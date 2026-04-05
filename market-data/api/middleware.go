package api

import (
	"log"
	"net/http"
	"sync"
	"time"

	"srfm/market-data/monitoring"
)

// Middleware holds middleware configuration.
type Middleware struct {
	metrics *monitoring.Metrics
	limiter *ipRateLimiter
}

// NewMiddleware creates the middleware chain.
func NewMiddleware(metrics *monitoring.Metrics) *Middleware {
	return &Middleware{
		metrics: metrics,
		limiter: newIPRateLimiter(100, 10), // 100 req/s, burst 10
	}
}

// Apply wraps the handler with all middleware.
func (m *Middleware) Apply(next http.Handler) http.Handler {
	return m.cors(m.rateLimit(m.logging(next)))
}

// logging logs each request with latency and status code.
func (m *Middleware) logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(rw, r)
		latency := time.Since(start)
		log.Printf("[http] %s %s %d %v %s",
			r.Method, r.URL.Path, rw.status, latency, r.RemoteAddr)
		m.metrics.HTTPRequest(r.URL.Path, rw.status, latency)
	})
}

// cors adds CORS headers for dashboard access.
func (m *Middleware) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// rateLimit enforces per-IP request limits.
func (m *Middleware) rateLimit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip rate limiting for WebSocket upgrades and metrics
		if r.URL.Path == "/stream" || r.URL.Path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}

		ip := extractIP(r)
		if !m.limiter.allow(ip) {
			writeError(w, "rate limit exceeded", http.StatusTooManyRequests)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// responseWriter wraps http.ResponseWriter to capture status code.
type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}

// extractIP gets the real client IP, respecting X-Forwarded-For.
func extractIP(r *http.Request) string {
	if fwd := r.Header.Get("X-Forwarded-For"); fwd != "" {
		return fwd
	}
	ip := r.RemoteAddr
	// Strip port
	for i := len(ip) - 1; i >= 0; i-- {
		if ip[i] == ':' {
			return ip[:i]
		}
	}
	return ip
}

// --- Token bucket rate limiter per IP ---

type tokenBucket struct {
	tokens     float64
	maxTokens  float64
	refillRate float64 // tokens per second
	lastRefill time.Time
}

func (tb *tokenBucket) allow() bool {
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tb.tokens += elapsed * tb.refillRate
	if tb.tokens > tb.maxTokens {
		tb.tokens = tb.maxTokens
	}
	tb.lastRefill = now
	if tb.tokens >= 1 {
		tb.tokens--
		return true
	}
	return false
}

type ipRateLimiter struct {
	mu      sync.Mutex
	buckets map[string]*tokenBucket
	rate    float64
	burst   float64
}

func newIPRateLimiter(ratePerSec, burst float64) *ipRateLimiter {
	l := &ipRateLimiter{
		buckets: make(map[string]*tokenBucket),
		rate:    ratePerSec,
		burst:   burst,
	}
	// Periodic cleanup to avoid unbounded growth
	go func() {
		for range time.Tick(5 * time.Minute) {
			l.cleanup()
		}
	}()
	return l
}

func (l *ipRateLimiter) allow(ip string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	tb, ok := l.buckets[ip]
	if !ok {
		tb = &tokenBucket{
			tokens:     l.burst,
			maxTokens:  l.burst,
			refillRate: l.rate,
			lastRefill: time.Now(),
		}
		l.buckets[ip] = tb
	}
	return tb.allow()
}

func (l *ipRateLimiter) cleanup() {
	l.mu.Lock()
	defer l.mu.Unlock()
	cutoff := time.Now().Add(-10 * time.Minute)
	for ip, tb := range l.buckets {
		if tb.lastRefill.Before(cutoff) {
			delete(l.buckets, ip)
		}
	}
}
