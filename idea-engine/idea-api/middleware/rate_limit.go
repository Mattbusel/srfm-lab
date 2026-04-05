package middleware

import (
	"encoding/json"
	"net"
	"net/http"
	"sync"
	"time"

	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

// ipLimiter tracks a rate limiter and its last-seen time for a single IP.
type ipLimiter struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

// RateLimiter is an HTTP middleware that applies per-IP token-bucket rate
// limiting. Clients that exceed the limit receive 429 Too Many Requests.
//
// Stale entries (IPs not seen for more than cleanupInterval) are evicted
// periodically to prevent unbounded memory growth.
type RateLimiter struct {
	mu              sync.Mutex
	limiters        map[string]*ipLimiter
	rps             rate.Limit
	burst           int
	cleanupInterval time.Duration
	log             *zap.Logger
}

// NewRateLimiter constructs a RateLimiter.
// rps is the sustained request rate per IP; burst is the maximum instantaneous
// burst allowed.
func NewRateLimiter(rps float64, burst int, log *zap.Logger) *RateLimiter {
	rl := &RateLimiter{
		limiters:        make(map[string]*ipLimiter),
		rps:             rate.Limit(rps),
		burst:           burst,
		cleanupInterval: 5 * time.Minute,
		log:             log,
	}
	go rl.cleanupLoop()
	return rl
}

// Middleware returns an http.Handler middleware that enforces the rate limit.
func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := extractIP(r)

		if !rl.allow(ip) {
			rl.log.Warn("rate limit exceeded",
				zap.String("ip", ip),
				zap.String("path", r.URL.Path),
			)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			enc := json.NewEncoder(w)
			enc.SetIndent("", "  ")
			_ = enc.Encode(map[string]interface{}{
				"error": "rate limit exceeded; please slow down",
				"code":  http.StatusTooManyRequests,
			})
			return
		}

		next.ServeHTTP(w, r)
	})
}

// allow returns true if the given IP is within its rate limit.
func (rl *RateLimiter) allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	lim, ok := rl.limiters[ip]
	if !ok {
		lim = &ipLimiter{
			limiter: rate.NewLimiter(rl.rps, rl.burst),
		}
		rl.limiters[ip] = lim
	}
	lim.lastSeen = time.Now()
	return lim.limiter.Allow()
}

// cleanupLoop periodically removes stale limiters to reclaim memory.
func (rl *RateLimiter) cleanupLoop() {
	ticker := time.NewTicker(rl.cleanupInterval)
	defer ticker.Stop()
	for range ticker.C {
		rl.cleanup()
	}
}

// cleanup removes IPs not seen in the last cleanupInterval.
func (rl *RateLimiter) cleanup() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	cutoff := time.Now().Add(-rl.cleanupInterval)
	before := len(rl.limiters)
	for ip, lim := range rl.limiters {
		if lim.lastSeen.Before(cutoff) {
			delete(rl.limiters, ip)
		}
	}
	after := len(rl.limiters)
	if before != after {
		rl.log.Debug("rate limiter cleanup",
			zap.Int("removed", before-after),
			zap.Int("remaining", after),
		)
	}
}

// ActiveCount returns the number of IPs currently being tracked.
func (rl *RateLimiter) ActiveCount() int {
	rl.mu.Lock()
	n := len(rl.limiters)
	rl.mu.Unlock()
	return n
}

// extractIP returns the client IP address from the request, consulting
// X-Forwarded-For and X-Real-IP headers before falling back to RemoteAddr.
func extractIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		// X-Forwarded-For can be a comma-separated list; take the first.
		if idx := len(xff); idx > 0 {
			for i := 0; i < len(xff); i++ {
				if xff[i] == ',' {
					xff = xff[:i]
					break
				}
			}
		}
		xff = trimSpace(xff)
		if xff != "" {
			return xff
		}
	}
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return trimSpace(xri)
	}
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return ip
}

// trimSpace strips leading and trailing ASCII spaces.
func trimSpace(s string) string {
	start, end := 0, len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t') {
		end--
	}
	return s[start:end]
}
