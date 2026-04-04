package api

import (
	"fmt"
	"net/http"
	"runtime/debug"
	"strconv"
	"time"

	"go.uber.org/zap"
)

// responseWriter wraps http.ResponseWriter to capture the status code.
type responseWriter struct {
	http.ResponseWriter
	status      int
	written     int64
	wroteHeader bool
}

func wrapResponseWriter(w http.ResponseWriter) *responseWriter {
	return &responseWriter{ResponseWriter: w, status: http.StatusOK}
}

func (rw *responseWriter) Status() int         { return rw.status }
func (rw *responseWriter) BytesWritten() int64 { return rw.written }

func (rw *responseWriter) WriteHeader(code int) {
	if rw.wroteHeader {
		return
	}
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
	rw.wroteHeader = true
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	n, err := rw.ResponseWriter.Write(b)
	rw.written += int64(n)
	return n, err
}

// CORSMiddleware adds CORS headers to all responses.
func CORSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")
		w.Header().Set("Access-Control-Max-Age", "86400")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// RequestLoggerMiddleware logs each HTTP request with duration and status.
func RequestLoggerMiddleware(log *zap.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			wrapped := wrapResponseWriter(w)
			next.ServeHTTP(wrapped, r)
			log.Info("http",
				zap.String("method", r.Method),
				zap.String("path", r.URL.Path),
				zap.String("query", r.URL.RawQuery),
				zap.Int("status", wrapped.Status()),
				zap.Int64("bytes", wrapped.BytesWritten()),
				zap.Duration("duration", time.Since(start)),
				zap.String("remote", r.RemoteAddr),
				zap.String("user_agent", r.UserAgent()),
			)
		})
	}
}

// RecoveryMiddleware catches panics and returns 500.
func RecoveryMiddleware(log *zap.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if rec := recover(); rec != nil {
					stack := debug.Stack()
					log.Error("http handler panic",
						zap.Any("panic", rec),
						zap.ByteString("stack", stack))
					writeJSON(w, http.StatusInternalServerError, map[string]string{
						"error": "internal server error",
					})
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// RateLimiter is a simple token-bucket rate limiter per remote IP.
type RateLimiter struct {
	tokens   map[string]*tokenBucket
	maxRate  int           // requests per window
	window   time.Duration
}

type tokenBucket struct {
	count    int
	resetAt  time.Time
}

// NewRateLimiter creates a RateLimiter.
func NewRateLimiter(maxRate int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		tokens:  make(map[string]*tokenBucket),
		maxRate: maxRate,
		window:  window,
	}
}

// Allow returns true if the IP is within the rate limit.
func (rl *RateLimiter) Allow(ip string) bool {
	now := time.Now()
	bucket, ok := rl.tokens[ip]
	if !ok || now.After(bucket.resetAt) {
		rl.tokens[ip] = &tokenBucket{count: 1, resetAt: now.Add(rl.window)}
		return true
	}
	if bucket.count >= rl.maxRate {
		return false
	}
	bucket.count++
	return true
}

// RateLimitMiddleware wraps a handler with IP-based rate limiting.
func RateLimitMiddleware(rl *RateLimiter) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ip := extractIP(r)
			if !rl.Allow(ip) {
				w.Header().Set("Retry-After", "60")
				writeJSON(w, http.StatusTooManyRequests, map[string]string{
					"error": "rate limit exceeded",
				})
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

func extractIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		return xff
	}
	return r.RemoteAddr
}

// PrometheusMiddleware records HTTP request metrics.
func PrometheusMiddleware(observe func(method, path string, status int, duration float64)) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			wrapped := wrapResponseWriter(w)
			next.ServeHTTP(wrapped, r)
			observe(r.Method, r.URL.Path, wrapped.Status(), time.Since(start).Seconds())
		})
	}
}

// CacheControlMiddleware sets cache headers on bar query responses.
func CacheControlMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Don't cache WebSocket upgrade or health checks.
		if r.Header.Get("Upgrade") == "websocket" || r.URL.Path == "/health" {
			next.ServeHTTP(w, r)
			return
		}
		// Historical bar queries can be cached briefly.
		if r.URL.Path != "" && len(r.URL.Path) > 5 {
			w.Header().Set("Cache-Control", "public, max-age=5")
			w.Header().Set("ETag", fmt.Sprintf(`"%d"`, time.Now().Unix()/5))
		}
		next.ServeHTTP(w, r)
	})
}

// TimeoutMiddleware wraps a handler with a request timeout.
func TimeoutMiddleware(timeout time.Duration) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctx := r.Context()
			done := make(chan struct{})
			panicChan := make(chan interface{}, 1)

			go func() {
				defer func() {
					if p := recover(); p != nil {
						panicChan <- p
					}
				}()
				next.ServeHTTP(w, r.WithContext(ctx))
				close(done)
			}()

			timer := time.NewTimer(timeout)
			defer timer.Stop()

			select {
			case p := <-panicChan:
				panic(p)
			case <-done:
				return
			case <-timer.C:
				writeJSON(w, http.StatusGatewayTimeout, map[string]string{
					"error": "request timeout after " + strconv.FormatFloat(timeout.Seconds(), 'f', 1, 64) + "s",
				})
			}
		})
	}
}
