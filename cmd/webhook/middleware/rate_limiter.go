// cmd/webhook/middleware/rate_limiter.go -- Per-route token bucket rate limiter.
//
// TokenBucket per route with configurable capacity and refill rate.
// Returns HTTP 429 with Retry-After header when the bucket is exhausted.

package middleware

import (
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// TokenBucket
// ---------------------------------------------------------------------------

// TokenBucket implements a token bucket algorithm for rate limiting.
// Tokens refill at `refillRate` tokens per second up to `capacity`.
type TokenBucket struct {
	mu         sync.Mutex
	tokens     float64
	capacity   float64
	refillRate float64 // tokens per second
	lastRefill time.Time
}

// newTokenBucket creates a full token bucket.
func newTokenBucket(capacity float64, refillRate float64) *TokenBucket {
	return &TokenBucket{
		tokens:     capacity,
		capacity:   capacity,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

// Allow attempts to consume one token. Returns true and 0 wait time if
// a token was available, or false and the time to wait if the bucket is empty.
func (b *TokenBucket) Allow() (allowed bool, retryAfter time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(b.lastRefill).Seconds()
	b.lastRefill = now

	// Refill tokens.
	b.tokens += elapsed * b.refillRate
	if b.tokens > b.capacity {
		b.tokens = b.capacity
	}

	if b.tokens >= 1.0 {
		b.tokens--
		return true, 0
	}

	// Calculate when the next token will be available.
	wait := time.Duration((1.0-b.tokens)/b.refillRate*1000) * time.Millisecond
	return false, wait
}

// ---------------------------------------------------------------------------
// RateLimiter middleware
// ---------------------------------------------------------------------------

// RateLimiter manages per-route token buckets and provides HTTP middleware.
type RateLimiter struct {
	mu      sync.RWMutex
	buckets map[string]*TokenBucket
	logger  *slog.Logger
}

// NewRateLimiter creates a new RateLimiter.
func NewRateLimiter(logger *slog.Logger) *RateLimiter {
	return &RateLimiter{
		buckets: make(map[string]*TokenBucket),
		logger:  logger,
	}
}

// Register pre-creates a named bucket. This is optional; Limit will create
// buckets on demand if they do not exist.
func (rl *RateLimiter) Register(route string, capacity float64, refillRate float64) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.buckets[route] = newTokenBucket(capacity, refillRate)
}

// bucket returns the named bucket, creating it with the given defaults if absent.
func (rl *RateLimiter) bucket(route string, capacity, refillRate float64) *TokenBucket {
	rl.mu.RLock()
	b, ok := rl.buckets[route]
	rl.mu.RUnlock()
	if ok {
		return b
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()
	// Double-check after acquiring write lock.
	if b, ok = rl.buckets[route]; ok {
		return b
	}
	b = newTokenBucket(capacity, refillRate)
	rl.buckets[route] = b
	return b
}

// Limit returns an HTTP middleware that enforces the token bucket for the named
// route. capacity is the burst capacity; refillRate is tokens per second.
func (rl *RateLimiter) Limit(route string, capacity, refillRate float64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b := rl.bucket(route, capacity, refillRate)
			allowed, wait := b.Allow()
			if !allowed {
				retrySeconds := int(wait.Seconds()) + 1
				w.Header().Set("Retry-After", fmt.Sprintf("%d", retrySeconds))
				w.Header().Set("X-RateLimit-Route", route)
				rl.logger.Warn("rate limit exceeded",
					"route", route,
					"remote", r.RemoteAddr,
					"retry_after_secs", retrySeconds,
				)
				http.Error(w,
					fmt.Sprintf("rate limit exceeded for route %q; retry after %ds", route, retrySeconds),
					http.StatusTooManyRequests,
				)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// Stats returns current token counts for all registered buckets.
// This is useful for debugging and monitoring.
func (rl *RateLimiter) Stats() map[string]float64 {
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	out := make(map[string]float64, len(rl.buckets))
	for name, b := range rl.buckets {
		b.mu.Lock()
		out[name] = b.tokens
		b.mu.Unlock()
	}
	return out
}
