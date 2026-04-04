// ratelimiter.go — Per-client and global broadcast rate limiting.
package wshub

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

// ─────────────────────────────────────────────────────────────────────────────
// Rate limiter configuration
// ─────────────────────────────────────────────────────────────────────────────

// RateLimiterConfig configures the hub rate limiter.
type RateLimiterConfig struct {
	// GlobalMessagesPerSec is the server-wide inbound message rate limit.
	GlobalMessagesPerSec float64
	// GlobalBurst is the burst size for the global limiter.
	GlobalBurst int

	// PerClientMessagesPerSec is the per-connection inbound rate limit.
	PerClientMessagesPerSec float64
	// PerClientBurst is the burst size per client.
	PerClientBurst int

	// BroadcastMessagesPerSec is the max outbound broadcast rate per room.
	BroadcastMessagesPerSec float64
	// BroadcastBurst is the burst size for room broadcasts.
	BroadcastBurst int

	// CleanupInterval removes stale per-client limiters.
	CleanupInterval time.Duration
}

// DefaultRateLimiterConfig returns sensible defaults.
func DefaultRateLimiterConfig() RateLimiterConfig {
	return RateLimiterConfig{
		GlobalMessagesPerSec:    10_000,
		GlobalBurst:             20_000,
		PerClientMessagesPerSec: 100,
		PerClientBurst:          200,
		BroadcastMessagesPerSec: 1_000,
		BroadcastBurst:          5_000,
		CleanupInterval:         5 * time.Minute,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// HubRateLimiter
// ─────────────────────────────────────────────────────────────────────────────

// HubRateLimiter enforces message rate limits at the hub level.
type HubRateLimiter struct {
	cfg     RateLimiterConfig
	log     *zap.Logger
	global  *rate.Limiter
	clients sync.Map // clientID → *clientRateLimiter
	rooms   sync.Map // roomName → *rate.Limiter

	// Metrics.
	globalDrops atomic.Int64
	clientDrops atomic.Int64
	roomDrops   atomic.Int64
}

type clientRateLimiter struct {
	limiter    *rate.Limiter
	lastAccess atomic.Value // time.Time
}

// NewHubRateLimiter creates a HubRateLimiter and starts the cleanup goroutine.
func NewHubRateLimiter(cfg RateLimiterConfig, log *zap.Logger) *HubRateLimiter {
	rl := &HubRateLimiter{
		cfg:    cfg,
		log:    log,
		global: rate.NewLimiter(rate.Limit(cfg.GlobalMessagesPerSec), cfg.GlobalBurst),
	}
	go rl.cleanupLoop()
	return rl
}

// AllowInbound checks if an inbound client message is allowed.
// Returns false if the global or per-client rate limit is exceeded.
func (rl *HubRateLimiter) AllowInbound(clientID string) bool {
	if !rl.global.Allow() {
		rl.globalDrops.Add(1)
		hubRateLimitedTotal.WithLabelValues("global").Inc()
		return false
	}

	crl := rl.getOrCreateClient(clientID)
	if !crl.limiter.Allow() {
		rl.clientDrops.Add(1)
		hubRateLimitedTotal.WithLabelValues("client").Inc()
		return false
	}
	crl.lastAccess.Store(time.Now())
	return true
}

// AllowBroadcast checks if a broadcast to a room is allowed.
func (rl *HubRateLimiter) AllowBroadcast(room string) bool {
	lim := rl.getOrCreateRoomLimiter(room)
	if !lim.Allow() {
		rl.roomDrops.Add(1)
		hubRateLimitedTotal.WithLabelValues("room").Inc()
		return false
	}
	return true
}

// WaitInbound blocks until the global rate limiter allows the request.
func (rl *HubRateLimiter) WaitInbound(ctx context.Context, clientID string) error {
	if err := rl.global.Wait(ctx); err != nil {
		return fmt.Errorf("global rate limit: %w", err)
	}
	crl := rl.getOrCreateClient(clientID)
	if err := crl.limiter.Wait(ctx); err != nil {
		return fmt.Errorf("client rate limit: %w", err)
	}
	crl.lastAccess.Store(time.Now())
	return nil
}

// RemoveClient removes the per-client rate limiter (call on disconnect).
func (rl *HubRateLimiter) RemoveClient(clientID string) {
	rl.clients.Delete(clientID)
}

// RateLimiterStats returns current rate limiter statistics.
type RateLimiterStats struct {
	GlobalDrops int64
	ClientDrops int64
	RoomDrops   int64
}

func (rl *HubRateLimiter) Stats() RateLimiterStats {
	return RateLimiterStats{
		GlobalDrops: rl.globalDrops.Load(),
		ClientDrops: rl.clientDrops.Load(),
		RoomDrops:   rl.roomDrops.Load(),
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

func (rl *HubRateLimiter) getOrCreateClient(clientID string) *clientRateLimiter {
	if v, ok := rl.clients.Load(clientID); ok {
		return v.(*clientRateLimiter)
	}
	crl := &clientRateLimiter{
		limiter: rate.NewLimiter(
			rate.Limit(rl.cfg.PerClientMessagesPerSec),
			rl.cfg.PerClientBurst,
		),
	}
	crl.lastAccess.Store(time.Now())
	actual, _ := rl.clients.LoadOrStore(clientID, crl)
	return actual.(*clientRateLimiter)
}

func (rl *HubRateLimiter) getOrCreateRoomLimiter(room string) *rate.Limiter {
	if v, ok := rl.rooms.Load(room); ok {
		return v.(*rate.Limiter)
	}
	lim := rate.NewLimiter(
		rate.Limit(rl.cfg.BroadcastMessagesPerSec),
		rl.cfg.BroadcastBurst,
	)
	actual, _ := rl.rooms.LoadOrStore(room, lim)
	return actual.(*rate.Limiter)
}

// cleanupLoop periodically evicts stale per-client limiters.
func (rl *HubRateLimiter) cleanupLoop() {
	ticker := time.NewTicker(rl.cfg.CleanupInterval)
	defer ticker.Stop()
	for range ticker.C {
		cutoff := time.Now().Add(-rl.cfg.CleanupInterval)
		rl.clients.Range(func(k, v interface{}) bool {
			crl := v.(*clientRateLimiter)
			last, _ := crl.lastAccess.Load().(time.Time)
			if last.Before(cutoff) {
				rl.clients.Delete(k)
			}
			return true
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// ThrottledBroadcaster — wraps Hub.Broadcast with per-room throttling
// ─────────────────────────────────────────────────────────────────────────────

// ThrottledBroadcaster wraps a Hub and applies rate limiting to broadcasts.
type ThrottledBroadcaster struct {
	hub *Hub
	rl  *HubRateLimiter
	log *zap.Logger
}

// NewThrottledBroadcaster creates a ThrottledBroadcaster.
func NewThrottledBroadcaster(hub *Hub, rl *HubRateLimiter, log *zap.Logger) *ThrottledBroadcaster {
	return &ThrottledBroadcaster{hub: hub, rl: rl, log: log}
}

// BroadcastUpdate sends a typed update to a room if not rate-limited.
func (tb *ThrottledBroadcaster) BroadcastUpdate(room, dataType string, payload interface{}) error {
	if !tb.rl.AllowBroadcast(room) {
		hubRateLimitedTotal.WithLabelValues("broadcast").Inc()
		return nil // silently drop over-limit broadcasts
	}
	return tb.hub.BroadcastUpdate(room, dataType, payload)
}
