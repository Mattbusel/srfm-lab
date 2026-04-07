// Package subscription -- throttled_publisher.go implements a rate-limited,
// deduplicating bar broadcast layer that sits in front of SubscriptionManager.
//
// Limits:
//   - Sustained rate: 1000 bars/second across all symbols
//   - Burst allowance: up to 5000 bars/second for 1 second
//   - Deduplication: bars with the same (symbol, timeframe, timestamp) are
//     dropped if already published within the current window.
package subscription

import (
	"sync"
	"time"
)

const (
	sustainedRate  = 1000 // bars per second (sustained)
	burstRate      = 5000 // bars per second (burst ceiling)
	burstWindowSec = 1    // burst window duration in seconds
)

// seenKey uniquely identifies a bar for deduplication.
type seenKey struct {
	symbol    string
	timeframe string
	ts        int64 // Unix nano
}

// ThrottledPublisher wraps a SubscriptionManager with rate-limiting and
// deduplication.
type ThrottledPublisher struct {
	mu sync.Mutex
	sm *SubscriptionManager

	// Token-bucket counters (updated per Publish call).
	tokens     float64   // available tokens (max = burstRate)
	lastRefill time.Time // last time tokens were refilled

	// Deduplication: tracks bars seen in the last dedup window.
	seen       map[seenKey]struct{}
	seenExpiry time.Time // when to reset the seen map
}

// NewThrottledPublisher creates a ThrottledPublisher backed by sm.
func NewThrottledPublisher(sm *SubscriptionManager) *ThrottledPublisher {
	return &ThrottledPublisher{
		sm:         sm,
		tokens:     burstRate,
		lastRefill: time.Now(),
		seen:       make(map[seenKey]struct{}),
		seenExpiry: time.Now().Add(time.Second),
	}
}

// Publish attempts to publish bar to the underlying SubscriptionManager.
// Returns true if the bar was published, false if it was throttled or
// identified as a duplicate.
func (p *ThrottledPublisher) Publish(symbol string, timeframe string, bar Bar) bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	// -- Deduplication check --
	key := seenKey{symbol, timeframe, bar.Timestamp.UnixNano()}
	now := time.Now()

	// Reset seen map once per second to bound memory usage.
	if now.After(p.seenExpiry) {
		p.seen = make(map[seenKey]struct{})
		p.seenExpiry = now.Add(time.Second)
	}
	if _, dup := p.seen[key]; dup {
		return false
	}

	// -- Token bucket refill --
	elapsed := now.Sub(p.lastRefill).Seconds()
	p.lastRefill = now
	p.tokens += elapsed * sustainedRate
	if p.tokens > burstRate {
		p.tokens = burstRate
	}

	if p.tokens < 1 {
		// Throttled -- not enough tokens.
		return false
	}
	p.tokens--

	// Mark as seen and publish.
	p.seen[key] = struct{}{}
	p.mu.Unlock()
	p.sm.Publish(symbol, timeframe, bar)
	p.mu.Lock() // re-acquire for deferred unlock
	return true
}

// TokensAvailable returns the current token count (for diagnostics).
func (p *ThrottledPublisher) TokensAvailable() float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.tokens
}

// SeenCount returns the number of unique bars tracked in the current dedup
// window (for diagnostics).
func (p *ThrottledPublisher) SeenCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.seen)
}
