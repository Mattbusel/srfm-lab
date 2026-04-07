// Package subscription manages per-symbol bar subscriptions with fan-out to
// registered callbacks. It supports up to 500 concurrent subscriptions and
// automatically removes dead (panicking) callbacks.
package subscription

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Bar is a minimal bar type used for subscription callbacks.
// It intentionally mirrors the cache.Bar structure so the two packages can
// interoperate without a hard import cycle.
type Bar struct {
	Symbol    string
	Timeframe string
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	Timestamp time.Time
}

// BarCallback is a function invoked when a new bar is published for a
// subscribed symbol/timeframe combination.
type BarCallback func(symbol string, timeframe string, bar Bar)

// SubscriptionID is an opaque identifier returned by Subscribe.
type SubscriptionID uint64

// maxSubscriptions is the hard cap on concurrent active subscriptions.
const maxSubscriptions = 500

// subscription holds the internal state of one subscription.
type subscription struct {
	id         SubscriptionID
	symbol     string
	timeframes map[string]struct{}
	cb         BarCallback
}

// SubscriptionManager manages symbol subscriptions and fans out bar events
// to all matching callbacks.
type SubscriptionManager struct {
	mu   sync.RWMutex
	subs map[SubscriptionID]*subscription

	// nextID is incremented atomically for each new subscription.
	nextID uint64
}

// NewSubscriptionManager creates an empty SubscriptionManager.
func NewSubscriptionManager() *SubscriptionManager {
	return &SubscriptionManager{
		subs: make(map[SubscriptionID]*subscription),
	}
}

// Subscribe registers cb to receive bars for symbol on the given timeframes.
// Returns an ID that can be passed to Unsubscribe.
// Returns an error if the subscription cap (500) would be exceeded.
func (m *SubscriptionManager) Subscribe(
	symbol string, timeframes []string, cb BarCallback,
) (SubscriptionID, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.subs) >= maxSubscriptions {
		return 0, fmt.Errorf("subscription: limit of %d reached", maxSubscriptions)
	}

	id := SubscriptionID(atomic.AddUint64(&m.nextID, 1))
	tfs := make(map[string]struct{}, len(timeframes))
	for _, tf := range timeframes {
		tfs[tf] = struct{}{}
	}
	m.subs[id] = &subscription{
		id:         id,
		symbol:     symbol,
		timeframes: tfs,
		cb:         cb,
	}
	return id, nil
}

// Unsubscribe removes the subscription with the given ID. It is a no-op if
// the ID does not exist.
func (m *SubscriptionManager) Unsubscribe(id SubscriptionID) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.subs, id)
}

// Publish fans out a bar to all callbacks whose subscription matches both
// symbol and timeframe. Panicking callbacks are caught and the offending
// subscription is removed (dead callback cleanup).
func (m *SubscriptionManager) Publish(symbol string, timeframe string, bar Bar) {
	m.mu.RLock()
	// Collect matching subscriptions while holding the read lock.
	type entry struct {
		id SubscriptionID
		cb BarCallback
	}
	var targets []entry
	for id, sub := range m.subs {
		if sub.symbol != symbol {
			continue
		}
		if _, ok := sub.timeframes[timeframe]; !ok {
			continue
		}
		targets = append(targets, entry{id, sub.cb})
	}
	m.mu.RUnlock()

	// Invoke callbacks outside the lock to avoid deadlocks.
	var dead []SubscriptionID
	for _, t := range targets {
		id, cb := t.id, t.cb
		func() {
			defer func() {
				if r := recover(); r != nil {
					dead = append(dead, id)
				}
			}()
			cb(symbol, timeframe, bar)
		}()
	}

	// Remove dead subscriptions.
	if len(dead) > 0 {
		m.mu.Lock()
		for _, id := range dead {
			delete(m.subs, id)
		}
		m.mu.Unlock()
	}
}

// ActiveSymbols returns a deduplicated list of symbols that currently have at
// least one subscription.
func (m *SubscriptionManager) ActiveSymbols() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	seen := make(map[string]struct{})
	for _, sub := range m.subs {
		seen[sub.symbol] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for sym := range seen {
		out = append(out, sym)
	}
	return out
}

// SubscriberCount returns the number of active subscriptions for the given
// symbol (across all timeframes).
func (m *SubscriptionManager) SubscriberCount(symbol string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	count := 0
	for _, sub := range m.subs {
		if sub.symbol == symbol {
			count++
		}
	}
	return count
}

// Subscriptions returns the total number of active subscriptions.
func (m *SubscriptionManager) Subscriptions() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.subs)
}
