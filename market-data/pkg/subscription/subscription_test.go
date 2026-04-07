package subscription

import (
	"sync/atomic"
	"testing"
	"time"
)

func makeBar(sym, tf string, ts time.Time) Bar {
	return Bar{Symbol: sym, Timeframe: tf, Close: 100, Timestamp: ts}
}

// TestSubscribe_Receive verifies a callback receives the published bar.
func TestSubscribe_Receive(t *testing.T) {
	sm := NewSubscriptionManager()
	var received atomic.Int32

	id, err := sm.Subscribe("AAPL", []string{"1h"}, func(sym, tf string, bar Bar) {
		received.Add(1)
	})
	if err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	_ = id

	sm.Publish("AAPL", "1h", makeBar("AAPL", "1h", time.Now()))
	if received.Load() != 1 {
		t.Errorf("expected 1 callback, got %d", received.Load())
	}
}

// TestSubscribe_WrongTimeframe verifies no callback for non-subscribed timeframe.
func TestSubscribe_WrongTimeframe(t *testing.T) {
	sm := NewSubscriptionManager()
	var received atomic.Int32

	sm.Subscribe("AAPL", []string{"1d"}, func(sym, tf string, bar Bar) {
		received.Add(1)
	})
	sm.Publish("AAPL", "1h", makeBar("AAPL", "1h", time.Now()))
	if received.Load() != 0 {
		t.Error("should not receive bar for non-subscribed timeframe")
	}
}

// TestSubscribe_WrongSymbol verifies no callback for non-subscribed symbol.
func TestSubscribe_WrongSymbol(t *testing.T) {
	sm := NewSubscriptionManager()
	var received atomic.Int32

	sm.Subscribe("GOOGL", []string{"1h"}, func(sym, tf string, bar Bar) {
		received.Add(1)
	})
	sm.Publish("AAPL", "1h", makeBar("AAPL", "1h", time.Now()))
	if received.Load() != 0 {
		t.Error("should not receive bar for non-subscribed symbol")
	}
}

// TestUnsubscribe verifies removing a subscription stops delivery.
func TestUnsubscribe(t *testing.T) {
	sm := NewSubscriptionManager()
	var received atomic.Int32

	id, _ := sm.Subscribe("TSLA", []string{"5m"}, func(sym, tf string, bar Bar) {
		received.Add(1)
	})
	sm.Publish("TSLA", "5m", makeBar("TSLA", "5m", time.Now()))
	if received.Load() != 1 {
		t.Fatal("expected 1 delivery before unsubscribe")
	}

	sm.Unsubscribe(id)
	sm.Publish("TSLA", "5m", makeBar("TSLA", "5m", time.Now()))
	if received.Load() != 1 {
		t.Errorf("expected no delivery after unsubscribe, got %d total", received.Load())
	}
}

// TestFanOut_MultipleSubscribers verifies all subscribers receive the bar.
func TestFanOut_MultipleSubscribers(t *testing.T) {
	sm := NewSubscriptionManager()
	var total atomic.Int32

	for i := 0; i < 5; i++ {
		sm.Subscribe("SPY", []string{"1m"}, func(sym, tf string, bar Bar) {
			total.Add(1)
		})
	}
	sm.Publish("SPY", "1m", makeBar("SPY", "1m", time.Now()))
	if total.Load() != 5 {
		t.Errorf("expected 5 callbacks, got %d", total.Load())
	}
}

// TestActiveSymbols verifies ActiveSymbols returns correct deduplicated list.
func TestActiveSymbols(t *testing.T) {
	sm := NewSubscriptionManager()
	sm.Subscribe("AAPL", []string{"1h"}, func(_, _ string, _ Bar) {})
	sm.Subscribe("AAPL", []string{"1d"}, func(_, _ string, _ Bar) {})
	sm.Subscribe("MSFT", []string{"1h"}, func(_, _ string, _ Bar) {})

	syms := sm.ActiveSymbols()
	symSet := make(map[string]bool)
	for _, s := range syms {
		symSet[s] = true
	}
	if !symSet["AAPL"] || !symSet["MSFT"] {
		t.Errorf("unexpected active symbols: %v", syms)
	}
	if len(syms) != 2 {
		t.Errorf("expected 2 unique symbols, got %d", len(syms))
	}
}

// TestSubscriberCount verifies per-symbol counts.
func TestSubscriberCount(t *testing.T) {
	sm := NewSubscriptionManager()
	sm.Subscribe("NVDA", []string{"1h"}, func(_, _ string, _ Bar) {})
	sm.Subscribe("NVDA", []string{"1d"}, func(_, _ string, _ Bar) {})
	sm.Subscribe("AMD", []string{"1h"}, func(_, _ string, _ Bar) {})

	if sm.SubscriberCount("NVDA") != 2 {
		t.Errorf("expected 2 subscribers for NVDA, got %d", sm.SubscriberCount("NVDA"))
	}
	if sm.SubscriberCount("AMD") != 1 {
		t.Errorf("expected 1 subscriber for AMD, got %d", sm.SubscriberCount("AMD"))
	}
}

// TestSubscriptionCap verifies the 500-subscription limit is enforced.
func TestSubscriptionCap(t *testing.T) {
	sm := NewSubscriptionManager()
	cb := func(_, _ string, _ Bar) {}
	for i := 0; i < maxSubscriptions; i++ {
		if _, err := sm.Subscribe("SYM", []string{"1h"}, cb); err != nil {
			t.Fatalf("unexpected error at subscription %d: %v", i, err)
		}
	}
	_, err := sm.Subscribe("SYM", []string{"1h"}, cb)
	if err == nil {
		t.Error("expected error when cap exceeded")
	}
}

// TestDeadCallbackCleanup verifies that a panicking callback is removed.
func TestDeadCallbackCleanup(t *testing.T) {
	sm := NewSubscriptionManager()
	before := sm.Subscriptions()

	sm.Subscribe("BTC", []string{"1h"}, func(_, _ string, _ Bar) {
		panic("simulated panic")
	})
	if sm.Subscriptions() != before+1 {
		t.Fatal("subscription not added")
	}

	// Publish -- the panic should be caught and subscription removed.
	sm.Publish("BTC", "1h", makeBar("BTC", "1h", time.Now()))

	if sm.Subscriptions() != before {
		t.Errorf("expected panicking subscription to be removed, count=%d", sm.Subscriptions())
	}
}

// TestThrottledPublisher_Dedup verifies duplicate bars are dropped.
func TestThrottledPublisher_Dedup(t *testing.T) {
	sm := NewSubscriptionManager()
	var received atomic.Int32
	sm.Subscribe("ETH", []string{"1h"}, func(_, _ string, _ Bar) {
		received.Add(1)
	})

	tp := NewThrottledPublisher(sm)
	bar := makeBar("ETH", "1h", time.Now())

	tp.Publish("ETH", "1h", bar)
	tp.Publish("ETH", "1h", bar) // same bar -- should be deduplicated
	tp.Publish("ETH", "1h", bar) // same bar again

	if received.Load() != 1 {
		t.Errorf("expected 1 delivery after dedup, got %d", received.Load())
	}
}

// TestThrottledPublisher_DifferentTimestamps verifies distinct bars are published.
func TestThrottledPublisher_DifferentTimestamps(t *testing.T) {
	sm := NewSubscriptionManager()
	var received atomic.Int32
	sm.Subscribe("BTC", []string{"1h"}, func(_, _ string, _ Bar) {
		received.Add(1)
	})

	tp := NewThrottledPublisher(sm)
	now := time.Now()
	tp.Publish("BTC", "1h", makeBar("BTC", "1h", now))
	tp.Publish("BTC", "1h", makeBar("BTC", "1h", now.Add(time.Hour)))

	if received.Load() != 2 {
		t.Errorf("expected 2 deliveries for distinct timestamps, got %d", received.Load())
	}
}
