package hub

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// SubscriptionFilter defines what data a subscriber wants.
type SubscriptionFilter struct {
	// Symbols is the set of symbols to subscribe to. Empty = all.
	Symbols map[string]struct{}
	// Timeframes is the set of timeframes to subscribe to. Empty = all.
	Timeframes map[string]struct{}
	// EventTypes is the set of event types to subscribe to.
	// Valid: "bar", "trade", "quote". Empty = bar only.
	EventTypes map[string]struct{}
	// IncludePartial, if true, includes partial (in-progress) bars.
	IncludePartial bool
}

// NewSubscriptionFilter creates an empty filter that accepts all bars.
func NewSubscriptionFilter() SubscriptionFilter {
	return SubscriptionFilter{
		Symbols:    make(map[string]struct{}),
		Timeframes: make(map[string]struct{}),
		EventTypes: map[string]struct{}{"bar": {}},
	}
}

// Matches returns true if the filter matches the given parameters.
func (f *SubscriptionFilter) Matches(symbol, timeframe, eventType string, isPartial bool) bool {
	if isPartial && !f.IncludePartial {
		return false
	}
	if len(f.Symbols) > 0 {
		if _, ok := f.Symbols[symbol]; !ok {
			return false
		}
	}
	if len(f.Timeframes) > 0 && timeframe != "" {
		if _, ok := f.Timeframes[timeframe]; !ok {
			return false
		}
	}
	if len(f.EventTypes) > 0 {
		if _, ok := f.EventTypes[eventType]; !ok {
			return false
		}
	}
	return true
}

// Clone returns a deep copy of the filter.
func (f *SubscriptionFilter) Clone() SubscriptionFilter {
	c := SubscriptionFilter{
		Symbols:        make(map[string]struct{}, len(f.Symbols)),
		Timeframes:     make(map[string]struct{}, len(f.Timeframes)),
		EventTypes:     make(map[string]struct{}, len(f.EventTypes)),
		IncludePartial: f.IncludePartial,
	}
	for k, v := range f.Symbols {
		c.Symbols[k] = v
	}
	for k, v := range f.Timeframes {
		c.Timeframes[k] = v
	}
	for k, v := range f.EventTypes {
		c.EventTypes[k] = v
	}
	return c
}

// SubscriptionStats tracks usage statistics for a subscription.
type SubscriptionStats struct {
	MessagesReceived  int64
	MessagesSent      int64
	MessagesDropped   int64
	BytesSent         int64
	ConnectedAt       time.Time
	LastMessageAt     time.Time
}

// SubscriptionRegistry manages all active subscriptions and provides
// bulk management operations.
type SubscriptionRegistry struct {
	mu   sync.RWMutex
	subs map[uint64]*registryEntry
}

type registryEntry struct {
	id      uint64
	filter  SubscriptionFilter
	stats   SubscriptionStats
	send    chan []byte
	closeCh chan struct{}
}

// NewSubscriptionRegistry creates a SubscriptionRegistry.
func NewSubscriptionRegistry() *SubscriptionRegistry {
	return &SubscriptionRegistry{
		subs: make(map[uint64]*registryEntry),
	}
}

// Register adds a new subscription.
func (sr *SubscriptionRegistry) Register(id uint64, filter SubscriptionFilter, send chan []byte) {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	sr.subs[id] = &registryEntry{
		id:      id,
		filter:  filter,
		send:    send,
		closeCh: make(chan struct{}),
		stats:   SubscriptionStats{ConnectedAt: time.Now()},
	}
}

// Unregister removes a subscription.
func (sr *SubscriptionRegistry) Unregister(id uint64) {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	if entry, ok := sr.subs[id]; ok {
		select {
		case <-entry.closeCh:
		default:
			close(entry.closeCh)
		}
		delete(sr.subs, id)
	}
}

// UpdateFilter replaces the filter for an existing subscription.
func (sr *SubscriptionRegistry) UpdateFilter(id uint64, filter SubscriptionFilter) bool {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	entry, ok := sr.subs[id]
	if !ok {
		return false
	}
	entry.filter = filter.Clone()
	return true
}

// GetFilter returns the current filter for a subscription.
func (sr *SubscriptionRegistry) GetFilter(id uint64) (SubscriptionFilter, bool) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	entry, ok := sr.subs[id]
	if !ok {
		return SubscriptionFilter{}, false
	}
	return entry.filter.Clone(), true
}

// Fanout delivers a message to all matching subscriptions.
func (sr *SubscriptionRegistry) Fanout(symbol, timeframe, eventType string, isPartial bool, msg []byte) (sent, dropped int) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	for _, entry := range sr.subs {
		if !entry.filter.Matches(symbol, timeframe, eventType, isPartial) {
			continue
		}
		select {
		case entry.send <- msg:
			entry.stats.MessagesSent++
			entry.stats.BytesSent += int64(len(msg))
			entry.stats.LastMessageAt = time.Now()
			sent++
		default:
			entry.stats.MessagesDropped++
			dropped++
		}
	}
	return sent, dropped
}

// Count returns the number of registered subscriptions.
func (sr *SubscriptionRegistry) Count() int {
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	return len(sr.subs)
}

// Stats returns a snapshot of stats for all subscriptions.
func (sr *SubscriptionRegistry) Stats() []map[string]interface{} {
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	out := make([]map[string]interface{}, 0, len(sr.subs))
	for _, entry := range sr.subs {
		out = append(out, map[string]interface{}{
			"id":               fmt.Sprintf("%d", entry.id),
			"symbols":          setKeys(entry.filter.Symbols),
			"timeframes":       setKeys(entry.filter.Timeframes),
			"event_types":      setKeys(entry.filter.EventTypes),
			"messages_sent":    entry.stats.MessagesSent,
			"messages_dropped": entry.stats.MessagesDropped,
			"bytes_sent":       entry.stats.BytesSent,
			"connected_at":     entry.stats.ConnectedAt,
			"last_message_at":  entry.stats.LastMessageAt,
			"queue_depth":      len(entry.send),
		})
	}
	return out
}

func setKeys(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// WelcomeMessage builds the initial message sent to a new WS subscriber.
func WelcomeMessage(subscriberID uint64, activeSymbols []string) ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"type":           "welcome",
		"subscriber_id":  fmt.Sprintf("%d", subscriberID),
		"active_symbols": activeSymbols,
		"timestamp":      time.Now(),
	})
}
