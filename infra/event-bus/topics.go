// Package eventbus implements a Redis-backed pub/sub market event bus with
// durable Streams, consumer groups, dead-letter queues, and replay.
package eventbus

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Topic hierarchy and metadata
// ─────────────────────────────────────────────────────────────────────────────

// Topic represents a typed event bus channel.
type Topic string

// Well-known topic patterns.
// Wildcards: use '*' at a segment to match any value.
const (
	// Market data topics.
	TopicMarketBars   Topic = "market.bars.{symbol}"   // e.g. market.bars.AAPL
	TopicMarketQuotes Topic = "market.quotes.{symbol}" // e.g. market.quotes.AAPL
	TopicMarketTrades Topic = "market.trades.{symbol}"
	TopicMarketBook   Topic = "market.book.{symbol}"

	// Strategy / signal topics.
	TopicBHSignal  Topic = "bh.signal.{symbol}"   // e.g. bh.signal.AAPL
	TopicBHState   Topic = "bh.state.{symbol}"
	TopicDeltaScore Topic = "bh.delta.{symbol}.{bucket}"

	// Execution and order topics.
	TopicTradeExecuted  Topic = "trade.executed"
	TopicOrderSubmitted Topic = "order.submitted"
	TopicOrderCancelled Topic = "order.cancelled"
	TopicOrderFilled    Topic = "order.filled"

	// Risk topics.
	TopicRiskBreach   Topic = "risk.breach"
	TopicRiskEvent    Topic = "risk.event.{account_id}"
	TopicMarginCall   Topic = "risk.margin_call"

	// Portfolio topics.
	TopicPortfolioUpdate Topic = "portfolio.update.{account_id}"
	TopicPnLUpdate       Topic = "portfolio.pnl.{account_id}"

	// System topics.
	TopicSystemHeartbeat Topic = "system.heartbeat"
	TopicSystemAlert     Topic = "system.alert"
)

// TopicMeta holds metadata for a topic pattern.
type TopicMeta struct {
	Pattern     Topic
	Description string
	Persistent  bool          // if true, use Redis Streams for durability
	TTL         time.Duration // 0 = no expiry
	MaxLen      int64         // Redis Stream MAXLEN; 0 = unlimited
	Partitions  int           // number of stream partitions (sharding)
}

var topicRegistry = []TopicMeta{
	{TopicMarketBars, "OHLCV bar data per symbol", true, 24 * time.Hour, 100_000, 4},
	{TopicMarketQuotes, "NBBO quote updates", true, 1 * time.Hour, 500_000, 8},
	{TopicMarketTrades, "Individual trade prints", true, 4 * time.Hour, 1_000_000, 8},
	{TopicMarketBook, "Order book updates", false, 0, 0, 0},
	{TopicBHSignal, "BH signal events", true, 7 * 24 * time.Hour, 50_000, 2},
	{TopicBHState, "BH state snapshots", true, 7 * 24 * time.Hour, 20_000, 2},
	{TopicDeltaScore, "Delta score events", true, 24 * time.Hour, 100_000, 2},
	{TopicTradeExecuted, "Executed trade confirmations", true, 90 * 24 * time.Hour, 1_000_000, 1},
	{TopicOrderSubmitted, "New order submissions", true, 30 * 24 * time.Hour, 500_000, 1},
	{TopicOrderCancelled, "Order cancellations", true, 30 * 24 * time.Hour, 500_000, 1},
	{TopicOrderFilled, "Order fill events", true, 90 * 24 * time.Hour, 500_000, 1},
	{TopicRiskBreach, "Risk limit breach events", true, 365 * 24 * time.Hour, 100_000, 1},
	{TopicRiskEvent, "Per-account risk events", true, 90 * 24 * time.Hour, 100_000, 1},
	{TopicMarginCall, "Margin call events", true, 365 * 24 * time.Hour, 10_000, 1},
	{TopicPortfolioUpdate, "Portfolio state updates", true, 7 * 24 * time.Hour, 100_000, 1},
	{TopicPnLUpdate, "PnL update events", true, 30 * 24 * time.Hour, 100_000, 1},
	{TopicSystemHeartbeat, "System health heartbeats", false, 0, 0, 0},
	{TopicSystemAlert, "System alert notifications", true, 30 * 24 * time.Hour, 10_000, 1},
}

// topicMetaMap is loaded once at init.
var topicMetaMap map[Topic]*TopicMeta
var topicMetaOnce sync.Once

func loadTopicMeta() map[Topic]*TopicMeta {
	topicMetaOnce.Do(func() {
		topicMetaMap = make(map[Topic]*TopicMeta, len(topicRegistry))
		for i := range topicRegistry {
			topicMetaMap[topicRegistry[i].Pattern] = &topicRegistry[i]
		}
	})
	return topicMetaMap
}

// GetTopicMeta returns metadata for a topic pattern, or nil if not registered.
func GetTopicMeta(pattern Topic) *TopicMeta {
	return loadTopicMeta()[pattern]
}

// ─────────────────────────────────────────────────────────────────────────────
// Topic resolution — pattern → concrete channel name
// ─────────────────────────────────────────────────────────────────────────────

// ResolveTopic replaces {variable} placeholders in a topic pattern with actual values.
// vars maps variable name (without braces) to its value.
//
//	ResolveTopic("market.bars.{symbol}", map[string]string{"symbol": "AAPL"})
//	→ "market.bars.AAPL"
func ResolveTopic(pattern Topic, vars map[string]string) Topic {
	s := string(pattern)
	for k, v := range vars {
		s = strings.ReplaceAll(s, "{"+k+"}", v)
	}
	return Topic(s)
}

// ForSymbol resolves a single-symbol topic pattern.
func ForSymbol(pattern Topic, symbol string) Topic {
	return ResolveTopic(pattern, map[string]string{"symbol": symbol})
}

// ForAccount resolves an account topic pattern.
func ForAccount(pattern Topic, accountID string) Topic {
	return ResolveTopic(pattern, map[string]string{"account_id": accountID})
}

// ForSymbolBucket resolves a symbol+bucket topic pattern.
func ForSymbolBucket(pattern Topic, symbol, bucket string) Topic {
	return ResolveTopic(pattern, map[string]string{"symbol": symbol, "bucket": bucket})
}

// ─────────────────────────────────────────────────────────────────────────────
// Wildcard matching
// ─────────────────────────────────────────────────────────────────────────────

// TopicRouter manages wildcard subscriptions.
// Subscriptions may use '*' as a segment wildcard, e.g. "market.bars.*".
type TopicRouter struct {
	mu          sync.RWMutex
	exact       map[Topic][]subscriptionHandle  // exact matches
	wildcards   []wildcardEntry                  // ordered list of wildcard patterns
}

type subscriptionHandle struct {
	id string
	ch chan<- *Event
}

type wildcardEntry struct {
	pattern  string
	segments []string
	subs     []subscriptionHandle
}

// NewTopicRouter creates a new router.
func NewTopicRouter() *TopicRouter {
	return &TopicRouter{
		exact: make(map[Topic][]subscriptionHandle),
	}
}

// Subscribe registers ch to receive events on topic (may contain '*' wildcards).
func (r *TopicRouter) Subscribe(topic Topic, id string, ch chan<- *Event) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !strings.Contains(string(topic), "*") {
		r.exact[topic] = append(r.exact[topic], subscriptionHandle{id, ch})
		return
	}
	// Find or create wildcard entry.
	for i := range r.wildcards {
		if r.wildcards[i].pattern == string(topic) {
			r.wildcards[i].subs = append(r.wildcards[i].subs, subscriptionHandle{id, ch})
			return
		}
	}
	r.wildcards = append(r.wildcards, wildcardEntry{
		pattern:  string(topic),
		segments: strings.Split(string(topic), "."),
		subs:     []subscriptionHandle{{id, ch}},
	})
}

// Unsubscribe removes all subscriptions with the given id from topic.
func (r *TopicRouter) Unsubscribe(topic Topic, id string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	subs := r.exact[topic]
	updated := subs[:0]
	for _, s := range subs {
		if s.id != id {
			updated = append(updated, s)
		}
	}
	r.exact[topic] = updated

	for i := range r.wildcards {
		wsubs := r.wildcards[i].subs
		wupdated := wsubs[:0]
		for _, s := range wsubs {
			if s.id != id {
				wupdated = append(wupdated, s)
			}
		}
		r.wildcards[i].subs = wupdated
	}
}

// Dispatch sends an event to all matching subscribers.
func (r *TopicRouter) Dispatch(evt *Event) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	topic := evt.Topic

	// Exact match.
	for _, s := range r.exact[topic] {
		select {
		case s.ch <- evt:
		default:
		}
	}

	// Wildcard match.
	topicSegs := strings.Split(string(topic), ".")
	for _, wc := range r.wildcards {
		if matchSegments(topicSegs, wc.segments) {
			for _, s := range wc.subs {
				select {
				case s.ch <- evt:
				default:
				}
			}
		}
	}
}

// matchSegments returns true if topic matches pattern, where '*' in pattern
// matches any single segment.
func matchSegments(topic, pattern []string) bool {
	if len(topic) != len(pattern) {
		return false
	}
	for i, p := range pattern {
		if p != "*" && p != topic[i] {
			return false
		}
	}
	return true
}

// SubscriberCount returns the total number of subscribers for a topic.
func (r *TopicRouter) SubscriberCount(topic Topic) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.exact[topic])
}

// TopicList returns all topics that have at least one subscriber.
func (r *TopicRouter) TopicList() []Topic {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Topic, 0, len(r.exact))
	for t, subs := range r.exact {
		if len(subs) > 0 {
			out = append(out, t)
		}
	}
	return out
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream key helpers
// ─────────────────────────────────────────────────────────────────────────────

// StreamKey returns the Redis Stream key for a given topic (+ optional partition).
func StreamKey(topic Topic, partition int) string {
	if partition <= 0 {
		return fmt.Sprintf("stream:%s", topic)
	}
	return fmt.Sprintf("stream:%s:%d", topic, partition)
}

// DLQKey returns the dead-letter queue stream key for a topic.
func DLQKey(topic Topic) string {
	return fmt.Sprintf("dlq:%s", topic)
}

// ConsumerGroupName returns the standard consumer group name for a service+topic.
func ConsumerGroupName(serviceName string, topic Topic) string {
	safe := strings.ReplaceAll(string(topic), ".", "_")
	safe = strings.ReplaceAll(safe, "{", "")
	safe = strings.ReplaceAll(safe, "}", "")
	safe = strings.ReplaceAll(safe, "*", "wildcard")
	return fmt.Sprintf("cg:%s:%s", serviceName, safe)
}
