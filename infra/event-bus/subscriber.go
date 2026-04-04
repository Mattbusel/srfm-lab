// subscriber.go — Typed subscribers with goroutine workers and retry semantics.
package eventbus

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// Generic TypedSubscriber
// ─────────────────────────────────────────────────────────────────────────────

// TypedSubscriber[T] is a generic subscriber that automatically deserializes
// the event payload into type T before calling the handler.
type TypedSubscriber[T any] struct {
	bus      *EventBus
	log      *zap.Logger
	subID    string
	topic    Topic
	handler  func(ctx context.Context, payload *T, evt *Event) error
	workers  int

	// Stats.
	received  atomic.Int64
	processed atomic.Int64
	errors    atomic.Int64
}

// NewTypedSubscriber creates a TypedSubscriber and registers it with the bus.
// workers controls how many concurrent goroutines process events.
func NewTypedSubscriber[T any](
	bus *EventBus,
	topic Topic,
	workers int,
	handler func(ctx context.Context, payload *T, evt *Event) error,
	log *zap.Logger,
) (*TypedSubscriber[T], error) {
	if workers <= 0 {
		workers = 1
	}

	ts := &TypedSubscriber[T]{
		bus:     bus,
		log:     log,
		topic:   topic,
		handler: handler,
		workers: workers,
	}

	// Wrap the typed handler in the bus's generic EventHandler interface.
	subID, err := bus.Subscribe(topic, func(ctx context.Context, evt *Event) error {
		return ts.dispatch(ctx, evt)
	})
	if err != nil {
		return nil, fmt.Errorf("subscribe to %s: %w", topic, err)
	}
	ts.subID = subID
	return ts, nil
}

// dispatch deserializes the event payload and calls the typed handler.
func (ts *TypedSubscriber[T]) dispatch(ctx context.Context, evt *Event) error {
	ts.received.Add(1)
	var payload T
	if err := json.Unmarshal(evt.Payload, &payload); err != nil {
		ts.errors.Add(1)
		ts.log.Error("payload unmarshal failed",
			zap.String("topic", string(ts.topic)),
			zap.String("event_id", evt.ID),
			zap.Error(err))
		return err
	}
	if err := ts.handler(ctx, &payload, evt); err != nil {
		ts.errors.Add(1)
		return err
	}
	ts.processed.Add(1)
	return nil
}

// Unsubscribe removes this subscriber from the bus.
func (ts *TypedSubscriber[T]) Unsubscribe() {
	ts.bus.Unsubscribe(ts.subID)
}

// Stats returns subscriber counters.
type SubscriberStats struct {
	Received  int64
	Processed int64
	Errors    int64
}

func (ts *TypedSubscriber[T]) Stats() SubscriberStats {
	return SubscriberStats{
		Received:  ts.received.Load(),
		Processed: ts.processed.Load(),
		Errors:    ts.errors.Load(),
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Domain-specific typed subscribers
// ─────────────────────────────────────────────────────────────────────────────

// BarSubscriber subscribes to bar events for one or more symbols.
type BarSubscriber struct {
	subs []*TypedSubscriber[BarEvent]
	log  *zap.Logger
}

// NewBarSubscriber subscribes to bar events for each symbol in the list.
// If symbols is empty, subscribes to a wildcard "market.bars.*".
func NewBarSubscriber(
	bus *EventBus,
	symbols []string,
	workers int,
	handler func(ctx context.Context, bar *BarEvent, evt *Event) error,
	log *zap.Logger,
) (*BarSubscriber, error) {
	bs := &BarSubscriber{log: log}
	if len(symbols) == 0 {
		symbols = []string{"*"}
	}
	for _, sym := range symbols {
		var topic Topic
		if sym == "*" {
			topic = "market.bars.*"
		} else {
			topic = ForSymbol(TopicMarketBars, sym)
		}
		ts, err := NewTypedSubscriber[BarEvent](bus, topic, workers, handler, log)
		if err != nil {
			bs.Close()
			return nil, fmt.Errorf("subscribe bars %s: %w", sym, err)
		}
		bs.subs = append(bs.subs, ts)
	}
	return bs, nil
}

func (bs *BarSubscriber) Close() {
	for _, ts := range bs.subs {
		ts.Unsubscribe()
	}
}

// SignalSubscriber subscribes to BH signal events.
type SignalSubscriber struct {
	subs []*TypedSubscriber[SignalEvent]
	log  *zap.Logger
}

// NewSignalSubscriber subscribes to signal events for each symbol.
func NewSignalSubscriber(
	bus *EventBus,
	symbols []string,
	strategyIDs []string,
	workers int,
	handler func(ctx context.Context, sig *SignalEvent, evt *Event) error,
	log *zap.Logger,
) (*SignalSubscriber, error) {
	ss := &SignalSubscriber{log: log}
	if len(symbols) == 0 {
		symbols = []string{"*"}
	}

	for _, sym := range symbols {
		var topic Topic
		if sym == "*" {
			topic = "bh.signal.*"
		} else {
			topic = ForSymbol(TopicBHSignal, sym)
		}

		// Wrap handler with strategy filter.
		strategyFilter := strategyIDs
		wrappedHandler := func(ctx context.Context, sig *SignalEvent, evt *Event) error {
			if len(strategyFilter) > 0 {
				found := false
				for _, id := range strategyFilter {
					if sig.StrategyID == id {
						found = true
						break
					}
				}
				if !found {
					return nil // skip this signal
				}
			}
			return handler(ctx, sig, evt)
		}

		ts, err := NewTypedSubscriber[SignalEvent](bus, topic, workers, wrappedHandler, log)
		if err != nil {
			ss.Close()
			return nil, err
		}
		ss.subs = append(ss.subs, ts)
	}
	return ss, nil
}

func (ss *SignalSubscriber) Close() {
	for _, ts := range ss.subs {
		ts.Unsubscribe()
	}
}

// TradeSubscriber subscribes to trade execution events.
type TradeSubscriber struct {
	sub *TypedSubscriber[TradeEvent]
}

// NewTradeSubscriber subscribes to all trade execution events.
func NewTradeSubscriber(
	bus *EventBus,
	workers int,
	handler func(ctx context.Context, trade *TradeEvent, evt *Event) error,
	log *zap.Logger,
) (*TradeSubscriber, error) {
	ts, err := NewTypedSubscriber[TradeEvent](bus, TopicTradeExecuted, workers, handler, log)
	if err != nil {
		return nil, err
	}
	return &TradeSubscriber{sub: ts}, nil
}

func (ts *TradeSubscriber) Close() { ts.sub.Unsubscribe() }

// RiskSubscriber subscribes to risk breach events.
type RiskSubscriber struct {
	subs []*TypedSubscriber[RiskBreachEvent]
	log  *zap.Logger
}

// NewRiskSubscriber subscribes to risk breach events, optionally filtered by account.
func NewRiskSubscriber(
	bus *EventBus,
	accountIDs []string,
	minSeverity string,
	workers int,
	handler func(ctx context.Context, breach *RiskBreachEvent, evt *Event) error,
	log *zap.Logger,
) (*RiskSubscriber, error) {
	rs := &RiskSubscriber{log: log}

	// Global breach topic.
	globalHandler := func(ctx context.Context, breach *RiskBreachEvent, evt *Event) error {
		if len(accountIDs) > 0 {
			for _, id := range accountIDs {
				if breach.AccountID == id {
					goto deliver
				}
			}
			return nil // filtered out
		}
	deliver:
		if minSeverity != "" && !meetsSeverity(breach.Severity, minSeverity) {
			return nil
		}
		return handler(ctx, breach, evt)
	}

	ts, err := NewTypedSubscriber[RiskBreachEvent](bus, TopicRiskBreach, workers, globalHandler, log)
	if err != nil {
		return nil, err
	}
	rs.subs = append(rs.subs, ts)

	// Per-account topics.
	for _, acctID := range accountIDs {
		topic := ForAccount(TopicRiskEvent, acctID)
		acctTS, err := NewTypedSubscriber[RiskBreachEvent](bus, topic, workers, func(ctx context.Context, breach *RiskBreachEvent, evt *Event) error {
			if minSeverity != "" && !meetsSeverity(breach.Severity, minSeverity) {
				return nil
			}
			return handler(ctx, breach, evt)
		}, log)
		if err != nil {
			rs.Close()
			return nil, err
		}
		rs.subs = append(rs.subs, acctTS)
	}
	return rs, nil
}

func (rs *RiskSubscriber) Close() {
	for _, ts := range rs.subs {
		ts.Unsubscribe()
	}
}

func meetsSeverity(actual, minimum string) bool {
	order := map[string]int{"info": 0, "warning": 1, "critical": 2, "halt": 3}
	return order[actual] >= order[minimum]
}

// PortfolioSubscriber subscribes to portfolio update events for an account.
type PortfolioSubscriber struct {
	subs []*TypedSubscriber[PortfolioUpdateEvent]
}

// NewPortfolioSubscriber subscribes to portfolio updates for the given account IDs.
func NewPortfolioSubscriber(
	bus *EventBus,
	accountIDs []string,
	workers int,
	handler func(ctx context.Context, update *PortfolioUpdateEvent, evt *Event) error,
	log *zap.Logger,
) (*PortfolioSubscriber, error) {
	ps := &PortfolioSubscriber{}
	if len(accountIDs) == 0 {
		accountIDs = []string{"*"}
	}
	for _, id := range accountIDs {
		var topic Topic
		if id == "*" {
			topic = "portfolio.update.*"
		} else {
			topic = ForAccount(TopicPortfolioUpdate, id)
		}
		ts, err := NewTypedSubscriber[PortfolioUpdateEvent](bus, topic, workers, handler, log)
		if err != nil {
			ps.Close()
			return nil, err
		}
		ps.subs = append(ps.subs, ts)
	}
	return ps, nil
}

func (ps *PortfolioSubscriber) Close() {
	for _, ts := range ps.subs {
		ts.Unsubscribe()
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchSubscriber — aggregates events into batches for efficient bulk processing
// ─────────────────────────────────────────────────────────────────────────────

// BatchConfig configures batch delivery.
type BatchConfig struct {
	MaxSize     int           // max events per batch
	MaxWait     time.Duration // max time to wait before flushing
	Workers     int
}

// BatchHandler is called with a batch of events.
type BatchHandler[T any] func(ctx context.Context, batch []*T) error

// BatchSubscriber buffers events and delivers them in batches.
type BatchSubscriber[T any] struct {
	mu      sync.Mutex
	buf     []*T
	timer   *time.Timer
	cfg     BatchConfig
	handler BatchHandler[T]
	inner   *TypedSubscriber[T]
	ctx     context.Context
	cancel  context.CancelFunc
	log     *zap.Logger
}

// NewBatchSubscriber creates a BatchSubscriber.
func NewBatchSubscriber[T any](
	bus *EventBus,
	topic Topic,
	cfg BatchConfig,
	handler BatchHandler[T],
	log *zap.Logger,
) (*BatchSubscriber[T], error) {
	if cfg.MaxSize <= 0 {
		cfg.MaxSize = 100
	}
	if cfg.MaxWait <= 0 {
		cfg.MaxWait = 100 * time.Millisecond
	}
	if cfg.Workers <= 0 {
		cfg.Workers = 1
	}

	bsCtx, cancel := context.WithCancel(context.Background())

	bs := &BatchSubscriber[T]{
		buf:     make([]*T, 0, cfg.MaxSize),
		cfg:     cfg,
		handler: handler,
		ctx:     bsCtx,
		cancel:  cancel,
		log:     log,
	}
	bs.timer = time.AfterFunc(cfg.MaxWait, bs.flush)

	inner, err := NewTypedSubscriber[T](bus, topic, cfg.Workers, func(ctx context.Context, payload *T, evt *Event) error {
		bs.add(payload)
		return nil
	}, log)
	if err != nil {
		cancel()
		return nil, err
	}
	bs.inner = inner
	return bs, nil
}

func (bs *BatchSubscriber[T]) add(item *T) {
	bs.mu.Lock()
	bs.buf = append(bs.buf, item)
	shouldFlush := len(bs.buf) >= bs.cfg.MaxSize
	bs.mu.Unlock()

	if shouldFlush {
		bs.flush()
	}
}

func (bs *BatchSubscriber[T]) flush() {
	bs.mu.Lock()
	if len(bs.buf) == 0 {
		bs.mu.Unlock()
		bs.timer.Reset(bs.cfg.MaxWait)
		return
	}
	batch := make([]*T, len(bs.buf))
	copy(batch, bs.buf)
	bs.buf = bs.buf[:0]
	bs.mu.Unlock()

	bs.timer.Reset(bs.cfg.MaxWait)

	if err := bs.handler(bs.ctx, batch); err != nil {
		bs.log.Error("batch handler error", zap.Error(err))
	}
}

func (bs *BatchSubscriber[T]) Close() {
	bs.cancel()
	bs.timer.Stop()
	bs.flush()
	bs.inner.Unsubscribe()
}
