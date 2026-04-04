// bus.go — EventBus: the central coordinator for all pub/sub operations.
package eventbus

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// Event — the universal message envelope
// ─────────────────────────────────────────────────────────────────────────────

// Event is the universal envelope for all messages on the bus.
type Event struct {
	// ID is a globally unique event identifier (UUID v4).
	ID string `json:"id" msgpack:"id"`

	// Topic is the routing key.
	Topic Topic `json:"topic" msgpack:"topic"`

	// Type is a human-readable event type discriminator.
	Type string `json:"type" msgpack:"type"`

	// Payload contains the JSON-encoded event body.
	Payload json.RawMessage `json:"payload" msgpack:"payload"`

	// Source identifies the publishing service.
	Source string `json:"source" msgpack:"source"`

	// Timestamp is the event creation time (UTC, nanosecond precision).
	Timestamp time.Time `json:"timestamp" msgpack:"timestamp"`

	// Sequence is a monotonic per-topic counter (set by the bus).
	Sequence int64 `json:"sequence" msgpack:"sequence"`

	// TraceID carries the OpenTelemetry trace ID for distributed tracing.
	TraceID string `json:"trace_id,omitempty" msgpack:"trace_id,omitempty"`

	// SchemaVersion allows payload version negotiation.
	SchemaVersion string `json:"schema_version,omitempty" msgpack:"schema_version,omitempty"`

	// Metadata holds arbitrary key-value pairs for routing/filtering.
	Metadata map[string]string `json:"metadata,omitempty" msgpack:"metadata,omitempty"`
}

// NewEvent constructs an Event with a generated UUID and current timestamp.
func NewEvent(topic Topic, evtType string, payload interface{}, source string) (*Event, error) {
	var raw json.RawMessage
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal payload: %w", err)
		}
		raw = b
	}
	return &Event{
		ID:        uuid.New().String(),
		Topic:     topic,
		Type:      evtType,
		Payload:   raw,
		Source:    source,
		Timestamp: time.Now().UTC(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Bus configuration
// ─────────────────────────────────────────────────────────────────────────────

// BusConfig holds all tuning parameters for EventBus.
type BusConfig struct {
	// RedisAddr is the Redis server address (default: "localhost:6379").
	RedisAddr string
	// RedisPassword is the Redis AUTH password (empty = no auth).
	RedisPassword string
	// RedisDB is the Redis database index.
	RedisDB int

	// ChannelBufferSize is the internal publish channel buffer depth.
	ChannelBufferSize int

	// WorkerCount is the number of goroutines dispatching local subscriptions.
	WorkerCount int

	// MaxRetries for Redis operations.
	MaxRetries int

	// Serialization format: "json" or "msgpack".
	Serialization string

	// EnablePersistence writes events to Redis Streams in addition to Pub/Sub.
	EnablePersistence bool

	// DLQEnabled enables dead-letter queue for failed deliveries.
	DLQEnabled bool

	// DLQMaxRetries is the number of times a message is retried before DLQ.
	DLQMaxRetries int
}

// DefaultBusConfig returns production-ready defaults.
func DefaultBusConfig() BusConfig {
	return BusConfig{
		RedisAddr:         "localhost:6379",
		ChannelBufferSize: 4096,
		WorkerCount:       8,
		MaxRetries:        3,
		Serialization:     "json",
		EnablePersistence: true,
		DLQEnabled:        true,
		DLQMaxRetries:     3,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// EventBus
// ─────────────────────────────────────────────────────────────────────────────

// EventBus is the central pub/sub coordinator.
// It provides:
//   - Redis Pub/Sub for low-latency fanout
//   - Redis Streams for durable event log
//   - In-process topic router for zero-copy local delivery
//   - Dead-letter queue for undeliverable messages
//   - At-least-once delivery via consumer groups
type EventBus struct {
	cfg    BusConfig
	log    *zap.Logger
	rdb    *redis.Client
	router *TopicRouter
	ser    Serializer

	// Sequence counters per topic.
	seqMu  sync.Mutex
	seqMap map[Topic]*int64

	// Publish pipeline.
	publishCh chan publishJob

	// Subscription registry.
	subMu sync.RWMutex
	subs  map[string]*subscription // sub ID → subscription

	// Redis Pub/Sub connections (one per subscribed channel set).
	redisPubSubs []*redis.PubSub

	// Metrics.
	metrics *busMetrics

	// Lifecycle.
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

type publishJob struct {
	event  *Event
	doneCh chan error
}

type subscription struct {
	id        string
	topic     Topic
	handler   EventHandler
	workerCh  chan *Event
	retries   int
}

// EventHandler is the callback invoked when an event arrives.
type EventHandler func(ctx context.Context, evt *Event) error

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

// NewEventBus creates an EventBus and connects to Redis.
func NewEventBus(cfg BusConfig, log *zap.Logger) (*EventBus, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:         cfg.RedisAddr,
		Password:     cfg.RedisPassword,
		DB:           cfg.RedisDB,
		MaxRetries:   cfg.MaxRetries,
		PoolSize:     cfg.WorkerCount * 2,
		MinIdleConns: cfg.WorkerCount,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping: %w", err)
	}

	var ser Serializer
	if cfg.Serialization == "msgpack" {
		ser = &MsgpackSerializer{}
	} else {
		ser = &JSONSerializer{}
	}

	busCtx, busCancel := context.WithCancel(context.Background())

	b := &EventBus{
		cfg:       cfg,
		log:       log,
		rdb:       rdb,
		router:    NewTopicRouter(),
		ser:       ser,
		seqMap:    make(map[Topic]*int64),
		publishCh: make(chan publishJob, cfg.ChannelBufferSize),
		subs:      make(map[string]*subscription),
		metrics:   newBusMetrics(),
		ctx:       busCtx,
		cancel:    busCancel,
	}

	// Start publish workers.
	for i := 0; i < cfg.WorkerCount; i++ {
		b.wg.Add(1)
		go b.publishWorker()
	}

	return b, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Publish
// ─────────────────────────────────────────────────────────────────────────────

// Publish sends an event synchronously. It returns when the event has been
// dispatched to local subscribers and optionally written to Redis Streams.
func (b *EventBus) Publish(ctx context.Context, evt *Event) error {
	if evt.ID == "" {
		evt.ID = uuid.New().String()
	}
	evt.Sequence = b.nextSeq(evt.Topic)
	if evt.Timestamp.IsZero() {
		evt.Timestamp = time.Now().UTC()
	}

	doneCh := make(chan error, 1)
	select {
	case b.publishCh <- publishJob{event: evt, doneCh: doneCh}:
	case <-ctx.Done():
		return ctx.Err()
	}

	select {
	case err := <-doneCh:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// PublishAsync sends an event without waiting for delivery. Fire-and-forget.
func (b *EventBus) PublishAsync(evt *Event) {
	if evt.ID == "" {
		evt.ID = uuid.New().String()
	}
	evt.Sequence = b.nextSeq(evt.Topic)
	if evt.Timestamp.IsZero() {
		evt.Timestamp = time.Now().UTC()
	}
	select {
	case b.publishCh <- publishJob{event: evt, doneCh: nil}:
	default:
		b.log.Warn("publish channel full, dropping event", zap.String("topic", string(evt.Topic)))
		b.metrics.dropped.Add(1)
	}
}

// publishWorker drains the publish channel.
func (b *EventBus) publishWorker() {
	defer b.wg.Done()
	for {
		select {
		case <-b.ctx.Done():
			return
		case job := <-b.publishCh:
			err := b.doPublish(b.ctx, job.event)
			if job.doneCh != nil {
				job.doneCh <- err
			}
		}
	}
}

// doPublish performs the actual delivery: local router + Redis.
func (b *EventBus) doPublish(ctx context.Context, evt *Event) error {
	start := time.Now()

	// 1. Local in-process dispatch.
	b.router.Dispatch(evt)

	// 2. Redis Pub/Sub (low-latency, at-most-once).
	data, err := b.ser.Marshal(evt)
	if err != nil {
		b.log.Error("serialize event failed", zap.Error(err))
		b.metrics.errors.Add(1)
		return err
	}

	if err := b.rdb.Publish(ctx, string(evt.Topic), data).Err(); err != nil {
		b.log.Warn("Redis PUBLISH failed", zap.String("topic", string(evt.Topic)), zap.Error(err))
		b.metrics.errors.Add(1)
		// Don't return — still try Streams.
	}

	// 3. Redis Streams (durable, at-least-once).
	if b.cfg.EnablePersistence {
		meta := GetTopicMeta(evt.Topic)
		if meta == nil || meta.Persistent {
			streamKey := StreamKey(evt.Topic, 0)
			args := &redis.XAddArgs{
				Stream: streamKey,
				Values: map[string]interface{}{
					"id":        evt.ID,
					"type":      evt.Type,
					"source":    evt.Source,
					"timestamp": evt.Timestamp.UnixNano(),
					"payload":   string(data),
				},
			}
			if meta != nil && meta.MaxLen > 0 {
				args.MaxLen = meta.MaxLen
				args.Approx = true
			}
			if _, err := b.rdb.XAdd(ctx, args).Err(); err != nil {
				b.log.Warn("Redis XADD failed", zap.String("stream", streamKey), zap.Error(err))
				// Non-fatal for Pub/Sub mode.
			}
		}
	}

	elapsed := time.Since(start)
	b.metrics.publishLatency.observe(elapsed)
	b.metrics.published.Add(1)
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Subscribe
// ─────────────────────────────────────────────────────────────────────────────

// Subscribe registers a handler for events on the given topic.
// Returns a subscription ID that can be used to unsubscribe.
func (b *EventBus) Subscribe(topic Topic, handler EventHandler) (string, error) {
	subID := uuid.New().String()
	workerCh := make(chan *Event, 256)

	sub := &subscription{
		id:       subID,
		topic:    topic,
		handler:  handler,
		workerCh: workerCh,
	}

	b.subMu.Lock()
	b.subs[subID] = sub
	b.subMu.Unlock()

	// Register with local router.
	b.router.Subscribe(topic, subID, workerCh)

	// Subscribe to Redis Pub/Sub channel.
	ps := b.rdb.Subscribe(b.ctx, string(topic))
	b.redisPubSubs = append(b.redisPubSubs, ps)

	// Start a goroutine to relay Redis Pub/Sub messages to the handler.
	b.wg.Add(1)
	go b.redisPubSubRelay(ps, sub)

	// Start a goroutine that processes the local channel.
	b.wg.Add(1)
	go b.subscriptionWorker(sub)

	b.metrics.subscribers.Add(1)
	return subID, nil
}

// Unsubscribe removes a subscription by ID.
func (b *EventBus) Unsubscribe(subID string) {
	b.subMu.Lock()
	sub, ok := b.subs[subID]
	if ok {
		delete(b.subs, subID)
	}
	b.subMu.Unlock()

	if ok {
		b.router.Unsubscribe(sub.topic, subID)
		close(sub.workerCh)
		b.metrics.subscribers.Add(-1)
	}
}

// redisPubSubRelay reads Redis Pub/Sub messages and pushes them to the handler channel.
func (b *EventBus) redisPubSubRelay(ps *redis.PubSub, sub *subscription) {
	defer b.wg.Done()
	ch := ps.Channel()
	for {
		select {
		case <-b.ctx.Done():
			_ = ps.Close()
			return
		case msg, ok := <-ch:
			if !ok {
				return
			}
			evt, err := b.ser.Unmarshal([]byte(msg.Payload))
			if err != nil {
				b.log.Warn("deserialize Redis PubSub message failed", zap.Error(err))
				b.metrics.errors.Add(1)
				continue
			}
			select {
			case sub.workerCh <- evt:
				b.metrics.received.Add(1)
			default:
				b.log.Warn("subscription worker channel full",
					zap.String("sub_id", sub.id),
					zap.String("topic", string(sub.topic)))
				b.metrics.dropped.Add(1)
				if b.cfg.DLQEnabled {
					b.sendToDLQ(evt)
				}
			}
		}
	}
}

// subscriptionWorker processes events from the local channel.
func (b *EventBus) subscriptionWorker(sub *subscription) {
	defer b.wg.Done()
	for evt := range sub.workerCh {
		b.deliverWithRetry(sub, evt)
	}
}

// deliverWithRetry invokes the handler, retrying on error up to DLQMaxRetries.
func (b *EventBus) deliverWithRetry(sub *subscription, evt *Event) {
	var lastErr error
	for attempt := 0; attempt <= b.cfg.DLQMaxRetries; attempt++ {
		if err := sub.handler(b.ctx, evt); err != nil {
			lastErr = err
			if attempt < b.cfg.DLQMaxRetries {
				wait := time.Duration(50*(1<<attempt)) * time.Millisecond
				time.Sleep(wait)
				continue
			}
			b.log.Warn("handler failed after retries, sending to DLQ",
				zap.String("sub_id", sub.id),
				zap.String("event_id", evt.ID),
				zap.Error(lastErr))
			b.metrics.dlqMessages.Add(1)
			if b.cfg.DLQEnabled {
				b.sendToDLQ(evt)
			}
		}
		return
	}
}

// sendToDLQ writes an undeliverable event to the dead-letter queue stream.
func (b *EventBus) sendToDLQ(evt *Event) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	data, err := b.ser.Marshal(evt)
	if err != nil {
		return
	}

	dlqKey := DLQKey(evt.Topic)
	_ = b.rdb.XAdd(ctx, &redis.XAddArgs{
		Stream: dlqKey,
		MaxLen: 10_000,
		Approx: true,
		Values: map[string]interface{}{
			"event_id":   evt.ID,
			"topic":      string(evt.Topic),
			"payload":    string(data),
			"failed_at":  time.Now().UnixNano(),
		},
	}).Err()
}

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

// Close gracefully shuts down the EventBus.
func (b *EventBus) Close() error {
	b.cancel()
	b.wg.Wait()

	for _, ps := range b.redisPubSubs {
		_ = ps.Close()
	}
	return b.rdb.Close()
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequence counter
// ─────────────────────────────────────────────────────────────────────────────

func (b *EventBus) nextSeq(topic Topic) int64 {
	b.seqMu.Lock()
	defer b.seqMu.Unlock()
	if _, ok := b.seqMap[topic]; !ok {
		var zero int64
		b.seqMap[topic] = &zero
	}
	atomic.AddInt64(b.seqMap[topic], 1)
	return atomic.LoadInt64(b.seqMap[topic])
}

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────

type busMetrics struct {
	published   atomic.Int64
	received    atomic.Int64
	dropped     atomic.Int64
	errors      atomic.Int64
	subscribers atomic.Int64
	dlqMessages atomic.Int64

	publishLatency *latencyHistogram
}

func newBusMetrics() *busMetrics {
	return &busMetrics{publishLatency: newLatencyHistogram(32)}
}

// Stats returns a snapshot of current bus metrics.
func (b *EventBus) Stats() BusStats {
	return BusStats{
		Published:       b.metrics.published.Load(),
		Received:        b.metrics.received.Load(),
		Dropped:         b.metrics.dropped.Load(),
		Errors:          b.metrics.errors.Load(),
		Subscribers:     b.metrics.subscribers.Load(),
		DLQMessages:     b.metrics.dlqMessages.Load(),
		P50LatencyUS:    b.metrics.publishLatency.percentile(50),
		P95LatencyUS:    b.metrics.publishLatency.percentile(95),
		P99LatencyUS:    b.metrics.publishLatency.percentile(99),
	}
}

// BusStats is a snapshot of bus performance metrics.
type BusStats struct {
	Published    int64
	Received     int64
	Dropped      int64
	Errors       int64
	Subscribers  int64
	DLQMessages  int64
	P50LatencyUS float64
	P95LatencyUS float64
	P99LatencyUS float64
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple latency histogram (lock-free ring buffer)
// ─────────────────────────────────────────────────────────────────────────────

type latencyHistogram struct {
	mu      sync.Mutex
	samples []float64
	size    int
	pos     int
	count   int
}

func newLatencyHistogram(size int) *latencyHistogram {
	return &latencyHistogram{samples: make([]float64, size), size: size}
}

func (h *latencyHistogram) observe(d time.Duration) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.samples[h.pos%h.size] = float64(d.Microseconds())
	h.pos++
	if h.count < h.size {
		h.count++
	}
}

func (h *latencyHistogram) percentile(p float64) float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.count == 0 {
		return 0
	}
	cp := make([]float64, h.count)
	copy(cp, h.samples[:h.count])
	// Simple sort.
	n := len(cp)
	for i := 1; i < n; i++ {
		for j := i; j > 0 && cp[j] < cp[j-1]; j-- {
			cp[j], cp[j-1] = cp[j-1], cp[j]
		}
	}
	idx := int(p/100*float64(n)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}
	return cp[idx]
}
