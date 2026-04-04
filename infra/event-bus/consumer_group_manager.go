// consumer_group_manager.go — Higher-level consumer group lifecycle management.
// Handles group creation, health monitoring, lag alerts, and graceful shutdown.
package eventbus

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// ConsumerGroupManager
// ─────────────────────────────────────────────────────────────────────────────

// ConsumerGroupManager coordinates multiple ConsumerGroups across topics.
// It provides:
//   - Centralised lag monitoring
//   - Automatic rebalancing hints
//   - Consumer heartbeating
//   - Admin API (list groups, reset offsets, delete groups)
type ConsumerGroupManager struct {
	rdb     *redis.Client
	ser     Serializer
	log     *zap.Logger
	cfg     ConsumerGroupManagerConfig

	mu     sync.RWMutex
	groups map[string]*ConsumerGroup // groupName:streamKey → group

	// Lag monitoring.
	lagAlertCh chan LagAlert

	// Lifecycle.
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// ConsumerGroupManagerConfig configures the manager.
type ConsumerGroupManagerConfig struct {
	// LagAlertThreshold: alert when lag exceeds this many messages.
	LagAlertThreshold int64
	// MonitorInterval is how often to check lag.
	MonitorInterval time.Duration
	// HeartbeatInterval is how often consumer heartbeats are written to Redis.
	HeartbeatInterval time.Duration
	// MaxLagAlertCh is the capacity of the lag alert channel.
	MaxLagAlertCh int
}

// DefaultConsumerGroupManagerConfig returns sensible defaults.
func DefaultConsumerGroupManagerConfig() ConsumerGroupManagerConfig {
	return ConsumerGroupManagerConfig{
		LagAlertThreshold: 10_000,
		MonitorInterval:   15 * time.Second,
		HeartbeatInterval: 5 * time.Second,
		MaxLagAlertCh:     128,
	}
}

// LagAlert represents a consumer group lag alert.
type LagAlert struct {
	GroupName  string
	StreamKey  string
	Lag        int64
	AlertedAt  time.Time
}

// NewConsumerGroupManager creates a ConsumerGroupManager.
func NewConsumerGroupManager(rdb *redis.Client, ser Serializer, cfg ConsumerGroupManagerConfig, log *zap.Logger) *ConsumerGroupManager {
	ctx, cancel := context.WithCancel(context.Background())
	return &ConsumerGroupManager{
		rdb:        rdb,
		ser:        ser,
		log:        log,
		cfg:        cfg,
		groups:     make(map[string]*ConsumerGroup),
		lagAlertCh: make(chan LagAlert, cfg.MaxLagAlertCh),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// RegisterGroup registers a ConsumerGroup with the manager for monitoring.
func (m *ConsumerGroupManager) RegisterGroup(group *ConsumerGroup) {
	key := fmt.Sprintf("%s:%s", group.cfg.GroupName, group.cfg.ConsumerName)
	m.mu.Lock()
	m.groups[key] = group
	m.mu.Unlock()
}

// Start begins monitoring goroutines.
func (m *ConsumerGroupManager) Start() {
	m.wg.Add(1)
	go m.monitorLoop()

	m.wg.Add(1)
	go m.heartbeatLoop()
}

// Stop shuts down all managed consumer groups and the manager.
func (m *ConsumerGroupManager) Stop() {
	m.cancel()
	m.mu.RLock()
	groups := make([]*ConsumerGroup, 0, len(m.groups))
	for _, g := range m.groups {
		groups = append(groups, g)
	}
	m.mu.RUnlock()

	for _, g := range groups {
		g.Stop()
	}
	m.wg.Wait()
}

// LagAlerts returns the channel on which lag alerts are published.
func (m *ConsumerGroupManager) LagAlerts() <-chan LagAlert {
	return m.lagAlertCh
}

// monitorLoop periodically checks consumer group lag.
func (m *ConsumerGroupManager) monitorLoop() {
	defer m.wg.Done()
	ticker := time.NewTicker(m.cfg.MonitorInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.checkLag()
		}
	}
}

// checkLag inspects lag for all known consumer groups.
func (m *ConsumerGroupManager) checkLag() {
	m.mu.RLock()
	groups := make([]*ConsumerGroup, 0, len(m.groups))
	for _, g := range m.groups {
		groups = append(groups, g)
	}
	m.mu.RUnlock()

	ctx, cancel := context.WithTimeout(m.ctx, 10*time.Second)
	defer cancel()

	for _, group := range groups {
		for _, entry := range group.topics {
			for _, streamKey := range entry.streamKeys {
				lag, err := StreamLag(ctx, m.rdb, streamKey, group.cfg.GroupName)
				if err != nil {
					m.log.Warn("lag check failed",
						zap.String("stream", streamKey),
						zap.String("group", group.cfg.GroupName),
						zap.Error(err))
					continue
				}
				streamLagGauge.WithLabelValues(streamKey, group.cfg.GroupName).Set(float64(lag))

				if lag > m.cfg.LagAlertThreshold {
					alert := LagAlert{
						GroupName: group.cfg.GroupName,
						StreamKey: streamKey,
						Lag:       lag,
						AlertedAt: time.Now().UTC(),
					}
					select {
					case m.lagAlertCh <- alert:
					default:
						m.log.Warn("lag alert channel full",
							zap.String("stream", streamKey),
							zap.Int64("lag", lag))
					}
				}
			}
		}
	}
}

// heartbeatLoop writes consumer heartbeats to Redis for membership tracking.
func (m *ConsumerGroupManager) heartbeatLoop() {
	defer m.wg.Done()
	ticker := time.NewTicker(m.cfg.HeartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.writeHeartbeats()
		}
	}
}

func (m *ConsumerGroupManager) writeHeartbeats() {
	m.mu.RLock()
	groups := make([]*ConsumerGroup, 0, len(m.groups))
	for _, g := range m.groups {
		groups = append(groups, g)
	}
	m.mu.RUnlock()

	ctx, cancel := context.WithTimeout(m.ctx, 5*time.Second)
	defer cancel()

	for _, group := range groups {
		hbKey := fmt.Sprintf("consumer:heartbeat:%s:%s",
			group.cfg.GroupName, group.cfg.ConsumerName)
		_ = m.rdb.Set(ctx, hbKey, time.Now().UnixNano(), 30*time.Second).Err()
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Admin operations
// ─────────────────────────────────────────────────────────────────────────────

// GroupInfo holds information about a consumer group.
type GroupInfo struct {
	Name        string
	StreamKey   string
	Consumers   int64
	Pending     int64
	Lag         int64
	LastEntryID string
}

// ListGroups returns information about all consumer groups on a stream.
func (m *ConsumerGroupManager) ListGroups(ctx context.Context, streamKey string) ([]GroupInfo, error) {
	groups, err := m.rdb.XInfoGroups(ctx, streamKey).Result()
	if err != nil {
		return nil, fmt.Errorf("XINFO GROUPS %s: %w", streamKey, err)
	}

	out := make([]GroupInfo, len(groups))
	for i, g := range groups {
		out[i] = GroupInfo{
			Name:      g.Name,
			StreamKey: streamKey,
			Consumers: g.Consumers,
			Pending:   g.Pending,
			Lag:       g.Lag,
		}
	}
	return out, nil
}

// ResetOffset resets a consumer group to read from a specific stream offset.
// Use "0" to replay from the beginning, or "$" for only new messages.
func (m *ConsumerGroupManager) ResetOffset(ctx context.Context, streamKey, groupName, offset string) error {
	err := m.rdb.XGroupSetID(ctx, streamKey, groupName, offset).Err()
	if err != nil {
		return fmt.Errorf("XGROUP SETID: %w", err)
	}
	m.log.Info("consumer group offset reset",
		zap.String("stream", streamKey),
		zap.String("group", groupName),
		zap.String("offset", offset))
	return nil
}

// DeleteGroup removes a consumer group from a stream.
func (m *ConsumerGroupManager) DeleteGroup(ctx context.Context, streamKey, groupName string) error {
	if err := m.rdb.XGroupDestroy(ctx, streamKey, groupName).Err(); err != nil {
		return fmt.Errorf("XGROUP DESTROY: %w", err)
	}
	m.log.Info("consumer group deleted",
		zap.String("stream", streamKey),
		zap.String("group", groupName))
	return nil
}

// PurgeStream removes all entries from a stream and resets the consumer group.
func (m *ConsumerGroupManager) PurgeStream(ctx context.Context, streamKey, groupName string) error {
	if err := m.rdb.XTrimMaxLen(ctx, streamKey, 0).Err(); err != nil {
		return fmt.Errorf("XTRIM: %w", err)
	}
	m.log.Info("stream purged", zap.String("stream", streamKey))
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline — wraps multiple consumer groups behind a single Start/Stop interface
// ─────────────────────────────────────────────────────────────────────────────

// Pipeline combines multiple ConsumerGroups and a ConsumerGroupManager into
// a single unit that can be started and stopped together.
type Pipeline struct {
	manager *ConsumerGroupManager
	groups  []*ConsumerGroup
	log     *zap.Logger
}

// NewPipeline creates a Pipeline.
func NewPipeline(manager *ConsumerGroupManager, log *zap.Logger) *Pipeline {
	return &Pipeline{manager: manager, log: log}
}

// AddGroup adds a ConsumerGroup to the pipeline.
func (p *Pipeline) AddGroup(group *ConsumerGroup) {
	p.groups = append(p.groups, group)
	p.manager.RegisterGroup(group)
}

// Start begins all consumer groups and the manager.
func (p *Pipeline) Start() {
	for _, g := range p.groups {
		g.Start()
	}
	p.manager.Start()
}

// Stop gracefully shuts down the pipeline.
func (p *Pipeline) Stop() {
	p.manager.Stop()
}

// LagAlerts returns the lag alert channel from the manager.
func (p *Pipeline) LagAlerts() <-chan LagAlert {
	return p.manager.LagAlerts()
}

// ─────────────────────────────────────────────────────────────────────────────
// EventFilter — server-side event filtering middleware
// ─────────────────────────────────────────────────────────────────────────────

// FilterFunc is a predicate applied to events.
type FilterFunc func(evt *Event) bool

// FilteredHandler wraps an EventHandler with filter predicates.
// The event is only delivered if ALL filters return true.
func FilteredHandler(filters []FilterFunc, handler EventHandler) EventHandler {
	return func(ctx context.Context, evt *Event) error {
		for _, f := range filters {
			if !f(evt) {
				return nil // filtered out, not an error
			}
		}
		return handler(ctx, evt)
	}
}

// FilterByType returns a FilterFunc that accepts only events of a specific type.
func FilterByType(evtTypes ...string) FilterFunc {
	allowed := make(map[string]bool, len(evtTypes))
	for _, t := range evtTypes {
		allowed[t] = true
	}
	return func(evt *Event) bool {
		return allowed[evt.Type]
	}
}

// FilterBySource returns a FilterFunc that accepts only events from specific sources.
func FilterBySource(sources ...string) FilterFunc {
	allowed := make(map[string]bool, len(sources))
	for _, s := range sources {
		allowed[s] = true
	}
	return func(evt *Event) bool {
		return allowed[evt.Source]
	}
}

// FilterByMetadata returns a FilterFunc that checks event metadata key-value pairs.
func FilterByMetadata(key, value string) FilterFunc {
	return func(evt *Event) bool {
		if evt.Metadata == nil {
			return false
		}
		return evt.Metadata[key] == value
	}
}

// FilterBefore accepts only events whose timestamp is before t.
func FilterBefore(t time.Time) FilterFunc {
	return func(evt *Event) bool {
		return evt.Timestamp.Before(t)
	}
}

// FilterAfter accepts only events whose timestamp is after t.
func FilterAfter(t time.Time) FilterFunc {
	return func(evt *Event) bool {
		return evt.Timestamp.After(t)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// EventTransformer — transforms events before delivery
// ─────────────────────────────────────────────────────────────────────────────

// TransformFunc transforms an event before delivery.
type TransformFunc func(evt *Event) *Event

// TransformedHandler wraps an EventHandler with transformer functions.
func TransformedHandler(transforms []TransformFunc, handler EventHandler) EventHandler {
	return func(ctx context.Context, evt *Event) error {
		transformed := evt
		for _, t := range transforms {
			transformed = t(transformed)
			if transformed == nil {
				return nil // transform dropped the event
			}
		}
		return handler(ctx, transformed)
	}
}

// SetSource returns a TransformFunc that overrides the event source.
func SetSource(source string) TransformFunc {
	return func(evt *Event) *Event {
		cp := *evt
		cp.Source = source
		return &cp
	}
}

// AddMetadata returns a TransformFunc that adds metadata key-value pairs.
func AddMetadata(kv map[string]string) TransformFunc {
	return func(evt *Event) *Event {
		cp := *evt
		if cp.Metadata == nil {
			cp.Metadata = make(map[string]string, len(kv))
		} else {
			// Clone metadata.
			m := make(map[string]string, len(cp.Metadata)+len(kv))
			for k, v := range cp.Metadata {
				m[k] = v
			}
			cp.Metadata = m
		}
		for k, v := range kv {
			cp.Metadata[k] = v
		}
		return &cp
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// EventAggregator — aggregates events over a time window
// ─────────────────────────────────────────────────────────────────────────────

// Aggregation functions.
type AggregateFunc func(events []*Event) *Event

// EventAggregator buffers events and emits aggregated events on a timer.
type EventAggregator struct {
	bus     *EventBus
	mu      sync.Mutex
	buffer  []*Event
	ticker  *time.Ticker
	window  time.Duration
	aggFn   AggregateFunc
	outTopic Topic
	source  string
	ctx     context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup
}

// NewEventAggregator creates an EventAggregator.
func NewEventAggregator(bus *EventBus, window time.Duration, outTopic Topic, source string, aggFn AggregateFunc) *EventAggregator {
	ctx, cancel := context.WithCancel(context.Background())
	ea := &EventAggregator{
		bus:      bus,
		buffer:   make([]*Event, 0, 100),
		ticker:   time.NewTicker(window),
		window:   window,
		aggFn:    aggFn,
		outTopic: outTopic,
		source:   source,
		ctx:      ctx,
		cancel:   cancel,
	}
	ea.wg.Add(1)
	go ea.flushLoop()
	return ea
}

// Add adds an event to the aggregation buffer.
func (ea *EventAggregator) Add(evt *Event) {
	ea.mu.Lock()
	ea.buffer = append(ea.buffer, evt)
	ea.mu.Unlock()
}

func (ea *EventAggregator) flushLoop() {
	defer ea.wg.Done()
	for {
		select {
		case <-ea.ctx.Done():
			ea.flush()
			return
		case <-ea.ticker.C:
			ea.flush()
		}
	}
}

func (ea *EventAggregator) flush() {
	ea.mu.Lock()
	if len(ea.buffer) == 0 {
		ea.mu.Unlock()
		return
	}
	batch := make([]*Event, len(ea.buffer))
	copy(batch, ea.buffer)
	ea.buffer = ea.buffer[:0]
	ea.mu.Unlock()

	aggregated := ea.aggFn(batch)
	if aggregated == nil {
		return
	}
	aggregated.Topic = ea.outTopic
	aggregated.Source = ea.source

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := ea.bus.Publish(ctx, aggregated); err != nil {
		// Log but don't propagate.
		_ = err
	}
}

// Stop halts the aggregator and flushes any remaining events.
func (ea *EventAggregator) Stop() {
	ea.ticker.Stop()
	ea.cancel()
	ea.wg.Wait()
}
