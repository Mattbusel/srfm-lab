// mock.go — In-memory mock EventBus for unit testing without Redis.
package eventbus

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// ─────────────────────────────────────────────────────────────────────────────
// MockEventBus
// ─────────────────────────────────────────────────────────────────────────────

// MockEventBus is a fully in-process EventBus implementation.
// It is safe for concurrent use and suitable for unit tests.
type MockEventBus struct {
	mu     sync.RWMutex
	router *TopicRouter
	events []*Event // audit log of all published events

	subMu sync.RWMutex
	subs  map[string]*mockSubscription

	seqMu  sync.Mutex
	seqMap map[Topic]*int64

	published  atomic.Int64
	dropped    atomic.Int64

	// Optional: simulated latency for testing backpressure.
	SimulatedLatency time.Duration

	// Optional: drop ratio for chaos testing.
	DropRatio float64

	closed bool
}

type mockSubscription struct {
	id       string
	topic    Topic
	handler  EventHandler
	ch       chan *Event
	cancelFn context.CancelFunc
	wg       sync.WaitGroup
}

// NewMockEventBus creates an in-memory MockEventBus.
func NewMockEventBus() *MockEventBus {
	return &MockEventBus{
		router: NewTopicRouter(),
		subs:   make(map[string]*mockSubscription),
		seqMap: make(map[Topic]*int64),
	}
}

// Publish delivers an event to all subscribers synchronously.
func (m *MockEventBus) Publish(ctx context.Context, evt *Event) error {
	if m.closed {
		return fmt.Errorf("bus is closed")
	}
	if evt.ID == "" {
		evt.ID = uuid.New().String()
	}
	evt.Sequence = m.nextSeq(evt.Topic)
	if evt.Timestamp.IsZero() {
		evt.Timestamp = time.Now().UTC()
	}

	// Chaos: random drops.
	if m.DropRatio > 0 {
		// Simple deterministic drop based on sequence.
		if float64(evt.Sequence%100)/100.0 < m.DropRatio {
			m.dropped.Add(1)
			return nil
		}
	}

	if m.SimulatedLatency > 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(m.SimulatedLatency):
		}
	}

	m.mu.Lock()
	m.events = append(m.events, evt)
	m.published.Add(1)
	m.mu.Unlock()

	m.router.Dispatch(evt)

	// Deliver to registered handler subs.
	m.subMu.RLock()
	var handlers []*mockSubscription
	for _, sub := range m.subs {
		if topicMatchesMock(sub.topic, evt.Topic) {
			handlers = append(handlers, sub)
		}
	}
	m.subMu.RUnlock()

	for _, sub := range handlers {
		select {
		case sub.ch <- evt:
		default:
			m.dropped.Add(1)
		}
	}
	return nil
}

// PublishAsync publishes without waiting.
func (m *MockEventBus) PublishAsync(evt *Event) {
	_ = m.Publish(context.Background(), evt)
}

// Subscribe registers a handler for the given topic. Returns a subscription ID.
func (m *MockEventBus) Subscribe(topic Topic, handler EventHandler) (string, error) {
	subID := uuid.New().String()
	ch := make(chan *Event, 256)
	subCtx, cancel := context.WithCancel(context.Background())

	sub := &mockSubscription{
		id:       subID,
		topic:    topic,
		handler:  handler,
		ch:       ch,
		cancelFn: cancel,
	}

	m.subMu.Lock()
	m.subs[subID] = sub
	m.subMu.Unlock()

	m.router.Subscribe(topic, subID, ch)

	sub.wg.Add(1)
	go func() {
		defer sub.wg.Done()
		for {
			select {
			case <-subCtx.Done():
				return
			case evt, ok := <-ch:
				if !ok {
					return
				}
				_ = handler(subCtx, evt)
			}
		}
	}()

	return subID, nil
}

// Unsubscribe removes a subscription.
func (m *MockEventBus) Unsubscribe(subID string) {
	m.subMu.Lock()
	sub, ok := m.subs[subID]
	if ok {
		delete(m.subs, subID)
	}
	m.subMu.Unlock()

	if ok {
		m.router.Unsubscribe(sub.topic, subID)
		sub.cancelFn()
		close(sub.ch)
		sub.wg.Wait()
	}
}

// Close shuts down the mock bus.
func (m *MockEventBus) Close() error {
	m.mu.Lock()
	m.closed = true
	m.mu.Unlock()

	m.subMu.Lock()
	ids := make([]string, 0, len(m.subs))
	for id := range m.subs {
		ids = append(ids, id)
	}
	m.subMu.Unlock()

	for _, id := range ids {
		m.Unsubscribe(id)
	}
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

// AllEvents returns a copy of all events published to the bus.
func (m *MockEventBus) AllEvents() []*Event {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cp := make([]*Event, len(m.events))
	copy(cp, m.events)
	return cp
}

// EventsForTopic returns all published events matching the given topic.
func (m *MockEventBus) EventsForTopic(topic Topic) []*Event {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var out []*Event
	for _, evt := range m.events {
		if topicMatchesMock(topic, evt.Topic) {
			out = append(out, evt)
		}
	}
	return out
}

// EventsOfType returns all events with the given event type.
func (m *MockEventBus) EventsOfType(evtType string) []*Event {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var out []*Event
	for _, evt := range m.events {
		if evt.Type == evtType {
			out = append(out, evt)
		}
	}
	return out
}

// Reset clears the event log and subscriber list. Useful between test cases.
func (m *MockEventBus) Reset() {
	m.mu.Lock()
	m.events = m.events[:0]
	m.mu.Unlock()
}

// WaitForEvents blocks until n events matching the topic have been published or timeout occurs.
func (m *MockEventBus) WaitForEvents(topic Topic, n int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if len(m.EventsForTopic(topic)) >= n {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return false
}

// PublishedCount returns the total number of events published.
func (m *MockEventBus) PublishedCount() int64 {
	return m.published.Load()
}

// DroppedCount returns the total number of events dropped.
func (m *MockEventBus) DroppedCount() int64 {
	return m.dropped.Load()
}

// Stats returns current mock bus statistics.
func (m *MockEventBus) Stats() BusStats {
	return BusStats{
		Published: m.published.Load(),
		Dropped:   m.dropped.Load(),
	}
}

func (m *MockEventBus) nextSeq(topic Topic) int64 {
	m.seqMu.Lock()
	defer m.seqMu.Unlock()
	if _, ok := m.seqMap[topic]; !ok {
		var zero int64
		m.seqMap[topic] = &zero
	}
	atomic.AddInt64(m.seqMap[topic], 1)
	return atomic.LoadInt64(m.seqMap[topic])
}

// topicMatchesMock checks if an event topic matches a subscription topic pattern.
func topicMatchesMock(pattern, evtTopic Topic) bool {
	if pattern == evtTopic {
		return true
	}
	// Wildcard match.
	return matchSegments(
		splitTopic(string(evtTopic)),
		splitTopic(string(pattern)),
	)
}

func splitTopic(s string) []string {
	out := []string{}
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == '.' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return out
}

// ─────────────────────────────────────────────────────────────────────────────
// MockPublishers — convenience wrappers that use MockEventBus
// ─────────────────────────────────────────────────────────────────────────────

// MockBarPublisher wraps BarPublisher over a MockEventBus for tests.
type MockBarPublisher struct {
	*BarPublisher
	mock *MockEventBus
}

// NewMockBarPublisher creates a MockBarPublisher backed by the given mock bus.
func NewMockBarPublisher(mock *MockEventBus, source string) *MockBarPublisher {
	// We need the real BarPublisher struct, but pointed at our mock bus.
	// Since BarPublisher uses *EventBus which has unexported fields, we
	// embed it and directly call the real EventBus methods via mock adapter.
	return &MockBarPublisher{
		BarPublisher: &BarPublisher{bus: mockToBus(mock), source: source},
		mock:         mock,
	}
}

// mockToBus creates a minimal *EventBus wrapper around a MockEventBus.
// This is a compile-time shim; in tests prefer using MockEventBus directly.
func mockToBus(m *MockEventBus) *EventBus {
	// We construct an EventBus with no Redis connection.
	// The mock's Publish/Subscribe methods are used instead.
	busCtx, cancel := context.WithCancel(context.Background())
	b := &EventBus{
		router:    m.router,
		ser:       &JSONSerializer{},
		seqMap:    m.seqMap,
		publishCh: make(chan publishJob, 256),
		subs:      make(map[string]*subscription),
		metrics:   newBusMetrics(),
		ctx:       busCtx,
		cancel:    cancel,
	}
	// Override doPublish to use mock.
	// Since doPublish is not an interface, we wire publish via channel.
	go func() {
		for {
			select {
			case <-busCtx.Done():
				return
			case job := <-b.publishCh:
				evt := job.event
				m.mu.Lock()
				m.events = append(m.events, evt)
				m.published.Add(1)
				m.mu.Unlock()
				b.router.Dispatch(evt)
				if job.doneCh != nil {
					job.doneCh <- nil
				}
			}
		}
	}()
	return b
}
