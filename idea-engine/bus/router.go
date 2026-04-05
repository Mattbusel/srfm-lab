package bus

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"srfm-lab/idea-engine/bus/events"
)

// SubscriptionID uniquely identifies a subscription on the router.
type SubscriptionID string

// subscription holds a handler function registered for a specific topic.
type subscription struct {
	id      SubscriptionID
	topic   string
	handler func(events.Event)
}

// Router is an in-memory pub/sub router. It is safe for concurrent use by
// multiple goroutines. Every published event is delivered synchronously to
// all subscribers registered for that topic before Publish returns, so
// handlers must not block for extended periods.
//
// For crash recovery the router delegates persistence to a Persister
// (implemented by persistence.go) so that every event is also written to
// SQLite.
type Router struct {
	mu      sync.RWMutex
	subs    map[string][]subscription // topic -> subscriptions
	log     *zap.Logger
	persist Persister
}

// Persister is the interface the router calls after every successful publish.
type Persister interface {
	PersistEvent(event events.Event) error
}

// NewRouter constructs a Router. If persist is nil no persistence is applied.
func NewRouter(log *zap.Logger, persist Persister) *Router {
	return &Router{
		subs:    make(map[string][]subscription),
		log:     log,
		persist: persist,
	}
}

// Subscribe registers handler to be called whenever an event is published on
// topic. It returns a SubscriptionID that can be passed to Unsubscribe later.
func (r *Router) Subscribe(topic string, handler func(events.Event)) SubscriptionID {
	id := SubscriptionID(fmt.Sprintf("sub-%s-%d", topic, time.Now().UnixNano()))

	r.mu.Lock()
	r.subs[topic] = append(r.subs[topic], subscription{
		id:      id,
		topic:   topic,
		handler: handler,
	})
	r.mu.Unlock()

	r.log.Debug("bus subscription registered",
		zap.String("topic", topic),
		zap.String("subscription_id", string(id)),
	)
	return id
}

// Unsubscribe removes the subscription identified by id from the router.
// It is a no-op if the id does not exist.
func (r *Router) Unsubscribe(id SubscriptionID) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for topic, subs := range r.subs {
		for i, s := range subs {
			if s.id == id {
				r.subs[topic] = append(subs[:i], subs[i+1:]...)
				r.log.Debug("bus subscription removed",
					zap.String("topic", topic),
					zap.String("subscription_id", string(id)),
				)
				return
			}
		}
	}
}

// Publish marshals payload as JSON and delivers the resulting Event to every
// subscriber registered for topic. The event is also persisted via the
// Persister if one was supplied at construction time.
//
// Publish returns an error only for structural failures (unknown topic,
// marshal error, persist error); handler panics are recovered and logged but
// do not cause Publish to return an error.
func (r *Router) Publish(topic string, producerName string, payload interface{}) error {
	if !IsValidTopic(topic) {
		return fmt.Errorf("bus: unknown topic %q", topic)
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("bus: marshal payload for topic %q: %w", topic, err)
	}

	evt := events.Event{
		EventID:      uuid.NewString(),
		Topic:        topic,
		Payload:      json.RawMessage(raw),
		ProducedAt:   time.Now().UTC(),
		ProducerName: producerName,
	}

	// Persist before delivering so that the event is durably recorded even if
	// a subscriber panics.
	if r.persist != nil {
		if err := r.persist.PersistEvent(evt); err != nil {
			r.log.Error("bus: failed to persist event",
				zap.String("topic", topic),
				zap.String("event_id", evt.EventID),
				zap.Error(err),
			)
			// Non-fatal: continue delivering to in-memory subscribers.
		}
	}

	// Snapshot the subscriber list under read lock so we don't hold the lock
	// while calling handlers (which may themselves publish).
	r.mu.RLock()
	subs := make([]subscription, len(r.subs[topic]))
	copy(subs, r.subs[topic])
	r.mu.RUnlock()

	for _, s := range subs {
		r.deliverSafe(s, evt)
	}

	r.log.Debug("bus: event published",
		zap.String("topic", topic),
		zap.String("event_id", evt.EventID),
		zap.Int("subscribers", len(subs)),
	)
	return nil
}

// deliverSafe calls the subscription handler, recovering from any panics so
// that a misbehaving handler cannot crash the whole bus.
func (r *Router) deliverSafe(s subscription, evt events.Event) {
	defer func() {
		if rec := recover(); rec != nil {
			r.log.Error("bus: subscriber panicked",
				zap.String("topic", evt.Topic),
				zap.String("subscription_id", string(s.id)),
				zap.Any("panic", rec),
			)
		}
	}()
	s.handler(evt)
}

// SubscriberCount returns the number of active subscriptions for topic.
// Primarily useful for tests and health endpoints.
func (r *Router) SubscriberCount(topic string) int {
	r.mu.RLock()
	n := len(r.subs[topic])
	r.mu.RUnlock()
	return n
}

// TopicCounts returns a snapshot map of topic -> subscriber count for all
// topics that have at least one subscription.
func (r *Router) TopicCounts() map[string]int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]int, len(r.subs))
	for topic, subs := range r.subs {
		if len(subs) > 0 {
			out[topic] = len(subs)
		}
	}
	return out
}
