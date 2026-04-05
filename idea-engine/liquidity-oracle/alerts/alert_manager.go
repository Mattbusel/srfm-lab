// Package alerts manages, deduplicates, and publishes liquidity alerts.
package alerts

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"srfm-lab/idea-engine/liquidity-oracle/scorer"
)

const (
	defaultBusURL       = "http://localhost:8768"
	alertRateLimit      = 15 * time.Minute
	alertTopic          = "liquidity.alert"
	alertProducer       = "liquidity-oracle"
	maxStoredAlerts     = 1000
)

// LiquidityAlert is a fired alert for a symbol.
type LiquidityAlert struct {
	AlertID        string                 `json:"alert_id"`
	Symbol         string                 `json:"symbol"`
	Score          scorer.LiquidityScore  `json:"score"`
	Reason         string                 `json:"reason"`
	FiredAt        time.Time              `json:"fired_at"`
}

// Subscriber receives alerts over HTTP POST.
type Subscriber struct {
	URL string
}

// Manager deduplicates and dispatches alerts.
type Manager struct {
	mu         sync.Mutex
	lastFired  map[string]time.Time // symbol -> last alert time
	active     []LiquidityAlert
	subscribers []Subscriber
	busURL     string
	client     *http.Client
	log        *slog.Logger
	counter    int64
}

// New creates an alert Manager.
func New(busURL string, log *slog.Logger) *Manager {
	if busURL == "" {
		busURL = defaultBusURL
	}
	return &Manager{
		lastFired: make(map[string]time.Time),
		busURL:    busURL,
		client:    &http.Client{Timeout: 5 * time.Second},
		log:       log,
	}
}

// MaybeAlert fires an alert for symbol if the score is below threshold and
// the rate limit has not been exceeded.
func (m *Manager) MaybeAlert(ctx context.Context, sc scorer.LiquidityScore) {
	if sc.Recommendation == scorer.RecommendationOK {
		return
	}

	m.mu.Lock()
	last, ok := m.lastFired[sc.Symbol]
	if ok && time.Since(last) < alertRateLimit {
		m.mu.Unlock()
		return
	}
	m.counter++
	id := fmt.Sprintf("liq-alert-%d", m.counter)
	m.lastFired[sc.Symbol] = time.Now()
	m.mu.Unlock()

	reason := fmt.Sprintf("composite=%.3f recommendation=%s spread=%.5f",
		sc.Composite, sc.Recommendation, sc.SpreadPct)

	alert := LiquidityAlert{
		AlertID: id,
		Symbol:  sc.Symbol,
		Score:   sc,
		Reason:  reason,
		FiredAt: time.Now().UTC(),
	}

	m.mu.Lock()
	m.active = append(m.active, alert)
	if len(m.active) > maxStoredAlerts {
		m.active = m.active[len(m.active)-maxStoredAlerts:]
	}
	subs := make([]Subscriber, len(m.subscribers))
	copy(subs, m.subscribers)
	m.mu.Unlock()

	m.log.Info("liquidity alert fired",
		"symbol", sc.Symbol,
		"composite", sc.Composite,
		"recommendation", sc.Recommendation,
	)

	// Publish to IAE bus.
	go m.publishToBus(ctx, alert)

	// Notify HTTP subscribers.
	for _, sub := range subs {
		go m.notifySubscriber(ctx, sub, alert)
	}
}

// ActiveAlerts returns the current stored alerts.
func (m *Manager) ActiveAlerts() []LiquidityAlert {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]LiquidityAlert, len(m.active))
	copy(out, m.active)
	return out
}

// AddSubscriber registers a new HTTP subscriber URL.
func (m *Manager) AddSubscriber(sub Subscriber) {
	m.mu.Lock()
	m.subscribers = append(m.subscribers, sub)
	m.mu.Unlock()
}

type busEvent struct {
	EventID      string          `json:"event_id"`
	Topic        string          `json:"topic"`
	Payload      json.RawMessage `json:"payload"`
	ProducedAt   time.Time       `json:"produced_at"`
	ProducerName string          `json:"producer_name"`
}

func (m *Manager) publishToBus(ctx context.Context, alert LiquidityAlert) {
	raw, err := json.Marshal(alert)
	if err != nil {
		m.log.Error("alert: marshal payload", "err", err)
		return
	}
	evt := busEvent{
		EventID:      alert.AlertID,
		Topic:        alertTopic,
		Payload:      json.RawMessage(raw),
		ProducedAt:   time.Now().UTC(),
		ProducerName: alertProducer,
	}
	body, err := json.Marshal(evt)
	if err != nil {
		m.log.Error("alert: marshal event", "err", err)
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.busURL+"/publish", bytes.NewReader(body))
	if err != nil {
		m.log.Error("alert: build bus request", "err", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := m.client.Do(req)
	if err != nil {
		m.log.Warn("alert: bus publish failed", "err", err)
		return
	}
	resp.Body.Close()
}

func (m *Manager) notifySubscriber(ctx context.Context, sub Subscriber, alert LiquidityAlert) {
	body, err := json.Marshal(alert)
	if err != nil {
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, sub.URL, bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := m.client.Do(req)
	if err != nil {
		m.log.Warn("alert: notify subscriber failed", "url", sub.URL, "err", err)
		return
	}
	resp.Body.Close()
}
