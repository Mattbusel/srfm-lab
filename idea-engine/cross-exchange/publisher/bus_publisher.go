// Package publisher sends divergence and basis signals to the IAE event bus.
package publisher

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"srfm-lab/idea-engine/cross-exchange/basis"
	"srfm-lab/idea-engine/cross-exchange/divergence"
)

const (
	defaultBusURL      = "http://localhost:8768"
	publishPath        = "/publish"
	producerName       = "cross-exchange"
	topicDivergence    = "market.divergence"
	topicBasis         = "market.basis"
)

// BusEvent is the envelope sent to the IAE event bus.
type BusEvent struct {
	EventID      string          `json:"event_id"`
	Topic        string          `json:"topic"`
	Payload      json.RawMessage `json:"payload"`
	ProducedAt   time.Time       `json:"produced_at"`
	ProducerName string          `json:"producer_name"`
}

// Publisher posts events to the IAE bus via HTTP.
type Publisher struct {
	busURL string
	client *http.Client
	log    *slog.Logger
}

// New creates a Publisher targeting busURL (defaults to http://localhost:8768).
func New(busURL string, log *slog.Logger) *Publisher {
	if busURL == "" {
		busURL = defaultBusURL
	}
	return &Publisher{
		busURL: busURL,
		client: &http.Client{Timeout: 5 * time.Second},
		log:    log,
	}
}

// PublishDivergence sends a divergence signal to the bus.
func (p *Publisher) PublishDivergence(ctx context.Context, sig divergence.DivergenceSignal) error {
	return p.publish(ctx, topicDivergence, sig)
}

// PublishBasis sends a basis signal to the bus.
func (p *Publisher) PublishBasis(ctx context.Context, sig basis.BasisSignal) error {
	return p.publish(ctx, topicBasis, sig)
}

func (p *Publisher) publish(ctx context.Context, topic string, payload interface{}) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("publisher: marshal payload: %w", err)
	}

	evt := BusEvent{
		EventID:      fmt.Sprintf("%s-%d", producerName, time.Now().UnixNano()),
		Topic:        topic,
		Payload:      json.RawMessage(raw),
		ProducedAt:   time.Now().UTC(),
		ProducerName: producerName,
	}

	body, err := json.Marshal(evt)
	if err != nil {
		return fmt.Errorf("publisher: marshal event: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.busURL+publishPath, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("publisher: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("publisher: post to bus: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		return fmt.Errorf("publisher: bus returned %d for topic %s", resp.StatusCode, topic)
	}

	p.log.Debug("publisher: event sent", "topic", topic)
	return nil
}
