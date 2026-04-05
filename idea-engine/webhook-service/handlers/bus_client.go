// Package handlers — bus_client.go
//
// BusClient is an HTTP client that publishes events to the idea-bus at :8768.
// It supports simple publish and publish-with-retry (exponential back-off).
package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"time"

	"go.uber.org/zap"
)

// BusEvent is the payload sent to the idea-bus.
type BusEvent struct {
	// Topic is the bus routing key, e.g. "signal.external" or "regime.change".
	Topic string `json:"topic"`
	// Source identifies the origin of the event (e.g. "tradingview", "custom").
	Source string `json:"source"`
	// EventType is a discriminator within the topic (e.g. "buy_signal").
	EventType string `json:"event_type"`
	// Timestamp is the UTC event creation time in RFC3339 format.
	Timestamp string `json:"timestamp"`
	// Payload is the event-specific data.
	Payload map[string]interface{} `json:"payload"`
}

// BusClient publishes events to the idea-bus.
type BusClient struct {
	baseURL    string
	httpClient *http.Client
	logger     *zap.Logger
}

// NewBusClient creates a new BusClient pointing at busBaseURL (e.g. "http://localhost:8768").
func NewBusClient(busBaseURL string, logger *zap.Logger) *BusClient {
	return &BusClient{
		baseURL: busBaseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		logger: logger,
	}
}

// Publish sends a single event to the bus.  It does NOT retry on failure.
// Returns an error if the bus returns a non-2xx status or if the network fails.
func (c *BusClient) Publish(ctx context.Context, event BusEvent) error {
	event.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)

	body, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("marshal event: %w", err)
	}

	url := fmt.Sprintf("%s/publish?topic=%s", c.baseURL, event.Topic)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("bus request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("bus returned status %d", resp.StatusCode)
	}

	c.logger.Debug("published to bus",
		zap.String("topic", event.Topic),
		zap.String("event_type", event.EventType),
		zap.String("source", event.Source),
	)
	return nil
}

// PublishWithRetry retries the publish up to maxRetries times using exponential
// back-off with ±20% random jitter.
//
// Back-off schedule (before jitter):
//
//	attempt 1 → 200 ms
//	attempt 2 → 400 ms
//	attempt 3 → 800 ms
//	attempt N → min(200 * 2^(N-1), 10 000) ms
func (c *BusClient) PublishWithRetry(ctx context.Context, event BusEvent, maxRetries int) error {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			delay := backoffDelay(attempt)
			c.logger.Warn("retrying bus publish",
				zap.Int("attempt", attempt),
				zap.Duration("delay", delay),
				zap.Error(lastErr),
			)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		if err := c.Publish(ctx, event); err != nil {
			lastErr = err
			continue
		}
		return nil // success
	}
	return fmt.Errorf("bus publish failed after %d retries: %w", maxRetries, lastErr)
}

// PublishSignal is a convenience wrapper for publishing external trading signals.
func (c *BusClient) PublishSignal(ctx context.Context, source, ticker, action string, extra map[string]interface{}) error {
	payload := map[string]interface{}{
		"ticker": ticker,
		"action": action,
	}
	for k, v := range extra {
		payload[k] = v
	}
	return c.PublishWithRetry(ctx, BusEvent{
		Topic:     "signal.external",
		Source:    source,
		EventType: fmt.Sprintf("signal.%s", action),
		Payload:   payload,
	}, 3)
}

// PublishRegimeChange publishes a regime change notification.
func (c *BusClient) PublishRegimeChange(ctx context.Context, source, regime string, confidence float64, extra map[string]interface{}) error {
	payload := map[string]interface{}{
		"regime":     regime,
		"confidence": confidence,
	}
	for k, v := range extra {
		payload[k] = v
	}
	return c.PublishWithRetry(ctx, BusEvent{
		Topic:     "regime.change",
		Source:    source,
		EventType: "regime.change",
		Payload:   payload,
	}, 3)
}

// PublishNewsEvent publishes a news event signal.
func (c *BusClient) PublishNewsEvent(ctx context.Context, source, headline, sentiment string, extra map[string]interface{}) error {
	payload := map[string]interface{}{
		"headline":  headline,
		"sentiment": sentiment,
	}
	for k, v := range extra {
		payload[k] = v
	}
	return c.PublishWithRetry(ctx, BusEvent{
		Topic:     "signal.news",
		Source:    source,
		EventType: "news.event",
		Payload:   payload,
	}, 3)
}

// backoffDelay computes an exponential back-off delay with ±20% jitter.
func backoffDelay(attempt int) time.Duration {
	base := 200.0 * math.Pow(2, float64(attempt-1))
	maxMs := 10_000.0
	if base > maxMs {
		base = maxMs
	}
	// ±20% jitter
	jitter := base * 0.2 * (rand.Float64()*2 - 1) //nolint:gosec
	ms := base + jitter
	if ms < 50 {
		ms = 50
	}
	return time.Duration(ms) * time.Millisecond
}
