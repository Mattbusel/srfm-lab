// Package handlers — custom.go
//
// CustomHandler handles generic and specialised webhook endpoints:
//
//   POST /webhooks/custom       — generic event injection
//   POST /webhooks/regime-alert — external regime change notifications
//   POST /webhooks/news-event   — major news/announcement signals
//
// All handlers validate the incoming payload, map the event type to the
// appropriate bus topic, and publish via the BusClient.
package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// ---------------------------------------------------------------------------
// Payload types
// ---------------------------------------------------------------------------

// CustomWebhookPayload is the generic webhook body.
type CustomWebhookPayload struct {
	// EventType is a dot-separated event identifier, e.g. "signal.entry".
	EventType string `json:"event_type"`
	// Source identifies the system that generated the event.
	Source string `json:"source"`
	// Timestamp is an optional ISO8601 time; defaults to server receive time.
	Timestamp string `json:"timestamp,omitempty"`
	// Data is the arbitrary event payload.
	Data map[string]interface{} `json:"data"`
}

func (p *CustomWebhookPayload) Validate() error {
	if strings.TrimSpace(p.EventType) == "" {
		return fmt.Errorf("event_type is required")
	}
	if strings.TrimSpace(p.Source) == "" {
		return fmt.Errorf("source is required")
	}
	if p.Data == nil {
		return fmt.Errorf("data must not be null")
	}
	return nil
}

// eventTypeToBusTopic maps custom event types to bus topics.
// Unmapped types fall back to "webhook.custom".
var eventTypeToBusTopic = map[string]string{
	"signal.entry":       "signal.external",
	"signal.exit":        "signal.external",
	"signal.buy":         "signal.external",
	"signal.sell":        "signal.external",
	"regime.change":      "regime.change",
	"regime.alert":       "regime.change",
	"news.event":         "signal.news",
	"risk.alert":         "risk.alert",
	"portfolio.rebalance": "portfolio.event",
	"system.heartbeat":   "system.heartbeat",
}

func resolveTopic(eventType string) string {
	if topic, ok := eventTypeToBusTopic[strings.ToLower(eventType)]; ok {
		return topic
	}
	// Hierarchical fallback: "foo.bar.baz" → try "foo.bar" → try "foo"
	parts := strings.Split(strings.ToLower(eventType), ".")
	for i := len(parts) - 1; i > 0; i-- {
		prefix := strings.Join(parts[:i], ".")
		if topic, ok := eventTypeToBusTopic[prefix]; ok {
			return topic
		}
	}
	return "webhook.custom"
}

// ---------------------------------------------------------------------------
// Regime alert payload
// ---------------------------------------------------------------------------

// RegimeAlertPayload is the body for POST /webhooks/regime-alert.
type RegimeAlertPayload struct {
	// Regime is the new market regime identifier (e.g. "bull", "bear", "neutral").
	Regime string `json:"regime"`
	// Confidence is the model's confidence in [0, 1].
	Confidence float64 `json:"confidence"`
	// Model is the name/version of the regime detection model.
	Model string `json:"model,omitempty"`
	// Ticker is the instrument the regime applies to (optional, "" = global).
	Ticker string `json:"ticker,omitempty"`
	// Timeframe is the analysis timeframe (e.g. "1D", "1W").
	Timeframe string `json:"timeframe,omitempty"`
	// PreviousRegime is the previous regime, for transition tracking.
	PreviousRegime string `json:"previous_regime,omitempty"`
	// Source identifies the originating system.
	Source string `json:"source"`
	// Timestamp is optional; defaults to receive time.
	Timestamp string `json:"timestamp,omitempty"`
	// Extra is any additional key-value metadata.
	Extra map[string]interface{} `json:"extra,omitempty"`
}

func (p *RegimeAlertPayload) Validate() error {
	if strings.TrimSpace(p.Regime) == "" {
		return fmt.Errorf("regime is required")
	}
	if p.Confidence < 0 || p.Confidence > 1 {
		return fmt.Errorf("confidence must be in [0, 1], got %v", p.Confidence)
	}
	if strings.TrimSpace(p.Source) == "" {
		return fmt.Errorf("source is required")
	}
	return nil
}

// ---------------------------------------------------------------------------
// News event payload
// ---------------------------------------------------------------------------

// NewsEventPayload is the body for POST /webhooks/news-event.
type NewsEventPayload struct {
	// Headline is the news headline text.
	Headline string `json:"headline"`
	// Summary is an optional longer summary.
	Summary string `json:"summary,omitempty"`
	// Sentiment is the pre-computed sentiment: "positive" | "negative" | "neutral".
	Sentiment string `json:"sentiment"`
	// SentimentScore is a numeric score in [-1, 1].
	SentimentScore float64 `json:"sentiment_score,omitempty"`
	// Tickers is a list of affected tickers.
	Tickers []string `json:"tickers,omitempty"`
	// Categories are event categories (e.g. "earnings", "macro", "regulatory").
	Categories []string `json:"categories,omitempty"`
	// Source is the news source name.
	Source string `json:"source"`
	// URL is a link to the full article (optional).
	URL string `json:"url,omitempty"`
	// Timestamp is optional; defaults to receive time.
	Timestamp string `json:"timestamp,omitempty"`
	// Urgency is a qualitative urgency level: "low" | "medium" | "high" | "critical".
	Urgency string `json:"urgency,omitempty"`
}

func (p *NewsEventPayload) Validate() error {
	if strings.TrimSpace(p.Headline) == "" {
		return fmt.Errorf("headline is required")
	}
	switch strings.ToLower(p.Sentiment) {
	case "positive", "negative", "neutral", "":
		// ok
	default:
		return fmt.Errorf("sentiment must be positive|negative|neutral, got %q", p.Sentiment)
	}
	if strings.TrimSpace(p.Source) == "" {
		return fmt.Errorf("source is required")
	}
	if p.SentimentScore < -1 || p.SentimentScore > 1 {
		return fmt.Errorf("sentiment_score must be in [-1, 1]")
	}
	return nil
}

// ---------------------------------------------------------------------------
// CustomHandler
// ---------------------------------------------------------------------------

// CustomHandler processes generic and specialised webhook endpoints.
type CustomHandler struct {
	bus     *BusClient
	logger  *zap.Logger
	metrics *MetricsRegistry
}

// NewCustomHandler creates a new CustomHandler.
func NewCustomHandler(bus *BusClient, logger *zap.Logger, metrics *MetricsRegistry) *CustomHandler {
	return &CustomHandler{bus: bus, logger: logger, metrics: metrics}
}

// HandleCustom processes POST /webhooks/custom.
func (h *CustomHandler) HandleCustom(w http.ResponseWriter, r *http.Request) {
	h.metrics.WebhooksReceived.Add(1)

	body, err := readBody(r, 256*1024)
	if err != nil {
		respondError(w, http.StatusBadRequest, "could not read body: "+err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	var payload CustomWebhookPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		respondError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	if err := payload.Validate(); err != nil {
		respondError(w, http.StatusUnprocessableEntity, err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	if payload.Timestamp == "" {
		payload.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	}

	topic := resolveTopic(payload.EventType)
	eventID := uuid.New().String()

	h.logger.Info("custom webhook received",
		zap.String("event_type", payload.EventType),
		zap.String("source", payload.Source),
		zap.String("topic", topic),
		zap.String("event_id", eventID),
	)

	busPayload := map[string]interface{}{
		"event_id":   eventID,
		"event_type": payload.EventType,
		"source":     payload.Source,
		"timestamp":  payload.Timestamp,
		"data":       payload.Data,
	}

	event := BusEvent{
		Topic:     topic,
		Source:    payload.Source,
		EventType: payload.EventType,
		Payload:   busPayload,
	}

	if err := h.bus.PublishWithRetry(r.Context(), event, 3); err != nil {
		h.logger.Error("bus publish failed", zap.Error(err))
		respondError(w, http.StatusServiceUnavailable, "could not publish to event bus")
		h.metrics.WebhookErrors.Add(1)
		return
	}

	h.metrics.WebhooksProcessed.Add(1)
	respondJSON(w, http.StatusAccepted, map[string]interface{}{
		"status":     "accepted",
		"event_id":   eventID,
		"bus_topic":  topic,
		"event_type": payload.EventType,
	})
}

// HandleRegimeAlert processes POST /webhooks/regime-alert.
func (h *CustomHandler) HandleRegimeAlert(w http.ResponseWriter, r *http.Request) {
	h.metrics.WebhooksReceived.Add(1)

	body, err := readBody(r, 64*1024)
	if err != nil {
		respondError(w, http.StatusBadRequest, "could not read body: "+err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	var payload RegimeAlertPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		respondError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	if err := payload.Validate(); err != nil {
		respondError(w, http.StatusUnprocessableEntity, err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	if payload.Timestamp == "" {
		payload.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	}

	eventID := uuid.New().String()

	h.logger.Info("regime alert received",
		zap.String("regime", payload.Regime),
		zap.Float64("confidence", payload.Confidence),
		zap.String("source", payload.Source),
		zap.String("ticker", payload.Ticker),
		zap.String("event_id", eventID),
	)

	extra := map[string]interface{}{
		"event_id":        eventID,
		"model":           payload.Model,
		"ticker":          payload.Ticker,
		"timeframe":       payload.Timeframe,
		"previous_regime": payload.PreviousRegime,
		"timestamp":       payload.Timestamp,
	}
	if payload.Extra != nil {
		for k, v := range payload.Extra {
			extra[k] = v
		}
	}

	if err := h.bus.PublishRegimeChange(r.Context(), payload.Source, payload.Regime, payload.Confidence, extra); err != nil {
		h.logger.Error("bus publish failed", zap.Error(err))
		respondError(w, http.StatusServiceUnavailable, "could not publish to event bus")
		h.metrics.WebhookErrors.Add(1)
		return
	}

	h.metrics.WebhooksProcessed.Add(1)
	respondJSON(w, http.StatusAccepted, map[string]interface{}{
		"status":     "accepted",
		"event_id":   eventID,
		"bus_topic":  "regime.change",
		"regime":     payload.Regime,
		"confidence": payload.Confidence,
	})
}

// HandleNewsEvent processes POST /webhooks/news-event.
func (h *CustomHandler) HandleNewsEvent(w http.ResponseWriter, r *http.Request) {
	h.metrics.WebhooksReceived.Add(1)

	body, err := readBody(r, 256*1024)
	if err != nil {
		respondError(w, http.StatusBadRequest, "could not read body: "+err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	var payload NewsEventPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		respondError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	if err := payload.Validate(); err != nil {
		respondError(w, http.StatusUnprocessableEntity, err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	if payload.Timestamp == "" {
		payload.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	}
	if payload.Urgency == "" {
		payload.Urgency = "medium"
	}

	eventID := uuid.New().String()

	h.logger.Info("news event received",
		zap.String("headline", truncate(payload.Headline, 80)),
		zap.String("sentiment", payload.Sentiment),
		zap.Float64("sentiment_score", payload.SentimentScore),
		zap.String("source", payload.Source),
		zap.String("urgency", payload.Urgency),
		zap.String("event_id", eventID),
	)

	extra := map[string]interface{}{
		"event_id":        eventID,
		"summary":         payload.Summary,
		"sentiment_score": payload.SentimentScore,
		"tickers":         payload.Tickers,
		"categories":      payload.Categories,
		"url":             payload.URL,
		"urgency":         payload.Urgency,
		"timestamp":       payload.Timestamp,
	}

	if err := h.bus.PublishNewsEvent(r.Context(), payload.Source, payload.Headline, payload.Sentiment, extra); err != nil {
		h.logger.Error("bus publish failed", zap.Error(err))
		respondError(w, http.StatusServiceUnavailable, "could not publish to event bus")
		h.metrics.WebhookErrors.Add(1)
		return
	}

	h.metrics.WebhooksProcessed.Add(1)
	respondJSON(w, http.StatusAccepted, map[string]interface{}{
		"status":          "accepted",
		"event_id":        eventID,
		"bus_topic":       "signal.news",
		"sentiment":       payload.Sentiment,
		"sentiment_score": payload.SentimentScore,
		"urgency":         payload.Urgency,
	})
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func readBody(r *http.Request, maxBytes int64) ([]byte, error) {
	return io.ReadAll(io.LimitReader(r.Body, maxBytes))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
