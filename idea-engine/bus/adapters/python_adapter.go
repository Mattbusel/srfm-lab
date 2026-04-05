// Package adapters contains inbound HTTP adapters that allow non-Go services
// (primarily Python modules) to publish events onto the bus.
package adapters

import (
	"encoding/json"
	"net/http"
	"time"

	"go.uber.org/zap"

	"srfm-lab/idea-engine/bus"
	"srfm-lab/idea-engine/bus/events"
)

// publishRequest is the JSON body expected by the /events endpoint.
type publishRequest struct {
	// Topic must be one of the registered topic constants.
	Topic string `json:"topic"`
	// ProducerName identifies the Python module sending the event.
	ProducerName string `json:"producer_name"`
	// Payload is an arbitrary JSON object; it is stored verbatim on the Event.
	Payload json.RawMessage `json:"payload"`
}

// publishResponse is the JSON body returned on 202 Accepted.
type publishResponse struct {
	EventID    string    `json:"event_id"`
	Topic      string    `json:"topic"`
	AcceptedAt time.Time `json:"accepted_at"`
}

// PythonAdapter exposes an HTTP endpoint at POST /events that Python
// services call to inject events into the Go bus. It validates the topic,
// unmarshals the JSON body, calls Router.Publish, and returns 202 Accepted
// immediately (fire-and-forget from the caller's perspective).
type PythonAdapter struct {
	router *bus.Router
	log    *zap.Logger
}

// NewPythonAdapter constructs a PythonAdapter backed by the given router.
func NewPythonAdapter(router *bus.Router, log *zap.Logger) *PythonAdapter {
	return &PythonAdapter{router: router, log: log}
}

// HandlePublish is the HTTP handler for POST /events.
// It accepts a JSON body, validates the topic, publishes to the bus, and
// responds 202 Accepted without waiting for subscriber delivery.
func (a *PythonAdapter) HandlePublish(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{
			"error": "method not allowed; use POST",
		})
		return
	}

	// Limit the request body to 4 MiB to avoid memory exhaustion.
	r.Body = http.MaxBytesReader(w, r.Body, 4<<20)

	var req publishRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		a.log.Warn("python_adapter: bad request body", zap.Error(err))
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "invalid JSON body: " + err.Error(),
		})
		return
	}

	if req.Topic == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "field 'topic' is required",
		})
		return
	}

	if !bus.IsValidTopic(req.Topic) {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "unknown topic: " + req.Topic,
		})
		return
	}

	if req.ProducerName == "" {
		req.ProducerName = "python-unknown"
	}

	if len(req.Payload) == 0 {
		req.Payload = json.RawMessage(`{}`)
	}

	// Publish is synchronous but fast; the adapter returns 202 regardless of
	// delivery latency because all handlers should be non-blocking.
	if err := a.router.Publish(req.Topic, req.ProducerName, req.Payload); err != nil {
		a.log.Error("python_adapter: publish failed",
			zap.String("topic", req.Topic),
			zap.Error(err),
		)
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": "publish failed: " + err.Error(),
		})
		return
	}

	a.log.Info("python_adapter: event accepted",
		zap.String("topic", req.Topic),
		zap.String("producer", req.ProducerName),
	)

	// We don't have the EventID here because Publish constructs it internally.
	// Return a minimal acknowledgement; the event_id will be visible in the DB.
	writeJSON(w, http.StatusAccepted, publishResponse{
		Topic:      req.Topic,
		AcceptedAt: time.Now().UTC(),
	})
}

// HandleTopics lists all valid topic names. Useful for Python services to
// discover available topics at startup.
func (a *PythonAdapter) HandleTopics(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"topics": bus.AllTopics(),
		"count":  len(bus.AllTopics()),
	})
}

// HandleReplay allows Python services to request a replay of events for a
// given topic since a given timestamp. Query params: topic, since (RFC3339).
func (a *PythonAdapter) HandleReplay(store ReplayStore, w http.ResponseWriter, r *http.Request) {
	topic := r.URL.Query().Get("topic")
	sinceStr := r.URL.Query().Get("since")

	if topic == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "query param 'topic' is required",
		})
		return
	}
	if !bus.IsValidTopic(topic) {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "unknown topic: " + topic,
		})
		return
	}

	since := time.Time{}
	if sinceStr != "" {
		var err error
		since, err = time.Parse(time.RFC3339, sinceStr)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{
				"error": "invalid 'since' timestamp; use RFC3339 format",
			})
			return
		}
	}

	evts, err := store.ReplayTopic(topic, since)
	if err != nil {
		a.log.Error("python_adapter: replay failed",
			zap.String("topic", topic),
			zap.Error(err),
		)
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": "replay failed: " + err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"topic":  topic,
		"since":  since.UTC().Format(time.RFC3339),
		"events": evts,
		"count":  len(evts),
	})
}

// ReplayStore is the interface the HandleReplay helper requires.
type ReplayStore interface {
	ReplayTopic(topic string, since time.Time) ([]events.Event, error)
}

// writeJSON writes v as indented JSON with the given status code.
func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}
