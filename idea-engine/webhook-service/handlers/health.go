// Package handlers — health.go
//
// HealthHandler provides:
//
//   GET /health   — liveness probe (always 200 if the process is running)
//   GET /ready    — readiness probe (200 once the bus is reachable)
//   GET /metrics  — JSON service counters
package handlers

import (
	"context"
	"net/http"
	"runtime"
	"time"

	"go.uber.org/zap"
)

// HealthHandler serves health and metrics endpoints.
type HealthHandler struct {
	metrics   *MetricsRegistry
	logger    *zap.Logger
	startTime time.Time
	busClient *BusClient // optional; used for readiness check
}

// NewHealthHandler creates a new HealthHandler.
// busClient may be nil; if set, /ready will ping the bus.
func NewHealthHandler(metrics *MetricsRegistry, logger *zap.Logger) *HealthHandler {
	return &HealthHandler{
		metrics:   metrics,
		logger:    logger,
		startTime: time.Now(),
	}
}

// WithBusClient attaches a bus client for the readiness check.
func (h *HealthHandler) WithBusClient(bus *BusClient) *HealthHandler {
	h.busClient = bus
	return h
}

// Health handles GET /health.
// Always returns 200 with a JSON body while the process is alive.
func (h *HealthHandler) Health(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "ok",
		"service": "webhook-service",
		"uptime":  h.metrics.UptimeSeconds(),
	})
}

// Ready handles GET /ready.
// Returns 200 if the service is ready to handle traffic, 503 otherwise.
// Readiness is determined by a lightweight bus connectivity check (optional).
func (h *HealthHandler) Ready(w http.ResponseWriter, r *http.Request) {
	if h.busClient != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()

		// We do a dry-run publish to the bus health topic.
		// The bus is expected to accept and discard system.ping events.
		event := BusEvent{
			Topic:     "system.ping",
			Source:    "webhook-service",
			EventType: "health.ping",
			Payload:   map[string]interface{}{"check": "readiness"},
		}
		if err := h.busClient.Publish(ctx, event); err != nil {
			h.logger.Warn("readiness check: bus unreachable", zap.Error(err))
			respondJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
				"status": "not_ready",
				"reason": "bus_unreachable",
				"error":  err.Error(),
			})
			return
		}
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "ready",
		"service": "webhook-service",
	})
}

// Metrics handles GET /metrics.
// Returns JSON counters for monitoring dashboards.
func (h *HealthHandler) Metrics(w http.ResponseWriter, r *http.Request) {
	snap := h.metrics.Snapshot()

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	respondJSON(w, http.StatusOK, map[string]interface{}{
		// Request counters
		"webhooks_received":  snap.WebhooksReceived,
		"webhooks_processed": snap.WebhooksProcessed,
		"errors":             snap.Errors,
		"uptime_seconds":     snap.UptimeSeconds,

		// Derived rates (per-minute, computed over lifetime)
		"received_per_min": ratePerMin(snap.WebhooksReceived, snap.UptimeSeconds),
		"errors_per_min":   ratePerMin(snap.Errors, snap.UptimeSeconds),

		// Process stats
		"goroutines":      runtime.NumGoroutine(),
		"heap_alloc_mb":   float64(mem.HeapAlloc) / 1024 / 1024,
		"heap_sys_mb":     float64(mem.HeapSys) / 1024 / 1024,
		"gc_num":          mem.NumGC,

		// Service info
		"go_version": runtime.Version(),
		"service":    "webhook-service",
	})
}

func ratePerMin(count int64, uptimeSec float64) float64 {
	if uptimeSec < 1 {
		return 0
	}
	return float64(count) / (uptimeSec / 60.0)
}
