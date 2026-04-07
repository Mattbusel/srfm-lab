// cmd/webhook/handlers/health_webhook.go -- Service health webhook handler.
//
// POST /webhook/health/{service}
//   A service reports its own health status.
//   If degraded, immediately triggers an alert via the alerter service.
//
// GET /webhook/health/summary
//   Returns the aggregated health map for all reporting services.

package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Health status types
// ---------------------------------------------------------------------------

// HealthStatus is the severity of a service's self-reported health.
type HealthStatus string

const (
	HealthOK       HealthStatus = "ok"
	HealthDegraded HealthStatus = "degraded"
	HealthDown     HealthStatus = "down"
)

// ServiceHealthReport is the JSON body expected from each service.
type ServiceHealthReport struct {
	Status    HealthStatus   `json:"status"`
	Message   string         `json:"message,omitempty"`
	Details   map[string]any `json:"details,omitempty"`
	Timestamp time.Time      `json:"timestamp,omitempty"`
	Version   string         `json:"version,omitempty"`
	Uptime    string         `json:"uptime,omitempty"`
}

// serviceHealthEntry is the internal record stored per service.
type serviceHealthEntry struct {
	Service    string
	Status     HealthStatus
	Message    string
	Details    map[string]any
	Version    string
	Uptime     string
	LastSeen   time.Time
	ReportedAt time.Time
}

// ---------------------------------------------------------------------------
// HealthWebhookHandler
// ---------------------------------------------------------------------------

// HealthWebhookHandler aggregates self-reported health from all services
// and exposes a summary endpoint.
type HealthWebhookHandler struct {
	mu          sync.RWMutex
	services    map[string]*serviceHealthEntry
	alerterAddr string
	client      *http.Client
	logger      *slog.Logger
}

// NewHealthWebhookHandler constructs a HealthWebhookHandler.
func NewHealthWebhookHandler(alerterAddr string, logger *slog.Logger) *HealthWebhookHandler {
	return &HealthWebhookHandler{
		services:    make(map[string]*serviceHealthEntry),
		alerterAddr: alerterAddr,
		client:      &http.Client{Timeout: 5 * time.Second},
		logger:      logger,
	}
}

// HandleServiceHealth handles POST /webhook/health/{service}.
func (h *HealthWebhookHandler) HandleServiceHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract service name from path: /webhook/health/{service}
	serviceName := strings.TrimPrefix(r.URL.Path, "/webhook/health/")
	serviceName = strings.Trim(serviceName, "/")
	if serviceName == "" || serviceName == "summary" {
		http.Error(w, "service name required in path", http.StatusBadRequest)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 64*1024))
	if err != nil {
		http.Error(w, "read error", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var report ServiceHealthReport
	if err := json.Unmarshal(body, &report); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Normalize status.
	report.Status = HealthStatus(strings.ToLower(string(report.Status)))
	if report.Timestamp.IsZero() {
		report.Timestamp = time.Now().UTC()
	}

	entry := &serviceHealthEntry{
		Service:    serviceName,
		Status:     report.Status,
		Message:    report.Message,
		Details:    report.Details,
		Version:    report.Version,
		Uptime:     report.Uptime,
		LastSeen:   time.Now(),
		ReportedAt: report.Timestamp,
	}

	h.mu.Lock()
	previous := h.services[serviceName]
	h.services[serviceName] = entry
	h.mu.Unlock()

	h.logger.Info("health report received",
		"service", serviceName,
		"status", report.Status,
		"message", report.Message,
	)

	// Trigger an alert if the service is degraded or down.
	if report.Status == HealthDegraded || report.Status == HealthDown {
		// Only alert if status changed or this is the first report with a bad status.
		shouldAlert := previous == nil ||
			previous.Status == HealthOK ||
			previous.Status != report.Status
		if shouldAlert {
			go h.triggerAlert(context.Background(), serviceName, entry)
		}
	}

	w.WriteHeader(http.StatusNoContent)
}

// HandleSummary handles GET /webhook/health/summary.
func (h *HealthWebhookHandler) HandleSummary(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	h.mu.RLock()
	defer h.mu.RUnlock()

	type summaryEntry struct {
		Status     string         `json:"status"`
		Message    string         `json:"message,omitempty"`
		Version    string         `json:"version,omitempty"`
		Uptime     string         `json:"uptime,omitempty"`
		LastSeen   string         `json:"last_seen"`
		StaleMinutes float64      `json:"stale_minutes,omitempty"`
		Details    map[string]any `json:"details,omitempty"`
	}

	summary := make(map[string]summaryEntry, len(h.services))
	degradedCount := 0
	downCount := 0
	staleCount := 0

	for name, e := range h.services {
		stale := time.Since(e.LastSeen)
		entry := summaryEntry{
			Status:   string(e.Status),
			Message:  e.Message,
			Version:  e.Version,
			Uptime:   e.Uptime,
			LastSeen: e.LastSeen.Format(time.RFC3339),
			Details:  e.Details,
		}
		// Flag entries that haven't reported in > 5 minutes as potentially stale.
		if stale > 5*time.Minute {
			entry.StaleMinutes = stale.Minutes()
			staleCount++
		}
		summary[name] = entry

		switch e.Status {
		case HealthDegraded:
			degradedCount++
		case HealthDown:
			downCount++
		}
	}

	// Derive overall status.
	overall := "ok"
	switch {
	case downCount > 0:
		overall = "down"
	case degradedCount > 0 || staleCount > 0:
		overall = "degraded"
	}

	writeJSONHandler(w, http.StatusOK, map[string]any{
		"overall":        overall,
		"service_count":  len(h.services),
		"degraded_count": degradedCount,
		"down_count":     downCount,
		"stale_count":    staleCount,
		"services":       summary,
		"as_of":          time.Now().UTC().Format(time.RFC3339),
	})
}

// triggerAlert sends an alert request to the alerter service for a degraded service.
func (h *HealthWebhookHandler) triggerAlert(ctx context.Context, serviceName string, e *serviceHealthEntry) {
	// POST to the alerter's webhook intake -- the alerter will route the alert
	// through its severity pipeline.
	payload := map[string]any{
		"source":   "webhook-health",
		"event":    "service_health_degraded",
		"service":  serviceName,
		"status":   string(e.Status),
		"message":  fmt.Sprintf("Service %q reported status %s: %s", serviceName, e.Status, e.Message),
		"severity": h.statusToSeverity(e.Status),
		"details":  e.Details,
		"reported_at": e.ReportedAt.Format(time.RFC3339),
	}

	body, err := json.Marshal(payload)
	if err != nil {
		h.logger.Error("health alert: marshal", "err", err)
		return
	}

	url := h.alerterAddr + "/alerts/ingest"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		h.logger.Error("health alert: build request", "err", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-SRFM-Source", "webhook-health")

	resp, err := h.client.Do(req)
	if err != nil {
		// Alerter may not be running in test environments.
		h.logger.Warn("health alert: alerter unreachable",
			"service", serviceName, "err", err)
		return
	}
	defer resp.Body.Close()

	h.logger.Info("health alert dispatched",
		"service", serviceName, "status", e.Status,
		"alerter_response", resp.StatusCode)
}

func (h *HealthWebhookHandler) statusToSeverity(s HealthStatus) string {
	switch s {
	case HealthDown:
		return "CRITICAL"
	case HealthDegraded:
		return "WARNING"
	default:
		return "INFO"
	}
}

// ServiceCount returns how many services have reported health.
func (h *HealthWebhookHandler) ServiceCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.services)
}

// GetServiceStatus returns the current status of a named service.
// Returns empty string if the service has never reported.
func (h *HealthWebhookHandler) GetServiceStatus(name string) HealthStatus {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if e, ok := h.services[name]; ok {
		return e.Status
	}
	return ""
}
