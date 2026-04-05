// Package handlers — metrics.go
//
// MetricsRegistry is a lightweight in-process counter store used by all
// handlers.  It is goroutine-safe via sync/atomic.
package handlers

import (
	"sync/atomic"
	"time"
)

// MetricsRegistry holds service-wide counters.
type MetricsRegistry struct {
	WebhooksReceived  atomic.Int64
	WebhooksProcessed atomic.Int64
	WebhookErrors     atomic.Int64
	startTime         time.Time
}

// NewMetricsRegistry creates a new registry, recording the service start time.
func NewMetricsRegistry() *MetricsRegistry {
	return &MetricsRegistry{startTime: time.Now()}
}

// UptimeSeconds returns the number of seconds since the service started.
func (m *MetricsRegistry) UptimeSeconds() float64 {
	return time.Since(m.startTime).Seconds()
}

// Snapshot returns a point-in-time copy of the counters.
func (m *MetricsRegistry) Snapshot() MetricsSnapshot {
	return MetricsSnapshot{
		WebhooksReceived:  m.WebhooksReceived.Load(),
		WebhooksProcessed: m.WebhooksProcessed.Load(),
		Errors:            m.WebhookErrors.Load(),
		UptimeSeconds:     m.UptimeSeconds(),
	}
}

// MetricsSnapshot is a serialisable point-in-time view of the metrics.
type MetricsSnapshot struct {
	WebhooksReceived  int64   `json:"webhooks_received"`
	WebhooksProcessed int64   `json:"webhooks_processed"`
	Errors            int64   `json:"errors"`
	UptimeSeconds     float64 `json:"uptime_seconds"`
}
