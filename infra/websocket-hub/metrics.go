// metrics.go — Prometheus metrics for the WebSocket hub.
package wshub

import (
	"encoding/json"
	"io"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ─────────────────────────────────────────────────────────────────────────────
// Prometheus instruments
// ─────────────────────────────────────────────────────────────────────────────

var (
	hubConnectedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "connections_total",
		Help:      "Total WebSocket connections accepted.",
	})

	hubUpgradesTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "upgrades_total",
		Help:      "Total successful HTTP→WebSocket upgrades.",
	})

	hubRejectedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "rejected_total",
		Help:      "Total connection rejections by reason.",
	}, []string{"reason"})

	hubActiveConnections = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "active_connections",
		Help:      "Number of currently connected WebSocket clients.",
	})

	hubActiveRooms = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "active_rooms",
		Help:      "Number of active rooms.",
	})

	hubRoomSubscriptions = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "room_subscriptions",
		Help:      "Number of subscriptions per room.",
	}, []string{"room"})

	hubBroadcastTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "broadcasts_total",
		Help:      "Total broadcast messages enqueued.",
	})

	hubMessagesSentTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "messages_sent_total",
		Help:      "Total messages sent to clients.",
	})

	hubDroppedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "messages_dropped_total",
		Help:      "Total messages dropped by reason.",
	}, []string{"reason"})

	hubRateLimitedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "rate_limited_total",
		Help:      "Total messages dropped due to rate limiting by scope.",
	}, []string{"scope"})

	hubBroadcastQueueDepth = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "broadcast_queue_depth",
		Help:      "Current depth of the broadcast work queue.",
	})

	hubMessageSizeBytes = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "message_size_bytes",
		Help:      "Size distribution of WebSocket messages.",
		Buckets:   prometheus.ExponentialBuckets(64, 2, 12), // 64B to 256KB
	}, []string{"direction"})

	hubClientRooms = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "client_room_count",
		Help:      "Distribution of room counts per client.",
		Buckets:   []float64{1, 2, 5, 10, 20, 50},
	})

	hubConnectionDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "srfm",
		Subsystem: "wshub",
		Name:      "connection_duration_seconds",
		Help:      "Duration of WebSocket connections.",
		Buckets:   []float64{1, 10, 30, 60, 300, 900, 1800, 3600, 86400},
	})
)

// ─────────────────────────────────────────────────────────────────────────────
// json.Encoder helper (avoids handler.go referencing encoding/json directly)
// ─────────────────────────────────────────────────────────────────────────────

// newJSONEncoder returns a new json.Encoder writing to w.
func newJSONEncoder(w io.Writer) *json.Encoder {
	return json.NewEncoder(w)
}
