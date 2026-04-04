// Package metrics exposes Prometheus instrumentation for the gateway.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics groups all Prometheus instruments used by the gateway.
type Metrics struct {
	// BarsReceivedTotal counts bars received per symbol and source.
	BarsReceivedTotal *prometheus.CounterVec

	// BarsPerSecond is the current bar ingestion rate.
	BarsPerSecond prometheus.Gauge

	// WSSubscribers is the number of active WebSocket subscribers.
	WSSubscribers prometheus.Gauge

	// CacheHitRate is the current bar cache hit ratio.
	CacheHitRate prometheus.Gauge

	// FeedReconnectsTotal counts reconnection attempts per feed.
	FeedReconnectsTotal *prometheus.CounterVec

	// BarLatencyMs tracks the latency from bar timestamp to ingestion (ms).
	BarLatencyMs *prometheus.HistogramVec

	// AggregatedBarsTotal counts completed aggregated bars per timeframe.
	AggregatedBarsTotal *prometheus.CounterVec

	// EventChannelDepth tracks how full the ingestion channel is.
	EventChannelDepth prometheus.Gauge

	// ActiveSymbols is the number of symbols currently being tracked.
	ActiveSymbols prometheus.Gauge

	// ParquetWriteErrors counts parquet write errors.
	ParquetWriteErrors *prometheus.CounterVec

	// HTTPRequestDuration tracks REST API latency.
	HTTPRequestDuration *prometheus.HistogramVec
}

// New registers all Prometheus metrics and returns a Metrics instance.
func New(reg prometheus.Registerer) *Metrics {
	if reg == nil {
		reg = prometheus.DefaultRegisterer
	}
	factory := promauto.With(reg)

	return &Metrics{
		BarsReceivedTotal: factory.NewCounterVec(prometheus.CounterOpts{
			Namespace: "gateway",
			Name:      "bars_received_total",
			Help:      "Total number of bars received, labelled by symbol and source.",
		}, []string{"symbol", "source"}),

		BarsPerSecond: factory.NewGauge(prometheus.GaugeOpts{
			Namespace: "gateway",
			Name:      "bars_per_second",
			Help:      "Current bar ingestion rate (bars/s) measured over a rolling window.",
		}),

		WSSubscribers: factory.NewGauge(prometheus.GaugeOpts{
			Namespace: "gateway",
			Name:      "ws_subscribers",
			Help:      "Number of active WebSocket subscriber connections.",
		}),

		CacheHitRate: factory.NewGauge(prometheus.GaugeOpts{
			Namespace: "gateway",
			Name:      "cache_hit_rate",
			Help:      "Bar cache hit rate (0-1).",
		}),

		FeedReconnectsTotal: factory.NewCounterVec(prometheus.CounterOpts{
			Namespace: "gateway",
			Name:      "feed_reconnects_total",
			Help:      "Total reconnection attempts by feed source.",
		}, []string{"source"}),

		BarLatencyMs: factory.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "gateway",
			Name:      "bar_latency_ms",
			Help:      "Time from bar timestamp to ingestion (milliseconds).",
			Buckets:   []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000},
		}, []string{"source"}),

		AggregatedBarsTotal: factory.NewCounterVec(prometheus.CounterOpts{
			Namespace: "gateway",
			Name:      "aggregated_bars_total",
			Help:      "Total number of completed aggregated bars emitted.",
		}, []string{"timeframe"}),

		EventChannelDepth: factory.NewGauge(prometheus.GaugeOpts{
			Namespace: "gateway",
			Name:      "event_channel_depth",
			Help:      "Current depth of the internal event processing channel.",
		}),

		ActiveSymbols: factory.NewGauge(prometheus.GaugeOpts{
			Namespace: "gateway",
			Name:      "active_symbols",
			Help:      "Number of symbols currently receiving data.",
		}),

		ParquetWriteErrors: factory.NewCounterVec(prometheus.CounterOpts{
			Namespace: "gateway",
			Name:      "parquet_write_errors_total",
			Help:      "Total parquet write errors by symbol.",
		}, []string{"symbol"}),

		HTTPRequestDuration: factory.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "gateway",
			Name:      "http_request_duration_seconds",
			Help:      "HTTP request duration in seconds.",
			Buckets:   prometheus.DefBuckets,
		}, []string{"method", "path", "status"}),
	}
}
