// metrics.go — Prometheus metrics for the event bus.
package eventbus

import (
	"context"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// Prometheus gauges, counters, histograms
// ─────────────────────────────────────────────────────────────────────────────

var (
	eventsPublishedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "events_published_total",
		Help:      "Total events published to the bus.",
	}, []string{"topic", "event_type"})

	eventsReceivedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "events_received_total",
		Help:      "Total events received by subscribers.",
	}, []string{"topic", "event_type"})

	eventsDroppedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "events_dropped_total",
		Help:      "Total events dropped (slow consumers, channel full).",
	}, []string{"topic", "reason"})

	eventsDLQTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "dlq_messages_total",
		Help:      "Total messages sent to dead-letter queue.",
	}, []string{"topic"})

	publishLatencyHistogram = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "publish_latency_us",
		Help:      "End-to-end publish latency in microseconds.",
		Buckets:   []float64{50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000},
	}, []string{"topic"})

	subscribeDeliveryLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "delivery_latency_us",
		Help:      "Time from event publish to subscriber handler invocation.",
		Buckets:   []float64{100, 250, 500, 1000, 2500, 5000, 10000, 50000, 100000},
	}, []string{"topic"})

	subscriberCount = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "subscribers",
		Help:      "Current number of active subscribers per topic.",
	}, []string{"topic"})

	streamLagGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "stream_lag",
		Help:      "Consumer group lag (unread messages) per stream.",
	}, []string{"stream_key", "group_name"})

	streamLengthGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "stream_length",
		Help:      "Number of entries in each Redis Stream.",
	}, []string{"stream_key"})

	pendingMessagesGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "pending_messages",
		Help:      "Number of pending (unacknowledged) messages per consumer group.",
	}, []string{"stream_key", "group_name"})

	dlqLengthGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "dlq_length",
		Help:      "Number of messages waiting in the dead-letter queue per topic.",
	}, []string{"topic"})

	backpressureDetectedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "backpressure_detected_total",
		Help:      "Number of times backpressure was detected (channel full).",
	}, []string{"topic"})

	throughputGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "eventbus",
		Name:      "throughput_events_per_sec",
		Help:      "Smoothed events-per-second throughput.",
	}, []string{"topic"})
)

// ─────────────────────────────────────────────────────────────────────────────
// MetricsCollector — background goroutine that scrapes Redis metrics
// ─────────────────────────────────────────────────────────────────────────────

// MetricsCollectorConfig configures the background metrics collector.
type MetricsCollectorConfig struct {
	// CollectInterval is how often to scrape Redis metrics.
	CollectInterval time.Duration
	// MonitoredTopics is the list of topics to monitor.
	MonitoredTopics []Topic
	// ConsumerGroups maps stream key pattern → consumer group name.
	ConsumerGroups map[string]string
}

// MetricsCollector polls Redis for stream metrics and exposes them via Prometheus.
type MetricsCollector struct {
	rdb *redis.Client
	cfg MetricsCollectorConfig
	log *zap.Logger

	// Throughput tracking.
	prevCounts map[string]int64
}

// NewMetricsCollector creates a MetricsCollector.
func NewMetricsCollector(rdb *redis.Client, cfg MetricsCollectorConfig, log *zap.Logger) *MetricsCollector {
	if cfg.CollectInterval == 0 {
		cfg.CollectInterval = 15 * time.Second
	}
	return &MetricsCollector{
		rdb:        rdb,
		cfg:        cfg,
		log:        log,
		prevCounts: make(map[string]int64),
	}
}

// Run starts the metrics collection loop. Blocks until ctx is cancelled.
func (mc *MetricsCollector) Run(ctx context.Context) {
	ticker := time.NewTicker(mc.cfg.CollectInterval)
	defer ticker.Stop()

	mc.collect(ctx) // immediate first collection

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mc.collect(ctx)
		}
	}
}

func (mc *MetricsCollector) collect(ctx context.Context) {
	for _, topic := range mc.cfg.MonitoredTopics {
		meta := GetTopicMeta(topic)
		partitions := 1
		if meta != nil && meta.Partitions > 0 {
			partitions = meta.Partitions
		}

		for p := 0; p < partitions; p++ {
			streamKey := StreamKey(topic, p)
			mc.collectStreamMetrics(ctx, streamKey, string(topic))
		}

		// DLQ length.
		dlqKey := DLQKey(topic)
		if length, err := mc.rdb.XLen(ctx, dlqKey).Result(); err == nil {
			dlqLengthGauge.WithLabelValues(string(topic)).Set(float64(length))
		}
	}

	// Consumer group metrics.
	for streamKey, groupName := range mc.cfg.ConsumerGroups {
		if lag, err := StreamLag(ctx, mc.rdb, streamKey, groupName); err == nil {
			streamLagGauge.WithLabelValues(streamKey, groupName).Set(float64(lag))
		}
		if pending, err := PendingCount(ctx, mc.rdb, streamKey, groupName); err == nil {
			pendingMessagesGauge.WithLabelValues(streamKey, groupName).Set(float64(pending))
		}
	}
}

func (mc *MetricsCollector) collectStreamMetrics(ctx context.Context, streamKey, topicStr string) {
	length, err := mc.rdb.XLen(ctx, streamKey).Result()
	if err != nil {
		if err != redis.Nil {
			mc.log.Warn("XLEN failed", zap.String("stream", streamKey), zap.Error(err))
		}
		return
	}
	streamLengthGauge.WithLabelValues(streamKey).Set(float64(length))

	// Throughput: events per second since last collection.
	prev := mc.prevCounts[streamKey]
	delta := length - prev
	mc.prevCounts[streamKey] = length
	if mc.cfg.CollectInterval > 0 && delta >= 0 {
		eps := float64(delta) / mc.cfg.CollectInterval.Seconds()
		throughputGauge.WithLabelValues(topicStr).Set(eps)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Instrumented publish/receive helpers
// ─────────────────────────────────────────────────────────────────────────────

// RecordPublish updates Prometheus counters after a successful publish.
func RecordPublish(topic Topic, evtType string, latencyUS int64) {
	eventsPublishedTotal.WithLabelValues(string(topic), evtType).Inc()
	publishLatencyHistogram.WithLabelValues(string(topic)).Observe(float64(latencyUS))
}

// RecordReceive updates counters when a subscriber receives an event.
func RecordReceive(topic Topic, evtType string) {
	eventsReceivedTotal.WithLabelValues(string(topic), evtType).Inc()
}

// RecordDrop records a dropped event.
func RecordDrop(topic Topic, reason string) {
	eventsDroppedTotal.WithLabelValues(string(topic), reason).Inc()
	backpressureDetectedTotal.WithLabelValues(string(topic)).Inc()
}

// RecordDLQ records a message sent to the dead-letter queue.
func RecordDLQ(topic Topic) {
	eventsDLQTotal.WithLabelValues(string(topic)).Inc()
}

// RecordDeliveryLatency records time from event creation to subscriber delivery.
func RecordDeliveryLatency(topic Topic, createdAt time.Time) {
	latencyUS := time.Since(createdAt).Microseconds()
	subscribeDeliveryLatency.WithLabelValues(string(topic)).Observe(float64(latencyUS))
}

// UpdateSubscriberCount updates the gauge for subscriber counts.
func UpdateSubscriberCount(topic Topic, count int) {
	subscriberCount.WithLabelValues(string(topic)).Set(float64(count))
}

// ─────────────────────────────────────────────────────────────────────────────
// Backpressure detector
// ─────────────────────────────────────────────────────────────────────────────

// BackpressureDetector monitors the bus and logs/alerts when consumer lag is high.
type BackpressureDetector struct {
	bus    *EventBus
	log    *zap.Logger
	rdb    *redis.Client
	topics []Topic

	// Threshold: lag ratio above which we consider backpressure active.
	lagThreshold float64
}

// NewBackpressureDetector creates a BackpressureDetector.
func NewBackpressureDetector(bus *EventBus, rdb *redis.Client, topics []Topic, lagThreshold float64, log *zap.Logger) *BackpressureDetector {
	return &BackpressureDetector{
		bus:          bus,
		log:          log,
		rdb:          rdb,
		topics:       topics,
		lagThreshold: lagThreshold,
	}
}

// Run starts the backpressure monitoring loop.
func (bd *BackpressureDetector) Run(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			bd.check(ctx)
		}
	}
}

func (bd *BackpressureDetector) check(ctx context.Context) {
	stats := bd.bus.Stats()

	// Check drop ratio.
	if stats.Published > 0 {
		dropRatio := float64(stats.Dropped) / float64(stats.Published)
		if dropRatio > bd.lagThreshold {
			bd.log.Warn("backpressure detected: high drop ratio",
				zap.Float64("drop_ratio", dropRatio),
				zap.Int64("published", stats.Published),
				zap.Int64("dropped", stats.Dropped))
		}
	}

	// Check DLQ growth.
	if stats.DLQMessages > 0 {
		bd.log.Warn("DLQ has messages — check dead letter queues",
			zap.Int64("dlq_messages", stats.DLQMessages))
	}

	// Check per-topic stream lag in Redis.
	for _, topic := range bd.topics {
		streamKey := StreamKey(topic, 0)
		length, err := bd.rdb.XLen(ctx, streamKey).Result()
		if err != nil {
			continue
		}
		if length > 100_000 {
			bd.log.Warn("stream length high",
				zap.String("stream", streamKey),
				zap.Int64("length", length))
		}
	}
}
