package monitoring

import (
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// latencyHistogram tracks a rough histogram of latency values in buckets.
type latencyHistogram struct {
	mu      sync.Mutex
	buckets []int64  // counts per bucket
	bounds  []float64 // upper bounds in ms
	sum     float64
	count   int64
}

func newLatencyHistogram() *latencyHistogram {
	return &latencyHistogram{
		bounds:  []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000},
		buckets: make([]int64, 11), // 10 bounds + overflow
	}
}

func (h *latencyHistogram) observe(ms float64) {
	h.mu.Lock()
	h.sum += ms
	h.count++
	for i, b := range h.bounds {
		if ms <= b {
			h.buckets[i]++
			h.mu.Unlock()
			return
		}
	}
	h.buckets[len(h.buckets)-1]++
	h.mu.Unlock()
}

func (h *latencyHistogram) snapshot() (sum float64, count int64, buckets []int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	b := make([]int64, len(h.buckets))
	copy(b, h.buckets)
	return h.sum, h.count, b
}

// Metrics tracks all in-memory counters and histograms.
type Metrics struct {
	mu sync.RWMutex

	// Counters
	barsReceivedTotal atomic.Int64
	barsStoredTotal   atomic.Int64
	cacheHits         atomic.Int64
	cacheMisses       atomic.Int64
	feedErrors        sync.Map // feed name -> *atomic.Int64
	feedBarsRcvd      sync.Map // feed name -> *atomic.Int64
	failovers         atomic.Int64
	httpRequests      atomic.Int64
	httpErrors        atomic.Int64

	// Gauges
	wsClientsActive atomic.Int64

	// Histograms
	feedLatency   *latencyHistogram
	httpLatency   *latencyHistogram

	// Per-timeframe stored counts
	tfStored sync.Map // timeframe -> *atomic.Int64

	// HTTP path counters
	pathCounts sync.Map // path -> *atomic.Int64
}

// NewMetrics creates a Metrics instance.
func NewMetrics() *Metrics {
	return &Metrics{
		feedLatency: newLatencyHistogram(),
		httpLatency: newLatencyHistogram(),
	}
}

func (m *Metrics) BarReceived(feed string) {
	m.barsReceivedTotal.Add(1)
	v, _ := m.feedBarsRcvd.LoadOrStore(feed, &atomic.Int64{})
	v.(*atomic.Int64).Add(1)
}

func (m *Metrics) BarStored(timeframe string) {
	m.barsStoredTotal.Add(1)
	v, _ := m.tfStored.LoadOrStore(timeframe, &atomic.Int64{})
	v.(*atomic.Int64).Add(1)
}

func (m *Metrics) FeedError(feed string) {
	v, _ := m.feedErrors.LoadOrStore(feed, &atomic.Int64{})
	v.(*atomic.Int64).Add(1)
}

func (m *Metrics) FeedLatency(feed string, latencyMs float64) {
	_ = feed
	m.feedLatency.observe(latencyMs)
}

func (m *Metrics) WSClientsActive(n int) {
	m.wsClientsActive.Store(int64(n))
}

func (m *Metrics) CacheHit() {
	m.cacheHits.Add(1)
}

func (m *Metrics) CacheMiss() {
	m.cacheMisses.Add(1)
}

func (m *Metrics) RecordFailover(from, to string) {
	m.failovers.Add(1)
}

func (m *Metrics) HTTPRequest(path string, status int, latency time.Duration) {
	m.httpRequests.Add(1)
	if status >= 400 {
		m.httpErrors.Add(1)
	}
	m.httpLatency.observe(float64(latency.Milliseconds()))
	v, _ := m.pathCounts.LoadOrStore(path, &atomic.Int64{})
	v.(*atomic.Int64).Add(1)
}

// Snapshot returns a JSON-serializable snapshot of all metrics.
func (m *Metrics) Snapshot() map[string]interface{} {
	hits := m.cacheHits.Load()
	misses := m.cacheMisses.Load()
	total := hits + misses
	cacheHitRate := 0.0
	if total > 0 {
		cacheHitRate = float64(hits) / float64(total)
	}

	feedBars := make(map[string]int64)
	m.feedBarsRcvd.Range(func(k, v interface{}) bool {
		feedBars[k.(string)] = v.(*atomic.Int64).Load()
		return true
	})

	feedErrs := make(map[string]int64)
	m.feedErrors.Range(func(k, v interface{}) bool {
		feedErrs[k.(string)] = v.(*atomic.Int64).Load()
		return true
	})

	tfCounts := make(map[string]int64)
	m.tfStored.Range(func(k, v interface{}) bool {
		tfCounts[k.(string)] = v.(*atomic.Int64).Load()
		return true
	})

	_, latCount, _ := m.feedLatency.snapshot()

	return map[string]interface{}{
		"bars_received_total":  m.barsReceivedTotal.Load(),
		"bars_stored_total":    m.barsStoredTotal.Load(),
		"cache_hits":           hits,
		"cache_misses":         misses,
		"cache_hit_rate":       cacheHitRate,
		"ws_clients_active":    m.wsClientsActive.Load(),
		"failovers_total":      m.failovers.Load(),
		"http_requests_total":  m.httpRequests.Load(),
		"http_errors_total":    m.httpErrors.Load(),
		"feed_bars_received":   feedBars,
		"feed_errors":          feedErrs,
		"timeframe_stored":     tfCounts,
		"feed_latency_samples": latCount,
	}
}

// Handler returns an http.HandlerFunc that exposes Prometheus-compatible metrics.
func (m *Metrics) Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")

		var sb strings.Builder

		write := func(name, help, typ string, value interface{}) {
			sb.WriteString(fmt.Sprintf("# HELP %s %s\n", name, help))
			sb.WriteString(fmt.Sprintf("# TYPE %s %s\n", name, typ))
			sb.WriteString(fmt.Sprintf("%s %v\n", name, value))
		}

		write("srfm_bars_received_total", "Total bars received from all feeds", "counter", m.barsReceivedTotal.Load())
		write("srfm_bars_stored_total", "Total bars stored to SQLite", "counter", m.barsStoredTotal.Load())
		write("srfm_cache_hits_total", "Cache hits", "counter", m.cacheHits.Load())
		write("srfm_cache_misses_total", "Cache misses", "counter", m.cacheMisses.Load())
		write("srfm_ws_clients_active", "Active WebSocket clients", "gauge", m.wsClientsActive.Load())
		write("srfm_failovers_total", "Feed failover count", "counter", m.failovers.Load())
		write("srfm_http_requests_total", "HTTP requests", "counter", m.httpRequests.Load())
		write("srfm_http_errors_total", "HTTP error responses", "counter", m.httpErrors.Load())

		// Per-feed bars
		m.feedBarsRcvd.Range(func(k, v interface{}) bool {
			sb.WriteString(fmt.Sprintf("srfm_feed_bars_received{feed=%q} %d\n", k, v.(*atomic.Int64).Load()))
			return true
		})

		// Feed latency histogram
		sum, count, buckets := m.feedLatency.snapshot()
		bounds := m.feedLatency.bounds
		cumulative := int64(0)
		for i, bound := range bounds {
			cumulative += buckets[i]
			sb.WriteString(fmt.Sprintf("srfm_feed_latency_ms_bucket{le=\"%.0f\"} %d\n", bound, cumulative))
		}
		cumulative += buckets[len(buckets)-1]
		sb.WriteString(fmt.Sprintf("srfm_feed_latency_ms_bucket{le=\"+Inf\"} %d\n", cumulative))
		sb.WriteString(fmt.Sprintf("srfm_feed_latency_ms_sum %v\n", sum))
		sb.WriteString(fmt.Sprintf("srfm_feed_latency_ms_count %d\n", count))

		w.Write([]byte(sb.String()))
	}
}
