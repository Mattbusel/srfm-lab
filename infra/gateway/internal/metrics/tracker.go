package metrics

import (
	"sync"
	"sync/atomic"
	"time"
)

// RateTracker computes a rolling events-per-second rate.
type RateTracker struct {
	mu       sync.Mutex
	buckets  []int64       // ring of count buckets
	times    []time.Time   // ring of bucket start times
	cap      int
	head     int
	window   time.Duration // total window over all buckets
	bucketW  time.Duration // per-bucket width
}

// NewRateTracker creates a RateTracker with nBuckets buckets over window.
func NewRateTracker(window time.Duration, nBuckets int) *RateTracker {
	if nBuckets <= 0 {
		nBuckets = 10
	}
	return &RateTracker{
		buckets: make([]int64, nBuckets),
		times:   make([]time.Time, nBuckets),
		cap:     nBuckets,
		window:  window,
		bucketW: window / time.Duration(nBuckets),
	}
}

// Inc increments the counter for the current time bucket.
func (rt *RateTracker) Inc() {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	now := time.Now()
	rt.advance(now)
	rt.buckets[rt.head]++
}

// Rate returns the events-per-second rate over the tracking window.
func (rt *RateTracker) Rate() float64 {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	now := time.Now()
	rt.advance(now)

	var total int64
	cutoff := now.Add(-rt.window)
	for i := 0; i < rt.cap; i++ {
		if rt.times[i].After(cutoff) {
			total += rt.buckets[i]
		}
	}
	return float64(total) / rt.window.Seconds()
}

// advance moves the head pointer to the bucket for now, zeroing stale buckets.
func (rt *RateTracker) advance(now time.Time) {
	if rt.times[rt.head].IsZero() {
		rt.times[rt.head] = now
		return
	}
	// How many new buckets do we need?
	elapsed := now.Sub(rt.times[rt.head])
	steps := int(elapsed / rt.bucketW)
	if steps <= 0 {
		return
	}
	if steps > rt.cap {
		steps = rt.cap
	}
	for i := 0; i < steps; i++ {
		rt.head = (rt.head + 1) % rt.cap
		rt.buckets[rt.head] = 0
		rt.times[rt.head] = rt.times[(rt.head-1+rt.cap)%rt.cap].Add(rt.bucketW)
	}
}

// LatencyTracker records p50/p95/p99 latencies using a fixed histogram.
type LatencyTracker struct {
	mu      sync.Mutex
	samples []float64
	maxSamp int
	pos     int
	full    bool
}

// NewLatencyTracker creates a LatencyTracker holding up to maxSamples observations.
func NewLatencyTracker(maxSamples int) *LatencyTracker {
	if maxSamples <= 0 {
		maxSamples = 10000
	}
	return &LatencyTracker{
		samples: make([]float64, maxSamples),
		maxSamp: maxSamples,
	}
}

// Observe records a latency sample in milliseconds.
func (lt *LatencyTracker) Observe(ms float64) {
	lt.mu.Lock()
	lt.samples[lt.pos] = ms
	lt.pos = (lt.pos + 1) % lt.maxSamp
	if lt.pos == 0 {
		lt.full = true
	}
	lt.mu.Unlock()
}

// Percentile returns the p-th percentile latency (p in [0,1]).
func (lt *LatencyTracker) Percentile(p float64) float64 {
	lt.mu.Lock()
	n := lt.pos
	if lt.full {
		n = lt.maxSamp
	}
	if n == 0 {
		lt.mu.Unlock()
		return 0
	}
	snap := make([]float64, n)
	copy(snap, lt.samples[:n])
	lt.mu.Unlock()

	sortFloat64s(snap)
	idx := int(p * float64(n-1))
	if idx >= len(snap) {
		idx = len(snap) - 1
	}
	return snap[idx]
}

// sortFloat64s sorts a float64 slice in place (insertion sort for small slices).
func sortFloat64s(a []float64) {
	n := len(a)
	// Use a simple quicksort-like partition for larger slices.
	if n <= 1 {
		return
	}
	quickSortF64(a, 0, n-1)
}

func quickSortF64(a []float64, lo, hi int) {
	if lo >= hi {
		return
	}
	pivot := a[(lo+hi)/2]
	i, j := lo, hi
	for i <= j {
		for a[i] < pivot {
			i++
		}
		for a[j] > pivot {
			j--
		}
		if i <= j {
			a[i], a[j] = a[j], a[i]
			i++
			j--
		}
	}
	quickSortF64(a, lo, j)
	quickSortF64(a, i, hi)
}

// Counter is a thread-safe int64 counter.
type Counter struct {
	v int64
}

func (c *Counter) Inc()           { atomic.AddInt64(&c.v, 1) }
func (c *Counter) Add(n int64)    { atomic.AddInt64(&c.v, n) }
func (c *Counter) Load() int64    { return atomic.LoadInt64(&c.v) }
func (c *Counter) Reset()         { atomic.StoreInt64(&c.v, 0) }

// GaugeSnapshot is a point-in-time snapshot of gateway runtime metrics.
type GaugeSnapshot struct {
	Timestamp        time.Time
	BarsPerSecond    float64
	WSSubscribers    int
	CacheHitRate     float64
	ActiveSymbols    int
	TotalBars        int
	EventChanDepth   int
	P50LatencyMs     float64
	P95LatencyMs     float64
	P99LatencyMs     float64
	GoroutineCount   int
	HeapAllocMB      float64
}

// MetricCollector aggregates all real-time metrics.
type MetricCollector struct {
	BarRate      *RateTracker
	Latency      *LatencyTracker
	BarCount     Counter
	TradeCount   Counter
	QuoteCount   Counter
	DropCount    Counter
	ErrorCount   Counter
}

// NewMetricCollector creates a MetricCollector.
func NewMetricCollector() *MetricCollector {
	return &MetricCollector{
		BarRate: NewRateTracker(60*time.Second, 60),
		Latency: NewLatencyTracker(100000),
	}
}

// RecordBar records a bar ingestion event.
func (mc *MetricCollector) RecordBar(latencyMs float64) {
	mc.BarCount.Inc()
	mc.BarRate.Inc()
	mc.Latency.Observe(latencyMs)
}
