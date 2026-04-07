/// pipeline_metrics.rs -- performance metrics for the data ingestion pipeline.
///
/// Tracks bars processed, errors, throughput, latency (HDR-style histogram),
/// and rolling quality scores. All public methods are thread-safe via atomic
/// operations and a Mutex-guarded latency histogram.

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Latency histogram ─────────────────────────────────────────────────────────

/// HDR-style power-of-2 latency histogram (buckets in nanoseconds).
/// Bucket i covers [2^i, 2^(i+1)) ns.  We use 64 buckets (covers up to ~584s).
const NUM_BUCKETS: usize = 64;

#[derive(Debug, Clone)]
struct LatencyHistogram {
    buckets:     [u64; NUM_BUCKETS],
    total_count: u64,
    total_ns:    u64,
}

impl LatencyHistogram {
    fn new() -> Self {
        LatencyHistogram {
            buckets:     [0; NUM_BUCKETS],
            total_count: 0,
            total_ns:    0,
        }
    }

    fn record(&mut self, ns: u64) {
        let bucket = bucket_index(ns);
        self.buckets[bucket] += 1;
        self.total_count += 1;
        self.total_ns += ns;
    }

    /// Compute percentile (0..=100) from histogram. Returns nanoseconds.
    fn percentile(&self, pct: f64) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        let target = ((pct / 100.0) * self.total_count as f64).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                // Midpoint of bucket [2^i, 2^(i+1)).
                let lo = 1u64 << i;
                let hi = if i + 1 < 64 { 1u64 << (i + 1) } else { u64::MAX };
                return ((lo + hi) / 2) as f64;
            }
        }
        // Should not reach here.
        (1u64 << (NUM_BUCKETS - 1)) as f64
    }

    fn mean_ns(&self) -> f64 {
        if self.total_count == 0 { return 0.0; }
        self.total_ns as f64 / self.total_count as f64
    }

    fn reset(&mut self) {
        self.buckets = [0; NUM_BUCKETS];
        self.total_count = 0;
        self.total_ns = 0;
    }
}

fn bucket_index(ns: u64) -> usize {
    if ns == 0 { return 0; }
    let bit = 63 - ns.leading_zeros() as usize; // floor(log2(ns))
    bit.min(NUM_BUCKETS - 1)
}

// ── Rolling window for throughput ─────────────────────────────────────────────

/// Fixed-size ring buffer of (timestamp_s, count) pairs used to compute
/// throughput over a sliding window.
const RING_SIZE: usize = 128;

#[derive(Clone)]
struct ThroughputRing {
    timestamps: [i64; RING_SIZE],
    counts:     [u64; RING_SIZE],
    head:       usize,
    len:        usize,
}

impl ThroughputRing {
    fn new() -> Self {
        ThroughputRing {
            timestamps: [0; RING_SIZE],
            counts:     [0; RING_SIZE],
            head:       0,
            len:        0,
        }
    }

    fn push(&mut self, ts_s: i64, count: u64) {
        let idx = (self.head + self.len) % RING_SIZE;
        self.timestamps[idx] = ts_s;
        self.counts[idx] = count;
        if self.len < RING_SIZE {
            self.len += 1;
        } else {
            // Overwrite oldest entry.
            self.head = (self.head + 1) % RING_SIZE;
        }
    }

    /// Sum counts within the last `window_s` seconds.
    fn sum_within(&self, window_s: i64, now_s: i64) -> u64 {
        let cutoff = now_s - window_s;
        let mut total = 0u64;
        for i in 0..self.len {
            let idx = (self.head + i) % RING_SIZE;
            if self.timestamps[idx] >= cutoff {
                total += self.counts[idx];
            }
        }
        total
    }
}

// ── PipelineMetrics ───────────────────────────────────────────────────────────

/// Snapshot of pipeline performance at a point in time.
#[derive(Debug, Clone)]
pub struct PipelineSnapshot {
    /// Total bars processed since last reset.
    pub bars_processed:   u64,
    /// Total error events since last reset.
    pub errors:           u64,
    /// Current throughput estimate (bars per second).
    pub throughput_per_s: f64,
    /// P99 processing latency in milliseconds.
    pub latency_p99_ms:   f64,
    /// Average quality score across processed bars (0..1).
    pub quality_score_avg: f64,
    /// Unix timestamp (seconds) when snapshot was taken.
    pub timestamp:         i64,
}

/// Thread-safe pipeline performance tracker.
///
/// Use `Arc<PipelineMetrics>` to share across threads.
pub struct PipelineMetrics {
    bars_processed:  AtomicU64,
    errors:          AtomicU64,
    quality_sum:     Mutex<f64>,
    quality_count:   AtomicU64,
    // Start timestamp for throughput baseline.
    start_ts:        AtomicI64,
    // Latency histogram (protected by mutex).
    histogram:       Mutex<LatencyHistogram>,
    // Throughput ring buffer.
    ring:            Mutex<ThroughputRing>,
}

impl PipelineMetrics {
    pub fn new() -> Self {
        let now = now_s();
        PipelineMetrics {
            bars_processed: AtomicU64::new(0),
            errors:         AtomicU64::new(0),
            quality_sum:    Mutex::new(0.0),
            quality_count:  AtomicU64::new(0),
            start_ts:       AtomicI64::new(now),
            histogram:      Mutex::new(LatencyHistogram::new()),
            ring:            Mutex::new(ThroughputRing::new()),
        }
    }

    /// Create a reference-counted metrics instance suitable for sharing.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Record one bar's processing result.
    ///
    /// `processing_time_ns` -- wall-clock time to process the bar.
    /// `quality_score`       -- data quality score for this bar (0..1).
    /// `error`               -- true if the bar triggered an error.
    pub fn record_bar(&self, processing_time_ns: u64, quality_score: f64, error: bool) {
        self.bars_processed.fetch_add(1, Ordering::Relaxed);
        if error {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }

        // Quality accumulator.
        {
            let mut q = self.quality_sum.lock().unwrap();
            *q += quality_score;
            self.quality_count.fetch_add(1, Ordering::Relaxed);
        }

        // Latency histogram.
        {
            let mut hist = self.histogram.lock().unwrap();
            hist.record(processing_time_ns);
        }

        // Throughput ring -- record one bar at current second.
        {
            let ts = now_s();
            let mut ring = self.ring.lock().unwrap();
            ring.push(ts, 1);
        }
    }

    /// Take an instantaneous snapshot of all metrics.
    pub fn snapshot(&self) -> PipelineSnapshot {
        let ts = now_s();
        let bars = self.bars_processed.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);

        let elapsed = (ts - self.start_ts.load(Ordering::Relaxed)).max(1) as f64;
        let throughput = bars as f64 / elapsed;

        let latency_p99_ms = {
            let hist = self.histogram.lock().unwrap();
            hist.percentile(99.0) / 1_000_000.0 // ns -> ms
        };

        let quality_score_avg = {
            let q = self.quality_sum.lock().unwrap();
            let count = self.quality_count.load(Ordering::Relaxed);
            if count == 0 { 0.0 } else { *q / count as f64 }
        };

        PipelineSnapshot {
            bars_processed: bars,
            errors,
            throughput_per_s: throughput,
            latency_p99_ms,
            quality_score_avg,
            timestamp: ts,
        }
    }

    /// Compute bars per second over the last `window_s` seconds.
    pub fn throughput_over(&self, window_s: u64) -> f64 {
        let ts = now_s();
        let ring = self.ring.lock().unwrap();
        let count = ring.sum_within(window_s as i64, ts);
        count as f64 / window_s.max(1) as f64
    }

    /// Reset all counters and histograms.
    pub fn reset_counters(&self) {
        self.bars_processed.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
        self.quality_count.store(0, Ordering::Relaxed);
        *self.quality_sum.lock().unwrap() = 0.0;
        self.start_ts.store(now_s(), Ordering::Relaxed);
        self.histogram.lock().unwrap().reset();
        *self.ring.lock().unwrap() = ThroughputRing::new();
    }

    /// Current bars processed count.
    pub fn bars_processed(&self) -> u64 {
        self.bars_processed.load(Ordering::Relaxed)
    }

    /// Current error count.
    pub fn error_count(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Mean latency in milliseconds.
    pub fn mean_latency_ms(&self) -> f64 {
        self.histogram.lock().unwrap().mean_ns() / 1_000_000.0
    }

    /// P50 latency in milliseconds.
    pub fn latency_p50_ms(&self) -> f64 {
        self.histogram.lock().unwrap().percentile(50.0) / 1_000_000.0
    }

    /// P99 latency in milliseconds.
    pub fn latency_p99_ms(&self) -> f64 {
        self.histogram.lock().unwrap().percentile(99.0) / 1_000_000.0
    }

    /// Error rate as a fraction (errors / bars_processed).
    pub fn error_rate(&self) -> f64 {
        let bars = self.bars_processed.load(Ordering::Relaxed);
        let errs = self.errors.load(Ordering::Relaxed);
        if bars == 0 { return 0.0; }
        errs as f64 / bars as f64
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self { Self::new() }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_s() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_snapshot_zeros() {
        let m = PipelineMetrics::new();
        let snap = m.snapshot();
        assert_eq!(snap.bars_processed, 0);
        assert_eq!(snap.errors, 0);
        assert_eq!(snap.quality_score_avg, 0.0);
    }

    #[test]
    fn test_record_bar_increments_count() {
        let m = PipelineMetrics::new();
        m.record_bar(1_000_000, 0.95, false);
        m.record_bar(2_000_000, 0.80, false);
        assert_eq!(m.bars_processed(), 2);
    }

    #[test]
    fn test_record_error_tracked() {
        let m = PipelineMetrics::new();
        m.record_bar(500_000, 0.0, true);
        m.record_bar(500_000, 1.0, false);
        assert_eq!(m.error_count(), 1);
    }

    #[test]
    fn test_error_rate_calculation() {
        let m = PipelineMetrics::new();
        for _ in 0..8 {
            m.record_bar(100_000, 1.0, false);
        }
        for _ in 0..2 {
            m.record_bar(100_000, 0.0, true);
        }
        let rate = m.error_rate();
        assert!((rate - 0.2).abs() < 1e-9, "expected 20% error rate, got {}", rate);
    }

    #[test]
    fn test_quality_score_average() {
        let m = PipelineMetrics::new();
        m.record_bar(100_000, 1.0, false);
        m.record_bar(100_000, 0.0, false);
        let snap = m.snapshot();
        assert!((snap.quality_score_avg - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_latency_p99_recorded() {
        let m = PipelineMetrics::new();
        // Record 100 fast bars and 1 slow bar.
        for _ in 0..100 {
            m.record_bar(1_000_000, 1.0, false); // 1 ms
        }
        m.record_bar(500_000_000, 1.0, false); // 500 ms
        let p99_ms = m.latency_p99_ms();
        assert!(p99_ms > 0.0, "p99 should be positive");
    }

    #[test]
    fn test_reset_counters_clears_state() {
        let m = PipelineMetrics::new();
        m.record_bar(100_000, 0.9, false);
        m.record_bar(100_000, 0.8, true);
        m.reset_counters();
        assert_eq!(m.bars_processed(), 0);
        assert_eq!(m.error_count(), 0);
        let snap = m.snapshot();
        assert_eq!(snap.quality_score_avg, 0.0);
    }

    #[test]
    fn test_throughput_over_window() {
        let m = PipelineMetrics::new();
        for _ in 0..100 {
            m.record_bar(10_000, 1.0, false);
        }
        // Window of 10s -- should return reasonable value.
        let tput = m.throughput_over(10);
        assert!(tput >= 0.0);
    }

    #[test]
    fn test_bucket_index_boundaries() {
        assert_eq!(bucket_index(0), 0);
        assert_eq!(bucket_index(1), 0);
        assert_eq!(bucket_index(2), 1);
        assert_eq!(bucket_index(4), 2);
        assert_eq!(bucket_index(1_000_000_000), 29); // ~1 second in ns
    }

    #[test]
    fn test_histogram_percentile_single_value() {
        let mut h = LatencyHistogram::new();
        h.record(1_000_000); // 1ms in ns
        // P50 and P99 should both land in the same bucket.
        let p50 = h.percentile(50.0);
        let p99 = h.percentile(99.0);
        assert!(p50 > 0.0);
        assert!(p99 >= p50);
    }

    #[test]
    fn test_shared_arc_usage() {
        let m = PipelineMetrics::shared();
        let m2 = Arc::clone(&m);
        m.record_bar(50_000, 1.0, false);
        m2.record_bar(50_000, 1.0, false);
        assert_eq!(m.bars_processed(), 2);
    }

    #[test]
    fn test_p50_below_p99() {
        let m = PipelineMetrics::new();
        // Mixed latencies.
        for i in 1..=100u64 {
            m.record_bar(i * 100_000, 1.0, false);
        }
        let p50 = m.latency_p50_ms();
        let p99 = m.latency_p99_ms();
        assert!(p99 >= p50, "p99 ({}) should be >= p50 ({})", p99, p50);
    }
}
