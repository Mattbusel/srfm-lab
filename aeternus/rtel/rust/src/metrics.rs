// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// metrics.rs — Prometheus metrics and performance monitoring for Rust RTEL
// =============================================================================
//! In-process metrics collection for the Rust RTEL components.
//! Exports Prometheus text format metrics.
//! Tracks: publish rates, latency histograms, ring buffer utilization,
//! sequence lag, error counts.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::collections::HashMap;

use crate::now_ns;

// ---------------------------------------------------------------------------
// AtomicHistogram — lock-free log2 histogram (64 buckets)
// ---------------------------------------------------------------------------
pub struct AtomicHistogram {
    buckets: [AtomicU64; 64],
    count:   AtomicU64,
    sum:     AtomicU64,
    min:     AtomicU64,
    max:     AtomicU64,
}

impl Default for AtomicHistogram {
    fn default() -> Self {
        const INIT: AtomicU64 = AtomicU64::new(0);
        Self {
            buckets: [INIT; 64],
            count:   AtomicU64::new(0),
            sum:     AtomicU64::new(0),
            min:     AtomicU64::new(u64::MAX),
            max:     AtomicU64::new(0),
        }
    }
}

impl AtomicHistogram {
    pub fn record(&self, value: u64) {
        let bucket = (64 - value.leading_zeros().min(64)) as usize;
        let bucket = bucket.min(63);
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(value, Ordering::Relaxed);
        // Update min
        let mut old = self.min.load(Ordering::Relaxed);
        while value < old {
            match self.min.compare_exchange_weak(old, value, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => old = x,
            }
        }
        // Update max
        let mut old = self.max.load(Ordering::Relaxed);
        while value > old {
            match self.max.compare_exchange_weak(old, value, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => old = x,
            }
        }
    }

    pub fn percentile(&self, p: f64) -> u64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 { return 0; }
        let target = (p / 100.0 * total as f64) as u64;
        let mut cum = 0u64;
        for (b, bucket) in self.buckets.iter().enumerate() {
            cum += bucket.load(Ordering::Relaxed);
            if cum > target {
                return if b == 0 { 1 } else { 3u64 << (b - 1) };
            }
        }
        3u64 << 62
    }

    pub fn p50(&self)  -> u64 { self.percentile(50.0) }
    pub fn p95(&self)  -> u64 { self.percentile(95.0) }
    pub fn p99(&self)  -> u64 { self.percentile(99.0) }
    pub fn p999(&self) -> u64 { self.percentile(99.9) }

    pub fn count(&self) -> u64 { self.count.load(Ordering::Relaxed) }
    pub fn sum(&self)   -> u64 { self.sum.load(Ordering::Relaxed) }
    pub fn mean(&self)  -> f64 {
        let c = self.count();
        if c == 0 { 0.0 } else { self.sum() as f64 / c as f64 }
    }
    pub fn min(&self)   -> u64 {
        let m = self.min.load(Ordering::Relaxed);
        if m == u64::MAX { 0 } else { m }
    }
    pub fn max(&self)   -> u64 { self.max.load(Ordering::Relaxed) }

    pub fn reset(&self) {
        for b in &self.buckets { b.store(0, Ordering::Relaxed); }
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// ChannelMetrics — per-channel metric counters
// ---------------------------------------------------------------------------
pub struct ChannelMetrics {
    pub channel_name:      String,
    pub publish_total:     AtomicU64,
    pub consume_total:     AtomicU64,
    pub ring_full_total:   AtomicU64,
    pub bytes_written:     AtomicU64,
    pub bytes_read:        AtomicU64,
    pub publish_latency:   AtomicHistogram,
    pub consume_latency:   AtomicHistogram,
    pub sequence_lag:      AtomicU64,
}

impl ChannelMetrics {
    pub fn new(name: &str) -> Self {
        Self {
            channel_name:    name.to_owned(),
            publish_total:   AtomicU64::new(0),
            consume_total:   AtomicU64::new(0),
            ring_full_total: AtomicU64::new(0),
            bytes_written:   AtomicU64::new(0),
            bytes_read:      AtomicU64::new(0),
            publish_latency: AtomicHistogram::default(),
            consume_latency: AtomicHistogram::default(),
            sequence_lag:    AtomicU64::new(0),
        }
    }

    pub fn record_publish(&self, bytes: u64, latency_ns: u64) {
        self.publish_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
        self.publish_latency.record(latency_ns);
    }

    pub fn record_consume(&self, bytes: u64, latency_ns: u64) {
        self.consume_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        self.consume_latency.record(latency_ns);
    }

    pub fn record_ring_full(&self) {
        self.ring_full_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_lag(&self, lag: u64) {
        self.sequence_lag.store(lag, Ordering::Relaxed);
    }

    pub fn to_prometheus(&self) -> String {
        let n = &self.channel_name;
        let safe = n.replace('.', "_").replace('-', "_");
        let mut out = String::new();

        out.push_str(&format!("rtel_publish_total{{channel=\"{}\"}} {}\n",
            n, self.publish_total.load(Ordering::Relaxed)));
        out.push_str(&format!("rtel_consume_total{{channel=\"{}\"}} {}\n",
            n, self.consume_total.load(Ordering::Relaxed)));
        out.push_str(&format!("rtel_ring_full_total{{channel=\"{}\"}} {}\n",
            n, self.ring_full_total.load(Ordering::Relaxed)));
        out.push_str(&format!("rtel_bytes_written{{channel=\"{}\"}} {}\n",
            n, self.bytes_written.load(Ordering::Relaxed)));
        out.push_str(&format!("rtel_publish_latency_p50_ns{{channel=\"{}\"}} {}\n",
            n, self.publish_latency.p50()));
        out.push_str(&format!("rtel_publish_latency_p99_ns{{channel=\"{}\"}} {}\n",
            n, self.publish_latency.p99()));
        out.push_str(&format!("rtel_sequence_lag{{channel=\"{}\"}} {}\n",
            n, self.sequence_lag.load(Ordering::Relaxed)));
        out
    }
}

// ---------------------------------------------------------------------------
// RTELMetrics — global metrics registry
// ---------------------------------------------------------------------------
pub struct RTELMetrics {
    channels:      std::sync::Mutex<HashMap<String, Arc<ChannelMetrics>>>,
    start_time_ns: u64,
    // Global counters
    pub lob_publishes:       AtomicU64,
    pub vol_publishes:       AtomicU64,
    pub pipeline_runs:       AtomicU64,
    pub serialization_errors:AtomicU64,
    pub connection_errors:   AtomicU64,
}

impl Default for RTELMetrics {
    fn default() -> Self {
        Self {
            channels:             std::sync::Mutex::new(HashMap::new()),
            start_time_ns:        now_ns(),
            lob_publishes:        AtomicU64::new(0),
            vol_publishes:        AtomicU64::new(0),
            pipeline_runs:        AtomicU64::new(0),
            serialization_errors: AtomicU64::new(0),
            connection_errors:    AtomicU64::new(0),
        }
    }
}

impl RTELMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn channel(&self, name: &str) -> Arc<ChannelMetrics> {
        let mut map = self.channels.lock().unwrap();
        map.entry(name.to_owned())
            .or_insert_with(|| Arc::new(ChannelMetrics::new(name)))
            .clone()
    }

    pub fn uptime_ns(&self) -> u64 {
        now_ns().saturating_sub(self.start_time_ns)
    }

    pub fn export_prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str("# HELP rtel_uptime_ns RTEL uptime in nanoseconds\n");
        out.push_str("# TYPE rtel_uptime_ns gauge\n");
        out.push_str(&format!("rtel_uptime_ns {}\n", self.uptime_ns()));
        out.push_str(&format!("rtel_lob_publishes_total {}\n",
            self.lob_publishes.load(Ordering::Relaxed)));
        out.push_str(&format!("rtel_vol_publishes_total {}\n",
            self.vol_publishes.load(Ordering::Relaxed)));
        out.push_str(&format!("rtel_pipeline_runs_total {}\n",
            self.pipeline_runs.load(Ordering::Relaxed)));

        let map = self.channels.lock().unwrap();
        out.push_str("# HELP rtel_channel_metrics Per-channel metrics\n");
        for cm in map.values() {
            out.push_str(&cm.to_prometheus());
        }
        out
    }

    pub fn print_summary(&self) {
        println!("=== RTEL Rust Metrics ===");
        println!("  Uptime:          {} s", self.uptime_ns() / 1_000_000_000);
        println!("  LOB publishes:   {}", self.lob_publishes.load(Ordering::Relaxed));
        println!("  Vol publishes:   {}", self.vol_publishes.load(Ordering::Relaxed));
        println!("  Pipeline runs:   {}", self.pipeline_runs.load(Ordering::Relaxed));
        println!("  Errors:          {}",
            self.serialization_errors.load(Ordering::Relaxed)
            + self.connection_errors.load(Ordering::Relaxed));
        let map = self.channels.lock().unwrap();
        for (name, cm) in map.iter() {
            println!("  Chan {:40} pub={} cons={} lag={}",
                name,
                cm.publish_total.load(Ordering::Relaxed),
                cm.consume_total.load(Ordering::Relaxed),
                cm.sequence_lag.load(Ordering::Relaxed));
        }
    }
}

// Global instance
use std::sync::OnceLock;
static GLOBAL_METRICS: OnceLock<RTELMetrics> = OnceLock::new();

pub fn global_metrics() -> &'static RTELMetrics {
    GLOBAL_METRICS.get_or_init(RTELMetrics::new)
}

// ---------------------------------------------------------------------------
// Throughput monitor — rolling window ops/sec
// ---------------------------------------------------------------------------
pub struct ThroughputMonitor {
    window:  Vec<u64>,
    head:    std::sync::atomic::AtomicUsize,
    count:   AtomicU64,
}

impl ThroughputMonitor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: vec![0u64; window_size],
            head:   std::sync::atomic::AtomicUsize::new(0),
            count:  AtomicU64::new(0),
        }
    }

    pub fn record(&self) {
        let t = now_ns();
        let idx = self.head.fetch_add(1, Ordering::Relaxed) % self.window.len();
        // Safety: we own this index in the rolling window
        unsafe {
            let ptr = self.window.as_ptr().add(idx) as *mut u64;
            *ptr = t;
        }
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn rate_per_second(&self) -> f64 {
        let mut min_t = u64::MAX;
        let mut max_t = 0u64;
        let mut valid = 0;
        for &t in &self.window {
            if t > 0 {
                min_t = min_t.min(t);
                max_t = max_t.max(t);
                valid += 1;
            }
        }
        if valid < 2 || max_t <= min_t { return 0.0; }
        let elapsed_s = (max_t - min_t) as f64 / 1e9;
        valid as f64 / elapsed_s
    }

    pub fn total(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// LatencyTracker — records end-to-end latency for LOB publish pipeline
// ---------------------------------------------------------------------------
pub struct LatencyTracker {
    pub publish_to_gsr:  AtomicHistogram,
    pub shm_write:       AtomicHistogram,
    pub serialization:   AtomicHistogram,
    pub end_to_end:      AtomicHistogram,
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self {
            publish_to_gsr: AtomicHistogram::default(),
            shm_write:      AtomicHistogram::default(),
            serialization:  AtomicHistogram::default(),
            end_to_end:     AtomicHistogram::default(),
        }
    }
}

impl LatencyTracker {
    pub fn report(&self) -> String {
        format!(
            "publish_to_gsr: p50={} p99={} ns | shm_write: p50={} p99={} ns | e2e: p50={} p99={} ns",
            self.publish_to_gsr.p50(), self.publish_to_gsr.p99(),
            self.shm_write.p50(),      self.shm_write.p99(),
            self.end_to_end.p50(),     self.end_to_end.p99(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let h = AtomicHistogram::default();
        for i in 1u64..=100 {
            h.record(i * 1000);
        }
        assert_eq!(h.count(), 100);
        assert!(h.p50() > 0);
        assert!(h.p99() >= h.p50());
        assert!(h.max() >= h.min());
        println!("p50={} p99={} mean={:.1}", h.p50(), h.p99(), h.mean());
    }

    #[test]
    fn test_channel_metrics() {
        let m = ChannelMetrics::new("test.channel");
        m.record_publish(1024, 5000);
        m.record_publish(2048, 8000);
        m.record_consume(1024, 3000);
        assert_eq!(m.publish_total.load(Ordering::Relaxed), 2);
        assert_eq!(m.consume_total.load(Ordering::Relaxed), 1);
        assert_eq!(m.bytes_written.load(Ordering::Relaxed), 3072);
        let prom = m.to_prometheus();
        assert!(prom.contains("rtel_publish_total"));
    }

    #[test]
    fn test_throughput_monitor() {
        let mon = ThroughputMonitor::new(1000);
        for _ in 0..100 {
            mon.record();
            std::thread::sleep(std::time::Duration::from_micros(10));
        }
        assert_eq!(mon.total(), 100);
        let rate = mon.rate_per_second();
        println!("ThroughputMonitor rate: {:.1} ops/s", rate);
        // Rate should be very roughly 100 / (100 * 10µs) = ~100K ops/s
        // (but can vary widely in tests)
    }

    #[test]
    fn test_global_metrics() {
        let m = global_metrics();
        m.lob_publishes.fetch_add(1, Ordering::Relaxed);
        let prom = m.export_prometheus();
        assert!(prom.contains("rtel_lob_publishes_total 1"));
    }
}
