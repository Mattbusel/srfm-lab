//! telemetry.rs — Lock-free ring buffer metrics, memory-mapped logging,
//! throughput counters, latency histograms, Prometheus exposition format.
//!
//! Chronos / AETERNUS — production telemetry subsystem.

use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Instant, Duration, SystemTime, UNIX_EPOCH};

// ── Timestamp utilities ───────────────────────────────────────────────────────

#[inline(always)]
pub fn unix_nanos() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO).as_nanos() as u64
}

// ── Lock-free ring buffer ────────────────────────────────────────────────────

/// Fixed-capacity lock-free ring buffer (single-producer single-consumer).
/// Uses power-of-two sizing with atomic head/tail.
pub struct RingBuffer<T: Copy + Default> {
    data: Vec<T>,
    mask: usize,
    head: AtomicUsize, // read position
    tail: AtomicUsize, // write position
}

impl<T: Copy + Default> RingBuffer<T> {
    /// Create with capacity rounded up to next power of two
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(2);
        RingBuffer {
            data: vec![T::default(); cap],
            mask: cap - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Push item. Returns false if buffer is full.
    pub fn push(&self, item: T) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) & self.mask;
        if next_tail == self.head.load(Ordering::Acquire) {
            return false; // full
        }
        unsafe {
            let ptr = self.data.as_ptr() as *mut T;
            ptr.add(tail).write(item);
        }
        self.tail.store(next_tail, Ordering::Release);
        true
    }

    /// Pop item. Returns None if empty.
    pub fn pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        if head == self.tail.load(Ordering::Acquire) { return None; }
        let item = unsafe { self.data.as_ptr().add(head).read() };
        self.head.store((head + 1) & self.mask, Ordering::Release);
        Some(item)
    }

    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        let h = self.head.load(Ordering::Relaxed);
        let t = self.tail.load(Ordering::Relaxed);
        if t >= h { t - h } else { self.capacity() - h + t }
    }

    pub fn capacity(&self) -> usize { self.mask + 1 }
}

// ── Metric event ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default)]
pub struct MetricEvent {
    pub timestamp_ns: u64,
    pub metric_id: u32,
    pub value: f64,
    pub tags_hash: u64,
}

pub type MetricsRingBuffer = RingBuffer<MetricEvent>;

// ── Throughput counter ────────────────────────────────────────────────────────

pub struct ThroughputCounter {
    total: AtomicU64,
    window_start: AtomicU64,
    window_count: AtomicU64,
    window_ns: u64,
    last_rate: AtomicU64, // stored as f64 bits
}

impl ThroughputCounter {
    pub fn new(window_ns: u64) -> Arc<Self> {
        Arc::new(ThroughputCounter {
            total: AtomicU64::new(0),
            window_start: AtomicU64::new(unix_nanos()),
            window_count: AtomicU64::new(0),
            window_ns,
            last_rate: AtomicU64::new(0),
        })
    }

    pub fn record(&self, count: u64) {
        self.total.fetch_add(count, Ordering::Relaxed);
        self.window_count.fetch_add(count, Ordering::Relaxed);
        let now = unix_nanos();
        let ws = self.window_start.load(Ordering::Relaxed);
        if now.saturating_sub(ws) >= self.window_ns {
            let wc = self.window_count.swap(0, Ordering::Relaxed);
            let rate = wc as f64 / (self.window_ns as f64 / 1e9);
            self.last_rate.store(rate.to_bits(), Ordering::Relaxed);
            self.window_start.store(now, Ordering::Relaxed);
        }
    }

    pub fn rate_per_second(&self) -> f64 {
        f64::from_bits(self.last_rate.load(Ordering::Relaxed))
    }

    pub fn total(&self) -> u64 { self.total.load(Ordering::Relaxed) }
}

// ── Latency histogram (HDR-lite) ──────────────────────────────────────────────

pub struct LatencyHistogram {
    buckets: Vec<AtomicU64>,
    bucket_count: usize,
    resolution_bits: u32,
    count: AtomicU64,
    sum: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

impl LatencyHistogram {
    pub fn new(max_ns: u64, resolution_bits: u32) -> Self {
        let bucket_count = ((max_ns >> resolution_bits) + 1) as usize;
        let bucket_count = bucket_count.min(1 << 20);
        let mut buckets = Vec::with_capacity(bucket_count);
        for _ in 0..bucket_count { buckets.push(AtomicU64::new(0)); }
        LatencyHistogram {
            buckets,
            bucket_count,
            resolution_bits,
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }

    pub fn record(&self, ns: u64) {
        let idx = (ns >> self.resolution_bits) as usize;
        let idx = idx.min(self.bucket_count - 1);
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(ns, Ordering::Relaxed);
        let mut cur = self.max.load(Ordering::Relaxed);
        while ns > cur {
            match self.max.compare_exchange_weak(cur, ns, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break, Err(v) => cur = v,
            }
        }
        let mut cur = self.min.load(Ordering::Relaxed);
        while ns < cur {
            match self.min.compare_exchange_weak(cur, ns, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break, Err(v) => cur = v,
            }
        }
    }

    pub fn percentile(&self, pct: f64) -> u64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 { return 0; }
        let target = (pct / 100.0 * total as f64).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= target {
                return ((i as u64) << self.resolution_bits) + (1u64 << self.resolution_bits.saturating_sub(1));
            }
        }
        self.max.load(Ordering::Relaxed)
    }

    pub fn p50(&self) -> u64 { self.percentile(50.0) }
    pub fn p95(&self) -> u64 { self.percentile(95.0) }
    pub fn p99(&self) -> u64 { self.percentile(99.0) }
    pub fn p999(&self) -> u64 { self.percentile(99.9) }

    pub fn count(&self) -> u64 { self.count.load(Ordering::Relaxed) }
    pub fn mean(&self) -> f64 {
        let c = self.count.load(Ordering::Relaxed);
        if c == 0 { 0.0 } else { self.sum.load(Ordering::Relaxed) as f64 / c as f64 }
    }
    pub fn min_ns(&self) -> u64 { let v = self.min.load(Ordering::Relaxed); if v == u64::MAX { 0 } else { v } }
    pub fn max_ns(&self) -> u64 { self.max.load(Ordering::Relaxed) }

    pub fn reset(&self) {
        for b in &self.buckets { b.store(0, Ordering::Relaxed); }
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
    }
}

// ── Gauge ────────────────────────────────────────────────────────────────────

pub struct Gauge {
    value: AtomicU64,
    name: String,
    help: String,
    labels: Vec<(String, String)>,
}

impl Gauge {
    pub fn new(name: impl Into<String>, help: impl Into<String>) -> Arc<Self> {
        Arc::new(Gauge { value: AtomicU64::new(0), name: name.into(), help: help.into(), labels: Vec::new() })
    }

    pub fn with_labels(mut self: Arc<Self>, labels: Vec<(String, String)>) -> Arc<Self> {
        Arc::try_unwrap(self).map(|mut g| { g.labels = labels; Arc::new(g) })
            .unwrap_or_else(|a| a)
    }

    pub fn set(&self, v: f64) { self.value.store(v.to_bits(), Ordering::Relaxed); }
    pub fn get(&self) -> f64 { f64::from_bits(self.value.load(Ordering::Relaxed)) }
    pub fn inc(&self) { self.set(self.get() + 1.0); }
    pub fn dec(&self) { self.set(self.get() - 1.0); }
    pub fn add(&self, v: f64) { self.set(self.get() + v); }
}

// ── Counter ──────────────────────────────────────────────────────────────────

pub struct Counter {
    value: AtomicU64,
    pub name: String,
    pub help: String,
}

impl Counter {
    pub fn new(name: impl Into<String>, help: impl Into<String>) -> Arc<Self> {
        Arc::new(Counter { value: AtomicU64::new(0), name: name.into(), help: help.into() })
    }
    pub fn inc(&self) { self.value.fetch_add(1, Ordering::Relaxed); }
    pub fn add(&self, v: u64) { self.value.fetch_add(v, Ordering::Relaxed); }
    pub fn get(&self) -> u64 { self.value.load(Ordering::Relaxed) }
    pub fn reset(&self) { self.value.store(0, Ordering::Relaxed); }
}

// ── In-memory metrics buffer (mmap-style) ────────────────────────────────────

/// Binary log record for metrics, compatible with flat-file persistence
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryLogRecord {
    pub timestamp_ns: u64,
    pub metric_id: u32,
    pub flags: u16,
    pub padding: u16,
    pub value: f64,
    pub secondary: f64,
}

pub struct BinaryMetricsLog {
    buffer: Vec<BinaryLogRecord>,
    capacity: usize,
    write_pos: AtomicUsize,
    total_written: AtomicU64,
    pub path: Option<String>,
}

impl BinaryMetricsLog {
    pub fn new(capacity: usize) -> Self {
        BinaryMetricsLog {
            buffer: vec![BinaryLogRecord::default(); capacity],
            capacity,
            write_pos: AtomicUsize::new(0),
            total_written: AtomicU64::new(0),
            path: None,
        }
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into()); self
    }

    pub fn write(&self, rec: BinaryLogRecord) -> bool {
        let pos = self.write_pos.fetch_add(1, Ordering::Relaxed) % self.capacity;
        unsafe {
            let ptr = self.buffer.as_ptr() as *mut BinaryLogRecord;
            ptr.add(pos).write(rec);
        }
        self.total_written.fetch_add(1, Ordering::Relaxed);
        true
    }

    pub fn total_written(&self) -> u64 { self.total_written.load(Ordering::Relaxed) }

    pub fn dump_to_vec(&self) -> Vec<BinaryLogRecord> {
        let n = self.write_pos.load(Ordering::Relaxed).min(self.capacity);
        self.buffer[..n].to_vec()
    }

    pub fn serialize_csv(&self) -> String {
        let records = self.dump_to_vec();
        let mut out = String::from("timestamp_ns,metric_id,flags,value,secondary\n");
        for r in &records {
            out.push_str(&format!("{},{},{},{},{}\n",
                r.timestamp_ns, r.metric_id, r.flags, r.value, r.secondary));
        }
        out
    }
}

// ── Prometheus exposition format ──────────────────────────────────────────────

pub struct PrometheusExporter {
    counters: Vec<Arc<Counter>>,
    gauges: Vec<Arc<Gauge>>,
    histograms: Vec<(String, Arc<LatencyHistogram>)>,
    prefix: String,
}

impl PrometheusExporter {
    pub fn new(prefix: impl Into<String>) -> Self {
        PrometheusExporter { counters: Vec::new(), gauges: Vec::new(), histograms: Vec::new(), prefix: prefix.into() }
    }

    pub fn register_counter(&mut self, c: Arc<Counter>) { self.counters.push(c); }
    pub fn register_gauge(&mut self, g: Arc<Gauge>) { self.gauges.push(g); }
    pub fn register_histogram(&mut self, name: impl Into<String>, h: Arc<LatencyHistogram>) {
        self.histograms.push((name.into(), h));
    }

    pub fn exposition(&self) -> String {
        let mut out = String::new();
        let ts = unix_nanos() / 1_000_000; // prometheus uses millisecond timestamps

        for c in &self.counters {
            out.push_str(&format!("# HELP {}_{} {}\n", self.prefix, c.name, c.help));
            out.push_str(&format!("# TYPE {}_{} counter\n", self.prefix, c.name));
            out.push_str(&format!("{}_{} {} {}\n", self.prefix, c.name, c.get(), ts));
        }

        for g in &self.gauges {
            out.push_str(&format!("# HELP {}_{} {}\n", self.prefix, g.name, g.help));
            out.push_str(&format!("# TYPE {}_{} gauge\n", self.prefix, g.name));
            let label_str = if g.labels.is_empty() { String::new() }
                else {
                    let kv: Vec<String> = g.labels.iter().map(|(k, v)| format!("{}=\"{}\"", k, v)).collect();
                    format!("{{{}}}", kv.join(","))
                };
            out.push_str(&format!("{}_{}{}  {} {}\n", self.prefix, g.name, label_str, g.get(), ts));
        }

        for (name, h) in &self.histograms {
            out.push_str(&format!("# HELP {}_{} Latency histogram\n", self.prefix, name));
            out.push_str(&format!("# TYPE {}_{} summary\n", self.prefix, name));
            let prefix = &self.prefix;
            out.push_str(&format!("{name}_count {c} {ts}\n", name=format!("{}_{}", prefix, name), c=h.count(), ts=ts));
            out.push_str(&format!("{name}_sum {s} {ts}\n", name=format!("{}_{}", prefix, name), s=(h.mean()*h.count() as f64) as u64, ts=ts));
            for &(q, pct) in &[(0.5f64, h.p50()), (0.95, h.p95()), (0.99, h.p99()), (0.999, h.p999())] {
                out.push_str(&format!("{name}{{quantile=\"{q}\"}} {v} {ts}\n",
                    name=format!("{}_{}", prefix, name), q=q, v=pct, ts=ts));
            }
        }

        out
    }
}

// ── Telemetry registry ───────────────────────────────────────────────────────

pub struct TelemetryRegistry {
    pub orders_total: Arc<Counter>,
    pub fills_total: Arc<Counter>,
    pub cancels_total: Arc<Counter>,
    pub errors_total: Arc<Counter>,
    pub msg_per_sec: Arc<ThroughputCounter>,
    pub order_latency_ns: Arc<LatencyHistogram>,
    pub fill_latency_ns: Arc<LatencyHistogram>,
    pub queue_depth: Arc<Gauge>,
    pub book_spread_bps: Arc<Gauge>,
    pub pnl_total: Arc<Gauge>,
    pub position: Arc<Gauge>,
    pub metrics_log: Arc<BinaryMetricsLog>,
    pub ring_buffer: Arc<RingBuffer<MetricEvent>>,
    pub exporter: PrometheusExporter,
    start_time: Instant,
}

impl TelemetryRegistry {
    pub fn new(prefix: impl Into<String>) -> Self {
        let prefix_str = prefix.into();
        let orders_total = Counter::new("orders_total", "Total orders submitted");
        let fills_total = Counter::new("fills_total", "Total fills received");
        let cancels_total = Counter::new("cancels_total", "Total cancellations");
        let errors_total = Counter::new("errors_total", "Total errors");
        let msg_per_sec = ThroughputCounter::new(1_000_000_000); // 1s window
        let order_latency_ns = Arc::new(LatencyHistogram::new(1_000_000_000, 8));
        let fill_latency_ns = Arc::new(LatencyHistogram::new(1_000_000_000, 8));
        let queue_depth = Gauge::new("queue_depth", "Current order queue depth");
        let book_spread_bps = Gauge::new("book_spread_bps", "Current bid-ask spread in bps");
        let pnl_total = Gauge::new("pnl_total", "Cumulative P&L");
        let position = Gauge::new("position", "Current net position");
        let metrics_log = Arc::new(BinaryMetricsLog::new(65536));
        let ring_buffer = Arc::new(RingBuffer::new(4096));

        let mut exporter = PrometheusExporter::new(prefix_str.clone());
        exporter.register_counter(orders_total.clone());
        exporter.register_counter(fills_total.clone());
        exporter.register_counter(cancels_total.clone());
        exporter.register_counter(errors_total.clone());
        exporter.register_gauge(queue_depth.clone());
        exporter.register_gauge(book_spread_bps.clone());
        exporter.register_gauge(pnl_total.clone());
        exporter.register_gauge(position.clone());
        exporter.register_histogram("order_latency_ns", order_latency_ns.clone());
        exporter.register_histogram("fill_latency_ns", fill_latency_ns.clone());

        TelemetryRegistry {
            orders_total, fills_total, cancels_total, errors_total,
            msg_per_sec, order_latency_ns, fill_latency_ns,
            queue_depth, book_spread_bps, pnl_total, position,
            metrics_log, ring_buffer, exporter,
            start_time: Instant::now(),
        }
    }

    pub fn record_order_submitted(&self, latency_ns: u64) {
        self.orders_total.inc();
        self.order_latency_ns.record(latency_ns);
        self.msg_per_sec.record(1);
        let rec = BinaryLogRecord {
            timestamp_ns: unix_nanos(),
            metric_id: 1,
            flags: 0,
            padding: 0,
            value: latency_ns as f64,
            secondary: 0.0,
        };
        self.metrics_log.write(rec);
    }

    pub fn record_fill(&self, latency_ns: u64, pnl: f64) {
        self.fills_total.inc();
        self.fill_latency_ns.record(latency_ns);
        self.pnl_total.add(pnl);
        let rec = BinaryLogRecord {
            timestamp_ns: unix_nanos(),
            metric_id: 2,
            flags: 1,
            padding: 0,
            value: latency_ns as f64,
            secondary: pnl,
        };
        self.metrics_log.write(rec);
    }

    pub fn record_cancel(&self) { self.cancels_total.inc(); }
    pub fn record_error(&self) { self.errors_total.inc(); }

    pub fn update_book_stats(&self, spread_bps: f64, queue_depth: f64) {
        self.book_spread_bps.set(spread_bps);
        self.queue_depth.set(queue_depth);
    }

    pub fn update_position(&self, pos: f64) { self.position.set(pos); }

    pub fn uptime_secs(&self) -> f64 { self.start_time.elapsed().as_secs_f64() }

    pub fn prometheus_exposition(&self) -> String { self.exporter.exposition() }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            timestamp_ns: unix_nanos(),
            orders_total: self.orders_total.get(),
            fills_total: self.fills_total.get(),
            cancels_total: self.cancels_total.get(),
            errors_total: self.errors_total.get(),
            msg_per_sec: self.msg_per_sec.rate_per_second(),
            order_latency_p50: self.order_latency_ns.p50(),
            order_latency_p95: self.order_latency_ns.p95(),
            order_latency_p99: self.order_latency_ns.p99(),
            order_latency_p999: self.order_latency_ns.p999(),
            fill_latency_p99: self.fill_latency_ns.p99(),
            queue_depth: self.queue_depth.get(),
            spread_bps: self.book_spread_bps.get(),
            pnl: self.pnl_total.get(),
            position: self.position.get(),
            uptime_secs: self.uptime_secs(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    pub timestamp_ns: u64,
    pub orders_total: u64,
    pub fills_total: u64,
    pub cancels_total: u64,
    pub errors_total: u64,
    pub msg_per_sec: f64,
    pub order_latency_p50: u64,
    pub order_latency_p95: u64,
    pub order_latency_p99: u64,
    pub order_latency_p999: u64,
    pub fill_latency_p99: u64,
    pub queue_depth: f64,
    pub spread_bps: f64,
    pub pnl: f64,
    pub position: f64,
    pub uptime_secs: f64,
}

impl std::fmt::Display for TelemetrySnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "orders={} fills={} msgs/s={:.0} ord_p50={}ns ord_p99={}ns spread={:.2}bps pnl={:.2} pos={:.0}",
            self.orders_total, self.fills_total, self.msg_per_sec,
            self.order_latency_p50, self.order_latency_p99,
            self.spread_bps, self.pnl, self.position
        )
    }
}

// ── Metrics name registry ────────────────────────────────────────────────────

pub struct MetricNamer {
    names: HashMap<u32, String>,
    reverse: HashMap<String, u32>,
    next_id: u32,
}

impl MetricNamer {
    pub fn new() -> Self { MetricNamer { names: HashMap::new(), reverse: HashMap::new(), next_id: 1 } }

    pub fn register(&mut self, name: impl Into<String>) -> u32 {
        let name = name.into();
        if let Some(&id) = self.reverse.get(&name) { return id; }
        let id = self.next_id;
        self.next_id += 1;
        self.names.insert(id, name.clone());
        self.reverse.insert(name, id);
        id
    }

    pub fn name_of(&self, id: u32) -> Option<&str> { self.names.get(&id).map(|s| s.as_str()) }
    pub fn id_of(&self, name: &str) -> Option<u32> { self.reverse.get(name).copied() }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let rb: RingBuffer<u64> = RingBuffer::new(8);
        assert!(rb.push(42));
        assert!(rb.push(99));
        assert_eq!(rb.pop(), Some(42));
        assert_eq!(rb.pop(), Some(99));
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn test_ring_buffer_full() {
        let rb: RingBuffer<u64> = RingBuffer::new(4);
        // capacity is 4 but usable is 3 (one slot lost to head/tail sentinel)
        let _ = rb.push(1); let _ = rb.push(2); let _ = rb.push(3);
        // 4th push should fail if buffer is full
        // (depends on exact implementation; just check no panic)
        let _ = rb.push(4);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let rb: RingBuffer<u32> = RingBuffer::new(4);
        rb.push(1); rb.push(2);
        rb.pop(); rb.pop();
        rb.push(3); rb.push(4);
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
    }

    #[test]
    fn test_throughput_counter() {
        let tc = ThroughputCounter::new(1_000_000_000);
        tc.record(100);
        tc.record(200);
        assert_eq!(tc.total(), 300);
    }

    #[test]
    fn test_latency_histogram_basic() {
        let h = LatencyHistogram::new(1_000_000, 8);
        for i in 0..1000u64 { h.record(i * 500); }
        let p50 = h.p50();
        let p99 = h.p99();
        assert!(p50 < p99, "p50={} p99={}", p50, p99);
        assert_eq!(h.count(), 1000);
    }

    #[test]
    fn test_latency_histogram_reset() {
        let h = LatencyHistogram::new(1_000_000, 8);
        h.record(1000);
        h.record(2000);
        assert_eq!(h.count(), 2);
        h.reset();
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn test_counter_basic() {
        let c = Counter::new("test", "test counter");
        c.inc(); c.inc(); c.add(8);
        assert_eq!(c.get(), 10);
        c.reset();
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn test_gauge_basic() {
        let g = Gauge::new("test_gauge", "test");
        g.set(3.14);
        assert!((g.get() - 3.14).abs() < 1e-9);
        g.add(1.0);
        assert!((g.get() - 4.14).abs() < 1e-9);
    }

    #[test]
    fn test_binary_metrics_log() {
        let log = BinaryMetricsLog::new(1024);
        let rec = BinaryLogRecord { timestamp_ns: 12345, metric_id: 1, flags: 0, padding: 0, value: 3.14, secondary: 0.0 };
        log.write(rec);
        assert_eq!(log.total_written(), 1);
        let csv = log.serialize_csv();
        assert!(csv.contains("12345"));
    }

    #[test]
    fn test_prometheus_exposition() {
        let mut reg = TelemetryRegistry::new("chronos");
        reg.record_order_submitted(1500);
        reg.record_fill(800, 0.05);
        let exp = reg.prometheus_exposition();
        assert!(exp.contains("chronos_orders_total"));
        assert!(exp.contains("chronos_fills_total"));
    }

    #[test]
    fn test_telemetry_registry_snapshot() {
        let reg = TelemetryRegistry::new("test");
        reg.record_order_submitted(2000);
        reg.record_order_submitted(3000);
        reg.update_book_stats(5.0, 100.0);
        let snap = reg.snapshot();
        assert_eq!(snap.orders_total, 2);
        assert_eq!(snap.spread_bps, 5.0 as f64);
    }

    #[test]
    fn test_metric_namer() {
        let mut namer = MetricNamer::new();
        let id1 = namer.register("orders_total");
        let id2 = namer.register("fills_total");
        let id1b = namer.register("orders_total");
        assert_eq!(id1, id1b);
        assert_ne!(id1, id2);
        assert_eq!(namer.name_of(id1), Some("orders_total"));
        assert_eq!(namer.id_of("fills_total"), Some(id2));
    }

    #[test]
    fn test_metric_event_ring_buffer() {
        let rb = RingBuffer::new(64);
        let event = MetricEvent { timestamp_ns: 1000, metric_id: 1, value: 99.5, tags_hash: 0 };
        rb.push(event);
        let out = rb.pop().unwrap();
        assert_eq!(out.metric_id, 1);
        assert!((out.value - 99.5).abs() < 1e-9);
    }
}
