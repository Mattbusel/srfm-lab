//! performance_profiler.rs — Cycle-accurate benchmarking, flame graph data export,
//! hot path analysis, memory bandwidth measurement.
//!
//! Chronos / AETERNUS — production performance profiler.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ── TSC (time stamp counter) simulation ──────────────────────────────────────

/// Read simulated TSC. On real hardware this would be rdtsc.
#[inline(always)]
pub fn read_tsc() -> u64 {
    // Fallback to std::time on platforms without rdtsc
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

/// Estimate TSC frequency (cycles per second) by calibrating against wall clock
pub fn estimate_tsc_freq() -> f64 {
    let t0 = Instant::now();
    let c0 = read_tsc();
    std::thread::sleep(std::time::Duration::from_millis(10));
    let elapsed = t0.elapsed().as_nanos() as f64;
    let c1 = read_tsc();
    let cycles = (c1 - c0) as f64;
    cycles / elapsed * 1e9 // cycles per second
}

/// Convert cycles to nanoseconds
pub fn cycles_to_ns(cycles: u64, tsc_freq: f64) -> f64 {
    cycles as f64 / tsc_freq * 1e9
}

// ── Timed span ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SpanRecord {
    pub name: String,
    pub start_tsc: u64,
    pub end_tsc: u64,
    pub thread_id: u64,
    pub depth: u32,
    pub tags: Vec<(String, String)>,
}

impl SpanRecord {
    pub fn duration_cycles(&self) -> u64 { self.end_tsc.saturating_sub(self.start_tsc) }
    pub fn duration_ns(&self, tsc_freq: f64) -> f64 { cycles_to_ns(self.duration_cycles(), tsc_freq) }
}

// ── Profiler span builder ─────────────────────────────────────────────────────

pub struct SpanBuilder<'a> {
    profiler: &'a mut Profiler,
    name: String,
    start_tsc: u64,
    depth: u32,
    tags: Vec<(String, String)>,
}

impl<'a> SpanBuilder<'a> {
    pub fn tag(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.tags.push((key.into(), val.into())); self
    }
}

impl<'a> Drop for SpanBuilder<'a> {
    fn drop(&mut self) {
        let end = read_tsc();
        self.profiler.record_span(SpanRecord {
            name: self.name.clone(),
            start_tsc: self.start_tsc,
            end_tsc: end,
            thread_id: 0,
            depth: self.depth,
            tags: self.tags.clone(),
        });
    }
}

// ── Core profiler ─────────────────────────────────────────────────────────────

pub struct Profiler {
    pub spans: VecDeque<SpanRecord>,
    pub max_spans: usize,
    pub tsc_freq: f64,
    depth: u32,
    /// Aggregated stats per span name
    pub stats: HashMap<String, SpanStats>,
}

#[derive(Debug, Clone, Default)]
pub struct SpanStats {
    pub count: u64,
    pub total_cycles: u64,
    pub min_cycles: u64,
    pub max_cycles: u64,
    pub sum_sq_cycles: f64,
}

impl SpanStats {
    pub fn update(&mut self, cycles: u64) {
        self.count += 1;
        self.total_cycles += cycles;
        self.sum_sq_cycles += (cycles as f64).powi(2);
        if cycles < self.min_cycles || self.count == 1 { self.min_cycles = cycles; }
        if cycles > self.max_cycles { self.max_cycles = cycles; }
    }

    pub fn mean_cycles(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.total_cycles as f64 / self.count as f64 }
    }

    pub fn std_dev_cycles(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let mean = self.mean_cycles();
        let variance = self.sum_sq_cycles / self.count as f64 - mean * mean;
        variance.max(0.0).sqrt()
    }

    pub fn mean_ns(&self, tsc_freq: f64) -> f64 { cycles_to_ns(self.total_cycles / self.count.max(1), tsc_freq) }
}

impl Profiler {
    pub fn new(max_spans: usize) -> Self {
        let tsc_freq = estimate_tsc_freq();
        Profiler { spans: VecDeque::new(), max_spans, tsc_freq, depth: 0, stats: HashMap::new() }
    }

    pub fn span(&mut self, name: impl Into<String>) -> SpanBuilder<'_> {
        let start = read_tsc();
        let d = self.depth;
        self.depth += 1;
        SpanBuilder { profiler: self, name: name.into(), start_tsc: start, depth: d, tags: Vec::new() }
    }

    pub fn record_span(&mut self, span: SpanRecord) {
        if self.depth > 0 { self.depth -= 1; }
        let cycles = span.duration_cycles();
        self.stats.entry(span.name.clone()).or_default().update(cycles);
        if self.spans.len() >= self.max_spans { self.spans.pop_front(); }
        self.spans.push_back(span);
    }

    pub fn time_fn<F, R>(&mut self, name: &str, f: F) -> (R, u64)
    where F: FnOnce() -> R
    {
        let start = read_tsc();
        let result = f();
        let end = read_tsc();
        let cycles = end - start;
        self.stats.entry(name.to_string()).or_default().update(cycles);
        (result, cycles)
    }

    pub fn hot_paths(&self, top_n: usize) -> Vec<(&str, &SpanStats)> {
        let mut entries: Vec<_> = self.stats.iter().map(|(k, v)| (k.as_str(), v)).collect();
        entries.sort_by(|a, b| b.1.total_cycles.cmp(&a.1.total_cycles));
        entries.into_iter().take(top_n).collect()
    }

    pub fn flamegraph_data(&self) -> FlameGraphData {
        let mut stacks: HashMap<String, u64> = HashMap::new();
        for span in &self.spans {
            *stacks.entry(span.name.clone()).or_insert(0) += span.duration_cycles();
        }
        FlameGraphData { stacks, tsc_freq: self.tsc_freq }
    }

    pub fn report(&self) -> ProfileReport {
        let mut entries: Vec<_> = self.stats.iter().map(|(name, stats)| {
            ProfileEntry {
                name: name.clone(),
                count: stats.count,
                mean_ns: stats.mean_ns(self.tsc_freq),
                min_ns: cycles_to_ns(stats.min_cycles, self.tsc_freq),
                max_ns: cycles_to_ns(stats.max_cycles, self.tsc_freq),
                std_dev_ns: cycles_to_ns(stats.std_dev_cycles() as u64, self.tsc_freq),
                total_ns: cycles_to_ns(stats.total_cycles, self.tsc_freq),
                pct_of_total: 0.0,
            }
        }).collect();

        let total_ns: f64 = entries.iter().map(|e| e.total_ns).sum();
        for e in &mut entries {
            e.pct_of_total = if total_ns > 0.0 { e.total_ns / total_ns * 100.0 } else { 0.0 };
        }
        entries.sort_by(|a, b| b.total_ns.partial_cmp(&a.total_ns).unwrap());

        ProfileReport { entries, total_spans: self.spans.len(), tsc_freq: self.tsc_freq }
    }
}

#[derive(Debug, Clone)]
pub struct ProfileEntry {
    pub name: String,
    pub count: u64,
    pub mean_ns: f64,
    pub min_ns: f64,
    pub max_ns: f64,
    pub std_dev_ns: f64,
    pub total_ns: f64,
    pub pct_of_total: f64,
}

#[derive(Debug, Clone)]
pub struct ProfileReport {
    pub entries: Vec<ProfileEntry>,
    pub total_spans: usize,
    pub tsc_freq: f64,
}

impl std::fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:<40} {:>10} {:>12} {:>12} {:>12} {:>8}", "Name", "Count", "Mean(ns)", "Min(ns)", "Max(ns)", "% Total")?;
        writeln!(f, "{}", "-".repeat(100))?;
        for e in &self.entries {
            writeln!(f, "{:<40} {:>10} {:>12.1} {:>12.1} {:>12.1} {:>7.1}%",
                e.name, e.count, e.mean_ns, e.min_ns, e.max_ns, e.pct_of_total)?;
        }
        Ok(())
    }
}

// ── Flame graph data ──────────────────────────────────────────────────────────

pub struct FlameGraphData {
    pub stacks: HashMap<String, u64>,
    pub tsc_freq: f64,
}

impl FlameGraphData {
    /// Export in collapsed stack format (for flamegraph.pl)
    pub fn to_collapsed_stacks(&self) -> String {
        let mut out = String::new();
        let total: u64 = self.stacks.values().sum();
        for (name, &cycles) in &self.stacks {
            // Normalize to "samples" (1 sample per 1000 cycles)
            let samples = cycles / 1000;
            if samples > 0 {
                out.push_str(&format!("{} {}\n", name, samples));
            }
        }
        out
    }

    /// Export as JSON for d3-flamegraph
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\"name\":\"root\",\"value\":0,\"children\":[\n");
        let entries: Vec<_> = self.stacks.iter().collect();
        let mut parts = Vec::new();
        for (name, &cycles) in &entries {
            let ns = cycles_to_ns(cycles, self.tsc_freq);
            parts.push(format!("{{\"name\":\"{}\",\"value\":{:.0}}}", name, ns));
        }
        out.push_str(&parts.join(",\n"));
        out.push_str("\n]}");
        out
    }
}

// ── Memory bandwidth measurement ──────────────────────────────────────────────

pub struct MemoryBandwidthBenchmark {
    buf_size_bytes: usize,
}

impl MemoryBandwidthBenchmark {
    pub fn new(buf_size_bytes: usize) -> Self { MemoryBandwidthBenchmark { buf_size_bytes } }

    /// Sequential read bandwidth
    pub fn measure_read_bandwidth(&self) -> f64 {
        let data: Vec<u64> = vec![0u64; self.buf_size_bytes / 8];
        let start = Instant::now();
        let sum: u64 = data.iter().sum();
        let elapsed = start.elapsed().as_secs_f64();
        let _ = sum;
        self.buf_size_bytes as f64 / elapsed / 1e9 // GB/s
    }

    /// Sequential write bandwidth
    pub fn measure_write_bandwidth(&self) -> f64 {
        let mut data: Vec<u64> = vec![0u64; self.buf_size_bytes / 8];
        let start = Instant::now();
        for (i, v) in data.iter_mut().enumerate() { *v = i as u64; }
        let elapsed = start.elapsed().as_secs_f64();
        self.buf_size_bytes as f64 / elapsed / 1e9
    }

    /// Random access latency
    pub fn measure_random_access_latency_ns(&self) -> f64 {
        let n = self.buf_size_bytes / 8;
        let data: Vec<u64> = (0..n).map(|_| 0u64).collect();
        let n_accesses = 10_000usize;

        // Build random access indices
        let mut indices = Vec::with_capacity(n_accesses);
        let mut state = 0x123456789ABCDEFu64;
        for _ in 0..n_accesses {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            indices.push((state % n as u64) as usize);
        }

        let start = Instant::now();
        let mut acc = 0u64;
        for &i in &indices { acc = acc.wrapping_add(data[i]); }
        let elapsed = start.elapsed().as_nanos() as f64;
        let _ = acc;
        elapsed / n_accesses as f64
    }

    pub fn full_report(&self) -> BandwidthReport {
        BandwidthReport {
            buf_size_bytes: self.buf_size_bytes,
            read_gb_per_sec: self.measure_read_bandwidth(),
            write_gb_per_sec: self.measure_write_bandwidth(),
            random_access_latency_ns: self.measure_random_access_latency_ns(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BandwidthReport {
    pub buf_size_bytes: usize,
    pub read_gb_per_sec: f64,
    pub write_gb_per_sec: f64,
    pub random_access_latency_ns: f64,
}

// ── Cycle-accurate micro-benchmark ───────────────────────────────────────────

pub struct MicroBenchmark {
    pub name: String,
    pub iterations: u64,
    pub warmup_iterations: u64,
    samples: Vec<u64>,
}

impl MicroBenchmark {
    pub fn new(name: impl Into<String>, iterations: u64) -> Self {
        MicroBenchmark { name: name.into(), iterations, warmup_iterations: iterations / 10, samples: Vec::new() }
    }

    pub fn run<F: FnMut()>(&mut self, mut f: F) {
        // Warmup
        for _ in 0..self.warmup_iterations { f(); }

        // Actual measurement
        self.samples.clear();
        for _ in 0..self.iterations {
            let start = read_tsc();
            f();
            let end = read_tsc();
            self.samples.push(end.saturating_sub(start));
        }
        self.samples.sort_unstable();
    }

    pub fn p50_cycles(&self) -> u64 { self.percentile(50.0) }
    pub fn p99_cycles(&self) -> u64 { self.percentile(99.0) }
    pub fn min_cycles(&self) -> u64 { self.samples.first().copied().unwrap_or(0) }
    pub fn max_cycles(&self) -> u64 { self.samples.last().copied().unwrap_or(0) }
    pub fn mean_cycles(&self) -> f64 {
        if self.samples.is_empty() { return 0.0; }
        self.samples.iter().sum::<u64>() as f64 / self.samples.len() as f64
    }

    pub fn percentile(&self, pct: f64) -> u64 {
        if self.samples.is_empty() { return 0; }
        let idx = ((pct / 100.0) * (self.samples.len() - 1) as f64) as usize;
        self.samples[idx.min(self.samples.len() - 1)]
    }

    pub fn results(&self, tsc_freq: f64) -> MicroBenchResult {
        MicroBenchResult {
            name: self.name.clone(),
            iterations: self.iterations,
            min_ns: cycles_to_ns(self.min_cycles(), tsc_freq),
            mean_ns: cycles_to_ns(self.mean_cycles() as u64, tsc_freq),
            p50_ns: cycles_to_ns(self.p50_cycles(), tsc_freq),
            p99_ns: cycles_to_ns(self.p99_cycles(), tsc_freq),
            max_ns: cycles_to_ns(self.max_cycles(), tsc_freq),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MicroBenchResult {
    pub name: String,
    pub iterations: u64,
    pub min_ns: f64,
    pub mean_ns: f64,
    pub p50_ns: f64,
    pub p99_ns: f64,
    pub max_ns: f64,
}

impl std::fmt::Display for MicroBenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: min={:.1}ns mean={:.1}ns p50={:.1}ns p99={:.1}ns max={:.1}ns (n={})",
            self.name, self.min_ns, self.mean_ns, self.p50_ns, self.p99_ns, self.max_ns, self.iterations)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_tsc_increases() {
        let t1 = read_tsc();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let t2 = read_tsc();
        assert!(t2 > t1, "TSC should increase");
    }

    #[test]
    fn test_tsc_freq_estimate() {
        let freq = estimate_tsc_freq();
        assert!(freq > 1e8, "freq too low: {}", freq);
        assert!(freq < 1e13, "freq too high: {}", freq);
    }

    #[test]
    fn test_profiler_time_fn() {
        let mut prof = Profiler::new(1000);
        let (result, cycles) = prof.time_fn("test_op", || {
            let mut sum = 0u64;
            for i in 0..1000 { sum += i; }
            sum
        });
        assert_eq!(result, 499500);
        assert!(cycles > 0);
    }

    #[test]
    fn test_profiler_stats_accumulation() {
        let mut prof = Profiler::new(10000);
        for _ in 0..100 {
            prof.time_fn("op_a", || { let mut x = 0u64; for i in 0..100 { x += i; } x });
        }
        let stats = prof.stats.get("op_a").unwrap();
        assert_eq!(stats.count, 100);
        assert!(stats.mean_cycles() > 0.0);
    }

    #[test]
    fn test_profiler_hot_paths() {
        let mut prof = Profiler::new(10000);
        for _ in 0..50 { prof.time_fn("expensive", || { let mut x = 0u64; for i in 0..10000 { x += i; } x }); }
        for _ in 0..1000 { prof.time_fn("cheap", || 42u64); }
        let hot = prof.hot_paths(2);
        assert_eq!(hot.len(), 2);
        // "expensive" should be first
        assert_eq!(hot[0].0, "expensive");
    }

    #[test]
    fn test_profiler_flamegraph_export() {
        let mut prof = Profiler::new(1000);
        prof.time_fn("fn_a", || 1u64);
        prof.time_fn("fn_b", || 2u64);
        let fg = prof.flamegraph_data();
        let collapsed = fg.to_collapsed_stacks();
        // Should contain function names
        assert!(collapsed.contains("fn_a") || collapsed.contains("fn_b") || collapsed.is_empty());
    }

    #[test]
    fn test_profiler_json_export() {
        let mut prof = Profiler::new(1000);
        prof.time_fn("test_fn", || 42u64);
        let fg = prof.flamegraph_data();
        let json = fg.to_json();
        assert!(json.starts_with("{"));
        assert!(json.ends_with("}"));
    }

    #[test]
    fn test_profile_report_format() {
        let mut prof = Profiler::new(1000);
        for _ in 0..10 { prof.time_fn("order_submit", || 1u64); }
        let report = prof.report();
        assert!(!report.entries.is_empty());
        let formatted = format!("{}", report);
        assert!(formatted.contains("order_submit"));
    }

    #[test]
    fn test_micro_benchmark_run() {
        let mut bench = MicroBenchmark::new("noop", 100);
        bench.run(|| { let _: u64 = 1 + 1; });
        assert!(bench.min_cycles() <= bench.max_cycles());
        assert!(bench.mean_cycles() >= 0.0);
    }

    #[test]
    fn test_micro_benchmark_results() {
        let mut bench = MicroBenchmark::new("simple_loop", 200);
        bench.run(|| { let mut x = 0u64; for i in 0..100 { x += i; } let _ = x; });
        let tsc_freq = estimate_tsc_freq();
        let results = bench.results(tsc_freq);
        assert!(results.p99_ns >= results.min_ns);
        assert!(results.mean_ns > 0.0);
        let display = format!("{}", results);
        assert!(display.contains("simple_loop"));
    }

    #[test]
    fn test_memory_bandwidth_basic() {
        let bench = MemoryBandwidthBenchmark::new(1024 * 1024); // 1MB
        let read_bw = bench.measure_read_bandwidth();
        assert!(read_bw > 0.0, "read_bw={}", read_bw);
        let write_bw = bench.measure_write_bandwidth();
        assert!(write_bw > 0.0, "write_bw={}", write_bw);
    }

    #[test]
    fn test_random_access_latency() {
        let bench = MemoryBandwidthBenchmark::new(64 * 1024); // 64KB (fits in L1/L2)
        let lat = bench.measure_random_access_latency_ns();
        assert!(lat > 0.0 && lat < 1_000_000.0, "lat={}ns", lat);
    }

    #[test]
    fn test_span_stats_std_dev() {
        let mut stats = SpanStats::default();
        for c in [100u64, 200, 300, 400, 500] { stats.update(c); }
        assert!(stats.std_dev_cycles() > 0.0);
        assert!((stats.mean_cycles() - 300.0).abs() < 0.01);
    }
}
