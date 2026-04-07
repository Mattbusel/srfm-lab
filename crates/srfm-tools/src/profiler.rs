// profiler.rs -- execution profiling utilities for SRFM
// Records timing samples, computes percentile summaries, exports flamegraph and CSV.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ── Core sample type ───────────────────────────────────────────────────────

/// A single timing sample for a labelled code section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSample {
    /// Label identifying the code section or function.
    pub label: String,
    /// Nanosecond timestamp at the start of the measured section.
    pub start_ns: u64,
    /// Nanosecond timestamp at the end of the measured section.
    pub end_ns: u64,
    /// Arbitrary numeric metadata attached to this sample (e.g. input size).
    pub metadata: HashMap<String, f64>,
}

impl ProfileSample {
    /// Create a new sample with the given start/end times.
    pub fn new(label: impl Into<String>, start_ns: u64, end_ns: u64) -> Self {
        Self {
            label: label.into(),
            start_ns,
            end_ns,
            metadata: HashMap::new(),
        }
    }

    /// Duration of this sample in nanoseconds.
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }

    /// Attach a metadata key/value pair.
    pub fn with_meta(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

// ── Per-label summary ──────────────────────────────────────────────────────

/// Aggregated statistics for one label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSummary {
    pub label: String,
    pub call_count: u64,
    pub total_ns: u64,
    pub mean_ns: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub p50_ns: f64,
    pub p95_ns: f64,
    pub p99_ns: f64,
    pub std_ns: f64,
}

impl LabelSummary {
    fn from_durations(label: &str, mut durations: Vec<u64>) -> Self {
        durations.sort_unstable();
        let n = durations.len() as u64;
        let total_ns: u64 = durations.iter().sum();
        let mean_ns = total_ns as f64 / n as f64;

        let p50_ns = percentile_ns(&durations, 50.0);
        let p95_ns = percentile_ns(&durations, 95.0);
        let p99_ns = percentile_ns(&durations, 99.0);

        let variance = durations
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean_ns;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let std_ns = variance.sqrt();

        Self {
            label: label.to_string(),
            call_count: n,
            total_ns,
            mean_ns,
            min_ns: *durations.first().unwrap_or(&0),
            max_ns: *durations.last().unwrap_or(&0),
            p50_ns,
            p95_ns,
            p99_ns,
            std_ns,
        }
    }
}

fn percentile_ns(sorted: &[u64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)] as f64
}

// ── Full profile summary ───────────────────────────────────────────────────

/// Aggregated summary across all labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSummary {
    /// Per-label statistics, sorted by total time descending.
    pub labels: Vec<LabelSummary>,
    /// Total wall time recorded across all samples (sum of durations), nanoseconds.
    pub total_wall_ns: u64,
    /// Number of distinct labels seen.
    pub distinct_labels: usize,
    /// Total sample count.
    pub total_samples: usize,
}

impl ProfileSummary {
    /// Return the label with the highest total time.
    pub fn hottest_label(&self) -> Option<&LabelSummary> {
        self.labels.first()
    }

    /// Render a simple ASCII table.
    pub fn display_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "\n{:<40} {:>10} {:>12} {:>12} {:>12} {:>12}\n",
            "Label", "Calls", "Mean(us)", "P95(us)", "P99(us)", "Total(ms)"
        ));
        out.push_str(&"-".repeat(100));
        out.push('\n');
        for s in &self.labels {
            out.push_str(&format!(
                "{:<40} {:>10} {:>12.2} {:>12.2} {:>12.2} {:>12.3}\n",
                s.label,
                s.call_count,
                s.mean_ns / 1_000.0,
                s.p95_ns / 1_000.0,
                s.p99_ns / 1_000.0,
                s.total_ns as f64 / 1_000_000.0,
            ));
        }
        out
    }
}

// ── ExecutionProfiler ──────────────────────────────────────────────────────

/// Central profiler that accumulates timing samples across the process lifetime.
#[derive(Debug, Default, Clone)]
pub struct ExecutionProfiler {
    /// Raw samples in insertion order.
    pub samples: Vec<ProfileSample>,
    /// Duration-only fast-path: maps label -> list of duration_ns values.
    pub labels: HashMap<String, Vec<u64>>,
}

impl ExecutionProfiler {
    /// Create a new empty profiler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed sample.
    pub fn record_sample(&mut self, sample: ProfileSample) {
        let dur = sample.duration_ns();
        self.labels
            .entry(sample.label.clone())
            .or_default()
            .push(dur);
        self.samples.push(sample);
    }

    /// Record a raw duration for a label (fast path -- no sample stored).
    pub fn record(&mut self, label: &str, duration_ns: u64) {
        self.labels.entry(label.to_string()).or_default().push(duration_ns);
    }

    /// Time a closure and record the result under `label`.
    pub fn time<F, R>(&mut self, label: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let t0 = Instant::now();
        let result = f();
        let elapsed_ns = t0.elapsed().as_nanos() as u64;
        self.record(label, elapsed_ns);
        result
    }

    /// Compute aggregated summary across all recorded labels.
    pub fn summary(&self) -> ProfileSummary {
        let mut label_summaries: Vec<LabelSummary> = self
            .labels
            .iter()
            .map(|(label, durations)| LabelSummary::from_durations(label, durations.clone()))
            .collect();

        // Sort by total_ns descending so hottest functions appear first.
        label_summaries.sort_by(|a, b| b.total_ns.cmp(&a.total_ns));

        let total_wall_ns = label_summaries.iter().map(|s| s.total_ns).sum();
        let distinct_labels = label_summaries.len();
        let total_samples = label_summaries.iter().map(|s| s.call_count as usize).sum();

        ProfileSummary {
            labels: label_summaries,
            total_wall_ns,
            distinct_labels,
            total_samples,
        }
    }

    /// Clear all recorded data.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.labels.clear();
    }

    /// Export in flamegraph folded-stack format.
    /// Each line is: `label;label count` where count is the total duration in ns.
    /// Compatible with Brendan Gregg's flamegraph.pl and inferno.
    pub fn to_flamegraph_folded(&self) -> String {
        let summary = self.summary();
        summary
            .labels
            .iter()
            .map(|s| format!("{} {}", s.label, s.total_ns))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Export all label summaries as CSV.
    pub fn to_csv(&self) -> String {
        let mut out =
            "label,call_count,mean_ns,p50_ns,p95_ns,p99_ns,min_ns,max_ns,std_ns,total_ns\n"
                .to_string();
        let summary = self.summary();
        for s in &summary.labels {
            out.push_str(&format!(
                "{},{},{:.2},{:.2},{:.2},{:.2},{},{},{:.2},{}\n",
                s.label,
                s.call_count,
                s.mean_ns,
                s.p50_ns,
                s.p95_ns,
                s.p99_ns,
                s.min_ns,
                s.max_ns,
                s.std_ns,
                s.total_ns,
            ));
        }
        out
    }

    /// Export raw samples as CSV for offline analysis.
    pub fn raw_samples_csv(&self) -> String {
        let mut out = "label,start_ns,end_ns,duration_ns\n".to_string();
        for s in &self.samples {
            out.push_str(&format!(
                "{},{},{},{}\n",
                s.label,
                s.start_ns,
                s.end_ns,
                s.duration_ns()
            ));
        }
        out
    }

    /// Merge another profiler's data into this one.
    pub fn merge(&mut self, other: &ExecutionProfiler) {
        for (label, durations) in &other.labels {
            let entry = self.labels.entry(label.clone()).or_default();
            entry.extend_from_slice(durations);
        }
        self.samples.extend(other.samples.iter().cloned());
    }

    /// Return labels that exceeded `threshold_ns` for at least one sample.
    pub fn hot_paths(&self, threshold_ns: u64) -> Vec<String> {
        self.labels
            .iter()
            .filter(|(_, durations)| durations.iter().any(|&d| d >= threshold_ns))
            .map(|(label, _)| label.clone())
            .collect()
    }
}

// ── Macro: profile_fn! ─────────────────────────────────────────────────────

/// Time an expression and print the elapsed time to stderr.
/// Returns the value produced by the expression.
///
/// Usage:
///   let result = profile_fn!("my_label", some_expensive_fn(arg));
#[macro_export]
macro_rules! profile_fn {
    ($label:expr, $expr:expr) => {{
        let _pf_t0 = std::time::Instant::now();
        let _pf_result = $expr;
        let _pf_elapsed = _pf_t0.elapsed().as_nanos();
        eprintln!("[profile] {} = {}ns", $label, _pf_elapsed);
        _pf_result
    }};
}

/// Time an expression, recording into a provided `ExecutionProfiler`.
///
/// Usage:
///   let result = profile_into!(profiler, "my_label", some_fn());
#[macro_export]
macro_rules! profile_into {
    ($profiler:expr, $label:expr, $expr:expr) => {{
        let _pi_t0 = std::time::Instant::now();
        let _pi_result = $expr;
        let _pi_elapsed = _pi_t0.elapsed().as_nanos() as u64;
        $profiler.record($label, _pi_elapsed);
        _pi_result
    }};
}

// ── Scoped timer ───────────────────────────────────────────────────────────

/// RAII timer that records elapsed time into a profiler on drop.
pub struct ScopedTimer<'a> {
    label: String,
    start: Instant,
    profiler: &'a mut ExecutionProfiler,
}

impl<'a> ScopedTimer<'a> {
    pub fn new(profiler: &'a mut ExecutionProfiler, label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
            profiler,
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        let elapsed_ns = self.start.elapsed().as_nanos() as u64;
        self.profiler.record(&self.label, elapsed_ns);
    }
}

// ── Thread-local global profiler ───────────────────────────────────────────

use std::cell::RefCell;

thread_local! {
    static GLOBAL_PROFILER: RefCell<ExecutionProfiler> = RefCell::new(ExecutionProfiler::new());
}

/// Record into the thread-local global profiler.
pub fn global_record(label: &str, duration_ns: u64) {
    GLOBAL_PROFILER.with(|p| p.borrow_mut().record(label, duration_ns));
}

/// Extract a summary from the thread-local global profiler.
pub fn global_summary() -> ProfileSummary {
    GLOBAL_PROFILER.with(|p| p.borrow().summary())
}

/// Reset the thread-local global profiler.
pub fn global_reset() {
    GLOBAL_PROFILER.with(|p| p.borrow_mut().reset());
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profile_sample_duration() {
        let s = ProfileSample::new("test", 1000, 5000);
        assert_eq!(s.duration_ns(), 4000);
    }

    #[test]
    fn test_profile_sample_metadata() {
        let s = ProfileSample::new("test", 0, 100)
            .with_meta("size", 1024.0)
            .with_meta("depth", 3.0);
        assert_eq!(s.metadata["size"], 1024.0);
        assert_eq!(s.metadata["depth"], 3.0);
    }

    #[test]
    fn test_record_and_summary() {
        let mut p = ExecutionProfiler::new();
        for i in 0..10u64 {
            p.record("alpha", i * 100);
        }
        let summary = p.summary();
        assert_eq!(summary.distinct_labels, 1);
        assert_eq!(summary.total_samples, 10);
        let label = &summary.labels[0];
        assert_eq!(label.call_count, 10);
        assert_eq!(label.min_ns, 0);
        assert_eq!(label.max_ns, 900);
    }

    #[test]
    fn test_multiple_labels_sorted_by_total() {
        let mut p = ExecutionProfiler::new();
        // "heavy" gets much more total time than "light"
        p.record("light", 10);
        p.record("heavy", 100_000);
        p.record("heavy", 200_000);
        let summary = p.summary();
        assert_eq!(summary.labels[0].label, "heavy");
        assert_eq!(summary.labels[1].label, "light");
    }

    #[test]
    fn test_time_closure() {
        let mut p = ExecutionProfiler::new();
        let result = p.time("sleep_test", || {
            // Light spin to avoid zero-duration on fast machines.
            let mut x = 0u64;
            for i in 0..1000u64 {
                x = x.wrapping_add(i);
            }
            x
        });
        assert!(result > 0);
        let summary = p.summary();
        assert_eq!(summary.labels[0].label, "sleep_test");
        assert!(summary.labels[0].total_ns > 0);
    }

    #[test]
    fn test_csv_export_header() {
        let mut p = ExecutionProfiler::new();
        p.record("fn_a", 500);
        let csv = p.to_csv();
        assert!(csv.starts_with("label,call_count,mean_ns"));
        assert!(csv.contains("fn_a"));
    }

    #[test]
    fn test_flamegraph_folded_format() {
        let mut p = ExecutionProfiler::new();
        p.record("foo", 1000);
        p.record("bar", 2000);
        let folded = p.to_flamegraph_folded();
        // Each line should have a space separating label from count.
        for line in folded.lines() {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            assert_eq!(parts.len(), 2, "malformed line: {}", line);
            parts[1].parse::<u64>().expect("count should be integer");
        }
    }

    #[test]
    fn test_merge_profilers() {
        let mut p1 = ExecutionProfiler::new();
        p1.record("shared", 100);
        let mut p2 = ExecutionProfiler::new();
        p2.record("shared", 200);
        p2.record("exclusive", 50);
        p1.merge(&p2);
        let summary = p1.summary();
        let shared = summary.labels.iter().find(|s| s.label == "shared").unwrap();
        assert_eq!(shared.call_count, 2);
        assert!(summary.labels.iter().any(|s| s.label == "exclusive"));
    }

    #[test]
    fn test_reset() {
        let mut p = ExecutionProfiler::new();
        p.record("x", 100);
        p.reset();
        let summary = p.summary();
        assert_eq!(summary.total_samples, 0);
    }

    #[test]
    fn test_hot_paths() {
        let mut p = ExecutionProfiler::new();
        p.record("fast", 100);
        p.record("slow", 10_000_000);
        let hot = p.hot_paths(1_000_000);
        assert!(hot.contains(&"slow".to_string()));
        assert!(!hot.contains(&"fast".to_string()));
    }

    #[test]
    fn test_percentile_calculation() {
        let mut p = ExecutionProfiler::new();
        // 100 samples: 1, 2, ..., 100 (in ns)
        for i in 1u64..=100 {
            p.record("percentile_test", i);
        }
        let summary = p.summary();
        let label = &summary.labels[0];
        // p50 of [1..100] should be around 50.
        assert!(label.p50_ns >= 49.0 && label.p50_ns <= 51.0);
        // p99 should be near 99.
        assert!(label.p99_ns >= 98.0 && label.p99_ns <= 100.0);
    }

    #[test]
    fn test_global_profiler_thread_local() {
        // Verify the global profiler works on the current thread.
        global_reset();
        global_record("global_fn", 12345);
        let summary = global_summary();
        assert_eq!(summary.total_samples, 1);
        assert_eq!(summary.labels[0].label, "global_fn");
        global_reset();
    }

    #[test]
    fn test_profile_into_macro() {
        let mut p = ExecutionProfiler::new();
        let val = profile_into!(p, "macro_test", 2u64 + 2u64);
        assert_eq!(val, 4);
        let summary = p.summary();
        assert!(summary.labels.iter().any(|s| s.label == "macro_test"));
    }

    #[test]
    fn test_scoped_timer() {
        let mut p = ExecutionProfiler::new();
        {
            let _timer = ScopedTimer::new(&mut p, "scoped");
            // Some work.
            let mut _x = 0u64;
            for i in 0..500u64 {
                _x = _x.wrapping_add(i);
            }
        }
        let summary = p.summary();
        assert_eq!(summary.labels[0].label, "scoped");
        assert!(summary.labels[0].total_ns > 0);
    }

    #[test]
    fn test_raw_samples_csv() {
        let mut p = ExecutionProfiler::new();
        p.record_sample(ProfileSample::new("fn_x", 1000, 2000));
        let csv = p.raw_samples_csv();
        assert!(csv.contains("fn_x"));
        assert!(csv.contains("1000"));
        assert!(csv.contains("2000"));
        assert!(csv.contains("1000")); // duration
    }

    #[test]
    fn test_display_table_output() {
        let mut p = ExecutionProfiler::new();
        p.record("alpha", 5_000);
        p.record("alpha", 10_000);
        let summary = p.summary();
        let table = summary.display_table();
        assert!(table.contains("alpha"));
        assert!(table.contains("Calls"));
    }
}
