// benchmark_runner.rs -- micro-benchmark utilities for SRFM
// Criterion-style warmup + measurement, baseline comparison, markdown output.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

// ── Core types ─────────────────────────────────────────────────────────────

/// A single benchmark function with its run configuration.
pub struct Benchmark {
    pub name: String,
    pub iterations: u64,
    pub warmup: u64,
    pub fn_ptr: Box<dyn Fn() -> f64 + Send + Sync>,
}

impl Benchmark {
    /// Create a new benchmark with default iteration counts.
    pub fn new(name: impl Into<String>, f: impl Fn() -> f64 + Send + Sync + 'static) -> Self {
        Self {
            name: name.into(),
            iterations: 1000,
            warmup: 100,
            fn_ptr: Box::new(f),
        }
    }

    /// Override iteration count.
    pub fn with_iterations(mut self, n: u64) -> Self {
        self.iterations = n;
        self
    }

    /// Override warmup count.
    pub fn with_warmup(mut self, n: u64) -> Self {
        self.warmup = n;
        self
    }
}

/// A suite of related benchmarks.
pub struct BenchmarkSuite {
    pub name: String,
    pub benchmarks: Vec<Benchmark>,
}

impl BenchmarkSuite {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), benchmarks: vec![] }
    }

    pub fn add(mut self, bench: Benchmark) -> Self {
        self.benchmarks.push(bench);
        self
    }
}

/// Aggregated statistics from one benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub mean_ns: f64,
    pub std_ns: f64,
    pub min_ns: f64,
    pub max_ns: f64,
    pub median_ns: f64,
    pub p95_ns: f64,
    pub iterations: u64,
    pub throughput_per_sec: f64,
}

impl BenchmarkResult {
    fn from_samples(name: &str, mut samples_ns: Vec<f64>, iterations: u64) -> Self {
        samples_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = samples_ns.len() as f64;
        let mean_ns = samples_ns.iter().sum::<f64>() / n;
        let variance = samples_ns
            .iter()
            .map(|&x| (x - mean_ns).powi(2))
            .sum::<f64>()
            / n;
        let std_ns = variance.sqrt();
        let min_ns = samples_ns.first().copied().unwrap_or(0.0);
        let max_ns = samples_ns.last().copied().unwrap_or(0.0);
        let median_ns = percentile_f64(&samples_ns, 50.0);
        let p95_ns = percentile_f64(&samples_ns, 95.0);
        // Throughput: how many calls per second if each call takes mean_ns.
        let throughput_per_sec = if mean_ns > 0.0 {
            1_000_000_000.0 / mean_ns
        } else {
            f64::INFINITY
        };

        Self {
            name: name.to_string(),
            mean_ns,
            std_ns,
            min_ns,
            max_ns,
            median_ns,
            p95_ns,
            iterations,
            throughput_per_sec,
        }
    }
}

fn percentile_f64(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Runner ─────────────────────────────────────────────────────────────────

/// Run a single benchmark and return its result.
pub fn run_benchmark(bench: &Benchmark) -> BenchmarkResult {
    // Warmup phase: run fn but discard results to heat up caches/branch predictors.
    for _ in 0..bench.warmup {
        let _ = std::hint::black_box((bench.fn_ptr)());
    }

    // Measurement phase.
    let mut samples_ns: Vec<f64> = Vec::with_capacity(bench.iterations as usize);
    for _ in 0..bench.iterations {
        let t0 = Instant::now();
        let _ = std::hint::black_box((bench.fn_ptr)());
        let elapsed = t0.elapsed().as_nanos() as f64;
        samples_ns.push(elapsed);
    }

    BenchmarkResult::from_samples(&bench.name, samples_ns, bench.iterations)
}

/// Run an entire suite and return all results.
pub fn run_suite(suite: &BenchmarkSuite) -> Vec<BenchmarkResult> {
    suite.benchmarks.iter().map(run_benchmark).collect()
}

// ── Baseline comparison ────────────────────────────────────────────────────

/// A stored baseline entry for one benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineEntry {
    pub name: String,
    pub mean_ns: f64,
    pub std_ns: f64,
}

/// Comparison against a stored baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionCheck {
    pub name: String,
    pub baseline_mean_ns: f64,
    pub current_mean_ns: f64,
    /// Fractional change: positive = slower, negative = faster.
    pub delta_pct: f64,
    /// True if the change exceeds the regression threshold.
    pub is_regression: bool,
}

impl RegressionCheck {
    pub fn new(baseline: &BaselineEntry, current: &BenchmarkResult, threshold_pct: f64) -> Self {
        let delta_pct = (current.mean_ns - baseline.mean_ns) / baseline.mean_ns * 100.0;
        let is_regression = delta_pct > threshold_pct;
        Self {
            name: current.name.clone(),
            baseline_mean_ns: baseline.mean_ns,
            current_mean_ns: current.mean_ns,
            delta_pct,
            is_regression,
        }
    }
}

/// Load a baseline from a JSON file.
pub fn load_baseline(path: &Path) -> Result<HashMap<String, BaselineEntry>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read baseline: {e}"))?;
    let entries: Vec<BaselineEntry> = serde_json::from_str(&content)
        .map_err(|e| format!("Cannot parse baseline JSON: {e}"))?;
    Ok(entries.into_iter().map(|e| (e.name.clone(), e)).collect())
}

/// Save current results as a new baseline JSON file.
pub fn save_baseline(results: &[BenchmarkResult], path: &Path) -> Result<(), String> {
    let entries: Vec<BaselineEntry> = results
        .iter()
        .map(|r| BaselineEntry {
            name: r.name.clone(),
            mean_ns: r.mean_ns,
            std_ns: r.std_ns,
        })
        .collect();
    let json = serde_json::to_string_pretty(&entries)
        .map_err(|e| format!("Serialize error: {e}"))?;
    std::fs::write(path, json).map_err(|e| format!("Write error: {e}"))
}

/// Compare results against a baseline and report regressions.
/// Returns a list of regression checks; items with `is_regression = true`
/// exceeded the threshold.
pub fn check_regressions(
    results: &[BenchmarkResult],
    baseline: &HashMap<String, BaselineEntry>,
    threshold_pct: f64,
) -> Vec<RegressionCheck> {
    results
        .iter()
        .filter_map(|r| {
            baseline
                .get(&r.name)
                .map(|b| RegressionCheck::new(b, r, threshold_pct))
        })
        .collect()
}

// ── Output formatters ──────────────────────────────────────────────────────

/// Render results as a markdown table suitable for PR comments.
pub fn to_markdown_table(results: &[BenchmarkResult]) -> String {
    let mut out = String::new();
    out.push_str("| Benchmark | Mean (us) | Std (us) | Min (us) | Max (us) | P95 (us) | Throughput/s |\n");
    out.push_str("|-----------|----------:|---------:|---------:|---------:|---------:|-------------:|\n");
    for r in results {
        out.push_str(&format!(
            "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.0} |\n",
            r.name,
            r.mean_ns / 1_000.0,
            r.std_ns / 1_000.0,
            r.min_ns / 1_000.0,
            r.max_ns / 1_000.0,
            r.p95_ns / 1_000.0,
            r.throughput_per_sec,
        ));
    }
    out
}

/// Render regression checks as a markdown table.
pub fn regressions_to_markdown(checks: &[RegressionCheck]) -> String {
    let mut out = String::new();
    out.push_str("## Benchmark Regression Report\n\n");

    let regressions: Vec<&RegressionCheck> = checks.iter().filter(|c| c.is_regression).collect();
    if regressions.is_empty() {
        out.push_str("**No regressions detected.**\n");
    } else {
        out.push_str(&format!(
            "**{} regression(s) detected!**\n\n",
            regressions.len()
        ));
    }

    out.push('\n');
    out.push_str("| Benchmark | Baseline (us) | Current (us) | Delta% | Status |\n");
    out.push_str("|-----------|-------------:|-------------:|-------:|--------|\n");

    for c in checks {
        let status = if c.is_regression { "REGRESS" } else { "OK" };
        out.push_str(&format!(
            "| {} | {:.2} | {:.2} | {:.1}% | {} |\n",
            c.name,
            c.baseline_mean_ns / 1_000.0,
            c.current_mean_ns / 1_000.0,
            c.delta_pct,
            status,
        ));
    }
    out
}

/// Render results as a plain-text summary for terminal output.
pub fn to_text_summary(results: &[BenchmarkResult], suite_name: &str) -> String {
    let mut out = format!("\n=== Benchmark Suite: {} ===\n", suite_name);
    out.push_str(&format!(
        "{:<40} {:>12} {:>12} {:>12} {:>14}\n",
        "Benchmark", "Mean(us)", "P95(us)", "Std(us)", "Throughput/s"
    ));
    out.push_str(&"-".repeat(92));
    out.push('\n');
    for r in results {
        out.push_str(&format!(
            "{:<40} {:>12.3} {:>12.3} {:>12.3} {:>14.0}\n",
            r.name,
            r.mean_ns / 1_000.0,
            r.p95_ns / 1_000.0,
            r.std_ns / 1_000.0,
            r.throughput_per_sec,
        ));
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn noop_bench(name: &str) -> Benchmark {
        let n = name.to_string();
        Benchmark::new(n, || {
            let mut x = 0.0f64;
            for i in 0..100 {
                x += i as f64;
            }
            x
        })
        .with_iterations(50)
        .with_warmup(10)
    }

    #[test]
    fn test_run_benchmark_produces_result() {
        let bench = noop_bench("test_noop");
        let result = run_benchmark(&bench);
        assert_eq!(result.name, "test_noop");
        assert_eq!(result.iterations, 50);
        assert!(result.mean_ns > 0.0);
        assert!(result.min_ns <= result.mean_ns);
        assert!(result.mean_ns <= result.max_ns);
    }

    #[test]
    fn test_throughput_inversely_proportional_to_mean() {
        let bench = noop_bench("throughput_test");
        let result = run_benchmark(&bench);
        // throughput = 1e9 / mean_ns
        let expected = 1_000_000_000.0 / result.mean_ns;
        let diff = (result.throughput_per_sec - expected).abs();
        assert!(diff < 1.0, "throughput mismatch: expected ~{expected}, got {}", result.throughput_per_sec);
    }

    #[test]
    fn test_run_suite() {
        let suite = BenchmarkSuite::new("test_suite")
            .add(noop_bench("bench_a"))
            .add(noop_bench("bench_b"));
        let results = run_suite(&suite);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].name, "bench_a");
        assert_eq!(results[1].name, "bench_b");
    }

    #[test]
    fn test_markdown_table_format() {
        let results = vec![BenchmarkResult {
            name: "foo".to_string(),
            mean_ns: 5000.0,
            std_ns: 100.0,
            min_ns: 4800.0,
            max_ns: 6000.0,
            median_ns: 4950.0,
            p95_ns: 5800.0,
            iterations: 1000,
            throughput_per_sec: 200_000.0,
        }];
        let md = to_markdown_table(&results);
        assert!(md.contains("| Benchmark |"));
        assert!(md.contains("foo"));
        assert!(md.contains("5.00")); // mean_ns/1000
    }

    #[test]
    fn test_regression_detected() {
        let baseline = BaselineEntry {
            name: "fn_x".to_string(),
            mean_ns: 1000.0,
            std_ns: 50.0,
        };
        let current = BenchmarkResult {
            name: "fn_x".to_string(),
            mean_ns: 1200.0, // 20% slower
            std_ns: 60.0,
            min_ns: 1100.0,
            max_ns: 1400.0,
            median_ns: 1190.0,
            p95_ns: 1380.0,
            iterations: 1000,
            throughput_per_sec: 833_333.0,
        };
        let check = RegressionCheck::new(&baseline, &current, 10.0);
        assert!(check.is_regression);
        assert!((check.delta_pct - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_no_regression_within_threshold() {
        let baseline = BaselineEntry {
            name: "fn_y".to_string(),
            mean_ns: 1000.0,
            std_ns: 50.0,
        };
        let current = BenchmarkResult {
            name: "fn_y".to_string(),
            mean_ns: 1050.0, // 5% slower -- within 10% threshold
            std_ns: 55.0,
            min_ns: 950.0,
            max_ns: 1100.0,
            median_ns: 1045.0,
            p95_ns: 1090.0,
            iterations: 1000,
            throughput_per_sec: 952_381.0,
        };
        let check = RegressionCheck::new(&baseline, &current, 10.0);
        assert!(!check.is_regression);
    }

    #[test]
    fn test_save_and_load_baseline() {
        let results = vec![
            BenchmarkResult {
                name: "bench_1".to_string(),
                mean_ns: 100.0,
                std_ns: 5.0,
                min_ns: 90.0,
                max_ns: 120.0,
                median_ns: 99.0,
                p95_ns: 115.0,
                iterations: 500,
                throughput_per_sec: 10_000_000.0,
            },
        ];
        let tmp = NamedTempFile::new().expect("temp");
        save_baseline(&results, tmp.path()).expect("save");
        let loaded = load_baseline(tmp.path()).expect("load");
        let entry = loaded.get("bench_1").expect("key");
        assert!((entry.mean_ns - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_check_regressions_missing_baseline() {
        // If a benchmark has no baseline entry, it is omitted from checks.
        let results = vec![BenchmarkResult {
            name: "new_bench".to_string(),
            mean_ns: 500.0,
            std_ns: 20.0,
            min_ns: 480.0,
            max_ns: 550.0,
            median_ns: 495.0,
            p95_ns: 540.0,
            iterations: 100,
            throughput_per_sec: 2_000_000.0,
        }];
        let baseline: HashMap<String, BaselineEntry> = HashMap::new();
        let checks = check_regressions(&results, &baseline, 10.0);
        assert!(checks.is_empty());
    }

    #[test]
    fn test_regressions_to_markdown_no_regressions() {
        let checks = vec![RegressionCheck {
            name: "ok_bench".to_string(),
            baseline_mean_ns: 1000.0,
            current_mean_ns: 980.0,
            delta_pct: -2.0,
            is_regression: false,
        }];
        let md = regressions_to_markdown(&checks);
        assert!(md.contains("No regressions detected"));
        assert!(md.contains("ok_bench"));
    }

    #[test]
    fn test_regressions_to_markdown_with_regressions() {
        let checks = vec![RegressionCheck {
            name: "slow_bench".to_string(),
            baseline_mean_ns: 1000.0,
            current_mean_ns: 1500.0,
            delta_pct: 50.0,
            is_regression: true,
        }];
        let md = regressions_to_markdown(&checks);
        assert!(md.contains("1 regression"));
        assert!(md.contains("REGRESS"));
    }

    #[test]
    fn test_percentile_sorted() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile_f64(&sorted, 0.0), 1.0);
        assert_eq!(percentile_f64(&sorted, 100.0), 10.0);
        // p50 of 10 items -> index 5 (rounded) -> value 6.0
        let p50 = percentile_f64(&sorted, 50.0);
        assert!(p50 >= 5.0 && p50 <= 6.0);
    }

    #[test]
    fn test_benchmark_result_from_samples_statistics() {
        let samples: Vec<f64> = (1..=100).map(|i| i as f64 * 10.0).collect(); // 10..1000 ns
        let result = BenchmarkResult::from_samples("stat_test", samples, 100);
        assert_eq!(result.min_ns, 10.0);
        assert_eq!(result.max_ns, 1000.0);
        assert!((result.mean_ns - 505.0).abs() < 1.0);
        assert!(result.std_ns > 0.0);
    }

    #[test]
    fn test_to_text_summary() {
        let results = vec![BenchmarkResult {
            name: "summary_bench".to_string(),
            mean_ns: 2500.0,
            std_ns: 200.0,
            min_ns: 2200.0,
            max_ns: 3100.0,
            median_ns: 2490.0,
            p95_ns: 3000.0,
            iterations: 200,
            throughput_per_sec: 400_000.0,
        }];
        let txt = to_text_summary(&results, "MySuite");
        assert!(txt.contains("MySuite"));
        assert!(txt.contains("summary_bench"));
    }
}
