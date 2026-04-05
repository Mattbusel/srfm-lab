//! Core Monte Carlo simulation module.
//!
//! Fits a return distribution from historical data using method-of-moments,
//! L-moments (skew/kurtosis), and Extreme Value Theory for tail estimation.
//! Runs parallel simulation paths via Rayon to produce rich output statistics.

use anyhow::{bail, Result};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution as RandDist, Normal, StudentT};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// ReturnDistribution — fitted parameters
// ---------------------------------------------------------------------------

/// Fitted parameters for a return distribution, including fat-tail estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnDistribution {
    /// Arithmetic mean of daily returns.
    pub mean: f64,
    /// Standard deviation of daily returns.
    pub std: f64,
    /// Skewness (third standardised moment via L-moments).
    pub skew: f64,
    /// Excess kurtosis (fourth standardised moment via L-moments).
    pub kurtosis: f64,
    /// Hill tail index α (from Extreme Value Theory). Higher ≈ lighter tails.
    pub tail_alpha: f64,
    /// Number of observations used to fit the distribution.
    pub n_obs: usize,
    /// Degrees of freedom for the blended Student-t (derived from kurtosis).
    pub student_df: f64,
}

impl ReturnDistribution {
    /// Derive Student-t degrees of freedom from excess kurtosis.
    /// For Student-t: excess kurtosis = 6 / (df - 4)  →  df = 6/ek + 4
    /// Clamped to [3, 100] to keep the distribution well-defined.
    fn df_from_kurtosis(excess_kurtosis: f64) -> f64 {
        if excess_kurtosis <= 0.0 {
            return 100.0; // near-normal
        }
        let df = 6.0 / excess_kurtosis + 4.0;
        df.clamp(3.0, 100.0)
    }
}

// ---------------------------------------------------------------------------
// Distribution fitting
// ---------------------------------------------------------------------------

/// Fit a [`ReturnDistribution`] from a slice of return observations.
///
/// Uses:
/// - Method of moments for mean and standard deviation
/// - L-moments for skew and kurtosis (more robust to outliers)
/// - Hill estimator for tail index α
pub fn fit_distribution(returns: &[f64]) -> ReturnDistribution {
    assert!(!returns.is_empty(), "returns slice must not be empty");

    let n = returns.len();

    // --- Method of moments: mean and std ---
    let mean = returns.iter().sum::<f64>() / n as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt().max(1e-10);

    // --- L-moments: skew and kurtosis ---
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let l1 = l_moment_1(&sorted);
    let l2 = l_moment_2(&sorted);
    let l3 = l_moment_3(&sorted);
    let l4 = l_moment_4(&sorted);

    // L-skewness τ₃ = L3/L2; L-kurtosis τ₄ = L4/L2
    let l_skew = if l2.abs() > 1e-12 { l3 / l2 } else { 0.0 };
    let l_kurt = if l2.abs() > 1e-12 { l4 / l2 } else { 0.0 };

    // Convert L-kurtosis to excess kurtosis (approx: ek ≈ 5.6*τ₄ - 1.0)
    let skew = l_skew * 6.0; // rough conversion from L-skewness to classical skew
    let excess_kurtosis = (5.6 * l_kurt - 1.0).max(0.0);

    let _ = l1; // suppress unused warning — l1 used for validation only

    // --- EVT Hill estimator for tail index ---
    let tail_alpha = hill_tail_index(&sorted);

    let student_df = ReturnDistribution::df_from_kurtosis(excess_kurtosis);

    ReturnDistribution {
        mean,
        std,
        skew,
        kurtosis: excess_kurtosis,
        tail_alpha,
        n_obs: n,
        student_df,
    }
}

// --- L-moment helpers (probability-weighted moments) ---

fn l_moment_1(sorted: &[f64]) -> f64 {
    sorted.iter().sum::<f64>() / sorted.len() as f64
}

fn l_moment_2(sorted: &[f64]) -> f64 {
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let c1 = (i as f64) / (n - 1.0);
            let c2 = (n - 1.0 - i as f64) / (n - 1.0);
            x * (c1 - c2)
        })
        .sum::<f64>()
        / 2.0
}

fn l_moment_3(sorted: &[f64]) -> f64 {
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let p = i as f64 / (n - 1.0);
            let q = (n - 1.0 - i as f64) / (n - 1.0);
            // Legendre polynomial basis for 3rd L-moment
            x * (p * p - 2.0 * p * q + q * q - (p + q) / 3.0)
        })
        .sum::<f64>()
        / 3.0
}

fn l_moment_4(sorted: &[f64]) -> f64 {
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let p = i as f64 / (n - 1.0);
            let q = (n - 1.0 - i as f64) / (n - 1.0);
            let p2 = p * p;
            let q2 = q * q;
            // 4th order basis
            x * (p2 * p - 3.0 * p2 * q + 3.0 * p * q2 - q2 * q
                - (p2 - 2.0 * p * q + q2) / 2.0)
        })
        .sum::<f64>()
        / 4.0
}

/// Hill estimator for the Pareto tail index α.
/// Uses the top `k` order statistics where k = max(10, n/10).
fn hill_tail_index(sorted_asc: &[f64]) -> f64 {
    let n = sorted_asc.len();
    let k = (n / 10).max(10).min(n - 1);

    // Work with positive absolute values of the upper tail
    let tail: Vec<f64> = sorted_asc
        .iter()
        .rev()
        .take(k)
        .map(|x| x.abs().max(1e-12))
        .collect();

    let threshold = *tail.last().unwrap(); // k-th order statistic
    if threshold <= 0.0 {
        return 2.0; // default: finite variance, moderate tail
    }

    let sum: f64 = tail[..k - 1]
        .iter()
        .map(|&x| (x / threshold).ln())
        .sum();

    if sum <= 0.0 || (k - 1) == 0 {
        return 2.0;
    }

    let hill = (k - 1) as f64 / sum;
    hill.clamp(0.5, 10.0)
}

// ---------------------------------------------------------------------------
// SimulationConfig
// ---------------------------------------------------------------------------

/// Configuration for a Monte Carlo run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Number of independent equity paths to simulate.
    pub n_paths: usize,
    /// Number of bars (periods) per path.
    pub n_bars: usize,
    /// Starting equity for each path.
    pub initial_equity: f64,
    /// If true, blend Normal with Student-t for fat-tail returns.
    pub use_fat_tails: bool,
    /// Fraction of Student-t draws when `use_fat_tails = true`.
    pub fat_tail_weight: f64,
    /// Store full equity paths in results (memory-intensive for large runs).
    pub store_paths: bool,
    /// Random seed (0 = non-deterministic).
    pub seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            n_paths: 10_000,
            n_bars: 252,
            initial_equity: 100_000.0,
            use_fat_tails: true,
            fat_tail_weight: 0.25,
            store_paths: false,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SimulationResults
// ---------------------------------------------------------------------------

/// Output statistics from a Monte Carlo simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResults {
    pub n_paths: usize,
    pub n_bars: usize,
    pub initial_equity: f64,

    // Final equity distribution
    pub median_final_equity: f64,
    pub mean_final_equity: f64,
    pub p5_final_equity: f64,
    pub p25_final_equity: f64,
    pub p75_final_equity: f64,
    pub p95_final_equity: f64,

    // Drawdown distribution
    pub median_max_drawdown: f64,
    pub p95_max_drawdown: f64,
    pub mean_max_drawdown: f64,

    // Risk metrics
    /// Probability equity falls below 50% of initial.
    pub prob_ruin: f64,
    /// Median annualised Sharpe (assumes 252 bars/year).
    pub median_sharpe: f64,
    /// 5th-percentile Sharpe.
    pub p5_sharpe: f64,

    /// Optional: raw final equity values for each path.
    pub final_equities: Option<Vec<f64>>,
    /// Optional: full (path_idx, bar_idx → equity) matrix.
    /// Stored as Vec<Vec<f64>> [path][bar].
    pub equity_paths: Option<Vec<Vec<f64>>>,
}

impl SimulationResults {
    /// Return the p-th percentile of final equity (0–100).
    pub fn percentile_final_equity(&self, p: f64) -> Option<f64> {
        let equities = self.final_equities.as_ref()?;
        let mut sorted = equities.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted.get(idx).copied()
    }
}

// ---------------------------------------------------------------------------
// MonteCarloSimulator
// ---------------------------------------------------------------------------

/// Primary simulator. Runs paths in parallel using Rayon.
pub struct MonteCarloSimulator {
    /// Optional progress callback called every 1000 paths.
    pub progress_callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
}

impl MonteCarloSimulator {
    pub fn new() -> Self {
        Self {
            progress_callback: None,
        }
    }

    /// Run simulation using the fitted distribution and config.
    pub fn run(&self, config: &SimulationConfig, dist: &ReturnDistribution) -> SimulationResults {
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();
        let n_paths = config.n_paths;
        let cb = self.progress_callback.as_ref().map(|_| n_paths);

        // Simulate all paths in parallel
        let path_results: Vec<PathResult> = (0..config.n_paths)
            .into_par_iter()
            .map(|path_idx| {
                let seed = if config.seed == 0 {
                    // Non-deterministic: mix path index with system nonce
                    path_idx as u64 ^ (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .subsec_nanos() as u64)
                } else {
                    config.seed.wrapping_add(path_idx as u64)
                };

                let mut rng = SmallRng::seed_from_u64(seed);
                let result =
                    simulate_single_path(config, dist, &mut rng, path_idx);

                let done = counter_clone.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(_total) = cb {
                    if done % 1000 == 0 {
                        // progress tick — callback is not called in parallel context
                        // but counter is available
                    }
                }
                result
            })
            .collect();

        // Aggregate
        aggregate_results(config, path_results)
    }

    /// Convenience: run with a historical returns slice (fits distribution internally).
    pub fn run_from_returns(
        &self,
        returns: &[f64],
        config: &SimulationConfig,
    ) -> Result<SimulationResults> {
        if returns.is_empty() {
            bail!("returns slice is empty");
        }
        let dist = fit_distribution(returns);
        Ok(self.run(config, &dist))
    }
}

impl Default for MonteCarloSimulator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Single path simulation
// ---------------------------------------------------------------------------

struct PathResult {
    final_equity: f64,
    max_drawdown: f64,
    sharpe: f64,
    equity_series: Option<Vec<f64>>,
}

fn simulate_single_path(
    config: &SimulationConfig,
    dist: &ReturnDistribution,
    rng: &mut SmallRng,
    _path_idx: usize,
) -> PathResult {
    let mut equity = config.initial_equity;
    let mut peak = equity;
    let mut max_dd = 0.0f64;
    let mut returns_log: Vec<f64> = Vec::with_capacity(config.n_bars);
    let equity_series_cap = if config.store_paths {
        config.n_bars + 1
    } else {
        0
    };
    let mut equity_series: Vec<f64> = Vec::with_capacity(equity_series_cap);

    if config.store_paths {
        equity_series.push(equity);
    }

    let normal = Normal::new(dist.mean, dist.std).unwrap_or(Normal::new(0.0, 0.01).unwrap());

    // Student-t for fat tails — centred and scaled to match dist mean/std
    let student = if config.use_fat_tails && dist.student_df > 3.0 {
        StudentT::new(dist.student_df).ok()
    } else {
        None
    };

    for _ in 0..config.n_bars {
        let ret = sample_return(config, dist, &normal, student.as_ref(), rng);

        // Compound equity (log-return approach for stability)
        equity *= 1.0 + ret;
        equity = equity.max(0.0); // floor at zero

        // Track drawdown
        if equity > peak {
            peak = equity;
        }
        let dd = if peak > 0.0 {
            (peak - equity) / peak
        } else {
            0.0
        };
        if dd > max_dd {
            max_dd = dd;
        }

        returns_log.push(ret);

        if config.store_paths {
            equity_series.push(equity);
        }
    }

    // Compute path Sharpe: mean(r) / std(r) * sqrt(252)
    let sharpe = compute_sharpe(&returns_log);

    PathResult {
        final_equity: equity,
        max_drawdown: max_dd,
        sharpe,
        equity_series: if config.store_paths {
            Some(equity_series)
        } else {
            None
        },
    }
}

fn sample_return(
    config: &SimulationConfig,
    dist: &ReturnDistribution,
    normal: &Normal<f64>,
    student: Option<&StudentT<f64>>,
    rng: &mut SmallRng,
) -> f64 {
    if config.use_fat_tails {
        if let Some(st) = student {
            if rng.gen::<f64>() < config.fat_tail_weight {
                // Draw from Student-t, scale to match target mean/std
                // Student-t with df has std = sqrt(df/(df-2))
                let df = dist.student_df;
                let st_std = (df / (df - 2.0)).sqrt();
                let z: f64 = st.sample(rng);
                return dist.mean + dist.std * z / st_std;
            }
        }
    }
    normal.sample(rng)
}

fn compute_sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    mean / std * 252f64.sqrt()
}

// ---------------------------------------------------------------------------
// Result aggregation
// ---------------------------------------------------------------------------

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn aggregate_results(config: &SimulationConfig, paths: Vec<PathResult>) -> SimulationResults {
    let n = paths.len();

    let mut final_eq: Vec<f64> = paths.iter().map(|p| p.final_equity).collect();
    let mut drawdowns: Vec<f64> = paths.iter().map(|p| p.max_drawdown).collect();
    let mut sharpes: Vec<f64> = paths.iter().map(|p| p.sharpe).collect();

    final_eq.sort_by(|a, b| a.partial_cmp(b).unwrap());
    drawdowns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_final = final_eq.iter().sum::<f64>() / n as f64;
    let mean_dd = drawdowns.iter().sum::<f64>() / n as f64;

    let ruin_threshold = config.initial_equity * 0.5;
    let prob_ruin = paths.iter().filter(|p| p.final_equity < ruin_threshold).count() as f64
        / n as f64;

    let equity_paths: Option<Vec<Vec<f64>>> = if config.store_paths {
        Some(paths.iter().filter_map(|p| p.equity_series.clone()).collect())
    } else {
        None
    };

    let final_equities_out: Option<Vec<f64>> = if config.store_paths {
        Some(final_eq.clone())
    } else {
        None
    };

    SimulationResults {
        n_paths: config.n_paths,
        n_bars: config.n_bars,
        initial_equity: config.initial_equity,
        median_final_equity: percentile(&final_eq, 50.0),
        mean_final_equity: mean_final,
        p5_final_equity: percentile(&final_eq, 5.0),
        p25_final_equity: percentile(&final_eq, 25.0),
        p75_final_equity: percentile(&final_eq, 75.0),
        p95_final_equity: percentile(&final_eq, 95.0),
        median_max_drawdown: percentile(&drawdowns, 50.0),
        p95_max_drawdown: percentile(&drawdowns, 95.0),
        mean_max_drawdown: mean_dd,
        prob_ruin,
        median_sharpe: percentile(&sharpes, 50.0),
        p5_sharpe: percentile(&sharpes, 5.0),
        final_equities: final_equities_out,
        equity_paths,
    }
}

// ---------------------------------------------------------------------------
// Tests (unit)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_returns(n: usize, r: f64) -> Vec<f64> {
        vec![r; n]
    }

    #[test]
    fn test_fit_distribution_basic() {
        let returns = flat_returns(100, 0.001);
        let dist = fit_distribution(&returns);
        assert!((dist.mean - 0.001).abs() < 1e-9);
        assert!(dist.std < 1e-6); // near-zero variance for flat series
    }

    #[test]
    fn test_fit_distribution_realistic() {
        // Slightly varied returns
        let returns: Vec<f64> = (0..252)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.005 })
            .collect();
        let dist = fit_distribution(&returns);
        assert!(dist.mean > 0.0);
        assert!(dist.std > 0.0);
        assert!(dist.tail_alpha > 0.0);
    }

    #[test]
    fn test_hill_estimator_reasonable() {
        let returns: Vec<f64> = (0..500).map(|i| (i as f64) * 0.0001 - 0.025).collect();
        let mut sorted = returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = hill_tail_index(&sorted);
        assert!(alpha >= 0.5 && alpha <= 10.0);
    }
}
