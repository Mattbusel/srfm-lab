// monte_carlo_backtest.rs -- Monte Carlo backtest analysis for SRFM
// Bootstraps equity curves from return series to estimate distribution of outcomes.

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::distributions::Uniform;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ── Resampling methods ─────────────────────────────────────────────────────

/// Method used to resample the return series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResampleMethod {
    /// IID bootstrap: each return is drawn uniformly at random with replacement.
    Bootstrap,
    /// Circular block bootstrap: preserves local autocorrelation structure.
    CircularBlock { block_size: usize },
    /// Stationary bootstrap (Politis & Romano): variable block size with
    /// geometric distribution parameterised by `p` (mean block = 1/p).
    Stationary { p: f64 },
}

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration for a Monte Carlo simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCConfig {
    /// Number of simulated equity paths.
    pub n_paths: usize,
    /// Number of bars per simulated path.
    pub n_bars: usize,
    /// Resampling method.
    pub resample_method: ResampleMethod,
    /// Initial equity level (default: 1.0).
    pub initial_equity: f64,
    /// Ruin threshold: fraction of initial equity that counts as ruin.
    /// Default: 0.80 (i.e. -20% drawdown from peak).
    pub ruin_threshold: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for MCConfig {
    fn default() -> Self {
        Self {
            n_paths: 1000,
            n_bars: 252,
            resample_method: ResampleMethod::Bootstrap,
            initial_equity: 1.0,
            ruin_threshold: 0.80,
            seed: 42,
        }
    }
}

// ── Result types ───────────────────────────────────────────────────────────

/// Full Monte Carlo simulation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCResult {
    /// All simulated equity paths (n_paths x n_bars+1 including initial equity).
    pub paths: Vec<Vec<f64>>,
    /// 5th percentile of terminal equity.
    pub percentile_5: f64,
    /// 25th percentile of terminal equity.
    pub percentile_25: f64,
    /// Median terminal equity.
    pub median: f64,
    /// 75th percentile of terminal equity.
    pub percentile_75: f64,
    /// 95th percentile of terminal equity.
    pub percentile_95: f64,
    /// Mean terminal equity.
    pub mean_terminal: f64,
    /// Fraction of paths that hit ruin (drawdown >= (1 - ruin_threshold)).
    pub prob_ruin: f64,
    /// Fraction of paths with positive terminal equity vs initial.
    pub prob_profit: f64,
    /// Maximum drawdown averaged across all paths.
    pub mean_max_drawdown: f64,
}

impl MCResult {
    /// Returns empty/zero result.
    pub fn empty() -> Self {
        Self {
            paths: vec![],
            percentile_5: 0.0,
            percentile_25: 0.0,
            median: 0.0,
            percentile_75: 0.0,
            percentile_95: 0.0,
            mean_terminal: 0.0,
            prob_ruin: 0.0,
            prob_profit: 0.0,
            mean_max_drawdown: 0.0,
        }
    }
}

// ── Percentile helper ──────────────────────────────────────────────────────

fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Maximum drawdown ───────────────────────────────────────────────────────

/// Compute maximum drawdown for an equity path.
/// Returns the maximum fractional decline from any peak to any subsequent trough.
pub fn max_drawdown(equity: &[f64]) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let mut peak = equity[0];
    let mut max_dd = 0.0f64;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        let dd = (peak - e) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

// ── Resampler ──────────────────────────────────────────────────────────────

/// Generate one resampled return series of length `n` from `returns` using
/// the specified method.
fn resample_returns<R: Rng>(
    returns: &[f64],
    n: usize,
    method: &ResampleMethod,
    rng: &mut R,
) -> Vec<f64> {
    if returns.is_empty() || n == 0 {
        return vec![0.0; n];
    }

    match method {
        ResampleMethod::Bootstrap => {
            let dist = Uniform::new(0, returns.len());
            (0..n).map(|_| returns[rng.sample(dist)]).collect()
        }

        ResampleMethod::CircularBlock { block_size } => {
            let block_size = (*block_size).max(1);
            let m = returns.len();
            let mut result = Vec::with_capacity(n);
            let start_dist = Uniform::new(0, m);

            while result.len() < n {
                let start = rng.sample(start_dist);
                for i in 0..block_size {
                    if result.len() >= n {
                        break;
                    }
                    result.push(returns[(start + i) % m]);
                }
            }
            result.truncate(n);
            result
        }

        ResampleMethod::Stationary { p } => {
            // Politis-Romano stationary bootstrap.
            // With probability p, start a new block; otherwise continue current block.
            let p = p.clamp(1e-6, 1.0);
            let m = returns.len();
            let mut result = Vec::with_capacity(n);
            let start_dist = Uniform::new(0, m);
            let mut pos = rng.sample(start_dist);

            for _ in 0..n {
                result.push(returns[pos % m]);
                // With probability p, jump to a new random start.
                if rng.gen::<f64>() < p {
                    pos = rng.sample(start_dist);
                } else {
                    pos += 1;
                }
            }
            result
        }
    }
}

/// Convert a return series into an equity curve starting at `initial`.
fn returns_to_equity(returns: &[f64], initial: f64) -> Vec<f64> {
    let mut curve = Vec::with_capacity(returns.len() + 1);
    curve.push(initial);
    let mut equity = initial;
    for &r in returns {
        equity *= 1.0 + r;
        curve.push(equity);
    }
    curve
}

// ── Core simulation ────────────────────────────────────────────────────────

/// Run a Monte Carlo simulation on a return series.
///
/// `returns` should be a sequence of per-bar returns (e.g. 0.002 = +0.2%).
pub fn run_monte_carlo(returns: &[f64], config: &MCConfig) -> MCResult {
    if returns.is_empty() {
        return MCResult::empty();
    }

    let seed = config.seed;
    let method = config.resample_method.clone();
    let n_bars = config.n_bars;
    let initial = config.initial_equity;
    let ruin_thresh = config.ruin_threshold;

    // Generate paths in parallel chunks to avoid per-thread seed collision.
    let chunk_size = 64usize;
    let n_chunks = (config.n_paths + chunk_size - 1) / chunk_size;

    let paths: Vec<Vec<f64>> = (0..n_chunks)
        .into_par_iter()
        .flat_map(|chunk_idx| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(chunk_idx as u64 * 1_000_003));
            let paths_in_chunk = chunk_size.min(
                config.n_paths.saturating_sub(chunk_idx * chunk_size),
            );
            let mut local_paths = Vec::with_capacity(paths_in_chunk);
            for _ in 0..paths_in_chunk {
                let sampled = resample_returns(returns, n_bars, &method, &mut rng);
                local_paths.push(returns_to_equity(&sampled, initial));
            }
            local_paths
        })
        .collect();

    if paths.is_empty() {
        return MCResult::empty();
    }

    // Extract terminal equities.
    let mut terminal: Vec<f64> = paths
        .iter()
        .map(|p| *p.last().unwrap_or(&initial))
        .collect();
    terminal.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pct_5 = percentile_sorted(&terminal, 5.0);
    let pct_25 = percentile_sorted(&terminal, 25.0);
    let median = percentile_sorted(&terminal, 50.0);
    let pct_75 = percentile_sorted(&terminal, 75.0);
    let pct_95 = percentile_sorted(&terminal, 95.0);
    let mean_terminal = terminal.iter().sum::<f64>() / terminal.len() as f64;
    let prob_profit = terminal.iter().filter(|&&e| e > initial).count() as f64
        / terminal.len() as f64;

    // Ruin: any path that at any point dropped below ruin_threshold * peak.
    let n_ruin = paths.iter().filter(|path| hit_ruin(path, ruin_thresh)).count();
    let prob_ruin = n_ruin as f64 / paths.len() as f64;

    let mean_max_dd = paths.iter().map(|p| max_drawdown(p)).sum::<f64>() / paths.len() as f64;

    MCResult {
        paths,
        percentile_5: pct_5,
        percentile_25: pct_25,
        median,
        percentile_75: pct_75,
        percentile_95: pct_95,
        mean_terminal,
        prob_ruin,
        prob_profit,
        mean_max_drawdown: mean_max_dd,
    }
}

/// True if the equity path at any point hit a ruin-level drawdown from its
/// running peak. ruin_threshold=0.80 means a -20% drawdown from peak.
fn hit_ruin(equity: &[f64], ruin_threshold: f64) -> bool {
    if equity.is_empty() {
        return false;
    }
    let mut peak = equity[0];
    for &e in equity {
        if e > peak {
            peak = e;
        }
        if peak > 0.0 && e / peak < ruin_threshold {
            return true;
        }
    }
    false
}

// ── Confidence bands ───────────────────────────────────────────────────────

/// Compute (p5, median, p95) confidence bands at `n_points` evenly-spaced bar
/// positions across all simulated paths.
///
/// Returns a Vec of length `n_points`, each element being (p5, median, p95).
pub fn confidence_bands(mc: &MCResult, n_points: usize) -> Vec<(f64, f64, f64)> {
    if mc.paths.is_empty() || n_points == 0 {
        return vec![];
    }

    let path_len = mc.paths[0].len();
    if path_len == 0 {
        return vec![];
    }

    let step = if n_points == 1 {
        0
    } else {
        (path_len - 1) / (n_points - 1)
    };

    (0..n_points)
        .map(|i| {
            let bar_idx = (i * step).min(path_len - 1);
            let mut vals: Vec<f64> = mc.paths.iter().map(|p| p[bar_idx]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p5 = percentile_sorted(&vals, 5.0);
            let median = percentile_sorted(&vals, 50.0);
            let p95 = percentile_sorted(&vals, 95.0);
            (p5, median, p95)
        })
        .collect()
}

// ── Summary stats helpers ──────────────────────────────────────────────────

/// Compute annualised return from a terminal equity level.
pub fn annualised_return(terminal_equity: f64, initial_equity: f64, n_bars: usize, bars_per_year: f64) -> f64 {
    if initial_equity <= 0.0 || n_bars == 0 {
        return 0.0;
    }
    let years = n_bars as f64 / bars_per_year;
    if years <= 0.0 {
        return 0.0;
    }
    (terminal_equity / initial_equity).powf(1.0 / years) - 1.0
}

/// Compute Sharpe ratio from a return series (excess return / std, annualised).
pub fn sharpe_from_returns(returns: &[f64], bars_per_year: f64, risk_free_rate: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let excess = mean - risk_free_rate / bars_per_year;
    let variance = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    if std_dev < 1e-12 {
        return 0.0;
    }
    excess / std_dev * bars_per_year.sqrt()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_returns(n: usize, r: f64) -> Vec<f64> {
        vec![r; n]
    }

    #[test]
    fn test_returns_to_equity_compounding() {
        // 10% return each bar for 3 bars starting at 1.0
        let returns = flat_returns(3, 0.10);
        let eq = returns_to_equity(&returns, 1.0);
        assert_eq!(eq.len(), 4);
        assert!((eq[0] - 1.0).abs() < 1e-12);
        assert!((eq[3] - 1.331).abs() < 1e-6);
    }

    #[test]
    fn test_max_drawdown_flat() {
        let equity = vec![1.0, 1.1, 1.2, 1.3];
        assert!((max_drawdown(&equity)).abs() < 1e-12);
    }

    #[test]
    fn test_max_drawdown_single_drop() {
        // Peak at 2.0, trough at 1.0 -> 50% drawdown.
        let equity = vec![1.0, 2.0, 1.0];
        assert!((max_drawdown(&equity) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_hit_ruin_false() {
        let equity = vec![1.0, 1.05, 1.1, 1.08];
        assert!(!hit_ruin(&equity, 0.8));
    }

    #[test]
    fn test_hit_ruin_true() {
        // Peak 1.0, then drops to 0.75 -> 25% drawdown -- exceeds 20% threshold.
        let equity = vec![1.0, 0.75];
        assert!(hit_ruin(&equity, 0.8));
    }

    #[test]
    fn test_bootstrap_length() {
        let returns = flat_returns(100, 0.001);
        let mut rng = StdRng::seed_from_u64(1);
        let resampled = resample_returns(&returns, 50, &ResampleMethod::Bootstrap, &mut rng);
        assert_eq!(resampled.len(), 50);
    }

    #[test]
    fn test_circular_block_length() {
        let returns: Vec<f64> = (0..50).map(|i| i as f64 * 0.001).collect();
        let mut rng = StdRng::seed_from_u64(2);
        let resampled = resample_returns(
            &returns,
            40,
            &ResampleMethod::CircularBlock { block_size: 5 },
            &mut rng,
        );
        assert_eq!(resampled.len(), 40);
    }

    #[test]
    fn test_stationary_bootstrap_length() {
        let returns = flat_returns(100, 0.002);
        let mut rng = StdRng::seed_from_u64(3);
        let resampled = resample_returns(
            &returns,
            60,
            &ResampleMethod::Stationary { p: 0.1 },
            &mut rng,
        );
        assert_eq!(resampled.len(), 60);
    }

    #[test]
    fn test_monte_carlo_paths_count() {
        let returns = flat_returns(252, 0.001);
        let config = MCConfig {
            n_paths: 100,
            n_bars: 252,
            seed: 7,
            ..Default::default()
        };
        let result = run_monte_carlo(&returns, &config);
        assert_eq!(result.paths.len(), 100);
        for path in &result.paths {
            assert_eq!(path.len(), 253); // n_bars + 1 (initial equity)
        }
    }

    #[test]
    fn test_monte_carlo_prob_profit_positive_returns() {
        // With consistently positive returns, prob_profit should be close to 1.
        let returns = flat_returns(100, 0.002);
        let config = MCConfig {
            n_paths: 200,
            n_bars: 100,
            seed: 11,
            ..Default::default()
        };
        let result = run_monte_carlo(&returns, &config);
        assert!(result.prob_profit > 0.95, "prob_profit = {}", result.prob_profit);
    }

    #[test]
    fn test_monte_carlo_prob_ruin_negative_returns() {
        // Large negative returns -> should almost always hit ruin.
        let returns = flat_returns(50, -0.02); // -2% per bar
        let config = MCConfig {
            n_paths: 200,
            n_bars: 50,
            seed: 13,
            ruin_threshold: 0.80,
            ..Default::default()
        };
        let result = run_monte_carlo(&returns, &config);
        // After 50 bars at -2%: equity = 0.98^50 ~ 0.364 -- deep ruin.
        assert!(result.prob_ruin > 0.5, "prob_ruin = {}", result.prob_ruin);
    }

    #[test]
    fn test_confidence_bands_length() {
        let returns = flat_returns(100, 0.001);
        let config = MCConfig { n_paths: 50, n_bars: 100, seed: 5, ..Default::default() };
        let mc = run_monte_carlo(&returns, &config);
        let bands = confidence_bands(&mc, 10);
        assert_eq!(bands.len(), 10);
    }

    #[test]
    fn test_confidence_bands_ordering() {
        let returns: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 0.01 } else { -0.005 }).collect();
        let config = MCConfig { n_paths: 100, n_bars: 200, seed: 9, ..Default::default() };
        let mc = run_monte_carlo(&returns, &config);
        let bands = confidence_bands(&mc, 5);
        for (p5, median, p95) in &bands {
            assert!(p5 <= median, "p5={p5}, median={median}");
            assert!(median <= p95, "median={median}, p95={p95}");
        }
    }

    #[test]
    fn test_sharpe_from_returns_positive() {
        let returns = flat_returns(252, 0.002);
        let sharpe = sharpe_from_returns(&returns, 252.0, 0.0);
        // All returns identical -> std=0 -> sharpe=0 (std guard).
        assert_eq!(sharpe, 0.0);
    }

    #[test]
    fn test_annualised_return_calculation() {
        // 2x in 252 bars with 252 bars/year -> CAGR = 100% per year.
        let ann = annualised_return(2.0, 1.0, 252, 252.0);
        assert!((ann - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mc_empty_returns() {
        let result = run_monte_carlo(&[], &MCConfig::default());
        assert!(result.paths.is_empty());
    }

    #[test]
    fn test_percentile_sorted_edges() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_sorted(&sorted, 0.0), 1.0);
        assert_eq!(percentile_sorted(&sorted, 100.0), 5.0);
    }
}
