// walk_forward.rs -- parallel walk-forward optimization for SRFM backtesting
// Splits a time series into train/test folds and runs optimization on each.

use crate::backtest::{run_backtest, BacktestResult};
use crate::bar_data::DataStore;
use crate::optimizer::multi_objective_optimize;
use crate::params::{ParameterSpace, StrategyParams};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Configuration ──────────────────────────────────────────────────────────

/// Controls how the time series is split and how many threads to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    /// Number of bars in each training window.
    pub train_bars: usize,
    /// Number of bars in each out-of-sample test window.
    pub test_bars: usize,
    /// Number of bars to step forward between consecutive folds.
    pub step_bars: usize,
    /// Parallelism level for fold execution.
    pub n_threads: usize,
    /// Minimum number of folds required for results to be meaningful.
    pub min_folds: usize,
    /// Number of optimisation trials per fold (passed to the optimizer).
    pub optimizer_trials: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            train_bars: 5040,   // ~1 year of 15-min bars
            test_bars: 1260,    // ~3 months
            step_bars: 1260,    // non-overlapping test windows
            n_threads: 4,
            min_folds: 3,
            optimizer_trials: 50,
            seed: 42,
        }
    }
}

// ── Fold definition ────────────────────────────────────────────────────────

/// Describes one train/test split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardFold {
    pub fold_idx: usize,
    /// Inclusive start index of the training window.
    pub train_start: usize,
    /// Exclusive end index of the training window.
    pub train_end: usize,
    /// Inclusive start index of the test window.
    pub test_start: usize,
    /// Exclusive end index of the test window.
    pub test_end: usize,
}

impl WalkForwardFold {
    pub fn train_len(&self) -> usize {
        self.train_end.saturating_sub(self.train_start)
    }

    pub fn test_len(&self) -> usize {
        self.test_end.saturating_sub(self.test_start)
    }
}

/// Generate all folds for a given total bar count and config.
pub fn generate_folds(total_bars: usize, config: &WalkForwardConfig) -> Vec<WalkForwardFold> {
    let mut folds = Vec::new();
    let window = config.train_bars + config.test_bars;

    if total_bars < window {
        return folds;
    }

    let mut fold_idx = 0usize;
    let mut train_start = 0usize;

    loop {
        let train_end = train_start + config.train_bars;
        let test_start = train_end;
        let test_end = test_start + config.test_bars;

        if test_end > total_bars {
            break;
        }

        folds.push(WalkForwardFold {
            fold_idx,
            train_start,
            train_end,
            test_start,
            test_end,
        });

        fold_idx += 1;
        train_start += config.step_bars;
    }

    folds
}

// ── Results ────────────────────────────────────────────────────────────────

/// Results from a single fold: best params found in training plus OOS metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    pub fold_idx: usize,
    pub train_sharpe: f64,
    pub test_sharpe: f64,
    pub train_max_dd: f64,
    pub test_max_dd: f64,
    /// Best parameters found during in-sample optimisation.
    pub params: HashMap<String, f64>,
    /// Number of optimisation trials run.
    pub n_trials: usize,
}

impl FoldResult {
    /// Degradation: in-sample minus out-of-sample Sharpe.
    pub fn sharpe_degradation(&self) -> f64 {
        self.train_sharpe - self.test_sharpe
    }
}

/// Aggregated walk-forward results across all folds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardResult {
    pub folds: Vec<FoldResult>,
    /// Mean OOS Sharpe across all folds.
    pub aggregate_sharpe: f64,
    /// Fraction of folds where OOS Sharpe > 0.
    pub consistency_ratio: f64,
    /// Overfitting score: mean(train - test) / std(test).
    /// Higher values indicate worse overfitting.
    pub overfitting_score: f64,
    /// Mean train Sharpe across folds.
    pub mean_train_sharpe: f64,
    /// Total number of folds run.
    pub n_folds: usize,
}

impl WalkForwardResult {
    fn compute(folds: Vec<FoldResult>) -> Self {
        if folds.is_empty() {
            return Self {
                folds: vec![],
                aggregate_sharpe: 0.0,
                consistency_ratio: 0.0,
                overfitting_score: f64::INFINITY,
                mean_train_sharpe: 0.0,
                n_folds: 0,
            };
        }

        let n = folds.len() as f64;
        let mean_test = folds.iter().map(|f| f.test_sharpe).sum::<f64>() / n;
        let mean_train = folds.iter().map(|f| f.train_sharpe).sum::<f64>() / n;
        let consistency = folds.iter().filter(|f| f.test_sharpe > 0.0).count() as f64 / n;

        let mean_degradation = folds.iter().map(|f| f.sharpe_degradation()).sum::<f64>() / n;
        let test_variance = folds
            .iter()
            .map(|f| (f.test_sharpe - mean_test).powi(2))
            .sum::<f64>()
            / n;
        let test_std = test_variance.sqrt();
        let overfitting_score = if test_std > 1e-9 {
            mean_degradation / test_std
        } else {
            f64::INFINITY
        };

        let n_folds = folds.len();
        Self {
            folds,
            aggregate_sharpe: mean_test,
            consistency_ratio: consistency,
            overfitting_score,
            mean_train_sharpe: mean_train,
            n_folds,
        }
    }

    /// True if results suggest no excessive overfitting and positive expectancy.
    pub fn is_robust(&self, min_consistency: f64, max_overfit: f64) -> bool {
        self.aggregate_sharpe > 0.0
            && self.consistency_ratio >= min_consistency
            && self.overfitting_score <= max_overfit
    }
}

// ── Bar slice helper ───────────────────────────────────────────────────────

/// Extract a subset of bars from a DataStore given bar index range [start, end).
fn slice_datastore(data: &DataStore, start: usize, end: usize) -> DataStore {
    data.iter()
        .map(|(sym, bars)| {
            let sliced: Vec<_> = bars
                .iter()
                .skip(start)
                .take(end.saturating_sub(start))
                .cloned()
                .collect();
            (sym.clone(), sliced)
        })
        .collect()
}

/// Compute Sharpe ratio from a BacktestResult's equity curve implied data.
/// Uses CAGR / max_drawdown as a proxy when stddev isn't available.
fn extract_sharpe(result: &BacktestResult) -> f64 {
    // Use the Sharpe already computed by the backtest engine.
    if result.sharpe.is_finite() {
        result.sharpe
    } else {
        0.0
    }
}

/// Params -> HashMap<String, f64> for serialisation.
fn params_to_map(p: &StrategyParams) -> HashMap<String, f64> {
    let mut m = HashMap::new();
    m.insert("min_hold_bars".to_string(), p.min_hold_bars as f64);
    m.insert("stale_15m_move".to_string(), p.stale_15m_move);
    m.insert("winner_protection_pct".to_string(), p.winner_protection_pct);
    m.insert("garch_target_vol".to_string(), p.garch_target_vol);
    m.insert("corr_normal".to_string(), p.corr_normal);
    m.insert("corr_stress".to_string(), p.corr_stress);
    m.insert("hour_boost_multiplier".to_string(), p.hour_boost_multiplier);
    m
}

// ── Public entry point ─────────────────────────────────────────────────────

/// Run a full walk-forward optimization.
///
/// For each fold: optimize on the training window, then evaluate the best
/// params on the out-of-sample test window. All folds are run in parallel
/// using the thread count from `config`.
pub fn run_walk_forward(data: &DataStore, config: &WalkForwardConfig) -> WalkForwardResult {
    // Determine total bar count from the first symbol.
    let total_bars = data
        .values()
        .map(|bars| bars.len())
        .min()
        .unwrap_or(0);

    let folds = generate_folds(total_bars, config);

    if folds.len() < config.min_folds {
        eprintln!(
            "walk_forward: only {} folds generated (need {}). Returning empty result.",
            folds.len(),
            config.min_folds
        );
        return WalkForwardResult::compute(vec![]);
    }

    // Configure rayon thread pool for this run.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.n_threads)
        .build()
        .unwrap_or_else(|_| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("rayon pool")
        });

    let fold_results: Vec<FoldResult> = pool.install(|| {
        folds
            .par_iter()
            .map(|fold| {
                let train_data = slice_datastore(data, fold.train_start, fold.train_end);
                let test_data = slice_datastore(data, fold.test_start, fold.test_end);

                // Optimize on training data using default parameter space.
                let space = ParameterSpace::default();
                let pareto = multi_objective_optimize(&train_data, &space);

                let best_params = pareto
                    .first()
                    .map(|p| p.params.clone())
                    .unwrap_or_default();

                // Evaluate best params on training window (in-sample confirmation).
                let train_result = run_backtest(&train_data, &best_params);
                // Evaluate on test window (out-of-sample).
                let test_result = run_backtest(&test_data, &best_params);

                FoldResult {
                    fold_idx: fold.fold_idx,
                    train_sharpe: extract_sharpe(&train_result),
                    test_sharpe: extract_sharpe(&test_result),
                    train_max_dd: train_result.max_drawdown,
                    test_max_dd: test_result.max_drawdown,
                    params: params_to_map(&best_params),
                    n_trials: config.optimizer_trials,
                }
            })
            .collect()
    });

    WalkForwardResult::compute(fold_results)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_folds_basic() {
        let config = WalkForwardConfig {
            train_bars: 100,
            test_bars: 20,
            step_bars: 20,
            ..Default::default()
        };
        let folds = generate_folds(200, &config);
        // With step=20, test=20, train=100: start at 0, 20, 40, 60
        // fold 0: train [0,100), test [100,120) -- test_end=120 <= 200 OK
        // fold 1: train [20,120), test [120,140) OK
        // fold 2: train [40,140), test [140,160) OK
        // fold 3: train [60,160), test [160,180) OK
        // fold 4: train [80,180), test [180,200) OK
        assert!(!folds.is_empty());
        // All folds should have correct lengths.
        for f in &folds {
            assert_eq!(f.train_len(), 100);
            assert_eq!(f.test_len(), 20);
        }
    }

    #[test]
    fn test_generate_folds_insufficient_data() {
        let config = WalkForwardConfig {
            train_bars: 1000,
            test_bars: 500,
            step_bars: 500,
            ..Default::default()
        };
        // Only 100 bars -- not enough for even one fold.
        let folds = generate_folds(100, &config);
        assert!(folds.is_empty());
    }

    #[test]
    fn test_generate_folds_exact_fit() {
        let config = WalkForwardConfig {
            train_bars: 50,
            test_bars: 50,
            step_bars: 50,
            ..Default::default()
        };
        let folds = generate_folds(100, &config);
        assert_eq!(folds.len(), 1);
        assert_eq!(folds[0].train_start, 0);
        assert_eq!(folds[0].train_end, 50);
        assert_eq!(folds[0].test_start, 50);
        assert_eq!(folds[0].test_end, 100);
    }

    #[test]
    fn test_fold_indices_are_sequential() {
        let config = WalkForwardConfig {
            train_bars: 100,
            test_bars: 25,
            step_bars: 25,
            ..Default::default()
        };
        let folds = generate_folds(300, &config);
        for (i, fold) in folds.iter().enumerate() {
            assert_eq!(fold.fold_idx, i);
        }
    }

    #[test]
    fn test_walk_forward_result_consistency_ratio() {
        let folds = vec![
            FoldResult { fold_idx: 0, train_sharpe: 1.5, test_sharpe: 0.8, train_max_dd: 0.1, test_max_dd: 0.12, params: HashMap::new(), n_trials: 10 },
            FoldResult { fold_idx: 1, train_sharpe: 1.2, test_sharpe: -0.3, train_max_dd: 0.08, test_max_dd: 0.15, params: HashMap::new(), n_trials: 10 },
            FoldResult { fold_idx: 2, train_sharpe: 0.9, test_sharpe: 0.4, train_max_dd: 0.09, test_max_dd: 0.11, params: HashMap::new(), n_trials: 10 },
            FoldResult { fold_idx: 3, train_sharpe: 1.1, test_sharpe: 0.2, train_max_dd: 0.07, test_max_dd: 0.09, params: HashMap::new(), n_trials: 10 },
        ];
        let result = WalkForwardResult::compute(folds);
        // 3 out of 4 folds have test_sharpe > 0 -> consistency = 0.75
        assert!((result.consistency_ratio - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_walk_forward_result_aggregate_sharpe() {
        let folds = vec![
            FoldResult { fold_idx: 0, train_sharpe: 2.0, test_sharpe: 1.0, train_max_dd: 0.1, test_max_dd: 0.1, params: HashMap::new(), n_trials: 5 },
            FoldResult { fold_idx: 1, train_sharpe: 2.0, test_sharpe: 3.0, train_max_dd: 0.1, test_max_dd: 0.1, params: HashMap::new(), n_trials: 5 },
        ];
        let result = WalkForwardResult::compute(folds);
        assert!((result.aggregate_sharpe - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_folds_result() {
        let result = WalkForwardResult::compute(vec![]);
        assert_eq!(result.n_folds, 0);
        assert_eq!(result.aggregate_sharpe, 0.0);
        assert_eq!(result.consistency_ratio, 0.0);
    }

    #[test]
    fn test_overfitting_score_calculation() {
        // If all test sharpes are identical, std=0 -> score=infinity.
        let folds = vec![
            FoldResult { fold_idx: 0, train_sharpe: 2.0, test_sharpe: 1.0, train_max_dd: 0.1, test_max_dd: 0.1, params: HashMap::new(), n_trials: 5 },
            FoldResult { fold_idx: 1, train_sharpe: 2.0, test_sharpe: 1.0, train_max_dd: 0.1, test_max_dd: 0.1, params: HashMap::new(), n_trials: 5 },
        ];
        let result = WalkForwardResult::compute(folds);
        // mean degradation = 1.0, std(test) = 0.0 -> infinity
        assert!(result.overfitting_score.is_infinite());
    }

    #[test]
    fn test_is_robust() {
        let folds: Vec<FoldResult> = (0..5)
            .map(|i| FoldResult {
                fold_idx: i,
                train_sharpe: 1.5,
                test_sharpe: 0.7,
                train_max_dd: 0.1,
                test_max_dd: 0.12,
                params: HashMap::new(),
                n_trials: 20,
            })
            .collect();
        let result = WalkForwardResult::compute(folds);
        assert!(result.aggregate_sharpe > 0.0);
        assert_eq!(result.consistency_ratio, 1.0);
        // With identical test sharpes, std=0 -> overfit=infinity; test that
        // is_robust returns false when overfit threshold is exceeded.
        assert!(!result.is_robust(0.6, 3.0)); // overfit=inf exceeds 3.0
    }

    #[test]
    fn test_params_to_map_contains_keys() {
        let p = StrategyParams::default();
        let m = params_to_map(&p);
        assert!(m.contains_key("min_hold_bars"));
        assert!(m.contains_key("garch_target_vol"));
        assert!(m.contains_key("corr_normal"));
    }

    #[test]
    fn test_slice_datastore_length() {
        let mut ds: DataStore = HashMap::new();
        let bars: Vec<crate::bar_data::BarData> = (0..200)
            .map(|i| crate::bar_data::BarData {
                timestamp: i as i64 * 900,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: 1000.0,
            })
            .collect();
        ds.insert("BTC".to_string(), bars);
        let sliced = slice_datastore(&ds, 50, 150);
        assert_eq!(sliced["BTC"].len(), 100);
        assert_eq!(sliced["BTC"][0].timestamp, 50 * 900);
    }

    #[test]
    fn test_fold_train_test_no_overlap() {
        let config = WalkForwardConfig {
            train_bars: 80,
            test_bars: 20,
            step_bars: 20,
            ..Default::default()
        };
        let folds = generate_folds(200, &config);
        for fold in &folds {
            assert!(fold.train_end <= fold.test_start, "train and test overlap at fold {}", fold.fold_idx);
        }
    }
}
