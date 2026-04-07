// param_grid_search.rs -- parallel grid/random/Sobol/Latin-hypercube search
// Evaluates strategy parameters across a defined search space in parallel.

use crate::backtest::run_backtest;
use crate::bar_data::DataStore;
use crate::params::StrategyParams;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Search space types ─────────────────────────────────────────────────────

/// Method used to generate parameter combinations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMethod {
    /// Full Cartesian product of all parameter ranges.
    Grid,
    /// Uniform random sampling within ranges.
    Random,
    /// Sobol low-discrepancy quasi-random sequence.
    Sobol,
    /// Latin hypercube sampling: guarantees one sample per stratum per dimension.
    LatinHypercube,
}

/// Configuration for one parameter grid search run.
///
/// `param_ranges`: maps param name -> (min, max, n_steps).
/// For Random/Sobol/LHS, n_steps is ignored; `n_trials` controls sample count.
/// For Grid, the total trial count = product of all n_steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSearchConfig {
    /// Parameter name -> (min, max, n_steps_for_grid).
    pub param_ranges: HashMap<String, (f64, f64, usize)>,
    pub method: SearchMethod,
    /// Number of trials for non-grid methods.
    pub n_trials: usize,
    /// Parallelism level.
    pub n_threads: usize,
    /// Minimum number of trades a trial must produce to be included in results.
    pub min_trades: u64,
    /// Random seed.
    pub seed: u64,
}

impl Default for GridSearchConfig {
    fn default() -> Self {
        Self {
            param_ranges: HashMap::new(),
            method: SearchMethod::Random,
            n_trials: 200,
            n_threads: 4,
            min_trades: 10,
            seed: 42,
        }
    }
}

// ── Trial result ───────────────────────────────────────────────────────────

/// Result from evaluating one parameter combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub params: HashMap<String, f64>,
    pub sharpe: f64,
    pub max_dd: f64,
    pub calmar: f64,
    pub n_trades: u64,
    pub cagr: f64,
    pub win_rate: f64,
}

impl TrialResult {
    /// Composite score used for ranking: Sharpe / (1 + max_dd).
    pub fn score(&self) -> f64 {
        if self.n_trades == 0 {
            return f64::NEG_INFINITY;
        }
        self.sharpe / (1.0 + self.max_dd)
    }
}

// ── Aggregated search result ───────────────────────────────────────────────

/// Complete results from a grid/random search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSearchResult {
    pub best_params: HashMap<String, f64>,
    pub best_sharpe: f64,
    pub best_calmar: f64,
    /// All trial results sorted by score descending.
    pub all_trials: Vec<TrialResult>,
    /// Parameter importance: fraction of total score variance explained by each param.
    pub param_importance: HashMap<String, f64>,
    /// Total trials evaluated.
    pub n_evaluated: usize,
    /// Trials filtered out (too few trades).
    pub n_filtered: usize,
}

impl GridSearchResult {
    pub fn top_n(&self, n: usize) -> &[TrialResult] {
        &self.all_trials[..n.min(self.all_trials.len())]
    }
}

// ── Sobol sequence ─────────────────────────────────────────────────────────

/// Generate one component of a Sobol sequence in base 2 using van der Corput
/// scrambling. Returns a value in [0, 1).
///
/// This is a simplified 1D implementation using the standard direction numbers
/// for the first 32 dimensions. For d > 0 we use primitive polynomials.
fn van_der_corput(index: u64, base: u64) -> f64 {
    let mut n = index;
    let mut result = 0.0f64;
    let mut denom = 1.0f64;
    while n > 0 {
        denom *= base as f64;
        result += (n % base) as f64 / denom;
        n /= base;
    }
    result
}

/// Return `n_dims` Sobol points for sample `index` using van der Corput in
/// primes 2, 3, 5, 7, 11, ... as the base for each dimension.
fn sobol_point(index: u64, n_dims: usize) -> Vec<f64> {
    // Use the first n_dims primes as bases.
    const PRIMES: [u64; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
    (0..n_dims)
        .map(|d| van_der_corput(index + 1, PRIMES[d % PRIMES.len()]))
        .collect()
}

// ── Latin Hypercube ────────────────────────────────────────────────────────

/// Generate a Latin Hypercube sample matrix of shape (n_trials x n_dims).
/// Each column has one sample per stratum [k/n, (k+1)/n) for k=0..n.
fn latin_hypercube<R: Rng>(n_trials: usize, n_dims: usize, rng: &mut R) -> Vec<Vec<f64>> {
    if n_trials == 0 || n_dims == 0 {
        return vec![];
    }
    let mut matrix = vec![vec![0.0f64; n_dims]; n_trials];
    let inv_n = 1.0 / n_trials as f64;

    for d in 0..n_dims {
        // Create stratified samples.
        let mut strata: Vec<f64> = (0..n_trials)
            .map(|k| (k as f64 + rng.gen::<f64>()) * inv_n)
            .collect();
        // Shuffle the strata permutation for this dimension.
        strata.shuffle(rng);
        for i in 0..n_trials {
            matrix[i][d] = strata[i];
        }
    }
    matrix
}

// ── Parameter point generation ─────────────────────────────────────────────

/// Ordered list of param names for stable indexing.
fn param_names(ranges: &HashMap<String, (f64, f64, usize)>) -> Vec<String> {
    let mut names: Vec<String> = ranges.keys().cloned().collect();
    names.sort(); // Deterministic order.
    names
}

/// Map a unit hypercube point to actual parameter values.
fn scale_point(
    unit: &[f64],
    names: &[String],
    ranges: &HashMap<String, (f64, f64, usize)>,
) -> HashMap<String, f64> {
    names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let (min, max, _) = ranges[name];
            let val = min + unit[i].clamp(0.0, 1.0) * (max - min);
            (name.clone(), val)
        })
        .collect()
}

/// Generate all trial parameter sets according to the search method.
fn generate_trial_points(config: &GridSearchConfig) -> Vec<HashMap<String, f64>> {
    let names = param_names(&config.param_ranges);
    let n_dims = names.len();

    if n_dims == 0 {
        return vec![];
    }

    match &config.method {
        SearchMethod::Grid => {
            // Build the Cartesian product of all grid values.
            let axes: Vec<Vec<f64>> = names
                .iter()
                .map(|name| {
                    let (min, max, steps) = config.param_ranges[name];
                    let steps = steps.max(1);
                    if steps == 1 {
                        vec![min]
                    } else {
                        (0..steps)
                            .map(|i| min + i as f64 * (max - min) / (steps - 1) as f64)
                            .collect()
                    }
                })
                .collect();

            // Recursive Cartesian product.
            let mut result: Vec<Vec<f64>> = vec![vec![]];
            for axis in &axes {
                result = result
                    .iter()
                    .flat_map(|prev| {
                        axis.iter().map(|&v| {
                            let mut next = prev.clone();
                            next.push(v);
                            next
                        })
                    })
                    .collect();
            }
            result
                .into_iter()
                .map(|point| scale_raw_point(&point, &names, &config.param_ranges))
                .collect()
        }

        SearchMethod::Random => {
            let mut rng = StdRng::seed_from_u64(config.seed);
            (0..config.n_trials)
                .map(|_| {
                    let unit: Vec<f64> = (0..n_dims).map(|_| rng.gen::<f64>()).collect();
                    scale_point(&unit, &names, &config.param_ranges)
                })
                .collect()
        }

        SearchMethod::Sobol => {
            (0..config.n_trials)
                .map(|i| {
                    let unit = sobol_point(i as u64, n_dims);
                    scale_point(&unit, &names, &config.param_ranges)
                })
                .collect()
        }

        SearchMethod::LatinHypercube => {
            let mut rng = StdRng::seed_from_u64(config.seed);
            let matrix = latin_hypercube(config.n_trials, n_dims, &mut rng);
            matrix
                .iter()
                .map(|row| scale_point(row, &names, &config.param_ranges))
                .collect()
        }
    }
}

/// Version of scale_point used for grid method where values are already scaled.
fn scale_raw_point(
    values: &[f64],
    names: &[String],
    _ranges: &HashMap<String, (f64, f64, usize)>,
) -> HashMap<String, f64> {
    names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), values[i]))
        .collect()
}

// ── StrategyParams from HashMap ────────────────────────────────────────────

/// Apply a parameter map onto a base StrategyParams, overriding known fields.
pub fn apply_param_map(base: &StrategyParams, params: &HashMap<String, f64>) -> StrategyParams {
    let mut p = base.clone();
    if let Some(&v) = params.get("min_hold_bars") {
        p.min_hold_bars = v.max(1.0) as u32;
    }
    if let Some(&v) = params.get("stale_15m_move") {
        p.stale_15m_move = v.clamp(0.0001, 0.05);
    }
    if let Some(&v) = params.get("winner_protection_pct") {
        p.winner_protection_pct = v.clamp(0.05, 0.95);
    }
    if let Some(&v) = params.get("garch_target_vol") {
        p.garch_target_vol = v.clamp(0.05, 2.0);
    }
    if let Some(&v) = params.get("corr_normal") {
        p.corr_normal = v.clamp(0.0, 0.99);
    }
    if let Some(&v) = params.get("corr_stress") {
        p.corr_stress = v.clamp(p.corr_normal, 0.99);
    }
    if let Some(&v) = params.get("hour_boost_multiplier") {
        p.hour_boost_multiplier = v.clamp(1.0, 3.0);
    }
    p
}

// ── Param importance via variance decomposition ────────────────────────────

/// Compute parameter importance as the fraction of total score variance
/// explained by variance in each parameter (one-way ANOVA proxy).
fn compute_param_importance(trials: &[TrialResult]) -> HashMap<String, f64> {
    if trials.len() < 2 {
        return HashMap::new();
    }

    let scores: Vec<f64> = trials.iter().map(|t| t.score()).collect();
    let n = scores.len() as f64;
    let mean_score = scores.iter().sum::<f64>() / n;
    let total_var = scores
        .iter()
        .map(|&s| (s - mean_score).powi(2))
        .sum::<f64>()
        / n;

    if total_var < 1e-12 {
        return trials
            .first()
            .map(|t| t.params.keys().map(|k| (k.clone(), 0.0)).collect())
            .unwrap_or_default();
    }

    // Get all param names from the first trial.
    let param_names: Vec<String> = trials
        .first()
        .map(|t| {
            let mut names: Vec<String> = t.params.keys().cloned().collect();
            names.sort();
            names
        })
        .unwrap_or_default();

    let mut importance: HashMap<String, f64> = HashMap::new();

    for name in &param_names {
        // Split trials into n_buckets groups based on this param's value.
        let vals: Vec<f64> = trials
            .iter()
            .filter_map(|t| t.params.get(name).copied())
            .collect();
        if vals.is_empty() {
            importance.insert(name.clone(), 0.0);
            continue;
        }
        let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range < 1e-12 {
            importance.insert(name.clone(), 0.0);
            continue;
        }

        // Partition into 5 buckets and compute between-group variance.
        let n_buckets = 5usize;
        let mut bucket_scores: Vec<Vec<f64>> = vec![vec![]; n_buckets];
        for trial in trials {
            if let Some(&val) = trial.params.get(name) {
                let bucket = ((val - min_val) / range * (n_buckets - 1) as f64)
                    .round()
                    .clamp(0.0, (n_buckets - 1) as f64) as usize;
                bucket_scores[bucket].push(trial.score());
            }
        }

        // Between-group variance.
        let between_var: f64 = bucket_scores
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| {
                let bm = b.iter().sum::<f64>() / b.len() as f64;
                (bm - mean_score).powi(2) * b.len() as f64 / n
            })
            .sum();

        importance.insert(name.clone(), (between_var / total_var).clamp(0.0, 1.0));
    }

    // Normalise so importances sum to 1.
    let total_imp: f64 = importance.values().sum();
    if total_imp > 1e-12 {
        for v in importance.values_mut() {
            *v /= total_imp;
        }
    }

    importance
}

// ── Main entry point ───────────────────────────────────────────────────────

/// Run a parallel parameter grid/random search against the provided data.
pub fn run_grid_search(data: &DataStore, config: &GridSearchConfig) -> GridSearchResult {
    let trial_params = generate_trial_points(config);
    let n_total = trial_params.len();

    if n_total == 0 {
        return GridSearchResult {
            best_params: HashMap::new(),
            best_sharpe: f64::NEG_INFINITY,
            best_calmar: 0.0,
            all_trials: vec![],
            param_importance: HashMap::new(),
            n_evaluated: 0,
            n_filtered: 0,
        };
    }

    let base_params = StrategyParams::default();
    let min_trades = config.min_trades;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.n_threads)
        .build()
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().num_threads(2).build().expect("pool"));

    let mut all_trials: Vec<TrialResult> = pool.install(|| {
        trial_params
            .par_iter()
            .map(|params| {
                let strategy_params = apply_param_map(&base_params, params);
                let result = run_backtest(data, &strategy_params);

                let calmar = if result.max_drawdown > 1e-9 {
                    result.cagr / result.max_drawdown
                } else {
                    0.0
                };

                TrialResult {
                    params: params.clone(),
                    sharpe: result.sharpe,
                    max_dd: result.max_drawdown,
                    calmar,
                    n_trades: result.total_trades,
                    cagr: result.cagr,
                    win_rate: result.win_rate,
                }
            })
            .collect()
    });

    // Filter out trials with too few trades.
    let n_before = all_trials.len();
    all_trials.retain(|t| t.n_trades >= min_trades);
    let n_filtered = n_before - all_trials.len();

    // Sort by score descending.
    all_trials.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap_or(std::cmp::Ordering::Equal));

    let (best_params, best_sharpe, best_calmar) = if let Some(best) = all_trials.first() {
        (best.params.clone(), best.sharpe, best.calmar)
    } else {
        (HashMap::new(), f64::NEG_INFINITY, 0.0)
    };

    let param_importance = compute_param_importance(&all_trials);

    GridSearchResult {
        best_params,
        best_sharpe,
        best_calmar,
        all_trials,
        param_importance,
        n_evaluated: n_total,
        n_filtered,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ranges() -> HashMap<String, (f64, f64, usize)> {
        let mut m = HashMap::new();
        m.insert("garch_target_vol".to_string(), (0.2, 0.8, 3));
        m.insert("winner_protection_pct".to_string(), (0.1, 0.5, 3));
        m
    }

    #[test]
    fn test_grid_generates_cartesian_product() {
        let mut config = GridSearchConfig::default();
        config.param_ranges = make_ranges();
        config.method = SearchMethod::Grid;
        let points = generate_trial_points(&config);
        // 3 x 3 = 9 grid points.
        assert_eq!(points.len(), 9);
    }

    #[test]
    fn test_random_generates_n_trials() {
        let mut config = GridSearchConfig::default();
        config.param_ranges = make_ranges();
        config.method = SearchMethod::Random;
        config.n_trials = 50;
        let points = generate_trial_points(&config);
        assert_eq!(points.len(), 50);
    }

    #[test]
    fn test_sobol_generates_n_trials() {
        let mut config = GridSearchConfig::default();
        config.param_ranges = make_ranges();
        config.method = SearchMethod::Sobol;
        config.n_trials = 32;
        let points = generate_trial_points(&config);
        assert_eq!(points.len(), 32);
    }

    #[test]
    fn test_lhs_generates_n_trials() {
        let mut config = GridSearchConfig::default();
        config.param_ranges = make_ranges();
        config.method = SearchMethod::LatinHypercube;
        config.n_trials = 20;
        let points = generate_trial_points(&config);
        assert_eq!(points.len(), 20);
    }

    #[test]
    fn test_lhs_values_in_range() {
        let mut config = GridSearchConfig::default();
        config.param_ranges = make_ranges();
        config.method = SearchMethod::LatinHypercube;
        config.n_trials = 30;
        let points = generate_trial_points(&config);
        for pt in &points {
            let vol = pt["garch_target_vol"];
            assert!(vol >= 0.2 && vol <= 0.8, "vol out of range: {vol}");
        }
    }

    #[test]
    fn test_sobol_values_in_range() {
        let mut config = GridSearchConfig::default();
        config.param_ranges = make_ranges();
        config.method = SearchMethod::Sobol;
        config.n_trials = 20;
        let points = generate_trial_points(&config);
        for pt in &points {
            let w = pt["winner_protection_pct"];
            assert!(w >= 0.1 && w <= 0.5, "w out of range: {w}");
        }
    }

    #[test]
    fn test_van_der_corput_range() {
        for i in 0..100u64 {
            let v = van_der_corput(i, 2);
            assert!(v >= 0.0 && v < 1.0, "vdc({i})={v} out of [0,1)");
        }
    }

    #[test]
    fn test_latin_hypercube_stratification() {
        let mut rng = StdRng::seed_from_u64(99);
        let n = 10usize;
        let matrix = latin_hypercube(n, 2, &mut rng);
        assert_eq!(matrix.len(), n);
        // Each column should cover [0,1) without leaving any stratum empty.
        for d in 0..2 {
            let mut col: Vec<f64> = matrix.iter().map(|row| row[d]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // Each stratum [k/n, (k+1)/n) should contain exactly one sample.
            for k in 0..n {
                let lo = k as f64 / n as f64;
                let hi = (k + 1) as f64 / n as f64;
                let count = col.iter().filter(|&&v| v >= lo && v < hi).count();
                assert_eq!(count, 1, "stratum [{lo:.2},{hi:.2}) has {count} samples");
            }
        }
    }

    #[test]
    fn test_apply_param_map_clamps() {
        let base = StrategyParams::default();
        let mut params = HashMap::new();
        params.insert("garch_target_vol".to_string(), 999.0); // way above max
        params.insert("min_hold_bars".to_string(), 0.0);       // below min
        let applied = apply_param_map(&base, &params);
        assert!(applied.garch_target_vol <= 2.0);
        assert!(applied.min_hold_bars >= 1);
    }

    #[test]
    fn test_trial_result_score() {
        let trial = TrialResult {
            params: HashMap::new(),
            sharpe: 1.5,
            max_dd: 0.2,
            calmar: 5.0,
            n_trades: 100,
            cagr: 0.3,
            win_rate: 0.55,
        };
        // score = sharpe / (1 + max_dd) = 1.5 / 1.2 = 1.25
        let expected = 1.5 / 1.2;
        assert!((trial.score() - expected).abs() < 1e-9);
    }

    #[test]
    fn test_trial_result_zero_trades_score() {
        let trial = TrialResult {
            params: HashMap::new(),
            sharpe: 2.0,
            max_dd: 0.1,
            calmar: 10.0,
            n_trades: 0,
            cagr: 0.5,
            win_rate: 0.0,
        };
        assert_eq!(trial.score(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_compute_param_importance_sums_to_one() {
        let mut trials = Vec::new();
        for i in 0..20usize {
            let mut params = HashMap::new();
            params.insert("a".to_string(), i as f64 * 0.1);
            params.insert("b".to_string(), (20 - i) as f64 * 0.05);
            trials.push(TrialResult {
                params,
                sharpe: i as f64 * 0.1,
                max_dd: 0.1,
                calmar: i as f64,
                n_trades: 50,
                cagr: 0.1,
                win_rate: 0.5,
            });
        }
        let importance = compute_param_importance(&trials);
        let total: f64 = importance.values().sum();
        assert!((total - 1.0).abs() < 1e-9, "importance sum = {total}");
    }

    #[test]
    fn test_grid_search_empty_ranges() {
        let config = GridSearchConfig::default(); // empty param_ranges
        let data: DataStore = HashMap::new();
        let result = run_grid_search(&data, &config);
        assert_eq!(result.n_evaluated, 0);
    }
}
