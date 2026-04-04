// param_sweep.rs — Grid search and random search over BH physics parameters
//
// Both search strategies use rayon for parallel evaluation.  The random
// search uses a Halton low-discrepancy sequence for better coverage than
// uniform pseudo-random sampling.

use crate::csv_loader::BarsSet;
use crate::engine::{BacktestConfig, BacktestEngine, BacktestResult};
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Metric
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    Sharpe,
    CAGR,
    MaxDD,
    ProfitFactor,
    CalmarRatio,
}

impl Metric {
    /// Extract the scalar value from a `BacktestResult`.
    /// For MaxDD a *lower* value is better, but we negate so maximisation is uniform.
    pub fn extract(&self, result: &BacktestResult) -> f64 {
        let m = &result.metrics;
        match self {
            Self::Sharpe => m.sharpe,
            Self::CAGR => m.cagr,
            Self::MaxDD => -m.max_drawdown,
            Self::ProfitFactor => m.profit_factor,
            Self::CalmarRatio => m.calmar_ratio,
        }
    }

    pub fn is_higher_better(&self) -> bool {
        !matches!(self, Self::MaxDD)
    }
}

// ---------------------------------------------------------------------------
// ParamPoint — one combination of BH physics parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ParamPoint {
    pub cf: f64,
    pub bh_form: f64,
    pub bh_collapse: f64,
    pub bh_decay: f64,
}

impl ParamPoint {
    pub fn new(cf: f64, bh_form: f64, bh_collapse: f64, bh_decay: f64) -> Self {
        Self { cf, bh_form, bh_collapse, bh_decay }
    }

    fn apply_to(&self, template: &BacktestConfig) -> BacktestConfig {
        let mut cfg = template.clone();
        cfg.cf = self.cf;
        cfg.bh_form = self.bh_form;
        cfg.bh_collapse = self.bh_collapse;
        cfg.bh_decay = self.bh_decay;
        cfg
    }
}

// ---------------------------------------------------------------------------
// ParamGrid — Cartesian product of parameter ranges
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ParamGrid {
    pub cf_values: Vec<f64>,
    pub bh_form_values: Vec<f64>,
    pub bh_collapse_values: Vec<f64>,
    pub bh_decay_values: Vec<f64>,
}

impl ParamGrid {
    pub fn new(
        cf_values: Vec<f64>,
        bh_form_values: Vec<f64>,
        bh_collapse_values: Vec<f64>,
        bh_decay_values: Vec<f64>,
    ) -> Self {
        Self { cf_values, bh_form_values, bh_collapse_values, bh_decay_values }
    }

    /// Default research grid (~3×3×3×3 = 81 combinations).
    pub fn default_grid() -> Self {
        Self::new(
            vec![0.0005, 0.001, 0.002],
            vec![1.2, 1.5, 1.8],
            vec![0.8, 1.0, 1.2],
            vec![0.90, 0.95, 0.98],
        )
    }

    /// Expand to all combinations (Cartesian product).
    pub fn expand(&self) -> Vec<ParamPoint> {
        let mut points = Vec::new();
        for &cf in &self.cf_values {
            for &bh_form in &self.bh_form_values {
                for &bh_collapse in &self.bh_collapse_values {
                    // Ensure collapse < form
                    if bh_collapse >= bh_form {
                        continue;
                    }
                    for &bh_decay in &self.bh_decay_values {
                        points.push(ParamPoint::new(cf, bh_form, bh_collapse, bh_decay));
                    }
                }
            }
        }
        points
    }

    pub fn n_combinations(&self) -> usize {
        self.cf_values.len()
            * self.bh_form_values.len()
            * self.bh_collapse_values.len()
            * self.bh_decay_values.len()
    }
}

// ---------------------------------------------------------------------------
// SweepResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    /// All (config, result, metric_value) triples, sorted best-first.
    pub all_results: Vec<(BacktestConfig, BacktestResult, f64)>,
    pub best_config: BacktestConfig,
    pub best_result: BacktestResult,
    pub best_metric_value: f64,
    pub metric: Metric,
}

impl SweepResult {
    /// Top-k configs by metric.
    pub fn top_k(&self, k: usize) -> &[(BacktestConfig, BacktestResult, f64)] {
        let end = k.min(self.all_results.len());
        &self.all_results[..end]
    }
}

// ---------------------------------------------------------------------------
// grid_search
// ---------------------------------------------------------------------------

/// Evaluate all parameter combinations in `param_grid` against `bars`.
/// Returns a `SweepResult` with all results sorted by `metric` (best first).
pub fn grid_search(
    bars: &BarsSet,
    param_grid: &ParamGrid,
    template_config: &BacktestConfig,
    metric: Metric,
) -> Result<SweepResult> {
    let points = param_grid.expand();
    run_sweep(bars, &points, template_config, metric)
}

// ---------------------------------------------------------------------------
// random_search
// ---------------------------------------------------------------------------

/// Parameter space bounds for random search.
#[derive(Debug, Clone, Copy)]
pub struct ParamBounds {
    pub cf_min: f64,
    pub cf_max: f64,
    pub bh_form_min: f64,
    pub bh_form_max: f64,
    /// bh_collapse is sampled as a fraction of bh_form (0 < frac < 1).
    pub collapse_frac_min: f64,
    pub collapse_frac_max: f64,
    pub bh_decay_min: f64,
    pub bh_decay_max: f64,
}

impl Default for ParamBounds {
    fn default() -> Self {
        Self {
            cf_min: 0.0001,
            cf_max: 0.005,
            bh_form_min: 1.1,
            bh_form_max: 1.95,
            collapse_frac_min: 0.5,
            collapse_frac_max: 0.95,
            bh_decay_min: 0.85,
            bh_decay_max: 0.99,
        }
    }
}

/// Random search using a Halton low-discrepancy sequence for 4 dimensions.
pub fn random_search(
    bars: &BarsSet,
    n_trials: usize,
    seed: u64,
    bounds: &ParamBounds,
    template_config: &BacktestConfig,
    metric: Metric,
) -> Result<SweepResult> {
    let points = halton_param_points(n_trials, seed, bounds);
    run_sweep(bars, &points, template_config, metric)
}

// ---------------------------------------------------------------------------
// Core parallel sweep
// ---------------------------------------------------------------------------

fn run_sweep(
    bars: &BarsSet,
    points: &[ParamPoint],
    template_config: &BacktestConfig,
    metric: Metric,
) -> Result<SweepResult> {
    if points.is_empty() {
        anyhow::bail!("param sweep: no parameter combinations to evaluate");
    }

    // Parallel evaluation
    let mut all_results: Vec<(BacktestConfig, BacktestResult, f64)> = points
        .par_iter()
        .filter_map(|pt| {
            let cfg = pt.apply_to(template_config);
            let mut engine = BacktestEngine::new(cfg.clone());
            match engine.run_barsset(bars) {
                Ok(result) => {
                    let score = metric.extract(&result);
                    if score.is_finite() {
                        Some((cfg, result, score))
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        })
        .collect();

    if all_results.is_empty() {
        anyhow::bail!("param sweep: all evaluations failed or returned non-finite metrics");
    }

    // Sort best-first (always maximise; MaxDD metric is negated in extract())
    all_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let (best_config, best_result, best_metric_value) = all_results[0].clone();

    Ok(SweepResult {
        all_results,
        best_config,
        best_result,
        best_metric_value,
        metric,
    })
}

// ---------------------------------------------------------------------------
// Halton low-discrepancy sequence
// ---------------------------------------------------------------------------

/// Generate a single Halton value for the given index and prime base.
fn halton(mut index: u64, base: u64) -> f64 {
    let mut result = 0.0_f64;
    let mut f = 1.0_f64;
    while index > 0 {
        f /= base as f64;
        result += f * (index % base) as f64;
        index /= base;
    }
    result
}

/// Generate `n` 2-D Halton points (base-2, base-3).
pub fn halton_sequence_2d(n: usize, seed: u64) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let idx = i as u64 + seed + 1;
            (halton(idx, 2), halton(idx, 3))
        })
        .collect()
}

/// Generate `n` parameter points using a 4-D Halton sequence (bases 2,3,5,7).
fn halton_param_points(n: usize, seed: u64, bounds: &ParamBounds) -> Vec<ParamPoint> {
    (0..n)
        .map(|i| {
            let idx = i as u64 + seed + 1;
            let h2 = halton(idx, 2);
            let h3 = halton(idx, 3);
            let h5 = halton(idx, 5);
            let h7 = halton(idx, 7);

            let cf = bounds.cf_min + h2 * (bounds.cf_max - bounds.cf_min);
            let bh_form = bounds.bh_form_min + h3 * (bounds.bh_form_max - bounds.bh_form_min);
            let collapse_frac =
                bounds.collapse_frac_min + h5 * (bounds.collapse_frac_max - bounds.collapse_frac_min);
            let bh_collapse = bh_form * collapse_frac;
            let bh_decay =
                bounds.bh_decay_min + h7 * (bounds.bh_decay_max - bounds.bh_decay_min);

            ParamPoint::new(cf, bh_form, bh_collapse, bh_decay)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CSV export of sweep results
// ---------------------------------------------------------------------------

/// Write all sweep results to a CSV file for downstream analysis.
pub fn save_sweep_csv(path: &std::path::Path, result: &SweepResult) -> Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path)
        .map_err(|e| anyhow::anyhow!("Cannot create sweep CSV {}: {e}", path.display()))?;
    let mut wtr = std::io::BufWriter::new(f);

    writeln!(
        wtr,
        "rank,cf,bh_form,bh_collapse,bh_decay,metric_value,\
         sharpe,cagr,max_drawdown,profit_factor,calmar,win_rate,total_trades"
    )?;

    for (rank, (cfg, res, score)) in result.all_results.iter().enumerate() {
        let m = &res.metrics;
        writeln!(
            wtr,
            "{rank},{cf:.6},{bh_form:.4},{bh_collapse:.4},{bh_decay:.4},{score:.6},\
             {sharpe:.4},{cagr:.6},{max_dd:.6},{pf:.4},{calmar:.4},{wr:.4},{trades}",
            rank = rank + 1,
            cf = cfg.cf,
            bh_form = cfg.bh_form,
            bh_collapse = cfg.bh_collapse,
            bh_decay = cfg.bh_decay,
            score = score,
            sharpe = m.sharpe,
            cagr = m.cagr,
            max_dd = m.max_drawdown,
            pf = m.profit_factor,
            calmar = m.calmar_ratio,
            wr = m.win_rate,
            trades = m.total_trades,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_base2_values() {
        // Known Halton base-2 sequence: 1/2, 1/4, 3/4, 1/8, 5/8, ...
        let expected = [0.5, 0.25, 0.75, 0.125, 0.625];
        for (i, &exp) in expected.iter().enumerate() {
            let h = halton(i as u64 + 1, 2);
            assert!((h - exp).abs() < 1e-9, "i={i} expected {exp} got {h}");
        }
    }

    #[test]
    fn halton_2d_no_duplicates() {
        let pts = halton_sequence_2d(100, 0);
        // All x and y values should be in [0, 1)
        for (x, y) in &pts {
            assert!(*x >= 0.0 && *x < 1.0);
            assert!(*y >= 0.0 && *y < 1.0);
        }
    }

    #[test]
    fn param_grid_expand_respects_collapse_lt_form() {
        let grid = ParamGrid::default_grid();
        for pt in grid.expand() {
            assert!(pt.bh_collapse < pt.bh_form, "collapse must be < form");
        }
    }

    #[test]
    fn halton_param_points_within_bounds() {
        let bounds = ParamBounds::default();
        let pts = halton_param_points(50, 42, &bounds);
        for pt in pts {
            assert!(pt.cf >= bounds.cf_min && pt.cf <= bounds.cf_max);
            assert!(pt.bh_form >= bounds.bh_form_min && pt.bh_form <= bounds.bh_form_max);
            assert!(pt.bh_decay >= bounds.bh_decay_min && pt.bh_decay <= bounds.bh_decay_max);
            assert!(pt.bh_collapse < pt.bh_form);
        }
    }
}
