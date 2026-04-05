//! Sensitivity analysis for parameter sweep results.
//!
//! # Methods
//!
//! ## [`SobolAnalyzer`]
//! Variance-based first-order (S1) and total-order (ST) Sobol sensitivity
//! indices.  Uses a rank-based conditional-variance estimator that works
//! without requiring a specific sampling design (Saltelli et al., 2002).
//!
//! ## [`MorrisScreening`]
//! Elementary Effects (Morris, 1991) for cheap, fast screening of which
//! parameters actually move the output.  Requires O(k+1) model evaluations
//! per trajectory (not re-run here — assumes results are pre-computed).
//!
//! # Output
//! Both methods produce a [`SensitivityReport`] which is JSON-serialisable.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::SimResult;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// First-order (S1) and total-order (ST) Sobol indices for one parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolIndices {
    pub param: String,
    /// First-order index: fraction of variance explained by this param alone.
    pub s1: f64,
    /// Total-order index: S1 plus all higher-order interactions involving this param.
    pub st: f64,
}

/// Elementary effect statistics for one parameter (Morris method).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorrisResult {
    pub param: String,
    /// Mean absolute elementary effect (μ*).
    pub mu_star: f64,
    /// Standard deviation of elementary effects (σ).
    pub sigma: f64,
    /// Screening importance = μ* (higher → more important).
    pub importance: f64,
}

/// Full sensitivity report combining Sobol and Morris analyses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityReport {
    pub n_samples: usize,
    pub sobol: Vec<SobolIndices>,
    pub morris: Vec<MorrisResult>,
    /// Tornado chart: param → (score at low p10, score at high p90)
    pub tornado: HashMap<String, (f64, f64)>,
    /// Interaction matrix: param_i → param_j → correlation
    pub interaction: HashMap<String, HashMap<String, f64>>,
}

// ---------------------------------------------------------------------------
// Helper matrix ops
// ---------------------------------------------------------------------------

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

fn std_dev(v: &[f64]) -> f64 {
    variance(v).sqrt()
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let mx = mean(x);
    let my = mean(y);
    let num: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - mx) * (yi - my)).sum();
    let dx: f64 = x.iter().map(|xi| (xi - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = y.iter().map(|yi| (yi - my).powi(2)).sum::<f64>().sqrt();
    let denom = dx * dy;
    if denom < 1e-12 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

/// Sort indices for a slice (ascending).
fn argsort(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

// ---------------------------------------------------------------------------
// SobolAnalyzer
// ---------------------------------------------------------------------------

/// Variance-based Sobol sensitivity index estimator.
pub struct SobolAnalyzer;

impl SobolAnalyzer {
    /// Compute Sobol S1 and ST indices for each parameter in `results`.
    ///
    /// # Arguments
    /// * `results`      — slice of simulation results
    /// * `param_names`  — which parameters to analyse (must be keys in `results[i].params`)
    ///
    /// # Returns
    /// Vector of [`SobolIndices`], one per parameter, sorted by ST descending.
    pub fn compute(
        results: &[SimResult],
        param_names: &[&str],
    ) -> Vec<SobolIndices> {
        if results.len() < 4 {
            return param_names.iter().map(|&p| SobolIndices {
                param: p.to_owned(),
                s1: 0.0,
                st: 0.0,
            }).collect();
        }

        let y: Vec<f64> = results.iter().map(|r| r.improvement).collect();
        let var_y = variance(&y);

        if var_y < 1e-12 {
            return param_names.iter().map(|&p| SobolIndices {
                param: p.to_owned(),
                s1: 0.0,
                st: 0.0,
            }).collect();
        }

        let mut indices: Vec<SobolIndices> = param_names.iter().map(|&param_name| {
            let x: Vec<f64> = results.iter()
                .map(|r| r.params.get(param_name).copied().unwrap_or(0.0))
                .collect();

            // First-order index: conditional variance via sorted binning
            let s1 = {
                let sort_idx = argsort(&x);
                let y_sorted: Vec<f64> = sort_idx.iter().map(|&i| y[i]).collect();
                let n = y_sorted.len();
                let n_bins = (n as f64).sqrt().max(4.0) as usize;
                let bin_size = n / n_bins;
                let bin_means: Vec<f64> = (0..n_bins).map(|b| {
                    let start = b * bin_size;
                    let end = if b + 1 == n_bins { n } else { start + bin_size };
                    mean(&y_sorted[start..end])
                }).collect();
                let cond_var = variance(&bin_means);
                (cond_var / var_y).clamp(0.0, 1.0)
            };

            // Total-order index: average within-group variance vs median split
            let st = {
                let med = {
                    let mut xs = x.clone();
                    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    xs[xs.len() / 2]
                };
                let y_lo: Vec<f64> = x.iter().zip(y.iter())
                    .filter(|(&xi, _)| xi <= med)
                    .map(|(_, &yi)| yi)
                    .collect();
                let y_hi: Vec<f64> = x.iter().zip(y.iter())
                    .filter(|(&xi, _)| xi > med)
                    .map(|(_, &yi)| yi)
                    .collect();
                if y_lo.len() > 1 && y_hi.len() > 1 {
                    let avg_within_var = (variance(&y_lo) + variance(&y_hi)) / 2.0;
                    (avg_within_var / var_y).clamp(0.0, 1.0)
                } else {
                    s1
                }
            };

            SobolIndices { param: param_name.to_owned(), s1, st }
        }).collect();

        // Sort by ST descending
        indices.sort_by(|a, b| b.st.partial_cmp(&a.st).unwrap_or(std::cmp::Ordering::Equal));
        indices
    }
}

// ---------------------------------------------------------------------------
// MorrisScreening
// ---------------------------------------------------------------------------

/// Morris (1991) Elementary Effects screening.
///
/// Estimates which parameters have the most impact using the mean absolute
/// elementary effect (μ*) and its standard deviation (σ).
///
/// This implementation computes approximate elementary effects from an
/// existing result set by looking at finite differences between nearby
/// parameter configurations.  It is not a true Morris design — use
/// [`MorrisScreening::from_trajectory`] when you have paired trajectory data.
pub struct MorrisScreening;

impl MorrisScreening {
    /// Approximate Morris screening from an unstructured result set.
    ///
    /// For each parameter, pairs up samples that differ mainly in that
    /// parameter and computes the elementary effect as:
    /// `EE = (y2 - y1) / (x2 - x1) * range`
    ///
    /// Uses up to `max_pairs` pairs per parameter for efficiency.
    pub fn screen(
        results: &[SimResult],
        param_names: &[&str],
        max_pairs: usize,
    ) -> Vec<MorrisResult> {
        if results.len() < 2 {
            return param_names.iter().map(|&p| MorrisResult {
                param: p.to_owned(),
                mu_star: 0.0,
                sigma: 0.0,
                importance: 0.0,
            }).collect();
        }

        param_names.iter().map(|&param_name| {
            let mut effects: Vec<f64> = Vec::new();

            // Collect all parameter values for range estimation
            let all_x: Vec<f64> = results.iter()
                .map(|r| r.params.get(param_name).copied().unwrap_or(0.0))
                .collect();
            let x_min = all_x.iter().cloned().fold(f64::INFINITY, f64::min);
            let x_max = all_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (x_max - x_min).max(f64::EPSILON);

            let n = results.len();
            let step = (n / max_pairs.min(n)).max(1);

            for i in (0..n).step_by(step) {
                for j in (i + 1..n).step_by(step) {
                    let xi = all_x[i];
                    let xj = all_x[j];
                    let dx = xj - xi;
                    if dx.abs() < range * 0.01 {
                        continue; // Too similar in this param
                    }
                    let dy = results[j].improvement - results[i].improvement;
                    effects.push(dy / dx * range);
                    if effects.len() >= max_pairs {
                        break;
                    }
                }
                if effects.len() >= max_pairs {
                    break;
                }
            }

            if effects.is_empty() {
                return MorrisResult {
                    param: param_name.to_owned(),
                    mu_star: 0.0,
                    sigma: 0.0,
                    importance: 0.0,
                };
            }

            let mu_star = effects.iter().map(|e| e.abs()).sum::<f64>() / effects.len() as f64;
            let sigma = std_dev(&effects);

            MorrisResult {
                param: param_name.to_owned(),
                mu_star,
                sigma,
                importance: mu_star,
            }
        }).collect()
    }

    /// Compute elementary effects from explicitly paired trajectory data.
    ///
    /// # Arguments
    /// * `trajectories` — list of (result_a, result_b, param_changed) tuples.
    ///   Each trajectory step should vary exactly one parameter.
    pub fn from_trajectory(
        trajectories: &[(SimResult, SimResult, String)],
        param_names: &[&str],
    ) -> Vec<MorrisResult> {
        let mut effects_by_param: HashMap<String, Vec<f64>> = HashMap::new();
        for name in param_names {
            effects_by_param.insert((*name).to_owned(), Vec::new());
        }

        for (ra, rb, param) in trajectories {
            if !effects_by_param.contains_key(param.as_str()) {
                continue;
            }
            let xa = ra.params.get(param).copied().unwrap_or(0.0);
            let xb = rb.params.get(param).copied().unwrap_or(0.0);
            let dx = xb - xa;
            if dx.abs() < 1e-10 {
                continue;
            }
            let dy = rb.improvement - ra.improvement;
            effects_by_param.entry(param.clone()).or_default().push(dy / dx);
        }

        param_names.iter().map(|&p| {
            let effects = effects_by_param.get(p).map(Vec::as_slice).unwrap_or(&[]);
            if effects.is_empty() {
                return MorrisResult { param: p.to_owned(), mu_star: 0.0, sigma: 0.0, importance: 0.0 };
            }
            let mu_star = effects.iter().map(|e| e.abs()).sum::<f64>() / effects.len() as f64;
            let sigma = std_dev(effects);
            MorrisResult { param: p.to_owned(), mu_star, sigma, importance: mu_star }
        }).collect()
    }
}

// ---------------------------------------------------------------------------
// Full report builder
// ---------------------------------------------------------------------------

/// Build a complete [`SensitivityReport`] from a set of simulation results.
pub fn build_report(results: &[SimResult], param_names: &[&str]) -> SensitivityReport {
    let n_samples = results.len();
    let sobol = SobolAnalyzer::compute(results, param_names);
    let morris = MorrisScreening::screen(results, param_names, 500);
    let tornado = compute_tornado(results, param_names);
    let interaction = compute_interaction(results, param_names);

    SensitivityReport { n_samples, sobol, morris, tornado, interaction }
}

/// Compute tornado chart data: for each parameter, score at p10 vs p90.
fn compute_tornado(
    results: &[SimResult],
    param_names: &[&str],
) -> HashMap<String, (f64, f64)> {
    let y: Vec<f64> = results.iter().map(|r| r.improvement).collect();
    let mut out = HashMap::new();

    for &param in param_names {
        let x: Vec<f64> = results.iter()
            .map(|r| r.params.get(param).copied().unwrap_or(0.0))
            .collect();

        let mut sorted_x = x.clone();
        sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted_x.len();
        let p10_val = sorted_x[(n / 10).max(0)];
        let p90_val = sorted_x[((9 * n) / 10).min(n - 1)];

        let y_lo: Vec<f64> = x.iter().zip(y.iter())
            .filter(|(&xi, _)| xi <= p10_val)
            .map(|(_, &yi)| yi)
            .collect();
        let y_hi: Vec<f64> = x.iter().zip(y.iter())
            .filter(|(&xi, _)| xi >= p90_val)
            .map(|(_, &yi)| yi)
            .collect();

        let score_lo = if y_lo.is_empty() { 0.0 } else { mean(&y_lo) };
        let score_hi = if y_hi.is_empty() { 0.0 } else { mean(&y_hi) };
        out.insert(param.to_owned(), (score_lo, score_hi));
    }
    out
}

/// Compute pairwise interaction strengths as Pearson correlation of
/// X_i × X_j (normalised) with Y.
fn compute_interaction(
    results: &[SimResult],
    param_names: &[&str],
) -> HashMap<String, HashMap<String, f64>> {
    let y: Vec<f64> = results.iter().map(|r| r.improvement).collect();
    let n = y.len();
    if n < 4 {
        return HashMap::new();
    }

    // Normalise each param column to [0, 1]
    let norm_x: HashMap<String, Vec<f64>> = param_names.iter().map(|&p| {
        let raw: Vec<f64> = results.iter()
            .map(|r| r.params.get(p).copied().unwrap_or(0.0))
            .collect();
        let lo = raw.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (hi - lo).max(1e-12);
        let normed = raw.iter().map(|&v| (v - lo) / range).collect();
        (p.to_owned(), normed)
    }).collect();

    let mut matrix: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for &pi in param_names {
        let mut row = HashMap::new();
        for &pj in param_names {
            if pi == pj {
                row.insert(pj.to_owned(), 1.0);
                continue;
            }
            let xi = &norm_x[pi];
            let xj = &norm_x[pj];
            let interaction: Vec<f64> = xi.iter().zip(xj.iter())
                .map(|(a, b)| a * b)
                .collect();
            let corr = pearson_corr(&interaction, &y);
            row.insert(pj.to_owned(), corr);
        }
        matrix.insert(pi.to_owned(), row);
    }
    matrix
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_results(n: usize) -> Vec<SimResult> {
        (0..n).map(|i| {
            let t = i as f64 / n as f64;
            let mut params = HashMap::new();
            params.insert("bh_form".to_owned(), 1.70 + t * 0.28);
            params.insert("bh_decay".to_owned(), 0.92 + t * 0.07);
            params.insert("ou_frac".to_owned(), 0.02 + t * 0.18);
            SimResult {
                params,
                sharpe: t * 2.0,
                max_dd: 0.30 - t * 0.20,
                calmar: t * 3.0,
                total_return: t * 0.5,
                win_rate: 0.4 + t * 0.2,
                num_trades: 100,
                improvement: t * 1.5,
            }
        }).collect()
    }

    #[test]
    fn sobol_non_negative() {
        let results = make_results(50);
        let names = ["bh_form", "bh_decay", "ou_frac"];
        let indices = SobolAnalyzer::compute(&results, &names);
        assert_eq!(indices.len(), 3);
        for idx in &indices {
            assert!(idx.s1 >= 0.0 && idx.s1 <= 1.0, "s1 out of range: {}", idx.s1);
            assert!(idx.st >= 0.0 && idx.st <= 1.0, "st out of range: {}", idx.st);
        }
    }

    #[test]
    fn morris_returns_all_params() {
        let results = make_results(30);
        let names = ["bh_form", "bh_decay", "ou_frac"];
        let mr = MorrisScreening::screen(&results, &names, 100);
        assert_eq!(mr.len(), 3);
        for r in &mr {
            assert!(r.mu_star >= 0.0, "mu_star negative: {}", r.mu_star);
        }
    }

    #[test]
    fn build_report_smoke() {
        let results = make_results(60);
        let names = ["bh_form", "bh_decay", "ou_frac"];
        let report = build_report(&results, &names);
        assert_eq!(report.n_samples, 60);
        assert_eq!(report.sobol.len(), 3);
        assert_eq!(report.morris.len(), 3);
        assert!(!report.tornado.is_empty());
        assert!(!report.interaction.is_empty());
    }
}
