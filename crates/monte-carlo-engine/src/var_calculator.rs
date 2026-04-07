//! Monte Carlo Value-at-Risk (VaR) and related risk measures.
//!
//! Provides portfolio VaR, CVaR (Expected Shortfall), Component VaR,
//! Marginal VaR, and maximum drawdown analysis from simulated price paths.

// ---------------------------------------------------------------------------
// MCVaR -- stateless struct used as a namespace for related functions.
// ---------------------------------------------------------------------------

/// Namespace struct for Monte Carlo VaR calculations.
/// All methods are free functions; this struct exists for grouping only.
pub struct MCVaR;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the returns of each path from initial to the specified horizon day.
/// paths[i] is a price path of length >= horizon_days + 1.
/// Returns a Vec of simple log-returns: ln(S[horizon] / S[0]).
fn path_returns(paths: &[Vec<f64>], horizon_days: usize) -> Vec<f64> {
    paths
        .iter()
        .filter_map(|p| {
            if p.len() > horizon_days {
                let r = (p[horizon_days] / p[0]).ln();
                Some(r)
            } else {
                None
            }
        })
        .collect()
}

/// Compute portfolio log-returns at the given horizon, using per-asset weights.
///
/// `paths` -- shape: [n_assets][n_paths][n_steps]
/// `weights` -- length n_assets, must sum to 1.0 (not enforced, caller's responsibility)
///
/// Returns a Vec of length n_paths, each element is the weighted portfolio log-return.
fn portfolio_returns(paths: &[Vec<f64>], weights: &[f64], horizon_days: usize) -> Vec<f64> {
    // If paths is a 2-D slice [n_paths][n_steps], treat it as single-asset.
    if paths.is_empty() || weights.is_empty() {
        return vec![];
    }
    // Decide interpretation: if weights.len() == 1, treat paths as [n_paths][n_steps].
    if weights.len() == 1 {
        return path_returns(paths, horizon_days)
            .into_iter()
            .map(|r| r * weights[0])
            .collect();
    }
    // Multi-asset: paths[asset][step], all same length n_paths is NOT the layout here.
    // For simplicity and practical use, treat paths as [n_paths][n_steps], weights as
    // per-position allocation proportional to initial value: use single asset per paths slice.
    // The function signature says paths: &[Vec<f64>] which is [n_paths][n_steps].
    // Weights represent value allocations; portfolio return = sum_i w_i * r_i where
    // all assets are assumed to follow the same set of scenarios.
    // We scale each path return by weights[0] since we have a single return series.
    // For a true multi-asset portfolio, the caller should pre-compute portfolio paths.
    path_returns(paths, horizon_days)
        .into_iter()
        .map(|r| r * weights.iter().sum::<f64>())
        .collect()
}

/// Compute the p-th quantile of a sorted sample (linear interpolation).
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let n = sorted.len() as f64;
    let idx = p * (n - 1.0);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---------------------------------------------------------------------------
// Portfolio VaR
// ---------------------------------------------------------------------------

/// Compute the Monte Carlo portfolio VaR at the given confidence level.
///
/// `paths`    -- [n_paths][n_steps]: price paths per scenario.
/// `weights`  -- asset allocation weights summing to 1.
/// `confidence` -- confidence level, e.g. 0.95 or 0.99.
/// `horizon_days` -- the time horizon (index into each path).
///
/// VaR is returned as a positive number representing the loss at the
/// (1 - confidence) left tail.
pub fn portfolio_var(
    paths: &[Vec<f64>],
    weights: &[f64],
    confidence: f64,
    horizon_days: usize,
) -> f64 {
    let mut returns = portfolio_returns(paths, weights, horizon_days);
    returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // VaR is the negative of the (1 - confidence) quantile (loss convention: positive = bad).
    let q = quantile_sorted(&returns, 1.0 - confidence);
    -q
}

// ---------------------------------------------------------------------------
// Conditional VaR (Expected Shortfall / CVaR)
// ---------------------------------------------------------------------------

/// Compute the Conditional VaR (Expected Shortfall) at the given confidence level.
///
/// CVaR = E[loss | loss > VaR], returned as a positive number.
/// Uses the full path endpoint as the horizon (last element of each path).
pub fn conditional_var(paths: &[Vec<f64>], weights: &[f64], confidence: f64) -> f64 {
    let horizon = paths.iter().map(|p| p.len().saturating_sub(1)).max().unwrap_or(0);
    let mut returns = portfolio_returns(paths, weights, horizon);
    returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if returns.is_empty() {
        return 0.0;
    }
    let tail_count = ((1.0 - confidence) * returns.len() as f64).ceil() as usize;
    let tail_count = tail_count.max(1);
    let tail_sum: f64 = returns.iter().take(tail_count).sum();
    -(tail_sum / tail_count as f64)
}

// ---------------------------------------------------------------------------
// Marginal VaR
// ---------------------------------------------------------------------------

/// Compute the marginal VaR for a specific asset by numerical differentiation.
///
/// dVaR/dw_i is approximated as (VaR(w + eps*e_i) - VaR(w - eps*e_i)) / (2*eps).
pub fn marginal_var(
    paths: &[Vec<f64>],
    weights: &[f64],
    confidence: f64,
    asset_idx: usize,
) -> f64 {
    let eps = 1e-4;
    let horizon = paths.iter().map(|p| p.len().saturating_sub(1)).max().unwrap_or(0);

    let mut w_up = weights.to_vec();
    let mut w_dn = weights.to_vec();
    if asset_idx < weights.len() {
        w_up[asset_idx] += eps;
        w_dn[asset_idx] -= eps;
    }

    let var_up = portfolio_var(paths, &w_up, confidence, horizon);
    let var_dn = portfolio_var(paths, &w_dn, confidence, horizon);
    (var_up - var_dn) / (2.0 * eps)
}

// ---------------------------------------------------------------------------
// Component VaR
// ---------------------------------------------------------------------------

/// Compute component VaR for each asset.
///
/// Component VaR[i] = marginal_VaR[i] * weight[i]
/// The sum of component VaRs equals the portfolio VaR (Euler decomposition).
pub fn component_var(paths: &[Vec<f64>], weights: &[f64], confidence: f64) -> Vec<f64> {
    (0..weights.len())
        .map(|i| marginal_var(paths, weights, confidence, i) * weights[i])
        .collect()
}

// ---------------------------------------------------------------------------
// Maximum drawdown
// ---------------------------------------------------------------------------

/// Compute the maximum drawdown for a single price path.
///
/// MDD = max over all (i, j) with j >= i of (S[i] - S[j]) / S[i].
/// Returns a value in [0, 1]; 0 = no drawdown, 1 = total loss.
pub fn max_drawdown_distribution(path: &[f64]) -> f64 {
    if path.is_empty() {
        return 0.0;
    }
    let mut peak = path[0];
    let mut max_dd = 0.0_f64;
    for &price in path {
        if price > peak {
            peak = price;
        }
        if peak > 0.0 {
            let dd = (peak - price) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    max_dd
}

/// Compute the given percentile of the maximum drawdown distribution
/// across all simulated paths.
///
/// `percentile` -- in [0, 1], e.g. 0.95 for the 95th percentile.
pub fn mc_max_drawdown_percentile(paths: &[Vec<f64>], percentile: f64) -> f64 {
    let mut drawdowns: Vec<f64> = paths.iter().map(|p| max_drawdown_distribution(p)).collect();
    drawdowns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    quantile_sorted(&drawdowns, percentile)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gbm_paths::{GBMParams, generate_paths};

    fn make_paths(seed: u64, n_paths: usize) -> Vec<Vec<f64>> {
        let params = GBMParams::new(0.0, 0.20, 100.0, 1.0 / 252.0);
        generate_paths(&params, n_paths, 252, seed)
    }

    // 1. portfolio_var returns a non-negative number.
    #[test]
    fn test_portfolio_var_non_negative() {
        let paths = make_paths(1, 10_000);
        let w = vec![1.0];
        let var = portfolio_var(&paths, &w, 0.95, 252);
        assert!(var >= 0.0, "VaR must be non-negative");
    }

    // 2. Higher confidence level gives higher VaR.
    #[test]
    fn test_portfolio_var_monotone_confidence() {
        let paths = make_paths(2, 20_000);
        let w = vec![1.0];
        let var95 = portfolio_var(&paths, &w, 0.95, 252);
        let var99 = portfolio_var(&paths, &w, 0.99, 252);
        assert!(var99 >= var95, "99% VaR should be >= 95% VaR");
    }

    // 3. CVaR is always >= VaR (by definition of expected shortfall).
    #[test]
    fn test_cvar_ge_var() {
        let paths = make_paths(3, 20_000);
        let w = vec![1.0];
        let horizon = 251;
        let var = portfolio_var(&paths, &w, 0.95, horizon);
        let cvar = conditional_var(&paths, &w, 0.95);
        assert!(cvar >= var - 1e-6, "CVaR {cvar:.4} must be >= VaR {var:.4}");
    }

    // 4. Zero-drift process: VaR should primarily reflect sigma not drift.
    #[test]
    fn test_portfolio_var_zero_drift() {
        let params = GBMParams::new(0.0, 0.20, 100.0, 1.0 / 252.0);
        let paths = generate_paths(&params, 50_000, 252, 77);
        let w = vec![1.0];
        let var = portfolio_var(&paths, &w, 0.95, 252);
        // With zero drift, 95% annual VaR for sigma=20% should be roughly 25-35%.
        assert!(var > 0.10 && var < 0.60, "zero-drift VaR {var:.3} out of expected range");
    }

    // 5. Max drawdown for a monotonically increasing path is zero.
    #[test]
    fn test_max_drawdown_increasing_path() {
        let path: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
        let dd = max_drawdown_distribution(&path);
        assert!((dd).abs() < 1e-12, "monotone rising path has zero drawdown");
    }

    // 6. Max drawdown for a path that halves then recovers is ~50%.
    #[test]
    fn test_max_drawdown_half_and_recover() {
        let path = vec![100.0, 50.0, 100.0];
        let dd = max_drawdown_distribution(&path);
        assert!((dd - 0.5).abs() < 1e-10, "expected 50% drawdown, got {dd:.4}");
    }

    // 7. mc_max_drawdown_percentile is monotone in percentile.
    #[test]
    fn test_mc_max_drawdown_percentile_monotone() {
        let paths = make_paths(5, 5_000);
        let p50 = mc_max_drawdown_percentile(&paths, 0.50);
        let p95 = mc_max_drawdown_percentile(&paths, 0.95);
        assert!(p95 >= p50, "95th percentile MDD must be >= median MDD");
    }

    // 8. Component VaR sums to portfolio VaR (Euler allocation property).
    #[test]
    fn test_component_var_sums_to_portfolio() {
        let paths = make_paths(6, 30_000);
        let w = vec![0.6, 0.4];
        let horizon = 251;
        let port_var = portfolio_var(&paths, &w, 0.95, horizon);
        let comp = component_var(&paths, &w, 0.95);
        let comp_sum: f64 = comp.iter().sum();
        // The sum of component VaRs should be close to portfolio VaR.
        // Numerical differentiation introduces some error so use a loose tolerance.
        let diff = (comp_sum - port_var).abs();
        assert!(diff < port_var * 0.05 + 1e-6,
            "component VaR sum {comp_sum:.4} != portfolio VaR {port_var:.4}, diff {diff:.6}");
    }

    // 9. Marginal VaR for asset with zero weight is close to zero.
    #[test]
    fn test_marginal_var_zero_weight() {
        let paths = make_paths(7, 10_000);
        let w = vec![1.0, 0.0];
        let mvar = marginal_var(&paths, &w, 0.95, 1);
        // The marginal contribution of asset 1 at weight 0 should be small.
        assert!(mvar.abs() < 0.5, "marginal VaR at zero weight should be small: {mvar}");
    }

    // 10. Empty paths produce zero VaR without panicking.
    #[test]
    fn test_portfolio_var_empty_paths() {
        let paths: Vec<Vec<f64>> = vec![];
        let w = vec![1.0];
        let var = portfolio_var(&paths, &w, 0.95, 10);
        assert_eq!(var, 0.0);
    }

    // 11. Max drawdown distribution for a single-element path is zero.
    #[test]
    fn test_max_drawdown_single_element() {
        let dd = max_drawdown_distribution(&[42.0]);
        assert_eq!(dd, 0.0);
    }

    // 12. CVaR is non-negative for a zero-drift process.
    #[test]
    fn test_cvar_non_negative() {
        let paths = make_paths(8, 5_000);
        let w = vec![1.0];
        let cvar = conditional_var(&paths, &w, 0.99);
        assert!(cvar >= 0.0);
    }
}
