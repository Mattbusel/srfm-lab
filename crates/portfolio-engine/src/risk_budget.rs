// risk_budget.rs -- Risk budgeting framework
// Provides equal-risk contribution, signal-weighted, and regime-adjusted
// risk budget construction, plus realized contribution diagnostics.

use std::collections::HashMap;

/// Regime state used to adjust risk budgets.
#[derive(Debug, Clone, PartialEq)]
pub enum RegimeState {
    /// Low-volatility bull market.
    BullLowVol,
    /// High-volatility bull market (momentum regime).
    BullHighVol,
    /// Low-volatility bear market (mean-reverting range).
    BearLowVol,
    /// High-volatility bear market / crisis.
    BearHighVol,
    /// Neutral -- no strong regime signal.
    Neutral,
}

/// Per-asset risk allocation as a fraction of total portfolio risk.
/// Entries should sum to 1.0.
#[derive(Debug, Clone)]
pub struct RiskBudget {
    /// Symbol -> fractional risk budget.
    pub allocations: HashMap<String, f64>,
}

impl RiskBudget {
    /// Construct a RiskBudget from parallel symbol and weight vectors.
    pub fn from_vecs(symbols: &[String], budgets: &[f64]) -> Self {
        assert_eq!(symbols.len(), budgets.len(), "symbols and budgets must have equal length");
        let allocations = symbols
            .iter()
            .zip(budgets.iter())
            .map(|(s, &b)| (s.clone(), b))
            .collect();
        RiskBudget { allocations }
    }

    /// Normalize allocations so they sum to 1.0.
    pub fn normalize(&mut self) {
        let total: f64 = self.allocations.values().sum();
        if total > 1e-12 {
            for v in self.allocations.values_mut() {
                *v /= total;
            }
        }
    }

    /// Return allocations as an ordered vector given a symbol ordering.
    pub fn to_vec(&self, symbols: &[String]) -> Vec<f64> {
        symbols
            .iter()
            .map(|s| self.allocations.get(s).copied().unwrap_or(0.0))
            .collect()
    }
}

/// Risk budget optimizer -- constructs target risk budgets and evaluates realized contributions.
pub struct RiskBudgetOptimizer;

impl RiskBudgetOptimizer {
    // -----------------------------------------------------------------------
    // Budget construction
    // -----------------------------------------------------------------------

    /// Equal risk contribution: each asset gets 1/n of total risk budget.
    pub fn equal_risk(n_assets: usize) -> Vec<f64> {
        if n_assets == 0 {
            return Vec::new();
        }
        vec![1.0 / n_assets as f64; n_assets]
    }

    /// Signal-weighted budget: risk proportional to signal strength.
    ///
    /// All signal_strengths must be >= 0. Negatives are clamped to zero.
    /// If all signals are zero, falls back to equal risk.
    pub fn signal_weighted_budget(signal_strengths: &[f64]) -> Vec<f64> {
        let clamped: Vec<f64> = signal_strengths.iter().map(|&s| s.max(0.0)).collect();
        let total: f64 = clamped.iter().sum();
        if total < 1e-12 {
            return Self::equal_risk(signal_strengths.len());
        }
        clamped.iter().map(|&s| s / total).collect()
    }

    /// Regime-adjusted budget: scale base budgets based on current regime.
    ///
    /// Scaling rules:
    /// - BullLowVol:  full budgets (no change)
    /// - BullHighVol: concentrate top-budget assets (apply softmax-like sharpening)
    /// - BearLowVol:  more equal distribution (flatten toward 1/n)
    /// - BearHighVol: defensive -- flatten heavily and reduce top allocations
    /// - Neutral:     mild flattening
    ///
    /// Result is re-normalized to sum to 1.0.
    pub fn regime_adjusted_budget(base_budget: &[f64], regime: &RegimeState) -> Vec<f64> {
        let n = base_budget.len();
        if n == 0 {
            return Vec::new();
        }
        let equal = 1.0 / n as f64;

        let adjusted: Vec<f64> = match regime {
            RegimeState::BullLowVol => {
                // Full budgets unchanged.
                base_budget.to_vec()
            }
            RegimeState::BullHighVol => {
                // Sharpen: amplify differences from equal, favor high-signal assets.
                // Use temperature scaling: b_i -> b_i^(1/T) where T < 1 sharpens.
                let temperature = 0.7_f64;
                let sharpened: Vec<f64> = base_budget
                    .iter()
                    .map(|&b| b.max(1e-12).powf(1.0 / temperature))
                    .collect();
                let sum: f64 = sharpened.iter().sum();
                sharpened.iter().map(|&s| s / sum).collect()
            }
            RegimeState::BearLowVol => {
                // Flatten toward equal: blend 50% toward 1/n.
                let blend = 0.50_f64;
                base_budget
                    .iter()
                    .map(|&b| blend * equal + (1.0 - blend) * b)
                    .collect()
            }
            RegimeState::BearHighVol => {
                // Defensive: heavy flatten (80% toward equal).
                let blend = 0.80_f64;
                base_budget
                    .iter()
                    .map(|&b| blend * equal + (1.0 - blend) * b)
                    .collect()
            }
            RegimeState::Neutral => {
                // Mild flattening (20% toward equal).
                let blend = 0.20_f64;
                base_budget
                    .iter()
                    .map(|&b| blend * equal + (1.0 - blend) * b)
                    .collect()
            }
        };

        // Re-normalize.
        let sum: f64 = adjusted.iter().sum();
        if sum < 1e-12 {
            return Self::equal_risk(n);
        }
        adjusted.iter().map(|&v| v / sum).collect()
    }

    // -----------------------------------------------------------------------
    // Realized risk contributions
    // -----------------------------------------------------------------------

    /// Compute realized marginal risk contributions for a portfolio.
    ///
    /// MRC_i = w_i * (Sigma * w)_i / sigma_p
    ///
    /// where sigma_p = sqrt(w' * Sigma * w).
    ///
    /// Returns a vector of fractional risk contributions summing to 1.0.
    /// If portfolio volatility is effectively zero, returns equal contributions.
    pub fn compute_realized_contributions(weights: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
        let n = weights.len();
        assert_eq!(cov.len(), n, "covariance matrix rows must match weights length");
        for row in cov {
            assert_eq!(row.len(), n, "covariance matrix must be square");
        }

        // Compute Sigma * w (matrix-vector product).
        let mut sigma_w = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                sigma_w[i] += cov[i][j] * weights[j];
            }
        }

        // Portfolio variance: w' * Sigma * w
        let port_var: f64 = weights.iter().zip(sigma_w.iter()).map(|(&w, &sw)| w * sw).sum();
        let port_vol = port_var.sqrt();

        if port_vol < 1e-12 {
            return Self::equal_risk(n);
        }

        // MRC_i = w_i * (Sigma * w)_i
        let mrc: Vec<f64> = weights
            .iter()
            .zip(sigma_w.iter())
            .map(|(&w, &sw)| w * sw / port_vol)
            .collect();

        // MRC sums to port_vol, so divide by port_vol to get fractional contributions.
        mrc.iter().map(|&m| m / port_vol).collect()
    }

    /// Compute the L2 distance between realized and target risk contributions.
    ///
    /// This is sqrt(sum((realized_i - target_i)^2)).
    pub fn budget_deviation(realized: &[f64], target: &[f64]) -> f64 {
        assert_eq!(
            realized.len(),
            target.len(),
            "realized and target must have equal length"
        );
        let sum_sq: f64 = realized
            .iter()
            .zip(target.iter())
            .map(|(&r, &t)| (r - t).powi(2))
            .sum();
        sum_sq.sqrt()
    }

    /// Find weights that minimize the budget deviation from a target risk budget.
    ///
    /// Uses an iterative scaling approach (risk parity convergence):
    /// w_i <- w_i * sqrt(b_i / rc_i)
    /// until convergence or max_iter reached.
    ///
    /// Returns portfolio weights (long-only, sum to 1).
    pub fn optimize_to_budget(
        target_budget: &[f64],
        cov: &[Vec<f64>],
        max_iter: usize,
        tol: f64,
    ) -> Vec<f64> {
        let n = target_budget.len();
        if n == 0 {
            return Vec::new();
        }

        // Initialize with equal weights.
        let mut weights = vec![1.0 / n as f64; n];

        for _iter in 0..max_iter {
            let rc = Self::compute_realized_contributions(&weights, cov);
            let dev = Self::budget_deviation(&rc, target_budget);
            if dev < tol {
                break;
            }
            // Scale each weight by sqrt(target / realized).
            for i in 0..n {
                if rc[i] > 1e-12 {
                    weights[i] *= (target_budget[i] / rc[i]).sqrt();
                }
            }
            // Re-normalize to sum to 1.
            let sum: f64 = weights.iter().sum();
            if sum > 1e-12 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
        }

        weights
    }

    /// Compute the portfolio volatility given weights and covariance matrix.
    pub fn portfolio_volatility(weights: &[f64], cov: &[Vec<f64>]) -> f64 {
        let n = weights.len();
        let mut port_var = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                port_var += weights[i] * cov[i][j] * weights[j];
            }
        }
        port_var.sqrt()
    }

    /// Compute the concentration ratio (Herfindahl index) of a risk budget.
    /// Returns a value in [1/n, 1] where 1/n is fully diversified.
    pub fn herfindahl_index(budget: &[f64]) -> f64 {
        budget.iter().map(|&b| b * b).sum()
    }

    /// Compute the effective number of risk contributors (1 / HHI).
    pub fn effective_n(budget: &[f64]) -> f64 {
        let hhi = Self::herfindahl_index(budget);
        if hhi < 1e-12 {
            return budget.len() as f64;
        }
        1.0 / hhi
    }

    /// Check whether a budget vector is valid (non-negative, sums to ~1).
    pub fn is_valid_budget(budget: &[f64]) -> bool {
        if budget.is_empty() {
            return false;
        }
        let sum: f64 = budget.iter().sum();
        let all_non_neg = budget.iter().all(|&b| b >= 0.0);
        all_non_neg && (sum - 1.0).abs() < 1e-6
    }

    /// Blend two budgets linearly: result = alpha * a + (1 - alpha) * b.
    pub fn blend_budgets(a: &[f64], b: &[f64], alpha: f64) -> Vec<f64> {
        assert_eq!(a.len(), b.len(), "budget vectors must have equal length");
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| alpha * ai + (1.0 - alpha) * bi)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_cov(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect()
    }

    #[test]
    fn test_equal_risk() {
        let budget = RiskBudgetOptimizer::equal_risk(4);
        assert_eq!(budget.len(), 4);
        let sum: f64 = budget.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        for &b in &budget {
            assert!((b - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_signal_weighted_budget() {
        let signals = vec![1.0, 2.0, 3.0, 4.0];
        let budget = RiskBudgetOptimizer::signal_weighted_budget(&signals);
        let sum: f64 = budget.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(budget[3] > budget[2]);
        assert!(budget[2] > budget[1]);
        assert!(budget[1] > budget[0]);
    }

    #[test]
    fn test_regime_adjusted_bear_flattens() {
        let base = vec![0.5, 0.3, 0.2];
        let adjusted =
            RiskBudgetOptimizer::regime_adjusted_budget(&base, &RegimeState::BearHighVol);
        let equal = 1.0 / 3.0;
        // After heavy flattening, all entries should be closer to 1/3.
        for &a in &adjusted {
            assert!((a - equal).abs() < 0.20, "adjusted={}", a);
        }
    }

    #[test]
    fn test_realized_contributions_identity() {
        // With identity covariance, equal weights give equal contributions.
        let n = 4;
        let weights = vec![0.25; n];
        let cov = identity_cov(n);
        let rc = RiskBudgetOptimizer::compute_realized_contributions(&weights, &cov);
        let sum: f64 = rc.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        for &r in &rc {
            assert!((r - 0.25).abs() < 1e-8);
        }
    }

    #[test]
    fn test_budget_deviation_zero() {
        let v = vec![0.25, 0.25, 0.25, 0.25];
        let dev = RiskBudgetOptimizer::budget_deviation(&v, &v);
        assert!(dev.abs() < 1e-10);
    }
}
