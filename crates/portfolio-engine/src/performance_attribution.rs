// performance_attribution.rs -- Returns attribution for portfolio performance
// Implements Brinson-Hood-Beebower (BHB) three-factor attribution,
// factor regression-based attribution, and time-series contribution tables.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BHB Attribution
// ---------------------------------------------------------------------------

/// BHB three-factor attribution result for a single period or asset.
#[derive(Debug, Clone)]
pub struct BHBAttribution {
    /// Per-asset allocation effects.
    pub allocation: Vec<f64>,
    /// Per-asset selection effects.
    pub selection: Vec<f64>,
    /// Per-asset interaction effects.
    pub interaction: Vec<f64>,
    /// Total attribution = sum(allocation + selection + interaction).
    pub total_active_return: f64,
    /// Benchmark return for the period.
    pub benchmark_return: f64,
    /// Portfolio return for the period.
    pub portfolio_return: f64,
}

impl BHBAttribution {
    /// Sum of all allocation effects.
    pub fn total_allocation(&self) -> f64 {
        self.allocation.iter().sum()
    }

    /// Sum of all selection effects.
    pub fn total_selection(&self) -> f64 {
        self.selection.iter().sum()
    }

    /// Sum of all interaction effects.
    pub fn total_interaction(&self) -> f64 {
        self.interaction.iter().sum()
    }

    /// Verify that allocation + selection + interaction sums to total_active_return.
    pub fn check_consistency(&self) -> bool {
        let computed =
            self.total_allocation() + self.total_selection() + self.total_interaction();
        (computed - self.total_active_return).abs() < 1e-9
    }
}

/// Factor attribution result from OLS regression.
#[derive(Debug, Clone)]
pub struct FactorAttribution {
    /// Factor names in order.
    pub factor_names: Vec<String>,
    /// Factor loadings (betas).
    pub loadings: Vec<f64>,
    /// Factor returns for the period.
    pub factor_returns: Vec<f64>,
    /// Contribution of each factor: loading_i * factor_return_i.
    pub factor_contributions: Vec<f64>,
    /// Alpha (unexplained return).
    pub alpha: f64,
    /// R-squared of the regression.
    pub r_squared: f64,
}

impl FactorAttribution {
    /// Total return explained by factors (sum of contributions).
    pub fn explained_return(&self) -> f64 {
        self.factor_contributions.iter().sum()
    }

    /// Total return = explained + alpha.
    pub fn total_return(&self) -> f64 {
        self.explained_return() + self.alpha
    }
}

/// Per-period attribution record for a time-series attribution table.
#[derive(Debug, Clone)]
pub struct PeriodAttribution {
    /// Period index (day/month/quarter).
    pub period: usize,
    /// Portfolio return for this period.
    pub portfolio_return: f64,
    /// Benchmark return for this period.
    pub benchmark_return: f64,
    /// Active return (portfolio - benchmark).
    pub active_return: f64,
    /// Allocation effect.
    pub allocation: f64,
    /// Selection effect.
    pub selection: f64,
    /// Interaction effect.
    pub interaction: f64,
}

/// Core attribution engine.
pub struct AttributionEngine;

impl AttributionEngine {
    // -----------------------------------------------------------------------
    // BHB Three-Factor Attribution
    // -----------------------------------------------------------------------

    /// Compute BHB attribution for one period.
    ///
    /// Inputs (all vectors of length n_assets):
    /// - `portfolio_weights`: w_p_i -- portfolio weight in asset i
    /// - `benchmark_weights`: w_b_i -- benchmark weight in asset i
    /// - `portfolio_returns`:  r_p_i -- portfolio's return for asset i
    /// - `benchmark_returns`:  r_b_i -- benchmark's return for asset i
    ///
    /// BHB formulae:
    /// - Allocation:   A_i = (w_p_i - w_b_i) * (r_b_i - r_b)
    /// - Selection:    S_i = w_b_i * (r_p_i - r_b_i)
    /// - Interaction:  I_i = (w_p_i - w_b_i) * (r_p_i - r_b_i)
    ///
    /// where r_b = sum_i(w_b_i * r_b_i) is the benchmark's total return.
    pub fn compute_attribution(
        portfolio_weights: &[f64],
        benchmark_weights: &[f64],
        portfolio_returns: &[f64],
        benchmark_returns: &[f64],
    ) -> BHBAttribution {
        let n = portfolio_weights.len();
        assert_eq!(benchmark_weights.len(), n);
        assert_eq!(portfolio_returns.len(), n);
        assert_eq!(benchmark_returns.len(), n);

        // Total benchmark return.
        let r_b: f64 = benchmark_weights
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(&w, &r)| w * r)
            .sum();

        // Total portfolio return.
        let r_p: f64 = portfolio_weights
            .iter()
            .zip(portfolio_returns.iter())
            .map(|(&w, &r)| w * r)
            .sum();

        let mut allocation = vec![0.0_f64; n];
        let mut selection = vec![0.0_f64; n];
        let mut interaction = vec![0.0_f64; n];

        for i in 0..n {
            let w_p = portfolio_weights[i];
            let w_b = benchmark_weights[i];
            let r_p_i = portfolio_returns[i];
            let r_b_i = benchmark_returns[i];

            allocation[i] = (w_p - w_b) * (r_b_i - r_b);
            selection[i] = w_b * (r_p_i - r_b_i);
            interaction[i] = (w_p - w_b) * (r_p_i - r_b_i);
        }

        let total_active_return = r_p - r_b;

        BHBAttribution {
            allocation,
            selection,
            interaction,
            total_active_return,
            benchmark_return: r_b,
            portfolio_return: r_p,
        }
    }

    /// Compute time-series BHB attribution -- one record per period.
    ///
    /// `weights_ts` and `returns_ts` are (n_periods x n_assets) matrices stored
    /// as Vec<Vec<f64>>.
    pub fn time_series_attribution(
        portfolio_weights_ts: &[Vec<f64>],
        benchmark_weights_ts: &[Vec<f64>],
        portfolio_returns_ts: &[Vec<f64>],
        benchmark_returns_ts: &[Vec<f64>],
    ) -> Vec<PeriodAttribution> {
        let n_periods = portfolio_weights_ts.len();
        assert_eq!(benchmark_weights_ts.len(), n_periods);
        assert_eq!(portfolio_returns_ts.len(), n_periods);
        assert_eq!(benchmark_returns_ts.len(), n_periods);

        let mut records = Vec::with_capacity(n_periods);
        for t in 0..n_periods {
            let bhb = Self::compute_attribution(
                &portfolio_weights_ts[t],
                &benchmark_weights_ts[t],
                &portfolio_returns_ts[t],
                &benchmark_returns_ts[t],
            );
            records.push(PeriodAttribution {
                period: t,
                portfolio_return: bhb.portfolio_return,
                benchmark_return: bhb.benchmark_return,
                active_return: bhb.total_active_return,
                allocation: bhb.total_allocation(),
                selection: bhb.total_selection(),
                interaction: bhb.total_interaction(),
            });
        }
        records
    }

    /// Compute cumulative attribution across periods by compounding.
    /// Returns (cumulative_portfolio_return, cumulative_benchmark_return, cumulative_active).
    pub fn cumulative_attribution(periods: &[PeriodAttribution]) -> (f64, f64, f64) {
        let cum_p: f64 = periods
            .iter()
            .fold(1.0, |acc, p| acc * (1.0 + p.portfolio_return))
            - 1.0;
        let cum_b: f64 = periods
            .iter()
            .fold(1.0, |acc, p| acc * (1.0 + p.benchmark_return))
            - 1.0;
        let cum_active = cum_p - cum_b;
        (cum_p, cum_b, cum_active)
    }

    // -----------------------------------------------------------------------
    // Factor Attribution via OLS
    // -----------------------------------------------------------------------

    /// Regress portfolio returns on factor returns using OLS.
    ///
    /// `portfolio_returns`: T-length vector of portfolio returns.
    /// `factor_returns_matrix`: (T x K) matrix, each inner vec is one observation of K factors.
    /// `factor_names`: length-K names.
    ///
    /// Returns FactorAttribution for the average / total period.
    pub fn factor_attribution_ols(
        portfolio_returns: &[f64],
        factor_returns_matrix: &[Vec<f64>],
        factor_names: &[String],
    ) -> FactorAttribution {
        let t = portfolio_returns.len();
        assert_eq!(factor_returns_matrix.len(), t);
        let k = factor_names.len();
        for row in factor_returns_matrix {
            assert_eq!(row.len(), k);
        }

        // Add intercept: augment each factor row with a 1.0 prepended.
        // X is (T x (K+1)), first column = 1 (intercept), rest = factor returns.
        let x: Vec<Vec<f64>> = factor_returns_matrix
            .iter()
            .map(|row| {
                let mut r = vec![1.0_f64];
                r.extend_from_slice(row);
                r
            })
            .collect();

        // Solve OLS: beta = (X'X)^{-1} X'y
        let p = k + 1; // including intercept
        let xt_x = Self::xt_x_matrix(&x, p);
        let xt_y = Self::xt_y_vector(&x, portfolio_returns, p);
        let beta = Self::solve_linear(xt_x, xt_y);

        let alpha = beta[0];
        let loadings: Vec<f64> = beta[1..].to_vec();

        // Compute fitted values and R-squared.
        let y_mean = portfolio_returns.iter().sum::<f64>() / t as f64;
        let fitted: Vec<f64> = x
            .iter()
            .map(|row| row.iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum())
            .collect();
        let ss_res: f64 = portfolio_returns
            .iter()
            .zip(fitted.iter())
            .map(|(&y, &yhat)| (y - yhat).powi(2))
            .sum();
        let ss_tot: f64 = portfolio_returns
            .iter()
            .map(|&y| (y - y_mean).powi(2))
            .sum();
        let r_squared = if ss_tot < 1e-14 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        };

        // Mean factor returns over the period.
        let mean_factor_returns: Vec<f64> = (0..k)
            .map(|j| {
                factor_returns_matrix.iter().map(|row| row[j]).sum::<f64>() / t as f64
            })
            .collect();

        let factor_contributions: Vec<f64> = loadings
            .iter()
            .zip(mean_factor_returns.iter())
            .map(|(&b, &fr)| b * fr)
            .collect();

        FactorAttribution {
            factor_names: factor_names.to_vec(),
            loadings,
            factor_returns: mean_factor_returns,
            factor_contributions,
            alpha,
            r_squared,
        }
    }

    // -----------------------------------------------------------------------
    // Attribution with named assets
    // -----------------------------------------------------------------------

    /// Compute per-asset contribution to portfolio return.
    ///
    /// contribution_i = weight_i * return_i
    pub fn asset_contributions(weights: &[f64], returns: &[f64]) -> Vec<f64> {
        assert_eq!(weights.len(), returns.len());
        weights.iter().zip(returns.iter()).map(|(&w, &r)| w * r).collect()
    }

    /// Return a map of symbol -> return contribution for a given period.
    pub fn named_contributions(
        symbols: &[String],
        weights: &[f64],
        returns: &[f64],
    ) -> HashMap<String, f64> {
        assert_eq!(symbols.len(), weights.len());
        assert_eq!(symbols.len(), returns.len());
        symbols
            .iter()
            .zip(weights.iter().zip(returns.iter()))
            .map(|(sym, (&w, &r))| (sym.clone(), w * r))
            .collect()
    }

    /// Compute information ratio: mean(active_return) / std(active_return).
    pub fn information_ratio(active_returns: &[f64]) -> f64 {
        let n = active_returns.len();
        if n < 2 {
            return 0.0;
        }
        let mean = active_returns.iter().sum::<f64>() / n as f64;
        let var = active_returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
            / (n as f64 - 1.0);
        let std_dev = var.sqrt();
        if std_dev < 1e-12 {
            0.0
        } else {
            mean / std_dev * (252.0_f64).sqrt()
        }
    }

    // -----------------------------------------------------------------------
    // Internal OLS helpers
    // -----------------------------------------------------------------------

    /// Compute X'X matrix of shape (p x p).
    fn xt_x_matrix(x: &[Vec<f64>], p: usize) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0_f64; p]; p];
        for row in x {
            for i in 0..p {
                for j in 0..p {
                    out[i][j] += row[i] * row[j];
                }
            }
        }
        out
    }

    /// Compute X'y vector of length p.
    fn xt_y_vector(x: &[Vec<f64>], y: &[f64], p: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; p];
        for (row, &yi) in x.iter().zip(y.iter()) {
            for i in 0..p {
                out[i] += row[i] * yi;
            }
        }
        out
    }

    /// Solve a linear system Ax = b via Gaussian elimination with partial pivoting.
    /// Returns the solution vector x. Panics if the system is singular.
    fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for col in 0..n {
            // Find pivot.
            let mut max_row = col;
            let mut max_val = a[col][col].abs();
            for row in (col + 1)..n {
                if a[row][col].abs() > max_val {
                    max_val = a[row][col].abs();
                    max_row = row;
                }
            }
            a.swap(col, max_row);
            b.swap(col, max_row);

            let pivot = a[col][col];
            if pivot.abs() < 1e-14 {
                // Singular -- return zeros rather than panicking in degenerate data.
                return vec![0.0; n];
            }

            for row in (col + 1)..n {
                let factor = a[row][col] / pivot;
                for k in col..n {
                    let val = a[col][k] * factor;
                    a[row][k] -= val;
                }
                b[row] -= b[col] * factor;
            }
        }

        // Back substitution.
        let mut x = vec![0.0_f64; n];
        for i in (0..n).rev() {
            x[i] = b[i];
            for j in (i + 1)..n {
                let ax = a[i][j] * x[j];
                x[i] -= ax;
            }
            x[i] /= a[i][i];
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_bhb() -> BHBAttribution {
        let pw = vec![0.6, 0.4];
        let bw = vec![0.5, 0.5];
        let pr = vec![0.10, 0.05];
        let br = vec![0.08, 0.04];
        AttributionEngine::compute_attribution(&pw, &bw, &pr, &br)
    }

    #[test]
    fn test_bhb_allocation_effect_sign() {
        let bhb = simple_bhb();
        // Asset 0: (w_p - w_b) = +0.1, (r_b_0 - r_b) = ?
        // r_b = 0.5*0.08 + 0.5*0.04 = 0.06
        // allocation[0] = 0.1 * (0.08 - 0.06) = 0.002 > 0
        assert!(bhb.allocation[0] > 0.0, "Overweight in better asset => positive allocation effect");
    }

    #[test]
    fn test_bhb_attribution_sums_to_total() {
        let bhb = simple_bhb();
        assert!(
            bhb.check_consistency(),
            "BHB components must sum to total active return"
        );
    }

    #[test]
    fn test_time_series_attribution_length() {
        let pw = vec![vec![0.6, 0.4]; 5];
        let bw = vec![vec![0.5, 0.5]; 5];
        let pr = vec![vec![0.10, 0.05]; 5];
        let br = vec![vec![0.08, 0.04]; 5];
        let periods = AttributionEngine::time_series_attribution(&pw, &bw, &pr, &br);
        assert_eq!(periods.len(), 5);
    }

    #[test]
    fn test_information_ratio_zero_for_constant() {
        let ar = vec![0.01; 100];
        // All same returns -> std = 0 -> IR = 0
        let ir = AttributionEngine::information_ratio(&ar);
        assert_eq!(ir, 0.0);
    }

    #[test]
    fn test_factor_attribution_r_squared_bounds() {
        let y = vec![0.01, 0.02, -0.01, 0.03, 0.00];
        let x: Vec<Vec<f64>> = y.iter().map(|&r| vec![r * 0.9]).collect();
        let names = vec!["MOM".to_string()];
        let fa = AttributionEngine::factor_attribution_ols(&y, &x, &names);
        assert!(fa.r_squared >= 0.0 && fa.r_squared <= 1.0 + 1e-6);
    }
}
