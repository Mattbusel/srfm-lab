//! Factor risk model: factor covariance matrix, specific risk, total portfolio variance.
//!
//! Implements a Barra-style multi-factor risk model:
//! Total Risk = Factor Risk + Specific Risk
//! Var(r_p) = h' * F * Cov_f * F' * h + h' * D * h
//!
//! where h = portfolio weights, F = factor exposure matrix, D = specific variance diagonal.

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};

/// Barra-style multi-factor risk model.
#[derive(Debug, Clone)]
pub struct FactorRiskModel {
    /// Factor covariance matrix (n_factors x n_factors), annualized
    pub factor_cov: Array2<f64>,
    /// Factor exposure matrix (n_assets x n_factors)
    pub factor_exposures: Array2<f64>,
    /// Specific variance for each asset (diagonal of D), annualized
    pub specific_variance: Array1<f64>,
    /// Factor names
    pub factor_names: Vec<String>,
    /// Asset names
    pub asset_names: Vec<String>,
}

/// Portfolio risk decomposition result.
#[derive(Debug, Clone)]
pub struct PortfolioRisk {
    /// Total portfolio variance (annualized)
    pub total_variance: f64,
    /// Total portfolio volatility (annualized)
    pub total_vol: f64,
    /// Factor risk variance component
    pub factor_variance: f64,
    /// Factor risk volatility component
    pub factor_vol: f64,
    /// Specific (idiosyncratic) risk variance component
    pub specific_variance: f64,
    /// Specific risk volatility component
    pub specific_vol: f64,
    /// Fraction of variance from factors
    pub factor_fraction: f64,
    /// Per-factor variance contribution
    pub factor_variance_contributions: Vec<f64>,
    /// Per-asset specific variance contribution
    pub asset_specific_contributions: Vec<f64>,
    /// Marginal contribution to risk (MCTR) per asset
    pub mctr: Vec<f64>,
}

impl FactorRiskModel {
    /// Create a new factor risk model.
    pub fn new(
        factor_cov: Array2<f64>,
        factor_exposures: Array2<f64>,
        specific_variance: Array1<f64>,
        factor_names: Vec<String>,
        asset_names: Vec<String>,
    ) -> Result<Self> {
        let n_assets = factor_exposures.nrows();
        let n_factors = factor_exposures.ncols();

        if factor_cov.dim() != (n_factors, n_factors) {
            return Err(FactorError::ShapeMismatch {
                msg: format!(
                    "factor_cov must be ({n_factors},{n_factors}), got {:?}",
                    factor_cov.dim()
                ),
            });
        }
        if specific_variance.len() != n_assets {
            return Err(FactorError::DimensionMismatch {
                expected: n_assets,
                got: specific_variance.len(),
            });
        }

        Ok(FactorRiskModel {
            factor_cov,
            factor_exposures,
            specific_variance,
            factor_names,
            asset_names,
        })
    }

    /// Compute the full asset covariance matrix.
    ///
    /// Sigma = F * Cov_f * F' + D
    /// where F is (n_assets x n_factors), D = diag(specific_variance).
    pub fn asset_covariance(&self) -> Array2<f64> {
        let (n_assets, n_factors) = self.factor_exposures.dim();

        // Compute F * Cov_f (n_assets x n_factors)
        let mut f_cov = Array2::<f64>::zeros((n_assets, n_factors));
        for i in 0..n_assets {
            for j in 0..n_factors {
                let mut sum = 0.0;
                for k in 0..n_factors {
                    sum += self.factor_exposures[[i, k]] * self.factor_cov[[k, j]];
                }
                f_cov[[i, j]] = sum;
            }
        }

        // Compute (F * Cov_f) * F' = factor contribution (n_assets x n_assets)
        let mut sigma = Array2::<f64>::zeros((n_assets, n_assets));
        for i in 0..n_assets {
            for j in 0..n_assets {
                let mut sum = 0.0;
                for k in 0..n_factors {
                    sum += f_cov[[i, k]] * self.factor_exposures[[j, k]];
                }
                sigma[[i, j]] = sum;
            }
        }

        // Add specific variance to diagonal
        for i in 0..n_assets {
            sigma[[i, i]] += self.specific_variance[i];
        }

        sigma
    }

    /// Compute portfolio risk decomposition.
    ///
    /// # Arguments
    /// * `weights` -- portfolio weights (n_assets,)
    pub fn portfolio_risk(&self, weights: &Array1<f64>) -> Result<PortfolioRisk> {
        let n_assets = self.factor_exposures.nrows();
        let n_factors = self.factor_exposures.ncols();

        if weights.len() != n_assets {
            return Err(FactorError::DimensionMismatch {
                expected: n_assets,
                got: weights.len(),
            });
        }

        // Compute portfolio factor exposures: h_f = F' * h
        let mut h_f = vec![0.0f64; n_factors];
        for k in 0..n_factors {
            for i in 0..n_assets {
                h_f[k] += self.factor_exposures[[i, k]] * weights[i];
            }
        }

        // Factor variance: h_f' * Cov_f * h_f
        let mut cov_h_f = vec![0.0f64; n_factors];
        for j in 0..n_factors {
            for k in 0..n_factors {
                cov_h_f[j] += self.factor_cov[[j, k]] * h_f[k];
            }
        }
        let factor_variance: f64 = h_f.iter().zip(cov_h_f.iter()).map(|(a, b)| a * b).sum();

        // Specific variance: sum(h_i^2 * sigma_i^2)
        let specific_var: f64 = weights
            .iter()
            .zip(self.specific_variance.iter())
            .map(|(w, s)| w * w * s)
            .sum();

        let total_variance = factor_variance + specific_var;
        let total_vol = total_variance.sqrt();
        let factor_vol = factor_variance.max(0.0).sqrt();
        let specific_vol = specific_var.max(0.0).sqrt();
        let factor_fraction = if total_variance > 1e-14 {
            factor_variance / total_variance
        } else {
            0.0
        };

        // Per-factor variance contribution: h_f_k * (Cov_f * h_f)_k
        let factor_variance_contributions: Vec<f64> = h_f
            .iter()
            .zip(cov_h_f.iter())
            .map(|(h, c)| h * c)
            .collect();

        // Per-asset specific variance contribution: h_i^2 * sigma_i^2
        let asset_specific_contributions: Vec<f64> = weights
            .iter()
            .zip(self.specific_variance.iter())
            .map(|(w, s)| w * w * s)
            .collect();

        // Marginal Contribution to Risk (MCTR) = (Sigma * h) / portfolio_vol
        // Sigma * h = factor_part + specific_part
        let mut sigma_h = vec![0.0f64; n_assets];

        // Factor part: F * Cov_f * F' * h = F * cov_h_f
        for i in 0..n_assets {
            for k in 0..n_factors {
                sigma_h[i] += self.factor_exposures[[i, k]] * cov_h_f[k];
            }
        }

        // Specific part: D * h
        for i in 0..n_assets {
            sigma_h[i] += self.specific_variance[i] * weights[i];
        }

        let mctr: Vec<f64> = if total_vol > 1e-12 {
            sigma_h.iter().map(|&s| s / total_vol).collect()
        } else {
            vec![0.0; n_assets]
        };

        Ok(PortfolioRisk {
            total_variance,
            total_vol,
            factor_variance,
            factor_vol,
            specific_variance: specific_var,
            specific_vol,
            factor_fraction,
            factor_variance_contributions,
            asset_specific_contributions,
            mctr,
        })
    }

    /// Compute active risk (tracking error) relative to benchmark.
    ///
    /// Active weights = portfolio weights - benchmark weights.
    pub fn active_risk(
        &self,
        portfolio_weights: &Array1<f64>,
        benchmark_weights: &Array1<f64>,
    ) -> Result<PortfolioRisk> {
        if portfolio_weights.len() != benchmark_weights.len() {
            return Err(FactorError::DimensionMismatch {
                expected: portfolio_weights.len(),
                got: benchmark_weights.len(),
            });
        }
        let active_weights: Array1<f64> = portfolio_weights - benchmark_weights;
        self.portfolio_risk(&active_weights)
    }

    /// Estimate specific variance from time series of residual returns.
    ///
    /// For each asset, specific variance = var(residuals) where residuals are
    /// asset returns minus factor model fitted returns.
    pub fn estimate_specific_variance(
        returns: &Array2<f64>,
        factor_exposures: &Array2<f64>,
        factor_returns: &Array2<f64>,
        annualization: f64,
    ) -> Result<Array1<f64>> {
        let (n_periods, n_assets) = returns.dim();
        let (n_periods_f, n_factors) = factor_returns.dim();

        if n_periods != n_periods_f {
            return Err(FactorError::DimensionMismatch {
                expected: n_periods,
                got: n_periods_f,
            });
        }

        let mut specific_var = Array1::<f64>::zeros(n_assets);

        for i in 0..n_assets {
            let mut residuals = Vec::with_capacity(n_periods);
            for t in 0..n_periods {
                let r_i = returns[[t, i]];
                // Fitted return = sum_k(exposure_ik * factor_return_k_t)
                let fitted: f64 = (0..n_factors)
                    .map(|k| factor_exposures[[i, k]] * factor_returns[[t, k]])
                    .sum();
                residuals.push(r_i - fitted);
            }

            // Estimate variance of residuals
            let n = residuals.len();
            if n < 2 {
                specific_var[i] = 0.01; // default 10% specific vol
                continue;
            }
            let mean = residuals.iter().sum::<f64>() / n as f64;
            let var = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (n - 1) as f64;
            specific_var[i] = var * annualization;
        }

        Ok(specific_var)
    }
}

/// Build a simple risk model from returns and pre-computed factor exposures.
///
/// Estimates factor returns via cross-sectional regression, computes factor
/// covariance from time series of factor returns, and estimates specific variances.
pub fn build_risk_model_from_panel(
    returns_panel: &Array2<f64>,
    factor_exposure_panel: &Array2<f64>,
    factor_names: Vec<String>,
    asset_names: Vec<String>,
    annualization: f64,
) -> Result<FactorRiskModel> {
    let (n_periods, n_assets) = returns_panel.dim();
    let n_factors = factor_names.len();

    if factor_exposure_panel.dim() != (n_assets, n_factors) {
        return Err(FactorError::ShapeMismatch {
            msg: format!(
                "factor_exposure_panel must be ({}, {}), got {:?}",
                n_assets,
                n_factors,
                factor_exposure_panel.dim()
            ),
        });
    }

    // Estimate factor returns via cross-sectional OLS each period
    let mut factor_returns = Array2::<f64>::zeros((n_periods, n_factors));

    for t in 0..n_periods {
        let period_returns: Vec<f64> = returns_panel.row(t).to_vec();
        match crate::cross_section::neutralize::ols_full(
            &period_returns,
            factor_exposure_panel,
            true,
        ) {
            Ok(ols) => {
                for k in 0..n_factors {
                    factor_returns[[t, k]] = ols.coefficients[k + 1]; // skip intercept
                }
            }
            Err(_) => {
                for k in 0..n_factors {
                    factor_returns[[t, k]] = f64::NAN;
                }
            }
        }
    }

    // Factor covariance = sample cov of factor return time series, annualized
    let mut raw_cov = crate::cross_section::composite::sample_covariance(&factor_returns)?;
    raw_cov.mapv_inplace(|v| v * annualization);

    // Specific variance
    let specific_var = FactorRiskModel::estimate_specific_variance(
        returns_panel,
        factor_exposure_panel,
        &factor_returns,
        annualization,
    )?;

    FactorRiskModel::new(raw_cov, factor_exposure_panel.clone(), specific_var, factor_names, asset_names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_risk_model(n_assets: usize, n_factors: usize) -> FactorRiskModel {
        let factor_cov = Array2::from_shape_fn((n_factors, n_factors), |(i, j)| {
            if i == j { 0.04 } else { 0.01 }
        });
        let exposures = Array2::from_shape_fn((n_assets, n_factors), |(i, k)| {
            ((i + k) as f64 * 0.3).sin()
        });
        let specific_var = Array1::from_elem(n_assets, 0.02);
        let factor_names: Vec<String> = (0..n_factors).map(|k| format!("f{}", k)).collect();
        let asset_names: Vec<String> = (0..n_assets).map(|i| format!("asset_{}", i)).collect();

        FactorRiskModel::new(factor_cov, exposures, specific_var, factor_names, asset_names).unwrap()
    }

    #[test]
    fn test_asset_covariance_symmetric() {
        let model = make_risk_model(10, 3);
        let cov = model.asset_covariance();
        assert_eq!(cov.dim(), (10, 10));
        for i in 0..10 {
            for j in 0..10 {
                assert!((cov[[i, j]] - cov[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_portfolio_risk() {
        let model = make_risk_model(10, 3);
        let weights = Array1::from_elem(10, 0.1);
        let risk = model.portfolio_risk(&weights).unwrap();
        assert!(risk.total_variance > 0.0);
        assert!((risk.factor_variance + risk.specific_variance - risk.total_variance).abs() < 1e-10);
        assert!(risk.factor_fraction >= 0.0 && risk.factor_fraction <= 1.0);
    }

    #[test]
    fn test_mctr_sums_to_vol() {
        let model = make_risk_model(10, 3);
        let weights = Array1::from_elem(10, 0.1);
        let risk = model.portfolio_risk(&weights).unwrap();
        // Sum of MCTR * weight should equal total vol
        let mctr_sum: f64 = risk.mctr.iter().zip(weights.iter()).map(|(m, w)| m * w).sum();
        assert!((mctr_sum - risk.total_vol).abs() < 1e-8, "mctr_sum={}, vol={}", mctr_sum, risk.total_vol);
    }

    #[test]
    fn test_build_risk_model() {
        let n_periods = 50;
        let n_assets = 20;
        let n_factors = 3;

        let returns = Array2::from_shape_fn((n_periods, n_assets), |(t, i)| {
            0.001 + 0.02 * ((t + i) as f64 * 0.1).sin()
        });
        let exposures = Array2::from_shape_fn((n_assets, n_factors), |(i, k)| {
            ((i + k) as f64 * 0.3).sin()
        });

        let factor_names: Vec<String> = (0..n_factors).map(|k| format!("f{}", k)).collect();
        let asset_names: Vec<String> = (0..n_assets).map(|i| format!("a{}", i)).collect();

        let model = build_risk_model_from_panel(
            &returns,
            &exposures,
            factor_names,
            asset_names,
            252.0,
        ).unwrap();

        assert_eq!(model.factor_cov.dim(), (n_factors, n_factors));
        assert_eq!(model.specific_variance.len(), n_assets);
    }
}
