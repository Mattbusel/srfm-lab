// factor_alpha.rs
// Factor-based alpha measurement.
// Regresses returns on factor exposures (momentum, value, quality, vol).
// Residual is the factor-adjusted alpha (Jensen's alpha).
// Rolling 63-day regression windows.
// Fama-MacBeth cross-sectional regression with Newey-West standard errors.

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};
use crate::{newey_west_lags, newey_west_variance};

/// Standard factor names used in decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactorName {
    Momentum,
    Value,
    Quality,
    LowVol,
    Market,
}

impl FactorName {
    pub fn all() -> [FactorName; 5] {
        [
            FactorName::Momentum,
            FactorName::Value,
            FactorName::Quality,
            FactorName::LowVol,
            FactorName::Market,
        ]
    }

    pub fn n_factors() -> usize {
        5
    }
}

/// Factor exposure for a single asset at a single time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorExposure {
    pub asset_id: String,
    pub bar: usize,
    /// Exposures ordered: momentum, value, quality, low_vol, market.
    pub exposures: Vec<f64>,
}

impl FactorExposure {
    pub fn new(asset_id: String, bar: usize, exposures: Vec<f64>) -> Self {
        assert_eq!(exposures.len(), FactorName::n_factors());
        FactorExposure { asset_id, bar, exposures }
    }
}

/// Alpha decomposition for a single asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaDecomposition {
    pub asset_id: String,
    pub bar: usize,
    /// Total return.
    pub total_return: f64,
    /// Factor-explained return.
    pub factor_return: f64,
    /// Residual alpha.
    pub alpha: f64,
    /// Factor betas.
    pub betas: Vec<f64>,
    /// R-squared of the factor model.
    pub r_squared: f64,
    /// Information ratio of the residual alpha series.
    pub alpha_ir: f64,
}

/// A cross-section observation: returns and factor exposures for all assets.
struct CrossSectionObs {
    bar: usize,
    returns: Vec<f64>,
    exposures: Vec<Vec<f64>>, // [asset_idx][factor_idx]
    asset_ids: Vec<String>,
}

/// OLS regression: y = X * beta. Returns beta vector.
/// X has shape [n x k], y has length n.
fn ols_multivariate(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let k = x[0].len();
    // Compute X'X and X'y manually.
    let mut xtx = vec![vec![0.0f64; k]; k];
    let mut xty = vec![0.0f64; k];
    for i in 0..n {
        let xi = &x[i];
        for j in 0..k {
            xty[j] += xi[j] * y[i];
            for l in 0..k {
                xtx[j][l] += xi[j] * xi[l];
            }
        }
    }
    // Solve via Cholesky or Gaussian elimination.
    gauss_solve(&xtx, &xty)
}

/// Gaussian elimination with partial pivoting for square system Ax = b.
fn gauss_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Find pivot.
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j] * factor;
                aug[row][j] -= val;
            }
        }
    }
    // Back substitution.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        let denom = aug[i][i];
        if denom.abs() > 1e-14 {
            x[i] /= denom;
        }
    }
    x
}

/// Rolling factor alpha model.
pub struct FactorAlphaModel {
    /// Rolling window in bars for regression (63 trading days).
    window: usize,
    /// Buffer of cross-section observations.
    history: VecDeque<CrossSectionObs>,
    /// Fama-MacBeth time series of cross-sectional betas.
    fm_betas: VecDeque<Vec<f64>>,
    fm_window: usize,
}

impl FactorAlphaModel {
    pub fn new(window: usize) -> Self {
        FactorAlphaModel {
            window: window.max(10),
            history: VecDeque::new(),
            fm_betas: VecDeque::new(),
            fm_window: 252,
        }
    }

    /// Push a new cross-section of returns and factor exposures.
    pub fn push(
        &mut self,
        bar: usize,
        returns: Vec<f64>,
        exposures: Vec<Vec<f64>>,
        asset_ids: Vec<String>,
    ) {
        assert_eq!(returns.len(), exposures.len());
        assert_eq!(returns.len(), asset_ids.len());

        // Run Fama-MacBeth cross-sectional OLS for this bar.
        if returns.len() >= FactorName::n_factors() + 2 {
            let betas = ols_multivariate(&exposures, &returns);
            self.fm_betas.push_back(betas);
            if self.fm_betas.len() > self.fm_window {
                self.fm_betas.pop_front();
            }
        }

        self.history.push_back(CrossSectionObs {
            bar,
            returns,
            exposures,
            asset_ids,
        });
        if self.history.len() > self.window {
            self.history.pop_front();
        }
    }

    /// Decompose return of a specific asset using the pooled rolling window betas.
    pub fn decompose_asset(
        &self,
        asset_id: &str,
        bar: usize,
    ) -> Option<AlphaDecomposition> {
        // Collect (return, exposure) pairs for this asset across the window.
        let pairs: Vec<(f64, Vec<f64>)> = self
            .history
            .iter()
            .filter_map(|cs| {
                let idx = cs.asset_ids.iter().position(|a| a == asset_id)?;
                Some((cs.returns[idx], cs.exposures[idx].clone()))
            })
            .collect();

        if pairs.len() < 10 {
            return None;
        }

        let returns: Vec<f64> = pairs.iter().map(|(r, _)| *r).collect();
        let exposures: Vec<Vec<f64>> = pairs.iter().map(|(_, e)| e.clone()).collect();

        let betas = ols_multivariate(&exposures, &returns);
        let k = betas.len();

        let fitted: Vec<f64> = exposures
            .iter()
            .map(|exp| exp.iter().zip(betas.iter()).map(|(e, b)| e * b).sum())
            .collect();

        let residuals: Vec<f64> = returns
            .iter()
            .zip(fitted.iter())
            .map(|(r, f)| r - f)
            .collect();

        let n = returns.len() as f64;
        let mean_ret = returns.iter().sum::<f64>() / n;
        let ss_tot: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum();
        let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
        let r_squared = if ss_tot > 1e-14 { 1.0 - ss_res / ss_tot } else { 1.0 };

        // Most recent observation's factor return.
        let last_exp = &exposures[exposures.len() - 1];
        let factor_return: f64 = last_exp.iter().zip(betas.iter()).map(|(e, b)| e * b).sum();
        let last_ret = returns[returns.len() - 1];
        let alpha = last_ret - factor_return;

        // Alpha IR from residual series.
        let mean_resid = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let std_resid = (residuals.iter().map(|v| (v - mean_resid).powi(2)).sum::<f64>()
            / residuals.len() as f64)
            .sqrt();
        let alpha_ir = if std_resid > 1e-10 { mean_resid / std_resid } else { 0.0 };

        Some(AlphaDecomposition {
            asset_id: asset_id.to_string(),
            bar,
            total_return: last_ret,
            factor_return,
            alpha,
            betas,
            r_squared,
            alpha_ir,
        })
    }

    /// Fama-MacBeth average betas with Newey-West standard errors.
    /// Returns Vec<(mean_beta, nw_se, t_stat)> for each factor.
    pub fn fama_macbeth_betas(&self) -> Vec<(f64, f64, f64)> {
        if self.fm_betas.is_empty() {
            return Vec::new();
        }
        let k = self.fm_betas[0].len();
        let t = self.fm_betas.len();
        let lags = newey_west_lags(t);

        (0..k)
            .map(|factor_idx| {
                let beta_series: Vec<f64> = self
                    .fm_betas
                    .iter()
                    .map(|betas| {
                        if factor_idx < betas.len() {
                            betas[factor_idx]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let mean_beta = beta_series.iter().sum::<f64>() / t as f64;
                let nw_var = newey_west_variance(&beta_series, lags);
                let se = (nw_var / t as f64).sqrt();
                let t_stat = if se > 1e-14 { mean_beta / se } else { 0.0 };
                (mean_beta, se, t_stat)
            })
            .collect()
    }

    /// Cross-sectional alpha for the most recent bar.
    /// Returns Vec of (asset_id, alpha) for all assets.
    pub fn current_cross_section_alphas(&self) -> Vec<(String, f64)> {
        let cs = match self.history.back() {
            Some(cs) => cs,
            None => return Vec::new(),
        };

        // Use Fama-MacBeth mean betas.
        let fm = self.fama_macbeth_betas();
        if fm.is_empty() {
            return Vec::new();
        }
        let mean_betas: Vec<f64> = fm.iter().map(|(b, _, _)| *b).collect();

        cs.returns
            .iter()
            .zip(cs.exposures.iter())
            .zip(cs.asset_ids.iter())
            .map(|((ret, exp), id)| {
                let factor_return: f64 = exp
                    .iter()
                    .zip(mean_betas.iter())
                    .map(|(e, b)| e * b)
                    .sum();
                (id.clone(), ret - factor_return)
            })
            .collect()
    }

    /// Compute alpha IC: correlation of predicted alphas vs subsequent realized alphas.
    pub fn alpha_ic(&self, predicted_alphas: &[(String, f64)], realized_returns: &[(String, f64)]) -> f64 {
        let mut pred = Vec::new();
        let mut real = Vec::new();
        for (id, pa) in predicted_alphas {
            if let Some((_, rr)) = realized_returns.iter().find(|(rid, _)| rid == id) {
                pred.push(*pa);
                real.push(*rr);
            }
        }
        if pred.len() < 5 {
            return 0.0;
        }
        crate::spearman_rank_corr(&pred, &real)
    }

    pub fn n_obs(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_factor_data(n_bars: usize, n_assets: usize) -> FactorAlphaModel {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(123);
        let mut model = FactorAlphaModel::new(63);
        let k = FactorName::n_factors();

        // True factor betas.
        let true_betas: Vec<f64> = (0..k).map(|i| 0.1 + i as f64 * 0.05).collect();

        for bar in 0..n_bars {
            let asset_ids: Vec<String> = (0..n_assets).map(|i| format!("ASSET_{}", i)).collect();
            let exposures: Vec<Vec<f64>> = (0..n_assets)
                .map(|_| (0..k).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
                .collect();
            let returns: Vec<f64> = exposures
                .iter()
                .map(|exp| {
                    let factor_ret: f64 = exp.iter().zip(true_betas.iter()).map(|(e, b)| e * b).sum();
                    factor_ret + 0.001 * (rng.gen::<f64>() - 0.5)
                })
                .collect();
            model.push(bar, returns, exposures, asset_ids);
        }
        model
    }

    #[test]
    fn test_fama_macbeth_recovers_betas() {
        let model = make_factor_data(100, 50);
        let fm = model.fama_macbeth_betas();
        assert!(!fm.is_empty());
        // Check that the first beta (momentum, true value 0.10) is roughly recovered.
        let (b0, _, _) = fm[0];
        assert!((b0 - 0.10).abs() < 0.05, "Beta recovery failed: {}", b0);
    }

    #[test]
    fn test_asset_decomposition() {
        let model = make_factor_data(80, 30);
        let result = model.decompose_asset("ASSET_0", 79);
        assert!(result.is_some());
        let decomp = result.unwrap();
        assert!(decomp.r_squared >= 0.0 && decomp.r_squared <= 1.1);
    }
}
