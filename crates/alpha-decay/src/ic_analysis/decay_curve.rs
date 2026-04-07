// decay_curve.rs
// IC decay curve modeling.
// Fits exponential decay: IC(h) = IC(1) * exp(-lambda * h)
// Half-life = ln(2) / lambda
// Also fits power law: IC(h) = IC(1) * h^(-beta)
// AIC/BIC model selection. Bootstrap confidence intervals on half-life.

use serde::{Deserialize, Serialize};
use crate::{pearson_corr, ols_simple};

/// Supported decay model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecayModel {
    Exponential,
    PowerLaw,
}

/// Fitted parameters for a decay model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayFit {
    pub model: DecayModel,
    /// IC(1): intercept / initial IC.
    pub ic1: f64,
    /// Exponential decay rate lambda (Exponential model).
    pub lambda: Option<f64>,
    /// Power law exponent beta (PowerLaw model).
    pub beta: Option<f64>,
    /// R-squared of fit.
    pub r_squared: f64,
    /// Log-likelihood (Gaussian residuals assumption).
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Number of data points used.
    pub n: usize,
    /// Residuals (observed - fitted).
    pub residuals: Vec<f64>,
}

impl DecayFit {
    /// Predict IC at horizon h.
    pub fn predict(&self, h: f64) -> f64 {
        match self.model {
            DecayModel::Exponential => {
                let lambda = self.lambda.unwrap_or(0.0);
                self.ic1 * (-lambda * h).exp()
            }
            DecayModel::PowerLaw => {
                let beta = self.beta.unwrap_or(0.0);
                if h <= 0.0 {
                    self.ic1
                } else {
                    self.ic1 * h.powf(-beta)
                }
            }
        }
    }

    /// Half-life of the signal (bars to decay to 50% of initial IC).
    pub fn half_life(&self) -> Option<f64> {
        match self.model {
            DecayModel::Exponential => {
                let lambda = self.lambda?;
                if lambda > 1e-10 {
                    Some(std::f64::consts::LN_2 / lambda)
                } else {
                    None
                }
            }
            DecayModel::PowerLaw => {
                let beta = self.beta?;
                if beta > 1e-10 {
                    // h^(-beta) = 0.5 => h = 2^(1/beta)
                    Some(2.0_f64.powf(1.0 / beta))
                } else {
                    None
                }
            }
        }
    }
}

/// Half-life estimate with bootstrap confidence interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalfLifeEstimate {
    pub point_estimate: f64,
    /// Lower bound of 95% CI.
    pub ci_lower: f64,
    /// Upper bound of 95% CI.
    pub ci_upper: f64,
    /// Standard error from bootstrap.
    pub se: f64,
    /// Number of bootstrap replications.
    pub n_boot: usize,
    pub model: DecayModel,
}

/// IC decay curve: holds observed (horizon, IC) pairs and fits decay models.
pub struct DecayCurve {
    /// (horizon_bars, mean_ic) pairs.
    data: Vec<(f64, f64)>,
}

impl DecayCurve {
    /// Create from (horizon, ic) observations.
    pub fn new(data: Vec<(f64, f64)>) -> Self {
        DecayCurve { data }
    }

    /// Fit exponential decay model via log-linear OLS.
    /// ln(IC(h)) = ln(IC(1)) - lambda * h
    /// Filters to positive IC values only.
    pub fn fit_exponential(&self) -> Option<DecayFit> {
        let pos: Vec<(f64, f64)> = self
            .data
            .iter()
            .copied()
            .filter(|&(h, ic)| h > 0.0 && ic > 1e-12)
            .collect();
        if pos.len() < 3 {
            return None;
        }
        let h_vals: Vec<f64> = pos.iter().map(|(h, _)| *h).collect();
        let ln_ic: Vec<f64> = pos.iter().map(|(_, ic)| ic.ln()).collect();

        // OLS: ln(IC) = intercept - lambda * h
        let (slope, intercept) = ols_simple(&h_vals, &ln_ic);
        let lambda = -slope;
        let ic1 = intercept.exp();

        let n = pos.len();
        let fitted: Vec<f64> = h_vals.iter().map(|&h| (intercept + slope * h).exp()).collect();
        let obs_ic: Vec<f64> = pos.iter().map(|(_, ic)| *ic).collect();
        let residuals: Vec<f64> = obs_ic.iter().zip(fitted.iter()).map(|(o, f)| o - f).collect();

        let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
        let mean_obs = obs_ic.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = obs_ic.iter().map(|v| (v - mean_obs).powi(2)).sum();
        let r_squared = if ss_tot < 1e-14 { 1.0 } else { 1.0 - ss_res / ss_tot };

        let sigma2 = ss_res / n as f64;
        let log_likelihood = if sigma2 > 1e-14 {
            -0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln()
                - ss_res / (2.0 * sigma2)
        } else {
            f64::INFINITY
        };

        // 2 parameters: ic1, lambda
        let k = 2.0;
        let aic = 2.0 * k - 2.0 * log_likelihood;
        let bic = k * (n as f64).ln() - 2.0 * log_likelihood;

        Some(DecayFit {
            model: DecayModel::Exponential,
            ic1,
            lambda: Some(lambda),
            beta: None,
            r_squared,
            log_likelihood,
            aic,
            bic,
            n,
            residuals,
        })
    }

    /// Fit power law decay model via log-log OLS.
    /// ln(IC(h)) = ln(IC(1)) - beta * ln(h)
    pub fn fit_power_law(&self) -> Option<DecayFit> {
        let pos: Vec<(f64, f64)> = self
            .data
            .iter()
            .copied()
            .filter(|&(h, ic)| h > 0.0 && ic > 1e-12)
            .collect();
        if pos.len() < 3 {
            return None;
        }
        let ln_h: Vec<f64> = pos.iter().map(|(h, _)| h.ln()).collect();
        let ln_ic: Vec<f64> = pos.iter().map(|(_, ic)| ic.ln()).collect();

        let (slope, intercept) = ols_simple(&ln_h, &ln_ic);
        let beta = -slope;
        let ic1 = intercept.exp();

        let n = pos.len();
        let obs_ic: Vec<f64> = pos.iter().map(|(_, ic)| *ic).collect();
        let fitted: Vec<f64> = pos
            .iter()
            .map(|(h, _)| {
                let val = (intercept + slope * h.ln()).exp();
                val
            })
            .collect();
        let residuals: Vec<f64> = obs_ic.iter().zip(fitted.iter()).map(|(o, f)| o - f).collect();

        let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
        let mean_obs = obs_ic.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = obs_ic.iter().map(|v| (v - mean_obs).powi(2)).sum();
        let r_squared = if ss_tot < 1e-14 { 1.0 } else { 1.0 - ss_res / ss_tot };

        let sigma2 = ss_res / n as f64;
        let log_likelihood = if sigma2 > 1e-14 {
            -0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln()
                - ss_res / (2.0 * sigma2)
        } else {
            f64::INFINITY
        };

        let k = 2.0;
        let aic = 2.0 * k - 2.0 * log_likelihood;
        let bic = k * (n as f64).ln() - 2.0 * log_likelihood;

        Some(DecayFit {
            model: DecayModel::PowerLaw,
            ic1,
            lambda: None,
            beta: Some(beta),
            r_squared,
            log_likelihood,
            aic,
            bic,
            n,
            residuals,
        })
    }

    /// Select best model by AIC. Returns both fits and the winner.
    pub fn select_best(&self) -> (Option<DecayFit>, Option<DecayFit>, DecayModel) {
        let exp_fit = self.fit_exponential();
        let pow_fit = self.fit_power_law();
        let winner = match (&exp_fit, &pow_fit) {
            (Some(e), Some(p)) => {
                if e.aic <= p.aic {
                    DecayModel::Exponential
                } else {
                    DecayModel::PowerLaw
                }
            }
            (Some(_), None) => DecayModel::Exponential,
            _ => DecayModel::PowerLaw,
        };
        (exp_fit, pow_fit, winner)
    }

    /// Bootstrap confidence interval for half-life.
    /// Resamples (horizon, IC) pairs with replacement and refits the model.
    pub fn bootstrap_half_life(
        &self,
        model: DecayModel,
        n_boot: usize,
        seed: u64,
    ) -> Option<HalfLifeEstimate> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        let n = self.data.len();
        if n < 3 {
            return None;
        }

        let point_fit = match model {
            DecayModel::Exponential => self.fit_exponential()?,
            DecayModel::PowerLaw => self.fit_power_law()?,
        };
        let point_hl = point_fit.half_life()?;

        let mut boot_hls: Vec<f64> = Vec::with_capacity(n_boot);
        for _ in 0..n_boot {
            // Resample with replacement.
            let sample: Vec<(f64, f64)> = (0..n)
                .map(|_| self.data[rng.gen_range(0..n)])
                .collect();
            let boot_curve = DecayCurve::new(sample);
            let fit = match model {
                DecayModel::Exponential => boot_curve.fit_exponential(),
                DecayModel::PowerLaw => boot_curve.fit_power_law(),
            };
            if let Some(f) = fit {
                if let Some(hl) = f.half_life() {
                    if hl.is_finite() && hl > 0.0 {
                        boot_hls.push(hl);
                    }
                }
            }
        }

        if boot_hls.is_empty() {
            return None;
        }

        boot_hls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lower_idx = (0.025 * boot_hls.len() as f64) as usize;
        let upper_idx = (0.975 * boot_hls.len() as f64) as usize;
        let ci_lower = boot_hls[lower_idx.min(boot_hls.len() - 1)];
        let ci_upper = boot_hls[upper_idx.min(boot_hls.len() - 1)];

        let mean_boot = boot_hls.iter().sum::<f64>() / boot_hls.len() as f64;
        let se = (boot_hls
            .iter()
            .map(|v| (v - mean_boot).powi(2))
            .sum::<f64>()
            / boot_hls.len() as f64)
            .sqrt();

        Some(HalfLifeEstimate {
            point_estimate: point_hl,
            ci_lower,
            ci_upper,
            se,
            n_boot: boot_hls.len(),
            model,
        })
    }

    /// Compute goodness of fit: correlation between observed and fitted IC.
    pub fn fit_correlation(&self, fit: &DecayFit) -> f64 {
        let obs: Vec<f64> = self.data.iter().map(|(_, ic)| *ic).collect();
        let fitted: Vec<f64> = self.data.iter().map(|(h, _)| fit.predict(*h)).collect();
        pearson_corr(&obs, &fitted)
    }

    /// Predict the entire IC profile using the fitted model.
    pub fn predicted_profile(&self, fit: &DecayFit) -> Vec<(f64, f64)> {
        self.data
            .iter()
            .map(|(h, _)| (*h, fit.predict(*h)))
            .collect()
    }

    /// Compute the area under the IC decay curve (trapezoidal rule).
    /// Represents total expected alpha contribution across all horizons.
    pub fn area_under_curve(&self, fit: &DecayFit, max_horizon: f64, steps: usize) -> f64 {
        let step = max_horizon / steps as f64;
        let mut area = 0.0;
        for i in 0..steps {
            let h0 = (i as f64 + 0.5) * step;
            area += fit.predict(h0) * step;
        }
        area
    }

    /// Given a holding period (bars), compute the average IC over that period.
    pub fn average_ic_over_holding(&self, fit: &DecayFit, holding_bars: usize) -> f64 {
        if holding_bars == 0 {
            return fit.predict(1.0);
        }
        let sum: f64 = (1..=holding_bars)
            .map(|h| fit.predict(h as f64))
            .sum();
        sum / holding_bars as f64
    }

    /// Return the raw data points.
    pub fn data(&self) -> &[(f64, f64)] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exp_data(lambda: f64, ic1: f64, noise: f64) -> Vec<(f64, f64)> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(7);
        let horizons = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0];
        horizons
            .iter()
            .map(|&h| {
                let true_ic = ic1 * (-lambda * h).exp();
                let noisy = true_ic + noise * (rng.gen::<f64>() - 0.5) * 0.1;
                (h, noisy.max(1e-6))
            })
            .collect()
    }

    #[test]
    fn test_exponential_fit_recovers_lambda() {
        let lambda = 0.05;
        let data = make_exp_data(lambda, 0.10, 0.01);
        let curve = DecayCurve::new(data);
        let fit = curve.fit_exponential().unwrap();
        let recovered = fit.lambda.unwrap();
        assert!(
            (recovered - lambda).abs() < 0.01,
            "Lambda mismatch: got {}, expected {}",
            recovered,
            lambda
        );
    }

    #[test]
    fn test_half_life_calculation() {
        let lambda = 0.1;
        let data = make_exp_data(lambda, 0.08, 0.005);
        let curve = DecayCurve::new(data);
        let fit = curve.fit_exponential().unwrap();
        let hl = fit.half_life().unwrap();
        let expected_hl = std::f64::consts::LN_2 / lambda;
        // Allow up to 30% relative error due to small-sample noise.
        let rel_err = (hl - expected_hl).abs() / expected_hl;
        assert!(rel_err < 0.30, "Half-life off: got {:.2}, expected {:.2}", hl, expected_hl);
    }

    #[test]
    fn test_bootstrap_confidence_interval() {
        let data = make_exp_data(0.05, 0.10, 0.01);
        let curve = DecayCurve::new(data);
        let hl_est = curve
            .bootstrap_half_life(DecayModel::Exponential, 200, 99)
            .unwrap();
        assert!(hl_est.ci_lower < hl_est.point_estimate);
        assert!(hl_est.point_estimate < hl_est.ci_upper);
    }

    #[test]
    fn test_aic_selects_exponential() {
        // True model is exponential.
        let data = make_exp_data(0.04, 0.12, 0.002);
        let curve = DecayCurve::new(data);
        let (exp_fit, _pow_fit, winner) = curve.select_best();
        assert!(exp_fit.is_some());
        // Winner should be exponential for clean exponential data.
        assert_eq!(winner, DecayModel::Exponential);
    }
}
