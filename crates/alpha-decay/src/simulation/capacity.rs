// capacity.rs
// Alpha capacity estimation.
// Model: alpha(AUM) = alpha_0 * (1 - AUM / capacity)
// Fit capacity from historical (AUM, alpha) observations using NLS or linear transform.
// Estimate at what AUM alpha decays to zero.

use serde::{Deserialize, Serialize};
use crate::ols_simple;

/// A single (AUM, realized_alpha) data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityObservation {
    /// AUM in dollars (or normalized units).
    pub aum: f64,
    /// Realized alpha (annualized) at that AUM level.
    pub realized_alpha: f64,
}

/// Fitted capacity model parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityEstimate {
    /// alpha_0: alpha at zero AUM.
    pub alpha_0: f64,
    /// Capacity: AUM at which alpha reaches zero.
    pub capacity: f64,
    /// R-squared of fit.
    pub r_squared: f64,
    /// Standard error of capacity estimate.
    pub capacity_se: f64,
    /// Standard error of alpha_0 estimate.
    pub alpha_0_se: f64,
    /// Recommended max AUM for target_alpha_fraction of alpha_0.
    pub max_aum_half_alpha: f64,
    /// Number of observations used.
    pub n_obs: usize,
    /// Fit residuals.
    pub residuals: Vec<f64>,
    /// Model confidence: high, medium, low.
    pub confidence: CapacityConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacityConfidence {
    High,
    Medium,
    Low,
}

impl CapacityEstimate {
    /// Predict alpha at a given AUM.
    pub fn predict_alpha(&self, aum: f64) -> f64 {
        if aum >= self.capacity {
            return 0.0;
        }
        self.alpha_0 * (1.0 - aum / self.capacity)
    }

    /// Predict alpha decay fraction at a given AUM.
    pub fn alpha_decay_fraction(&self, aum: f64) -> f64 {
        1.0 - (aum / self.capacity).min(1.0)
    }

    /// AUM level that captures `fraction` of alpha_0.
    pub fn aum_for_alpha_fraction(&self, fraction: f64) -> f64 {
        self.capacity * (1.0 - fraction)
    }

    /// Return curve: Vec of (aum, alpha) from 0 to capacity.
    pub fn alpha_curve(&self, steps: usize) -> Vec<(f64, f64)> {
        (0..=steps)
            .map(|i| {
                let aum = self.capacity * i as f64 / steps as f64;
                (aum, self.predict_alpha(aum))
            })
            .collect()
    }
}

/// Capacity model fitter.
pub struct CapacityModel {
    observations: Vec<CapacityObservation>,
}

impl CapacityModel {
    pub fn new() -> Self {
        CapacityModel {
            observations: Vec::new(),
        }
    }

    /// Add a (AUM, alpha) observation.
    pub fn push(&mut self, aum: f64, realized_alpha: f64) {
        self.observations.push(CapacityObservation { aum, realized_alpha });
    }

    /// Push a batch of observations.
    pub fn push_batch(&mut self, obs: Vec<CapacityObservation>) {
        self.observations.extend(obs);
    }

    /// Fit the linear capacity model: alpha = alpha_0 * (1 - AUM/C)
    /// Rearranges to: alpha = alpha_0 - (alpha_0/C) * AUM
    /// Let a = alpha_0, b = alpha_0/C => C = a/b.
    /// OLS: alpha = a + b*AUM => b is negative.
    pub fn fit(&self) -> Option<CapacityEstimate> {
        let n = self.observations.len();
        if n < 4 {
            return None;
        }

        let aum: Vec<f64> = self.observations.iter().map(|o| o.aum).collect();
        let alpha: Vec<f64> = self.observations.iter().map(|o| o.realized_alpha).collect();

        let (slope, intercept) = ols_simple(&aum, &alpha);
        // slope should be negative (alpha decays with AUM).
        let alpha_0 = intercept;
        if slope.abs() < 1e-14 || alpha_0 <= 0.0 {
            return None;
        }
        // C = -alpha_0 / slope (slope is negative).
        let capacity = -alpha_0 / slope;
        if capacity <= 0.0 {
            return None;
        }

        let fitted: Vec<f64> = aum.iter().map(|&a| intercept + slope * a).collect();
        let residuals: Vec<f64> = alpha.iter().zip(fitted.iter()).map(|(o, f)| o - f).collect();

        let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
        let mean_alpha = alpha.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = alpha.iter().map(|v| (v - mean_alpha).powi(2)).sum();
        let r_squared = if ss_tot > 1e-14 { 1.0 - ss_res / ss_tot } else { 1.0 };

        // Standard errors via OLS formula.
        let mse = ss_res / (n as f64 - 2.0).max(1.0);
        let mean_aum = aum.iter().sum::<f64>() / n as f64;
        let sxx: f64 = aum.iter().map(|a| (a - mean_aum).powi(2)).sum();
        let se_slope = if sxx > 1e-14 { (mse / sxx).sqrt() } else { 1e6 };
        let se_intercept = if sxx > 1e-14 {
            (mse * (1.0 / n as f64 + mean_aum.powi(2) / sxx)).sqrt()
        } else {
            1e6
        };

        // Delta method for capacity SE: C = -a/b, dC/da = -1/b, dC/db = a/b^2.
        let da = -1.0 / slope;
        let db = alpha_0 / slope.powi(2);
        let capacity_se = (da * da * se_intercept * se_intercept
            + db * db * se_slope * se_slope)
            .sqrt();
        let alpha_0_se = se_intercept;

        let max_aum_half_alpha = capacity * 0.5;

        let confidence = if r_squared > 0.7 && n >= 20 {
            CapacityConfidence::High
        } else if r_squared > 0.4 && n >= 10 {
            CapacityConfidence::Medium
        } else {
            CapacityConfidence::Low
        };

        Some(CapacityEstimate {
            alpha_0,
            capacity,
            r_squared,
            capacity_se,
            alpha_0_se,
            max_aum_half_alpha,
            n_obs: n,
            residuals,
            confidence,
        })
    }

    /// Fit using a power model: alpha = alpha_0 * (AUM / AUM_scale)^{-gamma}.
    /// Useful when decay is not strictly linear.
    pub fn fit_power_decay(&self) -> Option<PowerCapacityEstimate> {
        let n = self.observations.len();
        if n < 4 {
            return None;
        }

        let pos: Vec<(f64, f64)> = self
            .observations
            .iter()
            .filter(|o| o.aum > 0.0 && o.realized_alpha > 0.0)
            .map(|o| (o.aum, o.realized_alpha))
            .collect();
        if pos.len() < 4 {
            return None;
        }

        let ln_aum: Vec<f64> = pos.iter().map(|(a, _)| a.ln()).collect();
        let ln_alpha: Vec<f64> = pos.iter().map(|(_, al)| al.ln()).collect();

        let (slope, intercept) = ols_simple(&ln_aum, &ln_alpha);
        let gamma = -slope; // decay exponent.
        let alpha_scale = intercept.exp();

        let n_pos = pos.len() as f64;
        let fitted: Vec<f64> = ln_aum.iter().map(|la| (intercept + slope * la).exp()).collect();
        let obs_alpha: Vec<f64> = pos.iter().map(|(_, al)| *al).collect();
        let residuals: Vec<f64> = obs_alpha.iter().zip(fitted.iter()).map(|(o, f)| o - f).collect();
        let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
        let mean_obs = obs_alpha.iter().sum::<f64>() / n_pos;
        let ss_tot: f64 = obs_alpha.iter().map(|v| (v - mean_obs).powi(2)).sum();
        let r_squared = if ss_tot > 1e-14 { 1.0 - ss_res / ss_tot } else { 1.0 };

        Some(PowerCapacityEstimate {
            alpha_scale,
            gamma,
            r_squared,
            n_obs: pos.len(),
        })
    }

    /// Bootstrap confidence interval on fitted capacity.
    pub fn bootstrap_capacity_ci(&self, n_boot: usize, seed: u64) -> Option<(f64, f64, f64)> {
        use rand::prelude::*;
        let n = self.observations.len();
        if n < 4 {
            return None;
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut boot_capacities: Vec<f64> = Vec::with_capacity(n_boot);
        for _ in 0..n_boot {
            let sample: Vec<CapacityObservation> = (0..n)
                .map(|_| self.observations[rng.gen_range(0..n)].clone())
                .collect();
            let mut boot_model = CapacityModel::new();
            boot_model.push_batch(sample);
            if let Some(est) = boot_model.fit() {
                if est.capacity.is_finite() && est.capacity > 0.0 {
                    boot_capacities.push(est.capacity);
                }
            }
        }
        if boot_capacities.is_empty() {
            return None;
        }
        boot_capacities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = boot_capacities[(0.025 * boot_capacities.len() as f64) as usize];
        let hi = boot_capacities[(0.975 * boot_capacities.len() as f64).min(boot_capacities.len() as f64 - 1.0) as usize];
        let point = self.fit()?.capacity;
        Some((lo, point, hi))
    }

    pub fn n_obs(&self) -> usize {
        self.observations.len()
    }

    pub fn observations(&self) -> &[CapacityObservation] {
        &self.observations
    }
}

/// Power law capacity model estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerCapacityEstimate {
    /// Alpha at reference AUM level.
    pub alpha_scale: f64,
    /// Decay exponent.
    pub gamma: f64,
    pub r_squared: f64,
    pub n_obs: usize,
}

impl PowerCapacityEstimate {
    pub fn predict_alpha(&self, aum: f64) -> f64 {
        if aum <= 0.0 {
            return self.alpha_scale;
        }
        self.alpha_scale * aum.powf(-self.gamma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_capacity_data(alpha_0: f64, capacity: f64) -> CapacityModel {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let mut model = CapacityModel::new();
        for i in 0..30 {
            let aum = capacity * i as f64 / 30.0;
            let true_alpha = alpha_0 * (1.0 - aum / capacity);
            let noisy = true_alpha + 0.001 * (rng.gen::<f64>() - 0.5);
            model.push(aum, noisy.max(1e-6));
        }
        model
    }

    #[test]
    fn test_capacity_fit_recovers_params() {
        let true_capacity = 1_000_000_000.0;
        let true_alpha_0 = 0.10;
        let model = make_capacity_data(true_alpha_0, true_capacity);
        let est = model.fit().unwrap();
        assert!(
            (est.capacity - true_capacity).abs() / true_capacity < 0.05,
            "Capacity off: {:.0} vs expected {:.0}",
            est.capacity,
            true_capacity
        );
    }

    #[test]
    fn test_predict_zero_at_capacity() {
        let model = make_capacity_data(0.08, 5e8);
        let est = model.fit().unwrap();
        let alpha_at_cap = est.predict_alpha(est.capacity);
        assert!(alpha_at_cap.abs() < 0.001);
    }

    #[test]
    fn test_bootstrap_ci_bounds() {
        let model = make_capacity_data(0.12, 2e8);
        let ci = model.bootstrap_capacity_ci(200, 7).unwrap();
        assert!(ci.0 < ci.1 && ci.1 < ci.2, "CI should bracket the point estimate");
    }
}
