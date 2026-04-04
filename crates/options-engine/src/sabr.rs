use crate::OptionsError;
use crate::heston::nelder_mead;

/// SABR model parameters
#[derive(Debug, Clone, Copy)]
pub struct SabrParams {
    /// Initial volatility
    pub alpha: f64,
    /// Elasticity (CEV exponent), 0 <= beta <= 1
    pub beta: f64,
    /// Correlation between spot and vol
    pub rho: f64,
    /// Vol-of-vol
    pub nu: f64,
}

impl SabrParams {
    pub fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> Result<Self, OptionsError> {
        if beta < 0.0 || beta > 1.0 {
            return Err(OptionsError::InvalidParameter(format!("beta must be in [0,1], got {}", beta)));
        }
        if alpha <= 0.0 {
            return Err(OptionsError::InvalidParameter(format!("alpha must be positive, got {}", alpha)));
        }
        if rho <= -1.0 || rho >= 1.0 {
            return Err(OptionsError::InvalidParameter(format!("rho must be in (-1,1), got {}", rho)));
        }
        if nu <= 0.0 {
            return Err(OptionsError::InvalidParameter(format!("nu must be positive, got {}", nu)));
        }
        Ok(SabrParams { alpha, beta, rho, nu })
    }

    pub fn new_unchecked(alpha: f64, beta: f64, rho: f64, nu: f64) -> Self {
        SabrParams { alpha, beta, rho, nu }
    }
}

pub struct SabrModel {
    params: SabrParams,
}

impl SabrModel {
    pub fn new(params: SabrParams) -> Self {
        SabrModel { params }
    }

    /// Hagan et al. (2002) implied vol approximation
    /// Returns Black-Scholes implied vol for strike K, forward F, expiry T
    pub fn implied_vol(&self, f: f64, k: f64, t: f64) -> f64 {
        let p = &self.params;
        let alpha = p.alpha;
        let beta = p.beta;
        let rho = p.rho;
        let nu = p.nu;

        if (f - k).abs() < 1e-10 {
            // ATM formula
            return self.atm_implied_vol(f, t);
        }

        let fk = f * k;
        let fk_mid = fk.powf((1.0 - beta) / 2.0);
        let log_fk = (f / k).ln();

        // z = (nu / alpha) * (F*K)^((1-beta)/2) * log(F/K)
        let z = (nu / alpha) * fk_mid * log_fk;

        // x(z) = log[(sqrt(1-2*rho*z+z^2) + z - rho) / (1-rho)]
        let xz = {
            let sqrt_term = (1.0 - 2.0 * rho * z + z * z).sqrt();
            ((sqrt_term + z - rho) / (1.0 - rho)).ln()
        };

        let z_over_xz = if xz.abs() < 1e-12 { 1.0 } else { z / xz };

        // Denominator term: 1 + ((1-beta)^2/24)*log^2(F/K) + ((1-beta)^4/1920)*log^4(F/K)
        let log_sq = log_fk * log_fk;
        let beta1 = 1.0 - beta;
        let denom = 1.0 + (beta1 * beta1 / 24.0) * log_sq + (beta1.powi(4) / 1920.0) * log_sq * log_sq;

        // Numerator: alpha / ((F*K)^((1-beta)/2) * denom)
        let numerator_part = alpha / (fk_mid * denom);

        // Correction term: 1 + [...] * T
        let correction = 1.0 + (
            (beta1 * beta1 * alpha * alpha) / (24.0 * fk.powf(1.0 - beta))
            + (rho * beta * nu * alpha) / (4.0 * fk.powf((1.0 - beta) / 2.0))
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        ) * t;

        numerator_part * z_over_xz * correction
    }

    /// ATM implied vol (F = K)
    pub fn atm_implied_vol(&self, f: f64, t: f64) -> f64 {
        let p = &self.params;
        let alpha = p.alpha;
        let beta = p.beta;
        let rho = p.rho;
        let nu = p.nu;
        let beta1 = 1.0 - beta;
        let f_pow = f.powf(1.0 - beta);

        let correction = 1.0 + (
            (beta1 * beta1 * alpha * alpha) / (24.0 * f.powf(2.0 * (1.0 - beta)))
            + (rho * beta * nu * alpha) / (4.0 * f_pow)
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        ) * t;

        (alpha / f_pow) * correction
    }

    /// Implied vol for a range of strikes (smile)
    pub fn vol_smile(&self, f: f64, strikes: &[f64], t: f64) -> Vec<f64> {
        strikes.iter().map(|&k| self.implied_vol(f, k, t)).collect()
    }

    /// Compute SABR density (numerical derivative of CDF)
    pub fn density(&self, f: f64, k: f64, t: f64, r: f64) -> f64 {
        use crate::black_scholes::BlackScholes;
        use crate::black_scholes::OptionType;
        let h = k * 0.0001;
        let c_up = {
            let vol = self.implied_vol(f, k + h, t);
            BlackScholes::price(f * (-r * t).exp(), k + h, 0.0, r, vol, t, OptionType::Call)
        };
        let c_mid = {
            let vol = self.implied_vol(f, k, t);
            BlackScholes::price(f * (-r * t).exp(), k, 0.0, r, vol, t, OptionType::Call)
        };
        let c_dn = {
            let vol = self.implied_vol(f, k - h, t);
            BlackScholes::price(f * (-r * t).exp(), k - h, 0.0, r, vol, t, OptionType::Call)
        };
        ((-r * t).exp()) * (c_up - 2.0 * c_mid + c_dn) / (h * h)
    }

    /// Calibrate SABR (fixing beta) to market implied vols
    pub fn calibrate(
        &self,
        forward: f64,
        expiry: f64,
        strikes: &[f64],
        market_vols: &[f64],
        beta: f64,
        max_iter: usize,
    ) -> Result<SabrParams, OptionsError> {
        if strikes.len() != market_vols.len() {
            return Err(OptionsError::InvalidParameter("strikes and vols must have same length".into()));
        }

        let objective = |x: &[f64]| -> f64 {
            let alpha = x[0].max(1e-6);
            let rho = x[1].max(-0.9999).min(0.9999);
            let nu = x[2].max(1e-6);
            let params = SabrParams::new_unchecked(alpha, beta, rho, nu);
            let model = SabrModel::new(params);
            strikes.iter().zip(market_vols.iter()).map(|(&k, &mv)| {
                let model_vol = model.implied_vol(forward, k, expiry);
                (model_vol - mv).powi(2)
            }).sum()
        };

        // Initial guess using ATM vol as alpha seed
        let atm_vol = market_vols.iter().cloned().fold(f64::INFINITY, f64::min);
        let atm_alpha = atm_vol * forward.powf(1.0 - beta);
        let x0 = vec![atm_alpha.max(0.01), -0.3, 0.4];

        let result = nelder_mead(objective, x0, max_iter, 1e-12)?;

        SabrParams::new(result[0].max(1e-6), beta, result[1].max(-0.9999).min(0.9999), result[2].max(1e-6))
    }

    pub fn params(&self) -> SabrParams { self.params }

    /// Convert SABR implied vol to option price
    pub fn price(
        &self,
        s: f64, k: f64, r: f64, q: f64, t: f64,
        opt_type: crate::black_scholes::OptionType,
    ) -> f64 {
        use crate::black_scholes::BlackScholes;
        let f = BlackScholes::forward(s, r, q, t);
        let sigma = self.implied_vol(f, k, t);
        BlackScholes::price(s, k, r, q, sigma, t, opt_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> SabrParams {
        SabrParams::new_unchecked(0.2, 0.5, -0.3, 0.4)
    }

    #[test]
    fn test_sabr_atm_vol_positive() {
        let model = SabrModel::new(default_params());
        let vol = model.atm_implied_vol(100.0, 1.0);
        assert!(vol > 0.0, "ATM vol should be positive, got {}", vol);
    }

    #[test]
    fn test_sabr_smile_shape() {
        let model = SabrModel::new(default_params());
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let vols = model.vol_smile(100.0, &strikes, 1.0);
        assert_eq!(vols.len(), 5);
        for &v in &vols {
            assert!(v > 0.0, "vol should be positive, got {}", v);
        }
        // With negative rho, should see skew (lower vols for higher strikes approximately)
        // The smile should be non-flat
        let atm = vols[2];
        assert!(vols[0] != atm); // skew present
    }

    #[test]
    fn test_sabr_calibration() {
        let true_params = SabrParams::new_unchecked(0.2, 0.5, -0.3, 0.4);
        let model = SabrModel::new(true_params);
        let forward = 100.0;
        let expiry = 1.0;
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let market_vols: Vec<f64> = strikes.iter().map(|&k| model.implied_vol(forward, k, expiry)).collect();

        let calibrated = model.calibrate(forward, expiry, &strikes, &market_vols, 0.5, 500).unwrap();

        // Check that calibrated params reproduce vols
        let cal_model = SabrModel::new(calibrated);
        for (&k, &mv) in strikes.iter().zip(market_vols.iter()) {
            let fitted_vol = cal_model.implied_vol(forward, k, expiry);
            assert!((fitted_vol - mv).abs() < 0.005,
                "Calibration error at K={}: fitted={:.4} market={:.4}", k, fitted_vol, mv);
        }
    }

    #[test]
    fn test_sabr_invalid_params() {
        assert!(SabrParams::new(0.2, 1.5, -0.3, 0.4).is_err());
        assert!(SabrParams::new(-0.1, 0.5, -0.3, 0.4).is_err());
        assert!(SabrParams::new(0.2, 0.5, 1.5, 0.4).is_err());
        assert!(SabrParams::new(0.2, 0.5, -0.3, -0.1).is_err());
    }

    #[test]
    fn test_sabr_lognormal_beta1() {
        // With beta=1, SABR reduces to lognormal-like model
        let params = SabrParams::new_unchecked(0.2, 1.0, 0.0, 0.0001);
        let model = SabrModel::new(params);
        let vol = model.implied_vol(100.0, 100.0, 1.0);
        // Should be approximately alpha = 0.2
        assert!((vol - 0.2).abs() < 0.01, "Beta=1 vol={:.4} expected ~0.2", vol);
    }
}
