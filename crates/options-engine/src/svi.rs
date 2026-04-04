use crate::OptionsError;
use crate::heston::nelder_mead;

/// SVI (Stochastic Volatility Inspired) raw parametrization
/// w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
/// where k = log(K/F) is log-moneyness, w is total implied variance
#[derive(Debug, Clone, Copy)]
pub struct SviParams {
    pub a: f64,      // overall level of variance
    pub b: f64,      // slope/width of the wings
    pub rho: f64,    // rotation/skew, in (-1, 1)
    pub m: f64,      // translation along k-axis
    pub sigma: f64,  // smoothness of vertex
}

impl SviParams {
    pub fn new(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Result<Self, OptionsError> {
        if b < 0.0 {
            return Err(OptionsError::InvalidParameter(format!("SVI b must be >= 0, got {}", b)));
        }
        if rho <= -1.0 || rho >= 1.0 {
            return Err(OptionsError::InvalidParameter(format!("SVI rho must be in (-1,1), got {}", rho)));
        }
        if sigma <= 0.0 {
            return Err(OptionsError::InvalidParameter(format!("SVI sigma must be > 0, got {}", sigma)));
        }
        Ok(SviParams { a, b, rho, m, sigma })
    }

    pub fn new_unchecked(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Self {
        SviParams { a, b, rho, m, sigma }
    }

    /// Total implied variance w(k) for log-moneyness k
    pub fn total_variance(&self, k: f64) -> f64 {
        let d = k - self.m;
        self.a + self.b * (self.rho * d + (d * d + self.sigma * self.sigma).sqrt())
    }

    /// Implied vol sigma_impl(k) = sqrt(w(k) / T)
    pub fn implied_vol(&self, k: f64, t: f64) -> Result<f64, OptionsError> {
        let w = self.total_variance(k);
        if w < 0.0 {
            return Err(OptionsError::ArbitrageViolation(format!(
                "Negative total variance w={:.6} at k={:.4}", w, k
            )));
        }
        Ok((w / t).sqrt())
    }

    /// Check butterfly arbitrage: w''(k) >= 0 for all k (second derivative of w must be >= 0)
    /// More precisely: (1 - k*w'/(2w))^2 - w'^2/4*(1/w + 1/4) + w''/2 >= 0
    pub fn check_butterfly_arbitrage(&self) -> bool {
        let test_ks = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        for &k in &test_ks {
            let g = self.durrleman_g(k);
            if g < -1e-6 {
                return false;
            }
        }
        true
    }

    /// Durrleman's g function for butterfly arbitrage check
    /// g(k) = (1 - k*w'/(2w))^2 - w'^2/4*(1/w + 1/4) + w''/2
    fn durrleman_g(&self, k: f64) -> f64 {
        let d = k - self.m;
        let sq = (d * d + self.sigma * self.sigma).sqrt();
        let w = self.total_variance(k);

        let w_prime = self.b * (self.rho + d / sq);
        let w_dbl_prime = self.b * self.sigma * self.sigma / (sq * sq * sq);

        if w < 1e-12 { return 0.0; }

        let term1 = (1.0 - k * w_prime / (2.0 * w)).powi(2);
        let term2 = w_prime * w_prime / 4.0 * (1.0 / w + 0.25);
        let term3 = w_dbl_prime / 2.0;

        term1 - term2 + term3
    }

    /// Check calendar spread arbitrage: w(k; T1) <= w(k; T2) for T1 < T2
    /// (Total variance must be non-decreasing in T for each k)
    pub fn check_calendar_arbitrage(
        &self,
        other: &SviParams,
        test_ks: &[f64],
    ) -> bool {
        for &k in test_ks {
            let w1 = self.total_variance(k);
            let w2 = other.total_variance(k);
            if w1 > w2 + 1e-8 {
                return false;
            }
        }
        true
    }

    /// Jump-Wing (JW) parametrization conversion
    /// Converts from JW to raw SVI
    pub fn from_jw(v_t: f64, psi_t: f64, p_t: f64, c_t: f64, v_tilde_t: f64) -> Result<Self, OptionsError> {
        // JW params: v_t (ATM variance), psi_t (ATM slope), p_t (put wing), c_t (call wing), v_tilde_t (min variance)
        // Convert to raw SVI
        let b = 0.5 * (p_t + c_t);
        if b < 0.0 {
            return Err(OptionsError::InvalidParameter("JW: b would be negative".into()));
        }
        let rho_raw = 1.0 - p_t / b;
        let rho = rho_raw.max(-0.9999).min(0.9999);
        let beta = rho - 2.0 * psi_t / b;
        let alpha = beta.signum() * (beta * beta - (4.0 * (v_t - v_tilde_t) * b) / (1.0 + beta.abs())).sqrt();
        // m and sigma from the ATM conditions
        let m = (v_t - v_tilde_t) / (b * (rho - alpha));
        let sigma = (alpha * alpha + v_tilde_t / b).sqrt().max(1e-6);
        let a = v_tilde_t - b * sigma;
        SviParams::new(a, b, rho, m, sigma)
    }
}

pub struct SviModel {
    slices: Vec<(f64, SviParams)>, // (expiry T, params)
}

impl SviModel {
    pub fn new() -> Self {
        SviModel { slices: Vec::new() }
    }

    pub fn add_slice(&mut self, t: f64, params: SviParams) {
        // Insert in sorted order by expiry
        let pos = self.slices.partition_point(|&(t2, _)| t2 < t);
        self.slices.insert(pos, (t, params));
    }

    /// Implied vol for (strike, expiry) via SVI surface
    pub fn implied_vol(&self, log_moneyness: f64, t: f64) -> Result<f64, OptionsError> {
        if self.slices.is_empty() {
            return Err(OptionsError::ModelError("No SVI slices".into()));
        }

        // Find bracketing slices
        let pos = self.slices.partition_point(|&(t2, _)| t2 <= t);
        if pos == 0 {
            return self.slices[0].1.implied_vol(log_moneyness, self.slices[0].0);
        }
        if pos >= self.slices.len() {
            let last = &self.slices[self.slices.len() - 1];
            return last.1.implied_vol(log_moneyness, last.0);
        }

        // Linear interpolation in total variance
        let (t1, ref p1) = self.slices[pos - 1];
        let (t2, ref p2) = self.slices[pos];
        let w1 = p1.total_variance(log_moneyness);
        let w2 = p2.total_variance(log_moneyness);

        // Interpolate total variance linearly in T
        let alpha = (t - t1) / (t2 - t1);
        let w = w1 + alpha * (w2 - w1);

        if w < 0.0 {
            return Err(OptionsError::ArbitrageViolation(format!(
                "Negative interpolated total variance w={:.6} at k={:.4} T={:.4}", w, log_moneyness, t
            )));
        }
        Ok((w / t).sqrt())
    }

    /// Calibrate a single SVI slice to market data
    pub fn calibrate_slice(
        forward: f64,
        expiry: f64,
        strikes: &[f64],
        market_vols: &[f64],
        max_iter: usize,
    ) -> Result<SviParams, OptionsError> {
        if strikes.len() != market_vols.len() {
            return Err(OptionsError::InvalidParameter("strikes and vols must have same length".into()));
        }

        let log_ks: Vec<f64> = strikes.iter().map(|&k| (k / forward).ln()).collect();
        let target_variances: Vec<f64> = market_vols.iter().map(|&v| v * v * expiry).collect();

        let objective = |x: &[f64]| -> f64 {
            let a = x[0];
            let b = x[1].max(0.0);
            let rho = x[2].max(-0.9999).min(0.9999);
            let m = x[3];
            let sigma = x[4].max(1e-6);
            let params = SviParams::new_unchecked(a, b, rho, m, sigma);
            log_ks.iter().zip(target_variances.iter()).map(|(&k, &tv)| {
                let w = params.total_variance(k);
                (w - tv).powi(2)
            }).sum()
        };

        // Initial guess: flat vol
        let avg_var = target_variances.iter().sum::<f64>() / target_variances.len() as f64;
        let x0 = vec![avg_var * 0.9, 0.1, -0.3, 0.0, 0.1];

        let result = nelder_mead(objective, x0, max_iter, 1e-12)?;

        SviParams::new(result[0], result[1].max(0.0), result[2], result[3], result[4].max(1e-6))
    }

    pub fn slices(&self) -> &[(f64, SviParams)] {
        &self.slices
    }

    /// Check for calendar spread arbitrage across all slices
    pub fn check_calendar_arbitrage(&self) -> bool {
        let test_ks: Vec<f64> = (-30..=30).map(|i| i as f64 * 0.1).collect();
        for i in 1..self.slices.len() {
            let (_, ref p_prev) = self.slices[i - 1];
            let (_, ref p_curr) = self.slices[i];
            if !p_prev.check_calendar_arbitrage(p_curr, &test_ks) {
                return false;
            }
        }
        true
    }

    /// Check butterfly arbitrage for all slices
    pub fn check_butterfly_arbitrage(&self) -> bool {
        self.slices.iter().all(|(_, p)| p.check_butterfly_arbitrage())
    }
}

impl Default for SviModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params() -> SviParams {
        SviParams::new_unchecked(0.04, 0.1, -0.3, 0.0, 0.1)
    }

    #[test]
    fn test_total_variance_atm() {
        let p = sample_params();
        let w_atm = p.total_variance(0.0);
        // w(0) = a + b*(rho*0 + sigma) = a + b*sigma = 0.04 + 0.1*0.1 = 0.05
        assert!((w_atm - 0.05).abs() < 1e-10, "w(0) = {:.6}", w_atm);
    }

    #[test]
    fn test_implied_vol_positive() {
        let p = sample_params();
        let vol = p.implied_vol(0.0, 1.0).unwrap();
        assert!(vol > 0.0);
        assert!((vol - 0.05_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_butterfly_arbitrage_free() {
        // This SVI smile should be butterfly-arb-free
        let p = SviParams::new_unchecked(0.04, 0.1, -0.3, 0.0, 0.3);
        assert!(p.check_butterfly_arbitrage());
    }

    #[test]
    fn test_calibrate_slice() {
        let true_params = sample_params();
        let forward = 100.0_f64;
        let expiry = 1.0_f64;
        let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
        let market_vols: Vec<f64> = strikes.iter()
            .map(|&k| {
                let log_k = (k / forward).ln();
                true_params.implied_vol(log_k, expiry).unwrap()
            })
            .collect();

        let calibrated = SviModel::calibrate_slice(forward, expiry, &strikes, &market_vols, 2000).unwrap();
        let cal_model = {
            let mut m = SviModel::new();
            m.add_slice(expiry, calibrated);
            m
        };

        for (&k, &mv) in strikes.iter().zip(market_vols.iter()) {
            let log_k = (k / forward).ln();
            let fitted_vol = cal_model.implied_vol(log_k, expiry).unwrap();
            assert!((fitted_vol - mv).abs() < 0.002,
                "SVI calibration error at K={}: fitted={:.4} market={:.4}", k, fitted_vol, mv);
        }
    }

    #[test]
    fn test_calendar_arbitrage_check() {
        let p1 = SviParams::new_unchecked(0.03, 0.08, -0.3, 0.0, 0.1);
        let p2 = SviParams::new_unchecked(0.05, 0.12, -0.3, 0.0, 0.1);
        let test_ks: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.1).collect();
        assert!(p1.check_calendar_arbitrage(&p2, &test_ks));
    }

    #[test]
    fn test_svi_surface_interpolation() {
        let mut surface = SviModel::new();
        surface.add_slice(0.25, SviParams::new_unchecked(0.03, 0.08, -0.2, 0.0, 0.08));
        surface.add_slice(1.0, SviParams::new_unchecked(0.04, 0.1, -0.3, 0.0, 0.1));

        let vol_mid = surface.implied_vol(0.0, 0.5).unwrap();
        let vol_short = surface.implied_vol(0.0, 0.25).unwrap();
        let vol_long = surface.implied_vol(0.0, 1.0).unwrap();

        // Mid should be between short and long term vols
        assert!(vol_mid > vol_short.min(vol_long) - 0.01);
        assert!(vol_mid < vol_short.max(vol_long) + 0.01);
    }
}
