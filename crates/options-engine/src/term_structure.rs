use crate::OptionsError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YieldCurveType {
    Flat,
    NelsonSiegel,
    Svensson,
    PiecewiseLinear,
}

/// Yield curve abstraction
#[derive(Debug, Clone)]
pub struct YieldCurve {
    curve_type: YieldCurveType,
    params: Vec<f64>,
    /// Pillar maturities for piecewise interpolation
    pillars: Vec<f64>,
    /// Rates at pillars
    pillar_rates: Vec<f64>,
}

impl YieldCurve {
    /// Flat yield curve
    pub fn flat(rate: f64) -> Self {
        YieldCurve {
            curve_type: YieldCurveType::Flat,
            params: vec![rate],
            pillars: Vec::new(),
            pillar_rates: Vec::new(),
        }
    }

    /// Nelson-Siegel yield curve
    /// r(t) = beta0 + beta1*(1-exp(-t/tau))*tau/t + beta2*((1-exp(-t/tau))*tau/t - exp(-t/tau))
    pub fn nelson_siegel(beta0: f64, beta1: f64, beta2: f64, tau: f64) -> Self {
        YieldCurve {
            curve_type: YieldCurveType::NelsonSiegel,
            params: vec![beta0, beta1, beta2, tau],
            pillars: Vec::new(),
            pillar_rates: Vec::new(),
        }
    }

    /// Svensson yield curve (extended Nelson-Siegel)
    /// r(t) = beta0 + beta1*L1(t,tau1) + beta2*L2(t,tau1) + beta3*L2(t,tau2)
    pub fn svensson(beta0: f64, beta1: f64, beta2: f64, beta3: f64, tau1: f64, tau2: f64) -> Self {
        YieldCurve {
            curve_type: YieldCurveType::Svensson,
            params: vec![beta0, beta1, beta2, beta3, tau1, tau2],
            pillars: Vec::new(),
            pillar_rates: Vec::new(),
        }
    }

    /// Piecewise linear yield curve from pillars
    pub fn piecewise_linear(maturities: Vec<f64>, rates: Vec<f64>) -> Result<Self, OptionsError> {
        if maturities.len() != rates.len() {
            return Err(OptionsError::InvalidParameter(
                "maturities and rates must have equal length".into()
            ));
        }
        if maturities.is_empty() {
            return Err(OptionsError::InvalidParameter("must have at least one pillar".into()));
        }
        // Check maturities are sorted
        for i in 1..maturities.len() {
            if maturities[i] <= maturities[i - 1] {
                return Err(OptionsError::InvalidParameter(
                    "maturities must be strictly increasing".into()
                ));
            }
        }
        Ok(YieldCurve {
            curve_type: YieldCurveType::PiecewiseLinear,
            params: Vec::new(),
            pillars: maturities,
            pillar_rates: rates,
        })
    }

    /// Zero rate r(t) for maturity t
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 0.0 { return self.params.first().copied().unwrap_or(0.0); }
        match self.curve_type {
            YieldCurveType::Flat => self.params[0],
            YieldCurveType::NelsonSiegel => self.nelson_siegel_rate(t),
            YieldCurveType::Svensson => self.svensson_rate(t),
            YieldCurveType::PiecewiseLinear => self.piecewise_rate(t),
        }
    }

    fn nelson_siegel_rate(&self, t: f64) -> f64 {
        let (b0, b1, b2, tau) = (self.params[0], self.params[1], self.params[2], self.params[3]);
        if tau <= 0.0 { return b0; }
        let exp_term = (-t / tau).exp();
        let loading1 = (1.0 - exp_term) * tau / t;
        let loading2 = loading1 - exp_term;
        b0 + b1 * loading1 + b2 * loading2
    }

    fn svensson_rate(&self, t: f64) -> f64 {
        let (b0, b1, b2, b3, tau1, tau2) = (
            self.params[0], self.params[1], self.params[2], self.params[3],
            self.params[4], self.params[5]
        );
        let ns_base = {
            let exp1 = (-t / tau1).exp();
            let l1 = (1.0 - exp1) * tau1 / t;
            let l2 = l1 - exp1;
            b0 + b1 * l1 + b2 * l2
        };
        let extra = if tau2 > 0.0 {
            let exp2 = (-t / tau2).exp();
            let l2_tau2 = (1.0 - exp2) * tau2 / t - exp2;
            b3 * l2_tau2
        } else { 0.0 };
        ns_base + extra
    }

    fn piecewise_rate(&self, t: f64) -> f64 {
        let n = self.pillars.len();
        if t <= self.pillars[0] { return self.pillar_rates[0]; }
        if t >= self.pillars[n - 1] { return self.pillar_rates[n - 1]; }
        let pos = self.pillars.partition_point(|&p| p <= t);
        let t1 = self.pillars[pos - 1];
        let t2 = self.pillars[pos];
        let r1 = self.pillar_rates[pos - 1];
        let r2 = self.pillar_rates[pos];
        let alpha = (t - t1) / (t2 - t1);
        r1 + alpha * (r2 - r1)
    }

    /// Discount factor P(0, t) = exp(-r(t)*t)
    pub fn discount_factor(&self, t: f64) -> f64 {
        if t <= 0.0 { return 1.0; }
        (-self.zero_rate(t) * t).exp()
    }

    /// Instantaneous forward rate f(t) = -d/dt[ln P(0,t)] = r(t) + t * r'(t)
    pub fn forward_rate(&self, t: f64) -> f64 {
        if t <= 0.0 { return self.zero_rate(0.0001); }
        // Numerical differentiation: f(t) = -(ln P(t+h) - ln P(t-h)) / (2h)
        let h = t * 0.001 + 1e-6;
        let p_up = self.discount_factor(t + h);
        let p_dn = self.discount_factor((t - h).max(1e-8));
        -(p_up.ln() - p_dn.ln()) / (2.0 * h)
    }

    /// Forward rate between two dates: f(T1, T2)
    pub fn forward_rate_between(&self, t1: f64, t2: f64) -> Result<f64, OptionsError> {
        if t2 <= t1 {
            return Err(OptionsError::InvalidParameter(
                format!("t2={} must be > t1={}", t2, t1)
            ));
        }
        let p1 = self.discount_factor(t1);
        let p2 = self.discount_factor(t2);
        if p2 <= 0.0 {
            return Err(OptionsError::ModelError("Discount factor is zero or negative".into()));
        }
        Ok((p1 / p2 - 1.0) / (t2 - t1))
    }

    /// Par yield for maturity T (annual coupon bond)
    pub fn par_yield(&self, t: f64, freq: f64) -> f64 {
        let n = (t * freq).ceil() as usize;
        if n == 0 { return self.zero_rate(t); }
        let dt = 1.0 / freq;
        let sum_df: f64 = (1..=n).map(|i| self.discount_factor(i as f64 * dt)).sum();
        let df_t = self.discount_factor(t);
        freq * (1.0 - df_t) / sum_df
    }

    /// Bootstrap: create a flat-extrapolated version for short maturities
    pub fn zero_curve_points(&self, maturities: &[f64]) -> Vec<(f64, f64)> {
        maturities.iter().map(|&t| (t, self.zero_rate(t))).collect()
    }

    pub fn curve_type(&self) -> YieldCurveType { self.curve_type }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_curve() {
        let curve = YieldCurve::flat(0.05);
        assert!((curve.zero_rate(1.0) - 0.05).abs() < 1e-12);
        assert!((curve.zero_rate(5.0) - 0.05).abs() < 1e-12);
        assert!((curve.discount_factor(1.0) - (-0.05_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_discount_factor_at_zero() {
        let curve = YieldCurve::flat(0.05);
        assert!((curve.discount_factor(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_nelson_siegel_shape() {
        // Typical NS parameterization: upward sloping curve
        let curve = YieldCurve::nelson_siegel(0.05, -0.02, 0.03, 2.0);
        let r1 = curve.zero_rate(0.5);
        let r5 = curve.zero_rate(5.0);
        let r10 = curve.zero_rate(10.0);
        // All rates should be positive for these params
        assert!(r1 > 0.0 && r5 > 0.0 && r10 > 0.0);
        // Short end lower than long end typically
        println!("NS: r(0.5)={:.4} r(5)={:.4} r(10)={:.4}", r1, r5, r10);
    }

    #[test]
    fn test_svensson_curve() {
        let curve = YieldCurve::svensson(0.05, -0.02, 0.03, -0.01, 1.5, 5.0);
        let r = curve.zero_rate(1.0);
        assert!(r > 0.0 && r < 0.15);
    }

    #[test]
    fn test_piecewise_linear() {
        let curve = YieldCurve::piecewise_linear(
            vec![0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            vec![0.03, 0.035, 0.04, 0.045, 0.05, 0.055],
        ).unwrap();
        // At a pillar
        assert!((curve.zero_rate(1.0) - 0.04).abs() < 1e-12);
        // Interpolated
        let r_0_75 = curve.zero_rate(0.75);
        assert!((r_0_75 - 0.0375).abs() < 1e-10);
    }

    #[test]
    fn test_piecewise_extrapolation() {
        let curve = YieldCurve::piecewise_linear(
            vec![1.0, 5.0],
            vec![0.03, 0.05],
        ).unwrap();
        // Below lower pillar
        assert!((curve.zero_rate(0.5) - 0.03).abs() < 1e-12);
        // Above upper pillar
        assert!((curve.zero_rate(10.0) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_forward_rate() {
        let curve = YieldCurve::flat(0.05);
        // For a flat curve, forward rate = spot rate
        let fwd = curve.forward_rate(1.0);
        assert!((fwd - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_forward_rate_between() {
        let curve = YieldCurve::flat(0.05);
        let fwd = curve.forward_rate_between(1.0, 2.0).unwrap();
        // For flat 5% curve, simple forward rate between T1=1 and T2=2
        let df1 = (-0.05_f64).exp();
        let df2 = (-0.10_f64).exp();
        let expected = (df1 / df2 - 1.0); // per year in [1,2]
        assert!((fwd - expected).abs() < 0.01);
    }

    #[test]
    fn test_par_yield() {
        let curve = YieldCurve::flat(0.05);
        // For flat curve, par yield should be close to spot rate
        let par = curve.par_yield(5.0, 1.0);
        assert!((par - 0.05).abs() < 0.01, "par yield = {:.6}", par);
    }

    #[test]
    fn test_piecewise_invalid() {
        assert!(YieldCurve::piecewise_linear(vec![1.0, 0.5], vec![0.03, 0.04]).is_err());
        assert!(YieldCurve::piecewise_linear(vec![1.0], vec![0.03, 0.04]).is_err());
        assert!(YieldCurve::piecewise_linear(vec![], vec![]).is_err());
    }
}
