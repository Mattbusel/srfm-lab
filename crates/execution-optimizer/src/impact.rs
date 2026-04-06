/// Market-impact models.
///
/// Temporary impact: cost of trading q shares in interval τ.
/// Permanent impact: lasting price shift from cumulative trading.

use serde::Serialize;

/// Trait for temporary market-impact models.
pub trait ImpactModel: Send + Sync {
    /// Expected temporary impact cost for trading `qty` shares in `time_secs`.
    fn temp_impact(&self, qty: f64, time_secs: f64) -> f64;
    /// Expected permanent impact from `qty` cumulative shares traded.
    fn perm_impact(&self, qty: f64) -> f64;
}

/// Linear temporary impact: cost = η * (qty / time)
#[derive(Debug, Clone, Serialize)]
pub struct LinearImpact {
    /// Temporary impact coefficient
    pub eta:   f64,
    /// Permanent impact coefficient
    pub gamma: f64,
}

impl LinearImpact {
    pub fn new(eta: f64, gamma: f64) -> Self { Self { eta, gamma } }

    /// Calibrate from ADV and estimated half-spread.
    /// η ≈ half_spread / ADV, γ ≈ half_spread / (2 * ADV)
    pub fn calibrate(adv: f64, half_spread: f64) -> Self {
        Self { eta: half_spread / adv, gamma: half_spread / (2.0 * adv) }
    }
}

impl ImpactModel for LinearImpact {
    fn temp_impact(&self, qty: f64, time_secs: f64) -> f64 {
        if time_secs <= 0.0 { return f64::INFINITY; }
        self.eta * qty.abs() / time_secs
    }

    fn perm_impact(&self, qty: f64) -> f64 {
        self.gamma * qty.abs()
    }
}

/// Square-root temporary impact (more realistic for large orders):
/// cost = η * σ * √(qty / ADV)
#[derive(Debug, Clone, Serialize)]
pub struct SquareRootImpact {
    /// Calibration constant (typical: 0.314 for US equities)
    pub eta:   f64,
    /// Daily volatility of price
    pub sigma: f64,
    /// Average daily volume
    pub adv:   f64,
    /// Permanent impact coefficient
    pub gamma: f64,
}

impl SquareRootImpact {
    pub fn new(eta: f64, sigma: f64, adv: f64, gamma: f64) -> Self {
        Self { eta, sigma, adv, gamma }
    }

    /// Almgren et al. (2005) empirical calibration for US equities.
    pub fn us_equity(sigma: f64, adv: f64) -> Self {
        Self { eta: 0.314, sigma, adv, gamma: 0.142 }
    }
}

impl ImpactModel for SquareRootImpact {
    fn temp_impact(&self, qty: f64, _time_secs: f64) -> f64 {
        if self.adv <= 0.0 { return 0.0; }
        self.eta * self.sigma * (qty.abs() / self.adv).sqrt()
    }

    fn perm_impact(&self, qty: f64) -> f64 {
        if self.adv <= 0.0 { return 0.0; }
        self.gamma * self.sigma * (qty.abs() / self.adv).sqrt()
    }
}

/// Power-law impact: cost = η * σ * (qty/ADV)^α
#[derive(Debug, Clone, Serialize)]
pub struct PowerLawImpact {
    pub eta:   f64,
    pub sigma: f64,
    pub adv:   f64,
    pub alpha: f64,  // typically 0.4–0.6
    pub gamma: f64,
}

impl PowerLawImpact {
    pub fn new(eta: f64, sigma: f64, adv: f64, alpha: f64, gamma: f64) -> Self {
        Self { eta, sigma, adv, alpha, gamma }
    }
}

impl ImpactModel for PowerLawImpact {
    fn temp_impact(&self, qty: f64, _time_secs: f64) -> f64 {
        if self.adv <= 0.0 { return 0.0; }
        self.eta * self.sigma * (qty.abs() / self.adv).powf(self.alpha)
    }

    fn perm_impact(&self, qty: f64) -> f64 {
        if self.adv <= 0.0 { return 0.0; }
        self.gamma * self.sigma * (qty.abs() / self.adv).powf(self.alpha)
    }
}

/// Implementation cost estimator: given a proposed execution plan,
/// estimate total impact cost.
pub fn estimate_total_impact<M: ImpactModel>(
    model:      &M,
    trades:     &[f64],
    time_steps: &[f64],
) -> f64 {
    assert_eq!(trades.len(), time_steps.len(), "trades and time_steps must have same length");
    trades.iter().zip(time_steps.iter())
        .map(|(&qty, &dt)| model.temp_impact(qty, dt) * qty.abs())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_impact_scales_with_rate() {
        let m = LinearImpact::new(0.01, 0.005);
        let i1 = m.temp_impact(100.0, 10.0);
        let i2 = m.temp_impact(100.0, 5.0);
        // Faster trading → higher impact
        assert!(i2 > i1, "faster trade should have higher impact: {} vs {}", i2, i1);
    }

    #[test]
    fn sqrt_impact_positive() {
        let m = SquareRootImpact::us_equity(0.02, 1_000_000.0);
        assert!(m.temp_impact(50_000.0, 3600.0) > 0.0);
    }

    #[test]
    fn power_law_concave() {
        // α < 1 → concave (halving qty reduces impact by more than half)
        let m = PowerLawImpact::new(0.3, 0.02, 1e6, 0.5, 0.1);
        let i1 = m.temp_impact(10_000.0, 1.0);
        let i2 = m.temp_impact(40_000.0, 1.0);
        // i2/i1 should be < 4 (concave)
        assert!(i2 / i1 < 4.0, "expected concave scaling, ratio={}", i2/i1);
    }
}
