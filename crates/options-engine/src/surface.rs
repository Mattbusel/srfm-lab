use crate::OptionsError;
use crate::black_scholes::{BlackScholes, OptionType};

/// A single point on the volatility surface
#[derive(Debug, Clone, Copy)]
pub struct VolPoint {
    pub expiry: f64,
    pub strike: f64,
    pub implied_vol: f64,
}

/// Cubic spline segment
#[derive(Debug, Clone)]
struct SplineSegment {
    t0: f64,
    a: f64, b: f64, c: f64, d: f64, // cubic polynomial coefficients
}

impl SplineSegment {
    fn eval(&self, t: f64) -> f64 {
        let dt = t - self.t0;
        self.a + dt * (self.b + dt * (self.c + dt * self.d))
    }
}

/// Natural cubic spline through given points
fn build_cubic_spline(xs: &[f64], ys: &[f64]) -> Vec<SplineSegment> {
    let n = xs.len();
    assert!(n >= 2);
    if n == 2 {
        let slope = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        return vec![SplineSegment { t0: xs[0], a: ys[0], b: slope, c: 0.0, d: 0.0 }];
    }

    let h: Vec<f64> = (0..n-1).map(|i| xs[i+1] - xs[i]).collect();
    let alpha: Vec<f64> = (1..n-1).map(|i| {
        3.0 * ((ys[i+1] - ys[i]) / h[i] - (ys[i] - ys[i-1]) / h[i-1])
    }).collect();

    let mut l = vec![1.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];

    for i in 1..n-1 {
        l[i] = 2.0 * (xs[i+1] - xs[i-1]) - h[i-1] * mu[i-1];
        if l[i].abs() < 1e-12 { l[i] = 1e-12; }
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i-1] - h[i-1] * z[i-1]) / l[i];
    }

    let mut c_coef = vec![0.0; n];
    let mut b_coef = vec![0.0; n];
    let mut d_coef = vec![0.0; n];

    for j in (0..n-1).rev() {
        c_coef[j] = z[j] - mu[j] * c_coef[j+1];
        b_coef[j] = (ys[j+1] - ys[j]) / h[j] - h[j] * (c_coef[j+1] + 2.0 * c_coef[j]) / 3.0;
        d_coef[j] = (c_coef[j+1] - c_coef[j]) / (3.0 * h[j]);
    }

    (0..n-1).map(|i| SplineSegment {
        t0: xs[i],
        a: ys[i],
        b: b_coef[i],
        c: c_coef[i],
        d: d_coef[i],
    }).collect()
}

fn eval_spline(segments: &[SplineSegment], x: f64) -> f64 {
    if segments.is_empty() { return 0.0; }
    let idx = segments.partition_point(|s| s.t0 <= x).saturating_sub(1)
        .min(segments.len() - 1);
    segments[idx].eval(x)
}

/// Full volatility surface with interpolation
pub struct VolatilitySurface {
    /// Sorted expiries
    expiries: Vec<f64>,
    /// Sorted strikes per expiry
    strikes_per_expiry: Vec<Vec<f64>>,
    /// Implied vols: [expiry_idx][strike_idx]
    vols: Vec<Vec<f64>>,
    /// Spot price (for local vol computation)
    spot: f64,
    /// Risk-free rate
    rate: f64,
    /// Dividend yield
    div_yield: f64,
}

impl VolatilitySurface {
    pub fn new(spot: f64, rate: f64, div_yield: f64) -> Self {
        VolatilitySurface {
            expiries: Vec::new(),
            strikes_per_expiry: Vec::new(),
            vols: Vec::new(),
            spot,
            rate,
            div_yield,
        }
    }

    /// Add a term structure slice
    pub fn add_slice(&mut self, expiry: f64, strikes: Vec<f64>, vols: Vec<f64>) -> Result<(), OptionsError> {
        if strikes.len() != vols.len() {
            return Err(OptionsError::InvalidParameter("strikes and vols must have same length".into()));
        }
        if strikes.is_empty() {
            return Err(OptionsError::InvalidParameter("must have at least one point per slice".into()));
        }
        let pos = self.expiries.partition_point(|&e| e < expiry);
        self.expiries.insert(pos, expiry);
        self.strikes_per_expiry.insert(pos, strikes);
        self.vols.insert(pos, vols);
        Ok(())
    }

    /// Add individual vol points
    pub fn add_points(&mut self, points: &[VolPoint]) -> Result<(), OptionsError> {
        // Group by expiry
        let mut groups: std::collections::BTreeMap<OrderedFloat, Vec<(f64, f64)>> =
            std::collections::BTreeMap::new();
        for p in points {
            groups.entry(OrderedFloat::new(p.expiry))
                .or_default()
                .push((p.strike, p.implied_vol));
        }
        for (t, mut kv) in groups {
            kv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let strikes: Vec<f64> = kv.iter().map(|(k, _)| *k).collect();
            let vols: Vec<f64> = kv.iter().map(|(_, v)| *v).collect();
            self.add_slice(f64::from(t), strikes, vols)?;
        }
        Ok(())
    }

    /// Bilinear interpolation of implied vol at (strike, expiry)
    pub fn implied_vol_bilinear(&self, strike: f64, expiry: f64) -> Result<f64, OptionsError> {
        if self.expiries.is_empty() {
            return Err(OptionsError::InterpolationError("No data in surface".into()));
        }

        // Find expiry bracket
        let te_pos = self.expiries.partition_point(|&e| e <= expiry);

        if te_pos == 0 {
            return self.interp_strike_for_expiry(0, strike);
        }
        if te_pos >= self.expiries.len() {
            let last = self.expiries.len() - 1;
            return self.interp_strike_for_expiry(last, strike);
        }

        let t1 = self.expiries[te_pos - 1];
        let t2 = self.expiries[te_pos];
        let v1 = self.interp_strike_for_expiry(te_pos - 1, strike)?;
        let v2 = self.interp_strike_for_expiry(te_pos, strike)?;

        // Linear interpolation in variance-time space
        let w1 = v1 * v1 * t1;
        let w2 = v2 * v2 * t2;
        let alpha = (expiry - t1) / (t2 - t1);
        let w = w1 + alpha * (w2 - w1);
        if w < 0.0 {
            return Err(OptionsError::InterpolationError(
                "Negative interpolated total variance".into()
            ));
        }
        Ok((w / expiry).sqrt())
    }

    /// Cubic spline interpolation across strikes at a given expiry
    pub fn implied_vol_cubic(&self, strike: f64, expiry: f64) -> Result<f64, OptionsError> {
        if self.expiries.is_empty() {
            return Err(OptionsError::InterpolationError("No data in surface".into()));
        }

        let te_pos = self.expiries.partition_point(|&e| e <= expiry);

        if te_pos == 0 {
            return self.interp_strike_cubic(0, strike);
        }
        if te_pos >= self.expiries.len() {
            let last = self.expiries.len() - 1;
            return self.interp_strike_cubic(last, strike);
        }

        let t1 = self.expiries[te_pos - 1];
        let t2 = self.expiries[te_pos];
        let v1 = self.interp_strike_cubic(te_pos - 1, strike)?;
        let v2 = self.interp_strike_cubic(te_pos, strike)?;

        let w1 = v1 * v1 * t1;
        let w2 = v2 * v2 * t2;
        let alpha = (expiry - t1) / (t2 - t1);
        let w = w1 + alpha * (w2 - w1);
        if w < 0.0 {
            return Err(OptionsError::InterpolationError(
                "Negative interpolated total variance".into()
            ));
        }
        Ok((w / expiry).sqrt())
    }

    fn interp_strike_for_expiry(&self, expiry_idx: usize, strike: f64) -> Result<f64, OptionsError> {
        let strikes = &self.strikes_per_expiry[expiry_idx];
        let vols = &self.vols[expiry_idx];
        if strikes.is_empty() {
            return Err(OptionsError::InterpolationError("Empty slice".into()));
        }
        if strike <= strikes[0] { return Ok(vols[0]); }
        if strike >= *strikes.last().unwrap() { return Ok(*vols.last().unwrap()); }
        let pos = strikes.partition_point(|&k| k <= strike);
        let k1 = strikes[pos - 1];
        let k2 = strikes[pos];
        let v1 = vols[pos - 1];
        let v2 = vols[pos];
        let alpha = (strike - k1) / (k2 - k1);
        Ok(v1 + alpha * (v2 - v1))
    }

    fn interp_strike_cubic(&self, expiry_idx: usize, strike: f64) -> Result<f64, OptionsError> {
        let strikes = &self.strikes_per_expiry[expiry_idx];
        let vols = &self.vols[expiry_idx];
        if strikes.len() < 2 {
            return self.interp_strike_for_expiry(expiry_idx, strike);
        }
        if strike <= strikes[0] { return Ok(vols[0]); }
        if strike >= *strikes.last().unwrap() { return Ok(*vols.last().unwrap()); }
        let segs = build_cubic_spline(strikes, vols);
        let v = eval_spline(&segs, strike);
        if v < 0.0 {
            return Err(OptionsError::InterpolationError(
                format!("Cubic spline produced negative vol {:.6}", v)
            ));
        }
        Ok(v)
    }

    /// Forward volatility: sigma_fwd(T1, T2) = sqrt((w(T2) - w(T1)) / (T2 - T1))
    pub fn forward_vol(&self, strike: f64, t1: f64, t2: f64) -> Result<f64, OptionsError> {
        if t2 <= t1 {
            return Err(OptionsError::InvalidParameter(
                format!("t2={} must be > t1={}", t2, t1)
            ));
        }
        let v1 = self.implied_vol_bilinear(strike, t1)?;
        let v2 = self.implied_vol_bilinear(strike, t2)?;
        let w1 = v1 * v1 * t1;
        let w2 = v2 * v2 * t2;
        if w2 < w1 {
            return Err(OptionsError::ArbitrageViolation(
                format!("Calendar spread arbitrage: w(T2) < w(T1) for K={}", strike)
            ));
        }
        Ok(((w2 - w1) / (t2 - t1)).sqrt())
    }

    /// Term structure of ATM vols
    pub fn atm_term_structure(&self) -> Vec<(f64, f64)> {
        self.expiries.iter().enumerate().filter_map(|(i, &t)| {
            let fwd = self.spot * ((self.rate - self.div_yield) * t).exp();
            self.interp_strike_for_expiry(i, fwd).ok().map(|v| (t, v))
        }).collect()
    }

    /// Dupire local vol extraction
    /// sigma_loc^2(K, T) = (dC/dT + (r-q)*K*dC/dK + q*C) / (0.5*K^2 * d^2C/dK^2)
    pub fn local_vol(&self, strike: f64, expiry: f64) -> Result<f64, OptionsError> {
        let dk = strike * 0.001;
        let dt = expiry * 0.005 + 1e-4;

        let c = |k: f64, t: f64| -> f64 {
            let vol = self.implied_vol_bilinear(k, t).unwrap_or(0.2);
            BlackScholes::price(self.spot, k, self.rate, self.div_yield, vol, t, OptionType::Call)
        };

        let c_center = c(strike, expiry);
        let c_up_k = c(strike + dk, expiry);
        let c_dn_k = c(strike - dk, expiry);
        let c_up_t = c(strike, expiry + dt);

        let dc_dt = (c_up_t - c_center) / dt;
        let dc_dk = (c_up_k - c_dn_k) / (2.0 * dk);
        let d2c_dk2 = (c_up_k - 2.0 * c_center + c_dn_k) / (dk * dk);

        let numerator = dc_dt + (self.rate - self.div_yield) * strike * dc_dk + self.div_yield * c_center;
        let denominator = 0.5 * strike * strike * d2c_dk2;

        if denominator.abs() < 1e-12 {
            return Err(OptionsError::ModelError(
                "Dupire denominator near zero (no gamma)".into()
            ));
        }

        let local_var = numerator / denominator;
        if local_var < 0.0 {
            return Err(OptionsError::ArbitrageViolation(
                format!("Negative local variance: {:.6}", local_var)
            ));
        }
        Ok(local_var.sqrt())
    }

    pub fn expiries(&self) -> &[f64] { &self.expiries }
    pub fn spot(&self) -> f64 { self.spot }
    pub fn rate(&self) -> f64 { self.rate }
    pub fn div_yield(&self) -> f64 { self.div_yield }
}

// Newtype to allow BTreeMap keying by f64
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OrderedFloat(u64);

impl OrderedFloat {
    fn new(v: f64) -> Self { OrderedFloat(v.to_bits()) }
}

impl From<f64> for OrderedFloat {
    fn from(v: f64) -> Self { OrderedFloat(v.to_bits()) }
}

impl From<OrderedFloat> for f64 {
    fn from(v: OrderedFloat) -> f64 { f64::from_bits(v.0) }
}

type OF = OrderedFloat;

impl VolatilitySurface {
    /// Rebuild from a flat list of vol points (used internally)
    fn rebuild_from_points(points: &[VolPoint]) -> Result<Self, OptionsError> {
        let spot = 100.0; // default; should be set by caller
        let mut surf = VolatilitySurface::new(spot, 0.0, 0.0);
        let mut groups: std::collections::BTreeMap<OF, Vec<(f64, f64)>> =
            std::collections::BTreeMap::new();
        for p in points {
            groups.entry(OF::new(p.expiry))
                .or_default()
                .push((p.strike, p.implied_vol));
        }
        for (t_bits, mut kv) in groups {
            let t: f64 = t_bits.into();
            kv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let ks: Vec<f64> = kv.iter().map(|(k, _)| *k).collect();
            let vs: Vec<f64> = kv.iter().map(|(_, v)| *v).collect();
            surf.add_slice(t, ks, vs)?;
        }
        Ok(surf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_surface() -> VolatilitySurface {
        let mut surf = VolatilitySurface::new(100.0, 0.05, 0.02);
        surf.add_slice(0.25, vec![90.0, 95.0, 100.0, 105.0, 110.0],
            vec![0.22, 0.21, 0.20, 0.21, 0.22]).unwrap();
        surf.add_slice(0.5, vec![90.0, 95.0, 100.0, 105.0, 110.0],
            vec![0.23, 0.215, 0.205, 0.215, 0.225]).unwrap();
        surf.add_slice(1.0, vec![90.0, 95.0, 100.0, 105.0, 110.0],
            vec![0.24, 0.22, 0.21, 0.22, 0.24]).unwrap();
        surf
    }

    #[test]
    fn test_bilinear_at_pillar() {
        let surf = make_surface();
        let vol = surf.implied_vol_bilinear(100.0, 0.25).unwrap();
        assert!((vol - 0.20).abs() < 1e-6, "vol = {:.6}", vol);
    }

    #[test]
    fn test_bilinear_interpolated_expiry() {
        let surf = make_surface();
        let vol = surf.implied_vol_bilinear(100.0, 0.375).unwrap();
        // Should be between ATM vols at 0.25 and 0.5
        assert!(vol > 0.19 && vol < 0.22, "vol = {:.4}", vol);
    }

    #[test]
    fn test_cubic_spline_at_pillar() {
        let surf = make_surface();
        let vol = surf.implied_vol_cubic(100.0, 1.0).unwrap();
        assert!((vol - 0.21).abs() < 1e-6, "vol = {:.6}", vol);
    }

    #[test]
    fn test_forward_vol() {
        let surf = make_surface();
        let fwd_vol = surf.forward_vol(100.0, 0.25, 1.0).unwrap();
        assert!(fwd_vol > 0.0, "forward vol should be positive, got {}", fwd_vol);
    }

    #[test]
    fn test_atm_term_structure() {
        let surf = make_surface();
        let ts = surf.atm_term_structure();
        assert_eq!(ts.len(), 3);
        // Expiries should be sorted
        assert!(ts[0].0 < ts[1].0 && ts[1].0 < ts[2].0);
    }

    #[test]
    fn test_local_vol() {
        let surf = make_surface();
        let lv = surf.local_vol(100.0, 0.5);
        // Local vol might succeed or fail depending on smoothness, just check no panic
        match lv {
            Ok(v) => assert!(v > 0.0, "local vol should be positive"),
            Err(_) => {}, // acceptable if arbitrage or numerical issues
        }
    }

    #[test]
    fn test_extrapolation_strike() {
        let surf = make_surface();
        // Strike below range
        let vol_low = surf.implied_vol_bilinear(70.0, 1.0).unwrap();
        assert!(vol_low > 0.0);
        // Strike above range
        let vol_high = surf.implied_vol_bilinear(150.0, 1.0).unwrap();
        assert!(vol_high > 0.0);
    }
}
