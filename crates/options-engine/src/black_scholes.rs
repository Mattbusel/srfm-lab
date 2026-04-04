use crate::OptionsError;
use std::f64::consts::{PI, SQRT_2};

/// Standard normal CDF using Hart's approximation (accurate to ~1e-7)
pub fn norm_cdf(x: f64) -> f64 {
    if x < -8.0 { return 0.0; }
    if x > 8.0 { return 1.0; }
    0.5 * (1.0 + erf_approx(x / SQRT_2))
}

/// Standard normal PDF
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Error function via numerical approximation (Abramowitz & Stegun 7.1.26, max |err| < 1.5e-7)
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    // Horner's method evaluation
    let t = 1.0 / (1.0 + 0.3275911_f64 * x_abs);
    let poly = t * (0.254829592_f64
        + t * (-0.284496736_f64
        + t * (1.421413741_f64
        + t * (-1.453152027_f64
        + t * 1.061405429_f64))));
    sign * (1.0_f64 - poly * (-x_abs * x_abs).exp())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    pub fn sign(self) -> f64 {
        match self {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        }
    }

    pub fn flip(self) -> Self {
        match self {
            OptionType::Call => OptionType::Put,
            OptionType::Put => OptionType::Call,
        }
    }
}

/// All first-order and second-order Greeks
#[derive(Debug, Clone, Copy)]
pub struct Greeks {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
    pub vanna: f64,   // d(delta)/d(vol) = d(vega)/dS
    pub volga: f64,   // d^2(price)/d(vol)^2
    pub charm: f64,   // d(delta)/dt
    pub speed: f64,   // d(gamma)/dS
}

pub struct BlackScholes;

impl BlackScholes {
    /// Compute d1 and d2 from BSM formula
    pub fn d1_d2(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> (f64, f64) {
        let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        (d1, d2)
    }

    /// BSM price
    pub fn price(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, opt_type: OptionType) -> f64 {
        if t <= 0.0 {
            return match opt_type {
                OptionType::Call => (s - k).max(0.0),
                OptionType::Put => (k - s).max(0.0),
            };
        }
        let (d1, d2) = Self::d1_d2(s, k, r, q, sigma, t);
        let df = (-r * t).exp();
        let fwd_df = (-q * t).exp();
        match opt_type {
            OptionType::Call => s * fwd_df * norm_cdf(d1) - k * df * norm_cdf(d2),
            OptionType::Put => k * df * norm_cdf(-d2) - s * fwd_df * norm_cdf(-d1),
        }
    }

    /// Delta: dP/dS
    pub fn delta(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, opt_type: OptionType) -> f64 {
        if t <= 0.0 {
            return match opt_type {
                OptionType::Call => if s > k { 1.0 } else { 0.0 },
                OptionType::Put => if s < k { -1.0 } else { 0.0 },
            };
        }
        let (d1, _) = Self::d1_d2(s, k, r, q, sigma, t);
        let fwd_df = (-q * t).exp();
        match opt_type {
            OptionType::Call => fwd_df * norm_cdf(d1),
            OptionType::Put => fwd_df * (norm_cdf(d1) - 1.0),
        }
    }

    /// Gamma: d^2P/dS^2
    pub fn gamma(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
        if t <= 0.0 { return 0.0; }
        let (d1, _) = Self::d1_d2(s, k, r, q, sigma, t);
        (-q * t).exp() * norm_pdf(d1) / (s * sigma * t.sqrt())
    }

    /// Theta: dP/dt (per calendar day, annualized by /365)
    pub fn theta(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, opt_type: OptionType) -> f64 {
        if t <= 0.0 { return 0.0; }
        let (d1, d2) = Self::d1_d2(s, k, r, q, sigma, t);
        let fwd_df = (-q * t).exp();
        let df = (-r * t).exp();
        let common = -s * fwd_df * norm_pdf(d1) * sigma / (2.0 * t.sqrt());
        let theta_annual = match opt_type {
            OptionType::Call => common - r * k * df * norm_cdf(d2) + q * s * fwd_df * norm_cdf(d1),
            OptionType::Put => common + r * k * df * norm_cdf(-d2) - q * s * fwd_df * norm_cdf(-d1),
        };
        theta_annual / 365.0
    }

    /// Vega: dP/d(sigma), per 1% move in vol
    pub fn vega(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
        if t <= 0.0 { return 0.0; }
        let (d1, _) = Self::d1_d2(s, k, r, q, sigma, t);
        s * (-q * t).exp() * norm_pdf(d1) * t.sqrt() / 100.0
    }

    /// Rho: dP/dr, per 1% move in rate
    pub fn rho(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, opt_type: OptionType) -> f64 {
        if t <= 0.0 { return 0.0; }
        let (_, d2) = Self::d1_d2(s, k, r, q, sigma, t);
        let df = (-r * t).exp();
        match opt_type {
            OptionType::Call => k * t * df * norm_cdf(d2) / 100.0,
            OptionType::Put => -k * t * df * norm_cdf(-d2) / 100.0,
        }
    }

    /// Vanna: d(delta)/d(vol) = d(vega)/dS
    pub fn vanna(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
        if t <= 0.0 { return 0.0; }
        let (d1, d2) = Self::d1_d2(s, k, r, q, sigma, t);
        -(-q * t).exp() * norm_pdf(d1) * d2 / sigma
    }

    /// Volga (vomma): d^2P/d(sigma)^2
    pub fn volga(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
        if t <= 0.0 { return 0.0; }
        let (d1, d2) = Self::d1_d2(s, k, r, q, sigma, t);
        s * (-q * t).exp() * norm_pdf(d1) * t.sqrt() * d1 * d2 / sigma
    }

    /// Charm: d(delta)/dt (delta decay)
    pub fn charm(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, opt_type: OptionType) -> f64 {
        if t <= 0.0 { return 0.0; }
        let (d1, d2) = Self::d1_d2(s, k, r, q, sigma, t);
        let fwd_df = (-q * t).exp();
        let common = fwd_df * norm_pdf(d1) * (2.0 * (r - q) * t - d2 * sigma * t.sqrt()) / (2.0 * t * sigma * t.sqrt());
        match opt_type {
            OptionType::Call => -q * fwd_df * norm_cdf(d1) + common,
            OptionType::Put => q * fwd_df * norm_cdf(-d1) + common,
        }
    }

    /// Speed: d(gamma)/dS
    pub fn speed(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
        if t <= 0.0 { return 0.0; }
        let (d1, _) = Self::d1_d2(s, k, r, q, sigma, t);
        let gamma = Self::gamma(s, k, r, q, sigma, t);
        -gamma / s * (d1 / (sigma * t.sqrt()) + 1.0)
    }

    /// Compute all Greeks at once (more efficient than calling individually)
    pub fn all_greeks(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, opt_type: OptionType) -> Greeks {
        Greeks {
            price: Self::price(s, k, r, q, sigma, t, opt_type),
            delta: Self::delta(s, k, r, q, sigma, t, opt_type),
            gamma: Self::gamma(s, k, r, q, sigma, t),
            theta: Self::theta(s, k, r, q, sigma, t, opt_type),
            vega: Self::vega(s, k, r, q, sigma, t),
            rho: Self::rho(s, k, r, q, sigma, t, opt_type),
            vanna: Self::vanna(s, k, r, q, sigma, t),
            volga: Self::volga(s, k, r, q, sigma, t),
            charm: Self::charm(s, k, r, q, sigma, t, opt_type),
            speed: Self::speed(s, k, r, q, sigma, t),
        }
    }

    /// Put-Call parity: C - P = S*exp(-q*T) - K*exp(-r*T)
    pub fn put_call_parity_check(
        call_price: f64, put_price: f64,
        s: f64, k: f64, r: f64, q: f64, t: f64
    ) -> f64 {
        let lhs = call_price - put_price;
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        (lhs - rhs).abs()
    }

    /// Implied volatility via Newton-Raphson, with Brent fallback
    pub fn implied_vol(
        market_price: f64,
        s: f64, k: f64, r: f64, q: f64, t: f64,
        opt_type: OptionType,
        max_iter: usize,
        tol: f64,
    ) -> Result<f64, OptionsError> {
        if t <= 0.0 {
            return Err(OptionsError::InvalidParameter("Time to expiry must be positive".into()));
        }
        // Validate price bounds
        let intrinsic = match opt_type {
            OptionType::Call => (s * (-q * t).exp() - k * (-r * t).exp()).max(0.0),
            OptionType::Put => (k * (-r * t).exp() - s * (-q * t).exp()).max(0.0),
        };
        if market_price < intrinsic - 1e-8 {
            return Err(OptionsError::InvalidParameter(format!(
                "Market price {:.6} below intrinsic value {:.6}", market_price, intrinsic
            )));
        }

        // Newton-Raphson
        let nr_result = Self::implied_vol_newton(market_price, s, k, r, q, t, opt_type, max_iter, tol);
        if nr_result.is_ok() {
            return nr_result;
        }

        // Brent's method fallback
        Self::implied_vol_brent(market_price, s, k, r, q, t, opt_type, max_iter, tol)
    }

    fn implied_vol_newton(
        market_price: f64,
        s: f64, k: f64, r: f64, q: f64, t: f64,
        opt_type: OptionType,
        max_iter: usize,
        tol: f64,
    ) -> Result<f64, OptionsError> {
        // Initial guess: Brenner-Subrahmanyam approximation
        let atm_vol = (2.0 * PI / t).sqrt() * market_price / s;
        let mut sigma = atm_vol.max(0.001).min(5.0);

        for _ in 0..max_iter {
            let price = Self::price(s, k, r, q, sigma, t, opt_type);
            let vega_raw = {
                let (d1, _) = Self::d1_d2(s, k, r, q, sigma, t);
                s * (-q * t).exp() * norm_pdf(d1) * t.sqrt()
            };
            let diff = price - market_price;
            if diff.abs() < tol {
                if sigma > 0.0 && sigma < 20.0 {
                    return Ok(sigma);
                }
            }
            if vega_raw.abs() < 1e-12 {
                break;
            }
            sigma -= diff / vega_raw;
            sigma = sigma.max(1e-6).min(20.0);
        }
        Err(OptionsError::ConvergenceFailure("Newton-Raphson did not converge".into()))
    }

    fn implied_vol_brent(
        market_price: f64,
        s: f64, k: f64, r: f64, q: f64, t: f64,
        opt_type: OptionType,
        max_iter: usize,
        tol: f64,
    ) -> Result<f64, OptionsError> {
        let f = |sigma: f64| Self::price(s, k, r, q, sigma, t, opt_type) - market_price;

        let mut a = 1e-6_f64;
        let mut b = 10.0_f64;
        let mut fa = f(a);
        let mut fb = f(b);

        // Expand bracket if needed
        if fa * fb > 0.0 {
            // Try wider bracket
            b = 20.0;
            fb = f(b);
            if fa * fb > 0.0 {
                return Err(OptionsError::ConvergenceFailure(
                    "Brent: could not bracket root".into()
                ));
            }
        }

        let mut c = a;
        let mut fc = fa;
        let mut d = b - a;
        let mut e = d;

        for _ in 0..max_iter {
            if fb.abs() < fa.abs() {
                a = b; b = c; c = a;
                fa = fb; fb = fc; fc = fa;
            }
            let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * tol;
            let xm = 0.5 * (c - b);
            if xm.abs() <= tol1 || fb.abs() < tol {
                return Ok(b);
            }
            if e.abs() >= tol1 && fa.abs() > fb.abs() {
                let s_brent = fb / fa;
                let (p, q_brent) = if a == c {
                    (2.0 * xm * s_brent, 1.0 - s_brent)
                } else {
                    let q2 = fa / fc;
                    let r2 = fb / fc;
                    (s_brent * (2.0 * xm * q2 * (q2 - r2) - (b - a) * (r2 - 1.0)),
                     (q2 - 1.0) * (r2 - 1.0) * (s_brent - 1.0))
                };
                let (mut p, mut q_brent) = if p > 0.0 { (p, -q_brent) } else { (-p, q_brent) };
                if 2.0 * p < (3.0 * xm * q_brent - (tol1 * q_brent).abs()).min(e.abs() * q_brent.abs()) {
                    e = d;
                    d = p / q_brent;
                } else {
                    d = xm; e = d;
                }
            } else {
                d = xm; e = d;
            }
            a = b; fa = fb;
            b += if d.abs() > tol1 { d } else if xm > 0.0 { tol1 } else { -tol1 };
            fb = f(b);
            if fb * fc > 0.0 {
                c = a; fc = fa; d = b - a; e = d;
            }
        }
        Err(OptionsError::ConvergenceFailure("Brent's method did not converge".into()))
    }

    /// Forward price
    pub fn forward(s: f64, r: f64, q: f64, t: f64) -> f64 {
        s * ((r - q) * t).exp()
    }

    /// Convert implied vol between different moneyness conventions
    pub fn vol_to_delta_space(
        sigma: f64, s: f64, k: f64, r: f64, q: f64, t: f64,
    ) -> f64 {
        if t <= 0.0 { return 0.5; }
        let (d1, _) = Self::d1_d2(s, k, r, q, sigma, t);
        (-q * t).exp() * norm_cdf(d1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: f64 = 100.0;
    const K: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.02;
    const SIGMA: f64 = 0.2;
    const T: f64 = 1.0;

    #[test]
    fn test_bsm_call_price_atm() {
        let call = BlackScholes::price(S, K, R, Q, SIGMA, T, OptionType::Call);
        // S=100, K=100, r=0.05, q=0.02, sigma=0.2, T=1: approximately 9.22
        assert!(call > 7.0 && call < 12.0, "Call price = {:.4} out of expected range", call);
    }

    #[test]
    fn test_put_call_parity() {
        let call = BlackScholes::price(S, K, R, Q, SIGMA, T, OptionType::Call);
        let put = BlackScholes::price(S, K, R, Q, SIGMA, T, OptionType::Put);
        let parity_err = BlackScholes::put_call_parity_check(call, put, S, K, R, Q, T);
        assert!(parity_err < 1e-10, "Put-call parity error = {:.2e}", parity_err);
    }

    #[test]
    fn test_implied_vol_roundtrip() {
        let market_price = BlackScholes::price(S, K, R, Q, SIGMA, T, OptionType::Call);
        let iv = BlackScholes::implied_vol(market_price, S, K, R, Q, T, OptionType::Call, 100, 1e-8).unwrap();
        assert!((iv - SIGMA).abs() < 1e-6, "IV={:.6} expected={}", iv, SIGMA);
    }

    #[test]
    fn test_implied_vol_itm_put() {
        let sigma = 0.25;
        let market_price = BlackScholes::price(S, 90.0, R, Q, sigma, T, OptionType::Put);
        let iv = BlackScholes::implied_vol(market_price, S, 90.0, R, Q, T, OptionType::Put, 100, 1e-8).unwrap();
        assert!((iv - sigma).abs() < 1e-5, "IV={:.6} expected={}", iv, sigma);
    }

    #[test]
    fn test_delta_bounds() {
        let call_delta = BlackScholes::delta(S, K, R, Q, SIGMA, T, OptionType::Call);
        let put_delta = BlackScholes::delta(S, K, R, Q, SIGMA, T, OptionType::Put);
        assert!(call_delta > 0.0 && call_delta < 1.0);
        assert!(put_delta > -1.0 && put_delta < 0.0);
        // Call delta - Put delta = exp(-q*T) for same strike
        let diff = call_delta - put_delta;
        let expected = (-Q * T).exp();
        assert!((diff - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_positive() {
        let gamma = BlackScholes::gamma(S, K, R, Q, SIGMA, T);
        assert!(gamma > 0.0);
    }

    #[test]
    fn test_vega_positive() {
        let vega = BlackScholes::vega(S, K, R, Q, SIGMA, T);
        assert!(vega > 0.0);
    }

    #[test]
    fn test_theta_negative_long() {
        let theta_call = BlackScholes::theta(S, K, R, Q, SIGMA, T, OptionType::Call);
        assert!(theta_call < 0.0, "Long call theta should be negative, got {}", theta_call);
    }

    #[test]
    fn test_all_greeks_consistency() {
        let greeks = BlackScholes::all_greeks(S, K, R, Q, SIGMA, T, OptionType::Call);
        let price_check = BlackScholes::price(S, K, R, Q, SIGMA, T, OptionType::Call);
        assert!((greeks.price - price_check).abs() < 1e-12);
        let delta_check = BlackScholes::delta(S, K, R, Q, SIGMA, T, OptionType::Call);
        assert!((greeks.delta - delta_check).abs() < 1e-12);
    }

    #[test]
    fn test_intrinsic_at_expiry() {
        let call_itm = BlackScholes::price(110.0, K, R, Q, SIGMA, 0.0, OptionType::Call);
        assert!((call_itm - 10.0).abs() < 1e-10);
        let call_otm = BlackScholes::price(90.0, K, R, Q, SIGMA, 0.0, OptionType::Call);
        assert_eq!(call_otm, 0.0);
    }

    #[test]
    fn test_norm_cdf_properties() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!(norm_cdf(5.0) > 0.9999);
        assert!(norm_cdf(-5.0) < 0.0001);
        assert!((norm_cdf(1.96) - 0.975).abs() < 0.001);
    }

    #[test]
    fn test_high_vol_implied_vol() {
        let sigma = 0.8;
        let price = BlackScholes::price(S, K, R, Q, sigma, 0.25, OptionType::Call);
        let iv = BlackScholes::implied_vol(price, S, K, R, Q, 0.25, OptionType::Call, 200, 1e-8).unwrap();
        assert!((iv - sigma).abs() < 1e-5);
    }
}
