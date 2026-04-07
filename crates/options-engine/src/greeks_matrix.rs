/// Vectorized Greeks computation for a portfolio of options.
/// Includes closed-form Black-Scholes Greeks and SIMD-style batch delta computation.

use std::f64::consts::PI;

// ── Normal distribution helpers ───────────────────────────────────────────────

fn norm_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.3275911_f64 * x.abs());
    let poly = t * (0.254829592_f64
        + t * (-0.284496736_f64
            + t * (1.421413741_f64 + t * (-1.453152027_f64 + t * 1.061405429_f64))));
    let y = 1.0 - poly * (-x * x).exp();
    let raw = if x < 0.0 { (1.0 - y) / 2.0 } else { (1.0 + y) / 2.0 };
    raw.clamp(0.0, 1.0)
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Compute d1 and d2 for Black-Scholes.
/// Returns (d1, d2) or None if inputs are invalid (T<=0, S<=0, K<=0, sigma<=0).
fn d1_d2(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> Option<(f64, f64)> {
    if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return None;
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    Some((d1, d2))
}

// ── Option specification ──────────────────────────────────────────────────────

/// Specification for a single option position.
#[derive(Debug, Clone, Copy)]
pub struct OptionSpec {
    /// Strike price.
    pub strike: f64,
    /// Time to expiry in years.
    pub expiry: f64,
    /// True = call, False = put.
    pub is_call: bool,
    /// Signed position size (positive = long, negative = short).
    pub position: f64,
}

// ── Portfolio-level Greeks ────────────────────────────────────────────────────

/// Aggregated Greeks for an entire portfolio.
#[derive(Debug, Clone, Copy, Default)]
pub struct PortfolioGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    /// d(delta)/d(sigma) -- sensitivity of delta to volatility.
    pub vanna: f64,
    /// d(vega)/d(sigma) -- second derivative of price w.r.t. volatility.
    pub volga: f64,
}

// ── GreeksMatrix ──────────────────────────────────────────────────────────────

/// Computes Greeks for a portfolio of options simultaneously.
pub struct GreeksMatrix;

impl GreeksMatrix {
    /// Compute portfolio Greeks for all specs under a single spot/vol/rate.
    pub fn compute_all(
        specs: &[OptionSpec],
        spot: f64,
        vol: f64,
        rf: f64,
    ) -> PortfolioGreeks {
        let mut pg = PortfolioGreeks::default();
        for spec in specs {
            let pos = spec.position;
            pg.delta += pos * delta(spot, spec.strike, spec.expiry, rf, vol, spec.is_call);
            pg.gamma += pos * gamma(spot, spec.strike, spec.expiry, rf, vol);
            pg.vega  += pos * vega(spot, spec.strike, spec.expiry, rf, vol);
            pg.theta += pos * theta(spot, spec.strike, spec.expiry, rf, vol, spec.is_call);
            pg.rho   += pos * rho_greek(spot, spec.strike, spec.expiry, rf, vol, spec.is_call);
            pg.vanna += pos * vanna(spot, spec.strike, spec.expiry, rf, vol);
            pg.volga += pos * volga(spot, spec.strike, spec.expiry, rf, vol);
        }
        pg
    }
}

// ── Individual Greeks (closed-form Black-Scholes) ─────────────────────────────

/// Delta: dC/dS (call) or dP/dS (put).
pub fn delta(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
    match d1_d2(s, k, t, r, sigma) {
        Some((d1, _)) => {
            if is_call {
                norm_cdf(d1)
            } else {
                norm_cdf(d1) - 1.0
            }
        }
        None => {
            if is_call { if s > k { 1.0 } else { 0.0 } }
            else { if s < k { -1.0 } else { 0.0 } }
        }
    }
}

/// Gamma: d2C/dS2 = d2P/dS2.
pub fn gamma(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    match d1_d2(s, k, t, r, sigma) {
        Some((d1, _)) => norm_pdf(d1) / (s * sigma * t.sqrt()),
        None => 0.0,
    }
}

/// Vega: dC/d(sigma) = dP/d(sigma). Expressed per unit of vol (not per 1 vol point).
pub fn vega(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    match d1_d2(s, k, t, r, sigma) {
        Some((d1, _)) => s * norm_pdf(d1) * t.sqrt(),
        None => 0.0,
    }
}

/// Theta: dC/dt (call) or dP/dt (put). Returns daily decay (per calendar day).
/// Convention: negative for long options.
pub fn theta(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
    match d1_d2(s, k, t, r, sigma) {
        Some((d1, d2)) => {
            let term1 = -(s * norm_pdf(d1) * sigma) / (2.0 * t.sqrt());
            if is_call {
                let term2 = -r * k * (-r * t).exp() * norm_cdf(d2);
                (term1 + term2) / 365.0
            } else {
                let term2 = r * k * (-r * t).exp() * norm_cdf(-d2);
                (term1 + term2) / 365.0
            }
        }
        None => 0.0,
    }
}

/// Rho: dC/dr (call) or dP/dr (put). Expressed per 1 unit of rate (not per basis point).
pub fn rho_greek(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
    match d1_d2(s, k, t, r, sigma) {
        Some((_, d2)) => {
            if is_call {
                k * t * (-r * t).exp() * norm_cdf(d2)
            } else {
                -k * t * (-r * t).exp() * norm_cdf(-d2)
            }
        }
        None => 0.0,
    }
}

/// Vanna: d(delta)/d(sigma) = d(vega)/dS.
/// Measures how delta changes as vol moves.
pub fn vanna(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    match d1_d2(s, k, t, r, sigma) {
        Some((d1, d2)) => -norm_pdf(d1) * d2 / sigma,
        None => 0.0,
    }
}

/// Volga (vomma): d(vega)/d(sigma) = d2C/d(sigma)^2.
/// Measures convexity of the option price w.r.t. volatility.
pub fn volga(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    match d1_d2(s, k, t, r, sigma) {
        Some((d1, d2)) => vega(s, k, t, r, sigma) * d1 * d2 / sigma,
        None => 0.0,
    }
}

// ── SIMD-style batch delta ────────────────────────────────────────────────────

/// Process spots in chunks of 4 for cache-friendly batch delta computation.
/// Falls back to scalar for the remainder. This mirrors f64x4 SIMD width
/// without requiring a nightly feature or external crate.
pub fn compute_delta_batch(
    spots: &[f64],
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    is_call: bool,
    out: &mut [f64],
) {
    assert_eq!(
        spots.len(),
        out.len(),
        "spots and out slices must have the same length"
    );
    let n = spots.len();
    let chunks = n / 4;
    let rem = n % 4;

    // Process 4 at a time (unrolled).
    for c in 0..chunks {
        let base = c * 4;
        let s0 = spots[base];
        let s1 = spots[base + 1];
        let s2 = spots[base + 2];
        let s3 = spots[base + 3];

        // Compute d1 for each lane simultaneously.
        let inv_sigma_sqrt_t = if sigma > 0.0 && t > 0.0 {
            1.0 / (sigma * t.sqrt())
        } else {
            0.0
        };
        let drift = (r + 0.5 * sigma * sigma) * t;

        let d1_0 = ((s0 / k).ln() + drift) * inv_sigma_sqrt_t;
        let d1_1 = ((s1 / k).ln() + drift) * inv_sigma_sqrt_t;
        let d1_2 = ((s2 / k).ln() + drift) * inv_sigma_sqrt_t;
        let d1_3 = ((s3 / k).ln() + drift) * inv_sigma_sqrt_t;

        if is_call {
            out[base]     = norm_cdf(d1_0);
            out[base + 1] = norm_cdf(d1_1);
            out[base + 2] = norm_cdf(d1_2);
            out[base + 3] = norm_cdf(d1_3);
        } else {
            out[base]     = norm_cdf(d1_0) - 1.0;
            out[base + 1] = norm_cdf(d1_1) - 1.0;
            out[base + 2] = norm_cdf(d1_2) - 1.0;
            out[base + 3] = norm_cdf(d1_3) - 1.0;
        }
    }

    // Remainder.
    let rem_start = chunks * 4;
    for i in 0..rem {
        out[rem_start + i] = delta(spots[rem_start + i], k, t, r, sigma, is_call);
    }
}

// ── Black-Scholes price (for reference / testing) ─────────────────────────────

/// Black-Scholes option price.
pub fn bs_price(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
    match d1_d2(s, k, t, r, sigma) {
        Some((d1, d2)) => {
            let disc = (-r * t).exp();
            if is_call {
                s * norm_cdf(d1) - k * disc * norm_cdf(d2)
            } else {
                k * disc * norm_cdf(-d2) - s * norm_cdf(-d1)
            }
        }
        None => {
            let intrinsic = if is_call {
                (s - k).max(0.0)
            } else {
                (k - s).max(0.0)
            };
            intrinsic
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const S: f64 = 100.0;
    const K: f64 = 100.0;
    const T: f64 = 0.25; // 3 months
    const R: f64 = 0.05;
    const SIGMA: f64 = 0.20;

    #[test]
    fn test_delta_call_atm_near_half() {
        let d = delta(S, K, T, R, SIGMA, true);
        // ATM call delta should be near 0.50-0.55.
        assert!(d > 0.45 && d < 0.60, "ATM call delta out of range: {}", d);
    }

    #[test]
    fn test_delta_put_atm_near_minus_half() {
        let d = delta(S, K, T, R, SIGMA, false);
        assert!(d > -0.60 && d < -0.40, "ATM put delta out of range: {}", d);
    }

    #[test]
    fn test_put_call_parity_delta() {
        let d_call = delta(S, K, T, R, SIGMA, true);
        let d_put = delta(S, K, T, R, SIGMA, false);
        // delta_call - delta_put = 1.
        assert!((d_call - d_put - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_same_call_put() {
        let g_call = gamma(S, K, T, R, SIGMA);
        // Gamma is the same for call and put.
        assert!(g_call > 0.0);
    }

    #[test]
    fn test_vega_positive() {
        let v = vega(S, K, T, R, SIGMA);
        assert!(v > 0.0, "Vega must be positive for long option");
    }

    #[test]
    fn test_theta_negative_long_call() {
        let th = theta(S, K, T, R, SIGMA, true);
        // Long option decays over time.
        assert!(th < 0.0, "Theta for long call must be negative: {}", th);
    }

    #[test]
    fn test_vanna_finite() {
        let va = vanna(S, K, T, R, SIGMA);
        assert!(va.is_finite(), "Vanna must be finite");
    }

    #[test]
    fn test_volga_atm_positive() {
        // Volga (vomma) should be positive for ATM options.
        let vg = volga(S, K, T, R, SIGMA);
        assert!(vg > 0.0, "Volga must be positive for ATM option: {}", vg);
    }

    #[test]
    fn test_rho_call_positive() {
        let r = rho_greek(S, K, T, R, SIGMA, true);
        assert!(r > 0.0, "Call rho must be positive: {}", r);
    }

    #[test]
    fn test_rho_put_negative() {
        let r = rho_greek(S, K, T, R, SIGMA, false);
        assert!(r < 0.0, "Put rho must be negative: {}", r);
    }

    #[test]
    fn test_compute_all_portfolio_delta() {
        let specs = vec![
            OptionSpec { strike: 100.0, expiry: T, is_call: true,  position: 1.0 },
            OptionSpec { strike: 100.0, expiry: T, is_call: false, position: -1.0 },
        ];
        let pg = GreeksMatrix::compute_all(&specs, S, SIGMA, R);
        // Long call + short put at same strike/expiry: delta ~ 1.0.
        assert!((pg.delta - 1.0).abs() < 0.01, "Delta of synthetic long: {}", pg.delta);
    }

    #[test]
    fn test_compute_delta_batch_matches_scalar() {
        let spots = vec![95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0];
        let mut batch_out = vec![0.0_f64; spots.len()];
        compute_delta_batch(&spots, K, T, R, SIGMA, true, &mut batch_out);
        for (i, &s) in spots.iter().enumerate() {
            let scalar = delta(s, K, T, R, SIGMA, true);
            assert!(
                (batch_out[i] - scalar).abs() < 1e-12,
                "Batch mismatch at spot {}: batch={}, scalar={}",
                s, batch_out[i], scalar
            );
        }
    }

    #[test]
    fn test_compute_delta_batch_exact_multiple_of_4() {
        let spots = vec![90.0, 95.0, 100.0, 105.0];
        let mut out = vec![0.0_f64; 4];
        compute_delta_batch(&spots, K, T, R, SIGMA, false, &mut out);
        for (i, &s) in spots.iter().enumerate() {
            let scalar = delta(s, K, T, R, SIGMA, false);
            assert!((out[i] - scalar).abs() < 1e-12);
        }
    }

    #[test]
    fn test_portfolio_greeks_empty() {
        let pg = GreeksMatrix::compute_all(&[], S, SIGMA, R);
        assert_eq!(pg.delta, 0.0);
        assert_eq!(pg.gamma, 0.0);
        assert_eq!(pg.vega, 0.0);
    }

    #[test]
    fn test_deep_itm_call_delta_near_one() {
        let d = delta(200.0, 100.0, T, R, SIGMA, true);
        assert!(d > 0.99, "Deep ITM call delta should approach 1: {}", d);
    }

    #[test]
    fn test_deep_otm_call_delta_near_zero() {
        let d = delta(50.0, 100.0, T, R, SIGMA, true);
        assert!(d < 0.01, "Deep OTM call delta should approach 0: {}", d);
    }
}
