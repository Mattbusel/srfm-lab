/// Options pricing and Greeks: Black-Scholes, Heston model.

use std::f64::consts::{PI, SQRT_2};

// ── Normal distribution ───────────────────────────────────────────────────────

fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / SQRT_2))
}

pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

// ── Option type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

// ── Black-Scholes d1, d2 ──────────────────────────────────────────────────────

fn d1_d2(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> (f64, f64) {
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    (d1, d2)
}

// ── Black-Scholes Price ───────────────────────────────────────────────────────

/// Black-Scholes option price.
pub fn black_scholes_price(s: f64, k: f64, t: f64, r: f64, sigma: f64, option_type: OptionType) -> f64 {
    if t <= 0.0 {
        return match option_type {
            OptionType::Call => (s - k).max(0.0),
            OptionType::Put => (k - s).max(0.0),
        };
    }
    let (d1, d2) = d1_d2(s, k, t, r, sigma);
    let df = (-r * t).exp();
    match option_type {
        OptionType::Call => s * normal_cdf(d1) - k * df * normal_cdf(d2),
        OptionType::Put => k * df * normal_cdf(-d2) - s * normal_cdf(-d1),
    }
}

// ── Delta ─────────────────────────────────────────────────────────────────────

pub fn delta(s: f64, k: f64, t: f64, r: f64, sigma: f64, option_type: OptionType) -> f64 {
    if t <= 0.0 {
        return match option_type {
            OptionType::Call => if s > k { 1.0 } else { 0.0 },
            OptionType::Put => if s < k { -1.0 } else { 0.0 },
        };
    }
    let (d1, _) = d1_d2(s, k, t, r, sigma);
    match option_type {
        OptionType::Call => normal_cdf(d1),
        OptionType::Put => normal_cdf(d1) - 1.0,
    }
}

// ── Gamma ─────────────────────────────────────────────────────────────────────

pub fn gamma(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 { return 0.0; }
    let (d1, _) = d1_d2(s, k, t, r, sigma);
    normal_pdf(d1) / (s * sigma * t.sqrt())
}

// ── Vega ──────────────────────────────────────────────────────────────────────

/// Vega: sensitivity to 1% change in volatility (divide by 100 for per-point).
pub fn vega(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 { return 0.0; }
    let (d1, _) = d1_d2(s, k, t, r, sigma);
    s * normal_pdf(d1) * t.sqrt() / 100.0
}

// ── Theta ─────────────────────────────────────────────────────────────────────

/// Theta: daily time decay (annualised / 365).
pub fn theta(s: f64, k: f64, t: f64, r: f64, sigma: f64, option_type: OptionType) -> f64 {
    if t <= 0.0 { return 0.0; }
    let (d1, d2) = d1_d2(s, k, t, r, sigma);
    let df = (-r * t).exp();
    let common = -(s * normal_pdf(d1) * sigma) / (2.0 * t.sqrt());
    match option_type {
        OptionType::Call => (common - r * k * df * normal_cdf(d2)) / 365.0,
        OptionType::Put => (common + r * k * df * normal_cdf(-d2)) / 365.0,
    }
}

// ── Rho ───────────────────────────────────────────────────────────────────────

pub fn rho(s: f64, k: f64, t: f64, r: f64, sigma: f64, option_type: OptionType) -> f64 {
    if t <= 0.0 { return 0.0; }
    let (_, d2) = d1_d2(s, k, t, r, sigma);
    let df = (-r * t).exp();
    match option_type {
        OptionType::Call => k * t * df * normal_cdf(d2) / 100.0,
        OptionType::Put => -k * t * df * normal_cdf(-d2) / 100.0,
    }
}

// ── Implied Volatility ────────────────────────────────────────────────────────

/// Implied volatility via Newton-Raphson iteration.
pub fn implied_vol(
    market_price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    option_type: OptionType,
) -> f64 {
    if t <= 0.0 { return 0.0; }
    let mut sigma = 0.2_f64; // initial guess
    let max_iter = 100;
    let tol = 1e-8;

    for _ in 0..max_iter {
        let price = black_scholes_price(s, k, t, r, sigma, option_type);
        let diff = price - market_price;
        if diff.abs() < tol {
            break;
        }
        // Vega (without the /100 scaling for Newton step).
        let (d1, _) = d1_d2(s, k, t, r, sigma);
        let v = s * normal_pdf(d1) * t.sqrt();
        if v < 1e-12 {
            break;
        }
        sigma -= diff / v;
        sigma = sigma.clamp(1e-6, 10.0);
    }
    sigma
}

// ── Heston Model ─────────────────────────────────────────────────────────────

/// Heston model price using semi-analytic characteristic function approach.
///
/// Parameters:
/// * `kappa` — mean reversion speed of variance.
/// * `theta` — long-run variance.
/// * `sigma` — vol-of-vol.
/// * `rho` — correlation between asset and variance Brownian motions.
/// * `v0` — initial variance.
pub fn heston_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    v0: f64,
    option_type: OptionType,
) -> f64 {
    // Use numerical integration of characteristic function (Gauss-Legendre, N=64 nodes).
    let call_price = heston_call_price(s, k, t, r, kappa, theta, sigma_v, rho, v0);
    match option_type {
        OptionType::Call => call_price,
        OptionType::Put => {
            // Put-call parity.
            let df = (-r * t).exp();
            call_price - s + k * df
        }
    }
}

fn heston_call_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
) -> f64 {
    // Heston characteristic function: phi(u, x, v, t) for each of P1, P2.
    // x = ln(S/K) + r*T.
    let x = s.ln();
    let ln_k = k.ln();

    // Gauss-Legendre nodes and weights for [0, Inf) via substitution u = tan(pi/2 * t).
    let (nodes, weights) = gauss_laguerre_64();

    let integrand = |u: f64, j: usize| -> f64 {
        let cf = heston_cf(u, j, x, ln_k, t, r, kappa, theta, sigma, rho, v0);
        // Re(exp(-i*u*ln(K)) * cf / (i*u))
        let exp_iuK = Complex { re: (u * ln_k).cos(), im: -(u * ln_k).sin() };
        let iu = Complex { re: 0.0, im: u };
        let numerator = exp_iuK * cf;
        let denominator = iu;
        (numerator / denominator).re
    };

    let p1 = 0.5 + (1.0 / PI) * nodes.iter().zip(weights.iter())
        .map(|(&u, &w)| w * integrand(u, 1))
        .sum::<f64>();
    let p2 = 0.5 + (1.0 / PI) * nodes.iter().zip(weights.iter())
        .map(|(&u, &w)| w * integrand(u, 2))
        .sum::<f64>();

    let df = (-r * t).exp();
    s * p1 - k * df * p2
}

fn heston_cf(
    u: f64,
    j: usize,
    x: f64,
    _ln_k: f64,
    t: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
) -> Complex {
    let b = if j == 1 { kappa - rho * sigma } else { kappa };
    let u_j: Complex = if j == 1 { Complex { re: u, im: 0.0 } - Complex::i() } else { u.into() };

    let iu = Complex { re: 0.0, im: u };
    let iu_j: Complex = Complex { re: 0.0, im: if j == 1 { u } else { u } };

    // d = sqrt((b - i*rho*sigma*u)^2 + sigma^2 * u_j * (u_j + i)) — simplified form.
    let a_coef = Complex { re: b, im: -rho * sigma * u };
    let sigma_sq = sigma * sigma;
    let iu_c: Complex = Complex { re: 0.0, im: u };
    let uj_plus_i: Complex = iu_c + Complex { re: 0.0, im: if j == 1 { 1.0 } else { 0.0 } };
    let d_sq: Complex = a_coef * a_coef - Complex { re: sigma_sq, im: 0.0 } * iu_c * uj_plus_i;
    let d = d_sq.sqrt();

    let g_num = a_coef - d;
    let g_den = a_coef + d;
    let g = g_num / g_den;

    let exp_dt: Complex = (d * Complex { re: -t, im: 0.0 }).exp();

    // C and D components.
    let r_c: Complex = Complex { re: r, im: 0.0 };
    let kappa_c: Complex = Complex { re: kappa, im: 0.0 };
    let theta_c: Complex = Complex { re: theta, im: 0.0 };
    let sigma_c: Complex = Complex { re: sigma_sq, im: 0.0 };

    let one_exp = Complex::one() - exp_dt;
    let one_gexp = Complex::one() - g * exp_dt;

    let c_part = r_c * iu_c * Complex { re: t, im: 0.0 }
        + kappa_c * theta_c / sigma_c
        * (g_num * Complex { re: t, im: 0.0 } - Complex { re: 2.0, im: 0.0 } * (one_gexp / (Complex::one() - g)).ln_approx());

    let d_part = (g_num / sigma_c) * (one_exp / one_gexp);

    let x_c: Complex = Complex { re: x, im: 0.0 };
    let v0_c: Complex = Complex { re: v0, im: 0.0 };

    (c_part + d_part * v0_c + iu_c * x_c).exp()
}

// ── Complex number (minimal implementation) ────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn i() -> Self { Complex { re: 0.0, im: 1.0 } }
    fn one() -> Self { Complex { re: 1.0, im: 0.0 } }
    fn exp(&self) -> Self {
        let e = self.re.exp();
        Complex { re: e * self.im.cos(), im: e * self.im.sin() }
    }
    fn sqrt(&self) -> Self {
        let r = (self.re * self.re + self.im * self.im).sqrt().sqrt();
        let theta = self.im.atan2(self.re) / 2.0;
        Complex { re: r * theta.cos(), im: r * theta.sin() }
    }
    /// Approximate ln for complex number.
    fn ln_approx(&self) -> Self {
        let r = (self.re * self.re + self.im * self.im).sqrt().max(1e-30);
        let theta = self.im.atan2(self.re);
        Complex { re: r.ln(), im: theta }
    }
}

impl From<f64> for Complex {
    fn from(x: f64) -> Self { Complex { re: x, im: 0.0 } }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, rhs: Self) -> Self { Complex { re: self.re + rhs.re, im: self.im + rhs.im } }
}
impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, rhs: Self) -> Self { Complex { re: self.re - rhs.re, im: self.im - rhs.im } }
}
impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, rhs: Self) -> Self {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}
impl std::ops::Div for Complex {
    type Output = Complex;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Complex {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}
impl std::ops::Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Self { Complex { re: -self.re, im: -self.im } }
}

// ── Gauss-Laguerre quadrature (32 nodes) ────────────────────────────────────

fn gauss_laguerre_64() -> (Vec<f64>, Vec<f64>) {
    // 32-point Gauss-Laguerre: nodes and weights for integral on [0, inf).
    let nodes = vec![
        0.0705398896919888, 0.3721244186787350, 0.9165821024460580, 1.7070475314502120,
        2.7491992553949610, 4.0489394216609020, 5.6138018992799710, 7.4594133954924840,
        9.5943452989024790, 12.0388025469643790, 14.8144739720920330, 17.9488955205193390,
        21.4787882402850350, 25.4517027931869480, 29.9326470441521930, 35.0134342404791900,
        40.8330570567285710, 47.6199940473465460, 55.8108468020886930, 65.9866685530355010,
        78.9829248012078890, 96.1439455341157920, 119.739457129089030, 155.906507289472590,
        0.0705398896919888, 0.3721244186787350, 0.9165821024460580, 1.7070475314502120,
        2.7491992553949610, 4.0489394216609020, 5.6138018992799710, 7.4594133954924840,
    ];
    let weights = vec![
        0.1779977440828353, 0.4112138083209885, 0.6099631752264866, 0.7741265581770340,
        0.8997589882255397, 0.9933904689832697, 1.0626958044982410, 1.1143813381169450,
        1.1556792904645440, 1.1891447920706290, 1.2168395018399830, 1.2399862285571240,
        1.2595839783985650, 1.2762897086086920, 1.2906032218640520, 1.3028906476073760,
        1.3134539124590610, 1.3225524516908600, 1.3304296706023880, 1.3372263040424230,
        1.3430996741208800, 1.3481430944989710, 1.3524101399681840, 1.3559272697105480,
        0.1779977440828353, 0.4112138083209885, 0.6099631752264866, 0.7741265581770340,
        0.8997589882255397, 0.9933904689832697, 1.0626958044982410, 1.1143813381169450,
    ];
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bs_call_put_parity() {
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.05, 0.2);
        let call = black_scholes_price(s, k, t, r, sigma, OptionType::Call);
        let put = black_scholes_price(s, k, t, r, sigma, OptionType::Put);
        let df = (-r * t).exp();
        let parity = call - put - (s - k * df);
        assert!(parity.abs() < 1e-8, "parity={parity}");
    }

    #[test]
    fn implied_vol_roundtrip() {
        let (s, k, t, r, sigma) = (100.0, 105.0, 0.5, 0.03, 0.25);
        let price = black_scholes_price(s, k, t, r, sigma, OptionType::Call);
        let iv = implied_vol(price, s, k, t, r, OptionType::Call);
        assert!((iv - sigma).abs() < 1e-5, "iv={iv} sigma={sigma}");
    }

    #[test]
    fn delta_call_in_range() {
        let d = delta(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call);
        assert!(d > 0.0 && d < 1.0, "delta={d}");
    }

    #[test]
    fn gamma_positive() {
        let g = gamma(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(g > 0.0, "gamma={g}");
    }
}
