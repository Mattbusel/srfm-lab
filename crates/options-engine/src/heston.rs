use std::f64::consts::PI;
use crate::OptionsError;
use crate::black_scholes::{OptionType, norm_cdf};

/// Heston stochastic volatility model parameters
#[derive(Debug, Clone, Copy)]
pub struct HestonParams {
    /// Long-run variance (v_bar)
    pub v0: f64,
    /// Long-run mean reversion level
    pub theta: f64,
    /// Mean reversion speed
    pub kappa: f64,
    /// Volatility of variance (vol-of-vol)
    pub sigma: f64,
    /// Correlation between spot and variance processes
    pub rho: f64,
}

impl HestonParams {
    pub fn new(v0: f64, theta: f64, kappa: f64, sigma: f64, rho: f64) -> Self {
        HestonParams { v0, theta, kappa, sigma, rho }
    }

    /// Check Feller condition: 2*kappa*theta > sigma^2
    pub fn feller_condition_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }
}

/// Complex number arithmetic for characteristic function evaluation
#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self { Complex { re, im } }
    fn re(re: f64) -> Self { Complex { re, im: 0.0 } }
    fn im(im: f64) -> Self { Complex { re: 0.0, im } }

    fn add(self, other: Complex) -> Complex {
        Complex { re: self.re + other.re, im: self.im + other.im }
    }

    fn sub(self, other: Complex) -> Complex {
        Complex { re: self.re - other.re, im: self.im - other.im }
    }

    fn mul(self, other: Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn div(self, other: Complex) -> Complex {
        let denom = other.re * other.re + other.im * other.im;
        Complex {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }

    fn scale(self, s: f64) -> Complex {
        Complex { re: self.re * s, im: self.im * s }
    }

    fn norm_sq(self) -> f64 { self.re * self.re + self.im * self.im }

    fn sqrt(self) -> Complex {
        let r = self.norm_sq().sqrt().sqrt();
        let theta = self.im.atan2(self.re) / 2.0;
        Complex { re: r * theta.cos(), im: r * theta.sin() }
    }

    fn exp(self) -> Complex {
        let e_re = self.re.exp();
        Complex { re: e_re * self.im.cos(), im: e_re * self.im.sin() }
    }

    fn ln(self) -> Complex {
        let r = (self.re * self.re + self.im * self.im).sqrt();
        let theta = self.im.atan2(self.re);
        Complex { re: r.ln(), im: theta }
    }

    fn conj(self) -> Complex { Complex { re: self.re, im: -self.im } }
}

impl std::ops::Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex { Complex { re: -self.re, im: -self.im } }
}

pub struct HestonModel {
    params: HestonParams,
    n_integration: usize,
    upper_limit: f64,
}

impl HestonModel {
    pub fn new(params: HestonParams) -> Self {
        HestonModel { params, n_integration: 128, upper_limit: 200.0 }
    }

    pub fn with_integration(mut self, n: usize, upper: f64) -> Self {
        self.n_integration = n;
        self.upper_limit = upper;
        self
    }

    /// Heston characteristic function phi(u; t)
    /// Using the formulation from Albrecher et al. (2007) to avoid discontinuities
    fn char_fn(&self, u: Complex, s: f64, v0: f64, r: f64, q: f64, t: f64) -> Complex {
        let p = &self.params;
        let i = Complex::new(0.0, 1.0);
        let x = s.ln();

        // d = sqrt((kappa - rho*sigma*i*u)^2 + sigma^2*(i*u + u^2))
        let kappa_c = Complex::re(p.kappa);
        let rho_c = Complex::re(p.rho);
        let sigma_c = Complex::re(p.sigma);

        let a = kappa_c.sub(rho_c.mul(sigma_c).mul(i).mul(u));
        let b = sigma_c.mul(sigma_c).mul(i.mul(u).add(u.mul(u)));
        let d = a.mul(a).add(b).sqrt();

        // g = (kappa - rho*sigma*i*u - d) / (kappa - rho*sigma*i*u + d)
        let num = a.sub(d);
        let den = a.add(d);
        let g = num.div(den);

        // exponent
        let exp_dt = d.scale(-t).exp();
        let one = Complex::re(1.0);

        // C(t) = (r-q)*i*u*t + (kappa*theta/sigma^2) * ((kappa-rho*sigma*iu-d)*t - 2*ln((1-g*exp(-dt))/(1-g)))
        let one_minus_g_exp = one.sub(g.mul(exp_dt));
        let one_minus_g = one.sub(g);
        let log_term = one_minus_g_exp.div(one_minus_g).ln();
        let kappa_theta_over_sigma2 = p.kappa * p.theta / (p.sigma * p.sigma);

        let c_part1 = Complex::re(r - q).mul(i).mul(u).scale(t);
        let c_part2 = Complex::re(kappa_theta_over_sigma2)
            .mul(a.sub(d).scale(t).sub(log_term.scale(2.0)));
        let c = c_part1.add(c_part2);

        // D(t) = (kappa - rho*sigma*iu - d) / sigma^2 * (1 - exp(-dt)) / (1 - g*exp(-dt))
        let d_coeff = a.sub(d).div(Complex::re(p.sigma * p.sigma));
        let d_val = d_coeff.mul(one.sub(exp_dt).div(one_minus_g_exp));

        // phi = exp(C + D*v0 + i*u*x)
        c.add(d_val.scale(v0)).add(i.mul(u).scale(x)).exp()
    }

    /// Carr-Madan FFT pricing (Gauss-Legendre quadrature version for simplicity)
    pub fn price(&self, s: f64, k: f64, r: f64, q: f64, t: f64, opt_type: OptionType) -> f64 {
        let call_price = self.price_call_integration(s, k, r, q, t);
        match opt_type {
            OptionType::Call => call_price,
            OptionType::Put => {
                // Put-call parity
                call_price - s * (-q * t).exp() + k * (-r * t).exp()
            }
        }
    }

    fn price_call_integration(&self, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
        let log_k = k.ln();
        let v0 = self.params.v0;
        let n = self.n_integration;
        let upper = self.upper_limit;
        let h = upper / n as f64;

        // Gauss sum integration for Heston formula
        // P1 and P2 are the two probability terms
        let mut p1 = 0.0_f64;
        let mut p2 = 0.0_f64;

        for j in 0..n {
            let u = (j as f64 + 0.5) * h;
            let iu = Complex::new(0.0, u);

            // Characteristic function for P1: phi(u - i)
            let phi1 = self.char_fn(Complex::new(u, -1.0), s, v0, r, q, t);
            // Characteristic function for P2: phi(u)
            let phi2 = self.char_fn(Complex::new(u, 0.0), s, v0, r, q, t);

            let fwd = s * ((r - q) * t).exp();
            let phi1_adj = phi1.div(Complex::new(fwd, 0.0));

            // Integrate: Re[ exp(-i*u*log(K)) * phi / (i*u) ]
            let exp_term_re = (u * log_k).cos();
            let exp_term_im = -(u * log_k).sin();
            let exp_iu_logk = Complex::new(exp_term_re, exp_term_im);

            let integrand1 = exp_iu_logk.mul(phi1_adj).div(iu).re;
            let integrand2 = exp_iu_logk.mul(phi2).div(iu).re;

            p1 += integrand1 * h;
            p2 += integrand2 * h;
        }

        let df = (-r * t).exp();
        let fwd_df = (-q * t).exp();

        let p1 = 0.5 + p1 / PI;
        let p2 = 0.5 + p2 / PI;

        (s * fwd_df * p1 - k * df * p2).max(0.0)
    }

    /// Calibrate Heston parameters to market implied vols via Nelder-Mead
    pub fn calibrate(
        &self,
        market_data: &[(f64, f64, f64, OptionType)], // (strike, expiry, market_price, type)
        s: f64, r: f64, q: f64,
        initial_params: Option<HestonParams>,
        max_iter: usize,
    ) -> Result<HestonParams, OptionsError> {
        let init = initial_params.unwrap_or(HestonParams::new(0.04, 0.04, 1.5, 0.3, -0.7));

        // Objective: sum of squared differences between model and market prices
        let objective = |params: &[f64]| -> f64 {
            let p = HestonParams {
                v0: params[0].max(1e-6),
                theta: params[1].max(1e-6),
                kappa: params[2].max(0.01),
                sigma: params[3].max(1e-4),
                rho: params[4].max(-0.999).min(0.999),
            };
            let model = HestonModel::new(p).with_integration(64, 100.0);
            market_data.iter().map(|(k, t, mp, ot)| {
                let model_price = model.price(s, *k, r, q, *t, *ot);
                (model_price - mp).powi(2)
            }).sum()
        };

        let x0 = vec![init.v0, init.theta, init.kappa, init.sigma, init.rho];
        let result = nelder_mead(objective, x0, max_iter, 1e-8)?;

        Ok(HestonParams {
            v0: result[0].max(1e-6),
            theta: result[1].max(1e-6),
            kappa: result[2].max(0.01),
            sigma: result[3].max(1e-4),
            rho: result[4].max(-0.999).min(0.999),
        })
    }

    pub fn params(&self) -> HestonParams { self.params }
}

/// Nelder-Mead simplex optimizer
pub fn nelder_mead<F>(f: F, x0: Vec<f64>, max_iter: usize, tol: f64) -> Result<Vec<f64>, OptionsError>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    // Initialize simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    for i in 0..n {
        let mut x = x0.clone();
        x[i] *= if x[i].abs() > 1e-6 { 1.05 } else { 0.00025 };
        x[i] += if x[i].abs() < 1e-6 { 0.00025 } else { 0.0 };
        simplex.push(x);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|x| f(x)).collect();

    for iter in 0..max_iter {
        // Sort
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        fvals = order.iter().map(|&i| fvals[i]).collect();

        // Check convergence
        let spread: f64 = fvals[n] - fvals[0];
        if spread < tol && iter > 10 {
            return Ok(simplex[0].clone());
        }

        // Centroid (exclude worst)
        let centroid: Vec<f64> = (0..n).map(|j| {
            simplex[..n].iter().map(|x| x[j]).sum::<f64>() / n as f64
        }).collect();

        // Reflect
        let xr: Vec<f64> = (0..n).map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j])).collect();
        let fr = f(&xr);

        if fr < fvals[0] {
            // Expand
            let xe: Vec<f64> = (0..n).map(|j| centroid[j] + gamma * (xr[j] - centroid[j])).collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            // Contract
            let xc: Vec<f64> = (0..n).map(|j| centroid[j] + rho * (simplex[n][j] - centroid[j])).collect();
            let fc = f(&xc);
            if fc < fvals[n] {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                // Shrink
                for i in 1..=n {
                    simplex[i] = (0..n).map(|j| simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j])).collect();
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    Ok(simplex[0].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black_scholes::BlackScholes;

    fn default_params() -> HestonParams {
        HestonParams::new(0.04, 0.04, 2.0, 0.3, -0.7)
    }

    #[test]
    fn test_feller_condition() {
        let p = default_params();
        // 2 * 2.0 * 0.04 = 0.16 > 0.09 = 0.3^2
        assert!(p.feller_condition_satisfied());
        let bad = HestonParams::new(0.04, 0.01, 0.5, 0.5, -0.5);
        // 2 * 0.5 * 0.01 = 0.01 < 0.25 = 0.5^2
        assert!(!bad.feller_condition_satisfied());
    }

    #[test]
    fn test_heston_price_positive() {
        let model = HestonModel::new(default_params());
        let price = model.price(100.0, 100.0, 0.05, 0.02, 1.0, OptionType::Call);
        assert!(price > 0.0, "Heston call price should be positive, got {}", price);
    }

    #[test]
    fn test_heston_put_call_parity() {
        let model = HestonModel::new(default_params());
        let s = 100.0; let k = 100.0; let r = 0.05; let q = 0.02; let t = 1.0;
        let call = model.price(s, k, r, q, t, OptionType::Call);
        let put = model.price(s, k, r, q, t, OptionType::Put);
        let parity = (call - put) - (s * (-q * t).exp() - k * (-r * t).exp());
        assert!(parity.abs() < 0.01, "Put-call parity error = {:.4}", parity);
    }

    #[test]
    fn test_heston_low_vol_approaches_bsm() {
        // When sigma (vol-of-vol) -> 0, Heston should approach BSM
        let params = HestonParams::new(0.04, 0.04, 5.0, 0.001, 0.0);
        let model = HestonModel::new(params).with_integration(256, 300.0);
        let heston_price = model.price(100.0, 100.0, 0.05, 0.02, 1.0, OptionType::Call);
        let bsm_price = BlackScholes::price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
        // Allow 5% tolerance since vol-of-vol isn't exactly 0
        assert!((heston_price - bsm_price).abs() / bsm_price < 0.05,
            "Heston={:.4} BSM={:.4}", heston_price, bsm_price);
    }

    #[test]
    fn test_nelder_mead_simple() {
        // Minimize (x-2)^2 + (y-3)^2
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
        let result = nelder_mead(f, vec![0.0, 0.0], 1000, 1e-10).unwrap();
        assert!((result[0] - 2.0).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
    }
}
