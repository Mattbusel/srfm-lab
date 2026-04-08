// distributions.rs — Probability distributions, RNG, sampling

use std::f64::consts::{PI, E, SQRT_2};

// ============================================================
// Random Number Generation: Xoshiro256++
// ============================================================

/// Xoshiro256++ PRNG
#[derive(Clone, Debug)]
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to initialize state
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1)
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in [a, b)
    #[inline]
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next_f64()
    }

    /// Standard normal via Box-Muller
    pub fn normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 1e-30 {
                return (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            }
        }
    }

    /// Normal with given mean and std
    pub fn normal_params(&mut self, mean: f64, std: f64) -> f64 {
        mean + std * self.normal()
    }

    /// Pair of standard normals (Box-Muller)
    pub fn normal_pair(&mut self) -> (f64, f64) {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 1e-30 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * PI * u2;
                return (r * theta.cos(), r * theta.sin());
            }
        }
    }

    /// Exponential distribution
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        loop {
            let u = self.next_f64();
            if u > 1e-30 { return -u.ln() / lambda; }
        }
    }

    /// Gamma distribution (Marsaglia and Tsang)
    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        if shape < 1.0 {
            let u = self.next_f64();
            return self.gamma(shape + 1.0, scale) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 { continue; }
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * x * x * x * x { return d * v * scale; }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) { return d * v * scale; }
        }
    }

    /// Beta distribution
    pub fn beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let x = self.gamma(alpha, 1.0);
        let y = self.gamma(beta, 1.0);
        x / (x + y)
    }

    /// Chi-squared
    pub fn chi_squared(&mut self, df: f64) -> f64 {
        self.gamma(df / 2.0, 2.0)
    }

    /// Student's t
    pub fn student_t(&mut self, df: f64) -> f64 {
        let n = self.normal();
        let c = self.chi_squared(df);
        n / (c / df).sqrt()
    }

    /// F distribution
    pub fn f_dist(&mut self, d1: f64, d2: f64) -> f64 {
        let x1 = self.chi_squared(d1) / d1;
        let x2 = self.chi_squared(d2) / d2;
        x1 / x2
    }

    /// Poisson (Knuth for small lambda, rejection for large)
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda < 30.0 {
            let l = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.next_f64();
                if p <= l { return k - 1; }
            }
        } else {
            // Normal approximation with correction
            let n = self.normal() * lambda.sqrt() + lambda;
            n.round().max(0.0) as u64
        }
    }

    /// Bernoulli
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Binomial
    pub fn binomial(&mut self, n: u64, p: f64) -> u64 {
        if n < 20 {
            let mut count = 0;
            for _ in 0..n { if self.bernoulli(p) { count += 1; } }
            count
        } else {
            // Normal approximation
            let mean = n as f64 * p;
            let std = (n as f64 * p * (1.0 - p)).sqrt();
            (self.normal() * std + mean).round().clamp(0.0, n as f64) as u64
        }
    }

    /// Geometric distribution (number of trials until first success)
    pub fn geometric(&mut self, p: f64) -> u64 {
        ((1.0 - self.next_f64()).ln() / (1.0 - p).ln()).ceil() as u64
    }

    /// Lognormal
    pub fn lognormal(&mut self, mu: f64, sigma: f64) -> f64 {
        (mu + sigma * self.normal()).exp()
    }

    /// Multivariate normal: mean + L * z, where L is Cholesky of covariance
    pub fn multivariate_normal(&mut self, mean: &[f64], chol_l: &[Vec<f64>]) -> Vec<f64> {
        let n = mean.len();
        let z: Vec<f64> = (0..n).map(|_| self.normal()).collect();
        let mut result = vec![0.0; n];
        for i in 0..n {
            let mut s = mean[i];
            for j in 0..=i { s += chol_l[i][j] * z[j]; }
            result[i] = s;
        }
        result
    }

    /// Dirichlet distribution
    pub fn dirichlet(&mut self, alpha: &[f64]) -> Vec<f64> {
        let mut samples: Vec<f64> = alpha.iter().map(|&a| self.gamma(a, 1.0)).collect();
        let sum: f64 = samples.iter().sum();
        for s in &mut samples { *s /= sum; }
        samples
    }

    /// Discrete distribution from weights (unnormalized)
    pub fn discrete(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut u = self.next_f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            u -= w;
            if u <= 0.0 { return i; }
        }
        weights.len() - 1
    }

    /// Shuffle a slice in-place (Fisher-Yates)
    pub fn shuffle<T>(&mut self, arr: &mut [T]) {
        for i in (1..arr.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            arr.swap(i, j);
        }
    }

    /// Sample k items without replacement
    pub fn sample_without_replacement(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        let k = k.min(n);
        for i in 0..k {
            let j = i + (self.next_u64() as usize) % (n - i);
            pool.swap(i, j);
        }
        pool[..k].to_vec()
    }
}

// ============================================================
// Normal Distribution
// ============================================================

/// Standard normal PDF
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF (Abramowitz & Stegun, max error 1.5e-7)
pub fn norm_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let z = x.abs() / SQRT_2;
    let t = 1.0 / (1.0 + p * z);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-z * z).exp();
    0.5 * (1.0 + sign * y)
}

/// Normal CDF — higher precision rational approximation (Hart)
pub fn norm_cdf_precise(x: f64) -> f64 {
    if x < -37.0 { return 0.0; }
    if x > 37.0 { return 1.0; }
    let cdf_pos = |z: f64| -> f64 {
        let b0 = 0.2316419;
        let b1 = 0.319381530;
        let b2 = -0.356563782;
        let b3 = 1.781477937;
        let b4 = -1.821255978;
        let b5 = 1.330274429;
        let t = 1.0 / (1.0 + b0 * z);
        let t2 = t * t; let t3 = t2 * t; let t4 = t3 * t; let t5 = t4 * t;
        1.0 - norm_pdf(z) * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5)
    };
    if x >= 0.0 { cdf_pos(x) } else { 1.0 - cdf_pos(-x) }
}

/// Normal quantile function (Beasley-Springer-Moro)
pub fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    let a = [
        -3.969683028665376e1, 2.209460984245205e2,
        -2.759285104469687e2, 1.383577518672690e2,
        -3.066479806614716e1, 2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1, 1.615858368580409e2,
        -1.556989798598866e2, 6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3, -3.223964580411365e-1,
        -2.400758277161838e0, -2.549732539343734e0,
        4.374664141464968e0, 2.938163982698783e0,
    ];
    let d = [
        7.784695709041462e-3, 3.224671290700398e-1,
        2.445134137142996e0, 3.754408661907416e0,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    }
}

/// Normal PDF with mean and std
pub fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    norm_pdf(z) / sigma
}

/// Normal CDF with mean and std
pub fn normal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    norm_cdf((x - mu) / sigma)
}

/// Normal quantile with mean and std
pub fn normal_ppf(p: f64, mu: f64, sigma: f64) -> f64 {
    mu + sigma * norm_ppf(p)
}

// ============================================================
// Student's t Distribution
// ============================================================

/// Student's t PDF
pub fn student_t_pdf(x: f64, df: f64) -> f64 {
    let coeff = lgamma(0.5 * (df + 1.0)) - lgamma(0.5 * df) - 0.5 * (df * PI).ln();
    (coeff - 0.5 * (df + 1.0) * (1.0 + x * x / df).ln()).exp()
}

/// Student's t CDF via regularized incomplete beta
pub fn student_t_cdf(x: f64, df: f64) -> f64 {
    let t = df / (df + x * x);
    let ib = regularized_incomplete_beta(0.5 * df, 0.5, t);
    if x >= 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
}

/// Student's t quantile via Newton's method
pub fn student_t_ppf(p: f64, df: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    // Initial guess from normal
    let mut x = norm_ppf(p);
    for _ in 0..50 {
        let cdf = student_t_cdf(x, df);
        let pdf = student_t_pdf(x, df);
        if pdf.abs() < 1e-30 { break; }
        let dx = (cdf - p) / pdf;
        x -= dx;
        if dx.abs() < 1e-12 { break; }
    }
    x
}

// ============================================================
// Chi-Squared Distribution
// ============================================================

/// Chi-squared PDF
pub fn chi2_pdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let half_k = k / 2.0;
    ((half_k - 1.0) * x.ln() - x / 2.0 - half_k * 2.0_f64.ln() - lgamma(half_k)).exp()
}

/// Chi-squared CDF via regularized lower incomplete gamma
pub fn chi2_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    regularized_gamma_p(k / 2.0, x / 2.0)
}

/// Chi-squared quantile via Newton's method
pub fn chi2_ppf(p: f64, k: f64) -> f64 {
    if p <= 0.0 { return 0.0; }
    if p >= 1.0 { return f64::INFINITY; }
    // Wilson-Hilferty initial approximation
    let mut x = k * (1.0 - 2.0 / (9.0 * k) + norm_ppf(p) * (2.0 / (9.0 * k)).sqrt()).powi(3);
    x = x.max(0.001);
    for _ in 0..100 {
        let cdf = chi2_cdf(x, k);
        let pdf = chi2_pdf(x, k);
        if pdf.abs() < 1e-30 { break; }
        let dx = (cdf - p) / pdf;
        x -= dx;
        x = x.max(1e-10);
        if dx.abs() < 1e-10 { break; }
    }
    x
}

// ============================================================
// F Distribution
// ============================================================

/// F distribution PDF
pub fn f_pdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let half_d1 = d1 / 2.0;
    let half_d2 = d2 / 2.0;
    let num = (d1 * x).powf(half_d1) * d2.powf(half_d2);
    let den = (d1 * x + d2).powf(half_d1 + half_d2);
    let beta_val = (lgamma(half_d1) + lgamma(half_d2) - lgamma(half_d1 + half_d2)).exp();
    num / (den * x * beta_val)
}

/// F distribution CDF
pub fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let z = d1 * x / (d1 * x + d2);
    regularized_incomplete_beta(d1 / 2.0, d2 / 2.0, z)
}

/// F distribution quantile
pub fn f_ppf(p: f64, d1: f64, d2: f64) -> f64 {
    if p <= 0.0 { return 0.0; }
    if p >= 1.0 { return f64::INFINITY; }
    let mut x = 1.0; // initial guess
    for _ in 0..100 {
        let cdf = f_cdf(x, d1, d2);
        let pdf = f_pdf(x, d1, d2);
        if pdf.abs() < 1e-30 { break; }
        let dx = (cdf - p) / pdf;
        x -= dx;
        x = x.max(1e-10);
        if dx.abs() < 1e-10 { break; }
    }
    x
}

// ============================================================
// Beta Distribution
// ============================================================

/// Beta PDF
pub fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 { return 0.0; }
    let log_beta = lgamma(a) + lgamma(b) - lgamma(a + b);
    ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - log_beta).exp()
}

/// Beta CDF
pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    regularized_incomplete_beta(a, b, x)
}

/// Beta quantile via Newton's method
pub fn beta_ppf(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 { return 0.0; }
    if p >= 1.0 { return 1.0; }
    let mut x = p; // initial guess
    // Better initial guess
    if a > 1.0 && b > 1.0 {
        x = (a - 1.0) / (a + b - 2.0);
    }
    for _ in 0..100 {
        let cdf = beta_cdf(x, a, b);
        let pdf = beta_pdf(x, a, b);
        if pdf.abs() < 1e-30 { break; }
        let dx = (cdf - p) / pdf;
        x -= dx;
        x = x.clamp(1e-10, 1.0 - 1e-10);
        if dx.abs() < 1e-12 { break; }
    }
    x
}

// ============================================================
// Gamma Distribution
// ============================================================

/// Gamma PDF
pub fn gamma_pdf(x: f64, shape: f64, scale: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let rate = 1.0 / scale;
    ((shape - 1.0) * x.ln() - rate * x + shape * rate.ln() - lgamma(shape)).exp()
}

/// Gamma CDF
pub fn gamma_cdf(x: f64, shape: f64, scale: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    regularized_gamma_p(shape, x / scale)
}

/// Gamma quantile
pub fn gamma_ppf(p: f64, shape: f64, scale: f64) -> f64 {
    if p <= 0.0 { return 0.0; }
    if p >= 1.0 { return f64::INFINITY; }
    let mut x = shape * scale; // initial guess (mean)
    if shape >= 1.0 {
        let z = norm_ppf(p);
        x = shape * scale * (1.0 + z / (9.0 * shape).sqrt()).powi(3).max(0.01);
    }
    for _ in 0..100 {
        let cdf = gamma_cdf(x, shape, scale);
        let pdf = gamma_pdf(x, shape, scale);
        if pdf.abs() < 1e-30 { break; }
        let dx = (cdf - p) / pdf;
        x -= dx;
        x = x.max(1e-10);
        if dx.abs() < 1e-10 { break; }
    }
    x
}

// ============================================================
// Poisson Distribution
// ============================================================

/// Poisson PMF
pub fn poisson_pmf(k: u64, lambda: f64) -> f64 {
    (k as f64 * lambda.ln() - lambda - lgamma(k as f64 + 1.0)).exp()
}

/// Poisson CDF
pub fn poisson_cdf(k: u64, lambda: f64) -> f64 {
    1.0 - regularized_gamma_p(k as f64 + 1.0, lambda)
}

// ============================================================
// Exponential Distribution
// ============================================================

/// Exponential PDF
pub fn exponential_pdf(x: f64, lambda: f64) -> f64 {
    if x < 0.0 { 0.0 } else { lambda * (-lambda * x).exp() }
}

/// Exponential CDF
pub fn exponential_cdf(x: f64, lambda: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 - (-lambda * x).exp() }
}

/// Exponential quantile
pub fn exponential_ppf(p: f64, lambda: f64) -> f64 {
    -(1.0 - p).ln() / lambda
}

// ============================================================
// Lognormal Distribution
// ============================================================

pub fn lognormal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let z = (x.ln() - mu) / sigma;
    (-0.5 * z * z).exp() / (x * sigma * (2.0 * PI).sqrt())
}

pub fn lognormal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    norm_cdf((x.ln() - mu) / sigma)
}

pub fn lognormal_ppf(p: f64, mu: f64, sigma: f64) -> f64 {
    (mu + sigma * norm_ppf(p)).exp()
}

// ============================================================
// Uniform Distribution
// ============================================================

pub fn uniform_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x >= a && x <= b { 1.0 / (b - a) } else { 0.0 }
}

pub fn uniform_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x < a { 0.0 } else if x > b { 1.0 } else { (x - a) / (b - a) }
}

pub fn uniform_ppf(p: f64, a: f64, b: f64) -> f64 {
    a + p * (b - a)
}

// ============================================================
// Bernoulli Distribution
// ============================================================

pub fn bernoulli_pmf(k: u64, p: f64) -> f64 {
    if k == 0 { 1.0 - p } else if k == 1 { p } else { 0.0 }
}

// ============================================================
// Weibull Distribution
// ============================================================

pub fn weibull_pdf(x: f64, shape: f64, scale: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    let z = x / scale;
    (shape / scale) * z.powf(shape - 1.0) * (-z.powf(shape)).exp()
}

pub fn weibull_cdf(x: f64, shape: f64, scale: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    1.0 - (-(x / scale).powf(shape)).exp()
}

pub fn weibull_ppf(p: f64, shape: f64, scale: f64) -> f64 {
    scale * (-(1.0 - p).ln()).powf(1.0 / shape)
}

// ============================================================
// Pareto Distribution
// ============================================================

pub fn pareto_pdf(x: f64, alpha: f64, x_m: f64) -> f64 {
    if x < x_m { 0.0 } else { alpha * x_m.powf(alpha) / x.powf(alpha + 1.0) }
}

pub fn pareto_cdf(x: f64, alpha: f64, x_m: f64) -> f64 {
    if x < x_m { 0.0 } else { 1.0 - (x_m / x).powf(alpha) }
}

pub fn pareto_ppf(p: f64, alpha: f64, x_m: f64) -> f64 {
    x_m / (1.0 - p).powf(1.0 / alpha)
}

// ============================================================
// Generalized Pareto Distribution (GPD)
// ============================================================

pub fn gpd_pdf(x: f64, xi: f64, sigma: f64) -> f64 {
    if x < 0.0 || sigma <= 0.0 { return 0.0; }
    if xi.abs() < 1e-10 {
        (-x / sigma).exp() / sigma
    } else {
        let z = x / sigma;
        if 1.0 + xi * z <= 0.0 { return 0.0; }
        (1.0 / sigma) * (1.0 + xi * z).powf(-1.0 / xi - 1.0)
    }
}

pub fn gpd_cdf(x: f64, xi: f64, sigma: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if xi.abs() < 1e-10 {
        1.0 - (-x / sigma).exp()
    } else {
        let z = x / sigma;
        if 1.0 + xi * z <= 0.0 { return 1.0; }
        1.0 - (1.0 + xi * z).powf(-1.0 / xi)
    }
}

pub fn gpd_ppf(p: f64, xi: f64, sigma: f64) -> f64 {
    if xi.abs() < 1e-10 {
        -sigma * (1.0 - p).ln()
    } else {
        sigma / xi * ((1.0 - p).powf(-xi) - 1.0)
    }
}

// ============================================================
// Inverse CDF (general) sampling using bisection
// ============================================================

/// Generic inverse CDF via bisection
pub fn inverse_cdf_bisection<F: Fn(f64) -> f64>(
    cdf: F, p: f64, lo: f64, hi: f64, tol: f64, max_iter: usize,
) -> f64 {
    let mut a = lo;
    let mut b = hi;
    for _ in 0..max_iter {
        let mid = 0.5 * (a + b);
        if cdf(mid) < p { a = mid; } else { b = mid; }
        if b - a < tol { break; }
    }
    0.5 * (a + b)
}

// ============================================================
// Multivariate Normal helpers
// ============================================================

/// Multivariate normal PDF
pub fn multivariate_normal_pdf(x: &[f64], mean: &[f64], cov_inv: &[Vec<f64>], cov_det: f64) -> f64 {
    let n = x.len();
    let diff: Vec<f64> = x.iter().zip(mean).map(|(a, b)| a - b).collect();
    let mut quad = 0.0;
    for i in 0..n {
        for j in 0..n {
            quad += diff[i] * cov_inv[i][j] * diff[j];
        }
    }
    let log_norm = -0.5 * (n as f64 * (2.0 * PI).ln() + cov_det.ln());
    (log_norm - 0.5 * quad).exp()
}

// ============================================================
// Copulas
// ============================================================

/// Gaussian copula: transform uniform marginals to correlated normals and back
pub fn gaussian_copula_sample(rng: &mut Xoshiro256PlusPlus, rho: f64, n_samples: usize) -> Vec<(f64, f64)> {
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let z1 = rng.normal();
        let z2 = rho * z1 + (1.0 - rho * rho).sqrt() * rng.normal();
        let u1 = norm_cdf(z1);
        let u2 = norm_cdf(z2);
        samples.push((u1, u2));
    }
    samples
}

/// Clayton copula sampling
pub fn clayton_copula_sample(rng: &mut Xoshiro256PlusPlus, theta: f64, n_samples: usize) -> Vec<(f64, f64)> {
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let u1 = rng.next_f64();
        let v = rng.next_f64();
        let u2 = (u1.powf(-theta) * (v.powf(-theta / (1.0 + theta)) - 1.0) + 1.0).powf(-1.0 / theta);
        samples.push((u1, u2.clamp(0.0, 1.0)));
    }
    samples
}

/// Student-t copula sampling
pub fn student_t_copula_sample(
    rng: &mut Xoshiro256PlusPlus, rho: f64, df: f64, n_samples: usize,
) -> Vec<(f64, f64)> {
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let z1 = rng.normal();
        let z2 = rho * z1 + (1.0 - rho * rho).sqrt() * rng.normal();
        let chi2 = rng.chi_squared(df);
        let scale = (df / chi2).sqrt();
        let t1 = z1 * scale;
        let t2 = z2 * scale;
        let u1 = student_t_cdf(t1, df);
        let u2 = student_t_cdf(t2, df);
        samples.push((u1, u2));
    }
    samples
}

// ============================================================
// Helper functions (lgamma, incomplete beta/gamma)
// ============================================================

/// Log-gamma via Stirling's approximation with Lanczos coefficients
pub fn lgamma(x: f64) -> f64 {
    if x <= 0.0 { return f64::INFINITY; }
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        let reflect = PI / (PI * x).sin();
        return reflect.abs().ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut sum = c[0];
    for i in 1..9 { sum += c[i] / (x + i as f64); }
    let t = x + g + 0.5;
    0.5 * (2.0 * PI).ln() + (t).ln() * (x + 0.5) - t + sum.ln()
}

/// Regularized lower incomplete gamma P(a, x)
pub fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 { return 0.0; }
    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction
        1.0 - gamma_cf(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-14 { break; }
    }
    sum * (-x + a * x.ln() - lgamma(a)).exp()
}

fn gamma_cf(a: f64, x: f64) -> f64 {
    let mut f = 1e-30_f64;
    let mut c = 1e-30_f64;
    let mut d = 1.0 / (x + 1.0 - a);
    f = d;
    for n in 1..200 {
        let an = -(n as f64) * (n as f64 - a);
        let bn = x + 2.0 * n as f64 + 1.0 - a;
        d = 1.0 / (bn + an * d);
        c = bn + an / c;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-14 { break; }
    }
    f * (-x + a * x.ln() - lgamma(a)).exp()
}

/// Regularized incomplete beta I_x(a, b)
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let log_beta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - log_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(1e-30);
    let mut h = d;
    for m in 1..200 {
        let m_f = m as f64;
        // Even step
        let num = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 / (1.0 + num * d).max(1e-30);
        c = (1.0 + num / c).max(1e-30);
        h *= d * c;
        // Odd step
        let num = -((a + m_f) * (a + b + m_f) * x) / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 / (1.0 + num * d).max(1e-30);
        c = (1.0 + num / c).max(1e-30);
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-14 { break; }
    }
    h
}

// ============================================================
// Mixture distributions
// ============================================================

/// Gaussian mixture model evaluation
pub fn gmm_pdf(x: f64, weights: &[f64], means: &[f64], stds: &[f64]) -> f64 {
    weights.iter().zip(means.iter().zip(stds.iter()))
        .map(|(&w, (&m, &s))| w * normal_pdf(x, m, s))
        .sum()
}

/// Gaussian mixture CDF
pub fn gmm_cdf(x: f64, weights: &[f64], means: &[f64], stds: &[f64]) -> f64 {
    weights.iter().zip(means.iter().zip(stds.iter()))
        .map(|(&w, (&m, &s))| w * normal_cdf(x, m, s))
        .sum()
}

// ============================================================
// Stable distribution (approximate)
// ============================================================

/// Generate stable random variable (Chambers-Mallows-Stuck)
pub fn stable_random(rng: &mut Xoshiro256PlusPlus, alpha: f64, beta: f64, sigma: f64, mu: f64) -> f64 {
    let u = rng.uniform(-PI / 2.0, PI / 2.0);
    let w = rng.exponential(1.0);

    if (alpha - 1.0).abs() < 1e-10 {
        let x = ((PI / 2.0 + beta * u) * u.tan() - beta * ((PI / 2.0 * w * u.cos()) / (PI / 2.0 + beta * u)).ln()) * 2.0 / PI;
        sigma * x + mu
    } else {
        let zeta = -beta * (PI * alpha / 2.0).tan();
        let xi = (1.0 + zeta * zeta).sqrt().atan() / alpha;
        let s = (1.0 + zeta * zeta).powf(1.0 / (2.0 * alpha));

        let val = s * (alpha * (u + xi)).sin() / (u.cos()).powf(1.0 / alpha)
            * (u.cos() - alpha * (u + xi).sin() * w).powf((1.0 - alpha) / alpha);
        sigma * val + mu
    }
}

// ============================================================
// Distribution fitting
// ============================================================

/// Fit normal distribution via MLE
pub fn fit_normal(data: &[f64]) -> (f64, f64) {
    let mu = data.iter().sum::<f64>() / data.len() as f64;
    let sigma2 = data.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / data.len() as f64;
    (mu, sigma2.sqrt())
}

/// Fit exponential distribution via MLE
pub fn fit_exponential(data: &[f64]) -> f64 {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    1.0 / mean
}

/// Fit gamma distribution via MLE (moment matching)
pub fn fit_gamma(data: &[f64]) -> (f64, f64) {
    let m = data.iter().sum::<f64>() / data.len() as f64;
    let v = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() as f64 - 1.0);
    let shape = m * m / v;
    let scale = v / m;
    (shape, scale)
}

/// Fit GPD via probability-weighted moments
pub fn fit_gpd(data: &[f64]) -> (f64, f64) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    let b0: f64 = sorted.iter().sum::<f64>() / n;
    let b1: f64 = sorted.iter().enumerate()
        .map(|(i, &x)| i as f64 / (n - 1.0) * x)
        .sum::<f64>() / n;
    let xi = b0 / (2.0 * b1 - b0) - 2.0;
    let sigma = 2.0 * b0 * b1 / (2.0 * b1 - b0);
    (xi.max(-0.5), sigma.max(1e-10))
}

/// Negative log-likelihood for normal
pub fn normal_nll(data: &[f64], mu: f64, sigma: f64) -> f64 {
    let n = data.len() as f64;
    n * sigma.ln() + 0.5 * n * (2.0 * PI).ln()
        + data.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (2.0 * sigma * sigma)
}

/// AIC
pub fn aic(nll: f64, k: usize) -> f64 {
    2.0 * k as f64 + 2.0 * nll
}

/// BIC
pub fn bic(nll: f64, k: usize, n: usize) -> f64 {
    k as f64 * (n as f64).ln() + 2.0 * nll
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_cdf_symmetry() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((norm_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((norm_cdf(-1.96) - 0.025).abs() < 1e-3);
    }

    #[test]
    fn test_norm_ppf_inverse() {
        for &p in &[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99] {
            let x = norm_ppf(p);
            let p2 = norm_cdf(x);
            assert!((p - p2).abs() < 1e-4, "ppf/cdf inverse failed for p={}", p);
        }
    }

    #[test]
    fn test_xoshiro() {
        let mut rng = Xoshiro256PlusPlus::new(42);
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n { sum += rng.normal(); }
        let m = sum / n as f64;
        assert!(m.abs() < 0.05, "mean of normals too far from 0: {}", m);
    }

    #[test]
    fn test_student_t_cdf_symmetry() {
        let cdf_0 = student_t_cdf(0.0, 10.0);
        assert!((cdf_0 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_chi2_cdf() {
        // chi2(2) is exponential(0.5)
        let x = 2.0;
        let cdf = chi2_cdf(x, 2.0);
        let exp_cdf = 1.0 - (-x / 2.0_f64).exp();
        assert!((cdf - exp_cdf).abs() < 1e-4);
    }

    #[test]
    fn test_beta_cdf_boundary() {
        assert!((beta_cdf(0.5, 1.0, 1.0) - 0.5).abs() < 1e-6); // uniform on [0,1]
    }
}
