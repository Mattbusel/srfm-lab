// special_functions.rs — erf, gamma, beta, Bessel, Airy, hypergeometric, Fresnel

use std::f64::consts::PI;

/// Error function erf(x)
pub fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.28 (max error 2.5e-5 for simple, use better)
    // Use Horner form of rational approximation
    if x.abs() > 6.0 { return x.signum(); }
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -result } else { result }
}

/// Complementary error function erfc(x) = 1 - erf(x)
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Scaled complementary error function erfcx(x) = exp(x^2) * erfc(x)
pub fn erfcx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 * (x * x).exp() - erfcx(-x);
    }
    // Continued fraction for large x
    if x > 5.0 {
        let mut r = 0.0;
        for n in (1..=20).rev() {
            r = n as f64 * 0.5 / (x + r);
        }
        return 1.0 / ((x + r) * PI.sqrt());
    }
    (x * x).exp() * erfc(x)
}

/// Inverse error function erfinv(x)
pub fn erfinv(x: f64) -> f64 {
    if x <= -1.0 { return f64::NEG_INFINITY; }
    if x >= 1.0 { return f64::INFINITY; }
    if x.abs() < 1e-15 { return 0.0; }

    let a = 0.147;
    let ln_term = (1.0 - x * x).ln();
    let term = 2.0 / (PI * a) + ln_term / 2.0;
    let result = (term * term - ln_term / a).sqrt() - term;
    let result = result.sqrt();
    if x < 0.0 { -result } else { result }
}

/// Gamma function Γ(x) via Lanczos approximation
pub fn gamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() { return f64::INFINITY; }
    lgamma(x).exp() * if x > 0.0 || (x.floor() as i64) % 2 != 0 { 1.0 } else { -1.0 }
}

/// Log-gamma function ln(|Γ(x)|) via Lanczos approximation
pub fn lgamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() { return f64::INFINITY; }
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
    let t = x + 7.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// 1/Γ(x) — reciprocal gamma, well-defined everywhere
pub fn rgamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() { return 0.0; }
    (-lgamma(x)).exp()
}

/// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)
pub fn beta(a: f64, b: f64) -> f64 {
    (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()
}

/// Log-beta function
pub fn lbeta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Lower incomplete gamma γ(s, x) = ∫₀ˣ t^{s-1} e^{-t} dt
pub fn lower_incomplete_gamma(s: f64, x: f64) -> f64 {
    regularized_gamma_p(s, x) * gamma(s)
}

/// Upper incomplete gamma Γ(s, x) = ∫ₓ^∞ t^{s-1} e^{-t} dt
pub fn upper_incomplete_gamma(s: f64, x: f64) -> f64 {
    (1.0 - regularized_gamma_p(s, x)) * gamma(s)
}

/// Regularized lower incomplete gamma P(s, x) = γ(s,x)/Γ(s)
pub fn regularized_gamma_p(s: f64, x: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 { return 0.0; }
    if x < s + 1.0 {
        gamma_series_p(s, x)
    } else {
        1.0 - gamma_cf_q(s, x)
    }
}

/// Regularized upper incomplete gamma Q(s, x) = 1 - P(s, x)
pub fn regularized_gamma_q(s: f64, x: f64) -> f64 {
    1.0 - regularized_gamma_p(s, x)
}

fn gamma_series_p(s: f64, x: f64) -> f64 {
    let mut sum = 1.0 / s;
    let mut term = 1.0 / s;
    for n in 1..300 {
        term *= x / (s + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-15 { break; }
    }
    sum * (-x + s * x.ln() - lgamma(s)).exp()
}

fn gamma_cf_q(s: f64, x: f64) -> f64 {
    let mut f = 1e-30_f64;
    let mut c = 1e-30_f64;
    let mut d = 1.0 / (x + 1.0 - s);
    f = d;
    for n in 1..300 {
        let an = -(n as f64) * (n as f64 - s);
        let bn = x + 2.0 * n as f64 + 1.0 - s;
        d = 1.0 / (bn + an * d).max(1e-30);
        c = (bn + an / c).max(1e-30);
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 { break; }
    }
    f * (-x + s * x.ln() - lgamma(s)).exp()
}

/// Incomplete beta function B(x; a, b) = ∫₀ˣ t^{a-1}(1-t)^{b-1} dt
pub fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    regularized_incomplete_beta(a, b, x) * beta(a, b)
}

/// Regularized incomplete beta I_x(a, b)
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let log_beta = lbeta(a, b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - log_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let mut c = 1.0;
    let mut d = (1.0 - (a + b) * x / (a + 1.0)).recip().max(1e-30);
    let mut h = d;
    for m in 1..300 {
        let mf = m as f64;
        let num = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = (1.0 + num * d).recip().max(1e-30);
        c = (1.0 + num / c).max(1e-30);
        h *= d * c;
        let num = -((a + mf) * (a + b + mf) * x) / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = (1.0 + num * d).recip().max(1e-30);
        c = (1.0 + num / c).max(1e-30);
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-15 { break; }
    }
    h
}

/// Digamma function ψ(x) = d/dx ln Γ(x)
pub fn digamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() { return f64::NAN; }
    let mut result = 0.0;
    let mut x = x;
    // Recurrence: ψ(x+1) = ψ(x) + 1/x
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion for large x
    result += x.ln() - 0.5 / x;
    let x2 = 1.0 / (x * x);
    result -= x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0 - x2 * (1.0/240.0 - x2 / 132.0))));
    result
}

/// Trigamma function ψ₁(x) = d²/dx² ln Γ(x)
pub fn trigamma(x: f64) -> f64 {
    let mut result = 0.0;
    let mut x = x;
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let x2 = 1.0 / (x * x);
    result += 1.0 / x + x2 * (0.5 + x2 * (1.0/6.0 - x2 * (1.0/30.0 - x2 * (1.0/42.0 - x2 / 30.0))));
    result
}

/// Polygamma function ψ^(n)(x) for n >= 0
pub fn polygamma(n: usize, x: f64) -> f64 {
    match n {
        0 => digamma(x),
        1 => trigamma(x),
        _ => {
            // Finite difference approximation
            let h = 1e-5;
            let pn1_plus = polygamma(n - 1, x + h);
            let pn1_minus = polygamma(n - 1, x - h);
            (pn1_plus - pn1_minus) / (2.0 * h)
        }
    }
}

/// Bessel function of the first kind J₀(x)
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let p = -2.2002690873032541e6 + y * (7.2374568611970124e7 + y * (-5.6827243015866594e8
            + y * (1.7710478032601086e9 + y * (-2.2213455048177972e9 + y * 7.5038560685498384e8))));
        let q = 1.0 + y * (1.7710478032601086e-2 + y * (1.2472079693809423e-4
            + y * (4.4141399782594284e-7 + y * 7.0690239598498036e-10)));
        // Simplified series
        bessel_j_series(0, x)
    } else {
        let z = 8.0 / ax;
        let z2 = z * z;
        let theta = ax - 0.25 * PI;
        let p0 = 1.0 + z2 * (-1.098628627e-2 + z2 * (2.734510407e-4 + z2 * (-2.073370639e-5 + z2 * 2.093887211e-6)));
        let q0 = z * (-1.562499995e-1 + z2 * (1.430488765e-3 + z2 * (-6.911147651e-5 + z2 * (7.621095161e-6 - z2 * 9.34945152e-7))));
        (2.0 / (PI * ax)).sqrt() * (p0 * theta.cos() - q0 * theta.sin())
    }
}

/// Bessel function of the first kind J₁(x)
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        bessel_j_series(1, x)
    } else {
        let z = 8.0 / ax;
        let z2 = z * z;
        let theta = ax - 0.75 * PI;
        let p1 = 1.0 + z2 * (1.83105e-2 + z2 * (-2.5155e-4 + z2 * 1.7105e-5));
        let q1 = z * (4.6875e-2 + z2 * (-2.002e-3 + z2 * (8.449e-5 + z2 * (-8.82e-6))));
        let result = (2.0 / (PI * ax)).sqrt() * (p1 * theta.cos() - q1 * theta.sin());
        if x < 0.0 { -result } else { result }
    }
}

/// Bessel J_n via series expansion
fn bessel_j_series(n: i32, x: f64) -> f64 {
    let x_half = x / 2.0;
    let mut term = x_half.powi(n) / gamma(n as f64 + 1.0);
    let mut sum = term;
    let x2_quarter = -x * x / 4.0;
    for k in 1..100 {
        term *= x2_quarter / (k as f64 * (k as f64 + n as f64));
        sum += term;
        if term.abs() < sum.abs() * 1e-15 { break; }
    }
    sum
}

/// Bessel J_n for integer order n
pub fn bessel_jn(n: i32, x: f64) -> f64 {
    if n < 0 {
        let result = bessel_jn(-n, x);
        return if (-n) % 2 == 0 { result } else { -result };
    }
    match n {
        0 => bessel_j0(x),
        1 => bessel_j1(x),
        _ => {
            if x.abs() < 1e-15 { return 0.0; }
            // Miller's backward recurrence for stability
            let ax = x.abs();
            if ax > n as f64 {
                // Forward recurrence OK
                let mut jm1 = bessel_j0(x);
                let mut j = bessel_j1(x);
                for k in 1..n {
                    let jnew = 2.0 * k as f64 / x * j - jm1;
                    jm1 = j;
                    j = jnew;
                }
                j
            } else {
                bessel_j_series(n, x)
            }
        }
    }
}

/// Bessel function of the second kind Y₀(x)
pub fn bessel_y0(x: f64) -> f64 {
    if x <= 0.0 { return f64::NEG_INFINITY; }
    if x < 8.0 {
        let j0 = bessel_j0(x);
        let euler = 0.5772156649015329;
        // Y0 = (2/π)(J0 ln(x/2) + γ J0 + series correction)
        // Use series
        let mut sum = 0.0;
        let x2_quarter = -x * x / 4.0;
        let mut term = 1.0;
        let mut h = 0.0_f64; // harmonic number
        for k in 1..60 {
            term *= x2_quarter / (k as f64 * k as f64);
            h += 1.0 / k as f64;
            sum += term * h;
        }
        (2.0 / PI) * ((x / 2.0).ln() + euler) * j0 + (2.0 / PI) * sum
    } else {
        let z = 8.0 / x;
        let z2 = z * z;
        let theta = x - 0.25 * PI;
        let p0 = 1.0 + z2 * (-1.098628627e-2 + z2 * 2.734510407e-4);
        let q0 = z * (-1.562499995e-1 + z2 * 1.430488765e-3);
        (2.0 / (PI * x)).sqrt() * (p0 * theta.sin() + q0 * theta.cos())
    }
}

/// Bessel function of the second kind Y₁(x)
pub fn bessel_y1(x: f64) -> f64 {
    if x <= 0.0 { return f64::NEG_INFINITY; }
    if x < 8.0 {
        let j1 = bessel_j1(x);
        let euler = 0.5772156649015329;
        // Series for Y1
        let mut sum = -2.0 / (PI * x);
        let x2 = x * x;
        let mut term = x / 2.0;
        for k in 1..60 {
            let kf = k as f64;
            term *= -x2 / (4.0 * kf * (kf + 1.0));
            let hk: f64 = (1..=k).map(|j| 1.0 / j as f64).sum();
            let hk1: f64 = (1..=(k + 1)).map(|j| 1.0 / j as f64).sum();
            sum += term * (hk + hk1);
        }
        (2.0 / PI) * ((x / 2.0).ln() + euler) * j1 + sum / PI
    } else {
        let z = 8.0 / x;
        let z2 = z * z;
        let theta = x - 0.75 * PI;
        let p1 = 1.0 + z2 * 1.83105e-2;
        let q1 = z * 4.6875e-2;
        (2.0 / (PI * x)).sqrt() * (p1 * theta.sin() + q1 * theta.cos())
    }
}

/// Modified Bessel function I₀(x)
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        let result = (0.39894228 + y * (0.01328592 + y * (0.00225319
            + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706
            + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
            / ax.sqrt();
        result * ax.exp()
    }
}

/// Modified Bessel function I₁(x)
pub fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let result = if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
            + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))))
    } else {
        let y = 3.75 / ax;
        let r = 0.02282967 + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059));
        let r = 0.39894228 + y * (-0.03988024 + y * (-0.00362018 + y * (0.00163801
            + y * (-0.01031555 + y * r))));
        r * ax.exp() / ax.sqrt()
    };
    if x < 0.0 { -result } else { result }
}

/// Modified Bessel function K₀(x)
pub fn bessel_k0(x: f64) -> f64 {
    if x <= 0.0 { return f64::INFINITY; }
    if x <= 2.0 {
        let y = x * x / 4.0;
        (-x.ln() / 2.0 + 0.5772156649015329) * bessel_i0(x)
            + y * (0.42278420 + y * (0.23069756 + y * (0.03488590
            + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))))
    } else {
        let y = 2.0 / x;
        (1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446
            + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))))))
            / x.sqrt() * (-x).exp()
    }
}

/// Modified Bessel function K₁(x)
pub fn bessel_k1(x: f64) -> f64 {
    if x <= 0.0 { return f64::INFINITY; }
    if x <= 2.0 {
        let y = x * x / 4.0;
        (x.ln() / 2.0) * bessel_i1(x) + (1.0 / x) * (1.0 + y * (0.15443144
            + y * (-0.67278579 + y * (-0.18156897 + y * (-0.01919402
            + y * (-0.00110404 + y * -0.00004686))))))
    } else {
        let y = 2.0 / x;
        (1.25331414 + y * (0.23498619 + y * (-0.03655620 + y * (0.01504268
            + y * (-0.00780353 + y * (0.00325614 + y * -0.00068245))))))
            / x.sqrt() * (-x).exp()
    }
}

/// Airy function Ai(x)
pub fn airy_ai(x: f64) -> f64 {
    if x > 5.0 {
        // Asymptotic for large positive x
        let zeta = 2.0 / 3.0 * x.powf(1.5);
        0.5 * x.powf(-0.25) / PI.sqrt() * (-zeta).exp()
    } else if x < -5.0 {
        // Asymptotic for large negative x
        let zeta = 2.0 / 3.0 * (-x).powf(1.5);
        (-x).powf(-0.25) / PI.sqrt() * (zeta - PI / 4.0).cos()
    } else {
        // Series expansion
        let mut f = 1.0;
        let mut g = x;
        let mut sum_f = f;
        let mut sum_g = g;
        for k in 1..100 {
            f *= x * x * x / ((3 * k) as f64 * (3 * k - 1) as f64);
            g *= x * x * x / ((3 * k + 1) as f64 * (3 * k) as f64);
            sum_f += f;
            sum_g += g;
            if f.abs() < 1e-15 && g.abs() < 1e-15 { break; }
        }
        let c1 = 1.0 / (3.0_f64.powf(2.0 / 3.0) * gamma(2.0 / 3.0));
        let c2 = -1.0 / (3.0_f64.powf(1.0 / 3.0) * gamma(1.0 / 3.0));
        c1 * sum_f + c2 * sum_g
    }
}

/// Airy function Bi(x)
pub fn airy_bi(x: f64) -> f64 {
    if x > 5.0 {
        let zeta = 2.0 / 3.0 * x.powf(1.5);
        x.powf(-0.25) / PI.sqrt() * zeta.exp()
    } else if x < -5.0 {
        let zeta = 2.0 / 3.0 * (-x).powf(1.5);
        (-x).powf(-0.25) / PI.sqrt() * (zeta - PI / 4.0).sin().abs()
    } else {
        let mut f = 1.0;
        let mut g = x;
        let mut sum_f = f;
        let mut sum_g = g;
        for k in 1..100 {
            f *= x * x * x / ((3 * k) as f64 * (3 * k - 1) as f64);
            g *= x * x * x / ((3 * k + 1) as f64 * (3 * k) as f64);
            sum_f += f;
            sum_g += g;
            if f.abs() < 1e-15 && g.abs() < 1e-15 { break; }
        }
        let c1 = 1.0 / (3.0_f64.powf(1.0 / 6.0) * gamma(2.0 / 3.0));
        let c2 = 3.0_f64.powf(1.0 / 6.0) / gamma(1.0 / 3.0);
        c1 * sum_f + c2 * sum_g
    }
}

/// Hypergeometric function ₂F₁(a, b; c; z)
pub fn hyper_2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
    if z.abs() < 1.0 {
        hyper_2f1_series(a, b, c, z)
    } else if z < 0.0 {
        // Euler transformation: 2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1))
        let w = z / (z - 1.0);
        (1.0 - z).powf(-a) * hyper_2f1_series(a, c - b, c, w)
    } else {
        // Attempt analytic continuation or just return NaN for |z| >= 1
        f64::NAN
    }
}

fn hyper_2f1_series(a: f64, b: f64, c: f64, z: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for n in 0..500 {
        let nf = n as f64;
        term *= (a + nf) * (b + nf) / ((c + nf) * (nf + 1.0)) * z;
        sum += term;
        if term.abs() < sum.abs() * 1e-15 { break; }
    }
    sum
}

/// Confluent hypergeometric function ₁F₁(a; b; z) = M(a, b, z)
pub fn hyper_1f1(a: f64, b: f64, z: f64) -> f64 {
    if z.abs() < 20.0 {
        let mut sum = 1.0;
        let mut term = 1.0;
        for n in 0..300 {
            let nf = n as f64;
            term *= (a + nf) / ((b + nf) * (nf + 1.0)) * z;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 { break; }
        }
        sum
    } else if z > 0.0 {
        // Asymptotic: M(a,b,z) ~ Γ(b)/Γ(a) * e^z * z^{a-b}
        let coeff = (lgamma(b) - lgamma(a)).exp();
        coeff * z.exp() * z.powf(a - b)
    } else {
        // Kummer transformation: M(a,b,z) = e^z M(b-a, b, -z)
        z.exp() * hyper_1f1(b - a, b, -z)
    }
}

/// Fresnel integrals S(x) and C(x)
/// S(x) = ∫₀ˣ sin(πt²/2) dt
/// C(x) = ∫₀ˣ cos(πt²/2) dt
pub fn fresnel(x: f64) -> (f64, f64) {
    let ax = x.abs();
    let (s, c) = if ax < 1.5 {
        fresnel_series(ax)
    } else if ax < 5.0 {
        fresnel_auxiliary(ax)
    } else {
        fresnel_asymptotic(ax)
    };
    if x < 0.0 { (-s, -c) } else { (s, c) }
}

fn fresnel_series(x: f64) -> (f64, f64) {
    let x2 = x * x;
    let x3 = x2 * x;
    let pi_half = PI / 2.0;
    let mut s = 0.0;
    let mut c = 0.0;
    let mut term_s = x3 * pi_half / 3.0;
    let mut term_c = x;
    s = term_s;
    c = term_c;
    for k in 1..50 {
        let kf = k as f64;
        term_c *= -pi_half * pi_half * x2 * x2 / ((4.0 * kf - 1.0) * (4.0 * kf) * (2.0 * kf));
        // Actually just use direct power series
        // S(x) = Σ (-1)^k (πx²/2)^{2k+1} x / ((2k+1)(4k+3))
        // Simplified approach:
        break;
    }

    // Direct summation
    let mut ss = 0.0;
    let mut sc = 0.0;
    let t = PI * x * x / 2.0;
    let mut term = t * x / 3.0;
    ss = term;
    sc = x;
    let mut sign = -1.0;
    for k in 1..60 {
        let kf = k as f64;
        // S series
        term = sign * t.powi(2 * k as i32 + 1) * x / (factorial_approx(2 * k + 1) * (4.0 * kf + 3.0));
        ss += sign * (PI / 2.0).powi(2 * k as i32 + 1) * x.powi(4 * k as i32 + 3)
            / (factorial_approx(2 * k + 1) * (4.0 * kf + 3.0));
        // Simpler: just integrate numerically
        break;
    }

    // Fall back to numerical integration
    let n = 100;
    let h = x / n as f64;
    let mut s_val = 0.0;
    let mut c_val = 0.0;
    for i in 0..n {
        let t0 = i as f64 * h;
        let t1 = (i + 1) as f64 * h;
        let tm = 0.5 * (t0 + t1);
        // Simpson
        s_val += h / 6.0 * ((PI * t0 * t0 / 2.0).sin() + 4.0 * (PI * tm * tm / 2.0).sin() + (PI * t1 * t1 / 2.0).sin());
        c_val += h / 6.0 * ((PI * t0 * t0 / 2.0).cos() + 4.0 * (PI * tm * tm / 2.0).cos() + (PI * t1 * t1 / 2.0).cos());
    }
    (s_val, c_val)
}

fn fresnel_auxiliary(x: f64) -> (f64, f64) {
    fresnel_numerical(x)
}

fn fresnel_asymptotic(x: f64) -> (f64, f64) {
    let t = PI * x * x;
    let s = 0.5 - (1.0 / (PI * x)) * (t / 2.0).cos();
    let c = 0.5 + (1.0 / (PI * x)) * (t / 2.0).sin();
    (s, c)
}

fn fresnel_numerical(x: f64) -> (f64, f64) {
    let n = 200;
    let h = x / n as f64;
    let mut s = 0.0;
    let mut c = 0.0;
    for i in 0..n {
        let t0 = i as f64 * h;
        let t1 = (i + 1) as f64 * h;
        let tm = 0.5 * (t0 + t1);
        s += h / 6.0 * ((PI * t0 * t0 / 2.0).sin() + 4.0 * (PI * tm * tm / 2.0).sin() + (PI * t1 * t1 / 2.0).sin());
        c += h / 6.0 * ((PI * t0 * t0 / 2.0).cos() + 4.0 * (PI * tm * tm / 2.0).cos() + (PI * t1 * t1 / 2.0).cos());
    }
    (s, c)
}

fn factorial_approx(n: usize) -> f64 {
    gamma(n as f64 + 1.0)
}

/// Pochhammer symbol (rising factorial) (a)_n
pub fn pochhammer(a: f64, n: usize) -> f64 {
    let mut result = 1.0;
    for k in 0..n { result *= a + k as f64; }
    result
}

/// Binomial coefficient C(n, k)
pub fn binomial_coeff(n: u64, k: u64) -> f64 {
    if k > n { return 0.0; }
    (lgamma(n as f64 + 1.0) - lgamma(k as f64 + 1.0) - lgamma((n - k) as f64 + 1.0)).exp()
}

/// Catalan number C_n
pub fn catalan(n: u64) -> f64 {
    binomial_coeff(2 * n, n) / (n as f64 + 1.0)
}

/// Bernoulli numbers B_n (first few)
pub fn bernoulli_number(n: usize) -> f64 {
    // Precomputed for small n
    match n {
        0 => 1.0,
        1 => -0.5,
        2 => 1.0 / 6.0,
        4 => -1.0 / 30.0,
        6 => 1.0 / 42.0,
        8 => -1.0 / 30.0,
        10 => 5.0 / 66.0,
        12 => -691.0 / 2730.0,
        14 => 7.0 / 6.0,
        _ => {
            if n % 2 == 1 && n > 1 { return 0.0; }
            // Compute via Akiyama-Tanigawa algorithm
            let mut a = Vec::with_capacity(n + 1);
            for m in 0..=n {
                a.push(1.0 / (m + 1) as f64);
                for j in (1..=m).rev() {
                    a[j - 1] = j as f64 * (a[j - 1] - a[j]);
                }
            }
            a[0]
        }
    }
}

/// Euler numbers E_n (even only, odd are 0)
pub fn euler_number(n: usize) -> f64 {
    if n % 2 == 1 { return 0.0; }
    match n {
        0 => 1.0,
        2 => -1.0,
        4 => 5.0,
        6 => -61.0,
        8 => 1385.0,
        10 => -50521.0,
        _ => {
            // Sum formula
            let m = n / 2;
            let mut sum = 0.0;
            for k in 0..=m {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                sum += sign * binomial_coeff(n as u64, (2 * k) as u64);
            }
            sum
        }
    }
}

/// Riemann zeta function ζ(s) for real s > 1
pub fn zeta(s: f64) -> f64 {
    if s <= 1.0 { return f64::NAN; }
    if (s - 2.0).abs() < 1e-15 { return PI * PI / 6.0; }
    if (s - 4.0).abs() < 1e-15 { return PI.powi(4) / 90.0; }

    // Borwein algorithm
    let n = 20;
    let mut dk = vec![0.0; n + 1];
    dk[0] = 1.0;
    for k in 1..=n {
        dk[k] = dk[k - 1] + n as f64 * binomial_coeff(n as u64, k as u64) / binomial_coeff(2 * n as u64, k as u64);
    }
    let dn = dk[n];

    let mut sum = 0.0;
    for k in 0..n {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * (dk[k] - dn) / (k as f64 + 1.0).powf(s);
    }
    -sum / (dn * (1.0 - 2.0_f64.powf(1.0 - s)))
}

/// Exponential integral Ei(x)
pub fn exp_integral_ei(x: f64) -> f64 {
    if x.abs() < 1e-15 { return f64::NEG_INFINITY; }
    let euler = 0.5772156649015329;
    if x.abs() < 40.0 {
        let mut sum = euler + x.abs().ln();
        let mut term = x;
        for k in 1..200 {
            sum += term / (k as f64 * k as f64);
            term *= x / (k + 1) as f64;
            if term.abs() < 1e-15 { break; }
        }
        sum
    } else {
        // Asymptotic
        let mut sum = 1.0;
        let mut term = 1.0;
        for k in 1..20 {
            term *= k as f64 / x;
            sum += term;
            if term.abs() < 1e-10 { break; }
        }
        x.exp() / x * sum
    }
}

/// Sine integral Si(x) = ∫₀ˣ sin(t)/t dt
pub fn si(x: f64) -> f64 {
    let n = (x.abs() * 10.0 + 50.0) as usize;
    let n = n.max(100).min(1000);
    let h = x / n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let t0 = (i as f64 + 0.001) * h;
        let t1 = (i as f64 + 0.5) * h;
        let t2 = (i as f64 + 0.999) * h;
        if h.abs() < 1e-15 { continue; }
        sum += h / 6.0 * (sinc(t0) + 4.0 * sinc(t1) + sinc(t2));
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-15 { 1.0 } else { x.sin() / x }
}

/// Cosine integral Ci(x) = γ + ln(x) + ∫₀ˣ (cos(t)-1)/t dt
pub fn ci(x: f64) -> f64 {
    if x <= 0.0 { return f64::NEG_INFINITY; }
    let euler = 0.5772156649015329;
    let n = (x * 10.0 + 50.0) as usize;
    let n = n.max(100).min(1000);
    let h = x / n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let t0 = (i as f64 + 0.01) * h;
        let tm = (i as f64 + 0.5) * h;
        let t1 = (i as f64 + 0.99) * h;
        let f0 = if t0.abs() < 1e-15 { 0.0 } else { (t0.cos() - 1.0) / t0 };
        let fm = if tm.abs() < 1e-15 { 0.0 } else { (tm.cos() - 1.0) / tm };
        let f1 = if t1.abs() < 1e-15 { 0.0 } else { (t1.cos() - 1.0) / t1 };
        sum += h / 6.0 * (f0 + 4.0 * fm + f1);
    }
    euler + x.ln() + sum
}

/// Debye function D_n(x) = (n/x^n) ∫₀ˣ t^n/(e^t - 1) dt
pub fn debye(n: usize, x: f64) -> f64 {
    if x.abs() < 1e-15 { return 1.0; }
    let steps = 200;
    let h = x / steps as f64;
    let nf = n as f64;
    let mut sum = 0.0;
    for i in 1..steps {
        let t = i as f64 * h;
        let f = t.powi(n as i32) / (t.exp() - 1.0);
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * f;
    }
    // Endpoints: t=0 limit is 0 for n>=1, t=x
    let fx = x.powi(n as i32) / (x.exp() - 1.0);
    sum += fx;
    sum *= h / 3.0;
    nf / x.powi(n as i32) * sum
}

/// Lambert W function (principal branch W₀)
pub fn lambert_w0(x: f64) -> f64 {
    if x < -1.0 / std::f64::consts::E + 1e-10 { return f64::NAN; }
    if x.abs() < 1e-10 { return x; }
    // Initial guess
    let mut w = if x < 1.0 { x } else { x.ln() - x.ln().ln() };
    // Halley's method
    for _ in 0..50 {
        let ew = w.exp();
        let wew = w * ew;
        let f = wew - x;
        let fp = ew * (w + 1.0);
        let fpp = ew * (w + 2.0);
        let dw = f / (fp - f * fpp / (2.0 * fp));
        w -= dw;
        if dw.abs() < 1e-14 { break; }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf() {
        assert!(erf(0.0).abs() < 1e-10);
        assert!((erf(1.0) - 0.8427007929).abs() < 1e-4);
        assert!((erf(-1.0) + 0.8427007929).abs() < 1e-4);
    }

    #[test]
    fn test_gamma() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-10);
        assert!((gamma(5.0) - 24.0).abs() < 1e-6);
        assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_beta() {
        assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-10);
        assert!((beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_j0() {
        assert!((bessel_j0(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_digamma() {
        assert!((digamma(1.0) + 0.5772156649).abs() < 1e-6);
    }

    #[test]
    fn test_lambert_w() {
        let w = lambert_w0(1.0);
        assert!((w * w.exp() - 1.0).abs() < 1e-10);
    }
}
