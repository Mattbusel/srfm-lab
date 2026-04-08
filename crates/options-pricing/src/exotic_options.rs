use std::f64::consts::PI;
use crate::black_scholes::{norm_cdf, norm_pdf, norm_inv, BSParams, OptionType, bs_price};

// ═══════════════════════════════════════════════════════════════════════════
// BARRIER OPTIONS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierType {
    DownIn,
    DownOut,
    UpIn,
    UpOut,
}

#[derive(Debug, Clone, Copy)]
pub struct BarrierOption {
    pub spot: f64,
    pub strike: f64,
    pub barrier: f64,
    pub rebate: f64,
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub time_to_expiry: f64,
    pub opt_type: OptionType,
    pub barrier_type: BarrierType,
}

impl BarrierOption {
    pub fn new(
        spot: f64, strike: f64, barrier: f64, rebate: f64,
        rate: f64, dividend: f64, vol: f64, tte: f64,
        opt_type: OptionType, barrier_type: BarrierType,
    ) -> Self {
        Self { spot, strike, barrier, rebate, rate, dividend, vol, time_to_expiry: tte, opt_type, barrier_type }
    }

    /// Check if barrier has already been breached.
    pub fn is_knocked(&self) -> bool {
        match self.barrier_type {
            BarrierType::DownIn | BarrierType::DownOut => self.spot <= self.barrier,
            BarrierType::UpIn | BarrierType::UpOut => self.spot >= self.barrier,
        }
    }
}

/// Closed-form barrier option pricing (Merton-Reiner-Rubinstein).
/// Handles all 16 cases (4 barrier types x 2 option types x 2 strike vs barrier).
pub fn barrier_price(opt: &BarrierOption) -> f64 {
    let s = opt.spot;
    let k = opt.strike;
    let h = opt.barrier;
    let r = opt.rate;
    let q = opt.dividend;
    let v = opt.vol;
    let t = opt.time_to_expiry;
    let rebate = opt.rebate;

    if t <= 0.0 {
        let payoff = match opt.opt_type {
            OptionType::Call => (s - k).max(0.0),
            OptionType::Put => (k - s).max(0.0),
        };
        return match opt.barrier_type {
            BarrierType::DownIn | BarrierType::UpIn => {
                if opt.is_knocked() { payoff } else { rebate }
            }
            BarrierType::DownOut | BarrierType::UpOut => {
                if opt.is_knocked() { rebate } else { payoff }
            }
        };
    }

    let v2 = v * v;
    let sqrt_t = t.sqrt();
    let mu = (r - q - 0.5 * v2) / v2;
    let lambda = (mu * mu + 2.0 * r / v2).sqrt();
    let x1 = (s / k).ln() / (v * sqrt_t) + (1.0 + mu) * v * sqrt_t;
    let x2 = (s / h).ln() / (v * sqrt_t) + (1.0 + mu) * v * sqrt_t;
    let y1 = (h * h / (s * k)).ln() / (v * sqrt_t) + (1.0 + mu) * v * sqrt_t;
    let y2 = (h / s).ln() / (v * sqrt_t) + (1.0 + mu) * v * sqrt_t;
    let z = (h / s).ln() / (v * sqrt_t) + lambda * v * sqrt_t;

    let phi = if matches!(opt.opt_type, OptionType::Call) { 1.0 } else { -1.0 };
    let eta = match opt.barrier_type {
        BarrierType::DownIn | BarrierType::DownOut => 1.0,
        BarrierType::UpIn | BarrierType::UpOut => -1.0,
    };

    let df = (-r * t).exp();
    let qf = (-q * t).exp();

    let a = phi * s * qf * norm_cdf(phi * x1)
        - phi * k * df * norm_cdf(phi * x1 - phi * v * sqrt_t);

    let b = phi * s * qf * norm_cdf(phi * x2)
        - phi * k * df * norm_cdf(phi * x2 - phi * v * sqrt_t);

    let c = phi * s * qf * (h / s).powf(2.0 * (mu + 1.0)) * norm_cdf(eta * y1)
        - phi * k * df * (h / s).powf(2.0 * mu) * norm_cdf(eta * y1 - eta * v * sqrt_t);

    let d_val = phi * s * qf * (h / s).powf(2.0 * (mu + 1.0)) * norm_cdf(eta * y2)
        - phi * k * df * (h / s).powf(2.0 * mu) * norm_cdf(eta * y2 - eta * v * sqrt_t);

    let e_val = rebate * df * (norm_cdf(eta * x2 - eta * v * sqrt_t)
        - (h / s).powf(2.0 * mu) * norm_cdf(eta * y2 - eta * v * sqrt_t));

    let f_val = rebate * ((h / s).powf(mu + lambda) * norm_cdf(eta * z)
        + (h / s).powf(mu - lambda) * norm_cdf(eta * z - 2.0 * eta * lambda * v * sqrt_t));

    match (opt.opt_type, opt.barrier_type) {
        // Down-and-in call
        (OptionType::Call, BarrierType::DownIn) => {
            if k > h {
                c + e_val
            } else {
                a - b + d_val + e_val
            }
        }
        // Up-and-in call
        (OptionType::Call, BarrierType::UpIn) => {
            if k > h {
                a + e_val
            } else {
                b - c + d_val + e_val
            }
        }
        // Down-and-in put
        (OptionType::Put, BarrierType::DownIn) => {
            if k > h {
                b - c + d_val + e_val
            } else {
                a + e_val
            }
        }
        // Up-and-in put
        (OptionType::Put, BarrierType::UpIn) => {
            if k > h {
                a - b + d_val + e_val
            } else {
                c + e_val
            }
        }
        // Down-and-out call
        (OptionType::Call, BarrierType::DownOut) => {
            if k > h {
                a - c + f_val
            } else {
                b - d_val + f_val
            }
        }
        // Up-and-out call
        (OptionType::Call, BarrierType::UpOut) => {
            if k > h {
                f_val
            } else {
                a - b + c - d_val + f_val
            }
        }
        // Down-and-out put
        (OptionType::Put, BarrierType::DownOut) => {
            if k > h {
                a - b + c - d_val + f_val
            } else {
                f_val
            }
        }
        // Up-and-out put
        (OptionType::Put, BarrierType::UpOut) => {
            if k > h {
                b - d_val + f_val
            } else {
                a - c + f_val
            }
        }
    }
}

/// In-out parity: knock-in + knock-out = vanilla + rebate_value
pub fn barrier_in_out_parity(
    spot: f64, strike: f64, barrier: f64, rebate: f64,
    rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType, is_down: bool,
) -> f64 {
    let bt_in = if is_down { BarrierType::DownIn } else { BarrierType::UpIn };
    let bt_out = if is_down { BarrierType::DownOut } else { BarrierType::UpOut };
    let opt_in = BarrierOption::new(spot, strike, barrier, rebate, rate, dividend, vol, tte, opt_type, bt_in);
    let opt_out = BarrierOption::new(spot, strike, barrier, rebate, rate, dividend, vol, tte, opt_type, bt_out);
    let p = BSParams::new(spot, strike, rate, dividend, vol, tte);
    let vanilla = bs_price(&p, opt_type);
    let knock_in = barrier_price(&opt_in);
    let knock_out = barrier_price(&opt_out);
    // Should be approximately zero
    knock_in + knock_out - vanilla - rebate * (-rate * tte).exp()
}

/// Barrier option delta via finite difference.
pub fn barrier_delta(opt: &BarrierOption, ds: f64) -> f64 {
    let mut up = *opt;
    up.spot += ds;
    let mut dn = *opt;
    dn.spot -= ds;
    (barrier_price(&up) - barrier_price(&dn)) / (2.0 * ds)
}

/// Barrier option gamma via finite difference.
pub fn barrier_gamma(opt: &BarrierOption, ds: f64) -> f64 {
    let mut up = *opt;
    up.spot += ds;
    let mut dn = *opt;
    dn.spot -= ds;
    (barrier_price(&up) - 2.0 * barrier_price(opt) + barrier_price(&dn)) / (ds * ds)
}

/// Barrier option vega via finite difference.
pub fn barrier_vega(opt: &BarrierOption, dvol: f64) -> f64 {
    let mut up = *opt;
    up.vol += dvol;
    let mut dn = *opt;
    dn.vol -= dvol;
    (barrier_price(&up) - barrier_price(&dn)) / (2.0 * dvol)
}

/// Barrier option theta via finite difference.
pub fn barrier_theta(opt: &BarrierOption, dt: f64) -> f64 {
    let mut short = *opt;
    short.time_to_expiry = (opt.time_to_expiry - dt).max(0.0);
    (barrier_price(&short) - barrier_price(opt)) / dt
}

/// Double barrier option (knock-out): S must stay between lower and upper barriers.
pub fn double_barrier_knockout(
    spot: f64, strike: f64, lower: f64, upper: f64,
    rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType, n_terms: usize,
) -> f64 {
    if spot <= lower || spot >= upper {
        return 0.0;
    }
    let phi = match opt_type { OptionType::Call => 1.0, OptionType::Put => -1.0 };
    let df = (-rate * tte).exp();
    let sqrt_t = tte.sqrt();
    let log_ratio = (upper / lower).ln();
    let mut price = 0.0;

    for n in 1..=n_terms {
        let nf = n as f64;
        let beta_n = PI * nf / log_ratio;
        let alpha_n = -0.5 * vol * vol * beta_n * beta_n;
        let drift = rate - dividend - 0.5 * vol * vol;

        // Fourier sine series coefficients
        let sin_term_s = (beta_n * (spot / lower).ln()).sin();
        let integral = fourier_barrier_integral(strike, lower, upper, beta_n, phi);
        let decay = ((alpha_n + drift * beta_n) * tte).exp();
        // Actually using the Ikeda-Kunitomo expansion
        price += sin_term_s * integral * decay;
    }

    price * 2.0 * df / log_ratio
}

fn fourier_barrier_integral(strike: f64, lower: f64, upper: f64, beta: f64, phi: f64) -> f64 {
    // Numerical integration of payoff * sin(beta * ln(x/L)) over [L, U]
    let n_steps = 200;
    let dx = (upper - lower) / n_steps as f64;
    let mut sum = 0.0;
    for i in 0..=n_steps {
        let x = lower + i as f64 * dx;
        let payoff = if phi > 0.0 { (x - strike).max(0.0) } else { (strike - x).max(0.0) };
        let sin_val = (beta * (x / lower).ln()).sin();
        let w = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
        sum += w * payoff * sin_val / x;
    }
    sum * dx
}

// ═══════════════════════════════════════════════════════════════════════════
// LOOKBACK OPTIONS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub struct LookbackFixed {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub time_to_expiry: f64,
    pub s_max: f64,  // running maximum for calls
    pub s_min: f64,  // running minimum for puts
    pub opt_type: OptionType,
}

/// Fixed-strike lookback call: payoff = max(S_max - K, 0)
pub fn lookback_fixed_call(opt: &LookbackFixed) -> f64 {
    let s = opt.spot;
    let k = opt.strike;
    let r = opt.rate;
    let q = opt.dividend;
    let v = opt.vol;
    let t = opt.time_to_expiry;
    let m = opt.s_max;

    if t <= 0.0 {
        return (m - k).max(0.0);
    }

    let v2 = v * v;
    let sqrt_t = t.sqrt();
    let df = (-r * t).exp();
    let qf = (-q * t).exp();

    if m >= k {
        // Already in the money from historical max
        let d1 = ((s / m).ln() + (r - q + 0.5 * v2) * t) / (v * sqrt_t);
        let d2 = d1 - v * sqrt_t;
        let e1 = ((s / k).ln() + (r - q + 0.5 * v2) * t) / (v * sqrt_t);
        let e2 = e1 - v * sqrt_t;
        let term1 = s * qf * norm_cdf(e1) - k * df * norm_cdf(e2);
        let term2 = m * df * norm_cdf(-d2) - s * qf * norm_cdf(-d1);
        let term3 = s * qf * v2 / (2.0 * (r - q))
            * (-(s / m).powf(-2.0 * (r - q) / v2) * norm_cdf(d1 - 2.0 * (r - q) / v * sqrt_t)
                + (r * t).exp() * norm_cdf(d1));
        term1 + term2 + term3.max(0.0)
    } else {
        // OTM lookback call
        let e1 = ((s / k).ln() + (r - q + 0.5 * v2) * t) / (v * sqrt_t);
        let e2 = e1 - v * sqrt_t;
        let term1 = s * qf * norm_cdf(e1) - k * df * norm_cdf(e2);
        let term2 = s * qf * v2 / (2.0 * (r - q))
            * (-(s / k).powf(-2.0 * (r - q) / v2) * norm_cdf(e1 - 2.0 * (r - q) / v * sqrt_t)
                + (r * t).exp() * norm_cdf(e1));
        term1 + term2.max(0.0)
    }
}

/// Fixed-strike lookback put: payoff = max(K - S_min, 0)
pub fn lookback_fixed_put(opt: &LookbackFixed) -> f64 {
    let s = opt.spot;
    let k = opt.strike;
    let r = opt.rate;
    let q = opt.dividend;
    let v = opt.vol;
    let t = opt.time_to_expiry;
    let m = opt.s_min;

    if t <= 0.0 {
        return (k - m).max(0.0);
    }

    let v2 = v * v;
    let sqrt_t = t.sqrt();
    let df = (-r * t).exp();
    let qf = (-q * t).exp();

    if m <= k {
        let d1 = ((s / m).ln() + (r - q + 0.5 * v2) * t) / (v * sqrt_t);
        let d2 = d1 - v * sqrt_t;
        let e1 = ((s / k).ln() + (r - q + 0.5 * v2) * t) / (v * sqrt_t);
        let e2 = e1 - v * sqrt_t;
        let term1 = k * df * norm_cdf(-e2) - s * qf * norm_cdf(-e1);
        let term2 = s * qf * norm_cdf(d1) - m * df * norm_cdf(d2);
        let rq = r - q;
        let term3 = if rq.abs() > 1e-10 {
            s * qf * v2 / (2.0 * rq)
                * ((s / m).powf(-2.0 * rq / v2) * norm_cdf(-d1 + 2.0 * rq / v * sqrt_t)
                    - (r * t).exp() * norm_cdf(-d1))
        } else {
            s * qf * v * sqrt_t * (norm_pdf(d1) + d1 * norm_cdf(d1))
        };
        term1 - term2 + term3.max(0.0)
    } else {
        let e1 = ((s / k).ln() + (r - q + 0.5 * v2) * t) / (v * sqrt_t);
        let e2 = e1 - v * sqrt_t;
        let rq = r - q;
        let term1 = k * df * norm_cdf(-e2) - s * qf * norm_cdf(-e1);
        let term2 = if rq.abs() > 1e-10 {
            s * qf * v2 / (2.0 * rq)
                * ((s / k).powf(-2.0 * rq / v2) * norm_cdf(-e1 + 2.0 * rq / v * sqrt_t)
                    - (r * t).exp() * norm_cdf(-e1))
        } else {
            s * qf * v * sqrt_t * (norm_pdf(e1) - e1 * norm_cdf(-e1))
        };
        term1 + term2.max(0.0)
    }
}

/// Floating-strike lookback call: payoff = S_T - S_min
pub fn lookback_floating_call(
    spot: f64, s_min: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
) -> f64 {
    if tte <= 0.0 {
        return (spot - s_min).max(0.0);
    }
    let v2 = vol * vol;
    let sqrt_t = tte.sqrt();
    let df = (-rate * tte).exp();
    let qf = (-dividend * tte).exp();
    let rq = rate - dividend;

    let a1 = ((spot / s_min).ln() + (rq + 0.5 * v2) * tte) / (vol * sqrt_t);
    let a2 = a1 - vol * sqrt_t;

    if rq.abs() > 1e-10 {
        let term1 = spot * qf * norm_cdf(a1) - s_min * df * norm_cdf(a2);
        let term2 = spot * df * v2 / (2.0 * rq)
            * (-(spot / s_min).powf(-2.0 * rq / v2)
                * norm_cdf(-a1 + 2.0 * rq / vol * sqrt_t)
                + qf / df * norm_cdf(-a1));
        term1 - term2
    } else {
        let term1 = spot * qf * norm_cdf(a1) - s_min * df * norm_cdf(a2);
        let term2 = spot * df * vol * sqrt_t * (norm_pdf(a1) + a1 * norm_cdf(a1));
        term1 + term2
    }
}

/// Floating-strike lookback put: payoff = S_max - S_T
pub fn lookback_floating_put(
    spot: f64, s_max: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
) -> f64 {
    if tte <= 0.0 {
        return (s_max - spot).max(0.0);
    }
    let v2 = vol * vol;
    let sqrt_t = tte.sqrt();
    let df = (-rate * tte).exp();
    let qf = (-dividend * tte).exp();
    let rq = rate - dividend;

    let b1 = ((s_max / spot).ln() + (-rq + 0.5 * v2) * tte) / (vol * sqrt_t);
    let b2 = b1 - vol * sqrt_t;

    if rq.abs() > 1e-10 {
        let term1 = s_max * df * norm_cdf(b2) - spot * qf * norm_cdf(b1 - vol * sqrt_t) ;
        // Simplified: use symmetry argument
        let a1 = ((spot / s_max).ln() + (rq + 0.5 * v2) * tte) / (vol * sqrt_t);
        let term2 = spot * df * v2 / (2.0 * rq)
            * ((spot / s_max).powf(-2.0 * rq / v2)
                * norm_cdf(a1 - 2.0 * rq / vol * sqrt_t)
                - qf / df * norm_cdf(a1));
        s_max * df * norm_cdf(b2) - spot * qf * norm_cdf(b2 - vol * sqrt_t) + term2.abs()
    } else {
        s_max * df * norm_cdf(b2) - spot * qf * norm_cdf(b2) + spot * df * vol * sqrt_t * (norm_pdf(b1) + b1 * norm_cdf(b1))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ASIAN OPTIONS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsianAverageType {
    Geometric,
    Arithmetic,
}

#[derive(Debug, Clone)]
pub struct AsianOption {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub time_to_expiry: f64,
    pub opt_type: OptionType,
    pub avg_type: AsianAverageType,
    pub n_observations: usize,
    /// Running average so far (for partially elapsed options)
    pub running_avg: f64,
    pub observations_so_far: usize,
}

/// Geometric Asian option (closed form via Kemna-Vorst).
pub fn asian_geometric_price(opt: &AsianOption) -> f64 {
    let s = opt.spot;
    let k = opt.strike;
    let r = opt.rate;
    let q = opt.dividend;
    let v = opt.vol;
    let t = opt.time_to_expiry;
    let n = opt.n_observations as f64;

    if t <= 0.0 {
        let payoff = match opt.opt_type {
            OptionType::Call => (opt.running_avg - k).max(0.0),
            OptionType::Put => (k - opt.running_avg).max(0.0),
        };
        return payoff;
    }

    // Adjusted parameters for geometric average
    let adj_vol = v / 3.0_f64.sqrt();
    let adj_r = 0.5 * (r - q - v * v / 6.0);
    let adj_div = r - adj_r;

    let p = BSParams::new(s, k, r, adj_div, adj_vol, t);
    bs_price(&p, opt.opt_type)
}

/// Geometric Asian option with discrete monitoring (Levy).
pub fn asian_geometric_discrete(opt: &AsianOption) -> f64 {
    let s = opt.spot;
    let k = opt.strike;
    let r = opt.rate;
    let q = opt.dividend;
    let v = opt.vol;
    let t = opt.time_to_expiry;
    let n = opt.n_observations as f64;

    if n < 1.0 || t <= 0.0 {
        return 0.0;
    }

    let dt = t / n;
    // Mean of log of geometric average
    let mu_g = ((s).ln() + (r - q - 0.5 * v * v) * dt * (n + 1.0) / 2.0);
    // Variance of log of geometric average
    let v2_g = v * v * dt * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n);
    let sig_g = v2_g.sqrt();

    let d1 = (mu_g - k.ln() + v2_g) / sig_g;
    let d2 = d1 - sig_g;
    let df = (-r * t).exp();

    match opt.opt_type {
        OptionType::Call => df * ((mu_g + 0.5 * v2_g).exp() * norm_cdf(d1) - k * norm_cdf(d2)),
        OptionType::Put => df * (k * norm_cdf(-d2) - (mu_g + 0.5 * v2_g).exp() * norm_cdf(-d1)),
    }
}

/// Arithmetic Asian option via moment-matching (Turnbull-Wakeman).
pub fn asian_arithmetic_price(opt: &AsianOption) -> f64 {
    let s = opt.spot;
    let k = opt.strike;
    let r = opt.rate;
    let q = opt.dividend;
    let v = opt.vol;
    let t = opt.time_to_expiry;
    let n = opt.n_observations as f64;

    if t <= 0.0 {
        let payoff = match opt.opt_type {
            OptionType::Call => (opt.running_avg - k).max(0.0),
            OptionType::Put => (k - opt.running_avg).max(0.0),
        };
        return payoff;
    }

    // First moment (mean) of arithmetic average
    let rq = r - q;
    let m1 = if rq.abs() > 1e-10 {
        s * ((rq * t).exp() - 1.0) / (rq * t)
    } else {
        s * (1.0 + 0.5 * rq * t)
    };

    // Second moment of arithmetic average
    let m2 = compute_asian_second_moment(s, r, q, v, t, n);

    // Lognormal approximation parameters
    let var_a = (m2 / (m1 * m1)).ln();
    if var_a <= 0.0 {
        // Fallback to geometric
        return asian_geometric_price(opt);
    }
    let vol_a = (var_a / t).sqrt();

    let p = BSParams::new(m1, k, r, 0.0, vol_a, t);
    let df = (-r * t).exp();
    match opt.opt_type {
        OptionType::Call => {
            let d1 = ((m1 / k).ln() + 0.5 * var_a) / var_a.sqrt();
            let d2 = d1 - var_a.sqrt();
            df * (m1 * norm_cdf(d1) - k * norm_cdf(d2))
        }
        OptionType::Put => {
            let d1 = ((m1 / k).ln() + 0.5 * var_a) / var_a.sqrt();
            let d2 = d1 - var_a.sqrt();
            df * (k * norm_cdf(-d2) - m1 * norm_cdf(-d1))
        }
    }
}

fn compute_asian_second_moment(s: f64, r: f64, q: f64, v: f64, t: f64, n: f64) -> f64 {
    let dt = t / n;
    let rq = r - q;
    let v2 = v * v;
    let mut m2 = 0.0;
    for i in 0..n as usize {
        for j in 0..=i {
            let ti = (i as f64 + 1.0) * dt;
            let tj = (j as f64 + 1.0) * dt;
            let t_min = tj.min(ti);
            m2 += s * s * (rq * (ti + tj) + v2 * t_min).exp();
        }
    }
    2.0 * m2 / (n * n) - {
        let mut diag = 0.0;
        for i in 0..n as usize {
            let ti = (i as f64 + 1.0) * dt;
            diag += s * s * (2.0 * rq * ti + v2 * ti).exp();
        }
        diag / (n * n)
    }
}

/// Asian option delta via bump.
pub fn asian_delta(opt: &AsianOption, ds: f64) -> f64 {
    let mut up = opt.clone();
    up.spot += ds;
    let mut dn = opt.clone();
    dn.spot -= ds;
    let price_fn = match opt.avg_type {
        AsianAverageType::Geometric => asian_geometric_price,
        AsianAverageType::Arithmetic => asian_arithmetic_price,
    };
    (price_fn(&up) - price_fn(&dn)) / (2.0 * ds)
}

/// Control variate Asian pricing: use geometric as control.
pub fn asian_arithmetic_cv(opt: &AsianOption, n_sims: usize, seed: u64) -> f64 {
    let geo_analytical = asian_geometric_price(opt);
    let dt = opt.time_to_expiry / opt.n_observations as f64;
    let drift = (opt.rate - opt.dividend - 0.5 * opt.vol * opt.vol) * dt;
    let vol_sqrt_dt = opt.vol * dt.sqrt();

    let mut rng_state = seed;
    let mut sum_arith = 0.0;
    let mut sum_geo = 0.0;

    for _ in 0..n_sims {
        let mut s = opt.spot;
        let mut sum_s = 0.0;
        let mut prod_s = 0.0;

        for _ in 0..opt.n_observations {
            rng_state = lcg_next(rng_state);
            let u1 = rng_state as f64 / u64::MAX as f64;
            rng_state = lcg_next(rng_state);
            let u2 = rng_state as f64 / u64::MAX as f64;
            let z = box_muller(u1.max(1e-15), u2);
            s *= (drift + vol_sqrt_dt * z).exp();
            sum_s += s;
            prod_s += s.ln();
        }

        let arith_avg = sum_s / opt.n_observations as f64;
        let geo_avg = (prod_s / opt.n_observations as f64).exp();

        let payoff_a = match opt.opt_type {
            OptionType::Call => (arith_avg - opt.strike).max(0.0),
            OptionType::Put => (opt.strike - arith_avg).max(0.0),
        };
        let payoff_g = match opt.opt_type {
            OptionType::Call => (geo_avg - opt.strike).max(0.0),
            OptionType::Put => (opt.strike - geo_avg).max(0.0),
        };

        sum_arith += payoff_a;
        sum_geo += payoff_g;
    }

    let df = (-opt.rate * opt.time_to_expiry).exp();
    let mc_arith = df * sum_arith / n_sims as f64;
    let mc_geo = df * sum_geo / n_sims as f64;

    // Control variate adjustment
    mc_arith + (geo_analytical - mc_geo)
}

fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ═══════════════════════════════════════════════════════════════════════════
// CHOOSER OPTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Simple chooser option: holder chooses call or put at time t_choose.
/// Rubinstein (1991) formula.
pub fn chooser_option(
    spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64,
    t_choose: f64, t_expiry: f64,
) -> f64 {
    let t = t_expiry;
    let tc = t_choose;
    let sqrt_tc = tc.sqrt();
    let sqrt_t = t.sqrt();

    let d = ((spot / strike).ln() + (rate - dividend + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    let y = ((spot / strike).ln() + (rate - dividend) * t + 0.5 * vol * vol * tc) / (vol * sqrt_tc);

    let qf = (-dividend * t).exp();
    let df = (-rate * t).exp();
    let qf_c = (-dividend * (t - tc)).exp();
    let df_c = (-rate * (t - tc)).exp();

    // Chooser = Call(S, K, T) + Put(S, K*exp(-(r-q)(T-tc)), tc)
    let p = BSParams::new(spot, strike, rate, dividend, vol, t);
    let call_val = crate::black_scholes::bs_call_price(&p);

    // Or equivalently via the direct formula:
    let term1 = spot * qf * norm_cdf(d) - strike * df * norm_cdf(d - vol * sqrt_t);
    let term2 = strike * df * norm_cdf(-y + vol * sqrt_tc) - spot * qf * norm_cdf(-y);
    // But term1 is just the call. We add the "extra" put value from choosing.
    // Direct: chooser = call(T) + e^{-(r-q)(T-tc)} * put(tc, adjusted)
    call_val + spot * (-dividend * t).exp() * norm_cdf(-y)
        - strike * (-rate * t).exp() * norm_cdf(-y + vol * sqrt_tc)
        - (spot * (-dividend * t).exp() * norm_cdf(-d)
            - strike * (-rate * t).exp() * norm_cdf(-d + vol * sqrt_t))
}

/// Complex chooser: different strikes/expiries for call and put legs.
pub fn complex_chooser(
    spot: f64,
    k_call: f64,
    k_put: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    t_choose: f64,
    t_call: f64,
    t_put: f64,
) -> f64 {
    // At t_choose, holder picks max(Call(S, K_c, T_c - t_c), Put(S, K_p, T_p - t_c))
    // Use critical spot S* where Call = Put, then bivariate normal
    let s_star = find_chooser_critical_spot(
        spot, k_call, k_put, rate, dividend, vol, t_choose, t_call, t_put,
    );

    let sqrt_tc = t_choose.sqrt();
    let sqrt_t_call = t_call.sqrt();
    let sqrt_t_put = t_put.sqrt();
    let rho_c = (t_choose / t_call).sqrt();
    let rho_p = (t_choose / t_put).sqrt();

    let d1 = ((spot / s_star).ln() + (rate - dividend + 0.5 * vol * vol) * t_choose) / (vol * sqrt_tc);
    let d2 = d1 - vol * sqrt_tc;

    let y1_c = ((spot / k_call).ln() + (rate - dividend + 0.5 * vol * vol) * t_call) / (vol * sqrt_t_call);
    let y2_c = y1_c - vol * sqrt_t_call;

    let y1_p = ((spot / k_put).ln() + (rate - dividend + 0.5 * vol * vol) * t_put) / (vol * sqrt_t_put);
    let y2_p = y1_p - vol * sqrt_t_put;

    let qf_c = (-dividend * t_call).exp();
    let df_c = (-rate * t_call).exp();
    let qf_p = (-dividend * t_put).exp();
    let df_p = (-rate * t_put).exp();

    // Bivariate normal CDF approximation
    let call_part = spot * qf_c * bivariate_normal_cdf(d1, y1_c, rho_c)
        - k_call * df_c * bivariate_normal_cdf(d2, y2_c, rho_c);
    let put_part = k_put * df_p * bivariate_normal_cdf(-d2, -y2_p, rho_p)
        - spot * qf_p * bivariate_normal_cdf(-d1, -y1_p, rho_p);

    call_part + put_part
}

fn find_chooser_critical_spot(
    spot: f64, k_call: f64, k_put: f64, rate: f64, dividend: f64, vol: f64,
    t_choose: f64, t_call: f64, t_put: f64,
) -> f64 {
    // Newton's method to find S* where Call(S*, K_c, T_c-t_c) = Put(S*, K_p, T_p-t_c)
    let mut s = spot;
    for _ in 0..50 {
        let tau_c = t_call - t_choose;
        let tau_p = t_put - t_choose;
        let pc = BSParams::new(s, k_call, rate, dividend, vol, tau_c);
        let pp = BSParams::new(s, k_put, rate, dividend, vol, tau_p);
        let call_v = crate::black_scholes::bs_call_price(&pc);
        let put_v = crate::black_scholes::bs_put_price(&pp);
        let diff = call_v - put_v;
        if diff.abs() < 1e-10 {
            break;
        }
        let dc = crate::black_scholes::delta(&pc, OptionType::Call);
        let dp = crate::black_scholes::delta(&pp, OptionType::Put);
        let deriv = dc - dp;
        if deriv.abs() < 1e-15 {
            break;
        }
        s -= diff / deriv;
        s = s.max(0.01);
    }
    s
}

/// Bivariate standard normal CDF approximation (Drezner-Wesolowsky).
pub fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    if rho.abs() < 1e-12 {
        return norm_cdf(x) * norm_cdf(y);
    }
    if (rho - 1.0).abs() < 1e-12 {
        return norm_cdf(x.min(y));
    }
    if (rho + 1.0).abs() < 1e-12 {
        return (norm_cdf(x) - norm_cdf(-y)).max(0.0);
    }

    // Gauss-Legendre quadrature approximation
    let weights = [0.1713244923791703, 0.3607615730481386, 0.4679139345726910];
    let abscissae = [0.9324695142031521, 0.6612093864662645, 0.2386191860831969];

    let a = if rho >= 0.0 { 0.0 } else { -1.0 };
    let b = (1.0 - rho * rho).sqrt();

    let mut sum = 0.0;
    for i in 0..3 {
        for sign in &[-1.0, 1.0] {
            let si = (abscissae[i] * sign + 1.0) / 2.0;
            let r = a + b * si;
            let rr = 1.0 - r * r;
            if rr > 0.0 {
                let rr_sqrt = rr.sqrt();
                let val = (-(x * x - 2.0 * r * x * y + y * y) / (2.0 * rr)).exp() / (2.0 * PI * rr_sqrt);
                sum += weights[i] * val;
            }
        }
    }
    sum *= b / 2.0;

    if rho >= 0.0 {
        sum + norm_cdf(x).min(norm_cdf(y))
    } else {
        let base = (norm_cdf(x) + norm_cdf(y) - 1.0).max(0.0);
        sum + base
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPOUND OPTIONS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundType {
    CallOnCall,
    CallOnPut,
    PutOnCall,
    PutOnPut,
}

#[derive(Debug, Clone, Copy)]
pub struct CompoundOption {
    pub spot: f64,
    pub strike_outer: f64,  // strike of the compound option
    pub strike_inner: f64,  // strike of the underlying option
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub t_outer: f64,       // expiry of compound option
    pub t_inner: f64,       // expiry of underlying option
    pub compound_type: CompoundType,
}

/// Geske (1979) compound option pricing.
pub fn compound_option_price(opt: &CompoundOption) -> f64 {
    let s = opt.spot;
    let k1 = opt.strike_outer;
    let k2 = opt.strike_inner;
    let r = opt.rate;
    let q = opt.dividend;
    let v = opt.vol;
    let t1 = opt.t_outer;
    let t2 = opt.t_inner;

    if t1 <= 0.0 || t2 <= t1 {
        return 0.0;
    }

    // Find critical underlying option price at t1
    let s_star = find_compound_critical_spot(k1, k2, r, q, v, t1, t2, opt.compound_type);

    let sqrt_t1 = t1.sqrt();
    let sqrt_t2 = t2.sqrt();
    let rho = (t1 / t2).sqrt();
    let qf1 = (-q * t1).exp();
    let qf2 = (-q * t2).exp();
    let df1 = (-r * t1).exp();
    let df2 = (-r * t2).exp();

    let a1 = ((s / s_star).ln() + (r - q + 0.5 * v * v) * t1) / (v * sqrt_t1);
    let a2 = a1 - v * sqrt_t1;
    let b1 = ((s / k2).ln() + (r - q + 0.5 * v * v) * t2) / (v * sqrt_t2);
    let b2 = b1 - v * sqrt_t2;

    match opt.compound_type {
        CompoundType::CallOnCall => {
            s * qf2 * bivariate_normal_cdf(a1, b1, rho)
                - k2 * df2 * bivariate_normal_cdf(a2, b2, rho)
                - k1 * df1 * norm_cdf(a2)
        }
        CompoundType::CallOnPut => {
            -s * qf2 * bivariate_normal_cdf(-a1, -b1, rho)
                + k2 * df2 * bivariate_normal_cdf(-a2, -b2, rho)
                - k1 * df1 * norm_cdf(-a2)
        }
        CompoundType::PutOnCall => {
            -s * qf2 * bivariate_normal_cdf(-a1, b1, -rho)
                + k2 * df2 * bivariate_normal_cdf(-a2, b2, -rho)
                + k1 * df1 * norm_cdf(-a2)
        }
        CompoundType::PutOnPut => {
            s * qf2 * bivariate_normal_cdf(a1, -b1, -rho)
                - k2 * df2 * bivariate_normal_cdf(a2, -b2, -rho)
                + k1 * df1 * norm_cdf(a2)
        }
    }
}

fn find_compound_critical_spot(
    k1: f64, k2: f64, r: f64, q: f64, v: f64, t1: f64, t2: f64,
    compound_type: CompoundType,
) -> f64 {
    // Find S* where the underlying option value = k1
    let tau = t2 - t1;
    let inner_type = match compound_type {
        CompoundType::CallOnCall | CompoundType::PutOnCall => OptionType::Call,
        CompoundType::CallOnPut | CompoundType::PutOnPut => OptionType::Put,
    };

    let mut s = k2; // initial guess
    for _ in 0..100 {
        let p = BSParams::new(s, k2, r, q, v, tau);
        let val = crate::black_scholes::bs_price(&p, inner_type);
        let diff = val - k1;
        if diff.abs() < 1e-10 {
            break;
        }
        let d = crate::black_scholes::delta(&p, inner_type);
        if d.abs() < 1e-15 {
            break;
        }
        s -= diff / d;
        s = s.max(0.01);
    }
    s
}

// ═══════════════════════════════════════════════════════════════════════════
// EXCHANGE OPTIONS (MARGRABE)
// ═══════════════════════════════════════════════════════════════════════════

/// Margrabe's formula: option to exchange asset 2 for asset 1.
/// Payoff = max(S1 - S2, 0)
pub fn margrabe_exchange(
    s1: f64, s2: f64, q1: f64, q2: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    if tte <= 0.0 {
        return (s1 - s2).max(0.0);
    }
    let vol = (vol1 * vol1 + vol2 * vol2 - 2.0 * corr * vol1 * vol2).sqrt();
    let sqrt_t = tte.sqrt();
    let d1 = ((s1 / s2).ln() + (q2 - q1 + 0.5 * vol * vol) * tte) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    s1 * (-q1 * tte).exp() * norm_cdf(d1) - s2 * (-q2 * tte).exp() * norm_cdf(d2)
}

/// Exchange option delta w.r.t. S1.
pub fn margrabe_delta_s1(
    s1: f64, s2: f64, q1: f64, q2: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    let vol = (vol1 * vol1 + vol2 * vol2 - 2.0 * corr * vol1 * vol2).sqrt();
    let sqrt_t = tte.sqrt();
    let d1 = ((s1 / s2).ln() + (q2 - q1 + 0.5 * vol * vol) * tte) / (vol * sqrt_t);
    (-q1 * tte).exp() * norm_cdf(d1)
}

/// Exchange option delta w.r.t. S2.
pub fn margrabe_delta_s2(
    s1: f64, s2: f64, q1: f64, q2: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    let vol = (vol1 * vol1 + vol2 * vol2 - 2.0 * corr * vol1 * vol2).sqrt();
    let sqrt_t = tte.sqrt();
    let d1 = ((s1 / s2).ln() + (q2 - q1 + 0.5 * vol * vol) * tte) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    -(-q2 * tte).exp() * norm_cdf(d2)
}

/// Exchange option gamma (cross-gamma d²V/(dS1 dS2)).
pub fn margrabe_cross_gamma(
    s1: f64, s2: f64, q1: f64, q2: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    let vol = (vol1 * vol1 + vol2 * vol2 - 2.0 * corr * vol1 * vol2).sqrt();
    let sqrt_t = tte.sqrt();
    let d1 = ((s1 / s2).ln() + (q2 - q1 + 0.5 * vol * vol) * tte) / (vol * sqrt_t);
    -(-q1 * tte).exp() * norm_pdf(d1) / (s2 * vol * sqrt_t)
}

// ═══════════════════════════════════════════════════════════════════════════
// SPREAD OPTIONS (KIRK'S APPROXIMATION)
// ═══════════════════════════════════════════════════════════════════════════

/// Kirk's approximation for spread option: max(S1 - S2 - K, 0)
pub fn kirk_spread_call(
    s1: f64, s2: f64, strike: f64, rate: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    if tte <= 0.0 {
        return (s1 - s2 - strike).max(0.0);
    }
    let df = (-rate * tte).exp();
    let f1 = s1 / df;
    let f2 = s2 / df;
    let k = strike;

    // Kirk's adjusted vol
    let ratio = f2 / (f2 + k);
    let vol_kirk = (vol1 * vol1 - 2.0 * corr * vol1 * vol2 * ratio + vol2 * vol2 * ratio * ratio).sqrt();

    let sqrt_t = tte.sqrt();
    let d1 = ((f1 / (f2 + k)).ln() + 0.5 * vol_kirk * vol_kirk * tte) / (vol_kirk * sqrt_t);
    let d2 = d1 - vol_kirk * sqrt_t;

    df * (f1 * norm_cdf(d1) - (f2 + k) * norm_cdf(d2))
}

/// Kirk's spread put: max(K + S2 - S1, 0)
pub fn kirk_spread_put(
    s1: f64, s2: f64, strike: f64, rate: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    if tte <= 0.0 {
        return (strike + s2 - s1).max(0.0);
    }
    let call = kirk_spread_call(s1, s2, strike, rate, vol1, vol2, corr, tte);
    let df = (-rate * tte).exp();
    // Put-call parity for spreads
    call - df * (s1 / df - s2 / df - strike)
}

/// Spread option delta w.r.t. S1 via bump.
pub fn kirk_spread_delta_s1(
    s1: f64, s2: f64, strike: f64, rate: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64, ds: f64,
) -> f64 {
    let up = kirk_spread_call(s1 + ds, s2, strike, rate, vol1, vol2, corr, tte);
    let dn = kirk_spread_call(s1 - ds, s2, strike, rate, vol1, vol2, corr, tte);
    (up - dn) / (2.0 * ds)
}

/// Spread option delta w.r.t. S2 via bump.
pub fn kirk_spread_delta_s2(
    s1: f64, s2: f64, strike: f64, rate: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64, ds: f64,
) -> f64 {
    let up = kirk_spread_call(s1, s2 + ds, strike, rate, vol1, vol2, corr, tte);
    let dn = kirk_spread_call(s1, s2 - ds, strike, rate, vol1, vol2, corr, tte);
    (up - dn) / (2.0 * ds)
}

// ═══════════════════════════════════════════════════════════════════════════
// BASKET OPTIONS (GENTLE'S METHOD / MOMENT MATCHING)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct BasketOption {
    pub spots: Vec<f64>,
    pub weights: Vec<f64>,
    pub strike: f64,
    pub rate: f64,
    pub dividends: Vec<f64>,
    pub vols: Vec<f64>,
    pub corr_matrix: Vec<Vec<f64>>,
    pub time_to_expiry: f64,
    pub opt_type: OptionType,
}

/// Gentle's geometric conditioning for basket options.
pub fn basket_gentle(opt: &BasketOption) -> f64 {
    let n = opt.spots.len();
    let t = opt.time_to_expiry;
    let r = opt.rate;

    if t <= 0.0 || n == 0 {
        let basket_val: f64 = opt.spots.iter().zip(opt.weights.iter()).map(|(s, w)| s * w).sum();
        return match opt.opt_type {
            OptionType::Call => (basket_val - opt.strike).max(0.0),
            OptionType::Put => (opt.strike - basket_val).max(0.0),
        };
    }

    // Forward prices
    let forwards: Vec<f64> = (0..n)
        .map(|i| opt.spots[i] * ((r - opt.dividends[i]) * t).exp())
        .collect();

    // First moment of basket
    let m1: f64 = (0..n).map(|i| opt.weights[i] * forwards[i]).sum();

    // Second moment of basket
    let mut m2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let cov_ij = opt.corr_matrix[i][j] * opt.vols[i] * opt.vols[j] * t;
            m2 += opt.weights[i] * opt.weights[j] * forwards[i] * forwards[j] * cov_ij.exp();
        }
    }

    // Lognormal approximation
    let var_log = (m2 / (m1 * m1)).ln();
    if var_log <= 0.0 {
        let df = (-r * t).exp();
        return match opt.opt_type {
            OptionType::Call => df * (m1 - opt.strike).max(0.0),
            OptionType::Put => df * (opt.strike - m1).max(0.0),
        };
    }

    let vol_basket = (var_log / t).sqrt();
    let df = (-r * t).exp();
    let sqrt_t = t.sqrt();

    let d1 = ((m1 / opt.strike).ln() + 0.5 * var_log) / var_log.sqrt();
    let d2 = d1 - var_log.sqrt();

    match opt.opt_type {
        OptionType::Call => df * (m1 * norm_cdf(d1) - opt.strike * norm_cdf(d2)),
        OptionType::Put => df * (opt.strike * norm_cdf(-d2) - m1 * norm_cdf(-d1)),
    }
}

/// Basket option via Monte Carlo with antithetic variates.
pub fn basket_mc(opt: &BasketOption, n_sims: usize, seed: u64) -> f64 {
    let n = opt.spots.len();
    let t = opt.time_to_expiry;
    let r = opt.rate;
    let df = (-r * t).exp();

    // Cholesky decomposition of correlation matrix
    let chol = cholesky_decompose(&opt.corr_matrix);

    let mut rng_state = seed;
    let mut sum = 0.0;

    for _ in 0..n_sims {
        // Generate correlated normals
        let mut z = vec![0.0; n];
        for i in 0..n {
            rng_state = lcg_next(rng_state);
            let u1 = (rng_state as f64 / u64::MAX as f64).max(1e-15);
            rng_state = lcg_next(rng_state);
            let u2 = rng_state as f64 / u64::MAX as f64;
            z[i] = box_muller(u1, u2);
        }

        let mut corr_z = vec![0.0; n];
        for i in 0..n {
            for j in 0..=i {
                corr_z[i] += chol[i][j] * z[j];
            }
        }

        // Simulate terminal prices and compute basket value
        let mut basket_val = 0.0;
        let mut basket_val_anti = 0.0;
        for i in 0..n {
            let drift = (r - opt.dividends[i] - 0.5 * opt.vols[i] * opt.vols[i]) * t;
            let diffusion = opt.vols[i] * t.sqrt() * corr_z[i];
            let s_t = opt.spots[i] * (drift + diffusion).exp();
            let s_t_anti = opt.spots[i] * (drift - diffusion).exp();
            basket_val += opt.weights[i] * s_t;
            basket_val_anti += opt.weights[i] * s_t_anti;
        }

        let payoff = match opt.opt_type {
            OptionType::Call => (basket_val - opt.strike).max(0.0),
            OptionType::Put => (opt.strike - basket_val).max(0.0),
        };
        let payoff_anti = match opt.opt_type {
            OptionType::Call => (basket_val_anti - opt.strike).max(0.0),
            OptionType::Put => (opt.strike - basket_val_anti).max(0.0),
        };

        sum += 0.5 * (payoff + payoff_anti);
    }

    df * sum / n_sims as f64
}

fn cholesky_decompose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let val = matrix[i][i] - sum;
                l[i][j] = if val > 0.0 { val.sqrt() } else { 0.0 };
            } else {
                l[i][j] = if l[j][j].abs() > 1e-15 {
                    (matrix[i][j] - sum) / l[j][j]
                } else {
                    0.0
                };
            }
        }
    }
    l
}

// ═══════════════════════════════════════════════════════════════════════════
// RAINBOW OPTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Best-of-two call: max(S1, S2) - K
pub fn rainbow_best_of_two_call(
    s1: f64, s2: f64, strike: f64,
    r: f64, q1: f64, q2: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    if tte <= 0.0 {
        return (s1.max(s2) - strike).max(0.0);
    }

    // max(S1, S2) = S2 + max(S1 - S2, 0)
    // Price = Call(S2, K) + Exchange(S1, S2) - part where S1>S2 and S1<K
    // Use Stulz (1982) formula
    let vol = (vol1 * vol1 + vol2 * vol2 - 2.0 * corr * vol1 * vol2).sqrt();
    let sqrt_t = tte.sqrt();
    let df = (-r * tte).exp();
    let qf1 = (-q1 * tte).exp();
    let qf2 = (-q2 * tte).exp();

    let rho1 = (vol1 - corr * vol2) / vol;
    let rho2 = (vol2 - corr * vol1) / vol;

    let d = ((s1 / s2).ln() + (q2 - q1 + 0.5 * vol * vol) * tte) / (vol * sqrt_t);

    let y1 = ((s1 / strike).ln() + (r - q1 + 0.5 * vol1 * vol1) * tte) / (vol1 * sqrt_t);
    let y2 = ((s2 / strike).ln() + (r - q2 + 0.5 * vol2 * vol2) * tte) / (vol2 * sqrt_t);

    s1 * qf1 * bivariate_normal_cdf(y1, d, -rho1)
        + s2 * qf2 * bivariate_normal_cdf(y2, -d + vol * sqrt_t, -rho2)
        - strike * df * (1.0
            - bivariate_normal_cdf(-y1 + vol1 * sqrt_t, -y2 + vol2 * sqrt_t, corr))
}

/// Worst-of-two call: min(S1, S2) - K
pub fn rainbow_worst_of_two_call(
    s1: f64, s2: f64, strike: f64,
    r: f64, q1: f64, q2: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    if tte <= 0.0 {
        return (s1.min(s2) - strike).max(0.0);
    }

    // min(S1, S2) = S1 + S2 - max(S1, S2)
    // Call on min = Call on S1 + Call on S2 - Call on max
    let p1 = BSParams::new(s1, strike, r, q1, vol1, tte);
    let p2 = BSParams::new(s2, strike, r, q2, vol2, tte);
    let call1 = crate::black_scholes::bs_call_price(&p1);
    let call2 = crate::black_scholes::bs_call_price(&p2);
    let best_of = rainbow_best_of_two_call(s1, s2, strike, r, q1, q2, vol1, vol2, corr, tte);

    call1 + call2 - best_of
}

/// Best-of-two put: K - min(S1, S2)
pub fn rainbow_best_of_two_put(
    s1: f64, s2: f64, strike: f64,
    r: f64, q1: f64, q2: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    if tte <= 0.0 {
        return (strike - s1.min(s2)).max(0.0);
    }

    let p1 = BSParams::new(s1, strike, r, q1, vol1, tte);
    let p2 = BSParams::new(s2, strike, r, q2, vol2, tte);
    let put1 = crate::black_scholes::bs_put_price(&p1);
    let put2 = crate::black_scholes::bs_put_price(&p2);

    // Put on max = Put1 + Put2 - Put on min
    // By put-call parity for rainbow...
    let vol = (vol1 * vol1 + vol2 * vol2 - 2.0 * corr * vol1 * vol2).sqrt();
    let sqrt_t = tte.sqrt();
    let df = (-r * tte).exp();
    let qf1 = (-q1 * tte).exp();
    let qf2 = (-q2 * tte).exp();

    let d = ((s1 / s2).ln() + (q2 - q1 + 0.5 * vol * vol) * tte) / (vol * sqrt_t);
    let y1 = ((s1 / strike).ln() + (r - q1 + 0.5 * vol1 * vol1) * tte) / (vol1 * sqrt_t);
    let y2 = ((s2 / strike).ln() + (r - q2 + 0.5 * vol2 * vol2) * tte) / (vol2 * sqrt_t);

    // Put on max(S1,S2): K - max(S1,S2) when max < K
    strike * df * bivariate_normal_cdf(-y1 + vol1 * sqrt_t, -y2 + vol2 * sqrt_t, corr)
        - s1 * qf1 * bivariate_normal_cdf(-y1, -d, -(-vol1 + corr * vol2) / vol)
        - s2 * qf2 * bivariate_normal_cdf(-y2, d - vol * sqrt_t, -(- vol2 + corr * vol1) / vol)
}

/// Worst-of-two put: K - max(S1, S2)
pub fn rainbow_worst_of_two_put(
    s1: f64, s2: f64, strike: f64,
    r: f64, q1: f64, q2: f64,
    vol1: f64, vol2: f64, corr: f64, tte: f64,
) -> f64 {
    if tte <= 0.0 {
        return (strike - s1.max(s2)).max(0.0);
    }

    let p1 = BSParams::new(s1, strike, r, q1, vol1, tte);
    let p2 = BSParams::new(s2, strike, r, q2, vol2, tte);
    let put1 = crate::black_scholes::bs_put_price(&p1);
    let put2 = crate::black_scholes::bs_put_price(&p2);
    let best_put = rainbow_best_of_two_put(s1, s2, strike, r, q1, q2, vol1, vol2, corr, tte);
    put1 + put2 - best_put
}

/// Rainbow option: max(S1, S2, ..., Sn) via MC simulation.
pub fn rainbow_best_of_n_call_mc(
    spots: &[f64],
    strike: f64,
    rate: f64,
    dividends: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    tte: f64,
    n_sims: usize,
    seed: u64,
) -> f64 {
    let n = spots.len();
    let df = (-rate * tte).exp();
    let chol = cholesky_decompose(corr_matrix);
    let mut rng_state = seed;
    let mut sum = 0.0;

    for _ in 0..n_sims {
        let mut z = vec![0.0; n];
        for i in 0..n {
            rng_state = lcg_next(rng_state);
            let u1 = (rng_state as f64 / u64::MAX as f64).max(1e-15);
            rng_state = lcg_next(rng_state);
            let u2 = rng_state as f64 / u64::MAX as f64;
            z[i] = box_muller(u1, u2);
        }
        let mut corr_z = vec![0.0; n];
        for i in 0..n {
            for j in 0..=i {
                corr_z[i] += chol[i][j] * z[j];
            }
        }
        let mut best = f64::NEG_INFINITY;
        for i in 0..n {
            let drift = (rate - dividends[i] - 0.5 * vols[i] * vols[i]) * tte;
            let s_t = spots[i] * (drift + vols[i] * tte.sqrt() * corr_z[i]).exp();
            if s_t > best {
                best = s_t;
            }
        }
        sum += (best - strike).max(0.0);
    }
    df * sum / n_sims as f64
}

/// Rainbow option: min(S1, S2, ..., Sn) via MC simulation.
pub fn rainbow_worst_of_n_call_mc(
    spots: &[f64],
    strike: f64,
    rate: f64,
    dividends: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    tte: f64,
    n_sims: usize,
    seed: u64,
) -> f64 {
    let n = spots.len();
    let df = (-rate * tte).exp();
    let chol = cholesky_decompose(corr_matrix);
    let mut rng_state = seed;
    let mut sum = 0.0;

    for _ in 0..n_sims {
        let mut z = vec![0.0; n];
        for i in 0..n {
            rng_state = lcg_next(rng_state);
            let u1 = (rng_state as f64 / u64::MAX as f64).max(1e-15);
            rng_state = lcg_next(rng_state);
            let u2 = rng_state as f64 / u64::MAX as f64;
            z[i] = box_muller(u1, u2);
        }
        let mut corr_z = vec![0.0; n];
        for i in 0..n {
            for j in 0..=i {
                corr_z[i] += chol[i][j] * z[j];
            }
        }
        let mut worst = f64::INFINITY;
        for i in 0..n {
            let drift = (rate - dividends[i] - 0.5 * vols[i] * vols[i]) * tte;
            let s_t = spots[i] * (drift + vols[i] * tte.sqrt() * corr_z[i]).exp();
            if s_t < worst {
                worst = s_t;
            }
        }
        sum += (worst - strike).max(0.0);
    }
    df * sum / n_sims as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// CLIQUET / RATCHET OPTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Cliquet option: sum of capped/floored forward-starting returns.
pub fn cliquet_price(
    spot: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    reset_times: &[f64],
    local_cap: f64,
    local_floor: f64,
    global_cap: f64,
    global_floor: f64,
    n_sims: usize,
    seed: u64,
) -> f64 {
    let n = reset_times.len();
    if n < 2 {
        return 0.0;
    }

    let tte = *reset_times.last().unwrap();
    let df = (-rate * tte).exp();
    let mut rng_state = seed;
    let mut sum = 0.0;

    for _ in 0..n_sims {
        let mut s = spot;
        let mut total_return = 0.0;

        for i in 0..n - 1 {
            let dt = reset_times[i + 1] - reset_times[i];
            let drift = (rate - dividend - 0.5 * vol * vol) * dt;
            let vol_sqrt = vol * dt.sqrt();

            rng_state = lcg_next(rng_state);
            let u1 = (rng_state as f64 / u64::MAX as f64).max(1e-15);
            rng_state = lcg_next(rng_state);
            let u2 = rng_state as f64 / u64::MAX as f64;
            let z = box_muller(u1, u2);

            let s_new = s * (drift + vol_sqrt * z).exp();
            let period_return = (s_new / s - 1.0).max(local_floor).min(local_cap);
            total_return += period_return;
            s = s_new;
        }

        let capped_return = total_return.max(global_floor).min(global_cap);
        sum += capped_return.max(0.0);
    }

    df * spot * sum / n_sims as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// QUANTO OPTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Quanto option: foreign asset, domestic settlement, fixed FX rate.
pub fn quanto_option(
    spot_foreign: f64,
    strike: f64,
    rate_dom: f64,
    rate_for: f64,
    vol_asset: f64,
    vol_fx: f64,
    corr_asset_fx: f64,
    fx_rate: f64,
    tte: f64,
    opt_type: OptionType,
) -> f64 {
    let quanto_adj = corr_asset_fx * vol_asset * vol_fx;
    let adj_div = rate_for + quanto_adj;
    let p = BSParams::new(spot_foreign, strike, rate_dom, adj_div, vol_asset, tte);
    fx_rate * crate::black_scholes::bs_price(&p, opt_type)
}

// ═══════════════════════════════════════════════════════════════════════════
// BINARY BARRIER OPTIONS (ONE-TOUCH / NO-TOUCH)
// ═══════════════════════════════════════════════════════════════════════════

/// One-touch option: pays 1 if barrier is hit before expiry.
pub fn one_touch(
    spot: f64, barrier: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    is_up: bool,
) -> f64 {
    if tte <= 0.0 {
        return 0.0;
    }
    let mu = (rate - dividend - 0.5 * vol * vol) / (vol * vol);
    let lambda = (mu * mu + 2.0 * rate / (vol * vol)).sqrt();
    let sqrt_t = tte.sqrt();

    let h_s = if is_up { barrier / spot } else { spot / barrier };
    if h_s <= 0.0 {
        return 0.0;
    }
    let x = h_s.ln() / (vol * sqrt_t);

    if is_up && spot >= barrier {
        return (-rate * tte).exp();
    }
    if !is_up && spot <= barrier {
        return (-rate * tte).exp();
    }

    let sign = if is_up { -1.0 } else { 1.0 };
    let log_hs = (barrier / spot).ln();

    let term1 = (barrier / spot).powf(mu + lambda) * norm_cdf(sign * (-log_hs / (vol * sqrt_t) - lambda * vol * sqrt_t));
    let term2 = (barrier / spot).powf(mu - lambda) * norm_cdf(sign * (-log_hs / (vol * sqrt_t) + lambda * vol * sqrt_t));

    (term1 + term2).min(1.0).max(0.0)
}

/// No-touch: 1 - one_touch
pub fn no_touch(
    spot: f64, barrier: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    is_up: bool,
) -> f64 {
    let df = (-rate * tte).exp();
    df - one_touch(spot, barrier, rate, dividend, vol, tte, is_up)
}

/// Double no-touch: spot must stay between lower and upper barriers.
pub fn double_no_touch(
    spot: f64, lower: f64, upper: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    n_terms: usize,
) -> f64 {
    if spot <= lower || spot >= upper {
        return 0.0;
    }
    let df = (-rate * tte).exp();
    let log_range = (upper / lower).ln();
    let log_spot = (spot / lower).ln();
    let mu = (rate - dividend - 0.5 * vol * vol) / (vol * vol);

    let mut sum = 0.0;
    for n in 1..=n_terms {
        let nf = n as f64;
        let beta = nf * PI / log_range;
        let sin_term = (beta * log_spot).sin();
        let exp_term = (-0.5 * vol * vol * beta * beta * tte).exp();
        let factor = 2.0 / (nf * PI) * (1.0 - (-1.0_f64).powi(n as i32));
        sum += factor * sin_term * exp_term;
    }

    df * (spot / lower).powf(-mu) * sum.max(0.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// VARIANCE / VOLATILITY SWAPS
// ═══════════════════════════════════════════════════════════════════════════

/// Fair variance strike for a variance swap (using log strip of options).
pub fn variance_swap_fair_strike(
    spot: f64,
    rate: f64,
    dividend: f64,
    vol_func: impl Fn(f64) -> f64, // vol as function of strike
    tte: f64,
    n_strikes: usize,
) -> f64 {
    let forward = spot * ((rate - dividend) * tte).exp();
    let df = (-rate * tte).exp();
    let k_low = forward * 0.5;
    let k_high = forward * 1.5;
    let dk = (k_high - k_low) / n_strikes as f64;

    let mut var_strike = 0.0;

    for i in 0..=n_strikes {
        let k = k_low + i as f64 * dk;
        let vol = vol_func(k);
        let w = if i == 0 || i == n_strikes { 0.5 } else { 1.0 };

        let p = BSParams::new(spot, k, rate, dividend, vol, tte);
        let price = if k < forward {
            crate::black_scholes::bs_put_price(&p)
        } else {
            crate::black_scholes::bs_call_price(&p)
        };

        var_strike += w * price / (k * k) * dk;
    }

    var_strike * 2.0 / (df * tte)
}

/// Vol swap fair strike approximation: K_vol ≈ sqrt(K_var) - vvol²/(8*K_var^{3/2})
pub fn vol_swap_fair_strike(var_strike: f64, vol_of_vol: f64) -> f64 {
    let sqrt_kv = var_strike.sqrt();
    sqrt_kv - vol_of_vol * vol_of_vol / (8.0 * sqrt_kv * var_strike)
}

/// Variance swap mark-to-market.
pub fn variance_swap_mtm(
    realized_var: f64,
    fair_var: f64,
    notional: f64,
    days_elapsed: f64,
    days_total: f64,
) -> f64 {
    let accrued = realized_var * days_elapsed / days_total;
    let remaining_implied = fair_var; // simplified
    let expected_final = accrued + remaining_implied * (1.0 - days_elapsed / days_total);
    notional * (expected_final - fair_var)
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_in_out_parity() {
        let diff = barrier_in_out_parity(
            100.0, 100.0, 90.0, 0.0,
            0.05, 0.02, 0.25, 1.0,
            OptionType::Call, true,
        );
        assert!(diff.abs() < 0.01, "Barrier in-out parity: {}", diff);
    }

    #[test]
    fn test_margrabe_zero_strike() {
        // Exchange option with S2=0 should equal S1
        let price = margrabe_exchange(100.0, 0.001, 0.0, 0.0, 0.2, 0.2, 0.5, 1.0);
        assert!((price - 100.0).abs() < 1.0, "Margrabe with S2~0: {}", price);
    }

    #[test]
    fn test_asian_geometric_vs_vanilla() {
        // Geometric asian should be cheaper than vanilla
        let opt = AsianOption {
            spot: 100.0, strike: 100.0, rate: 0.05, dividend: 0.0, vol: 0.2,
            time_to_expiry: 1.0, opt_type: OptionType::Call,
            avg_type: AsianAverageType::Geometric, n_observations: 12,
            running_avg: 0.0, observations_so_far: 0,
        };
        let asian_price = asian_geometric_price(&opt);
        let p = BSParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let vanilla = crate::black_scholes::bs_call_price(&p);
        assert!(asian_price < vanilla, "Asian geo {} should be < vanilla {}", asian_price, vanilla);
    }

    #[test]
    fn test_kirk_spread_positive() {
        let price = kirk_spread_call(100.0, 90.0, 5.0, 0.05, 0.2, 0.25, 0.5, 1.0);
        assert!(price > 0.0, "Spread call should be positive: {}", price);
    }

    #[test]
    fn test_bivariate_normal_independent() {
        let result = bivariate_normal_cdf(0.0, 0.0, 0.0);
        assert!((result - 0.25).abs() < 0.05, "BVN(0,0,0) should be ~0.25: {}", result);
    }
}
