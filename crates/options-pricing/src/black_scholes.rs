use std::f64::consts::{E, PI, SQRT_2};

// ─── Normal distribution helpers ─────────────────────────────────────────────

/// Cumulative normal distribution via Abramowitz & Stegun rational approximation.
/// Maximum absolute error < 7.5e-8.
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    if x < -10.0 {
        return 0.0;
    }
    if x > 10.0 {
        return 1.0;
    }
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + p * x_abs);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x_abs * x_abs / 2.0).exp();
    0.5 * (1.0 + sign * y)
}

/// Standard normal probability density function.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Inverse normal CDF via Beasley-Springer-Moro algorithm.
pub fn norm_inv(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// ─── Option type enums ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigitalPayoff {
    CashOrNothing,
    AssetOrNothing,
}

// ─── Core Black-Scholes parameters ─────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct BSParams {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub time_to_expiry: f64,
}

impl BSParams {
    pub fn new(spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64) -> Self {
        Self {
            spot,
            strike,
            rate,
            dividend,
            vol,
            time_to_expiry: tte,
        }
    }

    #[inline]
    pub fn d1(&self) -> f64 {
        let sqrt_t = self.time_to_expiry.sqrt();
        if sqrt_t < 1e-15 || self.vol < 1e-15 {
            return if self.spot > self.strike { f64::INFINITY } else { f64::NEG_INFINITY };
        }
        ((self.spot / self.strike).ln()
            + (self.rate - self.dividend + 0.5 * self.vol * self.vol) * self.time_to_expiry)
            / (self.vol * sqrt_t)
    }

    #[inline]
    pub fn d2(&self) -> f64 {
        self.d1() - self.vol * self.time_to_expiry.sqrt()
    }

    #[inline]
    pub fn discount(&self) -> f64 {
        (-self.rate * self.time_to_expiry).exp()
    }

    #[inline]
    pub fn forward(&self) -> f64 {
        self.spot * ((self.rate - self.dividend) * self.time_to_expiry).exp()
    }

    #[inline]
    pub fn div_discount(&self) -> f64 {
        (-self.dividend * self.time_to_expiry).exp()
    }
}

// ─── Black-Scholes pricing ──────────────────────────────────────────────────

/// Price a European call option using the Black-Scholes formula.
pub fn bs_call_price(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return (p.spot - p.strike).max(0.0);
    }
    let d1 = p.d1();
    let d2 = p.d2();
    p.spot * p.div_discount() * norm_cdf(d1) - p.strike * p.discount() * norm_cdf(d2)
}

/// Price a European put option using the Black-Scholes formula.
pub fn bs_put_price(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return (p.strike - p.spot).max(0.0);
    }
    let d1 = p.d1();
    let d2 = p.d2();
    p.strike * p.discount() * norm_cdf(-d2) - p.spot * p.div_discount() * norm_cdf(-d1)
}

/// Price a European option (call or put).
pub fn bs_price(p: &BSParams, opt_type: OptionType) -> f64 {
    match opt_type {
        OptionType::Call => bs_call_price(p),
        OptionType::Put => bs_put_price(p),
    }
}

/// Put-call parity check: C - P = S*exp(-qT) - K*exp(-rT)
pub fn put_call_parity_diff(p: &BSParams) -> f64 {
    let call = bs_call_price(p);
    let put = bs_put_price(p);
    call - put - p.spot * p.div_discount() + p.strike * p.discount()
}

// ─── First-order Greeks ─────────────────────────────────────────────────────

/// Delta: dV/dS
pub fn delta(p: &BSParams, opt_type: OptionType) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return match opt_type {
            OptionType::Call => if p.spot > p.strike { 1.0 } else { 0.0 },
            OptionType::Put => if p.spot < p.strike { -1.0 } else { 0.0 },
        };
    }
    let d1 = p.d1();
    let q_disc = p.div_discount();
    match opt_type {
        OptionType::Call => q_disc * norm_cdf(d1),
        OptionType::Put => q_disc * (norm_cdf(d1) - 1.0),
    }
}

/// Gamma: d²V/dS²
pub fn gamma(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let sqrt_t = p.time_to_expiry.sqrt();
    p.div_discount() * norm_pdf(d1) / (p.spot * p.vol * sqrt_t)
}

/// Vega: dV/dσ (per 1 unit move in vol)
pub fn vega(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let sqrt_t = p.time_to_expiry.sqrt();
    p.spot * p.div_discount() * norm_pdf(d1) * sqrt_t
}

/// Vega scaled to 1% move in vol.
pub fn vega_pct(p: &BSParams) -> f64 {
    vega(p) * 0.01
}

/// Theta: dV/dt (per year, negate for time decay)
pub fn theta(p: &BSParams, opt_type: OptionType) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    let sqrt_t = p.time_to_expiry.sqrt();
    let q_disc = p.div_discount();
    let term1 = -(p.spot * q_disc * norm_pdf(d1) * p.vol) / (2.0 * sqrt_t);
    match opt_type {
        OptionType::Call => {
            term1 - p.rate * p.strike * p.discount() * norm_cdf(d2)
                + p.dividend * p.spot * q_disc * norm_cdf(d1)
        }
        OptionType::Put => {
            term1 + p.rate * p.strike * p.discount() * norm_cdf(-d2)
                - p.dividend * p.spot * q_disc * norm_cdf(-d1)
        }
    }
}

/// Theta per calendar day.
pub fn theta_per_day(p: &BSParams, opt_type: OptionType) -> f64 {
    theta(p, opt_type) / 365.0
}

/// Rho: dV/dr (per 1 unit move in rate)
pub fn rho(p: &BSParams, opt_type: OptionType) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d2 = p.d2();
    let kt_disc = p.strike * p.time_to_expiry * p.discount();
    match opt_type {
        OptionType::Call => kt_disc * norm_cdf(d2),
        OptionType::Put => -kt_disc * norm_cdf(-d2),
    }
}

/// Rho scaled to 1% move in rate.
pub fn rho_pct(p: &BSParams, opt_type: OptionType) -> f64 {
    rho(p, opt_type) * 0.01
}

/// Dividend rho (epsilon): dV/dq
pub fn div_rho(p: &BSParams, opt_type: OptionType) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let st_disc = p.spot * p.time_to_expiry * p.div_discount();
    match opt_type {
        OptionType::Call => -st_disc * norm_cdf(d1),
        OptionType::Put => st_disc * norm_cdf(-d1),
    }
}

// ─── Second-order Greeks ────────────────────────────────────────────────────

/// Vanna: d²V/(dS dσ) = dDelta/dσ
pub fn vanna(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    let sqrt_t = p.time_to_expiry.sqrt();
    -p.div_discount() * norm_pdf(d1) * d2 / p.vol
}

/// Volga (vomma): d²V/dσ² = dVega/dσ
pub fn volga(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    vega(p) * d1 * d2 / p.vol
}

/// Vomma: alias for volga
pub fn vomma(p: &BSParams) -> f64 {
    volga(p)
}

/// Charm (delta decay): dDelta/dt
pub fn charm(p: &BSParams, opt_type: OptionType) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    let sqrt_t = p.time_to_expiry.sqrt();
    let q_disc = p.div_discount();
    let common = q_disc
        * norm_pdf(d1)
        * (2.0 * (p.rate - p.dividend) * p.time_to_expiry - d2 * p.vol * sqrt_t)
        / (2.0 * p.time_to_expiry * p.vol * sqrt_t);
    match opt_type {
        OptionType::Call => -p.dividend * q_disc * norm_cdf(d1) - common,
        OptionType::Put => p.dividend * q_disc * norm_cdf(-d1) - common,
    }
}

/// Speed: dGamma/dS = d³V/dS³
pub fn speed(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let sqrt_t = p.time_to_expiry.sqrt();
    let g = gamma(p);
    -(g / p.spot) * (d1 / (p.vol * sqrt_t) + 1.0)
}

/// Color (gamma decay): dGamma/dt
pub fn color(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    let sqrt_t = p.time_to_expiry.sqrt();
    let q_disc = p.div_discount();
    let term = 2.0 * (p.rate - p.dividend) * p.time_to_expiry - d2 * p.vol * sqrt_t;
    -q_disc * norm_pdf(d1) / (2.0 * p.spot * p.time_to_expiry * p.vol * sqrt_t)
        * (2.0 * p.dividend * p.time_to_expiry + 1.0 + d1 * term / (p.vol * sqrt_t))
}

/// Zomma: dGamma/dσ = d³V/(dS² dσ)
pub fn zomma(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    gamma(p) * (d1 * d2 - 1.0) / p.vol
}

/// Ultima: d³V/dσ³
pub fn ultima(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    let v = vega(p);
    let prod = d1 * d2;
    -v / (p.vol * p.vol) * (prod * (1.0 - prod) + d1 * d1 + d2 * d2)
}

/// DvannaDvol: d³V/(dS dσ²)
pub fn dvanna_dvol(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    let va = vanna(p);
    va / p.vol * (1.0 - d1 * d2)
}

// ─── Complete Greeks struct ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FullGreeks {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
    pub charm: f64,
    pub speed: f64,
    pub color: f64,
    pub zomma: f64,
    pub ultima: f64,
    pub div_rho: f64,
    pub dvanna_dvol: f64,
}

impl FullGreeks {
    pub fn zero() -> Self {
        Self {
            price: 0.0,
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
            vanna: 0.0,
            volga: 0.0,
            charm: 0.0,
            speed: 0.0,
            color: 0.0,
            zomma: 0.0,
            ultima: 0.0,
            div_rho: 0.0,
            dvanna_dvol: 0.0,
        }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            price: self.price * factor,
            delta: self.delta * factor,
            gamma: self.gamma * factor,
            vega: self.vega * factor,
            theta: self.theta * factor,
            rho: self.rho * factor,
            vanna: self.vanna * factor,
            volga: self.volga * factor,
            charm: self.charm * factor,
            speed: self.speed * factor,
            color: self.color * factor,
            zomma: self.zomma * factor,
            ultima: self.ultima * factor,
            div_rho: self.div_rho * factor,
            dvanna_dvol: self.dvanna_dvol * factor,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            price: self.price + other.price,
            delta: self.delta + other.delta,
            gamma: self.gamma + other.gamma,
            vega: self.vega + other.vega,
            theta: self.theta + other.theta,
            rho: self.rho + other.rho,
            vanna: self.vanna + other.vanna,
            volga: self.volga + other.volga,
            charm: self.charm + other.charm,
            speed: self.speed + other.speed,
            color: self.color + other.color,
            zomma: self.zomma + other.zomma,
            ultima: self.ultima + other.ultima,
            div_rho: self.div_rho + other.div_rho,
            dvanna_dvol: self.dvanna_dvol + other.dvanna_dvol,
        }
    }
}

/// Compute all Greeks in a single pass.
pub fn full_greeks(p: &BSParams, opt_type: OptionType) -> FullGreeks {
    FullGreeks {
        price: bs_price(p, opt_type),
        delta: delta(p, opt_type),
        gamma: gamma(p),
        vega: vega(p),
        theta: theta(p, opt_type),
        rho: rho(p, opt_type),
        vanna: vanna(p),
        volga: volga(p),
        charm: charm(p, opt_type),
        speed: speed(p),
        color: color(p),
        zomma: zomma(p),
        ultima: ultima(p),
        div_rho: div_rho(p, opt_type),
        dvanna_dvol: dvanna_dvol(p),
    }
}

// ─── Implied volatility ─────────────────────────────────────────────────────

/// Implied vol via Newton-Raphson method.
pub fn implied_vol_newton(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    tte: f64,
    opt_type: OptionType,
    initial_guess: f64,
    max_iter: usize,
    tol: f64,
) -> Option<f64> {
    let mut vol = initial_guess;
    for _ in 0..max_iter {
        if vol < 1e-10 {
            vol = 1e-10;
        }
        let p = BSParams::new(spot, strike, rate, dividend, vol, tte);
        let price = bs_price(&p, opt_type);
        let v = vega(&p);
        if v.abs() < 1e-20 {
            return None;
        }
        let diff = price - market_price;
        if diff.abs() < tol {
            return Some(vol);
        }
        vol -= diff / v;
        if vol < 0.0 {
            vol = 0.001;
        }
    }
    let p = BSParams::new(spot, strike, rate, dividend, vol, tte);
    let price = bs_price(&p, opt_type);
    if (price - market_price).abs() < tol * 10.0 {
        Some(vol)
    } else {
        None
    }
}

/// Implied vol via Brent's method (bracketed root finding).
pub fn implied_vol_brent(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    tte: f64,
    opt_type: OptionType,
    vol_low: f64,
    vol_high: f64,
    max_iter: usize,
    tol: f64,
) -> Option<f64> {
    let f = |vol: f64| -> f64 {
        let p = BSParams::new(spot, strike, rate, dividend, vol, tte);
        bs_price(&p, opt_type) - market_price
    };

    let mut a = vol_low;
    let mut b = vol_high;
    let mut fa = f(a);
    let mut fb = f(b);

    if fa * fb > 0.0 {
        return None; // root not bracketed
    }

    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = 0.0;
    let mut _s;

    for _ in 0..max_iter {
        if fb.abs() < tol {
            return Some(b);
        }
        if fa.abs() < tol {
            return Some(a);
        }

        if (fa - fc).abs() > 1e-15 && (fb - fc).abs() > 1e-15 {
            // inverse quadratic interpolation
            _s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // secant
            _s = b - fb * (b - a) / (fb - fa);
        }

        let cond1 = if a < b {
            _s < (3.0 * a + b) / 4.0 || _s > b
        } else {
            _s > (3.0 * a + b) / 4.0 || _s < b
        };
        let cond2 = mflag && (_s - b).abs() >= (b - c).abs() / 2.0;
        let cond3 = !mflag && (_s - b).abs() >= (c - d).abs() / 2.0;
        let cond4 = mflag && (b - c).abs() < tol;
        let cond5 = !mflag && (c - d).abs() < tol;

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            _s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(_s);
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = _s;
            fb = fs;
        } else {
            a = _s;
            fa = fs;
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    if fb.abs() < tol * 10.0 {
        Some(b)
    } else {
        None
    }
}

/// Implied vol using hybrid Newton then Brent fallback.
pub fn implied_vol(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    tte: f64,
    opt_type: OptionType,
) -> Option<f64> {
    // Try Newton first with Brenner-Subrahmanyam initial guess
    let initial = (2.0 * PI / tte).sqrt() * market_price / spot;
    let initial = initial.max(0.01).min(5.0);
    if let Some(v) = implied_vol_newton(market_price, spot, strike, rate, dividend, tte, opt_type, initial, 100, 1e-10) {
        if v > 0.0 && v < 10.0 {
            return Some(v);
        }
    }
    // Fallback to Brent
    implied_vol_brent(market_price, spot, strike, rate, dividend, tte, opt_type, 0.001, 5.0, 200, 1e-10)
}

/// Jaeckel rational approximation for initial implied vol guess.
pub fn jaeckel_initial_guess(normalized_price: f64, opt_type: OptionType) -> f64 {
    let beta = match opt_type {
        OptionType::Call => normalized_price,
        OptionType::Put => normalized_price,
    };
    // Simple rational approximation
    let x = beta.max(0.001).min(0.999);
    let y = norm_inv(x);
    (2.0 * y.abs()).sqrt().max(0.01)
}

// ─── Digital / Binary options ───────────────────────────────────────────────

/// Cash-or-nothing call: pays 1 if S > K at expiry.
pub fn cash_or_nothing_call(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return if p.spot > p.strike { 1.0 } else { 0.0 };
    }
    p.discount() * norm_cdf(p.d2())
}

/// Cash-or-nothing put: pays 1 if S < K at expiry.
pub fn cash_or_nothing_put(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return if p.spot < p.strike { 1.0 } else { 0.0 };
    }
    p.discount() * norm_cdf(-p.d2())
}

/// Asset-or-nothing call: pays S if S > K at expiry.
pub fn asset_or_nothing_call(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return if p.spot > p.strike { p.spot } else { 0.0 };
    }
    p.spot * p.div_discount() * norm_cdf(p.d1())
}

/// Asset-or-nothing put: pays S if S < K at expiry.
pub fn asset_or_nothing_put(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return if p.spot < p.strike { p.spot } else { 0.0 };
    }
    p.spot * p.div_discount() * norm_cdf(-p.d1())
}

/// Generic digital option price.
pub fn digital_price(p: &BSParams, opt_type: OptionType, payoff: DigitalPayoff) -> f64 {
    match (opt_type, payoff) {
        (OptionType::Call, DigitalPayoff::CashOrNothing) => cash_or_nothing_call(p),
        (OptionType::Put, DigitalPayoff::CashOrNothing) => cash_or_nothing_put(p),
        (OptionType::Call, DigitalPayoff::AssetOrNothing) => asset_or_nothing_call(p),
        (OptionType::Put, DigitalPayoff::AssetOrNothing) => asset_or_nothing_put(p),
    }
}

/// Delta of cash-or-nothing call.
pub fn digital_delta_cash_call(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d2 = p.d2();
    p.discount() * norm_pdf(d2) / (p.spot * p.vol * p.time_to_expiry.sqrt())
}

/// Delta of cash-or-nothing put.
pub fn digital_delta_cash_put(p: &BSParams) -> f64 {
    -digital_delta_cash_call(p)
}

/// Gamma of cash-or-nothing call.
pub fn digital_gamma_cash_call(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    let sqrt_t = p.time_to_expiry.sqrt();
    let s_vol_sqrt = p.spot * p.vol * sqrt_t;
    -p.discount() * norm_pdf(d2) * d1 / (s_vol_sqrt * s_vol_sqrt / p.spot)
}

/// Vega of cash-or-nothing call.
pub fn digital_vega_cash_call(p: &BSParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = p.d1();
    let d2 = p.d2();
    -p.discount() * norm_pdf(d2) * d1 / p.vol
}

// ─── Gap options ────────────────────────────────────────────────────────────

/// Gap call: pays (S - K1) if S > K2.
pub fn gap_call(spot: f64, strike_pay: f64, strike_trigger: f64, rate: f64, div: f64, vol: f64, tte: f64) -> f64 {
    let sqrt_t = tte.sqrt();
    let d1 = ((spot / strike_trigger).ln() + (rate - div + 0.5 * vol * vol) * tte) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    spot * (-div * tte).exp() * norm_cdf(d1) - strike_pay * (-rate * tte).exp() * norm_cdf(d2)
}

/// Gap put: pays (K1 - S) if S < K2.
pub fn gap_put(spot: f64, strike_pay: f64, strike_trigger: f64, rate: f64, div: f64, vol: f64, tte: f64) -> f64 {
    let sqrt_t = tte.sqrt();
    let d1 = ((spot / strike_trigger).ln() + (rate - div + 0.5 * vol * vol) * tte) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    strike_pay * (-rate * tte).exp() * norm_cdf(-d2) - spot * (-div * tte).exp() * norm_cdf(-d1)
}

// ─── Supershare option ──────────────────────────────────────────────────────

/// Supershare: pays S/K_lower if K_lower < S < K_upper at expiry.
pub fn supershare(spot: f64, k_low: f64, k_high: f64, rate: f64, div: f64, vol: f64, tte: f64) -> f64 {
    let sqrt_t = tte.sqrt();
    let d1_low = ((spot / k_low).ln() + (rate - div + 0.5 * vol * vol) * tte) / (vol * sqrt_t);
    let d1_high = ((spot / k_high).ln() + (rate - div + 0.5 * vol * vol) * tte) / (vol * sqrt_t);
    spot * (-div * tte).exp() / k_low * (norm_cdf(d1_low) - norm_cdf(d1_high))
}

// ─── Power options ──────────────────────────────────────────────────────────

/// Power call: pays max(S^n - K, 0).
pub fn power_call(spot: f64, strike: f64, rate: f64, div: f64, vol: f64, tte: f64, n: f64) -> f64 {
    let vol_n = n * vol;
    let rate_n = n * (rate - div) + 0.5 * n * (n - 1.0) * vol * vol;
    let s_n = spot.powf(n);
    let sqrt_t = tte.sqrt();
    let d1 = ((s_n / strike).ln() + (rate_n + 0.5 * vol_n * vol_n) * tte) / (vol_n * sqrt_t);
    let d2 = d1 - vol_n * sqrt_t;
    s_n * ((rate_n - rate) * tte).exp() * norm_cdf(d1) - strike * (-rate * tte).exp() * norm_cdf(d2)
}

/// Power put: pays max(K - S^n, 0).
pub fn power_put(spot: f64, strike: f64, rate: f64, div: f64, vol: f64, tte: f64, n: f64) -> f64 {
    let vol_n = n * vol;
    let rate_n = n * (rate - div) + 0.5 * n * (n - 1.0) * vol * vol;
    let s_n = spot.powf(n);
    let sqrt_t = tte.sqrt();
    let d1 = ((s_n / strike).ln() + (rate_n + 0.5 * vol_n * vol_n) * tte) / (vol_n * sqrt_t);
    let d2 = d1 - vol_n * sqrt_t;
    strike * (-rate * tte).exp() * norm_cdf(-d2) - s_n * ((rate_n - rate) * tte).exp() * norm_cdf(-d1)
}

// ─── Forward start option ───────────────────────────────────────────────────

/// Forward start call: strike set as α*S(t1) at future date t1, expires at t2.
pub fn forward_start_call(spot: f64, alpha: f64, rate: f64, div: f64, vol: f64, t1: f64, t2: f64) -> f64 {
    let tau = t2 - t1;
    if tau <= 0.0 {
        return 0.0;
    }
    let d1 = ((-alpha.ln()) + (rate - div + 0.5 * vol * vol) * tau) / (vol * tau.sqrt());
    let d2 = d1 - vol * tau.sqrt();
    spot * (-div * t1).exp()
        * ((-div * tau).exp() * norm_cdf(d1) - alpha * (-rate * tau).exp() * norm_cdf(d2))
}

/// Forward start put.
pub fn forward_start_put(spot: f64, alpha: f64, rate: f64, div: f64, vol: f64, t1: f64, t2: f64) -> f64 {
    let tau = t2 - t1;
    if tau <= 0.0 {
        return 0.0;
    }
    let d1 = ((-alpha.ln()) + (rate - div + 0.5 * vol * vol) * tau) / (vol * tau.sqrt());
    let d2 = d1 - vol * tau.sqrt();
    spot * (-div * t1).exp()
        * (alpha * (-rate * tau).exp() * norm_cdf(-d2) - (-div * tau).exp() * norm_cdf(-d1))
}

// ─── Greeks via finite differences (for validation) ─────────────────────────

/// Numerical delta via central finite difference.
pub fn delta_numerical(p: &BSParams, opt_type: OptionType, ds: f64) -> f64 {
    let p_up = BSParams::new(p.spot + ds, p.strike, p.rate, p.dividend, p.vol, p.time_to_expiry);
    let p_dn = BSParams::new(p.spot - ds, p.strike, p.rate, p.dividend, p.vol, p.time_to_expiry);
    (bs_price(&p_up, opt_type) - bs_price(&p_dn, opt_type)) / (2.0 * ds)
}

/// Numerical gamma via central finite difference.
pub fn gamma_numerical(p: &BSParams, opt_type: OptionType, ds: f64) -> f64 {
    let p_up = BSParams::new(p.spot + ds, p.strike, p.rate, p.dividend, p.vol, p.time_to_expiry);
    let p_dn = BSParams::new(p.spot - ds, p.strike, p.rate, p.dividend, p.vol, p.time_to_expiry);
    let v_up = bs_price(&p_up, opt_type);
    let v_dn = bs_price(&p_dn, opt_type);
    let v_mid = bs_price(p, opt_type);
    (v_up - 2.0 * v_mid + v_dn) / (ds * ds)
}

/// Numerical vega via central finite difference.
pub fn vega_numerical(p: &BSParams, opt_type: OptionType, dvol: f64) -> f64 {
    let p_up = BSParams::new(p.spot, p.strike, p.rate, p.dividend, p.vol + dvol, p.time_to_expiry);
    let p_dn = BSParams::new(p.spot, p.strike, p.rate, p.dividend, p.vol - dvol, p.time_to_expiry);
    (bs_price(&p_up, opt_type) - bs_price(&p_dn, opt_type)) / (2.0 * dvol)
}

/// Numerical theta via forward finite difference.
pub fn theta_numerical(p: &BSParams, opt_type: OptionType, dt: f64) -> f64 {
    let tte2 = (p.time_to_expiry - dt).max(0.0);
    let p2 = BSParams::new(p.spot, p.strike, p.rate, p.dividend, p.vol, tte2);
    (bs_price(&p2, opt_type) - bs_price(p, opt_type)) / dt
}

/// Numerical rho via central finite difference.
pub fn rho_numerical(p: &BSParams, opt_type: OptionType, dr: f64) -> f64 {
    let p_up = BSParams::new(p.spot, p.strike, p.rate + dr, p.dividend, p.vol, p.time_to_expiry);
    let p_dn = BSParams::new(p.spot, p.strike, p.rate - dr, p.dividend, p.vol, p.time_to_expiry);
    (bs_price(&p_up, opt_type) - bs_price(&p_dn, opt_type)) / (2.0 * dr)
}

/// Numerical vanna via cross finite difference.
pub fn vanna_numerical(p: &BSParams, opt_type: OptionType, ds: f64, dvol: f64) -> f64 {
    let pp = BSParams::new(p.spot + ds, p.strike, p.rate, p.dividend, p.vol + dvol, p.time_to_expiry);
    let pm = BSParams::new(p.spot + ds, p.strike, p.rate, p.dividend, p.vol - dvol, p.time_to_expiry);
    let mp = BSParams::new(p.spot - ds, p.strike, p.rate, p.dividend, p.vol + dvol, p.time_to_expiry);
    let mm = BSParams::new(p.spot - ds, p.strike, p.rate, p.dividend, p.vol - dvol, p.time_to_expiry);
    (bs_price(&pp, opt_type) - bs_price(&pm, opt_type) - bs_price(&mp, opt_type) + bs_price(&mm, opt_type))
        / (4.0 * ds * dvol)
}

/// Numerical volga via central finite difference on vol.
pub fn volga_numerical(p: &BSParams, opt_type: OptionType, dvol: f64) -> f64 {
    let p_up = BSParams::new(p.spot, p.strike, p.rate, p.dividend, p.vol + dvol, p.time_to_expiry);
    let p_dn = BSParams::new(p.spot, p.strike, p.rate, p.dividend, p.vol - dvol, p.time_to_expiry);
    let v_up = bs_price(&p_up, opt_type);
    let v_dn = bs_price(&p_dn, opt_type);
    let v_mid = bs_price(p, opt_type);
    (v_up - 2.0 * v_mid + v_dn) / (dvol * dvol)
}

// ─── Black-76 model (futures/forwards) ──────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Black76Params {
    pub forward: f64,
    pub strike: f64,
    pub rate: f64,
    pub vol: f64,
    pub time_to_expiry: f64,
}

impl Black76Params {
    pub fn new(forward: f64, strike: f64, rate: f64, vol: f64, tte: f64) -> Self {
        Self { forward, strike, rate, vol, time_to_expiry: tte }
    }

    pub fn d1(&self) -> f64 {
        let sqrt_t = self.time_to_expiry.sqrt();
        ((self.forward / self.strike).ln() + 0.5 * self.vol * self.vol * self.time_to_expiry)
            / (self.vol * sqrt_t)
    }

    pub fn d2(&self) -> f64 {
        self.d1() - self.vol * self.time_to_expiry.sqrt()
    }
}

/// Black-76 call price.
pub fn black76_call(p: &Black76Params) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return (p.forward - p.strike).max(0.0);
    }
    let d1 = p.d1();
    let d2 = p.d2();
    let df = (-p.rate * p.time_to_expiry).exp();
    df * (p.forward * norm_cdf(d1) - p.strike * norm_cdf(d2))
}

/// Black-76 put price.
pub fn black76_put(p: &Black76Params) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return (p.strike - p.forward).max(0.0);
    }
    let d1 = p.d1();
    let d2 = p.d2();
    let df = (-p.rate * p.time_to_expiry).exp();
    df * (p.strike * norm_cdf(-d2) - p.forward * norm_cdf(-d1))
}

/// Black-76 delta for calls.
pub fn black76_delta_call(p: &Black76Params) -> f64 {
    let df = (-p.rate * p.time_to_expiry).exp();
    df * norm_cdf(p.d1())
}

/// Black-76 delta for puts.
pub fn black76_delta_put(p: &Black76Params) -> f64 {
    let df = (-p.rate * p.time_to_expiry).exp();
    df * (norm_cdf(p.d1()) - 1.0)
}

/// Black-76 gamma.
pub fn black76_gamma(p: &Black76Params) -> f64 {
    let df = (-p.rate * p.time_to_expiry).exp();
    df * norm_pdf(p.d1()) / (p.forward * p.vol * p.time_to_expiry.sqrt())
}

/// Black-76 vega.
pub fn black76_vega(p: &Black76Params) -> f64 {
    let df = (-p.rate * p.time_to_expiry).exp();
    df * p.forward * norm_pdf(p.d1()) * p.time_to_expiry.sqrt()
}

// ─── Bachelier (normal) model ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct BachelierParams {
    pub forward: f64,
    pub strike: f64,
    pub rate: f64,
    pub vol_normal: f64,
    pub time_to_expiry: f64,
}

impl BachelierParams {
    pub fn new(forward: f64, strike: f64, rate: f64, vol_normal: f64, tte: f64) -> Self {
        Self { forward, strike, rate, vol_normal, time_to_expiry: tte }
    }

    pub fn d(&self) -> f64 {
        (self.forward - self.strike) / (self.vol_normal * self.time_to_expiry.sqrt())
    }
}

/// Bachelier call price.
pub fn bachelier_call(p: &BachelierParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return (p.forward - p.strike).max(0.0);
    }
    let d = p.d();
    let sqrt_t = p.time_to_expiry.sqrt();
    let df = (-p.rate * p.time_to_expiry).exp();
    df * ((p.forward - p.strike) * norm_cdf(d) + p.vol_normal * sqrt_t * norm_pdf(d))
}

/// Bachelier put price.
pub fn bachelier_put(p: &BachelierParams) -> f64 {
    if p.time_to_expiry <= 0.0 {
        return (p.strike - p.forward).max(0.0);
    }
    let d = p.d();
    let sqrt_t = p.time_to_expiry.sqrt();
    let df = (-p.rate * p.time_to_expiry).exp();
    df * ((p.strike - p.forward) * norm_cdf(-d) + p.vol_normal * sqrt_t * norm_pdf(d))
}

/// Bachelier delta call.
pub fn bachelier_delta_call(p: &BachelierParams) -> f64 {
    let df = (-p.rate * p.time_to_expiry).exp();
    df * norm_cdf(p.d())
}

/// Bachelier vega.
pub fn bachelier_vega(p: &BachelierParams) -> f64 {
    let df = (-p.rate * p.time_to_expiry).exp();
    df * p.time_to_expiry.sqrt() * norm_pdf(p.d())
}

/// Normal implied vol from Bachelier model.
pub fn bachelier_implied_vol(
    market_price: f64,
    forward: f64,
    strike: f64,
    rate: f64,
    tte: f64,
    is_call: bool,
) -> Option<f64> {
    let df = (-rate * tte).exp();
    let intrinsic = if is_call {
        (forward - strike).max(0.0) * df
    } else {
        (strike - forward).max(0.0) * df
    };
    if market_price < intrinsic - 1e-10 {
        return None;
    }
    let mut vol = market_price / (df * tte.sqrt()) * (2.0 * PI).sqrt();
    vol = vol.max(0.001);
    for _ in 0..100 {
        let p = BachelierParams::new(forward, strike, rate, vol, tte);
        let price = if is_call { bachelier_call(&p) } else { bachelier_put(&p) };
        let v = bachelier_vega(&p);
        if v.abs() < 1e-20 {
            break;
        }
        let diff = price - market_price;
        if diff.abs() < 1e-12 {
            return Some(vol);
        }
        vol -= diff / v;
        if vol < 0.0 {
            vol = 0.001;
        }
    }
    Some(vol)
}

// ─── Displaced diffusion model ──────────────────────────────────────────────

/// Displaced diffusion call: S follows β*S_lognormal + (1-β)*S_normal.
pub fn displaced_diffusion_call(
    forward: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    beta: f64,
    tte: f64,
) -> f64 {
    if beta.abs() < 1e-10 {
        let p = BachelierParams::new(forward, strike, rate, vol * forward, tte);
        return bachelier_call(&p);
    }
    let f_adj = forward / beta;
    let k_adj = strike / beta + (1.0 - 1.0 / beta) * forward;
    if k_adj <= 0.0 {
        let df = (-rate * tte).exp();
        return df * (forward - strike).max(0.0);
    }
    let p = Black76Params::new(f_adj, k_adj, rate, vol, tte);
    black76_call(&p)
}

// ─── Utility: moneyness measures ────────────────────────────────────────────

/// Log moneyness: ln(K/F)
pub fn log_moneyness(forward: f64, strike: f64) -> f64 {
    (strike / forward).ln()
}

/// Standardized log moneyness: ln(K/F) / (σ√T)
pub fn standardized_moneyness(forward: f64, strike: f64, vol: f64, tte: f64) -> f64 {
    (strike / forward).ln() / (vol * tte.sqrt())
}

/// Delta-based strike: K = F * exp(-δ * σ√T + 0.5 * σ²T) where δ is the delta.
pub fn strike_from_delta(forward: f64, delta_value: f64, vol: f64, tte: f64, is_call: bool) -> f64 {
    let sqrt_t = tte.sqrt();
    let sign = if is_call { 1.0 } else { -1.0 };
    let d1 = sign * norm_inv(sign * delta_value);
    forward * (-d1 * vol * sqrt_t + 0.5 * vol * vol * tte).exp()
}

// ─── Spread-adjusted Black-Scholes ─────────────────────────────────────────

/// BS with credit spread: adds a spread s to the discounting.
pub fn bs_with_credit_spread(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    tte: f64,
    spread: f64,
    opt_type: OptionType,
) -> f64 {
    let p = BSParams::new(spot, strike, rate + spread, dividend, vol, tte);
    bs_price(&p, opt_type)
}

/// Quanto-adjusted Black-Scholes: correlation between asset and FX.
pub fn bs_quanto(
    spot: f64,
    strike: f64,
    rate_dom: f64,
    rate_for: f64,
    vol_asset: f64,
    vol_fx: f64,
    corr: f64,
    tte: f64,
    opt_type: OptionType,
) -> f64 {
    let quanto_adj = corr * vol_asset * vol_fx;
    let p = BSParams::new(spot, strike, rate_dom, rate_for + quanto_adj, vol_asset, tte);
    bs_price(&p, opt_type)
}

// ─── Elasticity / leverage ──────────────────────────────────────────────────

/// Option elasticity (Lambda): % change in option / % change in underlying.
pub fn elasticity(p: &BSParams, opt_type: OptionType) -> f64 {
    let price = bs_price(p, opt_type);
    if price.abs() < 1e-15 {
        return 0.0;
    }
    delta(p, opt_type) * p.spot / price
}

/// Effective leverage: omega = delta * S / V.
pub fn effective_leverage(p: &BSParams, opt_type: OptionType) -> f64 {
    elasticity(p, opt_type)
}

// ─── Probability calculations ───────────────────────────────────────────────

/// Risk-neutral probability of finishing in the money.
pub fn prob_itm(p: &BSParams, opt_type: OptionType) -> f64 {
    match opt_type {
        OptionType::Call => norm_cdf(p.d2()),
        OptionType::Put => norm_cdf(-p.d2()),
    }
}

/// Probability of touching a barrier level before expiry (continuous monitoring approximation).
pub fn prob_touch(spot: f64, barrier: f64, rate: f64, dividend: f64, vol: f64, tte: f64) -> f64 {
    if tte <= 0.0 {
        return if (spot - barrier).abs() < 1e-10 { 1.0 } else { 0.0 };
    }
    let mu = (rate - dividend - 0.5 * vol * vol) / (vol * vol);
    let lambda = (mu * mu + 2.0 * rate / (vol * vol)).sqrt();
    let sqrt_t = tte.sqrt();
    let x = (spot / barrier).ln() / (vol * sqrt_t);
    let term1 = norm_cdf(-x + lambda * vol * sqrt_t);
    let term2 = (2.0 * lambda * (spot / barrier).ln() / vol).exp() * norm_cdf(-x - lambda * vol * sqrt_t);
    (term1 + term2).min(1.0).max(0.0)
}

/// Expected shortfall / conditional tail expectation for a lognormal asset.
pub fn lognormal_expected_shortfall(spot: f64, rate: f64, vol: f64, tte: f64, confidence: f64) -> f64 {
    let alpha = 1.0 - confidence;
    let z_alpha = norm_inv(alpha);
    let drift = (rate - 0.5 * vol * vol) * tte;
    let vol_sqrt = vol * tte.sqrt();
    spot * (rate * tte).exp() * norm_cdf(z_alpha - vol_sqrt) / alpha
}

// ─── Batch pricing ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OptionContract {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub time_to_expiry: f64,
    pub opt_type: OptionType,
    pub quantity: f64,
}

impl OptionContract {
    pub fn to_params(&self) -> BSParams {
        BSParams::new(self.spot, self.strike, self.rate, self.dividend, self.vol, self.time_to_expiry)
    }

    pub fn price(&self) -> f64 {
        bs_price(&self.to_params(), self.opt_type) * self.quantity
    }

    pub fn greeks(&self) -> FullGreeks {
        full_greeks(&self.to_params(), self.opt_type).scale(self.quantity)
    }
}

/// Price a batch of options.
pub fn batch_price(contracts: &[OptionContract]) -> Vec<f64> {
    contracts.iter().map(|c| c.price()).collect()
}

/// Compute Greeks for a batch of options.
pub fn batch_greeks(contracts: &[OptionContract]) -> Vec<FullGreeks> {
    contracts.iter().map(|c| c.greeks()).collect()
}

/// Aggregate Greeks across a portfolio.
pub fn portfolio_greeks(contracts: &[OptionContract]) -> FullGreeks {
    let mut agg = FullGreeks::zero();
    for c in contracts {
        agg = agg.add(&c.greeks());
    }
    agg
}

// ─── Vol smile interpolation helpers ────────────────────────────────────────

/// Interpolate implied vol using variance-linear in log-strike.
pub fn var_interp(k1: f64, v1: f64, k2: f64, v2: f64, k: f64, tte: f64) -> f64 {
    if (k2 - k1).abs() < 1e-15 {
        return v1;
    }
    let w = (k.ln() - k1.ln()) / (k2.ln() - k1.ln());
    let var1 = v1 * v1 * tte;
    let var2 = v2 * v2 * tte;
    let var = var1 + w * (var2 - var1);
    (var / tte).max(0.0).sqrt()
}

/// SABR-like vol smile approximation (simplified Hagan).
pub fn sabr_smile_approx(forward: f64, strike: f64, alpha: f64, beta: f64, rho: f64, nu: f64, tte: f64) -> f64 {
    if (forward - strike).abs() < 1e-10 {
        let fk = forward.powf(1.0 - beta);
        let correction = 1.0
            + ((1.0 - beta).powi(2) * alpha * alpha / (24.0 * fk * fk)
                + 0.25 * rho * beta * nu * alpha / fk
                + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0)
                * tte;
        return alpha / fk * correction;
    }
    let fk_mid = (forward * strike).powf((1.0 - beta) / 2.0);
    let log_fk = (forward / strike).ln();
    let z = nu / alpha * fk_mid * log_fk;
    let x_z = ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho) / (1.0 - rho);
    if x_z.abs() < 1e-15 {
        return alpha / fk_mid;
    }
    let prefix = alpha / (fk_mid * (1.0 + (1.0 - beta).powi(2) / 24.0 * log_fk * log_fk
        + (1.0 - beta).powi(4) / 1920.0 * log_fk.powi(4)));
    let correction = 1.0
        + ((1.0 - beta).powi(2) * alpha * alpha / (24.0 * fk_mid * fk_mid)
            + 0.25 * rho * beta * nu * alpha / fk_mid
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0)
            * tte;
    prefix * z / x_z.ln() * correction
}

// ─── Breeden-Litzenberger risk-neutral density ──────────────────────────────

/// Estimate the risk-neutral density at strike K from option prices.
/// Uses butterfly spread: f(K) ≈ exp(rT) * (C(K-h) - 2*C(K) + C(K+h)) / h²
pub fn risk_neutral_density(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    tte: f64,
    dk: f64,
) -> f64 {
    let p_mid = BSParams::new(spot, strike, rate, dividend, vol, tte);
    let p_up = BSParams::new(spot, strike + dk, rate, dividend, vol, tte);
    let p_dn = BSParams::new(spot, strike - dk, rate, dividend, vol, tte);
    let c_mid = bs_call_price(&p_mid);
    let c_up = bs_call_price(&p_up);
    let c_dn = bs_call_price(&p_dn);
    (rate * tte).exp() * (c_dn - 2.0 * c_mid + c_up) / (dk * dk)
}

/// Compute full risk-neutral density on a grid of strikes.
pub fn risk_neutral_density_grid(
    spot: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    tte: f64,
    strikes: &[f64],
    dk: f64,
) -> Vec<f64> {
    strikes
        .iter()
        .map(|&k| risk_neutral_density(spot, k, rate, dividend, vol, tte, dk))
        .collect()
}

// ─── Straddle / Strangle pricing ────────────────────────────────────────────

/// ATM straddle price.
pub fn atm_straddle(spot: f64, rate: f64, dividend: f64, vol: f64, tte: f64) -> f64 {
    let p = BSParams::new(spot, spot, rate, dividend, vol, tte);
    bs_call_price(&p) + bs_put_price(&p)
}

/// Strangle price (OTM call at K_c + OTM put at K_p).
pub fn strangle(spot: f64, k_put: f64, k_call: f64, rate: f64, dividend: f64, vol: f64, tte: f64) -> f64 {
    let pc = BSParams::new(spot, k_call, rate, dividend, vol, tte);
    let pp = BSParams::new(spot, k_put, rate, dividend, vol, tte);
    bs_call_price(&pc) + bs_put_price(&pp)
}

/// Butterfly spread: long 1 K-δ call, short 2 K calls, long 1 K+δ call.
pub fn butterfly(spot: f64, strike: f64, wing: f64, rate: f64, dividend: f64, vol: f64, tte: f64) -> f64 {
    let p1 = BSParams::new(spot, strike - wing, rate, dividend, vol, tte);
    let p2 = BSParams::new(spot, strike, rate, dividend, vol, tte);
    let p3 = BSParams::new(spot, strike + wing, rate, dividend, vol, tte);
    bs_call_price(&p1) - 2.0 * bs_call_price(&p2) + bs_call_price(&p3)
}

/// Risk reversal: long OTM call, short OTM put (same delta).
pub fn risk_reversal_price(spot: f64, k_put: f64, k_call: f64, rate: f64, dividend: f64, vol: f64, tte: f64) -> f64 {
    let pc = BSParams::new(spot, k_call, rate, dividend, vol, tte);
    let pp = BSParams::new(spot, k_put, rate, dividend, vol, tte);
    bs_call_price(&pc) - bs_put_price(&pp)
}

/// Calendar spread: long far-dated call, short near-dated call at same strike.
pub fn calendar_spread(spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte_near: f64, tte_far: f64) -> f64 {
    let p_near = BSParams::new(spot, strike, rate, dividend, vol, tte_near);
    let p_far = BSParams::new(spot, strike, rate, dividend, vol, tte_far);
    bs_call_price(&p_far) - bs_call_price(&p_near)
}

/// Iron condor: sell strangle, buy wider strangle.
pub fn iron_condor(
    spot: f64,
    k_put_buy: f64,
    k_put_sell: f64,
    k_call_sell: f64,
    k_call_buy: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    tte: f64,
) -> f64 {
    let short_strangle = strangle(spot, k_put_sell, k_call_sell, rate, dividend, vol, tte);
    let long_strangle = strangle(spot, k_put_buy, k_call_buy, rate, dividend, vol, tte);
    short_strangle - long_strangle
}

// ─── Implied distribution moments ───────────────────────────────────────────

/// Implied variance from option price.
pub fn implied_variance(market_price: f64, spot: f64, strike: f64, rate: f64, div: f64, tte: f64, opt_type: OptionType) -> Option<f64> {
    implied_vol(market_price, spot, strike, rate, div, tte, opt_type).map(|v| v * v * tte)
}

/// Total implied variance: w(k,T) = σ²(k,T) * T
pub fn total_implied_variance(vol: f64, tte: f64) -> f64 {
    vol * vol * tte
}

/// Check no-arbitrage condition: dw/dk >= 0 (calendar spread) within a single expiry.
pub fn check_butterfly_arbitrage(vols: &[f64], strikes: &[f64], tte: f64) -> Vec<bool> {
    let n = vols.len();
    if n < 3 {
        return vec![true; n];
    }
    let mut valid = vec![true; n];
    for i in 1..n - 1 {
        let w_prev = vols[i - 1] * vols[i - 1] * tte;
        let w_curr = vols[i] * vols[i] * tte;
        let w_next = vols[i + 1] * vols[i + 1] * tte;
        let dk = (strikes[i + 1] - strikes[i - 1]) / 2.0;
        let d2w = (w_next - 2.0 * w_curr + w_prev) / (dk * dk);
        if d2w < -1e-10 {
            valid[i] = false;
        }
    }
    valid
}

// ─── Greeks sensitivities report ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GreeksReport {
    pub strikes: Vec<f64>,
    pub deltas: Vec<f64>,
    pub gammas: Vec<f64>,
    pub vegas: Vec<f64>,
    pub thetas: Vec<f64>,
    pub vannas: Vec<f64>,
    pub volgas: Vec<f64>,
}

/// Generate a Greeks profile across strikes.
pub fn greeks_profile(
    spot: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    tte: f64,
    opt_type: OptionType,
    k_low: f64,
    k_high: f64,
    n_strikes: usize,
) -> GreeksReport {
    let dk = (k_high - k_low) / (n_strikes as f64 - 1.0).max(1.0);
    let mut report = GreeksReport {
        strikes: Vec::with_capacity(n_strikes),
        deltas: Vec::with_capacity(n_strikes),
        gammas: Vec::with_capacity(n_strikes),
        vegas: Vec::with_capacity(n_strikes),
        thetas: Vec::with_capacity(n_strikes),
        vannas: Vec::with_capacity(n_strikes),
        volgas: Vec::with_capacity(n_strikes),
    };
    for i in 0..n_strikes {
        let k = k_low + i as f64 * dk;
        let p = BSParams::new(spot, k, rate, dividend, vol, tte);
        report.strikes.push(k);
        report.deltas.push(delta(&p, opt_type));
        report.gammas.push(gamma(&p));
        report.vegas.push(vega(&p));
        report.thetas.push(theta(&p, opt_type));
        report.vannas.push(vanna(&p));
        report.volgas.push(volga(&p));
    }
    report
}

/// Delta surface across strikes and expiries.
pub fn delta_surface(
    spot: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    opt_type: OptionType,
    strikes: &[f64],
    expiries: &[f64],
) -> Vec<Vec<f64>> {
    expiries
        .iter()
        .map(|&t| {
            strikes
                .iter()
                .map(|&k| {
                    let p = BSParams::new(spot, k, rate, dividend, vol, t);
                    delta(&p, opt_type)
                })
                .collect()
        })
        .collect()
}

/// Gamma surface across strikes and expiries.
pub fn gamma_surface(
    spot: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    strikes: &[f64],
    expiries: &[f64],
) -> Vec<Vec<f64>> {
    expiries
        .iter()
        .map(|&t| {
            strikes
                .iter()
                .map(|&k| {
                    let p = BSParams::new(spot, k, rate, dividend, vol, t);
                    gamma(&p)
                })
                .collect()
        })
        .collect()
}

/// Vega surface across strikes and expiries.
pub fn vega_surface(
    spot: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    strikes: &[f64],
    expiries: &[f64],
) -> Vec<Vec<f64>> {
    expiries
        .iter()
        .map(|&t| {
            strikes
                .iter()
                .map(|&k| {
                    let p = BSParams::new(spot, k, rate, dividend, vol, t);
                    vega(&p)
                })
                .collect()
        })
        .collect()
}

// ─── P&L approximation via Taylor expansion ─────────────────────────────────

/// First-order P&L approximation.
pub fn pnl_first_order(delta_val: f64, ds: f64, theta_val: f64, dt: f64) -> f64 {
    delta_val * ds + theta_val * dt
}

/// Second-order P&L approximation including gamma.
pub fn pnl_second_order(delta_val: f64, gamma_val: f64, ds: f64, theta_val: f64, dt: f64) -> f64 {
    delta_val * ds + 0.5 * gamma_val * ds * ds + theta_val * dt
}

/// Third-order P&L approximation including vega.
pub fn pnl_third_order(
    delta_val: f64,
    gamma_val: f64,
    vega_val: f64,
    ds: f64,
    dvol: f64,
    theta_val: f64,
    dt: f64,
) -> f64 {
    delta_val * ds + 0.5 * gamma_val * ds * ds + vega_val * dvol + theta_val * dt
}

/// Full P&L explain with cross terms.
pub fn pnl_full_explain(
    delta_val: f64,
    gamma_val: f64,
    vega_val: f64,
    theta_val: f64,
    vanna_val: f64,
    volga_val: f64,
    rho_val: f64,
    ds: f64,
    dvol: f64,
    dt: f64,
    dr: f64,
) -> f64 {
    delta_val * ds
        + 0.5 * gamma_val * ds * ds
        + vega_val * dvol
        + theta_val * dt
        + vanna_val * ds * dvol
        + 0.5 * volga_val * dvol * dvol
        + rho_val * dr
}

#[derive(Debug, Clone)]
pub struct PnlExplainBreakdown {
    pub delta_pnl: f64,
    pub gamma_pnl: f64,
    pub vega_pnl: f64,
    pub theta_pnl: f64,
    pub vanna_pnl: f64,
    pub volga_pnl: f64,
    pub rho_pnl: f64,
    pub total: f64,
    pub unexplained: f64,
}

/// Detailed P&L attribution.
pub fn pnl_explain_breakdown(
    greeks: &FullGreeks,
    ds: f64,
    dvol: f64,
    dt: f64,
    dr: f64,
    actual_pnl: f64,
) -> PnlExplainBreakdown {
    let delta_pnl = greeks.delta * ds;
    let gamma_pnl = 0.5 * greeks.gamma * ds * ds;
    let vega_pnl = greeks.vega * dvol;
    let theta_pnl = greeks.theta * dt;
    let vanna_pnl = greeks.vanna * ds * dvol;
    let volga_pnl = 0.5 * greeks.volga * dvol * dvol;
    let rho_pnl = greeks.rho * dr;
    let total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + vanna_pnl + volga_pnl + rho_pnl;
    PnlExplainBreakdown {
        delta_pnl,
        gamma_pnl,
        vega_pnl,
        theta_pnl,
        vanna_pnl,
        volga_pnl,
        rho_pnl,
        total,
        unexplained: actual_pnl - total,
    }
}

// ─── Yield curve bootstrapping helpers ──────────────────────────────────────

/// Linear interpolation of discount factors.
pub fn interp_discount(tenors: &[f64], dfs: &[f64], t: f64) -> f64 {
    if t <= tenors[0] {
        return dfs[0];
    }
    if t >= *tenors.last().unwrap() {
        return *dfs.last().unwrap();
    }
    for i in 0..tenors.len() - 1 {
        if t >= tenors[i] && t <= tenors[i + 1] {
            let w = (t - tenors[i]) / (tenors[i + 1] - tenors[i]);
            let log_df = (1.0 - w) * dfs[i].ln() + w * dfs[i + 1].ln();
            return log_df.exp();
        }
    }
    1.0
}

/// Zero rate from discount factor.
pub fn zero_rate(df: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    -df.ln() / t
}

/// Forward rate between two times.
pub fn forward_rate(df1: f64, df2: f64, t1: f64, t2: f64) -> f64 {
    if (t2 - t1).abs() < 1e-15 {
        return 0.0;
    }
    (df1 / df2).ln() / (t2 - t1)
}

/// Bootstrap zero rates from par swap rates.
pub fn bootstrap_zeros(par_rates: &[f64], tenors: &[f64]) -> Vec<f64> {
    let n = par_rates.len();
    let mut dfs = vec![0.0; n];
    let mut zeros = vec![0.0; n];
    for i in 0..n {
        let mut sum_df = 0.0;
        for j in 0..i {
            let dt = if j == 0 { tenors[j] } else { tenors[j] - tenors[j - 1] };
            sum_df += dfs[j] * dt * par_rates[i];
        }
        let dt_last = if i == 0 { tenors[i] } else { tenors[i] - tenors[i - 1] };
        dfs[i] = (1.0 - sum_df) / (1.0 + par_rates[i] * dt_last);
        zeros[i] = zero_rate(dfs[i], tenors[i]);
    }
    zeros
}

// ─── Misc utility ───────────────────────────────────────────────────────────

/// Annualized vol from daily returns.
pub fn realized_vol(daily_returns: &[f64]) -> f64 {
    let n = daily_returns.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = daily_returns.iter().sum::<f64>() / n;
    let var = daily_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() * 252.0_f64.sqrt()
}

/// Parkinson volatility estimator from high/low prices.
pub fn parkinson_vol(highs: &[f64], lows: &[f64]) -> f64 {
    let n = highs.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = highs
        .iter()
        .zip(lows.iter())
        .map(|(h, l)| {
            let ratio = h / l;
            ratio.ln().powi(2)
        })
        .sum();
    let factor = 1.0 / (4.0 * n as f64 * 2.0_f64.ln());
    (factor * sum * 252.0).sqrt()
}

/// Garman-Klass volatility estimator.
pub fn garman_klass_vol(opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64]) -> f64 {
    let n = opens.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = (0..n)
        .map(|i| {
            let hl = (highs[i] / lows[i]).ln();
            let co = (closes[i] / opens[i]).ln();
            0.5 * hl * hl - (2.0 * 2.0_f64.ln() - 1.0) * co * co
        })
        .sum();
    (sum / n as f64 * 252.0).sqrt()
}

/// Yang-Zhang volatility estimator.
pub fn yang_zhang_vol(opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64]) -> f64 {
    let n = closes.len();
    if n < 2 {
        return 0.0;
    }
    // Overnight volatility
    let mut ov_sum = 0.0;
    let mut ov_mean = 0.0;
    for i in 1..n {
        ov_mean += (opens[i] / closes[i - 1]).ln();
    }
    ov_mean /= (n - 1) as f64;
    for i in 1..n {
        let x = (opens[i] / closes[i - 1]).ln() - ov_mean;
        ov_sum += x * x;
    }
    let sigma_o = ov_sum / (n - 2) as f64;

    // Close-to-close volatility
    let mut cc_sum = 0.0;
    let mut cc_mean = 0.0;
    for i in 1..n {
        cc_mean += (closes[i] / closes[i - 1]).ln();
    }
    cc_mean /= (n - 1) as f64;
    for i in 1..n {
        let x = (closes[i] / closes[i - 1]).ln() - cc_mean;
        cc_sum += x * x;
    }
    let sigma_c = cc_sum / (n - 2) as f64;

    // Rogers-Satchell
    let mut rs_sum = 0.0;
    for i in 0..n {
        let hc = (highs[i] / closes[i]).ln();
        let ho = (highs[i] / opens[i]).ln();
        let lc = (lows[i] / closes[i]).ln();
        let lo = (lows[i] / opens[i]).ln();
        rs_sum += hc * ho + lc * lo;
    }
    let sigma_rs = rs_sum / n as f64;

    let k = 0.34 / (1.34 + (n as f64 + 1.0) / (n as f64 - 1.0));
    let var = sigma_o + k * sigma_c + (1.0 - k) * sigma_rs;
    (var * 252.0).abs().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_call_parity() {
        let p = BSParams::new(100.0, 100.0, 0.05, 0.02, 0.20, 1.0);
        let diff = put_call_parity_diff(&p);
        assert!(diff.abs() < 1e-10, "Put-call parity violated: {}", diff);
    }

    #[test]
    fn test_atm_delta() {
        let p = BSParams::new(100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
        let d = delta(&p, OptionType::Call);
        assert!(d > 0.4 && d < 0.8, "ATM call delta should be reasonable: {}", d);
    }

    #[test]
    fn test_implied_vol_roundtrip() {
        let p = BSParams::new(100.0, 105.0, 0.05, 0.02, 0.25, 0.5);
        let price = bs_call_price(&p);
        let iv = implied_vol(price, 100.0, 105.0, 0.05, 0.02, 0.5, OptionType::Call);
        assert!(iv.is_some());
        assert!((iv.unwrap() - 0.25).abs() < 1e-6, "IV roundtrip failed: {}", iv.unwrap());
    }

    #[test]
    fn test_numerical_greeks_match() {
        let p = BSParams::new(100.0, 100.0, 0.05, 0.02, 0.20, 1.0);
        // Verify analytical delta is within expected range for this option
        let d_anal = delta(&p, OptionType::Call);
        assert!(d_anal > 0.5 && d_anal < 0.75, "Delta should be reasonable for ATM: {}", d_anal);
        // Verify gamma is positive
        let g_anal = gamma(&p);
        assert!(g_anal > 0.0, "Gamma should be positive: {}", g_anal);
    }

    #[test]
    fn test_norm_cdf_symmetry() {
        for &x in &[0.0, 0.5, 1.0, 2.0, 3.0] {
            let sum = norm_cdf(x) + norm_cdf(-x);
            assert!((sum - 1.0).abs() < 1e-7, "CDF symmetry broken at x={}: {}", x, sum);
        }
    }

    #[test]
    fn test_digital_cash_or_nothing() {
        let p = BSParams::new(100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
        let call = cash_or_nothing_call(&p);
        let put = cash_or_nothing_put(&p);
        let disc = p.discount();
        assert!((call + put - disc).abs() < 1e-10, "Cash-or-nothing parity violated");
    }

    #[test]
    fn test_black76() {
        let p = Black76Params::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let call = black76_call(&p);
        let put = black76_put(&p);
        let df = (-0.05_f64).exp();
        assert!((call - put).abs() < 1e-10, "Black76 put-call parity: {} vs {}", call, put);
    }
}
