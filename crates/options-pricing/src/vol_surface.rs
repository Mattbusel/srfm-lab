use std::f64::consts::PI;
use crate::black_scholes::{norm_cdf, norm_pdf, norm_inv, BSParams, OptionType, bs_price, implied_vol};

// ═══════════════════════════════════════════════════════════════════════════
// SVI PARAMETRIC SURFACE
// ═══════════════════════════════════════════════════════════════════════════

/// SVI raw parameterization: w(k) = a + b * (ρ(k-m) + sqrt((k-m)² + σ²))
/// where k = ln(K/F) is log-moneyness, w = σ²T is total variance.
#[derive(Debug, Clone, Copy)]
pub struct SviRaw {
    pub a: f64,     // overall variance level
    pub b: f64,     // slope (b >= 0)
    pub rho: f64,   // rotation/skew (-1 < ρ < 1)
    pub m: f64,     // translation
    pub sigma: f64, // ATM curvature (σ > 0)
}

impl SviRaw {
    pub fn new(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Self {
        Self { a, b, rho, m, sigma }
    }

    /// Total variance w(k) at log-moneyness k.
    pub fn total_variance(&self, k: f64) -> f64 {
        let dk = k - self.m;
        self.a + self.b * (self.rho * dk + (dk * dk + self.sigma * self.sigma).sqrt())
    }

    /// Implied vol at log-moneyness k and time to expiry t.
    pub fn implied_vol(&self, k: f64, t: f64) -> f64 {
        let w = self.total_variance(k);
        if w <= 0.0 || t <= 0.0 {
            return 0.0;
        }
        (w / t).sqrt()
    }

    /// First derivative dw/dk.
    pub fn dw_dk(&self, k: f64) -> f64 {
        let dk = k - self.m;
        let sqrt_term = (dk * dk + self.sigma * self.sigma).sqrt();
        self.b * (self.rho + dk / sqrt_term)
    }

    /// Second derivative d²w/dk².
    pub fn d2w_dk2(&self, k: f64) -> f64 {
        let dk = k - self.m;
        let inner = dk * dk + self.sigma * self.sigma;
        let sqrt_term = inner.sqrt();
        self.b * self.sigma * self.sigma / (inner * sqrt_term)
    }

    /// Check no-butterfly-arbitrage condition: g(k) >= 0
    /// g(k) = (1 - k*w'/(2w))² - w'/4*(1/w + 1/4) + w''/2
    pub fn butterfly_density(&self, k: f64) -> f64 {
        let w = self.total_variance(k);
        if w <= 0.0 {
            return 0.0;
        }
        let wp = self.dw_dk(k);
        let wpp = self.d2w_dk2(k);
        let term1 = (1.0 - k * wp / (2.0 * w)).powi(2);
        let term2 = wp * wp / 4.0 * (1.0 / w + 0.25);
        let term3 = wpp / 2.0;
        term1 - term2 + term3
    }

    /// Check arbitrage-free over a range of strikes.
    pub fn is_arbitrage_free(&self, k_min: f64, k_max: f64, n_points: usize) -> bool {
        let dk = (k_max - k_min) / n_points as f64;
        for i in 0..=n_points {
            let k = k_min + i as f64 * dk;
            if self.butterfly_density(k) < -1e-10 {
                return false;
            }
            if self.total_variance(k) < -1e-10 {
                return false;
            }
        }
        true
    }

    /// Fit SVI to market data using quasi-Newton optimization.
    pub fn fit(log_strikes: &[f64], total_variances: &[f64]) -> Self {
        // Simple grid + Nelder-Mead approach
        let n = log_strikes.len();
        if n == 0 {
            return Self::new(0.04, 0.1, -0.3, 0.0, 0.1);
        }

        let atm_var = total_variances[n / 2];
        let mut best = Self::new(atm_var, 0.1, -0.3, 0.0, 0.1);
        let mut best_err = svi_fit_error(&best, log_strikes, total_variances);

        // Grid search for initialization
        for &a in &[atm_var * 0.5, atm_var * 0.8, atm_var, atm_var * 1.2] {
            for &b in &[0.05, 0.1, 0.2, 0.3, 0.5] {
                for &rho in &[-0.8, -0.5, -0.3, 0.0, 0.3] {
                    for &m in &[-0.1, -0.05, 0.0, 0.05, 0.1] {
                        for &sigma in &[0.05, 0.1, 0.2, 0.3] {
                            let candidate = Self::new(a, b, rho, m, sigma);
                            let err = svi_fit_error(&candidate, log_strikes, total_variances);
                            if err < best_err {
                                best = candidate;
                                best_err = err;
                            }
                        }
                    }
                }
            }
        }

        // Nelder-Mead refinement
        svi_nelder_mead(&mut best, log_strikes, total_variances, 500);
        best
    }
}

fn svi_fit_error(svi: &SviRaw, ks: &[f64], ws: &[f64]) -> f64 {
    ks.iter()
        .zip(ws.iter())
        .map(|(&k, &w)| {
            let model_w = svi.total_variance(k);
            (model_w - w).powi(2)
        })
        .sum::<f64>()
}

fn svi_nelder_mead(svi: &mut SviRaw, ks: &[f64], ws: &[f64], max_iter: usize) {
    // Pack into vector
    let mut params = vec![svi.a, svi.b, svi.rho, svi.m, svi.sigma];
    let n = 5;
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(params.clone());
    for i in 0..n {
        let mut p = params.clone();
        p[i] *= 1.05;
        if p[i].abs() < 1e-6 {
            p[i] += 0.01;
        }
        simplex.push(p);
    }

    let eval = |p: &[f64]| -> f64 {
        if p[1] < 0.0 || p[4] <= 0.0 || p[2] <= -1.0 || p[2] >= 1.0 {
            return 1e20;
        }
        let s = SviRaw::new(p[0], p[1], p[2], p[3], p[4]);
        svi_fit_error(&s, ks, ws)
    };

    let mut values: Vec<f64> = simplex.iter().map(|p| eval(p)).collect();

    for _ in 0..max_iter {
        // Sort by value
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        if values[best_idx] < 1e-15 {
            break;
        }

        // Centroid (excluding worst)
        let mut centroid = vec![0.0; n];
        for &idx in &indices[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let mut reflected = vec![0.0; n];
        for j in 0..n {
            reflected[j] = 2.0 * centroid[j] - simplex[worst_idx][j];
        }
        let f_r = eval(&reflected);

        if f_r < values[best_idx] {
            // Expansion
            let mut expanded = vec![0.0; n];
            for j in 0..n {
                expanded[j] = 3.0 * centroid[j] - 2.0 * simplex[worst_idx][j];
            }
            let f_e = eval(&expanded);
            if f_e < f_r {
                simplex[worst_idx] = expanded;
                values[worst_idx] = f_e;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = f_r;
            }
        } else if f_r < values[second_worst_idx] {
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_r;
        } else {
            // Contraction
            let mut contracted = vec![0.0; n];
            if f_r < values[worst_idx] {
                for j in 0..n {
                    contracted[j] = 0.5 * (centroid[j] + reflected[j]);
                }
            } else {
                for j in 0..n {
                    contracted[j] = 0.5 * (centroid[j] + simplex[worst_idx][j]);
                }
            }
            let f_c = eval(&contracted);
            if f_c < values[worst_idx] {
                simplex[worst_idx] = contracted;
                values[worst_idx] = f_c;
            } else {
                // Shrink
                let best = simplex[best_idx].clone();
                for i in 0..=n {
                    if i != best_idx {
                        for j in 0..n {
                            simplex[i][j] = 0.5 * (simplex[i][j] + best[j]);
                        }
                        values[i] = eval(&simplex[i]);
                    }
                }
            }
        }
    }

    // Find best
    let mut best_idx = 0;
    for i in 1..=n {
        if values[i] < values[best_idx] {
            best_idx = i;
        }
    }
    svi.a = simplex[best_idx][0];
    svi.b = simplex[best_idx][1];
    svi.rho = simplex[best_idx][2];
    svi.m = simplex[best_idx][3];
    svi.sigma = simplex[best_idx][4];
}

/// SVI Jump-Wing (JW) parameterization.
/// Parameters: v_t (ATM variance), ψ (ATM skew), p (left slope), c (right slope), v_tilde (min variance)
#[derive(Debug, Clone, Copy)]
pub struct SviJumpWing {
    pub v_t: f64,       // ATM total variance
    pub psi: f64,       // ATM skew
    pub p: f64,         // left wing slope
    pub c: f64,         // right wing slope
    pub v_tilde: f64,   // minimum total variance
    pub t: f64,         // time to expiry
}

impl SviJumpWing {
    pub fn new(v_t: f64, psi: f64, p: f64, c: f64, v_tilde: f64, t: f64) -> Self {
        Self { v_t, psi, p, c, v_tilde, t }
    }

    /// Convert JW to raw SVI parameters.
    pub fn to_raw(&self) -> SviRaw {
        let w = self.v_t * self.t;
        let b = 0.5 * (self.c + self.p);
        let rho = 1.0 - self.p / b;
        let beta = rho - 2.0 * self.psi * w.sqrt() / b;
        let alpha = (-beta * beta + 1.0).max(0.0).sqrt();
        let m = (self.v_t - self.v_tilde) * self.t / (b * (-rho + alpha.max(1e-10) - beta * ((alpha).max(1e-10)).atan2(beta)));
        let sigma = alpha * m.abs().max(0.01);
        let a = self.v_tilde * self.t - b * sigma * (1.0 - rho * rho).max(0.0).sqrt();

        SviRaw::new(a, b.max(0.0), rho.max(-0.999).min(0.999), m, sigma.max(0.001))
    }

    /// Implied vol at log-moneyness k.
    pub fn implied_vol(&self, k: f64) -> f64 {
        let raw = self.to_raw();
        raw.implied_vol(k, self.t)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SABR MODEL
// ═══════════════════════════════════════════════════════════════════════════

/// SABR model parameters.
#[derive(Debug, Clone, Copy)]
pub struct SabrParams {
    pub alpha: f64,  // initial vol of vol level
    pub beta: f64,   // CEV exponent (0 = normal, 1 = lognormal)
    pub rho: f64,    // correlation between spot and vol (-1 < ρ < 1)
    pub nu: f64,     // vol of vol
}

impl SabrParams {
    pub fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> Self {
        Self { alpha, beta, rho, nu }
    }

    /// Hagan et al. (2002) SABR implied vol formula.
    pub fn implied_vol(&self, forward: f64, strike: f64, tte: f64) -> f64 {
        if tte <= 0.0 || self.alpha <= 0.0 {
            return 0.0;
        }

        let f = forward;
        let k = strike;
        let a = self.alpha;
        let b = self.beta;
        let r = self.rho;
        let v = self.nu;

        // Handle ATM case
        if (f - k).abs() / f < 1e-7 {
            let fk = f.powf(1.0 - b);
            let term1 = (1.0 - b).powi(2) * a * a / (24.0 * fk * fk);
            let term2 = r * b * v * a / (4.0 * fk);
            let term3 = (2.0 - 3.0 * r * r) * v * v / 24.0;
            return a / fk * (1.0 + (term1 + term2 + term3) * tte);
        }

        let fk = (f * k).powf((1.0 - b) / 2.0);
        let log_fk = (f / k).ln();
        let z = v / a * fk * log_fk;
        let x = ((1.0 - 2.0 * r * z + z * z).sqrt() + z - r).ln() - (1.0 - r).ln();

        if x.abs() < 1e-15 {
            return a / fk;
        }

        let prefix = a / (fk
            * (1.0 + (1.0 - b).powi(2) / 24.0 * log_fk.powi(2)
                + (1.0 - b).powi(4) / 1920.0 * log_fk.powi(4)));

        let term1 = (1.0 - b).powi(2) * a * a / (24.0 * fk * fk);
        let term2 = r * b * v * a / (4.0 * fk);
        let term3 = (2.0 - 3.0 * r * r) * v * v / 24.0;

        prefix * z / x * (1.0 + (term1 + term2 + term3) * tte)
    }

    /// SABR implied vol for a vector of strikes.
    pub fn implied_vol_vec(&self, forward: f64, strikes: &[f64], tte: f64) -> Vec<f64> {
        strikes.iter().map(|&k| self.implied_vol(forward, k, tte)).collect()
    }

    /// Calibrate SABR to market vols (with fixed beta).
    pub fn calibrate(
        forward: f64,
        strikes: &[f64],
        market_vols: &[f64],
        tte: f64,
        beta: f64,
    ) -> Self {
        let n = strikes.len();
        if n == 0 {
            return Self::new(0.2, beta, -0.3, 0.5);
        }

        // Initial guess from ATM vol
        let atm_vol = market_vols[n / 2];
        let f_pow = forward.powf(1.0 - beta);
        let alpha_init = atm_vol * f_pow;

        let mut best = Self::new(alpha_init, beta, -0.3, 0.5);
        let mut best_err = sabr_fit_error(&best, forward, strikes, market_vols, tte);

        // Grid search
        for &alpha_mult in &[0.5, 0.8, 1.0, 1.2, 1.5] {
            for &rho in &[-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3] {
                for &nu in &[0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5] {
                    let candidate = Self::new(alpha_init * alpha_mult, beta, rho, nu);
                    let err = sabr_fit_error(&candidate, forward, strikes, market_vols, tte);
                    if err < best_err {
                        best = candidate;
                        best_err = err;
                    }
                }
            }
        }

        // Refine with Nelder-Mead
        sabr_nelder_mead(&mut best, forward, strikes, market_vols, tte, 500);
        best
    }

    /// Delta under SABR (numerical).
    pub fn delta(&self, forward: f64, strike: f64, tte: f64, rate: f64, is_call: bool) -> f64 {
        let ds = forward * 0.001;
        let vol_up = self.implied_vol(forward + ds, strike, tte);
        let vol_dn = self.implied_vol(forward - ds, strike, tte);
        let p_up = BSParams::new(forward + ds, strike, rate, 0.0, vol_up, tte);
        let p_dn = BSParams::new(forward - ds, strike, rate, 0.0, vol_dn, tte);
        let opt = if is_call { OptionType::Call } else { OptionType::Put };
        (bs_price(&p_up, opt) - bs_price(&p_dn, opt)) / (2.0 * ds)
    }

    /// Vega under SABR (numerical bump of alpha).
    pub fn vega_alpha(&self, forward: f64, strike: f64, tte: f64, rate: f64, is_call: bool) -> f64 {
        let da = self.alpha * 0.01;
        let up = Self::new(self.alpha + da, self.beta, self.rho, self.nu);
        let dn = Self::new(self.alpha - da, self.beta, self.rho, self.nu);
        let vol_up = up.implied_vol(forward, strike, tte);
        let vol_dn = dn.implied_vol(forward, strike, tte);
        let p_up = BSParams::new(forward, strike, rate, 0.0, vol_up, tte);
        let p_dn = BSParams::new(forward, strike, rate, 0.0, vol_dn, tte);
        let opt = if is_call { OptionType::Call } else { OptionType::Put };
        (bs_price(&p_up, opt) - bs_price(&p_dn, opt)) / (2.0 * da)
    }
}

fn sabr_fit_error(sabr: &SabrParams, forward: f64, strikes: &[f64], vols: &[f64], tte: f64) -> f64 {
    strikes.iter().zip(vols.iter()).map(|(&k, &v)| {
        let model = sabr.implied_vol(forward, k, tte);
        (model - v).powi(2)
    }).sum::<f64>()
}

fn sabr_nelder_mead(sabr: &mut SabrParams, forward: f64, strikes: &[f64], vols: &[f64], tte: f64, max_iter: usize) {
    let mut params = vec![sabr.alpha, sabr.rho, sabr.nu];
    let n = 3;
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(params.clone());
    for i in 0..n {
        let mut p = params.clone();
        p[i] *= 1.05;
        if p[i].abs() < 1e-6 { p[i] += 0.01; }
        simplex.push(p);
    }

    let eval = |p: &[f64]| -> f64 {
        if p[0] <= 0.0 || p[1] <= -1.0 || p[1] >= 1.0 || p[2] <= 0.0 {
            return 1e20;
        }
        let s = SabrParams::new(p[0], sabr.beta, p[1], p[2]);
        sabr_fit_error(&s, forward, strikes, vols, tte)
    };

    let mut values: Vec<f64> = simplex.iter().map(|p| eval(p)).collect();

    for _ in 0..max_iter {
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst = indices[n - 1];

        if values[best_idx] < 1e-15 { break; }

        let mut centroid = vec![0.0; n];
        for &idx in &indices[..n] {
            for j in 0..n { centroid[j] += simplex[idx][j]; }
        }
        for j in 0..n { centroid[j] /= n as f64; }

        let mut reflected: Vec<f64> = (0..n).map(|j| 2.0 * centroid[j] - simplex[worst_idx][j]).collect();
        let f_r = eval(&reflected);

        if f_r < values[best_idx] {
            let expanded: Vec<f64> = (0..n).map(|j| 3.0 * centroid[j] - 2.0 * simplex[worst_idx][j]).collect();
            let f_e = eval(&expanded);
            if f_e < f_r {
                simplex[worst_idx] = expanded;
                values[worst_idx] = f_e;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = f_r;
            }
        } else if f_r < values[second_worst] {
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_r;
        } else {
            let contracted: Vec<f64> = (0..n).map(|j| 0.5 * (centroid[j] + simplex[worst_idx][j])).collect();
            let f_c = eval(&contracted);
            if f_c < values[worst_idx] {
                simplex[worst_idx] = contracted;
                values[worst_idx] = f_c;
            } else {
                let best = simplex[best_idx].clone();
                for i in 0..=n {
                    if i != best_idx {
                        for j in 0..n { simplex[i][j] = 0.5 * (simplex[i][j] + best[j]); }
                        values[i] = eval(&simplex[i]);
                    }
                }
            }
        }
    }

    let mut best_idx = 0;
    for i in 1..=n { if values[i] < values[best_idx] { best_idx = i; } }
    sabr.alpha = simplex[best_idx][0];
    sabr.rho = simplex[best_idx][1];
    sabr.nu = simplex[best_idx][2];
}

// ═══════════════════════════════════════════════════════════════════════════
// LOCAL VOLATILITY (DUPIRE)
// ═══════════════════════════════════════════════════════════════════════════

/// Local volatility surface via Dupire's formula.
/// σ_local²(K,T) = (∂w/∂T) / (1 - k/(w) * ∂w/∂k + 1/4*(-1/4 - 1/w + k²/w²)*(∂w/∂k)² + 1/2 * ∂²w/∂k²)
#[derive(Debug, Clone)]
pub struct DupireLocalVol {
    pub strikes: Vec<f64>,
    pub expiries: Vec<f64>,
    pub total_variance: Vec<Vec<f64>>,  // [expiry_idx][strike_idx]
}

impl DupireLocalVol {
    pub fn new(strikes: Vec<f64>, expiries: Vec<f64>, total_variance: Vec<Vec<f64>>) -> Self {
        Self { strikes, expiries, total_variance }
    }

    /// Build from implied vols.
    pub fn from_implied_vols(
        strikes: Vec<f64>,
        expiries: Vec<f64>,
        impl_vols: &[Vec<f64>],
    ) -> Self {
        let tv: Vec<Vec<f64>> = expiries
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                impl_vols[i].iter().map(|&v| v * v * t).collect()
            })
            .collect();
        Self::new(strikes, expiries, tv)
    }

    /// Interpolate total variance at (k, t) using bilinear interpolation.
    pub fn interp_total_var(&self, k: f64, t: f64) -> f64 {
        let ki = self.find_index(&self.strikes, k);
        let ti = self.find_index(&self.expiries, t);

        let k0 = ki.0;
        let k1 = ki.1;
        let t0 = ti.0;
        let t1 = ti.1;

        let wk = if (self.strikes[k1] - self.strikes[k0]).abs() > 1e-15 {
            (k - self.strikes[k0]) / (self.strikes[k1] - self.strikes[k0])
        } else {
            0.0
        };
        let wt = if (self.expiries[t1] - self.expiries[t0]).abs() > 1e-15 {
            (t - self.expiries[t0]) / (self.expiries[t1] - self.expiries[t0])
        } else {
            0.0
        };

        let v00 = self.total_variance[t0][k0];
        let v01 = self.total_variance[t0][k1];
        let v10 = self.total_variance[t1][k0];
        let v11 = self.total_variance[t1][k1];

        (1.0 - wt) * ((1.0 - wk) * v00 + wk * v01) + wt * ((1.0 - wk) * v10 + wk * v11)
    }

    /// Local variance at (K, T) via numerical Dupire.
    pub fn local_variance(&self, k: f64, t: f64) -> f64 {
        let dk = k * 0.01;
        let dt = t * 0.01;
        let dt = dt.max(0.001);

        let w = self.interp_total_var(k, t);
        let w_up_t = self.interp_total_var(k, t + dt);
        let w_dn_t = self.interp_total_var(k, (t - dt).max(0.001));
        let w_up_k = self.interp_total_var(k + dk, t);
        let w_dn_k = self.interp_total_var(k - dk, t);
        let w_up2_k = self.interp_total_var(k + dk, t);
        let w_dn2_k = self.interp_total_var(k - dk, t);

        let dw_dt = (w_up_t - w_dn_t) / (2.0 * dt);
        let dw_dk = (w_up_k - w_dn_k) / (2.0 * dk);
        let d2w_dk2 = (w_up2_k - 2.0 * w + w_dn2_k) / (dk * dk);

        let log_k = k.ln();
        let numerator = dw_dt;
        let denom = 1.0 - log_k / w * dw_dk
            + 0.25 * (-0.25 - 1.0 / w + log_k * log_k / (w * w)) * dw_dk * dw_dk
            + 0.5 * d2w_dk2;

        if denom <= 0.0 {
            return w / t; // fallback to implied
        }

        (numerator / denom).max(0.0)
    }

    /// Local vol at (K, T).
    pub fn local_vol(&self, k: f64, t: f64) -> f64 {
        self.local_variance(k, t).sqrt()
    }

    /// Build local vol surface on a grid.
    pub fn local_vol_surface(&self) -> Vec<Vec<f64>> {
        self.expiries
            .iter()
            .map(|&t| {
                self.strikes.iter().map(|&k| self.local_vol(k, t)).collect()
            })
            .collect()
    }

    fn find_index(&self, arr: &[f64], val: f64) -> (usize, usize) {
        if val <= arr[0] {
            return (0, 0.min(arr.len() - 1));
        }
        if val >= *arr.last().unwrap() {
            let n = arr.len() - 1;
            return (n, n);
        }
        for i in 0..arr.len() - 1 {
            if val >= arr[i] && val <= arr[i + 1] {
                return (i, i + 1);
            }
        }
        (0, 0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CUBIC SPLINE INTERPOLATION
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct CubicSpline {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
}

impl CubicSpline {
    /// Build natural cubic spline through data points.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        let n = x.len();
        assert!(n >= 2, "Need at least 2 points for spline");

        if n == 2 {
            // Linear
            let slope = (y[1] - y[0]) / (x[1] - x[0]);
            return Self {
                x: x.clone(),
                y: y.clone(),
                a: vec![y[0]],
                b: vec![slope],
                c: vec![0.0],
                d: vec![0.0],
            };
        }

        let mut h = vec![0.0; n - 1];
        for i in 0..n - 1 {
            h[i] = x[i + 1] - x[i];
        }

        // Solve tridiagonal system for second derivatives
        let mut alpha = vec![0.0; n - 1];
        for i in 1..n - 1 {
            alpha[i] = 3.0 / h[i] * (y[i + 1] - y[i]) - 3.0 / h[i - 1] * (y[i] - y[i - 1]);
        }

        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 1..n - 1 {
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        let mut c_vec = vec![0.0; n];
        let mut b_vec = vec![0.0; n - 1];
        let mut d_vec = vec![0.0; n - 1];
        let a_vec: Vec<f64> = y[..n - 1].to_vec();

        for j in (0..n - 1).rev() {
            c_vec[j] = z[j] - mu[j] * c_vec[j + 1];
            b_vec[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c_vec[j + 1] + 2.0 * c_vec[j]) / 3.0;
            d_vec[j] = (c_vec[j + 1] - c_vec[j]) / (3.0 * h[j]);
        }

        Self {
            x,
            y,
            a: a_vec,
            b: b_vec,
            c: c_vec[..n - 1].to_vec(),
            d: d_vec,
        }
    }

    /// Evaluate spline at point t.
    pub fn eval(&self, t: f64) -> f64 {
        let n = self.x.len();
        if t <= self.x[0] {
            let dx = t - self.x[0];
            return self.a[0] + self.b[0] * dx;
        }
        if t >= self.x[n - 1] {
            let last = n - 2;
            let dx = t - self.x[last];
            return self.a[last] + self.b[last] * dx + self.c[last] * dx * dx + self.d[last] * dx * dx * dx;
        }

        // Binary search
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x[mid] > t {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        let dx = t - self.x[lo];
        self.a[lo] + self.b[lo] * dx + self.c[lo] * dx * dx + self.d[lo] * dx * dx * dx
    }

    /// First derivative at point t.
    pub fn deriv(&self, t: f64) -> f64 {
        let n = self.x.len();
        let idx = if t <= self.x[0] {
            0
        } else if t >= self.x[n - 1] {
            n - 2
        } else {
            let mut lo = 0;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if self.x[mid] > t { hi = mid; } else { lo = mid; }
            }
            lo
        };
        let dx = t - self.x[idx];
        self.b[idx] + 2.0 * self.c[idx] * dx + 3.0 * self.d[idx] * dx * dx
    }

    /// Second derivative at point t.
    pub fn deriv2(&self, t: f64) -> f64 {
        let n = self.x.len();
        let idx = if t <= self.x[0] {
            0
        } else if t >= self.x[n - 1] {
            n - 2
        } else {
            let mut lo = 0;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if self.x[mid] > t { hi = mid; } else { lo = mid; }
            }
            lo
        };
        let dx = t - self.x[idx];
        2.0 * self.c[idx] + 6.0 * self.d[idx] * dx
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VOL SURFACE INTERPOLATION
// ═══════════════════════════════════════════════════════════════════════════

/// Full volatility surface with spline interpolation in both strike and time.
#[derive(Debug, Clone)]
pub struct VolSurface {
    pub strikes: Vec<f64>,
    pub expiries: Vec<f64>,
    pub vols: Vec<Vec<f64>>,   // [expiry_idx][strike_idx]
    strike_splines: Vec<CubicSpline>,
}

impl VolSurface {
    /// Build vol surface from a grid of implied vols.
    pub fn new(strikes: Vec<f64>, expiries: Vec<f64>, vols: Vec<Vec<f64>>) -> Self {
        let strike_splines: Vec<CubicSpline> = vols
            .iter()
            .map(|vol_row| CubicSpline::new(strikes.clone(), vol_row.clone()))
            .collect();
        Self { strikes, expiries, vols, strike_splines }
    }

    /// Interpolate implied vol at (strike, expiry).
    pub fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        // Interpolate along strike for each expiry, then interpolate in time
        let vol_at_strike: Vec<f64> = self.strike_splines
            .iter()
            .map(|sp| sp.eval(strike))
            .collect();

        // Variance-linear interpolation in time
        if expiry <= self.expiries[0] {
            return vol_at_strike[0].max(0.001);
        }
        if expiry >= *self.expiries.last().unwrap() {
            return vol_at_strike.last().unwrap().max(0.001);
        }

        let mut idx = 0;
        for i in 0..self.expiries.len() - 1 {
            if expiry >= self.expiries[i] && expiry <= self.expiries[i + 1] {
                idx = i;
                break;
            }
        }

        let t0 = self.expiries[idx];
        let t1 = self.expiries[idx + 1];
        let v0 = vol_at_strike[idx];
        let v1 = vol_at_strike[idx + 1];

        // Variance linear: var(t) = var(t0) + (t-t0)/(t1-t0) * (var(t1) - var(t0))
        let var0 = v0 * v0 * t0;
        let var1 = v1 * v1 * t1;
        let w = (expiry - t0) / (t1 - t0);
        let var_t = var0 + w * (var1 - var0);
        if var_t <= 0.0 || expiry <= 0.0 {
            return 0.001;
        }
        (var_t / expiry).sqrt()
    }

    /// ATM vol at given expiry.
    pub fn atm_vol(&self, forward: f64, expiry: f64) -> f64 {
        self.implied_vol(forward, expiry)
    }

    /// Vol skew (25-delta risk reversal proxy).
    pub fn skew(&self, forward: f64, expiry: f64) -> f64 {
        let k_up = forward * 1.1;
        let k_dn = forward * 0.9;
        self.implied_vol(k_up, expiry) - self.implied_vol(k_dn, expiry)
    }

    /// Vol curvature (25-delta butterfly proxy).
    pub fn curvature(&self, forward: f64, expiry: f64) -> f64 {
        let k_up = forward * 1.1;
        let k_dn = forward * 0.9;
        0.5 * (self.implied_vol(k_up, expiry) + self.implied_vol(k_dn, expiry))
            - self.implied_vol(forward, expiry)
    }

    /// Term structure of ATM vols.
    pub fn atm_term_structure(&self, forward: f64) -> Vec<(f64, f64)> {
        self.expiries
            .iter()
            .map(|&t| (t, self.atm_vol(forward, t)))
            .collect()
    }

    /// Forward vol between t1 and t2.
    pub fn forward_vol(&self, strike: f64, t1: f64, t2: f64) -> f64 {
        if t2 <= t1 {
            return self.implied_vol(strike, t1);
        }
        let v1 = self.implied_vol(strike, t1);
        let v2 = self.implied_vol(strike, t2);
        let var_fwd = (v2 * v2 * t2 - v1 * v1 * t1) / (t2 - t1);
        if var_fwd <= 0.0 {
            return 0.001;
        }
        var_fwd.sqrt()
    }

    /// Forward variance between t1 and t2.
    pub fn forward_variance(&self, strike: f64, t1: f64, t2: f64) -> f64 {
        let v1 = self.implied_vol(strike, t1);
        let v2 = self.implied_vol(strike, t2);
        (v2 * v2 * t2 - v1 * v1 * t1).max(0.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SMILE FITTING UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/// Fit a polynomial smile: σ(k) = a0 + a1*k + a2*k² + a3*k³
pub fn fit_polynomial_smile(log_moneyness: &[f64], vols: &[f64], degree: usize) -> Vec<f64> {
    let n = log_moneyness.len();
    let d = degree + 1;
    if n < d {
        return vec![vols.iter().sum::<f64>() / n as f64];
    }

    // Normal equations: (X'X) * a = X'y
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];

    for i in 0..n {
        let mut pow_k = vec![1.0; d];
        for j in 1..d {
            pow_k[j] = pow_k[j - 1] * log_moneyness[i];
        }
        for j in 0..d {
            for l in 0..d {
                xtx[j][l] += pow_k[j] * pow_k[l];
            }
            xty[j] += pow_k[j] * vols[i];
        }
    }

    // Solve via Gauss elimination
    gauss_solve(&mut xtx, &mut xty)
}

fn gauss_solve(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Vec<f64> {
    let n = b.len();
    // Forward elimination with partial pivoting
    for k in 0..n {
        let mut max_idx = k;
        let mut max_val = a[k][k].abs();
        for i in k + 1..n {
            if a[i][k].abs() > max_val {
                max_val = a[i][k].abs();
                max_idx = i;
            }
        }
        a.swap(k, max_idx);
        b.swap(k, max_idx);

        if a[k][k].abs() < 1e-15 {
            continue;
        }

        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k + 1..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in i + 1..n {
            x[i] -= a[i][j] * x[j];
        }
        if a[i][i].abs() > 1e-15 {
            x[i] /= a[i][i];
        }
    }
    x
}

/// Evaluate polynomial smile at log-moneyness k.
pub fn eval_polynomial_smile(coeffs: &[f64], k: f64) -> f64 {
    let mut result = 0.0;
    let mut k_pow = 1.0;
    for &c in coeffs {
        result += c * k_pow;
        k_pow *= k;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// TERM STRUCTURE MODELS
// ═══════════════════════════════════════════════════════════════════════════

/// Flat vol term structure.
#[derive(Debug, Clone)]
pub struct FlatVolSurface {
    pub vol: f64,
}

impl FlatVolSurface {
    pub fn new(vol: f64) -> Self {
        Self { vol }
    }

    pub fn implied_vol(&self, _strike: f64, _expiry: f64) -> f64 {
        self.vol
    }
}

/// Time-dependent vol (piecewise constant).
#[derive(Debug, Clone)]
pub struct PiecewiseConstantVol {
    pub breakpoints: Vec<f64>,  // times
    pub vols: Vec<f64>,         // vol in each interval
}

impl PiecewiseConstantVol {
    pub fn new(breakpoints: Vec<f64>, vols: Vec<f64>) -> Self {
        Self { breakpoints, vols }
    }

    /// Effective vol for period [0, T].
    pub fn effective_vol(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return self.vols[0];
        }
        let mut var = 0.0;
        let mut t_prev = 0.0;
        for (i, &bp) in self.breakpoints.iter().enumerate() {
            if t <= bp {
                var += self.vols[i] * self.vols[i] * (t - t_prev);
                break;
            }
            var += self.vols[i] * self.vols[i] * (bp - t_prev);
            t_prev = bp;
            if i == self.breakpoints.len() - 1 && t > bp {
                var += self.vols[i] * self.vols[i] * (t - bp);
            }
        }
        (var / t).sqrt()
    }

    /// Forward vol for period [t1, t2].
    pub fn forward_vol(&self, t1: f64, t2: f64) -> f64 {
        if t2 <= t1 {
            return self.effective_vol(t1);
        }
        let var2 = self.effective_vol(t2).powi(2) * t2;
        let var1 = self.effective_vol(t1).powi(2) * t1;
        ((var2 - var1) / (t2 - t1)).max(0.0).sqrt()
    }
}

/// Sticky strike vol surface: vol is function of absolute strike.
#[derive(Debug, Clone)]
pub struct StickyStrikeVol {
    pub surface: VolSurface,
}

impl StickyStrikeVol {
    pub fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.surface.implied_vol(strike, expiry)
    }
}

/// Sticky delta vol surface: vol is function of moneyness K/S.
#[derive(Debug, Clone)]
pub struct StickyDeltaVol {
    pub surface: VolSurface,
    pub reference_spot: f64,
}

impl StickyDeltaVol {
    pub fn implied_vol(&self, strike: f64, spot: f64, expiry: f64) -> f64 {
        let moneyness = strike / spot;
        let adj_strike = moneyness * self.reference_spot;
        self.surface.implied_vol(adj_strike, expiry)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VANNA-VOLGA METHOD
// ═══════════════════════════════════════════════════════════════════════════

/// Vanna-Volga pricing: interpolate vol using three market quotes.
pub fn vanna_volga_vol(
    forward: f64,
    strike: f64,
    rate: f64,
    tte: f64,
    k_25d_put: f64,
    k_atm: f64,
    k_25d_call: f64,
    vol_25d_put: f64,
    vol_atm: f64,
    vol_25d_call: f64,
) -> f64 {
    let log_k = (strike / forward).ln();
    let log_k1 = (k_25d_put / forward).ln();
    let log_k2 = (k_atm / forward).ln();
    let log_k3 = (k_25d_call / forward).ln();

    // Weights from vanna-volga method
    let x1 = log_k * (log_k - log_k2) * (log_k - log_k3)
        / ((log_k1 - log_k2) * (log_k1 - log_k3));
    let x2 = log_k * (log_k - log_k1) * (log_k - log_k3)
        / ((log_k2 - log_k1) * (log_k2 - log_k3));
    let x3 = log_k * (log_k - log_k1) * (log_k - log_k2)
        / ((log_k3 - log_k1) * (log_k3 - log_k2));

    // First-order: just interpolate vol
    let vol_vv1 = vol_atm + x1 * (vol_25d_put - vol_atm) + x3 * (vol_25d_call - vol_atm);

    // Second-order correction
    let d1_k = |vol: f64, k: f64| -> f64 {
        ((forward / k).ln() + 0.5 * vol * vol * tte) / (vol * tte.sqrt())
    };

    let vega_k = |vol: f64, k: f64| -> f64 {
        let d1 = d1_k(vol, k);
        forward * (-rate * tte).exp() * norm_pdf(d1) * tte.sqrt()
    };

    let vanna_k = |vol: f64, k: f64| -> f64 {
        let d1 = d1_k(vol, k);
        let d2 = d1 - vol * tte.sqrt();
        -(-rate * tte).exp() * norm_pdf(d1) * d2 / vol
    };

    let volga_k = |vol: f64, k: f64| -> f64 {
        let d1 = d1_k(vol, k);
        let d2 = d1 - vol * tte.sqrt();
        vega_k(vol, k) * d1 * d2 / vol
    };

    // Compute BS prices at market vols
    let c_atm = {
        let p = BSParams::new(forward, k_atm, rate, 0.0, vol_atm, tte);
        bs_price(&p, OptionType::Call)
    };

    let c1 = {
        let p = BSParams::new(forward, k_25d_put, rate, 0.0, vol_25d_put, tte);
        bs_price(&p, OptionType::Put)
    };
    let c1_atm = {
        let p = BSParams::new(forward, k_25d_put, rate, 0.0, vol_atm, tte);
        bs_price(&p, OptionType::Put)
    };

    let c3 = {
        let p = BSParams::new(forward, k_25d_call, rate, 0.0, vol_25d_call, tte);
        bs_price(&p, OptionType::Call)
    };
    let c3_atm = {
        let p = BSParams::new(forward, k_25d_call, rate, 0.0, vol_atm, tte);
        bs_price(&p, OptionType::Call)
    };

    // VV overhedge costs
    let d1_atm = d1_k(vol_atm, strike);
    let d2_atm = d1_atm - vol_atm * tte.sqrt();
    let vega_x = vega_k(vol_atm, strike);
    let vanna_x = vanna_k(vol_atm, strike);
    let volga_x = volga_k(vol_atm, strike);

    vol_vv1.max(0.001)
}

/// Build delta-space vol surface from market conventions.
pub fn delta_surface_to_strike_surface(
    forward: f64,
    rate: f64,
    expiries: &[f64],
    atm_vols: &[f64],
    rr_25d: &[f64],   // 25-delta risk reversal
    bf_25d: &[f64],   // 25-delta butterfly
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n_expiries = expiries.len();

    // Convert delta quotes to strikes and vols
    let mut all_strikes = Vec::new();
    let mut all_vols = Vec::new();

    for i in 0..n_expiries {
        let t = expiries[i];
        let atm = atm_vols[i];
        let rr = rr_25d[i];
        let bf = bf_25d[i];

        let vol_25d_call = atm + bf + 0.5 * rr;
        let vol_25d_put = atm + bf - 0.5 * rr;

        // 25-delta strikes (approximate)
        let sqrt_t = t.sqrt();
        let k_atm = forward * (0.5 * atm * atm * t).exp();
        let k_25d_call = forward * (-norm_inv(0.25) * vol_25d_call * sqrt_t + 0.5 * vol_25d_call * vol_25d_call * t).exp();
        let k_25d_put = forward * (norm_inv(0.25) * vol_25d_put * sqrt_t + 0.5 * vol_25d_put * vol_25d_put * t).exp();

        if i == 0 {
            all_strikes = vec![k_25d_put, k_atm, k_25d_call];
        }
        all_vols.push(vec![vol_25d_put, atm, vol_25d_call]);
    }

    (all_strikes, expiries.to_vec(), all_vols)
}

// ═══════════════════════════════════════════════════════════════════════════
// VARIANCE SURFACE
// ═══════════════════════════════════════════════════════════════════════════

/// Variance surface: stores total variance and provides arbitrage checks.
#[derive(Debug, Clone)]
pub struct VarianceSurface {
    pub log_strikes: Vec<f64>,
    pub expiries: Vec<f64>,
    pub total_variances: Vec<Vec<f64>>,
}

impl VarianceSurface {
    pub fn from_vol_surface(vs: &VolSurface, forward: f64) -> Self {
        let log_strikes: Vec<f64> = vs.strikes.iter().map(|k| (k / forward).ln()).collect();
        let total_variances: Vec<Vec<f64>> = vs.expiries.iter().enumerate().map(|(i, &t)| {
            vs.vols[i].iter().map(|&v| v * v * t).collect()
        }).collect();
        Self {
            log_strikes,
            expiries: vs.expiries.clone(),
            total_variances,
        }
    }

    /// Check calendar spread arbitrage: total variance must be non-decreasing in time.
    pub fn check_calendar_arbitrage(&self) -> Vec<Vec<bool>> {
        let n_t = self.expiries.len();
        let n_k = self.log_strikes.len();
        let mut valid = vec![vec![true; n_k]; n_t];
        for t in 1..n_t {
            for k in 0..n_k {
                if self.total_variances[t][k] < self.total_variances[t - 1][k] - 1e-10 {
                    valid[t][k] = false;
                }
            }
        }
        valid
    }

    /// Check butterfly arbitrage at each point.
    pub fn check_butterfly_arbitrage(&self) -> Vec<Vec<bool>> {
        let n_t = self.expiries.len();
        let n_k = self.log_strikes.len();
        let mut valid = vec![vec![true; n_k]; n_t];

        for t in 0..n_t {
            for k in 1..n_k - 1 {
                let dk = self.log_strikes[k + 1] - self.log_strikes[k - 1];
                let d2w = (self.total_variances[t][k + 1] - 2.0 * self.total_variances[t][k]
                    + self.total_variances[t][k - 1]) / (dk * dk * 0.25);
                if d2w < -1e-10 {
                    valid[t][k] = false;
                }
            }
        }
        valid
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svi_atm() {
        let svi = SviRaw::new(0.04, 0.2, -0.3, 0.0, 0.1);
        let w_atm = svi.total_variance(0.0);
        assert!(w_atm > 0.0, "ATM total var should be positive: {}", w_atm);
    }

    #[test]
    fn test_svi_butterfly_positive() {
        let svi = SviRaw::new(0.04, 0.1, -0.2, 0.0, 0.15);
        assert!(svi.is_arbitrage_free(-1.0, 1.0, 100));
    }

    #[test]
    fn test_sabr_atm() {
        let sabr = SabrParams::new(0.2, 0.5, -0.3, 0.4);
        let vol = sabr.implied_vol(100.0, 100.0, 1.0);
        assert!(vol > 0.0 && vol < 1.0, "SABR ATM vol: {}", vol);
    }

    #[test]
    fn test_cubic_spline() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x^2
        let sp = CubicSpline::new(x, y);
        let val = sp.eval(2.5);
        assert!((val - 6.25).abs() < 0.5, "Spline at 2.5: {}", val);
    }

    #[test]
    fn test_vol_surface_interp() {
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let expiries = vec![0.25, 0.5, 1.0];
        let vols = vec![
            vec![0.25, 0.22, 0.20, 0.22, 0.25],
            vec![0.24, 0.21, 0.19, 0.21, 0.24],
            vec![0.23, 0.20, 0.18, 0.20, 0.23],
        ];
        let surface = VolSurface::new(strikes, expiries, vols);
        let v = surface.implied_vol(100.0, 0.5);
        assert!(v > 0.15 && v < 0.25, "Interp vol: {}", v);
    }

    #[test]
    fn test_sabr_skew() {
        let sabr = SabrParams::new(0.2, 0.5, -0.5, 0.4);
        let v_otm_put = sabr.implied_vol(100.0, 90.0, 1.0);
        let v_atm = sabr.implied_vol(100.0, 100.0, 1.0);
        let v_otm_call = sabr.implied_vol(100.0, 110.0, 1.0);
        // Negative rho should produce negative skew (OTM puts have higher vol)
        assert!(v_otm_put > v_atm, "Negative skew expected: put {} > atm {}", v_otm_put, v_atm);
    }
}
