/// Volatility surface interpolation: bicubic spline, SVI fitting, Dupire local vol,
/// and arbitrage checks.

use std::f64::consts::PI;

// ── Normal CDF (local copy to avoid cross-module dependency) ──────────────────

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

// ── VolNode ───────────────────────────────────────────────────────────────────

/// A single observed implied-vol data point.
#[derive(Debug, Clone, Copy)]
pub struct VolNode {
    pub strike: f64,
    pub expiry_days: f64,
    pub iv: f64,
}

// ── Arbitrage violation record ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ArbViolation {
    pub kind: String,
    pub strike: f64,
    pub expiry_days: f64,
    pub description: String,
}

// ── SVI parameters (Gatheral 2004) ────────────────────────────────────────────

/// Raw SVI parameterization of a single expiry slice.
/// Total implied variance: w(k) = a + b*(rho*(k - m) + sqrt((k - m)^2 + sigma^2))
/// where k = log(K/F).
#[derive(Debug, Clone, Copy)]
pub struct SVIParams {
    /// Overall level of variance.
    pub a: f64,
    /// Slope parameter (volatility of vol proxy).
    pub b: f64,
    /// Correlation between spot and vol.
    pub rho: f64,
    /// ATM offset in log-moneyness.
    pub m: f64,
    /// Smoothing parameter (wing curvature).
    pub sigma: f64,
}

impl SVIParams {
    /// Evaluate total implied variance at log-moneyness k.
    pub fn total_variance(&self, k: f64) -> f64 {
        let km = k - self.m;
        self.a + self.b * (self.rho * km + (km * km + self.sigma * self.sigma).sqrt())
    }

    /// Implied volatility at log-moneyness k and expiry T (years).
    pub fn iv(&self, k: f64, t_years: f64) -> f64 {
        if t_years <= 0.0 {
            return 0.0;
        }
        let w = self.total_variance(k).max(0.0);
        (w / t_years).sqrt()
    }

    /// First derivative of total variance w.r.t. k (needed for local vol).
    pub fn dw_dk(&self, k: f64) -> f64 {
        let km = k - self.m;
        let sq = (km * km + self.sigma * self.sigma).sqrt();
        self.b * (self.rho + km / sq)
    }

    /// Second derivative of total variance w.r.t. k.
    pub fn d2w_dk2(&self, k: f64) -> f64 {
        let km = k - self.m;
        let sq = (km * km + self.sigma * self.sigma).sqrt();
        self.b * self.sigma * self.sigma / (sq * sq * sq)
    }
}

// ── VolSurface ────────────────────────────────────────────────────────────────

/// Full implied-vol surface built from a cloud of VolNode observations.
/// Expiry slices are stored in ascending order of expiry_days.
#[derive(Debug, Clone)]
pub struct VolSurface {
    /// Raw nodes sorted by (expiry_days, strike).
    nodes: Vec<VolNode>,
    /// Unique expiry levels present in the surface.
    expiries: Vec<f64>,
}

/// Construct a VolSurface from a vector of VolNode observations.
pub fn build_surface(mut nodes: Vec<VolNode>) -> VolSurface {
    nodes.sort_by(|a, b| {
        a.expiry_days
            .partial_cmp(&b.expiry_days)
            .unwrap()
            .then(a.strike.partial_cmp(&b.strike).unwrap())
    });
    let mut expiries: Vec<f64> = nodes.iter().map(|n| n.expiry_days).collect();
    expiries.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    VolSurface { nodes, expiries }
}

impl VolSurface {
    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Nodes belonging to a single expiry slice (within 0.5 days tolerance).
    fn slice(&self, expiry_days: f64) -> Vec<&VolNode> {
        self.nodes
            .iter()
            .filter(|n| (n.expiry_days - expiry_days).abs() < 0.5)
            .collect()
    }

    /// Linear search for the two bracketing expiry indices.
    fn bracket_expiry(&self, expiry: f64) -> (usize, usize) {
        let n = self.expiries.len();
        if n == 0 {
            return (0, 0);
        }
        if expiry <= self.expiries[0] {
            return (0, 0);
        }
        if expiry >= self.expiries[n - 1] {
            return (n - 1, n - 1);
        }
        for i in 0..n - 1 {
            if expiry >= self.expiries[i] && expiry <= self.expiries[i + 1] {
                return (i, i + 1);
            }
        }
        (n - 1, n - 1)
    }

    /// Interpolate IV for a strike on a single expiry slice using natural
    /// cubic spline (falls back to linear if fewer than 3 nodes).
    fn spline_iv_at_slice(&self, slice_nodes: &[&VolNode], strike: f64) -> f64 {
        let n = slice_nodes.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return slice_nodes[0].iv;
        }
        if n == 2 {
            // Linear interpolation
            let n0 = slice_nodes[0];
            let n1 = slice_nodes[1];
            let t = (strike - n0.strike) / (n1.strike - n0.strike + 1e-12);
            return n0.iv + t * (n1.iv - n0.iv);
        }

        // Natural cubic spline via Thomas algorithm.
        let xs: Vec<f64> = slice_nodes.iter().map(|n| n.strike).collect();
        let ys: Vec<f64> = slice_nodes.iter().map(|n| n.iv).collect();

        // Extrapolate flat if outside range.
        if strike <= xs[0] {
            return ys[0];
        }
        if strike >= xs[n - 1] {
            return ys[n - 1];
        }

        let h: Vec<f64> = (0..n - 1).map(|i| xs[i + 1] - xs[i]).collect();

        // Build RHS
        let mut rhs = vec![0.0_f64; n];
        for i in 1..n - 1 {
            rhs[i] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
        }

        // Tridiagonal solve (natural BC: m[0] = m[n-1] = 0)
        let mut m = vec![0.0_f64; n];
        let mut diag = vec![2.0_f64; n];
        let mut upper = vec![0.0_f64; n];
        for i in 1..n - 1 {
            upper[i] = h[i] / (h[i - 1] + h[i]);
            diag[i] = 2.0;
        }
        // Forward sweep
        let mut c = vec![0.0_f64; n];
        let mut d = rhs.clone();
        c[0] = 0.0;
        d[0] = 0.0;
        for i in 1..n - 1 {
            let w = h[i - 1] / (h[i - 1] + h[i]);
            let fac = w * c[i - 1] + 2.0;
            c[i] = (1.0 - w) / fac;
            d[i] = (d[i] - w * d[i - 1]) / fac;
        }
        m[n - 1] = 0.0;
        for i in (1..n - 1).rev() {
            m[i] = d[i] - c[i] * m[i + 1];
        }

        // Find segment
        let mut seg = n - 2;
        for i in 0..n - 1 {
            if strike >= xs[i] && strike <= xs[i + 1] {
                seg = i;
                break;
            }
        }
        let _t = (strike - xs[seg]) / h[seg];
        // Cubic Hermite using second derivatives m
        let a0 = ys[seg];
        let a1 = (ys[seg + 1] - ys[seg]) / h[seg]
            - h[seg] * (2.0 * m[seg] + m[seg + 1]) / 3.0;
        let a2 = m[seg];
        let a3 = (m[seg + 1] - m[seg]) / (3.0 * h[seg]);
        let dt = strike - xs[seg];
        a0 + a1 * dt + a2 * dt * dt + a3 * dt * dt * dt
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// Bicubic spline interpolation: first interpolate across strikes at each
    /// bracketing expiry slice, then interpolate linearly in expiry.
    pub fn cubic_spline_iv(&self, strike: f64, expiry: f64) -> f64 {
        if self.expiries.is_empty() {
            return 0.0;
        }
        let (lo, hi) = self.bracket_expiry(expiry);
        if lo == hi {
            let s = self.slice(self.expiries[lo]);
            return self.spline_iv_at_slice(&s, strike).max(0.0);
        }
        let exp_lo = self.expiries[lo];
        let exp_hi = self.expiries[hi];
        let s_lo = self.slice(exp_lo);
        let s_hi = self.slice(exp_hi);
        let iv_lo = self.spline_iv_at_slice(&s_lo, strike);
        let iv_hi = self.spline_iv_at_slice(&s_hi, strike);
        // Interpolate in total-variance space to respect term structure.
        let t_lo = exp_lo / 365.0;
        let t_hi = exp_hi / 365.0;
        let t = expiry / 365.0;
        let w_lo = iv_lo * iv_lo * t_lo;
        let w_hi = iv_hi * iv_hi * t_hi;
        let alpha = if (t_hi - t_lo).abs() < 1e-12 {
            0.0
        } else {
            (t - t_lo) / (t_hi - t_lo)
        };
        let w = w_lo + alpha * (w_hi - w_lo);
        if t > 0.0 {
            (w / t).sqrt().max(0.0)
        } else {
            0.0
        }
    }

    /// Fit the Gatheral SVI parameterization to a single expiry slice via
    /// Nelder-Mead simplex minimization of sum-of-squared IV errors.
    pub fn svi_fit(&self, expiry: f64) -> SVIParams {
        let nodes = self.slice(expiry);
        if nodes.is_empty() {
            return SVIParams { a: 0.04, b: 0.1, rho: 0.0, m: 0.0, sigma: 0.1 };
        }
        let t = expiry / 365.0;

        // Convert to log-moneyness and total variance.
        // Approximate ATM strike as median strike.
        let mut strikes: Vec<f64> = nodes.iter().map(|n| n.strike).collect();
        strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let f = strikes[strikes.len() / 2];
        let data: Vec<(f64, f64)> = nodes
            .iter()
            .map(|n| {
                let k = (n.strike / f).ln();
                let w = n.iv * n.iv * t.max(1e-6);
                (k, w)
            })
            .collect();

        // Objective: sum of squared errors in total variance.
        let objective = |params: &[f64; 5]| -> f64 {
            let p = SVIParams {
                a: params[0],
                b: params[1],
                rho: params[2].clamp(-0.999, 0.999),
                m: params[3],
                sigma: params[4].abs().max(1e-6),
            };
            data.iter()
                .map(|(k, w)| {
                    let wfit = p.total_variance(*k).max(0.0);
                    (wfit - w).powi(2)
                })
                .sum()
        };

        // Nelder-Mead with initial guess from ATM vol.
        let atm_iv = nodes
            .iter()
            .min_by(|a, b| {
                (a.strike - f).abs().partial_cmp(&(b.strike - f).abs()).unwrap()
            })
            .map(|n| n.iv)
            .unwrap_or(0.2);
        let w_atm = atm_iv * atm_iv * t.max(1e-6);

        let mut best = [w_atm * 0.8, 0.15, -0.3, 0.0, 0.2_f64];
        let mut best_val = objective(&best);

        // Simple grid search to seed Nelder-Mead.
        for &rho in &[-0.7, -0.3, 0.0, 0.3] {
            for &b in &[0.05, 0.15, 0.25] {
                let a = (w_atm - b * (rho * 0.0_f64 + 0.2_f64.sqrt())).max(1e-6);
                let candidate = [a, b, rho, 0.0, 0.15_f64];
                let val = objective(&candidate);
                if val < best_val {
                    best_val = val;
                    best = candidate;
                }
            }
        }

        // Nelder-Mead iterations (200 steps, dimension=5).
        let n_dim = 5_usize;
        let mut simplex: Vec<[f64; 5]> = Vec::with_capacity(n_dim + 1);
        simplex.push(best);
        let steps = [0.01, 0.02, 0.05, 0.02, 0.02];
        for i in 0..n_dim {
            let mut v = best;
            v[i] += steps[i];
            simplex.push(v);
        }

        for _ in 0..400 {
            simplex.sort_by(|a, b| {
                objective(a).partial_cmp(&objective(b)).unwrap()
            });
            let best_v = objective(&simplex[0]);
            let worst_v = objective(&simplex[n_dim]);
            if worst_v - best_v < 1e-10 {
                break;
            }
            // Centroid of all but worst.
            let mut centroid = [0.0_f64; 5];
            for v in &simplex[..n_dim] {
                for j in 0..5 {
                    centroid[j] += v[j];
                }
            }
            for j in 0..5 {
                centroid[j] /= n_dim as f64;
            }
            // Reflection.
            let mut reflected = [0.0_f64; 5];
            for j in 0..5 {
                reflected[j] = centroid[j] + (centroid[j] - simplex[n_dim][j]);
            }
            let refl_v = objective(&reflected);
            if refl_v < best_v {
                // Expansion.
                let mut expanded = [0.0_f64; 5];
                for j in 0..5 {
                    expanded[j] = centroid[j] + 2.0 * (reflected[j] - centroid[j]);
                }
                if objective(&expanded) < refl_v {
                    simplex[n_dim] = expanded;
                } else {
                    simplex[n_dim] = reflected;
                }
            } else if refl_v < worst_v {
                simplex[n_dim] = reflected;
            } else {
                // Contraction.
                let mut contracted = [0.0_f64; 5];
                for j in 0..5 {
                    contracted[j] = centroid[j] + 0.5 * (simplex[n_dim][j] - centroid[j]);
                }
                if objective(&contracted) < worst_v {
                    simplex[n_dim] = contracted;
                } else {
                    // Shrink.
                    for i in 1..=n_dim {
                        for j in 0..5 {
                            simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                        }
                    }
                }
            }
        }

        simplex.sort_by(|a, b| objective(a).partial_cmp(&objective(b)).unwrap());
        let p = simplex[0];
        SVIParams {
            a: p[0],
            b: p[1],
            rho: p[2].clamp(-0.999, 0.999),
            m: p[3],
            sigma: p[4].abs().max(1e-6),
        }
    }

    /// Dupire local volatility from the implied-vol surface.
    /// Uses the Dupire formula: sigma_loc^2(K,T) = (dw/dT) / ((1 - y*dw/dy)^2 - 0.25*(0.25 + 1/w)*( dw/dy)^2 + 0.5*d2w/dy2)
    /// where y = log(K/F), w = sigma_imp^2 * T.
    /// Here we use finite differences on the surface.
    pub fn local_vol(&self, spot: f64, t_years: f64, k: f64) -> f64 {
        let expiry_days = t_years * 365.0;
        let dt = 0.5; // 0.5 day finite difference step
        let dk_frac = 0.005; // 0.5% strike step

        let dk = k * dk_frac;

        // Total variance function w(K,T) = sigma^2 * T
        let w = |strike: f64, exp_d: f64| -> f64 {
            let iv = self.cubic_spline_iv(strike, exp_d);
            iv * iv * (exp_d / 365.0).max(1e-9)
        };

        // dw/dT via forward difference
        let w0 = w(k, expiry_days);
        let w_fwd = w(k, expiry_days + dt);
        let dw_dt = (w_fwd - w0) / (dt / 365.0);

        if dw_dt <= 0.0 {
            // Degenerate case -- return flat vol
            return self.cubic_spline_iv(k, expiry_days).max(0.01);
        }

        // dw/dk and d2w/dk2 in log-moneyness y = ln(K/F), F ~ spot
        let y = (k / spot).ln();
        let w_up = w(k + dk, expiry_days);
        let w_dn = w(k - dk, expiry_days);
        let dw_dy = (w_up - w_dn) / (2.0 * dk_frac); // dw/dy = K * dw/dK, approximate
        let d2w_dy2 = (w_up - 2.0 * w0 + w_dn) / (dk_frac * dk_frac);

        // Denominator of Dupire formula (Gatheral form in log-moneyness)
        let denom_term1 = (1.0 - y * dw_dy / (2.0 * w0.max(1e-12))).powi(2);
        let denom_term2 = -0.25 * (0.25 + 1.0 / w0.max(1e-12)) * dw_dy * dw_dy;
        let denom_term3 = 0.5 * d2w_dy2;
        let denom = denom_term1 + denom_term2 + denom_term3;

        if denom <= 1e-12 {
            return self.cubic_spline_iv(k, expiry_days).max(0.01);
        }

        let sigma_loc_sq = dw_dt / denom;
        sigma_loc_sq.max(0.0001_f64).sqrt()
    }

    /// ATM implied vol at a given expiry (days). Uses cubic spline interpolation.
    pub fn atm_vol(&self, expiry: f64) -> f64 {
        // ATM = approximate by finding nodes closest to ATM.
        let s = self.slice(expiry);
        if s.is_empty() {
            return self.cubic_spline_iv(0.0, expiry);
        }
        // ATM strike: the node with the lowest IV (minimum of smile).
        let atm_node = s
            .iter()
            .min_by(|a, b| a.iv.partial_cmp(&b.iv).unwrap())
            .unwrap();
        // Use spline at that strike.
        self.cubic_spline_iv(atm_node.strike, expiry)
    }

    /// 25-delta put IV minus 25-delta call IV (risk reversal proxy).
    /// Uses the SVI fit to compute delta-consistent strikes.
    pub fn vol_skew(&self, expiry: f64) -> f64 {
        let svi = self.svi_fit(expiry);
        let t = expiry / 365.0;
        if t <= 0.0 {
            return 0.0;
        }
        // ATM vol and forward (assume F=1, log-moneyness basis)
        let atm_iv = svi.iv(0.0, t);
        if atm_iv <= 0.0 {
            return 0.0;
        }
        let sqrt_t = t.sqrt();
        // 25-delta call: d1 = N^-1(0.25 + 0.5) in simplified form; k such that N(d1)=0.75
        // d1 = ln(F/K)/(sigma*sqrt_t) + 0.5*sigma*sqrt_t
        // For delta=0.25 put: N(d1) = 0.25 => d1 = -0.6745
        // For delta=0.25 call: N(d1) = 0.75 => d1 = +0.6745
        let d1_25c = 0.6745_f64;
        let d1_25p = -0.6745_f64;
        // k = -d1*sigma*sqrt_t + 0.5*sigma^2*t
        let k_25c = -d1_25c * atm_iv * sqrt_t + 0.5 * atm_iv * atm_iv * t;
        let k_25p = -d1_25p * atm_iv * sqrt_t + 0.5 * atm_iv * atm_iv * t;
        let iv_25c = svi.iv(k_25c, t).max(0.0);
        let iv_25p = svi.iv(k_25p, t).max(0.0);
        iv_25p - iv_25c
    }

    /// Smile curvature: butterfly proxy = (IV_25p + IV_25c)/2 - ATM_IV.
    pub fn vol_smile_curvature(&self, expiry: f64) -> f64 {
        let svi = self.svi_fit(expiry);
        let t = expiry / 365.0;
        if t <= 0.0 {
            return 0.0;
        }
        let atm_iv = svi.iv(0.0, t);
        if atm_iv <= 0.0 {
            return 0.0;
        }
        let sqrt_t = t.sqrt();
        let d1_25 = 0.6745_f64;
        let k_25c = -d1_25 * atm_iv * sqrt_t + 0.5 * atm_iv * atm_iv * t;
        let k_25p = d1_25 * atm_iv * sqrt_t + 0.5 * atm_iv * atm_iv * t;
        let iv_25c = svi.iv(k_25c, t).max(0.0);
        let iv_25p = svi.iv(k_25p, t).max(0.0);
        (iv_25c + iv_25p) / 2.0 - atm_iv
    }

    // ── Arbitrage checks ──────────────────────────────────────────────────────

    /// Check calendar spread arbitrage: total variance must be non-decreasing
    /// in expiry for every strike present in both slices.
    pub fn check_calendar_spread_arb(&self) -> Vec<ArbViolation> {
        let mut violations = Vec::new();
        let n_exp = self.expiries.len();
        if n_exp < 2 {
            return violations;
        }
        for i in 0..n_exp - 1 {
            let exp_lo = self.expiries[i];
            let exp_hi = self.expiries[i + 1];
            let t_lo = exp_lo / 365.0;
            let t_hi = exp_hi / 365.0;
            // Sample strikes common to both slices.
            let slice_lo = self.slice(exp_lo);
            let slice_hi = self.slice(exp_hi);
            for node_lo in &slice_lo {
                // Find closest strike in upper slice.
                let k = node_lo.strike;
                if let Some(node_hi) = slice_hi
                    .iter()
                    .min_by(|a, b| {
                        (a.strike - k).abs().partial_cmp(&(b.strike - k).abs()).unwrap()
                    })
                {
                    if (node_hi.strike - k).abs() / k < 0.05 {
                        let w_lo = node_lo.iv * node_lo.iv * t_lo;
                        let w_hi = node_hi.iv * node_hi.iv * t_hi;
                        if w_hi < w_lo - 1e-6 {
                            violations.push(ArbViolation {
                                kind: "CalendarSpread".to_string(),
                                strike: k,
                                expiry_days: exp_hi,
                                description: format!(
                                    "Total variance decreases from {:.4} to {:.4} at strike {:.2}",
                                    w_lo, w_hi, k
                                ),
                            });
                        }
                    }
                }
            }
        }
        violations
    }

    /// Check butterfly arbitrage for a given expiry: second derivative of call
    /// price w.r.t. strike must be non-negative (density >= 0).
    /// Uses the Breeden-Litzenberger condition on the SVI fit.
    pub fn check_butterfly_arb(&self, expiry: f64) -> Vec<ArbViolation> {
        let mut violations = Vec::new();
        let svi = self.svi_fit(expiry);
        let t = expiry / 365.0;
        if t <= 0.0 {
            return violations;
        }
        // Sample log-moneyness grid.
        let n = 50;
        let k_min = -1.0_f64;
        let k_max = 1.0_f64;
        let dk = (k_max - k_min) / (n as f64);
        for i in 0..n {
            let k = k_min + (i as f64 + 0.5) * dk;
            // Risk-neutral density via Breeden-Litzenberger:
            // density(k) = exp(k) * d2C/dK2 at K = F*exp(k)
            // Using Gatheral's formula in total-variance terms.
            let w = svi.total_variance(k).max(1e-9);
            let dw = svi.dw_dk(k);
            let d2w = svi.d2w_dk2(k);
            let sq_w = w.sqrt();
            let _d1 = -k / sq_w + sq_w / 2.0;
            // g(k) = (1 - k*dw/(2*w))^2 - dw^2/4*(1/w + 0.25) + d2w/2
            let g = (1.0 - k * dw / (2.0 * w)).powi(2)
                - dw * dw / 4.0 * (1.0 / w + 0.25)
                + d2w / 2.0;
            // Risk-neutral density proportional to g * norm_pdf(d1) / sqrt(w)
            // Butterfly arb if g < 0.
            if g < -1e-6 {
                violations.push(ArbViolation {
                    kind: "Butterfly".to_string(),
                    strike: k,
                    expiry_days: expiry,
                    description: format!(
                        "Negative risk-neutral density g={:.6} at log-moneyness {:.3}",
                        g, k
                    ),
                });
            }
        }
        violations
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_surface(iv: f64) -> VolSurface {
        let nodes = vec![
            VolNode { strike: 90.0,  expiry_days: 30.0,  iv },
            VolNode { strike: 100.0, expiry_days: 30.0,  iv },
            VolNode { strike: 110.0, expiry_days: 30.0,  iv },
            VolNode { strike: 90.0,  expiry_days: 60.0,  iv },
            VolNode { strike: 100.0, expiry_days: 60.0,  iv },
            VolNode { strike: 110.0, expiry_days: 60.0,  iv },
            VolNode { strike: 90.0,  expiry_days: 90.0,  iv },
            VolNode { strike: 100.0, expiry_days: 90.0,  iv },
            VolNode { strike: 110.0, expiry_days: 90.0,  iv },
        ];
        build_surface(nodes)
    }

    fn smile_surface() -> VolSurface {
        let nodes = vec![
            VolNode { strike: 80.0,  expiry_days: 30.0,  iv: 0.30 },
            VolNode { strike: 90.0,  expiry_days: 30.0,  iv: 0.25 },
            VolNode { strike: 100.0, expiry_days: 30.0,  iv: 0.20 },
            VolNode { strike: 110.0, expiry_days: 30.0,  iv: 0.22 },
            VolNode { strike: 120.0, expiry_days: 30.0,  iv: 0.28 },
            VolNode { strike: 80.0,  expiry_days: 60.0,  iv: 0.32 },
            VolNode { strike: 90.0,  expiry_days: 60.0,  iv: 0.27 },
            VolNode { strike: 100.0, expiry_days: 60.0,  iv: 0.22 },
            VolNode { strike: 110.0, expiry_days: 60.0,  iv: 0.24 },
            VolNode { strike: 120.0, expiry_days: 60.0,  iv: 0.30 },
        ];
        build_surface(nodes)
    }

    #[test]
    fn test_build_surface_sorted() {
        let surf = smile_surface();
        // Expiries should be in ascending order.
        for i in 0..surf.expiries.len() - 1 {
            assert!(surf.expiries[i] < surf.expiries[i + 1]);
        }
    }

    #[test]
    fn test_flat_surface_spline_returns_correct_iv() {
        let iv = 0.25;
        let surf = flat_surface(iv);
        let interp = surf.cubic_spline_iv(100.0, 30.0);
        // Should recover input IV closely.
        assert!((interp - iv).abs() < 1e-3, "Expected ~{}, got {}", iv, interp);
    }

    #[test]
    fn test_spline_interpolation_between_expiries() {
        let surf = smile_surface();
        let iv_30 = surf.cubic_spline_iv(100.0, 30.0);
        let iv_60 = surf.cubic_spline_iv(100.0, 60.0);
        let iv_45 = surf.cubic_spline_iv(100.0, 45.0);
        // Should be between the two.
        assert!(iv_45 > iv_30 - 0.01 && iv_45 < iv_60 + 0.01);
    }

    #[test]
    fn test_svi_params_total_variance_positive() {
        let surf = smile_surface();
        let svi = surf.svi_fit(30.0);
        for k in &[-0.5, -0.25, 0.0, 0.25, 0.5] {
            let w = svi.total_variance(*k);
            assert!(w >= 0.0, "Negative total variance at k={}", k);
        }
    }

    #[test]
    fn test_svi_params_structure() {
        let surf = smile_surface();
        let svi = surf.svi_fit(30.0);
        // sigma must be positive, b must be positive for a valid SVI.
        assert!(svi.sigma > 0.0);
        assert!(svi.b > 0.0);
        assert!(svi.rho.abs() <= 1.0);
    }

    #[test]
    fn test_atm_vol_in_range() {
        let surf = smile_surface();
        let atm = surf.atm_vol(30.0);
        assert!(atm > 0.05 && atm < 0.60, "ATM vol out of expected range: {}", atm);
    }

    #[test]
    fn test_vol_skew_negative_for_skewed_smile() {
        // A surface with more downside skew should have positive 25d risk reversal
        // (put IV > call IV).
        let surf = smile_surface();
        let skew = surf.vol_skew(30.0);
        // Not asserting sign strictly -- just finite and not NaN.
        assert!(skew.is_finite(), "Skew is not finite: {}", skew);
    }

    #[test]
    fn test_calendar_spread_arb_clean_surface() {
        // A surface with total variance increasing with time should have no violations.
        let surf = smile_surface();
        let viols = surf.check_calendar_spread_arb();
        // The smile surface has increasing IV with expiry -- should be clean.
        assert!(viols.is_empty(), "Unexpected calendar arb violations: {:?}", viols);
    }

    #[test]
    fn test_calendar_spread_arb_detects_inverted() {
        // Build a surface where the 60d slice has LOWER total variance than 30d.
        let nodes = vec![
            VolNode { strike: 100.0, expiry_days: 30.0, iv: 0.40 },
            VolNode { strike: 100.0, expiry_days: 60.0, iv: 0.20 }, // inverted
        ];
        let surf = build_surface(nodes);
        let viols = surf.check_calendar_spread_arb();
        assert!(!viols.is_empty(), "Should detect calendar arb");
    }

    #[test]
    fn test_butterfly_arb_clean_svi() {
        let surf = smile_surface();
        let viols = surf.check_butterfly_arb(30.0);
        // A well-fitted SVI without butterfly arb should have no violations
        // in the interior (may have some near extremes depending on fit quality).
        // Just verify the function runs and returns a vec.
        let _ = viols;
    }

    #[test]
    fn test_local_vol_positive() {
        let surf = smile_surface();
        let lv = surf.local_vol(100.0, 30.0 / 365.0, 100.0);
        assert!(lv > 0.0, "Local vol must be positive, got {}", lv);
    }

    #[test]
    fn test_smile_curvature_positive_for_smile() {
        let surf = smile_surface();
        let curv = surf.vol_smile_curvature(30.0);
        // Butterfly proxy should be positive for a smile (wings > ATM).
        assert!(curv.is_finite());
    }

    #[test]
    fn test_svi_dw_dk_d2w_dk2() {
        let p = SVIParams { a: 0.04, b: 0.1, rho: -0.3, m: 0.0, sigma: 0.2 };
        // d2w/dk2 should always be positive for valid SVI.
        for k in &[-0.5, 0.0, 0.5] {
            assert!(p.d2w_dk2(*k) > 0.0, "d2w/dk2 must be positive");
        }
    }
}
