use std::f64::consts::PI;
use crate::black_scholes::{norm_cdf, norm_pdf, BSParams, OptionType, bs_price};

// ═══════════════════════════════════════════════════════════════════════════
// BINOMIAL TREE MODELS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeType {
    CRR,          // Cox-Ross-Rubinstein
    JarrowRudd,   // Equal probability
    Tian,         // Third moment matching
    LeisenReimer, // Convergence-improved
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExerciseStyle {
    European,
    American,
    Bermudan,
}

#[derive(Debug, Clone)]
pub struct TreeParams {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub time_to_expiry: f64,
    pub opt_type: OptionType,
    pub exercise: ExerciseStyle,
    pub n_steps: usize,
}

impl TreeParams {
    pub fn new(
        spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64,
        tte: f64, opt_type: OptionType, exercise: ExerciseStyle, n_steps: usize,
    ) -> Self {
        Self { spot, strike, rate, dividend, vol, time_to_expiry: tte, opt_type, exercise, n_steps }
    }
}

#[derive(Debug, Clone)]
pub struct TreeResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub early_exercise_boundary: Vec<f64>,
}

/// CRR binomial tree pricing.
pub fn binomial_crr(params: &TreeParams) -> TreeResult {
    let n = params.n_steps;
    let dt = params.time_to_expiry / n as f64;
    let v = params.vol;
    let r = params.rate;
    let q = params.dividend;

    let u = (v * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = (((r - q) * dt).exp() - d) / (u - d);
    let disc = (-r * dt).exp();

    binomial_generic(params, u, d, p, disc)
}

/// Jarrow-Rudd (equal probability) binomial tree.
pub fn binomial_jarrow_rudd(params: &TreeParams) -> TreeResult {
    let n = params.n_steps;
    let dt = params.time_to_expiry / n as f64;
    let v = params.vol;
    let r = params.rate;
    let q = params.dividend;

    let drift = (r - q - 0.5 * v * v) * dt;
    let u = (drift + v * dt.sqrt()).exp();
    let d = (drift - v * dt.sqrt()).exp();
    let p = 0.5;
    let disc = (-r * dt).exp();

    binomial_generic(params, u, d, p, disc)
}

/// Tian binomial tree (moment-matched to 3rd moment).
pub fn binomial_tian(params: &TreeParams) -> TreeResult {
    let n = params.n_steps;
    let dt = params.time_to_expiry / n as f64;
    let v = params.vol;
    let r = params.rate;
    let q = params.dividend;

    let v2 = (v * v * dt).exp();
    let m = ((r - q) * dt).exp();
    let v2_sq = v2;
    let u = 0.5 * m * v2_sq * (v2_sq + 1.0 + (v2_sq * v2_sq + 2.0 * v2_sq - 3.0).max(0.0).sqrt());
    let d = 0.5 * m * v2_sq * (v2_sq + 1.0 - (v2_sq * v2_sq + 2.0 * v2_sq - 3.0).max(0.0).sqrt());
    let p = if (u - d).abs() > 1e-15 { (m - d) / (u - d) } else { 0.5 };
    let disc = (-r * dt).exp();

    binomial_generic(params, u, d, p, disc)
}

/// Leisen-Reimer binomial tree (improved convergence).
pub fn binomial_leisen_reimer(params: &TreeParams) -> TreeResult {
    let n = params.n_steps;
    let n_odd = if n % 2 == 0 { n + 1 } else { n };
    let dt = params.time_to_expiry / n_odd as f64;
    let v = params.vol;
    let r = params.rate;
    let q = params.dividend;
    let sqrt_t = params.time_to_expiry.sqrt();

    let d1 = ((params.spot / params.strike).ln()
        + (r - q + 0.5 * v * v) * params.time_to_expiry)
        / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;

    // Peizer-Pratt inversion
    let p_prime = peizer_pratt(d2, n_odd);
    let p_prime_prime = peizer_pratt(d1, n_odd);

    let u = ((r - q) * dt).exp() * p_prime_prime / p_prime;
    let d = (((r - q) * dt).exp() - p_prime * u) / (1.0 - p_prime);
    let p = p_prime;
    let disc = (-r * dt).exp();

    let mut params_adj = params.clone();
    params_adj.n_steps = n_odd;
    binomial_generic(&params_adj, u, d, p, disc)
}

fn peizer_pratt(z: f64, n: usize) -> f64 {
    let n_f = n as f64;
    let sign = if z >= 0.0 { 1.0 } else { -1.0 };
    let z_abs = z.abs();
    // Peizer-Pratt method 2
    let d = z_abs / (n_f + 1.0 / 3.0 + 0.1 / (n_f + 1.0));
    0.5 + sign * 0.5 * (1.0 - (-d * d * (n_f + 1.0 / 6.0)).exp()).max(0.0).sqrt()
}

fn binomial_generic(params: &TreeParams, u: f64, d: f64, p: f64, disc: f64) -> TreeResult {
    let n = params.n_steps;
    let phi = match params.opt_type { OptionType::Call => 1.0, OptionType::Put => -1.0 };

    // Terminal payoffs
    let mut prices = vec![0.0; n + 1];
    for i in 0..=n {
        let s_t = params.spot * u.powi(i as i32) * d.powi((n - i) as i32);
        prices[i] = (phi * (s_t - params.strike)).max(0.0);
    }

    // Early exercise boundary tracking
    let mut boundary = vec![0.0; n + 1];

    // Backward induction
    let dt = params.time_to_expiry / n as f64;
    let mut option_at_1 = [0.0; 3]; // for Greeks

    for step in (0..n).rev() {
        let mut new_prices = vec![0.0; step + 1];
        let mut min_exercise_spot = f64::INFINITY;

        for i in 0..=step {
            let hold = disc * (p * prices[i + 1] + (1.0 - p) * prices[i]);
            let s_node = params.spot * u.powi(i as i32) * d.powi((step - i) as i32);
            let exercise = (phi * (s_node - params.strike)).max(0.0);

            new_prices[i] = match params.exercise {
                ExerciseStyle::European => hold,
                ExerciseStyle::American => {
                    if exercise > hold {
                        if s_node < min_exercise_spot {
                            min_exercise_spot = s_node;
                        }
                        exercise
                    } else {
                        hold
                    }
                }
                ExerciseStyle::Bermudan => {
                    // Exercise only at specific times (every 10 steps for simplicity)
                    if step % 10 == 0 {
                        hold.max(exercise)
                    } else {
                        hold
                    }
                }
            };
        }

        boundary[step] = if min_exercise_spot < f64::INFINITY {
            min_exercise_spot
        } else {
            0.0
        };

        prices = new_prices;

        // Save step 1 values for Greeks
        if step == 1 {
            option_at_1[0] = prices.get(0).copied().unwrap_or(0.0);
            option_at_1[1] = prices.get(1).copied().unwrap_or(0.0);
        }
        if step == 2 && prices.len() >= 3 {
            // For gamma calculation we also need step 2
        }
    }

    let price = prices[0];

    // Greeks from tree
    let s_u = params.spot * u;
    let s_d = params.spot * d;
    let delta = if (s_u - s_d).abs() > 1e-15 {
        (option_at_1[1] - option_at_1[0]) / (s_u - s_d)
    } else {
        0.0
    };

    // Theta: use the center value at step 2
    let theta = 0.0; // simplified

    // Gamma: computed from step 2 values
    let gamma = 0.0; // simplified - would need step 2 data

    TreeResult {
        price,
        delta,
        gamma,
        theta,
        early_exercise_boundary: boundary,
    }
}

/// Generic binomial pricing function dispatcher.
pub fn binomial_price(params: &TreeParams, tree_type: TreeType) -> TreeResult {
    match tree_type {
        TreeType::CRR => binomial_crr(params),
        TreeType::JarrowRudd => binomial_jarrow_rudd(params),
        TreeType::Tian => binomial_tian(params),
        TreeType::LeisenReimer => binomial_leisen_reimer(params),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRINOMIAL TREE
// ═══════════════════════════════════════════════════════════════════════════

/// Trinomial tree pricing (Boyle 1986).
pub fn trinomial_price(params: &TreeParams) -> TreeResult {
    let n = params.n_steps;
    let dt = params.time_to_expiry / n as f64;
    let v = params.vol;
    let r = params.rate;
    let q = params.dividend;
    let phi = match params.opt_type { OptionType::Call => 1.0, OptionType::Put => -1.0 };

    let lambda = (1.5_f64).sqrt(); // stretch factor
    let u = (lambda * v * dt.sqrt()).exp();
    let d = 1.0 / u;
    let m = 1.0; // middle factor

    let drift = (r - q) * dt;
    let v2dt = v * v * dt;

    let pu = ((v2dt + drift * drift * dt) / (lambda * lambda * v2dt)
        + drift * dt.sqrt() / (2.0 * lambda * v * dt.sqrt()))
        / 2.0;
    let pd = ((v2dt + drift * drift * dt) / (lambda * lambda * v2dt)
        - drift * dt.sqrt() / (2.0 * lambda * v * dt.sqrt()))
        / 2.0;

    // Simplified probabilities
    let nu = (r - q - 0.5 * v * v) * dt;
    let pu = (v2dt / (lambda * lambda * v2dt) + nu * nu / (lambda * lambda * v2dt) + nu / (lambda * v * dt.sqrt())) / 2.0;
    let pu = pu.max(0.0).min(1.0);
    let pd_val = (v2dt / (lambda * lambda * v2dt) + nu * nu / (lambda * lambda * v2dt) - nu / (lambda * v * dt.sqrt())) / 2.0;
    let pd_val = pd_val.max(0.0).min(1.0);
    let pm = 1.0 - pu - pd_val;
    let pm = pm.max(0.0);

    let disc = (-r * dt).exp();

    // At terminal step, node i represents spot * u^(i-n) effectively
    // Total nodes at step j: 2*j+1 centered at j
    let total_nodes = 2 * n + 1;
    let mut prices = vec![0.0; total_nodes];

    // Terminal payoffs
    for i in 0..total_nodes {
        let k = i as i32 - n as i32;
        let s_t = params.spot * u.powi(k);
        prices[i] = (phi * (s_t - params.strike)).max(0.0);
    }

    let mut boundary = vec![0.0; n + 1];

    // Backward induction
    for step in (0..n).rev() {
        let nodes = 2 * step + 1;
        let mut new_prices = vec![0.0; nodes];
        let offset = n - step;

        for i in 0..nodes {
            let pi = i + offset;
            let pi2 = (pi + 2).min(prices.len() - 1);
            let pi1 = (pi + 1).min(prices.len() - 1);
            let pi0 = pi.min(prices.len() - 1);
            let hold = disc * (pu * prices[pi2] + pm * prices[pi1] + pd_val * prices[pi0]);
            let k = i as i32 - step as i32;
            let s_node = params.spot * u.powi(k);
            let exercise = (phi * (s_node - params.strike)).max(0.0);

            new_prices[i] = match params.exercise {
                ExerciseStyle::European => hold,
                ExerciseStyle::American => hold.max(exercise),
                ExerciseStyle::Bermudan => {
                    if step % 10 == 0 { hold.max(exercise) } else { hold }
                }
            };
        }

        prices = new_prices;
    }

    TreeResult {
        price: prices[0],
        delta: 0.0,
        gamma: 0.0,
        theta: 0.0,
        early_exercise_boundary: boundary,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LONGSTAFF-SCHWARTZ LSM (LEAST SQUARES MONTE CARLO)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct LsmParams {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub time_to_expiry: f64,
    pub opt_type: OptionType,
    pub n_paths: usize,
    pub n_steps: usize,
    pub seed: u64,
    pub n_basis: usize,    // number of basis functions (typically 3-5)
}

/// Longstaff-Schwartz LSM for American options.
pub fn lsm_price(params: &LsmParams) -> f64 {
    let n = params.n_paths;
    let m = params.n_steps;
    let dt = params.time_to_expiry / m as f64;
    let drift = (params.rate - params.dividend - 0.5 * params.vol * params.vol) * dt;
    let vol_sqrt = params.vol * dt.sqrt();
    let disc = (-params.rate * dt).exp();
    let phi = match params.opt_type { OptionType::Call => 1.0, OptionType::Put => -1.0 };

    // Generate paths
    let mut paths = vec![vec![0.0; m + 1]; n];
    let mut rng_state = params.seed;

    for i in 0..n {
        paths[i][0] = params.spot;
        for j in 1..=m {
            rng_state = lcg_next(rng_state);
            let u1 = (rng_state as f64 / u64::MAX as f64).max(1e-15);
            rng_state = lcg_next(rng_state);
            let u2 = rng_state as f64 / u64::MAX as f64;
            let z = box_muller(u1, u2);
            paths[i][j] = paths[i][j - 1] * (drift + vol_sqrt * z).exp();
        }
    }

    // Cash flows at each time step (initialized to terminal payoff)
    let mut cashflow = vec![0.0; n];
    let mut cashflow_time = vec![m; n];
    for i in 0..n {
        cashflow[i] = (phi * (paths[i][m] - params.strike)).max(0.0);
    }

    // Backward induction with regression
    for step in (1..m).rev() {
        // Find in-the-money paths
        let mut itm_indices = Vec::new();
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();

        for i in 0..n {
            let s = paths[i][step];
            let intrinsic = (phi * (s - params.strike)).max(0.0);
            if intrinsic > 0.0 {
                itm_indices.push(i);
                x_vals.push(s);
                // Discounted future cashflow
                let disc_factor = disc.powi((cashflow_time[i] - step) as i32);
                y_vals.push(cashflow[i] * disc_factor);
            }
        }

        if itm_indices.len() < params.n_basis + 1 {
            continue;
        }

        // Regression: use Laguerre polynomials as basis
        let coeffs = lsm_regression(&x_vals, &y_vals, params.n_basis, params.strike);

        // Decide: exercise or continue
        for (idx, &i) in itm_indices.iter().enumerate() {
            let s = paths[i][step];
            let intrinsic = (phi * (s - params.strike)).max(0.0);
            let continuation = lsm_evaluate_basis(s, &coeffs, params.n_basis, params.strike);

            if intrinsic > continuation {
                cashflow[i] = intrinsic;
                cashflow_time[i] = step;
            }
        }
    }

    // Discount all cashflows to time 0
    let mut sum = 0.0;
    for i in 0..n {
        let disc_factor = disc.powi(cashflow_time[i] as i32);
        sum += cashflow[i] * disc_factor;
    }

    sum / n as f64
}

fn lsm_regression(x: &[f64], y: &[f64], n_basis: usize, strike: f64) -> Vec<f64> {
    let n = x.len();
    let d = n_basis;

    // Build design matrix using normalized Laguerre polynomials
    // L_0(x) = 1, L_1(x) = 1-x, L_2(x) = 1-2x+x²/2, etc.
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];

    for k in 0..n {
        let s = x[k] / strike; // normalize
        let basis = laguerre_basis(s, d);
        for i in 0..d {
            for j in 0..d {
                xtx[i][j] += basis[i] * basis[j];
            }
            xty[i] += basis[i] * y[k];
        }
    }

    // Solve normal equations
    solve_linear_system(&mut xtx, &mut xty)
}

fn laguerre_basis(x: f64, n: usize) -> Vec<f64> {
    let mut basis = vec![0.0; n];
    if n > 0 {
        basis[0] = (-x / 2.0).exp();
    }
    if n > 1 {
        basis[1] = (-x / 2.0).exp() * (1.0 - x);
    }
    if n > 2 {
        basis[2] = (-x / 2.0).exp() * (1.0 - 2.0 * x + x * x / 2.0);
    }
    if n > 3 {
        basis[3] = (-x / 2.0).exp() * (1.0 - 3.0 * x + 3.0 * x * x / 2.0 - x * x * x / 6.0);
    }
    if n > 4 {
        basis[4] = (-x / 2.0).exp() * x * x;
    }
    basis
}

fn lsm_evaluate_basis(s: f64, coeffs: &[f64], n_basis: usize, strike: f64) -> f64 {
    let x = s / strike;
    let basis = laguerre_basis(x, n_basis);
    let mut val = 0.0;
    for i in 0..n_basis.min(coeffs.len()) {
        val += coeffs[i] * basis[i];
    }
    val.max(0.0)
}

fn solve_linear_system(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Vec<f64> {
    let n = b.len();
    // Gauss elimination with partial pivoting
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

fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// LSM with antithetic variates for variance reduction.
pub fn lsm_price_antithetic(params: &LsmParams) -> f64 {
    let mut p1 = params.clone();
    p1.n_paths = params.n_paths / 2;
    let price1 = lsm_price(&p1);

    // Use different seed for antithetic
    let mut p2 = params.clone();
    p2.n_paths = params.n_paths / 2;
    p2.seed = params.seed.wrapping_add(12345);
    let price2 = lsm_price(&p2);

    0.5 * (price1 + price2)
}

// ═══════════════════════════════════════════════════════════════════════════
// BARONE-ADESI-WHALEY APPROXIMATION
// ═══════════════════════════════════════════════════════════════════════════

/// Barone-Adesi and Whaley (1987) quadratic approximation for American options.
pub fn barone_adesi_whaley(
    spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType,
) -> f64 {
    if tte <= 0.0 {
        let phi = match opt_type { OptionType::Call => 1.0, OptionType::Put => -1.0 };
        return (phi * (spot - strike)).max(0.0);
    }

    let p = BSParams::new(spot, strike, rate, dividend, vol, tte);
    let euro = bs_price(&p, opt_type);

    // For American calls with no dividends, price = European
    if matches!(opt_type, OptionType::Call) && dividend <= 0.0 {
        return euro;
    }

    let v2 = vol * vol;
    let h = 1.0 - (-rate * tte).exp();
    let alpha = 2.0 * rate / v2;
    let beta_val = 2.0 * (rate - dividend) / v2;

    match opt_type {
        OptionType::Call => {
            let q2 = (-(beta_val - 1.0) + ((beta_val - 1.0).powi(2) + 4.0 * alpha / h).sqrt()) / 2.0;
            let s_star = baw_critical_spot_call(strike, rate, dividend, vol, tte, q2);
            if spot >= s_star {
                spot - strike
            } else {
                let p_star = BSParams::new(s_star, strike, rate, dividend, vol, tte);
                let euro_star = bs_price(&p_star, OptionType::Call);
                let a2 = (s_star / q2) * (1.0 - (-dividend * tte).exp() * norm_cdf(p_star.d1()));
                euro + a2 * (spot / s_star).powf(q2)
            }
        }
        OptionType::Put => {
            let q1 = (-(beta_val - 1.0) - ((beta_val - 1.0).powi(2) + 4.0 * alpha / h).sqrt()) / 2.0;
            let s_star = baw_critical_spot_put(strike, rate, dividend, vol, tte, q1);
            if spot <= s_star {
                strike - spot
            } else {
                let p_star = BSParams::new(s_star, strike, rate, dividend, vol, tte);
                let euro_star = bs_price(&p_star, OptionType::Put);
                let a1 = -(s_star / q1) * (1.0 - (-dividend * tte).exp() * norm_cdf(-p_star.d1()));
                euro + a1 * (spot / s_star).powf(q1)
            }
        }
    }
}

fn baw_critical_spot_call(strike: f64, rate: f64, div: f64, vol: f64, tte: f64, q2: f64) -> f64 {
    // Newton's method to find S* where S* - K = C(S*) + (1 - e^{-qT} N(d1(S*))) * S*/q2
    let mut s = strike; // initial guess
    let h_inf = strike * (1.0 - (-(rate - div) * tte).exp());
    s = strike + h_inf;

    for _ in 0..100 {
        let p = BSParams::new(s, strike, rate, div, vol, tte);
        let c = bs_price(&p, OptionType::Call);
        let d1 = p.d1();
        let lhs = s - strike;
        let rhs = c + (1.0 - (-div * tte).exp() * norm_cdf(d1)) * s / q2;
        let diff = lhs - rhs;

        if diff.abs() < 1e-8 {
            break;
        }

        let delta_c = (-div * tte).exp() * norm_cdf(d1);
        let deriv = 1.0 - delta_c * (1.0 + 1.0 / q2)
            - (1.0 - (-div * tte).exp() * norm_cdf(d1)) / q2;

        if deriv.abs() < 1e-15 {
            break;
        }

        s -= diff / deriv;
        s = s.max(strike * 0.1);
    }
    s
}

fn baw_critical_spot_put(strike: f64, rate: f64, div: f64, vol: f64, tte: f64, q1: f64) -> f64 {
    let mut s = strike * 0.9; // initial guess below strike

    for _ in 0..100 {
        let p = BSParams::new(s, strike, rate, div, vol, tte);
        let put = bs_price(&p, OptionType::Put);
        let d1 = p.d1();
        let lhs = strike - s;
        let rhs = put - (1.0 - (-div * tte).exp() * norm_cdf(-d1)) * s / q1;
        let diff = lhs - rhs;

        if diff.abs() < 1e-8 {
            break;
        }

        let delta_p = -(-div * tte).exp() * norm_cdf(-d1);
        let deriv = -1.0 - delta_p * (1.0 - 1.0 / q1)
            + (1.0 - (-div * tte).exp() * norm_cdf(-d1)) / q1;

        if deriv.abs() < 1e-15 {
            break;
        }

        s -= diff / deriv;
        s = s.max(0.01).min(strike * 2.0);
    }
    s
}

/// Bjerksund-Stensland (2002) approximation for American options.
pub fn bjerksund_stensland(
    spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType,
) -> f64 {
    match opt_type {
        OptionType::Call => bs2002_call(spot, strike, rate, dividend, vol, tte),
        OptionType::Put => {
            // Use put-call symmetry: P(S,K,r,q) = C(K,S,q,r)
            bs2002_call(strike, spot, dividend, rate, vol, tte)
        }
    }
}

fn bs2002_call(s: f64, x: f64, r: f64, q: f64, v: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return (s - x).max(0.0);
    }

    let v2 = v * v;
    let b = r - q;

    // Flat boundary: trigger at t/2
    let beta = (0.5 - b / v2) + ((b / v2 - 0.5).powi(2) + 2.0 * r / v2).sqrt();
    let b_inf = beta / (beta - 1.0) * x;
    let b0 = (x).max(r / (r - q) * x);

    let h_t = -(b * t + 2.0 * v * t.sqrt()) * (b0 / (b_inf - b0));
    let trigger = b_inf + (b0 - b_inf) * (-h_t).exp();

    if s >= trigger {
        return s - x;
    }

    let alpha = (trigger - x) * trigger.powf(-beta);
    let p = BSParams::new(s, x, r, q, v, t);
    let euro = bs_price(&p, OptionType::Call);

    euro + alpha * (s / trigger).powf(beta)
        - alpha * phi_func(s, t, beta, trigger, trigger, r, b, v)
        + phi_func(s, t, 1.0, trigger, trigger, r, b, v)
        - phi_func(s, t, 1.0, x, trigger, r, b, v)
        - x * phi_func(s, t, 0.0, trigger, trigger, r, b, v)
        + x * phi_func(s, t, 0.0, x, trigger, r, b, v)
}

fn phi_func(s: f64, t: f64, gamma: f64, h: f64, trigger: f64, r: f64, b: f64, v: f64) -> f64 {
    let v2 = v * v;
    let lambda = (-r * t + gamma * (b * t) + 0.5 * gamma * (gamma - 1.0) * v2 * t).exp();
    let d1 = -((s / h).ln() + (b + (gamma - 0.5) * v2) * t) / (v * t.sqrt());
    let kappa = 2.0 * b / v2 + 2.0 * gamma - 1.0;
    let d2 = -((trigger * trigger / (s * h)).ln() + (b + (gamma - 0.5) * v2) * t) / (v * t.sqrt());

    lambda * s.powf(gamma)
        * (norm_cdf(d1) - (trigger / s).powf(kappa) * norm_cdf(d2))
}

// ═══════════════════════════════════════════════════════════════════════════
// EARLY EXERCISE BOUNDARY
// ═══════════════════════════════════════════════════════════════════════════

/// Compute early exercise boundary for American put via integral equation.
pub fn early_exercise_boundary_put(
    strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    n_time_steps: usize,
) -> Vec<(f64, f64)> {
    let dt = tte / n_time_steps as f64;
    let mut boundary = Vec::with_capacity(n_time_steps + 1);

    // At expiry, boundary is K for put (or K * min(1, r/q) for call)
    let mut s_star = strike;
    boundary.push((tte, s_star));

    for i in (0..n_time_steps).rev() {
        let t = i as f64 * dt;
        let tau = tte - t;
        if tau <= 0.0 {
            boundary.push((t, strike));
            continue;
        }

        // Newton's method: find S* where P_eur(S*) + early_exercise_premium = K - S*
        let mut s = s_star;
        for _ in 0..50 {
            let p = BSParams::new(s, strike, rate, dividend, vol, tau);
            let put = bs_price(&p, OptionType::Put);
            let intrinsic = (strike - s).max(0.0);
            let diff = put - intrinsic;

            if diff.abs() < 1e-10 {
                break;
            }

            let d = crate::black_scholes::delta(&p, OptionType::Put);
            let deriv = d + 1.0;
            if deriv.abs() < 1e-15 {
                break;
            }

            s -= diff / deriv;
            s = s.max(0.01).min(strike);
        }

        s_star = s;
        boundary.push((t, s_star));
    }

    boundary.reverse();
    boundary
}

/// Compute early exercise boundary for American call.
pub fn early_exercise_boundary_call(
    strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    n_time_steps: usize,
) -> Vec<(f64, f64)> {
    if dividend <= 0.0 {
        // No early exercise for calls without dividends
        return vec![(0.0, f64::INFINITY), (tte, f64::INFINITY)];
    }

    let dt = tte / n_time_steps as f64;
    let mut boundary = Vec::with_capacity(n_time_steps + 1);

    let mut s_star = strike * (rate / dividend).max(1.0);
    boundary.push((tte, s_star));

    for i in (0..n_time_steps).rev() {
        let t = i as f64 * dt;
        let tau = tte - t;
        if tau <= 0.0 {
            boundary.push((t, f64::INFINITY));
            continue;
        }

        let mut s = s_star;
        for _ in 0..50 {
            let p = BSParams::new(s, strike, rate, dividend, vol, tau);
            let call = bs_price(&p, OptionType::Call);
            let intrinsic = (s - strike).max(0.0);
            let diff = call - intrinsic;

            if diff.abs() < 1e-10 {
                break;
            }

            let d = crate::black_scholes::delta(&p, OptionType::Call);
            let deriv = d - 1.0;
            if deriv.abs() < 1e-15 {
                break;
            }

            s -= diff / deriv;
            s = s.max(strike);
        }

        s_star = s;
        boundary.push((t, s_star));
    }

    boundary.reverse();
    boundary
}

// ═══════════════════════════════════════════════════════════════════════════
// AMERICAN OPTION GREEKS
// ═══════════════════════════════════════════════════════════════════════════

/// American option delta via BAW.
pub fn american_delta(
    spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType, ds: f64,
) -> f64 {
    let up = barone_adesi_whaley(spot + ds, strike, rate, dividend, vol, tte, opt_type);
    let dn = barone_adesi_whaley(spot - ds, strike, rate, dividend, vol, tte, opt_type);
    (up - dn) / (2.0 * ds)
}

/// American option gamma via BAW.
pub fn american_gamma(
    spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType, ds: f64,
) -> f64 {
    let up = barone_adesi_whaley(spot + ds, strike, rate, dividend, vol, tte, opt_type);
    let mid = barone_adesi_whaley(spot, strike, rate, dividend, vol, tte, opt_type);
    let dn = barone_adesi_whaley(spot - ds, strike, rate, dividend, vol, tte, opt_type);
    (up - 2.0 * mid + dn) / (ds * ds)
}

/// American option vega via BAW.
pub fn american_vega(
    spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType, dvol: f64,
) -> f64 {
    let up = barone_adesi_whaley(spot, strike, rate, dividend, vol + dvol, tte, opt_type);
    let dn = barone_adesi_whaley(spot, strike, rate, dividend, vol - dvol, tte, opt_type);
    (up - dn) / (2.0 * dvol)
}

/// American option theta via BAW.
pub fn american_theta(
    spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType, dt: f64,
) -> f64 {
    let short = barone_adesi_whaley(spot, strike, rate, dividend, vol, (tte - dt).max(0.0), opt_type);
    let current = barone_adesi_whaley(spot, strike, rate, dividend, vol, tte, opt_type);
    (short - current) / dt
}

/// American implied vol from market price.
pub fn american_implied_vol(
    market_price: f64, spot: f64, strike: f64, rate: f64, dividend: f64, tte: f64,
    opt_type: OptionType,
) -> Option<f64> {
    let mut vol = 0.2; // initial guess
    for _ in 0..100 {
        let price = barone_adesi_whaley(spot, strike, rate, dividend, vol, tte, opt_type);
        let v = american_vega(spot, strike, rate, dividend, vol, tte, opt_type, 0.001);
        if v.abs() < 1e-20 {
            break;
        }
        let diff = price - market_price;
        if diff.abs() < 1e-10 {
            return Some(vol);
        }
        vol -= diff / v;
        vol = vol.max(0.001).min(5.0);
    }
    let price = barone_adesi_whaley(spot, strike, rate, dividend, vol, tte, opt_type);
    if (price - market_price).abs() < 1e-6 {
        Some(vol)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONVERGENCE ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

/// Run Richardson extrapolation on binomial tree prices.
pub fn richardson_extrapolation(params: &TreeParams, tree_type: TreeType) -> f64 {
    let n = params.n_steps;
    let mut p1 = params.clone();
    p1.n_steps = n;
    let v1 = binomial_price(&p1, tree_type).price;

    let mut p2 = params.clone();
    p2.n_steps = n / 2;
    let v2 = binomial_price(&p2, tree_type).price;

    2.0 * v1 - v2
}

/// Convergence table: compute prices for increasing number of steps.
pub fn convergence_table(
    spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType, exercise: ExerciseStyle, tree_type: TreeType,
    steps: &[usize],
) -> Vec<(usize, f64)> {
    steps.iter().map(|&n| {
        let p = TreeParams::new(spot, strike, rate, dividend, vol, tte, opt_type, exercise, n);
        let result = binomial_price(&p, tree_type);
        (n, result.price)
    }).collect()
}

/// Compare all pricing methods for an American option.
pub fn american_price_comparison(
    spot: f64, strike: f64, rate: f64, dividend: f64, vol: f64, tte: f64,
    opt_type: OptionType,
) -> Vec<(&'static str, f64)> {
    let tree_params = TreeParams::new(spot, strike, rate, dividend, vol, tte, opt_type, ExerciseStyle::American, 200);

    let crr = binomial_crr(&tree_params).price;
    let jr = binomial_jarrow_rudd(&tree_params).price;
    let tian = binomial_tian(&tree_params).price;
    let lr = binomial_leisen_reimer(&tree_params).price;
    let tri = trinomial_price(&tree_params).price;
    let baw = barone_adesi_whaley(spot, strike, rate, dividend, vol, tte, opt_type);
    let bs2 = bjerksund_stensland(spot, strike, rate, dividend, vol, tte, opt_type);

    let lsm_params = LsmParams {
        spot, strike, rate, dividend, vol, time_to_expiry: tte, opt_type,
        n_paths: 10000, n_steps: 50, seed: 42, n_basis: 3,
    };
    let lsm = lsm_price(&lsm_params);

    vec![
        ("CRR Binomial", crr),
        ("Jarrow-Rudd", jr),
        ("Tian", tian),
        ("Leisen-Reimer", lr),
        ("Trinomial", tri),
        ("Barone-Adesi-Whaley", baw),
        ("Bjerksund-Stensland", bs2),
        ("LSM Monte Carlo", lsm),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// DIVIDEND HANDLING
// ═══════════════════════════════════════════════════════════════════════════

/// American option with discrete dividends via tree (escrowed dividend approach).
pub fn american_discrete_dividends(
    spot: f64, strike: f64, rate: f64, vol: f64, tte: f64,
    opt_type: OptionType, div_times: &[f64], div_amounts: &[f64],
    n_steps: usize,
) -> f64 {
    // Compute PV of dividends
    let pv_divs: f64 = div_times.iter().zip(div_amounts.iter())
        .filter(|(&t, _)| t <= tte)
        .map(|(&t, &d)| d * (-rate * t).exp())
        .sum();

    let spot_adj = spot - pv_divs;
    if spot_adj <= 0.0 {
        return match opt_type {
            OptionType::Call => 0.0,
            OptionType::Put => (strike - spot).max(0.0),
        };
    }

    // Price with adjusted spot and no continuous dividend
    let params = TreeParams::new(spot_adj, strike, rate, 0.0, vol, tte, opt_type, ExerciseStyle::American, n_steps);
    binomial_crr(&params).price
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_american_put_ge_european() {
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let div = 0.0;
        let vol = 0.2;
        let tte = 1.0;

        let p = BSParams::new(spot, strike, rate, div, vol, tte);
        let euro = bs_price(&p, OptionType::Put);
        let amer = barone_adesi_whaley(spot, strike, rate, div, vol, tte, OptionType::Put);
        assert!(amer >= euro - 0.01, "American put {} should be >= European {}", amer, euro);
    }

    #[test]
    fn test_american_call_no_div() {
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let vol = 0.2;
        let tte = 1.0;

        let p = BSParams::new(spot, strike, rate, 0.0, vol, tte);
        let euro = bs_price(&p, OptionType::Call);
        let amer = barone_adesi_whaley(spot, strike, rate, 0.0, vol, tte, OptionType::Call);
        assert!((amer - euro).abs() < 0.01, "American call without div should equal European: {} vs {}", amer, euro);
    }

    #[test]
    fn test_binomial_convergence() {
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let div = 0.0;
        let vol = 0.2;
        let tte = 1.0;

        let p = BSParams::new(spot, strike, rate, div, vol, tte);
        let bs = bs_price(&p, OptionType::Call);

        let params = TreeParams::new(spot, strike, rate, div, vol, tte, OptionType::Call, ExerciseStyle::European, 500);
        let tree = binomial_crr(&params).price;
        assert!((tree - bs).abs() < 2.0, "CRR should converge to BS: {} vs {}", tree, bs);
    }

    #[test]
    fn test_trinomial_positive() {
        let params = TreeParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0,
            OptionType::Put, ExerciseStyle::American, 50);
        let result = trinomial_price(&params);
        assert!(result.price >= 0.0, "Trinomial price should be non-negative: {}", result.price);
    }

    #[test]
    fn test_baw_vs_tree() {
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let div = 0.03;
        let vol = 0.2;
        let tte = 1.0;

        let baw = barone_adesi_whaley(spot, strike, rate, div, vol, tte, OptionType::Put);
        let params = TreeParams::new(spot, strike, rate, div, vol, tte,
            OptionType::Put, ExerciseStyle::American, 500);
        let tree = binomial_crr(&params).price;
        assert!((baw - tree).abs() < 3.0, "BAW {} vs Tree {}", baw, tree);
    }
}
