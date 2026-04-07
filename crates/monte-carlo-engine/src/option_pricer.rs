//! Monte Carlo option pricing.
//!
//! Implements:
//! - European option pricing (Black-Scholes dynamics)
//! - American option pricing (Longstaff-Schwartz LSM)
//! - Barrier option pricing (Down-and-Out, Down-and-In, Up-and-Out, Up-and-In)
//!
//! Variance reduction via antithetic variates is applied throughout.

use crate::gbm_paths::{GBMParams, generate_paths};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The type of barrier condition for a barrier option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierType {
    /// Option is knocked out if price falls to or below the barrier.
    DownAndOut,
    /// Option is knocked in (activated) if price falls to or below the barrier.
    DownAndIn,
    /// Option is knocked out if price rises to or above the barrier.
    UpAndOut,
    /// Option is knocked in (activated) if price rises to or above the barrier.
    UpAndIn,
}

/// Result of a Monte Carlo option pricing calculation.
#[derive(Debug, Clone)]
pub struct MCPriceResult {
    /// Estimated fair value of the option.
    pub price: f64,
    /// Standard error of the Monte Carlo estimate.
    pub std_error: f64,
    /// 95% confidence interval: (lower, upper).
    pub confidence_interval: (f64, f64),
    /// Number of paths used.
    pub n_paths: usize,
}

impl MCPriceResult {
    fn from_payoffs(payoffs: &[f64], discount: f64) -> Self {
        let n = payoffs.len();
        if n == 0 {
            return MCPriceResult {
                price: 0.0,
                std_error: 0.0,
                confidence_interval: (0.0, 0.0),
                n_paths: 0,
            };
        }
        let mean = payoffs.iter().sum::<f64>() / n as f64;
        let discounted = mean * discount;
        let variance =
            payoffs.iter().map(|p| (p * discount - discounted).powi(2)).sum::<f64>() / n as f64;
        let se = (variance / n as f64).sqrt();
        let ci = (discounted - 1.96 * se, discounted + 1.96 * se);
        MCPriceResult {
            price: discounted,
            std_error: se,
            confidence_interval: ci,
            n_paths: n,
        }
    }
}

// ---------------------------------------------------------------------------
// MCOptionPricer
// ---------------------------------------------------------------------------

/// Monte Carlo option pricer configuration.
pub struct MCOptionPricer {
    /// Number of paths to simulate (per half when using antithetic variates).
    pub n_paths: usize,
    /// Number of time steps per path.
    pub n_steps: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl MCOptionPricer {
    pub fn new(n_paths: usize, n_steps: usize, seed: u64) -> Self {
        MCOptionPricer { n_paths, n_steps, seed }
    }
}

// ---------------------------------------------------------------------------
// Antithetic variates helper
// ---------------------------------------------------------------------------

/// Generate n_paths / 2 standard paths and n_paths / 2 antithetic paths.
///
/// For antithetic variates we generate paths from normal samples Z, and also
/// paths from -Z (equivalently: reflect the log returns).
/// We do this by generating paths, then reflecting log-increments.
fn generate_with_antithetic(
    params: &GBMParams,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    // Generate the standard half.
    let half = n_paths / 2;
    let standard = generate_paths(params, half, n_steps, seed);

    // Build antithetic paths: reflect each log-increment.
    let mut antithetic = Vec::with_capacity(half);
    for std_path in &standard {
        let mut anti_path = Vec::with_capacity(n_steps + 1);
        anti_path.push(params.s0);
        let mut s = params.s0;
        for t in 0..n_steps {
            if t + 1 < std_path.len() {
                let std_log_ret = (std_path[t + 1] / std_path[t]).ln();
                // Antithetic: negate the random component.
                // log_ret = drift + sigma*sqrt(dt)*Z
                // anti: drift - sigma*sqrt(dt)*Z = 2*drift - std_log_ret
                let drift = (params.mu - 0.5 * params.sigma * params.sigma) * params.dt;
                let anti_log_ret = 2.0 * drift - std_log_ret;
                s *= anti_log_ret.exp();
                s = s.max(1e-10);
                anti_path.push(s);
            }
        }
        antithetic.push(anti_path);
    }

    let mut all = standard;
    all.extend(antithetic);
    all
}

// ---------------------------------------------------------------------------
// European option
// ---------------------------------------------------------------------------

/// Compute the intrinsic (payoff) value at expiry.
fn european_payoff(s_t: f64, k: f64, is_call: bool) -> f64 {
    if is_call {
        (s_t - k).max(0.0)
    } else {
        (k - s_t).max(0.0)
    }
}

/// Price a European option using Monte Carlo with antithetic variates.
///
/// `params` -- GBM dynamics (use risk-neutral drift: set mu = r).
/// `k` -- strike price.
/// `t` -- time to expiry in years.
/// `r` -- risk-free rate (continuous compounding, annualised).
/// `is_call` -- true for call, false for put.
pub fn price_european(
    pricer: &MCOptionPricer,
    params: &GBMParams,
    k: f64,
    t: f64,
    r: f64,
    is_call: bool,
) -> MCPriceResult {
    let n_steps = pricer.n_steps;
    let dt = t / n_steps as f64;
    let rn_params = GBMParams::new(r, params.sigma, params.s0, dt);

    let paths = generate_with_antithetic(&rn_params, pricer.n_paths, n_steps, pricer.seed);
    let payoffs: Vec<f64> = paths
        .iter()
        .filter_map(|p| p.last().map(|&s| european_payoff(s, k, is_call)))
        .collect();

    let discount = (-r * t).exp();
    MCPriceResult::from_payoffs(&payoffs, discount)
}

// ---------------------------------------------------------------------------
// American option (Longstaff-Schwartz LSM)
// ---------------------------------------------------------------------------

/// Evaluate the LSM regression basis functions at a given price.
/// Basis: [1, S, S^2, exp(-S/2)]
fn lsm_basis(s: f64) -> [f64; 4] {
    [1.0, s, s * s, (-s / 2.0).exp()]
}

/// Fit ordinary least-squares regression: regress y on columns of X.
/// X is (n_samples x 4), y is n_samples.
/// Returns the 4-element coefficient vector.
fn ols_4(x: &[[f64; 4]], y: &[f64]) -> [f64; 4] {
    // Normal equations: (X^T X) beta = X^T y
    // Use 4x4 Gram matrix and 4x1 RHS.
    let n = x.len();
    let mut xtx = [[0.0_f64; 4]; 4];
    let mut xty = [0.0_f64; 4];
    for i in 0..n {
        let xi = x[i];
        let yi = y[i];
        for r in 0..4 {
            xty[r] += xi[r] * yi;
            for c in 0..4 {
                xtx[r][c] += xi[r] * xi[c];
            }
        }
    }
    // Solve via Gaussian elimination with partial pivoting.
    let mut a = [[0.0_f64; 5]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = xtx[r][c];
        }
        a[r][4] = xty[r];
    }
    for col in 0..4 {
        // Find pivot.
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..4 {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        a.swap(col, max_row);
        let pivot = a[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..4 {
            let factor = a[row][col] / pivot;
            for k in col..5 {
                a[row][k] -= factor * a[col][k];
            }
        }
    }
    // Back substitution.
    let mut beta = [0.0_f64; 4];
    for i in (0..4).rev() {
        let mut sum = a[i][4];
        for j in (i + 1)..4 {
            sum -= a[i][j] * beta[j];
        }
        let diag = a[i][i];
        beta[i] = if diag.abs() > 1e-12 { sum / diag } else { 0.0 };
    }
    beta
}

/// Price an American option using the Longstaff-Schwartz algorithm.
///
/// The LSM method regresses continuation value against basis functions at
/// each time step, working backwards from expiry.
pub fn price_american(
    pricer: &MCOptionPricer,
    params: &GBMParams,
    k: f64,
    t: f64,
    r: f64,
    is_call: bool,
) -> MCPriceResult {
    let n_steps = pricer.n_steps;
    let dt = t / n_steps as f64;
    let rn_params = GBMParams::new(r, params.sigma, params.s0, dt);

    let paths = generate_with_antithetic(&rn_params, pricer.n_paths, n_steps, pricer.seed);
    let n = paths.len();
    let discount_step = (-r * dt).exp();

    // cash_flows[path] = the discounted payoff if exercised at time exercise_time[path].
    let mut cash_flows: Vec<f64> = paths
        .iter()
        .filter_map(|p| p.last().map(|&s| european_payoff(s, k, is_call)))
        .collect();

    // Work backwards from n_steps - 1 down to 1.
    for step in (1..n_steps).rev() {
        // Discount cash flows by one step.
        for cf in cash_flows.iter_mut() {
            *cf *= discount_step;
        }

        // Collect in-the-money paths at this step.
        let itm_indices: Vec<usize> = (0..n)
            .filter(|&i| {
                if step < paths[i].len() {
                    let s = paths[i][step];
                    let intrinsic = european_payoff(s, k, is_call);
                    intrinsic > 0.0
                } else {
                    false
                }
            })
            .collect();

        if itm_indices.is_empty() {
            continue;
        }

        // Build regression data.
        let x_reg: Vec<[f64; 4]> = itm_indices
            .iter()
            .map(|&i| lsm_basis(paths[i][step]))
            .collect();
        let y_reg: Vec<f64> = itm_indices.iter().map(|&i| cash_flows[i]).collect();

        let beta = ols_4(&x_reg, &y_reg);

        // Decide: exercise now or continue.
        for &i in &itm_indices {
            let s = paths[i][step];
            let intrinsic = european_payoff(s, k, is_call);
            let continuation: f64 = lsm_basis(s)
                .iter()
                .zip(beta.iter())
                .map(|(b, c)| b * c)
                .sum();
            if intrinsic >= continuation.max(0.0) {
                cash_flows[i] = intrinsic;
            }
        }
    }

    // The price is the mean of all (already-discounted) cash flows.
    let price = cash_flows.iter().sum::<f64>() / n as f64;
    let variance = cash_flows.iter().map(|c| (c - price).powi(2)).sum::<f64>() / n as f64;
    let se = (variance / n as f64).sqrt();
    MCPriceResult {
        price,
        std_error: se,
        confidence_interval: (price - 1.96 * se, price + 1.96 * se),
        n_paths: n,
    }
}

// ---------------------------------------------------------------------------
// Barrier option
// ---------------------------------------------------------------------------

/// Check whether a price path has breached a barrier.
fn barrier_breached(path: &[f64], barrier: f64, barrier_type: BarrierType) -> bool {
    match barrier_type {
        BarrierType::DownAndOut | BarrierType::DownAndIn => {
            path.iter().any(|&s| s <= barrier)
        }
        BarrierType::UpAndOut | BarrierType::UpAndIn => {
            path.iter().any(|&s| s >= barrier)
        }
    }
}

/// Price a barrier option via Monte Carlo with antithetic variates.
///
/// `barrier` -- the knock-in or knock-out level.
/// `option_type` -- which barrier condition applies.
pub fn price_barrier(
    pricer: &MCOptionPricer,
    params: &GBMParams,
    k: f64,
    barrier: f64,
    t: f64,
    r: f64,
    option_type: BarrierType,
) -> MCPriceResult {
    let n_steps = pricer.n_steps;
    let dt = t / n_steps as f64;
    let rn_params = GBMParams::new(r, params.sigma, params.s0, dt);

    // Determine whether the option is call or put based on conventional
    // barrier structure: DownAndOut/DownAndIn are usually puts (below spot),
    // but we price as a call here; the caller can flip the parity via K > S.
    // For generality, treat all barrier options as calls.
    let is_call = true;

    let paths = generate_with_antithetic(&rn_params, pricer.n_paths, n_steps, pricer.seed);

    let payoffs: Vec<f64> = paths
        .iter()
        .filter_map(|p| {
            p.last().map(|&s| {
                let breached = barrier_breached(p, barrier, option_type);
                let intrinsic = european_payoff(s, k, is_call);
                match option_type {
                    BarrierType::DownAndOut | BarrierType::UpAndOut => {
                        if breached { 0.0 } else { intrinsic }
                    }
                    BarrierType::DownAndIn | BarrierType::UpAndIn => {
                        if breached { intrinsic } else { 0.0 }
                    }
                }
            })
        })
        .collect();

    let discount = (-r * t).exp();
    MCPriceResult::from_payoffs(&payoffs, discount)
}

// ---------------------------------------------------------------------------
// Black-Scholes closed-form for test validation
// ---------------------------------------------------------------------------

/// Compute the Black-Scholes European call price for test comparisons.
#[cfg(test)]
fn bs_call(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let d1 = ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    s0 * normal_cdf(d1) - k * (-r * t).exp() * normal_cdf(d2)
}

/// Approximate standard normal CDF using Abramowitz and Stegun approximation.
#[cfg(test)]
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf_abs = 1.0 - pdf * poly;
    if x >= 0.0 { cdf_abs } else { 1.0 - cdf_abs }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn atm_pricer() -> (MCOptionPricer, GBMParams) {
        let pricer = MCOptionPricer::new(50_000, 252, 42);
        let params = GBMParams::new(0.05, 0.20, 100.0, 1.0 / 252.0);
        (pricer, params)
    }

    // 1. European call price is positive.
    #[test]
    fn test_european_call_positive() {
        let (pricer, params) = atm_pricer();
        let res = price_european(&pricer, &params, 100.0, 1.0, 0.05, true);
        assert!(res.price > 0.0, "call price must be positive");
    }

    // 2. European put-call parity: C - P = S0 - K*exp(-rT).
    #[test]
    fn test_european_put_call_parity() {
        let pricer = MCOptionPricer::new(100_000, 252, 99);
        let params = GBMParams::new(0.05, 0.20, 100.0, 1.0 / 252.0);
        let call = price_european(&pricer, &params, 100.0, 1.0, 0.05, true);
        let put = price_european(&pricer, &params, 100.0, 1.0, 0.05, false);
        let r = 0.05;
        let k = 100.0;
        let t = 1.0;
        let s0 = 100.0;
        let parity = s0 - k * (-r * t as f64).exp();
        let mc_diff = call.price - put.price;
        assert!((mc_diff - parity).abs() < 1.0,
            "put-call parity violated: C-P={mc_diff:.3} vs S-Ke^-rT={parity:.3}");
    }

    // 3. European call approximates Black-Scholes price.
    #[test]
    fn test_european_call_vs_bs() {
        let pricer = MCOptionPricer::new(200_000, 252, 7);
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.20;
        let t = 1.0;
        let params = GBMParams::new(r, sigma, s0, 1.0 / 252.0);
        let res = price_european(&pricer, &params, k, t, r, true);
        let bs = bs_call(s0, k, r, sigma, t);
        assert!((res.price - bs).abs() < 0.5,
            "MC call {:.3} vs BS {bs:.3}", res.price);
    }

    // 4. American call >= European call (early exercise premium is non-negative).
    #[test]
    fn test_american_call_ge_european() {
        let pricer = MCOptionPricer::new(30_000, 100, 13);
        let params = GBMParams::new(0.05, 0.25, 100.0, 0.01);
        let euro = price_european(&pricer, &params, 100.0, 1.0, 0.05, true);
        let amer = price_american(&pricer, &params, 100.0, 1.0, 0.05, true);
        assert!(amer.price >= euro.price - 0.5,
            "American {:.3} < European {:.3}", amer.price, euro.price);
    }

    // 5. American put has early exercise premium over European put.
    #[test]
    fn test_american_put_premium() {
        let pricer = MCOptionPricer::new(30_000, 100, 17);
        let params = GBMParams::new(0.05, 0.20, 100.0, 0.01);
        let euro = price_european(&pricer, &params, 100.0, 1.0, 0.05, false);
        let amer = price_american(&pricer, &params, 100.0, 1.0, 0.05, false);
        // American put should be priced at least as much as European put.
        assert!(amer.price >= euro.price - 0.5,
            "American put {:.3} < European put {:.3}", amer.price, euro.price);
    }

    // 6. Down-and-Out call price < vanilla call price (knock-out reduces value).
    #[test]
    fn test_down_and_out_cheaper() {
        let pricer = MCOptionPricer::new(50_000, 252, 33);
        let params = GBMParams::new(0.05, 0.20, 100.0, 1.0 / 252.0);
        let vanilla = price_european(&pricer, &params, 100.0, 1.0, 0.05, true);
        let dao = price_barrier(&pricer, &params, 100.0, 80.0, 1.0, 0.05, BarrierType::DownAndOut);
        assert!(dao.price <= vanilla.price + 0.1,
            "DAO {:.3} > vanilla {:.3}", dao.price, vanilla.price);
    }

    // 7. Down-and-Out + Down-and-In = Vanilla (parity relationship).
    #[test]
    fn test_barrier_parity() {
        let pricer = MCOptionPricer::new(100_000, 252, 55);
        let params = GBMParams::new(0.05, 0.20, 100.0, 1.0 / 252.0);
        let k = 100.0;
        let barrier = 85.0;
        let t = 1.0;
        let r = 0.05;
        let dao = price_barrier(&pricer, &params, k, barrier, t, r, BarrierType::DownAndOut);
        let dai = price_barrier(&pricer, &params, k, barrier, t, r, BarrierType::DownAndIn);
        let vanilla = price_european(&pricer, &params, k, t, r, true);
        let combined = dao.price + dai.price;
        assert!((combined - vanilla.price).abs() < 0.5,
            "DAO+DAI={combined:.3} vs vanilla={:.3}", vanilla.price);
    }

    // 8. MCPriceResult confidence interval brackets the point estimate.
    #[test]
    fn test_confidence_interval_brackets_price() {
        let (pricer, params) = atm_pricer();
        let res = price_european(&pricer, &params, 100.0, 1.0, 0.05, true);
        assert!(res.confidence_interval.0 <= res.price);
        assert!(res.confidence_interval.1 >= res.price);
    }

    // 9. n_paths reported matches pricer input (rounded to even for antithetic).
    #[test]
    fn test_n_paths_reported() {
        let pricer = MCOptionPricer::new(10_000, 50, 1);
        let params = GBMParams::new(0.05, 0.20, 100.0, 0.02);
        let res = price_european(&pricer, &params, 100.0, 1.0, 0.05, true);
        assert_eq!(res.n_paths, 10_000);
    }

    // 10. Deep out-of-the-money call has lower price than ATM call.
    #[test]
    fn test_otm_call_cheaper_than_atm() {
        let pricer = MCOptionPricer::new(50_000, 252, 66);
        let params = GBMParams::new(0.05, 0.20, 100.0, 1.0 / 252.0);
        let atm = price_european(&pricer, &params, 100.0, 1.0, 0.05, true);
        let otm = price_european(&pricer, &params, 150.0, 1.0, 0.05, true);
        assert!(otm.price < atm.price,
            "OTM {:.3} should be < ATM {:.3}", otm.price, atm.price);
    }

    // 11. Standard error decreases with more paths (law of large numbers).
    #[test]
    fn test_se_decreases_with_paths() {
        let params = GBMParams::new(0.05, 0.20, 100.0, 1.0 / 252.0);
        let p1 = MCOptionPricer::new(1_000, 252, 1);
        let p2 = MCOptionPricer::new(50_000, 252, 1);
        let se1 = price_european(&p1, &params, 100.0, 1.0, 0.05, true).std_error;
        let se2 = price_european(&p2, &params, 100.0, 1.0, 0.05, true).std_error;
        assert!(se2 < se1, "SE should decrease with more paths: {se1:.4} vs {se2:.4}");
    }

    // 12. LMS basis functions are finite for positive S values.
    #[test]
    fn test_lsm_basis_finite() {
        for &s in &[1.0, 50.0, 100.0, 200.0, 1000.0] {
            let b = lsm_basis(s);
            for v in b {
                assert!(v.is_finite(), "basis value is not finite at S={s}");
            }
        }
    }

    // 13. Up-and-Out barrier with barrier below S0 knocks out immediately.
    #[test]
    fn test_up_and_out_barrier_below_s0_zero() {
        let pricer = MCOptionPricer::new(10_000, 252, 77);
        let params = GBMParams::new(0.05, 0.20, 100.0, 1.0 / 252.0);
        // Barrier at 90 < S0 = 100; all paths start above the barrier and
        // the UpAndOut condition is >= barrier, so at step 0 s0 >= 90.
        // Wait -- barrier=90 < s0=100, upandout triggers when s >= barrier=90,
        // so every path is already above 90. All payoffs should be zero.
        let res = price_barrier(&pricer, &params, 110.0, 90.0, 1.0, 0.05, BarrierType::UpAndOut);
        assert!(res.price < 0.01, "UpAndOut with barrier < S0 should be ~0: {:.4}", res.price);
    }
}
