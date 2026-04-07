//! Geometric Brownian Motion and related stochastic process path generators.
//!
//! Implements:
//! - Standard GBM (Black-Scholes dynamics)
//! - Merton jump-diffusion
//! - Heston stochastic volatility (full truncation scheme)
//! - Correlated multi-asset GBM via Cholesky decomposition

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// GBM Parameters
// ---------------------------------------------------------------------------

/// Parameters for a standard Geometric Brownian Motion process.
#[derive(Debug, Clone)]
pub struct GBMParams {
    /// Drift rate (annualised, e.g. 0.08 for 8% per year).
    pub mu: f64,
    /// Volatility (annualised, e.g. 0.20 for 20% per year).
    pub sigma: f64,
    /// Initial price / starting value of the process.
    pub s0: f64,
    /// Time step size in years (e.g. 1/252 for daily steps).
    pub dt: f64,
}

impl GBMParams {
    pub fn new(mu: f64, sigma: f64, s0: f64, dt: f64) -> Self {
        GBMParams { mu, sigma, s0, dt }
    }
}

// ---------------------------------------------------------------------------
// Box-Muller transform
// ---------------------------------------------------------------------------

/// Generate a pair of independent standard normal samples from two uniform
/// samples using the Box-Muller transform.
/// Returns (z1, z2), both N(0,1).
fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    // Guard against degenerate u1 = 0 which would produce -inf in log.
    let u1 = u1.max(1e-15);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Draw `n` standard normal samples from the given RNG using Box-Muller.
fn sample_normals(rng: &mut SmallRng, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let (z1, z2) = box_muller(u1, u2);
        out.push(z1);
        if i + 1 < n {
            out.push(z2);
        }
        i += 2;
    }
    out.truncate(n);
    out
}

// ---------------------------------------------------------------------------
// Standard GBM path generation
// ---------------------------------------------------------------------------

/// Generate `n_paths` independent GBM price paths each of length `n_steps + 1`
/// (the extra element is the starting price s0).
///
/// Uses the exact discrete solution:
///   S(t + dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
/// where Z ~ N(0,1) drawn via Box-Muller.
pub fn generate_paths(
    params: &GBMParams,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let drift = (params.mu - 0.5 * params.sigma * params.sigma) * params.dt;
    let diffusion_scale = params.sigma * params.dt.sqrt();

    let total_draws = n_paths * n_steps;
    let normals = sample_normals(&mut rng, total_draws);

    let mut paths = Vec::with_capacity(n_paths);
    for p in 0..n_paths {
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(params.s0);
        let mut s = params.s0;
        for t in 0..n_steps {
            let z = normals[p * n_steps + t];
            s *= (drift + diffusion_scale * z).exp();
            path.push(s);
        }
        paths.push(path);
    }
    paths
}

// ---------------------------------------------------------------------------
// Jump-Diffusion (Merton) parameters
// ---------------------------------------------------------------------------

/// Parameters for Merton's jump-diffusion model.
///
/// The process follows GBM between jumps, with jumps arriving as a
/// Poisson process with intensity `lambda` (jumps per year).
/// Each jump multiplier is drawn from LogNormal(jump_mu, jump_sigma).
#[derive(Debug, Clone)]
pub struct JumpDiffusionParams {
    /// Underlying GBM parameters (mu, sigma, s0, dt).
    pub gbm: GBMParams,
    /// Poisson jump intensity: expected number of jumps per year.
    pub lambda: f64,
    /// Mean of the log of each jump size.
    pub jump_mu: f64,
    /// Standard deviation of the log of each jump size.
    pub jump_sigma: f64,
}

impl JumpDiffusionParams {
    pub fn new(gbm: GBMParams, lambda: f64, jump_mu: f64, jump_sigma: f64) -> Self {
        JumpDiffusionParams { gbm, lambda, jump_mu, jump_sigma }
    }
}

/// Simulate a Poisson random variable with mean `lambda` using the Knuth
/// algorithm.  Suitable for small lambda (< ~30) as used here per step.
fn poisson_draw(rng: &mut SmallRng, lambda: f64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k = 0u32;
    let mut p = 1.0_f64;
    loop {
        k += 1;
        p *= rng.gen::<f64>();
        if p <= l {
            break;
        }
    }
    k - 1
}

/// Generate `n_paths` Merton jump-diffusion paths.
///
/// For each time step:
///   1. Apply the usual GBM increment (adjusted drift removes the jump
///      compensator: mu_adj = mu - lambda*(exp(jump_mu + 0.5*jump_sigma^2) - 1))
///   2. Draw the number of jumps from Poisson(lambda*dt)
///   3. Apply each jump: S *= exp(jump_mu + jump_sigma * Z_jump)
pub fn generate_jump_diffusion_paths(
    params: &JumpDiffusionParams,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let gbm = &params.gbm;
    let dt = gbm.dt;

    // Expected jump return per unit time (compensator).
    let jump_mean = (params.jump_mu + 0.5 * params.jump_sigma * params.jump_sigma).exp() - 1.0;
    let mu_adj = gbm.mu - params.lambda * jump_mean;
    let drift = (mu_adj - 0.5 * gbm.sigma * gbm.sigma) * dt;
    let diffusion_scale = gbm.sigma * dt.sqrt();
    let lambda_dt = params.lambda * dt;

    let mut paths = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(gbm.s0);
        let mut s = gbm.s0;
        for _ in 0..n_steps {
            // Continuous diffusion component.
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let (z, _) = box_muller(u1, u2);
            s *= (drift + diffusion_scale * z).exp();

            // Jump component: Poisson number of jumps this step.
            let n_jumps = poisson_draw(&mut rng, lambda_dt);
            for _ in 0..n_jumps {
                let uj1: f64 = rng.gen();
                let uj2: f64 = rng.gen();
                let (zj, _) = box_muller(uj1, uj2);
                let log_jump = params.jump_mu + params.jump_sigma * zj;
                s *= log_jump.exp();
            }
            path.push(s);
        }
        paths.push(path);
    }
    paths
}

// ---------------------------------------------------------------------------
// Heston stochastic volatility parameters
// ---------------------------------------------------------------------------

/// Parameters for the Heston (1993) stochastic volatility model.
///
/// dS = mu*S*dt + sqrt(V)*S*dW_S
/// dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V
/// dW_S * dW_V = rho * dt
#[derive(Debug, Clone)]
pub struct HestonParams {
    /// Drift of the asset price process (annualised).
    pub mu: f64,
    /// Initial variance (sigma^2, not sigma).
    pub v0: f64,
    /// Mean reversion speed of the variance process.
    pub kappa: f64,
    /// Long-run (steady state) variance.
    pub theta: f64,
    /// Volatility of the variance process (vol-of-vol).
    pub xi: f64,
    /// Correlation between price and variance Brownian motions.
    pub rho: f64,
    /// Initial asset price.
    pub s0: f64,
    /// Time step size in years.
    pub dt: f64,
}

impl HestonParams {
    pub fn new(
        mu: f64,
        v0: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
        s0: f64,
        dt: f64,
    ) -> Self {
        HestonParams { mu, v0, kappa, theta, xi, rho, s0, dt }
    }
}

/// Generate `n_paths` Heston model paths using the Euler-Maruyama discretisation
/// with the full truncation scheme: variance is clamped to max(V, 0) before use
/// but the drift of V still uses the untruncated value to prevent bias.
///
/// Returns a vector of (price_path, variance_path) pairs where each path has
/// `n_steps + 1` elements including the initial values.
pub fn generate_heston_paths(
    params: &HestonParams,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let dt = params.dt;
    let sqrt_dt = dt.sqrt();
    // Cholesky decomposition for correlated Brownian motions:
    // W_V = Z1
    // W_S = rho*Z1 + sqrt(1 - rho^2)*Z2
    let rho_perp = (1.0 - params.rho * params.rho).sqrt();

    let mut out = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut price_path = Vec::with_capacity(n_steps + 1);
        let mut var_path = Vec::with_capacity(n_steps + 1);
        price_path.push(params.s0);
        var_path.push(params.v0);

        let mut s = params.s0;
        let mut v = params.v0;

        for _ in 0..n_steps {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let u3: f64 = rng.gen();
            let u4: f64 = rng.gen();
            let (z1, _) = box_muller(u1, u2);
            let (z2, _) = box_muller(u3, u4);

            let dw_v = z1 * sqrt_dt;
            let dw_s = (params.rho * z1 + rho_perp * z2) * sqrt_dt;

            // Full truncation: use max(v, 0) in the diffusion/drift for V,
            // but allow v to go below zero temporarily so the drift can pull
            // it back (prevents reflecting boundary bias).
            let v_plus = v.max(0.0);
            let sqrt_v = v_plus.sqrt();

            // Variance update (Euler-Maruyama on full (possibly negative) v).
            let dv = params.kappa * (params.theta - v_plus) * dt + params.xi * sqrt_v * dw_v;
            v += dv;

            // Price update -- uses clamped variance.
            let ds = params.mu * s * dt + sqrt_v * s * dw_s;
            s += ds;
            s = s.max(1e-10); // Prices cannot be negative.

            price_path.push(s);
            var_path.push(v.max(0.0)); // Store the truncated variance.
        }
        out.push((price_path, var_path));
    }
    out
}

// ---------------------------------------------------------------------------
// Correlated multi-asset GBM
// ---------------------------------------------------------------------------

/// Compute the lower-triangular Cholesky factor L of a positive-definite
/// matrix A (stored as a flat slice of size n*n, row-major).
///
/// Returns L as a flat Vec<f64> in row-major order.
fn cholesky(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let val = matrix[i][i] - sum;
                l[i][j] = if val > 0.0 { val.sqrt() } else { 0.0 };
            } else {
                let ljj = l[j][j];
                l[i][j] = if ljj.abs() > 1e-12 {
                    (matrix[i][j] - sum) / ljj
                } else {
                    0.0
                };
            }
        }
    }
    l
}

/// Generate correlated multi-asset GBM paths.
///
/// `params` -- per-asset GBM parameters (all must share the same dt).
/// `corr_matrix` -- n_assets x n_assets correlation matrix (must be PD).
/// `seed` -- reproducibility seed.
///
/// Returns a Vec of shape [n_assets][n_paths][n_steps+1].
pub fn correlated_gbm(
    params: &[GBMParams],
    corr_matrix: &[Vec<f64>],
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<Vec<Vec<f64>>> {
    let n_assets = params.len();
    assert_eq!(corr_matrix.len(), n_assets, "corr_matrix row count must equal n_assets");
    for row in corr_matrix {
        assert_eq!(row.len(), n_assets, "corr_matrix must be square");
    }

    let chol = cholesky(corr_matrix);

    let mut rng = SmallRng::seed_from_u64(seed);

    // Pre-compute per-asset drift and diffusion scaling.
    let drifts: Vec<f64> = params
        .iter()
        .map(|p| (p.mu - 0.5 * p.sigma * p.sigma) * p.dt)
        .collect();
    let diffusions: Vec<f64> = params.iter().map(|p| p.sigma * p.dt.sqrt()).collect();

    // Output: [n_assets][n_paths][n_steps+1]
    let mut out: Vec<Vec<Vec<f64>>> = (0..n_assets)
        .map(|a| {
            let mut paths = vec![vec![params[a].s0; n_steps + 1]; n_paths];
            for path in paths.iter_mut() {
                path[0] = params[a].s0;
            }
            paths
        })
        .collect();

    // We need n_assets independent normals per step per path.
    // Draw them on the fly.
    for p in 0..n_paths {
        // Initialise state per asset for this path.
        let mut s: Vec<f64> = params.iter().map(|p| p.s0).collect();

        for t in 0..n_steps {
            // Draw n_assets independent normals.
            let raw: Vec<f64> = sample_normals(&mut rng, n_assets);

            // Correlate via Cholesky: z_corr[i] = sum_j L[i][j] * raw[j]
            let mut z_corr = vec![0.0_f64; n_assets];
            for i in 0..n_assets {
                for j in 0..=i {
                    z_corr[i] += chol[i][j] * raw[j];
                }
            }

            // Advance each asset.
            for a in 0..n_assets {
                s[a] *= (drifts[a] + diffusions[a] * z_corr[a]).exp();
                out[a][p][t + 1] = s[a];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_gbm() -> GBMParams {
        GBMParams::new(0.08, 0.20, 100.0, 1.0 / 252.0)
    }

    // 1. GBM path dimensions are correct.
    #[test]
    fn test_generate_paths_shape() {
        let params = default_gbm();
        let paths = generate_paths(&params, 100, 252, 42);
        assert_eq!(paths.len(), 100);
        for path in &paths {
            assert_eq!(path.len(), 253); // n_steps + 1
        }
    }

    // 2. First element of every path equals s0.
    #[test]
    fn test_generate_paths_starts_at_s0() {
        let params = default_gbm();
        let paths = generate_paths(&params, 50, 100, 7);
        for path in &paths {
            assert!((path[0] - 100.0).abs() < 1e-12);
        }
    }

    // 3. All prices remain strictly positive.
    #[test]
    fn test_generate_paths_positive_prices() {
        let params = default_gbm();
        let paths = generate_paths(&params, 200, 252, 99);
        for path in &paths {
            for &price in path {
                assert!(price > 0.0, "price must be positive");
            }
        }
    }

    // 4. Reproducibility: same seed produces same paths.
    #[test]
    fn test_generate_paths_reproducible() {
        let params = default_gbm();
        let p1 = generate_paths(&params, 10, 10, 1234);
        let p2 = generate_paths(&params, 10, 10, 1234);
        for (a, b) in p1.iter().zip(p2.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-15);
            }
        }
    }

    // 5. GBM drift: mean log-return per step should be close to (mu - sigma^2/2)*dt.
    #[test]
    fn test_generate_paths_drift() {
        let mu = 0.10;
        let sigma = 0.20;
        let dt = 1.0 / 252.0;
        let params = GBMParams::new(mu, sigma, 100.0, dt);
        let n_paths = 50_000;
        let n_steps = 1;
        let paths = generate_paths(&params, n_paths, n_steps, 1);
        let expected_log_return = (mu - 0.5 * sigma * sigma) * dt;
        let actual: f64 = paths.iter().map(|p| (p[1] / p[0]).ln()).sum::<f64>() / n_paths as f64;
        assert!((actual - expected_log_return).abs() < 1e-4,
            "drift off: got {actual:.6}, expected {expected_log_return:.6}");
    }

    // 6. Jump-diffusion paths have correct shape.
    #[test]
    fn test_jump_diffusion_shape() {
        let gbm = default_gbm();
        let params = JumpDiffusionParams::new(gbm, 2.0, -0.05, 0.10);
        let paths = generate_jump_diffusion_paths(&params, 50, 252, 42);
        assert_eq!(paths.len(), 50);
        for path in &paths {
            assert_eq!(path.len(), 253);
        }
    }

    // 7. Jump-diffusion prices stay positive.
    #[test]
    fn test_jump_diffusion_positive() {
        let gbm = default_gbm();
        let params = JumpDiffusionParams::new(gbm, 5.0, -0.10, 0.20);
        let paths = generate_jump_diffusion_paths(&params, 100, 252, 8);
        for path in &paths {
            for &p in path {
                assert!(p > 0.0);
            }
        }
    }

    // 8. Heston output shape is correct.
    #[test]
    fn test_heston_shape() {
        let p = HestonParams::new(0.08, 0.04, 2.0, 0.04, 0.30, -0.70, 100.0, 1.0 / 252.0);
        let out = generate_heston_paths(&p, 100, 252, 42);
        assert_eq!(out.len(), 100);
        for (price_path, var_path) in &out {
            assert_eq!(price_path.len(), 253);
            assert_eq!(var_path.len(), 253);
        }
    }

    // 9. Heston variance paths are non-negative (full truncation guarantee).
    #[test]
    fn test_heston_variance_non_negative() {
        let p = HestonParams::new(0.0, 0.04, 1.0, 0.04, 0.50, -0.50, 100.0, 1.0 / 252.0);
        let out = generate_heston_paths(&p, 200, 252, 17);
        for (_, var_path) in &out {
            for &v in var_path {
                assert!(v >= 0.0, "variance must be non-negative after truncation");
            }
        }
    }

    // 10. Correlated GBM: shape is [n_assets][n_paths][n_steps+1].
    #[test]
    fn test_correlated_gbm_shape() {
        let params = vec![
            GBMParams::new(0.08, 0.20, 100.0, 1.0 / 252.0),
            GBMParams::new(0.06, 0.15, 50.0, 1.0 / 252.0),
        ];
        let corr = vec![vec![1.0, 0.6], vec![0.6, 1.0]];
        let out = correlated_gbm(&params, &corr, 50, 100, 42);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 50);
        assert_eq!(out[0][0].len(), 101);
    }

    // 11. Correlated GBM: initial prices match s0 for each asset.
    #[test]
    fn test_correlated_gbm_initial_prices() {
        let params = vec![
            GBMParams::new(0.10, 0.25, 200.0, 1.0 / 252.0),
            GBMParams::new(0.05, 0.12, 80.0, 1.0 / 252.0),
            GBMParams::new(0.07, 0.18, 50.0, 1.0 / 252.0),
        ];
        let corr = vec![
            vec![1.0, 0.4, 0.3],
            vec![0.4, 1.0, 0.5],
            vec![0.3, 0.5, 1.0],
        ];
        let out = correlated_gbm(&params, &corr, 30, 50, 11);
        for (a, asset_paths) in out.iter().enumerate() {
            for path in asset_paths {
                assert!((path[0] - params[a].s0).abs() < 1e-12);
            }
        }
    }

    // 12. Box-Muller produces values statistically close to N(0,1).
    #[test]
    fn test_box_muller_distribution() {
        let mut rng = SmallRng::seed_from_u64(555);
        let normals = sample_normals(&mut rng, 100_000);
        let mean = normals.iter().sum::<f64>() / normals.len() as f64;
        let var = normals.iter().map(|z| z * z).sum::<f64>() / normals.len() as f64;
        assert!(mean.abs() < 0.01, "mean {mean:.4} not close to 0");
        assert!((var - 1.0).abs() < 0.01, "variance {var:.4} not close to 1");
    }

    // 13. Cholesky decomposition: L * L^T should recover the input matrix.
    #[test]
    fn test_cholesky_correctness() {
        let corr = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];
        let l = cholesky(&corr);
        let n = 3;
        // Compute L * L^T and compare to corr.
        for i in 0..n {
            for j in 0..n {
                let val: f64 = (0..n).map(|k| l[i][k] * l[j][k]).sum();
                assert!((val - corr[i][j]).abs() < 1e-10,
                    "L*L^T[{i}][{j}] = {val:.6} but corr = {:.6}", corr[i][j]);
            }
        }
    }
}
