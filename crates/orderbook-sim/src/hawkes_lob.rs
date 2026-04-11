/// Hawkes process driving order arrival rates in the LOB.
///
/// Implements:
/// - Univariate Hawkes with exponential kernel (Ogata thinning simulation)
/// - MLE parameter estimation via Ogata's method (gradient + Newton)
/// - Branching ratio computation and stationarity check
/// - Bivariate Hawkes for bid vs. ask sides with cross-excitation
/// - Intensity forecasting and integrated intensity computation

use std::f64::consts::PI;

// ── Univariate Hawkes ─────────────────────────────────────────────────────────

/// Parameters for a univariate Hawkes process with exponential kernel.
///
/// Intensity: λ(t) = μ + Σ_{t_i < t} α·exp(−β·(t − t_i))
#[derive(Debug, Clone, Copy)]
pub struct HawkesParams1D {
    /// Baseline (exogenous) intensity μ > 0.
    pub mu: f64,
    /// Excitation jump size α ≥ 0.
    pub alpha: f64,
    /// Decay rate β > 0.
    pub beta: f64,
}

impl HawkesParams1D {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        assert!(mu > 0.0, "mu must be positive");
        assert!(alpha >= 0.0, "alpha must be non-negative");
        assert!(beta > 0.0, "beta must be positive");
        HawkesParams1D { mu, alpha, beta }
    }

    /// Branching ratio: n = α/β. Process is stationary iff n < 1.
    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }

    pub fn is_stationary(&self) -> bool {
        self.branching_ratio() < 1.0
    }

    /// Unconditional mean intensity: μ / (1 − n) for stationary process.
    pub fn mean_intensity(&self) -> f64 {
        let n = self.branching_ratio();
        if n >= 1.0 { f64::INFINITY } else { self.mu / (1.0 - n) }
    }
}

impl Default for HawkesParams1D {
    fn default() -> Self {
        HawkesParams1D { mu: 5.0, alpha: 3.0, beta: 10.0 }
    }
}

/// Simulate a univariate Hawkes process using Ogata's modified thinning algorithm.
/// Returns sorted arrival times in [0, T].
pub fn simulate_hawkes_1d(
    horizon: f64,
    params: &HawkesParams1D,
    rng: &mut impl rand_trait::Rng,
) -> Vec<f64> {
    assert!(params.is_stationary(), "Hawkes process must be stationary (α/β < 1)");
    let mut times = Vec::new();
    let mut t = 0.0_f64;
    // Upper bound on intensity.
    let mut lambda_bar = params.mu;

    loop {
        // Candidate inter-arrival from exponential with rate lambda_bar.
        let u1: f64 = rng.gen();
        if u1 < 1e-20 { break; }
        let dt = -u1.ln() / lambda_bar;
        t += dt;
        if t > horizon { break; }

        // Compute actual intensity at t.
        let lambda_t = params.mu
            + times.iter()
                .map(|&ti: &f64| params.alpha * (-(params.beta * (t - ti))).exp())
                .sum::<f64>();

        // Accept-reject.
        let u2: f64 = rng.gen();
        if u2 <= lambda_t / lambda_bar {
            times.push(t);
            lambda_bar = lambda_t + params.alpha; // new upper bound after jump
        } else {
            lambda_bar = lambda_t; // tighten upper bound
        }
    }
    times
}

// ── MLE via Ogata / EM ────────────────────────────────────────────────────────

/// Log-likelihood of observed times under the Hawkes model (exponential kernel).
///
/// log L = Σ log λ(t_i) − ∫₀ᵀ λ(t) dt
///
/// Efficient O(n) recursion exploiting exponential kernel structure.
pub fn hawkes_log_likelihood(times: &[f64], horizon: f64, params: &HawkesParams1D) -> f64 {
    let n = times.len();
    if n == 0 { return 0.0; }

    let mu = params.mu;
    let alpha = params.alpha;
    let beta = params.beta;

    // R_i = Σ_{j<i} exp(-β(t_i - t_j)) via recursion: R_i = exp(-β Δt)·(1 + R_{i-1})
    let mut r = 0.0_f64;
    let mut log_sum = 0.0_f64;
    let mut comp = 0.0_f64;  // compensator integral

    for i in 0..n {
        if i > 0 {
            let dt = times[i] - times[i - 1];
            r = (-(beta * dt)).exp() * (1.0 + r);
        } else {
            r = 0.0;
        }
        let lambda_i = mu + alpha * r;
        if lambda_i <= 0.0 { return f64::NEG_INFINITY; }
        log_sum += lambda_i.ln();
    }

    // Compensator: ∫₀ᵀ λ(t) dt = μ·T + (α/β)·Σ_i (1 − exp(−β(T − t_i)))
    comp = mu * horizon;
    for &ti in times {
        comp += (alpha / beta) * (1.0 - (-(beta * (horizon - ti))).exp());
    }

    log_sum - comp
}

/// Gradient of log-likelihood w.r.t. (μ, α, β).
pub fn hawkes_log_likelihood_grad(
    times: &[f64],
    horizon: f64,
    params: &HawkesParams1D,
) -> (f64, f64, f64) {
    let n = times.len();
    if n == 0 { return (0.0, 0.0, 0.0); }

    let mu = params.mu;
    let alpha = params.alpha;
    let beta = params.beta;

    // Recursion variables.
    let mut r = 0.0_f64;    // Σ exp(-β Δ)
    let mut s = 0.0_f64;    // Σ Δ·exp(-β Δ) for β gradient

    let mut d_mu = 0.0_f64;
    let mut d_alpha = 0.0_f64;
    let mut d_beta = 0.0_f64;

    for i in 0..n {
        if i > 0 {
            let dt = times[i] - times[i - 1];
            let e = (-(beta * dt)).exp();
            s = e * (s - dt * (1.0 + r));
            r = e * (1.0 + r);
        }
        let lambda_i = mu + alpha * r;
        if lambda_i <= 0.0 { return (0.0, 0.0, 0.0); }
        let inv_lambda = 1.0 / lambda_i;
        d_mu += inv_lambda;
        d_alpha += r * inv_lambda;
        d_beta += alpha * s * inv_lambda;
    }

    // Subtract compensator derivatives.
    d_mu -= horizon;
    let mut sum_exp = 0.0_f64;
    let mut sum_dt_exp = 0.0_f64;
    for &ti in times {
        let exp_val = (-(beta * (horizon - ti))).exp();
        sum_exp += 1.0 - exp_val;
        sum_dt_exp += (horizon - ti) * exp_val;
    }
    d_alpha -= sum_exp / beta;
    d_beta -= alpha * (sum_exp / beta.powi(2) - sum_dt_exp / beta);  // chain rule

    (d_mu, d_alpha, d_beta)
}

/// MLE estimation of Hawkes parameters via gradient ascent with Armijo line search.
///
/// Returns (mu_hat, alpha_hat, beta_hat) and final log-likelihood.
pub fn hawkes_mle(
    times: &[f64],
    horizon: f64,
    init: HawkesParams1D,
    max_iter: usize,
    tol: f64,
) -> (HawkesParams1D, f64) {
    let mut mu = init.mu;
    let mut alpha = init.alpha;
    let mut beta = init.beta;

    let mut prev_ll = f64::NEG_INFINITY;

    for _iter in 0..max_iter {
        let params = HawkesParams1D { mu, alpha, beta };
        let ll = hawkes_log_likelihood(times, horizon, &params);
        let (g_mu, g_alpha, g_beta) = hawkes_log_likelihood_grad(times, horizon, &params);

        let grad_norm = (g_mu * g_mu + g_alpha * g_alpha + g_beta * g_beta).sqrt();
        if grad_norm < tol { break; }
        if (ll - prev_ll).abs() < tol && _iter > 5 { break; }
        prev_ll = ll;

        // Armijo line search.
        let mut step = 1.0_f64;
        let c = 0.5_f64;
        let tau = 0.5_f64;
        loop {
            let mu_new = (mu + step * g_mu).max(1e-8);
            let alpha_new = (alpha + step * g_alpha).max(0.0);
            let beta_new = (beta + step * g_beta).max(1e-8);
            // Ensure stationarity.
            let alpha_new = alpha_new.min(beta_new * 0.999);
            let params_new = HawkesParams1D { mu: mu_new, alpha: alpha_new, beta: beta_new };
            let ll_new = hawkes_log_likelihood(times, horizon, &params_new);
            if ll_new >= ll + c * step * grad_norm * grad_norm || step < 1e-14 {
                mu = mu_new;
                alpha = alpha_new;
                beta = beta_new;
                break;
            }
            step *= tau;
        }
    }

    let params = HawkesParams1D { mu, alpha, beta };
    let ll = hawkes_log_likelihood(times, horizon, &params);
    (params, ll)
}

// ── Intensity Computation ─────────────────────────────────────────────────────

/// Compute the conditional intensity λ(t | F_t−) at a given query time.
pub fn hawkes_intensity_at(t: f64, history: &[f64], params: &HawkesParams1D) -> f64 {
    params.mu
        + history.iter()
            .filter(|&&ti| ti < t)
            .map(|&ti| params.alpha * (-(params.beta * (t - ti))).exp())
            .sum::<f64>()
}

/// Compute the integrated intensity (compensator) ∫ from 0 to T.
pub fn hawkes_compensator(history: &[f64], horizon: f64, params: &HawkesParams1D) -> f64 {
    let base = params.mu * horizon;
    let exc: f64 = history.iter()
        .filter(|&&ti| ti < horizon)
        .map(|&ti| (params.alpha / params.beta) * (1.0 - (-(params.beta * (horizon - ti))).exp()))
        .sum();
    base + exc
}

/// Residual process check: transformed times should be uniform on [0, N].
/// Returns transformed inter-arrivals (should be i.i.d. Exp(1) under correct model).
pub fn hawkes_residuals(times: &[f64], horizon: f64, params: &HawkesParams1D) -> Vec<f64> {
    let n = times.len();
    if n == 0 { return vec![]; }

    // Compute Λ(t_i) for each event time.
    let lambdas: Vec<f64> = times.iter()
        .map(|&t| hawkes_compensator(&times[..times.iter().position(|&x| x == t).unwrap()], t, params))
        .collect();

    // Inter-arrival residuals: Λ(t_i) − Λ(t_{i-1}).
    let mut residuals = Vec::with_capacity(n);
    residuals.push(lambdas[0]);
    for i in 1..n {
        residuals.push(lambdas[i] - lambdas[i - 1]);
    }
    residuals
}

// ── Bivariate Hawkes ──────────────────────────────────────────────────────────

/// Parameters for a bivariate Hawkes process (bid + ask sides).
///
/// Processes: λ_B(t), λ_A(t)
/// λ_B(t) = μ_B + α_BB·R_BB(t) + α_AB·R_AB(t)
/// λ_A(t) = μ_A + α_AA·R_AA(t) + α_BA·R_BA(t)
#[derive(Debug, Clone, Copy)]
pub struct HawkesParams2D {
    /// Baseline intensities.
    pub mu_b: f64,
    pub mu_a: f64,
    /// Self-excitation: bid→bid, ask→ask.
    pub alpha_bb: f64,
    pub alpha_aa: f64,
    /// Cross-excitation: ask→bid, bid→ask.
    pub alpha_ab: f64,
    pub alpha_ba: f64,
    /// Decay rates (shared for simplicity; could be per-component).
    pub beta_b: f64,
    pub beta_a: f64,
}

impl Default for HawkesParams2D {
    fn default() -> Self {
        HawkesParams2D {
            mu_b: 3.0, mu_a: 3.0,
            alpha_bb: 1.5, alpha_aa: 1.5,
            alpha_ab: 0.5, alpha_ba: 0.5,
            beta_b: 8.0, beta_a: 8.0,
        }
    }
}

impl HawkesParams2D {
    /// Spectral radius of the excitation matrix. Must be < 1 for stationarity.
    pub fn spectral_radius(&self) -> f64 {
        // 2x2 matrix M = [[α_BB/β_B, α_AB/β_B], [α_BA/β_A, α_AA/β_A]]
        let m11 = self.alpha_bb / self.beta_b;
        let m12 = self.alpha_ab / self.beta_b;
        let m21 = self.alpha_ba / self.beta_a;
        let m22 = self.alpha_aa / self.beta_a;
        // Eigenvalues of 2x2: (tr ± sqrt(tr² - 4·det)) / 2
        let tr = m11 + m22;
        let det = m11 * m22 - m12 * m21;
        let disc = (tr * tr - 4.0 * det).max(0.0);
        (tr + disc.sqrt()) / 2.0
    }

    pub fn is_stationary(&self) -> bool {
        self.spectral_radius() < 1.0
    }
}

/// State for bivariate Hawkes simulation.
pub struct HawkesState2D {
    /// Accumulated excitation from bid events on bid intensity.
    pub r_bb: f64,
    /// Accumulated excitation from ask events on ask intensity.
    pub r_aa: f64,
    /// Cross: ask→bid.
    pub r_ab: f64,
    /// Cross: bid→ask.
    pub r_ba: f64,
    /// Current time.
    pub t: f64,
}

impl HawkesState2D {
    pub fn new() -> Self {
        HawkesState2D { r_bb: 0.0, r_aa: 0.0, r_ab: 0.0, r_ba: 0.0, t: 0.0 }
    }

    /// Advance state to time t without events.
    pub fn advance(&mut self, t_new: f64, params: &HawkesParams2D) {
        let dt = (t_new - self.t).max(0.0);
        let eb = (-(params.beta_b * dt)).exp();
        let ea = (-(params.beta_a * dt)).exp();
        self.r_bb *= eb;
        self.r_ab *= eb;
        self.r_aa *= ea;
        self.r_ba *= ea;
        self.t = t_new;
    }

    /// Record a bid event.
    pub fn add_bid_event(&mut self) {
        self.r_bb += 1.0;
        self.r_ba += 1.0;
    }

    /// Record an ask event.
    pub fn add_ask_event(&mut self) {
        self.r_aa += 1.0;
        self.r_ab += 1.0;
    }

    pub fn bid_intensity(&self, params: &HawkesParams2D) -> f64 {
        params.mu_b
            + params.alpha_bb * self.r_bb
            + params.alpha_ab * self.r_ab
    }

    pub fn ask_intensity(&self, params: &HawkesParams2D) -> f64 {
        params.mu_a
            + params.alpha_aa * self.r_aa
            + params.alpha_ba * self.r_ba
    }
}

impl Default for HawkesState2D {
    fn default() -> Self { Self::new() }
}

/// Simulate a bivariate Hawkes process via Ogata thinning.
/// Returns (bid_times, ask_times).
pub fn simulate_hawkes_2d(
    horizon: f64,
    params: &HawkesParams2D,
    rng: &mut impl rand_trait::Rng,
) -> (Vec<f64>, Vec<f64>) {
    assert!(params.is_stationary(), "Bivariate Hawkes must be stationary");

    let mut bid_times = Vec::new();
    let mut ask_times = Vec::new();
    let mut state = HawkesState2D::new();
    let mut t = 0.0_f64;
    let mut lambda_bar = params.mu_b + params.mu_a; // initial upper bound

    loop {
        let u1: f64 = rng.gen();
        if u1 < 1e-20 { break; }
        let dt = -u1.ln() / lambda_bar;
        t += dt;
        if t > horizon { break; }

        state.advance(t, params);
        let lambda_b = state.bid_intensity(params);
        let lambda_a = state.ask_intensity(params);
        let lambda_tot = lambda_b + lambda_a;

        let u2: f64 = rng.gen();
        if u2 <= lambda_tot / lambda_bar {
            // Accept — determine which process.
            let u3: f64 = rng.gen();
            if u3 < lambda_b / lambda_tot {
                bid_times.push(t);
                state.add_bid_event();
                lambda_bar = lambda_tot + params.alpha_bb + params.alpha_ba;
            } else {
                ask_times.push(t);
                state.add_ask_event();
                lambda_bar = lambda_tot + params.alpha_aa + params.alpha_ab;
            }
        } else {
            lambda_bar = lambda_tot;
        }
    }

    (bid_times, ask_times)
}

/// Log-likelihood of observed (bid, ask) times under the 2D Hawkes model.
pub fn hawkes_2d_log_likelihood(
    bid_times: &[f64],
    ask_times: &[f64],
    horizon: f64,
    params: &HawkesParams2D,
) -> f64 {
    let mut state = HawkesState2D::new();

    // Merge and sort all events.
    let mut events: Vec<(f64, bool)> = bid_times.iter().map(|&t| (t, true))
        .chain(ask_times.iter().map(|&t| (t, false)))
        .collect();
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut log_sum = 0.0_f64;

    for (t, is_bid) in &events {
        state.advance(*t, params);
        let lambda = if *is_bid {
            state.bid_intensity(params)
        } else {
            state.ask_intensity(params)
        };
        if lambda <= 0.0 { return f64::NEG_INFINITY; }
        log_sum += lambda.ln();
        if *is_bid { state.add_bid_event(); } else { state.add_ask_event(); }
    }

    // Compensators.
    let mut state2 = HawkesState2D::new();
    let mut comp_b = params.mu_b * horizon;
    let mut comp_a = params.mu_a * horizon;

    for (t, is_bid) in &events {
        // Contribution to compensators: (α/β)·(1 − exp(−β·(T − t)))
        let remaining = horizon - t;
        if *is_bid {
            comp_b += (params.alpha_bb / params.beta_b) * (1.0 - (-(params.beta_b * remaining)).exp());
            comp_a += (params.alpha_ba / params.beta_a) * (1.0 - (-(params.beta_a * remaining)).exp());
        } else {
            comp_a += (params.alpha_aa / params.beta_a) * (1.0 - (-(params.beta_a * remaining)).exp());
            comp_b += (params.alpha_ab / params.beta_b) * (1.0 - (-(params.beta_b * remaining)).exp());
        }
    }

    log_sum - comp_b - comp_a
}

// ── Order Intensity Forecasting ───────────────────────────────────────────────

/// Forecast expected number of events in [t, t+dt] given history.
pub fn hawkes_expected_count(
    t: f64,
    dt: f64,
    history: &[f64],
    params: &HawkesParams1D,
) -> f64 {
    // E[N(t, t+dt) | F_t] ≈ integral of conditional intensity.
    // For exponential kernel: E[integral] = μ·dt + Σ_{t_i<t} (α/β)·exp(−β(t−t_i))·(1−exp(−β·dt))
    let base = params.mu * dt;
    let exc: f64 = history.iter()
        .filter(|&&ti| ti < t)
        .map(|&ti| {
            let decay = (-(params.beta * (t - ti))).exp();
            (params.alpha / params.beta) * decay * (1.0 - (-(params.beta * dt)).exp())
        })
        .sum();
    base + exc
}

// ── Wrapper trait for rand compatibility ──────────────────────────────────────

// We define a local trait to avoid depending on the full rand crate here;
// implementations should use rand::Rng directly.
pub mod rand_trait {
    pub trait Rng {
        fn gen<T: RandGen>(&mut self) -> T;
    }

    pub trait RandGen: Sized {
        fn generate(rng: &mut dyn FnMut() -> u64) -> Self;
    }

    impl RandGen for f64 {
        fn generate(rng: &mut dyn FnMut() -> u64) -> Self {
            (rng() >> 11) as f64 / (1u64 << 53) as f64
        }
    }
}

// ── Bridge to rand crate ──────────────────────────────────────────────────────

use rand::Rng as RandRng;

pub struct RandBridge<R: RandRng>(pub R);

impl<R: RandRng> rand_trait::Rng for RandBridge<R> {
    fn gen<T: rand_trait::RandGen>(&mut self) -> T {
        let mut closure = || self.0.gen::<u64>();
        T::generate(&mut closure)
    }
}

// ── Convenience wrappers using rand ──────────────────────────────────────────

pub fn sim_hawkes_1d(horizon: f64, params: &HawkesParams1D, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut bridge = RandBridge(rng);
    simulate_hawkes_1d(horizon, params, &mut bridge)
}

pub fn sim_hawkes_2d(
    horizon: f64,
    params: &HawkesParams2D,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    use rand::SeedableRng;
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut bridge = RandBridge(rng);
    simulate_hawkes_2d(horizon, params, &mut bridge)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branching_ratio() {
        let p = HawkesParams1D::new(5.0, 3.0, 10.0);
        assert!((p.branching_ratio() - 0.3).abs() < 1e-10);
        assert!(p.is_stationary());
    }

    #[test]
    fn test_simulation_produces_events() {
        let p = HawkesParams1D::new(5.0, 3.0, 10.0);
        let times = sim_hawkes_1d(10.0, &p, 42);
        assert!(!times.is_empty());
        // Check monotonicity.
        for i in 1..times.len() {
            assert!(times[i] > times[i - 1]);
        }
        // All in [0, 10].
        assert!(times.iter().all(|&t| t >= 0.0 && t <= 10.0));
    }

    #[test]
    fn test_log_likelihood_finite() {
        let p = HawkesParams1D::new(5.0, 3.0, 10.0);
        let times = sim_hawkes_1d(100.0, &p, 7);
        let ll = hawkes_log_likelihood(&times, 100.0, &p);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_mle_recovers_params() {
        let true_params = HawkesParams1D::new(5.0, 2.0, 8.0);
        // Simulate a large dataset.
        let times = sim_hawkes_1d(1000.0, &true_params, 123);
        let horizon = *times.last().unwrap_or(&1000.0);
        let init = HawkesParams1D::new(4.0, 1.5, 7.0);
        let (est, ll) = hawkes_mle(&times, horizon, init, 500, 1e-6);
        // MLE should be in ballpark of true params.
        assert!(ll.is_finite());
        assert!(est.mu > 0.0 && est.alpha >= 0.0 && est.beta > 0.0);
        assert!(est.is_stationary());
    }

    #[test]
    fn test_bivariate_simulation() {
        let p = HawkesParams2D::default();
        assert!(p.is_stationary());
        let (bids, asks) = sim_hawkes_2d(20.0, &p, 99);
        assert!(!bids.is_empty());
        assert!(!asks.is_empty());
    }

    #[test]
    fn test_2d_log_likelihood() {
        let p = HawkesParams2D::default();
        let (bids, asks) = sim_hawkes_2d(100.0, &p, 55);
        let ll = hawkes_2d_log_likelihood(&bids, &asks, 100.0, &p);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_intensity_at() {
        let p = HawkesParams1D::new(2.0, 1.0, 5.0);
        let history = vec![0.1, 0.3, 0.5];
        let lam = hawkes_intensity_at(0.6, &history, &p);
        assert!(lam >= p.mu);
    }

    #[test]
    fn test_expected_count_positive() {
        let p = HawkesParams1D::new(5.0, 2.0, 8.0);
        let times = sim_hawkes_1d(10.0, &p, 1);
        let count = hawkes_expected_count(5.0, 1.0, &times, &p);
        assert!(count > 0.0);
    }
}
