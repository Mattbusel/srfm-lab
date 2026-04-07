//! mutation_strategies.rs -- Mutation operators for the SRFM genome engine.
//!
//! All operators implement the `Mutation` trait and work in-place on genome slices.
//! Bounds are enforced inside each operator; callers do not need to clamp afterward.
//!
//! Mathematical references:
//!   -- Polynomial mutation: Deb & Goyal (1996), distribution index eta_m
//!   -- 1/5 success rule: Rechenberg (1973)
//!   -- Non-uniform mutation: Michalewicz (1994)
//!   -- Cauchy mutation: Yao, Liu & Lin (1999) in Fast Evolutionary Programming

use rand::rngs::SmallRng;
use rand::Rng;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Mutation trait
// ---------------------------------------------------------------------------

/// Common interface for all mutation operators.
///
/// `genome` is mutated in-place.
/// `bounds` contains (min, max) per gene -- same length as genome.
/// `generation` is the current generation index (0-based).
/// `rng` provides randomness.
pub trait Mutation: Send + Sync {
    fn mutate(
        &mut self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        generation: usize,
        rng: &mut SmallRng,
    );

    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Helper: standard normal via Box-Muller transform
// ---------------------------------------------------------------------------

/// Sample one N(0,1) variate using Box-Muller.
fn sample_normal(rng: &mut SmallRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Sample from Cauchy(0, 1) using the inverse CDF method.
fn sample_cauchy(rng: &mut SmallRng) -> f64 {
    let u: f64 = rng.gen::<f64>();
    // Avoid the singularities at 0 and 1
    let u_clamped = u.clamp(1e-6, 1.0 - 1e-6);
    (PI * (u_clamped - 0.5)).tan()
}

// ---------------------------------------------------------------------------
// GaussianMutation
// ---------------------------------------------------------------------------

/// Gaussian perturbation mutation with generation-dependent sigma.
///
/// sigma(generation) = sigma0 / sqrt(generation + 1)
///
/// This provides large initial exploration that narrows as the search progresses.
/// Each gene is mutated independently with probability `mutation_rate`.
pub struct GaussianMutation {
    /// Initial standard deviation (in normalized [0,1] space per parameter).
    pub sigma0: f64,
    /// Per-gene mutation probability.
    pub mutation_rate: f64,
}

impl Default for GaussianMutation {
    fn default() -> Self {
        GaussianMutation {
            sigma0: 0.1,
            mutation_rate: 1.0 / 15.0, // 1/n_params heuristic
        }
    }
}

impl Mutation for GaussianMutation {
    fn mutate(
        &mut self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        generation: usize,
        rng: &mut SmallRng,
    ) {
        let sigma = self.sigma0 / ((generation + 1) as f64).sqrt();
        let n = genome.len().min(bounds.len());

        for i in 0..n {
            if rng.gen::<f64>() >= self.mutation_rate {
                continue;
            }
            let (lo, hi) = bounds[i];
            let range = hi - lo;
            if range < 1e-14 {
                continue;
            }
            let noise = sample_normal(rng) * sigma * range;
            genome[i] = (genome[i] + noise).clamp(lo, hi);
        }
    }

    fn name(&self) -> &'static str {
        "GaussianMutation"
    }
}

// ---------------------------------------------------------------------------
// CauchyMutation
// ---------------------------------------------------------------------------

/// Cauchy distribution mutation -- heavier tails than Gaussian.
///
/// The Cauchy distribution has no finite variance, meaning it occasionally
/// produces large jumps that help escape local optima. Scale parameter t
/// controls the spread (analogous to sigma in Gaussian mutation).
///
/// t(generation) = t0 / (1 + generation * decay_rate)
pub struct CauchyMutation {
    /// Initial scale parameter.
    pub t0: f64,
    /// Linear decay rate per generation.
    pub decay_rate: f64,
    /// Per-gene mutation probability.
    pub mutation_rate: f64,
}

impl Default for CauchyMutation {
    fn default() -> Self {
        CauchyMutation {
            t0: 0.1,
            decay_rate: 0.001,
            mutation_rate: 1.0 / 15.0,
        }
    }
}

impl Mutation for CauchyMutation {
    fn mutate(
        &mut self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        generation: usize,
        rng: &mut SmallRng,
    ) {
        let t = self.t0 / (1.0 + generation as f64 * self.decay_rate);
        let n = genome.len().min(bounds.len());

        for i in 0..n {
            if rng.gen::<f64>() >= self.mutation_rate {
                continue;
            }
            let (lo, hi) = bounds[i];
            let range = hi - lo;
            if range < 1e-14 {
                continue;
            }
            let noise = sample_cauchy(rng) * t * range;
            genome[i] = (genome[i] + noise).clamp(lo, hi);
        }
    }

    fn name(&self) -> &'static str {
        "CauchyMutation"
    }
}

// ---------------------------------------------------------------------------
// PolynomialMutation
// ---------------------------------------------------------------------------

/// Polynomial mutation (PM) -- bounded perturbation with polynomial distribution.
///
/// The perturbation magnitude is drawn from a polynomial distribution that
/// respects the [lo, hi] bounds of each gene. The distribution index eta_m
/// controls the sharpness: higher eta_m => smaller perturbations (exploitation).
///
/// Algorithm per gene:
///   u ~ Uniform(0, 1)
///   delta1 = (x - lo) / (hi - lo)
///   delta2 = (hi - x) / (hi - lo)
///   if u < 0.5:
///     delta = (2u + (1-2u)*(1-delta1)^(eta_m+1))^(1/(eta_m+1)) - 1
///   else:
///     delta = 1 - (2(1-u) + 2*(u-0.5)*(1-delta2)^(eta_m+1))^(1/(eta_m+1))
///   x' = x + delta * (hi - lo)
pub struct PolynomialMutation {
    pub eta_m: f64,
    pub mutation_rate: f64,
}

impl Default for PolynomialMutation {
    fn default() -> Self {
        PolynomialMutation {
            eta_m: 20.0,
            mutation_rate: 1.0 / 15.0,
        }
    }
}

impl Mutation for PolynomialMutation {
    fn mutate(
        &mut self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        generation: usize,
        rng: &mut SmallRng,
    ) {
        let _ = generation; // PM does not use generation
        let exp = 1.0 / (self.eta_m + 1.0);
        let n = genome.len().min(bounds.len());

        for i in 0..n {
            if rng.gen::<f64>() >= self.mutation_rate {
                continue;
            }
            let (lo, hi) = bounds[i];
            let range = hi - lo;
            if range < 1e-14 {
                continue;
            }

            let x = genome[i];
            let delta1 = (x - lo) / range;
            let delta2 = (hi - x) / range;
            let u: f64 = rng.gen();

            let delta_q = if u < 0.5 {
                let base = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1).powf(self.eta_m + 1.0);
                base.powf(exp) - 1.0
            } else {
                let base =
                    2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2).powf(self.eta_m + 1.0);
                1.0 - base.powf(exp)
            };

            genome[i] = (x + delta_q * range).clamp(lo, hi);
        }
    }

    fn name(&self) -> &'static str {
        "PolynomialMutation"
    }
}

// ---------------------------------------------------------------------------
// AdaptiveSigmaMutation (1/5 success rule)
// ---------------------------------------------------------------------------

/// Self-adaptive Gaussian mutation using Rechenberg's 1/5 success rule.
///
/// Tracks the fraction of mutations that improved fitness over a sliding
/// window of `window_size` mutations. Every `update_interval` calls to
/// `mutate()`, the sigma is updated:
///   -- if success_rate > 0.2: sigma *= 1 / 0.817 (increase by factor ~1.22)
///   -- if success_rate < 0.2: sigma *= 0.817      (decrease by factor ~0.82)
///   -- if success_rate == 0.2: sigma unchanged
///
/// The caller must call `record_outcome(improved: bool)` after each evaluation
/// to feed the success history. This separates evaluation from mutation.
pub struct AdaptiveSigmaMutation {
    pub sigma: f64,
    /// Minimum allowed sigma.
    pub sigma_min: f64,
    /// Maximum allowed sigma.
    pub sigma_max: f64,
    /// Per-gene mutation probability.
    pub mutation_rate: f64,
    /// Success history ring buffer.
    success_history: Vec<bool>,
    history_head: usize,
    history_count: usize,
    /// How many `mutate()` calls between sigma updates.
    pub update_interval: usize,
    calls_since_update: usize,
    /// Rechenberg adjustment factor.
    adjustment: f64,
}

impl AdaptiveSigmaMutation {
    pub fn new(sigma_init: f64, window_size: usize) -> Self {
        AdaptiveSigmaMutation {
            sigma: sigma_init,
            sigma_min: 1e-6,
            sigma_max: 0.5,
            mutation_rate: 1.0 / 15.0,
            success_history: vec![false; window_size],
            history_head: 0,
            history_count: 0,
            update_interval: window_size,
            calls_since_update: 0,
            adjustment: 0.817, // Rechenberg's constant
        }
    }

    /// Record whether the last mutation improved fitness.
    /// Must be called once per individual after evaluation.
    pub fn record_outcome(&mut self, improved: bool) {
        self.success_history[self.history_head] = improved;
        self.history_head = (self.history_head + 1) % self.success_history.len();
        if self.history_count < self.success_history.len() {
            self.history_count += 1;
        }
    }

    /// Return the current success rate over the history window.
    pub fn success_rate(&self) -> f64 {
        if self.history_count == 0 {
            return 0.2; // neutral
        }
        let successes: usize = self.success_history[..self.history_count]
            .iter()
            .filter(|&&s| s)
            .count();
        successes as f64 / self.history_count as f64
    }

    fn maybe_update_sigma(&mut self) {
        self.calls_since_update += 1;
        if self.calls_since_update < self.update_interval {
            return;
        }
        self.calls_since_update = 0;

        let rate = self.success_rate();
        if rate > 0.2 {
            self.sigma = (self.sigma / self.adjustment).min(self.sigma_max);
        } else if rate < 0.2 {
            self.sigma = (self.sigma * self.adjustment).max(self.sigma_min);
        }
        // rate == 0.2: no change
    }
}

impl Mutation for AdaptiveSigmaMutation {
    fn mutate(
        &mut self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        generation: usize,
        rng: &mut SmallRng,
    ) {
        let _ = generation;
        self.maybe_update_sigma();
        let sigma = self.sigma;
        let n = genome.len().min(bounds.len());

        for i in 0..n {
            if rng.gen::<f64>() >= self.mutation_rate {
                continue;
            }
            let (lo, hi) = bounds[i];
            let range = hi - lo;
            if range < 1e-14 {
                continue;
            }
            let noise = sample_normal(rng) * sigma * range;
            genome[i] = (genome[i] + noise).clamp(lo, hi);
        }
    }

    fn name(&self) -> &'static str {
        "AdaptiveSigmaMutation"
    }
}

// ---------------------------------------------------------------------------
// NonUniformMutation
// ---------------------------------------------------------------------------

/// Non-uniform mutation from Michalewicz (1994).
///
/// The perturbation size decreases as generations increase, following:
///   delta(t, y) = y * (1 - r^((1 - t/T)^b))
///
/// where:
///   t = current generation
///   T = max_generations (upper bound, softly enforced)
///   r ~ Uniform(0, 1)
///   b = non_uniformity_factor (controls decay rate; b=5 is typical)
///   y = distance to bound in the mutation direction
///
/// This provides uniform-like mutation early (t << T) and nearly zero
/// perturbation as t approaches T.
pub struct NonUniformMutation {
    pub max_generations: usize,
    pub b: f64,
    pub mutation_rate: f64,
}

impl Default for NonUniformMutation {
    fn default() -> Self {
        NonUniformMutation {
            max_generations: 500,
            b: 5.0,
            mutation_rate: 1.0 / 15.0,
        }
    }
}

impl Mutation for NonUniformMutation {
    fn mutate(
        &mut self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        generation: usize,
        rng: &mut SmallRng,
    ) {
        let t = generation.min(self.max_generations) as f64;
        let big_t = self.max_generations as f64;
        let tau = (1.0 - t / big_t).max(0.0);
        let n = genome.len().min(bounds.len());

        for i in 0..n {
            if rng.gen::<f64>() >= self.mutation_rate {
                continue;
            }
            let (lo, hi) = bounds[i];
            let x = genome[i];
            let r: f64 = rng.gen();
            let exponent = tau.powf(self.b);
            // Randomly choose direction: toward hi or toward lo
            let delta = if rng.gen::<bool>() {
                (hi - x) * (1.0 - r.powf(exponent))
            } else {
                -(x - lo) * (1.0 - r.powf(exponent))
            };
            genome[i] = (x + delta).clamp(lo, hi);
        }
    }

    fn name(&self) -> &'static str {
        "NonUniformMutation"
    }
}

// ---------------------------------------------------------------------------
// Convenience builder
// ---------------------------------------------------------------------------

/// Build a mutation operator by name string.
pub fn mutation_from_name(name: &str) -> Option<Box<dyn Mutation>> {
    match name {
        "gaussian" => Some(Box::new(GaussianMutation::default())),
        "cauchy" => Some(Box::new(CauchyMutation::default())),
        "polynomial" => Some(Box::new(PolynomialMutation::default())),
        "adaptive_sigma" => Some(Box::new(AdaptiveSigmaMutation::new(0.1, 20))),
        "non_uniform" => Some(Box::new(NonUniformMutation::default())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(0xC0FFEE)
    }

    fn unit_bounds(n: usize) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); n]
    }

    #[test]
    fn gaussian_stays_in_bounds() {
        let mut op = GaussianMutation { sigma0: 0.5, mutation_rate: 1.0 };
        let bounds = unit_bounds(15);
        let mut genome = vec![0.5f64; 15];
        let mut r = rng();
        for gen in 0..100 {
            op.mutate(&mut genome, &bounds, gen, &mut r);
            for (&g, &(lo, hi)) in genome.iter().zip(bounds.iter()) {
                assert!(g >= lo && g <= hi);
            }
        }
    }

    #[test]
    fn cauchy_stays_in_bounds() {
        let mut op = CauchyMutation { t0: 0.3, decay_rate: 0.0, mutation_rate: 1.0 };
        let bounds = unit_bounds(15);
        let mut genome = vec![0.5f64; 15];
        let mut r = rng();
        for gen in 0..50 {
            op.mutate(&mut genome, &bounds, gen, &mut r);
            for (&g, &(lo, hi)) in genome.iter().zip(bounds.iter()) {
                assert!(g >= lo && g <= hi);
            }
        }
    }

    #[test]
    fn polynomial_stays_in_bounds() {
        let mut op = PolynomialMutation { eta_m: 20.0, mutation_rate: 1.0 };
        let bounds = unit_bounds(15);
        let mut genome = vec![0.5f64; 15];
        let mut r = rng();
        for _ in 0..100 {
            op.mutate(&mut genome, &bounds, 0, &mut r);
            for (&g, &(lo, hi)) in genome.iter().zip(bounds.iter()) {
                assert!(g >= lo - 1e-10 && g <= hi + 1e-10);
            }
        }
    }

    #[test]
    fn adaptive_sigma_adjusts_down_on_failure() {
        let mut op = AdaptiveSigmaMutation::new(0.1, 20);
        let sigma_initial = op.sigma;
        // Record all failures
        for _ in 0..20 {
            op.record_outcome(false);
        }
        let bounds = unit_bounds(15);
        let mut genome = vec![0.5f64; 15];
        let mut r = rng();
        op.mutate(&mut genome, &bounds, 0, &mut r);
        assert!(op.sigma < sigma_initial, "sigma should decrease after all failures");
    }

    #[test]
    fn non_uniform_shrinks_at_late_generation() {
        let mut op = NonUniformMutation { max_generations: 100, b: 5.0, mutation_rate: 1.0 };
        let bounds = unit_bounds(15);
        let mut genome_early = vec![0.5f64; 15];
        let mut genome_late = vec![0.5f64; 15];
        let mut r1 = SmallRng::seed_from_u64(1);
        let mut r2 = SmallRng::seed_from_u64(1);
        op.mutate(&mut genome_early, &bounds, 5, &mut r1);
        op.mutate(&mut genome_late, &bounds, 95, &mut r2);
        let change_early: f64 = genome_early.iter().map(|g| (g - 0.5).abs()).sum();
        let change_late: f64 = genome_late.iter().map(|g| (g - 0.5).abs()).sum();
        // Late generation changes should generally be smaller
        assert!(change_late <= change_early + 0.5); // soft check due to randomness
    }
}
