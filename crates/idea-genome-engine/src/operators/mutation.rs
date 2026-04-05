/// Mutation operators: Gaussian mutation and adaptive sigma decay.

use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::genome::Genome;

// ---------------------------------------------------------------------------
// Gaussian mutation
// ---------------------------------------------------------------------------

/// Apply Gaussian perturbation to a genome.
///
/// For each gene `i`, with probability `mutation_rate`:
///   gene += N(0, sigma_frac * range_i)
///
/// After all perturbations the genome is clamped to its declared bounds.
///
/// # Arguments
/// * `genome`        — genome to mutate (modified in place).
/// * `sigma_frac`    — standard deviation expressed as a fraction of each
///                     parameter's range.  Default recommendation: 0.05.
/// * `mutation_rate` — probability of mutating each individual gene (0..1).
/// * `rng`           — caller-supplied RNG.
pub fn gaussian_mutate(
    genome: &mut Genome,
    sigma_frac: f64,
    mutation_rate: f64,
    rng: &mut impl Rng,
) {
    debug_assert!(
        (0.0..=1.0).contains(&mutation_rate),
        "mutation_rate must be in [0, 1]"
    );
    debug_assert!(sigma_frac > 0.0, "sigma_frac must be positive");

    for (i, param) in genome.parameters.iter_mut().enumerate() {
        if rng.gen_bool(mutation_rate.clamp(0.0, 1.0)) {
            let (lo, hi) = genome.bounds[i];
            let range = hi - lo;
            let sigma = sigma_frac * range;

            // Normal::new returns an error only when sigma is NaN or infinite.
            // We guard against that with the clamp above.
            if let Ok(dist) = Normal::new(0.0_f64, sigma) {
                *param += dist.sample(rng);
            }
        }
    }

    genome.clamp();
}

// ---------------------------------------------------------------------------
// Adaptive sigma mutation
// ---------------------------------------------------------------------------

/// Adaptive Gaussian mutation whose sigma decays from `sigma_max` to
/// `sigma_min` as the algorithm progresses through generations.
///
/// The schedule is linear in `generation / max_gen`:
///   sigma_frac = sigma_max − (sigma_max − sigma_min) * (generation / max_gen)
///
/// This encourages broad exploration early and fine-grained exploitation late.
///
/// Each gene is mutated independently with the default rate of 1/N_PARAMS
/// (expected one mutation per genome per call, common in practice).
///
/// # Arguments
/// * `genome`     — genome to mutate.
/// * `generation` — current generation number (0-based).
/// * `max_gen`    — total number of generations planned.
/// * `rng`        — caller-supplied RNG.
pub fn adaptive_mutate(genome: &mut Genome, generation: u32, max_gen: u32, rng: &mut impl Rng) {
    const SIGMA_MAX: f64 = 0.15;
    const SIGMA_MIN: f64 = 0.02;

    let progress = if max_gen == 0 {
        1.0
    } else {
        (generation as f64 / max_gen as f64).clamp(0.0, 1.0)
    };

    let sigma_frac = SIGMA_MAX - (SIGMA_MAX - SIGMA_MIN) * progress;

    // Typical EA convention: expected one gene mutated per individual.
    let mutation_rate = (1.0 / genome.parameters.len() as f64).clamp(0.01, 1.0);

    gaussian_mutate(genome, sigma_frac, mutation_rate, rng);
}

/// Polynomial mutation (alternative, parameter-aware).
///
/// A common alternative to Gaussian in real-coded GAs.  Each gene is
/// perturbed with probability `mutation_rate` using a polynomial distribution
/// controlled by `eta_m` (distribution index, typically 20).
///
/// # Arguments
/// * `genome`        — genome to mutate.
/// * `eta_m`         — polynomial distribution index (higher = smaller perturbations).
/// * `mutation_rate` — per-gene mutation probability.
/// * `rng`           — caller-supplied RNG.
pub fn polynomial_mutate(
    genome: &mut Genome,
    eta_m: f64,
    mutation_rate: f64,
    rng: &mut impl Rng,
) {
    debug_assert!(eta_m > 0.0, "eta_m must be positive");

    for (i, param) in genome.parameters.iter_mut().enumerate() {
        if rng.gen_bool(mutation_rate.clamp(0.0, 1.0)) {
            let (lo, hi) = genome.bounds[i];
            let range = hi - lo;

            if range < f64::EPSILON {
                continue;
            }

            let u: f64 = rng.gen_range(0.0..1.0);
            let delta = if u < 0.5 {
                let base = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - (*param - lo) / range).powf(eta_m + 1.0);
                base.powf(1.0 / (eta_m + 1.0)) - 1.0
            } else {
                let base = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - (hi - *param) / range).powf(eta_m + 1.0);
                1.0 - base.powf(1.0 / (eta_m + 1.0))
            };

            *param += delta * range;
        }
    }

    genome.clamp();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn make_genome(seed: u64) -> Genome {
        let mut rng = SmallRng::seed_from_u64(seed);
        Genome::new_random(&mut rng)
    }

    #[test]
    fn gaussian_mutate_stays_in_bounds() {
        let mut rng = SmallRng::seed_from_u64(77);
        for _ in 0..100 {
            let mut g = make_genome(rng.gen());
            gaussian_mutate(&mut g, 0.05, 0.3, &mut rng);
            assert!(g.is_within_bounds(), "genome out of bounds after Gaussian mutation");
        }
    }

    #[test]
    fn gaussian_mutate_rate_zero_no_change() {
        let mut rng = SmallRng::seed_from_u64(88);
        let mut g = make_genome(1);
        let original = g.parameters.clone();
        gaussian_mutate(&mut g, 0.05, 0.0, &mut rng);
        assert_eq!(g.parameters, original, "rate=0 should produce no changes");
    }

    #[test]
    fn adaptive_mutate_stays_in_bounds() {
        let mut rng = SmallRng::seed_from_u64(55);
        for gen in [0, 10, 25, 49, 50] {
            let mut g = make_genome(rng.gen());
            adaptive_mutate(&mut g, gen, 50, &mut rng);
            assert!(g.is_within_bounds(), "genome out of bounds at gen {}", gen);
        }
    }

    #[test]
    fn adaptive_sigma_decreases_over_time() {
        // Verify that early mutation produces larger average displacement than late mutation.
        use crate::genome::PARAM_META;

        let mut rng = SmallRng::seed_from_u64(11);
        let n_trials = 500;
        let mut early_disp = 0.0_f64;
        let mut late_disp = 0.0_f64;

        for _ in 0..n_trials {
            let seed: u64 = rng.gen();
            let mut g_early = make_genome(seed);
            let orig_early = g_early.parameters.clone();
            adaptive_mutate(&mut g_early, 0, 50, &mut rng);
            early_disp += g_early
                .parameters
                .iter()
                .zip(orig_early.iter())
                .zip(PARAM_META.iter())
                .map(|((a, b), (_, lo, hi))| (a - b).abs() / (hi - lo))
                .sum::<f64>();

            let mut g_late = make_genome(seed);
            let orig_late = g_late.parameters.clone();
            adaptive_mutate(&mut g_late, 49, 50, &mut rng);
            late_disp += g_late
                .parameters
                .iter()
                .zip(orig_late.iter())
                .zip(PARAM_META.iter())
                .map(|((a, b), (_, lo, hi))| (a - b).abs() / (hi - lo))
                .sum::<f64>();
        }

        assert!(
            early_disp >= late_disp,
            "early sigma ({:.4}) should produce >= displacement vs late ({:.4})",
            early_disp,
            late_disp,
        );
    }

    #[test]
    fn polynomial_mutate_stays_in_bounds() {
        let mut rng = SmallRng::seed_from_u64(33);
        for _ in 0..100 {
            let mut g = make_genome(rng.gen());
            polynomial_mutate(&mut g, 20.0, 0.2, &mut rng);
            assert!(g.is_within_bounds(), "genome out of bounds after polynomial mutation");
        }
    }
}
