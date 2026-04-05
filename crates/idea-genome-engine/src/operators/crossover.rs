/// Crossover operators: BLX-α and SBX (Simulated Binary Crossover).

use rand::Rng;
use rand::distributions::Uniform;

use crate::genome::{Genome, N_PARAMS};

// ---------------------------------------------------------------------------
// BLX-α Blend Crossover
// ---------------------------------------------------------------------------

/// BLX-α blend crossover.
///
/// For each parameter `i`:
///   d = |a_i − b_i|
///   lo_ext = min(a_i, b_i) − α * d
///   hi_ext = max(a_i, b_i) + α * d
///   child gene ~ Uniform(lo_ext, hi_ext)
///
/// Both children are clamped to declared parameter bounds after crossover.
///
/// # Arguments
/// * `parent_a` / `parent_b` — the two parents.
/// * `alpha`   — extension factor (0.0 = pure interpolation, 0.5 = typical).
///               The specification requests `alpha = 0.3`.
/// * `rng`     — caller-supplied RNG.
pub fn blx_alpha(
    parent_a: &Genome,
    parent_b: &Genome,
    alpha: f64,
    rng: &mut impl Rng,
) -> (Genome, Genome) {
    assert_eq!(parent_a.parameters.len(), N_PARAMS);
    assert_eq!(parent_b.parameters.len(), N_PARAMS);

    let mut child_a_params = Vec::with_capacity(N_PARAMS);
    let mut child_b_params = Vec::with_capacity(N_PARAMS);

    for i in 0..N_PARAMS {
        let a = parent_a.parameters[i];
        let b = parent_b.parameters[i];

        let lo = a.min(b);
        let hi = a.max(b);
        let d = hi - lo;

        let lo_ext = lo - alpha * d;
        let hi_ext = hi + alpha * d;

        // If lo_ext == hi_ext (identical parents on this gene), both children get that value.
        if (hi_ext - lo_ext).abs() < f64::EPSILON {
            child_a_params.push(lo_ext);
            child_b_params.push(lo_ext);
        } else {
            let dist = Uniform::new(lo_ext, hi_ext);
            child_a_params.push(rng.sample(dist));
            child_b_params.push(rng.sample(dist));
        }
    }

    let next_gen = parent_a.generation.max(parent_b.generation) + 1;
    let parent_ids = vec![parent_a.id.clone(), parent_b.id.clone()];

    let mut child_a = Genome::from_parameters(child_a_params, next_gen, parent_ids.clone());
    let mut child_b = Genome::from_parameters(child_b_params, next_gen, parent_ids);

    child_a.clamp();
    child_b.clamp();

    (child_a, child_b)
}

// ---------------------------------------------------------------------------
// SBX — Simulated Binary Crossover
// ---------------------------------------------------------------------------

/// Simulated Binary Crossover (Deb & Agrawal, 1995).
///
/// Mimics the distribution of offspring produced by single-point crossover on
/// binary-coded strings.  The distribution parameter `eta` controls how
/// closely offspring cluster around parents:
///   * small `eta` → more exploration (spread-out offspring)
///   * large `eta` → more exploitation (offspring close to parents)
///
/// Typical values: `eta = 2` (exploration) … `eta = 20` (exploitation).
pub fn sbx(
    parent_a: &Genome,
    parent_b: &Genome,
    eta: f64,
    rng: &mut impl Rng,
) -> (Genome, Genome) {
    assert_eq!(parent_a.parameters.len(), N_PARAMS);
    assert_eq!(parent_b.parameters.len(), N_PARAMS);
    assert!(eta > 0.0, "SBX eta must be positive");

    let mut child_a_params = Vec::with_capacity(N_PARAMS);
    let mut child_b_params = Vec::with_capacity(N_PARAMS);

    for i in 0..N_PARAMS {
        let (lo_bound, hi_bound) = parent_a.bounds[i];
        let x1 = parent_a.parameters[i].min(parent_b.parameters[i]);
        let x2 = parent_a.parameters[i].max(parent_b.parameters[i]);

        if (x2 - x1).abs() < 1e-14 {
            // Identical genes — copy unchanged.
            child_a_params.push(x1);
            child_b_params.push(x1);
            continue;
        }

        let u: f64 = rng.gen_range(0.0..1.0_f64);

        // Spread factor β based on boundary constraints.
        let beta_lo = 1.0 + 2.0 * (x1 - lo_bound) / (x2 - x1);
        let beta_hi = 1.0 + 2.0 * (hi_bound - x2) / (x2 - x1);

        let alpha_lo = 2.0 - beta_lo.powf(-(eta + 1.0));
        let alpha_hi = 2.0 - beta_hi.powf(-(eta + 1.0));

        let beta_q_lo = sbx_beta_q(u, alpha_lo, eta);
        let beta_q_hi = sbx_beta_q(u, alpha_hi, eta);

        let c1 = 0.5 * ((x1 + x2) - beta_q_lo * (x2 - x1));
        let c2 = 0.5 * ((x1 + x2) + beta_q_hi * (x2 - x1));

        child_a_params.push(c1);
        child_b_params.push(c2);
    }

    let next_gen = parent_a.generation.max(parent_b.generation) + 1;
    let parent_ids = vec![parent_a.id.clone(), parent_b.id.clone()];

    let mut child_a = Genome::from_parameters(child_a_params, next_gen, parent_ids.clone());
    let mut child_b = Genome::from_parameters(child_b_params, next_gen, parent_ids);

    child_a.clamp();
    child_b.clamp();

    (child_a, child_b)
}

/// Compute the SBX spread factor β_q.
fn sbx_beta_q(u: f64, alpha: f64, eta: f64) -> f64 {
    if u <= 1.0 / alpha {
        (u * alpha).powf(1.0 / (eta + 1.0))
    } else {
        (1.0 / (2.0 - u * alpha)).powf(1.0 / (eta + 1.0))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn make_parents(seed_a: u64, seed_b: u64) -> (Genome, Genome) {
        let mut rng_a = SmallRng::seed_from_u64(seed_a);
        let mut rng_b = SmallRng::seed_from_u64(seed_b);
        (Genome::new_random(&mut rng_a), Genome::new_random(&mut rng_b))
    }

    #[test]
    fn blx_children_within_bounds() {
        let (pa, pb) = make_parents(10, 20);
        let mut rng = SmallRng::seed_from_u64(99);
        for _ in 0..50 {
            let (ca, cb) = blx_alpha(&pa, &pb, 0.3, &mut rng);
            assert!(ca.is_within_bounds(), "BLX child A out of bounds");
            assert!(cb.is_within_bounds(), "BLX child B out of bounds");
        }
    }

    #[test]
    fn blx_parent_ids_set() {
        let (pa, pb) = make_parents(1, 2);
        let mut rng = SmallRng::seed_from_u64(7);
        let (ca, _) = blx_alpha(&pa, &pb, 0.3, &mut rng);
        assert_eq!(ca.parent_ids.len(), 2);
        assert!(ca.parent_ids.contains(&pa.id));
        assert!(ca.parent_ids.contains(&pb.id));
    }

    #[test]
    fn blx_generation_increments() {
        let (pa, pb) = make_parents(5, 6);
        let mut rng = SmallRng::seed_from_u64(3);
        let (ca, _) = blx_alpha(&pa, &pb, 0.3, &mut rng);
        assert!(ca.generation >= pa.generation.max(pb.generation) + 1);
    }

    #[test]
    fn sbx_children_within_bounds() {
        let (pa, pb) = make_parents(30, 40);
        let mut rng = SmallRng::seed_from_u64(55);
        for _ in 0..50 {
            let (ca, cb) = sbx(&pa, &pb, 2.0, &mut rng);
            assert!(ca.is_within_bounds(), "SBX child A out of bounds");
            assert!(cb.is_within_bounds(), "SBX child B out of bounds");
        }
    }

    #[test]
    fn blx_alpha_zero_interpolation() {
        // With alpha=0, children must lie between parents on every gene.
        let (pa, pb) = make_parents(11, 22);
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..30 {
            let (ca, _) = blx_alpha(&pa, &pb, 0.0, &mut rng);
            for i in 0..N_PARAMS {
                let lo = pa.parameters[i].min(pb.parameters[i]);
                let hi = pa.parameters[i].max(pb.parameters[i]);
                assert!(
                    ca.parameters[i] >= lo - 1e-9 && ca.parameters[i] <= hi + 1e-9,
                    "alpha=0 child gene {} out of interpolation range",
                    i
                );
            }
        }
    }
}
