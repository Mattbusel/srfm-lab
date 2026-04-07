//! crossover_strategies.rs -- Advanced crossover operators for the SRFM genome engine.
//!
//! All operators implement the `Crossover` trait and are composable with any
//! `Selection` and `Mutation` combination in the evolution loop.
//!
//! Mathematical references:
//!   -- BLX-alpha: Eshelman & Schaffer (1993)
//!   -- SBX: Deb & Agrawal (1995)
//!   -- DE/rand/1/bin: Storn & Price (1997)
//!   -- Beta(2,2) arithmetic: standard convex combination with symmetric Bell-shaped lambda

use rand::rngs::SmallRng;
use rand::Rng;

// ---------------------------------------------------------------------------
// Crossover trait
// ---------------------------------------------------------------------------

/// Common interface for all crossover operators.
///
/// Each operator takes two parent slices and returns two children.
/// Implementations must not modify the parent slices.
/// The caller is responsible for applying bounds clamping after crossover.
pub trait Crossover: Send + Sync {
    fn cross(
        &self,
        parent1: &[f64],
        parent2: &[f64],
        rng: &mut SmallRng,
    ) -> (Vec<f64>, Vec<f64>);

    /// Human-readable name for logging.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// UniformCrossover
// ---------------------------------------------------------------------------

/// Each gene is independently drawn from parent1 or parent2 with probability 0.5.
///
/// High disruption rate -- good for initial exploration when parents are distant.
/// Both children are complementary: where child1 takes from parent1, child2 takes from parent2.
pub struct UniformCrossover {
    /// Probability that a gene is taken from parent1 (default 0.5).
    pub p_parent1: f64,
}

impl Default for UniformCrossover {
    fn default() -> Self {
        UniformCrossover { p_parent1: 0.5 }
    }
}

impl Crossover for UniformCrossover {
    fn cross(
        &self,
        parent1: &[f64],
        parent2: &[f64],
        rng: &mut SmallRng,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = parent1.len().min(parent2.len());
        let mut child1 = Vec::with_capacity(n);
        let mut child2 = Vec::with_capacity(n);

        for i in 0..n {
            if rng.gen::<f64>() < self.p_parent1 {
                child1.push(parent1[i]);
                child2.push(parent2[i]);
            } else {
                child1.push(parent2[i]);
                child2.push(parent1[i]);
            }
        }

        (child1, child2)
    }

    fn name(&self) -> &'static str {
        "UniformCrossover"
    }
}

// ---------------------------------------------------------------------------
// BlendCrossoverAlpha (BLX-alpha)
// ---------------------------------------------------------------------------

/// BLX-alpha crossover: each gene is sampled from an extended interval.
///
/// For genes (p1, p2), the interval is extended by alpha * |p2 - p1| on each side:
///   lo = min(p1, p2) - alpha * d
///   hi = max(p1, p2) + alpha * d
///   child_gene ~ Uniform(lo, hi)
///
/// alpha = 0.3 provides moderate exploration beyond the parent range.
/// Both children are independently sampled from the same extended interval.
pub struct BlendCrossoverAlpha {
    pub alpha: f64,
}

impl Default for BlendCrossoverAlpha {
    fn default() -> Self {
        BlendCrossoverAlpha { alpha: 0.3 }
    }
}

impl Crossover for BlendCrossoverAlpha {
    fn cross(
        &self,
        parent1: &[f64],
        parent2: &[f64],
        rng: &mut SmallRng,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = parent1.len().min(parent2.len());
        let mut child1 = Vec::with_capacity(n);
        let mut child2 = Vec::with_capacity(n);

        for i in 0..n {
            let (lo_p, hi_p) = if parent1[i] < parent2[i] {
                (parent1[i], parent2[i])
            } else {
                (parent2[i], parent1[i])
            };
            let d = hi_p - lo_p;
            let lo = lo_p - self.alpha * d;
            let hi = hi_p + self.alpha * d;

            // Sample two independent points from the extended interval
            let g1 = if (hi - lo).abs() < 1e-14 {
                lo_p
            } else {
                rng.gen_range(lo..hi)
            };
            let g2 = if (hi - lo).abs() < 1e-14 {
                lo_p
            } else {
                rng.gen_range(lo..hi)
            };

            child1.push(g1);
            child2.push(g2);
        }

        (child1, child2)
    }

    fn name(&self) -> &'static str {
        "BlendCrossoverAlpha"
    }
}

// ---------------------------------------------------------------------------
// SimulatedBinaryCrossover (SBX)
// ---------------------------------------------------------------------------

/// SBX crossover mimics single-point binary crossover in real space.
///
/// Uses a polynomial probability distribution controlled by the distribution
/// index eta. Higher eta => children closer to parents (exploitation).
/// eta = 20 is a common default for continuous optimization.
///
/// Algorithm:
///   u ~ Uniform(0, 1)
///   if u <= 0.5: beta = (2u)^(1/(eta+1))
///   else:        beta = (1/(2-2u))^(1/(eta+1))
///   child1 = 0.5 * ((1+beta)*p1 + (1-beta)*p2)
///   child2 = 0.5 * ((1-beta)*p1 + (1+beta)*p2)
pub struct SimulatedBinaryCrossover {
    pub eta: f64,
    /// Crossover probability per gene (default 1.0 -- always cross).
    pub prob: f64,
}

impl Default for SimulatedBinaryCrossover {
    fn default() -> Self {
        SimulatedBinaryCrossover { eta: 20.0, prob: 1.0 }
    }
}

impl Crossover for SimulatedBinaryCrossover {
    fn cross(
        &self,
        parent1: &[f64],
        parent2: &[f64],
        rng: &mut SmallRng,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = parent1.len().min(parent2.len());
        let mut child1 = Vec::with_capacity(n);
        let mut child2 = Vec::with_capacity(n);
        let exp = 1.0 / (self.eta + 1.0);

        for i in 0..n {
            if rng.gen::<f64>() > self.prob || (parent1[i] - parent2[i]).abs() < 1e-14 {
                child1.push(parent1[i]);
                child2.push(parent2[i]);
                continue;
            }

            let u: f64 = rng.gen();
            let beta = if u <= 0.5 {
                (2.0 * u).powf(exp)
            } else {
                (1.0 / (2.0 - 2.0 * u)).powf(exp)
            };

            let g1 = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i]);
            let g2 = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i]);
            child1.push(g1);
            child2.push(g2);
        }

        (child1, child2)
    }

    fn name(&self) -> &'static str {
        "SimulatedBinaryCrossover"
    }
}

// ---------------------------------------------------------------------------
// ArithmeticCrossover
// ---------------------------------------------------------------------------

/// Convex combination crossover with lambda drawn from Beta(2,2).
///
/// Beta(2,2) is bell-shaped on (0,1), concentrating mass near 0.5.
/// This means children are usually close to the midpoint, with some
/// probability of being closer to one parent.
///
/// child1 = lambda * p1 + (1-lambda) * p2
/// child2 = (1-lambda) * p1 + lambda * p2
///
/// Beta(2,2) sampled via the Johnk method: if X ~ Beta(a,b), use
///   U1^(1/a) + U2^(1/b) <= 1 rejection sampling for small a,b.
/// For Beta(2,2) we use the direct formula: lambda = sin^2(pi/2 * u1)
/// when u1 and u2 satisfy a certain condition -- here we use a simpler
/// approach: lambda = (u1 + u2) / 2 where u1, u2 ~ Uniform(0,1),
/// which approximates Beta(2,2) via the sum of two uniforms (triangular
/// distribution on [0,1]), scaled to have the same mean and mode.
pub struct ArithmeticCrossover;

impl ArithmeticCrossover {
    /// Sample from Beta(2,2) using the sum-of-two-uniforms approximation.
    /// The triangular distribution on [0,1] has the same mode (0.5) as Beta(2,2).
    fn sample_lambda(rng: &mut SmallRng) -> f64 {
        // Use the actual Johnk rejection sampler for Beta(2,2):
        // X ~ Beta(2,2) via order statistics: sort 3 uniforms, take middle one.
        let mut vals = [rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()];
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals[1] // median of 3 uniforms ~ Beta(2,2) up to scaling
    }
}

impl Crossover for ArithmeticCrossover {
    fn cross(
        &self,
        parent1: &[f64],
        parent2: &[f64],
        rng: &mut SmallRng,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = parent1.len().min(parent2.len());
        let lambda = Self::sample_lambda(rng);

        let child1: Vec<f64> = (0..n)
            .map(|i| lambda * parent1[i] + (1.0 - lambda) * parent2[i])
            .collect();
        let child2: Vec<f64> = (0..n)
            .map(|i| (1.0 - lambda) * parent1[i] + lambda * parent2[i])
            .collect();

        (child1, child2)
    }

    fn name(&self) -> &'static str {
        "ArithmeticCrossover"
    }
}

// ---------------------------------------------------------------------------
// DifferentialEvolutionCrossover (DE/rand/1/bin)
// ---------------------------------------------------------------------------

/// DE/rand/1/bin crossover.
///
/// Mutant vector: v = r1 + F * (r2 - r3)
/// Trial vector: u[i] = v[i] if rand() < CR or i == j_rand, else parent[i]
///
/// F = 0.8 (scale factor), CR = 0.9 (crossover rate).
/// `parent1` is the target vector; `parent2`, `r2`, `r3` should come from
/// randomly selected population members. For the two-parent interface, we
/// treat parent2 as r1 and derive r2/r3 via internal perturbation to stay
/// compatible with the Crossover trait contract.
///
/// In practice the evolution loop should call with three distinct parents;
/// we provide the standard two-parent signature and internally create r2/r3
/// from Gaussian perturbations of parent2 scaled by the inter-parent distance.
pub struct DifferentialEvolutionCrossover {
    pub f: f64,
    pub cr: f64,
}

impl Default for DifferentialEvolutionCrossover {
    fn default() -> Self {
        DifferentialEvolutionCrossover { f: 0.8, cr: 0.9 }
    }
}

impl Crossover for DifferentialEvolutionCrossover {
    fn cross(
        &self,
        parent1: &[f64],
        parent2: &[f64],
        rng: &mut SmallRng,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = parent1.len().min(parent2.len());

        // Compute per-gene inter-parent distance to scale the surrogate r2/r3 noise
        let scale: f64 = parent1
            .iter()
            .zip(parent2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / (n as f64).sqrt();
        let noise_std = (scale * 0.3).max(1e-8);

        // Build surrogate r2 and r3 by perturbing parent2 with zero-mean Gaussian noise
        let r2: Vec<f64> = parent2
            .iter()
            .map(|&x| x + rng.gen::<f64>() * noise_std * 2.0 - noise_std)
            .collect();
        let r3: Vec<f64> = parent2
            .iter()
            .map(|&x| x + rng.gen::<f64>() * noise_std * 2.0 - noise_std)
            .collect();

        // Mutant vector: v = parent2 + F * (r2 - r3)
        let mutant: Vec<f64> = (0..n)
            .map(|i| parent2[i] + self.f * (r2[i] - r3[i]))
            .collect();

        // Mandatory crossover index
        let j_rand = rng.gen_range(0..n);

        // Trial vector 1: target = parent1, donor = mutant
        let trial1: Vec<f64> = (0..n)
            .map(|i| {
                if i == j_rand || rng.gen::<f64>() < self.cr {
                    mutant[i]
                } else {
                    parent1[i]
                }
            })
            .collect();

        // Trial vector 2: target = parent2, donor = parent1 difference-based mutant
        let mutant2: Vec<f64> = (0..n)
            .map(|i| parent1[i] + self.f * (r3[i] - r2[i]))
            .collect();
        let trial2: Vec<f64> = (0..n)
            .map(|i| {
                if i == j_rand || rng.gen::<f64>() < self.cr {
                    mutant2[i]
                } else {
                    parent2[i]
                }
            })
            .collect();

        (trial1, trial2)
    }

    fn name(&self) -> &'static str {
        "DifferentialEvolutionCrossover"
    }
}

// ---------------------------------------------------------------------------
// Convenience builder
// ---------------------------------------------------------------------------

/// Build a crossover operator by name string.
/// Returns None if the name is not recognized.
pub fn crossover_from_name(name: &str) -> Option<Box<dyn Crossover>> {
    match name {
        "uniform" => Some(Box::new(UniformCrossover::default())),
        "blx_alpha" => Some(Box::new(BlendCrossoverAlpha::default())),
        "sbx" => Some(Box::new(SimulatedBinaryCrossover::default())),
        "arithmetic" => Some(Box::new(ArithmeticCrossover)),
        "de" => Some(Box::new(DifferentialEvolutionCrossover::default())),
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
        SmallRng::seed_from_u64(0xDEAD_BEEF)
    }

    fn dummy_parents(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut r = SmallRng::seed_from_u64(seed);
        let p1: Vec<f64> = (0..n).map(|_| r.gen::<f64>()).collect();
        let p2: Vec<f64> = (0..n).map(|_| r.gen::<f64>()).collect();
        (p1, p2)
    }

    #[test]
    fn uniform_crossover_length() {
        let (p1, p2) = dummy_parents(15, 1);
        let op = UniformCrossover::default();
        let (c1, c2) = op.cross(&p1, &p2, &mut rng());
        assert_eq!(c1.len(), 15);
        assert_eq!(c2.len(), 15);
    }

    #[test]
    fn blx_alpha_extends_range() {
        let p1 = vec![0.0f64; 10];
        let p2 = vec![1.0f64; 10];
        let op = BlendCrossoverAlpha { alpha: 0.5 };
        let mut r = rng();
        for _ in 0..50 {
            let (c1, _) = op.cross(&p1, &p2, &mut r);
            for &g in &c1 {
                // With alpha=0.5 range is [-0.5, 1.5]
                assert!(g >= -0.501 && g <= 1.501, "gene={} out of extended range", g);
            }
        }
    }

    #[test]
    fn sbx_children_near_parents_high_eta() {
        let p1 = vec![0.0f64; 10];
        let p2 = vec![1.0f64; 10];
        let op = SimulatedBinaryCrossover { eta: 100.0, prob: 1.0 };
        let mut r = rng();
        let (c1, c2) = op.cross(&p1, &p2, &mut r);
        // With very high eta, children should be close to parents
        let mid1 = c1.iter().sum::<f64>() / c1.len() as f64;
        let mid2 = c2.iter().sum::<f64>() / c2.len() as f64;
        assert!(mid1 < 0.3 || mid1 > 0.7, "expected child near a parent");
        let _ = mid2;
    }

    #[test]
    fn arithmetic_crossover_convex_combination() {
        let p1 = vec![0.0f64; 15];
        let p2 = vec![1.0f64; 15];
        let op = ArithmeticCrossover;
        let mut r = rng();
        let (c1, c2) = op.cross(&p1, &p2, &mut r);
        for (&g1, &g2) in c1.iter().zip(c2.iter()) {
            // Both children should sum to 1.0 (convex combination property)
            assert!((g1 + g2 - 1.0).abs() < 1e-10, "g1+g2={} != 1.0", g1 + g2);
        }
    }

    #[test]
    fn de_crossover_returns_correct_length() {
        let (p1, p2) = dummy_parents(15, 99);
        let op = DifferentialEvolutionCrossover::default();
        let (c1, c2) = op.cross(&p1, &p2, &mut rng());
        assert_eq!(c1.len(), 15);
        assert_eq!(c2.len(), 15);
    }
}
