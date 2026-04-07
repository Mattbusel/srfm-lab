//! selection_strategies.rs -- Parent selection methods for the SRFM genome engine.
//!
//! All operators implement the `Selection` trait. The trait returns a reference
//! to a selected individual from the provided population slice.
//!
//! Design notes:
//!   -- All selectors handle empty population gracefully (panic with clear message)
//!   -- Fitness-based selectors use `genome.sharpe()` as the scalar fitness proxy
//!   -- NicheSelection requires pre-computed crowding distances from NSGA-II

use rand::rngs::SmallRng;
use rand::Rng;

use crate::genome::Genome;

// ---------------------------------------------------------------------------
// Individual wrapper alias
// ---------------------------------------------------------------------------

/// Re-exported for caller convenience -- selection operates on Genome slices.
pub type Individual = Genome;

// ---------------------------------------------------------------------------
// Selection trait
// ---------------------------------------------------------------------------

/// Common interface for all selection operators.
///
/// Returns a reference to the selected individual.
/// The caller owns the population and must keep it live.
pub trait Selection: Send + Sync {
    fn select<'a>(&self, population: &'a [Individual], rng: &mut SmallRng) -> &'a Individual;

    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// TournamentSelection
// ---------------------------------------------------------------------------

/// k-tournament selection: sample k candidates uniformly at random and return
/// the one with the highest scalar fitness (Sharpe ratio).
///
/// k=3 provides a good balance between selection pressure and diversity.
/// Larger k increases selection pressure (fewer weak individuals survive).
pub struct TournamentSelection {
    /// Tournament size.
    pub k: usize,
}

impl Default for TournamentSelection {
    fn default() -> Self {
        TournamentSelection { k: 3 }
    }
}

impl Selection for TournamentSelection {
    fn select<'a>(&self, population: &'a [Individual], rng: &mut SmallRng) -> &'a Individual {
        assert!(!population.is_empty(), "TournamentSelection: population is empty");
        let n = population.len();
        let k = self.k.max(1).min(n);

        let mut best_idx = rng.gen_range(0..n);
        for _ in 1..k {
            let candidate = rng.gen_range(0..n);
            if population[candidate].sharpe() > population[best_idx].sharpe() {
                best_idx = candidate;
            }
        }

        &population[best_idx]
    }

    fn name(&self) -> &'static str {
        "TournamentSelection"
    }
}

// ---------------------------------------------------------------------------
// RankSelection
// ---------------------------------------------------------------------------

/// Linear ranking selection with selective pressure parameter s.
///
/// Individuals are ranked by Sharpe (rank 1 = worst). Selection probability:
///   P(rank i) = (2 - s) / N + 2 * (i - 1) * (s - 1) / (N * (N - 1))
/// where i is 1-based rank (i=N is the best), s in [1, 2].
///
/// s = 1.5 is the recommended default; s = 2.0 maximizes selection pressure.
pub struct RankSelection {
    /// Selective pressure in [1.0, 2.0].
    pub s: f64,
}

impl Default for RankSelection {
    fn default() -> Self {
        RankSelection { s: 1.5 }
    }
}

impl Selection for RankSelection {
    fn select<'a>(&self, population: &'a [Individual], rng: &mut SmallRng) -> &'a Individual {
        assert!(!population.is_empty(), "RankSelection: population is empty");
        let n = population.len();

        // Build sorted indices: index 0 = worst sharpe, index n-1 = best sharpe
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            population[a]
                .sharpe()
                .partial_cmp(&population[b].sharpe())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compute selection probabilities for each rank position
        let s = self.s.clamp(1.0, 2.0);
        let n_f = n as f64;
        let mut probs: Vec<f64> = (0..n)
            .map(|rank| {
                let i = (rank + 1) as f64; // 1-based rank
                (2.0 - s) / n_f + 2.0 * (i - 1.0) * (s - 1.0) / (n_f * (n_f - 1.0))
            })
            .collect();

        // Normalize to ensure sum = 1 (handles floating point drift)
        let sum: f64 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Roulette wheel selection over ranks
        let r: f64 = rng.gen();
        let mut cumulative = 0.0f64;
        let mut selected_rank = n - 1;
        for (rank, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                selected_rank = rank;
                break;
            }
        }

        &population[indices[selected_rank]]
    }

    fn name(&self) -> &'static str {
        "RankSelection"
    }
}

// ---------------------------------------------------------------------------
// BoltzmannSelection
// ---------------------------------------------------------------------------

/// Boltzmann (simulated annealing inspired) selection.
///
/// Selection probability proportional to exp(fitness / T(generation)).
/// Temperature schedule: T(g) = T0 * alpha^g
///
/// At high T: nearly uniform selection (exploration).
/// At low T: almost greedy (exploitation).
///
/// Handles negative fitness (Sharpe can be negative) by shifting all values
/// by the minimum before exponentiation.
pub struct BoltzmannSelection {
    /// Initial temperature.
    pub t0: f64,
    /// Per-generation cooling rate in (0, 1).
    pub alpha: f64,
    /// Minimum temperature floor to prevent numerical underflow.
    pub t_min: f64,
}

impl Default for BoltzmannSelection {
    fn default() -> Self {
        BoltzmannSelection {
            t0: 10.0,
            alpha: 0.95,
            t_min: 0.01,
        }
    }
}

impl BoltzmannSelection {
    fn temperature(&self, generation: usize) -> f64 {
        let t = self.t0 * self.alpha.powi(generation as i32);
        t.max(self.t_min)
    }
}

impl Selection for BoltzmannSelection {
    fn select<'a>(&self, population: &'a [Individual], rng: &mut SmallRng) -> &'a Individual {
        assert!(!population.is_empty(), "BoltzmannSelection: population is empty");
        let n = population.len();

        // Compute current temperature from average generation of the population
        let avg_gen = population.iter().map(|g| g.generation as f64).sum::<f64>()
            / n as f64;
        let temp = self.temperature(avg_gen as usize);

        // Gather fitness values (Sharpe)
        let fitnesses: Vec<f64> = population.iter().map(|g| g.sharpe()).collect();

        // Shift by minimum to handle negatives and improve numerical stability
        let f_min = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let shifted: Vec<f64> = fitnesses.iter().map(|&f| f - f_min).collect();

        // Compute Boltzmann weights
        let weights: Vec<f64> = shifted.iter().map(|&f| (f / temp).exp()).collect();
        let total: f64 = weights.iter().sum();

        if total < 1e-12 {
            // Degenerate: all fitnesses equal, fall back to uniform
            return &population[rng.gen_range(0..n)];
        }

        // Roulette wheel sampling
        let r: f64 = rng.gen::<f64>() * total;
        let mut cumulative = 0.0f64;
        let mut selected = n - 1;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if r <= cumulative {
                selected = i;
                break;
            }
        }

        &population[selected]
    }

    fn name(&self) -> &'static str {
        "BoltzmannSelection"
    }
}

// ---------------------------------------------------------------------------
// NicheSelection
// ---------------------------------------------------------------------------

/// Niche-based selection using fitness sharing and crowding distance.
///
/// Fitness sharing reduces the effective fitness of individuals in crowded
/// regions of the search space, promoting diversity. The shared fitness is:
///   f'(i) = f(i) / sum_j sh(d(i,j))
/// where sh(d) = 1 - (d / sigma_share)^2 if d < sigma_share, else 0.
///
/// Crowding distance (from NSGA-II) is used as a secondary ranking criterion
/// when two individuals have equal Pareto rank. Here we use it as a tiebreaker
/// in the tournament step.
///
/// Implementation: run a 2-tournament, but replace raw fitness with shared fitness.
pub struct NicheSelection {
    /// Niche radius -- individuals within this distance share fitness.
    pub sigma_share: f64,
    /// Tournament size for the underlying tournament (default 2).
    pub k: usize,
}

impl Default for NicheSelection {
    fn default() -> Self {
        NicheSelection {
            sigma_share: 0.3,
            k: 2,
        }
    }
}

impl NicheSelection {
    /// Compute pairwise normalized parameter distance between two genomes.
    /// Uses the same normalized Euclidean distance as Genome::parameter_distance.
    fn distance(a: &Individual, b: &Individual) -> f64 {
        a.parameter_distance(b)
    }

    /// Sharing function: triangular kernel.
    fn sharing(&self, d: f64) -> f64 {
        if d >= self.sigma_share {
            0.0
        } else {
            1.0 - (d / self.sigma_share).powi(2)
        }
    }

    /// Compute shared fitness for individual at index `idx` over the full population.
    fn shared_fitness(&self, idx: usize, population: &[Individual]) -> f64 {
        let raw = population[idx].sharpe();
        if raw == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }

        let niche_count: f64 = population
            .iter()
            .enumerate()
            .map(|(j, other)| {
                if j == idx {
                    1.0 // self-sharing is always 1
                } else {
                    self.sharing(Self::distance(&population[idx], other))
                }
            })
            .sum();

        if niche_count < 1e-12 {
            raw
        } else {
            raw / niche_count
        }
    }

    /// Crowding distance for individual idx within the population.
    /// Approximates NSGA-II crowding distance using Sharpe as the single objective axis.
    fn crowding_distance(&self, idx: usize, population: &[Individual]) -> f64 {
        let n = population.len();
        if n <= 2 {
            return f64::INFINITY;
        }

        // Sort indices by Sharpe
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| {
            population[a]
                .sharpe()
                .partial_cmp(&population[b].sharpe())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let f_min = population[sorted[0]].sharpe();
        let f_max = population[sorted[n - 1]].sharpe();
        let f_range = (f_max - f_min).abs();

        if f_range < 1e-12 {
            return 1.0;
        }

        // Find position of idx in sorted order
        let pos = sorted.iter().position(|&s| s == idx).unwrap_or(n / 2);
        if pos == 0 || pos == n - 1 {
            return f64::INFINITY;
        }

        let f_left = population[sorted[pos - 1]].sharpe();
        let f_right = population[sorted[pos + 1]].sharpe();
        (f_right - f_left).abs() / f_range
    }
}

impl Selection for NicheSelection {
    fn select<'a>(&self, population: &'a [Individual], rng: &mut SmallRng) -> &'a Individual {
        assert!(!population.is_empty(), "NicheSelection: population is empty");
        let n = population.len();
        let k = self.k.max(1).min(n);

        let mut best_idx = rng.gen_range(0..n);
        let mut best_shared = self.shared_fitness(best_idx, population);

        for _ in 1..k {
            let candidate = rng.gen_range(0..n);
            let candidate_shared = self.shared_fitness(candidate, population);

            let prefer_candidate = if (candidate_shared - best_shared).abs() < 1e-10 {
                // Tiebreak: prefer larger crowding distance (more isolated)
                self.crowding_distance(candidate, population)
                    > self.crowding_distance(best_idx, population)
            } else {
                candidate_shared > best_shared
            };

            if prefer_candidate {
                best_idx = candidate;
                best_shared = candidate_shared;
            }
        }

        &population[best_idx]
    }

    fn name(&self) -> &'static str {
        "NicheSelection"
    }
}

// ---------------------------------------------------------------------------
// Convenience builder
// ---------------------------------------------------------------------------

/// Build a selection operator by name string.
pub fn selection_from_name(name: &str) -> Option<Box<dyn Selection>> {
    match name {
        "tournament" => Some(Box::new(TournamentSelection::default())),
        "rank" => Some(Box::new(RankSelection::default())),
        "boltzmann" => Some(Box::new(BoltzmannSelection::default())),
        "niche" => Some(Box::new(NicheSelection::default())),
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
        SmallRng::seed_from_u64(0xABCD_1234)
    }

    fn make_population(sharpes: &[f64]) -> Vec<Individual> {
        use rand::SeedableRng;
        let mut r = SmallRng::seed_from_u64(99);
        sharpes
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let mut g = Individual::new_random(&mut r);
                g.fitness = Some(crate::fitness::FitnessVec {
                    sharpe: s,
                    max_dd: 0.1,
                    calmar: s.max(0.0),
                    win_rate: 0.5,
                    profit_factor: 1.2,
                    n_trades: 50,
                    is_oos_spread: 0.1,
                });
                g.generation = i as u32;
                g
            })
            .collect()
    }

    #[test]
    fn tournament_selects_best_under_pressure() {
        // With k=population_size, always selects the best
        let pop = make_population(&[1.0, 2.0, 3.0, 0.5, -1.0]);
        let op = TournamentSelection { k: 5 };
        let selected = op.select(&pop, &mut rng());
        assert!((selected.sharpe() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn rank_selection_returns_valid_member() {
        let pop = make_population(&[0.5, 1.5, 2.5, -0.5, 0.0]);
        let op = RankSelection::default();
        let mut r = rng();
        for _ in 0..50 {
            let selected = op.select(&pop, &mut r);
            assert!(pop.iter().any(|g| g.id == selected.id));
        }
    }

    #[test]
    fn boltzmann_high_temp_samples_broadly() {
        // At very high temperature all individuals should be selected occasionally
        let pop = make_population(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        let op = BoltzmannSelection { t0: 1000.0, alpha: 1.0, t_min: 1000.0 };
        let mut r = rng();
        let mut counts = std::collections::HashMap::new();
        for _ in 0..500 {
            let selected = op.select(&pop, &mut r);
            *counts.entry(selected.id.clone()).or_insert(0usize) += 1;
        }
        // All 5 individuals should be selected at least once
        assert_eq!(counts.len(), 5, "expected all 5 individuals to be selected");
    }

    #[test]
    fn niche_selection_returns_valid_member() {
        let pop = make_population(&[1.0, 1.0, 1.0, 2.0, 2.0]);
        let op = NicheSelection::default();
        let mut r = rng();
        for _ in 0..20 {
            let selected = op.select(&pop, &mut r);
            assert!(pop.iter().any(|g| g.id == selected.id));
        }
    }
}
