/// Genetic evolver: orchestrates island-model evolution across multiple generations.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use crate::fitness::FitnessEvaluator;
use crate::genome::Genome;
use crate::operators::crossover::blx_alpha;
use crate::operators::mutation::adaptive_mutate;
use crate::operators::selection::tournament_select;
use crate::population::Population;

// ---------------------------------------------------------------------------
// Island label
// ---------------------------------------------------------------------------

/// Market regime label for each island.  The label is metadata only — the GA
/// mechanics are identical; the regime tag is used to filter which tick data
/// is used during evaluation (configured externally in the Python backtest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IslandRegime {
    Bull,
    Bear,
    Neutral,
}

impl std::fmt::Display for IslandRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IslandRegime::Bull => write!(f, "BULL"),
            IslandRegime::Bear => write!(f, "BEAR"),
            IslandRegime::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// Evolution result
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct IslandResult {
    pub regime: IslandRegime,
    /// Best genome (by Sharpe) found on this island.
    pub best_genome: Genome,
    /// Final diversity of the island population.
    pub final_diversity: f64,
    /// Sharpe trajectory across generations (one entry per generation).
    pub sharpe_history: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvolutionResult {
    pub island_results: Vec<IslandResult>,
    /// Non-dominated genomes across all islands at the end of evolution.
    pub pareto_front: Vec<Genome>,
    /// Overall best genome by Sharpe.
    pub global_best: Genome,
    /// Total wall-clock time in seconds.
    pub elapsed_secs: f64,
}

// ---------------------------------------------------------------------------
// GeneticEvolver
// ---------------------------------------------------------------------------

/// Top-level driver for the island-model genetic algorithm.
#[derive(Debug)]
pub struct GeneticEvolver {
    pub population_size: usize,
    pub n_generations: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    /// Fraction of top individuals carried over unchanged between generations.
    pub elite_frac: f64,
    pub use_islands: bool,
    /// Number of generations between island migrations.
    pub migration_interval: usize,
    /// Number of migrants exchanged per migration event per pair of islands.
    pub n_migrants: usize,
    /// Tournament size for parent selection.
    pub tournament_size: usize,
    /// BLX-alpha extension factor.
    pub blx_alpha: f64,
    pub evaluator: FitnessEvaluator,
}

impl GeneticEvolver {
    pub fn new(evaluator: FitnessEvaluator) -> Self {
        Self {
            population_size: 100,
            n_generations: 50,
            crossover_rate: 0.80,
            mutation_rate: 0.15,
            elite_frac: 0.10,
            use_islands: true,
            migration_interval: 5,
            n_migrants: 3,
            tournament_size: 5,
            blx_alpha: 0.30,
            evaluator,
        }
    }

    // ------------------------------------------------------------------
    // Main entry point
    // ------------------------------------------------------------------

    /// Run the full evolution and return the best solutions found.
    pub fn run(&mut self) -> EvolutionResult {
        let start = std::time::Instant::now();

        if self.use_islands {
            self.run_island_model(start)
        } else {
            self.run_single_population(start)
        }
    }

    // ------------------------------------------------------------------
    // Island model
    // ------------------------------------------------------------------

    fn run_island_model(&mut self, start: std::time::Instant) -> EvolutionResult {
        let regimes = [IslandRegime::Bull, IslandRegime::Bear, IslandRegime::Neutral];
        let island_pop_size = (self.population_size / regimes.len()).max(10);

        // Initialise one population per island with independent RNG seeds.
        let mut islands: Vec<(IslandRegime, Population, Vec<f64>)> = regimes
            .iter()
            .enumerate()
            .map(|(idx, &regime)| {
                let mut rng = SmallRng::seed_from_u64(idx as u64 * 0xDEAD_BEEF + 1);
                let pop = Population::new(island_pop_size, &mut rng);
                (regime, pop, Vec::new()) // (regime, population, sharpe_history)
            })
            .collect();

        for gen in 0..self.n_generations {
            // Evaluate all islands.
            for (regime, pop, history) in islands.iter_mut() {
                pop.generation = gen as u32;
                pop.evaluate_all_parallel(&self.evaluator);
                let best = pop.best_by_sharpe().map(|g| g.sharpe()).unwrap_or(f64::NAN);
                history.push(best);

                if gen % 10 == 0 || gen == self.n_generations - 1 {
                    println!(
                        "[Island {}] gen={:>3} best_sharpe={:.3} diversity={:.4}",
                        regime,
                        gen,
                        best,
                        pop.diversity_metric()
                    );
                }
            }

            // Evolve each island.
            let mut rng = SmallRng::seed_from_u64(gen as u64 * 0xCAFE_F00D);
            for (_, pop, _) in islands.iter_mut() {
                let new_pop = self.evolve_one_generation(pop, gen as u32, &mut rng);
                *pop = new_pop;
            }

            // Island migration.
            if self.use_islands && gen > 0 && gen % self.migration_interval == 0 {
                self.migrate_islands(&mut islands);
            }
        }

        // Collect results.
        let island_results: Vec<IslandResult> = islands
            .iter()
            .map(|(regime, pop, history)| {
                let best_genome = pop
                    .best_by_sharpe()
                    .cloned()
                    .unwrap_or_else(|| pop.genomes[0].clone());

                IslandResult {
                    regime: *regime,
                    best_genome,
                    final_diversity: pop.diversity_metric(),
                    sharpe_history: history.clone(),
                }
            })
            .collect();

        // Build combined Pareto front across all islands.
        let all_genomes: Vec<Genome> = islands
            .iter()
            .flat_map(|(_, pop, _)| pop.genomes.iter().filter(|g| g.fitness.is_some()).cloned())
            .collect();

        let combined_pop = Population::from_genomes(all_genomes, self.n_generations as u32);
        let pareto_front: Vec<Genome> = combined_pop
            .get_pareto_front()
            .into_iter()
            .cloned()
            .collect();

        let global_best = combined_pop
            .best_by_sharpe()
            .cloned()
            .unwrap_or_else(|| island_results[0].best_genome.clone());

        EvolutionResult {
            island_results,
            pareto_front,
            global_best,
            elapsed_secs: start.elapsed().as_secs_f64(),
        }
    }

    // ------------------------------------------------------------------
    // Single-population fallback
    // ------------------------------------------------------------------

    fn run_single_population(&mut self, start: std::time::Instant) -> EvolutionResult {
        let mut rng = SmallRng::seed_from_u64(0xBEEF_CAFE);
        let mut pop = Population::new(self.population_size, &mut rng);
        let mut history: Vec<f64> = Vec::new();

        for gen in 0..self.n_generations {
            pop.generation = gen as u32;
            pop.evaluate_all_parallel(&self.evaluator);

            let best = pop.best_by_sharpe().map(|g| g.sharpe()).unwrap_or(f64::NAN);
            history.push(best);

            if gen % 10 == 0 || gen == self.n_generations - 1 {
                pop.print_summary();
            }

            let new_pop = self.evolve_one_generation(&pop, gen as u32, &mut rng);
            pop = new_pop;
        }

        let pareto_front: Vec<Genome> = pop.get_pareto_front().into_iter().cloned().collect();
        let global_best = pop
            .best_by_sharpe()
            .cloned()
            .unwrap_or_else(|| pop.genomes[0].clone());

        let island_result = IslandResult {
            regime: IslandRegime::Neutral,
            best_genome: global_best.clone(),
            final_diversity: pop.diversity_metric(),
            sharpe_history: history,
        };

        EvolutionResult {
            island_results: vec![island_result],
            pareto_front,
            global_best,
            elapsed_secs: start.elapsed().as_secs_f64(),
        }
    }

    // ------------------------------------------------------------------
    // One-generation evolution step
    // ------------------------------------------------------------------

    fn evolve_one_generation(
        &self,
        current: &Population,
        generation: u32,
        rng: &mut impl Rng,
    ) -> Population {
        let n = current.genomes.len();
        let n_elites = ((n as f64 * self.elite_frac).round() as usize).max(1);

        // Elitism: copy top genomes unchanged.
        let elites: Vec<Genome> = current
            .elites(n_elites)
            .iter()
            .map(|g| (*g).clone())
            .collect();

        let mut next_genomes: Vec<Genome> = elites;

        // Fill remainder with offspring.
        while next_genomes.len() < n {
            let (pa, pb) = if current.n_evaluated() >= 2 {
                let a = tournament_select(current, self.tournament_size, rng).clone();
                let b = tournament_select(current, self.tournament_size, rng).clone();
                (a, b)
            } else {
                // Fallback for very early generations with few evaluated genomes.
                let idx_a = rng.gen_range(0..n);
                let idx_b = rng.gen_range(0..n);
                (current.genomes[idx_a].clone(), current.genomes[idx_b].clone())
            };

            if rng.gen_bool(self.crossover_rate) {
                let (mut ca, mut cb) = blx_alpha(&pa, &pb, self.blx_alpha, rng);
                ca.generation = generation + 1;
                cb.generation = generation + 1;

                adaptive_mutate(&mut ca, generation, self.n_generations as u32, rng);
                if next_genomes.len() < n {
                    next_genomes.push(ca);
                }
                adaptive_mutate(&mut cb, generation, self.n_generations as u32, rng);
                if next_genomes.len() < n {
                    next_genomes.push(cb);
                }
            } else {
                // Clone one parent with mutation only.
                let mut child = pa.clone();
                child.generation = generation + 1;
                child.fitness = None; // requires re-evaluation
                adaptive_mutate(&mut child, generation, self.n_generations as u32, rng);
                next_genomes.push(child);
            }
        }

        // Truncate in case we over-filled by one.
        next_genomes.truncate(n);

        // Apply NSGA-II selection pressure to trim if needed (keeps diversity).
        // Here it acts as a sort so the elites stay at the front.
        let new_pop = Population::from_genomes(next_genomes, generation + 1);
        // Elites already have fitness; the rest need evaluation next round.
        new_pop
    }

    // ------------------------------------------------------------------
    // Island migration
    // ------------------------------------------------------------------

    fn migrate_islands(
        &self,
        islands: &mut Vec<(IslandRegime, Population, Vec<f64>)>,
    ) {
        let n_islands = islands.len();
        if n_islands < 2 {
            return;
        }

        // Collect the top-k migrants from each island (cloned, fitness cleared).
        let migrants: Vec<Vec<Genome>> = islands
            .iter()
            .map(|(_, pop, _)| {
                pop.elites(self.n_migrants)
                    .iter()
                    .map(|g| {
                        let mut migrant = (*g).clone();
                        migrant.fitness = None; // will be re-evaluated in the target regime
                        migrant
                    })
                    .collect()
            })
            .collect();

        // Ring migration: island i receives migrants from island (i+1) % n_islands.
        for i in 0..n_islands {
            let source = (i + 1) % n_islands;
            let target_pop = &mut islands[i].1;

            // Replace the worst genomes (by Sharpe, ascending) with migrants.
            let mut worst_indices: Vec<usize> = (0..target_pop.genomes.len())
                .filter(|&idx| target_pop.genomes[idx].fitness.is_some())
                .collect();
            worst_indices.sort_by(|&a, &b| {
                target_pop.genomes[a]
                    .sharpe()
                    .partial_cmp(&target_pop.genomes[b].sharpe())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (k, migrant) in migrants[source].iter().enumerate() {
                if let Some(&replace_idx) = worst_indices.get(k) {
                    target_pop.genomes[replace_idx] = migrant.clone();
                } else {
                    target_pop.genomes.push(migrant.clone());
                }
            }
        }

        println!(
            "[Migration] {} migrants exchanged between {} islands",
            self.n_migrants, n_islands
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::{EvaluatorConfig, FitnessEvaluator};

    fn dry_evolver(pop_size: usize, generations: usize) -> GeneticEvolver {
        let eval = FitnessEvaluator::new(EvaluatorConfig {
            dry_run: true,
            ..Default::default()
        });
        let mut evolver = GeneticEvolver::new(eval);
        evolver.population_size = pop_size;
        evolver.n_generations = generations;
        evolver.use_islands = false;
        evolver
    }

    #[test]
    fn single_pop_evolution_completes() {
        let mut evolver = dry_evolver(20, 3);
        let result = evolver.run();
        assert!(!result.pareto_front.is_empty());
        assert!(result.global_best.fitness.is_some());
        assert!(result.elapsed_secs >= 0.0);
    }

    #[test]
    fn island_model_completes() {
        let eval = FitnessEvaluator::new(EvaluatorConfig {
            dry_run: true,
            ..Default::default()
        });
        let mut evolver = GeneticEvolver::new(eval);
        evolver.population_size = 15;
        evolver.n_generations = 2;
        evolver.use_islands = true;
        evolver.migration_interval = 1;
        evolver.n_migrants = 1;

        let result = evolver.run();
        assert_eq!(result.island_results.len(), 3, "should have 3 islands");
        for island in &result.island_results {
            assert_eq!(island.sharpe_history.len(), 2);
        }
    }
}
