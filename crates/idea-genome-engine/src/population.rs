/// Population management: collection of genomes, parallel evaluation, Pareto helpers.

use rayon::prelude::*;
use rand::Rng;

use crate::fitness::{FitnessEvaluator, FitnessVec};
use crate::genome::Genome;

// ---------------------------------------------------------------------------
// Population
// ---------------------------------------------------------------------------

/// A generation of candidate genomes.
#[derive(Debug)]
pub struct Population {
    /// All genomes in this generation (evaluated or not).
    pub genomes: Vec<Genome>,
    /// Zero-based generation index.
    pub generation: u32,
    /// Cached best fitness (updated after each evaluation sweep).
    pub best_fitness: Option<FitnessVec>,
}

impl Population {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create an initial random population of `size` genomes.
    pub fn new(size: usize, rng: &mut impl Rng) -> Self {
        let genomes: Vec<Genome> = (0..size).map(|_| Genome::new_random(rng)).collect();
        Self {
            genomes,
            generation: 0,
            best_fitness: None,
        }
    }

    /// Create a population from an existing list of genomes (used between generations).
    pub fn from_genomes(genomes: Vec<Genome>, generation: u32) -> Self {
        Self {
            genomes,
            generation,
            best_fitness: None,
        }
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// Evaluate all unevaluated genomes in parallel using Rayon.
    ///
    /// Genomes that already have `fitness` set are skipped (elites).
    pub fn evaluate_all_parallel(&mut self, evaluator: &FitnessEvaluator) {
        // Collect indices that need evaluation.
        let needs_eval: Vec<usize> = self
            .genomes
            .iter()
            .enumerate()
            .filter(|(_, g)| g.fitness.is_none())
            .map(|(i, _)| i)
            .collect();

        if needs_eval.is_empty() {
            return;
        }

        // Parallel evaluation — each thread gets a reference to the evaluator and the genome.
        let results: Vec<(usize, FitnessVec)> = needs_eval
            .par_iter()
            .map(|&idx| {
                let fv = evaluator.evaluate(&self.genomes[idx]);
                (idx, fv)
            })
            .collect();

        for (idx, fv) in results {
            self.genomes[idx].fitness = Some(fv);
        }

        // Update cached best.
        self.best_fitness = self
            .best_by_sharpe()
            .and_then(|g| g.fitness.clone());
    }

    // ------------------------------------------------------------------
    // Selection helpers
    // ------------------------------------------------------------------

    /// Return references to all Pareto-optimal (non-dominated) genomes.
    ///
    /// A genome `a` dominates `b` iff `a` is at least as good on every
    /// objective and strictly better on at least one (`Genome::dominates`).
    pub fn get_pareto_front(&self) -> Vec<&Genome> {
        let evaluated: Vec<&Genome> = self
            .genomes
            .iter()
            .filter(|g| g.fitness.is_some())
            .collect();

        evaluated
            .iter()
            .copied()
            .filter(|candidate| {
                !evaluated
                    .iter()
                    .any(|other| other.dominates(candidate) && other.id != candidate.id)
            })
            .collect()
    }

    /// Return the genome with the highest Sharpe ratio, or `None` if the
    /// population is empty / no genome has been evaluated.
    pub fn best_by_sharpe(&self) -> Option<&Genome> {
        self.genomes
            .iter()
            .filter(|g| g.fitness.is_some())
            .max_by(|a, b| a.sharpe().partial_cmp(&b.sharpe()).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Return the genome with the highest Calmar ratio.
    pub fn best_by_calmar(&self) -> Option<&Genome> {
        self.genomes
            .iter()
            .filter(|g| g.fitness.is_some())
            .max_by(|a, b| a.calmar().partial_cmp(&b.calmar()).unwrap_or(std::cmp::Ordering::Equal))
    }

    // ------------------------------------------------------------------
    // Diversity
    // ------------------------------------------------------------------

    /// Mean pairwise normalised Euclidean distance across all evaluated genomes.
    ///
    /// Returns 0.0 when fewer than 2 genomes have been evaluated.
    pub fn diversity_metric(&self) -> f64 {
        let evaluated: Vec<&Genome> = self
            .genomes
            .iter()
            .filter(|g| g.fitness.is_some())
            .collect();

        let n = evaluated.len();
        if n < 2 {
            return 0.0;
        }

        let mut total_dist = 0.0_f64;
        let mut count = 0u64;

        for i in 0..n {
            for j in (i + 1)..n {
                total_dist += evaluated[i].parameter_distance(evaluated[j]);
                count += 1;
            }
        }

        if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        }
    }

    /// Return up to `n` elite genomes sorted by Sharpe (descending).
    pub fn elites(&self, n: usize) -> Vec<&Genome> {
        let mut evaluated: Vec<&Genome> = self
            .genomes
            .iter()
            .filter(|g| g.fitness.is_some())
            .collect();

        evaluated.sort_by(|a, b| {
            b.sharpe()
                .partial_cmp(&a.sharpe())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        evaluated.into_iter().take(n).collect()
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    /// Number of evaluated genomes.
    pub fn n_evaluated(&self) -> usize {
        self.genomes.iter().filter(|g| g.fitness.is_some()).count()
    }

    /// Print a one-line summary to stdout.
    pub fn print_summary(&self) {
        let best_sharpe = self
            .best_by_sharpe()
            .map(|g| g.sharpe())
            .unwrap_or(f64::NAN);

        println!(
            "Gen {:>3} | size={} evaluated={} best_sharpe={:.3} diversity={:.4}",
            self.generation,
            self.genomes.len(),
            self.n_evaluated(),
            best_sharpe,
            self.diversity_metric(),
        );
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
    use crate::fitness::{EvaluatorConfig, FitnessEvaluator};

    fn dry_evaluator() -> FitnessEvaluator {
        FitnessEvaluator::new(EvaluatorConfig {
            dry_run: true,
            ..Default::default()
        })
    }

    #[test]
    fn population_new_correct_size() {
        let mut rng = SmallRng::seed_from_u64(1);
        let pop = Population::new(20, &mut rng);
        assert_eq!(pop.genomes.len(), 20);
        assert_eq!(pop.n_evaluated(), 0);
    }

    #[test]
    fn evaluate_all_parallel_fills_fitness() {
        let mut rng = SmallRng::seed_from_u64(2);
        let mut pop = Population::new(10, &mut rng);
        let eval = dry_evaluator();
        pop.evaluate_all_parallel(&eval);
        assert_eq!(pop.n_evaluated(), 10);
    }

    #[test]
    fn pareto_front_subset_of_population() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut pop = Population::new(20, &mut rng);
        let eval = dry_evaluator();
        pop.evaluate_all_parallel(&eval);
        let front = pop.get_pareto_front();
        assert!(!front.is_empty());
        assert!(front.len() <= pop.genomes.len());
    }

    #[test]
    fn diversity_metric_positive() {
        let mut rng = SmallRng::seed_from_u64(4);
        let mut pop = Population::new(10, &mut rng);
        let eval = dry_evaluator();
        pop.evaluate_all_parallel(&eval);
        let d = pop.diversity_metric();
        assert!(d >= 0.0, "diversity must be non-negative");
    }
}
