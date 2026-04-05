/// Population management for the genetic programming engine.
///
/// - Initialise with random trees of depth 2–5.
/// - Track Pareto front (IC vs complexity).
/// - Non-dominated sorting (NSGA-II style).

use crate::data_loader::BarData;
use crate::expression_tree::SignalTree;
use crate::fitness::{evaluate_fitness, FitnessVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A population of evolved signal trees.
#[derive(Debug)]
pub struct Population {
    pub individuals: Vec<SignalTree>,
    pub generation: u32,
    /// Historical Pareto-front individuals (archive).
    pub pareto_archive: Vec<SignalTree>,
}

impl Population {
    // ------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------

    /// Create a new population with `size` random individuals.
    pub fn new(size: usize, rng: &mut impl rand::Rng) -> Self {
        let individuals: Vec<SignalTree> = (0..size)
            .map(|_| SignalTree::random(5, 0, rng))
            .collect();
        Self {
            individuals,
            generation: 0,
            pareto_archive: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// Evaluate all unevaluated individuals against the bar history in parallel.
    pub fn evaluate_all(&mut self, bars: &[BarData]) {
        // Parallel fitness evaluation using rayon
        let fitnesses: Vec<FitnessVector> = self
            .individuals
            .par_iter()
            .map(|ind| evaluate_fitness(ind, bars))
            .collect();

        for (ind, fv) in self.individuals.iter_mut().zip(fitnesses.into_iter()) {
            ind.fitness = Some(fv);
        }
    }

    // ------------------------------------------------------------------
    // Pareto front computation
    // ------------------------------------------------------------------

    /// Compute the non-dominated Pareto front using NSGA-II fast non-dominated sort.
    /// Returns front indices sorted by decreasing IC.
    pub fn pareto_front(&self) -> Vec<usize> {
        let n = self.individuals.len();
        let mut dominated_count = vec![0usize; n];
        let mut dominates_set: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let fi = self.individuals[i].fitness.as_ref();
                let fj = self.individuals[j].fitness.as_ref();
                if let (Some(fi), Some(fj)) = (fi, fj) {
                    if fi.dominates(fj) {
                        dominates_set[i].push(j);
                    } else if fj.dominates(fi) {
                        dominated_count[i] += 1;
                    }
                }
            }
        }

        // Front 0: individuals not dominated by anyone
        let front: Vec<usize> = (0..n)
            .filter(|&i| {
                dominated_count[i] == 0
                    && self.individuals[i].fitness.is_some()
            })
            .collect();

        // Sort front by descending IC
        let mut front = front;
        front.sort_by(|&a, &b| {
            let ic_a = self.individuals[a].ic();
            let ic_b = self.individuals[b].ic();
            ic_b.partial_cmp(&ic_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        front
    }

    /// Update the Pareto archive with the current front.
    pub fn update_archive(&mut self) {
        let front_ids = self.pareto_front();
        for &idx in &front_ids {
            let ind = &self.individuals[idx];
            // Only add if it isn't already in the archive (by formula)
            let formula = ind.formula();
            if !self.pareto_archive.iter().any(|a| a.formula() == formula) {
                self.pareto_archive.push(ind.clone());
            }
        }
        // Trim archive to 50 best by IC
        self.pareto_archive
            .sort_by(|a, b| b.ic().partial_cmp(&a.ic()).unwrap_or(std::cmp::Ordering::Equal));
        self.pareto_archive.truncate(50);
    }

    // ------------------------------------------------------------------
    // Selection
    // ------------------------------------------------------------------

    /// Tournament selection: pick k individuals at random, return the one with highest IC.
    pub fn tournament_select(&self, k: usize, rng: &mut impl rand::Rng) -> &SignalTree {
        let n = self.individuals.len();
        let candidates: Vec<usize> = (0..k.min(n))
            .map(|_| rng.gen_range(0..n))
            .collect();
        candidates
            .iter()
            .max_by(|&&a, &&b| {
                self.individuals[a]
                    .ic()
                    .partial_cmp(&self.individuals[b].ic())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|&i| &self.individuals[i])
            .unwrap_or(&self.individuals[0])
    }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    /// Best IC in the current population.
    pub fn best_ic(&self) -> f64 {
        self.individuals
            .iter()
            .filter_map(|i| i.fitness.as_ref())
            .map(|f| f.ic)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Mean IC of the population.
    pub fn mean_ic(&self) -> f64 {
        let vals: Vec<f64> = self
            .individuals
            .iter()
            .filter_map(|i| i.fitness.as_ref())
            .map(|f| f.ic)
            .collect();
        if vals.is_empty() {
            return 0.0;
        }
        vals.iter().sum::<f64>() / vals.len() as f64
    }

    /// Individual with the highest IC.
    pub fn best_individual(&self) -> Option<&SignalTree> {
        self.individuals
            .iter()
            .filter(|i| i.fitness.is_some())
            .max_by(|a, b| a.ic().partial_cmp(&b.ic()).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Return sorted individuals by descending IC score.
    pub fn sorted_by_ic(&self) -> Vec<&SignalTree> {
        let mut sorted: Vec<&SignalTree> = self.individuals.iter().collect();
        sorted.sort_by(|a, b| b.ic().partial_cmp(&a.ic()).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Population summary statistics as a serialisable struct.
    pub fn stats(&self) -> PopulationStats {
        let evaluated: Vec<&FitnessVector> = self
            .individuals
            .iter()
            .filter_map(|i| i.fitness.as_ref())
            .collect();
        if evaluated.is_empty() {
            return PopulationStats::default();
        }
        let n = evaluated.len() as f64;
        let ics: Vec<f64> = evaluated.iter().map(|f| f.ic).collect();
        let complexities: Vec<f64> = self
            .individuals
            .iter()
            .map(|i| i.complexity() as f64)
            .collect();

        PopulationStats {
            generation: self.generation,
            size: self.individuals.len(),
            best_ic: ics.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            mean_ic: ics.iter().sum::<f64>() / n,
            pareto_front_size: self.pareto_front().len(),
            mean_complexity: complexities.iter().sum::<f64>() / complexities.len() as f64,
        }
    }
}

/// Summary statistics for a population at one generation.
#[derive(Debug, Serialize, Deserialize)]
pub struct PopulationStats {
    pub generation: u32,
    pub size: usize,
    pub best_ic: f64,
    pub mean_ic: f64,
    pub pareto_front_size: usize,
    pub mean_complexity: f64,
}

impl Default for PopulationStats {
    fn default() -> Self {
        Self {
            generation: 0,
            size: 0,
            best_ic: 0.0,
            mean_ic: 0.0,
            pareto_front_size: 0,
            mean_complexity: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::synthetic_bars;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    #[test]
    fn population_initialises_correct_size() {
        let mut rng = rng();
        let pop = Population::new(20, &mut rng);
        assert_eq!(pop.individuals.len(), 20);
    }

    #[test]
    fn evaluate_all_sets_fitness() {
        let mut rng = rng();
        let mut pop = Population::new(5, &mut rng);
        let bars = synthetic_bars(100, 100.0);
        pop.evaluate_all(&bars);
        for ind in &pop.individuals {
            assert!(ind.fitness.is_some(), "Fitness should be set after evaluation");
        }
    }

    #[test]
    fn best_ic_after_evaluation() {
        let mut rng = rng();
        let mut pop = Population::new(10, &mut rng);
        let bars = synthetic_bars(200, 100.0);
        pop.evaluate_all(&bars);
        let best = pop.best_ic();
        assert!(best.is_finite(), "Best IC should be finite");
    }

    #[test]
    fn pareto_front_non_empty_after_eval() {
        let mut rng = rng();
        let mut pop = Population::new(10, &mut rng);
        let bars = synthetic_bars(200, 100.0);
        pop.evaluate_all(&bars);
        let front = pop.pareto_front();
        assert!(!front.is_empty());
    }

    #[test]
    fn tournament_select_returns_individual() {
        let mut rng = rng();
        let mut pop = Population::new(20, &mut rng);
        let bars = synthetic_bars(100, 100.0);
        pop.evaluate_all(&bars);
        let mut rng2 = SmallRng::seed_from_u64(99);
        let selected = pop.tournament_select(7, &mut rng2);
        assert!(selected.fitness.is_some());
    }
}
