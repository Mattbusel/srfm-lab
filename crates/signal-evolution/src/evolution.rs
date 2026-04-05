/// Main evolution loop for the genetic programming signal engine.
///
/// Strategy:
///   - Tournament selection (k=7)
///   - Subtree crossover (p=0.8)
///   - Mutation: point / subtree / hoist (p=0.3)
///   - Elitism: keep top 10% unchanged
///   - Run for N generations
///   - Emit top signals as JSON

use crate::data_loader::BarData;
use crate::expression_tree::SignalTree;
use crate::operators::crossover::subtree_crossover;
use crate::operators::mutation::mutate;
use crate::population::{Population, PopulationStats};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for the evolution run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub generations: usize,
    /// Probability of applying crossover to a pair.
    pub crossover_prob: f64,
    /// Probability of applying mutation to an offspring.
    pub mutation_prob: f64,
    /// Tournament size for selection.
    pub tournament_k: usize,
    /// Fraction of best individuals preserved unchanged.
    pub elitism_frac: f64,
    /// Maximum tree depth.
    pub max_depth: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 50,
            crossover_prob: 0.8,
            mutation_prob: 0.3,
            tournament_k: 7,
            elitism_frac: 0.10,
            max_depth: 6,
        }
    }
}

/// Result of a completed evolution run.
#[derive(Debug, Serialize, Deserialize)]
pub struct EvolutionResult {
    pub config: EvolutionConfig,
    pub best_signals: Vec<SignalSummary>,
    pub generation_stats: Vec<PopulationStats>,
    pub pareto_archive: Vec<SignalSummary>,
}

/// Serialisable summary of a discovered signal.
#[derive(Debug, Serialize, Deserialize)]
pub struct SignalSummary {
    pub id: String,
    pub formula: String,
    pub depth: usize,
    pub complexity: usize,
    pub generation: u32,
    pub ic: f64,
    pub icir: f64,
    pub sharpe_contrib: f64,
    pub turnover: f64,
    pub score: f64,
}

impl From<&SignalTree> for SignalSummary {
    fn from(st: &SignalTree) -> Self {
        let fv = st.fitness.as_ref();
        Self {
            id: st.id.clone(),
            formula: st.formula(),
            depth: st.depth(),
            complexity: st.complexity(),
            generation: st.generation,
            ic: fv.map(|f| f.ic).unwrap_or(0.0),
            icir: fv.map(|f| f.icir).unwrap_or(0.0),
            sharpe_contrib: fv.map(|f| f.sharpe_contrib).unwrap_or(0.0),
            turnover: fv.map(|f| f.turnover).unwrap_or(0.0),
            score: fv.map(|f| f.score).unwrap_or(f64::NEG_INFINITY),
        }
    }
}

// ---------------------------------------------------------------------------
// Main evolution engine
// ---------------------------------------------------------------------------

/// Run the full genetic programming evolution loop.
pub fn run_evolution(
    bars: &[BarData],
    config: EvolutionConfig,
    rng: &mut impl Rng,
) -> EvolutionResult {
    let mut pop = Population::new(config.population_size, rng);
    pop.evaluate_all(bars);

    let mut generation_stats: Vec<PopulationStats> = Vec::with_capacity(config.generations);
    let n_elite = (config.population_size as f64 * config.elitism_frac).round() as usize;

    for gen in 0..config.generations {
        pop.generation = gen as u32;

        // Collect stats before breeding
        pop.update_archive();
        let stats = pop.stats();
        eprintln!(
            "Gen {:3} | Best IC: {:+.4} | Mean IC: {:+.4} | Front: {} | Avg complexity: {:.1}",
            gen, stats.best_ic, stats.mean_ic, stats.pareto_front_size, stats.mean_complexity
        );
        generation_stats.push(stats);

        // Build next generation
        let mut next_gen: Vec<SignalTree> = Vec::with_capacity(config.population_size);

        // Elitism: copy top n_elite individuals unchanged
        let sorted = pop.sorted_by_ic();
        for ind in sorted.iter().take(n_elite) {
            next_gen.push((*ind).clone());
        }

        // Fill remainder with crossover / mutation offspring
        while next_gen.len() < config.population_size {
            let parent_a = pop.tournament_select(config.tournament_k, rng).clone();
            let offspring = if rng.gen_bool(config.crossover_prob) {
                let parent_b = pop.tournament_select(config.tournament_k, rng);
                let child_tree = subtree_crossover(
                    &parent_a.tree,
                    &parent_b.tree,
                    config.max_depth,
                    rng,
                );
                SignalTree::new(child_tree, gen as u32 + 1)
            } else {
                parent_a.clone()
            };

            let final_offspring = if rng.gen_bool(config.mutation_prob) {
                let mutated_tree = mutate(&offspring.tree, config.max_depth, rng);
                SignalTree::new(mutated_tree, gen as u32 + 1)
            } else {
                offspring
            };

            next_gen.push(final_offspring);
        }

        pop.individuals = next_gen;
        pop.evaluate_all(bars);
    }

    // Final stats and archive update
    pop.update_archive();
    generation_stats.push(pop.stats());

    // Top 10 signals by IC
    let best_signals: Vec<SignalSummary> = pop
        .sorted_by_ic()
        .iter()
        .take(10)
        .map(|s| SignalSummary::from(*s))
        .collect();

    let pareto_archive: Vec<SignalSummary> = pop
        .pareto_archive
        .iter()
        .take(20)
        .map(SignalSummary::from)
        .collect();

    EvolutionResult {
        config,
        best_signals,
        generation_stats,
        pareto_archive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::synthetic_bars;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn short_evolution_run_completes() {
        let bars = synthetic_bars(200, 100.0);
        let config = EvolutionConfig {
            population_size: 10,
            generations: 3,
            ..Default::default()
        };
        let mut rng = SmallRng::seed_from_u64(42);
        let result = run_evolution(&bars, config, &mut rng);
        assert!(!result.best_signals.is_empty());
        assert_eq!(result.generation_stats.len(), 4); // 3 + 1 final
    }

    #[test]
    fn evolution_result_serialises() {
        let bars = synthetic_bars(150, 100.0);
        let config = EvolutionConfig {
            population_size: 5,
            generations: 2,
            ..Default::default()
        };
        let mut rng = SmallRng::seed_from_u64(7);
        let result = run_evolution(&bars, config, &mut rng);
        let json = serde_json::to_string(&result).expect("serialisation should succeed");
        assert!(json.contains("best_signals"));
    }
}
