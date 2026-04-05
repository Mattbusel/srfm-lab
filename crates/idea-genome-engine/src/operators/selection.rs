/// Selection operators: tournament selection and NSGA-II.

use rand::Rng;

use crate::genome::Genome;
use crate::population::Population;

// ---------------------------------------------------------------------------
// Tournament selection
// ---------------------------------------------------------------------------

/// Select one genome via tournament selection (single-objective, by Sharpe).
///
/// Randomly samples `tournament_size` genomes from the population (with
/// replacement) and returns a reference to the one with the highest Sharpe
/// ratio.  Unevaluated genomes (fitness = None) are treated as having
/// Sharpe = −∞ and will never win a tournament against an evaluated genome.
///
/// # Arguments
/// * `population`       — the current population.
/// * `tournament_size`  — number of competitors per tournament (≥ 1).
/// * `rng`              — caller-supplied RNG.
pub fn tournament_select<'a>(
    population: &'a Population,
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a Genome {
    assert!(
        !population.genomes.is_empty(),
        "cannot select from an empty population"
    );
    assert!(tournament_size >= 1, "tournament_size must be >= 1");

    let n = population.genomes.len();
    let ts = tournament_size.min(n);

    let mut best_idx = rng.gen_range(0..n);

    for _ in 1..ts {
        let candidate_idx = rng.gen_range(0..n);
        let candidate = &population.genomes[candidate_idx];
        let current_best = &population.genomes[best_idx];

        if candidate.sharpe() > current_best.sharpe() {
            best_idx = candidate_idx;
        }
    }

    &population.genomes[best_idx]
}

/// Select a pair of parents via two independent tournaments.
pub fn tournament_select_pair<'a>(
    population: &'a Population,
    tournament_size: usize,
    rng: &mut impl Rng,
) -> (&'a Genome, &'a Genome) {
    let parent_a = tournament_select(population, tournament_size, rng);
    // Run a second independent tournament; may occasionally select the same
    // individual (acceptable in large populations).
    let parent_b = tournament_select(population, tournament_size, rng);
    (parent_a, parent_b)
}

// ---------------------------------------------------------------------------
// NSGA-II multi-objective selection
// ---------------------------------------------------------------------------

/// NSGA-II selection: returns indices of the `target_size` genomes that should
/// survive into the next generation.
///
/// Algorithm:
///   1. Assign each evaluated genome a Pareto rank (front 0 = non-dominated,
///      front 1 = dominated only by front-0, …).
///   2. Within each front, compute the crowding distance to preserve diversity.
///   3. Sort by (rank ASC, crowding_distance DESC) and return the top
///      `target_size` indices.
///
/// Unevaluated genomes receive the worst rank and zero crowding distance.
pub fn nsga2_select(population: &Population, target_size: usize) -> Vec<usize> {
    let n = population.genomes.len();
    if n == 0 {
        return vec![];
    }

    // ---- Step 1: Non-dominated sorting ----
    let fronts = non_dominated_sort(&population.genomes);

    // Flatten fronts into a rank vector (rank[i] = front index for genome i).
    let mut rank = vec![usize::MAX; n];
    for (front_idx, front) in fronts.iter().enumerate() {
        for &genome_idx in front {
            rank[genome_idx] = front_idx;
        }
    }

    // ---- Step 2: Crowding distance ----
    let crowding = crowding_distance(&population.genomes, &fronts);

    // ---- Step 3: Sort and truncate ----
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let ra = rank[a];
        let rb = rank[b];
        if ra != rb {
            ra.cmp(&rb) // lower rank is better
        } else {
            // Higher crowding distance is better (more isolated → more diverse).
            crowding[b]
                .partial_cmp(&crowding[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    order.into_iter().take(target_size).collect()
}

// ---------------------------------------------------------------------------
// Pareto rank helpers
// ---------------------------------------------------------------------------

/// Non-dominated sorting: returns a list of fronts, each front being a sorted
/// list of genome indices.  Front 0 is the Pareto-optimal set.
pub fn non_dominated_sort(genomes: &[Genome]) -> Vec<Vec<usize>> {
    let n = genomes.len();

    // domination_count[i] = number of genomes that dominate genome i.
    let mut domination_count: Vec<usize> = vec![0; n];
    // dominated_by[i]      = list of genomes that genome i dominates.
    let mut dominated_by: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if genomes[i].dominates(&genomes[j]) {
                dominated_by[i].push(j);
            } else if genomes[j].dominates(&genomes[i]) {
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = (0..n)
        .filter(|&i| domination_count[i] == 0)
        .collect();

    while !current_front.is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &i in &current_front {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(std::mem::take(&mut current_front));
        current_front = next_front;
    }

    fronts
}

/// Compute NSGA-II crowding distance for all genomes.
///
/// Returns a `Vec<f64>` where index `i` gives the crowding distance of genome
/// `i`.  Boundary individuals within each front receive distance = ∞.
pub fn crowding_distance(genomes: &[Genome], fronts: &[Vec<usize>]) -> Vec<f64> {
    let n = genomes.len();
    let mut distance = vec![0.0_f64; n];

    // Objectives (all "higher is better" after sign flip for minimisation objectives).
    // We treat each metric independently.
    let n_objectives = 6;
    let objective = |g: &Genome, k: usize| -> f64 {
        let Some(ref f) = g.fitness else { return f64::NEG_INFINITY };
        match k {
            0 => f.sharpe,
            1 => f.calmar,
            2 => f.win_rate,
            3 => f.profit_factor,
            4 => -f.max_dd,
            5 => -f.is_oos_spread,
            _ => unreachable!(),
        }
    };

    for front in fronts {
        if front.len() <= 2 {
            // All individuals in a tiny front get infinite distance.
            for &idx in front {
                distance[idx] = f64::INFINITY;
            }
            continue;
        }

        for m in 0..n_objectives {
            // Sort front by objective m.
            let mut sorted = front.clone();
            sorted.sort_by(|&a, &b| {
                objective(&genomes[a], m)
                    .partial_cmp(&objective(&genomes[b], m))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Boundary individuals.
            distance[sorted[0]] = f64::INFINITY;
            distance[*sorted.last().unwrap()] = f64::INFINITY;

            let f_min = objective(&genomes[sorted[0]], m);
            let f_max = objective(&genomes[*sorted.last().unwrap()], m);
            let range = f_max - f_min;

            if range < f64::EPSILON {
                continue;
            }

            for k in 1..(sorted.len() - 1) {
                let prev = objective(&genomes[sorted[k - 1]], m);
                let next = objective(&genomes[sorted[k + 1]], m);
                if distance[sorted[k]].is_finite() {
                    distance[sorted[k]] += (next - prev) / range;
                }
            }
        }
    }

    distance
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
    use crate::population::Population;

    fn evaluated_population(size: usize, seed: u64) -> Population {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut pop = Population::new(size, &mut rng);
        let eval = FitnessEvaluator::new(EvaluatorConfig {
            dry_run: true,
            ..Default::default()
        });
        pop.evaluate_all_parallel(&eval);
        pop
    }

    #[test]
    fn tournament_returns_valid_genome() {
        let pop = evaluated_population(20, 1);
        let mut rng = SmallRng::seed_from_u64(42);
        let winner = tournament_select(&pop, 5, &mut rng);
        assert!(winner.fitness.is_some());
    }

    #[test]
    fn tournament_prefers_higher_sharpe() {
        // Over many trials the tournament winner should have above-median Sharpe.
        let pop = evaluated_population(30, 7);
        let mut rng = SmallRng::seed_from_u64(77);

        let evaluated: Vec<&Genome> = pop.genomes.iter().filter(|g| g.fitness.is_some()).collect();
        let sharpes: Vec<f64> = evaluated.iter().map(|g| g.sharpe()).collect();
        let median = {
            let mut s = sharpes.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        };

        let mut wins_above_median = 0usize;
        let trials = 200;
        for _ in 0..trials {
            let winner = tournament_select(&pop, 5, &mut rng);
            if winner.sharpe() >= median {
                wins_above_median += 1;
            }
        }
        // Tournament of 5 should heavily favour above-median genomes.
        assert!(
            wins_above_median > trials * 6 / 10,
            "tournament should favour above-median genomes (wins: {}/{})",
            wins_above_median,
            trials
        );
    }

    #[test]
    fn nsga2_select_returns_correct_count() {
        let pop = evaluated_population(30, 9);
        let selected = nsga2_select(&pop, 15);
        assert_eq!(selected.len(), 15);
        // All indices must be valid.
        for &idx in &selected {
            assert!(idx < pop.genomes.len());
        }
    }

    #[test]
    fn nsga2_select_no_duplicates() {
        let pop = evaluated_population(20, 11);
        let selected = nsga2_select(&pop, 10);
        let mut sorted = selected.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), selected.len(), "NSGA-II returned duplicate indices");
    }
}
