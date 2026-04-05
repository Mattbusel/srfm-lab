/// Multi-objective optimizer using NSGA-II on the strategy parameter space.
///
/// Objectives (all minimised internally):
///   - obj0: -Sharpe  (maximise Sharpe → minimise -Sharpe)
///   - obj1: MaxDrawdown  (minimise drawdown)
///   - obj2: -(WinRate)   (maximise win rate → minimise -win_rate)
use crate::backtest::{run_backtest, BacktestResult};
use crate::bar_data::DataStore;
use crate::params::{ParameterSpace, StrategyParams};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const POP_SIZE: usize = 200;
const N_GENERATIONS: usize = 100;
const TOP_PARETO: usize = 20;
const TOURNAMENT_K: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoResult {
    pub params: StrategyParams,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub cagr: f64,
    pub profit_factor: f64,
}

impl ParetoResult {
    fn objectives(&self) -> [f64; 3] {
        [-self.sharpe, self.max_drawdown, -self.win_rate]
    }
}

/// Returns `true` if `a` Pareto-dominates `b` (a is better or equal on all, strictly better on one).
fn dominates(a: &[f64; 3], b: &[f64; 3]) -> bool {
    let any_better = a.iter().zip(b.iter()).any(|(ai, bi)| ai < bi);
    let none_worse = a.iter().zip(b.iter()).all(|(ai, bi)| ai <= bi);
    any_better && none_worse
}

/// Compute non-dominated fronts (NSGA-II fast non-dominated sort).
fn non_dominated_sort(population: &[ParetoResult]) -> Vec<Vec<usize>> {
    let n = population.len();
    let mut dominated_by: Vec<Vec<usize>> = vec![vec![]; n];
    let mut domination_count: Vec<usize> = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = vec![vec![]];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let oi = population[i].objectives();
            let oj = population[j].objectives();
            if dominates(&oi, &oj) {
                dominated_by[i].push(j);
            } else if dominates(&oj, &oi) {
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            fronts[0].push(i);
        }
    }

    let mut front_idx = 0;
    while !fronts[front_idx].is_empty() {
        let mut next_front = vec![];
        for &i in &fronts[front_idx].clone() {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(next_front);
        front_idx += 1;
    }

    fronts.retain(|f| !f.is_empty());
    fronts
}

/// Crowding distance assignment within a front.
fn crowding_distance(front: &[usize], population: &[ParetoResult]) -> Vec<f64> {
    let n = front.len();
    let mut distances = vec![0.0_f64; n];

    for obj_idx in 0..3 {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            population[front[a]].objectives()[obj_idx]
                .partial_cmp(&population[front[b]].objectives()[obj_idx])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        distances[order[0]] = f64::INFINITY;
        distances[order[n - 1]] = f64::INFINITY;

        let min_val = population[front[order[0]]].objectives()[obj_idx];
        let max_val = population[front[order[n - 1]]].objectives()[obj_idx];
        let range = (max_val - min_val).max(1e-12);

        for k in 1..n - 1 {
            let prev = population[front[order[k - 1]]].objectives()[obj_idx];
            let next = population[front[order[k + 1]]].objectives()[obj_idx];
            distances[order[k]] += (next - prev) / range;
        }
    }

    distances
}

/// Tournament selection: select best individual by rank then crowding distance.
fn tournament_select<R: Rng>(
    population: &[ParetoResult],
    ranks: &[usize],
    crowding: &[f64],
    rng: &mut R,
) -> usize {
    let candidates: Vec<usize> = (0..TOURNAMENT_K)
        .map(|_| rng.gen_range(0..population.len()))
        .collect();

    candidates
        .into_iter()
        .min_by(|&a, &b| {
            // Lower rank = better; tie-break by higher crowding distance.
            ranks[a].cmp(&ranks[b]).then_with(|| {
                crowding[b]
                    .partial_cmp(&crowding[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
        .unwrap()
}

/// Run NSGA-II multi-objective optimisation.
///
/// # Arguments
/// * `data` — historical bar data for backtesting.
/// * `space` — parameter space to sample from.
///
/// # Returns
/// Top-20 Pareto-optimal parameter sets.
pub fn multi_objective_optimize(data: &DataStore, space: &ParameterSpace) -> Vec<ParetoResult> {
    let mut rng = rand::thread_rng();

    // Initialise population with LHS.
    let init_params = space.sample(POP_SIZE);
    let mut population: Vec<ParetoResult> = init_params
        .iter()
        .map(|p| {
            let r = run_backtest(data, p);
            result_to_pareto(p, &r)
        })
        .collect();

    eprintln!("[nsga2] Generation 0 / {} | pop={}", N_GENERATIONS, POP_SIZE);

    for gen in 1..=N_GENERATIONS {
        // Compute ranks and crowding distances.
        let fronts = non_dominated_sort(&population);
        let mut ranks = vec![0usize; population.len()];
        let mut crowding_all = vec![0.0_f64; population.len()];

        for (rank, front) in fronts.iter().enumerate() {
            let cd = crowding_distance(front, &population);
            for (local_i, &global_i) in front.iter().enumerate() {
                ranks[global_i] = rank;
                crowding_all[global_i] = cd[local_i];
            }
        }

        // Generate offspring.
        let mut offspring: Vec<ParetoResult> = Vec::with_capacity(POP_SIZE);
        while offspring.len() < POP_SIZE {
            let parent_idx = tournament_select(&population, &ranks, &crowding_all, &mut rng);
            let parent = &population[parent_idx].params;
            let child_params = parent.random_neighbor(&mut rng, 0.3);
            let r = run_backtest(data, &child_params);
            offspring.push(result_to_pareto(&child_params, &r));
        }

        // Merge parent + offspring, re-rank, keep top POP_SIZE.
        population.extend(offspring);
        let combined_fronts = non_dominated_sort(&population);

        let mut next_pop: Vec<ParetoResult> = Vec::with_capacity(POP_SIZE);
        let mut seen: HashSet<usize> = HashSet::new();

        'outer: for front in &combined_fronts {
            if next_pop.len() + front.len() <= POP_SIZE {
                for &i in front {
                    if seen.insert(i) {
                        next_pop.push(population[i].clone());
                    }
                }
            } else {
                // Fill remaining slots by crowding distance.
                let cd = crowding_distance(front, &population);
                let mut order: Vec<usize> = (0..front.len()).collect();
                order.sort_unstable_by(|&a, &b| {
                    cd[b].partial_cmp(&cd[a]).unwrap_or(std::cmp::Ordering::Equal)
                });
                for &local_i in &order {
                    let global_i = front[local_i];
                    if seen.insert(global_i) {
                        next_pop.push(population[global_i].clone());
                        if next_pop.len() >= POP_SIZE {
                            break 'outer;
                        }
                    }
                }
                break;
            }
        }

        population = next_pop;

        if gen % 10 == 0 {
            let best_sharpe = population
                .iter()
                .map(|p| p.sharpe)
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "[nsga2] Generation {} / {} | best_sharpe={:.3}",
                gen, N_GENERATIONS, best_sharpe
            );
        }
    }

    // Extract Pareto front from final population.
    let final_fronts = non_dominated_sort(&population);
    let pareto_indices = &final_fronts[0];

    let mut pareto: Vec<ParetoResult> = pareto_indices
        .iter()
        .map(|&i| population[i].clone())
        .collect();

    // Sort by Sharpe (secondary sort for determinism).
    pareto.sort_unstable_by(|a, b| {
        b.sharpe
            .partial_cmp(&a.sharpe)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    pareto.truncate(TOP_PARETO);
    pareto
}

fn result_to_pareto(params: &StrategyParams, r: &BacktestResult) -> ParetoResult {
    ParetoResult {
        params: params.clone(),
        sharpe: r.sharpe,
        max_drawdown: r.max_drawdown,
        win_rate: r.win_rate,
        cagr: r.cagr,
        profit_factor: r.profit_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominates_basic() {
        let a = [-2.0, 0.1, -0.6];
        let b = [-1.0, 0.3, -0.5];
        assert!(dominates(&a, &b), "a dominates b on all objectives");
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn test_dominates_equal() {
        let a = [1.0, 1.0, 1.0];
        assert!(!dominates(&a, &a), "no self-domination");
    }

    #[test]
    fn test_non_dominated_sort_single() {
        let pop = vec![ParetoResult {
            params: StrategyParams::default(),
            sharpe: 1.0,
            max_drawdown: 0.2,
            win_rate: 0.6,
            cagr: 0.5,
            profit_factor: 1.5,
        }];
        let fronts = non_dominated_sort(&pop);
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0], vec![0]);
    }

    #[test]
    fn test_non_dominated_sort_two_fronts() {
        let make = |s: f64, dd: f64, wr: f64| ParetoResult {
            params: StrategyParams::default(),
            sharpe: s, max_drawdown: dd, win_rate: wr, cagr: 0.0, profit_factor: 0.0,
        };
        let pop = vec![make(2.0, 0.10, 0.70), make(1.0, 0.30, 0.50)];
        let fronts = non_dominated_sort(&pop);
        // First individual dominates second on all objectives.
        assert!(fronts[0].contains(&0));
        assert!(fronts.len() >= 2);
        assert!(fronts[1].contains(&1));
    }
}
