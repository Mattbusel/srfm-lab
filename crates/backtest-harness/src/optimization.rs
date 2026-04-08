// optimization.rs — Grid search, random search, walk-forward optimization, parameter stability
use std::collections::HashMap;

/// Parameter specification
#[derive(Clone, Debug)]
pub enum ParamSpec {
    IntRange { min: i64, max: i64, step: i64 },
    FloatRange { min: f64, max: f64, step: f64 },
    Choice(Vec<f64>),
    LogRange { min: f64, max: f64, num_points: usize },
}

impl ParamSpec {
    pub fn values(&self) -> Vec<f64> {
        match self {
            ParamSpec::IntRange { min, max, step } => {
                let mut v = Vec::new();
                let mut i = *min;
                while i <= *max { v.push(i as f64); i += step; }
                v
            }
            ParamSpec::FloatRange { min, max, step } => {
                let mut v = Vec::new();
                let mut x = *min;
                while x <= *max + 1e-10 { v.push(x); x += step; }
                v
            }
            ParamSpec::Choice(vals) => vals.clone(),
            ParamSpec::LogRange { min, max, num_points } => {
                let log_min = min.ln();
                let log_max = max.ln();
                let n = *num_points;
                (0..n).map(|i| {
                    let t = i as f64 / (n - 1).max(1) as f64;
                    (log_min + t * (log_max - log_min)).exp()
                }).collect()
            }
        }
    }

    pub fn num_values(&self) -> usize { self.values().len() }
}

/// Parameter set (named parameters)
pub type ParamSet = HashMap<String, f64>;

/// Optimization result for a single parameter combination
#[derive(Clone, Debug)]
pub struct OptResult {
    pub params: ParamSet,
    pub objective: f64,
    pub metrics: HashMap<String, f64>,
}

/// Grid search over parameter space
pub struct GridSearch {
    pub param_specs: Vec<(String, ParamSpec)>,
}

impl GridSearch {
    pub fn new(specs: Vec<(String, ParamSpec)>) -> Self {
        Self { param_specs: specs }
    }

    pub fn total_combinations(&self) -> usize {
        self.param_specs.iter().map(|(_, s)| s.num_values()).product()
    }

    /// Generate all parameter combinations
    pub fn generate_grid(&self) -> Vec<ParamSet> {
        let values: Vec<Vec<f64>> = self.param_specs.iter().map(|(_, s)| s.values()).collect();
        let names: Vec<&str> = self.param_specs.iter().map(|(n, _)| n.as_str()).collect();
        let total = self.total_combinations();
        let mut grid = Vec::with_capacity(total);

        let mut indices = vec![0usize; self.param_specs.len()];
        for _ in 0..total {
            let mut params = ParamSet::new();
            for (i, &name) in names.iter().enumerate() {
                params.insert(name.to_string(), values[i][indices[i]]);
            }
            grid.push(params);

            // increment indices
            let mut carry = true;
            for i in (0..indices.len()).rev() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= values[i].len() {
                        indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }
        grid
    }

    /// Run grid search with objective function
    pub fn search<F>(&self, objective_fn: &mut F) -> Vec<OptResult>
    where F: FnMut(&ParamSet) -> (f64, HashMap<String, f64>)
    {
        let grid = self.generate_grid();
        let mut results = Vec::with_capacity(grid.len());
        for params in grid {
            let (obj, metrics) = objective_fn(&params);
            results.push(OptResult { params, objective: obj, metrics });
        }
        results.sort_by(|a, b| b.objective.partial_cmp(&a.objective).unwrap());
        results
    }

    pub fn best_result<F>(&self, objective_fn: &mut F) -> OptResult
    where F: FnMut(&ParamSet) -> (f64, HashMap<String, f64>)
    {
        let results = self.search(objective_fn);
        results.into_iter().next().unwrap()
    }
}

/// Random search
pub struct RandomSearch {
    pub param_specs: Vec<(String, ParamSpec)>,
    pub num_samples: usize,
    pub seed: u64,
}

impl RandomSearch {
    pub fn new(specs: Vec<(String, ParamSpec)>, num_samples: usize, seed: u64) -> Self {
        Self { param_specs: specs, num_samples, seed }
    }

    pub fn generate_samples(&self) -> Vec<ParamSet> {
        let mut state = self.seed;
        let mut samples = Vec::with_capacity(self.num_samples);

        for _ in 0..self.num_samples {
            let mut params = ParamSet::new();
            for (name, spec) in &self.param_specs {
                let vals = spec.values();
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (state >> 32) as usize % vals.len();
                params.insert(name.clone(), vals[idx]);
            }
            samples.push(params);
        }
        samples
    }

    pub fn search<F>(&self, objective_fn: &mut F) -> Vec<OptResult>
    where F: FnMut(&ParamSet) -> (f64, HashMap<String, f64>)
    {
        let samples = self.generate_samples();
        let mut results = Vec::with_capacity(samples.len());
        for params in samples {
            let (obj, metrics) = objective_fn(&params);
            results.push(OptResult { params, objective: obj, metrics });
        }
        results.sort_by(|a, b| b.objective.partial_cmp(&a.objective).unwrap());
        results
    }
}

/// Bayesian-inspired optimization (simplified: surrogate model via RBF interpolation)
pub struct BayesianOptimizer {
    pub param_specs: Vec<(String, ParamSpec)>,
    pub n_initial: usize,
    pub n_iterations: usize,
    pub seed: u64,
    pub observed_params: Vec<ParamSet>,
    pub observed_values: Vec<f64>,
}

impl BayesianOptimizer {
    pub fn new(specs: Vec<(String, ParamSpec)>, n_initial: usize, n_iterations: usize, seed: u64) -> Self {
        Self {
            param_specs: specs, n_initial, n_iterations, seed,
            observed_params: Vec::new(), observed_values: Vec::new(),
        }
    }

    fn param_to_vec(params: &ParamSet, names: &[String]) -> Vec<f64> {
        names.iter().map(|n| *params.get(n).unwrap_or(&0.0)).collect()
    }

    fn rbf_kernel(x: &[f64], y: &[f64], length_scale: f64) -> f64 {
        let sq_dist: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
        (-sq_dist / (2.0 * length_scale * length_scale)).exp()
    }

    fn acquisition_ucb(mean: f64, std: f64, beta: f64) -> f64 {
        mean + beta * std
    }

    pub fn search<F>(&mut self, objective_fn: &mut F) -> Vec<OptResult>
    where F: FnMut(&ParamSet) -> (f64, HashMap<String, f64>)
    {
        let names: Vec<String> = self.param_specs.iter().map(|(n, _)| n.clone()).collect();
        let rs = RandomSearch::new(self.param_specs.clone(), self.n_initial, self.seed);
        let initial_samples = rs.generate_samples();

        let mut all_results = Vec::new();

        // Initial evaluations
        for params in initial_samples {
            let (obj, metrics) = objective_fn(&params);
            self.observed_params.push(params.clone());
            self.observed_values.push(obj);
            all_results.push(OptResult { params, objective: obj, metrics });
        }

        // Iterative optimization
        let candidates = RandomSearch::new(self.param_specs.clone(), 100, self.seed + 999);
        for iter in 0..self.n_iterations {
            let candidate_samples = candidates.generate_samples();
            let obs_vecs: Vec<Vec<f64>> = self.observed_params.iter()
                .map(|p| Self::param_to_vec(p, &names)).collect();

            let length_scale = 1.0;
            let beta = 2.0;

            let mut best_acq = f64::NEG_INFINITY;
            let mut best_candidate = candidate_samples[0].clone();

            for cand in &candidate_samples {
                let x = Self::param_to_vec(cand, &names);
                // Compute kernel-weighted mean and variance
                let mut weights = Vec::new();
                let mut w_sum = 0.0;
                for obs in &obs_vecs {
                    let k = Self::rbf_kernel(&x, obs, length_scale);
                    weights.push(k);
                    w_sum += k;
                }
                if w_sum < 1e-15 { continue; }
                let mean: f64 = weights.iter().zip(self.observed_values.iter())
                    .map(|(&w, &v)| w * v).sum::<f64>() / w_sum;
                let var: f64 = weights.iter().zip(self.observed_values.iter())
                    .map(|(&w, &v)| w * (v - mean).powi(2)).sum::<f64>() / w_sum;
                let std = var.sqrt();
                let acq = Self::acquisition_ucb(mean, std, beta);
                if acq > best_acq {
                    best_acq = acq;
                    best_candidate = cand.clone();
                }
            }

            let (obj, metrics) = objective_fn(&best_candidate);
            self.observed_params.push(best_candidate.clone());
            self.observed_values.push(obj);
            all_results.push(OptResult { params: best_candidate, objective: obj, metrics });
        }

        all_results.sort_by(|a, b| b.objective.partial_cmp(&a.objective).unwrap());
        all_results
    }
}

/// Walk-forward optimization
pub struct WalkForwardOptimizer {
    pub param_specs: Vec<(String, ParamSpec)>,
    pub train_window: usize,
    pub test_window: usize,
    pub step_size: usize,
}

impl WalkForwardOptimizer {
    pub fn new(specs: Vec<(String, ParamSpec)>, train: usize, test: usize, step: usize) -> Self {
        Self { param_specs: specs, train_window: train, test_window: test, step_size: step }
    }

    /// Run walk-forward: for each window, optimize on train then evaluate on test
    pub fn run<F, G>(
        &self,
        total_bars: usize,
        train_fn: &mut F,  // (params, start, end) -> objective
        test_fn: &mut G,   // (params, start, end) -> (objective, metrics)
    ) -> Vec<WFResult>
    where
        F: FnMut(&ParamSet, usize, usize) -> f64,
        G: FnMut(&ParamSet, usize, usize) -> (f64, HashMap<String, f64>),
    {
        let mut results = Vec::new();
        let mut start = 0;

        while start + self.train_window + self.test_window <= total_bars {
            let train_end = start + self.train_window;
            let test_end = train_end + self.test_window;

            // Grid search on training period
            let grid = GridSearch::new(self.param_specs.clone());
            let mut best_params = ParamSet::new();
            let mut best_obj = f64::NEG_INFINITY;

            for params in grid.generate_grid() {
                let obj = train_fn(&params, start, train_end);
                if obj > best_obj {
                    best_obj = obj;
                    best_params = params;
                }
            }

            // Evaluate on test period
            let (test_obj, test_metrics) = test_fn(&best_params, train_end, test_end);

            results.push(WFResult {
                train_start: start,
                train_end,
                test_start: train_end,
                test_end,
                best_params: best_params,
                train_objective: best_obj,
                test_objective: test_obj,
                test_metrics,
            });

            start += self.step_size;
        }
        results
    }
}

#[derive(Clone, Debug)]
pub struct WFResult {
    pub train_start: usize,
    pub train_end: usize,
    pub test_start: usize,
    pub test_end: usize,
    pub best_params: ParamSet,
    pub train_objective: f64,
    pub test_objective: f64,
    pub test_metrics: HashMap<String, f64>,
}

/// Walk-forward efficiency: test performance / train performance
pub fn walk_forward_efficiency(results: &[WFResult]) -> f64 {
    if results.is_empty() { return 0.0; }
    let mut num = 0.0;
    let mut den = 0.0;
    for r in results {
        if r.train_objective.abs() > 1e-15 {
            num += r.test_objective;
            den += r.train_objective;
        }
    }
    if den.abs() < 1e-15 { return 0.0; }
    num / den
}

/// Aggregated walk-forward OOS performance
pub fn walk_forward_aggregate(results: &[WFResult]) -> f64 {
    results.iter().map(|r| r.test_objective).sum::<f64>() / results.len().max(1) as f64
}

/// Parameter stability analysis
#[derive(Clone, Debug)]
pub struct ParamStability {
    pub param_name: String,
    pub values: Vec<f64>,
    pub objectives: Vec<f64>,
    pub mean_objective: f64,
    pub std_objective: f64,
    pub sensitivity: f64,
    pub optimal_value: f64,
    pub plateau_width: f64,
}

pub fn analyze_parameter_stability(
    param_name: &str,
    results: &[OptResult],
) -> ParamStability {
    let mut values = Vec::new();
    let mut objectives = Vec::new();

    for r in results {
        if let Some(&v) = r.params.get(param_name) {
            values.push(v);
            objectives.push(r.objective);
        }
    }

    if values.is_empty() {
        return ParamStability {
            param_name: param_name.to_string(),
            values: vec![], objectives: vec![],
            mean_objective: 0.0, std_objective: 0.0,
            sensitivity: 0.0, optimal_value: 0.0, plateau_width: 0.0,
        };
    }

    let mean_obj = objectives.iter().sum::<f64>() / objectives.len() as f64;
    let std_obj = (objectives.iter().map(|&o| (o - mean_obj).powi(2)).sum::<f64>() / objectives.len() as f64).sqrt();

    // Best value
    let (best_idx, _) = objectives.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    let optimal = values[best_idx];

    // Sensitivity: correlation between param value and objective
    let mean_v = values.iter().sum::<f64>() / values.len() as f64;
    let std_v = (values.iter().map(|&v| (v - mean_v).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    let sensitivity = if std_v > 1e-15 && std_obj > 1e-15 {
        let cov: f64 = values.iter().zip(objectives.iter())
            .map(|(&v, &o)| (v - mean_v) * (o - mean_obj)).sum::<f64>() / values.len() as f64;
        (cov / (std_v * std_obj)).abs()
    } else { 0.0 };

    // Plateau: range of values within 5% of best
    let best_obj = objectives[best_idx];
    let threshold = best_obj * 0.95;
    let plateau_values: Vec<f64> = values.iter().zip(objectives.iter())
        .filter(|(_, &o)| o >= threshold)
        .map(|(&v, _)| v)
        .collect();
    let plateau_width = if plateau_values.len() > 1 {
        let min_p = plateau_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_p = plateau_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max_p - min_p
    } else { 0.0 };

    ParamStability {
        param_name: param_name.to_string(),
        values, objectives, mean_objective: mean_obj, std_objective: std_obj,
        sensitivity, optimal_value: optimal, plateau_width,
    }
}

/// Robust parameter selection: pick parameter that maximizes worst-case across windows
pub fn robust_param_selection(wf_results: &[WFResult], param_name: &str) -> f64 {
    let mut by_value: HashMap<i64, Vec<f64>> = HashMap::new(); // discretized value -> objectives
    for r in wf_results {
        if let Some(&v) = r.best_params.get(param_name) {
            let key = (v * 1000.0) as i64;
            by_value.entry(key).or_default().push(r.test_objective);
        }
    }
    // Pick value with best worst-case
    let mut best_value = 0.0;
    let mut best_worst = f64::NEG_INFINITY;
    for (&key, objs) in &by_value {
        let worst = objs.iter().cloned().fold(f64::INFINITY, f64::min);
        if worst > best_worst {
            best_worst = worst;
            best_value = key as f64 / 1000.0;
        }
    }
    best_value
}

/// Overfitting analysis: compare in-sample vs out-of-sample performance distribution
pub fn overfitting_probability(
    is_performance: &[f64],
    oos_performance: &[f64],
) -> f64 {
    // Probability that OOS performance is worse than IS
    let n = is_performance.len().min(oos_performance.len());
    if n == 0 { return 0.5; }
    let worse = (0..n).filter(|&i| oos_performance[i] < is_performance[i]).count();
    worse as f64 / n as f64
}

/// Deflated performance after multiple testing
pub fn deflated_optimal_performance(
    is_performances: &[f64],
    oos_performances: &[f64],
    num_trials: usize,
) -> f64 {
    if oos_performances.is_empty() { return 0.0; }
    let best_oos_idx = oos_performances.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap_or(0);
    let mean_oos = oos_performances.iter().sum::<f64>() / oos_performances.len() as f64;
    let std_oos = (oos_performances.iter().map(|&o| (o - mean_oos).powi(2)).sum::<f64>()
        / oos_performances.len() as f64).sqrt();
    if std_oos < 1e-15 { return mean_oos; }
    // Haircut for multiple testing
    let haircut = (2.0 * (num_trials as f64).ln()).sqrt() * std_oos;
    oos_performances[best_oos_idx] - haircut
}

/// Simulated annealing optimizer
pub struct SimulatedAnnealing {
    pub param_specs: Vec<(String, ParamSpec)>,
    pub initial_temp: f64,
    pub cooling_rate: f64,
    pub max_iter: usize,
    pub seed: u64,
}

impl SimulatedAnnealing {
    pub fn new(specs: Vec<(String, ParamSpec)>, max_iter: usize, seed: u64) -> Self {
        Self {
            param_specs: specs, initial_temp: 100.0, cooling_rate: 0.995,
            max_iter, seed,
        }
    }

    pub fn search<F>(&self, objective_fn: &mut F) -> OptResult
    where F: FnMut(&ParamSet) -> (f64, HashMap<String, f64>)
    {
        let mut state = self.seed;
        let mut rng = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        // Initialize with random
        let all_values: Vec<Vec<f64>> = self.param_specs.iter().map(|(_, s)| s.values()).collect();
        let names: Vec<String> = self.param_specs.iter().map(|(n, _)| n.clone()).collect();

        let mut current = ParamSet::new();
        for (i, name) in names.iter().enumerate() {
            let idx = (rng() * all_values[i].len() as f64) as usize % all_values[i].len();
            current.insert(name.clone(), all_values[i][idx]);
        }

        let (mut current_obj, mut current_metrics) = objective_fn(&current);
        let mut best = current.clone();
        let mut best_obj = current_obj;
        let mut best_metrics = current_metrics.clone();

        let mut temp = self.initial_temp;

        for _ in 0..self.max_iter {
            // Perturb one parameter
            let mut neighbor = current.clone();
            let pi = (rng() * names.len() as f64) as usize % names.len();
            let vals = &all_values[pi];
            let ni = (rng() * vals.len() as f64) as usize % vals.len();
            neighbor.insert(names[pi].clone(), vals[ni]);

            let (new_obj, new_metrics) = objective_fn(&neighbor);
            let delta = new_obj - current_obj;

            if delta > 0.0 || rng() < (delta / temp).exp() {
                current = neighbor;
                current_obj = new_obj;
                current_metrics = new_metrics;
            }

            if current_obj > best_obj {
                best = current.clone();
                best_obj = current_obj;
                best_metrics = current_metrics.clone();
            }

            temp *= self.cooling_rate;
        }

        OptResult { params: best, objective: best_obj, metrics: best_metrics }
    }
}

/// Genetic algorithm optimizer
pub struct GeneticAlgorithm {
    pub param_specs: Vec<(String, ParamSpec)>,
    pub population_size: usize,
    pub num_generations: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub seed: u64,
}

impl GeneticAlgorithm {
    pub fn new(specs: Vec<(String, ParamSpec)>, pop_size: usize, generations: usize, seed: u64) -> Self {
        Self {
            param_specs: specs, population_size: pop_size,
            num_generations: generations, mutation_rate: 0.1,
            crossover_rate: 0.8, seed,
        }
    }

    pub fn search<F>(&self, objective_fn: &mut F) -> Vec<OptResult>
    where F: FnMut(&ParamSet) -> (f64, HashMap<String, f64>)
    {
        let mut state = self.seed;
        let mut rng = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        let all_values: Vec<Vec<f64>> = self.param_specs.iter().map(|(_, s)| s.values()).collect();
        let names: Vec<String> = self.param_specs.iter().map(|(n, _)| n.clone()).collect();
        let n_params = names.len();

        // Initialize population
        let mut population: Vec<ParamSet> = (0..self.population_size).map(|_| {
            let mut p = ParamSet::new();
            for (i, name) in names.iter().enumerate() {
                let idx = (rng() * all_values[i].len() as f64) as usize % all_values[i].len();
                p.insert(name.clone(), all_values[i][idx]);
            }
            p
        }).collect();

        let mut all_results = Vec::new();

        for _gen in 0..self.num_generations {
            // Evaluate
            let mut scored: Vec<(ParamSet, f64, HashMap<String, f64>)> = population.iter().map(|p| {
                let (obj, metrics) = objective_fn(p);
                (p.clone(), obj, metrics)
            }).collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (p, obj, m) in &scored {
                all_results.push(OptResult { params: p.clone(), objective: *obj, metrics: m.clone() });
            }

            // Selection: top 50%
            let elite_count = self.population_size / 2;
            let elite: Vec<ParamSet> = scored[..elite_count].iter().map(|(p, _, _)| p.clone()).collect();

            // New population
            let mut new_pop = elite.clone();
            while new_pop.len() < self.population_size {
                let p1_idx = (rng() * elite.len() as f64) as usize % elite.len();
                let p2_idx = (rng() * elite.len() as f64) as usize % elite.len();

                let mut child = ParamSet::new();
                for (i, name) in names.iter().enumerate() {
                    // Crossover
                    let val = if rng() < self.crossover_rate {
                        if rng() < 0.5 { *elite[p1_idx].get(name).unwrap() } else { *elite[p2_idx].get(name).unwrap() }
                    } else {
                        *elite[p1_idx].get(name).unwrap()
                    };
                    // Mutation
                    let val = if rng() < self.mutation_rate {
                        let idx = (rng() * all_values[i].len() as f64) as usize % all_values[i].len();
                        all_values[i][idx]
                    } else {
                        val
                    };
                    child.insert(name.clone(), val);
                }
                new_pop.push(child);
            }
            population = new_pop;
        }

        all_results.sort_by(|a, b| b.objective.partial_cmp(&a.objective).unwrap());
        all_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_search() {
        let specs = vec![
            ("a".to_string(), ParamSpec::IntRange { min: 1, max: 3, step: 1 }),
            ("b".to_string(), ParamSpec::FloatRange { min: 0.1, max: 0.3, step: 0.1 }),
        ];
        let gs = GridSearch::new(specs);
        assert_eq!(gs.total_combinations(), 9);
        let grid = gs.generate_grid();
        assert_eq!(grid.len(), 9);
    }

    #[test]
    fn test_grid_search_run() {
        let specs = vec![
            ("x".to_string(), ParamSpec::FloatRange { min: -1.0, max: 1.0, step: 0.5 }),
        ];
        let gs = GridSearch::new(specs);
        let results = gs.search(&mut |params| {
            let x = params["x"];
            let obj = -(x * x); // maximize at x=0
            (obj, HashMap::new())
        });
        assert!((results[0].params["x"]).abs() < 1e-10);
    }

    #[test]
    fn test_random_search() {
        let specs = vec![
            ("x".to_string(), ParamSpec::FloatRange { min: -1.0, max: 1.0, step: 0.1 }),
        ];
        let rs = RandomSearch::new(specs, 20, 42);
        let samples = rs.generate_samples();
        assert_eq!(samples.len(), 20);
    }

    #[test]
    fn test_simulated_annealing() {
        let specs = vec![
            ("x".to_string(), ParamSpec::FloatRange { min: -5.0, max: 5.0, step: 0.1 }),
        ];
        let sa = SimulatedAnnealing::new(specs, 1000, 42);
        let result = sa.search(&mut |params| {
            let x = params["x"];
            (-(x * x), HashMap::new())
        });
        assert!(result.params["x"].abs() < 2.0);
    }

    #[test]
    fn test_param_stability() {
        let results: Vec<OptResult> = (0..10).map(|i| {
            let mut params = ParamSet::new();
            params.insert("x".to_string(), i as f64);
            OptResult { params, objective: -(i as f64 - 5.0).powi(2), metrics: HashMap::new() }
        }).collect();
        let stability = analyze_parameter_stability("x", &results);
        assert!((stability.optimal_value - 5.0).abs() < 1e-10);
        assert!(stability.plateau_width >= 0.0);
    }

    #[test]
    fn test_overfitting_prob() {
        let is = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let oos = vec![0.5, 1.5, 2.0, 3.0, 4.0];
        let prob = overfitting_probability(&is, &oos);
        assert!(prob > 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_genetic_algorithm() {
        let specs = vec![
            ("x".to_string(), ParamSpec::FloatRange { min: -5.0, max: 5.0, step: 0.5 }),
        ];
        let ga = GeneticAlgorithm::new(specs, 10, 5, 42);
        let results = ga.search(&mut |params| {
            let x = params["x"];
            (-(x * x), HashMap::new())
        });
        assert!(!results.is_empty());
        assert!(results[0].params["x"].abs() < 3.0);
    }

    #[test]
    fn test_log_range() {
        let spec = ParamSpec::LogRange { min: 0.001, max: 1.0, num_points: 5 };
        let vals = spec.values();
        assert_eq!(vals.len(), 5);
        assert!(vals[0] > 0.0 && vals[0] < vals[4]);
    }
}
