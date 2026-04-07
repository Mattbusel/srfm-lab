// alpha_simulator.rs
// Simulate alpha decay scenarios.
// Given half-life and initial IC, simulate future IC paths via mean-reverting (OU) process.
// Compute expected return contribution vs holding cost.

use serde::{Deserialize, Serialize};
use rand::prelude::*;

/// Parameters for a decay scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayScenario {
    /// Initial IC (e.g., 0.08).
    pub initial_ic: f64,
    /// Long-run mean IC (the equilibrium level the signal reverts to).
    pub long_run_ic: f64,
    /// Half-life in bars (time for IC to decay halfway to the long-run mean).
    pub half_life_bars: f64,
    /// Volatility of IC innovations.
    pub ic_vol: f64,
    /// Daily signal volatility (for return scaling).
    pub signal_vol: f64,
    /// Per-bar holding cost (e.g., financing, slippage amortized).
    pub holding_cost_per_bar: f64,
    /// Number of simulation bars.
    pub n_bars: usize,
}

impl DecayScenario {
    /// Mean-reversion speed from half-life.
    pub fn kappa(&self) -> f64 {
        std::f64::consts::LN_2 / self.half_life_bars.max(1.0)
    }

    /// Expected IC at horizon h bars from now.
    pub fn expected_ic_at(&self, h: f64) -> f64 {
        let kappa = self.kappa();
        self.long_run_ic + (self.initial_ic - self.long_run_ic) * (-kappa * h).exp()
    }

    /// Expected cumulative return contribution up to horizon h.
    /// Returns = IC * signal_vol * sqrt(2/pi) (expected absolute value of return).
    pub fn expected_cum_return(&self, h: usize) -> f64 {
        let scale = self.signal_vol * (2.0 / std::f64::consts::PI).sqrt();
        (1..=h)
            .map(|bar| self.expected_ic_at(bar as f64) * scale)
            .sum::<f64>()
    }

    /// Net expected value: cumulative return minus holding cost.
    pub fn net_expected_value(&self, h: usize) -> f64 {
        self.expected_cum_return(h) - self.holding_cost_per_bar * h as f64
    }

    /// Optimal holding horizon: maximize net expected value.
    pub fn optimal_holding_horizon(&self) -> usize {
        let max_h = self.n_bars.min(500);
        let mut best_h = 1;
        let mut best_nev = self.net_expected_value(1);
        for h in 2..=max_h {
            let nev = self.net_expected_value(h);
            if nev > best_nev {
                best_nev = nev;
                best_h = h;
            }
        }
        best_h
    }
}

/// Output from a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub scenario: DecayScenario,
    /// Simulated IC path for each path.
    pub ic_paths: Vec<Vec<f64>>,
    /// Cumulative return paths.
    pub cum_return_paths: Vec<Vec<f64>>,
    /// Net cumulative return (after holding cost) paths.
    pub net_cum_return_paths: Vec<Vec<f64>>,
    /// Mean IC at each bar across paths.
    pub mean_ic_path: Vec<f64>,
    /// 5th percentile IC at each bar.
    pub ic_p05: Vec<f64>,
    /// 95th percentile IC at each bar.
    pub ic_p95: Vec<f64>,
    /// Mean cumulative return at each bar.
    pub mean_cum_return: Vec<f64>,
    /// Expected net value at each bar.
    pub expected_net_value: Vec<f64>,
    /// Optimal holding horizon (bar).
    pub optimal_holding_horizon: usize,
    /// Probability of positive net value at optimal horizon.
    pub prob_positive_at_optimal: f64,
    /// Break-even IC (IC needed to cover holding cost).
    pub break_even_ic: f64,
}

/// Alpha scenario simulator using Ornstein-Uhlenbeck process for IC dynamics.
pub struct AlphaSimulator {
    seed: u64,
}

impl AlphaSimulator {
    pub fn new(seed: u64) -> Self {
        AlphaSimulator { seed }
    }

    /// Run simulation for a given scenario, generating `n_paths` IC trajectories.
    pub fn simulate(&self, scenario: &DecayScenario, n_paths: usize) -> SimulationResult {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let kappa = scenario.kappa();
        let n = scenario.n_bars;
        let dt = 1.0; // 1 bar per step.

        // Euler-Maruyama for OU process:
        // dIC = kappa * (mu - IC) * dt + sigma * sqrt(dt) * dW
        let sigma = scenario.ic_vol;
        let mu = scenario.long_run_ic;

        let mut ic_paths: Vec<Vec<f64>> = Vec::with_capacity(n_paths);
        let mut cum_return_paths: Vec<Vec<f64>> = Vec::with_capacity(n_paths);
        let mut net_cum_return_paths: Vec<Vec<f64>> = Vec::with_capacity(n_paths);

        let return_scale = scenario.signal_vol * (2.0 / std::f64::consts::PI).sqrt();

        for _ in 0..n_paths {
            let mut ic = scenario.initial_ic;
            let mut cum_return = 0.0f64;
            let mut ic_path = Vec::with_capacity(n);
            let mut cum_ret_path = Vec::with_capacity(n);
            let mut net_cum_ret_path = Vec::with_capacity(n);

            for bar in 0..n {
                // OU update.
                let innovation: f64 = rng.gen::<f64>() * 2.0 - 1.0; // Use uniform approx for speed.
                // Box-Muller for proper normal.
                let normal = Self::normal_sample(&mut rng);
                ic += kappa * (mu - ic) * dt + sigma * dt.sqrt() * normal;

                // Return for this bar: IC * signal_vol * sign(signal), expected value = IC * scale.
                let bar_return = ic * return_scale - scenario.holding_cost_per_bar;
                let _ = innovation; // suppress warning
                cum_return += bar_return;

                let net_cum = cum_return; // holding cost already included per bar.
                ic_path.push(ic);
                cum_ret_path.push(ic * return_scale * (bar + 1) as f64); // gross, ignoring compounding
                net_cum_ret_path.push(net_cum);
            }
            ic_paths.push(ic_path);
            cum_return_paths.push(cum_ret_path);
            net_cum_return_paths.push(net_cum_ret_path);
        }

        // Compute cross-path statistics.
        let mean_ic_path = Self::cross_path_mean(&ic_paths, n);
        let ic_p05 = Self::cross_path_percentile(&ic_paths, n, 0.05);
        let ic_p95 = Self::cross_path_percentile(&ic_paths, n, 0.95);
        let mean_cum_return = Self::cross_path_mean(&net_cum_return_paths, n);

        let expected_net_value: Vec<f64> = (0..n)
            .map(|h| scenario.net_expected_value(h + 1))
            .collect();

        let optimal_h = scenario.optimal_holding_horizon();

        let prob_positive_at_optimal = if optimal_h > 0 && optimal_h <= n {
            let idx = optimal_h - 1;
            let pos: f64 = net_cum_return_paths
                .iter()
                .filter(|path| path.get(idx).map(|&v| v > 0.0).unwrap_or(false))
                .count() as f64;
            pos / n_paths as f64
        } else {
            0.0
        };

        let break_even_ic = scenario.holding_cost_per_bar / return_scale.max(1e-10);

        SimulationResult {
            scenario: scenario.clone(),
            ic_paths,
            cum_return_paths,
            net_cum_return_paths,
            mean_ic_path,
            ic_p05,
            ic_p95,
            mean_cum_return,
            expected_net_value,
            optimal_holding_horizon: optimal_h,
            prob_positive_at_optimal,
            break_even_ic,
        }
    }

    /// Run simulations for multiple scenarios.
    pub fn simulate_multiple(
        &self,
        scenarios: &[DecayScenario],
        n_paths: usize,
    ) -> Vec<SimulationResult> {
        scenarios
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let sim = AlphaSimulator::new(self.seed.wrapping_add(i as u64 * 1000));
                sim.simulate(s, n_paths)
            })
            .collect()
    }

    /// Compare scenarios: return (scenario_index, optimal_horizon, expected_value) sorted by EV.
    pub fn compare_scenarios(
        &self,
        scenarios: &[DecayScenario],
        n_paths: usize,
    ) -> Vec<(usize, usize, f64)> {
        let results = self.simulate_multiple(scenarios, n_paths);
        let mut summary: Vec<(usize, usize, f64)> = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let opt_h = r.optimal_holding_horizon;
                let ev = r
                    .expected_net_value
                    .get(opt_h.saturating_sub(1))
                    .copied()
                    .unwrap_or(0.0);
                (i, opt_h, ev)
            })
            .collect();
        summary.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        summary
    }

    /// Sample standard normal using Box-Muller transform.
    fn normal_sample(rng: &mut StdRng) -> f64 {
        let u1: f64 = rng.gen::<f64>().max(1e-14);
        let u2: f64 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn cross_path_mean(paths: &[Vec<f64>], n: usize) -> Vec<f64> {
        let n_paths = paths.len() as f64;
        (0..n)
            .map(|t| paths.iter().map(|p| p.get(t).copied().unwrap_or(0.0)).sum::<f64>() / n_paths)
            .collect()
    }

    fn cross_path_percentile(paths: &[Vec<f64>], n: usize, p: f64) -> Vec<f64> {
        (0..n)
            .map(|t| {
                let mut vals: Vec<f64> = paths
                    .iter()
                    .map(|path| path.get(t).copied().unwrap_or(0.0))
                    .collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = (p * vals.len() as f64) as usize;
                vals[idx.min(vals.len() - 1)]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scenario() -> DecayScenario {
        DecayScenario {
            initial_ic: 0.08,
            long_run_ic: 0.03,
            half_life_bars: 20.0,
            ic_vol: 0.02,
            signal_vol: 0.01,
            holding_cost_per_bar: 0.0001,
            n_bars: 60,
        }
    }

    #[test]
    fn test_expected_ic_decay() {
        let scenario = make_scenario();
        let ic_0 = scenario.initial_ic;
        let ic_hl = scenario.expected_ic_at(scenario.half_life_bars);
        let mid = (ic_0 + scenario.long_run_ic) / 2.0;
        assert!(
            (ic_hl - mid).abs() < 0.001,
            "IC at half-life should be midpoint: got {}, expected {}",
            ic_hl,
            mid
        );
    }

    #[test]
    fn test_simulation_produces_paths() {
        let scenario = make_scenario();
        let sim = AlphaSimulator::new(42);
        let result = sim.simulate(&scenario, 100);
        assert_eq!(result.ic_paths.len(), 100);
        assert_eq!(result.ic_paths[0].len(), 60);
    }

    #[test]
    fn test_mean_ic_converges() {
        let scenario = make_scenario();
        let sim = AlphaSimulator::new(99);
        let result = sim.simulate(&scenario, 1000);
        // Mean IC at bar 0 should be close to initial_ic.
        let mean_ic_0 = result.mean_ic_path[0];
        assert!(
            (mean_ic_0 - scenario.initial_ic).abs() < 0.02,
            "Initial mean IC off: {}",
            mean_ic_0
        );
    }

    #[test]
    fn test_optimal_horizon_is_positive() {
        let scenario = make_scenario();
        assert!(scenario.optimal_holding_horizon() > 0);
    }

    #[test]
    fn test_break_even_ic() {
        let scenario = make_scenario();
        let sim = AlphaSimulator::new(1);
        let result = sim.simulate(&scenario, 50);
        assert!(result.break_even_ic > 0.0);
    }
}
