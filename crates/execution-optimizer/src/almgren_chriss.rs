/// Almgren-Chriss (2001) optimal execution model.
///
/// Minimises: E[cost] + λ · Var[cost]
///
/// Where cost = market-impact cost + timing risk (price volatility × position).
///
/// The optimal strategy is to liquidate X shares over T periods with
/// a trajectory that trades off impact cost vs. timing risk.
///
/// Reference: Almgren, R. & Chriss, N. (2001).
///            "Optimal Execution of Portfolio Transactions".
///            Journal of Risk, 3(2), 5–39.

use serde::Serialize;

/// Model parameters for Almgren-Chriss.
#[derive(Debug, Clone, Serialize)]
pub struct AcParams {
    /// Total shares to execute (positive = sell, negative = buy)
    pub x0:     f64,
    /// Number of execution intervals
    pub n:      usize,
    /// Time step size (e.g., seconds or fraction of day)
    pub tau:    f64,
    /// Annual volatility of asset (σ). Adjust to per-step: σ_step = σ * √τ
    pub sigma:  f64,
    /// Temporary impact coefficient η (linear in trade rate)
    pub eta:    f64,
    /// Permanent impact coefficient γ (linear in cumulative trade)
    pub gamma:  f64,
    /// Risk-aversion parameter λ (controls trade-off between cost and variance)
    pub lambda: f64,
}

impl AcParams {
    /// Convenience: create params for a typical equity liquidation.
    /// `shares` total to sell, `n_intervals` time buckets, `sigma_daily` daily vol.
    pub fn liquidation(
        shares:       f64,
        n_intervals:  usize,
        t_days:       f64,
        sigma_daily:  f64,
        eta:          f64,
        gamma:        f64,
        lambda:       f64,
    ) -> Self {
        let tau = t_days / n_intervals as f64;
        Self { x0: shares, n: n_intervals, tau, sigma: sigma_daily, eta, gamma, lambda }
    }
}

/// Optimal execution schedule derived from AC model.
#[derive(Debug, Clone, Serialize)]
pub struct AcSchedule {
    /// Remaining inventory at start of each interval: x_j (j=0..n)
    pub inventory:  Vec<f64>,
    /// Trade size at each interval: n_j = x_{j-1} - x_j
    pub trades:     Vec<f64>,
    /// Expected cost of the schedule
    pub expected_cost: f64,
    /// Variance of the cost
    pub cost_variance: f64,
    /// Efficient frontier utility: E[cost] + λ·Var[cost]
    pub utility:    f64,
    /// Params used
    pub params:     AcParams,
}

impl AcSchedule {
    /// Compute the optimal trajectory.
    pub fn compute(p: AcParams) -> Self {
        let n      = p.n;
        let tau    = p.tau;
        let sigma  = p.sigma;
        let eta    = p.eta;
        let gamma  = p.gamma;
        let lambda = p.lambda;
        let x0     = p.x0;

        // Per-step volatility
        let sigma_tau = sigma * tau.sqrt();

        // AC formula: κ² = λσ²/η̃   where  η̃ = η - γτ/2
        let eta_tilde = eta - gamma * tau / 2.0;
        let kappa_sq  = if eta_tilde.abs() > 1e-15 {
            (lambda * sigma_tau * sigma_tau) / eta_tilde
        } else { 1e-8 };

        let kappa = kappa_sq.max(0.0).sqrt();

        // Optimal trajectory: x_j = x0 · sinh(κ(T-t_j)) / sinh(κT)
        let t_total = n as f64 * tau;
        let sinh_kt = (kappa * t_total).sinh();

        let mut inventory = Vec::with_capacity(n + 1);
        for j in 0..=n {
            let t_j = j as f64 * tau;
            let remaining_t = t_total - t_j;
            let x_j = if sinh_kt.abs() > 1e-15 {
                x0 * (kappa * remaining_t).sinh() / sinh_kt
            } else {
                x0 * (1.0 - j as f64 / n as f64)   // linear fallback
            };
            inventory.push(x_j);
        }

        let trades: Vec<f64> = inventory.windows(2)
            .map(|w| w[0] - w[1])
            .collect();

        // Expected cost
        let expected_cost = gamma / 2.0 * x0 * x0
            + eta * (inventory.iter().skip(1).zip(trades.iter())
                       .map(|(_, n_j)| n_j * n_j)
                       .sum::<f64>() / tau);

        // Cost variance
        let cost_variance = sigma_tau * sigma_tau
            * inventory[1..].iter().map(|x| x * x).sum::<f64>();

        let utility = expected_cost + lambda * cost_variance;

        AcSchedule { inventory, trades, expected_cost, cost_variance, utility, params: p }
    }

    /// VWAP-like schedule: uniform liquidation (benchmark).
    pub fn twap(p: AcParams) -> Self {
        let n     = p.n;
        let x0    = p.x0;
        let trade = x0 / n as f64;

        let inventory: Vec<f64> = (0..=n).map(|j| x0 - j as f64 * trade).collect();
        let trades: Vec<f64>    = vec![trade; n];

        // Costs under TWAP (for comparison)
        let tau        = p.tau;
        let sigma      = p.sigma;
        let sigma_tau  = sigma * tau.sqrt();
        let expected_cost = p.gamma / 2.0 * x0 * x0
            + p.eta * trades.iter().map(|&t| t * t / tau).sum::<f64>();
        let cost_variance = sigma_tau * sigma_tau
            * inventory[1..].iter().map(|x| x * x).sum::<f64>();
        let utility = expected_cost + p.lambda * cost_variance;

        AcSchedule { inventory, trades, expected_cost, cost_variance, utility, params: p }
    }
}

/// Almgren-Chriss efficient frontier — sweep over λ values.
pub struct AlmgrenChriss;

impl AlmgrenChriss {
    /// Compute the efficient frontier by sweeping λ ∈ [λ_min, λ_max].
    /// Returns (expected_cost, std_cost) pairs.
    pub fn efficient_frontier(
        base_params:  &AcParams,
        lambdas:      &[f64],
    ) -> Vec<(f64, f64, AcSchedule)> {
        lambdas.iter().map(|&lam| {
            let mut p = base_params.clone();
            p.lambda  = lam;
            let sched = AcSchedule::compute(p);
            (sched.expected_cost, sched.cost_variance.sqrt(), sched)
        }).collect()
    }

    /// Optimal λ given a maximum acceptable variance budget.
    pub fn solve_for_variance_budget(
        base_params:   &AcParams,
        max_variance:  f64,
    ) -> Option<AcSchedule> {
        // Binary search on λ to hit variance budget
        let mut lo = 1e-10_f64;
        let mut hi = 1e6_f64;
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            let mut p = base_params.clone();
            p.lambda  = mid;
            let s = AcSchedule::compute(p);
            if s.cost_variance > max_variance { lo = mid; } else { hi = mid; }
        }
        let mut p = base_params.clone();
        p.lambda = (lo + hi) / 2.0;
        Some(AcSchedule::compute(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ac_schedule_sum_to_x0() {
        let params = AcParams::liquidation(10_000.0, 20, 1.0, 0.02, 0.1, 0.01, 1e-6);
        let sched  = AcSchedule::compute(params);
        let total: f64 = sched.trades.iter().sum();
        assert!((total - 10_000.0).abs() < 0.1, "trades sum {} ≠ 10000", total);
    }

    #[test]
    fn ac_schedule_monotone_liquidation() {
        let params = AcParams::liquidation(10_000.0, 10, 1.0, 0.02, 0.1, 0.01, 1e-6);
        let sched  = AcSchedule::compute(params);
        // All trades should be positive (selling)
        for (i, &t) in sched.trades.iter().enumerate() {
            assert!(t > 0.0, "trade[{}] = {} not positive", i, t);
        }
    }

    #[test]
    fn twap_uniform_trades() {
        let params = AcParams::liquidation(1_000.0, 10, 1.0, 0.02, 0.1, 0.01, 1e-6);
        let sched  = AcSchedule::twap(params);
        for &t in &sched.trades {
            assert!((t - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn ac_vs_twap_lower_utility() {
        // Optimal AC schedule should have lower utility than TWAP for same λ
        let params = AcParams::liquidation(10_000.0, 20, 1.0, 0.02, 0.1, 0.01, 1e-4);
        let ac   = AcSchedule::compute(params.clone());
        let twap = AcSchedule::twap(params);
        assert!(ac.utility <= twap.utility + 1e-6,
            "AC utility {} > TWAP utility {}", ac.utility, twap.utility);
    }

    #[test]
    fn efficient_frontier_monotone() {
        let params  = AcParams::liquidation(5_000.0, 10, 0.5, 0.02, 0.05, 0.005, 1e-5);
        let lambdas: Vec<f64> = (1..=10).map(|i| i as f64 * 1e-5).collect();
        let frontier = AlmgrenChriss::efficient_frontier(&params, &lambdas);
        // Higher λ → lower variance, higher expected cost
        let variances: Vec<f64> = frontier.iter().map(|(_, s, _)| *s).collect();
        for i in 1..variances.len() {
            assert!(variances[i] <= variances[i-1] + 1e-6,
                "variance not monotone decreasing at i={}", i);
        }
    }
}
