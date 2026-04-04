/// Market impact models for estimating execution cost.

/// Available impact model types.
#[derive(Debug, Clone, Copy)]
pub enum ImpactModel {
    SquareRoot,
    Linear,
    AlmgrenChriss,
}

/// Parameters for the square-root impact model.
#[derive(Debug, Clone, Copy)]
pub struct SquareRootParams {
    /// Permanent impact coefficient γ.
    pub gamma: f64,
    /// Temporary impact coefficient η.
    pub eta: f64,
}

impl Default for SquareRootParams {
    fn default() -> Self {
        SquareRootParams { gamma: 0.1, eta: 0.1 }
    }
}

/// Parameters for a linear impact model.
#[derive(Debug, Clone, Copy)]
pub struct LinearParams {
    /// Permanent impact per unit of participation rate.
    pub permanent: f64,
    /// Temporary impact per unit of participation rate.
    pub temporary: f64,
}

impl Default for LinearParams {
    fn default() -> Self {
        LinearParams { permanent: 0.05, temporary: 0.05 }
    }
}

/// Almgren-Chriss model parameters.
#[derive(Debug, Clone, Copy)]
pub struct AlmgrenChrissParams {
    /// Permanent impact coefficient (linear).
    pub permanent_coeff: f64,
    /// Temporary impact coefficient.
    pub temporary_coeff: f64,
    /// Asset daily volatility.
    pub sigma: f64,
    /// Risk aversion (λ). Higher = more aggressive schedule.
    pub risk_aversion: f64,
    /// Current mid price.
    pub mid_price: f64,
}

impl Default for AlmgrenChrissParams {
    fn default() -> Self {
        AlmgrenChrissParams {
            permanent_coeff: 2.5e-7,
            temporary_coeff: 2.5e-6,
            sigma: 0.02,
            risk_aversion: 1e-6,
            mid_price: 100.0,
        }
    }
}

/// Estimate the permanent market impact using a square-root model.
///
/// `permanent_impact = gamma * sigma * sqrt(Q / ADV)`
pub fn sqrt_permanent_impact(qty: f64, adv: f64, sigma: f64, gamma: f64) -> f64 {
    gamma * sigma * (qty / adv).sqrt()
}

/// Estimate the temporary market impact using a square-root model.
///
/// `temporary_impact = eta * sigma * (Q / ADV)^0.6`
pub fn sqrt_temporary_impact(qty: f64, adv: f64, sigma: f64, eta: f64) -> f64 {
    eta * sigma * (qty / adv).powf(0.6)
}

/// Almgren-Chriss optimal execution trajectory.
///
/// Minimises E[cost] + λ * Var[cost] over N trading intervals.
/// Returns a vector of (slice_qty, expected_price_impact) tuples,
/// where `slice_qty[i]` is the number of shares to trade in interval i.
pub struct AlmgrenChrissSchedule {
    pub slices: Vec<f64>,
    pub expected_cost: f64,
    pub cost_variance: f64,
}

impl AlmgrenChrissSchedule {
    /// Compute the optimal execution schedule.
    ///
    /// # Arguments
    /// * `total_qty` — shares to liquidate / acquire.
    /// * `n_intervals` — number of trading intervals.
    /// * `params` — model parameters.
    pub fn compute(total_qty: f64, n_intervals: usize, params: &AlmgrenChrissParams) -> Self {
        let n = n_intervals as f64;
        // Almgren-Chriss closed-form solution.
        // τ = T / N  (length of each interval, normalised to 1 here)
        let tau = 1.0; // each interval is 1 time unit
        let _t = n * tau;

        let gamma = params.permanent_coeff;
        let eta = params.temporary_coeff;
        let sigma = params.sigma;
        let lambda = params.risk_aversion;

        // κ² = λσ² / η  (decay rate of optimal strategy)
        let kappa_sq = lambda * sigma * sigma / eta;
        let kappa = kappa_sq.sqrt();

        // sinh(κτ) helper
        let sinh_kt = (kappa * tau).sinh();
        let sinh_kn_tau = (kappa * n * tau).sinh();

        // Precompute optimal holdings x_j = X * sinh(κ(N-j)τ) / sinh(κNτ)
        let mut slices = Vec::with_capacity(n_intervals);
        let mut prev_holding = total_qty;
        for j in 0..n_intervals {
            let j_f = j as f64;
            let holding = total_qty * (kappa * (n - j_f) * tau).sinh() / sinh_kn_tau;
            slices.push(prev_holding - holding);
            prev_holding = holding;
        }
        // Ensure rounding residual is captured.
        let traded: f64 = slices.iter().sum();
        if let Some(last) = slices.last_mut() {
            *last += total_qty - traded;
        }

        // Expected cost (simplified).
        let expected_cost = 0.5 * gamma * total_qty * total_qty
            + eta * total_qty.powi(2) * tau * kappa / sinh_kt;
        // Variance of cost.
        let cost_variance = 0.5 * lambda * sigma * sigma * total_qty.powi(2)
            * tau * (kappa * n * tau).cosh() / sinh_kn_tau.powi(2) * sinh_kt;

        AlmgrenChrissSchedule {
            slices,
            expected_cost,
            cost_variance: cost_variance.abs(),
        }
    }
}

/// High-level slippage estimator.
///
/// Returns expected slippage (fraction of price) for executing `qty`
/// shares when ADV = `adv` and daily volatility = `volatility`.
pub fn estimate_slippage(qty: f64, adv: f64, volatility: f64, model: ImpactModel) -> f64 {
    match model {
        ImpactModel::SquareRoot => {
            let params = SquareRootParams::default();
            sqrt_temporary_impact(qty, adv, volatility, params.eta)
        }
        ImpactModel::Linear => {
            let params = LinearParams::default();
            params.temporary * (qty / adv)
        }
        ImpactModel::AlmgrenChriss => {
            let params = AlmgrenChrissParams {
                sigma: volatility,
                ..AlmgrenChrissParams::default()
            };
            // Use a single-interval schedule as the instantaneous cost estimate.
            let schedule = AlmgrenChrissSchedule::compute(qty, 1, &params);
            schedule.expected_cost / (qty * params.mid_price)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_impact_positive() {
        let imp = sqrt_temporary_impact(1000.0, 1_000_000.0, 0.02, 0.1);
        assert!(imp > 0.0 && imp < 0.01, "impact={imp}");
    }

    #[test]
    fn ac_schedule_sums_to_total() {
        let params = AlmgrenChrissParams::default();
        let schedule = AlmgrenChrissSchedule::compute(10_000.0, 10, &params);
        let total: f64 = schedule.slices.iter().sum();
        assert!((total - 10_000.0).abs() < 1e-6, "total={total}");
    }
}
