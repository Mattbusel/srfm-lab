// systemic.rs — CoVaR, MES, SRISK, DebtRank, Eisenberg-Noe, fire sale, contagion

use quant_math::statistics;
use quant_math::distributions::{norm_ppf, norm_cdf, Xoshiro256PlusPlus};

/// CoVaR: VaR of system conditional on institution i being in distress
/// (Adrian-Brunnermeier)
pub fn covar(
    system_returns: &[f64], institution_returns: &[f64],
    confidence: f64,
) -> (f64, f64) {
    assert_eq!(system_returns.len(), institution_returns.len());
    let n = system_returns.len();

    // Institution VaR
    let mut sorted_inst: Vec<(f64, usize)> = institution_returns.iter()
        .copied().enumerate().map(|(i, r)| (r, i)).collect();
    sorted_inst.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = ((1.0 - confidence) * n as f64).floor() as usize;
    let cutoff = cutoff.max(1);

    // System returns when institution is at/below its VaR
    let distress_indices: Vec<usize> = sorted_inst[..cutoff].iter().map(|&(_, i)| i).collect();
    let conditional_returns: Vec<f64> = distress_indices.iter().map(|&i| system_returns[i]).collect();

    // CoVaR = VaR of system given institution distress
    let mut sorted_cond = conditional_returns.clone();
    sorted_cond.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cond_cutoff = ((1.0 - confidence) * sorted_cond.len() as f64).floor() as usize;
    let cond_cutoff = cond_cutoff.max(0).min(sorted_cond.len() - 1);
    let covar_val = -sorted_cond[cond_cutoff];

    // ΔCoVaR = CoVaR - VaR of system at median of institution
    let median_indices: Vec<usize> = sorted_inst[cutoff..].iter()
        .take(cutoff.max(1))
        .map(|&(_, i)| i)
        .collect();
    let median_returns: Vec<f64> = if !median_indices.is_empty() {
        median_indices.iter().map(|&i| system_returns[i]).collect()
    } else {
        system_returns.to_vec()
    };
    let mut sorted_med = median_returns;
    sorted_med.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_cutoff = ((1.0 - confidence) * sorted_med.len() as f64).floor() as usize;
    let var_median = -sorted_med[med_cutoff.min(sorted_med.len() - 1)];

    let delta_covar = covar_val - var_median;
    (covar_val, delta_covar)
}

/// Quantile regression CoVaR (simplified linear)
pub fn covar_quantile_regression(
    system_returns: &[f64], institution_returns: &[f64],
    confidence: f64,
) -> (f64, f64, f64) {
    // Linear quantile regression: Q_τ(R_sys | R_inst) = α + β * R_inst
    let tau = 1.0 - confidence;
    let n = system_returns.len();

    // Simple OLS as approximation (proper quantile regression is iterative)
    let mean_x = statistics::mean(institution_returns);
    let mean_y = statistics::mean(system_returns);
    let cov_xy = statistics::covariance(system_returns, institution_returns);
    let var_x = statistics::variance(institution_returns);

    let beta = if var_x > 1e-15 { cov_xy / var_x } else { 0.0 };
    let alpha = mean_y - beta * mean_x;

    // Adjust for quantile
    let residuals: Vec<f64> = (0..n).map(|i| {
        system_returns[i] - (alpha + beta * institution_returns[i])
    }).collect();
    let resid_quantile = {
        let mut sorted = residuals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (tau * n as f64).floor() as usize;
        sorted[idx.min(n - 1)]
    };

    let alpha_q = alpha + resid_quantile;

    // Institution VaR
    let inst_var = {
        let mut sorted = institution_returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (tau * n as f64).floor() as usize;
        sorted[idx.min(n - 1)]
    };

    let covar_val = -(alpha_q + beta * inst_var);
    let covar_median = -(alpha_q + beta * statistics::median(institution_returns));
    let delta_covar = covar_val - covar_median;

    (covar_val, delta_covar, beta)
}

/// Marginal Expected Shortfall (MES)
/// Expected loss of institution i when system is in its worst α% days
pub fn mes(
    system_returns: &[f64], institution_returns: &[f64],
    alpha: f64,
) -> f64 {
    assert_eq!(system_returns.len(), institution_returns.len());
    let n = system_returns.len();

    // Find worst α% days for system
    let mut indexed: Vec<(f64, usize)> = system_returns.iter()
        .copied().enumerate().map(|(i, r)| (r, i)).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = (alpha * n as f64).ceil() as usize;
    let cutoff = cutoff.max(1);

    // Average institution return on system's worst days
    let worst_inst: Vec<f64> = indexed[..cutoff].iter()
        .map(|&(_, i)| institution_returns[i])
        .collect();
    -statistics::mean(&worst_inst)
}

/// Long-run MES (Brownlees-Engle)
pub fn lrmes(mes_1day: f64, horizon_days: usize) -> f64 {
    // LRMES ≈ 1 - exp(-18 * MES_1day) for standard 6-month horizon
    // Simplified scaling
    1.0 - (-(horizon_days as f64).sqrt() * mes_1day).exp()
}

/// SRISK: Systemic Risk (Brownlees-Engle)
/// Capital shortfall conditional on crisis
pub fn srisk(
    market_cap: f64, book_debt: f64,
    lrmes: f64, prudential_ratio: f64,
) -> f64 {
    // SRISK = max(0, k(D + MV) - MV(1 - LRMES))
    // k = prudential capital ratio (e.g., 0.08)
    let total_assets = book_debt + market_cap;
    let shortfall = prudential_ratio * total_assets - market_cap * (1.0 - lrmes);
    shortfall.max(0.0)
}

/// Systemic importance ranking
pub fn systemic_importance_ranking(
    names: &[&str],
    market_caps: &[f64],
    srisk_values: &[f64],
    covar_values: &[f64],
    mes_values: &[f64],
) -> Vec<(String, f64)> {
    let n = names.len();
    assert_eq!(market_caps.len(), n);
    assert_eq!(srisk_values.len(), n);

    // Composite score: weighted combination
    let total_srisk: f64 = srisk_values.iter().sum::<f64>().max(1e-10);
    let max_covar = covar_values.iter().cloned().fold(1e-10_f64, f64::max);
    let max_mes = mes_values.iter().cloned().fold(1e-10_f64, f64::max);
    let total_mc: f64 = market_caps.iter().sum::<f64>().max(1e-10);

    let mut scores: Vec<(String, f64)> = (0..n).map(|i| {
        let score = 0.30 * (srisk_values[i] / total_srisk)
            + 0.25 * (covar_values[i] / max_covar)
            + 0.25 * (mes_values[i] / max_mes)
            + 0.20 * (market_caps[i] / total_mc);
        (names[i].to_string(), score)
    }).collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

// ============================================================
// DebtRank
// ============================================================

/// DebtRank network model
pub struct DebtRank {
    pub n_banks: usize,
    pub exposures: Vec<Vec<f64>>, // exposures[i][j] = bank i's exposure to bank j
    pub equity: Vec<f64>,
    pub external_assets: Vec<f64>,
}

impl DebtRank {
    pub fn new(exposures: Vec<Vec<f64>>, equity: Vec<f64>, external_assets: Vec<f64>) -> Self {
        let n = equity.len();
        Self { n_banks: n, exposures, equity, external_assets }
    }

    /// Compute DebtRank from initial shock
    pub fn compute(&self, initial_shocks: &[f64], max_rounds: usize) -> Vec<f64> {
        let n = self.n_banks;
        let mut h = initial_shocks.to_vec(); // cumulative loss fraction for each bank
        let mut active: Vec<bool> = vec![true; n];
        let mut total_equity: f64 = self.equity.iter().sum();

        // Relative economic value
        let total_assets: f64 = self.equity.iter().zip(&self.external_assets)
            .map(|(e, a)| e + a).sum();

        for _ in 0..max_rounds {
            let mut new_h = h.clone();
            let mut changed = false;

            for j in 0..n {
                if !active[j] { continue; }
                let mut stress = 0.0;
                for i in 0..n {
                    if i == j { continue; }
                    let delta_h = h[i]; // loss fraction of bank i
                    if delta_h > 0.0 && self.equity[j] > 1e-10 {
                        stress += self.exposures[j][i] * delta_h / self.equity[j];
                    }
                }
                let new_loss = (h[j] + stress).min(1.0);
                if (new_loss - new_h[j]).abs() > 1e-10 {
                    changed = true;
                }
                new_h[j] = new_loss;
                if new_h[j] >= 1.0 { active[j] = false; }
            }

            h = new_h;
            if !changed { break; }
        }

        // DebtRank = sum of value-weighted losses
        h
    }

    /// System-wide DebtRank
    pub fn system_debtrank(&self, initial_shocks: &[f64], max_rounds: usize) -> f64 {
        let h = self.compute(initial_shocks, max_rounds);
        let total_equity: f64 = self.equity.iter().sum();
        if total_equity < 1e-10 { return 0.0; }
        h.iter().zip(&self.equity).map(|(&hi, &ei)| hi * ei).sum::<f64>() / total_equity
    }

    /// Compute DebtRank for each bank failing individually
    pub fn individual_debtrank(&self, max_rounds: usize) -> Vec<f64> {
        let n = self.n_banks;
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let mut shock = vec![0.0; n];
            shock[i] = 1.0;
            results.push(self.system_debtrank(&shock, max_rounds));
        }
        results
    }
}

// ============================================================
// Eisenberg-Noe Clearing
// ============================================================

/// Eisenberg-Noe clearing payments model
pub struct EisenbergNoe {
    pub n_banks: usize,
    pub liabilities: Vec<Vec<f64>>, // liabilities[i][j] = bank i owes bank j
    pub external_assets: Vec<f64>,
}

impl EisenbergNoe {
    pub fn new(liabilities: Vec<Vec<f64>>, external_assets: Vec<f64>) -> Self {
        let n = external_assets.len();
        Self { n_banks: n, liabilities, external_assets }
    }

    /// Compute clearing payments via fixed-point iteration (Fictitious Default Algorithm)
    pub fn clearing_payments(&self, max_iter: usize, tol: f64) -> Vec<f64> {
        let n = self.n_banks;
        // Total obligations
        let total_liab: Vec<f64> = (0..n).map(|i| {
            self.liabilities[i].iter().sum::<f64>()
        }).collect();

        // Relative liabilities matrix: π[i][j] = L[i][j] / total_liab[i]
        let pi: Vec<Vec<f64>> = (0..n).map(|i| {
            if total_liab[i] > 1e-10 {
                self.liabilities[i].iter().map(|&l| l / total_liab[i]).collect()
            } else {
                vec![0.0; n]
            }
        }).collect();

        // Initialize payments to full obligations
        let mut p = total_liab.clone();

        for _ in 0..max_iter {
            let mut new_p = vec![0.0; n];
            let mut converged = true;

            for i in 0..n {
                // Cash available = external assets + payments received
                let received: f64 = (0..n).map(|j| pi[j][i] * p[j]).sum();
                let available = self.external_assets[i] + received;
                new_p[i] = available.min(total_liab[i]).max(0.0);

                if (new_p[i] - p[i]).abs() > tol { converged = false; }
            }

            p = new_p;
            if converged { break; }
        }
        p
    }

    /// Compute equity after clearing
    pub fn post_clearing_equity(&self) -> Vec<f64> {
        let n = self.n_banks;
        let payments = self.clearing_payments(1000, 1e-10);
        let total_liab: Vec<f64> = (0..n).map(|i| self.liabilities[i].iter().sum::<f64>()).collect();

        let pi: Vec<Vec<f64>> = (0..n).map(|i| {
            if total_liab[i] > 1e-10 {
                self.liabilities[i].iter().map(|&l| l / total_liab[i]).collect()
            } else {
                vec![0.0; n]
            }
        }).collect();

        (0..n).map(|i| {
            let received: f64 = (0..n).map(|j| pi[j][i] * payments[j]).sum();
            let available = self.external_assets[i] + received;
            (available - total_liab[i]).max(0.0)
        }).collect()
    }

    /// Identify defaulting banks
    pub fn defaults(&self) -> Vec<bool> {
        let n = self.n_banks;
        let payments = self.clearing_payments(1000, 1e-10);
        let total_liab: Vec<f64> = (0..n).map(|i| self.liabilities[i].iter().sum::<f64>()).collect();
        (0..n).map(|i| payments[i] < total_liab[i] - 1e-10).collect()
    }

    /// System losses from initial shock to external assets
    pub fn system_loss_from_shock(&self, asset_shocks: &[f64]) -> f64 {
        let n = self.n_banks;
        let shocked_assets: Vec<f64> = self.external_assets.iter().zip(asset_shocks)
            .map(|(&a, &s)| (a * (1.0 + s)).max(0.0))
            .collect();

        let shocked_model = EisenbergNoe::new(self.liabilities.clone(), shocked_assets);
        let equity_before = self.post_clearing_equity();
        let equity_after = shocked_model.post_clearing_equity();

        equity_before.iter().zip(&equity_after)
            .map(|(b, a)| (b - a).max(0.0))
            .sum()
    }
}

// ============================================================
// Fire Sale Externalities
// ============================================================

/// Fire sale model: forced selling causes price impact, causing more losses
pub struct FireSaleModel {
    pub n_banks: usize,
    pub n_assets: usize,
    pub holdings: Vec<Vec<f64>>,   // holdings[bank][asset]
    pub equity: Vec<f64>,
    pub leverage_limit: f64,       // max leverage ratio
    pub price_impact: Vec<f64>,    // price impact per unit sold for each asset
}

impl FireSaleModel {
    pub fn new(
        holdings: Vec<Vec<f64>>, equity: Vec<f64>,
        leverage_limit: f64, price_impact: Vec<f64>,
    ) -> Self {
        let n_banks = equity.len();
        let n_assets = price_impact.len();
        Self { n_banks, n_assets, holdings, equity, leverage_limit, price_impact }
    }

    /// Simulate fire sale cascade from initial price shock
    pub fn simulate(&self, initial_price_shock: &[f64], max_rounds: usize) -> FireSaleResult {
        let n = self.n_banks;
        let m = self.n_assets;

        let mut prices: Vec<f64> = vec![1.0; m]; // normalized to 1
        for j in 0..m { prices[j] += initial_price_shock[j]; }

        let mut total_sold = vec![vec![0.0; m]; n];
        let mut current_equity = self.equity.clone();

        for round in 0..max_rounds {
            let mut sales = vec![vec![0.0; m]; n];
            let mut any_sale = false;

            for i in 0..n {
                // Current portfolio value
                let portfolio_val: f64 = (0..m).map(|j| {
                    (self.holdings[i][j] - total_sold[i][j]) * prices[j]
                }).sum();
                let leverage = portfolio_val / current_equity[i].max(1e-10);

                if leverage > self.leverage_limit && current_equity[i] > 0.0 {
                    // Need to sell to reduce leverage
                    let target_val = current_equity[i] * self.leverage_limit;
                    let excess = portfolio_val - target_val;
                    if excess > 0.0 {
                        // Sell proportionally
                        for j in 0..m {
                            let holding_val = (self.holdings[i][j] - total_sold[i][j]) * prices[j];
                            let sell_val = excess * holding_val / portfolio_val.max(1e-10);
                            let sell_qty = sell_val / prices[j].max(1e-10);
                            sales[i][j] = sell_qty.min(self.holdings[i][j] - total_sold[i][j]);
                        }
                        any_sale = true;
                    }
                }
            }

            if !any_sale { break; }

            // Price impact from aggregate sales
            for j in 0..m {
                let total_sale: f64 = (0..n).map(|i| sales[i][j]).sum();
                prices[j] -= self.price_impact[j] * total_sale;
                prices[j] = prices[j].max(0.01);
            }

            // Update equity and track sales
            for i in 0..n {
                for j in 0..m {
                    total_sold[i][j] += sales[i][j];
                }
                current_equity[i] = (0..m).map(|j| {
                    (self.holdings[i][j] - total_sold[i][j]) * prices[j]
                }).sum::<f64>() - (self.equity[i] * (self.leverage_limit - 1.0));
                current_equity[i] = current_equity[i].max(0.0);
            }
        }

        let total_loss: f64 = self.equity.iter().zip(current_equity.iter())
            .map(|(e0, e1)| (e0 - e1).max(0.0))
            .sum();

        let defaults: Vec<bool> = self.equity.iter().zip(current_equity.iter())
            .map(|(&e0, &e1)| e1 < 0.01 * e0)
            .collect();

        FireSaleResult {
            final_prices: prices,
            final_equity: current_equity,
            total_system_loss: total_loss,
            defaults,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FireSaleResult {
    pub final_prices: Vec<f64>,
    pub final_equity: Vec<f64>,
    pub total_system_loss: f64,
    pub defaults: Vec<bool>,
}

// ============================================================
// Contagion Simulation
// ============================================================

/// Network contagion simulation (SIR-like model for financial distress)
pub fn contagion_simulation(
    adjacency: &[Vec<f64>], // adjacency[i][j] = exposure weight
    initial_infected: &[bool],
    infection_threshold: f64,
    max_rounds: usize,
) -> Vec<Vec<bool>> {
    let n = adjacency.len();
    let mut state: Vec<bool> = initial_infected.to_vec();
    let mut history = vec![state.clone()];

    for _ in 0..max_rounds {
        let mut new_state = state.clone();
        let mut changed = false;

        for i in 0..n {
            if state[i] { continue; } // already infected
            let stress: f64 = (0..n).map(|j| {
                if state[j] { adjacency[i][j] } else { 0.0 }
            }).sum();
            if stress > infection_threshold {
                new_state[i] = true;
                changed = true;
            }
        }

        state = new_state;
        history.push(state.clone());
        if !changed { break; }
    }
    history
}

/// Compute network centrality measures
pub fn degree_centrality(adjacency: &[Vec<f64>]) -> Vec<f64> {
    let n = adjacency.len();
    (0..n).map(|i| {
        adjacency[i].iter().filter(|&&w| w > 1e-10).count() as f64 / (n - 1) as f64
    }).collect()
}

/// Eigenvector centrality (via power iteration)
pub fn eigenvector_centrality(adjacency: &[Vec<f64>], max_iter: usize) -> Vec<f64> {
    let n = adjacency.len();
    let mut v = vec![1.0 / n as f64; n];
    for _ in 0..max_iter {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n { new_v[i] += adjacency[i][j] * v[j]; }
        }
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 { for x in &mut new_v { *x /= norm; } }
        v = new_v;
    }
    v
}

/// Betweenness centrality (simplified, unweighted)
pub fn betweenness_centrality(adjacency: &[Vec<f64>]) -> Vec<f64> {
    let n = adjacency.len();
    let mut bc = vec![0.0; n];

    for s in 0..n {
        // BFS from s
        let mut dist = vec![usize::MAX; n];
        let mut sigma = vec![0.0; n]; // number of shortest paths
        let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
        let mut queue = std::collections::VecDeque::new();

        dist[s] = 0;
        sigma[s] = 1.0;
        queue.push_back(s);

        let mut stack = Vec::new();

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for w in 0..n {
                if adjacency[v][w] < 1e-10 { continue; }
                if dist[w] == usize::MAX {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s { bc[w] += delta[w]; }
        }
    }

    // Normalize
    let factor = 2.0 / ((n as f64 - 1.0) * (n as f64 - 2.0)).max(1.0);
    for x in &mut bc { *x *= factor; }
    bc
}

/// Systemic risk summary
pub struct SystemicRiskSummary {
    pub total_srisk: f64,
    pub avg_mes: f64,
    pub max_covar: f64,
    pub network_density: f64,
    pub systemic_concentration: f64, // HHI of SRISK
}

impl SystemicRiskSummary {
    pub fn compute(
        srisk_values: &[f64], mes_values: &[f64], covar_values: &[f64],
        adjacency: &[Vec<f64>],
    ) -> Self {
        let n = srisk_values.len();
        let total_srisk: f64 = srisk_values.iter().sum();
        let avg_mes = statistics::mean(mes_values);
        let max_covar = covar_values.iter().cloned().fold(0.0_f64, f64::max);

        let n_adj = adjacency.len();
        let n_edges: usize = adjacency.iter().flat_map(|row| row.iter())
            .filter(|&&w| w > 1e-10).count();
        let max_edges = n_adj * (n_adj - 1);
        let network_density = if max_edges > 0 { n_edges as f64 / max_edges as f64 } else { 0.0 };

        let hhi: f64 = if total_srisk > 1e-10 {
            srisk_values.iter().map(|s| (s / total_srisk).powi(2)).sum()
        } else { 0.0 };

        Self {
            total_srisk,
            avg_mes,
            max_covar,
            network_density,
            systemic_concentration: hhi,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covar() {
        let sys = vec![-0.05, 0.02, -0.03, 0.01, -0.04, 0.03, -0.02, 0.01, -0.06, 0.02,
                       0.01, -0.01, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.02, -0.04];
        let inst = vec![-0.08, 0.01, -0.05, 0.02, -0.06, 0.04, -0.03, 0.01, -0.09, 0.03,
                        0.02, -0.02, 0.01, -0.04, 0.02, -0.03, 0.04, -0.01, 0.01, -0.07];
        let (cv, dcv) = covar(&sys, &inst, 0.95);
        assert!(cv > 0.0);
    }

    #[test]
    fn test_mes() {
        let sys = vec![-0.05, 0.02, -0.03, 0.01, -0.04, 0.03, -0.02, 0.01, -0.06, 0.02];
        let inst = vec![-0.08, 0.01, -0.05, 0.02, -0.06, 0.04, -0.03, 0.01, -0.09, 0.03];
        let m = mes(&sys, &inst, 0.05);
        assert!(m > 0.0);
    }

    #[test]
    fn test_eisenberg_noe() {
        let liab = vec![
            vec![0.0, 10.0, 5.0],
            vec![8.0, 0.0, 7.0],
            vec![3.0, 6.0, 0.0],
        ];
        let assets = vec![20.0, 18.0, 15.0];
        let model = EisenbergNoe::new(liab, assets);
        let payments = model.clearing_payments(100, 1e-10);
        assert!(payments.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_debtrank() {
        let exp = vec![
            vec![0.0, 5.0, 3.0],
            vec![4.0, 0.0, 2.0],
            vec![2.0, 3.0, 0.0],
        ];
        let equity = vec![10.0, 8.0, 6.0];
        let external = vec![20.0, 15.0, 12.0];
        let dr = DebtRank::new(exp, equity, external);
        let shocks = vec![1.0, 0.0, 0.0]; // bank 0 defaults
        let dr_val = dr.system_debtrank(&shocks, 10);
        assert!(dr_val > 0.0);
    }
}
