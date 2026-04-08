use crate::constraints::{BoxConstraint, TurnoverConstraint, CardinalityConstraint};

// ═══════════════════════════════════════════════════════════════════════════
// REBALANCING TYPES
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct PortfolioState {
    pub asset_names: Vec<String>,
    pub weights: Vec<f64>,
    pub market_values: Vec<f64>,
    pub total_value: f64,
    pub cash: f64,
}

impl PortfolioState {
    pub fn new(names: Vec<String>, values: Vec<f64>, cash: f64) -> Self {
        let total: f64 = values.iter().sum::<f64>() + cash;
        let weights: Vec<f64> = values.iter().map(|v| v / total).collect();
        Self {
            asset_names: names,
            weights,
            market_values: values,
            total_value: total,
            cash,
        }
    }

    pub fn n_assets(&self) -> usize {
        self.weights.len()
    }

    pub fn drift_from_target(&self, target: &[f64]) -> Vec<f64> {
        self.weights.iter().zip(target.iter())
            .map(|(w, t)| w - t)
            .collect()
    }

    pub fn max_drift(&self, target: &[f64]) -> f64 {
        self.drift_from_target(target).iter()
            .map(|d| d.abs())
            .fold(0.0, f64::max)
    }

    pub fn total_drift(&self, target: &[f64]) -> f64 {
        self.drift_from_target(target).iter()
            .map(|d| d.abs())
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub asset_idx: usize,
    pub asset_name: String,
    pub direction: TradeDirection,
    pub weight_change: f64,
    pub dollar_amount: f64,
    pub shares: f64,
    pub estimated_cost: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeDirection {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone)]
pub struct RebalanceResult {
    pub trades: Vec<Trade>,
    pub new_weights: Vec<f64>,
    pub total_turnover: f64,
    pub total_cost: f64,
    pub n_trades: usize,
    pub triggered: bool,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct TransactionCostModel {
    pub fixed_cost_per_trade: f64,
    pub proportional_cost_bps: f64,     // basis points
    pub market_impact_coeff: f64,       // sqrt impact coefficient
    pub min_trade_size: f64,            // minimum dollar trade
    pub per_asset_costs: Vec<f64>,      // asset-specific costs
}

impl TransactionCostModel {
    pub fn default_model(n: usize) -> Self {
        Self {
            fixed_cost_per_trade: 5.0,
            proportional_cost_bps: 5.0,
            market_impact_coeff: 0.1,
            min_trade_size: 100.0,
            per_asset_costs: vec![5.0; n],
        }
    }

    pub fn zero_cost() -> Self {
        Self {
            fixed_cost_per_trade: 0.0,
            proportional_cost_bps: 0.0,
            market_impact_coeff: 0.0,
            min_trade_size: 0.0,
            per_asset_costs: vec![],
        }
    }

    pub fn trade_cost(&self, dollar_amount: f64, asset_idx: usize) -> f64 {
        let abs_amount = dollar_amount.abs();
        if abs_amount < self.min_trade_size {
            return 0.0; // skip small trades
        }
        let proportional = abs_amount * self.proportional_cost_bps / 10000.0;
        let impact = self.market_impact_coeff * abs_amount.sqrt();
        let fixed = self.fixed_cost_per_trade;
        let asset_specific = if asset_idx < self.per_asset_costs.len() {
            abs_amount * self.per_asset_costs[asset_idx] / 10000.0
        } else {
            0.0
        };
        fixed + proportional + impact + asset_specific
    }

    pub fn total_cost(&self, trades: &[Trade]) -> f64 {
        trades.iter().map(|t| t.estimated_cost).sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// THRESHOLD-BASED REBALANCING
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct ThresholdRebalancer {
    pub absolute_threshold: f64,     // trigger if any asset drifts by this much
    pub relative_threshold: f64,     // trigger if drift / target > this
    pub total_drift_threshold: f64,  // trigger if sum of drifts exceeds this
    pub cost_model: TransactionCostModel,
    pub partial_rebalance: bool,     // if true, only trade assets that drifted
    pub rebalance_to_target: bool,   // if true, go to exact target; else, to band edge
}

impl ThresholdRebalancer {
    pub fn new(abs_threshold: f64, cost_model: TransactionCostModel) -> Self {
        Self {
            absolute_threshold: abs_threshold,
            relative_threshold: 0.25,
            total_drift_threshold: abs_threshold * 2.0,
            cost_model,
            partial_rebalance: false,
            rebalance_to_target: true,
        }
    }

    /// Check if rebalancing is triggered.
    pub fn should_rebalance(&self, current: &PortfolioState, target: &[f64]) -> (bool, String) {
        let max_drift = current.max_drift(target);
        if max_drift > self.absolute_threshold {
            return (true, format!("Max drift {:.4} > threshold {:.4}", max_drift, self.absolute_threshold));
        }

        // Relative threshold
        for (i, (&w, &t)) in current.weights.iter().zip(target.iter()).enumerate() {
            if t > 1e-10 {
                let rel_drift = (w - t).abs() / t;
                if rel_drift > self.relative_threshold {
                    return (true, format!("Asset {} relative drift {:.2}% > {:.2}%",
                        i, rel_drift * 100.0, self.relative_threshold * 100.0));
                }
            }
        }

        let total_drift = current.total_drift(target);
        if total_drift > self.total_drift_threshold {
            return (true, format!("Total drift {:.4} > threshold {:.4}", total_drift, self.total_drift_threshold));
        }

        (false, "Within bands".to_string())
    }

    /// Generate rebalancing trades.
    pub fn rebalance(&self, current: &PortfolioState, target: &[f64]) -> RebalanceResult {
        let (triggered, reason) = self.should_rebalance(current, target);

        if !triggered {
            return RebalanceResult {
                trades: vec![],
                new_weights: current.weights.clone(),
                total_turnover: 0.0,
                total_cost: 0.0,
                n_trades: 0,
                triggered: false,
                reason,
            };
        }

        let n = current.n_assets();
        let mut trades = Vec::new();
        let mut total_turnover = 0.0;
        let mut total_cost = 0.0;

        for i in 0..n {
            let drift = current.weights[i] - target[i];
            let should_trade = if self.partial_rebalance {
                drift.abs() > self.absolute_threshold
            } else {
                true
            };

            if !should_trade || drift.abs() < 1e-10 {
                continue;
            }

            let target_weight = if self.rebalance_to_target {
                target[i]
            } else {
                // Rebalance to band edge
                if drift > 0.0 {
                    target[i] + self.absolute_threshold * 0.5
                } else {
                    target[i] - self.absolute_threshold * 0.5
                }
            };

            let weight_change = target_weight - current.weights[i];
            let dollar_amount = weight_change * current.total_value;
            let cost = self.cost_model.trade_cost(dollar_amount, i);

            let direction = if weight_change > 1e-10 {
                TradeDirection::Buy
            } else if weight_change < -1e-10 {
                TradeDirection::Sell
            } else {
                TradeDirection::Hold
            };

            if dollar_amount.abs() >= self.cost_model.min_trade_size {
                trades.push(Trade {
                    asset_idx: i,
                    asset_name: current.asset_names.get(i).cloned().unwrap_or_default(),
                    direction,
                    weight_change,
                    dollar_amount,
                    shares: 0.0,
                    estimated_cost: cost,
                });
                total_turnover += weight_change.abs();
                total_cost += cost;
            }
        }

        let new_weights: Vec<f64> = (0..n).map(|i| {
            let trade = trades.iter().find(|t| t.asset_idx == i);
            match trade {
                Some(t) => current.weights[i] + t.weight_change,
                None => current.weights[i],
            }
        }).collect();

        RebalanceResult {
            n_trades: trades.len(),
            trades,
            new_weights,
            total_turnover,
            total_cost,
            triggered: true,
            reason,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CALENDAR-BASED REBALANCING
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalendarFrequency {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    SemiAnnually,
    Annually,
}

impl CalendarFrequency {
    pub fn days(&self) -> u32 {
        match self {
            CalendarFrequency::Daily => 1,
            CalendarFrequency::Weekly => 7,
            CalendarFrequency::Monthly => 30,
            CalendarFrequency::Quarterly => 91,
            CalendarFrequency::SemiAnnually => 182,
            CalendarFrequency::Annually => 365,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CalendarRebalancer {
    pub frequency: CalendarFrequency,
    pub days_since_last: u32,
    pub cost_model: TransactionCostModel,
    pub min_drift_to_trade: f64, // don't trade if drift is tiny even on schedule
}

impl CalendarRebalancer {
    pub fn new(freq: CalendarFrequency, cost_model: TransactionCostModel) -> Self {
        Self {
            frequency: freq,
            days_since_last: 0,
            cost_model,
            min_drift_to_trade: 0.001,
        }
    }

    pub fn should_rebalance(&self) -> bool {
        self.days_since_last >= self.frequency.days()
    }

    pub fn rebalance(&mut self, current: &PortfolioState, target: &[f64]) -> RebalanceResult {
        if !self.should_rebalance() {
            return RebalanceResult {
                trades: vec![], new_weights: current.weights.clone(),
                total_turnover: 0.0, total_cost: 0.0, n_trades: 0,
                triggered: false, reason: format!("{} days until next rebalance", self.frequency.days() - self.days_since_last),
            };
        }

        self.days_since_last = 0;

        let n = current.n_assets();
        let mut trades = Vec::new();
        let mut total_cost = 0.0;
        let mut total_turnover = 0.0;

        for i in 0..n {
            let weight_change = target[i] - current.weights[i];
            if weight_change.abs() < self.min_drift_to_trade {
                continue;
            }
            let dollar_amount = weight_change * current.total_value;
            let cost = self.cost_model.trade_cost(dollar_amount, i);
            let direction = if weight_change > 0.0 { TradeDirection::Buy } else { TradeDirection::Sell };

            trades.push(Trade {
                asset_idx: i,
                asset_name: current.asset_names.get(i).cloned().unwrap_or_default(),
                direction,
                weight_change,
                dollar_amount,
                shares: 0.0,
                estimated_cost: cost,
            });
            total_turnover += weight_change.abs();
            total_cost += cost;
        }

        let new_weights = target.to_vec();
        RebalanceResult {
            n_trades: trades.len(), trades, new_weights,
            total_turnover, total_cost, triggered: true,
            reason: "Calendar rebalance triggered".into(),
        }
    }

    pub fn advance_day(&mut self) {
        self.days_since_last += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RISK-TARGET REBALANCING
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RiskTargetRebalancer {
    pub target_volatility: f64,
    pub vol_tolerance: f64,         // rebalance if |vol - target| > tolerance
    pub covariance: Vec<Vec<f64>>,
    pub cost_model: TransactionCostModel,
}

impl RiskTargetRebalancer {
    pub fn new(target_vol: f64, tolerance: f64, cov: Vec<Vec<f64>>, cost_model: TransactionCostModel) -> Self {
        Self { target_volatility: target_vol, vol_tolerance: tolerance, covariance: cov, cost_model }
    }

    pub fn portfolio_vol(&self, weights: &[f64]) -> f64 {
        let n = weights.len();
        let mut var = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i < self.covariance.len() && j < self.covariance[i].len() {
                    var += weights[i] * weights[j] * self.covariance[i][j];
                }
            }
        }
        var.max(0.0).sqrt()
    }

    pub fn should_rebalance(&self, weights: &[f64]) -> (bool, f64) {
        let vol = self.portfolio_vol(weights);
        let diff = (vol - self.target_volatility).abs();
        (diff > self.vol_tolerance, vol)
    }

    /// Scale portfolio to hit target vol while maintaining relative weights.
    pub fn rebalance(&self, current: &PortfolioState, base_weights: &[f64]) -> RebalanceResult {
        let (triggered, current_vol) = self.should_rebalance(&current.weights);

        if !triggered {
            return RebalanceResult {
                trades: vec![], new_weights: current.weights.clone(),
                total_turnover: 0.0, total_cost: 0.0, n_trades: 0,
                triggered: false,
                reason: format!("Vol {:.2}% within tolerance of target {:.2}%",
                    current_vol * 100.0, self.target_volatility * 100.0),
            };
        }

        // Scale: new_weights = scale * base_weights + (1-scale) * cash
        let base_vol = self.portfolio_vol(base_weights);
        let scale = if base_vol > 1e-15 {
            self.target_volatility / base_vol
        } else {
            1.0
        };
        let scale = scale.min(2.0).max(0.0); // cap leverage

        let n = current.n_assets();
        let new_weights: Vec<f64> = base_weights.iter().map(|&w| w * scale).collect();

        let mut trades = Vec::new();
        let mut total_turnover = 0.0;
        let mut total_cost = 0.0;

        for i in 0..n {
            let weight_change = new_weights[i] - current.weights[i];
            if weight_change.abs() < 1e-6 {
                continue;
            }
            let dollar_amount = weight_change * current.total_value;
            let cost = self.cost_model.trade_cost(dollar_amount, i);
            let direction = if weight_change > 0.0 { TradeDirection::Buy } else { TradeDirection::Sell };

            trades.push(Trade {
                asset_idx: i,
                asset_name: current.asset_names.get(i).cloned().unwrap_or_default(),
                direction, weight_change, dollar_amount, shares: 0.0, estimated_cost: cost,
            });
            total_turnover += weight_change.abs();
            total_cost += cost;
        }

        RebalanceResult {
            n_trades: trades.len(), trades, new_weights,
            total_turnover, total_cost, triggered: true,
            reason: format!("Vol {:.2}% vs target {:.2}%, scale={:.4}",
                current_vol * 100.0, self.target_volatility * 100.0, scale),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TAX-AWARE REBALANCING
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct TaxLot {
    pub asset_idx: usize,
    pub shares: f64,
    pub cost_basis: f64,
    pub acquisition_date: u64, // days since epoch
    pub is_long_term: bool,
}

impl TaxLot {
    pub fn unrealized_gain(&self, current_price: f64) -> f64 {
        (current_price - self.cost_basis) * self.shares
    }

    pub fn gain_per_share(&self, current_price: f64) -> f64 {
        current_price - self.cost_basis
    }
}

#[derive(Debug, Clone)]
pub struct TaxAwareRebalancer {
    pub short_term_rate: f64,
    pub long_term_rate: f64,
    pub loss_harvesting_threshold: f64,  // min loss to harvest
    pub wash_sale_days: u32,
    pub cost_model: TransactionCostModel,
    pub max_tax_budget: f64,             // max taxes willing to pay
}

impl TaxAwareRebalancer {
    pub fn new(st_rate: f64, lt_rate: f64, cost_model: TransactionCostModel) -> Self {
        Self {
            short_term_rate: st_rate,
            long_term_rate: lt_rate,
            loss_harvesting_threshold: 1000.0,
            wash_sale_days: 30,
            cost_model,
            max_tax_budget: f64::INFINITY,
        }
    }

    /// Tax cost of selling a specific lot.
    pub fn tax_cost(&self, lot: &TaxLot, current_price: f64, shares_to_sell: f64) -> f64 {
        let gain = (current_price - lot.cost_basis) * shares_to_sell;
        if gain <= 0.0 {
            return gain * (if lot.is_long_term { self.long_term_rate } else { self.short_term_rate });
        }
        let rate = if lot.is_long_term { self.long_term_rate } else { self.short_term_rate };
        gain * rate
    }

    /// Choose lots to sell using tax-optimal strategy (HIFO: Highest In, First Out).
    pub fn select_lots_hifo(lots: &[TaxLot], shares_needed: f64) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = lots.iter().enumerate()
            .map(|(i, lot)| (i, lot.cost_basis))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // highest cost first

        let mut remaining = shares_needed;
        let mut selections = Vec::new();
        for &(idx, _) in &indexed {
            if remaining <= 0.0 { break; }
            let sell = remaining.min(lots[idx].shares);
            selections.push((idx, sell));
            remaining -= sell;
        }
        selections
    }

    /// Choose lots to sell using specific identification to minimize tax.
    pub fn select_lots_min_tax(lots: &[TaxLot], shares_needed: f64, current_price: f64) -> Vec<(usize, f64)> {
        // Sort by tax cost (sell losses first, then lowest-gain lots)
        let mut indexed: Vec<(usize, f64)> = lots.iter().enumerate()
            .map(|(i, lot)| (i, lot.gain_per_share(current_price)))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // lowest gain first

        let mut remaining = shares_needed;
        let mut selections = Vec::new();
        for &(idx, _) in &indexed {
            if remaining <= 0.0 { break; }
            let sell = remaining.min(lots[idx].shares);
            selections.push((idx, sell));
            remaining -= sell;
        }
        selections
    }

    /// Identify tax-loss harvesting opportunities.
    pub fn loss_harvesting_opportunities(
        &self, lots: &[TaxLot], prices: &[f64],
    ) -> Vec<(usize, f64, f64)> {
        // Returns: (lot_idx, unrealized_loss, tax_benefit)
        let mut opportunities = Vec::new();
        for (i, lot) in lots.iter().enumerate() {
            if lot.asset_idx < prices.len() {
                let loss = lot.unrealized_gain(prices[lot.asset_idx]);
                if loss < -self.loss_harvesting_threshold {
                    let rate = if lot.is_long_term { self.long_term_rate } else { self.short_term_rate };
                    let tax_benefit = -loss * rate;
                    opportunities.push((i, loss, tax_benefit));
                }
            }
        }
        opportunities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // largest loss first
        opportunities
    }

    /// Tax-aware rebalance: minimize tracking error subject to tax budget.
    pub fn rebalance(
        &self,
        current: &PortfolioState,
        target: &[f64],
        lots: &[TaxLot],
        prices: &[f64],
    ) -> RebalanceResult {
        let n = current.n_assets();
        let mut new_weights = target.to_vec();
        let mut trades = Vec::new();
        let mut total_tax = 0.0;
        let mut total_cost = 0.0;
        let mut total_turnover = 0.0;

        for i in 0..n {
            let weight_change = target[i] - current.weights[i];
            if weight_change.abs() < 1e-6 {
                continue;
            }

            let dollar_amount = weight_change * current.total_value;

            // Check tax cost for sells
            if weight_change < 0.0 && i < prices.len() {
                let asset_lots: Vec<&TaxLot> = lots.iter().filter(|l| l.asset_idx == i).collect();
                let shares_to_sell = (-dollar_amount) / prices[i].max(0.01);
                let mut tax_for_this_trade = 0.0;
                for lot in &asset_lots {
                    let gain = lot.gain_per_share(prices[i]);
                    let rate = if lot.is_long_term { self.long_term_rate } else { self.short_term_rate };
                    tax_for_this_trade += gain.max(0.0) * rate * shares_to_sell.min(lot.shares);
                }

                // If tax exceeds budget, reduce the trade
                if total_tax + tax_for_this_trade > self.max_tax_budget {
                    let remaining_budget = (self.max_tax_budget - total_tax).max(0.0);
                    let scale = if tax_for_this_trade > 0.0 { remaining_budget / tax_for_this_trade } else { 1.0 };
                    let adj_change = weight_change * scale;
                    new_weights[i] = current.weights[i] + adj_change;
                    total_tax += tax_for_this_trade * scale;
                } else {
                    total_tax += tax_for_this_trade;
                }
            }

            let actual_change = new_weights[i] - current.weights[i];
            if actual_change.abs() < 1e-6 { continue; }

            let actual_dollar = actual_change * current.total_value;
            let cost = self.cost_model.trade_cost(actual_dollar, i);
            let direction = if actual_change > 0.0 { TradeDirection::Buy } else { TradeDirection::Sell };

            trades.push(Trade {
                asset_idx: i,
                asset_name: current.asset_names.get(i).cloned().unwrap_or_default(),
                direction, weight_change: actual_change, dollar_amount: actual_dollar,
                shares: actual_dollar.abs() / prices.get(i).copied().unwrap_or(1.0),
                estimated_cost: cost,
            });
            total_turnover += actual_change.abs();
            total_cost += cost;
        }

        RebalanceResult {
            n_trades: trades.len(), trades, new_weights,
            total_turnover, total_cost, triggered: true,
            reason: format!("Tax-aware rebalance, est tax: ${:.2}", total_tax),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRANSACTION COST AWARE REBALANCING
// ═══════════════════════════════════════════════════════════════════════════

/// Net-of-cost optimization: only trade if improvement exceeds cost.
pub fn cost_aware_rebalance(
    current: &PortfolioState,
    target: &[f64],
    expected_returns: &[f64],
    covariance: &[Vec<f64>],
    cost_model: &TransactionCostModel,
    risk_aversion: f64,
) -> RebalanceResult {
    let n = current.n_assets();

    // Compute utility improvement from rebalancing
    let current_utility = portfolio_utility(&current.weights, expected_returns, covariance, risk_aversion);
    let target_utility = portfolio_utility(target, expected_returns, covariance, risk_aversion);

    // Estimate transaction costs
    let total_cost: f64 = (0..n).map(|i| {
        let dollar_trade = (target[i] - current.weights[i]).abs() * current.total_value;
        cost_model.trade_cost(dollar_trade, i)
    }).sum();
    let cost_drag = total_cost / current.total_value;

    // Only rebalance if utility improvement exceeds cost
    if target_utility - current_utility < cost_drag * 2.0 {
        return RebalanceResult {
            trades: vec![], new_weights: current.weights.clone(),
            total_turnover: 0.0, total_cost: 0.0, n_trades: 0,
            triggered: false,
            reason: format!("Utility improvement {:.6} < cost drag {:.6}",
                target_utility - current_utility, cost_drag),
        };
    }

    // Partial rebalance: trade only where benefit exceeds per-asset cost
    let mut new_weights = current.weights.clone();
    let mut trades = Vec::new();
    let mut total_turnover = 0.0;
    let mut actual_cost = 0.0;

    for i in 0..n {
        let weight_change = target[i] - current.weights[i];
        let dollar_trade = weight_change.abs() * current.total_value;
        let trade_cost = cost_model.trade_cost(dollar_trade, i);

        // Marginal utility improvement from this trade
        let mut marginal_utility = expected_returns[i] * weight_change;
        for j in 0..n {
            marginal_utility -= risk_aversion * covariance[i][j] * current.weights[j] * weight_change;
        }

        if marginal_utility.abs() > trade_cost / current.total_value || weight_change.abs() > 0.05 {
            new_weights[i] = target[i];
            let direction = if weight_change > 0.0 { TradeDirection::Buy } else { TradeDirection::Sell };
            trades.push(Trade {
                asset_idx: i,
                asset_name: current.asset_names.get(i).cloned().unwrap_or_default(),
                direction, weight_change, dollar_amount: dollar_trade * weight_change.signum(),
                shares: 0.0, estimated_cost: trade_cost,
            });
            total_turnover += weight_change.abs();
            actual_cost += trade_cost;
        }
    }

    RebalanceResult {
        n_trades: trades.len(), trades, new_weights,
        total_turnover, total_cost: actual_cost, triggered: true,
        reason: "Cost-aware partial rebalance".into(),
    }
}

fn portfolio_utility(weights: &[f64], returns: &[f64], cov: &[Vec<f64>], lambda: f64) -> f64 {
    let n = weights.len();
    let ret: f64 = weights.iter().zip(returns.iter()).map(|(w, r)| w * r).sum();
    let mut var = 0.0;
    for i in 0..n {
        for j in 0..n {
            var += weights[i] * weights[j] * cov[i][j];
        }
    }
    ret - 0.5 * lambda * var
}

// ═══════════════════════════════════════════════════════════════════════════
// MULTI-PERIOD LOOKAHEAD REBALANCING
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MultiPeriodRebalancer {
    pub n_periods: usize,
    pub period_days: u32,
    pub cost_model: TransactionCostModel,
    pub covariance: Vec<Vec<f64>>,
    pub expected_returns: Vec<f64>,
    pub discount_rate: f64,
}

impl MultiPeriodRebalancer {
    /// Approximate multi-period optimal rebalance using backward induction.
    pub fn optimal_trade(
        &self,
        current: &PortfolioState,
        target: &[f64],
    ) -> RebalanceResult {
        let n = current.n_assets();

        // For multi-period, we want to trade less now if:
        // 1. Expected future drift will naturally move us toward target
        // 2. Trading costs compound over multiple rebalances

        // Compute expected drift over one period
        let dt = self.period_days as f64 / 252.0;
        let expected_drift: Vec<f64> = (0..n).map(|i| {
            self.expected_returns[i] * dt
        }).collect();

        // Adjust target: account for expected drift
        let mut adjusted_target = target.to_vec();
        for i in 0..n {
            // If the asset is expected to appreciate, we need less of it now
            let drift_effect = expected_drift[i] * current.weights[i];
            adjusted_target[i] -= drift_effect * 0.5; // partial adjustment
        }

        // Normalize adjusted target
        let sum: f64 = adjusted_target.iter().sum();
        if sum > 0.0 {
            for w in adjusted_target.iter_mut() { *w /= sum; }
        }

        // Multi-period cost discount: trade fraction now
        let trade_fraction = 1.0 / self.n_periods as f64;
        let patience_factor = 1.0 - trade_fraction;

        // Generate trades with patience factor
        let mut trades = Vec::new();
        let mut total_turnover = 0.0;
        let mut total_cost = 0.0;

        for i in 0..n {
            let full_change = adjusted_target[i] - current.weights[i];
            let patient_change = full_change * (1.0 - patience_factor);

            if patient_change.abs() < 1e-6 { continue; }

            let dollar_amount = patient_change * current.total_value;
            let cost = self.cost_model.trade_cost(dollar_amount, i);
            let direction = if patient_change > 0.0 { TradeDirection::Buy } else { TradeDirection::Sell };

            trades.push(Trade {
                asset_idx: i,
                asset_name: current.asset_names.get(i).cloned().unwrap_or_default(),
                direction, weight_change: patient_change, dollar_amount,
                shares: 0.0, estimated_cost: cost,
            });
            total_turnover += patient_change.abs();
            total_cost += cost;
        }

        let new_weights: Vec<f64> = (0..n).map(|i| {
            let trade = trades.iter().find(|t| t.asset_idx == i);
            current.weights[i] + trade.map(|t| t.weight_change).unwrap_or(0.0)
        }).collect();

        RebalanceResult {
            n_trades: trades.len(), trades, new_weights,
            total_turnover, total_cost, triggered: true,
            reason: format!("Multi-period lookahead ({} periods)", self.n_periods),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// REBALANCING BACKTEST
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RebalanceBacktestResult {
    pub dates: Vec<u64>,
    pub portfolio_values: Vec<f64>,
    pub weights_history: Vec<Vec<f64>>,
    pub rebalance_dates: Vec<u64>,
    pub total_turnover: f64,
    pub total_costs: f64,
    pub n_rebalances: usize,
    pub annualized_return: f64,
    pub annualized_vol: f64,
    pub max_drawdown: f64,
}

/// Backtest a rebalancing strategy over historical returns.
pub fn backtest_rebalance(
    initial_weights: &[f64],
    initial_value: f64,
    daily_returns: &[Vec<f64>],  // [n_days][n_assets]
    target_weights: &[f64],
    threshold: f64,
    cost_model: &TransactionCostModel,
) -> RebalanceBacktestResult {
    let n_days = daily_returns.len();
    let n = initial_weights.len();

    let rebalancer = ThresholdRebalancer::new(threshold, cost_model.clone());

    let mut weights = initial_weights.to_vec();
    let mut value = initial_value;
    let mut values = vec![value];
    let mut weights_hist = vec![weights.clone()];
    let mut rebalance_dates = Vec::new();
    let mut total_turnover = 0.0;
    let mut total_costs = 0.0;
    let mut n_rebalances = 0;
    let mut peak = value;
    let mut max_dd = 0.0;

    for day in 0..n_days {
        // Apply returns
        let mut new_values = vec![0.0; n];
        for i in 0..n {
            new_values[i] = weights[i] * value * (1.0 + daily_returns[day][i]);
        }
        value = new_values.iter().sum::<f64>();
        for i in 0..n {
            weights[i] = new_values[i] / value;
        }

        // Check rebalancing
        let state = PortfolioState {
            asset_names: (0..n).map(|i| format!("Asset{}", i)).collect(),
            weights: weights.clone(),
            market_values: new_values,
            total_value: value,
            cash: 0.0,
        };

        let result = rebalancer.rebalance(&state, target_weights);
        if result.triggered {
            weights = result.new_weights;
            total_turnover += result.total_turnover;
            total_costs += result.total_cost;
            value -= result.total_cost; // deduct costs
            n_rebalances += 1;
            rebalance_dates.push(day as u64);
        }

        // Track drawdown
        if value > peak { peak = value; }
        let dd = (peak - value) / peak;
        if dd > max_dd { max_dd = dd; }

        values.push(value);
        weights_hist.push(weights.clone());
    }

    let total_return = value / initial_value - 1.0;
    let years = n_days as f64 / 252.0;
    let ann_return = (1.0 + total_return).powf(1.0 / years) - 1.0;

    // Compute annualized vol from daily portfolio returns
    let mut daily_rets = Vec::new();
    for i in 1..values.len() {
        daily_rets.push(values[i] / values[i - 1] - 1.0);
    }
    let mean_ret = daily_rets.iter().sum::<f64>() / daily_rets.len() as f64;
    let var: f64 = daily_rets.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
        / (daily_rets.len() - 1) as f64;
    let ann_vol = var.sqrt() * 252.0_f64.sqrt();

    RebalanceBacktestResult {
        dates: (0..=n_days as u64).collect(),
        portfolio_values: values,
        weights_history: weights_hist,
        rebalance_dates,
        total_turnover,
        total_costs,
        n_rebalances,
        annualized_return: ann_return,
        annualized_vol: ann_vol,
        max_drawdown: max_dd,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_no_trigger() {
        let rebalancer = ThresholdRebalancer::new(0.05, TransactionCostModel::zero_cost());
        let state = PortfolioState::new(
            vec!["A".into(), "B".into()],
            vec![5100.0, 4900.0],
            0.0,
        );
        let target = vec![0.50, 0.50];
        let result = rebalancer.rebalance(&state, &target);
        assert!(!result.triggered, "Should not trigger: drift = {}", state.max_drift(&target));
    }

    #[test]
    fn test_threshold_trigger() {
        let rebalancer = ThresholdRebalancer::new(0.05, TransactionCostModel::zero_cost());
        let state = PortfolioState::new(
            vec!["A".into(), "B".into()],
            vec![6000.0, 4000.0],
            0.0,
        );
        let target = vec![0.50, 0.50];
        let result = rebalancer.rebalance(&state, &target);
        assert!(result.triggered);
        assert!(result.n_trades > 0);
    }

    #[test]
    fn test_calendar_rebalancer() {
        let mut rebalancer = CalendarRebalancer::new(
            CalendarFrequency::Monthly,
            TransactionCostModel::zero_cost(),
        );
        for _ in 0..29 {
            rebalancer.advance_day();
        }
        assert!(!rebalancer.should_rebalance());
        rebalancer.advance_day();
        assert!(rebalancer.should_rebalance());
    }
}
