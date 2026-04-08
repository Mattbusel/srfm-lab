// liquidity.rs — Liquidity risk: Amihud, bid-ask, market impact, L-VaR, liquidity scoring

use quant_math::statistics;

/// Amihud illiquidity ratio: |return| / dollar volume
pub fn amihud_illiquidity(returns: &[f64], volumes: &[f64]) -> f64 {
    assert_eq!(returns.len(), volumes.len());
    let n = returns.len();
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..n {
        if volumes[i] > 1e-10 {
            sum += returns[i].abs() / volumes[i];
            count += 1;
        }
    }
    if count > 0 { sum / count as f64 } else { 0.0 }
}

/// Rolling Amihud illiquidity
pub fn rolling_amihud(returns: &[f64], volumes: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        if i + 1 < window {
            result.push(f64::NAN);
        } else {
            let start = i + 1 - window;
            result.push(amihud_illiquidity(&returns[start..=i], &volumes[start..=i]));
        }
    }
    result
}

/// Bid-ask spread (quoted)
pub fn quoted_spread(bid: f64, ask: f64) -> f64 {
    ask - bid
}

/// Relative spread (bid-ask / midpoint)
pub fn relative_spread(bid: f64, ask: f64) -> f64 {
    let mid = 0.5 * (bid + ask);
    if mid > 1e-15 { (ask - bid) / mid } else { 0.0 }
}

/// Effective spread from trade data
pub fn effective_spread(trade_price: f64, midpoint: f64) -> f64 {
    2.0 * (trade_price - midpoint).abs()
}

/// Average effective spread
pub fn avg_effective_spread(trades: &[(f64, f64)]) -> f64 {
    // trades: (trade_price, midpoint)
    if trades.is_empty() { return 0.0; }
    let sum: f64 = trades.iter().map(|&(tp, mp)| effective_spread(tp, mp)).sum();
    sum / trades.len() as f64
}

/// Realized spread (5-minute horizon)
pub fn realized_spread(trade_price: f64, midpoint_after: f64, trade_sign: f64) -> f64 {
    2.0 * trade_sign * (trade_price - midpoint_after)
}

/// Price impact component
pub fn price_impact(midpoint_before: f64, midpoint_after: f64, trade_sign: f64) -> f64 {
    2.0 * trade_sign * (midpoint_after - midpoint_before)
}

/// Roll's spread estimator from serial covariance of returns
pub fn roll_spread(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 3 { return 0.0; }
    let mut cov = 0.0;
    let mean = statistics::mean(returns);
    for i in 1..n {
        cov += (returns[i] - mean) * (returns[i - 1] - mean);
    }
    cov /= (n - 1) as f64;
    if cov >= 0.0 { return 0.0; } // Non-negative covariance implies no spread
    2.0 * (-cov).sqrt()
}

/// Corwin-Schultz spread estimator from high/low prices
pub fn corwin_schultz_spread(highs: &[f64], lows: &[f64]) -> Vec<f64> {
    let n = highs.len();
    assert_eq!(n, lows.len());
    let mut spreads = Vec::with_capacity(n);

    for i in 1..n {
        let beta = (highs[i].max(highs[i-1]) / lows[i].min(lows[i-1])).ln().powi(2);
        let gamma = (highs[i] / lows[i]).ln().powi(2) + (highs[i-1] / lows[i-1]).ln().powi(2);

        let alpha_num = (2.0_f64.sqrt() - 1.0) * beta.sqrt() - gamma.sqrt();
        let alpha_den = 3.0 - 2.0 * 2.0_f64.sqrt();

        let alpha = if alpha_den.abs() > 1e-15 { alpha_num / alpha_den } else { 0.0 };
        let spread = 2.0 * (alpha.exp() - 1.0) / (1.0 + alpha.exp());
        spreads.push(spread.max(0.0));
    }
    spreads
}

/// Kyle's lambda (market impact coefficient)
/// Estimated from regression: ΔP = λ * signed_volume + ε
pub fn kyle_lambda(price_changes: &[f64], signed_volumes: &[f64]) -> f64 {
    assert_eq!(price_changes.len(), signed_volumes.len());
    let n = price_changes.len();
    if n < 2 { return 0.0; }

    let sum_xy: f64 = price_changes.iter().zip(signed_volumes).map(|(p, v)| p * v).sum();
    let sum_x2: f64 = signed_volumes.iter().map(|v| v * v).sum();
    if sum_x2 < 1e-30 { return 0.0; }
    sum_xy / sum_x2
}

/// Square-root market impact model: ΔP/P = η * σ * sqrt(Q/V)
pub fn sqrt_market_impact(
    volatility: f64, quantity: f64, avg_daily_volume: f64, eta: f64,
) -> f64 {
    if avg_daily_volume < 1e-10 { return 0.0; }
    eta * volatility * (quantity / avg_daily_volume).sqrt()
}

/// Almgren-Chriss optimal execution cost
pub fn almgren_chriss_cost(
    shares: f64, total_time: f64, volatility: f64,
    permanent_impact: f64, temporary_impact: f64,
    risk_aversion: f64,
) -> f64 {
    let n = shares;
    let sigma = volatility;
    let gamma = permanent_impact;
    let eta = temporary_impact;
    let lambda = risk_aversion;

    // Kappa
    let kappa = (lambda * sigma * sigma / eta).sqrt();
    let sinh_kt = (kappa * total_time).sinh();
    let cosh_kt = (kappa * total_time).cosh();

    if sinh_kt.abs() < 1e-15 { return 0.0; }

    // Expected cost
    let permanent_cost = 0.5 * gamma * n * n;
    let temp_cost = eta * n * n * (kappa / (2.0 * (kappa * total_time).tanh()));

    permanent_cost + temp_cost
}

/// Days to liquidate a position
pub fn days_to_liquidate(
    position_value: f64, avg_daily_volume: f64,
    max_participation_rate: f64,
) -> f64 {
    if avg_daily_volume * max_participation_rate < 1e-10 { return f64::INFINITY; }
    position_value / (avg_daily_volume * max_participation_rate)
}

/// Liquidation cost (total)
pub fn liquidation_cost(
    position_value: f64, days_to_liq: f64,
    spread: f64, volatility: f64, market_impact_eta: f64,
) -> f64 {
    let spread_cost = 0.5 * spread * position_value;
    let vol_cost = volatility * position_value * (days_to_liq / 252.0).sqrt();
    let impact_cost = market_impact_eta * position_value * (1.0 / days_to_liq).sqrt();
    spread_cost + vol_cost + impact_cost
}

/// Liquidity-adjusted VaR (L-VaR)
pub fn liquidity_adjusted_var(
    var: f64, position_value: f64,
    bid_ask_spread: f64, spread_volatility: f64,
    confidence_z: f64,
) -> f64 {
    let mid_spread = 0.5 * bid_ask_spread * position_value;
    let spread_risk = 0.5 * confidence_z * spread_volatility * position_value;
    var + mid_spread + spread_risk
}

/// Exogenous L-VaR (Bangia et al. 1999)
pub fn bangia_lvar(
    var: f64, position_value: f64,
    mean_spread: f64, spread_std: f64,
    confidence_z: f64,
) -> f64 {
    let lc = 0.5 * position_value * (mean_spread + confidence_z * spread_std);
    var + lc
}

/// Endogenous L-VaR (considers market impact from unwinding)
pub fn endogenous_lvar(
    var: f64, position_value: f64,
    avg_daily_volume: f64, volatility: f64,
    participation_rate: f64,
) -> f64 {
    let days = position_value / (avg_daily_volume * participation_rate);
    let impact = sqrt_market_impact(volatility, position_value, avg_daily_volume, 0.5);
    let time_risk = volatility * position_value * (days / 252.0).sqrt();
    var + impact * position_value + time_risk
}

/// Liquidity score (composite, 0-100)
pub struct LiquidityScore {
    pub total_score: f64,
    pub volume_score: f64,
    pub spread_score: f64,
    pub impact_score: f64,
    pub depth_score: f64,
}

impl LiquidityScore {
    pub fn compute(
        avg_daily_volume: f64,
        bid_ask_spread: f64,
        market_impact: f64,
        avg_trade_size: f64,
    ) -> Self {
        // Volume score: higher is better, log scale
        let volume_score = (100.0 * (1.0 - (-avg_daily_volume / 1e7).exp())).clamp(0.0, 100.0);

        // Spread score: lower is better
        let spread_score = (100.0 * (1.0 - bid_ask_spread * 100.0)).clamp(0.0, 100.0);

        // Impact score: lower is better
        let impact_score = (100.0 * (1.0 - market_impact * 50.0)).clamp(0.0, 100.0);

        // Depth score
        let depth_score = (100.0 * (1.0 - (-avg_trade_size / 1e5).exp())).clamp(0.0, 100.0);

        let total_score = 0.3 * volume_score + 0.25 * spread_score + 0.25 * impact_score + 0.2 * depth_score;

        Self { total_score, volume_score, spread_score, impact_score, depth_score }
    }

    pub fn category(&self) -> &'static str {
        if self.total_score >= 80.0 { "Highly Liquid" }
        else if self.total_score >= 60.0 { "Liquid" }
        else if self.total_score >= 40.0 { "Moderately Liquid" }
        else if self.total_score >= 20.0 { "Illiquid" }
        else { "Highly Illiquid" }
    }
}

/// Dark pool analytics
pub struct DarkPoolMetrics {
    pub dark_volume_ratio: f64,
    pub price_improvement: f64,
    pub fill_rate: f64,
    pub information_leakage: f64,
}

impl DarkPoolMetrics {
    pub fn compute(
        dark_volume: f64, total_volume: f64,
        dark_avg_price: f64, lit_avg_price: f64,
        orders_submitted: usize, orders_filled: usize,
        pre_trade_vol: f64, post_trade_vol: f64,
    ) -> Self {
        let dark_volume_ratio = if total_volume > 0.0 { dark_volume / total_volume } else { 0.0 };
        let price_improvement = if lit_avg_price > 1e-15 {
            (lit_avg_price - dark_avg_price) / lit_avg_price
        } else { 0.0 };
        let fill_rate = if orders_submitted > 0 {
            orders_filled as f64 / orders_submitted as f64
        } else { 0.0 };
        let information_leakage = if pre_trade_vol > 1e-15 {
            (post_trade_vol - pre_trade_vol) / pre_trade_vol
        } else { 0.0 };

        Self { dark_volume_ratio, price_improvement, fill_rate, information_leakage }
    }
}

/// Intraday liquidity pattern analysis
pub struct IntradayLiquidity {
    pub hourly_volume: Vec<f64>,
    pub hourly_spread: Vec<f64>,
    pub hourly_volatility: Vec<f64>,
}

impl IntradayLiquidity {
    pub fn from_ticks(timestamps_hour: &[usize], volumes: &[f64], spreads: &[f64], returns: &[f64]) -> Self {
        let mut hourly_volume = vec![0.0; 24];
        let mut hourly_spread_sum = vec![0.0; 24];
        let mut hourly_spread_count = vec![0usize; 24];
        let mut hourly_returns: Vec<Vec<f64>> = (0..24).map(|_| Vec::new()).collect();

        for i in 0..timestamps_hour.len() {
            let h = timestamps_hour[i] % 24;
            hourly_volume[h] += volumes[i];
            if i < spreads.len() {
                hourly_spread_sum[h] += spreads[i];
                hourly_spread_count[h] += 1;
            }
            if i < returns.len() {
                hourly_returns[h].push(returns[i]);
            }
        }

        let hourly_spread: Vec<f64> = (0..24).map(|h| {
            if hourly_spread_count[h] > 0 { hourly_spread_sum[h] / hourly_spread_count[h] as f64 }
            else { 0.0 }
        }).collect();

        let hourly_volatility: Vec<f64> = hourly_returns.iter().map(|r| {
            if r.len() > 1 { statistics::std_dev(r) } else { 0.0 }
        }).collect();

        Self { hourly_volume, hourly_spread, hourly_volatility }
    }

    /// Best trading hours (lowest spread, highest volume)
    pub fn best_hours(&self, top_n: usize) -> Vec<usize> {
        let mut scores: Vec<(usize, f64)> = (0..24).map(|h| {
            let vol_norm = self.hourly_volume[h] / self.hourly_volume.iter().cloned().fold(1e-15, f64::max);
            let spread_norm = if self.hourly_spread[h] > 0.0 {
                1.0 / self.hourly_spread[h]
            } else { 0.0 };
            (h, vol_norm + spread_norm * 0.5)
        }).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.iter().take(top_n).map(|&(h, _)| h).collect()
    }
}

/// VWAP (Volume-Weighted Average Price) calculation
pub fn vwap(prices: &[f64], volumes: &[f64]) -> f64 {
    assert_eq!(prices.len(), volumes.len());
    let pv_sum: f64 = prices.iter().zip(volumes).map(|(p, v)| p * v).sum();
    let v_sum: f64 = volumes.iter().sum();
    if v_sum > 1e-15 { pv_sum / v_sum } else { 0.0 }
}

/// TWAP (Time-Weighted Average Price)
pub fn twap(prices: &[f64]) -> f64 {
    statistics::mean(prices)
}

/// Participation-weighted implementation shortfall
pub fn implementation_shortfall(
    decision_price: f64, execution_prices: &[f64], execution_quantities: &[f64],
    total_quantity: f64,
) -> f64 {
    let avg_exec = vwap(execution_prices, execution_quantities);
    let executed: f64 = execution_quantities.iter().sum();
    let unexecuted = total_quantity - executed;

    let explicit_cost = (avg_exec - decision_price) * executed;
    // Opportunity cost of unexecuted portion using last price
    let last_price = *execution_prices.last().unwrap_or(&decision_price);
    let opportunity_cost = (last_price - decision_price) * unexecuted;

    (explicit_cost + opportunity_cost) / (decision_price * total_quantity)
}

/// Turnover ratio
pub fn turnover_ratio(traded_value: f64, portfolio_value: f64) -> f64 {
    if portfolio_value > 1e-15 { traded_value / portfolio_value } else { 0.0 }
}

/// Liquidity coverage ratio (LCR)
pub fn lcr(hqla: f64, net_cash_outflows_30d: f64) -> f64 {
    if net_cash_outflows_30d > 1e-15 { hqla / net_cash_outflows_30d } else { f64::INFINITY }
}

/// Net stable funding ratio (NSFR)
pub fn nsfr(available_stable_funding: f64, required_stable_funding: f64) -> f64 {
    if required_stable_funding > 1e-15 {
        available_stable_funding / required_stable_funding
    } else { f64::INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amihud() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.01];
        let volumes = vec![1e6, 1.5e6, 0.8e6, 1.2e6, 1e6];
        let illiq = amihud_illiquidity(&returns, &volumes);
        assert!(illiq > 0.0);
    }

    #[test]
    fn test_roll_spread() {
        // Negative autocorrelation should give positive spread
        let returns = vec![0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01];
        let spread = roll_spread(&returns);
        assert!(spread > 0.0);
    }

    #[test]
    fn test_lvar() {
        let var = 100_000.0;
        let lvar = liquidity_adjusted_var(var, 5_000_000.0, 0.001, 0.0005, 2.33);
        assert!(lvar > var);
    }

    #[test]
    fn test_liquidity_score() {
        let score = LiquidityScore::compute(5e6, 0.001, 0.0001, 50000.0);
        assert!(score.total_score > 0.0 && score.total_score <= 100.0);
    }
}
