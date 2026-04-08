// drawdown.rs — Maximum drawdown, duration, CDaR, recovery, attribution, Kelly

use quant_math::statistics;

/// Compute drawdown series from returns
pub fn drawdown_series(returns: &[f64]) -> Vec<f64> {
    let n = returns.len();
    let mut dd = Vec::with_capacity(n);
    let mut peak = 1.0;
    let mut value = 1.0;
    for i in 0..n {
        value *= 1.0 + returns[i];
        if value > peak { peak = value; }
        dd.push((peak - value) / peak);
    }
    dd
}

/// Compute drawdown series from price levels
pub fn drawdown_series_prices(prices: &[f64]) -> Vec<f64> {
    let n = prices.len();
    let mut dd = Vec::with_capacity(n);
    let mut peak = prices[0];
    for i in 0..n {
        if prices[i] > peak { peak = prices[i]; }
        dd.push((peak - prices[i]) / peak);
    }
    dd
}

/// Maximum drawdown
pub fn max_drawdown(returns: &[f64]) -> f64 {
    let dd = drawdown_series(returns);
    dd.iter().cloned().fold(0.0_f64, f64::max)
}

/// Maximum drawdown from prices
pub fn max_drawdown_prices(prices: &[f64]) -> f64 {
    let dd = drawdown_series_prices(prices);
    dd.iter().cloned().fold(0.0_f64, f64::max)
}

/// Maximum drawdown with start/end indices
pub fn max_drawdown_period(returns: &[f64]) -> (f64, usize, usize, usize) {
    let n = returns.len();
    let mut value = 1.0;
    let mut peak = 1.0;
    let mut peak_idx = 0;
    let mut max_dd = 0.0;
    let mut dd_start = 0;
    let mut dd_end = 0;

    for i in 0..n {
        value *= 1.0 + returns[i];
        if value > peak {
            peak = value;
            peak_idx = i;
        }
        let dd = (peak - value) / peak;
        if dd > max_dd {
            max_dd = dd;
            dd_start = peak_idx;
            dd_end = i;
        }
    }

    // Find recovery point
    let mut recovery_idx = dd_end;
    let mut v = 1.0;
    for i in 0..n { v *= 1.0 + returns[i]; }
    let mut v2 = 1.0;
    for i in 0..=dd_start { v2 *= 1.0 + returns[i]; }
    let peak_value = v2;
    let mut val = 1.0;
    for i in 0..n {
        val *= 1.0 + returns[i];
        if i > dd_end && val >= peak_value {
            recovery_idx = i;
            break;
        }
    }

    (max_dd, dd_start, dd_end, recovery_idx)
}

/// All drawdown periods
pub fn drawdown_periods(returns: &[f64]) -> Vec<DrawdownPeriod> {
    let n = returns.len();
    let dd = drawdown_series(returns);
    let mut periods = Vec::new();
    let mut in_dd = false;
    let mut start = 0;
    let mut peak_dd = 0.0;
    let mut trough_idx = 0;

    for i in 0..n {
        if dd[i] > 0.0 {
            if !in_dd {
                in_dd = true;
                start = i;
                peak_dd = dd[i];
                trough_idx = i;
            } else {
                if dd[i] > peak_dd {
                    peak_dd = dd[i];
                    trough_idx = i;
                }
            }
        } else if in_dd {
            periods.push(DrawdownPeriod {
                start,
                trough: trough_idx,
                recovery: i,
                depth: peak_dd,
                duration: i - start,
                drawdown_duration: trough_idx - start,
                recovery_duration: i - trough_idx,
            });
            in_dd = false;
        }
    }
    // Still in drawdown at end
    if in_dd {
        periods.push(DrawdownPeriod {
            start,
            trough: trough_idx,
            recovery: n,
            depth: peak_dd,
            duration: n - start,
            drawdown_duration: trough_idx - start,
            recovery_duration: n - trough_idx,
        });
    }
    periods
}

#[derive(Clone, Debug)]
pub struct DrawdownPeriod {
    pub start: usize,
    pub trough: usize,
    pub recovery: usize,
    pub depth: f64,
    pub duration: usize,
    pub drawdown_duration: usize,
    pub recovery_duration: usize,
}

/// Average drawdown
pub fn average_drawdown(returns: &[f64]) -> f64 {
    let dd = drawdown_series(returns);
    let positive: Vec<f64> = dd.iter().filter(|&&d| d > 0.0).copied().collect();
    if positive.is_empty() { return 0.0; }
    statistics::mean(&positive)
}

/// Conditional Drawdown at Risk (CDaR)
/// The average of the worst α% drawdowns
pub fn cdar(returns: &[f64], alpha: f64) -> f64 {
    let dd = drawdown_series(returns);
    let mut sorted: Vec<f64> = dd.iter().filter(|&&d| d > 0.0).copied().collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = (alpha * sorted.len() as f64).ceil() as usize;
    let cutoff = cutoff.max(1).min(sorted.len());
    sorted[..cutoff].iter().sum::<f64>() / cutoff as f64
}

/// Maximum drawdown duration (in periods)
pub fn max_drawdown_duration(returns: &[f64]) -> usize {
    let dd = drawdown_series(returns);
    let mut max_dur = 0;
    let mut current_dur = 0;
    for d in &dd {
        if *d > 0.0 {
            current_dur += 1;
            max_dur = max_dur.max(current_dur);
        } else {
            current_dur = 0;
        }
    }
    max_dur
}

/// Calmar ratio: annualized return / max drawdown
pub fn calmar_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    let n = returns.len() as f64;
    let total_return: f64 = returns.iter().map(|r| (1.0 + r).ln()).sum::<f64>();
    let annual_return = (total_return / n * periods_per_year).exp() - 1.0;
    let mdd = max_drawdown(returns);
    if mdd > 1e-15 { annual_return / mdd } else { f64::INFINITY }
}

/// Sterling ratio: annualized return / (average drawdown + 10%)
pub fn sterling_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    let n = returns.len() as f64;
    let total_return: f64 = returns.iter().map(|r| (1.0 + r).ln()).sum::<f64>();
    let annual_return = (total_return / n * periods_per_year).exp() - 1.0;
    let avg_dd = average_drawdown(returns);
    let denom = avg_dd + 0.10;
    if denom > 1e-15 { annual_return / denom } else { f64::INFINITY }
}

/// Burke ratio: annualized return / sqrt(sum of squared drawdowns)
pub fn burke_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    let n = returns.len() as f64;
    let total_return: f64 = returns.iter().map(|r| (1.0 + r).ln()).sum::<f64>();
    let annual_return = (total_return / n * periods_per_year).exp() - 1.0;
    let dd = drawdown_series(returns);
    let sum_sq: f64 = dd.iter().map(|d| d * d).sum();
    let denom = (sum_sq / n).sqrt();
    if denom > 1e-15 { annual_return / denom } else { f64::INFINITY }
}

/// Pain index: average drawdown over entire period
pub fn pain_index(returns: &[f64]) -> f64 {
    let dd = drawdown_series(returns);
    statistics::mean(&dd)
}

/// Pain ratio: excess return / pain index
pub fn pain_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    let n = returns.len() as f64;
    let total_return: f64 = returns.iter().map(|r| (1.0 + r).ln()).sum::<f64>();
    let annual_return = (total_return / n * periods_per_year).exp() - 1.0;
    let pi = pain_index(returns);
    if pi > 1e-15 { (annual_return - risk_free_rate) / pi } else { f64::INFINITY }
}

/// Ulcer index: RMS of drawdown
pub fn ulcer_index(returns: &[f64]) -> f64 {
    let dd = drawdown_series(returns);
    let rms = (dd.iter().map(|d| d * d).sum::<f64>() / dd.len() as f64).sqrt();
    rms
}

/// Martin ratio (Ulcer performance index): excess return / ulcer index
pub fn martin_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    let n = returns.len() as f64;
    let total_return: f64 = returns.iter().map(|r| (1.0 + r).ln()).sum::<f64>();
    let annual_return = (total_return / n * periods_per_year).exp() - 1.0;
    let ui = ulcer_index(returns);
    if ui > 1e-15 { (annual_return - risk_free_rate) / ui } else { f64::INFINITY }
}

/// Drawdown-based Kelly criterion
pub fn kelly_from_drawdown(win_rate: f64, avg_win: f64, avg_loss: f64, max_dd_target: f64) -> f64 {
    // Kelly fraction
    let kelly = win_rate / avg_loss - (1.0 - win_rate) / avg_win;
    // Adjust for max drawdown target
    // Rule of thumb: max_dd ≈ 2 * kelly * volatility
    let adjusted = kelly * max_dd_target / 0.20; // assuming 20% base max dd at full Kelly
    adjusted.clamp(0.0, 1.0)
}

/// Optimal f via maximizing terminal wealth with drawdown constraint
pub fn optimal_f_constrained(returns: &[f64], max_dd_target: f64) -> f64 {
    let mut best_f = 0.0;
    let mut best_growth = f64::NEG_INFINITY;

    for f_pct in 1..=100 {
        let f = f_pct as f64 / 100.0;
        let mut value = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;
        let mut log_growth = 0.0;

        for &r in returns {
            value *= 1.0 + f * r;
            if value <= 0.0 { break; }
            log_growth += (1.0 + f * r).ln();
            if value > peak { peak = value; }
            let dd = (peak - value) / peak;
            if dd > max_dd { max_dd = dd; }
        }

        if max_dd <= max_dd_target && log_growth > best_growth {
            best_growth = log_growth;
            best_f = f;
        }
    }
    best_f
}

/// Drawdown-based deleverage triggers
pub struct DeleverageTrigger {
    pub warning_level: f64,   // e.g., 0.05 = 5% drawdown
    pub reduce_level: f64,    // e.g., 0.10 = 10% drawdown
    pub critical_level: f64,  // e.g., 0.15 = 15% drawdown
    pub target_leverage: f64, // base leverage
}

impl DeleverageTrigger {
    pub fn new(warning: f64, reduce: f64, critical: f64, base_leverage: f64) -> Self {
        Self {
            warning_level: warning,
            reduce_level: reduce,
            critical_level: critical,
            target_leverage: base_leverage,
        }
    }

    /// Compute current leverage given drawdown
    pub fn current_leverage(&self, drawdown: f64) -> f64 {
        if drawdown >= self.critical_level {
            0.0 // fully deleveraged
        } else if drawdown >= self.reduce_level {
            let t = (self.critical_level - drawdown) / (self.critical_level - self.reduce_level);
            self.target_leverage * 0.5 * t
        } else if drawdown >= self.warning_level {
            let t = (self.reduce_level - drawdown) / (self.reduce_level - self.warning_level);
            self.target_leverage * (0.5 + 0.5 * t)
        } else {
            self.target_leverage
        }
    }

    /// Simulate strategy with deleverage triggers
    pub fn simulate(&self, returns: &[f64]) -> Vec<f64> {
        let n = returns.len();
        let mut managed_returns = Vec::with_capacity(n);
        let dd = drawdown_series(returns);

        for i in 0..n {
            let leverage = self.current_leverage(if i > 0 { dd[i - 1] } else { 0.0 });
            managed_returns.push(returns[i] * leverage);
        }
        managed_returns
    }
}

/// Drawdown attribution by factor
pub fn drawdown_attribution(
    portfolio_returns: &[f64],
    factor_returns: &[Vec<f64>],
    factor_betas: &[f64],
) -> Vec<f64> {
    let n = portfolio_returns.len();
    let k = factor_returns.len();
    let dd = drawdown_series(portfolio_returns);

    // Find max drawdown period
    let mut max_dd = 0.0;
    let mut max_dd_end = 0;
    for i in 0..n {
        if dd[i] > max_dd { max_dd = dd[i]; max_dd_end = i; }
    }

    // Find start of this drawdown
    let mut dd_start = max_dd_end;
    while dd_start > 0 && dd[dd_start - 1] > 0.0 { dd_start -= 1; }

    // Attribution: cumulative contribution of each factor during drawdown
    let mut contributions = vec![0.0; k];
    for t in dd_start..=max_dd_end {
        for j in 0..k {
            if t < factor_returns[j].len() {
                contributions[j] += factor_betas[j] * factor_returns[j][t];
            }
        }
    }

    // Residual
    let total_factor: f64 = contributions.iter().sum();
    let actual: f64 = portfolio_returns[dd_start..=max_dd_end].iter().sum();
    let residual = actual - total_factor;

    let mut result = contributions;
    result.push(residual);
    result
}

/// Drawdown distribution statistics
pub struct DrawdownStats {
    pub count: usize,
    pub avg_depth: f64,
    pub max_depth: f64,
    pub avg_duration: f64,
    pub max_duration: usize,
    pub avg_recovery: f64,
    pub max_recovery: usize,
}

impl DrawdownStats {
    pub fn compute(returns: &[f64]) -> Self {
        let periods = drawdown_periods(returns);
        let count = periods.len();
        if count == 0 {
            return Self { count: 0, avg_depth: 0.0, max_depth: 0.0,
                avg_duration: 0.0, max_duration: 0, avg_recovery: 0.0, max_recovery: 0 };
        }

        let avg_depth = periods.iter().map(|p| p.depth).sum::<f64>() / count as f64;
        let max_depth = periods.iter().map(|p| p.depth).fold(0.0_f64, f64::max);
        let avg_duration = periods.iter().map(|p| p.duration as f64).sum::<f64>() / count as f64;
        let max_duration = periods.iter().map(|p| p.duration).max().unwrap_or(0);
        let avg_recovery = periods.iter().map(|p| p.recovery_duration as f64).sum::<f64>() / count as f64;
        let max_recovery = periods.iter().map(|p| p.recovery_duration).max().unwrap_or(0);

        Self { count, avg_depth, max_depth, avg_duration, max_duration, avg_recovery, max_recovery }
    }
}

/// Sortino drawdown ratio
pub fn sortino_drawdown(returns: &[f64], target: f64, periods_per_year: f64) -> f64 {
    let n = returns.len() as f64;
    let total_return: f64 = returns.iter().map(|r| (1.0 + r).ln()).sum::<f64>();
    let annual_return = (total_return / n * periods_per_year).exp() - 1.0;
    let downside: f64 = returns.iter()
        .map(|r| if *r < target { (r - target).powi(2) } else { 0.0 })
        .sum::<f64>();
    let downside_dev = (downside / n * periods_per_year).sqrt();
    if downside_dev > 1e-15 { (annual_return - target) / downside_dev } else { f64::INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.10, -0.15, 0.05, -0.20, 0.30, -0.05];
        let mdd = max_drawdown(&returns);
        assert!(mdd > 0.0 && mdd < 1.0);
    }

    #[test]
    fn test_cdar() {
        let returns = vec![0.01, -0.02, -0.03, 0.05, -0.04, 0.02, -0.01];
        let cd = cdar(&returns, 0.05);
        assert!(cd > 0.0);
    }

    #[test]
    fn test_calmar_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.03, -0.005, 0.015, 0.01, -0.02, 0.025, 0.005];
        let cr = calmar_ratio(&returns, 252.0);
        assert!(cr > 0.0);
    }

    #[test]
    fn test_deleverage() {
        let trigger = DeleverageTrigger::new(0.05, 0.10, 0.15, 2.0);
        assert!((trigger.current_leverage(0.0) - 2.0).abs() < 1e-10);
        assert!(trigger.current_leverage(0.15) < 0.01);
    }
}
