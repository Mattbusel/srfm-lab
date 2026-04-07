//! equity_curve.rs -- Equity curve analytics, drawdown series, and rolling Sharpe.
//!
//! All computations run in WASM for client-side chart rendering without server
//! round-trips. Accepts trade history arrays and returns pre-computed analytics
//! arrays ready for direct plotting.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// EquityAnalytics -- batch results struct
// ---------------------------------------------------------------------------

/// Complete equity curve analytics result.
/// All arrays have the same length as the input trade history.
#[wasm_bindgen]
pub struct EquityAnalytics {
    /// Equity curve (normalized: starts at 1.0).
    pub_equity: Vec<f64>,
    /// Drawdown series: 0.0 to -1.0 (e.g. -0.10 = 10% drawdown from peak).
    pub_drawdown: Vec<f64>,
    /// Rolling Sharpe ratio (annualized). NaN during warmup.
    pub_rolling_sharpe: Vec<f64>,
    /// Rolling Sortino ratio (annualized). NaN during warmup.
    pub_rolling_sortino: Vec<f64>,
    /// Running maximum equity (high-water mark).
    pub_hwm: Vec<f64>,
    /// Summary statistics as a flat array.
    /// [total_return, max_drawdown, sharpe, sortino, calmar, win_rate,
    ///  avg_win, avg_loss, profit_factor, n_trades]
    pub_summary: Vec<f64>,
}

#[wasm_bindgen]
impl EquityAnalytics {
    /// Returns the equity curve as Float64Array.
    pub fn equity(&self) -> Vec<f64> {
        self.pub_equity.clone()
    }

    /// Returns the drawdown series as Float64Array.
    pub fn drawdown(&self) -> Vec<f64> {
        self.pub_drawdown.clone()
    }

    /// Returns the rolling Sharpe series as Float64Array.
    pub fn rolling_sharpe(&self) -> Vec<f64> {
        self.pub_rolling_sharpe.clone()
    }

    /// Returns the rolling Sortino series as Float64Array.
    pub fn rolling_sortino(&self) -> Vec<f64> {
        self.pub_rolling_sortino.clone()
    }

    /// Returns the high-water mark series as Float64Array.
    pub fn high_water_mark(&self) -> Vec<f64> {
        self.pub_hwm.clone()
    }

    /// Returns summary stats array of length 10.
    /// Indices: [total_return, max_drawdown, sharpe, sortino, calmar,
    ///           win_rate, avg_win, avg_loss, profit_factor, n_trades]
    pub fn summary(&self) -> Vec<f64> {
        self.pub_summary.clone()
    }

    /// Returns total return as a fraction (e.g. 0.25 = 25% gain).
    pub fn total_return(&self) -> f64 {
        self.pub_summary[0]
    }

    /// Returns maximum drawdown as a positive fraction (e.g. 0.10 = 10%).
    pub fn max_drawdown(&self) -> f64 {
        self.pub_summary[1]
    }

    /// Returns annualized Sharpe ratio.
    pub fn sharpe(&self) -> f64 {
        self.pub_summary[2]
    }

    /// Returns annualized Sortino ratio.
    pub fn sortino(&self) -> f64 {
        self.pub_summary[3]
    }

    /// Returns Calmar ratio (total_return / max_drawdown).
    pub fn calmar(&self) -> f64 {
        self.pub_summary[4]
    }
}

// ---------------------------------------------------------------------------
// Core computation functions
// ---------------------------------------------------------------------------

/// Compute equity curve from per-bar returns. Returns normalized equity starting at 1.0.
fn compute_equity(returns: &[f64]) -> Vec<f64> {
    let n = returns.len();
    let mut equity = Vec::with_capacity(n + 1);
    equity.push(1.0f64);
    let mut e = 1.0f64;
    for &r in returns {
        e *= 1.0 + r;
        equity.push(e);
    }
    equity
}

/// Compute drawdown series from equity curve. Returns values in (-1, 0].
fn compute_drawdown_series(equity: &[f64]) -> (Vec<f64>, f64) {
    let n = equity.len();
    let mut dd = vec![0.0f64; n];
    let mut hwm = vec![0.0f64; n];
    let mut peak = equity[0];
    let mut max_dd = 0.0f64;

    for i in 0..n {
        if equity[i] > peak {
            peak = equity[i];
        }
        hwm[i] = peak;
        let d = (equity[i] - peak) / peak;
        dd[i] = d;
        if d < max_dd {
            max_dd = d;
        }
    }

    (dd, max_dd.abs())
}

/// Compute rolling Sharpe ratio over a window. Annualized assuming returns are daily.
/// annualization_factor = sqrt(252) for daily, sqrt(52) for weekly, etc.
fn rolling_sharpe(returns: &[f64], window: usize, annualization: f64) -> Vec<f64> {
    let n = returns.len();
    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &returns[i + 1 - window..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 = slice.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (window - 1) as f64;
        let std = var.sqrt();
        if std > 1e-12 {
            result[i] = (mean / std) * annualization;
        } else if mean > 0.0 {
            result[i] = f64::INFINITY;
        } else {
            result[i] = 0.0;
        }
    }
    result
}

/// Compute rolling Sortino ratio (penalizes downside deviation only).
fn rolling_sortino(returns: &[f64], window: usize, annualization: f64) -> Vec<f64> {
    let n = returns.len();
    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &returns[i + 1 - window..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let downside_var: f64 = slice.iter()
            .map(|&r| if r < 0.0 { r * r } else { 0.0 })
            .sum::<f64>() / window as f64;
        let downside_std = downside_var.sqrt();
        if downside_std > 1e-12 {
            result[i] = (mean / downside_std) * annualization;
        } else if mean > 0.0 {
            result[i] = f64::INFINITY;
        } else {
            result[i] = 0.0;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// WASM exports
// ---------------------------------------------------------------------------

/// Compute full equity curve analytics from a per-bar returns array.
/// Returns EquityAnalytics object with all series and summary stats.
/// - returns: per-bar returns (fractional, e.g. 0.01 = 1% gain)
/// - sharpe_window: rolling window for Sharpe/Sortino (e.g. 63 for quarterly)
/// - annualization_factor: sqrt(252) for daily bars = 15.875
#[wasm_bindgen]
pub fn compute_equity_analytics(
    returns: &[f64],
    sharpe_window: u32,
    annualization_factor: f64,
) -> EquityAnalytics {
    let n = returns.len();
    if n == 0 {
        return EquityAnalytics {
            pub_equity:         vec![1.0],
            pub_drawdown:       vec![0.0],
            pub_rolling_sharpe: vec![f64::NAN],
            pub_rolling_sortino:vec![f64::NAN],
            pub_hwm:            vec![1.0],
            pub_summary:        vec![0.0; 10],
        };
    }

    let equity = compute_equity(returns);
    let sw = (sharpe_window as usize).min(n).max(2);
    let (drawdown, max_dd) = compute_drawdown_series(&equity);
    let r_sharpe  = rolling_sharpe(returns, sw, annualization_factor);
    let r_sortino = rolling_sortino(returns, sw, annualization_factor);

    // High-water mark from drawdown computation
    let mut hwm = vec![0.0f64; equity.len()];
    {
        let mut peak = equity[0];
        for i in 0..equity.len() {
            if equity[i] > peak { peak = equity[i]; }
            hwm[i] = peak;
        }
    }

    // Summary statistics
    let total_return = equity.last().copied().unwrap_or(1.0) - 1.0;

    // Full-period Sharpe
    let mean_ret: f64 = returns.iter().sum::<f64>() / n as f64;
    let var_ret: f64  = if n > 1 {
        returns.iter().map(|&r| (r - mean_ret).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let std_ret = var_ret.sqrt();
    let sharpe = if std_ret > 1e-12 {
        (mean_ret / std_ret) * annualization_factor
    } else {
        0.0
    };

    // Full-period Sortino
    let downside_var: f64 = returns.iter()
        .map(|&r| if r < 0.0 { r * r } else { 0.0 })
        .sum::<f64>() / n as f64;
    let downside_std = downside_var.sqrt();
    let sortino = if downside_std > 1e-12 {
        (mean_ret / downside_std) * annualization_factor
    } else if mean_ret > 0.0 { f64::INFINITY } else { 0.0 };

    // Calmar
    let calmar = if max_dd > 1e-12 { total_return / max_dd } else { 0.0 };

    // Trade stats (count positive vs negative returns)
    let wins: Vec<f64>   = returns.iter().filter(|&&r| r > 0.0).copied().collect();
    let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    let n_trades = returns.len() as f64;
    let win_rate = wins.len() as f64 / n_trades;
    let avg_win  = if wins.is_empty()   { 0.0 } else { wins.iter().sum::<f64>()   / wins.len() as f64 };
    let avg_loss = if losses.is_empty() { 0.0 } else { losses.iter().map(|&x| x.abs()).sum::<f64>() / losses.len() as f64 };
    let gross_profit = wins.iter().sum::<f64>();
    let gross_loss   = losses.iter().map(|&x| x.abs()).sum::<f64>();
    let profit_factor = if gross_loss > 1e-12 { gross_profit / gross_loss } else { f64::INFINITY };

    let summary = vec![
        total_return,
        max_dd,
        sharpe,
        sortino,
        calmar,
        win_rate,
        avg_win,
        avg_loss,
        profit_factor,
        n_trades,
    ];

    EquityAnalytics {
        pub_equity:          equity,
        pub_drawdown:        drawdown,
        pub_rolling_sharpe:  r_sharpe,
        pub_rolling_sortino: r_sortino,
        pub_hwm:             hwm,
        pub_summary:         summary,
    }
}

/// Compute equity curve directly from a trade history.
/// Accepts parallel arrays of trade entry prices, exit prices, and position sizes.
/// Returns per-trade returns array (suitable as input to compute_equity_analytics).
#[wasm_bindgen]
pub fn trade_returns_from_history(
    entry_prices: &[f64],
    exit_prices: &[f64],
    position_sizes: &[f64],
    is_long: &[u8],
) -> Vec<f64> {
    let n = entry_prices.len().min(exit_prices.len()).min(position_sizes.len()).min(is_long.len());
    let mut returns = Vec::with_capacity(n);

    for i in 0..n {
        let entry  = entry_prices[i];
        let exit_p = exit_prices[i];
        let size   = position_sizes[i];
        if entry.abs() < 1e-12 {
            returns.push(0.0);
            continue;
        }
        let raw_ret = if is_long[i] != 0 {
            (exit_p - entry) / entry
        } else {
            (entry - exit_p) / entry
        };
        returns.push(raw_ret * size);
    }
    returns
}

/// Compute running maximum drawdown series (the worst drawdown seen so far at each bar).
/// Useful for drawdown progress charts.
#[wasm_bindgen]
pub fn compute_running_max_drawdown(returns: &[f64]) -> Vec<f64> {
    let equity = compute_equity(returns);
    let n = equity.len();
    let mut peak = equity[0];
    let mut max_dd = 0.0f64;
    let mut result = vec![0.0f64; n];

    for i in 0..n {
        if equity[i] > peak {
            peak = equity[i];
        }
        let dd = (equity[i] - peak) / peak;
        if dd < -max_dd {
            max_dd = -dd;
        }
        result[i] = max_dd;
    }
    result
}

/// Compute underwater equity (equity divided by its running maximum).
/// A value of 0.9 means the portfolio is 10% below its peak at that point.
#[wasm_bindgen]
pub fn compute_underwater_equity(returns: &[f64]) -> Vec<f64> {
    let equity = compute_equity(returns);
    let n = equity.len();
    let mut peak = equity[0];
    let mut result = vec![1.0f64; n];

    for i in 0..n {
        if equity[i] > peak {
            peak = equity[i];
        }
        result[i] = if peak > 1e-12 { equity[i] / peak } else { 1.0 };
    }
    result
}

/// Compute rolling CAGR (compound annual growth rate) over a rolling window.
/// assumes bars are in daily frequency (252 trading days/year).
#[wasm_bindgen]
pub fn compute_rolling_cagr(returns: &[f64], window: u32, bars_per_year: f64) -> Vec<f64> {
    let n = returns.len();
    let w = window as usize;
    let mut result = vec![f64::NAN; n];

    for i in (w - 1)..n {
        let slice = &returns[i + 1 - w..=i];
        let period_return: f64 = slice.iter().fold(1.0, |acc, &r| acc * (1.0 + r));
        let years = w as f64 / bars_per_year;
        if years > 0.0 {
            result[i] = period_return.powf(1.0 / years) - 1.0;
        }
    }
    result
}
