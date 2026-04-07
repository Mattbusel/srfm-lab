// report_generator.rs -- generates HTML and markdown backtest reports for SRFM
// Includes ASCII sparklines, monthly heatmaps, and performance metrics tables.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Trade record ───────────────────────────────────────────────────────────

/// A single completed trade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl: f64,
    pub direction: String, // "long" or "short"
    pub entry_ts: i64,
    pub exit_ts: i64,
    pub bars_held: u32,
}

impl Trade {
    /// Return in percent.
    pub fn return_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            (self.exit_price - self.entry_price) / self.entry_price * 100.0
        } else {
            0.0
        }
    }

    pub fn is_winner(&self) -> bool {
        self.pnl > 0.0
    }
}

// ── Performance metrics ────────────────────────────────────────────────────

/// Aggregated performance statistics for a backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub max_dd: f64,
    pub win_rate: f64,
    pub avg_win_loss_ratio: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub avg_bars_held: f64,
    pub best_trade_pct: f64,
    pub worst_trade_pct: f64,
}

impl PerformanceMetrics {
    /// Compute metrics from a slice of trades and an equity curve.
    pub fn compute(trades: &[Trade], equity_curve: &[f64], bars_per_year: f64) -> Self {
        let n = trades.len();
        if n == 0 || equity_curve.is_empty() {
            return Self::zero();
        }

        let initial = equity_curve[0];
        let final_eq = *equity_curve.last().unwrap();
        let total_return = (final_eq - initial) / initial;
        let n_bars = equity_curve.len() as f64;
        let years = n_bars / bars_per_year;
        let annualized_return = if years > 0.0 && initial > 0.0 {
            (final_eq / initial).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Max drawdown.
        let max_dd = {
            let mut peak = equity_curve[0];
            let mut max_dd = 0.0f64;
            for &e in equity_curve {
                if e > peak { peak = e; }
                let dd = (peak - e) / peak;
                if dd > max_dd { max_dd = dd; }
            }
            max_dd
        };

        // Sharpe from bar returns.
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        let sharpe = compute_sharpe(&returns, bars_per_year);
        let sortino = compute_sortino(&returns, bars_per_year);
        let calmar = if max_dd > 1e-9 { annualized_return / max_dd } else { 0.0 };

        // Trade stats.
        let winners: Vec<&Trade> = trades.iter().filter(|t| t.is_winner()).collect();
        let losers: Vec<&Trade> = trades.iter().filter(|t| !t.is_winner()).collect();
        let win_rate = winners.len() as f64 / n as f64;

        let avg_win = if !winners.is_empty() {
            winners.iter().map(|t| t.pnl).sum::<f64>() / winners.len() as f64
        } else {
            0.0
        };
        let avg_loss = if !losers.is_empty() {
            losers.iter().map(|t| t.pnl.abs()).sum::<f64>() / losers.len() as f64
        } else {
            1.0
        };
        let avg_win_loss_ratio = avg_win / avg_loss.max(1e-12);

        let gross_profit: f64 = winners.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losers.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 1e-12 { gross_profit / gross_loss } else { f64::INFINITY };

        let avg_bars_held = trades.iter().map(|t| t.bars_held as f64).sum::<f64>() / n as f64;

        let returns_pct: Vec<f64> = trades.iter().map(|t| t.return_pct()).collect();
        let best_trade_pct = returns_pct.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst_trade_pct = returns_pct.iter().cloned().fold(f64::INFINITY, f64::min);

        Self {
            total_return,
            annualized_return,
            sharpe,
            sortino,
            calmar,
            max_dd,
            win_rate,
            avg_win_loss_ratio,
            profit_factor,
            total_trades: n,
            avg_bars_held,
            best_trade_pct,
            worst_trade_pct,
        }
    }

    fn zero() -> Self {
        Self {
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe: 0.0,
            sortino: 0.0,
            calmar: 0.0,
            max_dd: 0.0,
            win_rate: 0.0,
            avg_win_loss_ratio: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            avg_bars_held: 0.0,
            best_trade_pct: 0.0,
            worst_trade_pct: 0.0,
        }
    }
}

fn compute_sharpe(returns: &[f64], bars_per_year: f64) -> f64 {
    if returns.len() < 2 { return 0.0; }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    if std < 1e-12 { return 0.0; }
    mean / std * bars_per_year.sqrt()
}

fn compute_sortino(returns: &[f64], bars_per_year: f64) -> f64 {
    if returns.len() < 2 { return 0.0; }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let downside_var = returns
        .iter()
        .filter(|&&r| r < 0.0)
        .map(|&r| r.powi(2))
        .sum::<f64>()
        / n;
    let downside_std = downside_var.sqrt();
    if downside_std < 1e-12 { return 0.0; }
    mean / downside_std * bars_per_year.sqrt()
}

// ── BacktestReport ─────────────────────────────────────────────────────────

/// Full backtest report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub strategy_name: String,
    /// ISO-8601 date range strings.
    pub date_range: (String, String),
    pub metrics: PerformanceMetrics,
    pub trades: Vec<Trade>,
    /// Equity curve as a sequence of equity values (e.g. normalised to 1.0).
    pub equity_curve: Vec<f64>,
    /// Monthly return: year -> [jan..dec] (None if month not in range).
    pub monthly_returns: HashMap<i32, [Option<f64>; 12]>,
}

impl BacktestReport {
    pub fn new(
        strategy_name: impl Into<String>,
        date_range: (impl Into<String>, impl Into<String>),
        trades: Vec<Trade>,
        equity_curve: Vec<f64>,
        bars_per_year: f64,
    ) -> Self {
        let metrics = PerformanceMetrics::compute(&trades, &equity_curve, bars_per_year);
        let monthly_returns = compute_monthly_returns(&trades);
        Self {
            strategy_name: strategy_name.into(),
            date_range: (date_range.0.into(), date_range.1.into()),
            metrics,
            trades,
            equity_curve,
            monthly_returns,
        }
    }
}

// ── Monthly returns computation ────────────────────────────────────────────

/// Aggregate trade PnL by calendar month.
/// `ts` is a Unix timestamp (seconds). Returns year -> [12 monthly returns].
fn compute_monthly_returns(trades: &[Trade]) -> HashMap<i32, [Option<f64>; 12]> {
    let mut by_month: HashMap<(i32, u32), f64> = HashMap::new();

    for trade in trades {
        let ts = trade.exit_ts;
        // Simple Julian-day-based year/month calculation (no external deps).
        let (year, month) = ts_to_year_month(ts);
        *by_month.entry((year, month)).or_insert(0.0) += trade.pnl;
    }

    let mut result: HashMap<i32, [Option<f64>; 12]> = HashMap::new();
    for ((year, month), pnl) in &by_month {
        let entry = result
            .entry(*year)
            .or_insert([None; 12]);
        if *month >= 1 && *month <= 12 {
            entry[(month - 1) as usize] = Some(*pnl);
        }
    }
    result
}

/// Convert Unix timestamp to (year, month) using a simple algorithm.
/// Accurate for dates from 1970 onward.
fn ts_to_year_month(ts: i64) -> (i32, u32) {
    let days = ts / 86400;
    // Gregorian calendar calculation.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y_adj = if m <= 2 { y + 1 } else { y };
    (y_adj as i32, m as u32)
}

// ── ASCII sparkline ────────────────────────────────────────────────────────

const SPARK_CHARS: [char; 8] = ['\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}',
                                  '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];

/// Render an equity curve as an 80-character wide ASCII sparkline.
pub fn equity_sparkline(equity: &[f64], width: usize) -> String {
    if equity.is_empty() || width == 0 {
        return String::new();
    }

    // Downsample to `width` points.
    let downsampled: Vec<f64> = if equity.len() <= width {
        equity.to_vec()
    } else {
        (0..width)
            .map(|i| {
                let idx = i * (equity.len() - 1) / (width - 1);
                equity[idx.min(equity.len() - 1)]
            })
            .collect()
    };

    let min_val = downsampled.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = downsampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    downsampled
        .iter()
        .map(|&v| {
            let normalized = if range < 1e-12 {
                0.5
            } else {
                (v - min_val) / range
            };
            let idx = (normalized * 7.0).round().clamp(0.0, 7.0) as usize;
            SPARK_CHARS[idx]
        })
        .collect()
}

// ── Monthly returns ASCII heatmap ──────────────────────────────────────────

/// Render a monthly returns heatmap as ASCII text.
/// Format: 12 columns (months) x N rows (years), with colour coding.
pub fn monthly_heatmap_ascii(monthly_returns: &HashMap<i32, [Option<f64>; 12]>) -> String {
    if monthly_returns.is_empty() {
        return "No monthly data available.\n".to_string();
    }

    let mut years: Vec<i32> = monthly_returns.keys().cloned().collect();
    years.sort();

    let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

    let mut out = String::new();

    // Header row.
    out.push_str(&format!("{:<6}", "Year"));
    for m in &months {
        out.push_str(&format!("{:>8}", m));
    }
    out.push_str(&format!("{:>8}\n", "Annual"));
    out.push_str(&"-".repeat(6 + 8 * 13));
    out.push('\n');

    for year in &years {
        let row = &monthly_returns[year];
        out.push_str(&format!("{:<6}", year));
        let mut annual = 0.0f64;
        let mut n_months = 0usize;
        for &maybe_ret in row.iter() {
            match maybe_ret {
                Some(r) => {
                    let pct = r * 100.0;
                    let cell = format!("{:>7.1}%", pct);
                    // Simple +/- indicator without ANSI for portability.
                    out.push_str(&cell);
                    annual += r;
                    n_months += 1;
                }
                None => {
                    out.push_str(&format!("{:>8}", "--"));
                }
            }
        }
        if n_months > 0 {
            out.push_str(&format!("{:>7.1}%\n", annual * 100.0));
        } else {
            out.push_str(&format!("{:>8}\n", "--"));
        }
    }
    out
}

// ── Markdown report ────────────────────────────────────────────────────────

/// Render a full markdown report from a BacktestReport.
pub fn to_markdown(report: &BacktestReport) -> String {
    let m = &report.metrics;
    let mut out = String::new();

    out.push_str(&format!("# Backtest Report: {}\n\n", report.strategy_name));
    out.push_str(&format!(
        "**Date range:** {} to {}\n\n",
        report.date_range.0, report.date_range.1
    ));

    // Performance metrics table.
    out.push_str("## Performance Metrics\n\n");
    out.push_str("| Metric | Value |\n");
    out.push_str("|--------|------:|\n");
    out.push_str(&format!("| Total Return | {:.2}% |\n", m.total_return * 100.0));
    out.push_str(&format!("| Annualized Return | {:.2}% |\n", m.annualized_return * 100.0));
    out.push_str(&format!("| Sharpe Ratio | {:.3} |\n", m.sharpe));
    out.push_str(&format!("| Sortino Ratio | {:.3} |\n", m.sortino));
    out.push_str(&format!("| Calmar Ratio | {:.3} |\n", m.calmar));
    out.push_str(&format!("| Max Drawdown | {:.2}% |\n", m.max_dd * 100.0));
    out.push_str(&format!("| Win Rate | {:.1}% |\n", m.win_rate * 100.0));
    out.push_str(&format!("| Avg Win/Loss Ratio | {:.2} |\n", m.avg_win_loss_ratio));
    out.push_str(&format!("| Profit Factor | {:.2} |\n", m.profit_factor));
    out.push_str(&format!("| Total Trades | {} |\n", m.total_trades));
    out.push_str(&format!("| Avg Bars Held | {:.1} |\n", m.avg_bars_held));
    out.push_str(&format!("| Best Trade | {:.2}% |\n", m.best_trade_pct));
    out.push_str(&format!("| Worst Trade | {:.2}% |\n", m.worst_trade_pct));
    out.push('\n');

    // Equity sparkline.
    if !report.equity_curve.is_empty() {
        out.push_str("## Equity Curve\n\n");
        out.push_str("```\n");
        out.push_str(&equity_sparkline(&report.equity_curve, 80));
        out.push('\n');
        out.push_str("```\n\n");
    }

    // Monthly heatmap.
    if !report.monthly_returns.is_empty() {
        out.push_str("## Monthly Returns\n\n");
        out.push_str("```\n");
        out.push_str(&monthly_heatmap_ascii(&report.monthly_returns));
        out.push_str("```\n\n");
    }

    // Top 10 trades.
    if !report.trades.is_empty() {
        out.push_str("## Top 10 Trades by PnL\n\n");
        out.push_str("| Symbol | Direction | PnL | Return% | Bars Held |\n");
        out.push_str("|--------|-----------|----:|-------:|----------:|\n");

        let mut sorted_trades = report.trades.clone();
        sorted_trades.sort_by(|a, b| b.pnl.partial_cmp(&a.pnl).unwrap_or(std::cmp::Ordering::Equal));

        for trade in sorted_trades.iter().take(10) {
            out.push_str(&format!(
                "| {} | {} | {:.2} | {:.2}% | {} |\n",
                trade.symbol,
                trade.direction,
                trade.pnl,
                trade.return_pct(),
                trade.bars_held,
            ));
        }
        out.push('\n');
    }

    out
}

// ── HTML report ───────────────────────────────────────────────────────────

/// Render a full HTML report with inline CSS and metrics cards.
pub fn to_html(report: &BacktestReport) -> String {
    let m = &report.metrics;
    let mut out = String::new();

    out.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    out.push_str("<meta charset=\"UTF-8\">\n");
    out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
    out.push_str(&format!("<title>Backtest: {}</title>\n", report.strategy_name));
    out.push_str("<style>\n");
    out.push_str(HTML_CSS);
    out.push_str("</style>\n</head>\n<body>\n");

    out.push_str(&format!(
        "<h1>Backtest Report: {}</h1>\n",
        html_escape(&report.strategy_name)
    ));
    out.push_str(&format!(
        "<p class=\"date-range\">Period: {} &ndash; {}</p>\n",
        report.date_range.0, report.date_range.1
    ));

    // Metric cards.
    out.push_str("<div class=\"cards\">\n");
    out.push_str(&metric_card("Total Return", &format!("{:.2}%", m.total_return * 100.0), m.total_return > 0.0));
    out.push_str(&metric_card("Ann. Return", &format!("{:.2}%", m.annualized_return * 100.0), m.annualized_return > 0.0));
    out.push_str(&metric_card("Sharpe", &format!("{:.3}", m.sharpe), m.sharpe > 1.0));
    out.push_str(&metric_card("Sortino", &format!("{:.3}", m.sortino), m.sortino > 1.0));
    out.push_str(&metric_card("Calmar", &format!("{:.3}", m.calmar), m.calmar > 1.0));
    out.push_str(&metric_card("Max DD", &format!("{:.2}%", m.max_dd * 100.0), m.max_dd < 0.15));
    out.push_str(&metric_card("Win Rate", &format!("{:.1}%", m.win_rate * 100.0), m.win_rate > 0.5));
    out.push_str(&metric_card("Profit Factor", &format!("{:.2}", m.profit_factor), m.profit_factor > 1.5));
    out.push_str("</div>\n");

    // Full metrics table.
    out.push_str("<h2>Full Metrics</h2>\n<table>\n");
    out.push_str("<tr><th>Metric</th><th>Value</th></tr>\n");
    let rows = [
        ("Total Trades", format!("{}", m.total_trades)),
        ("Avg Bars Held", format!("{:.1}", m.avg_bars_held)),
        ("Best Trade", format!("{:.2}%", m.best_trade_pct)),
        ("Worst Trade", format!("{:.2}%", m.worst_trade_pct)),
        ("Avg W/L Ratio", format!("{:.2}", m.avg_win_loss_ratio)),
    ];
    for (label, val) in &rows {
        out.push_str(&format!("<tr><td>{}</td><td>{}</td></tr>\n", label, val));
    }
    out.push_str("</table>\n");

    // Equity sparkline.
    if !report.equity_curve.is_empty() {
        let spark = equity_sparkline(&report.equity_curve, 80);
        out.push_str("<h2>Equity Curve</h2>\n");
        out.push_str("<pre class=\"sparkline\">");
        out.push_str(&html_escape(&spark));
        out.push_str("</pre>\n");
    }

    // Monthly heatmap.
    if !report.monthly_returns.is_empty() {
        out.push_str("<h2>Monthly Returns</h2>\n");
        out.push_str("<pre class=\"heatmap\">");
        out.push_str(&html_escape(&monthly_heatmap_ascii(&report.monthly_returns)));
        out.push_str("</pre>\n");
    }

    out.push_str("</body>\n</html>\n");
    out
}

const HTML_CSS: &str = "
body { font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 2rem; }
h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }
h2 { color: #79c0ff; margin-top: 2rem; }
.date-range { color: #8b949e; margin-bottom: 1.5rem; }
.cards { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem 1.5rem; min-width: 120px; }
.card.good { border-color: #238636; }
.card.bad { border-color: #da3633; }
.card-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
.card-value { font-size: 1.4rem; font-weight: 600; margin-top: 0.25rem; }
.card.good .card-value { color: #3fb950; }
.card.bad .card-value { color: #f85149; }
table { border-collapse: collapse; width: 100%; max-width: 600px; }
th, td { padding: 0.5rem 1rem; border: 1px solid #30363d; text-align: left; }
th { background: #161b22; color: #58a6ff; }
tr:nth-child(even) { background: #0d1117; }
.sparkline { font-size: 1.2rem; letter-spacing: 0.1em; background: #161b22; padding: 1rem; border-radius: 4px; overflow-x: auto; }
.heatmap { font-size: 0.85rem; background: #161b22; padding: 1rem; border-radius: 4px; overflow-x: auto; }
";

fn metric_card(label: &str, value: &str, is_good: bool) -> String {
    let class = if is_good { "card good" } else { "card bad" };
    format!(
        "<div class=\"{class}\">\
         <div class=\"card-label\">{label}</div>\
         <div class=\"card-value\">{value}</div>\
         </div>\n"
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trades(n: usize) -> Vec<Trade> {
        (0..n)
            .map(|i| Trade {
                symbol: "BTC".to_string(),
                entry_price: 100.0,
                exit_price: if i % 3 == 0 { 90.0 } else { 115.0 },
                pnl: if i % 3 == 0 { -10.0 } else { 15.0 },
                direction: "long".to_string(),
                entry_ts: (i as i64) * 86400 + 1_700_000_000,
                exit_ts: (i as i64) * 86400 + 1_700_000_000 + 3600,
                bars_held: 4,
            })
            .collect()
    }

    fn make_equity(n: usize, daily_return: f64) -> Vec<f64> {
        let mut eq = vec![1.0f64; n];
        for i in 1..n {
            eq[i] = eq[i - 1] * (1.0 + daily_return);
        }
        eq
    }

    #[test]
    fn test_performance_metrics_win_rate() {
        let trades = make_trades(9); // 3 losers, 6 winners
        let equity = make_equity(100, 0.001);
        let m = PerformanceMetrics::compute(&trades, &equity, 252.0);
        // 6/9 winners = 0.666...
        assert!((m.win_rate - 6.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_performance_metrics_profit_factor() {
        let trades = make_trades(9);
        let equity = make_equity(100, 0.001);
        let m = PerformanceMetrics::compute(&trades, &equity, 252.0);
        // gross_profit = 6 * 15 = 90, gross_loss = 3 * 10 = 30
        assert!((m.profit_factor - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_equity_sparkline_length() {
        let equity = make_equity(200, 0.001);
        let spark = equity_sparkline(&equity, 80);
        // Unicode chars: each is one char, but length in bytes will vary.
        assert_eq!(spark.chars().count(), 80);
    }

    #[test]
    fn test_equity_sparkline_empty() {
        let spark = equity_sparkline(&[], 80);
        assert_eq!(spark, "");
    }

    #[test]
    fn test_equity_sparkline_short() {
        // Fewer points than width -- should return as-is.
        let equity = vec![1.0, 1.1, 0.9, 1.2];
        let spark = equity_sparkline(&equity, 80);
        assert_eq!(spark.chars().count(), 4);
    }

    #[test]
    fn test_to_markdown_contains_strategy_name() {
        let trades = make_trades(20);
        let equity = make_equity(100, 0.001);
        let report = BacktestReport::new("TestStrategy", ("2024-01-01", "2024-12-31"), trades, equity, 252.0);
        let md = to_markdown(&report);
        assert!(md.contains("TestStrategy"));
        assert!(md.contains("Sharpe Ratio"));
        assert!(md.contains("Win Rate"));
    }

    #[test]
    fn test_to_html_contains_strategy_name() {
        let trades = make_trades(10);
        let equity = make_equity(50, 0.002);
        let report = BacktestReport::new("HTMLTest", ("2023-01-01", "2023-06-30"), trades, equity, 252.0);
        let html = to_html(&report);
        assert!(html.contains("HTMLTest"));
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_monthly_heatmap_ascii_structure() {
        let mut monthly: HashMap<i32, [Option<f64>; 12]> = HashMap::new();
        let mut row = [None; 12];
        row[0] = Some(0.05);
        row[11] = Some(-0.02);
        monthly.insert(2024, row);
        let hm = monthly_heatmap_ascii(&monthly);
        assert!(hm.contains("2024"));
        assert!(hm.contains("Jan"));
        assert!(hm.contains("Dec"));
    }

    #[test]
    fn test_ts_to_year_month_epoch() {
        // Unix epoch 0 = 1970-01-01
        let (year, month) = ts_to_year_month(0);
        assert_eq!(year, 1970);
        assert_eq!(month, 1);
    }

    #[test]
    fn test_ts_to_year_month_known_date() {
        // 2024-07-04 = 1720051200 seconds approx
        let ts = 1720051200i64;
        let (year, month) = ts_to_year_month(ts);
        assert_eq!(year, 2024);
        assert_eq!(month, 7);
    }

    #[test]
    fn test_max_drawdown_in_metrics() {
        // Equity goes up then crashes.
        let equity = vec![1.0, 1.5, 2.0, 1.0, 1.2];
        let trades: Vec<Trade> = vec![];
        let m = PerformanceMetrics::compute(&trades, &equity, 252.0);
        // Peak = 2.0, trough = 1.0 -> max dd = 50%.
        assert!((m.max_dd - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("a & b"), "a &amp; b");
    }

    #[test]
    fn test_compute_monthly_returns_bucketing() {
        // Trades in Jan 2024 and Dec 2024.
        let trades = vec![
            Trade {
                symbol: "BTC".to_string(),
                entry_price: 100.0,
                exit_price: 110.0,
                pnl: 10.0,
                direction: "long".to_string(),
                entry_ts: 1704067200, // 2024-01-01
                exit_ts: 1704067200,
                bars_held: 1,
            },
            Trade {
                symbol: "ETH".to_string(),
                entry_price: 200.0,
                exit_price: 190.0,
                pnl: -10.0,
                direction: "long".to_string(),
                entry_ts: 1735689600, // 2025-01-01 (checking boundary)
                exit_ts: 1735689600,
                bars_held: 1,
            },
        ];
        let monthly = compute_monthly_returns(&trades);
        // 2024 Jan should have pnl = 10.
        if let Some(row) = monthly.get(&2024) {
            assert_eq!(row[0], Some(10.0)); // January
        }
    }

    #[test]
    fn test_trade_return_pct() {
        let t = Trade {
            symbol: "BTC".to_string(),
            entry_price: 100.0,
            exit_price: 120.0,
            pnl: 20.0,
            direction: "long".to_string(),
            entry_ts: 0,
            exit_ts: 3600,
            bars_held: 4,
        };
        assert!((t.return_pct() - 20.0).abs() < 1e-9);
        assert!(t.is_winner());
    }
}
