// report.rs — HTML/text report generation, equity curve data, trade list, monthly returns, drawdown chart
use crate::analytics::{PerformanceMetrics, compute_metrics, drawdown_series, equity_curve, monthly_returns, annual_returns, format_metrics};
use crate::portfolio::{Portfolio, Trade, TradeStats, Roundtrip, extract_roundtrips};
use std::collections::HashMap;

/// Report data container
#[derive(Clone, Debug)]
pub struct BacktestReport {
    pub strategy_name: String,
    pub start_date: String,
    pub end_date: String,
    pub initial_capital: f64,
    pub final_equity: f64,
    pub metrics: PerformanceMetrics,
    pub equity_curve: Vec<(u64, f64)>,
    pub returns: Vec<f64>,
    pub drawdowns: Vec<f64>,
    pub trades: Vec<Trade>,
    pub trade_stats: TradeStats,
    pub monthly_returns: Vec<f64>,
    pub annual_returns: Vec<f64>,
    pub position_history: Vec<HashMap<String, f64>>,
    pub benchmark_returns: Option<Vec<f64>>,
}

impl BacktestReport {
    pub fn from_portfolio(
        portfolio: &Portfolio,
        equity_curve: &[(u64, f64)],
        strategy_name: &str,
        risk_free_rate: f64,
        periods_per_year: f64,
    ) -> Self {
        let returns = portfolio.returns();
        let metrics = compute_metrics(&returns, risk_free_rate, periods_per_year);
        let drawdowns = drawdown_series(&returns);
        let trade_stats = portfolio.trade_stats();
        let monthly = monthly_returns(&returns, (periods_per_year / 12.0) as usize);
        let annual = annual_returns(&returns, periods_per_year as usize);

        Self {
            strategy_name: strategy_name.to_string(),
            start_date: String::new(),
            end_date: String::new(),
            initial_capital: portfolio.initial_capital,
            final_equity: portfolio.total_equity(),
            metrics,
            equity_curve: equity_curve.to_vec(),
            returns,
            drawdowns,
            trades: portfolio.trade_log.clone(),
            trade_stats,
            monthly_returns: monthly,
            annual_returns: annual,
            position_history: Vec::new(),
            benchmark_returns: None,
        }
    }
}

/// Generate text report
pub fn text_report(report: &BacktestReport) -> String {
    let mut s = String::new();
    s.push_str("================================================================\n");
    s.push_str(&format!("  BACKTEST REPORT: {}\n", report.strategy_name));
    s.push_str("================================================================\n\n");

    s.push_str("--- SUMMARY ---\n");
    s.push_str(&format!("Initial Capital:    ${:.2}\n", report.initial_capital));
    s.push_str(&format!("Final Equity:       ${:.2}\n", report.final_equity));
    s.push_str(&format!("Net Profit:         ${:.2}\n", report.final_equity - report.initial_capital));
    s.push_str("\n");

    s.push_str("--- PERFORMANCE METRICS ---\n");
    s.push_str(&format_metrics(&report.metrics));
    s.push_str("\n");

    s.push_str("--- TRADE STATISTICS ---\n");
    s.push_str(&format!("Total Trades:       {}\n", report.trade_stats.total_trades));
    s.push_str(&format!("Winning Trades:     {}\n", report.trade_stats.winning_trades));
    s.push_str(&format!("Losing Trades:      {}\n", report.trade_stats.losing_trades));
    s.push_str(&format!("Win Rate:           {:.2}%\n", report.trade_stats.win_rate * 100.0));
    s.push_str(&format!("Avg Win:            {:.4}\n", report.trade_stats.avg_win));
    s.push_str(&format!("Avg Loss:           {:.4}\n", report.trade_stats.avg_loss));
    s.push_str(&format!("Profit Factor:      {:.3}\n", report.trade_stats.profit_factor));
    s.push_str(&format!("Expectancy:         {:.4}\n", report.trade_stats.expectancy));
    s.push_str(&format!("Max Win:            {:.4}\n", report.trade_stats.max_win));
    s.push_str(&format!("Max Loss:           {:.4}\n", report.trade_stats.max_loss));
    s.push_str(&format!("Total Commission:   ${:.2}\n", report.trade_stats.total_commission));
    s.push_str(&format!("Total Slippage:     ${:.2}\n", report.trade_stats.total_slippage));
    s.push_str("\n");

    // Monthly returns table
    if !report.monthly_returns.is_empty() {
        s.push_str("--- MONTHLY RETURNS ---\n");
        for (i, &mr) in report.monthly_returns.iter().enumerate() {
            s.push_str(&format!("  Month {:3}: {:+.2}%\n", i + 1, mr * 100.0));
        }
        s.push_str("\n");
    }

    // Annual returns
    if !report.annual_returns.is_empty() {
        s.push_str("--- ANNUAL RETURNS ---\n");
        for (i, &ar) in report.annual_returns.iter().enumerate() {
            s.push_str(&format!("  Year {:3}: {:+.2}%\n", i + 1, ar * 100.0));
        }
        s.push_str("\n");
    }

    // Top trades
    if report.trades.len() > 0 {
        s.push_str("--- RECENT TRADES (last 20) ---\n");
        let start = if report.trades.len() > 20 { report.trades.len() - 20 } else { 0 };
        for trade in &report.trades[start..] {
            let side = if trade.is_buy() { "BUY " } else { "SELL" };
            s.push_str(&format!("  {} {} {} @ {:.4} qty={:.2} pnl={:.4}\n",
                trade.timestamp, side, trade.symbol, trade.price, trade.quantity, trade.pnl));
        }
        s.push_str("\n");
    }

    s.push_str("================================================================\n");
    s
}

/// Generate HTML report
pub fn html_report(report: &BacktestReport) -> String {
    let mut s = String::new();
    s.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    s.push_str("<meta charset=\"utf-8\">\n");
    s.push_str(&format!("<title>Backtest Report: {}</title>\n", report.strategy_name));
    s.push_str("<style>\n");
    s.push_str("body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }\n");
    s.push_str("h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }\n");
    s.push_str("h2 { color: #555; margin-top: 30px; }\n");
    s.push_str(".card { background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n");
    s.push_str(".metric { display: inline-block; width: 200px; margin: 5px 10px; }\n");
    s.push_str(".metric-label { color: #888; font-size: 12px; }\n");
    s.push_str(".metric-value { font-size: 20px; font-weight: bold; }\n");
    s.push_str(".positive { color: #4CAF50; }\n");
    s.push_str(".negative { color: #f44336; }\n");
    s.push_str("table { border-collapse: collapse; width: 100%; margin: 10px 0; }\n");
    s.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }\n");
    s.push_str("th { background: #4CAF50; color: white; }\n");
    s.push_str("tr:nth-child(even) { background: #f2f2f2; }\n");
    s.push_str("</style>\n</head>\n<body>\n");

    s.push_str(&format!("<h1>Backtest Report: {}</h1>\n", report.strategy_name));

    // Summary card
    s.push_str("<div class=\"card\">\n<h2>Summary</h2>\n");
    let m = &report.metrics;
    let class = |v: f64| if v >= 0.0 { "positive" } else { "negative" };

    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Total Return</div><div class=\"metric-value {}\">{:+.2}%</div></div>\n",
        class(m.total_return), m.total_return * 100.0));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Sharpe Ratio</div><div class=\"metric-value {}\">{:.3}</div></div>\n",
        class(m.sharpe_ratio), m.sharpe_ratio));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Sortino Ratio</div><div class=\"metric-value {}\">{:.3}</div></div>\n",
        class(m.sortino_ratio), m.sortino_ratio));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Max Drawdown</div><div class=\"metric-value negative\">{:.2}%</div></div>\n",
        m.max_drawdown * 100.0));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Calmar Ratio</div><div class=\"metric-value {}\">{:.3}</div></div>\n",
        class(m.calmar_ratio), m.calmar_ratio));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Win Rate</div><div class=\"metric-value\">{:.1}%</div></div>\n",
        m.win_rate * 100.0));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Ann. Return</div><div class=\"metric-value {}\">{:+.2}%</div></div>\n",
        class(m.annualized_return), m.annualized_return * 100.0));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Ann. Volatility</div><div class=\"metric-value\">{:.2}%</div></div>\n",
        m.annualized_volatility * 100.0));
    s.push_str("</div>\n");

    // Equity curve data (as JSON for charts)
    s.push_str("<div class=\"card\">\n<h2>Equity Curve</h2>\n");
    s.push_str("<div id=\"equity-data\" style=\"display:none;\">\n");
    let eq = equity_curve(&report.returns, report.initial_capital);
    s.push_str("[");
    for (i, &e) in eq.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("{:.2}", e));
    }
    s.push_str("]\n</div>\n");
    // ASCII sparkline
    if !eq.is_empty() {
        let min_e = eq.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_e = eq.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_e - min_e;
        if range > 0.0 {
            s.push_str("<pre style=\"font-size:10px;line-height:1;\">");
            let width = 80.min(eq.len());
            let step = eq.len() / width;
            let height = 10;
            for row in (0..height).rev() {
                let threshold = min_e + (row as f64 / height as f64) * range;
                for col in 0..width {
                    let idx = col * step;
                    if eq[idx] >= threshold { s.push('#'); } else { s.push(' '); }
                }
                s.push('\n');
            }
            s.push_str("</pre>\n");
        }
    }
    s.push_str("</div>\n");

    // Drawdown
    s.push_str("<div class=\"card\">\n<h2>Drawdown</h2>\n");
    s.push_str("<div id=\"drawdown-data\" style=\"display:none;\">\n");
    s.push_str("[");
    for (i, &d) in report.drawdowns.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("{:.4}", d));
    }
    s.push_str("]\n</div>\n</div>\n");

    // Monthly returns table
    if !report.monthly_returns.is_empty() {
        s.push_str("<div class=\"card\">\n<h2>Monthly Returns</h2>\n");
        s.push_str("<table>\n<tr><th>Month</th><th>Return</th></tr>\n");
        for (i, &mr) in report.monthly_returns.iter().enumerate() {
            let cls = class(mr);
            s.push_str(&format!("<tr><td>{}</td><td class=\"{}\">{:+.2}%</td></tr>\n", i + 1, cls, mr * 100.0));
        }
        s.push_str("</table>\n</div>\n");
    }

    // Trade table
    s.push_str("<div class=\"card\">\n<h2>Trade Log</h2>\n");
    s.push_str(&format!("<p>Total: {} trades</p>\n", report.trades.len()));
    s.push_str("<table>\n<tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Commission</th><th>PnL</th></tr>\n");
    let display_trades = if report.trades.len() > 50 { &report.trades[report.trades.len() - 50..] } else { &report.trades };
    for trade in display_trades {
        let side = if trade.is_buy() { "BUY" } else { "SELL" };
        let pnl_cls = class(trade.pnl);
        s.push_str(&format!("<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.4}</td><td>{:.4}</td><td class=\"{}\">{:.4}</td></tr>\n",
            trade.timestamp, trade.symbol, side, trade.quantity, trade.price, trade.commission, pnl_cls, trade.pnl));
    }
    s.push_str("</table>\n</div>\n");

    // Risk metrics
    s.push_str("<div class=\"card\">\n<h2>Risk Metrics</h2>\n");
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">VaR 95%</div><div class=\"metric-value\">{:.4}</div></div>\n", m.var_95));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">CVaR 95%</div><div class=\"metric-value\">{:.4}</div></div>\n", m.cvar_95));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Skewness</div><div class=\"metric-value\">{:.3}</div></div>\n", m.skewness));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Kurtosis</div><div class=\"metric-value\">{:.3}</div></div>\n", m.kurtosis));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Ulcer Index</div><div class=\"metric-value\">{:.4}</div></div>\n", m.ulcer_index));
    s.push_str(&format!("<div class=\"metric\"><div class=\"metric-label\">Tail Ratio</div><div class=\"metric-value\">{:.3}</div></div>\n", m.tail_ratio));
    s.push_str("</div>\n");

    s.push_str("</body>\n</html>\n");
    s
}

/// Generate CSV trade list
pub fn csv_trade_list(trades: &[Trade]) -> String {
    let mut s = String::from("timestamp,symbol,side,quantity,price,commission,slippage,pnl,tag\n");
    for t in trades {
        let side = if t.is_buy() { "BUY" } else { "SELL" };
        s.push_str(&format!("{},{},{},{},{},{},{},{},{}\n",
            t.timestamp, t.symbol, side, t.quantity, t.price, t.commission, t.slippage, t.pnl, t.tag));
    }
    s
}

/// Generate CSV equity curve
pub fn csv_equity_curve(curve: &[(u64, f64)]) -> String {
    let mut s = String::from("timestamp,equity\n");
    for &(ts, eq) in curve {
        s.push_str(&format!("{},{:.4}\n", ts, eq));
    }
    s
}

/// Compare multiple strategies
pub fn comparison_table(reports: &[BacktestReport]) -> String {
    let mut s = String::new();
    s.push_str(&format!("{:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
        "Strategy", "Return", "Sharpe", "Sortino", "MaxDD", "WinRate", "Trades"));
    s.push_str(&"-".repeat(85));
    s.push_str("\n");
    for r in reports {
        s.push_str(&format!("{:<25} {:>9.2}% {:>10.3} {:>10.3} {:>9.2}% {:>9.1}% {:>10}\n",
            r.strategy_name,
            r.metrics.total_return * 100.0,
            r.metrics.sharpe_ratio,
            r.metrics.sortino_ratio,
            r.metrics.max_drawdown * 100.0,
            r.metrics.win_rate * 100.0,
            r.trade_stats.total_trades));
    }
    s
}

/// Roundtrip analysis report
pub fn roundtrip_report(trades: &[Trade]) -> String {
    let roundtrips = extract_roundtrips(trades);
    let mut s = String::new();
    s.push_str("--- ROUNDTRIP ANALYSIS ---\n");
    s.push_str(&format!("Total Roundtrips: {}\n", roundtrips.len()));

    if roundtrips.is_empty() { return s; }

    let pnls: Vec<f64> = roundtrips.iter().map(|r| r.pnl).collect();
    let total_pnl: f64 = pnls.iter().sum();
    let avg_pnl = total_pnl / pnls.len() as f64;
    let winners = roundtrips.iter().filter(|r| r.pnl > 0.0).count();
    let win_rate = winners as f64 / roundtrips.len() as f64;
    let max_pnl = pnls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_pnl = pnls.iter().cloned().fold(f64::INFINITY, f64::min);

    let pnl_pcts: Vec<f64> = roundtrips.iter().map(|r| r.pnl_pct).collect();
    let avg_pct = pnl_pcts.iter().sum::<f64>() / pnl_pcts.len() as f64;

    s.push_str(&format!("Total PnL:        ${:.2}\n", total_pnl));
    s.push_str(&format!("Average PnL:      ${:.2}\n", avg_pnl));
    s.push_str(&format!("Win Rate:         {:.1}%\n", win_rate * 100.0));
    s.push_str(&format!("Avg Return:       {:.2}%\n", avg_pct * 100.0));
    s.push_str(&format!("Best Trade:       ${:.2}\n", max_pnl));
    s.push_str(&format!("Worst Trade:      ${:.2}\n", min_pnl));
    s.push_str(&format!("Total Commission: ${:.2}\n",
        roundtrips.iter().map(|r| r.commission).sum::<f64>()));

    // PnL distribution
    s.push_str("\nPnL Distribution:\n");
    let buckets = [
        ("<-5%", -f64::INFINITY, -0.05),
        ("-5% to -2%", -0.05, -0.02),
        ("-2% to 0%", -0.02, 0.0),
        ("0% to 2%", 0.0, 0.02),
        ("2% to 5%", 0.02, 0.05),
        (">5%", 0.05, f64::INFINITY),
    ];
    for (label, lo, hi) in &buckets {
        let count = pnl_pcts.iter().filter(|&&p| p >= *lo && p < *hi).count();
        let bar = "#".repeat(count.min(40));
        s.push_str(&format!("  {:>12} {:>4} {}\n", label, count, bar));
    }

    s
}

/// Export report data as JSON
pub fn json_report(report: &BacktestReport) -> String {
    let mut s = String::from("{\n");
    s.push_str(&format!("  \"strategy\": \"{}\",\n", report.strategy_name));
    s.push_str(&format!("  \"initial_capital\": {:.2},\n", report.initial_capital));
    s.push_str(&format!("  \"final_equity\": {:.2},\n", report.final_equity));
    s.push_str(&format!("  \"total_return\": {:.6},\n", report.metrics.total_return));
    s.push_str(&format!("  \"annualized_return\": {:.6},\n", report.metrics.annualized_return));
    s.push_str(&format!("  \"annualized_volatility\": {:.6},\n", report.metrics.annualized_volatility));
    s.push_str(&format!("  \"sharpe_ratio\": {:.6},\n", report.metrics.sharpe_ratio));
    s.push_str(&format!("  \"sortino_ratio\": {:.6},\n", report.metrics.sortino_ratio));
    s.push_str(&format!("  \"calmar_ratio\": {:.6},\n", report.metrics.calmar_ratio));
    s.push_str(&format!("  \"max_drawdown\": {:.6},\n", report.metrics.max_drawdown));
    s.push_str(&format!("  \"win_rate\": {:.6},\n", report.metrics.win_rate));
    s.push_str(&format!("  \"profit_factor\": {:.6},\n", report.metrics.profit_factor));
    s.push_str(&format!("  \"num_trades\": {},\n", report.trade_stats.total_trades));
    s.push_str(&format!("  \"total_commission\": {:.4},\n", report.trade_stats.total_commission));
    s.push_str(&format!("  \"skewness\": {:.6},\n", report.metrics.skewness));
    s.push_str(&format!("  \"kurtosis\": {:.6},\n", report.metrics.kurtosis));
    s.push_str(&format!("  \"var_95\": {:.6},\n", report.metrics.var_95));
    s.push_str(&format!("  \"cvar_95\": {:.6}\n", report.metrics.cvar_95));
    s.push_str("}\n");
    s
}

/// Save report to file
pub fn save_report(path: &str, content: &str) -> std::io::Result<()> {
    std::fs::write(path, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portfolio::TradeSide;

    fn make_test_report() -> BacktestReport {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.003, 0.012];
        let metrics = compute_metrics(&returns, 0.0, 252.0);
        let eq: Vec<(u64, f64)> = (0..9).map(|i| (i as u64 * 60000, 100000.0)).collect();
        BacktestReport {
            strategy_name: "Test".to_string(),
            start_date: "2024-01-01".to_string(),
            end_date: "2024-12-31".to_string(),
            initial_capital: 100000.0,
            final_equity: 104000.0,
            metrics,
            equity_curve: eq,
            returns: returns.clone(),
            drawdowns: drawdown_series(&returns),
            trades: vec![
                Trade { timestamp: 1000, symbol: "AAPL".into(), side: TradeSide::Buy,
                    quantity: 100.0, price: 150.0, commission: 1.0, slippage: 0.1,
                    order_id: 1, pnl: 500.0, tag: String::new() },
            ],
            trade_stats: TradeStats { total_trades: 1, winning_trades: 1, losing_trades: 0,
                win_rate: 1.0, avg_win: 500.0, avg_loss: 0.0, profit_factor: f64::INFINITY,
                expectancy: 500.0, max_win: 500.0, max_loss: 0.0,
                total_commission: 1.0, total_slippage: 0.1 },
            monthly_returns: vec![0.03, 0.01],
            annual_returns: vec![0.04],
            position_history: Vec::new(),
            benchmark_returns: None,
        }
    }

    #[test]
    fn test_text_report() {
        let report = make_test_report();
        let text = text_report(&report);
        assert!(text.contains("BACKTEST REPORT"));
        assert!(text.contains("Sharpe"));
    }

    #[test]
    fn test_html_report() {
        let report = make_test_report();
        let html = html_report(&report);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Sharpe"));
    }

    #[test]
    fn test_json_report() {
        let report = make_test_report();
        let json = json_report(&report);
        assert!(json.contains("\"strategy\""));
        assert!(json.contains("\"sharpe_ratio\""));
    }

    #[test]
    fn test_csv_trades() {
        let trades = vec![
            Trade { timestamp: 1000, symbol: "X".into(), side: TradeSide::Buy,
                quantity: 10.0, price: 100.0, commission: 1.0, slippage: 0.0,
                order_id: 1, pnl: 0.0, tag: String::new() },
        ];
        let csv = csv_trade_list(&trades);
        assert!(csv.contains("timestamp,symbol"));
        assert!(csv.contains("BUY"));
    }

    #[test]
    fn test_comparison() {
        let r1 = make_test_report();
        let mut r2 = make_test_report();
        r2.strategy_name = "Other".to_string();
        let table = comparison_table(&[r1, r2]);
        assert!(table.contains("Test"));
        assert!(table.contains("Other"));
    }
}
