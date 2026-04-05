use anyhow::{Context, Result};
use clap::Parser;
use network_graph::{
    community_detection::detect_communities,
    correlation_matrix::CorrelationMatrix,
    granger_causality::granger_causality_dag,
    lead_lag::build_lead_lag_network,
    minimum_spanning_tree::minimum_spanning_tree,
    network_signal::generate_network_signals,
    regime_correlation::regime_correlation_analysis,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// Crypto market network analysis tool.
#[derive(Debug, Parser)]
#[command(name = "network-graph", version, about)]
struct Args {
    /// CSV file with return series. Expected columns: symbol, timestamp, return.
    /// Each row is one bar for one symbol.
    #[arg(long)]
    input: PathBuf,

    /// Output file for JSON report.
    #[arg(long, default_value = "network_report.json")]
    output: PathBuf,

    /// Rolling window for correlation computation (bars).
    #[arg(long, default_value_t = 252)]
    window: usize,

    /// Minimum |correlation| threshold for community graph edges.
    #[arg(long, default_value_t = 0.5)]
    threshold: f64,

    /// BTC symbol name in the data (for regime analysis).
    #[arg(long, default_value = "BTC")]
    btc_sym: String,

    /// Portfolio symbols to analyse for signals (comma-separated).
    #[arg(long, default_value = "BTC,ETH")]
    portfolio: String,

    /// Maximum Granger causality lag order.
    #[arg(long, default_value_t = 3)]
    granger_lag: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("[network-graph] Loading return series from {:?}", args.input);
    let returns = load_return_series(&args.input)?;
    let symbols: Vec<String> = {
        let mut v: Vec<String> = returns.keys().cloned().collect();
        v.sort();
        v
    };
    eprintln!(
        "[network-graph] Loaded {} symbols, up to {} bars each",
        symbols.len(),
        returns.values().map(|r| r.len()).max().unwrap_or(0)
    );

    // ── 1. Correlation matrix ──────────────────────────────────────────────
    eprintln!("[network-graph] Computing correlation matrix (window={})...", args.window);
    let corr = CorrelationMatrix::from_returns(&returns, args.window);
    eprintln!("[network-graph] Shrinkage (Ledoit-Wolf)...");
    let corr_shrunk = CorrelationMatrix::shrink_ledoit_wolf(&returns, args.window);

    // ── 2. Minimum spanning tree ───────────────────────────────────────────
    eprintln!("[network-graph] Computing MST...");
    let mst = minimum_spanning_tree(&corr.symbols, &corr.data);
    eprintln!("[network-graph] MST total weight: {:.4}", mst.total_weight);
    eprintln!("[network-graph] Top hubs:");
    for (sym, deg) in mst.hubs.iter().take(5) {
        eprintln!("  {} (degree {})", sym, deg);
    }

    // ── 3. Community detection ─────────────────────────────────────────────
    eprintln!(
        "[network-graph] Detecting communities (threshold={})...",
        args.threshold
    );
    let communities = detect_communities(&corr.symbols, &corr.data, args.threshold);
    eprintln!(
        "[network-graph] Found {} communities, modularity={:.4}",
        communities.n_communities, communities.modularity
    );
    for (cid, members) in &communities.communities {
        eprintln!("  Community {}: {:?}", cid, members);
    }

    // ── 4. Lead-lag network ────────────────────────────────────────────────
    eprintln!("[network-graph] Computing lead-lag network (max_lag=5)...");
    let lead_lag = build_lead_lag_network(&returns, 5, 0.3);
    eprintln!(
        "[network-graph] {} directed edges in lead-lag network",
        lead_lag.edges.len()
    );
    for edge in lead_lag.edges.iter().take(5) {
        eprintln!(
            "  {} → {} (lag={} bars, corr={:.3})",
            edge.leader, edge.follower, edge.lag_bars, edge.correlation
        );
    }

    // ── 5. Granger causality ───────────────────────────────────────────────
    eprintln!(
        "[network-graph] Granger causality tests (lag={})...",
        args.granger_lag
    );
    let granger = granger_causality_dag(&returns, args.granger_lag);
    eprintln!(
        "[network-graph] {} significant Granger edges",
        granger.len()
    );
    for e in granger.iter().take(5) {
        eprintln!(
            "  {} → {} (F={:.2}, p={:.4})",
            e.cause, e.effect, e.f_stat, e.p_value
        );
    }

    // ── 6. Regime correlation ──────────────────────────────────────────────
    eprintln!("[network-graph] Regime correlation analysis...");
    let regime = regime_correlation_analysis(&returns, &args.btc_sym, 0.02);
    eprintln!("  Avg corr (full): {:.3}", regime.avg_corr_full);
    eprintln!("  Avg corr (bull): {:.3}", regime.avg_corr_bull);
    eprintln!("  Avg corr (bear): {:.3}", regime.avg_corr_bear);
    eprintln!("  Avg corr (high_vol): {:.3}", regime.avg_corr_high_vol);

    // ── 7. Network signals ─────────────────────────────────────────────────
    let portfolio_symbols: Vec<String> = args
        .portfolio
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .collect();
    let recent_returns: HashMap<String, f64> = returns
        .iter()
        .filter_map(|(s, r)| r.last().map(|&v| (s.clone(), v)))
        .collect();
    let signals = generate_network_signals(
        &portfolio_symbols,
        &mst,
        &communities,
        &lead_lag,
        &recent_returns,
        3,
    );

    eprintln!("[network-graph] Network signals for {:?}:", portfolio_symbols);
    eprintln!("  Diversification score: {:.3}", signals.diversification_score);
    eprintln!("  Hub concentration risk: {}", signals.hub_concentration_risk);
    eprintln!("  Communities in portfolio: {}", signals.portfolio_community_count);
    if let Some(w) = &signals.community_overlap_warning {
        eprintln!("  WARNING: {}", w);
    }
    for sig in signals.lead_signals.iter().take(3) {
        eprintln!(
            "  Lead signal: {} → {} (lag={}, strength={:.3}, dir={})",
            sig.leader, sig.follower, sig.lag_bars, sig.signal_strength, sig.direction
        );
    }

    // ── 8. Write JSON report ───────────────────────────────────────────────
    let report = serde_json::json!({
        "symbols": symbols,
        "correlation_matrix": corr_shrunk.data,
        "mst": mst,
        "communities": communities,
        "lead_lag_network": lead_lag,
        "granger_dag": granger,
        "regime_correlation": regime,
        "network_signals": signals,
    });

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.output, json)?;
    eprintln!("[network-graph] Report written to {:?}", args.output);

    Ok(())
}

/// Load return series from a CSV file.
/// Expected format: rows with columns symbol,timestamp,return (or symbol,return).
fn load_return_series(path: &PathBuf) -> Result<HashMap<String, Vec<f64>>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("opening {:?}", path))?;

    let headers = reader.headers()?.clone();
    let h: Vec<&str> = headers.iter().collect();

    let i_sym = h.iter().position(|&c| c.to_ascii_lowercase() == "symbol")
        .unwrap_or(0);
    let i_ret = h.iter()
        .position(|&c| c.to_ascii_lowercase() == "return" || c.to_ascii_lowercase() == "ret")
        .or_else(|| h.iter().position(|&c| c.to_ascii_lowercase() == "close"))
        .unwrap_or(h.len().saturating_sub(1));

    let mut series: HashMap<String, Vec<f64>> = HashMap::new();
    let mut prev_close: HashMap<String, f64> = HashMap::new();

    for record in reader.records() {
        let rec = record?;
        let sym = rec.get(i_sym).unwrap_or("UNKNOWN").trim().to_uppercase();
        if let Some(val_str) = rec.get(i_ret) {
            if let Ok(val) = val_str.trim().parse::<f64>() {
                let ret = if h[i_ret].to_ascii_lowercase() == "close" {
                    // Treat as price, compute log return.
                    let prev = prev_close.get(&sym).copied().unwrap_or(val);
                    let r = if prev > 0.0 { (val / prev).ln() } else { 0.0 };
                    prev_close.insert(sym.clone(), val);
                    r
                } else {
                    val
                };
                series.entry(sym).or_default().push(ret);
            }
        }
    }

    Ok(series)
}
