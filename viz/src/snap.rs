use std::fs;
use serde_json::Value;
use chrono::Local;

pub fn run_snap(json_path: &str, _snap_type: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(json_path)?;
    let now = Local::now().format("%Y-%m-%dT%H:%M").to_string();

    if let Ok(arr) = serde_json::from_str::<Vec<Value>>(&content) {
        // Array of experiments
        if let Some(best) = arr.iter()
            .max_by(|a, b| {
                let sa = a["combined_score"].as_f64().unwrap_or(f64::NEG_INFINITY);
                let sb = b["combined_score"].as_f64().unwrap_or(f64::NEG_INFINITY);
                sa.partial_cmp(&sb).unwrap()
            })
        {
            let name = best["exp"].as_str().unwrap_or("?");
            let score = best["combined_score"].as_f64().unwrap_or(0.0);
            let arena = best["arena_sharpe"].as_f64().unwrap_or(0.0);
            let synth = best["synth_sharpe"].as_f64().unwrap_or(0.0);
            let baseline_sh = arr.iter()
                .find(|e| e["exp"].as_str() == Some("BASELINE"))
                .and_then(|b| b["arena_sharpe"].as_f64())
                .unwrap_or(0.0);
            println!(
                "SNAP  best={}  score={:+.3}  arena_sh={:.3}  synth_sh={:.3}  baseline_sh={:.3}  experiments={}  {}",
                name, score, arena, synth, baseline_sh, arr.len(), now
            );
        }
    } else if let Ok(obj) = serde_json::from_str::<Value>(&content) {
        // Backtest summary object
        let summary = &obj["summary"];
        if !summary.is_null() {
            let trades = summary["n_trades"].as_i64().unwrap_or(0);
            let wells = summary["n_wells"].as_i64().unwrap_or(0);
            let pnl = summary["total_pnl"].as_f64().unwrap_or(0.0);
            let ret = summary["total_return_pct"].as_f64().unwrap_or(0.0);
            let sharpe = summary["sharpe"].as_f64().unwrap_or(0.0);
            let dd = summary["max_dd_pct"].as_f64().unwrap_or(0.0);
            let conv_edge = summary["conv_edge_pct"].as_f64().unwrap_or(0.0);
            println!(
                "SNAP  trades={}  wells={}  net_pnl=${:.0}  return={:.1}%  sharpe={:.3}  dd={:.1}%  conv_edge={:.1}%  {}",
                trades, wells, pnl, ret, sharpe, dd, conv_edge, now
            );
        } else {
            println!("SNAP  file={}  {}", json_path, now);
        }
    }

    Ok(())
}
