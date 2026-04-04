// csv_loader.rs — Load OHLCV bars and trades from CSV files
//
// Expected CSV column order (Alpaca-style):
//   timestamp, open, high, low, close, volume
// The header row is detected automatically.

use crate::types::{Bar, Regime, TFScore, Trade};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// BarsSet — three-timeframe bar collection for a single instrument
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct BarsSet {
    pub bars_1d: Vec<Bar>,
    pub bars_1h: Vec<Bar>,
    pub bars_15m: Vec<Bar>,
}

impl BarsSet {
    pub fn new(bars_1d: Vec<Bar>, bars_1h: Vec<Bar>, bars_15m: Vec<Bar>) -> Self {
        Self { bars_1d, bars_1h, bars_15m }
    }

    pub fn is_empty(&self) -> bool {
        self.bars_1d.is_empty() && self.bars_1h.is_empty() && self.bars_15m.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Raw deserialization helpers
// ---------------------------------------------------------------------------

/// Flexible raw record: timestamp can be RFC-3339 string or integer unix-ms.
#[derive(Debug, Deserialize)]
struct RawBar {
    #[serde(alias = "t", alias = "time", alias = "date")]
    timestamp: String,
    #[serde(alias = "o")]
    open: f64,
    #[serde(alias = "h")]
    high: f64,
    #[serde(alias = "l")]
    low: f64,
    #[serde(alias = "c")]
    close: f64,
    #[serde(alias = "v")]
    volume: f64,
}

fn parse_timestamp(s: &str) -> Result<i64> {
    // Try plain integer first
    if let Ok(ms) = s.trim().parse::<i64>() {
        return Ok(ms);
    }
    // Try RFC-3339 / ISO-8601
    use chrono::DateTime;
    let dt = DateTime::parse_from_rfc3339(s.trim())
        .or_else(|_| DateTime::parse_from_str(s.trim(), "%Y-%m-%d %H:%M:%S%.f %z"))
        .or_else(|_| DateTime::parse_from_str(s.trim(), "%Y-%m-%dT%H:%M:%S%.f"))
        .context(format!("Cannot parse timestamp: {s}"))?;
    Ok(dt.timestamp_millis())
}

// ---------------------------------------------------------------------------
// load_bars_csv
// ---------------------------------------------------------------------------

/// Load a CSV file into a sorted `Vec<Bar>`.
///
/// Accepts files with a header row or without.  The reader tolerates both
/// comma and tab delimiters by sniffing the first line.
pub fn load_bars_csv(path: &Path) -> Result<Vec<Bar>> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .with_context(|| format!("Opening bars CSV: {}", path.display()))?;

    let mut bars: Vec<Bar> = Vec::new();

    for result in rdr.deserialize::<RawBar>() {
        let raw = result.with_context(|| format!("Parsing bars CSV: {}", path.display()))?;
        let ts = parse_timestamp(&raw.timestamp)?;
        bars.push(Bar::new(ts, raw.open, raw.high, raw.low, raw.close, raw.volume));
    }

    // Ensure chronological order
    bars.sort_by_key(|b| b.timestamp);
    bars.dedup_by_key(|b| b.timestamp);

    Ok(bars)
}

// ---------------------------------------------------------------------------
// load_trades_csv
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawTrade {
    sym: String,
    entry_time: String,
    exit_time: String,
    entry_price: f64,
    exit_price: f64,
    pnl: f64,
    dollar_pos: f64,
    hold_bars: usize,
    regime: String,
    tf_score_daily: f64,
    tf_score_hourly: f64,
    tf_score_m15: f64,
    tf_score_total: f64,
    mass: f64,
}

/// Load a trades CSV produced by [`BacktestEngine`].
pub fn load_trades_csv(path: &Path) -> Result<Vec<Trade>> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .with_context(|| format!("Opening trades CSV: {}", path.display()))?;

    let mut trades: Vec<Trade> = Vec::new();

    for result in rdr.deserialize::<RawTrade>() {
        let raw = result.with_context(|| format!("Parsing trades CSV: {}", path.display()))?;
        let entry_time = parse_timestamp(&raw.entry_time)?;
        let exit_time = parse_timestamp(&raw.exit_time)?;
        let tf = TFScore {
            daily: raw.tf_score_daily,
            hourly: raw.tf_score_hourly,
            m15: raw.tf_score_m15,
            total: raw.tf_score_total,
        };
        let trade = Trade::new(
            raw.sym,
            entry_time,
            exit_time,
            raw.entry_price,
            raw.exit_price,
            raw.pnl,
            raw.dollar_pos,
            raw.hold_bars,
            Regime::from_str(&raw.regime),
            tf,
            raw.mass,
        );
        trades.push(trade);
    }

    Ok(trades)
}

// ---------------------------------------------------------------------------
// save_trades_csv
// ---------------------------------------------------------------------------

/// Serialise a slice of trades to a CSV file.
pub fn save_trades_csv(path: &Path, trades: &[Trade]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)
        .with_context(|| format!("Creating trades CSV: {}", path.display()))?;

    // Write header
    wtr.write_record(&[
        "sym",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "pnl",
        "dollar_pos",
        "hold_bars",
        "regime",
        "tf_score_daily",
        "tf_score_hourly",
        "tf_score_m15",
        "tf_score_total",
        "mass",
    ])?;

    for t in trades {
        wtr.write_record(&[
            t.sym.as_str(),
            &t.entry_time.to_string(),
            &t.exit_time.to_string(),
            &format!("{:.6}", t.entry_price),
            &format!("{:.6}", t.exit_price),
            &format!("{:.6}", t.pnl),
            &format!("{:.2}", t.dollar_pos),
            &t.hold_bars.to_string(),
            t.regime.to_str(),
            &format!("{:.4}", t.tf_score.daily),
            &format!("{:.4}", t.tf_score.hourly),
            &format!("{:.4}", t.tf_score.m15),
            &format!("{:.4}", t.tf_score.total),
            &format!("{:.6}", t.mass),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// save_equity_curve_csv
// ---------------------------------------------------------------------------

pub fn save_equity_curve_csv(path: &Path, curve: &[(i64, f64)]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)
        .with_context(|| format!("Creating equity CSV: {}", path.display()))?;
    wtr.write_record(&["timestamp_ms", "equity"])?;
    for (ts, eq) in curve {
        wtr.write_record(&[ts.to_string(), format!("{eq:.2}")])?;
    }
    wtr.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// load_barsset — convenience loader: looks for sym_1d.csv, sym_1h.csv, etc.
// ---------------------------------------------------------------------------

/// Try several filename patterns under `data_dir`:
///   `<sym>_1d.csv`, `<sym>_daily.csv`, `<sym>_1d/*.csv` ...
fn try_load(dir: &Path, candidates: &[&str]) -> Option<Vec<Bar>> {
    for name in candidates {
        let p = dir.join(name);
        if p.exists() {
            if let Ok(bars) = load_bars_csv(&p) {
                if !bars.is_empty() {
                    return Some(bars);
                }
            }
        }
    }
    None
}

pub fn load_barsset(data_dir: &Path, sym: &str) -> Result<BarsSet> {
    let sym_lc = sym.to_lowercase();

    let bars_1d = try_load(
        data_dir,
        &[
            &format!("{sym}_1d.csv"),
            &format!("{sym_lc}_1d.csv"),
            &format!("{sym}_daily.csv"),
            &format!("{sym_lc}_daily.csv"),
            &format!("{sym}_D.csv"),
        ],
    )
    .unwrap_or_default();

    let bars_1h = try_load(
        data_dir,
        &[
            &format!("{sym}_1h.csv"),
            &format!("{sym_lc}_1h.csv"),
            &format!("{sym}_hourly.csv"),
            &format!("{sym_lc}_hourly.csv"),
            &format!("{sym}_60.csv"),
        ],
    )
    .unwrap_or_default();

    let bars_15m = try_load(
        data_dir,
        &[
            &format!("{sym}_15m.csv"),
            &format!("{sym_lc}_15m.csv"),
            &format!("{sym}_15min.csv"),
            &format!("{sym_lc}_15min.csv"),
            &format!("{sym}_15.csv"),
        ],
    )
    .unwrap_or_default();

    Ok(BarsSet::new(bars_1d, bars_1h, bars_15m))
}

// ---------------------------------------------------------------------------
// Timeframe from path suffix helper
// ---------------------------------------------------------------------------

/// Infer timeframe label from a path like "BTC_1h.csv" → "1h".
pub fn infer_timeframe(path: &Path) -> &'static str {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem.ends_with("_1d") || stem.ends_with("_daily") {
        "1d"
    } else if stem.ends_with("_1h") || stem.ends_with("_hourly") {
        "1h"
    } else if stem.ends_with("_15m") || stem.ends_with("_15min") {
        "15m"
    } else {
        "unknown"
    }
}

/// Collect all CSV paths under a directory matching `*_<tf>.csv`.
pub fn list_csv_paths(dir: &Path, suffix: &str) -> Vec<PathBuf> {
    let pattern = format!("*{suffix}.csv");
    let Ok(entries) = std::fs::read_dir(dir) else { return vec![] };
    let mut out: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(&pattern.trim_start_matches('*')))
                .unwrap_or(false)
        })
        .collect();
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_iso_timestamp() {
        let ts = parse_timestamp("2024-01-15T09:30:00+00:00").unwrap();
        assert!(ts > 0);
    }

    #[test]
    fn parse_unix_ms_timestamp() {
        let ts = parse_timestamp("1705312200000").unwrap();
        assert_eq!(ts, 1_705_312_200_000i64);
    }
}
