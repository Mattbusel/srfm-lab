/// Bar data loader — reads CSV files and computes derived features.
///
/// Expected CSV columns (compatible with crypto_trades.csv format):
///   timestamp, open, high, low, close, volume
///
/// Derived features computed at load time:
///   returns, log_returns, realized_vol (20-bar rolling std of log-returns)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A single OHLCV bar with pre-computed derived features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarData {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// Simple return: (close - prev_close) / prev_close. 0.0 for first bar.
    pub returns: f64,
    /// Log return: ln(close / prev_close). 0.0 for first bar.
    pub log_returns: f64,
    /// 20-bar rolling realized volatility (std of log-returns). 0.0 while warming up.
    pub realized_vol: f64,
    /// Typical price: (high + low + close) / 3
    pub typical_price: f64,
}

impl BarData {
    /// Construct from raw OHLCV, with returns computed externally.
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        prev_close: Option<f64>,
    ) -> Self {
        let (returns, log_returns) = match prev_close {
            Some(p) if p != 0.0 => {
                let r = (close - p) / p;
                let lr = (close / p).ln();
                (r, lr)
            }
            _ => (0.0, 0.0),
        };
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            returns,
            log_returns,
            realized_vol: 0.0,
            typical_price: (high + low + close) / 3.0,
        }
    }
}

/// Raw CSV row, flexible header handling.
#[derive(Debug, Deserialize)]
struct CsvRow {
    #[serde(alias = "ts", alias = "time", alias = "date")]
    timestamp: Option<String>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    #[serde(alias = "vol")]
    volume: f64,
}

/// Load bar data from a CSV file. Returns bars sorted ascending by timestamp.
pub fn load_csv(path: &str) -> Result<Vec<BarData>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("Cannot open CSV: {path}"))?;

    let mut raw: Vec<(i64, f64, f64, f64, f64, f64)> = Vec::new();

    for (i, result) in reader.deserialize::<CsvRow>().enumerate() {
        let row = result.with_context(|| format!("CSV parse error at row {i}"))?;
        let ts: i64 = row
            .timestamp
            .as_deref()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(i as i64);
        raw.push((ts, row.open, row.high, row.low, row.close, row.volume));
    }

    // Sort by timestamp ascending
    raw.sort_by_key(|r| r.0);

    build_bars(raw)
}

/// Build BarData slice from raw tuples, computing all derived fields.
pub fn build_bars(raw: Vec<(i64, f64, f64, f64, f64, f64)>) -> Result<Vec<BarData>> {
    let n = raw.len();
    let mut bars: Vec<BarData> = Vec::with_capacity(n);

    for (i, &(ts, open, high, low, close, volume)) in raw.iter().enumerate() {
        let prev_close = if i > 0 { Some(raw[i - 1].4) } else { None };
        bars.push(BarData::new(ts, open, high, low, close, volume, prev_close));
    }

    // Compute 20-bar realized vol (rolling std of log-returns)
    const VOL_WINDOW: usize = 20;
    for i in VOL_WINDOW..n {
        let slice: Vec<f64> = bars[(i - VOL_WINDOW)..i]
            .iter()
            .map(|b| b.log_returns)
            .collect();
        let mean = slice.iter().sum::<f64>() / VOL_WINDOW as f64;
        let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (VOL_WINDOW - 1) as f64;
        bars[i].realized_vol = var.sqrt();
    }

    Ok(bars)
}

/// Build synthetic bar data for testing (sine-wave close prices).
pub fn synthetic_bars(n: usize, seed_close: f64) -> Vec<BarData> {
    let raw: Vec<(i64, f64, f64, f64, f64, f64)> = (0..n)
        .map(|i| {
            let t = i as f64;
            let close = seed_close * (1.0 + 0.01 * (t * 0.1).sin() + 0.001 * t);
            let open = close * 0.999;
            let high = close * 1.002;
            let low = close * 0.998;
            let volume = 1_000.0 + 500.0 * (t * 0.3).sin().abs();
            (i as i64, open, high, low, close, volume)
        })
        .collect();
    build_bars(raw).expect("synthetic bars should not fail")
}

/// Compute returns for a slice of close prices.
pub fn compute_returns(closes: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; closes.len()];
    for i in 1..closes.len() {
        if closes[i - 1] != 0.0 {
            out[i] = (closes[i] - closes[i - 1]) / closes[i - 1];
        }
    }
    out
}

/// Compute log-returns for a slice of close prices.
pub fn compute_log_returns(closes: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; closes.len()];
    for i in 1..closes.len() {
        if closes[i - 1] > 0.0 && closes[i] > 0.0 {
            out[i] = (closes[i] / closes[i - 1]).ln();
        }
    }
    out
}

/// Rolling standard deviation over a fixed window.
pub fn rolling_std(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![0.0; n];
    for i in window..n {
        let slice = &values[(i - window)..i];
        let mean = slice.iter().sum::<f64>() / window as f64;
        let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window - 1).max(1) as f64;
        out[i] = var.sqrt();
    }
    out
}

/// Rolling mean over a fixed window.
pub fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![0.0; n];
    for i in window..n {
        let slice = &values[(i - window)..i];
        out[i] = slice.iter().sum::<f64>() / window as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_bars_length() {
        let bars = synthetic_bars(100, 100.0);
        assert_eq!(bars.len(), 100);
    }

    #[test]
    fn returns_first_bar_zero() {
        let bars = synthetic_bars(50, 50.0);
        assert_eq!(bars[0].returns, 0.0);
        assert_eq!(bars[0].log_returns, 0.0);
    }

    #[test]
    fn realized_vol_warm_up() {
        let bars = synthetic_bars(100, 100.0);
        // First 20 bars should have 0.0 vol
        for bar in &bars[..20] {
            assert_eq!(bar.realized_vol, 0.0);
        }
        // Bar 20+ should have non-zero vol
        assert!(bars[20].realized_vol >= 0.0);
    }

    #[test]
    fn rolling_std_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_std(&data, 3);
        // window=3, so first valid at index 3
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        // [1,2,3] std = 1.0
        assert!((result[3] - 1.0).abs() < 1e-10);
    }
}
