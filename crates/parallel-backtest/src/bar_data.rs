use anyhow::{Context, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::path::Path;

/// A single OHLCV bar for one symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarData {
    pub symbol: String,
    /// Unix timestamp in seconds.
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// All bars keyed by symbol, sorted ascending by timestamp.
pub type DataStore = HashMap<String, Vec<BarData>>;

/// Load one CSV file. Expected columns (flexible order):
///   symbol, timestamp, open, high, low, close, volume
///
/// Falls back to positional columns if header is not present.
/// Uses memory-mapped I/O for large files.
pub fn load_csv(path: &Path) -> Result<DataStore> {
    let file = File::open(path)
        .with_context(|| format!("opening {:?}", path))?;

    // Memory-map the file for fast sequential reads.
    let mmap = unsafe { Mmap::map(&file).with_context(|| format!("mmap {:?}", path))? };
    let mut store: DataStore = HashMap::new();

    let mut lines = mmap.split(|&b| b == b'\n');

    // Parse header.
    let header_line = match lines.next() {
        Some(h) => std::str::from_utf8(h).unwrap_or("").trim().to_ascii_lowercase(),
        None => return Ok(store),
    };

    let cols: Vec<&str> = header_line.split(',').collect();
    let idx = |name: &str| cols.iter().position(|&c| c == name);

    let i_sym = idx("symbol");
    let i_ts = idx("timestamp").or_else(|| idx("time")).or_else(|| idx("date"));
    let i_open = idx("open");
    let i_high = idx("high");
    let i_low = idx("low");
    let i_close = idx("close");
    let i_vol = idx("volume").or_else(|| idx("vol"));

    // Derive symbol from filename if column absent.
    let file_symbol = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("UNKNOWN")
        .to_uppercase();

    for raw_line in lines {
        let line = match std::str::from_utf8(raw_line) {
            Ok(s) => s.trim(),
            Err(_) => continue,
        };
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 2 {
            continue;
        }

        macro_rules! parse_f64 {
            ($idx:expr, $pos:expr) => {
                $idx.and_then(|i| fields.get(i))
                    .or_else(|| fields.get($pos))
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .unwrap_or(0.0)
            };
        }

        macro_rules! parse_i64 {
            ($idx:expr, $pos:expr) => {
                $idx.and_then(|i| fields.get(i))
                    .or_else(|| fields.get($pos))
                    .and_then(|s| s.trim().parse::<i64>().ok())
                    .unwrap_or(0)
            };
        }

        let symbol = i_sym
            .and_then(|i| fields.get(i))
            .map(|s| s.trim().to_uppercase())
            .unwrap_or_else(|| file_symbol.clone());

        let timestamp = parse_i64!(i_ts, 1);
        let open = parse_f64!(i_open, 2);
        let high = parse_f64!(i_high, 3);
        let low = parse_f64!(i_low, 4);
        let close = parse_f64!(i_close, 5);
        let volume = parse_f64!(i_vol, 6);

        store
            .entry(symbol.clone())
            .or_default()
            .push(BarData { symbol, timestamp, open, high, low, close, volume });
    }

    // Sort each symbol's bars by timestamp ascending.
    for bars in store.values_mut() {
        bars.sort_unstable_by_key(|b| b.timestamp);
    }

    Ok(store)
}

/// Load all CSV files from a directory, merging into one DataStore.
pub fn load_csv_dir(dir: &Path) -> Result<DataStore> {
    let mut merged: DataStore = HashMap::new();

    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("reading directory {:?}", dir))?;

    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("csv") {
            match load_csv(&p) {
                Ok(store) => {
                    for (sym, bars) in store {
                        merged.entry(sym).or_default().extend(bars);
                    }
                }
                Err(e) => eprintln!("Warning: skipping {:?}: {}", p, e),
            }
        }
    }

    // Re-sort after merge.
    for bars in merged.values_mut() {
        bars.sort_unstable_by_key(|b| b.timestamp);
    }

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_csv_basic() {
        let mut f = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(f, "symbol,timestamp,open,high,low,close,volume").unwrap();
        writeln!(f, "BTC,1000000,50000,51000,49000,50500,100.5").unwrap();
        writeln!(f, "BTC,1000060,50500,52000,50000,51000,200.0").unwrap();
        writeln!(f, "ETH,1000000,3000,3100,2900,3050,500.0").unwrap();

        let store = load_csv(f.path()).unwrap();
        assert_eq!(store["BTC"].len(), 2);
        assert_eq!(store["ETH"].len(), 1);
        assert!((store["BTC"][0].close - 50500.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_csv_no_symbol_column() {
        let mut f = NamedTempFile::with_suffix(".csv").unwrap();
        // Name file as "MYTOKEN.csv" — symbol comes from filename.
        let path = f.path().with_file_name("MYTOKEN.csv");
        let mut f2 = std::fs::File::create(&path).unwrap();
        writeln!(f2, "timestamp,open,high,low,close,volume").unwrap();
        writeln!(f2, "1000000,100,110,90,105,1000").unwrap();
        drop(f2);

        let store = load_csv(&path).unwrap();
        assert!(store.contains_key("MYTOKEN"), "expected MYTOKEN key, got {:?}", store.keys().collect::<Vec<_>>());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_bars_sorted_ascending() {
        let mut f = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(f, "symbol,timestamp,open,high,low,close,volume").unwrap();
        writeln!(f, "BTC,1000120,50500,52000,50000,51000,200.0").unwrap();
        writeln!(f, "BTC,1000000,50000,51000,49000,50500,100.5").unwrap();

        let store = load_csv(f.path()).unwrap();
        let bars = &store["BTC"];
        assert!(bars[0].timestamp < bars[1].timestamp);
    }
}
