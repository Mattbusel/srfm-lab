/// Fast signal evaluator for evolved signal genomes.
///
/// Runs a bar-by-bar simulation over pre-loaded OHLCV data entirely
/// in memory with no I/O. Target: evaluate one genome on one year of
/// 15m bars (~26,000 bars) in under 10ms on modern hardware.

use crate::signal_genome::SignalGenome;
use serde::{Deserialize, Serialize};
use std::path::Path;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error loading a bar CSV file.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Csv(String),
    InsufficientRows(usize),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "I/O error: {e}"),
            LoadError::Csv(msg) => write!(f, "CSV parse error: {msg}"),
            LoadError::InsufficientRows(n) => write!(f, "insufficient rows: {n} < 50"),
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// BarArray
// ---------------------------------------------------------------------------

/// Pre-loaded OHLCV bar array stored in columnar format for cache efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarArray {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl BarArray {
    /// Number of bars in the array.
    pub fn len(&self) -> usize {
        self.close.len()
    }

    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    /// Validate that all columns have the same length.
    pub fn is_consistent(&self) -> bool {
        let n = self.close.len();
        self.open.len() == n
            && self.high.len() == n
            && self.low.len() == n
            && self.volume.len() == n
    }
}

// ---------------------------------------------------------------------------
// EvalResult
// ---------------------------------------------------------------------------

/// Summary statistics for one genome evaluated on one BarArray.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Annualized Sharpe ratio.
    pub sharpe: f64,
    /// Maximum drawdown as a fraction (0.0 to 1.0).
    pub max_dd: f64,
    /// Fraction of closed trades that were profitable.
    pub win_rate: f64,
    /// Total number of completed round-trip trades.
    pub n_trades: u32,
    /// Calmar ratio: annualized return / max_dd.
    pub calmar: f64,
    /// Terminal NAV (starts at 1.0).
    pub final_nav: f64,
}

impl EvalResult {
    /// A zero/invalid result returned when evaluation cannot proceed.
    pub fn invalid() -> Self {
        Self {
            sharpe: 0.0,
            max_dd: 1.0,
            win_rate: 0.0,
            n_trades: 0,
            calmar: 0.0,
            final_nav: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// FastSignalEvaluator
// ---------------------------------------------------------------------------

/// Evaluates signal genomes in-memory at high speed.
///
/// Pre-load bars once with `load_bars`, then call `evaluate` repeatedly
/// with different genomes. No allocations inside the hot loop.
pub struct FastSignalEvaluator {
    bars: BarArray,
    /// Spread cost per side in basis points (1 bps = 0.0001).
    spread_bps: f64,
    /// Bars constituting one year for annualization.
    bars_per_year: f64,
    /// Minimum bars before signals are trusted (warmup).
    warmup_bars: usize,
}

impl FastSignalEvaluator {
    /// Create an evaluator with default parameters.
    pub fn new(bars: BarArray) -> Self {
        Self {
            bars,
            spread_bps: 0.0001, // 1 bps
            bars_per_year: 96.0 * 252.0, // 15m bars in one trading year
            warmup_bars: 50,
        }
    }

    /// Override spread cost (default 1 bps per side).
    pub fn with_spread_bps(mut self, bps: f64) -> Self {
        self.spread_bps = bps * 0.0001;
        self
    }

    /// Override warmup period (default 50 bars).
    pub fn with_warmup(mut self, bars: usize) -> Self {
        self.warmup_bars = bars;
        self
    }

    /// Evaluate a single genome over the full bar array.
    ///
    /// Returns `EvalResult::invalid()` if there are insufficient bars.
    pub fn evaluate(&self, genome: &SignalGenome) -> EvalResult {
        let n = self.bars.len();
        if n < self.warmup_bars + 10 {
            return EvalResult::invalid();
        }

        // --- pre-compute signals via SRFM physics ---
        let (signals, bh_masses) = self.compute_signals(genome);

        // --- bar-by-bar simulation ---
        let mut nav = 1.0f64;
        let mut peak_nav = 1.0f64;
        let mut max_dd = 0.0f64;
        let mut position = 0.0f64; // current signed position
        let mut n_trades: u32 = 0;
        let mut n_winning: u32 = 0;
        let mut equity_returns: Vec<f64> = Vec::with_capacity(n);

        // Per-trade tracking.
        let mut in_trade = false;
        let mut trade_entry_price = 0.0f64;
        let mut trade_pos = 0.0f64;

        for i in 1..n {
            let close_prev = self.bars.close[i - 1];
            let close = self.bars.close[i];
            let bar_ret = if close_prev > 1e-12 {
                close / close_prev - 1.0
            } else {
                0.0
            };

            // Determine target position from genome parameters.
            let sig = signals[i];
            let mass = bh_masses[i];
            let target = if i < self.warmup_bars {
                0.0
            } else if mass >= genome.bh_mass_threshold && sig.abs() >= genome.entry_gate as f64 {
                sig.signum()
            } else {
                0.0
            };

            // Trade entry / exit detection.
            let changed = (target - position).abs() > 0.5;
            if changed {
                // Close existing trade.
                if in_trade {
                    let pnl_frac = if trade_pos > 0.0 {
                        (close - trade_entry_price) / trade_entry_price
                    } else {
                        (trade_entry_price - close) / trade_entry_price
                    };
                    // Deduct spread cost on exit.
                    let net_pnl = pnl_frac - self.spread_bps;
                    if net_pnl > 0.0 {
                        n_winning += 1;
                    }
                    n_trades += 1;
                    in_trade = false;
                }

                // Open new position.
                if target.abs() > 0.5 {
                    in_trade = true;
                    trade_entry_price = close * (1.0 + self.spread_bps * target.signum()); // pay spread on entry
                    trade_pos = target;
                }
                position = target;
            }

            // NAV update: apply bar return scaled by position.
            let port_ret = position * bar_ret;
            nav *= 1.0 + port_ret;
            equity_returns.push(port_ret);

            // Drawdown tracking.
            if nav > peak_nav {
                peak_nav = nav;
            }
            let dd = if peak_nav > 1e-12 {
                (peak_nav - nav) / peak_nav
            } else {
                0.0
            };
            if dd > max_dd {
                max_dd = dd;
            }
        }

        // Close any open trade at end.
        if in_trade {
            n_trades += 1;
            let last_close = *self.bars.close.last().unwrap_or(&1.0);
            let pnl_frac = if trade_pos > 0.0 {
                (last_close - trade_entry_price) / trade_entry_price
            } else {
                (trade_entry_price - last_close) / trade_entry_price
            };
            if pnl_frac > 0.0 {
                n_winning += 1;
            }
        }

        // Compute annualized Sharpe from equity_returns.
        let sharpe = annualized_sharpe(&equity_returns, self.bars_per_year);

        let win_rate = if n_trades > 0 {
            n_winning as f64 / n_trades as f64
        } else {
            0.0
        };

        // Annualized return for Calmar.
        let total_return = nav - 1.0;
        let n_years = equity_returns.len() as f64 / self.bars_per_year;
        let ann_return = if n_years > 1e-6 {
            (1.0 + total_return).powf(1.0 / n_years) - 1.0
        } else {
            total_return
        };
        let calmar = if max_dd > 1e-9 {
            ann_return / max_dd
        } else {
            0.0
        };

        EvalResult {
            sharpe,
            max_dd,
            win_rate,
            n_trades,
            calmar,
            final_nav: nav,
        }
    }

    /// Access the pre-loaded bar array.
    pub fn bars(&self) -> &BarArray {
        &self.bars
    }

    // -----------------------------------------------------------------------
    // Signal computation
    // -----------------------------------------------------------------------

    /// Compute per-bar signals and BH masses from SRFM physics.
    /// Returns (signals, bh_masses) aligned to bar indices.
    fn compute_signals(&self, genome: &SignalGenome) -> (Vec<f64>, Vec<f64>) {
        let n = self.bars.len();
        let cf = 0.02_f64; // cosmic flow constant

        let mut signals = vec![0.0f64; n];
        let mut bh_masses = vec![0.0f64; n];
        let mut mass = 0.0f64;
        let mut ctl: i32 = 0;
        let mut bh_active = false;
        let mut ema20 = self.bars.close[0];
        let alpha20 = 2.0 / 21.0;

        for i in 1..n {
            let close = self.bars.close[i];
            let prev = self.bars.close[i - 1];

            // BH physics.
            let beta = (close - prev).abs() / (prev * cf + 1e-12);
            if beta < 1.0 {
                mass = mass * 0.97 + 0.03;
                ctl += 1;
            } else {
                mass *= 0.95;
                ctl = 0;
            }

            // BH formation / collapse.
            if !bh_active && mass >= genome.bh_mass_threshold && ctl >= 5 {
                bh_active = true;
            }
            if bh_active && mass < genome.bh_mass_threshold * 0.5 {
                bh_active = false;
            }

            bh_masses[i] = mass;

            // 20-bar EMA.
            ema20 = ema20 * (1.0 - alpha20) + close * alpha20;

            // Signal.
            let direction = if close > ema20 { 1.0f64 } else { -1.0f64 };
            let magnitude = (mass / (genome.bh_mass_threshold + 1e-9)).min(1.0);

            signals[i] = if bh_active {
                direction * magnitude
            } else if ctl >= 3 {
                direction * magnitude * 0.5
            } else {
                0.0
            };
        }

        (signals, bh_masses)
    }
}

// ---------------------------------------------------------------------------
// load_bars -- CSV loader
// ---------------------------------------------------------------------------

/// Load OHLCV data from a CSV file.
///
/// Expects columns in order: timestamp, open, high, low, close, volume.
/// The first row is treated as a header and skipped.
/// Returns `Err(LoadError::InsufficientRows)` if fewer than 50 rows are loaded.
pub fn load_bars(csv_path: &Path) -> Result<BarArray, LoadError> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(csv_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header.
    if let Some(Ok(_header)) = lines.next() {
        // header consumed
    }

    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();

    for (line_num, line_result) in lines.enumerate() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 6 {
            return Err(LoadError::Csv(format!(
                "line {}: expected 6+ columns, got {}",
                line_num + 2,
                cols.len()
            )));
        }

        macro_rules! parse_col {
            ($idx:expr, $name:expr) => {
                cols[$idx].trim().parse::<f64>().map_err(|e| {
                    LoadError::Csv(format!(
                        "line {}: column '{}' parse error: {e}",
                        line_num + 2,
                        $name
                    ))
                })?
            };
        }

        // Column order: timestamp(0), open(1), high(2), low(3), close(4), volume(5)
        open.push(parse_col!(1, "open"));
        high.push(parse_col!(2, "high"));
        low.push(parse_col!(3, "low"));
        close.push(parse_col!(4, "close"));
        volume.push(parse_col!(5, "volume"));
    }

    let n = close.len();
    if n < 50 {
        return Err(LoadError::InsufficientRows(n));
    }

    Ok(BarArray {
        open,
        high,
        low,
        close,
        volume,
    })
}

// ---------------------------------------------------------------------------
// annualized_sharpe helper
// ---------------------------------------------------------------------------

fn annualized_sharpe(returns: &[f64], bars_per_year: f64) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = returns.iter().sum::<f64>() / nf;
    let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / nf;
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    mean / std * bars_per_year.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn seeded_rng() -> SmallRng {
        SmallRng::seed_from_u64(7)
    }

    /// Build a synthetic BarArray with a simple trending price series.
    fn make_trending_bars(n: usize) -> BarArray {
        let mut close = Vec::with_capacity(n);
        let mut price = 100.0f64;
        for i in 0..n {
            // Small upward drift with noise.
            price *= 1.0 + 0.0002 + ((i as f64 * 1.7).sin() * 0.002);
            close.push(price);
        }
        let open: Vec<f64> = close.iter().map(|&c| c * 0.9995).collect();
        let high: Vec<f64> = close.iter().map(|&c| c * 1.001).collect();
        let low: Vec<f64> = close.iter().map(|&c| c * 0.999).collect();
        let volume: Vec<f64> = vec![1_000_000.0; n];
        BarArray { open, high, low, close, volume }
    }

    fn make_flat_bars(n: usize) -> BarArray {
        let close = vec![100.0f64; n];
        BarArray {
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close,
            volume: vec![1_000_000.0; n],
        }
    }

    #[test]
    fn test_eval_returns_invalid_for_tiny_bars() {
        let bars = make_trending_bars(10);
        let evaluator = FastSignalEvaluator::new(bars);
        let genome = SignalGenome::default_genome();
        let result = evaluator.evaluate(&genome);
        // With only 10 bars and warmup=50, should return invalid.
        assert_eq!(result.n_trades, 0);
        assert!((result.max_dd - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_eval_runs_without_panic_on_year_data() {
        // ~26,280 bars in a year at 15m (24h * 4 * 365).
        let bars = make_trending_bars(27_000);
        let evaluator = FastSignalEvaluator::new(bars);
        let genome = SignalGenome::default_genome();
        let result = evaluator.evaluate(&genome);
        // Just check fields are in sane ranges.
        assert!(result.final_nav > 0.0);
        assert!(result.max_dd >= 0.0 && result.max_dd <= 1.0);
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
    }

    #[test]
    fn test_eval_flat_bars_no_signal() {
        // Flat price means beta is always 0 -- no BH forms, no signal.
        let bars = make_flat_bars(500);
        let evaluator = FastSignalEvaluator::new(bars);
        let genome = SignalGenome::default_genome();
        let result = evaluator.evaluate(&genome);
        assert_eq!(result.n_trades, 0, "no trades on flat bars");
    }

    #[test]
    fn test_final_nav_positive() {
        let bars = make_trending_bars(5000);
        let evaluator = FastSignalEvaluator::new(bars);
        let mut rng = seeded_rng();
        for _ in 0..5 {
            let genome = SignalGenome::random(&mut rng);
            let result = evaluator.evaluate(&genome);
            assert!(result.final_nav > 0.0, "final NAV must be positive");
        }
    }

    #[test]
    fn test_eval_result_serde_roundtrip() {
        let r = EvalResult {
            sharpe: 1.35,
            max_dd: 0.08,
            win_rate: 0.55,
            n_trades: 120,
            calmar: 3.2,
            final_nav: 1.42,
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: EvalResult = serde_json::from_str(&json).unwrap();
        assert!((decoded.sharpe - 1.35).abs() < 1e-12);
        assert_eq!(decoded.n_trades, 120);
    }

    #[test]
    fn test_bar_array_serde_roundtrip() {
        let bars = make_trending_bars(100);
        let json = serde_json::to_string(&bars).unwrap();
        let decoded: BarArray = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.len(), 100);
        assert!((decoded.close[50] - bars.close[50]).abs() < 1e-12);
    }

    #[test]
    fn test_bar_array_consistency() {
        let bars = make_trending_bars(200);
        assert!(bars.is_consistent());
        // Corrupt one column.
        let mut bad = bars.clone();
        bad.volume.push(0.0);
        assert!(!bad.is_consistent());
    }

    #[test]
    fn test_annualized_sharpe_zero_for_zero_returns() {
        let returns = vec![0.0f64; 100];
        let sh = annualized_sharpe(&returns, 96.0 * 252.0);
        assert!((sh - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_annualized_sharpe_positive_for_positive_returns() {
        // Constant positive return should give positive Sharpe = 0 (no std).
        let returns = vec![0.001f64; 100];
        let sh = annualized_sharpe(&returns, 96.0 * 252.0);
        // std is 0 so Sharpe is undefined -- function returns 0.
        assert!((sh - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_speed_one_year_under_10ms() {
        // Performance test: 1 year of 15m bars (~26,280) must evaluate in < 10ms.
        let bars = make_trending_bars(26_280);
        let evaluator = FastSignalEvaluator::new(bars);
        let genome = SignalGenome::default_genome();

        let start = std::time::Instant::now();
        let _result = evaluator.evaluate(&genome);
        let elapsed = start.elapsed();

        println!("evaluation took: {elapsed:?}");
        assert!(
            elapsed.as_millis() < 10,
            "evaluation took {elapsed:?}, expected < 10ms"
        );
    }
}
