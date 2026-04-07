/// Aggregates tick/trade events into OHLCV bars at multiple timeframes
/// simultaneously.
///
/// Supports: 1m, 5m, 15m, 1h, 4h, 1d.
/// Bars close at clean UTC boundaries (e.g., 15:00, 15:15, 15:30, ...).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

// ── Timeframe ─────────────────────────────────────────────────────────────────

/// Supported aggregation timeframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    /// 1-minute bars.
    Min1,
    /// 5-minute bars.
    Min5,
    /// 15-minute bars.
    Min15,
    /// 1-hour bars.
    Hour1,
    /// 4-hour bars.
    Hour4,
    /// Daily bars (UTC midnight boundary).
    Daily,
}

impl Timeframe {
    /// Duration of this timeframe in seconds.
    pub fn seconds(&self) -> i64 {
        match self {
            Timeframe::Min1 => 60,
            Timeframe::Min5 => 300,
            Timeframe::Min15 => 900,
            Timeframe::Hour1 => 3_600,
            Timeframe::Hour4 => 14_400,
            Timeframe::Daily => 86_400,
        }
    }

    /// Bar-open timestamp (seconds since Unix epoch) for a given tick timestamp.
    ///
    /// Truncates `ts_sec` to the nearest clean boundary.
    pub fn bar_open_ts(&self, ts_sec: i64) -> i64 {
        let period = self.seconds();
        (ts_sec / period) * period
    }

    /// Bar-close timestamp (exclusive) for the bar that contains `ts_sec`.
    pub fn bar_close_ts(&self, ts_sec: i64) -> i64 {
        self.bar_open_ts(ts_sec) + self.seconds()
    }
}

impl std::fmt::Display for Timeframe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Timeframe::Min1 => write!(f, "1m"),
            Timeframe::Min5 => write!(f, "5m"),
            Timeframe::Min15 => write!(f, "15m"),
            Timeframe::Hour1 => write!(f, "1h"),
            Timeframe::Hour4 => write!(f, "4h"),
            Timeframe::Daily => write!(f, "1d"),
        }
    }
}

// ── Tick ──────────────────────────────────────────────────────────────────────

/// A single trade/tick event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    /// Symbol identifier.
    pub symbol: String,
    /// UTC timestamp in seconds since Unix epoch.
    pub timestamp_sec: i64,
    /// Trade price.
    pub price: f64,
    /// Trade quantity.
    pub quantity: f64,
}

impl Tick {
    pub fn new(symbol: impl Into<String>, ts: i64, price: f64, qty: f64) -> Self {
        Tick { symbol: symbol.into(), timestamp_sec: ts, price, quantity: qty }
    }
}

// ── CompletedBar ──────────────────────────────────────────────────────────────

/// A fully formed OHLCV bar for a specific symbol and timeframe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedBar {
    pub symbol: String,
    pub timeframe: Timeframe,
    /// UTC seconds of bar open (clean boundary).
    pub open_ts: i64,
    /// UTC seconds of bar close (exclusive).
    pub close_ts: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// Number of ticks that contributed to this bar.
    pub tick_count: u32,
}

// ── InProgressBar ─────────────────────────────────────────────────────────────

/// Mutable OHLCV accumulator for an open bar.
#[derive(Debug, Clone)]
struct InProgressBar {
    open_ts: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    tick_count: u32,
}

impl InProgressBar {
    fn new(open_ts: i64, price: f64, qty: f64) -> Self {
        InProgressBar {
            open_ts,
            open: price,
            high: price,
            low: price,
            close: price,
            volume: qty,
            tick_count: 1,
        }
    }

    fn update(&mut self, price: f64, qty: f64) {
        if price > self.high {
            self.high = price;
        }
        if price < self.low {
            self.low = price;
        }
        self.close = price;
        self.volume += qty;
        self.tick_count += 1;
    }

    fn complete(&self, symbol: &str, tf: Timeframe) -> CompletedBar {
        CompletedBar {
            symbol: symbol.to_string(),
            timeframe: tf,
            open_ts: self.open_ts,
            close_ts: self.open_ts + tf.seconds(),
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            tick_count: self.tick_count,
        }
    }
}

// ── SymbolAggregator ──────────────────────────────────────────────────────────

/// Per-symbol, per-timeframe bar accumulator.
struct SymbolAggregator {
    symbol: String,
    /// Current in-progress bars, keyed by timeframe.
    bars: HashMap<Timeframe, InProgressBar>,
    /// The set of timeframes to aggregate.
    timeframes: Vec<Timeframe>,
}

impl SymbolAggregator {
    fn new(symbol: String, timeframes: Vec<Timeframe>) -> Self {
        SymbolAggregator {
            symbol,
            bars: HashMap::new(),
            timeframes,
        }
    }

    /// Process a tick; return any newly completed bars.
    fn on_tick(&mut self, tick: &Tick) -> Vec<CompletedBar> {
        let ts = tick.timestamp_sec;
        let mut completed = Vec::new();

        for &tf in &self.timeframes {
            let bar_open = tf.bar_open_ts(ts);

            match self.bars.get_mut(&tf) {
                Some(bar) => {
                    if ts < bar.open_ts + tf.seconds() {
                        // Same bar -- just update.
                        bar.update(tick.price, tick.quantity);
                    } else {
                        // Tick belongs to a new bar -- close the old one.
                        let done = bar.complete(&self.symbol, tf);
                        completed.push(done);
                        // Open a new bar.
                        self.bars.insert(tf, InProgressBar::new(bar_open, tick.price, tick.quantity));
                    }
                }
                None => {
                    // First tick for this timeframe.
                    self.bars.insert(tf, InProgressBar::new(bar_open, tick.price, tick.quantity));
                }
            }
        }

        completed
    }

    /// Force-close all in-progress bars (e.g., end of session).
    fn flush(&mut self) -> Vec<CompletedBar> {
        let mut completed = Vec::new();
        for &tf in &self.timeframes {
            if let Some(bar) = self.bars.remove(&tf) {
                if bar.tick_count > 0 {
                    completed.push(bar.complete(&self.symbol, tf));
                }
            }
        }
        completed
    }

    /// Return the current in-progress bar for `tf`, if any.
    fn current_bar(&self, tf: Timeframe) -> Option<&InProgressBar> {
        self.bars.get(&tf)
    }
}

// ── BarAggregator ─────────────────────────────────────────────────────────────

/// Multi-symbol, multi-timeframe bar aggregator.
///
/// Thread-safe: uses an internal `Mutex<HashMap<symbol, SymbolAggregator>>`.
pub struct BarAggregator {
    /// Per-symbol aggregators, protected by a single mutex.
    inner: Mutex<HashMap<String, SymbolAggregator>>,
    /// Active timeframes.
    timeframes: Vec<Timeframe>,
}

impl BarAggregator {
    /// Create an aggregator for the given timeframes.
    pub fn new(timeframes: Vec<Timeframe>) -> Self {
        BarAggregator {
            inner: Mutex::new(HashMap::new()),
            timeframes,
        }
    }

    /// Create with all supported timeframes.
    pub fn all_timeframes() -> Self {
        Self::new(vec![
            Timeframe::Min1,
            Timeframe::Min5,
            Timeframe::Min15,
            Timeframe::Hour1,
            Timeframe::Hour4,
            Timeframe::Daily,
        ])
    }

    // ── on_tick ───────────────────────────────────────────────────────────

    /// Process a tick event.
    ///
    /// Returns a list of newly-completed bars (0, 1, or more, one per
    /// timeframe that just closed).
    pub fn on_tick(&self, tick: Tick) -> Vec<CompletedBar> {
        let mut guard = self.inner.lock().expect("aggregator mutex poisoned");
        let tfs = self.timeframes.clone();
        let agg = guard
            .entry(tick.symbol.clone())
            .or_insert_with(|| SymbolAggregator::new(tick.symbol.clone(), tfs));
        agg.on_tick(&tick)
    }

    /// Process a batch of ticks in order.
    ///
    /// Returns all completed bars across all ticks, in order.
    pub fn on_ticks(&self, ticks: Vec<Tick>) -> Vec<CompletedBar> {
        let mut completed = Vec::new();
        for tick in ticks {
            let mut bars = self.on_tick(tick);
            completed.append(&mut bars);
        }
        completed
    }

    // ── flush ─────────────────────────────────────────────────────────────

    /// Force-close all in-progress bars for all symbols.
    ///
    /// Call at end-of-session to retrieve partial bars.
    pub fn flush(&self) -> Vec<CompletedBar> {
        let mut guard = self.inner.lock().expect("aggregator mutex poisoned");
        let mut completed = Vec::new();
        for agg in guard.values_mut() {
            let mut bars = agg.flush();
            completed.append(&mut bars);
        }
        completed
    }

    /// Flush a specific symbol.
    pub fn flush_symbol(&self, symbol: &str) -> Vec<CompletedBar> {
        let mut guard = self.inner.lock().expect("aggregator mutex poisoned");
        if let Some(agg) = guard.get_mut(symbol) {
            agg.flush()
        } else {
            vec![]
        }
    }

    // ── Inspection ────────────────────────────────────────────────────────

    /// Return the current (in-progress) bar state for a symbol and timeframe.
    pub fn current_bar(&self, symbol: &str, tf: Timeframe) -> Option<CompletedBar> {
        let guard = self.inner.lock().expect("aggregator mutex poisoned");
        let agg = guard.get(symbol)?;
        let bar = agg.current_bar(tf)?;
        Some(bar.complete(symbol, tf))
    }

    /// List all currently tracked symbols.
    pub fn tracked_symbols(&self) -> Vec<String> {
        let guard = self.inner.lock().expect("aggregator mutex poisoned");
        let mut syms: Vec<String> = guard.keys().cloned().collect();
        syms.sort();
        syms
    }

    /// Number of symbols currently tracked.
    pub fn symbol_count(&self) -> usize {
        let guard = self.inner.lock().expect("aggregator mutex poisoned");
        guard.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sequence of ticks at 1-second intervals starting at `base_ts`.
    fn tick_seq(
        symbol: &str,
        base_ts: i64,
        count: usize,
        start_price: f64,
        price_step: f64,
    ) -> Vec<Tick> {
        (0..count)
            .map(|i| Tick::new(symbol, base_ts + i as i64, start_price + i as f64 * price_step, 1.0))
            .collect()
    }

    #[test]
    fn test_bar_open_ts_1m() {
        let tf = Timeframe::Min1;
        // Tick at 12:00:45 UTC => bar opens at 12:00:00.
        let ts = 12 * 3600 + 45; // 12:00:45
        assert_eq!(tf.bar_open_ts(ts), 12 * 3600);
    }

    #[test]
    fn test_bar_open_ts_15m() {
        let tf = Timeframe::Min15;
        // 12:17:00 => bar opens at 12:15:00.
        let ts = 12 * 3600 + 17 * 60;
        assert_eq!(tf.bar_open_ts(ts), 12 * 3600 + 15 * 60);
    }

    #[test]
    fn test_no_completed_bars_within_one_minute() {
        let agg = BarAggregator::new(vec![Timeframe::Min1]);
        let ticks = tick_seq("BTC", 0, 50, 100.0, 0.1);
        let completed: Vec<CompletedBar> = ticks.into_iter().flat_map(|t| agg.on_tick(t)).collect();
        // All 50 ticks fall within the same 1-minute bar => no completions.
        assert!(completed.is_empty(), "expected no completed bars, got {}", completed.len());
    }

    #[test]
    fn test_one_bar_completes_on_boundary_crossing() {
        let agg = BarAggregator::new(vec![Timeframe::Min1]);
        let base = 0i64;
        // 59 ticks in minute 0, then 1 tick in minute 1.
        let mut ticks = tick_seq("ETH", base, 59, 200.0, 0.1);
        ticks.push(Tick::new("ETH", base + 60, 210.0, 1.0)); // new minute
        let completed: Vec<CompletedBar> = ticks.into_iter().flat_map(|t| agg.on_tick(t)).collect();
        assert_eq!(completed.len(), 1, "expected 1 completed bar");
        let bar = &completed[0];
        assert_eq!(bar.timeframe, Timeframe::Min1);
        assert!((bar.open - 200.0).abs() < 1e-9);
        assert!((bar.close - (200.0 + 58.0 * 0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_timeframes_complete() {
        let agg = BarAggregator::new(vec![Timeframe::Min1, Timeframe::Min5]);
        let base = 0i64;
        // Push ticks spanning 6 minutes (360 seconds + 1).
        let mut ticks: Vec<Tick> = (0..=360)
            .map(|i| Tick::new("SOL", base + i as i64, 50.0 + i as f64 * 0.01, 1.0))
            .collect();
        ticks.push(Tick::new("SOL", base + 360, 53.6, 1.0));

        let completed: Vec<CompletedBar> = ticks.into_iter().flat_map(|t| agg.on_tick(t)).collect();

        let min1_bars: Vec<&CompletedBar> =
            completed.iter().filter(|b| b.timeframe == Timeframe::Min1).collect();
        let min5_bars: Vec<&CompletedBar> =
            completed.iter().filter(|b| b.timeframe == Timeframe::Min5).collect();

        assert!(min1_bars.len() >= 5, "expected >= 5 1m bars, got {}", min1_bars.len());
        assert!(min5_bars.len() >= 1, "expected >= 1 5m bar, got {}", min5_bars.len());
    }

    #[test]
    fn test_ohlcv_within_bar() {
        let agg = BarAggregator::new(vec![Timeframe::Min1]);
        // 10 ticks: prices 100, 105, 98, 102, ..., then a boundary crossing.
        let prices = [100.0, 105.0, 98.0, 102.0, 101.0, 103.0, 99.0, 104.0, 100.0, 101.5];
        let base = 0i64;
        for (i, &p) in prices.iter().enumerate() {
            agg.on_tick(Tick::new("BTC", base + i as i64, p, 10.0));
        }
        // Force complete.
        let completed = agg.on_tick(Tick::new("BTC", base + 60, 101.0, 10.0));
        assert_eq!(completed.len(), 1);
        let bar = &completed[0];
        assert!((bar.open - 100.0).abs() < 1e-9, "open mismatch");
        assert!((bar.high - 105.0).abs() < 1e-9, "high mismatch: {}", bar.high);
        assert!((bar.low - 98.0).abs() < 1e-9, "low mismatch: {}", bar.low);
        assert!((bar.close - 101.5).abs() < 1e-9, "close mismatch");
        assert!((bar.volume - 100.0).abs() < 1e-9, "volume mismatch: {}", bar.volume);
    }

    #[test]
    fn test_flush_returns_partial_bars() {
        let agg = BarAggregator::new(vec![Timeframe::Min1]);
        agg.on_tick(Tick::new("ADA", 0, 0.5, 1000.0));
        agg.on_tick(Tick::new("ADA", 5, 0.51, 500.0));
        let bars = agg.flush();
        assert!(!bars.is_empty(), "expected partial bars from flush");
        assert_eq!(bars[0].symbol, "ADA");
    }

    #[test]
    fn test_multi_symbol() {
        let agg = BarAggregator::all_timeframes();
        agg.on_tick(Tick::new("BTC", 0, 60_000.0, 0.1));
        agg.on_tick(Tick::new("ETH", 0, 3_000.0, 1.0));
        agg.on_tick(Tick::new("SOL", 0, 150.0, 10.0));
        assert_eq!(agg.symbol_count(), 3);
        let syms = agg.tracked_symbols();
        assert!(syms.contains(&"BTC".to_string()));
        assert!(syms.contains(&"ETH".to_string()));
        assert!(syms.contains(&"SOL".to_string()));
    }

    #[test]
    fn test_tick_count_in_bar() {
        let agg = BarAggregator::new(vec![Timeframe::Min1]);
        let n = 30usize;
        for i in 0..n {
            agg.on_tick(Tick::new("X", i as i64, 1.0, 1.0));
        }
        // Trigger completion.
        let bars = agg.on_tick(Tick::new("X", 60, 1.0, 1.0));
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].tick_count, n as u32);
    }

    #[test]
    fn test_daily_boundary() {
        let tf = Timeframe::Daily;
        // Tick at 2024-01-15 14:32:00 UTC = 86400 * 19737 + 14*3600 + 32*60
        let ts: i64 = 86_400 * 19_737 + 14 * 3_600 + 32 * 60;
        let bar_open = tf.bar_open_ts(ts);
        // Should be 00:00:00 on the same day.
        assert_eq!(bar_open % 86_400, 0);
        assert!(bar_open <= ts);
    }
}
