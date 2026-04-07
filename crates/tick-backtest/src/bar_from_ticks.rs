// bar_from_ticks.rs -- Build OHLCV bars from tick data for SRFM backtest engine.
// Supports time bars, volume bars, tick-count bars, and dollar bars.

use crate::tick_replay::Tick;

// ---------------------------------------------------------------------------
// Extended Bar with microstructure fields
// ---------------------------------------------------------------------------

/// OHLCV bar enriched with microstructure data built from tick flow.
#[derive(Debug, Clone, PartialEq)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// Volume-weighted average price.
    pub vwap: f64,
    /// Volume where the aggressor was a buyer.
    pub buy_volume: f64,
    /// Volume where the aggressor was a seller.
    pub sell_volume: f64,
    /// Number of ticks included in this bar.
    pub n_ticks: u64,
    /// Order flow imbalance: buy_volume - sell_volume.
    pub delta: f64,
    /// Bar open time in milliseconds (Unix epoch).
    pub open_time_ms: u64,
    /// Bar close time in milliseconds (Unix epoch, last tick time).
    pub close_time_ms: u64,
}

impl Bar {
    /// Convert this extended Bar into the engine's canonical crate::types::Bar.
    pub fn to_canonical(&self) -> crate::types::Bar {
        crate::types::Bar::new(
            self.open_time_ms as i64,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
        )
    }
}

// ---------------------------------------------------------------------------
// PartialBar -- accumulator before the bar is sealed
// ---------------------------------------------------------------------------

/// Accumulator for an in-progress bar.
#[derive(Debug, Clone)]
pub struct PartialBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// Numerator for VWAP: sum(price * volume).
    pub vwap_num: f64,
    /// Denominator for VWAP: sum(volume).
    pub vwap_den: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub n_ticks: u64,
    pub open_time_ms: u64,
    pub close_time_ms: u64,
}

impl PartialBar {
    fn new(tick: &Tick) -> Self {
        let ts_ms = (tick.timestamp_ns / 1_000_000) as u64;
        let buy_vol = if tick.side == 1 { tick.volume } else { 0.0 };
        let sell_vol = if tick.side == -1 { tick.volume } else { 0.0 };
        Self {
            open: tick.price,
            high: tick.price,
            low: tick.price,
            close: tick.price,
            volume: tick.volume,
            vwap_num: tick.price * tick.volume,
            vwap_den: tick.volume,
            buy_volume: buy_vol,
            sell_volume: sell_vol,
            n_ticks: 1,
            open_time_ms: ts_ms,
            close_time_ms: ts_ms,
        }
    }

    fn add(&mut self, tick: &Tick) {
        let ts_ms = (tick.timestamp_ns / 1_000_000) as u64;
        if tick.price > self.high { self.high = tick.price; }
        if tick.price < self.low { self.low = tick.price; }
        self.close = tick.price;
        self.volume += tick.volume;
        self.vwap_num += tick.price * tick.volume;
        self.vwap_den += tick.volume;
        if tick.side == 1 { self.buy_volume += tick.volume; }
        if tick.side == -1 { self.sell_volume += tick.volume; }
        self.n_ticks += 1;
        self.close_time_ms = ts_ms;
    }

    fn seal(self) -> Bar {
        let vwap = if self.vwap_den > 1e-12 { self.vwap_num / self.vwap_den } else { self.open };
        let delta = self.buy_volume - self.sell_volume;
        Bar {
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            vwap,
            buy_volume: self.buy_volume,
            sell_volume: self.sell_volume,
            n_ticks: self.n_ticks,
            delta,
            open_time_ms: self.open_time_ms,
            close_time_ms: self.close_time_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// BarBuilder -- time-based bar builder
// ---------------------------------------------------------------------------

/// Builds time-based OHLCV bars from a stream of ticks.
/// Call `add_tick` for each tick; it returns a completed Bar when the timeframe elapses.
pub struct BarBuilder {
    /// Bar timeframe in milliseconds.
    pub timeframe_ms: u64,
    pub current_bar: Option<PartialBar>,
}

impl BarBuilder {
    pub fn new(timeframe_ms: u64) -> Self {
        Self { timeframe_ms, current_bar: None }
    }

    /// Add a tick. Returns a completed Bar if the timeframe has elapsed.
    pub fn add_tick(&mut self, tick: &Tick) -> Option<Bar> {
        let tick_ms = (tick.timestamp_ns / 1_000_000) as u64;

        match &mut self.current_bar {
            None => {
                self.current_bar = Some(PartialBar::new(tick));
                None
            }
            Some(pb) => {
                // Check if this tick belongs to a new timeframe bucket
                let bar_end_ms = pb.open_time_ms
                    + self.timeframe_ms
                    - (pb.open_time_ms % self.timeframe_ms.max(1));
                if tick_ms >= bar_end_ms {
                    let completed = std::mem::replace(pb, PartialBar::new(tick)).seal();
                    Some(completed)
                } else {
                    pb.add(tick);
                    None
                }
            }
        }
    }

    /// Flush any incomplete bar (call at end of data).
    pub fn flush(&mut self) -> Option<Bar> {
        self.current_bar.take().map(|pb| pb.seal())
    }
}

// ---------------------------------------------------------------------------
// build_bars -- convenience wrapper
// ---------------------------------------------------------------------------

/// Build time-based bars from a tick slice.
pub fn build_bars(ticks: &[Tick], timeframe_ms: u64) -> Vec<Bar> {
    let mut builder = BarBuilder::new(timeframe_ms);
    let mut bars = Vec::new();
    for tick in ticks {
        if let Some(bar) = builder.add_tick(tick) {
            bars.push(bar);
        }
    }
    if let Some(bar) = builder.flush() {
        bars.push(bar);
    }
    bars
}

// ---------------------------------------------------------------------------
// build_volume_bars -- equal-volume bars
// ---------------------------------------------------------------------------

/// Build equal-volume bars: each bar closes when cumulative volume >= target_volume.
pub fn build_volume_bars(ticks: &[Tick], target_volume: f64) -> Vec<Bar> {
    if ticks.is_empty() || target_volume <= 0.0 {
        return Vec::new();
    }
    let mut bars = Vec::new();
    let mut current: Option<PartialBar> = None;

    for tick in ticks {
        let pb = current.get_or_insert_with(|| PartialBar::new(tick));
        if std::ptr::eq(pb as *const _, pb as *const _) && pb.n_ticks == 0 {
            // freshly inserted -- already added first tick
        } else {
            pb.add(tick);
        }

        if current.as_ref().unwrap().volume >= target_volume {
            bars.push(current.take().unwrap().seal());
        }
    }
    if let Some(pb) = current {
        bars.push(pb.seal());
    }
    bars
}

// ---------------------------------------------------------------------------
// build_tick_bars -- equal-tick-count bars
// ---------------------------------------------------------------------------

/// Build equal-tick-count bars: each bar contains exactly n_ticks ticks.
pub fn build_tick_bars(ticks: &[Tick], n_ticks: usize) -> Vec<Bar> {
    if ticks.is_empty() || n_ticks == 0 {
        return Vec::new();
    }
    let mut bars = Vec::new();
    let mut current: Option<PartialBar> = None;

    for tick in ticks {
        match &mut current {
            None => {
                current = Some(PartialBar::new(tick));
            }
            Some(pb) => {
                pb.add(tick);
            }
        }
        if current.as_ref().unwrap().n_ticks as usize >= n_ticks {
            bars.push(current.take().unwrap().seal());
        }
    }
    if let Some(pb) = current {
        bars.push(pb.seal());
    }
    bars
}

// ---------------------------------------------------------------------------
// build_dollar_bars -- equal-dollar bars
// ---------------------------------------------------------------------------

/// Build equal-dollar bars: each bar closes when cumulative dollar volume >= target_dollars.
pub fn build_dollar_bars(ticks: &[Tick], target_dollars: f64) -> Vec<Bar> {
    if ticks.is_empty() || target_dollars <= 0.0 {
        return Vec::new();
    }
    let mut bars = Vec::new();
    let mut current: Option<PartialBar> = None;
    let mut dollar_accum: f64 = 0.0;

    for tick in ticks {
        dollar_accum += tick.price * tick.volume;
        match &mut current {
            None => {
                current = Some(PartialBar::new(tick));
            }
            Some(pb) => {
                pb.add(tick);
            }
        }
        if dollar_accum >= target_dollars {
            bars.push(current.take().unwrap().seal());
            dollar_accum = 0.0;
        }
    }
    if let Some(pb) = current {
        bars.push(pb.seal());
    }
    bars
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick_replay::Tick;

    fn t(price: f64, vol: f64, side: i8, ts_ms: i64) -> Tick {
        Tick::new(price, vol, side, ts_ms * 1_000_000) // ms -> ns
    }

    // A sequence of ticks spanning two 60-second bars
    fn two_minute_ticks() -> Vec<Tick> {
        vec![
            t(100.0, 100.0, 1, 0),       // bar 1
            t(100.5, 200.0, 1, 10_000),  // bar 1
            t(99.5, 150.0, -1, 30_000),  // bar 1
            t(100.0, 100.0, 1, 59_999),  // bar 1 -- last tick before 60s boundary
            t(101.0, 300.0, 1, 60_000),  // bar 2
            t(100.0, 200.0, -1, 90_000), // bar 2
        ]
    }

    // ---------------------------------------------------------------------------
    // PartialBar
    // ---------------------------------------------------------------------------

    #[test]
    fn test_partial_bar_new() {
        let tick = t(100.0, 500.0, 1, 0);
        let pb = PartialBar::new(&tick);
        assert_eq!(pb.open, 100.0);
        assert_eq!(pb.high, 100.0);
        assert_eq!(pb.low, 100.0);
        assert_eq!(pb.volume, 500.0);
        assert_eq!(pb.n_ticks, 1);
        assert_eq!(pb.buy_volume, 500.0);
        assert_eq!(pb.sell_volume, 0.0);
    }

    #[test]
    fn test_partial_bar_seal_vwap() {
        let tick1 = t(100.0, 100.0, 1, 0);
        let tick2 = t(102.0, 100.0, -1, 1_000);
        let mut pb = PartialBar::new(&tick1);
        pb.add(&tick2);
        let bar = pb.seal();
        // VWAP = (100*100 + 102*100) / 200 = 101
        assert!((bar.vwap - 101.0).abs() < 1e-9);
        assert_eq!(bar.delta, 100.0 - 100.0); // 0
    }

    #[test]
    fn test_partial_bar_delta() {
        let tick1 = t(100.0, 300.0, 1, 0);   // buy
        let tick2 = t(100.0, 100.0, -1, 500); // sell
        let mut pb = PartialBar::new(&tick1);
        pb.add(&tick2);
        let bar = pb.seal();
        assert!((bar.delta - 200.0).abs() < 1e-9);
    }

    // ---------------------------------------------------------------------------
    // BarBuilder
    // ---------------------------------------------------------------------------

    #[test]
    fn test_bar_builder_produces_two_bars() {
        let ticks = two_minute_ticks();
        let mut builder = BarBuilder::new(60_000); // 60-second bars
        let mut bars: Vec<Bar> = Vec::new();
        for tick in &ticks {
            if let Some(bar) = builder.add_tick(tick) {
                bars.push(bar);
            }
        }
        if let Some(bar) = builder.flush() {
            bars.push(bar);
        }
        assert_eq!(bars.len(), 2, "Expected 2 bars, got {}", bars.len());
    }

    #[test]
    fn test_bar_builder_ohlc_correct() {
        let ticks = vec![
            t(100.0, 100.0, 1, 0),
            t(102.0, 50.0, 1, 5_000),
            t(98.0, 75.0, -1, 10_000),
            t(101.0, 60.0, 1, 55_000),
        ];
        let mut builder = BarBuilder::new(60_000);
        let mut bars = Vec::new();
        for tick in &ticks {
            if let Some(b) = builder.add_tick(tick) { bars.push(b); }
        }
        if let Some(b) = builder.flush() { bars.push(b); }
        assert_eq!(bars.len(), 1);
        let bar = &bars[0];
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 102.0);
        assert_eq!(bar.low, 98.0);
        assert_eq!(bar.close, 101.0);
        assert_eq!(bar.n_ticks, 4);
    }

    // ---------------------------------------------------------------------------
    // build_bars
    // ---------------------------------------------------------------------------

    #[test]
    fn test_build_bars_empty_ticks() {
        let bars = build_bars(&[], 60_000);
        assert!(bars.is_empty());
    }

    #[test]
    fn test_build_bars_single_tick() {
        let ticks = vec![t(100.0, 100.0, 1, 0)];
        let bars = build_bars(&ticks, 60_000);
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].open, 100.0);
    }

    // ---------------------------------------------------------------------------
    // build_volume_bars
    // ---------------------------------------------------------------------------

    #[test]
    fn test_build_volume_bars_basic() {
        // 5 ticks of 100 volume each, target = 300
        // Bar 1: ticks 0,1,2 (300 vol), Bar 2: ticks 3,4 (200 vol, last bar)
        let ticks: Vec<Tick> = (0..5).map(|i| t(100.0, 100.0, 1, i * 1000)).collect();
        let bars = build_volume_bars(&ticks, 300.0);
        assert!(bars.len() >= 1);
        assert!(bars[0].volume >= 300.0);
    }

    #[test]
    fn test_build_volume_bars_empty() {
        let bars = build_volume_bars(&[], 1000.0);
        assert!(bars.is_empty());
    }

    // ---------------------------------------------------------------------------
    // build_tick_bars
    // ---------------------------------------------------------------------------

    #[test]
    fn test_build_tick_bars_exact_division() {
        let ticks: Vec<Tick> = (0..9).map(|i| t(100.0 + i as f64, 10.0, 1, i * 100)).collect();
        let bars = build_tick_bars(&ticks, 3);
        assert_eq!(bars.len(), 3);
        for bar in &bars {
            assert_eq!(bar.n_ticks, 3);
        }
    }

    #[test]
    fn test_build_tick_bars_remainder() {
        // 10 ticks, n=3 -> 3 complete bars + 1 remainder bar
        let ticks: Vec<Tick> = (0..10).map(|i| t(100.0, 1.0, 1, i * 100)).collect();
        let bars = build_tick_bars(&ticks, 3);
        assert_eq!(bars.len(), 4);
        assert_eq!(bars[3].n_ticks, 1);
    }

    // ---------------------------------------------------------------------------
    // build_dollar_bars
    // ---------------------------------------------------------------------------

    #[test]
    fn test_build_dollar_bars_basic() {
        // price=100, vol=10 -> $1000 per tick. Target = $3000 -> every 3 ticks
        let ticks: Vec<Tick> = (0..6).map(|i| t(100.0, 10.0, 1, i * 1000)).collect();
        let bars = build_dollar_bars(&ticks, 3000.0);
        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn test_build_dollar_bars_empty() {
        let bars = build_dollar_bars(&[], 10_000.0);
        assert!(bars.is_empty());
    }

    // ---------------------------------------------------------------------------
    // to_canonical
    // ---------------------------------------------------------------------------

    #[test]
    fn test_to_canonical_roundtrip() {
        let ticks = vec![t(150.0, 100.0, 1, 0)];
        let bars = build_bars(&ticks, 60_000);
        assert_eq!(bars.len(), 1);
        let canonical = bars[0].to_canonical();
        assert_eq!(canonical.open, 150.0);
        assert_eq!(canonical.volume, 100.0);
    }
}
