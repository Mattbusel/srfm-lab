/// Tick/volume/dollar/imbalance bar aggregators.
///
/// Bars are completed when a threshold is crossed:
///   - Tick bars:    N trades
///   - Volume bars:  total_volume ≥ threshold
///   - Dollar bars:  total_dollar_volume ≥ threshold
///   - Imbalance bars: |signed_tick_imbalance| ≥ threshold (EMA-estimated)
///
/// All bar types return an `OhlcvBar` plus optional metadata.

use serde::Serialize;

use crate::order_flow::TickRule;
use crate::streaming_stats::ExponentialMovingStats;

/// OHLCV bar with microstructure metadata.
#[derive(Debug, Clone, Serialize)]
pub struct OhlcvBar {
    pub open:         f64,
    pub high:         f64,
    pub low:          f64,
    pub close:        f64,
    pub volume:       f64,
    pub dollar_vol:   f64,
    pub vwap:         f64,
    pub n_trades:     u64,
    pub buy_volume:   f64,
    pub sell_volume:  f64,
    /// Buy/sell imbalance in [-1, +1]
    pub imbalance:    f64,
    /// Bar open timestamp (UNIX seconds, optional)
    pub ts_open:      Option<f64>,
    pub ts_close:     Option<f64>,
}

impl OhlcvBar {
    fn new(open: f64, ts: Option<f64>) -> Self {
        Self {
            open, high: open, low: open, close: open,
            volume: 0.0, dollar_vol: 0.0, vwap: 0.0,
            n_trades: 0, buy_volume: 0.0, sell_volume: 0.0, imbalance: 0.0,
            ts_open: ts, ts_close: None,
        }
    }
}

/// Which event triggers bar completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarType {
    Tick,
    Volume,
    Dollar,
    TickImbalance,
    VolumeImbalance,
}

/// Unified bar aggregator supporting all bar types.
#[derive(Debug, Clone)]
pub struct TickAggregator {
    bar_type:   BarType,
    threshold:  f64,

    // Current bar state
    current:    Option<OhlcvBar>,
    tick_count: u64,
    total_vol:  f64,
    total_dv:   f64,
    sum_pv:     f64,    // Σ price × volume for VWAP
    buy_vol:    f64,
    sell_vol:   f64,
    signed_ticks: f64,  // for imbalance bars

    // Tick classifier
    tick_rule:  TickRule,

    // EMA for adaptive imbalance threshold
    imb_ema:    ExponentialMovingStats,

    // Completed bars
    completed:  Vec<OhlcvBar>,
}

impl TickAggregator {
    /// Create a new aggregator.
    /// `threshold` meaning depends on `bar_type`:
    ///   - Tick:  number of trades per bar
    ///   - Volume: volume units per bar
    ///   - Dollar: dollar value per bar
    ///   - *Imbalance: imbalance magnitude threshold
    pub fn new(bar_type: BarType, threshold: f64) -> Self {
        assert!(threshold > 0.0, "threshold must be positive");
        Self {
            bar_type,
            threshold,
            current:      None,
            tick_count:   0,
            total_vol:    0.0,
            total_dv:     0.0,
            sum_pv:       0.0,
            buy_vol:      0.0,
            sell_vol:     0.0,
            signed_ticks: 0.0,
            tick_rule:    TickRule::new(),
            imb_ema:      ExponentialMovingStats::new(0.1),
            completed:    Vec::new(),
        }
    }

    /// Process a single trade (price, volume, optional timestamp).
    /// Returns `true` if a new bar was completed.
    pub fn update(&mut self, price: f64, volume: f64, ts: Option<f64>) -> bool {
        // Initialise current bar if needed
        if self.current.is_none() {
            self.current = Some(OhlcvBar::new(price, ts));
        }

        let sign = self.tick_rule.classify(price);

        // Update running accumulators
        self.tick_count   += 1;
        self.total_vol    += volume;
        self.total_dv     += price * volume;
        self.sum_pv       += price * volume;
        if sign > 0 { self.buy_vol  += volume; }
        if sign < 0 { self.sell_vol += volume; }
        self.signed_ticks += sign as f64;

        // Update current bar OHLC
        let bar = self.current.as_mut().unwrap();
        if price > bar.high { bar.high = price; }
        if price < bar.low  { bar.low  = price; }
        bar.close = price;

        // Check completion condition
        let done = match self.bar_type {
            BarType::Tick            => self.tick_count    >= self.threshold as u64,
            BarType::Volume          => self.total_vol     >= self.threshold,
            BarType::Dollar          => self.total_dv      >= self.threshold,
            BarType::TickImbalance   => self.signed_ticks.abs() >= self.threshold,
            BarType::VolumeImbalance => (self.buy_vol - self.sell_vol).abs() >= self.threshold,
        };

        if done {
            self.close_bar(ts);
            return true;
        }
        false
    }

    fn close_bar(&mut self, ts: Option<f64>) {
        if let Some(mut bar) = self.current.take() {
            bar.volume      = self.total_vol;
            bar.dollar_vol  = self.total_dv;
            bar.vwap        = if self.total_vol > 0.0 { self.sum_pv / self.total_vol } else { bar.close };
            bar.n_trades    = self.tick_count;
            bar.buy_volume  = self.buy_vol;
            bar.sell_volume = self.sell_vol;
            let tot = self.buy_vol + self.sell_vol;
            bar.imbalance   = if tot > 0.0 { (self.buy_vol - self.sell_vol) / tot } else { 0.0 };
            bar.ts_close    = ts;

            // Feed imbalance into EMA for adaptive threshold (informational)
            self.imb_ema.update(bar.imbalance.abs());

            self.completed.push(bar);
        }
        // Reset accumulators
        self.tick_count   = 0;
        self.total_vol    = 0.0;
        self.total_dv     = 0.0;
        self.sum_pv       = 0.0;
        self.buy_vol      = 0.0;
        self.sell_vol     = 0.0;
        self.signed_ticks = 0.0;
    }

    /// Drain and return all completed bars since last call.
    pub fn drain_bars(&mut self) -> Vec<OhlcvBar> {
        std::mem::take(&mut self.completed)
    }

    /// Peek at completed bars without consuming them.
    pub fn bars(&self) -> &[OhlcvBar] { &self.completed }

    /// Number of completed bars.
    pub fn n_bars(&self) -> usize { self.completed.len() }

    /// Current partial bar state.
    pub fn current_bar(&self) -> Option<&OhlcvBar> { self.current.as_ref() }

    /// EMA of recent imbalance magnitudes (useful for adaptive thresholds).
    pub fn imbalance_ema(&self) -> f64 { self.imb_ema.mean() }
}

/// Batch-process a slice of (price, volume) ticks and return completed bars.
pub fn batch_bars(
    bar_type:  BarType,
    threshold: f64,
    ticks:     &[(f64, f64)],
) -> Vec<OhlcvBar> {
    let mut agg = TickAggregator::new(bar_type, threshold);
    for &(price, vol) in ticks {
        agg.update(price, vol, None);
    }
    agg.drain_bars()
}

/// Compute time-series of VWAP from completed bars.
pub fn vwap_series(bars: &[OhlcvBar]) -> Vec<f64> {
    let mut cum_pv  = 0.0_f64;
    let mut cum_vol = 0.0_f64;
    bars.iter().map(|b| {
        cum_pv  += b.vwap * b.volume;
        cum_vol += b.volume;
        if cum_vol > 0.0 { cum_pv / cum_vol } else { b.close }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ticks(n: usize, base_price: f64) -> Vec<(f64, f64)> {
        (0..n).map(|i| {
            let p = base_price + (i as f64 * 0.01).sin() * 0.5;
            (p, 100.0)
        }).collect()
    }

    #[test]
    fn tick_bars_correct_count() {
        let ticks = make_ticks(1000, 100.0);
        let bars  = batch_bars(BarType::Tick, 50.0, &ticks);
        assert_eq!(bars.len(), 20, "expected 1000/50=20 bars");
        for b in &bars { assert_eq!(b.n_trades, 50); }
    }

    #[test]
    fn volume_bars() {
        let ticks = make_ticks(500, 100.0);
        let bars  = batch_bars(BarType::Volume, 1000.0, &ticks);
        // Each tick = 100 vol, so 10 ticks per bar
        assert_eq!(bars.len(), 50);
        for b in &bars { assert!((b.volume - 1000.0).abs() < 1e-6); }
    }

    #[test]
    fn dollar_bars() {
        let ticks: Vec<(f64, f64)> = (0..200).map(|_| (100.0, 50.0)).collect();
        let bars  = batch_bars(BarType::Dollar, 100_000.0, &ticks);
        // Each tick = 100*50 = $5000; 20 ticks = $100K per bar
        assert_eq!(bars.len(), 10);
    }

    #[test]
    fn ohlcv_high_low_correct() {
        let ticks = vec![(100.0, 1.0), (101.0, 1.0), (99.5, 1.0), (100.5, 1.0), (100.2, 1.0)];
        let bars  = batch_bars(BarType::Tick, 5.0, &ticks);
        assert_eq!(bars.len(), 1);
        let b = &bars[0];
        assert_eq!(b.open,  100.0);
        assert_eq!(b.high,  101.0);
        assert_eq!(b.low,    99.5);
        assert_eq!(b.close, 100.2);
    }

    #[test]
    fn vwap_series_monotone_vol() {
        let bars: Vec<OhlcvBar> = (0..5).map(|i| {
            let mut b = OhlcvBar::new(100.0 + i as f64, None);
            b.volume = 100.0; b.vwap = 100.0 + i as f64;
            b
        }).collect();
        let vwap = vwap_series(&bars);
        assert_eq!(vwap.len(), 5);
        // Cumulative VWAP should be between 100 and 104
        assert!(vwap[4] > 100.0 && vwap[4] < 105.0);
    }
}
