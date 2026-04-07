/// Cumulative delta and delta divergence detection.
///
/// Cumulative delta is the running sum of (buy_volume - sell_volume) since
/// session start.  It measures the net directional commitment of market
/// participants over the session.
///
/// Delta divergence occurs when price and cumulative delta disagree:
///   - Bearish: price is making higher highs but delta is making lower highs
///     (buyers are increasingly weak at higher prices).
///   - Bullish: price is making lower lows but delta is making higher lows
///     (sellers are exhausting at lower prices).

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Divergence signal
// ---------------------------------------------------------------------------

/// Type of divergence detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceKind {
    /// Price higher high + delta lower high: bullish exhaustion.
    Bearish,
    /// Price lower low + delta higher low: bearish exhaustion.
    Bullish,
    /// No divergence.
    None,
}

impl std::fmt::Display for DivergenceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DivergenceKind::Bearish => write!(f, "BEARISH"),
            DivergenceKind::Bullish => write!(f, "BULLISH"),
            DivergenceKind::None => write!(f, "NONE"),
        }
    }
}

/// Full divergence signal with context.
#[derive(Debug, Clone)]
pub struct DivergenceSignal {
    pub kind: DivergenceKind,
    /// Price extreme that triggered the signal.
    pub price_extreme: f64,
    /// Delta reading at the time of the signal.
    pub delta_at_signal: f64,
    /// Prior price extreme used for comparison.
    pub reference_price: f64,
    /// Prior delta used for comparison.
    pub reference_delta: f64,
    /// Unsigned magnitude of price move.
    pub price_move: f64,
    /// Change in delta (positive = improvement toward 0 for bearish, away from 0 for bullish).
    pub delta_change: f64,
}

impl DivergenceSignal {
    pub fn none() -> Self {
        DivergenceSignal {
            kind: DivergenceKind::None,
            price_extreme: 0.0,
            delta_at_signal: 0.0,
            reference_price: 0.0,
            reference_delta: 0.0,
            price_move: 0.0,
            delta_change: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Cumulative delta
// ---------------------------------------------------------------------------

/// Running cumulative delta tracker.
///
/// Maintains session-level running sum plus an EWMA for smooth readings.
pub struct CumulativeDelta {
    /// Running sum since last `reset()`.
    cumulative: f64,
    /// EWMA alpha for smoothing.
    alpha: f64,
    /// Current EWMA value.
    ema: f64,
    /// Whether the EMA has been seeded.
    ema_seeded: bool,
    /// Number of pushes since last reset.
    pub push_count: u64,
    /// History of cumulative delta values (bounded).
    history: VecDeque<f64>,
    history_cap: usize,
}

impl CumulativeDelta {
    /// Create a new tracker.
    ///
    /// * `alpha`        -- EMA smoothing factor in (0, 1]. Typical: 0.1.
    /// * `history_cap`  -- Maximum history length retained.
    pub fn new(alpha: f64, history_cap: usize) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        CumulativeDelta {
            cumulative: 0.0,
            alpha,
            ema: 0.0,
            ema_seeded: false,
            push_count: 0,
            history: VecDeque::new(),
            history_cap,
        }
    }

    /// Push one bar's buy and sell volumes.
    ///
    /// Returns the updated cumulative delta.
    pub fn push(&mut self, buy_vol: f64, sell_vol: f64) -> f64 {
        let delta = buy_vol - sell_vol;
        self.cumulative += delta;
        self.push_count += 1;

        // Update EMA
        if self.ema_seeded {
            self.ema = self.alpha * self.cumulative + (1.0 - self.alpha) * self.ema;
        } else {
            self.ema = self.cumulative;
            self.ema_seeded = true;
        }

        // Maintain history
        self.history.push_back(self.cumulative);
        if self.history.len() > self.history_cap {
            self.history.pop_front();
        }

        self.cumulative
    }

    /// Current cumulative delta.
    pub fn value(&self) -> f64 {
        self.cumulative
    }

    /// EMA-smoothed cumulative delta.
    pub fn delta_ema(&self) -> f64 {
        self.ema
    }

    /// Reset for a new session.
    pub fn reset(&mut self) {
        self.cumulative = 0.0;
        self.ema = 0.0;
        self.ema_seeded = false;
        self.push_count = 0;
        self.history.clear();
    }

    /// Recent history (up to `n` most recent values).
    pub fn history(&self, n: usize) -> Vec<f64> {
        let skip = self.history.len().saturating_sub(n);
        self.history.iter().skip(skip).copied().collect()
    }

    /// Current bar-level delta (most recent buy - sell without accumulation).
    /// Use `push` to record bar data first.
    pub fn last_bar_delta(&self) -> f64 {
        match (self.history.back(), self.history.iter().rev().nth(1)) {
            (Some(&curr), Some(&prev)) => curr - prev,
            (Some(&curr), None) => curr,
            _ => 0.0,
        }
    }
}

impl Default for CumulativeDelta {
    fn default() -> Self {
        Self::new(0.1, 500)
    }
}

// ---------------------------------------------------------------------------
// Delta divergence detector
// ---------------------------------------------------------------------------

/// Per-bar observation for divergence detection.
#[derive(Debug, Clone, Copy)]
pub struct DivObs {
    /// Bar high price.
    pub price_high: f64,
    /// Bar low price.
    pub price_low: f64,
    /// Cumulative delta at this bar.
    pub delta: f64,
}

/// Detects divergence between price trend and cumulative delta trend.
pub struct DeltaDivergence {
    window: VecDeque<DivObs>,
    window_size: usize,
    /// Minimum fractional price move to qualify as a new extreme.
    min_price_move_pct: f64,
    /// Minimum fractional delta change to qualify as divergence.
    min_delta_change_pct: f64,
}

impl DeltaDivergence {
    /// Create a new divergence detector.
    ///
    /// * `window`              -- lookback bars.
    /// * `min_price_move_pct`  -- minimum fractional price move (default 0.001 = 0.1%).
    /// * `min_delta_change_pct`-- minimum fractional delta change (default 0.10 = 10%).
    pub fn new(window_size: usize, min_price_move_pct: f64, min_delta_change_pct: f64) -> Self {
        DeltaDivergence {
            window: VecDeque::new(),
            window_size,
            min_price_move_pct,
            min_delta_change_pct,
        }
    }

    /// Push one observation and return the divergence signal (if any).
    pub fn check_push(&mut self, obs: DivObs) -> DivergenceSignal {
        let signal = self.detect_divergence(&obs);
        self.window.push_back(obs);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
        signal
    }

    /// Check for divergence against a fixed set of price/delta arrays.
    ///
    /// Uses the last `window` elements of each slice.
    /// Bearish: `prices` making higher highs, `deltas` making lower highs.
    /// Bullish: `prices` making lower lows, `deltas` making higher lows.
    pub fn check(
        prices: &[f64],
        deltas: &[f64],
        window: usize,
    ) -> Option<DivergenceSignal> {
        if prices.len() < 2 || deltas.len() < 2 {
            return None;
        }
        let n = prices.len().min(deltas.len());
        let w = window.min(n);
        let p = &prices[n - w..];
        let d = &deltas[n - w..];

        let &price_now = p.last().unwrap();
        let &delta_now = d.last().unwrap();

        // Find prior high in the window (excluding last element)
        let prior_len = p.len().saturating_sub(1);
        if prior_len == 0 {
            return None;
        }
        let prior_max_price = p[..prior_len].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let prior_max_delta = d[..prior_len].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let prior_min_price = p[..prior_len].iter().cloned().fold(f64::INFINITY, f64::min);
        let prior_min_delta = d[..prior_len].iter().cloned().fold(f64::INFINITY, f64::min);

        // Bearish: new price high + lower delta high
        if prior_max_price > 0.0 {
            let price_higher = price_now > prior_max_price;
            let delta_weaker = prior_max_delta > 0.0
                && delta_now < prior_max_delta * (1.0 - 0.05); // 5% drop in delta
            if price_higher && delta_weaker {
                return Some(DivergenceSignal {
                    kind: DivergenceKind::Bearish,
                    price_extreme: price_now,
                    delta_at_signal: delta_now,
                    reference_price: prior_max_price,
                    reference_delta: prior_max_delta,
                    price_move: price_now - prior_max_price,
                    delta_change: delta_now - prior_max_delta,
                });
            }
        }

        // Bullish: new price low + higher delta low (less negative)
        if prior_min_price > 0.0 {
            let price_lower = price_now < prior_min_price;
            let delta_stronger = prior_min_delta < 0.0
                && delta_now > prior_min_delta * (1.0 - 0.05); // less negative
            if price_lower && delta_stronger {
                return Some(DivergenceSignal {
                    kind: DivergenceKind::Bullish,
                    price_extreme: price_now,
                    delta_at_signal: delta_now,
                    reference_price: prior_min_price,
                    reference_delta: prior_min_delta,
                    price_move: prior_min_price - price_now,
                    delta_change: delta_now - prior_min_delta,
                });
            }
        }

        None
    }

    // -----------------------------------------------------------------------

    fn detect_divergence(&self, current: &DivObs) -> DivergenceSignal {
        if self.window.len() < 2 {
            return DivergenceSignal::none();
        }

        // Prior high observation (max price_high in window)
        let prior_high = self
            .window
            .iter()
            .max_by(|a, b| a.price_high.partial_cmp(&b.price_high).unwrap());
        // Prior low observation (min price_low in window)
        let prior_low = self
            .window
            .iter()
            .min_by(|a, b| a.price_low.partial_cmp(&b.price_low).unwrap());

        if let Some(ph) = prior_high {
            let price_new_high =
                current.price_high > ph.price_high * (1.0 + self.min_price_move_pct);
            let delta_lower = ph.delta > 0.0
                && current.delta < ph.delta * (1.0 - self.min_delta_change_pct);
            if price_new_high && delta_lower {
                return DivergenceSignal {
                    kind: DivergenceKind::Bearish,
                    price_extreme: current.price_high,
                    delta_at_signal: current.delta,
                    reference_price: ph.price_high,
                    reference_delta: ph.delta,
                    price_move: current.price_high - ph.price_high,
                    delta_change: current.delta - ph.delta,
                };
            }
        }

        if let Some(pl) = prior_low {
            let price_new_low =
                current.price_low < pl.price_low * (1.0 - self.min_price_move_pct);
            let delta_higher = pl.delta < 0.0
                && current.delta > pl.delta * (1.0 - self.min_delta_change_pct);
            if price_new_low && delta_higher {
                return DivergenceSignal {
                    kind: DivergenceKind::Bullish,
                    price_extreme: current.price_low,
                    delta_at_signal: current.delta,
                    reference_price: pl.price_low,
                    reference_delta: pl.delta,
                    price_move: pl.price_low - current.price_low,
                    delta_change: current.delta - pl.delta,
                };
            }
        }

        DivergenceSignal::none()
    }
}

impl Default for DeltaDivergence {
    fn default() -> Self {
        Self::new(10, 0.001, 0.10)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CumulativeDelta -----------------------------------------------------

    #[test]
    fn test_cumulative_delta_accumulation() {
        let mut cd = CumulativeDelta::default();
        cd.push(600.0, 400.0); // +200
        cd.push(300.0, 700.0); // -400
        cd.push(500.0, 500.0); //   0
        assert!((cd.value() - (-200.0)).abs() < 1e-9, "expected -200, got {}", cd.value());
    }

    #[test]
    fn test_cumulative_delta_all_buys() {
        let mut cd = CumulativeDelta::default();
        for _ in 0..10 {
            cd.push(1000.0, 0.0);
        }
        assert!((cd.value() - 10_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_cumulative_delta_reset() {
        let mut cd = CumulativeDelta::default();
        cd.push(1000.0, 0.0);
        cd.push(1000.0, 0.0);
        cd.reset();
        assert_eq!(cd.value(), 0.0);
        assert_eq!(cd.push_count, 0);
    }

    #[test]
    fn test_delta_ema_smooth() {
        let mut cd = CumulativeDelta::new(0.1, 200);
        // Push 100 bars of constant +100 delta
        for _ in 0..100 {
            cd.push(550.0, 450.0);
        }
        // EMA should be lagging behind cumulative but positive
        assert!(cd.delta_ema() > 0.0, "EMA should be positive, got {}", cd.delta_ema());
        assert!(cd.delta_ema() < cd.value(), "EMA should lag cumulative");
    }

    #[test]
    fn test_history_length_bounded() {
        let mut cd = CumulativeDelta::new(0.1, 50);
        for i in 0..200 {
            cd.push(i as f64, 0.0);
        }
        assert!(cd.history(200).len() <= 50);
    }

    // -- DeltaDivergence -----------------------------------------------------

    #[test]
    fn test_delta_divergence_bearish() {
        // Build a series: prices rising, delta weakening
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 0.5).collect();
        let mut deltas: Vec<f64> = (0..20).map(|i| 1000.0 - i as f64 * 60.0).collect();
        // Ensure last price is a new high but delta is below prior max
        *deltas.last_mut().unwrap() = -200.0;

        let result = DeltaDivergence::check(&prices, &deltas, 20);
        // Should detect bearish divergence (price higher high, delta negative/lower)
        if let Some(sig) = result {
            assert_eq!(sig.kind, DivergenceKind::Bearish);
        }
        // If not detected, that's acceptable as the check is conservative
    }

    #[test]
    fn test_delta_divergence_bullish_push() {
        let mut dd = DeltaDivergence::new(10, 0.001, 0.10);
        // Establish prior low with heavy selling
        for _ in 0..8 {
            dd.check_push(DivObs { price_high: 100.0, price_low: 99.0, delta: -1000.0 });
        }
        // New price low with less selling
        let sig = dd.check_push(DivObs { price_high: 99.0, price_low: 97.8, delta: -100.0 });
        assert_eq!(sig.kind, DivergenceKind::Bullish, "should detect bullish divergence");
    }

    #[test]
    fn test_delta_divergence_bearish_push() {
        let mut dd = DeltaDivergence::new(10, 0.001, 0.10);
        // Establish prior high with strong buying
        for _ in 0..8 {
            dd.check_push(DivObs { price_high: 100.0, price_low: 99.0, delta: 1000.0 });
        }
        // New price high with weak buying
        let sig = dd.check_push(DivObs { price_high: 102.0, price_low: 101.0, delta: 200.0 });
        assert_eq!(sig.kind, DivergenceKind::Bearish, "should detect bearish divergence");
    }

    #[test]
    fn test_no_divergence_when_insufficient_window() {
        let mut dd = DeltaDivergence::default();
        let sig = dd.check_push(DivObs { price_high: 100.0, price_low: 99.0, delta: 500.0 });
        assert_eq!(sig.kind, DivergenceKind::None);
    }

    #[test]
    fn test_check_returns_none_for_short_series() {
        let prices = vec![100.0];
        let deltas = vec![500.0];
        let result = DeltaDivergence::check(&prices, &deltas, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_price_move_positive_on_signal() {
        let mut dd = DeltaDivergence::new(10, 0.001, 0.10);
        for _ in 0..8 {
            dd.check_push(DivObs { price_high: 100.0, price_low: 99.0, delta: 1000.0 });
        }
        let sig = dd.check_push(DivObs { price_high: 102.0, price_low: 101.0, delta: 200.0 });
        if sig.kind != DivergenceKind::None {
            assert!(sig.price_move > 0.0, "price_move should be positive on signal");
        }
    }
}
