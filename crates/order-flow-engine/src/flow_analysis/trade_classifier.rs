/// Trade direction classification.
///
/// Implements four classification algorithms:
///
/// 1. Tick Rule             -- original tick test (uptick = buy, downtick = sell).
/// 2. Lee-Ready (1991)      -- quote rule first, tick rule for midpoint trades.
/// 3. Bulk Volume (BVC)     -- bar-level buy/sell split via price-range position.
/// 4. Ellis-Michaely-O'Hara -- open-price variant of the quote rule.
///
/// References:
///   Lee, C.M.C. & Ready, M.J. (1991). "Inferring Trade Direction from Intraday Data."
///   Journal of Finance 46(2): 733-746.
///
///   Ellis, K., Michaely, R., & O'Hara, M. (2000). "The Accuracy of Trade
///   Classification Rules: Evidence from Nasdaq." Journal of Financial and
///   Quantitative Analysis 35(4): 529-551.
///
///   Easley, D., Lopez de Prado, M., & O'Hara, M. (2012). "Bulk Classification
///   of Trading Activity." Working paper.

// ---------------------------------------------------------------------------
// Side
// ---------------------------------------------------------------------------

/// Classified direction of a single trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Side {
    Buy,
    Sell,
    /// Cannot be determined (insufficient context).
    Unknown,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
            Side::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

// ---------------------------------------------------------------------------
// Classification method
// ---------------------------------------------------------------------------

/// Available classification algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationMethod {
    TickRule,
    LeeReady,
    BulkVolume,
    /// Ellis-Michaely-O'Hara rule.
    EMO,
}

// ---------------------------------------------------------------------------
// OHLCV bar (used by BVC and EMO)
// ---------------------------------------------------------------------------

/// A single price bar with OHLCV fields.
#[derive(Debug, Clone, Copy)]
pub struct OHLCVBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OHLCVBar {
    pub fn new(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        OHLCVBar { open, high, low, close, volume }
    }

    /// Range of the bar (high - low).
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Midpoint of bid/ask approximated as (high + low) / 2.
    pub fn midpoint(&self) -> f64 {
        (self.high + self.low) / 2.0
    }
}

// ---------------------------------------------------------------------------
// Main classifier struct
// ---------------------------------------------------------------------------

/// Multi-method trade classifier.
///
/// Stateful: maintains `prev_price` for tick rule and `last_side` for zero-tick
/// resolution.  Reset between sessions with `reset()`.
#[derive(Debug)]
pub struct TradeClassifier {
    pub method: ClassificationMethod,
    prev_price: Option<f64>,
    last_side: Side,
}

impl TradeClassifier {
    /// Create a classifier using the specified method.
    pub fn new(method: ClassificationMethod) -> Self {
        TradeClassifier { method, prev_price: None, last_side: Side::Unknown }
    }

    // -----------------------------------------------------------------------
    // Stateless methods (no &mut self)
    // -----------------------------------------------------------------------

    /// Tick rule: buy on up-tick, sell on down-tick, forward-fill on zero-tick.
    ///
    /// `prev_price` must be provided explicitly when calling the stateless form.
    pub fn tick_rule(price: f64, prev_price: f64) -> Side {
        if price > prev_price {
            Side::Buy
        } else if price < prev_price {
            Side::Sell
        } else {
            Side::Unknown // caller must handle zero-tick
        }
    }

    /// Lee-Ready rule.
    ///
    /// Priority:
    ///   1. Quote rule: trade above midpoint -> Buy, below -> Sell.
    ///   2. Tick rule for trades at midpoint.
    ///
    /// `bid` and `ask` are the prevailing best bid/ask at trade time.
    /// `prev_price` is used if quote rule is inconclusive.
    pub fn lee_ready(trade_price: f64, bid: f64, ask: f64, prev_price: Option<f64>) -> Side {
        if bid >= ask {
            // Degenerate quote -- fall through to tick rule
        } else {
            let mid = (bid + ask) / 2.0;
            if trade_price > mid {
                return Side::Buy;
            } else if trade_price < mid {
                return Side::Sell;
            }
            // At midpoint -- use tick rule below
        }
        match prev_price {
            None => Side::Unknown,
            Some(prev) => {
                if trade_price > prev {
                    Side::Buy
                } else if trade_price < prev {
                    Side::Sell
                } else {
                    Side::Unknown
                }
            }
        }
    }

    /// Bulk Volume Classification for a single OHLCV bar.
    ///
    /// Approximates the buy/sell volume split without tick data.
    ///
    ///   z = (close - open) / (high - low + epsilon)   clamped to [-1, 1]
    ///   buy_vol  = volume * (0.5 + 0.5 * z)
    ///   sell_vol = volume - buy_vol
    ///
    /// Returns `(buy_vol, sell_vol)`.
    pub fn bulk_volume_classify(bar: OHLCVBar) -> (f64, f64) {
        let range = bar.range() + 1e-8; // avoid division by zero
        let z = ((bar.close - bar.open) / range).clamp(-1.0, 1.0);
        let buy_vol = bar.volume * (0.5 + 0.5 * z);
        let sell_vol = bar.volume - buy_vol;
        (buy_vol.max(0.0), sell_vol.max(0.0))
    }

    /// Ellis-Michaely-O'Hara rule.
    ///
    /// Variant of the quote rule that also compares to the bar's opening price.
    ///
    ///   - trade >= ask          -> Buy  (hit the ask)
    ///   - trade <= bid          -> Sell (hit the bid)
    ///   - trade > open          -> Buy  (EMO extension)
    ///   - trade < open          -> Sell (EMO extension)
    ///   - otherwise             -> Unknown
    pub fn emo_classify(trade_price: f64, open: f64, bid: f64, ask: f64) -> Side {
        if trade_price >= ask {
            return Side::Buy;
        }
        if trade_price <= bid {
            return Side::Sell;
        }
        // At midpoint or between bid/ask -- use open comparison
        if trade_price > open {
            Side::Buy
        } else if trade_price < open {
            Side::Sell
        } else {
            Side::Unknown
        }
    }

    // -----------------------------------------------------------------------
    // Stateful classify (updates prev_price and last_side)
    // -----------------------------------------------------------------------

    /// Classify a trade tick using the configured method.
    ///
    /// For `LeeReady` and `EMO` the `bid`, `ask`, and (for EMO) `open` must be
    /// passed as `context`.  For `TickRule` only `price` is required.
    pub fn classify(
        &mut self,
        price: f64,
        bid: Option<f64>,
        ask: Option<f64>,
        open: Option<f64>,
    ) -> Side {
        let side = match self.method {
            ClassificationMethod::TickRule => {
                let s = match self.prev_price {
                    None => Side::Unknown,
                    Some(prev) => {
                        let raw = Self::tick_rule(price, prev);
                        if raw == Side::Unknown {
                            self.last_side // carry forward
                        } else {
                            raw
                        }
                    }
                };
                s
            }
            ClassificationMethod::LeeReady => {
                let (b, a) = match (bid, ask) {
                    (Some(b), Some(a)) => (b, a),
                    _ => (0.0, 0.0), // will trigger tick-rule fallback in lee_ready
                };
                let raw = Self::lee_ready(price, b, a, self.prev_price);
                if raw == Side::Unknown { self.last_side } else { raw }
            }
            ClassificationMethod::BulkVolume => {
                // For tick-level calls, BVC cannot be applied meaningfully.
                // Fall back to tick rule.
                match self.prev_price {
                    None => Side::Unknown,
                    Some(prev) => {
                        if price > prev { Side::Buy } else if price < prev { Side::Sell } else { self.last_side }
                    }
                }
            }
            ClassificationMethod::EMO => {
                let (b, a) = match (bid, ask) {
                    (Some(b), Some(a)) => (b, a),
                    _ => (price * 0.9999, price * 1.0001),
                };
                let o = open.unwrap_or(price);
                let raw = Self::emo_classify(price, o, b, a);
                if raw == Side::Unknown { self.last_side } else { raw }
            }
        };

        self.prev_price = Some(price);
        if side != Side::Unknown {
            self.last_side = side;
        }
        side
    }

    /// Reset state (call at session open).
    pub fn reset(&mut self) {
        self.prev_price = None;
        self.last_side = Side::Unknown;
    }
}

impl Default for TradeClassifier {
    fn default() -> Self {
        Self::new(ClassificationMethod::LeeReady)
    }
}

// ---------------------------------------------------------------------------
// Batch helpers
// ---------------------------------------------------------------------------

/// Classify a slice of `(price, volume)` ticks using the tick rule.
/// Returns `(total_buy_volume, total_sell_volume)`.
pub fn classify_tick_series(ticks: &[(f64, f64)]) -> (f64, f64) {
    let mut clf = TradeClassifier::new(ClassificationMethod::TickRule);
    let mut buy = 0.0f64;
    let mut sell = 0.0f64;
    for &(price, vol) in ticks {
        match clf.classify(price, None, None, None) {
            Side::Buy => buy += vol,
            Side::Sell => sell += vol,
            Side::Unknown => {
                // Split 50/50 on Unknown
                buy += vol * 0.5;
                sell += vol * 0.5;
            }
        }
    }
    (buy, sell)
}

/// Apply Bulk Volume Classification to a slice of OHLCV bars.
/// Returns a Vec of `(buy_vol, sell_vol)` pairs.
pub fn classify_bars_bulk(bars: &[OHLCVBar]) -> Vec<(f64, f64)> {
    bars.iter().map(|&b| TradeClassifier::bulk_volume_classify(b)).collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Tick rule -----------------------------------------------------------

    #[test]
    fn test_trade_classifier_tick_rule_uptick() {
        assert_eq!(TradeClassifier::tick_rule(100.1, 100.0), Side::Buy);
    }

    #[test]
    fn test_trade_classifier_tick_rule_downtick() {
        assert_eq!(TradeClassifier::tick_rule(99.9, 100.0), Side::Sell);
    }

    #[test]
    fn test_trade_classifier_tick_rule_zero() {
        assert_eq!(TradeClassifier::tick_rule(100.0, 100.0), Side::Unknown);
    }

    #[test]
    fn test_stateful_tick_carry_forward() {
        let mut clf = TradeClassifier::new(ClassificationMethod::TickRule);
        clf.classify(100.0, None, None, None); // first: Unknown
        clf.classify(100.5, None, None, None); // uptick -> Buy, sets last_side = Buy
        let side = clf.classify(100.5, None, None, None); // zero-tick -> carry Buy
        assert_eq!(side, Side::Buy);
    }

    // -- Lee-Ready -----------------------------------------------------------

    #[test]
    fn test_lee_ready_above_midpoint() {
        // mid = (99.8 + 100.2) / 2 = 100.0; trade at 100.15 > mid -> Buy
        let side = TradeClassifier::lee_ready(100.15, 99.8, 100.2, None);
        assert_eq!(side, Side::Buy);
    }

    #[test]
    fn test_lee_ready_below_midpoint() {
        let side = TradeClassifier::lee_ready(99.85, 99.8, 100.2, None);
        assert_eq!(side, Side::Sell);
    }

    #[test]
    fn test_lee_ready_at_midpoint_uses_tick_rule() {
        // mid = 100.0; trade at 100.0; prev = 99.8 (uptick) -> Buy
        let side = TradeClassifier::lee_ready(100.0, 99.8, 100.2, Some(99.8));
        assert_eq!(side, Side::Buy);
    }

    #[test]
    fn test_lee_ready_at_midpoint_downtick() {
        // mid = 100.0; trade at 100.0; prev = 100.2 (downtick) -> Sell
        let side = TradeClassifier::lee_ready(100.0, 99.8, 100.2, Some(100.2));
        assert_eq!(side, Side::Sell);
    }

    // -- Bulk Volume ---------------------------------------------------------

    #[test]
    fn test_bulk_volume_at_close_equals_open() {
        // close == open -> z = 0 -> 50/50 split
        let bar = OHLCVBar::new(100.0, 102.0, 98.0, 100.0, 1000.0);
        let (buy, sell) = TradeClassifier::bulk_volume_classify(bar);
        assert!((buy - 500.0).abs() < 1.0, "expected 50/50 split, got buy={}", buy);
        assert!((sell - 500.0).abs() < 1.0, "expected 50/50 split, got sell={}", sell);
    }

    #[test]
    fn test_bulk_volume_close_at_high() {
        // close == high -> z ~= 1 -> most volume is buy
        let bar = OHLCVBar::new(98.0, 102.0, 98.0, 102.0, 1000.0);
        let (buy, sell) = TradeClassifier::bulk_volume_classify(bar);
        assert!(buy > sell, "close at high -> more buy volume, buy={} sell={}", buy, sell);
        assert!((buy + sell - 1000.0).abs() < 1e-9, "volumes must sum to total");
    }

    #[test]
    fn test_bulk_volume_close_at_low() {
        // close == low -> z ~= -1 -> most volume is sell
        let bar = OHLCVBar::new(102.0, 102.0, 98.0, 98.0, 1000.0);
        let (buy, sell) = TradeClassifier::bulk_volume_classify(bar);
        assert!(sell > buy, "close at low -> more sell volume, buy={} sell={}", buy, sell);
        assert!((buy + sell - 1000.0).abs() < 1e-9, "volumes must sum to total");
    }

    #[test]
    fn test_bulk_volume_sums_to_total() {
        let bars = vec![
            OHLCVBar::new(100.0, 105.0, 95.0, 103.0, 2000.0),
            OHLCVBar::new(103.0, 106.0, 101.0, 101.5, 1500.0),
            OHLCVBar::new(101.5, 103.0, 99.0, 100.0, 3000.0),
        ];
        for bar in &bars {
            let (b, s) = TradeClassifier::bulk_volume_classify(*bar);
            assert!((b + s - bar.volume).abs() < 1e-9, "buy+sell must equal volume");
        }
    }

    // -- EMO -----------------------------------------------------------------

    #[test]
    fn test_emo_hit_ask() {
        // trade >= ask -> Buy
        let side = TradeClassifier::emo_classify(100.2, 100.0, 99.8, 100.2);
        assert_eq!(side, Side::Buy);
    }

    #[test]
    fn test_emo_hit_bid() {
        // trade <= bid -> Sell
        let side = TradeClassifier::emo_classify(99.8, 100.0, 99.8, 100.2);
        assert_eq!(side, Side::Sell);
    }

    #[test]
    fn test_emo_above_open_inside_spread() {
        // bid=99.8, ask=100.2, open=100.0, trade=100.1 > open -> Buy
        let side = TradeClassifier::emo_classify(100.1, 100.0, 99.8, 100.2);
        assert_eq!(side, Side::Buy);
    }

    #[test]
    fn test_emo_below_open_inside_spread() {
        // bid=99.8, ask=100.2, open=100.0, trade=99.9 < open -> Sell
        let side = TradeClassifier::emo_classify(99.9, 100.0, 99.8, 100.2);
        assert_eq!(side, Side::Sell);
    }

    // -- Batch helpers -------------------------------------------------------

    #[test]
    fn test_classify_tick_series_sums_to_total() {
        let ticks: Vec<(f64, f64)> = (0..100)
            .map(|i| (100.0 + (i as f64 % 5.0) * 0.01, 100.0))
            .collect();
        let (buy, sell) = classify_tick_series(&ticks);
        let total: f64 = ticks.iter().map(|(_, v)| v).sum();
        assert!((buy + sell - total).abs() < 1e-9, "buy+sell must equal total volume");
    }

    #[test]
    fn test_classify_bars_bulk_batch() {
        let bars: Vec<OHLCVBar> = (0..10)
            .map(|i| OHLCVBar::new(100.0, 105.0, 95.0, 100.0 + i as f64 * 0.5, 1000.0))
            .collect();
        let results = classify_bars_bulk(&bars);
        assert_eq!(results.len(), 10);
        for (buy, sell) in &results {
            assert!(*buy >= 0.0 && *sell >= 0.0);
        }
    }
}
