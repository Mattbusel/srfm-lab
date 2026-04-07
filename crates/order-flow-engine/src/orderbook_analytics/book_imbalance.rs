/// Order book imbalance analytics.
///
/// Provides several complementary measures of supply/demand imbalance in the
/// limit order book:
///
/// 1. `weighted_imbalance`  -- (bid_vol - ask_vol)/(bid_vol + ask_vol) weighted
///    by price proximity to midpoint across the top N levels.
/// 2. `depth_imbalance`     -- imbalance of all resting volume within a
///    percentage band of the midpoint.
/// 3. `iceberg_detector`    -- flags price levels where large volume is stable
///    despite heavy trade activity (hidden iceberg orders).
/// 4. `sweep_detector`      -- identifies rapid consecutive sweeps through
///    multiple book levels (aggressive institutional flow).

// ---------------------------------------------------------------------------
// Book snapshot
// ---------------------------------------------------------------------------

/// A snapshot of the limit order book at a single point in time.
///
/// Levels are ordered from best (tightest) to worst:
///   bid_prices[0] is the best bid, ask_prices[0] is the best ask.
#[derive(Debug, Clone)]
pub struct BookSnapshot {
    /// Bid prices, best-to-worst (descending).
    pub bid_prices: Vec<f64>,
    /// Bid sizes at each level.
    pub bid_sizes: Vec<f64>,
    /// Ask prices, best-to-worst (ascending).
    pub ask_prices: Vec<f64>,
    /// Ask sizes at each level.
    pub ask_sizes: Vec<f64>,
    /// Optional timestamp.
    pub timestamp_ms: u64,
}

impl BookSnapshot {
    pub fn new(
        bid_prices: Vec<f64>,
        bid_sizes: Vec<f64>,
        ask_prices: Vec<f64>,
        ask_sizes: Vec<f64>,
    ) -> Self {
        BookSnapshot {
            bid_prices,
            bid_sizes,
            ask_prices,
            ask_sizes,
            timestamp_ms: 0,
        }
    }

    /// Best bid price (None if book is empty).
    pub fn best_bid(&self) -> Option<f64> {
        self.bid_prices.first().copied()
    }

    /// Best ask price (None if book is empty).
    pub fn best_ask(&self) -> Option<f64> {
        self.ask_prices.first().copied()
    }

    /// Midpoint price.
    pub fn midpoint(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }

    /// Bid-ask spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    /// Validate that the snapshot has matching price/size arrays.
    pub fn is_valid(&self) -> bool {
        self.bid_prices.len() == self.bid_sizes.len()
            && self.ask_prices.len() == self.ask_sizes.len()
            && !self.bid_prices.is_empty()
            && !self.ask_prices.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Trade record (lightweight, for sweep/iceberg detection)
// ---------------------------------------------------------------------------

/// A single trade used in book analytics.
#[derive(Debug, Clone, Copy)]
pub struct BookTrade {
    pub price: f64,
    pub volume: f64,
    pub timestamp_ms: u64,
}

impl BookTrade {
    pub fn new(price: f64, volume: f64) -> Self {
        BookTrade { price, volume, timestamp_ms: 0 }
    }
}

// ---------------------------------------------------------------------------
// Iceberg indicator
// ---------------------------------------------------------------------------

/// Signal that an iceberg order may be present at a given price level.
#[derive(Debug, Clone)]
pub struct IcebergIndicator {
    /// Price level where the iceberg was detected.
    pub price: f64,
    /// Side of the book (true = bid, false = ask).
    pub is_bid: bool,
    /// Resting visible size at this level.
    pub resting_size: f64,
    /// Estimated volume traded through this level during the observation window.
    pub traded_volume: f64,
    /// Ratio traded_volume / resting_size (higher = more likely iceberg).
    pub iceberg_ratio: f64,
}

// ---------------------------------------------------------------------------
// Sweep event
// ---------------------------------------------------------------------------

/// A detected aggressive sweep through multiple book levels.
#[derive(Debug, Clone)]
pub struct SweepEvent {
    /// True = buy-side sweep (lifting the ask), False = sell-side sweep.
    pub is_buy: bool,
    /// Starting price of the sweep.
    pub price_start: f64,
    /// Ending price of the sweep.
    pub price_end: f64,
    /// Total volume of the sweep.
    pub total_volume: f64,
    /// Number of distinct price levels consumed.
    pub levels_consumed: usize,
    /// Timestamp of the first trade in the sweep window.
    pub timestamp_ms: u64,
}

impl SweepEvent {
    /// Net price move of the sweep.
    pub fn price_move(&self) -> f64 {
        (self.price_end - self.price_start).abs()
    }
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Order book imbalance analyzer.
pub struct BookImbalanceAnalyzer {
    /// Minimum iceberg ratio to flag a level.
    iceberg_threshold: f64,
    /// Minimum number of distinct levels consumed to classify a sweep.
    sweep_min_levels: usize,
}

impl BookImbalanceAnalyzer {
    /// Create an analyzer with custom thresholds.
    pub fn new(iceberg_threshold: f64, sweep_min_levels: usize) -> Self {
        BookImbalanceAnalyzer { iceberg_threshold, sweep_min_levels }
    }

    // -----------------------------------------------------------------------
    // 1. Weighted imbalance
    // -----------------------------------------------------------------------

    /// Compute price-proximity-weighted order book imbalance over the top N levels.
    ///
    /// Weight for level i = 1 / (1 + i) (bid levels in descending price order,
    /// ask levels in ascending order).
    ///
    /// Returns a value in [-1, 1]:
    ///   +1.0 = all visible volume is on the bid
    ///   -1.0 = all visible volume is on the ask
    pub fn weighted_imbalance(&self, snapshot: &BookSnapshot, levels: usize) -> f64 {
        if !snapshot.is_valid() {
            return 0.0;
        }

        let n_bid = levels.min(snapshot.bid_sizes.len());
        let n_ask = levels.min(snapshot.ask_sizes.len());

        let mut bid_w = 0.0f64;
        let mut ask_w = 0.0f64;

        for i in 0..n_bid {
            let weight = 1.0 / (1.0 + i as f64);
            bid_w += snapshot.bid_sizes[i] * weight;
        }
        for i in 0..n_ask {
            let weight = 1.0 / (1.0 + i as f64);
            ask_w += snapshot.ask_sizes[i] * weight;
        }

        let total = bid_w + ask_w;
        if total < 1e-12 {
            return 0.0;
        }
        ((bid_w - ask_w) / total).clamp(-1.0, 1.0)
    }

    // -----------------------------------------------------------------------
    // 2. Depth imbalance
    // -----------------------------------------------------------------------

    /// Compute the imbalance of all resting volume within `pct_range` of the midpoint.
    ///
    /// Only levels whose price is within `pct_range` (e.g. 0.02 = 2%) of the
    /// midpoint contribute.
    ///
    /// Returns 0.0 if no midpoint can be computed or no qualifying levels exist.
    pub fn depth_imbalance(&self, snapshot: &BookSnapshot, pct_range: f64) -> f64 {
        let mid = match snapshot.midpoint() {
            Some(m) => m,
            None => return 0.0,
        };

        let threshold = mid * pct_range;
        let mut bid_vol = 0.0f64;
        let mut ask_vol = 0.0f64;

        for (price, size) in snapshot.bid_prices.iter().zip(&snapshot.bid_sizes) {
            if (mid - price).abs() <= threshold {
                bid_vol += size;
            }
        }
        for (price, size) in snapshot.ask_prices.iter().zip(&snapshot.ask_sizes) {
            if (price - mid).abs() <= threshold {
                ask_vol += size;
            }
        }

        let total = bid_vol + ask_vol;
        if total < 1e-12 {
            return 0.0;
        }
        ((bid_vol - ask_vol) / total).clamp(-1.0, 1.0)
    }

    // -----------------------------------------------------------------------
    // 3. Iceberg detector
    // -----------------------------------------------------------------------

    /// Detect potential iceberg orders.
    ///
    /// A level is flagged as a potential iceberg when:
    ///   - It has resting visible size.
    ///   - The volume traded *at that exact price* during the trade window
    ///     significantly exceeds its visible size (ratio >= `iceberg_threshold`).
    ///
    /// In practice, icebergs replenish their visible quantity, so heavy trading
    /// at a stable size level is the tell.
    pub fn iceberg_detector(
        &self,
        snapshot: &BookSnapshot,
        trades: &[BookTrade],
    ) -> Vec<IcebergIndicator> {
        let mut indicators = Vec::new();

        // Aggregate traded volume at each price level from the trade tape
        let mut vol_at_price: std::collections::HashMap<u64, f64> =
            std::collections::HashMap::new();
        for trade in trades {
            // Quantize price to 4 decimal places to avoid float key issues
            let key = (trade.price * 10_000.0).round() as u64;
            *vol_at_price.entry(key).or_insert(0.0) += trade.volume;
        }

        // Check bid levels
        for (price, size) in snapshot.bid_prices.iter().zip(&snapshot.bid_sizes) {
            if *size < 1e-9 {
                continue;
            }
            let key = (price * 10_000.0).round() as u64;
            let traded = *vol_at_price.get(&key).unwrap_or(&0.0);
            let ratio = traded / size;
            if ratio >= self.iceberg_threshold {
                indicators.push(IcebergIndicator {
                    price: *price,
                    is_bid: true,
                    resting_size: *size,
                    traded_volume: traded,
                    iceberg_ratio: ratio,
                });
            }
        }

        // Check ask levels
        for (price, size) in snapshot.ask_prices.iter().zip(&snapshot.ask_sizes) {
            if *size < 1e-9 {
                continue;
            }
            let key = (price * 10_000.0).round() as u64;
            let traded = *vol_at_price.get(&key).unwrap_or(&0.0);
            let ratio = traded / size;
            if ratio >= self.iceberg_threshold {
                indicators.push(IcebergIndicator {
                    price: *price,
                    is_bid: false,
                    resting_size: *size,
                    traded_volume: traded,
                    iceberg_ratio: ratio,
                });
            }
        }

        indicators
    }

    // -----------------------------------------------------------------------
    // 4. Sweep detector
    // -----------------------------------------------------------------------

    /// Detect an aggressive sweep through multiple book levels.
    ///
    /// Looks at the last `window` trades for a sequence of:
    ///   - Monotonically ascending (buy sweep) or descending (sell sweep) prices.
    ///   - Spanning at least `sweep_min_levels` distinct price ticks.
    ///   - Occurring within the window.
    ///
    /// Returns the most recent sweep event, or `None` if no sweep is detected.
    pub fn sweep_detector(
        &self,
        trades: &[BookTrade],
        window: usize,
    ) -> Option<SweepEvent> {
        if trades.len() < 2 {
            return None;
        }

        let recent = {
            let n = trades.len().min(window);
            &trades[trades.len() - n..]
        };

        if recent.len() < 2 {
            return None;
        }

        // Identify the longest monotonic run in the recent window
        let best = self.longest_monotonic_run(recent);
        if best.levels_consumed >= self.sweep_min_levels {
            Some(best)
        } else {
            None
        }
    }

    /// Find the longest monotonically increasing or decreasing price run in `trades`.
    fn longest_monotonic_run(&self, trades: &[BookTrade]) -> SweepEvent {
        let n = trades.len();
        if n == 0 {
            return SweepEvent {
                is_buy: true,
                price_start: 0.0,
                price_end: 0.0,
                total_volume: 0.0,
                levels_consumed: 0,
                timestamp_ms: 0,
            };
        }

        let mut best_start = 0usize;
        let mut best_len = 1usize;
        let mut best_is_buy = true;
        let mut best_total_vol = trades[0].volume;

        let mut cur_start = 0usize;
        let mut cur_len = 1usize;
        let mut cur_is_buy = true;
        let mut cur_vol = trades[0].volume;

        for i in 1..n {
            let dp = trades[i].price - trades[i - 1].price;
            let going_up = dp >= 0.0;
            if i == 1 {
                cur_is_buy = going_up;
            }
            let continues = if cur_is_buy { dp >= 0.0 } else { dp <= 0.0 };

            if continues {
                cur_len += 1;
                cur_vol += trades[i].volume;
            } else {
                // New run starts
                cur_start = i;
                cur_len = 1;
                cur_is_buy = going_up;
                cur_vol = trades[i].volume;
            }

            // Count distinct price levels in current run
            let distinct = self.count_distinct_prices(&trades[cur_start..=i]);

            if distinct > best_len {
                best_len = distinct;
                best_start = cur_start;
                best_is_buy = cur_is_buy;
                best_total_vol = cur_vol;
            }
        }

        let run = &trades[best_start..best_start + best_len.min(n - best_start)];
        let total_vol: f64 = run.iter().map(|t| t.volume).sum();

        SweepEvent {
            is_buy: best_is_buy,
            price_start: run.first().map(|t| t.price).unwrap_or(0.0),
            price_end: run.last().map(|t| t.price).unwrap_or(0.0),
            total_volume: total_vol.max(best_total_vol),
            levels_consumed: best_len,
            timestamp_ms: run.first().map(|t| t.timestamp_ms).unwrap_or(0),
        }
    }

    fn count_distinct_prices(&self, trades: &[BookTrade]) -> usize {
        let mut prices: Vec<u64> =
            trades.iter().map(|t| (t.price * 10_000.0).round() as u64).collect();
        prices.sort_unstable();
        prices.dedup();
        prices.len()
    }
}

impl Default for BookImbalanceAnalyzer {
    fn default() -> Self {
        Self::new(3.0, 3)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric_book(levels: usize, base_bid: f64, base_ask: f64, size: f64) -> BookSnapshot {
        let bid_prices: Vec<f64> = (0..levels).map(|i| base_bid - i as f64 * 0.01).collect();
        let ask_prices: Vec<f64> = (0..levels).map(|i| base_ask + i as f64 * 0.01).collect();
        let bid_sizes = vec![size; levels];
        let ask_sizes = vec![size; levels];
        BookSnapshot::new(bid_prices, bid_sizes, ask_prices, ask_sizes)
    }

    fn skewed_book(bid_size: f64, ask_size: f64) -> BookSnapshot {
        BookSnapshot::new(
            vec![99.9, 99.8, 99.7],
            vec![bid_size, bid_size, bid_size],
            vec![100.1, 100.2, 100.3],
            vec![ask_size, ask_size, ask_size],
        )
    }

    // -- weighted_imbalance --------------------------------------------------

    #[test]
    fn test_book_imbalance_symmetry() {
        let analyzer = BookImbalanceAnalyzer::default();
        let book = symmetric_book(5, 99.9, 100.1, 100.0);
        let imb = analyzer.weighted_imbalance(&book, 5);
        assert!(
            imb.abs() < 1e-9,
            "symmetric book should have zero imbalance, got {}",
            imb
        );
    }

    #[test]
    fn test_book_imbalance_bid_dominated() {
        let analyzer = BookImbalanceAnalyzer::default();
        let book = skewed_book(1000.0, 100.0);
        let imb = analyzer.weighted_imbalance(&book, 3);
        assert!(imb > 0.0, "bid-heavy book should have positive imbalance, got {}", imb);
    }

    #[test]
    fn test_book_imbalance_ask_dominated() {
        let analyzer = BookImbalanceAnalyzer::default();
        let book = skewed_book(100.0, 1000.0);
        let imb = analyzer.weighted_imbalance(&book, 3);
        assert!(imb < 0.0, "ask-heavy book should have negative imbalance, got {}", imb);
    }

    #[test]
    fn test_book_imbalance_in_range() {
        let analyzer = BookImbalanceAnalyzer::default();
        let book = skewed_book(500.0, 300.0);
        let imb = analyzer.weighted_imbalance(&book, 3);
        assert!(imb >= -1.0 && imb <= 1.0, "imbalance out of [-1,1]: {}", imb);
    }

    #[test]
    fn test_weighted_imbalance_one_level() {
        let analyzer = BookImbalanceAnalyzer::default();
        // Single level: bid=200, ask=200 -> 0
        let book = BookSnapshot::new(
            vec![99.9],
            vec![200.0],
            vec![100.1],
            vec![200.0],
        );
        let imb = analyzer.weighted_imbalance(&book, 1);
        assert!(imb.abs() < 1e-9, "single symmetric level -> 0, got {}", imb);
    }

    // -- depth_imbalance -----------------------------------------------------

    #[test]
    fn test_depth_imbalance_within_range() {
        let analyzer = BookImbalanceAnalyzer::default();
        // Midpoint = 100.0; 2% range = 2.0
        let book = BookSnapshot::new(
            vec![99.9, 98.0], // 98.0 is outside 2% of 100 (100 - 2 = 98, boundary)
            vec![500.0, 500.0],
            vec![100.1, 102.5],
            vec![500.0, 500.0],
        );
        let imb = analyzer.depth_imbalance(&book, 0.02);
        // 99.9 is within 2%, 100.1 is within 2%
        // 98.0 is at boundary (|100-98|/100 = 2% = threshold): included
        // 102.5: |102.5-100|/100 = 2.5% > 2%: excluded from ask
        // bid = 500+500 = 1000, ask = 500 -> imb = (1000-500)/1500 > 0
        assert!(imb > 0.0, "more bid depth inside range -> positive, got {}", imb);
    }

    // -- iceberg_detector ----------------------------------------------------

    #[test]
    fn test_iceberg_detected_at_heavy_trade_level() {
        let analyzer = BookImbalanceAnalyzer::new(2.0, 3); // iceberg_threshold=2x
        let book = BookSnapshot::new(
            vec![99.9],
            vec![100.0],
            vec![100.1],
            vec![100.0],
        );
        // Trade 300 units through the ask at 100.1 (3x resting size)
        let trades = vec![
            BookTrade::new(100.1, 100.0),
            BookTrade::new(100.1, 100.0),
            BookTrade::new(100.1, 100.0),
        ];
        let icebergs = analyzer.iceberg_detector(&book, &trades);
        assert!(!icebergs.is_empty(), "should detect iceberg at ask level");
        assert_eq!(icebergs[0].price, 100.1);
        assert!(icebergs[0].iceberg_ratio >= 2.0);
    }

    #[test]
    fn test_iceberg_not_detected_on_low_volume() {
        let analyzer = BookImbalanceAnalyzer::default();
        let book = BookSnapshot::new(
            vec![99.9],
            vec![10_000.0],
            vec![100.1],
            vec![10_000.0],
        );
        // Only 10 units traded -- ratio = 10/10000 = 0.001 << 3.0 threshold
        let trades = vec![BookTrade::new(99.9, 10.0)];
        let icebergs = analyzer.iceberg_detector(&book, &trades);
        assert!(icebergs.is_empty(), "low trade volume should not trigger iceberg");
    }

    // -- sweep_detector ------------------------------------------------------

    #[test]
    fn test_sweep_detected_on_monotonic_run() {
        let analyzer = BookImbalanceAnalyzer::new(3.0, 3);
        let trades = vec![
            BookTrade::new(100.0, 500.0),
            BookTrade::new(100.1, 500.0),
            BookTrade::new(100.2, 500.0),
            BookTrade::new(100.3, 500.0),
        ];
        let sweep = analyzer.sweep_detector(&trades, 10);
        assert!(sweep.is_some(), "monotonic run should trigger sweep detection");
        let s = sweep.unwrap();
        assert!(s.is_buy, "ascending prices = buy sweep");
        assert!(s.levels_consumed >= 3);
    }

    #[test]
    fn test_no_sweep_for_short_window() {
        let analyzer = BookImbalanceAnalyzer::new(3.0, 5); // require 5 levels
        let trades = vec![
            BookTrade::new(100.0, 100.0),
            BookTrade::new(100.1, 100.0),
        ];
        let sweep = analyzer.sweep_detector(&trades, 10);
        assert!(sweep.is_none(), "only 2 distinct levels: should not trigger sweep");
    }

    #[test]
    fn test_sell_sweep_detected() {
        let analyzer = BookImbalanceAnalyzer::new(3.0, 3);
        let trades = vec![
            BookTrade::new(100.3, 500.0),
            BookTrade::new(100.2, 500.0),
            BookTrade::new(100.1, 500.0),
            BookTrade::new(100.0, 500.0),
        ];
        let sweep = analyzer.sweep_detector(&trades, 10);
        assert!(sweep.is_some(), "descending prices = sell sweep");
        let s = sweep.unwrap();
        assert!(!s.is_buy, "descending prices = sell sweep");
    }

    #[test]
    fn test_snapshot_midpoint() {
        let book = BookSnapshot::new(
            vec![99.8],
            vec![100.0],
            vec![100.2],
            vec![100.0],
        );
        let mid = book.midpoint().unwrap();
        assert!((mid - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_snapshot_spread() {
        let book = BookSnapshot::new(
            vec![99.8],
            vec![100.0],
            vec![100.2],
            vec![100.0],
        );
        let spread = book.spread().unwrap();
        assert!((spread - 0.4).abs() < 1e-9);
    }
}
