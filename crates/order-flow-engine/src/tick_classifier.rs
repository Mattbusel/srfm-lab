/// Classification of individual trades as buyer- or seller-initiated.

/// Direction of a trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TickSide {
    Buy,
    Sell,
    Unknown,
}

impl std::fmt::Display for TickSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TickSide::Buy => write!(f, "BUY"),
            TickSide::Sell => write!(f, "SELL"),
            TickSide::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// A single tick (trade) with price, volume, and bid/ask quotes.
#[derive(Debug, Clone)]
pub struct Tick {
    pub price: f64,
    pub volume: f64,
    /// Bid price at the time of the trade (optional)
    pub bid: Option<f64>,
    /// Ask price at the time of the trade (optional)
    pub ask: Option<f64>,
}

/// Classify using the Lee-Ready rule.
///
/// Priority:
/// 1. Quote rule: if bid/ask available, compare trade price to midpoint.
///    - price > mid  -> BUY
///    - price < mid  -> SELL
///    - price == mid -> fall through to tick rule
/// 2. Tick rule: compare to previous trade price.
///    - uptick  -> BUY
///    - downtick -> SELL
///    - zero-tick -> use last non-zero tick direction
pub struct LeeReadyClassifier {
    /// Previous trade price
    prev_price: Option<f64>,
    /// Last non-zero tick direction (for zero-tick resolution)
    last_tick_direction: TickSide,
}

impl LeeReadyClassifier {
    pub fn new() -> Self {
        LeeReadyClassifier {
            prev_price: None,
            last_tick_direction: TickSide::Unknown,
        }
    }

    /// Classify a single tick. Updates internal state.
    pub fn classify(&mut self, tick: &Tick) -> TickSide {
        let side = self.classify_inner(tick);
        // Update prev price
        self.prev_price = Some(tick.price);
        // Update last non-zero tick direction if not unknown
        if side != TickSide::Unknown {
            self.last_tick_direction = side;
        }
        side
    }

    fn classify_inner(&self, tick: &Tick) -> TickSide {
        // Quote rule (requires bid/ask)
        if let (Some(bid), Some(ask)) = (tick.bid, tick.ask) {
            let mid = (bid + ask) / 2.0;
            if tick.price > mid {
                return TickSide::Buy;
            } else if tick.price < mid {
                return TickSide::Sell;
            }
            // price == mid, fall through to tick rule
        }

        // Tick rule
        match self.prev_price {
            None => TickSide::Unknown,
            Some(prev) => {
                if tick.price > prev {
                    TickSide::Buy
                } else if tick.price < prev {
                    TickSide::Sell
                } else {
                    // Zero tick: use last non-zero tick direction
                    self.last_tick_direction
                }
            }
        }
    }

    /// Reset internal state (e.g., between trading sessions).
    pub fn reset(&mut self) {
        self.prev_price = None;
        self.last_tick_direction = TickSide::Unknown;
    }
}

impl Default for LeeReadyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Classify a slice of ticks and return (buy_volume, sell_volume).
pub fn classify_ticks(ticks: &[Tick]) -> (f64, f64) {
    let mut classifier = LeeReadyClassifier::new();
    let mut buy_vol = 0.0f64;
    let mut sell_vol = 0.0f64;
    for tick in ticks {
        match classifier.classify(tick) {
            TickSide::Buy => buy_vol += tick.volume,
            TickSide::Sell => sell_vol += tick.volume,
            TickSide::Unknown => {} // split evenly
        }
    }
    (buy_vol, sell_vol)
}

// ── Bulk Volume Classifier ────────────────────────────────────────────────────

/// BVC classifies bar-level OHLCV data without tick data.
///
/// proportion_buy = 0.5 + 0.5 * erf(Z / sqrt(2))
/// where Z = bar_return / (sigma * sqrt(2))
///
/// This is the CDF of a standard normal evaluated at bar_return / sigma,
/// giving the fraction of volume estimated as buyer-initiated.
#[derive(Debug, Clone)]
pub struct BulkVolumeClassifier {
    /// Rolling standard deviation of returns (annualised at bar frequency)
    pub sigma: f64,
    /// Window of recent returns used to estimate sigma
    return_window: Vec<f64>,
    window_size: usize,
}

impl BulkVolumeClassifier {
    pub fn new(window_size: usize) -> Self {
        BulkVolumeClassifier {
            sigma: 0.008, // default 0.8% per bar
            return_window: Vec::with_capacity(window_size + 1),
            window_size,
        }
    }

    /// Classify one OHLCV bar.
    ///
    /// Returns `(buy_volume, sell_volume)` estimated from bar return.
    pub fn classify_bar(&mut self, open: f64, close: f64, volume: f64) -> (f64, f64) {
        let ret = if open > 0.0 {
            (close - open) / open
        } else {
            0.0
        };

        // Update rolling sigma
        self.return_window.push(ret);
        if self.return_window.len() > self.window_size {
            self.return_window.remove(0);
        }
        if self.return_window.len() >= 5 {
            self.sigma = rolling_std(&self.return_window).max(1e-6);
        }

        let proportion_buy = proportion_buy(ret, self.sigma);
        let buy_vol = volume * proportion_buy;
        let sell_vol = volume * (1.0 - proportion_buy);
        (buy_vol, sell_vol)
    }

    /// Process a batch of (open, close, volume) tuples.
    pub fn classify_bars(&mut self, bars: &[(f64, f64, f64)]) -> Vec<(f64, f64)> {
        bars.iter()
            .map(|&(o, c, v)| self.classify_bar(o, c, v))
            .collect()
    }
}

/// proportion_buy = Φ(return / sigma) where Φ is the standard normal CDF.
/// Implemented via erf: Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
pub fn proportion_buy(bar_return: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.5;
    }
    let z = bar_return / sigma;
    // standard normal CDF
    let p = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    p.clamp(0.0, 1.0)
}

/// Approximate erf using Abramowitz & Stegun formula 7.1.26 (max error 1.5e-7).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    sign * result
}

fn rolling_std(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lee_ready_uptick_buy() {
        let mut clf = LeeReadyClassifier::new();
        let t1 = Tick { price: 100.0, volume: 10.0, bid: None, ask: None };
        let t2 = Tick { price: 100.5, volume: 10.0, bid: None, ask: None };
        clf.classify(&t1);
        assert_eq!(clf.classify(&t2), TickSide::Buy);
    }

    #[test]
    fn test_lee_ready_downtick_sell() {
        let mut clf = LeeReadyClassifier::new();
        let t1 = Tick { price: 100.0, volume: 10.0, bid: None, ask: None };
        let t2 = Tick { price: 99.5, volume: 10.0, bid: None, ask: None };
        clf.classify(&t1);
        assert_eq!(clf.classify(&t2), TickSide::Sell);
    }

    #[test]
    fn test_lee_ready_quote_rule() {
        let mut clf = LeeReadyClassifier::new();
        let t = Tick {
            price: 100.3,
            volume: 5.0,
            bid: Some(100.0),
            ask: Some(100.4),
        };
        // mid = 100.2, price > mid -> BUY
        assert_eq!(clf.classify(&t), TickSide::Buy);
    }

    #[test]
    fn test_lee_ready_zero_tick_inherits_last() {
        let mut clf = LeeReadyClassifier::new();
        // Establish uptick
        clf.classify(&Tick { price: 99.0, volume: 1.0, bid: None, ask: None });
        clf.classify(&Tick { price: 100.0, volume: 1.0, bid: None, ask: None }); // uptick=BUY
        // Zero tick
        let side = clf.classify(&Tick { price: 100.0, volume: 1.0, bid: None, ask: None });
        assert_eq!(side, TickSide::Buy);
    }

    #[test]
    fn test_proportion_buy_positive_return() {
        let p = proportion_buy(0.01, 0.01);
        assert!(p > 0.5, "positive return -> proportion_buy > 0.5, got {}", p);
    }

    #[test]
    fn test_proportion_buy_zero_return() {
        let p = proportion_buy(0.0, 0.01);
        assert!((p - 0.5).abs() < 0.01, "zero return -> ~0.5, got {}", p);
    }

    #[test]
    fn test_bvc_volume_sums_to_total() {
        let mut bvc = BulkVolumeClassifier::new(20);
        let (buy, sell) = bvc.classify_bar(100.0, 101.0, 1000.0);
        assert!((buy + sell - 1000.0).abs() < 1e-6, "buy+sell should equal volume");
    }

    #[test]
    fn test_erf_known_values() {
        // erf(0) = 0
        assert!((erf(0.0)).abs() < 1e-6);
        // erf(inf) ~ 1
        assert!((erf(5.0) - 1.0).abs() < 1e-4);
        // erf is odd
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-6);
    }
}
