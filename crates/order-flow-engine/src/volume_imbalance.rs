/// Order Flow Imbalance (OFI) calculation.
///
/// OFI = sum(buy_volume - sell_volume) / total_volume over a rolling window.
/// Range: [-1, 1].
/// OFI > +0.3  -> significant buy pressure
/// OFI < -0.3  -> significant sell pressure

pub const OFI_BUY_THRESHOLD: f64 = 0.3;
pub const OFI_SELL_THRESHOLD: f64 = -0.3;

/// A single bar's classified volumes.
#[derive(Debug, Clone, Copy)]
pub struct ClassifiedBar {
    pub buy_volume: f64,
    pub sell_volume: f64,
}

impl ClassifiedBar {
    pub fn new(buy_volume: f64, sell_volume: f64) -> Self {
        ClassifiedBar { buy_volume, sell_volume }
    }

    pub fn total_volume(&self) -> f64 {
        self.buy_volume + self.sell_volume
    }

    pub fn net_delta(&self) -> f64 {
        self.buy_volume - self.sell_volume
    }

    pub fn normalized_delta(&self) -> f64 {
        let total = self.total_volume();
        if total < 1e-9 {
            return 0.0;
        }
        self.net_delta() / total
    }
}

/// Pressure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Pressure {
    BuyDominated,
    SellDominated,
    Balanced,
}

impl std::fmt::Display for Pressure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pressure::BuyDominated => write!(f, "BUY_DOMINATED"),
            Pressure::SellDominated => write!(f, "SELL_DOMINATED"),
            Pressure::Balanced => write!(f, "BALANCED"),
        }
    }
}

/// Rolling OFI calculator.
///
/// Maintains a window of `ClassifiedBar` entries and computes the normalized
/// cumulative order flow imbalance over that window.
pub struct OFICalculator {
    window: std::collections::VecDeque<ClassifiedBar>,
    window_size: usize,
    /// Cumulative buy volume in window
    cum_buy: f64,
    /// Cumulative sell volume in window
    cum_sell: f64,
}

impl OFICalculator {
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0);
        OFICalculator {
            window: std::collections::VecDeque::with_capacity(window_size + 1),
            window_size,
            cum_buy: 0.0,
            cum_sell: 0.0,
        }
    }

    /// Push a new classified bar and return the updated OFI.
    pub fn push(&mut self, bar: ClassifiedBar) -> f64 {
        self.cum_buy += bar.buy_volume;
        self.cum_sell += bar.sell_volume;
        self.window.push_back(bar);

        // Evict oldest bar if window exceeds capacity
        if self.window.len() > self.window_size {
            let old = self.window.pop_front().unwrap();
            self.cum_buy -= old.buy_volume;
            self.cum_sell -= old.sell_volume;
        }

        self.ofi()
    }

    /// Current OFI value in [-1, 1].
    pub fn ofi(&self) -> f64 {
        let total = self.cum_buy + self.cum_sell;
        if total < 1e-9 {
            return 0.0;
        }
        ((self.cum_buy - self.cum_sell) / total).clamp(-1.0, 1.0)
    }

    /// Classify current OFI into a pressure label.
    pub fn pressure(&self) -> Pressure {
        let ofi = self.ofi();
        if ofi > OFI_BUY_THRESHOLD {
            Pressure::BuyDominated
        } else if ofi < OFI_SELL_THRESHOLD {
            Pressure::SellDominated
        } else {
            Pressure::Balanced
        }
    }

    /// Number of bars in the current window.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Compute per-bar delta (buy - sell) for each bar in the window.
    pub fn delta_series(&self) -> Vec<f64> {
        self.window.iter().map(|b| b.net_delta()).collect()
    }

    /// Cumulative delta (running sum of bar deltas) over the window.
    pub fn cumulative_delta(&self) -> f64 {
        self.cum_buy - self.cum_sell
    }

    /// Standard deviation of bar-level normalized deltas (imbalance volatility).
    pub fn delta_std(&self) -> f64 {
        let deltas: Vec<f64> = self
            .window
            .iter()
            .map(|b| b.normalized_delta())
            .collect();
        if deltas.len() < 2 {
            return 0.0;
        }
        let n = deltas.len() as f64;
        let mean = deltas.iter().sum::<f64>() / n;
        let var = deltas.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    }
}

/// Compute OFI for a complete slice of `ClassifiedBar` data with the given window.
/// Returns a Vec<f64> of OFI values, one per bar (first `window_size-1` values
/// are based on partial windows).
pub fn compute_ofi_series(bars: &[ClassifiedBar], window_size: usize) -> Vec<f64> {
    let mut calc = OFICalculator::new(window_size);
    bars.iter().map(|&b| calc.push(b)).collect()
}

/// Build a `ClassifiedBar` series from bulk-classified OHLCV using BVC.
pub fn ofi_from_ohlcv(
    ohlcv: &[(f64, f64, f64, f64, f64)], // (open, high, low, close, volume)
    window_size: usize,
) -> Vec<f64> {
    use crate::tick_classifier::BulkVolumeClassifier;
    let mut bvc = BulkVolumeClassifier::new(20);
    let bars: Vec<ClassifiedBar> = ohlcv
        .iter()
        .map(|&(o, _h, _l, c, v)| {
            let (buy, sell) = bvc.classify_bar(o, c, v);
            ClassifiedBar::new(buy, sell)
        })
        .collect();
    compute_ofi_series(&bars, window_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bar(buy: f64, sell: f64) -> ClassifiedBar {
        ClassifiedBar::new(buy, sell)
    }

    #[test]
    fn test_ofi_all_buys() {
        let mut calc = OFICalculator::new(5);
        for _ in 0..5 {
            calc.push(make_bar(1000.0, 0.0));
        }
        assert!((calc.ofi() - 1.0).abs() < 1e-6);
        assert_eq!(calc.pressure(), Pressure::BuyDominated);
    }

    #[test]
    fn test_ofi_all_sells() {
        let mut calc = OFICalculator::new(5);
        for _ in 0..5 {
            calc.push(make_bar(0.0, 1000.0));
        }
        assert!((calc.ofi() - (-1.0)).abs() < 1e-6);
        assert_eq!(calc.pressure(), Pressure::SellDominated);
    }

    #[test]
    fn test_ofi_balanced() {
        let mut calc = OFICalculator::new(5);
        for _ in 0..5 {
            calc.push(make_bar(500.0, 500.0));
        }
        assert!(calc.ofi().abs() < 1e-6);
        assert_eq!(calc.pressure(), Pressure::Balanced);
    }

    #[test]
    fn test_ofi_window_eviction() {
        let mut calc = OFICalculator::new(3);
        // Push 5 buy bars then 3 sell bars; window should only contain last 3
        for _ in 0..5 {
            calc.push(make_bar(1000.0, 0.0));
        }
        for _ in 0..3 {
            calc.push(make_bar(0.0, 1000.0));
        }
        assert!(calc.ofi() < -0.9, "window should be dominated by sells: {}", calc.ofi());
        assert_eq!(calc.len(), 3);
    }

    #[test]
    fn test_ofi_series_length() {
        let bars: Vec<ClassifiedBar> = (0..20)
            .map(|i| make_bar(i as f64 * 10.0, (20 - i) as f64 * 10.0))
            .collect();
        let series = compute_ofi_series(&bars, 5);
        assert_eq!(series.len(), 20);
    }

    #[test]
    fn test_normalized_delta_range() {
        let b = make_bar(300.0, 700.0);
        let nd = b.normalized_delta();
        assert!(nd >= -1.0 && nd <= 1.0);
        assert!((nd - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn test_cumulative_delta() {
        let mut calc = OFICalculator::new(10);
        calc.push(make_bar(600.0, 400.0)); // +200
        calc.push(make_bar(300.0, 700.0)); // -400
        // cumulative: 900 - 1100 = -200
        assert!((calc.cumulative_delta() - (-200.0)).abs() < 1e-6);
    }
}
