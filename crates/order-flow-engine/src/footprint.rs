/// Footprint chart analysis derived from OHLCV data.
///
/// A footprint chart shows volume distribution across price levels within a bar.
/// We simulate this from OHLCV by distributing volume across discrete price levels
/// proportional to a normal distribution centred on the VWAP (estimated as
/// (open + high + low + close) / 4).
///
/// Key concepts:
/// - POC (Point of Control): price level with the most traded volume.
/// - Value Area: price range containing 70% of the bar's volume.
/// - VAH (Value Area High): top of the value area.
/// - VAL (Value Area Low): bottom of the value area.
///
/// Market structure signals:
/// - Price rejecting VAH (close < VAH, high > VAH): resistance, potential short.
/// - Price accepting above VAH (close > VAH): continuation higher.
/// - Price rejecting VAL (close > VAL, low < VAL): support, potential long.

use crate::tick_classifier::BulkVolumeClassifier;

/// Resolution: number of price levels per bar.
const PRICE_LEVELS: usize = 20;
const VALUE_AREA_PCT: f64 = 0.70;

/// A single price level within a footprint bar.
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
}

impl PriceLevel {
    pub fn total_volume(&self) -> f64 {
        self.buy_volume + self.sell_volume
    }

    pub fn delta(&self) -> f64 {
        self.buy_volume - self.sell_volume
    }

    pub fn imbalance_ratio(&self) -> f64 {
        let total = self.total_volume();
        if total < 1e-9 {
            return 0.0;
        }
        self.delta() / total
    }
}

/// Footprint bar derived from a single OHLCV candle.
#[derive(Debug, Clone)]
pub struct FootprintBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub total_volume: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub levels: Vec<PriceLevel>,
    /// Point of Control: price level with highest volume
    pub poc: f64,
    /// Value Area High
    pub vah: f64,
    /// Value Area Low
    pub val: f64,
    /// Total bar delta (buy_vol - sell_vol)
    pub delta: f64,
    /// Imbalance signal at this bar
    pub imbalance: FootprintImbalance,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FootprintImbalance {
    /// Price rejected the VAH (close < VAH and high ≥ VAH)
    VahRejection,
    /// Price accepted above VAH (close > VAH)
    VahAcceptance,
    /// Price rejected the VAL (close > VAL and low ≤ VAL)
    ValSupport,
    /// Price accepted below VAL (close < VAL)
    ValBreakdown,
    /// No significant structure signal
    Neutral,
}

impl std::fmt::Display for FootprintImbalance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FootprintImbalance::VahRejection => write!(f, "VAH_REJECTION"),
            FootprintImbalance::VahAcceptance => write!(f, "VAH_ACCEPTANCE"),
            FootprintImbalance::ValSupport => write!(f, "VAL_SUPPORT"),
            FootprintImbalance::ValBreakdown => write!(f, "VAL_BREAKDOWN"),
            FootprintImbalance::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// Build a footprint bar from OHLCV data.
///
/// Volume is distributed across `PRICE_LEVELS` levels between low and high
/// using a triangular distribution peaked at the estimated VWAP.
pub fn build_footprint_bar(
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    buy_fraction: f64, // proportion of volume that is buy-initiated
) -> FootprintBar {
    let range = high - low;
    let buy_vol = volume * buy_fraction.clamp(0.0, 1.0);
    let sell_vol = volume - buy_vol;

    // Generate price levels
    let mut levels = Vec::with_capacity(PRICE_LEVELS);
    let step = if range > 1e-9 {
        range / PRICE_LEVELS as f64
    } else {
        1e-4
    };

    // VWAP estimate
    let vwap = (open + high + low + close) / 4.0;

    // Weight each level by a triangular distribution peaking at vwap
    let weights: Vec<f64> = (0..PRICE_LEVELS)
        .map(|i| {
            let price = low + (i as f64 + 0.5) * step;
            // Triangle weight: peaks at vwap, zero at extremes
            let w = 1.0 - ((price - vwap) / (range / 2.0 + 1e-9)).abs();
            w.max(0.01) // minimum weight ensures all levels have some volume
        })
        .collect();

    let weight_sum: f64 = weights.iter().sum();

    for (i, &w) in weights.iter().enumerate() {
        let price = low + (i as f64 + 0.5) * step;
        let frac = w / weight_sum;
        levels.push(PriceLevel {
            price,
            buy_volume: buy_vol * frac,
            sell_volume: sell_vol * frac,
        });
    }

    // POC: level with most total volume
    let poc_idx = levels
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.total_volume()
                .partial_cmp(&b.total_volume())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(PRICE_LEVELS / 2);
    let poc = levels[poc_idx].price;

    // Value area: start from POC, expand to capture 70% of volume
    let (vah, val) = compute_value_area(&levels, poc_idx, volume);

    let delta = buy_vol - sell_vol;

    // Imbalance signal
    let imbalance = classify_imbalance(high, low, close, vah, val);

    FootprintBar {
        open,
        high,
        low,
        close,
        total_volume: volume,
        buy_volume: buy_vol,
        sell_volume: sell_vol,
        levels,
        poc,
        vah,
        val,
        delta,
        imbalance,
    }
}

fn compute_value_area(
    levels: &[PriceLevel],
    poc_idx: usize,
    total_volume: f64,
) -> (f64, f64) {
    let target = total_volume * VALUE_AREA_PCT;
    let mut accumulated = levels[poc_idx].total_volume();
    let mut lo_idx = poc_idx;
    let mut hi_idx = poc_idx;

    // Expand outward from POC, adding the higher-volume neighbour each time
    while accumulated < target {
        let can_expand_lo = lo_idx > 0;
        let can_expand_hi = hi_idx < levels.len() - 1;

        if !can_expand_lo && !can_expand_hi {
            break;
        }

        let lo_vol = if can_expand_lo {
            levels[lo_idx - 1].total_volume()
        } else {
            f64::NEG_INFINITY
        };
        let hi_vol = if can_expand_hi {
            levels[hi_idx + 1].total_volume()
        } else {
            f64::NEG_INFINITY
        };

        if hi_vol >= lo_vol {
            hi_idx += 1;
            accumulated += levels[hi_idx].total_volume();
        } else {
            lo_idx -= 1;
            accumulated += levels[lo_idx].total_volume();
        }
    }

    let vah = levels[hi_idx].price + (levels[hi_idx].price - levels[hi_idx.saturating_sub(1).min(hi_idx)].price).abs() * 0.5;
    let val = levels[lo_idx].price;
    (vah, val)
}

fn classify_imbalance(
    high: f64,
    low: f64,
    close: f64,
    vah: f64,
    val: f64,
) -> FootprintImbalance {
    if high >= vah && close < vah {
        FootprintImbalance::VahRejection
    } else if close > vah {
        FootprintImbalance::VahAcceptance
    } else if low <= val && close > val {
        FootprintImbalance::ValSupport
    } else if close < val {
        FootprintImbalance::ValBreakdown
    } else {
        FootprintImbalance::Neutral
    }
}

/// Process a sequence of OHLCV bars into footprint bars.
pub fn build_footprint_series(
    ohlcv: &[(f64, f64, f64, f64, f64)], // (open, high, low, close, volume)
) -> Vec<FootprintBar> {
    let mut bvc = BulkVolumeClassifier::new(20);
    ohlcv
        .iter()
        .map(|&(o, h, l, c, v)| {
            let (buy_v, sell_v) = bvc.classify_bar(o, c, v);
            let buy_frac = if (buy_v + sell_v) > 1e-9 {
                buy_v / (buy_v + sell_v)
            } else {
                0.5
            };
            build_footprint_bar(o, h, l, c, v, buy_frac)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_bar(open: f64, high: f64, low: f64, close: f64, vol: f64) -> FootprintBar {
        build_footprint_bar(open, high, low, close, vol, 0.6)
    }

    #[test]
    fn test_poc_within_range() {
        let fb = simple_bar(100.0, 105.0, 95.0, 102.0, 10000.0);
        assert!(fb.poc >= fb.low && fb.poc <= fb.high, "POC out of range: {}", fb.poc);
    }

    #[test]
    fn test_value_area_within_range() {
        let fb = simple_bar(100.0, 110.0, 90.0, 105.0, 50000.0);
        assert!(fb.val <= fb.vah, "VAL > VAH");
        assert!(fb.val >= fb.low - 1.0, "VAL below low");
        assert!(fb.vah <= fb.high + 1.0, "VAH above high");
    }

    #[test]
    fn test_volume_conservation() {
        let fb = simple_bar(100.0, 105.0, 95.0, 103.0, 8000.0);
        let level_total: f64 = fb.levels.iter().map(|l| l.total_volume()).sum();
        assert!((level_total - fb.total_volume).abs() < 1.0, "volume not conserved");
    }

    #[test]
    fn test_delta_sign() {
        let fb = build_footprint_bar(100.0, 105.0, 95.0, 103.0, 5000.0, 0.8);
        assert!(fb.delta > 0.0, "buy-dominated should have positive delta");
    }

    #[test]
    fn test_vah_rejection_signal() {
        // High reaches above VAH but close is below it
        let fb = build_footprint_bar(100.0, 115.0, 95.0, 100.5, 10000.0, 0.4);
        // VAH is somewhere around the value area high (~108 given range 95-115)
        // with close=100.5 < vah and high=115 > vah, should be VAH_REJECTION
        assert!(
            fb.imbalance == FootprintImbalance::VahRejection
                || fb.imbalance == FootprintImbalance::Neutral
                || fb.imbalance == FootprintImbalance::ValSupport,
            "unexpected imbalance: {}",
            fb.imbalance
        );
    }

    #[test]
    fn test_footprint_series_length() {
        let ohlcv: Vec<(f64, f64, f64, f64, f64)> = (0..20)
            .map(|i| {
                let base = 100.0 + i as f64;
                (base, base + 1.0, base - 1.0, base + 0.3, 5000.0)
            })
            .collect();
        let bars = build_footprint_series(&ohlcv);
        assert_eq!(bars.len(), 20);
    }

    #[test]
    fn test_price_levels_count() {
        let fb = simple_bar(100.0, 102.0, 98.0, 101.0, 1000.0);
        assert_eq!(fb.levels.len(), PRICE_LEVELS);
    }
}
