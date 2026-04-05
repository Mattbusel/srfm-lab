/// Regime detector combining Hurst exponent and fractal dimension.
///
/// Classification:
///   TRENDING      : H > 0.6 AND FD < 1.4 → persistent directional moves
///   MEAN_REVERTING: H < 0.4 AND FD < 1.5 → oscillating around a level
///   CHOPPY        : H < 0.5 AND FD > 1.6 → noisy, high-frequency randomness
///   TRANSITIONING : else → regime is changing, sizing should be reduced
///
/// Also provides position-sizing multipliers calibrated to each regime.

use crate::fractal_dimension::{higuchi_fd, rolling_fd, FdClass};
use crate::hurst::{hurst_rs, rolling_hurst, HurstRegime};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Regime types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Regime {
    /// H > 0.6, FD < 1.4 — strong persistent trend
    Trending,
    /// H < 0.4, FD < 1.5 — clear mean reversion
    MeanReverting,
    /// H < 0.5, FD > 1.6 — noisy, disorganised
    Choppy,
    /// Otherwise — transition or ambiguous
    Transitioning,
    /// Insufficient data
    Unknown,
}

impl Regime {
    pub fn as_str(&self) -> &'static str {
        match self {
            Regime::Trending => "TRENDING",
            Regime::MeanReverting => "MEAN_REVERTING",
            Regime::Choppy => "CHOPPY",
            Regime::Transitioning => "TRANSITIONING",
            Regime::Unknown => "UNKNOWN",
        }
    }

    /// Position-sizing multiplier for this regime.
    /// Trending: full sizing. MeanReverting: moderate. Choppy: minimal.
    pub fn size_multiplier(&self) -> f64 {
        match self {
            Regime::Trending => 1.0,
            Regime::MeanReverting => 0.7,
            Regime::Transitioning => 0.5,
            Regime::Choppy => 0.2,
            Regime::Unknown => 0.1,
        }
    }

    /// Recommended strategy bias for this regime.
    pub fn strategy_bias(&self) -> &'static str {
        match self {
            Regime::Trending => "BH (trend-following, momentum)",
            Regime::MeanReverting => "OU (mean-reversion, stat-arb)",
            Regime::Transitioning => "reduce_size (mixed, await clarity)",
            Regime::Choppy => "neutral (market-making, very small size)",
            Regime::Unknown => "neutral (await data)",
        }
    }
}

// ---------------------------------------------------------------------------
// Regime snapshot
// ---------------------------------------------------------------------------

/// Full regime snapshot at a single bar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeSnapshot {
    pub bar_index: usize,
    pub regime: Regime,
    pub hurst: Option<f64>,
    pub hurst_regime: HurstRegime,
    pub fractal_dimension: Option<f64>,
    pub fd_class: Option<FdClass>,
    pub size_multiplier: f64,
    pub strategy_bias: String,
}

impl RegimeSnapshot {
    pub fn new(bar_index: usize, hurst: Option<f64>, fd: Option<f64>) -> Self {
        let hurst_regime = HurstRegime::from_h(hurst);
        let fd_class = fd.map(FdClass::from_fd);
        let regime = classify_regime(hurst, fd);
        let size_multiplier = regime.size_multiplier();
        let strategy_bias = regime.strategy_bias().to_string();
        Self {
            bar_index,
            regime,
            hurst,
            hurst_regime,
            fractal_dimension: fd,
            fd_class,
            size_multiplier,
            strategy_bias,
        }
    }
}

// ---------------------------------------------------------------------------
// Regime classification
// ---------------------------------------------------------------------------

/// Classify market regime from Hurst exponent and fractal dimension.
pub fn classify_regime(hurst: Option<f64>, fd: Option<f64>) -> Regime {
    match (hurst, fd) {
        (Some(h), Some(f)) => {
            if h > 0.6 && f < 1.4 {
                Regime::Trending
            } else if h < 0.4 && f < 1.5 {
                Regime::MeanReverting
            } else if h < 0.5 && f > 1.6 {
                Regime::Choppy
            } else {
                Regime::Transitioning
            }
        }
        (Some(h), None) => {
            if h > 0.65 {
                Regime::Trending
            } else if h < 0.35 {
                Regime::MeanReverting
            } else {
                Regime::Transitioning
            }
        }
        (None, Some(f)) => {
            if f < 1.3 {
                Regime::Trending
            } else if f > 1.7 {
                Regime::Choppy
            } else {
                Regime::Transitioning
            }
        }
        (None, None) => Regime::Unknown,
    }
}

// ---------------------------------------------------------------------------
// Rolling regime detector
// ---------------------------------------------------------------------------

/// Compute regime snapshots over a rolling window across a full price series.
pub fn rolling_regime(prices: &[f64], hurst_window: usize, fd_window: usize, fd_k_max: usize) -> Vec<RegimeSnapshot> {
    let n = prices.len();
    let h_vec = rolling_hurst(prices, hurst_window);
    let fd_vec = rolling_fd(prices, fd_window, fd_k_max);

    (0..n)
        .map(|i| {
            let h = if h_vec[i].is_finite() { Some(h_vec[i]) } else { None };
            let fd = if fd_vec[i].is_finite() { Some(fd_vec[i]) } else { None };
            RegimeSnapshot::new(i, h, fd)
        })
        .collect()
}

/// Compute regime for the latest window of prices.
pub fn current_regime(prices: &[f64], hurst_window: usize, fd_k_max: usize) -> RegimeSnapshot {
    let n = prices.len();
    let bar_index = n.saturating_sub(1);

    let hurst_slice = if n >= hurst_window {
        &prices[(n - hurst_window)..]
    } else {
        prices
    };
    let fd_slice = if n >= 64 {
        &prices[(n - 64)..]
    } else {
        prices
    };

    let hurst = hurst_rs(hurst_slice);
    let fd = higuchi_fd(fd_slice, fd_k_max).map(|fa| fa.fd);

    RegimeSnapshot::new(bar_index, hurst, fd)
}

// ---------------------------------------------------------------------------
// Regime statistics over a history
// ---------------------------------------------------------------------------

/// Summary statistics about regime distribution over a history.
#[derive(Debug, Serialize, Deserialize)]
pub struct RegimeStats {
    pub total_bars: usize,
    pub trending_pct: f64,
    pub mean_reverting_pct: f64,
    pub choppy_pct: f64,
    pub transitioning_pct: f64,
    pub unknown_pct: f64,
    pub mean_size_multiplier: f64,
}

impl RegimeStats {
    pub fn from_snapshots(snapshots: &[RegimeSnapshot]) -> Self {
        let n = snapshots.len();
        if n == 0 {
            return Self {
                total_bars: 0,
                trending_pct: 0.0,
                mean_reverting_pct: 0.0,
                choppy_pct: 0.0,
                transitioning_pct: 0.0,
                unknown_pct: 0.0,
                mean_size_multiplier: 0.0,
            };
        }
        let count = |target: &Regime| snapshots.iter().filter(|s| &s.regime == target).count();
        let trending = count(&Regime::Trending);
        let mean_rev = count(&Regime::MeanReverting);
        let choppy = count(&Regime::Choppy);
        let trans = count(&Regime::Transitioning);
        let unknown = count(&Regime::Unknown);
        let mean_size: f64 = snapshots.iter().map(|s| s.size_multiplier).sum::<f64>() / n as f64;
        Self {
            total_bars: n,
            trending_pct: trending as f64 / n as f64 * 100.0,
            mean_reverting_pct: mean_rev as f64 / n as f64 * 100.0,
            choppy_pct: choppy as f64 / n as f64 * 100.0,
            transitioning_pct: trans as f64 / n as f64 * 100.0,
            unknown_pct: unknown as f64 / n as f64 * 100.0,
            mean_size_multiplier: mean_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn trending_prices(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
    }

    fn noisy_prices(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut price = 100.0f64;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
                price *= 1.0 + r * 0.02;
                price
            })
            .collect()
    }

    #[test]
    fn regime_trending_classification() {
        let r = classify_regime(Some(0.75), Some(1.2));
        assert_eq!(r, Regime::Trending);
    }

    #[test]
    fn regime_mean_reverting_classification() {
        let r = classify_regime(Some(0.3), Some(1.4));
        assert_eq!(r, Regime::MeanReverting);
    }

    #[test]
    fn regime_choppy_classification() {
        let r = classify_regime(Some(0.4), Some(1.7));
        assert_eq!(r, Regime::Choppy);
    }

    #[test]
    fn regime_unknown_without_data() {
        let r = classify_regime(None, None);
        assert_eq!(r, Regime::Unknown);
    }

    #[test]
    fn regime_snapshot_creates_correctly() {
        let snap = RegimeSnapshot::new(100, Some(0.7), Some(1.3));
        assert_eq!(snap.regime, Regime::Trending);
        assert!(snap.size_multiplier > 0.0);
    }

    #[test]
    fn current_regime_no_panic() {
        let prices = trending_prices(200);
        let snap = current_regime(&prices, 128, 8);
        // Just ensure no panic and valid output
        let _ = snap.regime.as_str();
    }

    #[test]
    fn regime_stats_totals_100_pct() {
        let prices = noisy_prices(300, 42);
        let snapshots = rolling_regime(&prices, 128, 64, 8);
        let stats = RegimeStats::from_snapshots(&snapshots);
        let total = stats.trending_pct
            + stats.mean_reverting_pct
            + stats.choppy_pct
            + stats.transitioning_pct
            + stats.unknown_pct;
        assert!((total - 100.0).abs() < 0.01, "Regime percentages should sum to 100%, got {total}");
    }

    #[test]
    fn size_multipliers_ordered() {
        assert!(Regime::Trending.size_multiplier() > Regime::MeanReverting.size_multiplier());
        assert!(Regime::MeanReverting.size_multiplier() > Regime::Choppy.size_multiplier());
        assert!(Regime::Choppy.size_multiplier() > Regime::Unknown.size_multiplier());
    }
}
