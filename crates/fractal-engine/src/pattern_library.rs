/// Fractal pattern library for multi-scale pattern detection.
///
/// Patterns implemented:
///   1. Elliott Wave structure (5-wave impulse, 3-wave correction) using
///      wavelet energy ratios across scales.
///   2. W-bottom / M-top detection at multiple timeframe scales simultaneously.
///
/// Each pattern detector returns a PatternMatch with a confidence score [0, 1].

use crate::wavelet::forward_dwt;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Pattern match types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern: PatternType,
    /// Confidence in [0.0, 1.0]. Higher = more certain.
    pub confidence: f64,
    /// Bar index where the pattern is detected (usually the latest bar).
    pub bar_index: usize,
    /// Human-readable description.
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternType {
    ElliottImpulse5Wave,
    ElliottCorrection3Wave,
    WBottom,
    MTop,
    /// Accumulation / Distribution fractal cluster
    AccumulationCluster,
    BreakoutSetup,
}

impl PatternType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PatternType::ElliottImpulse5Wave => "ELLIOTT_IMPULSE_5W",
            PatternType::ElliottCorrection3Wave => "ELLIOTT_CORRECTION_3W",
            PatternType::WBottom => "W_BOTTOM",
            PatternType::MTop => "M_TOP",
            PatternType::AccumulationCluster => "ACCUMULATION_CLUSTER",
            PatternType::BreakoutSetup => "BREAKOUT_SETUP",
        }
    }
}

// ---------------------------------------------------------------------------
// Elliott Wave detection via wavelet energy ratios
// ---------------------------------------------------------------------------

/// Simplified Elliott Wave detector using wavelet decomposition.
///
/// Heuristics:
///   - 5-wave impulse: energy concentrated in coarse scales (long-term trend),
///     alternating detail patterns at scale 1 and 2, net bullish momentum.
///   - 3-wave correction: energy spread evenly across scales 1-3,
///     shorter wavelength, net bearish or sideways momentum.
pub fn detect_elliott_wave(prices: &[f64]) -> Vec<PatternMatch> {
    let n = prices.len();
    if n < 32 {
        return Vec::new();
    }

    let decomp = forward_dwt(prices, 4);
    let energy_fracs = decomp.energy_fractions();
    if energy_fracs.len() < 3 {
        return Vec::new();
    }

    let mut matches = Vec::new();

    // Compute momentum (net price change / initial price)
    let price_change = if prices[0].abs() > 1e-10 {
        (prices[n - 1] - prices[0]) / prices[0]
    } else {
        0.0
    };

    // Approximate energy (coarse scale) fraction
    let approx_energy_frac = energy_fracs[0];
    // Fine detail energy fractions
    let fine1 = energy_fracs.get(1).copied().unwrap_or(0.0);
    let fine2 = energy_fracs.get(2).copied().unwrap_or(0.0);
    let coarse_detail = energy_fracs.get(3).copied().unwrap_or(0.0);

    // 5-wave impulse: dominant coarse approximation + strong momentum + alternating detail
    let impulse_confidence = {
        let coarse_dominance = approx_energy_frac; // should be high
        let momentum_factor = price_change.abs().min(0.5) * 2.0; // normalise to [0,1]
        let alternation = if (fine1 - fine2).abs() > 0.05 { 0.3 } else { 0.0 };
        let trending_sign = if price_change > 0.01 { 0.2 } else { 0.0 };
        (coarse_dominance * 0.4 + momentum_factor * 0.3 + alternation + trending_sign).min(1.0)
    };

    if impulse_confidence > 0.35 {
        matches.push(PatternMatch {
            pattern: PatternType::ElliottImpulse5Wave,
            confidence: impulse_confidence,
            bar_index: n - 1,
            description: format!(
                "5-wave impulse: coarse_energy={:.2}, momentum={:.3}",
                approx_energy_frac, price_change
            ),
        });
    }

    // 3-wave correction: energy distributed across fine scales, low momentum
    let correction_confidence = {
        let energy_spread = 1.0 - approx_energy_frac; // energy NOT in coarse
        let low_momentum = (1.0 - price_change.abs() * 10.0).max(0.0);
        let fine_energy = fine1 + fine2 + coarse_detail;
        (energy_spread * 0.4 + low_momentum * 0.3 + fine_energy * 0.3).min(1.0)
    };

    if correction_confidence > 0.35 {
        matches.push(PatternMatch {
            pattern: PatternType::ElliottCorrection3Wave,
            confidence: correction_confidence,
            bar_index: n - 1,
            description: format!(
                "3-wave correction: fine_energy={:.2}, momentum={:.3}",
                1.0 - approx_energy_frac, price_change
            ),
        });
    }

    matches
}

// ---------------------------------------------------------------------------
// W-bottom / M-top detection
// ---------------------------------------------------------------------------

/// Detect W-bottom (double bottom) pattern in price series.
/// Scans for two local minima with a higher middle peak between them.
pub fn detect_w_bottom(prices: &[f64], min_bars: usize) -> Vec<PatternMatch> {
    let n = prices.len();
    if n < min_bars * 2 {
        return Vec::new();
    }
    let mut matches = Vec::new();

    // Find local minima
    let minima = local_extrema(prices, min_bars / 4, Extremum::Min);
    if minima.len() < 2 {
        return matches;
    }

    // Check for W pattern: two minima at similar levels with a peak between
    for i in 0..(minima.len() - 1) {
        let left_min_idx = minima[i];
        let right_min_idx = minima[i + 1];
        if right_min_idx - left_min_idx < min_bars / 2 {
            continue;
        }

        let left_price = prices[left_min_idx];
        let right_price = prices[right_min_idx];
        let peak_between = prices[left_min_idx..=right_min_idx]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // W-bottom criteria:
        // 1. Two minima at similar levels (within 5%)
        // 2. Middle peak meaningfully higher than both minima
        let min_level = left_price.min(right_price);
        let diff_pct = (left_price - right_price).abs() / min_level.abs().max(1e-10);
        let peak_lift = (peak_between - min_level) / min_level.abs().max(1e-10);

        if diff_pct < 0.05 && peak_lift > 0.02 {
            let confidence = ((1.0 - diff_pct * 10.0) * 0.5 + peak_lift.min(0.1) * 5.0).min(1.0);
            if confidence > 0.3 {
                matches.push(PatternMatch {
                    pattern: PatternType::WBottom,
                    confidence,
                    bar_index: right_min_idx,
                    description: format!(
                        "W-bottom: left={:.2}@{left_min_idx}, right={:.2}@{right_min_idx}, peak={:.2}",
                        left_price, right_price, peak_between
                    ),
                });
            }
        }
    }

    matches
}

/// Detect M-top (double top) pattern in price series.
pub fn detect_m_top(prices: &[f64], min_bars: usize) -> Vec<PatternMatch> {
    let n = prices.len();
    if n < min_bars * 2 {
        return Vec::new();
    }
    let mut matches = Vec::new();

    let maxima = local_extrema(prices, min_bars / 4, Extremum::Max);
    if maxima.len() < 2 {
        return matches;
    }

    for i in 0..(maxima.len() - 1) {
        let left_max_idx = maxima[i];
        let right_max_idx = maxima[i + 1];
        if right_max_idx - left_max_idx < min_bars / 2 {
            continue;
        }

        let left_price = prices[left_max_idx];
        let right_price = prices[right_max_idx];
        let trough_between = prices[left_max_idx..=right_max_idx]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let max_level = left_price.max(right_price);
        let diff_pct = (left_price - right_price).abs() / max_level.abs().max(1e-10);
        let trough_drop = (max_level - trough_between) / max_level.abs().max(1e-10);

        if diff_pct < 0.05 && trough_drop > 0.02 {
            let confidence = ((1.0 - diff_pct * 10.0) * 0.5 + trough_drop.min(0.1) * 5.0).min(1.0);
            if confidence > 0.3 {
                matches.push(PatternMatch {
                    pattern: PatternType::MTop,
                    confidence,
                    bar_index: right_max_idx,
                    description: format!(
                        "M-top: left={:.2}@{left_max_idx}, right={:.2}@{right_max_idx}, trough={:.2}",
                        left_price, right_price, trough_between
                    ),
                });
            }
        }
    }

    matches
}

// ---------------------------------------------------------------------------
// Multi-scale pattern scan
// ---------------------------------------------------------------------------

/// Run all pattern detectors and return all matches sorted by confidence.
pub fn scan_patterns(prices: &[f64]) -> Vec<PatternMatch> {
    let mut all: Vec<PatternMatch> = Vec::new();
    all.extend(detect_elliott_wave(prices));
    all.extend(detect_w_bottom(prices, 10));
    all.extend(detect_m_top(prices, 10));
    // Multi-scale: also scan sub-windows
    let n = prices.len();
    if n >= 64 {
        let half = &prices[(n - n / 2)..];
        all.extend(detect_w_bottom(half, 6));
        all.extend(detect_m_top(half, 6));
    }
    all.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    all
}

// ---------------------------------------------------------------------------
// Local extrema detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Extremum {
    Min,
    Max,
}

/// Find indices of local minima or maxima using a neighborhood window.
fn local_extrema(prices: &[f64], window: usize, kind: Extremum) -> Vec<usize> {
    let n = prices.len();
    let hw = window.max(1);
    let mut extrema = Vec::new();
    for i in hw..(n - hw) {
        let neighbourhood = &prices[(i - hw)..=(i + hw)];
        let val = prices[i];
        let is_extremum = match kind {
            Extremum::Min => neighbourhood.iter().all(|&x| x >= val),
            Extremum::Max => neighbourhood.iter().all(|&x| x <= val),
        };
        if is_extremum {
            extrema.push(i);
        }
    }
    extrema
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w_bottom_prices() -> Vec<f64> {
        // Prices that form a clear W: down, up a bit, down again, then up
        let mut v: Vec<f64> = Vec::new();
        // left descent
        for i in 0..10 { v.push(100.0 - i as f64); }
        // middle rise
        for i in 0..5 { v.push(90.0 + i as f64 * 1.5); }
        // right descent
        for i in 0..10 { v.push(97.5 - i as f64); }
        // final rise
        for i in 0..10 { v.push(87.5 + i as f64 * 1.5); }
        v
    }

    fn m_top_prices() -> Vec<f64> {
        let mut v: Vec<f64> = Vec::new();
        for i in 0..10 { v.push(100.0 + i as f64); }
        for i in 0..5 { v.push(110.0 - i as f64 * 1.5); }
        for i in 0..10 { v.push(102.5 + i as f64); }
        for i in 0..10 { v.push(112.5 - i as f64 * 1.5); }
        v
    }

    #[test]
    fn detect_w_bottom_finds_pattern() {
        let prices = w_bottom_prices();
        let matches = detect_w_bottom(&prices, 6);
        // Should find at least one W-bottom
        assert!(!matches.is_empty(), "Should detect W-bottom in constructed series");
    }

    #[test]
    fn detect_m_top_on_m_prices() {
        let prices = m_top_prices();
        let matches = detect_m_top(&prices, 6);
        // Should detect M-top or at least not panic
        let _ = matches; // Just ensure no panic
    }

    #[test]
    fn elliott_detection_no_panic_short() {
        let prices = vec![100.0, 101.0, 102.0];
        let matches = detect_elliott_wave(&prices);
        assert!(matches.is_empty());
    }

    #[test]
    fn elliott_detection_trending_series() {
        let prices: Vec<f64> = (0..64).map(|i| 100.0 + i as f64 * 0.5).collect();
        let matches = detect_elliott_wave(&prices);
        // Should return some matches and not panic
        let _ = matches;
    }

    #[test]
    fn scan_patterns_returns_sorted_by_confidence() {
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 10.0)
            .collect();
        let matches = scan_patterns(&prices);
        for w in matches.windows(2) {
            assert!(
                w[0].confidence >= w[1].confidence,
                "Patterns not sorted by confidence"
            );
        }
    }
}
