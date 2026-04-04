/// Rule-based and composite regime classification.

use serde::{Deserialize, Serialize};

// ── Regime Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Regime {
    Bull,
    Bear,
    Sideways,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolRegime {
    LowVol,
    NormalVol,
    HighVol,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MacroRegime {
    RiskOn,
    RiskOff,
    Stagflation,
    Goldilocks,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeRegime {
    pub trend: Regime,
    pub vol: VolRegime,
    pub macro_regime: MacroRegime,
    /// Overall confidence [0, 1].
    pub confidence: f64,
    /// Human-readable summary.
    pub label: String,
}

// ── Moving Average helpers ────────────────────────────────────────────────────

fn sma_last(prices: &[f64], window: usize) -> f64 {
    let n = prices.len();
    if n < window { return prices.last().copied().unwrap_or(0.0); }
    prices[n - window..].iter().sum::<f64>() / window as f64
}

fn rolling_std(series: &[f64], window: usize) -> f64 {
    let n = series.len();
    let w = window.min(n);
    if w < 2 { return 0.0; }
    let slice = &series[n - w..];
    let m = slice.iter().sum::<f64>() / w as f64;
    let var = slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (w - 1) as f64;
    var.sqrt()
}

// ── Trend Regime ──────────────────────────────────────────────────────────────

/// Classify trend regime using two moving averages.
pub fn trend_regime(prices: &[f64], ma_fast: usize, ma_slow: usize) -> Regime {
    if prices.len() < ma_slow {
        return Regime::Sideways;
    }
    let fast = sma_last(prices, ma_fast);
    let slow = sma_last(prices, ma_slow);
    let price = *prices.last().unwrap();
    let threshold = slow * 0.005; // 0.5% threshold to avoid noise

    if fast > slow + threshold && price > fast {
        Regime::Bull
    } else if fast < slow - threshold && price < fast {
        Regime::Bear
    } else {
        Regime::Sideways
    }
}

/// Trend regime with trend strength score in [-1, 1].
pub fn trend_regime_with_score(prices: &[f64], ma_fast: usize, ma_slow: usize) -> (Regime, f64) {
    if prices.len() < ma_slow {
        return (Regime::Sideways, 0.0);
    }
    let fast = sma_last(prices, ma_fast);
    let slow = sma_last(prices, ma_slow);
    let score = (fast - slow) / slow.abs().max(1e-10);
    let regime = if score > 0.005 {
        Regime::Bull
    } else if score < -0.005 {
        Regime::Bear
    } else {
        Regime::Sideways
    };
    (regime, score.clamp(-1.0, 1.0))
}

// ── Volatility Regime ─────────────────────────────────────────────────────────

/// Classify volatility regime based on rolling standard deviation of returns.
pub fn volatility_regime(
    returns: &[f64],
    vol_window: usize,
    high_threshold: f64,
    low_threshold: f64,
) -> VolRegime {
    let vol = rolling_std(returns, vol_window) * (252.0_f64).sqrt(); // annualised
    if vol >= high_threshold {
        VolRegime::HighVol
    } else if vol <= low_threshold {
        VolRegime::LowVol
    } else {
        VolRegime::NormalVol
    }
}

// ── Macro Regime ──────────────────────────────────────────────────────────────

/// Classify macro regime using VIX, yield curve slope, and credit spread.
///
/// * `vix` — current VIX level.
/// * `yield_curve_slope` — 10yr - 2yr spread (bps).
/// * `credit_spread` — IG/HY OAS (bps).
pub fn macro_regime(vix: f64, yield_curve_slope: f64, credit_spread: f64) -> MacroRegime {
    let risk_off_signal = vix > 25.0 || credit_spread > 200.0;
    let yield_curve_inverted = yield_curve_slope < 0.0;
    let growth_weak = yield_curve_inverted && credit_spread > 150.0;

    if risk_off_signal {
        if growth_weak {
            MacroRegime::Stagflation
        } else {
            MacroRegime::RiskOff
        }
    } else {
        if yield_curve_slope > 100.0 && vix < 18.0 {
            MacroRegime::Goldilocks
        } else {
            MacroRegime::RiskOn
        }
    }
}

// ── Composite Regime ──────────────────────────────────────────────────────────

/// Composite regime combining trend, vol, and macro signals.
pub fn composite_regime(
    trend: Regime,
    vol: VolRegime,
    macro_r: MacroRegime,
) -> CompositeRegime {
    // Compute confidence based on signal agreement.
    let mut bullish_votes = 0.0_f64;
    let mut total_votes = 0.0_f64;

    // Trend signal.
    match trend {
        Regime::Bull => bullish_votes += 1.0,
        Regime::Bear => {}
        Regime::Sideways => bullish_votes += 0.5,
    }
    total_votes += 1.0;

    // Vol signal: low vol is bullish, high vol is bearish.
    match vol {
        VolRegime::LowVol => bullish_votes += 1.0,
        VolRegime::NormalVol => bullish_votes += 0.5,
        VolRegime::HighVol => {}
    }
    total_votes += 1.0;

    // Macro signal.
    match macro_r {
        MacroRegime::Goldilocks => bullish_votes += 1.0,
        MacroRegime::RiskOn => bullish_votes += 0.75,
        MacroRegime::RiskOff => {}
        MacroRegime::Stagflation => bullish_votes += 0.1,
    }
    total_votes += 1.0;

    let bull_score = bullish_votes / total_votes;
    let confidence = (bull_score - 0.5).abs() * 2.0;

    let label = format!(
        "Trend:{:?} Vol:{:?} Macro:{:?} (bull_score={:.2})",
        trend, vol, macro_r, bull_score
    );

    CompositeRegime {
        trend,
        vol,
        macro_regime: macro_r,
        confidence,
        label,
    }
}

// ── Advanced Classification ───────────────────────────────────────────────────

/// Breadth-based momentum score.
///
/// `pct_above_ma` — fraction of assets in universe trading above their N-day MA.
/// Returns a score in [-1, 1].
pub fn breadth_momentum_score(pct_above_ma: &[f64]) -> f64 {
    if pct_above_ma.is_empty() { return 0.0; }
    let avg = pct_above_ma.iter().sum::<f64>() / pct_above_ma.len() as f64;
    (avg - 0.5) * 2.0
}

/// Relative strength index regime.
/// RSI > 70 = overbought (potential reversal), RSI < 30 = oversold.
pub fn rsi_regime(rsi: f64) -> &'static str {
    if rsi > 70.0 { "Overbought" }
    else if rsi < 30.0 { "Oversold" }
    else if rsi > 50.0 { "Bullish_Momentum" }
    else { "Bearish_Momentum" }
}

/// Four-quadrant equity regime based on price momentum and earnings revision.
///
/// `price_momentum` — 12-1 month price return.
/// `earnings_revision` — net earnings revision (e.g., breadth).
#[derive(Debug, Clone, Copy)]
pub enum QuadrantRegime {
    /// Rising price + rising earnings: strong bull.
    BullAccelerating,
    /// Rising price + falling earnings: late cycle.
    BullDecelerating,
    /// Falling price + rising earnings: potential recovery.
    BearRecovering,
    /// Falling price + falling earnings: deep bear.
    BearAccelerating,
}

pub fn quadrant_regime(price_momentum: f64, earnings_revision: f64) -> QuadrantRegime {
    match (price_momentum > 0.0, earnings_revision > 0.0) {
        (true, true) => QuadrantRegime::BullAccelerating,
        (true, false) => QuadrantRegime::BullDecelerating,
        (false, true) => QuadrantRegime::BearRecovering,
        (false, false) => QuadrantRegime::BearAccelerating,
    }
}

/// Compute a rolling regime-change indicator (number of regime transitions
/// in the past `window` bars).
pub fn regime_instability(regime_series: &[Regime], window: usize) -> Vec<f64> {
    let n = regime_series.len();
    if n == 0 { return vec![]; }
    (0..n)
        .map(|i| {
            let start = i.saturating_sub(window);
            let slice = &regime_series[start..i.min(n)];
            if slice.len() < 2 { return 0.0; }
            let transitions = slice.windows(2).filter(|w| w[0] != w[1]).count();
            transitions as f64 / (slice.len() - 1) as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trend_bull_with_strong_uptrend() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
        let regime = trend_regime(&prices, 10, 30);
        assert_eq!(regime, Regime::Bull);
    }

    #[test]
    fn vol_regime_high_vol() {
        let returns: Vec<f64> = (0..252).map(|i| if i % 2 == 0 { 0.05 } else { -0.05 }).collect();
        let regime = volatility_regime(&returns, 20, 0.30, 0.10);
        assert_eq!(regime, VolRegime::HighVol);
    }

    #[test]
    fn composite_confidence_in_range() {
        let c = composite_regime(Regime::Bull, VolRegime::LowVol, MacroRegime::Goldilocks);
        assert!(c.confidence >= 0.0 && c.confidence <= 1.0);
    }
}
