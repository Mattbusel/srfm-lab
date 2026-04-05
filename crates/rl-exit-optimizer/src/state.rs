/// Fixed-length state vector for the RL exit agent.
/// All features are normalized to [-1, 1].
///
/// Index layout:
///  0  position_pnl_pct          unrealized P&L as fraction of entry (clipped ±50%)
///  1  bars_held_norm             bars held, normalized (0 = just entered, 1 = 100 bars)
///  2  bh_mass_norm               BH mass normalized to [-1, 1] (raw range 0..1 -> clipped)
///  3  bh_active                  +1 if BH still active, -1 if dead
///  4  atr_ratio_norm             current_ATR / entry_ATR, normalized around 1
///  5  market_return              market return since entry (clipped ±10%)
///  6  momentum_15m               short-term momentum (clipped ±5%)
///  7  time_of_day_sin            sin of (UTC hour / 24 * 2π)
///  8  time_of_day_cos            cos of (UTC hour / 24 * 2π)
///  9  drawdown_from_peak         how far below peak P&L we are (negative, clipped -50%..0)
/// 10  pnl_acceleration           change in pnl_pct per bar (clipped ±5%)
/// 11  vol_regime                 atr_ratio > 1.5 -> high vol (+1), < 0.7 -> low vol (-1), else 0

pub const STATE_DIM: usize = 12;
pub const NUM_BINS: usize = 5;

/// Raw trade state before normalization.
#[derive(Debug, Clone, Default)]
pub struct TradeStateRaw {
    /// Unrealized P&L as a fraction of entry price (e.g. 0.03 = 3% up)
    pub position_pnl_pct: f64,
    /// Number of bars we have been in this position
    pub bars_held: u32,
    /// Current BH mass value (0..1 scale, higher = stronger BH signal)
    pub bh_mass: f64,
    /// Whether the BH signal is still considered active
    pub bh_active: bool,
    /// Ratio of current ATR to ATR at entry
    pub atr_ratio: f64,
    /// Return of market index since trade entry (fraction)
    pub market_return_since_entry: f64,
    /// 15-minute price momentum (fraction)
    pub momentum_15m: f64,
    /// UTC hour of day (0..23)
    pub utc_hour: f64,
    /// How far below the position's peak P&L we currently sit (negative fraction)
    pub drawdown_from_peak: f64,
    /// Change in pnl_pct from previous bar (acceleration term)
    pub pnl_acceleration: f64,
}

/// Normalized, fixed-length observation vector fed to the Q-network.
#[derive(Debug, Clone, PartialEq)]
pub struct StateVector(pub [f64; STATE_DIM]);

impl StateVector {
    /// Build a `StateVector` from raw inputs, normalizing each feature to [-1, 1].
    pub fn from_raw(raw: &TradeStateRaw) -> Self {
        // pnl_pct: clip to ±50%, then scale (÷0.5 -> [-1,1])
        let pnl = clamp(raw.position_pnl_pct / 0.50, -1.0, 1.0);

        // bars_held: 0 bars = -1, 100+ bars = +1 (linear in [0..100])
        let bars = clamp(raw.bars_held as f64 / 50.0 - 1.0, -1.0, 1.0);

        // bh_mass: raw in [0..1], map to [-1, 1]
        let bh_mass = clamp(raw.bh_mass * 2.0 - 1.0, -1.0, 1.0);

        // bh_active: binary
        let bh_active = if raw.bh_active { 1.0 } else { -1.0 };

        // atr_ratio: 1.0 = no change. Normalize: (ratio - 1) / 1 clipped ±1
        let atr_ratio = clamp(raw.atr_ratio - 1.0, -1.0, 1.0);

        // market_return: clip ±10%
        let mkt = clamp(raw.market_return_since_entry / 0.10, -1.0, 1.0);

        // momentum_15m: clip ±5%
        let mom = clamp(raw.momentum_15m / 0.05, -1.0, 1.0);

        // time of day: cyclical encoding
        let angle = raw.utc_hour / 24.0 * 2.0 * std::f64::consts::PI;
        let tod_sin = angle.sin();
        let tod_cos = angle.cos();

        // drawdown_from_peak: clip to [-50%..0], normalize to [-1..0]
        let dd = clamp(raw.drawdown_from_peak / 0.50, -1.0, 0.0);

        // pnl_acceleration: clip ±5%
        let accel = clamp(raw.pnl_acceleration / 0.05, -1.0, 1.0);

        // vol regime: derived from atr_ratio
        let vol_regime = if raw.atr_ratio > 1.5 {
            1.0
        } else if raw.atr_ratio < 0.7 {
            -1.0
        } else {
            0.0
        };

        StateVector([
            pnl, bars, bh_mass, bh_active, atr_ratio, mkt, mom, tod_sin, tod_cos, dd, accel,
            vol_regime,
        ])
    }

    /// Return the raw slice of feature values.
    #[inline]
    pub fn features(&self) -> &[f64; STATE_DIM] {
        &self.0
    }

    /// Discretize each feature into `NUM_BINS` bins (0..NUM_BINS-1).
    /// Feature range is [-1, 1]; bins are evenly spaced.
    pub fn discretize(&self) -> [usize; STATE_DIM] {
        let mut bins = [0usize; STATE_DIM];
        for (i, &v) in self.0.iter().enumerate() {
            // Map [-1, 1] to [0, NUM_BINS)
            let scaled = (v + 1.0) / 2.0; // [0, 1]
            let bin = (scaled * NUM_BINS as f64).floor() as isize;
            bins[i] = bin.clamp(0, NUM_BINS as isize - 1) as usize;
        }
        bins
    }
}

impl Default for StateVector {
    fn default() -> Self {
        StateVector([0.0; STATE_DIM])
    }
}

#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_default() -> TradeStateRaw {
        TradeStateRaw {
            position_pnl_pct: 0.0,
            bars_held: 50,
            bh_mass: 0.5,
            bh_active: true,
            atr_ratio: 1.0,
            market_return_since_entry: 0.0,
            momentum_15m: 0.0,
            utc_hour: 12.0,
            drawdown_from_peak: 0.0,
            pnl_acceleration: 0.0,
        }
    }

    #[test]
    fn test_normalization_bounds() {
        let raw = TradeStateRaw {
            position_pnl_pct: 10.0, // way out of range
            bars_held: 9999,
            bh_mass: 2.0,
            bh_active: false,
            atr_ratio: 50.0,
            market_return_since_entry: 5.0,
            momentum_15m: 3.0,
            utc_hour: 23.0,
            drawdown_from_peak: -100.0,
            pnl_acceleration: 10.0,
        };
        let sv = StateVector::from_raw(&raw);
        for &v in sv.features() {
            assert!(v >= -1.0 && v <= 1.0, "feature out of range: {}", v);
        }
    }

    #[test]
    fn test_mid_values() {
        let raw = raw_default();
        let sv = StateVector::from_raw(&raw);
        assert_eq!(sv.0[0], 0.0, "pnl=0 should map to 0");
        assert_eq!(sv.0[3], 1.0, "bh_active=true should map to 1");
        assert_eq!(sv.0[4], 0.0, "atr_ratio=1 should map to 0");
    }

    #[test]
    fn test_discretize_range() {
        let raw = raw_default();
        let sv = StateVector::from_raw(&raw);
        let bins = sv.discretize();
        for &b in &bins {
            assert!(b < NUM_BINS, "bin out of range: {}", b);
        }
    }

    #[test]
    fn test_discretize_extremes() {
        let low = StateVector([-1.0; STATE_DIM]);
        let high = StateVector([1.0; STATE_DIM]);
        let low_bins = low.discretize();
        let high_bins = high.discretize();
        for &b in &low_bins {
            assert_eq!(b, 0);
        }
        for &b in &high_bins {
            assert_eq!(b, NUM_BINS - 1);
        }
    }
}
