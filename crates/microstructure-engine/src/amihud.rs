/// Amihud Illiquidity Ratio — rolling estimator.
///
/// ILLIQ_t = (1/N) Σ |r_i| / DollarVolume_i
///
/// Higher values = more price impact per dollar traded = less liquid.
/// Uses a rolling window so stale observations are dropped.

use crate::streaming_stats::RollingWindow;

/// Single-bar observation for Amihud computation.
#[derive(Debug, Clone, Copy)]
pub struct BarObs {
    /// Absolute return: |close - open| / open  (or |Δlog p|)
    pub abs_return:    f64,
    /// Dollar volume traded in this bar (price * volume)
    pub dollar_volume: f64,
}

/// Rolling Amihud illiquidity estimator.
#[derive(Debug, Clone)]
pub struct AmihudIlliquidity {
    window:    RollingWindow<BarObs>,
    sum_ratio: f64,   // running sum of |r|/dv for fast O(1) mean update
}

impl AmihudIlliquidity {
    pub fn new(window: usize) -> Self {
        Self {
            window:    RollingWindow::new(window),
            sum_ratio: 0.0,
        }
    }

    /// Push a new bar observation.
    pub fn update(&mut self, abs_return: f64, dollar_volume: f64) {
        let ratio = if dollar_volume > 0.0 { abs_return / dollar_volume } else { 0.0 };
        let obs   = BarObs { abs_return, dollar_volume };

        if let Some(evicted) = self.window.push(obs) {
            let ev_ratio = if evicted.dollar_volume > 0.0 {
                evicted.abs_return / evicted.dollar_volume
            } else { 0.0 };
            self.sum_ratio -= ev_ratio;
        }
        self.sum_ratio += ratio;
    }

    /// Current Amihud illiquidity estimate. `None` if window is empty.
    pub fn illiquidity(&self) -> Option<f64> {
        let n = self.window.len();
        if n == 0 { return None; }
        Some(self.sum_ratio / n as f64)
    }

    /// Annualised illiquidity (multiply by trading days per year).
    pub fn annualised(&self, trading_days: f64) -> Option<f64> {
        self.illiquidity().map(|il| il * trading_days)
    }

    pub fn count(&self) -> usize { self.window.len() }
    pub fn is_full(&self) -> bool { self.window.is_full() }

    /// Convenience: accept (open, close, volume) directly.
    pub fn update_ohlcv(&mut self, open: f64, close: f64, volume: f64, price: f64) {
        if open <= 0.0 { return; }
        let abs_ret = ((close - open) / open).abs();
        let dv      = price * volume;
        self.update(abs_ret, dv);
    }
}

/// Normalised illiquidity — z-score of current ILLIQ against its own history.
#[derive(Debug, Clone)]
pub struct NormalisedAmihud {
    inner:        AmihudIlliquidity,
    history:      RollingWindow<f64>,  // rolling ILLIQ values
}

impl NormalisedAmihud {
    pub fn new(inner_window: usize, history_window: usize) -> Self {
        Self {
            inner:   AmihudIlliquidity::new(inner_window),
            history: RollingWindow::new(history_window),
        }
    }

    pub fn update(&mut self, abs_return: f64, dollar_volume: f64) {
        self.inner.update(abs_return, dollar_volume);
        if let Some(il) = self.inner.illiquidity() {
            self.history.push(il);
        }
    }

    /// Z-score of current ILLIQ vs historical distribution.
    pub fn zscore(&self) -> Option<f64> {
        let current = self.inner.illiquidity()?;
        let mean    = self.history.mean()?;
        let std     = self.history.std()?;
        if std < 1e-30 { return Some(0.0); }
        Some((current - mean) / std)
    }

    pub fn illiquidity(&self) -> Option<f64> { self.inner.illiquidity() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amihud_basic() {
        let mut a = AmihudIlliquidity::new(10);
        // |return| = 0.01, dollar_vol = 1_000_000 → ratio = 1e-8
        for _ in 0..5 {
            a.update(0.01, 1_000_000.0);
        }
        let il = a.illiquidity().unwrap();
        assert!((il - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn amihud_eviction() {
        let mut a = AmihudIlliquidity::new(3);
        a.update(0.01, 1_000_000.0);  // ratio = 1e-8
        a.update(0.02, 2_000_000.0);  // ratio = 1e-8
        a.update(0.01, 1_000_000.0);  // ratio = 1e-8
        // Push one more — first obs evicted but same ratio so result stable
        a.update(0.03, 3_000_000.0);  // ratio = 1e-8
        let il = a.illiquidity().unwrap();
        assert!((il - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn amihud_zero_volume() {
        let mut a = AmihudIlliquidity::new(5);
        a.update(0.05, 0.0);  // zero dv — should not panic, contributes 0
        a.update(0.01, 1e6);
        assert!(a.illiquidity().is_some());
    }

    #[test]
    fn normalised_amihud_zscore() {
        let mut na = NormalisedAmihud::new(5, 50);
        for i in 1..=100 {
            let dv = 1_000_000.0 + i as f64 * 10_000.0;
            na.update(0.01, dv);
        }
        // zscore should be computable after enough history
        assert!(na.zscore().is_some());
    }
}
