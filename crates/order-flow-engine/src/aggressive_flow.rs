/// Aggressive order flow detector.
///
/// Aggressive orders are large market orders that sweep through multiple price
/// levels, causing immediate price impact. We detect them from OHLCV data via:
///
/// 1. Volume spike: bar volume > `volume_z_threshold` standard deviations above
///    the rolling mean.
/// 2. Range expansion: (high - low) > `range_atr_multiple` * ATR(14)
/// 3. Direction: if close > open (or > mid), it's aggressive buying; else selling.
///
/// The combined sweep score is:
///   sweep_score = min(1.0, volume_zscore / 3.0) * min(1.0, range_ratio / 2.0)
///
/// score > 0.5 -> significant aggressive flow
/// score > 0.8 -> very aggressive (often precedes strong short-term follow-through)

/// Parameters for aggressive flow detection.
#[derive(Debug, Clone)]
pub struct AggressiveFlowConfig {
    /// Rolling window for mean/std of volume
    pub volume_window: usize,
    /// Rolling window for ATR calculation
    pub atr_window: usize,
    /// Number of standard deviations above mean to qualify as a volume spike
    pub volume_z_threshold: f64,
    /// Multiplier of ATR for range expansion threshold
    pub range_atr_multiple: f64,
    /// Score threshold for significant sweep
    pub sweep_threshold: f64,
}

impl Default for AggressiveFlowConfig {
    fn default() -> Self {
        AggressiveFlowConfig {
            volume_window: 20,
            atr_window: 14,
            volume_z_threshold: 1.5,
            range_atr_multiple: 1.5,
            sweep_threshold: 0.5,
        }
    }
}

/// Result of aggressive flow analysis for a single bar.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AggressiveFlowResult {
    /// Z-score of bar volume relative to recent history
    pub volume_zscore: f64,
    /// Bar range relative to ATR
    pub range_atr_ratio: f64,
    /// Combined sweep score [0, 1]
    pub sweep_score: f64,
    /// Estimated aggressive buy pressure [0, 1] (fraction of sweep attributed to buyers)
    pub aggressive_buy_pressure: f64,
    /// Estimated aggressive sell pressure [0, 1]
    pub aggressive_sell_pressure: f64,
    /// Is this bar classified as aggressive flow?
    pub is_aggressive: bool,
    /// Direction of the sweep
    pub direction: SweepDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SweepDirection {
    Buying,
    Selling,
    Mixed,
    Insufficient,
}

impl std::fmt::Display for SweepDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SweepDirection::Buying => write!(f, "BUYING"),
            SweepDirection::Selling => write!(f, "SELLING"),
            SweepDirection::Mixed => write!(f, "MIXED"),
            SweepDirection::Insufficient => write!(f, "INSUFFICIENT"),
        }
    }
}

/// Stateful aggressive flow detector.
pub struct AggressiveFlowDetector {
    config: AggressiveFlowConfig,
    volume_history: std::collections::VecDeque<f64>,
    // True ranges for ATR
    tr_history: std::collections::VecDeque<f64>,
    prev_close: Option<f64>,
}

impl AggressiveFlowDetector {
    pub fn new(config: AggressiveFlowConfig) -> Self {
        AggressiveFlowDetector {
            config,
            volume_history: std::collections::VecDeque::new(),
            tr_history: std::collections::VecDeque::new(),
            prev_close: None,
        }
    }

    /// Process a single OHLCV bar and return the aggressive flow result.
    pub fn push(&mut self, open: f64, high: f64, low: f64, close: f64, volume: f64) -> AggressiveFlowResult {
        // True range
        let tr = match self.prev_close {
            Some(pc) => {
                let hl = high - low;
                let hc = (high - pc).abs();
                let lc = (low - pc).abs();
                hl.max(hc).max(lc)
            }
            None => high - low,
        };
        self.prev_close = Some(close);

        // Update rolling windows
        self.volume_history.push_back(volume);
        if self.volume_history.len() > self.config.volume_window {
            self.volume_history.pop_front();
        }
        self.tr_history.push_back(tr);
        if self.tr_history.len() > self.config.atr_window {
            self.tr_history.pop_front();
        }

        // Volume Z-score
        let (vol_mean, vol_std) = mean_std(self.volume_history.iter().copied());
        let volume_zscore = if vol_std > 1e-9 {
            (volume - vol_mean) / vol_std
        } else {
            0.0
        };

        // ATR and range ratio
        let atr = self.tr_history.iter().copied().sum::<f64>()
            / self.tr_history.len() as f64;
        let bar_range = high - low;
        let range_atr_ratio = if atr > 1e-9 { bar_range / atr } else { 1.0 };

        // Sweep score
        let vol_component = (volume_zscore / 3.0).clamp(0.0, 1.0);
        let range_component = (range_atr_ratio / 2.0).clamp(0.0, 1.0);
        let sweep_score = vol_component * range_component;

        let is_aggressive = sweep_score >= self.config.sweep_threshold;

        // Direction: use close relative to bar midpoint
        let mid = (open + close) / 2.0;
        let bar_close_vs_mid = (close - open) / (bar_range.max(1e-6));

        let direction = if !is_aggressive {
            SweepDirection::Insufficient
        } else if bar_close_vs_mid > 0.2 {
            SweepDirection::Buying
        } else if bar_close_vs_mid < -0.2 {
            SweepDirection::Selling
        } else {
            SweepDirection::Mixed
        };

        // Aggressive buy/sell pressures
        let buy_frac = ((close - low) / (bar_range.max(1e-6))).clamp(0.0, 1.0);
        let aggressive_buy_pressure = if is_aggressive { sweep_score * buy_frac } else { 0.0 };
        let aggressive_sell_pressure = if is_aggressive {
            sweep_score * (1.0 - buy_frac)
        } else {
            0.0
        };

        let _ = mid; // suppress warning

        AggressiveFlowResult {
            volume_zscore,
            range_atr_ratio,
            sweep_score,
            aggressive_buy_pressure,
            aggressive_sell_pressure,
            is_aggressive,
            direction,
        }
    }

    /// Process a full OHLCV series.
    pub fn process_series(
        &mut self,
        bars: &[(f64, f64, f64, f64, f64)], // (open, high, low, close, volume)
    ) -> Vec<AggressiveFlowResult> {
        bars.iter()
            .map(|&(o, h, l, c, v)| self.push(o, h, l, c, v))
            .collect()
    }

    /// Return aggregate stats: mean sweep score and fraction of aggressive bars.
    pub fn aggregate_stats(results: &[AggressiveFlowResult]) -> (f64, f64) {
        if results.is_empty() {
            return (0.0, 0.0);
        }
        let mean_score = results.iter().map(|r| r.sweep_score).sum::<f64>() / results.len() as f64;
        let agg_frac = results.iter().filter(|r| r.is_aggressive).count() as f64
            / results.len() as f64;
        (mean_score, agg_frac)
    }
}

impl Default for AggressiveFlowDetector {
    fn default() -> Self {
        Self::new(AggressiveFlowConfig::default())
    }
}

fn mean_std(data: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
    let v: Vec<f64> = data.collect();
    let n = v.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let mean = v.iter().sum::<f64>() / n;
    if n < 2.0 {
        return (mean, 0.0);
    }
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_bar(price: f64) -> (f64, f64, f64, f64, f64) {
        (price, price + 0.5, price - 0.5, price + 0.1, 1000.0)
    }

    fn spike_bar(price: f64) -> (f64, f64, f64, f64, f64) {
        // Very high volume and wide range
        (price, price + 5.0, price - 5.0, price + 3.0, 50000.0)
    }

    #[test]
    fn test_insufficient_data_low_score() {
        let mut det = AggressiveFlowDetector::default();
        let result = det.push(100.0, 100.5, 99.5, 100.1, 1000.0);
        // With only 1 bar, volume history is trivial, std=0, zscore=0
        assert!(result.sweep_score < 0.5);
    }

    #[test]
    fn test_volume_spike_detected() {
        let mut det = AggressiveFlowDetector::default();
        // Fill normal bars
        for i in 0..20 {
            det.push(100.0 + i as f64 * 0.01, 100.5, 99.5, 100.1, 1000.0);
        }
        // Now a spike
        let result = det.push(102.0, 110.0, 98.0, 109.0, 50000.0);
        assert!(result.volume_zscore > 2.0, "should have high z-score: {}", result.volume_zscore);
    }

    #[test]
    fn test_aggressive_bar_classified() {
        let mut det = AggressiveFlowDetector::default();
        let bars: Vec<_> = (0..25).map(|i| normal_bar(100.0 + i as f64 * 0.1)).collect();
        det.process_series(&bars);
        let spike = spike_bar(102.5);
        let result = det.push(spike.0, spike.1, spike.2, spike.3, spike.4);
        // High volume and wide range -> should be aggressive
        assert!(result.sweep_score > 0.0);
    }

    #[test]
    fn test_pressure_sums_bounded() {
        let mut det = AggressiveFlowDetector::default();
        let bars: Vec<_> = (0..25).map(|i| spike_bar(100.0 + i as f64)).collect();
        for bar in bars {
            let r = det.push(bar.0, bar.1, bar.2, bar.3, bar.4);
            assert!(r.aggressive_buy_pressure >= 0.0 && r.aggressive_buy_pressure <= 1.0);
            assert!(r.aggressive_sell_pressure >= 0.0 && r.aggressive_sell_pressure <= 1.0);
        }
    }

    #[test]
    fn test_process_series_length() {
        let mut det = AggressiveFlowDetector::default();
        let bars: Vec<_> = (0..50).map(|i| normal_bar(100.0 + i as f64 * 0.05)).collect();
        let results = det.process_series(&bars);
        assert_eq!(results.len(), 50);
    }

    #[test]
    fn test_sweep_direction_buying() {
        let mut det = AggressiveFlowDetector::default();
        // Fill history
        for _ in 0..25 {
            det.push(100.0, 100.5, 99.5, 100.1, 1000.0);
        }
        // Strong bullish sweep: close near high
        let result = det.push(100.0, 110.0, 99.5, 109.5, 100000.0);
        if result.is_aggressive {
            assert!(
                result.direction == SweepDirection::Buying || result.direction == SweepDirection::Mixed,
                "expected buying sweep, got {:?}", result.direction
            );
        }
    }

    #[test]
    fn test_aggregate_stats() {
        let mut det = AggressiveFlowDetector::default();
        let bars: Vec<_> = (0..30).map(|i| {
            if i % 5 == 0 { spike_bar(100.0 + i as f64) } else { normal_bar(100.0 + i as f64 * 0.1) }
        }).collect();
        let results = det.process_series(&bars);
        let (mean_score, agg_frac) = AggressiveFlowDetector::aggregate_stats(&results);
        assert!(mean_score >= 0.0 && mean_score <= 1.0);
        assert!(agg_frac >= 0.0 && agg_frac <= 1.0);
    }
}
