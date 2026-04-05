/// Delta divergence detector.
///
/// Detects when price action and order flow delta diverge:
///
/// Bearish divergence (weakening bullish pressure):
///   Price makes a new high, but delta (buy_vol - sell_vol) makes a *lower* high.
///   Signals that buyers are losing conviction at higher prices.
///
/// Bullish divergence (weakening bearish pressure):
///   Price makes a new low, but delta makes a *higher* low (less negative).
///   Signals that sellers are exhausting at lower prices.
///
/// The detector operates on a rolling window of (price_high, price_low, delta)
/// observations and emits a `DivergenceSignal` at each step.

/// A divergence signal emitted by the detector.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DivergenceType {
    /// Price new high + delta lower high: buyer conviction weakening
    BearishDivergence,
    /// Price new low + delta higher low: seller exhaustion
    BullishDivergence,
    /// No divergence detected
    None,
}

impl std::fmt::Display for DivergenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DivergenceType::BearishDivergence => write!(f, "BEARISH_DIVERGENCE"),
            DivergenceType::BullishDivergence => write!(f, "BULLISH_DIVERGENCE"),
            DivergenceType::None => write!(f, "NONE"),
        }
    }
}

/// Full divergence signal with context.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DivergenceSignal {
    pub divergence_type: DivergenceType,
    /// Current bar's high
    pub current_high: f64,
    /// Current bar's low
    pub current_low: f64,
    /// Current bar's delta
    pub current_delta: f64,
    /// Reference bar's high (from which divergence is measured)
    pub reference_high: f64,
    /// Reference bar's low
    pub reference_low: f64,
    /// Reference bar's delta
    pub reference_delta: f64,
    /// Strength: magnitude of price move vs delta move (higher = stronger signal)
    pub strength: f64,
}

impl DivergenceSignal {
    pub fn none(high: f64, low: f64, delta: f64) -> Self {
        DivergenceSignal {
            divergence_type: DivergenceType::None,
            current_high: high,
            current_low: low,
            current_delta: delta,
            reference_high: high,
            reference_low: low,
            reference_delta: delta,
            strength: 0.0,
        }
    }
}

/// Observation for one bar fed into the divergence detector.
#[derive(Debug, Clone, Copy)]
pub struct DivergenceObs {
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub delta: f64, // buy_volume - sell_volume for this bar
}

/// Rolling divergence detector.
pub struct DeltaDivergenceDetector {
    window: std::collections::VecDeque<DivergenceObs>,
    window_size: usize,
    /// Minimum price move (fraction) to qualify as a new high/low
    min_price_move: f64,
    /// Minimum delta change to qualify as meaningful divergence
    min_delta_change_pct: f64,
}

impl DeltaDivergenceDetector {
    /// `window_size`: number of bars to look back for prior swing high/low.
    /// `min_price_move`: minimum fractional price move to consider as new extreme.
    /// `min_delta_change_pct`: minimum relative delta change to signal divergence.
    pub fn new(window_size: usize, min_price_move: f64, min_delta_change_pct: f64) -> Self {
        DeltaDivergenceDetector {
            window: std::collections::VecDeque::new(),
            window_size,
            min_price_move,
            min_delta_change_pct,
        }
    }

    /// Feed a new observation and return the divergence signal.
    pub fn push(&mut self, obs: DivergenceObs) -> DivergenceSignal {
        // First, compute signal from current window
        let signal = self.detect(&obs);
        // Then update window
        self.window.push_back(obs);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
        signal
    }

    fn detect(&self, current: &DivergenceObs) -> DivergenceSignal {
        if self.window.len() < 2 {
            return DivergenceSignal::none(current.high, current.low, current.delta);
        }

        // Find the most recent prior high and low in the window
        let prior_high_obs = self
            .window
            .iter()
            .max_by(|a, b| a.high.partial_cmp(&b.high).unwrap_or(std::cmp::Ordering::Equal));
        let prior_low_obs = self
            .window
            .iter()
            .min_by(|a, b| a.low.partial_cmp(&b.low).unwrap_or(std::cmp::Ordering::Equal));

        let (prior_high, prior_low) = match (prior_high_obs, prior_low_obs) {
            (Some(h), Some(l)) => (h, l),
            _ => return DivergenceSignal::none(current.high, current.low, current.delta),
        };

        // Check bearish divergence: price new high + lower delta high
        let price_higher = current.high > prior_high.high * (1.0 + self.min_price_move);
        let delta_lower_high = prior_high.delta > 0.0
            && current.delta < prior_high.delta * (1.0 - self.min_delta_change_pct);

        if price_higher && delta_lower_high {
            let price_move = (current.high - prior_high.high) / prior_high.high;
            let delta_move = if prior_high.delta.abs() > 1e-9 {
                (prior_high.delta - current.delta) / prior_high.delta.abs()
            } else {
                0.0
            };
            let strength = (price_move + delta_move) / 2.0;

            return DivergenceSignal {
                divergence_type: DivergenceType::BearishDivergence,
                current_high: current.high,
                current_low: current.low,
                current_delta: current.delta,
                reference_high: prior_high.high,
                reference_low: prior_high.low,
                reference_delta: prior_high.delta,
                strength,
            };
        }

        // Check bullish divergence: price new low + higher delta low (less negative)
        let price_lower = current.low < prior_low.low * (1.0 - self.min_price_move);
        let delta_higher_low = prior_low.delta < 0.0
            && current.delta > prior_low.delta * (1.0 - self.min_delta_change_pct);
        // delta_higher_low: current delta is less negative than prior (closer to 0)

        if price_lower && delta_higher_low {
            let price_move = (prior_low.low - current.low) / prior_low.low;
            let delta_move = if prior_low.delta.abs() > 1e-9 {
                (current.delta - prior_low.delta) / prior_low.delta.abs()
            } else {
                0.0
            };
            let strength = (price_move + delta_move) / 2.0;

            return DivergenceSignal {
                divergence_type: DivergenceType::BullishDivergence,
                current_high: current.high,
                current_low: current.low,
                current_delta: current.delta,
                reference_high: prior_low.high,
                reference_low: prior_low.low,
                reference_delta: prior_low.delta,
                strength,
            };
        }

        DivergenceSignal::none(current.high, current.low, current.delta)
    }

    /// Process a full series of observations.
    pub fn process_series(&mut self, obs: &[DivergenceObs]) -> Vec<DivergenceSignal> {
        obs.iter().map(|o| self.push(*o)).collect()
    }

    /// Count divergences in the last N signals.
    pub fn count_divergences(signals: &[DivergenceSignal]) -> (usize, usize) {
        let bearish = signals
            .iter()
            .filter(|s| s.divergence_type == DivergenceType::BearishDivergence)
            .count();
        let bullish = signals
            .iter()
            .filter(|s| s.divergence_type == DivergenceType::BullishDivergence)
            .count();
        (bearish, bullish)
    }
}

impl Default for DeltaDivergenceDetector {
    fn default() -> Self {
        Self::new(20, 0.001, 0.10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(high: f64, low: f64, close: f64, delta: f64) -> DivergenceObs {
        DivergenceObs { high, low, close, delta }
    }

    #[test]
    fn test_no_divergence_flat_market() {
        let mut det = DeltaDivergenceDetector::default();
        for _ in 0..20 {
            let sig = det.push(obs(100.0, 99.0, 99.5, 100.0));
            assert_eq!(sig.divergence_type, DivergenceType::None);
        }
    }

    #[test]
    fn test_bearish_divergence_detected() {
        let mut det = DeltaDivergenceDetector::new(10, 0.001, 0.1);
        // Establish a prior high with strong delta
        for _ in 0..5 {
            det.push(obs(100.0, 99.0, 99.5, 500.0));
        }
        // Now make a new price high with weak delta
        let sig = det.push(obs(101.5, 100.5, 101.0, 200.0)); // price > 100*1.001, delta < 500*0.9
        assert_eq!(
            sig.divergence_type,
            DivergenceType::BearishDivergence,
            "should detect bearish divergence"
        );
    }

    #[test]
    fn test_bullish_divergence_detected() {
        let mut det = DeltaDivergenceDetector::new(10, 0.001, 0.1);
        // Establish a prior low with heavy selling
        for _ in 0..5 {
            det.push(obs(100.0, 98.0, 98.5, -500.0));
        }
        // New price low with less selling pressure
        let sig = det.push(obs(98.0, 97.8, 98.0, -100.0));
        // price lower and delta higher (less negative) -> bullish
        assert_eq!(
            sig.divergence_type,
            DivergenceType::BullishDivergence,
            "should detect bullish divergence"
        );
    }

    #[test]
    fn test_insufficient_window_returns_none() {
        let mut det = DeltaDivergenceDetector::default();
        let sig = det.push(obs(100.0, 99.0, 99.5, 100.0));
        assert_eq!(sig.divergence_type, DivergenceType::None);
    }

    #[test]
    fn test_process_series_length() {
        let mut det = DeltaDivergenceDetector::default();
        let observations: Vec<DivergenceObs> = (0..30)
            .map(|i| obs(100.0 + i as f64 * 0.1, 99.0 + i as f64 * 0.1, 99.5, 100.0))
            .collect();
        let signals = det.process_series(&observations);
        assert_eq!(signals.len(), 30);
    }

    #[test]
    fn test_strength_positive_on_divergence() {
        let mut det = DeltaDivergenceDetector::new(10, 0.001, 0.1);
        for _ in 0..5 {
            det.push(obs(100.0, 99.0, 99.5, 500.0));
        }
        let sig = det.push(obs(102.0, 101.0, 101.5, 100.0));
        if sig.divergence_type == DivergenceType::BearishDivergence {
            assert!(sig.strength > 0.0);
        }
    }
}
