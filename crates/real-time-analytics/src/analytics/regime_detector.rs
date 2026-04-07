// analytics/regime_detector.rs -- Real-time regime detection.
//
// Implements BHMassRegime, HurstRegime, VolRegime, CompositeRegime.
// All implement the RegimeDetector trait.

use std::collections::VecDeque;
use crate::tick_aggregator::Bar;

// ─── RegimeDetector trait ─────────────────────────────────────────────────────

/// A regime change event emitted when the detected regime transitions.
#[derive(Debug, Clone, PartialEq)]
pub struct RegimeChange {
    pub old_regime: String,
    pub new_regime: String,
    /// Confidence in [0, 1].
    pub confidence: f64,
}

/// Trait for all regime detectors.
pub trait RegimeDetector {
    /// The regime label type returned by this detector.
    type Regime: Clone + PartialEq + std::fmt::Debug + std::fmt::Display;

    /// Feed a new bar. Returns Some(RegimeChange) if the regime shifted.
    fn update(&mut self, bar: &Bar) -> Option<RegimeChange>;

    /// Return the current regime label.
    fn current_regime(&self) -> Self::Regime;

    /// Return confidence in the current regime estimate in [0, 1].
    fn confidence(&self) -> f64;
}

// ─── BHMassRegime ─────────────────────────────────────────────────────────────

/// BH (Buy-side/Hold) mass accumulation regimes.
///
/// Uses a momentum-based proxy for "mass" (sustained directional energy).
/// mass = sum of (close - open) / atr for the last N bars (normalized directional moves).
///
/// Regimes:
///   HIGH_MASS : mass > 1.92  -- strong accumulation
///   FORMING   : 1.0 <= mass <= 1.92  -- building
///   LOW       : mass < 1.0  -- no regime / distribution
#[derive(Debug, Clone, PartialEq)]
pub enum BHMassLevel {
    HighMass,
    Forming,
    Low,
}

impl std::fmt::Display for BHMassLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BHMassLevel::HighMass => write!(f, "HIGH_MASS"),
            BHMassLevel::Forming => write!(f, "FORMING"),
            BHMassLevel::Low => write!(f, "LOW"),
        }
    }
}

#[derive(Debug)]
pub struct BHMassRegime {
    window: usize,
    atr_period: usize,
    buf: VecDeque<f64>,
    atr_buf: VecDeque<f64>,
    current_atr: f64,
    prev_close: Option<f64>,
    regime: BHMassLevel,
    confidence: f64,
}

impl BHMassRegime {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            atr_period: 14,
            buf: VecDeque::with_capacity(window + 1),
            atr_buf: VecDeque::with_capacity(15),
            current_atr: 0.0,
            prev_close: None,
            regime: BHMassLevel::Low,
            confidence: 0.0,
        }
    }

    fn compute_atr(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let tr = match self.prev_close {
            Some(pc) => {
                let hl = high - low;
                let hpc = (high - pc).abs();
                let lpc = (low - pc).abs();
                hl.max(hpc).max(lpc)
            }
            None => high - low,
        };
        self.prev_close = Some(close);

        if self.atr_buf.len() < self.atr_period {
            self.atr_buf.push_back(tr);
            if self.atr_buf.len() == self.atr_period {
                self.current_atr = self.atr_buf.iter().sum::<f64>() / self.atr_period as f64;
            }
            return self.current_atr;
        }
        self.current_atr =
            (self.current_atr * (self.atr_period as f64 - 1.0) + tr) / self.atr_period as f64;
        self.current_atr
    }
}

impl RegimeDetector for BHMassRegime {
    type Regime = BHMassLevel;

    fn update(&mut self, bar: &Bar) -> Option<RegimeChange> {
        let atr = self.compute_atr(bar.high, bar.low, bar.close);
        if atr < 1e-12 {
            return None;
        }

        let directional = (bar.close - bar.open) / atr;
        if self.buf.len() == self.window {
            self.buf.pop_front();
        }
        self.buf.push_back(directional);

        if self.buf.len() < self.window {
            return None;
        }

        // Mass = mean of absolute directional moves, scaled by sign alignment
        let pos: f64 = self.buf.iter().filter(|&&v| v > 0.0).sum();
        let neg: f64 = self.buf.iter().filter(|&&v| v < 0.0).map(|v| v.abs()).sum();
        let mass = (pos - neg).abs() / self.window as f64;

        let new_regime = if mass > 1.92 {
            BHMassLevel::HighMass
        } else if mass >= 1.0 {
            BHMassLevel::Forming
        } else {
            BHMassLevel::Low
        };

        // Confidence based on distance from thresholds
        self.confidence = if mass > 1.92 {
            ((mass - 1.92) / 0.5).min(1.0)
        } else if mass >= 1.0 {
            (mass - 1.0) / 0.92
        } else {
            (1.0 - mass).min(1.0)
        };

        if new_regime != self.regime {
            let change = RegimeChange {
                old_regime: self.regime.to_string(),
                new_regime: new_regime.to_string(),
                confidence: self.confidence,
            };
            self.regime = new_regime;
            Some(change)
        } else {
            self.regime = new_regime;
            None
        }
    }

    fn current_regime(&self) -> BHMassLevel { self.regime.clone() }
    fn confidence(&self) -> f64 { self.confidence }
}

// ─── HurstRegime ─────────────────────────────────────────────────────────────

/// Hurst exponent-based regime detection via rolling R/S analysis.
///
/// Regimes:
///   TRENDING      : H > 0.58
///   NEUTRAL       : 0.42 <= H <= 0.58
///   MEAN_REVERTING: H < 0.42
#[derive(Debug, Clone, PartialEq)]
pub enum HurstLevel {
    Trending,
    Neutral,
    MeanReverting,
}

impl std::fmt::Display for HurstLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HurstLevel::Trending => write!(f, "TRENDING"),
            HurstLevel::Neutral => write!(f, "NEUTRAL"),
            HurstLevel::MeanReverting => write!(f, "MEAN_REVERTING"),
        }
    }
}

#[derive(Debug)]
pub struct HurstRegime {
    window: usize,
    price_buf: VecDeque<f64>,
    regime: HurstLevel,
    confidence: f64,
    last_hurst: f64,
}

impl HurstRegime {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            price_buf: VecDeque::with_capacity(window + 1),
            regime: HurstLevel::Neutral,
            confidence: 0.0,
            last_hurst: 0.5,
        }
    }

    /// Compute R/S Hurst exponent from a slice of prices.
    ///
    /// Uses two sub-period lengths and OLS log-log regression.
    fn compute_hurst(prices: &[f64]) -> Option<f64> {
        let n = prices.len();
        if n < 8 {
            return None;
        }

        // Compute log returns
        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| if w[0] > 1e-12 { (w[1] / w[0]).ln() } else { 0.0 })
            .collect();

        if returns.len() < 4 {
            return None;
        }

        // Compute R/S for different sub-period lengths
        let mut log_ns = Vec::new();
        let mut log_rs = Vec::new();

        let periods = [returns.len() / 4, returns.len() / 2, returns.len()];
        for &period in &periods {
            if period < 4 {
                continue;
            }
            let num_chunks = returns.len() / period;
            if num_chunks == 0 {
                continue;
            }

            let mut rs_vals = Vec::new();
            for c in 0..num_chunks {
                let chunk = &returns[c * period..(c + 1) * period];
                let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
                let deviations: Vec<f64> = chunk.iter().map(|&r| r - mean).collect();
                let cum: Vec<f64> = deviations
                    .iter()
                    .scan(0.0_f64, |acc, &d| { *acc += d; Some(*acc) })
                    .collect();
                let r = cum.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    - cum.iter().cloned().fold(f64::INFINITY, f64::min);
                let std = {
                    let var = chunk.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
                        / chunk.len() as f64;
                    var.sqrt()
                };
                if std > 1e-12 {
                    rs_vals.push(r / std);
                }
            }

            if !rs_vals.is_empty() {
                let avg_rs = rs_vals.iter().sum::<f64>() / rs_vals.len() as f64;
                if avg_rs > 1e-12 {
                    log_ns.push((period as f64).ln());
                    log_rs.push(avg_rs.ln());
                }
            }
        }

        if log_ns.len() < 2 {
            return None;
        }

        // OLS slope = Hurst exponent
        let n_pts = log_ns.len() as f64;
        let mean_x = log_ns.iter().sum::<f64>() / n_pts;
        let mean_y = log_rs.iter().sum::<f64>() / n_pts;
        let num: f64 = log_ns.iter().zip(log_rs.iter()).map(|(&x, &y)| (x - mean_x) * (y - mean_y)).sum();
        let den: f64 = log_ns.iter().map(|&x| (x - mean_x).powi(2)).sum();

        if den.abs() < 1e-15 {
            return None;
        }
        Some((num / den).max(0.0).min(1.0))
    }

    pub fn last_hurst(&self) -> f64 { self.last_hurst }
}

impl RegimeDetector for HurstRegime {
    type Regime = HurstLevel;

    fn update(&mut self, bar: &Bar) -> Option<RegimeChange> {
        if self.price_buf.len() == self.window {
            self.price_buf.pop_front();
        }
        self.price_buf.push_back(bar.close);

        if self.price_buf.len() < self.window {
            return None;
        }

        let prices: Vec<f64> = self.price_buf.iter().cloned().collect();
        let hurst = match Self::compute_hurst(&prices) {
            Some(h) => h,
            None => return None,
        };
        self.last_hurst = hurst;

        let new_regime = if hurst > 0.58 {
            HurstLevel::Trending
        } else if hurst < 0.42 {
            HurstLevel::MeanReverting
        } else {
            HurstLevel::Neutral
        };

        self.confidence = if hurst > 0.58 {
            ((hurst - 0.58) / 0.42).min(1.0)
        } else if hurst < 0.42 {
            ((0.42 - hurst) / 0.42).min(1.0)
        } else {
            1.0 - (hurst - 0.5).abs() / 0.08
        };

        if new_regime != self.regime {
            let change = RegimeChange {
                old_regime: self.regime.to_string(),
                new_regime: new_regime.to_string(),
                confidence: self.confidence,
            };
            self.regime = new_regime;
            Some(change)
        } else {
            self.regime = new_regime;
            None
        }
    }

    fn current_regime(&self) -> HurstLevel { self.regime.clone() }
    fn confidence(&self) -> f64 { self.confidence }
}

// ─── VolRegime ────────────────────────────────────────────────────────────────

/// GARCH(1,1)-based volatility regime.
///
/// Compares current conditional variance to a rolling 252-bar percentile.
///
/// Regimes:
///   LOW_VOL    : current variance < 25th percentile
///   NORMAL     : 25th-75th percentile
///   HIGH_VOL   : > 75th percentile
#[derive(Debug, Clone, PartialEq)]
pub enum VolLevel {
    LowVol,
    Normal,
    HighVol,
}

impl std::fmt::Display for VolLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VolLevel::LowVol => write!(f, "LOW_VOL"),
            VolLevel::Normal => write!(f, "NORMAL"),
            VolLevel::HighVol => write!(f, "HIGH_VOL"),
        }
    }
}

#[derive(Debug)]
pub struct VolRegime {
    /// GARCH(1,1) parameters: omega, alpha, beta.
    omega: f64,
    alpha: f64,
    beta: f64,
    /// Current conditional variance.
    cond_var: f64,
    /// Rolling history of conditional variances for percentile computation.
    var_history: VecDeque<f64>,
    history_window: usize,
    prev_return: f64,
    prev_close: Option<f64>,
    regime: VolLevel,
    confidence: f64,
}

impl VolRegime {
    /// Create a new VolRegime detector.
    ///
    /// Uses typical GARCH(1,1) parameters: omega=1e-6, alpha=0.1, beta=0.85.
    pub fn new(history_window: usize) -> Self {
        Self {
            omega: 1e-6,
            alpha: 0.1,
            beta: 0.85,
            cond_var: 1e-4,
            var_history: VecDeque::with_capacity(history_window + 1),
            history_window,
            prev_return: 0.0,
            prev_close: None,
            regime: VolLevel::Normal,
            confidence: 0.0,
        }
    }

    pub fn with_garch_params(mut self, omega: f64, alpha: f64, beta: f64) -> Self {
        self.omega = omega;
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    pub fn conditional_variance(&self) -> f64 { self.cond_var }
}

impl RegimeDetector for VolRegime {
    type Regime = VolLevel;

    fn update(&mut self, bar: &Bar) -> Option<RegimeChange> {
        if let Some(pc) = self.prev_close {
            if pc > 1e-12 {
                let r = (bar.close / pc).ln();
                // GARCH(1,1): h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
                self.cond_var = self.omega
                    + self.alpha * self.prev_return.powi(2)
                    + self.beta * self.cond_var;
                self.cond_var = self.cond_var.max(1e-10);
                self.prev_return = r;

                if self.var_history.len() == self.history_window {
                    self.var_history.pop_front();
                }
                self.var_history.push_back(self.cond_var);

                if self.var_history.len() >= 20 {
                    let mut sorted: Vec<f64> = self.var_history.iter().cloned().collect();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let n = sorted.len();
                    let p25 = sorted[n / 4];
                    let p75 = sorted[3 * n / 4];

                    let new_regime = if self.cond_var < p25 {
                        VolLevel::LowVol
                    } else if self.cond_var > p75 {
                        VolLevel::HighVol
                    } else {
                        VolLevel::Normal
                    };

                    let range = (p75 - p25).max(1e-15);
                    self.confidence = if self.cond_var < p25 {
                        ((p25 - self.cond_var) / p25).min(1.0)
                    } else if self.cond_var > p75 {
                        ((self.cond_var - p75) / range).min(1.0)
                    } else {
                        let mid = (p25 + p75) / 2.0;
                        1.0 - ((self.cond_var - mid).abs() / (range / 2.0)).min(1.0)
                    };

                    if new_regime != self.regime {
                        let change = RegimeChange {
                            old_regime: self.regime.to_string(),
                            new_regime: new_regime.to_string(),
                            confidence: self.confidence,
                        };
                        self.regime = new_regime;
                        self.prev_close = Some(bar.close);
                        return Some(change);
                    }
                    self.regime = new_regime;
                }
            }
        }
        self.prev_close = Some(bar.close);
        None
    }

    fn current_regime(&self) -> VolLevel { self.regime.clone() }
    fn confidence(&self) -> f64 { self.confidence }
}

// ─── CompositeRegime ─────────────────────────────────────────────────────────

/// Composite regime combining BHMass + Hurst + Vol via majority vote.
///
/// Each sub-regime votes for: RISK_ON, NEUTRAL, RISK_OFF.
/// Confidence = fraction of sub-detectors agreeing + weighted confidence scores.
#[derive(Debug, Clone, PartialEq)]
pub enum CompositeLevel {
    RiskOn,
    Neutral,
    RiskOff,
}

impl std::fmt::Display for CompositeLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompositeLevel::RiskOn => write!(f, "RISK_ON"),
            CompositeLevel::Neutral => write!(f, "NEUTRAL"),
            CompositeLevel::RiskOff => write!(f, "RISK_OFF"),
        }
    }
}

#[derive(Debug)]
pub struct CompositeRegime {
    pub bh: BHMassRegime,
    pub hurst: HurstRegime,
    pub vol: VolRegime,
    regime: CompositeLevel,
    confidence: f64,
}

impl CompositeRegime {
    pub fn new() -> Self {
        Self {
            bh: BHMassRegime::new(20),
            hurst: HurstRegime::new(100),
            vol: VolRegime::new(252),
            regime: CompositeLevel::Neutral,
            confidence: 0.0,
        }
    }

    /// Map sub-regime labels to a common vote.
    fn bh_vote(&self) -> (i8, f64) {
        match self.bh.current_regime() {
            BHMassLevel::HighMass => (1, self.bh.confidence()),
            BHMassLevel::Forming => (0, self.bh.confidence()),
            BHMassLevel::Low => (-1, self.bh.confidence()),
        }
    }

    fn hurst_vote(&self) -> (i8, f64) {
        match self.hurst.current_regime() {
            HurstLevel::Trending => (1, self.hurst.confidence()),
            HurstLevel::Neutral => (0, self.hurst.confidence()),
            HurstLevel::MeanReverting => (-1, self.hurst.confidence()),
        }
    }

    fn vol_vote(&self) -> (i8, f64) {
        match self.vol.current_regime() {
            VolLevel::LowVol => (1, self.vol.confidence()),  // low vol = risk-on
            VolLevel::Normal => (0, self.vol.confidence()),
            VolLevel::HighVol => (-1, self.vol.confidence()), // high vol = risk-off
        }
    }
}

impl Default for CompositeRegime {
    fn default() -> Self { Self::new() }
}

impl RegimeDetector for CompositeRegime {
    type Regime = CompositeLevel;

    fn update(&mut self, bar: &Bar) -> Option<RegimeChange> {
        self.bh.update(bar);
        self.hurst.update(bar);
        self.vol.update(bar);

        let (bv, bc) = self.bh_vote();
        let (hv, hc) = self.hurst_vote();
        let (vv, vc) = self.vol_vote();

        // Weighted vote
        let score = bv as f64 * bc + hv as f64 * hc + vv as f64 * vc;
        let total_conf = bc + hc + vc;

        let new_regime = if score > 0.2 * total_conf.max(1.0) {
            CompositeLevel::RiskOn
        } else if score < -0.2 * total_conf.max(1.0) {
            CompositeLevel::RiskOff
        } else {
            CompositeLevel::Neutral
        };

        // Count agreements
        let votes = [bv, hv, vv];
        let majority_sign: i8 = if score > 0.0 { 1 } else if score < 0.0 { -1 } else { 0 };
        let agreements = votes.iter().filter(|&&v| v == majority_sign).count();
        self.confidence = (agreements as f64 / 3.0) * (score.abs() / total_conf.max(1.0)).min(1.0);

        if new_regime != self.regime {
            let change = RegimeChange {
                old_regime: self.regime.to_string(),
                new_regime: new_regime.to_string(),
                confidence: self.confidence,
            };
            self.regime = new_regime;
            Some(change)
        } else {
            self.regime = new_regime;
            None
        }
    }

    fn current_regime(&self) -> CompositeLevel { self.regime.clone() }
    fn confidence(&self) -> f64 { self.confidence }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick_aggregator::{Bar, BarType};
    use chrono::Utc;

    fn make_bar(close: f64, open: f64, high: f64, low: f64) -> Bar {
        Bar {
            symbol: "TEST".into(),
            bar_type: BarType::Time,
            open_time: Utc::now(),
            close_time: Utc::now(),
            open,
            high,
            low,
            close,
            volume: 1000.0,
            dollar_volume: close * 1000.0,
            vwap: close,
            num_ticks: 10,
            imbalance: 0.0,
            tick_imbalance: 0.0,
        }
    }

    #[test]
    fn hurst_regime_initializes() {
        let mut h = HurstRegime::new(100);
        for i in 0..150 {
            let p = 100.0 + (i as f64 * 0.1).sin() * 5.0;
            let bar = make_bar(p, p - 0.05, p + 0.1, p - 0.1);
            h.update(&bar);
        }
        let h_val = h.last_hurst();
        assert!(h_val >= 0.0 && h_val <= 1.0, "Hurst must be in [0,1]");
    }

    #[test]
    fn composite_regime_is_displayable() {
        let c = CompositeRegime::new();
        let regime = c.current_regime();
        let s = regime.to_string();
        assert!(!s.is_empty());
    }
}
