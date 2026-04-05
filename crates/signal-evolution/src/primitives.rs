/// Terminal node primitives for the signal expression tree.
///
/// Each terminal reads from a BarData slice and produces a Vec<f64> signal series.
/// All computations are pure functions of the bar history.

use crate::data_loader::BarData;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Terminal enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Terminal {
    /// Raw close price
    Price,
    /// Raw volume
    Volume,
    /// Average True Range (14-bar default)
    ATR { period: usize },
    /// Relative Strength Index
    RSI { period: usize },
    /// Exponential moving average
    EMA { period: usize },
    /// Simple moving average
    SMA { period: usize },
    /// n-bar returns: (close[t] - close[t-n]) / close[t-n]
    Returns { n: usize },
    /// Rolling realized volatility (std of log-returns)
    Volatility { window: usize },
    /// Black Hole mass proxy: cumulative volume-weighted price deviation
    BHMass,
    /// Black Hole active flag: 1.0 if mass > threshold, else 0.0
    BHActive,
    /// Ornstein-Uhlenbeck z-score: (price - rolling_mean) / rolling_std
    OUZScore { window: usize },
}

impl Terminal {
    /// Evaluate terminal over full bar history, returning one value per bar.
    pub fn evaluate(&self, bars: &[BarData]) -> Vec<f64> {
        let n = bars.len();
        match self {
            Terminal::Price => bars.iter().map(|b| b.close).collect(),

            Terminal::Volume => bars.iter().map(|b| b.volume).collect(),

            Terminal::ATR { period } => compute_atr(bars, *period),

            Terminal::RSI { period } => compute_rsi(bars, *period),

            Terminal::EMA { period } => compute_ema(bars, *period),

            Terminal::SMA { period } => {
                let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
                compute_sma(&closes, *period)
            }

            Terminal::Returns { n: lag } => {
                let mut out = vec![0.0; n];
                for i in *lag..n {
                    let p0 = bars[i - lag].close;
                    if p0 != 0.0 {
                        out[i] = (bars[i].close - p0) / p0;
                    }
                }
                out
            }

            Terminal::Volatility { window } => {
                let log_rets: Vec<f64> = bars.iter().map(|b| b.log_returns).collect();
                crate::data_loader::rolling_std(&log_rets, *window)
            }

            Terminal::BHMass => compute_bh_mass(bars),

            Terminal::BHActive => {
                let mass = compute_bh_mass(bars);
                let threshold = mass.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 0.7;
                mass.iter().map(|&m| if m > threshold { 1.0 } else { 0.0 }).collect()
            }

            Terminal::OUZScore { window } => {
                let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
                compute_ou_zscore(&closes, *window)
            }
        }
    }

    /// Human-readable name for display in formulas.
    pub fn name(&self) -> String {
        match self {
            Terminal::Price => "Price".to_string(),
            Terminal::Volume => "Volume".to_string(),
            Terminal::ATR { period } => format!("ATR({period})"),
            Terminal::RSI { period } => format!("RSI({period})"),
            Terminal::EMA { period } => format!("EMA({period})"),
            Terminal::SMA { period } => format!("SMA({period})"),
            Terminal::Returns { n } => format!("Returns({n})"),
            Terminal::Volatility { window } => format!("Vol({window})"),
            Terminal::BHMass => "BHMass".to_string(),
            Terminal::BHActive => "BHActive".to_string(),
            Terminal::OUZScore { window } => format!("OUZ({window})"),
        }
    }

    /// All available terminal variants for random sampling.
    pub fn all_variants() -> Vec<Terminal> {
        vec![
            Terminal::Price,
            Terminal::Volume,
            Terminal::ATR { period: 14 },
            Terminal::ATR { period: 7 },
            Terminal::RSI { period: 14 },
            Terminal::RSI { period: 7 },
            Terminal::EMA { period: 10 },
            Terminal::EMA { period: 20 },
            Terminal::EMA { period: 50 },
            Terminal::SMA { period: 10 },
            Terminal::SMA { period: 20 },
            Terminal::SMA { period: 50 },
            Terminal::Returns { n: 1 },
            Terminal::Returns { n: 5 },
            Terminal::Returns { n: 20 },
            Terminal::Volatility { window: 20 },
            Terminal::Volatility { window: 10 },
            Terminal::BHMass,
            Terminal::BHActive,
            Terminal::OUZScore { window: 20 },
            Terminal::OUZScore { window: 10 },
        ]
    }
}

// ---------------------------------------------------------------------------
// Indicator implementations
// ---------------------------------------------------------------------------

fn compute_atr(bars: &[BarData], period: usize) -> Vec<f64> {
    let n = bars.len();
    let mut tr = vec![0.0f64; n];
    for i in 1..n {
        let hl = bars[i].high - bars[i].low;
        let hc = (bars[i].high - bars[i - 1].close).abs();
        let lc = (bars[i].low - bars[i - 1].close).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    // Wilder smoothing
    let mut atr = vec![0.0f64; n];
    if n > period {
        atr[period] = tr[1..=period].iter().sum::<f64>() / period as f64;
        let alpha = 1.0 / period as f64;
        for i in (period + 1)..n {
            atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha;
        }
    }
    atr
}

fn compute_rsi(bars: &[BarData], period: usize) -> Vec<f64> {
    let n = bars.len();
    let mut rsi = vec![50.0f64; n];
    if n <= period {
        return rsi;
    }
    let mut gains = vec![0.0f64; n];
    let mut losses = vec![0.0f64; n];
    for i in 1..n {
        let delta = bars[i].close - bars[i - 1].close;
        if delta > 0.0 {
            gains[i] = delta;
        } else {
            losses[i] = -delta;
        }
    }
    // Initial averages
    let avg_gain_init: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
    let avg_loss_init: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_gain = avg_gain_init;
    let mut avg_loss = avg_loss_init;
    let alpha = 1.0 / period as f64;
    rsi[period] = if avg_loss == 0.0 {
        100.0
    } else {
        100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    };
    for i in (period + 1)..n {
        avg_gain = avg_gain * (1.0 - alpha) + gains[i] * alpha;
        avg_loss = avg_loss * (1.0 - alpha) + losses[i] * alpha;
        rsi[i] = if avg_loss == 0.0 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        };
    }
    rsi
}

fn compute_ema(bars: &[BarData], period: usize) -> Vec<f64> {
    let n = bars.len();
    let mut ema = vec![0.0f64; n];
    if n == 0 {
        return ema;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    ema[0] = bars[0].close;
    for i in 1..n {
        ema[i] = ema[i - 1] * (1.0 - alpha) + bars[i].close * alpha;
    }
    ema
}

pub fn compute_sma(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut sma = vec![0.0f64; n];
    for i in period..n {
        sma[i] = values[(i - period)..i].iter().sum::<f64>() / period as f64;
    }
    sma
}

fn compute_bh_mass(bars: &[BarData]) -> Vec<f64> {
    // BH mass: volume-weighted cumulative deviation of price from 20-bar SMA
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let sma = compute_sma(&closes, 20);
    let n = bars.len();
    let mut mass = vec![0.0f64; n];
    let decay = 0.97_f64;
    for i in 1..n {
        let dev = if sma[i] != 0.0 {
            (bars[i].close - sma[i]).abs() / sma[i]
        } else {
            0.0
        };
        let vol_norm = bars[i].volume / (bars[i - 1].volume.max(1.0));
        mass[i] = mass[i - 1] * decay + dev * vol_norm;
    }
    mass
}

fn compute_ou_zscore(closes: &[f64], window: usize) -> Vec<f64> {
    let n = closes.len();
    let mut out = vec![0.0f64; n];
    let sma = compute_sma(closes, window);
    for i in window..n {
        let slice = &closes[(i - window)..i];
        let mean = sma[i];
        let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window - 1).max(1) as f64;
        let std = var.sqrt();
        if std > 1e-10 {
            out[i] = (closes[i] - mean) / std;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::synthetic_bars;

    #[test]
    fn price_terminal_matches_close() {
        let bars = synthetic_bars(50, 100.0);
        let vals = Terminal::Price.evaluate(&bars);
        for (v, b) in vals.iter().zip(bars.iter()) {
            assert!((v - b.close).abs() < 1e-12);
        }
    }

    #[test]
    fn rsi_within_range() {
        let bars = synthetic_bars(100, 100.0);
        let vals = Terminal::RSI { period: 14 }.evaluate(&bars);
        for v in &vals {
            assert!(*v >= 0.0 && *v <= 100.0, "RSI out of range: {v}");
        }
    }

    #[test]
    fn ema_length_matches_input() {
        let bars = synthetic_bars(80, 50.0);
        let vals = Terminal::EMA { period: 20 }.evaluate(&bars);
        assert_eq!(vals.len(), bars.len());
    }

    #[test]
    fn all_variants_evaluate_without_panic() {
        let bars = synthetic_bars(100, 100.0);
        for t in Terminal::all_variants() {
            let v = t.evaluate(&bars);
            assert_eq!(v.len(), bars.len(), "Terminal {:?} wrong len", t);
        }
    }
}
