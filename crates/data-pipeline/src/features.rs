/// ML feature engineering for price data.

use crate::ohlcv::{Bar, BarSeries, returns};
use crate::indicators::{sma, ema, rsi, atr};
use chrono::{DateTime, Utc, Datelike, Weekday};

pub type Matrix = Vec<Vec<f64>>;

fn mean_f(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_f(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean_f(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

fn skewness(v: &[f64]) -> f64 {
    if v.len() < 3 { return 0.0; }
    let n = v.len() as f64;
    let m = mean_f(v);
    let s = std_f(v).max(1e-12);
    v.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

fn kurtosis(v: &[f64]) -> f64 {
    if v.len() < 4 { return 0.0; }
    let n = v.len() as f64;
    let m = mean_f(v);
    let s = std_f(v).max(1e-12);
    v.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n - 3.0
}

// ── Price Features ────────────────────────────────────────────────────────────

/// Compute price-based features at multiple lookback horizons.
///
/// For each bar t and each lookback L, computes:
///   - return: log(close_t / close_{t-L})
///   - volatility: std(log-returns over last L bars)
///   - skewness of returns
///   - kurtosis of returns
///   - price relative to its L-bar SMA
///   - L-bar price range normalised by current price
///
/// Returns a (T - max_lookback) × (6 * num_lookbacks) matrix.
pub fn price_features(bars: &BarSeries, lookbacks: &[usize]) -> Matrix {
    let closes = bars.closes();
    let log_rets = returns(bars); // length n-1
    let n = closes.len();
    let max_lb = *lookbacks.iter().max().unwrap_or(&1);
    if n <= max_lb { return vec![]; }

    let num_features = 6 * lookbacks.len();
    let t_start = max_lb;

    (t_start..n)
        .map(|t| {
            let mut row = Vec::with_capacity(num_features);
            for &lb in lookbacks {
                // Return over lookback.
                let ret = if lb > 0 && t >= lb {
                    (closes[t] / closes[t - lb].max(1e-10)).ln()
                } else { 0.0 };

                // Volatility over lookback (std of log returns).
                let vol = if t >= lb && lb >= 2 {
                    let slice = &log_rets[t.saturating_sub(lb)..t.saturating_sub(1).min(log_rets.len())];
                    std_f(slice)
                } else { 0.0 };

                // Skewness.
                let skew = if t >= lb && lb >= 3 {
                    let slice = &log_rets[t.saturating_sub(lb)..t.saturating_sub(1).min(log_rets.len())];
                    skewness(slice)
                } else { 0.0 };

                // Kurtosis.
                let kurt = if t >= lb && lb >= 4 {
                    let slice = &log_rets[t.saturating_sub(lb)..t.saturating_sub(1).min(log_rets.len())];
                    kurtosis(slice)
                } else { 0.0 };

                // Price vs SMA.
                let sma_val = if t >= lb {
                    closes[t - lb..=t].iter().sum::<f64>() / (lb + 1) as f64
                } else { closes[t] };
                let price_to_sma = closes[t] / sma_val.max(1e-10) - 1.0;

                // Normalised range.
                let hi = bars.highs()[t - lb.min(t)..=t].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lo = bars.lows()[t - lb.min(t)..=t].iter().copied().fold(f64::INFINITY, f64::min);
                let norm_range = (hi - lo) / closes[t].max(1e-10);

                row.extend_from_slice(&[ret, vol, skew, kurt, price_to_sma, norm_range]);
            }
            row
        })
        .collect()
}

// ── Microstructure Features ───────────────────────────────────────────────────

/// Microstructure proxy features: spread proxy, Amihud illiquidity, Roll spread.
///
/// Returns (T-1) × 3 matrix: [spread_proxy, amihud, roll_spread].
pub fn microstructure_features(bars: &BarSeries) -> Matrix {
    let n = bars.len();
    if n < 3 { return vec![]; }
    let closes = bars.closes();
    let highs = bars.highs();
    let lows = bars.lows();
    let vols = bars.volumes();
    let log_rets = returns(bars);

    (1..n)
        .map(|t| {
            // Spread proxy: (High - Low) / Close (Corwin-Schultz-like).
            let spread_proxy = (highs[t] - lows[t]) / closes[t].max(1e-10);

            // Amihud: |r_t| / dollar_volume.
            let dollar_vol = closes[t] * vols[t];
            let amihud = if dollar_vol > 0.0 {
                log_rets[t - 1].abs() / dollar_vol
            } else { 0.0 };

            // Roll spread estimate from last 2 return pairs.
            let roll = if t >= 2 {
                let cov = (log_rets[t - 2] - mean_f(&log_rets[t.saturating_sub(5)..t]))
                    * (log_rets[t - 1] - mean_f(&log_rets[t.saturating_sub(5)..t]));
                if cov < 0.0 { 2.0 * (-cov).sqrt() } else { 0.0 }
            } else { 0.0 };

            vec![spread_proxy, amihud * 1e6, roll]
        })
        .collect()
}

// ── Calendar Features ─────────────────────────────────────────────────────────

/// Calendar-based features: day-of-week effects, month, quarter, seasonality.
///
/// Returns T × 8 matrix per timestamp:
///   [dow_0..4 (one-hot), is_month_end, month_sin, month_cos]
pub fn calendar_features(timestamps: &[DateTime<Utc>]) -> Matrix {
    timestamps
        .iter()
        .map(|ts| {
            let dow = ts.weekday().num_days_from_monday() as f64; // 0=Mon..4=Fri
            let month = ts.month() as f64;
            let day = ts.day() as f64;
            let days_in_month = days_in_month(ts.month(), ts.year()) as f64;
            let is_month_end = if day >= days_in_month - 1.0 { 1.0 } else { 0.0 };
            let is_monday = if ts.weekday() == Weekday::Mon { 1.0 } else { 0.0 };
            let is_friday = if ts.weekday() == Weekday::Fri { 1.0 } else { 0.0 };
            let month_sin = (2.0 * std::f64::consts::PI * month / 12.0).sin();
            let month_cos = (2.0 * std::f64::consts::PI * month / 12.0).cos();
            let quarter = ((month - 1.0) / 3.0).floor() + 1.0;
            vec![
                dow / 4.0, // normalised 0-1
                is_month_end,
                is_monday,
                is_friday,
                month_sin,
                month_cos,
                quarter / 4.0,
                day / days_in_month, // day-of-month normalised
            ]
        })
        .collect()
}

fn days_in_month(month: u32, year: i32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 29 } else { 28 },
        _ => 30,
    }
}

// ── Regime Features ───────────────────────────────────────────────────────────

/// Regime-based features: trend strength at 3 timeframes, vol regime, BH mass.
///
/// Returns (T - max_period) × 9 matrix:
///   [trend_20, trend_50, trend_200, vol_20, vol_50, vol_200, above_20ma, above_50ma, above_200ma]
pub fn regime_features(bars: &BarSeries) -> Matrix {
    let n = bars.len();
    if n < 200 { return vec![]; }
    let closes = bars.closes();
    let log_rets = returns(bars);

    let sma20 = sma(&closes, 20);
    let sma50 = sma(&closes, 50);
    let sma200 = sma(&closes, 200);

    (200..n)
        .map(|t| {
            let c = closes[t];
            let off20 = sma20.len() - (n - t + 20 - 1).min(sma20.len());
            let off50 = sma50.len() - (n - t + 50 - 1).min(sma50.len());
            let off200 = sma200.len() - (n - t + 200 - 1).min(sma200.len());

            let i20 = if off20 < sma20.len() { sma20[off20] } else { c };
            let i50 = if off50 < sma50.len() { sma50[off50] } else { c };
            let i200 = if off200 < sma200.len() { sma200[off200] } else { c };

            let trend20 = (c / i20.max(1e-10) - 1.0).clamp(-0.5, 0.5);
            let trend50 = (c / i50.max(1e-10) - 1.0).clamp(-0.5, 0.5);
            let trend200 = (c / i200.max(1e-10) - 1.0).clamp(-0.5, 0.5);

            let vol20 = std_f(&log_rets[t.saturating_sub(20)..t.min(log_rets.len())]);
            let vol50 = std_f(&log_rets[t.saturating_sub(50)..t.min(log_rets.len())]);
            let vol200 = std_f(&log_rets[t.saturating_sub(200)..t.min(log_rets.len())]);

            let above_20 = if c > i20 { 1.0 } else { 0.0 };
            let above_50 = if c > i50 { 1.0 } else { 0.0 };
            let above_200 = if c > i200 { 1.0 } else { 0.0 };

            vec![trend20, trend50, trend200, vol20, vol50, vol200, above_20, above_50, above_200]
        })
        .collect()
}

// ── Cross-sectional Features ──────────────────────────────────────────────────

/// Cross-sectional features across a universe of bar series.
///
/// For each bar (aligned by position), computes:
///   - cross-sectional rank of 1M momentum
///   - average pairwise return correlation
///
/// Returns T × 2 matrix.
pub fn cross_sectional_features(all_bars: &[BarSeries]) -> Matrix {
    let n_assets = all_bars.len();
    if n_assets == 0 { return vec![]; }

    let min_len = all_bars.iter().map(|b| b.len()).min().unwrap_or(0);
    if min_len < 21 { return vec![]; }

    (21..min_len)
        .map(|t| {
            // 1-month momentum for each asset.
            let moms: Vec<f64> = all_bars
                .iter()
                .map(|b| {
                    let c = b.bars[t].close;
                    let c_prev = b.bars[t - 20].close;
                    (c / c_prev.max(1e-10)).ln()
                })
                .collect();

            // Cross-sectional rank of momentum.
            let rank = cross_sectional_rank(&moms);
            let mean_rank = mean_f(&rank);

            // Average correlation (simplified: use returns comovement sign).
            let returns_today: Vec<f64> = all_bars
                .iter()
                .map(|b| (b.bars[t].close / b.bars[t - 1].close.max(1e-10)).ln())
                .collect();
            let mean_ret = mean_f(&returns_today);
            let corr_proxy = returns_today.iter().map(|r| (r - mean_ret).abs()).sum::<f64>()
                / n_assets as f64;

            vec![mean_rank, corr_proxy]
        })
        .collect()
}

fn cross_sectional_rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0_f64; n];
    for (rank, (orig_idx, _)) in indexed.into_iter().enumerate() {
        ranks[orig_idx] = rank as f64 / (n - 1).max(1) as f64;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn make_bars(n: usize) -> BarSeries {
        let start = Utc::now();
        let bars: Vec<Bar> = (0..n)
            .map(|i| {
                let p = 100.0 + i as f64 * 0.1;
                Bar::new(start + Duration::days(i as i64), p, p + 0.5, p - 0.3, p + 0.2, 1000.0)
            })
            .collect();
        BarSeries::from_bars(bars, "TEST")
    }

    #[test]
    fn price_features_shape() {
        let bars = make_bars(100);
        let feats = price_features(&bars, &[5, 20, 60]);
        assert!(!feats.is_empty());
        assert_eq!(feats[0].len(), 18); // 6 features × 3 lookbacks
    }

    #[test]
    fn calendar_features_shape() {
        let now = Utc::now();
        let ts: Vec<DateTime<Utc>> = (0..10).map(|i| now + Duration::days(i)).collect();
        let feats = calendar_features(&ts);
        assert_eq!(feats.len(), 10);
        assert_eq!(feats[0].len(), 8);
    }
}
