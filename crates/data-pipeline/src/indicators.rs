/// Technical indicators — all rolling, strictly no lookahead.

use crate::ohlcv::Bar;

// ── Simple helpers ────────────────────────────────────────────────────────────

fn mean_window(v: &[f64], start: usize, n: usize) -> f64 {
    v[start..start + n].iter().sum::<f64>() / n as f64
}

// ── SMA ───────────────────────────────────────────────────────────────────────

pub fn sma(series: &[f64], period: usize) -> Vec<f64> {
    let n = series.len();
    if n < period { return vec![]; }
    let mut out = Vec::with_capacity(n - period + 1);
    let mut sum: f64 = series[..period].iter().sum();
    out.push(sum / period as f64);
    for i in period..n {
        sum += series[i] - series[i - period];
        out.push(sum / period as f64);
    }
    out
}

// ── EMA ───────────────────────────────────────────────────────────────────────

pub fn ema(series: &[f64], period: usize) -> Vec<f64> {
    let n = series.len();
    if n == 0 { return vec![]; }
    let k = 2.0 / (period as f64 + 1.0);
    let mut out = Vec::with_capacity(n);
    let mut ema_val = series[0];
    out.push(ema_val);
    for &x in &series[1..] {
        ema_val = x * k + ema_val * (1.0 - k);
        out.push(ema_val);
    }
    out
}

// ── RSI ───────────────────────────────────────────────────────────────────────

pub fn rsi(closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    if n <= period { return vec![]; }
    let mut gains = 0.0_f64;
    let mut losses = 0.0_f64;
    for i in 1..=period {
        let diff = closes[i] - closes[i - 1];
        if diff >= 0.0 { gains += diff; } else { losses -= diff; }
    }
    let mut avg_gain = gains / period as f64;
    let mut avg_loss = losses / period as f64;
    let mut out = Vec::with_capacity(n - period);

    let rs = if avg_loss < 1e-10 { 100.0 } else { avg_gain / avg_loss };
    out.push(100.0 - 100.0 / (1.0 + rs));

    for i in period + 1..n {
        let diff = closes[i] - closes[i - 1];
        let g = if diff >= 0.0 { diff } else { 0.0 };
        let l = if diff < 0.0 { -diff } else { 0.0 };
        avg_gain = (avg_gain * (period - 1) as f64 + g) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + l) / period as f64;
        let rs = if avg_loss < 1e-10 { 100.0 } else { avg_gain / avg_loss };
        out.push(100.0 - 100.0 / (1.0 + rs));
    }
    out
}

// ── MACD ──────────────────────────────────────────────────────────────────────

/// Returns Vec<(macd, signal, histogram)>.
pub fn macd(closes: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<(f64, f64, f64)> {
    let ema_fast = ema(closes, fast);
    let ema_slow = ema(closes, slow);
    let n = ema_fast.len().min(ema_slow.len());
    // Align: slow EMA starts later.
    let offset = ema_fast.len() - n;
    let macd_line: Vec<f64> = (0..n).map(|i| ema_fast[i + offset] - ema_slow[i]).collect();
    let signal_line = ema(&macd_line, signal);
    let m = signal_line.len();
    let macd_offset = macd_line.len() - m;
    (0..m)
        .map(|i| {
            let macd_val = macd_line[i + macd_offset];
            let sig_val = signal_line[i];
            (macd_val, sig_val, macd_val - sig_val)
        })
        .collect()
}

// ── Bollinger Bands ───────────────────────────────────────────────────────────

/// Returns Vec<(upper, middle, lower)>.
pub fn bollinger_bands(closes: &[f64], period: usize, num_std: f64) -> Vec<(f64, f64, f64)> {
    let n = closes.len();
    if n < period { return vec![]; }
    (period - 1..n)
        .map(|i| {
            let slice = &closes[i + 1 - period..=i];
            let m = slice.iter().sum::<f64>() / period as f64;
            let std = (slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / period as f64).sqrt();
            (m + num_std * std, m, m - num_std * std)
        })
        .collect()
}

// ── ATR ───────────────────────────────────────────────────────────────────────

pub fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len().min(lows.len()).min(closes.len());
    if n <= 1 { return vec![]; }
    let trs: Vec<f64> = (1..n)
        .map(|i| {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            hl.max(hc).max(lc)
        })
        .collect();
    // Wilder's smoothing.
    let m = trs.len();
    if m < period { return vec![]; }
    let mut atr_val: f64 = trs[..period].iter().sum::<f64>() / period as f64;
    let mut out = vec![atr_val];
    for i in period..m {
        atr_val = (atr_val * (period - 1) as f64 + trs[i]) / period as f64;
        out.push(atr_val);
    }
    out
}

// ── ADX ───────────────────────────────────────────────────────────────────────

pub fn adx(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len().min(lows.len()).min(closes.len());
    if n <= 1 { return vec![]; }

    let mut tr_vec = Vec::with_capacity(n - 1);
    let mut dm_pos = Vec::with_capacity(n - 1);
    let mut dm_neg = Vec::with_capacity(n - 1);

    for i in 1..n {
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
        let up = highs[i] - highs[i - 1];
        let down = lows[i - 1] - lows[i];
        dm_pos.push(if up > down && up > 0.0 { up } else { 0.0 });
        dm_neg.push(if down > up && down > 0.0 { down } else { 0.0 });
        tr_vec.push(tr);
    }

    let m = tr_vec.len();
    if m < period { return vec![]; }

    // Wilder smoothing.
    let mut atr_s = tr_vec[..period].iter().sum::<f64>();
    let mut dm_pos_s = dm_pos[..period].iter().sum::<f64>();
    let mut dm_neg_s = dm_neg[..period].iter().sum::<f64>();

    let mut dx_vals: Vec<f64> = Vec::new();
    let compute_dx = |atr_s: f64, dpos: f64, dneg: f64| -> f64 {
        let di_pos = 100.0 * dpos / atr_s.max(1e-10);
        let di_neg = 100.0 * dneg / atr_s.max(1e-10);
        100.0 * (di_pos - di_neg).abs() / (di_pos + di_neg).max(1e-10)
    };
    dx_vals.push(compute_dx(atr_s, dm_pos_s, dm_neg_s));

    for i in period..m {
        atr_s = atr_s - atr_s / period as f64 + tr_vec[i];
        dm_pos_s = dm_pos_s - dm_pos_s / period as f64 + dm_pos[i];
        dm_neg_s = dm_neg_s - dm_neg_s / period as f64 + dm_neg[i];
        dx_vals.push(compute_dx(atr_s, dm_pos_s, dm_neg_s));
    }

    // ADX = Wilder MA of DX.
    if dx_vals.len() < period { return vec![]; }
    let mut adx_val = dx_vals[..period].iter().sum::<f64>() / period as f64;
    let mut out = vec![adx_val];
    for &dx in &dx_vals[period..] {
        adx_val = (adx_val * (period - 1) as f64 + dx) / period as f64;
        out.push(adx_val);
    }
    out
}

// ── Stochastic ────────────────────────────────────────────────────────────────

/// Returns Vec<(%K, %D)>.
pub fn stochastic(highs: &[f64], lows: &[f64], closes: &[f64], k_period: usize, d_period: usize) -> Vec<(f64, f64)> {
    let n = closes.len().min(highs.len()).min(lows.len());
    if n < k_period { return vec![]; }
    let k: Vec<f64> = (k_period - 1..n)
        .map(|i| {
            let h = highs[i + 1 - k_period..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let l = lows[i + 1 - k_period..=i].iter().copied().fold(f64::INFINITY, f64::min);
            if (h - l).abs() < 1e-10 { 50.0 } else { 100.0 * (closes[i] - l) / (h - l) }
        })
        .collect();
    let d = sma(&k, d_period);
    let m = k.len().min(d.len());
    let k_off = k.len() - m;
    (0..m).map(|i| (k[i + k_off], d[i])).collect()
}

// ── OBV ───────────────────────────────────────────────────────────────────────

pub fn obv(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = closes.len().min(volumes.len());
    if n == 0 { return vec![]; }
    let mut out = vec![0.0_f64];
    let mut running = 0.0_f64;
    for i in 1..n {
        if closes[i] > closes[i - 1] {
            running += volumes[i];
        } else if closes[i] < closes[i - 1] {
            running -= volumes[i];
        }
        out.push(running);
    }
    out
}

// ── VWAP (rolling) ────────────────────────────────────────────────────────────

pub fn vwap_rolling(highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len().min(lows.len()).min(closes.len()).min(volumes.len());
    if n < period { return vec![]; }
    (period - 1..n)
        .map(|i| {
            let pv: f64 = (i + 1 - period..=i)
                .map(|j| ((highs[j] + lows[j] + closes[j]) / 3.0) * volumes[j])
                .sum();
            let vol: f64 = (i + 1 - period..=i).map(|j| volumes[j]).sum::<f64>().max(1e-10);
            pv / vol
        })
        .collect()
}

// ── Donchian Channel ──────────────────────────────────────────────────────────

/// Returns Vec<(upper, lower)>.
pub fn donchian_channel(highs: &[f64], lows: &[f64], period: usize) -> Vec<(f64, f64)> {
    let n = highs.len().min(lows.len());
    if n < period { return vec![]; }
    (period - 1..n)
        .map(|i| {
            let h = highs[i + 1 - period..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let l = lows[i + 1 - period..=i].iter().copied().fold(f64::INFINITY, f64::min);
            (h, l)
        })
        .collect()
}

// ── Keltner Channel ───────────────────────────────────────────────────────────

/// Returns Vec<(upper, middle, lower)>.
pub fn keltner_channel(highs: &[f64], lows: &[f64], closes: &[f64], period: usize, multiplier: f64) -> Vec<(f64, f64, f64)> {
    let ema_c = ema(closes, period);
    let atr_v = atr(highs, lows, closes, period);
    let m = ema_c.len().min(atr_v.len());
    let ema_off = ema_c.len() - m;
    (0..m)
        .map(|i| {
            let mid = ema_c[i + ema_off];
            let band = multiplier * atr_v[i];
            (mid + band, mid, mid - band)
        })
        .collect()
}

// ── Ichimoku ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IchimokuPoint {
    pub tenkan_sen: f64,     // 9-period
    pub kijun_sen: f64,      // 26-period
    pub senkou_span_a: f64,  // (tenkan + kijun) / 2, displaced +26
    pub senkou_span_b: f64,  // 52-period midpoint, displaced +26
    pub chikou_span: f64,    // close displaced -26
}

pub fn ichimoku(highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<IchimokuPoint> {
    let n = highs.len().min(lows.len()).min(closes.len());
    let min_len = 52 + 26;
    if n < min_len { return vec![]; }

    let midpoint = |h_slice: &[f64], l_slice: &[f64]| -> f64 {
        let h = h_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let l = l_slice.iter().copied().fold(f64::INFINITY, f64::min);
        (h + l) / 2.0
    };

    let mut out = Vec::new();
    for i in 51..n {
        if i + 26 > n { break; }
        let tenkan = midpoint(&highs[i - 8..=i], &lows[i - 8..=i]);
        let kijun = midpoint(&highs[i - 25..=i], &lows[i - 25..=i]);
        let span_a = (tenkan + kijun) / 2.0;
        let span_b = midpoint(&highs[i - 51..=i], &lows[i - 51..=i]);
        let chikou = if i >= 26 { closes[i - 26] } else { closes[0] };
        out.push(IchimokuPoint {
            tenkan_sen: tenkan,
            kijun_sen: kijun,
            senkou_span_a: span_a,
            senkou_span_b: span_b,
            chikou_span: chikou,
        });
    }
    out
}

// ── Heikin-Ashi ───────────────────────────────────────────────────────────────

pub fn heikin_ashi(bars: &[Bar]) -> Vec<Bar> {
    if bars.is_empty() { return vec![]; }
    let mut out = Vec::with_capacity(bars.len());
    let mut prev_ha_open = (bars[0].open + bars[0].close) / 2.0;
    let mut prev_ha_close = (bars[0].open + bars[0].high + bars[0].low + bars[0].close) / 4.0;
    out.push(Bar::new(
        bars[0].timestamp,
        prev_ha_open, bars[0].high, bars[0].low, prev_ha_close, bars[0].volume,
    ));
    for bar in bars.iter().skip(1) {
        let ha_close = (bar.open + bar.high + bar.low + bar.close) / 4.0;
        let ha_open = (prev_ha_open + prev_ha_close) / 2.0;
        let ha_high = bar.high.max(ha_open).max(ha_close);
        let ha_low = bar.low.min(ha_open).min(ha_close);
        out.push(Bar::new(bar.timestamp, ha_open, ha_high, ha_low, ha_close, bar.volume));
        prev_ha_open = ha_open;
        prev_ha_close = ha_close;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sma_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = sma(&data, 3);
        assert_eq!(s.len(), 3);
        assert!((s[0] - 2.0).abs() < 1e-10);
        assert!((s[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn rsi_in_range() {
        let closes: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0).collect();
        let r = rsi(&closes, 14);
        for v in &r {
            assert!(*v >= 0.0 && *v <= 100.0, "rsi={v}");
        }
    }

    #[test]
    fn macd_has_three_components() {
        let closes: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();
        let m = macd(&closes, 12, 26, 9);
        assert!(!m.is_empty());
    }

    #[test]
    fn bollinger_bands_upper_ge_lower() {
        let closes: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.1).collect();
        for (u, m, l) in bollinger_bands(&closes, 20, 2.0) {
            assert!(u >= l, "u={u} l={l}");
        }
    }

    #[test]
    fn obv_monotone_ascending() {
        let closes = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vols = vec![100.0; 5];
        let o = obv(&closes, &vols);
        for i in 1..o.len() {
            assert!(o[i] >= o[i - 1]);
        }
    }
}
