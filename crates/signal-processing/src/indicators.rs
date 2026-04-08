use std::collections::VecDeque;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Simple Moving Average (SMA)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Sma {
    period: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

impl Sma {
    pub fn new(period: usize) -> Self {
        assert!(period > 0);
        Self { period, buffer: VecDeque::with_capacity(period + 1), sum: 0.0 }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        self.sum += value;
        if self.buffer.len() > self.period {
            self.sum -= self.buffer.pop_front().unwrap();
        }
        if self.buffer.len() == self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }

    pub fn current(&self) -> Option<f64> {
        if self.buffer.len() == self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut sma = Self::new(period);
        data.iter().map(|&v| sma.update(v)).collect()
    }

    pub fn period(&self) -> usize {
        self.period
    }

    pub fn is_ready(&self) -> bool {
        self.buffer.len() == self.period
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Exponential Moving Average (EMA)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Ema {
    period: usize,
    multiplier: f64,
    current: Option<f64>,
    count: usize,
    sum: f64,
}

impl Ema {
    pub fn new(period: usize) -> Self {
        assert!(period > 0);
        let multiplier = 2.0 / (period as f64 + 1.0);
        Self { period, multiplier, current: None, count: 0, sum: 0.0 }
    }

    pub fn with_multiplier(period: usize, multiplier: f64) -> Self {
        Self { period, multiplier, current: None, count: 0, sum: 0.0 }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.count += 1;
        match self.current {
            Some(prev) => {
                let ema = value * self.multiplier + prev * (1.0 - self.multiplier);
                self.current = Some(ema);
                Some(ema)
            }
            None => {
                self.sum += value;
                if self.count >= self.period {
                    let sma = self.sum / self.period as f64;
                    self.current = Some(sma);
                    Some(sma)
                } else {
                    None
                }
            }
        }
    }

    pub fn current(&self) -> Option<f64> {
        self.current
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut ema = Self::new(period);
        data.iter().map(|&v| ema.update(v)).collect()
    }

    pub fn is_ready(&self) -> bool {
        self.current.is_some()
    }

    pub fn reset(&mut self) {
        self.current = None;
        self.count = 0;
        self.sum = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Double Exponential Moving Average (DEMA)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Dema {
    ema1: Ema,
    ema2: Ema,
}

impl Dema {
    pub fn new(period: usize) -> Self {
        Self { ema1: Ema::new(period), ema2: Ema::new(period) }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        let e1 = self.ema1.update(value);
        if let Some(e1v) = e1 {
            let e2 = self.ema2.update(e1v);
            if let Some(e2v) = e2 {
                return Some(2.0 * e1v - e2v);
            }
        }
        None
    }

    pub fn current(&self) -> Option<f64> {
        match (self.ema1.current(), self.ema2.current()) {
            (Some(e1), Some(e2)) => Some(2.0 * e1 - e2),
            _ => None,
        }
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut d = Self::new(period);
        data.iter().map(|&v| d.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
    }
}

// ---------------------------------------------------------------------------
// Triple Exponential Moving Average (TEMA)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Tema {
    ema1: Ema,
    ema2: Ema,
    ema3: Ema,
}

impl Tema {
    pub fn new(period: usize) -> Self {
        Self { ema1: Ema::new(period), ema2: Ema::new(period), ema3: Ema::new(period) }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        let e1 = self.ema1.update(value);
        if let Some(e1v) = e1 {
            let e2 = self.ema2.update(e1v);
            if let Some(e2v) = e2 {
                let e3 = self.ema3.update(e2v);
                if let Some(e3v) = e3 {
                    return Some(3.0 * e1v - 3.0 * e2v + e3v);
                }
            }
        }
        None
    }

    pub fn current(&self) -> Option<f64> {
        match (self.ema1.current(), self.ema2.current(), self.ema3.current()) {
            (Some(e1), Some(e2), Some(e3)) => Some(3.0 * e1 - 3.0 * e2 + e3),
            _ => None,
        }
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut t = Self::new(period);
        data.iter().map(|&v| t.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
        self.ema3.reset();
    }
}

// ---------------------------------------------------------------------------
// Weighted Moving Average (WMA)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Wma {
    period: usize,
    buffer: VecDeque<f64>,
    denom: f64,
}

impl Wma {
    pub fn new(period: usize) -> Self {
        assert!(period > 0);
        let denom = (period * (period + 1)) as f64 / 2.0;
        Self { period, buffer: VecDeque::with_capacity(period + 1), denom }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() == self.period {
            let mut s = 0.0;
            for (i, &v) in self.buffer.iter().enumerate() {
                s += v * (i + 1) as f64;
            }
            Some(s / self.denom)
        } else {
            None
        }
    }

    pub fn current(&self) -> Option<f64> {
        if self.buffer.len() == self.period {
            let mut s = 0.0;
            for (i, &v) in self.buffer.iter().enumerate() {
                s += v * (i + 1) as f64;
            }
            Some(s / self.denom)
        } else {
            None
        }
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut w = Self::new(period);
        data.iter().map(|&v| w.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Kaufman Adaptive Moving Average (KAMA)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Kama {
    period: usize,
    fast_sc: f64,
    slow_sc: f64,
    buffer: VecDeque<f64>,
    current: Option<f64>,
}

impl Kama {
    pub fn new(period: usize, fast_period: usize, slow_period: usize) -> Self {
        let fast_sc = 2.0 / (fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (slow_period as f64 + 1.0);
        Self {
            period,
            fast_sc,
            slow_sc,
            buffer: VecDeque::with_capacity(period + 2),
            current: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }
        if self.buffer.len() <= self.period {
            if self.buffer.len() == self.period {
                self.current = Some(value);
                return Some(value);
            }
            return None;
        }
        let direction = (value - self.buffer[0]).abs();
        let mut volatility = 0.0;
        for i in 1..self.buffer.len() {
            volatility += (self.buffer[i] - self.buffer[i - 1]).abs();
        }
        let er = if volatility > 1e-15 { direction / volatility } else { 0.0 };
        let sc = er * (self.fast_sc - self.slow_sc) + self.slow_sc;
        let sc2 = sc * sc;
        let prev = self.current.unwrap();
        let kama = prev + sc2 * (value - prev);
        self.current = Some(kama);
        Some(kama)
    }

    pub fn current(&self) -> Option<f64> {
        self.current
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut k = Self::new(period, 2, 30);
        data.iter().map(|&v| k.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.current = None;
    }
}

// ---------------------------------------------------------------------------
// Hull Moving Average (HMA)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Hma {
    wma_half: Wma,
    wma_full: Wma,
    wma_sqrt: Wma,
    buffer: VecDeque<f64>,
    period: usize,
    sqrt_period: usize,
}

impl Hma {
    pub fn new(period: usize) -> Self {
        let half = period / 2;
        let sqrt_period = (period as f64).sqrt() as usize;
        Self {
            wma_half: Wma::new(if half > 0 { half } else { 1 }),
            wma_full: Wma::new(period),
            wma_sqrt: Wma::new(if sqrt_period > 0 { sqrt_period } else { 1 }),
            buffer: VecDeque::new(),
            period,
            sqrt_period,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        let h = self.wma_half.update(value);
        let f = self.wma_full.update(value);
        match (h, f) {
            (Some(hv), Some(fv)) => {
                let diff = 2.0 * hv - fv;
                self.wma_sqrt.update(diff)
            }
            _ => None,
        }
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut h = Self::new(period);
        data.iter().map(|&v| h.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.wma_half.reset();
        self.wma_full.reset();
        self.wma_sqrt.reset();
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Volume Weighted Moving Average (VWMA)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Vwma {
    period: usize,
    prices: VecDeque<f64>,
    volumes: VecDeque<f64>,
    pv_sum: f64,
    v_sum: f64,
}

impl Vwma {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prices: VecDeque::with_capacity(period + 1),
            volumes: VecDeque::with_capacity(period + 1),
            pv_sum: 0.0,
            v_sum: 0.0,
        }
    }

    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.prices.push_back(price);
        self.volumes.push_back(volume);
        self.pv_sum += price * volume;
        self.v_sum += volume;
        if self.prices.len() > self.period {
            let old_p = self.prices.pop_front().unwrap();
            let old_v = self.volumes.pop_front().unwrap();
            self.pv_sum -= old_p * old_v;
            self.v_sum -= old_v;
        }
        if self.prices.len() == self.period && self.v_sum.abs() > 1e-15 {
            Some(self.pv_sum / self.v_sum)
        } else {
            None
        }
    }

    pub fn current(&self) -> Option<f64> {
        if self.prices.len() == self.period && self.v_sum.abs() > 1e-15 {
            Some(self.pv_sum / self.v_sum)
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.prices.clear();
        self.volumes.clear();
        self.pv_sum = 0.0;
        self.v_sum = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Relative Strength Index (RSI)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Rsi {
    period: usize,
    avg_gain: f64,
    avg_loss: f64,
    prev_value: Option<f64>,
    count: usize,
    gains: Vec<f64>,
    losses: Vec<f64>,
}

impl Rsi {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev_value: None,
            count: 0,
            gains: Vec::new(),
            losses: Vec::new(),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        if let Some(prev) = self.prev_value {
            let change = value - prev;
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };
            self.count += 1;
            if self.count < self.period {
                self.gains.push(gain);
                self.losses.push(loss);
                self.prev_value = Some(value);
                return None;
            } else if self.count == self.period {
                self.gains.push(gain);
                self.losses.push(loss);
                self.avg_gain = self.gains.iter().sum::<f64>() / self.period as f64;
                self.avg_loss = self.losses.iter().sum::<f64>() / self.period as f64;
            } else {
                self.avg_gain = (self.avg_gain * (self.period as f64 - 1.0) + gain) / self.period as f64;
                self.avg_loss = (self.avg_loss * (self.period as f64 - 1.0) + loss) / self.period as f64;
            }
            self.prev_value = Some(value);
            if self.avg_loss < 1e-15 {
                return Some(100.0);
            }
            let rs = self.avg_gain / self.avg_loss;
            Some(100.0 - 100.0 / (1.0 + rs))
        } else {
            self.prev_value = Some(value);
            None
        }
    }

    pub fn current(&self) -> Option<f64> {
        if self.count >= self.period {
            if self.avg_loss < 1e-15 {
                return Some(100.0);
            }
            let rs = self.avg_gain / self.avg_loss;
            Some(100.0 - 100.0 / (1.0 + rs))
        } else {
            None
        }
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut r = Self::new(period);
        data.iter().map(|&v| r.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.prev_value = None;
        self.count = 0;
        self.gains.clear();
        self.losses.clear();
    }
}

// ---------------------------------------------------------------------------
// Stochastic RSI
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct StochasticRsi {
    rsi: Rsi,
    period: usize,
    k_period: usize,
    d_period: usize,
    rsi_buffer: VecDeque<f64>,
    k_buffer: VecDeque<f64>,
}

impl StochasticRsi {
    pub fn new(rsi_period: usize, stoch_period: usize, k_smooth: usize, d_smooth: usize) -> Self {
        Self {
            rsi: Rsi::new(rsi_period),
            period: stoch_period,
            k_period: k_smooth,
            d_period: d_smooth,
            rsi_buffer: VecDeque::with_capacity(stoch_period + 1),
            k_buffer: VecDeque::with_capacity(d_smooth + 1),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let rsi_val = self.rsi.update(value);
        if let Some(rv) = rsi_val {
            self.rsi_buffer.push_back(rv);
            if self.rsi_buffer.len() > self.period {
                self.rsi_buffer.pop_front();
            }
            if self.rsi_buffer.len() == self.period {
                let min = self.rsi_buffer.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = self.rsi_buffer.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let stoch_rsi = if (max - min).abs() > 1e-15 {
                    (rv - min) / (max - min)
                } else {
                    0.5
                };
                self.k_buffer.push_back(stoch_rsi);
                if self.k_buffer.len() > self.d_period {
                    self.k_buffer.pop_front();
                }
                let k = stoch_rsi * 100.0;
                let d = if self.k_buffer.len() >= self.d_period {
                    self.k_buffer.iter().sum::<f64>() / self.k_buffer.len() as f64 * 100.0
                } else {
                    k
                };
                return Some((k, d));
            }
        }
        None
    }

    pub fn calculate(data: &[f64], rsi_period: usize, stoch_period: usize) -> Vec<Option<(f64, f64)>> {
        let mut s = Self::new(rsi_period, stoch_period, 3, 3);
        data.iter().map(|&v| s.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.rsi.reset();
        self.rsi_buffer.clear();
        self.k_buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// MACD (Moving Average Convergence Divergence)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MacdOutput {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
}

#[derive(Debug, Clone)]
pub struct Macd {
    fast_ema: Ema,
    slow_ema: Ema,
    signal_ema: Ema,
}

impl Macd {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_ema: Ema::new(fast_period),
            slow_ema: Ema::new(slow_period),
            signal_ema: Ema::new(signal_period),
        }
    }

    pub fn default_periods() -> Self {
        Self::new(12, 26, 9)
    }

    pub fn update(&mut self, value: f64) -> Option<MacdOutput> {
        let fast = self.fast_ema.update(value);
        let slow = self.slow_ema.update(value);
        match (fast, slow) {
            (Some(f), Some(s)) => {
                let macd_line = f - s;
                let sig = self.signal_ema.update(macd_line);
                match sig {
                    Some(signal_line) => Some(MacdOutput {
                        macd_line,
                        signal_line,
                        histogram: macd_line - signal_line,
                    }),
                    None => Some(MacdOutput {
                        macd_line,
                        signal_line: macd_line,
                        histogram: 0.0,
                    }),
                }
            }
            _ => None,
        }
    }

    pub fn calculate(data: &[f64]) -> Vec<Option<MacdOutput>> {
        let mut m = Self::default_periods();
        data.iter().map(|&v| m.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_ema.reset();
    }
}

// ---------------------------------------------------------------------------
// Bollinger Bands
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct BollingerOutput {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
    pub percent_b: f64,
    pub bandwidth: f64,
}

#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    num_std: f64,
    buffer: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl BollingerBands {
    pub fn new(period: usize, num_std: f64) -> Self {
        Self {
            period,
            num_std,
            buffer: VecDeque::with_capacity(period + 1),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    pub fn default_params() -> Self {
        Self::new(20, 2.0)
    }

    pub fn update(&mut self, value: f64) -> Option<BollingerOutput> {
        self.buffer.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
        if self.buffer.len() > self.period {
            let old = self.buffer.pop_front().unwrap();
            self.sum -= old;
            self.sum_sq -= old * old;
        }
        if self.buffer.len() == self.period {
            let mean = self.sum / self.period as f64;
            let variance = self.sum_sq / self.period as f64 - mean * mean;
            let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
            let upper = mean + self.num_std * std_dev;
            let lower = mean - self.num_std * std_dev;
            let bandwidth = if mean.abs() > 1e-15 { (upper - lower) / mean } else { 0.0 };
            let percent_b = if (upper - lower).abs() > 1e-15 {
                (value - lower) / (upper - lower)
            } else {
                0.5
            };
            Some(BollingerOutput { upper, middle: mean, lower, percent_b, bandwidth })
        } else {
            None
        }
    }

    pub fn calculate(data: &[f64], period: usize, num_std: f64) -> Vec<Option<BollingerOutput>> {
        let mut b = Self::new(period, num_std);
        data.iter().map(|&v| b.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Average True Range (ATR)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Atr {
    period: usize,
    prev_close: Option<f64>,
    count: usize,
    sum: f64,
    current_atr: Option<f64>,
}

impl Atr {
    pub fn new(period: usize) -> Self {
        Self { period, prev_close: None, count: 0, sum: 0.0, current_atr: None }
    }

    pub fn true_range(high: f64, low: f64, prev_close: Option<f64>) -> f64 {
        match prev_close {
            Some(pc) => {
                let hl = high - low;
                let hpc = (high - pc).abs();
                let lpc = (low - pc).abs();
                hl.max(hpc).max(lpc)
            }
            None => high - low,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = Self::true_range(high, low, self.prev_close);
        self.prev_close = Some(close);
        self.count += 1;
        if self.current_atr.is_none() {
            self.sum += tr;
            if self.count >= self.period {
                let atr = self.sum / self.period as f64;
                self.current_atr = Some(atr);
                return Some(atr);
            }
            return None;
        }
        let prev_atr = self.current_atr.unwrap();
        let atr = (prev_atr * (self.period as f64 - 1.0) + tr) / self.period as f64;
        self.current_atr = Some(atr);
        Some(atr)
    }

    pub fn current(&self) -> Option<f64> {
        self.current_atr
    }

    pub fn reset(&mut self) {
        self.prev_close = None;
        self.count = 0;
        self.sum = 0.0;
        self.current_atr = None;
    }
}

// ---------------------------------------------------------------------------
// ADX (Average Directional Index) with +DI / -DI
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AdxOutput {
    pub adx: f64,
    pub plus_di: f64,
    pub minus_di: f64,
}

#[derive(Debug, Clone)]
pub struct Adx {
    period: usize,
    prev_high: Option<f64>,
    prev_low: Option<f64>,
    prev_close: Option<f64>,
    smoothed_plus_dm: f64,
    smoothed_minus_dm: f64,
    smoothed_tr: f64,
    dx_buffer: VecDeque<f64>,
    adx: Option<f64>,
    count: usize,
    initialized: bool,
    plus_dm_sum: f64,
    minus_dm_sum: f64,
    tr_sum: f64,
}

impl Adx {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prev_high: None,
            prev_low: None,
            prev_close: None,
            smoothed_plus_dm: 0.0,
            smoothed_minus_dm: 0.0,
            smoothed_tr: 0.0,
            dx_buffer: VecDeque::with_capacity(period + 1),
            adx: None,
            count: 0,
            initialized: false,
            plus_dm_sum: 0.0,
            minus_dm_sum: 0.0,
            tr_sum: 0.0,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<AdxOutput> {
        if self.prev_high.is_none() {
            self.prev_high = Some(high);
            self.prev_low = Some(low);
            self.prev_close = Some(close);
            return None;
        }
        let ph = self.prev_high.unwrap();
        let pl = self.prev_low.unwrap();
        let pc = self.prev_close.unwrap();

        let up_move = high - ph;
        let down_move = pl - low;
        let plus_dm = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
        let minus_dm = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };

        let tr = Atr::true_range(high, low, Some(pc));

        self.prev_high = Some(high);
        self.prev_low = Some(low);
        self.prev_close = Some(close);
        self.count += 1;

        if !self.initialized {
            self.plus_dm_sum += plus_dm;
            self.minus_dm_sum += minus_dm;
            self.tr_sum += tr;
            if self.count >= self.period {
                self.smoothed_plus_dm = self.plus_dm_sum;
                self.smoothed_minus_dm = self.minus_dm_sum;
                self.smoothed_tr = self.tr_sum;
                self.initialized = true;
            } else {
                return None;
            }
        } else {
            self.smoothed_plus_dm = self.smoothed_plus_dm - self.smoothed_plus_dm / self.period as f64 + plus_dm;
            self.smoothed_minus_dm = self.smoothed_minus_dm - self.smoothed_minus_dm / self.period as f64 + minus_dm;
            self.smoothed_tr = self.smoothed_tr - self.smoothed_tr / self.period as f64 + tr;
        }

        let plus_di = if self.smoothed_tr > 1e-15 {
            100.0 * self.smoothed_plus_dm / self.smoothed_tr
        } else {
            0.0
        };
        let minus_di = if self.smoothed_tr > 1e-15 {
            100.0 * self.smoothed_minus_dm / self.smoothed_tr
        } else {
            0.0
        };

        let di_sum = plus_di + minus_di;
        let dx = if di_sum > 1e-15 {
            100.0 * (plus_di - minus_di).abs() / di_sum
        } else {
            0.0
        };

        self.dx_buffer.push_back(dx);
        if self.dx_buffer.len() > self.period {
            self.dx_buffer.pop_front();
        }

        if self.adx.is_none() {
            if self.dx_buffer.len() >= self.period {
                let avg = self.dx_buffer.iter().sum::<f64>() / self.period as f64;
                self.adx = Some(avg);
                return Some(AdxOutput { adx: avg, plus_di, minus_di });
            }
            return None;
        }

        let prev_adx = self.adx.unwrap();
        let new_adx = (prev_adx * (self.period as f64 - 1.0) + dx) / self.period as f64;
        self.adx = Some(new_adx);
        Some(AdxOutput { adx: new_adx, plus_di, minus_di })
    }

    pub fn current(&self) -> Option<f64> {
        self.adx
    }

    pub fn reset(&mut self) {
        self.prev_high = None;
        self.prev_low = None;
        self.prev_close = None;
        self.smoothed_plus_dm = 0.0;
        self.smoothed_minus_dm = 0.0;
        self.smoothed_tr = 0.0;
        self.dx_buffer.clear();
        self.adx = None;
        self.count = 0;
        self.initialized = false;
        self.plus_dm_sum = 0.0;
        self.minus_dm_sum = 0.0;
        self.tr_sum = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Commodity Channel Index (CCI)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Cci {
    period: usize,
    buffer: VecDeque<f64>,
    constant: f64,
}

impl Cci {
    pub fn new(period: usize) -> Self {
        Self { period, buffer: VecDeque::with_capacity(period + 1), constant: 0.015 }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tp = (high + low + close) / 3.0;
        self.buffer.push_back(tp);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() == self.period {
            let mean = self.buffer.iter().sum::<f64>() / self.period as f64;
            let mad = self.buffer.iter().map(|&v| (v - mean).abs()).sum::<f64>() / self.period as f64;
            if mad.abs() > 1e-15 {
                Some((tp - mean) / (self.constant * mad))
            } else {
                Some(0.0)
            }
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Williams %R
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct WilliamsR {
    period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
}

impl WilliamsR {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            highs: VecDeque::with_capacity(period + 1),
            lows: VecDeque::with_capacity(period + 1),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.highs.push_back(high);
        self.lows.push_back(low);
        if self.highs.len() > self.period {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        if self.highs.len() == self.period {
            let hh = self.highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ll = self.lows.iter().cloned().fold(f64::INFINITY, f64::min);
            if (hh - ll).abs() > 1e-15 {
                Some(-100.0 * (hh - close) / (hh - ll))
            } else {
                Some(0.0)
            }
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
    }
}

// ---------------------------------------------------------------------------
// Money Flow Index (MFI)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Mfi {
    period: usize,
    prev_tp: Option<f64>,
    pos_mf: VecDeque<f64>,
    neg_mf: VecDeque<f64>,
}

impl Mfi {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prev_tp: None,
            pos_mf: VecDeque::with_capacity(period + 1),
            neg_mf: VecDeque::with_capacity(period + 1),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        let tp = (high + low + close) / 3.0;
        let raw_mf = tp * volume;
        if let Some(prev) = self.prev_tp {
            if tp > prev {
                self.pos_mf.push_back(raw_mf);
                self.neg_mf.push_back(0.0);
            } else {
                self.pos_mf.push_back(0.0);
                self.neg_mf.push_back(raw_mf);
            }
            if self.pos_mf.len() > self.period {
                self.pos_mf.pop_front();
                self.neg_mf.pop_front();
            }
            self.prev_tp = Some(tp);
            if self.pos_mf.len() == self.period {
                let pos_sum: f64 = self.pos_mf.iter().sum();
                let neg_sum: f64 = self.neg_mf.iter().sum();
                if neg_sum.abs() > 1e-15 {
                    let mfr = pos_sum / neg_sum;
                    Some(100.0 - 100.0 / (1.0 + mfr))
                } else {
                    Some(100.0)
                }
            } else {
                None
            }
        } else {
            self.prev_tp = Some(tp);
            None
        }
    }

    pub fn reset(&mut self) {
        self.prev_tp = None;
        self.pos_mf.clear();
        self.neg_mf.clear();
    }
}

// ---------------------------------------------------------------------------
// On-Balance Volume (OBV)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Obv {
    prev_close: Option<f64>,
    obv: f64,
}

impl Obv {
    pub fn new() -> Self {
        Self { prev_close: None, obv: 0.0 }
    }

    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        if let Some(prev) = self.prev_close {
            if close > prev {
                self.obv += volume;
            } else if close < prev {
                self.obv -= volume;
            }
        }
        self.prev_close = Some(close);
        self.obv
    }

    pub fn current(&self) -> f64 {
        self.obv
    }

    pub fn calculate(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
        let mut o = Self::new();
        closes.iter().zip(volumes.iter()).map(|(&c, &v)| o.update(c, v)).collect()
    }

    pub fn reset(&mut self) {
        self.prev_close = None;
        self.obv = 0.0;
    }
}

// ---------------------------------------------------------------------------
// VWAP (Volume Weighted Average Price)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Vwap {
    cum_pv: f64,
    cum_vol: f64,
    cum_pv2: f64,
    count: usize,
}

impl Vwap {
    pub fn new() -> Self {
        Self { cum_pv: 0.0, cum_vol: 0.0, cum_pv2: 0.0, count: 0 }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let tp = (high + low + close) / 3.0;
        self.cum_pv += tp * volume;
        self.cum_vol += volume;
        self.cum_pv2 += tp * tp * volume;
        self.count += 1;
        if self.cum_vol.abs() > 1e-15 {
            self.cum_pv / self.cum_vol
        } else {
            tp
        }
    }

    pub fn current(&self) -> f64 {
        if self.cum_vol.abs() > 1e-15 {
            self.cum_pv / self.cum_vol
        } else {
            0.0
        }
    }

    pub fn std_dev(&self) -> f64 {
        if self.cum_vol.abs() > 1e-15 {
            let mean = self.cum_pv / self.cum_vol;
            let var = self.cum_pv2 / self.cum_vol - mean * mean;
            if var > 0.0 { var.sqrt() } else { 0.0 }
        } else {
            0.0
        }
    }

    pub fn upper_band(&self, mult: f64) -> f64 {
        self.current() + mult * self.std_dev()
    }

    pub fn lower_band(&self, mult: f64) -> f64 {
        self.current() - mult * self.std_dev()
    }

    pub fn reset(&mut self) {
        self.cum_pv = 0.0;
        self.cum_vol = 0.0;
        self.cum_pv2 = 0.0;
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// Ichimoku Cloud
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct IchimokuOutput {
    pub tenkan_sen: f64,
    pub kijun_sen: f64,
    pub senkou_span_a: f64,
    pub senkou_span_b: f64,
    pub chikou_span: f64,
}

#[derive(Debug, Clone)]
pub struct Ichimoku {
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
    displacement: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    closes: VecDeque<f64>,
}

impl Ichimoku {
    pub fn new(tenkan: usize, kijun: usize, senkou_b: usize, displacement: usize) -> Self {
        let max_period = tenkan.max(kijun).max(senkou_b);
        Self {
            tenkan_period: tenkan,
            kijun_period: kijun,
            senkou_b_period: senkou_b,
            displacement,
            highs: VecDeque::with_capacity(max_period + 1),
            lows: VecDeque::with_capacity(max_period + 1),
            closes: VecDeque::with_capacity(displacement + 1),
        }
    }

    pub fn default_params() -> Self {
        Self::new(9, 26, 52, 26)
    }

    fn highest(deque: &VecDeque<f64>, period: usize) -> f64 {
        let start = if deque.len() > period { deque.len() - period } else { 0 };
        deque.iter().skip(start).cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    fn lowest(deque: &VecDeque<f64>, period: usize) -> f64 {
        let start = if deque.len() > period { deque.len() - period } else { 0 };
        deque.iter().skip(start).cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<IchimokuOutput> {
        self.highs.push_back(high);
        self.lows.push_back(low);
        self.closes.push_back(close);
        let max_needed = self.senkou_b_period.max(self.displacement);
        if self.highs.len() > max_needed + 1 {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        if self.closes.len() > self.displacement + 1 {
            self.closes.pop_front();
        }
        if self.highs.len() >= self.senkou_b_period {
            let tenkan_high = Self::highest(&self.highs, self.tenkan_period);
            let tenkan_low = Self::lowest(&self.lows, self.tenkan_period);
            let tenkan_sen = (tenkan_high + tenkan_low) / 2.0;

            let kijun_high = Self::highest(&self.highs, self.kijun_period);
            let kijun_low = Self::lowest(&self.lows, self.kijun_period);
            let kijun_sen = (kijun_high + kijun_low) / 2.0;

            let senkou_span_a = (tenkan_sen + kijun_sen) / 2.0;

            let senkou_b_high = Self::highest(&self.highs, self.senkou_b_period);
            let senkou_b_low = Self::lowest(&self.lows, self.senkou_b_period);
            let senkou_span_b = (senkou_b_high + senkou_b_low) / 2.0;

            let chikou_span = close;

            Some(IchimokuOutput {
                tenkan_sen,
                kijun_sen,
                senkou_span_a,
                senkou_span_b,
                chikou_span,
            })
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
        self.closes.clear();
    }
}

// ---------------------------------------------------------------------------
// Parabolic SAR
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ParabolicSar {
    af_start: f64,
    af_step: f64,
    af_max: f64,
    af: f64,
    sar: f64,
    ep: f64,
    is_long: bool,
    initialized: bool,
    prev_high: f64,
    prev_low: f64,
}

impl ParabolicSar {
    pub fn new(af_start: f64, af_step: f64, af_max: f64) -> Self {
        Self {
            af_start,
            af_step,
            af_max,
            af: af_start,
            sar: 0.0,
            ep: 0.0,
            is_long: true,
            initialized: false,
            prev_high: 0.0,
            prev_low: 0.0,
        }
    }

    pub fn default_params() -> Self {
        Self::new(0.02, 0.02, 0.2)
    }

    pub fn update(&mut self, high: f64, low: f64) -> f64 {
        if !self.initialized {
            self.initialized = true;
            self.sar = low;
            self.ep = high;
            self.is_long = true;
            self.prev_high = high;
            self.prev_low = low;
            return self.sar;
        }

        let mut sar = self.sar + self.af * (self.ep - self.sar);

        if self.is_long {
            sar = sar.min(self.prev_low).min(low);
            if high > self.ep {
                self.ep = high;
                self.af = (self.af + self.af_step).min(self.af_max);
            }
            if low < sar {
                self.is_long = false;
                sar = self.ep;
                self.ep = low;
                self.af = self.af_start;
            }
        } else {
            sar = sar.max(self.prev_high).max(high);
            if low < self.ep {
                self.ep = low;
                self.af = (self.af + self.af_step).min(self.af_max);
            }
            if high > sar {
                self.is_long = true;
                sar = self.ep;
                self.ep = high;
                self.af = self.af_start;
            }
        }

        self.sar = sar;
        self.prev_high = high;
        self.prev_low = low;
        sar
    }

    pub fn is_bullish(&self) -> bool {
        self.is_long
    }

    pub fn reset(&mut self) {
        self.af = self.af_start;
        self.sar = 0.0;
        self.ep = 0.0;
        self.is_long = true;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Donchian Channel
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct DonchianOutput {
    pub upper: f64,
    pub lower: f64,
    pub middle: f64,
}

#[derive(Debug, Clone)]
pub struct DonchianChannel {
    period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
}

impl DonchianChannel {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            highs: VecDeque::with_capacity(period + 1),
            lows: VecDeque::with_capacity(period + 1),
        }
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<DonchianOutput> {
        self.highs.push_back(high);
        self.lows.push_back(low);
        if self.highs.len() > self.period {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        if self.highs.len() == self.period {
            let upper = self.highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lower = self.lows.iter().cloned().fold(f64::INFINITY, f64::min);
            Some(DonchianOutput { upper, lower, middle: (upper + lower) / 2.0 })
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
    }
}

// ---------------------------------------------------------------------------
// Keltner Channel
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct KeltnerOutput {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

#[derive(Debug, Clone)]
pub struct KeltnerChannel {
    ema: Ema,
    atr: Atr,
    multiplier: f64,
}

impl KeltnerChannel {
    pub fn new(ema_period: usize, atr_period: usize, multiplier: f64) -> Self {
        Self {
            ema: Ema::new(ema_period),
            atr: Atr::new(atr_period),
            multiplier,
        }
    }

    pub fn default_params() -> Self {
        Self::new(20, 10, 1.5)
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<KeltnerOutput> {
        let ema_val = self.ema.update(close);
        let atr_val = self.atr.update(high, low, close);
        match (ema_val, atr_val) {
            (Some(e), Some(a)) => Some(KeltnerOutput {
                upper: e + self.multiplier * a,
                middle: e,
                lower: e - self.multiplier * a,
            }),
            _ => None,
        }
    }

    pub fn reset(&mut self) {
        self.ema.reset();
        self.atr.reset();
    }
}

// ---------------------------------------------------------------------------
// SuperTrend
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct SuperTrendOutput {
    pub value: f64,
    pub direction: i32,
}

#[derive(Debug, Clone)]
pub struct SuperTrend {
    atr: Atr,
    multiplier: f64,
    prev_upper: f64,
    prev_lower: f64,
    prev_close: f64,
    prev_supertrend: f64,
    direction: i32,
    initialized: bool,
}

impl SuperTrend {
    pub fn new(period: usize, multiplier: f64) -> Self {
        Self {
            atr: Atr::new(period),
            multiplier,
            prev_upper: 0.0,
            prev_lower: 0.0,
            prev_close: 0.0,
            prev_supertrend: 0.0,
            direction: 1,
            initialized: false,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<SuperTrendOutput> {
        let atr_val = self.atr.update(high, low, close);
        if let Some(atr) = atr_val {
            let mid = (high + low) / 2.0;
            let mut upper = mid + self.multiplier * atr;
            let mut lower = mid - self.multiplier * atr;

            if self.initialized {
                if lower > self.prev_lower || self.prev_close < self.prev_lower {
                    // keep lower
                } else {
                    lower = self.prev_lower;
                }
                if upper < self.prev_upper || self.prev_close > self.prev_upper {
                    // keep upper
                } else {
                    upper = self.prev_upper;
                }
            }

            let supertrend;
            if !self.initialized {
                supertrend = lower;
                self.direction = 1;
                self.initialized = true;
            } else if self.prev_supertrend == self.prev_upper {
                if close > upper {
                    supertrend = lower;
                    self.direction = 1;
                } else {
                    supertrend = upper;
                    self.direction = -1;
                }
            } else {
                if close < lower {
                    supertrend = upper;
                    self.direction = -1;
                } else {
                    supertrend = lower;
                    self.direction = 1;
                }
            }

            self.prev_upper = upper;
            self.prev_lower = lower;
            self.prev_close = close;
            self.prev_supertrend = supertrend;
            Some(SuperTrendOutput { value: supertrend, direction: self.direction })
        } else {
            self.prev_close = close;
            None
        }
    }

    pub fn reset(&mut self) {
        self.atr.reset();
        self.prev_upper = 0.0;
        self.prev_lower = 0.0;
        self.prev_close = 0.0;
        self.prev_supertrend = 0.0;
        self.direction = 1;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Aroon (Up / Down / Oscillator)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AroonOutput {
    pub aroon_up: f64,
    pub aroon_down: f64,
    pub oscillator: f64,
}

#[derive(Debug, Clone)]
pub struct Aroon {
    period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
}

impl Aroon {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            highs: VecDeque::with_capacity(period + 2),
            lows: VecDeque::with_capacity(period + 2),
        }
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<AroonOutput> {
        self.highs.push_back(high);
        self.lows.push_back(low);
        if self.highs.len() > self.period + 1 {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        if self.highs.len() == self.period + 1 {
            let mut max_idx = 0;
            let mut max_val = f64::NEG_INFINITY;
            let mut min_idx = 0;
            let mut min_val = f64::INFINITY;
            for (i, (&h, &l)) in self.highs.iter().zip(self.lows.iter()).enumerate() {
                if h >= max_val {
                    max_val = h;
                    max_idx = i;
                }
                if l <= min_val {
                    min_val = l;
                    min_idx = i;
                }
            }
            let aroon_up = 100.0 * max_idx as f64 / self.period as f64;
            let aroon_down = 100.0 * min_idx as f64 / self.period as f64;
            Some(AroonOutput {
                aroon_up,
                aroon_down,
                oscillator: aroon_up - aroon_down,
            })
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
    }
}

// ---------------------------------------------------------------------------
// Chaikin Money Flow (CMF)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ChaikinMoneyFlow {
    period: usize,
    mfv_buffer: VecDeque<f64>,
    vol_buffer: VecDeque<f64>,
}

impl ChaikinMoneyFlow {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            mfv_buffer: VecDeque::with_capacity(period + 1),
            vol_buffer: VecDeque::with_capacity(period + 1),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        let mfm = if (high - low).abs() > 1e-15 {
            ((close - low) - (high - close)) / (high - low)
        } else {
            0.0
        };
        let mfv = mfm * volume;
        self.mfv_buffer.push_back(mfv);
        self.vol_buffer.push_back(volume);
        if self.mfv_buffer.len() > self.period {
            self.mfv_buffer.pop_front();
            self.vol_buffer.pop_front();
        }
        if self.mfv_buffer.len() == self.period {
            let vol_sum: f64 = self.vol_buffer.iter().sum();
            if vol_sum.abs() > 1e-15 {
                Some(self.mfv_buffer.iter().sum::<f64>() / vol_sum)
            } else {
                Some(0.0)
            }
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.mfv_buffer.clear();
        self.vol_buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Accumulation/Distribution Line
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AccumulationDistribution {
    ad: f64,
}

impl AccumulationDistribution {
    pub fn new() -> Self {
        Self { ad: 0.0 }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let mfm = if (high - low).abs() > 1e-15 {
            ((close - low) - (high - close)) / (high - low)
        } else {
            0.0
        };
        self.ad += mfm * volume;
        self.ad
    }

    pub fn current(&self) -> f64 {
        self.ad
    }

    pub fn reset(&mut self) {
        self.ad = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Force Index
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ForceIndex {
    ema: Ema,
    prev_close: Option<f64>,
}

impl ForceIndex {
    pub fn new(period: usize) -> Self {
        Self { ema: Ema::new(period), prev_close: None }
    }

    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        if let Some(prev) = self.prev_close {
            let fi = (close - prev) * volume;
            self.prev_close = Some(close);
            self.ema.update(fi)
        } else {
            self.prev_close = Some(close);
            None
        }
    }

    pub fn reset(&mut self) {
        self.ema.reset();
        self.prev_close = None;
    }
}

// ---------------------------------------------------------------------------
// Elder Ray (Bull Power / Bear Power)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ElderRayOutput {
    pub bull_power: f64,
    pub bear_power: f64,
}

#[derive(Debug, Clone)]
pub struct ElderRay {
    ema: Ema,
}

impl ElderRay {
    pub fn new(period: usize) -> Self {
        Self { ema: Ema::new(period) }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<ElderRayOutput> {
        let ema_val = self.ema.update(close);
        ema_val.map(|e| ElderRayOutput {
            bull_power: high - e,
            bear_power: low - e,
        })
    }

    pub fn reset(&mut self) {
        self.ema.reset();
    }
}

// ---------------------------------------------------------------------------
// TRIX
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Trix {
    ema1: Ema,
    ema2: Ema,
    ema3: Ema,
    prev_triple: Option<f64>,
}

impl Trix {
    pub fn new(period: usize) -> Self {
        Self {
            ema1: Ema::new(period),
            ema2: Ema::new(period),
            ema3: Ema::new(period),
            prev_triple: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        let e1 = self.ema1.update(value);
        if let Some(e1v) = e1 {
            let e2 = self.ema2.update(e1v);
            if let Some(e2v) = e2 {
                let e3 = self.ema3.update(e2v);
                if let Some(e3v) = e3 {
                    let result = if let Some(prev) = self.prev_triple {
                        if prev.abs() > 1e-15 {
                            Some(100.0 * (e3v - prev) / prev)
                        } else {
                            Some(0.0)
                        }
                    } else {
                        None
                    };
                    self.prev_triple = Some(e3v);
                    return result;
                }
            }
        }
        None
    }

    pub fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
        self.ema3.reset();
        self.prev_triple = None;
    }
}

// ---------------------------------------------------------------------------
// Ultimate Oscillator
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct UltimateOscillator {
    period1: usize,
    period2: usize,
    period3: usize,
    prev_close: Option<f64>,
    bp_buffer: VecDeque<f64>,
    tr_buffer: VecDeque<f64>,
}

impl UltimateOscillator {
    pub fn new(period1: usize, period2: usize, period3: usize) -> Self {
        let max_p = period1.max(period2).max(period3);
        Self {
            period1,
            period2,
            period3,
            prev_close: None,
            bp_buffer: VecDeque::with_capacity(max_p + 1),
            tr_buffer: VecDeque::with_capacity(max_p + 1),
        }
    }

    pub fn default_params() -> Self {
        Self::new(7, 14, 28)
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if let Some(pc) = self.prev_close {
            let tl = low.min(pc);
            let bp = close - tl;
            let tr = Atr::true_range(high, low, Some(pc));
            self.bp_buffer.push_back(bp);
            self.tr_buffer.push_back(tr);
            let max_p = self.period1.max(self.period2).max(self.period3);
            if self.bp_buffer.len() > max_p {
                self.bp_buffer.pop_front();
                self.tr_buffer.pop_front();
            }
            self.prev_close = Some(close);
            if self.bp_buffer.len() >= max_p {
                let len = self.bp_buffer.len();
                let sum = |period: usize| -> (f64, f64) {
                    let start = len - period;
                    let bp_s: f64 = self.bp_buffer.iter().skip(start).sum();
                    let tr_s: f64 = self.tr_buffer.iter().skip(start).sum();
                    (bp_s, tr_s)
                };
                let (bp1, tr1) = sum(self.period1);
                let (bp2, tr2) = sum(self.period2);
                let (bp3, tr3) = sum(self.period3);
                let avg1 = if tr1.abs() > 1e-15 { bp1 / tr1 } else { 0.0 };
                let avg2 = if tr2.abs() > 1e-15 { bp2 / tr2 } else { 0.0 };
                let avg3 = if tr3.abs() > 1e-15 { bp3 / tr3 } else { 0.0 };
                Some(100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0)
            } else {
                None
            }
        } else {
            self.prev_close = Some(close);
            None
        }
    }

    pub fn reset(&mut self) {
        self.prev_close = None;
        self.bp_buffer.clear();
        self.tr_buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Rate of Change (ROC)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Roc {
    period: usize,
    buffer: VecDeque<f64>,
}

impl Roc {
    pub fn new(period: usize) -> Self {
        Self { period, buffer: VecDeque::with_capacity(period + 2) }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }
        if self.buffer.len() == self.period + 1 {
            let prev = self.buffer[0];
            if prev.abs() > 1e-15 {
                Some(100.0 * (value - prev) / prev)
            } else {
                Some(0.0)
            }
        } else {
            None
        }
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut r = Self::new(period);
        data.iter().map(|&v| r.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Momentum
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Momentum {
    period: usize,
    buffer: VecDeque<f64>,
}

impl Momentum {
    pub fn new(period: usize) -> Self {
        Self { period, buffer: VecDeque::with_capacity(period + 2) }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }
        if self.buffer.len() == self.period + 1 {
            Some(value - self.buffer[0])
        } else {
            None
        }
    }

    pub fn calculate(data: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut m = Self::new(period);
        data.iter().map(|&v| m.update(v)).collect()
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Percentage Price Oscillator (PPO)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct PpoOutput {
    pub ppo: f64,
    pub signal: f64,
    pub histogram: f64,
}

#[derive(Debug, Clone)]
pub struct Ppo {
    fast_ema: Ema,
    slow_ema: Ema,
    signal_ema: Ema,
}

impl Ppo {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast_ema: Ema::new(fast),
            slow_ema: Ema::new(slow),
            signal_ema: Ema::new(signal),
        }
    }

    pub fn default_params() -> Self {
        Self::new(12, 26, 9)
    }

    pub fn update(&mut self, value: f64) -> Option<PpoOutput> {
        let fast = self.fast_ema.update(value);
        let slow = self.slow_ema.update(value);
        match (fast, slow) {
            (Some(f), Some(s)) => {
                let ppo = if s.abs() > 1e-15 { 100.0 * (f - s) / s } else { 0.0 };
                let sig = self.signal_ema.update(ppo);
                let signal = sig.unwrap_or(ppo);
                Some(PpoOutput { ppo, signal, histogram: ppo - signal })
            }
            _ => None,
        }
    }

    pub fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_ema.reset();
    }
}

// ---------------------------------------------------------------------------
// Detrended Price Oscillator (DPO)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Dpo {
    period: usize,
    sma: Sma,
    buffer: VecDeque<f64>,
    shift: usize,
}

impl Dpo {
    pub fn new(period: usize) -> Self {
        let shift = period / 2 + 1;
        Self {
            period,
            sma: Sma::new(period),
            buffer: VecDeque::with_capacity(shift + 2),
            shift,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        let sma_val = self.sma.update(value);
        if self.buffer.len() > self.shift + 1 {
            self.buffer.pop_front();
        }
        if let Some(sma) = sma_val {
            if self.buffer.len() > self.shift {
                let shifted = self.buffer[0];
                return Some(shifted - sma);
            }
        }
        None
    }

    pub fn reset(&mut self) {
        self.sma.reset();
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Mass Index
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MassIndex {
    period: usize,
    sum_period: usize,
    ema1: Ema,
    ema2: Ema,
    ratio_buffer: VecDeque<f64>,
}

impl MassIndex {
    pub fn new(ema_period: usize, sum_period: usize) -> Self {
        Self {
            period: ema_period,
            sum_period,
            ema1: Ema::new(ema_period),
            ema2: Ema::new(ema_period),
            ratio_buffer: VecDeque::with_capacity(sum_period + 1),
        }
    }

    pub fn default_params() -> Self {
        Self::new(9, 25)
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        let range = high - low;
        let e1 = self.ema1.update(range);
        if let Some(e1v) = e1 {
            let e2 = self.ema2.update(e1v);
            if let Some(e2v) = e2 {
                let ratio = if e2v.abs() > 1e-15 { e1v / e2v } else { 1.0 };
                self.ratio_buffer.push_back(ratio);
                if self.ratio_buffer.len() > self.sum_period {
                    self.ratio_buffer.pop_front();
                }
                if self.ratio_buffer.len() == self.sum_period {
                    return Some(self.ratio_buffer.iter().sum());
                }
            }
        }
        None
    }

    pub fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
        self.ratio_buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Vortex Indicator
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct VortexOutput {
    pub vi_plus: f64,
    pub vi_minus: f64,
}

#[derive(Debug, Clone)]
pub struct VortexIndicator {
    period: usize,
    prev_high: Option<f64>,
    prev_low: Option<f64>,
    prev_close: Option<f64>,
    vm_plus_buf: VecDeque<f64>,
    vm_minus_buf: VecDeque<f64>,
    tr_buf: VecDeque<f64>,
}

impl VortexIndicator {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prev_high: None,
            prev_low: None,
            prev_close: None,
            vm_plus_buf: VecDeque::with_capacity(period + 1),
            vm_minus_buf: VecDeque::with_capacity(period + 1),
            tr_buf: VecDeque::with_capacity(period + 1),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<VortexOutput> {
        if let (Some(ph), Some(pl), Some(pc)) = (self.prev_high, self.prev_low, self.prev_close) {
            let vm_plus = (high - pl).abs();
            let vm_minus = (low - ph).abs();
            let tr = Atr::true_range(high, low, Some(pc));
            self.vm_plus_buf.push_back(vm_plus);
            self.vm_minus_buf.push_back(vm_minus);
            self.tr_buf.push_back(tr);
            if self.vm_plus_buf.len() > self.period {
                self.vm_plus_buf.pop_front();
                self.vm_minus_buf.pop_front();
                self.tr_buf.pop_front();
            }
            self.prev_high = Some(high);
            self.prev_low = Some(low);
            self.prev_close = Some(close);
            if self.vm_plus_buf.len() == self.period {
                let tr_sum: f64 = self.tr_buf.iter().sum();
                if tr_sum.abs() > 1e-15 {
                    return Some(VortexOutput {
                        vi_plus: self.vm_plus_buf.iter().sum::<f64>() / tr_sum,
                        vi_minus: self.vm_minus_buf.iter().sum::<f64>() / tr_sum,
                    });
                }
            }
            None
        } else {
            self.prev_high = Some(high);
            self.prev_low = Some(low);
            self.prev_close = Some(close);
            None
        }
    }

    pub fn reset(&mut self) {
        self.prev_high = None;
        self.prev_low = None;
        self.prev_close = None;
        self.vm_plus_buf.clear();
        self.vm_minus_buf.clear();
        self.tr_buf.clear();
    }
}

// ---------------------------------------------------------------------------
// Know Sure Thing (KST)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct KstOutput {
    pub kst: f64,
    pub signal: f64,
}

#[derive(Debug, Clone)]
pub struct KnowSureThing {
    roc1: Roc,
    roc2: Roc,
    roc3: Roc,
    roc4: Roc,
    sma1: Sma,
    sma2: Sma,
    sma3: Sma,
    sma4: Sma,
    signal_sma: Sma,
}

impl KnowSureThing {
    pub fn new(
        roc1_p: usize, roc2_p: usize, roc3_p: usize, roc4_p: usize,
        sma1_p: usize, sma2_p: usize, sma3_p: usize, sma4_p: usize,
        signal_p: usize,
    ) -> Self {
        Self {
            roc1: Roc::new(roc1_p),
            roc2: Roc::new(roc2_p),
            roc3: Roc::new(roc3_p),
            roc4: Roc::new(roc4_p),
            sma1: Sma::new(sma1_p),
            sma2: Sma::new(sma2_p),
            sma3: Sma::new(sma3_p),
            sma4: Sma::new(sma4_p),
            signal_sma: Sma::new(signal_p),
        }
    }

    pub fn default_params() -> Self {
        Self::new(10, 15, 20, 30, 10, 10, 10, 15, 9)
    }

    pub fn update(&mut self, value: f64) -> Option<KstOutput> {
        let r1 = self.roc1.update(value);
        let r2 = self.roc2.update(value);
        let r3 = self.roc3.update(value);
        let r4 = self.roc4.update(value);

        let s1 = r1.and_then(|v| self.sma1.update(v));
        let s2 = r2.and_then(|v| self.sma2.update(v));
        let s3 = r3.and_then(|v| self.sma3.update(v));
        let s4 = r4.and_then(|v| self.sma4.update(v));

        match (s1, s2, s3, s4) {
            (Some(v1), Some(v2), Some(v3), Some(v4)) => {
                let kst = v1 * 1.0 + v2 * 2.0 + v3 * 3.0 + v4 * 4.0;
                let sig = self.signal_sma.update(kst);
                Some(KstOutput { kst, signal: sig.unwrap_or(kst) })
            }
            _ => None,
        }
    }

    pub fn reset(&mut self) {
        self.roc1.reset();
        self.roc2.reset();
        self.roc3.reset();
        self.roc4.reset();
        self.sma1.reset();
        self.sma2.reset();
        self.sma3.reset();
        self.sma4.reset();
        self.signal_sma.reset();
    }
}

// ---------------------------------------------------------------------------
// Coppock Curve
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct CoppockCurve {
    roc_long: Roc,
    roc_short: Roc,
    wma: Wma,
}

impl CoppockCurve {
    pub fn new(wma_period: usize, long_roc: usize, short_roc: usize) -> Self {
        Self {
            roc_long: Roc::new(long_roc),
            roc_short: Roc::new(short_roc),
            wma: Wma::new(wma_period),
        }
    }

    pub fn default_params() -> Self {
        Self::new(10, 14, 11)
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        let rl = self.roc_long.update(value);
        let rs = self.roc_short.update(value);
        match (rl, rs) {
            (Some(l), Some(s)) => self.wma.update(l + s),
            _ => None,
        }
    }

    pub fn reset(&mut self) {
        self.roc_long.reset();
        self.roc_short.reset();
        self.wma.reset();
    }
}

// ---------------------------------------------------------------------------
// Connors RSI
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ConnorsRsi {
    rsi: Rsi,
    streak_rsi: Rsi,
    roc_period: usize,
    streak: i32,
    prev_close: Option<f64>,
    roc_buffer: VecDeque<f64>,
    count: usize,
}

impl ConnorsRsi {
    pub fn new(rsi_period: usize, streak_period: usize, rank_period: usize) -> Self {
        Self {
            rsi: Rsi::new(rsi_period),
            streak_rsi: Rsi::new(streak_period),
            roc_period: rank_period,
            streak: 0,
            prev_close: None,
            roc_buffer: VecDeque::with_capacity(rank_period + 2),
            count: 0,
        }
    }

    pub fn default_params() -> Self {
        Self::new(3, 2, 100)
    }

    pub fn update(&mut self, close: f64) -> Option<f64> {
        // Update streak
        if let Some(prev) = self.prev_close {
            if close > prev {
                self.streak = if self.streak > 0 { self.streak + 1 } else { 1 };
            } else if close < prev {
                self.streak = if self.streak < 0 { self.streak - 1 } else { -1 };
            } else {
                self.streak = 0;
            }
        }
        self.prev_close = Some(close);
        self.count += 1;

        // RSI of price
        let rsi_val = self.rsi.update(close);
        // RSI of streak
        let streak_rsi_val = self.streak_rsi.update(self.streak as f64);

        // Percent rank of ROC(1)
        if self.count >= 2 {
            // Simple 1-period ROC
            // We store closes and compute percent rank
            self.roc_buffer.push_back(close);
            if self.roc_buffer.len() > self.roc_period + 1 {
                self.roc_buffer.pop_front();
            }
        } else {
            self.roc_buffer.push_back(close);
        }

        let pct_rank = if self.roc_buffer.len() >= 3 {
            // Compute 1-period ROC for all available
            let len = self.roc_buffer.len();
            let current_roc = self.roc_buffer[len - 1] - self.roc_buffer[len - 2];
            let mut count_less = 0usize;
            let total = len - 2; // number of historical ROCs
            for i in 1..len - 1 {
                let roc = self.roc_buffer[i] - self.roc_buffer[i - 1];
                if roc < current_roc {
                    count_less += 1;
                }
            }
            if total > 0 {
                Some(100.0 * count_less as f64 / total as f64)
            } else {
                Some(50.0)
            }
        } else {
            None
        };

        match (rsi_val, streak_rsi_val, pct_rank) {
            (Some(r), Some(sr), Some(pr)) => Some((r + sr + pr) / 3.0),
            _ => None,
        }
    }

    pub fn reset(&mut self) {
        self.rsi.reset();
        self.streak_rsi.reset();
        self.streak = 0;
        self.prev_close = None;
        self.roc_buffer.clear();
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// Helper: batch calculation utilities
// ---------------------------------------------------------------------------
pub fn sma_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Sma::calculate(data, period)
}

pub fn ema_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Ema::calculate(data, period)
}

pub fn rsi_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Rsi::calculate(data, period)
}

pub fn macd_batch(data: &[f64]) -> Vec<Option<MacdOutput>> {
    Macd::calculate(data)
}

pub fn bollinger_batch(data: &[f64], period: usize, num_std: f64) -> Vec<Option<BollingerOutput>> {
    BollingerBands::calculate(data, period, num_std)
}

pub fn roc_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Roc::calculate(data, period)
}

pub fn momentum_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Momentum::calculate(data, period)
}

pub fn obv_batch(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    Obv::calculate(closes, volumes)
}

pub fn wma_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Wma::calculate(data, period)
}

pub fn dema_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Dema::calculate(data, period)
}

pub fn tema_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Tema::calculate(data, period)
}

pub fn hma_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Hma::calculate(data, period)
}

pub fn kama_batch(data: &[f64], period: usize) -> Vec<Option<f64>> {
    Kama::calculate(data, period)
}

pub fn stochastic_rsi_batch(data: &[f64], rsi_period: usize, stoch_period: usize) -> Vec<Option<(f64, f64)>> {
    StochasticRsi::calculate(data, rsi_period, stoch_period)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_basic() {
        let mut sma = Sma::new(3);
        assert!(sma.update(1.0).is_none());
        assert!(sma.update(2.0).is_none());
        let v = sma.update(3.0).unwrap();
        assert!((v - 2.0).abs() < 1e-10);
        let v = sma.update(4.0).unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_basic() {
        let mut ema = Ema::new(3);
        assert!(ema.update(1.0).is_none());
        assert!(ema.update(2.0).is_none());
        let v = ema.update(3.0).unwrap();
        assert!((v - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi_basic() {
        let mut rsi = Rsi::new(14);
        for i in 0..20 {
            let _ = rsi.update(i as f64);
        }
        let val = rsi.current().unwrap();
        assert!(val > 50.0 && val <= 100.0);
    }

    #[test]
    fn test_macd_basic() {
        let mut macd = Macd::default_periods();
        for i in 0..50 {
            let _ = macd.update(100.0 + (i as f64) * 0.5);
        }
    }

    #[test]
    fn test_bollinger_basic() {
        let mut bb = BollingerBands::default_params();
        for i in 0..25 {
            let _ = bb.update(100.0 + (i as f64) * 0.1);
        }
    }

    #[test]
    fn test_obv_basic() {
        let mut obv = Obv::new();
        obv.update(10.0, 100.0);
        let v = obv.update(11.0, 200.0);
        assert!((v - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_atr_basic() {
        let mut atr = Atr::new(5);
        for i in 0..10 {
            let h = 100.0 + i as f64;
            let l = 99.0 + i as f64;
            let c = 99.5 + i as f64;
            let _ = atr.update(h, l, c);
        }
        assert!(atr.current().is_some());
    }

    #[test]
    fn test_williams_r() {
        let mut wr = WilliamsR::new(14);
        for i in 0..20 {
            let h = 100.0 + i as f64;
            let l = 99.0 + i as f64;
            let c = 99.5 + i as f64;
            let _ = wr.update(h, l, c);
        }
    }

    #[test]
    fn test_vwap_basic() {
        let mut vwap = Vwap::new();
        vwap.update(101.0, 99.0, 100.0, 1000.0);
        let v = vwap.current();
        assert!((v - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_donchian() {
        let mut dc = DonchianChannel::new(5);
        for i in 0..5 {
            let _ = dc.update(100.0 + i as f64, 99.0 + i as f64);
        }
    }

    #[test]
    fn test_supertrend() {
        let mut st = SuperTrend::new(10, 3.0);
        for i in 0..20 {
            let h = 100.0 + (i as f64) * 0.5;
            let l = 99.0 + (i as f64) * 0.5;
            let c = 99.5 + (i as f64) * 0.5;
            let _ = st.update(h, l, c);
        }
    }

    #[test]
    fn test_aroon() {
        let mut ar = Aroon::new(14);
        for i in 0..20 {
            let _ = ar.update(100.0 + i as f64, 99.0 + i as f64);
        }
    }

    #[test]
    fn test_roc() {
        let mut roc = Roc::new(5);
        for i in 0..10 {
            let _ = roc.update(100.0 + i as f64);
        }
    }

    #[test]
    fn test_momentum() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = Momentum::calculate(&vals, 3);
        assert!(result[3].is_some());
        assert!((result[3].unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ppo() {
        let mut ppo = Ppo::default_params();
        for i in 0..50 {
            let _ = ppo.update(100.0 + (i as f64) * 0.3);
        }
    }

    #[test]
    fn test_trix() {
        let mut trix = Trix::new(15);
        for i in 0..100 {
            let _ = trix.update(100.0 + (i as f64).sin());
        }
    }

    #[test]
    fn test_kst() {
        let mut kst = KnowSureThing::default_params();
        for i in 0..100 {
            let _ = kst.update(100.0 + (i as f64) * 0.2);
        }
    }

    #[test]
    fn test_coppock() {
        let mut cc = CoppockCurve::default_params();
        for i in 0..50 {
            let _ = cc.update(100.0 + (i as f64) * 0.4);
        }
    }

    #[test]
    fn test_connors_rsi() {
        let mut crsi = ConnorsRsi::default_params();
        for i in 0..200 {
            let _ = crsi.update(100.0 + (i as f64).sin() * 5.0);
        }
    }
}
