// data.rs — Bar/tick data structures, CSV/binary reader, alignment, corp actions, missing data
use std::collections::HashMap;

/// OHLCV bar
#[derive(Clone, Debug)]
pub struct Bar {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: Option<f64>,
    pub num_trades: Option<u64>,
}

impl Bar {
    pub fn new(ts: u64, o: f64, h: f64, l: f64, c: f64, v: f64) -> Self {
        Self { timestamp: ts, open: o, high: h, low: l, close: c, volume: v, vwap: None, num_trades: None }
    }

    pub fn typical_price(&self) -> f64 { (self.high + self.low + self.close) / 3.0 }
    pub fn mid_price(&self) -> f64 { (self.high + self.low) / 2.0 }
    pub fn range(&self) -> f64 { self.high - self.low }
    pub fn body(&self) -> f64 { (self.close - self.open).abs() }
    pub fn is_green(&self) -> bool { self.close >= self.open }
    pub fn is_red(&self) -> bool { self.close < self.open }
    pub fn upper_shadow(&self) -> f64 { self.high - self.close.max(self.open) }
    pub fn lower_shadow(&self) -> f64 { self.close.min(self.open) - self.low }

    pub fn returns(&self) -> f64 {
        if self.open.abs() < 1e-15 { 0.0 } else { (self.close - self.open) / self.open }
    }

    pub fn log_returns(&self) -> f64 {
        if self.open <= 0.0 || self.close <= 0.0 { 0.0 } else { (self.close / self.open).ln() }
    }

    pub fn validate(&self) -> bool {
        self.high >= self.low && self.high >= self.open && self.high >= self.close
            && self.low <= self.open && self.low <= self.close && self.volume >= 0.0
    }
}

/// Tick data
#[derive(Clone, Debug)]
pub struct Tick {
    pub timestamp: u64,
    pub price: f64,
    pub size: f64,
    pub side: TickSide,
    pub exchange_id: u16,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TickSide {
    Buy,
    Sell,
    Unknown,
}

impl Tick {
    pub fn new(ts: u64, price: f64, size: f64, side: TickSide) -> Self {
        Self { timestamp: ts, price, size, side, exchange_id: 0 }
    }

    pub fn notional(&self) -> f64 { self.price * self.size }
}

/// Quote data (BBO)
#[derive(Clone, Debug)]
pub struct Quote {
    pub timestamp: u64,
    pub bid_price: f64,
    pub bid_size: f64,
    pub ask_price: f64,
    pub ask_size: f64,
}

impl Quote {
    pub fn mid(&self) -> f64 { (self.bid_price + self.ask_price) / 2.0 }
    pub fn spread(&self) -> f64 { self.ask_price - self.bid_price }
    pub fn spread_bps(&self) -> f64 { self.spread() / self.mid() * 10000.0 }
    pub fn weighted_mid(&self) -> f64 {
        let total = self.bid_size + self.ask_size;
        if total < 1e-15 { return self.mid(); }
        (self.bid_price * self.ask_size + self.ask_price * self.bid_size) / total
    }
}

/// Time series of bars for one asset
#[derive(Clone, Debug)]
pub struct BarSeries {
    pub symbol: String,
    pub bars: Vec<Bar>,
}

impl BarSeries {
    pub fn new(symbol: &str) -> Self { Self { symbol: symbol.to_string(), bars: Vec::new() } }

    pub fn push(&mut self, bar: Bar) { self.bars.push(bar); }
    pub fn len(&self) -> usize { self.bars.len() }
    pub fn is_empty(&self) -> bool { self.bars.is_empty() }

    pub fn close_prices(&self) -> Vec<f64> { self.bars.iter().map(|b| b.close).collect() }
    pub fn volumes(&self) -> Vec<f64> { self.bars.iter().map(|b| b.volume).collect() }
    pub fn timestamps(&self) -> Vec<u64> { self.bars.iter().map(|b| b.timestamp).collect() }

    pub fn returns(&self) -> Vec<f64> {
        if self.bars.len() < 2 { return vec![]; }
        self.bars.windows(2).map(|w| {
            if w[0].close.abs() < 1e-15 { 0.0 } else { (w[1].close - w[0].close) / w[0].close }
        }).collect()
    }

    pub fn log_returns(&self) -> Vec<f64> {
        if self.bars.len() < 2 { return vec![]; }
        self.bars.windows(2).map(|w| {
            if w[0].close <= 0.0 || w[1].close <= 0.0 { 0.0 } else { (w[1].close / w[0].close).ln() }
        }).collect()
    }

    pub fn slice(&self, start: usize, end: usize) -> BarSeries {
        BarSeries { symbol: self.symbol.clone(), bars: self.bars[start..end].to_vec() }
    }

    pub fn get_bar_at(&self, timestamp: u64) -> Option<&Bar> {
        self.bars.iter().find(|b| b.timestamp == timestamp)
    }

    pub fn last_bar(&self) -> Option<&Bar> { self.bars.last() }
}

/// Tick series
#[derive(Clone, Debug)]
pub struct TickSeries {
    pub symbol: String,
    pub ticks: Vec<Tick>,
}

impl TickSeries {
    pub fn new(symbol: &str) -> Self { Self { symbol: symbol.to_string(), ticks: Vec::new() } }
    pub fn push(&mut self, tick: Tick) { self.ticks.push(tick); }
    pub fn len(&self) -> usize { self.ticks.len() }

    pub fn aggregate_to_bars(&self, interval_ms: u64) -> BarSeries {
        if self.ticks.is_empty() { return BarSeries::new(&self.symbol); }
        let mut bars = Vec::new();
        let mut bucket_start = self.ticks[0].timestamp / interval_ms * interval_ms;
        let mut o = self.ticks[0].price;
        let mut h = o; let mut l = o; let mut c = o;
        let mut vol = 0.0;
        let mut vwap_num = 0.0;

        for tick in &self.ticks {
            let bucket = tick.timestamp / interval_ms * interval_ms;
            if bucket != bucket_start {
                let mut bar = Bar::new(bucket_start, o, h, l, c, vol);
                if vol > 0.0 { bar.vwap = Some(vwap_num / vol); }
                bars.push(bar);
                bucket_start = bucket;
                o = tick.price; h = tick.price; l = tick.price;
                vol = 0.0; vwap_num = 0.0;
            }
            h = h.max(tick.price);
            l = l.min(tick.price);
            c = tick.price;
            vol += tick.size;
            vwap_num += tick.price * tick.size;
        }
        let mut bar = Bar::new(bucket_start, o, h, l, c, vol);
        if vol > 0.0 { bar.vwap = Some(vwap_num / vol); }
        bars.push(bar);

        BarSeries { symbol: self.symbol.clone(), bars }
    }
}

/// Multi-asset dataset
#[derive(Clone, Debug)]
pub struct MultiAssetData {
    pub series: HashMap<String, BarSeries>,
    pub aligned_timestamps: Vec<u64>,
}

impl MultiAssetData {
    pub fn new() -> Self { Self { series: HashMap::new(), aligned_timestamps: Vec::new() } }

    pub fn add_series(&mut self, series: BarSeries) {
        self.series.insert(series.symbol.clone(), series);
    }

    pub fn symbols(&self) -> Vec<&str> {
        self.series.keys().map(|s| s.as_str()).collect()
    }

    /// Align all series to common timestamps (intersection)
    pub fn align_intersection(&mut self) {
        if self.series.is_empty() { return; }
        let mut ts_sets: Vec<std::collections::HashSet<u64>> = self.series.values()
            .map(|s| s.bars.iter().map(|b| b.timestamp).collect())
            .collect();
        let mut common: std::collections::HashSet<u64> = ts_sets.remove(0);
        for set in &ts_sets {
            common = common.intersection(set).cloned().collect();
        }
        let mut aligned: Vec<u64> = common.into_iter().collect();
        aligned.sort();
        self.aligned_timestamps = aligned;
    }

    /// Align all series (union with forward fill)
    pub fn align_union_ffill(&mut self) {
        if self.series.is_empty() { return; }
        let mut all_ts = std::collections::BTreeSet::new();
        for s in self.series.values() {
            for b in &s.bars { all_ts.insert(b.timestamp); }
        }
        let timestamps: Vec<u64> = all_ts.into_iter().collect();

        for (_, series) in self.series.iter_mut() {
            let ts_map: HashMap<u64, usize> = series.bars.iter().enumerate()
                .map(|(i, b)| (b.timestamp, i)).collect();
            let mut new_bars = Vec::with_capacity(timestamps.len());
            let mut last_bar: Option<Bar> = None;
            for &ts in &timestamps {
                if let Some(&idx) = ts_map.get(&ts) {
                    let bar = series.bars[idx].clone();
                    last_bar = Some(bar.clone());
                    new_bars.push(bar);
                } else if let Some(ref lb) = last_bar {
                    let mut filled = lb.clone();
                    filled.timestamp = ts;
                    filled.volume = 0.0;
                    new_bars.push(filled);
                } else {
                    // no previous bar, skip or insert NaN
                    new_bars.push(Bar::new(ts, f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0.0));
                }
            }
            series.bars = new_bars;
        }
        self.aligned_timestamps = timestamps;
    }

    pub fn get_closes_at(&self, idx: usize) -> HashMap<&str, f64> {
        let mut result = HashMap::new();
        for (sym, series) in &self.series {
            if idx < series.bars.len() {
                result.insert(sym.as_str(), series.bars[idx].close);
            }
        }
        result
    }

    pub fn num_timestamps(&self) -> usize { self.aligned_timestamps.len() }
}

/// CSV bar reader
pub fn parse_bars_csv(content: &str, has_header: bool) -> BarSeries {
    let mut series = BarSeries::new("unknown");
    let mut lines = content.lines();
    if has_header { lines.next(); }
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6 { continue; }
        let ts = fields[0].trim().parse::<u64>().unwrap_or(0);
        let o = fields[1].trim().parse::<f64>().unwrap_or(0.0);
        let h = fields[2].trim().parse::<f64>().unwrap_or(0.0);
        let l = fields[3].trim().parse::<f64>().unwrap_or(0.0);
        let c = fields[4].trim().parse::<f64>().unwrap_or(0.0);
        let v = fields[5].trim().parse::<f64>().unwrap_or(0.0);
        series.push(Bar::new(ts, o, h, l, c, v));
    }
    series
}

pub fn bars_to_csv(series: &BarSeries) -> String {
    let mut out = String::from("timestamp,open,high,low,close,volume\n");
    for b in &series.bars {
        out.push_str(&format!("{},{},{},{},{},{}\n", b.timestamp, b.open, b.high, b.low, b.close, b.volume));
    }
    out
}

/// Binary bar format: [u64 ts, f64 o, f64 h, f64 l, f64 c, f64 v] = 48 bytes per bar
pub fn serialize_bars(series: &BarSeries) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 + series.bars.len() * 48);
    bytes.extend_from_slice(b"BARS");
    bytes.extend_from_slice(&(series.bars.len() as u32).to_le_bytes());
    for b in &series.bars {
        bytes.extend_from_slice(&b.timestamp.to_le_bytes());
        bytes.extend_from_slice(&b.open.to_le_bytes());
        bytes.extend_from_slice(&b.high.to_le_bytes());
        bytes.extend_from_slice(&b.low.to_le_bytes());
        bytes.extend_from_slice(&b.close.to_le_bytes());
        bytes.extend_from_slice(&b.volume.to_le_bytes());
    }
    bytes
}

pub fn deserialize_bars(bytes: &[u8]) -> Option<BarSeries> {
    if bytes.len() < 8 { return None; }
    if &bytes[0..4] != b"BARS" { return None; }
    let n = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    if bytes.len() < 8 + n * 48 { return None; }
    let mut series = BarSeries::new("unknown");
    let mut pos = 8;
    for _ in 0..n {
        let ts = u64::from_le_bytes(bytes[pos..pos + 8].try_into().ok()?); pos += 8;
        let o = f64::from_le_bytes(bytes[pos..pos + 8].try_into().ok()?); pos += 8;
        let h = f64::from_le_bytes(bytes[pos..pos + 8].try_into().ok()?); pos += 8;
        let l = f64::from_le_bytes(bytes[pos..pos + 8].try_into().ok()?); pos += 8;
        let c = f64::from_le_bytes(bytes[pos..pos + 8].try_into().ok()?); pos += 8;
        let v = f64::from_le_bytes(bytes[pos..pos + 8].try_into().ok()?); pos += 8;
        series.push(Bar::new(ts, o, h, l, c, v));
    }
    Some(series)
}

/// Corporate action types
#[derive(Clone, Debug)]
pub enum CorporateAction {
    Split { timestamp: u64, ratio: f64 },
    Dividend { timestamp: u64, amount: f64 },
    Merger { timestamp: u64, exchange_ratio: f64 },
    SpinOff { timestamp: u64, ratio: f64, new_symbol: String },
}

/// Adjust bar series for splits
pub fn adjust_for_splits(series: &mut BarSeries, splits: &[(u64, f64)]) {
    for &(ts, ratio) in splits {
        for bar in series.bars.iter_mut() {
            if bar.timestamp < ts {
                bar.open /= ratio;
                bar.high /= ratio;
                bar.low /= ratio;
                bar.close /= ratio;
                bar.volume *= ratio;
            }
        }
    }
}

/// Adjust for dividends (price only)
pub fn adjust_for_dividends(series: &mut BarSeries, dividends: &[(u64, f64)]) {
    for &(ts, amount) in dividends {
        // Find price at ex-date
        let price_at = series.bars.iter().find(|b| b.timestamp >= ts)
            .map(|b| b.close).unwrap_or(100.0);
        let factor = 1.0 - amount / price_at;
        for bar in series.bars.iter_mut() {
            if bar.timestamp < ts {
                bar.open *= factor;
                bar.high *= factor;
                bar.low *= factor;
                bar.close *= factor;
            }
        }
    }
}

/// Handle missing data in bar series
#[derive(Clone, Copy, Debug)]
pub enum FillMethod {
    ForwardFill,
    BackwardFill,
    Linear,
    Zero,
    Drop,
}

pub fn fill_missing_bars(series: &mut BarSeries, method: FillMethod) {
    match method {
        FillMethod::ForwardFill => {
            let mut last_valid: Option<Bar> = None;
            for bar in series.bars.iter_mut() {
                if bar.close.is_nan() {
                    if let Some(ref lv) = last_valid {
                        bar.open = lv.close;
                        bar.high = lv.close;
                        bar.low = lv.close;
                        bar.close = lv.close;
                    }
                } else {
                    last_valid = Some(bar.clone());
                }
            }
        }
        FillMethod::BackwardFill => {
            let n = series.bars.len();
            let mut next_valid: Option<Bar> = None;
            for i in (0..n).rev() {
                if series.bars[i].close.is_nan() {
                    if let Some(ref nv) = next_valid {
                        series.bars[i].open = nv.open;
                        series.bars[i].high = nv.open;
                        series.bars[i].low = nv.open;
                        series.bars[i].close = nv.open;
                    }
                } else {
                    next_valid = Some(series.bars[i].clone());
                }
            }
        }
        FillMethod::Linear => {
            let n = series.bars.len();
            let mut i = 0;
            while i < n {
                if series.bars[i].close.is_nan() {
                    let start = if i > 0 { i - 1 } else { i };
                    let mut end = i;
                    while end < n && series.bars[end].close.is_nan() { end += 1; }
                    if start < n && end < n && !series.bars[start].close.is_nan() && !series.bars[end].close.is_nan() {
                        let gap = end - start;
                        for j in (start + 1)..end {
                            let frac = (j - start) as f64 / gap as f64;
                            let interp = series.bars[start].close * (1.0 - frac) + series.bars[end].close * frac;
                            series.bars[j].open = interp;
                            series.bars[j].high = interp;
                            series.bars[j].low = interp;
                            series.bars[j].close = interp;
                        }
                    }
                    i = end;
                } else {
                    i += 1;
                }
            }
        }
        FillMethod::Zero => {
            for bar in series.bars.iter_mut() {
                if bar.close.is_nan() {
                    bar.open = 0.0; bar.high = 0.0; bar.low = 0.0; bar.close = 0.0;
                }
            }
        }
        FillMethod::Drop => {
            series.bars.retain(|b| !b.close.is_nan());
        }
    }
}

/// Data validation
pub fn validate_series(series: &BarSeries) -> Vec<String> {
    let mut issues = Vec::new();
    for (i, bar) in series.bars.iter().enumerate() {
        if !bar.validate() {
            issues.push(format!("Bar {} (ts={}): invalid OHLCV", i, bar.timestamp));
        }
        if i > 0 && bar.timestamp <= series.bars[i - 1].timestamp {
            issues.push(format!("Bar {} (ts={}): non-increasing timestamp", i, bar.timestamp));
        }
        if bar.close.is_nan() || bar.close.is_infinite() {
            issues.push(format!("Bar {} (ts={}): NaN/Inf close", i, bar.timestamp));
        }
    }
    issues
}

/// Generate synthetic bar data for testing
pub fn generate_synthetic_bars(num_bars: usize, initial_price: f64, volatility: f64, seed: u64) -> BarSeries {
    let mut series = BarSeries::new("SYNTH");
    let mut price = initial_price;
    let mut state = seed;
    let dt = 60000u64; // 1 minute bars

    for i in 0..num_bars {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
        // Box-Muller
        let z = (-2.0 * u1.max(1e-15).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let ret = z * volatility;
        let new_price = price * (1.0 + ret);

        let o = price;
        let c = new_price;
        let h = o.max(c) * (1.0 + u1 * volatility * 0.5);
        let l = o.min(c) * (1.0 - u2 * volatility * 0.5);
        let v = (10000.0 * (1.0 + z.abs())).round();

        series.push(Bar::new(i as u64 * dt, o, h, l, c, v));
        price = new_price;
    }
    series
}

/// Volume-weighted bar aggregation
pub fn volume_bars(ticks: &[Tick], volume_threshold: f64) -> BarSeries {
    let mut bars = BarSeries::new("VBAR");
    if ticks.is_empty() { return bars; }
    let mut o = ticks[0].price;
    let mut h = o; let mut l = o; let mut c = o;
    let mut vol = 0.0;
    let mut ts = ticks[0].timestamp;
    for tick in ticks {
        h = h.max(tick.price);
        l = l.min(tick.price);
        c = tick.price;
        vol += tick.size;
        if vol >= volume_threshold {
            bars.push(Bar::new(ts, o, h, l, c, vol));
            o = tick.price; h = o; l = o;
            vol = 0.0;
            ts = tick.timestamp;
        }
    }
    if vol > 0.0 { bars.push(Bar::new(ts, o, h, l, c, vol)); }
    bars
}

/// Dollar bars
pub fn dollar_bars(ticks: &[Tick], dollar_threshold: f64) -> BarSeries {
    let mut bars = BarSeries::new("DBAR");
    if ticks.is_empty() { return bars; }
    let mut o = ticks[0].price;
    let mut h = o; let mut l = o; let mut c = o;
    let mut vol = 0.0;
    let mut dollar_vol = 0.0;
    let mut ts = ticks[0].timestamp;
    for tick in ticks {
        h = h.max(tick.price);
        l = l.min(tick.price);
        c = tick.price;
        vol += tick.size;
        dollar_vol += tick.notional();
        if dollar_vol >= dollar_threshold {
            bars.push(Bar::new(ts, o, h, l, c, vol));
            o = tick.price; h = o; l = o;
            vol = 0.0; dollar_vol = 0.0;
            ts = tick.timestamp;
        }
    }
    if vol > 0.0 { bars.push(Bar::new(ts, o, h, l, c, vol)); }
    bars
}

/// Tick imbalance bars (TIB)
pub fn tick_imbalance_bars(ticks: &[Tick], expected_imbalance: f64) -> BarSeries {
    let mut bars = BarSeries::new("TIB");
    if ticks.is_empty() { return bars; }
    let mut o = ticks[0].price;
    let mut h = o; let mut l = o; let mut c = o;
    let mut vol = 0.0;
    let mut imbalance = 0.0f64;
    let mut ts = ticks[0].timestamp;
    let mut prev_price = ticks[0].price;
    for tick in ticks {
        let direction = if tick.price > prev_price { 1.0 } else if tick.price < prev_price { -1.0 } else { 0.0 };
        imbalance += direction;
        h = h.max(tick.price);
        l = l.min(tick.price);
        c = tick.price;
        vol += tick.size;
        if imbalance.abs() >= expected_imbalance {
            bars.push(Bar::new(ts, o, h, l, c, vol));
            o = tick.price; h = o; l = o;
            vol = 0.0; imbalance = 0.0;
            ts = tick.timestamp;
        }
        prev_price = tick.price;
    }
    if vol > 0.0 { bars.push(Bar::new(ts, o, h, l, c, vol)); }
    bars
}

/// Resample bars to lower frequency
pub fn resample_bars(series: &BarSeries, factor: usize) -> BarSeries {
    let mut result = BarSeries::new(&series.symbol);
    for chunk in series.bars.chunks(factor) {
        if chunk.is_empty() { continue; }
        let ts = chunk[0].timestamp;
        let o = chunk[0].open;
        let h = chunk.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
        let l = chunk.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
        let c = chunk.last().unwrap().close;
        let v: f64 = chunk.iter().map(|b| b.volume).sum();
        result.push(Bar::new(ts, o, h, l, c, v));
    }
    result
}

/// Compute rolling statistics on bar series
pub fn rolling_volatility_bars(series: &BarSeries, window: usize) -> Vec<f64> {
    let rets = series.returns();
    if rets.len() < window { return vec![]; }
    (0..=rets.len() - window).map(|i| {
        let w = &rets[i..i + window];
        let mean = w.iter().sum::<f64>() / window as f64;
        (w.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / window as f64).sqrt()
    }).collect()
}

/// Compute rolling mean
pub fn rolling_mean_bars(series: &BarSeries, window: usize) -> Vec<f64> {
    let prices = series.close_prices();
    if prices.len() < window { return vec![]; }
    (0..=prices.len() - window).map(|i| {
        prices[i..i + window].iter().sum::<f64>() / window as f64
    }).collect()
}

/// Compute Bollinger Bands
pub fn bollinger_bands(series: &BarSeries, window: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let prices = series.close_prices();
    if prices.len() < window { return (vec![], vec![], vec![]); }
    let n = prices.len() - window + 1;
    let mut upper = Vec::with_capacity(n);
    let mut middle = Vec::with_capacity(n);
    let mut lower = Vec::with_capacity(n);
    for i in 0..n {
        let sl = &prices[i..i + window];
        let mean = sl.iter().sum::<f64>() / window as f64;
        let std = (sl.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / window as f64).sqrt();
        middle.push(mean);
        upper.push(mean + num_std * std);
        lower.push(mean - num_std * std);
    }
    (upper, middle, lower)
}

/// Compute RSI
pub fn compute_rsi(series: &BarSeries, period: usize) -> Vec<f64> {
    let prices = series.close_prices();
    if prices.len() < period + 1 { return vec![]; }
    let mut result = Vec::new();
    for i in period..prices.len() {
        let mut gains = 0.0;
        let mut losses = 0.0;
        for j in (i - period)..i {
            let diff = prices[j + 1] - prices[j];
            if diff > 0.0 { gains += diff; } else { losses -= diff; }
        }
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        let rsi = if avg_loss < 1e-15 { 100.0 } else { 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) };
        result.push(rsi);
    }
    result
}

/// Compute MACD
pub fn compute_macd(series: &BarSeries, fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let prices = series.close_prices();
    if prices.len() < slow { return (vec![], vec![], vec![]); }

    let ema = |data: &[f64], period: usize| -> Vec<f64> {
        let alpha = 2.0 / (period + 1) as f64;
        let mut result = vec![data[0]];
        for i in 1..data.len() {
            result.push(alpha * data[i] + (1.0 - alpha) * result[i - 1]);
        }
        result
    };

    let fast_ema = ema(&prices, fast);
    let slow_ema = ema(&prices, slow);
    let macd_line: Vec<f64> = fast_ema.iter().zip(slow_ema.iter()).map(|(&f, &s)| f - s).collect();
    let signal_line = ema(&macd_line, signal);
    let histogram: Vec<f64> = macd_line.iter().zip(signal_line.iter()).map(|(&m, &s)| m - s).collect();
    (macd_line, signal_line, histogram)
}

/// Average True Range
pub fn compute_atr(series: &BarSeries, period: usize) -> Vec<f64> {
    if series.len() < period + 1 { return vec![]; }
    let mut tr = Vec::with_capacity(series.len() - 1);
    for i in 1..series.len() {
        let bar = &series.bars[i];
        let prev_close = series.bars[i - 1].close;
        let t = (bar.high - bar.low)
            .max((bar.high - prev_close).abs())
            .max((bar.low - prev_close).abs());
        tr.push(t);
    }
    if tr.len() < period { return vec![]; }
    let mut atr = Vec::with_capacity(tr.len() - period + 1);
    let first: f64 = tr[..period].iter().sum::<f64>() / period as f64;
    atr.push(first);
    for i in period..tr.len() {
        let prev = atr.last().unwrap();
        atr.push((prev * (period - 1) as f64 + tr[i]) / period as f64);
    }
    atr
}

/// On-balance volume
pub fn compute_obv(series: &BarSeries) -> Vec<f64> {
    let mut obv = vec![0.0];
    for i in 1..series.len() {
        let prev = obv.last().unwrap();
        let dir = if series.bars[i].close > series.bars[i - 1].close { 1.0 }
            else if series.bars[i].close < series.bars[i - 1].close { -1.0 }
            else { 0.0 };
        obv.push(prev + dir * series.bars[i].volume);
    }
    obv
}

/// Commodity Channel Index
pub fn compute_cci(series: &BarSeries, period: usize) -> Vec<f64> {
    if series.len() < period { return vec![]; }
    let tp: Vec<f64> = series.bars.iter().map(|b| b.typical_price()).collect();
    let mut result = Vec::new();
    for i in (period - 1)..tp.len() {
        let sl = &tp[i + 1 - period..=i];
        let mean = sl.iter().sum::<f64>() / period as f64;
        let mean_dev = sl.iter().map(|&p| (p - mean).abs()).sum::<f64>() / period as f64;
        let cci = if mean_dev > 1e-15 { (tp[i] - mean) / (0.015 * mean_dev) } else { 0.0 };
        result.push(cci);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bar() {
        let bar = Bar::new(0, 100.0, 105.0, 95.0, 103.0, 1000.0);
        assert!(bar.validate());
        assert!(bar.is_green());
        assert!((bar.typical_price() - 101.0).abs() < 1e-10);
    }

    #[test]
    fn test_csv_parse() {
        let csv = "ts,o,h,l,c,v\n1000,100,105,95,103,500\n2000,103,108,101,106,600\n";
        let series = parse_bars_csv(csv, true);
        assert_eq!(series.len(), 2);
        assert_eq!(series.bars[0].close, 103.0);
    }

    #[test]
    fn test_binary_roundtrip() {
        let series = generate_synthetic_bars(10, 100.0, 0.01, 42);
        let bytes = serialize_bars(&series);
        let series2 = deserialize_bars(&bytes).unwrap();
        assert_eq!(series2.len(), 10);
        assert!((series2.bars[0].close - series.bars[0].close).abs() < 1e-10);
    }

    #[test]
    fn test_returns() {
        let mut series = BarSeries::new("TEST");
        series.push(Bar::new(0, 100.0, 100.0, 100.0, 100.0, 100.0));
        series.push(Bar::new(1, 100.0, 100.0, 100.0, 110.0, 100.0));
        let rets = series.returns();
        assert_eq!(rets.len(), 1);
        assert!((rets[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_tick_aggregation() {
        let mut ticks = TickSeries::new("TEST");
        ticks.push(Tick::new(0, 100.0, 10.0, TickSide::Buy));
        ticks.push(Tick::new(50, 101.0, 5.0, TickSide::Sell));
        ticks.push(Tick::new(1000, 102.0, 8.0, TickSide::Buy));
        let bars = ticks.aggregate_to_bars(1000);
        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn test_synthetic() {
        let series = generate_synthetic_bars(100, 100.0, 0.02, 12345);
        assert_eq!(series.len(), 100);
        for bar in &series.bars { assert!(bar.validate()); }
    }

    #[test]
    fn test_split_adjustment() {
        let mut series = BarSeries::new("TEST");
        series.push(Bar::new(100, 200.0, 200.0, 200.0, 200.0, 500.0));
        series.push(Bar::new(200, 100.0, 100.0, 100.0, 100.0, 1000.0));
        adjust_for_splits(&mut series, &[(200, 2.0)]);
        assert!((series.bars[0].close - 100.0).abs() < 1e-10);
        assert!((series.bars[0].volume - 1000.0).abs() < 1e-10);
    }
}
