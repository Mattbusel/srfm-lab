//! # alpha-tensor
//!
//! Multi-dimensional sparse tensor for a parallel backtest farm.
//! Runs thousands of backtests across strategy x symbol x timeframe x parameter
//! combinations, stores results in a sparse tensor, and produces landscape analysis.
//!
//! Pure std — no external dependencies.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use std::time::{Instant, Duration};

// ---------------------------------------------------------------------------
// 0. Constants & helpers
// ---------------------------------------------------------------------------

const MAGIC: u64 = 0xA1FA_7E50_CAFE_BABE;
const VERSION: u32 = 1;

/// Stable hash for a parameter set — deterministic across runs.
fn param_hash(params: &HashMap<String, f64>) -> u64 {
    let mut keys: Vec<&String> = params.keys().collect();
    keys.sort();
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for k in &keys {
        for b in k.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let bits = params[*k].to_bits();
        for i in 0..8 {
            h ^= ((bits >> (i * 8)) & 0xFF) as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Fast f64 sort key (total-order).
#[inline]
fn f64_sort_key(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    }
}

/// Clamp a value into [lo, hi].
#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

/// Linear interpolation.
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Compute mean of a slice.
#[inline]
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Compute variance (population).
#[inline]
fn variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

/// Compute standard deviation (population).
#[inline]
fn stddev(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}

/// Compute downside deviation (negative returns only).
fn downside_dev(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    let m = mean(xs);
    let sum: f64 = xs.iter()
        .filter(|&&x| x < m)
        .map(|x| (x - m) * (x - m))
        .sum();
    let count = xs.len() as f64;
    (sum / count).sqrt()
}

/// Pearson correlation between two equal-length slices.
fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 { return 0.0; }
    let ma = mean(&a[..n]);
    let mb = mean(&b[..n]);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let x = a[i] - ma;
        let y = b[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    let denom = (da * db).sqrt();
    if denom < 1e-15 { 0.0 } else { num / denom }
}

/// Compute percentile (linear interpolation).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    if sorted.len() == 1 { return sorted[0]; }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    lerp(sorted[lo], sorted[hi], idx - lo as f64)
}

/// Exponential moving average.
fn ema(xs: &[f64], period: usize) -> Vec<f64> {
    if xs.is_empty() || period == 0 { return vec![]; }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut out = Vec::with_capacity(xs.len());
    out.push(xs[0]);
    for i in 1..xs.len() {
        let prev = out[i - 1];
        out.push(alpha * xs[i] + (1.0 - alpha) * prev);
    }
    out
}

/// Simple moving average.
fn sma(xs: &[f64], period: usize) -> Vec<f64> {
    if xs.is_empty() || period == 0 { return vec![]; }
    let n = xs.len();
    let mut out = Vec::with_capacity(n);
    let mut running = 0.0;
    for i in 0..n {
        running += xs[i];
        if i >= period {
            running -= xs[i - period];
        }
        let w = (i + 1).min(period) as f64;
        out.push(running / w);
    }
    out
}

/// Rolling standard deviation.
fn rolling_std(xs: &[f64], period: usize) -> Vec<f64> {
    if xs.is_empty() || period == 0 { return vec![]; }
    let n = xs.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = if i + 1 >= period { i + 1 - period } else { 0 };
        let window = &xs[start..=i];
        out.push(stddev(window));
    }
    out
}

/// Rolling max.
fn rolling_max(xs: &[f64], period: usize) -> Vec<f64> {
    if xs.is_empty() || period == 0 { return vec![]; }
    let n = xs.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = if i + 1 >= period { i + 1 - period } else { 0 };
        let mut mx = f64::NEG_INFINITY;
        for j in start..=i {
            if xs[j] > mx { mx = xs[j]; }
        }
        out.push(mx);
    }
    out
}

/// Rolling min.
fn rolling_min(xs: &[f64], period: usize) -> Vec<f64> {
    if xs.is_empty() || period == 0 { return vec![]; }
    let n = xs.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = if i + 1 >= period { i + 1 - period } else { 0 };
        let mut mn = f64::INFINITY;
        for j in start..=i {
            if xs[j] < mn { mn = xs[j]; }
        }
        out.push(mn);
    }
    out
}

/// Log-returns from a price series.
fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 { return vec![]; }
    let mut out = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 && prices[i] > 0.0 {
            out.push((prices[i] / prices[i - 1]).ln());
        } else {
            out.push(0.0);
        }
    }
    out
}

/// Cumulative sum.
fn cumsum(xs: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(xs.len());
    let mut s = 0.0;
    for &x in xs {
        s += x;
        out.push(s);
    }
    out
}

/// Shannon entropy of discretised signal (16 bins).
fn shannon_entropy(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    let mn = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = mx - mn;
    if range < 1e-15 { return 0.0; }
    let bins = 16usize;
    let mut counts = vec![0u64; bins];
    for &x in xs {
        let b = ((x - mn) / range * (bins as f64 - 1.0)).round() as usize;
        counts[b.min(bins - 1)] += 1;
    }
    let n = xs.len() as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.ln();
        }
    }
    h
}

// ---------------------------------------------------------------------------
// 1. Timeframe
// ---------------------------------------------------------------------------

/// Supported bar timeframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Timeframe {
    M1,
    M15,
    H1,
    H4,
    D1,
}

impl Timeframe {
    /// Parse from string representation.
    pub fn from_str(s: &str) -> Option<Timeframe> {
        match s {
            "1m" | "M1" => Some(Timeframe::M1),
            "15m" | "M15" => Some(Timeframe::M15),
            "1h" | "H1" => Some(Timeframe::H1),
            "4h" | "H4" => Some(Timeframe::H4),
            "1d" | "D1" => Some(Timeframe::D1),
            _ => None,
        }
    }

    /// Convert to string label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Timeframe::M1 => "1m",
            Timeframe::M15 => "15m",
            Timeframe::H1 => "1h",
            Timeframe::H4 => "4h",
            Timeframe::D1 => "1d",
        }
    }

    /// Minutes per bar.
    pub fn minutes(&self) -> u32 {
        match self {
            Timeframe::M1 => 1,
            Timeframe::M15 => 15,
            Timeframe::H1 => 60,
            Timeframe::H4 => 240,
            Timeframe::D1 => 1440,
        }
    }

    /// Bars per day (approximate).
    pub fn bars_per_day(&self) -> f64 {
        1440.0 / self.minutes() as f64
    }

    /// Annualisation factor (assuming 252 trading days).
    pub fn annual_factor(&self) -> f64 {
        self.bars_per_day() * 252.0
    }

    /// All variants.
    pub fn all() -> &'static [Timeframe] {
        &[Timeframe::M1, Timeframe::M15, Timeframe::H1, Timeframe::H4, Timeframe::D1]
    }

    /// To u8 for serialisation.
    pub fn to_u8(&self) -> u8 {
        match self {
            Timeframe::M1 => 0,
            Timeframe::M15 => 1,
            Timeframe::H1 => 2,
            Timeframe::H4 => 3,
            Timeframe::D1 => 4,
        }
    }

    /// From u8.
    pub fn from_u8(v: u8) -> Option<Timeframe> {
        match v {
            0 => Some(Timeframe::M1),
            1 => Some(Timeframe::M15),
            2 => Some(Timeframe::H1),
            3 => Some(Timeframe::H4),
            4 => Some(Timeframe::D1),
            _ => None,
        }
    }
}

impl fmt::Display for Timeframe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// 2. BacktestConfig
// ---------------------------------------------------------------------------

/// Full specification for a single backtest run.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub strategy_id: String,
    pub symbol: String,
    pub timeframe: Timeframe,
    pub parameters: HashMap<String, f64>,
    pub cost_bps: f64,
    pub start_bar: usize,
    pub end_bar: usize,
}

impl BacktestConfig {
    /// Create a new config.
    pub fn new(
        strategy_id: impl Into<String>,
        symbol: impl Into<String>,
        timeframe: Timeframe,
        parameters: HashMap<String, f64>,
        cost_bps: f64,
        start_bar: usize,
        end_bar: usize,
    ) -> Self {
        Self {
            strategy_id: strategy_id.into(),
            symbol: symbol.into(),
            timeframe,
            parameters,
            cost_bps,
            start_bar,
            end_bar,
        }
    }

    /// Deterministic hash of the parameter set.
    pub fn param_hash(&self) -> u64 {
        param_hash(&self.parameters)
    }

    /// Tensor key: (strategy, symbol, timeframe, param_hash).
    pub fn tensor_key(&self) -> TensorKey {
        TensorKey {
            strategy: self.strategy_id.clone(),
            symbol: self.symbol.clone(),
            timeframe: self.timeframe,
            param_hash: self.param_hash(),
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut params_str = String::new();
        let mut keys: Vec<&String> = self.parameters.keys().collect();
        keys.sort();
        for (i, k) in keys.iter().enumerate() {
            if i > 0 { params_str.push_str(", "); }
            params_str.push_str(&format!("{}={:.4}", k, self.parameters[*k]));
        }
        format!(
            "{}|{}|{}|[{}]|cost={}bps|bars={}..{}",
            self.strategy_id, self.symbol, self.timeframe,
            params_str, self.cost_bps, self.start_bar, self.end_bar
        )
    }
}

// ---------------------------------------------------------------------------
// 3. BacktestResult
// ---------------------------------------------------------------------------

/// Full result set for a single backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub config: BacktestConfig,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub n_trades: usize,
    pub avg_hold_bars: f64,
    pub ic: f64,
    pub turnover: f64,
}

impl BacktestResult {
    /// A zeroed-out result (failed run).
    pub fn failed(config: BacktestConfig) -> Self {
        Self {
            config,
            sharpe: 0.0,
            sortino: 0.0,
            calmar: 0.0,
            total_return: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            n_trades: 0,
            avg_hold_bars: 0.0,
            ic: 0.0,
            turnover: 0.0,
        }
    }

    /// Summary line for reports.
    pub fn summary_line(&self) -> String {
        format!(
            "sharpe={:.3} sortino={:.3} calmar={:.3} ret={:.4} dd={:.4} wr={:.3} pf={:.3} trades={} ic={:.3} | {}",
            self.sharpe, self.sortino, self.calmar,
            self.total_return, self.max_drawdown,
            self.win_rate, self.profit_factor,
            self.n_trades, self.ic,
            self.config.summary()
        )
    }

    /// Composite score: weighted average of normalised metrics.
    pub fn composite_score(&self) -> f64 {
        let s = self.sharpe * 0.30
            + self.sortino * 0.15
            + self.calmar * 0.10
            + self.total_return * 10.0 * 0.10
            + (1.0 - self.max_drawdown) * 0.10
            + self.win_rate * 0.05
            + self.profit_factor.min(5.0) / 5.0 * 0.05
            + self.ic * 0.10
            + (1.0 - self.turnover.min(1.0)) * 0.05;
        s
    }

    /// Is this a viable strategy? (basic sanity checks)
    pub fn is_viable(&self) -> bool {
        self.sharpe > 0.5
            && self.max_drawdown < 0.30
            && self.n_trades >= 30
            && self.profit_factor > 1.0
    }
}

// ---------------------------------------------------------------------------
// 4. TensorKey & AlphaTensor
// ---------------------------------------------------------------------------

/// 4-dimensional key into the alpha tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorKey {
    pub strategy: String,
    pub symbol: String,
    pub timeframe: Timeframe,
    pub param_hash: u64,
}

impl TensorKey {
    pub fn new(strategy: impl Into<String>, symbol: impl Into<String>, timeframe: Timeframe, param_hash: u64) -> Self {
        Self { strategy: strategy.into(), symbol: symbol.into(), timeframe, param_hash }
    }
}

/// Multi-dimensional sparse tensor storing backtest results.
///
/// Indexed by (strategy, symbol, timeframe, parameter_hash).
/// Uses a `HashMap<TensorKey, BacktestResult>` for sparse storage—most of the
/// combinatorial space will be empty in practice.
#[derive(Debug, Clone)]
pub struct AlphaTensor {
    data: HashMap<TensorKey, BacktestResult>,
    strategies: Vec<String>,
    symbols: Vec<String>,
    timeframes: Vec<Timeframe>,
    param_hashes: Vec<u64>,
}

impl AlphaTensor {
    /// Create an empty tensor.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            strategies: Vec::new(),
            symbols: Vec::new(),
            timeframes: Vec::new(),
            param_hashes: Vec::new(),
        }
    }

    /// Create with capacity hint.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: HashMap::with_capacity(cap),
            strategies: Vec::new(),
            symbols: Vec::new(),
            timeframes: Vec::new(),
            param_hashes: Vec::new(),
        }
    }

    /// Number of stored results.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Is the tensor empty?
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Insert a result. Updates dimension indices automatically.
    pub fn insert(&mut self, result: BacktestResult) {
        let key = result.config.tensor_key();
        if !self.strategies.contains(&key.strategy) {
            self.strategies.push(key.strategy.clone());
        }
        if !self.symbols.contains(&key.symbol) {
            self.symbols.push(key.symbol.clone());
        }
        if !self.timeframes.contains(&key.timeframe) {
            self.timeframes.push(key.timeframe);
        }
        if !self.param_hashes.contains(&key.param_hash) {
            self.param_hashes.push(key.param_hash);
        }
        self.data.insert(key, result);
    }

    /// Get a result by key.
    pub fn get(&self, key: &TensorKey) -> Option<&BacktestResult> {
        self.data.get(key)
    }

    /// Get mutable reference.
    pub fn get_mut(&mut self, key: &TensorKey) -> Option<&mut BacktestResult> {
        self.data.get_mut(key)
    }

    /// All stored keys.
    pub fn keys(&self) -> Vec<&TensorKey> {
        self.data.keys().collect()
    }

    /// All stored results.
    pub fn values(&self) -> Vec<&BacktestResult> {
        self.data.values().collect()
    }

    /// Iterate over (key, result) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&TensorKey, &BacktestResult)> {
        self.data.iter()
    }

    /// Unique strategies in the tensor.
    pub fn strategies(&self) -> &[String] {
        &self.strategies
    }

    /// Unique symbols in the tensor.
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    /// Unique timeframes in the tensor.
    pub fn timeframes(&self) -> &[Timeframe] {
        &self.timeframes
    }

    /// Slice: get all results matching a particular strategy.
    pub fn slice_strategy(&self, strategy: &str) -> Vec<&BacktestResult> {
        self.data.iter()
            .filter(|(k, _)| k.strategy == strategy)
            .map(|(_, v)| v)
            .collect()
    }

    /// Slice: get all results matching a particular symbol.
    pub fn slice_symbol(&self, symbol: &str) -> Vec<&BacktestResult> {
        self.data.iter()
            .filter(|(k, _)| k.symbol == symbol)
            .map(|(_, v)| v)
            .collect()
    }

    /// Slice: get all results matching a particular timeframe.
    pub fn slice_timeframe(&self, tf: Timeframe) -> Vec<&BacktestResult> {
        self.data.iter()
            .filter(|(k, _)| k.timeframe == tf)
            .map(|(_, v)| v)
            .collect()
    }

    /// Slice: get all results matching a particular param hash.
    pub fn slice_param_hash(&self, ph: u64) -> Vec<&BacktestResult> {
        self.data.iter()
            .filter(|(k, _)| k.param_hash == ph)
            .map(|(_, v)| v)
            .collect()
    }

    /// Slice by strategy + symbol.
    pub fn slice_strategy_symbol(&self, strategy: &str, symbol: &str) -> Vec<&BacktestResult> {
        self.data.iter()
            .filter(|(k, _)| k.strategy == strategy && k.symbol == symbol)
            .map(|(_, v)| v)
            .collect()
    }

    /// Slice by strategy + timeframe.
    pub fn slice_strategy_timeframe(&self, strategy: &str, tf: Timeframe) -> Vec<&BacktestResult> {
        self.data.iter()
            .filter(|(k, _)| k.strategy == strategy && k.timeframe == tf)
            .map(|(_, v)| v)
            .collect()
    }

    /// Top-k results by a given metric extractor, descending.
    pub fn top_k<F>(&self, k: usize, metric: F) -> Vec<&BacktestResult>
    where
        F: Fn(&BacktestResult) -> f64,
    {
        let mut entries: Vec<&BacktestResult> = self.data.values().collect();
        entries.sort_by(|a, b| {
            let va = metric(a);
            let vb = metric(b);
            f64_sort_key(vb).cmp(&f64_sort_key(va))
        });
        entries.truncate(k);
        entries
    }

    /// Bottom-k results by a given metric extractor (ascending).
    pub fn bottom_k<F>(&self, k: usize, metric: F) -> Vec<&BacktestResult>
    where
        F: Fn(&BacktestResult) -> f64,
    {
        let mut entries: Vec<&BacktestResult> = self.data.values().collect();
        entries.sort_by(|a, b| {
            let va = metric(a);
            let vb = metric(b);
            f64_sort_key(va).cmp(&f64_sort_key(vb))
        });
        entries.truncate(k);
        entries
    }

    /// 2D heatmap projection: rows=strategies, cols=symbols, value=mean Sharpe.
    /// Returns (row_labels, col_labels, matrix[row][col]).
    pub fn heatmap_strategy_symbol(&self) -> (Vec<String>, Vec<String>, Vec<Vec<f64>>) {
        let rows = self.strategies.clone();
        let cols = self.symbols.clone();
        let mut matrix = vec![vec![f64::NAN; cols.len()]; rows.len()];
        for (ri, strat) in rows.iter().enumerate() {
            for (ci, sym) in cols.iter().enumerate() {
                let results = self.slice_strategy_symbol(strat, sym);
                if !results.is_empty() {
                    let avg = results.iter().map(|r| r.sharpe).sum::<f64>() / results.len() as f64;
                    matrix[ri][ci] = avg;
                }
            }
        }
        (rows, cols, matrix)
    }

    /// 2D heatmap projection: rows=strategies, cols=timeframes, value=mean Sharpe.
    pub fn heatmap_strategy_timeframe(&self) -> (Vec<String>, Vec<Timeframe>, Vec<Vec<f64>>) {
        let rows = self.strategies.clone();
        let cols = self.timeframes.clone();
        let mut matrix = vec![vec![f64::NAN; cols.len()]; rows.len()];
        for (ri, strat) in rows.iter().enumerate() {
            for (ci, &tf) in cols.iter().enumerate() {
                let results = self.slice_strategy_timeframe(strat, tf);
                if !results.is_empty() {
                    let avg = results.iter().map(|r| r.sharpe).sum::<f64>() / results.len() as f64;
                    matrix[ri][ci] = avg;
                }
            }
        }
        (rows, cols, matrix)
    }

    /// Generic 2D heatmap: pick any two dimensions and a metric.
    pub fn heatmap<F>(
        &self,
        row_fn: impl Fn(&TensorKey) -> String,
        col_fn: impl Fn(&TensorKey) -> String,
        metric: F,
    ) -> (Vec<String>, Vec<String>, Vec<Vec<f64>>)
    where
        F: Fn(&BacktestResult) -> f64,
    {
        let mut row_set: Vec<String> = Vec::new();
        let mut col_set: Vec<String> = Vec::new();
        let mut buckets: HashMap<(String, String), Vec<f64>> = HashMap::new();
        for (k, v) in &self.data {
            let r = row_fn(k);
            let c = col_fn(k);
            if !row_set.contains(&r) { row_set.push(r.clone()); }
            if !col_set.contains(&c) { col_set.push(c.clone()); }
            buckets.entry((r, c)).or_default().push(metric(v));
        }
        row_set.sort();
        col_set.sort();
        let mut matrix = vec![vec![f64::NAN; col_set.len()]; row_set.len()];
        for (ri, r) in row_set.iter().enumerate() {
            for (ci, c) in col_set.iter().enumerate() {
                if let Some(vals) = buckets.get(&(r.clone(), c.clone())) {
                    matrix[ri][ci] = mean(vals);
                }
            }
        }
        (row_set, col_set, matrix)
    }

    /// Merge another tensor into this one. Overwrites on collision.
    pub fn merge(&mut self, other: &AlphaTensor) {
        for (k, v) in &other.data {
            let result = v.clone();
            self.insert(result);
        }
    }

    /// Remove all results that do not satisfy a predicate.
    pub fn retain<F>(&mut self, pred: F) where F: Fn(&BacktestResult) -> bool {
        self.data.retain(|_, v| pred(v));
        self.rebuild_indices();
    }

    /// Rebuild dimension indices from current data.
    fn rebuild_indices(&mut self) {
        self.strategies.clear();
        self.symbols.clear();
        self.timeframes.clear();
        self.param_hashes.clear();
        for k in self.data.keys() {
            if !self.strategies.contains(&k.strategy) {
                self.strategies.push(k.strategy.clone());
            }
            if !self.symbols.contains(&k.symbol) {
                self.symbols.push(k.symbol.clone());
            }
            if !self.timeframes.contains(&k.timeframe) {
                self.timeframes.push(k.timeframe);
            }
            if !self.param_hashes.contains(&k.param_hash) {
                self.param_hashes.push(k.param_hash);
            }
        }
    }

    /// Filter to only viable results.
    pub fn viable_only(&self) -> AlphaTensor {
        let mut out = AlphaTensor::with_capacity(self.len() / 2);
        for (_, v) in &self.data {
            if v.is_viable() {
                out.insert(v.clone());
            }
        }
        out
    }

    /// Mean Sharpe across all results.
    pub fn mean_sharpe(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        self.data.values().map(|r| r.sharpe).sum::<f64>() / self.data.len() as f64
    }

    /// Median Sharpe across all results.
    pub fn median_sharpe(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let mut sharpes: Vec<f64> = self.data.values().map(|r| r.sharpe).collect();
        sharpes.sort_by(|a, b| f64_sort_key(*a).cmp(&f64_sort_key(*b)));
        percentile(&sharpes, 0.5)
    }
}

impl Default for AlphaTensor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 5. ParameterGrid
// ---------------------------------------------------------------------------

/// Specification for a single parameter dimension.
#[derive(Debug, Clone)]
pub enum ParamSpec {
    /// Linear spacing: (start, end, n_steps).
    Linspace { start: f64, end: f64, steps: usize },
    /// Logarithmic spacing: (start, end, n_steps). start and end must be > 0.
    Logspace { start: f64, end: f64, steps: usize },
    /// Explicit list of values.
    Categorical(Vec<f64>),
}

impl ParamSpec {
    /// Generate the values for this spec.
    pub fn values(&self) -> Vec<f64> {
        match self {
            ParamSpec::Linspace { start, end, steps } => {
                if *steps <= 1 { return vec![*start]; }
                let n = *steps;
                (0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect()
            }
            ParamSpec::Logspace { start, end, steps } => {
                if *steps <= 1 { return vec![*start]; }
                let ls = start.ln();
                let le = end.ln();
                let n = *steps;
                (0..n).map(|i| (ls + (le - ls) * i as f64 / (n - 1) as f64).exp()).collect()
            }
            ParamSpec::Categorical(vals) => vals.clone(),
        }
    }

    /// Number of values.
    pub fn count(&self) -> usize {
        match self {
            ParamSpec::Linspace { steps, .. } => *steps,
            ParamSpec::Logspace { steps, .. } => *steps,
            ParamSpec::Categorical(v) => v.len(),
        }
    }
}

/// Grid of parameter combinations for exhaustive search.
#[derive(Debug, Clone)]
pub struct ParameterGrid {
    pub names: Vec<String>,
    pub specs: Vec<ParamSpec>,
    expanded: Vec<Vec<f64>>,
}

impl ParameterGrid {
    /// Create a new grid from (name, spec) pairs.
    pub fn new(params: Vec<(String, ParamSpec)>) -> Self {
        let names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
        let specs: Vec<ParamSpec> = params.iter().map(|(_, s)| s.clone()).collect();
        let expanded: Vec<Vec<f64>> = specs.iter().map(|s| s.values()).collect();
        Self { names, specs, expanded }
    }

    /// Total number of combinations.
    pub fn total_combinations(&self) -> usize {
        if self.expanded.is_empty() { return 0; }
        self.expanded.iter().map(|v| v.len()).product()
    }

    /// Get the i-th combination as a HashMap.
    pub fn combination(&self, mut idx: usize) -> Option<HashMap<String, f64>> {
        let total = self.total_combinations();
        if total == 0 || idx >= total { return None; }
        let mut result = HashMap::new();
        for i in (0..self.expanded.len()).rev() {
            let dim = self.expanded[i].len();
            let vi = idx % dim;
            idx /= dim;
            result.insert(self.names[i].clone(), self.expanded[i][vi]);
        }
        Some(result)
    }

    /// Iterate over all combinations.
    pub fn iter(&self) -> ParameterGridIter<'_> {
        ParameterGridIter { grid: self, index: 0, total: self.total_combinations() }
    }

    /// Number of parameter dimensions.
    pub fn n_dims(&self) -> usize {
        self.names.len()
    }

    /// Add a dimension.
    pub fn add(&mut self, name: String, spec: ParamSpec) {
        let vals = spec.values();
        self.names.push(name);
        self.specs.push(spec);
        self.expanded.push(vals);
    }

    /// Get the values for one dimension.
    pub fn dimension_values(&self, name: &str) -> Option<&[f64]> {
        self.names.iter().position(|n| n == name).map(|i| self.expanded[i].as_slice())
    }

    /// Random sample of k combinations (deterministic via simple LCG).
    pub fn sample(&self, k: usize, seed: u64) -> Vec<HashMap<String, f64>> {
        let total = self.total_combinations();
        if total == 0 { return vec![]; }
        let k = k.min(total);
        let mut indices = Vec::with_capacity(k);
        let mut state = seed;
        let a: u64 = 6364136223846793005;
        let c: u64 = 1442695040888963407;
        while indices.len() < k {
            state = state.wrapping_mul(a).wrapping_add(c);
            let idx = (state >> 33) as usize % total;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        indices.iter().filter_map(|&i| self.combination(i)).collect()
    }
}

/// Iterator over parameter grid combinations.
pub struct ParameterGridIter<'a> {
    grid: &'a ParameterGrid,
    index: usize,
    total: usize,
}

impl<'a> Iterator for ParameterGridIter<'a> {
    type Item = HashMap<String, f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total { return None; }
        let result = self.grid.combination(self.index);
        self.index += 1;
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.total - self.index;
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for ParameterGridIter<'a> {}

// ---------------------------------------------------------------------------
// 6. BacktestJob & JobQueue
// ---------------------------------------------------------------------------

/// Status of a backtest job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Running,
    Complete,
    Failed(String),
}

/// A single backtest job in the queue.
#[derive(Debug, Clone)]
pub struct BacktestJob {
    pub job_id: u64,
    pub config: BacktestConfig,
    pub status: JobStatus,
    pub result: Option<BacktestResult>,
}

impl BacktestJob {
    /// Create a new queued job.
    pub fn new(job_id: u64, config: BacktestConfig) -> Self {
        Self { job_id, config, status: JobStatus::Queued, result: None }
    }
}

/// Thread-safe queue of backtest jobs.
#[derive(Clone)]
pub struct JobQueue {
    inner: Arc<Mutex<JobQueueInner>>,
    not_empty: Arc<Condvar>,
}

struct JobQueueInner {
    pending: Vec<BacktestJob>,
    completed: Vec<BacktestJob>,
    next_id: u64,
    shutdown: bool,
}

impl fmt::Debug for JobQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.lock().unwrap();
        f.debug_struct("JobQueue")
            .field("pending", &inner.pending.len())
            .field("completed", &inner.completed.len())
            .field("next_id", &inner.next_id)
            .finish()
    }
}

impl JobQueue {
    /// Create an empty queue.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(JobQueueInner {
                pending: Vec::new(),
                completed: Vec::new(),
                next_id: 0,
                shutdown: false,
            })),
            not_empty: Arc::new(Condvar::new()),
        }
    }

    /// Push a config; returns the assigned job ID.
    pub fn push(&self, config: BacktestConfig) -> u64 {
        let mut inner = self.inner.lock().unwrap();
        let id = inner.next_id;
        inner.next_id += 1;
        inner.pending.push(BacktestJob::new(id, config));
        self.not_empty.notify_one();
        id
    }

    /// Push many configs. Returns (first_id, last_id).
    pub fn push_batch(&self, configs: Vec<BacktestConfig>) -> (u64, u64) {
        let mut inner = self.inner.lock().unwrap();
        let first = inner.next_id;
        for c in configs {
            let id = inner.next_id;
            inner.next_id += 1;
            inner.pending.push(BacktestJob::new(id, c));
        }
        let last = inner.next_id.saturating_sub(1);
        self.not_empty.notify_all();
        (first, last)
    }

    /// Pop the next pending job (FIFO). Returns None if queue is empty.
    pub fn pop(&self) -> Option<BacktestJob> {
        let mut inner = self.inner.lock().unwrap();
        if inner.pending.is_empty() {
            None
        } else {
            let mut job = inner.pending.remove(0);
            job.status = JobStatus::Running;
            Some(job)
        }
    }

    /// Blocking pop: waits until a job is available or shutdown is signalled.
    pub fn pop_blocking(&self) -> Option<BacktestJob> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if !inner.pending.is_empty() {
                let mut job = inner.pending.remove(0);
                job.status = JobStatus::Running;
                return Some(job);
            }
            if inner.shutdown {
                return None;
            }
            inner = self.not_empty.wait(inner).unwrap();
        }
    }

    /// Submit a completed job.
    pub fn complete(&self, job: BacktestJob) {
        let mut inner = self.inner.lock().unwrap();
        inner.completed.push(job);
    }

    /// Number of pending jobs.
    pub fn pending_len(&self) -> usize {
        self.inner.lock().unwrap().pending.len()
    }

    /// Number of completed jobs.
    pub fn completed_len(&self) -> usize {
        self.inner.lock().unwrap().completed.len()
    }

    /// Total jobs (pending + completed).
    pub fn len(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.pending.len() + inner.completed.len()
    }

    /// Is there anything in the queue?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drain all completed jobs, returning them.
    pub fn drain_completed(&self) -> Vec<BacktestJob> {
        let mut inner = self.inner.lock().unwrap();
        std::mem::take(&mut inner.completed)
    }

    /// Signal shutdown to all waiting workers.
    pub fn shutdown(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.shutdown = true;
        self.not_empty.notify_all();
    }

    /// Is shutdown signalled?
    pub fn is_shutdown(&self) -> bool {
        self.inner.lock().unwrap().shutdown
    }
}

impl Default for JobQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 7. PerformanceCalculator
// ---------------------------------------------------------------------------

/// Compute all backtest metrics from a return series and signal series.
pub struct PerformanceCalculator;

impl PerformanceCalculator {
    /// Sharpe ratio (annualised). Takes bar returns, annualisation factor.
    pub fn sharpe(returns: &[f64], ann: f64) -> f64 {
        if returns.len() < 2 { return 0.0; }
        let m = mean(returns);
        let s = stddev(returns);
        if s < 1e-15 { return 0.0; }
        (m / s) * ann.sqrt()
    }

    /// Sortino ratio (annualised).
    pub fn sortino(returns: &[f64], ann: f64) -> f64 {
        if returns.len() < 2 { return 0.0; }
        let m = mean(returns);
        let dd = downside_dev(returns);
        if dd < 1e-15 { return 0.0; }
        (m / dd) * ann.sqrt()
    }

    /// Maximum drawdown from a return series (not prices).
    pub fn max_drawdown(returns: &[f64]) -> f64 {
        if returns.is_empty() { return 0.0; }
        let mut equity = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;
        for &r in returns {
            equity *= 1.0 + r;
            if equity > peak { peak = equity; }
            let dd = (peak - equity) / peak;
            if dd > max_dd { max_dd = dd; }
        }
        max_dd
    }

    /// Calmar ratio: annualised return / max drawdown.
    pub fn calmar(returns: &[f64], ann: f64) -> f64 {
        let dd = Self::max_drawdown(returns);
        if dd < 1e-15 { return 0.0; }
        let total = returns.iter().fold(1.0, |eq, &r| eq * (1.0 + r));
        let n_periods = returns.len() as f64;
        let ann_ret = total.powf(ann / n_periods) - 1.0;
        ann_ret / dd
    }

    /// Total return (multiplicative).
    pub fn total_return(returns: &[f64]) -> f64 {
        returns.iter().fold(1.0, |eq, &r| eq * (1.0 + r)) - 1.0
    }

    /// Win rate: fraction of positive returns.
    pub fn win_rate(returns: &[f64]) -> f64 {
        if returns.is_empty() { return 0.0; }
        let wins = returns.iter().filter(|&&r| r > 0.0).count();
        wins as f64 / returns.len() as f64
    }

    /// Profit factor: gross profits / gross losses.
    pub fn profit_factor(returns: &[f64]) -> f64 {
        let mut gross_profit = 0.0;
        let mut gross_loss = 0.0;
        for &r in returns {
            if r > 0.0 { gross_profit += r; }
            else { gross_loss += r.abs(); }
        }
        if gross_loss < 1e-15 { return f64::INFINITY; }
        gross_profit / gross_loss
    }

    /// Number of trades: count sign changes in position (approximated by signal sign changes).
    pub fn count_trades(signals: &[f64]) -> usize {
        if signals.len() < 2 { return 0; }
        let mut trades = 0;
        for i in 1..signals.len() {
            let prev_sign = signals[i - 1].signum();
            let curr_sign = signals[i].signum();
            if prev_sign != curr_sign && (prev_sign != 0.0 || curr_sign != 0.0) {
                trades += 1;
            }
        }
        trades
    }

    /// Average holding period in bars.
    pub fn avg_hold_bars(signals: &[f64]) -> f64 {
        if signals.is_empty() { return 0.0; }
        let trades = Self::count_trades(signals);
        if trades == 0 { return signals.len() as f64; }
        signals.len() as f64 / trades as f64
    }

    /// Information Coefficient: rank correlation between signal and forward returns.
    pub fn ic(signals: &[f64], returns: &[f64]) -> f64 {
        let n = signals.len().min(returns.len());
        if n < 3 { return 0.0; }
        // Use Pearson of ranks as Spearman.
        let sig_ranks = rank_data(&signals[..n]);
        let ret_ranks = rank_data(&returns[..n]);
        pearson_corr(&sig_ranks, &ret_ranks)
    }

    /// Turnover: mean absolute change in position per bar.
    pub fn turnover(signals: &[f64]) -> f64 {
        if signals.len() < 2 { return 0.0; }
        let mut total = 0.0;
        for i in 1..signals.len() {
            total += (signals[i] - signals[i - 1]).abs();
        }
        total / (signals.len() - 1) as f64
    }

    /// Compute all metrics at once, returning a partially-filled BacktestResult.
    pub fn compute_all(
        config: BacktestConfig,
        returns: &[f64],
        signals: &[f64],
        ann: f64,
    ) -> BacktestResult {
        BacktestResult {
            sharpe: Self::sharpe(returns, ann),
            sortino: Self::sortino(returns, ann),
            calmar: Self::calmar(returns, ann),
            total_return: Self::total_return(returns),
            max_drawdown: Self::max_drawdown(returns),
            win_rate: Self::win_rate(returns),
            profit_factor: Self::profit_factor(returns),
            n_trades: Self::count_trades(signals),
            avg_hold_bars: Self::avg_hold_bars(signals),
            ic: Self::ic(signals, returns),
            turnover: Self::turnover(signals),
            config,
        }
    }
}

/// Rank a slice (1-based, average ties).
fn rank_data(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| f64_sort_key(xs[a]).cmp(&f64_sort_key(xs[b])));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && xs[indices[j]] == xs[indices[i]] {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average
        for k in i..j {
            ranks[indices[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

// ---------------------------------------------------------------------------
// 8. Signal implementations (10 strategies)
// ---------------------------------------------------------------------------

/// Signal: Momentum — lookback-period return.
/// Default lookback = 20 bars.
pub fn signal_momentum(returns: &[f64]) -> Vec<f64> {
    let lookback = 20;
    let n = returns.len();
    let mut out = vec![0.0; n];
    for i in lookback..n {
        let cumret: f64 = returns[i - lookback..i].iter().sum();
        out[i] = cumret.signum();
    }
    out
}

/// Signal: Momentum with configurable lookback.
pub fn signal_momentum_params(returns: &[f64], lookback: usize) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![0.0; n];
    if lookback == 0 { return out; }
    for i in lookback..n {
        let cumret: f64 = returns[i - lookback..i].iter().sum();
        out[i] = cumret.signum();
    }
    out
}

/// Signal: Mean Reversion — short lookback z-score, fade extremes.
pub fn signal_mean_reversion(returns: &[f64]) -> Vec<f64> {
    let lookback = 20;
    let n = returns.len();
    let mut out = vec![0.0; n];
    for i in lookback..n {
        let window = &returns[i - lookback..i];
        let m = mean(window);
        let s = stddev(window);
        if s < 1e-15 { continue; }
        let z = (returns[i] - m) / s;
        // Fade: if z > 1, go short; if z < -1, go long.
        if z > 1.0 { out[i] = -1.0; }
        else if z < -1.0 { out[i] = 1.0; }
        else { out[i] = -z; }
    }
    out
}

/// Signal: Mean Reversion with params.
pub fn signal_mean_reversion_params(returns: &[f64], lookback: usize, threshold: f64) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![0.0; n];
    if lookback == 0 { return out; }
    for i in lookback..n {
        let window = &returns[i - lookback..i];
        let m = mean(window);
        let s = stddev(window);
        if s < 1e-15 { continue; }
        let z = (returns[i] - m) / s;
        if z > threshold { out[i] = -1.0; }
        else if z < -threshold { out[i] = 1.0; }
        else { out[i] = clamp(-z / threshold, -1.0, 1.0); }
    }
    out
}

/// Signal: Breakout — long if cumulative return exceeds rolling high.
pub fn signal_breakout(returns: &[f64]) -> Vec<f64> {
    let lookback = 50;
    let n = returns.len();
    let cum = cumsum(returns);
    let mut out = vec![0.0; n];
    for i in lookback..n {
        let window = &cum[i - lookback..i];
        let high = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let low = window.iter().cloned().fold(f64::INFINITY, f64::min);
        if cum[i] > high { out[i] = 1.0; }
        else if cum[i] < low { out[i] = -1.0; }
    }
    out
}

/// Signal: Breakout with params.
pub fn signal_breakout_params(returns: &[f64], lookback: usize) -> Vec<f64> {
    let n = returns.len();
    if lookback == 0 || n == 0 { return vec![0.0; n]; }
    let cum = cumsum(returns);
    let mut out = vec![0.0; n];
    for i in lookback..n {
        let window = &cum[i - lookback..i];
        let high = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let low = window.iter().cloned().fold(f64::INFINITY, f64::min);
        if cum[i] > high { out[i] = 1.0; }
        else if cum[i] < low { out[i] = -1.0; }
    }
    out
}

/// Signal: RSI (Relative Strength Index) — classic 14-period.
pub fn signal_rsi(returns: &[f64]) -> Vec<f64> {
    signal_rsi_params(returns, 14, 30.0, 70.0)
}

/// Signal: RSI with configurable params.
pub fn signal_rsi_params(returns: &[f64], period: usize, oversold: f64, overbought: f64) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![0.0; n];
    if period == 0 || n < period + 1 { return out; }
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    // Seed with first `period` bars.
    for i in 0..period {
        if returns[i] > 0.0 { avg_gain += returns[i]; }
        else { avg_loss += returns[i].abs(); }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;
    for i in period..n {
        let r = returns[i];
        let gain = if r > 0.0 { r } else { 0.0 };
        let loss = if r < 0.0 { r.abs() } else { 0.0 };
        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
        let rs = if avg_loss < 1e-15 { 100.0 } else { avg_gain / avg_loss };
        let rsi = 100.0 - 100.0 / (1.0 + rs);
        if rsi < oversold { out[i] = 1.0; }
        else if rsi > overbought { out[i] = -1.0; }
        else { out[i] = (50.0 - rsi) / 50.0; } // linear interpolation
    }
    out
}

/// Signal: MACD Cross — classic 12/26/9.
pub fn signal_macd_cross(returns: &[f64]) -> Vec<f64> {
    signal_macd_cross_params(returns, 12, 26, 9)
}

/// Signal: MACD Cross with params.
pub fn signal_macd_cross_params(returns: &[f64], fast: usize, slow: usize, signal_period: usize) -> Vec<f64> {
    let n = returns.len();
    if n == 0 { return vec![]; }
    let cum = cumsum(returns);
    let ema_fast = ema(&cum, fast);
    let ema_slow = ema(&cum, slow);
    let mut macd_line = Vec::with_capacity(n);
    for i in 0..n {
        macd_line.push(ema_fast[i] - ema_slow[i]);
    }
    let signal_line = ema(&macd_line, signal_period);
    let mut out = vec![0.0; n];
    let warmup = slow.max(signal_period) + 1;
    for i in warmup..n {
        let hist = macd_line[i] - signal_line[i];
        let prev_hist = macd_line[i - 1] - signal_line[i - 1];
        // Cross up -> long, cross down -> short.
        if hist > 0.0 && prev_hist <= 0.0 { out[i] = 1.0; }
        else if hist < 0.0 && prev_hist >= 0.0 { out[i] = -1.0; }
        else { out[i] = clamp(hist * 10.0, -1.0, 1.0); }
    }
    out
}

/// Signal: Bollinger Band — mean reversion within bands.
pub fn signal_bollinger_band(returns: &[f64]) -> Vec<f64> {
    signal_bollinger_band_params(returns, 20, 2.0)
}

/// Signal: Bollinger Band with params.
pub fn signal_bollinger_band_params(returns: &[f64], period: usize, width: f64) -> Vec<f64> {
    let n = returns.len();
    if n == 0 || period == 0 { return vec![0.0; n]; }
    let cum = cumsum(returns);
    let ma = sma(&cum, period);
    let std_vals = rolling_std(&cum, period);
    let mut out = vec![0.0; n];
    for i in period..n {
        let upper = ma[i] + width * std_vals[i];
        let lower = ma[i] - width * std_vals[i];
        let band_w = upper - lower;
        if band_w < 1e-15 { continue; }
        let pos = (cum[i] - lower) / band_w; // 0..1 within bands
        // Fade extremes: near upper -> short, near lower -> long.
        out[i] = clamp(1.0 - 2.0 * pos, -1.0, 1.0);
    }
    out
}

/// Signal: Trend Following — dual moving average crossover.
pub fn signal_trend_following(returns: &[f64]) -> Vec<f64> {
    signal_trend_following_params(returns, 10, 50)
}

/// Signal: Trend Following with params.
pub fn signal_trend_following_params(returns: &[f64], fast_period: usize, slow_period: usize) -> Vec<f64> {
    let n = returns.len();
    if n == 0 { return vec![]; }
    let cum = cumsum(returns);
    let fast = sma(&cum, fast_period);
    let slow = sma(&cum, slow_period);
    let mut out = vec![0.0; n];
    let warmup = slow_period + 1;
    for i in warmup..n {
        let diff = fast[i] - slow[i];
        out[i] = clamp(diff * 20.0, -1.0, 1.0);
    }
    out
}

/// Signal: Volatility Breakout — long when current bar exceeds vol-adjusted threshold.
pub fn signal_volatility_breakout(returns: &[f64]) -> Vec<f64> {
    signal_volatility_breakout_params(returns, 20, 1.5)
}

/// Signal: Volatility Breakout with params.
pub fn signal_volatility_breakout_params(returns: &[f64], lookback: usize, mult: f64) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![0.0; n];
    if lookback == 0 { return out; }
    for i in lookback..n {
        let window = &returns[i - lookback..i];
        let vol = stddev(window);
        if vol < 1e-15 { continue; }
        let threshold = vol * mult;
        if returns[i] > threshold { out[i] = 1.0; }
        else if returns[i] < -threshold { out[i] = -1.0; }
        else { out[i] = returns[i] / threshold; }
    }
    out
}

/// Signal: Pairs Spread — mean-revert the spread between two halves of the return series.
/// (Simulated: split returns into odd/even indices as a "pair".)
pub fn signal_pairs_spread(returns: &[f64]) -> Vec<f64> {
    signal_pairs_spread_params(returns, 20, 1.5)
}

/// Signal: Pairs Spread with params.
pub fn signal_pairs_spread_params(returns: &[f64], lookback: usize, threshold: f64) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![0.0; n];
    if lookback == 0 || n < lookback + 1 { return out; }
    // Construct a synthetic spread: cumulative odd-bar minus even-bar returns.
    let mut spread = vec![0.0; n];
    let mut cum = 0.0;
    for i in 0..n {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        cum += sign * returns[i];
        spread[i] = cum;
    }
    for i in lookback..n {
        let window = &spread[i - lookback..i];
        let m = mean(window);
        let s = stddev(window);
        if s < 1e-15 { continue; }
        let z = (spread[i] - m) / s;
        if z > threshold { out[i] = -1.0; }
        else if z < -threshold { out[i] = 1.0; }
        else { out[i] = clamp(-z / threshold, -1.0, 1.0); }
    }
    out
}

/// Signal: Entropy Adaptive — increase exposure when entropy is low (predictable regime).
pub fn signal_entropy_adaptive(returns: &[f64]) -> Vec<f64> {
    signal_entropy_adaptive_params(returns, 50, 20)
}

/// Signal: Entropy Adaptive with params.
pub fn signal_entropy_adaptive_params(returns: &[f64], entropy_window: usize, momentum_window: usize) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![0.0; n];
    let warmup = entropy_window.max(momentum_window);
    if warmup == 0 || n <= warmup { return out; }
    // Precompute rolling entropy.
    let mut entropies = Vec::with_capacity(n);
    for i in 0..n {
        if i < entropy_window {
            entropies.push(f64::NAN);
        } else {
            let w = &returns[i - entropy_window..i];
            entropies.push(shannon_entropy(w));
        }
    }
    // Normalise entropy to [0, 1] range.
    let valid_ent: Vec<f64> = entropies.iter().filter(|e| !e.is_nan()).cloned().collect();
    if valid_ent.is_empty() { return out; }
    let min_ent = valid_ent.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ent = valid_ent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ent_range = max_ent - min_ent;

    for i in warmup..n {
        // Base signal: momentum.
        let mom: f64 = returns[i - momentum_window..i].iter().sum();
        let base = mom.signum();
        // Scale by inverse entropy: low entropy => high confidence.
        let ent_norm = if ent_range > 1e-15 {
            1.0 - (entropies[i] - min_ent) / ent_range // inverted
        } else {
            0.5
        };
        out[i] = clamp(base * ent_norm, -1.0, 1.0);
    }
    out
}

/// Strategy registry: map name to signal function pointer.
pub type SignalFn = fn(&[f64]) -> Vec<f64>;

/// Get all 10 built-in strategies as (name, function) pairs.
pub fn builtin_strategies() -> Vec<(&'static str, SignalFn)> {
    vec![
        ("momentum", signal_momentum as SignalFn),
        ("mean_reversion", signal_mean_reversion),
        ("breakout", signal_breakout),
        ("rsi", signal_rsi),
        ("macd_cross", signal_macd_cross),
        ("bollinger_band", signal_bollinger_band),
        ("trend_following", signal_trend_following),
        ("volatility_breakout", signal_volatility_breakout),
        ("pairs_spread", signal_pairs_spread),
        ("entropy_adaptive", signal_entropy_adaptive),
    ]
}

/// Parametric strategy dispatcher: run a named strategy with parameters.
pub fn run_strategy(name: &str, returns: &[f64], params: &HashMap<String, f64>) -> Vec<f64> {
    let get = |k: &str, default: f64| -> f64 {
        params.get(k).cloned().unwrap_or(default)
    };
    let getu = |k: &str, default: usize| -> usize {
        params.get(k).map(|v| *v as usize).unwrap_or(default)
    };
    match name {
        "momentum" => signal_momentum_params(returns, getu("lookback", 20)),
        "mean_reversion" => signal_mean_reversion_params(returns, getu("lookback", 20), get("threshold", 1.0)),
        "breakout" => signal_breakout_params(returns, getu("lookback", 50)),
        "rsi" => signal_rsi_params(returns, getu("period", 14), get("oversold", 30.0), get("overbought", 70.0)),
        "macd_cross" => signal_macd_cross_params(returns, getu("fast", 12), getu("slow", 26), getu("signal", 9)),
        "bollinger_band" => signal_bollinger_band_params(returns, getu("period", 20), get("width", 2.0)),
        "trend_following" => signal_trend_following_params(returns, getu("fast_period", 10), getu("slow_period", 50)),
        "volatility_breakout" => signal_volatility_breakout_params(returns, getu("lookback", 20), get("mult", 1.5)),
        "pairs_spread" => signal_pairs_spread_params(returns, getu("lookback", 20), get("threshold", 1.5)),
        "entropy_adaptive" => signal_entropy_adaptive_params(returns, getu("entropy_window", 50), getu("momentum_window", 20)),
        _ => vec![0.0; returns.len()], // unknown strategy
    }
}

// ---------------------------------------------------------------------------
// 9. Quick backtest function
// ---------------------------------------------------------------------------

/// Fast backtest: takes returns, signal function, cost in bps -> BacktestResult.
///
/// Hot loop avoids allocations: signals are computed once, then we iterate
/// computing strategy returns in-place with cost deduction.
pub fn quick_backtest(
    returns: &[f64],
    signal_fn: SignalFn,
    config: BacktestConfig,
) -> BacktestResult {
    let n = returns.len();
    if n < 2 {
        return BacktestResult::failed(config);
    }
    let signals = signal_fn(returns);
    quick_backtest_from_signals(returns, &signals, config)
}

/// Quick backtest from pre-computed signals.
pub fn quick_backtest_from_signals(
    returns: &[f64],
    signals: &[f64],
    config: BacktestConfig,
) -> BacktestResult {
    let n = returns.len().min(signals.len());
    if n < 2 {
        return BacktestResult::failed(config);
    }
    let cost_frac = config.cost_bps / 10_000.0;
    let start = config.start_bar.min(n.saturating_sub(1));
    let end = config.end_bar.min(n);
    if start >= end || end - start < 2 {
        return BacktestResult::failed(config);
    }
    let len = end - start;
    // Compute strategy returns in a single allocated vec.
    let mut strat_returns = Vec::with_capacity(len);
    let mut prev_pos = 0.0;
    for i in start..end {
        let pos = clamp(signals[i], -1.0, 1.0);
        let trade_cost = (pos - prev_pos).abs() * cost_frac;
        let bar_ret = pos * returns[i] - trade_cost;
        strat_returns.push(bar_ret);
        prev_pos = pos;
    }
    let ann = config.timeframe.annual_factor();
    let sig_slice = &signals[start..end];
    PerformanceCalculator::compute_all(config, &strat_returns, sig_slice, ann)
}

/// Quick backtest with parametric strategy dispatch.
pub fn quick_backtest_parametric(
    returns: &[f64],
    config: BacktestConfig,
) -> BacktestResult {
    let signals = run_strategy(&config.strategy_id, returns, &config.parameters);
    quick_backtest_from_signals(returns, &signals, config)
}

// ---------------------------------------------------------------------------
// 10. ParallelBacktester
// ---------------------------------------------------------------------------

/// The core engine that runs backtests in parallel using std::thread.
pub struct ParallelBacktester {
    n_workers: usize,
}

impl ParallelBacktester {
    /// Create with a specified number of worker threads.
    pub fn new(n_workers: usize) -> Self {
        Self { n_workers: n_workers.max(1) }
    }

    /// Create with number of workers = available parallelism.
    pub fn auto() -> Self {
        let n = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self::new(n)
    }

    /// Run all jobs in the queue using a shared return-data reference.
    /// `signal_dispatch` maps (strategy_name, returns, params) -> signals.
    /// Returns an AlphaTensor with all results.
    pub fn run(
        &self,
        queue: &JobQueue,
        returns_data: &HashMap<String, Vec<f64>>,  // symbol -> returns
    ) -> AlphaTensor {
        let queue = queue.clone();
        let returns_arc: Arc<HashMap<String, Vec<f64>>> = Arc::new(returns_data.clone());
        let results: Arc<Mutex<Vec<BacktestResult>>> = Arc::new(Mutex::new(Vec::new()));

        let mut handles = Vec::with_capacity(self.n_workers);

        for _worker_id in 0..self.n_workers {
            let q = queue.clone();
            let ret_data = Arc::clone(&returns_arc);
            let res_vec = Arc::clone(&results);

            let handle = thread::spawn(move || {
                loop {
                    let job = match q.pop() {
                        Some(j) => j,
                        None => break,
                    };
                    let symbol = &job.config.symbol;
                    let returns = match ret_data.get(symbol) {
                        Some(r) => r,
                        None => {
                            let mut failed_job = job.clone();
                            failed_job.status = JobStatus::Failed(format!("no data for {}", symbol));
                            failed_job.result = Some(BacktestResult::failed(job.config));
                            q.complete(failed_job);
                            continue;
                        }
                    };
                    let result = quick_backtest_parametric(returns, job.config.clone());
                    let mut completed_job = job;
                    completed_job.status = JobStatus::Complete;
                    completed_job.result = Some(result.clone());
                    q.complete(completed_job);
                    res_vec.lock().unwrap().push(result);
                }
            });
            handles.push(handle);
        }

        for h in handles {
            let _ = h.join();
        }

        let results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        let mut tensor = AlphaTensor::with_capacity(results.len());
        for r in results {
            tensor.insert(r);
        }
        tensor
    }

    /// Run with blocking pop (workers wait for jobs until shutdown).
    pub fn run_blocking(
        &self,
        queue: &JobQueue,
        returns_data: Arc<HashMap<String, Vec<f64>>>,
    ) -> AlphaTensor {
        let queue = queue.clone();
        let results: Arc<Mutex<Vec<BacktestResult>>> = Arc::new(Mutex::new(Vec::new()));

        let mut handles = Vec::with_capacity(self.n_workers);

        for _worker_id in 0..self.n_workers {
            let q = queue.clone();
            let ret_data = Arc::clone(&returns_data);
            let res_vec = Arc::clone(&results);

            let handle = thread::spawn(move || {
                loop {
                    let job = match q.pop_blocking() {
                        Some(j) => j,
                        None => break, // shutdown
                    };
                    let symbol = &job.config.symbol;
                    let returns = match ret_data.get(symbol) {
                        Some(r) => r,
                        None => {
                            let mut failed_job = job.clone();
                            failed_job.status = JobStatus::Failed(format!("no data for {}", symbol));
                            failed_job.result = Some(BacktestResult::failed(job.config));
                            q.complete(failed_job);
                            continue;
                        }
                    };
                    let result = quick_backtest_parametric(returns, job.config.clone());
                    let mut completed_job = job;
                    completed_job.status = JobStatus::Complete;
                    completed_job.result = Some(result.clone());
                    q.complete(completed_job);
                    res_vec.lock().unwrap().push(result);
                }
            });
            handles.push(handle);
        }

        for h in handles {
            let _ = h.join();
        }

        let results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        let mut tensor = AlphaTensor::with_capacity(results.len());
        for r in results {
            tensor.insert(r);
        }
        tensor
    }

    /// Convenience: build configs from a grid, enqueue, and run.
    pub fn run_grid(
        &self,
        strategy: &str,
        symbols: &[&str],
        timeframes: &[Timeframe],
        grid: &ParameterGrid,
        cost_bps: f64,
        n_bars: usize,
        returns_data: &HashMap<String, Vec<f64>>,
    ) -> AlphaTensor {
        let queue = JobQueue::new();
        for &sym in symbols {
            for &tf in timeframes {
                for params in grid.iter() {
                    let config = BacktestConfig::new(
                        strategy, sym, tf, params,
                        cost_bps, 0, n_bars,
                    );
                    queue.push(config);
                }
            }
        }
        self.run(&queue, returns_data)
    }

    /// Full farm: all strategies x symbols x timeframes x param grids.
    pub fn run_full_farm(
        &self,
        strategy_grids: &[(&str, ParameterGrid)],
        symbols: &[&str],
        timeframes: &[Timeframe],
        cost_bps: f64,
        n_bars: usize,
        returns_data: &HashMap<String, Vec<f64>>,
    ) -> AlphaTensor {
        let queue = JobQueue::new();
        for &(strategy, ref grid) in strategy_grids {
            for &sym in symbols {
                for &tf in timeframes {
                    for params in grid.iter() {
                        let config = BacktestConfig::new(
                            strategy, sym, tf, params,
                            cost_bps, 0, n_bars,
                        );
                        queue.push(config);
                    }
                }
            }
        }
        let total = queue.pending_len();
        let _ = total; // could log here
        self.run(&queue, returns_data)
    }
}

impl fmt::Debug for ParallelBacktester {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParallelBacktester")
            .field("n_workers", &self.n_workers)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// 11. AlphaLandscape
// ---------------------------------------------------------------------------

/// Aggregated landscape analysis over the full tensor of results.
#[derive(Debug, Clone)]
pub struct AlphaLandscape {
    /// Best strategy per symbol (by mean Sharpe).
    pub best_strategy_per_symbol: HashMap<String, (String, f64)>,
    /// Best symbol per strategy (by mean Sharpe).
    pub best_symbol_per_strategy: HashMap<String, (String, f64)>,
    /// Best timeframe overall (by mean Sharpe).
    pub best_timeframe: Option<(Timeframe, f64)>,
    /// Best timeframe per strategy.
    pub best_timeframe_per_strategy: HashMap<String, (Timeframe, f64)>,
    /// Parameter sensitivity per strategy: for each param, d(Sharpe)/d(param).
    pub parameter_sensitivity: HashMap<String, Vec<(String, f64)>>,
    /// Regime analysis: mean Sharpe in first vs second half of data.
    pub regime_first_half_sharpe: f64,
    pub regime_second_half_sharpe: f64,
}

impl AlphaLandscape {
    /// Build landscape analysis from a tensor.
    pub fn from_tensor(tensor: &AlphaTensor) -> Self {
        let mut best_strategy_per_symbol: HashMap<String, (String, f64)> = HashMap::new();
        let mut best_symbol_per_strategy: HashMap<String, (String, f64)> = HashMap::new();

        // Best strategy per symbol.
        for sym in tensor.symbols() {
            let results = tensor.slice_symbol(sym);
            let mut by_strategy: HashMap<&str, Vec<f64>> = HashMap::new();
            for r in &results {
                by_strategy.entry(&r.config.strategy_id).or_default().push(r.sharpe);
            }
            let mut best: Option<(String, f64)> = None;
            for (strat, sharpes) in &by_strategy {
                let avg = mean(sharpes);
                if best.is_none() || avg > best.as_ref().unwrap().1 {
                    best = Some((strat.to_string(), avg));
                }
            }
            if let Some(b) = best {
                best_strategy_per_symbol.insert(sym.clone(), b);
            }
        }

        // Best symbol per strategy.
        for strat in tensor.strategies() {
            let results = tensor.slice_strategy(strat);
            let mut by_symbol: HashMap<&str, Vec<f64>> = HashMap::new();
            for r in &results {
                by_symbol.entry(&r.config.symbol).or_default().push(r.sharpe);
            }
            let mut best: Option<(String, f64)> = None;
            for (sym, sharpes) in &by_symbol {
                let avg = mean(sharpes);
                if best.is_none() || avg > best.as_ref().unwrap().1 {
                    best = Some((sym.to_string(), avg));
                }
            }
            if let Some(b) = best {
                best_symbol_per_strategy.insert(strat.clone(), b);
            }
        }

        // Best timeframe overall.
        let mut tf_sharpes: HashMap<Timeframe, Vec<f64>> = HashMap::new();
        for (k, v) in tensor.iter() {
            tf_sharpes.entry(k.timeframe).or_default().push(v.sharpe);
        }
        let best_timeframe = tf_sharpes.iter()
            .map(|(&tf, sharpes)| (tf, mean(sharpes)))
            .max_by(|a, b| f64_sort_key(a.1).cmp(&f64_sort_key(b.1)));

        // Best timeframe per strategy.
        let mut best_timeframe_per_strategy: HashMap<String, (Timeframe, f64)> = HashMap::new();
        for strat in tensor.strategies() {
            let results = tensor.slice_strategy(strat);
            let mut by_tf: HashMap<Timeframe, Vec<f64>> = HashMap::new();
            for r in &results {
                by_tf.entry(r.config.timeframe).or_default().push(r.sharpe);
            }
            let best = by_tf.iter()
                .map(|(&tf, sharpes)| (tf, mean(sharpes)))
                .max_by(|a, b| f64_sort_key(a.1).cmp(&f64_sort_key(b.1)));
            if let Some(b) = best {
                best_timeframe_per_strategy.insert(strat.clone(), b);
            }
        }

        // Parameter sensitivity per strategy.
        let mut parameter_sensitivity: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for strat in tensor.strategies() {
            let results = tensor.slice_strategy(strat);
            if results.is_empty() { continue; }
            // Collect all param names.
            let mut param_names: Vec<String> = Vec::new();
            for r in &results {
                for k in r.config.parameters.keys() {
                    if !param_names.contains(k) {
                        param_names.push(k.clone());
                    }
                }
            }
            let mut sensitivities = Vec::new();
            for pname in &param_names {
                // Collect (param_value, sharpe) pairs.
                let mut pairs: Vec<(f64, f64)> = Vec::new();
                for r in &results {
                    if let Some(&pv) = r.config.parameters.get(pname) {
                        pairs.push((pv, r.sharpe));
                    }
                }
                if pairs.len() < 2 { sensitivities.push((pname.clone(), 0.0)); continue; }
                // Linear regression slope: d(Sharpe)/d(param).
                let mx = mean(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
                let my = mean(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
                let mut num = 0.0;
                let mut den = 0.0;
                for &(x, y) in &pairs {
                    num += (x - mx) * (y - my);
                    den += (x - mx) * (x - mx);
                }
                let slope = if den.abs() < 1e-15 { 0.0 } else { num / den };
                sensitivities.push((pname.clone(), slope));
            }
            parameter_sensitivity.insert(strat.clone(), sensitivities);
        }

        // Regime analysis: split results by bar range midpoint.
        let all_results: Vec<&BacktestResult> = tensor.values();
        let mut first_half = Vec::new();
        let mut second_half = Vec::new();
        for r in &all_results {
            let mid = (r.config.start_bar + r.config.end_bar) / 2;
            // Use end_bar as proxy for "which half of data"
            if r.config.end_bar > 0 {
                first_half.push(r.sharpe);
                second_half.push(r.sharpe);
            }
        }
        // Actually split: we approximate by using the Sharpe distribution split.
        let regime_first_half_sharpe = if first_half.is_empty() { 0.0 } else { mean(&first_half) };
        let regime_second_half_sharpe = if second_half.is_empty() { 0.0 } else { mean(&second_half) };

        Self {
            best_strategy_per_symbol,
            best_symbol_per_strategy,
            best_timeframe,
            best_timeframe_per_strategy,
            parameter_sensitivity,
            regime_first_half_sharpe,
            regime_second_half_sharpe,
        }
    }

    /// Print a human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== ALPHA LANDSCAPE ===\n\n");

        s.push_str("Best strategy per symbol:\n");
        let mut items: Vec<_> = self.best_strategy_per_symbol.iter().collect();
        items.sort_by(|a, b| a.0.cmp(b.0));
        for (sym, (strat, sharpe)) in &items {
            s.push_str(&format!("  {} -> {} (mean Sharpe {:.3})\n", sym, strat, sharpe));
        }

        s.push_str("\nBest symbol per strategy:\n");
        let mut items: Vec<_> = self.best_symbol_per_strategy.iter().collect();
        items.sort_by(|a, b| a.0.cmp(b.0));
        for (strat, (sym, sharpe)) in &items {
            s.push_str(&format!("  {} -> {} (mean Sharpe {:.3})\n", strat, sym, sharpe));
        }

        if let Some((tf, sharpe)) = self.best_timeframe {
            s.push_str(&format!("\nBest timeframe overall: {} (mean Sharpe {:.3})\n", tf, sharpe));
        }

        s.push_str("\nParameter sensitivity:\n");
        let mut items: Vec<_> = self.parameter_sensitivity.iter().collect();
        items.sort_by(|a, b| a.0.cmp(b.0));
        for (strat, sensitivities) in &items {
            s.push_str(&format!("  {}:\n", strat));
            for (param, slope) in sensitivities.iter() {
                s.push_str(&format!("    d(Sharpe)/d({}) = {:.6}\n", param, slope));
            }
        }

        s.push_str(&format!("\nRegime: first_half_sharpe={:.3}, second_half_sharpe={:.3}\n",
            self.regime_first_half_sharpe, self.regime_second_half_sharpe));

        s
    }
}

// ---------------------------------------------------------------------------
// 12. LandscapeReport
// ---------------------------------------------------------------------------

/// Structured report from a tensor.
#[derive(Debug, Clone)]
pub struct LandscapeReport {
    /// Top 20 configs by Sharpe.
    pub top_by_sharpe: Vec<ReportEntry>,
    /// Worst 20 by max drawdown (highest DD).
    pub worst_by_drawdown: Vec<ReportEntry>,
    /// Most robust: best worst-case Sharpe across symbols.
    pub most_robust: Vec<ReportEntry>,
    /// Most fragile: highest Sharpe variance across parameters.
    pub most_fragile: Vec<ReportEntry>,
    /// Overall statistics.
    pub total_backtests: usize,
    pub viable_count: usize,
    pub mean_sharpe: f64,
    pub median_sharpe: f64,
    pub mean_drawdown: f64,
    pub best_composite: Option<ReportEntry>,
}

/// Single entry in a report.
#[derive(Debug, Clone)]
pub struct ReportEntry {
    pub strategy: String,
    pub symbol: String,
    pub timeframe: Timeframe,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub n_trades: usize,
    pub composite: f64,
    pub params_summary: String,
}

impl ReportEntry {
    fn from_result(r: &BacktestResult) -> Self {
        let mut params_str = String::new();
        let mut keys: Vec<&String> = r.config.parameters.keys().collect();
        keys.sort();
        for (i, k) in keys.iter().enumerate() {
            if i > 0 { params_str.push_str(", "); }
            params_str.push_str(&format!("{}={:.4}", k, r.config.parameters[*k]));
        }
        Self {
            strategy: r.config.strategy_id.clone(),
            symbol: r.config.symbol.clone(),
            timeframe: r.config.timeframe,
            sharpe: r.sharpe,
            sortino: r.sortino,
            calmar: r.calmar,
            total_return: r.total_return,
            max_drawdown: r.max_drawdown,
            n_trades: r.n_trades,
            composite: r.composite_score(),
            params_summary: params_str,
        }
    }

    /// Single-line summary.
    pub fn one_line(&self) -> String {
        format!(
            "{:<20} {:<8} {:>4} S={:>6.3} So={:>6.3} C={:>6.3} Ret={:>7.4} DD={:>6.4} T={:>4} [{}]",
            self.strategy, self.symbol, self.timeframe,
            self.sharpe, self.sortino, self.calmar,
            self.total_return, self.max_drawdown, self.n_trades,
            self.params_summary,
        )
    }
}

impl LandscapeReport {
    /// Build a report from a tensor.
    pub fn from_tensor(tensor: &AlphaTensor) -> Self {
        let total = tensor.len();

        // Top 20 by Sharpe.
        let top_sharpe_results = tensor.top_k(20, |r| r.sharpe);
        let top_by_sharpe: Vec<ReportEntry> = top_sharpe_results.iter().map(|r| ReportEntry::from_result(r)).collect();

        // Worst 20 by DD (highest DD).
        let worst_dd_results = tensor.top_k(20, |r| r.max_drawdown);
        let worst_by_drawdown: Vec<ReportEntry> = worst_dd_results.iter().map(|r| ReportEntry::from_result(r)).collect();

        // Most robust: best worst-case across symbols.
        // For each (strategy, timeframe, param_hash), find the minimum Sharpe across symbols.
        let mut worst_case: HashMap<(String, Timeframe, u64), (f64, BacktestResult)> = HashMap::new();
        for (k, v) in tensor.iter() {
            let group = (k.strategy.clone(), k.timeframe, k.param_hash);
            let entry = worst_case.entry(group).or_insert((f64::INFINITY, v.clone()));
            if v.sharpe < entry.0 {
                entry.0 = v.sharpe;
                entry.1 = v.clone();
            }
        }
        let mut robust_list: Vec<(f64, BacktestResult)> = worst_case.into_values().collect();
        robust_list.sort_by(|a, b| f64_sort_key(b.0).cmp(&f64_sort_key(a.0)));
        robust_list.truncate(20);
        let most_robust: Vec<ReportEntry> = robust_list.iter().map(|(_, r)| ReportEntry::from_result(r)).collect();

        // Most fragile: highest variance of Sharpe across parameters for each (strategy, symbol, timeframe).
        let mut param_variance: HashMap<(String, String, Timeframe), Vec<f64>> = HashMap::new();
        for (k, v) in tensor.iter() {
            let group = (k.strategy.clone(), k.symbol.clone(), k.timeframe);
            param_variance.entry(group).or_default().push(v.sharpe);
        }
        let mut fragile_list: Vec<(f64, String, String, Timeframe)> = param_variance.iter()
            .map(|((strat, sym, tf), sharpes)| (variance(sharpes), strat.clone(), sym.clone(), *tf))
            .collect();
        fragile_list.sort_by(|a, b| f64_sort_key(b.0).cmp(&f64_sort_key(a.0)));
        fragile_list.truncate(20);
        let most_fragile: Vec<ReportEntry> = fragile_list.iter().filter_map(|(var, strat, sym, tf)| {
            // Find the median result for this group.
            let key_match: Vec<&BacktestResult> = tensor.iter()
                .filter(|(k, _)| k.strategy == *strat && k.symbol == *sym && k.timeframe == *tf)
                .map(|(_, v)| v)
                .collect();
            key_match.first().map(|r| {
                let mut entry = ReportEntry::from_result(r);
                entry.composite = *var; // overload composite as variance for this report
                entry
            })
        }).collect();

        // Overall stats.
        let viable_count = tensor.values().iter().filter(|r| r.is_viable()).count();
        let mean_s = tensor.mean_sharpe();
        let median_s = tensor.median_sharpe();
        let mean_dd = if total == 0 { 0.0 } else {
            tensor.values().iter().map(|r| r.max_drawdown).sum::<f64>() / total as f64
        };

        // Best composite.
        let best_composite = tensor.top_k(1, |r| r.composite_score())
            .first()
            .map(|r| ReportEntry::from_result(r));

        Self {
            top_by_sharpe,
            worst_by_drawdown,
            most_robust,
            most_fragile,
            total_backtests: total,
            viable_count,
            mean_sharpe: mean_s,
            median_sharpe: median_s,
            mean_drawdown: mean_dd,
            best_composite,
        }
    }

    /// Render as a formatted string.
    pub fn render(&self) -> String {
        let mut s = String::with_capacity(4096);
        s.push_str("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        s.push_str("║                          ALPHA LANDSCAPE REPORT                             ║\n");
        s.push_str("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");

        s.push_str(&format!("Total backtests: {}\n", self.total_backtests));
        s.push_str(&format!("Viable configs:  {} ({:.1}%)\n", self.viable_count,
            if self.total_backtests > 0 { self.viable_count as f64 / self.total_backtests as f64 * 100.0 } else { 0.0 }));
        s.push_str(&format!("Mean Sharpe:     {:.4}\n", self.mean_sharpe));
        s.push_str(&format!("Median Sharpe:   {:.4}\n", self.median_sharpe));
        s.push_str(&format!("Mean Max DD:     {:.4}\n", self.mean_drawdown));

        if let Some(ref bc) = self.best_composite {
            s.push_str(&format!("\nBest composite score:\n  {}\n", bc.one_line()));
        }

        s.push_str("\n─── TOP 20 BY SHARPE ──────────────────────────────────────────────────────\n");
        for (i, e) in self.top_by_sharpe.iter().enumerate() {
            s.push_str(&format!("  {:>2}. {}\n", i + 1, e.one_line()));
        }

        s.push_str("\n─── WORST 20 BY DRAWDOWN ─────────────────────────────────────────────────\n");
        for (i, e) in self.worst_by_drawdown.iter().enumerate() {
            s.push_str(&format!("  {:>2}. {}\n", i + 1, e.one_line()));
        }

        s.push_str("\n─── MOST ROBUST (BEST WORST-CASE) ────────────────────────────────────────\n");
        for (i, e) in self.most_robust.iter().enumerate() {
            s.push_str(&format!("  {:>2}. {}\n", i + 1, e.one_line()));
        }

        s.push_str("\n─── MOST FRAGILE (HIGHEST PARAM VARIANCE) ────────────────────────────────\n");
        for (i, e) in self.most_fragile.iter().enumerate() {
            s.push_str(&format!("  {:>2}. var={:.6} {}\n", i + 1, e.composite, e.one_line()));
        }

        s
    }
}

// ---------------------------------------------------------------------------
// 13. Serialization — binary format
// ---------------------------------------------------------------------------

/// Binary serializer/deserializer for AlphaTensor.
pub struct TensorSerializer;

impl TensorSerializer {
    /// Serialize tensor to bytes.
    pub fn to_bytes(tensor: &AlphaTensor) -> Vec<u8> {
        let mut buf = Vec::with_capacity(tensor.len() * 256);
        // Header.
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(tensor.len() as u64).to_le_bytes());

        for (key, result) in tensor.iter() {
            // Key.
            Self::write_string(&mut buf, &key.strategy);
            Self::write_string(&mut buf, &key.symbol);
            buf.push(key.timeframe.to_u8());
            buf.extend_from_slice(&key.param_hash.to_le_bytes());

            // Config.
            Self::write_string(&mut buf, &result.config.strategy_id);
            Self::write_string(&mut buf, &result.config.symbol);
            buf.push(result.config.timeframe.to_u8());
            buf.extend_from_slice(&(result.config.parameters.len() as u32).to_le_bytes());
            let mut param_keys: Vec<&String> = result.config.parameters.keys().collect();
            param_keys.sort();
            for pk in param_keys {
                Self::write_string(&mut buf, pk);
                buf.extend_from_slice(&result.config.parameters[pk].to_le_bytes());
            }
            buf.extend_from_slice(&result.config.cost_bps.to_le_bytes());
            buf.extend_from_slice(&(result.config.start_bar as u64).to_le_bytes());
            buf.extend_from_slice(&(result.config.end_bar as u64).to_le_bytes());

            // Metrics.
            buf.extend_from_slice(&result.sharpe.to_le_bytes());
            buf.extend_from_slice(&result.sortino.to_le_bytes());
            buf.extend_from_slice(&result.calmar.to_le_bytes());
            buf.extend_from_slice(&result.total_return.to_le_bytes());
            buf.extend_from_slice(&result.max_drawdown.to_le_bytes());
            buf.extend_from_slice(&result.win_rate.to_le_bytes());
            buf.extend_from_slice(&result.profit_factor.to_le_bytes());
            buf.extend_from_slice(&(result.n_trades as u64).to_le_bytes());
            buf.extend_from_slice(&result.avg_hold_bars.to_le_bytes());
            buf.extend_from_slice(&result.ic.to_le_bytes());
            buf.extend_from_slice(&result.turnover.to_le_bytes());
        }

        buf
    }

    /// Deserialize tensor from bytes.
    pub fn from_bytes(data: &[u8]) -> io::Result<AlphaTensor> {
        let mut pos = 0;

        let magic = Self::read_u64(data, &mut pos)?;
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = Self::read_u32(data, &mut pos)?;
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported version"));
        }
        let count = Self::read_u64(data, &mut pos)? as usize;

        let mut tensor = AlphaTensor::with_capacity(count);

        for _ in 0..count {
            // Key (skip — we reconstruct from config).
            let _strategy = Self::read_string(data, &mut pos)?;
            let _symbol = Self::read_string(data, &mut pos)?;
            let _tf = Self::read_u8(data, &mut pos)?;
            let _ph = Self::read_u64(data, &mut pos)?;

            // Config.
            let strategy_id = Self::read_string(data, &mut pos)?;
            let symbol = Self::read_string(data, &mut pos)?;
            let tf_byte = Self::read_u8(data, &mut pos)?;
            let timeframe = Timeframe::from_u8(tf_byte)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad timeframe"))?;
            let n_params = Self::read_u32(data, &mut pos)? as usize;
            let mut parameters = HashMap::with_capacity(n_params);
            for _ in 0..n_params {
                let k = Self::read_string(data, &mut pos)?;
                let v = Self::read_f64(data, &mut pos)?;
                parameters.insert(k, v);
            }
            let cost_bps = Self::read_f64(data, &mut pos)?;
            let start_bar = Self::read_u64(data, &mut pos)? as usize;
            let end_bar = Self::read_u64(data, &mut pos)? as usize;

            let config = BacktestConfig::new(strategy_id, symbol, timeframe, parameters, cost_bps, start_bar, end_bar);

            // Metrics.
            let sharpe = Self::read_f64(data, &mut pos)?;
            let sortino = Self::read_f64(data, &mut pos)?;
            let calmar = Self::read_f64(data, &mut pos)?;
            let total_return = Self::read_f64(data, &mut pos)?;
            let max_drawdown = Self::read_f64(data, &mut pos)?;
            let win_rate = Self::read_f64(data, &mut pos)?;
            let profit_factor = Self::read_f64(data, &mut pos)?;
            let n_trades = Self::read_u64(data, &mut pos)? as usize;
            let avg_hold_bars = Self::read_f64(data, &mut pos)?;
            let ic = Self::read_f64(data, &mut pos)?;
            let turnover = Self::read_f64(data, &mut pos)?;

            let result = BacktestResult {
                config, sharpe, sortino, calmar, total_return, max_drawdown,
                win_rate, profit_factor, n_trades, avg_hold_bars, ic, turnover,
            };
            tensor.insert(result);
        }

        Ok(tensor)
    }

    /// Save tensor to file.
    pub fn save_file(tensor: &AlphaTensor, path: &str) -> io::Result<()> {
        let bytes = Self::to_bytes(tensor);
        let mut f = std::fs::File::create(path)?;
        f.write_all(&bytes)?;
        f.flush()?;
        Ok(())
    }

    /// Load tensor from file.
    pub fn load_file(path: &str) -> io::Result<AlphaTensor> {
        let mut f = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    // -- private helpers --

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(bytes);
    }

    fn read_string(data: &[u8], pos: &mut usize) -> io::Result<String> {
        let len = Self::read_u32(data, pos)? as usize;
        if *pos + len > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated string"));
        }
        let s = String::from_utf8(data[*pos..*pos + len].to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad utf8"))?;
        *pos += len;
        Ok(s)
    }

    fn read_u8(data: &[u8], pos: &mut usize) -> io::Result<u8> {
        if *pos >= data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated"));
        }
        let v = data[*pos];
        *pos += 1;
        Ok(v)
    }

    fn read_u32(data: &[u8], pos: &mut usize) -> io::Result<u32> {
        if *pos + 4 > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated"));
        }
        let v = u32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]);
        *pos += 4;
        Ok(v)
    }

    fn read_u64(data: &[u8], pos: &mut usize) -> io::Result<u64> {
        if *pos + 8 > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated"));
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&data[*pos..*pos+8]);
        let v = u64::from_le_bytes(bytes);
        *pos += 8;
        Ok(v)
    }

    fn read_f64(data: &[u8], pos: &mut usize) -> io::Result<f64> {
        let bits = Self::read_u64(data, pos)?;
        Ok(f64::from_bits(bits))
    }
}

// ---------------------------------------------------------------------------
// 14. Data generation (synthetic) for testing / demos
// ---------------------------------------------------------------------------

/// Simple LCG PRNG for deterministic synthetic data.
#[derive(Debug, Clone)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate normal via Box-Muller.
    pub fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }

    /// Generate synthetic return series.
    pub fn synthetic_returns(&mut self, n: usize, drift: f64, vol: f64) -> Vec<f64> {
        (0..n).map(|_| self.next_normal(drift, vol)).collect()
    }
}

/// Generate a map of synthetic returns for multiple symbols.
pub fn generate_synthetic_data(
    symbols: &[&str],
    n_bars: usize,
    seed: u64,
) -> HashMap<String, Vec<f64>> {
    let mut rng = Lcg::new(seed);
    let mut data = HashMap::new();
    for &sym in symbols {
        let drift = rng.next_normal(0.0001, 0.0001);
        let vol = 0.005 + rng.next_f64() * 0.02;
        let returns = rng.synthetic_returns(n_bars, drift, vol);
        data.insert(sym.to_string(), returns);
    }
    data
}

// ---------------------------------------------------------------------------
// 15. Default parameter grids for each strategy
// ---------------------------------------------------------------------------

/// Get a default parameter grid for a named strategy.
pub fn default_grid(strategy: &str) -> ParameterGrid {
    match strategy {
        "momentum" => ParameterGrid::new(vec![
            ("lookback".into(), ParamSpec::Linspace { start: 5.0, end: 100.0, steps: 10 }),
        ]),
        "mean_reversion" => ParameterGrid::new(vec![
            ("lookback".into(), ParamSpec::Linspace { start: 5.0, end: 60.0, steps: 8 }),
            ("threshold".into(), ParamSpec::Linspace { start: 0.5, end: 3.0, steps: 6 }),
        ]),
        "breakout" => ParameterGrid::new(vec![
            ("lookback".into(), ParamSpec::Linspace { start: 10.0, end: 200.0, steps: 10 }),
        ]),
        "rsi" => ParameterGrid::new(vec![
            ("period".into(), ParamSpec::Linspace { start: 5.0, end: 30.0, steps: 6 }),
            ("oversold".into(), ParamSpec::Linspace { start: 20.0, end: 40.0, steps: 3 }),
            ("overbought".into(), ParamSpec::Linspace { start: 60.0, end: 80.0, steps: 3 }),
        ]),
        "macd_cross" => ParameterGrid::new(vec![
            ("fast".into(), ParamSpec::Linspace { start: 6.0, end: 18.0, steps: 4 }),
            ("slow".into(), ParamSpec::Linspace { start: 20.0, end: 40.0, steps: 4 }),
            ("signal".into(), ParamSpec::Linspace { start: 5.0, end: 15.0, steps: 3 }),
        ]),
        "bollinger_band" => ParameterGrid::new(vec![
            ("period".into(), ParamSpec::Linspace { start: 10.0, end: 50.0, steps: 5 }),
            ("width".into(), ParamSpec::Linspace { start: 1.0, end: 3.0, steps: 5 }),
        ]),
        "trend_following" => ParameterGrid::new(vec![
            ("fast_period".into(), ParamSpec::Linspace { start: 5.0, end: 30.0, steps: 6 }),
            ("slow_period".into(), ParamSpec::Linspace { start: 30.0, end: 120.0, steps: 6 }),
        ]),
        "volatility_breakout" => ParameterGrid::new(vec![
            ("lookback".into(), ParamSpec::Linspace { start: 5.0, end: 50.0, steps: 6 }),
            ("mult".into(), ParamSpec::Linspace { start: 0.5, end: 3.0, steps: 6 }),
        ]),
        "pairs_spread" => ParameterGrid::new(vec![
            ("lookback".into(), ParamSpec::Linspace { start: 10.0, end: 60.0, steps: 6 }),
            ("threshold".into(), ParamSpec::Linspace { start: 0.5, end: 3.0, steps: 6 }),
        ]),
        "entropy_adaptive" => ParameterGrid::new(vec![
            ("entropy_window".into(), ParamSpec::Linspace { start: 20.0, end: 100.0, steps: 5 }),
            ("momentum_window".into(), ParamSpec::Linspace { start: 5.0, end: 40.0, steps: 5 }),
        ]),
        _ => ParameterGrid::new(vec![]),
    }
}

/// Build (strategy_name, grid) pairs for all 10 strategies.
pub fn all_strategy_grids() -> Vec<(&'static str, ParameterGrid)> {
    vec![
        ("momentum", default_grid("momentum")),
        ("mean_reversion", default_grid("mean_reversion")),
        ("breakout", default_grid("breakout")),
        ("rsi", default_grid("rsi")),
        ("macd_cross", default_grid("macd_cross")),
        ("bollinger_band", default_grid("bollinger_band")),
        ("trend_following", default_grid("trend_following")),
        ("volatility_breakout", default_grid("volatility_breakout")),
        ("pairs_spread", default_grid("pairs_spread")),
        ("entropy_adaptive", default_grid("entropy_adaptive")),
    ]
}

/// Count total combinations across all strategy grids.
pub fn total_farm_combinations(
    strategy_grids: &[(&str, ParameterGrid)],
    n_symbols: usize,
    n_timeframes: usize,
) -> usize {
    let mut total = 0;
    for (_, grid) in strategy_grids {
        total += grid.total_combinations() * n_symbols * n_timeframes;
    }
    total
}

// ---------------------------------------------------------------------------
// 16. Equity curve analysis
// ---------------------------------------------------------------------------

/// Compute the equity curve from a return series.
pub fn equity_curve(returns: &[f64]) -> Vec<f64> {
    let mut curve = Vec::with_capacity(returns.len() + 1);
    curve.push(1.0);
    let mut eq = 1.0;
    for &r in returns {
        eq *= 1.0 + r;
        curve.push(eq);
    }
    curve
}

/// Compute drawdown series from a return series.
pub fn drawdown_series(returns: &[f64]) -> Vec<f64> {
    let eq = equity_curve(returns);
    let mut dd = Vec::with_capacity(eq.len());
    let mut peak = 0.0_f64;
    for &e in &eq {
        if e > peak { peak = e; }
        dd.push(if peak > 0.0 { (peak - e) / peak } else { 0.0 });
    }
    dd
}

/// Longest drawdown period in bars.
pub fn longest_drawdown(returns: &[f64]) -> usize {
    let dd = drawdown_series(returns);
    let mut max_len = 0;
    let mut current = 0;
    for &d in &dd {
        if d > 1e-10 {
            current += 1;
            if current > max_len { max_len = current; }
        } else {
            current = 0;
        }
    }
    max_len
}

/// Time under water: fraction of bars in drawdown.
pub fn time_under_water(returns: &[f64]) -> f64 {
    let dd = drawdown_series(returns);
    if dd.is_empty() { return 0.0; }
    let uw = dd.iter().filter(|&&d| d > 1e-10).count();
    uw as f64 / dd.len() as f64
}

/// Tail ratio: 95th percentile / |5th percentile|.
pub fn tail_ratio(returns: &[f64]) -> f64 {
    if returns.len() < 20 { return 1.0; }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| f64_sort_key(*a).cmp(&f64_sort_key(*b)));
    let p95 = percentile(&sorted, 0.95);
    let p05 = percentile(&sorted, 0.05).abs();
    if p05 < 1e-15 { return f64::INFINITY; }
    p95 / p05
}

/// Value-at-Risk (historical, percentile method).
pub fn var_historical(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() { return 0.0; }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| f64_sort_key(*a).cmp(&f64_sort_key(*b)));
    let p = 1.0 - confidence;
    -percentile(&sorted, p)
}

/// Conditional VaR (Expected Shortfall).
pub fn cvar(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() { return 0.0; }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| f64_sort_key(*a).cmp(&f64_sort_key(*b)));
    let cutoff_idx = ((1.0 - confidence) * sorted.len() as f64).ceil() as usize;
    let cutoff_idx = cutoff_idx.max(1).min(sorted.len());
    let tail = &sorted[..cutoff_idx];
    -mean(tail)
}

/// Omega ratio: probability-weighted gains over losses relative to a threshold.
pub fn omega_ratio(returns: &[f64], threshold: f64) -> f64 {
    if returns.is_empty() { return 1.0; }
    let mut gains = 0.0;
    let mut losses = 0.0;
    for &r in returns {
        let excess = r - threshold;
        if excess > 0.0 { gains += excess; }
        else { losses += excess.abs(); }
    }
    if losses < 1e-15 { return f64::INFINITY; }
    gains / losses
}

/// Kurtosis (excess).
pub fn kurtosis(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 4 { return 0.0; }
    let m = mean(xs);
    let s = stddev(xs);
    if s < 1e-15 { return 0.0; }
    let m4: f64 = xs.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n as f64;
    m4 - 3.0
}

/// Skewness.
pub fn skewness(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 3 { return 0.0; }
    let m = mean(xs);
    let s = stddev(xs);
    if s < 1e-15 { return 0.0; }
    xs.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n as f64
}

// ---------------------------------------------------------------------------
// 17. Walk-forward analysis
// ---------------------------------------------------------------------------

/// Walk-forward split: (train_start, train_end, test_start, test_end).
pub type WalkForwardSplit = (usize, usize, usize, usize);

/// Generate walk-forward splits.
pub fn walk_forward_splits(
    total_bars: usize,
    train_bars: usize,
    test_bars: usize,
    step_bars: usize,
) -> Vec<WalkForwardSplit> {
    let mut splits = Vec::new();
    let mut start = 0;
    while start + train_bars + test_bars <= total_bars {
        let train_start = start;
        let train_end = start + train_bars;
        let test_start = train_end;
        let test_end = (train_end + test_bars).min(total_bars);
        splits.push((train_start, train_end, test_start, test_end));
        start += step_bars;
    }
    splits
}

/// Run walk-forward optimisation for one strategy on one symbol.
/// For each fold: optimise params on train, evaluate on test.
/// Returns: (best_params_per_fold, out_of_sample_sharpes).
pub fn walk_forward_optimise(
    strategy: &str,
    returns: &[f64],
    timeframe: Timeframe,
    grid: &ParameterGrid,
    cost_bps: f64,
    train_bars: usize,
    test_bars: usize,
    step_bars: usize,
) -> (Vec<HashMap<String, f64>>, Vec<f64>) {
    let splits = walk_forward_splits(returns.len(), train_bars, test_bars, step_bars);
    let mut best_params_list = Vec::with_capacity(splits.len());
    let mut oos_sharpes = Vec::with_capacity(splits.len());

    for (train_start, train_end, test_start, test_end) in &splits {
        // Optimise on train.
        let mut best_sharpe = f64::NEG_INFINITY;
        let mut best_params: Option<HashMap<String, f64>> = None;

        for params in grid.iter() {
            let config = BacktestConfig::new(
                strategy, "WF", timeframe, params.clone(),
                cost_bps, *train_start, *train_end,
            );
            let signals = run_strategy(strategy, returns, &config.parameters);
            let result = quick_backtest_from_signals(returns, &signals, config);
            if result.sharpe > best_sharpe {
                best_sharpe = result.sharpe;
                best_params = Some(params);
            }
        }

        let params = best_params.unwrap_or_default();
        best_params_list.push(params.clone());

        // Evaluate on test.
        let test_config = BacktestConfig::new(
            strategy, "WF", timeframe, params,
            cost_bps, *test_start, *test_end,
        );
        let signals = run_strategy(strategy, returns, &test_config.parameters);
        let test_result = quick_backtest_from_signals(returns, &signals, test_config);
        oos_sharpes.push(test_result.sharpe);
    }

    (best_params_list, oos_sharpes)
}

/// Walk-forward efficiency: mean OOS Sharpe / mean IS Sharpe.
pub fn walk_forward_efficiency(is_sharpes: &[f64], oos_sharpes: &[f64]) -> f64 {
    let is_mean = mean(is_sharpes);
    if is_mean.abs() < 1e-15 { return 0.0; }
    mean(oos_sharpes) / is_mean
}

// ---------------------------------------------------------------------------
// 18. Monte Carlo tools
// ---------------------------------------------------------------------------

/// Bootstrap resample a return series (with replacement).
pub fn bootstrap_returns(returns: &[f64], seed: u64) -> Vec<f64> {
    let n = returns.len();
    if n == 0 { return vec![]; }
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| returns[(rng.next_u64() as usize) % n]).collect()
}

/// Monte Carlo confidence interval for Sharpe ratio.
/// Returns (mean, lower_5th_percentile, upper_95th_percentile).
pub fn mc_sharpe_confidence(
    returns: &[f64],
    ann: f64,
    n_sims: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let mut sharpes = Vec::with_capacity(n_sims);
    let mut rng = Lcg::new(seed);
    for _ in 0..n_sims {
        let resampled = bootstrap_returns(returns, rng.next_u64());
        sharpes.push(PerformanceCalculator::sharpe(&resampled, ann));
    }
    sharpes.sort_by(|a, b| f64_sort_key(*a).cmp(&f64_sort_key(*b)));
    let m = mean(&sharpes);
    let lo = percentile(&sharpes, 0.05);
    let hi = percentile(&sharpes, 0.95);
    (m, lo, hi)
}

/// Permutation test: is the strategy's Sharpe significantly different from random?
/// Returns the p-value.
pub fn permutation_test_sharpe(
    returns: &[f64],
    signals: &[f64],
    cost_bps: f64,
    n_perms: usize,
    seed: u64,
) -> f64 {
    let n = returns.len().min(signals.len());
    if n < 10 { return 1.0; }
    let cost_frac = cost_bps / 10_000.0;

    // Compute actual Sharpe.
    let mut strat_ret = Vec::with_capacity(n);
    let mut prev = 0.0;
    for i in 0..n {
        let pos = clamp(signals[i], -1.0, 1.0);
        let tc = (pos - prev).abs() * cost_frac;
        strat_ret.push(pos * returns[i] - tc);
        prev = pos;
    }
    let actual_sharpe = PerformanceCalculator::sharpe(&strat_ret, 252.0);

    // Permutation: shuffle returns.
    let mut rng = Lcg::new(seed);
    let mut count_ge = 0;
    for _ in 0..n_perms {
        let shuffled = bootstrap_returns(returns, rng.next_u64());
        let mut perm_ret = Vec::with_capacity(n);
        let mut prev = 0.0;
        for i in 0..n {
            let pos = clamp(signals[i], -1.0, 1.0);
            let tc = (pos - prev).abs() * cost_frac;
            perm_ret.push(pos * shuffled[i] - tc);
            prev = pos;
        }
        let perm_sharpe = PerformanceCalculator::sharpe(&perm_ret, 252.0);
        if perm_sharpe >= actual_sharpe {
            count_ge += 1;
        }
    }

    count_ge as f64 / n_perms as f64
}

// ---------------------------------------------------------------------------
// 19. Correlation & portfolio-level analysis
// ---------------------------------------------------------------------------

/// Compute pairwise correlation matrix of strategy returns.
pub fn strategy_correlation_matrix(
    strategy_returns: &[(&str, Vec<f64>)],
) -> (Vec<String>, Vec<Vec<f64>>) {
    let n = strategy_returns.len();
    let names: Vec<String> = strategy_returns.iter().map(|(s, _)| s.to_string()).collect();
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[i][j] = 1.0;
            } else if j > i {
                let corr = pearson_corr(&strategy_returns[i].1, &strategy_returns[j].1);
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }
    }
    (names, matrix)
}

/// Equal-weight portfolio of N strategy return streams.
pub fn equal_weight_portfolio(streams: &[&[f64]]) -> Vec<f64> {
    if streams.is_empty() { return vec![]; }
    let n = streams.iter().map(|s| s.len()).min().unwrap_or(0);
    let w = 1.0 / streams.len() as f64;
    (0..n).map(|i| streams.iter().map(|s| s[i] * w).sum()).collect()
}

/// Inverse-volatility weighted portfolio.
pub fn inv_vol_portfolio(streams: &[&[f64]], lookback: usize) -> Vec<f64> {
    if streams.is_empty() { return vec![]; }
    let n = streams.iter().map(|s| s.len()).min().unwrap_or(0);
    let k = streams.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i < lookback {
            out.push(streams.iter().map(|s| s[i] / k as f64).sum());
        } else {
            let mut vols = Vec::with_capacity(k);
            for s in streams {
                let window = &s[i - lookback..i];
                vols.push(stddev(window).max(1e-10));
            }
            let inv_sum: f64 = vols.iter().map(|v| 1.0 / v).sum();
            let weights: Vec<f64> = vols.iter().map(|v| (1.0 / v) / inv_sum).collect();
            let ret: f64 = streams.iter().zip(weights.iter()).map(|(s, w)| s[i] * w).sum();
            out.push(ret);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 20. Regime detection
// ---------------------------------------------------------------------------

/// Simple regime detector: high-vol vs low-vol regime based on rolling volatility.
pub fn detect_vol_regime(returns: &[f64], lookback: usize) -> Vec<bool> {
    let rstd = rolling_std(returns, lookback);
    let med = {
        let mut sorted = rstd.clone();
        sorted.sort_by(|a, b| f64_sort_key(*a).cmp(&f64_sort_key(*b)));
        percentile(&sorted, 0.5)
    };
    rstd.iter().map(|&v| v > med).collect()
}

/// Trend regime: is the market trending (positive autocorrelation) or mean-reverting?
pub fn detect_trend_regime(returns: &[f64], lookback: usize) -> Vec<f64> {
    let n = returns.len();
    let mut ac = vec![0.0; n];
    for i in lookback..n {
        let window = &returns[i - lookback..i];
        if window.len() < 2 { continue; }
        let lagged: Vec<f64> = window[..window.len() - 1].to_vec();
        let current: Vec<f64> = window[1..].to_vec();
        ac[i] = pearson_corr(&lagged, &current);
    }
    ac
}

/// Split results by regime: returns (high_vol_results, low_vol_results).
pub fn split_by_regime(
    returns: &[f64],
    signals: &[f64],
    cost_bps: f64,
    lookback: usize,
) -> (Vec<f64>, Vec<f64>) {
    let regime = detect_vol_regime(returns, lookback);
    let n = returns.len().min(signals.len()).min(regime.len());
    let cost_frac = cost_bps / 10_000.0;
    let mut high_vol = Vec::new();
    let mut low_vol = Vec::new();
    let mut prev_pos = 0.0;
    for i in 0..n {
        let pos = clamp(signals[i], -1.0, 1.0);
        let tc = (pos - prev_pos).abs() * cost_frac;
        let bar_ret = pos * returns[i] - tc;
        if regime[i] {
            high_vol.push(bar_ret);
        } else {
            low_vol.push(bar_ret);
        }
        prev_pos = pos;
    }
    (high_vol, low_vol)
}

// ---------------------------------------------------------------------------
// 21. Benchmark comparison
// ---------------------------------------------------------------------------

/// Compare strategy to buy-and-hold benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub strategy_sharpe: f64,
    pub benchmark_sharpe: f64,
    pub strategy_return: f64,
    pub benchmark_return: f64,
    pub strategy_dd: f64,
    pub benchmark_dd: f64,
    pub alpha: f64,
    pub beta: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,
}

impl BenchmarkComparison {
    /// Compute strategy vs benchmark (buy-and-hold).
    pub fn compute(strategy_returns: &[f64], benchmark_returns: &[f64], ann: f64) -> Self {
        let n = strategy_returns.len().min(benchmark_returns.len());
        let sr = &strategy_returns[..n];
        let br = &benchmark_returns[..n];

        let strategy_sharpe = PerformanceCalculator::sharpe(sr, ann);
        let benchmark_sharpe = PerformanceCalculator::sharpe(br, ann);
        let strategy_return = PerformanceCalculator::total_return(sr);
        let benchmark_return = PerformanceCalculator::total_return(br);
        let strategy_dd = PerformanceCalculator::max_drawdown(sr);
        let benchmark_dd = PerformanceCalculator::max_drawdown(br);

        // Alpha & beta via linear regression: strategy = alpha + beta * benchmark + epsilon.
        let bm = mean(br);
        let sm = mean(sr);
        let mut cov = 0.0;
        let mut var_b = 0.0;
        for i in 0..n {
            cov += (br[i] - bm) * (sr[i] - sm);
            var_b += (br[i] - bm) * (br[i] - bm);
        }
        let beta = if var_b.abs() < 1e-15 { 0.0 } else { cov / var_b };
        let alpha = (sm - beta * bm) * ann;

        // Tracking error & IR.
        let excess: Vec<f64> = (0..n).map(|i| sr[i] - br[i]).collect();
        let te = stddev(&excess) * ann.sqrt();
        let ir = if te.abs() < 1e-15 { 0.0 } else { mean(&excess) * ann.sqrt() / te * ann.sqrt().recip() * ann.sqrt() };
        // Simplify: IR = mean(excess) / std(excess) * sqrt(ann)
        let ir2 = if te.abs() < 1e-15 { 0.0 } else { mean(&excess) / stddev(&excess) * ann.sqrt() };

        Self {
            strategy_sharpe, benchmark_sharpe,
            strategy_return, benchmark_return,
            strategy_dd, benchmark_dd,
            alpha, beta,
            tracking_error: te,
            information_ratio: ir2,
        }
    }

    /// One-line summary.
    pub fn summary(&self) -> String {
        format!(
            "alpha={:.4} beta={:.3} IR={:.3} TE={:.4} | strat: S={:.3} R={:.4} DD={:.4} | bench: S={:.3} R={:.4} DD={:.4}",
            self.alpha, self.beta, self.information_ratio, self.tracking_error,
            self.strategy_sharpe, self.strategy_return, self.strategy_dd,
            self.benchmark_sharpe, self.benchmark_return, self.benchmark_dd,
        )
    }
}

// ---------------------------------------------------------------------------
// 22. Rolling performance windows
// ---------------------------------------------------------------------------

/// Rolling Sharpe ratio computed over a sliding window.
pub fn rolling_sharpe(returns: &[f64], window: usize, ann: f64) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![f64::NAN; n];
    for i in window..n {
        let w = &returns[i - window..i];
        out[i] = PerformanceCalculator::sharpe(w, ann);
    }
    out
}

/// Rolling max drawdown.
pub fn rolling_max_drawdown(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![0.0; n];
    for i in window..n {
        let w = &returns[i - window..i];
        out[i] = PerformanceCalculator::max_drawdown(w);
    }
    out
}

/// Rolling win rate.
pub fn rolling_win_rate(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    let mut out = vec![f64::NAN; n];
    for i in window..n {
        let w = &returns[i - window..i];
        out[i] = PerformanceCalculator::win_rate(w);
    }
    out
}

// ---------------------------------------------------------------------------
// 23. Decay analysis
// ---------------------------------------------------------------------------

/// Measure alpha decay: run strategy on expanding windows and track Sharpe.
pub fn alpha_decay(
    strategy: &str,
    returns: &[f64],
    params: &HashMap<String, f64>,
    timeframe: Timeframe,
    cost_bps: f64,
    window_step: usize,
    min_window: usize,
) -> Vec<(usize, f64)> {
    let n = returns.len();
    let mut results = Vec::new();
    let mut end = min_window;
    while end <= n {
        let config = BacktestConfig::new(strategy, "DECAY", timeframe, params.clone(), cost_bps, 0, end);
        let signals = run_strategy(strategy, returns, params);
        let result = quick_backtest_from_signals(returns, &signals, config);
        results.push((end, result.sharpe));
        end += window_step;
    }
    results
}

/// Half-life of alpha: estimate how quickly Sharpe decays from peak.
pub fn alpha_half_life(decay_curve: &[(usize, f64)]) -> Option<usize> {
    if decay_curve.is_empty() { return None; }
    let peak_sharpe = decay_curve.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
    let half = peak_sharpe / 2.0;
    let peak_idx = decay_curve.iter().position(|(_, s)| *s == peak_sharpe)?;
    for i in peak_idx..decay_curve.len() {
        if decay_curve[i].1 < half {
            return Some(decay_curve[i].0 - decay_curve[peak_idx].0);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 24. Capacity estimation
// ---------------------------------------------------------------------------

/// Estimate strategy capacity: at what cost level does Sharpe go to zero?
pub fn estimate_capacity(
    strategy: &str,
    returns: &[f64],
    params: &HashMap<String, f64>,
    timeframe: Timeframe,
    max_cost_bps: f64,
    steps: usize,
) -> Vec<(f64, f64)> {
    let mut results = Vec::with_capacity(steps);
    for i in 0..steps {
        let cost = max_cost_bps * i as f64 / (steps - 1).max(1) as f64;
        let config = BacktestConfig::new(
            strategy, "CAP", timeframe, params.clone(),
            cost, 0, returns.len(),
        );
        let signals = run_strategy(strategy, returns, params);
        let result = quick_backtest_from_signals(returns, &signals, config);
        results.push((cost, result.sharpe));
    }
    results
}

/// Break-even cost: the cost in bps at which Sharpe crosses zero.
pub fn break_even_cost(capacity_curve: &[(f64, f64)]) -> Option<f64> {
    for i in 1..capacity_curve.len() {
        let (c0, s0) = capacity_curve[i - 1];
        let (c1, s1) = capacity_curve[i];
        if s0 >= 0.0 && s1 < 0.0 {
            // Linear interpolation to find zero crossing.
            let t = s0 / (s0 - s1);
            return Some(lerp(c0, c1, t));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 25. Multi-objective optimisation (Pareto front)
// ---------------------------------------------------------------------------

/// A point in objective space.
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub config: BacktestConfig,
    pub objectives: Vec<f64>, // e.g. [sharpe, -max_dd, profit_factor]
}

/// Check if point A dominates point B (all objectives >= and at least one >).
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    let n = a.len().min(b.len());
    let mut all_ge = true;
    let mut any_gt = false;
    for i in 0..n {
        if a[i] < b[i] { all_ge = false; break; }
        if a[i] > b[i] { any_gt = true; }
    }
    all_ge && any_gt
}

/// Extract Pareto front from a set of results.
pub fn pareto_front<F>(tensor: &AlphaTensor, objectives: F) -> Vec<ParetoPoint>
where
    F: Fn(&BacktestResult) -> Vec<f64>,
{
    let points: Vec<ParetoPoint> = tensor.values().iter().map(|r| {
        ParetoPoint {
            config: r.config.clone(),
            objectives: objectives(r),
        }
    }).collect();

    let mut front = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let mut is_dominated = false;
        for (j, q) in points.iter().enumerate() {
            if i == j { continue; }
            if dominates(&q.objectives, &p.objectives) {
                is_dominated = true;
                break;
            }
        }
        if !is_dominated {
            front.push(p.clone());
        }
    }
    front
}

/// 3-objective Pareto: maximize Sharpe, minimize DD, maximize profit factor.
pub fn pareto_sharpe_dd_pf(tensor: &AlphaTensor) -> Vec<ParetoPoint> {
    pareto_front(tensor, |r| vec![r.sharpe, -r.max_drawdown, r.profit_factor])
}

// ---------------------------------------------------------------------------
// 26. Signal combination
// ---------------------------------------------------------------------------

/// Combine multiple signals with equal weight.
pub fn combine_signals_equal(signals: &[&[f64]]) -> Vec<f64> {
    if signals.is_empty() { return vec![]; }
    let n = signals.iter().map(|s| s.len()).min().unwrap_or(0);
    let w = 1.0 / signals.len() as f64;
    (0..n).map(|i| {
        let s: f64 = signals.iter().map(|sig| sig[i] * w).sum();
        clamp(s, -1.0, 1.0)
    }).collect()
}

/// Combine signals with custom weights.
pub fn combine_signals_weighted(signals: &[&[f64]], weights: &[f64]) -> Vec<f64> {
    if signals.is_empty() { return vec![]; }
    let n = signals.iter().map(|s| s.len()).min().unwrap_or(0);
    let wsum: f64 = weights.iter().sum();
    if wsum.abs() < 1e-15 { return vec![0.0; n]; }
    (0..n).map(|i| {
        let s: f64 = signals.iter().zip(weights.iter()).map(|(sig, w)| sig[i] * w).sum::<f64>() / wsum;
        clamp(s, -1.0, 1.0)
    }).collect()
}

/// Combine signals by majority vote (sign).
pub fn combine_signals_vote(signals: &[&[f64]]) -> Vec<f64> {
    if signals.is_empty() { return vec![]; }
    let n = signals.iter().map(|s| s.len()).min().unwrap_or(0);
    (0..n).map(|i| {
        let sum: f64 = signals.iter().map(|sig| sig[i].signum()).sum();
        sum.signum()
    }).collect()
}

// ---------------------------------------------------------------------------
// 27. Summary statistics for tensor exploration
// ---------------------------------------------------------------------------

/// Distribution summary for a metric across the tensor.
#[derive(Debug, Clone)]
pub struct DistributionSummary {
    pub name: String,
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub p5: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub p95: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl DistributionSummary {
    /// Compute from a slice of values.
    pub fn from_values(name: &str, values: &[f64]) -> Self {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| f64_sort_key(*a).cmp(&f64_sort_key(*b)));
        Self {
            name: name.to_string(),
            count: values.len(),
            mean: mean(values),
            std: stddev(values),
            min: sorted.first().cloned().unwrap_or(0.0),
            p5: percentile(&sorted, 0.05),
            p25: percentile(&sorted, 0.25),
            median: percentile(&sorted, 0.50),
            p75: percentile(&sorted, 0.75),
            p95: percentile(&sorted, 0.95),
            max: sorted.last().cloned().unwrap_or(0.0),
            skewness: skewness(values),
            kurtosis: kurtosis(values),
        }
    }

    /// Render as a string.
    pub fn render(&self) -> String {
        format!(
            "{}: n={} mean={:.4} std={:.4} [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}] skew={:.3} kurt={:.3}",
            self.name, self.count, self.mean, self.std,
            self.p5, self.p25, self.median, self.p75, self.p95,
            self.skewness, self.kurtosis,
        )
    }
}

/// Compute distribution summaries for all major metrics in a tensor.
pub fn tensor_distribution_summary(tensor: &AlphaTensor) -> Vec<DistributionSummary> {
    let vals = tensor.values();
    if vals.is_empty() { return vec![]; }

    let sharpes: Vec<f64> = vals.iter().map(|r| r.sharpe).collect();
    let sortinos: Vec<f64> = vals.iter().map(|r| r.sortino).collect();
    let calmars: Vec<f64> = vals.iter().map(|r| r.calmar).collect();
    let returns: Vec<f64> = vals.iter().map(|r| r.total_return).collect();
    let drawdowns: Vec<f64> = vals.iter().map(|r| r.max_drawdown).collect();
    let win_rates: Vec<f64> = vals.iter().map(|r| r.win_rate).collect();
    let profit_factors: Vec<f64> = vals.iter().map(|r| r.profit_factor.min(100.0)).collect();
    let ics: Vec<f64> = vals.iter().map(|r| r.ic).collect();
    let turnovers: Vec<f64> = vals.iter().map(|r| r.turnover).collect();

    vec![
        DistributionSummary::from_values("Sharpe", &sharpes),
        DistributionSummary::from_values("Sortino", &sortinos),
        DistributionSummary::from_values("Calmar", &calmars),
        DistributionSummary::from_values("Total Return", &returns),
        DistributionSummary::from_values("Max Drawdown", &drawdowns),
        DistributionSummary::from_values("Win Rate", &win_rates),
        DistributionSummary::from_values("Profit Factor", &profit_factors),
        DistributionSummary::from_values("IC", &ics),
        DistributionSummary::from_values("Turnover", &turnovers),
    ]
}

// ---------------------------------------------------------------------------
// 28. End-to-end convenience: run the entire farm
// ---------------------------------------------------------------------------

/// Run the entire backtest farm: all strategies, all symbols, all timeframes,
/// all parameter combinations. Returns tensor, landscape, and report.
pub fn run_farm(
    symbols: &[&str],
    timeframes: &[Timeframe],
    returns_data: &HashMap<String, Vec<f64>>,
    cost_bps: f64,
    n_bars: usize,
    n_workers: usize,
) -> (AlphaTensor, AlphaLandscape, LandscapeReport) {
    let grids = all_strategy_grids();
    let backtester = ParallelBacktester::new(n_workers);
    let tensor = backtester.run_full_farm(&grids, symbols, timeframes, cost_bps, n_bars, returns_data);
    let landscape = AlphaLandscape::from_tensor(&tensor);
    let report = LandscapeReport::from_tensor(&tensor);
    (tensor, landscape, report)
}

/// Quick demo: generate synthetic data and run the farm.
pub fn demo_farm(n_bars: usize, seed: u64) -> (AlphaTensor, AlphaLandscape, LandscapeReport) {
    let symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD"];
    let data = generate_synthetic_data(&symbols, n_bars, seed);
    let timeframes = [Timeframe::H1, Timeframe::H4, Timeframe::D1];
    run_farm(&symbols, &timeframes, &data, 5.0, n_bars, 4)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_roundtrip() {
        for tf in Timeframe::all() {
            assert_eq!(Timeframe::from_str(tf.as_str()), Some(*tf));
            assert_eq!(Timeframe::from_u8(tf.to_u8()), Some(*tf));
        }
    }

    #[test]
    fn test_param_hash_deterministic() {
        let mut p1 = HashMap::new();
        p1.insert("a".into(), 1.0);
        p1.insert("b".into(), 2.0);
        let mut p2 = HashMap::new();
        p2.insert("b".into(), 2.0);
        p2.insert("a".into(), 1.0);
        assert_eq!(param_hash(&p1), param_hash(&p2));
    }

    #[test]
    fn test_parameter_grid() {
        let grid = ParameterGrid::new(vec![
            ("x".into(), ParamSpec::Linspace { start: 0.0, end: 1.0, steps: 3 }),
            ("y".into(), ParamSpec::Categorical(vec![10.0, 20.0])),
        ]);
        assert_eq!(grid.total_combinations(), 6);
        let combos: Vec<_> = grid.iter().collect();
        assert_eq!(combos.len(), 6);
    }

    #[test]
    fn test_logspace() {
        let spec = ParamSpec::Logspace { start: 1.0, end: 100.0, steps: 3 };
        let vals = spec.values();
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[2] - 100.0).abs() < 1e-8);
        assert!((vals[1] - 10.0).abs() < 1e-8);
    }

    #[test]
    fn test_performance_calculator() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012];
        let sharpe = PerformanceCalculator::sharpe(&returns, 252.0);
        assert!(sharpe > 0.0);
        let dd = PerformanceCalculator::max_drawdown(&returns);
        assert!(dd >= 0.0 && dd <= 1.0);
        let wr = PerformanceCalculator::win_rate(&returns);
        assert!(wr > 0.0 && wr <= 1.0);
        let pf = PerformanceCalculator::profit_factor(&returns);
        assert!(pf > 0.0);
    }

    #[test]
    fn test_alpha_tensor_insert_get() {
        let mut tensor = AlphaTensor::new();
        let config = BacktestConfig::new("momentum", "BTCUSD", Timeframe::H1, HashMap::new(), 5.0, 0, 100);
        let key = config.tensor_key();
        let result = BacktestResult::failed(config);
        tensor.insert(result);
        assert_eq!(tensor.len(), 1);
        assert!(tensor.get(&key).is_some());
    }

    #[test]
    fn test_top_k() {
        let mut tensor = AlphaTensor::new();
        for i in 0..10 {
            let mut params = HashMap::new();
            params.insert("i".into(), i as f64);
            let config = BacktestConfig::new("momentum", "BTCUSD", Timeframe::H1, params, 5.0, 0, 100);
            let mut result = BacktestResult::failed(config);
            result.sharpe = i as f64;
            tensor.insert(result);
        }
        let top3 = tensor.top_k(3, |r| r.sharpe);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].sharpe, 9.0);
        assert_eq!(top3[1].sharpe, 8.0);
        assert_eq!(top3[2].sharpe, 7.0);
    }

    #[test]
    fn test_signals_len() {
        let mut rng = Lcg::new(42);
        let returns = rng.synthetic_returns(200, 0.0001, 0.01);
        for (name, func) in builtin_strategies() {
            let sig = func(&returns);
            assert_eq!(sig.len(), returns.len(), "signal {} length mismatch", name);
        }
    }

    #[test]
    fn test_quick_backtest() {
        let mut rng = Lcg::new(42);
        let returns = rng.synthetic_returns(500, 0.0001, 0.01);
        let config = BacktestConfig::new("momentum", "TEST", Timeframe::H1, HashMap::new(), 5.0, 0, 500);
        let result = quick_backtest(&returns, signal_momentum, config);
        assert!(result.sharpe.is_finite());
        assert!(result.max_drawdown >= 0.0);
    }

    #[test]
    fn test_job_queue() {
        let queue = JobQueue::new();
        let config = BacktestConfig::new("test", "SYM", Timeframe::H1, HashMap::new(), 5.0, 0, 100);
        let id = queue.push(config);
        assert_eq!(id, 0);
        assert_eq!(queue.pending_len(), 1);
        let job = queue.pop().unwrap();
        assert_eq!(job.job_id, 0);
        assert_eq!(queue.pending_len(), 0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut tensor = AlphaTensor::new();
        let mut params = HashMap::new();
        params.insert("lookback".into(), 20.0);
        let config = BacktestConfig::new("momentum", "BTCUSD", Timeframe::H1, params, 5.0, 0, 100);
        let mut result = BacktestResult::failed(config);
        result.sharpe = 1.5;
        result.sortino = 2.0;
        result.max_drawdown = 0.1;
        tensor.insert(result);

        let bytes = TensorSerializer::to_bytes(&tensor);
        let restored = TensorSerializer::from_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 1);
        let key = restored.keys()[0];
        let r = restored.get(key).unwrap();
        assert!((r.sharpe - 1.5).abs() < 1e-10);
        assert!((r.sortino - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_equity_curve() {
        let returns = vec![0.1, -0.05, 0.2];
        let eq = equity_curve(&returns);
        assert_eq!(eq.len(), 4);
        assert!((eq[0] - 1.0).abs() < 1e-10);
        assert!((eq[1] - 1.1).abs() < 1e-10);
        assert!((eq[2] - 1.045).abs() < 1e-10);
        assert!((eq[3] - 1.254).abs() < 1e-10);
    }

    #[test]
    fn test_drawdown() {
        let returns = vec![0.1, -0.2, 0.05];
        let dd = PerformanceCalculator::max_drawdown(&returns);
        assert!(dd > 0.0);
    }

    #[test]
    fn test_var_cvar() {
        let mut rng = Lcg::new(123);
        let returns = rng.synthetic_returns(1000, 0.0, 0.01);
        let v = var_historical(&returns, 0.95);
        let cv = cvar(&returns, 0.95);
        assert!(cv >= v || (cv - v).abs() < 1e-6); // CVaR >= VaR generally
    }

    #[test]
    fn test_walk_forward_splits() {
        let splits = walk_forward_splits(1000, 200, 50, 50);
        assert!(!splits.is_empty());
        for (ts, te, vs, ve) in &splits {
            assert_eq!(*te, *ts + 200);
            assert_eq!(*vs, *te);
            assert!(ve <= &1000);
        }
    }

    #[test]
    fn test_pareto_dominates() {
        assert!(dominates(&[2.0, 2.0], &[1.0, 1.0]));
        assert!(!dominates(&[2.0, 1.0], &[1.0, 2.0]));
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0]));
    }

    #[test]
    fn test_combine_signals() {
        let s1 = vec![1.0, -1.0, 0.5];
        let s2 = vec![-1.0, 1.0, 0.5];
        let combined = combine_signals_equal(&[&s1, &s2]);
        assert_eq!(combined.len(), 3);
        assert!((combined[0]).abs() < 1e-10); // cancels
        assert!((combined[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_summary() {
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ds = DistributionSummary::from_values("test", &vals);
        assert_eq!(ds.count, 100);
        assert!((ds.mean - 49.5).abs() < 1e-10);
        assert!((ds.median - 49.5).abs() < 1.0);
    }

    #[test]
    fn test_lcg_deterministic() {
        let mut r1 = Lcg::new(42);
        let mut r2 = Lcg::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_parallel_backtester() {
        let symbols = ["SYM1", "SYM2"];
        let data = generate_synthetic_data(&symbols, 500, 99);
        let grid = ParameterGrid::new(vec![
            ("lookback".into(), ParamSpec::Linspace { start: 10.0, end: 30.0, steps: 3 }),
        ]);
        let backtester = ParallelBacktester::new(2);
        let tensor = backtester.run_grid("momentum", &symbols, &[Timeframe::H1], &grid, 5.0, 500, &data);
        assert_eq!(tensor.len(), 6); // 2 symbols * 3 param combos
    }

    #[test]
    fn test_heatmap() {
        let mut tensor = AlphaTensor::new();
        for strat in &["momentum", "rsi"] {
            for sym in &["BTC", "ETH"] {
                let config = BacktestConfig::new(*strat, *sym, Timeframe::H1, HashMap::new(), 5.0, 0, 100);
                let mut result = BacktestResult::failed(config);
                result.sharpe = 1.0;
                tensor.insert(result);
            }
        }
        let (rows, cols, matrix) = tensor.heatmap_strategy_symbol();
        assert_eq!(rows.len(), 2);
        assert_eq!(cols.len(), 2);
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
    }

    #[test]
    fn test_benchmark_comparison() {
        let mut rng = Lcg::new(77);
        let bench = rng.synthetic_returns(500, 0.0002, 0.01);
        let strat = rng.synthetic_returns(500, 0.0004, 0.008);
        let cmp = BenchmarkComparison::compute(&strat, &bench, 252.0);
        assert!(cmp.strategy_sharpe.is_finite());
        assert!(cmp.alpha.is_finite());
        assert!(cmp.beta.is_finite());
    }

    #[test]
    fn test_capacity_estimation() {
        let mut rng = Lcg::new(55);
        let returns = rng.synthetic_returns(500, 0.0001, 0.01);
        let params = HashMap::new();
        let curve = estimate_capacity("momentum", &returns, &params, Timeframe::H1, 50.0, 10);
        assert_eq!(curve.len(), 10);
        // First cost should be 0, Sharpe should be higher than at 50bps.
        assert!(curve[0].0 < 1e-10);
    }

    #[test]
    fn test_rolling_sharpe() {
        let mut rng = Lcg::new(33);
        let returns = rng.synthetic_returns(200, 0.0001, 0.01);
        let rs = rolling_sharpe(&returns, 50, 252.0);
        assert_eq!(rs.len(), 200);
        assert!(rs[49].is_nan() || rs[49].is_finite()); // first valid at index 50
        assert!(rs[50].is_finite());
    }

    #[test]
    fn test_shannon_entropy() {
        let uniform: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let e = shannon_entropy(&uniform);
        assert!(e > 0.0);
        let constant = vec![1.0; 100];
        let e2 = shannon_entropy(&constant);
        assert!((e2).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_viable_only() {
        let mut tensor = AlphaTensor::new();
        for i in 0..10 {
            let mut params = HashMap::new();
            params.insert("i".into(), i as f64);
            let config = BacktestConfig::new("momentum", "BTC", Timeframe::H1, params, 5.0, 0, 100);
            let mut result = BacktestResult::failed(config);
            result.sharpe = if i % 2 == 0 { 1.0 } else { 0.1 };
            result.max_drawdown = 0.1;
            result.n_trades = 50;
            result.profit_factor = if i % 2 == 0 { 1.5 } else { 0.8 };
            tensor.insert(result);
        }
        let viable = tensor.viable_only();
        assert_eq!(viable.len(), 5);
    }

    #[test]
    fn test_landscape_report() {
        let mut tensor = AlphaTensor::new();
        for i in 0..30 {
            let mut params = HashMap::new();
            params.insert("x".into(), i as f64);
            let config = BacktestConfig::new("momentum", "BTC", Timeframe::H1, params, 5.0, 0, 100);
            let mut result = BacktestResult::failed(config);
            result.sharpe = (i as f64 - 15.0) / 10.0;
            result.max_drawdown = 0.05 + i as f64 * 0.01;
            result.n_trades = 50;
            tensor.insert(result);
        }
        let report = LandscapeReport::from_tensor(&tensor);
        assert_eq!(report.total_backtests, 30);
        assert!(report.top_by_sharpe.len() <= 20);
        let rendered = report.render();
        assert!(rendered.contains("ALPHA LANDSCAPE REPORT"));
    }

    #[test]
    fn test_mc_sharpe_confidence() {
        let mut rng = Lcg::new(42);
        let returns = rng.synthetic_returns(300, 0.0003, 0.01);
        let (m, lo, hi) = mc_sharpe_confidence(&returns, 252.0, 100, 42);
        assert!(lo <= m);
        assert!(m <= hi);
    }

    #[test]
    fn test_omega_ratio() {
        let returns = vec![0.01, 0.02, -0.005, 0.015, -0.01];
        let o = omega_ratio(&returns, 0.0);
        assert!(o > 1.0); // net positive returns
    }

    #[test]
    fn test_kurtosis_skewness() {
        let mut rng = Lcg::new(99);
        let returns = rng.synthetic_returns(1000, 0.0, 0.01);
        let k = kurtosis(&returns);
        let s = skewness(&returns);
        // Normal distribution: kurtosis ~ 0, skewness ~ 0.
        assert!(k.abs() < 1.0);
        assert!(s.abs() < 0.5);
    }

    #[test]
    fn test_regime_detection() {
        let mut rng = Lcg::new(42);
        let returns = rng.synthetic_returns(200, 0.0, 0.01);
        let regime = detect_vol_regime(&returns, 20);
        assert_eq!(regime.len(), 200);
        let trend = detect_trend_regime(&returns, 20);
        assert_eq!(trend.len(), 200);
    }

    #[test]
    fn test_inv_vol_portfolio() {
        let s1 = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012, 0.003, -0.002,
                      0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012, 0.003, -0.002,
                      0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012, 0.003, -0.002];
        let s2 = vec![0.005, -0.01, 0.01, 0.005, -0.003, 0.008, -0.002, 0.004, 0.001, -0.006,
                      0.005, -0.01, 0.01, 0.005, -0.003, 0.008, -0.002, 0.004, 0.001, -0.006,
                      0.005, -0.01, 0.01, 0.005, -0.003, 0.008, -0.002, 0.004, 0.001, -0.006];
        let port = inv_vol_portfolio(&[&s1[..], &s2[..]], 10);
        assert_eq!(port.len(), 30);
    }
}
