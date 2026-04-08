// strategy.rs — Strategy trait, signal-based, multi-signal composite, regime filter, position sizing
use crate::data::Bar;
use crate::portfolio::{Portfolio, TradeSide};
use crate::engine::{Order, OrderType};
use std::collections::HashMap;

/// Signal value: [-1, 1] where -1 = max short, +1 = max long
#[derive(Clone, Debug)]
pub struct Signal {
    pub symbol: String,
    pub value: f64,
    pub confidence: f64,
    pub timestamp: u64,
    pub source: String,
}

impl Signal {
    pub fn new(symbol: &str, value: f64, confidence: f64, ts: u64) -> Self {
        Self { symbol: symbol.to_string(), value: value.max(-1.0).min(1.0), confidence: confidence.max(0.0).min(1.0), timestamp: ts, source: String::new() }
    }

    pub fn with_source(mut self, source: &str) -> Self { self.source = source.to_string(); self }
    pub fn is_long(&self) -> bool { self.value > 0.0 }
    pub fn is_short(&self) -> bool { self.value < 0.0 }
    pub fn is_neutral(&self) -> bool { self.value.abs() < 1e-10 }
    pub fn direction(&self) -> f64 { self.value.signum() }
    pub fn strength(&self) -> f64 { self.value.abs() }
    pub fn weighted_value(&self) -> f64 { self.value * self.confidence }
}

/// Generic strategy interface
pub trait Strategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, portfolio: &Portfolio) -> Vec<Signal>;
    fn name(&self) -> &str;
}

/// Simple moving average crossover
#[derive(Clone, Debug)]
pub struct SMACrossover {
    pub short_window: usize,
    pub long_window: usize,
    pub price_history: HashMap<String, Vec<f64>>,
}

impl SMACrossover {
    pub fn new(short: usize, long: usize) -> Self {
        Self { short_window: short, long_window: long, price_history: HashMap::new() }
    }

    fn sma(prices: &[f64], window: usize) -> f64 {
        if prices.len() < window { return f64::NAN; }
        let sl = &prices[prices.len() - window..];
        sl.iter().sum::<f64>() / window as f64
    }
}

impl Strategy for SMACrossover {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, _portfolio: &Portfolio) -> Vec<Signal> {
        let mut signals = Vec::new();
        for (sym, bar) in bars {
            let history = self.price_history.entry(sym.to_string()).or_default();
            history.push(bar.close);
            if history.len() < self.long_window { continue; }
            let short_sma = Self::sma(history, self.short_window);
            let long_sma = Self::sma(history, self.long_window);
            if short_sma.is_nan() || long_sma.is_nan() { continue; }
            let value = if short_sma > long_sma { 1.0 } else { -1.0 };
            signals.push(Signal::new(sym, value, 1.0, timestamp).with_source("SMA"));
        }
        signals
    }
    fn name(&self) -> &str { "SMA_Crossover" }
}

/// RSI-based mean reversion
#[derive(Clone, Debug)]
pub struct RSIStrategy {
    pub period: usize,
    pub overbought: f64,
    pub oversold: f64,
    pub price_history: HashMap<String, Vec<f64>>,
}

impl RSIStrategy {
    pub fn new(period: usize, overbought: f64, oversold: f64) -> Self {
        Self { period, overbought, oversold, price_history: HashMap::new() }
    }

    fn compute_rsi(prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 { return 50.0; }
        let mut gains = 0.0;
        let mut losses = 0.0;
        let start = prices.len() - period - 1;
        for i in (start + 1)..prices.len() {
            let diff = prices[i] - prices[i - 1];
            if diff > 0.0 { gains += diff; } else { losses -= diff; }
        }
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        if avg_loss < 1e-15 { return 100.0; }
        let rs = avg_gain / avg_loss;
        100.0 - 100.0 / (1.0 + rs)
    }
}

impl Strategy for RSIStrategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, _portfolio: &Portfolio) -> Vec<Signal> {
        let mut signals = Vec::new();
        for (sym, bar) in bars {
            let history = self.price_history.entry(sym.to_string()).or_default();
            history.push(bar.close);
            let rsi = Self::compute_rsi(history, self.period);
            let value = if rsi > self.overbought {
                -((rsi - self.overbought) / (100.0 - self.overbought)).min(1.0)
            } else if rsi < self.oversold {
                ((self.oversold - rsi) / self.oversold).min(1.0)
            } else {
                0.0
            };
            signals.push(Signal::new(sym, value, 0.8, timestamp).with_source("RSI"));
        }
        signals
    }
    fn name(&self) -> &str { "RSI_MeanReversion" }
}

/// Bollinger Band breakout
#[derive(Clone, Debug)]
pub struct BollingerStrategy {
    pub period: usize,
    pub num_std: f64,
    pub price_history: HashMap<String, Vec<f64>>,
}

impl BollingerStrategy {
    pub fn new(period: usize, num_std: f64) -> Self {
        Self { period, num_std, price_history: HashMap::new() }
    }
}

impl Strategy for BollingerStrategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, _portfolio: &Portfolio) -> Vec<Signal> {
        let mut signals = Vec::new();
        for (sym, bar) in bars {
            let history = self.price_history.entry(sym.to_string()).or_default();
            history.push(bar.close);
            if history.len() < self.period { continue; }
            let sl = &history[history.len() - self.period..];
            let mean = sl.iter().sum::<f64>() / self.period as f64;
            let var = sl.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / self.period as f64;
            let std = var.sqrt();
            let upper = mean + self.num_std * std;
            let lower = mean - self.num_std * std;
            let value = if bar.close > upper { -1.0 } else if bar.close < lower { 1.0 } else { 0.0 };
            let confidence = if std > 1e-10 { ((bar.close - mean) / std).abs().min(3.0) / 3.0 } else { 0.0 };
            signals.push(Signal::new(sym, value, confidence, timestamp).with_source("Bollinger"));
        }
        signals
    }
    fn name(&self) -> &str { "Bollinger_Breakout" }
}

/// Momentum strategy
#[derive(Clone, Debug)]
pub struct MomentumStrategy {
    pub lookback: usize,
    pub price_history: HashMap<String, Vec<f64>>,
}

impl MomentumStrategy {
    pub fn new(lookback: usize) -> Self {
        Self { lookback, price_history: HashMap::new() }
    }
}

impl Strategy for MomentumStrategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, _portfolio: &Portfolio) -> Vec<Signal> {
        let mut signals = Vec::new();
        for (sym, bar) in bars {
            let history = self.price_history.entry(sym.to_string()).or_default();
            history.push(bar.close);
            if history.len() < self.lookback + 1 { continue; }
            let past = history[history.len() - self.lookback - 1];
            let ret = (bar.close - past) / past;
            let value = ret.max(-1.0).min(1.0);
            signals.push(Signal::new(sym, value, 0.7, timestamp).with_source("Momentum"));
        }
        signals
    }
    fn name(&self) -> &str { "Momentum" }
}

/// Multi-signal composite strategy
pub struct CompositeStrategy {
    pub strategies: Vec<Box<dyn Strategy>>,
    pub weights: Vec<f64>,
    pub combination: CombinationMethod,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CombinationMethod {
    WeightedAverage,
    MajorityVote,
    MaxConfidence,
    RankAverage,
}

impl CompositeStrategy {
    pub fn new(strategies: Vec<Box<dyn Strategy>>, weights: Vec<f64>, method: CombinationMethod) -> Self {
        Self { strategies, weights, combination: method }
    }

    pub fn equal_weight(strategies: Vec<Box<dyn Strategy>>) -> Self {
        let n = strategies.len();
        let w = vec![1.0 / n as f64; n];
        Self::new(strategies, w, CombinationMethod::WeightedAverage)
    }
}

impl Strategy for CompositeStrategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, portfolio: &Portfolio) -> Vec<Signal> {
        let mut all_signals: Vec<Vec<Signal>> = Vec::new();
        for strat in self.strategies.iter_mut() {
            all_signals.push(strat.generate_signals(timestamp, bars, portfolio));
        }

        // Group by symbol
        let mut by_symbol: HashMap<String, Vec<(f64, f64, usize)>> = HashMap::new(); // (value, confidence, strategy_idx)
        for (si, sigs) in all_signals.iter().enumerate() {
            for sig in sigs {
                by_symbol.entry(sig.symbol.clone()).or_default().push((sig.value, sig.confidence, si));
            }
        }

        let mut combined = Vec::new();
        for (sym, entries) in &by_symbol {
            let value = match self.combination {
                CombinationMethod::WeightedAverage => {
                    let mut sum = 0.0;
                    let mut w_sum = 0.0;
                    for &(v, c, si) in entries {
                        let w = self.weights[si] * c;
                        sum += v * w;
                        w_sum += w;
                    }
                    if w_sum > 1e-15 { sum / w_sum } else { 0.0 }
                }
                CombinationMethod::MajorityVote => {
                    let long = entries.iter().filter(|&&(v, _, _)| v > 0.0).count();
                    let short = entries.iter().filter(|&&(v, _, _)| v < 0.0).count();
                    if long > short { 1.0 } else if short > long { -1.0 } else { 0.0 }
                }
                CombinationMethod::MaxConfidence => {
                    entries.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|&(v, _, _)| v).unwrap_or(0.0)
                }
                CombinationMethod::RankAverage => {
                    let mut vals: Vec<f64> = entries.iter().map(|&(v, _, _)| v).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    vals.iter().sum::<f64>() / vals.len() as f64
                }
            };
            let confidence = entries.iter().map(|&(_, c, _)| c).sum::<f64>() / entries.len() as f64;
            combined.push(Signal::new(sym, value, confidence, timestamp).with_source("Composite"));
        }
        combined
    }
    fn name(&self) -> &str { "Composite" }
}

/// Regime filter: only pass signals through when regime matches
pub struct RegimeFilteredStrategy {
    pub inner: Box<dyn Strategy>,
    pub regime_detector: Box<dyn Strategy>,
    pub allowed_regimes: Vec<i32>, // discretized regime values
    pub current_regime: i32,
}

impl RegimeFilteredStrategy {
    pub fn new(inner: Box<dyn Strategy>, detector: Box<dyn Strategy>, allowed: Vec<i32>) -> Self {
        Self { inner, regime_detector: detector, allowed_regimes: allowed, current_regime: 0 }
    }
}

impl Strategy for RegimeFilteredStrategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, portfolio: &Portfolio) -> Vec<Signal> {
        let regime_signals = self.regime_detector.generate_signals(timestamp, bars, portfolio);
        if let Some(rs) = regime_signals.first() {
            self.current_regime = rs.value.round() as i32;
        }
        if !self.allowed_regimes.contains(&self.current_regime) {
            return vec![]; // no signals when regime doesn't match
        }
        self.inner.generate_signals(timestamp, bars, portfolio)
    }
    fn name(&self) -> &str { "RegimeFiltered" }
}

/// Position sizing methods
#[derive(Clone, Debug)]
pub enum PositionSizer {
    FixedFraction(f64),
    FixedDollar(f64),
    Kelly { lookback: usize },
    VolTarget { target_vol: f64, lookback: usize },
    RiskParity,
    EqualWeight,
    InverseVolatility { lookback: usize },
    MaxSharpeSizing,
    ATRBased { atr_period: usize, risk_per_trade: f64 },
}

/// Position sizing context
#[derive(Clone, Debug)]
pub struct SizingContext {
    pub equity: f64,
    pub signal: f64,
    pub confidence: f64,
    pub current_position: f64,
    pub price: f64,
    pub volatility: f64,
    pub atr: f64,
    pub returns_history: Vec<f64>,
}

impl PositionSizer {
    pub fn compute_size(&self, ctx: &SizingContext) -> f64 {
        let direction = ctx.signal.signum();
        let raw_size = match self {
            PositionSizer::FixedFraction(frac) => {
                let notional = ctx.equity * frac * ctx.signal.abs();
                notional / ctx.price
            }
            PositionSizer::FixedDollar(amount) => {
                amount * ctx.signal.abs() / ctx.price
            }
            PositionSizer::Kelly { lookback } => {
                let rets = &ctx.returns_history;
                if rets.len() < *lookback { return 0.0; }
                let window = &rets[rets.len() - lookback..];
                let wins: Vec<&f64> = window.iter().filter(|&&r| r > 0.0).collect();
                let losses: Vec<&f64> = window.iter().filter(|&&r| r < 0.0).collect();
                if losses.is_empty() || wins.is_empty() { return 0.0; }
                let win_rate = wins.len() as f64 / window.len() as f64;
                let avg_win = wins.iter().copied().sum::<f64>() / wins.len() as f64;
                let avg_loss = losses.iter().copied().sum::<f64>().abs() / losses.len() as f64;
                if avg_loss < 1e-15 { return 0.0; }
                let kelly = win_rate - (1.0 - win_rate) / (avg_win / avg_loss);
                let frac = kelly.max(0.0).min(0.25); // half-Kelly cap
                ctx.equity * frac * ctx.signal.abs() / ctx.price
            }
            PositionSizer::VolTarget { target_vol, lookback } => {
                let vol = if ctx.volatility > 1e-10 { ctx.volatility } else {
                    if ctx.returns_history.len() < *lookback { return 0.0; }
                    let w = &ctx.returns_history[ctx.returns_history.len() - lookback..];
                    let mean = w.iter().sum::<f64>() / w.len() as f64;
                    let var = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / w.len() as f64;
                    var.sqrt() * (252.0f64).sqrt()
                };
                if vol < 1e-10 { return 0.0; }
                let scale = target_vol / vol;
                ctx.equity * scale * ctx.signal.abs() / ctx.price
            }
            PositionSizer::ATRBased { atr_period: _, risk_per_trade } => {
                if ctx.atr < 1e-10 { return 0.0; }
                let risk_dollars = ctx.equity * risk_per_trade;
                risk_dollars / ctx.atr
            }
            PositionSizer::EqualWeight => {
                ctx.equity / ctx.price * ctx.signal.abs()
            }
            PositionSizer::InverseVolatility { lookback } => {
                if ctx.volatility < 1e-10 { return 0.0; }
                let inv_vol = 1.0 / ctx.volatility;
                ctx.equity * 0.1 * inv_vol * ctx.signal.abs() / ctx.price // scaled
            }
            _ => ctx.equity * 0.1 * ctx.signal.abs() / ctx.price,
        };
        raw_size * direction
    }
}

/// Signal to orders converter
pub fn signals_to_orders(
    signals: &[Signal],
    portfolio: &Portfolio,
    sizer: &PositionSizer,
    prices: &HashMap<String, f64>,
) -> Vec<Order> {
    let mut orders = Vec::new();
    for signal in signals {
        let price = match prices.get(&signal.symbol) {
            Some(&p) => p,
            None => continue,
        };
        let current_qty = portfolio.position_qty(&signal.symbol);
        let ctx = SizingContext {
            equity: portfolio.total_equity(),
            signal: signal.value,
            confidence: signal.confidence,
            current_position: current_qty,
            price,
            volatility: 0.0,
            atr: 0.0,
            returns_history: Vec::new(),
        };
        let target_qty = sizer.compute_size(&ctx);
        let delta = target_qty - current_qty;

        if delta.abs() < 1e-6 { continue; }

        let (side, qty) = if delta > 0.0 {
            (TradeSide::Buy, delta)
        } else {
            (TradeSide::Sell, delta.abs())
        };

        orders.push(Order::market(0, &signal.symbol, side, qty));
    }
    orders
}

/// Risk manager: applies constraints to orders
#[derive(Clone, Debug)]
pub struct RiskManager {
    pub max_position_pct: f64,
    pub max_total_exposure_pct: f64,
    pub max_single_order_pct: f64,
    pub max_daily_loss_pct: f64,
    pub daily_loss: f64,
    pub stop_loss_pct: Option<f64>,
    pub take_profit_pct: Option<f64>,
}

impl RiskManager {
    pub fn default_rm() -> Self {
        Self {
            max_position_pct: 0.2,
            max_total_exposure_pct: 1.0,
            max_single_order_pct: 0.1,
            max_daily_loss_pct: 0.05,
            daily_loss: 0.0,
            stop_loss_pct: None,
            take_profit_pct: None,
        }
    }

    pub fn conservative() -> Self {
        Self {
            max_position_pct: 0.05,
            max_total_exposure_pct: 0.5,
            max_single_order_pct: 0.02,
            max_daily_loss_pct: 0.02,
            daily_loss: 0.0,
            stop_loss_pct: Some(0.02),
            take_profit_pct: Some(0.05),
        }
    }

    pub fn filter_orders(&self, orders: &[Order], portfolio: &Portfolio) -> Vec<Order> {
        let equity = portfolio.total_equity();
        if equity < 1e-10 { return vec![]; }

        // Check daily loss limit
        if self.daily_loss / equity > self.max_daily_loss_pct {
            return vec![];
        }

        let mut filtered = Vec::new();
        for order in orders {
            let price = order.fill_price.unwrap_or(100.0); // estimate
            let order_notional = order.quantity * price;

            // Max single order
            if order_notional / equity > self.max_single_order_pct {
                let max_qty = equity * self.max_single_order_pct / price;
                let mut adj = order.clone();
                adj.quantity = max_qty;
                filtered.push(adj);
                continue;
            }

            // Max position size
            let current_exposure = portfolio.position(order.symbol.as_str())
                .map_or(0.0, |p| p.notional_exposure());
            let new_exposure = current_exposure + order_notional;
            if new_exposure / equity > self.max_position_pct {
                continue;
            }

            // Max total exposure
            let total = portfolio.total_exposure() + order_notional;
            if total / equity > self.max_total_exposure_pct {
                continue;
            }

            filtered.push(order.clone());
        }
        filtered
    }

    pub fn check_stop_orders(&self, portfolio: &Portfolio) -> Vec<Order> {
        let mut orders = Vec::new();
        for (sym, pos) in &portfolio.positions {
            if pos.is_flat() { continue; }
            let pnl_pct = pos.return_pct();

            if let Some(sl) = self.stop_loss_pct {
                if pnl_pct < -sl {
                    let side = if pos.is_long() { TradeSide::Sell } else { TradeSide::Buy };
                    orders.push(Order::market(0, sym, side, pos.quantity.abs()));
                }
            }
            if let Some(tp) = self.take_profit_pct {
                if pnl_pct > tp {
                    let side = if pos.is_long() { TradeSide::Sell } else { TradeSide::Buy };
                    orders.push(Order::market(0, sym, side, pos.quantity.abs()));
                }
            }
        }
        orders
    }

    pub fn update_daily_loss(&mut self, loss: f64) { self.daily_loss += loss; }
    pub fn reset_daily(&mut self) { self.daily_loss = 0.0; }
}

/// Signal smoothing / filtering
pub fn ema_signal(signals: &[f64], alpha: f64) -> Vec<f64> {
    if signals.is_empty() { return vec![]; }
    let mut result = vec![signals[0]];
    for i in 1..signals.len() {
        result.push(alpha * signals[i] + (1.0 - alpha) * result[i - 1]);
    }
    result
}

pub fn signal_threshold(signal: f64, threshold: f64) -> f64 {
    if signal.abs() < threshold { 0.0 } else { signal }
}

pub fn signal_discretize(signal: f64, levels: &[f64]) -> f64 {
    let mut best = levels[0];
    let mut best_dist = (signal - levels[0]).abs();
    for &l in &levels[1..] {
        let d = (signal - l).abs();
        if d < best_dist { best = l; best_dist = d; }
    }
    best
}

/// Turnover calculator
pub fn compute_turnover(weights_before: &HashMap<String, f64>, weights_after: &HashMap<String, f64>) -> f64 {
    let mut turnover = 0.0;
    let mut all_symbols: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for k in weights_before.keys() { all_symbols.insert(k.as_str()); }
    for k in weights_after.keys() { all_symbols.insert(k.as_str()); }

    for sym in all_symbols {
        let w_before = weights_before.get(sym).copied().unwrap_or(0.0);
        let w_after = weights_after.get(sym).copied().unwrap_or(0.0);
        turnover += (w_after - w_before).abs();
    }
    turnover / 2.0
}

/// Mean reversion strategy: fade large moves
#[derive(Clone, Debug)]
pub struct MeanReversionStrategy {
    pub lookback: usize,
    pub entry_z: f64,
    pub exit_z: f64,
    pub price_history: HashMap<String, Vec<f64>>,
}

impl MeanReversionStrategy {
    pub fn new(lookback: usize, entry_z: f64, exit_z: f64) -> Self {
        Self { lookback, entry_z, exit_z, price_history: HashMap::new() }
    }
}

impl Strategy for MeanReversionStrategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, _portfolio: &Portfolio) -> Vec<Signal> {
        let mut signals = Vec::new();
        for (sym, bar) in bars {
            let history = self.price_history.entry(sym.to_string()).or_default();
            history.push(bar.close);
            if history.len() < self.lookback { continue; }
            let sl = &history[history.len() - self.lookback..];
            let mean = sl.iter().sum::<f64>() / self.lookback as f64;
            let std = (sl.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / self.lookback as f64).sqrt();
            if std < 1e-15 { continue; }
            let z = (bar.close - mean) / std;
            let value = if z > self.entry_z { -(z / self.entry_z).min(1.0) }
                else if z < -self.entry_z { (-z / self.entry_z).min(1.0) }
                else if z.abs() < self.exit_z { 0.0 }
                else { 0.0 };
            signals.push(Signal::new(sym, value, (z.abs() / 3.0).min(1.0), timestamp).with_source("MeanRev"));
        }
        signals
    }
    fn name(&self) -> &str { "MeanReversion" }
}

/// Pairs trading strategy
#[derive(Clone, Debug)]
pub struct PairsTradingStrategy {
    pub symbol_a: String,
    pub symbol_b: String,
    pub lookback: usize,
    pub entry_z: f64,
    pub exit_z: f64,
    pub hedge_ratio: f64,
    pub spread_history: Vec<f64>,
    pub price_a_history: Vec<f64>,
    pub price_b_history: Vec<f64>,
}

impl PairsTradingStrategy {
    pub fn new(sym_a: &str, sym_b: &str, lookback: usize, entry_z: f64, hedge_ratio: f64) -> Self {
        Self {
            symbol_a: sym_a.to_string(), symbol_b: sym_b.to_string(),
            lookback, entry_z, exit_z: 0.5, hedge_ratio,
            spread_history: Vec::new(), price_a_history: Vec::new(), price_b_history: Vec::new(),
        }
    }
}

impl Strategy for PairsTradingStrategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, _portfolio: &Portfolio) -> Vec<Signal> {
        let price_a = bars.get(self.symbol_a.as_str()).map(|b| b.close);
        let price_b = bars.get(self.symbol_b.as_str()).map(|b| b.close);
        let (pa, pb) = match (price_a, price_b) {
            (Some(a), Some(b)) => (a, b),
            _ => return vec![],
        };
        self.price_a_history.push(pa);
        self.price_b_history.push(pb);
        let spread = pa - self.hedge_ratio * pb;
        self.spread_history.push(spread);
        if self.spread_history.len() < self.lookback { return vec![]; }
        let sl = &self.spread_history[self.spread_history.len() - self.lookback..];
        let mean = sl.iter().sum::<f64>() / self.lookback as f64;
        let std = (sl.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / self.lookback as f64).sqrt();
        if std < 1e-15 { return vec![]; }
        let z = (spread - mean) / std;
        let mut signals = Vec::new();
        if z > self.entry_z {
            signals.push(Signal::new(&self.symbol_a, -1.0, (z / 3.0).min(1.0), timestamp).with_source("Pairs"));
            signals.push(Signal::new(&self.symbol_b, 1.0, (z / 3.0).min(1.0), timestamp).with_source("Pairs"));
        } else if z < -self.entry_z {
            signals.push(Signal::new(&self.symbol_a, 1.0, (-z / 3.0).min(1.0), timestamp).with_source("Pairs"));
            signals.push(Signal::new(&self.symbol_b, -1.0, (-z / 3.0).min(1.0), timestamp).with_source("Pairs"));
        } else if z.abs() < self.exit_z {
            signals.push(Signal::new(&self.symbol_a, 0.0, 1.0, timestamp).with_source("Pairs"));
            signals.push(Signal::new(&self.symbol_b, 0.0, 1.0, timestamp).with_source("Pairs"));
        }
        signals
    }
    fn name(&self) -> &str { "PairsTrading" }
}

/// Breakout strategy: buy on new high, sell on new low
#[derive(Clone, Debug)]
pub struct BreakoutStrategy {
    pub lookback: usize,
    pub price_history: HashMap<String, Vec<f64>>,
}

impl BreakoutStrategy {
    pub fn new(lookback: usize) -> Self {
        Self { lookback, price_history: HashMap::new() }
    }
}

impl Strategy for BreakoutStrategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, _portfolio: &Portfolio) -> Vec<Signal> {
        let mut signals = Vec::new();
        for (sym, bar) in bars {
            let history = self.price_history.entry(sym.to_string()).or_default();
            history.push(bar.close);
            if history.len() < self.lookback + 1 { continue; }
            let window = &history[history.len() - self.lookback - 1..history.len() - 1];
            let high = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let low = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let value = if bar.close > high { 1.0 }
                else if bar.close < low { -1.0 }
                else { 0.0 };
            signals.push(Signal::new(sym, value, 0.7, timestamp).with_source("Breakout"));
        }
        signals
    }
    fn name(&self) -> &str { "Breakout" }
}

/// Volume-weighted strategy
#[derive(Clone, Debug)]
pub struct VolumeStrategy {
    pub vol_lookback: usize,
    pub vol_threshold: f64,
    pub volume_history: HashMap<String, Vec<f64>>,
    pub price_history: HashMap<String, Vec<f64>>,
}

impl VolumeStrategy {
    pub fn new(lookback: usize, threshold: f64) -> Self {
        Self { vol_lookback: lookback, vol_threshold: threshold, volume_history: HashMap::new(), price_history: HashMap::new() }
    }
}

impl Strategy for VolumeStrategy {
    fn generate_signals(&mut self, timestamp: u64, bars: &HashMap<String, &Bar>, _portfolio: &Portfolio) -> Vec<Signal> {
        let mut signals = Vec::new();
        for (sym, bar) in bars {
            let vhist = self.volume_history.entry(sym.to_string()).or_default();
            let phist = self.price_history.entry(sym.to_string()).or_default();
            vhist.push(bar.volume);
            phist.push(bar.close);
            if vhist.len() < self.vol_lookback + 1 { continue; }
            let avg_vol = vhist[vhist.len() - self.vol_lookback - 1..vhist.len() - 1].iter().sum::<f64>() / self.vol_lookback as f64;
            let vol_ratio = bar.volume / avg_vol.max(1.0);
            if vol_ratio > self.vol_threshold {
                let ret = (bar.close - phist[phist.len() - 2]) / phist[phist.len() - 2];
                let value = ret.signum();
                let confidence = ((vol_ratio - 1.0) / 3.0).min(1.0);
                signals.push(Signal::new(sym, value, confidence, timestamp).with_source("Volume"));
            }
        }
        signals
    }
    fn name(&self) -> &str { "Volume" }
}

/// Dual-timeframe strategy: use slow TF for direction, fast for entry
pub fn dual_timeframe_signal(
    fast_signal: f64, slow_signal: f64, alignment_weight: f64,
) -> f64 {
    if fast_signal.signum() == slow_signal.signum() {
        fast_signal * alignment_weight
    } else {
        fast_signal * (1.0 - alignment_weight) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Bar;

    #[test]
    fn test_signal() {
        let s = Signal::new("AAPL", 0.8, 0.9, 1000);
        assert!(s.is_long());
        assert!((s.weighted_value() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_sma_crossover() {
        let mut strat = SMACrossover::new(5, 20);
        let mut bars: HashMap<String, &Bar> = HashMap::new();
        let bar = Bar::new(1, 100.0, 105.0, 95.0, 103.0, 1000.0);
        bars.insert("TEST".to_string(), &bar);
        let portfolio = Portfolio::new(100000.0);
        // Need to feed enough bars to compute SMA
        for _ in 0..25 {
            strat.generate_signals(1, &bars, &portfolio);
        }
    }

    #[test]
    fn test_position_sizer_fixed_frac() {
        let sizer = PositionSizer::FixedFraction(0.1);
        let ctx = SizingContext {
            equity: 100000.0, signal: 1.0, confidence: 1.0,
            current_position: 0.0, price: 50.0, volatility: 0.2,
            atr: 2.0, returns_history: vec![],
        };
        let size = sizer.compute_size(&ctx);
        assert!((size - 200.0).abs() < 1e-10); // 100000 * 0.1 / 50
    }

    #[test]
    fn test_risk_manager() {
        let rm = RiskManager::default_rm();
        let portfolio = Portfolio::new(100000.0);
        let orders = vec![Order::market(1, "TEST", TradeSide::Buy, 1000.0)];
        let filtered = rm.filter_orders(&orders, &portfolio);
        // With estimated price 100, order = 100K which is 100% — should be limited
    }

    #[test]
    fn test_ema_signal() {
        let signals = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let smoothed = ema_signal(&signals, 0.3);
        assert_eq!(smoothed.len(), 5);
        assert!(smoothed[1] < 1.0); // smoothed down from 1
    }

    #[test]
    fn test_turnover() {
        let mut before = HashMap::new();
        before.insert("A".to_string(), 0.5);
        before.insert("B".to_string(), 0.5);
        let mut after = HashMap::new();
        after.insert("A".to_string(), 0.3);
        after.insert("B".to_string(), 0.7);
        let t = compute_turnover(&before, &after);
        assert!((t - 0.2).abs() < 1e-10);
    }
}
