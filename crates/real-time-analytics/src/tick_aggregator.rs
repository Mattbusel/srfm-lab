// tick_aggregator.rs — Aggregates raw ticks into OHLCV bars on multiple time bases.
//
// Supports: time bars, volume bars, dollar bars, and tick imbalance bars.
// Async channel-based: feed Tick via sender, receive Bar via receiver.

use std::collections::VecDeque;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use anyhow::Result;

// ─── Primitives ───────────────────────────────────────────────────────────────

/// A single raw trade tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub size: f64,
    /// True = buyer-initiated (aggressive buy), false = seller-initiated.
    pub is_buy: bool,
}

impl Tick {
    pub fn new(symbol: impl Into<String>, price: f64, size: f64, is_buy: bool) -> Self {
        Self {
            symbol: symbol.into(),
            timestamp: Utc::now(),
            price,
            size,
            is_buy,
        }
    }

    pub fn dollar_value(&self) -> f64 { self.price * self.size }

    pub fn signed_size(&self) -> f64 {
        if self.is_buy { self.size } else { -self.size }
    }
}

/// A completed OHLCV bar of any type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub symbol: String,
    pub bar_type: BarType,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub dollar_volume: f64,
    pub vwap: f64,
    pub num_ticks: u64,
    /// Net signed flow: sum of signed sizes. Positive = net buy pressure.
    pub imbalance: f64,
    /// Tick imbalance: |buy_ticks - sell_ticks| / num_ticks.
    pub tick_imbalance: f64,
}

impl Bar {
    fn from_state(state: &BarState, bar_type: BarType) -> Self {
        let vwap = if state.volume > 0.0 { state.dollar_volume / state.volume } else { 0.0 };
        let tick_imbalance = if state.num_ticks > 0 {
            (state.buy_ticks as f64 - state.sell_ticks as f64).abs() / state.num_ticks as f64
        } else { 0.0 };
        Self {
            symbol: state.symbol.clone(),
            bar_type,
            open_time: state.open_time,
            close_time: state.last_tick_time,
            open: state.open,
            high: state.high,
            low: state.low,
            close: state.close,
            volume: state.volume,
            dollar_volume: state.dollar_volume,
            vwap,
            num_ticks: state.num_ticks,
            imbalance: state.signed_volume,
            tick_imbalance,
        }
    }
}

/// Snapshot of the currently-open (incomplete) bar for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarSnapshot {
    pub bar: Bar,
    pub pct_complete: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BarType {
    Time,
    Volume,
    Dollar,
    Imbalance,
}

// ─── Internal Bar Construction State ─────────────────────────────────────────

#[derive(Debug, Clone)]
struct BarState {
    symbol: String,
    open_time: DateTime<Utc>,
    last_tick_time: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    dollar_volume: f64,
    signed_volume: f64,
    num_ticks: u64,
    buy_ticks: u64,
    sell_ticks: u64,
}

impl BarState {
    fn new(tick: &Tick) -> Self {
        Self {
            symbol: tick.symbol.clone(),
            open_time: tick.timestamp,
            last_tick_time: tick.timestamp,
            open: tick.price,
            high: tick.price,
            low: tick.price,
            close: tick.price,
            volume: tick.size,
            dollar_volume: tick.dollar_value(),
            signed_volume: tick.signed_size(),
            num_ticks: 1,
            buy_ticks: if tick.is_buy { 1 } else { 0 },
            sell_ticks: if tick.is_buy { 0 } else { 1 },
        }
    }

    fn update(&mut self, tick: &Tick) {
        self.last_tick_time = tick.timestamp;
        if tick.price > self.high { self.high = tick.price; }
        if tick.price < self.low { self.low = tick.price; }
        self.close = tick.price;
        self.volume += tick.size;
        self.dollar_volume += tick.dollar_value();
        self.signed_volume += tick.signed_size();
        self.num_ticks += 1;
        if tick.is_buy { self.buy_ticks += 1; } else { self.sell_ticks += 1; }
    }
}

// ─── TickAggregator ───────────────────────────────────────────────────────────

/// Aggregates raw ticks into bars on one of four sampling schemes.
///
/// # Usage
/// ```ignore
/// let (agg, bar_rx) = TickAggregator::new_volume_bar("AAPL", 10_000.0);
/// agg.sender().send(tick).await?;
/// while let Some(bar) = bar_rx.recv().await { ... }
/// ```
pub struct TickAggregator {
    sender: mpsc::Sender<Tick>,
}

impl TickAggregator {
    /// Time bar: closes every `duration`.
    pub fn new_time_bar(symbol: impl Into<String>, duration: Duration) -> (Self, mpsc::Receiver<Bar>) {
        let (tick_tx, tick_rx) = mpsc::channel(4096);
        let (bar_tx, bar_rx) = mpsc::channel(256);
        let sym = symbol.into();
        tokio::spawn(async move {
            time_bar_task(tick_rx, bar_tx, sym, duration).await;
        });
        (Self { sender: tick_tx }, bar_rx)
    }

    /// Volume bar: closes when `threshold` units of volume have traded.
    pub fn new_volume_bar(symbol: impl Into<String>, threshold: f64) -> (Self, mpsc::Receiver<Bar>) {
        let (tick_tx, tick_rx) = mpsc::channel(4096);
        let (bar_tx, bar_rx) = mpsc::channel(256);
        let sym = symbol.into();
        tokio::spawn(async move {
            threshold_bar_task(tick_rx, bar_tx, sym, BarType::Volume, threshold).await;
        });
        (Self { sender: tick_tx }, bar_rx)
    }

    /// Dollar bar: closes when `threshold` dollar value has traded.
    pub fn new_dollar_bar(symbol: impl Into<String>, threshold: f64) -> (Self, mpsc::Receiver<Bar>) {
        let (tick_tx, tick_rx) = mpsc::channel(4096);
        let (bar_tx, bar_rx) = mpsc::channel(256);
        let sym = symbol.into();
        tokio::spawn(async move {
            threshold_bar_task(tick_rx, bar_tx, sym, BarType::Dollar, threshold).await;
        });
        (Self { sender: tick_tx }, bar_rx)
    }

    /// Tick imbalance bar: closes when |buy_ticks - sell_ticks| ≥ `threshold`.
    pub fn new_imbalance_bar(symbol: impl Into<String>, threshold: u64) -> (Self, mpsc::Receiver<Bar>) {
        let (tick_tx, tick_rx) = mpsc::channel(4096);
        let (bar_tx, bar_rx) = mpsc::channel(256);
        let sym = symbol.into();
        tokio::spawn(async move {
            imbalance_bar_task(tick_rx, bar_tx, sym, threshold).await;
        });
        (Self { sender: tick_tx }, bar_rx)
    }

    pub fn sender(&self) -> &mpsc::Sender<Tick> { &self.sender }

    pub async fn send(&self, tick: Tick) -> Result<()> {
        self.sender.send(tick).await.map_err(|e| anyhow::anyhow!("tick send failed: {e}"))
    }
}

// ─── Bar Construction Tasks ───────────────────────────────────────────────────

async fn time_bar_task(
    mut tick_rx: mpsc::Receiver<Tick>,
    bar_tx: mpsc::Sender<Bar>,
    symbol: String,
    duration: Duration,
) {
    let mut current: Option<BarState> = None;
    let bar_nanos = duration.num_nanoseconds().unwrap_or(60_000_000_000) as u64;

    while let Some(tick) = tick_rx.recv().await {
        if tick.symbol != symbol { continue; }

        let tick_bucket = tick.timestamp.timestamp_nanos_opt().unwrap_or(0) as u64 / bar_nanos;

        match &mut current {
            None => {
                current = Some(BarState::new(&tick));
            }
            Some(state) => {
                let state_bucket = state.open_time.timestamp_nanos_opt().unwrap_or(0) as u64 / bar_nanos;
                if tick_bucket != state_bucket {
                    let bar = Bar::from_state(state, BarType::Time);
                    if bar_tx.send(bar).await.is_err() { return; }
                    current = Some(BarState::new(&tick));
                } else {
                    state.update(&tick);
                }
            }
        }
    }

    // Flush partial bar on channel close.
    if let Some(state) = current {
        let _ = bar_tx.send(Bar::from_state(&state, BarType::Time)).await;
    }
}

async fn threshold_bar_task(
    mut tick_rx: mpsc::Receiver<Tick>,
    bar_tx: mpsc::Sender<Bar>,
    symbol: String,
    bar_type: BarType,
    threshold: f64,
) {
    let mut current: Option<BarState> = None;

    while let Some(tick) = tick_rx.recv().await {
        if tick.symbol != symbol { continue; }

        let accumulated = match &current {
            None => { current = Some(BarState::new(&tick)); continue; }
            Some(s) => match bar_type {
                BarType::Volume => s.volume,
                BarType::Dollar => s.dollar_volume,
                _ => s.volume,
            },
        };

        let state = current.as_mut().unwrap();
        state.update(&tick);

        let new_accumulated = match bar_type {
            BarType::Volume => state.volume,
            BarType::Dollar => state.dollar_volume,
            _ => state.volume,
        };

        // Close bar when threshold crossed.
        if new_accumulated >= threshold && accumulated < threshold {
            let bar = Bar::from_state(state, bar_type);
            if bar_tx.send(bar).await.is_err() { return; }
            current = None;
        }
    }

    if let Some(state) = current {
        let _ = bar_tx.send(Bar::from_state(&state, bar_type)).await;
    }
}

async fn imbalance_bar_task(
    mut tick_rx: mpsc::Receiver<Tick>,
    bar_tx: mpsc::Sender<Bar>,
    symbol: String,
    threshold: u64,
) {
    let mut current: Option<BarState> = None;

    while let Some(tick) = tick_rx.recv().await {
        if tick.symbol != symbol { continue; }

        match &mut current {
            None => { current = Some(BarState::new(&tick)); }
            Some(state) => {
                state.update(&tick);
                let imbalance = (state.buy_ticks as i64 - state.sell_ticks as i64).unsigned_abs();
                if imbalance >= threshold {
                    let bar = Bar::from_state(state, BarType::Imbalance);
                    if bar_tx.send(bar).await.is_err() { return; }
                    current = None;
                }
            }
        }
    }

    if let Some(state) = current {
        let _ = bar_tx.send(Bar::from_state(&state, BarType::Imbalance)).await;
    }
}

// ─── Multi-Symbol Multi-Bar Dispatcher ───────────────────────────────────────

/// Routes ticks for multiple symbols to per-symbol aggregators.
pub struct MultiSymbolAggregator {
    senders: std::collections::HashMap<String, mpsc::Sender<Tick>>,
}

impl MultiSymbolAggregator {
    pub fn new() -> Self {
        Self { senders: std::collections::HashMap::new() }
    }

    /// Register a symbol with a pre-created TickAggregator.
    pub fn register(&mut self, symbol: impl Into<String>, agg: &TickAggregator) {
        self.senders.insert(symbol.into(), agg.sender.clone());
    }

    /// Route a tick to the appropriate symbol's aggregator.
    pub async fn dispatch(&self, tick: Tick) -> Result<()> {
        if let Some(tx) = self.senders.get(&tick.symbol) {
            tx.send(tick).await.map_err(|e| anyhow::anyhow!("dispatch failed: {e}"))?;
        }
        Ok(())
    }
}

impl Default for MultiSymbolAggregator {
    fn default() -> Self { Self::new() }
}

// ─── Bar Buffer ───────────────────────────────────────────────────────────────

/// Fixed-size ring buffer for completed bars, useful for pattern recognition.
pub struct BarBuffer {
    buf: VecDeque<Bar>,
    capacity: usize,
}

impl BarBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { buf: VecDeque::with_capacity(capacity), capacity }
    }

    pub fn push(&mut self, bar: Bar) {
        if self.buf.len() == self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(bar);
    }

    pub fn len(&self) -> usize { self.buf.len() }
    pub fn is_full(&self) -> bool { self.buf.len() == self.capacity }
    pub fn last(&self) -> Option<&Bar> { self.buf.back() }
    pub fn bars(&self) -> &VecDeque<Bar> { &self.buf }

    /// Returns (open, high, low, close) arrays for the last N bars.
    pub fn ohlc_arrays(&self, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let bars: Vec<&Bar> = self.buf.iter().rev().take(n).collect::<Vec<_>>()
            .into_iter().rev().collect();
        let o = bars.iter().map(|b| b.open).collect();
        let h = bars.iter().map(|b| b.high).collect();
        let l = bars.iter().map(|b| b.low).collect();
        let c = bars.iter().map(|b| b.close).collect();
        (o, h, l, c)
    }

    /// Returns close prices for last N bars.
    pub fn closes(&self, n: usize) -> Vec<f64> {
        self.buf.iter().rev().take(n).map(|b| b.close).collect::<Vec<_>>()
            .into_iter().rev().collect()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    fn make_tick(sym: &str, price: f64, size: f64, is_buy: bool) -> Tick {
        Tick::new(sym, price, size, is_buy)
    }

    #[test]
    fn test_volume_bar_closes_at_threshold() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let (agg, mut bar_rx) = TickAggregator::new_volume_bar("AAPL", 100.0);
            for _ in 0..10 {
                agg.send(make_tick("AAPL", 150.0, 10.0, true)).await.unwrap();
            }
            // One bar should be ready (100 volume = 10 ticks × 10).
            drop(agg);
            let bar = bar_rx.recv().await.unwrap();
            assert_eq!(bar.num_ticks, 10);
            assert!((bar.volume - 100.0).abs() < 1e-9);
        });
    }

    #[test]
    fn test_dollar_bar() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let (agg, mut bar_rx) = TickAggregator::new_dollar_bar("SPY", 10_000.0);
            // 10 ticks × 100 shares × $100 = $100,000 > threshold after 1 tick
            for _ in 0..5 {
                agg.send(make_tick("SPY", 100.0, 100.0, false)).await.unwrap();
            }
            drop(agg);
            let bar = bar_rx.recv().await;
            assert!(bar.is_some());
        });
    }

    #[test]
    fn test_bar_buffer_ring() {
        let mut buf = BarBuffer::new(3);
        for i in 0..5 {
            let p = i as f64;
            buf.push(Bar {
                symbol: "X".into(),
                bar_type: BarType::Volume,
                open_time: Utc::now(),
                close_time: Utc::now(),
                open: p, high: p, low: p, close: p,
                volume: 1.0, dollar_volume: p, vwap: p,
                num_ticks: 1, imbalance: 0.0, tick_imbalance: 0.0,
            });
        }
        assert_eq!(buf.len(), 3);
        assert!((buf.last().unwrap().close - 4.0).abs() < 1e-9);
    }
}
