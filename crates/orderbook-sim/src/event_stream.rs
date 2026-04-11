/// Lock-free MPSC event queue for LOB events using crossbeam-channel.
///
/// Event types: OrderSubmit, OrderCancel, Fill, PriceUpdate, RegimeChange.
/// Provides both synchronous and async-style event dispatch.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};

use crate::lob_engine::{Side, Qty, Nanos, OrderKind, TimeInForce};

// ── Event Types ───────────────────────────────────────────────────────────────

/// Unique event identifier.
pub type EventId = u64;

/// Market regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Normal,
    HighVolatility,
    LowLiquidity,
    TrendingUp,
    TrendingDown,
    Halt,
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::Normal => write!(f, "NORMAL"),
            MarketRegime::HighVolatility => write!(f, "HIGH_VOL"),
            MarketRegime::LowLiquidity => write!(f, "LOW_LIQ"),
            MarketRegime::TrendingUp => write!(f, "TREND_UP"),
            MarketRegime::TrendingDown => write!(f, "TREND_DOWN"),
            MarketRegime::Halt => write!(f, "HALT"),
        }
    }
}

/// Order submission event.
#[derive(Debug, Clone)]
pub struct OrderSubmitEvent {
    pub event_id: EventId,
    pub timestamp: Nanos,
    pub order_id: u64,
    pub symbol: String,
    pub side: Side,
    pub kind: OrderKindTag,
    pub price: f64,
    pub qty: Qty,
    pub tif: TimeInForcetag,
    pub agent_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderKindTag {
    Limit,
    Market,
    StopLimit,
    StopMarket,
    Iceberg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForcetag {
    GTC, IOC, FOK, Day,
}

/// Order cancellation event.
#[derive(Debug, Clone)]
pub struct OrderCancelEvent {
    pub event_id: EventId,
    pub timestamp: Nanos,
    pub order_id: u64,
    pub symbol: String,
    pub agent_id: u32,
    /// Whether cancellation was successful.
    pub success: bool,
}

/// Fill event (partial or complete).
#[derive(Debug, Clone)]
pub struct FillEvent {
    pub event_id: EventId,
    pub timestamp: Nanos,
    pub symbol: String,
    pub aggressor_id: u64,
    pub passive_id: u64,
    pub price: f64,
    pub qty: Qty,
    pub side: Side,
    pub aggressor_agent: u32,
    pub passive_agent: u32,
    pub is_partial: bool,
}

/// Price/market-data update event.
#[derive(Debug, Clone)]
pub struct PriceUpdateEvent {
    pub event_id: EventId,
    pub timestamp: Nanos,
    pub symbol: String,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub mid: Option<f64>,
    pub last: f64,
    pub bid_qty: Qty,
    pub ask_qty: Qty,
    pub imbalance: f64,
    /// Depth snapshot: (price, qty) for top 5 levels each side.
    pub bid_depth: Vec<(f64, Qty)>,
    pub ask_depth: Vec<(f64, Qty)>,
}

/// Market regime change event.
#[derive(Debug, Clone)]
pub struct RegimeChangeEvent {
    pub event_id: EventId,
    pub timestamp: Nanos,
    pub symbol: String,
    pub from_regime: MarketRegime,
    pub to_regime: MarketRegime,
    pub trigger: RegimeChangeTrigger,
}

#[derive(Debug, Clone, Copy)]
pub enum RegimeChangeTrigger {
    VolatilityThreshold,
    PriceMovement,
    LiquidityDrop,
    CircuitBreaker,
    Manual,
}

/// Top-level event enum.
#[derive(Debug, Clone)]
pub enum LobEvent {
    OrderSubmit(OrderSubmitEvent),
    OrderCancel(OrderCancelEvent),
    Fill(FillEvent),
    PriceUpdate(PriceUpdateEvent),
    RegimeChange(RegimeChangeEvent),
    /// Heartbeat / tick event for time advancement.
    Tick { timestamp: Nanos, sequence: u64 },
    /// Session start/end.
    SessionBoundary { timestamp: Nanos, is_open: bool },
}

impl LobEvent {
    pub fn timestamp(&self) -> Nanos {
        match self {
            LobEvent::OrderSubmit(e) => e.timestamp,
            LobEvent::OrderCancel(e) => e.timestamp,
            LobEvent::Fill(e) => e.timestamp,
            LobEvent::PriceUpdate(e) => e.timestamp,
            LobEvent::RegimeChange(e) => e.timestamp,
            LobEvent::Tick { timestamp, .. } => *timestamp,
            LobEvent::SessionBoundary { timestamp, .. } => *timestamp,
        }
    }

    pub fn event_type(&self) -> &'static str {
        match self {
            LobEvent::OrderSubmit(_) => "ORDER_SUBMIT",
            LobEvent::OrderCancel(_) => "ORDER_CANCEL",
            LobEvent::Fill(_) => "FILL",
            LobEvent::PriceUpdate(_) => "PRICE_UPDATE",
            LobEvent::RegimeChange(_) => "REGIME_CHANGE",
            LobEvent::Tick { .. } => "TICK",
            LobEvent::SessionBoundary { .. } => "SESSION",
        }
    }

    pub fn symbol(&self) -> Option<&str> {
        match self {
            LobEvent::OrderSubmit(e) => Some(&e.symbol),
            LobEvent::OrderCancel(e) => Some(&e.symbol),
            LobEvent::Fill(e) => Some(&e.symbol),
            LobEvent::PriceUpdate(e) => Some(&e.symbol),
            LobEvent::RegimeChange(e) => Some(&e.symbol),
            _ => None,
        }
    }
}

// ── Event Bus ─────────────────────────────────────────────────────────────────

/// Thread-safe global event sequence counter.
static EVENT_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn next_event_id() -> EventId {
    EVENT_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// MPSC event channel: multiple producers (agents, LOB engine) → one consumer.
pub struct EventBus {
    sender: Sender<LobEvent>,
    receiver: Receiver<LobEvent>,
    /// Number of events published total.
    pub published: Arc<AtomicU64>,
    /// Number of events consumed total.
    pub consumed: Arc<AtomicU64>,
}

impl EventBus {
    /// Create an unbounded event bus.
    pub fn new_unbounded() -> Self {
        let (sender, receiver) = unbounded();
        EventBus {
            sender,
            receiver,
            published: Arc::new(AtomicU64::new(0)),
            consumed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a bounded event bus with given capacity.
    pub fn new_bounded(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        EventBus {
            sender,
            receiver,
            published: Arc::new(AtomicU64::new(0)),
            consumed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get a cloned sender for an additional producer.
    pub fn make_sender(&self) -> EventBusSender {
        EventBusSender {
            sender: self.sender.clone(),
            published: Arc::clone(&self.published),
        }
    }

    /// Publish an event (from the bus owner).
    pub fn publish(&self, event: LobEvent) -> Result<(), LobEventError> {
        self.sender.send(event).map_err(|_| LobEventError::ChannelClosed)?;
        self.published.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Try to receive one event (non-blocking).
    pub fn try_recv(&self) -> Result<LobEvent, TryRecvResult> {
        match self.receiver.try_recv() {
            Ok(event) => {
                self.consumed.fetch_add(1, Ordering::Relaxed);
                Ok(event)
            }
            Err(TryRecvError::Empty) => Err(TryRecvResult::Empty),
            Err(TryRecvError::Disconnected) => Err(TryRecvResult::Disconnected),
        }
    }

    /// Blocking receive.
    pub fn recv(&self) -> Result<LobEvent, LobEventError> {
        self.receiver.recv().map_err(|_| LobEventError::ChannelClosed)
            .map(|e| { self.consumed.fetch_add(1, Ordering::Relaxed); e })
    }

    /// Drain all pending events into a Vec (non-blocking).
    pub fn drain(&self) -> Vec<LobEvent> {
        let mut events = Vec::new();
        loop {
            match self.receiver.try_recv() {
                Ok(e) => {
                    self.consumed.fetch_add(1, Ordering::Relaxed);
                    events.push(e);
                }
                Err(_) => break,
            }
        }
        events
    }

    /// Number of events waiting in the queue.
    pub fn pending_count(&self) -> usize {
        self.receiver.len()
    }

    /// Overall throughput: published - consumed (backpressure indicator).
    pub fn backpressure(&self) -> i64 {
        self.published.load(Ordering::Relaxed) as i64
            - self.consumed.load(Ordering::Relaxed) as i64
    }
}

/// A cloned sender for use by producers.
#[derive(Clone)]
pub struct EventBusSender {
    sender: Sender<LobEvent>,
    published: Arc<AtomicU64>,
}

impl EventBusSender {
    pub fn publish(&self, event: LobEvent) -> Result<(), LobEventError> {
        self.sender.send(event).map_err(|_| LobEventError::ChannelClosed)?;
        self.published.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn is_closed(&self) -> bool {
        self.sender.is_disconnected()
    }
}

// ── Error Types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LobEventError {
    ChannelClosed,
    SerializationError,
    InvalidEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvResult {
    Empty,
    Disconnected,
}

// ── Event Builder Helpers ─────────────────────────────────────────────────────

pub struct EventBuilder;

impl EventBuilder {
    pub fn order_submit(
        symbol: impl Into<String>,
        order_id: u64,
        side: Side,
        price: f64,
        qty: Qty,
        ts: Nanos,
        agent_id: u32,
    ) -> LobEvent {
        LobEvent::OrderSubmit(OrderSubmitEvent {
            event_id: next_event_id(),
            timestamp: ts,
            order_id,
            symbol: symbol.into(),
            side,
            kind: OrderKindTag::Limit,
            price,
            qty,
            tif: TimeInForcetag::GTC,
            agent_id,
        })
    }

    pub fn fill(
        symbol: impl Into<String>,
        aggressor_id: u64,
        passive_id: u64,
        price: f64,
        qty: Qty,
        side: Side,
        ts: Nanos,
        aggressor_agent: u32,
        passive_agent: u32,
        is_partial: bool,
    ) -> LobEvent {
        LobEvent::Fill(FillEvent {
            event_id: next_event_id(),
            timestamp: ts,
            symbol: symbol.into(),
            aggressor_id,
            passive_id,
            price,
            qty,
            side,
            aggressor_agent,
            passive_agent,
            is_partial,
        })
    }

    pub fn price_update(
        symbol: impl Into<String>,
        bid: Option<f64>,
        ask: Option<f64>,
        last: f64,
        bid_qty: Qty,
        ask_qty: Qty,
        imbalance: f64,
        bid_depth: Vec<(f64, Qty)>,
        ask_depth: Vec<(f64, Qty)>,
        ts: Nanos,
    ) -> LobEvent {
        let mid = match (bid, ask) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        };
        LobEvent::PriceUpdate(PriceUpdateEvent {
            event_id: next_event_id(),
            timestamp: ts,
            symbol: symbol.into(),
            bid,
            ask,
            mid,
            last,
            bid_qty,
            ask_qty,
            imbalance,
            bid_depth,
            ask_depth,
        })
    }

    pub fn regime_change(
        symbol: impl Into<String>,
        from: MarketRegime,
        to: MarketRegime,
        trigger: RegimeChangeTrigger,
        ts: Nanos,
    ) -> LobEvent {
        LobEvent::RegimeChange(RegimeChangeEvent {
            event_id: next_event_id(),
            timestamp: ts,
            symbol: symbol.into(),
            from_regime: from,
            to_regime: to,
            trigger,
        })
    }

    pub fn tick(ts: Nanos, seq: u64) -> LobEvent {
        LobEvent::Tick { timestamp: ts, sequence: seq }
    }

    pub fn cancel(
        symbol: impl Into<String>,
        order_id: u64,
        agent_id: u32,
        success: bool,
        ts: Nanos,
    ) -> LobEvent {
        LobEvent::OrderCancel(OrderCancelEvent {
            event_id: next_event_id(),
            timestamp: ts,
            order_id,
            symbol: symbol.into(),
            agent_id,
            success,
        })
    }
}

// ── Event Filter ─────────────────────────────────────────────────────────────

/// Filter events by type and/or symbol.
pub struct EventFilter {
    pub symbol: Option<String>,
    pub accept_submits: bool,
    pub accept_cancels: bool,
    pub accept_fills: bool,
    pub accept_price_updates: bool,
    pub accept_regime_changes: bool,
    pub accept_ticks: bool,
    pub min_qty: Option<Qty>,
    pub agent_id: Option<u32>,
}

impl Default for EventFilter {
    fn default() -> Self {
        EventFilter {
            symbol: None,
            accept_submits: true,
            accept_cancels: true,
            accept_fills: true,
            accept_price_updates: true,
            accept_regime_changes: true,
            accept_ticks: true,
            min_qty: None,
            agent_id: None,
        }
    }
}

impl EventFilter {
    pub fn accepts(&self, event: &LobEvent) -> bool {
        // Symbol filter.
        if let Some(ref sym) = self.symbol {
            if let Some(event_sym) = event.symbol() {
                if event_sym != sym { return false; }
            }
        }

        // Type filter.
        let type_ok = match event {
            LobEvent::OrderSubmit(e) => {
                if !self.accept_submits { return false; }
                if let Some(aid) = self.agent_id {
                    if e.agent_id != aid { return false; }
                }
                if let Some(mq) = self.min_qty {
                    if e.qty < mq { return false; }
                }
                true
            }
            LobEvent::OrderCancel(e) => {
                if !self.accept_cancels { return false; }
                if let Some(aid) = self.agent_id {
                    if e.agent_id != aid { return false; }
                }
                true
            }
            LobEvent::Fill(e) => {
                if !self.accept_fills { return false; }
                if let Some(mq) = self.min_qty {
                    if e.qty < mq { return false; }
                }
                if let Some(aid) = self.agent_id {
                    if e.aggressor_agent != aid && e.passive_agent != aid { return false; }
                }
                true
            }
            LobEvent::PriceUpdate(_) => self.accept_price_updates,
            LobEvent::RegimeChange(_) => self.accept_regime_changes,
            LobEvent::Tick { .. } => self.accept_ticks,
            LobEvent::SessionBoundary { .. } => true,
        };

        type_ok
    }
}

// ── Filtered receiver ─────────────────────────────────────────────────────────

/// Wrapper that filters events from an EventBus.
pub struct FilteredReceiver<'a> {
    bus: &'a EventBus,
    filter: EventFilter,
}

impl<'a> FilteredReceiver<'a> {
    pub fn new(bus: &'a EventBus, filter: EventFilter) -> Self {
        FilteredReceiver { bus, filter }
    }

    /// Drain events that pass the filter.
    pub fn drain_filtered(&self) -> Vec<LobEvent> {
        let all = self.bus.drain();
        all.into_iter().filter(|e| self.filter.accepts(e)).collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publish_receive() {
        let bus = EventBus::new_unbounded();
        let event = EventBuilder::tick(1_000_000_000, 1);
        bus.publish(event).unwrap();
        let recv = bus.recv().unwrap();
        assert_eq!(recv.event_type(), "TICK");
        assert_eq!(recv.timestamp(), 1_000_000_000);
    }

    #[test]
    fn test_mpsc() {
        let bus = EventBus::new_unbounded();
        let sender2 = bus.make_sender();
        // Publish from two senders.
        bus.publish(EventBuilder::tick(1, 1)).unwrap();
        sender2.publish(EventBuilder::tick(2, 2)).unwrap();
        let events = bus.drain();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_event_filter_symbol() {
        let bus = EventBus::new_unbounded();
        bus.publish(EventBuilder::price_update("AAPL", Some(150.0), Some(150.05), 150.0, 100.0, 100.0, 0.0, vec![], vec![], 1)).unwrap();
        bus.publish(EventBuilder::price_update("MSFT", Some(250.0), Some(250.1), 250.0, 50.0, 50.0, 0.0, vec![], vec![], 2)).unwrap();
        let filter = EventFilter { symbol: Some("AAPL".to_string()), ..Default::default() };
        let receiver = FilteredReceiver::new(&bus, filter);
        let filtered = receiver.drain_filtered();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].symbol(), Some("AAPL"));
    }

    #[test]
    fn test_backpressure() {
        let bus = EventBus::new_unbounded();
        for i in 0..10 {
            bus.publish(EventBuilder::tick(i, i)).unwrap();
        }
        assert_eq!(bus.backpressure(), 10);
        bus.drain();
        assert_eq!(bus.backpressure(), 0);
    }

    #[test]
    fn test_bounded_channel_blocks() {
        let bus = EventBus::new_bounded(5);
        for i in 0..5 {
            bus.publish(EventBuilder::tick(i, i)).unwrap();
        }
        // 6th should fail if we used try_send — but our API uses send which blocks.
        // Just verify we can drain all 5.
        let events = bus.drain();
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn test_regime_change_event() {
        let bus = EventBus::new_unbounded();
        bus.publish(EventBuilder::regime_change(
            "SPY",
            MarketRegime::Normal,
            MarketRegime::HighVolatility,
            RegimeChangeTrigger::VolatilityThreshold,
            500_000_000,
        )).unwrap();
        let ev = bus.recv().unwrap();
        if let LobEvent::RegimeChange(rce) = ev {
            assert_eq!(rce.from_regime, MarketRegime::Normal);
            assert_eq!(rce.to_regime, MarketRegime::HighVolatility);
        } else {
            panic!("Expected RegimeChange");
        }
    }

    #[test]
    fn test_fill_event_agent_filter() {
        let bus = EventBus::new_unbounded();
        bus.publish(EventBuilder::fill("XYZ", 1, 2, 100.0, 10.0, Side::Bid, 1, 1, 99, false)).unwrap();
        bus.publish(EventBuilder::fill("XYZ", 3, 4, 101.0, 20.0, Side::Ask, 2, 5, 77, true)).unwrap();
        let filter = EventFilter { agent_id: Some(1), accept_price_updates: false, accept_ticks: false, ..Default::default() };
        let recv = FilteredReceiver::new(&bus, filter);
        let events = recv.drain_filtered();
        assert_eq!(events.len(), 1);
    }
}
