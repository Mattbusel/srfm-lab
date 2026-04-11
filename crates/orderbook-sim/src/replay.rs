/// Event replay engine for LOB backtesting.
///
/// Replays a sequence of LOB events at configurable speed with:
/// - Real-time speed control (time_multiplier)
/// - Pause/resume/seek capabilities
/// - Event filtering during replay
/// - Callback hooks for strategy injection
/// - Progress tracking and statistics

use std::time::{Duration, Instant};
use std::thread;

use crate::event_stream::{LobEvent, EventBus, EventBuilder, EventFilter, MarketRegime};
use crate::lob_engine::{LobEngine, LobOrder, LobFill, Side, Qty, Nanos};

// ── Replay Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// How fast to replay: 1.0 = real-time, 10.0 = 10× speed, 0.0 = as-fast-as-possible.
    pub time_multiplier: f64,
    /// Start time offset (nanoseconds) within the event stream.
    pub start_offset_ns: u64,
    /// Optional end time (nanoseconds). None = replay all.
    pub end_time_ns: Option<u64>,
    /// Whether to publish replayed events to a bus.
    pub publish_events: bool,
    /// Whether to re-simulate matching engine (true) or just publish events (false).
    pub simulate_matching: bool,
    /// Maximum events to replay (None = unlimited).
    pub max_events: Option<usize>,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        ReplayConfig {
            time_multiplier: 0.0,  // as fast as possible
            start_offset_ns: 0,
            end_time_ns: None,
            publish_events: true,
            simulate_matching: true,
            max_events: None,
        }
    }
}

// ── Replay State ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayState {
    Ready,
    Running,
    Paused,
    Completed,
    Error,
}

// ── Replay Statistics ─────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct ReplayStats {
    pub events_processed: u64,
    pub fills_generated: u64,
    pub orders_submitted: u64,
    pub orders_cancelled: u64,
    pub wall_time_elapsed_ms: u64,
    pub sim_time_elapsed_ns: u64,
    pub replay_rate: f64,   // events per second wall time
    pub first_event_ts: u64,
    pub last_event_ts: u64,
}

impl ReplayStats {
    pub fn update_rate(&mut self, wall_elapsed_ms: u64) {
        if wall_elapsed_ms > 0 {
            self.replay_rate = self.events_processed as f64 / (wall_elapsed_ms as f64 / 1000.0);
        }
    }
}

// ── Replay Engine ─────────────────────────────────────────────────────────────

/// Full event replay engine.
pub struct ReplayEngine {
    pub config: ReplayConfig,
    /// Ordered list of events to replay.
    events: Vec<LobEvent>,
    /// Current position in the event stream.
    cursor: usize,
    /// Replay state.
    pub state: ReplayState,
    /// Statistics.
    pub stats: ReplayStats,
    /// Optional LOB engine for re-simulation.
    lob: Option<LobEngine>,
    /// Wall-clock start time of replay.
    wall_start: Option<Instant>,
    /// First event nanosecond timestamp.
    event_origin_ns: u64,
}

impl ReplayEngine {
    pub fn new(events: Vec<LobEvent>, config: ReplayConfig) -> Self {
        let origin = events.first().map(|e| e.timestamp()).unwrap_or(0);
        ReplayEngine {
            config,
            events,
            cursor: 0,
            state: ReplayState::Ready,
            stats: ReplayStats::default(),
            lob: None,
            wall_start: None,
            event_origin_ns: origin,
        }
    }

    pub fn with_lob(mut self, lob: LobEngine) -> Self {
        self.lob = Some(lob);
        self
    }

    pub fn total_events(&self) -> usize {
        self.events.len()
    }

    pub fn progress_pct(&self) -> f64 {
        if self.events.is_empty() { return 100.0; }
        100.0 * self.cursor as f64 / self.events.len() as f64
    }

    /// Start replay, calling the callback for each event.
    /// `callback(event, lob_opt) -> bool` — return false to stop replay.
    pub fn run<F>(&mut self, mut callback: F) -> ReplayStats
    where
        F: FnMut(&LobEvent, Option<&mut LobEngine>) -> bool,
    {
        self.state = ReplayState::Running;
        self.wall_start = Some(Instant::now());
        let start_wall = self.wall_start.unwrap();

        // Skip to start_offset.
        while self.cursor < self.events.len()
            && self.events[self.cursor].timestamp() < self.event_origin_ns + self.config.start_offset_ns
        {
            self.cursor += 1;
        }

        let mut events_this_session: u64 = 0;

        loop {
            if self.state == ReplayState::Paused {
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            if self.cursor >= self.events.len() {
                self.state = ReplayState::Completed;
                break;
            }

            if let Some(max) = self.config.max_events {
                if events_this_session >= max as u64 {
                    self.state = ReplayState::Completed;
                    break;
                }
            }

            let event = &self.events[self.cursor];
            let event_ts = event.timestamp();

            // End time check.
            if let Some(end) = self.config.end_time_ns {
                if event_ts > end {
                    self.state = ReplayState::Completed;
                    break;
                }
            }

            // Time-based throttling for real-time replay.
            if self.config.time_multiplier > 0.0 {
                let elapsed_wall_ns = start_wall.elapsed().as_nanos() as u64;
                let sim_elapsed_ns = event_ts.saturating_sub(self.event_origin_ns + self.config.start_offset_ns);
                let target_wall_ns = (sim_elapsed_ns as f64 / self.config.time_multiplier) as u64;

                if target_wall_ns > elapsed_wall_ns {
                    let sleep_ns = target_wall_ns - elapsed_wall_ns;
                    if sleep_ns > 100_000 {  // Only sleep if > 100 µs
                        thread::sleep(Duration::from_nanos(sleep_ns.min(10_000_000)));
                    }
                    continue;
                }
            }

            // Process the event.
            let lob_ref = self.lob.as_mut();
            let should_continue = callback(event, lob_ref);

            // Update stats.
            match event {
                LobEvent::Fill(_) => self.stats.fills_generated += 1,
                LobEvent::OrderSubmit(_) => self.stats.orders_submitted += 1,
                LobEvent::OrderCancel(_) => self.stats.orders_cancelled += 1,
                _ => {}
            }

            if self.stats.first_event_ts == 0 {
                self.stats.first_event_ts = event_ts;
            }
            self.stats.last_event_ts = event_ts;
            self.stats.events_processed += 1;
            self.stats.sim_time_elapsed_ns = event_ts.saturating_sub(self.stats.first_event_ts);
            events_this_session += 1;
            self.cursor += 1;

            if !should_continue {
                self.state = ReplayState::Completed;
                break;
            }
        }

        let wall_ms = start_wall.elapsed().as_millis() as u64;
        self.stats.wall_time_elapsed_ms = wall_ms;
        self.stats.update_rate(wall_ms);
        self.stats.clone()
    }

    /// Pause replay (only meaningful in threaded context).
    pub fn pause(&mut self) {
        if self.state == ReplayState::Running {
            self.state = ReplayState::Paused;
        }
    }

    pub fn resume(&mut self) {
        if self.state == ReplayState::Paused {
            self.state = ReplayState::Running;
        }
    }

    /// Seek to a specific timestamp. Returns true if found.
    pub fn seek(&mut self, target_ts: u64) -> bool {
        if let Some(pos) = self.events.iter().position(|e| e.timestamp() >= target_ts) {
            self.cursor = pos;
            true
        } else {
            false
        }
    }

    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.state = ReplayState::Ready;
        self.stats = ReplayStats::default();
        self.wall_start = None;
    }

    /// Get a window of events around the current cursor (for display).
    pub fn peek_window(&self, n: usize) -> &[LobEvent] {
        let start = self.cursor;
        let end = (start + n).min(self.events.len());
        &self.events[start..end]
    }
}

// ── Recorded Session ──────────────────────────────────────────────────────────

/// A complete recorded trading session that can be replayed.
#[derive(Default)]
pub struct RecordedSession {
    pub symbol: String,
    pub start_ts: u64,
    pub end_ts: u64,
    events: Vec<LobEvent>,
    /// Total fills recorded.
    pub fill_count: u64,
    /// Total orders submitted.
    pub order_count: u64,
}

impl RecordedSession {
    pub fn new(symbol: impl Into<String>) -> Self {
        RecordedSession { symbol: symbol.into(), ..Default::default() }
    }

    pub fn record(&mut self, event: LobEvent) {
        let ts = event.timestamp();
        if self.start_ts == 0 { self.start_ts = ts; }
        self.end_ts = ts;
        match &event {
            LobEvent::Fill(_) => self.fill_count += 1,
            LobEvent::OrderSubmit(_) => self.order_count += 1,
            _ => {}
        }
        self.events.push(event);
    }

    pub fn event_count(&self) -> usize { self.events.len() }

    pub fn duration_ns(&self) -> u64 {
        self.end_ts.saturating_sub(self.start_ts)
    }

    pub fn duration_secs(&self) -> f64 {
        self.duration_ns() as f64 / 1e9
    }

    /// Build a replay engine for this session.
    pub fn make_replay(&self, config: ReplayConfig) -> ReplayEngine {
        ReplayEngine::new(self.events.clone(), config)
    }

    /// Build a replay engine with a fresh LOB for re-simulation.
    pub fn make_simulation_replay(&self, config: ReplayConfig) -> ReplayEngine {
        let lob = LobEngine::new(&self.symbol);
        ReplayEngine::new(self.events.clone(), config).with_lob(lob)
    }

    /// Sort events by timestamp (for sessions assembled out of order).
    pub fn sort_events(&mut self) {
        self.events.sort_by_key(|e| e.timestamp());
    }

    /// Filter to only events matching the predicate.
    pub fn filter_events<F>(&self, predicate: F) -> RecordedSession
    where
        F: Fn(&LobEvent) -> bool,
    {
        let filtered: Vec<LobEvent> = self.events.iter()
            .filter(|e| predicate(e))
            .cloned()
            .collect();
        let mut session = RecordedSession::new(&self.symbol);
        for e in filtered {
            session.record(e);
        }
        session
    }
}

// ── Backtesting Loop ──────────────────────────────────────────────────────────

/// Result of running a strategy over a replayed session.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub symbol: String,
    pub total_fills: u64,
    pub total_pnl: f64,
    pub total_volume: f64,
    pub avg_fill_price: f64,
    pub replay_stats: ReplayStats,
    pub mid_prices: Vec<f64>,
    pub fill_prices: Vec<f64>,
    pub timestamps: Vec<u64>,
}

/// Simple backtesting wrapper: replays session and feeds price updates to a strategy.
pub fn run_backtest<S>(
    session: &RecordedSession,
    mut strategy: S,
    config: ReplayConfig,
) -> BacktestResult
where
    S: FnMut(f64, f64, f64, u64) -> Option<(Side, f64, Qty)>,
    // (mid, spread, imbalance, ts_ns) -> Option<(side, price, qty)>
{
    let symbol = session.symbol.clone();
    let mut replay = session.make_simulation_replay(config);

    let mut total_fills = 0u64;
    let mut total_pnl = 0.0_f64;
    let mut total_volume = 0.0_f64;
    let mut fill_notional = 0.0_f64;
    let mut mid_prices = Vec::new();
    let mut fill_prices = Vec::new();
    let mut timestamps = Vec::new();
    let mut order_id_counter = 1_000_000_000u64;

    let stats = replay.run(|event, lob| {
        match event {
            LobEvent::PriceUpdate(pu) => {
                let mid = pu.mid.unwrap_or(pu.last);
                let spread = match (pu.bid, pu.ask) {
                    (Some(b), Some(a)) => a - b,
                    _ => 0.0,
                };
                let imbalance = pu.imbalance;
                let ts = pu.timestamp;

                mid_prices.push(mid);
                timestamps.push(ts);

                if let Some((side, price, qty)) = strategy(mid, spread, imbalance, ts) {
                    if let Some(lob_engine) = lob {
                        let order = LobOrder::new_limit(order_id_counter, side, price, qty, ts, 1);
                        order_id_counter += 1;
                        let fills = lob_engine.submit(order);
                        for fill in &fills {
                            total_fills += 1;
                            total_volume += fill.qty;
                            fill_notional += fill.price * fill.qty;
                            fill_prices.push(fill.price);
                            let signed = match side {
                                Side::Bid => -fill.price * fill.qty,
                                Side::Ask => fill.price * fill.qty,
                            };
                            total_pnl += signed;
                        }
                    }
                }
            }
            _ => {}
        }
        true
    });

    let avg_fill = if total_volume > 1e-9 { fill_notional / total_volume } else { 0.0 };

    BacktestResult {
        symbol,
        total_fills,
        total_pnl,
        total_volume,
        avg_fill_price: avg_fill,
        replay_stats: stats,
        mid_prices,
        fill_prices,
        timestamps,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_stream::EventBuilder;

    fn make_price_event(mid: f64, ts: u64) -> LobEvent {
        EventBuilder::price_update(
            "TEST",
            Some(mid - 0.01),
            Some(mid + 0.01),
            mid,
            100.0,
            100.0,
            0.0,
            vec![],
            vec![],
            ts,
        )
    }

    fn make_fill_event(price: f64, qty: f64, ts: u64) -> LobEvent {
        EventBuilder::fill("TEST", 1, 2, price, qty, Side::Bid, ts, 1, 99, false)
    }

    #[test]
    fn test_recorded_session_record() {
        let mut session = RecordedSession::new("TEST");
        for i in 0..10 {
            session.record(make_price_event(100.0 + i as f64 * 0.1, i as u64 * 1_000_000));
        }
        session.record(make_fill_event(100.5, 50.0, 5_000_000));
        assert_eq!(session.event_count(), 11);
        assert_eq!(session.fill_count, 1);
        assert_eq!(session.order_count, 0);
    }

    #[test]
    fn test_replay_all_events() {
        let mut session = RecordedSession::new("TEST");
        for i in 0..20 {
            session.record(make_price_event(100.0 + i as f64 * 0.01, i as u64 * 1_000_000));
        }

        let config = ReplayConfig { time_multiplier: 0.0, ..Default::default() };
        let mut replay = session.make_replay(config);
        let mut count = 0usize;
        let stats = replay.run(|_event, _lob| { count += 1; true });
        assert_eq!(count, 20);
        assert_eq!(stats.events_processed, 20);
    }

    #[test]
    fn test_replay_max_events() {
        let mut session = RecordedSession::new("TEST");
        for i in 0..50 {
            session.record(make_price_event(100.0, i as u64 * 1_000_000));
        }
        let config = ReplayConfig { time_multiplier: 0.0, max_events: Some(10), ..Default::default() };
        let mut replay = session.make_replay(config);
        let stats = replay.run(|_, _| true);
        assert_eq!(stats.events_processed, 10);
        assert_eq!(replay.state, ReplayState::Completed);
    }

    #[test]
    fn test_replay_seek() {
        let mut session = RecordedSession::new("TEST");
        for i in 0..100 {
            session.record(make_price_event(100.0, i as u64 * 1_000_000_000)); // 1-second intervals
        }
        let config = ReplayConfig::default();
        let mut replay = session.make_replay(config);
        let found = replay.seek(50_000_000_000); // seek to t=50s
        assert!(found);
        assert_eq!(replay.cursor, 50);
    }

    #[test]
    fn test_replay_with_end_time() {
        let mut session = RecordedSession::new("TEST");
        for i in 0..100 {
            session.record(make_price_event(100.0, i as u64 * 1_000_000_000));
        }
        let config = ReplayConfig {
            time_multiplier: 0.0,
            end_time_ns: Some(10_000_000_000), // stop at t=10s
            ..Default::default()
        };
        let mut replay = session.make_replay(config);
        let stats = replay.run(|_, _| true);
        assert!(stats.events_processed <= 11); // 0..=10 seconds
    }

    #[test]
    fn test_backtest_passthrough() {
        let mut session = RecordedSession::new("TEST");
        // Seed price events.
        for i in 0..50 {
            session.record(make_price_event(100.0 + (i as f64 * 0.001), i as u64 * 1_000_000_000));
        }
        let config = ReplayConfig::default();
        // Strategy that never trades.
        let result = run_backtest(&session, |_, _, _, _| None, config);
        assert_eq!(result.total_fills, 0);
        assert!(!result.mid_prices.is_empty());
    }

    #[test]
    fn test_filter_events() {
        let mut session = RecordedSession::new("TEST");
        for i in 0..5 {
            session.record(make_price_event(100.0, i as u64 * 1_000_000));
        }
        for i in 0..3 {
            session.record(make_fill_event(100.0, 10.0, (5 + i) as u64 * 1_000_000));
        }
        let fills_only = session.filter_events(|e| matches!(e, LobEvent::Fill(_)));
        assert_eq!(fills_only.fill_count, 3);
        assert_eq!(fills_only.event_count(), 3);
    }
}
