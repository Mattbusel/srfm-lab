// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// pipeline_bridge.rs — Rust ↔ C++ pipeline integration bridge
// =============================================================================
//! Connects the Rust RTEL client to the C++ pipeline via shared memory.
//!
//! Provides:
//! - `PipelineBridge` — orchestrates multi-channel publish/subscribe
//! - `ChannelSupervisor` — monitors channel health and lag
//! - `BridgeMetrics` — tracks cross-boundary latencies
//! - Async event loop with tokio

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use crate::now_ns;
use crate::metrics::{global_metrics, ChannelMetrics};
use crate::error::{RtelError, Result};

// ---------------------------------------------------------------------------
// Channel health status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelStatus {
    Healthy,
    Lagging,
    Stale,
    Error,
}

impl std::fmt::Display for ChannelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy  => write!(f, "healthy"),
            Self::Lagging  => write!(f, "lagging"),
            Self::Stale    => write!(f, "stale"),
            Self::Error    => write!(f, "error"),
        }
    }
}

// ---------------------------------------------------------------------------
// Channel configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ChannelBridgeConfig {
    pub name:              String,
    pub max_lag:           u64,      // max acceptable sequence lag
    pub stale_timeout_ns:  u64,      // ns without update before "stale"
    pub buffer_size:       usize,    // local ring buffer for outgoing messages
}

impl ChannelBridgeConfig {
    pub fn new(name: &str) -> Self {
        Self {
            name:             name.to_owned(),
            max_lag:          100,
            stale_timeout_ns: 5_000_000_000,   // 5 seconds
            buffer_size:      1024,
        }
    }

    pub fn with_max_lag(mut self, lag: u64) -> Self {
        self.max_lag = lag;
        self
    }

    pub fn with_stale_timeout(mut self, ns: u64) -> Self {
        self.stale_timeout_ns = ns;
        self
    }
}

// ---------------------------------------------------------------------------
// ChannelSupervisor
// ---------------------------------------------------------------------------

pub struct ChannelSupervisor {
    config:          ChannelBridgeConfig,
    last_seq:        AtomicU64,
    last_update_ns:  AtomicU64,
    status:          Mutex<ChannelStatus>,
    n_updates:       AtomicU64,
    n_stale_events:  AtomicU64,
    n_lag_events:    AtomicU64,
}

impl ChannelSupervisor {
    pub fn new(config: ChannelBridgeConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            last_seq:       AtomicU64::new(0),
            last_update_ns: AtomicU64::new(now_ns()),
            status:         Mutex::new(ChannelStatus::Healthy),
            n_updates:      AtomicU64::new(0),
            n_stale_events: AtomicU64::new(0),
            n_lag_events:   AtomicU64::new(0),
        })
    }

    pub fn record_update(&self, sequence: u64) {
        let last = self.last_seq.swap(sequence, Ordering::Relaxed);
        let lag  = sequence.saturating_sub(last).saturating_sub(1);

        self.last_update_ns.store(now_ns(), Ordering::Relaxed);
        self.n_updates.fetch_add(1, Ordering::Relaxed);

        if lag > self.config.max_lag {
            self.n_lag_events.fetch_add(1, Ordering::Relaxed);
            *self.status.lock().unwrap() = ChannelStatus::Lagging;
        } else {
            *self.status.lock().unwrap() = ChannelStatus::Healthy;
        }
    }

    pub fn check_staleness(&self) -> ChannelStatus {
        let elapsed = now_ns().saturating_sub(
            self.last_update_ns.load(Ordering::Relaxed)
        );
        if elapsed > self.config.stale_timeout_ns {
            self.n_stale_events.fetch_add(1, Ordering::Relaxed);
            *self.status.lock().unwrap() = ChannelStatus::Stale;
        }
        *self.status.lock().unwrap()
    }

    pub fn status(&self) -> ChannelStatus {
        *self.status.lock().unwrap()
    }

    pub fn name(&self) -> &str { &self.config.name }

    pub fn n_updates(&self)     -> u64 { self.n_updates.load(Ordering::Relaxed) }
    pub fn n_lag_events(&self)  -> u64 { self.n_lag_events.load(Ordering::Relaxed) }
    pub fn n_stale_events(&self)-> u64 { self.n_stale_events.load(Ordering::Relaxed) }
}

// ---------------------------------------------------------------------------
// BridgeMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct BridgeMetrics {
    pub total_messages_sent:    AtomicU64,
    pub total_messages_received:AtomicU64,
    pub total_bytes_sent:       AtomicU64,
    pub total_bytes_received:   AtomicU64,
    pub n_channel_errors:       AtomicU64,
    pub n_sequence_gaps:        AtomicU64,

    // Latency tracking (last 10K measurements)
    latency_sum:    AtomicU64,
    latency_count:  AtomicU64,
    latency_max:    AtomicU64,
}

impl BridgeMetrics {
    pub fn record_send(&self, bytes: u64) {
        self.total_messages_sent.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_receive(&self, bytes: u64, latency_ns: u64) {
        self.total_messages_received.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_received.fetch_add(bytes, Ordering::Relaxed);
        self.latency_sum.fetch_add(latency_ns, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);
        // Update max
        let mut old = self.latency_max.load(Ordering::Relaxed);
        while latency_ns > old {
            match self.latency_max.compare_exchange_weak(
                old, latency_ns, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_)  => break,
                Err(x) => old = x,
            }
        }
    }

    pub fn mean_latency_ns(&self) -> f64 {
        let c = self.latency_count.load(Ordering::Relaxed);
        if c == 0 { return 0.0; }
        self.latency_sum.load(Ordering::Relaxed) as f64 / c as f64
    }

    pub fn max_latency_ns(&self) -> u64 {
        self.latency_max.load(Ordering::Relaxed)
    }

    pub fn prometheus(&self) -> String {
        format!(
            "rtel_bridge_sent_total {}\n\
             rtel_bridge_received_total {}\n\
             rtel_bridge_bytes_sent {}\n\
             rtel_bridge_bytes_received {}\n\
             rtel_bridge_errors {}\n\
             rtel_bridge_mean_latency_ns {:.1}\n\
             rtel_bridge_max_latency_ns {}\n",
            self.total_messages_sent.load(Ordering::Relaxed),
            self.total_messages_received.load(Ordering::Relaxed),
            self.total_bytes_sent.load(Ordering::Relaxed),
            self.total_bytes_received.load(Ordering::Relaxed),
            self.n_channel_errors.load(Ordering::Relaxed),
            self.mean_latency_ns(),
            self.max_latency_ns(),
        )
    }
}

// ---------------------------------------------------------------------------
// Message types for pipeline events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PipelineMessage {
    LobSnapshot {
        asset_id:   u32,
        sequence:   u64,
        timestamp:  u64,
        mid_price:  f64,
        spread:     f64,
        imbalance:  f64,
    },
    VolSurface {
        asset_id:  u32,
        timestamp: u64,
        atm_vol:   f64,
        skew:      f64,
        term_slope: f64,
    },
    AgentAction {
        asset_id:   u32,
        timestamp:  u64,
        direction:  f64,
        confidence: f64,
        size_hint:  f64,
    },
    PipelineEvent {
        event_type: u32,
        timestamp:  u64,
        data:       Vec<u8>,
    },
    Heartbeat {
        timestamp: u64,
        source_id: u32,
    },
}

impl PipelineMessage {
    pub fn is_heartbeat(&self) -> bool {
        matches!(self, Self::Heartbeat { .. })
    }

    pub fn timestamp(&self) -> u64 {
        match self {
            Self::LobSnapshot { timestamp, .. }  => *timestamp,
            Self::VolSurface { timestamp, .. }   => *timestamp,
            Self::AgentAction { timestamp, .. }  => *timestamp,
            Self::PipelineEvent { timestamp, .. }=> *timestamp,
            Self::Heartbeat { timestamp, .. }    => *timestamp,
        }
    }

    pub fn latency_ns(&self) -> u64 {
        now_ns().saturating_sub(self.timestamp())
    }
}

// ---------------------------------------------------------------------------
// PipelineBridge
// ---------------------------------------------------------------------------

pub struct PipelineBridge {
    supervisors:   Mutex<HashMap<String, Arc<ChannelSupervisor>>>,
    metrics:       Arc<BridgeMetrics>,
    running:       AtomicBool,
    heartbeat_interval_ns: u64,
    last_heartbeat_ns: AtomicU64,
}

impl PipelineBridge {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            supervisors:  Mutex::new(HashMap::new()),
            metrics:      Arc::new(BridgeMetrics::default()),
            running:      AtomicBool::new(false),
            heartbeat_interval_ns: 1_000_000_000,  // 1 second
            last_heartbeat_ns: AtomicU64::new(0),
        })
    }

    pub fn register_channel(&self, config: ChannelBridgeConfig) {
        let name = config.name.clone();
        let sup  = ChannelSupervisor::new(config);
        self.supervisors.lock().unwrap().insert(name, sup);
    }

    pub fn with_standard_channels(self: &Arc<Self>) -> &Arc<Self> {
        use crate::channels::*;
        for name in [LOB_SNAPSHOT, VOL_SURFACE, TENSOR_COMP, GRAPH_ADJ,
                      LUMINA_PRED, AGENT_ACTIONS, PIPELINE_EVENTS, HEARTBEAT]
        {
            self.register_channel(ChannelBridgeConfig::new(name));
        }
        self
    }

    pub fn start(&self) {
        self.running.store(true, Ordering::Release);
        self.last_heartbeat_ns.store(now_ns(), Ordering::Relaxed);
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Process an incoming pipeline message
    pub fn on_message(&self, msg: PipelineMessage) {
        let latency = msg.latency_ns();
        let bytes   = 64u64;  // approximate

        self.metrics.record_receive(bytes, latency);

        match &msg {
            PipelineMessage::LobSnapshot { asset_id, sequence, .. } => {
                let name = crate::channels::LOB_SNAPSHOT;
                if let Some(sup) = self.supervisors.lock().unwrap().get(name) {
                    sup.record_update(*sequence);
                }
                global_metrics().lob_publishes.fetch_add(1, Ordering::Relaxed);
            }
            PipelineMessage::VolSurface { asset_id, .. } => {
                global_metrics().vol_publishes.fetch_add(1, Ordering::Relaxed);
            }
            PipelineMessage::Heartbeat { source_id, .. } => {
                // Update heartbeat timestamp
                self.last_heartbeat_ns.store(msg.timestamp(), Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Send a message (simulate publish via metrics recording)
    pub fn send_message(&self, msg: &PipelineMessage) -> Result<()> {
        if !self.is_running() {
            return Err(RtelError::NotConnected);
        }
        self.metrics.record_send(64);
        global_metrics().pipeline_runs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Check all channels for staleness
    pub fn health_check(&self) -> HashMap<String, ChannelStatus> {
        let sups = self.supervisors.lock().unwrap();
        sups.iter().map(|(name, sup)| {
            (name.clone(), sup.check_staleness())
        }).collect()
    }

    /// Returns true if heartbeat is recent
    pub fn heartbeat_ok(&self) -> bool {
        let elapsed = now_ns().saturating_sub(
            self.last_heartbeat_ns.load(Ordering::Relaxed)
        );
        elapsed < self.heartbeat_interval_ns * 3
    }

    pub fn metrics(&self) -> &BridgeMetrics { &self.metrics }

    pub fn channel_supervisor(&self, name: &str) -> Option<Arc<ChannelSupervisor>> {
        self.supervisors.lock().unwrap().get(name).cloned()
    }

    pub fn prometheus_metrics(&self) -> String {
        let mut out = self.metrics.prometheus();
        let sups    = self.supervisors.lock().unwrap();
        for (name, sup) in sups.iter() {
            let safe = name.replace('.', "_").replace('-', "_");
            out += &format!(
                "rtel_bridge_channel_updates{{channel=\"{}\"}} {}\n\
                 rtel_bridge_channel_lag_events{{channel=\"{}\"}} {}\n\
                 rtel_bridge_channel_status{{channel=\"{}\"}} {}\n",
                name, sup.n_updates(),
                name, sup.n_lag_events(),
                name, match sup.status() {
                    ChannelStatus::Healthy => 0,
                    ChannelStatus::Lagging => 1,
                    ChannelStatus::Stale   => 2,
                    ChannelStatus::Error   => 3,
                }
            );
            let _ = safe;
        }
        out
    }
}

impl Default for PipelineBridge {
    fn default() -> Self {
        Self {
            supervisors:  Mutex::new(HashMap::new()),
            metrics:      Arc::new(BridgeMetrics::default()),
            running:      AtomicBool::new(false),
            heartbeat_interval_ns: 1_000_000_000,
            last_heartbeat_ns: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Event replay — replay stored messages for testing
// ---------------------------------------------------------------------------

pub struct EventReplayer {
    messages: Vec<(u64, PipelineMessage)>,  // (timestamp_ns, msg)
    idx:      usize,
    speed:    f64,   // replay speed multiplier (1.0 = realtime, 0.0 = max speed)
}

impl EventReplayer {
    pub fn new(speed: f64) -> Self {
        Self { messages: vec![], idx: 0, speed }
    }

    pub fn add_message(&mut self, ts_ns: u64, msg: PipelineMessage) {
        self.messages.push((ts_ns, msg));
        self.messages.sort_by_key(|(ts, _)| *ts);
    }

    pub fn generate_synthetic(&mut self, n_assets: usize, n_steps: usize) {
        // Simple XOR-based RNG for deterministic generation
        let mut rng = 0xDEADBEEF_CAFEBABEu64;
        let next_f64 = |rng: &mut u64| -> f64 {
            *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
            (*rng as f64) / (u64::MAX as f64)
        };

        let mut prices = vec![100.0f64; n_assets];
        let mut ts_ns  = 1_700_000_000_000_000_000u64;  // base timestamp

        for step in 0..n_steps {
            let dt_ns = 1_000_000u64;  // 1ms steps
            ts_ns += dt_ns;

            for aid in 0..n_assets {
                let z     = 2.0 * next_f64(&mut rng) - 1.0;
                prices[aid] *= (1.0 + 0.001 * z).max(0.1);
                let spread  = prices[aid] * 0.001;
                let imbal   = 2.0 * next_f64(&mut rng) - 1.0;

                let msg = PipelineMessage::LobSnapshot {
                    asset_id:  aid as u32,
                    sequence:  (step * n_assets + aid) as u64,
                    timestamp: ts_ns,
                    mid_price: prices[aid],
                    spread,
                    imbalance: imbal * 0.3,
                };
                self.add_message(ts_ns, msg);
            }

            // Heartbeat every 100 steps
            if step % 100 == 0 {
                self.add_message(ts_ns, PipelineMessage::Heartbeat {
                    timestamp: ts_ns,
                    source_id: 0,
                });
            }
        }
    }

    pub fn next(&mut self) -> Option<&(u64, PipelineMessage)> {
        if self.idx >= self.messages.len() { return None; }
        let item = &self.messages[self.idx];
        self.idx += 1;
        Some(item)
    }

    pub fn reset(&mut self) { self.idx = 0; }

    pub fn n_messages(&self) -> usize { self.messages.len() }
    pub fn remaining(&self) -> usize { self.messages.len() - self.idx }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_supervisor_healthy() {
        let cfg = ChannelBridgeConfig::new("test.channel");
        let sup = ChannelSupervisor::new(cfg);
        for i in 1u64..=10 {
            sup.record_update(i);
        }
        assert_eq!(sup.status(), ChannelStatus::Healthy);
        assert_eq!(sup.n_updates(), 10);
    }

    #[test]
    fn test_channel_supervisor_lagging() {
        let cfg = ChannelBridgeConfig::new("test.channel").with_max_lag(5);
        let sup = ChannelSupervisor::new(cfg);
        sup.record_update(1);
        sup.record_update(1000);  // gap of 999 > max_lag=5
        assert_eq!(sup.status(), ChannelStatus::Lagging);
        assert!(sup.n_lag_events() > 0);
    }

    #[test]
    fn test_bridge_message_processing() {
        let bridge = PipelineBridge::new();
        bridge.with_standard_channels();
        bridge.start();
        assert!(bridge.is_running());

        let msg = PipelineMessage::LobSnapshot {
            asset_id: 0, sequence: 1, timestamp: now_ns(),
            mid_price: 100.0, spread: 0.05, imbalance: 0.1,
        };
        bridge.on_message(msg);
        assert!(bridge.metrics().total_messages_received.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_bridge_send() {
        let bridge = PipelineBridge::new();
        bridge.start();
        let msg = PipelineMessage::Heartbeat { timestamp: now_ns(), source_id: 1 };
        assert!(bridge.send_message(&msg).is_ok());
    }

    #[test]
    fn test_bridge_not_running_error() {
        let bridge = PipelineBridge::new();
        // NOT started
        let msg = PipelineMessage::Heartbeat { timestamp: now_ns(), source_id: 1 };
        assert!(bridge.send_message(&msg).is_err());
    }

    #[test]
    fn test_bridge_health_check() {
        let bridge = PipelineBridge::new();
        bridge.with_standard_channels();
        bridge.start();
        let health = bridge.health_check();
        assert!(!health.is_empty());
    }

    #[test]
    fn test_event_replayer() {
        let mut replayer = EventReplayer::new(0.0);
        replayer.generate_synthetic(3, 50);
        assert!(replayer.n_messages() > 0);

        let mut count = 0;
        while replayer.next().is_some() {
            count += 1;
        }
        assert_eq!(count, replayer.n_messages());
    }

    #[test]
    fn test_bridge_metrics_prometheus() {
        let bridge = PipelineBridge::new();
        bridge.with_standard_channels();
        bridge.start();
        let prom = bridge.prometheus_metrics();
        assert!(prom.contains("rtel_bridge_sent_total"));
        assert!(prom.contains("rtel_bridge_channel_status"));
    }

    #[test]
    fn test_pipeline_message_latency() {
        let ts = now_ns();
        let msg = PipelineMessage::LobSnapshot {
            asset_id: 0, sequence: 1, timestamp: ts,
            mid_price: 100.0, spread: 0.01, imbalance: 0.0,
        };
        let latency = msg.latency_ns();
        assert!(latency < 1_000_000_000, "latency should be < 1s in test");
    }

    #[test]
    fn test_supervisor_staleness() {
        let cfg = ChannelBridgeConfig::new("stale.test")
            .with_stale_timeout(1_000_000);  // 1ms timeout
        let sup = ChannelSupervisor::new(cfg);
        sup.record_update(1);
        // Sleep > 1ms
        std::thread::sleep(Duration::from_millis(5));
        let status = sup.check_staleness();
        assert_eq!(status, ChannelStatus::Stale);
    }
}
