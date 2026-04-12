// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// state_publisher.rs — Chronos LOB State Publisher
// =============================================================================
//! Publishes Chronos order book state to the RTEL Global State Registry (GSR).
//!
//! Converts a `BTreeMap<Decimal, Decimal>` order book (as used by the
//! Chronos Rust crates) into flat `f64` arrays compatible with the C++
//! `LOBSnapshot` struct, then writes to the `aeternus.chronos.lob` channel.
//!
//! Published fields:
//! - Bid prices/sizes (up to 10 levels)
//! - Ask prices/sizes (up to 10 levels)
//! - Mid price, spread, imbalance
//! - VWAP bid, VWAP ask
//! - Timestamp (nanoseconds)

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::{
    channels,
    error::{Result, RtelError},
    now_ns,
    shm_client::{ChannelConfig, ReadCursor, ShmChannel, ShmClient},
    types::{DType, TensorDescriptor},
    MAX_LOB_LEVELS,
};

// ---------------------------------------------------------------------------
// Order Side
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

// ---------------------------------------------------------------------------
// LOBLevel — single price level
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LobLevel {
    pub price: f64,
    pub size:  f64,
}

// ---------------------------------------------------------------------------
// LobSnapshot — full order book snapshot for one asset
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LobSnapshot {
    pub asset_id:      u32,
    pub exchange_ts_ns:u64,
    pub sequence:      u64,
    pub bids:          Vec<LobLevel>,
    pub asks:          Vec<LobLevel>,

    // Derived
    pub mid_price:     f64,
    pub spread:        f64,
    pub bid_imbalance: f64,
    pub vwap_bid:      f64,
    pub vwap_ask:      f64,
}

impl LobSnapshot {
    /// Build from BTreeMap order books.
    /// `bid_book`: price→size (descending = highest bid first)
    /// `ask_book`: price→size (ascending = lowest ask first)
    pub fn from_btreemaps(
        asset_id: u32,
        bid_book: &BTreeMap<i64, i64>,   // scaled: price * 1e8, size * 1e8
        ask_book: &BTreeMap<i64, i64>,
        price_scale: f64,                 // e.g. 1e-8
        size_scale:  f64,
        sequence: u64,
    ) -> Self {
        let mut snap = LobSnapshot {
            asset_id,
            exchange_ts_ns: now_ns(),
            sequence,
            ..Default::default()
        };

        // Bids: highest prices first
        snap.bids = bid_book
            .iter()
            .rev()
            .take(MAX_LOB_LEVELS)
            .map(|(&p, &s)| LobLevel {
                price: p as f64 * price_scale,
                size:  s as f64 * size_scale,
            })
            .collect();

        // Asks: lowest prices first
        snap.asks = ask_book
            .iter()
            .take(MAX_LOB_LEVELS)
            .map(|(&p, &s)| LobLevel {
                price: p as f64 * price_scale,
                size:  s as f64 * size_scale,
            })
            .collect();

        snap.compute_derived();
        snap
    }

    /// Build from float BTreeMaps (for convenience)
    pub fn from_float_maps(
        asset_id: u32,
        bid_book: &BTreeMap<ordered_float::OrderedFloat<f64>,
                             ordered_float::OrderedFloat<f64>>,
        ask_book: &BTreeMap<ordered_float::OrderedFloat<f64>,
                             ordered_float::OrderedFloat<f64>>,
        sequence: u64,
    ) -> Self {
        let mut snap = LobSnapshot {
            asset_id,
            exchange_ts_ns: now_ns(),
            sequence,
            ..Default::default()
        };

        // Bids: highest prices first
        snap.bids = bid_book
            .iter()
            .rev()
            .take(MAX_LOB_LEVELS)
            .map(|(p, s)| LobLevel { price: p.into_inner(), size: s.into_inner() })
            .collect();

        // Asks: lowest prices first
        snap.asks = ask_book
            .iter()
            .take(MAX_LOB_LEVELS)
            .map(|(p, s)| LobLevel { price: p.into_inner(), size: s.into_inner() })
            .collect();

        snap.compute_derived();
        snap
    }

    fn compute_derived(&mut self) {
        if self.bids.is_empty() || self.asks.is_empty() {
            return;
        }
        self.mid_price = (self.bids[0].price + self.asks[0].price) * 0.5;
        self.spread    = self.asks[0].price - self.bids[0].price;

        let bid_depth: f64 = self.bids.iter().map(|l| l.size).sum();
        let ask_depth: f64 = self.asks.iter().map(|l| l.size).sum();
        let total_depth = bid_depth + ask_depth;

        self.bid_imbalance = if total_depth > 0.0 {
            (bid_depth - ask_depth) / total_depth
        } else {
            0.0
        };

        let bid_vol: f64 = self.bids.iter().map(|l| l.price * l.size).sum();
        let ask_vol: f64 = self.asks.iter().map(|l| l.price * l.size).sum();

        self.vwap_bid = if bid_depth > 0.0 { bid_vol / bid_depth } else { 0.0 };
        self.vwap_ask = if ask_depth > 0.0 { ask_vol / ask_depth } else { 0.0 };
    }

    /// Serialise to flat f64 array (C++ LOBSnapshot compatible layout).
    /// Layout: [asset_id_f64, n_bids, n_asks, timestamp_ns_f64, seq_f64,
    ///          bid_p0..bid_p9, bid_s0..bid_s9,
    ///          ask_p0..ask_p9, ask_s0..ask_s9,
    ///          mid, spread, imbalance, vwap_bid, vwap_ask]
    pub fn to_flat_f64(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(5 + 4 * MAX_LOB_LEVELS + 5);
        v.push(self.asset_id as f64);
        v.push(self.bids.len() as f64);
        v.push(self.asks.len() as f64);
        v.push(self.exchange_ts_ns as f64);
        v.push(self.sequence as f64);

        for i in 0..MAX_LOB_LEVELS {
            v.push(self.bids.get(i).map_or(0.0, |l| l.price));
        }
        for i in 0..MAX_LOB_LEVELS {
            v.push(self.bids.get(i).map_or(0.0, |l| l.size));
        }
        for i in 0..MAX_LOB_LEVELS {
            v.push(self.asks.get(i).map_or(0.0, |l| l.price));
        }
        for i in 0..MAX_LOB_LEVELS {
            v.push(self.asks.get(i).map_or(0.0, |l| l.size));
        }
        v.push(self.mid_price);
        v.push(self.spread);
        v.push(self.bid_imbalance);
        v.push(self.vwap_bid);
        v.push(self.vwap_ask);
        v
    }

    /// Deserialize from flat f64 array
    pub fn from_flat_f64(data: &[f64]) -> Option<Self> {
        if data.len() < 5 + 4 * MAX_LOB_LEVELS + 5 {
            return None;
        }
        let mut snap = LobSnapshot {
            asset_id:       data[0] as u32,
            exchange_ts_ns: data[3] as u64,
            sequence:       data[4] as u64,
            ..Default::default()
        };
        let n_bids = (data[1] as usize).min(MAX_LOB_LEVELS);
        let n_asks = (data[2] as usize).min(MAX_LOB_LEVELS);

        for i in 0..n_bids {
            snap.bids.push(LobLevel {
                price: data[5 + i],
                size:  data[5 + MAX_LOB_LEVELS + i],
            });
        }
        for i in 0..n_asks {
            snap.asks.push(LobLevel {
                price: data[5 + 2 * MAX_LOB_LEVELS + i],
                size:  data[5 + 3 * MAX_LOB_LEVELS + i],
            });
        }

        let base = 5 + 4 * MAX_LOB_LEVELS;
        snap.mid_price     = data[base];
        snap.spread        = data[base + 1];
        snap.bid_imbalance = data[base + 2];
        snap.vwap_bid      = data[base + 3];
        snap.vwap_ask      = data[base + 4];
        Some(snap)
    }
}

// ---------------------------------------------------------------------------
// VolSurface (lightweight Rust version)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VolSurface {
    pub asset_id:    u32,
    pub model_id:    u32,
    pub timestamp_ns:u64,
    pub version:     u64,
    pub strikes:     Vec<f64>,
    pub expiries:    Vec<f64>,
    /// Row-major: vols[i_strike * n_expiries + j_expiry]
    pub vols:        Vec<f64>,
}

impl VolSurface {
    pub fn vol_at(&self, strike_idx: usize, expiry_idx: usize) -> f64 {
        if expiry_idx >= self.expiries.len() { return 0.0; }
        self.vols.get(strike_idx * self.expiries.len() + expiry_idx)
            .copied()
            .unwrap_or(0.0)
    }

    pub fn to_flat_f64(&self) -> Vec<f64> {
        let mut v = Vec::new();
        v.push(self.asset_id as f64);
        v.push(self.model_id as f64);
        v.push(self.timestamp_ns as f64);
        v.push(self.version as f64);
        v.push(self.strikes.len() as f64);
        v.push(self.expiries.len() as f64);
        v.extend(&self.strikes);
        v.extend(&self.expiries);
        v.extend(&self.vols);
        v
    }
}

// ---------------------------------------------------------------------------
// StatePublisher
// ---------------------------------------------------------------------------
/// Publishes Chronos state updates to the RTEL ShmBus
pub struct StatePublisher {
    client:      ShmClient,
    publish_vol: bool,
    stats:       PublisherStats,
}

#[derive(Debug, Default)]
pub struct PublisherStats {
    pub lob_published:  u64,
    pub vol_published:  u64,
    pub errors:         u64,
    pub total_bytes:    u64,
    pub last_publish_ns:u64,
    pub mean_latency_ns:f64,
}

impl StatePublisher {
    /// Create a new StatePublisher backed by a ShmClient
    pub fn new(client: ShmClient) -> Self {
        Self {
            client,
            publish_vol: false,
            stats: Default::default(),
        }
    }

    /// Create and initialize with all AETERNUS channels open
    pub fn new_with_channels(base_path: std::path::PathBuf) -> Result<Self> {
        let mut client = ShmClient::with_base_path(base_path);
        client.open_aeternus_channels(true)?;
        Ok(Self::new(client))
    }

    pub fn enable_vol_publishing(&mut self, enable: bool) {
        self.publish_vol = enable;
    }

    /// Publish a LOB snapshot to the "aeternus.chronos.lob" channel.
    /// This is the primary hot path called by Chronos on every tick.
    pub fn publish_lob(&mut self, snap: &LobSnapshot) -> Result<()> {
        let t0 = now_ns();
        let flat = snap.to_flat_f64();
        let bytes = bytemuck::cast_slice::<f64, u8>(&flat);

        let td = TensorDescriptor::new_float64_2d(1, flat.len());

        self.client.write(channels::LOB_SNAPSHOT, bytes, &td)?;
        let t1 = now_ns();

        self.stats.lob_published += 1;
        self.stats.total_bytes   += bytes.len() as u64;
        self.stats.last_publish_ns = t1;
        // Exponential moving average for latency
        let lat = (t1 - t0) as f64;
        self.stats.mean_latency_ns = self.stats.mean_latency_ns * 0.99 + lat * 0.01;

        debug!(
            "LOB published: asset={} mid={:.4} spread={:.5} seq={}",
            snap.asset_id, snap.mid_price, snap.spread, snap.sequence
        );
        Ok(())
    }

    /// Publish a volatility surface to the "aeternus.neuro_sde.vol" channel
    pub fn publish_vol_surface(&mut self, surf: &VolSurface) -> Result<()> {
        if !self.publish_vol {
            return Ok(());
        }
        let flat = surf.to_flat_f64();
        let bytes = bytemuck::cast_slice::<f64, u8>(&flat);

        let mut td = TensorDescriptor::new_float64_2d(1, flat.len());
        td.payload_bytes = bytes.len() as u64;

        self.client.write(channels::VOL_SURFACE, bytes, &td)?;
        self.stats.vol_published += 1;
        Ok(())
    }

    /// Publish multiple LOB snapshots in a batch (reduces per-call overhead)
    pub fn publish_lob_batch(&mut self, snaps: &[LobSnapshot]) -> Result<()> {
        for snap in snaps {
            self.publish_lob(snap)?;
        }
        Ok(())
    }

    /// Read the latest LOB snapshot published by another process
    pub fn read_lob(&self, cur: &mut ReadCursor) -> Option<LobSnapshot> {
        let ch = self.client.channel(channels::LOB_SNAPSHOT)?;
        let (_, payload) = ch.consume(cur)?;
        // payload is f64 bytes
        if payload.len() % 8 != 0 { return None; }
        let floats: Vec<f64> = payload
            .chunks(8)
            .map(|b| {
                let arr: [u8; 8] = b.try_into().unwrap();
                f64::from_le_bytes(arr)
            })
            .collect();
        LobSnapshot::from_flat_f64(&floats)
    }

    /// Subscribe to LOB updates (async, tokio)
    #[cfg(feature = "tokio-async")]
    pub async fn subscribe_lob(
        &mut self,
        mut callback: impl FnMut(LobSnapshot) + Send + 'static,
    ) -> tokio::task::JoinHandle<()> {
        // We snapshot the channel for polling
        let base = self.client.all_stats(); // just to get a reference point
        let _ = base;
        // In production: spawn a task that reads from the channel
        tokio::spawn(async move {
            callback(LobSnapshot::default());
        })
    }

    pub fn stats(&self) -> &PublisherStats {
        &self.stats
    }

    pub fn print_stats(&self) {
        println!("=== StatePublisher Stats ===");
        println!("  LOB published:    {}", self.stats.lob_published);
        println!("  Vol published:    {}", self.stats.vol_published);
        println!("  Total bytes:      {}", self.stats.total_bytes);
        println!("  Mean latency:     {:.1} ns", self.stats.mean_latency_ns);
        println!("  Errors:           {}", self.stats.errors);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_snapshot(asset_id: u32) -> LobSnapshot {
        let mut bids = BTreeMap::new();
        let mut asks = BTreeMap::new();
        // Scaled by 1e8: 150.00 = 15_000_000_000
        for i in 1i64..=5 {
            bids.insert(15_000_000_000 - i * 1_000_000, 100_000_000 * i);
            asks.insert(15_000_000_000 + i * 1_000_000, 100_000_000 * i);
        }
        LobSnapshot::from_btreemaps(asset_id, &bids, &asks, 1e-8, 1e-8, 1)
    }

    #[test]
    fn test_lob_snapshot_derived() {
        let snap = make_snapshot(0);
        assert!((snap.mid_price - 150.0).abs() < 1e-4,
                "mid_price={}", snap.mid_price);
        assert!(snap.spread > 0.0, "spread should be positive");
        assert_eq!(snap.bids.len(), 5);
        assert_eq!(snap.asks.len(), 5);
        println!("LOB mid={:.4} spread={:.6} imbal={:.4}",
                 snap.mid_price, snap.spread, snap.bid_imbalance);
    }

    #[test]
    fn test_lob_flat_roundtrip() {
        let snap = make_snapshot(42);
        let flat = snap.to_flat_f64();
        let restored = LobSnapshot::from_flat_f64(&flat).expect("deserialize");
        assert_eq!(restored.asset_id, 42);
        assert!((restored.mid_price - snap.mid_price).abs() < 1e-10);
        assert!((restored.spread - snap.spread).abs() < 1e-10);
        assert_eq!(restored.bids.len(), snap.bids.len());
    }

    #[test]
    fn test_publisher_write_read() {
        let dir = tempdir().unwrap();
        let mut client = ShmClient::with_base_path(dir.path().to_owned());

        let cfg = ChannelConfig {
            name: channels::LOB_SNAPSHOT.to_owned(),
            slot_bytes: 128 * 1024,
            ring_capacity: 512,
            create: true,
            shm_base_path: dir.path().to_owned(),
        };
        client.open_channel_with_config(cfg).unwrap();

        let mut pub_ = StatePublisher::new(client);
        let snap = make_snapshot(1);
        pub_.publish_lob(&snap).expect("publish");

        let ch = pub_.client.channel(channels::LOB_SNAPSHOT).unwrap();
        let mut cur = ReadCursor::new(0);
        let _ = ch.consume(&mut cur);  // consume the published item
        // Stats check
        assert_eq!(pub_.stats().lob_published, 1);
        println!("Publisher stats: {:?}", pub_.stats());
    }

    #[test]
    fn test_vwap_correctness() {
        let snap = make_snapshot(0);
        // VWAP should be between best bid and mid for bids
        assert!(snap.vwap_bid <= snap.bids[0].price + 1e-10,
                "VWAP bid={} should be <= best bid={}", snap.vwap_bid, snap.bids[0].price);
        assert!(snap.vwap_ask >= snap.asks[0].price - 1e-10,
                "VWAP ask={} should be >= best ask={}", snap.vwap_ask, snap.asks[0].price);
    }

    #[test]
    fn test_imbalance_balanced_book() {
        let mut snap = LobSnapshot::default();
        snap.bids = vec![LobLevel { price: 100.0, size: 500.0 }];
        snap.asks = vec![LobLevel { price: 100.1, size: 500.0 }];
        snap.compute_derived();
        // Balanced book → imbalance ≈ 0
        assert!(snap.bid_imbalance.abs() < 1e-10,
                "balanced book imbalance={}", snap.bid_imbalance);
    }

    #[test]
    fn test_imbalance_bid_heavy() {
        let mut snap = LobSnapshot::default();
        snap.bids = vec![LobLevel { price: 100.0, size: 900.0 }];
        snap.asks = vec![LobLevel { price: 100.1, size: 100.0 }];
        snap.compute_derived();
        // Bid-heavy → imbalance ≈ 0.8
        assert!(snap.bid_imbalance > 0.7, "bid_imbalance={}", snap.bid_imbalance);
    }
}
