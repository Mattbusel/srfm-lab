// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// lib.rs — Rust RTEL Client Library Root
// =============================================================================
//! Rust client for the AETERNUS RTEL shared-memory bus.
//!
//! Provides:
//! - [`ShmClient`] — read/write named RTEL channels via `mmap`
//! - [`StatePublisher`] — publishes Chronos LOB state to the GSR
//! - Async tokio interface for non-blocking channel operations
//!
//! # Architecture
//!
//! ```text
//! Chronos Rust crates
//!       │
//!       ▼
//!  StatePublisher
//!       │ (converts BTreeMap<Decimal, Decimal> order book → flat arrays)
//!       ▼
//!  ShmClient::write("aeternus.chronos.lob")
//!       │ (zero-copy mmap)
//!       ▼
//! C++ ShmBus ring buffer
//!       │
//!       ▼
//!  Python shm_reader.py / TensorNet / Lumina ...
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, clippy::pedantic)]
#![allow(clippy::module_name_repetitions, clippy::cast_possible_truncation)]

pub mod shm_client;
pub mod state_publisher;
pub mod types;
pub mod error;
pub mod metrics;
pub mod order_book;
pub mod execution_engine;
pub mod backtest_engine;
pub mod pipeline_bridge;
pub mod sim_engine;

pub use shm_client::{ShmClient, ShmChannel, ReadCursor, ChannelConfig};
pub use state_publisher::{StatePublisher, LobSnapshot, VolSurface};
pub use types::{DType, TensorDescriptor, SlotHeader, RingControl};
pub use error::{RtelError, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Magic value for RTEL blobs
pub const RTEL_MAGIC: u64 = 0xAE7E_4E55_5254_4C00;

/// Cache line size in bytes
pub const CACHE_LINE_SIZE: usize = 64;

/// Default slot size in bytes (64 KB)
pub const DEFAULT_SLOT_BYTES: usize = 64 * 1024;

/// Default ring capacity (must be power of 2)
pub const DEFAULT_RING_CAPACITY: usize = 1024;

/// Maximum number of assets supported
pub const MAX_ASSETS: usize = 512;

/// Maximum LOB levels
pub const MAX_LOB_LEVELS: usize = 10;

/// Maximum number of strikes in vol surface
pub const MAX_STRIKES: usize = 50;

/// Maximum number of expiries in vol surface
pub const MAX_EXPIRIES: usize = 12;

// Standard AETERNUS channel names
pub mod channels {
    pub const LOB_SNAPSHOT:    &str = "aeternus.chronos.lob";
    pub const VOL_SURFACE:     &str = "aeternus.neuro_sde.vol";
    pub const TENSOR_COMP:     &str = "aeternus.tensornet.compressed";
    pub const GRAPH_ADJ:       &str = "aeternus.omni_graph.adj";
    pub const LUMINA_PRED:     &str = "aeternus.lumina.predictions";
    pub const AGENT_ACTIONS:   &str = "aeternus.hyper_agent.actions";
    pub const AGENT_WEIGHTS:   &str = "aeternus.hyper_agent.weights";
    pub const PIPELINE_EVENTS: &str = "aeternus.rtel.pipeline_events";
    pub const HEARTBEAT:       &str = "aeternus.rtel.heartbeat";
}

/// Get current wall-clock time in nanoseconds
pub fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Align `value` up to the nearest multiple of `align` (must be power of 2)
#[inline]
pub const fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

/// Check if n is a power of two
#[inline]
pub const fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Round up to next power of two
pub fn next_power_of_two(mut n: usize) -> usize {
    if n == 0 { return 1; }
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(1024), 1024);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(is_power_of_two(1024));
    }

    #[test]
    fn test_now_ns() {
        let t0 = now_ns();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let t1 = now_ns();
        assert!(t1 > t0, "now_ns should be monotonically increasing");
        assert!(t1 - t0 < 100_000_000, "sleep should be < 100ms");
    }
}
