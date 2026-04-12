// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// shm_client.rs — Rust Shared Memory Bus Client
// =============================================================================
//! Rust client for reading/writing RTEL shared-memory bus channels.
//!
//! Uses `memmap2` for zero-copy mmap access, compatible with the C++ ShmBus.
//! Provides both synchronous and async (tokio) interfaces.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use memmap2::{MmapMut, MmapOptions};
use tracing::{debug, error, info, warn};

use crate::{
    align_up, channels, now_ns, next_power_of_two,
    error::{Result, RtelError},
    types::{DType, RingControl, SlotHeader, TensorDescriptor},
    CACHE_LINE_SIZE, DEFAULT_RING_CAPACITY, DEFAULT_SLOT_BYTES, RTEL_MAGIC,
};

// ---------------------------------------------------------------------------
// ChannelConfig
// ---------------------------------------------------------------------------
/// Configuration for opening a shared-memory channel
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Channel name (e.g. "aeternus.chronos.lob")
    pub name: String,
    /// Slot size in bytes (default 64 KB)
    pub slot_bytes: usize,
    /// Ring capacity in slots (must be power of 2, default 1024)
    pub ring_capacity: usize,
    /// Create the channel if it doesn't exist
    pub create: bool,
    /// Base path for shm files (default: /tmp)
    pub shm_base_path: PathBuf,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            slot_bytes: DEFAULT_SLOT_BYTES,
            ring_capacity: DEFAULT_RING_CAPACITY,
            create: true,
            shm_base_path: PathBuf::from("/tmp"),
        }
    }
}

impl ChannelConfig {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            ..Default::default()
        }
    }

    pub fn with_slot_bytes(mut self, bytes: usize) -> Self {
        self.slot_bytes = bytes;
        self
    }

    pub fn with_ring_capacity(mut self, cap: usize) -> Self {
        self.ring_capacity = next_power_of_two(cap);
        self
    }

    pub fn readonly(mut self) -> Self {
        self.create = false;
        self
    }

    fn shm_file_path(&self) -> PathBuf {
        let safe_name = self.name.replace('.', "_");
        self.shm_base_path.join(format!("aeternus_rtel_{}", safe_name))
    }

    fn total_bytes(&self) -> usize {
        let ctrl_size = align_up(std::mem::size_of::<RingControl>(), CACHE_LINE_SIZE);
        ctrl_size + self.ring_capacity * self.slot_bytes
    }
}

// ---------------------------------------------------------------------------
// ReadCursor — per-consumer read position
// ---------------------------------------------------------------------------
/// Tracks a consumer's position in the ring
#[derive(Debug, Default, Clone)]
pub struct ReadCursor {
    pub next_seq: u64,
}

impl ReadCursor {
    pub fn new(start_seq: u64) -> Self {
        Self { next_seq: start_seq }
    }
}

// ---------------------------------------------------------------------------
// ChannelStats
// ---------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct ChannelStats {
    pub published_total: u64,
    pub consumed_total:  u64,
    pub dropped_total:   u64,
    pub bytes_written:   u64,
    pub bytes_read:      u64,
}

// ---------------------------------------------------------------------------
// ShmChannel — single named channel
// ---------------------------------------------------------------------------
/// A single shared-memory ring channel
pub struct ShmChannel {
    config:      ChannelConfig,
    mmap:        MmapMut,
    ctrl_offset: usize,
    ring_offset: usize,

    // Per-process stats
    stat_published: AtomicU64,
    stat_consumed:  AtomicU64,
    stat_dropped:   AtomicU64,
    stat_bw:        AtomicU64,
    stat_br:        AtomicU64,
}

// Safety: ShmChannel uses only atomic operations on the mmap region
unsafe impl Send for ShmChannel {}
unsafe impl Sync for ShmChannel {}

impl ShmChannel {
    /// Open or create a channel
    pub fn open(cfg: ChannelConfig) -> Result<Self> {
        let total = cfg.total_bytes();
        let path  = cfg.shm_file_path();

        debug!("ShmChannel::open '{}' at {:?} ({} bytes)", cfg.name, path, total);

        let file = if cfg.create {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&path)?
        } else {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)?
        };

        // Resize to required length
        if cfg.create {
            file.set_len(total as u64)?;
        }

        let mmap = unsafe {
            MmapOptions::new()
                .len(total)
                .map_mut(&file)?
        };

        let ctrl_offset = 0;
        let ring_offset = align_up(std::mem::size_of::<RingControl>(), CACHE_LINE_SIZE);

        let mut ch = ShmChannel {
            config: cfg.clone(),
            mmap,
            ctrl_offset,
            ring_offset,
            stat_published: AtomicU64::new(0),
            stat_consumed:  AtomicU64::new(0),
            stat_dropped:   AtomicU64::new(0),
            stat_bw:        AtomicU64::new(0),
            stat_br:        AtomicU64::new(0),
        };

        if cfg.create {
            ch.initialize_ring();
        } else {
            ch.validate_ring()?;
        }

        info!("ShmChannel '{}' opened ({} slots × {} bytes)",
              cfg.name, cfg.ring_capacity, cfg.slot_bytes);
        Ok(ch)
    }

    fn initialize_ring(&mut self) {
        let ctrl = self.ring_control_mut();
        ctrl.magic      = RTEL_MAGIC;
        ctrl.slot_bytes = self.config.slot_bytes as u64;
        ctrl.ring_cap   = self.config.ring_capacity as u64;
        ctrl.schema_ver = 1;
        ctrl.write_seq  = 1;
        ctrl.min_read_seq = 0;
        // Copy channel name
        let name_bytes = self.config.name.as_bytes();
        let copy_len = name_bytes.len().min(63);
        ctrl.channel_name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
    }

    fn validate_ring(&self) -> Result<()> {
        let ctrl = self.ring_control();
        if ctrl.magic != RTEL_MAGIC {
            return Err(RtelError::BadMagic);
        }
        Ok(())
    }

    fn ring_control(&self) -> &RingControl {
        unsafe {
            &*(self.mmap.as_ptr().add(self.ctrl_offset) as *const RingControl)
        }
    }

    fn ring_control_mut(&mut self) -> &mut RingControl {
        unsafe {
            &mut *(self.mmap.as_mut_ptr().add(self.ctrl_offset) as *mut RingControl)
        }
    }

    fn slot_ptr(&self, idx: usize) -> *const SlotHeader {
        let idx = idx & (self.config.ring_capacity - 1);
        unsafe {
            self.mmap.as_ptr()
                .add(self.ring_offset + idx * self.config.slot_bytes)
                as *const SlotHeader
        }
    }

    fn slot_ptr_mut(&mut self, idx: usize) -> *mut SlotHeader {
        let idx = idx & (self.config.ring_capacity - 1);
        unsafe {
            self.mmap.as_mut_ptr()
                .add(self.ring_offset + idx * self.config.slot_bytes)
                as *mut SlotHeader
        }
    }

    fn payload_ptr(&self, idx: usize) -> *const u8 {
        let idx = idx & (self.config.ring_capacity - 1);
        unsafe {
            self.mmap.as_ptr()
                .add(self.ring_offset + idx * self.config.slot_bytes
                     + std::mem::size_of::<SlotHeader>())
        }
    }

    fn payload_ptr_mut(&mut self, idx: usize) -> *mut u8 {
        let idx = idx & (self.config.ring_capacity - 1);
        unsafe {
            self.mmap.as_mut_ptr()
                .add(self.ring_offset + idx * self.config.slot_bytes
                     + std::mem::size_of::<SlotHeader>())
        }
    }

    fn payload_capacity(&self) -> usize {
        self.config.slot_bytes - std::mem::size_of::<SlotHeader>()
    }

    /// Claim the next write slot. Returns the sequence number of the claimed slot.
    pub fn claim(&mut self) -> Result<u64> {
        let ctrl = self.ring_control_mut();
        // Atomically increment write_seq
        let seq = unsafe {
            let ptr = &ctrl.write_seq as *const u64 as *const AtomicU64;
            (*ptr).fetch_add(1, Ordering::AcqRel)
        };

        let idx = seq as usize & (self.config.ring_capacity - 1);
        let slot = unsafe { &mut *self.slot_ptr_mut(idx) };

        // Check if ring is full (slot not yet consumed)
        let prev_seq = unsafe {
            let ptr = &slot.sequence as *const u64 as *const AtomicU64;
            (*ptr).load(Ordering::Acquire)
        };

        if prev_seq + self.config.ring_capacity as u64 <= seq && prev_seq != 0 {
            warn!("Ring full for channel '{}'", self.config.name);
            return Err(RtelError::RingFull);
        }

        // Initialize slot header
        slot.magic      = RTEL_MAGIC;
        slot.flags      = 0;
        slot.schema_ver = 1;
        Ok(seq)
    }

    /// Publish a previously claimed slot
    pub fn publish_slot(
        &mut self,
        seq: u64,
        data: &[u8],
        td: &TensorDescriptor,
    ) -> Result<()> {
        if data.len() > self.payload_capacity() {
            return Err(RtelError::InvalidArg(format!(
                "Data too large: {} > {}", data.len(), self.payload_capacity()
            )));
        }

        let idx = seq as usize & (self.config.ring_capacity - 1);
        let slot = unsafe { &mut *self.slot_ptr_mut(idx) };

        slot.timestamp_ns = now_ns();
        slot.tensor = *td;
        slot.tensor.payload_bytes = data.len() as u64;
        slot.flags = SlotHeader::FLAG_VALID;

        // Copy payload
        unsafe {
            let dst = self.payload_ptr_mut(idx);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        // Store published sequence (seq + 1 → readable)
        let seq_ptr = unsafe {
            &slot.sequence as *const u64 as *const AtomicU64
        };
        unsafe { (*seq_ptr).store(seq + 1, Ordering::Release); }

        self.stat_published.fetch_add(1, Ordering::Relaxed);
        self.stat_bw.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Convenience: claim + copy + publish in one call
    pub fn write(&mut self, data: &[u8], td: &TensorDescriptor) -> Result<()> {
        let seq = self.claim()?;
        self.publish_slot(seq, data, td)
    }

    /// Peek at next available slot; returns None if no data at cursor
    pub fn peek<'a>(&'a self, cur: &ReadCursor) -> Option<(&'a SlotHeader, &'a [u8])> {
        let idx = cur.next_seq as usize & (self.config.ring_capacity - 1);
        let slot = unsafe { &*self.slot_ptr(idx) };

        let seq = unsafe {
            let ptr = &slot.sequence as *const u64 as *const AtomicU64;
            (*ptr).load(Ordering::Acquire)
        };

        if seq == cur.next_seq + 1 {
            let payload = unsafe {
                let ptr = self.payload_ptr(idx);
                let len = slot.tensor.payload_bytes.min(self.payload_capacity() as u64) as usize;
                std::slice::from_raw_parts(ptr, len)
            };
            Some((slot, payload))
        } else {
            None
        }
    }

    /// Consume next available slot; advances cursor
    pub fn consume<'a>(&'a self, cur: &mut ReadCursor) -> Option<(&'a SlotHeader, &'a [u8])> {
        let result = self.peek(cur);
        if result.is_some() {
            cur.next_seq += 1;
            self.stat_consumed.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Read raw bytes (copies into caller buffer); returns bytes read or 0
    pub fn read_into(&self, cur: &mut ReadCursor, buf: &mut Vec<u8>) -> Option<TensorDescriptor> {
        let (slot, payload) = self.consume(cur)?;
        buf.clear();
        buf.extend_from_slice(payload);
        self.stat_br.fetch_add(payload.len() as u64, Ordering::Relaxed);
        Some(slot.tensor)
    }

    /// Wait (blocking) until a slot is available, up to `timeout`
    pub fn wait_for_data(&self, cur: &ReadCursor, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if self.peek(cur).is_some() {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::yield_now();
        }
    }

    /// Current write sequence (for consumer initialization)
    pub fn current_seq(&self) -> u64 {
        let ctrl = self.ring_control();
        unsafe {
            let ptr = &ctrl.write_seq as *const u64 as *const AtomicU64;
            (*ptr).load(Ordering::Acquire)
        }
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub fn stats(&self) -> ChannelStats {
        ChannelStats {
            published_total: self.stat_published.load(Ordering::Relaxed),
            consumed_total:  self.stat_consumed.load(Ordering::Relaxed),
            dropped_total:   self.stat_dropped.load(Ordering::Relaxed),
            bytes_written:   self.stat_bw.load(Ordering::Relaxed),
            bytes_read:      self.stat_br.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// ShmClient — manages multiple channels
// ---------------------------------------------------------------------------
/// High-level client managing multiple shm-bus channels
pub struct ShmClient {
    channels: std::collections::HashMap<String, ShmChannel>,
    base_path: PathBuf,
}

impl ShmClient {
    /// Create a new ShmClient
    pub fn new() -> Self {
        Self::with_base_path(PathBuf::from("/tmp"))
    }

    pub fn with_base_path(base: PathBuf) -> Self {
        Self {
            channels: Default::default(),
            base_path: base,
        }
    }

    /// Open a channel (create=true) or attach to existing (create=false)
    pub fn open_channel(&mut self, name: &str, create: bool) -> Result<()> {
        if self.channels.contains_key(name) {
            return Ok(());
        }
        let cfg = ChannelConfig {
            name: name.to_owned(),
            create,
            shm_base_path: self.base_path.clone(),
            ..Default::default()
        };
        let ch = ShmChannel::open(cfg)?;
        self.channels.insert(name.to_owned(), ch);
        Ok(())
    }

    /// Open a channel with custom config
    pub fn open_channel_with_config(&mut self, cfg: ChannelConfig) -> Result<()> {
        let name = cfg.name.clone();
        if self.channels.contains_key(&name) {
            return Ok(());
        }
        let ch = ShmChannel::open(cfg)?;
        self.channels.insert(name, ch);
        Ok(())
    }

    /// Open all standard AETERNUS channels
    pub fn open_aeternus_channels(&mut self, create: bool) -> Result<()> {
        let names = [
            (channels::LOB_SNAPSHOT,    128 * 1024usize, 512usize),
            (channels::VOL_SURFACE,     64  * 1024,      256),
            (channels::TENSOR_COMP,     256 * 1024,      256),
            (channels::GRAPH_ADJ,       64  * 1024,      256),
            (channels::LUMINA_PRED,     64  * 1024,      512),
            (channels::AGENT_ACTIONS,   16  * 1024,     1024),
            (channels::PIPELINE_EVENTS, 4   * 1024,     2048),
            (channels::HEARTBEAT,       1   * 1024,       64),
        ];
        for (name, slot_bytes, cap) in &names {
            let cfg = ChannelConfig {
                name: name.to_string(),
                slot_bytes: *slot_bytes,
                ring_capacity: *cap,
                create,
                shm_base_path: self.base_path.clone(),
            };
            if let Err(e) = self.open_channel_with_config(cfg) {
                warn!("Failed to open channel '{}': {}", name, e);
            }
        }
        Ok(())
    }

    /// Get channel by name
    pub fn channel(&self, name: &str) -> Option<&ShmChannel> {
        self.channels.get(name)
    }

    pub fn channel_mut(&mut self, name: &str) -> Option<&mut ShmChannel> {
        self.channels.get_mut(name)
    }

    /// Write data to a named channel
    pub fn write(&mut self, channel: &str, data: &[u8], td: &TensorDescriptor) -> Result<()> {
        let ch = self.channels.get_mut(channel)
            .ok_or_else(|| RtelError::ChannelNotFound(channel.to_owned()))?;
        ch.write(data, td)
    }

    /// Read from a named channel; returns (TensorDescriptor, data_bytes) or None
    pub fn read(&self, channel: &str, cur: &mut ReadCursor) -> Option<(TensorDescriptor, Vec<u8>)> {
        let ch = self.channels.get(channel)?;
        let (slot, payload) = ch.consume(cur)?;
        Some((slot.tensor, payload.to_vec()))
    }

    /// Async write (tokio)
    #[cfg(feature = "tokio-async")]
    pub async fn write_async(
        &mut self,
        channel: &str,
        data: &[u8],
        td: &TensorDescriptor,
    ) -> Result<()> {
        // For non-blocking write, we simply try and return
        self.write(channel, data, td)
    }

    /// Async read with timeout (tokio)
    #[cfg(feature = "tokio-async")]
    pub async fn read_async(
        &self,
        channel: &str,
        cur: &mut ReadCursor,
        timeout_ms: u64,
    ) -> Option<(TensorDescriptor, Vec<u8>)> {
        let deadline = tokio::time::Instant::now()
            + Duration::from_millis(timeout_ms);
        loop {
            if let Some(result) = self.read(channel, cur) {
                return Some(result);
            }
            if tokio::time::Instant::now() >= deadline {
                return None;
            }
            tokio::task::yield_now().await;
        }
    }

    /// Subscribe to a channel: returns a stream of (TensorDescriptor, Vec<u8>)
    #[cfg(feature = "tokio-async")]
    pub fn subscribe(
        self_arc: Arc<tokio::sync::Mutex<ShmClient>>,
        channel: String,
        poll_interval_ms: u64,
    ) -> tokio_stream::wrappers::ReceiverStream<(TensorDescriptor, Vec<u8>)> {
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        tokio::spawn(async move {
            let mut cur = {
                let client = self_arc.lock().await;
                let seq = client.channel(&channel)
                    .map(|ch| ch.current_seq())
                    .unwrap_or(0);
                ReadCursor::new(seq)
            };

            loop {
                let result = {
                    let client = self_arc.lock().await;
                    client.channels.get(&channel)
                        .and_then(|ch| {
                            ch.consume(&mut cur)
                                .map(|(slot, payload)| (slot.tensor, payload.to_vec()))
                        })
                };
                match result {
                    Some(data) => {
                        if tx.send(data).await.is_err() {
                            break;
                        }
                    }
                    None => {
                        tokio::time::sleep(Duration::from_millis(poll_interval_ms)).await;
                    }
                }
            }
        });
        tokio_stream::wrappers::ReceiverStream::new(rx)
    }

    pub fn list_channels(&self) -> Vec<String> {
        self.channels.keys().cloned().collect()
    }

    pub fn all_stats(&self) -> std::collections::HashMap<String, ChannelStats> {
        self.channels.iter()
            .map(|(k, v)| (k.clone(), v.stats()))
            .collect()
    }

    pub fn print_stats(&self) {
        println!("{:<45} {:>10} {:>10} {:>12}", "Channel", "Published", "Consumed", "BytesWritten");
        println!("{}", "-".repeat(82));
        for (name, stats) in self.all_stats() {
            println!("{:<45} {:>10} {:>10} {:>12}",
                     name, stats.published_total, stats.consumed_total, stats.bytes_written);
        }
    }
}

impl Default for ShmClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_channel_config_defaults() {
        let cfg = ChannelConfig::new("test.channel");
        assert_eq!(cfg.slot_bytes, DEFAULT_SLOT_BYTES);
        assert_eq!(cfg.ring_capacity, DEFAULT_RING_CAPACITY);
        assert!(cfg.create);
    }

    #[test]
    fn test_channel_open_write_read() {
        let dir = tempdir().unwrap();
        let cfg = ChannelConfig {
            name: "test.channel".to_owned(),
            slot_bytes: 4096,
            ring_capacity: 16,
            create: true,
            shm_base_path: dir.path().to_owned(),
        };
        let mut ch = ShmChannel::open(cfg).expect("open channel");
        let mut cur = ReadCursor::new(ch.current_seq());

        let data = b"hello rtel rust";
        let td = TensorDescriptor::new_float32_1d(data.len() / 4);
        ch.write(data, &td).expect("write");

        let (header, payload) = ch.consume(&mut cur).expect("consume");
        assert_eq!(payload, data);
        assert!(header.is_valid());
        println!("Rust channel test OK: {} bytes", payload.len());
    }

    #[test]
    fn test_shm_client_multiple_channels() {
        let dir = tempdir().unwrap();
        let mut client = ShmClient::with_base_path(dir.path().to_owned());

        for i in 0..3 {
            let name = format!("test.ch{}", i);
            let cfg = ChannelConfig {
                name: name.clone(),
                slot_bytes: 4096,
                ring_capacity: 8,
                create: true,
                shm_base_path: dir.path().to_owned(),
            };
            client.open_channel_with_config(cfg).unwrap();
        }

        assert_eq!(client.list_channels().len(), 3);
    }

    #[test]
    fn test_read_cursor_independence() {
        let dir = tempdir().unwrap();
        let cfg = ChannelConfig {
            name: "test.cursor".to_owned(),
            slot_bytes: 4096,
            ring_capacity: 32,
            create: true,
            shm_base_path: dir.path().to_owned(),
        };
        let mut ch = ShmChannel::open(cfg).unwrap();

        let start_seq = ch.current_seq();
        let mut c1 = ReadCursor::new(start_seq);
        let mut c2 = ReadCursor::new(start_seq);

        let td = TensorDescriptor::new_float32_1d(1);
        for i in 0..5u8 {
            ch.write(&[i, 0, 0, 0], &td).unwrap();
        }

        // Both cursors should read the same 5 items independently
        let mut sum1 = 0u32;
        let mut sum2 = 0u32;
        for _ in 0..5 {
            if let Some((_, p)) = ch.consume(&mut c1) { sum1 += p[0] as u32; }
            if let Some((_, p)) = ch.consume(&mut c2) { sum2 += p[0] as u32; }
        }
        assert_eq!(sum1, sum2);
        assert_eq!(sum1, 0+1+2+3+4);
    }
}
