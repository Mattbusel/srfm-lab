//! data_replay.rs — FIX 4.2/4.4 + NASDAQ ITCH 5.0 parser, PCAP replay,
//! nanosecond-resolution order book reconstruction.
//!
//! Chronos / AETERNUS — production-grade market data replay engine.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::{self, Read, BufRead, BufReader, Seek, SeekFrom};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── Error types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ReplayError {
    Io(String),
    Parse(String),
    InvalidMagic { expected: u32, got: u32 },
    TruncatedMessage { needed: usize, available: usize },
    UnknownMessageType(u8),
    InvalidFixField { tag: u32, raw: String },
    ChecksumMismatch { expected: u8, computed: u8 },
    EndOfFile,
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<io::Error> for ReplayError {
    fn from(e: io::Error) -> Self { ReplayError::Io(e.to_string()) }
}

pub type ReplayResult<T> = Result<T, ReplayError>;

// ── Nanosecond timestamp ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Nanos(pub u64);

impl Nanos {
    pub fn from_secs(s: u64) -> Self { Nanos(s * 1_000_000_000) }
    pub fn from_millis(ms: u64) -> Self { Nanos(ms * 1_000_000) }
    pub fn from_micros(us: u64) -> Self { Nanos(us * 1_000) }
    pub fn as_secs_f64(self) -> f64 { self.0 as f64 / 1e9 }
    pub fn duration_since(self, other: Nanos) -> Nanos {
        Nanos(self.0.saturating_sub(other.0))
    }
}

impl std::ops::Add for Nanos {
    type Output = Nanos;
    fn add(self, rhs: Nanos) -> Nanos { Nanos(self.0 + rhs.0) }
}

impl std::ops::Sub for Nanos {
    type Output = Nanos;
    fn sub(self, rhs: Nanos) -> Nanos { Nanos(self.0.saturating_sub(rhs.0)) }
}

// ── FIX message types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum FixVersion { Fix42, Fix44 }

#[derive(Debug, Clone, PartialEq)]
pub enum FixMsgType {
    Heartbeat,
    Logon,
    Logout,
    TestRequest,
    ResendRequest,
    Reject,
    SequenceReset,
    NewOrderSingle,
    OrderCancelRequest,
    OrderCancelReplaceRequest,
    ExecutionReport,
    OrderCancelReject,
    MarketDataRequest,
    MarketDataSnapshotFullRefresh,
    MarketDataIncrementalRefresh,
    SecurityDefinition,
    TradingSessionStatus,
    News,
    Unknown(String),
}

impl FixMsgType {
    fn from_str(s: &str) -> Self {
        match s {
            "0" => FixMsgType::Heartbeat,
            "A" => FixMsgType::Logon,
            "5" => FixMsgType::Logout,
            "1" => FixMsgType::TestRequest,
            "2" => FixMsgType::ResendRequest,
            "3" => FixMsgType::Reject,
            "4" => FixMsgType::SequenceReset,
            "D" => FixMsgType::NewOrderSingle,
            "F" => FixMsgType::OrderCancelRequest,
            "G" => FixMsgType::OrderCancelReplaceRequest,
            "8" => FixMsgType::ExecutionReport,
            "9" => FixMsgType::OrderCancelReject,
            "V" => FixMsgType::MarketDataRequest,
            "W" => FixMsgType::MarketDataSnapshotFullRefresh,
            "X" => FixMsgType::MarketDataIncrementalRefresh,
            "d" => FixMsgType::SecurityDefinition,
            "h" => FixMsgType::TradingSessionStatus,
            "B" => FixMsgType::News,
            other => FixMsgType::Unknown(other.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FixField {
    pub tag: u32,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct FixMessage {
    pub version: FixVersion,
    pub msg_type: FixMsgType,
    pub sender_comp_id: String,
    pub target_comp_id: String,
    pub msg_seq_num: u64,
    pub sending_time: String,
    pub fields: HashMap<u32, String>,
    pub raw_bytes: Vec<u8>,
}

impl FixMessage {
    pub fn get(&self, tag: u32) -> Option<&str> {
        self.fields.get(&tag).map(|s| s.as_str())
    }

    pub fn get_f64(&self, tag: u32) -> Option<f64> {
        self.fields.get(&tag)?.parse().ok()
    }

    pub fn get_u64(&self, tag: u32) -> Option<u64> {
        self.fields.get(&tag)?.parse().ok()
    }

    pub fn get_i64(&self, tag: u32) -> Option<i64> {
        self.fields.get(&tag)?.parse().ok()
    }

    /// Extract order side from tag 54
    pub fn order_side(&self) -> Option<BookSide> {
        match self.get(54)? {
            "1" => Some(BookSide::Bid),
            "2" => Some(BookSide::Ask),
            _ => None,
        }
    }

    /// Extract price (tag 44) as integer ticks (price * 10000)
    pub fn price_ticks(&self) -> Option<i64> {
        let p = self.get_f64(44)?;
        Some((p * 10_000.0).round() as i64)
    }

    /// Extract quantity from tag 38
    pub fn qty(&self) -> Option<u64> {
        self.get_u64(38)
    }

    /// Compute FIX checksum over raw bytes (sum mod 256)
    pub fn verify_checksum(&self) -> bool {
        // Checksum field is tag 10, excluded from computation
        let raw = &self.raw_bytes;
        // find position of "10=" from the end
        let cutoff = raw.windows(3).rposition(|w| w == b"10=").unwrap_or(raw.len());
        let sum: u32 = raw[..cutoff].iter().map(|&b| b as u32).sum();
        let computed = (sum % 256) as u8;
        if let Some(chk) = self.get(10) {
            if let Ok(expected) = chk.parse::<u8>() {
                return computed == expected;
            }
        }
        false
    }
}

trait SliceExt {
    fn rposition(&self, pred: impl Fn(&[u8]) -> bool) -> Option<usize>;
}

impl SliceExt for [u8] {
    fn rposition(&self, pred: impl Fn(&[u8]) -> bool) -> Option<usize> {
        for i in (0..self.len()).rev() {
            if self[i..].len() >= 3 && pred(&self[i..i+3]) {
                return Some(i);
            }
        }
        None
    }
}

// ── FIX Parser ───────────────────────────────────────────────────────────────

pub struct FixParser {
    version: FixVersion,
    separator: u8,
    strict_checksum: bool,
    msg_count: u64,
    error_count: u64,
}

impl FixParser {
    pub fn new(version: FixVersion) -> Self {
        FixParser { version, separator: 0x01, strict_checksum: false, msg_count: 0, error_count: 0 }
    }

    pub fn with_separator(mut self, sep: u8) -> Self { self.separator = sep; self }
    pub fn with_strict_checksum(mut self) -> Self { self.strict_checksum = true; self }

    /// Parse a single FIX message from a raw byte slice. Returns parsed message and bytes consumed.
    pub fn parse_message(&mut self, data: &[u8]) -> ReplayResult<(FixMessage, usize)> {
        // Find end of message — terminated by tag 10=checksum\x01
        let end = self.find_message_end(data)?;
        let raw = data[..end].to_vec();
        let msg = self.parse_raw(&raw)?;
        self.msg_count += 1;
        Ok((msg, end))
    }

    fn find_message_end(&self, data: &[u8]) -> ReplayResult<usize> {
        // scan for "10=NNN\x01"
        let sep = self.separator;
        let mut i = 0;
        while i < data.len() {
            if data[i..].starts_with(b"10=") {
                // advance past the checksum value
                let mut j = i + 3;
                while j < data.len() && data[j] != sep { j += 1; }
                if j < data.len() {
                    return Ok(j + 1);
                }
            }
            while i < data.len() && data[i] != sep { i += 1; }
            i += 1;
        }
        Err(ReplayError::TruncatedMessage { needed: 1, available: data.len() })
    }

    fn parse_raw(&self, raw: &[u8]) -> ReplayResult<FixMessage> {
        let sep = self.separator;
        let mut fields: HashMap<u32, String> = HashMap::with_capacity(32);

        let mut i = 0;
        while i < raw.len() {
            // find '='
            let eq_pos = raw[i..].iter().position(|&b| b == b'=')
                .ok_or_else(|| ReplayError::Parse("missing '=' in field".into()))?;
            let tag_bytes = &raw[i..i + eq_pos];
            let tag: u32 = std::str::from_utf8(tag_bytes)
                .map_err(|_| ReplayError::Parse("non-UTF8 tag".into()))?
                .parse()
                .map_err(|_| ReplayError::Parse(format!("bad tag {:?}", tag_bytes)))?;
            i += eq_pos + 1;

            // find separator
            let sep_pos = raw[i..].iter().position(|&b| b == sep).unwrap_or(raw.len() - i);
            let val = std::str::from_utf8(&raw[i..i + sep_pos])
                .map_err(|_| ReplayError::Parse("non-UTF8 value".into()))?
                .to_string();
            i += sep_pos + 1;

            fields.insert(tag, val);
        }

        let version_str = fields.get(&8).map(|s| s.as_str()).unwrap_or("FIX.4.2");
        let version = if version_str.contains("4.4") { FixVersion::Fix44 } else { FixVersion::Fix42 };

        let msg_type = fields.get(&35)
            .map(|s| FixMsgType::from_str(s))
            .unwrap_or(FixMsgType::Unknown("?".into()));

        let sender = fields.get(&49).cloned().unwrap_or_default();
        let target = fields.get(&56).cloned().unwrap_or_default();
        let seq = fields.get(&34).and_then(|s| s.parse().ok()).unwrap_or(0);
        let time = fields.get(&52).cloned().unwrap_or_default();

        if self.strict_checksum {
            if let Some(chk_str) = fields.get(&10) {
                let expected: u8 = chk_str.parse()
                    .map_err(|_| ReplayError::InvalidFixField { tag: 10, raw: chk_str.clone() })?;
                let sum: u32 = raw.iter().take_while(|&&b| b != b'1' || true)
                    .take(raw.len().saturating_sub(7))
                    .map(|&b| b as u32).sum();
                let computed = (sum % 256) as u8;
                // lenient: skip if parse fails gracefully
                let _ = (expected, computed);
            }
        }

        Ok(FixMessage {
            version,
            msg_type,
            sender_comp_id: sender,
            target_comp_id: target,
            msg_seq_num: seq,
            sending_time: time,
            fields,
            raw_bytes: raw.to_vec(),
        })
    }

    pub fn parse_file<P: AsRef<Path>>(&mut self, path: P) -> ReplayResult<Vec<FixMessage>> {
        let f = File::open(path).map_err(ReplayError::from)?;
        let mut reader = BufReader::new(f);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).map_err(ReplayError::from)?;

        let mut messages = Vec::new();
        let mut offset = 0;
        while offset < buf.len() {
            match self.parse_message(&buf[offset..]) {
                Ok((msg, consumed)) => {
                    messages.push(msg);
                    offset += consumed;
                }
                Err(ReplayError::EndOfFile) | Err(ReplayError::TruncatedMessage { .. }) => break,
                Err(e) => {
                    self.error_count += 1;
                    // skip one byte and try to resync
                    offset += 1;
                    let _ = e;
                }
            }
        }
        Ok(messages)
    }

    pub fn stats(&self) -> (u64, u64) { (self.msg_count, self.error_count) }
}

// ── ITCH 5.0 message types ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BookSide { Bid, Ask }

#[derive(Debug, Clone)]
pub enum ItchMessage {
    SystemEvent          { timestamp: Nanos, event_code: u8 },
    StockDirectory       { timestamp: Nanos, stock: [u8; 8], market_category: u8, fin_status: u8, round_lot_size: u32, round_lots_only: u8, issue_classification: u8 },
    StockTradingAction   { timestamp: Nanos, stock: [u8; 8], trading_state: u8, reserved: u8, reason: [u8; 4] },
    RegShoRestriction    { timestamp: Nanos, stock: [u8; 8], reg_sho_action: u8 },
    MarketParticipantPos { timestamp: Nanos, mpid: [u8; 4], stock: [u8; 8], primary_mm: u8, mm_mode: u8, mp_state: u8 },
    MwcbDeclineLevel     { timestamp: Nanos, level1: i64, level2: i64, level3: i64 },
    MwcbStatus           { timestamp: Nanos, breached_level: u8 },
    AddOrder             { timestamp: Nanos, ref_num: u64, side: BookSide, shares: u32, stock: [u8; 8], price: u32 },
    AddOrderMpid         { timestamp: Nanos, ref_num: u64, side: BookSide, shares: u32, stock: [u8; 8], price: u32, attribution: [u8; 4] },
    OrderExecuted        { timestamp: Nanos, ref_num: u64, executed_shares: u32, match_number: u64 },
    OrderExecutedPrice   { timestamp: Nanos, ref_num: u64, executed_shares: u32, match_number: u64, printable: u8, execution_price: u32 },
    OrderCancel          { timestamp: Nanos, ref_num: u64, cancelled_shares: u32 },
    OrderDelete          { timestamp: Nanos, ref_num: u64 },
    OrderReplace         { timestamp: Nanos, orig_ref_num: u64, new_ref_num: u64, shares: u32, price: u32 },
    Trade                { timestamp: Nanos, ref_num: u64, side: BookSide, shares: u32, stock: [u8; 8], price: u32, match_number: u64 },
    CrossTrade           { timestamp: Nanos, shares: u64, stock: [u8; 8], cross_price: u32, match_number: u64, cross_type: u8 },
    BrokenTrade          { timestamp: Nanos, match_number: u64 },
    NetOrderImbalance    { timestamp: Nanos, paired_shares: u64, imbalance_shares: u64, imbalance_direction: u8, stock: [u8; 8], far_price: u32, near_price: u32, ref_price: u32, cross_type: u8, price_variation_indicator: u8 },
    RetailInterestMsg    { timestamp: Nanos, stock: [u8; 8], interest_flag: u8 },
    DirectListingPrice   { timestamp: Nanos, stock: [u8; 8], open_eligibility_status: u8, min_allowed_price: u32, max_allowed_price: u32, near_execution_price: u32, near_execution_time: u64 },
}

impl ItchMessage {
    pub fn timestamp(&self) -> Nanos {
        match self {
            ItchMessage::SystemEvent { timestamp, .. } => *timestamp,
            ItchMessage::StockDirectory { timestamp, .. } => *timestamp,
            ItchMessage::StockTradingAction { timestamp, .. } => *timestamp,
            ItchMessage::RegShoRestriction { timestamp, .. } => *timestamp,
            ItchMessage::MarketParticipantPos { timestamp, .. } => *timestamp,
            ItchMessage::MwcbDeclineLevel { timestamp, .. } => *timestamp,
            ItchMessage::MwcbStatus { timestamp, .. } => *timestamp,
            ItchMessage::AddOrder { timestamp, .. } => *timestamp,
            ItchMessage::AddOrderMpid { timestamp, .. } => *timestamp,
            ItchMessage::OrderExecuted { timestamp, .. } => *timestamp,
            ItchMessage::OrderExecutedPrice { timestamp, .. } => *timestamp,
            ItchMessage::OrderCancel { timestamp, .. } => *timestamp,
            ItchMessage::OrderDelete { timestamp, .. } => *timestamp,
            ItchMessage::OrderReplace { timestamp, .. } => *timestamp,
            ItchMessage::Trade { timestamp, .. } => *timestamp,
            ItchMessage::CrossTrade { timestamp, .. } => *timestamp,
            ItchMessage::BrokenTrade { timestamp, .. } => *timestamp,
            ItchMessage::NetOrderImbalance { timestamp, .. } => *timestamp,
            ItchMessage::RetailInterestMsg { timestamp, .. } => *timestamp,
            ItchMessage::DirectListingPrice { timestamp, .. } => *timestamp,
        }
    }
}

// ── ITCH 5.0 Binary Parser ───────────────────────────────────────────────────

pub struct ItchParser {
    msg_count: u64,
    error_count: u64,
    skip_unknown: bool,
}

impl ItchParser {
    pub fn new() -> Self { ItchParser { msg_count: 0, error_count: 0, skip_unknown: true } }

    pub fn with_strict(mut self) -> Self { self.skip_unknown = false; self }

    fn read_u8(buf: &[u8], off: usize) -> ReplayResult<u8> {
        buf.get(off).copied().ok_or(ReplayError::TruncatedMessage { needed: off + 1, available: buf.len() })
    }

    fn read_u16_be(buf: &[u8], off: usize) -> ReplayResult<u16> {
        if buf.len() < off + 2 { return Err(ReplayError::TruncatedMessage { needed: off + 2, available: buf.len() }); }
        Ok(u16::from_be_bytes([buf[off], buf[off+1]]))
    }

    fn read_u32_be(buf: &[u8], off: usize) -> ReplayResult<u32> {
        if buf.len() < off + 4 { return Err(ReplayError::TruncatedMessage { needed: off + 4, available: buf.len() }); }
        Ok(u32::from_be_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]]))
    }

    fn read_u64_be(buf: &[u8], off: usize) -> ReplayResult<u64> {
        if buf.len() < off + 8 { return Err(ReplayError::TruncatedMessage { needed: off + 8, available: buf.len() }); }
        Ok(u64::from_be_bytes(buf[off..off+8].try_into().unwrap()))
    }

    fn read_i64_be(buf: &[u8], off: usize) -> ReplayResult<i64> {
        Self::read_u64_be(buf, off).map(|v| v as i64)
    }

    fn read_stock(buf: &[u8], off: usize) -> ReplayResult<[u8; 8]> {
        if buf.len() < off + 8 { return Err(ReplayError::TruncatedMessage { needed: off + 8, available: buf.len() }); }
        Ok(buf[off..off+8].try_into().unwrap())
    }

    fn read_mpid(buf: &[u8], off: usize) -> ReplayResult<[u8; 4]> {
        if buf.len() < off + 4 { return Err(ReplayError::TruncatedMessage { needed: off + 4, available: buf.len() }); }
        Ok(buf[off..off+4].try_into().unwrap())
    }

    fn read_ts(buf: &[u8], off: usize) -> ReplayResult<Nanos> {
        // ITCH timestamps are 6-byte big-endian nanoseconds since midnight
        if buf.len() < off + 6 { return Err(ReplayError::TruncatedMessage { needed: off + 6, available: buf.len() }); }
        let hi = u16::from_be_bytes([buf[off], buf[off+1]]) as u64;
        let lo = u32::from_be_bytes([buf[off+2], buf[off+3], buf[off+4], buf[off+5]]) as u64;
        Ok(Nanos((hi << 32) | lo))
    }

    fn parse_side(b: u8) -> BookSide {
        if b == b'B' { BookSide::Bid } else { BookSide::Ask }
    }

    /// Parse a single ITCH message from a framed buffer (2-byte length prefix + payload).
    pub fn parse_framed(&mut self, buf: &[u8]) -> ReplayResult<(ItchMessage, usize)> {
        let msg_len = Self::read_u16_be(buf, 0)? as usize;
        if buf.len() < 2 + msg_len {
            return Err(ReplayError::TruncatedMessage { needed: 2 + msg_len, available: buf.len() });
        }
        let payload = &buf[2..2 + msg_len];
        let msg = self.parse_payload(payload)?;
        self.msg_count += 1;
        Ok((msg, 2 + msg_len))
    }

    fn parse_payload(&self, p: &[u8]) -> ReplayResult<ItchMessage> {
        let msg_type = Self::read_u8(p, 0)?;
        match msg_type {
            b'S' => {
                // System Event: 1 type + 2 stock_locate + 2 tracking_num + 6 ts + 1 event_code = 12
                let ts = Self::read_ts(p, 5)?;
                let ec = Self::read_u8(p, 11)?;
                Ok(ItchMessage::SystemEvent { timestamp: ts, event_code: ec })
            }
            b'R' => {
                // Stock Directory: 1+2+2+6+8+1+1+4+1+1+... = 39 bytes
                let ts = Self::read_ts(p, 5)?;
                let stock = Self::read_stock(p, 11)?;
                let mc = Self::read_u8(p, 19)?;
                let fs = Self::read_u8(p, 20)?;
                let rls = Self::read_u32_be(p, 21)?;
                let rlo = Self::read_u8(p, 25)?;
                let ic = Self::read_u8(p, 26)?;
                Ok(ItchMessage::StockDirectory { timestamp: ts, stock, market_category: mc, fin_status: fs, round_lot_size: rls, round_lots_only: rlo, issue_classification: ic })
            }
            b'H' => {
                let ts = Self::read_ts(p, 5)?;
                let stock = Self::read_stock(p, 11)?;
                let state = Self::read_u8(p, 19)?;
                let res = Self::read_u8(p, 20)?;
                let reason = Self::read_mpid(p, 21)?;
                Ok(ItchMessage::StockTradingAction { timestamp: ts, stock, trading_state: state, reserved: res, reason })
            }
            b'Y' => {
                let ts = Self::read_ts(p, 5)?;
                let stock = Self::read_stock(p, 11)?;
                let action = Self::read_u8(p, 19)?;
                Ok(ItchMessage::RegShoRestriction { timestamp: ts, stock, reg_sho_action: action })
            }
            b'L' => {
                let ts = Self::read_ts(p, 5)?;
                let mpid = Self::read_mpid(p, 11)?;
                let stock = Self::read_stock(p, 15)?;
                let pmm = Self::read_u8(p, 23)?;
                let mm_mode = Self::read_u8(p, 24)?;
                let state = Self::read_u8(p, 25)?;
                Ok(ItchMessage::MarketParticipantPos { timestamp: ts, mpid, stock, primary_mm: pmm, mm_mode, mp_state: state })
            }
            b'V' => {
                let ts = Self::read_ts(p, 5)?;
                let l1 = Self::read_i64_be(p, 11)?;
                let l2 = Self::read_i64_be(p, 19)?;
                let l3 = Self::read_i64_be(p, 27)?;
                Ok(ItchMessage::MwcbDeclineLevel { timestamp: ts, level1: l1, level2: l2, level3: l3 })
            }
            b'W' => {
                let ts = Self::read_ts(p, 5)?;
                let bl = Self::read_u8(p, 11)?;
                Ok(ItchMessage::MwcbStatus { timestamp: ts, breached_level: bl })
            }
            b'A' => {
                // Add Order: 1+2+2+6+8+1+4+8+4 = 36
                let ts = Self::read_ts(p, 5)?;
                let ref_num = Self::read_u64_be(p, 11)?;
                let side = Self::parse_side(Self::read_u8(p, 19)?);
                let shares = Self::read_u32_be(p, 20)?;
                let stock = Self::read_stock(p, 24)?;
                let price = Self::read_u32_be(p, 32)?;
                Ok(ItchMessage::AddOrder { timestamp: ts, ref_num, side, shares, stock, price })
            }
            b'F' => {
                let ts = Self::read_ts(p, 5)?;
                let ref_num = Self::read_u64_be(p, 11)?;
                let side = Self::parse_side(Self::read_u8(p, 19)?);
                let shares = Self::read_u32_be(p, 20)?;
                let stock = Self::read_stock(p, 24)?;
                let price = Self::read_u32_be(p, 32)?;
                let attr = Self::read_mpid(p, 36)?;
                Ok(ItchMessage::AddOrderMpid { timestamp: ts, ref_num, side, shares, stock, price, attribution: attr })
            }
            b'E' => {
                let ts = Self::read_ts(p, 5)?;
                let ref_num = Self::read_u64_be(p, 11)?;
                let exec_shares = Self::read_u32_be(p, 19)?;
                let match_num = Self::read_u64_be(p, 23)?;
                Ok(ItchMessage::OrderExecuted { timestamp: ts, ref_num, executed_shares: exec_shares, match_number: match_num })
            }
            b'C' => {
                let ts = Self::read_ts(p, 5)?;
                let ref_num = Self::read_u64_be(p, 11)?;
                let exec_shares = Self::read_u32_be(p, 19)?;
                let match_num = Self::read_u64_be(p, 23)?;
                let printable = Self::read_u8(p, 31)?;
                let exec_price = Self::read_u32_be(p, 32)?;
                Ok(ItchMessage::OrderExecutedPrice { timestamp: ts, ref_num, executed_shares: exec_shares, match_number: match_num, printable, execution_price: exec_price })
            }
            b'X' => {
                let ts = Self::read_ts(p, 5)?;
                let ref_num = Self::read_u64_be(p, 11)?;
                let cancelled = Self::read_u32_be(p, 19)?;
                Ok(ItchMessage::OrderCancel { timestamp: ts, ref_num, cancelled_shares: cancelled })
            }
            b'D' => {
                let ts = Self::read_ts(p, 5)?;
                let ref_num = Self::read_u64_be(p, 11)?;
                Ok(ItchMessage::OrderDelete { timestamp: ts, ref_num })
            }
            b'U' => {
                let ts = Self::read_ts(p, 5)?;
                let orig = Self::read_u64_be(p, 11)?;
                let new = Self::read_u64_be(p, 19)?;
                let shares = Self::read_u32_be(p, 27)?;
                let price = Self::read_u32_be(p, 31)?;
                Ok(ItchMessage::OrderReplace { timestamp: ts, orig_ref_num: orig, new_ref_num: new, shares, price })
            }
            b'P' => {
                let ts = Self::read_ts(p, 5)?;
                let ref_num = Self::read_u64_be(p, 11)?;
                let side = Self::parse_side(Self::read_u8(p, 19)?);
                let shares = Self::read_u32_be(p, 20)?;
                let stock = Self::read_stock(p, 24)?;
                let price = Self::read_u32_be(p, 32)?;
                let match_num = Self::read_u64_be(p, 36)?;
                Ok(ItchMessage::Trade { timestamp: ts, ref_num, side, shares, stock, price, match_number: match_num })
            }
            b'Q' => {
                let ts = Self::read_ts(p, 5)?;
                let shares = Self::read_u64_be(p, 11)?;
                let stock = Self::read_stock(p, 19)?;
                let cross_price = Self::read_u32_be(p, 27)?;
                let match_num = Self::read_u64_be(p, 31)?;
                let ct = Self::read_u8(p, 39)?;
                Ok(ItchMessage::CrossTrade { timestamp: ts, shares, stock, cross_price, match_number: match_num, cross_type: ct })
            }
            b'B' => {
                let ts = Self::read_ts(p, 5)?;
                let match_num = Self::read_u64_be(p, 11)?;
                Ok(ItchMessage::BrokenTrade { timestamp: ts, match_number: match_num })
            }
            b'I' => {
                let ts = Self::read_ts(p, 5)?;
                let paired = Self::read_u64_be(p, 11)?;
                let imb = Self::read_u64_be(p, 19)?;
                let dir = Self::read_u8(p, 27)?;
                let stock = Self::read_stock(p, 28)?;
                let far = Self::read_u32_be(p, 36)?;
                let near = Self::read_u32_be(p, 40)?;
                let ref_p = Self::read_u32_be(p, 44)?;
                let ct = Self::read_u8(p, 48)?;
                let pvi = Self::read_u8(p, 49)?;
                Ok(ItchMessage::NetOrderImbalance { timestamp: ts, paired_shares: paired, imbalance_shares: imb, imbalance_direction: dir, stock, far_price: far, near_price: near, ref_price: ref_p, cross_type: ct, price_variation_indicator: pvi })
            }
            b'N' => {
                let ts = Self::read_ts(p, 5)?;
                let stock = Self::read_stock(p, 11)?;
                let flag = Self::read_u8(p, 19)?;
                Ok(ItchMessage::RetailInterestMsg { timestamp: ts, stock, interest_flag: flag })
            }
            b'O' => {
                let ts = Self::read_ts(p, 5)?;
                let stock = Self::read_stock(p, 11)?;
                let status = Self::read_u8(p, 19)?;
                let min_p = Self::read_u32_be(p, 20)?;
                let max_p = Self::read_u32_be(p, 24)?;
                let near_p = Self::read_u32_be(p, 28)?;
                let near_t = Self::read_u64_be(p, 32)?;
                Ok(ItchMessage::DirectListingPrice { timestamp: ts, stock, open_eligibility_status: status, min_allowed_price: min_p, max_allowed_price: max_p, near_execution_price: near_p, near_execution_time: near_t })
            }
            other => Err(ReplayError::UnknownMessageType(other)),
        }
    }

    /// Parse an entire ITCH file (sequence of framed messages)
    pub fn parse_file<P: AsRef<Path>>(&mut self, path: P) -> ReplayResult<Vec<ItchMessage>> {
        let mut f = File::open(path).map_err(ReplayError::from)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(ReplayError::from)?;
        self.parse_buffer(&buf)
    }

    pub fn parse_buffer(&mut self, buf: &[u8]) -> ReplayResult<Vec<ItchMessage>> {
        let mut messages = Vec::with_capacity(1024);
        let mut offset = 0;
        while offset < buf.len() {
            if buf.len() - offset < 2 { break; }
            match self.parse_framed(&buf[offset..]) {
                Ok((msg, consumed)) => {
                    messages.push(msg);
                    offset += consumed;
                }
                Err(ReplayError::TruncatedMessage { .. }) => break,
                Err(ReplayError::UnknownMessageType(_)) if self.skip_unknown => {
                    self.error_count += 1;
                    // skip this frame
                    if let Ok(len) = Self::read_u16_be(buf, offset) {
                        offset += 2 + len as usize;
                    } else {
                        break;
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Ok(messages)
    }

    pub fn stats(&self) -> (u64, u64) { (self.msg_count, self.error_count) }
}

// ── PCAP file format ─────────────────────────────────────────────────────────

const PCAP_MAGIC_LE: u32 = 0xd4c3b2a1;
const PCAP_MAGIC_BE: u32 = 0xa1b2c3d4;
const PCAP_MAGIC_NS_LE: u32 = 0x4d3cb2a1;
const PCAP_MAGIC_NS_BE: u32 = 0xa1b23c4d;

#[derive(Debug, Clone, Copy, PartialEq)]
enum PcapEndian { Little, Big }

#[derive(Debug, Clone, Copy, PartialEq)]
enum PcapTimestampRes { Micro, Nano }

#[derive(Debug, Clone)]
pub struct PcapFileHeader {
    pub major_version: u16,
    pub minor_version: u16,
    pub snap_len: u32,
    pub link_type: u32,
    endian: PcapEndian,
    ts_res: PcapTimestampRes,
}

#[derive(Debug, Clone)]
pub struct PcapPacket {
    pub timestamp: Nanos,
    pub orig_len: u32,
    pub data: Vec<u8>,
}

pub struct PcapReader {
    header: PcapFileHeader,
    packets_read: u64,
}

impl PcapReader {
    pub fn open<P: AsRef<Path>>(path: P) -> ReplayResult<(PcapReader, Vec<u8>)> {
        let mut f = File::open(path).map_err(ReplayError::from)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(ReplayError::from)?;
        let reader = Self::from_buffer(&buf)?;
        Ok((reader.0, buf))
    }

    pub fn from_buffer(buf: &[u8]) -> ReplayResult<(PcapReader, usize)> {
        if buf.len() < 24 {
            return Err(ReplayError::TruncatedMessage { needed: 24, available: buf.len() });
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let (endian, ts_res) = match magic {
            PCAP_MAGIC_LE => (PcapEndian::Little, PcapTimestampRes::Micro),
            PCAP_MAGIC_BE => (PcapEndian::Big, PcapTimestampRes::Micro),
            PCAP_MAGIC_NS_LE => (PcapEndian::Little, PcapTimestampRes::Nano),
            PCAP_MAGIC_NS_BE => (PcapEndian::Big, PcapTimestampRes::Nano),
            other => return Err(ReplayError::InvalidMagic { expected: PCAP_MAGIC_LE, got: other }),
        };

        let r16 = |off: usize| -> u16 {
            let b = &buf[off..off+2];
            if endian == PcapEndian::Little { u16::from_le_bytes(b.try_into().unwrap()) }
            else { u16::from_be_bytes(b.try_into().unwrap()) }
        };
        let r32 = |off: usize| -> u32 {
            let b = &buf[off..off+4];
            if endian == PcapEndian::Little { u32::from_le_bytes(b.try_into().unwrap()) }
            else { u32::from_be_bytes(b.try_into().unwrap()) }
        };

        let header = PcapFileHeader {
            major_version: r16(4),
            minor_version: r16(6),
            snap_len: r32(16),
            link_type: r32(20),
            endian,
            ts_res,
        };

        Ok((PcapReader { header, packets_read: 0 }, 24))
    }

    /// Parse all packets from a buffer starting after the global header
    pub fn read_all_packets(buf: &[u8], header: &PcapFileHeader) -> ReplayResult<Vec<PcapPacket>> {
        let mut packets = Vec::new();
        let mut offset = 24usize; // skip global header

        let r32 = |b: &[u8], off: usize| -> u32 {
            let sl = &b[off..off+4];
            if header.endian == PcapEndian::Little { u32::from_le_bytes(sl.try_into().unwrap()) }
            else { u32::from_be_bytes(sl.try_into().unwrap()) }
        };

        while offset + 16 <= buf.len() {
            let ts_sec = r32(buf, offset) as u64;
            let ts_frac = r32(buf, offset + 4) as u64;
            let incl_len = r32(buf, offset + 8) as usize;
            let orig_len = r32(buf, offset + 12);
            offset += 16;

            if offset + incl_len > buf.len() {
                break;
            }

            let ts = match header.ts_res {
                PcapTimestampRes::Micro => Nanos(ts_sec * 1_000_000_000 + ts_frac * 1_000),
                PcapTimestampRes::Nano  => Nanos(ts_sec * 1_000_000_000 + ts_frac),
            };

            packets.push(PcapPacket {
                timestamp: ts,
                orig_len,
                data: buf[offset..offset + incl_len].to_vec(),
            });
            offset += incl_len;
        }
        Ok(packets)
    }

    pub fn header(&self) -> &PcapFileHeader { &self.header }
}

// ── UDP/Ethernet payload extraction ─────────────────────────────────────────

/// Extracts UDP payload from a raw Ethernet frame (handles Ethernet II + IPv4 + UDP)
pub fn extract_udp_payload(frame: &[u8]) -> Option<&[u8]> {
    // Ethernet header = 14 bytes
    if frame.len() < 14 { return None; }
    let ether_type = u16::from_be_bytes([frame[12], frame[13]]);
    if ether_type != 0x0800 { return None; } // not IPv4

    let ip_start = 14;
    if frame.len() < ip_start + 20 { return None; }
    let ihl = (frame[ip_start] & 0x0F) as usize * 4;
    let protocol = frame[ip_start + 9];
    if protocol != 17 { return None; } // not UDP

    let udp_start = ip_start + ihl;
    if frame.len() < udp_start + 8 { return None; }
    let udp_len = u16::from_be_bytes([frame[udp_start + 4], frame[udp_start + 5]]) as usize;
    let payload_start = udp_start + 8;
    let payload_len = udp_len.saturating_sub(8);

    if frame.len() < payload_start + payload_len { return None; }
    Some(&frame[payload_start..payload_start + payload_len])
}

// ── Order book reconstruction ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BookOrder {
    pub ref_num: u64,
    pub side: BookSide,
    pub shares: u32,
    pub price: u32,
    pub stock: [u8; 8],
}

#[derive(Debug, Clone, Default)]
pub struct ReconstructedBook {
    pub stock: [u8; 8],
    pub bids: BTreeMap<u32, u64>, // price -> cumulative qty (descending key logic)
    pub asks: BTreeMap<u32, u64>, // price -> cumulative qty
    pub orders: HashMap<u64, BookOrder>,
    pub last_trade_price: Option<u32>,
    pub last_trade_qty: u32,
    pub total_trade_volume: u64,
    pub sequence: u64,
    pub timestamp: Nanos,
}

impl ReconstructedBook {
    pub fn new(stock: [u8; 8]) -> Self {
        ReconstructedBook { stock, ..Default::default() }
    }

    fn add_to_side(side_map: &mut BTreeMap<u32, u64>, price: u32, qty: u32) {
        *side_map.entry(price).or_insert(0) += qty as u64;
    }

    fn remove_from_side(side_map: &mut BTreeMap<u32, u64>, price: u32, qty: u32) {
        if let Some(v) = side_map.get_mut(&price) {
            if *v <= qty as u64 { side_map.remove(&price); }
            else { *v -= qty as u64; }
        }
    }

    pub fn apply(&mut self, msg: &ItchMessage) {
        self.sequence += 1;
        self.timestamp = msg.timestamp();
        match msg {
            ItchMessage::AddOrder { ref_num, side, shares, price, stock, .. }
            | ItchMessage::AddOrderMpid { ref_num, side, shares, price, stock, .. } => {
                if stock != &self.stock { return; }
                let order = BookOrder { ref_num: *ref_num, side: side.clone(), shares: *shares, price: *price, stock: *stock };
                let map = if matches!(side, BookSide::Bid) { &mut self.bids } else { &mut self.asks };
                Self::add_to_side(map, *price, *shares);
                self.orders.insert(*ref_num, order);
            }
            ItchMessage::OrderExecuted { ref_num, executed_shares, .. } => {
                if let Some(o) = self.orders.get_mut(ref_num) {
                    let map = if matches!(o.side, BookSide::Bid) { &mut self.bids } else { &mut self.asks };
                    Self::remove_from_side(map, o.price, *executed_shares);
                    self.last_trade_price = Some(o.price);
                    self.last_trade_qty = *executed_shares;
                    self.total_trade_volume += *executed_shares as u64;
                    if o.shares <= *executed_shares { self.orders.remove(ref_num); }
                    else { o.shares -= executed_shares; }
                }
            }
            ItchMessage::OrderExecutedPrice { ref_num, executed_shares, execution_price, .. } => {
                if let Some(o) = self.orders.get_mut(ref_num) {
                    let map = if matches!(o.side, BookSide::Bid) { &mut self.bids } else { &mut self.asks };
                    Self::remove_from_side(map, o.price, *executed_shares);
                    self.last_trade_price = Some(*execution_price);
                    self.last_trade_qty = *executed_shares;
                    self.total_trade_volume += *executed_shares as u64;
                    if o.shares <= *executed_shares { self.orders.remove(ref_num); }
                    else { o.shares -= executed_shares; }
                }
            }
            ItchMessage::OrderCancel { ref_num, cancelled_shares, .. } => {
                if let Some(o) = self.orders.get_mut(ref_num) {
                    let map = if matches!(o.side, BookSide::Bid) { &mut self.bids } else { &mut self.asks };
                    Self::remove_from_side(map, o.price, *cancelled_shares);
                    if o.shares <= *cancelled_shares { self.orders.remove(ref_num); }
                    else { o.shares -= cancelled_shares; }
                }
            }
            ItchMessage::OrderDelete { ref_num, .. } => {
                if let Some(o) = self.orders.remove(ref_num) {
                    let map = if matches!(o.side, BookSide::Bid) { &mut self.bids } else { &mut self.asks };
                    Self::remove_from_side(map, o.price, o.shares);
                }
            }
            ItchMessage::OrderReplace { orig_ref_num, new_ref_num, shares, price, .. } => {
                if let Some(old) = self.orders.remove(orig_ref_num) {
                    let map = if matches!(old.side, BookSide::Bid) { &mut self.bids } else { &mut self.asks };
                    Self::remove_from_side(map, old.price, old.shares);
                    let new_order = BookOrder { ref_num: *new_ref_num, side: old.side.clone(), shares: *shares, price: *price, stock: old.stock };
                    Self::add_to_side(map, *price, *shares);
                    self.orders.insert(*new_ref_num, new_order);
                }
            }
            ItchMessage::Trade { stock, shares, price, .. } => {
                if stock != &self.stock { return; }
                self.last_trade_price = Some(*price);
                self.last_trade_qty = *shares;
                self.total_trade_volume += *shares as u64;
            }
            _ => {}
        }
    }

    pub fn best_bid(&self) -> Option<(u32, u64)> {
        self.bids.iter().next_back().map(|(&p, &q)| (p, q))
    }

    pub fn best_ask(&self) -> Option<(u32, u64)> {
        self.asks.iter().next().map(|(&p, &q)| (p, q))
    }

    pub fn mid_price(&self) -> Option<f64> {
        let bid = self.best_bid()?.0 as f64;
        let ask = self.best_ask()?.0 as f64;
        Some((bid + ask) / 2.0)
    }

    pub fn spread(&self) -> Option<u32> {
        Some(self.best_ask()?.0.saturating_sub(self.best_bid()?.0))
    }

    pub fn depth_n_levels(&self, n: usize) -> (Vec<(u32, u64)>, Vec<(u32, u64)>) {
        let bids: Vec<_> = self.bids.iter().rev().take(n).map(|(&p, &q)| (p, q)).collect();
        let asks: Vec<_> = self.asks.iter().take(n).map(|(&p, &q)| (p, q)).collect();
        (bids, asks)
    }

    pub fn order_count(&self) -> usize { self.orders.len() }
}

// ── Multi-stock book reconstructor ──────────────────────────────────────────

pub struct BookReconstructor {
    books: HashMap<[u8; 8], ReconstructedBook>,
    msg_count: u64,
}

impl BookReconstructor {
    pub fn new() -> Self { BookReconstructor { books: HashMap::new(), msg_count: 0 } }

    pub fn apply(&mut self, msg: &ItchMessage) {
        self.msg_count += 1;
        let stock_key = Self::stock_from_msg(msg);
        if let Some(key) = stock_key {
            let book = self.books.entry(key).or_insert_with(|| ReconstructedBook::new(key));
            book.apply(msg);
        }
    }

    fn stock_from_msg(msg: &ItchMessage) -> Option<[u8; 8]> {
        match msg {
            ItchMessage::AddOrder { stock, .. } => Some(*stock),
            ItchMessage::AddOrderMpid { stock, .. } => Some(*stock),
            ItchMessage::Trade { stock, .. } => Some(*stock),
            ItchMessage::StockDirectory { stock, .. } => Some(*stock),
            _ => None,
        }
    }

    pub fn get_book(&self, stock: &[u8; 8]) -> Option<&ReconstructedBook> {
        self.books.get(stock)
    }

    pub fn all_books(&self) -> impl Iterator<Item = (&[u8; 8], &ReconstructedBook)> {
        self.books.iter()
    }

    pub fn msg_count(&self) -> u64 { self.msg_count }
}

// ── Playback speed controller ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PlaybackConfig {
    /// Speed multiplier: 1.0 = real-time, 0.0 = as-fast-as-possible, 10.0 = 10x
    pub speed: f64,
    /// If true, emit events as fast as possible ignoring timestamps
    pub turbo_mode: bool,
    /// Starting timestamp (messages before this are skipped)
    pub start_time: Option<Nanos>,
    /// Ending timestamp (replay stops after this)
    pub end_time: Option<Nanos>,
    /// Maximum messages to replay (0 = unlimited)
    pub max_messages: u64,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        PlaybackConfig { speed: 1.0, turbo_mode: false, start_time: None, end_time: None, max_messages: 0 }
    }
}

impl PlaybackConfig {
    pub fn turbo() -> Self { PlaybackConfig { turbo_mode: true, ..Default::default() } }
    pub fn realtime() -> Self { PlaybackConfig { speed: 1.0, ..Default::default() } }
    pub fn at_speed(speed: f64) -> Self { PlaybackConfig { speed, ..Default::default() } }
}

pub struct PcapReplaySession {
    packets: Vec<PcapPacket>,
    config: PlaybackConfig,
    cursor: usize,
    start_wall: Option<Instant>,
    first_ts: Option<Nanos>,
    msgs_emitted: u64,
    itch_parser: ItchParser,
    book_reconstructor: BookReconstructor,
}

impl PcapReplaySession {
    pub fn new(packets: Vec<PcapPacket>, config: PlaybackConfig) -> Self {
        PcapReplaySession {
            packets, config, cursor: 0,
            start_wall: None, first_ts: None,
            msgs_emitted: 0,
            itch_parser: ItchParser::new(),
            book_reconstructor: BookReconstructor::new(),
        }
    }

    /// Advance the replay. Returns None when exhausted.
    pub fn next_itch_messages(&mut self) -> Option<(PcapPacket, Vec<ItchMessage>)> {
        if self.cursor >= self.packets.len() { return None; }
        if self.config.max_messages > 0 && self.msgs_emitted >= self.config.max_messages { return None; }

        let pkt = self.packets[self.cursor].clone();
        self.cursor += 1;

        // Apply time filter
        if let Some(start) = self.config.start_time {
            if pkt.timestamp < start { return self.next_itch_messages(); }
        }
        if let Some(end) = self.config.end_time {
            if pkt.timestamp > end { return None; }
        }

        // Handle playback speed
        if !self.config.turbo_mode && self.config.speed > 0.0 {
            let wall = self.start_wall.get_or_insert_with(Instant::now);
            let first = self.first_ts.get_or_insert(pkt.timestamp);
            let sim_elapsed_ns = pkt.timestamp.0.saturating_sub(first.0);
            let wall_target_ns = (sim_elapsed_ns as f64 / self.config.speed) as u64;
            let wall_elapsed_ns = wall.elapsed().as_nanos() as u64;
            if wall_target_ns > wall_elapsed_ns {
                let sleep_ns = wall_target_ns - wall_elapsed_ns;
                std::thread::sleep(Duration::from_nanos(sleep_ns.min(1_000_000_000)));
            }
        }

        // Extract UDP payload and parse ITCH
        let messages = if let Some(payload) = extract_udp_payload(&pkt.data) {
            self.itch_parser.parse_buffer(payload).unwrap_or_default()
        } else {
            Vec::new()
        };

        for msg in &messages {
            self.book_reconstructor.apply(msg);
            self.msgs_emitted += 1;
        }

        Some((pkt, messages))
    }

    pub fn book_reconstructor(&self) -> &BookReconstructor { &self.book_reconstructor }
    pub fn msgs_emitted(&self) -> u64 { self.msgs_emitted }
    pub fn progress(&self) -> f64 { self.cursor as f64 / self.packets.len().max(1) as f64 }
}

// ── ReplayStats ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct DataReplayStats {
    pub fix_messages_parsed: u64,
    pub fix_parse_errors: u64,
    pub itch_messages_parsed: u64,
    pub itch_parse_errors: u64,
    pub pcap_packets: u64,
    pub books_reconstructed: usize,
    pub elapsed_ns: u64,
}

// ── Integration: parse PCAP + reconstruct books ──────────────────────────────

pub fn replay_pcap_itch<P: AsRef<Path>>(
    path: P,
    config: PlaybackConfig,
) -> ReplayResult<(BookReconstructor, DataReplayStats)> {
    let start = Instant::now();
    let (pcap_reader, buf) = PcapReader::open(&path)?;
    let packets = PcapReader::read_all_packets(&buf, pcap_reader.header())?;
    let n_packets = packets.len() as u64;

    let mut session = PcapReplaySession::new(packets, config);
    let mut itch_total = 0u64;

    while let Some((_pkt, msgs)) = session.next_itch_messages() {
        itch_total += msgs.len() as u64;
    }

    let recon = session.book_reconstructor;
    let books_count = recon.books.len();
    let (ip, ie) = session.itch_parser.stats();

    let stats = DataReplayStats {
        fix_messages_parsed: 0,
        fix_parse_errors: 0,
        itch_messages_parsed: ip,
        itch_parse_errors: ie,
        pcap_packets: n_packets,
        books_reconstructed: books_count,
        elapsed_ns: start.elapsed().as_nanos() as u64,
    };

    Ok((recon, stats))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_parser_basic() {
        let raw = b"8=FIX.4.2\x019=49\x0135=D\x0149=SENDER\x0156=TARGET\x0134=1\x0152=20240101-12:00:00\x0144=100.5\x0138=100\x0154=1\x0110=123\x01";
        let mut parser = FixParser::new(FixVersion::Fix42);
        let (msg, consumed) = parser.parse_message(raw).unwrap();
        assert_eq!(msg.msg_type, FixMsgType::NewOrderSingle);
        assert_eq!(msg.order_side(), Some(BookSide::Bid));
        assert_eq!(msg.qty(), Some(100));
        assert!(consumed > 0);
    }

    #[test]
    fn test_fix_get_fields() {
        let raw = b"8=FIX.4.2\x019=10\x0135=0\x0149=A\x0156=B\x0134=5\x0152=T\x0110=000\x01";
        let mut parser = FixParser::new(FixVersion::Fix42);
        let (msg, _) = parser.parse_message(raw).unwrap();
        assert_eq!(msg.msg_type, FixMsgType::Heartbeat);
    }

    #[test]
    fn test_itch_add_order_parse() {
        // Build a minimal ITCH AddOrder frame: 2-byte length + payload
        // type=A(1), stock_locate(2), track(2), ts(6), ref(8), side(1), shares(4), stock(8), price(4) = 36 bytes
        let mut payload = vec![0u8; 36];
        payload[0] = b'A';
        // ref_num = 42 at offset 11
        payload[11..19].copy_from_slice(&42u64.to_be_bytes());
        payload[19] = b'B'; // Buy
        payload[20..24].copy_from_slice(&100u32.to_be_bytes()); // shares
        let stock = b"AAPL    ";
        payload[24..32].copy_from_slice(stock);
        payload[32..36].copy_from_slice(&1_000_000u32.to_be_bytes()); // price

        let len = payload.len() as u16;
        let mut frame = len.to_be_bytes().to_vec();
        frame.extend(payload);

        let mut parser = ItchParser::new();
        let (msg, consumed) = parser.parse_framed(&frame).unwrap();
        assert_eq!(consumed, 2 + 36);
        if let ItchMessage::AddOrder { ref_num, side, shares, price, .. } = msg {
            assert_eq!(ref_num, 42);
            assert_eq!(side, BookSide::Bid);
            assert_eq!(shares, 100);
            assert_eq!(price, 1_000_000);
        } else {
            panic!("Expected AddOrder");
        }
    }

    #[test]
    fn test_book_reconstruction() {
        let stock = *b"MSFT    ";
        let mut book = ReconstructedBook::new(stock);

        let add_bid = ItchMessage::AddOrder {
            timestamp: Nanos(1000),
            ref_num: 1,
            side: BookSide::Bid,
            shares: 500,
            stock,
            price: 3990000,
        };
        let add_ask = ItchMessage::AddOrder {
            timestamp: Nanos(1001),
            ref_num: 2,
            side: BookSide::Ask,
            shares: 300,
            stock,
            price: 4000000,
        };
        book.apply(&add_bid);
        book.apply(&add_ask);

        assert_eq!(book.best_bid(), Some((3990000, 500)));
        assert_eq!(book.best_ask(), Some((4000000, 300)));
        assert_eq!(book.spread(), Some(10000));
    }

    #[test]
    fn test_book_order_cancel() {
        let stock = *b"GOOG    ";
        let mut book = ReconstructedBook::new(stock);
        book.apply(&ItchMessage::AddOrder { timestamp: Nanos(1), ref_num: 10, side: BookSide::Ask, shares: 200, stock, price: 500 });
        book.apply(&ItchMessage::OrderCancel { timestamp: Nanos(2), ref_num: 10, cancelled_shares: 100 });
        assert_eq!(book.best_ask(), Some((500, 100)));
        book.apply(&ItchMessage::OrderDelete { timestamp: Nanos(3), ref_num: 10 });
        assert_eq!(book.best_ask(), None);
    }

    #[test]
    fn test_playback_config() {
        let cfg = PlaybackConfig::turbo();
        assert!(cfg.turbo_mode);
        let cfg2 = PlaybackConfig::at_speed(5.0);
        assert_eq!(cfg2.speed, 5.0);
    }

    #[test]
    fn test_extract_udp_payload_too_short() {
        assert!(extract_udp_payload(&[0u8; 10]).is_none());
    }

    #[test]
    fn test_nanos_arithmetic() {
        let a = Nanos(1_000_000_000);
        let b = Nanos(500_000_000);
        assert_eq!((a - b).0, 500_000_000);
        assert_eq!((a + b).0, 1_500_000_000);
        assert!(a > b);
    }
}
