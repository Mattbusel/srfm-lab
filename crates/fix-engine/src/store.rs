use std::collections::{BTreeMap, HashMap};
use crate::message::FixMessage;

/// In-memory FIX message store for resend and gap detection.
/// Stores sent messages indexed by sequence number.
#[derive(Debug)]
pub struct MessageStore {
    /// Sent messages: seqnum -> message
    sent: BTreeMap<u32, FixMessage>,
    /// Received messages: seqnum -> message
    received: BTreeMap<u32, FixMessage>,
    /// Next expected outbound sequence number
    next_sender_seq: u32,
    /// Next expected inbound sequence number
    next_target_seq: u32,
    /// Session identifier (sender:target)
    session_id: String,
    /// Maximum messages to retain (0 = unlimited)
    max_retain: usize,
}

impl MessageStore {
    pub fn new(sender: &str, target: &str) -> Self {
        MessageStore {
            sent: BTreeMap::new(),
            received: BTreeMap::new(),
            next_sender_seq: 1,
            next_target_seq: 1,
            session_id: format!("{}:{}", sender, target),
            max_retain: 10_000,
        }
    }

    pub fn with_max_retain(mut self, max: usize) -> Self {
        self.max_retain = max;
        self
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Store an outbound message
    pub fn store_sent(&mut self, seq_num: u32, msg: FixMessage) {
        self.sent.insert(seq_num, msg);
        self.prune_sent();
    }

    /// Store an inbound message
    pub fn store_received(&mut self, seq_num: u32, msg: FixMessage) {
        self.received.insert(seq_num, msg);
        self.prune_received();
    }

    fn prune_sent(&mut self) {
        if self.max_retain > 0 && self.sent.len() > self.max_retain {
            while self.sent.len() > self.max_retain {
                if let Some((&k, _)) = self.sent.iter().next() {
                    self.sent.remove(&k);
                } else {
                    break;
                }
            }
        }
    }

    fn prune_received(&mut self) {
        if self.max_retain > 0 && self.received.len() > self.max_retain {
            while self.received.len() > self.max_retain {
                if let Some((&k, _)) = self.received.iter().next() {
                    self.received.remove(&k);
                } else {
                    break;
                }
            }
        }
    }

    pub fn get_sent(&self, seq_num: u32) -> Option<&FixMessage> {
        self.sent.get(&seq_num)
    }

    pub fn get_received(&self, seq_num: u32) -> Option<&FixMessage> {
        self.received.get(&seq_num)
    }

    /// Get a range of sent messages for resend [begin, end] (inclusive)
    /// If end is 0, returns all from begin onwards
    pub fn get_sent_range(&self, begin: u32, end: u32) -> Vec<(u32, &FixMessage)> {
        if end == 0 {
            self.sent.range(begin..)
                .map(|(&k, v)| (k, v))
                .collect()
        } else {
            self.sent.range(begin..=end)
                .map(|(&k, v)| (k, v))
                .collect()
        }
    }

    /// Detect sequence gaps in received messages
    pub fn inbound_gaps(&self, expected_next: u32) -> Vec<(u32, u32)> {
        let mut gaps = Vec::new();
        if self.received.is_empty() {
            return gaps;
        }

        let mut expected = expected_next;
        for (&seq, _) in &self.received {
            if seq > expected {
                gaps.push((expected, seq - 1));
            }
            expected = seq + 1;
        }
        gaps
    }

    pub fn next_sender_seq_num(&self) -> u32 {
        self.next_sender_seq
    }

    pub fn next_target_seq_num(&self) -> u32 {
        self.next_target_seq
    }

    pub fn increment_sender_seq(&mut self) -> u32 {
        let n = self.next_sender_seq;
        self.next_sender_seq += 1;
        n
    }

    pub fn increment_target_seq(&mut self) {
        self.next_target_seq += 1;
    }

    pub fn set_sender_seq(&mut self, seq: u32) {
        self.next_sender_seq = seq;
    }

    pub fn set_target_seq(&mut self, seq: u32) {
        self.next_target_seq = seq;
    }

    /// Reset all sequence numbers to 1 and clear stored messages
    pub fn reset(&mut self) {
        self.sent.clear();
        self.received.clear();
        self.next_sender_seq = 1;
        self.next_target_seq = 1;
    }

    pub fn sent_count(&self) -> usize {
        self.sent.len()
    }

    pub fn received_count(&self) -> usize {
        self.received.len()
    }

    /// Check if a specific outbound seq num is stored (for duplicate detection)
    pub fn has_sent(&self, seq_num: u32) -> bool {
        self.sent.contains_key(&seq_num)
    }

    /// Check if a specific inbound seq num is already stored (duplicate detection)
    pub fn has_received(&self, seq_num: u32) -> bool {
        self.received.contains_key(&seq_num)
    }

    /// Get the highest stored sender sequence number
    pub fn highest_sent_seq(&self) -> Option<u32> {
        self.sent.keys().next_back().copied()
    }

    /// Get the highest stored received sequence number
    pub fn highest_received_seq(&self) -> Option<u32> {
        self.received.keys().next_back().copied()
    }
}

/// Gap fill tracking for resend requests
#[derive(Debug, Clone)]
pub struct GapFillRecord {
    pub begin_seq: u32,
    pub end_seq: u32,
    pub requested_at: std::time::Instant,
    pub retry_count: u32,
}

impl GapFillRecord {
    pub fn new(begin_seq: u32, end_seq: u32) -> Self {
        GapFillRecord {
            begin_seq,
            end_seq,
            requested_at: std::time::Instant::now(),
            retry_count: 0,
        }
    }

    pub fn retry(&mut self) {
        self.retry_count += 1;
        self.requested_at = std::time::Instant::now();
    }

    pub fn is_expired(&self, timeout_secs: u64) -> bool {
        self.requested_at.elapsed().as_secs() > timeout_secs
    }
}

/// Tracks pending gap fill requests
#[derive(Debug)]
pub struct GapFillTracker {
    pending: HashMap<u32, GapFillRecord>, // keyed by begin_seq
    max_retries: u32,
    timeout_secs: u64,
}

impl GapFillTracker {
    pub fn new(max_retries: u32, timeout_secs: u64) -> Self {
        GapFillTracker {
            pending: HashMap::new(),
            max_retries,
            timeout_secs,
        }
    }

    pub fn add_request(&mut self, begin_seq: u32, end_seq: u32) {
        self.pending.insert(begin_seq, GapFillRecord::new(begin_seq, end_seq));
    }

    pub fn acknowledge(&mut self, seq_num: u32) {
        // Remove any pending records that cover this seq num
        self.pending.retain(|_, rec| {
            !(rec.begin_seq <= seq_num && seq_num <= rec.end_seq)
        });
    }

    pub fn get_expired_retries(&mut self) -> Vec<GapFillRecord> {
        let timeout = self.timeout_secs;
        let max_retries = self.max_retries;
        let mut expired = Vec::new();
        for (_, rec) in &mut self.pending {
            if rec.is_expired(timeout) && rec.retry_count < max_retries {
                expired.push(rec.clone());
                rec.retry();
            }
        }
        expired
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::FixMessage;
    use crate::types::MsgType;

    fn dummy_msg(seq: u32) -> FixMessage {
        FixMessage::with_header("FIX.4.2", MsgType::Heartbeat.as_str(), "S", "T", seq)
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut store = MessageStore::new("S", "T");
        store.store_sent(1, dummy_msg(1));
        store.store_sent(2, dummy_msg(2));
        assert_eq!(store.sent_count(), 2);
        assert!(store.get_sent(1).is_some());
        assert!(store.get_sent(3).is_none());
    }

    #[test]
    fn test_sent_range() {
        let mut store = MessageStore::new("S", "T");
        for i in 1..=10 {
            store.store_sent(i, dummy_msg(i));
        }
        let range = store.get_sent_range(3, 7);
        assert_eq!(range.len(), 5);
        assert_eq!(range[0].0, 3);
        assert_eq!(range[4].0, 7);
    }

    #[test]
    fn test_sent_range_open_end() {
        let mut store = MessageStore::new("S", "T");
        for i in 1..=5 {
            store.store_sent(i, dummy_msg(i));
        }
        let range = store.get_sent_range(3, 0);
        assert_eq!(range.len(), 3); // 3,4,5
    }

    #[test]
    fn test_gap_detection() {
        let mut store = MessageStore::new("S", "T");
        // Receive 1, 2, 4, 5 (gap at 3)
        store.store_received(1, dummy_msg(1));
        store.store_received(2, dummy_msg(2));
        store.store_received(4, dummy_msg(4));
        store.store_received(5, dummy_msg(5));
        let gaps = store.inbound_gaps(1);
        assert_eq!(gaps, vec![(3, 3)]);
    }

    #[test]
    fn test_sequence_tracking() {
        let mut store = MessageStore::new("S", "T");
        assert_eq!(store.next_sender_seq_num(), 1);
        let seq = store.increment_sender_seq();
        assert_eq!(seq, 1);
        assert_eq!(store.next_sender_seq_num(), 2);
    }

    #[test]
    fn test_reset() {
        let mut store = MessageStore::new("S", "T");
        store.store_sent(1, dummy_msg(1));
        store.set_sender_seq(10);
        store.reset();
        assert_eq!(store.sent_count(), 0);
        assert_eq!(store.next_sender_seq_num(), 1);
    }

    #[test]
    fn test_pruning() {
        let mut store = MessageStore::new("S", "T").with_max_retain(5);
        for i in 1..=10u32 {
            store.store_sent(i, dummy_msg(i));
        }
        assert!(store.sent_count() <= 5);
    }

    #[test]
    fn test_gap_fill_tracker() {
        let mut tracker = GapFillTracker::new(3, 30);
        tracker.add_request(5, 8);
        assert!(tracker.has_pending());
        tracker.acknowledge(6);
        // Still pending because 5 and 7..8 not covered? Actually ack covers any record overlapping
        // The acknowledge removes records where begin<=seq<=end, so record (5,8) is removed at seq 6
        assert!(!tracker.has_pending());
    }
}
