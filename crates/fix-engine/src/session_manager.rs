// session_manager.rs -- FIX session lifecycle management for SRFM.
// Handles logon/logout, heartbeats, sequence numbers, and gap detection.

use std::collections::HashMap;
use thiserror::Error;
use crate::message::{FixMessage, FixField, tags};
use crate::types::UtcTimestamp;

// ---------------------------------------------------------------------------
// SessionError
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("Invalid state: operation not allowed in state {0:?}")]
    InvalidState(SessionState),
    #[error("Sequence gap: expected {expected}, received {received}")]
    SequenceGap { expected: u64, received: u64 },
    #[error("CompID mismatch: expected sender={expected_sender}, got {got_sender}")]
    CompIdMismatch { expected_sender: String, got_sender: String },
    #[error("Missing field: tag {0}")]
    MissingField(u32),
    #[error("Field error: {0}")]
    FieldError(#[from] crate::message::MessageError),
}

// ---------------------------------------------------------------------------
// SessionState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    Disconnected,
    LogonSent,
    Active,
    LogoutSent,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionState::Disconnected => write!(f, "DISCONNECTED"),
            SessionState::LogonSent   => write!(f, "LOGON_SENT"),
            SessionState::Active      => write!(f, "ACTIVE"),
            SessionState::LogoutSent  => write!(f, "LOGOUT_SENT"),
        }
    }
}

// ---------------------------------------------------------------------------
// SessionConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub sender_comp_id: String,
    pub target_comp_id: String,
    pub begin_string: String,
    /// Heartbeat interval in seconds (tag 108).
    pub heartbeat_interval_s: u64,
    /// If true, reset sequence numbers on logon.
    pub reset_on_logon: bool,
    /// If true, reset sequence numbers on logout.
    pub reset_on_logout: bool,
}

impl SessionConfig {
    pub fn new(
        sender: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            sender_comp_id: sender.into(),
            target_comp_id: target.into(),
            begin_string: "FIX.4.2".to_string(),
            heartbeat_interval_s: 30,
            reset_on_logon: false,
            reset_on_logout: false,
        }
    }

    pub fn with_begin_string(mut self, bs: impl Into<String>) -> Self {
        self.begin_string = bs.into();
        self
    }

    pub fn with_heartbeat_interval(mut self, secs: u64) -> Self {
        self.heartbeat_interval_s = secs;
        self
    }

    pub fn with_reset_on_logon(mut self, v: bool) -> Self {
        self.reset_on_logon = v;
        self
    }
}

// ---------------------------------------------------------------------------
// SessionEvent
// ---------------------------------------------------------------------------

/// Event returned by FIXSessionManager after processing a received message.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionEvent {
    /// No action required.
    NoOp,
    /// Session state changed to the given value.
    StateChange(SessionState),
    /// Send a heartbeat (optionally with TestReqID echoed back).
    SendHeartbeat,
    /// Request resend of messages from seq_from to seq_to (inclusive).
    /// seq_to = 0 means "to current".
    RequestResend(u64, u64),
    /// A non-session application message that should be delivered upstream.
    ApplicationMessage(FixMessage),
}

// ---------------------------------------------------------------------------
// FIXSessionManager
// ---------------------------------------------------------------------------

/// Manages a FIX session's lifecycle and sequence numbering.
pub struct FIXSessionManager {
    pub config: SessionConfig,
    pub state: SessionState,
    /// Our outbound sequence number (incremented on each message sent).
    outbound_seq_num: u64,
    /// Expected inbound sequence number (next message we expect to receive).
    expected_inbound_seq: u64,
    /// Buffer for storing outbound messages for potential resend.
    /// Maps seq_num -> encoded message bytes.
    sent_messages: HashMap<u64, Vec<u8>>,
}

impl FIXSessionManager {
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            state: SessionState::Disconnected,
            outbound_seq_num: 1,
            expected_inbound_seq: 1,
            sent_messages: HashMap::new(),
        }
    }

    // ---------------------------------------------------------------------------
    // Sequence numbers
    // ---------------------------------------------------------------------------

    /// Return and increment the next outbound sequence number.
    pub fn next_seq_num(&mut self) -> u64 {
        let n = self.outbound_seq_num;
        self.outbound_seq_num += 1;
        n
    }

    /// Peek at the current outbound sequence number without incrementing.
    pub fn current_seq_num(&self) -> u64 {
        self.outbound_seq_num
    }

    pub fn expected_inbound_seq(&self) -> u64 {
        self.expected_inbound_seq
    }

    pub fn reset_sequences(&mut self) {
        self.outbound_seq_num = 1;
        self.expected_inbound_seq = 1;
        self.sent_messages.clear();
    }

    // ---------------------------------------------------------------------------
    // Message building helpers
    // ---------------------------------------------------------------------------

    fn build_message(&mut self, msg_type: &str) -> FixMessage {
        let seq = self.next_seq_num();
        let mut msg = FixMessage::new(&self.config.begin_string, msg_type);
        msg.set_field_str(tags::MsgType, msg_type);
        msg.set_field_str(tags::SenderCompID, &self.config.sender_comp_id.clone());
        msg.set_field_str(tags::TargetCompID, &self.config.target_comp_id.clone());
        msg.set_field_str(tags::MsgSeqNum, &seq.to_string());
        msg.set_field_str(tags::SendingTime, &UtcTimestamp::now().to_fix_str());
        msg
    }

    fn encode_message(msg: &FixMessage) -> Vec<u8> {
        use crate::message::SOH;
        // Simple field serialization -- body only (no length/checksum for now)
        let mut body: Vec<u8> = Vec::new();
        for field in msg.fields() {
            body.extend_from_slice(&field.tag.to_string().as_bytes());
            body.push(b'=');
            body.extend_from_slice(&field.value);
            body.push(SOH);
        }
        body
    }

    fn store_sent(&mut self, seq: u64, bytes: Vec<u8>) {
        self.sent_messages.insert(seq, bytes);
        // Trim store to last 1000 messages to bound memory usage
        if self.sent_messages.len() > 1000 {
            let min_key = self.sent_messages.keys().copied().min().unwrap_or(0);
            self.sent_messages.remove(&min_key);
        }
    }

    // ---------------------------------------------------------------------------
    // Outbound message construction
    // ---------------------------------------------------------------------------

    /// Build and encode a Logon message (MsgType=A).
    pub fn send_logon(&mut self) -> Vec<u8> {
        if self.config.reset_on_logon {
            self.reset_sequences();
            // After reset, next_seq_num() will hand out 1 -- but build_message
            // calls next_seq_num() internally so we must call build_message first.
        }
        let mut msg = self.build_message("A");
        msg.set_field_str(tags::EncryptMethod, "0");
        msg.set_field_str(tags::HeartBtInt, &self.config.heartbeat_interval_s.to_string());
        if self.config.reset_on_logon {
            msg.set_field_str(tags::ResetSeqNumFlag, "Y");
        }
        let seq = msg.fields()
            .iter()
            .find(|f| f.tag == tags::MsgSeqNum)
            .and_then(|f| f.value_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1);
        self.state = SessionState::LogonSent;
        let bytes = Self::encode_message(&msg);
        self.store_sent(seq, bytes.clone());
        bytes
    }

    /// Build and encode a Heartbeat message (MsgType=0).
    /// Pass a TestReqID to echo back a test request; pass None for a scheduled heartbeat.
    pub fn send_heartbeat(&mut self, test_req_id: Option<&str>) -> Vec<u8> {
        let mut msg = self.build_message("0");
        if let Some(id) = test_req_id {
            msg.set_field_str(tags::TestReqID, id);
        }
        let seq = msg.fields()
            .iter()
            .find(|f| f.tag == tags::MsgSeqNum)
            .and_then(|f| f.value_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let bytes = Self::encode_message(&msg);
        self.store_sent(seq, bytes.clone());
        bytes
    }

    /// Build and encode a Logout message (MsgType=5).
    pub fn send_logout(&mut self, text: &str) -> Vec<u8> {
        let mut msg = self.build_message("5");
        if !text.is_empty() {
            msg.set_field_str(tags::Text, text);
        }
        let seq = msg.fields()
            .iter()
            .find(|f| f.tag == tags::MsgSeqNum)
            .and_then(|f| f.value_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        self.state = SessionState::LogoutSent;
        let bytes = Self::encode_message(&msg);
        self.store_sent(seq, bytes.clone());
        bytes
    }

    /// Build a ResendRequest message (MsgType=2) for the given sequence range.
    pub fn build_resend_request(&mut self, begin_seq: u64, end_seq: u64) -> Vec<u8> {
        let mut msg = self.build_message("2");
        msg.set_field_str(tags::BeginSeqNo, &begin_seq.to_string());
        msg.set_field_str(tags::EndSeqNo, &end_seq.to_string());
        Self::encode_message(&msg)
    }

    // ---------------------------------------------------------------------------
    // Inbound message processing
    // ---------------------------------------------------------------------------

    /// Process a received FIX message and advance session state.
    ///
    /// Performs:
    ///   - CompID validation
    ///   - Sequence number gap detection (triggers ResendRequest)
    ///   - Session-level message handling (Logon, Logout, Heartbeat, TestRequest,
    ///     ResendRequest, SequenceReset)
    ///   - Forwards application messages upstream via ApplicationMessage event
    pub fn on_message_received(&mut self, msg: FixMessage) -> SessionEvent {
        // Validate SenderCompID
        if let Some(Ok(sender)) = msg.get_str(tags::SenderCompID) {
            if sender != self.config.target_comp_id {
                return SessionEvent::NoOp; // ignore or could return error event
            }
        }

        // Extract and validate sequence number
        let recv_seq = msg.fields()
            .iter()
            .find(|f| f.tag == tags::MsgSeqNum)
            .and_then(|f| f.value_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        // Sequence gap detection (skip for SequenceReset and PossDupFlag)
        let is_gap_fill = msg.has_field(tags::GapFillFlag)
            && msg.get_field(tags::GapFillFlag)
                .and_then(|f| f.value_str().ok())
                .map(|v| v == "Y")
                .unwrap_or(false);
        let is_poss_dup = msg.has_field(tags::PossDupFlag)
            && msg.get_field(tags::PossDupFlag)
                .and_then(|f| f.value_str().ok())
                .map(|v| v == "Y")
                .unwrap_or(false);
        let msg_type = msg.msg_type.as_str();

        if !is_gap_fill && !is_poss_dup && msg_type != "4" && recv_seq > 0 {
            if recv_seq > self.expected_inbound_seq {
                // Gap detected -- request resend
                let from = self.expected_inbound_seq;
                let to = recv_seq - 1;
                self.expected_inbound_seq = recv_seq + 1;
                return SessionEvent::RequestResend(from, to);
            } else if recv_seq < self.expected_inbound_seq {
                // Duplicate or too-low sequence -- ignore
                return SessionEvent::NoOp;
            }
        }

        if recv_seq > 0 && !is_gap_fill {
            self.expected_inbound_seq = recv_seq + 1;
        }

        // Session-level message dispatch
        match msg_type {
            // Logon
            "A" => {
                let prev = self.state;
                self.state = SessionState::Active;
                if prev != SessionState::Active {
                    return SessionEvent::StateChange(SessionState::Active);
                }
                SessionEvent::NoOp
            }

            // Logout
            "5" => {
                if self.config.reset_on_logout {
                    self.reset_sequences();
                }
                let prev = self.state;
                self.state = SessionState::Disconnected;
                if prev != SessionState::Disconnected {
                    return SessionEvent::StateChange(SessionState::Disconnected);
                }
                SessionEvent::NoOp
            }

            // Heartbeat
            "0" => SessionEvent::NoOp,

            // TestRequest -- respond with heartbeat
            "1" => SessionEvent::SendHeartbeat,

            // ResendRequest
            "2" => {
                // The counterparty is asking us to replay sent messages.
                // Return a SessionEvent so the caller can handle this.
                let begin = msg.get_field(tags::BeginSeqNo)
                    .and_then(|f| f.value_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(1);
                let end = msg.get_field(tags::EndSeqNo)
                    .and_then(|f| f.value_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                SessionEvent::RequestResend(begin, end)
            }

            // SequenceReset
            "4" => {
                let new_seq = msg.get_field(tags::NewSeqNo)
                    .and_then(|f| f.value_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(self.expected_inbound_seq);
                self.expected_inbound_seq = new_seq;
                SessionEvent::NoOp
            }

            // Reject (session-level)
            "3" => SessionEvent::NoOp,

            // All other messages are application-level
            _ => {
                if self.state == SessionState::Active {
                    SessionEvent::ApplicationMessage(msg)
                } else {
                    SessionEvent::NoOp
                }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Resend support
    // ---------------------------------------------------------------------------

    /// Retrieve a previously sent message for resend by sequence number.
    pub fn get_sent_message(&self, seq: u64) -> Option<&Vec<u8>> {
        self.sent_messages.get(&seq)
    }

    /// Return a copy of all stored sent messages in range [from, to] inclusive.
    pub fn get_resend_range(&self, from: u64, to: u64) -> Vec<(u64, Vec<u8>)> {
        let mut result: Vec<(u64, Vec<u8>)> = self.sent_messages
            .iter()
            .filter(|(&seq, _)| seq >= from && (to == 0 || seq <= to))
            .map(|(&seq, msg)| (seq, msg.clone()))
            .collect();
        result.sort_by_key(|(seq, _)| *seq);
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{FixMessage, tags};

    fn make_mgr() -> FIXSessionManager {
        let config = SessionConfig::new("SRFM", "BROKER");
        FIXSessionManager::new(config)
    }

    fn make_logon_response() -> FixMessage {
        let mut msg = FixMessage::new("FIX.4.2", "A");
        msg.set_field_str(tags::SenderCompID, "BROKER");
        msg.set_field_str(tags::TargetCompID, "SRFM");
        msg.set_field_str(tags::MsgSeqNum, "1");
        msg.set_field_str(tags::EncryptMethod, "0");
        msg.set_field_str(tags::HeartBtInt, "30");
        msg
    }

    // ---------------------------------------------------------------------------

    #[test]
    fn test_initial_state_disconnected() {
        let mgr = make_mgr();
        assert_eq!(mgr.state, SessionState::Disconnected);
    }

    #[test]
    fn test_send_logon_transitions_to_logon_sent() {
        let mut mgr = make_mgr();
        let bytes = mgr.send_logon();
        assert!(!bytes.is_empty());
        assert_eq!(mgr.state, SessionState::LogonSent);
    }

    #[test]
    fn test_on_logon_response_transitions_to_active() {
        let mut mgr = make_mgr();
        mgr.send_logon();
        let resp = make_logon_response();
        let event = mgr.on_message_received(resp);
        assert!(matches!(event, SessionEvent::StateChange(SessionState::Active)));
        assert_eq!(mgr.state, SessionState::Active);
    }

    #[test]
    fn test_send_heartbeat_produces_bytes() {
        let mut mgr = make_mgr();
        mgr.send_logon();
        mgr.on_message_received(make_logon_response());
        let hb = mgr.send_heartbeat(None);
        assert!(!hb.is_empty());
    }

    #[test]
    fn test_send_heartbeat_with_test_req_id() {
        let mut mgr = make_mgr();
        let hb = mgr.send_heartbeat(Some("TEST-123"));
        // Should contain the TestReqID value
        let s = String::from_utf8_lossy(&hb);
        assert!(s.contains("TEST-123"), "Heartbeat should echo TestReqID: {s}");
    }

    #[test]
    fn test_send_logout_transitions_state() {
        let mut mgr = make_mgr();
        mgr.send_logon();
        mgr.on_message_received(make_logon_response());
        let bytes = mgr.send_logout("End of day");
        assert!(!bytes.is_empty());
        assert_eq!(mgr.state, SessionState::LogoutSent);
    }

    #[test]
    fn test_sequence_number_increments() {
        let mut mgr = make_mgr();
        let s1 = mgr.next_seq_num();
        let s2 = mgr.next_seq_num();
        let s3 = mgr.next_seq_num();
        assert_eq!(s1 + 1, s2);
        assert_eq!(s2 + 1, s3);
    }

    #[test]
    fn test_sequence_reset_resets_counters() {
        let mut mgr = make_mgr();
        mgr.next_seq_num();
        mgr.next_seq_num();
        assert_eq!(mgr.current_seq_num(), 3);
        mgr.reset_sequences();
        assert_eq!(mgr.current_seq_num(), 1);
        assert_eq!(mgr.expected_inbound_seq(), 1);
    }

    #[test]
    fn test_gap_detection_triggers_resend_request() {
        let mut mgr = make_mgr();
        mgr.send_logon();
        mgr.on_message_received(make_logon_response()); // seq 1 consumed

        // Send a message with seq 3, skipping 2 -- should trigger resend request
        let mut app_msg = FixMessage::new("FIX.4.2", "8");
        app_msg.set_field_str(tags::SenderCompID, "BROKER");
        app_msg.set_field_str(tags::MsgSeqNum, "3");
        app_msg.set_field_str(tags::ExecType, "0");
        let event = mgr.on_message_received(app_msg);
        assert!(matches!(event, SessionEvent::RequestResend(2, 2)), "Expected resend request, got {event:?}");
    }

    #[test]
    fn test_test_request_triggers_send_heartbeat() {
        let mut mgr = make_mgr();
        mgr.send_logon();
        mgr.on_message_received(make_logon_response());

        let mut tr = FixMessage::new("FIX.4.2", "1");
        tr.set_field_str(tags::SenderCompID, "BROKER");
        tr.set_field_str(tags::MsgSeqNum, "2");
        tr.set_field_str(tags::TestReqID, "PING-1");
        let event = mgr.on_message_received(tr);
        assert_eq!(event, SessionEvent::SendHeartbeat);
    }

    #[test]
    fn test_application_message_forwarded_when_active() {
        let mut mgr = make_mgr();
        mgr.send_logon();
        mgr.on_message_received(make_logon_response());

        let mut app = FixMessage::new("FIX.4.2", "8");
        app.set_field_str(tags::SenderCompID, "BROKER");
        app.set_field_str(tags::MsgSeqNum, "2");
        let event = mgr.on_message_received(app);
        assert!(matches!(event, SessionEvent::ApplicationMessage(_)));
    }

    #[test]
    fn test_logout_response_transitions_to_disconnected() {
        let mut mgr = make_mgr();
        mgr.send_logon();
        mgr.on_message_received(make_logon_response());
        mgr.send_logout("");

        let mut logout_resp = FixMessage::new("FIX.4.2", "5");
        logout_resp.set_field_str(tags::SenderCompID, "BROKER");
        logout_resp.set_field_str(tags::MsgSeqNum, "2");
        let event = mgr.on_message_received(logout_resp);
        assert_eq!(mgr.state, SessionState::Disconnected);
        assert!(matches!(event, SessionEvent::StateChange(SessionState::Disconnected)));
    }
}
