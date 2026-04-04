use std::time::{Duration, Instant};
use thiserror::Error;
use crate::message::{FixMessage, MessageError, tags};
use crate::store::MessageStore;
use crate::types::MsgType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    Disconnected,
    Logon,
    Active,
    Logout,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionState::Disconnected => write!(f, "DISCONNECTED"),
            SessionState::Logon => write!(f, "LOGON"),
            SessionState::Active => write!(f, "ACTIVE"),
            SessionState::Logout => write!(f, "LOGOUT"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub begin_string: String,
    pub sender_comp_id: String,
    pub target_comp_id: String,
    pub heartbeat_interval: Duration,
    pub logon_timeout: Duration,
    pub logout_timeout: Duration,
    pub test_request_delay: Duration,
    pub reset_on_logon: bool,
    pub reset_on_logout: bool,
    pub reset_on_disconnect: bool,
    pub max_resend_retries: u32,
    pub check_comp_id: bool,
    pub check_latency: bool,
    pub max_latency: Duration,
}

impl SessionConfig {
    pub fn new(begin_string: &str, sender: &str, target: &str) -> Self {
        SessionConfig {
            begin_string: begin_string.to_string(),
            sender_comp_id: sender.to_string(),
            target_comp_id: target.to_string(),
            heartbeat_interval: Duration::from_secs(30),
            logon_timeout: Duration::from_secs(10),
            logout_timeout: Duration::from_secs(2),
            test_request_delay: Duration::from_secs(1),
            reset_on_logon: false,
            reset_on_logout: false,
            reset_on_disconnect: false,
            max_resend_retries: 3,
            check_comp_id: true,
            check_latency: false,
            max_latency: Duration::from_secs(120),
        }
    }
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("Invalid state transition: {from} -> {to}")]
    InvalidTransition { from: SessionState, to: SessionState },
    #[error("Message error: {0}")]
    MessageError(#[from] MessageError),
    #[error("Sequence gap: expected {expected}, got {received}")]
    SequenceGap { expected: u32, received: u32 },
    #[error("Duplicate sequence number: {0}")]
    DuplicateSeqNum(u32),
    #[error("CompID mismatch: expected sender={expected_sender} target={expected_target}, got sender={got_sender} target={got_target}")]
    CompIdMismatch {
        expected_sender: String,
        expected_target: String,
        got_sender: String,
        got_target: String,
    },
    #[error("Logon rejected: {0}")]
    LogonRejected(String),
    #[error("Session not active")]
    NotActive,
    #[error("Logout in progress")]
    LogoutInProgress,
}

/// Action the session requires the caller to perform
#[derive(Debug, Clone)]
pub enum SessionAction {
    /// Send a message to the counterparty
    Send(FixMessage),
    /// Disconnect the transport
    Disconnect,
    /// Log a warning
    Warn(String),
    /// Log info
    Info(String),
    /// Resend messages in range [begin, end] (0=to-end)
    ResendRange { begin: u32, end: u32 },
}

pub struct FixSession {
    config: SessionConfig,
    state: SessionState,
    store: MessageStore,
    last_sent: Instant,
    last_received: Instant,
    logon_time: Option<Instant>,
    logout_time: Option<Instant>,
    pending_test_request: Option<String>,
    test_request_id_counter: u64,
}

impl FixSession {
    pub fn new(config: SessionConfig) -> Self {
        let store = MessageStore::new(&config.sender_comp_id, &config.target_comp_id);
        let now = Instant::now();
        FixSession {
            config,
            state: SessionState::Disconnected,
            store,
            last_sent: now,
            last_received: now,
            logon_time: None,
            logout_time: None,
            pending_test_request: None,
            test_request_id_counter: 0,
        }
    }

    pub fn state(&self) -> SessionState { self.state }
    pub fn is_active(&self) -> bool { self.state == SessionState::Active }
    pub fn config(&self) -> &SessionConfig { &self.config }
    pub fn store(&self) -> &MessageStore { &self.store }
    pub fn store_mut(&mut self) -> &mut MessageStore { &mut self.store }

    /// Begin logon process — returns the Logon message to send
    pub fn initiate_logon(&mut self) -> Result<Vec<SessionAction>, SessionError> {
        if self.state != SessionState::Disconnected {
            return Err(SessionError::InvalidTransition {
                from: self.state,
                to: SessionState::Logon,
            });
        }
        if self.config.reset_on_logon {
            self.store.reset();
        }
        self.state = SessionState::Logon;
        self.logon_time = Some(Instant::now());

        let msg = self.build_logon(self.config.heartbeat_interval.as_secs() as u32)?;
        let actions = vec![SessionAction::Send(msg)];
        Ok(actions)
    }

    fn build_logon(&mut self, heartbeat_int: u32) -> Result<FixMessage, SessionError> {
        let seq = self.store.increment_sender_seq();
        let mut msg = FixMessage::with_header(
            &self.config.begin_string,
            MsgType::Logon.as_str(),
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            seq,
        );
        msg.set_field_str(tags::EncryptMethod, "0");
        msg.set_field_str(tags::HeartBtInt, &heartbeat_int.to_string());
        if self.config.reset_on_logon {
            msg.set_field_str(tags::ResetSeqNumFlag, "Y");
        }
        self.store.store_sent(seq, msg.clone());
        self.last_sent = Instant::now();
        Ok(msg)
    }

    /// Process an incoming message, returns actions to perform
    pub fn process(&mut self, mut msg: FixMessage) -> Result<Vec<SessionAction>, SessionError> {
        let msg_type_str = msg.msg_type.clone();
        let msg_type = MsgType::from_str(&msg_type_str)
            .map_err(|_| MessageError::MissingField(tags::MsgType))?;

        self.last_received = Instant::now();

        let mut actions = Vec::new();

        // Validate comp IDs for non-logon messages
        if self.config.check_comp_id && msg_type != MsgType::Logon {
            let sender = msg.sender_comp_id().unwrap_or("").to_string();
            let target = msg.target_comp_id().unwrap_or("").to_string();
            if sender != self.config.target_comp_id || target != self.config.sender_comp_id {
                actions.push(SessionAction::Warn(format!(
                    "CompID mismatch: sender={} target={}", sender, target
                )));
            }
        }

        // Sequence number handling (skip only for SequenceReset; Logon uses seq tracking)
        if msg_type != MsgType::SequenceReset {
            if let Some(seq) = msg.seq_num() {
                let expected = self.store.next_target_seq_num();
                if seq > expected {
                    // Gap detected: send ResendRequest
                    let resend = self.build_resend_request(expected, seq - 1)?;
                    actions.push(SessionAction::Send(resend));
                    actions.push(SessionAction::Warn(format!(
                        "Sequence gap: expected {} got {}", expected, seq
                    )));
                    // Buffer the out-of-order message
                    self.store.store_received(seq, msg);
                    return Ok(actions);
                } else if seq < expected {
                    // Old sequence — check PossDupFlag
                    let poss_dup = msg.get_field(tags::PossDupFlag)
                        .and_then(|f| f.value_char().ok())
                        .map(|c| c == b'Y')
                        .unwrap_or(false);
                    if !poss_dup {
                        // Logout with error
                        let logout = self.build_logout("MsgSeqNum too low")?;
                        actions.push(SessionAction::Send(logout));
                        self.state = SessionState::Logout;
                        return Ok(actions);
                    }
                    // Duplicate with PossDup=Y: ignore
                    return Ok(actions);
                } else {
                    self.store.increment_target_seq();
                    self.store.store_received(seq, msg.clone());
                }
            }
        }

        // Dispatch by message type
        match msg_type {
            MsgType::Logon => {
                let mut sub_actions = self.handle_logon(&msg)?;
                actions.append(&mut sub_actions);
            }
            MsgType::Logout => {
                let mut sub_actions = self.handle_logout(&msg)?;
                actions.append(&mut sub_actions);
            }
            MsgType::Heartbeat => {
                // If we had a pending TestRequest, clear it
                if let Some(ref test_id) = self.pending_test_request.clone() {
                    let recv_test_id = msg.get_field(tags::TestReqID)
                        .and_then(|f| f.value_str().ok())
                        .unwrap_or("");
                    if recv_test_id == test_id {
                        self.pending_test_request = None;
                    }
                }
            }
            MsgType::TestRequest => {
                let test_req_id = msg.require_str(tags::TestReqID)
                    .unwrap_or("0")
                    .to_string();
                let hb = self.build_heartbeat(Some(&test_req_id))?;
                actions.push(SessionAction::Send(hb));
            }
            MsgType::ResendRequest => {
                let begin = msg.require_u32(tags::BeginSeqNo)?;
                let end = msg.require_u32(tags::EndSeqNo)?;
                actions.push(SessionAction::ResendRange { begin, end });
                let mut resend_actions = self.handle_resend_request(begin, end)?;
                actions.append(&mut resend_actions);
            }
            MsgType::SequenceReset => {
                let mut sub_actions = self.handle_sequence_reset(&msg)?;
                actions.append(&mut sub_actions);
            }
            MsgType::Reject => {
                let ref_seq = msg.get_field(tags::RefSeqNum)
                    .and_then(|f| f.value_u32().ok())
                    .unwrap_or(0);
                let text = msg.get_field(tags::Text)
                    .and_then(|f| f.value_str().ok())
                    .unwrap_or("")
                    .to_string();
                actions.push(SessionAction::Warn(format!("Session Reject: ref_seq={} text={}", ref_seq, text)));
            }
            _ => {
                // Application message — pass through
                if self.state != SessionState::Active {
                    actions.push(SessionAction::Warn(format!(
                        "Received app message {} in state {}", msg_type_str, self.state
                    )));
                }
            }
        }

        Ok(actions)
    }

    fn handle_logon(&mut self, msg: &FixMessage) -> Result<Vec<SessionAction>, SessionError> {
        let mut actions = Vec::new();
        match self.state {
            SessionState::Disconnected | SessionState::Logon => {
                // Accept logon
                let reset_flag = msg.get_field(tags::ResetSeqNumFlag)
                    .and_then(|f| f.value_char().ok())
                    .map(|c| c == b'Y')
                    .unwrap_or(false);
                if reset_flag {
                    self.store.set_target_seq(1);
                }

                let peer_hb_int = msg.get_field(tags::HeartBtInt)
                    .and_then(|f| f.value_u32().ok())
                    .unwrap_or(30);

                self.state = SessionState::Active;
                actions.push(SessionAction::Info("Session ACTIVE".to_string()));

                // If we were the acceptor (Disconnected -> Active), send our Logon
                if self.logon_time.is_none() {
                    let reply = self.build_logon(peer_hb_int)?;
                    actions.push(SessionAction::Send(reply));
                }
            }
            SessionState::Active => {
                // Re-logon: reject or handle reset
                actions.push(SessionAction::Warn("Received Logon while Active".to_string()));
            }
            SessionState::Logout => {
                actions.push(SessionAction::Warn("Received Logon while Logging out".to_string()));
            }
        }
        Ok(actions)
    }

    fn handle_logout(&mut self, msg: &FixMessage) -> Result<Vec<SessionAction>, SessionError> {
        let mut actions = Vec::new();
        match self.state {
            SessionState::Active => {
                // Respond with Logout
                let logout = self.build_logout("")?;
                actions.push(SessionAction::Send(logout));
                self.state = SessionState::Disconnected;
                actions.push(SessionAction::Disconnect);
            }
            SessionState::Logout => {
                // We initiated, received acknowledgement
                self.state = SessionState::Disconnected;
                actions.push(SessionAction::Disconnect);
            }
            _ => {
                actions.push(SessionAction::Warn(format!(
                    "Received Logout in state {}", self.state
                )));
                self.state = SessionState::Disconnected;
                actions.push(SessionAction::Disconnect);
            }
        }
        if self.config.reset_on_logout {
            self.store.reset();
        }
        Ok(actions)
    }

    fn handle_resend_request(&mut self, begin: u32, end: u32) -> Result<Vec<SessionAction>, SessionError> {
        let mut actions = Vec::new();
        let range = self.store.get_sent_range(begin, end);

        if range.is_empty() {
            // Gap fill the entire range
            let gap_fill = self.build_sequence_reset(begin, end + 1, true)?;
            actions.push(SessionAction::Send(gap_fill));
        } else {
            for (seq, msg) in range {
                let mut resend = msg.clone();
                resend.set_field_str(tags::PossDupFlag, "Y");
                // Update OrigSendingTime
                let sending_time = resend.get_field(tags::SendingTime)
                    .and_then(|f| f.value_str().ok())
                    .unwrap_or("")
                    .to_string();
                if !sending_time.is_empty() {
                    resend.set_field_str(tags::OrigSendingTime, &sending_time);
                }
                resend.set_field_str(tags::SendingTime, &crate::types::UtcTimestamp::now().to_fix_str());
                actions.push(SessionAction::Send(resend));
            }
        }
        Ok(actions)
    }

    fn handle_sequence_reset(&mut self, msg: &FixMessage) -> Result<Vec<SessionAction>, SessionError> {
        let mut actions = Vec::new();
        let new_seq = msg.require_u32(tags::NewSeqNo)?;
        let gap_fill = msg.get_field(tags::GapFillFlag)
            .and_then(|f| f.value_char().ok())
            .map(|c| c == b'Y')
            .unwrap_or(false);

        if gap_fill {
            // GapFill: advance expected seq num
            if new_seq <= self.store.next_target_seq_num() {
                actions.push(SessionAction::Warn(format!(
                    "SequenceReset GapFill with NewSeqNo {} <= expected {}",
                    new_seq, self.store.next_target_seq_num()
                )));
            } else {
                self.store.set_target_seq(new_seq);
            }
        } else {
            // Reset: unconditionally set seq num
            self.store.set_target_seq(new_seq);
            actions.push(SessionAction::Info(format!("Sequence Reset to {}", new_seq)));
        }
        Ok(actions)
    }

    fn build_resend_request(&mut self, begin: u32, end: u32) -> Result<FixMessage, SessionError> {
        let seq = self.store.increment_sender_seq();
        let mut msg = FixMessage::with_header(
            &self.config.begin_string,
            MsgType::ResendRequest.as_str(),
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            seq,
        );
        msg.set_field_str(tags::BeginSeqNo, &begin.to_string());
        msg.set_field_str(tags::EndSeqNo, &end.to_string());
        self.store.store_sent(seq, msg.clone());
        self.last_sent = Instant::now();
        Ok(msg)
    }

    fn build_sequence_reset(&mut self, gap_start: u32, new_seq: u32, gap_fill: bool) -> Result<FixMessage, SessionError> {
        let mut msg = FixMessage::with_header(
            &self.config.begin_string,
            MsgType::SequenceReset.as_str(),
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            gap_start,
        );
        msg.set_field_str(tags::GapFillFlag, if gap_fill { "Y" } else { "N" });
        msg.set_field_str(tags::NewSeqNo, &new_seq.to_string());
        if gap_fill {
            msg.set_field_str(tags::PossDupFlag, "Y");
        }
        self.last_sent = Instant::now();
        Ok(msg)
    }

    pub fn build_logout(&mut self, text: &str) -> Result<FixMessage, SessionError> {
        let seq = self.store.increment_sender_seq();
        let mut msg = FixMessage::with_header(
            &self.config.begin_string,
            MsgType::Logout.as_str(),
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            seq,
        );
        if !text.is_empty() {
            msg.set_field_str(tags::Text, text);
        }
        self.store.store_sent(seq, msg.clone());
        self.last_sent = Instant::now();
        Ok(msg)
    }

    pub fn build_heartbeat(&mut self, test_req_id: Option<&str>) -> Result<FixMessage, SessionError> {
        let seq = self.store.increment_sender_seq();
        let mut msg = FixMessage::with_header(
            &self.config.begin_string,
            MsgType::Heartbeat.as_str(),
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            seq,
        );
        if let Some(id) = test_req_id {
            msg.set_field_str(tags::TestReqID, id);
        }
        self.store.store_sent(seq, msg.clone());
        self.last_sent = Instant::now();
        Ok(msg)
    }

    pub fn build_test_request(&mut self) -> Result<FixMessage, SessionError> {
        self.test_request_id_counter += 1;
        let test_id = format!("TR{}", self.test_request_id_counter);
        self.pending_test_request = Some(test_id.clone());

        let seq = self.store.increment_sender_seq();
        let mut msg = FixMessage::with_header(
            &self.config.begin_string,
            MsgType::TestRequest.as_str(),
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            seq,
        );
        msg.set_field_str(tags::TestReqID, &test_id);
        self.store.store_sent(seq, msg.clone());
        self.last_sent = Instant::now();
        Ok(msg)
    }

    /// Send an application message (wraps with sequence number and stores)
    pub fn send_app_message(&mut self, mut msg: FixMessage) -> Result<FixMessage, SessionError> {
        if self.state != SessionState::Active {
            return Err(SessionError::NotActive);
        }
        let seq = self.store.increment_sender_seq();
        msg.set_field_str(tags::MsgSeqNum, &seq.to_string());
        msg.set_field_str(tags::SendingTime, &crate::types::UtcTimestamp::now().to_fix_str());
        self.store.store_sent(seq, msg.clone());
        self.last_sent = Instant::now();
        Ok(msg)
    }

    /// Timer tick — returns any actions needed (heartbeats, test requests, etc.)
    pub fn on_tick(&mut self) -> Vec<SessionAction> {
        let mut actions = Vec::new();
        let now = Instant::now();

        match self.state {
            SessionState::Active => {
                // Check if we need to send a heartbeat
                let since_sent = now.duration_since(self.last_sent);
                if since_sent >= self.config.heartbeat_interval {
                    if let Ok(hb) = self.build_heartbeat(None) {
                        actions.push(SessionAction::Send(hb));
                    }
                }

                // Check if we haven't received anything for too long
                let since_recv = now.duration_since(self.last_received);
                let threshold = self.config.heartbeat_interval + self.config.test_request_delay;
                if since_recv >= threshold {
                    if self.pending_test_request.is_none() {
                        if let Ok(tr) = self.build_test_request() {
                            actions.push(SessionAction::Send(tr));
                        }
                    } else {
                        // Test request pending too long
                        let extra_timeout = threshold + self.config.heartbeat_interval;
                        if since_recv >= extra_timeout {
                            actions.push(SessionAction::Warn("No response to TestRequest, disconnecting".to_string()));
                            actions.push(SessionAction::Disconnect);
                            self.state = SessionState::Disconnected;
                        }
                    }
                }
            }
            SessionState::Logon => {
                if let Some(logon_time) = self.logon_time {
                    if now.duration_since(logon_time) >= self.config.logon_timeout {
                        actions.push(SessionAction::Warn("Logon timeout".to_string()));
                        actions.push(SessionAction::Disconnect);
                        self.state = SessionState::Disconnected;
                    }
                }
            }
            SessionState::Logout => {
                if let Some(logout_time) = self.logout_time {
                    if now.duration_since(logout_time) >= self.config.logout_timeout {
                        actions.push(SessionAction::Disconnect);
                        self.state = SessionState::Disconnected;
                    }
                }
            }
            SessionState::Disconnected => {}
        }
        actions
    }

    /// Initiate a logout
    pub fn initiate_logout(&mut self, text: &str) -> Result<Vec<SessionAction>, SessionError> {
        if self.state != SessionState::Active {
            return Err(SessionError::InvalidTransition {
                from: self.state,
                to: SessionState::Logout,
            });
        }
        self.state = SessionState::Logout;
        self.logout_time = Some(Instant::now());
        let msg = self.build_logout(text)?;
        Ok(vec![SessionAction::Send(msg)])
    }

    pub fn on_disconnect(&mut self) {
        if self.config.reset_on_disconnect {
            self.store.reset();
        }
        self.state = SessionState::Disconnected;
        self.logon_time = None;
        self.logout_time = None;
        self.pending_test_request = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session() -> FixSession {
        let config = SessionConfig::new("FIX.4.2", "SENDER", "TARGET");
        FixSession::new(config)
    }

    fn make_logon_msg(sender: &str, target: &str, seq: u32) -> FixMessage {
        let mut msg = FixMessage::with_header("FIX.4.2", MsgType::Logon.as_str(), sender, target, seq);
        msg.set_field_str(tags::EncryptMethod, "0");
        msg.set_field_str(tags::HeartBtInt, "30");
        msg
    }

    #[test]
    fn test_initial_state() {
        let session = make_session();
        assert_eq!(session.state(), SessionState::Disconnected);
    }

    #[test]
    fn test_initiate_logon() {
        let mut session = make_session();
        let actions = session.initiate_logon().unwrap();
        assert_eq!(session.state(), SessionState::Logon);
        assert!(actions.iter().any(|a| matches!(a, SessionAction::Send(_))));
    }

    #[test]
    fn test_accept_logon() {
        let mut session = make_session();
        // Acceptor: start Disconnected, receive Logon
        let logon = make_logon_msg("TARGET", "SENDER", 1);
        let actions = session.process(logon).unwrap();
        assert_eq!(session.state(), SessionState::Active);
        // Should send a Logon response
        assert!(actions.iter().any(|a| matches!(a, SessionAction::Send(_))));
    }

    #[test]
    fn test_initiator_logon_flow() {
        let mut session = make_session();
        session.initiate_logon().unwrap();
        assert_eq!(session.state(), SessionState::Logon);
        // Receive Logon from counterparty
        let logon = make_logon_msg("TARGET", "SENDER", 1);
        session.process(logon).unwrap();
        assert_eq!(session.state(), SessionState::Active);
    }

    #[test]
    fn test_logout_flow() {
        let mut session = make_session();
        session.initiate_logon().unwrap();
        let logon = make_logon_msg("TARGET", "SENDER", 1);
        session.process(logon).unwrap();

        let actions = session.initiate_logout("Normal").unwrap();
        assert_eq!(session.state(), SessionState::Logout);

        let logout_msg = make_logout_msg("TARGET", "SENDER", 2);
        session.process(logout_msg).unwrap();
        assert_eq!(session.state(), SessionState::Disconnected);
    }

    fn make_logout_msg(sender: &str, target: &str, seq: u32) -> FixMessage {
        FixMessage::with_header("FIX.4.2", MsgType::Logout.as_str(), sender, target, seq)
    }

    #[test]
    fn test_sequence_gap_triggers_resend_request() {
        let mut session = make_session();
        session.initiate_logon().unwrap();
        let logon = make_logon_msg("TARGET", "SENDER", 1);
        session.process(logon).unwrap();

        // Send a message with seq 3 (gap at 2)
        let mut msg = FixMessage::with_header("FIX.4.2", MsgType::Heartbeat.as_str(), "TARGET", "SENDER", 3);
        let actions = session.process(msg).unwrap();
        assert!(actions.iter().any(|a| matches!(a, SessionAction::Send(_) | SessionAction::Warn(_))));
    }

    #[test]
    fn test_double_logon_is_rejected() {
        let mut session = make_session();
        assert!(session.initiate_logon().is_ok());
        // Second initiate_logon should fail
        assert!(session.initiate_logon().is_err());
    }

    #[test]
    fn test_send_app_message_increments_seq() {
        let mut session = make_session();
        session.initiate_logon().unwrap();
        let logon = make_logon_msg("TARGET", "SENDER", 1);
        session.process(logon).unwrap();

        let msg = FixMessage::new("FIX.4.2", "D");
        let sent = session.send_app_message(msg).unwrap();
        assert!(sent.seq_num().is_some());
    }
}
