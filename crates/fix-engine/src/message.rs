use std::collections::HashMap;
use std::fmt;
use thiserror::Error;
use crate::types::{MsgType, FixTypeError, UtcTimestamp};

pub const SOH: u8 = 0x01;

/// Well-known FIX tag numbers
#[allow(non_upper_case_globals)]
pub mod tags {
    pub const BeginString: u32 = 8;
    pub const BodyLength: u32 = 9;
    pub const MsgType: u32 = 35;
    pub const SenderCompID: u32 = 49;
    pub const TargetCompID: u32 = 56;
    pub const MsgSeqNum: u32 = 34;
    pub const SendingTime: u32 = 52;
    pub const CheckSum: u32 = 10;
    pub const PossDupFlag: u32 = 43;
    pub const PossResend: u32 = 97;
    pub const OrigSendingTime: u32 = 122;
    pub const OnBehalfOfCompID: u32 = 115;
    pub const DeliverToCompID: u32 = 128;

    // Order fields
    pub const ClOrdID: u32 = 11;
    pub const OrigClOrdID: u32 = 41;
    pub const OrderID: u32 = 37;
    pub const ExecID: u32 = 17;
    pub const ExecType: u32 = 150;
    pub const OrdStatus: u32 = 39;
    pub const Symbol: u32 = 55;
    pub const SecurityExchange: u32 = 207;
    pub const Side: u32 = 54;
    pub const OrderQty: u32 = 38;
    pub const OrdType: u32 = 40;
    pub const Price: u32 = 44;
    pub const StopPx: u32 = 99;
    pub const TimeInForce: u32 = 59;
    pub const TransactTime: u32 = 60;
    pub const CumQty: u32 = 14;
    pub const LeavesQty: u32 = 151;
    pub const LastQty: u32 = 32;
    pub const LastPx: u32 = 31;
    pub const AvgPx: u32 = 6;
    pub const Text: u32 = 58;
    pub const OrdRejReason: u32 = 103;
    pub const ExecInst: u32 = 18;
    pub const HandlInst: u32 = 21;
    pub const Currency: u32 = 15;
    pub const Account: u32 = 1;
    pub const IDSource: u32 = 22;
    pub const SecurityID: u32 = 48;
    pub const MaturityMonthYear: u32 = 200;
    pub const PutOrCall: u32 = 201;
    pub const StrikePrice: u32 = 202;
    pub const SecurityType: u32 = 167;

    // Market data fields
    pub const MDReqID: u32 = 262;
    pub const SubscriptionRequestType: u32 = 263;
    pub const MarketDepth: u32 = 264;
    pub const MDUpdateType: u32 = 265;
    pub const NoMDEntryTypes: u32 = 267;
    pub const NoMDEntries: u32 = 268;
    pub const MDEntryType: u32 = 269;
    pub const MDEntryPx: u32 = 270;
    pub const MDEntrySize: u32 = 271;
    pub const MDEntryDate: u32 = 272;
    pub const MDEntryTime: u32 = 273;
    pub const TickDirection: u32 = 274;
    pub const MDMkt: u32 = 275;
    pub const NoRelatedSym: u32 = 146;

    // Session fields
    pub const HeartBtInt: u32 = 108;
    pub const TestReqID: u32 = 112;
    pub const BeginSeqNo: u32 = 7;
    pub const EndSeqNo: u32 = 16;
    pub const NewSeqNo: u32 = 36;
    pub const GapFillFlag: u32 = 123;
    pub const ResetSeqNumFlag: u32 = 141;
    pub const EncryptMethod: u32 = 98;
    pub const RefSeqNum: u32 = 45;
    pub const RefTagID: u32 = 371;
    pub const RefMsgType: u32 = 372;
    pub const SessionRejectReason: u32 = 373;
    pub const BusinessRejectReason: u32 = 380;
    pub const BusinessRejectRefID: u32 = 379;
}

pub type FixTag = u32;

#[derive(Debug, Clone, PartialEq)]
pub struct FixField {
    pub tag: FixTag,
    pub value: Vec<u8>,
}

impl FixField {
    pub fn new(tag: FixTag, value: impl Into<Vec<u8>>) -> Self {
        FixField { tag, value: value.into() }
    }

    pub fn new_str(tag: FixTag, value: &str) -> Self {
        FixField { tag, value: value.as_bytes().to_vec() }
    }

    pub fn value_str(&self) -> Result<&str, MessageError> {
        std::str::from_utf8(&self.value)
            .map_err(|_| MessageError::Utf8Error(self.tag))
    }

    pub fn value_i64(&self) -> Result<i64, MessageError> {
        self.value_str()?
            .parse::<i64>()
            .map_err(|_| MessageError::ParseInt(self.tag))
    }

    pub fn value_u32(&self) -> Result<u32, MessageError> {
        self.value_str()?
            .parse::<u32>()
            .map_err(|_| MessageError::ParseInt(self.tag))
    }

    pub fn value_f64(&self) -> Result<f64, MessageError> {
        self.value_str()?
            .parse::<f64>()
            .map_err(|_| MessageError::ParseFloat(self.tag))
    }

    pub fn value_char(&self) -> Result<u8, MessageError> {
        if self.value.len() == 1 {
            Ok(self.value[0])
        } else {
            Err(MessageError::InvalidField(self.tag, "expected single char".into()))
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let tag_str = self.tag.to_string();
        out.extend_from_slice(tag_str.as_bytes());
        out.push(b'=');
        out.extend_from_slice(&self.value);
        out.push(SOH);
        out
    }
}

impl fmt::Display for FixField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.tag, String::from_utf8_lossy(&self.value))
    }
}

/// A complete FIX message with ordered fields
#[derive(Debug, Clone, PartialEq)]
pub struct FixMessage {
    /// All fields in wire order (includes header, body, trailer)
    fields: Vec<FixField>,
    /// Fast lookup: tag -> index in fields (first occurrence)
    index: HashMap<FixTag, usize>,
    /// Version string e.g. "FIX.4.2"
    pub begin_string: String,
    /// Message type
    pub msg_type: String,
}

impl FixMessage {
    pub fn new(begin_string: &str, msg_type: &str) -> Self {
        FixMessage {
            fields: Vec::new(),
            index: HashMap::new(),
            begin_string: begin_string.to_string(),
            msg_type: msg_type.to_string(),
        }
    }

    /// Create a new message with standard header fields populated
    pub fn with_header(
        begin_string: &str,
        msg_type: &str,
        sender: &str,
        target: &str,
        seq_num: u32,
    ) -> Self {
        let mut msg = Self::new(begin_string, msg_type);
        msg.set_field_str(tags::MsgType, msg_type);
        msg.set_field_str(tags::SenderCompID, sender);
        msg.set_field_str(tags::TargetCompID, target);
        msg.set_field_str(tags::MsgSeqNum, &seq_num.to_string());
        msg.set_field_str(tags::SendingTime, &UtcTimestamp::now().to_fix_str());
        msg
    }

    pub fn set_field(&mut self, field: FixField) {
        if let Some(&idx) = self.index.get(&field.tag) {
            self.fields[idx] = field;
        } else {
            let idx = self.fields.len();
            self.index.insert(field.tag, idx);
            self.fields.push(field);
        }
    }

    pub fn set_field_str(&mut self, tag: FixTag, value: &str) {
        self.set_field(FixField::new_str(tag, value));
    }

    pub fn set_field_bytes(&mut self, tag: FixTag, value: Vec<u8>) {
        self.set_field(FixField { tag, value });
    }

    pub fn get_field(&self, tag: FixTag) -> Option<&FixField> {
        self.index.get(&tag).map(|&idx| &self.fields[idx])
    }

    pub fn get_str(&self, tag: FixTag) -> Option<Result<&str, MessageError>> {
        self.get_field(tag).map(|f| f.value_str())
    }

    pub fn require_str(&self, tag: FixTag) -> Result<&str, MessageError> {
        self.get_field(tag)
            .ok_or(MessageError::MissingField(tag))?
            .value_str()
    }

    pub fn require_i64(&self, tag: FixTag) -> Result<i64, MessageError> {
        self.get_field(tag)
            .ok_or(MessageError::MissingField(tag))?
            .value_i64()
    }

    pub fn require_u32(&self, tag: FixTag) -> Result<u32, MessageError> {
        self.get_field(tag)
            .ok_or(MessageError::MissingField(tag))?
            .value_u32()
    }

    pub fn require_f64(&self, tag: FixTag) -> Result<f64, MessageError> {
        self.get_field(tag)
            .ok_or(MessageError::MissingField(tag))?
            .value_f64()
    }

    pub fn require_char(&self, tag: FixTag) -> Result<u8, MessageError> {
        self.get_field(tag)
            .ok_or(MessageError::MissingField(tag))?
            .value_char()
    }

    pub fn has_field(&self, tag: FixTag) -> bool {
        self.index.contains_key(&tag)
    }

    pub fn fields(&self) -> &[FixField] {
        &self.fields
    }

    pub fn remove_field(&mut self, tag: FixTag) -> Option<FixField> {
        if let Some(idx) = self.index.remove(&tag) {
            let field = self.fields.remove(idx);
            // Rebuild index after removal
            self.index.clear();
            for (i, f) in self.fields.iter().enumerate() {
                self.index.entry(f.tag).or_insert(i);
            }
            Some(field)
        } else {
            None
        }
    }

    /// Compute the FIX checksum (sum of all bytes mod 256) formatted as 3 digits
    pub fn compute_checksum(data: &[u8]) -> u8 {
        let sum: u32 = data.iter().map(|&b| b as u32).sum();
        (sum % 256) as u8
    }

    pub fn format_checksum(ck: u8) -> String {
        format!("{:03}", ck)
    }

    /// Validate checksum field of a complete FIX wire message (bytes)
    pub fn validate_checksum(wire: &[u8]) -> Result<(), MessageError> {
        // Find last SOH before "10="
        // The checksum tag covers everything before the "10=" field
        let ck_prefix = b"10=";
        let pos = wire.windows(3)
            .rposition(|w| w == ck_prefix)
            .ok_or(MessageError::MissingChecksum)?;
        let body = &wire[..pos];
        let computed = Self::compute_checksum(body);
        // Extract declared checksum
        let ck_field_start = pos + 3;
        let ck_field_end = wire[ck_field_start..]
            .iter()
            .position(|&b| b == SOH)
            .map(|p| ck_field_start + p)
            .ok_or(MessageError::MissingChecksum)?;
        let ck_str = std::str::from_utf8(&wire[ck_field_start..ck_field_end])
            .map_err(|_| MessageError::InvalidChecksum)?;
        let declared: u8 = ck_str.parse::<u32>()
            .map_err(|_| MessageError::InvalidChecksum)? as u8;
        if computed == declared {
            Ok(())
        } else {
            Err(MessageError::ChecksumMismatch { computed, declared })
        }
    }

    /// Encode the message to wire format with BodyLength and CheckSum computed
    pub fn encode(&self) -> Vec<u8> {
        // Build body (everything except BeginString, BodyLength, CheckSum)
        let mut body = Vec::new();
        for field in &self.fields {
            if field.tag == tags::BeginString
                || field.tag == tags::BodyLength
                || field.tag == tags::CheckSum
            {
                continue;
            }
            body.extend_from_slice(&field.encode());
        }

        let body_len = body.len();

        let mut out = Vec::new();
        // BeginString
        out.extend_from_slice(&FixField::new_str(tags::BeginString, &self.begin_string).encode());
        // BodyLength
        out.extend_from_slice(&FixField::new_str(tags::BodyLength, &body_len.to_string()).encode());
        // Body
        out.extend_from_slice(&body);
        // CheckSum
        let ck = Self::compute_checksum(&out);
        out.extend_from_slice(&FixField::new_str(tags::CheckSum, &Self::format_checksum(ck)).encode());
        out
    }

    pub fn msg_type_enum(&self) -> Result<MsgType, FixTypeError> {
        MsgType::from_str(&self.msg_type)
    }

    pub fn seq_num(&self) -> Option<u32> {
        self.get_field(tags::MsgSeqNum)?.value_u32().ok()
    }

    pub fn sender_comp_id(&self) -> Option<&str> {
        self.get_field(tags::SenderCompID)?.value_str().ok()
    }

    pub fn target_comp_id(&self) -> Option<&str> {
        self.get_field(tags::TargetCompID)?.value_str().ok()
    }

    /// Count all fields including possible duplicates in repeating groups
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

impl fmt::Display for FixMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for field in &self.fields {
            write!(f, "{}|", field)?;
        }
        Ok(())
    }
}

/// Repeating group container
#[derive(Debug, Clone)]
pub struct RepeatingGroup {
    pub delimiter_tag: FixTag,
    pub instances: Vec<Vec<FixField>>,
}

impl RepeatingGroup {
    pub fn new(delimiter_tag: FixTag) -> Self {
        RepeatingGroup {
            delimiter_tag,
            instances: Vec::new(),
        }
    }

    pub fn add_instance(&mut self, fields: Vec<FixField>) {
        self.instances.push(fields);
    }

    pub fn count(&self) -> usize {
        self.instances.len()
    }

    pub fn get_instance(&self, idx: usize) -> Option<&Vec<FixField>> {
        self.instances.get(idx)
    }

    pub fn get_field_in_instance(&self, idx: usize, tag: FixTag) -> Option<&FixField> {
        self.instances.get(idx)?.iter().find(|f| f.tag == tag)
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for instance in &self.instances {
            for field in instance {
                out.extend_from_slice(&field.encode());
            }
        }
        out
    }
}

#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Missing required field: tag {0}")]
    MissingField(FixTag),
    #[error("UTF-8 decode error for tag {0}")]
    Utf8Error(FixTag),
    #[error("Integer parse error for tag {0}")]
    ParseInt(FixTag),
    #[error("Float parse error for tag {0}")]
    ParseFloat(FixTag),
    #[error("Invalid field tag {0}: {1}")]
    InvalidField(FixTag, String),
    #[error("Missing checksum field (tag 10)")]
    MissingChecksum,
    #[error("Invalid checksum encoding")]
    InvalidChecksum,
    #[error("Checksum mismatch: computed {computed} declared {declared}")]
    ChecksumMismatch { computed: u8, declared: u8 },
    #[error("Type error: {0}")]
    TypeError(#[from] FixTypeError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MsgType;

    fn make_heartbeat() -> FixMessage {
        let mut msg = FixMessage::with_header("FIX.4.2", MsgType::Heartbeat.as_str(), "SENDER", "TARGET", 1);
        msg
    }

    #[test]
    fn test_encode_decode_checksum() {
        let msg = make_heartbeat();
        let wire = msg.encode();
        // Must not panic, checksum must validate
        FixMessage::validate_checksum(&wire).expect("checksum should be valid");
    }

    #[test]
    fn test_checksum_format() {
        assert_eq!(FixMessage::format_checksum(0), "000");
        assert_eq!(FixMessage::format_checksum(7), "007");
        assert_eq!(FixMessage::format_checksum(255), "255");
    }

    #[test]
    fn test_set_get_field() {
        let mut msg = FixMessage::new("FIX.4.2", "D");
        msg.set_field_str(tags::ClOrdID, "ORDER123");
        assert_eq!(msg.require_str(tags::ClOrdID).unwrap(), "ORDER123");
    }

    #[test]
    fn test_missing_field_error() {
        let msg = FixMessage::new("FIX.4.2", "D");
        assert!(matches!(msg.require_str(tags::ClOrdID), Err(MessageError::MissingField(11))));
    }

    #[test]
    fn test_checksum_mismatch() {
        let msg = make_heartbeat();
        let mut wire = msg.encode();
        // Corrupt a byte in the body
        if wire.len() > 10 {
            wire[10] ^= 0xFF;
        }
        assert!(FixMessage::validate_checksum(&wire).is_err());
    }

    #[test]
    fn test_remove_field() {
        let mut msg = FixMessage::new("FIX.4.2", "D");
        msg.set_field_str(tags::ClOrdID, "X1");
        msg.set_field_str(tags::Symbol, "AAPL");
        msg.remove_field(tags::ClOrdID);
        assert!(!msg.has_field(tags::ClOrdID));
        assert!(msg.has_field(tags::Symbol));
    }

    #[test]
    fn test_repeating_group() {
        let mut rg = RepeatingGroup::new(tags::MDEntryType);
        rg.add_instance(vec![
            FixField::new_str(tags::MDEntryType, "0"),
            FixField::new_str(tags::MDEntryPx, "100.50"),
        ]);
        rg.add_instance(vec![
            FixField::new_str(tags::MDEntryType, "1"),
            FixField::new_str(tags::MDEntryPx, "100.55"),
        ]);
        assert_eq!(rg.count(), 2);
        let f = rg.get_field_in_instance(1, tags::MDEntryPx).unwrap();
        assert_eq!(f.value_str().unwrap(), "100.55");
    }
}
