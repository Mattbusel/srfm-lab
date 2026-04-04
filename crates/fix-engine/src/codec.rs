use crate::message::{FixMessage, FixField, tags, SOH};
use crate::parser::{FixParser, ParseError};
use crate::types::{Price, Qty, UtcTimestamp, FixTypeError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),
    #[error("Type conversion error: {0}")]
    TypeConversion(#[from] FixTypeError),
    #[error("Missing required field: tag {0}")]
    MissingField(u32),
    #[error("Encode error: {0}")]
    Encode(String),
}

/// FIX message encoder
pub struct FixEncoder;

impl FixEncoder {
    /// Encode a FixMessage to wire bytes with proper BodyLength and CheckSum
    pub fn encode(msg: &FixMessage) -> Vec<u8> {
        msg.encode()
    }

    /// Encode a batch of messages
    pub fn encode_batch(msgs: &[FixMessage]) -> Vec<u8> {
        let mut out = Vec::new();
        for msg in msgs {
            out.extend_from_slice(&msg.encode());
        }
        out
    }

    /// Encode a Price field value
    pub fn encode_price(price: Price) -> String {
        // FIX prices typically use minimal decimal places
        let v = price.to_f64();
        // Trim trailing zeros
        let s = format!("{:.8}", v);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        if s.is_empty() { "0".to_string() } else { s.to_string() }
    }

    /// Encode a Qty field value
    pub fn encode_qty(qty: Qty) -> String {
        let whole = qty.0 / Qty::SCALE;
        let frac = qty.0.abs() % Qty::SCALE;
        if frac == 0 {
            whole.to_string()
        } else {
            format!("{}.{:04}", whole, frac).trim_end_matches('0').to_string()
        }
    }

    /// Encode a UTCTimestamp
    pub fn encode_timestamp(ts: &UtcTimestamp) -> String {
        ts.to_fix_str()
    }

    /// Build the raw FIX wire string (pipe-delimited for display)
    pub fn to_display_string(msg: &FixMessage) -> String {
        msg.to_string()
    }
}

/// FIX message decoder
pub struct FixDecoder {
    parser: FixParser,
}

impl FixDecoder {
    pub fn new() -> Self {
        FixDecoder {
            parser: FixParser::new(),
        }
    }

    /// Decode a single FIX message from wire bytes
    pub fn decode(&self, data: &[u8]) -> Result<FixMessage, CodecError> {
        let (msg, _) = self.parser.parse(data)?;
        Ok(msg)
    }

    /// Decode a Price from a FIX field value string
    pub fn decode_price(s: &str) -> Result<Price, CodecError> {
        let v: f64 = s.parse().map_err(|_| FixTypeError::ParseError(format!("invalid price: {}", s)))?;
        Ok(Price::from_f64(v))
    }

    /// Decode a Qty from a FIX field value string
    pub fn decode_qty(s: &str) -> Result<Qty, CodecError> {
        let v: f64 = s.parse().map_err(|_| FixTypeError::ParseError(format!("invalid qty: {}", s)))?;
        Ok(Qty::from_f64(v))
    }

    /// Decode a UtcTimestamp from FIX format
    pub fn decode_timestamp(s: &str) -> Result<UtcTimestamp, CodecError> {
        Ok(UtcTimestamp::from_fix_str(s)?)
    }

    /// Decode a boolean from Y/N char
    pub fn decode_bool(c: u8) -> bool {
        c == b'Y'
    }

    /// Extract a tag's value as a string slice from wire bytes (fast, no full parse)
    pub fn peek_tag<'a>(data: &'a [u8], tag: u32) -> Option<&'a str> {
        FixParser::extract_field(data, tag)
            .and_then(|b| std::str::from_utf8(b).ok())
    }

    /// Validate a complete FIX message's checksum
    pub fn validate_checksum(data: &[u8]) -> Result<(), CodecError> {
        crate::message::FixMessage::validate_checksum(data)
            .map_err(|e| CodecError::Encode(e.to_string()))
    }
}

impl Default for FixDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Field type conversion utilities
pub mod field_types {
    use super::*;
    use crate::types::{Side, OrdType, OrdStatus, ExecType, TimeInForce, MsgType};

    pub fn to_price(msg: &FixMessage, tag: u32) -> Result<Price, CodecError> {
        let s = msg.require_str(tag)
            .map_err(|_| CodecError::MissingField(tag))?;
        FixDecoder::decode_price(s)
    }

    pub fn to_qty(msg: &FixMessage, tag: u32) -> Result<Qty, CodecError> {
        let s = msg.require_str(tag)
            .map_err(|_| CodecError::MissingField(tag))?;
        FixDecoder::decode_qty(s)
    }

    pub fn to_timestamp(msg: &FixMessage, tag: u32) -> Result<UtcTimestamp, CodecError> {
        let s = msg.require_str(tag)
            .map_err(|_| CodecError::MissingField(tag))?;
        FixDecoder::decode_timestamp(s)
    }

    pub fn to_side(msg: &FixMessage) -> Result<Side, CodecError> {
        let c = msg.require_char(tags::Side)
            .map_err(|_| CodecError::MissingField(tags::Side))?;
        Side::from_fix_char(c).map_err(|e| CodecError::TypeConversion(e))
    }

    pub fn to_ord_type(msg: &FixMessage) -> Result<OrdType, CodecError> {
        let c = msg.require_char(tags::OrdType)
            .map_err(|_| CodecError::MissingField(tags::OrdType))?;
        OrdType::from_fix_char(c).map_err(|e| CodecError::TypeConversion(e))
    }

    pub fn to_ord_status(msg: &FixMessage) -> Result<OrdStatus, CodecError> {
        let c = msg.require_char(tags::OrdStatus)
            .map_err(|_| CodecError::MissingField(tags::OrdStatus))?;
        OrdStatus::from_fix_char(c).map_err(|e| CodecError::TypeConversion(e))
    }

    pub fn to_exec_type(msg: &FixMessage) -> Result<ExecType, CodecError> {
        let c = msg.require_char(tags::ExecType)
            .map_err(|_| CodecError::MissingField(tags::ExecType))?;
        ExecType::from_fix_char(c).map_err(|e| CodecError::TypeConversion(e))
    }

    pub fn to_time_in_force(msg: &FixMessage) -> Result<TimeInForce, CodecError> {
        let c = msg.require_char(tags::TimeInForce)
            .map_err(|_| CodecError::MissingField(tags::TimeInForce))?;
        TimeInForce::from_fix_char(c).map_err(|e| CodecError::TypeConversion(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::FixMessage;
    use crate::types::MsgType;

    fn make_new_order_single() -> FixMessage {
        let mut msg = FixMessage::with_header("FIX.4.2", MsgType::NewOrderSingle.as_str(), "SENDER", "TARGET", 1);
        msg.set_field_str(tags::ClOrdID, "ORD001");
        msg.set_field_str(tags::Symbol, "AAPL");
        msg.set_field_str(tags::Side, "1");
        msg.set_field_str(tags::OrderQty, "100");
        msg.set_field_str(tags::OrdType, "2");
        msg.set_field_str(tags::Price, "150.00");
        msg.set_field_str(tags::TransactTime, "20240101-12:00:00.000");
        msg
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let msg = make_new_order_single();
        let wire = FixEncoder::encode(&msg);
        let decoder = FixDecoder::new();
        let decoded = decoder.decode(&wire).unwrap();
        assert_eq!(decoded.require_str(tags::ClOrdID).unwrap(), "ORD001");
        assert_eq!(decoded.require_str(tags::Symbol).unwrap(), "AAPL");
    }

    #[test]
    fn test_checksum_validation() {
        let msg = make_new_order_single();
        let wire = FixEncoder::encode(&msg);
        FixDecoder::validate_checksum(&wire).expect("checksum must be valid");
    }

    #[test]
    fn test_price_encode() {
        let p = Price::from_f64(150.0);
        assert_eq!(FixEncoder::encode_price(p), "150");
        let p2 = Price::from_f64(100.25);
        assert_eq!(FixEncoder::encode_price(p2), "100.25");
    }

    #[test]
    fn test_qty_encode() {
        let q = Qty::from_f64(100.0);
        assert_eq!(FixEncoder::encode_qty(q), "100");
        let q2 = Qty::from_f64(50.5);
        let s = FixEncoder::encode_qty(q2);
        assert!(s.starts_with("50.5"));
    }

    #[test]
    fn test_decode_price() {
        let p = FixDecoder::decode_price("150.25").unwrap();
        assert!((p.to_f64() - 150.25).abs() < 1e-6);
    }

    #[test]
    fn test_decode_qty() {
        let q = FixDecoder::decode_qty("1000").unwrap();
        assert!((q.to_f64() - 1000.0).abs() < 1e-4);
    }

    #[test]
    fn test_peek_tag() {
        let msg = make_new_order_single();
        let wire = FixEncoder::encode(&msg);
        let symbol = FixDecoder::peek_tag(&wire, tags::Symbol);
        assert_eq!(symbol, Some("AAPL"));
    }

    #[test]
    fn test_field_type_conversions() {
        let msg = make_new_order_single();
        let side = field_types::to_side(&msg).unwrap();
        assert_eq!(side, crate::types::Side::Buy);
        let ord_type = field_types::to_ord_type(&msg).unwrap();
        assert_eq!(ord_type, crate::types::OrdType::Limit);
        let price = field_types::to_price(&msg, tags::Price).unwrap();
        assert!((price.to_f64() - 150.0).abs() < 1e-6);
        let qty = field_types::to_qty(&msg, tags::OrderQty).unwrap();
        assert!((qty.to_f64() - 100.0).abs() < 1e-4);
    }

    #[test]
    fn test_batch_encode() {
        let msg1 = make_new_order_single();
        let mut msg2 = make_new_order_single();
        msg2.set_field_str(tags::ClOrdID, "ORD002");
        let batch = FixEncoder::encode_batch(&[msg1, msg2]);
        // Should be parseable as two messages
        let parser = FixParser::new();
        let (msgs, consumed) = parser.parse_stream(&batch);
        assert_eq!(msgs.len(), 2);
        assert_eq!(consumed, batch.len());
    }
}
