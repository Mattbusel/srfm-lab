// execution_report_parser.rs -- Parse FIX ExecutionReport (MsgType=8) variants.
// Maps ExecType tag values to structured Rust enums and extracts fill data.

use thiserror::Error;
use crate::message::{FixMessage, tags};
use crate::types::ExecType;

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Missing required field tag {0}")]
    MissingField(u32),
    #[error("Invalid field value for tag {tag}: {msg}")]
    InvalidField { tag: u32, msg: String },
    #[error("Unknown ExecType: '{0}'")]
    UnknownExecType(String),
    #[error("Message type is not an ExecutionReport: got '{0}'")]
    WrongMsgType(String),
    #[error("Field parse error: {0}")]
    FieldError(#[from] crate::message::MessageError),
}

// ---------------------------------------------------------------------------
// ExecReport
// ---------------------------------------------------------------------------

/// Strongly typed representation of every ExecutionReport variant.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecReport {
    NewOrderAck {
        cl_ord_id: String,
        order_id: String,
        symbol: String,
        /// FIX Side char: '1'=Buy, '2'=Sell, etc.
        side: char,
        qty: f64,
    },
    PartialFill {
        cl_ord_id: String,
        order_id: String,
        symbol: String,
        side: char,
        last_qty: f64,
        last_px: f64,
        cum_qty: f64,
        avg_px: f64,
    },
    FullFill {
        cl_ord_id: String,
        order_id: String,
        symbol: String,
        side: char,
        last_qty: f64,
        last_px: f64,
        cum_qty: f64,
        avg_px: f64,
    },
    CancelAck {
        cl_ord_id: String,
        orig_cl_ord_id: String,
        symbol: String,
    },
    Reject {
        cl_ord_id: String,
        reject_reason: String,
        text: String,
    },
}

// ---------------------------------------------------------------------------
// FillInfo
// ---------------------------------------------------------------------------

/// Extracted fill data when an execution report contains a fill.
#[derive(Debug, Clone, PartialEq)]
pub struct FillInfo {
    pub last_qty: f64,
    pub last_px: f64,
    pub cum_qty: f64,
    pub avg_px: f64,
    /// Optional venue identifier (tag 207 SecurityExchange or similar).
    pub venue: Option<String>,
}

// ---------------------------------------------------------------------------
// classify_exec_type
// ---------------------------------------------------------------------------

/// Map a FIX ExecType string value (tag 150) to the typed ExecType enum.
/// Returns ExecType::OrderStatus as a fallback for unknown values.
pub fn classify_exec_type(exec_type: &str) -> ExecType {
    let s = exec_type.trim();
    if s.len() == 1 {
        let c = s.as_bytes()[0];
        ExecType::from_fix_char(c).unwrap_or(ExecType::OrderStatus)
    } else {
        // Some venues send textual descriptions
        match s.to_ascii_uppercase().as_str() {
            "NEW"           => ExecType::New,
            "PARTIAL_FILL" | "PARTIALFILL" => ExecType::PartialFill,
            "FILL"          => ExecType::Fill,
            "DONE_FOR_DAY"  => ExecType::DoneForDay,
            "CANCELED" | "CANCELLED" => ExecType::Canceled,
            "REPLACE"       => ExecType::Replace,
            "PENDING_CANCEL" => ExecType::PendingCancel,
            "STOPPED"       => ExecType::Stopped,
            "REJECTED"      => ExecType::Rejected,
            "SUSPENDED"     => ExecType::Suspended,
            "PENDING_NEW"   => ExecType::PendingNew,
            "CALCULATED"    => ExecType::Calculated,
            "EXPIRED"       => ExecType::Expired,
            "RESTATED"      => ExecType::Restated,
            "PENDING_REPLACE" => ExecType::PendingReplace,
            "TRADE"         => ExecType::Trade,
            "TRADE_CORRECT" => ExecType::TradeCorrect,
            "TRADE_CANCEL"  => ExecType::TradeCancel,
            _               => ExecType::OrderStatus,
        }
    }
}

// ---------------------------------------------------------------------------
// extract_fill
// ---------------------------------------------------------------------------

/// Extract fill data from a FIX message if both LastQty and LastPx are present.
/// Returns None if the message contains no fill fields.
pub fn extract_fill(msg: &FixMessage) -> Option<FillInfo> {
    let last_qty = msg.get_field(tags::LastQty)?.value_f64().ok()?;
    let last_px = msg.get_field(tags::LastPx)?.value_f64().ok()?;

    let cum_qty = msg.get_field(tags::CumQty)
        .and_then(|f| f.value_f64().ok())
        .unwrap_or(last_qty);

    let avg_px = msg.get_field(tags::AvgPx)
        .and_then(|f| f.value_f64().ok())
        .unwrap_or(last_px);

    let venue = msg.get_field(tags::SecurityExchange)
        .and_then(|f| f.value_str().ok())
        .map(|s| s.to_string());

    Some(FillInfo { last_qty, last_px, cum_qty, avg_px, venue })
}

// ---------------------------------------------------------------------------
// parse_exec_report
// ---------------------------------------------------------------------------

/// Parse a FixMessage into a strongly typed ExecReport.
///
/// Required tags: 35 (MsgType = "8"), 150 (ExecType), 11 (ClOrdID).
/// Additional tags required depend on ExecType:
///   New/fills: 37 (OrderID), 55 (Symbol), 54 (Side), 38 (OrderQty)
///   Fills:     32 (LastQty), 31 (LastPx), 14 (CumQty), 6 (AvgPx)
///   Cancel:    41 (OrigClOrdID), 55 (Symbol)
///   Reject:    58 (Text), 103 (OrdRejReason)
pub fn parse_exec_report(msg: &FixMessage) -> Result<ExecReport, ParseError> {
    // Validate message type
    if msg.msg_type != "8" {
        return Err(ParseError::WrongMsgType(msg.msg_type.clone()));
    }

    let exec_type_str = msg.require_str(tags::ExecType)?;
    let exec_type = classify_exec_type(exec_type_str);

    let cl_ord_id = msg.require_str(tags::ClOrdID)?.to_string();

    match exec_type {
        ExecType::New | ExecType::PendingNew => {
            let order_id = msg.get_field(tags::OrderID)
                .and_then(|f| f.value_str().ok())
                .unwrap_or("NONE")
                .to_string();
            let symbol = msg.require_str(tags::Symbol)?.to_string();
            let side = msg.require_char(tags::Side)? as char;
            let qty = msg.require_f64(tags::OrderQty)?;
            Ok(ExecReport::NewOrderAck { cl_ord_id, order_id, symbol, side, qty })
        }

        ExecType::PartialFill | ExecType::Trade => {
            let order_id = msg.require_str(tags::OrderID)?.to_string();
            let symbol = msg.require_str(tags::Symbol)?.to_string();
            let side = msg.require_char(tags::Side)? as char;
            let last_qty = msg.require_f64(tags::LastQty)?;
            let last_px = msg.require_f64(tags::LastPx)?;
            let cum_qty = msg.require_f64(tags::CumQty)?;
            let avg_px = msg.require_f64(tags::AvgPx)?;
            // Determine if this is a full fill by checking OrdStatus
            let ord_status = msg.get_field(tags::OrdStatus)
                .and_then(|f| f.value_char().ok())
                .unwrap_or(b'1');
            if ord_status == b'2' {
                // OrdStatus = Filled
                Ok(ExecReport::FullFill { cl_ord_id, order_id, symbol, side, last_qty, last_px, cum_qty, avg_px })
            } else {
                Ok(ExecReport::PartialFill { cl_ord_id, order_id, symbol, side, last_qty, last_px, cum_qty, avg_px })
            }
        }

        ExecType::Fill => {
            let order_id = msg.require_str(tags::OrderID)?.to_string();
            let symbol = msg.require_str(tags::Symbol)?.to_string();
            let side = msg.require_char(tags::Side)? as char;
            let last_qty = msg.require_f64(tags::LastQty)?;
            let last_px = msg.require_f64(tags::LastPx)?;
            let cum_qty = msg.require_f64(tags::CumQty)?;
            let avg_px = msg.require_f64(tags::AvgPx)?;
            Ok(ExecReport::FullFill { cl_ord_id, order_id, symbol, side, last_qty, last_px, cum_qty, avg_px })
        }

        ExecType::Canceled => {
            let orig_cl_ord_id = msg.get_field(tags::OrigClOrdID)
                .and_then(|f| f.value_str().ok())
                .unwrap_or("")
                .to_string();
            let symbol = msg.require_str(tags::Symbol)?.to_string();
            Ok(ExecReport::CancelAck { cl_ord_id, orig_cl_ord_id, symbol })
        }

        ExecType::Rejected => {
            let reject_reason = msg.get_field(tags::OrdRejReason)
                .and_then(|f| f.value_str().ok())
                .unwrap_or("0")
                .to_string();
            let text = msg.get_field(tags::Text)
                .and_then(|f| f.value_str().ok())
                .unwrap_or("")
                .to_string();
            Ok(ExecReport::Reject { cl_ord_id, reject_reason, text })
        }

        _ => {
            // For order status, restated, expired, etc. -- treat as NewOrderAck if
            // OrderID is present, otherwise return a minimal Reject variant
            if let Some(order_id_f) = msg.get_field(tags::OrderID) {
                let order_id = order_id_f.value_str().unwrap_or("").to_string();
                let symbol = msg.get_field(tags::Symbol)
                    .and_then(|f| f.value_str().ok())
                    .unwrap_or("")
                    .to_string();
                let side = msg.get_field(tags::Side)
                    .and_then(|f| f.value_char().ok())
                    .unwrap_or(b'1') as char;
                let qty = msg.get_field(tags::OrderQty)
                    .and_then(|f| f.value_f64().ok())
                    .unwrap_or(0.0);
                Ok(ExecReport::NewOrderAck { cl_ord_id, order_id, symbol, side, qty })
            } else {
                Ok(ExecReport::Reject {
                    cl_ord_id,
                    reject_reason: "0".to_string(),
                    text: format!("Unhandled ExecType: {}", exec_type_str),
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{FixMessage, tags};

    fn exec_report_base(exec_type: &str, cl_ord_id: &str) -> FixMessage {
        let mut msg = FixMessage::new("FIX.4.2", "8");
        msg.set_field_str(tags::ExecType, exec_type);
        msg.set_field_str(tags::ClOrdID, cl_ord_id);
        msg
    }

    fn add_order_fields(msg: &mut FixMessage, order_id: &str, symbol: &str, side: char, qty: f64) {
        msg.set_field_str(tags::OrderID, order_id);
        msg.set_field_str(tags::Symbol, symbol);
        msg.set_field_str(tags::Side, &side.to_string());
        msg.set_field_str(tags::OrderQty, &qty.to_string());
    }

    fn add_fill_fields(msg: &mut FixMessage, last_qty: f64, last_px: f64, cum_qty: f64, avg_px: f64) {
        msg.set_field_str(tags::LastQty, &last_qty.to_string());
        msg.set_field_str(tags::LastPx, &last_px.to_string());
        msg.set_field_str(tags::CumQty, &cum_qty.to_string());
        msg.set_field_str(tags::AvgPx, &avg_px.to_string());
    }

    // ---------------------------------------------------------------------------
    // classify_exec_type
    // ---------------------------------------------------------------------------

    #[test]
    fn test_classify_exec_type_single_char() {
        assert_eq!(classify_exec_type("0"), ExecType::New);
        assert_eq!(classify_exec_type("1"), ExecType::PartialFill);
        assert_eq!(classify_exec_type("2"), ExecType::Fill);
        assert_eq!(classify_exec_type("4"), ExecType::Canceled);
        assert_eq!(classify_exec_type("8"), ExecType::Rejected);
        assert_eq!(classify_exec_type("F"), ExecType::Trade);
    }

    #[test]
    fn test_classify_exec_type_text() {
        assert_eq!(classify_exec_type("NEW"), ExecType::New);
        assert_eq!(classify_exec_type("FILL"), ExecType::Fill);
        assert_eq!(classify_exec_type("CANCELLED"), ExecType::Canceled);
        assert_eq!(classify_exec_type("TRADE"), ExecType::Trade);
    }

    #[test]
    fn test_classify_exec_type_unknown_fallback() {
        assert_eq!(classify_exec_type("ZZZZ"), ExecType::OrderStatus);
        assert_eq!(classify_exec_type("Z"), ExecType::OrderStatus);
    }

    // ---------------------------------------------------------------------------
    // parse_exec_report
    // ---------------------------------------------------------------------------

    #[test]
    fn test_parse_new_order_ack() {
        let mut msg = exec_report_base("0", "ORD-001");
        add_order_fields(&mut msg, "SERVER-001", "AAPL", '1', 100.0);
        let report = parse_exec_report(&msg).unwrap();
        match report {
            ExecReport::NewOrderAck { cl_ord_id, order_id, symbol, side, qty } => {
                assert_eq!(cl_ord_id, "ORD-001");
                assert_eq!(order_id, "SERVER-001");
                assert_eq!(symbol, "AAPL");
                assert_eq!(side, '1');
                assert!((qty - 100.0).abs() < 1e-9);
            }
            other => panic!("Expected NewOrderAck, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_partial_fill() {
        let mut msg = exec_report_base("1", "ORD-002");
        add_order_fields(&mut msg, "SRV-002", "MSFT", '1', 200.0);
        add_fill_fields(&mut msg, 50.0, 299.0, 50.0, 299.0);
        msg.set_field_str(tags::OrdStatus, "1"); // PartiallyFilled
        let report = parse_exec_report(&msg).unwrap();
        match report {
            ExecReport::PartialFill { last_qty, last_px, cum_qty, .. } => {
                assert!((last_qty - 50.0).abs() < 1e-9);
                assert!((last_px - 299.0).abs() < 1e-9);
                assert!((cum_qty - 50.0).abs() < 1e-9);
            }
            other => panic!("Expected PartialFill, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_full_fill() {
        let mut msg = exec_report_base("2", "ORD-003");
        add_order_fields(&mut msg, "SRV-003", "GOOG", '2', 50.0);
        add_fill_fields(&mut msg, 50.0, 2800.0, 50.0, 2800.0);
        let report = parse_exec_report(&msg).unwrap();
        match report {
            ExecReport::FullFill { cum_qty, avg_px, .. } => {
                assert!((cum_qty - 50.0).abs() < 1e-9);
                assert!((avg_px - 2800.0).abs() < 1e-9);
            }
            other => panic!("Expected FullFill, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_cancel_ack() {
        let mut msg = exec_report_base("4", "ORD-004");
        msg.set_field_str(tags::OrigClOrdID, "ORD-003");
        msg.set_field_str(tags::Symbol, "TSLA");
        let report = parse_exec_report(&msg).unwrap();
        match report {
            ExecReport::CancelAck { cl_ord_id, orig_cl_ord_id, symbol } => {
                assert_eq!(cl_ord_id, "ORD-004");
                assert_eq!(orig_cl_ord_id, "ORD-003");
                assert_eq!(symbol, "TSLA");
            }
            other => panic!("Expected CancelAck, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_reject() {
        let mut msg = exec_report_base("8", "ORD-005");
        msg.set_field_str(tags::OrdRejReason, "3");
        msg.set_field_str(tags::Text, "Order size exceeds limit");
        let report = parse_exec_report(&msg).unwrap();
        match report {
            ExecReport::Reject { reject_reason, text, .. } => {
                assert_eq!(reject_reason, "3");
                assert!(text.contains("size"));
            }
            other => panic!("Expected Reject, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_wrong_msg_type_fails() {
        let msg = FixMessage::new("FIX.4.2", "D"); // NewOrderSingle, not ExecutionReport
        let result = parse_exec_report(&msg);
        assert!(matches!(result, Err(ParseError::WrongMsgType(_))));
    }

    // ---------------------------------------------------------------------------
    // extract_fill
    // ---------------------------------------------------------------------------

    #[test]
    fn test_extract_fill_present() {
        let mut msg = exec_report_base("1", "ORD-006");
        add_fill_fields(&mut msg, 25.0, 150.5, 25.0, 150.5);
        msg.set_field_str(tags::SecurityExchange, "XNAS");
        let fill = extract_fill(&msg).unwrap();
        assert!((fill.last_qty - 25.0).abs() < 1e-9);
        assert!((fill.last_px - 150.5).abs() < 1e-9);
        assert_eq!(fill.venue, Some("XNAS".to_string()));
    }

    #[test]
    fn test_extract_fill_absent_returns_none() {
        let msg = exec_report_base("0", "ORD-007");
        assert!(extract_fill(&msg).is_none());
    }

    #[test]
    fn test_extract_fill_no_venue() {
        let mut msg = exec_report_base("2", "ORD-008");
        add_fill_fields(&mut msg, 100.0, 50.0, 100.0, 50.0);
        let fill = extract_fill(&msg).unwrap();
        assert!(fill.venue.is_none());
    }
}
