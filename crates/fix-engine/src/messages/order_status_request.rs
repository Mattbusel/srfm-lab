use crate::message::{FixMessage, tags};
use crate::types::{Side, MsgType};
use crate::codec::{CodecError, field_types};

/// FIX OrderStatusRequest (H)
#[derive(Debug, Clone)]
pub struct OrderStatusRequest {
    pub cl_ord_id: String,
    pub order_id: Option<String>,
    pub symbol: String,
    pub side: Side,
    pub account: Option<String>,
}

impl OrderStatusRequest {
    pub fn new(cl_ord_id: String, symbol: String, side: Side) -> Self {
        OrderStatusRequest {
            cl_ord_id,
            order_id: None,
            symbol,
            side,
            account: None,
        }
    }

    pub fn with_order_id(mut self, oid: String) -> Self {
        self.order_id = Some(oid);
        self
    }

    pub fn with_account(mut self, account: String) -> Self {
        self.account = Some(account);
        self
    }

    pub fn to_fix_message(&self, begin_string: &str, sender: &str, target: &str, seq: u32) -> FixMessage {
        let mut msg = FixMessage::with_header(
            begin_string,
            MsgType::OrderStatusRequest.as_str(),
            sender,
            target,
            seq,
        );
        msg.set_field_str(tags::ClOrdID, &self.cl_ord_id);
        if let Some(ref oid) = self.order_id {
            msg.set_field_str(tags::OrderID, oid);
        }
        msg.set_field_str(tags::Symbol, &self.symbol);
        msg.set_field_str(tags::Side, &(self.side.to_fix_char() as char).to_string());
        if let Some(ref acc) = self.account {
            msg.set_field_str(tags::Account, acc);
        }
        msg
    }

    pub fn from_fix_message(msg: &FixMessage) -> Result<Self, CodecError> {
        let cl_ord_id = msg.require_str(tags::ClOrdID)
            .map_err(|_| CodecError::MissingField(tags::ClOrdID))?.to_string();
        let symbol = msg.require_str(tags::Symbol)
            .map_err(|_| CodecError::MissingField(tags::Symbol))?.to_string();
        let side = field_types::to_side(msg)?;

        Ok(OrderStatusRequest {
            cl_ord_id,
            order_id: msg.get_field(tags::OrderID).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            symbol,
            side,
            account: msg.get_field(tags::Account).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Side;

    #[test]
    fn test_order_status_request_roundtrip() {
        let req = OrderStatusRequest::new(
            "C001".to_string(),
            "TSLA".to_string(),
            Side::Buy,
        ).with_order_id("ORD999".to_string());

        let fix_msg = req.to_fix_message("FIX.4.2", "CLIENT", "BROKER", 3);
        let decoded = OrderStatusRequest::from_fix_message(&fix_msg).unwrap();
        assert_eq!(decoded.cl_ord_id, "C001");
        assert_eq!(decoded.order_id.as_deref(), Some("ORD999"));
    }
}
