use crate::message::{FixMessage, tags};
use crate::types::{Qty, Side, OrdType, UtcTimestamp, MsgType};
use crate::codec::{CodecError, field_types, FixEncoder};

/// FIX OrderCancelRequest (F)
#[derive(Debug, Clone)]
pub struct OrderCancelRequest {
    pub orig_cl_ord_id: String,
    pub cl_ord_id: String,
    pub order_id: Option<String>,
    pub symbol: String,
    pub side: Side,
    pub transact_time: UtcTimestamp,
    pub order_qty: Option<Qty>,
    pub ord_type: Option<OrdType>,
    pub account: Option<String>,
    pub security_exchange: Option<String>,
    pub text: Option<String>,
}

impl OrderCancelRequest {
    pub fn new(
        orig_cl_ord_id: String,
        cl_ord_id: String,
        symbol: String,
        side: Side,
    ) -> Self {
        OrderCancelRequest {
            orig_cl_ord_id,
            cl_ord_id,
            order_id: None,
            symbol,
            side,
            transact_time: UtcTimestamp::now(),
            order_qty: None,
            ord_type: None,
            account: None,
            security_exchange: None,
            text: None,
        }
    }

    pub fn with_order_id(mut self, order_id: String) -> Self {
        self.order_id = Some(order_id);
        self
    }

    pub fn with_order_qty(mut self, qty: Qty) -> Self {
        self.order_qty = Some(qty);
        self
    }

    pub fn with_text(mut self, text: String) -> Self {
        self.text = Some(text);
        self
    }

    pub fn to_fix_message(&self, begin_string: &str, sender: &str, target: &str, seq: u32) -> FixMessage {
        let mut msg = FixMessage::with_header(
            begin_string,
            MsgType::OrderCancelRequest.as_str(),
            sender,
            target,
            seq,
        );
        msg.set_field_str(tags::OrigClOrdID, &self.orig_cl_ord_id);
        msg.set_field_str(tags::ClOrdID, &self.cl_ord_id);
        if let Some(ref oid) = self.order_id {
            msg.set_field_str(tags::OrderID, oid);
        }
        msg.set_field_str(tags::Symbol, &self.symbol);
        msg.set_field_str(tags::Side, &(self.side.to_fix_char() as char).to_string());
        msg.set_field_str(tags::TransactTime, &self.transact_time.to_fix_str());
        if let Some(qty) = self.order_qty {
            msg.set_field_str(tags::OrderQty, &FixEncoder::encode_qty(qty));
        }
        if let Some(ot) = self.ord_type {
            msg.set_field_str(tags::OrdType, &(ot.to_fix_char() as char).to_string());
        }
        if let Some(ref acc) = self.account {
            msg.set_field_str(tags::Account, acc);
        }
        if let Some(ref exch) = self.security_exchange {
            msg.set_field_str(tags::SecurityExchange, exch);
        }
        if let Some(ref text) = self.text {
            msg.set_field_str(tags::Text, text);
        }
        msg
    }

    pub fn from_fix_message(msg: &FixMessage) -> Result<Self, CodecError> {
        let orig_cl_ord_id = msg.require_str(tags::OrigClOrdID)
            .map_err(|_| CodecError::MissingField(tags::OrigClOrdID))?.to_string();
        let cl_ord_id = msg.require_str(tags::ClOrdID)
            .map_err(|_| CodecError::MissingField(tags::ClOrdID))?.to_string();
        let symbol = msg.require_str(tags::Symbol)
            .map_err(|_| CodecError::MissingField(tags::Symbol))?.to_string();
        let side = field_types::to_side(msg)?;
        let transact_time = field_types::to_timestamp(msg, tags::TransactTime)?;

        Ok(OrderCancelRequest {
            orig_cl_ord_id,
            cl_ord_id,
            order_id: msg.get_field(tags::OrderID).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            symbol,
            side,
            transact_time,
            order_qty: if msg.has_field(tags::OrderQty) { Some(field_types::to_qty(msg, tags::OrderQty)?) } else { None },
            ord_type: if msg.has_field(tags::OrdType) {
                let c = msg.require_char(tags::OrdType).map_err(|_| CodecError::MissingField(tags::OrdType))?;
                Some(crate::types::OrdType::from_fix_char(c).map_err(|e| CodecError::TypeConversion(e))?)
            } else { None },
            account: msg.get_field(tags::Account).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            security_exchange: msg.get_field(tags::SecurityExchange).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            text: msg.get_field(tags::Text).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Side;

    #[test]
    fn test_order_cancel_roundtrip() {
        let req = OrderCancelRequest::new(
            "ORIG001".to_string(),
            "CXL001".to_string(),
            "AAPL".to_string(),
            Side::Buy,
        )
        .with_order_id("SRV001".to_string())
        .with_text("Cancel due to strategy".to_string());

        let fix_msg = req.to_fix_message("FIX.4.2", "CLIENT", "BROKER", 5);
        let decoded = OrderCancelRequest::from_fix_message(&fix_msg).unwrap();

        assert_eq!(decoded.orig_cl_ord_id, "ORIG001");
        assert_eq!(decoded.cl_ord_id, "CXL001");
        assert_eq!(decoded.order_id.as_deref(), Some("SRV001"));
        assert_eq!(decoded.text.as_deref(), Some("Cancel due to strategy"));
    }
}
