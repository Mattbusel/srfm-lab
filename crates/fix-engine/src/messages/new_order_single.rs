use crate::message::{FixMessage, tags};
use crate::types::{Price, Qty, Side, OrdType, TimeInForce, UtcTimestamp, MsgType};
use crate::codec::{CodecError, field_types, FixEncoder};

/// FIX NewOrderSingle (D)
#[derive(Debug, Clone)]
pub struct NewOrderSingle {
    pub cl_ord_id: String,
    pub symbol: String,
    pub side: Side,
    pub order_qty: Qty,
    pub ord_type: OrdType,
    pub price: Option<Price>,
    pub stop_px: Option<Price>,
    pub time_in_force: Option<TimeInForce>,
    pub transact_time: UtcTimestamp,
    pub account: Option<String>,
    pub security_exchange: Option<String>,
    pub handl_inst: Option<u8>,
    pub exec_inst: Option<String>,
    pub currency: Option<String>,
    pub text: Option<String>,
    pub security_id: Option<String>,
    pub id_source: Option<String>,
    pub security_type: Option<String>,
    pub maturity_month_year: Option<String>,
    pub put_or_call: Option<u8>,
    pub strike_price: Option<Price>,
}

impl NewOrderSingle {
    pub fn new(
        cl_ord_id: String,
        symbol: String,
        side: Side,
        order_qty: Qty,
        ord_type: OrdType,
    ) -> Self {
        NewOrderSingle {
            cl_ord_id,
            symbol,
            side,
            order_qty,
            ord_type,
            price: None,
            stop_px: None,
            time_in_force: None,
            transact_time: UtcTimestamp::now(),
            account: None,
            security_exchange: None,
            handl_inst: None,
            exec_inst: None,
            currency: None,
            text: None,
            security_id: None,
            id_source: None,
            security_type: None,
            maturity_month_year: None,
            put_or_call: None,
            strike_price: None,
        }
    }

    pub fn with_limit_price(mut self, price: Price) -> Self {
        self.price = Some(price);
        self
    }

    pub fn with_stop_price(mut self, stop_px: Price) -> Self {
        self.stop_px = Some(stop_px);
        self
    }

    pub fn with_time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = Some(tif);
        self
    }

    pub fn with_account(mut self, account: String) -> Self {
        self.account = Some(account);
        self
    }

    pub fn with_exchange(mut self, exchange: String) -> Self {
        self.security_exchange = Some(exchange);
        self
    }

    pub fn with_currency(mut self, currency: String) -> Self {
        self.currency = Some(currency);
        self
    }

    pub fn with_text(mut self, text: String) -> Self {
        self.text = Some(text);
        self
    }

    /// Convert to FixMessage (without session header fields)
    pub fn to_fix_message(&self, begin_string: &str, sender: &str, target: &str, seq: u32) -> FixMessage {
        let mut msg = FixMessage::with_header(
            begin_string,
            MsgType::NewOrderSingle.as_str(),
            sender,
            target,
            seq,
        );
        // Required
        msg.set_field_str(tags::ClOrdID, &self.cl_ord_id);
        msg.set_field_str(tags::HandlInst, &self.handl_inst.unwrap_or(b'1').to_string());
        msg.set_field_str(tags::Symbol, &self.symbol);
        msg.set_field_str(tags::Side, &(self.side.to_fix_char() as char).to_string());
        msg.set_field_str(tags::TransactTime, &self.transact_time.to_fix_str());
        msg.set_field_str(tags::OrderQty, &FixEncoder::encode_qty(self.order_qty));
        msg.set_field_str(tags::OrdType, &(self.ord_type.to_fix_char() as char).to_string());

        // Optional
        if let Some(p) = self.price {
            msg.set_field_str(tags::Price, &FixEncoder::encode_price(p));
        }
        if let Some(s) = self.stop_px {
            msg.set_field_str(tags::StopPx, &FixEncoder::encode_price(s));
        }
        if let Some(tif) = self.time_in_force {
            msg.set_field_str(tags::TimeInForce, &(tif.to_fix_char() as char).to_string());
        }
        if let Some(ref acc) = self.account {
            msg.set_field_str(tags::Account, acc);
        }
        if let Some(ref exch) = self.security_exchange {
            msg.set_field_str(tags::SecurityExchange, exch);
        }
        if let Some(ref exec_inst) = self.exec_inst {
            msg.set_field_str(tags::ExecInst, exec_inst);
        }
        if let Some(ref cur) = self.currency {
            msg.set_field_str(tags::Currency, cur);
        }
        if let Some(ref text) = self.text {
            msg.set_field_str(tags::Text, text);
        }
        if let Some(ref sec_id) = self.security_id {
            msg.set_field_str(tags::SecurityID, sec_id);
        }
        if let Some(ref id_src) = self.id_source {
            msg.set_field_str(tags::IDSource, id_src);
        }
        if let Some(ref sec_type) = self.security_type {
            msg.set_field_str(tags::SecurityType, sec_type);
        }
        if let Some(ref mmy) = self.maturity_month_year {
            msg.set_field_str(tags::MaturityMonthYear, mmy);
        }
        if let Some(poc) = self.put_or_call {
            msg.set_field_str(tags::PutOrCall, &poc.to_string());
        }
        if let Some(strike) = self.strike_price {
            msg.set_field_str(tags::StrikePrice, &FixEncoder::encode_price(strike));
        }
        msg
    }

    /// Parse from FixMessage
    pub fn from_fix_message(msg: &FixMessage) -> Result<Self, CodecError> {
        let cl_ord_id = msg.require_str(tags::ClOrdID)
            .map_err(|_| CodecError::MissingField(tags::ClOrdID))?.to_string();
        let symbol = msg.require_str(tags::Symbol)
            .map_err(|_| CodecError::MissingField(tags::Symbol))?.to_string();
        let side = field_types::to_side(msg)?;
        let order_qty = field_types::to_qty(msg, tags::OrderQty)?;
        let ord_type = field_types::to_ord_type(msg)?;
        let transact_time = field_types::to_timestamp(msg, tags::TransactTime)?;

        let price = if msg.has_field(tags::Price) {
            Some(field_types::to_price(msg, tags::Price)?)
        } else {
            None
        };
        let stop_px = if msg.has_field(tags::StopPx) {
            Some(field_types::to_price(msg, tags::StopPx)?)
        } else {
            None
        };
        let time_in_force = if msg.has_field(tags::TimeInForce) {
            let c = msg.require_char(tags::TimeInForce).map_err(|_| CodecError::MissingField(tags::TimeInForce))?;
            Some(crate::types::TimeInForce::from_fix_char(c).map_err(|e| CodecError::TypeConversion(e))?)
        } else {
            None
        };

        Ok(NewOrderSingle {
            cl_ord_id,
            symbol,
            side,
            order_qty,
            ord_type,
            price,
            stop_px,
            time_in_force,
            transact_time,
            account: msg.get_field(tags::Account).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            security_exchange: msg.get_field(tags::SecurityExchange).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            handl_inst: msg.get_field(tags::HandlInst).and_then(|f| f.value_char().ok()),
            exec_inst: msg.get_field(tags::ExecInst).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            currency: msg.get_field(tags::Currency).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            text: msg.get_field(tags::Text).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            security_id: msg.get_field(tags::SecurityID).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            id_source: msg.get_field(tags::IDSource).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            security_type: msg.get_field(tags::SecurityType).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            maturity_month_year: msg.get_field(tags::MaturityMonthYear).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            put_or_call: msg.get_field(tags::PutOrCall).and_then(|f| f.value_char().ok()),
            strike_price: if msg.has_field(tags::StrikePrice) {
                Some(field_types::to_price(msg, tags::StrikePrice)?)
            } else {
                None
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Side, OrdType, TimeInForce, Price, Qty};

    fn make_nos() -> NewOrderSingle {
        NewOrderSingle::new(
            "ORD001".to_string(),
            "AAPL".to_string(),
            Side::Buy,
            Qty::from_f64(100.0),
            OrdType::Limit,
        )
        .with_limit_price(Price::from_f64(150.25))
        .with_time_in_force(TimeInForce::Day)
        .with_account("ACC001".to_string())
    }

    #[test]
    fn test_nos_roundtrip() {
        let nos = make_nos();
        let fix_msg = nos.to_fix_message("FIX.4.2", "SENDER", "TARGET", 1);
        let decoded = NewOrderSingle::from_fix_message(&fix_msg).unwrap();
        assert_eq!(decoded.cl_ord_id, "ORD001");
        assert_eq!(decoded.symbol, "AAPL");
        assert!(matches!(decoded.side, Side::Buy));
        assert!(matches!(decoded.ord_type, OrdType::Limit));
        assert!((decoded.price.unwrap().to_f64() - 150.25).abs() < 1e-6);
        assert_eq!(decoded.account.as_deref(), Some("ACC001"));
    }

    #[test]
    fn test_nos_missing_required() {
        let msg = FixMessage::new("FIX.4.2", "D");
        let result = NewOrderSingle::from_fix_message(&msg);
        assert!(result.is_err());
    }

    #[test]
    fn test_nos_market_order_no_price() {
        let nos = NewOrderSingle::new(
            "ORD002".to_string(),
            "MSFT".to_string(),
            Side::Sell,
            Qty::from_f64(50.0),
            OrdType::Market,
        );
        let fix_msg = nos.to_fix_message("FIX.4.2", "SENDER", "TARGET", 2);
        let decoded = NewOrderSingle::from_fix_message(&fix_msg).unwrap();
        assert!(decoded.price.is_none());
        assert!(matches!(decoded.ord_type, OrdType::Market));
    }
}
