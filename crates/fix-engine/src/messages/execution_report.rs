use crate::message::{FixMessage, tags};
use crate::types::{Price, Qty, Side, OrdType, OrdStatus, ExecType, TimeInForce, UtcTimestamp, MsgType};
use crate::codec::{CodecError, field_types, FixEncoder};

/// FIX ExecutionReport (8)
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub order_id: String,
    pub cl_ord_id: String,
    pub orig_cl_ord_id: Option<String>,
    pub exec_id: String,
    pub exec_type: ExecType,
    pub ord_status: OrdStatus,
    pub symbol: String,
    pub side: Side,
    pub leaves_qty: Qty,
    pub cum_qty: Qty,
    pub avg_px: Price,
    pub order_qty: Option<Qty>,
    pub price: Option<Price>,
    pub stop_px: Option<Price>,
    pub last_qty: Option<Qty>,
    pub last_px: Option<Price>,
    pub ord_type: Option<OrdType>,
    pub time_in_force: Option<TimeInForce>,
    pub transact_time: UtcTimestamp,
    pub text: Option<String>,
    pub ord_rej_reason: Option<u32>,
    pub account: Option<String>,
    pub security_exchange: Option<String>,
    pub exec_inst: Option<String>,
    pub currency: Option<String>,
    pub security_id: Option<String>,
    pub id_source: Option<String>,
    pub security_type: Option<String>,
    pub maturity_month_year: Option<String>,
    pub put_or_call: Option<u8>,
    pub strike_price: Option<Price>,
}

impl ExecutionReport {
    pub fn new(
        order_id: String,
        cl_ord_id: String,
        exec_id: String,
        exec_type: ExecType,
        ord_status: OrdStatus,
        symbol: String,
        side: Side,
        leaves_qty: Qty,
        cum_qty: Qty,
        avg_px: Price,
    ) -> Self {
        ExecutionReport {
            order_id,
            cl_ord_id,
            orig_cl_ord_id: None,
            exec_id,
            exec_type,
            ord_status,
            symbol,
            side,
            leaves_qty,
            cum_qty,
            avg_px,
            order_qty: None,
            price: None,
            stop_px: None,
            last_qty: None,
            last_px: None,
            ord_type: None,
            time_in_force: None,
            transact_time: UtcTimestamp::now(),
            text: None,
            ord_rej_reason: None,
            account: None,
            security_exchange: None,
            exec_inst: None,
            currency: None,
            security_id: None,
            id_source: None,
            security_type: None,
            maturity_month_year: None,
            put_or_call: None,
            strike_price: None,
        }
    }

    pub fn with_fill(mut self, last_qty: Qty, last_px: Price) -> Self {
        self.last_qty = Some(last_qty);
        self.last_px = Some(last_px);
        self
    }

    pub fn with_order_details(mut self, order_qty: Qty, ord_type: OrdType, price: Option<Price>) -> Self {
        self.order_qty = Some(order_qty);
        self.ord_type = Some(ord_type);
        self.price = price;
        self
    }

    pub fn with_text(mut self, text: String) -> Self {
        self.text = Some(text);
        self
    }

    pub fn with_reject_reason(mut self, reason: u32) -> Self {
        self.ord_rej_reason = Some(reason);
        self
    }

    pub fn with_orig_cl_ord_id(mut self, orig: String) -> Self {
        self.orig_cl_ord_id = Some(orig);
        self
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.ord_status,
            OrdStatus::Filled | OrdStatus::Canceled | OrdStatus::Rejected | OrdStatus::Expired | OrdStatus::DoneForDay
        )
    }

    pub fn is_filled(&self) -> bool {
        matches!(self.ord_status, OrdStatus::Filled)
    }

    pub fn is_partial(&self) -> bool {
        matches!(self.ord_status, OrdStatus::PartiallyFilled)
    }

    pub fn to_fix_message(&self, begin_string: &str, sender: &str, target: &str, seq: u32) -> FixMessage {
        let mut msg = FixMessage::with_header(
            begin_string,
            MsgType::ExecutionReport.as_str(),
            sender,
            target,
            seq,
        );
        msg.set_field_str(tags::OrderID, &self.order_id);
        msg.set_field_str(tags::ClOrdID, &self.cl_ord_id);
        if let Some(ref orig) = self.orig_cl_ord_id {
            msg.set_field_str(tags::OrigClOrdID, orig);
        }
        msg.set_field_str(tags::ExecID, &self.exec_id);
        msg.set_field_str(tags::ExecType, &(self.exec_type.to_fix_char() as char).to_string());
        msg.set_field_str(tags::OrdStatus, &(self.ord_status.to_fix_char() as char).to_string());
        msg.set_field_str(tags::Symbol, &self.symbol);
        msg.set_field_str(tags::Side, &(self.side.to_fix_char() as char).to_string());
        msg.set_field_str(tags::LeavesQty, &FixEncoder::encode_qty(self.leaves_qty));
        msg.set_field_str(tags::CumQty, &FixEncoder::encode_qty(self.cum_qty));
        msg.set_field_str(tags::AvgPx, &FixEncoder::encode_price(self.avg_px));
        msg.set_field_str(tags::TransactTime, &self.transact_time.to_fix_str());

        if let Some(qty) = self.order_qty {
            msg.set_field_str(tags::OrderQty, &FixEncoder::encode_qty(qty));
        }
        if let Some(p) = self.price {
            msg.set_field_str(tags::Price, &FixEncoder::encode_price(p));
        }
        if let Some(s) = self.stop_px {
            msg.set_field_str(tags::StopPx, &FixEncoder::encode_price(s));
        }
        if let Some(lq) = self.last_qty {
            msg.set_field_str(tags::LastQty, &FixEncoder::encode_qty(lq));
        }
        if let Some(lp) = self.last_px {
            msg.set_field_str(tags::LastPx, &FixEncoder::encode_price(lp));
        }
        if let Some(ot) = self.ord_type {
            msg.set_field_str(tags::OrdType, &(ot.to_fix_char() as char).to_string());
        }
        if let Some(tif) = self.time_in_force {
            msg.set_field_str(tags::TimeInForce, &(tif.to_fix_char() as char).to_string());
        }
        if let Some(ref text) = self.text {
            msg.set_field_str(tags::Text, text);
        }
        if let Some(reason) = self.ord_rej_reason {
            msg.set_field_str(tags::OrdRejReason, &reason.to_string());
        }
        if let Some(ref acc) = self.account {
            msg.set_field_str(tags::Account, acc);
        }
        if let Some(ref exch) = self.security_exchange {
            msg.set_field_str(tags::SecurityExchange, exch);
        }
        if let Some(ref ei) = self.exec_inst {
            msg.set_field_str(tags::ExecInst, ei);
        }
        if let Some(ref cur) = self.currency {
            msg.set_field_str(tags::Currency, cur);
        }
        if let Some(ref sid) = self.security_id {
            msg.set_field_str(tags::SecurityID, sid);
        }
        if let Some(ref ids) = self.id_source {
            msg.set_field_str(tags::IDSource, ids);
        }
        if let Some(ref st) = self.security_type {
            msg.set_field_str(tags::SecurityType, st);
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

    pub fn from_fix_message(msg: &FixMessage) -> Result<Self, CodecError> {
        let order_id = msg.require_str(tags::OrderID)
            .map_err(|_| CodecError::MissingField(tags::OrderID))?.to_string();
        let cl_ord_id = msg.require_str(tags::ClOrdID)
            .map_err(|_| CodecError::MissingField(tags::ClOrdID))?.to_string();
        let exec_id = msg.require_str(tags::ExecID)
            .map_err(|_| CodecError::MissingField(tags::ExecID))?.to_string();
        let exec_type = field_types::to_exec_type(msg)?;
        let ord_status_c = msg.require_char(tags::OrdStatus)
            .map_err(|_| CodecError::MissingField(tags::OrdStatus))?;
        let ord_status = crate::types::OrdStatus::from_fix_char(ord_status_c)
            .map_err(|e| CodecError::TypeConversion(e))?;
        let symbol = msg.require_str(tags::Symbol)
            .map_err(|_| CodecError::MissingField(tags::Symbol))?.to_string();
        let side = field_types::to_side(msg)?;
        let leaves_qty = field_types::to_qty(msg, tags::LeavesQty)?;
        let cum_qty = field_types::to_qty(msg, tags::CumQty)?;
        let avg_px = field_types::to_price(msg, tags::AvgPx)?;
        let transact_time = field_types::to_timestamp(msg, tags::TransactTime)?;

        let ord_type = if msg.has_field(tags::OrdType) {
            let c = msg.require_char(tags::OrdType).map_err(|_| CodecError::MissingField(tags::OrdType))?;
            Some(crate::types::OrdType::from_fix_char(c).map_err(|e| CodecError::TypeConversion(e))?)
        } else { None };

        let time_in_force = if msg.has_field(tags::TimeInForce) {
            let c = msg.require_char(tags::TimeInForce).map_err(|_| CodecError::MissingField(tags::TimeInForce))?;
            Some(crate::types::TimeInForce::from_fix_char(c).map_err(|e| CodecError::TypeConversion(e))?)
        } else { None };

        Ok(ExecutionReport {
            order_id,
            cl_ord_id,
            orig_cl_ord_id: msg.get_field(tags::OrigClOrdID).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            exec_id,
            exec_type,
            ord_status,
            symbol,
            side,
            leaves_qty,
            cum_qty,
            avg_px,
            order_qty: if msg.has_field(tags::OrderQty) { Some(field_types::to_qty(msg, tags::OrderQty)?) } else { None },
            price: if msg.has_field(tags::Price) { Some(field_types::to_price(msg, tags::Price)?) } else { None },
            stop_px: if msg.has_field(tags::StopPx) { Some(field_types::to_price(msg, tags::StopPx)?) } else { None },
            last_qty: if msg.has_field(tags::LastQty) { Some(field_types::to_qty(msg, tags::LastQty)?) } else { None },
            last_px: if msg.has_field(tags::LastPx) { Some(field_types::to_price(msg, tags::LastPx)?) } else { None },
            ord_type,
            time_in_force,
            transact_time,
            text: msg.get_field(tags::Text).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            ord_rej_reason: msg.get_field(tags::OrdRejReason).and_then(|f| f.value_u32().ok()),
            account: msg.get_field(tags::Account).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            security_exchange: msg.get_field(tags::SecurityExchange).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            exec_inst: msg.get_field(tags::ExecInst).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            currency: msg.get_field(tags::Currency).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            security_id: msg.get_field(tags::SecurityID).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            id_source: msg.get_field(tags::IDSource).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            security_type: msg.get_field(tags::SecurityType).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            maturity_month_year: msg.get_field(tags::MaturityMonthYear).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            put_or_call: msg.get_field(tags::PutOrCall).and_then(|f| f.value_char().ok()),
            strike_price: if msg.has_field(tags::StrikePrice) { Some(field_types::to_price(msg, tags::StrikePrice)?) } else { None },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Side, OrdStatus, ExecType, Qty, Price};

    #[test]
    fn test_execution_report_roundtrip() {
        let er = ExecutionReport::new(
            "ORD001".to_string(),
            "CLORD001".to_string(),
            "EXEC001".to_string(),
            ExecType::Fill,
            OrdStatus::Filled,
            "AAPL".to_string(),
            Side::Buy,
            Qty::zero(),
            Qty::from_f64(100.0),
            Price::from_f64(150.25),
        )
        .with_fill(Qty::from_f64(100.0), Price::from_f64(150.25))
        .with_order_details(Qty::from_f64(100.0), OrdType::Limit, Some(Price::from_f64(150.00)));

        let fix_msg = er.to_fix_message("FIX.4.2", "BROKER", "CLIENT", 1);
        let decoded = ExecutionReport::from_fix_message(&fix_msg).unwrap();

        assert_eq!(decoded.order_id, "ORD001");
        assert!(decoded.is_filled());
        assert!(decoded.is_terminal());
        assert!((decoded.avg_px.to_f64() - 150.25).abs() < 1e-6);
        assert!((decoded.cum_qty.to_f64() - 100.0).abs() < 1e-4);
    }

    #[test]
    fn test_partial_fill() {
        let er = ExecutionReport::new(
            "O1".to_string(), "C1".to_string(), "E1".to_string(),
            ExecType::PartialFill, OrdStatus::PartiallyFilled,
            "MSFT".to_string(), Side::Sell,
            Qty::from_f64(50.0), Qty::from_f64(50.0), Price::from_f64(300.0),
        );
        assert!(er.is_partial());
        assert!(!er.is_terminal());
    }

    #[test]
    fn test_rejected_order() {
        let er = ExecutionReport::new(
            "O2".to_string(), "C2".to_string(), "E2".to_string(),
            ExecType::Rejected, OrdStatus::Rejected,
            "GOOG".to_string(), Side::Buy,
            Qty::zero(), Qty::zero(), Price::zero(),
        ).with_reject_reason(0).with_text("Insufficient funds".to_string());

        let fix_msg = er.to_fix_message("FIX.4.2", "BROKER", "CLIENT", 2);
        let decoded = ExecutionReport::from_fix_message(&fix_msg).unwrap();
        assert!(decoded.is_terminal());
        assert_eq!(decoded.text.as_deref(), Some("Insufficient funds"));
    }
}
