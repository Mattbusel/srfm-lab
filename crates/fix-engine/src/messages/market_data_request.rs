use crate::message::{FixMessage, tags};
use crate::types::MsgType;
use crate::codec::CodecError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscriptionRequestType {
    Snapshot,
    SnapshotAndUpdates,
    DisablePreviousSnapshot,
}

impl SubscriptionRequestType {
    pub fn to_char(self) -> u8 {
        match self {
            SubscriptionRequestType::Snapshot => b'0',
            SubscriptionRequestType::SnapshotAndUpdates => b'1',
            SubscriptionRequestType::DisablePreviousSnapshot => b'2',
        }
    }

    pub fn from_char(c: u8) -> Result<Self, CodecError> {
        match c {
            b'0' => Ok(SubscriptionRequestType::Snapshot),
            b'1' => Ok(SubscriptionRequestType::SnapshotAndUpdates),
            b'2' => Ok(SubscriptionRequestType::DisablePreviousSnapshot),
            _ => Err(CodecError::Encode(format!("Invalid SubscriptionRequestType: {}", c as char))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdUpdateType {
    FullRefresh,
    IncrementalRefresh,
}

impl MdUpdateType {
    pub fn to_char(self) -> u8 {
        match self {
            MdUpdateType::FullRefresh => b'0',
            MdUpdateType::IncrementalRefresh => b'1',
        }
    }

    pub fn from_char(c: u8) -> Result<Self, CodecError> {
        match c {
            b'0' => Ok(MdUpdateType::FullRefresh),
            b'1' => Ok(MdUpdateType::IncrementalRefresh),
            _ => Err(CodecError::Encode(format!("Invalid MdUpdateType: {}", c as char))),
        }
    }
}

/// FIX MarketDataRequest (V)
#[derive(Debug, Clone)]
pub struct MarketDataRequest {
    pub md_req_id: String,
    pub subscription_request_type: SubscriptionRequestType,
    pub market_depth: u32,
    pub md_update_type: Option<MdUpdateType>,
    pub md_entry_types: Vec<u8>,  // '0'=Bid, '1'=Offer, '2'=Trade, etc.
    pub symbols: Vec<String>,
}

impl MarketDataRequest {
    pub fn new(
        md_req_id: String,
        subscription_request_type: SubscriptionRequestType,
        market_depth: u32,
    ) -> Self {
        MarketDataRequest {
            md_req_id,
            subscription_request_type,
            market_depth,
            md_update_type: None,
            md_entry_types: Vec::new(),
            symbols: Vec::new(),
        }
    }

    pub fn with_bid_offer(mut self) -> Self {
        self.md_entry_types.push(b'0');
        self.md_entry_types.push(b'1');
        self
    }

    pub fn with_trades(mut self) -> Self {
        self.md_entry_types.push(b'2');
        self
    }

    pub fn with_symbol(mut self, symbol: String) -> Self {
        self.symbols.push(symbol);
        self
    }

    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = symbols;
        self
    }

    pub fn with_update_type(mut self, ut: MdUpdateType) -> Self {
        self.md_update_type = Some(ut);
        self
    }

    pub fn to_fix_message(&self, begin_string: &str, sender: &str, target: &str, seq: u32) -> FixMessage {
        let mut msg = FixMessage::with_header(
            begin_string,
            MsgType::MarketDataRequest.as_str(),
            sender,
            target,
            seq,
        );
        msg.set_field_str(tags::MDReqID, &self.md_req_id);
        msg.set_field_str(tags::SubscriptionRequestType, &(self.subscription_request_type.to_char() as char).to_string());
        msg.set_field_str(tags::MarketDepth, &self.market_depth.to_string());
        if let Some(ut) = self.md_update_type {
            msg.set_field_str(tags::MDUpdateType, &(ut.to_char() as char).to_string());
        }

        // NoMDEntryTypes repeating group
        msg.set_field_str(tags::NoMDEntryTypes, &self.md_entry_types.len().to_string());
        for &entry_type in &self.md_entry_types {
            msg.set_field_str(tags::MDEntryType, &(entry_type as char).to_string());
        }

        // NoRelatedSym repeating group
        msg.set_field_str(tags::NoRelatedSym, &self.symbols.len().to_string());
        for sym in &self.symbols {
            msg.set_field_str(tags::Symbol, sym);
        }
        msg
    }

    pub fn from_fix_message(msg: &FixMessage) -> Result<Self, CodecError> {
        let md_req_id = msg.require_str(tags::MDReqID)
            .map_err(|_| CodecError::MissingField(tags::MDReqID))?.to_string();
        let srt_c = msg.require_char(tags::SubscriptionRequestType)
            .map_err(|_| CodecError::MissingField(tags::SubscriptionRequestType))?;
        let subscription_request_type = SubscriptionRequestType::from_char(srt_c)?;
        let market_depth = msg.require_u32(tags::MarketDepth)
            .map_err(|_| CodecError::MissingField(tags::MarketDepth))?;

        let md_update_type = if msg.has_field(tags::MDUpdateType) {
            let c = msg.require_char(tags::MDUpdateType).map_err(|_| CodecError::MissingField(tags::MDUpdateType))?;
            Some(MdUpdateType::from_char(c)?)
        } else { None };

        // For simplicity, get the single MDEntryType field value as the entry types list
        let md_entry_types: Vec<u8> = msg.get_field(tags::MDEntryType)
            .and_then(|f| f.value_str().ok())
            .map(|s| s.bytes().collect())
            .unwrap_or_default();

        let symbols: Vec<String> = msg.get_field(tags::Symbol)
            .and_then(|f| f.value_str().ok())
            .map(|s| vec![s.to_string()])
            .unwrap_or_default();

        Ok(MarketDataRequest {
            md_req_id,
            subscription_request_type,
            market_depth,
            md_update_type,
            md_entry_types,
            symbols,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data_request_roundtrip() {
        let req = MarketDataRequest::new(
            "MDR001".to_string(),
            SubscriptionRequestType::SnapshotAndUpdates,
            10,
        )
        .with_bid_offer()
        .with_symbol("AAPL".to_string())
        .with_update_type(MdUpdateType::FullRefresh);

        let fix_msg = req.to_fix_message("FIX.4.2", "CLIENT", "MARKET", 1);
        let decoded = MarketDataRequest::from_fix_message(&fix_msg).unwrap();
        assert_eq!(decoded.md_req_id, "MDR001");
        assert_eq!(decoded.market_depth, 10);
        assert!(matches!(decoded.subscription_request_type, SubscriptionRequestType::SnapshotAndUpdates));
    }
}
