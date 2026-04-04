use crate::message::{FixMessage, tags};
use crate::types::{Price, Qty, UtcTimestamp, MsgType};
use crate::codec::{CodecError, FixEncoder, FixDecoder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdEntryType {
    Bid,
    Offer,
    Trade,
    IndexValue,
    OpeningPrice,
    ClosingPrice,
    SettlementPrice,
    TradingSessionHighPrice,
    TradingSessionLowPrice,
    TradingSessionVwapPrice,
}

impl MdEntryType {
    pub fn to_char(self) -> u8 {
        match self {
            MdEntryType::Bid => b'0',
            MdEntryType::Offer => b'1',
            MdEntryType::Trade => b'2',
            MdEntryType::IndexValue => b'3',
            MdEntryType::OpeningPrice => b'4',
            MdEntryType::ClosingPrice => b'5',
            MdEntryType::SettlementPrice => b'6',
            MdEntryType::TradingSessionHighPrice => b'7',
            MdEntryType::TradingSessionLowPrice => b'8',
            MdEntryType::TradingSessionVwapPrice => b'9',
        }
    }

    pub fn from_char(c: u8) -> Result<Self, CodecError> {
        match c {
            b'0' => Ok(MdEntryType::Bid),
            b'1' => Ok(MdEntryType::Offer),
            b'2' => Ok(MdEntryType::Trade),
            b'3' => Ok(MdEntryType::IndexValue),
            b'4' => Ok(MdEntryType::OpeningPrice),
            b'5' => Ok(MdEntryType::ClosingPrice),
            b'6' => Ok(MdEntryType::SettlementPrice),
            b'7' => Ok(MdEntryType::TradingSessionHighPrice),
            b'8' => Ok(MdEntryType::TradingSessionLowPrice),
            b'9' => Ok(MdEntryType::TradingSessionVwapPrice),
            _ => Err(CodecError::Encode(format!("Invalid MdEntryType: {}", c as char))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MdEntry {
    pub entry_type: MdEntryType,
    pub price: Price,
    pub size: Option<Qty>,
    pub md_mkt: Option<String>,
}

impl MdEntry {
    pub fn bid(price: Price, size: Qty) -> Self {
        MdEntry { entry_type: MdEntryType::Bid, price, size: Some(size), md_mkt: None }
    }

    pub fn offer(price: Price, size: Qty) -> Self {
        MdEntry { entry_type: MdEntryType::Offer, price, size: Some(size), md_mkt: None }
    }

    pub fn trade(price: Price, size: Qty) -> Self {
        MdEntry { entry_type: MdEntryType::Trade, price, size: Some(size), md_mkt: None }
    }

    pub fn with_mkt(mut self, mkt: String) -> Self {
        self.md_mkt = Some(mkt);
        self
    }
}

/// FIX MarketDataSnapshotFullRefresh (W)
#[derive(Debug, Clone)]
pub struct MarketDataSnapshotFullRefresh {
    pub md_req_id: Option<String>,
    pub symbol: String,
    pub security_exchange: Option<String>,
    pub entries: Vec<MdEntry>,
}

impl MarketDataSnapshotFullRefresh {
    pub fn new(symbol: String) -> Self {
        MarketDataSnapshotFullRefresh {
            md_req_id: None,
            symbol,
            security_exchange: None,
            entries: Vec::new(),
        }
    }

    pub fn with_req_id(mut self, id: String) -> Self {
        self.md_req_id = Some(id);
        self
    }

    pub fn with_exchange(mut self, exch: String) -> Self {
        self.security_exchange = Some(exch);
        self
    }

    pub fn add_entry(mut self, entry: MdEntry) -> Self {
        self.entries.push(entry);
        self
    }

    pub fn best_bid(&self) -> Option<&MdEntry> {
        self.entries.iter()
            .filter(|e| e.entry_type == MdEntryType::Bid)
            .max_by_key(|e| e.price)
    }

    pub fn best_offer(&self) -> Option<&MdEntry> {
        self.entries.iter()
            .filter(|e| e.entry_type == MdEntryType::Offer)
            .min_by_key(|e| e.price)
    }

    pub fn mid_price(&self) -> Option<Price> {
        let bid = self.best_bid()?.price;
        let offer = self.best_offer()?.price;
        Some(Price((bid.0 + offer.0) / 2))
    }

    pub fn spread(&self) -> Option<Price> {
        let bid = self.best_bid()?.price;
        let offer = self.best_offer()?.price;
        if offer.0 >= bid.0 {
            Some(Price(offer.0 - bid.0))
        } else {
            None
        }
    }

    pub fn to_fix_message(&self, begin_string: &str, sender: &str, target: &str, seq: u32) -> FixMessage {
        let mut msg = FixMessage::with_header(
            begin_string,
            MsgType::MarketDataSnapshotFullRefresh.as_str(),
            sender,
            target,
            seq,
        );
        if let Some(ref req_id) = self.md_req_id {
            msg.set_field_str(tags::MDReqID, req_id);
        }
        msg.set_field_str(tags::Symbol, &self.symbol);
        if let Some(ref exch) = self.security_exchange {
            msg.set_field_str(tags::SecurityExchange, exch);
        }
        msg.set_field_str(tags::NoMDEntries, &self.entries.len().to_string());
        for entry in &self.entries {
            msg.set_field_str(tags::MDEntryType, &(entry.entry_type.to_char() as char).to_string());
            msg.set_field_str(tags::MDEntryPx, &FixEncoder::encode_price(entry.price));
            if let Some(size) = entry.size {
                msg.set_field_str(tags::MDEntrySize, &FixEncoder::encode_qty(size));
            }
            if let Some(ref mkt) = entry.md_mkt {
                msg.set_field_str(tags::MDMkt, mkt);
            }
        }
        msg
    }

    pub fn from_fix_message(msg: &FixMessage) -> Result<Self, CodecError> {
        let symbol = msg.require_str(tags::Symbol)
            .map_err(|_| CodecError::MissingField(tags::Symbol))?.to_string();

        let n_entries = msg.get_field(tags::NoMDEntries)
            .and_then(|f| f.value_u32().ok())
            .unwrap_or(0) as usize;

        // Build entries from the fields we have (simplified: one entry from the fields)
        // In a full implementation, you'd use RepeatingGroup parsing
        let mut entries = Vec::new();
        if msg.has_field(tags::MDEntryType) {
            let entry_type_c = msg.require_char(tags::MDEntryType)
                .map_err(|_| CodecError::MissingField(tags::MDEntryType))?;
            let entry_type = MdEntryType::from_char(entry_type_c)?;
            let price = if msg.has_field(tags::MDEntryPx) {
                let s = msg.require_str(tags::MDEntryPx)
                    .map_err(|_| CodecError::MissingField(tags::MDEntryPx))?;
                FixDecoder::decode_price(s)?
            } else {
                Price::zero()
            };
            let size = if msg.has_field(tags::MDEntrySize) {
                let s = msg.require_str(tags::MDEntrySize)
                    .map_err(|_| CodecError::MissingField(tags::MDEntrySize))?;
                Some(FixDecoder::decode_qty(s)?)
            } else {
                None
            };
            entries.push(MdEntry {
                entry_type,
                price,
                size,
                md_mkt: msg.get_field(tags::MDMkt).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            });
        }

        Ok(MarketDataSnapshotFullRefresh {
            md_req_id: msg.get_field(tags::MDReqID).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            symbol,
            security_exchange: msg.get_field(tags::SecurityExchange).and_then(|f| f.value_str().ok().map(|s| s.to_string())),
            entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Price, Qty};

    #[test]
    fn test_snapshot_creation() {
        let snap = MarketDataSnapshotFullRefresh::new("AAPL".to_string())
            .with_req_id("MDR001".to_string())
            .add_entry(MdEntry::bid(Price::from_f64(150.00), Qty::from_f64(1000.0)))
            .add_entry(MdEntry::offer(Price::from_f64(150.05), Qty::from_f64(500.0)))
            .add_entry(MdEntry::trade(Price::from_f64(150.02), Qty::from_f64(100.0)));

        assert_eq!(snap.entries.len(), 3);
        let bid = snap.best_bid().unwrap();
        assert!((bid.price.to_f64() - 150.00).abs() < 1e-6);
        let offer = snap.best_offer().unwrap();
        assert!((offer.price.to_f64() - 150.05).abs() < 1e-6);
        let mid = snap.mid_price().unwrap();
        assert!((mid.to_f64() - 150.025).abs() < 1e-4);
        let spread = snap.spread().unwrap();
        assert!((spread.to_f64() - 0.05).abs() < 1e-5);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let snap = MarketDataSnapshotFullRefresh::new("MSFT".to_string())
            .add_entry(MdEntry::bid(Price::from_f64(300.0), Qty::from_f64(200.0)));

        let fix_msg = snap.to_fix_message("FIX.4.2", "MARKET", "CLIENT", 1);
        let decoded = MarketDataSnapshotFullRefresh::from_fix_message(&fix_msg).unwrap();
        assert_eq!(decoded.symbol, "MSFT");
        assert_eq!(decoded.entries.len(), 1);
        assert!(matches!(decoded.entries[0].entry_type, MdEntryType::Bid));
    }
}
