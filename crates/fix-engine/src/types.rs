use std::fmt;
use chrono::{DateTime, Utc, NaiveDateTime};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Price(pub i64); // stored as integer * 10^8

impl Price {
    pub const SCALE: i64 = 100_000_000;

    pub fn from_f64(v: f64) -> Self {
        Price((v * Self::SCALE as f64).round() as i64)
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    pub fn zero() -> Self { Price(0) }

    pub fn is_zero(self) -> bool { self.0 == 0 }

    pub fn checked_add(self, other: Price) -> Option<Price> {
        self.0.checked_add(other.0).map(Price)
    }

    pub fn checked_mul_f64(self, factor: f64) -> Price {
        Price::from_f64(self.to_f64() * factor)
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.8}", self.to_f64())
    }
}

impl From<f64> for Price {
    fn from(v: f64) -> Self { Price::from_f64(v) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Qty(pub i64); // stored as integer * 10^4

impl Qty {
    pub const SCALE: i64 = 10_000;

    pub fn from_f64(v: f64) -> Self {
        Qty((v * Self::SCALE as f64).round() as i64)
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    pub fn zero() -> Self { Qty(0) }

    pub fn is_zero(self) -> bool { self.0 == 0 }

    pub fn checked_add(self, other: Qty) -> Option<Qty> {
        self.0.checked_add(other.0).map(Qty)
    }
}

impl fmt::Display for Qty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIX quantities typically integer or up to 4 decimal places
        let whole = self.0 / Self::SCALE;
        let frac = self.0.abs() % Self::SCALE;
        if frac == 0 {
            write!(f, "{}", whole)
        } else {
            write!(f, "{}.{:04}", whole, frac)
        }
    }
}

impl From<f64> for Qty {
    fn from(v: f64) -> Self { Qty::from_f64(v) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixChar(pub u8);

impl FixChar {
    pub fn new(c: char) -> Option<Self> {
        if c.is_ascii() { Some(FixChar(c as u8)) } else { None }
    }

    pub fn as_char(self) -> char { self.0 as char }
}

impl fmt::Display for FixChar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0 as char)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UtcTimestamp(pub DateTime<Utc>);

impl UtcTimestamp {
    pub fn now() -> Self { UtcTimestamp(Utc::now()) }

    pub fn from_fix_str(s: &str) -> Result<Self, FixTypeError> {
        // FIX format: YYYYMMDD-HH:MM:SS or YYYYMMDD-HH:MM:SS.sss
        let fmt1 = "%Y%m%d-%H:%M:%S%.3f";
        let fmt2 = "%Y%m%d-%H:%M:%S";
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt1) {
            return Ok(UtcTimestamp(DateTime::from_naive_utc_and_offset(dt, Utc)));
        }
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt2) {
            return Ok(UtcTimestamp(DateTime::from_naive_utc_and_offset(dt, Utc)));
        }
        Err(FixTypeError::InvalidTimestamp(s.to_string()))
    }

    pub fn to_fix_str(&self) -> String {
        self.0.format("%Y%m%d-%H:%M:%S%.3f").to_string()
    }
}

impl fmt::Display for UtcTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_fix_str())
    }
}

// FIX standard enums
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
    BuyMinus,
    SellPlus,
    SellShort,
    SellShortExempt,
    Undisclosed,
    Cross,
    CrossShort,
}

impl Side {
    pub fn to_fix_char(self) -> u8 {
        match self {
            Side::Buy => b'1',
            Side::Sell => b'2',
            Side::BuyMinus => b'3',
            Side::SellPlus => b'4',
            Side::SellShort => b'5',
            Side::SellShortExempt => b'6',
            Side::Undisclosed => b'7',
            Side::Cross => b'8',
            Side::CrossShort => b'9',
        }
    }

    pub fn from_fix_char(c: u8) -> Result<Self, FixTypeError> {
        match c {
            b'1' => Ok(Side::Buy),
            b'2' => Ok(Side::Sell),
            b'3' => Ok(Side::BuyMinus),
            b'4' => Ok(Side::SellPlus),
            b'5' => Ok(Side::SellShort),
            b'6' => Ok(Side::SellShortExempt),
            b'7' => Ok(Side::Undisclosed),
            b'8' => Ok(Side::Cross),
            b'9' => Ok(Side::CrossShort),
            _ => Err(FixTypeError::InvalidEnum("Side", c as char)),
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_fix_char() as char)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrdType {
    Market,
    Limit,
    Stop,
    StopLimit,
    MarketOnClose,
    WithOrWithout,
    LimitOrBetter,
    LimitWithOrWithout,
    OnBasis,
    OnClose,
    Pegged,
}

impl OrdType {
    pub fn to_fix_char(self) -> u8 {
        match self {
            OrdType::Market => b'1',
            OrdType::Limit => b'2',
            OrdType::Stop => b'3',
            OrdType::StopLimit => b'4',
            OrdType::MarketOnClose => b'5',
            OrdType::WithOrWithout => b'6',
            OrdType::LimitOrBetter => b'7',
            OrdType::LimitWithOrWithout => b'8',
            OrdType::OnBasis => b'9',
            OrdType::OnClose => b'A',
            OrdType::Pegged => b'P',
        }
    }

    pub fn from_fix_char(c: u8) -> Result<Self, FixTypeError> {
        match c {
            b'1' => Ok(OrdType::Market),
            b'2' => Ok(OrdType::Limit),
            b'3' => Ok(OrdType::Stop),
            b'4' => Ok(OrdType::StopLimit),
            b'5' => Ok(OrdType::MarketOnClose),
            b'6' => Ok(OrdType::WithOrWithout),
            b'7' => Ok(OrdType::LimitOrBetter),
            b'8' => Ok(OrdType::LimitWithOrWithout),
            b'9' => Ok(OrdType::OnBasis),
            b'A' => Ok(OrdType::OnClose),
            b'P' => Ok(OrdType::Pegged),
            _ => Err(FixTypeError::InvalidEnum("OrdType", c as char)),
        }
    }
}

impl fmt::Display for OrdType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_fix_char() as char)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrdStatus {
    New,
    PartiallyFilled,
    Filled,
    DoneForDay,
    Canceled,
    Replaced,
    PendingCancel,
    Stopped,
    Rejected,
    Suspended,
    PendingNew,
    Calculated,
    Expired,
    AcceptedForBidding,
    PendingReplace,
}

impl OrdStatus {
    pub fn to_fix_char(self) -> u8 {
        match self {
            OrdStatus::New => b'0',
            OrdStatus::PartiallyFilled => b'1',
            OrdStatus::Filled => b'2',
            OrdStatus::DoneForDay => b'3',
            OrdStatus::Canceled => b'4',
            OrdStatus::Replaced => b'5',
            OrdStatus::PendingCancel => b'6',
            OrdStatus::Stopped => b'7',
            OrdStatus::Rejected => b'8',
            OrdStatus::Suspended => b'9',
            OrdStatus::PendingNew => b'A',
            OrdStatus::Calculated => b'B',
            OrdStatus::Expired => b'C',
            OrdStatus::AcceptedForBidding => b'D',
            OrdStatus::PendingReplace => b'E',
        }
    }

    pub fn from_fix_char(c: u8) -> Result<Self, FixTypeError> {
        match c {
            b'0' => Ok(OrdStatus::New),
            b'1' => Ok(OrdStatus::PartiallyFilled),
            b'2' => Ok(OrdStatus::Filled),
            b'3' => Ok(OrdStatus::DoneForDay),
            b'4' => Ok(OrdStatus::Canceled),
            b'5' => Ok(OrdStatus::Replaced),
            b'6' => Ok(OrdStatus::PendingCancel),
            b'7' => Ok(OrdStatus::Stopped),
            b'8' => Ok(OrdStatus::Rejected),
            b'9' => Ok(OrdStatus::Suspended),
            b'A' => Ok(OrdStatus::PendingNew),
            b'B' => Ok(OrdStatus::Calculated),
            b'C' => Ok(OrdStatus::Expired),
            b'D' => Ok(OrdStatus::AcceptedForBidding),
            b'E' => Ok(OrdStatus::PendingReplace),
            _ => Err(FixTypeError::InvalidEnum("OrdStatus", c as char)),
        }
    }
}

impl fmt::Display for OrdStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_fix_char() as char)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecType {
    New,
    PartialFill,
    Fill,
    DoneForDay,
    Canceled,
    Replace,
    PendingCancel,
    Stopped,
    Rejected,
    Suspended,
    PendingNew,
    Calculated,
    Expired,
    Restated,
    PendingReplace,
    Trade,
    TradeCorrect,
    TradeCancel,
    OrderStatus,
}

impl ExecType {
    pub fn to_fix_char(self) -> u8 {
        match self {
            ExecType::New => b'0',
            ExecType::PartialFill => b'1',
            ExecType::Fill => b'2',
            ExecType::DoneForDay => b'3',
            ExecType::Canceled => b'4',
            ExecType::Replace => b'5',
            ExecType::PendingCancel => b'6',
            ExecType::Stopped => b'7',
            ExecType::Rejected => b'8',
            ExecType::Suspended => b'9',
            ExecType::PendingNew => b'A',
            ExecType::Calculated => b'B',
            ExecType::Expired => b'C',
            ExecType::Restated => b'D',
            ExecType::PendingReplace => b'E',
            ExecType::Trade => b'F',
            ExecType::TradeCorrect => b'G',
            ExecType::TradeCancel => b'H',
            ExecType::OrderStatus => b'I',
        }
    }

    pub fn from_fix_char(c: u8) -> Result<Self, FixTypeError> {
        match c {
            b'0' => Ok(ExecType::New),
            b'1' => Ok(ExecType::PartialFill),
            b'2' => Ok(ExecType::Fill),
            b'3' => Ok(ExecType::DoneForDay),
            b'4' => Ok(ExecType::Canceled),
            b'5' => Ok(ExecType::Replace),
            b'6' => Ok(ExecType::PendingCancel),
            b'7' => Ok(ExecType::Stopped),
            b'8' => Ok(ExecType::Rejected),
            b'9' => Ok(ExecType::Suspended),
            b'A' => Ok(ExecType::PendingNew),
            b'B' => Ok(ExecType::Calculated),
            b'C' => Ok(ExecType::Expired),
            b'D' => Ok(ExecType::Restated),
            b'E' => Ok(ExecType::PendingReplace),
            b'F' => Ok(ExecType::Trade),
            b'G' => Ok(ExecType::TradeCorrect),
            b'H' => Ok(ExecType::TradeCancel),
            b'I' => Ok(ExecType::OrderStatus),
            _ => Err(FixTypeError::InvalidEnum("ExecType", c as char)),
        }
    }
}

impl fmt::Display for ExecType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_fix_char() as char)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeInForce {
    Day,
    GoodTillCancel,
    AtTheOpening,
    ImmediateOrCancel,
    FillOrKill,
    GoodTillCrossing,
    GoodTillDate,
    AtTheClose,
}

impl TimeInForce {
    pub fn to_fix_char(self) -> u8 {
        match self {
            TimeInForce::Day => b'0',
            TimeInForce::GoodTillCancel => b'1',
            TimeInForce::AtTheOpening => b'2',
            TimeInForce::ImmediateOrCancel => b'3',
            TimeInForce::FillOrKill => b'4',
            TimeInForce::GoodTillCrossing => b'5',
            TimeInForce::GoodTillDate => b'6',
            TimeInForce::AtTheClose => b'7',
        }
    }

    pub fn from_fix_char(c: u8) -> Result<Self, FixTypeError> {
        match c {
            b'0' => Ok(TimeInForce::Day),
            b'1' => Ok(TimeInForce::GoodTillCancel),
            b'2' => Ok(TimeInForce::AtTheOpening),
            b'3' => Ok(TimeInForce::ImmediateOrCancel),
            b'4' => Ok(TimeInForce::FillOrKill),
            b'5' => Ok(TimeInForce::GoodTillCrossing),
            b'6' => Ok(TimeInForce::GoodTillDate),
            b'7' => Ok(TimeInForce::AtTheClose),
            _ => Err(FixTypeError::InvalidEnum("TimeInForce", c as char)),
        }
    }
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_fix_char() as char)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MsgType {
    Heartbeat,
    TestRequest,
    ResendRequest,
    Reject,
    SequenceReset,
    Logout,
    ExecutionReport,
    OrderCancelReject,
    Logon,
    NewOrderSingle,
    OrderCancelRequest,
    OrderCancelReplaceRequest,
    OrderStatusRequest,
    MarketDataRequest,
    MarketDataSnapshotFullRefresh,
    MarketDataIncrementalRefresh,
    MarketDataRequestReject,
    BusinessMessageReject,
    News,
    Email,
}

impl MsgType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MsgType::Heartbeat => "0",
            MsgType::TestRequest => "1",
            MsgType::ResendRequest => "2",
            MsgType::Reject => "3",
            MsgType::SequenceReset => "4",
            MsgType::Logout => "5",
            MsgType::ExecutionReport => "8",
            MsgType::OrderCancelReject => "9",
            MsgType::Logon => "A",
            MsgType::NewOrderSingle => "D",
            MsgType::OrderCancelRequest => "F",
            MsgType::OrderCancelReplaceRequest => "G",
            MsgType::OrderStatusRequest => "H",
            MsgType::MarketDataRequest => "V",
            MsgType::MarketDataSnapshotFullRefresh => "W",
            MsgType::MarketDataIncrementalRefresh => "X",
            MsgType::MarketDataRequestReject => "Y",
            MsgType::BusinessMessageReject => "j",
            MsgType::News => "B",
            MsgType::Email => "C",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, FixTypeError> {
        match s {
            "0" => Ok(MsgType::Heartbeat),
            "1" => Ok(MsgType::TestRequest),
            "2" => Ok(MsgType::ResendRequest),
            "3" => Ok(MsgType::Reject),
            "4" => Ok(MsgType::SequenceReset),
            "5" => Ok(MsgType::Logout),
            "8" => Ok(MsgType::ExecutionReport),
            "9" => Ok(MsgType::OrderCancelReject),
            "A" => Ok(MsgType::Logon),
            "D" => Ok(MsgType::NewOrderSingle),
            "F" => Ok(MsgType::OrderCancelRequest),
            "G" => Ok(MsgType::OrderCancelReplaceRequest),
            "H" => Ok(MsgType::OrderStatusRequest),
            "V" => Ok(MsgType::MarketDataRequest),
            "W" => Ok(MsgType::MarketDataSnapshotFullRefresh),
            "X" => Ok(MsgType::MarketDataIncrementalRefresh),
            "Y" => Ok(MsgType::MarketDataRequestReject),
            "j" => Ok(MsgType::BusinessMessageReject),
            "B" => Ok(MsgType::News),
            "C" => Ok(MsgType::Email),
            _ => Err(FixTypeError::UnknownMsgType(s.to_string())),
        }
    }
}

impl fmt::Display for MsgType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Error)]
pub enum FixTypeError {
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),
    #[error("Invalid enum value for {0}: '{1}'")]
    InvalidEnum(&'static str, char),
    #[error("Unknown MsgType: {0}")]
    UnknownMsgType(String),
    #[error("Parse error: {0}")]
    ParseError(String),
}
