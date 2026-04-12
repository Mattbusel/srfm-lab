// error.rs — RTEL error types
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RtelError {
    #[error("Shared memory error: {0}")]
    ShmError(String),
    #[error("Ring full — no free slots")]
    RingFull,
    #[error("No data available")]
    NoData,
    #[error("Bad magic header")]
    BadMagic,
    #[error("Invalid argument: {0}")]
    InvalidArg(String),
    #[error("Channel not found: {0}")]
    ChannelNotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Timeout")]
    Timeout,
}

pub type Result<T> = std::result::Result<T, RtelError>;
