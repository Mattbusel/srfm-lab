// streaming/mod.rs -- Streaming signal processing and event detection.

pub mod adaptive_filter;
pub mod event_detector;

pub use adaptive_filter::{
    KalmanFilter1D, KalmanFilter2D, LMSFilter, RLSFilter, AdaptiveEMA, HodrickPrescottFilter,
};
pub use event_detector::{
    EventEmitter,
    VolatilityBreakout, VolatilityBreakoutEvent,
    MomentumShift, MomentumShiftEvent, MomentumDirection,
    VolumeAnomaly, VolumeAnomalyEvent,
    GapDetector, GapEvent, GapDirection,
    OrderFlowReversal, OrderFlowReversalEvent,
};
