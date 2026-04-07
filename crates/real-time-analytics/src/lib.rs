pub mod streaming_stats;
pub mod tick_aggregator;
pub mod market_microstructure;
pub mod risk_analytics;
pub mod streaming;
pub mod analytics;

pub use streaming_stats::{
    StreamingStats, RollingWindow, ExponentialMovingStats,
    StreamingQuantile, CorrelationTracker, Stats, QuantileSnapshot, CorrelationSnapshot,
};
pub use tick_aggregator::{
    TickAggregator, Tick, Bar, BarType, BarSnapshot,
};
pub use market_microstructure::{
    OrderFlowImbalance, KyleLambda, AmihudIlliquidity, BidAskBounce,
    InformationRatio, PinEstimator, TickSide, PinParams,
};
pub use risk_analytics::{
    VaREstimator, ExpectedShortfall, CornishFisher, MaxDrawdown,
    TailRiskDecomposition, StressTest, VaRMethod, DrawdownState, StressScenario,
};
pub use streaming::{
    KalmanFilter1D, KalmanFilter2D, LMSFilter, RLSFilter, AdaptiveEMA, HodrickPrescottFilter,
    EventEmitter,
    VolatilityBreakout, VolatilityBreakoutEvent,
    MomentumShift, MomentumShiftEvent, MomentumDirection,
    VolumeAnomaly, VolumeAnomalyEvent,
    GapDetector, GapEvent, GapDirection,
    OrderFlowReversal, OrderFlowReversalEvent,
};
pub use analytics::{
    RegimeDetector, RegimeChange,
    BHMassRegime, BHMassLevel,
    HurstRegime, HurstLevel,
    VolRegime, VolLevel,
    CompositeRegime, CompositeLevel,
    VPINEstimator, KyleLambdaEstimator, AmihudEstimator, SpreadEstimator, ToxicityMeter,
};
