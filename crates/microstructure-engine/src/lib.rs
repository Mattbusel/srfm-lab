/// microstructure-engine
/// =====================
/// High-performance market microstructure analytics.
///
/// Modules:
///   - kyle_lambda          : Kyle's price-impact coefficient (Λ)
///   - amihud               : Amihud illiquidity ratio (rolling)
///   - roll_spread          : Roll bid-ask spread estimator
///   - pin                  : Probability of Informed Trading (EM algorithm)
///   - order_flow           : Signed order-flow, tick rule, VPIN
///   - tick_bars            : Tick/volume/dollar/imbalance bar aggregators
///   - streaming_stats      : Online Welford mean/variance/skew/kurt
///   - trade_flow_analysis  : Trade flow, OFI, delta divergence, aggressiveness
///   - liquidity_metrics    : Effective/realized spread, price impact, resilience
///   - market_regime_signals: Microstructure-based regime signal computation

pub mod kyle_lambda;
pub mod amihud;
pub mod roll_spread;
pub mod pin;
pub mod order_flow;
pub mod tick_bars;
pub mod streaming_stats;
pub mod trade_flow_analysis;
pub mod liquidity_metrics;
pub mod market_regime_signals;

pub use kyle_lambda::KyleLambda;
pub use amihud::AmihudIlliquidity;
pub use roll_spread::RollSpreadEstimator;
pub use pin::PinEstimator;
pub use order_flow::{OrderFlowImbalance, TickRule};
pub use tick_bars::{TickAggregator, BarType, OhlcvBar};
pub use streaming_stats::{StreamingStats, RollingWindow};
pub use trade_flow_analysis::{
    TradeFlowAnalyzer, TradeFlowState, DeltaDivergence, tick_rule_classify,
};
pub use liquidity_metrics::{
    LiquidityMetrics, LiquiditySnapshot, ResilienceTracker,
    effective_spread, realized_spread, price_impact, quote_quality_score,
};
pub use market_regime_signals::{
    MicrostructureRegimeSignal, MicrostructureContext, MicroRegime,
    classify_microstructure_regime,
};
