pub mod adversarial_testing;
pub mod execution;
pub mod fills;
pub mod market_impact;
pub mod market_impact_sim;
pub mod microstructure;
pub mod order;
pub mod orderbook;
pub mod simulator;
pub mod synthetic_orderbook;

// ── Chronos / AETERNUS Module 1 additions ───────────────────────────────────
pub mod lob_engine;
pub mod hawkes_lob;
pub mod heston_vol;
pub mod agent_impact;
pub mod event_stream;
pub mod statistics;
pub mod replay;

// ── Chronos / AETERNUS Module 2 additions ───────────────────────────────────
pub mod data_replay;
pub mod latency_model;
pub mod market_impact_model;
pub mod telemetry;
pub mod order_types;
pub mod risk_checks;
#[cfg(test)]
pub mod tests;

pub use order::{Order, OrderSide, OrderStatus, OrderType};
pub use orderbook::OrderBook;
pub use fills::{Fill, FillSummary};
pub use market_impact::{
    AlmgrenChrissParams, AlmgrenChrissSchedule, ImpactModel, LinearParams, SquareRootParams,
    estimate_slippage, sqrt_permanent_impact, sqrt_temporary_impact,
};
pub use execution::{ExecutionSlice, twap_schedule, vwap_schedule, is_schedule, pov_schedule};
pub use microstructure::{
    amihud_illiquidity, bid_ask_spread_decomposition, effective_spread, kyle_lambda,
    realized_spread, roll_spread_estimator,
};
pub use simulator::{
    HawkesParams, OrderFlowParams, SimResult, generate_order_flow, run_simulation,
    simulate_hawkes,
};

// New Chronos exports.
pub use lob_engine::{
    LobEngine, LobOrder, LobFill, Side, OrderKind, TimeInForce, OrderStatus as LobOrderStatus,
    PriceLevel, Nanos, OrderId, TickPrice, Qty,
    to_tick, from_tick, PRICE_SCALE,
};
pub use hawkes_lob::{
    HawkesParams1D, HawkesParams2D, HawkesState2D,
    hawkes_log_likelihood, hawkes_mle, hawkes_intensity_at, hawkes_compensator,
    hawkes_2d_log_likelihood, hawkes_expected_count,
    sim_hawkes_1d, sim_hawkes_2d,
};
pub use heston_vol::{
    HestonParams, HestonStep, HestonLobDriver, ReturnStats,
    simulate_heston, calibrate_heston_mom, enforce_cir_constraints,
    heston_char_fn, heston_mean_return, heston_return_variance, heston_excess_kurtosis,
};
pub use agent_impact::{
    AgentFill, AgentImpactTracker, Footprint, SelfImpactModel,
    QueuePositionTracker, ImpactSummary,
};
pub use event_stream::{
    LobEvent, EventBus, EventBusSender, EventBuilder, EventFilter,
    OrderSubmitEvent, OrderCancelEvent, FillEvent, PriceUpdateEvent, RegimeChangeEvent,
    MarketRegime, RegimeChangeTrigger, LobEventError, FilteredReceiver,
    next_event_id,
};
pub use statistics::{
    LobStatistics, StatSnapshot, RunningVwap, RealizedVol, SpreadDecomposition,
    AmihudEstimator, KyleLambdaEstimator, RollEstimator, ArrivalRateEstimator,
    TradeRecord, CircularBuffer,
};
pub use replay::{
    ReplayEngine, ReplayConfig, ReplayState, ReplayStats,
    RecordedSession, BacktestResult, run_backtest,
};
