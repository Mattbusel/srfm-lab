/// execution-optimizer
/// ===================
/// Optimal trade scheduling algorithms.
///
/// Modules:
///   - almgren_chriss : Continuous-time optimal execution (AC 2001)
///   - twap           : Time-weighted average price slicer
///   - vwap           : Volume-weighted average price slicer
///   - impact         : Market-impact models (linear, square-root)
///   - schedule       : Unified execution schedule builder

pub mod almgren_chriss;
pub mod twap;
pub mod vwap;
pub mod impact;
pub mod schedule;
pub mod optimal_execution;
pub mod adaptive_optimizer;
pub mod execution_simulator;

pub use almgren_chriss::{AlmgrenChriss, AcParams, AcSchedule};
pub use twap::TwapSlicer;
pub use vwap::VwapSlicer;
pub use impact::{LinearImpact, SquareRootImpact, ImpactModel};
pub use schedule::{ExecutionSchedule, SliceOrder};
pub use optimal_execution::{
    OptimalExecutionEngine,
    ac_optimal_schedule, ac_expected_cost, ac_variance,
    twap_schedule, vwap_schedule, participation_rate_schedule,
};
pub use adaptive_optimizer::{
    AdaptiveExecutionOptimizer, MarketConditions, OptConfig,
};
pub use execution_simulator::{
    ExecutionSimulator, SimResult, ImpactModel as SimImpactModel,
    ScheduleComparison, backtest_schedule_strategies,
};
