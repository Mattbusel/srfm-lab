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

pub use almgren_chriss::{AlmgrenChriss, AcParams, AcSchedule};
pub use twap::TwapSlicer;
pub use vwap::VwapSlicer;
pub use impact::{LinearImpact, SquareRootImpact, ImpactModel};
pub use schedule::{ExecutionSchedule, SliceOrder};
