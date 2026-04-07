pub mod black_scholes;
pub mod heston;
pub mod sabr;
pub mod svi;
pub mod surface;
pub mod term_structure;
pub mod exotic;
pub mod risk;
pub mod vol_surface_interp;
pub mod greeks_matrix;
pub mod scenario_pnl;

pub use black_scholes::{BlackScholes, OptionType, Greeks};
pub use heston::{HestonModel, HestonParams};
pub use sabr::{SabrModel, SabrParams};
pub use svi::{SviParams, SviModel};
pub use surface::{VolatilitySurface, VolPoint};
pub use term_structure::{YieldCurve, YieldCurveType};
pub use exotic::{BarrierOption, BarrierType, AsianOption, LookbackOption, DigitalOption};
pub use risk::{PortfolioGreeks, PositionGreeks, GreeksAggregator};

/// Common error type for the options engine
#[derive(Debug, thiserror::Error)]
pub enum OptionsError {
    #[error("Numerical convergence failure: {0}")]
    ConvergenceFailure(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Interpolation error: {0}")]
    InterpolationError(String),
    #[error("Model error: {0}")]
    ModelError(String),
    #[error("Arbitrage violation: {0}")]
    ArbitrageViolation(String),
}
