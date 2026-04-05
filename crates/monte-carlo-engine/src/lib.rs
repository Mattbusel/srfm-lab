//! Monte Carlo Engine — fast simulation of strategy return distributions.
//!
//! This crate provides:
//! - Distribution fitting (moments, L-moments, EVT tail estimation)
//! - Parallel Monte Carlo simulation via Rayon
//! - Three bootstrap resampling methods (Stationary, Circular, Moving Block)
//! - Stress-test scenarios (market crash, crypto winter, etc.)
//!
//! # Quick Start
//! ```rust,no_run
//! use monte_carlo_engine::{
//!     simulation::{fit_distribution, MonteCarloSimulator, SimulationConfig},
//! };
//!
//! let returns = vec![0.01, -0.02, 0.005, 0.03, -0.01];
//! let dist = fit_distribution(&returns);
//! let config = SimulationConfig::default();
//! let results = MonteCarloSimulator::new().run(&config, &dist);
//! println!("Median final equity: {:.2}", results.median_final_equity);
//! ```

pub mod bootstrap;
pub mod simulation;
pub mod stress_tests;

// Re-export primary public types so callers don't need deep module paths.
pub use bootstrap::{
    Bootstrapper, CircularBlockBootstrap, MovingBlockBootstrap, StationaryBootstrap,
};
pub use simulation::{
    fit_distribution, MonteCarloSimulator, ReturnDistribution, SimulationConfig,
    SimulationResults,
};
pub use stress_tests::{
    conditional_drawdown_at_risk, scenario_analysis, tail_risk_contribution, ScenarioResult,
    StressScenario, StressTestEngine,
};
