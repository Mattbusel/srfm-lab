// lib.rs — tick-backtest: high-performance BH physics backtest engine
//
// Public module declarations and re-exports.

pub mod bar_from_ticks;
pub mod bh_physics;
pub mod csv_loader;
pub mod engine;
pub mod indicators;
pub mod intraday_patterns;
pub mod monte_carlo;
pub mod multi_engine;
pub mod param_sweep;
pub mod tick_replay;
pub mod types;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

// BH physics core
pub use bh_physics::{BHState, BHUpdate, BlackHoleDetector, HawkingTemperature, MinkowskiClassifier};

// Engine
pub use engine::{BacktestConfig, BacktestEngine, BacktestResult, compute_metrics};

// Multi-instrument engine
pub use multi_engine::{MultiBacktestConfig, MultiBacktestEngine, MultiBacktestResult};

// Parameter sweep
pub use param_sweep::{
    grid_search, random_search, Metric, ParamBounds, ParamGrid, ParamPoint, SweepResult,
};

// Monte Carlo
pub use monte_carlo::{run_mc, run_mc_bootstrap, run_mc_from_stats, MCConfig, MCResult};

// CSV I/O
pub use csv_loader::{
    load_bars_csv, load_barsset, load_trades_csv, save_equity_curve_csv, save_trades_csv, BarsSet,
};

// Sweep CSV export (defined in param_sweep)
pub use param_sweep::save_sweep_csv;

// Core types
pub use types::{
    BacktestMetrics, Bar, DeltaScore, Position, Regime, RegimeStats, TFScore, Trade,
};

// Indicators
pub use indicators::{
    ATR, BBands, BollingerBands, EMA, MACD, MACDValue, Momentum, RSI, RegimeClassifier,
    RollingVol, SMA, VWAP,
};
