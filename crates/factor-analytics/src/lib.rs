//! # factor-analytics
//!
//! Production-grade quantitative factor analytics for a multi-factor equity model.
//!
//! ## Modules
//!
//! * `factors` -- Factor computation (momentum, value, quality, volatility, liquidity, sentiment)
//! * `cross_section` -- Cross-sectional normalization, neutralization, and composite construction
//! * `decay` -- IC decay curves and factor stability analysis
//! * `attribution` -- Brinson-Hood-Beebower and factor return attribution
//! * `portfolio` -- Mean-variance optimization and factor risk models
//! * `backtest` -- Quintile/decile backtesting and Fama-MacBeth regression

pub mod attribution;
pub mod backtest;
pub mod cross_section;
pub mod decay;
pub mod error;
pub mod factors;
pub mod portfolio;

// -- Public API re-exports --

// Error types
pub use error::{FactorError, Result};

// Factor computation
pub use factors::momentum::{
    compute_momentum_factors, compute_panel_momentum, momentum_factor_names, MomentumConfig,
    MomentumFactors,
};
pub use factors::value::{
    compute_panel_value, compute_value_factors, value_factor_names, FundamentalData, ValueFactors,
};
pub use factors::quality::{
    compute_panel_quality, compute_piotroski, compute_quality_factors, quality_factor_names,
    FinancialStatements, PiotroskiFScore, QualityFactors,
};
pub use factors::volatility::{
    compute_panel_volatility, compute_volatility_factors, volatility_factor_names,
    VolatilityConfig, VolatilityFactors,
};
pub use factors::liquidity::{
    compute_liquidity_factors, compute_panel_liquidity, liquidity_factor_names,
    DailyMarketData, LiquidityConfig, LiquidityFactors,
};
pub use factors::sentiment::{
    compute_panel_sentiment, compute_sentiment_factors, sentiment_factor_names,
    AnalystData, EarningsSurpriseData, SentimentFactors, ShortInterestData,
};

// Cross-section
pub use cross_section::normalize::{
    mad_robust_scale, normalize_factor, normalize_factor_matrix, rank_normalize,
    winsorized_zscore, zscore_cross_section, NormalizationMethod,
};
pub use cross_section::neutralize::{
    ols_residualize, sector_and_cap_neutralize, sector_neutralize,
};
pub use cross_section::composite::{
    compute_composite, factor_correlation_matrix, sample_covariance, CompositeMethod,
};

// Decay analysis
pub use decay::ic_decay::{
    compute_ic, compute_ic_by_lag, compute_icir, ic_decay_analysis, IcDecayCurve,
};
pub use decay::factor_stability::{
    compute_stability_report, factor_turnover, rank_autocorrelation, FactorStabilityReport,
};

// Attribution
pub use attribution::brinson::{
    brinson_attribution, brinson_from_positions, BrinsonAttributionReport, SectorAttribution,
};
pub use attribution::factor_attr::{
    compute_factor_attribution, cumulative_factor_attribution, time_series_factor_attribution,
    FactorAttributionResult,
};

// Portfolio
pub use portfolio::optimizer::{
    factor_constrained_portfolio, max_ic_portfolio, max_sharpe_portfolio, min_variance_portfolio,
    OptimizationConstraints, OptimizationResult,
};
pub use portfolio::risk_model::{
    build_risk_model_from_panel, FactorRiskModel, PortfolioRisk,
};

// Backtest
pub use backtest::factor_backtest::{
    assign_buckets, format_backtest_summary, run_factor_backtest, BacktestConfig,
    BucketStats, FactorBacktestResult,
};
pub use backtest::stats::{
    compute_ic_statistics, fama_macbeth, newey_west_se, time_series_regression,
    FamaMacBethResult, IcStatistics,
};
