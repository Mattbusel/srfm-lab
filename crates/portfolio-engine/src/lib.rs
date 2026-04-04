pub mod covariance;
pub mod optimizer;
pub mod risk;
pub mod factor_model;
pub mod backtest;
pub mod attribution;

pub use covariance::{
    Matrix, Vector,
    sample_covariance, ledoit_wolf, exponential_weighted,
    dcc_garch_correlation, garch11_variance, factor_model_cov,
};
pub use optimizer::{
    mean_variance, min_variance, max_sharpe, risk_parity,
    black_litterman, hierarchical_risk_parity, kelly_portfolio,
    project_simplex, invert_matrix,
};
pub use risk::{
    portfolio_var, portfolio_cvar, component_var, marginal_var,
    parametric_var, historical_var, monte_carlo_var, expected_shortfall,
    stressed_var, var_backtest, VaRBacktest,
    beta, tracking_error, information_ratio,
    sortino_ratio, calmar_ratio, ulcer_index, max_drawdown, sharpe_ratio,
};
pub use factor_model::{
    estimate_loadings, factor_attribution, FactorAttribution,
    residualize, fama_french_3, FF3Result, ols,
};
pub use backtest::{
    run, RebalanceFreq, PortfolioBacktestResult, PortfolioMetrics,
    transaction_cost_impact,
};
pub use attribution::{
    brinson_hood_beebower, BrinsonResult,
    time_period_attribution, PeriodReturn,
    rolling_attribution, cumulative_attribution,
};
