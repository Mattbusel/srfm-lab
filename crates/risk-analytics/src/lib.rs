// risk-analytics: Comprehensive risk analytics library
// Built on quant-math, std-only Rust

/// Value at Risk: historical, parametric, Monte Carlo, EVT, backtesting.
pub mod var;

/// Stress testing: historical scenarios, hypothetical, reverse stress.
pub mod stress;

/// Credit risk: Merton, KMV, hazard rates, Gaussian copula, migration.
pub mod credit;

/// Liquidity risk: Amihud, bid-ask, market impact, L-VaR, scoring.
pub mod liquidity;

/// Drawdown analysis: max drawdown, CDaR, recovery, attribution.
pub mod drawdown;

/// Systemic risk: CoVaR, MES, SRISK, DebtRank, Eisenberg-Noe, contagion.
pub mod systemic;
