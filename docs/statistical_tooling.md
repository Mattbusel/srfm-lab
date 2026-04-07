# Statistical Tooling Reference

Reference guide for all Julia and R statistical modules in the srfm-lab codebase. Covers purpose, key functions/types, and usage patterns for every module across `julia/src/`, `idea-engine/stats-service/julia/`, `r/R/`, `r/research/`, and `idea-engine/stats-service/r/`.

---

## Julia Modules

Julia is the primary language for performance-critical financial mathematics: large-scale Monte Carlo simulation, numerical optimization, real-time signal computation, and stochastic process calibration. The module set spans two trees:

- `julia/src/` -- production library loaded by the stats service and backtester
- `idea-engine/stats-service/julia/` -- service-layer variants with HTTP route adapters

---

### Core Financial Mathematics

#### `BHPhysics.jl`

**Purpose.** Implements the Black-Hole (BH) confluence engine -- the lab's proprietary multi-factor gravity model. Drives the walk-forward validation harness and cross-sectional factor ranking.

**Key functions/types.**
- `BHEngine` -- struct holding factor weights, gravity coefficients, and regime state
- `compute_confluence(engine, signals)` -- collapses N factor scores into a single confluence signal
- `walk_forward_validate(strategy, data; n_splits, gap)` -- anchored expanding-window WFO returning OOS metrics
- `cross_sectional_rank(universe, factors, date)` -- z-score normalization + tilt ranking across instruments

**Usage example.**
```julia
using BHPhysics
engine = BHEngine(weights=factor_weights, lookback=252)
signal  = compute_confluence(engine, daily_signals)
results = walk_forward_validate(my_strategy, ohlcv_df; n_splits=8, gap=5)
```

---

#### `Stochastic.jl`

**Purpose.** Library of continuous-time stochastic process models used for simulation, calibration, and derivative pricing.

**Key functions/types.**
- `fit_garch11(returns)` -- MLE GARCH(1,1); returns `(ω, α, β)` parameter tuple
- `HestonModel` -- struct for Heston stochastic volatility (v₀, κ, θ, σᵥ, ρ)
- `simulate_heston(model, S0, T, n_paths, n_steps)` -- full-path Euler-Maruyama simulation
- `HawkesProcess` -- self-exciting point process with exponential kernel
- `simulate_hawkes(proc, T)` -- thinning algorithm event simulation
- `OUProcess` -- Ornstein-Uhlenbeck; `fit_ou(spread_series)` returns `(μ, θ, σ)`
- `MertonJump` -- Merton jump-diffusion; `simulate_merton(model, S0, T, n_paths)`

**Usage example.**
```julia
using Stochastic
garch = fit_garch11(log_returns)
heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
paths  = simulate_heston(heston, 100.0, 1.0, 10_000, 252)
```

---

#### `Bayesian.jl`

**Purpose.** Bayesian inference infrastructure: MCMC samplers, characteristic function (CF) estimation, and posterior predictive generation.

**Key functions/types.**
- `MCMCSampler` -- configurable Metropolis-Hastings / NUTS wrapper
- `run_mcmc(model, data; n_iter, burn_in, thin)` -- returns `MCMCChain`
- `bayesian_cf_estimate(returns, prior)` -- CF-based posterior over distribution parameters
- `posterior_predictive(chain, n_samples)` -- draws from the posterior predictive distribution
- `credible_interval(chain, param; level=0.95)` -- HPD interval extraction

**Usage example.**
```julia
using Bayesian
chain = run_mcmc(levy_model, returns; n_iter=20_000, burn_in=5_000)
pred  = posterior_predictive(chain, 1_000)
ci    = credible_interval(chain, :alpha; level=0.95)
```

---

#### `DerivativesPricing.jl`

**Purpose.** Analytical and Monte Carlo option pricing, along with full Greeks computation.

**Key functions/types.**
- `black_scholes(S, K, r, q, σ, T, type)` -- closed-form BSM price
- `bs_greeks(S, K, r, q, σ, T)` -- returns `NamedTuple` with delta, gamma, vega, theta, rho
- `mc_price(payoff_fn, model, S0, T; n_paths, n_steps, antithetic)` -- generic MC pricer
- `ImpliedVol` -- Newton-Raphson IV solver; `implied_vol(market_price, S, K, r, T, type)`
- `barrier_option_price(S, K, H, r, σ, T, type, barrier_type)` -- closed-form barrier pricing

**Usage example.**
```julia
using DerivativesPricing
price  = black_scholes(100.0, 105.0, 0.05, 0.0, 0.2, 0.25, :call)
greeks = bs_greeks(100.0, 105.0, 0.05, 0.0, 0.2, 0.25)
iv     = implied_vol(3.50, 100.0, 105.0, 0.05, 0.25, :call)
```

---

#### `VolatilitySurface.jl` / `volatility_surface.jl`

**Purpose.** Volatility surface construction, calibration, and arbitrage-free model fitting.

**Key functions/types.**
- `SVIParams` -- raw SVI parametrization `(a, b, ρ, m, σ)`; `fit_svi(strikes, ivols, F, T)`
- `SABRModel` -- SABR parameters `(α, β, ρ, ν)`; `calibrate_sabr(strikes, ivols, F, T)`
- `dupire_local_vol(surface, S, t, K)` -- Dupire equation local vol extraction
- `variance_swap_strike(surface, T)` -- model-free variance swap fair strike
- `surface_arbitrage_check(surface)` -- static arbitrage (calendar + butterfly) validation

**Usage example.**
```julia
using VolatilitySurface
svi  = fit_svi(strikes, ivols, forward, expiry)
lvol = dupire_local_vol(svi, S=100.0, t=0.25, K=105.0)
vs   = variance_swap_strike(svi, T=0.25)
```

---

### Risk and Portfolio

#### `RiskManagement.jl`

**Purpose.** Portfolio risk measurement: VaR, CVaR, drawdown analysis, stress scenario application, and hierarchical risk parity weighting.

**Key functions/types.**
- `historical_var(returns, α)` / `parametric_var(returns, α)` / `filtered_var(returns, α)`
- `cvar(returns, α)` -- Expected Shortfall (ES/CVaR)
- `hrp_weights(cov_matrix)` -- Hierarchical Risk Parity via single-linkage clustering
- `stress_scenario(portfolio, scenario_shocks)` -- applies factor shock dictionary to portfolio P&L
- `max_drawdown(equity_curve)` / `calmar_ratio(returns, rf)`

**Usage example.**
```julia
using RiskManagement
var95  = historical_var(daily_pnl, 0.05)
cvar95 = cvar(daily_pnl, 0.05)
w      = hrp_weights(Σ)
```

---

#### `SystemicRisk.jl`

**Purpose.** Systemic and network risk measures drawn from academic literature.

**Key functions/types.**
- `covar(institution_returns, market_returns; quantile=0.05)` -- Adrian-Brunnermeier CoVaR
- `mes(institution_returns, market_returns; quantile=0.05)` -- Marginal Expected Shortfall
- `srisk(mes, leverage, equity_ratio)` -- Brownlees-Engle SRISK
- `eisenberg_noe_clearing(L, e)` -- fixed-point interbank clearing vector
- `debt_rank(network_adj, equity_vec)` -- DebtRank contagion propagation

**Usage example.**
```julia
using SystemicRisk
cv  = covar(bank_rets, mkt_rets; quantile=0.05)
dr  = debt_rank(interbank_matrix, equity_vector)
```

---

#### `BayesianPortfolio.jl`

**Purpose.** Bayesian portfolio construction including Black-Litterman views and robust optimization under uncertainty.

**Key functions/types.**
- `black_litterman(Σ, Π, P, Q, Ω; τ=0.05)` -- posterior return estimate with investor views
- `robust_mean_variance(μ, Σ; uncertainty_set=:ellipsoidal, κ=1.0)` -- robust MVO
- `posterior_efficient_frontier(chain, n_points)` -- MC-based efficient frontier with uncertainty bands

**Usage example.**
```julia
using BayesianPortfolio
mu_bl = black_litterman(Σ, eq_returns, P_views, Q_views, Ω)
w_rob = robust_mean_variance(mu_bl, Σ; κ=2.0)
```

---

#### `PortfolioSimulator.jl`

**Purpose.** Monte Carlo simulation of multi-asset portfolio evolution across time horizons.

**Key functions/types.**
- `simulate_portfolio(w, μ, Σ, T; n_paths, rebalance_freq)` -- returns matrix of equity curves
- `portfolio_cvar_surface(simulated_paths, horizons, quantiles)` -- CVaR across horizon/level grid
- `bootstrap_sharpe(returns; n_boot=10_000)` -- bootstrap distribution of the Sharpe ratio

---

#### `portfolio_construction.jl` / `portfolio_attribution.jl`

**Purpose.** Signal-to-weight pipeline and Brinson-style performance attribution.

**Key functions.**
- `signal_to_weights(signals, method)` -- converts alpha signals to portfolio weights (rank, z-score, Kelly variants)
- `brinson_attribution(portfolio_weights, benchmark_weights, returns)` -- allocation and selection effects
- `factor_attribution(returns, factor_exposures, factor_returns)` -- factor model P&L decomposition

---

### Signal Research

#### `AlphaResearch.jl` / `alpha_research.jl`

**Purpose.** Full alpha signal evaluation pipeline: IC/ICIR computation, factor decay analysis, and quintile portfolio backtesting.

**Key functions/types.**
- `information_coefficient(factor_vals, fwd_returns)` -- Spearman rank IC
- `ic_decay(factor_vals, returns_matrix; max_lag=20)` -- IC across holding periods
- `factor_half_life(ic_series)` -- exponential fit to IC decay curve
- `quintile_backtest(factor_vals, returns; n_quintiles=5)` -- long-short quintile portfolio with turnover metrics
- `icir(ic_series)` -- IC information ratio (IC mean / IC std)

**Usage example.**
```julia
using AlphaResearch
ic   = information_coefficient(momentum_score, fwd_ret_5d)
hl   = factor_half_life(ic_decay(momentum_score, returns_wide))
qbt  = quintile_backtest(momentum_score, returns_1d)
```

---

#### `TimeSeriesML.jl` / `TimeSeriesAdvanced.jl` / `time_series_advanced.jl`

**Purpose.** Advanced time series modeling: SARIMA fitting, Kalman/RTS smoothing, dynamic factor models, and Granger causality testing.

**Key functions/types.**
- `fit_sarima(y; p, d, q, P, D, Q, s)` -- Box-Jenkins SARIMA with AIC-guided order selection
- `KalmanFilter` -- linear Gaussian state-space model; `kalman_smooth(kf, observations)` → RTS smoother
- `DynamicFactorModel` -- DFM with EM estimation; `extract_factors(dfm, panel_data)`
- `granger_causality(x, y; max_lag=10)` -- F-test Granger causality with lag selection

---

#### `SignalProcessing.jl`

**Purpose.** Frequency-domain analysis and wavelet decomposition for financial time series.

**Key functions/types.**
- `wavelet_decompose(signal; wavelet=:db4, levels=5)` -- DWT decomposition
- `spectral_density(returns; method=:periodogram)` -- power spectral density estimation
- `bandpass_filter(signal, f_low, f_high, fs)` -- zero-phase Butterworth bandpass filter
- `cycle_extract(returns; min_period=5, max_period=63)` -- dominant cycle detection via FFT

---

#### `OnlineLearning.jl` / `online_learning.jl`

**Purpose.** Online and adaptive learning algorithms for non-stationary financial data.

**Key functions/types.**
- `OnlineGradientDescent` -- struct with step-size schedule; `update!(ogd, x, y_true)`
- `AdaptiveLearner` -- decaying learning rate with concept drift detection
- `follow_regularized_leader(losses, regularizer)` -- FTRL for sequential prediction
- `online_sharpe_update(running_stats, new_return)` -- incremental Sharpe estimation

---

### Market Microstructure

#### `CryptoMicrostructure.jl` / `market_microstructure.jl`

**Purpose.** Microstructure analysis: spread decomposition, price impact estimation, and illiquidity measurement.

**Key functions/types.**
- `roll_spread(midprices)` -- Roll implicit spread estimator
- `kyle_lambda(trades, midprices)` -- Kyle's lambda price impact coefficient
- `amihud_illiquidity(returns, volume)` -- Amihud ILLIQ ratio
- `hasbrouck_information_share(bid, ask, trades)` -- information share decomposition
- `adverse_selection_component(trades, quotes)` -- spread decomposition into adverse selection, inventory, order processing

---

#### `ExecutionAnalytics.jl` / `execution_analytics.jl`

**Purpose.** Transaction cost analysis and optimal execution scheduling.

**Key functions/types.**
- `vwap_benchmark(fills, market_volume)` -- VWAP slippage vs. interval VWAP
- `twap_schedule(order_size, n_slices, horizon)` -- uniform time-sliced schedule
- `almgren_chriss(order_size, σ, η, γ, T; risk_aversion=0.1)` -- AC optimal execution trajectory
- `implementation_shortfall(decision_price, fill_prices, quantities)` -- IS decomposition
- `market_impact_model(order_size, adv, σ)` -- power-law permanent + transient impact

---

#### `HighFrequencyTools.jl` / `high_frequency_analysis.jl`

**Purpose.** Tick-level data processing, realized volatility estimation, and order flow toxicity.

**Key functions/types.**
- `realized_variance(tick_prices; kernel=:flat_top)` -- kernel-based realized variance
- `bipower_variation(tick_prices)` -- jump-robust BPV
- `vpin(volume_buckets, buy_vol, sell_vol)` -- Volume-Synchronized PIN (toxicity)
- `tick_rule(trades)` -- Lee-Ready trade direction classification
- `aggregate_ticks(ticks, bar_type; bar_size)` -- tick/volume/dollar bar construction

---

### Crypto-Specific

#### `CryptoDefi.jl` / `crypto_defi.jl`

**Purpose.** DeFi protocol mathematics: AMM pricing, Uniswap v3 concentrated liquidity, impermanent loss, and MEV analysis.

**Key functions/types.**
- `cpamm_price(reserve_x, reserve_y)` -- constant product AMM spot price
- `uniswap_v3_liquidity(amount_x, amount_y, price, price_lower, price_upper)` -- LP position liquidity
- `impermanent_loss(initial_price, current_price)` -- IL fraction vs. holding
- `mev_sandwich_pnl(victim_size, pool_depth, slippage)` -- sandwich attack expected PnL
- `arb_profit(price_a, price_b, size, fee_a, fee_b)` -- cross-pool arbitrage profitability

---

#### `CryptoMechanics.jl` / `crypto_mechanics.jl`

**Purpose.** Crypto derivatives mechanics: basis trading, funding rate arbitrage, and cross-exchange spread analysis.

**Key functions/types.**
- `basis(spot_price, futures_price, T)` -- annualized basis (carry)
- `funding_rate_arb(perp_funding, spot_borrow, hedge_cost)` -- net funding rate arbitrage yield
- `cross_exchange_spread(bids_a, asks_b)` -- best cross-venue spread with fee adjustment
- `basis_trade_pnl(entry_basis, exit_basis, notional)` -- realized carry P&L

---

#### `CryptoOnChain.jl`

**Purpose.** On-chain analytics and market cycle signals derived from blockchain data.

**Key functions/types.**
- `mvrv_ratio(market_cap, realized_cap)` -- Market Value to Realized Value
- `sopr(spent_output_value, cost_basis)` -- Spent Output Profit Ratio
- `exchange_reserve_delta(inflows, outflows)` -- net exchange flow signal
- `hash_rate_ribbon(hash_rate; ma_short=30, ma_long=60)` -- miner capitulation detector
- `nvt_ratio(network_value, tx_volume)` -- Network Value to Transactions

---

### Machine Learning

#### `ReinforcementLearning.jl` / `reinforcement_learning.jl`

**Purpose.** RL agents for trading: tabular Q-learning, policy gradient methods, and DQN with experience replay.

**Key functions/types.**
- `QLearningAgent` -- tabular Q with epsilon-greedy; `update!(agent, s, a, r, s_next)`
- `PolicyGradientAgent` -- REINFORCE with baseline
- `DQNAgent` -- deep Q-network with replay buffer and target network; `train_step!(agent, batch)`
- `TradingEnvironment` -- OpenAI-gym-style env wrapping OHLCV data

---

#### `AlternativeData.jl` / `alternative_data.jl`

**Purpose.** Processing and signal extraction from non-traditional data sources.

**Key functions/types.**
- `options_flow_signal(unusual_activity, open_interest_change)` -- options flow skew signal
- `dark_pool_signal(dark_volume_ratio, price_return)` -- dark pool print interpretation
- `satellite_sentiment(activity_index, baseline)` -- satellite-derived activity signal
- `aggregate_alt_signals(signal_dict; method=:rank_average)` -- ensemble combination

---

#### `MacroFactors.jl`

**Purpose.** Macro regime and factor signal construction from cross-asset data.

**Key functions/types.**
- `vix_regime(vix_level; low=15.0, high=25.0)` -- three-state VIX regime classifier
- `dxy_signal(dxy_returns, lag=5)` -- DXY momentum signal for risk assets
- `yield_curve_slope(t2y, t10y)` -- 2s10s slope with inversion flag
- `equity_momentum_factor(sector_returns; lookback=252, skip=21)` -- cross-sector momentum

---

## R Modules

R is used for statistical testing with small-to-medium datasets, Hidden Markov Models, White's Reality Check, regime analysis, and publication-quality visualization. The module set spans three directories:

- `r/R/` -- production R library
- `r/research/` -- standalone research scripts and studies
- `idea-engine/stats-service/r/` -- service-layer R modules

---

### Module Reference

| Module | Location | Purpose |
|---|---|---|
| `bh_analysis.R` | `r/R/` | BH confluence signal reconstruction and sensitivity analysis in R |
| `regime_models.R` | `r/R/` | HMM-based regime detection (depmixS4), regime-conditional statistics |
| `crypto_analytics.R` | `r/R/` | Crypto return distributions, correlation, rolling beta to BTC |
| `factor_research.R` | `r/R/` | Fama-French style factor construction, cross-sectional regressions |
| `volatility_models.R` | `r/R/` `idea-engine/stats-service/r/` | GARCH family (rugarch), realized vol, HAR-RV model |
| `spectral_analysis.R` | `idea-engine/stats-service/r/` | Spectral density, wavelet coherence (WaveletComp), cycle detection |
| `copula_analysis.R` | `idea-engine/stats-service/r/` | Gaussian/Student-t/Clayton copulas (copula package), tail dependence |
| `bayesian_portfolio.R` | `idea-engine/stats-service/r/` | Black-Litterman in R, MCMCpack posterior sampling |
| `systemic_risk.R` | `r/R/` `idea-engine/stats-service/r/` | CoVaR, MES, SRISK, network contagion (igraph) |
| `credit_risk.R` | `idea-engine/stats-service/r/` | Merton structural model, CDS spread calibration |
| `advanced_ml.R` | `idea-engine/stats-service/r/` | Random forests, gradient boosting (xgboost), LASSO signal selection |
| `alternative_data.R` | `r/R/` `idea-engine/stats-service/r/` | Text sentiment scoring, options flow signal aggregation |
| `execution_analytics.R` | `r/R/` `idea-engine/stats-service/r/` | TCA in R, VWAP/IS benchmarks, regression on fill quality |
| `numerical_methods.R` | `idea-engine/stats-service/r/` | Numerical optimization (optim), root finding, quadrature |
| `portfolio_attribution.R` | `r/R/` `idea-engine/stats-service/r/` | Brinson attribution, factor model P&L decomposition |
| `crypto_mechanics.R` | `idea-engine/stats-service/r/` | Funding rate time series analysis, basis seasonality |
| `volatility_surface.R` | `r/R/` `idea-engine/stats-service/r/` | SVI fitting, SABR calibration, surface smoothing |
| `defi_analytics.R` | `r/R/` `idea-engine/stats-service/r/` | LP return attribution, IL distribution analysis |
| `time_series_advanced.R` | `r/R/` `idea-engine/stats-service/r/` | VECM, structural breaks (strucchange), cointegration tests |
| `stress_testing.R` | `r/R/` | Historical scenario replays, factor shock analysis |
| `signal_research.R` | `r/R/` | White's Reality Check, SPA test, multiple testing correction |
| `advanced_portfolio.R` | `r/R/` | Robust optimization (CVXR), risk budgeting, min-CVaR |
| `walk_forward_analysis.R` | `idea-engine/stats-service/r/` | Walk-forward OOS evaluation with anchored/rolling windows |
| `high_frequency_stats.R` | `idea-engine/stats-service/r/` | Realized variance (highfrequency package), microstructure noise tests |

**Research scripts (`r/research/`).**
- `backtesting_framework.R` -- generic vectorized backtest with transaction costs
- `cross_asset_study.R` -- cross-asset correlation regime analysis
- `crypto_research_toolkit.R` -- convenience wrappers for crypto data + analytics
- `econometrics.R` -- panel data regressions (plm), IV estimation
- `hypothesis_testing.R` -- bootstrapped hypothesis tests, permutation tests
- `ml_backtesting.R` -- purged k-fold CV for financial ML
- `regime_trading_study.R` -- regime-conditional strategy simulation
- `volatility_research.R` -- realized/implied vol comparison studies
- `simulation_studies.R` -- Monte Carlo simulation experiments in R

---

### Usage Examples

```r
# Regime detection with HMM
source("r/R/regime_models.R")
regimes <- fit_hmm(crypto_returns, n_states=3)
plot_regime_overlay(regimes, prices)

# White's Reality Check for multiple strategies
source("r/R/signal_research.R")
pval <- whites_reality_check(strategy_returns_matrix, B=1000)

# Walk-forward validation
source("idea-engine/stats-service/r/walk_forward_analysis.R")
wf <- walk_forward(strategy_fn, data, n_splits=8, gap=5)
summary(wf$oos_sharpes)
```

---

## When to Use Julia vs R

| Criterion | Julia | R |
|---|---|---|
| Monte Carlo simulation (>10K paths) | Yes -- 10-100x faster | No -- too slow |
| Large-scale numerical optimization | Yes -- native JIT, sparse linear algebra | Only for small problems |
| Real-time / online signal computation | Yes | No |
| HMM regime detection | Possible | Yes -- depmixS4 is mature |
| White's Reality Check / SPA test | Custom implementation needed | Yes -- well-tested R packages |
| Granger causality, cointegration | Yes | Yes -- vars, urca packages |
| Cross-sectional statistical tests | Either | R preferred for small N |
| Visualization / reporting | Plots.jl (adequate) | ggplot2 (preferred) |
| Stochastic process calibration | Yes -- performance-critical | Only for prototype |
| GARCH family estimation | Yes -- custom GARCH(1,1) | Yes -- rugarch (richer family) |
| Bayesian MCMC | Yes -- custom samplers | MCMCpack, rstan |
| Factor model / panel regression | AlphaResearch.jl | plm, lfe packages |
| Walk-forward validation | BHPhysics.jl (anchored WFO) | walk_forward_analysis.R |

**Rule of thumb.** If the computation needs to run in a backtest loop, in the stats service, or over thousands of simulations, use Julia. If you need a battle-tested statistical package, need to interface with R-specific econometrics literature, or are producing a research notebook for human review, use R.
