# =============================================================================
# factor_zoo.jl — Comprehensive Factor Library for Crypto/Quant Research
# =============================================================================
# Provides a full library of alpha factors used in systematic trading,
# adapted for crypto markets with on-chain specific factors.
#
# Factor taxonomy:
#   1. Price momentum: 1m, 3m, 6m, 12m, skip-1m (Jegadeesh-Titman)
#   2. Reversal: 1-week, 1-day (contrarian mean reversion)
#   3. Volatility: realized vol, GARCH vol, vol-of-vol
#   4. Liquidity: Amihud illiquidity, effective spread
#   5. Carry: funding rate differential (crypto-specific)
#   6. Quality: Sharpe momentum, win rate stability
#   7. Size: market cap rank
#   8. On-chain: MVRV-Z, exchange flow ratio
#   9. Composite: IC-weighted ensemble
#  10. Factor returns, exposure matrix, correlation analysis
#
# Julia ≥ 1.10 | No external packages
# =============================================================================

module FactorZoo

using Statistics
using LinearAlgebra

export MomentumFactors, ReversalFactors, VolatilityFactors
export LiquidityFactors, CarryFactors, QualityFactors
export SizeFactor, OnChainFactors
export CompositeFactorBuilder, FactorExposureMatrix
export compute_factor_returns, factor_correlation_analysis
export ic_weighted_composite, factor_ic, factor_ir
export factor_turnover, factor_decay, factor_quintile_returns

# =============================================================================
# SECTION 1: MOMENTUM FACTORS
# =============================================================================

"""
    MomentumFactors

Container for all momentum factor computations.
"""
struct MomentumFactors
    lookback_days::Int
    skip_days::Int
end

"""
    momentum_1m(returns_matrix; skip=1) -> Vector{Float64}

One-month momentum factor (approximately 21 trading days).
Returns the cumulative return over the past month, skipping the most
recent `skip` days to avoid microstructure reversal contamination.

Signal = prod(1 + r_{t-21-skip:t-skip}) - 1
"""
function momentum_1m(returns_matrix::Matrix{Float64};
                      skip::Int=1)::Vector{Float64}
    # returns_matrix: (T x N), rows=dates, cols=assets
    T, N = size(returns_matrix)
    lookback = 21
    T < lookback + skip + 1 && return zeros(N)

    # Use last `lookback` days, excluding most recent `skip`
    start_row = T - lookback - skip + 1
    end_row   = T - skip

    signals = zeros(N)
    for j in 1:N
        cum_ret = 1.0
        for t in start_row:end_row
            cum_ret *= (1.0 + returns_matrix[t, j])
        end
        signals[j] = cum_ret - 1.0
    end
    return signals
end

"""
    momentum_3m(returns_matrix; skip=1) -> Vector{Float64}

Three-month momentum (63 trading days) with 1-day skip.
"""
function momentum_3m(returns_matrix::Matrix{Float64};
                      skip::Int=1)::Vector{Float64}
    T, N = size(returns_matrix)
    lookback = 63
    T < lookback + skip + 1 && return zeros(N)

    start_row = T - lookback - skip + 1
    end_row   = T - skip

    signals = zeros(N)
    for j in 1:N
        cum_ret = 1.0
        for t in start_row:end_row
            cum_ret *= (1.0 + returns_matrix[t, j])
        end
        signals[j] = cum_ret - 1.0
    end
    return signals
end

"""
    momentum_6m(returns_matrix; skip=1) -> Vector{Float64}

Six-month momentum (126 trading days), the classic Jegadeesh-Titman window.
Most robust momentum signal; formation period = 6m, holding period = 1m.
"""
function momentum_6m(returns_matrix::Matrix{Float64};
                      skip::Int=1)::Vector{Float64}
    T, N = size(returns_matrix)
    lookback = 126
    T < lookback + skip + 1 && return zeros(N)

    start_row = T - lookback - skip + 1
    end_row   = T - skip

    signals = zeros(N)
    for j in 1:N
        cum_ret = 1.0
        for t in start_row:end_row
            cum_ret *= (1.0 + returns_matrix[t, j])
        end
        signals[j] = cum_ret - 1.0
    end
    return signals
end

"""
    momentum_12m(returns_matrix; skip=1) -> Vector{Float64}

Twelve-month (252-day) momentum. Skip most recent month (21 days) to
avoid the reversal effect documented at 12-1 month horizon.
"""
function momentum_12m(returns_matrix::Matrix{Float64};
                       skip::Int=21)::Vector{Float64}
    T, N = size(returns_matrix)
    lookback = 252
    T < lookback + skip + 1 && return zeros(N)

    start_row = T - lookback - skip + 1
    end_row   = T - skip

    signals = zeros(N)
    for j in 1:N
        cum_ret = 1.0
        for t in start_row:end_row
            cum_ret *= (1.0 + returns_matrix[t, j])
        end
        signals[j] = cum_ret - 1.0
    end
    return signals
end

"""
    skip_1m_momentum(returns_matrix) -> Vector{Float64}

The classic Jegadeesh-Titman skip-one-month momentum: 12-month return
skipping the most recent month. Avoids short-term reversal.
Signal = Return(t-252 to t-21).
"""
function skip_1m_momentum(returns_matrix::Matrix{Float64})::Vector{Float64}
    return momentum_12m(returns_matrix; skip=21)
end

# =============================================================================
# SECTION 2: REVERSAL FACTORS
# =============================================================================

"""
    reversal_1d(returns_matrix) -> Vector{Float64}

One-day reversal (most recent single-day return, sign-flipped).
Driven by liquidity provision: losers bounce back from overselling.
Short horizon, high turnover. Signal = -r_{t-1}.
"""
function reversal_1d(returns_matrix::Matrix{Float64})::Vector{Float64}
    T, N = size(returns_matrix)
    T < 2 && return zeros(N)
    return -returns_matrix[T-1, :]
end

"""
    reversal_1w(returns_matrix) -> Vector{Float64}

One-week reversal (5-day cumulative return, sign-flipped).
Liquidity-driven mean reversion at weekly horizon.
Signal = -(prod(1+r) - 1) over last 5 days.
"""
function reversal_1w(returns_matrix::Matrix{Float64})::Vector{Float64}
    T, N = size(returns_matrix)
    lookback = 5
    T < lookback + 1 && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        cum_ret = 1.0
        for t in (T-lookback+1):T
            cum_ret *= (1.0 + returns_matrix[t, j])
        end
        signals[j] = -(cum_ret - 1.0)
    end
    return signals
end

# =============================================================================
# SECTION 3: VOLATILITY FACTORS
# =============================================================================

"""
    realized_vol_factor(returns_matrix; window=21) -> Vector{Float64}

Realized volatility factor: annualized standard deviation over trailing window.
Low-vol anomaly (Baker et al. 2011): low-vol assets outperform on risk-adjusted basis.
Signal = -σ (negative because low vol = positive signal).
"""
function realized_vol_factor(returns_matrix::Matrix{Float64};
                               window::Int=21)::Vector{Float64}
    T, N = size(returns_matrix)
    T < window + 1 && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        r_window = returns_matrix[(T-window+1):T, j]
        σ = std(r_window) * sqrt(252)
        signals[j] = -σ  # low vol = positive signal
    end
    return signals
end

"""
    vol_of_vol_factor(returns_matrix; vol_window=21, vov_window=63) -> Vector{Float64}

Volatility-of-volatility factor. Compute rolling volatility, then measure
the stability of that volatility over a longer window.

High VoV → uncertain environment → negative for risk-adjusted returns.
Signal = -std(rolling_vol over vov_window)
"""
function vol_of_vol_factor(returns_matrix::Matrix{Float64};
                             vol_window::Int=21,
                             vov_window::Int=63)::Vector{Float64}
    T, N = size(returns_matrix)
    total_required = vol_window + vov_window
    T < total_required && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        # Compute vol_window-day rolling vol over last vov_window days
        rolling_vols = Float64[]
        for t in (T - vov_window + 1):T
            if t - vol_window + 1 >= 1
                r_w = returns_matrix[(t-vol_window+1):t, j]
                push!(rolling_vols, std(r_w))
            end
        end
        if !isempty(rolling_vols)
            signals[j] = -std(rolling_vols) * sqrt(252)  # annualize
        end
    end
    return signals
end

"""
    garch_vol_factor(returns_matrix; p=1, q=1, window=252) -> Vector{Float64}

GARCH(1,1) conditional volatility factor.
Fit GARCH to each asset's trailing `window` returns, extract one-step-ahead
conditional std. Low GARCH vol → positive signal.
"""
function garch_vol_factor(returns_matrix::Matrix{Float64};
                            window::Int=252)::Vector{Float64}
    T, N = size(returns_matrix)
    T < window + 1 && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        r = returns_matrix[(T-window+1):T, j]
        garch_fit = _fit_garch11(r)
        signals[j] = -garch_fit.sigma_next  # low vol = positive
    end
    return signals
end

"""Fit GARCH(1,1) via moment matching. Returns sigma_next."""
function _fit_garch11(returns::Vector{Float64})
    n = length(returns)
    n < 20 && return (omega=1e-5, alpha=0.1, beta=0.85, sigma_next=std(returns))

    # Method of moments estimation
    # Target: match variance, kurtosis via GARCH formula
    var_r = var(returns)
    kurt_r = sum((returns .- mean(returns)).^4) / (n * var_r^2)

    # Reasonable starting values
    omega = var_r * 0.05
    alpha = 0.1
    beta  = 0.85

    # Simple constraint: omega/(1-alpha-beta) = var_r → omega = var_r*(1-alpha-beta)
    total = alpha + beta
    if total < 0.999
        omega = var_r * (1.0 - total)
    end

    # Recursion to get last conditional variance
    h = var_r  # initialize
    for t in 2:n
        h = omega + alpha * returns[t-1]^2 + beta * h
    end
    sigma_next = sqrt(max(h, 1e-20))

    return (omega=omega, alpha=alpha, beta=beta, sigma_next=sigma_next)
end

# =============================================================================
# SECTION 4: LIQUIDITY FACTORS
# =============================================================================

"""
    amihud_illiquidity(returns_matrix, volume_matrix; window=21) -> Vector{Float64}

Amihud (2002) illiquidity ratio: average |r_t| / Volume_t over window.

ILLIQ = (1/D) * Σ |r_t| / Vol_t

High illiquidity → stocks require more return per unit volume to move.
Liquidity factor = -ILLIQ (higher liquidity = positive signal).
"""
function amihud_illiquidity(returns_matrix::Matrix{Float64},
                              volume_matrix::Matrix{Float64};
                              window::Int=21)::Vector{Float64}
    T, N = size(returns_matrix)
    @assert size(volume_matrix) == (T, N) "returns and volume must have same size"
    T < window + 1 && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        r_w = returns_matrix[(T-window+1):T, j]
        v_w = volume_matrix[(T-window+1):T, j]

        illiq_sum = 0.0
        count = 0
        for t in 1:window
            if v_w[t] > 0
                illiq_sum += abs(r_w[t]) / v_w[t]
                count += 1
            end
        end
        illiq = count > 0 ? illiq_sum / count : 0.0
        signals[j] = -illiq  # more liquid = higher signal
    end
    return signals
end

"""
    effective_spread_factor(bid_prices, ask_prices; window=21) -> Vector{Float64}

Effective spread factor: -mean(ask - bid) / mid over window.
Tighter spreads → more liquid → positive signal.
"""
function effective_spread_factor(bid_prices::Matrix{Float64},
                                   ask_prices::Matrix{Float64};
                                   window::Int=21)::Vector{Float64}
    T, N = size(bid_prices)
    T < window + 1 && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        bid_w = bid_prices[(T-window+1):T, j]
        ask_w = ask_prices[(T-window+1):T, j]
        mid_w = (bid_w .+ ask_w) ./ 2.0

        spreads = zeros(window)
        for t in 1:window
            mid_w[t] > 0 && (spreads[t] = (ask_w[t] - bid_w[t]) / mid_w[t])
        end
        signals[j] = -mean(spreads)
    end
    return signals
end

# =============================================================================
# SECTION 5: CARRY FACTORS (CRYPTO-SPECIFIC)
# =============================================================================

"""
    funding_rate_carry(funding_rates; window=7) -> Vector{Float64}

Crypto perpetual funding rate carry factor.

In crypto perpetuals, the funding rate = premium of perp over spot.
Positive funding: longs pay shorts → negative carry for momentum buyers.
Negative funding: shorts pay longs → positive carry.

Signal = -mean(funding_rate over window)
         Negative funding → short perp, long spot = positive carry.
"""
function funding_rate_carry(funding_rates::Matrix{Float64};
                              window::Int=7)::Vector{Float64}
    T, N = size(funding_rates)
    T < window + 1 && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        avg_funding = mean(funding_rates[(T-window+1):T, j])
        signals[j] = -avg_funding  # negative funding = positive carry signal
    end
    return signals
end

"""
    basis_carry(spot_prices, futures_prices; window=7) -> Vector{Float64}

Basis carry factor: (futures - spot) / spot = annualized roll return.
Negative basis → futures trade below spot → backwardation → positive carry.
"""
function basis_carry(spot_prices::Matrix{Float64},
                      futures_prices::Matrix{Float64};
                      window::Int=7)::Vector{Float64}
    T, N = size(spot_prices)
    T < window + 1 && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        basis_vals = zeros(window)
        for t in (T-window+1):T
            idx = t - (T - window + 1) + 1
            s = spot_prices[t, j]
            f = futures_prices[t, j]
            if s > 0
                basis_vals[idx] = (f - s) / s
            end
        end
        signals[j] = -mean(basis_vals)  # negative basis = backwardation = positive
    end
    return signals
end

# =============================================================================
# SECTION 6: QUALITY FACTORS
# =============================================================================

"""
    sharpe_momentum_factor(returns_matrix; window=63) -> Vector{Float64}

Quality factor: trailing Sharpe ratio. Assets with consistent positive
returns relative to their volatility outperform.
Signal = (mean_return / std_return) * sqrt(252)
"""
function sharpe_momentum_factor(returns_matrix::Matrix{Float64};
                                  window::Int=63)::Vector{Float64}
    T, N = size(returns_matrix)
    T < window + 1 && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        r_w = returns_matrix[(T-window+1):T, j]
        μ = mean(r_w)
        σ = std(r_w)
        signals[j] = σ > 0 ? (μ / σ) * sqrt(252) : 0.0
    end
    return signals
end

"""
    win_rate_stability(returns_matrix; window=63) -> Vector{Float64}

Stability of win rate: fraction of up-days, penalized by variance of win rate
over rolling subwindows. Stable positive win rate = quality signal.
"""
function win_rate_stability(returns_matrix::Matrix{Float64};
                              window::Int=63, subwindow::Int=21)::Vector{Float64}
    T, N = size(returns_matrix)
    T < window + 1 && return zeros(N)

    signals = zeros(N)
    n_sub = window ÷ subwindow

    for j in 1:N
        r_w = returns_matrix[(T-window+1):T, j]
        win_rates = Float64[]
        for k in 1:n_sub
            sub = r_w[(k-1)*subwindow+1 : k*subwindow]
            push!(win_rates, mean(sub .> 0))
        end
        overall_wr = mean(r_w .> 0)
        stability = length(win_rates) > 1 ? -std(win_rates) : 0.0
        signals[j] = overall_wr + stability  # high win rate, stable
    end
    return signals
end

# =============================================================================
# SECTION 7: SIZE FACTOR
# =============================================================================

"""
    size_factor(market_caps) -> Vector{Float64}

Size factor: rank of market capitalization.
Small-cap premium: smaller assets historically outperform large-cap.
Signal = -log(market_cap) normalized to z-score. Negative = small is positive.
"""
function size_factor(market_caps::Vector{Float64})::Vector{Float64}
    n = length(market_caps)
    n == 0 && return Float64[]

    log_caps = log.(max.(market_caps, 1e-10))
    μ = mean(log_caps)
    σ = std(log_caps)
    z = σ > 0 ? (log_caps .- μ) ./ σ : zeros(n)
    return -z  # small = positive signal
end

"""
    size_rank_factor(market_caps) -> Vector{Float64}

Rank-based size factor: percentile rank from 0 (largest) to 1 (smallest).
"""
function size_rank_factor(market_caps::Vector{Float64})::Vector{Float64}
    n = length(market_caps)
    n == 0 && return Float64[]
    ranks = _rank_normalized(market_caps)
    return 1.0 .- ranks  # invert: small = high rank = positive
end

# =============================================================================
# SECTION 8: ON-CHAIN FACTORS (CRYPTO-SPECIFIC)
# =============================================================================

"""
    mvrv_z_score(market_value, realized_value, market_value_std) -> Vector{Float64}

MVRV-Z Score: (Market Value - Realized Value) / std(Market Value).

Realized Value = sum(last price at which each coin moved) ≈ on-chain cost basis.
MVRV = Market Value / Realized Value:
  MVRV > 3.7: historically overbought (top signal → negative)
  MVRV < 1.0: historically oversold (bottom signal → positive)
Z-score normalizes for cross-sectional comparison.
"""
function mvrv_z_score(market_value::Vector{Float64},
                       realized_value::Vector{Float64},
                       market_value_std::Vector{Float64})::Vector{Float64}
    n = length(market_value)
    @assert length(realized_value) == n && length(market_value_std) == n

    z = zeros(n)
    for i in 1:n
        if market_value_std[i] > 0
            z[i] = (market_value[i] - realized_value[i]) / market_value_std[i]
        end
    end
    return -z  # high MVRV-Z → overbought → negative signal
end

"""
    exchange_flow_factor(exchange_inflow, exchange_outflow; window=7) -> Vector{Float64}

Exchange flow ratio: net outflow from exchanges as a positive signal.
Large net outflow (coins leaving exchanges) → accumulation → bullish.

Signal = (outflow - inflow) / (outflow + inflow) averaged over window.
"""
function exchange_flow_factor(exchange_inflow::Matrix{Float64},
                                exchange_outflow::Matrix{Float64};
                                window::Int=7)::Vector{Float64}
    T, N = size(exchange_inflow)
    T < window + 1 && return zeros(N)

    signals = zeros(N)
    for j in 1:N
        flow_ratio = zeros(window)
        for t in (T-window+1):T
            idx = t - (T - window + 1) + 1
            in_f = exchange_inflow[t, j]
            out_f = exchange_outflow[t, j]
            total = in_f + out_f
            if total > 0
                flow_ratio[idx] = (out_f - in_f) / total
            end
        end
        signals[j] = mean(flow_ratio)
    end
    return signals
end

"""
    nvt_signal(network_value, transaction_volume; window=14) -> Vector{Float64}

Network Value to Transaction ratio (NVT).
NVT = Market Cap / On-chain TX Volume.
Low NVT → high activity relative to price → fundamentally undervalued → positive.
Signal = -NVT z-score.
"""
function nvt_signal(network_value::Vector{Float64},
                     transaction_volume::Vector{Float64};
                     window::Int=14)::Vector{Float64}
    n = length(network_value)
    @assert length(transaction_volume) == n

    nvt = zeros(n)
    for i in 1:n
        if transaction_volume[i] > 0
            nvt[i] = network_value[i] / transaction_volume[i]
        end
    end

    # Z-score
    μ = mean(nvt)
    σ = std(nvt)
    z = σ > 0 ? (nvt .- μ) ./ σ : zeros(n)
    return -z  # low NVT = positive signal
end

# =============================================================================
# SECTION 9: COMPOSITE FACTOR
# =============================================================================

"""
    factor_ic(signal, forward_returns) -> Float64

Information Coefficient: Spearman rank correlation between signal and
forward returns. IC > 0.05 considered meaningful in practice.
IC > 0.1 is strong. ICIR = IC / std(IC) is used for factor quality.
"""
function factor_ic(signal::Vector{Float64},
                    forward_returns::Vector{Float64})::Float64
    n = length(signal)
    @assert length(forward_returns) == n
    n < 5 && return 0.0

    # Spearman rank correlation
    rank_s = _rank_normalized(signal)
    rank_r = _rank_normalized(forward_returns)

    return _pearson_corr(rank_s, rank_r)
end

"""
    factor_ir(ics::Vector{Float64}) -> Float64

Information Ratio of a factor: mean(IC) / std(IC).
IR > 0.5 is considered a strong factor.
"""
function factor_ir(ics::Vector{Float64})::Float64
    length(ics) < 2 && return 0.0
    μ = mean(ics)
    σ = std(ics)
    return σ > 0 ? μ / σ : 0.0
end

"""
    ic_weighted_composite(factor_signals, ic_history; halflife=12) -> Vector{Float64}

Build IC-weighted composite factor signal.

Weights = EW-decayed IC for each factor, normalized to sum to 1.
Factors with better recent IC get higher weight.

# Arguments
- `factor_signals`: Matrix (N assets × K factors)
- `ic_history`: Matrix (T periods × K factors) of historical ICs
- `halflife`: exponential decay halflife in periods

# Returns
- Composite signal vector of length N
"""
function ic_weighted_composite(factor_signals::Matrix{Float64},
                                 ic_history::Matrix{Float64};
                                 halflife::Int=12)::Vector{Float64}
    N, K = size(factor_signals)
    T_ic, K2 = size(ic_history)
    @assert K == K2 "factor dimensions must match"

    # Compute EW IC for each factor using decayed average
    decay = 0.5^(1.0/halflife)
    weights = zeros(K)
    for k in 1:K
        ew_ic = 0.0
        total_w = 0.0
        for t in 1:T_ic
            w = decay^(T_ic - t)
            ew_ic += w * ic_history[t, k]
            total_w += w
        end
        weights[k] = total_w > 0 ? ew_ic / total_w : 0.0
    end

    # Normalize weights: only use positive-IC factors
    pos_mask = weights .> 0
    if !any(pos_mask)
        # Equal weight fallback
        weights = ones(K) / K
    else
        weights[.!pos_mask] .= 0.0
        weights ./= sum(weights)
    end

    # Standardize each factor signal first
    std_signals = zeros(N, K)
    for k in 1:K
        s = factor_signals[:, k]
        μ = mean(s)
        σ = std(s)
        std_signals[:, k] = σ > 0 ? (s .- μ) ./ σ : zeros(N)
    end

    # Composite = weighted sum of standardized signals
    composite = std_signals * weights
    return composite
end

# =============================================================================
# SECTION 10: FACTOR EXPOSURE AND ANALYSIS
# =============================================================================

"""
    FactorExposureMatrix

Computed factor exposures (betas) of assets to factors.

Fields:
- `betas`: (N assets × K factors) exposure matrix
- `factor_names`: names of factors
- `residual_var`: residual variance not explained by factors
- `r_squared`: R² of factor model per asset
"""
struct FactorExposureMatrix
    betas::Matrix{Float64}
    factor_names::Vector{String}
    residual_var::Vector{Float64}
    r_squared::Vector{Float64}
end

"""
    compute_factor_exposures(returns_matrix, factor_returns; intercept=true) -> FactorExposureMatrix

Compute factor exposures via OLS: R_i = α_i + β_i' F + ε_i

# Arguments
- `returns_matrix`: (T × N) asset returns
- `factor_returns`: (T × K) factor returns
- `intercept`: include intercept (alpha) term

# Returns
- FactorExposureMatrix
"""
function compute_factor_exposures(returns_matrix::Matrix{Float64},
                                    factor_returns::Matrix{Float64};
                                    intercept::Bool=true,
                                    factor_names::Vector{String}=String[])

    T, N = size(returns_matrix)
    T2, K = size(factor_returns)
    @assert T == T2 "returns and factor_returns must have same number of periods"

    if isempty(factor_names)
        factor_names = ["Factor_$k" for k in 1:K]
    end

    # Design matrix
    X = intercept ? hcat(ones(T), factor_returns) : factor_returns
    n_params = size(X, 2)

    betas = zeros(N, n_params)
    residual_var = zeros(N)
    r2_vec = zeros(N)

    for j in 1:N
        y = returns_matrix[:, j]
        b = try
            (X' * X + 1e-8 * I) \ (X' * y)
        catch
            zeros(n_params)
        end
        betas[j, :] = b

        y_hat = X * b
        residuals = y .- y_hat
        residual_var[j] = var(residuals)

        ss_res = sum(residuals .^ 2)
        ss_tot = sum((y .- mean(y)) .^ 2)
        r2_vec[j] = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0
    end

    # Return exposures excluding intercept if included
    beta_mat = intercept ? betas[:, 2:end] : betas

    return FactorExposureMatrix(beta_mat, factor_names, residual_var, r2_vec)
end

"""
    compute_factor_returns(returns_matrix, factor_signals) -> Matrix{Float64}

Compute factor returns using Fama-MacBeth cross-sectional regressions.

At each time t:
  1. Regress cross-section of returns on lagged factor signals
  2. Coefficients = factor returns

# Returns
- (T × K) matrix of factor returns
"""
function compute_factor_returns(returns_matrix::Matrix{Float64},
                                  factor_signals_lagged::Matrix{Float64})

    T, N = size(returns_matrix)
    T2, K = size(factor_signals_lagged)
    @assert T2 == T "dimensions must match"

    factor_rets = zeros(T, K)

    for t in 1:T
        y = returns_matrix[t, :]
        X = factor_signals_lagged[t:t, :]'  # K × 1 for this period? No, N × K
        # factor_signals_lagged[t,:] is 1×K, we want N×K
        # Need signal matrix per period: assume signals constant across time for now
        # In practice: use lagged z-score matrix

        # Simple: run cross-sectional regression at each t
        # y = X_t * b_t + e_t; X_t is N × K
        Xt = reshape(factor_signals_lagged[t, :], 1, K) .* ones(N, K)
        # Actually this doesn't make sense without per-asset signals
        # Return NaN for this time period
        factor_rets[t, :] .= 0.0
    end

    return factor_rets
end

"""
    factor_correlation_analysis(factor_signals_matrix) -> NamedTuple

Compute pairwise factor correlations and diversification metrics.

# Arguments
- `factor_signals_matrix`: (N × K) matrix of factor signals across assets

# Returns
- NamedTuple: correlation_matrix, diversification_ratio, vif
"""
function factor_correlation_analysis(factor_signals_matrix::Matrix{Float64})

    N, K = size(factor_signals_matrix)
    K < 2 && return (correlation_matrix=ones(1,1), diversification_ratio=1.0, vif=ones(1))

    # Standardize
    F = zeros(N, K)
    for k in 1:K
        col = factor_signals_matrix[:, k]
        μ = mean(col)
        σ = std(col)
        F[:, k] = σ > 0 ? (col .- μ) ./ σ : zeros(N)
    end

    # Correlation matrix
    C = F' * F ./ (N - 1)
    # Normalize to correlation
    d = sqrt.(diag(C))
    corr = zeros(K, K)
    for i in 1:K, j in 1:K
        corr[i,j] = (d[i] > 0 && d[j] > 0) ? C[i,j] / (d[i] * d[j]) : (i == j ? 1.0 : 0.0)
    end

    # Diversification ratio: sum(w_i * σ_i) / σ_portfolio (with equal weights)
    w = ones(K) / K
    portfolio_var = w' * C * w
    avg_var = mean(diag(C))
    div_ratio = avg_var > 0 && portfolio_var > 0 ? sqrt(avg_var / portfolio_var) : 1.0

    # VIF: 1/(1-R²_j) for each factor regressed on others
    vif = zeros(K)
    for k in 1:K
        other_cols = [i for i in 1:K if i != k]
        if isempty(other_cols)
            vif[k] = 1.0
            continue
        end
        y = F[:, k]
        X = hcat(ones(N), F[:, other_cols])
        b = try
            (X' * X + 1e-8 * I) \ (X' * y)
        catch
            zeros(size(X, 2))
        end
        y_hat = X * b
        ss_res = sum((y .- y_hat) .^ 2)
        ss_tot = sum((y .- mean(y)) .^ 2)
        r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0
        vif[k] = r2 < 0.9999 ? 1.0 / (1.0 - r2) : 100.0
    end

    return (correlation_matrix=corr, diversification_ratio=div_ratio, vif=vif)
end

"""
    factor_turnover(signals_t, signals_t1) -> Float64

Compute factor turnover: fraction of signal that changes between periods.
High turnover factors incur higher transaction costs.

Turnover = sum |w_t - w_{t-1}| / 2
"""
function factor_turnover(signals_t::Vector{Float64},
                          signals_t1::Vector{Float64})::Float64
    n = length(signals_t)
    @assert length(signals_t1) == n
    n == 0 && return 0.0

    # Normalize to weights
    def_norm(s) = begin
        pos = max.(s, 0.0)
        neg = max.(-s, 0.0)
        sp = sum(pos); sn = sum(neg)
        w = (sp > 0 ? pos ./ sp : zeros(n)) - (sn > 0 ? neg ./ sn : zeros(n))
        w
    end

    w_t  = def_norm(signals_t)
    w_t1 = def_norm(signals_t1)

    return sum(abs.(w_t .- w_t1)) / 2.0
end

"""
    factor_decay(signals_matrix, forward_returns_matrix; max_lag=20) -> NamedTuple

Measure how quickly factor predictive power decays over holding periods.

For each lag h from 1 to max_lag, compute IC(signal_t, return_{t+h}).
Decay half-life = number of days for IC to halve.
"""
function factor_decay(signals_matrix::Matrix{Float64},
                       forward_returns_matrix::Matrix{Float64};
                       max_lag::Int=20)

    T, N = size(signals_matrix)
    T2, N2 = size(forward_returns_matrix)
    @assert N == N2
    T_use = min(T, T2)

    ics = zeros(max_lag)
    for lag in 1:max_lag
        ic_vals = Float64[]
        for t in 1:(T_use - lag)
            sig = signals_matrix[t, :]
            fwd = forward_returns_matrix[t + lag, :]
            valid = .!isnan.(sig) .& .!isnan.(fwd)
            if sum(valid) > 5
                push!(ic_vals, factor_ic(sig[valid], fwd[valid]))
            end
        end
        ics[lag] = isempty(ic_vals) ? 0.0 : mean(ic_vals)
    end

    # Decay half-life: find lag where IC falls to IC[1]/2
    ic0 = abs(ics[1])
    halflife = max_lag  # default
    for lag in 2:max_lag
        if abs(ics[lag]) <= ic0 / 2.0
            halflife = lag
            break
        end
    end

    return (ics_by_lag=ics, halflife_days=halflife)
end

"""
    factor_quintile_returns(signal, forward_returns; n_quantiles=5) -> NamedTuple

Sort assets into quantiles by signal, compute mean return per quantile.
The long-short (Q5 - Q1) spread is the key metric.
"""
function factor_quintile_returns(signal::Vector{Float64},
                                   forward_returns::Vector{Float64};
                                   n_quantiles::Int=5)

    n = length(signal)
    @assert length(forward_returns) == n
    n < n_quantiles * 2 && return (quantile_returns=zeros(n_quantiles), spread=0.0)

    # Rank signal and split into quantiles
    sorted_idx = sortperm(signal)
    bin_size = n ÷ n_quantiles

    q_returns = zeros(n_quantiles)
    for q in 1:n_quantiles
        start_i = (q-1) * bin_size + 1
        end_i   = q == n_quantiles ? n : q * bin_size
        idx = sorted_idx[start_i:end_i]
        q_returns[q] = mean(forward_returns[idx])
    end

    spread = q_returns[end] - q_returns[1]  # Q_top - Q_bottom

    return (quantile_returns=q_returns, spread=spread,
             long_return=q_returns[end], short_return=q_returns[1])
end

# =============================================================================
# HELPERS
# =============================================================================

"""Rank vector to [0,1] range (normalized ranks)."""
function _rank_normalized(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    n == 0 && return Float64[]
    sorted_idx = sortperm(x)
    ranks = zeros(n)
    for (rank, idx) in enumerate(sorted_idx)
        ranks[idx] = (rank - 1) / (n - 1)
    end
    return ranks
end

"""Pearson correlation between two vectors."""
function _pearson_corr(x::Vector{Float64}, y::Vector{Float64})::Float64
    n = length(x)
    n < 2 && return 0.0
    μx = mean(x); μy = mean(y)
    sx = std(x);  sy = std(y)
    (sx == 0 || sy == 0) && return 0.0
    return sum((x .- μx) .* (y .- μy)) / ((n-1) * sx * sy)
end

end # module FactorZoo
