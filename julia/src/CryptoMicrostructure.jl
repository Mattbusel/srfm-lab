"""
CryptoMicrostructure — Crypto-specific market microstructure analysis.

Implements: Order flow imbalance, Lee-Ready trade sign inference,
effective/realised spread, Hasbrouck information share, Roll's model,
variance ratio test, Hurst exponent (R/S + DFA), intraday periodicity,
and quote stuffing detection.
"""
module CryptoMicrostructure

using Statistics
using LinearAlgebra
using Random

export OrderFlowImbalance, LeeReadySign, SpreadDecomposition
export HasbrouckInfoShare, RollModel, VarianceRatioTest
export HurstExponent, IntradayPeriodicity, QuoteStuffingDetection
export run_microstructure_analysis

# ─────────────────────────────────────────────────────────────────────────────
# Internal Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Lag a vector by k periods (prepend NaN)."""
function _lag(x::Vector{Float64}, k::Int=1)::Vector{Float64}
    n = length(x)
    y = fill(NaN, n)
    k < n && (y[(k+1):end] = x[1:(n-k)])
    return y
end

"""Running mean in a vector."""
function _running_mean(x::Vector{Float64}, w::Int)::Vector{Float64}
    n = length(x)
    out = fill(NaN, n)
    for i in w:n
        out[i] = mean(x[(i-w+1):i])
    end
    return out
end

"""OLS regression: returns (alpha, beta, residuals, R2)."""
function _ols(y::Vector{Float64}, X::Matrix{Float64})
    n, k = size(X)
    XtX = X' * X + 1e-10 * I
    β = XtX \ (X' * y)
    ŷ = X * β
    e = y .- ŷ
    SS_tot = sum((y .- mean(y)).^2)
    SS_res = sum(e.^2)
    R2 = 1.0 - SS_res / max(SS_tot, 1e-10)
    return (coef=β, fitted=ŷ, residuals=e, R2=R2)
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Order Flow Imbalance
# ─────────────────────────────────────────────────────────────────────────────

"""
    OrderFlowImbalance(bid_vol, ask_vol; window) → NamedTuple

Compute Order Flow Imbalance (OFI): the signed pressure of buyer vs seller
initiated volume at each timestamp.

OFI_t = (bid_vol_t - ask_vol_t) / (bid_vol_t + ask_vol_t)

# Arguments
- `bid_vol`  : bid-side volume at each timestamp
- `ask_vol`  : ask-side volume at each timestamp
- `window`   : smoothing window for rolling OFI (default 10)

# Returns
NamedTuple: (ofi, rolling_ofi, imbalance_sign, vol_total, cumulative_ofi)

# Example
```julia
n = 1000
bid = abs.(randn(n)) .* 100 .+ 50
ask = abs.(randn(n)) .* 100 .+ 50
result = OrderFlowImbalance(bid, ask; window=20)
```
"""
function OrderFlowImbalance(bid_vol::Vector{Float64}, ask_vol::Vector{Float64};
                             window::Int=10)
    length(bid_vol) == length(ask_vol) || error("bid and ask volumes must match")
    n = length(bid_vol)

    total_vol = bid_vol .+ ask_vol
    total_vol = max.(total_vol, 1e-10)

    # Raw OFI
    ofi = (bid_vol .- ask_vol) ./ total_vol

    # Rolling OFI
    rolling_ofi = _running_mean(ofi, window)

    # Imbalance sign: 1 = buy pressure, -1 = sell pressure, 0 = neutral
    imbalance_sign = sign.(ofi)

    # Cumulative OFI
    cumulative_ofi = cumsum(ofi)

    # Absorptive capacity: how quickly OFI mean-reverts
    lag_ofi = _lag(ofi)
    valid = .!isnan.(lag_ofi)
    autocorr_1 = if sum(valid) > 2
        cov(ofi[valid], lag_ofi[valid]) / max(var(ofi[valid]) * var(lag_ofi[valid])^0.5, 1e-10)
    else
        0.0
    end

    # Volumetric concentration: Herfindahl-Hirschman of size distribution
    hhi = sum((v / sum(total_vol))^2 for v in total_vol)

    # Buy/sell volume ratio
    buy_vol = sum(bid_vol[ofi .> 0.1])
    sell_vol = sum(ask_vol[ofi .< -0.1])
    bs_ratio = buy_vol / max(sell_vol, 1e-10)

    return (ofi=ofi, rolling_ofi=rolling_ofi, imbalance_sign=imbalance_sign,
            vol_total=total_vol, cumulative_ofi=cumulative_ofi,
            autocorr_1=autocorr_1, hhi=hhi, buy_sell_ratio=bs_ratio)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Lee-Ready Trade Sign Inference
# ─────────────────────────────────────────────────────────────────────────────

"""
    LeeReadySign(prices, bid, ask; tick_test_fallback) → NamedTuple

Infer trade direction using the Lee-Ready (1991) algorithm.

Rule:
1. Quote test: compare trade price to bid-ask midpoint
   - price > midpoint → buyer-initiated (+1)
   - price < midpoint → seller-initiated (-1)
   - price = midpoint → use tick test
2. Tick test: compare to previous trade price
   - uptick → buyer (+1), downtick → seller (-1)
   - zero uptick / zero downtick → use previous non-zero tick

# Arguments
- `prices` : trade price vector
- `bid`    : bid quote at each trade
- `ask`    : ask quote at each trade
- `tick_test_fallback` : if true, use tick test when bid/ask unavailable

# Returns
NamedTuple: (signs, buy_fraction, sell_fraction, effective_spread, midpoints)
"""
function LeeReadySign(prices::Vector{Float64}, bid::Vector{Float64},
                       ask::Vector{Float64}; tick_test_fallback::Bool=true)
    n = length(prices)
    (length(bid) == n && length(ask) == n) || error("bid/ask must match prices length")

    signs = zeros(Int, n)
    midpoints = (bid .+ ask) ./ 2.0

    for i in 1:n
        if ask[i] > bid[i] && !isnan(bid[i]) && !isnan(ask[i])
            mid = midpoints[i]
            if prices[i] > mid
                signs[i] = 1   # buyer-initiated
            elseif prices[i] < mid
                signs[i] = -1  # seller-initiated
            else
                # Tie → tick test
                signs[i] = _tick_test(prices, i)
            end
        elseif tick_test_fallback
            signs[i] = _tick_test(prices, i)
        end
    end

    buy_mask  = signs .> 0
    sell_mask = signs .< 0
    buy_frac  = sum(buy_mask) / n
    sell_frac = sum(sell_mask) / n

    # Effective spread per trade
    eff_spread = 2.0 .* abs.(prices .- midpoints) .* signs

    return (signs=signs, buy_fraction=buy_frac, sell_fraction=sell_frac,
            effective_spread=eff_spread, midpoints=midpoints,
            n_buy=sum(buy_mask), n_sell=sum(sell_mask))
end

"""Tick test: +1 if uptick, -1 if downtick, propagate for zero-ticks."""
function _tick_test(prices::Vector{Float64}, i::Int)::Int
    i == 1 && return 0
    for j in (i-1):-1:1
        Δ = prices[i] - prices[j]
        Δ > 0 && return 1   # uptick
        Δ < 0 && return -1  # downtick
    end
    return 0
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Effective Spread, Realised Spread, Price Impact
# ─────────────────────────────────────────────────────────────────────────────

"""
    SpreadDecomposition(prices, signs, midpoints; τ) → NamedTuple

Decompose the effective spread into:
- Realised spread (dealer's gross profit)
- Price impact (adverse selection component)

Effective spread = 2 · sign_t · (p_t - m_t)
Price impact     = 2 · sign_t · (m_{t+τ} - m_t)   [Glosten-Harris]
Realised spread  = 2 · sign_t · (p_t - m_{t+τ})

# Arguments
- `prices`    : transaction prices
- `signs`     : trade signs (+1/-1) from LeeReadySign
- `midpoints` : bid-ask midpoints
- `τ`         : lag for price-impact window (default 5 trades)
"""
function SpreadDecomposition(prices::Vector{Float64}, signs::Vector{Int},
                              midpoints::Vector{Float64}; τ::Int=5)
    n = length(prices)
    n_valid = n - τ

    eff_spread      = 2.0 .* signs .* (prices .- midpoints)
    price_impact    = zeros(n)
    realised_spread = zeros(n)

    for t in 1:n_valid
        price_impact[t]    = 2.0 * signs[t] * (midpoints[t+τ] - midpoints[t])
        realised_spread[t] = 2.0 * signs[t] * (prices[t] - midpoints[t+τ])
    end

    valid = 1:n_valid
    return (effective_spread    = mean(abs.(eff_spread[valid])),
            price_impact        = mean(price_impact[valid]),
            realised_spread     = mean(realised_spread[valid]),
            eff_spread_series   = eff_spread,
            price_impact_series = price_impact,
            adverse_selection_pct = mean(price_impact[valid]) /
                                     max(mean(abs.(eff_spread[valid])), 1e-10) * 100)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Hasbrouck's Information Share
# ─────────────────────────────────────────────────────────────────────────────

"""
    HasbrouckInfoShare(prices_matrix; max_lags) → NamedTuple

Estimate Hasbrouck's (1995) information share for price discovery
across multiple exchanges.

IS_i = (e_i · Σ_u · e_i') / (e · Σ_u · e')
where e is the cointegrating vector and Σ_u is the covariance of innovations.

# Arguments
- `prices_matrix` : n × K matrix of log-prices across K venues
- `max_lags`      : VAR lag order (default 5)

# Returns
NamedTuple: (info_share_upper, info_share_lower, info_share_mid, lead_lag_matrix)
"""
function HasbrouckInfoShare(prices_matrix::Matrix{Float64}; max_lags::Int=5)
    n, K = size(prices_matrix)
    returns = diff(prices_matrix, dims=1)   # (n-1) × K
    m = size(returns, 1)

    # Fit reduced-form VAR(p): Δp_t = ∑ A_k Δp_{t-k} + e_t
    p = min(max_lags, m÷10)
    p = max(p, 1)
    n_eff = m - p

    # Build regressor matrix [Δp_{t-1}, ..., Δp_{t-p}]
    Y = returns[(p+1):end, :]    # n_eff × K
    X = zeros(n_eff, K*p + 1)
    X[:, 1] .= 1.0
    for lag in 1:p
        X[:, (1+(lag-1)*K):(lag*K)] = returns[(p+1-lag):(end-lag), :]
    end

    # OLS equation by equation
    resids = zeros(n_eff, K)
    for k in 1:K
        XtX = X' * X + 1e-8*I
        β_k = XtX \ (X' * Y[:, k])
        resids[:, k] = Y[:, k] .- X * β_k
    end

    Σ_u = cov(resids)   # K × K innovation covariance

    # Cointegrating vector for random-walk component (all-ones for common price)
    e_vec = ones(K)

    # Cholesky factor for bounds on information share
    F_chol = try cholesky(Σ_u + 1e-8*I).L catch; I(K)*0.01 end

    # Information share: IS_i ∈ [IS_lower_i, IS_upper_i]
    info_share_upper = zeros(K)
    info_share_lower = zeros(K)

    for ordering in [1:K, K:-1:1]
        Σ_perm = Σ_u[ordering, ordering]
        F_perm = try cholesky(Σ_perm + 1e-8*I).L catch; I(K)*0.01 end
        denom = (e_vec' * F_perm * F_perm' * e_vec)[1]
        for (rank, k) in enumerate(ordering)
            numer = (e_vec[ordering[1:rank]]' * F_perm[1:rank, :] * F_perm' * e_vec)[1]
            IS_k = rank == 1 ? (e_vec[k] * F_perm[rank, rank])^2 / denom :
                               ((e_vec[ordering[1:rank]]' * F_perm[1:rank, :])' *
                                 (e_vec[ordering[1:rank]]' * F_perm[1:rank, :])[1] / denom)
            # Simplified: use diagonal approximation
            IS_k = (e_vec[k]^2 * Σ_u[k,k]) / max(e_vec' * Σ_u * e_vec, 1e-10)
            if ordering == collect(1:K)
                info_share_upper[k] = IS_k
            else
                info_share_lower[k] = IS_k
            end
        end
    end

    info_share_mid = (info_share_upper .+ info_share_lower) ./ 2.0
    info_share_mid ./= sum(info_share_mid)   # normalise

    # Lead-lag matrix: cross-correlation of returns
    lead_lag = zeros(K, K)
    for i in 1:K, j in 1:K
        i == j && continue
        r_i = returns[2:end, i]; r_j = returns[1:(end-1), j]
        lead_lag[i,j] = cov(r_i, r_j) / max(std(r_i)*std(r_j), 1e-10)
    end

    return (info_share_upper=info_share_upper, info_share_lower=info_share_lower,
            info_share_mid=info_share_mid, lead_lag_matrix=lead_lag,
            innovation_cov=Σ_u, n_venues=K, var_lags=p)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Roll's Model for Spread Estimation
# ─────────────────────────────────────────────────────────────────────────────

"""
    RollModel(prices) → NamedTuple

Estimate effective bid-ask spread from price data using Roll's (1984) model.

Roll's estimator: s = 2√(-Cov(Δp_t, Δp_{t-1}))
based on the observation that bid-ask bounces induce negative autocorrelation.

# Arguments
- `prices` : transaction price vector (or daily close prices)

# Returns
NamedTuple: (spread_estimate, autocovariance, effective_spread_bps,
             price_impact_component, roll_model_R2)
"""
function RollModel(prices::Vector{Float64})
    n = length(prices)
    returns = diff(log.(max.(prices, 1e-10)))
    m = length(returns)

    # Lag-1 autocovariance
    acov = cov(returns[2:end], returns[1:(end-1)])

    # Roll spread estimate
    spread = if acov < 0
        2.0 * sqrt(-acov)
    else
        0.0  # model breaks down (non-negative autocovariance)
    end

    # Effective spread in basis points
    avg_price = mean(prices)
    spread_bps = spread / avg_price * 10000

    # Variance decomposition
    σ2_r = var(returns)
    # σ2 = 2c² (Roll's model); price impact adds σ²_u
    σ2_bounce = 2.0 * (spread/2.0)^2
    σ2_impact = max(σ2_r - σ2_bounce, 0.0)
    price_impact_frac = σ2_impact / max(σ2_r, 1e-10)

    # Model fit: predicted autocovariance should be -c²
    predicted_acov = -(spread/2.0)^2
    roll_r2 = 1.0 - (acov - predicted_acov)^2 / max(acov^2, 1e-10)

    # Daily turnover proxy
    vol_proxy = std(returns) * avg_price * sqrt(252)

    return (spread_estimate=spread, autocovariance=acov,
            effective_spread_bps=spread_bps, price_impact_frac=price_impact_frac,
            roll_model_R2=clamp(roll_r2, 0.0, 1.0),
            vol_proxy=vol_proxy, avg_price=avg_price)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Variance Ratio Test
# ─────────────────────────────────────────────────────────────────────────────

"""
    VarianceRatioTest(prices; q_values, heteroskedastic) → NamedTuple

Lo-MacKinlay (1988) variance ratio test for the random walk hypothesis.

VR(q) = Var(r_t + ... + r_{t+q-1}) / (q · Var(r_t))
Under RWH, VR(q) = 1 for all q.

# Arguments
- `prices`          : price vector
- `q_values`        : holding periods to test (default [2,4,8,16])
- `heteroskedastic` : if true, use HC-robust z-statistic (default true)

# Returns
NamedTuple: (VR, Z_stats, p_values, reject_RWH, Hurst_approx)
"""
function VarianceRatioTest(prices::Vector{Float64};
                            q_values::Vector{Int}=Int[2,4,8,16],
                            heteroskedastic::Bool=true)
    n = length(prices)
    r = diff(log.(max.(prices, 1e-10)))
    m = length(r)

    mu = mean(r)
    sigma2_1 = var(r)

    VR_vec    = Float64[]
    Z_vec     = Float64[]
    p_vec     = Float64[]

    for q in q_values
        q >= m && continue

        # q-period returns
        n_q = m - q + 1
        r_q = [sum(r[t:(t+q-1)]) for t in 1:(m-q+1)]
        sigma2_q = var(r_q) / q

        VR = sigma2_q / max(sigma2_1, 1e-10)
        push!(VR_vec, VR)

        # Asymptotic variance of VR
        if heteroskedastic
            # Heteroskedasticity-robust (Lo-MacKinlay)
            δ = zeros(q-1)
            for j in 1:(q-1)
                num = sum((r[t] - mu)^2 * (r[t-j] - mu)^2 for t in (j+1):m)
                den = (sum((r[t] - mu)^2 for t in 1:m))^2
                δ[j] = num / max(den, 1e-20)
            end
            θ_q = sum(((2*(q-j)/q)^2 * δ[j]) for j in 1:(q-1))
        else
            θ_q = 2*(2*q - 1)*(q-1) / (3*q*m)
        end

        Z = (VR - 1.0) / max(sqrt(θ_q), 1e-10)
        push!(Z_vec, Z)

        # Two-sided p-value (normal approximation)
        p = 2.0 * (1.0 - _norm_cdf(abs(Z)))
        push!(p_vec, p)
    end

    reject_RWH = any(p_vec .< 0.05)

    # Hurst exponent from VR: VR(q) ≈ q^{2H-1}, so H = (log VR(q)/log q + 1)/2
    H_estimates = Float64[]
    for (i, q) in enumerate(q_values[1:length(VR_vec)])
        q > 1 && VR_vec[i] > 0 &&
            push!(H_estimates, (log(VR_vec[i]) / log(q) + 1.0) / 2.0)
    end
    H_approx = isempty(H_estimates) ? 0.5 : clamp(mean(H_estimates), 0.0, 1.0)

    return (VR=VR_vec, Z_stats=Z_vec, p_values=p_vec,
            q_values=q_values[1:length(VR_vec)],
            reject_RWH=reject_RWH, Hurst_approx=H_approx)
end

"""Normal CDF (inline)."""
function _norm_cdf(x::Float64)::Float64
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))))
    p = 1.0 - exp(-0.5*x^2)/sqrt(2π) * poly
    x >= 0.0 ? p : 1.0 - p
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Hurst Exponent
# ─────────────────────────────────────────────────────────────────────────────

"""
    HurstExponent(x; method, min_n, max_n, n_scales) → NamedTuple

Estimate the Hurst exponent via R/S analysis and Detrended Fluctuation Analysis.

- H < 0.5: anti-persistent (mean-reverting)
- H = 0.5: random walk (no memory)
- H > 0.5: persistent (trending)

# Arguments
- `x`        : time series (prices or returns)
- `method`   : :rs (R/S), :dfa, or :both (default :both)
- `min_n`    : minimum sub-series length (default 10)
- `max_n`    : maximum sub-series length (default n÷4)
- `n_scales` : number of logarithmically-spaced scales (default 15)
"""
function HurstExponent(x::Vector{Float64}; method::Symbol=:both,
                        min_n::Int=10, max_n::Int=0, n_scales::Int=15)
    n = length(x)
    max_n_eff = max_n > 0 ? max_n : n ÷ 4
    max_n_eff = max(max_n_eff, min_n + 1)

    # Logarithmic scale grid
    scales = unique(round.(Int, exp.(range(log(min_n), log(max_n_eff), length=n_scales))))
    filter!(s -> s >= min_n && s <= max_n_eff, scales)

    H_rs = H_dfa = 0.5

    if method == :rs || method == :both
        RS_vec = Float64[]
        for s in scales
            rs = _rs_statistic(x, s)
            rs > 0 && push!(RS_vec, rs)
        end
        if length(RS_vec) >= 3
            log_s = log.(Float64.(scales[1:length(RS_vec)]))
            log_RS = log.(RS_vec)
            slope, _ = _linreg(log_s, log_RS)
            H_rs = clamp(slope, 0.0, 1.0)
        end
    end

    if method == :dfa || method == :both
        DFA_vec = Float64[]
        for s in scales
            fa = _dfa_fluctuation(x, s)
            fa > 0 && push!(DFA_vec, fa)
        end
        if length(DFA_vec) >= 3
            log_s  = log.(Float64.(scales[1:length(DFA_vec)]))
            log_F  = log.(DFA_vec)
            slope, _ = _linreg(log_s, log_F)
            H_dfa = clamp(slope, 0.0, 1.0)
        end
    end

    H_combined = if method == :both
        0.5 * H_rs + 0.5 * H_dfa
    elseif method == :rs
        H_rs
    else
        H_dfa
    end

    interpretation = if H_combined < 0.45
        "anti-persistent (mean-reverting)"
    elseif H_combined > 0.55
        "persistent (trending)"
    else
        "random walk"
    end

    return (H_rs=H_rs, H_dfa=H_dfa, H=H_combined,
            interpretation=interpretation, scales=scales)
end

"""R/S statistic over sub-series of length s."""
function _rs_statistic(x::Vector{Float64}, s::Int)::Float64
    n = length(x)
    rs_values = Float64[]
    for start in 1:s:(n-s+1)
        sub = x[start:min(start+s-1, n)]
        mu = mean(sub)
        dev = cumsum(sub .- mu)
        R = maximum(dev) - minimum(dev)
        S = std(sub)
        S > 0 && push!(rs_values, R/S)
    end
    return isempty(rs_values) ? 0.0 : mean(rs_values)
end

"""DFA fluctuation function for scale s."""
function _dfa_fluctuation(x::Vector{Float64}, s::Int)::Float64
    n = length(x)
    # Integrate the series
    y = cumsum(x .- mean(x))
    F2_vals = Float64[]
    for start in 1:s:(n-s+1)
        seg = y[start:min(start+s-1, n)]
        len = length(seg)
        len < 4 && continue
        # Detrend by linear fit
        t_vec = Float64.(1:len)
        slope, intercept = _linreg(t_vec, seg)
        trend = intercept .+ slope .* t_vec
        push!(F2_vals, mean((seg .- trend).^2))
    end
    return isempty(F2_vals) ? 0.0 : sqrt(mean(F2_vals))
end

"""Simple linear regression: returns (slope, intercept)."""
function _linreg(x::Vector{Float64}, y::Vector{Float64})::Tuple{Float64,Float64}
    n = length(x)
    x_m = mean(x); y_m = mean(y)
    slope = cov(x, y) / max(var(x), 1e-12)
    intercept = y_m - slope * x_m
    return (slope, intercept)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Intraday Periodicity Estimation and Removal
# ─────────────────────────────────────────────────────────────────────────────

"""
    IntradayPeriodicity(returns, periods_per_day; method) → NamedTuple

Estimate and remove intraday seasonality (U-shape in volatility) from
high-frequency crypto returns.

Uses the FFF (Flexible Fourier Form) or simple grid-average approach.

# Arguments
- `returns`          : intraday return vector
- `periods_per_day`  : number of observations per day (e.g. 288 for 5-min)
- `method`           : :average, :fff, or :wavg (default :average)

# Returns
NamedTuple: (seasonality, adjusted_returns, diurnal_factor, peak_hour)
"""
function IntradayPeriodicity(returns::Vector{Float64}, periods_per_day::Int;
                              method::Symbol=:average)
    n = length(returns)
    n_days = n ÷ periods_per_day

    # Map each observation to intraday slot
    slot = mod1.((1:n), periods_per_day)

    if method == :average
        # Simple average absolute return per slot
        slot_vol = zeros(periods_per_day)
        slot_cnt = zeros(Int, periods_per_day)
        for t in 1:n
            s = slot[t]
            slot_vol[s] += abs(returns[t])
            slot_cnt[s] += 1
        end
        slot_cnt = max.(slot_cnt, 1)
        seasonality = slot_vol ./ slot_cnt

    elseif method == :fff
        # Flexible Fourier Form: regress |r_t| on sin/cos harmonics
        K = 3  # number of Fourier terms
        X = ones(n, 2K+1)
        for k in 1:K
            X[:, 2k]   = sin.(2π * k * slot ./ periods_per_day)
            X[:, 2k+1] = cos.(2π * k * slot ./ periods_per_day)
        end
        y = abs.(returns)
        XtX = X'*X + 1e-8*I
        β = XtX \ (X' * y)
        fitted = X * β
        # Reconstruct seasonality on grid
        t_grid = Float64.(1:periods_per_day)
        X_grid = ones(periods_per_day, 2K+1)
        for k in 1:K
            X_grid[:, 2k]   = sin.(2π * k * t_grid ./ periods_per_day)
            X_grid[:, 2k+1] = cos.(2π * k * t_grid ./ periods_per_day)
        end
        seasonality = X_grid * β

    else  # wavg: volatility-weighted average
        slot_vol = zeros(periods_per_day)
        slot_cnt = zeros(Int, periods_per_day)
        weights  = returns.^2
        for t in 1:n
            s = slot[t]
            slot_vol[s] += weights[t]
            slot_cnt[s] += 1
        end
        slot_cnt = max.(slot_cnt, 1)
        seasonality = sqrt.(slot_vol ./ slot_cnt)
    end

    # Normalise so mean seasonality = 1
    mean_s = mean(seasonality)
    mean_s > 0 && (seasonality ./= mean_s)

    # Diurnal factor for each observation
    diurnal = seasonality[slot]

    # Adjusted returns (remove seasonality)
    adjusted = returns ./ max.(diurnal, 0.1)

    # Peak volatility period
    peak_slot = argmax(seasonality)
    peak_hour = peak_slot / periods_per_day * 24.0  # approximate hour

    return (seasonality=seasonality, adjusted_returns=adjusted,
            diurnal_factor=diurnal, peak_slot=peak_slot, peak_hour=peak_hour,
            n_days=n_days, periods_per_day=periods_per_day)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Quote Stuffing Detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    QuoteStuffingDetection(quote_counts, timestamps; window_ms, threshold_z) → NamedTuple

Detect quote stuffing: abnormal bursts of quote activity within short windows.
A hallmark of certain high-frequency manipulation strategies.

# Arguments
- `quote_counts`  : number of quotes in each time bucket
- `timestamps`    : time stamps (in ms) for each bucket
- `window_ms`     : detection window in milliseconds (default 1000)
- `threshold_z`   : z-score threshold for anomaly (default 5.0)

# Returns
NamedTuple: (stuffing_events, event_times, severity_scores, baseline_rate,
             stuffing_fraction, z_scores)
"""
function QuoteStuffingDetection(quote_counts::Vector{Float64},
                                 timestamps::Vector{Float64};
                                 window_ms::Float64=1000.0,
                                 threshold_z::Float64=5.0)
    n = length(quote_counts)
    length(timestamps) == n || error("timestamps must match quote_counts")

    # Rolling window statistics
    rolling_mean = _running_mean(quote_counts, 20)
    rolling_std  = zeros(n)
    for i in 21:n
        rolling_std[i] = std(quote_counts[max(1,i-20):i])
    end
    rolling_std[1:20] .= std(quote_counts[1:min(20,n)])

    # Z-scores
    z_scores = zeros(n)
    for i in 1:n
        rolling_std[i] > 0 &&
            (z_scores[i] = (quote_counts[i] - rolling_mean[i]) / rolling_std[i])
    end

    # Detect stuffing events
    event_mask = z_scores .> threshold_z
    event_times = timestamps[event_mask]
    event_counts = quote_counts[event_mask]

    # Cluster events within window_ms
    stuffing_events = Int[]
    in_cluster = false
    cluster_start = 0.0
    for i in 1:n
        if event_mask[i]
            if !in_cluster || (timestamps[i] - cluster_start) > window_ms
                push!(stuffing_events, i)
                cluster_start = timestamps[i]
                in_cluster = true
            end
        else
            in_cluster = false
        end
    end

    # Severity: normalised excess quote rate
    baseline_rate = mean(quote_counts)
    severity = [q / max(baseline_rate, 1.0) - 1.0 for q in event_counts]

    stuffing_fraction = sum(event_mask) / n

    return (stuffing_events=stuffing_events, event_times=event_times,
            severity_scores=severity, baseline_rate=baseline_rate,
            stuffing_fraction=stuffing_fraction, z_scores=z_scores,
            n_events=length(stuffing_events), threshold_z=threshold_z)
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. Top-Level Driver
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_microstructure_analysis(prices, bid, ask; kwargs...) → Dict

Run the full crypto microstructure analysis pipeline.

# Arguments
- `prices`  : trade price vector
- `bid`     : bid quote vector (same length as prices)
- `ask`     : ask quote vector
- Keyword arguments forwarded to individual functions.

# Returns
Dict with all microstructure metrics.

# Example
```julia
using Random
rng = Random.MersenneTwister(42)
n = 2000
prices = cumsum(randn(rng, n) .* 10) .+ 50_000.0
prices = max.(prices, 1.0)
bid = prices .- abs.(randn(rng, n) .* 5)
ask = prices .+ abs.(randn(rng, n) .* 5)
bid_vol = abs.(randn(rng, n)) .* 1e6 .+ 5e5
ask_vol = abs.(randn(rng, n)) .* 1e6 .+ 5e5
results = run_microstructure_analysis(prices, bid, ask, bid_vol, ask_vol)
println("Effective spread (bps): ", results["spread"]["effective_spread_bps"])
println("Hurst exponent: ", results["hurst"]["H"])
```
"""
function run_microstructure_analysis(prices::Vector{Float64},
                                      bid::Vector{Float64},
                                      ask::Vector{Float64},
                                      bid_vol::Vector{Float64}=Float64[],
                                      ask_vol::Vector{Float64}=Float64[];
                                      periods_per_day::Int=288)
    n = length(prices)
    isempty(bid_vol) && (bid_vol = fill(1e6, n))
    isempty(ask_vol) && (ask_vol = fill(1e6, n))
    results = Dict{String, Any}()

    # ── Order Flow Imbalance ───────────────────────────────────────────────
    @info "Order flow imbalance..."
    ofi = OrderFlowImbalance(bid_vol, ask_vol; window=20)
    results["ofi"] = Dict(
        "buy_sell_ratio" => ofi.buy_sell_ratio,
        "autocorr_1"     => ofi.autocorr_1,
        "hhi"            => ofi.hhi,
        "mean_ofi"       => mean(ofi.ofi)
    )

    # ── Lee-Ready Signs ────────────────────────────────────────────────────
    @info "Lee-Ready trade sign classification..."
    lr = LeeReadySign(prices, bid, ask)
    results["lee_ready"] = Dict(
        "buy_fraction"  => lr.buy_fraction,
        "sell_fraction" => lr.sell_fraction,
        "n_buy"         => lr.n_buy,
        "n_sell"        => lr.n_sell,
        "avg_eff_spread" => mean(abs.(lr.effective_spread))
    )

    # ── Spread Decomposition ───────────────────────────────────────────────
    @info "Spread decomposition..."
    sd = SpreadDecomposition(prices, lr.signs, lr.midpoints; τ=5)
    results["spread"] = Dict(
        "effective_spread"       => sd.effective_spread,
        "price_impact"           => sd.price_impact,
        "realised_spread"        => sd.realised_spread,
        "adverse_selection_pct"  => sd.adverse_selection_pct,
        "effective_spread_bps"   => sd.effective_spread / mean(prices) * 10000
    )

    # ── Roll's Model ───────────────────────────────────────────────────────
    @info "Roll's model..."
    roll = RollModel(prices)
    results["roll_model"] = Dict(
        "spread_estimate"     => roll.spread_estimate,
        "spread_bps"          => roll.effective_spread_bps,
        "price_impact_frac"   => roll.price_impact_frac,
        "autocovariance"      => roll.autocovariance
    )

    # ── Hasbrouck Information Share ────────────────────────────────────────
    @info "Hasbrouck information share (using bid/ask as pseudo-venues)..."
    prices_2v = hcat(log.(prices), log.((bid .+ ask) ./ 2.0))
    hs = HasbrouckInfoShare(prices_2v; max_lags=3)
    results["hasbrouck"] = Dict(
        "info_share_mid" => hs.info_share_mid,
        "lead_lag"       => hs.lead_lag_matrix[1,2]
    )

    # ── Variance Ratio Test ────────────────────────────────────────────────
    @info "Variance ratio test (RWH)..."
    vr = VarianceRatioTest(prices; q_values=[2,4,8,16], heteroskedastic=true)
    results["variance_ratio"] = Dict(
        "VR"          => vr.VR,
        "Z_stats"     => vr.Z_stats,
        "p_values"    => vr.p_values,
        "reject_RWH"  => vr.reject_RWH,
        "Hurst_approx"=> vr.Hurst_approx
    )

    # ── Hurst Exponent ─────────────────────────────────────────────────────
    @info "Hurst exponent (R/S + DFA)..."
    log_prices = log.(prices)
    hurst = HurstExponent(log_prices; method=:both, min_n=10)
    results["hurst"] = Dict(
        "H"              => hurst.H,
        "H_rs"           => hurst.H_rs,
        "H_dfa"          => hurst.H_dfa,
        "interpretation" => hurst.interpretation
    )

    # ── Intraday Periodicity ───────────────────────────────────────────────
    @info "Intraday periodicity..."
    ret_series = diff(log.(prices))
    if length(ret_series) >= periods_per_day * 2
        idp = IntradayPeriodicity(ret_series, periods_per_day; method=:average)
        results["intraday_periodicity"] = Dict(
            "peak_slot"   => idp.peak_slot,
            "peak_hour"   => idp.peak_hour,
            "n_days"      => idp.n_days,
            "seasonality_range" => maximum(idp.seasonality) - minimum(idp.seasonality)
        )
    else
        results["intraday_periodicity"] = Dict("note" => "insufficient data for periodicity")
    end

    # ── Quote Stuffing ─────────────────────────────────────────────────────
    @info "Quote stuffing detection..."
    # Use bid/ask spread changes as proxy for quote activity
    spread_changes = abs.(diff(ask .- bid))
    timestamps_ms  = Float64.(1:length(spread_changes)) .* 1000.0
    qs = QuoteStuffingDetection(spread_changes, timestamps_ms;
                                 threshold_z=4.0)
    results["quote_stuffing"] = Dict(
        "n_events"           => qs.n_events,
        "stuffing_fraction"  => qs.stuffing_fraction,
        "baseline_rate"      => qs.baseline_rate
    )

    # ── Summary ────────────────────────────────────────────────────────────
    results["summary"] = Dict(
        "n_trades"              => n,
        "buy_fraction"          => lr.buy_fraction,
        "effective_spread_bps"  => sd.effective_spread / mean(prices) * 10000,
        "adverse_selection_pct" => sd.adverse_selection_pct,
        "Hurst"                 => hurst.H,
        "RWH_rejected"          => vr.reject_RWH,
        "price_impact_frac"     => roll.price_impact_frac,
        "quote_stuffing_events" => qs.n_events
    )

    return results
end

end  # module CryptoMicrostructure
