"""
HighFrequencyTools — HFT Analytics Library

Comprehensive toolkit for high-frequency trading research:
  - Realised variance estimators: RV, BPV, RK, TSRV, pre-averaging
  - Diurnal pattern estimation and removal
  - Jump detection and filtering
  - Trade classification: tick rule, Lee-Ready, bulk volume
  - Order book analytics: depth, imbalance, pressure
  - Adverse selection: PIN model estimation
  - Market impact: Amihud, Kyle lambda, square-root
  - Flash crash detection
"""
module HighFrequencyTools

using Statistics
using LinearAlgebra
using Random

export realized_variance, bipower_variation, two_scales_rv, pre_averaging_rv
export realized_kernel_parzen, multipower_variation
export diurnal_adjustment, fff_estimate, seasonal_factor
export jump_filter, bns_jump_test, lee_mykland_jump_test
export tick_rule, lee_ready_rule, bulk_volume_classification
export order_book_imbalance, book_pressure, effective_spread
export pin_model_estimate, easley_ohara_pin
export amihud_lambda, kyle_lambda_estimate, sqrt_market_impact
export flash_crash_detect, abnormal_vol_alert
export nadarray_signature_plot, roll_covariance

# =============================================================================
# SECTION 1: REALISED VARIANCE ESTIMATORS
# =============================================================================

"""
    realized_variance(log_prices; skip=1) -> Float64

Standard RV: sum of squared log-returns.
"""
function realized_variance(log_prices::Vector{Float64}; skip::Int=1)::Float64
    n = length(log_prices)
    n < 2 && return 0.0
    rv = 0.0
    i = 1
    while i + skip <= n
        rv += (log_prices[i+skip] - log_prices[i])^2
        i += skip
    end
    return rv
end

"""
    bipower_variation(log_prices) -> Float64

Barndorff-Nielsen & Shephard (2004) BPV = (π/2) Σ|r_t||r_{t-1}|.
Robust to finite-activity jumps.
"""
function bipower_variation(log_prices::Vector{Float64})::Float64
    returns = abs.(diff(log_prices))
    m = length(returns)
    m < 2 && return 0.0
    return (π/2) * sum(returns[2:end] .* returns[1:end-1])
end

"""
    multipower_variation(log_prices, powers) -> Float64

Generalised multipower variation: sum product of |r_t|^p_k over consecutive windows.
BPV is the special case powers = [1, 1] (scaled).
"""
function multipower_variation(log_prices::Vector{Float64},
                                powers::Vector{Float64})::Float64
    returns = abs.(diff(log_prices))
    m = length(returns)
    q = length(powers)
    m < q && return 0.0

    # Scaling constant: product of E[|Z|^p] for Z ~ N(0,1)
    function normal_abs_moment(p)
        p == 1 && return sqrt(2/π)
        p == 2 && return 1.0
        # Gamma function formula: E[|Z|^p] = 2^(p/2)*Γ((p+1)/2)/Γ(1/2)
        # Use simple recursion
        return 2^(p/2) * _gamma_half(p+1) / sqrt(π)
    end

    scale = prod(normal_abs_moment(p) for p in powers)
    scale = max(scale, 1e-20)

    total = 0.0
    for t in q:m
        prod_val = 1.0
        for (k, p) in enumerate(powers)
            prod_val *= returns[t - k + 1]^p
        end
        total += prod_val
    end

    return total / scale
end

function _gamma_half(n::Float64)::Float64
    # Gamma((n)/2) for small n via recursion
    x = n / 2.0
    if abs(x - round(x)) < 1e-6
        # Integer: use factorial
        k = round(Int, x)
        k <= 1 && return 1.0
        return prod(1:k-1) * 1.0
    else
        # Half-integer
        return sqrt(π)
    end
end

"""
    two_scales_rv(log_prices; K=nothing) -> NamedTuple

Zhang-Mykland-Aït-Sahalia (2005) TSRV noise-robust estimator.
"""
function two_scales_rv(log_prices::Vector{Float64};
                        K::Union{Int,Nothing}=nothing)

    n = length(log_prices)
    n < 10 && return (tsrv=0.0, rv_fast=0.0, rv_slow=0.0, noise_var=0.0, K=1)

    K_use = K === nothing ? max(1, round(Int, cbrt(n))) : K

    rv_fast = realized_variance(log_prices; skip=1)

    # Average over K subgrids
    rv_sub = zeros(K_use)
    for k in 1:K_use
        idx = collect(k:K_use:n)
        length(idx) < 2 && continue
        rv_sub[k] = realized_variance(log_prices[idx]; skip=1)
    end
    rv_slow = mean(rv_sub)

    n_slow = (n - K_use) / K_use
    n_fast = n - 1.0
    correction = n_slow / n_fast
    tsrv = max(rv_slow - correction * rv_fast, 0.0)
    noise_var = rv_fast / (2*n_fast)

    return (tsrv=tsrv, rv_fast=rv_fast, rv_slow=rv_slow, noise_var=noise_var, K=K_use)
end

"""
    pre_averaging_rv(log_prices; theta=0.5) -> NamedTuple

Jacod et al. (2009) pre-averaging estimator.
"""
function pre_averaging_rv(log_prices::Vector{Float64}; theta::Float64=0.5)
    n = length(log_prices)
    n < 10 && return (pav=0.0, kn=2, noise_var=0.0)

    kn = max(2, floor(Int, theta*sqrt(n)))
    g(x) = min(x, 1-x)  # triangular weight

    gw = [g(j/kn) for j in 1:(kn-1)]
    psi1 = sum(gw); psi2 = sum(gw.^2)

    dP = diff(log_prices)
    m = length(dP)
    num_avg = m - kn + 1
    num_avg < 1 && return (pav=0.0, kn=kn, noise_var=0.0)

    Y_avg = zeros(num_avg)
    for i in 1:num_avg
        for (jdx,j) in enumerate(1:(kn-1))
            i+j-1 <= m && (Y_avg[i] += gw[jdx]*dP[i+j-1])
        end
    end

    noise_var = sum(dP.^2)/(2m)
    pav_raw = sum(Y_avg.^2)/(kn*psi2/kn)
    pav = max(pav_raw - psi1^2/psi2*noise_var, 0.0)

    return (pav=pav, kn=kn, noise_var=noise_var)
end

"""
    realized_kernel_parzen(log_prices; H=nothing) -> NamedTuple

Barndorff-Nielsen et al. (2008) Parzen-kernel RV.
"""
function realized_kernel_parzen(log_prices::Vector{Float64};
                                  H::Union{Int,Nothing}=nothing)

    n = length(log_prices); n < 4 && return (rk=0.0, H=0)
    returns = diff(log_prices)
    m = length(returns)
    H_use = H === nothing ? max(1, round(Int, cbrt(m))) : H
    H_use = min(H_use, m-1)

    function parzen(x)
        ax = abs(x); ax > 1 && return 0.0
        ax <= 0.5 && return 1-6ax^2+6ax^3
        return 2(1-ax)^3
    end

    # Autocovariances
    acov = zeros(H_use+1)
    for h in 0:H_use
        for i in (h+1):m
            acov[h+1] += returns[i]*returns[i-h]
        end
    end

    rk = acov[1]
    for h in 1:H_use
        rk += 2*parzen(h/H_use)*acov[h+1]
    end

    return (rk=max(rk,0.0), H=H_use)
end

# =============================================================================
# SECTION 2: DIURNAL ADJUSTMENT
# =============================================================================

"""
    fff_estimate(time_of_day, returns; K=5) -> NamedTuple

Fourier Flexible Form seasonal estimator (Andersen-Bollerslev 1997).
"""
function fff_estimate(time_of_day::Vector{Float64},
                       returns::Vector{Float64}; K::Int=5)

    n = length(returns)
    y = log.(returns.^2 .+ 1e-10)
    n_params = 1 + 2K

    X = ones(n, n_params)
    for k in 1:K
        X[:, 2k]   = sin.(2π*k.*time_of_day)
        X[:, 2k+1] = cos.(2π*k.*time_of_day)
    end

    beta = try (X'*X + 1e-8I)\(X'*y) catch zeros(n_params) end

    function predict(tau::Vector{Float64})
        Xp = ones(length(tau), n_params)
        for k in 1:K
            Xp[:, 2k]   = sin.(2π*k.*tau)
            Xp[:, 2k+1] = cos.(2π*k.*tau)
        end
        exp.(Xp*beta)
    end

    yhat = X*beta
    r2 = 1 - sum((y-yhat).^2)/sum((y.-mean(y)).^2)

    return (beta=beta, predict=predict, r_squared=r2)
end

"""
    seasonal_factor(time_of_day, fff_fit) -> Vector{Float64}

Return multiplicative seasonal factors for given times.
"""
function seasonal_factor(time_of_day::Vector{Float64}, fff_fit)::Vector{Float64}
    s = fff_fit.predict(time_of_day)
    return s ./ mean(s)
end

"""
    diurnal_adjustment(returns, time_of_day; K=5) -> NamedTuple

Remove intraday seasonality.
"""
function diurnal_adjustment(returns::Vector{Float64},
                              time_of_day::Vector{Float64}; K::Int=5)

    n = length(returns)
    n < 20 && return (adjusted=returns, seasonal=ones(n))

    fit = fff_estimate(time_of_day, returns; K=K)
    s = seasonal_factor(time_of_day, fit)
    adjusted = returns ./ sqrt.(max.(s, 1e-10))

    return (adjusted=adjusted, seasonal=s, fit=fit)
end

# =============================================================================
# SECTION 3: JUMP DETECTION
# =============================================================================

"""
    bns_jump_test(log_prices; significance=0.05) -> NamedTuple

Barndorff-Nielsen & Shephard (2006) jump test.
"""
function bns_jump_test(log_prices::Vector{Float64}; significance::Float64=0.05)

    n = length(log_prices); n < 5 && return (stat=0.0, p=1.0, has_jump=false, rv=0.0, bpv=0.0)

    rv  = realized_variance(log_prices)
    bpv = bipower_variation(log_prices)

    returns = abs.(diff(log_prices))
    m = length(returns)
    mu1 = sqrt(2/π)

    tpq = 0.0
    for i in 3:m
        tpq += returns[i]^(4/3)*returns[i-1]^(4/3)*returns[i-2]^(4/3)
    end
    tpq *= m/(m-2) / mu1^3

    omega_sq = π^2/4 + π - 5
    test_stat = (rv-bpv)/rv
    var_stat = omega_sq * max(tpq,1e-20)/bpv^2
    Z = var_stat > 0 && m > 0 ? test_stat/sqrt(var_stat/m) : 0.0

    p_val = 1 - _normal_cdf(Z)

    return (stat=Z, p=p_val, has_jump=(p_val<significance), rv=rv, bpv=bpv,
             jump_component=max(rv-bpv,0.0))
end

"""
    lee_mykland_jump_test(log_prices, times; significance=0.05) -> NamedTuple

Lee & Mykland (2008): identify individual jump times.
"""
function lee_mykland_jump_test(log_prices::Vector{Float64},
                                 times::Vector{Float64};
                                 significance::Float64=0.05)

    n = length(log_prices); n < 5 && return (jump_times=Float64[], jump_sizes=Float64[], L=Float64[], threshold=0.0)

    returns = diff(log_prices)
    m = length(returns)
    K = max(3, ceil(Int, sqrt(m)))

    local_vol = zeros(m)
    for i in 1:m
        rng = returns[max(1,i-K):min(m,i+K)]
        if length(rng) > 2
            bpv_w = (π/2)*sum(abs.(rng[2:end]).*abs.(rng[1:end-1]))
            local_vol[i] = sqrt(bpv_w/max(length(rng)-1, 1))
        else
            local_vol[i] = std(returns)
        end
    end

    L = local_vol .> 0 ? abs.(returns) ./ local_vol : zeros(m)

    beta_m = (2*log(m))^0.5 - (log(π)+log(log(m)))/(2*(2*log(m))^0.5)
    c_m = 1/(2*log(m))^0.5
    threshold = beta_m - c_m*log(-log(1-significance))

    jump_idx = findall(L .> threshold)
    jump_times = isempty(jump_idx) ? Float64[] : times[jump_idx .+ 1]
    jump_sizes = isempty(jump_idx) ? Float64[] : returns[jump_idx]

    return (jump_times=jump_times, jump_sizes=jump_sizes, L=L, threshold=threshold)
end

"""
    jump_filter(log_prices; n_sigma=3.0) -> NamedTuple

Filter jumps from a price series using a simple threshold:
remove returns exceeding n_sigma standard deviations.
"""
function jump_filter(log_prices::Vector{Float64}; n_sigma::Float64=3.0)

    n = length(log_prices)
    returns = diff(log_prices)
    m = length(returns)

    mu_r = median(returns)
    # Robust std: MAD
    mad = median(abs.(returns .- mu_r))
    sigma_r = mad / 0.6745  # MAD to sigma conversion

    threshold = n_sigma * sigma_r
    jump_flags = abs.(returns .- mu_r) .> threshold

    # Reconstruct filtered prices
    filtered = copy(log_prices)
    for i in findall(jump_flags)
        # Replace jump return with median
        filtered[i+1] = filtered[i] + mu_r
    end

    return (filtered_prices=filtered, jump_flags=jump_flags,
             n_jumps=sum(jump_flags), threshold=threshold)
end

# =============================================================================
# SECTION 4: TRADE CLASSIFICATION
# =============================================================================

"""
    tick_rule(prices) -> Vector{Int}

Lee-Ready tick rule: classify buy (+1) / sell (-1) based on price movement.
"""
function tick_rule(prices::Vector{Float64})::Vector{Int}
    n = length(prices)
    signs = zeros(Int, n)
    last = 1
    for i in 2:n
        dp = prices[i] - prices[i-1]
        if dp > 0;     signs[i] = 1;  last = 1
        elseif dp < 0; signs[i] = -1; last = -1
        else;          signs[i] = last
        end
    end
    signs[1] = signs[2] != 0 ? signs[2] : 1
    return signs
end

"""
    lee_ready_rule(prices, quotes_mid; delay=0) -> Vector{Int}

Lee-Ready (1991) trade classification using quote midpoint:
- If trade at ask or above → buy (+1)
- If trade at bid or below → sell (-1)
- Otherwise → apply tick rule
"""
function lee_ready_rule(prices::Vector{Float64},
                          quotes_mid::Vector{Float64};
                          delay::Int=0)::Vector{Int}
    n = length(prices)
    @assert length(quotes_mid) == n
    signs = zeros(Int, n)

    tick = tick_rule(prices)

    for i in 1:n
        mid_idx = max(1, i - delay)
        mid = quotes_mid[mid_idx]
        if prices[i] > mid
            signs[i] = 1   # above mid → buy
        elseif prices[i] < mid
            signs[i] = -1  # below mid → sell
        else
            signs[i] = tick[i]  # at mid → tick rule
        end
    end

    return signs
end

"""
    bulk_volume_classification(prices, volumes; n_buckets=50) -> Vector{Float64}

Bulk Volume Classification (Easley et al. 2012).

Within each bucket: classify fraction of volume as buy vs sell
using price movement as signal.

Returns vector of signed volumes (positive = net buy volume).
"""
function bulk_volume_classification(prices::Vector{Float64},
                                      volumes::Vector{Float64};
                                      n_buckets::Int=50)::Vector{Float64}

    n = length(prices)
    @assert length(volumes) == n

    bucket_size = max(1, n ÷ n_buckets)
    signed_vols = zeros(n_buckets)

    for b in 1:n_buckets
        start_i = (b-1)*bucket_size + 1
        end_i   = min(b*bucket_size, n)
        start_i > n && break

        p_bucket = prices[start_i:end_i]
        v_bucket = volumes[start_i:end_i]
        total_vol = sum(v_bucket)

        if total_vol > 0 && length(p_bucket) > 1
            dp = p_bucket[end] - p_bucket[1]
            sigma_p = std(p_bucket)
            z = sigma_p > 0 ? dp / sigma_p : 0.0
            # Buy fraction via normal CDF
            buy_frac = _normal_cdf(z)
            signed_vols[b] = total_vol * (2*buy_frac - 1)
        end
    end

    return signed_vols
end

# =============================================================================
# SECTION 5: ORDER BOOK ANALYTICS
# =============================================================================

"""
    order_book_imbalance(bid_vols, ask_vols) -> Vector{Float64}

Queue imbalance: OI = (V_bid - V_ask) / (V_bid + V_ask) ∈ [-1, 1].
"""
function order_book_imbalance(bid_vols::Vector{Float64},
                                ask_vols::Vector{Float64})::Vector{Float64}
    n = length(bid_vols)
    @assert length(ask_vols) == n
    oi = zeros(n)
    for i in 1:n
        total = bid_vols[i] + ask_vols[i]
        total > 0 && (oi[i] = (bid_vols[i]-ask_vols[i])/total)
    end
    return oi
end

"""
    book_pressure(bid_prices, bid_sizes, ask_prices, ask_sizes; levels=5) -> NamedTuple

Compute order book pressure metrics using the top `levels` levels.

Bid pressure = Σ V_bid_i / (Ask_i - Mid)  (volume weighted by inverse distance)
Ask pressure = Σ V_ask_i / (Mid - Bid_i)

Higher bid pressure relative to ask → price likely to rise.
"""
function book_pressure(bid_prices::Matrix{Float64},
                         bid_sizes::Matrix{Float64},
                         ask_prices::Matrix{Float64},
                         ask_sizes::Matrix{Float64};
                         levels::Int=5)

    n_obs, n_lev = size(bid_prices)
    levels = min(levels, n_lev)

    bid_press = zeros(n_obs)
    ask_press = zeros(n_obs)
    imbalance  = zeros(n_obs)

    for t in 1:n_obs
        mid = (bid_prices[t,1] + ask_prices[t,1]) / 2

        bp = 0.0; ap = 0.0
        for l in 1:levels
            dist_bid = abs(mid - bid_prices[t,l])
            dist_ask = abs(mid - ask_prices[t,l])
            bp += dist_bid > 0 ? bid_sizes[t,l]/dist_bid : bid_sizes[t,l]*1000
            ap += dist_ask > 0 ? ask_sizes[t,l]/dist_ask : ask_sizes[t,l]*1000
        end

        bid_press[t] = bp; ask_press[t] = ap
        total_press = bp + ap
        imbalance[t] = total_press > 0 ? (bp-ap)/total_press : 0.0
    end

    return (bid_pressure=bid_press, ask_pressure=ask_press, imbalance=imbalance)
end

"""
    effective_spread(trade_prices, quotes_mid) -> Vector{Float64}

Effective spread = 2 * |price - midquote| / midquote.
"""
function effective_spread(trade_prices::Vector{Float64},
                            quotes_mid::Vector{Float64})::Vector{Float64}
    n = length(trade_prices)
    @assert length(quotes_mid) == n
    es = zeros(n)
    for i in 1:n
        quotes_mid[i] > 0 && (es[i] = 2*abs(trade_prices[i]-quotes_mid[i])/quotes_mid[i])
    end
    return es
end

# =============================================================================
# SECTION 6: ADVERSE SELECTION — PIN MODEL
# =============================================================================

"""
    pin_model_estimate(buy_trades, sell_trades; max_iter=500) -> NamedTuple

Easley-Kiefer-O'Hara-Paperman (1996) PIN model estimation via MLE.

Model:
  Probability of information event: α
  Probability of bad news | event: δ
  Arrival rate of informed traders: μ
  Arrival rate of uninformed buyers: ε_b
  Arrival rate of uninformed sellers: ε_s

  PIN = αμ / (αμ + ε_b + ε_s)

Likelihood:
  L = (1-α) * Poisson(B|ε_b) * Poisson(S|ε_s)
     + α*δ   * Poisson(B|ε_b) * Poisson(S|μ+ε_s)
     + α*(1-δ)* Poisson(B|μ+ε_b) * Poisson(S|ε_s)

# Arguments
- `buy_trades`: vector of daily buy-trade counts
- `sell_trades`: vector of daily sell-trade counts
"""
function pin_model_estimate(buy_trades::Vector{Float64},
                              sell_trades::Vector{Float64};
                              max_iter::Int=500)

    n = length(buy_trades)
    @assert length(sell_trades) == n

    mu_B = mean(buy_trades); mu_S = mean(sell_trades)

    function poisson_loglik(k::Float64, lambda::Float64)::Float64
        lambda <= 0 && return -Inf
        k * log(lambda) - lambda - _log_factorial(k)
    end

    function log_likelihood(params::Vector{Float64})::Float64
        alpha, delta, mu, eps_b, eps_s = params
        (alpha<0||alpha>1||delta<0||delta>1||mu<0||eps_b<0||eps_s<0) && return -Inf

        ll = 0.0
        for i in 1:n
            B = buy_trades[i]; S = sell_trades[i]
            l1 = log(max(1-alpha, 1e-10)) + poisson_loglik(B, eps_b) + poisson_loglik(S, eps_s)
            l2 = log(max(alpha*delta, 1e-10)) + poisson_loglik(B, eps_b) + poisson_loglik(S, mu+eps_s)
            l3 = log(max(alpha*(1-delta), 1e-10)) + poisson_loglik(B, mu+eps_b) + poisson_loglik(S, eps_s)

            # Log-sum-exp
            log_max = max(l1, l2, l3)
            ll += log_max + log(exp(l1-log_max) + exp(l2-log_max) + exp(l3-log_max))
        end
        return ll
    end

    # Initial guess
    best = [0.3, 0.5, max(mu_B, mu_S)*0.5, mu_B*0.7, mu_S*0.7]
    best_ll = log_likelihood(best)

    steps = [0.05, 0.05, max(mu_B,1.0)*0.1, mu_B*0.05, mu_S*0.05]

    for _ in 1:max_iter
        improved = false
        for dim in 1:5
            for dir in [1.0,-1.0]
                c = copy(best)
                c[dim] += dir*steps[dim]
                ll = log_likelihood(c)
                if ll > best_ll
                    best_ll = ll; best = c; improved = true
                end
            end
        end
        if !improved
            steps .*= 0.5
            all(steps .< 1e-9) && break
        end
    end

    alpha, delta, mu, eps_b, eps_s = best
    pin = alpha*mu / max(alpha*mu + eps_b + eps_s, 1e-10)

    return (alpha=alpha, delta=delta, mu=mu, eps_b=eps_b, eps_s=eps_s,
             pin=pin, loglik=best_ll)
end

"""
    easley_ohara_pin(buy_trades, sell_trades) -> Float64

Simplified PIN estimate using the Duarte-Young formula.
PIN ≈ (|B̄ - S̄|) / (B̄ + S̄) where B̄, S̄ are mean daily volumes.
"""
function easley_ohara_pin(buy_trades::Vector{Float64},
                            sell_trades::Vector{Float64})::Float64
    B = mean(buy_trades); S = mean(sell_trades)
    total = B + S
    return total > 0 ? abs(B-S)/total : 0.0
end

function _log_factorial(n::Float64)::Float64
    n <= 0 && return 0.0
    n > 170 && return n*log(n) - n + 0.5*log(2π*n)  # Stirling
    return sum(log(i) for i in 1:round(Int,n))
end

# =============================================================================
# SECTION 7: MARKET IMPACT
# =============================================================================

"""
    amihud_lambda(returns, volumes; window=21) -> Float64

Amihud (2002) illiquidity: average |r_t| / Volume_t.
Higher λ → illiquid, prices move more per dollar traded.
"""
function amihud_lambda(returns::Vector{Float64},
                         volumes::Vector{Float64};
                         window::Union{Int,Nothing}=nothing)::Float64

    n = length(returns)
    @assert length(volumes) == n

    w = window === nothing ? n : min(window, n)
    start = n - w + 1

    total = 0.0; count = 0
    for i in start:n
        volumes[i] > 0 && (total += abs(returns[i])/volumes[i]; count += 1)
    end

    return count > 0 ? total/count : 0.0
end

"""
    kyle_lambda_estimate(price_changes, signed_volumes) -> NamedTuple

Kyle (1985) lambda: slope of ΔP on signed order flow Q.
ΔP_t = λ Q_t + ε_t. Higher λ → more price impact per unit volume.
"""
function kyle_lambda_estimate(price_changes::Vector{Float64},
                                signed_volumes::Vector{Float64})

    n = length(price_changes)
    @assert length(signed_volumes) == n
    n < 5 && return (lambda=0.0, alpha=0.0, r2=0.0, t_stat=0.0)

    x = signed_volumes; y = price_changes
    xm = mean(x); ym = mean(y)
    sxx = sum((x.-xm).^2); sxy = sum((x.-xm).*(y.-ym))
    lam = sxx > 0 ? sxy/sxx : 0.0
    alpha = ym - lam*xm

    yhat = alpha .+ lam.*x
    ssr = sum((y.-yhat).^2); sst = sum((y.-ym).^2)
    r2 = sst > 0 ? 1-ssr/sst : 0.0
    se = (n>2&&sxx>0) ? sqrt(ssr/(n-2)/sxx) : 1e-10
    t = se > 0 ? lam/se : 0.0

    return (lambda=lam, alpha=alpha, r2=r2, t_stat=t)
end

"""
    sqrt_market_impact(order_size, sigma, ADV; eta=0.1, gamma=0.6) -> Float64

Square-root market impact (Almgren et al. 2005):
    MI = eta * sigma * sqrt(order_size / ADV) * (order_size / ADV)^gamma

where ADV = average daily volume, sigma = daily volatility.
"""
function sqrt_market_impact(order_size::Float64,
                              sigma::Float64,
                              ADV::Float64;
                              eta::Float64=0.1,
                              gamma::Float64=0.6)::Float64

    ADV <= 0 && return 0.0
    frac = order_size / ADV
    return eta * sigma * frac^(0.5 + gamma)
end

# =============================================================================
# SECTION 8: FLASH CRASH DETECTION
# =============================================================================

"""
    flash_crash_detect(prices, times; vol_window=20, price_drop_pct=0.02,
                        time_window_sec=300) -> NamedTuple

Detect flash crash events: rapid price drops followed by partial recovery.

Criteria:
1. Price drops > `price_drop_pct` within `time_window_sec` seconds
2. Drop velocity > n_sigma standard deviations
3. Followed by partial recovery within same window

# Returns
- NamedTuple: event_indices, event_times, magnitudes, recovery_ratios
"""
function flash_crash_detect(prices::Vector{Float64},
                              times::Vector{Float64};
                              vol_window::Int=20,
                              price_drop_pct::Float64=0.02,
                              time_window_sec::Float64=300.0,
                              n_sigma::Float64=5.0)

    n = length(prices)
    @assert length(times) == n

    log_returns = diff(log.(max.(prices, 1e-10)))
    m = length(log_returns)

    # Rolling volatility
    sigma_rolling = zeros(m)
    for i in 1:m
        w = log_returns[max(1,i-vol_window+1):i]
        sigma_rolling[i] = length(w) > 1 ? std(w) : 1e-4
    end

    event_indices = Int[]
    event_times   = Float64[]
    magnitudes    = Float64[]
    recovery_ratios = Float64[]

    i = 1
    while i <= m
        # Check for extreme negative return
        if log_returns[i] < -n_sigma * sigma_rolling[i] &&
           abs(log_returns[i]) > log(1 + price_drop_pct)

            # Found potential flash crash start
            crash_start = i
            crash_price = prices[i]
            min_price = crash_price
            min_idx = i

            # Look ahead within time window
            j = i + 1
            while j <= n && times[j] - times[i] <= time_window_sec
                min_price = min(min_price, prices[j])
                min_idx = j
                j += 1
            end

            magnitude = (crash_price - min_price) / crash_price

            if magnitude > price_drop_pct
                # Check for recovery
                recovery_end = min(n, j)
                recovery_price = prices[recovery_end]
                recovery_ratio = (recovery_price - min_price) / (crash_price - min_price)

                push!(event_indices, crash_start)
                push!(event_times, times[crash_start])
                push!(magnitudes, magnitude)
                push!(recovery_ratios, clamp(recovery_ratio, 0.0, 2.0))

                i = j  # skip ahead
            end
        end
        i += 1
    end

    return (event_indices=event_indices, event_times=event_times,
             magnitudes=magnitudes, recovery_ratios=recovery_ratios,
             n_events=length(event_indices))
end

"""
    abnormal_vol_alert(returns, window=20; z_threshold=3.0) -> NamedTuple

Detect abnormal volatility spikes.
"""
function abnormal_vol_alert(returns::Vector{Float64},
                              window::Int=20;
                              z_threshold::Float64=3.0)

    n = length(returns)
    n < window + 2 && return (alert_indices=Int[], z_scores=Float64[])

    # Rolling vol
    roll_vol = [std(returns[max(1,i-window+1):i]) for i in 1:n]

    # Z-score of current vol vs history
    vol_mean = mean(roll_vol)
    vol_std  = std(roll_vol)

    z_scores = vol_std > 0 ? (roll_vol .- vol_mean) ./ vol_std : zeros(n)
    alerts = findall(z_scores .> z_threshold)

    return (alert_indices=alerts, z_scores=z_scores, threshold=z_threshold)
end

# =============================================================================
# SECTION 9: SIGNATURE PLOT AND ROLL COVARIANCE
# =============================================================================

"""
    nadarray_signature_plot(log_prices; max_lag=120) -> NamedTuple

Signature plot: realized variance as function of sampling frequency.
Optimal frequency = flattest region of the plot.
"""
function nadarray_signature_plot(log_prices::Vector{Float64};
                                   max_lag::Int=120)

    n = length(log_prices)
    max_lag = min(max_lag, n÷4)

    lags = 1:max_lag
    rv_vals = [realized_variance(log_prices; skip=l) *
               ((n-1)÷l) / (n-1) for l in lags]  # annualize per-return

    # Find plateau
    window = max(3, max_lag÷8)
    best_cv = Inf; opt_lag = lags[end÷2]

    for i in window:(max_lag-window)
        seg = rv_vals[max(1,i-window):min(max_lag,i+window)]
        mu = mean(seg); s = std(seg)
        cv = mu > 0 ? s/mu : Inf
        if cv < best_cv; best_cv = cv; opt_lag = lags[i] end
    end

    # Noise variance from Roll
    rets = diff(log_prices)
    acov1 = length(rets) > 2 ? sum(rets[2:end].*rets[1:end-1])/(length(rets)-1) : 0.0
    noise_var = max(-acov1, 0.0)

    return (lags=collect(lags), rv_values=rv_vals, optimal_lag=opt_lag,
             noise_variance=noise_var)
end

"""
    roll_covariance(log_prices1, log_prices2) -> NamedTuple

Roll (1984) implied covariance between two assets.
Cov_Roll = -Cov(Δr1_t, Δr1_{t-1}) for volatility,
and cross-covariance for co-movement.
"""
function roll_covariance(log_prices1::Vector{Float64},
                           log_prices2::Vector{Float64})

    n = min(length(log_prices1), length(log_prices2))
    n < 3 && return (spread1=0.0, spread2=0.0, roll_cov=0.0)

    r1 = diff(log_prices1[1:n])
    r2 = diff(log_prices2[1:n])
    m = length(r1)

    # Roll spreads
    acov1 = sum(r1[2:end].*r1[1:end-1])/(m-1)
    acov2 = sum(r2[2:end].*r2[1:end-1])/(m-1)
    spread1 = 2*sqrt(max(-acov1, 0.0))
    spread2 = 2*sqrt(max(-acov2, 0.0))

    # Cross-roll covariance
    cross_cov = sum(r1[2:end].*r2[1:end-1] + r2[2:end].*r1[1:end-1]) / (2*(m-1))

    return (spread1=spread1, spread2=spread2, roll_cov=cross_cov)
end

# =============================================================================
# HELPERS
# =============================================================================

function _normal_cdf(x::Float64)::Float64
    x >= 8 && return 1.0; x <= -8 && return 0.0
    t = 1/(1+0.2316419*abs(x))
    poly = t*(0.319381530+t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))))
    cdf = 1 - exp(-0.5x^2)/sqrt(2π)*poly
    return x >= 0 ? cdf : 1-cdf
end

end # module HighFrequencyTools
