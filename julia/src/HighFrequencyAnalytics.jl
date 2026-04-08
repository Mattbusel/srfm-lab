module HighFrequencyAnalytics

using LinearAlgebra
using Statistics
using Random

# ============================================================================
# SECTION 1: Tick Data Processing
# ============================================================================

"""
    TickData

Single tick record: timestamp (seconds from midnight), price, volume, type.
"""
struct TickData
    timestamp::Float64
    price::Float64
    volume::Float64
    tick_type::Int  # 1=trade, 2=bid, 3=ask
end

"""
    QuoteData

Bid-ask quote record.
"""
struct QuoteData
    timestamp::Float64
    bid::Float64
    ask::Float64
    bid_size::Float64
    ask_size::Float64
end

"""
    TradeData

Trade record with direction classification.
"""
struct TradeData
    timestamp::Float64
    price::Float64
    volume::Float64
    direction::Int  # +1=buy, -1=sell, 0=unknown
end

"""
    align_trades_quotes(trades, quotes)

Align trades with prevailing quotes using backward-looking join.
Returns vector of (trade, prevailing_quote) pairs.
"""
function align_trades_quotes(trades::Vector{TradeData},
                              quotes::Vector{QuoteData})
    n_trades = length(trades)
    aligned = Vector{Tuple{TradeData, QuoteData}}()

    qi = 1
    for ti in 1:n_trades
        # Find latest quote before trade
        while qi < length(quotes) && quotes[qi + 1].timestamp <= trades[ti].timestamp
            qi += 1
        end

        if qi <= length(quotes) && quotes[qi].timestamp <= trades[ti].timestamp
            push!(aligned, (trades[ti], quotes[qi]))
        end
    end

    return aligned
end

"""
    lee_ready_classification(trade_price, bid, ask, prev_trade_price)

Lee-Ready (1991) trade classification algorithm.
Step 1: Quote test - compare to midpoint.
Step 2: Tick test - compare to previous trade.
Returns: +1 (buy), -1 (sell).
"""
function lee_ready_classification(trade_price::Float64, bid::Float64,
                                   ask::Float64, prev_trade_price::Float64)::Int
    midpoint = 0.5 * (bid + ask)

    # Quote test
    if trade_price > midpoint + 1e-10
        return 1   # Buy
    elseif trade_price < midpoint - 1e-10
        return -1  # Sell
    end

    # Tick test (at midpoint)
    if trade_price > prev_trade_price + 1e-10
        return 1   # Uptick -> buy
    elseif trade_price < prev_trade_price - 1e-10
        return -1  # Downtick -> sell
    end

    return 0  # Indeterminate
end

"""
    tick_rule_classification(prices)

Tick rule trade classification for a series of prices.
Uptick/zero-uptick -> buy, downtick/zero-downtick -> sell.
"""
function tick_rule_classification(prices::Vector{Float64})::Vector{Int}
    n = length(prices)
    directions = zeros(Int, n)

    last_nonzero_dir = 1
    for i in 2:n
        diff = prices[i] - prices[i-1]
        if diff > 1e-10
            directions[i] = 1
            last_nonzero_dir = 1
        elseif diff < -1e-10
            directions[i] = -1
            last_nonzero_dir = -1
        else
            directions[i] = last_nonzero_dir
        end
    end
    directions[1] = 1

    return directions
end

"""
    bulk_volume_classification(prices, volumes, num_bars)

Bulk Volume Classification (BVC) for VPIN.
Assign volume to buy/sell based on normalized price change.
"""
function bulk_volume_classification(prices::Vector{Float64},
                                     volumes::Vector{Float64},
                                     num_bars::Int)
    n = length(prices)
    bar_size = max(1, n ÷ num_bars)

    buy_volume = Vector{Float64}(undef, num_bars)
    sell_volume = Vector{Float64}(undef, num_bars)
    bar_prices = Vector{Float64}(undef, num_bars)
    bar_volumes = Vector{Float64}(undef, num_bars)

    # Estimate price volatility
    returns = diff(log.(max.(prices, 1e-10)))
    sigma = length(returns) > 1 ? std(returns) : 0.01
    sigma = max(sigma, 1e-8)

    for b in 1:num_bars
        start_idx = (b - 1) * bar_size + 1
        end_idx = min(b * bar_size, n)
        if start_idx > n
            buy_volume[b] = 0.0
            sell_volume[b] = 0.0
            bar_prices[b] = prices[end]
            bar_volumes[b] = 0.0
            continue
        end

        bar_vol = sum(volumes[start_idx:end_idx])
        bar_volumes[b] = bar_vol
        bar_prices[b] = prices[end_idx]

        # Normalized price change
        dp = (prices[end_idx] - prices[start_idx]) / (sigma * sqrt(bar_size))
        # CDF of normalized change gives buy fraction
        buy_frac = _normal_cdf_fast(dp)

        buy_volume[b] = bar_vol * buy_frac
        sell_volume[b] = bar_vol * (1.0 - buy_frac)
    end

    return buy_volume, sell_volume, bar_prices, bar_volumes
end

"""
    _normal_cdf_fast(x)

Fast normal CDF approximation.
"""
function _normal_cdf_fast(x::Float64)::Float64
    if x < -8.0
        return 0.0
    elseif x > 8.0
        return 1.0
    end
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    pdf_val = exp(-0.5 * x * x) / sqrt(2.0 * pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
           t * (-1.821255978 + t * 1.330274429))))
    cdf_val = 1.0 - pdf_val * poly
    return x >= 0.0 ? cdf_val : 1.0 - cdf_val
end

"""
    _normal_inv_fast(p)

Fast inverse normal CDF.
"""
function _normal_inv_fast(p::Float64)::Float64
    if p <= 0.0
        return -8.0
    elseif p >= 1.0
        return 8.0
    end
    a = [-3.969683028665376e1, 2.209460984245205e2,
         -2.759285104469687e2, 1.383577518672690e2,
         -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2,
         -1.556989798598866e2, 6.680131188771972e1,
         -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1,
         -2.400758277161838e0, -2.549732539343734e0,
          4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1,
         2.445134137142996e0, 3.754408661907416e0]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low
        q = sqrt(-2.0 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    elseif p <= p_high
        q = p - 0.5
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6]) * q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    end
end

# ============================================================================
# SECTION 2: Realized Measures
# ============================================================================

"""
    realized_variance(prices, sampling_type=:calendar; freq=300)

Realized variance from tick prices.
sampling_type: :calendar (fixed time), :tick (every n-th tick), :business (volume-based).
freq: seconds for calendar, ticks for tick, volume for business.
"""
function realized_variance(prices::Vector{Float64};
                            sampling_type::Symbol=:calendar,
                            freq::Int=300)::Float64
    log_prices = log.(max.(prices, 1e-10))
    n = length(log_prices)

    if sampling_type == :tick
        # Sample every freq-th observation
        sampled = log_prices[1:freq:n]
    else
        sampled = log_prices
    end

    m = length(sampled)
    if m < 2
        return 0.0
    end

    rv = 0.0
    for i in 2:m
        r = sampled[i] - sampled[i-1]
        rv += r^2
    end

    return rv
end

"""
    realized_variance_subsampled(prices, num_grids)

Subsampled realized variance (Zhang, Mykland, Ait-Sahalia 2005).
Average RV over num_grids shifted grids.
"""
function realized_variance_subsampled(prices::Vector{Float64},
                                       num_grids::Int)::Float64
    n = length(prices)
    log_prices = log.(max.(prices, 1e-10))

    rv_sum = 0.0
    for g in 0:(num_grids-1)
        sampled = log_prices[(g+1):num_grids:n]
        m = length(sampled)
        rv_g = 0.0
        for i in 2:m
            rv_g += (sampled[i] - sampled[i-1])^2
        end
        rv_sum += rv_g
    end

    return rv_sum / num_grids
end

"""
    two_scale_realized_variance(prices, slow_freq)

Two-Scale Realized Variance (TSRV) - Zhang, Mykland, Ait-Sahalia (2005).
Debiases for microstructure noise.
"""
function two_scale_realized_variance(prices::Vector{Float64},
                                      slow_freq::Int)::Float64
    n = length(prices)
    log_prices = log.(max.(prices, 1e-10))

    # Fast scale (all ticks)
    rv_fast = 0.0
    for i in 2:n
        rv_fast += (log_prices[i] - log_prices[i-1])^2
    end

    # Slow scale (subsampled)
    rv_slow = realized_variance_subsampled(prices, slow_freq)

    # Optimal combination
    n_bar = (n - 1) / slow_freq
    tsrv = rv_slow - (n_bar / (n - 1)) * rv_fast

    return max(tsrv, 0.0)
end

"""
    bipower_variation(prices)

Bipower Variation (BPV) - Barndorff-Nielsen & Shephard (2004).
BPV = (pi/2) * sum |r_i| * |r_{i-1}|
Robust to jumps.
"""
function bipower_variation(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    n = length(log_prices)
    if n < 3
        return realized_variance(prices)
    end

    returns = diff(log_prices)
    m = length(returns)

    bpv = 0.0
    for i in 2:m
        bpv += abs(returns[i]) * abs(returns[i-1])
    end

    mu1 = sqrt(2.0 / pi)
    bpv *= (pi / 2.0) * (m / (m - 1))

    return bpv
end

"""
    realized_quarticity(prices)

Realized quarticity: sum r_i^4 * (n/3).
"""
function realized_quarticity(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)
    rq = sum(r^4 for r in returns) * n / 3.0
    return rq
end

"""
    tripower_quarticity(prices)

Tripower quarticity: robust to jumps.
"""
function tripower_quarticity(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    if n < 4
        return realized_quarticity(prices)
    end

    mu43 = 2.0^(2.0/3.0) * gamma(7.0/6.0) / gamma(0.5)

    tpq = 0.0
    for i in 3:n
        tpq += abs(returns[i])^(4.0/3.0) * abs(returns[i-1])^(4.0/3.0) * abs(returns[i-2])^(4.0/3.0)
    end

    tpq *= n * n / ((n - 2) * mu43^3)
    return tpq
end

"""
    medRV(prices)

Median Realized Variance (Andersen, Dobrev, Schaumburg 2012).
medRV = (pi/(6-4*sqrt(3)+pi)) * (n/(n-2)) * sum median(|r_{i-1}|,|r_i|,|r_{i+1}|)^2
"""
function medRV(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    if n < 3
        return realized_variance(prices)
    end

    c_med = pi / (6.0 - 4.0 * sqrt(3.0) + pi)

    med_sum = 0.0
    for i in 2:(n-1)
        vals = sort([abs(returns[i-1]), abs(returns[i]), abs(returns[i+1])])
        med_val = vals[2]
        med_sum += med_val^2
    end

    return c_med * (n / (n - 2)) * med_sum
end

"""
    minRV(prices)

Minimum Realized Variance.
minRV = (pi/(pi-2)) * (n/(n-1)) * sum min(|r_i|, |r_{i+1}|)^2
"""
function minRV(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    if n < 2
        return realized_variance(prices)
    end

    c_min = pi / (pi - 2.0)

    min_sum = 0.0
    for i in 1:(n-1)
        min_val = min(abs(returns[i]), abs(returns[i+1]))
        min_sum += min_val^2
    end

    return c_min * (n / (n - 1)) * min_sum
end

"""
    jump_test_bns(prices; alpha=0.05)

Barndorff-Nielsen & Shephard (2006) jump test.
Test statistic: (RV - BPV) / sqrt(theta * max(TPQ, RQ))
Under H0 (no jumps): Z ~ N(0,1).
"""
function jump_test_bns(prices::Vector{Float64}; alpha::Float64=0.05)
    rv = realized_variance(prices)
    bpv = bipower_variation(prices)
    tpq = tripower_quarticity(prices)

    log_prices = log.(max.(prices, 1e-10))
    n = length(log_prices) - 1

    # Theta: (pi^2/4 + pi - 5) * (1/n)
    theta = (pi^2 / 4.0 + pi - 5.0)

    denominator = sqrt(theta * max(tpq, 1e-20))
    if denominator < 1e-15
        return (statistic=0.0, p_value=1.0, has_jump=false,
                jump_variation=0.0, continuous_variation=bpv)
    end

    z = (rv - bpv) / denominator
    z = max(z, 0.0)

    # One-sided test
    p_value = 1.0 - _normal_cdf_fast(z)
    z_crit = -_normal_inv_fast(alpha)

    jump_var = max(rv - bpv, 0.0)

    return (statistic=z, p_value=p_value, has_jump=z > z_crit,
            jump_variation=jump_var, continuous_variation=bpv)
end

"""
    jump_test_jiang_oomen(prices; alpha=0.05)

Jiang-Oomen (2008) swap-variance based jump test.
"""
function jump_test_jiang_oomen(prices::Vector{Float64}; alpha::Float64=0.05)
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    # Realized variance
    rv = sum(r^2 for r in returns)

    # Swap variance: 2 * sum(r_i - log(1 + r_i)) using prices
    # sv = 2 * sum((P_i - P_{i-1})/P_{i-1} - log(P_i/P_{i-1}))
    sv = 0.0
    for i in 2:length(prices)
        simple_return = (prices[i] - prices[i-1]) / max(prices[i-1], 1e-10)
        log_return = log(max(prices[i] / prices[i-1], 1e-10))
        sv += 2.0 * (simple_return - log_return)
    end

    # Test statistic
    diff_val = sv - rv
    bpv = bipower_variation(prices)
    tpq = tripower_quarticity(prices)

    se = sqrt(max(tpq, 1e-20) * 2.0 / n)
    z = diff_val / max(se, 1e-15)

    p_value = 1.0 - _normal_cdf_fast(abs(z))
    z_crit = -_normal_inv_fast(alpha / 2.0)

    return (statistic=z, p_value=p_value, has_jump=abs(z) > z_crit,
            rv=rv, sv=sv)
end

"""
    realized_kernel(prices; kernel_type=:parzen, bandwidth=0)

Realized Kernel estimator (Barndorff-Nielsen, Hansen, Lunde, Shephard 2008).
Robust to microstructure noise.
"""
function realized_kernel(prices::Vector{Float64};
                          kernel_type::Symbol=:parzen,
                          bandwidth::Int=0)::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    # Automatic bandwidth selection
    if bandwidth == 0
        # Rule of thumb: H ~ c * n^{2/3}
        bandwidth = max(1, round(Int, 0.5 * n^(2.0/3.0)))
    end

    # Kernel function
    function kernel_weight(x::Float64, ktype::Symbol)::Float64
        if ktype == :parzen
            if abs(x) <= 0.5
                return 1.0 - 6.0 * x^2 + 6.0 * abs(x)^3
            elseif abs(x) <= 1.0
                return 2.0 * (1.0 - abs(x))^3
            else
                return 0.0
            end
        elseif ktype == :bartlett
            return max(1.0 - abs(x), 0.0)
        elseif ktype == :tukey_hanning
            if abs(x) <= 1.0
                return (1.0 + cos(pi * x)) / 2.0
            else
                return 0.0
            end
        elseif ktype == :flat_top
            if abs(x) <= 0.5
                return 1.0
            elseif abs(x) <= 1.0
                return 2.0 * (1.0 - abs(x))
            else
                return 0.0
            end
        else
            return max(1.0 - abs(x), 0.0)
        end
    end

    # Compute autocovariances
    gamma = zeros(bandwidth + 1)
    for h in 0:bandwidth
        s = 0.0
        count = 0
        for i in (h+1):n
            s += returns[i] * returns[i - h]
            count += 1
        end
        gamma[h+1] = count > 0 ? s : 0.0
    end

    # Realized kernel
    rk = gamma[1]
    for h in 1:bandwidth
        w = kernel_weight(h / (bandwidth + 1.0), kernel_type)
        rk += 2.0 * w * gamma[h+1]
    end

    return max(rk, 0.0)
end

"""
    realized_semivariance(prices)

Realized semivariance: decompose into positive and negative components.
"""
function realized_semivariance(prices::Vector{Float64})
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)

    rs_pos = sum(r^2 for r in returns if r > 0.0)
    rs_neg = sum(r^2 for r in returns if r <= 0.0)

    return (positive=rs_pos, negative=rs_neg, total=rs_pos + rs_neg,
            signed=rs_neg - rs_pos)
end

"""
    realized_covariance(prices1, prices2, timestamps1, timestamps2)

Realized covariance using refresh-time sampling (Barndorff-Nielsen et al.).
"""
function realized_covariance(prices1::Vector{Float64}, prices2::Vector{Float64},
                              timestamps1::Vector{Float64}, timestamps2::Vector{Float64})::Float64
    # Refresh time: both assets must have ticked
    refresh_times = Float64[]
    refresh_p1 = Float64[]
    refresh_p2 = Float64[]

    i1, i2 = 1, 1
    n1, n2 = length(timestamps1), length(timestamps2)

    while i1 <= n1 && i2 <= n2
        t = max(timestamps1[i1], timestamps2[i2])
        # Advance both to at least t
        while i1 < n1 && timestamps1[i1+1] <= t
            i1 += 1
        end
        while i2 < n2 && timestamps2[i2+1] <= t
            i2 += 1
        end

        push!(refresh_times, t)
        push!(refresh_p1, prices1[i1])
        push!(refresh_p2, prices2[i2])

        i1 += 1
        i2 += 1
    end

    # Compute covariance from synchronized returns
    m = length(refresh_p1)
    if m < 2
        return 0.0
    end

    rcov = 0.0
    for i in 2:m
        r1 = log(max(refresh_p1[i] / refresh_p1[i-1], 1e-10))
        r2 = log(max(refresh_p2[i] / refresh_p2[i-1], 1e-10))
        rcov += r1 * r2
    end

    return rcov
end

# ============================================================================
# SECTION 3: Market Microstructure
# ============================================================================

"""
    bid_ask_bounce(prices)

Estimate bid-ask bounce effect.
Autocovariance of returns: gamma_1 = -sigma_s^2 / 4
where sigma_s is the spread.
"""
function bid_ask_bounce(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    if n < 2
        return 0.0
    end

    # First-order autocovariance
    mean_r = mean(returns)
    gamma1 = 0.0
    for i in 2:n
        gamma1 += (returns[i] - mean_r) * (returns[i-1] - mean_r)
    end
    gamma1 /= (n - 1)

    # Implied spread
    spread_sq = -4.0 * gamma1
    return spread_sq > 0.0 ? sqrt(spread_sq) : 0.0
end

"""
    roll_spread(prices)

Roll (1984) implied spread estimator.
s = 2 * sqrt(-cov(r_t, r_{t-1}))
"""
function roll_spread(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    if n < 2
        return 0.0
    end

    gamma1 = 0.0
    for i in 2:n
        gamma1 += returns[i] * returns[i-1]
    end
    gamma1 /= (n - 1)

    if gamma1 < 0.0
        return 2.0 * sqrt(-gamma1)
    else
        return 0.0
    end
end

"""
    effective_spread(trade_prices, midpoints)

Effective spread: 2 * |trade_price - midpoint| / midpoint.
"""
function effective_spread(trade_prices::Vector{Float64},
                          midpoints::Vector{Float64})::Float64
    n = min(length(trade_prices), length(midpoints))
    if n == 0
        return 0.0
    end

    total = 0.0
    for i in 1:n
        mid = midpoints[i]
        if mid > 0.0
            total += 2.0 * abs(trade_prices[i] - mid) / mid
        end
    end

    return total / n
end

"""
    quoted_spread(bids, asks)

Quoted spread statistics.
"""
function quoted_spread(bids::Vector{Float64}, asks::Vector{Float64})
    n = min(length(bids), length(asks))
    abs_spreads = [asks[i] - bids[i] for i in 1:n]
    rel_spreads = [2.0 * (asks[i] - bids[i]) / (asks[i] + bids[i])
                   for i in 1:n if asks[i] + bids[i] > 0.0]

    return (mean_abs=mean(abs_spreads), median_abs=median(abs_spreads),
            mean_rel=mean(rel_spreads), median_rel=median(rel_spreads),
            std_abs=std(abs_spreads))
end

"""
    realized_spread(trade_prices, midpoints_at_trade, midpoints_after, directions)

Realized spread: 2 * d_i * (P_i - M_{i+delta}) / M_i
where d_i is trade direction.
"""
function realized_spread(trade_prices::Vector{Float64},
                          midpoints_at_trade::Vector{Float64},
                          midpoints_after::Vector{Float64},
                          directions::Vector{Int})::Float64
    n = length(trade_prices)
    total = 0.0
    count = 0

    for i in 1:n
        mid = midpoints_at_trade[i]
        if mid > 0.0 && directions[i] != 0
            rs = 2.0 * directions[i] * (trade_prices[i] - midpoints_after[i]) / mid
            total += rs
            count += 1
        end
    end

    return count > 0 ? total / count : 0.0
end

"""
    price_impact(trade_prices, midpoints_before, midpoints_after, directions)

Price impact: 2 * d_i * (M_{i+delta} - M_i) / M_i.
"""
function price_impact(trade_prices::Vector{Float64},
                      midpoints_before::Vector{Float64},
                      midpoints_after::Vector{Float64},
                      directions::Vector{Int})::Float64
    n = length(trade_prices)
    total = 0.0
    count = 0

    for i in 1:n
        mid = midpoints_before[i]
        if mid > 0.0 && directions[i] != 0
            pi_val = 2.0 * directions[i] * (midpoints_after[i] - mid) / mid
            total += pi_val
            count += 1
        end
    end

    return count > 0 ? total / count : 0.0
end

"""
    amihud_illiquidity(returns, volumes; daily=true)

Amihud (2002) illiquidity measure: ILLIQ = (1/D) * sum |r_d| / volume_d.
"""
function amihud_illiquidity(returns::Vector{Float64},
                            volumes::Vector{Float64})::Float64
    n = min(length(returns), length(volumes))
    total = 0.0
    count = 0
    for i in 1:n
        if volumes[i] > 0.0
            total += abs(returns[i]) / volumes[i]
            count += 1
        end
    end
    return count > 0 ? total / count : 0.0
end

"""
    pastor_stambaugh_liquidity(returns, market_returns, volumes)

Pastor-Stambaugh (2003) liquidity measure.
gamma from regression: r_{t+1} = alpha + beta*r_t + gamma*sign(r_t)*v_t + eps.
"""
function pastor_stambaugh_liquidity(returns::Vector{Float64},
                                     market_returns::Vector{Float64},
                                     volumes::Vector{Float64})::Float64
    n = min(length(returns) - 1, length(market_returns) - 1, length(volumes) - 1)
    if n < 5
        return 0.0
    end

    # Construct regression: r_{t+1} = a + b*r_t + c*sign(r_t)*v_t + eps
    Y = returns[2:(n+1)]
    X = zeros(n, 3)
    for i in 1:n
        X[i, 1] = 1.0  # intercept
        X[i, 2] = returns[i]
        X[i, 3] = sign(returns[i]) * volumes[i]
    end

    beta = (X' * X) \ (X' * Y)
    return beta[3]  # gamma = liquidity measure
end

"""
    corwin_schultz_spread(high_prices, low_prices)

Corwin-Schultz (2012) spread estimator from daily high-low prices.
"""
function corwin_schultz_spread(high_prices::Vector{Float64},
                                low_prices::Vector{Float64})::Vector{Float64}
    n = length(high_prices)
    spreads = Vector{Float64}(undef, max(n - 1, 0))

    for i in 2:n
        # Beta
        beta_val = (log(high_prices[i] / low_prices[i]))^2 +
                   (log(high_prices[i-1] / low_prices[i-1]))^2

        # Gamma
        h2 = max(high_prices[i], high_prices[i-1])
        l2 = min(low_prices[i], low_prices[i-1])
        gamma_val = (log(h2 / l2))^2

        # Alpha
        alpha = (sqrt(2.0 * beta_val) - sqrt(beta_val)) / (3.0 - 2.0 * sqrt(2.0)) -
                sqrt(gamma_val / (3.0 - 2.0 * sqrt(2.0)))

        # Spread
        spreads[i-1] = 2.0 * (exp(alpha) - 1.0) / (1.0 + exp(alpha))
        spreads[i-1] = max(spreads[i-1], 0.0)
    end

    return spreads
end

# ============================================================================
# SECTION 4: VPIN (Volume-Synchronized PIN)
# ============================================================================

"""
    VPINResult

VPIN computation result.
"""
struct VPINResult
    vpin::Vector{Float64}
    timestamps::Vector{Float64}
    buy_volumes::Vector{Float64}
    sell_volumes::Vector{Float64}
end

"""
    compute_vpin(prices, volumes, bucket_size, num_buckets_window)

VPIN (Easley, Lopez de Prado, O'Hara 2012).
1. Partition volume into equal-sized buckets.
2. Classify volume in each bucket as buy/sell using BVC.
3. VPIN = sum|V^B - V^S| over rolling window / (n * bucket_size).
"""
function compute_vpin(prices::Vector{Float64}, volumes::Vector{Float64},
                      bucket_size::Float64, num_buckets_window::Int)::VPINResult
    n = length(prices)
    log_prices = log.(max.(prices, 1e-10))

    # Create volume buckets
    buckets_buy = Float64[]
    buckets_sell = Float64[]
    bucket_times = Float64[]

    cum_vol = 0.0
    bucket_buy = 0.0
    bucket_sell = 0.0

    # Price vol for BVC
    returns = diff(log_prices)
    sigma = length(returns) > 10 ? std(returns) : 0.01
    sigma = max(sigma, 1e-8)

    for i in 2:n
        dp = log_prices[i] - log_prices[i-1]
        z = dp / sigma
        buy_frac = _normal_cdf_fast(z)

        v = volumes[i]
        bucket_buy += v * buy_frac
        bucket_sell += v * (1.0 - buy_frac)
        cum_vol += v

        if cum_vol >= bucket_size
            push!(buckets_buy, bucket_buy)
            push!(buckets_sell, bucket_sell)
            push!(bucket_times, Float64(i))

            # Reset
            overflow = cum_vol - bucket_size
            bucket_buy = overflow * buy_frac
            bucket_sell = overflow * (1.0 - buy_frac)
            cum_vol = overflow
        end
    end

    # Compute VPIN over rolling window
    num_buckets = length(buckets_buy)
    vpin = Vector{Float64}(undef, max(num_buckets - num_buckets_window + 1, 0))
    vpin_times = Vector{Float64}(undef, length(vpin))

    for i in num_buckets_window:num_buckets
        window_start = i - num_buckets_window + 1
        order_imbalance = 0.0
        total_volume = 0.0

        for j in window_start:i
            order_imbalance += abs(buckets_buy[j] - buckets_sell[j])
            total_volume += buckets_buy[j] + buckets_sell[j]
        end

        idx = i - num_buckets_window + 1
        vpin[idx] = total_volume > 0.0 ? order_imbalance / total_volume : 0.0
        vpin_times[idx] = bucket_times[i]
    end

    return VPINResult(vpin, vpin_times, buckets_buy, buckets_sell)
end

"""
    vpin_cdf(vpin_values, current_vpin)

CDF of VPIN: percentile rank of current VPIN in historical distribution.
"""
function vpin_cdf(vpin_values::Vector{Float64}, current_vpin::Float64)::Float64
    n = length(vpin_values)
    if n == 0
        return 0.0
    end
    count = sum(v <= current_vpin for v in vpin_values)
    return count / n
end

"""
    vpin_toxicity_alert(vpin_result::VPINResult; threshold=0.9)

Generate toxicity alerts when VPIN exceeds threshold percentile.
"""
function vpin_toxicity_alert(vpin_result::VPINResult; threshold::Float64=0.9)
    n = length(vpin_result.vpin)
    if n == 0
        return Int[]
    end

    sorted_vpin = sort(vpin_result.vpin)
    cutoff_idx = ceil(Int, threshold * n)
    cutoff_val = sorted_vpin[min(cutoff_idx, n)]

    alerts = [i for i in 1:n if vpin_result.vpin[i] >= cutoff_val]
    return alerts
end

# ============================================================================
# SECTION 5: Kyle's Lambda
# ============================================================================

"""
    kyle_lambda(price_changes, order_flow, volumes)

Kyle's lambda: permanent price impact coefficient.
dp_t = lambda * OI_t + eps_t
where OI_t is signed order flow.
"""
function kyle_lambda(price_changes::Vector{Float64},
                     order_flow::Vector{Float64};
                     volume_weighted::Bool=false)
    n = min(length(price_changes), length(order_flow))
    if n < 3
        return (lambda=0.0, r_squared=0.0, t_stat=0.0)
    end

    y = price_changes[1:n]
    x = order_flow[1:n]

    # OLS regression: dp = alpha + lambda * OI + eps
    X = hcat(ones(n), x)
    beta = (X' * X) \ (X' * y)

    residuals = y - X * beta
    ss_res = sum(residuals.^2)
    ss_tot = sum((y .- mean(y)).^2)
    r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)

    # Standard error
    se_beta = sqrt(ss_res / (n - 2) / max(sum((x .- mean(x)).^2), 1e-15))
    t_stat = beta[2] / max(se_beta, 1e-15)

    return (lambda=beta[2], r_squared=r_sq, t_stat=t_stat, intercept=beta[1])
end

"""
    kyle_lambda_rolling(price_changes, order_flow, window)

Rolling Kyle's lambda.
"""
function kyle_lambda_rolling(price_changes::Vector{Float64},
                              order_flow::Vector{Float64},
                              window::Int)::Vector{Float64}
    n = min(length(price_changes), length(order_flow))
    lambdas = Vector{Float64}(undef, max(n - window + 1, 0))

    for i in window:n
        dp = price_changes[(i-window+1):i]
        oi = order_flow[(i-window+1):i]
        result = kyle_lambda(dp, oi)
        lambdas[i - window + 1] = result.lambda
    end

    return lambdas
end

"""
    kyle_lambda_nonlinear(price_changes, order_flow)

Nonlinear Kyle's lambda: dp = lambda * sign(OI) * |OI|^delta + eps.
Estimate (lambda, delta) via nonlinear least squares.
"""
function kyle_lambda_nonlinear(price_changes::Vector{Float64},
                                order_flow::Vector{Float64})
    n = min(length(price_changes), length(order_flow))
    y = price_changes[1:n]
    x = order_flow[1:n]

    # Grid search over delta
    best_sse = Inf
    best_lambda = 0.0
    best_delta = 1.0

    for delta in 0.1:0.05:2.0
        # Transform: z = sign(x) * |x|^delta
        z = [sign(x[i]) * abs(x[i])^delta for i in 1:n]
        Z = hcat(ones(n), z)
        beta = (Z' * Z) \ (Z' * y)
        residuals = y - Z * beta
        sse = sum(residuals.^2)

        if sse < best_sse
            best_sse = sse
            best_lambda = beta[2]
            best_delta = delta
        end
    end

    return (lambda=best_lambda, delta=best_delta, sse=best_sse)
end

# ============================================================================
# SECTION 6: Hasbrouck's Information Share
# ============================================================================

"""
    hasbrouck_information_share(prices_matrix, lags)

Hasbrouck (1995) information share via VECM and VMA representation.
prices_matrix: T x K matrix of co-integrated price series.
Returns upper and lower bounds of information share for each market.
"""
function hasbrouck_information_share(prices_matrix::Matrix{Float64}, lags::Int)
    T, K = size(prices_matrix)

    # Step 1: Estimate VECM
    # dp_t = alpha * beta' * p_{t-1} + sum Gamma_i * dp_{t-i} + eps_t
    dp = diff(prices_matrix, dims=1)
    n = size(dp, 1)

    # Cointegrating vector: assume [1, -1, ..., -1] for K prices
    # This assumes prices share a common efficient price
    beta_coint = ones(K)
    beta_coint[2:end] .= -1.0

    # Error correction term
    ec = prices_matrix[1:n, :] * beta_coint  # z_{t-1}

    # Build regression matrix
    p_rhs = lags * K + 1  # +1 for EC term
    if n <= p_rhs + lags
        return (lower=ones(K) / K, upper=ones(K) / K)
    end

    X = zeros(n - lags, p_rhs)
    Y = dp[(lags+1):n, :]

    for i in 1:(n - lags)
        X[i, 1] = ec[i + lags - 1]
        for l in 1:lags
            for k in 1:K
                col = 1 + (l - 1) * K + k
                X[i, col] = dp[i + lags - l, k]
            end
        end
    end

    # OLS for each equation
    coeffs = (X' * X) \ (X' * Y)
    residuals = Y - X * coeffs

    # Variance-covariance of residuals
    Omega = (residuals' * residuals) / (size(residuals, 1) - p_rhs)

    # Step 2: VMA representation - compute long-run impact
    # Psi(1) = (I - sum Gamma_i)^{-1}
    Gamma_sum = zeros(K, K)
    for l in 1:lags
        for k1 in 1:K
            for k2 in 1:K
                col = 1 + (l - 1) * K + k2
                Gamma_sum[k1, k2] += coeffs[col, k1]
            end
        end
    end

    Psi1 = inv(I - Gamma_sum)

    # Common row: all rows of Psi(1) should be proportional
    # Use first row as psi
    psi = Psi1[1, :]

    # Step 3: Information share
    # IS_j = (psi * Chol(Omega))_j^2 / (psi * Omega * psi')
    C = cholesky(Symmetric(Omega + 1e-10 * I)).L
    psi_C = psi' * C

    total = dot(psi, Omega * psi)

    # Upper bounds (maximize over rotations of Cholesky)
    upper = [(psi_C[j])^2 / max(total, 1e-15) for j in 1:K]

    # Lower bounds (minimize by permuting Cholesky columns)
    lower = copy(upper)

    # For 2 markets, exact bounds
    if K == 2
        # Upper bound for market 1 = lower bound for market 2 and vice versa
        sigma_12 = Omega[1, 2]
        sigma_11 = Omega[1, 1]
        sigma_22 = Omega[2, 2]

        # Permuted Cholesky (swap order)
        C_perm = cholesky(Symmetric(Omega[end:-1:1, end:-1:1] + 1e-10 * I)).L
        psi_perm = psi[end:-1:1]
        psi_C_perm = psi_perm' * C_perm

        for j in 1:K
            lower[j] = min(upper[j], (psi_C_perm[K+1-j])^2 / max(total, 1e-15))
        end
    end

    return (lower=lower, upper=upper, psi=psi, omega=Omega)
end

# ============================================================================
# SECTION 7: Order Flow Toxicity
# ============================================================================

"""
    flow_toxicity_index(prices, volumes, trade_directions, window)

Composite flow toxicity index combining multiple measures.
"""
function flow_toxicity_index(prices::Vector{Float64}, volumes::Vector{Float64},
                              trade_directions::Vector{Int}, window::Int)
    n = length(prices)
    if n < window + 1
        return (fti=Float64[], timestamps=Int[])
    end

    fti = Vector{Float64}(undef, n - window)

    for i in (window+1):n
        idx = (i-window):i

        # Component 1: Order imbalance
        net_flow = sum(trade_directions[j] * volumes[j] for j in idx)
        total_vol = sum(volumes[j] for j in idx)
        oi_ratio = total_vol > 0 ? abs(net_flow) / total_vol : 0.0

        # Component 2: Price volatility / volume
        sub_prices = prices[idx]
        log_returns = diff(log.(max.(sub_prices, 1e-10)))
        vol = length(log_returns) > 1 ? std(log_returns) : 0.0
        kyle_proxy = vol / max(sqrt(total_vol), 1e-10)

        # Component 3: Spread proxy (Roll)
        roll = roll_spread(sub_prices)

        # Composite
        fti[i - window] = 0.4 * oi_ratio + 0.3 * (kyle_proxy * 1000.0) + 0.3 * (roll * 100.0)
    end

    return (fti=fti, timestamps=collect((window+1):n))
end

"""
    pin_model(buy_trades, sell_trades; max_iter=200, tol=1e-8)

Estimate PIN (Probability of Informed Trading) from daily buy/sell counts.
Easley, Kiefer, O'Hara, Paperman (1996).
Parameters: (alpha, delta, mu, epsilon_b, epsilon_s)
"""
function pin_model(buy_trades::Vector{Int}, sell_trades::Vector{Int};
                   max_iter::Int=200, tol::Float64=1e-8)
    D = length(buy_trades)

    # Initial estimates
    total_buys = sum(buy_trades)
    total_sells = sum(sell_trades)
    eps_b = total_buys / D * 0.5
    eps_s = total_sells / D * 0.5
    mu = max((total_buys + total_sells) / D * 0.3, 1.0)
    alpha = 0.5
    delta = 0.5

    for iter in 1:max_iter
        old_params = [alpha, delta, mu, eps_b, eps_s]

        # E-step: compute posterior probabilities
        p_no_event = zeros(D)
        p_good_news = zeros(D)
        p_bad_news = zeros(D)

        for d in 1:D
            B = buy_trades[d]
            S = sell_trades[d]

            # Log-likelihoods (Poisson)
            ll_no = -eps_b - eps_s + B * log(max(eps_b, 1e-10)) + S * log(max(eps_s, 1e-10))
            ll_good = -(eps_b + mu) - eps_s + B * log(max(eps_b + mu, 1e-10)) + S * log(max(eps_s, 1e-10))
            ll_bad = -eps_b - (eps_s + mu) + B * log(max(eps_b, 1e-10)) + S * log(max(eps_s + mu, 1e-10))

            # Prior-weighted
            log_p_no = log(max(1.0 - alpha, 1e-10)) + ll_no
            log_p_good = log(max(alpha * (1.0 - delta), 1e-10)) + ll_good
            log_p_bad = log(max(alpha * delta, 1e-10)) + ll_bad

            max_ll = max(log_p_no, log_p_good, log_p_bad)
            p_no_event[d] = exp(log_p_no - max_ll)
            p_good_news[d] = exp(log_p_good - max_ll)
            p_bad_news[d] = exp(log_p_bad - max_ll)

            total = p_no_event[d] + p_good_news[d] + p_bad_news[d]
            p_no_event[d] /= total
            p_good_news[d] /= total
            p_bad_news[d] /= total
        end

        # M-step: update parameters
        alpha_new = 1.0 - sum(p_no_event) / D
        delta_new = sum(p_bad_news) / max(sum(p_good_news) + sum(p_bad_news), 1e-10)

        mu_new = 0.0
        eps_b_new = 0.0
        eps_s_new = 0.0

        for d in 1:D
            B = buy_trades[d]
            S = sell_trades[d]

            eps_b_new += p_no_event[d] * B + p_bad_news[d] * B
            eps_s_new += p_no_event[d] * S + p_good_news[d] * S

            if eps_b + mu > 0
                mu_new += p_good_news[d] * B * mu / (eps_b + mu)
            end
            if eps_s + mu > 0
                mu_new += p_bad_news[d] * S * mu / (eps_s + mu)
            end
        end

        denominator_b = sum(p_no_event) + sum(p_bad_news)
        denominator_s = sum(p_no_event) + sum(p_good_news)

        eps_b_new /= max(denominator_b, 1e-10)
        eps_s_new /= max(denominator_s, 1e-10)
        mu_new /= max(sum(p_good_news) + sum(p_bad_news), 1e-10)

        alpha = clamp(alpha_new, 0.001, 0.999)
        delta = clamp(delta_new, 0.001, 0.999)
        mu = max(mu_new, 0.1)
        eps_b = max(eps_b_new, 0.1)
        eps_s = max(eps_s_new, 0.1)

        new_params = [alpha, delta, mu, eps_b, eps_s]
        if norm(new_params - old_params) < tol
            break
        end
    end

    # PIN = alpha * mu / (alpha * mu + eps_b + eps_s)
    pin = alpha * mu / (alpha * mu + eps_b + eps_s)

    return (pin=pin, alpha=alpha, delta=delta, mu=mu,
            epsilon_buy=eps_b, epsilon_sell=eps_s)
end

# ============================================================================
# SECTION 8: Lead-Lag Analysis
# ============================================================================

"""
    hayashi_yoshida_correlation(prices1, times1, prices2, times2)

Hayashi-Yoshida (2005) estimator for asynchronous data.
No need for synchronization - uses overlapping intervals.
"""
function hayashi_yoshida_correlation(prices1::Vector{Float64}, times1::Vector{Float64},
                                     prices2::Vector{Float64}, times2::Vector{Float64})::Float64
    n1 = length(prices1) - 1
    n2 = length(prices2) - 1

    cov_hy = 0.0
    var1 = 0.0
    var2 = 0.0

    for i in 1:n1
        r1 = log(max(prices1[i+1] / prices1[i], 1e-10))
        t1_start = times1[i]
        t1_end = times1[i+1]
        var1 += r1^2

        for j in 1:n2
            t2_start = times2[j]
            t2_end = times2[j+1]

            # Check overlap
            if t1_start < t2_end && t2_start < t1_end
                r2 = log(max(prices2[j+1] / prices2[j], 1e-10))
                cov_hy += r1 * r2
            end
        end
    end

    for j in 1:n2
        r2 = log(max(prices2[j+1] / prices2[j], 1e-10))
        var2 += r2^2
    end

    denom = sqrt(max(var1, 1e-15)) * sqrt(max(var2, 1e-15))
    return cov_hy / max(denom, 1e-15)
end

"""
    lead_lag_correlation(prices1, prices2, max_lag)

Cross-correlation function at different lags.
Positive lag k means series 1 leads series 2 by k.
"""
function lead_lag_correlation(prices1::Vector{Float64}, prices2::Vector{Float64},
                               max_lag::Int)
    log_p1 = log.(max.(prices1, 1e-10))
    log_p2 = log.(max.(prices2, 1e-10))
    r1 = diff(log_p1)
    r2 = diff(log_p2)
    n = min(length(r1), length(r2))
    r1 = r1[1:n]
    r2 = r2[1:n]

    s1 = std(r1)
    s2 = std(r2)
    m1 = mean(r1)
    m2 = mean(r2)

    lags = collect(-max_lag:max_lag)
    correlations = Vector{Float64}(undef, length(lags))

    for (idx, k) in enumerate(lags)
        if k >= 0
            overlap = n - k
            if overlap < 1
                correlations[idx] = 0.0
                continue
            end
            c = 0.0
            for i in 1:overlap
                c += (r1[i] - m1) * (r2[i + k] - m2)
            end
            correlations[idx] = c / (overlap * max(s1 * s2, 1e-15))
        else
            overlap = n + k
            if overlap < 1
                correlations[idx] = 0.0
                continue
            end
            c = 0.0
            for i in 1:overlap
                c += (r1[i - k] - m1) * (r2[i] - m2)
            end
            correlations[idx] = c / (overlap * max(s1 * s2, 1e-15))
        end
    end

    # Detect lead-lag
    best_idx = argmax(abs.(correlations))
    best_lag = lags[best_idx]

    return (lags=lags, correlations=correlations, best_lag=best_lag,
            best_correlation=correlations[best_idx])
end

"""
    lead_lag_ratio(correlations_pos, correlations_neg)

Lead-lag ratio: LLR = sum(rho(k>0)^2) / sum(rho(k<0)^2).
LLR > 1 means series 1 leads, LLR < 1 means series 2 leads.
"""
function lead_lag_ratio(correlations::Vector{Float64}, lags::Vector{Int})::Float64
    pos_sum = sum(correlations[i]^2 for i in 1:length(lags) if lags[i] > 0)
    neg_sum = sum(correlations[i]^2 for i in 1:length(lags) if lags[i] < 0)
    return pos_sum / max(neg_sum, 1e-15)
end

# ============================================================================
# SECTION 9: Intraday Patterns
# ============================================================================

"""
    intraday_volume_pattern(timestamps, volumes, num_bins)

U-shaped volume pattern estimation.
timestamps: seconds from midnight.
"""
function intraday_volume_pattern(timestamps::Vector{Float64},
                                  volumes::Vector{Float64},
                                  num_bins::Int)
    n = length(timestamps)
    market_open = 9.5 * 3600.0   # 9:30 AM
    market_close = 16.0 * 3600.0  # 4:00 PM
    bin_width = (market_close - market_open) / num_bins

    bin_volumes = zeros(num_bins)
    bin_counts = zeros(Int, num_bins)

    for i in 1:n
        t = timestamps[i]
        if t >= market_open && t < market_close
            bin = min(floor(Int, (t - market_open) / bin_width) + 1, num_bins)
            bin_volumes[bin] += volumes[i]
            bin_counts[bin] += 1
        end
    end

    avg_volumes = [bin_counts[b] > 0 ? bin_volumes[b] / bin_counts[b] : 0.0
                   for b in 1:num_bins]

    # Fit U-shape: V(t) = a + b*(t-0.5)^2 + c*(t-0.5)^4
    bin_centers = [(b - 0.5) / num_bins for b in 1:num_bins]
    x1 = [(t - 0.5)^2 for t in bin_centers]
    x2 = [(t - 0.5)^4 for t in bin_centers]
    X = hcat(ones(num_bins), x1, x2)
    beta = (X' * X) \ (X' * avg_volumes)

    return (bin_volumes=avg_volumes, bin_counts=bin_counts,
            u_shape_coeffs=beta, bin_centers=bin_centers)
end

"""
    intraday_spread_pattern(timestamps, spreads, num_bins)

L-shaped spread pattern estimation.
"""
function intraday_spread_pattern(timestamps::Vector{Float64},
                                  spreads::Vector{Float64},
                                  num_bins::Int)
    n = length(timestamps)
    market_open = 9.5 * 3600.0
    market_close = 16.0 * 3600.0
    bin_width = (market_close - market_open) / num_bins

    bin_spreads = zeros(num_bins)
    bin_counts = zeros(Int, num_bins)

    for i in 1:n
        t = timestamps[i]
        if t >= market_open && t < market_close
            bin = min(floor(Int, (t - market_open) / bin_width) + 1, num_bins)
            bin_spreads[bin] += spreads[i]
            bin_counts[bin] += 1
        end
    end

    avg_spreads = [bin_counts[b] > 0 ? bin_spreads[b] / bin_counts[b] : 0.0
                   for b in 1:num_bins]

    return (avg_spreads=avg_spreads, bin_counts=bin_counts)
end

"""
    intraday_volatility_pattern(timestamps, prices, num_bins)

Intraday volatility pattern.
"""
function intraday_volatility_pattern(timestamps::Vector{Float64},
                                      prices::Vector{Float64},
                                      num_bins::Int)
    n = length(timestamps)
    market_open = 9.5 * 3600.0
    market_close = 16.0 * 3600.0
    bin_width = (market_close - market_open) / num_bins

    bin_returns_sq = zeros(num_bins)
    bin_counts = zeros(Int, num_bins)

    for i in 2:n
        t = timestamps[i]
        if t >= market_open && t < market_close
            bin = min(floor(Int, (t - market_open) / bin_width) + 1, num_bins)
            r = log(max(prices[i] / prices[i-1], 1e-10))
            bin_returns_sq[bin] += r^2
            bin_counts[bin] += 1
        end
    end

    avg_vol = [bin_counts[b] > 0 ? sqrt(bin_returns_sq[b] / bin_counts[b]) : 0.0
               for b in 1:num_bins]

    return (avg_volatility=avg_vol, bin_counts=bin_counts)
end

"""
    deseasonalize_intraday(timestamps, values, num_bins)

Remove intraday seasonality by dividing by time-of-day average.
"""
function deseasonalize_intraday(timestamps::Vector{Float64},
                                 values::Vector{Float64},
                                 num_bins::Int)::Vector{Float64}
    n = length(timestamps)
    market_open = 9.5 * 3600.0
    market_close = 16.0 * 3600.0
    bin_width = (market_close - market_open) / num_bins

    # Compute bin averages
    bin_sums = zeros(num_bins)
    bin_counts = zeros(Int, num_bins)

    for i in 1:n
        t = timestamps[i]
        if t >= market_open && t < market_close
            bin = min(floor(Int, (t - market_open) / bin_width) + 1, num_bins)
            bin_sums[bin] += values[i]
            bin_counts[bin] += 1
        end
    end

    bin_avgs = [bin_counts[b] > 0 ? bin_sums[b] / bin_counts[b] : 1.0
                for b in 1:num_bins]
    overall_avg = mean(filter(x -> x > 0, bin_avgs))

    # Deseasonalize
    result = copy(values)
    for i in 1:n
        t = timestamps[i]
        if t >= market_open && t < market_close
            bin = min(floor(Int, (t - market_open) / bin_width) + 1, num_bins)
            if bin_avgs[bin] > 0.0
                result[i] = values[i] * overall_avg / bin_avgs[bin]
            end
        end
    end

    return result
end

# ============================================================================
# SECTION 10: Signature Plot and Epps Effect
# ============================================================================

"""
    signature_plot(prices, timestamps, sampling_freqs)

Signature plot: realized variance vs sampling frequency.
Should be flat for a semimartingale; deviations indicate microstructure noise.
"""
function signature_plot(prices::Vector{Float64}, timestamps::Vector{Float64},
                        sampling_freqs::Vector{Int})
    n = length(prices)
    log_prices = log.(max.(prices, 1e-10))

    rvs = Vector{Float64}(undef, length(sampling_freqs))

    for (idx, freq) in enumerate(sampling_freqs)
        sampled = log_prices[1:freq:n]
        m = length(sampled)
        rv = 0.0
        for i in 2:m
            rv += (sampled[i] - sampled[i-1])^2
        end
        rvs[idx] = rv
    end

    # Annualized vols
    vols = sqrt.(rvs * 252.0)

    return (frequencies=sampling_freqs, realized_variances=rvs,
            annualized_vols=vols)
end

"""
    epps_effect(prices1, prices2, timestamps1, timestamps2, sampling_freqs)

Epps (1979) effect: correlation increases with sampling interval.
"""
function epps_effect(prices1::Vector{Float64}, prices2::Vector{Float64},
                     timestamps1::Vector{Float64}, timestamps2::Vector{Float64},
                     sampling_freqs::Vector{Int})
    correlations = Vector{Float64}(undef, length(sampling_freqs))

    for (idx, freq) in enumerate(sampling_freqs)
        # Subsample both series
        n1 = length(prices1)
        n2 = length(prices2)

        sub1 = prices1[1:freq:n1]
        sub2 = prices2[1:freq:n2]

        m = min(length(sub1), length(sub2))
        if m < 3
            correlations[idx] = 0.0
            continue
        end

        r1 = diff(log.(max.(sub1[1:m], 1e-10)))
        r2 = diff(log.(max.(sub2[1:m], 1e-10)))

        correlations[idx] = cor(r1, r2)
    end

    return (frequencies=sampling_freqs, correlations=correlations)
end

"""
    noise_variance_estimation(prices)

Estimate microstructure noise variance from first-order autocovariance.
sigma_noise^2 = -Cov(r_t, r_{t-1})
"""
function noise_variance_estimation(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    if n < 2
        return 0.0
    end

    gamma1 = 0.0
    for i in 2:n
        gamma1 += returns[i] * returns[i-1]
    end
    gamma1 /= (n - 1)

    return max(-gamma1, 0.0)
end

# ============================================================================
# SECTION 11: Market Maker Inventory Model
# ============================================================================

"""
    MRRModel

Madhavan-Richardson-Roomans (1997) model parameters.
"""
struct MRRModel
    theta::Float64   # Probability of informed trade
    phi::Float64     # Permanent impact (adverse selection)
    rho::Float64     # Autocovariance of order flow
    sigma_u::Float64 # Innovation variance
end

"""
    estimate_mrr(price_changes, trade_directions)

Estimate Madhavan-Richardson-Roomans model.
dp_t = phi * (x_t - rho * x_{t-1}) + theta * x_t + eps_t
where x_t = trade direction (+/-1).
"""
function estimate_mrr(price_changes::Vector{Float64},
                      trade_directions::Vector{Int})
    n = min(length(price_changes), length(trade_directions))
    if n < 5
        return MRRModel(0.0, 0.0, 0.0, 0.0)
    end

    y = price_changes[2:n]
    x_curr = Float64.(trade_directions[2:n])
    x_prev = Float64.(trade_directions[1:(n-1)])

    # GMM / OLS: dp_t = c1 * x_t + c2 * x_{t-1} + eps_t
    X = hcat(x_curr, x_prev)
    beta = (X' * X) \ (X' * y)

    c1 = beta[1]  # phi + theta
    c2 = beta[2]  # -phi * rho

    # Autocovariance of directions
    rho_est = 0.0
    for i in 2:length(x_curr)
        rho_est += x_curr[i] * x_curr[i-1]
    end
    rho_est /= (length(x_curr) - 1)

    phi_est = abs(rho_est) > 1e-10 ? -c2 / rho_est : 0.0
    theta_est = c1 - phi_est

    residuals = y - X * beta
    sigma_u = std(residuals)

    return MRRModel(max(theta_est, 0.0), max(phi_est, 0.0), rho_est, sigma_u)
end

"""
    inventory_model_ho_stoll(bid, ask, inventory, risk_aversion, volatility, dt)

Ho-Stoll (1981) inventory model for market maker quotes.
"""
function inventory_model_ho_stoll(bid::Float64, ask::Float64, inventory::Float64,
                                   risk_aversion::Float64, volatility::Float64,
                                   dt::Float64)
    mid = 0.5 * (bid + ask)
    spread = ask - bid

    # Optimal quotes
    inventory_adjustment = risk_aversion * volatility^2 * dt * inventory
    optimal_bid = mid - 0.5 * spread - inventory_adjustment
    optimal_ask = mid + 0.5 * spread - inventory_adjustment

    return (optimal_bid=optimal_bid, optimal_ask=optimal_ask,
            inventory_skew=inventory_adjustment)
end

"""
    avellaneda_stoikov_quotes(mid_price, inventory, risk_aversion, volatility,
                               remaining_time, arrival_rate)

Avellaneda-Stoikov (2008) optimal market making.
"""
function avellaneda_stoikov_quotes(mid_price::Float64, inventory::Float64,
                                    risk_aversion::Float64, volatility::Float64,
                                    remaining_time::Float64, arrival_rate::Float64)
    # Reservation price
    reservation = mid_price - inventory * risk_aversion * volatility^2 * remaining_time

    # Optimal spread
    optimal_spread = risk_aversion * volatility^2 * remaining_time +
                     2.0 / risk_aversion * log(1.0 + risk_aversion / arrival_rate)

    optimal_bid = reservation - 0.5 * optimal_spread
    optimal_ask = reservation + 0.5 * optimal_spread

    return (reservation_price=reservation, optimal_spread=optimal_spread,
            optimal_bid=optimal_bid, optimal_ask=optimal_ask)
end

# ============================================================================
# SECTION 12: Optimal Execution - Almgren-Chriss
# ============================================================================

"""
    AlmgrenChrissParams

Almgren-Chriss (2001) optimal execution parameters.
"""
struct AlmgrenChrissParams
    total_shares::Float64
    time_horizon::Float64
    num_steps::Int
    volatility::Float64
    permanent_impact::Float64   # gamma
    temporary_impact::Float64   # eta
    risk_aversion::Float64      # lambda
end

"""
    almgren_chriss_optimal_trajectory(params::AlmgrenChrissParams)

Compute optimal execution trajectory.
x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)
where kappa = sqrt(lambda * sigma^2 / eta)
"""
function almgren_chriss_optimal_trajectory(params::AlmgrenChrissParams)
    X = params.total_shares
    T = params.time_horizon
    N = params.num_steps
    sigma = params.volatility
    gamma = params.permanent_impact
    eta = params.temporary_impact
    lambda = params.risk_aversion

    tau = T / N
    kappa_sq = lambda * sigma^2 / eta
    kappa = sqrt(max(kappa_sq, 1e-15))

    # Optimal holdings trajectory
    holdings = Vector{Float64}(undef, N + 1)
    trade_list = Vector{Float64}(undef, N)

    sinh_kT = sinh(kappa * T)
    if abs(sinh_kT) < 1e-15
        # Linear trajectory (TWAP)
        for j in 0:N
            holdings[j+1] = X * (1.0 - j / N)
        end
    else
        for j in 0:N
            t_j = j * tau
            holdings[j+1] = X * sinh(kappa * (T - t_j)) / sinh_kT
        end
    end

    for j in 1:N
        trade_list[j] = holdings[j] - holdings[j+1]
    end

    # Expected cost and variance
    expected_cost = _ac_expected_cost(params, holdings, trade_list, tau)
    variance = _ac_variance(params, holdings, tau)

    return (holdings=holdings, trades=trade_list,
            expected_cost=expected_cost, variance=variance,
            efficient_frontier_point=(expected_cost, sqrt(variance)))
end

function _ac_expected_cost(params::AlmgrenChrissParams, holdings, trades, tau)
    gamma = params.permanent_impact
    eta = params.temporary_impact

    permanent_cost = 0.5 * gamma * params.total_shares^2
    temporary_cost = sum(eta * (trades[j] / tau)^2 * tau for j in 1:params.num_steps)

    return permanent_cost + temporary_cost
end

function _ac_variance(params::AlmgrenChrissParams, holdings, tau)
    sigma = params.volatility
    return sigma^2 * sum(holdings[j+1]^2 * tau for j in 1:params.num_steps)
end

"""
    almgren_chriss_efficient_frontier(params::AlmgrenChrissParams, num_points)

Compute efficient frontier by varying risk aversion.
"""
function almgren_chriss_efficient_frontier(params::AlmgrenChrissParams, num_points::Int)
    lambdas = exp.(range(log(1e-6), log(1e2), length=num_points))
    costs = Vector{Float64}(undef, num_points)
    risks = Vector{Float64}(undef, num_points)

    for (i, lam) in enumerate(lambdas)
        p = AlmgrenChrissParams(params.total_shares, params.time_horizon,
                                 params.num_steps, params.volatility,
                                 params.permanent_impact, params.temporary_impact, lam)
        result = almgren_chriss_optimal_trajectory(p)
        costs[i] = result.expected_cost
        risks[i] = sqrt(result.variance)
    end

    return (lambdas=lambdas, costs=costs, risks=risks)
end

"""
    almgren_chriss_with_constraints(params::AlmgrenChrissParams;
                                    max_participation=0.1, market_volume=nothing)

Almgren-Chriss with participation rate constraint.
"""
function almgren_chriss_with_constraints(params::AlmgrenChrissParams;
                                         max_participation::Float64=0.1,
                                         market_volume::Union{Nothing, Vector{Float64}}=nothing)
    result = almgren_chriss_optimal_trajectory(params)
    trades = copy(result.trades)
    tau = params.time_horizon / params.num_steps

    if market_volume !== nothing
        for j in 1:params.num_steps
            max_trade = max_participation * market_volume[min(j, length(market_volume))] * tau
            if abs(trades[j]) > max_trade
                trades[j] = sign(trades[j]) * max_trade
            end
        end

        # Redistribute remaining shares
        remaining = params.total_shares - sum(trades)
        if abs(remaining) > 1e-10
            for j in 1:params.num_steps
                trades[j] += remaining / params.num_steps
            end
        end
    end

    holdings = Vector{Float64}(undef, params.num_steps + 1)
    holdings[1] = params.total_shares
    for j in 1:params.num_steps
        holdings[j+1] = holdings[j] - trades[j]
    end

    return (holdings=holdings, trades=trades)
end

"""
    twap_trajectory(total_shares, num_steps)

Time-Weighted Average Price trajectory.
"""
function twap_trajectory(total_shares::Float64, num_steps::Int)
    trade_per_step = total_shares / num_steps
    trades = fill(trade_per_step, num_steps)
    holdings = [total_shares - i * trade_per_step for i in 0:num_steps]
    return (holdings=holdings, trades=trades)
end

"""
    vwap_trajectory(total_shares, volume_profile, num_steps)

Volume-Weighted Average Price trajectory.
"""
function vwap_trajectory(total_shares::Float64, volume_profile::Vector{Float64},
                          num_steps::Int)
    n = min(num_steps, length(volume_profile))
    vol_total = sum(volume_profile[1:n])

    trades = [total_shares * volume_profile[i] / max(vol_total, 1e-10) for i in 1:n]
    if n < num_steps
        append!(trades, zeros(num_steps - n))
    end

    holdings = Vector{Float64}(undef, num_steps + 1)
    holdings[1] = total_shares
    for j in 1:num_steps
        holdings[j+1] = holdings[j] - trades[j]
    end

    return (holdings=holdings, trades=trades)
end

"""
    implementation_shortfall(decision_prices, execution_prices, quantities, side)

Implementation shortfall analysis.
side: +1 for buy, -1 for sell.
"""
function implementation_shortfall(decision_prices::Vector{Float64},
                                   execution_prices::Vector{Float64},
                                   quantities::Vector{Float64},
                                   side::Int)
    n = length(decision_prices)
    total_quantity = sum(quantities)

    # Total cost
    execution_cost = sum(execution_prices[i] * quantities[i] for i in 1:n)
    benchmark_cost = sum(decision_prices[i] * quantities[i] for i in 1:n)

    shortfall = side * (execution_cost - benchmark_cost)
    shortfall_bps = shortfall / max(benchmark_cost, 1e-10) * 10000.0

    # Decomposition into timing and market impact
    arrival_prices = decision_prices  # Simplified
    delay_cost = 0.0
    market_impact_cost = 0.0

    for i in 1:n
        delay_cost += side * (arrival_prices[i] - decision_prices[i]) * quantities[i]
        market_impact_cost += side * (execution_prices[i] - arrival_prices[i]) * quantities[i]
    end

    return (total_shortfall=shortfall, shortfall_bps=shortfall_bps,
            delay_cost=delay_cost, market_impact=market_impact_cost,
            avg_execution_price=execution_cost / max(total_quantity, 1e-10))
end

# ============================================================================
# SECTION 13: High-Frequency Factor Model
# ============================================================================

"""
    intraday_pca(returns_matrix, num_factors)

Intraday PCA factor model from high-frequency return matrix.
returns_matrix: T x N matrix (T time intervals, N assets).
"""
function intraday_pca(returns_matrix::Matrix{Float64}, num_factors::Int)
    T, N = size(returns_matrix)

    # Demean
    means = vec(mean(returns_matrix, dims=1))
    demeaned = returns_matrix .- means'

    # Covariance matrix
    C = (demeaned' * demeaned) / (T - 1)

    # Eigendecomposition
    eig = eigen(Symmetric(C))
    idx = sortperm(eig.values, rev=true)

    eigenvalues = eig.values[idx]
    eigenvectors = eig.vectors[:, idx]

    # Select top factors
    K = min(num_factors, N)
    factor_loadings = eigenvectors[:, 1:K]
    factor_returns = demeaned * factor_loadings

    # Variance explained
    total_var = sum(eigenvalues)
    explained = cumsum(eigenvalues[1:K]) / max(total_var, 1e-15)

    # Residuals
    reconstructed = factor_returns * factor_loadings'
    residuals = demeaned - reconstructed
    idiosyncratic_var = vec(var(residuals, dims=1))

    return (loadings=factor_loadings, factors=factor_returns,
            eigenvalues=eigenvalues[1:K], explained_variance=explained,
            idiosyncratic_variance=idiosyncratic_var)
end

"""
    intraday_factor_model_regression(returns, factor_returns)

Time-series regression of asset returns on intraday factors.
"""
function intraday_factor_model_regression(returns::Vector{Float64},
                                           factor_returns::Matrix{Float64})
    T = length(returns)
    K = size(factor_returns, 2)

    X = hcat(ones(T), factor_returns)
    beta = (X' * X) \ (X' * returns)

    fitted = X * beta
    residuals = returns - fitted
    ss_res = sum(residuals.^2)
    ss_tot = sum((returns .- mean(returns)).^2)
    r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)

    return (alpha=beta[1], betas=beta[2:end], r_squared=r_sq,
            residual_vol=std(residuals))
end

"""
    realized_beta(prices_asset, prices_market, freq)

High-frequency realized beta.
"""
function realized_beta(prices_asset::Vector{Float64}, prices_market::Vector{Float64},
                        freq::Int)::Float64
    n = min(length(prices_asset), length(prices_market))
    log_a = log.(max.(prices_asset[1:n], 1e-10))
    log_m = log.(max.(prices_market[1:n], 1e-10))

    # Subsample
    sub_a = log_a[1:freq:n]
    sub_m = log_m[1:freq:n]

    r_a = diff(sub_a)
    r_m = diff(sub_m)

    cov_am = sum(r_a .* r_m)
    var_m = sum(r_m.^2)

    return var_m > 1e-15 ? cov_am / var_m : 0.0
end

"""
    jump_robust_covariance(prices_matrix)

Jump-robust covariance estimation using bipower variation approach.
"""
function jump_robust_covariance(prices_matrix::Matrix{Float64})::Matrix{Float64}
    T, N = size(prices_matrix)
    log_prices = log.(max.(prices_matrix, 1e-10))
    returns = diff(log_prices, dims=1)
    m = size(returns, 1)

    C = zeros(N, N)
    mu1 = sqrt(2.0 / pi)

    for i in 1:N
        for j in i:N
            bpcv = 0.0
            for t in 2:m
                bpcv += abs(returns[t, i]) * abs(returns[t-1, j]) +
                        abs(returns[t-1, i]) * abs(returns[t, j])
            end
            C[i, j] = (pi / 4.0) * bpcv / (m - 1)
            C[j, i] = C[i, j]
        end
    end

    return C
end

"""
    microstructure_noise_ratio(prices, slow_freq)

Noise-to-signal ratio estimation.
"""
function microstructure_noise_ratio(prices::Vector{Float64}, slow_freq::Int)::Float64
    rv_fast = realized_variance(prices)
    rv_slow = realized_variance_subsampled(prices, slow_freq)

    if rv_slow < 1e-15
        return 0.0
    end

    # Noise ratio ~ (RV_fast - RV_slow) / RV_slow
    return max((rv_fast - rv_slow) / rv_slow, 0.0)
end

"""
    optimal_sampling_frequency(prices)

Estimate optimal sampling frequency to minimize MSE.
Bandi-Russell (2008): n* ~ (IV / (4 * noise_var^2))^{1/3}
"""
function optimal_sampling_frequency(prices::Vector{Float64})::Int
    n = length(prices)
    noise_var = noise_variance_estimation(prices)

    # Estimate integrated variance from slow sampling
    slow_freq = max(1, round(Int, sqrt(n)))
    iv_est = realized_variance_subsampled(prices, slow_freq)

    if noise_var < 1e-15
        return 1
    end

    n_opt = (iv_est / (4.0 * noise_var^2))^(1.0/3.0)
    return max(1, round(Int, n / max(n_opt, 1.0)))
end

"""
    trade_informativeness(price_changes, trade_sizes, trade_directions, lags)

Measure trade informativeness via predictive regression.
dp_{t+k} = sum beta_j * f(trade_{t-j}) + eps
"""
function trade_informativeness(price_changes::Vector{Float64},
                                trade_sizes::Vector{Float64},
                                trade_directions::Vector{Int},
                                lags::Int)
    n = length(price_changes)
    if n < 2 * lags + 5
        return (coefficients=zeros(lags), r_squared=0.0)
    end

    # Build signed order flow
    signed_flow = [trade_directions[i] * trade_sizes[i] for i in 1:n]

    # Forward return as dependent variable
    Y = price_changes[(lags+1):n]
    m = length(Y)

    X = zeros(m, lags)
    for j in 1:lags
        for i in 1:m
            X[i, j] = signed_flow[i + lags - j]
        end
    end

    X_aug = hcat(ones(m), X)
    beta = (X_aug' * X_aug) \ (X_aug' * Y)

    residuals = Y - X_aug * beta
    ss_res = sum(residuals.^2)
    ss_tot = sum((Y .- mean(Y)).^2)
    r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)

    return (intercept=beta[1], coefficients=beta[2:end], r_squared=r_sq)
end

"""
    realized_skewness(prices)

Realized skewness from high-frequency data.
"""
function realized_skewness(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    rv = sum(r^2 for r in returns)
    rs3 = sum(r^3 for r in returns)

    rv32 = rv^1.5
    return rv32 > 1e-15 ? sqrt(n) * rs3 / rv32 : 0.0
end

"""
    realized_kurtosis(prices)

Realized kurtosis from high-frequency data.
"""
function realized_kurtosis(prices::Vector{Float64})::Float64
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    n = length(returns)

    rv = sum(r^2 for r in returns)
    rq = sum(r^4 for r in returns)

    rv2 = rv^2
    return rv2 > 1e-15 ? n * rq / rv2 : 3.0
end

"""
    volatility_of_volatility(prices, block_size)

Volatility of volatility (Vol-of-Vol) from high-frequency data.
"""
function volatility_of_volatility(prices::Vector{Float64}, block_size::Int)::Float64
    n = length(prices)
    num_blocks = n ÷ block_size

    if num_blocks < 3
        return 0.0
    end

    block_rvs = Vector{Float64}(undef, num_blocks)
    for b in 1:num_blocks
        start_idx = (b - 1) * block_size + 1
        end_idx = b * block_size
        block_prices = prices[start_idx:end_idx]
        block_rvs[b] = realized_variance(block_prices)
    end

    return std(block_rvs) / max(mean(block_rvs), 1e-15)
end

"""
    high_low_volatility(highs, lows)

Parkinson (1980) high-low volatility estimator.
sigma^2 = (1/4*ln(2)) * E[(ln(H/L))^2]
"""
function high_low_volatility(highs::Vector{Float64}, lows::Vector{Float64})::Float64
    n = length(highs)
    if n == 0
        return 0.0
    end
    sum_sq = sum((log(max(highs[i] / lows[i], 1e-10)))^2 for i in 1:n)
    return sqrt(sum_sq / (n * 4.0 * log(2.0)))
end

"""
    garman_klass_volatility(opens, highs, lows, closes)

Garman-Klass (1980) volatility estimator.
"""
function garman_klass_volatility(opens::Vector{Float64}, highs::Vector{Float64},
                                  lows::Vector{Float64}, closes::Vector{Float64})::Float64
    n = length(opens)
    if n == 0
        return 0.0
    end

    sum_val = 0.0
    for i in 1:n
        u = log(max(highs[i] / opens[i], 1e-10))
        d = log(max(lows[i] / opens[i], 1e-10))
        c = log(max(closes[i] / opens[i], 1e-10))

        sum_val += 0.5 * (u - d)^2 - (2.0 * log(2.0) - 1.0) * c^2
    end

    return sqrt(sum_val / n)
end

"""
    rogers_satchell_volatility(opens, highs, lows, closes)

Rogers-Satchell (1991) volatility estimator.
"""
function rogers_satchell_volatility(opens::Vector{Float64}, highs::Vector{Float64},
                                     lows::Vector{Float64}, closes::Vector{Float64})::Float64
    n = length(opens)
    if n == 0
        return 0.0
    end

    sum_val = 0.0
    for i in 1:n
        h = log(max(highs[i] / opens[i], 1e-10))
        l = log(max(lows[i] / opens[i], 1e-10))
        c = log(max(closes[i] / opens[i], 1e-10))

        sum_val += h * (h - c) + l * (l - c)
    end

    return sqrt(sum_val / n)
end

"""
    yang_zhang_volatility(opens, highs, lows, closes)

Yang-Zhang (2000) volatility estimator combining overnight and Rogers-Satchell.
"""
function yang_zhang_volatility(opens::Vector{Float64}, highs::Vector{Float64},
                                lows::Vector{Float64}, closes::Vector{Float64})::Float64
    n = length(opens)
    if n < 2
        return 0.0
    end

    # Overnight variance
    log_oc = [log(max(opens[i] / closes[max(i-1, 1)], 1e-10)) for i in 2:n]
    sigma_o_sq = var(log_oc)

    # Close-to-close variance
    log_cc = [log(max(closes[i] / closes[max(i-1, 1)], 1e-10)) for i in 2:n]
    sigma_c_sq = var(log_cc)

    # Rogers-Satchell
    rs_sq = rogers_satchell_volatility(opens, highs, lows, closes)^2

    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    sigma_sq = sigma_o_sq + k * sigma_c_sq + (1.0 - k) * rs_sq

    return sqrt(max(sigma_sq, 0.0))
end

"""
    detect_momentum_ignition(prices, volumes, window, threshold_mult)

Detect potential momentum ignition patterns.
Returns indices where rapid price + volume spike occurs.
"""
function detect_momentum_ignition(prices::Vector{Float64}, volumes::Vector{Float64},
                                   window::Int, threshold_mult::Float64)::Vector{Int}
    n = length(prices)
    if n < 2 * window
        return Int[]
    end

    log_prices = log.(max.(prices, 1e-10))
    alerts = Int[]

    for i in (window+1):(n-window)
        # Price movement in short window
        price_move = abs(log_prices[i] - log_prices[i - window])

        # Volume spike
        recent_vol = mean(volumes[max(1, i-window):i])
        baseline_vol = mean(volumes[max(1, i-3*window):(i-window)])

        # Historical volatility
        historical_returns = diff(log_prices[max(1, i-5*window):i])
        hist_vol = length(historical_returns) > 1 ? std(historical_returns) : 0.01

        # Alert conditions
        vol_spike = recent_vol > threshold_mult * baseline_vol
        price_spike = price_move > threshold_mult * hist_vol * sqrt(window)

        # Reversal check
        if i + window <= n
            reversal = abs(log_prices[i + window] - log_prices[i])
            partial_reversal = reversal > 0.5 * price_move
        else
            partial_reversal = false
        end

        if vol_spike && price_spike && partial_reversal
            push!(alerts, i)
        end
    end

    return alerts
end

"""
    quote_stuffing_detection(timestamps, num_quotes_per_second, threshold)

Detect potential quote stuffing by identifying abnormal message rates.
"""
function quote_stuffing_detection(timestamps::Vector{Float64},
                                   num_messages_per_interval::Vector{Int},
                                   threshold_mult::Float64)::Vector{Int}
    n = length(num_messages_per_interval)
    if n < 10
        return Int[]
    end

    baseline = median(Float64.(num_messages_per_interval))
    threshold = threshold_mult * baseline

    return [i for i in 1:n if num_messages_per_interval[i] > threshold]
end

"""
    market_quality_metrics(prices, volumes, bids, asks, timestamps)

Comprehensive market quality assessment.
"""
function market_quality_metrics(prices::Vector{Float64}, volumes::Vector{Float64},
                                 bids::Vector{Float64}, asks::Vector{Float64},
                                 timestamps::Vector{Float64})
    n = length(prices)

    # Spread measures
    qs = quoted_spread(bids, asks)
    midpoints = 0.5 .* (bids .+ asks)
    eff_spread = effective_spread(prices, midpoints)
    roll_est = roll_spread(prices)

    # Volatility measures
    rv = realized_variance(prices)
    bpv = bipower_variation(prices)
    jump_result = jump_test_bns(prices)

    # Depth
    avg_depth = mean(volumes)

    # Noise
    noise_var = noise_variance_estimation(prices)

    return (quoted_spread=qs.mean_rel, effective_spread=eff_spread,
            roll_spread=roll_est, realized_variance=rv,
            bipower_variation=bpv, jump_present=jump_result.has_jump,
            avg_depth=avg_depth, noise_variance=noise_var,
            noise_to_signal=noise_var / max(rv, 1e-15))
end

# ============================================================================
# SECTION 14: Additional HF Analytics
# ============================================================================

"""
    multi_scale_realized_variance(prices, scales)

Multi-scale realized variance across different sampling frequencies.
"""
function multi_scale_realized_variance(prices::Vector{Float64},
                                        scales::Vector{Int})
    rvs = [realized_variance_subsampled(prices, s) for s in scales]
    return (scales=scales, realized_variances=rvs,
            annualized_vols=sqrt.(rvs * 252.0))
end

"""
    pre_averaging_estimator(prices, theta)

Pre-averaging estimator for integrated variance (Jacod et al. 2009).
Handles microstructure noise by pre-averaging returns.
"""
function pre_averaging_estimator(prices::Vector{Float64}, theta::Float64)::Float64
    n = length(prices) - 1
    kn = max(2, round(Int, theta * sqrt(n)))

    log_prices = log.(max.(prices, 1e-10))

    # Pre-averaging weights: g(x) = min(x, 1-x) for x in [0,1]
    function g(x::Float64)::Float64
        return min(x, 1.0 - x)
    end

    # Pre-averaged returns
    y_bar = Vector{Float64}(undef, max(n - kn + 1, 0))
    for i in 1:length(y_bar)
        s = 0.0
        for j in 1:kn
            w = g(j / kn)
            if i + j <= n + 1
                s += w * (log_prices[i + j] - log_prices[i + j - 1])
            end
        end
        y_bar[i] = s
    end

    # Estimator
    psi1 = 1.0  # integral g(x)^2 dx for min(x,1-x)
    psi2 = 1.0 / 12.0  # integral g'(x)^2 dx

    pa_sum = sum(y_bar[i]^2 for i in 1:length(y_bar))

    # Bias correction
    noise_est = noise_variance_estimation(prices)
    bias = psi2 * kn * noise_est

    iv = (pa_sum / (psi1 * kn)) - bias
    return max(iv, 0.0)
end

"""
    flat_top_realized_kernel(prices; bandwidth=0)

Flat-top realized kernel: improved noise robustness.
"""
function flat_top_realized_kernel(prices::Vector{Float64}; bandwidth::Int=0)::Float64
    return realized_kernel(prices; kernel_type=:flat_top, bandwidth=bandwidth)
end

"""
    jump_variation_decomposition(prices)

Decompose total variation into continuous and jump components.
"""
function jump_variation_decomposition(prices::Vector{Float64})
    rv = realized_variance(prices)
    bpv = bipower_variation(prices)
    mrv = medRV(prices)

    # Jump variation estimates
    jump_bpv = max(rv - bpv, 0.0)
    jump_mrv = max(rv - mrv, 0.0)

    # Average estimate
    jump_avg = 0.5 * (jump_bpv + jump_mrv)
    continuous = rv - jump_avg

    return (total_variance=rv, continuous_variance=continuous,
            jump_variance=jump_avg, jump_fraction=jump_avg / max(rv, 1e-15),
            bpv_jump=jump_bpv, mrv_jump=jump_mrv)
end

"""
    intraday_leverage_effect(prices, block_size)

Leverage effect: correlation between returns and future volatility.
"""
function intraday_leverage_effect(prices::Vector{Float64}, block_size::Int)::Float64
    n = length(prices)
    log_prices = log.(max.(prices, 1e-10))
    returns = diff(log_prices)
    m = length(returns)

    num_blocks = m ÷ block_size
    if num_blocks < 3
        return 0.0
    end

    block_returns = Vector{Float64}(undef, num_blocks)
    block_rvs = Vector{Float64}(undef, num_blocks)

    for b in 1:num_blocks
        start_idx = (b - 1) * block_size + 1
        end_idx = b * block_size
        block_r = returns[start_idx:end_idx]
        block_returns[b] = sum(block_r)
        block_rvs[b] = sum(r^2 for r in block_r)
    end

    # Leverage = corr(r_t, rv_{t+1})
    if num_blocks < 3
        return 0.0
    end
    return cor(block_returns[1:end-1], block_rvs[2:end])
end

"""
    trade_duration_analysis(timestamps)

Analyze inter-trade durations.
"""
function trade_duration_analysis(timestamps::Vector{Float64})
    n = length(timestamps)
    if n < 2
        return (mean_duration=0.0, std_duration=0.0, overdispersion=0.0)
    end

    durations = diff(timestamps)
    durations = filter(d -> d > 0, durations)

    if isempty(durations)
        return (mean_duration=0.0, std_duration=0.0, overdispersion=0.0)
    end

    mu = mean(durations)
    sigma = std(durations)

    # Overdispersion: var/mean (>1 = clustered, <1 = regular)
    overdispersion = mu > 0 ? sigma^2 / mu : 0.0

    # Hazard rate (assuming exponential baseline)
    hazard = mu > 0 ? 1.0 / mu : 0.0

    # Autocorrelation of durations
    n_d = length(durations)
    if n_d > 2
        ac1 = cor(durations[1:end-1], durations[2:end])
    else
        ac1 = 0.0
    end

    return (mean_duration=mu, std_duration=sigma, overdispersion=overdispersion,
            hazard_rate=hazard, autocorrelation=ac1,
            num_trades=n, median_duration=median(durations))
end

"""
    realized_covariance_matrix(prices_matrix, timestamps; freq=300)

Multi-asset realized covariance matrix.
"""
function realized_covariance_matrix(prices_matrix::Matrix{Float64};
                                     freq::Int=1)::Matrix{Float64}
    T, N = size(prices_matrix)
    log_prices = log.(max.(prices_matrix, 1e-10))
    returns = diff(log_prices, dims=1)

    # Subsample
    sub_returns = returns[1:freq:end, :]
    m = size(sub_returns, 1)

    rcov = zeros(N, N)
    for i in 1:N
        for j in i:N
            for t in 1:m
                rcov[i, j] += sub_returns[t, i] * sub_returns[t, j]
            end
            rcov[j, i] = rcov[i, j]
        end
    end

    return rcov
end

"""
    tick_imbalance_bars(prices, volumes, directions, initial_theta)

Tick imbalance bars (Lopez de Prado 2018).
Form bars when cumulative tick imbalance exceeds threshold.
"""
function tick_imbalance_bars(prices::Vector{Float64}, volumes::Vector{Float64},
                              directions::Vector{Int}, initial_theta::Float64)
    n = length(prices)
    theta = initial_theta
    cum_imbalance = 0.0

    bar_indices = Int[1]
    bar_prices = Float64[prices[1]]
    bar_volumes = Float64[0.0]

    bar_vol_accum = 0.0

    for i in 2:n
        cum_imbalance += directions[i]
        bar_vol_accum += volumes[i]

        if abs(cum_imbalance) >= theta
            push!(bar_indices, i)
            push!(bar_prices, prices[i])
            push!(bar_volumes, bar_vol_accum)

            # Update theta using EWMA of bar sizes
            if length(bar_indices) > 2
                recent_sizes = diff(bar_indices[max(1, end-10):end])
                theta = mean(recent_sizes) * 0.5
                theta = max(theta, 1.0)
            end

            cum_imbalance = 0.0
            bar_vol_accum = 0.0
        end
    end

    return (indices=bar_indices, prices=bar_prices, volumes=bar_volumes)
end

"""
    volume_imbalance_bars(prices, volumes, directions, initial_theta)

Volume imbalance bars.
"""
function volume_imbalance_bars(prices::Vector{Float64}, volumes::Vector{Float64},
                                directions::Vector{Int}, initial_theta::Float64)
    n = length(prices)
    theta = initial_theta
    cum_vol_imbalance = 0.0

    bar_indices = Int[1]
    bar_prices = Float64[prices[1]]

    for i in 2:n
        signed_vol = directions[i] * volumes[i]
        cum_vol_imbalance += signed_vol

        if abs(cum_vol_imbalance) >= theta
            push!(bar_indices, i)
            push!(bar_prices, prices[i])

            # Update theta
            if length(bar_indices) > 5
                recent_sizes = diff(bar_indices[max(1, end-10):end])
                avg_size = mean(recent_sizes)
                avg_vol = mean(volumes[max(1, i-100):i])
                theta = avg_size * avg_vol * 0.5
                theta = max(theta, volumes[i])
            end

            cum_vol_imbalance = 0.0
        end
    end

    return (indices=bar_indices, prices=bar_prices)
end

"""
    dollar_bars(prices, volumes, dollar_threshold)

Dollar bars: sample when cumulative dollar volume exceeds threshold.
"""
function dollar_bars(prices::Vector{Float64}, volumes::Vector{Float64},
                      dollar_threshold::Float64)
    n = length(prices)
    cum_dollar = 0.0

    bar_indices = Int[1]
    bar_ohlc = Vector{NTuple{4, Float64}}()
    bar_open = prices[1]
    bar_high = prices[1]
    bar_low = prices[1]

    for i in 1:n
        dollar_vol = prices[i] * volumes[i]
        cum_dollar += dollar_vol
        bar_high = max(bar_high, prices[i])
        bar_low = min(bar_low, prices[i])

        if cum_dollar >= dollar_threshold
            push!(bar_indices, i)
            push!(bar_ohlc, (bar_open, bar_high, bar_low, prices[i]))

            cum_dollar = 0.0
            bar_open = prices[i]
            bar_high = prices[i]
            bar_low = prices[i]
        end
    end

    return (indices=bar_indices, ohlc=bar_ohlc)
end

"""
    entropy_of_order_flow(directions, window)

Shannon entropy of order flow direction distribution over rolling window.
Low entropy = one-sided flow (toxic), High entropy = balanced.
"""
function entropy_of_order_flow(directions::Vector{Int}, window::Int)::Vector{Float64}
    n = length(directions)
    if n < window
        return Float64[]
    end

    entropies = Vector{Float64}(undef, n - window + 1)

    for i in window:n
        sub = directions[(i-window+1):i]
        n_buy = count(x -> x > 0, sub)
        n_sell = count(x -> x < 0, sub)
        total = n_buy + n_sell
        if total == 0
            entropies[i - window + 1] = 0.0
            continue
        end

        p_buy = n_buy / total
        p_sell = n_sell / total

        H = 0.0
        if p_buy > 0
            H -= p_buy * log2(p_buy)
        end
        if p_sell > 0
            H -= p_sell * log2(p_sell)
        end

        entropies[i - window + 1] = H
    end

    return entropies
end

"""
    market_impact_model(trade_sizes, price_changes, participation_rates)

Estimate market impact model: dp = alpha * (v/V)^beta * sigma.
"""
function market_impact_model(trade_sizes::Vector{Float64},
                              price_changes::Vector{Float64},
                              participation_rates::Vector{Float64})
    n = min(length(trade_sizes), length(price_changes), length(participation_rates))

    # Log-linear model: log|dp| = log(alpha) + beta*log(v/V) + gamma*log(sigma) + eps
    # Simplified: log|dp| = a + b*log(participation) + eps
    y = [log(max(abs(price_changes[i]), 1e-15)) for i in 1:n]
    x = [log(max(participation_rates[i], 1e-15)) for i in 1:n]

    X = hcat(ones(n), x)
    beta = (X' * X) \ (X' * y)

    alpha = exp(beta[1])
    exponent = beta[2]

    residuals = y - X * beta
    r_sq = 1.0 - sum(residuals.^2) / max(sum((y .- mean(y)).^2), 1e-15)

    return (alpha=alpha, exponent=exponent, r_squared=r_sq)
end

"""
    hf_portfolio_risk(weights, realized_cov)

High-frequency portfolio risk from realized covariance.
"""
function hf_portfolio_risk(weights::Vector{Float64},
                            realized_cov::Matrix{Float64})
    port_var = dot(weights, realized_cov * weights)
    port_vol = sqrt(max(port_var, 0.0))

    # Marginal risk contribution
    N = length(weights)
    mrc = realized_cov * weights
    if port_vol > 1e-15
        mrc ./= port_vol
    end

    # Component risk
    component_risk = weights .* mrc
    total_risk = sum(component_risk)

    return (portfolio_variance=port_var, portfolio_vol=port_vol,
            marginal_risk=mrc, component_risk=component_risk,
            annualized_vol=port_vol * sqrt(252.0))
end

end # module HighFrequencyAnalytics
