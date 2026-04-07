module MarketMicrostructure

# Market microstructure models: Glosten-Milgrom spread decomposition,
# Kyle lambda, Amihud illiquidity, Corwin-Schultz spread, VPIN.
# For production use in the SRFM quant trading system.

using LinearAlgebra
using Statistics
using Random
using Test

export Trade, Quote
export GMResult, glosten_milgrom
export kyle_lambda
export amihud_illiquidity
export corwin_schultz
export vpin

# ---------------------------------------------------------------------------
# Core structs
# ---------------------------------------------------------------------------

"""
    Trade

Represents a single executed trade.

Fields:
- `price`: trade price
- `volume`: trade volume (absolute, positive)
- `side`: trade direction (+1 = buy, -1 = sell)
- `timestamp`: trade time (Float64 seconds or index)
"""
@kwdef struct Trade
    price::Float64
    volume::Float64
    side::Int  -- +1 buy, -1 sell
    timestamp::Float64
end

"""
    Quote

Represents a best bid/ask quote snapshot.

Fields:
- `bid`: best bid price
- `ask`: best ask price
- `bid_size`: quantity available at bid
- `ask_size`: quantity available at ask
- `timestamp`: quote time (Float64 seconds or index)
"""
@kwdef struct Quote
    bid::Float64
    ask::Float64
    bid_size::Float64
    ask_size::Float64
    timestamp::Float64
end

# ---------------------------------------------------------------------------
# Glosten-Milgrom Spread Decomposition
# ---------------------------------------------------------------------------

"""
    GMResult

Result of Glosten-Milgrom spread decomposition.

Fields:
- `adverse_selection_pct`: fraction of half-spread due to adverse selection
- `order_processing_pct`: fraction due to order processing costs
- `half_spread_mean`: mean quoted half-spread in price units
- `lambda`: adverse selection coefficient (price impact per trade)
- `n_trades`: number of trades used in estimation
"""
@kwdef struct GMResult
    adverse_selection_pct::Float64
    order_processing_pct::Float64
    half_spread_mean::Float64
    lambda::Float64
    n_trades::Int
end

"""
    glosten_milgrom(trades, quotes) -> GMResult

Estimate the Glosten-Milgrom (1985) adverse selection component of the bid-ask spread.

Uses the Huang-Stoll (1997) trade indicator regression:
    mid_{t+1} - mid_t = lambda * Q_t + eps_t
where Q_t = +1 for buyer-initiated, -1 for seller-initiated trade.

lambda captures the permanent price impact (adverse selection).
The adverse selection component as a fraction of the half-spread is: lambda / (half_spread).

# Arguments
- `trades`: vector of Trade structs
- `quotes`: vector of Quote structs (matched by nearest timestamp)

# Returns
GMResult struct.
"""
function glosten_milgrom(trades::Vector{Trade}, quotes::Vector{Quote})::GMResult
    n = min(length(trades), length(quotes) - 1)
    @assert n >= 10 "Need at least 10 paired observations"

    -- Compute midpoints from quotes
    mids = [(q.bid + q.ask) / 2.0 for q in quotes]
    spreads = [q.ask - q.bid for q in quotes]

    -- Build regression arrays
    n_obs = min(n, length(mids) - 1)
    Q = Float64[trades[t].side for t in 1:n_obs]
    delta_mid = Float64[mids[t+1] - mids[t] for t in 1:n_obs]

    -- OLS: delta_mid = lambda * Q + eps
    Q_mat = reshape(Q, :, 1)
    lambda_est = (dot(Q, delta_mid)) / max(dot(Q, Q), 1e-12)

    -- Residuals
    resid = delta_mid .- lambda_est .* Q
    sigma_eps = std(resid)

    mean_half_spread = mean(spreads[1:n_obs]) / 2.0
    adverse_pct = if mean_half_spread > 1e-12
        clamp(lambda_est / mean_half_spread, 0.0, 1.0)
    else
        0.0
    end

    return GMResult(
        adverse_selection_pct=adverse_pct,
        order_processing_pct=1.0 - adverse_pct,
        half_spread_mean=mean_half_spread,
        lambda=lambda_est,
        n_trades=n_obs
    )
end

# ---------------------------------------------------------------------------
# Kyle (1985) Lambda
# ---------------------------------------------------------------------------

"""
    kyle_lambda(prices, signed_volumes, window=100) -> Float64

Estimate Kyle's (1985) lambda: the price impact coefficient.

Regresses price changes on signed order flow:
    Delta_p_t = lambda * signed_flow_t + eps_t

Uses EWMA weighting with decay = 2 / (window + 1) for a rolling estimate.

# Arguments
- `prices`: vector of prices
- `signed_volumes`: vector of signed volume (positive = net buy, negative = net sell)
- `window`: lookback window for EWMA weighting (default 100)

# Returns
Scalar lambda estimate (price impact per unit signed flow).
"""
function kyle_lambda(prices::Vector{Float64},
                      signed_volumes::Vector{Float64},
                      window::Int=100)::Float64
    @assert length(prices) == length(signed_volumes) "prices and signed_volumes must be same length"
    n = length(prices)
    @assert n >= 3 "Need at least 3 observations"

    dp = diff(prices)
    sv = signed_volumes[2:n]
    m = length(dp)

    -- EWMA weights
    decay = 2.0 / (window + 1)
    weights = Float64[decay * (1 - decay)^(m - t) for t in 1:m]
    weights ./= sum(weights)

    -- Weighted OLS: dp = lambda * sv
    num = sum(weights .* sv .* dp)
    denom = sum(weights .* sv .^ 2)

    return denom > 1e-12 ? num / denom : 0.0
end

"""
    kyle_lambda_rolling(prices, signed_volumes, window=100) -> Vector{Float64}

Compute rolling Kyle lambda estimates.

# Returns
Vector of length n-window with rolling lambda values.
"""
function kyle_lambda_rolling(prices::Vector{Float64},
                               signed_volumes::Vector{Float64},
                               window::Int=100)::Vector{Float64}
    n = length(prices)
    @assert n > window "Series must be longer than window"

    result = Vector{Float64}(undef, n - window)
    for i in 1:(n - window)
        result[i] = kyle_lambda(prices[i:(i+window)], signed_volumes[i:(i+window)], window)
    end
    return result
end

# ---------------------------------------------------------------------------
# Amihud Illiquidity
# ---------------------------------------------------------------------------

"""
    amihud_illiquidity(returns, volumes, window=22) -> Vector{Float64}

Compute Amihud (2002) illiquidity measure: rolling mean of |r_t| / volume_t.

Higher values indicate greater price impact per unit of trading volume
(less liquid markets).

# Arguments
- `returns`: vector of returns
- `volumes`: vector of trading volumes (same length)
- `window`: rolling window length (default 22 trading days)

# Returns
Vector of rolling Amihud illiquidity ratios (length = n - window + 1).
"""
function amihud_illiquidity(returns::Vector{Float64},
                              volumes::Vector{Float64},
                              window::Int=22)::Vector{Float64}
    @assert length(returns) == length(volumes) "returns and volumes must be same length"
    n = length(returns)
    @assert n >= window "Series must be at least as long as window"

    illiq = abs.(returns) ./ max.(volumes, 1.0)  -- avoid division by zero

    result = Vector{Float64}(undef, n - window + 1)
    for i in 1:(n - window + 1)
        result[i] = mean(illiq[i:(i + window - 1)])
    end

    return result
end

# ---------------------------------------------------------------------------
# Corwin-Schultz Spread Estimator
# ---------------------------------------------------------------------------

"""
    corwin_schultz(highs, lows) -> Vector{Float64}

Estimate effective bid-ask spread from daily high-low price ranges using
the Corwin-Schultz (2012) estimator.

Uses the insight that the ratio of two-day to one-day high-low ranges
captures the spread, while adjusting for overnight volatility.

# Arguments
- `highs`: vector of daily high prices
- `lows`: vector of daily low prices

# Returns
Vector of estimated daily spreads (length = n - 1).
Negative estimates (occasionally occur due to rounding) are set to 0.
"""
function corwin_schultz(highs::Vector{Float64},
                          lows::Vector{Float64})::Vector{Float64}
    @assert length(highs) == length(lows) "highs and lows must be same length"
    n = length(highs)
    @assert n >= 2 "Need at least 2 observations"

    spreads = Vector{Float64}(undef, n - 1)

    for t in 1:(n-1)
        -- Single-day ranges
        h1 = highs[t]
        l1 = lows[t]
        h2 = highs[t+1]
        l2 = lows[t+1]

        -- Two-day range
        h2d = max(h1, h2)
        l2d = min(l1, l2)

        if l1 <= 0 || l2 <= 0 || l2d <= 0
            spreads[t] = 0.0
            continue
        end

        -- Log ratios
        beta1 = log(h1 / l1)^2
        beta2 = log(h2 / l2)^2
        beta = beta1 + beta2
        gamma = log(h2d / l2d)^2

        -- Corwin-Schultz formula
        alpha_num = sqrt(2.0 * beta) - sqrt(beta)
        alpha_denom = 3.0 - 2.0 * sqrt(2.0)
        alpha = alpha_num / max(alpha_denom, 1e-12) - sqrt(gamma / max(alpha_denom, 1e-12))

        spread = 2.0 * (exp(alpha) - 1.0) / (1.0 + exp(alpha))
        spreads[t] = max(spread, 0.0)
    end

    return spreads
end

# ---------------------------------------------------------------------------
# VPIN (Volume-Synchronized Probability of Informed Trading)
# ---------------------------------------------------------------------------

"""
    vpin(prices, volumes, n_buckets=50) -> Vector{Float64}

Compute Volume-Synchronized Probability of Informed Trading (VPIN).

Algorithm (Easley, Lopez de Prado, O'Hara 2012):
1. Divide total volume into equal-volume buckets of size V_bar = total_vol / n_buckets
2. For each bucket, classify fraction of buy volume using bulk volume classification:
   V_buy = V_bucket * CDF((p_end - p_start) / sigma_dp)
   V_sell = V_bucket - V_buy
3. VPIN = (1/n) * sum_i |V_buy_i - V_sell_i| / V_bar

# Arguments
- `prices`: vector of prices (tick or bar)
- `volumes`: vector of volumes (same length as prices)
- `n_buckets`: number of equal-volume buckets per VPIN sample (default 50)

# Returns
Vector of VPIN estimates. Length depends on data; at least one value returned
per full set of n_buckets complete buckets.
"""
function vpin(prices::Vector{Float64},
               volumes::Vector{Float64},
               n_buckets::Int=50)::Vector{Float64}
    @assert length(prices) == length(volumes) "prices and volumes must be same length"
    n = length(prices)
    @assert n >= n_buckets "Need at least n_buckets observations"

    total_vol = sum(volumes)
    V_bar = total_vol / n_buckets

    -- Price changes for bulk volume classification
    dp = diff(prices)
    sigma_dp = std(dp) > 0 ? std(dp) : 1e-8

    -- Fill buckets
    buy_vols = Float64[]
    sell_vols = Float64[]

    bucket_buy = 0.0
    bucket_sell = 0.0
    bucket_vol = 0.0
    bucket_start_price = prices[1]

    i = 1
    while i <= n
        vol_i = volumes[i]
        dp_i = i > 1 ? prices[i] - prices[i-1] : 0.0

        -- Bulk volume classification: fraction buy
        z = dp_i / sigma_dp
        frac_buy = _standard_normal_cdf(z)
        frac_sell = 1.0 - frac_buy

        remaining = vol_i

        while remaining > 1e-10
            space_in_bucket = V_bar - bucket_vol
            fill = min(remaining, space_in_bucket)

            bucket_buy += fill * frac_buy
            bucket_sell += fill * frac_sell
            bucket_vol += fill
            remaining -= fill

            if bucket_vol >= V_bar - 1e-10
                push!(buy_vols, bucket_buy)
                push!(sell_vols, bucket_sell)
                bucket_buy = 0.0
                bucket_sell = 0.0
                bucket_vol = 0.0
            end
        end
        i += 1
    end

    -- VPIN using rolling window of n_buckets buckets
    m = length(buy_vols)
    if m < n_buckets
        -- Not enough buckets; return single estimate if possible
        if m == 0
            return [0.0]
        end
        total_imbalance = sum(abs.(buy_vols .- sell_vols))
        return [total_imbalance / (m * V_bar)]
    end

    vpin_vals = Vector{Float64}(undef, m - n_buckets + 1)
    for t in 1:(m - n_buckets + 1)
        window_buy = buy_vols[t:(t + n_buckets - 1)]
        window_sell = sell_vols[t:(t + n_buckets - 1)]
        vpin_vals[t] = sum(abs.(window_buy .- window_sell)) / (n_buckets * V_bar)
    end

    return clamp.(vpin_vals, 0.0, 1.0)
end

"""
    _standard_normal_cdf(z) -> Float64

Approximate standard normal CDF using rational approximation.
Abramowitz and Stegun 26.2.17 (max error ~7.5e-8).
"""
function _standard_normal_cdf(z::Float64)::Float64
    if z > 6.0
        return 1.0
    elseif z < -6.0
        return 0.0
    end
    if z < 0.0
        return 1.0 - _standard_normal_cdf(-z)
    end
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 +
                t * (-0.356563782 +
                t * (1.781477937 +
                t * (-1.821255978 +
                t * 1.330274429))))
    phi = exp(-0.5 * z^2) / sqrt(2.0 * pi)
    return 1.0 - phi * poly
end

# ---------------------------------------------------------------------------
# Synthetic data generators for testing
# ---------------------------------------------------------------------------

"""
    _generate_trades(n, rng) -> Vector{Trade}

Generate synthetic trades with a midpoint random walk and random sides.
"""
function _generate_trades(n::Int, rng::AbstractRNG)::Vector{Trade}
    mid = 100.0
    trades = Vector{Trade}(undef, n)
    for i in 1:n
        mid += 0.05 * randn(rng)
        side = rand(rng) > 0.5 ? 1 : -1
        spread = 0.10
        price = mid + side * spread / 2.0
        vol = 100.0 + 900.0 * rand(rng)
        trades[i] = Trade(price=price, volume=vol, side=side, timestamp=Float64(i))
    end
    return trades
end

"""
    _generate_quotes(n, rng) -> Vector{Quote}

Generate synthetic quotes matching trade timestamps.
"""
function _generate_quotes(n::Int, rng::AbstractRNG)::Vector{Quote}
    mid = 100.0
    quotes = Vector{Quote}(undef, n)
    for i in 1:n
        mid += 0.05 * randn(rng)
        spread = 0.10 + 0.05 * abs(randn(rng))
        quotes[i] = Quote(
            bid=mid - spread/2,
            ask=mid + spread/2,
            bid_size=500.0 + 500.0 * rand(rng),
            ask_size=500.0 + 500.0 * rand(rng),
            timestamp=Float64(i)
        )
    end
    return quotes
end

# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

function run_tests()
    @testset "MarketMicrostructure Tests" begin

        rng = MersenneTwister(55)

        -- -- Struct construction --
        @testset "Trade struct" begin
            t = Trade(price=100.0, volume=500.0, side=1, timestamp=1.0)
            @test t.price == 100.0
            @test t.volume == 500.0
            @test t.side == 1
            @test t.timestamp == 1.0
        end

        @testset "Quote struct" begin
            q = Quote(bid=99.90, ask=100.10, bid_size=1000.0, ask_size=800.0, timestamp=1.0)
            @test q.ask > q.bid
            @test q.bid_size > 0.0
        end

        -- -- Glosten-Milgrom --
        @testset "glosten_milgrom basic" begin
            trades = _generate_trades(200, rng)
            quotes = _generate_quotes(201, rng)
            result = glosten_milgrom(trades, quotes)
            @test isa(result, GMResult)
            @test result.n_trades > 0
            @test result.half_spread_mean > 0.0
            @test 0.0 <= result.adverse_selection_pct <= 1.0
            @test isapprox(result.adverse_selection_pct + result.order_processing_pct, 1.0, atol=1e-10)
        end

        @testset "glosten_milgrom lambda_positive_when_buy_impact" begin
            -- Simulate: buy trades push price up
            n = 100
            trades2 = Vector{Trade}(undef, n)
            quotes2 = Vector{Quote}(undef, n + 1)
            mid = 100.0
            quotes2[1] = Quote(bid=mid-0.05, ask=mid+0.05, bid_size=1000.0, ask_size=1000.0, timestamp=0.0)
            for i in 1:n
                side = i % 2 == 0 ? 1 : -1
                mid += 0.01 * side  -- price moves with order direction
                quotes2[i+1] = Quote(bid=mid-0.05, ask=mid+0.05, bid_size=1000.0, ask_size=1000.0, timestamp=Float64(i))
                trades2[i] = Trade(price=mid + side*0.05, volume=100.0, side=side, timestamp=Float64(i))
            end
            result = glosten_milgrom(trades2, quotes2)
            @test result.lambda > 0.0
        end

        @testset "glosten_milgrom components_sum_to_one" begin
            trades = _generate_trades(150, rng)
            quotes = _generate_quotes(151, rng)
            result = glosten_milgrom(trades, quotes)
            @test isapprox(result.adverse_selection_pct + result.order_processing_pct, 1.0, atol=1e-10)
        end

        -- -- Kyle Lambda --
        @testset "kyle_lambda basic" begin
            n = 200
            prices = cumsum(randn(rng, n) .* 0.1) .+ 100.0
            sv = randn(rng, n) .* 1000.0
            lam = kyle_lambda(prices, sv, 50)
            @test isfinite(lam)
        end

        @testset "kyle_lambda positive_impact" begin
            -- Simulate: signed volume positively predicts price change
            n = 300
            prices = zeros(n)
            prices[1] = 100.0
            sv = randn(rng, n) .* 1000.0
            for t in 2:n
                prices[t] = prices[t-1] + 0.001 * sv[t] + 0.05 * randn(rng)
            end
            lam = kyle_lambda(prices, sv, 100)
            @test lam > 0.0
        end

        @testset "kyle_lambda_rolling length" begin
            n = 250
            prices = cumsum(randn(rng, n) .* 0.1) .+ 100.0
            sv = randn(rng, n) .* 1000.0
            rolling = kyle_lambda_rolling(prices, sv, 50)
            @test length(rolling) == n - 50
        end

        @testset "kyle_lambda zero_flow" begin
            n = 100
            prices = 100.0 .* ones(n)
            sv = zeros(n)
            lam = kyle_lambda(prices, sv, 50)
            @test lam == 0.0
        end

        -- -- Amihud Illiquidity --
        @testset "amihud_illiquidity basic" begin
            n = 100
            rets = randn(rng, n) .* 0.01
            vols = 1e6 .+ 1e5 .* randn(rng, n)
            vols = abs.(vols)
            illiq = amihud_illiquidity(rets, vols, 22)
            @test length(illiq) == n - 22 + 1
            @test all(illiq .>= 0.0)
            @test all(isfinite.(illiq))
        end

        @testset "amihud_illiquidity high_vol_high_illiq" begin
            n = 100
            -- High volatility, low volume -> high illiquidity
            rets_high = randn(rng, n) .* 0.05
            vols_low = 1e3 .* ones(n)
            -- Low volatility, high volume -> low illiquidity
            rets_low = randn(rng, n) .* 0.001
            vols_high = 1e8 .* ones(n)

            illiq_high = amihud_illiquidity(rets_high, vols_low, 22)
            illiq_low = amihud_illiquidity(rets_low, vols_high, 22)
            @test mean(illiq_high) > mean(illiq_low)
        end

        @testset "amihud_illiquidity window_22" begin
            n = 60
            rets = randn(rng, n) .* 0.01
            vols = abs.(randn(rng, n)) .* 1e6 .+ 1e6
            illiq = amihud_illiquidity(rets, vols, 22)
            @test length(illiq) == n - 22 + 1
        end

        -- -- Corwin-Schultz --
        @testset "corwin_schultz basic" begin
            n = 100
            prices = 100.0 .+ cumsum(randn(rng, n) .* 0.5)
            highs = prices .+ abs.(randn(rng, n) .* 0.3)
            lows = prices .- abs.(randn(rng, n) .* 0.3)

            spreads = corwin_schultz(highs, lows)
            @test length(spreads) == n - 1
            @test all(spreads .>= 0.0)
            @test all(isfinite.(spreads))
        end

        @testset "corwin_schultz wider_spread_with_wider_range" begin
            n = 100
            base = 100.0
            -- Narrow ranges -> small spread estimates
            highs_narrow = fill(base + 0.1, n)
            lows_narrow = fill(base - 0.1, n)
            -- Wide ranges -> larger spread estimates
            highs_wide = fill(base + 1.0, n)
            lows_wide = fill(base - 1.0, n)

            sp_narrow = corwin_schultz(highs_narrow, lows_narrow)
            sp_wide = corwin_schultz(highs_wide, lows_wide)
            @test mean(sp_wide) >= mean(sp_narrow) - 1e-8
        end

        @testset "corwin_schultz non_negative" begin
            n = 50
            mid = 100.0 .+ cumsum(randn(rng, n))
            h = mid .+ abs.(randn(rng, n))
            l = mid .- abs.(randn(rng, n))
            sp = corwin_schultz(h, l)
            @test all(sp .>= 0.0)
        end

        -- -- VPIN --
        @testset "vpin basic" begin
            n = 1000
            prices2 = 100.0 .+ cumsum(randn(rng, n) .* 0.1)
            volumes2 = abs.(randn(rng, n)) .* 1000.0 .+ 500.0

            vpin_vals = vpin(prices2, volumes2, 50)
            @test length(vpin_vals) >= 1
            @test all(0.0 .<= vpin_vals .<= 1.0)
            @test all(isfinite.(vpin_vals))
        end

        @testset "vpin range" begin
            n = 500
            prices2 = 100.0 .+ cumsum(randn(rng, n) .* 0.05)
            volumes2 = ones(n) .* 1000.0
            vpin_vals = vpin(prices2, volumes2, 20)
            @test all(vpin_vals .>= 0.0)
            @test all(vpin_vals .<= 1.0)
        end

        @testset "vpin high_with_one_sided_flow" begin
            -- One-sided flow should produce high VPIN
            n = 1000
            -- Prices always rising (all buys)
            prices_up = 100.0 .+ cumsum(ones(n) .* 0.1)
            vols_uniform = ones(n) .* 1000.0
            vpin_up = vpin(prices_up, vols_uniform, 20)
            @test mean(vpin_up) > 0.3  -- should be elevated
        end

        @testset "_standard_normal_cdf" begin
            @test isapprox(_standard_normal_cdf(0.0), 0.5, atol=1e-5)
            @test isapprox(_standard_normal_cdf(1.96), 0.975, atol=1e-3)
            @test isapprox(_standard_normal_cdf(-1.96), 0.025, atol=1e-3)
            @test _standard_normal_cdf(7.0) == 1.0
            @test _standard_normal_cdf(-7.0) == 0.0
        end

        @testset "vpin with_few_buckets" begin
            n = 200
            prices2 = 100.0 .+ cumsum(randn(rng, n) .* 0.1)
            vols2 = ones(n) .* 100.0
            vpin_vals = vpin(prices2, vols2, 10)
            @test length(vpin_vals) >= 1
        end

        @testset "kyle_lambda_rolling correctness" begin
            n = 200
            prices = 100.0 .+ cumsum(randn(rng, n) .* 0.1)
            sv = randn(rng, n) .* 1000.0
            window = 50
            rolling = kyle_lambda_rolling(prices, sv, window)
            -- Verify first element matches direct call
            lam_direct = kyle_lambda(prices[1:(1+window)], sv[1:(1+window)], window)
            @test isapprox(rolling[1], lam_direct, atol=1e-10)
        end

        @testset "amihud_illiquidity zero_returns" begin
            n = 50
            rets = zeros(n)
            vols = ones(n) .* 1e6
            illiq = amihud_illiquidity(rets, vols, 10)
            @test all(illiq .== 0.0)
        end

    end
end

end # module MarketMicrostructure
