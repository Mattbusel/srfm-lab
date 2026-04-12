"""
integration_chronos.jl — Bridge to Chronos LOB simulator output

Implements:
  1. CSV reader for Chronos order-book snapshots
  2. LOB microstructure feature extraction:
     - Mid-price, bid-ask spread, depth imbalance
     - Order flow imbalance (OFI)
     - Queue imbalance, queue depth
     - Trade arrival rates, trade sign sequences
  3. Mid-price SDE calibration from LOB data
  4. Market impact modelling (Almgren-Chriss, Kyle lambda)
  5. Hawkes process parameter estimation for order arrivals
  6. Integration with SDE calibration pipeline

Chronos CSV format (assumed):
  timestamp, bid_price, ask_price, bid_size, ask_size,
  [level_2_bid, level_2_ask, ...],
  [trade_price, trade_size, trade_side]

References:
  - Cont, Kukanov & Stoikov (2013) "The price impact of order book events"
  - Kyle (1985) "Continuous auctions and insider trading"
  - Almgren & Chriss (2001) "Optimal execution of portfolio transactions"
  - Hawkes (1971) "Spectra of some self-exciting and mutually exciting point processes"
"""

using Statistics
using LinearAlgebra
using Random
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA TYPES FOR CHRONOS OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

"""
    LOBSnapshot

Single limit order book snapshot from Chronos.
"""
struct LOBSnapshot
    timestamp  :: Float64
    bid_prices :: Vector{Float64}   # levels 1..L
    ask_prices :: Vector{Float64}
    bid_sizes  :: Vector{Float64}
    ask_sizes  :: Vector{Float64}
end

"""
    TradeRecord

A single trade execution record from Chronos.
"""
struct TradeRecord
    timestamp  :: Float64
    price      :: Float64
    size       :: Float64
    side       :: Int8     # +1 buyer-initiated, -1 seller-initiated
end

"""
    ChronosData

Container for a complete Chronos simulation output session.
"""
struct ChronosData
    snapshots   :: Vector{LOBSnapshot}
    trades      :: Vector{TradeRecord}
    symbol      :: String
    session_dt  :: Float64   # calendar time per snapshot (seconds)
    n_levels    :: Int
end

Base.length(d::ChronosData) = length(d.snapshots)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CSV READER
# ─────────────────────────────────────────────────────────────────────────────

"""
    parse_chronos_csv_line(line, n_levels) → LOBSnapshot

Parse a single Chronos CSV row into a LOBSnapshot.
Expected format: timestamp, b1, b2...bL, a1, a2...aL, bs1...bsL, as1...asL
"""
function parse_chronos_csv_line(line::String, n_levels::Int)
    fields = split(strip(line), ',')
    length(fields) < 1 + 4 * n_levels && return nothing

    ts  = parse(Float64, fields[1])
    bid_p = [parse(Float64, fields[1 + k])              for k in 1:n_levels]
    ask_p = [parse(Float64, fields[1 + n_levels + k])   for k in 1:n_levels]
    bid_s = [parse(Float64, fields[1 + 2*n_levels + k]) for k in 1:n_levels]
    ask_s = [parse(Float64, fields[1 + 3*n_levels + k]) for k in 1:n_levels]

    return LOBSnapshot(ts, bid_p, ask_p, bid_s, ask_s)
end

"""
    parse_trade_csv_line(line) → TradeRecord

Parse a trade record CSV row.
Expected format: timestamp, price, size, side
"""
function parse_trade_csv_line(line::String)
    fields = split(strip(line), ',')
    length(fields) < 4 && return nothing
    ts    = parse(Float64, fields[1])
    price = parse(Float64, fields[2])
    size  = parse(Float64, fields[3])
    side  = Int8(parse(Int, fields[4]))
    return TradeRecord(ts, price, size, side)
end

"""
    read_chronos_csv(lob_path, trades_path; n_levels=5, max_rows=nothing)
        → ChronosData

Read Chronos CSV output files.

- `lob_path`    : path to LOB snapshot CSV
- `trades_path` : path to trades CSV (can be same file with different prefix)
- `n_levels`    : number of price levels per side
- `max_rows`    : optional row limit (for testing)
"""
function read_chronos_csv(lob_path::String,
                           trades_path::String;
                           n_levels::Int           = 5,
                           max_rows::Union{Nothing, Int} = nothing,
                           symbol::String          = "UNKNOWN",
                           session_dt::Real        = 0.001)
    snapshots = LOBSnapshot[]
    trades    = TradeRecord[]

    # Read LOB snapshots
    open(lob_path, "r") do f
        header = readline(f)  # skip header
        row = 0
        for line in eachline(f)
            isempty(strip(line)) && continue
            snap = parse_chronos_csv_line(line, n_levels)
            !isnothing(snap) && push!(snapshots, snap)
            row += 1
            !isnothing(max_rows) && row >= max_rows && break
        end
    end

    # Read trade records (if separate file)
    if isfile(trades_path)
        open(trades_path, "r") do f
            header = readline(f)
            for line in eachline(f)
                isempty(strip(line)) && continue
                tr = parse_trade_csv_line(line)
                !isnothing(tr) && push!(trades, tr)
            end
        end
    end

    return ChronosData(snapshots, trades, symbol, Float64(session_dt), n_levels)
end

"""
    synthetic_chronos_data(n_steps; seed=42, n_levels=5) → ChronosData

Generate synthetic Chronos-format data for testing.
"""
function synthetic_chronos_data(n_steps::Int;
                                 seed::Int     = 42,
                                 n_levels::Int = 5,
                                 dt::Float64   = 0.001,
                                 S0::Float64   = 100.0,
                                 σ::Float64    = 0.001,
                                 spread0::Float64 = 0.01)
    rng = MersenneTwister(seed)
    snapshots = LOBSnapshot[]
    trades    = TradeRecord[]

    mid = S0
    for t in 1:n_steps
        # Evolve mid-price
        mid += σ * randn(rng) * sqrt(dt)
        half_spread = spread0 * (1 + 0.5 * abs(randn(rng)))

        bid1 = mid - half_spread
        ask1 = mid + half_spread

        bid_p = [bid1 - (k-1) * 0.01 for k in 1:n_levels]
        ask_p = [ask1 + (k-1) * 0.01 for k in 1:n_levels]
        bid_s = abs.(randn(rng, n_levels)) .* 100 .+ 50
        ask_s = abs.(randn(rng, n_levels)) .* 100 .+ 50

        push!(snapshots, LOBSnapshot(Float64(t) * dt, bid_p, ask_p, bid_s, ask_s))

        # Random trade
        if rand(rng) < 0.3
            side  = Int8(rand(rng) < 0.5 ? 1 : -1)
            tsize = abs(randn(rng)) * 50 + 10
            tprice = side == 1 ? ask1 : bid1
            push!(trades, TradeRecord(Float64(t) * dt, tprice, tsize, side))
        end
    end
    return ChronosData(snapshots, trades, "SYN", dt, n_levels)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: MICROSTRUCTURE FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

"""
    mid_price(snap::LOBSnapshot) → Float64

Best bid-ask midpoint.
"""
mid_price(snap::LOBSnapshot) = 0.5 * (snap.bid_prices[1] + snap.ask_prices[1])

"""
    bid_ask_spread(snap::LOBSnapshot) → Float64

Best bid-ask spread.
"""
bid_ask_spread(snap::LOBSnapshot) = snap.ask_prices[1] - snap.bid_prices[1]

"""
    depth_imbalance(snap::LOBSnapshot; n_levels=1) → Float64

Order book depth imbalance at top n_levels:
I = (bid_vol - ask_vol) / (bid_vol + ask_vol)
∈ [-1, 1], positive = bid-heavy.
"""
function depth_imbalance(snap::LOBSnapshot; n_levels::Int=1)
    bid_v = sum(snap.bid_sizes[1:min(n_levels, length(snap.bid_sizes))])
    ask_v = sum(snap.ask_sizes[1:min(n_levels, length(snap.ask_sizes))])
    denom = bid_v + ask_v
    return denom > 0 ? (bid_v - ask_v) / denom : 0.0
end

"""
    order_flow_imbalance(snap_prev::LOBSnapshot, snap_curr::LOBSnapshot) → Float64

Cont-Kukanov-Stoikov (2013) order flow imbalance:
OFI = ΔBid_vol_at_best - ΔAsk_vol_at_best
"""
function order_flow_imbalance(prev::LOBSnapshot, curr::LOBSnapshot)
    # Best bid changes
    if curr.bid_prices[1] > prev.bid_prices[1]
        δ_bid = curr.bid_sizes[1]
    elseif curr.bid_prices[1] == prev.bid_prices[1]
        δ_bid = curr.bid_sizes[1] - prev.bid_sizes[1]
    else
        δ_bid = -prev.bid_sizes[1]
    end

    # Best ask changes
    if curr.ask_prices[1] < prev.ask_prices[1]
        δ_ask = curr.ask_sizes[1]
    elseif curr.ask_prices[1] == prev.ask_prices[1]
        δ_ask = curr.ask_sizes[1] - prev.ask_sizes[1]
    else
        δ_ask = -prev.ask_sizes[1]
    end

    return δ_bid - δ_ask
end

"""
    extract_lob_features(data::ChronosData) → NamedTuple

Extract a comprehensive feature set from Chronos LOB data.

Returns named tuple with:
  - `timestamps`   : time vector
  - `mid_prices`   : mid-price series
  - `log_returns`  : log mid-price returns
  - `spreads`      : bid-ask spreads
  - `depth_imb`    : top-level depth imbalance
  - `ofi`          : order flow imbalance
  - `bid_depths`   : total bid depth (all levels)
  - `ask_depths`   : total ask depth
  - `trade_signs`  : signed trade flow (aggregated per snapshot)
  - `realized_var` : realised variance (rolling window)
"""
function extract_lob_features(data::ChronosData;
                               window::Int = 100)
    n_snap = length(data.snapshots)
    n_snap < 2 && error("Need at least 2 snapshots")

    ts    = [snap.timestamp for snap in data.snapshots]
    mids  = [mid_price(snap) for snap in data.snapshots]
    rets  = log.(mids[2:end] ./ mids[1:end-1])
    rets  = vcat(0.0, rets)

    spreads = [bid_ask_spread(snap) for snap in data.snapshots]
    dimb    = [depth_imbalance(snap) for snap in data.snapshots]

    ofi = zeros(n_snap)
    for t in 2:n_snap
        ofi[t] = order_flow_imbalance(data.snapshots[t-1], data.snapshots[t])
    end

    bid_depths = [sum(snap.bid_sizes) for snap in data.snapshots]
    ask_depths = [sum(snap.ask_sizes) for snap in data.snapshots]

    # Aggregate trade signs per snapshot interval
    trade_signs = zeros(n_snap)
    if !isempty(data.trades)
        for tr in data.trades
            idx = searchsortedfirst(ts, tr.timestamp)
            idx = clamp(idx, 1, n_snap)
            trade_signs[idx] += tr.side * tr.size
        end
    end

    # Realised variance (rolling)
    rv = zeros(n_snap)
    for t in window:n_snap
        rv[t] = sum(rets[t-window+1:t].^2)
    end

    return (
        timestamps  = ts,
        mid_prices  = mids,
        log_returns = rets,
        spreads     = spreads,
        depth_imb   = dimb,
        ofi         = ofi,
        bid_depths  = bid_depths,
        ask_depths  = ask_depths,
        trade_signs = trade_signs,
        realized_var = rv,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: KYLE LAMBDA (MARKET IMPACT)
# ─────────────────────────────────────────────────────────────────────────────

"""
    KyleLambdaResult

Result of Kyle lambda regression.
Δp_t = λ × OFI_t + ε_t
"""
struct KyleLambdaResult
    λ        :: Float64   # price impact coefficient
    α        :: Float64   # intercept
    r_squared :: Float64
    std_err  :: Float64   # standard error of λ
    t_stat   :: Float64
end

"""
    estimate_kyle_lambda(mid_returns, ofi; window=nothing) → KyleLambdaResult

Estimate Kyle (1985) λ via OLS regression of price changes on order flow.
"""
function estimate_kyle_lambda(mid_returns::AbstractVector,
                               ofi::AbstractVector;
                               window::Union{Nothing, Int} = nothing)
    @assert length(mid_returns) == length(ofi)
    if !isnothing(window)
        # Use last `window` observations
        n = length(mid_returns)
        mid_returns = mid_returns[max(1, n-window+1):n]
        ofi         = ofi[max(1, n-window+1):n]
    end

    # Remove NaN
    valid = .!isnan.(mid_returns) .& .!isnan.(ofi)
    y = mid_returns[valid]
    x = ofi[valid]
    n = length(y)
    n < 3 && return KyleLambdaResult(NaN, NaN, NaN, NaN, NaN)

    # OLS
    X  = hcat(ones(n), x)
    XtX = X' * X
    Xty = X' * y
    β   = XtX \ Xty

    α, λ = β[1], β[2]
    ŷ    = X * β
    res  = y .- ŷ
    σ2   = sum(res.^2) / (n - 2)
    var_β = σ2 * inv(XtX)

    se_λ  = sqrt(max(var_β[2,2], 0.0))
    t_st  = abs(se_λ) > 0 ? λ / se_λ : 0.0
    ss_tot = sum((y .- mean(y)).^2)
    ss_res = sum(res.^2)
    r2    = 1.0 - ss_res / max(ss_tot, 1e-12)

    return KyleLambdaResult(λ, α, r2, se_λ, t_st)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: HAWKES PROCESS FOR ORDER ARRIVALS
# ─────────────────────────────────────────────────────────────────────────────

"""
    HawkesParams

Parameters of univariate Hawkes process:
  λ(t) = μ + Σ_{t_i < t} α exp(-β (t - t_i))

where μ = baseline intensity, α = excitation amplitude, β = decay rate.
"""
struct HawkesParams
    μ :: Float64
    α :: Float64
    β :: Float64
end

"""
    hawkes_log_likelihood(events, T_end, p::HawkesParams) → Float64

Log-likelihood of Hawkes process via standard formula.
"""
function hawkes_log_likelihood(events::AbstractVector,
                                T_end::Real,
                                p::HawkesParams)
    μ, α, β = p.μ, p.α, p.β
    n = length(events)
    n == 0 && return -μ * T_end

    # Sort events
    t_sorted = sort(events)

    # Compensator integral: Λ(T) = μ T + α/β Σ_i (1 - exp(-β(T-t_i)))
    Λ = μ * T_end + α / β * sum(1.0 - exp(-β * (T_end - t_i)) for t_i in t_sorted)

    # Log-intensity at each event
    ll = -Λ
    A  = 0.0  # running sum: Σ_{t_j < t_i} exp(-β(t_i - t_j))
    for i in 1:n
        if i > 1
            A = exp(-β * (t_sorted[i] - t_sorted[i-1])) * (1 + A)
        end
        λ_t = μ + α * A
        λ_t < 1e-10 && (λ_t = 1e-10)
        ll += log(λ_t)
    end
    return ll
end

"""
    calibrate_hawkes(event_times, T_end; n_restarts=5, seed=42) → HawkesParams

Calibrate Hawkes process parameters via MLE.
"""
function calibrate_hawkes(event_times::AbstractVector,
                           T_end::Real;
                           n_restarts::Int = 5,
                           seed::Int       = 42)
    rng = MersenneTwister(seed)
    events = sort(collect(Float64, event_times))

    best_ll  = -Inf
    best_p   = HawkesParams(1.0, 0.5, 1.0)

    for restart in 1:n_restarts
        μ0 = rand(rng) * 2 + 0.1
        α0 = rand(rng) * 0.8 + 0.01
        β0 = rand(rng) * 3.0 + 0.1

        function neg_ll(x)
            μ_ = max(x[1], 1e-4)
            α_ = max(x[2], 1e-4)
            β_ = max(x[3], 1e-4)
            α_ >= β_ && return 1e8   # stationarity: α/β < 1
            return -hawkes_log_likelihood(events, T_end, HawkesParams(μ_, α_, β_))
        end

        try
            res = optimize(neg_ll, [μ0, α0, β0],
                           NelderMead(),
                           Optim.Options(iterations=3000, g_tol=1e-8))
            ll = -Optim.minimum(res)
            if ll > best_ll
                x = Optim.minimizer(res)
                best_p  = HawkesParams(max(x[1],1e-4), max(x[2],1e-4), max(x[3],1e-4))
                best_ll = ll
            end
        catch; continue; end
    end
    return best_p
end

"""
    hawkes_intensity(t, events_before_t, p::HawkesParams) → Float64

Compute instantaneous Hawkes intensity at time t.
"""
function hawkes_intensity(t::Real, events_before_t::AbstractVector,
                           p::HawkesParams)
    λ = p.μ
    for t_j in events_before_t
        t_j >= t && break
        λ += p.α * exp(-p.β * (t - t_j))
    end
    return λ
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ALMGREN-CHRISS MARKET IMPACT
# ─────────────────────────────────────────────────────────────────────────────

"""
    AlmgrenChrissParams

Parameters for Almgren-Chriss (2001) optimal execution model.

Fields:
  - η   : temporary price impact (linear, η > 0)
  - γ   : permanent price impact (linear, γ > 0)
  - σ   : volatility of mid-price
  - τ   : trading horizon
  - λ   : risk-aversion parameter
"""
struct AlmgrenChrissParams
    η :: Float64
    γ :: Float64
    σ :: Float64
    τ :: Float64
    λ :: Float64
end

"""
    optimal_twap(X0, T, n_slices) → Vector{Float64}

Time-Weighted Average Price (TWAP) execution schedule:
equal slices over T, as baseline.
"""
optimal_twap(X0::Real, T::Real, n_slices::Int) =
    fill(X0 / n_slices, n_slices)

"""
    almgren_chriss_schedule(X0, p::AlmgrenChrissParams, n_slices) → Vector{Float64}

Optimal Almgren-Chriss execution schedule minimising mean-variance cost.

x_j = X0 × sinh(κ (n-j) Δt) / sinh(κ n Δt)  where κ = √(λ σ² / η)
"""
function almgren_chriss_schedule(X0::Real,
                                  p::AlmgrenChrissParams,
                                  n_slices::Int)
    Δt = p.τ / n_slices
    κ  = sqrt(p.λ * p.σ^2 / (p.η + 1e-10))
    holdings = zeros(n_slices + 1)
    holdings[1] = X0
    for j in 1:n_slices
        t_j = j * Δt
        holdings[j+1] = X0 * sinh(κ * (p.τ - t_j)) / max(sinh(κ * p.τ), 1e-10)
    end
    return max.(holdings, 0.0)
end

"""
    execution_cost(schedule, p::AlmgrenChrissParams) → (mean_cost, risk)

Compute expected execution cost and risk under Almgren-Chriss model.
"""
function execution_cost(schedule::AbstractVector,
                         p::AlmgrenChrissParams)
    n   = length(schedule) - 1
    Δt  = p.τ / n
    trades = diff(schedule) * (-1)   # positive = sell

    mean_cost = 0.0
    var_cost  = 0.0
    for j in 1:n
        xj   = schedule[j]
        tj   = trades[j]
        # Permanent impact cost
        mean_cost += p.γ * xj * tj
        # Temporary impact cost
        mean_cost += p.η * (tj / Δt)^2 * Δt
        # Variance contribution: σ² x_j² Δt
        var_cost  += p.σ^2 * xj^2 * Δt
    end
    return mean_cost, sqrt(max(var_cost, 0.0))
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: PIPELINE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    ChronosFeatureSet

Comprehensive feature set extracted from Chronos LOB data,
ready for SDE calibration.
"""
struct ChronosFeatureSet
    timestamps    :: Vector{Float64}
    mid_prices    :: Vector{Float64}
    log_returns   :: Vector{Float64}
    realized_vol  :: Vector{Float64}   # annualised
    spreads       :: Vector{Float64}
    depth_imb     :: Vector{Float64}
    ofi           :: Vector{Float64}
    trade_signs   :: Vector{Float64}
    kyle_lambda   :: Float64           # single value per session
    hawkes_params :: Union{Nothing, HawkesParams}
    n_snapshots   :: Int
    n_trades      :: Int
end

"""
    extract_chronos_features(data::ChronosData; annualise=252*8*3600) → ChronosFeatureSet

Full feature extraction pipeline for Chronos LOB data.
`annualise` converts per-second variance to annualised vol (252 days × 8h × 3600s).
"""
function extract_chronos_features(data::ChronosData;
                                   annualise::Real  = 252.0 * 8.0 * 3600.0,
                                   vol_window::Int  = 100,
                                   hawkes_fit::Bool = true)
    feats = extract_lob_features(data; window=vol_window)

    # Annualised realised vol from variance per tick-interval
    dt_sec     = data.session_dt
    rv_ann     = sqrt.(feats.realized_var .* annualise / vol_window)

    # Kyle lambda
    kl = estimate_kyle_lambda(feats.log_returns, feats.ofi;
                               window=min(vol_window * 10, length(feats.log_returns)))

    # Hawkes for trade arrivals
    hawkes_p = nothing
    if hawkes_fit && !isempty(data.trades)
        t_events = [tr.timestamp for tr in data.trades]
        T_end    = feats.timestamps[end]
        try
            hawkes_p = calibrate_hawkes(t_events, T_end; n_restarts=3)
        catch
            hawkes_p = nothing
        end
    end

    return ChronosFeatureSet(
        feats.timestamps,
        feats.mid_prices,
        feats.log_returns,
        rv_ann,
        feats.spreads,
        feats.depth_imb,
        feats.ofi,
        feats.trade_signs,
        isnan(kl.λ) ? 0.0 : kl.λ,
        hawkes_p,
        length(data.snapshots),
        length(data.trades),
    )
end

"""
    chronos_to_sde_inputs(cfs::ChronosFeatureSet) → NamedTuple

Convert Chronos feature set into inputs for SDE calibration.

Returns:
  - `prices`  : mid-price series
  - `returns` : log-return series
  - `vol_est` : realised volatility series (proxy for V_t in Heston)
  - `r`       : drift proxy (annualised log-return mean)
  - `sigma0`  : initial vol estimate
"""
function chronos_to_sde_inputs(cfs::ChronosFeatureSet)
    valid     = .!isnan.(cfs.log_returns) .& .!isnan.(cfs.realized_vol)
    rets      = cfs.log_returns[valid]
    vols      = cfs.realized_vol[valid]

    μ_ann     = mean(rets) / mean(diff(cfs.timestamps[valid])) * (1.0)
    σ0        = quantile(vols[vols .> 0], 0.5)   # median vol

    return (
        prices   = cfs.mid_prices,
        returns  = rets,
        vol_est  = vols,
        r        = μ_ann,
        sigma0   = σ0,
        spread   = mean(cfs.spreads),
        kyle_λ   = cfs.kyle_lambda,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: DEMO
# ─────────────────────────────────────────────────────────────────────────────

"""
    demo_chronos_integration(; n_steps=2000, seed=1)

Demo: generate synthetic Chronos data and extract features.
"""
function demo_chronos_integration(; n_steps::Int=2000, seed::Int=1)
    @info "Generating synthetic Chronos LOB data (n=$(n_steps))..."
    data = synthetic_chronos_data(n_steps; seed=seed, n_levels=5)

    @info "Extracting microstructure features..."
    cfs = extract_chronos_features(data; hawkes_fit=true)

    @printf "─────────────────────────────────────────────────\n"
    @printf "  Chronos Integration Summary\n"
    @printf "─────────────────────────────────────────────────\n"
    @printf "  Snapshots   : %d\n"       cfs.n_snapshots
    @printf "  Trades      : %d\n"       cfs.n_trades
    @printf "  Mid-price   : %.4f\n"     cfs.mid_prices[end]
    @printf "  Mean spread : %.6f\n"     mean(cfs.spreads)
    @printf "  Mean depth imb: %.4f\n"   mean(cfs.depth_imb)
    @printf "  Kyle λ      : %.6f\n"     cfs.kyle_lambda
    if !isnothing(cfs.hawkes_params)
        hp = cfs.hawkes_params
        @printf "  Hawkes μ=%.4f α=%.4f β=%.4f\n" hp.μ hp.α hp.β
    end
    @printf "─────────────────────────────────────────────────\n"

    sde_inputs = chronos_to_sde_inputs(cfs)
    @printf "  Drift (ann) : %+.4f\n"   sde_inputs.r
    @printf "  Init σ      : %.4f\n"    sde_inputs.sigma0

    return (data=data, features=cfs, sde_inputs=sde_inputs)
end
