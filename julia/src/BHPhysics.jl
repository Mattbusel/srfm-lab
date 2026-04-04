"""
BHPhysics — Black-Hole physics engine ported to Julia.

Implements the full BH mass accumulation model, timelike/spacelike classification,
regime detection, and multi-timeframe backtest matching crypto_backtest_mc.py.
"""
module BHPhysics

using DataStructures   # CircularBuffer
using DataFrames
using Statistics
using Dates

export BHConfig, BHState, update!, mass_series, multi_tf_backtest
export bh_regime, bh_active_fraction, bh_mass_gradient
export TradeResult, BacktestResult, run_backtest

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & State
# ─────────────────────────────────────────────────────────────────────────────

"""
    BHConfig(cf, bh_form, bh_collapse, bh_decay, ctl_req)

Immutable configuration for a single BH instance.

Fields:
  cf          — critical frequency threshold (beta < cf → timelike)
  bh_form     — mass threshold to declare BH active
  bh_collapse — mass fraction that triggers collapse (mass * bh_collapse)
  bh_decay    — per-bar mass decay multiplier (< 1.0)
  ctl_req     — consecutive timelike bars required to form BH
"""
struct BHConfig
    cf::Float64
    bh_form::Float64
    bh_collapse::Float64
    bh_decay::Float64
    ctl_req::Int

    function BHConfig(; cf=0.003, bh_form=0.25, bh_collapse=0.15,
                       bh_decay=0.97, ctl_req=3)
        @assert 0 < cf < 1       "cf must be in (0,1)"
        @assert bh_form > 0      "bh_form must be positive"
        @assert 0 < bh_collapse < bh_form "bh_collapse must be in (0, bh_form)"
        @assert 0 < bh_decay < 1 "bh_decay must be in (0,1)"
        @assert ctl_req >= 1     "ctl_req must be >= 1"
        new(cf, bh_form, bh_collapse, bh_decay, ctl_req)
    end
end

"""
    BHState

Mutable runtime state for a single BH instance.
"""
mutable struct BHState
    config::BHConfig
    mass::Float64
    active::Bool
    bh_dir::Int          # direction: -1 = bearish, 0 = neutral, 1 = bullish
    ctl::Int             # consecutive timelike bar count
    prev_price::Float64
    prices::CircularBuffer{Float64}
    bars_active::Int
    peak_mass::Float64
    total_mass_absorbed::Float64
end

function BHState(config::BHConfig; window::Int=50)
    BHState(config, 0.0, false, 0, 0, NaN,
            CircularBuffer{Float64}(window), 0, 0.0, 0.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Core Physics
# ─────────────────────────────────────────────────────────────────────────────

"""
    beta(price, prev_price) → Float64

Compute the normalised price velocity (analogous to relativistic β = v/c).
Returns |Δlog(price)|.
"""
@inline function beta(price::Float64, prev_price::Float64)::Float64
    return abs(log(price / prev_price))
end

"""
    classify_bar(β, cf) → Symbol

Classify a bar as :timelike (slow, β < cf) or :spacelike (fast, β ≥ cf).
"""
@inline function classify_bar(β::Float64, cf::Float64)::Symbol
    return β < cf ? :timelike : :spacelike
end

"""
    mass_delta(β, cf, is_timelike) → Float64

Mass increment for a single bar.
- Timelike bars slowly deposit mass (Hawking-radiation-like infall).
- Spacelike bars contribute impulse mass proportional to how far β > cf.
"""
@inline function mass_delta(β::Float64, cf::Float64, is_timelike::Bool)::Float64
    if is_timelike
        return cf * 0.5      # steady infall
    else
        excess = β - cf
        return excess * 2.0  # impulsive accretion
    end
end

"""
    update!(state, price) → Bool

Advance the BH state by one bar. Returns true if the BH became active this bar.
"""
function update!(state::BHState, price::Float64)::Bool
    config = state.config
    newly_activated = false

    # Bootstrap: need prev_price
    if isnan(state.prev_price)
        state.prev_price = price
        push!(state.prices, price)
        return false
    end

    push!(state.prices, price)
    β = beta(price, state.prev_price)
    is_tl = β < config.cf

    # --- Consecutive timelike counter ---
    if is_tl
        state.ctl += 1
    else
        state.ctl = 0
    end

    # --- Mass dynamics ---
    dm = mass_delta(β, config.cf, is_tl)
    state.mass = state.mass * config.bh_decay + dm
    state.total_mass_absorbed += dm

    # --- Direction: based on price movement during active phase ---
    if length(state.prices) >= 2
        net = log(price / state.prices[1])
        if net > config.cf * 0.5
            state.bh_dir = 1
        elseif net < -config.cf * 0.5
            state.bh_dir = -1
        else
            state.bh_dir = 0
        end
    end

    # --- Activation ---
    if !state.active
        if state.mass >= config.bh_form && state.ctl >= config.ctl_req
            state.active = true
            newly_activated = true
            state.bars_active = 0
            state.peak_mass = state.mass
        end
    else
        state.bars_active += 1
        state.peak_mass = max(state.peak_mass, state.mass)

        # --- Collapse check ---
        if state.mass < config.bh_collapse
            state.active = false
            state.mass = 0.0
            state.ctl = 0
            state.bh_dir = 0
            state.peak_mass = 0.0
            state.bars_active = 0
        end
    end

    state.prev_price = price
    return newly_activated
end

"""
    reset!(state)

Reset BH state to initial conditions (preserves config).
"""
function reset!(state::BHState)
    state.mass = 0.0
    state.active = false
    state.bh_dir = 0
    state.ctl = 0
    state.prev_price = NaN
    state.bars_active = 0
    state.peak_mass = 0.0
    state.total_mass_absorbed = 0.0
    empty!(state.prices)
end

# ─────────────────────────────────────────────────────────────────────────────
# Series Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    mass_series(prices, config) → NamedTuple

Run BH physics across a price series, returning arrays of:
  masses   — BH mass at each bar
  active   — Bool: is BH active?
  ctl      — consecutive timelike count
  bh_dir   — direction signal
  betas    — raw β values
  timelike — Bool: is bar timelike?
"""
function mass_series(prices::Vector{Float64}, config::BHConfig)::NamedTuple
    n = length(prices)
    masses   = zeros(Float64, n)
    active   = zeros(Bool, n)
    ctl_arr  = zeros(Int, n)
    bh_dir_a = zeros(Int, n)
    betas    = zeros(Float64, n)
    timelike = zeros(Bool, n)

    state = BHState(config)

    for i in 1:n
        update!(state, prices[i])
        masses[i]   = state.mass
        active[i]   = state.active
        ctl_arr[i]  = state.ctl
        bh_dir_a[i] = state.bh_dir
        if i > 1
            b = beta(prices[i], prices[i-1])
            betas[i]    = b
            timelike[i] = b < config.cf
        end
    end

    return (masses=masses, active=active, ctl=ctl_arr,
            bh_dir=bh_dir_a, betas=betas, timelike=timelike)
end

# ─────────────────────────────────────────────────────────────────────────────
# Regime Detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    bh_regime(masses, active, bh_dir) → Vector{String}

Map BH state arrays to regime labels:
  "BH_BULL"   — active + bullish
  "BH_BEAR"   — active + bearish
  "BH_NEUTRAL"— active + neutral
  "DRIFT"     — inactive
"""
function bh_regime(masses::Vector{Float64}, active::Vector{Bool},
                   bh_dir::Vector{Int})::Vector{String}
    n = length(masses)
    @assert n == length(active) == length(bh_dir)
    regimes = Vector{String}(undef, n)
    for i in 1:n
        if active[i]
            if bh_dir[i] == 1
                regimes[i] = "BH_BULL"
            elseif bh_dir[i] == -1
                regimes[i] = "BH_BEAR"
            else
                regimes[i] = "BH_NEUTRAL"
            end
        else
            regimes[i] = "DRIFT"
        end
    end
    return regimes
end

"""
    bh_active_fraction(active, window) → Vector{Float64}

Rolling fraction of bars where BH was active over given window.
"""
function bh_active_fraction(active::Vector{Bool}, window::Int)::Vector{Float64}
    n = length(active)
    out = fill(NaN, n)
    for i in window:n
        out[i] = mean(active[i-window+1:i])
    end
    return out
end

"""
    bh_mass_gradient(masses, window) → Vector{Float64}

Rolling linear regression slope of BH mass (mass momentum).
"""
function bh_mass_gradient(masses::Vector{Float64}, window::Int)::Vector{Float64}
    n = length(masses)
    out = fill(NaN, n)
    x = collect(1.0:window)
    xm = mean(x)
    ss_x = sum((xi - xm)^2 for xi in x)
    for i in window:n
        y = masses[i-window+1:i]
        ym = mean(y)
        ss_xy = sum((x[j] - xm) * (y[j] - ym) for j in 1:window)
        out[i] = ss_xy / ss_x
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Trade Result & Backtest
# ─────────────────────────────────────────────────────────────────────────────

"""
    TradeResult

Stores all metadata for a single closed trade.
"""
struct TradeResult
    entry_bar::Int
    exit_bar::Int
    entry_price::Float64
    exit_price::Float64
    direction::Int        # +1 long, -1 short
    pnl::Float64          # log-return
    mfe::Float64          # max favourable excursion
    mae::Float64          # max adverse excursion
    duration::Int         # bars held
    peak_mass_entry::Float64
    regime::String        # regime at entry
    tf_score::Int         # multi-TF alignment score
end

"""
    BacktestResult

Full backtest output.
"""
struct BacktestResult
    trades::Vector{TradeResult}
    equity_curve::Vector{Float64}
    positions::Vector{Int}
    regime_series::Vector{String}
    mass_series_1d::Vector{Float64}
    n_trades::Int
    sharpe::Float64
    max_dd::Float64
    total_return::Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# Single-TF Backtest
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_backtest(prices, config; kwargs) → BacktestResult

Single-timeframe backtest. Signals:
  ENTRY LONG:  BH activates with bh_dir == 1
  ENTRY SHORT: BH activates with bh_dir == -1 (if !long_only)
  EXIT:        BH collapses (active goes false)

Kwargs:
  long_only   (Bool, false)
  commission  (Float64, 0.0004)  — per-side fraction
  slippage    (Float64, 0.0001)
  initial_eq  (Float64, 1.0)
"""
function run_backtest(prices::Vector{Float64}, config::BHConfig;
                      long_only::Bool=false,
                      commission::Float64=0.0004,
                      slippage::Float64=0.0001,
                      initial_eq::Float64=1.0)::BacktestResult

    n = length(prices)
    ms = mass_series(prices, config)
    regimes = bh_regime(ms.masses, ms.active, ms.bh_dir)

    trades      = TradeResult[]
    equity      = fill(initial_eq, n)
    positions   = zeros(Int, n)

    pos         = 0          # current position: 0, +1, -1
    entry_bar   = 0
    entry_price = 0.0
    entry_dir   = 0
    entry_mass  = 0.0
    entry_regime = ""
    running_mfe = 0.0
    running_mae = 0.0
    eq          = initial_eq

    cost = commission + slippage

    for i in 2:n
        prev_active = ms.active[i-1]
        curr_active = ms.active[i]
        dir         = ms.bh_dir[i]
        mass_now    = ms.masses[i]

        # --- Update equity ---
        if pos != 0 && i > entry_bar
            bar_ret = log(prices[i] / prices[i-1]) * pos
            eq *= exp(bar_ret)
            # Update MFE / MAE
            fav = bar_ret * pos
            if fav > 0
                running_mfe += fav
            else
                running_mae += abs(fav)
            end
        end
        equity[i] = eq
        positions[i] = pos

        # --- Entry signal: BH newly activated ---
        if pos == 0 && curr_active && !prev_active
            if dir == 1
                pos = 1
            elseif dir == -1 && !long_only
                pos = -1
            end
            if pos != 0
                entry_bar    = i
                entry_price  = prices[i] * (1 + cost * pos)
                entry_dir    = pos
                entry_mass   = mass_now
                entry_regime = regimes[i]
                running_mfe  = 0.0
                running_mae  = 0.0
                eq          *= (1 - cost)   # entry cost
            end
        end

        # --- Exit signal: BH collapsed ---
        if pos != 0 && !curr_active && prev_active
            exit_price = prices[i] * (1 - cost * pos)
            pnl = log(exit_price / entry_price) * entry_dir
            eq *= (1 - cost)   # exit cost

            push!(trades, TradeResult(
                entry_bar, i, entry_price, exit_price, entry_dir,
                pnl, running_mfe, running_mae, i - entry_bar,
                entry_mass, entry_regime, 1
            ))

            pos = 0
        end
    end

    # --- Force-close any open trade ---
    if pos != 0
        exit_price = prices[n] * (1 - cost * pos)
        pnl = log(exit_price / entry_price) * entry_dir
        push!(trades, TradeResult(
            entry_bar, n, entry_price, exit_price, entry_dir,
            pnl, running_mfe, running_mae, n - entry_bar,
            entry_mass, entry_regime, 1
        ))
    end

    # --- Summary statistics ---
    rets = diff(log.(equity))
    sharpe = length(rets) > 1 ? mean(rets) / (std(rets) + 1e-10) * sqrt(252) : 0.0
    peak = equity[1]; max_dd = 0.0
    for e in equity
        peak = max(peak, e)
        dd = (peak - e) / peak
        max_dd = max(max_dd, dd)
    end
    total_ret = log(equity[end] / equity[1])

    return BacktestResult(trades, equity, positions, regimes,
                          ms.masses, length(trades), sharpe, max_dd, total_ret)
end

# ─────────────────────────────────────────────────────────────────────────────
# Multi-Timeframe Backtest
# ─────────────────────────────────────────────────────────────────────────────

"""
    TFAlignment

Holds aligned multi-TF BH state for a single bar.
"""
struct TFAlignment
    active_1d::Bool
    active_1h::Bool
    active_15m::Bool
    dir_1d::Int
    dir_1h::Int
    dir_15m::Int
    score::Int     # -3 to +3: aligned bull/bear across TFs
end

"""
    compute_tf_score(align) → Int

Count directional agreement across timeframes. +1 per bullish TF, -1 per bearish.
"""
function compute_tf_score(a::TFAlignment)::Int
    score = 0
    for (act, d) in ((a.active_1d, a.dir_1d),
                      (a.active_1h, a.dir_1h),
                      (a.active_15m, a.dir_15m))
        if act
            score += d
        end
    end
    return score
end

"""
    upsample_index(prices_high_tf, prices_low_tf) → Vector{Int}

For each bar in prices_low_tf, find the corresponding bar index in prices_high_tf.
Assumes both series start at the same time and prices_high_tf has fewer bars.
"""
function upsample_index(n_high::Int, n_low::Int)::Vector{Int}
    ratio = n_low / n_high
    return [clamp(round(Int, i / ratio), 1, n_high) for i in 1:n_low]
end

"""
    multi_tf_backtest(prices_1d, prices_1h, prices_15m, configs) → BacktestResult

Full three-timeframe backtest.

configs keys: "1d", "1h", "15m" → BHConfig

Entry logic:
  - Primary signal from 15m BH activation
  - Confirmed by ≥2 TFs aligned in same direction (tf_score ≥ 2 or ≤ -2)
  - Exit: 15m BH collapses OR 1d BH collapses

Per-trade metadata includes MFE, MAE, tf_score, regime at entry.
"""
function multi_tf_backtest(prices_1d::Vector{Float64},
                            prices_1h::Vector{Float64},
                            prices_15m::Vector{Float64},
                            configs::Dict{String,BHConfig};
                            long_only::Bool=false,
                            commission::Float64=0.0004,
                            slippage::Float64=0.0001,
                            initial_eq::Float64=1.0,
                            min_tf_score::Int=2)::BacktestResult

    cost = commission + slippage

    # Compute full mass series per TF
    ms_1d  = mass_series(prices_1d,  configs["1d"])
    ms_1h  = mass_series(prices_1h,  configs["1h"])
    ms_15m = mass_series(prices_15m, configs["15m"])

    reg_1d  = bh_regime(ms_1d.masses,  ms_1d.active,  ms_1d.bh_dir)
    reg_15m = bh_regime(ms_15m.masses, ms_15m.active, ms_15m.bh_dir)

    n = length(prices_15m)
    n_1d = length(prices_1d)
    n_1h = length(prices_1h)

    # Build index mappings: 15m bar → 1d bar, 1h bar
    idx_1d = upsample_index(n_1d, n)
    idx_1h = upsample_index(n_1h, n)

    trades    = TradeResult[]
    equity    = fill(initial_eq, n)
    positions = zeros(Int, n)

    pos         = 0
    entry_bar   = 0
    entry_price = 0.0
    entry_dir   = 0
    entry_mass  = 0.0
    entry_regime = ""
    entry_tf_score = 0
    running_mfe = 0.0
    running_mae = 0.0
    eq          = initial_eq

    for i in 2:n
        i1d = idx_1d[i]
        i1h = idx_1h[i]

        align = TFAlignment(
            ms_1d.active[i1d],   ms_1h.active[i1h],   ms_15m.active[i],
            ms_1d.bh_dir[i1d],   ms_1h.bh_dir[i1h],   ms_15m.bh_dir[i],
            0  # placeholder
        )
        tf_score = compute_tf_score(align)

        # Update running equity
        if pos != 0 && i > entry_bar
            bar_ret = log(prices_15m[i] / prices_15m[i-1]) * pos
            eq *= exp(bar_ret)
            fav = bar_ret * pos
            if fav > 0
                running_mfe += fav
            else
                running_mae += abs(fav)
            end
        end
        equity[i] = eq
        positions[i] = pos

        prev_15m_active = ms_15m.active[i-1]
        curr_15m_active = ms_15m.active[i]
        curr_1d_active  = ms_1d.active[i1d]

        # --- Entry ---
        if pos == 0 && curr_15m_active && !prev_15m_active
            if tf_score >= min_tf_score
                pos = 1
            elseif tf_score <= -min_tf_score && !long_only
                pos = -1
            end
            if pos != 0
                entry_bar      = i
                entry_price    = prices_15m[i] * (1 + cost * pos)
                entry_dir      = pos
                entry_mass     = ms_15m.masses[i]
                entry_regime   = reg_15m[i]
                entry_tf_score = tf_score
                running_mfe    = 0.0
                running_mae    = 0.0
                eq            *= (1 - cost)
            end
        end

        # --- Exit: 15m collapse OR 1d collapse ---
        if pos != 0
            exit_cond = (!curr_15m_active && prev_15m_active) ||
                        (i1d > 1 && !curr_1d_active && ms_1d.active[max(1,i1d-1)])
            if exit_cond
                exit_price = prices_15m[i] * (1 - cost * pos)
                pnl = log(exit_price / entry_price) * entry_dir
                eq *= (1 - cost)
                push!(trades, TradeResult(
                    entry_bar, i, entry_price, exit_price, entry_dir,
                    pnl, running_mfe, running_mae, i - entry_bar,
                    entry_mass, entry_regime, entry_tf_score
                ))
                pos = 0
            end
        end
    end

    # Force close
    if pos != 0
        exit_price = prices_15m[n] * (1 - cost * pos)
        pnl = log(exit_price / entry_price) * entry_dir
        push!(trades, TradeResult(
            entry_bar, n, entry_price, exit_price, entry_dir,
            pnl, running_mfe, running_mae, n - entry_bar,
            entry_mass, entry_regime, entry_tf_score
        ))
    end

    rets = diff(log.(equity))
    sharpe = length(rets) > 1 ? mean(rets) / (std(rets) + 1e-10) * sqrt(252 * 96) : 0.0
    peak = equity[1]; max_dd = 0.0
    for e in equity
        peak = max(peak, e)
        dd = (peak - e) / peak
        max_dd = max(max_dd, dd)
    end
    total_ret = log(equity[end] / equity[1])

    return BacktestResult(trades, equity, positions, reg_15m,
                          ms_15m.masses, length(trades), sharpe, max_dd, total_ret)
end

# ─────────────────────────────────────────────────────────────────────────────
# Analysis Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    trades_to_dataframe(trades) → DataFrame

Convert vector of TradeResult to a tidy DataFrame for analysis.
"""
function trades_to_dataframe(trades::Vector{TradeResult})::DataFrame
    if isempty(trades)
        return DataFrame(entry_bar=Int[], exit_bar=Int[],
                         entry_price=Float64[], exit_price=Float64[],
                         direction=Int[], pnl=Float64[],
                         mfe=Float64[], mae=Float64[],
                         duration=Int[], peak_mass=Float64[],
                         regime=String[], tf_score=Int[])
    end
    return DataFrame(
        entry_bar   = [t.entry_bar    for t in trades],
        exit_bar    = [t.exit_bar     for t in trades],
        entry_price = [t.entry_price  for t in trades],
        exit_price  = [t.exit_price   for t in trades],
        direction   = [t.direction    for t in trades],
        pnl         = [t.pnl          for t in trades],
        mfe         = [t.mfe          for t in trades],
        mae         = [t.mae          for t in trades],
        duration    = [t.duration     for t in trades],
        peak_mass   = [t.peak_mass_entry for t in trades],
        regime      = [t.regime       for t in trades],
        tf_score    = [t.tf_score     for t in trades],
    )
end

"""
    parameter_sensitivity(prices, cf_range, form_range) → DataFrame

Grid search over (cf, bh_form) showing Sharpe, # trades, max DD per combo.
"""
function parameter_sensitivity(prices::Vector{Float64},
                                cf_range::AbstractVector{Float64},
                                form_range::AbstractVector{Float64})::DataFrame

    rows = NamedTuple{(:cf, :bh_form, :sharpe, :n_trades, :max_dd, :total_ret),
                      Tuple{Float64,Float64,Float64,Int,Float64,Float64}}[]

    for cf in cf_range, form in form_range
        try
            cfg = BHConfig(cf=cf, bh_form=form)
            res = run_backtest(prices, cfg)
            push!(rows, (cf=cf, bh_form=form, sharpe=res.sharpe,
                         n_trades=res.n_trades, max_dd=res.max_dd,
                         total_ret=res.total_return))
        catch
            continue
        end
    end
    return DataFrame(rows)
end

"""
    walk_forward_bh(prices, config, train_size, test_size, step) → DataFrame

Walk-forward test of a fixed BH config, reporting out-of-sample performance per window.
"""
function walk_forward_bh(prices::Vector{Float64}, config::BHConfig,
                          train_size::Int, test_size::Int, step::Int)::DataFrame

    n = length(prices)
    results = NamedTuple[]

    i = 1
    while i + train_size + test_size <= n
        train_end  = i + train_size - 1
        test_start = train_end + 1
        test_end   = min(test_start + test_size - 1, n)

        test_prices = prices[test_start:test_end]
        res = run_backtest(test_prices, config)

        push!(results, (
            train_start = i,
            train_end   = train_end,
            test_start  = test_start,
            test_end    = test_end,
            sharpe      = res.sharpe,
            n_trades    = res.n_trades,
            max_dd      = res.max_dd,
            total_ret   = res.total_return,
        ))
        i += step
    end
    return isempty(results) ? DataFrame() : DataFrame(results)
end

"""
    mass_cross_sectional(prices_dict, config) → DataFrame

Run mass_series on a dict of symbol → price_vector, return a DataFrame
with columns [:symbol, :bar, :mass, :active, :regime].
"""
function mass_cross_sectional(prices_dict::Dict{String, Vector{Float64}},
                               config::BHConfig)::DataFrame

    rows = NamedTuple[]
    for (sym, prices) in prices_dict
        ms = mass_series(prices, config)
        reg = bh_regime(ms.masses, ms.active, ms.bh_dir)
        for i in 1:length(prices)
            push!(rows, (symbol=sym, bar=i, mass=ms.masses[i],
                         active=ms.active[i], regime=reg[i]))
        end
    end
    return isempty(rows) ? DataFrame() : DataFrame(rows)
end

end # module BHPhysics
