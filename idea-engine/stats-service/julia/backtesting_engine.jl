# =============================================================================
# backtesting_engine.jl — Pure Julia Event-Driven Backtesting Engine
# =============================================================================
# A production-grade backtesting framework with:
#   - Event-driven bar-by-bar simulation with priority queue
#   - Realistic transaction costs (spread, slippage, commissions)
#   - Position management with limit/market orders, partial fills
#   - Portfolio-level risk controls
#   - Monte Carlo significance testing (permutation, block bootstrap)
#   - Walk-forward validation
#   - Comprehensive BacktestResult struct
#
# Julia ≥ 1.10 | No external packages
# =============================================================================

module BacktestingEngine

using Statistics
using LinearAlgebra

export BacktestConfig, BacktestResult, run_backtest
export WalkForwardResult, run_walk_forward
export MonteCarloResult, run_monte_carlo
export Order, Fill, Position, PortfolioState
export performance_report, buy_and_hold_benchmark

# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================

"""
    Order

Represents a pending order in the backtest.

Fields:
- `id`: unique order identifier
- `asset_idx`: column index of asset in price matrix
- `direction`: +1 (buy) or -1 (sell)
- `quantity`: number of units (positive)
- `order_type`: :market or :limit
- `limit_price`: relevant for :limit orders
- `timestamp`: bar index when order was placed
- `time_in_force`: :day (cancel if not filled same bar) or :gtc
"""
mutable struct Order
    id::Int
    asset_idx::Int
    direction::Int          # +1 buy, -1 sell
    quantity::Float64
    order_type::Symbol      # :market, :limit
    limit_price::Float64
    timestamp::Int
    time_in_force::Symbol   # :day, :gtc
end

"""
    Fill

Record of an executed order.
"""
struct Fill
    order_id::Int
    asset_idx::Int
    direction::Int
    quantity::Float64
    fill_price::Float64
    commission::Float64
    timestamp::Int
end

"""
    Position

Current position in a single asset.
"""
mutable struct Position
    asset_idx::Int
    quantity::Float64       # positive = long, negative = short
    avg_cost::Float64       # volume-weighted average cost
    realized_pnl::Float64
end

"""
    BacktestConfig

Configuration for the backtest engine.
"""
struct BacktestConfig
    # Cost model
    spread_bps::Float64         # bid-ask spread in basis points
    slippage_linear::Float64    # linear slippage coefficient (bps per unit size)
    slippage_sqrt::Float64      # square-root impact coefficient
    commission_per_trade::Float64  # fixed commission in $
    commission_pct::Float64     # proportional commission (fraction of notional)

    # Position limits
    max_position_pct::Float64   # max position as fraction of portfolio
    max_concentration::Float64  # max weight in any single asset
    max_drawdown_stop::Float64  # halt trading if drawdown exceeds this

    # Capital
    initial_capital::Float64

    # Benchmark
    benchmark_idx::Int          # column index for benchmark asset (-1 = none)
end

"""Default BacktestConfig."""
function BacktestConfig(;
    spread_bps::Float64=5.0,
    slippage_linear::Float64=0.5,
    slippage_sqrt::Float64=0.1,
    commission_per_trade::Float64=0.0,
    commission_pct::Float64=0.001,
    max_position_pct::Float64=0.20,
    max_concentration::Float64=0.30,
    max_drawdown_stop::Float64=0.25,
    initial_capital::Float64=1_000_000.0,
    benchmark_idx::Int=-1)

    BacktestConfig(spread_bps, slippage_linear, slippage_sqrt,
                   commission_per_trade, commission_pct,
                   max_position_pct, max_concentration, max_drawdown_stop,
                   initial_capital, benchmark_idx)
end

"""
    PortfolioState

Mutable state of the portfolio during simulation.
"""
mutable struct PortfolioState
    cash::Float64
    positions::Dict{Int, Position}
    equity_curve::Vector{Float64}
    fills::Vector{Fill}
    pending_orders::Vector{Order}
    next_order_id::Int
    peak_equity::Float64
    current_drawdown::Float64
    halted::Bool            # true if drawdown stop triggered
end

function PortfolioState(initial_capital::Float64)
    PortfolioState(
        initial_capital,
        Dict{Int, Position}(),
        Float64[initial_capital],
        Fill[],
        Order[],
        1,
        initial_capital,
        0.0,
        false
    )
end

"""
    BacktestResult

Comprehensive results from a backtest run.
"""
struct BacktestResult
    # Equity curve
    equity_curve::Vector{Float64}
    returns::Vector{Float64}
    benchmark_returns::Union{Vector{Float64}, Nothing}

    # Performance metrics
    total_return::Float64
    annualized_return::Float64
    annualized_vol::Float64
    sharpe_ratio::Float64
    sortino_ratio::Float64
    calmar_ratio::Float64
    max_drawdown::Float64
    max_drawdown_duration::Int   # in bars
    omega_ratio::Float64
    profit_factor::Float64
    win_rate::Float64

    # Trade statistics
    n_trades::Int
    avg_trade_return::Float64
    avg_holding_period::Float64
    total_commission::Float64
    total_slippage::Float64

    # Risk metrics
    var_95::Float64
    cvar_95::Float64
    var_99::Float64

    # Benchmark comparison
    information_ratio::Float64
    beta::Float64
    alpha_annualized::Float64
    tracking_error::Float64

    # Raw data
    fills::Vector{Fill}
    config::BacktestConfig
end

# =============================================================================
# SECTION 2: TRANSACTION COST MODEL
# =============================================================================

"""
    compute_execution_price(mid_price, direction, quantity, portfolio_value, config) -> Float64

Compute execution price accounting for spread, linear slippage, and square-root impact.

Total cost = spread/2 + linear_slippage * order_size_pct + sqrt_impact * sqrt(order_size_pct)

Impact is applied in the direction of the trade (buy = price up, sell = price down).
"""
function compute_execution_price(mid_price::Float64,
                                   direction::Int,
                                   quantity::Float64,
                                   portfolio_value::Float64,
                                   config::BacktestConfig)::Float64

    # Spread cost: half spread in direction of trade
    spread_cost = mid_price * (config.spread_bps / 2.0 / 10_000.0)

    # Order size as fraction of portfolio
    order_notional = quantity * mid_price
    size_pct = portfolio_value > 0 ? order_notional / portfolio_value : 0.0

    # Linear slippage
    linear_cost = mid_price * (config.slippage_linear / 10_000.0) * size_pct

    # Square-root market impact (Almgren et al. 2005)
    sqrt_cost = mid_price * config.slippage_sqrt * sqrt(max(size_pct, 0.0)) / 10_000.0

    total_impact = spread_cost + linear_cost + sqrt_cost
    return mid_price + direction * total_impact
end

"""
    compute_commission(fill_price, quantity, config) -> Float64

Compute total commission for a fill.
"""
function compute_commission(fill_price::Float64,
                              quantity::Float64,
                              config::BacktestConfig)::Float64
    notional = fill_price * quantity
    return config.commission_per_trade + config.commission_pct * notional
end

# =============================================================================
# SECTION 3: ORDER MANAGEMENT
# =============================================================================

"""
    place_market_order!(state, asset_idx, direction, quantity, bar_idx) -> Int

Place a market order, returning the order ID.
"""
function place_market_order!(state::PortfolioState,
                               asset_idx::Int,
                               direction::Int,
                               quantity::Float64,
                               bar_idx::Int)::Int
    id = state.next_order_id
    state.next_order_id += 1

    order = Order(id, asset_idx, direction, quantity, :market,
                   0.0, bar_idx, :day)
    push!(state.pending_orders, order)
    return id
end

"""
    place_limit_order!(state, asset_idx, direction, quantity, limit_price, bar_idx; tif=:gtc) -> Int

Place a limit order. Will fill only if price crosses limit.
"""
function place_limit_order!(state::PortfolioState,
                               asset_idx::Int,
                               direction::Int,
                               quantity::Float64,
                               limit_price::Float64,
                               bar_idx::Int;
                               tif::Symbol=:gtc)::Int
    id = state.next_order_id
    state.next_order_id += 1

    order = Order(id, asset_idx, direction, quantity, :limit,
                   limit_price, bar_idx, tif)
    push!(state.pending_orders, order)
    return id
end

"""
    process_fills!(state, prices, bar_idx, portfolio_value, config)

Process all pending orders for current bar. Fill market orders at bid/ask,
fill limit orders if price crosses limit level.
"""
function process_fills!(state::PortfolioState,
                          prices::Vector{Float64},
                          bar_idx::Int,
                          portfolio_value::Float64,
                          config::BacktestConfig)

    remaining_orders = Order[]

    for order in state.pending_orders
        price = prices[order.asset_idx]
        price <= 0 && continue

        # Check if limit order can fill
        if order.order_type == :limit
            if order.direction == 1 && price > order.limit_price
                # Buy limit: fill only if ask <= limit
                push!(remaining_orders, order)
                continue
            elseif order.direction == -1 && price < order.limit_price
                # Sell limit: fill only if bid >= limit
                push!(remaining_orders, order)
                continue
            end
        end

        # Day orders: cancel if from previous bar
        if order.time_in_force == :day && order.timestamp < bar_idx
            continue  # cancel
        end

        # Compute fill price
        fill_price = compute_execution_price(price, order.direction,
                                               order.quantity,
                                               portfolio_value, config)
        commission = compute_commission(fill_price, order.quantity, config)

        # Check portfolio constraints before filling
        fill_notional = fill_price * order.quantity
        if order.direction == 1
            # Buying: check we have enough cash
            total_cost = fill_notional + commission
            if total_cost > state.cash
                # Partial fill
                affordable = max(0.0, state.cash - commission - fill_price * 0.01) / fill_price
                affordable = floor(affordable * 100) / 100  # round down
                affordable <= 0 && continue
                fill_notional = fill_price * affordable
                commission = compute_commission(fill_price, affordable, config)
                order = Order(order.id, order.asset_idx, order.direction,
                               affordable, order.order_type, order.limit_price,
                               order.timestamp, order.time_in_force)
            end
        end

        # Execute fill
        fill = Fill(order.id, order.asset_idx, order.direction,
                     order.quantity, fill_price, commission, bar_idx)
        push!(state.fills, fill)

        # Update cash
        state.cash -= order.direction * fill_price * order.quantity + commission

        # Update position
        if haskey(state.positions, order.asset_idx)
            pos = state.positions[order.asset_idx]
            new_qty = pos.quantity + order.direction * order.quantity
            if abs(new_qty) < 1e-9
                # Position closed: compute realized PnL
                realized = order.direction == -1 ?
                    (fill_price - pos.avg_cost) * order.quantity :
                    (pos.avg_cost - fill_price) * order.quantity
                pos.realized_pnl += realized
                pos.quantity = 0.0
                pos.avg_cost = 0.0
            elseif sign(new_qty) == sign(pos.quantity)
                # Adding to position: update avg cost
                total_cost_basis = pos.avg_cost * abs(pos.quantity) +
                                   fill_price * order.quantity
                pos.avg_cost = total_cost_basis / abs(new_qty)
                pos.quantity = new_qty
            else
                # Reversing position
                pos.quantity = new_qty
                pos.avg_cost = fill_price
            end
        else
            state.positions[order.asset_idx] = Position(
                order.asset_idx,
                order.direction * order.quantity,
                fill_price,
                0.0
            )
        end
    end

    state.pending_orders = remaining_orders
end

# =============================================================================
# SECTION 4: PORTFOLIO VALUATION AND RISK MANAGEMENT
# =============================================================================

"""
    compute_portfolio_value(state, prices) -> Float64

Mark-to-market portfolio value: cash + Σ position_qty * price.
"""
function compute_portfolio_value(state::PortfolioState,
                                   prices::Vector{Float64})::Float64
    value = state.cash
    for (asset_idx, pos) in state.positions
        if 1 <= asset_idx <= length(prices)
            value += pos.quantity * prices[asset_idx]
        end
    end
    return value
end

"""
    check_risk_limits!(state, prices, config) -> Bool

Check portfolio-level risk constraints. Liquidate if needed.
Returns true if portfolio is within limits.
"""
function check_risk_limits!(state::PortfolioState,
                              prices::Vector{Float64},
                              config::BacktestConfig,
                              bar_idx::Int)::Bool

    pv = compute_portfolio_value(state, prices)
    pv <= 0 && (state.halted = true; return false)

    # Update peak and drawdown
    if pv > state.peak_equity
        state.peak_equity = pv
    end
    state.current_drawdown = (state.peak_equity - pv) / state.peak_equity

    # Check drawdown stop
    if state.current_drawdown >= config.max_drawdown_stop
        state.halted = true
        # Liquidate all positions
        for (asset_idx, pos) in state.positions
            if abs(pos.quantity) > 1e-9 && asset_idx <= length(prices)
                direction = pos.quantity > 0 ? -1 : 1
                qty = abs(pos.quantity)
                place_market_order!(state, asset_idx, direction, qty, bar_idx)
            end
        end
        return false
    end

    # Check concentration limits
    for (asset_idx, pos) in state.positions
        if asset_idx <= length(prices)
            notional = abs(pos.quantity * prices[asset_idx])
            concentration = notional / pv
            if concentration > config.max_concentration
                # Trim to max concentration
                target_notional = config.max_concentration * pv
                excess_notional = notional - target_notional
                excess_qty = excess_notional / prices[asset_idx]
                direction = pos.quantity > 0 ? -1 : 1
                place_market_order!(state, asset_idx, direction, excess_qty, bar_idx)
            end
        end
    end

    return true
end

# =============================================================================
# SECTION 5: MAIN BACKTEST LOOP
# =============================================================================

"""
    run_backtest(prices, signal_func, config; dates=nothing) -> BacktestResult

Run the event-driven backtest.

# Arguments
- `prices`: (T × N) matrix of asset prices (close prices)
- `signal_func`: function(t, prices_history, state) -> Dict{Int, Float64}
                 Returns target weights per asset at each bar.
                 Called at the start of each bar before order placement.
- `config`: BacktestConfig
- `dates`: optional date labels for output

# Signal function interface:
    signal_func(bar::Int, price_history::Matrix{Float64}, state::PortfolioState)
        -> Dict{Int, Float64}  # asset_idx => target_weight in [-1, 1]

# Returns
- BacktestResult
"""
function run_backtest(prices::Matrix{Float64},
                       signal_func::Function,
                       config::BacktestConfig;
                       dates=nothing)::BacktestResult

    T, N = size(prices)
    T < 2 && error("Need at least 2 bars to backtest")

    state = PortfolioState(config.initial_capital)
    total_slippage = 0.0

    # Main simulation loop
    for t in 2:T
        state.halted && break

        current_prices = prices[t, :]
        prev_prices    = prices[t-1, :]

        # Get signals: target weights for each asset
        target_weights = try
            signal_func(t, prices[1:t, :], state)
        catch e
            Dict{Int, Float64}()
        end

        # Current portfolio value
        pv = compute_portfolio_value(state, prev_prices)
        pv <= 0 && break

        # Generate orders based on target weights vs current weights
        for (asset_idx, target_w) in target_weights
            (asset_idx < 1 || asset_idx > N) && continue

            # Current weight
            pos = get(state.positions, asset_idx, Position(asset_idx, 0.0, 0.0, 0.0))
            current_notional = pos.quantity * prev_prices[asset_idx]
            current_w = current_notional / pv

            # Trade if deviation > threshold (5bps to avoid excessive turnover)
            delta_w = target_w - current_w
            abs(delta_w) < 0.005 && continue

            # Constrain to max position
            clamped_w = clamp(target_w, -config.max_position_pct, config.max_position_pct)
            delta_w = clamped_w - current_w
            abs(delta_w) < 0.005 && continue

            # Target quantity
            target_notional = clamped_w * pv
            target_qty = target_notional / max(prev_prices[asset_idx], 1e-10)
            delta_qty = target_qty - pos.quantity

            if abs(delta_qty) > 1e-6
                direction = delta_qty > 0 ? 1 : -1
                place_market_order!(state, asset_idx, direction, abs(delta_qty), t)
            end
        end

        # Close positions for assets not in target
        for (asset_idx, pos) in state.positions
            if abs(pos.quantity) > 1e-6 && !haskey(target_weights, asset_idx)
                direction = pos.quantity > 0 ? -1 : 1
                place_market_order!(state, asset_idx, direction, abs(pos.quantity), t)
            end
        end

        # Process fills at current bar prices
        process_fills!(state, current_prices, t, pv, config)

        # Risk management
        check_risk_limits!(state, current_prices, config, t)

        # Record equity
        current_pv = compute_portfolio_value(state, current_prices)
        push!(state.equity_curve, current_pv)
    end

    # Compute returns
    equity = state.equity_curve
    rets = diff(log.(max.(equity, 1e-10)))

    # Benchmark
    bm_rets = nothing
    if config.benchmark_idx > 0 && config.benchmark_idx <= N
        bm_prices = prices[:, config.benchmark_idx]
        bm_rets = diff(log.(max.(bm_prices, 1e-10)))
        n_use = min(length(rets), length(bm_rets))
        bm_rets = bm_rets[1:n_use]
        rets = rets[1:n_use]
    end

    # Compute all metrics
    return _compute_result(equity, rets, bm_rets, state, config)
end

# =============================================================================
# SECTION 6: PERFORMANCE ANALYTICS
# =============================================================================

"""Compute BacktestResult from equity curve and fills."""
function _compute_result(equity::Vector{Float64},
                          rets::Vector{Float64},
                          bm_rets::Union{Vector{Float64},Nothing},
                          state::PortfolioState,
                          config::BacktestConfig)::BacktestResult

    n = length(rets)
    n == 0 && return _empty_result(state, config)

    # Returns statistics
    total_return = equity[end] / equity[1] - 1.0
    ann_return   = (1.0 + total_return)^(252.0 / n) - 1.0
    ann_vol      = std(rets) * sqrt(252)

    sharpe = ann_vol > 0 ? ann_return / ann_vol : 0.0

    # Sortino ratio
    neg_rets = filter(r -> r < 0, rets)
    downside_vol = isempty(neg_rets) ? ann_vol : std(neg_rets) * sqrt(252)
    sortino = downside_vol > 0 ? ann_return / downside_vol : 0.0

    # Max drawdown
    max_dd, dd_dur = _max_drawdown(equity)
    calmar = max_dd > 0 ? ann_return / max_dd : 0.0

    # Omega ratio (threshold = 0)
    gains = sum(max.(rets, 0.0))
    losses = sum(max.(-rets, 0.0))
    omega = losses > 0 ? gains / losses : gains > 0 ? Inf : 1.0

    # Trade statistics
    n_trades = length(state.fills)
    total_comm = sum(f.commission for f in state.fills; init=0.0)

    # Profit factor
    gross_profits = sum(max(r, 0.0) for r in rets; init=0.0)
    gross_losses  = sum(max(-r, 0.0) for r in rets; init=0.0)
    profit_factor = gross_losses > 0 ? gross_profits / gross_losses : gross_profits > 0 ? Inf : 1.0

    win_rate = n > 0 ? sum(rets .> 0) / n : 0.0

    # VaR / CVaR
    sorted_rets = sort(rets)
    idx_95 = max(1, floor(Int, 0.05 * n))
    idx_99 = max(1, floor(Int, 0.01 * n))
    var_95 = -sorted_rets[idx_95]
    var_99 = -sorted_rets[idx_99]
    cvar_95 = -mean(sorted_rets[1:idx_95])

    # Benchmark comparison
    ir = 0.0; beta = 0.0; alpha_ann = 0.0; te = 0.0
    if bm_rets !== nothing && length(bm_rets) >= 5
        n_use = min(length(rets), length(bm_rets))
        r = rets[1:n_use]; b = bm_rets[1:n_use]
        active = r .- b
        te = std(active) * sqrt(252)
        ir = te > 0 ? mean(active) * 252 / te : 0.0

        sbb = sum((b .- mean(b)) .^ 2)
        srb = sum((r .- mean(r)) .* (b .- mean(b)))
        beta = sbb > 0 ? srb / sbb : 1.0
        alpha_ann = (mean(r) - beta * mean(b)) * 252
    end

    return BacktestResult(
        equity, rets, bm_rets,
        total_return, ann_return, ann_vol, sharpe, sortino, calmar,
        max_dd, dd_dur, omega, profit_factor, win_rate,
        n_trades, n > 0 ? mean(rets) : 0.0, 0.0, total_comm, 0.0,
        var_95, cvar_95, var_99,
        ir, beta, alpha_ann, te,
        state.fills, config
    )
end

function _empty_result(state::PortfolioState, config::BacktestConfig)
    BacktestResult(
        state.equity_curve, Float64[], nothing,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1.0, 1.0, 0.5,
        0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        state.fills, config
    )
end

"""Compute max drawdown and duration from equity curve."""
function _max_drawdown(equity::Vector{Float64})
    n = length(equity)
    n < 2 && return (0.0, 0)

    max_dd = 0.0
    max_dur = 0
    peak = equity[1]
    trough_start = 1
    in_drawdown = false
    dd_start = 1

    for i in 2:n
        if equity[i] > peak
            # New high: end drawdown
            if in_drawdown
                dur = i - dd_start
                max_dur = max(max_dur, dur)
                in_drawdown = false
            end
            peak = equity[i]
            trough_start = i
        else
            dd = (peak - equity[i]) / peak
            if dd > max_dd
                max_dd = dd
            end
            if !in_drawdown
                in_drawdown = true
                dd_start = i - 1
            end
        end
    end

    return max_dd, max_dur
end

"""
    performance_report(result::BacktestResult) -> Dict{String, Float64}

Return a dictionary of all performance metrics from a BacktestResult.
"""
function performance_report(result::BacktestResult)::Dict{String, Float64}
    return Dict(
        "total_return"       => result.total_return,
        "annualized_return"  => result.annualized_return,
        "annualized_vol"     => result.annualized_vol,
        "sharpe_ratio"       => result.sharpe_ratio,
        "sortino_ratio"      => result.sortino_ratio,
        "calmar_ratio"       => result.calmar_ratio,
        "max_drawdown"       => result.max_drawdown,
        "max_dd_duration"    => Float64(result.max_drawdown_duration),
        "omega_ratio"        => result.omega_ratio,
        "profit_factor"      => result.profit_factor,
        "win_rate"           => result.win_rate,
        "n_trades"           => Float64(result.n_trades),
        "total_commission"   => result.total_commission,
        "var_95"             => result.var_95,
        "cvar_95"            => result.cvar_95,
        "var_99"             => result.var_99,
        "information_ratio"  => result.information_ratio,
        "beta"               => result.beta,
        "alpha_annualized"   => result.alpha_annualized,
        "tracking_error"     => result.tracking_error,
    )
end

"""
    buy_and_hold_benchmark(prices, config; asset_idx=1) -> BacktestResult

Simple buy-and-hold benchmark for comparison.
"""
function buy_and_hold_benchmark(prices::Matrix{Float64},
                                  config::BacktestConfig;
                                  asset_idx::Int=1)::BacktestResult

    function bh_signal(t::Int, history::Matrix{Float64}, state::PortfolioState)
        Dict(asset_idx => 1.0)
    end

    return run_backtest(prices, bh_signal, config)
end

# =============================================================================
# SECTION 7: WALK-FORWARD VALIDATION
# =============================================================================

"""
    WalkForwardResult

Results of a walk-forward validation.
"""
struct WalkForwardResult
    is_results::Vector{BacktestResult}   # in-sample windows
    oos_results::Vector{BacktestResult}  # out-of-sample windows
    oos_sharpe_mean::Float64
    oos_sharpe_std::Float64
    degradation_ratio::Float64  # oos_sharpe / is_sharpe
    n_folds::Int
end

"""
    run_walk_forward(prices, signal_factory, config;
                     is_window, oos_window, step, refit_func) -> WalkForwardResult

Run walk-forward optimization and testing.

# Arguments
- `prices`: (T × N) price matrix
- `signal_factory`: function(is_prices) -> signal_func
                    Returns a new signal function fitted on IS data
- `config`: BacktestConfig
- `is_window`: in-sample training window (bars)
- `oos_window`: out-of-sample testing window (bars)
- `step`: how many bars to advance the window each fold

# Returns
- WalkForwardResult
"""
function run_walk_forward(prices::Matrix{Float64},
                            signal_factory::Function,
                            config::BacktestConfig;
                            is_window::Int=252,
                            oos_window::Int=63,
                            step::Int=21)::WalkForwardResult

    T, N = size(prices)
    T < is_window + oos_window && error("Not enough data for walk-forward")

    is_results  = BacktestResult[]
    oos_results = BacktestResult[]

    t_start = 1
    while t_start + is_window + oos_window - 1 <= T
        is_end  = t_start + is_window - 1
        oos_end = is_end + oos_window

        is_prices  = prices[t_start:is_end, :]
        oos_prices = prices[is_end:oos_end, :]  # include last IS bar for continuity

        # Fit signal on in-sample data
        signal_func = try
            signal_factory(is_prices)
        catch e
            (t, hist, st) -> Dict{Int, Float64}()
        end

        # Evaluate on in-sample
        is_res = run_backtest(is_prices, signal_func, config)
        push!(is_results, is_res)

        # Evaluate on out-of-sample (no refitting)
        oos_res = run_backtest(oos_prices, signal_func, config)
        push!(oos_results, oos_res)

        t_start += step
    end

    if isempty(oos_results)
        return WalkForwardResult(is_results, oos_results, 0.0, 0.0, 0.0, 0)
    end

    oos_sharpes = [r.sharpe_ratio for r in oos_results]
    is_sharpes  = [r.sharpe_ratio for r in is_results]

    oos_mean = mean(oos_sharpes)
    oos_std  = std(oos_sharpes)
    is_mean  = mean(is_sharpes)
    degradation = is_mean != 0.0 ? oos_mean / is_mean : 0.0

    return WalkForwardResult(is_results, oos_results, oos_mean, oos_std,
                              degradation, length(oos_results))
end

# =============================================================================
# SECTION 8: MONTE CARLO ANALYSIS
# =============================================================================

"""
    MonteCarloResult

Results of Monte Carlo permutation/bootstrap analysis.
"""
struct MonteCarloResult
    observed_sharpe::Float64
    simulated_sharpes::Vector{Float64}
    p_value::Float64
    is_significant::Bool
    confidence_interval_95::Tuple{Float64, Float64}
    method::Symbol
end

"""
    run_monte_carlo(prices, signal_func, config;
                    n_simulations=500, method=:permute_signals,
                    block_size=20) -> MonteCarloResult

Monte Carlo test for strategy significance.

Methods:
- `:permute_signals`: randomly shuffle trade signals across time
- `:block_bootstrap`: block bootstrap the return series
- `:random_entry`: random buy/sell signals with same frequency

The null hypothesis: observed Sharpe ≤ typical Sharpe from random signals.
p-value = fraction of simulations that exceed observed Sharpe.
"""
function run_monte_carlo(prices::Matrix{Float64},
                          signal_func::Function,
                          config::BacktestConfig;
                          n_simulations::Int=500,
                          method::Symbol=:permute_signals,
                          block_size::Int=20)::MonteCarloResult

    T, N = size(prices)

    # Observed result
    observed = run_backtest(prices, signal_func, config)
    obs_sharpe = observed.sharpe_ratio

    # Collect simulated Sharpes
    sim_sharpes = zeros(n_simulations)
    rng = 12345  # deterministic seed

    function lcg_rand(state::Int)
        state = (1664525 * state + 1013904223) % (2^32)
        (state, state / 2^32)
    end

    for sim in 1:n_simulations
        rng, _ = lcg_rand(rng)
        seed_this = rng

        if method == :permute_signals
            # Create a time-shuffled signal function
            # Build permuted time index
            perm = collect(1:T)
            # Fisher-Yates shuffle
            for i in T:-1:2
                seed_this, u = lcg_rand(seed_this)
                j = floor(Int, u * i) + 1
                j = clamp(j, 1, i)
                perm[i], perm[j] = perm[j], perm[i]
            end
            perm_prices = prices[perm, :]
            sim_result = run_backtest(perm_prices, signal_func, config)
            sim_sharpes[sim] = sim_result.sharpe_ratio

        elseif method == :block_bootstrap
            # Block bootstrap the price series
            n_blocks = ceil(Int, T / block_size)
            boot_indices = Int[]
            for _ in 1:n_blocks
                seed_this, u = lcg_rand(seed_this)
                start = floor(Int, u * (T - block_size)) + 1
                start = clamp(start, 1, T - block_size + 1)
                append!(boot_indices, start:(start + block_size - 1))
            end
            boot_indices = boot_indices[1:T]
            boot_prices = prices[boot_indices, :]
            sim_result = run_backtest(boot_prices, signal_func, config)
            sim_sharpes[sim] = sim_result.sharpe_ratio

        else  # :random_entry
            # Random entry/exit with same frequency as original
            original_trades = length(observed.fills)
            trade_prob = original_trades > 0 ? original_trades / T : 0.1

            function random_signal(t::Int, hist::Matrix{Float64}, st::PortfolioState)
                seed_this, u = lcg_rand(seed_this)
                if u < trade_prob
                    seed_this2, u2 = lcg_rand(seed_this)
                    seed_this = seed_this2
                    dir = u2 > 0.5 ? 1.0 : -1.0
                    return Dict(1 => dir * 0.5)
                end
                return Dict{Int, Float64}()
            end

            sim_result = run_backtest(prices, random_signal, config)
            sim_sharpes[sim] = sim_result.sharpe_ratio
        end
    end

    # p-value: fraction of simulations with Sharpe >= observed
    p_val = sum(sim_sharpes .>= obs_sharpe) / n_simulations
    is_sig = p_val < 0.05

    # 95% confidence interval
    sorted_sims = sort(sim_sharpes)
    n_sim = length(sorted_sims)
    ci_lo = sorted_sims[max(1, floor(Int, 0.025 * n_sim))]
    ci_hi = sorted_sims[min(n_sim, ceil(Int, 0.975 * n_sim))]

    return MonteCarloResult(obs_sharpe, sim_sharpes, p_val, is_sig,
                             (ci_lo, ci_hi), method)
end

end # module BacktestingEngine
