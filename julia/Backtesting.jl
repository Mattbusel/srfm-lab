"""
    Backtesting

High-performance backtesting engine for the SRFM quantitative trading system.
Implements vectorized event-driven simulation, LARSA signal replication,
walk-forward optimization, Combinatorial Purged Cross-Validation (CPCV),
Deflated Sharpe Ratio, and a transaction cost model.
"""
module Backtesting

using LinearAlgebra
using Statistics
using Distributed

export BacktestConfig, BacktestResult, run_backtest
export larsa_bh_mass, larsa_cf_cross, larsa_hurst_exponent, larsa_quaternion_nav
export walk_forward_optimize
export cpcv_splits, combinatorial_purged_cv
export deflated_sharpe_ratio
export TransactionCostModel, compute_transaction_costs

# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------

"""
    TransactionCostModel

Captures all components of transaction costs for a position change.
"""
struct TransactionCostModel
    commission_bps::Float64      # Commission in basis points
    spread_bps::Float64          # Half bid-ask spread in bps
    market_impact_coeff::Float64 # Almgren-Chriss linear impact coefficient
    slippage_bps::Float64        # Additional slippage in bps
    min_commission::Float64      # Minimum per-trade commission in dollars
end

"""
    TransactionCostModel(; commission_bps=2.0, spread_bps=1.0,
                          market_impact_coeff=0.1, slippage_bps=0.5,
                          min_commission=1.0)

Create a transaction cost model with typical institutional parameters.
"""
function TransactionCostModel(; commission_bps::Real=2.0,
                                spread_bps::Real=1.0,
                                market_impact_coeff::Real=0.1,
                                slippage_bps::Real=0.5,
                                min_commission::Real=1.0)::TransactionCostModel
    return TransactionCostModel(Float64(commission_bps), Float64(spread_bps),
                                 Float64(market_impact_coeff), Float64(slippage_bps),
                                 Float64(min_commission))
end

"""
    compute_transaction_costs(notional, adv, tc_model)

Compute total transaction cost for a trade.

# Arguments
- `notional`  : trade notional in dollars (absolute value)
- `adv`       : average daily volume in dollars
- `tc_model`  : TransactionCostModel

Returns total cost in dollars.
"""
function compute_transaction_costs(notional::Real, adv::Real,
                                    tc_model::TransactionCostModel)::Float64
    commission = max(notional * tc_model.commission_bps * 1e-4,
                     tc_model.min_commission)
    spread_cost = notional * tc_model.spread_bps * 1e-4
    # Almgren-Chriss linear market impact: sigma * (trade_size / ADV)^0.6
    participation = adv > 0.0 ? notional / adv : 1.0
    market_impact = notional * tc_model.market_impact_coeff * participation^0.6
    slippage = notional * tc_model.slippage_bps * 1e-4

    return commission + spread_cost + market_impact + slippage
end

# ---------------------------------------------------------------------------
# Backtest configuration and result
# ---------------------------------------------------------------------------

"""
    BacktestConfig

Configuration for a backtest run.
"""
struct BacktestConfig
    initial_capital::Float64
    tc_model::TransactionCostModel
    max_leverage::Float64
    position_sizing::Symbol     # :fixed, :kelly, :equal_weight, :signal_scaled
    signal_threshold::Float64   # minimum |signal| to trade
    rebalance_freq::Int         # rebalance every N bars
    stop_loss_pct::Float64      # 0.0 disables stop-loss
    take_profit_pct::Float64    # 0.0 disables take-profit
end

"""
    BacktestConfig(; kwargs...)

Create a backtest configuration with sensible defaults.
"""
function BacktestConfig(; initial_capital::Real=1_000_000.0,
                          tc_model::TransactionCostModel=TransactionCostModel(),
                          max_leverage::Real=2.0,
                          position_sizing::Symbol=:signal_scaled,
                          signal_threshold::Real=0.1,
                          rebalance_freq::Int=1,
                          stop_loss_pct::Real=0.05,
                          take_profit_pct::Real=0.0)::BacktestConfig
    return BacktestConfig(Float64(initial_capital), tc_model, Float64(max_leverage),
                          position_sizing, Float64(signal_threshold),
                          rebalance_freq, Float64(stop_loss_pct),
                          Float64(take_profit_pct))
end

"""
    BacktestResult

Stores the output of a backtest run.
"""
struct BacktestResult
    equity_curve::Vector{Float64}
    returns::Vector{Float64}
    positions::Matrix{Float64}     # (T x N) position sizes
    trade_costs::Vector{Float64}
    sharpe_ratio::Float64
    max_drawdown::Float64
    cagr::Float64
    calmar_ratio::Float64
    hit_rate::Float64
    profit_factor::Float64
    num_trades::Int
    turnover_avg::Float64
end

# ---------------------------------------------------------------------------
# Core vectorized backtest loop
# ---------------------------------------------------------------------------

"""
    run_backtest(prices, signals, config; adv=nothing)

Run a vectorized event-driven backtest.

# Arguments
- `prices`  : (T x N) matrix of asset prices
- `signals` : (T x N) matrix of signals in [-1, 1]
- `config`  : BacktestConfig
- `adv`     : (N,) average daily volume per asset; defaults to 1e6

Returns a BacktestResult.
"""
function run_backtest(prices::Matrix{Float64}, signals::Matrix{Float64},
                       config::BacktestConfig;
                       adv::Union{Vector{Float64}, Nothing}=nothing)::BacktestResult
    T, N = size(prices)
    @assert size(signals) == (T, N) "signals must have same shape as prices"

    adv_vec = isnothing(adv) ? fill(1e6, N) : adv

    # Returns
    log_rets = zeros(T - 1, N)
    for t in 2:T
        for n in 1:N
            prices[t-1, n] > 0.0 || continue
            log_rets[t-1, n] = log(prices[t, n] / prices[t-1, n])
        end
    end

    equity = Float64[config.initial_capital]
    positions = zeros(T, N)     # dollar positions
    trade_costs = zeros(T)
    daily_returns = Float64[]

    current_equity = config.initial_capital
    prev_positions = zeros(N)

    for t in 2:T
        # Determine rebalance
        should_rebalance = (t - 1) % config.rebalance_freq == 0

        if should_rebalance
            sig = signals[t - 1, :]

            # Apply threshold
            sig_filtered = ifelse.(abs.(sig) .>= config.signal_threshold, sig, 0.0)

            # Position sizing
            target_pos = compute_target_positions(sig_filtered, current_equity,
                                                   config, prices[t-1, :])

            # Compute trades
            trades = target_pos .- prev_positions

            # Transaction costs
            cost = sum(compute_transaction_costs(abs(trades[n]) * prices[t-1, n],
                                                  adv_vec[n], config.tc_model)
                       for n in 1:N)
            trade_costs[t] = cost
            positions[t, :] = target_pos
            prev_positions = target_pos
        else
            positions[t, :] = prev_positions
        end

        # P&L from price moves
        pnl = dot(prev_positions, log_rets[t-1, :])
        net_pnl = pnl * current_equity - trade_costs[t]

        # Stop-loss check
        if config.stop_loss_pct > 0.0 && pnl < -config.stop_loss_pct
            net_pnl = -config.stop_loss_pct * current_equity - trade_costs[t]
            positions[t, :] .= 0.0
            prev_positions = zeros(N)
        end

        # Take-profit check
        if config.take_profit_pct > 0.0 && pnl > config.take_profit_pct
            net_pnl = config.take_profit_pct * current_equity - trade_costs[t]
            positions[t, :] .= 0.0
            prev_positions = zeros(N)
        end

        current_equity = max(current_equity + net_pnl, 1.0)
        push!(equity, current_equity)
        push!(daily_returns, net_pnl / (current_equity - net_pnl + 1e-10))
    end

    # Compute statistics
    r = daily_returns
    sr = length(r) > 1 ? (mean(r) / (std(r) + 1e-10)) * sqrt(252) : 0.0
    mdd = max_drawdown(equity)
    n_days = length(equity)
    cagr_val = n_days > 1 ? (equity[end] / equity[1])^(252.0 / n_days) - 1.0 : 0.0
    calmar = mdd > 1e-8 ? cagr_val / mdd : 0.0
    hit = length(r) > 0 ? mean(r .> 0.0) : 0.5

    wins = filter(x -> x > 0.0, r)
    losses = filter(x -> x < 0.0, r)
    pf = (!isempty(wins) && !isempty(losses)) ?
         abs(sum(wins) / sum(losses)) : 1.0

    num_trades = sum(any(positions[t, :] .!= positions[max(1, t-1), :]) for t in 2:T)

    # Average turnover
    turnovers = [sum(abs.(positions[t, :] .- positions[t-1, :])) for t in 2:T]
    avg_turnover = isempty(turnovers) ? 0.0 : mean(turnovers)

    return BacktestResult(equity, r, positions, trade_costs, sr, mdd,
                           cagr_val, calmar, hit, pf, num_trades, avg_turnover)
end

"""
    compute_target_positions(signals, equity, config, prices)

Compute target dollar positions from signals.
"""
function compute_target_positions(signals::Vector{Float64}, equity::Real,
                                   config::BacktestConfig,
                                   prices::Vector{Float64})::Vector{Float64}
    N = length(signals)
    if config.position_sizing == :signal_scaled
        # Scale by signal strength, respect max leverage
        total_abs = sum(abs.(signals))
        if total_abs < 1e-10
            return zeros(N)
        end
        weights = signals ./ total_abs
        leverage = min(config.max_leverage, norm(weights, 1))
        return weights .* (equity * leverage)

    elseif config.position_sizing == :equal_weight
        active = abs.(signals) .>= config.signal_threshold
        n_active = sum(active)
        n_active == 0 && return zeros(N)
        pos = zeros(N)
        for i in 1:N
            active[i] || continue
            pos[i] = sign(signals[i]) * equity * config.max_leverage / n_active
        end
        return pos

    elseif config.position_sizing == :kelly
        # Simplified Kelly: f = mu / sigma^2 (requires extra info - use signal as proxy)
        pos = signals .* equity .* 0.25  # conservative Kelly fraction
        # Clip to max leverage
        total_exp = sum(abs.(pos))
        if total_exp > equity * config.max_leverage
            pos .*= equity * config.max_leverage / total_exp
        end
        return pos

    else  # :fixed
        return sign.(signals) .* (equity * config.max_leverage / N)
    end
end

"""
    max_drawdown(equity)

Compute the maximum drawdown of an equity curve.
Returns a positive number representing the fractional drawdown.
"""
function max_drawdown(equity::Vector{Float64})::Float64
    peak = equity[1]
    mdd = 0.0
    for e in equity
        e > peak && (peak = e)
        dd = (peak - e) / peak
        dd > mdd && (mdd = dd)
    end
    return mdd
end

# ---------------------------------------------------------------------------
# LARSA signal replication
# ---------------------------------------------------------------------------

"""
    larsa_bh_mass(prices; window=20)

Compute the Borscht-Hautain (BH) mass indicator used in LARSA signal generation.
Models price action as a mass distribution and computes the effective gravitational
center of mass. Higher mass = stronger trend.

# Arguments
- `prices` : vector of prices
- `window` : lookback window

Returns vector of BH mass values.
"""
function larsa_bh_mass(prices::Vector{Float64}; window::Int=20)::Vector{Float64}
    T = length(prices)
    mass = zeros(T)
    for t in window:T
        w = prices[(t - window + 1):t]
        vol = std(w)
        vol < 1e-10 && continue
        # Normalized price levels as mass weights
        w_norm = (w .- minimum(w)) ./ (maximum(w) - minimum(w) + 1e-10)
        # Center of mass position
        time_idx = collect(1:window)
        com = dot(time_idx, w_norm) / (sum(w_norm) + 1e-10)
        # Mass = concentration relative to midpoint
        mass[t] = (com - window / 2.0) / (window / 2.0)
    end
    return mass
end

"""
    larsa_cf_cross(fast_prices, slow_prices; fast_window=5, slow_window=20)

LARSA Centripetal Force (CF) crossing indicator.
Computes the difference between fast and slow exponentially weighted momentum,
normalized by volatility to give a z-score signal.

Returns a signal vector in approximately [-3, 3].
"""
function larsa_cf_cross(prices::Vector{Float64};
                         fast_window::Int=5, slow_window::Int=20)::Vector{Float64}
    T = length(prices)
    signal = zeros(T)
    fast_alpha = 2.0 / (fast_window + 1.0)
    slow_alpha = 2.0 / (slow_window + 1.0)

    ema_fast = prices[1]
    ema_slow = prices[1]
    ewm_var = 0.0

    for t in 2:T
        ema_fast = fast_alpha * prices[t] + (1.0 - fast_alpha) * ema_fast
        ema_slow = slow_alpha * prices[t] + (1.0 - slow_alpha) * ema_slow
        diff = ema_fast - ema_slow
        ewm_var = 0.94 * ewm_var + 0.06 * diff^2
        vol_adj = sqrt(ewm_var + 1e-10)
        signal[t] = diff / (vol_adj * prices[t] + 1e-10)
    end
    return signal
end

"""
    larsa_hurst_exponent(prices; min_window=10, max_window=100, n_steps=10)

Estimate the Hurst exponent via rescaled-range (R/S) analysis.
H > 0.5 suggests trend-following (persistent); H < 0.5 suggests mean-reversion.

Returns a single Hurst exponent estimate in [0, 1].
"""
function larsa_hurst_exponent(prices::Vector{Float64};
                               min_window::Int=10, max_window::Int=100,
                               n_steps::Int=10)::Float64
    T = length(prices)
    max_window = min(max_window, T ÷ 2)
    min_window < 4 && (min_window = 4)
    max_window <= min_window && return 0.5

    windows = round.(Int, exp.(range(log(min_window), log(max_window), length=n_steps)))
    windows = unique(windows)

    log_n = Float64[]
    log_rs = Float64[]

    for w in windows
        rs_vals = Float64[]
        for start in 1:(T - w + 1)
            sub = prices[start:(start + w - 1)]
            log_sub = log.(max.(sub, 1e-10))
            mean_log = mean(log_sub)
            deviations = cumsum(log_sub .- mean_log)
            R = maximum(deviations) - minimum(deviations)
            S = std(log_sub)
            S > 1e-10 && push!(rs_vals, R / S)
        end
        if !isempty(rs_vals)
            push!(log_n, log(w))
            push!(log_rs, log(mean(rs_vals)))
        end
    end

    length(log_n) < 2 && return 0.5

    # Linear regression: log(R/S) = H * log(n) + const
    X_reg = hcat(ones(length(log_n)), log_n)
    beta = (X_reg' * X_reg) \ (X_reg' * log_rs)
    return clamp(beta[2], 0.0, 1.0)
end

"""
    larsa_quaternion_nav(prices_x, prices_y, prices_z; window=20)

LARSA quaternion navigation signal: model three correlated price series as a
3D trajectory and compute the quaternion rotation rate as a trend strength measure.

Uses simplified quaternion representation: q = [w, x, y, z] where
w = cos(theta/2) and [x,y,z] = sin(theta/2) * rotation_axis.

Returns a scalar trend signal for each time step.
"""
function larsa_quaternion_nav(prices_x::Vector{Float64},
                               prices_y::Vector{Float64},
                               prices_z::Vector{Float64};
                               window::Int=20)::Vector{Float64}
    T = minimum(length.([prices_x, prices_y, prices_z]))
    signal = zeros(T)

    for t in (window + 1):T
        # Extract local window returns
        rx = diff(log.(max.(prices_x[(t - window):t], 1e-10)))
        ry = diff(log.(max.(prices_y[(t - window):t], 1e-10)))
        rz = diff(log.(max.(prices_z[(t - window):t], 1e-10)))

        # 3D velocity vector (cumulative log returns = log-price change direction)
        vx = sum(rx)
        vy = sum(ry)
        vz = sum(rz)

        # Quaternion rotation: compute rotation from early to late half
        half = window ÷ 2
        v_early = [sum(rx[1:half]), sum(ry[1:half]), sum(rz[1:half])]
        v_late  = [sum(rx[(half+1):end]), sum(ry[(half+1):end]), sum(rz[(half+1):end])]

        n_early = norm(v_early)
        n_late  = norm(v_late)
        n_early < 1e-10 || n_late < 1e-10 && continue

        v_early_hat = v_early ./ n_early
        v_late_hat  = v_late  ./ n_late

        # Quaternion dot product = cos(angle/2)
        cos_angle = clamp(dot(v_early_hat, v_late_hat), -1.0, 1.0)

        # Cross product magnitude = sin(angle)
        cross_prod = [v_early_hat[2]*v_late_hat[3] - v_early_hat[3]*v_late_hat[2],
                      v_early_hat[3]*v_late_hat[1] - v_early_hat[1]*v_late_hat[3],
                      v_early_hat[1]*v_late_hat[2] - v_early_hat[2]*v_late_hat[1]]
        sin_angle = norm(cross_prod)

        # Rotation rate = angle = atan(sin, cos)
        rotation_angle = atan(sin_angle, cos_angle)

        # Sign from main price direction
        direction = sign(vx + vy + vz)

        signal[t] = direction * rotation_angle / pi  # normalize to [-1, 1]
    end
    return signal
end

# ---------------------------------------------------------------------------
# Walk-forward optimization
# ---------------------------------------------------------------------------

"""
    walk_forward_optimize(prices, signal_generator, param_grid, config;
                           train_periods=252, test_periods=63, step_size=21)

Walk-forward optimization with parallel evaluation.

# Arguments
- `prices`           : (T x N) price matrix
- `signal_generator` : function(prices, params) -> (T x N) signals
- `param_grid`       : vector of parameter sets to evaluate
- `config`           : BacktestConfig
- `train_periods`    : number of bars for training window
- `test_periods`     : number of bars for out-of-sample test
- `step_size`        : step between walk-forward windows

Returns vector of (params, out_of_sample_result) tuples per window.
"""
function walk_forward_optimize(prices::Matrix{Float64},
                                 signal_generator::Function,
                                 param_grid::Vector,
                                 config::BacktestConfig;
                                 train_periods::Int=252,
                                 test_periods::Int=63,
                                 step_size::Int=21)
    T = size(prices, 1)
    results = []

    t_start = train_periods + 1
    while t_start + test_periods - 1 <= T
        train_idx = (t_start - train_periods):(t_start - 1)
        test_idx  = t_start:(t_start + test_periods - 1)

        prices_train = prices[train_idx, :]
        prices_test  = prices[test_idx, :]

        # Evaluate all parameter sets on training data
        train_scores = map(param_grid) do params
            sigs = signal_generator(prices_train, params)
            result = run_backtest(prices_train, sigs, config)
            return result.sharpe_ratio
        end

        # Select best parameters
        best_idx = argmax(train_scores)
        best_params = param_grid[best_idx]

        # Evaluate on test data
        sigs_test = signal_generator(prices_test, best_params)
        oos_result = run_backtest(prices_test, sigs_test, config)

        push!(results, (params=best_params, oos_sharpe=oos_result.sharpe_ratio,
                        oos_result=oos_result, train_sharpe=train_scores[best_idx],
                        test_start=t_start, test_end=t_start + test_periods - 1))

        t_start += step_size
    end

    return results
end

# ---------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation (CPCV)
# ---------------------------------------------------------------------------

"""
    cpcv_splits(T, n_groups, n_test_groups; embargo_pct=0.01)

Generate train/test splits for Combinatorial Purged Cross-Validation (CPCV)
as described by Lopez de Prado (2018).

# Arguments
- `T`             : total number of observations
- `n_groups`      : number of groups to split data into (phi)
- `n_test_groups` : number of groups used for testing per split (k)
- `embargo_pct`   : fraction of observations to embargo after each test set

Returns a vector of (train_idx, test_idx) pairs.
"""
function cpcv_splits(T::Int, n_groups::Int=6, n_test_groups::Int=2;
                      embargo_pct::Real=0.01)
    group_size = T ÷ n_groups
    embargo = max(1, round(Int, T * embargo_pct))

    # Create group boundaries
    groups = [(i * group_size + 1):min((i + 1) * group_size, T) for i in 0:(n_groups-1)]

    # All combinations of n_test_groups from n_groups
    splits = []
    combo_indices = collect(combinations_generator(1:n_groups, n_test_groups))

    for test_groups in combo_indices
        test_idx = Int[]
        for g in test_groups
            append!(test_idx, groups[g])
        end
        sort!(test_idx)

        # Train: all groups not in test, with embargo
        train_idx = Int[]
        for g in 1:n_groups
            g in test_groups && continue
            g_range = groups[g]
            # Check embargo: skip observations just before test group
            for idx in g_range
                # Purge: skip if idx is within embargo of any test index
                too_close = any(abs(idx - t_idx) <= embargo for t_idx in test_idx)
                too_close || push!(train_idx, idx)
            end
        end

        !isempty(train_idx) && !isempty(test_idx) && push!(splits, (train=train_idx, test=test_idx))
    end

    return splits
end

"""
    combinations_generator(items, k)

Generate all k-combinations of items (non-recursive).
"""
function combinations_generator(items, k::Int)
    n = length(items)
    k > n && return []
    result = []
    combo = collect(1:k)

    while true
        push!(result, items[combo])
        # Find rightmost element that can be incremented
        i = k
        while i >= 1 && combo[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        combo[i] += 1
        for j in (i+1):k
            combo[j] = combo[j-1] + 1
        end
    end
    return result
end

"""
    combinatorial_purged_cv(prices, signals_fn, param_grid, config;
                             n_groups=6, n_test_groups=2)

Run Combinatorial Purged Cross-Validation to evaluate strategy robustness.

Returns a distribution of out-of-sample Sharpe ratios.
"""
function combinatorial_purged_cv(prices::Matrix{Float64},
                                   signals_fn::Function,
                                   param_grid::Vector,
                                   config::BacktestConfig;
                                   n_groups::Int=6,
                                   n_test_groups::Int=2)
    T = size(prices, 1)
    splits = cpcv_splits(T, n_groups, n_test_groups)

    oos_sharpes = Float64[]

    for split in splits
        train_prices = prices[split.train, :]
        test_prices  = prices[split.test,  :]

        # Evaluate params on training split
        train_scores = [run_backtest(train_prices,
                                      signals_fn(train_prices, p), config).sharpe_ratio
                        for p in param_grid]

        best_params = param_grid[argmax(train_scores)]

        # Out-of-sample
        oos_sigs = signals_fn(test_prices, best_params)
        oos_result = run_backtest(test_prices, oos_sigs, config)
        push!(oos_sharpes, oos_result.sharpe_ratio)
    end

    return (oos_sharpes=oos_sharpes,
            mean_oos_sharpe=mean(oos_sharpes),
            std_oos_sharpe=std(oos_sharpes),
            pct_positive=mean(oos_sharpes .> 0.0))
end

# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

"""
    deflated_sharpe_ratio(sharpe_ratios, n_obs; sr_benchmark=0.0)

Compute the Deflated Sharpe Ratio (DSR) test statistic per Lopez de Prado (2014).
Accounts for multiple testing, non-normality, and backtest overfitting.

# Arguments
- `sharpe_ratios` : vector of Sharpe ratios from multiple strategy trials
- `n_obs`         : number of observations per trial
- `sr_benchmark`  : minimum acceptable Sharpe ratio (default 0)

Returns named tuple (dsr, pvalue, sr_star, sr_max).
"""
function deflated_sharpe_ratio(sharpe_ratios::Vector{Float64}, n_obs::Int;
                                 sr_benchmark::Real=0.0)
    N_trials = length(sharpe_ratios)
    sr_max = maximum(sharpe_ratios)

    # Expected maximum of N iid standard normal draws
    # E[max(Z_1,...,Z_N)] ~= (1 - gamma) * quantile(N(0,1), 1 - 1/N) + gamma * quantile(N(0,1), 1 - 1/(N*e))
    # Simplified: use Euler-Mascheroni approximation
    if N_trials > 1
        gamma_em = 0.5772156649
        mean_max = (1.0 - gamma_em) * quantile(Normal(), 1.0 - 1.0 / N_trials) +
                   gamma_em * quantile(Normal(), 1.0 - 1.0 / (N_trials * exp(1.0)))
    else
        mean_max = 0.0
    end

    # Variance of maximum (approximation)
    var_max = N_trials > 1 ?
              quantile(Normal(), 1.0 - 1.0 / N_trials)^2 * (1.0 - (1.0 - 1.0/N_trials)) /
              (N_trials * pdf(Normal(), quantile(Normal(), 1.0 - 1.0 / N_trials))^2) : 1.0
    var_max = max(var_max, 1e-8)

    # Adjust benchmark for selection bias
    sr_star = sr_benchmark + sqrt(var_max) * mean_max

    # Skewness and kurtosis correction (Opdyke 2007)
    # Using sample moments of the SR distribution
    skew_sr = skewness_sr(sr_max, n_obs)
    kurt_sr = kurtosis_sr(sr_max, n_obs)

    # Variance of SR estimate
    sr_var = (1.0 + 0.5 * sr_max^2 - skew_sr * sr_max + (kurt_sr - 1.0) / 4.0 * sr_max^2) / n_obs

    # Test statistic: is SR significantly greater than sr_star?
    z_stat = (sr_max - sr_star) / sqrt(max(sr_var, 1e-10))
    pval = 1.0 - cdf(Normal(), z_stat)

    # DSR: probability that the best SR exceeds benchmark
    dsr = cdf(Normal(), z_stat)

    return (dsr=dsr, pvalue=pval, sr_star=sr_star, sr_max=sr_max,
            z_statistic=z_stat, n_trials=N_trials)
end

"""
    skewness_sr(sr, n)

Approximate skewness of SR sampling distribution.
"""
function skewness_sr(sr::Real, n::Int)::Float64
    return (6.0 * sr^3 - 3.0 * sr) / sqrt(n)
end

"""
    kurtosis_sr(sr, n)

Approximate kurtosis correction for SR distribution.
"""
function kurtosis_sr(sr::Real, n::Int)::Float64
    return 3.0 + (12.0 * sr^4 - 24.0 * sr^2 + 6.0) / n
end

# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

"""
    compute_performance_metrics(returns; risk_free_rate=0.0, periods_per_year=252)

Compute comprehensive performance metrics from a return series.
"""
function compute_performance_metrics(returns::Vector{Float64};
                                      risk_free_rate::Real=0.0,
                                      periods_per_year::Int=252)
    n = length(returns)
    n < 2 && return nothing

    mu = mean(returns)
    sigma = std(returns)
    s = skewness_stat_bt(returns)
    k = kurtosis_stat_bt(returns)

    ann_return = (1.0 + mu)^periods_per_year - 1.0
    ann_vol    = sigma * sqrt(periods_per_year)
    sharpe     = ann_vol > 1e-10 ? (ann_return - risk_free_rate) / ann_vol : 0.0

    # Sortino ratio: downside deviation
    downside = filter(x -> x < risk_free_rate / periods_per_year, returns)
    downside_vol = isempty(downside) ? 1e-10 : std(downside) * sqrt(periods_per_year)
    sortino = downside_vol > 1e-10 ? (ann_return - risk_free_rate) / downside_vol : 0.0

    # Max drawdown
    cum_rets = cumprod(1.0 .+ returns)
    equity = vcat(1.0, cum_rets)
    mdd = max_drawdown(equity)

    # Information ratio (vs zero benchmark here)
    ir = ann_vol > 1e-10 ? ann_return / ann_vol : 0.0

    return (
        annualized_return=ann_return, annualized_vol=ann_vol,
        sharpe_ratio=sharpe, sortino_ratio=sortino,
        max_drawdown=mdd, calmar_ratio=mdd > 1e-8 ? ann_return / mdd : 0.0,
        skewness=s, excess_kurtosis=k,
        n_observations=n, hit_rate=mean(returns .> 0.0)
    )
end

function skewness_stat_bt(x::Vector{Float64})::Float64
    n = length(x)
    mu = mean(x); sigma = std(x)
    sigma < 1e-14 && return 0.0
    return sum((xi - mu)^3 for xi in x) / (n * sigma^3)
end

function kurtosis_stat_bt(x::Vector{Float64})::Float64
    n = length(x)
    mu = mean(x); sigma = std(x)
    sigma < 1e-14 && return 0.0
    return sum((xi - mu)^4 for xi in x) / (n * sigma^4) - 3.0
end

end  # module Backtesting
