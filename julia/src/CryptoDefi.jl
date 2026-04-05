module CryptoDefi

# ============================================================
# CryptoDefi.jl — DeFi mechanics, AMM math, liquidity, yield
# ============================================================
# Pure stdlib Julia. Covers:
#   - Constant-product AMM (Uniswap v2 style)
#   - Concentrated liquidity (Uniswap v3 style)
#   - Impermanent loss computation
#   - Liquidity provision analytics
#   - Yield optimization and APR/APY conversions
#   - Stablecoin curve (StableSwap invariant)
#   - Fee tier analysis
#   - MEV / sandwich attack estimation
#   - Rebalancing strategies
# ============================================================

using Statistics, LinearAlgebra

export AMMPool, UniV3Position, StablePool
export swap_output, swap_price_impact, add_liquidity, remove_liquidity
export impermanent_loss, il_breakeven_fee
export concentrated_liquidity_range, tick_to_price, price_to_tick
export stableswap_output, stableswap_dy
export yield_to_apr, apr_to_apy, continuous_to_apy
export optimal_rebalance_threshold, hodl_vs_lp_comparison
export fee_revenue_estimate, capital_efficiency_ratio
export sandwich_attack_profit, mev_exposure_estimate
export portfolio_defi_weights, defi_risk_score
export simulate_lp_returns, bootstrap_lp_sharpe

# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

"""Constant-product AMM pool (x * y = k)."""
struct AMMPool
    reserve_x::Float64   # token X reserve
    reserve_y::Float64   # token Y reserve
    fee_rate::Float64    # e.g. 0.003 for 30 bps
end

"""Uniswap v3 concentrated liquidity position."""
struct UniV3Position
    tick_lower::Int
    tick_upper::Int
    liquidity::Float64
    fee_tier::Float64    # 0.0005, 0.003, or 0.01
end

"""StableSwap (Curve-style) pool."""
struct StablePool
    reserves::Vector{Float64}  # n-token reserves
    amplification::Float64     # A parameter (e.g. 100–2000)
    fee_rate::Float64
end

# ──────────────────────────────────────────────────────────────
# Constant-product AMM mechanics
# ──────────────────────────────────────────────────────────────

"""
    swap_output(pool, dx, buy_y) -> (dy, new_pool)

Compute output from a swap and return updated pool.
`buy_y=true` means spending X to receive Y.
"""
function swap_output(pool::AMMPool, dx::Float64, buy_y::Bool=true)
    k = pool.reserve_x * pool.reserve_y
    fee = pool.fee_rate
    if buy_y
        dx_eff = dx * (1.0 - fee)
        new_x = pool.reserve_x + dx_eff
        new_y = k / new_x
        dy = pool.reserve_y - new_y
        new_pool = AMMPool(pool.reserve_x + dx, new_y, fee)
        return dy, new_pool
    else
        dy_eff = dx * (1.0 - fee)
        new_y = pool.reserve_y + dy_eff
        new_x = k / new_y
        dx_out = pool.reserve_x - new_x
        new_pool = AMMPool(new_x, pool.reserve_y + dx, fee)
        return dx_out, new_pool
    end
end

"""
    swap_price_impact(pool, dx) -> price_impact_fraction

Estimate price impact of a swap relative to mid-price.
"""
function swap_price_impact(pool::AMMPool, dx::Float64)
    mid_price = pool.reserve_y / pool.reserve_x
    dy, _ = swap_output(pool, dx, true)
    exec_price = dy / dx
    return abs(exec_price - mid_price) / mid_price
end

"""
    spot_price(pool) -> price of Y in terms of X
"""
spot_price(pool::AMMPool) = pool.reserve_y / pool.reserve_x

"""
    add_liquidity(pool, amount_x) -> (amount_y_required, lp_tokens, new_pool)

Add liquidity at current ratio.
"""
function add_liquidity(pool::AMMPool, amount_x::Float64)
    ratio = pool.reserve_y / pool.reserve_x
    amount_y = amount_x * ratio
    total_lp = sqrt(pool.reserve_x * pool.reserve_y)
    new_lp = sqrt((pool.reserve_x + amount_x) * (pool.reserve_y + amount_y)) - total_lp
    new_pool = AMMPool(pool.reserve_x + amount_x, pool.reserve_y + amount_y, pool.fee_rate)
    return amount_y, new_lp, new_pool
end

"""
    remove_liquidity(pool, lp_fraction) -> (amount_x, amount_y, new_pool)

Withdraw proportional share of the pool.
"""
function remove_liquidity(pool::AMMPool, lp_fraction::Float64)
    ax = pool.reserve_x * lp_fraction
    ay = pool.reserve_y * lp_fraction
    new_pool = AMMPool(pool.reserve_x - ax, pool.reserve_y - ay, pool.fee_rate)
    return ax, ay, new_pool
end

# ──────────────────────────────────────────────────────────────
# Impermanent loss
# ──────────────────────────────────────────────────────────────

"""
    impermanent_loss(price_ratio) -> IL fraction

Impermanent loss as a fraction of hold value given price_ratio = p1/p0.
IL = 2√r/(1+r) - 1
"""
function impermanent_loss(price_ratio::Float64)
    r = price_ratio
    return 2.0 * sqrt(r) / (1.0 + r) - 1.0
end

"""
    impermanent_loss_dollar(initial_value, price_ratio) -> IL in dollar terms
"""
function impermanent_loss_dollar(initial_value::Float64, price_ratio::Float64)
    return initial_value * abs(impermanent_loss(price_ratio))
end

"""
    il_breakeven_fee(price_ratio, trading_volume_fraction) -> required_fee_rate

Minimum fee rate needed to break even on IL given volume as fraction of TVL per day.
"""
function il_breakeven_fee(price_ratio::Float64, daily_volume_over_tvl::Float64)
    il = abs(impermanent_loss(price_ratio))
    # fees = fee_rate * volume; need fees >= il * tvl
    return il / daily_volume_over_tvl
end

"""
    il_curve(price_ratios) -> Vector of IL values
"""
function il_curve(price_ratios::Vector{Float64})
    return [impermanent_loss(r) for r in price_ratios]
end

# ──────────────────────────────────────────────────────────────
# Uniswap v3 concentrated liquidity
# ──────────────────────────────────────────────────────────────

const TICK_BASE = 1.0001

"""
    tick_to_price(tick) -> price

Convert a Uniswap v3 tick integer to price.
"""
tick_to_price(tick::Int) = TICK_BASE ^ tick

"""
    price_to_tick(price) -> tick

Convert a price to the nearest v3 tick.
"""
price_to_tick(price::Float64) = round(Int, log(price) / log(TICK_BASE))

"""
    concentrated_liquidity_range(pa, pb, current_price, liquidity)

Compute token amounts for a concentrated position between pa and pb.
Returns (amount_x, amount_y) of tokens deposited.
"""
function concentrated_liquidity_range(pa::Float64, pb::Float64,
                                       current_price::Float64, liquidity::Float64)
    p = current_price
    if p <= pa
        # All in token X
        amount_x = liquidity * (1.0/sqrt(pa) - 1.0/sqrt(pb))
        amount_y = 0.0
    elseif p >= pb
        # All in token Y
        amount_x = 0.0
        amount_y = liquidity * (sqrt(pb) - sqrt(pa))
    else
        amount_x = liquidity * (1.0/sqrt(p) - 1.0/sqrt(pb))
        amount_y = liquidity * (sqrt(p) - sqrt(pa))
    end
    return amount_x, amount_y
end

"""
    capital_efficiency_ratio(pa, pb, current_price) -> ratio vs full-range

How much more capital efficient v3 position is vs v2.
"""
function capital_efficiency_ratio(pa::Float64, pb::Float64, current_price::Float64)
    # CE = 1 / (1 - (pa/pb)^(1/4))  (approximate)
    r = (pa / pb)^0.25
    return 1.0 / (1.0 - r)
end

"""
    v3_fee_per_unit_liquidity(volume, fee_rate, liquidity_in_range, total_liquidity)

Estimate fee income for a v3 position.
"""
function v3_fee_per_unit_liquidity(volume::Float64, fee_rate::Float64,
                                    liquidity_in_range::Float64, total_liquidity::Float64)
    total_fees = volume * fee_rate
    share = liquidity_in_range / max(total_liquidity, 1e-12)
    return total_fees * share
end

# ──────────────────────────────────────────────────────────────
# StableSwap (Curve invariant)
# ──────────────────────────────────────────────────────────────

"""
    stableswap_invariant(reserves, A) -> D

Compute the Curve StableSwap invariant D.
Uses Newton's method to solve the implicit equation.
"""
function stableswap_invariant(reserves::Vector{Float64}, A::Float64)
    n = length(reserves)
    S = sum(reserves)
    D = S
    Ann = A * n^n
    for _ in 1:256
        D_P = D
        for xi in reserves
            D_P = D_P * D / (n * xi + 1e-18)
        end
        D_prev = D
        D = (Ann * S + n * D_P) * D / ((Ann - 1) * D + (n + 1) * D_P)
        if abs(D - D_prev) < 1e-12
            break
        end
    end
    return D
end

"""
    stableswap_output(pool, i, j, dx) -> dy

Compute output when swapping dx of token i for token j.
"""
function stableswap_output(pool::StablePool, i::Int, j::Int, dx::Float64)
    n = length(pool.reserves)
    A = pool.amplification
    Ann = A * n^n
    dx_eff = dx * (1.0 - pool.fee_rate)
    new_reserves = copy(pool.reserves)
    new_reserves[i] += dx_eff
    D = stableswap_invariant(pool.reserves, A)
    # Solve for new y (reserves[j])
    S_ = sum(new_reserves) - new_reserves[j]
    c = D
    for k in 1:n
        if k == j; continue; end
        c = c * D / (n * new_reserves[k])
    end
    c = c * D / (n * Ann)
    b = S_ + D / Ann
    y = D
    for _ in 1:256
        y_prev = y
        y = (y^2 + c) / (2*y + b - D)
        if abs(y - y_prev) < 1e-12
            break
        end
    end
    dy = pool.reserves[j] - y
    return max(dy, 0.0)
end

"""
    stableswap_dy(pool, i, j, dx) -> (dy_net, fee_collected)

Net output and fee for a StableSwap trade.
"""
function stableswap_dy(pool::StablePool, i::Int, j::Int, dx::Float64)
    dy_gross = stableswap_output(pool, i, j, dx / (1.0 - pool.fee_rate))
    dy_net = stableswap_output(pool, i, j, dx)
    fee = dy_gross - dy_net
    return dy_net, fee
end

# ──────────────────────────────────────────────────────────────
# Yield analytics
# ──────────────────────────────────────────────────────────────

"""
    yield_to_apr(yield_per_period, periods_per_year) -> APR
"""
yield_to_apr(y::Float64, n::Float64) = y * n

"""
    apr_to_apy(apr, compounding_periods) -> APY

Convert APR to APY with specified compounding frequency.
"""
apr_to_apy(apr::Float64, n::Float64) = (1.0 + apr / n)^n - 1.0

"""
    continuous_to_apy(r_continuous) -> APY
"""
continuous_to_apy(r::Float64) = exp(r) - 1.0

"""
    apy_to_continuous(apy) -> continuous rate
"""
apy_to_continuous(apy::Float64) = log(1.0 + apy)

"""
    compound_yield(principal, apy, days) -> final_value
"""
compound_yield(P::Float64, apy::Float64, days::Float64) = P * (1.0 + apy)^(days/365.0)

"""
    fee_revenue_estimate(tvl, daily_volume_ratio, fee_rate) -> daily_fee_revenue
"""
fee_revenue_estimate(tvl::Float64, vol_ratio::Float64, fee::Float64) = tvl * vol_ratio * fee

# ──────────────────────────────────────────────────────────────
# Rebalancing and strategy analytics
# ──────────────────────────────────────────────────────────────

"""
    optimal_rebalance_threshold(sigma, transaction_cost) -> threshold

Estimate optimal drift before rebalancing a 50/50 LP position.
Based on proportional transaction cost vs drift variance tradeoff.
"""
function optimal_rebalance_threshold(sigma::Float64, transaction_cost::Float64)
    # Simplified: threshold ∝ sqrt(2 * tx_cost / sigma^2)
    return sqrt(2.0 * transaction_cost / (sigma^2))
end

"""
    hodl_vs_lp_comparison(p0, p1, initial_value, fee_income)

Compare HODL strategy vs LP over a price move from p0 to p1.
Returns (hodl_value, lp_value, lp_advantage).
"""
function hodl_vs_lp_comparison(p0::Float64, p1::Float64,
                                 initial_value::Float64, fee_income::Float64)
    # Assume 50/50 initial split: V0/2 in X (x0 units) and V0/2 in Y
    x0 = (initial_value / 2.0) / p0
    y0 = initial_value / 2.0
    hodl_value = x0 * p1 + y0
    # LP value with IL
    r = p1 / p0
    lp_value = initial_value * (2.0 * sqrt(r) / (1.0 + r)) + fee_income
    return hodl_value, lp_value, lp_value - hodl_value
end

# ──────────────────────────────────────────────────────────────
# MEV and sandwich attack estimation
# ──────────────────────────────────────────────────────────────

"""
    sandwich_attack_profit(pool, victim_dx, attacker_dx_front) -> profit

Estimate profit of a sandwich attack.
Front-run with attacker_dx_front, victim executes victim_dx,
then attacker back-runs.
"""
function sandwich_attack_profit(pool::AMMPool, victim_dx::Float64, attacker_dx_front::Float64)
    # Front-run: attacker buys Y
    dy_front, pool_after_front = swap_output(pool, attacker_dx_front, true)
    # Victim swap
    _, pool_after_victim = swap_output(pool_after_front, victim_dx, true)
    # Back-run: attacker sells Y back
    dx_back, _ = swap_output(pool_after_victim, dy_front, false)
    profit = dx_back - attacker_dx_front
    return max(profit, 0.0)
end

"""
    mev_exposure_estimate(pool, expected_order_flow, attacker_capital_fraction) -> daily_mev

Rough estimate of MEV extracted per day from a pool.
"""
function mev_exposure_estimate(pool::AMMPool, daily_volume::Float64,
                                 attacker_capital_fraction::Float64)
    tvl = pool.reserve_x * spot_price(pool) + pool.reserve_y
    attacker_capital = tvl * attacker_capital_fraction
    # Expected sandwich profit scales with order_flow * price_impact
    avg_order = daily_volume / 100.0  # assume 100 orders/day
    avg_impact = swap_price_impact(pool, avg_order)
    mev_per_order = attacker_capital * avg_impact * 0.5
    return mev_per_order * 100.0
end

# ──────────────────────────────────────────────────────────────
# Portfolio-level DeFi analytics
# ──────────────────────────────────────────────────────────────

"""
    portfolio_defi_weights(apys, vols, lambda) -> weights

Mean-variance optimal allocation across DeFi protocols.
lambda = risk aversion parameter.
"""
function portfolio_defi_weights(apys::Vector{Float64}, vols::Vector{Float64}, lambda::Float64)
    n = length(apys)
    # Diagonal covariance (independent protocols)
    inv_cov = diagm(1.0 ./ (vols .^ 2))
    w = (1.0 / lambda) .* inv_cov * apys
    w = max.(w, 0.0)
    s = sum(w)
    return s > 0 ? w ./ s : fill(1.0/n, n)
end

"""
    defi_risk_score(protocol_tvl, audit_count, age_days, hack_history) -> score 0-100

Composite risk score for a DeFi protocol. Lower is safer.
"""
function defi_risk_score(tvl::Float64, audit_count::Int,
                           age_days::Int, hack_history::Bool)
    score = 100.0
    # TVL signal: larger TVL = lower risk
    score -= min(30.0, log10(max(tvl, 1.0)) * 5.0)
    # Audits: more audits = lower risk
    score -= min(20.0, audit_count * 5.0)
    # Age: older = lower risk
    score -= min(30.0, age_days / 30.0)
    # Hack history: penalize
    if hack_history; score += 25.0; end
    return clamp(score, 0.0, 100.0)
end

# ──────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────

"""
    simulate_lp_returns(p0, mu, sigma, T, fee_rate, vol_ratio, n_paths, n_steps)

Simulate LP returns vs HODL over T years using GBM price paths.
Returns (lp_returns, hodl_returns, il_series).
"""
function simulate_lp_returns(p0::Float64, mu::Float64, sigma::Float64,
                               T::Float64, fee_rate::Float64, vol_ratio::Float64,
                               n_paths::Int=1000, n_steps::Int=252)
    dt = T / n_steps
    lp_returns = zeros(n_paths)
    hodl_returns = zeros(n_paths)
    il_series = zeros(n_paths, n_steps)

    rng_state = 42
    function next_randn()
        # Simple LCG + Box-Muller for stdlib-only normal samples
        rng_state = (1664525 * rng_state + 1013904223) % 2^32
        u1 = rng_state / 2^32
        rng_state = (1664525 * rng_state + 1013904223) % 2^32
        u2 = rng_state / 2^32
        return sqrt(-2.0 * log(u1 + 1e-15)) * cos(2π * u2)
    end

    for i in 1:n_paths
        price = p0
        cumulative_fees = 0.0
        initial_value = 2.0 * p0  # 1 unit X + p0 units Y (arbitrary)
        hodl_x = 1.0
        hodl_y = p0

        for t in 1:n_steps
            z = next_randn()
            price *= exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z)
            # Daily fee income
            tvl = 2.0 * sqrt(p0 * price) * initial_value  # approx
            daily_fees = tvl * vol_ratio * fee_rate / 365.0
            cumulative_fees += daily_fees
            il_series[i, t] = impermanent_loss(price / p0)
        end

        hodl_value = hodl_x * price + hodl_y
        lp_value = initial_value * (2.0 * sqrt(price / p0) / (1.0 + price/p0)) + cumulative_fees
        lp_returns[i] = (lp_value - initial_value) / initial_value
        hodl_returns[i] = (hodl_value - initial_value) / initial_value
    end

    return lp_returns, hodl_returns, il_series
end

"""
    bootstrap_lp_sharpe(lp_returns, n_boot) -> (mean_sharpe, ci_lower, ci_upper)

Bootstrap confidence interval for LP Sharpe ratio.
"""
function bootstrap_lp_sharpe(lp_returns::Vector{Float64}, n_boot::Int=1000)
    n = length(lp_returns)
    sharpes = zeros(n_boot)
    for b in 1:n_boot
        idx = [mod(b * 1664525 + i * 1013904223, n) + 1 for i in 1:n]
        sample = lp_returns[idx]
        m = mean(sample)
        s = std(sample)
        sharpes[b] = s > 0 ? m / s * sqrt(252.0) : 0.0
    end
    sort!(sharpes)
    lo = sharpes[round(Int, 0.025 * n_boot)]
    hi = sharpes[round(Int, 0.975 * n_boot)]
    return mean(sharpes), lo, hi
end

# ──────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────

"""
    price_range_utilization(pa, pb, current_price) -> fraction in range

What fraction of time (uniform random walk) does price spend in [pa, pb]?
Approximation using log-normal coverage assumption.
"""
function price_range_utilization(pa::Float64, pb::Float64, current_price::Float64,
                                  sigma_annual::Float64, horizon_days::Float64)
    lp = log(pa / current_price)
    lh = log(pb / current_price)
    sigma_h = sigma_annual * sqrt(horizon_days / 365.0)
    # Probability current_price stays in [pa, pb] — rough normal approximation
    z_lo = lp / sigma_h
    z_hi = lh / sigma_h
    # Using error function approximation
    erf_approx(x) = tanh(1.2533141373 * x)
    return 0.5 * (erf_approx(z_hi / sqrt(2)) - erf_approx(z_lo / sqrt(2)))
end

"""
    tvl_weighted_apy(pools, tvls) -> weighted_apy

Volume-weighted average APY across a set of pools.
"""
function tvl_weighted_apy(apys::Vector{Float64}, tvls::Vector{Float64})
    total = sum(tvls)
    return total > 0 ? dot(apys, tvls) / total : 0.0
end


# ============================================================
# SECTION 2: LENDING PROTOCOL ANALYTICS
# ============================================================

struct LendingMarket
    asset::String
    total_supply::Float64
    total_borrow::Float64
    base_rate::Float64
    slope1::Float64
    slope2::Float64
    optimal_utilization::Float64
    liquidation_threshold::Float64
    liquidation_bonus::Float64
end

function utilization_rate(market::LendingMarket)
    return market.total_borrow / (market.total_supply + 1e-10)
end

function borrow_rate(market::LendingMarket)
    u = utilization_rate(market)
    u_opt = market.optimal_utilization
    if u <= u_opt
        return market.base_rate + market.slope1 * u / (u_opt + 1e-10)
    else
        excess = (u - u_opt) / (1 - u_opt + 1e-10)
        return market.base_rate + market.slope1 + market.slope2 * excess
    end
end

function supply_rate(market::LendingMarket; reserve_factor::Float64=0.1)
    u = utilization_rate(market)
    br = borrow_rate(market)
    return br * u * (1 - reserve_factor)
end

function health_factor_aave(collateral_value::Float64, debt_value::Float64,
                              liquidation_threshold::Float64)
    return (collateral_value * liquidation_threshold) / (debt_value + 1e-10)
end

function max_ltv_borrow(collateral_value::Float64, current_debt::Float64,
                          ltv::Float64)
    max_debt = collateral_value * ltv
    return max(max_debt - current_debt, 0.0)
end

function liquidation_profit(collateral_seized_value::Float64, debt_repaid::Float64,
                              liquidation_bonus::Float64)
    profit = collateral_seized_value * (1 + liquidation_bonus) - debt_repaid
    return (profit=profit, profit_pct=profit/(debt_repaid+1e-10))
end

function simulate_lending_tvl(initial_supply::Float64, n_periods::Int=365;
                               deposit_rate::Float64=0.02,
                               withdrawal_rate::Float64=0.015,
                               yield_boost::Float64=0.05)
    supply = initial_supply; tvl_series = [supply]
    for _ in 1:n_periods
        net_flow = supply * (deposit_rate - withdrawal_rate) / 365 +
                   supply * yield_boost / 365 * randn()
        supply = max(supply + net_flow, 0.0)
        push!(tvl_series, supply)
    end
    return tvl_series
end

function compound_interest_rate(apy::Float64; compounding_periods::Int=365)
    return (1 + apy)^(1/compounding_periods) - 1
end

function effective_borrowing_cost(nominal_rate::Float64, origination_fee::Float64,
                                    loan_duration_years::Float64)
    return nominal_rate + origination_fee / (loan_duration_years + 1e-10)
end

# ============================================================
# SECTION 3: MEV & FLASH LOAN ANALYTICS
# ============================================================

struct FlashLoanArb
    borrow_asset::String
    borrow_amount::Float64
    flash_loan_fee::Float64
    path::Vector{String}
    expected_profit::Float64
    net_profit::Float64
    is_profitable::Bool
end

function flash_loan_arbitrage(borrow_amount::Float64,
                               price_a_exchange1::Float64,
                               price_a_exchange2::Float64,
                               fee_rate::Float64=0.0009,
                               flash_fee_rate::Float64=0.0009)
    flash_cost = borrow_amount * flash_fee_rate
    # Buy cheap, sell expensive
    if price_a_exchange1 < price_a_exchange2
        buy_cost  = borrow_amount * price_a_exchange1 * (1 + fee_rate)
        sell_rev  = borrow_amount * price_a_exchange2 * (1 - fee_rate)
    else
        buy_cost  = borrow_amount * price_a_exchange2 * (1 + fee_rate)
        sell_rev  = borrow_amount * price_a_exchange1 * (1 - fee_rate)
    end
    gross = sell_rev - buy_cost
    net   = gross - flash_cost
    return FlashLoanArb("USDC", borrow_amount, flash_cost,
                         ["buy","swap","sell"], gross, net, net > 0)
end

function sandwich_attack_profit(victim_size_usd::Float64, pool_liquidity::Float64,
                                  swap_fee::Float64=0.003)
    # Simplified sandwich MEV model
    price_impact = victim_size_usd / (pool_liquidity + 1e-10)
    frontrun_size = victim_size_usd * 0.5  # attacker buys same size
    frontrun_cost = frontrun_size * swap_fee
    price_after_victim = price_impact * 2  # doubled after both trades
    gross_profit = frontrun_size * price_after_victim
    net_profit   = gross_profit - 2*frontrun_cost  # two gas txs
    return (gross=gross_profit, net=net_profit, profitable=(net_profit > 0))
end

function mev_extraction_rate(extracted_mev::Float64, total_volume::Float64)
    return extracted_mev / (total_volume + 1e-10) * 10000  # bps
end

function gas_auction_optimal_bid(base_fee::Float64, competitor_priority_fees::Vector{Float64},
                                   expected_profit::Float64)
    # Bid just above median competitor to win while maximizing profit
    if isempty(competitor_priority_fees)
        return min(base_fee * 0.1, expected_profit * 0.5)
    end
    median_fee = quantile(competitor_priority_fees, 0.5)
    max_bid = expected_profit * 0.8  # keep 20% profit margin
    return min(median_fee * 1.1, max_bid)
end

# ============================================================
# SECTION 4: GOVERNANCE & TOKEN ECONOMICS
# ============================================================

struct GovernanceToken
    symbol::String
    total_supply::Float64
    circulating_supply::Float64
    vesting_schedule::Vector{Tuple{Float64,Float64}}  # (time_years, unlock_pct)
    protocol_revenue_usd::Float64
    buyback_rate::Float64  # pct of revenue used for buyback
end

function token_fully_diluted_valuation(price::Float64, total_supply::Float64)
    return price * total_supply
end

function protocol_earnings_yield(protocol_revenue::Float64, market_cap::Float64)
    return protocol_revenue / (market_cap + 1e-10)
end

function token_inflation_rate(emission_schedule::Vector{Float64})
    # Annual inflation as % of previous supply
    n = length(emission_schedule)
    n < 2 && return 0.0
    return (emission_schedule[2] - emission_schedule[1]) / (emission_schedule[1] + 1e-10)
end

function vesting_unlock_schedule(vesting::Vector{Tuple{Float64,Float64}},
                                   query_times::Vector{Float64})
    # Returns cumulative unlocked % at each time
    unlocked = zeros(length(query_times))
    for (i, t) in enumerate(query_times)
        unlocked[i] = sum(pct for (unlock_t, pct) in vesting if unlock_t <= t)
    end
    return clamp.(unlocked, 0.0, 1.0)
end

function buyback_pressure(revenue::Float64, buyback_rate::Float64,
                            avg_daily_volume::Float64)
    daily_buyback = revenue * buyback_rate / 365
    return daily_buyback / (avg_daily_volume + 1e-10) * 100  # % of daily volume
end

function governance_participation_rate(votes_cast::Float64, circulating_supply::Float64)
    return votes_cast / (circulating_supply + 1e-10) * 100
end

function token_velocity(transaction_volume::Float64, market_cap::Float64)
    # MV = PQ; V = PQ/M
    return transaction_volume / (market_cap + 1e-10)
end

# ============================================================
# SECTION 5: YIELD FARMING & LIQUIDITY MINING ADVANCED
# ============================================================

struct YieldPosition
    protocol::String
    pool::String
    entry_price_token0::Float64
    entry_price_token1::Float64
    liquidity::Float64
    accumulated_fees::Float64
    token_rewards::Float64
    entry_time::Float64
    current_time::Float64
end

function total_position_value(pos::YieldPosition,
                               current_price_token0::Float64,
                               current_price_token1::Float64,
                               reward_token_price::Float64)
    # Simplified: use IL formula for base IL
    price_ratio = current_price_token0 / (pos.entry_price_token0 + 1e-10)
    il = 2*sqrt(price_ratio)/(1+price_ratio) - 1
    base_value = pos.liquidity * (1 + il)
    fee_value   = pos.accumulated_fees
    reward_value = pos.token_rewards * reward_token_price
    return base_value + fee_value + reward_value
end

function optimal_harvest_frequency(apy::Float64, gas_cost_usd::Float64,
                                     position_value::Float64)
    # Compound: harvest when benefit > 2x gas cost (buy & re-stake)
    daily_yield = position_value * apy / 365
    harvest_period_days = 2 * gas_cost_usd / (daily_yield + 1e-10)
    return max(harvest_period_days, 1.0)
end

function farm_migration_analysis(from_apy::Float64, to_apy::Float64,
                                   migration_cost_pct::Float64,
                                   position_value::Float64)
    daily_gain = position_value * (to_apy - from_apy) / 365
    migration_cost = position_value * migration_cost_pct
    breakeven_days = migration_cost / (daily_gain + 1e-10)
    return (daily_gain=daily_gain, migration_cost=migration_cost,
            breakeven_days=breakeven_days, worthwhile=(breakeven_days < 30))
end

function protocol_risk_premium(audit_score::Float64, tvl::Float64,
                                 age_days::Float64, bug_bounty_usd::Float64)
    # Higher score = lower risk (0-100)
    audit_factor  = audit_score / 100.0
    tvl_factor    = min(log10(tvl + 1e-10) / 9.0, 1.0)   # 1B TVL = perfect
    age_factor    = min(age_days / 365.0, 1.0)
    bounty_factor = min(log10(bug_bounty_usd + 1e-10) / 7.0, 1.0)  # 10M = perfect
    risk_score = (audit_factor + tvl_factor + age_factor + bounty_factor) / 4.0 * 100
    # Convert to yield premium needed
    risk_premium = (1 - risk_score/100) * 0.20  # max 20% premium for zero-score
    return (risk_score=risk_score, required_premium=risk_premium)
end

function defi_portfolio_optimizer(pools::Vector{NamedTuple}, total_capital::Float64;
                                   max_per_pool_pct::Float64=0.3,
                                   min_risk_score::Float64=60.0)
    # Filter by risk
    eligible = filter(p -> p.risk_score >= min_risk_score, pools)
    isempty(eligible) && return (weights=Float64[], pools=NamedTuple[])
    n = length(eligible)
    # Naive risk-adjusted APY weighting
    scores = [p.apy * (p.risk_score/100)^2 for p in eligible]
    total_score = sum(scores)
    weights = clamp.(scores ./ (total_score + 1e-10), 0.0, max_per_pool_pct)
    weights ./= sum(weights)
    return (weights=weights, pools=eligible, allocations=weights.*total_capital)
end

# ============================================================
# SECTION 6: CROSS-CHAIN & BRIDGE ANALYTICS
# ============================================================

function bridge_fee_comparison(bridges::Vector{NamedTuple}, amount::Float64)
    results = [(bridge=b.name,
                fee_usd=b.fixed_fee + amount*b.variable_fee,
                time_minutes=b.estimated_time,
                net_received=amount - b.fixed_fee - amount*b.variable_fee)
               for b in bridges]
    best = argmin([r.fee_usd for r in results])
    return (all_bridges=results, best_bridge=results[best])
end

function cross_chain_arb_profitability(asset_price_chain_a::Float64,
                                         asset_price_chain_b::Float64,
                                         bridge_fee_pct::Float64,
                                         gas_cost_total_usd::Float64,
                                         amount::Float64)
    gross_profit = abs(asset_price_chain_a - asset_price_chain_b) /
                   min(asset_price_chain_a, asset_price_chain_b) * amount
    net_profit = gross_profit - amount*bridge_fee_pct - gas_cost_total_usd
    return (gross=gross_profit, net=net_profit, profitable=(net_profit > 0))
end

function tvl_across_chains(chain_tvls::Dict{String,Float64})
    total = sum(values(chain_tvls))
    shares = Dict(k => v/total for (k,v) in chain_tvls)
    return (total_tvl=total, chain_shares=shares,
            dominant_chain=argmax(chain_tvls))
end

# ============================================================
# DEMO
# ============================================================

function demo_defi_extended()
    println("=== DeFi Extended Demo ===")

    # Lending
    market = LendingMarket("USDC", 100e6, 65e6, 0.02, 0.04, 0.75, 0.8, 0.85, 0.05)
    u = utilization_rate(market)
    br = borrow_rate(market)
    sr = supply_rate(market)
    println("Utilization: ", round(u*100,digits=1), "% Borrow APY: ",
            round(br*100,digits=2), "% Supply APY: ", round(sr*100,digits=2), "%")

    # Flash loan arb
    arb = flash_loan_arbitrage(1e6, 30000.0, 30150.0)
    println("Flash loan net profit: \$", round(arb.net_profit, digits=2),
            " Profitable: ", arb.is_profitable)

    # Sandwich
    sw = sandwich_attack_profit(100e3, 5e6)
    println("Sandwich net: \$", round(sw.net, sigdigits=3))

    # Governance
    gt = GovernanceToken("GOV", 1e9, 300e6,
                          [(0.25, 0.1),(0.5, 0.2),(1.0, 0.3),(2.0, 0.4)],
                          5e6, 0.3)
    bp = buyback_pressure(gt.protocol_revenue_usd, gt.buyback_rate, 50e6)
    println("Buyback pressure: ", round(bp, digits=3), "% of daily vol")

    # Yield farming
    harvest_days = optimal_harvest_frequency(0.15, 20.0, 50000.0)
    println("Optimal harvest every ", round(harvest_days, digits=1), " days")

    # Farm migration
    mig = farm_migration_analysis(0.10, 0.18, 0.005, 50000.0)
    println("Migration breakeven: ", round(mig.breakeven_days, digits=1), " days")
end

end # module CryptoDefi
