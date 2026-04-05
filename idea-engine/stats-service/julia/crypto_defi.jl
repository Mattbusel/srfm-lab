# crypto_defi.jl
# DeFi (Decentralized Finance) analytics for crypto/quant trading lab
# Pure Julia stdlib implementation

module CryptoDefiAnalytics

using Statistics, LinearAlgebra, Random

# ============================================================
# DATA STRUCTURES
# ============================================================

struct AMMPool
    token_a::String
    token_b::String
    reserve_a::Float64
    reserve_b::Float64
    fee_rate::Float64        # e.g., 0.003 for 0.3%
    protocol::Symbol         # :uniswap_v2, :uniswap_v3, :curve
end

struct UniV3Pool
    token_a::String
    token_b::String
    current_tick::Int
    tick_spacing::Int
    fee_tier::Float64        # 0.0005, 0.003, or 0.01
    liquidity::Float64       # current active liquidity
    sqrt_price::Float64      # √(price) in Q64.96 format (simplified)
    positions::Vector{Tuple{Int, Int, Float64}}  # (tick_lower, tick_upper, liquidity)
end

struct LiquidityPosition
    pool::AMMPool
    amount_a::Float64
    amount_b::Float64
    shares::Float64
    entry_price::Float64     # price of A in terms of B at entry
    entry_sqrt_k::Float64    # √(reserve_a * reserve_b) at entry
end

struct LendingPosition
    asset::String
    collateral::Float64
    borrowed::Float64
    collateral_factor::Float64   # e.g., 0.75 = 75% LTV allowed
    liquidation_threshold::Float64  # e.g., 0.80
    liquidation_penalty::Float64    # e.g., 0.05
end

struct YieldFarm
    name::String
    tvl::Float64
    apy_base::Float64        # base APY from protocol
    reward_token_price::Float64
    daily_rewards::Float64
    il_risk::Float64         # estimated IL annualized
end

struct MEVOpportunity
    type::Symbol             # :sandwich, :frontrun, :backrun, :arbitrage
    profit_gross::Float64
    gas_cost::Float64
    profit_net::Float64
    execution_block::Int
    target_tx_size::Float64
end

# ============================================================
# AMM: CONSTANT PRODUCT FORMULA (UniswapV2 / x*y=k)
# ============================================================

"""Create a new AMM pool."""
function create_pool(token_a::String, token_b::String,
                     reserve_a::Float64, reserve_b::Float64,
                     fee_rate::Float64=0.003)
    return AMMPool(token_a, token_b, reserve_a, reserve_b, fee_rate, :uniswap_v2)
end

"""Spot price of token A in terms of token B: P = reserve_B / reserve_A."""
function spot_price(pool::AMMPool)
    return pool.reserve_b / pool.reserve_a
end

"""
Price impact of buying amount_a of token A.
Returns: (amount_b_out, new_pool, price_impact_pct)
"""
function swap_a_for_b(pool::AMMPool, amount_a_in::Float64)
    k = pool.reserve_a * pool.reserve_b
    amount_a_in_fee = amount_a_in * (1 - pool.fee_rate)
    new_reserve_a = pool.reserve_a + amount_a_in_fee
    new_reserve_b = k / new_reserve_a
    amount_b_out = pool.reserve_b - new_reserve_b

    new_pool = AMMPool(pool.token_a, pool.token_b,
                       pool.reserve_a + amount_a_in,
                       new_reserve_b, pool.fee_rate, pool.protocol)

    exec_price = amount_b_out / amount_a_in
    price_impact = (spot_price(pool) - exec_price) / spot_price(pool)

    return (amount_b_out=amount_b_out, new_pool=new_pool,
            price_impact_pct=price_impact * 100,
            exec_price=exec_price)
end

"""Swap token B for token A."""
function swap_b_for_a(pool::AMMPool, amount_b_in::Float64)
    k = pool.reserve_a * pool.reserve_b
    amount_b_in_fee = amount_b_in * (1 - pool.fee_rate)
    new_reserve_b = pool.reserve_b + amount_b_in_fee
    new_reserve_a = k / new_reserve_b
    amount_a_out = pool.reserve_a - new_reserve_a

    new_pool = AMMPool(pool.token_a, pool.token_b,
                       new_reserve_a, pool.reserve_b + amount_b_in,
                       pool.fee_rate, pool.protocol)

    exec_price = amount_b_in / amount_a_out
    price_impact = (exec_price - spot_price(pool)) / spot_price(pool)

    return (amount_a_out=amount_a_out, new_pool=new_pool,
            price_impact_pct=price_impact * 100,
            exec_price=exec_price)
end

"""Market depth: amount of token A needed to move price by pct."""
function market_depth(pool::AMMPool, pct::Float64)
    # Price moves from P to P*(1+pct): solve for Δx
    # P_new = reserve_b' / reserve_a' = P*(1+pct)
    # Using x*y=k: reserve_a' = √(k/P_new) - reserve_a
    P = spot_price(pool)
    P_new = P * (1 + pct)
    k = pool.reserve_a * pool.reserve_b
    reserve_a_new = sqrt(k / P_new)
    delta_a = pool.reserve_a - reserve_a_new  # negative = selling A
    return delta_a / (1 - pool.fee_rate)
end

# ============================================================
# IMPERMANENT LOSS
# ============================================================

"""
Impermanent loss formula.
IL = 2*√r/(1+r) - 1 where r = P_current / P_entry
"""
function impermanent_loss(price_ratio::Float64)
    r = price_ratio
    r <= 0 && return -1.0
    return 2 * sqrt(r) / (1 + r) - 1.0
end

"""
Compute IL for a range of price scenarios.
Returns dict of price_ratio => IL_fraction.
"""
function il_scenarios(price_ratios::Vector{Float64}=collect(0.1:0.1:3.0))
    return [(r=r, il_pct=impermanent_loss(r)*100) for r in price_ratios]
end

"""
Liquidity provision P&L including fees earned.
fee_apy: annualized fee APY
holding_period_days: days in pool
price_entry, price_exit: prices of token A in token B
"""
function lp_pnl(initial_value::Float64, price_entry::Float64, price_exit::Float64,
                fee_apy::Float64=0.20, holding_period_days::Float64=30.0)
    price_ratio = price_exit / price_entry
    il = impermanent_loss(price_ratio)

    # Value if held (50/50 split held statically)
    hold_pnl = 0.5 * (price_ratio - 1)  # as fraction of initial

    # LP value with IL
    lp_value_ratio = (1 + il) * (1 + hold_pnl)

    # Fees earned
    fee_earned = fee_apy * holding_period_days / 365

    # Net LP P&L vs holding
    lp_total = lp_value_ratio - 1 + fee_earned
    hold_total = hold_pnl

    return (
        il_pct = il * 100,
        fee_earned_pct = fee_earned * 100,
        lp_pnl_pct = lp_total * 100,
        hold_pnl_pct = hold_total * 100,
        lp_vs_hold_pct = (lp_total - hold_total) * 100
    )
end

"""
Breakeven fee APY needed to offset impermanent loss.
"""
function breakeven_fee_apy(price_ratio::Float64, holding_period_days::Float64=30.0)
    il = impermanent_loss(price_ratio)
    return -il * 365.0 / holding_period_days  # APY needed to break even
end

# ============================================================
# UNISWAP V3: CONCENTRATED LIQUIDITY
# ============================================================

"""
UniswapV3 price from tick.
price = 1.0001^tick
"""
function tick_to_price(tick::Int)
    return 1.0001^tick
end

"""price to nearest tick."""
function price_to_tick(price::Float64, tick_spacing::Int=1)
    raw_tick = log(price) / log(1.0001)
    tick = Int(round(raw_tick / tick_spacing)) * tick_spacing
    return tick
end

"""
Liquidity amounts in V3 position [tick_lower, tick_upper].
Given current sqrt price, computes token amounts for given liquidity L.
"""
function v3_position_amounts(L::Float64, tick_lower::Int, tick_upper::Int,
                               current_price::Float64)
    p_lower = tick_to_price(tick_lower)
    p_upper = tick_to_price(tick_upper)
    p = clamp(current_price, p_lower, p_upper)

    sqrt_p = sqrt(p)
    sqrt_lower = sqrt(p_lower)
    sqrt_upper = sqrt(p_upper)

    if current_price <= p_lower
        # All in token A
        amount_a = L * (1/sqrt_lower - 1/sqrt_upper)
        amount_b = 0.0
    elseif current_price >= p_upper
        # All in token B
        amount_a = 0.0
        amount_b = L * (sqrt_upper - sqrt_lower)
    else
        # In range
        amount_a = L * (1/sqrt_p - 1/sqrt_upper)
        amount_b = L * (sqrt_p - sqrt_lower)
    end

    return (amount_a=amount_a, amount_b=amount_b)
end

"""V3 IL: more complex due to concentrated range."""
function v3_impermanent_loss(L::Float64, tick_lower::Int, tick_upper::Int,
                               price_entry::Float64, price_exit::Float64)
    # Value at entry
    entry = v3_position_amounts(L, tick_lower, tick_upper, price_entry)
    entry_value = entry.amount_a * price_entry + entry.amount_b

    # Value at exit
    exit_pos = v3_position_amounts(L, tick_lower, tick_upper, price_exit)
    exit_value = exit_pos.amount_a * price_exit + exit_pos.amount_b

    # Holding value at exit
    hold_a = entry.amount_a
    hold_b = entry.amount_b
    hold_value = hold_a * price_exit + hold_b

    il = (exit_value - hold_value) / max(entry_value, 1e-10)
    return il
end

# ============================================================
# YIELD FARMING
# ============================================================

"""
APY with continuous compounding.
Daily rate → APY = exp(rate * 365) - 1
"""
function apy_continuous(daily_rate::Float64)
    return exp(daily_rate * 365) - 1.0
end

"""
TVL dynamics model: simple growth/decay with compounding.
TVL(t) = TVL_0 * (1 + inflow_rate - outflow_rate)^t
"""
function tvl_dynamics(tvl0::Float64, inflow_rate::Float64, outflow_rate::Float64,
                       n_days::Int=30)
    tvl = zeros(n_days)
    tvl[1] = tvl0
    for t in 2:n_days
        tvl[t] = tvl[t-1] * (1 + inflow_rate - outflow_rate)
    end
    return tvl
end

"""
Yield farm APY breakdown:
- Base APY: from trading fees
- Reward APY: from emission tokens
- Net APY: after IL risk
"""
function yield_farm_apy(farm::YieldFarm, user_deposit::Float64=1e6)
    share_of_pool = user_deposit / max(farm.tvl, 1e-10)
    daily_reward_value = farm.daily_rewards * share_of_pool * farm.reward_token_price
    reward_apy = daily_reward_value * 365 / user_deposit

    total_apy = farm.apy_base + reward_apy
    net_apy = total_apy - farm.il_risk

    return (
        base_apy=farm.apy_base*100,
        reward_apy=reward_apy*100,
        total_apy=total_apy*100,
        il_risk_apy=farm.il_risk*100,
        net_apy=net_apy*100,
        daily_yield_usd=daily_reward_value + user_deposit * farm.apy_base / 365
    )
end

"""
Optimize yield farming allocation across multiple farms.
Maximize net APY * allocation, subject to total allocation = 1.
Uses greedy approach: allocate proportional to net APY.
"""
function optimize_yield_allocation(farms::Vector{YieldFarm}, total_capital::Float64=1e6)
    net_apys = [max(0.0, f.apy_base - f.il_risk) for f in farms]

    # Simple proportional allocation
    total_net = sum(net_apys)
    if total_net < 1e-10
        allocs = fill(total_capital / length(farms), length(farms))
    else
        allocs = net_apys ./ total_net .* total_capital
    end

    results = []
    for (farm, alloc) in zip(farms, allocs)
        apys = yield_farm_apy(farm, alloc)
        push!(results, (farm=farm.name, allocation=alloc, net_apy=apys.net_apy))
    end

    return allocs, results
end

# ============================================================
# LIQUIDATION MECHANICS
# ============================================================

"""
Health factor: HF = (collateral_value * liq_threshold) / borrowed_value
HF < 1 → liquidation triggered.
"""
function health_factor(pos::LendingPosition, collateral_price::Float64,
                        borrow_price::Float64=1.0)
    collateral_value = pos.collateral * collateral_price
    borrow_value = pos.borrowed * borrow_price
    borrow_value < 1e-10 && return Inf
    return collateral_value * pos.liquidation_threshold / borrow_value
end

"""
Maximum borrowable amount given collateral.
"""
function max_borrow(collateral::Float64, collateral_price::Float64,
                    collateral_factor::Float64, borrow_price::Float64=1.0)
    return collateral * collateral_price * collateral_factor / borrow_price
end

"""
Liquidation mechanics:
- Liquidator repays fraction of debt
- Receives collateral at discount (liquidation penalty)
Returns: liquidation profit for liquidator.
"""
function liquidate_position(pos::LendingPosition, collateral_price::Float64,
                              close_factor::Float64=0.5, borrow_price::Float64=1.0)
    hf = health_factor(pos, collateral_price, borrow_price)

    if hf >= 1.0
        return (can_liquidate=false, profit=0.0, debt_repaid=0.0,
                collateral_seized=0.0, remaining_hf=hf)
    end

    # Debt to repay
    debt_to_repay = pos.borrowed * close_factor * borrow_price

    # Collateral seized at discount
    liquidation_bonus = 1.0 + pos.liquidation_penalty
    collateral_seized_value = debt_to_repay * liquidation_bonus
    collateral_seized = collateral_seized_value / collateral_price

    # Liquidator profit
    profit = collateral_seized * collateral_price - debt_to_repay
    profit_pct = profit / debt_to_repay * 100

    return (can_liquidate=true, profit=profit, profit_pct=profit_pct,
            debt_repaid=debt_to_repay, collateral_seized=collateral_seized,
            remaining_hf=hf)
end

"""
Simulate liquidation cascade: price falls trigger liquidations
which further depress price.
"""
function liquidation_cascade(positions::Vector{LendingPosition},
                               initial_price::Float64, price_shock::Float64;
                               price_impact_per_liquidation::Float64=0.005,
                               n_rounds::Int=20)
    prices = Float64[initial_price]
    n_liquidated = Int[]
    total_collateral_sold = Float64[]

    current_price = initial_price * (1 - price_shock)
    push!(prices, current_price)
    remaining = copy(positions)

    for round in 1:n_rounds
        liquidated_this_round = 0
        collateral_sold = 0.0

        new_remaining = LendingPosition[]
        for pos in remaining
            result = liquidate_position(pos, current_price)
            if result.can_liquidate
                collateral_sold += result.collateral_seized
                liquidated_this_round += 1
            else
                push!(new_remaining, pos)
            end
        end

        push!(n_liquidated, liquidated_this_round)
        push!(total_collateral_sold, collateral_sold)

        # Price impact from forced selling
        if collateral_sold > 0
            current_price *= (1 - price_impact_per_liquidation * liquidated_this_round)
            push!(prices, current_price)
        end

        remaining = new_remaining
        liquidated_this_round == 0 && break
    end

    return (price_path=prices, liquidations_per_round=n_liquidated,
            total_liquidated=length(positions) - length(remaining),
            final_price=current_price,
            final_price_drop_pct=(initial_price - current_price)/initial_price*100)
end

# ============================================================
# FLASH LOAN ARBITRAGE
# ============================================================

"""
Detect price discrepancy between two pools for the same token pair.
"""
function detect_arbitrage(pool1::AMMPool, pool2::AMMPool)
    price1 = spot_price(pool1)
    price2 = spot_price(pool2)

    if price1 > price2 * (1 + pool1.fee_rate + pool2.fee_rate)
        # Buy on pool2, sell on pool1
        direction = :buy_pool2_sell_pool1
        price_diff_pct = (price1 - price2) / price2 * 100
        profit_pct = price_diff_pct - (pool1.fee_rate + pool2.fee_rate) * 100
    elseif price2 > price1 * (1 + pool1.fee_rate + pool2.fee_rate)
        direction = :buy_pool1_sell_pool2
        price_diff_pct = (price2 - price1) / price1 * 100
        profit_pct = price_diff_pct - (pool1.fee_rate + pool2.fee_rate) * 100
    else
        direction = :none
        price_diff_pct = abs(price1 - price2) / min(price1, price2) * 100
        profit_pct = 0.0
    end

    return (direction=direction, price1=price1, price2=price2,
            price_diff_pct=price_diff_pct, profit_pct=profit_pct)
end

"""
Optimal flash loan size for arbitrage.
Uses binary search to find size maximizing profit.
"""
function optimal_arb_size(pool1::AMMPool, pool2::AMMPool,
                           flash_loan_fee::Float64=0.0009,  # 0.09% Aave fee
                           gas_cost_usd::Float64=50.0)
    arb = detect_arbitrage(pool1, pool2)
    arb.direction == :none && return (optimal_size=0.0, max_profit=0.0)

    # Binary search for optimal trade size
    lo, hi = 0.0, min(pool1.reserve_a, pool2.reserve_a) * 0.3

    function profit_at_size(size::Float64)
        if arb.direction == :buy_pool2_sell_pool1
            result_buy = swap_b_for_a(pool2, size * spot_price(pool2))
            amount_a = result_buy.amount_a_out
            result_sell = swap_a_for_b(pool1, amount_a)
            gross = result_sell.amount_b_out - size * spot_price(pool2)
        else
            result_buy = swap_b_for_a(pool1, size * spot_price(pool1))
            amount_a = result_buy.amount_a_out
            result_sell = swap_a_for_b(pool2, amount_a)
            gross = result_sell.amount_b_out - size * spot_price(pool1)
        end
        flash_fee = size * spot_price(pool1) * flash_loan_fee
        return gross - flash_fee - gas_cost_usd
    end

    best_profit = -Inf
    best_size = 0.0

    for size in range(lo, hi, length=100)
        p = profit_at_size(size)
        if p > best_profit
            best_profit = p
            best_size = size
        end
    end

    return (optimal_size=best_size, max_profit=max(0.0, best_profit),
            profitable=best_profit > 0)
end

# ============================================================
# MEV: SANDWICH ATTACK MODELING
# ============================================================

"""
Sandwich attack P&L calculator.
Attacker: 1) front-run victim's buy, 2) victim buys, 3) back-run (sell).
"""
function sandwich_attack(pool::AMMPool, victim_amount_a::Float64,
                          max_frontrun_fraction::Float64=0.1)
    original_price = spot_price(pool)

    # Step 1: Front-run (attacker buys token A before victim)
    frontrun_size = pool.reserve_b * max_frontrun_fraction
    fr_result = swap_b_for_a(pool, frontrun_size)
    pool_after_fr = fr_result.new_pool
    attacker_a_received = fr_result.amount_a_out

    # Step 2: Victim transaction executes
    victim_result = swap_a_for_b(pool_after_fr, victim_amount_a)
    pool_after_victim = victim_result.new_pool

    # Victim's effective execution price vs expected
    victim_expected_b = victim_amount_a * original_price * (1 - pool.fee_rate)
    victim_actual_b = victim_result.amount_b_out
    victim_loss = victim_expected_b - victim_actual_b
    victim_slippage_pct = victim_loss / victim_expected_b * 100

    # Step 3: Attacker back-runs (sells token A received from front-run)
    backrun_result = swap_a_for_b(pool_after_victim, attacker_a_received)

    # Attacker P&L
    attacker_cost_b = frontrun_size  # USDC spent
    attacker_received_b = backrun_result.amount_b_out
    attacker_profit_b = attacker_received_b - attacker_cost_b

    return (
        attacker_profit=attacker_profit_b,
        victim_loss=victim_loss,
        victim_slippage_pct=victim_slippage_pct,
        frontrun_size=frontrun_size,
        price_impact_on_victim=(victim_result.exec_price / original_price - 1) * 100
    )
end

"""
Front-running cost for a given transaction.
Estimates extra cost paid by victim due to front-running.
"""
function frontrunning_cost(pool::AMMPool, victim_trade_size::Float64,
                            n_front_runners::Int=1)
    # Without front-running
    clean_result = swap_a_for_b(pool, victim_trade_size)
    clean_price = clean_result.exec_price

    # With front-running (each front-runner takes fraction of liquidity)
    current_pool = pool
    for i in 1:n_front_runners
        fr_size = victim_trade_size * 0.05  # 5% of victim size per attacker
        fr_result = swap_a_for_b(current_pool, fr_size)
        current_pool = fr_result.new_pool
    end

    # Victim executes after front-runners
    victim_result_with_fr = swap_a_for_b(current_pool, victim_trade_size)

    extra_cost = (clean_price - victim_result_with_fr.exec_price) * victim_trade_size
    extra_slippage_pct = (clean_price - victim_result_with_fr.exec_price) / clean_price * 100

    return (extra_cost=extra_cost, extra_slippage_pct=extra_slippage_pct,
            clean_price=clean_price, frontrun_price=victim_result_with_fr.exec_price)
end

# ============================================================
# GOVERNANCE TOKEN VALUATION
# ============================================================

"""
Governance token cash flow model.
Value = PV of protocol revenues + voting power premium.
"""
function governance_token_value(
    protocol_revenue_annual::Float64,
    fee_switch_fraction::Float64,    # fraction of revenue distributed to token holders
    token_supply::Float64,
    growth_rate::Float64=0.20,       # annual revenue growth
    discount_rate::Float64=0.25,     # required return
    terminal_growth::Float64=0.05,   # terminal growth rate
    voting_premium::Float64=0.10,    # premium for governance rights
    n_years::Int=10
)
    if discount_rate <= terminal_growth
        discount_rate = terminal_growth + 0.01
    end

    # DCF of fee distributions
    pv_fees = 0.0
    revenue = protocol_revenue_annual
    for t in 1:n_years
        revenue *= (1 + growth_rate)
        growth_rate = max(growth_rate * 0.9, terminal_growth)  # decay to terminal
        annual_dist = revenue * fee_switch_fraction
        pv_fees += annual_dist / (1 + discount_rate)^t
    end

    # Terminal value
    terminal_revenue = protocol_revenue_annual * (1 + growth_rate)^n_years
    terminal_dist = terminal_revenue * fee_switch_fraction
    terminal_value = terminal_dist / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate)^n_years

    total_pv = pv_fees + pv_terminal

    # Per-token value with voting premium
    per_token_intrinsic = total_pv / token_supply
    per_token_with_premium = per_token_intrinsic * (1 + voting_premium)

    return (
        pv_fees=pv_fees,
        pv_terminal=pv_terminal,
        total_pv=total_pv,
        per_token_intrinsic=per_token_intrinsic,
        per_token_with_premium=per_token_with_premium,
        implied_pe=token_supply * per_token_intrinsic / (protocol_revenue_annual * fee_switch_fraction)
    )
end

# ============================================================
# PROTOCOL REVENUE: FEE SWITCH + BUYBACK-AND-BURN
# ============================================================

"""
Protocol revenue model.
TVL → trading volume → fees → token buyback/burn mechanics.
"""
struct ProtocolRevenue
    tvl::Float64
    volume_to_tvl::Float64      # daily volume as fraction of TVL
    fee_rate::Float64           # fee on volume
    fee_split_lp::Float64       # fraction to LPs
    fee_split_protocol::Float64 # fraction to protocol treasury
    buyback_fraction::Float64   # fraction used for token buyback
    token_supply::Float64
    token_price::Float64
end

function compute_protocol_metrics(pr::ProtocolRevenue)
    daily_volume = pr.tvl * pr.volume_to_tvl
    daily_fees = daily_volume * pr.fee_rate
    daily_lp_fees = daily_fees * pr.fee_split_lp
    daily_protocol_fees = daily_fees * pr.fee_split_protocol
    daily_buyback = daily_protocol_fees * pr.buyback_fraction

    # Tokens burned per day
    tokens_burned = daily_buyback / pr.token_price

    # Annualized metrics
    annual_fees = daily_fees * 365
    annual_protocol_revenue = daily_protocol_fees * 365
    annual_buyback = daily_buyback * 365
    annual_burn_rate = tokens_burned * 365

    # P/E ratio (using fully diluted market cap)
    market_cap = pr.token_supply * pr.token_price
    pe_ratio = market_cap / annual_protocol_revenue

    # P/F ratio (price to fee)
    pf_ratio = market_cap / annual_fees

    return (
        daily_volume=daily_volume,
        daily_fees=daily_fees,
        daily_protocol_revenue=daily_protocol_fees,
        daily_buyback_usd=daily_buyback,
        daily_tokens_burned=tokens_burned,
        annual_protocol_revenue=annual_protocol_revenue,
        annual_burn_pct=annual_burn_rate / pr.token_supply * 100,
        pe_ratio=pe_ratio,
        pf_ratio=pf_ratio
    )
end

# ============================================================
# DeFi SYSTEMIC RISK: PROTOCOL FAILURE CASCADE
# ============================================================

struct DeFiProtocol
    name::String
    tvl::Float64
    dependencies::Vector{String}  # protocols this one depends on
    collateral_type::Symbol       # :stablecoin, :volatile, :lp_token, :synthetic
    oracle_risk::Float64          # 0-1 probability of oracle failure
    smart_contract_risk::Float64  # 0-1 probability of hack/exploit
end

"""
Simulate DeFi protocol failure cascade.
Uses dependency graph to propagate failures.
"""
function defi_failure_cascade(protocols::Vector{DeFiProtocol},
                               initial_failure::String;
                               cascade_threshold::Float64=0.3)
    n = length(protocols)
    protocol_map = Dict(p.name => i for (i, p) in enumerate(protocols))

    failed = Set{String}()
    push!(failed, initial_failure)

    tvl_at_risk = Float64[]
    rounds = 0

    for round in 1:n
        new_failures = String[]

        for p in protocols
            p.name in failed && continue

            # Check if dependencies have failed
            failed_deps = [d for d in p.dependencies if d in failed]
            if isempty(failed_deps)
                continue
            end

            # Fraction of TVL at risk from failed dependencies
            total_dep_tvl = sum(protocols[protocol_map[d]].tvl
                               for d in p.dependencies if haskey(protocol_map, d))
            failed_dep_tvl = sum(protocols[protocol_map[d]].tvl
                                for d in failed_deps if haskey(protocol_map, d))

            failure_fraction = total_dep_tvl > 0 ? failed_dep_tvl / total_dep_tvl : 0.0

            # Collateral type multiplier
            multiplier = p.collateral_type == :lp_token ? 1.5 :
                         p.collateral_type == :synthetic ? 1.3 :
                         p.collateral_type == :volatile ? 1.1 : 0.8

            if failure_fraction * multiplier > cascade_threshold
                push!(new_failures, p.name)
                push!(tvl_at_risk, p.tvl)
            end
        end

        for f in new_failures
            push!(failed, f)
        end

        rounds += 1
        isempty(new_failures) && break
    end

    total_tvl = sum(p.tvl for p in protocols)
    failed_tvl = sum(protocols[protocol_map[f]].tvl for f in failed if haskey(protocol_map, f))

    return (
        failed_protocols=collect(failed),
        n_failed=length(failed),
        failed_tvl=failed_tvl,
        failed_tvl_pct=failed_tvl/total_tvl*100,
        cascade_rounds=rounds,
        systemic_risk_score=length(failed) / n
    )
end

# ============================================================
# DEMO FUNCTIONS
# ============================================================

"""Demo: AMM mechanics."""
function demo_amm()
    println("=== AMM Demo: UniswapV2 ===")
    pool = create_pool("ETH", "USDC", 1000.0, 2_000_000.0, 0.003)
    println("Initial price: \$$(spot_price(pool))/ETH")

    # Swap 10 ETH for USDC
    result = swap_a_for_b(pool, 10.0)
    println("Swap 10 ETH → $(round(result.amount_b_out, digits=2)) USDC")
    println("Price impact: $(round(result.price_impact_pct, digits=4))%")

    # IL scenarios
    println("\n=== Impermanent Loss ===")
    for r in [0.5, 0.75, 1.0, 1.5, 2.0]
        il = impermanent_loss(r)
        println("  Price ratio $r: IL = $(round(il*100, digits=2))%")
    end

    # LP P&L
    pnl = lp_pnl(1e6, 2000.0, 2800.0, fee_apy=0.25)
    println("\n=== LP P&L (ETH: 2000→2800) ===")
    println("  IL: $(round(pnl.il_pct, digits=2))%")
    println("  Fees earned: $(round(pnl.fee_earned_pct, digits=2))%")
    println("  LP net: $(round(pnl.lp_pnl_pct, digits=2))%")
    println("  Hold return: $(round(pnl.hold_pnl_pct, digits=2))%")
    println("  LP vs Hold: $(round(pnl.lp_vs_hold_pct, digits=2))%")
end

"""Demo: Liquidation cascade."""
function demo_liquidation(; rng::AbstractRNG=MersenneTwister(42))
    println("\n=== Liquidation Cascade Demo ===")
    positions = [
        LendingPosition("ETH", 10.0, 15000.0, 0.75, 0.80, 0.05),
        LendingPosition("ETH", 5.0, 7000.0, 0.75, 0.80, 0.05),
        LendingPosition("ETH", 20.0, 25000.0, 0.75, 0.80, 0.05),
        LendingPosition("ETH", 8.0, 11000.0, 0.75, 0.80, 0.05),
    ]

    result = liquidation_cascade(positions, 2000.0, 0.15)
    println("Price path: $(round.(result.price_path, digits=2))")
    println("Total liquidated: $(result.total_liquidated)/$(length(positions)) positions")
    println("Final price drop: $(round(result.final_price_drop_pct, digits=2))%")
end

"""Demo: Sandwich attack."""
function demo_sandwich()
    println("\n=== Sandwich Attack Demo ===")
    pool = create_pool("ETH", "USDC", 1000.0, 2_000_000.0, 0.003)

    result = sandwich_attack(pool, 50.0)  # 50 ETH victim trade
    println("Victim trade: 50 ETH")
    println("Attacker profit: \$$(round(result.attacker_profit, digits=2))")
    println("Victim loss: \$$(round(result.victim_loss, digits=2))")
    println("Victim slippage: $(round(result.victim_slippage_pct, digits=2))%")
end

"""Demo: Governance token valuation."""
function demo_token_valuation()
    println("\n=== Governance Token Valuation ===")
    val = governance_token_value(
        1e8,     # $100M annual revenue
        0.20,    # 20% fee switch to token holders
        1e9,     # 1B token supply
        0.30,    # 30% annual growth
        0.25,    # 25% discount rate
        0.05     # 5% terminal growth
    )
    println("Per-token intrinsic value: \$$(round(val.per_token_intrinsic, digits=4))")
    println("Per-token with voting premium: \$$(round(val.per_token_with_premium, digits=4))")
    println("Implied P/E: $(round(val.implied_pe, digits=1))x")
end


# ============================================================
# ADDITIONAL DeFi ANALYTICS
# ============================================================

# ============================================================
# CURVE FINANCE: STABLESWAP INVARIANT
# ============================================================

"""
Curve StableSwap invariant: A*n^n*sum(x_i) + D = A*n^n*D + D^(n+1)/(n^n * prod(x_i))
Newton's method to find D.
"""
function curve_invariant_D(reserves::Vector{Float64}, A::Float64)
    n = length(reserves)
    S = sum(reserves); D = S
    Ann = A * n^n

    for _ in 1:256
        D_P = D
        for xi in reserves
            D_P = D_P * D / (n * max(xi, 1e-18))
        end
        D_prev = D
        D = (Ann * S + n * D_P) * D / ((Ann - 1) * D + (n + 1) * D_P)
        abs(D - D_prev) < 1e-12 && break
    end
    return D
end

"""
Curve swap: token i → token j.
Returns (dy, fee).
"""
function curve_swap(reserves::Vector{Float64}, A::Float64, fee_rate::Float64,
                     i::Int, j::Int, dx::Float64)
    n = length(reserves)
    Ann = A * n^n
    dx_eff = dx * (1 - fee_rate)
    new_reserves = copy(reserves)
    new_reserves[i] += dx_eff

    D = curve_invariant_D(reserves, A)

    # Solve for new reserves[j]
    S_prime = sum(new_reserves) - new_reserves[j]
    c = D
    for k in 1:n
        k == j && continue
        c = c * D / (n * new_reserves[k])
    end
    c = c * D / (n * Ann)
    b = S_prime + D / Ann

    y = D
    for _ in 1:256
        y_prev = y
        y = (y^2 + c) / (2*y + b - D)
        abs(y - y_prev) < 1e-12 && break
    end

    dy = new_reserves[j] - y
    return (dy=max(dy, 0.0), fee=dx - dx_eff, new_reserves=begin
        r = copy(new_reserves); r[j] = y; r
    end)
end

"""Curve price impact for a given trade size."""
function curve_price_impact(reserves::Vector{Float64}, A::Float64, fee_rate::Float64,
                              i::Int, j::Int, dx::Float64)
    price_before = reserves[j] / reserves[i]
    result = curve_swap(reserves, A, fee_rate, i, j, dx)
    price_after = result.new_reserves[j] / result.new_reserves[i]
    return abs(price_before - price_after) / price_before
end

# ============================================================
# AAVE v3 INTEREST RATE MODEL
# ============================================================

"""
AAVE v3 interest rate model.
Borrow rate depends on utilization U = total_borrowed / total_liquidity.
Below U_optimal: slope1
Above U_optimal: slope1 + slope2 * (U - U_optimal) / (1 - U_optimal)
"""
function aave_borrow_rate(total_borrowed::Float64, total_liquidity::Float64;
                           base_rate::Float64=0.0,
                           slope1::Float64=0.04,
                           slope2::Float64=0.60,
                           u_optimal::Float64=0.80)
    total_liquidity <= 0 && return base_rate
    U = total_borrowed / total_liquidity
    U = clamp(U, 0.0, 1.0)

    if U <= u_optimal
        return base_rate + U / u_optimal * slope1
    else
        excess = (U - u_optimal) / (1 - u_optimal)
        return base_rate + slope1 + excess * slope2
    end
end

"""Supply rate: borrow_rate * U * (1 - reserve_factor)."""
function aave_supply_rate(borrow_rate::Float64, utilization::Float64;
                           reserve_factor::Float64=0.10)
    return borrow_rate * utilization * (1 - reserve_factor)
end

"""
Simulate lending protocol TVL and interest rate dynamics.
"""
function simulate_lending_protocol(initial_tvl::Float64, initial_borrowed::Float64;
                                    n_days::Int=365, inflow_rate::Float64=0.001,
                                    outflow_sensitivity::Float64=2.0,
                                    rng::AbstractRNG=MersenneTwister(42))
    tvl = zeros(n_days); borrowed = zeros(n_days)
    borrow_rate = zeros(n_days); supply_rate = zeros(n_days)
    tvl[1] = initial_tvl; borrowed[1] = initial_borrowed

    for t in 2:n_days
        br = aave_borrow_rate(borrowed[t-1], tvl[t-1])
        sr = aave_supply_rate(br, borrowed[t-1]/max(tvl[t-1], 1e-10))
        borrow_rate[t] = br; supply_rate[t] = sr

        # TVL changes: attracted by high supply rate, repelled by high borrow rate
        daily_inflow = tvl[t-1] * inflow_rate * (1 + sr)
        daily_outflow = tvl[t-1] * 0.001 * outflow_sensitivity * (1 + br)

        # Borrow demand: increases with lower borrow rate
        borrow_demand = tvl[t-1] * 0.002 / (1 + br)
        borrow_repay = borrowed[t-1] * 0.005

        tvl[t] = max(tvl[t-1] + daily_inflow - daily_outflow + randn(rng)*tvl[t-1]*0.005, 0)
        borrowed[t] = clamp(borrowed[t-1] + borrow_demand - borrow_repay, 0, tvl[t])
    end

    return (tvl=tvl, borrowed=borrowed, borrow_rate=borrow_rate, supply_rate=supply_rate,
            avg_utilization=mean(borrowed ./ max.(tvl, 1e-10)))
end

# ============================================================
# PERPETUAL SWAP PRICING
# ============================================================

"""
Perpetual swap fair value model.
Fair value = spot + basis_adjustment
Basis = funding_rate * days_to_convergence
"""
function perp_fair_value(spot::Float64, expected_funding_8h::Float64;
                          convergence_days::Float64=30.0)
    annual_funding = expected_funding_8h * 3 * 365
    basis = spot * annual_funding * convergence_days / 365
    return spot + basis
end

"""
Funding arbitrage: long spot, short perp to earn funding.
P&L = funding_collected - hedging_costs - borrow_costs
"""
function funding_arbitrage_pnl(spot_price::Float64, perp_price::Float64,
                                funding_rate_8h::Float64, holding_days::Float64;
                                borrow_rate_annual::Float64=0.02,
                                execution_slippage::Float64=0.001,
                                fee_rate::Float64=0.0006)
    n_funding_periods = holding_days * 3  # 8h periods per day
    funding_collected = abs(funding_rate_8h) * n_funding_periods * spot_price

    borrow_cost = spot_price * borrow_rate_annual * holding_days / 365
    slippage_cost = spot_price * execution_slippage * 2  # entry + exit
    fee_cost = (spot_price + perp_price) * fee_rate * 2  # entry + exit

    net_pnl = funding_collected - borrow_cost - slippage_cost - fee_cost
    return (gross_funding=funding_collected, borrow_cost=borrow_cost,
            slippage_cost=slippage_cost, fee_cost=fee_cost,
            net_pnl=net_pnl, net_apy=net_pnl/spot_price*365/holding_days)
end

# ============================================================
# LIQUIDITY MINING SIMULATION
# ============================================================

"""
Simulate liquidity mining with token emission schedule.
Token price dynamics affect APY in real-time.
"""
function simulate_liquidity_mining(
    initial_tvl::Float64,
    token_emission_schedule::Vector{Float64},  # daily token emissions
    token_price_initial::Float64;
    price_decay_rate::Float64=0.002,           # daily % decline from inflation
    tvl_sensitivity_to_apy::Float64=0.5,       # TVL growth elasticity to APY
    fee_apy_base::Float64=0.10,
    rng::AbstractRNG=MersenneTwister(42)
)
    n_days = length(token_emission_schedule)
    tvl = zeros(n_days); token_price = zeros(n_days); apy = zeros(n_days)
    il_loss = zeros(n_days)

    tvl[1] = initial_tvl
    token_price[1] = token_price_initial

    for t in 2:n_days
        # Token price: inflation pressure reduces price
        emission_pressure = token_emission_schedule[t] / max(tvl[t-1]/token_price[t-1], 1e-10)
        token_price[t] = token_price[t-1] * (1 - price_decay_rate * (1 + emission_pressure))
        token_price[t] *= (1 + randn(rng) * 0.03)  # noise
        token_price[t] = max(token_price[t], 0.01)

        # APY
        daily_emission_value = token_emission_schedule[t] * token_price[t]
        emission_apy = daily_emission_value * 365 / max(tvl[t-1], 1e-10)
        total_apy = fee_apy_base + emission_apy
        apy[t] = total_apy

        # TVL: attracted by APY, decreased by IL and token price risk
        tvl_growth = tvl_sensitivity_to_apy * (total_apy - 0.15) / 365  # vs 15% hurdle
        il_daily = 0.001 * abs(randn(rng)) * token_price[t] / token_price[max(1, t-30)]
        il_loss[t] = il_daily

        tvl[t] = max(tvl[t-1] * (1 + tvl_growth - il_daily * 0.1), 1e6)
    end

    return (tvl=tvl, token_price=token_price, apy=apy, il_loss=il_loss,
            avg_apy=mean(apy[2:end]), total_il=sum(il_loss))
end

# ============================================================
# DEX AGGREGATOR ROUTING
# ============================================================

"""
Find optimal routing across multiple DEX pools.
Simple greedy: route fraction to each pool to minimize price impact.
"""
function optimal_route(pools::Vector{AMMPool}, amount_out_token::Bool,
                        total_amount::Float64)
    n_pools = length(pools)
    n_steps = 100

    best_output = 0.0
    best_split = fill(total_amount / n_pools, n_pools)

    # Grid search over split ratios (simplified for 2 pools)
    n_pools == 2 || return (best_split, 0.0)

    for frac in range(0.0, 1.0, length=n_steps)
        splits = [frac * total_amount, (1-frac) * total_amount]
        output = 0.0
        for (i, pool) in enumerate(pools)
            if splits[i] > 0
                r = swap_a_for_b(pool, splits[i])
                output += r.amount_b_out
            end
        end
        if output > best_output
            best_output = output
            best_split = splits
        end
    end

    return (split=best_split, total_output=best_output)
end

# ============================================================
# RISK METRICS FOR DeFi PORTFOLIOS
# ============================================================

"""
DeFi portfolio risk metrics: combines smart contract risk, IL risk, liquidity risk.
"""
function defi_portfolio_risk(
    positions::Vector{Tuple{String, Float64, Float64}};  # (protocol, tvl_share, il_vol)
    smart_contract_risk_scores::Vector{Float64}=ones(length(positions)),
    liquidity_risk_scores::Vector{Float64}=ones(length(positions)) * 0.5
)
    n = length(positions)
    names = [p[1] for p in positions]
    weights = [p[2] for p in positions] ./ sum([p[2] for p in positions])
    il_vols = [p[3] for p in positions]

    # IL risk (annualized expected)
    il_risk = sum(weights[i] * 0.5 * il_vols[i]^2 for i in 1:n)

    # Smart contract risk (hack probability × capital at risk)
    sc_risk = sum(weights[i] * smart_contract_risk_scores[i] for i in 1:n)

    # Liquidity risk (can we exit?)
    liq_risk = sum(weights[i] * liquidity_risk_scores[i] for i in 1:n)

    # Concentration risk (HHI)
    hhi = sum(w^2 for w in weights)

    # Composite risk score
    composite = (il_risk * 0.3 + sc_risk * 0.4 + liq_risk * 0.2 + hhi * 0.1)

    return (il_risk=il_risk, sc_risk=sc_risk, liq_risk=liq_risk,
            concentration=hhi, composite_risk=composite,
            weights=weights, protocol_names=names)
end

end # module CryptoDefiAnalytics
