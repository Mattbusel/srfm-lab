"""
CryptoMarketMechanics.jl — Deep crypto market mechanics module

Extends the existing CryptoMechanics.jl with:
  - FeeCalculator: tiered maker/taker/volume fee computation
  - MarginEngine: leverage, initial/maintenance margin, margin call logic
  - FundingRateModel: premium index, expected funding, historical analysis
  - BasisTrader: spot-perp convergence strategy, basis z-score signals
  - LiquidationEngine: cascade simulation, insurance fund modeling
  - SlippageModel: order book depth → expected slippage function
  - CircuitBreaker: price band detection, trading halt simulation
"""
module CryptoMarketMechanics

using Statistics, LinearAlgebra, Random

export FeeCalculator, MarginEngine, FundingRateModel, BasisTrader,
       LiquidationEngine, SlippageModel, CircuitBreaker

export compute_fee, effective_fee_tier, maker_rebate,
       initial_margin, maintenance_margin, margin_ratio, is_margin_call,
       compute_forced_liquidation_price, liquidation_penalty,
       compute_funding_rate, premium_index, expected_daily_funding,
       funding_carry_signal, funding_regime,
       basis_zscore, basis_signal, enter_basis_trade, exit_condition,
       simulated_cascade, insurance_fund_sufficiency,
       expected_slippage, market_impact, fill_price,
       is_circuit_breaker_triggered, halt_duration, price_band

# ─────────────────────────────────────────────────────────────────────────────
# 1. FeeCalculator
# ─────────────────────────────────────────────────────────────────────────────

"""
FeeCalculator: tiered maker/taker fee computation for crypto exchanges.

Exchange fee schedules use volume-based tiers where higher monthly trading
volume reduces fees. Makers (limit orders) pay less than takers (market
orders). Some exchanges offer maker rebates at high volumes.

Fields:
  exchange_name : exchange identifier
  tiers         : vector of (min_30d_vol_usd, maker_fee, taker_fee)
  bnb_discount  : optional token discount (e.g., Binance BNB = 0.25 off)
"""
struct FeeCalculator
    exchange_name::String
    tiers::Vector{Tuple{Float64, Float64, Float64}}  # (min_vol, maker, taker)
    bnb_discount::Float64
end

# Pre-defined exchange fee schedules
function binance_fees()
    return FeeCalculator("Binance", [
        (0.0,         0.0002,  0.0010),
        (1_000_000,   0.0002,  0.0009),
        (5_000_000,   0.0001,  0.0008),
        (20_000_000,  0.0001,  0.0007),
        (100_000_000, 0.0000,  0.0006),
        (500_000_000,-0.00005, 0.0005),
    ], 0.25)
end

function bybit_fees()
    return FeeCalculator("Bybit", [
        (0.0,          0.0001,  0.0006),
        (5_000_000,    0.0001,  0.0005),
        (25_000_000,   0.0000,  0.0005),
        (100_000_000, -0.0001,  0.0004),
    ], 0.0)
end

function okx_fees()
    return FeeCalculator("OKX", [
        (0.0,          0.0002,  0.0005),
        (10_000_000,   0.0001,  0.0004),
        (50_000_000,   0.0000,  0.0004),
        (200_000_000, -0.0001,  0.0003),
    ], 0.0)
end

function deribit_fees()
    return FeeCalculator("Deribit", [
        (0.0, 0.0000, 0.0003),
    ], 0.0)
end

"""
Look up effective fee for a given 30-day volume.
Returns (maker_fee, taker_fee).
"""
function effective_fee_tier(fc::FeeCalculator, monthly_vol::Float64)
    applicable_tier = fc.tiers[1]
    for tier in fc.tiers
        if monthly_vol >= tier[1]
            applicable_tier = tier
        end
    end
    return (maker=applicable_tier[2], taker=applicable_tier[3])
end

"""
Compute fee for a single trade.
is_maker: true = limit order that posted liquidity.
use_discount: whether to apply token discount (e.g., BNB).
"""
function compute_fee(fc::FeeCalculator, notional::Float64, monthly_vol::Float64;
                      is_maker::Bool=false, use_discount::Bool=false)
    tier = effective_fee_tier(fc, monthly_vol)
    fee_rate = is_maker ? tier.maker : tier.taker
    if use_discount && fc.bnb_discount > 0
        fee_rate *= (1 - fc.bnb_discount)
    end
    return abs(notional * fee_rate)
end

"""
Compute annualized fee drag for a strategy.
n_round_trips: number of complete round trips (entry + exit) per year.
"""
function annual_fee_drag(fc::FeeCalculator, avg_notional::Float64,
                          n_round_trips::Int, monthly_vol::Float64;
                          pct_maker::Float64=0.5, use_discount::Bool=false)
    tier = effective_fee_tier(fc, monthly_vol)
    avg_rate = pct_maker * tier.maker + (1 - pct_maker) * tier.taker
    if use_discount; avg_rate *= (1 - fc.bnb_discount); end
    return avg_notional * avg_rate * n_round_trips * 2  # entry + exit
end

"""Maker rebate: when fee is negative, return the rebate earned."""
function maker_rebate(fc::FeeCalculator, notional::Float64, monthly_vol::Float64)
    tier = effective_fee_tier(fc, monthly_vol)
    if tier.maker < 0
        return abs(notional * tier.maker)
    end
    return 0.0
end

"""Breakeven gross P&L per trade to cover fees."""
function breakeven_gross_bps(fc::FeeCalculator, monthly_vol::Float64;
                               pct_maker::Float64=0.5)
    tier = effective_fee_tier(fc, monthly_vol)
    avg_rate = pct_maker * tier.maker + (1 - pct_maker) * tier.taker
    return avg_rate * 2 * 10000  # round trip in bps
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. MarginEngine
# ─────────────────────────────────────────────────────────────────────────────

"""
MarginEngine: computes margin requirements and monitors positions.

Initial margin: required collateral to open position (1/leverage).
Maintenance margin: minimum equity before liquidation.
Margin call: triggered when equity falls below maintenance margin.

For cross-margin (portfolio margin): positions can share collateral.
"""
struct MarginEngine
    leverage_max::Float64         # maximum allowed leverage
    init_margin_rate::Float64     # initial margin as fraction of notional
    maint_margin_rate::Float64    # maintenance margin as fraction
    liquidation_fee::Float64      # fee charged on forced liquidation
    taker_fee::Float64            # fee applied to liquidation trade
end

function MarginEngine(; leverage_max::Float64=100.0,
                        init_margin_rate::Float64=0.01,
                        maint_margin_rate::Float64=0.005,
                        liquidation_fee::Float64=0.0015,
                        taker_fee::Float64=0.0005)
    return MarginEngine(leverage_max, init_margin_rate, maint_margin_rate,
                         liquidation_fee, taker_fee)
end

"""Initial margin required to open a position."""
function initial_margin(me::MarginEngine, notional::Float64; leverage::Float64=1.0)
    effective_rate = max(me.init_margin_rate, 1.0 / leverage)
    return notional * effective_rate
end

"""Maintenance margin: minimum equity to avoid liquidation."""
function maintenance_margin(me::MarginEngine, notional::Float64)
    return notional * me.maint_margin_rate
end

"""
Current margin ratio: equity / notional.
Position is long: equity = collateral + unrealized PnL.
"""
function margin_ratio(me::MarginEngine, collateral::Float64, unrealized_pnl::Float64,
                       notional::Float64)
    equity = collateral + unrealized_pnl
    return equity / max(notional, 1.0)
end

"""Is the position subject to a margin call?"""
function is_margin_call(me::MarginEngine, collateral::Float64,
                          unrealized_pnl::Float64, notional::Float64)
    return margin_ratio(me, collateral, unrealized_pnl, notional) < me.maint_margin_rate
end

"""
Forced liquidation price for a long position.
p_liq = p_entry × (1 - 1/leverage + maint_margin + liquidation_fee)
"""
function compute_forced_liquidation_price(me::MarginEngine, entry_price::Float64,
                                           leverage::Float64; direction::Int=1)
    if direction == 1  # long
        return entry_price * (1.0 - 1.0/leverage + me.maint_margin_rate + me.liquidation_fee)
    else  # short
        return entry_price * (1.0 + 1.0/leverage - me.maint_margin_rate - me.liquidation_fee)
    end
end

"""Liquidation penalty: total cost of being forcibly liquidated."""
function liquidation_penalty(me::MarginEngine, notional::Float64)
    return notional * (me.liquidation_fee + me.taker_fee)
end

"""
Maximum position size given available collateral and desired leverage.
"""
function max_position_size(me::MarginEngine, collateral::Float64, leverage::Float64)
    return collateral * leverage / me.init_margin_rate / leverage
end

"""
Portfolio margin: net margin requirement for a set of correlated positions.
positions: vector of (notional, direction, beta_to_market).
"""
function portfolio_margin(me::MarginEngine, positions::Vector{<:NamedTuple},
                           corr_matrix::Matrix{Float64}, asset_vols::Vector{Float64};
                           confidence::Float64=0.99)
    n = length(positions)
    w = [p.notional * p.direction for p in positions]
    D = Diagonal(asset_vols)
    Σ = D * corr_matrix * D
    port_var = dot(w, Σ * w)
    z = 2.326  # 99th percentile
    return z * sqrt(max(0.0, port_var))
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. FundingRateModel
# ─────────────────────────────────────────────────────────────────────────────

"""
FundingRateModel: models perpetual futures funding rates.

The funding rate mechanism ensures perpetual futures prices track spot.
Funding is exchanged between longs and shorts every 8 hours.
Positive funding: longs pay shorts (premium = bullish sentiment).
Negative funding: shorts pay longs (discount = bearish sentiment).

Premium index method:
  premium = (mark_price - index_price) / index_price
  funding_rate = clamp(premium + interest_component, -cap, cap)
"""
mutable struct FundingRateModel
    interest_rate_daily::Float64    # typically 0.0001 (0.01% per 8h)
    funding_cap::Float64            # per-interval cap (e.g., 0.0075 = 0.75%)
    mean_reversion_speed::Float64   # kappa for OU dynamics
    long_run_mean::Float64          # typical long-run premium
    vol_premium::Float64            # volatility of premium
    history::Vector{Float64}        # historical 8h funding rates
    daily_history::Vector{Float64}  # daily aggregated
    n_obs::Int
end

function FundingRateModel(; interest_rate_8h::Float64=0.0001,
                            funding_cap::Float64=0.0075,
                            kappa::Float64=0.20,
                            theta::Float64=0.0001,
                            vol::Float64=0.0005)
    return FundingRateModel(interest_rate_8h, funding_cap, kappa, theta, vol,
                             Float64[], Float64[], 0)
end

"""
Compute instantaneous premium index from mark and index prices.
"""
function premium_index(mark_price::Float64, index_price::Float64)
    return (mark_price - index_price) / (index_price + 1e-10)
end

"""
Compute funding rate from premium index.
funding = clamp(premium + interest_rate, -cap, cap)
"""
function compute_funding_rate(fm::FundingRateModel, premium::Float64)
    raw = premium + fm.interest_rate_daily
    return clamp(raw, -fm.funding_cap, fm.funding_cap)
end

"""
Simulate funding rate path using OU dynamics.
Returns vector of 8h funding rates.
"""
function simulate_funding_path(fm::FundingRateModel, n_8h_periods::Int;
                                 seed::Int=42, initial_rate::Float64=0.0001)
    rng = MersenneTwister(seed)
    dt = 1.0  # each step = 1 8h period
    rates = Float64[initial_rate]
    for _ in 2:n_8h_periods
        r = rates[end]
        dr = fm.mean_reversion_speed * (fm.long_run_mean - r) * dt + fm.vol_premium * sqrt(dt) * randn(rng)
        new_r = clamp(r + dr, -fm.funding_cap, fm.funding_cap)
        push!(rates, new_r)
    end
    return rates
end

"""
Update model with a new observed funding rate.
"""
function update!(fm::FundingRateModel, funding_8h::Float64)
    push!(fm.history, funding_8h)
    fm.n_obs += 1
    # Aggregate every 3 observations into daily
    if length(fm.history) >= 3 && length(fm.history) % 3 == 0
        daily = sum(fm.history[end-2:end])
        push!(fm.daily_history, daily)
    end
end

"""
Compute expected daily funding cost for a long position (fraction of notional).
"""
function expected_daily_funding(fm::FundingRateModel; window::Int=30)
    isempty(fm.daily_history) && return fm.long_run_mean * 3  # 3 x 8h per day
    n = min(window, length(fm.daily_history))
    return mean(fm.daily_history[end-n+1:end])
end

"""
Funding carry signal: negative when funding is high (expensive to be long),
positive when funding is negative (paid to be long).
"""
function funding_carry_signal(fm::FundingRateModel; lookback::Int=5, zscore_window::Int=30)
    isempty(fm.daily_history) && return NaN
    n = length(fm.daily_history)
    n < lookback && return NaN
    # Recent avg funding vs history
    recent_avg = mean(fm.daily_history[max(1,n-lookback+1):n])
    history_window = fm.daily_history[max(1,n-zscore_window+1):n]
    mu = mean(history_window)
    sg = std(history_window)
    return -(recent_avg - mu) / (sg + 1e-8)  # negative = high funding → sell signal
end

"""Classify funding regime."""
function funding_regime(fm::FundingRateModel; threshold::Float64=0.0003)
    isempty(fm.daily_history) && return :unknown
    recent = mean(fm.daily_history[max(1,end-4):end])
    recent > threshold && return :high_contango      # expensive longs
    recent < -threshold && return :backwardation     # paid to be long
    return :neutral
end

"""Annualized cost of holding a perpetual long position."""
function annualized_holding_cost(fm::FundingRateModel; window::Int=30)
    daily_cost = expected_daily_funding(fm; window=window)
    return daily_cost * 365 * 100  # in percent
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. BasisTrader
# ─────────────────────────────────────────────────────────────────────────────

"""
BasisTrader: spot-perp convergence strategy.

The basis = (perp_price - spot_price) / spot_price should mean-revert to
zero (modified by funding costs). When the basis is abnormally wide, the
strategy shorts the perp and longs spot, earning the basis convergence
plus funding income from the short perp.

Includes:
  - Basis z-score computation
  - Entry/exit logic
  - Expected P&L and Sharpe estimation
"""
mutable struct BasisTrader
    entry_threshold_z::Float64    # enter when |basis_z| > threshold
    exit_threshold_z::Float64     # exit when |basis_z| < exit
    lookback_days::Int            # history for z-score computation
    tcost_per_leg::Float64        # cost per leg (spot + perp)
    spot_history::Vector{Float64}
    perp_history::Vector{Float64}
    basis_history::Vector{Float64}  # raw basis
    zscore_history::Vector{Float64}
    in_trade::Bool
    entry_z::Float64
    entry_basis::Float64
    pnl_history::Vector{Float64}
    n_obs::Int
end

function BasisTrader(; entry_z::Float64=1.5, exit_z::Float64=0.3,
                       lookback::Int=20, tcost::Float64=0.0005)
    return BasisTrader(entry_z, exit_z, lookback, tcost,
                        Float64[], Float64[], Float64[], Float64[],
                        false, 0.0, 0.0, Float64[], 0)
end

"""
Compute basis: (perp - spot) / spot.
"""
function raw_basis(spot::Float64, perp::Float64)
    return (perp - spot) / (spot + 1e-10)
end

"""
Compute basis z-score relative to recent history.
"""
function basis_zscore(bt::BasisTrader)
    n = length(bt.basis_history)
    n < bt.lookback_days && return NaN
    window = bt.basis_history[max(1,n-bt.lookback_days+1):n]
    mu = mean(window)
    sg = std(window)
    return (bt.basis_history[end] - mu) / (sg + 1e-8)
end

"""
Update basis trader with new spot and perp prices.
"""
function update!(bt::BasisTrader, spot::Float64, perp::Float64,
                  funding_rate::Float64=0.0)
    push!(bt.spot_history, spot)
    push!(bt.perp_history, perp)
    b = raw_basis(spot, perp)
    push!(bt.basis_history, b)
    bt.n_obs += 1

    # Z-score
    z = basis_zscore(bt)
    push!(bt.zscore_history, isnan(z) ? 0.0 : z)

    # P&L from existing trade
    pnl = 0.0
    if bt.in_trade
        # Short perp + long spot: earn basis convergence + funding
        basis_chg = bt.entry_basis - b  # basis narrowed = profit
        pnl = basis_chg + funding_rate  # funding earned (positive if longs pay)
    end
    push!(bt.pnl_history, pnl)
end

"""
Basis signal for positioning (based on z-score).
Positive = short perp/long spot (basis too wide).
"""
function basis_signal(bt::BasisTrader)
    n = length(bt.zscore_history)
    n == 0 && return 0.0
    z = bt.zscore_history[end]
    # High positive z → wide contango → short perp/long spot (positive signal)
    return z
end

"""Entry condition for basis trade."""
function enter_basis_trade(bt::BasisTrader)
    z = basis_signal(bt)
    return !bt.in_trade && abs(z) > bt.entry_threshold_z
end

"""Exit condition."""
function exit_condition(bt::BasisTrader)
    z = basis_signal(bt)
    return bt.in_trade && abs(z) < bt.exit_threshold_z
end

"""
Simulate basis trading over historical data.
spots, perps, fundings: time series of prices and funding rates.
"""
function backtest_basis_strategy(bt::BasisTrader, spots::Vector{Float64},
                                   perps::Vector{Float64},
                                   fundings::Vector{Float64})
    n = min(length(spots), length(perps), length(fundings))
    positions = zeros(n)
    daily_pnl = zeros(n)
    n_trades = 0

    for t in 1:n
        update!(bt, spots[t], perps[t], fundings[t])

        if enter_basis_trade(bt)
            bt.in_trade = true
            bt.entry_z = bt.zscore_history[end]
            bt.entry_basis = bt.basis_history[end]
            daily_pnl[t] -= bt.tcost_per_leg * 2  # entry cost for both legs
            n_trades += 1
            positions[t] = bt.zscore_history[end] > 0 ? 1.0 : -1.0
        elseif exit_condition(bt) && bt.in_trade
            bt.in_trade = false
            daily_pnl[t] -= bt.tcost_per_leg * 2  # exit cost
            positions[t] = 0.0
        else
            positions[t] = bt.in_trade ? (bt.entry_z > 0 ? 1.0 : -1.0) : 0.0
        end

        if bt.in_trade && t > 1
            daily_pnl[t] += bt.pnl_history[end]
        end
    end

    ann_ret = mean(daily_pnl) * 252 * 100
    ann_vol = std(daily_pnl) * sqrt(252) * 100
    sharpe = ann_vol > 0 ? ann_ret / ann_vol : 0.0

    return (pnl=daily_pnl, positions=positions, n_trades=n_trades,
            ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. LiquidationEngine
# ─────────────────────────────────────────────────────────────────────────────

"""
LiquidationEngine: simulates liquidation cascades and insurance fund mechanics.

Models the feedback loop:
  price drop → positions liquidated → liquidation orders increase sell pressure
              → further price drop → more liquidations → cascade

Key parameters:
  market_depth  : total USD liquidity in order book
  impact_factor : fraction of market_depth that 1x notional affects price (Kyle's λ)
  insurance_fund: available funds before socialized loss
"""
mutable struct LiquidationEngine
    market_depth::Float64       # total USD depth in book
    impact_factor::Float64      # price impact per USD sold
    insurance_fund::Float64     # current fund balance
    insurance_rate::Float64     # fraction of liquidation fee going to fund
    socialized_loss_log::Vector{Float64}
    cascade_log::Vector{NamedTuple}
end

function LiquidationEngine(; market_depth::Float64=2e9,
                              impact_factor::Float64=0.3,
                              insurance_fund::Float64=5e7,
                              insurance_rate::Float64=0.3)
    return LiquidationEngine(market_depth, impact_factor, insurance_fund,
                               insurance_rate, Float64[], NamedTuple[])
end

"""
Simulate a liquidation cascade.
positions: NamedTuple with fields (notional, leverage, entry_price, maint_margin).
"""
function simulated_cascade(engine::LiquidationEngine,
                             positions::Vector{<:NamedTuple},
                             initial_price::Float64,
                             initial_drop::Float64;
                             max_rounds::Int=30)
    # Compute liquidation prices
    liq_prices = [pos.entry_price * (1 - 1/pos.leverage + pos.maint_margin)
                  for pos in positions]
    remaining = trues(length(positions))
    price = initial_price * (1 - initial_drop)
    prices = [initial_price, price]
    total_liq_usd = 0.0
    n_liq = 0
    rounds_log = NamedTuple[]

    for round in 1:max_rounds
        newly_liq_usd = 0.0
        newly_liq_n = 0
        for (i, pos) in enumerate(positions)
            if remaining[i] && price <= liq_prices[i]
                liq_size = pos.notional
                newly_liq_usd += liq_size
                newly_liq_n += 1
                remaining[i] = false
                # Insurance fund update
                fee_collected = liq_size * 0.001
                engine.insurance_fund += fee_collected * engine.insurance_rate
            end
        end

        newly_liq_usd < 1000.0 && break

        # Price impact from forced sells
        impact = newly_liq_usd / engine.market_depth * engine.impact_factor
        price *= (1 - impact)
        push!(prices, price)
        total_liq_usd += newly_liq_usd
        n_liq += newly_liq_n
        push!(rounds_log, (round=round, price=price, liq_usd=newly_liq_usd, n_liq=n_liq))
    end

    total_drop = (initial_price - price) / initial_price
    amplification = total_drop / initial_drop
    return (prices=prices, total_liq_usd=total_liq_usd, n_liquidated=n_liq,
            total_drop=total_drop, amplification=amplification, rounds=rounds_log)
end

"""
Assess insurance fund adequacy under a stress scenario.
"""
function insurance_fund_sufficiency(engine::LiquidationEngine,
                                     stress_loss_usd::Float64;
                                     confidence::Float64=0.99)
    adequate = engine.insurance_fund >= stress_loss_usd
    coverage_ratio = engine.insurance_fund / max(stress_loss_usd, 1.0)
    shortfall = max(0.0, stress_loss_usd - engine.insurance_fund)
    return (adequate=adequate, coverage_ratio=coverage_ratio,
            shortfall=shortfall, fund_balance=engine.insurance_fund)
end

"""
Socialized loss: when insurance fund is exhausted, remaining losses
are distributed to profitable traders as a haircut.
"""
function compute_socialized_loss(engine::LiquidationEngine, total_loss::Float64)
    covered = min(engine.insurance_fund, total_loss)
    socialized = max(0.0, total_loss - covered)
    push!(engine.socialized_loss_log, socialized)
    engine.insurance_fund = max(0.0, engine.insurance_fund - covered)
    return (covered=covered, socialized=socialized)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. SlippageModel
# ─────────────────────────────────────────────────────────────────────────────

"""
SlippageModel: maps order size to expected slippage.

Uses a power-law model:
  slippage_bps = alpha × (order_size / adv)^beta
where adv = average daily volume.

For small orders (< 1% ADV): linear regime (beta ≈ 1)
For large orders (> 10% ADV): square-root regime (beta ≈ 0.5-0.7)
"""
struct SlippageModel
    alpha::Float64    # slippage coefficient (in bps)
    beta::Float64     # power (0.5 = square root, 1.0 = linear)
    adv::Float64      # average daily volume in USD
    spread_bps::Float64  # bid-ask spread contribution
end

function SlippageModel(adv::Float64; alpha::Float64=10.0, beta::Float64=0.6,
                        spread_bps::Float64=2.0)
    return SlippageModel(alpha, beta, adv, spread_bps)
end

"""
Expected slippage in bps for a given order size.
Includes spread cost + market impact.
"""
function expected_slippage(sm::SlippageModel, order_size_usd::Float64;
                             is_aggressive::Bool=true)
    participation_rate = order_size_usd / max(sm.adv, 1.0)
    impact_bps = sm.alpha * participation_rate^sm.beta
    spread_cost = is_aggressive ? sm.spread_bps / 2 : 0.0  # taker pays half spread
    return impact_bps + spread_cost
end

"""
Estimate market impact in basis points.
"""
function market_impact(sm::SlippageModel, order_size_usd::Float64)
    return sm.alpha * (order_size_usd / max(sm.adv, 1.0))^sm.beta
end

"""
Compute the fill price for a given order, given current mid price.
direction: +1 buy, -1 sell.
"""
function fill_price(sm::SlippageModel, mid_price::Float64, order_size_usd::Float64;
                     direction::Int=1)
    slip_bps = expected_slippage(sm, order_size_usd)
    slip_frac = slip_bps / 10000
    return mid_price * (1.0 + direction * slip_frac)
end

"""
VWAP-based execution: split order across N child orders.
Returns total slippage reduction vs market order.
"""
function vwap_execution_savings(sm::SlippageModel, total_order::Float64,
                                  n_slices::Int=10;
                                  time_horizon_hours::Float64=1.0)
    # Child order size
    child_size = total_order / n_slices
    # ADV over time horizon (fraction of daily)
    adv_horizon = sm.adv * time_horizon_hours / 24.0
    sm_child = SlippageModel(adv_horizon; alpha=sm.alpha, beta=sm.beta, spread_bps=sm.spread_bps)

    slip_market_order = expected_slippage(sm, total_order)
    slip_vwap = sum(expected_slippage(sm_child, child_size) for _ in 1:n_slices) / n_slices
    savings_bps = slip_market_order - slip_vwap
    return (market_order_slip=slip_market_order, vwap_slip=slip_vwap, savings=savings_bps)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. CircuitBreaker
# ─────────────────────────────────────────────────────────────────────────────

"""
CircuitBreaker: models exchange-level and index-level circuit breakers.

Crypto exchanges implement:
  1. Price band limits: no trading outside a % band from reference price
  2. Volatility interruptions: halt if price moves X% in Y minutes
  3. Mark price deviation: funding rate extreme = reduce position limits

Includes simulation of trading halts and their impact on strategy.
"""
mutable struct CircuitBreaker
    price_band_pct::Float64       # maximum % move before halt (e.g., 0.15)
    volatility_threshold_pct::Float64  # % move in window to trigger
    volatility_window_mins::Int   # rolling window for vol check
    halt_duration_mins::Int       # how long the halt lasts
    reference_price::Float64
    price_history::Vector{Float64}
    time_history::Vector{Float64}  # in minutes
    halt_log::Vector{NamedTuple}
    n_obs::Int
end

function CircuitBreaker(; price_band::Float64=0.15, vol_threshold::Float64=0.10,
                           vol_window::Int=5, halt_duration::Int=10,
                           reference_price::Float64=50000.0)
    return CircuitBreaker(price_band, vol_threshold, vol_window, halt_duration,
                           reference_price, Float64[], Float64[], NamedTuple[], 0)
end

"""
Check if current price triggers any circuit breaker.
Returns (triggered, reason, halt_duration).
"""
function is_circuit_breaker_triggered(cb::CircuitBreaker, price::Float64,
                                        current_time_mins::Float64)
    push!(cb.price_history, price)
    push!(cb.time_history, current_time_mins)
    cb.n_obs += 1

    # Check 1: price band from reference
    band_move = abs(price - cb.reference_price) / cb.reference_price
    if band_move > cb.price_band_pct
        push!(cb.halt_log, (time=current_time_mins, reason="price_band",
                             price=price, move_pct=band_move*100))
        return (triggered=true, reason="price_band", halt_mins=cb.halt_duration_mins)
    end

    # Check 2: volatility in rolling window
    n = length(cb.price_history)
    if n > 2
        window_start = current_time_mins - cb.volatility_window_mins
        in_window = cb.time_history .>= window_start
        if sum(in_window) >= 2
            window_prices = cb.price_history[in_window]
            vol_move = (maximum(window_prices) - minimum(window_prices)) / minimum(window_prices)
            if vol_move > cb.volatility_threshold_pct
                push!(cb.halt_log, (time=current_time_mins, reason="volatility",
                                     price=price, move_pct=vol_move*100))
                return (triggered=true, reason="volatility", halt_mins=cb.halt_duration_mins)
            end
        end
    end

    return (triggered=false, reason="none", halt_mins=0)
end

"""
Price band: return allowed trading range around reference price.
"""
function price_band(cb::CircuitBreaker)
    lo = cb.reference_price * (1 - cb.price_band_pct)
    hi = cb.reference_price * (1 + cb.price_band_pct)
    return (lower=lo, upper=hi)
end

"""
Estimate halt duration based on historical volatility.
Longer halts for more extreme moves.
"""
function halt_duration(cb::CircuitBreaker, move_pct::Float64)
    # Base duration, extended for more extreme moves
    multiplier = 1.0 + max(0.0, (move_pct - cb.price_band_pct * 100) / 5.0)
    return round(Int, cb.halt_duration_mins * multiplier)
end

"""
Simulate a price path and identify circuit breaker events.
Returns list of halt events.
"""
function simulate_trading_session(cb::CircuitBreaker, price_path::Vector{Float64};
                                    interval_mins::Float64=1.0)
    halts = NamedTuple[]
    halt_end_time = -Inf
    total_halt_mins = 0.0

    for (t, price) in enumerate(price_path)
        time_mins = t * interval_mins
        time_mins <= halt_end_time && continue  # in halt

        result = is_circuit_breaker_triggered(cb, price, time_mins)
        if result.triggered
            push!(halts, (time=time_mins, reason=result.reason,
                           price=price, halt_mins=result.halt_mins))
            halt_end_time = time_mins + result.halt_mins
            total_halt_mins += result.halt_mins
        end
    end

    pct_time_halted = total_halt_mins / (length(price_path) * interval_mins) * 100
    return (halts=halts, n_halts=length(halts),
            total_halt_mins=total_halt_mins, pct_time_halted=pct_time_halted)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Module-level utilities
# ─────────────────────────────────────────────────────────────────────────────

"""
Generate a synthetic trading day for testing mechanics.
Returns prices, volumes, funding rates.
"""
function synthetic_trading_day(n_minutes::Int=1440; seed::Int=42,
                                 initial_price::Float64=85000.0,
                                 annual_vol::Float64=0.80)
    rng = MersenneTwister(seed)
    dt = 1.0 / (252 * 1440)  # 1 minute in years
    sigma_per_min = annual_vol * sqrt(dt)
    prices = Float64[initial_price]
    for _ in 2:n_minutes
        r = sigma_per_min * randn(rng)
        push!(prices, prices[end] * exp(r))
    end

    # Volume: U-shaped intraday pattern
    vol_pattern = [1.5 - cos(2*pi*t/1440) for t in 1:n_minutes]
    base_vol = 100_000_000.0  # $100M daily volume
    volumes = vol_pattern .* (base_vol / sum(vol_pattern))

    # Funding: 3 per day, driven by price moves
    funding_times = [480, 960, 1440]
    cum_ret = (prices[end] - prices[1]) / prices[1]
    funding_8h = 0.0001 + 0.003 * cum_ret + randn(rng) * 0.0002
    funding_8h = clamp(funding_8h, -0.0075, 0.0075)

    return (prices=prices, volumes=volumes, funding=funding_8h)
end

end  # module CryptoMarketMechanics
