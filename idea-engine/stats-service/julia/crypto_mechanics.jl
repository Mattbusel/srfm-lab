module CryptoMechanics

# ============================================================
# crypto_mechanics.jl -- Crypto Trading Mechanics & Strategies
# ============================================================
# Covers: funding rate arbitrage, cash-and-carry basis trading,
# cross-exchange spread capture, perpetual-spot convergence,
# calendar spreads, options basis, delta-neutral hedging,
# liquidation cascade risk, PnL attribution, position sizing,
# multi-leg strategy construction, Greeks for crypto options.
# Pure stdlib Julia -- no external packages.
# ============================================================

using Statistics, LinearAlgebra

struct PerpContract
    symbol::String
    mark_price::Float64
    index_price::Float64
    funding_rate_8h::Float64
    open_interest::Float64
    volume_24h::Float64
    max_leverage::Float64
end

struct SpotMarket
    symbol::String
    bid::Float64
    ask::Float64
    last::Float64
    volume_24h::Float64
    exchange::String
end

struct FundingArb
    long_spot::Float64
    short_perp::Float64
    net_notional::Float64
    funding_rate_8h::Float64
    holding_days::Float64
    borrow_rate_annual::Float64
end

struct CalendarSpread
    near_symbol::String
    far_symbol::String
    near_price::Float64
    far_price::Float64
    near_expiry_days::Int
    far_expiry_days::Int
    basis_pct::Float64
end

struct CrossExchangeSpread
    buy_exchange::String
    sell_exchange::String
    buy_price::Float64
    sell_price::Float64
    transfer_fee::Float64
    transfer_time_hrs::Float64
    spread_pct::Float64
end

struct CryptoOption
    underlying::String
    strike::Float64
    expiry_days::Int
    call_or_put::Symbol
    iv::Float64
    premium::Float64
    delta::Float64
    gamma::Float64
    vega::Float64
    theta::Float64
end

struct LiquidationLevel
    position_size::Float64
    entry_price::Float64
    leverage::Float64
    side::Symbol
    maintenance_margin::Float64
    liq_price::Float64
end

struct FundingRateHistory
    timestamps::Vector{Float64}
    rates_8h::Vector{Float64}
    mark_prices::Vector{Float64}
    index_prices::Vector{Float64}
end

# ---- 1. Funding Rate Analysis ----

function funding_rate_premium(mark::Float64, index::Float64)::Float64
    return (mark - index) / (index + 1e-8)
end

function funding_8h_annualised(rate_8h::Float64)::Float64
    return rate_8h * 3 * 365 * 100.0
end

function funding_arb_pnl(arb::FundingArb)::NamedTuple
    daily_funding = arb.funding_rate_8h * 3 * arb.net_notional
    total_funding = daily_funding * arb.holding_days
    borrow_cost   = arb.net_notional * arb.borrow_rate_annual * arb.holding_days / 365.0
    net_pnl       = total_funding - borrow_cost
    ann_yield     = net_pnl / (arb.net_notional + 1e-8) * (365.0 / arb.holding_days) * 100.0
    breakeven_fr  = arb.borrow_rate_annual / (3*365*100)
    return (
        daily_funding_income = daily_funding,
        total_funding_income = total_funding,
        borrow_cost          = borrow_cost,
        net_pnl              = net_pnl,
        annualised_yield_pct = ann_yield,
        breakeven_rate_8h    = breakeven_fr,
        is_profitable        = net_pnl > 0,
    )
end

function funding_rate_signal(hist::FundingRateHistory, lookback::Int=30)
    n = length(hist.rates_8h)
    recent = hist.rates_8h[max(1,n-lookback*3):n]
    avg_rate = mean(recent)
    std_rate = std(recent) + 1e-8
    latest = hist.rates_8h[end]
    z = (latest - avg_rate) / std_rate
    signal = z > 2 ? :extreme_longs_paying : z < -2 ? :extreme_shorts_paying : :normal
    cumulative_annual = sum(recent) * 3 * 365 / length(recent) * 100.0
    return (
        latest_rate = latest,
        avg_rate    = avg_rate,
        z_score     = z,
        signal      = signal,
        implied_annual_pct = cumulative_annual,
        percentile  = count(r -> r < latest, recent) / length(recent) * 100.0,
    )
end

function optimal_funding_entry(hist::FundingRateHistory, z_threshold::Float64=2.0)
    n = length(hist.rates_8h)
    entries = Int[]
    rates = hist.rates_8h
    for i in 21:n
        window = rates[i-20:i-1]
        z = (rates[i] - mean(window)) / (std(window) + 1e-8)
        if z > z_threshold
            push!(entries, i)
        end
    end
    return entries
end

# ---- 2. Basis Trading ----

function perp_spot_basis(perp_price::Float64, spot_price::Float64)::Float64
    return (perp_price - spot_price) / spot_price * 100.0
end

function basis_trade_pnl(perp_price::Float64, spot_price::Float64,
                          size::Float64, days::Float64,
                          funding_rate_daily::Float64, exit_basis_pct::Float64)
    entry_basis = perp_spot_basis(perp_price, spot_price)
    funding_income = funding_rate_daily * perp_price * size * days
    basis_convergence = (entry_basis - exit_basis_pct) / 100.0 * spot_price * size
    total_pnl = funding_income + basis_convergence
    ann_yield = total_pnl / (spot_price * size + 1e-8) * (365.0/days) * 100.0
    return (
        entry_basis_pct    = entry_basis,
        funding_income     = funding_income,
        basis_pnl          = basis_convergence,
        total_pnl          = total_pnl,
        annualised_yield   = ann_yield,
    )
end

function carry_adjusted_basis(basis_pct::Float64, funding_rate_8h::Float64,
                               holding_days::Float64)::Float64
    funding_carry = funding_rate_8h * 3 * holding_days * 100.0
    return basis_pct - funding_carry
end

function roll_yield(near_price::Float64, far_price::Float64,
                     days_to_near_expiry::Int, days_to_far_expiry::Int)::Float64
    spread = far_price - near_price
    days_diff = far_expiry_days - days_to_near_expiry
    return (spread / near_price) * (365.0 / (days_diff + 1e-8)) * 100.0
end

# ---- 3. Calendar Spreads ----

function calendar_spread_value(spread::CalendarSpread)::Float64
    return spread.far_price - spread.near_price
end

function calendar_implied_rate(spread::CalendarSpread)::Float64
    fwd_rate = (spread.far_price / spread.near_price - 1.0)
    days_diff = spread.far_expiry_days - spread.near_expiry_days
    return fwd_rate * (365.0 / (days_diff + 1e-8)) * 100.0
end

function optimal_calendar_leg_ratio(near_vega::Float64, far_vega::Float64)::Float64
    return far_vega / (near_vega + 1e-8)
end

function calendar_theta_decay(near_theta::Float64, far_theta::Float64,
                                leg_ratio::Float64)::Float64
    return far_theta * leg_ratio - near_theta
end

# ---- 4. Cross-Exchange Arbitrage ----

function cross_exchange_arb_profit(spread::CrossExchangeSpread,
                                    size::Float64)::NamedTuple
    gross_profit = (spread.sell_price - spread.buy_price) * size
    transfer_fees = spread.transfer_fee * size
    execution_slippage = (spread.sell_price + spread.buy_price) * 0.0005 * size
    net_profit = gross_profit - transfer_fees - execution_slippage
    return (
        gross_profit        = gross_profit,
        transfer_fees       = transfer_fees,
        slippage_cost       = execution_slippage,
        net_profit          = net_profit,
        spread_pct          = spread.spread_pct,
        is_profitable       = net_profit > 0,
        min_size_profitable = (transfer_fees + execution_slippage) /
                              max(gross_profit / size - 1e-8, 1e-8),
    )
end

function triangular_arb(price_ab::Float64, price_bc::Float64,
                          price_ac::Float64, fee_pct::Float64=0.001)::Float64
    # A -> B -> C -> A path profit
    fee = 1 - fee_pct
    return price_ab * fee * (1/price_bc) * fee * price_ac * fee - 1.0
end

function stat_arb_zscore(spread::Vector{Float64}, window::Int=30)::Vector{Float64}
    n = length(spread); z = fill(NaN, n)
    for i in (window+1):n
        h = spread[i-window:i-1]
        z[i] = (spread[i] - mean(h)) / (std(h) + 1e-8)
    end
    return z
end

# ---- 5. Liquidation Mechanics ----

function liquidation_price(liq::LiquidationLevel)::Float64
    mm = liq.maintenance_margin
    if liq.side == :long
        return liq.entry_price * (1.0 - 1.0/liq.leverage + mm)
    else
        return liq.entry_price * (1.0 + 1.0/liq.leverage - mm)
    end
end

function liquidation_cascade_volume(positions::Vector{LiquidationLevel},
                                     price_drop_pct::Float64)::Float64
    total_liq = 0.0
    for pos in positions
        lp = liquidation_price(pos)
        price = pos.entry_price * (1 - price_drop_pct)
        if pos.side == :long && price <= lp
            total_liq += pos.position_size * pos.entry_price
        elseif pos.side == :short && price >= lp
            total_liq += pos.position_size * pos.entry_price
        end
    end
    return total_liq
end

function max_position_size(account_equity::Float64, leverage::Float64,
                            price::Float64, risk_pct::Float64=0.02)::Float64
    risk_dollars = account_equity * risk_pct
    return risk_dollars * leverage / (price + 1e-8)
end

function kelly_position_size(win_rate::Float64, win_size::Float64,
                              loss_size::Float64)::Float64
    return win_rate/loss_size - (1-win_rate)/win_size
end

# ---- 6. Delta-Neutral Hedging ----

function delta_hedge_ratio(option::CryptoOption, position_size::Float64)::Float64
    return -option.delta * position_size
end

function gamma_scalp_pnl(gamma::Float64, price_move::Float64,
                           position_size::Float64)::Float64
    return 0.5 * gamma * price_move^2 * position_size
end

function vega_pnl(vega::Float64, vol_move::Float64, position_size::Float64)::Float64
    return vega * vol_move * position_size
end

function theta_decay_daily(theta::Float64, position_size::Float64)::Float64
    return theta * position_size / 365.0
end

function options_pnl_components(option::CryptoOption, price_move::Float64,
                                  vol_move::Float64, time_elapsed_days::Float64,
                                  position_size::Float64)::NamedTuple
    delta_pnl = option.delta * price_move * position_size
    gamma_pnl = 0.5 * option.gamma * price_move^2 * position_size
    vega_pnl_val = option.vega * vol_move * position_size
    theta_pnl = option.theta * time_elapsed_days * position_size
    total = delta_pnl + gamma_pnl + vega_pnl_val + theta_pnl
    return (delta=delta_pnl, gamma=gamma_pnl, vega=vega_pnl_val,
            theta=theta_pnl, total=total)
end

# ---- 7. Crypto BS Option Pricing ----

function crypto_bs_call(S::Float64, K::Float64, r::Float64, q::Float64,
                         sigma::Float64, T::Float64)::Float64
    d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T) + 1e-12)
    d2 = d1 - sigma*sqrt(T)
    nd1 = 0.5*(1 + erf(d1/sqrt(2))); nd2 = 0.5*(1 + erf(d2/sqrt(2)))
    return S*exp(-q*T)*nd1 - K*exp(-r*T)*nd2
end

function crypto_bs_put(S::Float64, K::Float64, r::Float64, q::Float64,
                        sigma::Float64, T::Float64)::Float64
    d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T) + 1e-12)
    d2 = d1 - sigma*sqrt(T)
    nd1m = 0.5*(1 - erf(d1/sqrt(2))); nd2m = 0.5*(1 - erf(d2/sqrt(2)))
    return K*exp(-r*T)*nd2m - S*exp(-q*T)*nd1m
end

function crypto_bs_greeks(S::Float64, K::Float64, r::Float64, q::Float64,
                           sigma::Float64, T::Float64, opt::Symbol=:call)::CryptoOption
    d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T) + 1e-12)
    d2 = d1 - sigma*sqrt(T)
    nd1 = 0.5*(1 + erf(d1/sqrt(2))); nd2 = 0.5*(1 + erf(d2/sqrt(2)))
    phi_d1 = exp(-0.5*d1^2)/sqrt(2pi)
    delta = opt==:call ? exp(-q*T)*nd1 : exp(-q*T)*(nd1-1)
    gamma = exp(-q*T)*phi_d1 / (S*sigma*sqrt(T) + 1e-12)
    vega  = S*exp(-q*T)*phi_d1*sqrt(T)
    theta = opt==:call ?
        (-S*exp(-q*T)*phi_d1*sigma/(2*sqrt(T)+1e-12) - r*K*exp(-r*T)*nd2 + q*S*exp(-q*T)*nd1) :
        (-S*exp(-q*T)*phi_d1*sigma/(2*sqrt(T)+1e-12) + r*K*exp(-r*T)*(1-nd2) - q*S*exp(-q*T)*(1-nd1))
    price = opt==:call ? crypto_bs_call(S,K,r,q,sigma,T) : crypto_bs_put(S,K,r,q,sigma,T)
    return CryptoOption("CRYPTO", K, round(Int,T*365), opt, sigma, price, delta, gamma, vega, theta)
end

# ---- 8. Market Microstructure (Crypto) ----

function bid_ask_spread_bps(bid::Float64, ask::Float64)::Float64
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 10000.0
end

function effective_spread(executed_price::Float64, bid::Float64, ask::Float64)::Float64
    mid = (bid + ask) / 2.0
    return 2.0 * abs(executed_price - mid) / mid * 10000.0
end

function order_book_imbalance(bid_qty::Float64, ask_qty::Float64)::Float64
    return (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-8)
end

function kyle_lambda(price_changes::Vector{Float64}, order_flows::Vector{Float64})::Float64
    n = length(price_changes)
    x_bar = mean(order_flows); y_bar = mean(price_changes)
    cov_val = sum((order_flows .- x_bar) .* (price_changes .- y_bar)) / (n-1+1e-8)
    var_x = var(order_flows) + 1e-12
    return cov_val / var_x
end

# ---- 9. Portfolio Risk for Crypto ----

function crypto_portfolio_var(prices::Matrix{Float64}, weights::Vector{Float64},
                               confidence::Float64=0.99, horizon_days::Int=1)::Float64
    log_rets = diff(log.(prices), dims=1)
    port_rets = log_rets * weights
    sorted = sort(port_rets)
    cutoff_idx = max(1, round(Int, (1-confidence)*length(sorted)))
    var_1d = -sorted[cutoff_idx]
    return var_1d * sqrt(horizon_days)
end

function max_loss_estimate(prices::Matrix{Float64}, weights::Vector{Float64},
                            window::Int=252)::Float64
    log_rets = diff(log.(prices), dims=1)
    n = size(log_rets, 1)
    if n < window; return NaN; end
    port_rets = log_rets[end-window+1:end, :] * weights
    return -minimum(port_rets)
end

# ---- 10. Strategy Backtesting Helpers ----

function funding_arb_backtest(hist::FundingRateHistory,
                               entry_z::Float64=2.0, exit_z::Float64=0.5)
    n = length(hist.rates_8h)
    pnl = zeros(n); in_trade = false; entry_rate = 0.0
    window = 30
    for i in (window+1):n
        recent = hist.rates_8h[i-window:i-1]
        z = (hist.rates_8h[i] - mean(recent)) / (std(recent)+1e-8)
        if !in_trade && z > entry_z
            in_trade = true; entry_rate = hist.rates_8h[i]
        elseif in_trade
            pnl[i] = hist.rates_8h[i]  # collect funding
            if abs(z) < exit_z
                in_trade = false
            end
        end
    end
    total_pnl = sum(pnl) * 3 * 365  # annualised
    return (pnl_series=pnl, total_annualised=total_pnl,
            n_trades=count(p -> p > 0, pnl))
end

function stat_arb_backtest(spread::Vector{Float64}, entry_z::Float64=2.0,
                             exit_z::Float64=0.0, window::Int=30)
    n = length(spread); pnl = zeros(n); in_trade = false; entry_spread = 0.0; side = 1
    for i in (window+1):n
        h = spread[i-window:i-1]
        mu = mean(h); sig = std(h)+1e-8; z = (spread[i]-mu)/sig
        if !in_trade
            if z > entry_z; in_trade=true; entry_spread=spread[i]; side=-1
            elseif z < -entry_z; in_trade=true; entry_spread=spread[i]; side=1
            end
        else
            pnl[i] = side * (spread[i] - entry_spread)
            if abs(z) < exit_z; in_trade=false; entry_spread=0.0; end
        end
    end
    return (pnl=pnl, total=sum(pnl), sharpe=mean(pnl[pnl.!=0])/(std(pnl[pnl.!=0])+1e-8)*sqrt(252))
end

# ---- Demo ----

function demo()
    println("=== CryptoMechanics Demo ===")

    arb = FundingArb(100.0, 100.0, 10000.0, 0.0003, 30.0, 0.05)
    res = funding_arb_pnl(arb)
    println("Funding arb (30d, FR=0.03% 8h):")
    println("  Total funding income: \$", round(res.total_funding_income, digits=2))
    println("  Borrow cost: \$", round(res.borrow_cost, digits=2))
    println("  Net PnL: \$", round(res.net_pnl, digits=2))
    println("  Annualised yield: ", round(res.annualised_yield_pct, digits=2), "%")

    spread_obj = CrossExchangeSpread("Binance","OKX",30000.0,30120.0,5.0,2.0,0.4)
    arb_res = cross_exchange_arb_profit(spread_obj, 1.0)
    println("\nCross-exchange arb:")
    println("  Gross profit: \$", round(arb_res.gross_profit, digits=2))
    println("  Net profit: \$", round(arb_res.net_profit, digits=2))
    println("  Profitable: ", arb_res.is_profitable)

    opt = crypto_bs_greeks(30000.0, 30000.0, 0.05, 0.0, 0.80, 30/365, :call)
    println("\nATM call (BTC 30d, vol=80%):")
    println("  Premium: \$", round(opt.premium, digits=2))
    println("  Delta: ", round(opt.delta, digits=4))
    println("  Gamma: ", round(opt.gamma, digits=6))
    println("  Vega:  ", round(opt.vega, digits=2))
    println("  Theta: ", round(opt.theta, digits=2))

    liq = LiquidationLevel(1.0, 30000.0, 10.0, :long, 0.005, 0.0)
    lp = liquidation_price(liq)
    println("\nLong 1 BTC at 30000 with 10x leverage:")
    println("  Liquidation price: \$", round(lp, digits=2))

    fr_times = collect(Float64, 1:100)
    fr_rates = 0.0001 .+ 0.0002 .* sin.(fr_times ./ 10)
    fr_rates[90] = 0.0015  # spike
    hist = FundingRateHistory(fr_times, fr_rates, 30000.0 .* ones(100), 29900.0 .* ones(100))
    sig = funding_rate_signal(hist, 20)
    println("\nFunding rate signal: ", sig.signal, " (z=", round(sig.z_score, digits=2), ")")

    basis = perp_spot_basis(30120.0, 30000.0)
    println("\nPerp-spot basis: ", round(basis, digits=3), "%")
    println("Annualised funding (0.03% 8h): ", round(funding_8h_annualised(0.0003), digits=2), "%")
end

# ---- Additional Crypto Mechanics Functions ----

function funding_rate_prediction(hist::FundingRateHistory, horizon::Int=3)::Float64
    n = length(hist.rates_8h)
    if n < 10; return hist.rates_8h[end]; end
    rates = hist.rates_8h
    slope = (rates[end] - rates[end-min(9,n-1)]) / min(9.0,Float64(n-1))
    return rates[end] + slope * horizon
end

function cross_exchange_volume_ratio(vol_a::Float64, vol_b::Float64)::Float64
    total = vol_a + vol_b + 1e-8
    return vol_a / total
end

function perp_premium_index(perp_price::Float64, spot_price::Float64,
                              risk_free::Float64=0.05)::Float64
    theo_premium = risk_free / (3*365) * spot_price
    actual_premium = perp_price - spot_price
    return (actual_premium - theo_premium) / spot_price * 1e4
end

function liquidation_heatmap(price_levels::Vector{Float64},
                               positions::Vector{LiquidationLevel})::Vector{Float64}
    n = length(price_levels); heatmap = zeros(n)
    for pos in positions
        lp = liquidation_price(pos)
        idx = argmin(abs.(price_levels .- lp))
        heatmap[idx] += pos.position_size * pos.entry_price
    end
    return heatmap
end

function portfolio_delta_neutral(options::Vector{CryptoOption},
                                   spot_holdings::Float64)::Float64
    total_delta = sum(o.delta for o in options) + spot_holdings
    return -total_delta
end

function gamma_scalping_threshold(gamma::Float64, theta::Float64,
                                    position_size::Float64)::Float64
    daily_theta = abs(theta) * position_size / 365.0
    min_move = sqrt(2 * daily_theta / (gamma * position_size + 1e-12))
    return min_move
end

function volatility_carry_trade(realised_vol::Float64, implied_vol::Float64,
                                  vega::Float64, position_size::Float64)::NamedTuple
    carry = implied_vol - realised_vol
    daily_pnl = vega * carry / sqrt(252.0) * position_size
    ann_carry = daily_pnl * 252.0
    breakeven_vol = implied_vol
    return (vol_carry=carry, daily_pnl=daily_pnl, ann_carry_pnl=ann_carry,
            breakeven_realised_vol=breakeven_vol)
end

function spot_futures_convergence_rate(basis_now::Float64, basis_at_expiry::Float64,
                                        days_to_expiry::Float64)::Float64
    if days_to_expiry < 0.5; return 0.0; end
    return (basis_now - basis_at_expiry) / days_to_expiry
end

function cross_margin_efficiency(margin_a::Float64, margin_b::Float64,
                                   correlation::Float64)::Float64
    combined_margin = sqrt(margin_a^2 + margin_b^2 + 2*correlation*margin_a*margin_b)
    sum_margins = margin_a + margin_b
    return 1.0 - combined_margin / (sum_margins + 1e-8)
end

function funding_carry_breakeven_vol(funding_rate_annual::Float64,
                                      delta::Float64, vega::Float64)::Float64
    return sqrt(abs(funding_rate_annual / (vega * delta + 1e-8)))
end

function multi_exchange_best_execution(exchanges::Vector{SpotMarket},
                                        order_size::Float64)::NamedTuple
    best_buy = argmin([ex.ask for ex in exchanges])
    best_sell = argmax([ex.bid for ex in exchanges])
    best_ask = exchanges[best_buy].ask
    best_bid = exchanges[best_sell].bid
    spread = best_bid - best_ask
    return (best_buy_exchange=exchanges[best_buy].exchange,
            best_sell_exchange=exchanges[best_sell].exchange,
            best_ask=best_ask, best_bid=best_bid,
            spread_bps=spread/best_ask*1e4,
            can_arb=best_bid > best_ask)
end

function perpetual_swap_pnl(entry_price::Float64, exit_price::Float64,
                              size::Float64, side::Symbol,
                              funding_payments::Vector{Float64})::NamedTuple
    sign_val = side == :long ? 1.0 : -1.0
    price_pnl = sign_val * (exit_price - entry_price) * size
    funding_pnl = side == :long ? -sum(funding_payments)*size :
                                    sum(funding_payments)*size
    total = price_pnl + funding_pnl
    return (price_pnl=price_pnl, funding_pnl=funding_pnl, total_pnl=total,
            return_pct=total/(entry_price*size)*100)
end


# ---- Crypto Mechanics Utilities (continued) ----

function basis_arbitrage_signal(basis_history::Vector{Float64}, window::Int=14)
    n = length(basis_history); if n < window + 1; return (z=NaN, signal=:insufficient); end
    hist = basis_history[end-window:end-1]; cur = basis_history[end]
    z = (cur - mean(hist)) / (std(hist) + 1e-8)
    sig = z > 2 ? :wide_basis_sell_perp : z < -2 ? :narrow_basis_buy_perp : :neutral
    return (z=z, signal=sig, current_basis=cur, hist_avg=mean(hist))
end

function cross_coin_correlation(prices_a::Vector{Float64}, prices_b::Vector{Float64},
                                  window::Int=30)::Float64
    n = min(length(prices_a), length(prices_b))
    if n < window + 1; return NaN; end
    ra = diff(log.(prices_a[end-window:end]))
    rb = diff(log.(prices_b[end-window:end]))
    return cor(ra, rb)
end

function volatility_regime(price_series::Vector{Float64},
                             window_short::Int=7, window_long::Int=30)
    n = length(price_series)
    if n < window_long + 1; return :insufficient_data; end
    rets = diff(log.(price_series))
    vol_s = std(rets[end-window_short+1:end]) * sqrt(365.0)
    vol_l = std(rets[end-window_long+1:end])  * sqrt(365.0)
    ratio = vol_s / (vol_l + 1e-8)
    return ratio > 1.5 ? :high_vol_regime : ratio < 0.67 ? :low_vol_regime : :normal_regime
end

function defi_vs_cefi_spread(defi_rate::Float64, cefi_rate::Float64)::Float64
    return (defi_rate - cefi_rate) * 100.0
end

function options_market_stress_indicator(skew_25d::Float64,
                                          term_slope::Float64,
                                          vol_level::Float64)::Float64
    skew_z = skew_25d / 0.05
    slope_z = -term_slope / 0.02
    vol_z   = (vol_level - 0.6) / 0.3
    return (skew_z + slope_z + vol_z) / 3.0
end

function miner_capitulation_signal(hash_rate::Vector{Float64},
                                    btc_price::Vector{Float64},
                                    window::Int=14)::Symbol
    n = min(length(hash_rate), length(btc_price))
    if n < window + 1; return :insufficient_data; end
    hr_mom = (hash_rate[n] - hash_rate[n-window]) / (hash_rate[n-window] + 1e-8)
    px_mom = (btc_price[n] - btc_price[n-window]) / (btc_price[n-window] + 1e-8)
    if hr_mom < -0.1 && px_mom < -0.1; return :capitulation
    elseif hr_mom > 0.1; return :expansion
    else; return :stable
    end
end

function stablecoin_depeg_risk(peg_price::Float64, current_price::Float64,
                                 backing_ratio::Float64)::NamedTuple
    deviation_bps = (current_price - peg_price) / peg_price * 1e4
    backing_buffer = (backing_ratio - 1.0) * 100.0
    risk = abs(deviation_bps) > 50 || backing_ratio < 1.05 ? :high :
           abs(deviation_bps) > 20 ? :medium : :low
    return (deviation_bps=deviation_bps, backing_buffer_pct=backing_buffer, risk=risk)
end

function crypto_tax_lot_pnl(purchase_price::Float64, current_price::Float64,
                              quantity::Float64, holding_days::Int)::NamedTuple
    unrealised_pnl = (current_price - purchase_price) * quantity
    pct_return = (current_price - purchase_price) / (purchase_price + 1e-8) * 100.0
    lt_eligible = holding_days >= 365
    return (unrealised_pnl=unrealised_pnl, pct_return=pct_return,
            long_term_eligible=lt_eligible, holding_days=holding_days)
end


# ============================================================
# SECTION 2: ADVANCED TRADING ANALYTICS
# ============================================================

struct TWAPOrder
    symbol::String
    total_qty::Float64
    start_time::Float64   # unix seconds
    end_time::Float64
    num_slices::Int
    filled_qty::Float64
    avg_fill_price::Float64
end

struct VWAPBenchmark
    symbol::String
    market_volume::Vector{Float64}
    market_price::Vector{Float64}
    participation_rate::Float64
    vwap_price::Float64
    vwap_slippage::Float64
end

function compute_vwap(prices::Vector{Float64}, volumes::Vector{Float64})
    total_vol = sum(volumes)
    total_vol < 1e-10 && return mean(prices)
    return sum(prices .* volumes) / total_vol
end

function twap_schedule(total_qty::Float64, duration_min::Float64,
                        num_slices::Int; randomize::Bool=false)
    slice_qty = total_qty / num_slices
    interval = duration_min / num_slices
    schedule = [(t * interval, randomize ? slice_qty * (0.9 + 0.2*rand()) : slice_qty)
                for t in 0:num_slices-1]
    return schedule
end

function vwap_schedule(total_qty::Float64, volume_profile::Vector{Float64})
    # Volume-proportional participation
    n = length(volume_profile)
    total_vol_profile = sum(volume_profile)
    if total_vol_profile < 1e-10
        return fill(total_qty / n, n)
    end
    return total_qty .* volume_profile ./ total_vol_profile
end

function implementation_shortfall(decision_price::Float64, execution_prices::Vector{Float64},
                                    quantities::Vector{Float64}, total_shares::Float64)
    # IS = (weighted avg execution price - decision price) / decision price
    wavg = sum(execution_prices .* quantities) / (sum(quantities) + 1e-10)
    delay_cost    = (execution_prices[1] - decision_price) / (decision_price + 1e-10)
    market_impact = (wavg - execution_prices[1]) / (execution_prices[1] + 1e-10)
    timing_risk   = std(execution_prices) / (mean(execution_prices) + 1e-10)
    total_is      = (wavg - decision_price) / (decision_price + 1e-10)
    return (total_is=total_is, delay_cost=delay_cost,
            market_impact=market_impact, timing_risk=timing_risk, wavg_px=wavg)
end

function almgren_chriss_optimal_trajectory(q0::Float64, T::Float64, sigma::Float64,
                                             eta::Float64, gamma::Float64,
                                             risk_aversion::Float64; n_steps::Int=20)
    # Almgren-Chriss liquidation model
    # eta: temporary impact, gamma: permanent impact
    kappa = sqrt(risk_aversion * sigma^2 / eta)
    sinh_kT = sinh(kappa * T)
    trajectory = Float64[]
    times = Float64[]
    for i in 0:n_steps
        t = i * T / n_steps
        qt = q0 * sinh(kappa * (T - t)) / (sinh_kT + 1e-15)
        push!(trajectory, qt); push!(times, t)
    end
    trade_list = -diff(trajectory)
    # Expected cost
    E_cost = 0.5 * gamma * q0^2 + eta * q0 * kappa / tanh(kappa * T / 2 + 1e-15)
    # Variance
    V_cost = 0.5 * sigma^2 * q0^2 * T / n_steps
    return (trajectory=trajectory, times=times, trade_list=trade_list,
            expected_cost=E_cost, cost_variance=V_cost)
end

function participation_rate_model(target_qty::Float64, market_volume_rate::Float64,
                                    max_participation::Float64=0.2)
    # How long to execute given participation rate constraint
    rate = min(max_participation * market_volume_rate, market_volume_rate)
    duration = target_qty / (rate + 1e-10)
    return (execution_rate=rate, duration_seconds=duration, pct_of_market=rate/market_volume_rate)
end

# ============================================================
# SECTION 3: FUNDING RATE ANALYTICS
# ============================================================

struct FundingAnalytics
    symbol::String
    funding_rates::Vector{Float64}  # 8h rates
    timestamps::Vector{Float64}
    annualized_avg::Float64
    carry_cost::Float64
    predicted_next::Float64
end

function analyze_funding_history(funding_rates_8h::Vector{Float64},
                                   timestamps::Vector{Float64},
                                   symbol::String="BTCUSDT")
    annualized = mean(funding_rates_8h) * 3 * 365  # 3 funding periods/day
    # Cumulative carry cost
    carry = sum(funding_rates_8h)
    # Simple AR(1) prediction
    n = length(funding_rates_8h)
    predicted = n >= 2 ? funding_rates_8h[end] + 0.5*(funding_rates_8h[end]-funding_rates_8h[end-1]) :
                         funding_rates_8h[end]
    predicted = clamp(predicted, -0.005, 0.005)  # cap at ±0.5%
    return FundingAnalytics(symbol, funding_rates_8h, timestamps, annualized, carry, predicted)
end

function funding_arbitrage_pnl(spot_entry::Float64, perp_entry::Float64,
                                funding_rates::Vector{Float64}, size_usd::Float64)
    # Long spot, short perp: collect positive funding when perp > spot
    basis = (perp_entry - spot_entry) / spot_entry
    funding_collected = sum(funding_rates) * size_usd
    basis_pnl = -basis * size_usd  # convergence of basis
    return (basis_initial=basis, funding_collected=funding_collected,
            basis_pnl=basis_pnl, total_pnl=funding_collected + basis_pnl)
end

function funding_rate_regime(funding_rates::Vector{Float64}; window::Int=3*24)
    # Classify funding regime: extreme_positive, positive, neutral, negative, extreme_negative
    recent = funding_rates[max(1,end-window+1):end]
    avg_rate = mean(recent)
    if avg_rate > 0.001; return "extreme_positive"
    elseif avg_rate > 0.0003; return "positive"
    elseif avg_rate > -0.0003; return "neutral"
    elseif avg_rate > -0.001; return "negative"
    else; return "extreme_negative"; end
end

function optimal_funding_entry(funding_rates::Vector{Float64},
                                spot_prices::Vector{Float64},
                                perp_prices::Vector{Float64};
                                min_annualized_yield::Float64=0.10)
    n = min(length(funding_rates), length(spot_prices), length(perp_prices))
    entry_signals = zeros(n)
    for i in 1:n
        annualized = funding_rates[i] * 3 * 365
        basis = (perp_prices[i] - spot_prices[i]) / (spot_prices[i] + 1e-10)
        # Enter when annualized funding exceeds threshold and basis is positive
        if annualized > min_annualized_yield && basis > 0
            entry_signals[i] = annualized
        end
    end
    return entry_signals
end

# ============================================================
# SECTION 4: LIQUIDATION CASCADE ANALYTICS
# ============================================================

struct LiquidationCluster
    price_level::Float64
    estimated_liq_volume::Float64
    direction::Symbol  # :long or :short
    impact_pct::Float64
end

function estimate_liquidation_map(open_interest::Float64,
                                    current_price::Float64,
                                    long_leverage_dist::Vector{Float64},
                                    long_entry_prices::Vector{Float64},
                                    short_leverage_dist::Vector{Float64},
                                    short_entry_prices::Vector{Float64})
    # Estimate liquidation prices for longs and shorts
    long_liq_prices  = long_entry_prices  .* (1.0 .- 1.0 ./ long_leverage_dist)
    short_liq_prices = short_entry_prices .* (1.0 .+ 1.0 ./ short_leverage_dist)
    return (long_liquidations=long_liq_prices, short_liquidations=short_liq_prices)
end

function cascade_scenario(initial_price::Float64, shock_pct::Float64,
                            liq_clusters::Vector{LiquidationCluster},
                            market_depth_usd_per_pct::Float64=1e7)
    price = initial_price
    total_cascaded_volume = 0.0
    max_drawdown_pct = 0.0
    log_entries = Tuple{Float64,Float64,Float64}[]

    for _ in 1:20  # max 20 cascade rounds
        price_new = price * (1 - shock_pct)
        # Find liquidations triggered
        triggered = filter(c -> c.direction == :long &&
                                c.price_level >= price_new &&
                                c.price_level <= price, liq_clusters)
        isempty(triggered) && break
        liq_vol = sum(c.estimated_liq_volume for c in triggered)
        total_cascaded_volume += liq_vol
        # Price impact of liquidations
        additional_drop = liq_vol / (market_depth_usd_per_pct + 1e-10) / 100.0
        shock_pct = additional_drop  # next round
        price = price_new
        push!(log_entries, (price, liq_vol, additional_drop))
        max_drawdown_pct = (initial_price - price) / initial_price
        additional_drop < 0.001 && break  # cascade dies out
    end
    return (final_price=price, max_drawdown_pct=max_drawdown_pct,
            total_cascaded_volume=total_cascaded_volume, log=log_entries)
end

function estimate_open_interest_risk(oi_longs::Float64, oi_shorts::Float64,
                                      current_price::Float64, avg_leverage::Float64)
    long_liq_at_pct_drop  = 100.0 / avg_leverage   # % price drop to trigger avg long liq
    short_liq_at_pct_rise = 100.0 / avg_leverage   # % price rise to trigger avg short liq
    long_liq_value  = oi_longs  * (1.0 / avg_leverage)  # collateral at risk
    short_liq_value = oi_shorts * (1.0 / avg_leverage)
    return (long_liq_price_drop_pct=long_liq_at_pct_drop,
            short_liq_price_rise_pct=short_liq_at_pct_rise,
            long_collateral_at_risk=long_liq_value,
            short_collateral_at_risk=short_liq_value,
            net_directional_risk=(oi_longs-oi_shorts)/(oi_longs+oi_shorts+1e-10))
end

# ============================================================
# SECTION 5: CROSS-EXCHANGE ARBITRAGE
# ============================================================

struct ExchangeQuote
    exchange::String
    bid::Float64
    ask::Float64
    bid_size::Float64
    ask_size::Float64
    latency_ms::Float64
    taker_fee::Float64
end

function find_cross_exchange_arb(quotes::Vector{ExchangeQuote})
    best_bid = maximum(q.bid for q in quotes)
    best_ask = minimum(q.ask for q in quotes)
    spread = best_bid - best_ask
    if spread > 0
        sell_exch = quotes[argmax([q.bid for q in quotes])]
        buy_exch  = quotes[argmin([q.ask for q in quotes])]
        gross_pnl = spread
        total_fee = sell_exch.taker_fee * sell_exch.bid + buy_exch.taker_fee * buy_exch.ask
        net_pnl = gross_pnl - total_fee
        max_size = min(sell_exch.bid_size, buy_exch.ask_size)
        return (profitable=(net_pnl > 0), spread=spread, net_pnl_pct=net_pnl/best_ask,
                sell_exchange=sell_exch.exchange, buy_exchange=buy_exch.exchange,
                max_size=max_size, total_pnl=net_pnl * max_size)
    end
    return (profitable=false, spread=spread, net_pnl_pct=0.0,
            sell_exchange="", buy_exchange="", max_size=0.0, total_pnl=0.0)
end

function triangular_arbitrage(prices::Dict{String,Float64};
                               fee_rate::Float64=0.001)
    # Check BTC/USDT, ETH/USDT, ETH/BTC triangle
    btc_usdt = get(prices, "BTC/USDT", NaN)
    eth_usdt = get(prices, "ETH/USDT", NaN)
    eth_btc  = get(prices, "ETH/BTC",  NaN)
    any(isnan, [btc_usdt, eth_usdt, eth_btc]) && return (profitable=false, gain_pct=0.0)
    # Path: USDT → BTC → ETH → USDT
    start_usdt = 10000.0
    btc_amt  = start_usdt / btc_usdt * (1 - fee_rate)
    eth_amt  = btc_amt    / eth_btc  * (1 - fee_rate)
    usdt_end = eth_amt    * eth_usdt * (1 - fee_rate)
    gain_pct = (usdt_end - start_usdt) / start_usdt
    return (profitable=(gain_pct > 0), gain_pct=gain_pct,
            usdt_end=usdt_end, path="USDT→BTC→ETH→USDT")
end

function statistical_arbitrage_pairs(price_a::Vector{Float64}, price_b::Vector{Float64};
                                      z_entry::Float64=2.0, z_exit::Float64=0.5)
    n = min(length(price_a), length(price_b))
    log_ratio = log.(price_a[1:n] ./ (price_b[1:n] .+ 1e-10))
    mu = mean(log_ratio); sigma = std(log_ratio)
    zscore = (log_ratio .- mu) ./ (sigma .+ 1e-10)
    signals = zeros(n)
    in_trade = 0  # +1 long A/short B, -1 short A/long B
    for i in 1:n
        if in_trade == 0
            if zscore[i] > z_entry;  in_trade = -1; signals[i] = -1.0
            elseif zscore[i] < -z_entry; in_trade = 1; signals[i] = 1.0; end
        else
            if abs(zscore[i]) < z_exit; in_trade = 0; signals[i] = 0.0
            else; signals[i] = Float64(in_trade); end
        end
    end
    return (zscore=zscore, signals=signals, half_life=mu)
end

# ============================================================
# SECTION 6: MARKET MICROSTRUCTURE & TICK DATA
# ============================================================

struct TickData
    timestamps::Vector{Float64}
    prices::Vector{Float64}
    sizes::Vector{Float64}
    sides::Vector{Int8}    # +1 buy, -1 sell
end

function classify_trade_direction_tick_rule(prices::Vector{Float64})
    n = length(prices)
    sides = zeros(Int8, n)
    for i in 2:n
        dp = prices[i] - prices[i-1]
        sides[i] = dp > 0 ? Int8(1) : dp < 0 ? Int8(-1) : sides[i-1]
    end
    return sides
end

function ohlcv_from_ticks(td::TickData, bar_seconds::Float64)
    isempty(td.timestamps) && return (open=Float64[], high=Float64[], low=Float64[],
                                       close=Float64[], volume=Float64[], times=Float64[])
    t_start = td.timestamps[1]
    t_end   = td.timestamps[end]
    n_bars  = max(1, ceil(Int, (t_end - t_start) / bar_seconds))
    opens = Float64[]; highs = Float64[]; lows = Float64[]
    closes = Float64[]; volumes = Float64[]; times = Float64[]
    for b in 0:n_bars-1
        t0 = t_start + b * bar_seconds
        t1 = t0 + bar_seconds
        idx = findall(t -> t0 <= t < t1, td.timestamps)
        isempty(idx) && continue
        bar_prices = td.prices[idx]; bar_vols = td.sizes[idx]
        push!(opens,   bar_prices[1])
        push!(highs,   maximum(bar_prices))
        push!(lows,    minimum(bar_prices))
        push!(closes,  bar_prices[end])
        push!(volumes, sum(bar_vols))
        push!(times,   t0)
    end
    return (open=opens, high=highs, low=lows, close=closes, volume=volumes, times=times)
end

function realized_volatility(prices::Vector{Float64}; annualize::Bool=true,
                               periods_per_year::Int=252)
    returns = diff(log.(prices .+ 1e-10))
    rv = sqrt(sum(returns.^2))
    annualize && (rv *= sqrt(periods_per_year))
    return rv
end

function bipower_variation(prices::Vector{Float64})
    returns = abs.(diff(log.(prices .+ 1e-10)))
    n = length(returns)
    bv = (π/2) * sum(returns[1:end-1] .* returns[2:end]) * n / (n-1)
    return bv
end

function jump_test(prices::Vector{Float64}; alpha::Float64=0.05)
    returns = diff(log.(prices .+ 1e-10))
    n = length(returns)
    rv = sum(returns.^2)
    bv = bipower_variation(prices)
    # Barndorff-Nielsen & Shephard test
    ratio = max(rv - bv, 0.0) / rv
    # Under no jumps, this converges to chi-sq / rv
    z_stat = (rv - bv) / sqrt((π/2)^2 + π - 5) * sqrt(n)
    critical = 1.645  # one-sided 5%
    return (ratio=ratio, z_stat=z_stat, has_jumps=(z_stat > critical))
end

function microstructure_noise_var(prices::Vector{Float64}; m::Int=5)
    # Subsample RV estimator for noise
    rv_full = realized_volatility(prices; annualize=false)^2
    # Subsample at 1/m frequency
    rv_sub = mean([realized_volatility(prices[1:m:end]; annualize=false)^2 for _ in 1:m])
    noise_var = (rv_full - rv_sub) / (length(prices) - 1)
    return max(noise_var, 0.0)
end

# ============================================================
# SECTION 7: DERIVATIVES & OPTIONS ON CRYPTO
# ============================================================

function black_scholes_call(S::Float64, K::Float64, r::Float64,
                              sigma::Float64, T::Float64)
    T <= 0 && return max(S-K, 0.0)
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    N(x) = 0.5*(1+erf(x/sqrt(2)))
    return S*N(d1) - K*exp(-r*T)*N(d2)
end

function black_scholes_put(S::Float64, K::Float64, r::Float64,
                             sigma::Float64, T::Float64)
    T <= 0 && return max(K-S, 0.0)
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    N(x) = 0.5*(1+erf(x/sqrt(2)))
    return K*exp(-r*T)*N(-d2) - S*N(-d1)
end

function bs_greeks(S::Float64, K::Float64, r::Float64, sigma::Float64, T::Float64;
                    option_type::Symbol=:call)
    T <= 1e-8 && return (delta=option_type==:call ? 1.0 : -1.0, gamma=0.0, vega=0.0,
                          theta=0.0, rho=0.0)
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    N(x) = 0.5*(1+erf(x/sqrt(2)))
    n_pdf(x) = exp(-0.5*x^2)/sqrt(2π)
    delta = option_type == :call ? N(d1) : N(d1) - 1.0
    gamma = n_pdf(d1) / (S*sigma*sqrt(T))
    vega  = S * n_pdf(d1) * sqrt(T) / 100.0
    if option_type == :call
        theta = (-S*n_pdf(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2)) / 365.0
        rho   = K*T*exp(-r*T)*N(d2) / 100.0
    else
        theta = (-S*n_pdf(d1)*sigma/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)) / 365.0
        rho   = -K*T*exp(-r*T)*N(-d2) / 100.0
    end
    return (delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
end

function implied_vol_bisection(market_price::Float64, S::Float64, K::Float64,
                                 r::Float64, T::Float64; option_type::Symbol=:call,
                                 tol::Float64=1e-6)
    T <= 0 && return NaN
    lo = 0.001; hi = 10.0
    f = option_type == :call ? black_scholes_call : black_scholes_put
    for _ in 1:200
        mid = (lo + hi) / 2.0
        val = f(S, K, r, mid, T)
        diff = val - market_price
        abs(diff) < tol && return mid
        diff > 0 ? (hi = mid) : (lo = mid)
    end
    return (lo + hi) / 2.0
end

function delta_hedged_pnl(option_prices::Vector{Float64},
                           spot_prices::Vector{Float64},
                           deltas::Vector{Float64},
                           rebalance_freq::Int=1)
    n = length(option_prices)
    pnl_series = zeros(n)
    hedge_position = -deltas[1]
    for i in 2:n
        d_option = option_prices[i] - option_prices[i-1]
        d_spot   = spot_prices[i] - spot_prices[i-1]
        pnl_series[i] = d_option + hedge_position * d_spot
        if i % rebalance_freq == 0
            hedge_position = -deltas[i]
        end
    end
    return (pnl=pnl_series, cum_pnl=cumsum(pnl_series))
end

function vol_surface_interpolation(strikes::Vector{Float64}, expiries::Vector{Float64},
                                    ivs::Matrix{Float64}, K_query::Float64, T_query::Float64)
    # Bilinear interpolation on vol surface
    i_k = searchsortedlast(strikes, K_query)
    i_k = clamp(i_k, 1, length(strikes)-1)
    i_t = searchsortedlast(expiries, T_query)
    i_t = clamp(i_t, 1, length(expiries)-1)
    tk = (K_query - strikes[i_k]) / (strikes[i_k+1] - strikes[i_k] + 1e-10)
    tt = (T_query - expiries[i_t]) / (expiries[i_t+1] - expiries[i_t] + 1e-10)
    return (1-tk)*(1-tt)*ivs[i_k,i_t]   + tk*(1-tt)*ivs[i_k+1,i_t] +
           (1-tk)*tt*ivs[i_k,i_t+1]     + tk*tt*ivs[i_k+1,i_t+1]
end

# ============================================================
# EXTENDED DEMO
# ============================================================

function demo_crypto_mechanics_extended()
    println("=== Crypto Mechanics Extended Demo ===")

    # TWAP schedule
    sched = twap_schedule(10.0, 60.0, 12; randomize=false)
    println("TWAP 1st slice: time=", round(sched[1][1],digits=1),
            " qty=", round(sched[1][2],digits=3))

    # Almgren-Chriss
    ac = almgren_chriss_optimal_trajectory(1000.0, 1.0, 0.02, 0.1, 0.0, 1e-6)
    println("AC expected cost: ", round(ac.expected_cost, sigdigits=4))

    # Funding analysis
    funding_hist = [0.0001*randn()+0.0003 for _ in 1:90]
    fa = analyze_funding_history(funding_hist, Float64.(1:90))
    println("Annualized funding: ", round(fa.annualized_avg*100, digits=2), "%")

    # Cascade scenario
    clusters = [LiquidationCluster(29000.0-i*100, 1e6, :long, 0.005) for i in 0:5]
    cascade = cascade_scenario(30000.0, 0.01, clusters)
    println("Cascade final price: ", round(cascade.final_price, digits=2),
            " drawdown: ", round(cascade.max_drawdown_pct*100, digits=2), "%")

    # Cross-exchange arb
    quotes = [ExchangeQuote("Binance", 29995.0, 30001.0, 1.0, 1.0, 5.0, 0.0004),
              ExchangeQuote("Bybit",   30002.0, 30008.0, 1.0, 1.0, 3.0, 0.0003)]
    arb = find_cross_exchange_arb(quotes)
    println("Arb profitable: ", arb.profitable, " spread: ", round(arb.spread, digits=2))

    # BS Greeks
    g = bs_greeks(30000.0, 30000.0, 0.05, 0.7, 30/365; option_type=:call)
    println("ATM call delta=", round(g.delta,digits=4), " gamma=", round(g.gamma,digits=6))

    # Implied vol
    market_px = black_scholes_call(30000.0, 30000.0, 0.05, 0.7, 30/365)
    iv = implied_vol_bisection(market_px, 30000.0, 30000.0, 0.05, 30/365)
    println("Round-trip IV: ", round(iv, digits=4))
end

end # module CryptoMechanics
