module PortfolioAttribution

# ============================================================
# portfolio_attribution.jl -- Performance Attribution Analysis
# ============================================================
# Covers: Brinson-Hood-Beebower (BHB), Brinson-Fachler,
# factor-based attribution (Barra-style), risk attribution
# (component VaR/CVaR), fixed-income attribution (carry,
# duration, convexity, spread, curve), currency attribution,
# transaction cost attribution, geometric compounding,
# information coefficient calculations.
# Pure stdlib Julia -- no external packages.
# ============================================================

using Statistics, LinearAlgebra

struct PortfolioWeights
    sector_names::Vector{String}
    portfolio_weights::Vector{Float64}
    benchmark_weights::Vector{Float64}
end

struct SectorReturns
    sector_names::Vector{String}
    portfolio_returns::Vector{Float64}
    benchmark_returns::Vector{Float64}
end

struct BHBResult
    sector_names::Vector{String}
    allocation_effect::Vector{Float64}
    selection_effect::Vector{Float64}
    interaction_effect::Vector{Float64}
    total_active_return::Float64
    total_allocation::Float64
    total_selection::Float64
    total_interaction::Float64
end

struct FactorModel
    factor_names::Vector{String}
    factor_exposures_port::Vector{Float64}
    factor_exposures_bench::Vector{Float64}
    factor_returns::Vector{Float64}
    specific_return_port::Float64
    specific_return_bench::Float64
end

struct FIAttribution
    carry::Float64
    duration::Float64
    convexity::Float64
    spread_change::Float64
    residual::Float64
    total::Float64
end

struct RiskAttribution
    factor_names::Vector{String}
    marginal_contributions::Vector{Float64}
    component_var::Vector{Float64}
    pct_contribution::Vector{Float64}
    portfolio_var::Float64
    diversification_ratio::Float64
end

struct TxCostAttribution
    gross_return::Float64
    commission::Float64
    market_impact::Float64
    spread_cost::Float64
    timing_cost::Float64
    net_return::Float64
    implementation_shortfall::Float64
end

# ---- 1. BHB Attribution ----

function bhb_attribution(weights::PortfolioWeights, returns::SectorReturns)::BHBResult
    n = length(weights.sector_names)
    wp = weights.portfolio_weights; wb = weights.benchmark_weights
    rp = returns.portfolio_returns; rb = returns.benchmark_returns
    R_b = dot(wb, rb)
    alloc = (wp .- wb) .* (rb .- R_b)
    sel   = wb .* (rp .- rb)
    inter = (wp .- wb) .* (rp .- rb)
    active = dot(wp,rp) - R_b
    return BHBResult(weights.sector_names, alloc, sel, inter,
                     active, sum(alloc), sum(sel), sum(inter))
end

function brinson_fachler(weights::PortfolioWeights, returns::SectorReturns)::BHBResult
    n = length(weights.sector_names)
    wp = weights.portfolio_weights; wb = weights.benchmark_weights
    rp = returns.portfolio_returns; rb = returns.benchmark_returns
    R_b = dot(wb, rb)
    alloc = (wp .- wb) .* (rb .- R_b)
    sel   = wp .* (rp .- rb)
    active = dot(wp,rp) - R_b
    return BHBResult(weights.sector_names, alloc, sel, zeros(n),
                     active, sum(alloc), sum(sel), 0.0)
end

function attribution_summary(bhb::BHBResult)
    println("\n--- BHB Attribution ---")
    println(rpad("Sector",20), rpad("Alloc%",10), rpad("Select%",10), "Inter%")
    println("-"^55)
    for i in eachindex(bhb.sector_names)
        println(rpad(bhb.sector_names[i],20),
                rpad(string(round(bhb.allocation_effect[i]*100,digits=3))*"%", 10),
                rpad(string(round(bhb.selection_effect[i]*100,digits=3))*"%", 10),
                string(round(bhb.interaction_effect[i]*100,digits=3))*"%")
    end
    println("-"^55)
    println(rpad("TOTAL",20),
            rpad(string(round(bhb.total_allocation*100,digits=3))*"%",10),
            rpad(string(round(bhb.total_selection*100,digits=3))*"%",10),
            string(round(bhb.total_interaction*100,digits=3))*"%")
    println("Active return: ", round(bhb.total_active_return*100,digits=3), "%")
end

# ---- 2. Factor Attribution ----

function factor_attribution(model::FactorModel)
    n = length(model.factor_names)
    active_beta = model.factor_exposures_port .- model.factor_exposures_bench
    contribs = active_beta .* model.factor_returns
    spec_active = model.specific_return_port - model.specific_return_bench
    total = sum(contribs) + spec_active
    pct = abs(total) > 1e-10 ? contribs./abs(total).*100 : zeros(n)
    return (factor_names=model.factor_names, active_exposures=active_beta,
            factor_returns=model.factor_returns, contributions=contribs,
            specific_active=spec_active, total_active=total, pct_contributions=pct)
end

function factor_return_decompose(port_rets::Vector{Float64},
                                   factor_rets::Matrix{Float64},
                                   betas::Vector{Float64})
    explained = factor_rets * betas
    residual  = port_rets .- explained
    r2 = 1 - var(residual)/(var(port_rets)+1e-12)
    te = std(residual)*sqrt(252.0)
    ir = mean(residual)*252 / (te+1e-12)
    return (explained=explained, residual=residual, r_squared=r2,
            tracking_error=te, information_ratio=ir)
end

function style_attribution(port_betas::Vector{Float64}, bench_betas::Vector{Float64},
                             style_rets::Vector{Float64}, names::Vector{String}=String[])
    ab = port_betas .- bench_betas
    contribs = ab .* style_rets
    nms = isempty(names) ? ["Style_$i" for i in eachindex(port_betas)] : names
    return (names=nms, active_betas=ab, contributions=contribs, total=sum(contribs))
end

# ---- 3. Risk Attribution ----

function component_var(weights::Vector{Float64}, cov_mat::Matrix{Float64};
                        names::Vector{String}=String[], conf::Float64=0.95)::RiskAttribution
    n = length(weights)
    nms = isempty(names) ? ["Asset_$i" for i in 1:n] : names
    sig2p = dot(weights, cov_mat*weights)
    sigp  = sqrt(max(sig2p, 0.0))
    z95   = 1.6449
    port_var = z95*sigp
    Cw  = cov_mat*weights
    mrc = Cw ./ (sigp+1e-12)
    cvr = weights .* mrc .* z95
    pct = cvr ./ (port_var+1e-12) .* 100
    indiv_vols = [sqrt(max(cov_mat[i,i],0.0)) for i in 1:n]
    dr = dot(weights, indiv_vols) / (sigp+1e-12)
    return RiskAttribution(nms, mrc, cvr, pct, port_var, dr)
end

function tracking_error_attr(active_w::Vector{Float64}, cov_mat::Matrix{Float64};
                               ann::Float64=252.0)
    te_var = dot(active_w, cov_mat*active_w)
    te_d = sqrt(max(te_var,0.0)); te_a = te_d*sqrt(ann)
    Cw = cov_mat*active_w
    mrc = active_w .* Cw ./ (te_d+1e-12)
    pct = mrc ./ (te_d+1e-12) .* 100
    return (te_daily=te_d, te_annual=te_a, component_te=mrc.*sqrt(ann), pct=pct)
end

function expected_shortfall_attr(returns::Matrix{Float64}, weights::Vector{Float64},
                                   alpha::Float64=0.05)
    T_len, n = size(returns)
    pr = returns*weights
    cutoff = sort(pr)[max(1,round(Int,alpha*T_len))]
    idx = findall(r -> r <= cutoff, pr)
    isempty(idx) && return (es=NaN, component_es=zeros(n), pct=zeros(n))
    es = -mean(pr[idx])
    comp_es = -mean(returns[idx,:], dims=1)[:].*weights
    return (es=es, component_es=comp_es, pct=comp_es./(es+1e-12).*100)
end

function maximum_drawdown_contribution(port_rets::Vector{Float64},
                                        asset_rets::Matrix{Float64},
                                        weights::Vector{Float64})
    T_len, n = size(asset_rets)
    cum = cumsum(port_rets); peak = -Inf; dd_start = 1; dd_end = 1; max_dd = 0.0
    for (i, c) in enumerate(cum)
        if c > peak; peak = c; dd_start = i; end
        if peak - c > max_dd; max_dd = peak - c; dd_end = i; end
    end
    range_rets = asset_rets[dd_start:dd_end, :]
    contribs = mean(range_rets, dims=1)[:] .* weights
    return (max_drawdown=max_dd, dd_contributions=contribs,
            dd_start=dd_start, dd_end=dd_end)
end

# ---- 4. Fixed Income Attribution ----

function fi_attribution(yield_start::Float64, yield_end::Float64,
                          duration::Float64, convexity::Float64,
                          spread_start::Float64, spread_end::Float64,
                          carry_days::Float64, coupon::Float64)::FIAttribution
    dy = yield_end - yield_start; ds = spread_end - spread_start
    rate_move = dy - ds
    carry  = coupon * carry_days/365.0
    dur_r  = -duration * rate_move
    conv_r = 0.5 * convexity * rate_move^2
    spr_r  = -duration * ds
    total  = carry + dur_r + conv_r + spr_r
    return FIAttribution(carry, dur_r, conv_r, spr_r, 0.0, total)
end

function key_rate_attribution(krd::Vector{Float64}, kr_changes::Vector{Float64})
    contribs = -krd .* kr_changes; total = sum(contribs)
    avg_chg = mean(kr_changes); parallel = -sum(krd)*avg_chg
    twist   = kr_changes[end] - kr_changes[1]
    return (total=total, contributions=contribs, parallel=parallel, twist=twist)
end

function carry_rolldown(yield::Float64, fwd_yield::Float64, duration::Float64,
                         holding_days::Float64)::Float64
    carry_component = yield * holding_days / 365.0
    rolldown = -duration * (fwd_yield - yield) * holding_days / 365.0
    return carry_component + rolldown
end

# ---- 5. Currency Attribution ----

function currency_attribution(local_w::Vector{Float64}, local_rets::Vector{Float64},
                                spot_chg::Vector{Float64}, hedge_ratios::Vector{Float64})
    total_rets = local_rets .+ spot_chg
    hedged = local_rets .+ spot_chg.*(1 .- hedge_ratios)
    local_eff    = local_rets .* local_w
    currency_eff = spot_chg  .* local_w
    hedge_eff    = -spot_chg .* local_w .* hedge_ratios
    return (total=dot(local_w,total_rets), local_return=sum(local_eff),
            currency_return=sum(currency_eff), hedge_return=sum(hedge_eff),
            per_ccy_local=local_eff, per_ccy_fx=currency_eff)
end

# ---- 6. Transaction Costs ----

function tx_cost_attribution(gross_ret::Float64, trade_sizes::Vector{Float64},
                               arrival_prices::Vector{Float64},
                               exec_prices::Vector{Float64};
                               commission_rate::Float64=0.0005,
                               spread_rate::Float64=0.0003,
                               impact_coef::Float64=0.1)::TxCostAttribution
    notional = sum(abs.(trade_sizes).*arrival_prices)
    commission   = notional*commission_rate
    spread_cost  = notional*spread_rate/2
    impacts = impact_coef.*abs.(trade_sizes)./arrival_prices
    mkt_impact = sum(abs.(trade_sizes).*arrival_prices.*impacts)
    timing_cost = sum(abs.(trade_sizes).*(exec_prices.-arrival_prices))
    net = gross_ret - (commission+spread_cost+mkt_impact+timing_cost)/(notional+1e-8)
    is  = (commission+spread_cost+sum(abs.(trade_sizes).*(exec_prices.-arrival_prices)))/(notional+1e-8)
    return TxCostAttribution(gross_ret, commission, mkt_impact, spread_cost, timing_cost, net, is)
end

# ---- 7. Portfolio Characteristics ----

function portfolio_characteristics(weights::Vector{Float64},
                                    returns::Matrix{Float64},
                                    bench_w::Vector{Float64},
                                    bench_rets::Matrix{Float64};
                                    rf::Float64=0.02/252)
    pr = returns*weights; br = bench_rets*bench_w; ar = pr.-br
    ann = 252.0
    ann_p = mean(pr)*ann; ann_b = mean(br)*ann; ann_a = mean(ar)*ann
    te = std(ar)*sqrt(ann); ir = ann_a/(te+1e-12)
    sharpe = (mean(pr)-rf)*ann/(std(pr)*sqrt(ann)+1e-12)
    cum = cumsum(pr); peak = -Inf; max_dd = 0.0
    for c in cum; if c>peak; peak=c; end; max_dd=max(max_dd,peak-c); end
    return (ann_return=ann_p, ann_bench=ann_b, ann_active=ann_a,
            te=te, ir=ir, sharpe=sharpe, max_dd=max_dd,
            hit_rate=count(r->r>0,ar)/length(ar))
end

function information_coefficient(scores::Vector{Float64}, fwd_rets::Vector{Float64})::Float64
    n = length(scores)
    rs = sortperm(sortperm(scores)); rr = sortperm(sortperm(fwd_rets))
    rf = Float64.(rs); rrr = Float64.(rr)
    rfb = mean(rf); rrb = mean(rrr)
    cov_val = sum((rf.-rfb).*(rrr.-rrb))/(n-1)
    return cov_val/(std(rf)*std(rrr)+1e-12)
end

function geometric_link(monthly_effects::Matrix{Float64})::Vector{Float64}
    n_months, n_eff = size(monthly_effects)
    return [prod(1.0 .+ monthly_effects[:,j]) - 1.0 for j in 1:n_eff]
end

function annualise_attribution(monthly_effects::Matrix{Float64})::Vector{Float64}
    return geometric_link(monthly_effects)
end

# ---- Demo ----

function demo()
    println("=== PortfolioAttribution Demo ===")
    sectors = ["Technology","Healthcare","Financials","Energy","Consumer"]
    wp = [0.35, 0.20, 0.25, 0.10, 0.10]
    wb = [0.25, 0.20, 0.20, 0.20, 0.15]
    rp = [0.08, 0.05, 0.03,-0.02, 0.04]
    rb = [0.06, 0.04, 0.04, 0.01, 0.03]
    wts = PortfolioWeights(sectors, wp, wb)
    rets = SectorReturns(sectors, rp, rb)
    bhb = bhb_attribution(wts, rets)
    attribution_summary(bhb)

    fm = FactorModel(["Market","Value","Momentum","Size","Quality"],
                     [1.05, 0.3, 0.2,-0.1, 0.4],[1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.06, 0.02, 0.03,-0.01, 0.01], 0.008, 0.001)
    fa = factor_attribution(fm)
    println("\n--- Factor Attribution ---")
    for i in eachindex(fa.factor_names)
        println("  ", rpad(fa.factor_names[i],12), " actB=",
                round(fa.active_exposures[i],digits=3), " contrib=",
                round(fa.contributions[i]*100,digits=3), "%")
    end
    println("  Specific: ", round(fa.specific_active*100,digits=3), "%")

    println("\n--- FI Attribution ---")
    fi = fi_attribution(0.035,0.032,6.5,55.0,0.01,0.008,90.0,0.04)
    println("  Carry: ", round(fi.carry*100,digits=4), "%")
    println("  Duration: ", round(fi.duration*100,digits=4), "%")
    println("  Spread: ", round(fi.spread_change*100,digits=4), "%")
    println("  Total: ", round(fi.total*100,digits=4), "%")

    println("\n--- Risk Attribution ---")
    cm = [0.04 0.01 0.005; 0.01 0.03 0.008; 0.005 0.008 0.025]
    ww = [0.5, 0.3, 0.2]
    ra = component_var(ww, cm; names=["Eq","FI","Alt"])
    println("  Portfolio VaR 95%: ", round(ra.portfolio_var*100,digits=3), "%")
    println("  Diversification ratio: ", round(ra.diversification_ratio,digits=4))
    for i in 1:3
        println("  ", ra.factor_names[i], ": ", round(ra.pct_contribution[i],digits=2), "%")
    end
end


# ============================================================
# EXTENDED PORTFOLIO ATTRIBUTION METHODS
# ============================================================

# ============================================================
# MULTI-PERIOD GEOMETRIC ATTRIBUTION
# ============================================================

"""
GRAP (Global Research Associates Presentation) geometric linking.
Links arithmetic period attributions into compound performance attribution.
"""
function grap_linking(period_alloc::Vector{Float64}, period_sel::Vector{Float64},
                       period_port_ret::Vector{Float64}, period_bench_ret::Vector{Float64})
    T = length(period_alloc)

    # Compound returns
    R_p = prod(1 .+ period_port_ret) - 1
    R_b = prod(1 .+ period_bench_ret) - 1

    # GRAP linking coefficients
    bench_cum = [prod(1 .+ period_bench_ret[t+1:end]) for t in 1:T]

    bench_total = prod(1 .+ period_bench_ret)

    linked_alloc = sum(period_alloc[t] * bench_cum[t] / max(bench_total, 1e-10) for t in 1:T)
    linked_alloc *= (bench_total - 1)

    linked_sel = sum(period_sel[t] * bench_cum[t] / max(bench_total, 1e-10) for t in 1:T)
    linked_sel *= (bench_total - 1)

    return (portfolio_return=R_p, benchmark_return=R_b, active_return=R_p-R_b,
            linked_allocation=linked_alloc, linked_selection=linked_sel,
            residual=(R_p-R_b) - linked_alloc - linked_sel)
end

"""
Carino logarithmic linking (alternative to GRAP).
Weights each period's contribution by the log of cumulative wealth.
"""
function carino_linking(period_effects::Vector{Float64},
                         period_port_ret::Vector{Float64},
                         period_bench_ret::Vector{Float64})
    T = length(period_effects)
    R_p = prod(1 .+ period_port_ret) - 1
    R_b = prod(1 .+ period_bench_ret) - 1

    # Carino k-factor
    k_T = R_p > -1 && R_b > -1 ?
          (log(1+R_p) - log(1+R_b)) / max(R_p - R_b, 1e-12) : 1.0

    k_t = [begin
        Rp_t, Rb_t = period_port_ret[t], period_bench_ret[t]
        Rp_t > -1 && Rb_t > -1 ?
            (log(1+Rp_t) - log(1+Rb_t)) / max(Rp_t - Rb_t, 1e-12) : 1.0
    end for t in 1:T]

    linked = sum(period_effects[t] * k_t[t] / max(k_T, 1e-10) for t in 1:T)
    return linked
end

# ============================================================
# CURRENCY ATTRIBUTION
# ============================================================

"""
Currency contribution to active return.
Active currency return = w_p * (s_p + currency_ret) - w_b * (s_b + currency_ret)
= (w_p - w_b) * currency_ret + active_local_return
"""
function currency_attribution(
    w_p::Vector{Float64},          # portfolio weights
    w_b::Vector{Float64},          # benchmark weights
    local_ret_p::Vector{Float64},  # local currency returns (portfolio)
    local_ret_b::Vector{Float64},  # local currency returns (benchmark)
    fx_ret::Vector{Float64};       # FX return of each currency vs base
    currency_names::Vector{String}=["Currency_$i" for i in 1:length(w_p)]
)
    N = length(w_p)

    # Total returns in base currency
    total_ret_p = local_ret_p + fx_ret
    total_ret_b = local_ret_b + fx_ret

    R_p_base = dot(w_p, total_ret_p)
    R_b_base = dot(w_b, total_ret_b)

    # Currency allocation: (w_p - w_b) * fx_ret
    currency_alloc = (w_p - w_b) .* fx_ret

    # Local return allocation: standard BHB on local returns
    R_b_local = dot(w_b, local_ret_b)
    local_alloc = (w_p - w_b) .* (local_ret_b .- R_b_local)
    local_sel = w_b .* (local_ret_p - local_ret_b)
    local_inter = (w_p - w_b) .* (local_ret_p - local_ret_b)

    return (
        active_total=R_p_base - R_b_base,
        currency_allocation=currency_alloc,
        local_allocation=local_alloc,
        local_selection=local_sel,
        local_interaction=local_inter,
        total_currency=sum(currency_alloc),
        total_local=sum(local_alloc) + sum(local_sel) + sum(local_inter),
        currency_names=currency_names
    )
end

# ============================================================
# SECTOR ROTATION ATTRIBUTION
# ============================================================

"""
Sector rotation analysis: how timing of sector tilts contributed to returns.
"""
function sector_rotation_analysis(
    sector_weights_history::Matrix{Float64},    # T x N: portfolio weights over time
    benchmark_weights_history::Matrix{Float64}, # T x N: benchmark weights
    sector_returns_history::Matrix{Float64}     # T x N: sector returns
)
    T, N = size(sector_weights_history)
    active_weights = sector_weights_history - benchmark_weights_history

    # Timing: correlation between active weight changes and next period returns
    timing_scores = zeros(N)
    for s in 1:N
        if T > 2
            dw = diff(active_weights[:, s])   # weight changes
            fwd_ret = sector_returns_history[2:T, s]
            len = min(length(dw), length(fwd_ret))
            if len >= 3
                timing_scores[s] = cor(dw[1:len], fwd_ret[1:len])
            end
        end
    end

    # Hit rate: fraction of periods where active weight and relative return have same sign
    hit_rates = zeros(N)
    for s in 1:N
        bench_r = sum(benchmark_weights_history[:, j] .* sector_returns_history[:, j] for j in 1:N) / N
        relative_ret = sector_returns_history[:, s] - vec(mean(sector_returns_history, dims=2))
        same_sign = sign.(active_weights[:, s]) .== sign.(relative_ret)
        hit_rates[s] = mean(same_sign)
    end

    return (timing_scores=timing_scores, hit_rates=hit_rates,
            avg_timing_score=mean(timing_scores),
            avg_hit_rate=mean(hit_rates))
end

# ============================================================
# ALPHA DECOMPOSITION
# ============================================================

"""
Decompose total portfolio return into:
1. Market beta return
2. Factor tilts (size, value, momentum, quality)
3. Active selection (residual)
"""
function decompose_alpha(portfolio_returns::Vector{Float64},
                          factor_returns::Matrix{Float64},
                          factor_names::Vector{String};
                          risk_free::Float64=0.0)
    T, K = size(factor_returns)
    excess = portfolio_returns .- risk_free

    X = hcat(ones(T), factor_returns)
    beta = (X'*X + 1e-8*I) \ (X'*excess)

    alpha = beta[1]
    factor_betas = beta[2:end]

    y_hat = X * beta
    residuals = excess - y_hat
    residual_var = var(residuals)
    factor_var = var(y_hat .- mean(y_hat))

    # Information ratio components
    ir_alpha = alpha * 252 / (std(residuals) * sqrt(252))
    ir_factor = sum(factor_betas .* vec(mean(factor_returns, dims=1))) * 252 /
                    (sqrt(factor_var) * sqrt(252))

    # Factor contribution to return
    factor_contributions = factor_betas .* vec(mean(factor_returns, dims=1)) .* T

    return (alpha_annualized=alpha*252, factor_betas=factor_betas,
            factor_contributions=factor_contributions,
            ir_alpha=ir_alpha, r_squared=1 - residual_var/max(var(excess), 1e-10),
            factor_names=factor_names, residuals=residuals)
end

# ============================================================
# TRANSACTION COST ANALYSIS (TCA)
# ============================================================

"""
Full TCA: implementation shortfall decomposition.
IS = Decision-to-execution slippage
   = Delay cost + Market impact + Spread cost + Opportunity cost
"""
struct TCAResult
    trade_id::String
    side::Symbol
    decision_price::Float64
    arrival_price::Float64
    avg_execution_price::Float64
    vwap_benchmark::Float64
    close_price::Float64
    # Components
    delay_cost_bps::Float64
    market_impact_bps::Float64
    spread_cost_bps::Float64
    timing_cost_bps::Float64
    opportunity_cost_bps::Float64
    # Aggregates
    is_bps::Float64              # total implementation shortfall
    vwap_slippage_bps::Float64
end

function compute_tca(trade_id::String, side::Symbol,
                      decision_price::Float64, arrival_price::Float64,
                      exec_prices::Vector{Float64}, exec_sizes::Vector{Float64},
                      vwap_price::Float64, close_price::Float64)
    sgn = side == :buy ? 1.0 : -1.0
    avg_exec = dot(exec_prices, exec_sizes) / max(sum(exec_sizes), 1e-10)

    to_bps(x) = x * 10000.0

    delay_cost = sgn * (arrival_price - decision_price) / max(decision_price, 1e-10)
    market_impact = sgn * (avg_exec - arrival_price) / max(arrival_price, 1e-10)
    spread_cost = 0.0005  # 5bps half-spread estimate
    timing_cost = sgn * (close_price - avg_exec) / max(avg_exec, 1e-10)
    opportunity_cost = 0.0  # unfilled portion (simplified)
    is = sgn * (avg_exec - decision_price) / max(decision_price, 1e-10)
    vwap_slip = sgn * (avg_exec - vwap_price) / max(vwap_price, 1e-10)

    return TCAResult(trade_id, side, decision_price, arrival_price, avg_exec,
                      vwap_price, close_price,
                      to_bps(delay_cost), to_bps(market_impact),
                      to_bps(spread_cost), to_bps(timing_cost),
                      to_bps(opportunity_cost),
                      to_bps(is), to_bps(vwap_slip))
end

"""Aggregate TCA across multiple trades."""
function aggregate_tca(trades::Vector{TCAResult})
    n = length(trades); isempty(trades) && return nothing

    return (
        n_trades=n,
        avg_is_bps=mean(t.is_bps for t in trades),
        avg_market_impact_bps=mean(t.market_impact_bps for t in trades),
        avg_vwap_slippage_bps=mean(t.vwap_slippage_bps for t in trades),
        avg_delay_cost_bps=mean(t.delay_cost_bps for t in trades),
        by_side=Dict(
            :buy => mean(t.is_bps for t in trades if t.side == :buy),
            :sell => mean(t.is_bps for t in trades if t.side == :sell)
        )
    )
end

# ============================================================
# FACTOR TIMING ATTRIBUTION
# ============================================================

"""
Factor timing: did we vary factor exposures profitably over time?
Compare dynamic vs static factor exposures.
"""
function factor_timing_attribution(
    dynamic_betas::Matrix{Float64},    # T x K: time-varying exposures
    factor_returns::Matrix{Float64},   # T x K
    static_betas::Vector{Float64}      # K: long-run average exposures
)
    T, K = size(factor_returns)

    # Dynamic strategy return
    dynamic_ret = [dot(dynamic_betas[t, :], factor_returns[t, :]) for t in 1:T]

    # Static strategy return (using long-run betas)
    static_ret = factor_returns * static_betas

    # Timing contribution: dynamic - static
    timing_contribution = dynamic_ret - static_ret

    # Factor-by-factor timing
    factor_timing = [(dynamic_betas[:, k] .- static_betas[k]) .* factor_returns[:, k]
                     for k in 1:K]

    return (timing_return=sum(timing_contribution),
            timing_sharpe=mean(timing_contribution) / max(std(timing_contribution), 1e-10) * sqrt(252),
            factor_timing_contributions=[mean(ft) for ft in factor_timing],
            dynamic_return=sum(dynamic_ret),
            static_return=sum(static_ret))
end

# ============================================================
# PORTFOLIO STRESS ATTRIBUTION
# ============================================================

"""
Stress test attribution: where does portfolio lose most in stress scenarios?
"""
function stress_attribution(
    weights::Vector{Float64},
    factor_loadings::Matrix{Float64},  # N x K
    factor_shocks::Vector{Float64};    # K stress shocks
    asset_names::Vector{String}=["Asset_$i" for i in 1:length(weights)]
)
    N, K = size(factor_loadings)

    # Asset-level stress returns
    asset_stress_rets = factor_loadings * factor_shocks

    # Portfolio stress return
    portfolio_stress_ret = dot(weights, asset_stress_rets)

    # Attribution: contribution of each asset to portfolio stress
    asset_contributions = weights .* asset_stress_rets
    pct_contribution = asset_contributions ./ max(abs(portfolio_stress_ret), 1e-10)

    # Factor contributions to portfolio stress
    factor_contributions = [dot(weights, factor_loadings[:, k]) * factor_shocks[k] for k in 1:K]

    return (portfolio_stress_return=portfolio_stress_ret,
            asset_contributions=asset_contributions,
            pct_by_asset=pct_contribution,
            factor_contributions=factor_contributions,
            worst_contributor=asset_names[argmin(asset_contributions)],
            best_contributor=asset_names[argmax(asset_contributions)])
end

# ============================================================
# ATTRIBUTION RECONCILIATION
# ============================================================

"""
Reconcile attribution: ensure all effects sum to active return.
"""
function reconcile_attribution(
    active_return::Float64,
    allocation::Float64,
    selection::Float64,
    interaction::Float64,
    currency::Float64=0.0,
    other::Float64=0.0;
    tol::Float64=1e-6
)
    total_attributed = allocation + selection + interaction + currency + other
    residual = active_return - total_attributed

    reconciled = abs(residual) < tol

    return (reconciled=reconciled, residual=residual,
            total_attributed=total_attributed, active_return=active_return,
            allocation=allocation, selection=selection,
            interaction=interaction, currency=currency, other=other)
end

# ============================================================
# PERFORMANCE REPORTING
# ============================================================

"""
Generate comprehensive performance attribution report.
"""
function attribution_report(
    portfolio_return::Float64,
    benchmark_return::Float64,
    sector_names::Vector{String},
    allocation::Vector{Float64},
    selection::Vector{Float64},
    interaction::Vector{Float64};
    title::String="Performance Attribution Report"
)
    active = portfolio_return - benchmark_return

    println("\n" * "="^60)
    println(" " * title)
    println("="^60)
    println("Portfolio Return:    $(round(portfolio_return*100, digits=3))%")
    println("Benchmark Return:    $(round(benchmark_return*100, digits=3))%")
    println("Active Return:       $(round(active*100, digits=3))%")
    println("")
    println("="^60)
    println(" Attribution Decomposition")
    println("="^60)
    println(rpad("Sector", 16), rpad("Alloc", 10), rpad("Select", 10), rpad("Interact", 10), "Total")
    println("-"^56)

    for (i, name) in enumerate(sector_names)
        total = allocation[i] + selection[i] + interaction[i]
        println(rpad(name, 16),
                rpad("$(round(allocation[i]*100, digits=3))%", 10),
                rpad("$(round(selection[i]*100, digits=3))%", 10),
                rpad("$(round(interaction[i]*100, digits=3))%", 10),
                "$(round(total*100, digits=3))%")
    end

    println("-"^56)
    tot_alloc = sum(allocation); tot_sel = sum(selection); tot_inter = sum(interaction)
    total_attr = tot_alloc + tot_sel + tot_inter
    println(rpad("TOTAL", 16),
            rpad("$(round(tot_alloc*100, digits=3))%", 10),
            rpad("$(round(tot_sel*100, digits=3))%", 10),
            rpad("$(round(tot_inter*100, digits=3))%", 10),
            "$(round(total_attr*100, digits=3))%")

    reconcile = reconcile_attribution(active, tot_alloc, tot_sel, tot_inter)
    if !reconcile.reconciled
        println("\nWarning: Residual = $(round(reconcile.residual*100, digits=5))%")
    else
        println("\nAttribution fully reconciled ✓")
    end

    println("="^60)
end

# ============================================================
# DEMO FOR EXTENDED ATTRIBUTION
# ============================================================

"""Extended demo: multi-period attribution and TCA."""
function demo_extended_attribution()
    println("=== Extended Attribution Demo ===")

    # Multi-period geometric linking
    T = 4
    alloc = [0.002, -0.001, 0.003, 0.001]
    sel = [0.001, 0.002, -0.001, 0.003]
    p_ret = [0.015, -0.005, 0.020, 0.012]
    b_ret = [0.010, -0.002, 0.015, 0.008]

    grap = grap_linking(alloc, sel, p_ret, b_ret)
    println("GRAP linked active return: $(round(grap.active_return*100, digits=4))%")
    println("Linked allocation: $(round(grap.linked_allocation*100, digits=4))%")

    # TCA analysis
    trades = [
        compute_tca("T001", :buy, 50000.0, 50100.0, [50150.0, 50200.0], [0.5, 0.5],
                     50300.0, 50400.0),
        compute_tca("T002", :sell, 50400.0, 50350.0, [50300.0, 50280.0], [0.6, 0.4],
                     50200.0, 50100.0)
    ]

    agg = aggregate_tca(trades)
    println("\nTCA Summary: Avg IS=$(round(agg.avg_is_bps, digits=2))bps, Impact=$(round(agg.avg_market_impact_bps, digits=2))bps")

    # Currency attribution
    w_p = [0.4, 0.35, 0.25]
    w_b = [0.35, 0.40, 0.25]
    local_r_p = [0.02, 0.015, 0.025]
    local_r_b = [0.018, 0.012, 0.022]
    fx_r = [0.005, -0.003, 0.001]

    curr = currency_attribution(w_p, w_b, local_r_p, local_r_b, fx_r,
                                  currency_names=["USD", "EUR", "GBP"])
    println("\nCurrency attribution total: $(round(curr.total_currency*100, digits=4))%")

    # Attribution report
    sectors = ["Technology", "Healthcare", "Financials", "Energy"]
    alloc_v = [0.0025, -0.0010, 0.0015, -0.0005]
    sel_v = [0.0015, 0.0020, -0.0005, 0.0010]
    inter_v = [0.0005, -0.0002, 0.0003, -0.0001]

    attribution_report(0.0350, 0.0230, sectors, alloc_v, sel_v, inter_v)
end


# ============================================================
# SECTION 2: MULTI-PERIOD & FACTOR ATTRIBUTION
# ============================================================

struct FactorAttribution
    factor_names::Vector{String}
    factor_returns::Vector{Float64}
    factor_exposures::Vector{Float64}
    factor_contributions::Vector{Float64}
    specific_return::Float64
    total_active::Float64
end

function factor_attribution(portfolio_return::Float64,
                              benchmark_return::Float64,
                              factor_returns::Vector{Float64},
                              portfolio_betas::Vector{Float64},
                              benchmark_betas::Vector{Float64},
                              factor_names::Vector{String})
    K = length(factor_names)
    active = portfolio_return - benchmark_return
    active_betas = portfolio_betas .- benchmark_betas
    factor_contributions = active_betas .* factor_returns
    specific = active - sum(factor_contributions)
    return FactorAttribution(factor_names, factor_returns, active_betas,
                              factor_contributions, specific, active)
end

function fama_french_3factor(portfolio_returns::Vector{Float64},
                               mkt_rf::Vector{Float64}, smb::Vector{Float64},
                               hml::Vector{Float64}, rf::Vector{Float64})
    T = length(portfolio_returns)
    excess_ret = portfolio_returns .- rf
    X = hcat(ones(T), mkt_rf, smb, hml)
    beta = (X'*X + 1e-6*I(4)) \ (X'*excess_ret)
    fitted = X*beta; residuals = excess_ret .- fitted
    r2 = 1.0 - var(residuals)/(var(excess_ret)+1e-10)
    alpha_t = beta[1] / (std(residuals)/sqrt(T) + 1e-10)
    return (alpha=beta[1], beta_mkt=beta[2], beta_smb=beta[3], beta_hml=beta[4],
            r2=r2, alpha_tstat=alpha_t, residuals=residuals)
end

function carhart_4factor(portfolio_returns::Vector{Float64},
                          mkt_rf::Vector{Float64}, smb::Vector{Float64},
                          hml::Vector{Float64}, mom::Vector{Float64},
                          rf::Vector{Float64})
    T = length(portfolio_returns)
    excess_ret = portfolio_returns .- rf
    X = hcat(ones(T), mkt_rf, smb, hml, mom)
    beta = (X'*X + 1e-6*I(5)) \ (X'*excess_ret)
    fitted = X*beta; residuals = excess_ret .- fitted
    r2 = 1.0 - var(residuals)/(var(excess_ret)+1e-10)
    return (alpha=beta[1], betas=beta[2:5], r2=r2, residuals=residuals,
            factor_names=["MKT-RF","SMB","HML","MOM"])
end

function rolling_factor_attribution(portfolio_returns::Vector{Float64},
                                     factor_matrix::Matrix{Float64};
                                     window::Int=60)
    T = length(portfolio_returns); K = size(factor_matrix, 2)
    rolling_alpha = zeros(T); rolling_betas = zeros(T, K)
    rolling_r2 = zeros(T)
    for t in window:T
        y = portfolio_returns[t-window+1:t]
        X = hcat(ones(window), factor_matrix[t-window+1:t,:])
        beta = (X'*X + 1e-6*I(K+1)) \ (X'*y)
        fitted = X*beta; res = y .- fitted
        rolling_alpha[t] = beta[1]
        rolling_betas[t,:] = beta[2:end]
        rolling_r2[t] = 1.0 - var(res)/(var(y)+1e-10)
    end
    return (alpha=rolling_alpha, betas=rolling_betas, r2=rolling_r2)
end

function style_attribution(portfolio_returns::Vector{Float64},
                             style_indices::Matrix{Float64},
                             style_names::Vector{String})
    # Sharpe style analysis via constrained regression (weights >= 0, sum = 1)
    T, K = size(style_indices)
    # Use simple non-negative least squares via projected gradient
    w = ones(K) ./ K
    X = style_indices; y = portfolio_returns
    for iter in 1:1000
        grad = -2 * X' * (y - X*w)
        w_new = w .- 0.001 .* grad
        w_new = max.(w_new, 0.0); w_new ./= (sum(w_new) + 1e-10)
        norm(w_new - w) < 1e-8 && break
        w = w_new
    end
    fitted = X*w; res = y .- fitted
    r2 = 1.0 - var(res)/(var(y)+1e-10)
    selection_return = mean(res)
    return (weights=w, r2=r2, selection_return=selection_return,
            style_names=style_names, fitted=fitted)
end

# ============================================================
# SECTION 3: RISK ATTRIBUTION
# ============================================================

struct RiskBudget
    assets::Vector{String}
    weights::Vector{Float64}
    marginal_risk_contrib::Vector{Float64}
    risk_contrib::Vector{Float64}
    risk_contrib_pct::Vector{Float64}
    portfolio_vol::Float64
end

function risk_budgeting(weights::Vector{Float64}, cov_matrix::Matrix{Float64},
                         asset_names::Vector{String})
    n = length(weights)
    port_var = dot(weights, cov_matrix * weights)
    port_vol = sqrt(max(port_var, 0.0))
    marginal = cov_matrix * weights ./ (port_vol + 1e-10)
    contrib  = weights .* marginal
    contrib_pct = contrib ./ (port_vol + 1e-10) .* 100
    return RiskBudget(asset_names, weights, marginal, contrib, contrib_pct, port_vol)
end

function equal_risk_contribution_weights(cov_matrix::Matrix{Float64};
                                           max_iter::Int=1000, tol::Float64=1e-8)
    n = size(cov_matrix, 1)
    w = ones(n) ./ n
    for _ in 1:max_iter
        port_vol = sqrt(dot(w, cov_matrix*w))
        marginal = cov_matrix*w ./ (port_vol+1e-10)
        contrib  = w .* marginal
        target   = port_vol / n
        # Update: increase underweight risk contributors, decrease overweight
        w_new = w .* (target ./ (contrib .+ 1e-10)).^0.3
        w_new ./= sum(w_new)
        norm(w_new - w) < tol && (w = w_new; break)
        w = w_new
    end
    return w
end

function component_var(weights::Vector{Float64}, returns_matrix::Matrix{Float64};
                        confidence::Float64=0.95)
    n = length(weights)
    port_returns = returns_matrix * weights
    var_level = quantile(port_returns, 1-confidence)
    cvar = mean([r for r in port_returns if r <= var_level])
    # Component VaR via marginal contribution
    comp_var = zeros(n)
    for i in 1:n
        cov_i = cov(returns_matrix[:,i], port_returns)
        comp_var[i] = weights[i] * cov_i / (std(port_returns)+1e-10) *
                      (-var_level) / (std(port_returns)+1e-10)
    end
    return (portfolio_var=-var_level, cvar=-cvar, component_var=comp_var)
end

function tracking_error_decomposition(active_weights::Vector{Float64},
                                        cov_matrix::Matrix{Float64},
                                        asset_names::Vector{String})
    TE = sqrt(max(dot(active_weights, cov_matrix*active_weights), 0.0))
    marginal_te = cov_matrix * active_weights ./ (TE + 1e-10)
    contrib_te  = active_weights .* marginal_te
    contrib_pct = contrib_te ./ (TE + 1e-10) .* 100
    return (tracking_error=TE, marginal_te=marginal_te,
            contribution_te=contrib_te, contribution_pct=contrib_pct,
            asset_names=asset_names)
end

function expected_shortfall_attribution(weights::Vector{Float64},
                                          returns_matrix::Matrix{Float64};
                                          alpha::Float64=0.05)
    port_returns = returns_matrix * weights
    T = length(port_returns)
    threshold = quantile(port_returns, alpha)
    tail_idx = findall(r -> r <= threshold, port_returns)
    isempty(tail_idx) && return zeros(length(weights))
    tail_returns = returns_matrix[tail_idx,:]
    # ES contribution: weight * mean(tail return_i)
    es_contribs = weights .* [mean(tail_returns[:,i]) for i in 1:length(weights)]
    return (es_contributions=es_contribs, portfolio_es=-mean(port_returns[tail_idx]))
end

function maximum_drawdown_contribution(weights::Vector{Float64},
                                         returns_matrix::Matrix{Float64})
    T, N = size(returns_matrix)
    port_ret = returns_matrix * weights
    cum_port = cumsum(port_ret)
    peak = cum_port[1]; mdd = 0.0; mdd_start = 1; mdd_end = 1
    peak_idx = 1
    for t in 1:T
        if cum_port[t] > peak; peak = cum_port[t]; peak_idx = t; end
        dd = peak - cum_port[t]
        if dd > mdd; mdd = dd; mdd_start = peak_idx; mdd_end = t; end
    end
    # Attribution during drawdown period
    dd_returns = returns_matrix[mdd_start:mdd_end,:]
    contrib = weights .* [sum(dd_returns[:,i]) for i in 1:N]
    return (mdd=mdd, mdd_start=mdd_start, mdd_end=mdd_end,
            asset_contributions=contrib)
end

# ============================================================
# SECTION 4: FIXED INCOME ATTRIBUTION
# ============================================================

struct FixedIncomeAttribution
    carry_return::Float64
    duration_return::Float64
    convexity_return::Float64
    spread_return::Float64
    currency_return::Float64
    residual::Float64
    total::Float64
end

function fixed_income_attribution_detailed(ytm_start::Float64, ytm_end::Float64,
                                            mod_duration::Float64, convexity::Float64,
                                            carry_period::Float64, spread_change::Float64,
                                            currency_return::Float64=0.0)
    dy = ytm_end - ytm_start
    carry_ret    = ytm_start * carry_period
    duration_ret = -mod_duration * dy
    convexity_ret = 0.5 * convexity * dy^2
    spread_ret    = -mod_duration * spread_change
    total_approx  = carry_ret + duration_ret + convexity_ret + spread_ret + currency_return
    # Residual (higher order terms)
    price_start = 1.0 / (1 + ytm_start)^(1/carry_period + 1)
    price_end   = 1.0 / (1 + ytm_end)^(1/carry_period + 1)
    actual_ret  = (price_end - price_start) / price_start
    residual    = actual_ret - total_approx
    return FixedIncomeAttribution(carry_ret, duration_ret, convexity_ret,
                                   spread_ret, currency_return, residual, total_approx)
end

function duration_bucket_attribution(portfolio_durations::Vector{Float64},
                                       benchmark_durations::Vector{Float64},
                                       yield_changes::Vector{Float64},
                                       weights_port::Vector{Float64},
                                       weights_bench::Vector{Float64},
                                       bucket_names::Vector{String})
    K = length(bucket_names)
    active_dur = portfolio_durations .- benchmark_durations
    active_wt  = weights_port .- weights_bench
    # Duration effect: active duration * yield change
    duration_attr = active_dur .* yield_changes .* weights_bench
    # Weight effect: active weight * benchmark duration * yield change
    weight_attr   = active_wt .* benchmark_durations .* yield_changes
    return (duration_attribution=duration_attr, weight_attribution=weight_attr,
            total=duration_attr + weight_attr, buckets=bucket_names)
end

function credit_attribution(portfolio_spreads::Vector{Float64},
                              benchmark_spreads::Vector{Float64},
                              mod_durations::Vector{Float64},
                              weights_port::Vector{Float64},
                              weights_bench::Vector{Float64})
    active_spread = portfolio_spreads .- benchmark_spreads
    active_wt     = weights_port .- weights_bench
    # Spread carry: active_wt * bench_spread * dt (dt=1 period)
    spread_carry  = active_wt .* benchmark_spreads
    # Spread change attribution
    spread_change = active_spread .* mod_durations .* weights_bench
    return (spread_carry=spread_carry, spread_change=spread_change,
            total_credit=spread_carry .+ spread_change)
end

# ============================================================
# SECTION 5: TRANSACTION COST ANALYSIS
# ============================================================

struct TCAResult
    symbol::String
    decision_price::Float64
    arrival_price::Float64
    vwap::Float64
    twap::Float64
    execution_price::Float64
    slippage_vs_arrival::Float64
    slippage_vs_vwap::Float64
    slippage_vs_twap::Float64
    market_impact_bps::Float64
    timing_alpha_bps::Float64
    spread_cost_bps::Float64
end

function compute_tca_detailed(symbol::String,
                               decision_price::Float64,
                               arrival_price::Float64,
                               execution_prices::Vector{Float64},
                               execution_volumes::Vector{Float64},
                               market_prices::Vector{Float64},
                               market_volumes::Vector{Float64},
                               bid_ask_spread::Float64,
                               side::Symbol=:buy)
    direction = side == :buy ? 1.0 : -1.0
    exec_vwap = sum(execution_prices .* execution_volumes) /
                (sum(execution_volumes) + 1e-10)
    market_vwap = sum(market_prices .* market_volumes) /
                  (sum(market_volumes) + 1e-10)
    market_twap = mean(market_prices)
    slip_arrival = direction * (exec_vwap - arrival_price) / arrival_price * 10000
    slip_vwap    = direction * (exec_vwap - market_vwap) / market_vwap * 10000
    slip_twap    = direction * (exec_vwap - market_twap) / market_twap * 10000
    timing_alpha = direction * (arrival_price - decision_price) / decision_price * 10000
    spread_cost  = bid_ask_spread / arrival_price * 10000 / 2
    return TCAResult(symbol, decision_price, arrival_price, market_vwap, market_twap,
                     exec_vwap, slip_arrival, slip_vwap, slip_twap,
                     slip_arrival, timing_alpha, spread_cost)
end

function tca_report(tca::TCAResult)
    println("=== TCA Report: ", tca.symbol, " ===")
    println("Decision price:   ", round(tca.decision_price, digits=4))
    println("Arrival price:    ", round(tca.arrival_price, digits=4))
    println("Execution VWAP:   ", round(tca.execution_price, digits=4))
    println("Market VWAP:      ", round(tca.vwap, digits=4))
    println("Slippage vs arr:  ", round(tca.slippage_vs_arrival, digits=2), " bps")
    println("Slippage vs VWAP: ", round(tca.slippage_vs_vwap, digits=2), " bps")
    println("Timing alpha:     ", round(tca.timing_alpha_bps, digits=2), " bps")
    println("Spread cost:      ", round(tca.spread_cost_bps, digits=2), " bps")
end

function aggregate_tca_stats(tca_list::Vector{TCAResult})
    n = length(tca_list)
    n == 0 && return nothing
    avg_slip   = mean(t.slippage_vs_arrival for t in tca_list)
    med_slip   = quantile([t.slippage_vs_arrival for t in tca_list], 0.5)
    avg_vwap   = mean(t.slippage_vs_vwap for t in tca_list)
    avg_timing = mean(t.timing_alpha_bps for t in tca_list)
    total_spread = sum(t.spread_cost_bps for t in tca_list)
    return (n_trades=n, avg_slippage_bps=avg_slip, median_slippage_bps=med_slip,
            avg_vwap_slippage_bps=avg_vwap, avg_timing_alpha_bps=avg_timing,
            total_spread_cost_bps=total_spread)
end

# ============================================================
# SECTION 6: PERFORMANCE MEASUREMENT
# ============================================================

function sharpe_ratio(returns::Vector{Float64}, rf::Float64=0.0;
                       annualize::Bool=true, periods::Int=252)
    excess = returns .- rf/periods
    sr = mean(excess) / (std(excess) + 1e-10)
    annualize && (sr *= sqrt(periods))
    return sr
end

function sortino_ratio(returns::Vector{Float64}, mar::Float64=0.0; periods::Int=252)
    excess = returns .- mar/periods
    downside_ret = [min(r, 0.0) for r in excess]
    downside_dev = sqrt(mean(downside_ret.^2))
    return mean(excess) / (downside_dev + 1e-10) * sqrt(periods)
end

function calmar_ratio(returns::Vector{Float64}; periods::Int=252)
    ann_return = mean(returns) * periods
    cum = cumsum(returns); peak = cum[1]; mdd = 0.0
    for v in cum; peak=max(peak,v); mdd=max(mdd,peak-v); end
    return ann_return / (mdd + 1e-10)
end

function omega_ratio(returns::Vector{Float64}, threshold::Float64=0.0)
    gains  = sum(max(r-threshold, 0.0) for r in returns)
    losses = sum(max(threshold-r, 0.0) for r in returns)
    return gains / (losses + 1e-10)
end

function information_ratio(active_returns::Vector{Float64}; periods::Int=252)
    return mean(active_returns) / (std(active_returns) + 1e-10) * sqrt(periods)
end

function upside_capture(portfolio_returns::Vector{Float64},
                          benchmark_returns::Vector{Float64})
    up_periods = findall(r -> r > 0, benchmark_returns)
    isempty(up_periods) && return 1.0
    return mean(portfolio_returns[up_periods]) / mean(benchmark_returns[up_periods])
end

function downside_capture(portfolio_returns::Vector{Float64},
                            benchmark_returns::Vector{Float64})
    dn_periods = findall(r -> r < 0, benchmark_returns)
    isempty(dn_periods) && return 1.0
    return mean(portfolio_returns[dn_periods]) / mean(benchmark_returns[dn_periods])
end

function active_share(portfolio_weights::Vector{Float64},
                       benchmark_weights::Vector{Float64})
    return 0.5 * sum(abs.(portfolio_weights .- benchmark_weights))
end

function portfolio_concentration(weights::Vector{Float64})
    hhi = sum(weights.^2)
    n = count(w -> w > 0, weights)
    effective_n = 1.0 / (hhi + 1e-10)
    return (hhi=hhi, effective_n=effective_n, n_positions=n)
end

function beta_adjusted_alpha(portfolio_returns::Vector{Float64},
                               benchmark_returns::Vector{Float64};
                               rf::Float64=0.0)
    T = length(portfolio_returns)
    X = hcat(ones(T), benchmark_returns .- rf)
    y = portfolio_returns .- rf
    beta_vec = (X'*X + 1e-8*I(2)) \ (X'*y)
    alpha = beta_vec[1] * 252  # annualized
    beta  = beta_vec[2]
    resid = y .- X*beta_vec
    treynor = mean(y) * 252 / (beta + 1e-10)
    return (alpha=alpha, beta=beta, treynor=treynor, residual_vol=std(resid)*sqrt(252))
end

# ============================================================
# SECTION 7: MULTI-ASSET PERFORMANCE ATTRIBUTION
# ============================================================

function cross_asset_attribution(asset_returns::Vector{Float64},
                                   asset_weights::Vector{Float64},
                                   bench_returns::Vector{Float64},
                                   bench_weights::Vector{Float64},
                                   asset_classes::Vector{String})
    n = length(asset_returns)
    port_return = dot(asset_returns, asset_weights)
    bench_return = dot(bench_returns, bench_weights)
    active = port_return - bench_return
    # Allocation effect
    alloc = (asset_weights .- bench_weights) .* (bench_returns .- bench_return)
    # Selection effect
    sel = bench_weights .* (asset_returns .- bench_returns)
    # Interaction
    inter = (asset_weights .- bench_weights) .* (asset_returns .- bench_returns)
    return (portfolio_return=port_return, benchmark_return=bench_return,
            active_return=active, allocation=alloc, selection=sel, interaction=inter,
            asset_classes=asset_classes)
end

function currency_attribution_detailed(local_returns::Vector{Float64},
                                         fx_returns::Vector{Float64},
                                         weights::Vector{Float64},
                                         bench_local_ret::Vector{Float64},
                                         bench_fx_ret::Vector{Float64},
                                         bench_weights::Vector{Float64},
                                         currency_names::Vector{String})
    # Total return = local return + FX return + local*FX (cross)
    port_total = (1 .+ local_returns) .* (1 .+ fx_returns) .- 1
    bench_total = (1 .+ bench_local_ret) .* (1 .+ bench_fx_ret) .- 1
    active_wt = weights .- bench_weights
    # Local return attribution (BHB on local)
    local_alloc = active_wt .* bench_local_ret
    local_sel   = bench_weights .* (local_returns .- bench_local_ret)
    # FX attribution
    fx_alloc = active_wt .* bench_fx_ret
    fx_sel   = bench_weights .* (fx_returns .- bench_fx_ret)
    return (local_allocation=local_alloc, local_selection=local_sel,
            fx_allocation=fx_alloc, fx_selection=fx_sel,
            currencies=currency_names)
end

function factor_timing_score(portfolio_betas::Matrix{Float64},
                               factor_returns::Matrix{Float64})
    # Score based on correlation of beta changes with factor return changes
    T, K = size(portfolio_betas)
    scores = zeros(K)
    for k in 1:K
        db = diff(portfolio_betas[:,k])
        fr = factor_returns[2:T, k]
        scores[k] = cor(db, fr)
    end
    return scores
end

function stress_test_attribution(weights::Vector{Float64},
                                   factor_shocks::Vector{Float64},
                                   factor_sensitivities::Matrix{Float64},
                                   scenario_name::String="Stress")
    # factor_sensitivities: N assets x K factors
    asset_returns = factor_sensitivities * factor_shocks
    portfolio_stress = dot(weights, asset_returns)
    contributions = weights .* asset_returns
    return (scenario=scenario_name, portfolio_stress_return=portfolio_stress,
            asset_contributions=contributions,
            worst_contributors=sortperm(contributions)[1:min(5,end)])
end

# ============================================================
# EXTENDED DEMO
# ============================================================

function demo_portfolio_attribution_extended()
    println("=== Portfolio Attribution Extended Demo ===")

    # Factor attribution
    fa = factor_attribution(0.08, 0.05, [0.06, 0.02, -0.01],
                             [1.1, 0.3, 0.2], [1.0, 0.0, 0.0],
                             ["Market","Size","Value"])
    println("Factor attribution: Market=", round(fa.factor_contributions[1]*100,digits=2),
            "% Spec=", round(fa.specific_return*100,digits=2), "%")

    # Risk budgeting
    n=5; w=fill(0.2,n)
    cov = 0.01*Matrix{Float64}(I,n,n) .+ 0.005*ones(n,n)
    rb = risk_budgeting(w, cov, ["A","B","C","D","E"])
    println("Portfolio vol: ", round(rb.portfolio_vol*100,digits=3), "%")
    println("Risk contrib %: ", round.(rb.risk_contrib_pct,digits=1))

    # Sharpe/Sortino
    rets = randn(252)*0.01 .+ 0.0003
    println("Sharpe: ", round(sharpe_ratio(rets), digits=2))
    println("Sortino: ", round(sortino_ratio(rets), digits=2))
    println("Active share example: ", round(active_share([0.3,0.4,0.3],[0.33,0.33,0.34]),digits=3))

    # TCA
    exec_px = [100.0, 100.2, 100.1, 100.3]; exec_vol = [250.0,250.0,250.0,250.0]
    mkt_px  = [100.0, 100.1, 100.2, 100.3]; mkt_vol  = [1000.0,800.0,1200.0,900.0]
    tca_res = compute_tca_detailed("BTCUSDT", 100.0, 100.0, exec_px, exec_vol,
                                    mkt_px, mkt_vol, 0.05)
    println("TCA slip vs VWAP: ", round(tca_res.slippage_vs_vwap, digits=2), " bps")

    # ERC weights
    erc_w = equal_risk_contribution_weights(cov)
    println("ERC weights: ", round.(erc_w, digits=3))

    # Stress test
    st = stress_test_attribution(w, [-0.1, -0.05, 0.02, 0.01, -0.03],
                                  randn(n,5), "2008 Crisis Scenario")
    println("Portfolio stress: ", round(st.portfolio_stress_return*100, digits=2), "%")
end

end # module PortfolioAttribution
