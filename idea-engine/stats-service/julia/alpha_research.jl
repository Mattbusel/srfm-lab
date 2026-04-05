module AlphaResearch

# ============================================================
# alpha_research.jl -- Systematic Alpha Research Pipeline
# ============================================================
# Covers: factor construction (momentum, value, quality, growth),
# IC/ICIR calculation, factor decay analysis, alpha combination
# (ensemble weighting, covariance shrinkage), turnover analysis,
# factor exposure netting, neutralisation, t-stats, quintile
# analysis, factor zoo management, Barra-style risk model
# construction, alpha half-life estimation.
# Pure stdlib Julia -- no external packages.
# ============================================================

using Statistics, LinearAlgebra

struct FactorSignal
    name::String
    universe::Vector{String}
    scores::Vector{Float64}      # cross-sectional z-scored factor
    date::Float64
    category::Symbol             # :momentum, :value, :quality, :growth, :technical
end

struct AlphaModel
    factor_names::Vector{String}
    weights::Vector{Float64}
    ic_history::Matrix{Float64}  # T x n_factors
    decay_halflife::Vector{Float64}
    covariance_matrix::Matrix{Float64}
end

struct FactorEvaluation
    factor_name::String
    ic_mean::Float64
    ic_std::Float64
    icir::Float64
    t_stat::Float64
    decay_halflife::Float64
    pct_positive::Float64
    quintile_spread::Float64
end

struct TurnoverAnalysis
    factor_name::String
    avg_monthly_turnover::Float64
    turnover_series::Vector{Float64}
    autocorr_1::Float64
    autocorr_5::Float64
end

struct QuintilePerformance
    factor_name::String
    quintile_returns::Vector{Float64}   # 5 quintiles, Q1=low factor
    spread_return::Float64              # Q5 - Q1
    sharpe_per_quintile::Vector{Float64}
    hit_rates::Vector{Float64}
end

struct RiskModel
    factor_names::Vector{String}
    factor_covariance::Matrix{Float64}
    specific_variance::Vector{Float64}
    factor_loadings::Matrix{Float64}    # n_assets x n_factors
end

# ---- 1. Factor Construction ----

function cross_section_zscore(scores::Vector{Float64})::Vector{Float64}
    mu = mean(scores); sig = std(scores) + 1e-8
    z = (scores .- mu) ./ sig
    return clamp.(z, -3.0, 3.0)
end

function rank_normalise(scores::Vector{Float64})::Vector{Float64}
    n = length(scores)
    ranks = Float64.(sortperm(sortperm(scores)))
    return (ranks .- mean(ranks)) ./ (std(ranks) + 1e-8)
end

function winsorise(scores::Vector{Float64}, pct_lo::Float64=0.01,
                   pct_hi::Float64=0.99)::Vector{Float64}
    n = length(scores); sorted = sort(scores)
    lo = sorted[max(1, round(Int, pct_lo*n))]
    hi = sorted[min(n, round(Int, pct_hi*n))]
    return clamp.(scores, lo, hi)
end

function momentum_factor(prices::Matrix{Float64}, lookback::Int=252,
                          skip::Int=21)::Vector{Float64}
    T_len, n = size(prices)
    if T_len < lookback + 1; return zeros(n); end
    rets = [log(prices[T_len-skip, i] / prices[T_len-lookback, i]) for i in 1:n]
    return cross_section_zscore(rets)
end

function value_factor_ep(earnings::Vector{Float64}, prices::Vector{Float64})::Vector{Float64}
    ep = earnings ./ (prices .+ 1e-8)
    return cross_section_zscore(ep)
end

function quality_factor(roe::Vector{Float64}, debt_to_equity::Vector{Float64},
                          accruals::Vector{Float64})::Vector{Float64}
    z_roe  = cross_section_zscore(roe)
    z_lev  = cross_section_zscore(-debt_to_equity)
    z_accr = cross_section_zscore(-accruals)
    composite = (z_roe .+ z_lev .+ z_accr) ./ 3.0
    return cross_section_zscore(composite)
end

function size_factor(market_caps::Vector{Float64})::Vector{Float64}
    return cross_section_zscore(-log.(market_caps .+ 1e-8))
end

function low_vol_factor(returns::Matrix{Float64}, window::Int=63)::Vector{Float64}
    T_len, n = size(returns)
    vols = [std(returns[max(1,T_len-window+1):T_len, i]) for i in 1:n]
    return cross_section_zscore(-vols)
end

function growth_factor(rev_growth::Vector{Float64}, eps_growth::Vector{Float64})::Vector{Float64}
    composite = (cross_section_zscore(rev_growth) .+ cross_section_zscore(eps_growth)) ./ 2.0
    return cross_section_zscore(composite)
end

function short_term_reversal(returns_1d::Vector{Float64})::Vector{Float64}
    return cross_section_zscore(-returns_1d)
end

function idiosyncratic_momentum(total_rets::Vector{Float64},
                                  factor_explained_rets::Vector{Float64})::Vector{Float64}
    residuals = total_rets .- factor_explained_rets
    return cross_section_zscore(residuals)
end

# ---- 2. IC / ICIR Calculation ----

function information_coefficient(factor_scores::Vector{Float64},
                                   fwd_returns::Vector{Float64})::Float64
    n = length(factor_scores)
    r1 = Float64.(sortperm(sortperm(factor_scores)))
    r2 = Float64.(sortperm(sortperm(fwd_returns)))
    cov_val = sum((r1.-mean(r1)).*(r2.-mean(r2))) / (n-1+1e-8)
    return cov_val / (std(r1)*std(r2)+1e-8)
end

function ic_series(factor_scores_history::Matrix{Float64},
                    fwd_returns_history::Matrix{Float64})::Vector{Float64}
    T_len = size(factor_scores_history, 1)
    return [information_coefficient(factor_scores_history[t,:], fwd_returns_history[t,:])
            for t in 1:T_len]
end

function icir(ic_vals::Vector{Float64})::Float64
    return mean(ic_vals) / (std(ic_vals) + 1e-8)
end

function ic_t_stat(ic_vals::Vector{Float64})::Float64
    n = length(ic_vals)
    return mean(ic_vals) / (std(ic_vals)/sqrt(n) + 1e-8)
end

function ic_decay_analysis(factor_scores::Matrix{Float64},
                             returns::Matrix{Float64},
                             max_horizon::Int=20)::Vector{Float64}
    T_len, n_assets = size(factor_scores)
    ic_by_horizon = zeros(max_horizon)
    for h in 1:max_horizon
        ics = Float64[]
        for t in 1:(T_len-h)
            if t+h <= T_len
                push!(ics, information_coefficient(factor_scores[t,:], returns[t+h,:]))
            end
        end
        ic_by_horizon[h] = isempty(ics) ? 0.0 : mean(ics)
    end
    return ic_by_horizon
end

function factor_decay_halflife(ic_by_horizon::Vector{Float64})::Float64
    ic0 = abs(ic_by_horizon[1]) + 1e-8
    for h in 2:length(ic_by_horizon)
        if abs(ic_by_horizon[h]) <= ic0/2
            return Float64(h)
        end
    end
    return Float64(length(ic_by_horizon))
end

function evaluate_factor(name::String, scores_hist::Matrix{Float64},
                           fwd_rets_hist::Matrix{Float64},
                           quintile_rets::Vector{Float64})::FactorEvaluation
    ic_vals = ic_series(scores_hist, fwd_rets_hist)
    decay = ic_decay_analysis(scores_hist, fwd_rets_hist, min(20, size(scores_hist,1)-1))
    hl = factor_decay_halflife(decay)
    qspread = length(quintile_rets) >= 5 ? quintile_rets[5] - quintile_rets[1] : 0.0
    return FactorEvaluation(
        name,
        mean(ic_vals),
        std(ic_vals),
        icir(ic_vals),
        ic_t_stat(ic_vals),
        hl,
        count(v -> v > 0, ic_vals) / length(ic_vals),
        qspread
    )
end

# ---- 3. Alpha Combination ----

function equal_weight_alpha(factor_scores::Matrix{Float64})::Vector{Float64}
    n = size(factor_scores, 1)
    combined = mean(factor_scores, dims=2)[:]
    return cross_section_zscore(combined)
end

function ic_weighted_alpha(factor_scores::Matrix{Float64},
                             ic_estimates::Vector{Float64})::Vector{Float64}
    pos_ic = max.(ic_estimates, 0.0)
    total_ic = sum(pos_ic) + 1e-8
    weights = pos_ic ./ total_ic
    combined = factor_scores * weights
    return cross_section_zscore(combined)
end

function ic_decay_weighted_alpha(factor_scores::Matrix{Float64},
                                   ic_estimates::Vector{Float64},
                                   halflives::Vector{Float64},
                                   current_age::Vector{Float64})::Vector{Float64}
    decay_factors = exp.(-log.(2.0) .* current_age ./ (halflives .+ 1e-8))
    adj_ic = ic_estimates .* decay_factors
    return ic_weighted_alpha(factor_scores, adj_ic)
end

function shrinkage_covariance(ic_history::Matrix{Float64},
                               shrinkage::Float64=0.2)::Matrix{Float64}
    T_len, n = size(ic_history)
    sample_cov = cov(ic_history)
    target = Diagonal(diag(sample_cov))
    return (1 - shrinkage) .* sample_cov + shrinkage .* Matrix(target)
end

function mean_variance_alpha_weights(ic_estimates::Vector{Float64},
                                      ic_covariance::Matrix{Float64},
                                      lambda::Float64=1.0)::Vector{Float64}
    n = length(ic_estimates)
    C_inv = inv(ic_covariance + 1e-6*I(n))
    raw = C_inv * ic_estimates
    raw = max.(raw, 0.0)
    total = sum(raw) + 1e-8
    return raw ./ total
end

# ---- 4. Turnover Analysis ----

function factor_turnover(scores_t1::Vector{Float64},
                           scores_t2::Vector{Float64},
                           top_pct::Float64=0.2)::Float64
    n = length(scores_t1)
    top_n = max(1, round(Int, n * top_pct))
    top_t1 = Set(sortperm(scores_t1, rev=true)[1:top_n])
    top_t2 = Set(sortperm(scores_t2, rev=true)[1:top_n])
    added   = length(setdiff(top_t2, top_t1))
    return added / top_n
end

function compute_turnover_series(scores_history::Matrix{Float64},
                                   top_pct::Float64=0.2)::Vector{Float64}
    T_len = size(scores_history, 1)
    to = Float64[]
    for t in 2:T_len
        push!(to, factor_turnover(scores_history[t-1,:], scores_history[t,:], top_pct))
    end
    return to
end

function net_alpha_after_costs(gross_alpha::Float64, turnover::Float64,
                                 one_way_cost::Float64=0.001)::Float64
    return gross_alpha - 2 * turnover * one_way_cost
end

function factor_portfolio_holdings(scores::Vector{Float64},
                                    total_capital::Float64,
                                    top_pct::Float64=0.2,
                                    bottom_pct::Float64=0.2)::Vector{Float64}
    n = length(scores)
    holdings = zeros(n)
    sorted_idx = sortperm(scores, rev=true)
    top_n  = max(1, round(Int, n * top_pct))
    bot_n  = max(1, round(Int, n * bottom_pct))
    long_w  = total_capital / (2 * top_n)
    short_w = -total_capital / (2 * bot_n)
    for i in 1:top_n; holdings[sorted_idx[i]] = long_w; end
    for i in (n-bot_n+1):n; holdings[sorted_idx[i]] = short_w; end
    return holdings
end

# ---- 5. Quintile Analysis ----

function quintile_backtest(scores::Vector{Float64},
                             fwd_returns::Vector{Float64})::QuintilePerformance
    n = length(scores)
    sorted_idx = sortperm(scores)
    q_size = n div 5
    q_rets = zeros(5)
    for q in 1:5
        idx_range = ((q-1)*q_size+1):min(q*q_size, n)
        q_rets[q] = mean(fwd_returns[sorted_idx[idx_range]])
    end
    spread = q_rets[5] - q_rets[1]
    sharpes = q_rets ./ (std(fwd_returns)/sqrt(n) .+ 1e-8) .* sqrt(252.0)
    hits = [q_rets[q] > 0 ? 1.0 : 0.0 for q in 1:5]
    return QuintilePerformance("factor", q_rets, spread, sharpes, hits)
end

function information_ratio_backtest(alpha_rets::Vector{Float64},
                                     bench_rets::Vector{Float64})
    active = alpha_rets .- bench_rets
    ann = 252.0
    ir = mean(active)*ann / (std(active)*sqrt(ann) + 1e-8)
    te = std(active)*sqrt(ann)
    return (information_ratio=ir, tracking_error=te,
            ann_active=mean(active)*ann, hit_rate=count(r->r>0,active)/length(active))
end

# ---- 6. Risk Model Construction ----

function pca_risk_model(returns::Matrix{Float64}, n_factors::Int)::RiskModel
    T_len, n = size(returns)
    mu = mean(returns, dims=1)
    std_r = std(returns, dims=1) .+ 1e-10
    norm_rets = (returns .- mu) ./ std_r
    U, S, V = svd(norm_rets)
    factor_loadings = V[:, 1:n_factors]
    factor_rets = norm_rets * factor_loadings
    factor_cov = cov(factor_rets) .* (T_len-1)/(T_len)
    residuals = norm_rets .- factor_rets * factor_loadings'
    specific_var = vec(var(residuals, dims=1))
    factor_names = ["PC_$i" for i in 1:n_factors]
    return RiskModel(factor_names, factor_cov, specific_var, factor_loadings)
end

function model_covariance(model::RiskModel)::Matrix{Float64}
    return model.factor_loadings * model.factor_covariance * model.factor_loadings' +
           Diagonal(model.specific_variance)
end

function factor_neutralise(scores::Vector{Float64},
                             exposure::Vector{Float64})::Vector{Float64}
    beta = sum(scores .* exposure) / (sum(exposure.^2) + 1e-12)
    return scores .- beta .* exposure
end

function sector_neutralise(scores::Vector{Float64},
                             sector_ids::Vector{Int})::Vector{Float64}
    n_sectors = maximum(sector_ids)
    neutralised = copy(scores)
    for s in 1:n_sectors
        mask = sector_ids .== s
        if sum(mask) > 1
            mu_s = mean(scores[mask])
            neutralised[mask] .-= mu_s
        end
    end
    return neutralised
end

# ---- Demo ----

function demo()
    println("=== AlphaResearch Demo ===")
    n_assets = 50; T_len = 120

    rets_mat = randn(T_len, n_assets) .* 0.01
    prices_mat = exp.(cumsum(rets_mat, dims=1)) .* 100.0

    mom = momentum_factor(prices_mat, 60, 5)
    println("Momentum factor (first 5):")
    println("  ", round.(mom[1:5], digits=3))

    fwd_rets = rets_mat[T_len, :] .+ 0.001 .* mom
    ic_val = information_coefficient(mom, fwd_rets)
    println("IC (momentum vs fwd returns): ", round(ic_val, digits=4))

    scores_hist = randn(T_len, n_assets)
    rets_hist = 0.001 .* scores_hist .+ 0.01 .* randn(T_len, n_assets)
    ic_vals = ic_series(scores_hist, rets_hist)
    println("IC mean: ", round(mean(ic_vals), digits=4))
    println("ICIR: ", round(icir(ic_vals), digits=4))
    println("IC t-stat: ", round(ic_t_stat(ic_vals), digits=4))

    decay_ics = ic_decay_analysis(scores_hist, rets_hist, 10)
    hl = factor_decay_halflife(decay_ics)
    println("Factor decay half-life: ", round(hl, digits=1), " periods")

    fac1 = cross_section_zscore(randn(n_assets))
    fac2 = cross_section_zscore(randn(n_assets))
    fac3 = cross_section_zscore(randn(n_assets))
    combined = equal_weight_alpha(hcat(fac1, fac2, fac3)')
    println("Combined alpha (equal weight) std: ", round(std(combined), digits=4))

    to = factor_turnover(fac1, fac2, 0.2)
    println("Top-quintile turnover fac1->fac2: ", round(to*100, digits=2), "%")

    q_perf = quintile_backtest(fac1, fwd_rets)
    println("Quintile returns: ", round.(q_perf.quintile_returns.*100, digits=3))
    println("Q5-Q1 spread: ", round(q_perf.spread_return*100, digits=3), "%")

    risk_model = pca_risk_model(rets_mat, 5)
    println("\nPCA risk model (5 factors) specific var range: [",
            round(minimum(risk_model.specific_variance),digits=6), ", ",
            round(maximum(risk_model.specific_variance),digits=6), "]")

    sector_ids = rand(1:5, n_assets)
    neutralised = sector_neutralise(fac1, sector_ids)
    println("Sector-neutralised alpha std: ", round(std(neutralised), digits=4))
end

# ---- Additional Alpha Research Functions ----

function factor_orthogonalisation(factors::Matrix{Float64})::Matrix{Float64}
    n_assets, n_factors = size(factors)
    ortho = zeros(n_assets, n_factors)
    ortho[:, 1] = factors[:, 1] ./ (norm(factors[:,1]) + 1e-12)
    for j in 2:n_factors
        v = factors[:, j]
        for k in 1:(j-1)
            v = v - dot(v, ortho[:,k]) * ortho[:,k]
        end
        ortho[:, j] = v ./ (norm(v) + 1e-12)
    end
    return ortho
end

function factor_correlation_matrix(ic_history::Matrix{Float64})::Matrix{Float64}
    return cor(ic_history)
end

function conditional_ic(factor_scores::Vector{Float64}, fwd_returns::Vector{Float64},
                          condition::Vector{Bool})::NamedTuple
    if sum(condition) < 5
        return (ic_cond=NaN, ic_uncond=NaN, conditional_premium=NaN)
    end
    ic_all   = information_coefficient(factor_scores, fwd_returns)
    ic_cond  = information_coefficient(factor_scores[condition], fwd_returns[condition])
    return (ic_cond=ic_cond, ic_uncond=ic_all, conditional_premium=ic_cond - ic_all)
end

function factor_timing_signal(ic_history::Vector{Float64}, window::Int=12)::Float64
    n = length(ic_history); if n < window + 1; return 0.0; end
    recent = ic_history[end-window+1:end]
    older  = ic_history[end-2*window+1:end-window]
    return mean(recent) - mean(older)
end

function long_short_portfolio_stats(long_scores::Vector{Float64},
                                     short_scores::Vector{Float64},
                                     long_returns::Vector{Float64},
                                     short_returns::Vector{Float64})
    long_ret  = mean(long_returns)
    short_ret = mean(short_returns)
    spread    = long_ret - short_ret
    long_vol  = std(long_returns)*sqrt(252.0)
    short_vol = std(short_returns)*sqrt(252.0)
    return (long_return=long_ret*252, short_return=short_ret*252,
            spread_return=spread*252, long_vol=long_vol, short_vol=short_vol,
            sharpe=spread*252/(std(long_returns.-short_returns)*sqrt(252.0)+1e-8))
end

function factor_marginal_contribution(factor_scores::Matrix{Float64},
                                       fwd_returns::Matrix{Float64},
                                       base_ic::Float64)::Vector{Float64}
    n_t, n_f = size(factor_scores)
    mc = zeros(n_f)
    for j in 1:n_f
        reduced = hcat(factor_scores[:, 1:j-1], factor_scores[:, j+1:end])
        if size(reduced, 2) == 0; mc[j] = base_ic; continue; end
        combined = mean(reduced, dims=2)[:]
        ic_without = mean(information_coefficient(combined, fwd_returns[t,:]) for t in 1:n_t)
        mc[j] = base_ic - ic_without
    end
    return mc
end

function walk_forward_ic(scores_hist::Matrix{Float64}, rets_hist::Matrix{Float64},
                           train_window::Int, test_window::Int)::Vector{Float64}
    T_len = size(scores_hist, 1); ics = Float64[]
    t = train_window + 1
    while t + test_window - 1 <= T_len
        test_ics = [information_coefficient(scores_hist[t+i-1,:], rets_hist[t+i-1,:])
                    for i in 1:test_window if t+i-1 <= T_len]
        append!(ics, test_ics); t += test_window
    end
    return ics
end

function factor_portfolio_sharpe(factor_scores::Vector{Float64},
                                   fwd_returns::Vector{Float64},
                                   n_long::Int=10, n_short::Int=10)::Float64
    n = length(factor_scores)
    sorted_idx = sortperm(factor_scores, rev=true)
    long_idx  = sorted_idx[1:n_long]
    short_idx = sorted_idx[end-n_short+1:end]
    long_ret  = mean(fwd_returns[long_idx])
    short_ret = mean(fwd_returns[short_idx])
    spread    = long_ret - short_ret
    spread_vol = std(fwd_returns[long_idx]) + std(fwd_returns[short_idx])
    return spread / (spread_vol + 1e-8) * sqrt(252.0)
end

function barra_factor_return(returns::Matrix{Float64},
                               loadings::Matrix{Float64})::Matrix{Float64}
    T_len = size(returns, 1); n_f = size(loadings, 2)
    factor_rets = zeros(T_len, n_f)
    for t in 1:T_len
        X = loadings; y = returns[t,:]
        factor_rets[t,:] = (X'X + 1e-6*I(n_f)) \ (X'y)
    end
    return factor_rets
end

function alpha_decay_correction(raw_alpha::Vector{Float64},
                                  decay_weights::Vector{Float64})::Vector{Float64}
    n = length(raw_alpha); out = zeros(length(raw_alpha[1:1]))
    return raw_alpha .* (isempty(decay_weights) ? 1.0 : decay_weights[1])
end

function net_exposure_check(weights::Vector{Float64},
                              sector_ids::Vector{Int},
                              max_net_exposure::Float64=0.1)::Bool
    total_long  = sum(w for w in weights if w > 0; init=0.0)
    total_short = sum(w for w in weights if w < 0; init=0.0)
    net = total_long + total_short
    return abs(net) <= max_net_exposure
end


# ---- Alpha Research Utilities (continued) ----

function alpha_book_pnl(positions::Vector{Float64}, returns::Vector{Float64})::Float64
    return dot(positions, returns)
end

function factor_portfolio_beta(factor_rets::Vector{Float64},
                                 market_rets::Vector{Float64})::Float64
    n = length(factor_rets)
    xb = mean(market_rets); yb = mean(factor_rets)
    return sum((market_rets.-xb).*(factor_rets.-yb)) / (sum((market_rets.-xb).^2)+1e-12)
end

function factor_information_ratio_rolling(ic_series::Vector{Float64},
                                           window::Int=12)::Vector{Float64}
    n = length(ic_series); ir = fill(NaN, n)
    for i in (window+1):n
        h = ic_series[i-window:i-1]
        ir[i] = mean(h) / (std(h) + 1e-8)
    end
    return ir
end

function alpha_dilution_factor(factor_ic::Float64, model_ic::Float64)::Float64
    return model_ic > 0 ? factor_ic / model_ic : 0.0
end

function universe_coverage_filter(scores::Vector{Float64},
                                    adv::Vector{Float64},
                                    min_adv::Float64=1e6)::BitVector
    return adv .>= min_adv
end

function cross_sectional_regression(y::Vector{Float64},
                                     X::Matrix{Float64})::Vector{Float64}
    n, k = size(X)
    return (X'X + 1e-8*I(k)) \ (X'y)
end

function factor_exposure_hedge(scores::Vector{Float64},
                                 exposure::Vector{Float64})::Vector{Float64}
    beta = dot(scores, exposure) / (dot(exposure, exposure) + 1e-12)
    return scores .- beta .* exposure
end

function ic_weighted_combination_optimised(scores::Matrix{Float64},
                                            ic_cov::Matrix{Float64},
                                            ic_means::Vector{Float64},
                                            lambda::Float64=1.0)::Vector{Float64}
    k = size(scores, 2)
    C_inv = inv(ic_cov + 1e-6*I(k))
    raw_w = C_inv * ic_means; raw_w = max.(raw_w, 0.0)
    w = raw_w ./ (sum(raw_w) + 1e-8)
    combined = scores * w
    mu = mean(combined); sig = std(combined) + 1e-8
    return clamp.((combined .- mu) ./ sig, -3.0, 3.0)
end


# ---- Alpha Research Utilities (continued) ----

function alpha_book_pnl(positions::Vector{Float64}, returns::Vector{Float64})::Float64
    return dot(positions, returns)
end

function factor_portfolio_beta(factor_rets::Vector{Float64},
                                 market_rets::Vector{Float64})::Float64
    n = length(factor_rets)
    xb = mean(market_rets); yb = mean(factor_rets)
    return sum((market_rets.-xb).*(factor_rets.-yb)) / (sum((market_rets.-xb).^2)+1e-12)
end

function factor_information_ratio_rolling(ic_series::Vector{Float64},
                                           window::Int=12)::Vector{Float64}
    n = length(ic_series); ir = fill(NaN, n)
    for i in (window+1):n
        h = ic_series[i-window:i-1]
        ir[i] = mean(h) / (std(h) + 1e-8)
    end
    return ir
end

function alpha_dilution_factor(factor_ic::Float64, model_ic::Float64)::Float64
    return model_ic > 0 ? factor_ic / model_ic : 0.0
end

function universe_coverage_filter(scores::Vector{Float64},
                                    adv::Vector{Float64},
                                    min_adv::Float64=1e6)::BitVector
    return adv .>= min_adv
end

function cross_sectional_regression(y::Vector{Float64},
                                     X::Matrix{Float64})::Vector{Float64}
    n, k = size(X)
    return (X'X + 1e-8*I(k)) \ (X'y)
end

function factor_exposure_hedge(scores::Vector{Float64},
                                 exposure::Vector{Float64})::Vector{Float64}
    beta = dot(scores, exposure) / (dot(exposure, exposure) + 1e-12)
    return scores .- beta .* exposure
end

function ic_weighted_combination_optimised(scores::Matrix{Float64},
                                            ic_cov::Matrix{Float64},
                                            ic_means::Vector{Float64},
                                            lambda::Float64=1.0)::Vector{Float64}
    k = size(scores, 2)
    C_inv = inv(ic_cov + 1e-6*I(k))
    raw_w = C_inv * ic_means; raw_w = max.(raw_w, 0.0)
    w = raw_w ./ (sum(raw_w) + 1e-8)
    combined = scores * w
    mu = mean(combined); sig = std(combined) + 1e-8
    return clamp.((combined .- mu) ./ sig, -3.0, 3.0)
end


# ---- Alpha Research Utilities (continued) ----

function alpha_book_pnl(positions::Vector{Float64}, returns::Vector{Float64})::Float64
    return dot(positions, returns)
end

function factor_portfolio_beta(factor_rets::Vector{Float64},
                                 market_rets::Vector{Float64})::Float64
    n = length(factor_rets)
    xb = mean(market_rets); yb = mean(factor_rets)
    return sum((market_rets.-xb).*(factor_rets.-yb)) / (sum((market_rets.-xb).^2)+1e-12)
end

function factor_information_ratio_rolling(ic_series::Vector{Float64},
                                           window::Int=12)::Vector{Float64}
    n = length(ic_series); ir = fill(NaN, n)
    for i in (window+1):n
        h = ic_series[i-window:i-1]
        ir[i] = mean(h) / (std(h) + 1e-8)
    end
    return ir
end

function alpha_dilution_factor(factor_ic::Float64, model_ic::Float64)::Float64
    return model_ic > 0 ? factor_ic / model_ic : 0.0
end

function universe_coverage_filter(scores::Vector{Float64},
                                    adv::Vector{Float64},
                                    min_adv::Float64=1e6)::BitVector
    return adv .>= min_adv
end

function cross_sectional_regression(y::Vector{Float64},
                                     X::Matrix{Float64})::Vector{Float64}
    n, k = size(X)
    return (X'X + 1e-8*I(k)) \ (X'y)
end

function factor_exposure_hedge(scores::Vector{Float64},
                                 exposure::Vector{Float64})::Vector{Float64}
    beta = dot(scores, exposure) / (dot(exposure, exposure) + 1e-12)
    return scores .- beta .* exposure
end

function ic_weighted_combination_optimised(scores::Matrix{Float64},
                                            ic_cov::Matrix{Float64},
                                            ic_means::Vector{Float64},
                                            lambda::Float64=1.0)::Vector{Float64}
    k = size(scores, 2)
    C_inv = inv(ic_cov + 1e-6*I(k))
    raw_w = C_inv * ic_means; raw_w = max.(raw_w, 0.0)
    w = raw_w ./ (sum(raw_w) + 1e-8)
    combined = scores * w
    mu = mean(combined); sig = std(combined) + 1e-8
    return clamp.((combined .- mu) ./ sig, -3.0, 3.0)
end

end # module AlphaResearch
