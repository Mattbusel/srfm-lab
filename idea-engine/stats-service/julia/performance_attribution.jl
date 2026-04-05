# =============================================================================
# performance_attribution.jl — Factor-Based Performance Attribution
# =============================================================================
# Provides:
#   - BrinsonAttribution     BHB decomposition: selection + timing + interaction
#   - FamaFrenchRegression   Factor regression: α, β_BTC, β_mktcap, β_vol
#   - RollingAttribution     90-day rolling attribution stability
#   - InformationRatioByFactor  IR per signal/factor
#   - run_performance_attribution  Top-level driver
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, JSON3
# =============================================================================

module PerformanceAttribution

using Statistics
using LinearAlgebra
using JSON3

export BrinsonAttribution, FamaFrenchRegression
export RollingAttribution, InformationRatioByFactor
export run_performance_attribution

# ── OLS Utilities ─────────────────────────────────────────────────────────────

"""OLS: y = X β + ε. Returns (β, residuals, R², t_stats, std_errors)."""
function _ols(y::Vector{Float64}, X::Matrix{Float64})
    n, k = size(X)
    n > k || error("OLS: need n > k (n=$n, k=$k)")

    β    = (X'X + I*1e-10) \ (X'y)
    ŷ    = X * β
    ε    = y - ŷ

    sse   = dot(ε, ε)
    sst   = sum((y .- mean(y)).^2)
    r_sq  = sst > 1e-12 ? 1.0 - sse / sst : 0.0
    σ²    = sse / max(n - k, 1)
    cov_β = σ² .* inv(X'X + I*1e-10)
    se    = sqrt.(max.(diag(cov_β), 0.0))
    t     = β ./ max.(se, 1e-12)

    (β=β, residuals=ε, r_squared=r_sq, t_stats=t, std_errors=se, fitted=ŷ, sigma2=σ²)
end

"""Robust OLS (iteratively re-weighted least squares, Huber weights)."""
function _robust_ols(y::Vector{Float64}, X::Matrix{Float64}; n_iter::Int=20)
    n, k = size(X)
    w     = ones(n)
    β     = zeros(k)

    for _ in 1:n_iter
        W     = Diagonal(w)
        β_new = ((X'W*X) + I*1e-10) \ (X'W*y)
        resid = y .- X*β_new
        mad   = median(abs.(resid))
        σ_rob = max(mad / 0.6745, 1e-12)   # MAD-based scale
        # Huber weights: w_i = min(1, c / |ε_i/σ|)  with c = 1.345
        w     = min.(1.0, 1.345 ./ max.(abs.(resid) ./ σ_rob, 1e-12))
        abs.(β_new - β) |> maximum < 1e-8 && (β = β_new; break)
        β = β_new
    end

    resid = y .- X * β
    sse   = dot(resid, resid)
    sst   = sum((y .- mean(y)).^2)
    r_sq  = sst > 1e-12 ? 1.0 - sse / sst : 0.0

    (β=β, residuals=resid, r_squared=r_sq)
end

# ── Brinson-Hood-Beebower Attribution ────────────────────────────────────────

"""
Brinson-Hood-Beebower (BHB) performance attribution.

Decomposes total active return into three effects:
  Allocation effect  = (w_p - w_b) * (r_b - r_total_b)
  Selection effect   = w_b * (r_p - r_b)
  Interaction effect = (w_p - w_b) * (r_p - r_b)

Where:
  w_p, w_b = portfolio and benchmark weights per bucket
  r_p, r_b = portfolio and benchmark returns per bucket
  r_total_b = total benchmark return

# Arguments
- `bucket_names`  : labels for each sub-group (e.g. instruments or strategies)
- `port_weights`  : portfolio weights per bucket (sum ≈ 1)
- `bench_weights` : benchmark weights per bucket
- `port_returns`  : portfolio returns per bucket
- `bench_returns` : benchmark returns per bucket

# Returns
NamedTuple: per-bucket effects + totals
"""
function BrinsonAttribution(
    bucket_names::Vector{String},
    port_weights::Vector{Float64},
    bench_weights::Vector{Float64},
    port_returns::Vector{Float64},
    bench_returns::Vector{Float64}
)
    n = length(bucket_names)
    n == length(port_weights) == length(bench_weights) ==
        length(port_returns) == length(bench_returns) ||
        error("All input vectors must have equal length")

    # Total benchmark return
    r_total_b = dot(bench_weights, bench_returns)

    # Per-bucket effects
    alloc    = (port_weights .- bench_weights) .* (bench_returns .- r_total_b)
    select   = bench_weights .* (port_returns .- bench_returns)
    interact = (port_weights .- bench_weights) .* (port_returns .- bench_returns)

    # Total active return
    active_return = dot(port_weights, port_returns) - r_total_b

    buckets = map(enumerate(bucket_names)) do (i, nm)
        (
            bucket        = nm,
            port_weight   = port_weights[i],
            bench_weight  = bench_weights[i],
            port_return   = port_returns[i],
            bench_return  = bench_returns[i],
            alloc_effect  = alloc[i],
            select_effect = select[i],
            interact_effect = interact[i],
            total_effect  = alloc[i] + select[i] + interact[i]
        )
    end

    (
        buckets           = buckets,
        total_allocation  = sum(alloc),
        total_selection   = sum(select),
        total_interaction = sum(interact),
        total_active      = active_return,
        attribution_check = sum(alloc) + sum(select) + sum(interact),   # ≈ active_return
        benchmark_return  = r_total_b,
        portfolio_return  = dot(port_weights, port_returns),
        dominant_effect   = argmax([abs(sum(alloc)), abs(sum(select)), abs(sum(interact))]) |>
            i -> ["allocation", "selection", "interaction"][i]
    )
end

# ── Fama-French Style Factor Regression ──────────────────────────────────────

"""
Fama-French style factor regression on strategy returns.

Model:
  R_strategy = α + β_BTC * R_BTC + β_mktcap * F_mktcap + β_vol * F_vol + ε

Also runs robust regression (IRLS) and reports the comparison.

# Arguments
- `strategy_returns` : T-vector of daily/hourly strategy returns
- `btc_returns`      : T-vector of BTC returns
- `mktcap_factor`    : T-vector of log market cap factor (e.g. large vs small cap)
- `vol_factor`       : T-vector of volatility factor (e.g. realised vol)
- `factor_names`     : optional override for factor names

# Returns
NamedTuple: (alpha, betas, t_stats, r_squared, robust_betas, residuals)
"""
function FamaFrenchRegression(
    strategy_returns::AbstractVector{<:Real},
    btc_returns::AbstractVector{<:Real},
    mktcap_factor::AbstractVector{<:Real},
    vol_factor::AbstractVector{<:Real};
    factor_names::Vector{String} = ["BTC", "MktCap", "Vol"]
)
    T = length(strategy_returns)
    T == length(btc_returns) == length(mktcap_factor) == length(vol_factor) ||
        error("All input vectors must have equal length")
    T < 10 && error("Need at least 10 observations for factor regression")

    y  = Float64.(strategy_returns)
    X  = hcat(
            ones(T),
            Float64.(btc_returns),
            Float64.(mktcap_factor),
            Float64.(vol_factor)
        )

    # OLS
    ols  = _ols(y, X)
    α    = ols.β[1]
    β    = ols.β[2:end]

    # Robust OLS
    rob  = _robust_ols(y, X)
    α_rob = rob.β[1]
    β_rob = rob.β[2:end]

    # Annualised alpha (assuming daily returns)
    α_ann = α * 252.0
    α_ann_rob = α_rob * 252.0

    # Information ratio: α / std(residuals)
    ir = std(ols.residuals) > 1e-12 ? α / std(ols.residuals) : 0.0

    (
        alpha            = α,
        alpha_annualised = α_ann,
        alpha_robust_ann = α_ann_rob,
        betas            = Dict(factor_names[k] => β[k] for k in 1:length(β)),
        betas_robust     = Dict(factor_names[k] => β_rob[k] for k in 1:length(β_rob)),
        t_stats          = Dict("alpha" => ols.t_stats[1],
                                 factor_names[1] => ols.t_stats[2],
                                 factor_names[2] => ols.t_stats[3],
                                 factor_names[3] => ols.t_stats[4]),
        std_errors       = ols.std_errors,
        r_squared        = ols.r_squared,
        information_ratio = ir,
        residuals        = ols.residuals,
        fitted           = ols.fitted,
        n_obs            = T,
        factor_names     = factor_names,
        interpretation   = begin
            sig_btc = abs(ols.t_stats[2]) > 2.0
            sig_vol = abs(ols.t_stats[4]) > 2.0
            if α > 0 && abs(ols.t_stats[1]) > 2.0
                "Significant positive alpha ($(round(α_ann*100; digits=2))%/yr); " *
                "$(sig_btc ? "BTC-exposed" : "BTC-independent"); " *
                "$(sig_vol ? "vol-sensitive" : "vol-neutral")"
            else
                "No significant alpha; return explained by factor exposures"
            end
        end
    )
end

# ── Rolling Attribution ───────────────────────────────────────────────────────

"""
Compute rolling 90-day Fama-French attribution to detect alpha stability.

# Arguments
- `strategy_returns`, `btc_returns`, `mktcap_factor`, `vol_factor` : time series
- `window` : rolling window in observations (default 90)
- `step`   : step size between windows (default 1)

# Returns
NamedTuple: rolling alpha, betas, R² over time
"""
function RollingAttribution(
    strategy_returns::AbstractVector{<:Real},
    btc_returns::AbstractVector{<:Real},
    mktcap_factor::AbstractVector{<:Real},
    vol_factor::AbstractVector{<:Real};
    window::Int = 90,
    step::Int   = 5,
    factor_names::Vector{String} = ["BTC", "MktCap", "Vol"]
)
    T     = length(strategy_returns)
    T > window || error("Series too short for rolling window=$window")

    idxs  = collect(window:step:T)
    n_win = length(idxs)

    rolling_alpha  = zeros(n_win)
    rolling_betas  = zeros(n_win, 3)
    rolling_r2     = zeros(n_win)
    rolling_ir     = zeros(n_win)

    for (k, t_end) in enumerate(idxs)
        t_start = t_end - window + 1
        slice   = t_start:t_end

        y_w  = Float64.(strategy_returns[slice])
        btc_w = Float64.(btc_returns[slice])
        mkt_w = Float64.(mktcap_factor[slice])
        vol_w = Float64.(vol_factor[slice])

        try
            res = FamaFrenchRegression(y_w, btc_w, mkt_w, vol_w; factor_names=factor_names)
            rolling_alpha[k]     = res.alpha_annualised
            rolling_betas[k, 1]  = res.betas[factor_names[1]]
            rolling_betas[k, 2]  = res.betas[factor_names[2]]
            rolling_betas[k, 3]  = res.betas[factor_names[3]]
            rolling_r2[k]        = res.r_squared
            rolling_ir[k]        = res.information_ratio
        catch
            # Window too small or degenerate — leave as zero
        end
    end

    # Alpha stability: fraction of windows with positive alpha
    frac_positive = mean(rolling_alpha .> 0)
    alpha_drift   = std(rolling_alpha)

    (
        window_ends       = idxs,
        rolling_alpha     = rolling_alpha,
        rolling_beta_btc  = rolling_betas[:, 1],
        rolling_beta_mktcap = rolling_betas[:, 2],
        rolling_beta_vol  = rolling_betas[:, 3],
        rolling_r2        = rolling_r2,
        rolling_ir        = rolling_ir,
        alpha_mean        = mean(rolling_alpha),
        alpha_std         = alpha_drift,
        alpha_fraction_positive = frac_positive,
        stability_interpretation = if frac_positive > 0.75 && alpha_drift < 0.05
            "STABLE: alpha consistently positive and low drift"
        elseif frac_positive > 0.5
            "MODERATE: alpha positive in majority of windows but drifting"
        else
            "UNSTABLE: alpha frequently negative — strategy may be deteriorating"
        end
    )
end

# ── Information Ratio by Factor ───────────────────────────────────────────────

"""
Compute Information Ratio (IR) contribution per signal/factor.

For each factor, regress it against residual strategy returns to measure
its incremental contribution to the information ratio.

IR per factor = α_factor / std(ε_factor)

# Arguments
- `strategy_returns` : total strategy returns
- `factors`          : matrix of factor/signal time series (T × k)
- `factor_names`     : names of each factor column

# Returns
NamedTuple: IR per factor, sorted by |IR|
"""
function InformationRatioByFactor(
    strategy_returns::AbstractVector{<:Real},
    factors::Matrix{Float64};
    factor_names::Vector{String} = String[]
)
    T, k = size(factors)
    T == length(strategy_returns) || error("strategy_returns and factors must have equal length")
    names = isempty(factor_names) ? ["F$i" for i in 1:k] : factor_names

    y       = Float64.(strategy_returns)
    y_mean  = mean(y)
    y_std   = std(y)

    ir_results = map(1:k) do fi
        f_raw = factors[:, fi]
        f     = (f_raw .- mean(f_raw)) ./ (std(f_raw) + 1e-12)

        # Regress strategy on factor
        X_f   = hcat(ones(T), f)
        ols   = _ols(y, X_f)

        # IC (information coefficient): correlation of factor with forward returns
        ic = cor(f, y)

        # Factor-specific IR
        ir_f = std(ols.residuals) > 1e-12 ?
            ols.β[1] / std(ols.residuals) : 0.0

        # Incremental R²: compare full model vs model without this factor
        ols_base = _ols(y, ones(T, 1))
        incr_r2  = ols.r_squared - ols_base.r_squared

        (
            factor         = names[fi],
            beta           = ols.β[2],
            alpha          = ols.β[1],
            t_stat_factor  = ols.t_stats[2],
            r_squared      = ols.r_squared,
            incremental_r2 = max(incr_r2, 0.0),
            ic             = ic,
            ir             = ir_f,
            significant    = abs(ols.t_stats[2]) > 2.0
        )
    end

    # Sort by |IR| descending
    sort!(ir_results, by = r -> -abs(r.ir))

    total_ir = sum(r.ir for r in ir_results if r.significant)

    (
        factor_ir     = ir_results,
        total_ir      = total_ir,
        top_factor    = ir_results[1].factor,
        n_significant = count(r -> r.significant, ir_results),
        summary       = "Top contributing factor: $(ir_results[1].factor) " *
                        "(IR=$(round(ir_results[1].ir; digits=3)), " *
                        "IC=$(round(ir_results[1].ic; digits=3)))"
    )
end

# ── Top-level driver ──────────────────────────────────────────────────────────

"""
Run full performance attribution pipeline.

Writes `performance_attribution_results.json` to `\$STATS_OUTPUT_DIR`.
"""
function run_performance_attribution(
    strategy_returns::Vector{Float64},
    btc_returns::Vector{Float64},
    mktcap_factor::Vector{Float64},
    vol_factor::Vector{Float64};
    # BHB attribution inputs
    bucket_names::Vector{String}   = String[],
    port_weights::Vector{Float64}  = Float64[],
    bench_weights::Vector{Float64} = Float64[],
    port_bucket_returns::Vector{Float64}  = Float64[],
    bench_bucket_returns::Vector{Float64} = Float64[],
    # Factor signals for IR computation
    signal_matrix::Matrix{Float64}      = Matrix{Float64}(undef, 0, 0),
    signal_names::Vector{String}        = String[],
    output_dir::String = get(ENV, "STATS_OUTPUT_DIR",
                              joinpath(@__DIR__, "..", "output"))
)
    println("[attribution] Running Fama-French factor regression...")
    ff = FamaFrenchRegression(strategy_returns, btc_returns, mktcap_factor, vol_factor)

    println("[attribution] Running rolling 90-day attribution...")
    rolling = RollingAttribution(strategy_returns, btc_returns, mktcap_factor, vol_factor;
        window=min(90, length(strategy_returns) ÷ 2))

    result = Dict{String,Any}(
        "fama_french" => Dict(
            "alpha_annualised_pct" => ff.alpha_annualised * 100,
            "alpha_robust_ann_pct" => ff.alpha_robust_ann * 100,
            "betas"               => ff.betas,
            "betas_robust"        => ff.betas_robust,
            "t_stats"             => ff.t_stats,
            "r_squared"           => ff.r_squared,
            "information_ratio"   => ff.information_ratio,
            "interpretation"      => ff.interpretation,
            "n_obs"               => ff.n_obs
        ),
        "rolling_attribution" => Dict(
            "window_ends"              => rolling.window_ends,
            "rolling_alpha_ann_pct"    => rolling.rolling_alpha .* 100,
            "rolling_beta_btc"         => rolling.rolling_beta_btc,
            "rolling_beta_vol"         => rolling.rolling_beta_vol,
            "rolling_r2"               => rolling.rolling_r2,
            "alpha_mean_ann_pct"       => rolling.alpha_mean * 100,
            "alpha_std_ann_pct"        => rolling.alpha_std * 100,
            "alpha_fraction_positive"  => rolling.alpha_fraction_positive,
            "stability"                => rolling.stability_interpretation
        )
    )

    # Optional BHB attribution
    if !isempty(bucket_names)
        println("[attribution] Running Brinson-Hood-Beebower attribution...")
        bhb = BrinsonAttribution(bucket_names, port_weights, bench_weights,
                                  port_bucket_returns, bench_bucket_returns)
        result["brinson_attribution"] = Dict(
            "total_active_return" => bhb.total_active,
            "allocation_effect"   => bhb.total_allocation,
            "selection_effect"    => bhb.total_selection,
            "interaction_effect"  => bhb.total_interaction,
            "dominant_effect"     => bhb.dominant_effect,
            "buckets" => map(b -> Dict(
                "bucket"   => b.bucket,
                "alloc"    => b.alloc_effect,
                "select"   => b.select_effect,
                "interact" => b.interact_effect
            ), bhb.buckets)
        )
    end

    # Optional IR by factor
    if !isempty(signal_matrix)
        println("[attribution] Computing IR by factor/signal...")
        ir_res = InformationRatioByFactor(strategy_returns, signal_matrix;
                                           factor_names=signal_names)
        result["ir_by_factor"] = Dict(
            "top_factor"    => ir_res.top_factor,
            "total_ir"      => ir_res.total_ir,
            "n_significant" => ir_res.n_significant,
            "summary"       => ir_res.summary,
            "factors"       => map(r -> Dict(
                "factor"  => r.factor,
                "ir"      => r.ir,
                "ic"      => r.ic,
                "beta"    => r.beta,
                "r2"      => r.r_squared
            ), ir_res.factor_ir)
        )
    end

    mkpath(output_dir)
    out_path = joinpath(output_dir, "performance_attribution_results.json")
    open(out_path, "w") do io
        write(io, JSON3.write(result))
    end
    println("[attribution] Results written to $out_path")

    result
end

end  # module PerformanceAttribution

# ── CLI self-test ─────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    using .PerformanceAttribution
    using Statistics, LinearAlgebra

    println("[performance_attribution] Running self-test...")

    T   = 300
    rng = let s = UInt64(999)
        () -> begin
            s = s * 6364136223846793005 + 1442695040888963407
            (Float64(s >> 11) / Float64(2^53) - 0.5) * 2
        end
    end

    btc   = [rng() * 0.03 for _ in 1:T]
    mktcap = [rng() * 0.01 for _ in 1:T]
    vol   = [rng() * 0.02 for _ in 1:T]
    strat = 0.0002 .+ 0.5 .* btc .+ 0.2 .* vol .+ [rng() * 0.005 for _ in 1:T]

    # BHB inputs
    buckets = ["BTC", "ETH", "SOL"]
    pw  = [0.50, 0.30, 0.20]
    bw  = [0.40, 0.35, 0.25]
    pr  = [0.08, 0.12, 0.15]
    br  = [0.06, 0.10, 0.11]

    # Signal matrix
    sigs = hcat(btc, vol, mktcap)

    result = run_performance_attribution(strat, btc, mktcap, vol;
        bucket_names=buckets, port_weights=pw, bench_weights=bw,
        port_bucket_returns=pr, bench_bucket_returns=br,
        signal_matrix=sigs, signal_names=["BTC_ret", "Vol_factor", "MktCap_factor"])

    ff = result["fama_french"]
    println("  FF α = $(round(ff["alpha_annualised_pct"]; digits=2))%/yr")
    println("  FF R² = $(round(ff["r_squared"]; digits=4))")
    println("  $(ff["interpretation"])")

    ra = result["rolling_attribution"]
    println("  Rolling alpha: mean=$(round(ra["alpha_mean_ann_pct"]; digits=2))% — $(ra["stability"])")

    if haskey(result, "brinson_attribution")
        bhb = result["brinson_attribution"]
        println("  BHB: alloc=$(round(bhb["allocation_effect"]*100; digits=3))pp " *
                "select=$(round(bhb["selection_effect"]*100; digits=3))pp " *
                "dominant=$(bhb["dominant_effect"])")
    end

    if haskey(result, "ir_by_factor")
        ir = result["ir_by_factor"]
        println("  IR: $(ir["summary"])")
    end

    println("[performance_attribution] Self-test complete.")
end
