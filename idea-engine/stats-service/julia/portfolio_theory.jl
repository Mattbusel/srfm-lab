# =============================================================================
# portfolio_theory.jl — Advanced Portfolio Theory
# =============================================================================
# Provides:
#   - BlackLitterman       BL model: market equilibrium + IAE views → optimal weights
#   - KellyCriterion       Full Kelly fraction from edge/variance
#   - FractionalKelly      Half-Kelly and custom fractions
#   - WeightComparison     BH weights vs Kelly vs BL comparison + hypothesis generation
#   - MeanVarianceOptimal  Unconstrained Markowitz optimisation
#   - run_portfolio_theory Top-level driver
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, JSON3
# =============================================================================

module PortfolioTheory

using Statistics
using LinearAlgebra
using JSON3

export BlackLitterman, KellyCriterion, FractionalKelly
export WeightComparison, MeanVarianceOptimal, run_portfolio_theory

# ── Utility ───────────────────────────────────────────────────────────────────

"""Clip values to [lo, hi] element-wise."""
_clip(v::Vector{Float64}, lo::Float64, hi::Float64) = clamp.(v, lo, hi)

"""Soft-max projection onto probability simplex (weights sum to 1, ≥ 0)."""
function _project_simplex(w::Vector{Float64})
    n  = length(w)
    u  = sort(w; rev=true)
    cs = cumsum(u)
    ρ  = findlast(u .> (cs .- 1.0) ./ collect(1:n))
    λ  = (cs[ρ] - 1.0) / ρ
    max.(w .- λ, 0.0)
end

"""Add Tikhonov regularisation to a matrix to ensure positive definiteness."""
function _regularise(Σ::Matrix{Float64}; ε::Float64=1e-6)
    Σ + ε * I
end

# ── Mean-Variance Optimisation ────────────────────────────────────────────────

"""
Compute Markowitz minimum-variance and tangency portfolio weights.

# Arguments
- `mu`       : expected return vector (n)
- `Sigma`    : covariance matrix (n × n)
- `rf`       : risk-free rate (annualised scalar)

# Returns
NamedTuple: (min_var_weights, tangency_weights, efficient_frontier)
"""
function MeanVarianceOptimal(
    mu::Vector{Float64},
    Sigma::Matrix{Float64};
    rf::Float64 = 0.0,
    n_frontier::Int = 50
)
    n    = length(mu)
    n == size(Sigma, 1) || error("mu and Sigma size mismatch")
    Σ_reg = _regularise(Sigma)

    # Minimum variance portfolio (long-only via projected gradient)
    ones_n  = ones(n)
    Σ_inv   = inv(Σ_reg)
    w_mv_raw = Σ_inv * ones_n
    w_mv     = w_mv_raw ./ sum(w_mv_raw)
    w_mv     = max.(w_mv, 0.0)
    w_mv     = w_mv ./ sum(w_mv)

    # Tangency portfolio (max Sharpe)
    excess_mu = mu .- rf
    w_tang_raw = Σ_inv * excess_mu
    # Long-only: project onto simplex
    w_tang = _project_simplex(w_tang_raw ./ max(sum(abs.(w_tang_raw)), 1e-12))

    # Efficient frontier: convex combinations
    frontier = map(1:n_frontier) do i
        α   = (i - 1) / (n_frontier - 1)
        w   = α .* w_tang .+ (1.0 - α) .* w_mv
        w   = _project_simplex(w)
        ret = dot(w, mu)
        vol = sqrt(dot(w, Sigma * w))
        (ret=ret, vol=vol, sharpe=(vol>1e-12 ? (ret-rf)/vol : 0.0), weights=w)
    end

    (
        min_var_weights  = w_mv,
        tangency_weights = w_tang,
        min_var_return   = dot(w_mv, mu),
        min_var_vol      = sqrt(max(dot(w_mv, Sigma * w_mv), 0.0)),
        tangency_return  = dot(w_tang, mu),
        tangency_vol     = sqrt(max(dot(w_tang, Sigma * w_tang), 0.0)),
        tangency_sharpe  = begin
            tv = sqrt(max(dot(w_tang, Sigma * w_tang), 0.0))
            tv > 1e-12 ? (dot(w_tang, mu) - rf) / tv : 0.0
        end,
        efficient_frontier = frontier
    )
end

# ── Black-Litterman Model ─────────────────────────────────────────────────────

"""
Black-Litterman model.

Combines CAPM equilibrium returns (Π) with investor views (Q, P) to produce
BL posterior expected returns (μ_BL) and optimal portfolio weights.

Model:
  r ~ N(μ, Σ)
  Equilibrium: Π = δ · Σ · w_mkt
  Views:       P · r = Q + ε,  ε ~ N(0, Ω)
  Posterior:   μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]

# Arguments
- `Sigma`      : n × n covariance matrix of asset returns
- `w_mkt`      : market-cap weights (length n, sum to 1)
- `view_matrix`: k × n matrix P (k views, n assets)
- `view_returns`: k-vector Q of view returns
- `view_conf`  : k-vector of view confidences ∈ (0,1]; scales Ω
- `delta`      : risk aversion parameter (default 2.5)
- `tau`        : scaling parameter for prior uncertainty (default 0.05)
- `asset_names`: optional vector of asset name strings
- `rf`         : risk-free rate

# Returns
NamedTuple: (mu_bl, w_bl_optimal, equilibrium_returns, posterior_cov)
"""
function BlackLitterman(
    Sigma::Matrix{Float64},
    w_mkt::Vector{Float64};
    view_matrix::Matrix{Float64}  = Matrix{Float64}(undef, 0, 0),
    view_returns::Vector{Float64} = Float64[],
    view_conf::Vector{Float64}    = Float64[],
    delta::Float64  = 2.5,
    tau::Float64    = 0.05,
    rf::Float64     = 0.0,
    asset_names::Vector{String} = String[]
)
    n = length(w_mkt)
    n == size(Sigma, 1) || error("w_mkt and Sigma size mismatch")
    abs(sum(w_mkt) - 1.0) < 0.01 || @warn "w_mkt does not sum to 1 (sum=$(sum(w_mkt)))"

    Σ_reg = _regularise(Sigma)
    names = isempty(asset_names) ? ["Asset_$i" for i in 1:n] : asset_names

    # CAPM equilibrium returns: Π = δ · Σ · w_mkt
    Pi = delta .* (Σ_reg * w_mkt)

    # Prior covariance on mean: τΣ
    tau_Sigma     = tau .* Σ_reg
    tau_Sigma_inv = inv(_regularise(tau_Sigma))

    has_views = !isempty(view_matrix) && size(view_matrix, 1) > 0
    k = has_views ? size(view_matrix, 1) : 0

    mu_bl     = copy(Pi)
    post_cov  = copy(tau_Sigma)

    if has_views
        k == length(view_returns) || error("view_matrix rows must match view_returns length")
        P   = view_matrix
        Q   = view_returns
        # Ω = diag(confidence-scaled residual variances)
        base_var = [dot(P[i, :], tau_Sigma * P[i, :]) for i in 1:k]
        conf      = isempty(view_conf) ? fill(0.5, k) : view_conf
        omega_diag = [base_var[i] * (1.0 - conf[i]) / max(conf[i], 0.01) for i in 1:k]
        Omega_inv  = Diagonal(1.0 ./ max.(omega_diag, 1e-10))

        # BL posterior
        M_inv  = _regularise(tau_Sigma_inv + P' * Omega_inv * P)
        M      = inv(M_inv)
        mu_bl  = M * (tau_Sigma_inv * Pi + P' * Omega_inv * Q)
        post_cov = M
    end

    # Optimal BL weights: w* = (δΣ)⁻¹ μ_BL → project onto simplex
    delta_Sigma_inv = inv(_regularise(delta .* Σ_reg))
    w_bl_raw  = delta_Sigma_inv * (mu_bl .- rf)
    w_bl      = _project_simplex(w_bl_raw ./ max(sum(abs.(w_bl_raw)), 1e-12))

    # Portfolio stats
    bl_return = dot(w_bl, mu_bl)
    bl_vol    = sqrt(max(dot(w_bl, Sigma * w_bl), 0.0))
    bl_sharpe = bl_vol > 1e-12 ? (bl_return - rf) / bl_vol : 0.0

    # Per-asset views attribution
    view_attributions = has_views ?
        map(1:k) do i
            (view_direction = view_returns[i] > 0 ? "bullish" : "bearish",
             view_return    = view_returns[i],
             confidence     = isempty(view_conf) ? 0.5 : view_conf[i],
             affected_assets = names[abs.(view_matrix[i, :]) .> 1e-6])
        end : []

    (
        equilibrium_returns  = Pi,
        bl_expected_returns  = mu_bl,
        bl_weights           = w_bl,
        mkt_weights          = w_mkt,
        posterior_cov        = post_cov,
        bl_portfolio_return  = bl_return,
        bl_portfolio_vol     = bl_vol,
        bl_sharpe            = bl_sharpe,
        n_views              = k,
        view_attributions    = view_attributions,
        asset_names          = names
    )
end

# ── Kelly Criterion ───────────────────────────────────────────────────────────

"""
Compute Kelly-optimal position size fraction for a single asset/strategy.

Full Kelly: f* = (μ - rf) / σ²
This is derived from maximising E[log(wealth)].

Also computes fractional Kelly variants for robustness.

# Arguments
- `mean_return` : expected return per period (e.g. per trade)
- `variance`    : return variance per period
- `win_rate`    : optional win rate (for discrete Kelly)
- `avg_win`     : optional average win size
- `avg_loss`    : optional average loss size (positive number)
- `rf`          : risk-free rate

# Returns
NamedTuple: full Kelly fraction, half-Kelly, quarter-Kelly, and continuous variant
"""
function KellyCriterion(
    mean_return::Float64,
    variance::Float64;
    win_rate::Float64    = NaN,
    avg_win::Float64     = NaN,
    avg_loss::Float64    = NaN,
    rf::Float64          = 0.0,
    max_fraction::Float64 = 1.0
)
    variance > 0 || error("variance must be positive")

    # Continuous Kelly (Gaussian returns)
    kelly_continuous = (mean_return - rf) / variance

    # Discrete Kelly (win/loss formulation)
    kelly_discrete = NaN
    if !isnan(win_rate) && !isnan(avg_win) && !isnan(avg_loss)
        avg_loss > 0 || error("avg_loss must be positive")
        # f* = (p * b - q) / b  where b = avg_win/avg_loss, p = win_rate, q = 1-p
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly_discrete = (win_rate * b - q) / b
    end

    # Use best estimate
    kelly_full = isnan(kelly_discrete) ? kelly_continuous : kelly_discrete
    kelly_full = clamp(kelly_full, -1.0, max_fraction)

    # Fractional Kelly variants
    half_kelly    = kelly_full * 0.5
    quarter_kelly = kelly_full * 0.25
    three_qtr     = kelly_full * 0.75

    # Growth rates at each fraction (Gaussian approximation)
    growth_rate(f) = f * (mean_return - rf) - 0.5 * f^2 * variance

    (
        kelly_full         = kelly_full,
        half_kelly         = half_kelly,
        quarter_kelly      = quarter_kelly,
        three_quarter_kelly = three_qtr,
        kelly_continuous   = kelly_continuous,
        kelly_discrete     = kelly_discrete,
        growth_rate_full   = growth_rate(kelly_full),
        growth_rate_half   = growth_rate(half_kelly),
        growth_rate_quarter = growth_rate(quarter_kelly),
        interpretation = if kelly_full > 0.5
            "Aggressive sizing recommended (edge is strong); consider half-Kelly for stability"
        elseif kelly_full > 0.1
            "Moderate sizing; half-Kelly is reasonable"
        elseif kelly_full > 0.0
            "Small positive edge; consider quarter-Kelly or fixed small fraction"
        else
            "Negative edge: Kelly = $(round(kelly_full*100; digits=1))%; DO NOT TRADE"
        end
    )
end

"""
Multi-asset Kelly (vector Kelly / Markowitz-Kelly equivalence).

Optimal fractions: f* = Σ⁻¹ μ  (in log-utility maximisation with Gaussian returns)
"""
function FractionalKelly(
    mu::Vector{Float64},
    Sigma::Matrix{Float64};
    fraction::Float64   = 0.5,
    rf::Float64         = 0.0,
    asset_names::Vector{String} = String[]
)
    n = length(mu)
    Σ_reg = _regularise(Sigma)
    Σ_inv = inv(Σ_reg)

    excess_mu   = mu .- rf
    kelly_full  = Σ_inv * excess_mu
    kelly_frac  = kelly_full .* fraction

    # Normalise to ≤ 100% total exposure
    total_exp   = sum(abs.(kelly_frac))
    if total_exp > 1.0
        kelly_frac = kelly_frac ./ total_exp
    end

    names = isempty(asset_names) ? ["Asset_$i" for i in 1:n] : asset_names

    kelly_per_asset = map(enumerate(names)) do (i, nm)
        (asset=nm, kelly_full=kelly_full[i], kelly_frac=kelly_frac[i])
    end

    (
        kelly_full_weights = kelly_full,
        kelly_frac_weights = kelly_frac,
        fraction           = fraction,
        total_exposure     = sum(abs.(kelly_frac)),
        per_asset          = kelly_per_asset,
        asset_names        = names
    )
end

# ── Weight Comparison & Hypothesis Generation ─────────────────────────────────

"""
Compare current BH weights against Kelly-optimal and BL-optimal weights.
Generate actionable hypotheses for underweight/overweight positions.

# Arguments
- `bh_weights`   : current BH strategy weights per asset (sums to 1)
- `kelly_weights`: fractional Kelly weights
- `bl_weights`   : Black-Litterman optimal weights
- `asset_names`  : asset labels
- `threshold`    : minimum deviation to flag as hypothesis (default 0.05 = 5 pp)

# Returns
NamedTuple: deviations, hypotheses, summary
"""
function WeightComparison(
    bh_weights::Vector{Float64},
    kelly_weights::Vector{Float64},
    bl_weights::Vector{Float64};
    asset_names::Vector{String} = String[],
    threshold::Float64 = 0.05
)
    n = length(bh_weights)
    n == length(kelly_weights) == length(bl_weights) ||
        error("All weight vectors must have equal length")

    names = isempty(asset_names) ? ["Asset_$i" for i in 1:n] : asset_names

    hypotheses = String[]
    deviations = map(enumerate(names)) do (i, nm)
        bh  = bh_weights[i]
        kel = kelly_weights[i]
        bl  = bl_weights[i]

        kelly_dev = kel - bh
        bl_dev    = bl - bh

        # Generate hypothesis if large deviation
        if abs(kelly_dev) > threshold
            dir    = kelly_dev > 0 ? "underweight" : "overweight"
            pct_bh  = round(bh   * 100; digits=1)
            pct_kel = round(kel  * 100; digits=1)
            push!(hypotheses,
                "$nm is $(round(abs(kelly_dev)*100; digits=1))pp $dir vs Kelly-optimal " *
                "(BH=$(pct_bh)%, Kelly=$(pct_kel)%): " *
                "consider $(kelly_dev > 0 ? "increasing" : "reducing") allocation")
        end
        if abs(bl_dev) > threshold
            dir    = bl_dev > 0 ? "underweight" : "overweight"
            pct_bh  = round(bh * 100; digits=1)
            pct_bl  = round(bl * 100; digits=1)
            push!(hypotheses,
                "$nm is $(round(abs(bl_dev)*100; digits=1))pp $dir vs BL-optimal " *
                "(BH=$(pct_bh)%, BL=$(pct_bl)%): view-driven rebalancing suggested")
        end

        (
            asset        = nm,
            bh_weight    = bh,
            kelly_weight = kel,
            bl_weight    = bl,
            kelly_deviation = kelly_dev,
            bl_deviation    = bl_dev,
            flagged      = abs(kelly_dev) > threshold || abs(bl_dev) > threshold
        )
    end

    # Portfolio-level stats
    tracking_error_kelly = sqrt(sum((bh_weights .- kelly_weights).^2))
    tracking_error_bl    = sqrt(sum((bh_weights .- bl_weights).^2))

    (
        deviations          = deviations,
        hypotheses          = hypotheses,
        n_flagged           = count(d -> d.flagged, deviations),
        tracking_error_kelly = tracking_error_kelly,
        tracking_error_bl   = tracking_error_bl,
        summary = "$(length(hypotheses)) weight-deviation hypotheses generated; " *
                  "TE vs Kelly=$(round(tracking_error_kelly*100; digits=1))pp"
    )
end

# ── Top-level driver ──────────────────────────────────────────────────────────

"""
Run full portfolio theory pipeline.

# Arguments
- `asset_names`    : e.g. ["BTC", "ETH", "SOL", "BNB"]
- `mu`             : expected returns per asset (annualised)
- `Sigma`          : return covariance matrix
- `bh_weights`     : current strategy weights
- `iae_confidence` : IAE hypothesis confidence per asset → translated to BL views

Writes `portfolio_theory_results.json` to `\$STATS_OUTPUT_DIR`.
"""
function run_portfolio_theory(
    asset_names::Vector{String},
    mu::Vector{Float64},
    Sigma::Matrix{Float64},
    bh_weights::Vector{Float64};
    iae_confidence::Vector{Float64} = fill(0.5, length(asset_names)),
    rf::Float64     = 0.02,
    delta::Float64  = 2.5,
    output_dir::String = get(ENV, "STATS_OUTPUT_DIR",
                              joinpath(@__DIR__, "..", "output"))
)
    n = length(asset_names)

    println("[portfolio] Running Markowitz mean-variance optimisation...")
    mv = MeanVarianceOptimal(mu, Sigma; rf=rf)

    println("[portfolio] Building Black-Litterman views from IAE confidence scores...")
    # View: each asset return deviates from equilibrium by (confidence - 0.5) * 2 * std
    asset_stds   = sqrt.(diag(Sigma))
    view_matrix  = Matrix{Float64}(I, n, n)   # k=n views: one per asset
    view_returns = mu .+ (iae_confidence .- 0.5) .* 2.0 .* asset_stds
    view_conf    = clamp.(iae_confidence, 0.01, 0.99)
    w_mkt        = bh_weights ./ max(sum(bh_weights), 1e-12)

    bl = BlackLitterman(Sigma, w_mkt;
        view_matrix  = view_matrix,
        view_returns = view_returns,
        view_conf    = view_conf,
        delta        = delta,
        tau          = 0.05,
        rf           = rf,
        asset_names  = asset_names
    )

    println("[portfolio] Computing Kelly criterion per asset...")
    fk = FractionalKelly(mu, Sigma; fraction=0.5, rf=rf, asset_names=asset_names)

    println("[portfolio] Comparing weights and generating hypotheses...")
    comp = WeightComparison(bh_weights, fk.kelly_frac_weights, bl.bl_weights;
        asset_names=asset_names, threshold=0.05)

    result = Dict(
        "assets"           => asset_names,
        "markowitz" => Dict(
            "min_var_weights"   => mv.min_var_weights,
            "tangency_weights"  => mv.tangency_weights,
            "tangency_sharpe"   => mv.tangency_sharpe,
            "tangency_return"   => mv.tangency_return,
            "tangency_vol"      => mv.tangency_vol
        ),
        "black_litterman" => Dict(
            "equilibrium_returns"  => bl.equilibrium_returns,
            "bl_expected_returns"  => bl.bl_expected_returns,
            "bl_weights"           => bl.bl_weights,
            "bl_sharpe"            => bl.bl_sharpe,
            "n_views"              => bl.n_views
        ),
        "kelly" => Dict(
            "full_weights"   => fk.kelly_full_weights,
            "half_weights"   => fk.kelly_frac_weights,
            "total_exposure" => fk.total_exposure,
            "per_asset"      => map(p -> Dict("asset"=>p.asset, "half_kelly"=>p.kelly_frac), fk.per_asset)
        ),
        "weight_comparison" => Dict(
            "bh_weights"              => bh_weights,
            "deviations"              => map(d -> Dict(
                "asset"           => d.asset,
                "bh"              => d.bh_weight,
                "kelly"           => d.kelly_weight,
                "bl"              => d.bl_weight,
                "kelly_dev"       => d.kelly_deviation,
                "bl_dev"          => d.bl_deviation,
                "flagged"         => d.flagged
            ), comp.deviations),
            "hypotheses"              => comp.hypotheses,
            "n_flagged"               => comp.n_flagged,
            "tracking_error_kelly_pp" => comp.tracking_error_kelly * 100,
            "tracking_error_bl_pp"    => comp.tracking_error_bl * 100,
            "summary"                 => comp.summary
        )
    )

    mkpath(output_dir)
    out_path = joinpath(output_dir, "portfolio_theory_results.json")
    open(out_path, "w") do io
        write(io, JSON3.write(result))
    end
    println("[portfolio] Results written to $out_path")

    result
end

end  # module PortfolioTheory

# ── CLI self-test ─────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    using .PortfolioTheory
    using Statistics, LinearAlgebra

    println("[portfolio_theory] Running self-test...")

    assets  = ["BTC", "ETH", "SOL", "BNB"]
    n       = length(assets)

    # Synthetic covariance (annualised)
    σs  = [0.80, 0.90, 1.10, 0.75]
    ρ   = [1.0  0.85  0.75  0.70;
           0.85 1.0   0.80  0.72;
           0.75 0.80  1.0   0.65;
           0.70 0.72  0.65  1.0]
    Σ   = Diagonal(σs) * ρ * Diagonal(σs)

    mu  = [0.30, 0.35, 0.50, 0.25]
    bh_w = [0.40, 0.30, 0.20, 0.10]
    conf = [0.65, 0.55, 0.70, 0.45]

    result = run_portfolio_theory(assets, mu, Σ, bh_w;
        iae_confidence=conf, rf=0.02, delta=2.5)

    mk = result["markowitz"]
    println("  Tangency portfolio: Sharpe=$(round(mk["tangency_sharpe"]; digits=3))")
    println("  Tangency weights: $(round.(mk["tangency_weights"]; digits=3))")

    bl = result["black_litterman"]
    println("  BL expected returns: $(round.(bl["bl_expected_returns"]; digits=3))")
    println("  BL weights: $(round.(bl["bl_weights"]; digits=3))")

    kl = result["kelly"]
    println("  Half-Kelly weights: $(round.([p["half_kelly"] for p in kl["per_asset"]]; digits=3))")

    wc = result["weight_comparison"]
    println("  Weight hypotheses generated: $(wc["n_flagged"])")
    for h in wc["hypotheses"]
        println("    → $h")
    end

    println("[portfolio_theory] Self-test complete.")
end
