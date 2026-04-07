module RiskAttribution

# Risk and performance attribution: Brinson-Hood-Beebower, Barra-style factor attribution,
# risk decomposition (component VaR, marginal VaR, diversification ratio).
# For production use in the SRFM quant trading system.

using LinearAlgebra
using Statistics
using Random
using Test

export BrinsonResult, brinson_attribution
export FactorAttribution, factor_attribution
export component_var, marginal_var, diversification_ratio

# ---------------------------------------------------------------------------
# Brinson-Hood-Beebower Attribution
# ---------------------------------------------------------------------------

"""
    BrinsonResult

Result of Brinson-Hood-Beebower performance attribution.

Fields:
- `allocation`: total allocation effect across all sectors
- `selection`: total selection effect
- `interaction`: total interaction effect
- `total_active`: total active return (allocation + selection + interaction)
- `by_sector`: Dict mapping sector name to NamedTuple (allocation, selection, interaction)
"""
@kwdef struct BrinsonResult
    allocation::Float64
    selection::Float64
    interaction::Float64
    total_active::Float64
    by_sector::Dict{String, NamedTuple}
end

"""
    brinson_attribution(port_weights, bench_weights, port_returns, bench_returns, sector_map)
        -> BrinsonResult

Compute Brinson-Hood-Beebower single-period performance attribution.

Decomposition:
- Allocation effect:  (w_p - w_b) * r_b
- Selection effect:   w_b * (r_p - r_b)
- Interaction effect: (w_p - w_b) * (r_p - r_b)

where weights and returns are aggregated to the sector level.

# Arguments
- `port_weights`: Dict{String, Float64} or Vector asset-level portfolio weights
- `bench_weights`: asset-level benchmark weights (same type/length as port_weights)
- `port_returns`: asset-level portfolio returns
- `bench_returns`: asset-level benchmark returns
- `sector_map`: Dict{Int, String} mapping asset index to sector name

All weight vectors must sum to approximately 1.0.

# Returns
BrinsonResult with attribution effects by sector and in aggregate.
"""
function brinson_attribution(port_weights::Vector{Float64},
                              bench_weights::Vector{Float64},
                              port_returns::Vector{Float64},
                              bench_returns::Vector{Float64},
                              sector_map::Dict{Int, String})::BrinsonResult
    n = length(port_weights)
    @assert length(bench_weights) == n "Weight vectors must be same length"
    @assert length(port_returns) == n "Return vectors must be same length"
    @assert length(bench_returns) == n "Return vectors must be same length"

    # Identify unique sectors
    sectors = unique(values(sector_map))

    by_sector = Dict{String, NamedTuple}()
    total_alloc = 0.0
    total_sel = 0.0
    total_inter = 0.0

    for sector in sectors
        # Indices belonging to this sector
        idx = [i for i in 1:n if get(sector_map, i, "") == sector]
        if isempty(idx)
            continue
        end

        # Sector-level weights and returns
        wp_s = sum(port_weights[idx])
        wb_s = sum(bench_weights[idx])

        # Weighted average sector returns
        rp_s = if wp_s > 1e-12
            sum(port_weights[idx] .* port_returns[idx]) / wp_s
        else
            0.0
        end

        rb_s = if wb_s > 1e-12
            sum(bench_weights[idx] .* bench_returns[idx]) / wb_s
        else
            0.0
        end

        # BHB effects
        alloc_s = (wp_s - wb_s) * rb_s
        sel_s = wb_s * (rp_s - rb_s)
        inter_s = (wp_s - wb_s) * (rp_s - rb_s)

        by_sector[sector] = (
            allocation=alloc_s,
            selection=sel_s,
            interaction=inter_s,
            port_weight=wp_s,
            bench_weight=wb_s,
            port_return=rp_s,
            bench_return=rb_s
        )

        total_alloc += alloc_s
        total_sel += sel_s
        total_inter += inter_s
    end

    total_active = total_alloc + total_sel + total_inter

    return BrinsonResult(
        allocation=total_alloc,
        selection=total_sel,
        interaction=total_inter,
        total_active=total_active,
        by_sector=by_sector
    )
end

"""
    brinson_attribution(port_weights, bench_weights, port_returns, bench_returns, sector_map)

Overload accepting sector_map as Dict{String, Vector{Int}} (sector name to asset indices).
"""
function brinson_attribution(port_weights::Vector{Float64},
                              bench_weights::Vector{Float64},
                              port_returns::Vector{Float64},
                              bench_returns::Vector{Float64},
                              sector_map::Dict{String, Vector{Int}})::BrinsonResult
    # Convert to Dict{Int, String}
    n = length(port_weights)
    int_map = Dict{Int, String}()
    for (sector, idxs) in sector_map
        for i in idxs
            int_map[i] = sector
        end
    end
    return brinson_attribution(port_weights, bench_weights, port_returns, bench_returns, int_map)
end

# ---------------------------------------------------------------------------
# Factor Attribution (Barra-style)
# ---------------------------------------------------------------------------

"""
    FactorAttribution

Result of factor-based performance attribution.

Fields:
- `factor_contributions`: Vector of factor contributions (length = n_factors)
- `specific_return`: residual return not explained by factors
- `total_return`: total portfolio return
- `r_squared`: variance explained by factor model
- `factor_names`: optional vector of factor names
"""
@kwdef struct FactorAttribution
    factor_contributions::Vector{Float64}
    specific_return::Float64
    total_return::Float64
    r_squared::Float64
    factor_names::Vector{String}
end

"""
    factor_attribution(returns, factor_exposures, factor_returns) -> FactorAttribution

Compute Barra-style factor attribution for a portfolio.

Portfolio return is decomposed as:
    r_p = sum_k (beta_k * f_k) + specific_return
where beta_k = sum_i w_i * B_ik is the portfolio's exposure to factor k.

# Arguments
- `returns`: portfolio returns vector (T,) or scalar single-period return
- `factor_exposures`: N x K matrix of asset factor loadings (N assets, K factors)
- `factor_returns`: K x T matrix of factor returns or K-vector for single period

If returns is a scalar and factor_returns is a K-vector, single-period attribution.
If returns is a T-vector and factor_returns is K x T, multi-period (average) attribution.

# Returns
FactorAttribution struct.
"""
function factor_attribution(portfolio_return::Float64,
                             factor_exposures::Vector{Float64},
                             factor_returns::Vector{Float64})::FactorAttribution
    # Single-period, portfolio already has exposures as a vector
    K = length(factor_exposures)
    @assert length(factor_returns) == K "factor_exposures and factor_returns must have equal length"

    factor_contribs = factor_exposures .* factor_returns
    factor_total = sum(factor_contribs)
    specific = portfolio_return - factor_total

    # R-squared: variance explained
    total_var = portfolio_return^2
    r2 = total_var > 1e-12 ? min(factor_total^2 / total_var, 1.0) : 0.0

    return FactorAttribution(
        factor_contributions=factor_contribs,
        specific_return=specific,
        total_return=portfolio_return,
        r_squared=r2,
        factor_names=["Factor_$k" for k in 1:K]
    )
end

function factor_attribution(portfolio_return::Float64,
                             factor_exposures::Vector{Float64},
                             factor_returns::Vector{Float64},
                             factor_names::Vector{String})::FactorAttribution
    result = factor_attribution(portfolio_return, factor_exposures, factor_returns)
    return FactorAttribution(
        factor_contributions=result.factor_contributions,
        specific_return=result.specific_return,
        total_return=result.total_return,
        r_squared=result.r_squared,
        factor_names=factor_names
    )
end

"""
    factor_attribution(weights, asset_returns, factor_exposures, factor_returns) -> FactorAttribution

Full multi-asset factor attribution.

# Arguments
- `weights`: N-vector of portfolio weights
- `asset_returns`: N-vector of asset returns for the period
- `factor_exposures`: N x K matrix of asset factor loadings
- `factor_returns`: K-vector of factor returns

# Returns
FactorAttribution struct with K factor contributions and residual.
"""
function factor_attribution(weights::Vector{Float64},
                             asset_returns::Vector{Float64},
                             factor_exposures::Matrix{Float64},
                             factor_returns::Vector{Float64})::FactorAttribution
    N, K = size(factor_exposures)
    @assert length(weights) == N "weights and factor_exposures row count must match"
    @assert length(asset_returns) == N "asset_returns and factor_exposures row count must match"
    @assert length(factor_returns) == K "factor_returns must match K factors"

    port_return = dot(weights, asset_returns)
    port_exposures = vec(weights' * factor_exposures)  -- K-vector of portfolio factor exposures

    factor_contribs = port_exposures .* factor_returns
    factor_total = sum(factor_contribs)
    specific = port_return - factor_total

    -- Compute R^2 using cross-sectional regression residuals
    predicted_returns = factor_exposures * factor_returns
    resid = asset_returns .- predicted_returns
    ss_tot = var(asset_returns) * (N - 1)
    ss_res = var(resid) * (N - 1)
    r2 = ss_tot > 1e-12 ? max(0.0, 1.0 - ss_res / ss_tot) : 0.0

    return FactorAttribution(
        factor_contributions=factor_contribs,
        specific_return=specific,
        total_return=port_return,
        r_squared=r2,
        factor_names=["Factor_$k" for k in 1:K]
    )
end

# ---------------------------------------------------------------------------
# Risk Decomposition
# ---------------------------------------------------------------------------

"""
    _portfolio_vol(weights, cov_matrix) -> Float64

Compute annualized portfolio volatility.
"""
function _portfolio_vol(weights::Vector{Float64}, cov_matrix::Matrix{Float64})::Float64
    pvar = dot(weights, cov_matrix * weights)
    return sqrt(max(pvar, 0.0))
end

"""
    _normal_quantile(p) -> Float64

Approximate quantile of standard normal distribution at probability p.
Uses rational Chebyshev approximation (Beasley-Springer-Moro algorithm).
"""
function _normal_quantile(p::Float64)::Float64
    @assert 0.0 < p < 1.0 "Probability must be strictly between 0 and 1"
    if p < 0.5
        return -_normal_quantile(1.0 - p)
    end
    q = p - 0.5
    r = q^2
    num = (((((2.515517 + r * 0.802853) + r^2 * 0.010328)))
    denom = (((((1.0 + r * 1.432788) + r^2 * 0.189269) + r^3 * 0.001308)))
    return q + q * (num / max(denom, 1e-12))
end

-- Fallback simple quantile
function _z_quantile(confidence::Float64)::Float64
    # For VaR: confidence = 0.99 -> z = 2.326, 0.95 -> z = 1.645
    if confidence >= 0.999
        return 3.090
    elseif confidence >= 0.99
        return 2.326
    elseif confidence >= 0.975
        return 1.960
    elseif confidence >= 0.95
        return 1.645
    elseif confidence >= 0.90
        return 1.282
    else
        # Linear interpolation approximation
        return 1.282 * (confidence - 0.5) / 0.4
    end
end

"""
    component_var(weights, cov_matrix, confidence=0.99) -> Vector{Float64}

Compute component VaR for each asset in the portfolio.

Component VaR_i = rho_i * VaR_i (correlation-weighted individual VaR).
Equivalently: CVaR_i = w_i * (Sigma * w)_i / sigma_p * z_alpha

The sum of component VaRs equals the total portfolio VaR.

# Arguments
- `weights`: N-vector of portfolio weights
- `cov_matrix`: N x N covariance matrix
- `confidence`: VaR confidence level (default 0.99)

# Returns
N-vector of component VaRs (positive values represent risk contributions).
"""
function component_var(weights::Vector{Float64},
                        cov_matrix::Matrix{Float64},
                        confidence::Float64=0.99)::Vector{Float64}
    N = length(weights)
    @assert size(cov_matrix) == (N, N) "Covariance matrix dimensions must match weights"

    z = _z_quantile(confidence)
    Sigma_w = cov_matrix * weights
    port_vol = sqrt(max(dot(weights, Sigma_w), 1e-12))

    -- Component VaR_i = z * w_i * (Sigma*w)_i / port_vol
    cvars = z .* weights .* Sigma_w ./ port_vol
    return cvars
end

"""
    marginal_var(weights, cov_matrix, confidence=0.99) -> Vector{Float64}

Compute marginal VaR for each asset.

Marginal VaR_i = dVaR/dw_i = z * (Sigma * w)_i / sigma_p

Represents the change in portfolio VaR per unit increase in asset i's weight.

# Arguments
- `weights`: N-vector of portfolio weights
- `cov_matrix`: N x N covariance matrix
- `confidence`: VaR confidence level (default 0.99)

# Returns
N-vector of marginal VaRs.
"""
function marginal_var(weights::Vector{Float64},
                       cov_matrix::Matrix{Float64},
                       confidence::Float64=0.99)::Vector{Float64}
    N = length(weights)
    @assert size(cov_matrix) == (N, N) "Covariance matrix dimensions must match weights"

    z = _z_quantile(confidence)
    Sigma_w = cov_matrix * weights
    port_vol = sqrt(max(dot(weights, Sigma_w), 1e-12))

    return z .* Sigma_w ./ port_vol
end

"""
    diversification_ratio(weights, cov_matrix) -> Float64

Compute the Diversification Ratio (DR) of a portfolio.

DR = (sum_i w_i * sigma_i) / sqrt(w' * Sigma * w)

A DR of 1.0 means no diversification benefit (all assets perfectly correlated).
Higher values indicate more diversification.

# Arguments
- `weights`: N-vector of portfolio weights
- `cov_matrix`: N x N covariance matrix

# Returns
Scalar diversification ratio >= 1.0.
"""
function diversification_ratio(weights::Vector{Float64},
                                 cov_matrix::Matrix{Float64})::Float64
    N = length(weights)
    @assert size(cov_matrix) == (N, N) "Covariance matrix dimensions must match weights"

    individual_vols = sqrt.(max.(diag(cov_matrix), 0.0))
    weighted_avg_vol = dot(weights, individual_vols)
    port_vol = _portfolio_vol(weights, cov_matrix)

    if port_vol < 1e-12
        return 1.0
    end

    return weighted_avg_vol / port_vol
end

"""
    portfolio_var(weights, cov_matrix, confidence=0.99) -> Float64

Total parametric portfolio VaR at given confidence level.

VaR = z_alpha * sqrt(w' * Sigma * w)

# Arguments
- `weights`: N-vector of portfolio weights
- `cov_matrix`: N x N covariance matrix
- `confidence`: confidence level (default 0.99)

# Returns
Scalar VaR (positive number representing loss).
"""
function portfolio_var(weights::Vector{Float64},
                        cov_matrix::Matrix{Float64},
                        confidence::Float64=0.99)::Float64
    z = _z_quantile(confidence)
    port_vol = _portfolio_vol(weights, cov_matrix)
    return z * port_vol
end

# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

function run_tests()
    @testset "RiskAttribution Tests" begin

        rng = MersenneTwister(7)
        N = 5

        -- Sector map: assets 1,2 in Tech; 3,4 in Finance; 5 in Energy
        sector_map = Dict{Int, String}(
            1 => "Tech", 2 => "Tech",
            3 => "Finance", 4 => "Finance",
            5 => "Energy"
        )

        -- Simple covariance matrix
        vols = [0.20, 0.25, 0.15, 0.18, 0.22]
        rho = 0.3 .* ones(N, N) + 0.7 .* I(N)
        cov_mat = Diagonal(vols) * rho * Diagonal(vols)

        -- -- Brinson-Hood-Beebower --
        @testset "brinson_attribution basic" begin
            pw = [0.25, 0.25, 0.20, 0.15, 0.15]
            bw = [0.20, 0.20, 0.25, 0.20, 0.15]
            pr = [0.05, 0.06, 0.03, 0.04, 0.07]
            br = [0.04, 0.05, 0.03, 0.03, 0.06]

            result = brinson_attribution(pw, bw, pr, br, sector_map)
            @test isa(result, BrinsonResult)
            @test isapprox(result.total_active,
                           result.allocation + result.selection + result.interaction, atol=1e-10)
            @test haskey(result.by_sector, "Tech")
            @test haskey(result.by_sector, "Finance")
            @test haskey(result.by_sector, "Energy")
        end

        @testset "brinson_attribution zero active weight" begin
            pw = [0.20, 0.20, 0.25, 0.20, 0.15]  -- same as benchmark
            bw = [0.20, 0.20, 0.25, 0.20, 0.15]
            pr = [0.05, 0.05, 0.03, 0.03, 0.06]
            br = [0.05, 0.05, 0.03, 0.03, 0.06]
            result = brinson_attribution(pw, bw, pr, br, sector_map)
            @test isapprox(result.allocation, 0.0, atol=1e-10)
            @test isapprox(result.selection, 0.0, atol=1e-10)
            @test isapprox(result.interaction, 0.0, atol=1e-10)
        end

        @testset "brinson_attribution sector_dict overload" begin
            pw = [0.25, 0.25, 0.20, 0.15, 0.15]
            bw = [0.20, 0.20, 0.25, 0.20, 0.15]
            pr = [0.05, 0.06, 0.03, 0.04, 0.07]
            br = [0.04, 0.05, 0.03, 0.03, 0.06]
            sector_map2 = Dict{String, Vector{Int}}(
                "Tech" => [1, 2],
                "Finance" => [3, 4],
                "Energy" => [5]
            )
            result = brinson_attribution(pw, bw, pr, br, sector_map2)
            @test isa(result, BrinsonResult)
            @test haskey(result.by_sector, "Tech")
        end

        @testset "brinson_attribution allocation effect" begin
            -- Pure allocation: same returns within sector, different weights
            pw = [0.30, 0.20, 0.20, 0.15, 0.15]
            bw = [0.20, 0.20, 0.25, 0.20, 0.15]
            pr = [0.04, 0.04, 0.03, 0.03, 0.06]  -- same as br within sector
            br = [0.04, 0.04, 0.03, 0.03, 0.06]
            result = brinson_attribution(pw, bw, pr, br, sector_map)
            @test isapprox(result.selection, 0.0, atol=1e-10)
            @test isapprox(result.interaction, 0.0, atol=1e-10)
        end

        @testset "brinson_attribution selection effect" begin
            -- Pure selection: same sector weights, different asset returns
            pw = [0.20, 0.20, 0.25, 0.20, 0.15]  -- same as benchmark
            bw = [0.20, 0.20, 0.25, 0.20, 0.15]
            pr = [0.06, 0.07, 0.04, 0.05, 0.07]  -- better stock picks
            br = [0.04, 0.05, 0.03, 0.03, 0.06]
            result = brinson_attribution(pw, bw, pr, br, sector_map)
            @test isapprox(result.allocation, 0.0, atol=1e-10)
            @test isapprox(result.interaction, 0.0, atol=1e-10)
            @test result.selection > 0.0
        end

        -- -- Factor Attribution --
        @testset "factor_attribution single_period" begin
            port_ret = 0.05
            exposures = [0.8, 0.3, 1.2]
            f_rets = [0.02, 0.01, 0.03]
            result = factor_attribution(port_ret, exposures, f_rets)
            @test length(result.factor_contributions) == 3
            @test isapprox(result.total_return, port_ret, atol=1e-10)
            @test isapprox(result.factor_contributions[1], 0.8 * 0.02, atol=1e-10)
            factor_sum = sum(result.factor_contributions)
            @test isapprox(result.specific_return, port_ret - factor_sum, atol=1e-10)
        end

        @testset "factor_attribution multi_asset" begin
            weights = [0.2, 0.3, 0.25, 0.15, 0.1]
            asset_rets = [0.05, 0.04, 0.03, 0.06, 0.02]
            B = randn(rng, 5, 3) .* 0.5
            f_rets = [0.02, 0.01, 0.03]
            result = factor_attribution(weights, asset_rets, B, f_rets)
            @test length(result.factor_contributions) == 3
            @test isapprox(result.total_return, dot(weights, asset_rets), atol=1e-10)
            @test 0.0 <= result.r_squared <= 1.0
        end

        @testset "factor_attribution specific_return" begin
            weights = ones(N) ./ N
            asset_rets = [0.05, 0.04, 0.03, 0.06, 0.02]
            B = zeros(N, 2)  -- zero exposures
            f_rets = [0.02, 0.01]
            result = factor_attribution(weights, asset_rets, B, f_rets)
            @test isapprox(result.specific_return, dot(weights, asset_rets), atol=1e-10)
            @test all(result.factor_contributions .== 0.0)
        end

        @testset "factor_attribution names" begin
            port_ret = 0.03
            exp = [0.5, 0.7]
            f_rets = [0.01, 0.02]
            names = ["Market", "Value"]
            result = factor_attribution(port_ret, exp, f_rets, names)
            @test result.factor_names == ["Market", "Value"]
        end

        -- -- Component VaR --
        @testset "component_var sum_equals_total" begin
            w = ones(N) ./ N
            cvars = component_var(w, cov_mat, 0.99)
            total_cvar = sum(cvars)
            total_var_val = portfolio_var(w, cov_mat, 0.99)
            @test isapprox(total_cvar, total_var_val, atol=1e-8)
        end

        @testset "component_var positive_weights" begin
            w = [0.3, 0.2, 0.25, 0.15, 0.1]
            cvars = component_var(w, cov_mat, 0.99)
            @test length(cvars) == N
            @test all(isfinite.(cvars))
        end

        @testset "component_var_95" begin
            w = ones(N) ./ N
            cvars_99 = component_var(w, cov_mat, 0.99)
            cvars_95 = component_var(w, cov_mat, 0.95)
            -- 99% VaR should exceed 95% VaR
            @test sum(cvars_99) > sum(cvars_95)
        end

        @testset "component_var identity_cov" begin
            w = ones(N) ./ N
            I_mat = Matrix{Float64}(I(N))
            cvars = component_var(w, I_mat, 0.99)
            -- With identity cov and equal weights, all CVaRs equal
            @test all(abs.(cvars .- cvars[1]) .< 1e-8)
        end

        -- -- Marginal VaR --
        @testset "marginal_var basic" begin
            w = ones(N) ./ N
            mvars = marginal_var(w, cov_mat, 0.99)
            @test length(mvars) == N
            @test all(isfinite.(mvars))
        end

        @testset "marginal_var relation_to_component" begin
            w = [0.3, 0.2, 0.25, 0.15, 0.1]
            mvars = marginal_var(w, cov_mat, 0.99)
            cvars = component_var(w, cov_mat, 0.99)
            -- CVaR_i = w_i * MVaR_i
            @test all(abs.(cvars .- w .* mvars) .< 1e-8)
        end

        @testset "marginal_var equal_weights_equal_cov" begin
            w = ones(N) ./ N
            I_mat = Matrix{Float64}(I(N))
            mvars = marginal_var(w, I_mat, 0.99)
            -- With identity cov and equal weights, all MVaRs equal
            @test all(abs.(mvars .- mvars[1]) .< 1e-8)
        end

        -- -- Diversification Ratio --
        @testset "diversification_ratio >= 1" begin
            w = ones(N) ./ N
            dr = diversification_ratio(w, cov_mat)
            @test dr >= 1.0 - 1e-8
        end

        @testset "diversification_ratio perfect_correlation" begin
            -- Perfectly correlated assets -> DR = 1
            vols2 = [0.2, 0.3, 0.25]
            N2 = 3
            perfect_rho = ones(N2, N2)
            cov_perfect = Diagonal(vols2) * perfect_rho * Diagonal(vols2)
            w2 = ones(N2) ./ N2
            dr = diversification_ratio(w2, cov_perfect)
            @test isapprox(dr, 1.0, atol=1e-6)
        end

        @testset "diversification_ratio uncorrelated" begin
            -- Uncorrelated assets -> higher DR
            N2 = 4
            vols2 = [0.2, 0.3, 0.15, 0.25]
            cov_uncorr = Diagonal(vols2 .^ 2)
            w2 = ones(N2) ./ N2
            dr_uncorr = diversification_ratio(w2, Matrix(cov_uncorr))
            @test dr_uncorr >= 1.0
        end

        @testset "diversification_ratio single_asset" begin
            w1 = [1.0]
            cov1 = reshape([0.04], 1, 1)
            dr = diversification_ratio(w1, cov1)
            @test isapprox(dr, 1.0, atol=1e-8)
        end

        @testset "portfolio_var basic" begin
            w = ones(N) ./ N
            pvar = portfolio_var(w, cov_mat, 0.99)
            @test pvar > 0.0
            @test isfinite(pvar)
        end

        @testset "portfolio_var confidence_monotone" begin
            w = ones(N) ./ N
            pvar_95 = portfolio_var(w, cov_mat, 0.95)
            pvar_99 = portfolio_var(w, cov_mat, 0.99)
            @test pvar_99 > pvar_95
        end

        @testset "brinson_by_sector_fields" begin
            pw = [0.25, 0.25, 0.20, 0.15, 0.15]
            bw = [0.20, 0.20, 0.25, 0.20, 0.15]
            pr = [0.05, 0.06, 0.03, 0.04, 0.07]
            br = [0.04, 0.05, 0.03, 0.03, 0.06]
            result = brinson_attribution(pw, bw, pr, br, sector_map)
            tech = result.by_sector["Tech"]
            @test haskey(Dict(pairs(tech)), :allocation)
            @test haskey(Dict(pairs(tech)), :selection)
            @test haskey(Dict(pairs(tech)), :interaction)
        end

    end
end

end # module RiskAttribution
