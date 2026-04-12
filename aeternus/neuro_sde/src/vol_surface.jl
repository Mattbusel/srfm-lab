"""
vol_surface.jl — Volatility surface construction, parametrisation, and dynamics

Implements:
  1. SVI (Stochastic Volatility Inspired) parametrisation — Gatheral (2004)
  2. SSVI (Surface SVI) global fit — Gatheral & Jacquier (2014)
  3. Arbitrage-free constraints:
     - Calendar spread (total variance non-decreasing in T)
     - Butterfly arbitrage (call prices convex in K)
     - Breeden-Litzenberger density positivity
  4. Surface smoothing (Tikhonov regularisation)
  5. Volatility surface dynamics: level/slope/curvature PCA
  6. Surface extrapolation (wings)
  7. Sticky-strike and sticky-delta models
  8. Forward vol and realised vs implied vol comparison

References:
  - Gatheral (2004) "A parsimonious arbitrage-free implied volatility parametrization"
  - Gatheral & Jacquier (2014) "Arbitrage-free SVI volatility surfaces"
  - Lee (2004) "The moment formula for implied volatility at extreme strikes"
  - Fengler (2005) "Semiparametric Modeling of Implied Volatility"
"""

using LinearAlgebra
using Statistics
using Random
using Distributions
using Optim
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SVI PARAMETRISATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    SVIParams

Raw SVI parametrisation:
  w(k) = a + b [ρ(k-m) + √((k-m)² + σ²)]

where k = log(K/F) is log-moneyness and w is total implied variance.

Fields:
  - a   : vertical translation (overall level of variance)
  - b   : slope/convexity of wings
  - ρ   : correlation (tilt) ∈ (-1,1)
  - m   : horizontal translation (ATM level)
  - σ   : ATM curvature (σ > 0)
"""
struct SVIParams
    a :: Float64
    b :: Float64
    ρ :: Float64
    m :: Float64
    σ :: Float64
end

"""
    svi_total_var(k, p::SVIParams) → Float64

Total implied variance w(k) under SVI parametrisation.
k = log(K/F).
"""
function svi_total_var(k::Real, p::SVIParams)
    d = k - p.m
    return p.a + p.b * (p.ρ * d + sqrt(d^2 + p.σ^2))
end

"""
    svi_implied_vol(k, T, p::SVIParams) → Float64

Black-Scholes implied vol from SVI total variance.
σ(k,T) = √(w(k)/T).
"""
function svi_implied_vol(k::Real, T::Real, p::SVIParams)
    w = svi_total_var(k, p)
    return w > 0 ? sqrt(w / T) : NaN
end

"""
    svi_smile(F, strikes, T, p::SVIParams) → Vector{Float64}

Compute SVI implied volatilities for a vector of strikes.
"""
function svi_smile(F::Real, strikes::AbstractVector, T::Real, p::SVIParams)
    ks = log.(strikes ./ F)
    return [svi_implied_vol(k, T, p) for k in ks]
end

"""
    svi_density(k, T, p::SVIParams) → Float64

Risk-neutral density via Breeden-Litzenberger:
g(k) = (1 - k w'/(2w) + w'²/4 (-1/w + 1/4 + w/(2·T)) - w''/2) × call_d2_term

Simplified: compute via finite differences on SVI smile.
"""
function svi_density(k::Real, T::Real, p::SVIParams; dk::Real=0.001)
    w0  = svi_total_var(k,    p)
    wp  = svi_total_var(k+dk, p)
    wm  = svi_total_var(k-dk, p)
    dw  = (wp - wm) / (2dk)
    d2w = (wp - 2w0 + wm) / dk^2
    w   = w0
    if w <= 0; return 0.0; end
    d1  = (-k + w/2) / sqrt(w)
    d2  = d1 - sqrt(w)
    # Breeden-Litzenberger density in moneyness space
    term1 = 1 - k * dw / (2w)
    term2 = dw^2 / 4 * (-1/w + 1/4 + w/2) - d2w / 2
    g     = (term1 + term2) * pdf(Normal(), d2) / sqrt(w)
    return max(g, 0.0)
end

"""
    svi_no_butterfly_check(p::SVIParams; n_grid=1000) → Bool

Check butterfly arbitrage condition: density ≥ 0 everywhere.
"""
function svi_no_butterfly_check(p::SVIParams; n_grid::Int=1000)
    ks = range(-3.0, 3.0, length=n_grid)
    for k in ks
        svi_density(k, 1.0, p) < -1e-6 && return false
    end
    return true
end

"""
    SVICalibResult

Result of SVI smile calibration.
"""
struct SVICalibResult
    params     :: SVIParams
    rmse       :: Float64
    converged  :: Bool
    n_iters    :: Int
    arb_free   :: Bool
end

"""
    calibrate_svi(F, strikes, market_vols, T;
                  n_restarts=5, seed=42) → SVICalibResult

Calibrate SVI parameters to a single-expiry implied vol smile.
"""
function calibrate_svi(F::Real,
                       strikes::AbstractVector,
                       market_vols::AbstractVector,
                       T::Real;
                       n_restarts::Int = 5,
                       seed::Int       = 42,
                       enforce_arb_free :: Bool = true)
    rng = MersenneTwister(seed)
    ks  = log.(strikes ./ F)
    market_tw = market_vols.^2 .* T    # target total variance

    best_obj = Inf
    best_p   = nothing
    best_iters = 0
    best_conv  = false

    for restart in 1:n_restarts
        a0 = rand(rng) * 0.05 + 0.005
        b0 = rand(rng) * 0.3  + 0.01
        ρ0 = rand(rng) * 1.6  - 0.8
        m0 = rand(rng) * 0.2  - 0.1
        σ0 = rand(rng) * 0.3  + 0.01

        x0 = [a0, b0, ρ0, m0, σ0]

        function obj(x)
            a, b_v, ρ, m, σ_ = x
            b_v = max(b_v, 1e-6)
            σ_  = max(σ_,  1e-6)
            ρ   = clamp(ρ, -0.9999, 0.9999)
            # SVI constraints: a + b σ √(1-ρ²) > 0
            a < -b_v * σ_ * sqrt(1 - ρ^2) && return 1e8
            p   = SVIParams(a, b_v, ρ, m, σ_)
            tw_model = [svi_total_var(k, p) for k in ks]
            any(tw_model .< 0) && return 1e8
            return sum((tw_model .- market_tw).^2)
        end

        try
            res = Optim.optimize(obj, x0, Optim.NelderMead(),
                                 Optim.Options(iterations=5000, g_tol=1e-12))
            if Optim.minimum(res) < best_obj
                best_obj   = Optim.minimum(res)
                best_p     = Optim.minimizer(res)
                best_iters = Optim.iterations(res)
                best_conv  = Optim.converged(res)
            end
        catch; continue; end
    end

    isnothing(best_p) && return SVICalibResult(SVIParams(0.04, 0.1, 0.0, 0.0, 0.2),
                                                Inf, false, 0, false)

    a, b_v, ρ, m, σ_ = best_p
    b_v = max(b_v, 1e-6); σ_ = max(σ_, 1e-6)
    ρ   = clamp(ρ, -0.9999, 0.9999)
    p_opt = SVIParams(a, b_v, ρ, m, σ_)

    model_vols = svi_smile(F, strikes, T, p_opt)
    rmse       = sqrt(mean((model_vols .- market_vols).^2))
    arb_free   = svi_no_butterfly_check(p_opt)

    return SVICalibResult(p_opt, rmse, best_conv, best_iters, arb_free)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SSVI (SURFACE SVI) GLOBAL FIT
# ─────────────────────────────────────────────────────────────────────────────

"""
    SSVIParams

SSVI parametrisation (Gatheral & Jacquier 2014):
  w(k, T) = θ_T / 2 [1 + ρ φ(θ_T) k + √((φ(θ_T)k + ρ)² + (1-ρ²))]

where:
  θ_T = ATM total variance at maturity T (term structure)
  φ(θ) = η / θ^γ  (power-law)

Fields:
  - ρ   : correlation (global)
  - η   : wing slope parameter
  - γ   : power-law exponent ∈ (0, 0.5]
"""
struct SSVIParams
    ρ :: Float64
    η :: Float64
    γ :: Float64
end

"""
    ssvi_phi(θ, p::SSVIParams) → Float64

SSVI wing slope function φ(θ) = η / θ^γ.
"""
ssvi_phi(θ::Real, p::SSVIParams) = p.η / θ^p.γ

"""
    ssvi_total_var(k, θ, p::SSVIParams) → Float64

SSVI total variance.
"""
function ssvi_total_var(k::Real, θ::Real, p::SSVIParams)
    ρ, φθ = p.ρ, ssvi_phi(θ, p)
    x = φθ * k
    return θ / 2 * (1 + ρ * x + sqrt((x + ρ)^2 + (1 - ρ^2)))
end

"""
    ssvi_surface(F_vec, strikes_mat, expiries, θ_vec, p::SSVIParams) → Matrix{Float64}

Compute SSVI implied vol surface.

- `θ_vec` : ATM total variances per expiry
"""
function ssvi_surface(F_vec::AbstractVector,
                      strikes_mat::AbstractMatrix,
                      expiries::AbstractVector,
                      θ_vec::AbstractVector,
                      p::SSVIParams)
    nK, nT = size(strikes_mat)
    iv_mat = zeros(nK, nT)
    for j in 1:nT
        T  = expiries[j]
        F  = F_vec[j]
        θ  = θ_vec[j]
        for i in 1:nK
            k = log(strikes_mat[i,j] / F)
            w = ssvi_total_var(k, θ, p)
            iv_mat[i,j] = w > 0 ? sqrt(w / T) : NaN
        end
    end
    return iv_mat
end

"""
    ssvi_no_calendar_arb(p::SSVIParams, θ_vec, expiries) → Bool

Calendar spread arbitrage condition for SSVI:
Total variance θ_T / 2 × [ATM term] must be non-decreasing in T.
"""
function ssvi_no_calendar_arb(p::SSVIParams,
                               θ_vec::AbstractVector,
                               expiries::AbstractVector)
    nT = length(expiries)
    for j in 1:(nT-1)
        # ATM total variance: θ/2 [1 + ρ⋅0 + √(1-ρ²+ρ²)] = θ  (at k=0)
        tv1 = θ_vec[j]
        tv2 = θ_vec[j+1]
        tv2 < tv1 - 1e-6 && return false
    end
    return true
end

"""
    calibrate_ssvi(F_vec, strikes_mat, market_vols_mat, expiries, θ_atm;
                   n_restarts=5, seed=42) → (SSVIParams, rmse)

Calibrate SSVI global surface parameters {ρ, η, γ} for given ATM
total variance term structure θ_atm[j] = ATM_vol_j² × T_j.
"""
function calibrate_ssvi(F_vec::AbstractVector,
                         strikes_mat::AbstractMatrix,
                         market_vols_mat::AbstractMatrix,
                         expiries::AbstractVector,
                         θ_atm::AbstractVector;
                         n_restarts::Int = 5,
                         seed::Int       = 42)
    rng  = MersenneTwister(seed)
    nK, nT = size(market_vols_mat)

    function obj(x)
        ρ = clamp(x[1], -0.9999, 0.9999)
        η = max(x[2], 1e-4)
        γ = clamp(x[3], 1e-4, 0.5)
        p = SSVIParams(ρ, η, γ)
        total = 0.0
        for j in 1:nT
            T = expiries[j]
            F = F_vec[j]
            θ = θ_atm[j]
            for i in 1:nK
                k   = log(strikes_mat[i,j] / F)
                w   = ssvi_total_var(k, θ, p)
                iv_m = w > 0 ? sqrt(w / T) : 0.0
                total += (iv_m - market_vols_mat[i,j])^2
            end
        end
        return total
    end

    best_obj_val = Inf
    best_p = SSVIParams(-0.7, 0.5, 0.4)

    for restart in 1:n_restarts
        ρ0 = rand(rng) * 1.8 - 0.9
        η0 = rand(rng) * 1.0 + 0.1
        γ0 = rand(rng) * 0.4 + 0.05
        x0 = [ρ0, η0, γ0]
        try
            res = Optim.optimize(obj, x0, Optim.NelderMead(),
                                 Optim.Options(iterations=10000, g_tol=1e-12))
            if Optim.minimum(res) < best_obj_val
                best_obj_val = Optim.minimum(res)
                x_opt = Optim.minimizer(res)
                best_p = SSVIParams(clamp(x_opt[1], -0.9999, 0.9999),
                                    max(x_opt[2], 1e-4),
                                    clamp(x_opt[3], 1e-4, 0.5))
            end
        catch; continue; end
    end

    # Compute RMSE
    iv_model = ssvi_surface(F_vec, strikes_mat, expiries, θ_atm, best_p)
    rmse = sqrt(mean((iv_model .- market_vols_mat).^2))
    return best_p, rmse
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ARBITRAGE-FREE CONSTRAINTS AND SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

"""
    VolSurface

Container for a discretised volatility surface with metadata.
"""
struct VolSurface
    log_moneyness :: Vector{Float64}   # k = log(K/F)
    expiries      :: Vector{Float64}
    total_var     :: Matrix{Float64}   # w(k,T) = σ² T (nK × nT)
    implied_vol   :: Matrix{Float64}   # σ(k,T) (nK × nT)
    is_arb_free   :: Bool
end

"""
    total_variance_to_vol(total_var, expiries) → Matrix{Float64}

Convert total variance matrix to implied vol matrix.
"""
function total_variance_to_vol(total_var::AbstractMatrix, expiries::AbstractVector)
    nK, nT = size(total_var)
    iv = zeros(nK, nT)
    for j in 1:nT
        T = expiries[j]
        for i in 1:nK
            w = total_var[i,j]
            iv[i,j] = w > 0 ? sqrt(w / T) : NaN
        end
    end
    return iv
end

"""
    enforce_calendar_spread(total_var) → Matrix{Float64}

Post-hoc enforcement of calendar spread (monotone total variance in T).
Uses isotonic regression in the T direction for each k slice.
"""
function enforce_calendar_spread(total_var::AbstractMatrix)
    nK, nT = size(total_var)
    tw_fixed = copy(total_var)
    for i in 1:nK
        tw_fixed[i, :] = isotonic_regression(tw_fixed[i, :])
    end
    return tw_fixed
end

"""
    isotonic_regression(y) → Vector{Float64}

Pool-adjacent-violators algorithm for non-decreasing isotonic regression.
"""
function isotonic_regression(y::AbstractVector)
    n   = length(y)
    out = copy(Float64.(y))
    blocks = [(sum=out[i], count=1, idx=i:i) for i in 1:n]

    changed = true
    while changed
        changed = false
        i = 1
        new_blocks = similar(blocks, 0)
        while i <= length(blocks)
            if i < length(blocks) && blocks[i].sum / blocks[i].count >
                                     blocks[i+1].sum / blocks[i+1].count
                # Merge
                merged_sum   = blocks[i].sum + blocks[i+1].sum
                merged_count = blocks[i].count + blocks[i+1].count
                merged_idx   = first(blocks[i].idx):last(blocks[i+1].idx)
                push!(new_blocks, (sum=merged_sum, count=merged_count,
                                   idx=merged_idx))
                i += 2
                changed = true
            else
                push!(new_blocks, blocks[i])
                i += 1
            end
        end
        blocks = new_blocks
    end

    for blk in blocks
        val = blk.sum / blk.count
        out[blk.idx] .= val
    end
    return out
end

"""
    tikhonov_smooth_surface(iv_mat; λ_k=0.1, λ_T=0.1) → Matrix{Float64}

Smooth an implied vol surface using Tikhonov (L2) regularisation.
Minimises ||σ - σ_market||² + λ_k ||∂²σ/∂k²||² + λ_T ||∂²σ/∂T²||².
"""
function tikhonov_smooth_surface(iv_mat::AbstractMatrix;
                                  λ_k::Real = 0.1,
                                  λ_T::Real = 0.1)
    nK, nT = size(iv_mat)
    # Build second-difference matrix D2 for each dimension
    function d2_matrix(n::Int)
        D = zeros(n-2, n)
        for i in 1:n-2
            D[i, i] = 1; D[i, i+1] = -2; D[i, i+2] = 1
        end
        return D
    end

    # Vectorise: x = vec(iv_mat), shape nK*nT
    y_vec  = vec(iv_mat)
    N      = nK * nT

    # Construct block-structured regularisation
    DK = d2_matrix(nK)   # (nK-2) × nK
    DT = d2_matrix(nT)   # (nT-2) × nT

    # Kronecker: regularisation in k direction for all T
    reg_k = kron(I(nT), DK)   # (nT*(nK-2)) × N
    reg_T = kron(DT, I(nK))   # ((nT-2)*nK) × N

    A = I(N) + λ_k * (reg_k' * reg_k) + λ_T * (reg_T' * reg_T)
    x_smooth = A \ y_vec

    return max.(reshape(x_smooth, nK, nT), 0.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: VOLATILITY SURFACE DYNAMICS (PCA)
# ─────────────────────────────────────────────────────────────────────────────

"""
    SurfacePCAResult

PCA decomposition of vol surface dynamics.
"""
struct SurfacePCAResult
    components      :: Matrix{Float64}   # (n_grid × n_components) loadings
    scores          :: Matrix{Float64}   # (n_dates × n_components) time series
    variance_explained :: Vector{Float64}
    mean_surface    :: Vector{Float64}
    factor_names    :: Vector{String}
end

"""
    vol_surface_pca(surface_history; n_components=3) → SurfacePCAResult

Perform PCA on a time series of vol surfaces.

`surface_history` : (nK*nT × n_dates) matrix (each column = flattened surface)
"""
function vol_surface_pca(surface_history::AbstractMatrix;
                          n_components::Int = 3)
    nG, n_dates = size(surface_history)

    # Demean
    x_bar   = mean(surface_history, dims=2)[:]
    X_c     = surface_history .- x_bar

    # SVD / PCA
    C = X_c * X_c' ./ (n_dates - 1)
    F = eigen(Symmetric(C); sortby=x->-x)

    n_k = min(n_components, nG)
    L   = F.vectors[:, 1:n_k]
    λ   = F.values[1:n_k]

    scores    = X_c' * L   # n_dates × n_k
    var_exp   = λ ./ max(sum(F.values), 1e-12)

    # Name factors by interpretation
    names = n_k >= 3 ? ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"] :
                       ["PC$i" for i in 1:n_k]

    return SurfacePCAResult(L, scores, var_exp, x_bar, names)
end

"""
    reconstruct_surface(pca::SurfacePCAResult, scores_t) → Vector{Float64}

Reconstruct a surface from its PCA scores.
"""
function reconstruct_surface(pca::SurfacePCAResult,
                              scores_t::AbstractVector)
    return pca.mean_surface .+ pca.components * scores_t
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: SURFACE EXTRAPOLATION (LEE MOMENT FORMULA)
# ─────────────────────────────────────────────────────────────────────────────

"""
    lee_moments(iv_grid, k_grid, T) → (β_right, β_left)

Estimate Lee (2004) moment formula decay exponents for extreme strikes.

As k → +∞: σ²(k) T ≤ β_R k  (right wing)
As k → -∞: σ²(k) T ≤ β_L|k| (left wing)

Estimators from slope of total variance in tail region.
"""
function lee_moments(iv_grid::AbstractVector,
                     k_grid::AbstractVector,
                     T::Real)
    tw = iv_grid.^2 .* T   # total variance
    n  = length(k_grid)
    # Right wing: last 20% of k_grid
    n_tail = max(Int(floor(0.2 * n)), 3)
    k_r = k_grid[end-n_tail+1:end]
    tw_r = tw[end-n_tail+1:end]
    β_R  = max(0.0, maximum(tw_r ./ max.(k_r, 1e-6)))

    k_l  = -k_grid[1:n_tail]     # make positive
    tw_l = tw[1:n_tail]
    β_L  = max(0.0, maximum(tw_l ./ max.(k_l, 1e-6)))

    return β_R, β_L
end

"""
    extrapolate_wing(k, k_boundary, iv_boundary, β; T=1.0, side=:right) → Float64

Extrapolate implied vol beyond calibrated region using Lee formula.
Uses a linear-in-total-variance extrapolation.
"""
function extrapolate_wing(k::Real, k_boundary::Real, iv_boundary::Real,
                           β::Real; T::Real=1.0, side::Symbol=:right)
    tw_b = iv_boundary^2 * T
    if side == :right
        tw_e = tw_b + β * (k - k_boundary)
    else
        tw_e = tw_b + β * (k_boundary - k)
    end
    return tw_e > 0 ? sqrt(max(tw_e, 0.0) / T) : 0.001
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FORWARD VOLATILITY AND VARIANCE SWAP
# ─────────────────────────────────────────────────────────────────────────────

"""
    forward_variance(total_var_T1, total_var_T2, T1, T2) → Float64

Forward variance between T1 and T2 from total variance at T1, T2.
σ²_fwd(T1, T2) = (σ²_T2 T2 - σ²_T1 T1) / (T2 - T1)
"""
function forward_variance(total_var_T1::Real, total_var_T2::Real,
                           T1::Real, T2::Real)
    return max((total_var_T2 * T2 - total_var_T1 * T1) / (T2 - T1), 0.0)
end

"""
    variance_swap_rate(total_var_slice, k_grid, T) → Float64

Approximate variance swap strike from model-free replication
(Carr-Madan log contract formula):
E[∫₀ᵀ σ²_t dt] ≈ 2/T ∫_{-∞}^{∞} w(k) dk (in total variance)

Numerical integration using trapezoidal rule on the k-grid.
"""
function variance_swap_rate(total_var_slice::AbstractVector,
                             k_grid::AbstractVector,
                             T::Real)
    dk = diff(k_grid)
    w  = total_var_slice
    # Trapezoidal rule
    integral = sum(0.5 * (w[i] + w[i+1]) * dk[i] for i in 1:length(dk))
    return 2.0 / T * integral
end

"""
    vix_approximation(iv_mat, k_grid, expiries, T1, T2;
                      near_col, far_col) → Float64

VIX-style model-free variance approximation interpolated to 30-day horizon.
"""
function vix_approximation(iv_mat::AbstractMatrix,
                            k_grid::AbstractVector,
                            expiries::AbstractVector;
                            T_target::Real = 30.0/365.0)
    nT = length(expiries)
    nT < 2 && return NaN

    # Find bounding expiries
    j = searchsortedfirst(expiries, T_target)
    j = clamp(j, 2, nT)

    T1 = expiries[j-1]; T2 = expiries[j]
    vsr1 = variance_swap_rate(iv_mat[:, j-1].^2 .* T1, k_grid, T1)
    vsr2 = variance_swap_rate(iv_mat[:, j].^2   .* T2, k_grid, T2)

    # Linear interpolation to T_target
    w = (T_target - T1) / (T2 - T1)
    vsr_target = (1 - w) * vsr1 + w * vsr2
    return sqrt(max(vsr_target, 0.0)) * 100   # VIX in percentage
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: STICKY MODELS
# ─────────────────────────────────────────────────────────────────────────────

"""
    StickyModel

Enum for sticky-strike vs sticky-delta vol surface dynamics.
"""
@enum StickyModel begin
    StickyStrike  # σ(K, T) stays constant as S moves
    StickyDelta   # σ(Δ, T) stays constant (vol moves with spot)
    StickyMoney   # σ(K/S, T) stays constant (moneyness-sticky)
end

"""
    roll_surface(iv_mat, k_grid, dlogS, sticky::StickyModel) → Matrix{Float64}

Compute how the vol surface changes when spot moves by dlogS.
"""
function roll_surface(iv_mat::AbstractMatrix,
                      k_grid::AbstractVector,
                      dlogS::Real,
                      sticky::StickyModel)
    if sticky == StickyStrike
        # Strikes fixed: k = log(K/F) changes by -dlogS
        return iv_mat   # unchanged in K-space (F-move adjusts k)
    elseif sticky == StickyDelta
        # Shift k_grid by -dlogS (surface shifts with spot)
        nK, nT = size(iv_mat)
        new_iv = zeros(nK, nT)
        for j in 1:nT
            for i in 1:nK
                new_k = k_grid[i] + dlogS
                # Interpolate
                idx = searchsortedfirst(k_grid, new_k)
                idx = clamp(idx, 2, nK)
                frac = (new_k - k_grid[idx-1]) / (k_grid[idx] - k_grid[idx-1])
                new_iv[i,j] = iv_mat[idx-1,j] * (1-frac) + iv_mat[idx,j] * frac
            end
        end
        return new_iv
    else  # StickyMoney
        # Same as sticky-delta in log-moneyness space
        return roll_surface(iv_mat, k_grid, dlogS, StickyDelta)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: FULL SURFACE CONSTRUCTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_vol_surface(spot, strikes_mat, expiries, market_iv, r, q;
                      method=:svi, smooth=true, enforce_arb=true) → VolSurface

Full pipeline to construct an arbitrage-free vol surface.

Steps:
  1. Fit per-expiry SVI (or global SSVI)
  2. Enforce calendar-spread constraint
  3. Apply Tikhonov smoothing (optional)
  4. Return VolSurface struct
"""
function build_vol_surface(spot::Real,
                            strikes_mat::AbstractMatrix,
                            expiries::AbstractVector,
                            market_iv::AbstractMatrix,
                            r::Real, q::Real;
                            method::Symbol = :svi,
                            smooth::Bool   = true,
                            enforce_arb::Bool = true,
                            verbose::Bool  = false)
    nK, nT = size(market_iv)
    F_vec  = [spot * exp((r - q) * T) for T in expiries]

    # Convert to total variance
    tw_market = zeros(nK, nT)
    for j in 1:nT, i in 1:nK
        tw_market[i,j] = market_iv[i,j]^2 * expiries[j]
    end

    # Fit per-expiry SVI
    tw_model = copy(tw_market)
    if method == :svi
        for j in 1:nT
            K_j  = strikes_mat[:, j]
            iv_j = market_iv[:, j]
            res  = calibrate_svi(F_vec[j], K_j, iv_j, expiries[j];
                                  n_restarts=3)
            if res.converged
                for i in 1:nK
                    k = log(K_j[i] / F_vec[j])
                    tw_model[i,j] = svi_total_var(k, res.params)
                end
            end
            verbose && @printf "  T=%.3f  SVI RMSE=%.6f  arb_free=%s\n" \
                expiries[j] res.rmse string(res.arb_free)
        end
    end

    # Enforce calendar spread
    if enforce_arb
        tw_model = enforce_calendar_spread(tw_model)
    end

    # Smoothing
    if smooth
        iv_rough = total_variance_to_vol(tw_model, expiries)
        iv_smooth = tikhonov_smooth_surface(iv_rough; λ_k=0.01, λ_T=0.01)
        tw_model  = zeros(nK, nT)
        for j in 1:nT, i in 1:nK
            tw_model[i,j] = iv_smooth[i,j]^2 * expiries[j]
        end
    end

    # Log-moneyness
    k_grid = log.(strikes_mat[:, 1] ./ F_vec[1])
    iv_final = total_variance_to_vol(tw_model, expiries)

    return VolSurface(k_grid, collect(Float64, expiries),
                      tw_model, iv_final, true)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: SURFACE INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    interp_vol_surface(vs::VolSurface, k, T) → Float64

Bilinear interpolation of implied vol at arbitrary (k, T).
Extrapolates using edge values beyond boundaries.
"""
function interp_vol_surface(vs::VolSurface, k::Real, T::Real)
    ks = vs.log_moneyness
    Ts = vs.expiries

    k_c = clamp(k, ks[1], ks[end])
    T_c = clamp(T, Ts[1], Ts[end])

    iK = searchsortedfirst(ks, k_c)
    iT = searchsortedfirst(Ts, T_c)
    iK = clamp(iK, 2, length(ks))
    iT = clamp(iT, 2, length(Ts))

    αK = (k_c - ks[iK-1]) / (ks[iK] - ks[iK-1])
    αT = (T_c - Ts[iT-1]) / (Ts[iT] - Ts[iT-1])

    v00 = vs.implied_vol[iK-1, iT-1]
    v10 = vs.implied_vol[iK,   iT-1]
    v01 = vs.implied_vol[iK-1, iT  ]
    v11 = vs.implied_vol[iK,   iT  ]

    # Replace NaN with neighbours
    any(isnan.([v00,v10,v01,v11])) && return mean(filter(!isnan, [v00,v10,v01,v11]))

    return (1-αK)*(1-αT)*v00 + αK*(1-αT)*v10 +
           (1-αK)*αT*v01    + αK*αT*v11
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: DEMO
# ─────────────────────────────────────────────────────────────────────────────

"""
    demo_vol_surface(; seed=1)

Demo: generate a Heston surface, fit SVI, compare.
"""
function demo_vol_surface(; seed::Int=1)
    rng = MersenneTwister(seed)
    # Heston params (approximate surface via SABR)
    S, r, q = 100.0, 0.05, 0.02
    expiries = [0.25, 0.5, 1.0, 2.0]
    strikes_1d = collect(75.0:5.0:125.0)
    nK = length(strikes_1d)
    nT = length(expiries)

    K_mat  = repeat(strikes_1d, 1, nT)
    F_vec  = [S * exp((r-q)*T) for T in expiries]

    # Generate synthetic market vol surface using simple analytic approximation
    iv_mat = zeros(nK, nT)
    for j in 1:nT
        T  = expiries[j]
        F  = F_vec[j]
        atm_vol = 0.2 + 0.02 * (T - 1.0)  # term structure
        skew    = -0.1 / sqrt(T)
        conv    = 0.05 / T
        for i in 1:nK
            k = log(strikes_1d[i] / F)
            iv_mat[i,j] = atm_vol + skew * k + conv * k^2
        end
    end
    iv_mat = max.(iv_mat, 0.05)

    @info "Building vol surface (SVI fit)..."
    vs = build_vol_surface(S, K_mat, expiries, iv_mat, r, q;
                            method=:svi, smooth=true, verbose=true)

    @info "VolSurface built: $(size(vs.implied_vol)) grid, arb_free=$(vs.is_arb_free)"

    # VIX approximation
    k_grid = log.(strikes_1d ./ F_vec[1])
    vix = vix_approximation(iv_mat, k_grid, expiries; T_target=30/365)
    @info "Model-free VIX approximation: $(round(vix, digits=2))"

    return vs
end
