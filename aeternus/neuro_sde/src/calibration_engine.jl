"""
calibration_engine.jl — Full market calibration engine for Neural SDE

Implements:
  1. Black-Scholes implied volatility extraction (Brent root-finding)
  2. SABR model calibration to volatility smile
  3. Heston model calibration via characteristic function / Carr-Madan FFT
  4. Local volatility surface via Dupire formula
  5. Parameter stability diagnostics across strikes/expiries
  6. Levenberg-Marquardt optimizer wrapper
  7. Calibration quality metrics (RMSE, MAPE, Vega-weighted errors)
  8. Joint calibration of multiple expiries
  9. Regularisation for stable surface interpolation
 10. Bootstrap uncertainty quantification for calibrated parameters

References:
  - Black & Scholes (1973) "The Pricing of Options and Corporate Liabilities"
  - Hagan et al. (2002) "Managing Smile Risk" (SABR model)
  - Heston (1993) "A Closed-Form Solution for Options with Stochastic Volatility"
  - Dupire (1994) "Pricing with a Smile"
  - Carr & Madan (1999) "Option valuation using the fast Fourier transform"
  - Levenberg (1944), Marquardt (1963) — LM algorithm
"""

using LinearAlgebra
using Statistics
using Random
using Distributions
using FFTW
using Optim
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: BLACK-SCHOLES FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────────────────

"""
    bs_d1(S, K, r, q, σ, T)

Compute the d1 term of the Black-Scholes formula.
"""
function bs_d1(S::Real, K::Real, r::Real, q::Real, σ::Real, T::Real)
    return (log(S / K) + (r - q + 0.5 * σ^2) * T) / (σ * sqrt(T))
end

"""
    bs_d2(S, K, r, q, σ, T)

Compute the d2 term: d2 = d1 - σ√T.
"""
function bs_d2(S::Real, K::Real, r::Real, q::Real, σ::Real, T::Real)
    return bs_d1(S, K, r, q, σ, T) - σ * sqrt(T)
end

"""
    bs_call(S, K, r, q, σ, T) → price

Black-Scholes call price for spot S, strike K, rate r,
dividend yield q, vol σ, maturity T.
"""
function bs_call(S::Real, K::Real, r::Real, q::Real, σ::Real, T::Real)
    if T <= 0.0 || σ <= 0.0
        return max(S * exp(-q * T) - K * exp(-r * T), 0.0)
    end
    d1 = bs_d1(S, K, r, q, σ, T)
    d2 = d1 - σ * sqrt(T)
    N  = x -> cdf(Normal(), x)
    return S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)
end

"""
    bs_put(S, K, r, q, σ, T) → price

Black-Scholes put price via put-call parity.
"""
function bs_put(S::Real, K::Real, r::Real, q::Real, σ::Real, T::Real)
    call = bs_call(S, K, r, q, σ, T)
    return call - S * exp(-q * T) + K * exp(-r * T)
end

"""
    bs_vega(S, K, r, q, σ, T) → vega

Black-Scholes vega: ∂C/∂σ.
"""
function bs_vega(S::Real, K::Real, r::Real, q::Real, σ::Real, T::Real)
    if T <= 0.0 || σ <= 0.0
        return 0.0
    end
    d1 = bs_d1(S, K, r, q, σ, T)
    return S * exp(-q * T) * pdf(Normal(), d1) * sqrt(T)
end

"""
    bs_delta(S, K, r, q, σ, T; call=true) → delta

Black-Scholes delta for call (default) or put.
"""
function bs_delta(S::Real, K::Real, r::Real, q::Real, σ::Real, T::Real; call::Bool=true)
    if T <= 0.0 || σ <= 0.0
        return call ? Float64(S > K) : Float64(S < K) - 1.0
    end
    d1 = bs_d1(S, K, r, q, σ, T)
    N  = x -> cdf(Normal(), x)
    if call
        return exp(-q * T) * N(d1)
    else
        return exp(-q * T) * (N(d1) - 1.0)
    end
end

"""
    bs_gamma(S, K, r, q, σ, T) → gamma

Second derivative ∂²C/∂S².
"""
function bs_gamma(S::Real, K::Real, r::Real, q::Real, σ::Real, T::Real)
    T <= 0.0 || σ <= 0.0 && return 0.0
    d1 = bs_d1(S, K, r, q, σ, T)
    return exp(-q * T) * pdf(Normal(), d1) / (S * σ * sqrt(T))
end

"""
    bs_theta(S, K, r, q, σ, T; call=true) → theta (per calendar day)

Time decay of the option price.
"""
function bs_theta(S::Real, K::Real, r::Real, q::Real, σ::Real, T::Real; call::Bool=true)
    T <= 0.0 || σ <= 0.0 && return 0.0
    d1 = bs_d1(S, K, r, q, σ, T)
    d2 = d1 - σ * sqrt(T)
    N  = x -> cdf(Normal(), x)
    ϕ  = pdf(Normal(), d1)
    if call
        θ = (-S * exp(-q * T) * ϕ * σ / (2 * sqrt(T))
             - r * K * exp(-r * T) * N(d2)
             + q * S * exp(-q * T) * N(d1))
    else
        θ = (-S * exp(-q * T) * ϕ * σ / (2 * sqrt(T))
             + r * K * exp(-r * T) * N(-d2)
             - q * S * exp(-q * T) * N(-d1))
    end
    return θ / 365.0   # per calendar day
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: IMPLIED VOLATILITY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

"""
    ImpliedVolResult

Result of implied-vol inversion.

Fields:
  - `sigma`      : extracted implied volatility (NaN if not converged)
  - `converged`  : boolean flag
  - `iterations` : number of Newton/Brent iterations used
  - `residual`   : final price residual |model - market|
"""
struct ImpliedVolResult
    sigma      :: Float64
    converged  :: Bool
    iterations :: Int
    residual   :: Float64
end

"""
    implied_vol_newton(market_price, S, K, r, q, T; call=true,
                       σ0=0.25, tol=1e-8, max_iter=200)

Newton-Raphson implied volatility inversion.  Falls back to Brent if
vega becomes too small (near-the-money at short expiry edge cases).

Returns `ImpliedVolResult`.
"""
function implied_vol_newton(market_price::Real, S::Real, K::Real,
                            r::Real, q::Real, T::Real;
                            call::Bool   = true,
                            σ0::Real     = 0.25,
                            tol::Real    = 1e-8,
                            max_iter::Int = 200)
    price_fn = call ? bs_call : bs_put
    σ = Float64(σ0)
    iter = 0
    res  = Inf
    for i in 1:max_iter
        iter  = i
        price = price_fn(S, K, r, q, σ, T)
        vega  = bs_vega(S, K, r, q, σ, T)
        res   = price - market_price
        abs(res) < tol && break
        if abs(vega) < 1e-12
            # fall back to bisection step
            σ = σ * (1.0 + (res > 0 ? -0.1 : 0.1))
        else
            σ -= res / vega
        end
        σ = clamp(σ, 1e-6, 20.0)
    end
    converged = abs(res) < tol * 10
    return ImpliedVolResult(σ, converged, iter, abs(res))
end

"""
    implied_vol_brent(market_price, S, K, r, q, T; call=true,
                      tol=1e-10, max_iter=500)

Brent's method for robust implied vol extraction (guaranteed convergence
on bracketed interval [1e-6, 20]).
"""
function implied_vol_brent(market_price::Real, S::Real, K::Real,
                           r::Real, q::Real, T::Real;
                           call::Bool    = true,
                           tol::Real     = 1e-10,
                           max_iter::Int = 500)
    price_fn = call ? bs_call : bs_put
    f = σ -> price_fn(S, K, r, q, σ, T) - market_price

    a, b = 1e-6, 20.0
    fa, fb = f(a), f(b)

    # check bracket
    if fa * fb > 0
        # try extending
        b = 5.0; fb = f(b)
        fa * fb > 0 && return ImpliedVolResult(NaN, false, 0, abs(f(0.25)))
    end

    c, fc = b, fb
    iter = 0
    s, fs = 0.0, 0.0
    mflag = true
    d = 0.0

    for i in 1:max_iter
        iter = i
        if abs(b - a) < tol
            break
        end
        if fa != fc && fb != fc
            # inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
               + b * fa * fc / ((fb - fa) * (fb - fc))
               + c * fa * fb / ((fc - fa) * (fc - fb)))
        else
            s = b - fb * (b - a) / (fb - fa)
        end
        cond1 = !((3a + b) / 4 < s < b || b < s < (3a + b) / 4)
        cond2 = mflag && abs(s - b) >= abs(b - c) / 2
        cond3 = !mflag && abs(s - b) >= abs(c - d) / 2
        cond4 = mflag && abs(b - c) < tol
        cond5 = !mflag && abs(c - d) < tol
        if cond1 || cond2 || cond3 || cond4 || cond5
            s = (a + b) / 2
            mflag = true
        else
            mflag = false
        end
        fs = f(s)
        d, c, fc = c, b, fb
        if fa * fs < 0
            b, fb = s, fs
        else
            a, fa = s, fs
        end
        if abs(fa) < abs(fb)
            a, fa, b, fb = b, fb, a, fa
        end
    end
    converged = abs(fs) < 1e-6
    return ImpliedVolResult(b, converged, iter, abs(fb))
end

"""
    implied_vol_surface(market_prices, S, strikes, expiries, r, q;
                        call=true, method=:newton) → Matrix{ImpliedVolResult}

Extract implied vols from a grid of market prices.

- `market_prices[i,j]` : price for strike i, expiry j
- Returns matrix of ImpliedVolResult of same dimensions.
"""
function implied_vol_surface(market_prices::AbstractMatrix,
                             S::Real,
                             strikes::AbstractVector,
                             expiries::AbstractVector,
                             r::Real, q::Real;
                             call::Bool   = true,
                             method::Symbol = :newton)
    nK, nT = length(strikes), length(expiries)
    @assert size(market_prices) == (nK, nT)
    results = Matrix{ImpliedVolResult}(undef, nK, nT)
    for j in 1:nT, i in 1:nK
        if method == :brent
            results[i,j] = implied_vol_brent(market_prices[i,j], S,
                                             strikes[i], r, q, expiries[j];
                                             call=call)
        else
            results[i,j] = implied_vol_newton(market_prices[i,j], S,
                                              strikes[i], r, q, expiries[j];
                                              call=call)
        end
    end
    return results
end

"""
    extract_vol_matrix(iv_results) → (sigma_matrix, converged_matrix)

Pull sigma values and convergence flags from ImpliedVolResult matrix.
"""
function extract_vol_matrix(iv_results::AbstractMatrix{ImpliedVolResult})
    sigma = [r.sigma for r in iv_results]
    conv  = [r.converged for r in iv_results]
    return sigma, conv
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SABR MODEL
# ─────────────────────────────────────────────────────────────────────────────

"""
    SABRParams

Parameters for the SABR stochastic volatility model:

  dF = σ Fᵝ dW₁
  dσ = α σ  dW₂
  ⟨dW₁, dW₂⟩ = ρ dt

Fields: α (vol-of-vol), β (elasticity), ρ (correlation), σ0 (initial vol).
"""
struct SABRParams
    α  :: Float64   # vol-of-vol
    β  :: Float64   # CEV exponent ∈ [0,1]
    ρ  :: Float64   # correlation ∈ (-1,1)
    σ0 :: Float64   # initial vol (ATM level)
end

"""
    sabr_implied_vol(F, K, T, p::SABRParams) → σ_imp

Hagan et al. (2002) asymptotic formula for SABR implied volatility.
Handles ATM case (K ≈ F) separately to avoid division by zero.
"""
function sabr_implied_vol(F::Real, K::Real, T::Real, p::SABRParams)
    α, β, ρ, σ0 = p.α, p.β, p.ρ, p.σ0
    if abs(F - K) < 1e-10 * F
        # ATM approximation
        FKβ   = F^(1 - β)
        term1 = σ0 / FKβ
        term2 = 1.0 + ((1 - β)^2 / 24 * σ0^2 / FKβ^2
                      + ρ * β * α * σ0 / (4 * FKβ)
                      + (2 - 3ρ^2) / 24 * α^2) * T
        return term1 * term2
    end

    FK     = F * K
    FKmid  = FK^((1 - β) / 2)
    logFK  = log(F / K)
    z      = α / σ0 * FKmid * logFK
    χ      = log((sqrt(1 - 2ρ * z + z^2) + z - ρ) / (1 - ρ))

    denom = FKmid * (1 + (1 - β)^2 / 24 * logFK^2
                      + (1 - β)^4 / 1920 * logFK^4)
    num   = σ0
    zχ    = abs(χ) < 1e-10 ? 1.0 : z / χ

    vol_lead = num / denom * zχ

    correction = (1 + ((1 - β)^2 / 24 * σ0^2 / FK^(1 - β)
                      + ρ * β * α * σ0 / (4 * FK^((1 - β) / 2))
                      + (2 - 3ρ^2) / 24 * α^2) * T)
    return vol_lead * correction
end

"""
    sabr_smile(F, strikes, T, p::SABRParams) → Vector{Float64}

Compute SABR implied vol for a vector of strikes at a single expiry.
"""
function sabr_smile(F::Real, strikes::AbstractVector, T::Real, p::SABRParams)
    return [sabr_implied_vol(F, K, T, p) for K in strikes]
end

"""
    SABRCalibResult

Result of SABR smile calibration.
"""
struct SABRCalibResult
    params      :: SABRParams
    rmse        :: Float64
    vega_wrmse  :: Float64
    converged   :: Bool
    n_iters     :: Int
end

"""
    calibrate_sabr(F, strikes, market_vols, T, S, r, q;
                   β=0.5, n_restarts=5, seed=42) → SABRCalibResult

Calibrate SABR parameters {α, ρ, σ0} for fixed β by minimising RMSE
between model and market implied vols.  β can optionally be calibrated too.
"""
function calibrate_sabr(F::Real,
                        strikes::AbstractVector,
                        market_vols::AbstractVector,
                        T::Real,
                        S::Real, r::Real, q::Real;
                        β::Real       = 0.5,
                        fit_β::Bool   = false,
                        n_restarts::Int = 5,
                        seed::Int     = 42)
    rng = MersenneTwister(seed)
    best_res = nothing
    best_obj = Inf

    # Vega weights from market
    vegas = [bs_vega(S, K, r, q, v, T) for (K, v) in zip(strikes, market_vols)]
    w = max.(vegas, 1e-8)
    w ./= sum(w)

    for restart in 1:n_restarts
        # Random initial guess
        α0  = rand(rng) * 0.8 + 0.1
        ρ0  = rand(rng) * 1.6 - 0.8
        σ00 = rand(rng) * 0.4 + 0.05
        β0  = fit_β ? rand(rng) : β

        if fit_β
            x0     = [α0, ρ0, σ00, β0]
            lb     = [1e-4, -0.999, 1e-4, 0.0]
            ub     = [5.0,   0.999, 5.0,  1.0]
        else
            x0     = [α0, ρ0, σ00]
            lb     = [1e-4, -0.999, 1e-4]
            ub     = [5.0,   0.999, 5.0]
        end

        function obj(x)
            α_  = x[1]
            ρ_  = clamp(x[2], -0.9999, 0.9999)
            σ0_ = x[3]
            β_  = fit_β ? clamp(x[4], 0.0, 1.0) : β
            p   = SABRParams(α_, β_, ρ_, σ0_)
            model_vols = sabr_smile(F, strikes, T, p)
            return sum(w[i] * (model_vols[i] - market_vols[i])^2
                       for i in 1:length(strikes))
        end

        try
            res = Optim.optimize(obj, x0,
                                 Optim.NelderMead(),
                                 Optim.Options(iterations=5000,
                                               g_tol=1e-10))
            if Optim.minimum(res) < best_obj
                best_obj = Optim.minimum(res)
                best_res = res
            end
        catch
            continue
        end
    end

    isnothing(best_res) && return SABRCalibResult(
        SABRParams(0.3, β, 0.0, 0.2), Inf, Inf, false, 0)

    x_opt = Optim.minimizer(best_res)
    α_opt  = x_opt[1]
    ρ_opt  = clamp(x_opt[2], -0.9999, 0.9999)
    σ0_opt = x_opt[3]
    β_opt  = fit_β ? clamp(x_opt[4], 0.0, 1.0) : β

    p_opt      = SABRParams(α_opt, β_opt, ρ_opt, σ0_opt)
    model_vols = sabr_smile(F, strikes, T, p_opt)
    rmse       = sqrt(mean((model_vols .- market_vols).^2))
    vw_rmse    = sqrt(sum(w .* (model_vols .- market_vols).^2))

    return SABRCalibResult(p_opt, rmse, vw_rmse,
                           Optim.converged(best_res),
                           Optim.iterations(best_res))
end

"""
    calibrate_sabr_surface(F_vec, strikes_mat, market_vols_mat, expiries;
                           β=0.5) → Vector{SABRCalibResult}

Calibrate SABR smile for each expiry independently.
"""
function calibrate_sabr_surface(F_vec::AbstractVector,
                                strikes_mat::AbstractMatrix,
                                market_vols_mat::AbstractMatrix,
                                expiries::AbstractVector,
                                S::Real, r::Real, q::Real;
                                β::Real = 0.5)
    nT = length(expiries)
    results = Vector{SABRCalibResult}(undef, nT)
    for j in 1:nT
        results[j] = calibrate_sabr(F_vec[j],
                                    strikes_mat[:, j],
                                    market_vols_mat[:, j],
                                    expiries[j],
                                    S, r, q; β=β)
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: HESTON MODEL — CHARACTERISTIC FUNCTION & CARR-MADAN FFT
# ─────────────────────────────────────────────────────────────────────────────

"""
    HestonParams

Parameters for the Heston (1993) stochastic volatility model:

  dS = (r - q) S dt + √V S dW₁
  dV = κ(θ - V) dt + ξ √V dW₂
  ⟨dW₁, dW₂⟩ = ρ dt

Fields:
  - κ  : mean-reversion speed
  - θ  : long-run variance
  - ξ  : vol-of-vol
  - ρ  : correlation
  - V0 : initial variance
"""
struct HestonParams
    κ  :: Float64
    θ  :: Float64
    ξ  :: Float64
    ρ  :: Float64
    V0 :: Float64
end

"""
    heston_char_fn(u, S, r, q, T, p::HestonParams) → ComplexF64

Heston characteristic function Φ(u) = E[exp(iu log(S_T/S_0))].
Uses the numerically stable 'Little Trap' formulation (Albrecher et al. 2007).
"""
function heston_char_fn(u::Real, S::Real, r::Real, q::Real, T::Real, p::HestonParams)
    κ, θ, ξ, ρ, V0 = p.κ, p.θ, p.ξ, p.ρ, p.V0
    iu = im * u
    d  = sqrt((ρ * ξ * iu - κ)^2 + ξ^2 * (iu + u^2))
    g  = (κ - ρ * ξ * iu - d) / (κ - ρ * ξ * iu + d)

    # Avoid overflow for large T: use alternative form
    exp_dT = exp(-d * T)
    B = (κ - ρ * ξ * iu - d) * (1 - exp_dT) / (ξ^2 * (1 - g * exp_dT))
    A = (iu * (r - q) * T
         + κ * θ / ξ^2 * ((κ - ρ * ξ * iu - d) * T
                           - 2 * log((1 - g * exp_dT) / (1 - g))))

    return exp(A + B * V0 + iu * log(S))
end

"""
    heston_call_fft(S, strikes, r, q, T, p::HestonParams;
                    N=4096, α=1.5, η=0.25) → Vector{Float64}

Price European calls using Carr-Madan FFT.

Parameters:
  - N  : FFT grid size (power of 2)
  - α  : damping parameter (typically 1.0–2.0)
  - η  : log-strike spacing
"""
function heston_call_fft(S::Real,
                         strikes::AbstractVector,
                         r::Real, q::Real, T::Real,
                         p::HestonParams;
                         N::Int    = 4096,
                         α::Real   = 1.5,
                         η::Real   = 0.25)
    λ    = 2π / (N * η)
    b    = N * λ / 2
    ku   = -b .+ λ .* (0:N-1)          # log-strike grid

    # Construct integrand
    νs = η .* (0:N-1)
    integrand = zeros(ComplexF64, N)
    for j in 1:N
        ν = νs[j]
        ψ = (heston_char_fn(ν - (α + 1) * im, S, r, q, T, p)
             / (α^2 + α - ν^2 + im * (2α + 1) * ν))
        # Simpson weights (modified)
        w = (j == 1 || j == N) ? 1.0 : (iseven(j) ? 4.0 : 2.0)
        integrand[j] = exp(-im * b * ν) * ψ * w * η / 3
    end

    fft_val = fft(integrand)
    call_prices_ku = real.(exp.(-α .* ku) ./ π .* fft_val)

    # Interpolate at requested log-strikes
    log_K  = log.(strikes ./ S)
    n_K    = length(strikes)
    prices = zeros(n_K)
    for (i, lk) in enumerate(log_K)
        # linear interpolation in log-strike space
        idx = searchsortedfirst(ku, lk)
        if idx <= 1
            prices[i] = call_prices_ku[1]
        elseif idx > N
            prices[i] = call_prices_ku[N]
        else
            frac = (lk - ku[idx-1]) / (ku[idx] - ku[idx-1])
            prices[i] = call_prices_ku[idx-1] * (1 - frac) + call_prices_ku[idx] * frac
        end
    end
    return max.(prices, 0.0)
end

"""
    heston_implied_vols(S, strikes, r, q, T, p::HestonParams;
                        N=4096) → Vector{Float64}

Compute Black-Scholes implied volatilities from Heston model prices.
"""
function heston_implied_vols(S::Real, strikes::AbstractVector,
                             r::Real, q::Real, T::Real, p::HestonParams;
                             N::Int = 4096)
    prices = heston_call_fft(S, strikes, r, q, T, p; N=N)
    ivs    = zeros(length(strikes))
    for (i, (K, price)) in enumerate(zip(strikes, prices))
        res = implied_vol_brent(price, S, K, r, q, T; call=true)
        ivs[i] = res.converged ? res.sigma : NaN
    end
    return ivs
end

"""
    HestonCalibResult

Result of Heston model calibration.
"""
struct HestonCalibResult
    params     :: HestonParams
    rmse       :: Float64
    vega_wrmse :: Float64
    converged  :: Bool
    n_iters    :: Int
    per_expiry_rmse :: Vector{Float64}
end

"""
    calibrate_heston(S, strikes_mat, market_vols_mat, expiries, r, q;
                     n_restarts=8, seed=1) → HestonCalibResult

Joint calibration of Heston parameters across all expiries by minimising
vega-weighted RMSE between model and market implied vols.
"""
function calibrate_heston(S::Real,
                          strikes_mat::AbstractMatrix,
                          market_vols_mat::AbstractMatrix,
                          expiries::AbstractVector,
                          r::Real, q::Real;
                          n_restarts::Int = 8,
                          seed::Int       = 1)
    rng  = MersenneTwister(seed)
    nK   = size(strikes_mat, 1)
    nT   = length(expiries)

    # Pre-compute vega weights
    vega_weights = zeros(nK, nT)
    for j in 1:nT, i in 1:nK
        vega_weights[i,j] = bs_vega(S, strikes_mat[i,j], r, q,
                                    market_vols_mat[i,j], expiries[j])
    end
    vega_weights .= max.(vega_weights, 1e-10)
    vega_weights ./= sum(vega_weights)

    function heston_obj(x)
        κ  = max(x[1], 1e-4)
        θ  = max(x[2], 1e-6)
        ξ  = max(x[3], 1e-4)
        ρ  = clamp(x[4], -0.9999, 0.9999)
        V0 = max(x[5], 1e-6)
        # Feller condition: 2κθ > ξ²
        p  = HestonParams(κ, θ, ξ, ρ, V0)
        total = 0.0
        for j in 1:nT
            K_j = strikes_mat[:, j]
            mv_j = market_vols_mat[:, j]
            try
                model_iv = heston_implied_vols(S, K_j, r, q, expiries[j], p)
                for i in 1:nK
                    isnan(model_iv[i]) && continue
                    total += vega_weights[i,j] * (model_iv[i] - mv_j[i])^2
                end
            catch
                return 1e8
            end
        end
        return total
    end

    best_params = nothing
    best_obj_val = Inf
    best_iters   = 0

    for restart in 1:n_restarts
        κ0  = rand(rng) * 4 + 0.5
        θ0  = rand(rng) * 0.1 + 0.01
        ξ0  = rand(rng) * 0.8 + 0.1
        ρ0  = rand(rng) * 1.6 - 0.8
        V00 = rand(rng) * 0.08 + 0.01

        x0  = [κ0, θ0, ξ0, ρ0, V00]

        try
            res = Optim.optimize(heston_obj, x0,
                                 Optim.NelderMead(),
                                 Optim.Options(iterations=10000,
                                               g_tol=1e-10,
                                               f_tol=1e-14))
            if Optim.minimum(res) < best_obj_val
                best_obj_val = Optim.minimum(res)
                best_params  = Optim.minimizer(res)
                best_iters   = Optim.iterations(res)
            end
        catch
            continue
        end
    end

    isnothing(best_params) && return HestonCalibResult(
        HestonParams(2.0, 0.04, 0.5, -0.7, 0.04), Inf, Inf, false, 0, Float64[])

    κ_o  = max(best_params[1], 1e-4)
    θ_o  = max(best_params[2], 1e-6)
    ξ_o  = max(best_params[3], 1e-4)
    ρ_o  = clamp(best_params[4], -0.9999, 0.9999)
    V0_o = max(best_params[5], 1e-6)
    p_o  = HestonParams(κ_o, θ_o, ξ_o, ρ_o, V0_o)

    # Compute per-expiry RMSE
    per_rmse = zeros(nT)
    all_sq_err = Float64[]
    for j in 1:nT
        K_j  = strikes_mat[:, j]
        mv_j = market_vols_mat[:, j]
        model_iv = heston_implied_vols(S, K_j, r, q, expiries[j], p_o)
        per_rmse[j] = sqrt(mean((model_iv .- mv_j).^2))
        append!(all_sq_err, (model_iv .- mv_j).^2)
    end
    rmse      = sqrt(mean(all_sq_err))
    vw_rmse   = sqrt(best_obj_val)

    return HestonCalibResult(p_o, rmse, vw_rmse,
                             best_iters > 0, best_iters, per_rmse)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: LOCAL VOLATILITY — DUPIRE FORMULA
# ─────────────────────────────────────────────────────────────────────────────

"""
    LocalVolSurface

Container for discretised local volatility surface σ_loc(K, T).

Fields:
  - `strikes`  : vector of strikes
  - `expiries` : vector of expiries
  - `sigma`    : (nK × nT) matrix of local vols
"""
struct LocalVolSurface
    strikes  :: Vector{Float64}
    expiries :: Vector{Float64}
    sigma    :: Matrix{Float64}
end

"""
    dupire_local_vol(S, strikes, expiries, call_prices, r, q) → LocalVolSurface

Compute local volatility surface from call prices via Dupire's formula:

  σ_loc²(K, T) = (∂C/∂T + (r-q)K ∂C/∂K + q C) / (½ K² ∂²C/∂K²)

Uses finite differences for the derivatives.
"""
function dupire_local_vol(S::Real,
                          strikes::AbstractVector,
                          expiries::AbstractVector,
                          call_prices::AbstractMatrix,
                          r::Real, q::Real)
    nK = length(strikes)
    nT = length(expiries)
    @assert size(call_prices) == (nK, nT)

    lv = zeros(nK, nT)

    for j in 1:nT
        T = expiries[j]
        for i in 2:(nK-1)
            K = strikes[i]

            # ∂C/∂T — forward difference (or central if not at boundary)
            if j < nT
                dCdT = (call_prices[i, j+1] - call_prices[i, j]) /
                       (expiries[j+1] - expiries[j])
            else
                dCdT = (call_prices[i, j] - call_prices[i, j-1]) /
                       (expiries[j] - expiries[j-1])
            end

            # ∂C/∂K — central difference
            dK1   = strikes[i+1] - strikes[i]
            dK2   = strikes[i] - strikes[i-1]
            dCdK  = (call_prices[i+1, j] - call_prices[i-1, j]) / (dK1 + dK2)

            # ∂²C/∂K² — central second difference
            d2CdK2 = 2 * (call_prices[i+1, j] / dK1 -
                          call_prices[i,   j] * (1/dK1 + 1/dK2) +
                          call_prices[i-1, j] / dK2) / (dK1 + dK2)

            numerator   = dCdT + (r - q) * K * dCdK + q * call_prices[i, j]
            denominator = 0.5 * K^2 * d2CdK2

            if denominator > 1e-12 && numerator > 0.0
                lv[i, j] = sqrt(numerator / denominator)
            else
                lv[i, j] = NaN
            end
        end
        # boundary values — extrapolate
        lv[1,   j] = lv[2,   j]
        lv[end, j] = lv[end-1, j]
    end
    return LocalVolSurface(collect(Float64, strikes),
                           collect(Float64, expiries), lv)
end

"""
    interp_local_vol(lv::LocalVolSurface, K, T) → Float64

Bilinear interpolation of local vol surface at arbitrary (K, T).
"""
function interp_local_vol(lv::LocalVolSurface, K::Real, T::Real)
    K = clamp(K, lv.strikes[1], lv.strikes[end])
    T = clamp(T, lv.expiries[1], lv.expiries[end])

    iK = searchsortedfirst(lv.strikes, K)
    iT = searchsortedfirst(lv.expiries, T)
    iK = clamp(iK, 2, length(lv.strikes))
    iT = clamp(iT, 2, length(lv.expiries))

    K0, K1 = lv.strikes[iK-1], lv.strikes[iK]
    T0, T1 = lv.expiries[iT-1], lv.expiries[iT]
    αK = (K - K0) / (K1 - K0)
    αT = (T - T0) / (T1 - T0)

    v00 = lv.sigma[iK-1, iT-1]
    v10 = lv.sigma[iK,   iT-1]
    v01 = lv.sigma[iK-1, iT  ]
    v11 = lv.sigma[iK,   iT  ]

    return ((1-αK) * (1-αT) * v00 + αK * (1-αT) * v10
           + (1-αK) * αT * v01    + αK * αT * v11)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: LEVENBERG-MARQUARDT OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

"""
    LMConfig

Configuration for Levenberg-Marquardt.

Fields:
  - max_iter  : maximum iterations
  - tol_f     : tolerance on ‖F‖
  - tol_x     : tolerance on parameter change ‖Δx‖
  - λ0        : initial damping
  - λ_up      : damping increase factor
  - λ_dn      : damping decrease factor
  - λ_max     : upper damping bound
"""
struct LMConfig
    max_iter :: Int
    tol_f    :: Float64
    tol_x    :: Float64
    λ0       :: Float64
    λ_up     :: Float64
    λ_dn     :: Float64
    λ_max    :: Float64
end

LMConfig(; max_iter=500, tol_f=1e-10, tol_x=1e-10,
           λ0=1e-3, λ_up=10.0, λ_dn=0.1, λ_max=1e16) =
    LMConfig(max_iter, tol_f, tol_x, λ0, λ_up, λ_dn, λ_max)

"""
    LMResult

Result of Levenberg-Marquardt optimisation.
"""
struct LMResult
    x          :: Vector{Float64}   # optimal parameters
    fval       :: Vector{Float64}   # residuals at optimum
    cost       :: Float64           # ½‖f‖²
    converged  :: Bool
    n_iters    :: Int
    history    :: Vector{Float64}   # cost per iteration
end

"""
    lm_jacobian(f, x; ε=1e-6) → Matrix{Float64}

Finite-difference Jacobian ∂f_i/∂x_j.
"""
function lm_jacobian(f::Function, x::AbstractVector; ε::Real=1e-6)
    f0 = f(x)
    m  = length(f0)
    n  = length(x)
    J  = zeros(m, n)
    for j in 1:n
        xp     = copy(x)
        xp[j] += ε
        J[:, j] = (f(xp) .- f0) ./ ε
    end
    return J
end

"""
    levenberg_marquardt(f, x0; cfg=LMConfig(), bounds=nothing) → LMResult

Levenberg-Marquardt algorithm for nonlinear least squares min ½‖f(x)‖².

`f` maps ℝⁿ → ℝᵐ (residual vector).
`bounds` is optionally a tuple (lb, ub) of vectors.
"""
function levenberg_marquardt(f::Function,
                             x0::AbstractVector;
                             cfg::LMConfig       = LMConfig(),
                             bounds::Union{Nothing, Tuple} = nothing)
    x       = copy(Float64.(x0))
    λ       = cfg.λ0
    history = Float64[]
    converged = false
    n_iters   = 0

    fval = f(x)
    cost = 0.5 * dot(fval, fval)
    push!(history, cost)

    for iter in 1:cfg.max_iter
        n_iters = iter
        J  = lm_jacobian(f, x)
        JtJ = J' * J
        Jtf = J' * fval

        # Regularised step: (JᵀJ + λI) Δx = -Jᵀf
        A  = JtJ + λ * I(length(x))
        Δx = -(A \ Jtf)

        x_new = x .+ Δx
        # Apply bounds if given
        if !isnothing(bounds)
            x_new = clamp.(x_new, bounds[1], bounds[2])
        end

        fval_new = f(x_new)
        cost_new = 0.5 * dot(fval_new, fval_new)

        # Actual vs predicted reduction
        pred_red = 0.5 * dot(fval, fval) - 0.5 * dot(J * Δx .+ fval, J * Δx .+ fval)
        ρ = (cost - cost_new) / (abs(pred_red) + 1e-20)

        if ρ > 0.25
            x    = x_new
            fval = fval_new
            cost = cost_new
            λ    = max(λ * cfg.λ_dn, 1e-20)
        else
            λ    = min(λ * cfg.λ_up, cfg.λ_max)
        end
        push!(history, cost)

        # Convergence checks
        if norm(fval) < cfg.tol_f
            converged = true; break
        end
        if norm(Δx) < cfg.tol_x
            converged = true; break
        end
    end
    return LMResult(x, fval, cost, converged, n_iters, history)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: PARAMETER STABILITY DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParameterStability

Diagnostics for parameter stability across strikes/expiries.
"""
struct ParameterStability
    param_name   :: String
    values       :: Vector{Float64}   # parameter values per slice
    labels       :: Vector{String}    # slice labels (e.g. "T=0.25")
    mean_val     :: Float64
    std_val      :: Float64
    cv           :: Float64           # coefficient of variation
    is_stable    :: Bool              # CV < threshold
end

"""
    assess_stability(values, labels, param_name; cv_threshold=0.2) → ParameterStability

Assess how stable a calibrated parameter is across different slices.
"""
function assess_stability(values::AbstractVector,
                          labels::AbstractVector{String},
                          param_name::String;
                          cv_threshold::Real = 0.2)
    μ  = mean(values)
    σ  = std(values)
    cv = abs(μ) > 1e-10 ? σ / abs(μ) : Inf
    return ParameterStability(param_name, collect(Float64, values),
                              labels, μ, σ, cv, cv < cv_threshold)
end

"""
    analyse_sabr_stability(calib_results, expiries) → Vector{ParameterStability}

Check stability of SABR parameters {α, ρ, σ0} across expiries.
"""
function analyse_sabr_stability(calib_results::Vector{SABRCalibResult},
                                expiries::AbstractVector)
    labels = ["T=$(round(T, digits=3))" for T in expiries]
    αs  = [r.params.α  for r in calib_results]
    ρs  = [r.params.ρ  for r in calib_results]
    σ0s = [r.params.σ0 for r in calib_results]
    βs  = [r.params.β  for r in calib_results]
    return [
        assess_stability(αs,  labels, "α"),
        assess_stability(ρs,  labels, "ρ"),
        assess_stability(σ0s, labels, "σ₀"),
        assess_stability(βs,  labels, "β"),
    ]
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: CALIBRATION QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    CalibQualityMetrics

Summary statistics measuring calibration quality.
"""
struct CalibQualityMetrics
    rmse          :: Float64   # root-mean-square error in vol
    mape          :: Float64   # mean absolute percentage error
    vega_wrmse    :: Float64   # vega-weighted RMSE
    max_abs_err   :: Float64   # worst-case absolute error
    n_obs         :: Int       # number of observations
    r_squared     :: Float64   # R²
    avg_bid_ask   :: Float64   # average half-spread (if provided)
    within_ba     :: Float64   # fraction within bid-ask
end

"""
    compute_calib_metrics(model_vols, market_vols, S, strikes, expiries,
                          r, q; bid_ask_spread=nothing) → CalibQualityMetrics

Compute a comprehensive set of calibration quality metrics.
"""
function compute_calib_metrics(model_vols::AbstractMatrix,
                               market_vols::AbstractMatrix,
                               S::Real,
                               strikes::AbstractMatrix,
                               expiries::AbstractVector,
                               r::Real, q::Real;
                               bid_ask_spread::Union{Nothing, AbstractMatrix} = nothing)
    errors    = model_vols .- market_vols
    abs_errors = abs.(errors)
    nK, nT   = size(model_vols)

    # RMSE
    rmse = sqrt(mean(errors.^2))

    # MAPE
    mape = mean(abs_errors ./ max.(abs.(market_vols), 1e-8)) * 100

    # Vega weights
    vegas = zeros(nK, nT)
    for j in 1:nT, i in 1:nK
        vegas[i,j] = bs_vega(S, strikes[i,j], r, q, market_vols[i,j], expiries[j])
    end
    w = max.(vegas, 1e-10)
    w ./= sum(w)
    vega_wrmse = sqrt(sum(w .* errors.^2))

    # Max error
    max_err = maximum(abs_errors)

    # R²
    ss_res = sum(errors.^2)
    ss_tot = sum((market_vols .- mean(market_vols)).^2)
    r2     = 1.0 - ss_res / max(ss_tot, 1e-16)

    # Bid-ask
    avg_ba   = isnothing(bid_ask_spread) ? NaN : mean(bid_ask_spread) / 2
    frac_in  = isnothing(bid_ask_spread) ? NaN :
               mean(abs_errors .<= bid_ask_spread ./ 2)

    return CalibQualityMetrics(rmse, mape, vega_wrmse, max_err,
                               nK * nT, r2, avg_ba, frac_in)
end

"""
    print_calib_metrics(m::CalibQualityMetrics)

Pretty-print calibration quality metrics.
"""
function print_calib_metrics(m::CalibQualityMetrics)
    @printf "──────────────────────────────────────────\n"
    @printf "  Calibration Quality Metrics\n"
    @printf "──────────────────────────────────────────\n"
    @printf "  Observations      : %d\n"        m.n_obs
    @printf "  RMSE (vol)        : %.6f\n"      m.rmse
    @printf "  MAPE (%%)          : %.4f\n"      m.mape
    @printf "  Vega-Weighted RMSE: %.6f\n"      m.vega_wrmse
    @printf "  Max Abs Error     : %.6f\n"      m.max_abs_err
    @printf "  R²                : %.6f\n"      m.r_squared
    isnan(m.avg_bid_ask) || @printf "  Avg Bid-Ask Half  : %.6f\n" m.avg_bid_ask
    isnan(m.within_ba)   || @printf "  Within B/A (%%)    : %.2f\n" m.within_ba * 100
    @printf "──────────────────────────────────────────\n"
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: JOINT MULTI-EXPIRY CALIBRATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

"""
    MarketVolSurface

Complete market data structure for a volatility surface.

Fields:
  - `spot`     : spot price S₀
  - `r`        : risk-free rate
  - `q`        : dividend yield
  - `strikes`  : (nK × nT) matrix of strikes
  - `expiries` : nT vector of expiries
  - `market_iv`: (nK × nT) market implied vols
  - `bid_iv`   : optional bid vols
  - `ask_iv`   : optional ask vols
"""
struct MarketVolSurface
    spot     :: Float64
    r        :: Float64
    q        :: Float64
    strikes  :: Matrix{Float64}
    expiries :: Vector{Float64}
    market_iv :: Matrix{Float64}
    bid_iv   :: Union{Nothing, Matrix{Float64}}
    ask_iv   :: Union{Nothing, Matrix{Float64}}
end

"""
    CalibrationResult

Top-level container for full surface calibration results.
"""
struct CalibrationResult
    surface       :: MarketVolSurface
    heston        :: Union{Nothing, HestonCalibResult}
    sabr_by_expiry :: Union{Nothing, Vector{SABRCalibResult}}
    local_vol     :: Union{Nothing, LocalVolSurface}
    metrics       :: Union{Nothing, CalibQualityMetrics}
    stability     :: Union{Nothing, Vector{ParameterStability}}
    timestamp     :: Float64
end

"""
    calibrate_surface(surf::MarketVolSurface;
                      models=[:heston, :sabr],
                      n_restarts=5) → CalibrationResult

Full calibration pipeline: fit Heston and/or SABR models to a given
market vol surface, compute quality metrics and stability diagnostics.
"""
function calibrate_surface(surf::MarketVolSurface;
                           models::Vector{Symbol}  = [:heston, :sabr],
                           n_restarts::Int         = 5,
                           verbose::Bool           = false)
    S, r, q  = surf.spot, surf.r, surf.q
    K_mat    = surf.strikes
    T_vec    = surf.expiries
    iv_mat   = surf.market_iv
    nK, nT   = size(iv_mat)

    heston_res   = nothing
    sabr_results = nothing
    lv_surface   = nothing

    if :heston in models
        verbose && @info "Calibrating Heston model..."
        heston_res = calibrate_heston(S, K_mat, iv_mat, T_vec, r, q;
                                      n_restarts=n_restarts)
        verbose && @printf "  Heston RMSE = %.6f\n" heston_res.rmse
    end

    if :sabr in models
        verbose && @info "Calibrating SABR model per expiry..."
        F_vec     = [S * exp((r - q) * T) for T in T_vec]
        sabr_results = Vector{SABRCalibResult}(undef, nT)
        for j in 1:nT
            sabr_results[j] = calibrate_sabr(F_vec[j],
                                             K_mat[:, j],
                                             iv_mat[:, j],
                                             T_vec[j], S, r, q)
        end
        verbose && @info "  SABR calibration done."
    end

    if :localvol in models
        verbose && @info "Computing Dupire local vol surface..."
        # Need call prices from market vols
        call_prices = zeros(nK, nT)
        for j in 1:nT, i in 1:nK
            call_prices[i,j] = bs_call(S, K_mat[i,j], r, q, iv_mat[i,j], T_vec[j])
        end
        lv_surface = dupire_local_vol(S, K_mat[:, 1], T_vec, call_prices, r, q)
    end

    # Metrics from Heston
    metrics  = nothing
    if !isnothing(heston_res)
        model_iv = zeros(nK, nT)
        for j in 1:nT
            model_iv[:, j] = heston_implied_vols(S, K_mat[:, j], r, q,
                                                 T_vec[j], heston_res.params)
        end
        ba_spread = isnothing(surf.bid_iv) ? nothing :
                    surf.ask_iv .- surf.bid_iv
        metrics = compute_calib_metrics(model_iv, iv_mat, S, K_mat, T_vec, r, q;
                                        bid_ask_spread=ba_spread)
        verbose && print_calib_metrics(metrics)
    end

    # Stability
    stability = nothing
    if !isnothing(sabr_results)
        labels = ["T=$(round(T, digits=3))" for T in T_vec]
        stability = analyse_sabr_stability(sabr_results, T_vec)
    end

    return CalibrationResult(surf, heston_res, sabr_results, lv_surface,
                             metrics, stability, time())
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: BOOTSTRAP UNCERTAINTY FOR CALIBRATED PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

"""
    BootstrapCalibResult

Bootstrap confidence intervals for calibrated Heston parameters.
"""
struct BootstrapCalibResult
    param_names :: Vector{String}
    means       :: Vector{Float64}
    stds        :: Vector{Float64}
    ci_lower    :: Vector{Float64}  # 2.5th percentile
    ci_upper    :: Vector{Float64}  # 97.5th percentile
    n_bootstrap :: Int
end

"""
    bootstrap_heston(surf::MarketVolSurface; B=200, noise_level=0.005,
                     seed=42, n_restarts=3) → BootstrapCalibResult

Bootstrap uncertainty quantification for Heston parameters.
Adds random perturbations to market vols and re-calibrates B times.
"""
function bootstrap_heston(surf::MarketVolSurface;
                          B::Int            = 200,
                          noise_level::Real = 0.005,
                          seed::Int         = 42,
                          n_restarts::Int   = 3)
    rng       = MersenneTwister(seed)
    nK, nT    = size(surf.market_iv)
    param_mat = zeros(B, 5)

    for b in 1:B
        # Perturb implied vols
        noise    = noise_level * randn(rng, nK, nT)
        iv_noisy = max.(surf.market_iv .+ noise, 1e-4)

        surf_b = MarketVolSurface(surf.spot, surf.r, surf.q,
                                  surf.strikes, surf.expiries,
                                  iv_noisy, nothing, nothing)
        try
            res = calibrate_heston(surf.spot, surf.strikes, iv_noisy,
                                   surf.expiries, surf.r, surf.q;
                                   n_restarts=n_restarts)
            p = res.params
            param_mat[b, :] = [p.κ, p.θ, p.ξ, p.ρ, p.V0]
        catch
            param_mat[b, :] .= NaN
        end
    end

    # Remove NaN rows
    valid = .!any(isnan.(param_mat), dims=2)[:, 1]
    PM    = param_mat[valid, :]
    n_ok  = sum(valid)

    names  = ["κ", "θ", "ξ", "ρ", "V₀"]
    means  = vec(mean(PM, dims=1))
    stds   = vec(std(PM, dims=1))
    lo     = [quantile(PM[:, j], 0.025) for j in 1:5]
    hi     = [quantile(PM[:, j], 0.975) for j in 1:5]

    return BootstrapCalibResult(names, means, stds, lo, hi, n_ok)
end

"""
    print_bootstrap_ci(bc::BootstrapCalibResult)

Display bootstrap confidence intervals.
"""
function print_bootstrap_ci(bc::BootstrapCalibResult)
    @printf "──────────────────────────────────────────────────────────\n"
    @printf "  Heston Bootstrap Confidence Intervals  (n=%d)\n" bc.n_bootstrap
    @printf "  %-8s  %8s  %8s  %12s  %12s\n" "Param" "Mean" "Std" "CI 2.5%%" "CI 97.5%%"
    @printf "──────────────────────────────────────────────────────────\n"
    for i in 1:length(bc.param_names)
        @printf "  %-8s  %8.5f  %8.5f  %12.5f  %12.5f\n" \
            bc.param_names[i] bc.means[i] bc.stds[i] bc.ci_lower[i] bc.ci_upper[i]
    end
    @printf "──────────────────────────────────────────────────────────\n"
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
    moneyness(S, K, r, q, T) → Float64

Log-moneyness: log(F/K) where F = S exp((r-q)T).
"""
moneyness(S, K, r, q, T) = log(S * exp((r-q)*T) / K)

"""
    atm_vol(surf::MarketVolSurface) → Vector{Float64}

Approximate ATM implied vol for each expiry (interpolated at K=F).
"""
function atm_vol(surf::MarketVolSurface)
    S, r, q = surf.spot, surf.r, surf.q
    atm_vols = zeros(length(surf.expiries))
    for j in 1:length(surf.expiries)
        T = surf.expiries[j]
        F = S * exp((r - q) * T)
        K_j  = surf.strikes[:, j]
        iv_j = surf.market_iv[:, j]
        # Find bracketing indices
        idx = searchsortedfirst(K_j, F)
        if idx <= 1
            atm_vols[j] = iv_j[1]
        elseif idx > length(K_j)
            atm_vols[j] = iv_j[end]
        else
            frac = (F - K_j[idx-1]) / (K_j[idx] - K_j[idx-1])
            atm_vols[j] = iv_j[idx-1] * (1 - frac) + iv_j[idx] * frac
        end
    end
    return atm_vols
end

"""
    vol_skew(surf::MarketVolSurface; delta=0.25) → Vector{Float64}

25-delta risk reversal proxy: vol at 25Δ put minus vol at 25Δ call.
"""
function vol_skew(surf::MarketVolSurface; delta::Real=0.25)
    S, r, q = surf.spot, surf.r, surf.q
    skews   = zeros(length(surf.expiries))
    for j in 1:length(surf.expiries)
        T    = surf.expiries[j]
        K_j  = surf.strikes[:, j]
        iv_j = surf.market_iv[:, j]
        # Find approximate strikes for ±25Δ using bisection on delta
        skews[j] = NaN  # placeholder when data insufficient
        length(K_j) < 3 && continue
        # Coarse: use interpolation at K ≈ F * exp(-σ√T * Φ⁻¹(Δ))
        atm_σ  = iv_j[div(length(iv_j), 2)]
        Φinv25 = quantile(Normal(), delta)
        K_put  = S * exp((r-q)*T) * exp(-atm_σ * sqrt(T) * Φinv25)
        K_call = S * exp((r-q)*T) * exp( atm_σ * sqrt(T) * Φinv25)
        # Interpolate
        function interp_iv(K)
            idx = searchsortedfirst(K_j, K)
            idx <= 1   && return iv_j[1]
            idx > length(K_j) && return iv_j[end]
            frac = (K - K_j[idx-1]) / (K_j[idx] - K_j[idx-1])
            return iv_j[idx-1] * (1-frac) + iv_j[idx] * frac
        end
        skews[j] = interp_iv(K_put) - interp_iv(K_call)
    end
    return skews
end

"""
    arbitrage_free_check(surf::MarketVolSurface) → NamedTuple

Check for calendar spread and butterfly arbitrage in the vol surface.

Returns:
  - `calendar_ok` : Bool — no calendar spread arbitrage
  - `butterfly_ok`: Bool — no butterfly arbitrage
  - `violations`  : Vector of (type, i, j, value) tuples
"""
function arbitrage_free_check(surf::MarketVolSurface)
    S, r, q  = surf.spot, surf.r, surf.q
    K_mat    = surf.strikes
    T_vec    = surf.expiries
    iv_mat   = surf.market_iv
    nK, nT   = size(iv_mat)

    violations = NamedTuple{(:type, :i, :j, :value), Tuple{Symbol,Int,Int,Float64}}[]
    cal_ok  = true
    but_ok  = true

    # Calendar spread: total variance T*σ² must be non-decreasing in T
    for i in 1:nK
        for j in 1:(nT-1)
            tv1 = T_vec[j]   * iv_mat[i, j]^2
            tv2 = T_vec[j+1] * iv_mat[i, j+1]^2
            if tv2 < tv1 - 1e-6
                cal_ok = false
                push!(violations, (type=:calendar_spread, i=i, j=j, value=tv1-tv2))
            end
        end
    end

    # Butterfly: call price must be convex in K
    for j in 1:nT
        T = T_vec[j]
        call_prices = [bs_call(S, K_mat[i,j], r, q, iv_mat[i,j], T) for i in 1:nK]
        for i in 2:(nK-1)
            butterfly = call_prices[i-1] - 2*call_prices[i] + call_prices[i+1]
            if butterfly < -1e-6
                but_ok = false
                push!(violations, (type=:butterfly, i=i, j=j, value=butterfly))
            end
        end
    end
    return (calendar_ok=cal_ok, butterfly_ok=but_ok, violations=violations)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: SIMPLE DEMO / SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

"""
    demo_calibration_engine()

Quick smoke test: generate synthetic Heston surface, recover parameters.
"""
function demo_calibration_engine(; verbose::Bool=true)
    S, r, q = 100.0, 0.05, 0.02
    true_p  = HestonParams(2.0, 0.04, 0.6, -0.7, 0.04)

    strikes_1d = collect(80.0:5.0:120.0)
    expiries   = [0.25, 0.5, 1.0, 2.0]
    nK, nT     = length(strikes_1d), length(expiries)

    K_mat  = repeat(strikes_1d, 1, nT)
    iv_mat = zeros(nK, nT)
    for j in 1:nT
        iv_mat[:, j] = heston_implied_vols(S, strikes_1d, r, q, expiries[j], true_p)
    end

    surf = MarketVolSurface(S, r, q, K_mat, expiries, iv_mat, nothing, nothing)

    verbose && @info "Running Heston calibration on synthetic surface..."
    res = calibrate_surface(surf; models=[:heston], n_restarts=3, verbose=verbose)

    if !isnothing(res.heston)
        verbose && @printf "True  κ=%.2f θ=%.4f ξ=%.2f ρ=%.2f V0=%.4f\n" \
            true_p.κ true_p.θ true_p.ξ true_p.ρ true_p.V0
        p_hat = res.heston.params
        verbose && @printf "Calib κ=%.2f θ=%.4f ξ=%.2f ρ=%.2f V0=%.4f\n" \
            p_hat.κ p_hat.θ p_hat.ξ p_hat.ρ p_hat.V0
        verbose && @printf "RMSE: %.6f\n" res.heston.rmse
    end
    return res
end


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: EXTENDED HESTON FFT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    heston_price_put_fft(S, strikes, r, q, T, p; N=4096) → Vector{Float64}

Price European puts via Carr-Madan with call prices plus put-call parity.
"""
function heston_price_put_fft(S::Real, strikes::AbstractVector,
                               r::Real, q::Real, T::Real, p::HestonParams;
                               N::Int=4096)
    calls = heston_call_fft(S, strikes, r, q, T, p; N=N)
    puts  = [calls[i] - (S*exp(-q*T) - strikes[i]*exp(-r*T))
             for i in 1:length(strikes)]
    return max.(puts, 0.0)
end

"""
    heston_variance_swap_vol(S, r, q, T, p; N=1024) → Float64

Model-free variance swap strike from Heston model via log-strip replication.
"""
function heston_variance_swap_vol(S::Real, r::Real, q::Real, T::Real,
                                   p::HestonParams; N::Int=1024)
    log_moneyness = range(-1.5, 1.5, length=N)
    strikes       = S .* exp.(log_moneyness)
    calls         = heston_call_fft(S, strikes, r, q, T, p)
    puts          = heston_price_put_fft(S, strikes, r, q, T, p)
    dk_log        = step(log_moneyness)
    integrand = zeros(N)
    for i in 1:N
        K = strikes[i]
        integrand[i] = 2 * (K < S ? puts[i] : calls[i]) / (K^2 * T)
    end
    total = sum(0.5*(integrand[i]*strikes[i] + integrand[i-1]*strikes[i-1])*dk_log
                for i in 2:N)
    return sqrt(max(total, 0.0))
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14: CALIBRATION DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

"""
    interpolate_missing_vols(iv_mat, strikes, expiries) → Matrix{Float64}

Fill missing (NaN) entries in a vol surface via linear interpolation in strike.
"""
function interpolate_missing_vols(iv_mat::AbstractMatrix,
                                   strikes::AbstractMatrix,
                                   expiries::AbstractVector)
    nK, nT = size(iv_mat)
    iv_fill = copy(iv_mat)
    for j in 1:nT, i in 1:nK
        isnan(iv_fill[i,j]) || continue
        above = findfirst(!isnan, @view iv_fill[i:end, j])
        below = findlast( !isnan, @view iv_fill[1:i, j])
        if !isnothing(above) && !isnothing(below)
            ia = i + above - 1; ib = below
            frac = (i - ib) / max(ia - ib, 1)
            iv_fill[i,j] = iv_fill[ib,j]*(1-frac) + iv_fill[ia,j]*frac
        elseif !isnothing(above)
            iv_fill[i,j] = iv_fill[i + above - 1, j]
        elseif !isnothing(below)
            iv_fill[i,j] = iv_fill[below, j]
        end
    end
    return iv_fill
end

"""
    CleanedSurface

Vol surface after data cleaning.
"""
struct CleanedSurface
    market_vol_surface :: MarketVolSurface
    n_removed          :: Int
    n_interpolated     :: Int
    cleaning_log       :: Vector{String}
end

"""
    clean_vol_surface(surf::MarketVolSurface;
                      min_vol=0.01, max_vol=3.0) → CleanedSurface

Data cleaning pipeline for market vol surface: removes outliers and
interpolates missing entries.
"""
function clean_vol_surface(surf::MarketVolSurface;
                            min_vol::Real       = 0.01,
                            max_vol::Real       = 3.0,
                            min_moneyness::Real = -0.6,
                            max_moneyness::Real =  0.6)
    iv      = copy(surf.market_iv)
    log_str = String[]
    n_rm    = 0

    S, r, q = surf.spot, surf.r, surf.q
    nK, nT  = size(iv)

    for j in 1:nT, i in 1:nK
        v = iv[i,j]; isnan(v) && continue
        if v < min_vol || v > max_vol
            push!(log_str, "Removed iv[$i,$j]=$(round(v,digits=4)) (range)")
            iv[i,j] = NaN; n_rm += 1; continue
        end
        T = surf.expiries[j]; K = surf.strikes[i,j]
        F = S * exp((r-q)*T)
        k = log(K/F)
        if k < min_moneyness || k > max_moneyness
            push!(log_str, "Removed iv[$i,$j] k=$(round(k,digits=3)) (moneyness)")
            iv[i,j] = NaN; n_rm += 1
        end
    end

    iv_clean = interpolate_missing_vols(iv, surf.strikes, surf.expiries)
    n_int    = sum(isnan.(iv) .& .!isnan.(iv_clean))

    new_surf = MarketVolSurface(surf.spot, surf.r, surf.q, surf.strikes,
                                 surf.expiries, iv_clean, surf.bid_iv, surf.ask_iv)
    return CleanedSurface(new_surf, n_rm, n_int, log_str)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15: OPTION GREEKS SURFACE
# ─────────────────────────────────────────────────────────────────────────────

"""
    GreeksSurface

Container for all option Greeks over a vol surface grid.
"""
struct GreeksSurface
    delta  :: Matrix{Float64}
    gamma  :: Matrix{Float64}
    vega   :: Matrix{Float64}
    theta  :: Matrix{Float64}
    vanna  :: Matrix{Float64}
    volga  :: Matrix{Float64}
end

"""
    compute_greeks_surface(surf::MarketVolSurface; call=true) → GreeksSurface

Compute full Greeks surface from market vol surface.
"""
function compute_greeks_surface(surf::MarketVolSurface; call::Bool=true)
    S, r, q = surf.spot, surf.r, surf.q
    nK, nT  = size(surf.market_iv)
    delta  = zeros(nK,nT); gamma  = zeros(nK,nT); vega = zeros(nK,nT)
    theta  = zeros(nK,nT); vanna  = zeros(nK,nT); volga = zeros(nK,nT)
    for j in 1:nT, i in 1:nK
        T = surf.expiries[j]; K = surf.strikes[i,j]; σ = surf.market_iv[i,j]
        isnan(σ) && continue
        delta[i,j] = bs_delta(S, K, r, q, σ, T; call=call)
        gamma[i,j] = bs_gamma(S, K, r, q, σ, T)
        vega[i,j]  = bs_vega(S, K, r, q, σ, T)
        theta[i,j] = bs_theta(S, K, r, q, σ, T; call=call)
        if T > 0 && σ > 0
            d1 = bs_d1(S, K, r, q, σ, T)
            d2 = d1 - σ * sqrt(T)
            vanna[i,j] = -vega[i,j] / S * d2 / (σ * sqrt(T))
            volga[i,j] = vega[i,j] * d1 * d2 / σ
        end
    end
    return GreeksSurface(delta, gamma, vega, theta, vanna, volga)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16: VOLATILITY RISK PREMIUM
# ─────────────────────────────────────────────────────────────────────────────

"""
    vol_risk_premium(implied_vols, realized_vols; lag=1) → Vector{Float64}

Volatility risk premium: VRP_t = IV_{t-lag} - RV_t.
Positive = market pays for variance insurance.
"""
function vol_risk_premium(implied_vols::AbstractVector,
                           realized_vols::AbstractVector;
                           lag::Int = 1)
    n   = min(length(implied_vols), length(realized_vols))
    vrp = zeros(n)
    for t in (lag+1):n
        vrp[t] = implied_vols[t-lag] - realized_vols[t]
    end
    return vrp
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17: ENSEMBLE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    EnsembleCalibration

Weighted ensemble of calibrated models for improved robustness.
"""
struct EnsembleCalibration
    models  :: Vector{Any}
    weights :: Vector{Float64}
    names   :: Vector{String}
end

"""
    ensemble_implied_vol(ens, S, K, r, q, T) → Float64

Weighted average implied vol from calibration ensemble.
"""
function ensemble_implied_vol(ens::EnsembleCalibration,
                               S::Real, K::Real, r::Real, q::Real, T::Real;
                               N_fft::Int=2048)
    iv_sum = 0.0
    for (m, w) in zip(ens.models, ens.weights)
        w < 1e-6 && continue
        if m isa HestonParams
            ivv = heston_implied_vols(S, [K], r, q, T, m; N=N_fft)
            isnan(ivv[1]) || (iv_sum += w * ivv[1])
        elseif m isa SABRParams
            F = S * exp((r-q)*T)
            iv_sum += w * sabr_implied_vol(F, K, T, m)
        end
    end
    return iv_sum
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18: CALIBRATION LOG AND DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

"""
    CalibrationLog

Time-stamped log of calibration results for drift monitoring.
"""
mutable struct CalibrationLog
    entries :: Vector{NamedTuple}
end
CalibrationLog() = CalibrationLog(NamedTuple[])

"""
    log_calibration!(clog, result, timestamp)
"""
function log_calibration!(clog::CalibrationLog,
                           result::HestonCalibResult,
                           timestamp::Float64)
    p = result.params
    push!(clog.entries, (
        timestamp=timestamp, κ=p.κ, θ=p.θ, ξ=p.ξ, ρ=p.ρ, V0=p.V0,
        rmse=result.rmse, converged=result.converged
    ))
end

"""
    parameter_drift(clog::CalibrationLog) → NamedTuple

Detect significant parameter drift across logged calibrations.
"""
function parameter_drift(clog::CalibrationLog)
    n = length(clog.entries); n < 2 && return (drift_detected=false, max_cv=0.0)
    κs = [e.κ for e in clog.entries]
    θs = [e.θ for e in clog.entries]
    ρs = [e.ρ for e in clog.entries]
    cvs = [std(κs)/max(mean(κs),1e-8), std(θs)/max(mean(θs),1e-8),
           std(ρs)/max(abs(mean(ρs)),1e-8)]
    return (drift_detected=maximum(cvs) > 0.2, max_cv=maximum(cvs))
end

end  # (no module — included into NeuroSDE)
