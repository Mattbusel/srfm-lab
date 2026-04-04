"""
VolatilitySurface — Volatility surface modeling and calibration for SRFM research.

Implements:
  - SVI parametrization: raw SVI, JW SVI, surface SVI, arbitrage-free constraints
  - SABR model: Hagan formula, exact MC simulation, beta calibration
  - Local volatility: Dupire equation, finite-difference PDE
  - Surface interpolation: bilinear, cubic spline, linear-in-time
  - Greeks on surface: vanna, volga, vomma, term structure slope
  - Arbitrage checks: butterfly, calendar spread
"""
module VolatilitySurface

using LinearAlgebra
using Statistics
using Distributions
using Optim
using DataFrames
using Random

export SVIParams, svi_raw, svi_implied_vol, svi_calibrate, svi_jw_to_raw
export JWSVIParams, svi_jw, svi_arbitrage_free_check
export SABRParams, sabr_hagan_vol, sabr_calibrate, sabr_mc_simulate
export LocalVolSurface, dupire_local_vol, build_local_vol_surface
export VolSurface, build_vol_surface, interpolate_vol
export surface_greeks, vanna_surface, volga_surface
export butterfly_arbitrage_check, calendar_spread_check

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: SVI Parametrization
# ─────────────────────────────────────────────────────────────────────────────

"""
    SVIParams

Raw SVI parameters for a single maturity slice.
Total implied variance: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
where k = log(K/F) is log-moneyness.
"""
struct SVIParams
    a::Float64      # vertical translation (overall vol level)
    b::Float64      # angle (ATM curvature)
    rho::Float64    # asymmetry (-1 < rho < 1)
    m::Float64      # horizontal translation (ATM)
    sigma::Float64  # smoothness at the kink (sigma > 0)
    T::Float64      # maturity (for computing implied vol from total variance)
end

"""
    svi_raw(k::Float64, p::SVIParams) -> Float64

Evaluate raw SVI total implied variance w(k) at log-moneyness k.
"""
function svi_raw(k::Float64, p::SVIParams)
    x = k - p.m
    return p.a + p.b * (p.rho * x + sqrt(x^2 + p.sigma^2))
end

"""
    svi_raw(k::Vector{Float64}, p::SVIParams) -> Vector{Float64}

Vectorized SVI evaluation.
"""
svi_raw(k::Vector{Float64}, p::SVIParams) = [svi_raw(ki, p) for ki in k]

"""
    svi_implied_vol(k::Float64, p::SVIParams) -> Float64

Convert SVI total variance to Black-Scholes implied vol.
sigma_BS = sqrt(w(k) / T)
"""
function svi_implied_vol(k::Float64, p::SVIParams)
    w = svi_raw(k, p)
    if w <= 0 || p.T <= 0
        return 0.0
    end
    return sqrt(max(w, 0.0) / p.T)
end

svi_implied_vol(k::Vector{Float64}, p::SVIParams) = [svi_implied_vol(ki, p) for ki in k]

"""
    svi_no_butterfly_arbitrage(p::SVIParams) -> Bool

Check SVI no-butterfly-arbitrage condition.
Necessary condition: g(k) >= 0 for all k, where
g(k) = (1 - k*w'(k)/(2w(k)))^2 - w'(k)^2/4*(1/w(k) + 1/4) + w''(k)/2
"""
function svi_no_butterfly_arbitrage(p::SVIParams; n_points::Int=200)
    k_grid = range(-4.0, 4.0, length=n_points)
    for k in k_grid
        x = k - p.m
        denom = sqrt(x^2 + p.sigma^2)

        w = svi_raw(k, p)
        wp = p.b * (p.rho + x / denom)  # dw/dk
        wpp = p.b * p.sigma^2 / denom^3  # d^2w/dk^2

        if w <= 0
            return false
        end

        g = (1.0 - k * wp / (2.0 * w))^2 - wp^2 / 4.0 * (1.0 / w + 0.25) + wpp / 2.0
        if g < -1e-8
            return false
        end
    end
    return true
end

"""
    svi_calibrate(k::Vector{Float64}, market_vols::Vector{Float64}, T::Float64;
                  n_trials=20) -> SVIParams

Calibrate SVI parameters to market implied volatilities using least-squares.
Uses multiple random restarts to avoid local minima.
"""
function svi_calibrate(k::Vector{Float64}, market_vols::Vector{Float64}, T::Float64;
                        n_trials::Int=20, rng::AbstractRNG=Random.GLOBAL_RNG)
    market_var = market_vols .^ 2 .* T

    atm_var = market_var[argmin(abs.(k))]
    var_range = maximum(market_var) - minimum(market_var)

    best_params = nothing
    best_error = Inf

    for trial in 1:n_trials
        # Random initialization
        a0 = atm_var * (0.5 + rand(rng))
        b0 = 0.1 * (0.5 + rand(rng))
        rho0 = -0.5 + rand(rng)
        m0 = 0.1 * randn(rng)
        sigma0 = 0.1 + 0.2 * rand(rng)

        x0 = [a0, b0, rho0, m0, sigma0]

        function obj(x)
            a, b, rho, m, sigma = x
            # Constraints
            if b < 0 || sigma <= 0 || abs(rho) >= 1
                return 1e10
            end
            if a + b * sigma * sqrt(1 - rho^2) < 0  # negative variance check
                return 1e10
            end
            p = SVIParams(a, b, rho, m, sigma, T)
            pred_var = [svi_raw(ki, p) for ki in k]
            return sum((pred_var .- market_var) .^ 2)
        end

        try
            result = optimize(obj, x0, NelderMead();
                options=Optim.Options(iterations=5000, f_tol=1e-12))
            err = Optim.minimum(result)
            if err < best_error
                x_opt = Optim.minimizer(result)
                best_error = err
                best_params = SVIParams(x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], T)
            end
        catch
            continue
        end
    end

    if isnothing(best_params)
        # Fallback: flat vol surface
        return SVIParams(atm_var, 0.1, -0.3, 0.0, 0.1, T)
    end

    return best_params
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: JW SVI Parametrization
# ─────────────────────────────────────────────────────────────────────────────

"""
    JWSVIParams

Jim Gatheral's "quasi-explicit" (JW) SVI parametrization.
More natural for term-structure analysis.

# Fields
- `v::Float64`: ATM variance (sigma_ATM^2 * T)
- `psi::Float64`: ATM skew (d sigma_BS / d k at k=0)
- `p::Float64`: slope of left wing
- `c::Float64`: slope of right wing
- `v_tilde::Float64`: minimum variance
- `T::Float64`: maturity
"""
struct JWSVIParams
    v::Float64     # ATM total variance
    psi::Float64   # ATM skew
    p::Float64     # slope of left wing (negative)
    c::Float64     # slope of right wing (positive)
    v_tilde::Float64  # minimum implied variance
    T::Float64
end

"""
    svi_jw_to_raw(jw::JWSVIParams) -> SVIParams

Convert JW SVI to raw SVI parametrization.
"""
function svi_jw_to_raw(jw::JWSVIParams)
    v = jw.v
    psi = jw.psi
    p = jw.p
    c = jw.c
    vt = jw.v_tilde
    T = jw.T

    w = v  # ATM total variance
    b = sqrt(w) / 2.0 * (c + p)
    if b < 1e-10
        b = 1e-10
    end
    rho = 1.0 - p * sqrt(w) / b
    rho = clamp(rho, -0.999, 0.999)
    beta = rho - 2.0 * psi * sqrt(w) / b
    beta = clamp(beta, -0.999, 0.999)
    alpha = sign(beta) * sqrt(1.0 / beta^2 - 1.0)

    m_val = (w - vt) / (b * (-rho + sign(alpha) * sqrt(1.0 + alpha^2) - alpha * beta))
    if !isfinite(m_val)
        m_val = 0.0
    end
    sigma_val = alpha * m_val
    if sigma_val <= 0
        sigma_val = 0.001
    end
    a_val = vt - b * sigma_val * sqrt(1.0 - rho^2)

    return SVIParams(a_val, b, rho, m_val, sigma_val, T)
end

"""
    svi_jw(k::Float64, jw::JWSVIParams) -> Float64

Evaluate total variance using JW parametrization.
"""
function svi_jw(k::Float64, jw::JWSVIParams)
    raw = svi_jw_to_raw(jw)
    return svi_raw(k, raw)
end

"""
    svi_arbitrage_free_check(slices::Vector{SVIParams}) -> NamedTuple

Check both butterfly and calendar-spread arbitrage across the vol surface.
"""
function svi_arbitrage_free_check(slices::Vector{SVIParams})
    n = length(slices)
    butterfly_free = [svi_no_butterfly_arbitrage(s) for s in slices]

    # Calendar spread: w(k, T2) >= w(k, T1) for T2 > T1
    calendar_free = trues(n - 1)
    k_grid = range(-3.0, 3.0, length=100)
    sort_idx = sortperm([s.T for s in slices])
    sorted = slices[sort_idx]

    for i in 1:(n-1)
        for k in k_grid
            w1 = svi_raw(k, sorted[i])
            w2 = svi_raw(k, sorted[i+1])
            if w2 < w1 - 1e-8
                calendar_free[i] = false
                break
            end
        end
    end

    return (
        butterfly_free=butterfly_free,
        calendar_free=calendar_free,
        all_clear=all(butterfly_free) && all(calendar_free)
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: SABR Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    SABRParams

SABR model parameters:
dF = sigma * F^beta * dW1
dsigma = alpha * sigma * dW2
where E[dW1*dW2] = rho*dt
"""
struct SABRParams
    alpha::Float64   # vol of vol
    beta::Float64    # CEV exponent (0 <= beta <= 1)
    rho::Float64     # correlation
    nu::Float64      # initial vol (renamed from sigma to avoid clash)
    T::Float64       # time to maturity
    F::Float64       # forward price
end

"""
    sabr_hagan_vol(K::Float64, p::SABRParams) -> Float64

Compute SABR implied volatility using Hagan et al. (2002) formula.
"""
function sabr_hagan_vol(K::Float64, p::SABRParams)
    F = p.F
    alpha = p.alpha
    beta = p.beta
    rho = p.rho
    nu = p.nu
    T = p.T

    if K <= 0 || F <= 0
        return alpha
    end

    if abs(F - K) < 1e-10
        # ATM formula
        FK_beta = F^(1.0 - beta)
        term1 = alpha / FK_beta
        term2 = 1.0 + ((1 - beta)^2 / 24 * alpha^2 / FK_beta^2 +
                        rho * beta * nu * alpha / (4 * FK_beta) +
                        (2 - 3 * rho^2) / 24 * nu^2) * T
        return term1 * term2
    end

    # General formula
    FK = F * K
    FK_mid_beta = FK^((1.0 - beta) / 2.0)
    log_FK = log(F / K)

    # z parameter
    z = (nu / alpha) * FK_mid_beta * log_FK

    # chi(z)
    chi = log((sqrt(1.0 - 2.0 * rho * z + z^2) + z - rho) / (1.0 - rho))

    if abs(chi) < 1e-10
        z_over_chi = 1.0
    else
        z_over_chi = z / chi
    end

    # Numerator and denominator terms
    numer = alpha * z_over_chi
    denom = FK_mid_beta * (1.0 + (1.0 - beta)^2 / 24.0 * log_FK^2 +
                            (1.0 - beta)^4 / 1920.0 * log_FK^4)

    correction = 1.0 + ((1.0 - beta)^2 * alpha^2 / (24.0 * FK_mid_beta^2) +
                         rho * beta * nu * alpha / (4.0 * FK_mid_beta) +
                         (2.0 - 3.0 * rho^2) / 24.0 * nu^2) * T

    return numer / denom * correction
end

"""
    sabr_hagan_vol(K::Vector{Float64}, p::SABRParams) -> Vector{Float64}

Vectorized SABR implied vol.
"""
sabr_hagan_vol(K::Vector{Float64}, p::SABRParams) = [sabr_hagan_vol(Ki, p) for Ki in K]

"""
    sabr_calibrate(K::Vector{Float64}, market_vols::Vector{Float64},
                   F::Float64, T::Float64; beta=0.5) -> SABRParams

Calibrate SABR parameters to market implied vols (with fixed beta).
"""
function sabr_calibrate(K::Vector{Float64}, market_vols::Vector{Float64},
                         F::Float64, T::Float64; beta::Float64=0.5,
                         rng::AbstractRNG=Random.GLOBAL_RNG)
    # ATM vol estimate for initialization
    atm_idx = argmin(abs.(K .- F))
    atm_vol = market_vols[atm_idx]

    # Initial guess
    alpha0 = atm_vol * F^(1 - beta)
    rho0 = -0.3
    nu0 = 0.4

    function objective(x)
        alpha, rho, nu = x
        if abs(rho) >= 1 || alpha <= 0 || nu <= 0
            return 1e10
        end
        p = SABRParams(alpha, beta, rho, nu, T, F)
        try
            model_vols = sabr_hagan_vol(K, p)
            return sum((model_vols .- market_vols) .^ 2)
        catch
            return 1e10
        end
    end

    result = optimize(objective, [alpha0, rho0, nu0], NelderMead();
        options=Optim.Options(iterations=5000, f_tol=1e-12))

    x_opt = Optim.minimizer(result)
    return SABRParams(
        clamp(x_opt[1], 1e-6, 10.0),
        beta,
        clamp(x_opt[2], -0.999, 0.999),
        clamp(x_opt[3], 1e-6, 10.0),
        T, F
    )
end

"""
    sabr_beta_calibrate(K::Vector{Float64}, market_vols::Vector{Float64},
                        F::Float64, T::Float64) -> SABRParams

Calibrate all four SABR parameters including beta.
"""
function sabr_beta_calibrate(K::Vector{Float64}, market_vols::Vector{Float64},
                              F::Float64, T::Float64)
    # Estimate beta from log-log regression of ATM vol vs forward
    # (requires multiple expiries; here use a single-expiry heuristic)
    # Try beta = 0, 0.5, 1.0 and pick best
    betas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best = nothing
    best_err = Inf

    for b in betas
        p = sabr_calibrate(K, market_vols, F, T; beta=b)
        model_vols = sabr_hagan_vol(K, p)
        err = sum((model_vols .- market_vols) .^ 2)
        if err < best_err
            best_err = err
            best = p
        end
    end

    return best
end

"""
    sabr_mc_simulate(p::SABRParams, N::Int, n_paths::Int;
                     rng=Random.GLOBAL_RNG) -> NamedTuple

Exact Monte Carlo simulation of SABR process.
Uses log-Euler scheme for sigma and reflection for F (ensures F >= 0).
"""
function sabr_mc_simulate(p::SABRParams, N::Int, n_paths::Int;
                            rng::AbstractRNG=Random.GLOBAL_RNG)
    dt = p.T / N
    alpha = p.alpha
    beta = p.beta
    rho = p.rho
    nu = p.nu
    F0 = p.F

    F_paths = zeros(n_paths, N + 1)
    sig_paths = zeros(n_paths, N + 1)
    F_paths[:, 1] .= F0
    sig_paths[:, 1] .= nu

    for t in 1:N
        z1 = randn(rng, n_paths)
        z2 = randn(rng, n_paths)
        # Correlated BMs
        dW1 = z1 .* sqrt(dt)
        dW2 = (rho .* z1 .+ sqrt(1 - rho^2) .* z2) .* sqrt(dt)

        F_prev = max.(F_paths[:, t], 0.0)
        sig_prev = max.(sig_paths[:, t], 1e-10)

        # Update F: dF = sigma * F^beta * dW1
        F_new = F_prev .+ sig_prev .* F_prev .^ beta .* dW1 .-
                0.5 .* sig_prev .^ 2 .* beta .* F_prev .^ (2 * beta - 1) .* dt
        F_paths[:, t+1] = max.(F_new, 0.0)

        # Update sigma: log-normal SDE (exact)
        sig_new = sig_prev .* exp.((- 0.5 * alpha^2) .* dt .+ alpha .* dW2)
        sig_paths[:, t+1] = sig_new
    end

    return (F=F_paths, sigma=sig_paths)
end

"""
    sabr_implied_vol_mc(p::SABRParams, K::Float64, N::Int, n_paths::Int;
                        rng=Random.GLOBAL_RNG) -> Float64

Compute SABR implied vol by MC simulation and Black-Scholes inversion.
"""
function sabr_implied_vol_mc(p::SABRParams, K::Float64, N::Int, n_paths::Int;
                               rng::AbstractRNG=Random.GLOBAL_RNG)
    sim = sabr_mc_simulate(p, N, n_paths; rng=rng)
    F_T = sim.F[:, end]

    payoff = mean(max.(F_T .- K, 0.0))

    # Discount (no rates in SABR)
    # Invert BS formula for implied vol
    F = p.F
    T = p.T
    d = Normal()

    function bs_call(vol)
        if vol <= 0
            return max(F - K, 0.0)
        end
        d1 = (log(F / K) + 0.5 * vol^2 * T) / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        return F * cdf(d, d1) - K * cdf(d, d2)
    end

    # Bisect for implied vol
    vol_lo, vol_hi = 0.001, 5.0
    if bs_call(vol_lo) >= payoff
        return vol_lo
    end
    if bs_call(vol_hi) <= payoff
        return vol_hi
    end

    for _ in 1:100
        vol_mid = (vol_lo + vol_hi) / 2
        if bs_call(vol_mid) > payoff
            vol_hi = vol_mid
        else
            vol_lo = vol_mid
        end
        if abs(vol_hi - vol_lo) < 1e-7
            break
        end
    end

    return (vol_lo + vol_hi) / 2
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Local Volatility Surface (Dupire)
# ─────────────────────────────────────────────────────────────────────────────

"""
    LocalVolSurface

Calibrated local volatility surface.

# Fields
- `K_grid::Vector{Float64}`: Strike grid
- `T_grid::Vector{Float64}`: Maturity grid
- `sigma_loc::Matrix{Float64}`: Local vol matrix (n_K × n_T)
"""
struct LocalVolSurface
    K_grid::Vector{Float64}
    T_grid::Vector{Float64}
    sigma_loc::Matrix{Float64}   # n_K × n_T
end

"""
    dupire_local_vol(
        K::Float64, T::Float64,
        C::Function,          # C(K, T): call price function
        r::Float64=0.0,       # risk-free rate
        q::Float64=0.0        # dividend yield
    ) -> Float64

Compute Dupire local volatility from call price function:
sigma_loc^2(K,T) = (dC/dT + q*C + (r-q)*K*dC/dK) /
                   (0.5 * K^2 * d^2C/dK^2)
"""
function dupire_local_vol(K::Float64, T::Float64, C::Function;
                           r::Float64=0.0, q::Float64=0.0,
                           dK::Float64=0.01, dT::Float64=0.01)
    # Numerical differentiation
    C0 = C(K, T)

    dCdT = (C(K, T + dT) - C(K, max(T - dT, dT/2))) / (2 * dT)
    dCdK = (C(K + dK, T) - C(K - dK, T)) / (2 * dK)
    d2CdK2 = (C(K + dK, T) - 2 * C0 + C(K - dK, T)) / dK^2

    numerator = dCdT + q * C0 + (r - q) * K * dCdK
    denominator = 0.5 * K^2 * d2CdK2

    if denominator <= 1e-12
        return NaN
    end

    lv2 = numerator / denominator
    return lv2 < 0 ? NaN : sqrt(lv2)
end

"""
    build_local_vol_surface(
        K_grid::Vector{Float64},
        T_grid::Vector{Float64},
        implied_vol_fn::Function,   # sigma_BS(K, T) -> Float64
        S0::Float64,
        r::Float64=0.0,
        q::Float64=0.0
    ) -> LocalVolSurface

Build local vol surface from implied vol surface using Dupire equation.
"""
function build_local_vol_surface(
    K_grid::Vector{Float64},
    T_grid::Vector{Float64},
    implied_vol_fn::Function,
    S0::Float64;
    r::Float64=0.0,
    q::Float64=0.0
)
    n_K = length(K_grid)
    n_T = length(T_grid)
    sigma_loc = zeros(n_K, n_T)
    d = Normal()

    # Build call price function from implied vols
    function call_price(K, T)
        sig = implied_vol_fn(K, T)
        if sig <= 0 || T <= 0
            return max(S0 * exp(-q * T) - K * exp(-r * T), 0.0)
        end
        F = S0 * exp((r - q) * T)
        d1 = (log(F / K) + 0.5 * sig^2 * T) / (sig * sqrt(T))
        d2 = d1 - sig * sqrt(T)
        return exp(-r * T) * (F * cdf(d, d1) - K * cdf(d, d2))
    end

    for (j, T) in enumerate(T_grid)
        for (i, K) in enumerate(K_grid)
            lv = dupire_local_vol(K, T, call_price; r=r, q=q,
                                   dK=K * 0.005, dT=T * 0.01)
            sigma_loc[i, j] = isnan(lv) ? implied_vol_fn(K, T) : lv
        end
    end

    return LocalVolSurface(K_grid, T_grid, sigma_loc)
end

"""
    local_vol_pde(lvs::LocalVolSurface, S0::Float64, K::Float64, T::Float64;
                   r=0.0, n_S=200, n_T=100) -> Float64

Price a European call using local volatility PDE (finite difference).
Solves: dC/dt + 0.5 * sigma_loc^2 * S^2 * d^2C/dS^2 + r*S*dC/dS - r*C = 0
"""
function local_vol_pde(lvs::LocalVolSurface, S0::Float64, K::Float64, T::Float64;
                        r::Float64=0.0, n_S::Int=200, n_T::Int=100)
    S_max = S0 * 5.0
    S_min = S0 * 0.01
    dS = (S_max - S_min) / (n_S - 1)
    dt = T / n_T
    S_grid = [S_min + (i - 1) * dS for i in 1:n_S]

    # Terminal condition: payoff
    C = max.(S_grid .- K, 0.0)

    # Backward time-stepping (implicit Crank-Nicolson)
    for t_step in n_T:-1:1
        t = t_step * dt

        # Build tridiagonal matrix
        alpha_vec = zeros(n_S)
        beta_vec = zeros(n_S)
        gamma_vec = zeros(n_S)

        for i in 2:(n_S-1)
            S_i = S_grid[i]
            # Interpolate local vol
            sig = interpolate_local_vol(lvs, S_i, t)
            a = 0.5 * sig^2 * S_i^2 / dS^2
            b_coef = r * S_i / (2 * dS)
            alpha_vec[i] = -dt * (a - b_coef) / 2
            beta_vec[i] = 1.0 + dt * (a + r / 2)
            gamma_vec[i] = -dt * (a + b_coef) / 2
        end
        beta_vec[1] = 1.0
        beta_vec[n_S] = 1.0

        # RHS
        rhs = copy(C)
        for i in 2:(n_S-1)
            rhs[i] = (1 - dt * (0.5 * (0.5 * interpolate_local_vol(lvs, S_grid[i], t)^2 *
                                        S_grid[i]^2 / dS^2) + r / 2)) * C[i] +
                     dt / 2 * (0.5 * interpolate_local_vol(lvs, S_grid[i], t)^2 *
                                S_grid[i]^2 / dS^2 - r * S_grid[i] / (2 * dS)) * C[i-1] +
                     dt / 2 * (0.5 * interpolate_local_vol(lvs, S_grid[i], t)^2 *
                                S_grid[i]^2 / dS^2 + r * S_grid[i] / (2 * dS)) * C[i+1]
        end
        rhs[1] = 0.0
        rhs[n_S] = S_max - K * exp(-r * (T - t))

        # Solve tridiagonal system
        C_new = solve_tridiagonal(alpha_vec, beta_vec, gamma_vec, rhs)
        C = C_new
    end

    # Interpolate at S0
    idx = searchsortedfirst(S_grid, S0)
    if idx <= 1
        return max(C[1], 0.0)
    end
    if idx > n_S
        return max(C[n_S], 0.0)
    end
    t1, t2 = S_grid[idx-1], S_grid[idx]
    c1, c2 = C[idx-1], C[idx]
    c_interp = c1 + (c2 - c1) * (S0 - t1) / (t2 - t1)
    return max(c_interp, 0.0)
end

"""
    interpolate_local_vol(lvs::LocalVolSurface, S::Float64, T::Float64) -> Float64

Bilinear interpolation of local volatility surface.
"""
function interpolate_local_vol(lvs::LocalVolSurface, S::Float64, T::Float64)
    K_grid = lvs.K_grid
    T_grid = lvs.T_grid
    sig = lvs.sigma_loc

    # Clamp to grid bounds
    S = clamp(S, K_grid[1], K_grid[end])
    T = clamp(T, T_grid[1], T_grid[end])

    # Find surrounding indices
    ik = searchsortedfirst(K_grid, S)
    it = searchsortedfirst(T_grid, T)

    ik = clamp(ik, 2, length(K_grid))
    it = clamp(it, 2, length(T_grid))

    K1, K2 = K_grid[ik-1], K_grid[ik]
    T1, T2 = T_grid[it-1], T_grid[it]

    wK = (S - K1) / (K2 - K1)
    wT = (T - T1) / (T2 - T1)

    return (sig[ik-1, it-1] * (1 - wK) * (1 - wT) +
            sig[ik, it-1] * wK * (1 - wT) +
            sig[ik-1, it] * (1 - wK) * wT +
            sig[ik, it] * wK * wT)
end

"""
    solve_tridiagonal(a::Vector, b::Vector, c::Vector, d::Vector) -> Vector

Solve tridiagonal system using Thomas algorithm.
a: lower diagonal, b: main diagonal, c: upper diagonal, d: RHS.
"""
function solve_tridiagonal(a::Vector{Float64}, b::Vector{Float64},
                            c::Vector{Float64}, d::Vector{Float64})
    n = length(d)
    c_prime = zeros(n)
    d_prime = zeros(n)
    x = zeros(n)

    c_prime[1] = c[1] / b[1]
    d_prime[1] = d[1] / b[1]

    for i in 2:n
        m = b[i] - a[i] * c_prime[i-1]
        if abs(m) < 1e-14
            m = 1e-14
        end
        c_prime[i] = c[i] / m
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / m
    end

    x[n] = d_prime[n]
    for i in (n-1):-1:1
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    end

    return x
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Vol Surface Interpolation
# ─────────────────────────────────────────────────────────────────────────────

"""
    VolSurface

Complete implied volatility surface with interpolation capability.
"""
struct VolSurface
    K_grid::Vector{Float64}     # strike grid
    T_grid::Vector{Float64}     # maturity grid
    vols::Matrix{Float64}       # implied vol matrix n_K × n_T
    F0::Vector{Float64}         # forward prices per maturity
    method::Symbol              # :bilinear, :cubic_strike, :svi
    svi_slices::Union{Vector{SVIParams}, Nothing}
end

"""
    build_vol_surface(
        K_grid, T_grid, market_vols, F0;
        method=:cubic_strike
    ) -> VolSurface

Construct an interpolated volatility surface.
"""
function build_vol_surface(
    K_grid::Vector{Float64},
    T_grid::Vector{Float64},
    market_vols::Matrix{Float64},
    F0::Vector{Float64};
    method::Symbol=:svi
)
    n_K = length(K_grid)
    n_T = length(T_grid)
    @assert size(market_vols) == (n_K, n_T)
    @assert length(F0) == n_T

    svi_slices = nothing

    if method == :svi
        svi_slices = SVIParams[]
        for j in 1:n_T
            T = T_grid[j]
            F = F0[j]
            k_j = log.(K_grid ./ F)
            vols_j = market_vols[:, j]
            push!(svi_slices, svi_calibrate(k_j, vols_j, T))
        end
    end

    return VolSurface(K_grid, T_grid, market_vols, F0, method, svi_slices)
end

"""
    interpolate_vol(vs::VolSurface, K::Float64, T::Float64) -> Float64

Interpolate implied volatility at (K, T).
"""
function interpolate_vol(vs::VolSurface, K::Float64, T::Float64)
    if vs.method == :bilinear
        return _bilinear_vol(vs, K, T)
    elseif vs.method == :svi
        return _svi_vol(vs, K, T)
    else
        return _cubic_strike_vol(vs, K, T)
    end
end

function _bilinear_vol(vs::VolSurface, K::Float64, T::Float64)
    K_grid = vs.K_grid
    T_grid = vs.T_grid
    K = clamp(K, K_grid[1], K_grid[end])
    T = clamp(T, T_grid[1], T_grid[end])

    ik = clamp(searchsortedfirst(K_grid, K), 2, length(K_grid))
    it = clamp(searchsortedfirst(T_grid, T), 2, length(T_grid))

    K1, K2 = K_grid[ik-1], K_grid[ik]
    T1, T2 = T_grid[it-1], T_grid[it]
    wK = (K - K1) / (K2 - K1)
    wT = (T - T1) / (T2 - T1)

    v = vs.vols
    return (v[ik-1, it-1] * (1-wK)*(1-wT) + v[ik, it-1]*wK*(1-wT) +
            v[ik-1, it]*(1-wK)*wT + v[ik, it]*wK*wT)
end

function _svi_vol(vs::VolSurface, K::Float64, T::Float64)
    slices = vs.svi_slices
    if isnothing(slices)
        return _bilinear_vol(vs, K, T)
    end

    T = clamp(T, vs.T_grid[1], vs.T_grid[end])
    T_grid = vs.T_grid

    it = clamp(searchsortedfirst(T_grid, T), 2, length(T_grid))
    T1, T2 = T_grid[it-1], T_grid[it]
    wT = (T - T1) / (T2 - T1)

    # Interpolate total variance linearly in T
    F1 = vs.F0[it-1]
    F2 = vs.F0[it]
    F = F1 * (1 - wT) + F2 * wT  # linear interpolation of forward

    k1 = log(K / F1)
    k2 = log(K / F2)

    w1 = svi_raw(k1, slices[it-1])
    w2 = svi_raw(k2, slices[it])

    # Linear interpolation of total variance
    w = w1 * (1 - wT) + w2 * wT
    if w <= 0 || T <= 0
        return 0.01
    end
    return sqrt(w / T)
end

function _cubic_strike_vol(vs::VolSurface, K::Float64, T::Float64)
    # Piecewise cubic in strike, linear in time
    T = clamp(T, vs.T_grid[1], vs.T_grid[end])
    T_grid = vs.T_grid
    K_grid = vs.K_grid

    it = clamp(searchsortedfirst(T_grid, T), 2, length(T_grid))
    T1, T2 = T_grid[it-1], T_grid[it]
    wT = (T - T1) / (T2 - T1)

    # Cubic spline in strike for each maturity slice
    function cs_interp(K_val, vols_slice)
        K_c = clamp(K_val, K_grid[1], K_grid[end])
        ik = clamp(searchsortedfirst(K_grid, K_c), 2, length(K_grid))
        K1, K2 = K_grid[ik-1], K_grid[ik]
        v1, v2 = vols_slice[ik-1], vols_slice[ik]
        # Simple linear for now (full cubic needs endpoint derivatives)
        wK = (K_c - K1) / (K2 - K1)
        return v1 * (1-wK) + v2 * wK
    end

    v1 = cs_interp(K, vs.vols[:, it-1])
    v2 = cs_interp(K, vs.vols[:, it])
    return v1 * (1 - wT) + v2 * wT
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Greeks on Volatility Surface
# ─────────────────────────────────────────────────────────────────────────────

"""
    surface_greeks(vs::VolSurface, S::Float64, K::Float64, T::Float64,
                   r::Float64, q::Float64) -> NamedTuple

Compute full set of Black-Scholes Greeks including vol surface Greeks:
delta, gamma, vega, theta, rho, vanna, volga, vomma
"""
function surface_greeks(vs::VolSurface, S::Float64, K::Float64, T::Float64,
                        r::Float64, q::Float64)
    sig = interpolate_vol(vs, K, T)

    if sig <= 0 || T <= 0
        return (delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho_greek=0.0,
                vanna=0.0, volga=0.0, vomma=0.0, sigma=sig)
    end

    d = Normal()
    F = S * exp((r - q) * T)
    d1 = (log(S / K) + (r - q + 0.5 * sig^2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    pdf_d1 = pdf(d, d1)

    # Standard Greeks
    delta = exp(-q * T) * cdf(d, d1)
    gamma = exp(-q * T) * pdf_d1 / (S * sig * sqrt(T))
    vega = S * exp(-q * T) * pdf_d1 * sqrt(T)
    theta = (-S * exp(-q * T) * pdf_d1 * sig / (2 * sqrt(T)) -
              r * K * exp(-r * T) * cdf(d, d2) +
              q * S * exp(-q * T) * cdf(d, d1))

    rho_greek = K * T * exp(-r * T) * cdf(d, d2)

    # Vol surface Greeks
    # Vanna = d(delta)/d(sigma) = -exp(-q*T) * d2 * pdf(d1) / sigma
    vanna = -exp(-q * T) * pdf_d1 * d2 / sig

    # Volga = d(vega)/d(sigma) = vega * d1 * d2 / sigma
    volga = vega * d1 * d2 / sig

    # Vomma = second derivative of price w.r.t. vol = vega * (d1*d2 / sigma)
    vomma = volga  # vomma = volga by definition

    return (
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho_greek=rho_greek,
        vanna=vanna,
        volga=volga,
        vomma=vomma,
        sigma=sig,
        d1=d1,
        d2=d2
    )
end

"""
    vanna_surface(vs::VolSurface, S::Float64, T::Float64,
                  r::Float64, q::Float64;
                  K_range=nothing) -> DataFrame

Compute vanna across strike range for a given maturity.
"""
function vanna_surface(vs::VolSurface, S::Float64, T::Float64,
                        r::Float64, q::Float64;
                        K_range::Union{Nothing,AbstractRange}=nothing)
    if isnothing(K_range)
        K_range = range(S * 0.7, S * 1.3, length=50)
    end

    results = DataFrame(K=Float64[], vanna=Float64[], volga=Float64[],
                         sigma=Float64[], delta=Float64[])
    for K in K_range
        g = surface_greeks(vs, S, K, T, r, q)
        push!(results, (K=K, vanna=g.vanna, volga=g.volga,
                         sigma=g.sigma, delta=g.delta))
    end
    return results
end

"""
    volga_surface(vs::VolSurface, S::Float64, T::Float64, r::Float64, q::Float64;
                  K_range=nothing) -> DataFrame
"""
function volga_surface(vs::VolSurface, S::Float64, T::Float64, r::Float64, q::Float64;
                        K_range::Union{Nothing,AbstractRange}=nothing)
    return vanna_surface(vs, S, T, r, q; K_range=K_range)
end

"""
    term_structure_slope(vs::VolSurface, K_frac::Float64=1.0) -> DataFrame

Compute term structure slope (ATM vol change per unit T).
"""
function term_structure_slope(vs::VolSurface, K_frac::Float64=1.0)
    T_grid = vs.T_grid
    n_T = length(T_grid)
    slopes = zeros(n_T - 1)
    for i in 1:(n_T-1)
        T1 = T_grid[i]
        T2 = T_grid[i+1]
        F1 = vs.F0[i]
        F2 = vs.F0[i+1]
        K1 = F1 * K_frac
        K2 = F2 * K_frac
        v1 = interpolate_vol(vs, K1, T1)
        v2 = interpolate_vol(vs, K2, T2)
        slopes[i] = (v2 - v1) / (T2 - T1)
    end
    T_mids = [(T_grid[i] + T_grid[i+1]) / 2 for i in 1:n_T-1]
    return DataFrame(T=T_mids, slope=slopes)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Arbitrage Checks
# ─────────────────────────────────────────────────────────────────────────────

"""
    butterfly_arbitrage_check(
        K_grid::Vector{Float64},
        vols::Vector{Float64},
        T::Float64,
        F::Float64,
        r::Float64=0.0
    ) -> NamedTuple

Check butterfly arbitrage: call spreads must be decreasing in K.
Also verifies density is positive: d^2C/dK^2 >= 0.
"""
function butterfly_arbitrage_check(
    K_grid::Vector{Float64},
    vols::Vector{Float64},
    T::Float64,
    F::Float64,
    r::Float64=0.0
)
    n = length(K_grid)
    @assert length(vols) == n

    d = Normal()
    function call_price(K, vol)
        if vol <= 0 || T <= 0
            return max(F - K, 0.0) * exp(-r * T)
        end
        d1 = (log(F / K) + 0.5 * vol^2 * T) / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        return exp(-r * T) * (F * cdf(d, d1) - K * cdf(d, d2))
    end

    prices = [call_price(K_grid[i], vols[i]) for i in 1:n]

    # Check call spread: C(K1) >= C(K2) for K1 < K2
    call_spread_ok = trues(n - 1)
    for i in 1:(n-1)
        if prices[i] < prices[i+1] - 1e-8
            call_spread_ok[i] = false
        end
    end

    # Check butterfly: C(K1) - 2*C(K2) + C(K3) >= 0
    butterfly_ok = trues(n - 2)
    for i in 2:(n-1)
        w1 = (K_grid[i+1] - K_grid[i]) / (K_grid[i+1] - K_grid[i-1])
        w2 = (K_grid[i] - K_grid[i-1]) / (K_grid[i+1] - K_grid[i-1])
        butterfly = prices[i-1] * w1 - prices[i] + prices[i+1] * w2
        if butterfly < -1e-8
            butterfly_ok[i-1] = false
        end
    end

    return (
        call_spread_ok=call_spread_ok,
        butterfly_ok=butterfly_ok,
        prices=prices,
        no_call_spread_arb=all(call_spread_ok),
        no_butterfly_arb=all(butterfly_ok)
    )
end

"""
    calendar_spread_check(
        T_grid::Vector{Float64},
        slices::Vector{SVIParams},
        K_ref::Vector{Float64}
    ) -> NamedTuple

Check calendar spread arbitrage: total variance must be non-decreasing in T.
"""
function calendar_spread_check(
    T_grid::Vector{Float64},
    slices::Vector{SVIParams},
    K_ref::Vector{Float64}
)
    n_T = length(T_grid)
    n_K = length(K_ref)
    violations = Int[]

    sort_idx = sortperm(T_grid)
    sorted_T = T_grid[sort_idx]
    sorted_slices = slices[sort_idx]

    for i in 1:(n_T-1)
        for k in K_ref
            w1 = svi_raw(k, sorted_slices[i])
            w2 = svi_raw(k, sorted_slices[i+1])
            if w2 < w1 - 1e-8
                push!(violations, i)
                break
            end
        end
    end

    return (
        no_calendar_arb=isempty(violations),
        violation_indices=violations,
        n_violations=length(violations)
    )
end

"""
    full_surface_arbitrage_check(vs::VolSurface) -> NamedTuple

Run complete arbitrage checks on a full vol surface.
"""
function full_surface_arbitrage_check(vs::VolSurface)
    K_grid = vs.K_grid
    T_grid = vs.T_grid
    F0 = vs.F0

    butterfly_results = []
    for j in eachindex(T_grid)
        T = T_grid[j]
        vols_j = vs.vols[:, j]
        F = F0[j]
        check = butterfly_arbitrage_check(K_grid, vols_j, T, F)
        push!(butterfly_results, check)
    end

    calendar_violations = Int[]
    for i in eachindex(K_grid)
        for j in 1:(length(T_grid)-1)
            T1, T2 = T_grid[j], T_grid[j+1]
            v1, v2 = vs.vols[i, j], vs.vols[i, j+1]
            w1, w2 = v1^2 * T1, v2^2 * T2
            if w2 < w1 - 1e-8
                push!(calendar_violations, j)
                break
            end
        end
    end

    no_butterfly = all(r.no_butterfly_arb for r in butterfly_results)
    no_calendar = isempty(calendar_violations)

    return (
        butterfly_results=butterfly_results,
        calendar_violations=calendar_violations,
        no_butterfly_arb=no_butterfly,
        no_calendar_arb=no_calendar,
        surface_arbitrage_free=no_butterfly && no_calendar
    )
end

end # module VolatilitySurface
