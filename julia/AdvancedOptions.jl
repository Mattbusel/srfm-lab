"""
    AdvancedOptions

Advanced options pricing and volatility surface analytics for the SRFM quantitative
trading system. Implements Heston, SABR, Merton jump-diffusion, Dupire local vol,
variance swaps, volatility cones, and vanna-volga exotic approximations.

All pricing functions accept scalar or array inputs and return arrays.
"""
module AdvancedOptions

using LinearAlgebra
using Statistics
using Distributions

export heston_char_func, heston_call_price, heston_put_price
export gauss_legendre_nodes_weights
export sabr_implied_vol, sabr_normal_vol, calibrate_sabr
export variance_swap_fair_strike, variance_swap_replication_weights
export volatility_cone
export svi_params_to_local_vol, dupire_local_vol
export merton_char_func, merton_call_price, merton_put_price
export vanna, volga, compute_vanna_volga_price
export black_scholes_call, black_scholes_put, bs_delta, bs_vega, bs_gamma

# ---------------------------------------------------------------------------
# Black-Scholes baseline functions (used as reference and building blocks)
# ---------------------------------------------------------------------------

"""
    black_scholes_call(S, K, r, q, sigma, T)

Compute Black-Scholes European call price.

# Arguments
- `S`     : spot price
- `K`     : strike price
- `r`     : risk-free rate
- `q`     : continuous dividend yield
- `sigma` : implied volatility
- `T`     : time to expiry in years
"""
function black_scholes_call(S::Real, K::Real, r::Real, q::Real, sigma::Real, T::Real)::Float64
    T <= 0.0 && return max(S * exp(-q * T) - K * exp(-r * T), 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    N = Normal()
    return S * exp(-q * T) * cdf(N, d1) - K * exp(-r * T) * cdf(N, d2)
end

function black_scholes_call(S, K, r, q, sigma, T)
    return black_scholes_call.(S, K, r, q, sigma, T)
end

"""
    black_scholes_put(S, K, r, q, sigma, T)

Compute Black-Scholes European put price via put-call parity.
"""
function black_scholes_put(S::Real, K::Real, r::Real, q::Real, sigma::Real, T::Real)::Float64
    call = black_scholes_call(S, K, r, q, sigma, T)
    return call - S * exp(-q * T) + K * exp(-r * T)
end

function black_scholes_put(S, K, r, q, sigma, T)
    return black_scholes_put.(S, K, r, q, sigma, T)
end

"""
    bs_delta(S, K, r, q, sigma, T; option_type=:call)

Black-Scholes delta.
"""
function bs_delta(S::Real, K::Real, r::Real, q::Real, sigma::Real, T::Real; option_type::Symbol=:call)::Float64
    T <= 0.0 && return option_type == :call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    N = Normal()
    if option_type == :call
        return exp(-q * T) * cdf(N, d1)
    else
        return exp(-q * T) * (cdf(N, d1) - 1.0)
    end
end

"""
    bs_vega(S, K, r, q, sigma, T)

Black-Scholes vega (dV/dsigma).
"""
function bs_vega(S::Real, K::Real, r::Real, q::Real, sigma::Real, T::Real)::Float64
    T <= 0.0 && return 0.0
    d1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    N = Normal()
    return S * exp(-q * T) * pdf(N, d1) * sqrt(T)
end

"""
    bs_gamma(S, K, r, q, sigma, T)

Black-Scholes gamma (d2V/dS2).
"""
function bs_gamma(S::Real, K::Real, r::Real, q::Real, sigma::Real, T::Real)::Float64
    T <= 0.0 && return 0.0
    d1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    N = Normal()
    return exp(-q * T) * pdf(N, d1) / (S * sigma * sqrt(T))
end

# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature
# ---------------------------------------------------------------------------

"""
    gauss_legendre_nodes_weights(n)

Return (nodes, weights) for n-point Gauss-Legendre quadrature on [-1, 1].
Uses the Golub-Welsch algorithm via symmetric tridiagonal eigenvalue problem.
"""
function gauss_legendre_nodes_weights(n::Int)
    # Build symmetric tridiagonal matrix
    beta = [i / sqrt(4.0 * i^2 - 1.0) for i in 1:(n - 1)]
    T = SymTridiagonal(zeros(n), beta)
    eig = eigen(T)
    nodes = eig.values
    weights = 2.0 .* eig.vectors[1, :] .^ 2
    # Sort ascending
    idx = sortperm(nodes)
    return nodes[idx], weights[idx]
end

"""
    gl_integrate(f, a, b, n)

Integrate f from a to b using n-point Gauss-Legendre quadrature.
"""
function gl_integrate(f::Function, a::Real, b::Real, n::Int=64)::Float64
    nodes, weights = gauss_legendre_nodes_weights(n)
    # Transform from [-1,1] to [a,b]
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    return half * sum(weights[i] * f(mid + half * nodes[i]) for i in 1:n)
end

# ---------------------------------------------------------------------------
# Heston stochastic volatility model
# ---------------------------------------------------------------------------

"""
    heston_char_func(u, S, K, r, q, T, kappa, theta, sigma_v, rho, v0)

Characteristic function of the Heston (1993) model using the Albrecher et al.
(2007) formulation that avoids discontinuities in the complex logarithm.

# Arguments
- `u`       : complex frequency variable
- `S`       : spot price
- `r`       : risk-free rate
- `q`       : dividend yield
- `T`       : time to expiry
- `kappa`   : mean reversion speed
- `theta`   : long-run variance
- `sigma_v` : vol-of-vol
- `rho`     : spot-vol correlation
- `v0`      : initial variance
"""
function heston_char_func(u::Complex{Float64}, S::Real, r::Real, q::Real, T::Real,
                          kappa::Real, theta::Real, sigma_v::Real, rho::Real, v0::Real)::Complex{Float64}
    i = im
    x = log(S)
    a = kappa * theta
    b_val = kappa - rho * sigma_v * u * i
    d = sqrt(b_val^2 + sigma_v^2 * (u^2 + i * u))
    g = (b_val - d) / (b_val + d)
    exp_dT = exp(-d * T)
    C = (r - q) * u * i * T + (a / sigma_v^2) * (
            (b_val - d) * T - 2.0 * log((1.0 - g * exp_dT) / (1.0 - g))
        )
    D = ((b_val - d) / sigma_v^2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    return exp(C + D * v0 + i * u * x)
end

"""
    heston_integrand(phi, S, K, r, q, T, kappa, theta, sigma_v, rho, v0, j)

Internal integrand for Heston option pricing via Gil-Pelaez inversion.
j=1 gives the first integral (for delta-like term), j=2 gives the second.
"""
function heston_integrand(phi::Real, S::Real, K::Real, r::Real, q::Real, T::Real,
                          kappa::Real, theta::Real, sigma_v::Real, rho::Real, v0::Real,
                          j::Int)::Float64
    i = im
    u_j = j == 1 ? 0.5 + phi * im : -0.5 + phi * im
    b_j = j == 1 ? kappa - rho * sigma_v : kappa

    a_coeff = kappa * theta
    xi = b_j - rho * sigma_v * phi * i
    d = sqrt(xi^2 + sigma_v^2 * (phi * i + phi^2))
    g2 = (xi - d) / (xi + d)
    exp_dT = exp(-d * T)

    C = (r - q) * phi * i * T + (a_coeff / sigma_v^2) * (
            (xi - d) * T - 2.0 * log((1.0 - g2 * exp_dT) / (1.0 - g2))
        )
    D = ((xi - d) / sigma_v^2) * ((1.0 - exp_dT) / (1.0 - g2 * exp_dT))

    cf = exp(C + D * v0 + i * phi * log(S))
    integrand_val = real(exp(-i * phi * log(K)) * cf / (i * phi))
    return integrand_val
end

"""
    heston_call_price(S, K, r, q, T, kappa, theta, sigma_v, rho, v0; n_quad=128)

Price a European call option under the Heston (1993) stochastic volatility model
using Gauss-Legendre quadrature for the Gil-Pelaez Fourier inversion.

Returns the call price as a Float64.
"""
function heston_call_price(S::Real, K::Real, r::Real, q::Real, T::Real,
                           kappa::Real, theta::Real, sigma_v::Real, rho::Real, v0::Real;
                           n_quad::Int=128)::Float64
    upper_limit = 200.0
    P1 = 0.5 + (1.0 / pi) * gl_integrate(
        phi -> heston_integrand(phi, S, K, r, q, T, kappa, theta, sigma_v, rho, v0, 1),
        1e-8, upper_limit, n_quad
    )
    P2 = 0.5 + (1.0 / pi) * gl_integrate(
        phi -> heston_integrand(phi, S, K, r, q, T, kappa, theta, sigma_v, rho, v0, 2),
        1e-8, upper_limit, n_quad
    )
    return S * exp(-q * T) * P1 - K * exp(-r * T) * P2
end

"""
    heston_put_price(S, K, r, q, T, kappa, theta, sigma_v, rho, v0; n_quad=128)

Price a European put under Heston via put-call parity.
"""
function heston_put_price(S::Real, K::Real, r::Real, q::Real, T::Real,
                          kappa::Real, theta::Real, sigma_v::Real, rho::Real, v0::Real;
                          n_quad::Int=128)::Float64
    call = heston_call_price(S, K, r, q, T, kappa, theta, sigma_v, rho, v0; n_quad=n_quad)
    return call - S * exp(-q * T) + K * exp(-r * T)
end

# Vectorized versions
function heston_call_price(S, K, r, q, T, kappa, theta, sigma_v, rho, v0; n_quad::Int=128)
    return [heston_call_price(s, k, r, q, t, kappa, theta, sigma_v, rho, v0; n_quad=n_quad)
            for (s, k, t) in zip(S, K, T)]
end

# ---------------------------------------------------------------------------
# SABR model
# ---------------------------------------------------------------------------

"""
    sabr_implied_vol(F, K, T, alpha, beta, rho, nu)

Hagan et al. (2002) SABR lognormal implied volatility approximation.
F is the forward price, K is the strike, T is expiry.
Returns approximate Black lognormal implied volatility.
"""
function sabr_implied_vol(F::Real, K::Real, T::Real,
                          alpha::Real, beta::Real, rho::Real, nu::Real)::Float64
    if abs(F - K) < 1e-10
        # ATM formula
        FK_mid = F
        term1 = alpha / (FK_mid^(1.0 - beta))
        term2 = 1.0 + ((1.0 - beta)^2 / 24.0 * alpha^2 / FK_mid^(2.0 - 2.0 * beta)
                       + 0.25 * rho * beta * nu * alpha / FK_mid^(1.0 - beta)
                       + (2.0 - 3.0 * rho^2) / 24.0 * nu^2) * T
        return term1 * term2
    end

    FK = sqrt(F * K)
    log_FK = log(F / K)

    z = (nu / alpha) * FK^(1.0 - beta) * log_FK
    chi_z = log((sqrt(1.0 - 2.0 * rho * z + z^2) + z - rho) / (1.0 - rho))

    A = alpha / (FK^(1.0 - beta) * (1.0 + (1.0 - beta)^2 / 24.0 * log_FK^2
                                     + (1.0 - beta)^4 / 1920.0 * log_FK^4))
    B = (z / chi_z)
    C = 1.0 + ((1.0 - beta)^2 / 24.0 * alpha^2 / FK^(2.0 - 2.0 * beta)
               + 0.25 * rho * beta * nu * alpha / FK^(1.0 - beta)
               + (2.0 - 3.0 * rho^2) / 24.0 * nu^2) * T

    return A * B * C
end

function sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
    return sabr_implied_vol.(F, K, T, alpha, beta, rho, nu)
end

"""
    sabr_normal_vol(F, K, T, alpha, beta, rho, nu)

Normal (Bachelier) SABR approximation for negative rate environments.
Returns the normal (absolute) implied volatility.
"""
function sabr_normal_vol(F::Real, K::Real, T::Real,
                         alpha::Real, beta::Real, rho::Real, nu::Real)::Float64
    if abs(F - K) < 1e-10
        # ATM normal vol
        return alpha * F^beta * (1.0 + ((beta * (beta - 2.0) * alpha^2 / (24.0 * F^(2.0 - 2.0 * beta)))
                                        + 0.25 * rho * nu * alpha * beta / F^(1.0 - beta)
                                        + (2.0 - 3.0 * rho^2) / 24.0 * nu^2) * T)
    end

    FK = sqrt(F * K)
    log_FK = log(F / K)
    z = (nu / alpha) * FK^(1.0 - beta) * log_FK
    chi_z = log((sqrt(1.0 - 2.0 * rho * z + z^2) + z - rho) / (1.0 - rho))

    A = alpha * (F - K) / (FK^(1.0 - beta) * log_FK) * (z / chi_z)
    B = 1.0 + ((beta * (beta - 2.0) / 24.0 * alpha^2 / FK^(2.0 - 2.0 * beta))
               + 0.25 * rho * nu * alpha * beta / FK^(1.0 - beta)
               + (2.0 - 3.0 * rho^2) / 24.0 * nu^2) * T

    return A * B
end

"""
    sabr_loss(params, F, strikes, T, market_vols, beta)

Internal least-squares loss for SABR calibration (alpha, rho, nu free; beta fixed).
"""
function sabr_loss(params::Vector{Float64}, F::Real, strikes::Vector{Float64},
                   T::Real, market_vols::Vector{Float64}, beta::Real)::Float64
    alpha, rho, nu = params
    # Penalty for out-of-range parameters
    if alpha <= 0.0 || nu <= 0.0 || abs(rho) >= 1.0
        return 1e10
    end
    model_vols = [sabr_implied_vol(F, k, T, alpha, beta, rho, nu) for k in strikes]
    return sum((model_vols .- market_vols) .^ 2)
end

"""
    calibrate_sabr(F, strikes, T, market_vols; beta=0.5)

Calibrate SABR alpha, rho, nu parameters to market implied vols using
Nelder-Mead simplex optimization with beta fixed.

Returns named tuple (alpha, beta, rho, nu, rmse).
"""
function calibrate_sabr(F::Real, strikes::Vector{Float64}, T::Real,
                        market_vols::Vector{Float64}; beta::Real=0.5)
    # Initial guess: ATM vol heuristic
    atm_idx = argmin(abs.(strikes .- F))
    atm_vol = market_vols[atm_idx]
    alpha0 = atm_vol * F^(1.0 - beta)
    rho0 = -0.3
    nu0 = 0.4

    params0 = [alpha0, rho0, nu0]
    best_params = copy(params0)
    best_loss = sabr_loss(params0, F, strikes, T, market_vols, beta)

    # Simple Nelder-Mead via finite difference gradient descent (light implementation)
    step_sizes = [0.01 * alpha0, 0.05, 0.05]
    for iter in 1:500
        improved = false
        for j in 1:3
            for sign in [-1.0, 1.0]
                trial = copy(best_params)
                trial[j] += sign * step_sizes[j]
                loss = sabr_loss(trial, F, strikes, T, market_vols, beta)
                if loss < best_loss
                    best_loss = loss
                    best_params = trial
                    improved = true
                end
            end
        end
        if !improved
            step_sizes .*= 0.7
            all(step_sizes .< 1e-8) && break
        end
    end

    alpha, rho, nu = best_params
    rmse = sqrt(best_loss / length(strikes))
    return (alpha=alpha, beta=beta, rho=rho, nu=nu, rmse=rmse)
end

"""
    sabr_smile(F, T, alpha, beta, rho, nu; n_strikes=50)

Generate the SABR implied vol smile across a range of strikes.
Returns (strikes, implied_vols) as a tuple.
"""
function sabr_smile(F::Real, T::Real, alpha::Real, beta::Real, rho::Real, nu::Real;
                    n_strikes::Int=50)
    strikes = collect(range(0.5 * F, 1.5 * F, length=n_strikes))
    vols = [sabr_implied_vol(F, k, T, alpha, beta, rho, nu) for k in strikes]
    return strikes, vols
end

# ---------------------------------------------------------------------------
# Variance swap pricing
# ---------------------------------------------------------------------------

"""
    variance_swap_fair_strike(S, r, q, T, call_strikes, call_prices, put_strikes, put_prices)

Compute the fair variance strike of a variance swap via static replication
using the Demeterfi et al. (1999) / Carr-Madan log-contract approach.

Returns the annualized variance strike K_var.
"""
function variance_swap_fair_strike(S::Real, r::Real, q::Real, T::Real,
                                    call_strikes::Vector{Float64}, call_prices::Vector{Float64},
                                    put_strikes::Vector{Float64}, put_prices::Vector{Float64})::Float64
    F = S * exp((r - q) * T)

    # Integral over puts (strikes < F)
    put_integral = 0.0
    valid_puts = put_strikes .< F
    if sum(valid_puts) > 1
        ks = put_strikes[valid_puts]
        ps = put_prices[valid_puts]
        # Trapezoid rule: 2/T * integral(P(K)/K^2 dK)
        for i in 2:length(ks)
            dK = ks[i] - ks[i-1]
            put_integral += 0.5 * (ps[i] / ks[i]^2 + ps[i-1] / ks[i-1]^2) * dK
        end
    end

    # Integral over calls (strikes > F)
    call_integral = 0.0
    valid_calls = call_strikes .> F
    if sum(valid_calls) > 1
        ks = call_strikes[valid_calls]
        cs = call_prices[valid_calls]
        for i in 2:length(ks)
            dK = ks[i] - ks[i-1]
            call_integral += 0.5 * (cs[i] / ks[i]^2 + cs[i-1] / ks[i-1]^2) * dK
        end
    end

    K_var = (2.0 / T) * exp(r * T) * (put_integral + call_integral)
    return K_var
end

"""
    variance_swap_replication_weights(S, F, T, strikes)

Compute the static replication portfolio weights (density) for a variance swap.
Returns the weight w(K) = 2/(T*K^2) for each strike, split into put/call sides.
"""
function variance_swap_replication_weights(S::Real, F::Real, T::Real,
                                            strikes::Vector{Float64})::Vector{Float64}
    return [2.0 / (T * k^2) for k in strikes]
end

# ---------------------------------------------------------------------------
# Volatility cone
# ---------------------------------------------------------------------------

"""
    volatility_cone(log_returns, horizons_days; annualization=252)

Compute realized volatility percentiles at multiple horizons to form a vol cone.

# Arguments
- `log_returns`     : vector of daily log returns
- `horizons_days`   : vector of horizon lengths in days (e.g., [5, 10, 21, 63, 126])
- `annualization`   : trading days per year

Returns a named tuple with fields: horizons, p10, p25, p50, p75, p90, current.
"""
function volatility_cone(log_returns::Vector{Float64}, horizons_days::Vector{Int};
                         annualization::Int=252)
    n = length(log_returns)
    p10 = Float64[]
    p25 = Float64[]
    p50 = Float64[]
    p75 = Float64[]
    p90 = Float64[]
    current = Float64[]

    for h in horizons_days
        if h >= n
            push!(p10, NaN); push!(p25, NaN); push!(p50, NaN)
            push!(p75, NaN); push!(p90, NaN); push!(current, NaN)
            continue
        end
        # Rolling realized vol
        rolling_vols = Float64[]
        for i in h:n
            window = log_returns[(i - h + 1):i]
            rv = std(window) * sqrt(annualization)
            push!(rolling_vols, rv)
        end
        push!(p10, quantile(rolling_vols, 0.10))
        push!(p25, quantile(rolling_vols, 0.25))
        push!(p50, quantile(rolling_vols, 0.50))
        push!(p75, quantile(rolling_vols, 0.75))
        push!(p90, quantile(rolling_vols, 0.90))
        # Most recent window
        push!(current, rolling_vols[end])
    end

    return (horizons=horizons_days, p10=p10, p25=p25, p50=p50,
            p75=p75, p90=p90, current=current)
end

# ---------------------------------------------------------------------------
# SVI parametrization and Dupire local volatility
# ---------------------------------------------------------------------------

"""
    svi_total_variance(k, a, b, rho_svi, m, sigma_svi)

Raw SVI (Stochastic Volatility Inspired) parametrization.
Returns total implied variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
where k = log(K/F) is log-moneyness.
"""
function svi_total_variance(k::Real, a::Real, b::Real, rho_svi::Real,
                             m::Real, sigma_svi::Real)::Float64
    inner = sqrt((k - m)^2 + sigma_svi^2)
    return a + b * (rho_svi * (k - m) + inner)
end

"""
    svi_params_to_local_vol(F, r, q, T1, T2, svi_params_T1, svi_params_T2, log_moneyness)

Extract Dupire local volatility from two SVI slices at tenors T1 < T2 using the
Dupire (1994) formula in log-moneyness space.

svi_params = (a, b, rho, m, sigma) tuple.
Returns local vol at the midpoint tenor and given log-moneyness.
"""
function svi_params_to_local_vol(F::Real, r::Real, q::Real,
                                  T1::Real, T2::Real,
                                  svi_T1::NTuple{5, Float64},
                                  svi_T2::NTuple{5, Float64},
                                  log_moneyness::Real)::Float64
    dT = T2 - T1
    dT <= 0.0 && error("T2 must be greater than T1")
    T_mid = 0.5 * (T1 + T2)
    k = log_moneyness

    w1 = svi_total_variance(k, svi_T1...)
    w2 = svi_total_variance(k, svi_T2...)

    # dw/dT (finite difference)
    dw_dT = (w2 - w1) / dT
    dw_dT < 0.0 && return 0.0  # Calendar arbitrage violation

    # dw/dk and d2w/dk2 at T_mid using weighted average SVI
    a_m = 0.5 * (svi_T1[1] + svi_T2[1])
    b_m = 0.5 * (svi_T1[2] + svi_T2[2])
    r_m = 0.5 * (svi_T1[3] + svi_T2[3])
    m_m = 0.5 * (svi_T1[4] + svi_T2[4])
    s_m = 0.5 * (svi_T1[5] + svi_T2[5])

    dk = 1e-4
    w_p = svi_total_variance(k + dk, a_m, b_m, r_m, m_m, s_m)
    w_c = svi_total_variance(k,      a_m, b_m, r_m, m_m, s_m)
    w_m2 = svi_total_variance(k - dk, a_m, b_m, r_m, m_m, s_m)

    dw_dk  = (w_p - w_m2) / (2.0 * dk)
    d2w_dk2 = (w_p - 2.0 * w_c + w_m2) / dk^2

    w_mid = w_c
    # Dupire local variance formula (Gatheral form)
    g_k = (1.0 - k * dw_dk / (2.0 * w_mid))^2 - dw_dk^2 / 4.0 * (1.0 / w_mid + 0.25) + 0.5 * d2w_dk2
    g_k = max(g_k, 1e-8)

    local_var = dw_dT / g_k
    local_var = max(local_var, 0.0)
    return sqrt(local_var / T_mid)
end

"""
    dupire_local_vol(F_grid, T_grid, implied_vols; r=0.0, q=0.0)

Extract a full local volatility surface from a grid of implied vols using
finite differences of the Dupire PDE.

# Arguments
- `F_grid`      : vector of forward price values
- `T_grid`      : vector of expiry times
- `implied_vols`: matrix of implied vols, size (length(F_grid), length(T_grid))

Returns a matrix of local vols of same size.
"""
function dupire_local_vol(F_grid::Vector{Float64}, T_grid::Vector{Float64},
                          implied_vols::Matrix{Float64}; r::Real=0.0, q::Real=0.0)::Matrix{Float64}
    nF = length(F_grid)
    nT = length(T_grid)
    local_vols = zeros(nF, nT)

    for j in 2:(nT - 1)
        T = T_grid[j]
        dT_fwd = T_grid[j+1] - T_grid[j]
        dT_bwd = T_grid[j] - T_grid[j-1]

        for i in 2:(nF - 1)
            K = F_grid[i]
            dK = F_grid[i+1] - F_grid[i-1]
            dK2 = 0.5 * ((F_grid[i+1] - F_grid[i]) + (F_grid[i] - F_grid[i-1]))

            sigma = implied_vols[i, j]
            sigma_Kp = implied_vols[i+1, j]
            sigma_Km = implied_vols[i-1, j]
            sigma_Tp = implied_vols[i, j+1]
            sigma_Tm = implied_vols[i, j-1]

            # Total variance w = sigma^2 * T
            w = sigma^2 * T
            w_Kp = sigma_Kp^2 * T
            w_Km = sigma_Km^2 * T
            w_Tp = sigma_Tp^2 * T_grid[j+1]
            w_Tm = sigma_Tm^2 * T_grid[j-1]

            dw_dT = (w_Tp - w_Tm) / (T_grid[j+1] - T_grid[j-1])
            dw_dK = (w_Kp - w_Km) / dK
            d2w_dK2 = (w_Kp - 2.0 * w + w_Km) / dK2^2

            k_lm = log(K / F_grid[i])  # approx log-moneyness

            g = (1.0 - k_lm / w * dw_dK * K)^2 + 0.5 * K^2 * d2w_dK2 - 0.25 * K^2 * dw_dK^2 * (0.25 + 1.0 / w)
            g = max(g, 1e-8)
            lv2 = dw_dT / g
            local_vols[i, j] = max(sqrt(max(lv2, 0.0)), 0.0)
        end
        # Boundary: copy from neighbors
        local_vols[1, j] = local_vols[2, j]
        local_vols[nF, j] = local_vols[nF-1, j]
    end
    # Boundary in T
    local_vols[:, 1] = local_vols[:, 2]
    local_vols[:, nT] = local_vols[:, nT-1]
    return local_vols
end

# ---------------------------------------------------------------------------
# Merton jump-diffusion model
# ---------------------------------------------------------------------------

"""
    merton_char_func(u, S, r, q, T, sigma, lambda, mu_j, sigma_j)

Characteristic function for the Merton (1976) jump-diffusion model.
Jumps are log-normally distributed with mean mu_j and std sigma_j.

# Arguments
- `lambda`  : Poisson jump intensity (average jumps per year)
- `mu_j`    : mean of log-jump size
- `sigma_j` : std dev of log-jump size
"""
function merton_char_func(u::Complex{Float64}, S::Real, r::Real, q::Real, T::Real,
                          sigma::Real, lambda::Real, mu_j::Real, sigma_j::Real)::Complex{Float64}
    i = im
    k_bar = exp(mu_j + 0.5 * sigma_j^2) - 1.0  # mean percentage jump
    # Drift adjustment so that E[S(T)] = S * exp((r-q)*T)
    mu_adj = r - q - 0.5 * sigma^2 - lambda * k_bar

    log_S = log(S)
    exp_jump_cf = exp(mu_j * i * u - 0.5 * sigma_j^2 * u^2) - 1.0

    exponent = (i * u * (log_S + mu_adj * T)
                - 0.5 * sigma^2 * u^2 * T
                + lambda * T * exp_jump_cf)
    return exp(exponent)
end

"""
    merton_call_price(S, K, r, q, T, sigma, lambda, mu_j, sigma_j; n_terms=30)

Price a European call under the Merton (1976) jump-diffusion model using
the infinite series representation (Poisson mixture of Black-Scholes).
"""
function merton_call_price(S::Real, K::Real, r::Real, q::Real, T::Real,
                           sigma::Real, lambda::Real, mu_j::Real, sigma_j::Real;
                           n_terms::Int=30)::Float64
    k_bar = exp(mu_j + 0.5 * sigma_j^2) - 1.0
    lambda_prime = lambda * (1.0 + k_bar)
    price = 0.0

    for n in 0:n_terms
        # Adjusted parameters for n-jump scenario
        sigma_n = sqrt(sigma^2 + n * sigma_j^2 / T)
        r_n = r - lambda * k_bar + n * (mu_j + 0.5 * sigma_j^2) / T
        # Poisson weight
        pois_weight = exp(-lambda_prime * T) * (lambda_prime * T)^n / factorial(big(n))
        bs_price = black_scholes_call(S, K, r_n, q, sigma_n, T)
        price += Float64(pois_weight) * bs_price
    end
    return price
end

"""
    merton_put_price(S, K, r, q, T, sigma, lambda, mu_j, sigma_j; n_terms=30)

Price a European put under the Merton (1976) jump-diffusion model.
"""
function merton_put_price(S::Real, K::Real, r::Real, q::Real, T::Real,
                          sigma::Real, lambda::Real, mu_j::Real, sigma_j::Real;
                          n_terms::Int=30)::Float64
    call = merton_call_price(S, K, r, q, T, sigma, lambda, mu_j, sigma_j; n_terms=n_terms)
    return call - S * exp(-q * T) + K * exp(-r * T)
end

function merton_call_price(S, K, r, q, T, sigma, lambda, mu_j, sigma_j; n_terms::Int=30)
    return [merton_call_price(s, k, r, q, t, sigma, lambda, mu_j, sigma_j; n_terms=n_terms)
            for (s, k, t) in zip(S, K, T)]
end

# ---------------------------------------------------------------------------
# Greeks: vanna, volga, vanna-volga
# ---------------------------------------------------------------------------

"""
    vanna(S, K, r, q, sigma, T)

Compute vanna = dDelta/dsigma = dVega/dS.
"""
function vanna(S::Real, K::Real, r::Real, q::Real, sigma::Real, T::Real)::Float64
    T <= 0.0 && return 0.0
    d1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    N = Normal()
    return -exp(-q * T) * pdf(N, d1) * d2 / sigma
end

"""
    volga(S, K, r, q, sigma, T)

Compute volga (vomma) = dVega/dsigma = d2V/dsigma2.
"""
function volga(S::Real, K::Real, r::Real, q::Real, sigma::Real, T::Real)::Float64
    T <= 0.0 && return 0.0
    d1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    vega = bs_vega(S, K, r, q, sigma, T)
    return vega * d1 * d2 / sigma
end

"""
    compute_vanna_volga_price(S, K, r, q, T, sigma_atm, sigma_25d_call, sigma_25d_put,
                               exotic_price_bsm; option_type=:call)

Vanna-Volga (Castagna-Mercurio) first-generation approximation for exotic options.
Uses three market instruments: ATM straddle, 25-delta risk reversal, 25-delta butterfly.

# Arguments
- `sigma_atm`      : ATM implied vol
- `sigma_25d_call` : 25-delta call implied vol
- `sigma_25d_put`  : 25-delta put implied vol
- `exotic_price_bsm`: exotic price under flat vol = sigma_atm

Returns the vanna-volga adjusted exotic price.
"""
function compute_vanna_volga_price(S::Real, K::Real, r::Real, q::Real, T::Real,
                                    sigma_atm::Real, sigma_25d_call::Real, sigma_25d_put::Real,
                                    exotic_price_bsm::Real; option_type::Symbol=:call)::Float64
    # Compute strikes for 25-delta instruments
    N = Normal()
    d_atm = (log(S / S) + (r - q + 0.5 * sigma_atm^2) * T) / (sigma_atm * sqrt(T))
    K_atm = S  # ATM forward approximation

    K_25c = S * exp(-quantile(N, 0.75) * sigma_25d_call * sqrt(T) +
                    (r - q + 0.5 * sigma_25d_call^2) * T)
    K_25p = S * exp(-quantile(N, 0.25) * sigma_25d_put * sqrt(T) +
                    (r - q + 0.5 * sigma_25d_put^2) * T)

    # Vanna and volga of exotic
    vanna_x = vanna(S, K, r, q, sigma_atm, T)
    volga_x = volga(S, K, r, q, sigma_atm, T)

    # Vanna and volga of hedging instruments
    vanna_rr = vanna(S, K_25c, r, q, sigma_25d_call, T) - vanna(S, K_25p, r, q, sigma_25d_put, T)
    volga_bf = 0.5 * (volga(S, K_25c, r, q, sigma_25d_call, T) + volga(S, K_25p, r, q, sigma_25d_put, T)) -
               volga(S, K_atm, r, q, sigma_atm, T)

    # Cost of hedging
    cost_rr = black_scholes_call(S, K_25c, r, q, sigma_25d_call, T) -
              black_scholes_put(S, K_25p, r, q, sigma_25d_put, T) -
              black_scholes_call(S, K_25c, r, q, sigma_atm, T) +
              black_scholes_put(S, K_25p, r, q, sigma_atm, T)

    cost_bf = 0.5 * (black_scholes_call(S, K_25c, r, q, sigma_25d_call, T) +
                     black_scholes_put(S, K_25p, r, q, sigma_25d_put, T) -
                     black_scholes_call(S, K_25c, r, q, sigma_atm, T) -
                     black_scholes_put(S, K_25p, r, q, sigma_atm, T))

    # Solve for hedge ratios
    det = vanna_rr * volga_bf - volga(S, K_25c, r, q, sigma_25d_call, T) * vanna(S, K_25p, r, q, sigma_25d_put, T)
    abs(det) < 1e-12 && return exotic_price_bsm

    x_rr = abs(vanna_rr) > 1e-10 ? vanna_x / vanna_rr : 0.0
    x_bf = abs(volga_bf) > 1e-10 ? volga_x / volga_bf : 0.0

    return exotic_price_bsm + x_rr * cost_rr + x_bf * cost_bf
end

# ---------------------------------------------------------------------------
# Utility: implied vol inversion via bisection
# ---------------------------------------------------------------------------

"""
    implied_vol(market_price, S, K, r, q, T; option_type=:call, tol=1e-8)

Compute implied volatility from a market option price using the bisection method.
"""
function implied_vol(market_price::Real, S::Real, K::Real, r::Real, q::Real, T::Real;
                     option_type::Symbol=:call, tol::Real=1e-8)::Float64
    intrinsic = option_type == :call ? max(S * exp(-q*T) - K * exp(-r*T), 0.0) :
                                       max(K * exp(-r*T) - S * exp(-q*T), 0.0)
    market_price <= intrinsic + tol && return 0.0

    lo, hi = 1e-6, 10.0
    pricer = option_type == :call ? black_scholes_call : black_scholes_put

    for _ in 1:200
        mid = 0.5 * (lo + hi)
        p = pricer(S, K, r, q, mid, T)
        if abs(p - market_price) < tol
            return mid
        elseif p < market_price
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

end  # module AdvancedOptions
