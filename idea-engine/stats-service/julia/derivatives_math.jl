# =============================================================================
# derivatives_math.jl — Derivatives Pricing in Pure Julia
# =============================================================================
# Complete options pricing library covering:
#   - Black-Scholes / Black-76 with full Greeks
#   - Implied volatility via Brent's bisection
#   - Heston model (characteristic function + Gil-Pelaez inversion)
#   - FFT option pricing (Carr-Madan algorithm)
#   - Dupire local volatility surface
#   - SABR model: Hagan approximation
#   - Crypto perpetual mechanics
#   - SVI vol surface parametrization
#   - Arbitrage detection (calendar, butterfly)
#
# Julia ≥ 1.10 | No external packages (stdlib only)
# =============================================================================

module DerivativesMath

using Statistics
using LinearAlgebra

export bs_price, bs_greeks, black76_price
export implied_vol, implied_vol_batch
export heston_price, heston_cf, fft_option_price
export dupire_local_vol, local_vol_surface
export sabr_implied_vol, sabr_calibrate
export svi_fit, svi_total_variance, svi_no_arbitrage_check
export perpetual_funding_pnl, mark_price_perp, liquidation_price
export calendar_arbitrage_check, butterfly_arbitrage_check
export vol_surface_from_svi

# =============================================================================
# SECTION 1: BLACK-SCHOLES MODEL
# =============================================================================

"""
    bs_price(S, K, T, r, sigma, q=0.0; call=true) -> Float64

Black-Scholes option price.

# Arguments
- `S`: spot price
- `K`: strike price
- `T`: time to expiry (years)
- `r`: risk-free rate (continuously compounded)
- `sigma`: implied volatility (annualized)
- `q`: dividend yield / funding rate
- `call`: true for call, false for put

# Formula
d₁ = (log(S/K) + (r - q + σ²/2)T) / (σ√T)
d₂ = d₁ - σ√T
C = S·e^{-qT}·N(d₁) - K·e^{-rT}·N(d₂)
P = K·e^{-rT}·N(-d₂) - S·e^{-qT}·N(-d₁)
"""
function bs_price(S::Float64, K::Float64, T::Float64, r::Float64,
                   sigma::Float64, q::Float64=0.0; call::Bool=true)::Float64

    (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) && begin
        # Intrinsic value at expiry
        intrinsic = call ? max(S - K, 0.0) : max(K - S, 0.0)
        return intrinsic
    end

    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if call
        return S * exp(-q * T) * _N(d1) - K * exp(-r * T) * _N(d2)
    else
        return K * exp(-r * T) * _N(-d2) - S * exp(-q * T) * _N(-d1)
    end
end

"""
    bs_greeks(S, K, T, r, sigma, q=0.0; call=true) -> NamedTuple

Black-Scholes Greeks: delta, gamma, theta, vega, rho.

Greeks are per-unit of notional:
- Delta: ∂V/∂S
- Gamma: ∂²V/∂S²
- Theta: ∂V/∂T (per calendar day: divide by 365)
- Vega:  ∂V/∂σ (per 1 point of vol, i.e., per 100%)
- Rho:   ∂V/∂r (per 1 point of rate)
"""
function bs_greeks(S::Float64, K::Float64, T::Float64, r::Float64,
                    sigma::Float64, q::Float64=0.0; call::Bool=true)

    if T <= 0 || sigma <= 0 || S <= 0 || K <= 0
        delta = call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0)
        return (delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
                 price=bs_price(S, K, T, r, sigma, q; call=call))
    end

    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    phi_d1 = _phi(d1)  # standard normal PDF at d1
    N_d1 = _N(d1)
    N_d2 = _N(d2)

    price = bs_price(S, K, T, r, sigma, q; call=call)

    # Delta
    delta = call ? exp(-q * T) * N_d1 : exp(-q * T) * (N_d1 - 1.0)

    # Gamma (same for call and put)
    gamma = exp(-q * T) * phi_d1 / (S * sigma * sqrtT)

    # Theta (per day: divided by 365)
    theta_common = -exp(-q * T) * S * phi_d1 * sigma / (2.0 * sqrtT)
    if call
        theta = (theta_common
                  - r * K * exp(-r * T) * N_d2
                  + q * S * exp(-q * T) * N_d1) / 365.0
    else
        theta = (theta_common
                  + r * K * exp(-r * T) * _N(-d2)
                  - q * S * exp(-q * T) * _N(-d1)) / 365.0
    end

    # Vega (per 1% vol move: divide by 100)
    vega = S * exp(-q * T) * phi_d1 * sqrtT / 100.0

    # Rho (per 1% rate move: divide by 100)
    rho = call ?  K * T * exp(-r * T) * N_d2  / 100.0 :
                 -K * T * exp(-r * T) * _N(-d2) / 100.0

    return (delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho, price=price)
end

"""
    black76_price(F, K, T, r, sigma; call=true) -> Float64

Black-76 model for futures/forwards options.

Black-76 treats the forward price F as the underlying; the drift is zero.
d₁ = (log(F/K) + σ²T/2) / (σ√T)
d₂ = d₁ - σ√T
C = e^{-rT}[F·N(d₁) - K·N(d₂)]

Used for: interest rate options, commodity options, crypto futures options.
"""
function black76_price(F::Float64, K::Float64, T::Float64, r::Float64,
                        sigma::Float64; call::Bool=true)::Float64

    (T <= 0 || sigma <= 0 || F <= 0 || K <= 0) && begin
        intrinsic = call ? max(F - K, 0.0) : max(K - F, 0.0)
        return intrinsic * exp(-r * T)
    end

    sqrtT = sqrt(T)
    d1 = (log(F / K) + 0.5 * sigma^2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc = exp(-r * T)
    if call
        return disc * (F * _N(d1) - K * _N(d2))
    else
        return disc * (K * _N(-d2) - F * _N(-d1))
    end
end

# =============================================================================
# SECTION 2: IMPLIED VOLATILITY
# =============================================================================

"""
    implied_vol(price, S, K, T, r, q=0.0; call=true, tol=1e-8, max_iter=50) -> Float64

Implied volatility via Brent's bisection method.

Brent's method combines bisection, secant, and inverse quadratic interpolation
for superlinear convergence with guaranteed progress.
50 iterations at 1e-8 tolerance is sufficient for all practical cases.

# Returns
- Implied volatility, or NaN if no solution found (e.g., price < intrinsic)
"""
function implied_vol(price::Float64, S::Float64, K::Float64, T::Float64,
                      r::Float64, q::Float64=0.0;
                      call::Bool=true,
                      tol::Float64=1e-8,
                      max_iter::Int=50)::Float64

    # Check bounds
    if T <= 0 || S <= 0 || K <= 0
        return NaN
    end

    # Intrinsic value
    disc_fwd = S * exp(-q * T)
    intrinsic = call ? max(disc_fwd - K * exp(-r * T), 0.0) :
                       max(K * exp(-r * T) - disc_fwd, 0.0)
    price < intrinsic - 1e-10 * S && return NaN

    # Objective function
    obj(sigma) = bs_price(S, K, T, r, sigma, q; call=call) - price

    # Bracket: [sigma_lo, sigma_hi]
    sigma_lo, sigma_hi = 1e-6, 10.0

    # Check bracket signs
    f_lo = obj(sigma_lo)
    f_hi = obj(sigma_hi)

    # If both same sign, the price might be at intrinsic
    if f_lo * f_hi > 0
        # Try wider bracket
        sigma_hi = 20.0
        f_hi = obj(sigma_hi)
        f_lo * f_hi > 0 && return NaN
    end

    # Brent's method
    a, b = sigma_lo, sigma_hi
    fa, fb = f_lo, f_hi

    if abs(fa) < abs(fb)
        a, b = b, a
        fa, fb = fb, fa
    end

    c = a; fc = fa
    mflag = true
    s = 0.0; fs = 0.0
    d = 0.0

    for _ in 1:max_iter
        abs(fb) < tol && return b
        abs(b - a) < tol && return b

        if fa != fc && fb != fc
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else
            # Secant method
            s = b - fb * (b - a) / (fb - fa)
        end

        # Conditions to use bisection instead
        cond1 = !((3a + b)/4 < s < b || b < s < (3a + b)/4)
        cond2 =  mflag && abs(s - b) >= abs(b - c) / 2
        cond3 = !mflag && abs(s - b) >= abs(c - d) / 2
        cond4 =  mflag && abs(b - c) < tol
        cond5 = !mflag && abs(c - d) < tol

        if cond1 || cond2 || cond3 || cond4 || cond5
            s = (a + b) / 2
            mflag = true
        else
            mflag = false
        end

        s = max(1e-8, s)  # ensure positive vol
        fs = obj(s)
        d = c; c = b; fc = fb

        if fa * fs < 0
            b = s; fb = fs
        else
            a = s; fa = fs
        end

        if abs(fa) < abs(fb)
            a, b = b, a
            fa, fb = fb, fa
        end
    end

    return b
end

"""
    implied_vol_batch(prices, S, K, T, r; call=true) -> Vector{Float64}

Compute implied vols for multiple strikes at once.
"""
function implied_vol_batch(prices::Vector{Float64},
                             S::Float64,
                             strikes::Vector{Float64},
                             T::Float64,
                             r::Float64;
                             call::Bool=true)::Vector{Float64}
    n = length(prices)
    @assert length(strikes) == n
    return [implied_vol(prices[i], S, strikes[i], T, r; call=call) for i in 1:n]
end

# =============================================================================
# SECTION 3: HESTON MODEL
# =============================================================================

"""
    heston_cf(phi, S, K, T, r, kappa, theta, sigma_v, rho, v0) -> ComplexF64

Characteristic function of log-return under the Heston (1993) model.

dS = r S dt + √v S dW₁
dv = κ(θ - v) dt + σᵥ √v dW₂
Corr(dW₁, dW₂) = ρ

The characteristic function is:
    φ(u) = exp(iuX + A(u) + B(u)*v₀)

where X = log(S) + (r - 0)T and A, B are complex functions of u, κ, θ, σᵥ, ρ.
"""
function heston_cf(phi::ComplexF64,
                    S::Float64, K::Float64, T::Float64, r::Float64,
                    kappa::Float64, theta::Float64,
                    sigma_v::Float64, rho::Float64, v0::Float64)::ComplexF64

    i = 1im
    x = log(S)

    # Heston characteristic function components
    d = sqrt((rho * sigma_v * i * phi - kappa)^2 +
             sigma_v^2 * (i * phi + phi^2))

    g = (kappa - rho * sigma_v * i * phi - d) /
        (kappa - rho * sigma_v * i * phi + d)

    exp_dT = exp(-d * T)

    C = r * i * phi * T +
        (kappa * theta / sigma_v^2) * (
            (kappa - rho * sigma_v * i * phi - d) * T -
            2.0 * log((1.0 - g * exp_dT) / (1.0 - g))
        )

    D = ((kappa - rho * sigma_v * i * phi - d) / sigma_v^2) *
        ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    return exp(C + D * v0 + i * phi * x)
end

"""
    heston_price(S, K, T, r, kappa, theta, sigma_v, rho, v0; call=true, n_points=256) -> Float64

Price a European option under the Heston stochastic volatility model
using the Gil-Pelaez Fourier inversion theorem.

P(S > K) = 1/2 + (1/π) ∫₀^∞ Re[e^{-iφlog(K)} φ(φ)/iφ] dφ

Integration via Gauss-Laguerre quadrature with n_points nodes.
"""
function heston_price(S::Float64, K::Float64, T::Float64, r::Float64,
                       kappa::Float64, theta::Float64,
                       sigma_v::Float64, rho::Float64, v0::Float64;
                       call::Bool=true,
                       n_points::Int=256)::Float64

    log_K = log(K)

    # Numerical integration via composite Simpson's rule on [0, upper_limit]
    upper_lim = 100.0
    h = upper_lim / n_points
    phi_vals = [(j + 0.5) * h for j in 0:(n_points-1)]

    # Two probabilities P1, P2 from Gil-Pelaez
    P1_sum = 0.0
    P2_sum = 0.0

    for phi in phi_vals
        # P1: probability measure under stock numeraire
        cf1 = heston_cf(phi - 1.0im, S, K, T, r, kappa, theta, sigma_v, rho, v0)
        cf0 = heston_cf(0.0 - 1.0im, S, K, T, r, kappa, theta, sigma_v, rho, v0)
        if abs(cf0) > 1e-20
            integrand1 = real(exp(-1im * phi * log_K) * cf1 / (1im * phi * cf0))
            P1_sum += integrand1 * h
        end

        # P2: risk-neutral probability
        cf2 = heston_cf(complex(phi, 0.0), S, K, T, r, kappa, theta, sigma_v, rho, v0)
        integrand2 = real(exp(-1im * phi * log_K) * cf2 / (1im * phi))
        P2_sum += integrand2 * h
    end

    P1 = 0.5 + P1_sum / π
    P2 = 0.5 + P2_sum / π

    P1 = clamp(P1, 0.0, 1.0)
    P2 = clamp(P2, 0.0, 1.0)

    call_price = S * P1 - K * exp(-r * T) * P2
    call_price = max(call_price, max(S - K * exp(-r * T), 0.0))

    if call
        return call_price
    else
        # Put-call parity
        return call_price - S + K * exp(-r * T)
    end
end

# =============================================================================
# SECTION 4: FFT OPTION PRICING (CARR-MADAN)
# =============================================================================

"""
    fft_option_price(S, strikes, T, r, log_cf; alpha=1.5, n=4096, eta=0.25) -> Vector{Float64}

Carr-Madan (1999) FFT option pricing algorithm.

Prices a strip of European call options at multiple strikes simultaneously
using the FFT of the modified characteristic function.

C(K) = e^{-α log K} / π ∫₀^∞ e^{-iv log K} ψ(v) dv

where ψ(v) = e^{-rT} φ(v - (α+1)i) / (α² + α - v² + i(2α+1)v)

# Arguments
- `S`: spot price
- `strikes`: vector of target strike prices
- `T`: time to expiry
- `r`: risk-free rate
- `log_cf`: function(u::ComplexF64) -> ComplexF64, characteristic function of log return
- `alpha`: damping parameter (default 1.5, must be > 0)
- `n`: number of FFT points (power of 2)
- `eta`: step size in frequency domain

# Returns
- Vector of call prices at requested strikes (via interpolation of FFT grid)
"""
function fft_option_price(S::Float64,
                            strikes::Vector{Float64},
                            T::Float64,
                            r::Float64,
                            log_cf::Function;
                            alpha::Float64=1.5,
                            n::Int=4096,
                            eta::Float64=0.25)::Vector{Float64}

    # FFT grid parameters
    lambda = 2π / (n * eta)     # log-strike spacing
    b = n * lambda / 2.0         # log-strike range is [-b, b]

    # Frequency grid v_j = eta * (j - 1), j = 1..N
    j_vec = 1:n
    v_vec = eta * (j_vec .- 1)

    # Modified characteristic function
    psi = ComplexF64[]
    for v in v_vec
        u = v - (alpha + 1.0) * 1im
        phi_u = try
            log_cf(complex(real(u), imag(u)))
        catch
            complex(0.0, 0.0)
        end

        denom = alpha^2 + alpha - v^2 + 1im * (2alpha + 1) * v
        if abs(denom) < 1e-20
            push!(psi, complex(0.0, 0.0))
        else
            push!(psi, exp(-r * T) * phi_u / denom)
        end
    end

    # Apply Simpson's rule weights
    simpson_weights = ones(n)
    simpson_weights[1] = 1/3
    for j in 2:(n-1)
        simpson_weights[j] = j % 2 == 0 ? 4/3 : 2/3
    end
    simpson_weights[n] = 1/3

    # Build FFT input
    k_vec = -b .+ lambda * (j_vec .- 1)  # log-strike grid
    x = psi .* exp.(1im * v_vec * b) .* simpson_weights * eta

    # FFT
    X = _fft(x)

    # Extract call prices
    call_prices = real(exp.(-alpha .* k_vec) ./ π .* X)

    # Interpolate to requested strikes
    log_K = log.(max.(strikes, 1e-10))
    result = zeros(length(strikes))
    for (i, lk) in enumerate(log_K)
        # Find nearest grid points and interpolate
        idx = searchsortedfirst(k_vec, lk)
        if idx <= 1
            result[i] = max(call_prices[1], 0.0)
        elseif idx > n
            result[i] = max(call_prices[n], 0.0)
        else
            # Linear interpolation
            t = (lk - k_vec[idx-1]) / (k_vec[idx] - k_vec[idx-1])
            c = (1-t) * call_prices[idx-1] + t * call_prices[idx]
            result[i] = max(c, 0.0)
        end
    end

    return result
end

"""Simple DFT (for small n; in production use FFTW)."""
function _fft(x::Vector{ComplexF64})::Vector{ComplexF64}
    n = length(x)
    # Cooley-Tukey radix-2 FFT
    if n == 1
        return x
    end
    if n & (n-1) != 0
        # Fall back to DFT for non-power-of-2
        return _dft(x)
    end

    even = _fft(x[1:2:end])
    odd  = _fft(x[2:2:end])

    T_arr = [exp(-2π * 1im * k / n) * odd[k+1] for k in 0:(n÷2 - 1)]

    return vcat(even .+ T_arr, even .- T_arr)
end

function _dft(x::Vector{ComplexF64})::Vector{ComplexF64}
    n = length(x)
    result = zeros(ComplexF64, n)
    for k in 0:(n-1)
        for j in 0:(n-1)
            result[k+1] += x[j+1] * exp(-2π * 1im * j * k / n)
        end
    end
    return result
end

# =============================================================================
# SECTION 5: LOCAL VOLATILITY (DUPIRE)
# =============================================================================

"""
    dupire_local_vol(K, T, C_grid, K_grid, T_grid, r, q) -> Float64

Dupire (1994) formula for local volatility surface.

Local vol is the vol that makes BS consistent with all market prices:
    σ²_loc(K,T) = [∂C/∂T + (r-q)K ∂C/∂K + qC] /
                  [K²/2 * ∂²C/∂K²]

Numerically: use finite differences on the option price surface.

# Arguments
- `K`, `T`: point where local vol is requested
- `C_grid`: (n_T × n_K) matrix of call prices
- `K_grid`, `T_grid`: strike and maturity grid vectors
- `r`, `q`: interest/dividend rates
"""
function dupire_local_vol(K::Float64, T::Float64,
                            C_grid::Matrix{Float64},
                            K_grid::Vector{Float64},
                            T_grid::Vector{Float64},
                            r::Float64, q::Float64)::Float64

    n_T = length(T_grid)
    n_K = length(K_grid)

    # Find indices
    k_idx = max(2, min(n_K-1, searchsortedfirst(K_grid, K)))
    t_idx = max(2, min(n_T-1, searchsortedfirst(T_grid, T)))

    dK = K_grid[k_idx+1] - K_grid[k_idx-1]
    dT = T_grid[t_idx] - T_grid[t_idx-1]

    C   = C_grid[t_idx, k_idx]
    dC_dT = (C_grid[t_idx, k_idx] - C_grid[t_idx-1, k_idx]) / dT
    dC_dK = (C_grid[t_idx, k_idx+1] - C_grid[t_idx, k_idx-1]) / dK
    d2C_dK2 = (C_grid[t_idx, k_idx+1] - 2*C + C_grid[t_idx, k_idx-1]) / ((dK/2)^2)

    numerator   = dC_dT + (r - q) * K * dC_dK + q * C
    denominator = 0.5 * K^2 * d2C_dK2

    if denominator <= 1e-15 || numerator < 0
        # Fall back to ATM vol approximation
        return 0.2
    end

    return sqrt(numerator / denominator)
end

"""
    local_vol_surface(C_grid, K_grid, T_grid, r, q) -> Matrix{Float64}

Compute full local volatility surface from call price grid.
"""
function local_vol_surface(C_grid::Matrix{Float64},
                             K_grid::Vector{Float64},
                             T_grid::Vector{Float64},
                             r::Float64, q::Float64)::Matrix{Float64}

    n_T = length(T_grid)
    n_K = length(K_grid)
    lv = zeros(n_T, n_K)

    for t_idx in 2:(n_T-1), k_idx in 2:(n_K-1)
        lv[t_idx, k_idx] = dupire_local_vol(
            K_grid[k_idx], T_grid[t_idx],
            C_grid, K_grid, T_grid, r, q
        )
    end

    # Fill edges with nearest values
    for t_idx in 1:n_T
        lv[t_idx, 1] = lv[t_idx, 2]
        lv[t_idx, end] = lv[t_idx, end-1]
    end
    lv[1, :] .= lv[2, :]
    lv[end, :] .= lv[end-1, :]

    return lv
end

# =============================================================================
# SECTION 6: SABR MODEL
# =============================================================================

"""
    sabr_implied_vol(F, K, T, alpha, beta, rho, nu) -> Float64

Hagan et al. (2002) SABR model implied volatility approximation.

SABR dynamics:
    dF = σ F^β dW₁
    dσ = ν σ dW₂
    Corr(dW₁, dW₂) = ρ

Hagan approximation (expanded to O(ε²)):
    σ_BS(K,T) = α·z/x(z) / [FK)^{(1-β)/2} · (...)]

Valid for K near F. The approximation breaks down for deep ITM/OTM.

# Arguments
- `F`: forward price
- `K`: strike
- `T`: time to expiry
- `alpha`: initial vol (σ₀)
- `beta`: CEV exponent ∈ [0,1]
- `rho`: vol-fwd correlation
- `nu`: vol-of-vol

# Returns
- Black-Scholes implied volatility
"""
function sabr_implied_vol(F::Float64, K::Float64, T::Float64,
                            alpha::Float64, beta::Float64,
                            rho::Float64, nu::Float64)::Float64

    F <= 0 || K <= 0 || T <= 0 && return alpha

    FK_mid = sqrt(F * K)
    log_FK = log(F / K)
    one_minus_beta = 1.0 - beta

    if abs(log_FK) < 1e-7
        # ATM formula (F ≈ K)
        # σ_ATM = α / [F^(1-β) * (1 + ...)]
        atm_factor = alpha / FK_mid^one_minus_beta

        correction1 = 1.0 + (one_minus_beta^2 / 24.0 * alpha^2 / FK_mid^(2*one_minus_beta) +
                              0.25 * rho * beta * nu * alpha / FK_mid^one_minus_beta +
                              (2.0 - 3.0*rho^2) / 24.0 * nu^2) * T

        return atm_factor * correction1
    end

    # General formula
    z = (nu / alpha) * FK_mid^one_minus_beta * log_FK
    x_z = log((sqrt(1.0 - 2.0*rho*z + z^2) + z - rho) / (1.0 - rho))

    numerator_factor = alpha
    denominator_factor = FK_mid^one_minus_beta * (
        1.0 + one_minus_beta^2/24.0 * log_FK^2 +
        one_minus_beta^4/1920.0 * log_FK^4
    )

    eps_correction = 1.0 + (
        one_minus_beta^2 * alpha^2 / (24.0 * FK_mid^(2*one_minus_beta)) +
        0.25 * rho * beta * nu * alpha / FK_mid^one_minus_beta +
        (2.0 - 3.0*rho^2) / 24.0 * nu^2
    ) * T

    z_over_xz = abs(x_z) < 1e-10 ? 1.0 : z / x_z

    return (numerator_factor / denominator_factor) * z_over_xz * eps_correction
end

"""
    sabr_calibrate(F, strikes, T, market_vols; beta=0.5) -> NamedTuple

Calibrate SABR parameters (alpha, rho, nu) to market implied vols.
Beta is usually fixed (0.5 is standard).

Uses coordinate descent minimizing sum of squared vol errors.

# Returns
- NamedTuple: alpha, beta, rho, nu, rmse
"""
function sabr_calibrate(F::Float64,
                          strikes::Vector{Float64},
                          T::Float64,
                          market_vols::Vector{Float64};
                          beta::Float64=0.5)

    n = length(strikes)
    @assert length(market_vols) == n

    # Initial guess: alpha from ATM vol, rho=0, nu=0.3
    atm_idx = argmin(abs.(strikes .- F))
    atm_vol = market_vols[atm_idx]
    alpha0 = atm_vol * F^(1 - beta)
    rho0 = 0.0
    nu0 = 0.3

    function objective(params)
        alpha, rho, nu = params
        (alpha <= 0 || abs(rho) >= 1 || nu <= 0) && return 1e15
        sse = 0.0
        for i in 1:n
            model_vol = sabr_implied_vol(F, strikes[i], T, alpha, beta, rho, nu)
            sse += (model_vol - market_vols[i])^2
        end
        return sse
    end

    # Coordinate descent
    best = [alpha0, rho0, nu0]
    best_obj = objective(best)

    step_sizes = [alpha0 * 0.1, 0.05, 0.05]
    tol = 1e-8

    for outer_iter in 1:100
        improved = false
        for dim in 1:3
            for direction in [1.0, -1.0]
                candidate = copy(best)
                candidate[dim] += direction * step_sizes[dim]
                # Clamp constraints
                candidate[1] = max(1e-6, candidate[1])
                candidate[2] = clamp(candidate[2], -0.99, 0.99)
                candidate[3] = max(1e-6, candidate[3])

                obj_val = objective(candidate)
                if obj_val < best_obj
                    best_obj = obj_val
                    best = candidate
                    improved = true
                end
            end
        end
        if !improved
            step_sizes .*= 0.5
            all(step_sizes .< tol) && break
        end
    end

    rmse = sqrt(best_obj / n)
    return (alpha=best[1], beta=beta, rho=best[2], nu=best[3], rmse=rmse)
end

# =============================================================================
# SECTION 7: SVI VOL SURFACE
# =============================================================================

"""
    svi_total_variance(k, a, b, rho, m, sigma) -> Float64

SVI (Stochastic Volatility Inspired) parametrization of Gatheral (2004).

Total variance w(k) = a + b(ρ(k-m) + √((k-m)²+σ²))

where k = log(K/F) is log-moneyness.

Parameters:
- a: overall level of variance
- b: angle between left and right wings
- rho: orientation (skew), ∈ (-1, 1)
- m: center of the smile
- sigma: at-the-money curvature (smoothness)
"""
function svi_total_variance(k::Float64,
                              a::Float64, b::Float64, rho::Float64,
                              m::Float64, sigma::Float64)::Float64
    x = k - m
    return a + b * (rho * x + sqrt(x^2 + sigma^2))
end

"""
    svi_fit(log_moneyness, total_variance; init=nothing) -> NamedTuple

Calibrate SVI parameters to market total variance (= implied_vol² * T).

# Arguments
- `log_moneyness`: k = log(K/F) for each option
- `total_variance`: w = σ²_BS * T for each option

# Returns
- NamedTuple: a, b, rho, m, sigma, rmse, converged
"""
function svi_fit(log_moneyness::Vector{Float64},
                  total_variance::Vector{Float64};
                  init::Union{Nothing, Vector{Float64}}=nothing)

    n = length(log_moneyness)
    @assert length(total_variance) == n

    # Initial parameters
    atm_var = total_variance[argmin(abs.(log_moneyness))]
    a0 = atm_var * 0.5
    b0 = 0.5
    rho0 = 0.0
    m0 = 0.0
    sigma0 = 0.2

    params0 = init === nothing ? [a0, b0, rho0, m0, sigma0] : init

    function objective(p)
        a, b, rho, m, sigma = p
        (b <= 0 || abs(rho) >= 1 || sigma <= 0 || a < 0) && return 1e15
        # No-arbitrage: a + b*sigma*(1-rho²) >= 0
        (a + b * sigma * sqrt(1 - rho^2) < 0) && return 1e15

        sse = 0.0
        for i in 1:n
            model_w = svi_total_variance(log_moneyness[i], a, b, rho, m, sigma)
            model_w < 0 && return 1e15
            sse += (model_w - total_variance[i])^2
        end
        return sse
    end

    best = copy(params0)
    best_obj = objective(best)

    steps = [abs(a0)*0.1 + 1e-4, 0.05, 0.05, 0.05, 0.05]
    converged = false

    for outer in 1:200
        improved = false
        for dim in 1:5
            for dir in [1.0, -1.0]
                cand = copy(best)
                cand[dim] += dir * steps[dim]
                obj = objective(cand)
                if obj < best_obj
                    best_obj = obj
                    best = cand
                    improved = true
                end
            end
        end
        if !improved
            steps .*= 0.5
            if all(steps .< 1e-9)
                converged = true
                break
            end
        end
    end

    a, b, rho, m, sigma = best
    rmse = sqrt(best_obj / n)

    return (a=a, b=b, rho=rho, m=m, sigma=sigma, rmse=rmse, converged=converged)
end

"""
    svi_no_arbitrage_check(params, log_moneyness_grid) -> NamedTuple

Check SVI parametrization for calendar spread and butterfly arbitrage.

No butterfly arbitrage: ∂²C/∂K² ≥ 0 ↔ g(k) ≥ 0 where
g(k) = (1 - k*w'/(2w))² - (w'/4)*(1/w + 1/4) + w''/2

# Returns
- NamedTuple: has_butterfly_arb, has_calendar_arb, min_g
"""
function svi_no_arbitrage_check(a::Float64, b::Float64, rho::Float64,
                                  m::Float64, sigma::Float64,
                                  log_moneyness_grid::Vector{Float64})

    dk = 1e-4
    min_g = Inf

    for k in log_moneyness_grid
        w   = svi_total_variance(k, a, b, rho, m, sigma)
        w_p = (svi_total_variance(k+dk, a, b, rho, m, sigma) -
               svi_total_variance(k-dk, a, b, rho, m, sigma)) / (2dk)
        w_pp = (svi_total_variance(k+dk, a, b, rho, m, sigma) -
                2w + svi_total_variance(k-dk, a, b, rho, m, sigma)) / dk^2

        if w > 0
            g = (1.0 - k * w_p / (2w))^2 -
                (w_p^2 / 4) * (1.0/w + 0.25) +
                w_pp / 2.0
            min_g = min(min_g, g)
        end
    end

    has_butterfly_arb = min_g < -1e-6

    return (has_butterfly_arb=has_butterfly_arb,
             has_calendar_arb=false,  # calendar arb requires multiple maturities
             min_g=min_g)
end

# =============================================================================
# SECTION 8: CRYPTO PERPETUALS
# =============================================================================

"""
    mark_price_perp(index_price, fair_price_premium, impact_bid, impact_ask) -> Float64

Compute the mark price for a crypto perpetual futures contract.

Mark price is used for PnL and liquidation calculations.
Typical formula: Mark = Median(Index, Fair Price, Impact Mid)

where:
  Fair Price = Index * (1 + Funding Rate)
  Impact Mid = (Impact Bid + Impact Ask) / 2
  Impact Bid/Ask = TWAP of bid/ask prices that can absorb margin_impact_notional

# Arguments
- `index_price`: current spot index price
- `fair_price_premium`: (funding rate) * index_price
- `impact_bid`, `impact_ask`: impact prices for liquidation calc

# Returns
- Mark price
"""
function mark_price_perp(index_price::Float64,
                           fair_price_premium::Float64,
                           impact_bid::Float64,
                           impact_ask::Float64)::Float64

    fair_price = index_price + fair_price_premium
    impact_mid = (impact_bid + impact_ask) / 2.0

    # Median of three
    prices = sort([index_price, fair_price, impact_mid])
    return prices[2]  # median
end

"""
    perpetual_funding_pnl(position_size, entry_price, mark_price, funding_rate,
                          n_periods; long=true) -> NamedTuple

Compute the cumulative PnL from a perpetual futures position including:
1. Mark-to-market unrealized PnL
2. Funding payments (paid/received every 8 hours typically)

Funding PnL (long): -position_size * mark_price * funding_rate per period
Funding PnL (short): +position_size * mark_price * funding_rate per period

# Returns
- NamedTuple: unrealized_pnl, funding_pnl, total_pnl, effective_entry
"""
function perpetual_funding_pnl(position_size::Float64,
                                 entry_price::Float64,
                                 mark_price::Float64,
                                 funding_rate::Float64,
                                 n_periods::Int;
                                 long::Bool=true)

    direction = long ? 1.0 : -1.0

    # Mark-to-market PnL
    unrealized_pnl = direction * position_size * (mark_price - entry_price)

    # Cumulative funding PnL
    # Each period: funding payment = -direction * position_size * mark_price * funding_rate
    # Approximate as constant mark price (simplification)
    funding_pnl = -direction * position_size * mark_price * funding_rate * n_periods

    total_pnl = unrealized_pnl + funding_pnl

    # Effective entry price accounting for funding
    effective_entry = entry_price - funding_pnl / (position_size * direction)

    return (unrealized_pnl=unrealized_pnl, funding_pnl=funding_pnl,
             total_pnl=total_pnl, effective_entry=effective_entry)
end

"""
    liquidation_price(entry_price, margin, maintenance_margin_rate,
                      position_size, mark_price; long=true) -> Float64

Compute the liquidation price for a crypto futures position.

Liquidation occurs when: Equity < Maintenance Margin
    Equity = Margin + Unrealized PnL
    Unrealized PnL = ±position_size * (mark_price - entry_price)

Solving for mark_price:
    Long:  liq_price = entry_price - (margin - maintenance_margin) / position_size
    Short: liq_price = entry_price + (margin - maintenance_margin) / position_size

where maintenance_margin = position_size * mark_price * maintenance_margin_rate
"""
function liquidation_price(entry_price::Float64,
                             margin::Float64,
                             maintenance_margin_rate::Float64,
                             position_size::Float64;
                             long::Bool=true)::Float64

    # Approximate: maintenance_margin ≈ position_size * entry_price * maintenance_margin_rate
    maintenance_margin = position_size * entry_price * maintenance_margin_rate

    available_margin = margin - maintenance_margin
    available_margin < 0 && return entry_price  # already under

    move_per_unit = available_margin / position_size

    if long
        return max(0.0, entry_price - move_per_unit)
    else
        return entry_price + move_per_unit
    end
end

# =============================================================================
# SECTION 9: ARBITRAGE DETECTION
# =============================================================================

"""
    calendar_arbitrage_check(total_variances, maturities) -> NamedTuple

Check for calendar spread arbitrage in a vol surface.

No calendar arbitrage: total variance w(K,T) is non-decreasing in T for each K.
∂w/∂T ≥ 0 for all K, T.

# Returns
- NamedTuple: has_arb, violation_indices, max_violation
"""
function calendar_arbitrage_check(total_variances::Matrix{Float64},
                                    maturities::Vector{Float64})

    n_T, n_K = size(total_variances)
    n_T < 2 && return (has_arb=false, violation_indices=Tuple{Int,Int}[], max_violation=0.0)

    violations = Tuple{Int,Int}[]
    max_viol = 0.0

    for t_idx in 2:n_T
        for k_idx in 1:n_K
            w_prev = total_variances[t_idx-1, k_idx]
            w_curr = total_variances[t_idx, k_idx]
            if w_curr < w_prev - 1e-8
                push!(violations, (t_idx, k_idx))
                max_viol = max(max_viol, w_prev - w_curr)
            end
        end
    end

    return (has_arb=!isempty(violations), violation_indices=violations,
             max_violation=max_viol)
end

"""
    butterfly_arbitrage_check(calls, strikes, F, r, T) -> NamedTuple

Check for butterfly arbitrage: call prices must be convex in strike.

No butterfly arb: ∂²C/∂K² ≥ 0, or equivalently:
C(K-dK) + C(K+dK) ≥ 2*C(K)

Violation means free money from a butterfly spread.

# Returns
- NamedTuple: has_arb, violation_strikes, max_violation
"""
function butterfly_arbitrage_check(calls::Vector{Float64},
                                     strikes::Vector{Float64},
                                     F::Float64, r::Float64, T::Float64)

    n = length(calls)
    n < 3 && return (has_arb=false, violation_strikes=Float64[], max_violation=0.0)
    @assert length(strikes) == n

    violations = Float64[]
    max_viol = 0.0

    for i in 2:(n-1)
        dK = (strikes[i+1] - strikes[i-1]) / 2.0
        # Second derivative approximation
        butterfly_value = calls[i-1] - 2*calls[i] + calls[i+1]
        if butterfly_value < -1e-8
            push!(violations, strikes[i])
            max_viol = max(max_viol, -butterfly_value)
        end
    end

    # Also check: call spread must be non-increasing in K
    for i in 2:n
        call_spread = calls[i-1] - calls[i]
        if call_spread < -(strikes[i] - strikes[i-1]) * exp(-r*T)
            push!(violations, strikes[i])
        end
    end

    return (has_arb=!isempty(violations), violation_strikes=unique(violations),
             max_violation=max_viol)
end

"""
    vol_surface_from_svi(F, strikes, maturities, svi_params_per_maturity) -> Matrix{Float64}

Build a full implied vol surface from SVI parameters per maturity.

# Arguments
- `F`: forward price (or Vector if term structure)
- `strikes`: K vector
- `maturities`: T vector
- `svi_params_per_maturity`: Vector of (a,b,rho,m,sigma) NamedTuples

# Returns
- (n_T × n_K) implied vol matrix
"""
function vol_surface_from_svi(F::Float64,
                                strikes::Vector{Float64},
                                maturities::Vector{Float64},
                                svi_params::Vector)::Matrix{Float64}

    n_T = length(maturities)
    n_K = length(strikes)
    vols = zeros(n_T, n_K)

    for (t_idx, T) in enumerate(maturities)
        T <= 0 && continue
        p = svi_params[t_idx]
        for (k_idx, K) in enumerate(strikes)
            K <= 0 && continue
            lm = log(K / F)
            w = svi_total_variance(lm, p.a, p.b, p.rho, p.m, p.sigma)
            vols[t_idx, k_idx] = w > 0 ? sqrt(w / T) : 0.0
        end
    end

    return vols
end

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

"""Standard normal CDF (Hart approximation, accurate to 7 decimal places)."""
function _N(x::Float64)::Float64
    x >= 8.0  && return 1.0
    x <= -8.0 && return 0.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 +
           t * (-0.356563782 +
           t * (1.781477937 +
           t * (-1.821255978 +
           t * 1.330274429))))
    phi = exp(-0.5 * x^2) / sqrt(2π)
    cdf = 1.0 - phi * poly
    return x >= 0 ? cdf : 1.0 - cdf
end

"""Standard normal PDF."""
function _phi(x::Float64)::Float64
    exp(-0.5 * x^2) / sqrt(2π)
end

end # module DerivativesMath
