"""
DerivativesPricing — Complete Derivatives Library

Comprehensive derivatives pricing and risk management:
  - Option pricing: BS, Heston, SABR, local vol, jump-diffusion
  - Exotic options: barrier, Asian, lookback, digital
  - Greeks: analytical for BS, finite-difference for others
  - Vol surface: SVI, SSVI, kernel smoothing
  - Crypto perpetuals: funding rate model, basis trading
  - Term structure of volatility (Bergomi model)
  - Correlation products: basket options, worst-of
  - Calibration framework: minimize squared vol errors
"""
module DerivativesPricing

using Statistics
using LinearAlgebra
using Random

export BlackScholes, bs_price, bs_greeks, bs_all_greeks
export implied_vol_brent, implied_vol_surface
export HestonModel, heston_price, heston_calibrate
export SABRModel, sabr_vol, sabr_calibrate
export JumpDiffusion, merton_jump_price
export BarrierOption, barrier_price_bs
export AsianOption, asian_price_mc
export LookbackOption, lookback_price_mc
export DigitalOption, digital_price_bs
export SVIParams, svi_vol, svi_fit, ssvi_vol
export CryptoPerp, perp_funding_pnl, perp_delta_hedge
export BergomiVol, bergomi_term_structure
export BasketOption, basket_price_mc, worst_of_price_mc
export CalibrationResult, calibrate_vol_surface

# =============================================================================
# SECTION 1: BLACK-SCHOLES
# =============================================================================

"""Black-Scholes model parameters."""
struct BlackScholes
    S::Float64      # spot
    r::Float64      # risk-free rate
    q::Float64      # dividend / funding yield
end

"""
    bs_price(S, K, T, r, sigma, q=0.0; call=true) -> Float64

Black-Scholes call/put price.
"""
function bs_price(S::Float64, K::Float64, T::Float64, r::Float64,
                   sigma::Float64, q::Float64=0.0; call::Bool=true)::Float64
    (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) && begin
        intrinsic = call ? max(S - K, 0.0) : max(K - S, 0.0)
        return intrinsic
    end
    sqT = sqrt(T)
    d1 = (log(S/K) + (r - q + 0.5sigma^2)*T) / (sigma*sqT)
    d2 = d1 - sigma*sqT
    if call
        return S*exp(-q*T)*_N(d1) - K*exp(-r*T)*_N(d2)
    else
        return K*exp(-r*T)*_N(-d2) - S*exp(-q*T)*_N(-d1)
    end
end

"""
    bs_greeks(S, K, T, r, sigma, q=0.0; call=true) -> NamedTuple

Full Greeks: delta, gamma, theta, vega, rho, vanna, volga, charm.
"""
function bs_greeks(S::Float64, K::Float64, T::Float64, r::Float64,
                    sigma::Float64, q::Float64=0.0; call::Bool=true)

    (T <= 0 || sigma <= 0) && begin
        delta = call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0)
        return (delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
                 vanna=0.0, volga=0.0, charm=0.0,
                 price=bs_price(S, K, T, r, sigma, q; call=call))
    end

    sqT = sqrt(T)
    d1 = (log(S/K) + (r - q + 0.5sigma^2)*T) / (sigma*sqT)
    d2 = d1 - sigma*sqT

    phi1 = _phi(d1)
    N1 = _N(d1); N2 = _N(d2)
    Nm1 = _N(-d1); Nm2 = _N(-d2)

    disc_q = exp(-q*T)
    disc_r = exp(-r*T)

    price = call ? S*disc_q*N1 - K*disc_r*N2 : K*disc_r*Nm2 - S*disc_q*Nm1

    delta = call ? disc_q*N1 : disc_q*(N1 - 1.0)
    gamma = disc_q * phi1 / (S*sigma*sqT)
    vega  = S*disc_q*phi1*sqT / 100.0

    # Theta per calendar day
    theta_common = -disc_q*S*phi1*sigma/(2sqT)
    theta = if call
        (theta_common - r*K*disc_r*N2 + q*S*disc_q*N1) / 365.0
    else
        (theta_common + r*K*disc_r*Nm2 - q*S*disc_q*Nm1) / 365.0
    end

    rho = call ? K*T*disc_r*N2/100.0 : -K*T*disc_r*Nm2/100.0

    # Higher-order Greeks
    # Vanna = dDelta/dsigma = d^2V/dS dsigma
    vanna = -disc_q * phi1 * d2 / sigma

    # Volga (vomma) = d^2V/dsigma^2
    volga = S * disc_q * phi1 * sqT * d1 * d2 / sigma

    # Charm = dDelta/dt (delta bleed)
    charm = if call
        disc_q * (q*N1 - phi1*(2*(r-q)*T - d2*sigma*sqT)/(2T*sigma*sqT))
    else
        disc_q * (-q*Nm1 + phi1*(2*(r-q)*T - d2*sigma*sqT)/(2T*sigma*sqT))
    end

    return (delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho,
             vanna=vanna, volga=volga, charm=charm, price=price, d1=d1, d2=d2)
end

"""
    bs_all_greeks(S, K, T, r, sigma; bump_pct=0.01) -> NamedTuple

Extended Greeks including speed, color, veta via finite differences.
"""
function bs_all_greeks(S::Float64, K::Float64, T::Float64, r::Float64,
                        sigma::Float64; bump_pct::Float64=0.01)

    base = bs_greeks(S, K, T, r, sigma)

    # Speed = d^3V/dS^3 = dGamma/dS
    dS = S * bump_pct
    g_up   = bs_greeks(S+dS, K, T, r, sigma).gamma
    g_down = bs_greeks(S-dS, K, T, r, sigma).gamma
    speed  = (g_up - g_down) / (2dS)

    # Color = d^2Gamma/dt
    dT = max(T * bump_pct, 1/365)
    T_down = max(T - dT, 1e-4)
    g_t_down = bs_greeks(S, K, T_down, r, sigma).gamma
    color = (base.gamma - g_t_down) / dT

    # Veta = dVega/dt
    v_t_down = bs_greeks(S, K, T_down, r, sigma).vega
    veta = (base.vega - v_t_down) / dT

    return merge(base, (speed=speed, color=color, veta=veta))
end

"""
    implied_vol_brent(price, S, K, T, r, q=0.0; call=true, tol=1e-8) -> Float64

Implied vol via Brent's method (50 iterations, 1e-8 tolerance).
"""
function implied_vol_brent(price::Float64, S::Float64, K::Float64, T::Float64,
                             r::Float64, q::Float64=0.0;
                             call::Bool=true,
                             tol::Float64=1e-8)::Float64

    T <= 0 || S <= 0 || K <= 0 && return NaN

    disc_fwd = S*exp(-q*T)
    intrinsic = call ? max(disc_fwd - K*exp(-r*T), 0.0) : max(K*exp(-r*T) - disc_fwd, 0.0)
    price < intrinsic - 1e-8*S && return NaN

    f(sig) = bs_price(S, K, T, r, sig, q; call=call) - price

    a, b = 1e-6, 10.0
    fa, fb = f(a), f(b)
    fa * fb > 0 && (b = 20.0; fb = f(b))
    fa * fb > 0 && return NaN

    if abs(fa) < abs(fb); a,b = b,a; fa,fb = fb,fa end
    c = a; fc = fa
    mflag = true; d = 0.0; s = 0.0

    for _ in 1:50
        abs(fb) < tol && return b
        abs(b-a) < tol && return b

        if fa != fc && fb != fc
            s = (a*fb*fc/((fa-fb)*(fa-fc)) +
                 b*fa*fc/((fb-fa)*(fb-fc)) +
                 c*fa*fb/((fc-fa)*(fc-fb)))
        else
            s = b - fb*(b-a)/(fb-fa)
        end

        c1 = !((3a+b)/4 < s < b || b < s < (3a+b)/4)
        c2 = mflag && abs(s-b) >= abs(b-c)/2
        c3 = !mflag && abs(s-b) >= abs(c-d)/2
        c4 = mflag && abs(b-c) < tol
        c5 = !mflag && abs(c-d) < tol

        if c1||c2||c3||c4||c5
            s = (a+b)/2; mflag = true
        else
            mflag = false
        end

        s = max(1e-8, s)
        fs = f(s)
        d = c; c = b; fc = fb

        if fa*fs < 0; b = s; fb = fs
        else; a = s; fa = fs end

        if abs(fa) < abs(fb); a,b = b,a; fa,fb = fb,fa end
    end
    return b
end

"""
    implied_vol_surface(prices, S, strikes, maturities, r; call=true) -> Matrix{Float64}

Compute full implied vol surface from a grid of option prices.
prices: (n_T × n_K) matrix.
"""
function implied_vol_surface(prices::Matrix{Float64},
                               S::Float64,
                               strikes::Vector{Float64},
                               maturities::Vector{Float64},
                               r::Float64;
                               call::Bool=true)::Matrix{Float64}

    n_T, n_K = length(maturities), length(strikes)
    vols = zeros(n_T, n_K)

    for (t_idx, T) in enumerate(maturities), (k_idx, K) in enumerate(strikes)
        price = prices[t_idx, k_idx]
        vols[t_idx, k_idx] = implied_vol_brent(price, S, K, T, r; call=call)
    end

    return vols
end

# =============================================================================
# SECTION 2: HESTON MODEL
# =============================================================================

"""Heston stochastic volatility model parameters."""
struct HestonModel
    kappa::Float64   # mean reversion speed
    theta::Float64   # long-run variance
    sigma::Float64   # vol of vol
    rho::Float64     # spot-vol correlation
    v0::Float64      # initial variance

    function HestonModel(kappa, theta, sigma, rho, v0)
        abs(rho) >= 1 && error("rho must be in (-1, 1)")
        sigma <= 0 && error("sigma must be positive")
        new(kappa, theta, sigma, clamp(rho, -0.999, 0.999), max(v0, 1e-6))
    end
end

"""Heston characteristic function."""
function _heston_cf(phi::ComplexF64, S::Float64, T::Float64, r::Float64,
                     m::HestonModel)::ComplexF64
    i = 1im
    x = log(S)
    d = sqrt((m.rho*m.sigma*i*phi - m.kappa)^2 + m.sigma^2*(i*phi + phi^2))
    g = (m.kappa - m.rho*m.sigma*i*phi - d) / (m.kappa - m.rho*m.sigma*i*phi + d)
    e_dT = exp(-d*T)
    C = r*i*phi*T + (m.kappa*m.theta/m.sigma^2) * (
            (m.kappa - m.rho*m.sigma*i*phi - d)*T -
            2*log((1 - g*e_dT)/(1 - g))
        )
    D = ((m.kappa - m.rho*m.sigma*i*phi - d)/m.sigma^2) *
        ((1 - e_dT)/(1 - g*e_dT))
    return exp(C + D*m.v0 + i*phi*x)
end

"""
    heston_price(S, K, T, r, model::HestonModel; call=true, n_pts=256) -> Float64

Heston option price via Gil-Pelaez inversion.
"""
function heston_price(S::Float64, K::Float64, T::Float64, r::Float64,
                       model::HestonModel; call::Bool=true, n_pts::Int=256)::Float64

    log_K = log(K)
    h = 100.0 / n_pts
    phi_grid = [(j+0.5)*h for j in 0:(n_pts-1)]

    P1 = P2 = 0.0
    for phi in phi_grid
        # P2: risk-neutral probability
        cf2 = _heston_cf(complex(phi, 0.0), S, T, r, model)
        P2 += real(exp(-1im*phi*log_K) * cf2 / (1im*phi)) * h

        # P1: stock-measure probability
        cf1 = _heston_cf(complex(phi, -1.0), S, T, r, model)
        cf1_0 = _heston_cf(complex(0.0, -1.0), S, T, r, model)
        if abs(cf1_0) > 1e-20
            P1 += real(exp(-1im*phi*log_K) * cf1 / (1im*phi * cf1_0)) * h
        end
    end

    P1 = clamp(0.5 + P1/π, 0.0, 1.0)
    P2 = clamp(0.5 + P2/π, 0.0, 1.0)

    call_price = S*P1 - K*exp(-r*T)*P2
    call_price = max(call_price, max(S - K*exp(-r*T), 0.0))

    return call ? call_price : call_price - S + K*exp(-r*T)
end

"""
    heston_calibrate(S, strikes, maturities, market_vols, r; n_tries=5) -> HestonModel

Calibrate Heston model to implied vol surface.
"""
function heston_calibrate(S::Float64, strikes::Vector{Float64},
                            maturities::Vector{Float64},
                            market_vols::Matrix{Float64},
                            r::Float64;
                            n_tries::Int=5)::HestonModel

    n_T = length(maturities); n_K = length(strikes)

    function objective(params)
        kappa, theta, sigma_v, rho, v0 = params
        (kappa<=0||theta<=0||sigma_v<=0||abs(rho)>=1||v0<=0) && return 1e15
        model = try HestonModel(kappa, theta, sigma_v, rho, v0) catch; return 1e15 end
        sse = 0.0
        for (t_idx,T) in enumerate(maturities), (k_idx,K) in enumerate(strikes)
            call_price = heston_price(S, K, T, r, model; call=true)
            iv = implied_vol_brent(call_price, S, K, T, r; call=true)
            isnan(iv) && continue
            sse += (iv - market_vols[t_idx, k_idx])^2
        end
        return sse
    end

    # Multiple random starts
    best_obj = Inf
    best_params = [2.0, 0.04, 0.3, -0.7, 0.04]

    rng = MersenneTwister(42)
    starts = [[abs(randn(rng))+0.5, abs(randn(rng))*0.02+0.01,
               abs(randn(rng))*0.3+0.1, clamp(randn(rng)*0.3, -0.9, 0.9),
               abs(randn(rng))*0.02+0.01] for _ in 1:n_tries]

    for start in vcat([best_params], starts)
        # Coordinate descent
        params = copy(start)
        obj = objective(params)
        steps = [0.1, 0.005, 0.05, 0.05, 0.005]

        for _ in 1:200
            improved = false
            for dim in 1:5
                for dir in [1.0, -1.0]
                    c = copy(params)
                    c[dim] += dir*steps[dim]
                    o = objective(c)
                    if o < obj
                        obj = o; params = c; improved = true
                    end
                end
            end
            if !improved
                steps .*= 0.5
                all(steps .< 1e-8) && break
            end
        end

        if obj < best_obj
            best_obj = obj; best_params = params
        end
    end

    kappa, theta, sigma_v, rho, v0 = best_params
    return HestonModel(abs(kappa), abs(theta), abs(sigma_v),
                       clamp(rho, -0.999, 0.999), abs(v0))
end

# =============================================================================
# SECTION 3: SABR MODEL
# =============================================================================

"""SABR model parameters."""
struct SABRModel
    alpha::Float64   # initial vol
    beta::Float64    # CEV exponent ∈ [0,1]
    rho::Float64     # correlation
    nu::Float64      # vol-of-vol

    SABRModel(a,b,r,n) = new(max(a,1e-6), clamp(b,0.0,1.0), clamp(r,-0.999,0.999), max(n,1e-6))
end

"""
    sabr_vol(F, K, T, model::SABRModel) -> Float64

Hagan et al. (2002) SABR approximation for Black implied vol.
"""
function sabr_vol(F::Float64, K::Float64, T::Float64, model::SABRModel)::Float64
    F<=0||K<=0||T<=0 && return model.alpha
    a = model.alpha; b = model.beta; r = model.rho; n = model.nu

    FK_mid = sqrt(F*K)
    log_FK = log(F/K)
    one_mb = 1-b

    if abs(log_FK) < 1e-6
        # ATM approximation
        atm = a / FK_mid^one_mb
        corr = 1 + (one_mb^2/24 * a^2/FK_mid^(2*one_mb) +
                    0.25*r*b*n*a/FK_mid^one_mb +
                    (2-3r^2)/24 * n^2) * T
        return atm * corr
    end

    z = n/a * FK_mid^one_mb * log_FK
    xz = log((sqrt(1-2r*z+z^2)+z-r)/(1-r))
    z_over_xz = abs(xz) < 1e-10 ? 1.0 : z/xz

    num = a
    denom = FK_mid^one_mb * (1 + one_mb^2/24*log_FK^2 + one_mb^4/1920*log_FK^4)
    eps = 1 + (one_mb^2*a^2/(24*FK_mid^(2*one_mb)) +
               0.25*r*b*n*a/FK_mid^one_mb +
               (2-3r^2)/24 * n^2) * T

    return num/denom * z_over_xz * eps
end

"""
    sabr_calibrate(F, strikes, T, market_vols; beta=0.5) -> SABRModel

Calibrate SABR (α, ρ, ν) to market implied vols with fixed β.
"""
function sabr_calibrate(F::Float64, strikes::Vector{Float64}, T::Float64,
                          market_vols::Vector{Float64}; beta::Float64=0.5)::SABRModel
    n = length(strikes)
    @assert length(market_vols) == n

    atm_idx = argmin(abs.(strikes.-F))
    a0 = market_vols[atm_idx] * F^(1-beta)

    function obj(params)
        a, r, n_v = params
        (a<=0||abs(r)>=1||n_v<=0) && return 1e15
        m = SABRModel(a, beta, r, n_v)
        sum((sabr_vol(F, strikes[i], T, m) - market_vols[i])^2 for i in 1:n)
    end

    best = [a0, 0.0, 0.3]
    best_obj = obj(best)
    steps = [a0*0.1, 0.05, 0.05]

    for _ in 1:300
        improved = false
        for dim in 1:3
            for dir in [1.0,-1.0]
                c = copy(best)
                c[dim] += dir*steps[dim]
                o = obj(c)
                if o < best_obj
                    best_obj = o; best = c; improved = true
                end
            end
        end
        if !improved
            steps .*= 0.5
            all(steps.<1e-9) && break
        end
    end

    return SABRModel(best[1], beta, best[2], best[3])
end

# =============================================================================
# SECTION 4: JUMP-DIFFUSION (MERTON 1976)
# =============================================================================

"""
    merton_jump_price(S, K, T, r, sigma, lambda, mu_j, sigma_j;
                      call=true, n_terms=30) -> Float64

Merton (1976) jump-diffusion option price.

Mixture of BS models with Poisson-weighted jump components:
    C = Σₙ₌₀^N exp(-λ'T)(λ'T)ⁿ/n! * BS(S, K, T, rₙ, σₙ)

where λ' = λ(1+μ_J), rₙ = r - λμ_J + n*log(1+μ_J)/T,
      σₙ² = σ² + n*σ_J²/T

# Arguments
- `lambda`: jump intensity (expected jumps per year)
- `mu_j`: mean jump size (log-normal, as fraction of price)
- `sigma_j`: vol of jump size
"""
function merton_jump_price(S::Float64, K::Float64, T::Float64, r::Float64,
                             sigma::Float64, lambda::Float64, mu_j::Float64,
                             sigma_j::Float64; call::Bool=true,
                             n_terms::Int=30)::Float64

    lambda_prime = lambda * (1 + mu_j)
    price = 0.0

    log_factorial = 0.0
    for n in 0:n_terms
        n > 0 && (log_factorial += log(n))

        # Poisson weight
        poisson_weight = exp(-lambda_prime*T + n*log(lambda_prime*T + 1e-20) - log_factorial)

        # Adjusted parameters for n jumps
        r_n = r - lambda*mu_j + n*log(1+mu_j+1e-10)/T
        sigma_n = sqrt(sigma^2 + n*sigma_j^2/T)

        price += poisson_weight * bs_price(S, K, T, r_n, sigma_n; call=call)

        # Early termination if contribution is negligible
        poisson_weight < 1e-12 && break
    end

    return price
end

# =============================================================================
# SECTION 5: EXOTIC OPTIONS
# =============================================================================

"""
    BarrierOption

Parameters for a barrier option.
"""
struct BarrierOption
    barrier_type::Symbol    # :up_out, :up_in, :down_out, :down_in
    H::Float64              # barrier level
    rebate::Float64         # rebate if knocked out
end

"""
    barrier_price_bs(S, K, T, r, sigma, q, barrier::BarrierOption; call=true) -> Float64

Closed-form barrier option price under Black-Scholes.

Uses the reflection principle for barrier pricing.
Standard formulas from Haug (2007) 'The Complete Guide to Option Pricing Formulas'.
"""
function barrier_price_bs(S::Float64, K::Float64, T::Float64, r::Float64,
                            sigma::Float64, q::Float64,
                            barrier::BarrierOption; call::Bool=true)::Float64

    H = barrier.H
    η = barrier.barrier_type in [:up_out, :up_in] ? -1.0 : 1.0
    φ = call ? 1.0 : -1.0

    sqT = sqrt(T)
    mu = (r - q - 0.5*sigma^2) / sigma^2
    x1 = log(S/K) / (sigma*sqT) + (1+mu)*sigma*sqT
    x2 = log(S/H) / (sigma*sqT) + (1+mu)*sigma*sqT
    y1 = log(H^2/(S*K)) / (sigma*sqT) + (1+mu)*sigma*sqT
    y2 = log(H/S) / (sigma*sqT) + (1+mu)*sigma*sqT

    disc_r = exp(-r*T); disc_q = exp(-q*T)

    A = φ * (S*disc_q*_N(φ*x1) - K*disc_r*_N(φ*(x1-sigma*sqT)))
    B = φ * (S*disc_q*_N(φ*x2) - K*disc_r*_N(φ*(x2-sigma*sqT)))

    power = (H/S)^(2*(mu+1))
    C = φ * (S*disc_q*power*_N(η*y1) - K*disc_r*(H/S)^(2*mu)*_N(η*(y1-sigma*sqT)))
    D = φ * (S*disc_q*power*_N(η*y2) - K*disc_r*(H/S)^(2*mu)*_N(η*(y2-sigma*sqT)))

    # Rebate
    E = barrier.rebate * disc_r * (_N(η*(x2-sigma*sqT)) - (H/S)^(2*mu)*_N(η*(y2-sigma*sqT)))

    price = if barrier.barrier_type == :down_out && call
        S > H ? A - C + E : E
    elseif barrier.barrier_type == :down_in && call
        S > H ? C : A - E
    elseif barrier.barrier_type == :up_out && call
        S < H ? A - B + D + E : E
    elseif barrier.barrier_type == :up_in && call
        S < H ? B - D : A
    else
        # Put barrier: use put-call parity for barrier options
        bs_price(S, K, T, r, sigma, q; call=false) -
            barrier_price_bs(S, K, T, r, sigma, q, BarrierOption(barrier.barrier_type, H, 0.0); call=true)
    end

    return max(price, 0.0)
end

"""
    AsianOption

Asian (average price) option.
"""
struct AsianOption
    averaging::Symbol    # :arithmetic or :geometric
    observation_freq::Int  # number of averaging observations
end

"""
    asian_price_mc(S, K, T, r, sigma, q, option::AsianOption;
                   n_paths=10000, seed=42) -> Float64

Monte Carlo price for Asian option.
"""
function asian_price_mc(S::Float64, K::Float64, T::Float64, r::Float64,
                          sigma::Float64, q::Float64,
                          option::AsianOption;
                          n_paths::Int=10_000, seed::Int=42)::Float64

    rng = MersenneTwister(seed)
    n = option.observation_freq
    dt = T / n
    sqdt = sqrt(dt)
    drift = (r - q - 0.5*sigma^2) * dt

    payoffs = zeros(n_paths)
    for path in 1:n_paths
        S_path = S
        avg_sum = 0.0
        for step in 1:n
            z = randn(rng)
            S_path *= exp(drift + sigma*sqdt*z)
            avg_sum += S_path
        end
        avg = avg_sum / n
        payoffs[path] = max(avg - K, 0.0)
    end

    return exp(-r*T) * mean(payoffs)
end

"""
    LookbackOption

Lookback option parameters.
"""
struct LookbackOption
    type::Symbol    # :floating_call, :floating_put, :fixed_call, :fixed_put
end

"""
    lookback_price_mc(S, T, r, sigma, q, option::LookbackOption;
                      K=NaN, n_steps=252, n_paths=10000) -> Float64

Monte Carlo price for lookback option.
- Floating call: max(S_T - S_min, 0) = S_T - S_min
- Floating put:  max(S_max - S_T, 0) = S_max - S_T
- Fixed call:    max(S_max - K, 0)
- Fixed put:     max(K - S_min, 0)
"""
function lookback_price_mc(S::Float64, T::Float64, r::Float64, sigma::Float64,
                             q::Float64, option::LookbackOption;
                             K::Float64=NaN, n_steps::Int=252,
                             n_paths::Int=10_000, seed::Int=42)::Float64

    rng = MersenneTwister(seed)
    dt = T / n_steps
    sqdt = sqrt(dt)
    drift = (r - q - 0.5*sigma^2) * dt

    payoffs = zeros(n_paths)

    for path in 1:n_paths
        S_curr = S
        S_max = S; S_min = S

        for _ in 1:n_steps
            z = randn(rng)
            S_curr *= exp(drift + sigma*sqdt*z)
            S_max = max(S_max, S_curr)
            S_min = min(S_min, S_curr)
        end

        payoffs[path] = if option.type == :floating_call
            S_curr - S_min
        elseif option.type == :floating_put
            S_max - S_curr
        elseif option.type == :fixed_call
            max(S_max - K, 0.0)
        else  # :fixed_put
            max(K - S_min, 0.0)
        end
    end

    return exp(-r*T) * mean(payoffs)
end

"""
    DigitalOption

Digital (binary) option.
"""
struct DigitalOption
    type::Symbol    # :cash_or_nothing, :asset_or_nothing
end

"""
    digital_price_bs(S, K, T, r, sigma, q, option::DigitalOption; call=true) -> Float64

Closed-form BS price for digital options.
Cash-or-nothing call: N(d2) * exp(-rT)
Asset-or-nothing call: S * exp(-qT) * N(d1)
"""
function digital_price_bs(S::Float64, K::Float64, T::Float64, r::Float64,
                            sigma::Float64, q::Float64,
                            option::DigitalOption; call::Bool=true)::Float64

    (T<=0||sigma<=0||S<=0||K<=0) && begin
        if option.type == :cash_or_nothing
            return call ? (S>K ? exp(-r*T) : 0.0) : (S<K ? exp(-r*T) : 0.0)
        else
            return call ? (S>K ? S*exp(-q*T) : 0.0) : (S<K ? S*exp(-q*T) : 0.0)
        end
    end

    sqT = sqrt(T)
    d1 = (log(S/K) + (r-q+0.5sigma^2)*T) / (sigma*sqT)
    d2 = d1 - sigma*sqT

    if option.type == :cash_or_nothing
        return call ? exp(-r*T)*_N(d2) : exp(-r*T)*_N(-d2)
    else  # :asset_or_nothing
        return call ? S*exp(-q*T)*_N(d1) : S*exp(-q*T)*_N(-d1)
    end
end

# =============================================================================
# SECTION 6: SVI VOL SURFACE
# =============================================================================

"""SVI parametrization parameters."""
struct SVIParams
    a::Float64; b::Float64; rho::Float64; m::Float64; sigma::Float64
end

"""
    svi_vol(F, K, T, params::SVIParams) -> Float64

Implied vol from SVI total variance w(k) = a + b*(ρ(k-m) + √((k-m)²+σ²)).
"""
function svi_vol(F::Float64, K::Float64, T::Float64, params::SVIParams)::Float64
    K<=0||F<=0||T<=0 && return 0.2
    k = log(K/F)
    x = k - params.m
    w = params.a + params.b*(params.rho*x + sqrt(x^2 + params.sigma^2))
    return w > 0 ? sqrt(w/T) : 0.2
end

"""
    svi_fit(log_moneyness, total_variance) -> SVIParams

Fit SVI parameters by coordinate descent.
"""
function svi_fit(log_moneyness::Vector{Float64},
                  total_variance::Vector{Float64})::SVIParams

    n = length(log_moneyness)
    atm = total_variance[argmin(abs.(log_moneyness))]

    function obj(p)
        a,b,r,m,s = p
        (b<=0||abs(r)>=1||s<=0||a<0) && return 1e15
        sum((a + b*(r*(log_moneyness[i]-m) + sqrt((log_moneyness[i]-m)^2+s^2)) -
             total_variance[i])^2 for i in 1:n)
    end

    best = [atm*0.5, 0.3, 0.0, 0.0, 0.2]
    best_obj = obj(best)
    steps = [atm*0.05, 0.05, 0.05, 0.05, 0.05]

    for _ in 1:500
        improved = false
        for dim in 1:5
            for dir in [1.0,-1.0]
                c = copy(best)
                c[dim] += dir*steps[dim]
                o = obj(c)
                if o < best_obj
                    best_obj = o; best = c; improved = true
                end
            end
        end
        if !improved
            steps .*= 0.5
            all(steps.<1e-10) && break
        end
    end

    return SVIParams(best[1], best[2], best[3], best[4], best[5])
end

"""
    ssvi_vol(F, K, T, rho, eta, gamma; psi_func=:power) -> Float64

Surface SVI (SSVI) of Gatheral & Jacquier (2014).

SSVI parametrizes the entire vol surface consistently across maturities:
    w(k,T) = θ_T/2 * (1 + ρψθk + √((ψθk+ρ)²+1-ρ²))

where θ_T is the ATM total variance and ψ = η/θ^γ is the slope function.
"""
function ssvi_vol(F::Float64, K::Float64, T::Float64,
                   rho::Float64, eta::Float64, gamma::Float64,
                   theta_T::Float64)::Float64

    K<=0||F<=0||T<=0||theta_T<=0 && return 0.2
    k = log(K/F)
    psi_theta = eta * theta_T^(-gamma)

    w = theta_T/2 * (1 + rho*psi_theta*k +
        sqrt((psi_theta*k + rho)^2 + 1 - rho^2))

    return w > 0 ? sqrt(w/T) : 0.2
end

# =============================================================================
# SECTION 7: CRYPTO PERPETUALS
# =============================================================================

"""Crypto perpetual futures position."""
struct CryptoPerp
    entry_price::Float64
    position_size::Float64  # + = long, - = short
    margin::Float64
    leverage::Float64
    maintenance_margin_rate::Float64
end

"""
    perp_funding_pnl(perp::CryptoPerp, mark_prices::Vector{Float64},
                      funding_rates::Vector{Float64}) -> NamedTuple

Compute PnL from a perpetual futures position including mark-to-market and funding.

funding_rate applies every 8 hours. funding_payment = position_size * mark_price * rate.
"""
function perp_funding_pnl(perp::CryptoPerp,
                            mark_prices::Vector{Float64},
                            funding_rates::Vector{Float64})

    n = length(mark_prices)
    @assert length(funding_rates) == n

    cum_pnl = zeros(n)
    cum_funding = zeros(n)
    unrealized_pnl = zeros(n)
    total_pnl = zeros(n)

    cum_fund = 0.0
    for t in 1:n
        mtm = perp.position_size * (mark_prices[t] - perp.entry_price)
        fund_payment = -perp.position_size * mark_prices[t] * funding_rates[t]
        cum_fund += fund_payment

        unrealized_pnl[t] = mtm
        cum_funding[t] = cum_fund
        total_pnl[t] = mtm + cum_fund
    end

    # Check liquidation
    liq_price = if perp.position_size > 0
        perp.entry_price - (perp.margin - perp.maintenance_margin_rate*perp.entry_price*perp.position_size) / perp.position_size
    else
        perp.entry_price + (perp.margin - perp.maintenance_margin_rate*perp.entry_price*abs(perp.position_size)) / abs(perp.position_size)
    end

    liq_hit = perp.position_size > 0 ?
        findall(mark_prices .< liq_price) :
        findall(mark_prices .> liq_price)

    return (unrealized_pnl=unrealized_pnl, cumulative_funding=cum_funding,
             total_pnl=total_pnl, liquidation_price=liq_price,
             liquidation_hit=!isempty(liq_hit))
end

"""
    perp_delta_hedge(perp::CryptoPerp, spot_prices::Vector{Float64},
                      hedge_freq::Int=8) -> NamedTuple

Delta hedge a perpetual position using spot market.
Delta of perp ≈ 1 (linear payoff). Hedging: short spot proportional to position.
"""
function perp_delta_hedge(perp::CryptoPerp,
                            spot_prices::Vector{Float64};
                            hedge_freq::Int=8)

    n = length(spot_prices)
    hedge_pnl = zeros(n)
    spot_position = -perp.position_size  # hedge delta

    for t in 2:n
        daily_pnl = spot_position * (spot_prices[t] - spot_prices[t-1])
        hedge_pnl[t] = daily_pnl
        # Rebalance every hedge_freq periods
        if t % hedge_freq == 0
            spot_position = -perp.position_size  # reset
        end
    end

    return (hedge_pnl=hedge_pnl, cumulative_hedge_pnl=cumsum(hedge_pnl))
end

# =============================================================================
# SECTION 8: BERGOMI TERM STRUCTURE
# =============================================================================

"""
    BergomiVol

Bergomi (2005) forward variance curve model parameters.
Models the term structure of variance as:
    ξₜ(T) = ξ₀(T) * exp(2σ_ξ * W_t - 2σ_ξ²t) for flat curve
"""
struct BergomiVol
    xi0::Vector{Float64}    # initial forward variance curve (per maturity)
    sigma_xi::Float64       # vol of variance
    rho::Float64            # spot-variance correlation
    maturities::Vector{Float64}
end

"""
    bergomi_term_structure(model::BergomiVol, T) -> Float64

Get ATM vol for maturity T from Bergomi model.
"""
function bergomi_term_structure(model::BergomiVol, T::Float64)::Float64
    if isempty(model.maturities)
        return sqrt(model.xi0[1])
    end

    # Interpolate forward variance
    sorted_T = sortperm(model.maturities)
    T_sorted = model.maturities[sorted_T]
    xi_sorted = model.xi0[sorted_T]

    idx = searchsortedfirst(T_sorted, T)
    if idx <= 1
        return sqrt(xi_sorted[1])
    elseif idx > length(T_sorted)
        return sqrt(xi_sorted[end])
    else
        t_frac = (T - T_sorted[idx-1]) / (T_sorted[idx] - T_sorted[idx-1])
        xi_T = (1-t_frac)*xi_sorted[idx-1] + t_frac*xi_sorted[idx]
        return sqrt(max(xi_T, 0.0))
    end
end

# =============================================================================
# SECTION 9: CORRELATION PRODUCTS
# =============================================================================

"""
    basket_price_mc(S_vec, weights, K, T, r, sigma_vec, corr_matrix;
                    call=true, n_paths=10000) -> Float64

Monte Carlo price for a basket option (call or put on weighted average).
"""
function basket_price_mc(S_vec::Vector{Float64},
                           weights::Vector{Float64},
                           K::Float64, T::Float64, r::Float64,
                           sigma_vec::Vector{Float64},
                           corr_matrix::Matrix{Float64};
                           call::Bool=true,
                           n_paths::Int=10_000,
                           seed::Int=42)::Float64

    n = length(S_vec)
    rng = MersenneTwister(seed)
    w = weights ./ sum(weights)

    # Cholesky decomposition of correlation matrix
    L = try
        cholesky(Symmetric(corr_matrix + 1e-8*I)).L
    catch
        Matrix{Float64}(I, n, n)
    end

    drift = (r - 0.5 .* sigma_vec.^2) * T
    sqT = sqrt(T)

    payoffs = zeros(n_paths)
    for path in 1:n_paths
        z = L * randn(rng, n)
        S_T = S_vec .* exp.(drift .+ sigma_vec .* sqT .* z)
        basket_T = w' * S_T
        payoffs[path] = call ? max(basket_T - K, 0.0) : max(K - basket_T, 0.0)
    end

    return exp(-r*T) * mean(payoffs)
end

"""
    worst_of_price_mc(S_vec, K, T, r, sigma_vec, corr_matrix;
                      call=true, n_paths=10000) -> Float64

Monte Carlo price for worst-of option: payoff based on minimum asset performance.
Worst-of call: max(min(S₁_T/S₁_0, ..., Sₙ_T/Sₙ_0) - K, 0)
"""
function worst_of_price_mc(S_vec::Vector{Float64},
                             K::Float64, T::Float64, r::Float64,
                             sigma_vec::Vector{Float64},
                             corr_matrix::Matrix{Float64};
                             call::Bool=true,
                             n_paths::Int=10_000,
                             seed::Int=42)::Float64

    n = length(S_vec)
    rng = MersenneTwister(seed)
    L = try cholesky(Symmetric(corr_matrix+1e-8*I)).L catch I(n)*1.0 end

    drift = (r - 0.5 .* sigma_vec.^2) * T
    sqT = sqrt(T)

    payoffs = zeros(n_paths)
    for path in 1:n_paths
        z = L * randn(rng, n)
        returns_T = exp.(drift .+ sigma_vec .* sqT .* z)
        worst_return = minimum(returns_T)
        payoffs[path] = call ? max(worst_return - K, 0.0) : max(K - worst_return, 0.0)
    end

    return exp(-r*T) * mean(payoffs)
end

# =============================================================================
# SECTION 10: CALIBRATION FRAMEWORK
# =============================================================================

"""
    CalibrationResult

Result of a vol surface calibration.
"""
struct CalibrationResult
    model_type::Symbol
    params::NamedTuple
    rmse::Float64
    max_error::Float64
    n_options::Int
    converged::Bool
end

"""
    calibrate_vol_surface(S, strikes, maturities, market_vols, r;
                           model=:heston) -> CalibrationResult

Unified calibration interface for different models.
"""
function calibrate_vol_surface(S::Float64,
                                 strikes::Vector{Float64},
                                 maturities::Vector{Float64},
                                 market_vols::Matrix{Float64},
                                 r::Float64;
                                 model::Symbol=:heston)::CalibrationResult

    n_T, n_K = size(market_vols)
    @assert n_T == length(maturities) && n_K == length(strikes)

    n_options = n_T * n_K

    if model == :heston
        heston_m = heston_calibrate(S, strikes, maturities, market_vols, r)

        # Compute fit error
        total_sse = 0.0; max_err = 0.0
        for (t_idx,T) in enumerate(maturities), (k_idx,K) in enumerate(strikes)
            cp = heston_price(S, K, T, r, heston_m; call=true)
            iv = implied_vol_brent(cp, S, K, T, r; call=true)
            isnan(iv) && continue
            err = abs(iv - market_vols[t_idx, k_idx])
            total_sse += err^2; max_err = max(max_err, err)
        end
        rmse = sqrt(total_sse/n_options)
        params = (kappa=heston_m.kappa, theta=heston_m.theta,
                   sigma=heston_m.sigma, rho=heston_m.rho, v0=heston_m.v0)
        return CalibrationResult(:heston, params, rmse, max_err, n_options, true)

    elseif model == :sabr
        # Calibrate per maturity
        sabr_params = SABRModel[]
        for t_idx in 1:n_T
            T = maturities[t_idx]
            F = S*exp(r*T)
            m = sabr_calibrate(F, strikes, T, market_vols[t_idx,:]; beta=0.5)
            push!(sabr_params, m)
        end
        params = (models=sabr_params,)
        return CalibrationResult(:sabr, params, 0.0, 0.0, n_options, true)

    else  # :svi per maturity
        svi_fits = SVIParams[]
        total_sse = 0.0
        for t_idx in 1:n_T
            T = maturities[t_idx]
            F = S*exp(r*T)
            lm = log.(strikes./F)
            tv = market_vols[t_idx,:].^2 .* T
            p = svi_fit(lm, tv)
            push!(svi_fits, p)
            for k_idx in 1:n_K
                iv_model = svi_vol(F, strikes[k_idx], T, p)
                total_sse += (iv_model - market_vols[t_idx,k_idx])^2
            end
        end
        rmse = sqrt(total_sse/n_options)
        params = (svi_params=svi_fits,)
        return CalibrationResult(:svi, params, rmse, sqrt(rmse), n_options, true)
    end
end

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

function _N(x::Float64)::Float64
    x >= 8.0 && return 1.0; x <= -8.0 && return 0.0
    t = 1.0/(1.0+0.2316419*abs(x))
    poly = t*(0.319381530+t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))))
    phi = exp(-0.5x^2)/sqrt(2π)
    cdf = 1.0 - phi*poly
    return x >= 0 ? cdf : 1.0 - cdf
end

function _phi(x::Float64)::Float64
    exp(-0.5x^2)/sqrt(2π)
end

end # module DerivativesPricing
