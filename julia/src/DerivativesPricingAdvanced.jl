module DerivativesPricingAdvanced

using LinearAlgebra
using Statistics
using Random

# ============================================================================
# Utility Functions
# ============================================================================

function normal_cdf(x::Float64)::Float64
    if x < -8.0 return 0.0 end
    if x > 8.0 return 1.0 end
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    pdf_val = exp(-0.5 * x * x) / sqrt(2.0 * pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
           t * (-1.821255978 + t * 1.330274429))))
    cdf_val = 1.0 - pdf_val * poly
    return x >= 0.0 ? cdf_val : 1.0 - cdf_val
end

function normal_pdf(x::Float64)::Float64
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)
end

function normal_inv(p::Float64)::Float64
    if p <= 0.0 return -8.0 end
    if p >= 1.0 return 8.0 end
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    p_low = 0.02425
    if p < p_low
        q = sqrt(-2.0 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    elseif p <= 1.0 - p_low
        q = p - 0.5; r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6]) * q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    end
end

# ============================================================================
# SECTION 1: Black-Scholes Full Greeks
# ============================================================================

struct BSResult
    price::Float64
    delta::Float64
    gamma::Float64
    theta::Float64
    vega::Float64
    rho::Float64
    vanna::Float64
    volga::Float64
    charm::Float64
    speed::Float64
    color::Float64
    dual_delta::Float64
end

function bs_d1d2(S::Float64, K::Float64, r::Float64, q::Float64,
                  sigma::Float64, T::Float64)
    d1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2
end

"""
    bs_call_full(S, K, r, q, sigma, T)

Black-Scholes call with full Greeks including higher-order.
"""
function bs_call_full(S::Float64, K::Float64, r::Float64, q::Float64,
                       sigma::Float64, T::Float64)::BSResult
    if T <= 0.0
        price = max(S * exp(-q * T) - K * exp(-r * T), 0.0)
        delta = S > K ? exp(-q * T) : 0.0
        return BSResult(price, delta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end

    d1, d2 = bs_d1d2(S, K, r, q, sigma, T)
    sqrtT = sqrt(T)
    Nd1 = normal_cdf(d1)
    Nd2 = normal_cdf(d2)
    nd1 = normal_pdf(d1)
    eqT = exp(-q * T)
    erT = exp(-r * T)

    price = S * eqT * Nd1 - K * erT * Nd2

    # First-order Greeks
    delta = eqT * Nd1
    gamma = eqT * nd1 / (S * sigma * sqrtT)
    theta = -(S * eqT * nd1 * sigma) / (2.0 * sqrtT) +
            q * S * eqT * Nd1 - r * K * erT * Nd2
    vega = S * eqT * nd1 * sqrtT
    rho_val = K * T * erT * Nd2

    # Second-order Greeks
    vanna = -eqT * nd1 * d2 / sigma  # dDelta/dSigma
    volga = vega * d1 * d2 / sigma     # dVega/dSigma
    charm = -eqT * (nd1 * (2.0 * (r - q) * T - d2 * sigma * sqrtT) /
            (2.0 * T * sigma * sqrtT) + q * Nd1)  # dDelta/dT

    # Third-order Greeks
    speed = -gamma / S * (d1 / (sigma * sqrtT) + 1.0)
    color = -eqT * nd1 / (2.0 * S * T * sigma * sqrtT) *
            (2.0 * q * T + 1.0 + (2.0 * (r - q) * T - d2 * sigma * sqrtT) /
             (sigma * sqrtT) * d1)

    dual_delta = -erT * Nd2

    return BSResult(price, delta, gamma, theta / 365.0, vega / 100.0, rho_val / 100.0,
                     vanna, volga, charm, speed, color, dual_delta)
end

"""
    bs_put_full(S, K, r, q, sigma, T)

Black-Scholes put with full Greeks.
"""
function bs_put_full(S::Float64, K::Float64, r::Float64, q::Float64,
                      sigma::Float64, T::Float64)::BSResult
    if T <= 0.0
        price = max(K * exp(-r * T) - S * exp(-q * T), 0.0)
        delta = S < K ? -exp(-q * T) : 0.0
        return BSResult(price, delta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end

    d1, d2 = bs_d1d2(S, K, r, q, sigma, T)
    sqrtT = sqrt(T)
    Nmd1 = normal_cdf(-d1)
    Nmd2 = normal_cdf(-d2)
    nd1 = normal_pdf(d1)
    eqT = exp(-q * T)
    erT = exp(-r * T)

    price = K * erT * Nmd2 - S * eqT * Nmd1
    delta = -eqT * Nmd1
    gamma = eqT * nd1 / (S * sigma * sqrtT)
    theta = -(S * eqT * nd1 * sigma) / (2.0 * sqrtT) -
            q * S * eqT * Nmd1 + r * K * erT * Nmd2
    vega = S * eqT * nd1 * sqrtT
    rho_val = -K * T * erT * Nmd2

    vanna = -eqT * nd1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    charm = -eqT * (nd1 * (2.0 * (r - q) * T - d2 * sigma * sqrtT) /
            (2.0 * T * sigma * sqrtT) - q * Nmd1)
    speed = -gamma / S * (d1 / (sigma * sqrtT) + 1.0)
    color = 0.0  # Simplified
    dual_delta = erT * Nmd2

    return BSResult(price, delta, gamma, theta / 365.0, vega / 100.0, rho_val / 100.0,
                     vanna, volga, charm, speed, color, dual_delta)
end

"""
    bs_digital_call(S, K, r, q, sigma, T)

Digital (binary) call option: pays 1 if S_T > K.
"""
function bs_digital_call(S::Float64, K::Float64, r::Float64, q::Float64,
                          sigma::Float64, T::Float64)::Float64
    if T <= 0.0
        return S > K ? exp(-r * T) : 0.0
    end
    _, d2 = bs_d1d2(S, K, r, q, sigma, T)
    return exp(-r * T) * normal_cdf(d2)
end

"""
    bs_digital_put(S, K, r, q, sigma, T)

Digital put: pays 1 if S_T < K.
"""
function bs_digital_put(S::Float64, K::Float64, r::Float64, q::Float64,
                         sigma::Float64, T::Float64)::Float64
    if T <= 0.0
        return S < K ? exp(-r * T) : 0.0
    end
    _, d2 = bs_d1d2(S, K, r, q, sigma, T)
    return exp(-r * T) * normal_cdf(-d2)
end

"""
    bs_power_option(S, K, r, q, sigma, T, n)

Power call option: max(S^n - K, 0).
"""
function bs_power_option(S::Float64, K::Float64, r::Float64, q::Float64,
                          sigma::Float64, T::Float64, n::Float64)::Float64
    if T <= 0.0
        return max(S^n - K, 0.0)
    end
    # Forward value of S^n
    F_n = S^n * exp((n * (r - q) + 0.5 * n * (n - 1) * sigma^2) * T)
    sigma_n = n * sigma
    d1 = (log(F_n / K) + 0.5 * sigma_n^2 * T) / (sigma_n * sqrt(T))
    d2 = d1 - sigma_n * sqrt(T)
    return exp(-r * T) * (F_n * normal_cdf(d1) - K * normal_cdf(d2))
end

"""
    bs_implied_vol(market_price, S, K, r, q, T, is_call; tol=1e-8, max_iter=100)

Newton-Raphson implied volatility solver.
"""
function bs_implied_vol(market_price::Float64, S::Float64, K::Float64,
                        r::Float64, q::Float64, T::Float64, is_call::Bool;
                        tol::Float64=1e-8, max_iter::Int=100)::Float64
    sigma = 0.2  # Initial guess

    for _ in 1:max_iter
        if is_call
            result = bs_call_full(S, K, r, q, sigma, T)
        else
            result = bs_put_full(S, K, r, q, sigma, T)
        end

        diff = result.price - market_price
        vega_raw = S * exp(-q * T) * normal_pdf(bs_d1d2(S, K, r, q, sigma, T)[1]) * sqrt(T)

        if abs(vega_raw) < 1e-15
            break
        end

        sigma -= diff / vega_raw
        sigma = clamp(sigma, 0.001, 5.0)

        if abs(diff) < tol
            break
        end
    end

    return sigma
end

# ============================================================================
# SECTION 2: Heston Model
# ============================================================================

"""
    HestonParams

Heston (1993) stochastic volatility model parameters.
dS = (r-q)*S*dt + sqrt(v)*S*dW_1
dv = kappa*(theta-v)*dt + xi*sqrt(v)*dW_2
corr(dW_1, dW_2) = rho
"""
struct HestonParams
    v0::Float64      # Initial variance
    kappa::Float64   # Mean reversion speed
    theta::Float64   # Long-run variance
    xi::Float64      # Vol of vol
    rho::Float64     # Correlation
end

"""
    heston_characteristic_function(u, S, K, r, q, T, params::HestonParams)

Heston characteristic function phi(u) = E[exp(iu*ln(S_T))].
"""
function heston_characteristic_function(u::ComplexF64, S::Float64, r::Float64,
                                         q::Float64, T::Float64,
                                         params::HestonParams)::ComplexF64
    kappa = params.kappa
    theta = params.theta
    xi = params.xi
    rho = params.rho
    v0 = params.v0

    d = sqrt((rho * xi * im * u - kappa)^2 + xi^2 * (im * u + u^2))
    g = (kappa - rho * xi * im * u - d) / (kappa - rho * xi * im * u + d)

    C = (r - q) * im * u * T +
        (kappa * theta / xi^2) * ((kappa - rho * xi * im * u - d) * T -
         2.0 * log((1.0 - g * exp(-d * T)) / (1.0 - g)))

    D = ((kappa - rho * xi * im * u - d) / xi^2) *
        ((1.0 - exp(-d * T)) / (1.0 - g * exp(-d * T)))

    return exp(C + D * v0 + im * u * log(S))
end

"""
    heston_call_price_fft(S, K, r, q, T, params::HestonParams; N=4096, alpha=1.5, eta=0.25)

Carr-Madan FFT pricing for Heston model.
"""
function heston_call_price_fft(S::Float64, K::Float64, r::Float64, q::Float64,
                                T::Float64, params::HestonParams;
                                N::Int=4096, alpha::Float64=1.5,
                                eta::Float64=0.25)::Float64
    lambda = 2.0 * pi / (N * eta)
    b = N * lambda / 2.0

    # Construct FFT input
    x = Vector{ComplexF64}(undef, N)
    for j in 1:N
        v_j = (j - 1) * eta
        u = v_j - (alpha + 1.0) * im

        # Modified characteristic function
        phi = heston_characteristic_function(u, S, r, q, T, params)

        # Damped call transform
        psi = exp(-r * T) * phi / (alpha^2 + alpha - v_j^2 + im * (2.0 * alpha + 1.0) * v_j)

        # Simpson's weights
        if j == 1
            simpson = eta / 3.0
        elseif j % 2 == 0
            simpson = eta * 4.0 / 3.0
        else
            simpson = eta * 2.0 / 3.0
        end

        x[j] = exp(im * b * v_j) * psi * simpson
    end

    # FFT (Cooley-Tukey radix-2 DIT)
    fft_result = _fft(x)

    # Extract price at desired strike
    log_K = log(K)
    k_grid = [-b + (j - 1) * lambda for j in 1:N]

    # Find nearest grid point to log(K)
    idx = argmin(abs.(k_grid .- log_K))
    call_price = real(exp(-alpha * k_grid[idx]) * fft_result[idx] / pi)

    return max(call_price, 0.0)
end

"""
    _fft(x)

Radix-2 Cooley-Tukey FFT.
"""
function _fft(x::Vector{ComplexF64})::Vector{ComplexF64}
    N = length(x)
    if N == 1
        return x
    end

    if N & (N - 1) != 0
        # Not power of 2, use DFT
        return _dft(x)
    end

    even = _fft(x[1:2:end])
    odd = _fft(x[2:2:end])

    result = Vector{ComplexF64}(undef, N)
    for k in 1:(N÷2)
        twiddle = exp(-2.0 * pi * im * (k - 1) / N)
        result[k] = even[k] + twiddle * odd[k]
        result[k + N÷2] = even[k] - twiddle * odd[k]
    end

    return result
end

function _dft(x::Vector{ComplexF64})::Vector{ComplexF64}
    N = length(x)
    result = Vector{ComplexF64}(undef, N)
    for k in 1:N
        s = 0.0 + 0.0im
        for n in 1:N
            s += x[n] * exp(-2.0 * pi * im * (k - 1) * (n - 1) / N)
        end
        result[k] = s
    end
    return result
end

"""
    heston_call_price_quad(S, K, r, q, T, params::HestonParams; num_points=200)

Heston call price via direct numerical integration (Gauss-Laguerre).
"""
function heston_call_price_quad(S::Float64, K::Float64, r::Float64, q::Float64,
                                 T::Float64, params::HestonParams;
                                 num_points::Int=200)::Float64
    du = 0.5
    integral_1 = 0.0
    integral_2 = 0.0

    for j in 1:num_points
        u = j * du

        phi1 = heston_characteristic_function(ComplexF64(u - im), S, r, q, T, params)
        phi1 /= (S * exp((r - q) * T))  # Normalize for P1

        integrand1 = real(exp(-im * u * log(K)) * phi1 / (im * u))
        integral_1 += integrand1 * du

        phi2 = heston_characteristic_function(ComplexF64(u), S, r, q, T, params)
        integrand2 = real(exp(-im * u * log(K)) * phi2 / (im * u))
        integral_2 += integrand2 * du
    end

    P1 = 0.5 + integral_1 / pi
    P2 = 0.5 + integral_2 / pi

    call = S * exp(-q * T) * P1 - K * exp(-r * T) * P2
    return max(call, 0.0)
end

"""
    heston_calibrate_lm(market_prices, strikes, maturities, S, r, q, is_call;
                         max_iter=100, tol=1e-6)

Calibrate Heston model via Levenberg-Marquardt.
"""
function heston_calibrate_lm(market_prices::Vector{Float64},
                              strikes::Vector{Float64},
                              maturities::Vector{Float64},
                              S::Float64, r::Float64, q::Float64,
                              is_call::Vector{Bool};
                              max_iter::Int=100, tol::Float64=1e-6)::HestonParams
    n = length(market_prices)

    # Initial guess: [v0, kappa, theta, xi, rho]
    x = [0.04, 2.0, 0.04, 0.3, -0.7]
    lambda_lm = 0.01

    function model_prices(params_vec)
        hp = HestonParams(max(params_vec[1], 0.001), max(params_vec[2], 0.01),
                           max(params_vec[3], 0.001), max(params_vec[4], 0.01),
                           clamp(params_vec[5], -0.99, 0.99))
        prices = Vector{Float64}(undef, n)
        for i in 1:n
            price = heston_call_price_quad(S, strikes[i], r, q, maturities[i], hp;
                                            num_points=100)
            if !is_call[i]
                # Put-call parity
                price = price - S * exp(-q * maturities[i]) + strikes[i] * exp(-r * maturities[i])
            end
            prices[i] = max(price, 0.0)
        end
        return prices
    end

    for iter in 1:max_iter
        prices = model_prices(x)
        residuals = prices - market_prices
        sse = sum(residuals.^2)

        if sse < tol
            break
        end

        # Jacobian via finite differences
        J = zeros(n, 5)
        eps_fd = 1e-5
        for j in 1:5
            x_bump = copy(x)
            x_bump[j] += eps_fd
            prices_bump = model_prices(x_bump)
            J[:, j] = (prices_bump - prices) / eps_fd
        end

        # LM update: (J'J + lambda*diag(J'J)) * dx = -J'r
        JtJ = J' * J
        Jtr = J' * residuals
        damping = lambda_lm * Diagonal(diag(JtJ) .+ 1e-10)

        dx = -(JtJ + damping) \ Jtr

        x_new = x + dx
        # Enforce bounds
        x_new[1] = max(x_new[1], 0.001)
        x_new[2] = max(x_new[2], 0.01)
        x_new[3] = max(x_new[3], 0.001)
        x_new[4] = max(x_new[4], 0.01)
        x_new[5] = clamp(x_new[5], -0.99, 0.99)

        prices_new = model_prices(x_new)
        sse_new = sum((prices_new - market_prices).^2)

        if sse_new < sse
            x = x_new
            lambda_lm = max(lambda_lm / 10.0, 1e-10)
        else
            lambda_lm = min(lambda_lm * 10.0, 1e6)
        end
    end

    return HestonParams(x[1], x[2], x[3], x[4], x[5])
end

"""
    heston_mc(S, K, r, q, T, params::HestonParams, num_paths, num_steps;
              seed=42, is_call=true)

Heston model Monte Carlo with QE scheme (Andersen 2008).
"""
function heston_mc(S::Float64, K::Float64, r::Float64, q::Float64, T::Float64,
                    params::HestonParams, num_paths::Int, num_steps::Int;
                    seed::Int=42, is_call::Bool=true)::Float64
    rng = Random.MersenneTwister(seed)
    dt = T / num_steps
    kappa = params.kappa
    theta = params.theta
    xi = params.xi
    rho = params.rho

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        s = log(S)
        v = params.v0

        for step in 1:num_steps
            z1 = randn(rng)
            z2 = rho * z1 + sqrt(1.0 - rho^2) * randn(rng)

            v_plus = max(v, 0.0)
            s += (r - q - 0.5 * v_plus) * dt + sqrt(v_plus * dt) * z1
            v += kappa * (theta - v_plus) * dt + xi * sqrt(v_plus * dt) * z2
            v = max(v, 0.0)
        end

        S_T = exp(s)
        if is_call
            payoffs[path] = max(S_T - K, 0.0)
        else
            payoffs[path] = max(K - S_T, 0.0)
        end
    end

    return exp(-r * T) * mean(payoffs)
end

# ============================================================================
# SECTION 3: SABR Model
# ============================================================================

"""
    SABRParams

SABR model parameters.
"""
struct SABRParams
    alpha::Float64   # Initial vol level
    beta::Float64    # CEV exponent (0 to 1)
    rho::Float64     # Correlation
    nu::Float64      # Vol of vol
end

"""
    sabr_implied_vol(F, K, T, params::SABRParams)

Hagan (2002) SABR implied volatility formula.
"""
function sabr_implied_vol(F::Float64, K::Float64, T::Float64,
                           params::SABRParams)::Float64
    alpha = params.alpha
    beta = params.beta
    rho = params.rho
    nu = params.nu

    if abs(F - K) < 1e-10 * F
        # ATM formula
        Fmid = F
        sigma = alpha / Fmid^(1.0 - beta) *
                (1.0 + ((1.0 - beta)^2 / 24.0 * alpha^2 / Fmid^(2.0 * (1.0 - beta)) +
                 0.25 * rho * beta * nu * alpha / Fmid^(1.0 - beta) +
                 (2.0 - 3.0 * rho^2) / 24.0 * nu^2) * T)
        return max(sigma, 1e-6)
    end

    FK = F * K
    FK_beta = FK^((1.0 - beta) / 2.0)
    log_FK = log(F / K)

    z = (nu / alpha) * FK_beta * log_FK
    x = log((sqrt(1.0 - 2.0 * rho * z + z^2) + z - rho) / (1.0 - rho))

    if abs(x) < 1e-10
        x = z
    end

    # Leading term
    A = alpha / (FK_beta * (1.0 + (1.0 - beta)^2 / 24.0 * log_FK^2 +
                 (1.0 - beta)^4 / 1920.0 * log_FK^4))

    B = z / x

    # Correction term
    C = 1.0 + ((1.0 - beta)^2 / 24.0 * alpha^2 / FK^(1.0 - beta) +
         0.25 * rho * beta * nu * alpha / FK_beta +
         (2.0 - 3.0 * rho^2) / 24.0 * nu^2) * T

    sigma = A * B * C
    return max(sigma, 1e-6)
end

"""
    sabr_calibrate(F, strikes, market_vols, T, beta)

Calibrate SABR (alpha, rho, nu) given fixed beta.
"""
function sabr_calibrate(F::Float64, strikes::Vector{Float64},
                        market_vols::Vector{Float64}, T::Float64,
                        beta::Float64)::SABRParams
    n = length(strikes)

    # Grid search + refinement
    best_sse = Inf
    best_alpha = 0.2
    best_rho = 0.0
    best_nu = 0.3

    for alpha in 0.01:0.02:1.0
        for rho in -0.9:0.1:0.9
            for nu in 0.05:0.05:2.0
                params = SABRParams(alpha, beta, rho, nu)
                sse = 0.0
                for i in 1:n
                    model_vol = sabr_implied_vol(F, strikes[i], T, params)
                    sse += (model_vol - market_vols[i])^2
                end
                if sse < best_sse
                    best_sse = sse
                    best_alpha = alpha
                    best_rho = rho
                    best_nu = nu
                end
            end
        end
    end

    # Refinement via Newton-like steps
    x = [best_alpha, best_rho, best_nu]
    for iter in 1:50
        params = SABRParams(x[1], beta, x[2], x[3])
        residuals = [sabr_implied_vol(F, strikes[i], T, params) - market_vols[i] for i in 1:n]

        J = zeros(n, 3)
        eps_fd = 1e-5
        for j in 1:3
            xb = copy(x)
            xb[j] += eps_fd
            params_b = SABRParams(xb[1], beta, xb[2], xb[3])
            for i in 1:n
                J[i, j] = (sabr_implied_vol(F, strikes[i], T, params_b) -
                           sabr_implied_vol(F, strikes[i], T, params)) / eps_fd
            end
        end

        dx = -(J' * J + 0.001 * I) \ (J' * residuals)
        x += 0.5 * dx
        x[1] = max(x[1], 0.001)
        x[2] = clamp(x[2], -0.99, 0.99)
        x[3] = max(x[3], 0.001)
    end

    return SABRParams(x[1], beta, x[2], x[3])
end

# ============================================================================
# SECTION 4: Local Volatility
# ============================================================================

"""
    dupire_local_vol(strikes, maturities, implied_vols, S, r, q)

Dupire (1994) local volatility from implied volatility surface.
sigma_local^2(K,T) = (dC/dT + (r-q)*K*dC/dK + q*C) / (0.5*K^2*d2C/dK2)
"""
function dupire_local_vol(strikes::Vector{Float64}, maturities::Vector{Float64},
                           implied_vols::Matrix{Float64}, S::Float64,
                           r::Float64, q::Float64)::Matrix{Float64}
    nK = length(strikes)
    nT = length(maturities)
    local_vol = zeros(nK, nT)

    # Compute call prices on grid
    call_prices = zeros(nK, nT)
    for i in 1:nK
        for j in 1:nT
            sigma = implied_vols[i, j]
            call_prices[i, j] = bs_call_full(S, strikes[i], r, q, sigma, maturities[j]).price
        end
    end

    for i in 2:(nK-1)
        for j in 2:(nT-1)
            K = strikes[i]
            T_val = maturities[j]

            # dC/dT
            dCdT = (call_prices[i, j+1] - call_prices[i, j-1]) /
                    (maturities[j+1] - maturities[j-1])

            # dC/dK
            dCdK = (call_prices[i+1, j] - call_prices[i-1, j]) /
                    (strikes[i+1] - strikes[i-1])

            # d2C/dK2
            d2CdK2 = (call_prices[i+1, j] - 2.0 * call_prices[i, j] + call_prices[i-1, j]) /
                      ((strikes[i+1] - strikes[i]) * (strikes[i] - strikes[i-1]))

            numerator = dCdT + (r - q) * K * dCdK + q * call_prices[i, j]
            denominator = 0.5 * K^2 * d2CdK2

            if denominator > 1e-10
                local_vol[i, j] = sqrt(max(numerator / denominator, 0.0))
            else
                local_vol[i, j] = implied_vols[i, j]
            end
        end
    end

    # Fill boundaries
    for j in 1:nT
        local_vol[1, j] = local_vol[2, j]
        local_vol[nK, j] = local_vol[nK-1, j]
    end
    for i in 1:nK
        local_vol[i, 1] = local_vol[i, 2]
        local_vol[i, nT] = local_vol[i, nT-1]
    end

    return local_vol
end

"""
    trinomial_tree_local_vol(S, r, q, T, num_steps, local_vol_func)

Trinomial tree pricing with local volatility.
local_vol_func(S, t) returns local vol at (S, t).
"""
function trinomial_tree_local_vol(S::Float64, K::Float64, r::Float64, q::Float64,
                                   T::Float64, num_steps::Int,
                                   local_vol_func::Function;
                                   is_call::Bool=true)::Float64
    dt = T / num_steps
    # Tree parameters
    dx = sqrt(3.0 * dt)  # Price step in log-space

    # Max tree width
    max_j = 2 * num_steps + 1
    offset = num_steps + 1  # Index of center node

    # Node values and option values
    stock_prices = zeros(max_j)
    option_vals = zeros(max_j)
    new_option_vals = zeros(max_j)

    # Initialize terminal values
    for j in 1:max_j
        log_S = log(S) + (j - offset) * dx
        stock_prices[j] = exp(log_S)
        if is_call
            option_vals[j] = max(exp(log_S) - K, 0.0)
        else
            option_vals[j] = max(K - exp(log_S), 0.0)
        end
    end

    # Backward induction
    for step in num_steps:-1:1
        t = (step - 1) * dt
        fill!(new_option_vals, 0.0)

        for j in 1:max_j
            log_S_j = log(S) + (j - offset) * dx
            S_j = exp(log_S_j)
            sigma = local_vol_func(S_j, t)

            # Transition probabilities
            mu = (r - q - 0.5 * sigma^2) * dt
            var = sigma^2 * dt

            # Trinomial probabilities
            p_u = 0.5 * (var / dx^2 + mu / dx)  # Up
            p_d = 0.5 * (var / dx^2 - mu / dx)  # Down
            p_m = 1.0 - var / dx^2               # Middle

            p_u = clamp(p_u, 0.0, 1.0)
            p_d = clamp(p_d, 0.0, 1.0)
            p_m = clamp(p_m, 0.0, 1.0)
            s = p_u + p_m + p_d
            if s > 0
                p_u /= s; p_m /= s; p_d /= s
            end

            j_u = min(j + 1, max_j)
            j_d = max(j - 1, 1)

            disc = exp(-r * dt)
            new_option_vals[j] = disc * (p_u * option_vals[j_u] +
                                          p_m * option_vals[j] +
                                          p_d * option_vals[j_d])
        end

        option_vals = copy(new_option_vals)
    end

    return option_vals[offset]
end

# ============================================================================
# SECTION 5: Variance Swaps
# ============================================================================

"""
    variance_swap_fair_strike(strikes, implied_vols, S, r, q, T)

Fair variance swap strike from vol surface using replication.
K_var = (2/T) * integral (C(K)/K^2 dK for K>F, P(K)/K^2 dK for K<F)
"""
function variance_swap_fair_strike(strikes::Vector{Float64},
                                    implied_vols::Vector{Float64},
                                    S::Float64, r::Float64, q::Float64,
                                    T::Float64)::Float64
    F = S * exp((r - q) * T)
    n = length(strikes)

    integral = 0.0
    for i in 2:n
        K = strikes[i]
        K_prev = strikes[i-1]
        dK = K - K_prev
        K_mid = 0.5 * (K + K_prev)
        sigma = 0.5 * (implied_vols[i] + implied_vols[i-1])

        if K_mid < F
            # OTM put
            price = bs_put_full(S, K_mid, r, q, sigma, T).price
        else
            # OTM call
            price = bs_call_full(S, K_mid, r, q, sigma, T).price
        end

        integral += price * dK / (K_mid^2)
    end

    K_var = (2.0 * exp(r * T) / T) * integral
    return K_var
end

"""
    realized_variance_leg(prices, T)

Realized variance for variance swap settlement.
"""
function realized_variance_leg(prices::Vector{Float64}, T::Float64)::Float64
    n = length(prices) - 1
    if n <= 0
        return 0.0
    end
    dt = T / n
    rv = 0.0
    for i in 2:length(prices)
        rv += (log(prices[i] / prices[i-1]))^2
    end
    return rv / T * (252.0 / n * n)  # Annualized
end

"""
    variance_swap_mtm(fair_strike, current_rv, remaining_T, total_T, notional)

Mark-to-market of variance swap.
"""
function variance_swap_mtm(fair_strike::Float64, current_rv::Float64,
                            remaining_T::Float64, total_T::Float64,
                            notional::Float64)::Float64
    elapsed = total_T - remaining_T
    weighted_rv = (elapsed * current_rv + remaining_T * fair_strike) / total_T
    return notional * (weighted_rv - fair_strike)
end

# ============================================================================
# SECTION 6: Barrier Options
# ============================================================================

"""
    barrier_option_analytical(S, K, H, r, q, sigma, T, barrier_type, rebate)

Analytical barrier option pricing (Reiner-Rubinstein 1991).
barrier_type: :down_and_out_call, :down_and_in_call, :up_and_out_call, :up_and_in_call,
              :down_and_out_put, :down_and_in_put, :up_and_out_put, :up_and_in_put
"""
function barrier_option_analytical(S::Float64, K::Float64, H::Float64,
                                    r::Float64, q::Float64, sigma::Float64,
                                    T::Float64, barrier_type::Symbol,
                                    rebate::Float64=0.0)::Float64
    sqrtT = sqrt(T)
    mu = (r - q - 0.5 * sigma^2) / sigma^2
    lambda = sqrt(mu^2 + 2.0 * r / sigma^2)

    x1 = log(S / K) / (sigma * sqrtT) + (1.0 + mu) * sigma * sqrtT
    x2 = log(S / H) / (sigma * sqrtT) + (1.0 + mu) * sigma * sqrtT
    y1 = log(H^2 / (S * K)) / (sigma * sqrtT) + (1.0 + mu) * sigma * sqrtT
    y2 = log(H / S) / (sigma * sqrtT) + (1.0 + mu) * sigma * sqrtT

    eqT = exp(-q * T)
    erT = exp(-r * T)

    A = S * eqT * normal_cdf(x1) - K * erT * normal_cdf(x1 - sigma * sqrtT)
    B = S * eqT * normal_cdf(x2) - K * erT * normal_cdf(x2 - sigma * sqrtT)
    C = S * eqT * (H / S)^(2.0 * (mu + 1.0)) * normal_cdf(y1) -
        K * erT * (H / S)^(2.0 * mu) * normal_cdf(y1 - sigma * sqrtT)
    D_val = S * eqT * (H / S)^(2.0 * (mu + 1.0)) * normal_cdf(y2) -
        K * erT * (H / S)^(2.0 * mu) * normal_cdf(y2 - sigma * sqrtT)

    # Rebate terms
    E_val = rebate * erT * (normal_cdf(x2 - sigma * sqrtT) -
            (H / S)^(2.0 * mu) * normal_cdf(y2 - sigma * sqrtT))
    F_val = rebate * ((H / S)^(mu + lambda) * normal_cdf(y2 * lambda / (mu + 1.0)) +
            (H / S)^(mu - lambda) * normal_cdf(y2 * lambda / (mu + 1.0) - 2.0 * lambda * sigma * sqrtT))

    if barrier_type == :down_and_out_call
        if K > H
            return A - C + F_val
        else
            return B - D_val + F_val
        end
    elseif barrier_type == :down_and_in_call
        vanilla = bs_call_full(S, K, r, q, sigma, T).price
        return vanilla - barrier_option_analytical(S, K, H, r, q, sigma, T, :down_and_out_call, rebate)
    elseif barrier_type == :up_and_out_call
        if K > H
            return F_val
        else
            return A - B + D_val - C + F_val
        end
    elseif barrier_type == :up_and_in_call
        vanilla = bs_call_full(S, K, r, q, sigma, T).price
        return vanilla - barrier_option_analytical(S, K, H, r, q, sigma, T, :up_and_out_call, rebate)
    elseif barrier_type == :down_and_out_put
        if K > H
            return A - B + D_val - C + F_val  # Adjusted for put
        else
            return F_val
        end
    elseif barrier_type == :down_and_in_put
        vanilla = bs_put_full(S, K, r, q, sigma, T).price
        return vanilla - barrier_option_analytical(S, K, H, r, q, sigma, T, :down_and_out_put, rebate)
    elseif barrier_type == :up_and_out_put
        if K > H
            return F_val
        else
            return B - D_val + F_val  # Put analog
        end
    elseif barrier_type == :up_and_in_put
        vanilla = bs_put_full(S, K, r, q, sigma, T).price
        return vanilla - barrier_option_analytical(S, K, H, r, q, sigma, T, :up_and_out_put, rebate)
    end

    return 0.0
end

"""
    barrier_option_mc(S, K, H, r, q, sigma, T, barrier_type, num_paths, num_steps;
                      seed=42, use_brownian_bridge=true)

Monte Carlo barrier option with Brownian bridge correction.
"""
function barrier_option_mc(S::Float64, K::Float64, H::Float64,
                            r::Float64, q::Float64, sigma::Float64, T::Float64,
                            barrier_type::Symbol, num_paths::Int, num_steps::Int;
                            seed::Int=42, use_brownian_bridge::Bool=true)::Float64
    rng = Random.MersenneTwister(seed)
    dt = T / num_steps
    drift = (r - q - 0.5 * sigma^2) * dt

    is_call = occursin("call", string(barrier_type))
    is_down = occursin("down", string(barrier_type))
    is_out = occursin("out", string(barrier_type))

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        S_curr = S
        S_prev = S
        hit_barrier = false

        for step in 1:num_steps
            z = randn(rng)
            S_curr = S_prev * exp(drift + sigma * sqrt(dt) * z)

            # Check barrier
            if is_down && S_curr <= H
                hit_barrier = true
            elseif !is_down && S_curr >= H
                hit_barrier = true
            end

            # Brownian bridge correction
            if use_brownian_bridge && !hit_barrier
                if is_down && S_curr > H && S_prev > H
                    # Probability of crossing H between S_prev and S_curr
                    p_cross = exp(-2.0 * log(S_prev / H) * log(S_curr / H) / (sigma^2 * dt))
                    if rand(rng) < p_cross
                        hit_barrier = true
                    end
                elseif !is_down && S_curr < H && S_prev < H
                    p_cross = exp(-2.0 * log(H / S_prev) * log(H / S_curr) / (sigma^2 * dt))
                    if rand(rng) < p_cross
                        hit_barrier = true
                    end
                end
            end

            S_prev = S_curr
        end

        if is_call
            intrinsic = max(S_curr - K, 0.0)
        else
            intrinsic = max(K - S_curr, 0.0)
        end

        if is_out
            payoffs[path] = hit_barrier ? 0.0 : intrinsic
        else
            payoffs[path] = hit_barrier ? intrinsic : 0.0
        end
    end

    return exp(-r * T) * mean(payoffs)
end

# ============================================================================
# SECTION 7: Asian Options
# ============================================================================

"""
    asian_geometric_exact(S, K, r, q, sigma, T, num_obs; is_call=true)

Exact pricing of geometric average Asian option.
"""
function asian_geometric_exact(S::Float64, K::Float64, r::Float64, q::Float64,
                                sigma::Float64, T::Float64, num_obs::Int;
                                is_call::Bool=true)::Float64
    n = num_obs
    dt = T / n

    # Adjusted parameters for geometric average
    sigma_a = sigma * sqrt((2.0 * n + 1.0) / (6.0 * (n + 1.0)))
    mu_a = (r - q - 0.5 * sigma^2) * (n + 1.0) / (2.0 * n) + 0.5 * sigma_a^2

    d1 = (log(S / K) + (mu_a + 0.5 * sigma_a^2) * T) / (sigma_a * sqrt(T))
    d2 = d1 - sigma_a * sqrt(T)

    if is_call
        return exp(-r * T) * (S * exp(mu_a * T) * normal_cdf(d1) - K * normal_cdf(d2))
    else
        return exp(-r * T) * (K * normal_cdf(-d2) - S * exp(mu_a * T) * normal_cdf(-d1))
    end
end

"""
    asian_arithmetic_tw(S, K, r, q, sigma, T, num_obs; is_call=true)

Turnbull-Wakeman (1991) approximation for arithmetic average Asian option.
"""
function asian_arithmetic_tw(S::Float64, K::Float64, r::Float64, q::Float64,
                              sigma::Float64, T::Float64, num_obs::Int;
                              is_call::Bool=true)::Float64
    n = num_obs
    dt = T / n

    # First moment of arithmetic average
    M1 = S * (exp((r - q) * T) - 1.0) / ((r - q) * T)
    if abs(r - q) < 1e-10
        M1 = S
    end

    # Second moment
    t_sum = sum(exp((r - q) * (T - i * dt)) for i in 0:(n-1)) / n
    M1 = S * t_sum

    M2 = 0.0
    for i in 0:(n-1)
        for j in 0:(n-1)
            ti = i * dt
            tj = j * dt
            t_min = min(ti, tj)
            M2 += exp(2.0 * (r - q) * T - (r - q) * (ti + tj) + sigma^2 * t_min)
        end
    end
    M2 *= S^2 / n^2

    # Match to lognormal
    sigma_a_sq = log(M2 / M1^2) / T
    sigma_a_sq = max(sigma_a_sq, 1e-10)
    sigma_a = sqrt(sigma_a_sq)

    d1 = (log(M1 / K) + 0.5 * sigma_a_sq * T) / (sigma_a * sqrt(T))
    d2 = d1 - sigma_a * sqrt(T)

    if is_call
        return exp(-r * T) * (M1 * normal_cdf(d1) - K * normal_cdf(d2))
    else
        return exp(-r * T) * (K * normal_cdf(-d2) - M1 * normal_cdf(-d1))
    end
end

"""
    asian_mc(S, K, r, q, sigma, T, num_obs, num_paths; seed=42, average_type=:arithmetic, is_call=true)

Monte Carlo Asian option.
"""
function asian_mc(S::Float64, K::Float64, r::Float64, q::Float64,
                   sigma::Float64, T::Float64, num_obs::Int, num_paths::Int;
                   seed::Int=42, average_type::Symbol=:arithmetic,
                   is_call::Bool=true)::Float64
    rng = Random.MersenneTwister(seed)
    dt = T / num_obs
    drift = (r - q - 0.5 * sigma^2) * dt

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        S_curr = S
        if average_type == :arithmetic
            avg = 0.0
        else
            avg = 0.0  # log-sum for geometric
        end

        for obs in 1:num_obs
            z = randn(rng)
            S_curr *= exp(drift + sigma * sqrt(dt) * z)
            if average_type == :arithmetic
                avg += S_curr
            else
                avg += log(S_curr)
            end
        end

        if average_type == :arithmetic
            A = avg / num_obs
        else
            A = exp(avg / num_obs)
        end

        if is_call
            payoffs[path] = max(A - K, 0.0)
        else
            payoffs[path] = max(K - A, 0.0)
        end
    end

    return exp(-r * T) * mean(payoffs)
end

# ============================================================================
# SECTION 8: Lookback Options
# ============================================================================

"""
    lookback_fixed_call(S, K, r, q, sigma, T)

Fixed-strike lookback call: max(max(S_t) - K, 0).
"""
function lookback_fixed_call(S::Float64, K::Float64, r::Float64, q::Float64,
                              sigma::Float64, T::Float64)::Float64
    sqrtT = sqrt(T)
    a1 = (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrtT)
    a2 = a1 - sigma * sqrtT
    a3 = (log(S / K) + (-r + q + 0.5 * sigma^2) * T) / (sigma * sqrtT)

    eqT = exp(-q * T)
    erT = exp(-r * T)

    price = S * eqT * normal_cdf(a1) - K * erT * normal_cdf(a2) +
            S * erT * sigma^2 / (2.0 * (r - q)) *
            (-eqT / erT * (S / K)^(-2.0 * (r - q) / sigma^2) * normal_cdf(a3) +
             exp((r - q) * T) * normal_cdf(a1))

    return max(price, 0.0)
end

"""
    lookback_floating_call(S, S_min, r, q, sigma, T)

Floating-strike lookback call: S_T - min(S_t).
"""
function lookback_floating_call(S::Float64, S_min::Float64, r::Float64,
                                 q::Float64, sigma::Float64, T::Float64)::Float64
    sqrtT = sqrt(T)
    if abs(r - q) < 1e-10
        # Special case
        d1 = (log(S / S_min) + 0.5 * sigma^2 * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return S * normal_cdf(d1) - S_min * exp(-r * T) * normal_cdf(d2) +
               S * exp(-r * T) * sigma * sqrtT * (normal_pdf(d1) + d1 * normal_cdf(d1))
    end

    a1 = (log(S / S_min) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrtT)
    a2 = a1 - sigma * sqrtT
    a3 = (log(S / S_min) + (-r + q + 0.5 * sigma^2) * T) / (sigma * sqrtT)

    eqT = exp(-q * T)
    erT = exp(-r * T)
    eta = 2.0 * (r - q) / sigma^2

    price = S * eqT * normal_cdf(a1) - S_min * erT * normal_cdf(a2) -
            S * erT * sigma^2 / (2.0 * (r - q)) *
            (eqT / erT * (S / S_min)^(-eta) * normal_cdf(a3) - exp((r - q) * T) * normal_cdf(a1))

    return max(price, 0.0)
end

"""
    lookback_floating_put(S, S_max, r, q, sigma, T)

Floating-strike lookback put: max(S_t) - S_T.
"""
function lookback_floating_put(S::Float64, S_max::Float64, r::Float64,
                                q::Float64, sigma::Float64, T::Float64)::Float64
    sqrtT = sqrt(T)
    b1 = (log(S / S_max) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrtT)
    b2 = b1 - sigma * sqrtT
    b3 = (log(S / S_max) + (-r + q + 0.5 * sigma^2) * T) / (sigma * sqrtT)

    eqT = exp(-q * T)
    erT = exp(-r * T)
    eta = 2.0 * (r - q) / sigma^2

    price = -S * eqT * normal_cdf(-b1) + S_max * erT * normal_cdf(-b2) +
            S * erT * sigma^2 / (2.0 * max(r - q, 1e-10)) *
            (eqT / erT * (S / S_max)^(-eta) * normal_cdf(-b3) - exp((r - q) * T) * normal_cdf(-b1))

    return max(price, 0.0)
end

# ============================================================================
# SECTION 9: American Options
# ============================================================================

"""
    american_put_baw(S, K, r, q, sigma, T)

Barone-Adesi-Whaley (1987) American put approximation.
"""
function american_put_baw(S::Float64, K::Float64, r::Float64, q::Float64,
                           sigma::Float64, T::Float64)::Float64
    european = bs_put_full(S, K, r, q, sigma, T).price

    if T <= 0.0
        return max(K - S, 0.0)
    end

    M = 2.0 * r / sigma^2
    N_val = 2.0 * (r - q) / sigma^2
    k = 1.0 - exp(-r * T)
    q_2 = (-(N_val - 1.0) - sqrt((N_val - 1.0)^2 + 4.0 * M / k)) / 2.0

    # Find critical stock price S* via Newton's method
    S_star = K
    for _ in 1:50
        d1, _ = bs_d1d2(S_star, K, r, q, sigma, T)
        put_val = bs_put_full(S_star, K, r, q, sigma, T).price
        lhs = K - S_star - put_val
        rhs = -S_star * (1.0 - exp(-q * T) * normal_cdf(-d1)) / q_2

        f = lhs - rhs
        # Derivative
        d_lhs = -1.0 + exp(-q * T) * normal_cdf(-d1)  # Approximate
        d_rhs = -(1.0 - exp(-q * T) * normal_cdf(-d1)) / q_2

        deriv = d_lhs - d_rhs
        if abs(deriv) < 1e-15
            break
        end

        S_star -= f / deriv
        S_star = clamp(S_star, 0.01, K * 2.0)
    end

    if S <= S_star
        return max(K - S, european)
    end

    A_2 = -(S_star / q_2) * (1.0 - exp(-q * T) * normal_cdf(-bs_d1d2(S_star, K, r, q, sigma, T)[1]))
    return european + A_2 * (S / S_star)^q_2
end

"""
    american_lsm(S, K, r, q, sigma, T, num_paths, num_steps;
                  seed=42, is_call=false, num_basis=4)

Longstaff-Schwartz (2001) Least Squares Monte Carlo.
"""
function american_lsm(S::Float64, K::Float64, r::Float64, q::Float64,
                       sigma::Float64, T::Float64, num_paths::Int, num_steps::Int;
                       seed::Int=42, is_call::Bool=false, num_basis::Int=4)::Float64
    rng = Random.MersenneTwister(seed)
    dt = T / num_steps
    drift = (r - q - 0.5 * sigma^2) * dt
    disc = exp(-r * dt)

    # Generate paths
    paths = zeros(num_paths, num_steps + 1)
    paths[:, 1] .= S
    for step in 1:num_steps
        for path in 1:num_paths
            z = randn(rng)
            paths[path, step+1] = paths[path, step] * exp(drift + sigma * sqrt(dt) * z)
        end
    end

    # Payoff function
    payoff_func(s) = is_call ? max(s - K, 0.0) : max(K - s, 0.0)

    # Cash flows at maturity
    cashflows = [payoff_func(paths[p, num_steps+1]) for p in 1:num_paths]
    exercise_time = fill(num_steps, num_paths)

    # Backward induction
    for step in (num_steps-1):-1:1
        # Intrinsic values
        intrinsic = [payoff_func(paths[p, step+1]) for p in 1:num_paths]

        # Find in-the-money paths
        itm = findall(intrinsic .> 0.0)
        if length(itm) < num_basis + 1
            continue
        end

        # Discounted future cashflows
        future_cf = [cashflows[p] * disc^(exercise_time[p] - step) for p in itm]

        # Regression: continuation value ~ polynomial of stock price
        X_reg = zeros(length(itm), num_basis)
        for (idx, p) in enumerate(itm)
            s = paths[p, step+1] / S  # Normalize
            for b in 1:num_basis
                X_reg[idx, b] = s^b
            end
        end

        # OLS
        beta = (X_reg' * X_reg + 1e-8 * I) \ (X_reg' * future_cf)
        continuation = X_reg * beta

        # Exercise decision
        for (idx, p) in enumerate(itm)
            if intrinsic[p] > continuation[idx]
                cashflows[p] = intrinsic[p]
                exercise_time[p] = step
            end
        end
    end

    # Discount to present
    pv = mean(cashflows[p] * disc^exercise_time[p] for p in 1:num_paths)
    return max(pv, 0.0)
end

# ============================================================================
# SECTION 10: Spread and Basket Options
# ============================================================================

"""
    kirk_spread_option(S1, S2, K, r, q1, q2, sigma1, sigma2, rho, T; is_call=true)

Kirk's approximation for spread option: max(S1 - S2 - K, 0).
"""
function kirk_spread_option(S1::Float64, S2::Float64, K::Float64,
                             r::Float64, q1::Float64, q2::Float64,
                             sigma1::Float64, sigma2::Float64,
                             rho::Float64, T::Float64; is_call::Bool=true)::Float64
    F1 = S1 * exp((r - q1) * T)
    F2 = S2 * exp((r - q2) * T)

    F2_K = F2 + K * exp(-r * T)
    if F2_K < 1e-10
        return is_call ? max(F1 - F2 - K, 0.0) * exp(-r * T) : 0.0
    end

    sigma_adj = sqrt(sigma1^2 - 2.0 * rho * sigma1 * sigma2 * F2 / F2_K +
                     (sigma2 * F2 / F2_K)^2)

    d1 = (log(F1 / F2_K) + 0.5 * sigma_adj^2 * T) / (sigma_adj * sqrt(T))
    d2 = d1 - sigma_adj * sqrt(T)

    if is_call
        return exp(-r * T) * (F1 * normal_cdf(d1) - F2_K * normal_cdf(d2))
    else
        return exp(-r * T) * (F2_K * normal_cdf(-d2) - F1 * normal_cdf(-d1))
    end
end

"""
    basket_option_levy(S, weights, K, r, q, sigmas, corr_matrix, T; is_call=true)

Levy (1992) approximation for basket option.
Match first two moments of basket to lognormal.
"""
function basket_option_levy(S::Vector{Float64}, weights::Vector{Float64},
                             K::Float64, r::Float64, q::Vector{Float64},
                             sigmas::Vector{Float64}, corr_matrix::Matrix{Float64},
                             T::Float64; is_call::Bool=true)::Float64
    N = length(S)
    # Forward values
    F = [S[i] * exp((r - q[i]) * T) for i in 1:N]

    # First moment of basket
    M1 = sum(weights[i] * F[i] for i in 1:N)

    # Second moment
    M2 = 0.0
    for i in 1:N
        for j in 1:N
            M2 += weights[i] * weights[j] * F[i] * F[j] *
                  exp(corr_matrix[i, j] * sigmas[i] * sigmas[j] * T)
        end
    end

    # Match to lognormal
    sigma_b_sq = log(M2 / M1^2) / T
    sigma_b_sq = max(sigma_b_sq, 1e-10)
    sigma_b = sqrt(sigma_b_sq)

    d1 = (log(M1 / K) + 0.5 * sigma_b_sq * T) / (sigma_b * sqrt(T))
    d2 = d1 - sigma_b * sqrt(T)

    if is_call
        return exp(-r * T) * (M1 * normal_cdf(d1) - K * normal_cdf(d2))
    else
        return exp(-r * T) * (K * normal_cdf(-d2) - M1 * normal_cdf(-d1))
    end
end

"""
    basket_option_mc(S, weights, K, r, q, sigmas, corr_matrix, T, num_paths;
                      seed=42, is_call=true)

Monte Carlo basket option with control variate (geometric basket).
"""
function basket_option_mc(S::Vector{Float64}, weights::Vector{Float64},
                           K::Float64, r::Float64, q::Vector{Float64},
                           sigmas::Vector{Float64}, corr_matrix::Matrix{Float64},
                           T::Float64, num_paths::Int;
                           seed::Int=42, is_call::Bool=true)::Float64
    rng = Random.MersenneTwister(seed)
    N_assets = length(S)

    # Cholesky decomposition
    L = cholesky(Symmetric(corr_matrix + 1e-8 * I)).L

    arith_payoffs = Vector{Float64}(undef, num_paths)
    geom_payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        z = randn(rng, N_assets)
        corr_z = L * z

        S_T = [S[i] * exp((r - q[i] - 0.5 * sigmas[i]^2) * T +
               sigmas[i] * sqrt(T) * corr_z[i]) for i in 1:N_assets]

        # Arithmetic basket
        basket_arith = sum(weights[i] * S_T[i] for i in 1:N_assets)

        # Geometric basket
        log_basket = sum(weights[i] * log(S_T[i]) for i in 1:N_assets)
        basket_geom = exp(log_basket)

        if is_call
            arith_payoffs[path] = max(basket_arith - K, 0.0)
            geom_payoffs[path] = max(basket_geom - K, 0.0)
        else
            arith_payoffs[path] = max(K - basket_arith, 0.0)
            geom_payoffs[path] = max(K - basket_geom, 0.0)
        end
    end

    # Control variate adjustment
    geom_analytical = basket_option_levy(S, weights, K, r, q, sigmas, corr_matrix, T;
                                          is_call=is_call)
    geom_mc = exp(-r * T) * mean(geom_payoffs)

    beta_cv = cov(arith_payoffs, geom_payoffs) / max(var(geom_payoffs), 1e-15)
    adjusted_payoffs = arith_payoffs .- beta_cv * (geom_payoffs .- geom_mc * exp(r * T))

    return exp(-r * T) * mean(adjusted_payoffs)
end

# ============================================================================
# SECTION 11: Cliquet and Forward-Starting Options
# ============================================================================

"""
    forward_starting_option(S, r, q, sigma, T_start, T_end, moneyness; is_call=true)

Forward-starting option: strike set as moneyness * S_{T_start} at T_start.
"""
function forward_starting_option(S::Float64, r::Float64, q::Float64,
                                  sigma::Float64, T_start::Float64,
                                  T_end::Float64, moneyness::Float64;
                                  is_call::Bool=true)::Float64
    T_option = T_end - T_start
    # Equivalent to moneyness * BS with S=1, K=moneyness
    if is_call
        result = bs_call_full(1.0, moneyness, r, q, sigma, T_option)
    else
        result = bs_put_full(1.0, moneyness, r, q, sigma, T_option)
    end

    return S * exp(-q * T_start) * result.price
end

"""
    cliquet_option(S, r, q, sigma, reset_times, cap, floor, global_cap, global_floor)

Cliquet (ratchet) option: sum of capped/floored periodic returns.
"""
function cliquet_option(S::Float64, r::Float64, q::Float64, sigma::Float64,
                        reset_times::Vector{Float64}, cap::Float64, floor::Float64,
                        global_cap::Float64, global_floor::Float64;
                        num_paths::Int=50000, seed::Int=42)::Float64
    rng = Random.MersenneTwister(seed)
    n_periods = length(reset_times)
    T = reset_times[end]

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        S_curr = S
        total_return = 0.0

        for p in 1:n_periods
            dt = p == 1 ? reset_times[1] : reset_times[p] - reset_times[p-1]
            z = randn(rng)
            S_next = S_curr * exp((r - q - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * z)

            period_return = S_next / S_curr - 1.0
            capped_return = min(max(period_return, floor), cap)
            total_return += capped_return

            S_curr = S_next
        end

        total_return = min(max(total_return, global_floor), global_cap)
        payoffs[path] = max(total_return, 0.0) * S  # Notional = S
    end

    return exp(-r * T) * mean(payoffs)
end

# ============================================================================
# SECTION 12: Convertible Bonds
# ============================================================================

"""
    convertible_bond_tree(S, face, coupon_rate, conversion_ratio, r, credit_spread,
                          sigma, T, num_steps; call_price=Inf, put_price=0.0)

Convertible bond pricing via binomial tree with credit risk.
"""
function convertible_bond_tree(S::Float64, face::Float64, coupon_rate::Float64,
                                conversion_ratio::Float64, r::Float64,
                                credit_spread::Float64, sigma::Float64, T::Float64,
                                num_steps::Int; call_price::Float64=Inf,
                                put_price::Float64=0.0)::Float64
    dt = T / num_steps
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    p = (exp((r - 0.0) * dt) - d) / (u - d)
    p = clamp(p, 0.01, 0.99)

    r_risky = r + credit_spread
    disc_risky = exp(-r_risky * dt)
    disc_rf = exp(-r * dt)

    coupon = face * coupon_rate * dt

    # Build stock tree
    stock_tree = zeros(num_steps + 1, num_steps + 1)
    for i in 0:num_steps
        for j in 0:i
            stock_tree[j+1, i+1] = S * u^j * d^(i - j)
        end
    end

    # CB values at maturity
    cb_values = zeros(num_steps + 1)
    for j in 0:num_steps
        stock_val = stock_tree[j+1, num_steps+1]
        conversion_value = conversion_ratio * stock_val
        bond_value = face + coupon
        cb_values[j+1] = max(conversion_value, bond_value)
    end

    # Backward induction
    for i in (num_steps-1):-1:0
        new_values = zeros(i + 1)
        for j in 0:i
            stock_val = stock_tree[j+1, i+1]
            conversion_value = conversion_ratio * stock_val

            # Continuation value (discount at risky rate for bond component)
            hold_value = disc_risky * (p * cb_values[j+2] + (1.0 - p) * cb_values[j+1]) + coupon

            # Conversion value (discount at risk-free)
            # Holder can convert at any time
            node_value = max(hold_value, conversion_value)

            # Issuer call
            if node_value > call_price
                node_value = max(call_price, conversion_value)
            end

            # Holder put
            if node_value < put_price
                node_value = put_price
            end

            new_values[j+1] = node_value
        end
        cb_values = new_values
    end

    return cb_values[1]
end

# ============================================================================
# SECTION 13: Interest Rate Derivatives
# ============================================================================

"""
    black76_call(F, K, r, sigma, T)

Black (1976) formula for options on futures/forwards.
"""
function black76_call(F::Float64, K::Float64, r::Float64, sigma::Float64,
                       T::Float64)::Float64
    if T <= 0.0
        return max(F - K, 0.0) * exp(-r * T)
    end
    d1 = (log(F / K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return exp(-r * T) * (F * normal_cdf(d1) - K * normal_cdf(d2))
end

"""
    black76_put(F, K, r, sigma, T)

Black (1976) put on futures.
"""
function black76_put(F::Float64, K::Float64, r::Float64, sigma::Float64,
                      T::Float64)::Float64
    if T <= 0.0
        return max(K - F, 0.0) * exp(-r * T)
    end
    d1 = (log(F / K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return exp(-r * T) * (K * normal_cdf(-d2) - F * normal_cdf(-d1))
end

"""
    bachelier_call(F, K, r, sigma_n, T)

Bachelier (normal) model for interest rate options.
sigma_n is normal (absolute) volatility.
"""
function bachelier_call(F::Float64, K::Float64, r::Float64, sigma_n::Float64,
                         T::Float64)::Float64
    if T <= 0.0
        return max(F - K, 0.0) * exp(-r * T)
    end
    d = (F - K) / (sigma_n * sqrt(T))
    return exp(-r * T) * ((F - K) * normal_cdf(d) + sigma_n * sqrt(T) * normal_pdf(d))
end

"""
    bachelier_put(F, K, r, sigma_n, T)

Bachelier put.
"""
function bachelier_put(F::Float64, K::Float64, r::Float64, sigma_n::Float64,
                        T::Float64)::Float64
    if T <= 0.0
        return max(K - F, 0.0) * exp(-r * T)
    end
    d = (F - K) / (sigma_n * sqrt(T))
    return exp(-r * T) * ((K - F) * normal_cdf(-d) + sigma_n * sqrt(T) * normal_pdf(d))
end

"""
    swaption_black(swap_rate, strike, annuity, sigma, T_expiry, is_payer)

Black's model for swaption pricing.
"""
function swaption_black(swap_rate::Float64, strike::Float64, annuity::Float64,
                         sigma::Float64, T_expiry::Float64, is_payer::Bool)::Float64
    if T_expiry <= 0.0
        if is_payer
            return max(swap_rate - strike, 0.0) * annuity
        else
            return max(strike - swap_rate, 0.0) * annuity
        end
    end

    d1 = (log(swap_rate / strike) + 0.5 * sigma^2 * T_expiry) / (sigma * sqrt(T_expiry))
    d2 = d1 - sigma * sqrt(T_expiry)

    if is_payer
        return annuity * (swap_rate * normal_cdf(d1) - strike * normal_cdf(d2))
    else
        return annuity * (strike * normal_cdf(-d2) - swap_rate * normal_cdf(-d1))
    end
end

"""
    caplet_black(forward_rate, strike, notional, day_count_frac, df, sigma, T)

Black's model for caplet.
"""
function caplet_black(forward_rate::Float64, strike::Float64, notional::Float64,
                       day_count_frac::Float64, df::Float64, sigma::Float64,
                       T::Float64)::Float64
    if T <= 0.0
        return max(forward_rate - strike, 0.0) * notional * day_count_frac * df
    end

    d1 = (log(forward_rate / strike) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return notional * day_count_frac * df *
           (forward_rate * normal_cdf(d1) - strike * normal_cdf(d2))
end

"""
    floorlet_black(forward_rate, strike, notional, day_count_frac, df, sigma, T)

Black's model for floorlet.
"""
function floorlet_black(forward_rate::Float64, strike::Float64, notional::Float64,
                          day_count_frac::Float64, df::Float64, sigma::Float64,
                          T::Float64)::Float64
    if T <= 0.0
        return max(strike - forward_rate, 0.0) * notional * day_count_frac * df
    end

    d1 = (log(forward_rate / strike) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return notional * day_count_frac * df *
           (strike * normal_cdf(-d2) - forward_rate * normal_cdf(-d1))
end

# ============================================================================
# SECTION 14: Exotic Path-Dependent Options
# ============================================================================

"""
    autocallable_mc(S, barriers, coupons, r, q, sigma, observation_times,
                     final_barrier, num_paths; seed=42)

Auto-callable structured note pricing.
Knocks out (redeems) at observation dates if S > barrier.
"""
function autocallable_mc(S::Float64, barriers::Vector{Float64},
                          coupons::Vector{Float64}, r::Float64, q::Float64,
                          sigma::Float64, observation_times::Vector{Float64},
                          final_barrier::Float64, num_paths::Int;
                          seed::Int=42)::Float64
    rng = Random.MersenneTwister(seed)
    n_obs = length(observation_times)
    T = observation_times[end]

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        S_curr = S
        redeemed = false
        t_prev = 0.0

        for obs in 1:n_obs
            dt = observation_times[obs] - t_prev
            z = randn(rng)
            S_curr = S_curr * exp((r - q - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * z)

            if S_curr >= barriers[obs] * S
                # Autocall triggered
                payoffs[path] = (1.0 + coupons[obs]) * exp(-r * observation_times[obs])
                redeemed = true
                break
            end

            t_prev = observation_times[obs]
        end

        if !redeemed
            # Final payoff at maturity
            if S_curr >= final_barrier * S
                payoffs[path] = 1.0 * exp(-r * T)  # Par redemption
            else
                payoffs[path] = (S_curr / S) * exp(-r * T)  # Loss
            end
        end
    end

    return S * mean(payoffs)
end

"""
    worst_of_option_mc(S_vec, K_frac, r, q_vec, sigmas, corr_matrix, T,
                        num_paths; seed=42, is_call=false)

Worst-of put option on multiple underlyings.
"""
function worst_of_option_mc(S_vec::Vector{Float64}, K_frac::Float64,
                             r::Float64, q_vec::Vector{Float64},
                             sigmas::Vector{Float64}, corr_matrix::Matrix{Float64},
                             T::Float64, num_paths::Int;
                             seed::Int=42, is_call::Bool=false)::Float64
    rng = Random.MersenneTwister(seed)
    N = length(S_vec)
    L = cholesky(Symmetric(corr_matrix + 1e-8 * I)).L

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        z = randn(rng, N)
        corr_z = L * z

        worst_perf = Inf
        for i in 1:N
            S_T = S_vec[i] * exp((r - q_vec[i] - 0.5 * sigmas[i]^2) * T +
                  sigmas[i] * sqrt(T) * corr_z[i])
            perf = S_T / S_vec[i]
            worst_perf = min(worst_perf, perf)
        end

        if is_call
            payoffs[path] = max(worst_perf - K_frac, 0.0)
        else
            payoffs[path] = max(K_frac - worst_perf, 0.0)
        end
    end

    return exp(-r * T) * mean(payoffs) * sum(S_vec) / N
end

"""
    rainbow_best_of_mc(S_vec, K, r, q_vec, sigmas, corr_matrix, T, num_paths;
                        seed=42, is_call=true)

Best-of rainbow option.
"""
function rainbow_best_of_mc(S_vec::Vector{Float64}, K::Float64,
                              r::Float64, q_vec::Vector{Float64},
                              sigmas::Vector{Float64}, corr_matrix::Matrix{Float64},
                              T::Float64, num_paths::Int;
                              seed::Int=42, is_call::Bool=true)::Float64
    rng = Random.MersenneTwister(seed)
    N = length(S_vec)
    L = cholesky(Symmetric(corr_matrix + 1e-8 * I)).L

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        z = randn(rng, N)
        corr_z = L * z

        best_val = -Inf
        for i in 1:N
            S_T = S_vec[i] * exp((r - q_vec[i] - 0.5 * sigmas[i]^2) * T +
                  sigmas[i] * sqrt(T) * corr_z[i])
            best_val = max(best_val, S_T)
        end

        if is_call
            payoffs[path] = max(best_val - K, 0.0)
        else
            payoffs[path] = max(K - best_val, 0.0)
        end
    end

    return exp(-r * T) * mean(payoffs)
end

"""
    rainbow_two_asset_correlation(S1, S2, K1, K2, r, q1, q2, sigma1, sigma2, rho, T)

Two-asset correlation option: max(S1_T - K1, 0) if S2_T > K2.
"""
function rainbow_two_asset_correlation(S1::Float64, S2::Float64, K1::Float64,
                                        K2::Float64, r::Float64, q1::Float64,
                                        q2::Float64, sigma1::Float64,
                                        sigma2::Float64, rho::Float64,
                                        T::Float64)::Float64
    sqrtT = sqrt(T)
    d1 = (log(S2 / K2) + (r - q2 - 0.5 * sigma2^2) * T) / (sigma2 * sqrtT)
    d2 = (log(S1 / K1) + (r - q1 - 0.5 * sigma1^2) * T) / (sigma1 * sqrtT)

    # Bivariate normal CDF approximation
    M_d2_d1 = _bivariate_normal_cdf(d2, d1, rho)
    M_d2s_d1 = _bivariate_normal_cdf(d2 + sigma1 * sqrtT, d1 + rho * sigma1 * sqrtT, rho)

    price = S1 * exp(-q1 * T) * M_d2s_d1 - K1 * exp(-r * T) * M_d2_d1
    return max(price, 0.0)
end

"""
    _bivariate_normal_cdf(a, b, rho)

Drezner-Wesolowsky (1990) approximation for bivariate normal CDF.
"""
function _bivariate_normal_cdf(a::Float64, b::Float64, rho::Float64)::Float64
    if abs(rho) < 1e-10
        return normal_cdf(a) * normal_cdf(b)
    end
    if abs(rho - 1.0) < 1e-10
        return normal_cdf(min(a, b))
    end
    if abs(rho + 1.0) < 1e-10
        return max(normal_cdf(a) + normal_cdf(b) - 1.0, 0.0)
    end

    # Gauss-Legendre quadrature
    x_gl = [0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992]
    w_gl = [0.11846345, 0.23931434, 0.28444444, 0.23931434, 0.11846345]

    result = 0.0
    for i in 1:5
        r_i = rho * x_gl[i]
        sq = sqrt(1.0 - r_i^2)
        result += w_gl[i] * exp(-(a^2 - 2.0 * r_i * a * b + b^2) / (2.0 * (1.0 - r_i^2))) / sq
    end

    result *= rho / (2.0 * pi)
    result += normal_cdf(a) * normal_cdf(b)

    return clamp(result, 0.0, 1.0)
end

"""
    quanto_option(S, K, r_d, r_f, sigma_s, sigma_fx, rho, T, fx_rate; is_call=true)

Quanto option: foreign asset, domestic payout.
"""
function quanto_option(S::Float64, K::Float64, r_d::Float64, r_f::Float64,
                        sigma_s::Float64, sigma_fx::Float64, rho::Float64,
                        T::Float64, fx_rate::Float64; is_call::Bool=true)::Float64
    # Quanto adjustment: drift becomes r_d - r_f - rho*sigma_s*sigma_fx
    q_adj = r_f + rho * sigma_s * sigma_fx

    d1 = (log(S / K) + (r_d - q_adj + 0.5 * sigma_s^2) * T) / (sigma_s * sqrt(T))
    d2 = d1 - sigma_s * sqrt(T)

    if is_call
        return fx_rate * exp(-r_d * T) * (S * exp((r_d - q_adj) * T) * normal_cdf(d1) -
               K * normal_cdf(d2))
    else
        return fx_rate * exp(-r_d * T) * (K * normal_cdf(-d2) -
               S * exp((r_d - q_adj) * T) * normal_cdf(-d1))
    end
end

"""
    chooser_option(S, K, r, q, sigma, T_choose, T_call, T_put)

Chooser option: holder chooses call or put at T_choose.
"""
function chooser_option(S::Float64, K::Float64, r::Float64, q::Float64,
                         sigma::Float64, T_choose::Float64,
                         T_call::Float64, T_put::Float64)::Float64
    # At T_choose: value = max(Call(T_call - T_choose), Put(T_put - T_choose))
    # Rubinstein (1991) formula
    d = (log(S / K) + (r - q + 0.5 * sigma^2) * T_call) / (sigma * sqrt(T_call))
    y = (log(S / K) + (r - q) * T_call + 0.5 * sigma^2 * T_choose) / (sigma * sqrt(T_choose))

    call_part = bs_call_full(S, K, r, q, sigma, T_call).price
    put_adjustment = -S * exp(-q * T_put) * normal_cdf(-y) +
                     K * exp(-r * T_put) * normal_cdf(-y + sigma * sqrt(T_choose))

    return call_part + put_adjustment
end

"""
    compound_option(S, K1, K2, r, q, sigma, T1, T2, outer_call, inner_call)

Compound option: option on option.
K1 = strike of outer, K2 = strike of inner.
T1 = expiry of outer, T2 = expiry of inner.
"""
function compound_option(S::Float64, K1::Float64, K2::Float64,
                          r::Float64, q::Float64, sigma::Float64,
                          T1::Float64, T2::Float64,
                          outer_call::Bool, inner_call::Bool)::Float64
    # Find critical S* where inner option value = K1
    S_star = K2  # Initial guess
    for _ in 1:50
        if inner_call
            inner_val = bs_call_full(S_star, K2, r, q, sigma, T2 - T1).price
        else
            inner_val = bs_put_full(S_star, K2, r, q, sigma, T2 - T1).price
        end
        vega = S_star * exp(-q * (T2 - T1)) * normal_pdf(
            bs_d1d2(S_star, K2, r, q, sigma, T2 - T1)[1]) * sqrt(T2 - T1)

        diff = inner_val - K1
        if abs(vega) < 1e-15
            break
        end
        S_star -= diff / vega
        S_star = max(S_star, 0.01)

        if abs(diff) < 1e-8
            break
        end
    end

    sqrtT1 = sqrt(T1)
    sqrtT2 = sqrt(T2)

    a1 = (log(S / S_star) + (r - q + 0.5 * sigma^2) * T1) / (sigma * sqrtT1)
    a2 = a1 - sigma * sqrtT1

    b1 = (log(S / K2) + (r - q + 0.5 * sigma^2) * T2) / (sigma * sqrtT2)
    b2 = b1 - sigma * sqrtT2

    rho_12 = sqrt(T1 / T2)

    if outer_call && inner_call
        # Call on call
        M1 = _bivariate_normal_cdf(a1, b1, rho_12)
        M2 = _bivariate_normal_cdf(a2, b2, rho_12)
        return S * exp(-q * T2) * M1 - K2 * exp(-r * T2) * M2 -
               K1 * exp(-r * T1) * normal_cdf(a2)
    elseif outer_call && !inner_call
        # Call on put
        M1 = _bivariate_normal_cdf(a1, -b1, -rho_12)
        M2 = _bivariate_normal_cdf(a2, -b2, -rho_12)
        return K2 * exp(-r * T2) * M2 - S * exp(-q * T2) * M1 -
               K1 * exp(-r * T1) * normal_cdf(a2)
    elseif !outer_call && inner_call
        # Put on call
        M1 = _bivariate_normal_cdf(-a1, b1, -rho_12)
        M2 = _bivariate_normal_cdf(-a2, b2, -rho_12)
        return K2 * exp(-r * T2) * M2 - S * exp(-q * T2) * M1 +
               K1 * exp(-r * T1) * normal_cdf(-a2)
    else
        # Put on put
        M1 = _bivariate_normal_cdf(-a1, -b1, rho_12)
        M2 = _bivariate_normal_cdf(-a2, -b2, rho_12)
        return S * exp(-q * T2) * M1 - K2 * exp(-r * T2) * M2 +
               K1 * exp(-r * T1) * normal_cdf(-a2)
    end
end

"""
    exchange_option(S1, S2, q1, q2, sigma1, sigma2, rho, T)

Margrabe (1978) exchange option: max(S1 - S2, 0) at T.
"""
function exchange_option(S1::Float64, S2::Float64, q1::Float64, q2::Float64,
                          sigma1::Float64, sigma2::Float64, rho::Float64,
                          T::Float64)::Float64
    sigma = sqrt(sigma1^2 + sigma2^2 - 2.0 * rho * sigma1 * sigma2)
    d1 = (log(S1 / S2) + (q2 - q1 + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S1 * exp(-q1 * T) * normal_cdf(d1) - S2 * exp(-q2 * T) * normal_cdf(d2)
end

"""
    gap_option(S, K_trigger, K_strike, r, q, sigma, T; is_call=true)

Gap option: pays (S - K_strike) if S > K_trigger (call).
"""
function gap_option(S::Float64, K_trigger::Float64, K_strike::Float64,
                     r::Float64, q::Float64, sigma::Float64, T::Float64;
                     is_call::Bool=true)::Float64
    d1 = (log(S / K_trigger) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if is_call
        return S * exp(-q * T) * normal_cdf(d1) - K_strike * exp(-r * T) * normal_cdf(d2)
    else
        return K_strike * exp(-r * T) * normal_cdf(-d2) - S * exp(-q * T) * normal_cdf(-d1)
    end
end

"""
    supershare_option(S, K_lower, K_upper, r, q, sigma, T)

Supershare option: pays S/K_lower if K_lower < S_T < K_upper, else 0.
"""
function supershare_option(S::Float64, K_lower::Float64, K_upper::Float64,
                            r::Float64, q::Float64, sigma::Float64, T::Float64)::Float64
    d1_lower = (log(S / K_lower) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d1_upper = (log(S / K_upper) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))

    return (S / K_lower) * exp(-q * T) * (normal_cdf(d1_lower) - normal_cdf(d1_upper))
end

"""
    vol_surface_smile(S, r, q, T, strikes, base_vol, skew, convexity)

Generate implied vol smile: sigma(K) = base + skew*(K/S-1) + convexity*(K/S-1)^2
"""
function vol_surface_smile(S::Float64, r::Float64, q::Float64, T::Float64,
                            strikes::Vector{Float64}, base_vol::Float64,
                            skew::Float64, convexity::Float64)::Vector{Float64}
    return [base_vol + skew * (K / S - 1.0) + convexity * (K / S - 1.0)^2
            for K in strikes]
end

"""
    vol_surface_svi(k, a, b, rho, m, sigma_svi)

SVI (Stochastic Volatility Inspired) parameterization.
w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
where k = log(K/F) is log-moneyness, w = sigma^2 * T.
"""
function vol_surface_svi(k::Float64, a::Float64, b::Float64, rho_svi::Float64,
                          m::Float64, sigma_svi::Float64)::Float64
    w = a + b * (rho_svi * (k - m) + sqrt((k - m)^2 + sigma_svi^2))
    return max(w, 1e-6)
end

# ============================================================================
# SECTION 15: Additional Exotic and Utility Functions
# ============================================================================

"""
    double_barrier_option_mc(S, K, H_upper, H_lower, r, q, sigma, T,
                              num_paths, num_steps; seed=42, is_call=true)

Double barrier knock-out option via Monte Carlo.
"""
function double_barrier_option_mc(S::Float64, K::Float64, H_upper::Float64,
                                   H_lower::Float64, r::Float64, q::Float64,
                                   sigma::Float64, T::Float64,
                                   num_paths::Int, num_steps::Int;
                                   seed::Int=42, is_call::Bool=true)::Float64
    rng = Random.MersenneTwister(seed)
    dt = T / num_steps
    drift = (r - q - 0.5 * sigma^2) * dt

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        S_curr = S
        knocked_out = false

        for step in 1:num_steps
            z = randn(rng)
            S_curr *= exp(drift + sigma * sqrt(dt) * z)

            if S_curr >= H_upper || S_curr <= H_lower
                knocked_out = true
                break
            end
        end

        if knocked_out
            payoffs[path] = 0.0
        else
            if is_call
                payoffs[path] = max(S_curr - K, 0.0)
            else
                payoffs[path] = max(K - S_curr, 0.0)
            end
        end
    end

    return exp(-r * T) * mean(payoffs)
end

"""
    parisian_barrier_mc(S, K, H, r, q, sigma, T, window, num_paths, num_steps;
                         seed=42, is_down=true, is_out=true, is_call=true)

Parisian barrier option: barrier must be breached for a continuous period.
"""
function parisian_barrier_mc(S::Float64, K::Float64, H::Float64,
                              r::Float64, q::Float64, sigma::Float64, T::Float64,
                              window::Float64, num_paths::Int, num_steps::Int;
                              seed::Int=42, is_down::Bool=true,
                              is_out::Bool=true, is_call::Bool=true)::Float64
    rng = Random.MersenneTwister(seed)
    dt = T / num_steps
    drift = (r - q - 0.5 * sigma^2) * dt
    window_steps = max(1, round(Int, window / dt))

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        S_curr = S
        consecutive_breach = 0
        parisian_triggered = false

        for step in 1:num_steps
            z = randn(rng)
            S_curr *= exp(drift + sigma * sqrt(dt) * z)

            breaching = is_down ? (S_curr <= H) : (S_curr >= H)

            if breaching
                consecutive_breach += 1
                if consecutive_breach >= window_steps
                    parisian_triggered = true
                    break
                end
            else
                consecutive_breach = 0
            end
        end

        intrinsic = is_call ? max(S_curr - K, 0.0) : max(K - S_curr, 0.0)

        if is_out
            payoffs[path] = parisian_triggered ? 0.0 : intrinsic
        else
            payoffs[path] = parisian_triggered ? intrinsic : 0.0
        end
    end

    return exp(-r * T) * mean(payoffs)
end

"""
    timer_option_mc(S, K, r, q, sigma, variance_budget, max_T, num_paths, num_steps;
                     seed=42, is_call=true)

Timer option: expires when realized variance reaches budget.
"""
function timer_option_mc(S::Float64, K::Float64, r::Float64, q::Float64,
                          sigma::Float64, variance_budget::Float64,
                          max_T::Float64, num_paths::Int, num_steps::Int;
                          seed::Int=42, is_call::Bool=true)::Float64
    rng = Random.MersenneTwister(seed)
    dt = max_T / num_steps
    drift = (r - q - 0.5 * sigma^2) * dt

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        S_curr = S
        cum_var = 0.0
        exercise_time = max_T

        for step in 1:num_steps
            z = randn(rng)
            log_return = drift + sigma * sqrt(dt) * z
            S_curr *= exp(log_return)
            cum_var += log_return^2

            if cum_var >= variance_budget
                exercise_time = step * dt
                break
            end
        end

        df = exp(-r * exercise_time)
        if is_call
            payoffs[path] = df * max(S_curr - K, 0.0)
        else
            payoffs[path] = df * max(K - S_curr, 0.0)
        end
    end

    return mean(payoffs)
end

"""
    corridor_variance_swap(prices, upper_barrier, lower_barrier, T)

Corridor variance swap: realized variance only when price is in corridor.
"""
function corridor_variance_swap(prices::Vector{Float64}, upper_barrier::Float64,
                                 lower_barrier::Float64, T::Float64)::Float64
    n = length(prices) - 1
    if n <= 0
        return 0.0
    end

    corridor_var = 0.0
    in_corridor_count = 0

    for i in 2:length(prices)
        if prices[i-1] >= lower_barrier && prices[i-1] <= upper_barrier
            r = log(prices[i] / prices[i-1])
            corridor_var += r^2
            in_corridor_count += 1
        end
    end

    if in_corridor_count == 0
        return 0.0
    end

    return corridor_var / T * 252.0
end

"""
    gamma_swap_fair_strike(strikes, implied_vols, S, r, q, T)

Gamma swap fair strike (variance swap weighted by price level).
"""
function gamma_swap_fair_strike(strikes::Vector{Float64}, implied_vols::Vector{Float64},
                                 S::Float64, r::Float64, q::Float64, T::Float64)::Float64
    F = S * exp((r - q) * T)
    n = length(strikes)

    integral = 0.0
    for i in 2:n
        K = strikes[i]
        K_prev = strikes[i-1]
        dK = K - K_prev
        K_mid = 0.5 * (K + K_prev)
        sigma = 0.5 * (implied_vols[i] + implied_vols[i-1])

        if K_mid < F
            price = bs_put_full(S, K_mid, r, q, sigma, T).price
        else
            price = bs_call_full(S, K_mid, r, q, sigma, T).price
        end

        # Weight by 1/K instead of 1/K^2 for gamma swap
        integral += price * dK / K_mid
    end

    return (2.0 * exp(r * T) / (T * F)) * integral
end

"""
    volatility_swap_fair_strike(var_swap_strike, T, kurtosis_adjustment)

Volatility swap fair strike from variance swap strike.
E[vol] ~ sqrt(K_var) * (1 - kurtosis_adj / (8 * K_var * T))
"""
function volatility_swap_fair_strike(var_swap_strike::Float64, T::Float64,
                                      kurtosis_adjustment::Float64)::Float64
    if var_swap_strike <= 0
        return 0.0
    end
    return sqrt(var_swap_strike) * (1.0 - kurtosis_adjustment / (8.0 * var_swap_strike * T))
end

"""
    discrete_barrier_adjustment(continuous_price, S, H, sigma, num_obs, T)

Broadie-Glasserman-Kou (1997) continuity correction for discrete barriers.
"""
function discrete_barrier_adjustment(H::Float64, sigma::Float64,
                                      num_obs::Int, T::Float64,
                                      is_down::Bool)::Float64
    dt = T / num_obs
    beta = 0.5826  # Zeta(1/2) / sqrt(2*pi)
    adjustment = exp(beta * sigma * sqrt(dt))

    if is_down
        return H / adjustment
    else
        return H * adjustment
    end
end

"""
    local_vol_from_dupire_mc(S, K, r, q, T, local_vol_grid, K_grid, T_grid,
                              num_paths, num_steps; seed=42, is_call=true)

Monte Carlo pricing with interpolated local vol surface.
"""
function local_vol_from_dupire_mc(S::Float64, K::Float64, r::Float64, q::Float64,
                                   T::Float64, local_vol_grid::Matrix{Float64},
                                   K_grid::Vector{Float64}, T_grid::Vector{Float64},
                                   num_paths::Int, num_steps::Int;
                                   seed::Int=42, is_call::Bool=true)::Float64
    rng = Random.MersenneTwister(seed)
    dt = T / num_steps
    nK = length(K_grid)
    nT = length(T_grid)

    function interp_local_vol(s_val::Float64, t_val::Float64)::Float64
        # Bilinear interpolation
        ki = 1
        for k in 1:(nK-1)
            if s_val >= K_grid[k] && s_val <= K_grid[k+1]
                ki = k
                break
            end
        end
        ki = clamp(ki, 1, nK - 1)

        ti = 1
        for t in 1:(nT-1)
            if t_val >= T_grid[t] && t_val <= T_grid[t+1]
                ti = t
                break
            end
        end
        ti = clamp(ti, 1, nT - 1)

        wk = (s_val - K_grid[ki]) / max(K_grid[ki+1] - K_grid[ki], 1e-10)
        wk = clamp(wk, 0.0, 1.0)
        wt = (t_val - T_grid[ti]) / max(T_grid[ti+1] - T_grid[ti], 1e-10)
        wt = clamp(wt, 0.0, 1.0)

        v00 = local_vol_grid[ki, ti]
        v10 = local_vol_grid[ki+1, ti]
        v01 = local_vol_grid[ki, ti+1]
        v11 = local_vol_grid[ki+1, ti+1]

        return (1-wk)*(1-wt)*v00 + wk*(1-wt)*v10 + (1-wk)*wt*v01 + wk*wt*v11
    end

    payoffs = Vector{Float64}(undef, num_paths)

    for path in 1:num_paths
        s = S
        for step in 1:num_steps
            t = (step - 1) * dt
            sigma_local = interp_local_vol(s, t)
            sigma_local = max(sigma_local, 0.001)

            z = randn(rng)
            s *= exp((r - q - 0.5 * sigma_local^2) * dt + sigma_local * sqrt(dt) * z)
            s = max(s, 1e-10)
        end

        if is_call
            payoffs[path] = max(s - K, 0.0)
        else
            payoffs[path] = max(K - s, 0.0)
        end
    end

    return exp(-r * T) * mean(payoffs)
end

"""
    straddle_price(S, K, r, q, sigma, T)

Straddle = Call + Put at same strike.
"""
function straddle_price(S::Float64, K::Float64, r::Float64, q::Float64,
                         sigma::Float64, T::Float64)::Float64
    return bs_call_full(S, K, r, q, sigma, T).price + bs_put_full(S, K, r, q, sigma, T).price
end

"""
    strangle_price(S, K_call, K_put, r, q, sigma, T)

Strangle = OTM Call + OTM Put.
"""
function strangle_price(S::Float64, K_call::Float64, K_put::Float64,
                         r::Float64, q::Float64, sigma::Float64, T::Float64)::Float64
    return bs_call_full(S, K_call, r, q, sigma, T).price +
           bs_put_full(S, K_put, r, q, sigma, T).price
end

"""
    butterfly_price(S, K_low, K_mid, K_high, r, q, sigma, T)

Butterfly spread: long K_low call + long K_high call + 2 short K_mid call.
"""
function butterfly_price(S::Float64, K_low::Float64, K_mid::Float64,
                          K_high::Float64, r::Float64, q::Float64,
                          sigma::Float64, T::Float64)::Float64
    return bs_call_full(S, K_low, r, q, sigma, T).price +
           bs_call_full(S, K_high, r, q, sigma, T).price -
           2.0 * bs_call_full(S, K_mid, r, q, sigma, T).price
end

"""
    risk_reversal_price(S, K_call, K_put, r, q, sigma_call, sigma_put, T)

Risk reversal: long OTM call + short OTM put.
"""
function risk_reversal_price(S::Float64, K_call::Float64, K_put::Float64,
                              r::Float64, q::Float64, sigma_call::Float64,
                              sigma_put::Float64, T::Float64)::Float64
    return bs_call_full(S, K_call, r, q, sigma_call, T).price -
           bs_put_full(S, K_put, r, q, sigma_put, T).price
end

"""
    calendar_spread_price(S, K, r, q, sigma_near, sigma_far, T_near, T_far)

Calendar spread: long far-dated, short near-dated.
"""
function calendar_spread_price(S::Float64, K::Float64, r::Float64, q::Float64,
                                sigma_near::Float64, sigma_far::Float64,
                                T_near::Float64, T_far::Float64)::Float64
    return bs_call_full(S, K, r, q, sigma_far, T_far).price -
           bs_call_full(S, K, r, q, sigma_near, T_near).price
end

"""
    iron_condor_price(S, K1, K2, K3, K4, r, q, sigma, T)

Iron condor: bear call spread (K3, K4) + bull put spread (K1, K2).
K1 < K2 < K3 < K4.
"""
function iron_condor_price(S::Float64, K1::Float64, K2::Float64,
                            K3::Float64, K4::Float64, r::Float64, q::Float64,
                            sigma::Float64, T::Float64)::Float64
    # Bull put spread
    bull_put = bs_put_full(S, K2, r, q, sigma, T).price -
               bs_put_full(S, K1, r, q, sigma, T).price
    # Bear call spread
    bear_call = bs_call_full(S, K3, r, q, sigma, T).price -
                bs_call_full(S, K4, r, q, sigma, T).price

    return bull_put + bear_call
end

"""
    vega_notional_to_variance_notional(vega_notional, fair_strike)

Convert vega notional to variance notional.
Var_notional = Vega_notional / (2 * sqrt(K_var))
"""
function vega_notional_to_variance_notional(vega_notional::Float64,
                                             fair_strike::Float64)::Float64
    if fair_strike <= 0
        return 0.0
    end
    return vega_notional / (2.0 * sqrt(fair_strike))
end

"""
    delta_hedging_pnl(S_path, K, r, sigma, T, num_rebalances)

Simulate delta hedging P&L for a short call position.
"""
function delta_hedging_pnl(S_path::Vector{Float64}, K::Float64, r::Float64,
                            sigma::Float64, T::Float64, num_rebalances::Int)
    n = length(S_path) - 1
    dt = T / n
    rebalance_freq = max(1, n ÷ num_rebalances)

    # Initial position
    S0 = S_path[1]
    call_price = bs_call_full(S0, K, r, 0.0, sigma, T).price
    delta = bs_call_full(S0, K, r, 0.0, sigma, T).delta

    cash = call_price - delta * S0
    shares = delta
    hedge_pnl_series = Vector{Float64}(undef, n)

    for i in 1:n
        t = i * dt
        remaining = T - t
        S_curr = S_path[i + 1]

        # Portfolio value before rebalance
        portfolio_value = shares * S_curr + cash * exp(r * dt)

        if remaining > 0.001 && i % rebalance_freq == 0
            new_delta = bs_call_full(S_curr, K, r, 0.0, sigma, remaining).delta
            trade = new_delta - shares
            cash = cash * exp(r * dt) - trade * S_curr
            shares = new_delta
        else
            cash *= exp(r * dt)
        end

        # Current option value
        if remaining > 0.001
            option_value = bs_call_full(S_curr, K, r, 0.0, sigma, remaining).price
        else
            option_value = max(S_curr - K, 0.0)
        end

        hedge_pnl_series[i] = shares * S_curr + cash - option_value
    end

    # Final P&L
    S_T = S_path[end]
    final_payoff = max(S_T - K, 0.0)
    final_pnl = shares * S_T + cash - final_payoff

    return (final_pnl=final_pnl, pnl_series=hedge_pnl_series,
            hedge_error=final_pnl - call_price)
end

"""
    theta_decay_profile(S, K, r, q, sigma, T; num_points=100)

Theta decay profile from now to expiry.
"""
function theta_decay_profile(S::Float64, K::Float64, r::Float64, q::Float64,
                              sigma::Float64, T::Float64; num_points::Int=100)
    times = range(T, 0.001, length=num_points)
    prices = [bs_call_full(S, K, r, q, sigma, t).price for t in times]
    thetas = [bs_call_full(S, K, r, q, sigma, t).theta for t in times]
    gammas = [bs_call_full(S, K, r, q, sigma, t).gamma for t in times]

    return (times=collect(times), prices=prices, thetas=thetas, gammas=gammas)
end

"""
    pin_risk_analysis(S, K, r, q, sigma, T_remaining; price_range_pct=0.05)

Gamma/pin risk analysis near expiry.
"""
function pin_risk_analysis(S::Float64, K::Float64, r::Float64, q::Float64,
                            sigma::Float64, T_remaining::Float64;
                            price_range_pct::Float64=0.05)
    prices = range(S * (1.0 - price_range_pct), S * (1.0 + price_range_pct), length=50)

    deltas = [bs_call_full(p, K, r, q, sigma, T_remaining).delta for p in prices]
    gammas = [bs_call_full(p, K, r, q, sigma, T_remaining).gamma for p in prices]
    dollar_gamma = [g * p^2 * 0.01 for (g, p) in zip(gammas, prices)]

    return (prices=collect(prices), deltas=deltas, gammas=gammas,
            dollar_gamma=dollar_gamma,
            max_gamma=maximum(gammas),
            max_dollar_gamma=maximum(dollar_gamma))
end

"""
    vol_surface_arbitrage_check(strikes, maturities, implied_vols, S, r, q)

Check vol surface for calendar and butterfly arbitrage.
"""
function vol_surface_arbitrage_check(strikes::Vector{Float64},
                                      maturities::Vector{Float64},
                                      implied_vols::Matrix{Float64},
                                      S::Float64, r::Float64, q::Float64)
    nK = length(strikes)
    nT = length(maturities)

    calendar_violations = 0
    butterfly_violations = 0

    # Calendar arbitrage: total variance must be non-decreasing in T
    for i in 1:nK
        for j in 2:nT
            tv_prev = implied_vols[i, j-1]^2 * maturities[j-1]
            tv_curr = implied_vols[i, j]^2 * maturities[j]
            if tv_curr < tv_prev - 1e-10
                calendar_violations += 1
            end
        end
    end

    # Butterfly arbitrage: d2C/dK2 >= 0 (call prices convex in K)
    for j in 1:nT
        for i in 2:(nK-1)
            c_low = bs_call_full(S, strikes[i-1], r, q, implied_vols[i-1, j], maturities[j]).price
            c_mid = bs_call_full(S, strikes[i], r, q, implied_vols[i, j], maturities[j]).price
            c_high = bs_call_full(S, strikes[i+1], r, q, implied_vols[i+1, j], maturities[j]).price

            butterfly = c_low - 2.0 * c_mid + c_high
            if butterfly < -1e-8
                butterfly_violations += 1
            end
        end
    end

    return (calendar_violations=calendar_violations,
            butterfly_violations=butterfly_violations,
            is_arbitrage_free=calendar_violations == 0 && butterfly_violations == 0)
end

end # module DerivativesPricingAdvanced
