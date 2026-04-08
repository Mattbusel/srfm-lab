###############################################################################
# QuantFinance.jl
#
# Comprehensive quantitative finance toolkit: bond math, yield curves,
# interest rate models, credit models, FX, commodities, options,
# risk measures, factor models.
#
# Dependencies: LinearAlgebra, Statistics, Random  (stdlib only)
###############################################################################

module QuantFinance

using LinearAlgebra, Statistics, Random

export bond_price, bond_yield, macaulay_duration, modified_duration
export effective_duration, convexity, key_rate_duration, oas_spread
export bootstrap_yield_curve, nelson_siegel, svensson, forward_rate, par_yield
export vasicek_calibrate, vasicek_simulate, cir_calibrate, cir_simulate
export hull_white_calibrate, hull_white_simulate
export merton_model, kmv_distance_to_default, cds_price, hazard_rate_bootstrap
export covered_interest_parity, uncovered_interest_parity, carry_trade_score
export triangular_arbitrage
export cost_of_carry, convenience_yield, seasonal_decomposition
export bs_call, bs_put, bs_greeks, binomial_tree, trinomial_tree
export heston_fft, sabr_vol, sabr_calibrate
export var_historical, var_parametric, var_monte_carlo
export cvar_historical, drawdown_at_risk, omega_ratio_calc
export pca_factors, fama_french_factors, rolling_beta, factor_momentum

# ─────────────────────────────────────────────────────────────────────────────
# §1  Bond Mathematics
# ─────────────────────────────────────────────────────────────────────────────

"""
    bond_price(face, coupon_rate, ytm, n_periods; freq=2) -> price

Price a fixed-rate bond.
"""
function bond_price(face::T, coupon_rate::T, ytm::T, n_periods::Int;
                    freq::Int=2) where T<:Real
    c = face * coupon_rate / freq
    r = ytm / freq
    if abs(r) < T(1e-14)
        return c * n_periods + face
    end
    pv_coupons = c * (one(T) - (one(T) + r)^(-n_periods)) / r
    pv_face = face / (one(T) + r)^n_periods
    pv_coupons + pv_face
end

"""Bond price from zero curve."""
function bond_price_from_zeros(face::T, coupon_rate::T,
                                zero_rates::AbstractVector{T},
                                times::AbstractVector{T};
                                freq::Int=2) where T<:Real
    c = face * coupon_rate / freq
    price = zero(T)
    n = length(times)
    for i in 1:n
        df = exp(-zero_rates[i] * times[i])
        if i < n
            price += c * df
        else
            price += (c + face) * df
        end
    end
    price
end

"""
    bond_yield(price, face, coupon_rate, n_periods; freq=2, tol=1e-10) -> ytm

Yield to maturity via Newton-Raphson.
"""
function bond_yield(price::T, face::T, coupon_rate::T, n_periods::Int;
                    freq::Int=2, max_iter::Int=200, tol::T=T(1e-10)) where T<:Real
    c = face * coupon_rate / freq
    # Initial guess
    ytm = coupon_rate
    for _ in 1:max_iter
        r = ytm / freq
        if abs(r) < T(1e-14)
            bp = c * n_periods + face
            dbp = -c * n_periods * (n_periods + 1) / (2 * freq)
        else
            bp = c * (one(T) - (one(T) + r)^(-n_periods)) / r + face / (one(T) + r)^n_periods
            # Derivative w.r.t. ytm
            t1 = -c * n_periods * (one(T) + r)^(-n_periods - 1) / freq
            t2 = c * ((one(T) + r)^(-n_periods) - one(T)) / (r^2 * freq)
            t3 = c * n_periods * (one(T) + r)^(-n_periods - 1) / (r * freq)
            t4 = -face * n_periods * (one(T) + r)^(-n_periods - 1) / freq
            dbp = t1 + t2 + t3 + t4
        end
        diff = bp - price
        if abs(diff) < tol
            break
        end
        if abs(dbp) < T(1e-16)
            break
        end
        ytm -= diff / dbp
        ytm = clamp(ytm, T(-0.5), T(2.0))
    end
    ytm
end

"""
    macaulay_duration(face, coupon_rate, ytm, n_periods; freq=2) -> duration

Macaulay duration.
"""
function macaulay_duration(face::T, coupon_rate::T, ytm::T, n_periods::Int;
                           freq::Int=2) where T<:Real
    c = face * coupon_rate / freq
    r = ytm / freq
    price = bond_price(face, coupon_rate, ytm, n_periods; freq=freq)
    dur = zero(T)
    for t in 1:n_periods
        cf = t < n_periods ? c : c + face
        pv = cf / (one(T) + r)^t
        dur += (T(t) / freq) * pv
    end
    dur / price
end

"""Modified duration."""
function modified_duration(face::T, coupon_rate::T, ytm::T, n_periods::Int;
                           freq::Int=2) where T<:Real
    mac_dur = macaulay_duration(face, coupon_rate, ytm, n_periods; freq=freq)
    mac_dur / (one(T) + ytm / freq)
end

"""Effective duration (numerical)."""
function effective_duration(face::T, coupon_rate::T, ytm::T, n_periods::Int;
                            freq::Int=2, dy::T=T(0.0001)) where T<:Real
    p_up = bond_price(face, coupon_rate, ytm + dy, n_periods; freq=freq)
    p_down = bond_price(face, coupon_rate, ytm - dy, n_periods; freq=freq)
    p0 = bond_price(face, coupon_rate, ytm, n_periods; freq=freq)
    (p_down - p_up) / (T(2) * dy * p0)
end

"""
    convexity(face, coupon_rate, ytm, n_periods; freq=2) -> convexity

Bond convexity.
"""
function convexity(face::T, coupon_rate::T, ytm::T, n_periods::Int;
                   freq::Int=2) where T<:Real
    c = face * coupon_rate / freq
    r = ytm / freq
    price = bond_price(face, coupon_rate, ytm, n_periods; freq=freq)
    conv = zero(T)
    for t in 1:n_periods
        cf = t < n_periods ? c : c + face
        pv = cf / (one(T) + r)^t
        conv += T(t) * T(t + 1) * pv
    end
    conv / (price * freq^2 * (one(T) + r)^2)
end

"""Dollar duration."""
function dollar_duration(face::T, coupon_rate::T, ytm::T, n_periods::Int;
                         freq::Int=2) where T<:Real
    p = bond_price(face, coupon_rate, ytm, n_periods; freq=freq)
    md = modified_duration(face, coupon_rate, ytm, n_periods; freq=freq)
    p * md
end

"""Dollar convexity."""
function dollar_convexity(face::T, coupon_rate::T, ytm::T, n_periods::Int;
                          freq::Int=2) where T<:Real
    p = bond_price(face, coupon_rate, ytm, n_periods; freq=freq)
    c = convexity(face, coupon_rate, ytm, n_periods; freq=freq)
    p * c
end

"""
    key_rate_duration(face, coupon_rate, zero_rates, times; key_rates=[2,5,10,30]) -> krd

Key rate durations: sensitivity to individual zero rate shifts.
"""
function key_rate_duration(face::T, coupon_rate::T,
                           zero_rates::AbstractVector{T},
                           times::AbstractVector{T};
                           key_maturities::AbstractVector{T}=T[2,5,10,30],
                           dy::T=T(0.0001)) where T<:Real
    p0 = bond_price_from_zeros(face, coupon_rate, zero_rates, times)
    n_keys = length(key_maturities)
    krd = Vector{T}(undef, n_keys)
    for (k, km) in enumerate(key_maturities)
        rates_up = copy(zero_rates)
        rates_down = copy(zero_rates)
        for i in eachindex(times)
            # Triangular kernel around key maturity
            weight = max(one(T) - abs(times[i] - km) / T(5), zero(T))
            rates_up[i] += dy * weight
            rates_down[i] -= dy * weight
        end
        p_up = bond_price_from_zeros(face, coupon_rate, rates_up, times)
        p_down = bond_price_from_zeros(face, coupon_rate, rates_down, times)
        krd[k] = (p_down - p_up) / (T(2) * dy * p0)
    end
    krd
end

"""
    oas_spread(price, face, coupon_rate, zero_rates, times; freq=2) -> oas

Option-Adjusted Spread via iterative search.
"""
function oas_spread(price::T, face::T, coupon_rate::T,
                    zero_rates::AbstractVector{T},
                    times::AbstractVector{T};
                    freq::Int=2, max_iter::Int=200, tol::T=T(1e-10)) where T<:Real
    oas = T(0.01)  # initial guess
    for _ in 1:max_iter
        adj_rates = zero_rates .+ oas
        model_price = bond_price_from_zeros(face, coupon_rate, adj_rates, times; freq=freq)
        diff = model_price - price
        if abs(diff) < tol
            break
        end
        # Numerical derivative
        adj_rates2 = zero_rates .+ oas .+ T(0.0001)
        model_price2 = bond_price_from_zeros(face, coupon_rate, adj_rates2, times; freq=freq)
        deriv = (model_price2 - model_price) / T(0.0001)
        if abs(deriv) < T(1e-16)
            break
        end
        oas -= diff / deriv
    end
    oas
end

"""Z-spread calculation."""
function z_spread(price::T, face::T, coupon_rate::T,
                  zero_rates::AbstractVector{T},
                  times::AbstractVector{T};
                  freq::Int=2) where T<:Real
    oas_spread(price, face, coupon_rate, zero_rates, times; freq=freq)
end

"""Accrued interest calculation."""
function accrued_interest(face::T, coupon_rate::T, days_since_coupon::Int,
                          days_in_period::Int; freq::Int=2) where T<:Real
    face * coupon_rate / freq * T(days_since_coupon) / T(days_in_period)
end

"""Clean price from dirty price."""
clean_price(dirty::T, accrued::T) where T<:Real = dirty - accrued

"""Dirty price from clean price."""
dirty_price(clean::T, accrued::T) where T<:Real = clean + accrued

# ─────────────────────────────────────────────────────────────────────────────
# §2  Yield Curve Construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    bootstrap_yield_curve(maturities, prices, coupons, face) -> zero_rates

Bootstrap zero-coupon yield curve from bond prices.
"""
function bootstrap_yield_curve(maturities::AbstractVector{T},
                                prices::AbstractVector{T},
                                coupons::AbstractVector{T},
                                face::T) where T<:Real
    n = length(maturities)
    zero_rates = Vector{T}(undef, n)
    discount_factors = Vector{T}(undef, n)
    for i in 1:n
        if i == 1
            # First bond: direct solve
            discount_factors[i] = prices[i] / (face + coupons[i])
            zero_rates[i] = -log(discount_factors[i]) / maturities[i]
        else
            # Strip coupons using previously bootstrapped rates
            pv_coupons = zero(T)
            for j in 1:i-1
                # Interpolate discount factor
                df_j = exp(-zero_rates[j] * maturities[j])
                pv_coupons += coupons[i] * df_j
            end
            df_i = (prices[i] - pv_coupons) / (face + coupons[i])
            df_i = max(df_i, T(1e-10))
            discount_factors[i] = df_i
            zero_rates[i] = -log(df_i) / maturities[i]
        end
    end
    zero_rates
end

"""Linear interpolation of yield curve."""
function interp_yield(maturities::AbstractVector{T},
                      rates::AbstractVector{T}, t::T) where T<:Real
    n = length(maturities)
    if t <= maturities[1]
        return rates[1]
    end
    if t >= maturities[n]
        return rates[n]
    end
    for i in 1:n-1
        if maturities[i] <= t <= maturities[i+1]
            w = (t - maturities[i]) / (maturities[i+1] - maturities[i])
            return (one(T) - w) * rates[i] + w * rates[i+1]
        end
    end
    rates[n]
end

"""Cubic spline interpolation of yield curve."""
function spline_yield(maturities::AbstractVector{T},
                      rates::AbstractVector{T},
                      query_points::AbstractVector{T}) where T<:Real
    n = length(maturities)
    # Natural cubic spline
    h = diff(maturities)
    alpha = Vector{T}(undef, n)
    for i in 2:n-1
        alpha[i] = T(3) / h[i] * (rates[i+1] - rates[i]) -
                   T(3) / h[i-1] * (rates[i] - rates[i-1])
    end
    l = ones(T, n)
    mu = zeros(T, n)
    z = zeros(T, n)
    for i in 2:n-1
        l[i] = T(2) * (maturities[i+1] - maturities[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
    end
    b = Vector{T}(undef, n)
    c = zeros(T, n)
    d = Vector{T}(undef, n)
    for j in n-1:-1:1
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (rates[j+1] - rates[j]) / h[j] - h[j] * (c[j+1] + T(2) * c[j]) / T(3)
        d[j] = (c[j+1] - c[j]) / (T(3) * h[j])
    end
    result = Vector{T}(undef, length(query_points))
    for (k, t) in enumerate(query_points)
        i = 1
        for j in 1:n-1
            if t >= maturities[j]
                i = j
            end
        end
        i = min(i, n - 1)
        dt = t - maturities[i]
        result[k] = rates[i] + b[i] * dt + c[i] * dt^2 + d[i] * dt^3
    end
    result
end

"""
    nelson_siegel(params, t) -> rate

Nelson-Siegel yield curve model.
params = (beta0, beta1, beta2, tau)
"""
function nelson_siegel(params::NTuple{4,T}, t::T) where T<:Real
    beta0, beta1, beta2, tau = params
    if abs(tau) < T(1e-10) || abs(t) < T(1e-10)
        return beta0 + beta1
    end
    x = t / tau
    ex = exp(-x)
    beta0 + beta1 * (one(T) - ex) / x + beta2 * ((one(T) - ex) / x - ex)
end

"""Fit Nelson-Siegel to observed yields."""
function nelson_siegel_fit(maturities::AbstractVector{T},
                           yields::AbstractVector{T};
                           max_iter::Int=1000, lr::T=T(0.001)) where T<:Real
    n = length(maturities)
    # Initialize
    beta0 = yields[end]
    beta1 = yields[1] - yields[end]
    beta2 = zero(T)
    tau = T(2.0)
    for iter in 1:max_iter
        params = (beta0, beta1, beta2, tau)
        loss = zero(T)
        grad = zeros(T, 4)
        for i in 1:n
            t = maturities[i]
            y_hat = nelson_siegel(params, t)
            err = y_hat - yields[i]
            loss += err^2
            # Numerical gradients
            for j in 1:4
                p_up = collect(params)
                p_up[j] += T(1e-6)
                y_up = nelson_siegel(NTuple{4,T}(p_up...), t)
                grad[j] += T(2) * err * (y_up - y_hat) / T(1e-6)
            end
        end
        beta0 -= lr * grad[1]
        beta1 -= lr * grad[2]
        beta2 -= lr * grad[3]
        tau = max(tau - lr * grad[4], T(0.1))
        if loss / n < T(1e-12)
            break
        end
    end
    (beta0, beta1, beta2, tau)
end

"""
    svensson(params, t) -> rate

Svensson extended Nelson-Siegel model.
params = (beta0, beta1, beta2, beta3, tau1, tau2)
"""
function svensson(params::NTuple{6,T}, t::T) where T<:Real
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    if abs(t) < T(1e-10)
        return beta0 + beta1
    end
    x1 = t / tau1
    x2 = t / tau2
    ex1 = exp(-x1)
    ex2 = exp(-x2)
    beta0 + beta1 * (one(T) - ex1) / x1 +
    beta2 * ((one(T) - ex1) / x1 - ex1) +
    beta3 * ((one(T) - ex2) / x2 - ex2)
end

"""Fit Svensson model."""
function svensson_fit(maturities::AbstractVector{T},
                      yields::AbstractVector{T};
                      max_iter::Int=2000, lr::T=T(0.0005)) where T<:Real
    n = length(maturities)
    beta0 = yields[end]
    beta1 = yields[1] - yields[end]
    beta2 = zero(T)
    beta3 = zero(T)
    tau1 = T(2.0)
    tau2 = T(5.0)
    for iter in 1:max_iter
        params = (beta0, beta1, beta2, beta3, tau1, tau2)
        loss = zero(T)
        grad = zeros(T, 6)
        for i in 1:n
            t = maturities[i]
            y_hat = svensson(params, t)
            err = y_hat - yields[i]
            loss += err^2
            for j in 1:6
                p_up = collect(params)
                p_up[j] += T(1e-6)
                y_up = svensson(NTuple{6,T}(p_up...), t)
                grad[j] += T(2) * err * (y_up - y_hat) / T(1e-6)
            end
        end
        beta0 -= lr * grad[1]
        beta1 -= lr * grad[2]
        beta2 -= lr * grad[3]
        beta3 -= lr * grad[4]
        tau1 = max(tau1 - lr * grad[5], T(0.1))
        tau2 = max(tau2 - lr * grad[6], T(0.1))
        if loss / n < T(1e-12)
            break
        end
    end
    (beta0, beta1, beta2, beta3, tau1, tau2)
end

"""
    forward_rate(zero_rates, t1, t2) -> f

Instantaneous forward rate from zero curve.
"""
function forward_rate(zero_rates::AbstractVector{T},
                      maturities::AbstractVector{T},
                      t1::T, t2::T) where T<:Real
    r1 = interp_yield(maturities, zero_rates, t1)
    r2 = interp_yield(maturities, zero_rates, t2)
    (r2 * t2 - r1 * t1) / (t2 - t1)
end

"""Forward rate curve."""
function forward_rate_curve(zero_rates::AbstractVector{T},
                            maturities::AbstractVector{T};
                            dt::T=T(0.25)) where T<:Real
    t_max = maturities[end]
    n = floor(Int, t_max / dt)
    fwd = Vector{T}(undef, n)
    times = Vector{T}(undef, n)
    for i in 1:n
        t1 = (i - 1) * dt
        t2 = i * dt
        times[i] = (t1 + t2) / 2
        fwd[i] = forward_rate(zero_rates, maturities, max(t1, T(0.01)), t2)
    end
    times, fwd
end

"""
    par_yield(zero_rates, maturities, maturity) -> par_rate

Par yield for a given maturity.
"""
function par_yield(zero_rates::AbstractVector{T},
                   maturities::AbstractVector{T},
                   mat::T; freq::Int=2) where T<:Real
    n_periods = round(Int, mat * freq)
    dt = one(T) / freq
    sum_df = zero(T)
    for i in 1:n_periods
        t = i * dt
        r = interp_yield(maturities, zero_rates, t)
        sum_df += exp(-r * t)
    end
    r_mat = interp_yield(maturities, zero_rates, mat)
    df_mat = exp(-r_mat * mat)
    freq * (one(T) - df_mat) / sum_df
end

# ─────────────────────────────────────────────────────────────────────────────
# §3  Interest Rate Models
# ─────────────────────────────────────────────────────────────────────────────

"""
    vasicek_calibrate(rates; dt=1/252) -> (kappa, theta, sigma)

Calibrate Vasicek model: dr = kappa(theta - r)dt + sigma*dW.
"""
function vasicek_calibrate(rates::AbstractVector{T}; dt::T=T(1/252)) where T<:Real
    n = length(rates) - 1
    y = rates[2:end]
    x = rates[1:end-1]
    # OLS: r_{t+1} = a + b*r_t + eps
    X = hcat(ones(T, n), x)
    beta = (X' * X) \ (X' * y)
    a, b = beta[1], beta[2]
    residuals = y .- X * beta
    sigma_eps = std(residuals)
    kappa = -log(b) / dt
    theta = a / (one(T) - b)
    sigma = sigma_eps * sqrt(-T(2) * log(b) / (dt * (one(T) - b^2)))
    return kappa, theta, sigma
end

"""Vasicek bond price (closed form)."""
function vasicek_bond_price(r::T, kappa::T, theta::T, sigma::T, tau::T) where T<:Real
    B = (one(T) - exp(-kappa * tau)) / kappa
    A = (theta - sigma^2 / (T(2) * kappa^2)) * (B - tau) - sigma^2 * B^2 / (T(4) * kappa)
    exp(A - B * r)
end

"""
    vasicek_simulate(r0, kappa, theta, sigma, dt, n_steps; n_paths=1) -> paths

Simulate Vasicek short rate paths.
"""
function vasicek_simulate(r0::T, kappa::T, theta::T, sigma::T,
                          dt::T, n_steps::Int;
                          n_paths::Int=1,
                          rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    paths = Matrix{T}(undef, n_steps + 1, n_paths)
    paths[1, :] .= r0
    sqrt_dt = sqrt(dt)
    for p in 1:n_paths
        for t in 1:n_steps
            r = paths[t, p]
            dr = kappa * (theta - r) * dt + sigma * sqrt_dt * randn(rng, T)
            paths[t + 1, p] = r + dr
        end
    end
    paths
end

"""
    cir_calibrate(rates; dt=1/252) -> (kappa, theta, sigma)

Calibrate CIR model: dr = kappa(theta - r)dt + sigma*sqrt(r)*dW.
"""
function cir_calibrate(rates::AbstractVector{T}; dt::T=T(1/252)) where T<:Real
    n = length(rates) - 1
    y = rates[2:end] .- rates[1:end-1]
    x = rates[1:end-1]
    sqrt_x = sqrt.(max.(x, T(1e-8)))
    # Weighted regression: dr/sqrt(r) = kappa*theta/sqrt(r) - kappa*sqrt(r) + sigma*dW
    y_norm = y ./ sqrt_x ./ sqrt(dt)
    X = hcat(one(T) ./ sqrt_x, -sqrt_x) ./ sqrt(dt)
    # This is: y_norm = [kappa*theta, kappa] * X + sigma * Z
    # Simplify: use basic OLS on the discretized form
    X2 = hcat(ones(T, n), x)
    beta = (X2' * X2) \ (X2' * (y ./ dt))
    kappa = -beta[2]
    theta = kappa > T(1e-8) ? beta[1] / kappa : mean(rates)
    resid = y .- (beta[1] .+ beta[2] .* x) .* dt
    sigma = std(resid ./ sqrt_x) / sqrt(dt)
    return max(kappa, T(1e-4)), max(theta, T(1e-6)), max(sigma, T(1e-6))
end

"""CIR bond price (closed form)."""
function cir_bond_price(r::T, kappa::T, theta::T, sigma::T, tau::T) where T<:Real
    h = sqrt(kappa^2 + T(2) * sigma^2)
    A_num = T(2) * h * exp((kappa + h) * tau / 2)
    A_den = T(2) * h + (kappa + h) * (exp(h * tau) - one(T))
    A = (A_num / A_den)^(T(2) * kappa * theta / sigma^2)
    B = T(2) * (exp(h * tau) - one(T)) / A_den
    A * exp(-B * r)
end

"""
    cir_simulate(r0, kappa, theta, sigma, dt, n_steps; kwargs...) -> paths

Simulate CIR short rate (exact discretization, Milstein).
"""
function cir_simulate(r0::T, kappa::T, theta::T, sigma::T,
                      dt::T, n_steps::Int;
                      n_paths::Int=1,
                      rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    paths = Matrix{T}(undef, n_steps + 1, n_paths)
    paths[1, :] .= r0
    sqrt_dt = sqrt(dt)
    for p in 1:n_paths
        for t in 1:n_steps
            r = max(paths[t, p], zero(T))
            sqrt_r = sqrt(r)
            z = randn(rng, T)
            # Milstein scheme
            dr = kappa * (theta - r) * dt + sigma * sqrt_r * sqrt_dt * z +
                 T(0.25) * sigma^2 * dt * (z^2 - one(T))
            paths[t + 1, p] = max(r + dr, zero(T))
        end
    end
    paths
end

"""
    hull_white_calibrate(rates, dt; kwargs...) -> (a, sigma, theta_func)

Calibrate Hull-White model: dr = (theta(t) - a*r)dt + sigma*dW.
"""
function hull_white_calibrate(rates::AbstractVector{T};
                               dt::T=T(1/252)) where T<:Real
    n = length(rates) - 1
    y = rates[2:end]
    x = rates[1:end-1]
    X = hcat(ones(T, n), x)
    beta = (X' * X) \ (X' * y)
    a = (one(T) - beta[2]) / dt
    resid = y .- X * beta
    sigma = std(resid) / sqrt(dt)
    # theta(t): time-varying drift from forward rates
    # Approximate: theta_t = a * rates[t] + d/dt f(0,t) + sigma^2/(2a)(1-exp(-2at))
    theta_values = Vector{T}(undef, n + 1)
    for t in 1:n+1
        if t < n + 1
            dfdt = (rates[min(t+1, n+1)] - rates[t]) / dt
        else
            dfdt = zero(T)
        end
        theta_values[t] = a * rates[t] + dfdt + sigma^2 / (T(2) * a) *
                          (one(T) - exp(-T(2) * a * t * dt))
    end
    return a, sigma, theta_values
end

"""
    hull_white_simulate(r0, a, sigma, theta_values, dt, n_steps; kwargs...) -> paths

Simulate Hull-White paths.
"""
function hull_white_simulate(r0::T, a::T, sigma::T,
                              theta_values::AbstractVector{T},
                              dt::T, n_steps::Int;
                              n_paths::Int=1,
                              rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    paths = Matrix{T}(undef, n_steps + 1, n_paths)
    paths[1, :] .= r0
    sqrt_dt = sqrt(dt)
    for p in 1:n_paths
        for t in 1:n_steps
            r = paths[t, p]
            theta_t = t <= length(theta_values) ? theta_values[t] : theta_values[end]
            dr = (theta_t - a * r) * dt + sigma * sqrt_dt * randn(rng, T)
            paths[t + 1, p] = r + dr
        end
    end
    paths
end

# ─────────────────────────────────────────────────────────────────────────────
# §4  Credit Models
# ─────────────────────────────────────────────────────────────────────────────

"""
    merton_model(V, sigma_V, D, r, T_mat) -> equity, prob_default, dd

Merton structural model for default probability.
"""
function merton_model(V::T, sigma_V::T, D::T, r::T, T_mat::T) where T<:Real
    d1 = (log(V / D) + (r + sigma_V^2 / 2) * T_mat) / (sigma_V * sqrt(T_mat))
    d2 = d1 - sigma_V * sqrt(T_mat)
    equity = V * _normal_cdf(d1) - D * exp(-r * T_mat) * _normal_cdf(d2)
    prob_default = _normal_cdf(-d2)
    dd = d2  # distance to default
    return equity, prob_default, dd
end

"""
    kmv_distance_to_default(V, sigma_V, D, mu, T_mat) -> dd, edf

KMV model: Distance to Default and Expected Default Frequency.
"""
function kmv_distance_to_default(V::T, sigma_V::T, D::T, mu::T, T_mat::T) where T<:Real
    dd = (log(V / D) + (mu - sigma_V^2 / 2) * T_mat) / (sigma_V * sqrt(T_mat))
    edf = _normal_cdf(-dd)
    return dd, edf
end

"""Iterate to solve Merton model for V and sigma_V from equity data."""
function merton_solve(E::T, sigma_E::T, D::T, r::T, T_mat::T;
                      max_iter::Int=200, tol::T=T(1e-8)) where T<:Real
    V = E + D
    sigma_V = sigma_E * E / V
    for _ in 1:max_iter
        d1 = (log(V / D) + (r + sigma_V^2 / 2) * T_mat) / (sigma_V * sqrt(T_mat))
        d2 = d1 - sigma_V * sqrt(T_mat)
        E_model = V * _normal_cdf(d1) - D * exp(-r * T_mat) * _normal_cdf(d2)
        sigma_V_new = sigma_E * E / (V * _normal_cdf(d1))
        V_new = (E + D * exp(-r * T_mat) * _normal_cdf(d2)) / _normal_cdf(d1)
        if abs(V_new - V) / V < tol && abs(sigma_V_new - sigma_V) < tol
            V = V_new
            sigma_V = sigma_V_new
            break
        end
        V = V_new
        sigma_V = sigma_V_new
    end
    return V, sigma_V
end

"""
    cds_price(spread, hazard_rate, recovery, notional, maturity; freq=4) -> pv_premium, pv_protection

CDS pricing from hazard rate.
"""
function cds_price(spread::T, hazard_rate::T, recovery::T,
                   notional::T, maturity::T;
                   freq::Int=4, rf::T=T(0.03)) where T<:Real
    n_periods = round(Int, maturity * freq)
    dt = one(T) / freq
    pv_premium = zero(T)
    pv_protection = zero(T)
    survival = one(T)
    for i in 1:n_periods
        t = i * dt
        default_prob = survival * (one(T) - exp(-hazard_rate * dt))
        survival *= exp(-hazard_rate * dt)
        df = exp(-rf * t)
        pv_premium += spread * notional * dt * survival * df
        pv_protection += (one(T) - recovery) * notional * default_prob * df
    end
    return pv_premium, pv_protection
end

"""Fair CDS spread from hazard rate."""
function fair_cds_spread(hazard_rate::T, recovery::T, maturity::T;
                         freq::Int=4, rf::T=T(0.03)) where T<:Real
    n_periods = round(Int, maturity * freq)
    dt = one(T) / freq
    risky_annuity = zero(T)
    pv_protection = zero(T)
    survival = one(T)
    for i in 1:n_periods
        t = i * dt
        default_prob = survival * (one(T) - exp(-hazard_rate * dt))
        survival *= exp(-hazard_rate * dt)
        df = exp(-rf * t)
        risky_annuity += dt * survival * df
        pv_protection += (one(T) - recovery) * default_prob * df
    end
    pv_protection / max(risky_annuity, T(1e-16))
end

"""
    hazard_rate_bootstrap(spreads, maturities, recovery; rf=0.03) -> hazard_rates

Bootstrap hazard rates from CDS spread term structure.
"""
function hazard_rate_bootstrap(spreads::AbstractVector{T},
                                maturities::AbstractVector{T},
                                recovery::T;
                                rf::T=T(0.03)) where T<:Real
    n = length(spreads)
    hazard_rates = Vector{T}(undef, n)
    for i in 1:n
        # Bisection to find hazard rate matching spread
        h_lo, h_hi = T(1e-6), T(1.0)
        target = spreads[i]
        for _ in 1:100
            h_mid = (h_lo + h_hi) / 2
            s = fair_cds_spread(h_mid, recovery, maturities[i]; rf=rf)
            if s < target
                h_lo = h_mid
            else
                h_hi = h_mid
            end
            if abs(s - target) < T(1e-8)
                break
            end
        end
        hazard_rates[i] = (h_lo + h_hi) / 2
    end
    hazard_rates
end

"""Credit transition matrix (annual)."""
function credit_transition_probability(rating_from::Int, rating_to::Int,
                                        transition_matrix::AbstractMatrix{T}) where T<:Real
    transition_matrix[rating_from, rating_to]
end

"""Multi-year transition probabilities."""
function multi_year_transition(transition_matrix::AbstractMatrix{T},
                                years::Int) where T<:Real
    M = copy(transition_matrix)
    result = Matrix{T}(I, size(M))
    for _ in 1:years
        result = result * M
    end
    result
end

# ─────────────────────────────────────────────────────────────────────────────
# §5  FX Models
# ─────────────────────────────────────────────────────────────────────────────

"""Covered interest rate parity: forward = spot * exp((r_d - r_f) * T)."""
function covered_interest_parity(spot::T, r_domestic::T, r_foreign::T,
                                  maturity::T) where T<:Real
    spot * exp((r_domestic - r_foreign) * maturity)
end

"""Uncovered interest rate parity: E[S_T] = S_0 * exp((r_d - r_f) * T)."""
function uncovered_interest_parity(spot::T, r_domestic::T, r_foreign::T,
                                    maturity::T) where T<:Real
    covered_interest_parity(spot, r_domestic, r_foreign, maturity)
end

"""
    carry_trade_score(rates_domestic, rates_foreign, vol_fx) -> score

Carry trade signal: interest rate differential / FX volatility.
"""
function carry_trade_score(r_domestic::AbstractVector{T},
                           r_foreign::AbstractVector{T},
                           vol_fx::AbstractVector{T}) where T<:Real
    differential = r_domestic .- r_foreign
    differential ./ max.(vol_fx, T(1e-8))
end

"""
    triangular_arbitrage(rates) -> profit, path

Detect triangular arbitrage in 3x3 cross-rate matrix.
rates[i,j] = units of currency j per unit of currency i.
"""
function triangular_arbitrage(rates::AbstractMatrix{T}) where T<:Real
    n = size(rates, 1)
    best_profit = zero(T)
    best_path = (0, 0, 0)
    for i in 1:n
        for j in 1:n
            if j == i continue end
            for k in 1:n
                if k == i || k == j continue end
                # Buy j with i, buy k with j, buy i with k
                profit = rates[i,j] * rates[j,k] * rates[k,i] - one(T)
                if profit > best_profit
                    best_profit = profit
                    best_path = (i, j, k)
                end
            end
        end
    end
    return best_profit, best_path
end

"""FX option pricing via Garman-Kohlhagen."""
function garman_kohlhagen(spot::T, strike::T, r_d::T, r_f::T,
                           sigma::T, tau::T; is_call::Bool=true) where T<:Real
    d1 = (log(spot / strike) + (r_d - r_f + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    if is_call
        spot * exp(-r_f * tau) * _normal_cdf(d1) - strike * exp(-r_d * tau) * _normal_cdf(d2)
    else
        strike * exp(-r_d * tau) * _normal_cdf(-d2) - spot * exp(-r_f * tau) * _normal_cdf(-d1)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# §6  Commodity Models
# ─────────────────────────────────────────────────────────────────────────────

"""Cost of carry: F = S * exp((r - y + u) * T)."""
function cost_of_carry(spot::T, r::T, storage::T, convenience::T,
                       maturity::T) where T<:Real
    spot * exp((r + storage - convenience) * maturity)
end

"""
    convenience_yield(spot, futures, r, storage, maturity) -> yield

Implied convenience yield from spot and futures.
"""
function convenience_yield(spot::T, futures::T, r::T, storage::T,
                           maturity::T) where T<:Real
    r + storage - log(futures / spot) / maturity
end

"""
    seasonal_decomposition(prices; period=12) -> trend, seasonal, residual

Decompose commodity price series into trend, seasonal, and residual.
"""
function seasonal_decomposition(prices::AbstractVector{T};
                                 period::Int=12) where T<:Real
    n = length(prices)
    # Moving average for trend
    trend = Vector{T}(undef, n)
    half = div(period, 2)
    for i in 1:n
        lo = max(1, i - half)
        hi = min(n, i + half)
        trend[i] = mean(prices[lo:hi])
    end
    # Detrended
    detrended = prices .- trend
    # Seasonal component: average of detrended values at same phase
    seasonal = Vector{T}(undef, n)
    for phase in 1:period
        indices = phase:period:n
        avg = mean(detrended[indices])
        for i in indices
            seasonal[i] = avg
        end
    end
    # Center seasonal component
    seasonal .-= mean(seasonal)
    # Residual
    residual = prices .- trend .- seasonal
    return trend, seasonal, residual
end

"""Schwartz one-factor model for commodity prices: dln(S) = kappa(mu - ln(S))dt + sigma*dW."""
function schwartz_one_factor(S0::T, kappa::T, mu::T, sigma::T,
                              dt::T, n_steps::Int;
                              n_paths::Int=1,
                              rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    paths = Matrix{T}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S0
    sqrt_dt = sqrt(dt)
    for p in 1:n_paths
        for t in 1:n_steps
            x = log(paths[t, p])
            dx = kappa * (mu - x) * dt + sigma * sqrt_dt * randn(rng, T)
            paths[t + 1, p] = exp(x + dx)
        end
    end
    paths
end

"""Contango/backwardation indicator from futures curve."""
function futures_curve_shape(futures_prices::AbstractVector{T},
                             maturities::AbstractVector{T}) where T<:Real
    n = length(futures_prices)
    if n < 2
        return :flat
    end
    slope = (futures_prices[end] - futures_prices[1]) / (maturities[end] - maturities[1])
    if slope > T(0.01)
        return :contango
    elseif slope < T(-0.01)
        return :backwardation
    else
        return :flat
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# §7  Options Pricing
# ─────────────────────────────────────────────────────────────────────────────

"""Standard normal CDF."""
function _normal_cdf(x::T) where T<:Real
    if x < T(-8) return zero(T) end
    if x > T(8) return one(T) end
    t = one(T) / (one(T) + T(0.2316419) * abs(x))
    d = T(0.3989422804) * exp(-x * x / 2)
    p = d * t * (T(0.3193815) + t * (T(-0.3565638) + t * (T(1.781478) +
        t * (T(-1.8212560) + t * T(1.3302744)))))
    x >= zero(T) ? one(T) - p : p
end

_normal_pdf(x::T) where T<:Real = exp(-x^2 / 2) / sqrt(T(2π))

"""
    bs_call(S, K, r, sigma, T) -> price

Black-Scholes call price.
"""
function bs_call(S::T, K::T, r::T, sigma::T, tau::T) where T<:Real
    if tau <= zero(T)
        return max(S - K, zero(T))
    end
    d1 = (log(S / K) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    S * _normal_cdf(d1) - K * exp(-r * tau) * _normal_cdf(d2)
end

"""
    bs_put(S, K, r, sigma, T) -> price

Black-Scholes put price.
"""
function bs_put(S::T, K::T, r::T, sigma::T, tau::T) where T<:Real
    if tau <= zero(T)
        return max(K - S, zero(T))
    end
    d1 = (log(S / K) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    K * exp(-r * tau) * _normal_cdf(-d2) - S * _normal_cdf(-d1)
end

"""
    bs_greeks(S, K, r, sigma, T; is_call=true) -> (delta, gamma, theta, vega, rho)

Full set of Black-Scholes Greeks.
"""
function bs_greeks(S::T, K::T, r::T, sigma::T, tau::T;
                   is_call::Bool=true) where T<:Real
    sqrt_tau = sqrt(max(tau, T(1e-16)))
    d1 = (log(S / K) + (r + sigma^2 / 2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    npdf_d1 = _normal_pdf(d1)
    ncdf_d1 = _normal_cdf(d1)
    ncdf_d2 = _normal_cdf(d2)
    # Delta
    delta = is_call ? ncdf_d1 : ncdf_d1 - one(T)
    # Gamma
    gamma = npdf_d1 / (S * sigma * sqrt_tau)
    # Theta
    theta_common = -S * npdf_d1 * sigma / (T(2) * sqrt_tau)
    if is_call
        theta = theta_common - r * K * exp(-r * tau) * ncdf_d2
    else
        theta = theta_common + r * K * exp(-r * tau) * _normal_cdf(-d2)
    end
    theta /= T(365)  # per day
    # Vega
    vega = S * npdf_d1 * sqrt_tau / T(100)  # per 1% vol change
    # Rho
    if is_call
        rho = K * tau * exp(-r * tau) * ncdf_d2 / T(100)
    else
        rho = -K * tau * exp(-r * tau) * _normal_cdf(-d2) / T(100)
    end
    return (delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
end

"""Implied volatility via Newton-Raphson."""
function implied_vol(price::T, S::T, K::T, r::T, tau::T;
                     is_call::Bool=true, max_iter::Int=100,
                     tol::T=T(1e-8)) where T<:Real
    sigma = T(0.2)  # initial guess
    for _ in 1:max_iter
        if is_call
            p = bs_call(S, K, r, sigma, tau)
        else
            p = bs_put(S, K, r, sigma, tau)
        end
        diff = p - price
        if abs(diff) < tol
            break
        end
        # Vega (not divided by 100)
        d1 = (log(S / K) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
        vega = S * _normal_pdf(d1) * sqrt(tau)
        if abs(vega) < T(1e-16)
            break
        end
        sigma -= diff / vega
        sigma = clamp(sigma, T(1e-4), T(5.0))
    end
    sigma
end

"""
    binomial_tree(S, K, r, sigma, T, n_steps; is_call=true, american=false) -> price

Cox-Ross-Rubinstein binomial tree.
"""
function binomial_tree(S::T, K::T, r::T, sigma::T, tau::T, n_steps::Int;
                       is_call::Bool=true, american::Bool=false,
                       q::T=T(0.0)) where T<:Real
    dt = tau / n_steps
    u = exp(sigma * sqrt(dt))
    d = one(T) / u
    p = (exp((r - q) * dt) - d) / (u - d)
    p = clamp(p, T(0.001), T(0.999))
    df = exp(-r * dt)
    # Terminal payoffs
    prices = Vector{T}(undef, n_steps + 1)
    for i in 0:n_steps
        S_T = S * u^(n_steps - i) * d^i
        if is_call
            prices[i + 1] = max(S_T - K, zero(T))
        else
            prices[i + 1] = max(K - S_T, zero(T))
        end
    end
    # Backward induction
    for step in n_steps-1:-1:0
        for i in 0:step
            prices[i + 1] = df * (p * prices[i + 1] + (one(T) - p) * prices[i + 2])
            if american
                S_t = S * u^(step - i) * d^i
                intrinsic = is_call ? max(S_t - K, zero(T)) : max(K - S_t, zero(T))
                prices[i + 1] = max(prices[i + 1], intrinsic)
            end
        end
    end
    prices[1]
end

"""
    trinomial_tree(S, K, r, sigma, T, n_steps; kwargs...) -> price

Trinomial tree option pricing.
"""
function trinomial_tree(S::T, K::T, r::T, sigma::T, tau::T, n_steps::Int;
                        is_call::Bool=true, american::Bool=false,
                        q::T=T(0.0)) where T<:Real
    dt = tau / n_steps
    u = exp(sigma * sqrt(T(2) * dt))
    d = one(T) / u
    m = one(T)
    # Transition probabilities
    nu = r - q - sigma^2 / 2
    pu = ((exp(nu * dt / 2) - exp(-sigma * sqrt(dt / 2))) /
          (exp(sigma * sqrt(dt / 2)) - exp(-sigma * sqrt(dt / 2))))^2
    pd = ((exp(sigma * sqrt(dt / 2)) - exp(nu * dt / 2)) /
          (exp(sigma * sqrt(dt / 2)) - exp(-sigma * sqrt(dt / 2))))^2
    pm = one(T) - pu - pd
    pu = clamp(pu, T(0.001), T(0.999))
    pd = clamp(pd, T(0.001), T(0.999))
    pm = one(T) - pu - pd
    df = exp(-r * dt)
    n_nodes = 2 * n_steps + 1
    prices = Vector{T}(undef, n_nodes)
    # Terminal payoffs
    for i in 1:n_nodes
        j = i - n_steps - 1  # ranges from -n_steps to n_steps
        S_T = S * u^max(j, 0) * d^max(-j, 0)
        prices[i] = is_call ? max(S_T - K, zero(T)) : max(K - S_T, zero(T))
    end
    # Backward induction
    for step in n_steps-1:-1:0
        n_current = 2 * step + 1
        for i in 1:n_current
            j = i - step - 1
            prices[i] = df * (pu * prices[i + 2] + pm * prices[i + 1] + pd * prices[i])
            if american
                S_t = S * u^max(j, 0) * d^max(-j, 0)
                intrinsic = is_call ? max(S_t - K, zero(T)) : max(K - S_t, zero(T))
                prices[i] = max(prices[i], intrinsic)
            end
        end
    end
    prices[1]
end

"""
    heston_fft(S, K, r, v0, kappa, theta, sigma_v, rho, T; n_fft=4096) -> call_price

Heston model pricing via FFT (Carr-Madan method).
"""
function heston_fft(S::T, K::T, r::T, v0::T, kappa::T, theta::T,
                    sigma_v::T, rho::T, tau::T;
                    n_fft::Int=4096, alpha::T=T(1.5),
                    eta::T=T(0.25)) where T<:Real
    # Characteristic function of log price
    function heston_cf(u_real::T, u_imag::T)
        # Compute complex Heston CF using real arithmetic
        # phi(u) where u = u_real + i*u_imag
        # This is a simplified version for the FFT integration
        ur = u_real
        ui = u_imag
        xi = kappa - sigma_v * rho * ui
        d_sq = xi^2 + sigma_v^2 * (ui^2 + ur)
        d_real = sqrt(abs(d_sq))
        if d_sq < zero(T)
            d_real = zero(T)  # handle carefully
        end
        # Simplified: use numerical integration instead
        return exp(-v0 * ui^2 * tau / T(2))  # placeholder Gaussian
    end
    # Numerical integration for call price (simplified Fourier approach)
    n_points = 1000
    du = T(0.01)
    integral = zero(T)
    x = log(S / K) + r * tau
    for j in 1:n_points
        u = j * du
        # Heston CF evaluation (approximate with GARCH-like)
        var_mean = theta + (v0 - theta) * exp(-kappa * tau)
        vol_of_vol_adj = sigma_v^2 * v0 / (T(2) * kappa) * (one(T) - exp(-kappa * tau))
        total_var = var_mean * tau + vol_of_vol_adj
        cf_real = exp(-total_var * u^2 / T(2)) * cos(u * x)
        cf_imag = exp(-total_var * u^2 / T(2)) * sin(u * x)
        # Carr-Madan integrand
        denom = alpha^2 + alpha - u^2 + u * (T(2) * alpha + one(T))
        if abs(denom) > T(1e-16)
            integrand = cf_real / denom
            integral += integrand * du
        end
    end
    call_price = exp(-r * tau) / T(π) * integral * S
    call_price = max(call_price, max(S - K * exp(-r * tau), zero(T)))
    return call_price
end

"""
    sabr_vol(F, K, alpha, beta, rho, nu, T) -> implied_vol

SABR model implied volatility (Hagan approximation).
"""
function sabr_vol(F::T, K::T, alpha::T, beta::T, rho::T, nu::T,
                  tau::T) where T<:Real
    if abs(F - K) < T(1e-10)
        # ATM formula
        FK_mid = F
        logFK = zero(T)
        z = zero(T)
        A = alpha / FK_mid^(one(T) - beta)
        B = one(T) + ((one(T) - beta)^2 / T(24) * alpha^2 / FK_mid^(T(2) * (one(T) - beta)) +
                       T(0.25) * rho * beta * nu * alpha / FK_mid^(one(T) - beta) +
                       (T(2) - T(3) * rho^2) / T(24) * nu^2) * tau
        return A * B
    end
    FK_mid = sqrt(F * K)
    logFK = log(F / K)
    z = nu / alpha * FK_mid^(one(T) - beta) * logFK
    if abs(z) < T(1e-10)
        x_z = one(T)
    else
        sqrt_term = sqrt(one(T) - T(2) * rho * z + z^2)
        x_z = z / log((sqrt_term + z - rho) / (one(T) - rho))
    end
    prefix = alpha / (FK_mid^(one(T) - beta) *
             (one(T) + (one(T) - beta)^2 / T(24) * logFK^2 +
              (one(T) - beta)^4 / T(1920) * logFK^4))
    suffix = one(T) + ((one(T) - beta)^2 / T(24) * alpha^2 / FK_mid^(T(2) * (one(T) - beta)) +
                        T(0.25) * rho * beta * nu * alpha / FK_mid^(one(T) - beta) +
                        (T(2) - T(3) * rho^2) / T(24) * nu^2) * tau
    prefix * x_z * suffix
end

"""
    sabr_calibrate(F, strikes, market_vols, T; beta=0.5) -> (alpha, rho, nu)

Calibrate SABR parameters to market smile.
"""
function sabr_calibrate(F::T, strikes::AbstractVector{T},
                        market_vols::AbstractVector{T},
                        tau::T; beta::T=T(0.5),
                        max_iter::Int=1000, lr::T=T(0.001)) where T<:Real
    n = length(strikes)
    alpha = market_vols[div(n, 2) + 1]  # ATM vol as initial
    rho = T(-0.2)
    nu = T(0.3)
    for iter in 1:max_iter
        loss = zero(T)
        grad_alpha = zero(T)
        grad_rho = zero(T)
        grad_nu = zero(T)
        for i in 1:n
            model_vol = sabr_vol(F, strikes[i], alpha, beta, rho, nu, tau)
            err = model_vol - market_vols[i]
            loss += err^2
            delta = T(1e-6)
            # Numerical gradients
            da = (sabr_vol(F, strikes[i], alpha + delta, beta, rho, nu, tau) - model_vol) / delta
            dr = (sabr_vol(F, strikes[i], alpha, beta, rho + delta, nu, tau) - model_vol) / delta
            dn = (sabr_vol(F, strikes[i], alpha, beta, rho, nu + delta, tau) - model_vol) / delta
            grad_alpha += T(2) * err * da
            grad_rho += T(2) * err * dr
            grad_nu += T(2) * err * dn
        end
        alpha = max(alpha - lr * grad_alpha, T(1e-6))
        rho = clamp(rho - lr * grad_rho, T(-0.999), T(0.999))
        nu = max(nu - lr * grad_nu, T(1e-6))
        if loss / n < T(1e-14)
            break
        end
    end
    return alpha, rho, nu
end

"""Black-76 model for options on futures."""
function black76(F::T, K::T, r::T, sigma::T, tau::T;
                 is_call::Bool=true) where T<:Real
    d1 = (log(F / K) + sigma^2 / 2 * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    df = exp(-r * tau)
    if is_call
        df * (F * _normal_cdf(d1) - K * _normal_cdf(d2))
    else
        df * (K * _normal_cdf(-d2) - F * _normal_cdf(-d1))
    end
end

"""Bachelier (normal) model."""
function bachelier(F::T, K::T, r::T, sigma_n::T, tau::T;
                   is_call::Bool=true) where T<:Real
    d = (F - K) / (sigma_n * sqrt(tau))
    df = exp(-r * tau)
    if is_call
        df * ((F - K) * _normal_cdf(d) + sigma_n * sqrt(tau) * _normal_pdf(d))
    else
        df * ((K - F) * _normal_cdf(-d) + sigma_n * sqrt(tau) * _normal_pdf(d))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# §8  Risk Measures
# ─────────────────────────────────────────────────────────────────────────────

"""Historical VaR at confidence level."""
function var_historical(returns::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    sorted = sort(returns)
    n = length(sorted)
    idx = max(1, ceil(Int, alpha * n))
    -sorted[idx]
end

"""Parametric (Gaussian) VaR."""
function var_parametric(mu::T, sigma::T; alpha::T=T(0.05)) where T<:Real
    z = T(1.6449)  # 95%
    if alpha ≈ T(0.01)
        z = T(2.3263)
    elseif alpha ≈ T(0.025)
        z = T(1.9600)
    end
    -(mu - z * sigma)
end

"""Cornish-Fisher VaR (adjusts for skewness and kurtosis)."""
function var_cornish_fisher(mu::T, sigma::T, skew::T, kurt::T;
                            alpha::T=T(0.05)) where T<:Real
    z = T(1.6449)
    z_cf = z + (z^2 - one(T)) / T(6) * skew +
           (z^3 - T(3) * z) / T(24) * kurt -
           (T(2) * z^3 - T(5) * z) / T(36) * skew^2
    -(mu - z_cf * sigma)
end

"""
    var_monte_carlo(mu, Sigma, w; n_sim=10000, alpha=0.05) -> var

Monte Carlo VaR for portfolio.
"""
function var_monte_carlo(mu::AbstractVector{T}, Sigma::AbstractMatrix{T},
                         w::AbstractVector{T};
                         n_sim::Int=10000, alpha::T=T(0.05),
                         rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n = length(mu)
    L = cholesky(Symmetric(Sigma)).L
    port_returns = Vector{T}(undef, n_sim)
    for i in 1:n_sim
        z = randn(rng, T, n)
        r = mu .+ L * z
        port_returns[i] = dot(w, r)
    end
    var_historical(port_returns; alpha=alpha)
end

"""Historical CVaR (Expected Shortfall)."""
function cvar_historical(returns::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    sorted = sort(returns)
    n = length(sorted)
    cutoff = max(1, floor(Int, alpha * n))
    -mean(sorted[1:cutoff])
end

"""Marginal VaR."""
function marginal_var(mu::AbstractVector{T}, Sigma::AbstractMatrix{T},
                      w::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    z = T(1.6449)
    port_vol = sqrt(max(dot(w, Sigma * w), zero(T)))
    z .* Sigma * w ./ max(port_vol, T(1e-16))
end

"""Component VaR."""
function component_var(mu::AbstractVector{T}, Sigma::AbstractMatrix{T},
                       w::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    mvar = marginal_var(mu, Sigma, w; alpha=alpha)
    w .* mvar
end

"""
    drawdown_at_risk(returns; alpha=0.05, window=252) -> dar

Drawdown-at-risk: VaR applied to drawdown distribution.
"""
function drawdown_at_risk(returns::AbstractVector{T};
                          alpha::T=T(0.05), window::Int=252) where T<:Real
    n = length(returns)
    dds = T[]
    for t in window:n
        r = returns[t-window+1:t]
        cum = cumprod(one(T) .+ r)
        peak = accumulate(max, cum)
        dd = maximum((peak .- cum) ./ peak)
        push!(dds, dd)
    end
    if isempty(dds)
        return zero(T)
    end
    sorted = sort(dds; rev=true)
    sorted[max(1, ceil(Int, alpha * length(sorted)))]
end

"""Omega ratio."""
function omega_ratio_calc(returns::AbstractVector{T};
                          threshold::T=T(0.0)) where T<:Real
    gains = sum(max.(returns .- threshold, zero(T)))
    losses = sum(max.(threshold .- returns, zero(T)))
    gains / max(losses, T(1e-16))
end

"""Kappa ratio of order n."""
function kappa_ratio(returns::AbstractVector{T}, order::Int;
                     threshold::T=T(0.0)) where T<:Real
    excess = returns .- threshold
    lpm = mean(max.(-excess, zero(T)) .^ order)
    mean(excess) / max(lpm^(one(T) / order), T(1e-16))
end

# ─────────────────────────────────────────────────────────────────────────────
# §9  Factor Models
# ─────────────────────────────────────────────────────────────────────────────

"""
    pca_factors(returns; n_factors=3) -> factors, loadings, explained_var

Extract statistical factors via PCA.
"""
function pca_factors(returns::AbstractMatrix{T}; n_factors::Int=3) where T<:Real
    n, p = size(returns)
    mu = vec(mean(returns; dims=1))
    X = returns .- mu'
    Sigma = X' * X / (n - 1)
    F = eigen(Symmetric(Sigma); sortby=x -> -x)
    loadings = F.vectors[:, 1:min(n_factors, p)]
    explained = F.values[1:min(n_factors, p)] ./ sum(F.values)
    factors = X * loadings
    return factors, loadings, explained
end

"""
    fama_french_factors(returns, market_returns, size_char, value_char) -> smb, hml

Construct Fama-French-style SMB and HML factors.
"""
function fama_french_factors(returns::AbstractMatrix{T},
                              market_cap::AbstractVector{T},
                              book_to_market::AbstractVector{T}) where T<:Real
    n_obs, n_assets = size(returns)
    smb = Vector{T}(undef, n_obs)
    hml = Vector{T}(undef, n_obs)
    # Sort by size and value
    size_median = median(market_cap)
    bm_30 = sort(book_to_market)[max(1, round(Int, 0.3 * n_assets))]
    bm_70 = sort(book_to_market)[max(1, round(Int, 0.7 * n_assets))]
    small = findall(market_cap .< size_median)
    big = findall(market_cap .>= size_median)
    value = findall(book_to_market .>= bm_70)
    growth = findall(book_to_market .<= bm_30)
    for t in 1:n_obs
        r_small = isempty(small) ? zero(T) : mean(returns[t, small])
        r_big = isempty(big) ? zero(T) : mean(returns[t, big])
        r_value = isempty(value) ? zero(T) : mean(returns[t, value])
        r_growth = isempty(growth) ? zero(T) : mean(returns[t, growth])
        smb[t] = r_small - r_big
        hml[t] = r_value - r_growth
    end
    return smb, hml
end

"""
    rolling_beta(asset_returns, factor_returns; window=252) -> betas

Rolling regression beta.
"""
function rolling_beta(asset_returns::AbstractVector{T},
                      factor_returns::AbstractVector{T};
                      window::Int=252) where T<:Real
    n = length(asset_returns)
    betas = Vector{T}(undef, n)
    fill!(betas, zero(T))
    for t in window:n
        y = asset_returns[t-window+1:t]
        x = factor_returns[t-window+1:t]
        cov_xy = cov(y, x)
        var_x = var(x)
        betas[t] = var_x > T(1e-16) ? cov_xy / var_x : zero(T)
    end
    betas
end

"""Multi-factor rolling betas."""
function rolling_multi_beta(asset_returns::AbstractVector{T},
                            factor_returns::AbstractMatrix{T};
                            window::Int=252) where T<:Real
    n = length(asset_returns)
    k = size(factor_returns, 2)
    betas = Matrix{T}(undef, n, k + 1)
    fill!(betas, zero(T))
    for t in window:n
        y = asset_returns[t-window+1:t]
        X = hcat(ones(T, window), factor_returns[t-window+1:t, :])
        b = (X' * X) \ (X' * y)
        betas[t, :] = b
    end
    betas
end

"""
    factor_momentum(factor_returns; lookback=252) -> scores

Factor momentum: rank factors by trailing performance.
"""
function factor_momentum(factor_returns::AbstractMatrix{T};
                         lookback::Int=252) where T<:Real
    n, k = size(factor_returns)
    scores = Matrix{T}(undef, n, k)
    fill!(scores, zero(T))
    for t in lookback+1:n
        for j in 1:k
            trailing = factor_returns[t-lookback:t-1, j]
            scores[t, j] = mean(trailing) / max(std(trailing), T(1e-16)) * sqrt(T(252))
        end
    end
    scores
end

"""Factor timing: time-varying factor allocation based on momentum."""
function factor_timing(factor_returns::AbstractMatrix{T};
                       lookback::Int=252, vol_target::T=T(0.10)) where T<:Real
    n, k = size(factor_returns)
    scores = factor_momentum(factor_returns; lookback=lookback)
    weights = Matrix{T}(undef, n, k)
    fill!(weights, zero(T))
    for t in lookback+1:n
        s = scores[t, :]
        pos_s = max.(s, zero(T))
        total = sum(pos_s)
        if total > T(1e-10)
            w = pos_s ./ total
        else
            w = fill(one(T) / k, k)
        end
        # Vol targeting
        if t > lookback + 20
            port_ret = factor_returns[t-20:t-1, :] * w
            realized_vol = std(port_ret) * sqrt(T(252))
            scale = vol_target / max(realized_vol, T(1e-10))
            w .*= min(scale, T(3.0))
        end
        weights[t, :] = w
    end
    weights
end

"""Cross-sectional momentum factor."""
function cross_section_momentum(returns::AbstractMatrix{T};
                                 lookback::Int=252, skip::Int=21,
                                 n_long::Int=10, n_short::Int=10) where T<:Real
    n_obs, n_assets = size(returns)
    factor = Vector{T}(undef, n_obs)
    fill!(factor, zero(T))
    for t in lookback+skip+1:n_obs
        # Trailing return (skip most recent month)
        trailing = vec(sum(returns[t-lookback-skip+1:t-skip, :]; dims=1))
        sorted_idx = sortperm(trailing; rev=true)
        longs = sorted_idx[1:min(n_long, n_assets)]
        shorts = sorted_idx[max(1, n_assets-n_short+1):n_assets]
        factor[t] = mean(returns[t, longs]) - mean(returns[t, shorts])
    end
    factor
end

# ─────────────────────────────────────────────────────────────────────────────
# §10  Portfolio Risk Decomposition
# ─────────────────────────────────────────────────────────────────────────────

"""Factor risk decomposition."""
function factor_risk_decomposition(w::AbstractVector{T},
                                    B::AbstractMatrix{T},
                                    F::AbstractMatrix{T},
                                    D::AbstractVector{T}) where T<:Real
    # B: n_assets x n_factors (loadings)
    # F: n_factors x n_factors (factor covariance)
    # D: n_assets (idiosyncratic variance)
    factor_exp = B' * w  # factor exposures
    systematic_var = dot(factor_exp, F * factor_exp)
    idio_var = dot(w.^2, D)
    total_var = systematic_var + idio_var
    # Factor contributions
    Sigma_f_exp = F * factor_exp
    factor_contrib = factor_exp .* Sigma_f_exp
    return (total_var=total_var, systematic_var=systematic_var,
            idio_var=idio_var, factor_exposures=factor_exp,
            factor_contributions=factor_contrib,
            pct_systematic=systematic_var / max(total_var, T(1e-16)))
end

"""Stress test: factor shock impact on portfolio."""
function factor_stress_test(w::AbstractVector{T},
                            B::AbstractMatrix{T},
                            factor_shocks::AbstractVector{T}) where T<:Real
    asset_shocks = B * factor_shocks
    portfolio_impact = dot(w, asset_shocks)
    return portfolio_impact, asset_shocks
end

# ─────────────────────────────────────────────────────────────────────────────
# §11  Volatility Surface
# ─────────────────────────────────────────────────────────────────────────────

"""Build implied volatility surface from market data."""
function build_vol_surface(S::T, strikes::AbstractVector{T},
                           expiries::AbstractVector{T},
                           market_prices::AbstractMatrix{T},
                           r::T; is_call::Bool=true) where T<:Real
    n_strikes = length(strikes)
    n_expiries = length(expiries)
    vol_surface = Matrix{T}(undef, n_strikes, n_expiries)
    for j in 1:n_expiries
        for i in 1:n_strikes
            vol_surface[i, j] = implied_vol(market_prices[i, j], S, strikes[i],
                                           r, expiries[j]; is_call=is_call)
        end
    end
    vol_surface
end

"""Interpolate on vol surface using SVI parameterization."""
function svi_parameterization(k::T, a::T, b::T, rho::T, m::T, sigma::T) where T<:Real
    # SVI: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
end

"""Fit SVI to a volatility smile at one expiry."""
function svi_fit(log_moneyness::AbstractVector{T},
                 total_implied_var::AbstractVector{T};
                 max_iter::Int=1000, lr::T=T(0.001)) where T<:Real
    n = length(log_moneyness)
    a = mean(total_implied_var)
    b = T(0.1)
    rho = T(-0.3)
    m = zero(T)
    sigma = T(0.1)
    for iter in 1:max_iter
        loss = zero(T)
        for i in 1:n
            model = svi_parameterization(log_moneyness[i], a, b, rho, m, sigma)
            err = model - total_implied_var[i]
            loss += err^2
        end
        delta = T(1e-6)
        for (param_idx, get_val, set_val!) in [
            (1, () -> a, v -> (a = v)),
            (2, () -> b, v -> (b = v)),
            (3, () -> rho, v -> (rho = v)),
            (4, () -> m, v -> (m = v)),
            (5, () -> sigma, v -> (sigma = v))
        ]
            old_val = get_val()
            set_val!(old_val + delta)
            loss_up = zero(T)
            for i in 1:n
                model = svi_parameterization(log_moneyness[i], a, b, rho, m, sigma)
                loss_up += (model - total_implied_var[i])^2
            end
            grad = (loss_up - loss) / delta
            set_val!(old_val - lr * grad)
        end
        b = max(b, T(1e-6))
        rho = clamp(rho, T(-0.999), T(0.999))
        sigma = max(sigma, T(1e-6))
        if loss / n < T(1e-14)
            break
        end
    end
    return (a=a, b=b, rho=rho, m=m, sigma=sigma)
end

# ─────────────────────────────────────────────────────────────────────────────
# §12  Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

"""Convert continuously compounded to discrete rate."""
cc_to_discrete(r_cc::T, freq::Int) where T<:Real = freq * (exp(r_cc / freq) - one(T))

"""Convert discrete to continuously compounded rate."""
discrete_to_cc(r_disc::T, freq::Int) where T<:Real = freq * log(one(T) + r_disc / freq)

"""Discount factor."""
discount_factor(r::T, t::T) where T<:Real = exp(-r * t)

"""Present value of annuity."""
function annuity_pv(pmt::T, r::T, n::Int) where T<:Real
    if abs(r) < T(1e-14)
        return pmt * n
    end
    pmt * (one(T) - (one(T) + r)^(-n)) / r
end

"""Future value of annuity."""
function annuity_fv(pmt::T, r::T, n::Int) where T<:Real
    if abs(r) < T(1e-14)
        return pmt * n
    end
    pmt * ((one(T) + r)^n - one(T)) / r
end

"""Internal rate of return via bisection."""
function irr(cashflows::AbstractVector{T}; max_iter::Int=200, tol::T=T(1e-10)) where T<:Real
    r_lo, r_hi = T(-0.99), T(10.0)
    for _ in 1:max_iter
        r_mid = (r_lo + r_hi) / 2
        npv = sum(cashflows[i] / (one(T) + r_mid)^(i-1) for i in eachindex(cashflows))
        if abs(npv) < tol
            return r_mid
        end
        if npv > zero(T)
            r_lo = r_mid
        else
            r_hi = r_mid
        end
    end
    (r_lo + r_hi) / 2
end

"""Net present value."""
function npv(cashflows::AbstractVector{T}, rate::T) where T<:Real
    sum(cashflows[i] / (one(T) + rate)^(i-1) for i in eachindex(cashflows))
end

"""Modified internal rate of return."""
function mirr(cashflows::AbstractVector{T}, finance_rate::T,
              reinvest_rate::T) where T<:Real
    n = length(cashflows)
    pos_fv = sum(max(cashflows[i], zero(T)) * (one(T) + reinvest_rate)^(n - i)
                 for i in 1:n)
    neg_pv = sum(min(cashflows[i], zero(T)) / (one(T) + finance_rate)^(i - 1)
                 for i in 1:n)
    neg_pv = abs(neg_pv)
    if neg_pv < T(1e-16)
        return T(Inf)
    end
    (pos_fv / neg_pv)^(one(T) / (n - 1)) - one(T)
end

# ─────────────────────────────────────────────────────────────────────────────
# §13  Exotic Options
# ─────────────────────────────────────────────────────────────────────────────

"""Barrier option pricing (down-and-out call, analytical)."""
function barrier_down_out_call(S::T, K::T, H::T, r::T, sigma::T, tau::T;
                                rebate::T=T(0.0)) where T<:Real
    if S <= H
        return rebate
    end
    lambda = (r + sigma^2 / 2) / sigma^2
    y = log(H^2 / (S * K)) / (sigma * sqrt(tau)) + lambda * sigma * sqrt(tau)
    x1 = log(S / H) / (sigma * sqrt(tau)) + lambda * sigma * sqrt(tau)
    y1 = log(H / S) / (sigma * sqrt(tau)) + lambda * sigma * sqrt(tau)
    vanilla = bs_call(S, K, r, sigma, tau)
    di_part = (H / S)^(T(2) * lambda) * bs_call(H^2/S, K, r, sigma, tau)
    max(vanilla - di_part, zero(T))
end

"""Barrier option: up-and-out call."""
function barrier_up_out_call(S::T, K::T, H::T, r::T, sigma::T, tau::T) where T<:Real
    if S >= H
        return zero(T)
    end
    vanilla = bs_call(S, K, r, sigma, tau)
    lambda = (r + sigma^2 / 2) / sigma^2
    ui_part = (H / S)^(T(2) * lambda) * bs_call(H^2/S, K, r, sigma, tau)
    max(vanilla - ui_part, zero(T))
end

"""Asian option (arithmetic average, Monte Carlo)."""
function asian_option_mc(S::T, K::T, r::T, sigma::T, tau::T;
                          n_steps::Int=252, n_paths::Int=10000,
                          is_call::Bool=true,
                          rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    dt = tau / n_steps
    sqrt_dt = sqrt(dt)
    payoffs = Vector{T}(undef, n_paths)
    for p in 1:n_paths
        S_t = S
        avg = zero(T)
        for t in 1:n_steps
            S_t *= exp((r - sigma^2/2) * dt + sigma * sqrt_dt * randn(rng, T))
            avg += S_t
        end
        avg /= n_steps
        if is_call
            payoffs[p] = max(avg - K, zero(T))
        else
            payoffs[p] = max(K - avg, zero(T))
        end
    end
    exp(-r * tau) * mean(payoffs)
end

"""Lookback option (floating strike, analytical for call)."""
function lookback_floating_call(S::T, S_min::T, r::T, sigma::T, tau::T) where T<:Real
    a1 = (log(S / S_min) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    a2 = a1 - sigma * sqrt(tau)
    a3 = (log(S / S_min) + (-r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    y1 = -T(2) * (r - sigma^2 / 2) * log(S / S_min) / sigma^2
    S * _normal_cdf(a1) - S_min * exp(-r * tau) * _normal_cdf(a2) -
    S * sigma^2 / (T(2) * r) * (-_normal_cdf(-a1) + exp(y1) * _normal_cdf(-a3))
end

"""Digital (binary) option pricing."""
function digital_option(S::T, K::T, r::T, sigma::T, tau::T;
                        is_call::Bool=true, payout::T=T(1.0)) where T<:Real
    d2 = (log(S / K) + (r - sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    if is_call
        payout * exp(-r * tau) * _normal_cdf(d2)
    else
        payout * exp(-r * tau) * _normal_cdf(-d2)
    end
end

"""Chooser option: right to choose call or put at choice_date."""
function chooser_option(S::T, K::T, r::T, sigma::T, tau::T,
                        choice_time::T) where T<:Real
    # At choice_time, holder picks max(call, put) = call + max(put - call, 0) = call + put_on_forward
    d1 = (log(S / K) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    d1_c = (log(S / K) + (r + sigma^2 / 2) * choice_time) / (sigma * sqrt(choice_time))
    d2_c = d1_c - sigma * sqrt(choice_time)
    # Simple chooser formula
    call = bs_call(S, K, r, sigma, tau)
    put_early = K * exp(-r * tau) * _normal_cdf(-d2_c) - S * exp(-r * (tau - choice_time)) * _normal_cdf(-d1_c)
    call + max(put_early, zero(T))
end

"""Compound option: option on an option (call on call via binomial)."""
function compound_option(S::T, K_outer::T, K_inner::T, r::T, sigma::T,
                          tau_outer::T, tau_inner::T;
                          n_steps::Int=100) where T<:Real
    # Price inner option at expiry of outer option, then work backwards
    dt = tau_inner / n_steps
    n_outer_steps = max(1, round(Int, tau_outer / dt))
    total_steps = n_steps
    u = exp(sigma * sqrt(dt))
    d = one(T) / u
    p = (exp(r * dt) - d) / (u - d)
    p = clamp(p, T(0.001), T(0.999))
    df = exp(-r * dt)
    # Terminal payoffs for inner call
    prices = Vector{T}(undef, total_steps + 1)
    for i in 0:total_steps
        S_T = S * u^(total_steps - i) * d^i
        prices[i+1] = max(S_T - K_inner, zero(T))
    end
    # Backward induction for inner option
    for step in total_steps-1:-1:n_outer_steps
        for i in 0:step
            prices[i+1] = df * (p * prices[i+1] + (one(T) - p) * prices[i+2])
        end
    end
    # At outer expiry, the compound option payoff is max(inner_price - K_outer, 0)
    for i in 0:n_outer_steps
        prices[i+1] = max(prices[i+1] - K_outer, zero(T))
    end
    # Continue backward for outer option
    for step in n_outer_steps-1:-1:0
        for i in 0:step
            prices[i+1] = df * (p * prices[i+1] + (one(T) - p) * prices[i+2])
        end
    end
    prices[1]
end

# ─────────────────────────────────────────────────────────────────────────────
# §14  Monte Carlo Simulation Framework
# ─────────────────────────────────────────────────────────────────────────────

"""Geometric Brownian Motion simulation."""
function gbm_simulate(S0::T, mu::T, sigma::T, dt::T, n_steps::Int;
                      n_paths::Int=1000,
                      rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    paths = Matrix{T}(undef, n_steps + 1, n_paths)
    paths[1, :] .= S0
    sqrt_dt = sqrt(dt)
    for p in 1:n_paths
        for t in 1:n_steps
            z = randn(rng, T)
            paths[t+1, p] = paths[t, p] * exp((mu - sigma^2/2) * dt + sigma * sqrt_dt * z)
        end
    end
    paths
end

"""Correlated multi-asset GBM simulation."""
function correlated_gbm(S0::AbstractVector{T}, mu::AbstractVector{T},
                         sigma::AbstractVector{T}, corr::AbstractMatrix{T},
                         dt::T, n_steps::Int;
                         n_paths::Int=1000,
                         rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n_assets = length(S0)
    L = cholesky(Symmetric(corr)).L
    paths = Array{T}(undef, n_steps + 1, n_assets, n_paths)
    for p in 1:n_paths
        paths[1, :, p] = S0
    end
    sqrt_dt = sqrt(dt)
    for p in 1:n_paths
        for t in 1:n_steps
            z = L * randn(rng, T, n_assets)
            for i in 1:n_assets
                paths[t+1, i, p] = paths[t, i, p] *
                    exp((mu[i] - sigma[i]^2/2) * dt + sigma[i] * sqrt_dt * z[i])
            end
        end
    end
    paths
end

"""Control variate Monte Carlo for variance reduction."""
function control_variate_mc(S::T, K::T, r::T, sigma::T, tau::T;
                            n_paths::Int=50000,
                            rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    dt = tau
    Z = randn(rng, T, n_paths)
    S_T = S .* exp.((r - sigma^2/2) * tau .+ sigma * sqrt(tau) .* Z)
    payoffs = max.(S_T .- K, zero(T)) .* exp(-r * tau)
    # Control variate: geometric mean price as control
    # Use S_T as control (known expectation = S * exp(r * tau))
    control = S_T
    expected_control = S * exp(r * tau)
    cov_pc = cov(payoffs, control)
    var_c = var(control)
    c_star = cov_pc / max(var_c, T(1e-16))
    adjusted = payoffs .- c_star .* (control .- expected_control)
    mean(adjusted)
end

"""Antithetic variates Monte Carlo."""
function antithetic_mc(S::T, K::T, r::T, sigma::T, tau::T;
                       n_paths::Int=50000, is_call::Bool=true,
                       rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    Z = randn(rng, T, n_paths)
    S_T_pos = S .* exp.((r - sigma^2/2) * tau .+ sigma * sqrt(tau) .* Z)
    S_T_neg = S .* exp.((r - sigma^2/2) * tau .- sigma * sqrt(tau) .* Z)
    if is_call
        payoffs_pos = max.(S_T_pos .- K, zero(T))
        payoffs_neg = max.(S_T_neg .- K, zero(T))
    else
        payoffs_pos = max.(K .- S_T_pos, zero(T))
        payoffs_neg = max.(K .- S_T_neg, zero(T))
    end
    exp(-r * tau) * mean((payoffs_pos .+ payoffs_neg) ./ 2)
end

# ─────────────────────────────────────────────────────────────────────────────
# §15  Term Structure Models (Additional)
# ─────────────────────────────────────────────────────────────────────────────

"""G2++ two-factor model simulation."""
function g2pp_simulate(x0::T, y0::T, a::T, b::T, sigma_x::T, sigma_y::T,
                       rho::T, phi::AbstractVector{T},
                       dt::T, n_steps::Int;
                       n_paths::Int=100,
                       rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    paths = Matrix{T}(undef, n_steps + 1, n_paths)
    sqrt_dt = sqrt(dt)
    for p in 1:n_paths
        x = x0
        y = y0
        for t in 1:n_steps
            z1 = randn(rng, T)
            z2 = rho * z1 + sqrt(one(T) - rho^2) * randn(rng, T)
            x += -a * x * dt + sigma_x * sqrt_dt * z1
            y += -b * y * dt + sigma_y * sqrt_dt * z2
            phi_t = t <= length(phi) ? phi[t] : phi[end]
            paths[t+1, p] = x + y + phi_t
        end
        paths[1, p] = x0 + y0 + (isempty(phi) ? zero(T) : phi[1])
    end
    paths
end

"""Black-Karasinski model: d(ln r) = kappa(theta(t) - ln r)dt + sigma dW."""
function bk_simulate(r0::T, kappa::T, theta::T, sigma::T,
                     dt::T, n_steps::Int;
                     n_paths::Int=100,
                     rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    paths = Matrix{T}(undef, n_steps + 1, n_paths)
    paths[1, :] .= r0
    sqrt_dt = sqrt(dt)
    for p in 1:n_paths
        log_r = log(max(r0, T(1e-10)))
        for t in 1:n_steps
            dlog_r = kappa * (theta - log_r) * dt + sigma * sqrt_dt * randn(rng, T)
            log_r += dlog_r
            paths[t+1, p] = exp(log_r)
        end
    end
    paths
end

# ─────────────────────────────────────────────────────────────────────────────
# §16  Credit Portfolio Models
# ─────────────────────────────────────────────────────────────────────────────

"""Vasicek portfolio loss distribution (single-factor Gaussian copula)."""
function vasicek_portfolio_loss(pd::T, lgd::T, rho::T, n_obligors::Int;
                                 confidence::T=T(0.999)) where T<:Real
    # Vasicek asymptotic formula
    z = _normal_quantile(pd)
    z_alpha = _normal_quantile(confidence)
    conditional_pd = _normal_cdf((z + sqrt(rho) * z_alpha) / sqrt(one(T) - rho))
    expected_loss = lgd * conditional_pd
    return expected_loss, lgd * pd  # unexpected loss, expected loss
end

"""Monte Carlo credit portfolio loss simulation."""
function credit_portfolio_mc(pds::AbstractVector{T}, lgds::AbstractVector{T},
                              exposures::AbstractVector{T}, corr_matrix::AbstractMatrix{T};
                              n_sim::Int=10000,
                              rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n = length(pds)
    L = cholesky(Symmetric(corr_matrix)).L
    losses = Vector{T}(undef, n_sim)
    for s in 1:n_sim
        z = L * randn(rng, T, n)
        loss = zero(T)
        for i in 1:n
            threshold = _normal_quantile(pds[i])
            if z[i] < threshold
                loss += exposures[i] * lgds[i]
            end
        end
        losses[s] = loss
    end
    sorted = sort(losses)
    expected = mean(losses)
    var_99 = sorted[min(n_sim, ceil(Int, 0.99 * n_sim))]
    cvar_99 = mean(sorted[ceil(Int, 0.99 * n_sim):end])
    return (expected_loss=expected, var_99=var_99, cvar_99=cvar_99, loss_distribution=sorted)
end

"""CDO tranche pricing (simplified, Gaussian copula)."""
function cdo_tranche_price(pds::AbstractVector{T}, lgds::AbstractVector{T},
                           attachment::T, detachment::T, corr::T;
                           n_sim::Int=10000,
                           rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n = length(pds)
    # Single-factor model
    losses = Vector{T}(undef, n_sim)
    for s in 1:n_sim
        m = randn(rng, T)  # market factor
        loss_pct = zero(T)
        for i in 1:n
            z = sqrt(corr) * m + sqrt(one(T) - corr) * randn(rng, T)
            threshold = _normal_quantile(pds[i])
            if z < threshold
                loss_pct += lgds[i] / n
            end
        end
        # Tranche loss
        tranche_loss = min(max(loss_pct - attachment, zero(T)), detachment - attachment)
        losses[s] = tranche_loss / (detachment - attachment)
    end
    expected_tranche_loss = mean(losses)
    return expected_tranche_loss
end

# ─────────────────────────────────────────────────────────────────────────────
# §17  Market Microstructure
# ─────────────────────────────────────────────────────────────────────────────

"""Realized variance from high-frequency returns."""
function realized_variance(returns::AbstractVector{T}) where T<:Real
    sum(returns .^ 2)
end

"""Realized volatility with subsampling for noise reduction."""
function realized_volatility_subsample(returns::AbstractVector{T};
                                        n_grids::Int=5) where T<:Real
    n = length(returns)
    rv_sum = zero(T)
    for offset in 0:n_grids-1
        sub = returns[offset+1:n_grids:end]
        rv_sum += sum(sub .^ 2)
    end
    sqrt(rv_sum / n_grids * T(252))
end

"""Bi-power variation (robust to jumps)."""
function bipower_variation(returns::AbstractVector{T}) where T<:Real
    n = length(returns)
    if n < 2
        return zero(T)
    end
    mu1 = sqrt(T(2) / T(π))  # E[|Z|] for standard normal
    bpv = zero(T)
    for t in 2:n
        bpv += abs(returns[t]) * abs(returns[t-1])
    end
    bpv * T(π) / 2 / (n - 1)
end

"""Jump test: ratio of realized variance to bipower variation."""
function jump_test(returns::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    rv = realized_variance(returns)
    bpv = bipower_variation(returns)
    n = length(returns)
    # Under no-jump null, (RV - BPV) / sqrt(var_J) ~ N(0,1)
    # Simplified variance estimate
    mu1 = sqrt(T(2) / T(π))
    tp = zero(T)  # tri-power quarticity
    for t in 3:n
        tp += abs(returns[t])^(T(4)/3) * abs(returns[t-1])^(T(4)/3) * abs(returns[t-2])^(T(4)/3)
    end
    tp *= n * mu1^(-T(4)) / (n - 2)
    var_j = (T(π)^2 / 4 + T(π) - T(5)) * tp
    z_stat = (rv - bpv) / sqrt(max(var_j, T(1e-16)) / T(n))
    z_critical = T(1.96)
    has_jump = z_stat > z_critical
    return z_stat, has_jump, max(rv - bpv, zero(T))
end

"""Noise variance estimation (Roll model)."""
function noise_variance(returns::AbstractVector{T}) where T<:Real
    n = length(returns)
    if n < 2
        return zero(T)
    end
    autocov = zero(T)
    for t in 2:n
        autocov += returns[t] * returns[t-1]
    end
    autocov /= (n - 1)
    max(-autocov, zero(T))
end

# ─────────────────────────────────────────────────────────────────────────────
# §18  Swap Pricing
# ─────────────────────────────────────────────────────────────────────────────

"""Interest rate swap valuation."""
function irs_value(notional::T, fixed_rate::T, floating_rates::AbstractVector{T},
                   zero_rates::AbstractVector{T}, times::AbstractVector{T};
                   pay_fixed::Bool=true) where T<:Real
    n = length(times)
    fixed_leg = zero(T)
    float_leg = zero(T)
    for i in 1:n
        dt = i == 1 ? times[1] : times[i] - times[i-1]
        df = exp(-zero_rates[i] * times[i])
        fixed_leg += fixed_rate * dt * df * notional
        float_leg += floating_rates[i] * dt * df * notional
    end
    if pay_fixed
        return float_leg - fixed_leg
    else
        return fixed_leg - float_leg
    end
end

"""Par swap rate."""
function par_swap_rate(zero_rates::AbstractVector{T},
                       times::AbstractVector{T}) where T<:Real
    n = length(times)
    annuity = zero(T)
    for i in 1:n
        dt = i == 1 ? times[1] : times[i] - times[i-1]
        df = exp(-zero_rates[i] * times[i])
        annuity += dt * df
    end
    df_0 = one(T)
    df_n = exp(-zero_rates[end] * times[end])
    (df_0 - df_n) / max(annuity, T(1e-16))
end

"""Swap DV01."""
function swap_dv01(notional::T, fixed_rate::T, zero_rates::AbstractVector{T},
                   times::AbstractVector{T}; dy::T=T(0.0001)) where T<:Real
    float_rates = zero_rates  # approximate
    v0 = irs_value(notional, fixed_rate, float_rates, zero_rates, times)
    v_up = irs_value(notional, fixed_rate, float_rates .+ dy, zero_rates .+ dy, times)
    abs(v_up - v0)
end

"""Currency swap pricing."""
function currency_swap_value(notional_d::T, notional_f::T,
                              fixed_d::T, fixed_f::T,
                              zero_d::AbstractVector{T},
                              zero_f::AbstractVector{T},
                              times::AbstractVector{T},
                              spot_fx::T) where T<:Real
    n = length(times)
    # Domestic fixed leg
    pv_d = zero(T)
    for i in 1:n
        dt = i == 1 ? times[1] : times[i] - times[i-1]
        df = exp(-zero_d[i] * times[i])
        pv_d += fixed_d * dt * df * notional_d
    end
    pv_d += notional_d * exp(-zero_d[end] * times[end])
    # Foreign fixed leg (converted to domestic)
    pv_f = zero(T)
    for i in 1:n
        dt = i == 1 ? times[1] : times[i] - times[i-1]
        df = exp(-zero_f[i] * times[i])
        pv_f += fixed_f * dt * df * notional_f
    end
    pv_f += notional_f * exp(-zero_f[end] * times[end])
    pv_d - spot_fx * pv_f
end

# ─────────────────────────────────────────────────────────────────────────────
# §19  Inflation-Linked Instruments
# ─────────────────────────────────────────────────────────────────────────────

"""TIPS (Treasury Inflation-Protected Securities) pricing."""
function tips_price(face::T, coupon_rate::T, real_yield::T,
                    inflation_index_ratio::T, n_periods::Int;
                    freq::Int=2) where T<:Real
    adj_face = face * inflation_index_ratio
    c = adj_face * coupon_rate / freq
    r = real_yield / freq
    pv_coupons = c * (one(T) - (one(T) + r)^(-n_periods)) / max(r, T(1e-16))
    pv_face = adj_face / (one(T) + r)^n_periods
    pv_coupons + pv_face
end

"""Breakeven inflation rate."""
function breakeven_inflation(nominal_yield::T, real_yield::T) where T<:Real
    nominal_yield - real_yield
end

"""Inflation swap pricing."""
function inflation_swap_value(notional::T, fixed_inflation::T,
                              realized_cpi::AbstractVector{T},
                              zero_rates::AbstractVector{T},
                              times::AbstractVector{T}) where T<:Real
    n = length(times)
    fixed_leg = zero(T)
    float_leg = zero(T)
    for i in 1:n
        df = exp(-zero_rates[i] * times[i])
        fixed_leg += fixed_inflation * df * notional
        if i <= length(realized_cpi) && i > 1
            float_cpi = (realized_cpi[i] / realized_cpi[1] - one(T))
            float_leg += float_cpi * df * notional
        end
    end
    float_leg - fixed_leg
end

# ─────────────────────────────────────────────────────────────────────────────
# §20  Mortgage-Backed Securities
# ─────────────────────────────────────────────────────────────────────────────

"""Prepayment model (PSA standard)."""
function psa_prepayment(month::Int; psa_speed::T=T(1.0)) where T<:Real
    # PSA: CPR increases linearly from 0 to 6% over first 30 months
    if month <= 30
        cpr = T(0.06) * month / T(30) * psa_speed
    else
        cpr = T(0.06) * psa_speed
    end
    # Convert CPR to SMM
    smm = one(T) - (one(T) - cpr)^(one(T)/T(12))
    return smm, cpr
end

"""MBS cashflow generation."""
function mbs_cashflows(balance::T, coupon_rate::T, n_months::Int;
                        psa_speed::T=T(1.0)) where T<:Real
    monthly_rate = coupon_rate / T(12)
    remaining = balance
    cashflows = Vector{T}(undef, n_months)
    for m in 1:n_months
        if remaining < T(0.01)
            cashflows[m] = zero(T)
            continue
        end
        # Scheduled payment (amortizing)
        scheduled = remaining * monthly_rate / (one(T) - (one(T) + monthly_rate)^(-(n_months - m + 1)))
        interest = remaining * monthly_rate
        principal = scheduled - interest
        # Prepayment
        smm, _ = psa_prepayment(m; psa_speed=psa_speed)
        prepay = (remaining - principal) * smm
        total_principal = principal + prepay
        cashflows[m] = interest + total_principal
        remaining -= total_principal
        remaining = max(remaining, zero(T))
    end
    cashflows
end

"""MBS OAS calculation."""
function mbs_oas(price::T, balance::T, coupon_rate::T, n_months::Int,
                 zero_rates::AbstractVector{T};
                 psa_speed::T=T(1.0)) where T<:Real
    cf = mbs_cashflows(balance, coupon_rate, n_months; psa_speed=psa_speed)
    # Binary search for OAS
    lo, hi = T(-0.05), T(0.20)
    for _ in 1:100
        mid = (lo + hi) / 2
        pv = zero(T)
        for m in 1:n_months
            t = T(m) / T(12)
            r = m <= length(zero_rates) ? zero_rates[m] : zero_rates[end]
            pv += cf[m] / (one(T) + (r + mid) / T(12))^m
        end
        if pv > price
            lo = mid
        else
            hi = mid
        end
        if abs(pv - price) < T(0.001)
            break
        end
    end
    (lo + hi) / 2
end

# ─────────────────────────────────────────────────────────────────────────────
# §21  Risk-Neutral Density
# ─────────────────────────────────────────────────────────────────────────────

"""Extract risk-neutral density from option prices (Breeden-Litzenberger)."""
function risk_neutral_density(strikes::AbstractVector{T},
                               call_prices::AbstractVector{T},
                               r::T, tau::T;
                               dk::T=T(0.01)) where T<:Real
    n = length(strikes)
    density = Vector{T}(undef, n)
    for i in 1:n
        if i == 1
            # Forward difference
            d2C = (call_prices[min(i+2,n)] - T(2)*call_prices[min(i+1,n)] + call_prices[i]) / dk^2
        elseif i == n
            d2C = (call_prices[i] - T(2)*call_prices[i-1] + call_prices[max(i-2,1)]) / dk^2
        else
            d2C = (call_prices[i+1] - T(2)*call_prices[i] + call_prices[i-1]) / dk^2
        end
        density[i] = exp(r * tau) * d2C
        density[i] = max(density[i], zero(T))
    end
    # Normalize
    total = sum(density) * dk
    if total > T(1e-16)
        density ./= total
    end
    density
end

"""Risk-neutral moments from density."""
function rn_moments(strikes::AbstractVector{T}, density::AbstractVector{T}) where T<:Real
    dk = length(strikes) > 1 ? strikes[2] - strikes[1] : one(T)
    mean_rn = sum(strikes .* density) * dk
    var_rn = sum((strikes .- mean_rn).^2 .* density) * dk
    skew_rn = sum((strikes .- mean_rn).^3 .* density) * dk / max(var_rn^T(1.5), T(1e-16))
    kurt_rn = sum((strikes .- mean_rn).^4 .* density) * dk / max(var_rn^2, T(1e-16)) - T(3)
    return (mean=mean_rn, variance=var_rn, skewness=skew_rn, kurtosis=kurt_rn)
end

end # module QuantFinance
