"""
InterestRates — Fixed income and interest rate derivatives pricing for SRFM research.

Implements:
  - Short rate models: Vasicek, CIR, Hull-White
  - HJM framework: drift condition, Gaussian HJM, simulation
  - LIBOR Market Model (BGM): caplet/swaption pricing
  - Yield curve bootstrap
  - Duration/convexity, DV01, key rate durations, swap valuation
"""
module InterestRates

using LinearAlgebra
using Statistics
using Distributions
using Random
using Optim
using DataFrames

export VasicekModel, vasicek_bond_price, vasicek_bond_option, vasicek_simulate
export CIRModel, cir_bond_price, cir_bond_option, cir_simulate, cir_check_feller
export HullWhiteModel, hw_bond_price, hw_simulate, hw_trinomial_tree
export HJMModel, hjm_drift_condition, hjm_simulate
export LMMModel, lmm_caplet_price, lmm_swaption_price, lmm_simulate
export YieldCurve, bootstrap_yield_curve, zero_to_forward, forward_to_zero
export SwapContract, swap_value, swap_par_rate, swap_dv01
export bond_duration, bond_convexity, dv01, key_rate_durations
export CapFloor, cap_black_price, floor_black_price
export Swaption, swaption_black_price

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Vasicek Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    VasicekModel

Parameters for the Vasicek short rate model:
dr_t = kappa*(theta - r_t)*dt + sigma*dW_t
"""
struct VasicekModel
    kappa::Float64   # mean reversion speed
    theta::Float64   # long-run mean
    sigma::Float64   # volatility
    r0::Float64      # initial rate
end

"""
    vasicek_A(m::VasicekModel, T::Float64) -> Float64

Compute A(T) factor in Vasicek bond price: P(0,T) = A(T)*exp(-B(T)*r0)
"""
function vasicek_A(m::VasicekModel, T::Float64)
    B = vasicek_B(m, T)
    kappa = m.kappa
    theta = m.theta
    sigma = m.sigma
    term = (theta - sigma^2 / (2 * kappa^2)) * (B - T) - sigma^2 * B^2 / (4 * kappa)
    return exp(term)
end

"""
    vasicek_B(m::VasicekModel, T::Float64) -> Float64

Compute B(T) factor: B(T) = (1 - exp(-kappa*T)) / kappa
"""
function vasicek_B(m::VasicekModel, T::Float64)
    if abs(m.kappa) < 1e-10
        return T
    end
    return (1.0 - exp(-m.kappa * T)) / m.kappa
end

"""
    vasicek_bond_price(m::VasicekModel, t::Float64, T::Float64, r::Float64) -> Float64

Compute zero-coupon bond price P(t,T) under Vasicek model.
P(t,T) = A(T-t) * exp(-B(T-t) * r_t)
"""
function vasicek_bond_price(m::VasicekModel, t::Float64, T::Float64, r::Float64)
    tau = T - t
    if tau <= 0
        return 1.0
    end
    B = vasicek_B(m, tau)
    A = vasicek_A(m, tau)
    return A * exp(-B * r)
end

"""
    vasicek_bond_price(m::VasicekModel, T::Float64) -> Float64

Bond price at t=0 with r=r0.
"""
vasicek_bond_price(m::VasicekModel, T::Float64) = vasicek_bond_price(m, 0.0, T, m.r0)

"""
    vasicek_yield(m::VasicekModel, T::Float64) -> Float64

Compute zero rate for maturity T under Vasicek model.
y(T) = -ln(P(0,T)) / T
"""
function vasicek_yield(m::VasicekModel, T::Float64)
    P = vasicek_bond_price(m, T)
    if P <= 0 || T <= 0
        return m.r0
    end
    return -log(P) / T
end

"""
    vasicek_bond_option(m::VasicekModel, T_option::Float64, T_bond::Float64,
                         K::Float64; call=true) -> Float64

Price a European option on a zero-coupon bond under Vasicek model.

# Arguments
- `T_option`: Option expiry
- `T_bond`: Bond maturity
- `K`: Strike price
- `call`: true for call, false for put
"""
function vasicek_bond_option(m::VasicekModel, T_option::Float64, T_bond::Float64,
                              K::Float64; call::Bool=true)
    # Jamshidian (1989) formula
    kappa = m.kappa
    sigma = m.sigma

    B_sT = vasicek_B(m, T_bond - T_option)
    P_bond = vasicek_bond_price(m, T_bond)
    P_option = vasicek_bond_price(m, T_option)

    # Volatility of ln(P(T_option, T_bond))
    sigma_P = sigma * B_sT * sqrt((1.0 - exp(-2.0 * kappa * T_option)) / (2.0 * kappa))

    if sigma_P < 1e-12
        if call
            return max(P_bond - K * P_option, 0.0)
        else
            return max(K * P_option - P_bond, 0.0)
        end
    end

    h = log(P_bond / (P_option * K)) / sigma_P + 0.5 * sigma_P
    d = Normal()

    if call
        return P_bond * cdf(d, h) - K * P_option * cdf(d, h - sigma_P)
    else
        return K * P_option * cdf(d, -(h - sigma_P)) - P_bond * cdf(d, -h)
    end
end

"""
    vasicek_simulate(m::VasicekModel, T::Float64, N::Int, n_paths::Int;
                     rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Simulate Vasicek short rate paths using Euler-Maruyama discretization.
Returns n_paths × (N+1) matrix.
"""
function vasicek_simulate(m::VasicekModel, T::Float64, N::Int, n_paths::Int;
                           rng::AbstractRNG=Random.GLOBAL_RNG)
    dt = T / N
    paths = zeros(n_paths, N + 1)
    paths[:, 1] .= m.r0

    for t in 1:N
        r_prev = paths[:, t]
        dW = randn(rng, n_paths) * sqrt(dt)
        r_new = r_prev .+ m.kappa .* (m.theta .- r_prev) .* dt .+ m.sigma .* dW
        paths[:, t+1] = r_new
    end

    return paths
end

"""
    vasicek_simulate_exact(m::VasicekModel, T::Float64, N::Int, n_paths::Int;
                            rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Exact simulation of Vasicek process (conditional normal distribution).
"""
function vasicek_simulate_exact(m::VasicekModel, T::Float64, N::Int, n_paths::Int;
                                 rng::AbstractRNG=Random.GLOBAL_RNG)
    dt = T / N
    kappa = m.kappa
    theta = m.theta
    sigma = m.sigma

    paths = zeros(n_paths, N + 1)
    paths[:, 1] .= m.r0

    # Exact conditional distribution:
    # r(t+dt) | r(t) ~ N(mu, variance)
    # mu = r(t)*exp(-kappa*dt) + theta*(1 - exp(-kappa*dt))
    # variance = sigma^2/(2*kappa) * (1 - exp(-2*kappa*dt))
    exp_kdt = exp(-kappa * dt)
    cond_mean_const = theta * (1.0 - exp_kdt)
    cond_var = sigma^2 / (2.0 * kappa) * (1.0 - exp(-2.0 * kappa * dt))
    cond_std = sqrt(cond_var)

    for t in 1:N
        r_prev = paths[:, t]
        eps = randn(rng, n_paths)
        paths[:, t+1] = r_prev .* exp_kdt .+ cond_mean_const .+ cond_std .* eps
    end

    return paths
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: CIR Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    CIRModel

Parameters for the Cox-Ingersoll-Ross short rate model:
dr_t = kappa*(theta - r_t)*dt + sigma*sqrt(r_t)*dW_t

Feller condition for positivity: 2*kappa*theta > sigma^2
"""
struct CIRModel
    kappa::Float64
    theta::Float64
    sigma::Float64
    r0::Float64
end

"""
    cir_check_feller(m::CIRModel) -> Bool

Check whether the Feller condition 2*kappa*theta > sigma^2 is satisfied.
"""
function cir_check_feller(m::CIRModel)
    return 2.0 * m.kappa * m.theta > m.sigma^2
end

"""
    cir_h(m::CIRModel) -> Float64

Auxiliary parameter h = sqrt(kappa^2 + 2*sigma^2)
"""
function cir_h(m::CIRModel)
    return sqrt(m.kappa^2 + 2.0 * m.sigma^2)
end

"""
    cir_A(m::CIRModel, tau::Float64) -> Float64

Compute A(tau) for CIR bond price formula.
"""
function cir_A(m::CIRModel, tau::Float64)
    kappa = m.kappa
    theta = m.theta
    sigma = m.sigma
    h = cir_h(m)

    numerator = 2.0 * h * exp((kappa + h) * tau / 2.0)
    denominator = (kappa + h) * (exp(h * tau) - 1.0) + 2.0 * h
    A = (numerator / denominator)^(2.0 * kappa * theta / sigma^2)
    return A
end

"""
    cir_B(m::CIRModel, tau::Float64) -> Float64

Compute B(tau) for CIR bond price formula.
"""
function cir_B(m::CIRModel, tau::Float64)
    kappa = m.kappa
    sigma = m.sigma
    h = cir_h(m)

    numerator = 2.0 * (exp(h * tau) - 1.0)
    denominator = (kappa + h) * (exp(h * tau) - 1.0) + 2.0 * h
    return numerator / denominator
end

"""
    cir_bond_price(m::CIRModel, t::Float64, T::Float64, r::Float64) -> Float64

Compute zero-coupon bond price P(t,T) under CIR model.
P(t,T) = A(T-t) * exp(-B(T-t) * r_t)
"""
function cir_bond_price(m::CIRModel, t::Float64, T::Float64, r::Float64)
    tau = T - t
    if tau <= 0
        return 1.0
    end
    A = cir_A(m, tau)
    B = cir_B(m, tau)
    return A * exp(-B * r)
end

cir_bond_price(m::CIRModel, T::Float64) = cir_bond_price(m, 0.0, T, m.r0)

"""
    cir_bond_option(m::CIRModel, T_option::Float64, T_bond::Float64,
                    K::Float64; call=true) -> Float64

Price a European option on a zero-coupon bond under CIR model.
Uses the non-central chi-squared distribution (Longstaff 1989 formula).
"""
function cir_bond_option(m::CIRModel, T_option::Float64, T_bond::Float64,
                          K::Float64; call::Bool=true)
    kappa = m.kappa
    theta = m.theta
    sigma = m.sigma
    r0 = m.r0
    h = cir_h(m)

    # Parameters for noncentral chi-sq distribution
    phi = 2.0 * h / (sigma^2 * (exp(h * T_option) - 1.0))
    u = phi * r0 * exp(h * T_option)

    # Degree of freedom
    nu = 4.0 * kappa * theta / sigma^2

    A_sT = cir_A(m, T_bond - T_option)
    B_sT = cir_B(m, T_bond - T_option)
    A_T = cir_A(m, T_bond)
    B_T = cir_B(m, T_bond)

    # Critical short rate r* such that P(T_option, T_bond; r*) = K
    if K <= 0 || A_sT <= 0
        return call ? max(cir_bond_price(m, T_bond) - K * cir_bond_price(m, T_option), 0.0) : 0.0
    end

    r_star = log(A_sT / K) / B_sT

    # Non-centrality parameters
    psi = 2.0 * phi * (phi + B_T)
    psi_s = 2.0 * phi * (phi + B_sT)

    lambda_1 = 2.0 * u * psi / (phi + B_T)
    lambda_2 = 2.0 * u * psi_s / (phi + B_sT)

    P_bond = cir_bond_price(m, T_bond)
    P_option = cir_bond_price(m, T_option)

    # CDF evaluations using noncentral chi-squared
    d1 = NoncentralChisq(nu, lambda_1)
    d2 = NoncentralChisq(nu, lambda_2)

    r_star_scaled_1 = 2.0 * r_star * psi
    r_star_scaled_2 = 2.0 * r_star * psi_s

    if call
        return P_bond * (1.0 - cdf(d1, r_star_scaled_1)) -
               K * P_option * (1.0 - cdf(d2, r_star_scaled_2))
    else
        return K * P_option * cdf(d2, r_star_scaled_2) -
               P_bond * cdf(d1, r_star_scaled_1)
    end
end

"""
    cir_simulate(m::CIRModel, T::Float64, N::Int, n_paths::Int;
                 rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Simulate CIR short rate paths using the Milstein scheme (ensures positivity via reflection).
"""
function cir_simulate(m::CIRModel, T::Float64, N::Int, n_paths::Int;
                       rng::AbstractRNG=Random.GLOBAL_RNG)
    dt = T / N
    kappa = m.kappa
    theta = m.theta
    sigma = m.sigma

    paths = zeros(n_paths, N + 1)
    paths[:, 1] .= m.r0

    for t in 1:N
        r_prev = max.(paths[:, t], 0.0)
        dW = randn(rng, n_paths) * sqrt(dt)
        # Milstein scheme
        r_new = r_prev .+
                kappa .* (theta .- r_prev) .* dt .+
                sigma .* sqrt.(r_prev) .* dW .+
                0.25 .* sigma^2 .* (dW .^ 2 .- dt)
        paths[:, t+1] = max.(r_new, 0.0)
    end

    return paths
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Hull-White Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    HullWhiteModel

Time-dependent Hull-White (extended Vasicek) model:
dr_t = (theta(t) - a*r_t)*dt + sigma*dW_t
where theta(t) is chosen to fit the initial yield curve.
"""
struct HullWhiteModel
    a::Float64          # mean reversion speed
    sigma::Float64      # volatility
    r0::Float64         # initial rate
    # Yield curve for calibration
    maturities::Vector{Float64}
    market_zeros::Vector{Float64}
end

"""
    hw_P_market(m::HullWhiteModel, T::Float64) -> Float64

Interpolate market bond price P(0,T) from bootstrapped zero rates.
"""
function hw_P_market(m::HullWhiteModel, T::Float64)
    if T <= 0
        return 1.0
    end
    # Linear interpolation of zero rates
    mats = m.maturities
    zeros = m.market_zeros

    if T <= mats[1]
        y = zeros[1]
    elseif T >= mats[end]
        y = zeros[end]
    else
        idx = searchsortedfirst(mats, T)
        if idx > length(mats)
            y = zeros[end]
        elseif idx == 1
            y = zeros[1]
        else
            t1, t2 = mats[idx-1], mats[idx]
            y1, y2 = zeros[idx-1], zeros[idx]
            y = y1 + (y2 - y1) * (T - t1) / (t2 - t1)
        end
    end
    return exp(-y * T)
end

"""
    hw_f_market(m::HullWhiteModel, T::Float64) -> Float64

Compute instantaneous forward rate from market zero curve: f(0,T) = -d/dT ln P(0,T)
"""
function hw_f_market(m::HullWhiteModel, T::Float64; eps::Float64=1e-5)
    P_plus = hw_P_market(m, T + eps)
    P_minus = hw_P_market(m, max(T - eps, 1e-8))
    return -(log(P_plus) - log(P_minus)) / (2.0 * eps)
end

"""
    hw_bond_price(m::HullWhiteModel, t::Float64, T::Float64, r_t::Float64) -> Float64

Compute P(t,T) under Hull-White model using the analytical formula.
P(t,T) = A(t,T) * exp(-B(t,T) * r_t)

where B(t,T) = (1 - exp(-a*(T-t))) / a
and ln A(t,T) = ln P(0,T)/P(0,t) + B(t,T)*f(0,t) - sigma^2/(4a) * (1-exp(-2at)) * B(t,T)^2
"""
function hw_bond_price(m::HullWhiteModel, t::Float64, T::Float64, r_t::Float64)
    tau = T - t
    if tau <= 0
        return 1.0
    end
    a = m.a
    sigma = m.sigma

    B = (1.0 - exp(-a * tau)) / a
    P_0T = hw_P_market(m, T)
    P_0t = hw_P_market(m, t)
    f_0t = hw_f_market(m, t)

    if P_0t <= 0
        return 1.0
    end

    lnA = log(P_0T / P_0t) + B * f_0t - sigma^2 / (4.0 * a) * (1.0 - exp(-2.0 * a * t)) * B^2
    A = exp(lnA)
    return A * exp(-B * r_t)
end

hw_bond_price(m::HullWhiteModel, T::Float64) = hw_bond_price(m, 0.0, T, m.r0)

"""
    hw_caplet_price(m::HullWhiteModel, t_start::Float64, t_end::Float64,
                    K::Float64) -> Float64

Price a caplet under Hull-White model (is equivalent to put on bond).
Caplet pays max(L(t_start, t_end) - K, 0) at t_end.
"""
function hw_caplet_price(m::HullWhiteModel, t_start::Float64, t_end::Float64, K::Float64)
    alpha = t_end - t_start
    K_bond = 1.0 / (1.0 + K * alpha)
    # Caplet = (1 + K*alpha) * put on bond P(t_start, t_end) with strike K_bond
    a = m.a
    sigma = m.sigma

    sigma_P = sigma / a * (1.0 - exp(-a * alpha)) * sqrt((1.0 - exp(-2.0 * a * t_start)) / (2.0 * a))

    P_bond = hw_bond_price(m, t_end)
    P_option = hw_bond_price(m, t_start)

    if sigma_P < 1e-12
        return max((1.0 / K_bond) * P_option - P_bond, 0.0)
    end

    h = log(P_bond / (P_option * K_bond)) / sigma_P + 0.5 * sigma_P
    d = Normal()
    # Put on bond = Floorlet
    put_value = K_bond * P_option * cdf(d, -(h - sigma_P)) - P_bond * cdf(d, -h)
    return (1.0 + K * alpha) * put_value
end

"""
    hw_simulate(m::HullWhiteModel, T::Float64, N::Int, n_paths::Int;
                rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Simulate Hull-White short rate paths (exact conditional normal).
"""
function hw_simulate(m::HullWhiteModel, T::Float64, N::Int, n_paths::Int;
                      rng::AbstractRNG=Random.GLOBAL_RNG)
    dt = T / N
    a = m.a
    sigma = m.sigma

    paths = zeros(n_paths, N + 1)
    paths[:, 1] .= m.r0

    exp_adt = exp(-a * dt)
    cond_var = sigma^2 / (2.0 * a) * (1.0 - exp(-2.0 * a * dt))
    cond_std = sqrt(cond_var)

    for t in 1:N
        t_now = (t - 1) * dt
        t_next = t * dt
        # theta(t) from market: d/dT f(0,t) + a*f(0,t) + sigma^2/(2a)*(1-exp(-2at))
        f_now = hw_f_market(m, t_now)
        f_next = hw_f_market(m, t_next)
        df_dt = (f_next - f_now) / dt
        theta_t = df_dt + a * f_now + sigma^2 / (2.0 * a) * (1.0 - exp(-2.0 * a * t_now))

        r_prev = paths[:, t]
        eps = randn(rng, n_paths)
        cond_mean = r_prev .* exp_adt .+ (theta_t / a) .* (1.0 - exp_adt)
        paths[:, t+1] = cond_mean .+ cond_std .* eps
    end

    return paths
end

"""
    hw_trinomial_tree(m::HullWhiteModel, T::Float64, N::Int) -> NamedTuple

Build Hull-White trinomial tree for option pricing.
Uses the standard trinomial construction (Hull-White 1994).

Returns: (r_grid, probs, discount_factors, dt, dr)
"""
function hw_trinomial_tree(m::HullWhiteModel, T::Float64, N::Int)
    dt = T / N
    a = m.a
    sigma = m.sigma

    # Spacing: dr = sigma * sqrt(3*dt)
    dr = sigma * sqrt(3.0 * dt)

    # Maximum number of nodes at each level
    j_max = ceil(Int, 0.184 / (a * dt))  # Hull-White branching criterion

    n_nodes = 2 * j_max + 1

    # Arrays for alpha (drift adjustment), node rates, transition probs
    alpha = zeros(N + 1)  # tree drift adjustment
    r_grid = zeros(N + 1, n_nodes)  # r_{t,j} = alpha_t + j*dr

    # Transition probabilities: pu, pm, pd for each node
    pu = zeros(N, n_nodes)
    pm = zeros(N, n_nodes)
    pd = zeros(N, n_nodes)

    # Initialize
    alpha[1] = m.r0

    # Forward induction to find alpha_t matching market bond prices
    # Arrow-Debreu prices Q[t,j]: present value of $1 if node (t,j) is reached
    Q = zeros(N + 1, n_nodes)
    j0 = j_max + 1  # index of center node
    Q[1, j0] = 1.0  # start at center node

    for t in 1:N
        # Compute alpha_t+1 from market price P(0, (t+1)*dt)
        P_market = hw_P_market(m, (t + 1) * dt)
        if P_market <= 0
            P_market = 1e-10
        end

        # Compute transition probs for level t
        for j in (-j_max):j_max
            ji = j + j0
            if Q[t, ji] == 0
                continue
            end
            r_j = alpha[t] + j * dr
            # Mean reversion: mu = -a*j*dr (relative to alpha)
            mu = -a * j * dr * dt  # expected change in x = r - alpha

            # Branching factor
            # Standard HW: j* = round(mu/dr)
            j_star = round(Int, mu / dr)
            j_star = clamp(j_star + j, -(j_max), j_max) - j  # relative

            # Probabilities (trinomial)
            eta_sq = sigma^2 * dt / dr^2
            pu[t, ji] = 0.5 * eta_sq + 0.5 * (mu / dr)^2 + 0.5 * (mu / dr)
            pm[t, ji] = 1.0 - eta_sq - (mu / dr)^2
            pd[t, ji] = 0.5 * eta_sq + 0.5 * (mu / dr)^2 - 0.5 * (mu / dr)

            # Clamp probabilities to valid range
            pu[t, ji] = clamp(pu[t, ji], 0.0, 1.0)
            pd[t, ji] = clamp(pd[t, ji], 0.0, 1.0)
            pm[t, ji] = 1.0 - pu[t, ji] - pd[t, ji]
            pm[t, ji] = clamp(pm[t, ji], 0.0, 1.0)
        end

        # Compute alpha_{t+1} to match P(0, (t+1)*dt)
        # sum_j Q[t,j] * exp(-r_j * dt) = P(0, (t+1)*dt)
        sum_Q = 0.0
        for j in (-j_max):j_max
            ji = j + j0
            r_j = j * dr  # without alpha
            sum_Q += Q[t, ji] * exp(-r_j * dt)
        end
        if sum_Q > 0
            alpha[t+1] = -log(P_market / sum_Q) / dt - alpha[t]
        end

        # Update Arrow-Debreu prices for next level
        for j in (-j_max):j_max
            ji_next = j + j0
            Q[t+1, ji_next] = 0.0
        end
        for j in (-j_max):j_max
            ji = j + j0
            if Q[t, ji] == 0
                continue
            end
            r_j = alpha[t] + j * dr
            disc = exp(-r_j * dt)

            if j + 1 <= j_max
                Q[t+1, j+1+j0] += Q[t, ji] * pu[t, ji] * disc
            end
            Q[t+1, j+j0] += Q[t, ji] * pm[t, ji] * disc
            if j - 1 >= -j_max
                Q[t+1, j-1+j0] += Q[t, ji] * pd[t, ji] * disc
            end
        end

        # Set r_grid for level t+1
        for j in (-j_max):j_max
            ji = j + j0
            r_grid[t+1, ji] = alpha[t+1] + j * dr
        end
    end

    r_grid[1, j0] = m.r0

    return (
        r_grid=r_grid,
        Q=Q,
        pu=pu, pm=pm, pd=pd,
        alpha=alpha,
        dt=dt, dr=dr,
        j_max=j_max,
        N=N
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: HJM Framework
# ─────────────────────────────────────────────────────────────────────────────

"""
    HJMModel

Heath-Jarrow-Morton framework parameters.
Supports multi-factor Gaussian HJM.
"""
struct HJMModel
    n_factors::Int
    sigma_fns::Vector{Function}  # sigma_i(t, T) for each factor
    f0::Function                 # initial forward curve f(0,T)
    T_max::Float64
end

"""
    hjm_drift_condition(m::HJMModel, t::Float64, T::Float64) -> Float64

Compute the HJM no-arbitrage drift condition:
mu(t,T) = sum_i sigma_i(t,T) * integral_t^T sigma_i(t,s) ds
"""
function hjm_drift_condition(m::HJMModel, t::Float64, T::Float64;
                               n_quad::Int=50)
    total_drift = 0.0
    s_grid = range(t, T, length=n_quad)
    ds = (T - t) / (n_quad - 1)

    for i in 1:m.n_factors
        sig_iT = m.sigma_fns[i](t, T)
        # Numerical integral of sigma_i(t,s) from t to T
        integral = 0.0
        for s in s_grid
            integral += m.sigma_fns[i](t, s) * ds
        end
        total_drift += sig_iT * integral
    end

    return total_drift
end

"""
    hjm_simulate(m::HJMModel, T_sim::Float64, N_time::Int, N_maturities::Int,
                 n_paths::Int; rng=Random.GLOBAL_RNG) -> NamedTuple

Simulate forward rate curves under HJM model on a discrete grid.
Returns forward curves f(t, T) for each simulated path.
"""
function hjm_simulate(m::HJMModel, T_sim::Float64, N_time::Int, N_maturities::Int,
                       n_paths::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    dt = T_sim / N_time
    T_max = m.T_max
    dT = T_max / N_maturities

    # Maturity grid
    maturities = [j * dT for j in 0:N_maturities]

    # Initialize forward curves
    # f[path, time, maturity_idx]
    f = zeros(n_paths, N_time + 1, N_maturities + 1)

    # Initial condition f(0, T) from m.f0
    for j in 0:N_maturities
        T = j * dT
        f0_val = m.f0(T)
        for p in 1:n_paths
            f[p, 1, j+1] = f0_val
        end
    end

    # Simulate
    for t_idx in 1:N_time
        t = (t_idx - 1) * dt
        # Brownian increments for each factor
        dW = randn(rng, n_paths, m.n_factors) .* sqrt(dt)

        for j in 0:N_maturities
            T_mat = j * dT
            if T_mat < t
                # Past maturity — set to short rate
                for p in 1:n_paths
                    f[p, t_idx+1, j+1] = f[p, t_idx, 1]  # use short rate
                end
                continue
            end

            # Compute drift (HJM condition)
            drift = hjm_drift_condition(m, t, T_mat)

            # Diffusion terms
            diffusion = zeros(n_paths)
            for i in 1:m.n_factors
                sig = m.sigma_fns[i](t, T_mat)
                diffusion .+= sig .* dW[:, i]
            end

            for p in 1:n_paths
                f[p, t_idx+1, j+1] = f[p, t_idx, j+1] + drift * dt + diffusion[p]
            end
        end
    end

    # Short rate path: r_t = f(t, t)
    r_paths = zeros(n_paths, N_time + 1)
    for t_idx in 1:(N_time + 1)
        t = (t_idx - 1) * dt
        j = clamp(round(Int, t / dT), 0, N_maturities)
        r_paths[:, t_idx] = f[:, t_idx, j+1]
    end

    return (
        f=f,
        r_paths=r_paths,
        maturities=maturities,
        times=[i * dt for i in 0:N_time]
    )
end

"""
    gaussian_hjm_model(a::Float64, sigma::Float64, f0::Function, T_max::Float64) -> HJMModel

Create single-factor Gaussian HJM equivalent to Hull-White.
sigma(t,T) = sigma * exp(-a*(T-t))
"""
function gaussian_hjm_model(a::Float64, sigma_val::Float64, f0::Function, T_max::Float64)
    sigma_fn = (t, T) -> sigma_val * exp(-a * (T - t))
    return HJMModel(1, [sigma_fn], f0, T_max)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: LIBOR Market Model (BGM)
# ─────────────────────────────────────────────────────────────────────────────

"""
    LMMModel

LIBOR Market Model (Brace-Gatarek-Musiela) parameters.

# Fields
- `T_dates::Vector{Float64}`: Tenor structure [T_0, T_1, ..., T_n]
- `L0::Vector{Float64}`: Initial LIBOR rates L(0; T_i, T_{i+1})
- `sigma::Vector{Float64}`: LIBOR volatilities per tenor
- `rho::Matrix{Float64}`: Correlation matrix between LIBORs
"""
struct LMMModel
    T_dates::Vector{Float64}
    L0::Vector{Float64}
    sigma::Vector{Float64}
    rho::Matrix{Float64}

    function LMMModel(T_dates, L0, sigma, rho)
        n = length(L0)
        @assert length(T_dates) == n + 1
        @assert length(sigma) == n
        @assert size(rho) == (n, n)
        new(T_dates, L0, sigma, rho)
    end
end

"""
    lmm_caplet_price(m::LMMModel, i::Int, K::Float64) -> Float64

Price caplet on L(T_{i-1}, T_i) using Black's formula.
Payoff = delta_i * max(L(T_{i-1}, T_i) - K, 0) at T_i.
"""
function lmm_caplet_price(m::LMMModel, i::Int, K::Float64)
    @assert 1 <= i <= length(m.L0)
    T_start = m.T_dates[i]
    T_end = m.T_dates[i+1]
    delta = T_end - T_start

    L = m.L0[i]
    sig = m.sigma[i]
    T = T_start

    if T <= 0 || sig <= 0
        return delta * max(L - K, 0.0)
    end

    # Discount factor P(0, T_end) from LIBOR strip
    P_Tend = lmm_discount(m, T_end)

    d1 = (log(L / K) + 0.5 * sig^2 * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    d = Normal()

    caplet = P_Tend * delta * (L * cdf(d, d1) - K * cdf(d, d2))
    return caplet
end

"""
    lmm_discount(m::LMMModel, T::Float64) -> Float64

Compute discount factor P(0,T) from LIBOR strip.
"""
function lmm_discount(m::LMMModel, T::Float64)
    T_dates = m.T_dates
    L = m.L0

    if T <= T_dates[1]
        return 1.0
    end

    P = 1.0
    for i in eachindex(L)
        T_i = T_dates[i]
        T_i1 = T_dates[i+1]
        delta = T_i1 - T_i
        P /= (1.0 + L[i] * delta)
        if T_i1 >= T
            break
        end
    end
    return P
end

"""
    lmm_cap_price(m::LMMModel, K::Float64) -> Float64

Price a cap (portfolio of caplets) from T_1 to T_n.
"""
function lmm_cap_price(m::LMMModel, K::Float64)
    return sum(lmm_caplet_price(m, i, K) for i in eachindex(m.L0))
end

"""
    lmm_swaption_price(m::LMMModel, T_start_idx::Int, T_end_idx::Int,
                        K_swap::Float64; n_paths=10000, rng=Random.GLOBAL_RNG) -> Float64

Price a European payer swaption via Monte Carlo under LMM.
Swaption pays max(swap_value, 0) at T_start.
"""
function lmm_swaption_price(m::LMMModel, T_start_idx::Int, T_end_idx::Int,
                              K_swap::Float64; n_paths::Int=10000,
                              rng::AbstractRNG=Random.GLOBAL_RNG)
    N = length(m.L0)
    @assert 1 <= T_start_idx < T_end_idx <= N + 1

    T_start = m.T_dates[T_start_idx]
    dt = T_start / max(50, round(Int, T_start * 365))
    n_steps = max(50, round(Int, T_start / dt))
    dt = T_start / n_steps

    # Cholesky decomposition of correlation matrix
    rho_sub = m.rho[T_start_idx:T_end_idx-1, T_start_idx:T_end_idx-1]
    C = cholesky(Symmetric(rho_sub + 1e-8 * I)).L

    n_sub = T_end_idx - T_start_idx  # number of relevant LIBORs
    sigma_sub = m.sigma[T_start_idx:T_end_idx-1]

    payoffs = zeros(n_paths)

    for p in 1:n_paths
        L_sim = copy(m.L0[T_start_idx:T_end_idx-1])

        for step in 1:n_steps
            t = (step - 1) * dt
            z = C * randn(rng, n_sub)

            # LMM drift under spot measure (approximate)
            drift = zeros(n_sub)
            for i in 1:n_sub
                for j in (i+1):n_sub
                    T_j1 = m.T_dates[T_start_idx+j]
                    T_j = m.T_dates[T_start_idx+j-1]
                    delta_j = T_j1 - T_j
                    L_j = L_sim[j]
                    drift[i] += rho_sub[i, j] * sigma_sub[i] * sigma_sub[j] *
                                 delta_j * L_j / (1.0 + delta_j * L_j)
                end
            end

            for i in 1:n_sub
                L_sim[i] *= exp((drift[i] - 0.5 * sigma_sub[i]^2) * dt +
                                sigma_sub[i] * sqrt(dt) * z[i])
            end
        end

        # Compute swap value at T_start
        swap_val = 0.0
        annuity = 0.0
        P_running = 1.0
        for i in 1:n_sub
            T_i = m.T_dates[T_start_idx+i-1]
            T_i1 = m.T_dates[T_start_idx+i]
            delta_i = T_i1 - T_i
            P_running /= (1.0 + L_sim[i] * delta_i)
            annuity += delta_i * P_running
        end
        # Swap value = P(T_start, T_start) - P(T_start, T_end) - K * annuity
        P_T_start = 1.0
        P_T_end = P_running
        swap_val = P_T_start - P_T_end - K_swap * annuity

        payoffs[p] = max(swap_val, 0.0)
    end

    # Discount to today
    P_T_start = lmm_discount(m, T_start)
    return P_T_start * mean(payoffs)
end

"""
    lmm_simulate(m::LMMModel, n_paths::Int; rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Simulate LIBOR rates to expiry of each tenor using terminal measure.
Returns n_paths × n_libors matrix of simulated terminal LIBOR rates.
"""
function lmm_simulate(m::LMMModel, n_paths::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    N = length(m.L0)
    C = cholesky(Symmetric(m.rho + 1e-8 * I)).L

    L_terminal = zeros(n_paths, N)

    for i in 1:N
        T = m.T_dates[i]  # time to expiry of L_i
        sig = m.sigma[i]

        # Compute drift correction (spot measure drift up to T)
        # Approximate: use initial LIBOR values for drift
        drift = 0.0
        for j in (i+1):N
            delta_j = m.T_dates[j+1] - m.T_dates[j]
            L_j = m.L0[j]
            drift += m.rho[i, j] * sig * m.sigma[j] *
                     delta_j * L_j / (1.0 + delta_j * L_j)
        end
        # Annual drift * T
        total_drift = (drift - 0.5 * sig^2) * T

        for p in 1:n_paths
            z = dot(C[i, :], randn(rng, N))
            L_terminal[p, i] = m.L0[i] * exp(total_drift + sig * sqrt(T) * z)
        end
    end

    return L_terminal
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Yield Curve Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

"""
    YieldCurve

A bootstrapped yield curve.

# Fields
- `maturities::Vector{Float64}`: Maturities in years
- `zero_rates::Vector{Float64}`: Continuously compounded zero rates
- `discount_factors::Vector{Float64}`: P(0,T) = exp(-r(T)*T)
- `forward_rates::Vector{Float64}`: Instantaneous forward rates
"""
struct YieldCurve
    maturities::Vector{Float64}
    zero_rates::Vector{Float64}
    discount_factors::Vector{Float64}
    forward_rates::Vector{Float64}
end

"""
    bootstrap_yield_curve(
        deposit_rates::Vector{Tuple{Float64,Float64}},
        fra_rates::Vector{Tuple{Float64,Float64,Float64}},
        swap_rates::Vector{Tuple{Float64,Float64}}
    ) -> YieldCurve

Bootstrap a yield curve from:
- `deposit_rates`: Vector of (maturity, rate) pairs (e.g. [(0.25, 0.04), ...])
- `fra_rates`: Vector of (start, end, rate) for Forward Rate Agreements
- `swap_rates`: Vector of (maturity, par_swap_rate) pairs

Rates assumed continuously compounded unless noted.
"""
function bootstrap_yield_curve(
    deposit_rates::Vector{Tuple{Float64,Float64}},
    fra_rates::Vector{Tuple{Float64,Float64,Float64}},
    swap_rates::Vector{Tuple{Float64,Float64}}
)
    mats = Float64[]
    zeros = Float64[]

    # 1. Deposits: P(0,T) = exp(-r*T), simple rate implied
    for (T, r) in sort(deposit_rates)
        # Convert simple rate to continuous
        # P = 1/(1 + r*T) so r_cont = -ln(P)/T = ln(1+r*T)/T
        r_cont = log(1.0 + r * T) / T
        push!(mats, T)
        push!(zeros, r_cont)
    end

    # Interpolation helper
    function interp_zero(T)
        if isempty(mats)
            return 0.04
        end
        if T <= mats[1]
            return zeros[1]
        end
        if T >= mats[end]
            return zeros[end]
        end
        idx = searchsortedfirst(mats, T)
        t1, t2 = mats[idx-1], mats[idx]
        z1, z2 = zeros[idx-1], zeros[idx]
        return z1 + (z2 - z1) * (T - t1) / (t2 - t1)
    end

    function discount(T)
        z = interp_zero(T)
        return exp(-z * T)
    end

    # 2. FRAs: given P(0, T_start), solve for P(0, T_end)
    for (T_start, T_end, fra_rate) in sort(fra_rates, by=x->x[2])
        P_start = discount(T_start)
        alpha = T_end - T_start
        # FRA rate (simply compounded): (P(T_start)/P(T_end) - 1) / alpha = fra_rate
        P_end = P_start / (1.0 + fra_rate * alpha)
        z_end = -log(P_end) / T_end
        push!(mats, T_end)
        push!(zeros, z_end)

        # Re-sort
        order = sortperm(mats)
        mats = mats[order]
        zeros = zeros[order]
    end

    # 3. Swap rates: bootstrap via annuity approach
    for (T_swap, S) in sort(swap_rates)
        # Par swap: floating leg = fixed leg
        # Floating: 1 - P(0, T_swap)
        # Fixed: S * sum_i [ delta_i * P(0, T_i) ]
        # Annual payments assumed (delta = 1.0)

        n_periods = round(Int, T_swap)
        if n_periods < 1
            continue
        end

        payment_times = [i * 1.0 for i in 1:n_periods]

        # Annuity of known payments (all except last)
        annuity_known = sum(1.0 * discount(t) for t in payment_times[1:end-1]; init=0.0)

        # Solve for P(0, T_swap):
        # 1 - P(0,T) = S * (annuity_known + P(0,T))
        # P(0,T) = (1 - S * annuity_known) / (1 + S)
        P_T = (1.0 - S * annuity_known) / (1.0 + S)
        if P_T <= 0
            continue
        end
        z_T = -log(P_T) / T_swap

        push!(mats, T_swap)
        push!(zeros, z_T)

        order = sortperm(mats)
        mats = mats[order]
        zeros = zeros[order]
    end

    # Remove duplicate maturities (keep last)
    unique_mask = [true; diff(mats) .> 1e-10]
    mats = mats[unique_mask]
    zeros = zeros[unique_mask]

    # Compute discount factors
    dfs = exp.(-zeros .* mats)

    # Compute instantaneous forward rates: f(T) = -d/dT ln P(0,T)
    n = length(mats)
    fwds = zeros(n)
    for i in 1:n
        if i == n
            # Forward extrapolation
            fwds[i] = zeros[i] + mats[i] * (zeros[i] - zeros[max(1, i-1)]) /
                       max(mats[i] - mats[max(1, i-1)], 1e-8)
        else
            dT = mats[i+1] - mats[i]
            dlnP = log(dfs[i+1]) - log(dfs[i])
            fwds[i] = -dlnP / dT
        end
    end

    return YieldCurve(mats, zeros, dfs, fwds)
end

"""
    zero_to_forward(yc::YieldCurve, T1::Float64, T2::Float64) -> Float64

Compute simply-compounded forward rate F(T1, T2) from zero curve.
"""
function zero_to_forward(yc::YieldCurve, T1::Float64, T2::Float64)
    P1 = interpolate_discount(yc, T1)
    P2 = interpolate_discount(yc, T2)
    alpha = T2 - T1
    return (P1 / P2 - 1.0) / alpha
end

"""
    forward_to_zero(forward_rates::Vector{Float64}, times::Vector{Float64}) -> Vector{Float64}

Convert a strip of forward rates to zero rates.
"""
function forward_to_zero(forward_rates::Vector{Float64}, times::Vector{Float64})
    n = length(forward_rates)
    @assert length(times) == n
    zeros_out = zeros(n)
    for i in 1:n
        # Trapezoidal integration of forward rates
        if i == 1
            zeros_out[i] = forward_rates[1]
        else
            # integral of f(t) dt from 0 to T_i
            integral = 0.0
            for j in 2:i
                dt = times[j] - times[j-1]
                integral += 0.5 * (forward_rates[j-1] + forward_rates[j]) * dt
            end
            zeros_out[i] = integral / times[i]
        end
    end
    return zeros_out
end

"""
    interpolate_discount(yc::YieldCurve, T::Float64) -> Float64

Interpolate discount factor at maturity T using log-linear interpolation.
"""
function interpolate_discount(yc::YieldCurve, T::Float64)
    if T <= 0
        return 1.0
    end
    mats = yc.maturities
    dfs = yc.discount_factors

    if T <= mats[1]
        # Extrapolate with first rate
        return exp(-yc.zero_rates[1] * T)
    end
    if T >= mats[end]
        return exp(-yc.zero_rates[end] * T)
    end

    idx = searchsortedfirst(mats, T)
    t1, t2 = mats[idx-1], mats[idx]
    lnP1, lnP2 = log(dfs[idx-1]), log(dfs[idx])
    lnP = lnP1 + (lnP2 - lnP1) * (T - t1) / (t2 - t1)
    return exp(lnP)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Swap Valuation
# ─────────────────────────────────────────────────────────────────────────────

"""
    SwapContract

Interest rate swap specification.
"""
struct SwapContract
    notional::Float64
    T_start::Float64
    T_end::Float64
    fixed_rate::Float64
    payment_freq::Float64  # payments per year (e.g. 1 = annual, 2 = semi)
    pay_fixed::Bool        # true = paying fixed, receiving floating
end

"""
    swap_annuity(yc::YieldCurve, swap::SwapContract) -> Float64

Compute the annuity factor (PV of fixed cash flows per unit rate).
A = sum_i delta_i * P(0, T_i)
"""
function swap_annuity(yc::YieldCurve, swap::SwapContract)
    delta = 1.0 / swap.payment_freq
    times = collect(swap.T_start + delta : delta : swap.T_end)
    annuity = sum(delta * interpolate_discount(yc, t) for t in times)
    return annuity
end

"""
    swap_par_rate(yc::YieldCurve, swap::SwapContract) -> Float64

Compute the par (fair) swap rate: S such that swap value = 0.
S = (P(0, T_start) - P(0, T_end)) / A
"""
function swap_par_rate(yc::YieldCurve, swap::SwapContract)
    P_start = interpolate_discount(yc, swap.T_start)
    P_end = interpolate_discount(yc, swap.T_end)
    A = swap_annuity(yc, swap)
    if A <= 0
        return 0.0
    end
    return (P_start - P_end) / A
end

"""
    swap_value(yc::YieldCurve, swap::SwapContract) -> Float64

Compute current value of swap.
Value of payer swap = N * (A * (S_par - K))
where S_par is current par rate, K is fixed rate.
"""
function swap_value(yc::YieldCurve, swap::SwapContract)
    S = swap_par_rate(yc, swap)
    A = swap_annuity(yc, swap)
    N = swap.notional
    sign = swap.pay_fixed ? -1.0 : 1.0
    return sign * N * A * (S - swap.fixed_rate)
end

"""
    swap_dv01(yc::YieldCurve, swap::SwapContract; bump_bps=1.0) -> Float64

Compute DV01 (dollar value of 1bp) of the swap via parallel bump.
"""
function swap_dv01(yc::YieldCurve, swap::SwapContract; bump_bps::Float64=1.0)
    bump = bump_bps / 10000.0
    # Bump all zero rates up by 1bp
    yc_up = YieldCurve(
        yc.maturities,
        yc.zero_rates .+ bump,
        exp.(-(yc.zero_rates .+ bump) .* yc.maturities),
        yc.forward_rates .+ bump
    )
    v_base = swap_value(yc, swap)
    v_up = swap_value(yc_up, swap)
    return v_up - v_base
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Bond Analytics
# ─────────────────────────────────────────────────────────────────────────────

"""
    bond_duration(cash_flows::Vector{Float64}, times::Vector{Float64},
                  yc::YieldCurve) -> NamedTuple

Compute Macaulay and modified duration of a bond.
"""
function bond_duration(cash_flows::Vector{Float64}, times::Vector{Float64},
                       yc::YieldCurve)
    n = length(cash_flows)
    @assert length(times) == n

    pv_total = 0.0
    weighted_time = 0.0

    for i in 1:n
        P = interpolate_discount(yc, times[i])
        pv_cf = cash_flows[i] * P
        pv_total += pv_cf
        weighted_time += times[i] * pv_cf
    end

    macaulay = pv_total > 0 ? weighted_time / pv_total : 0.0

    # Yield to maturity for modified duration
    function bond_pv(y)
        sum(cash_flows[i] * exp(-y * times[i]) for i in 1:n)
    end

    # Solve for YTM
    ytm_result = optimize(y -> (bond_pv(y[1]) - pv_total)^2, [0.05], LBFGS())
    ytm = Optim.minimizer(ytm_result)[1]

    modified = macaulay / (1.0 + ytm)

    return (macaulay=macaulay, modified=modified, ytm=ytm, price=pv_total)
end

"""
    bond_convexity(cash_flows::Vector{Float64}, times::Vector{Float64},
                   yc::YieldCurve) -> Float64

Compute convexity of a bond.
Convexity = (1/P) * sum_i t_i^2 * PV(CF_i)
"""
function bond_convexity(cash_flows::Vector{Float64}, times::Vector{Float64},
                        yc::YieldCurve)
    n = length(cash_flows)
    pv_total = sum(cash_flows[i] * interpolate_discount(yc, times[i]) for i in 1:n)
    conv = sum(times[i]^2 * cash_flows[i] * interpolate_discount(yc, times[i]) for i in 1:n)
    return pv_total > 0 ? conv / pv_total : 0.0
end

"""
    dv01(cash_flows::Vector{Float64}, times::Vector{Float64}, yc::YieldCurve) -> Float64

Dollar value of 1 basis point (parallel shift) for a cash flow stream.
DV01 = -dP/dy * 0.0001 ≈ (modified_duration * price) * 0.0001
"""
function dv01(cash_flows::Vector{Float64}, times::Vector{Float64}, yc::YieldCurve)
    dur = bond_duration(cash_flows, times, yc)
    return dur.modified * dur.price * 0.0001
end

"""
    key_rate_durations(
        cash_flows::Vector{Float64},
        times::Vector{Float64},
        yc::YieldCurve,
        key_maturities::Vector{Float64};
        bump_bps=1.0
    ) -> Vector{Float64}

Compute key rate durations by bumping each key rate tenor by 1bp.
"""
function key_rate_durations(
    cash_flows::Vector{Float64},
    times::Vector{Float64},
    yc::YieldCurve,
    key_maturities::Vector{Float64};
    bump_bps::Float64=1.0
)
    bump = bump_bps / 10000.0
    base_pv = sum(cash_flows[i] * interpolate_discount(yc, times[i]) for i in eachindex(cash_flows))
    krd = zeros(length(key_maturities))

    for (k, mat_k) in enumerate(key_maturities)
        # Localized bump: triangular function centered at mat_k
        bump_width = length(key_maturities) > 1 ?
            (k < length(key_maturities) ? key_maturities[k+1] - mat_k :
             mat_k - key_maturities[k-1]) : 1.0

        # Bumped zero rates
        bumped_zeros = copy(yc.zero_rates)
        for (j, T_j) in enumerate(yc.maturities)
            dist = abs(T_j - mat_k)
            if dist < bump_width
                weight = 1.0 - dist / bump_width
                bumped_zeros[j] += bump * weight
            end
        end

        bumped_dfs = exp.(-bumped_zeros .* yc.maturities)
        yc_bumped = YieldCurve(yc.maturities, bumped_zeros, bumped_dfs, yc.forward_rates)

        bumped_pv = sum(cash_flows[i] * interpolate_discount(yc_bumped, times[i])
                        for i in eachindex(cash_flows))
        krd[k] = -(bumped_pv - base_pv) / (base_pv * bump)
    end

    return krd
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Cap/Floor and Swaption Pricing
# ─────────────────────────────────────────────────────────────────────────────

"""
    CapFloor

Cap or floor specification.
"""
struct CapFloor
    notional::Float64
    T_start::Float64
    T_end::Float64
    strike::Float64
    is_cap::Bool
    payment_freq::Float64
end

"""
    cap_black_price(yc::YieldCurve, cf::CapFloor, vol::Float64) -> Float64

Price a cap using Black's model (sum of caplets).
"""
function cap_black_price(yc::YieldCurve, cf::CapFloor, vol::Float64)
    delta = 1.0 / cf.payment_freq
    times = collect(cf.T_start + delta : delta : cf.T_end)
    N = cf.notional
    K = cf.strike
    d = Normal()
    total = 0.0

    for (i, T_end) in enumerate(times)
        T_start = T_end - delta
        if T_start < 0
            continue
        end
        P_end = interpolate_discount(yc, T_end)
        P_start = interpolate_discount(yc, T_start)
        F = (P_start / P_end - 1.0) / delta  # forward LIBOR

        if vol * sqrt(T_start) < 1e-10
            if cf.is_cap
                total += N * delta * P_end * max(F - K, 0.0)
            else
                total += N * delta * P_end * max(K - F, 0.0)
            end
            continue
        end

        d1 = (log(F / K) + 0.5 * vol^2 * T_start) / (vol * sqrt(T_start))
        d2 = d1 - vol * sqrt(T_start)

        if cf.is_cap
            total += N * delta * P_end * (F * cdf(d, d1) - K * cdf(d, d2))
        else
            total += N * delta * P_end * (K * cdf(d, -d2) - F * cdf(d, -d1))
        end
    end

    return total
end

"""
    floor_black_price(yc::YieldCurve, cf::CapFloor, vol::Float64) -> Float64

Convenience wrapper for floor pricing.
"""
function floor_black_price(yc::YieldCurve, cf::CapFloor, vol::Float64)
    floor_cf = CapFloor(cf.notional, cf.T_start, cf.T_end, cf.strike, false, cf.payment_freq)
    return cap_black_price(yc, floor_cf, vol)
end

"""
    Swaption

European swaption specification.
"""
struct Swaption
    notional::Float64
    T_option::Float64    # expiry of option
    T_swap_start::Float64
    T_swap_end::Float64
    fixed_rate::Float64
    is_payer::Bool       # payer swaption if true
    payment_freq::Float64
end

"""
    swaption_black_price(yc::YieldCurve, s::Swaption, vol::Float64) -> Float64

Price a European swaption using Black's model.
Value = N * A * [S_par * N(d1) - K * N(d2)] (payer)
"""
function swaption_black_price(yc::YieldCurve, s::Swaption, vol::Float64)
    # Forward par swap rate
    swap_temp = SwapContract(s.notional, s.T_swap_start, s.T_swap_end,
                              s.fixed_rate, s.payment_freq, true)
    S = swap_par_rate(yc, swap_temp)
    A = swap_annuity(yc, swap_temp)
    K = s.fixed_rate
    T = s.T_option
    N = s.notional
    d = Normal()

    if vol * sqrt(T) < 1e-12
        if s.is_payer
            return N * A * max(S - K, 0.0)
        else
            return N * A * max(K - S, 0.0)
        end
    end

    d1 = (log(S / K) + 0.5 * vol^2 * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)

    if s.is_payer
        return N * A * (S * cdf(d, d1) - K * cdf(d, d2))
    else
        return N * A * (K * cdf(d, -d2) - S * cdf(d, -d1))
    end
end

"""
    implied_vol_swaption(yc::YieldCurve, s::Swaption, market_price::Float64) -> Float64

Implied volatility from market swaption price (Black inversion via bisection).
"""
function implied_vol_swaption(yc::YieldCurve, s::Swaption, market_price::Float64)
    f = vol -> swaption_black_price(yc, s, vol) - market_price
    vol_lo, vol_hi = 0.001, 5.0

    if f(vol_lo) * f(vol_hi) > 0
        return NaN
    end

    for _ in 1:100
        vol_mid = (vol_lo + vol_hi) / 2
        if f(vol_mid) * f(vol_lo) <= 0
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

end # module InterestRates
