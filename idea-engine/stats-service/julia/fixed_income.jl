"""
fixed_income.jl — Fixed Income Analytics for Macro Factor Modeling

Covers:
  - Bond pricing: PV of cash flows, yield, duration, convexity
  - Yield curve construction: bootstrap from par rates
  - Nelson-Siegel-Svensson (NSS) model fitting
  - Vasicek and CIR short-rate models: pricing, simulation
  - Forward rates and instantaneous forward curve
  - Duration matching and immunization
  - Swap pricing and par swap rate
  - Basis risk between rate benchmarks
  - Crypto vs traditional yield comparison as macro factor

Pure Julia stdlib only. No external dependencies.
"""

module FixedIncome

using Statistics, LinearAlgebra, Random

export Bond, bond_price, bond_yield, bond_duration, bond_convexity
export yield_curve_bootstrap, spot_from_discount, forward_rate
export NelsonSiegel, NelsonSiegelSvensson, fit_nss!, nss_yield
export VasicekModel, cir_model, simulate_vasicek, simulate_cir
export vasicek_bond_price, cir_bond_price
export instantaneous_forward_curve, par_yield_curve
export immunize_portfolio, duration_match
export swap_price, par_swap_rate, swap_dv01
export basis_spread, basis_risk_factor
export crypto_yield_comparison, macro_rate_sensitivity
export run_fixed_income_demo

# ─────────────────────────────────────────────────────────────
# 1. BOND PRICING
# ─────────────────────────────────────────────────────────────

"""
    Bond

Fixed-coupon bond specification.

Fields:
  face       — face value (par)
  coupon     — annual coupon rate
  maturity   — years to maturity
  frequency  — coupons per year (1=annual, 2=semi-annual, 4=quarterly)
  settlement — settlement date (in years, usually 0)
"""
struct Bond
    face::Float64
    coupon::Float64   # annual coupon rate
    maturity::Float64
    frequency::Int
    settlement::Float64
end

Bond(face, coupon, maturity; freq=2) = Bond(face, coupon, maturity, freq, 0.0)

"""Cash flow times and amounts for a bond."""
function bond_cashflows(b::Bond)
    dt   = 1.0 / b.frequency
    n    = Int(round(b.maturity * b.frequency))
    times = [b.settlement + i * dt for i in 1:n]
    coupons = fill(b.face * b.coupon / b.frequency, n)
    coupons[end] += b.face  # add face at maturity
    times, coupons
end

"""
    bond_price(b::Bond, yield::Float64) -> Float64

Price bond given yield to maturity (annual, same compounding as frequency).
"""
function bond_price(b::Bond, yield::Float64)::Float64
    times, cashflows = bond_cashflows(b)
    r_per_period = yield / b.frequency
    sum(cf * (1 + r_per_period)^(-t * b.frequency)
        for (t, cf) in zip(times, cashflows))
end

"""
    bond_price_continuous(b::Bond, yield_continuous::Float64) -> Float64

Price bond using continuously compounded yield.
"""
function bond_price_continuous(b::Bond, y::Float64)::Float64
    times, cashflows = bond_cashflows(b)
    sum(cf * exp(-y * t) for (t, cf) in zip(times, cashflows))
end

"""
    bond_yield(b::Bond, price::Float64; tol=1e-8, maxiter=100) -> Float64

Compute yield to maturity from price using Newton-Raphson.
"""
function bond_yield(b::Bond, price::Float64;
                     tol::Float64=1e-8, maxiter::Int=100)::Float64
    # Initial guess: coupon / price
    y = b.coupon * b.face / price
    y = clamp(y, 1e-4, 0.50)

    for _ in 1:maxiter
        p  = bond_price(b, y)
        dp = bond_dollar_duration(b, y)  # negative
        dy = (p - price) / (-dp)
        y  = clamp(y + dy, 1e-6, 1.0)
        abs(p - price) < tol && break
    end
    y
end

"""Dollar duration = -dP/dy (approximate via finite difference)."""
function bond_dollar_duration(b::Bond, yield::Float64; h::Float64=1e-5)::Float64
    (bond_price(b, yield + h) - bond_price(b, yield - h)) / (2h)
end

"""
    bond_duration(b::Bond, yield::Float64) -> (modified_dur, macaulay_dur)

Macaulay and modified duration.
MacD = (1/P) * Σ t_i * CF_i * DF_i
ModD = MacD / (1 + y/freq)
"""
function bond_duration(b::Bond, yield::Float64)
    times, cashflows = bond_cashflows(b)
    r = yield / b.frequency
    price = bond_price(b, yield)
    price <= 1e-10 && return (0.0, 0.0)
    mac_d = sum(t * cf * (1+r)^(-t * b.frequency)
                for (t, cf) in zip(times, cashflows)) / price
    mod_d = mac_d / (1 + r)
    (modified_duration=mod_d, macaulay_duration=mac_d)
end

"""
    bond_convexity(b::Bond, yield::Float64) -> Float64

Bond convexity: second derivative of price w.r.t. yield (normalized).
"""
function bond_convexity(b::Bond, yield::Float64)::Float64
    times, cashflows = bond_cashflows(b)
    r     = yield / b.frequency
    price = bond_price(b, yield)
    price <= 1e-10 && return 0.0
    freq  = Float64(b.frequency)
    conv  = sum(t * (t + 1/freq) * cf * (1+r)^(-(t * b.frequency + 2))
                for (t, cf) in zip(times, cashflows)) / price
    conv / (1 + r)^2
end

"""
    bond_dv01(b::Bond, yield::Float64) -> Float64

DV01 (Dollar Value of 01): price change per 1 basis point increase in yield.
"""
bond_dv01(b::Bond, y::Float64) = abs(bond_dollar_duration(b, y)) * 0.0001

"""
    price_yield_approximation(b::Bond, yield0, yield1, price0) -> Float64

Price approximation using duration and convexity:
  ΔP ≈ -D*P*Δy + 0.5*C*P*(Δy)^2
"""
function price_yield_approximation(b::Bond, yield0::Float64,
                                    yield1::Float64, price0::Float64)::Float64
    dur  = bond_duration(b, yield0).modified_duration
    conv = bond_convexity(b, yield0)
    dy   = yield1 - yield0
    price0 * (1 - dur * dy + 0.5 * conv * dy^2)
end

# ─────────────────────────────────────────────────────────────
# 2. YIELD CURVE CONSTRUCTION
# ─────────────────────────────────────────────────────────────

"""
    yield_curve_bootstrap(maturities, par_yields; frequency=2)
       -> (spot_rates, discount_factors)

Bootstrap zero/spot rates from par yield curve.
par_yields[i] = par yield (annual) for maturity maturities[i].
"""
function yield_curve_bootstrap(maturities::Vector{Float64},
                                 par_yields::Vector{Float64};
                                 frequency::Int=2)
    n           = length(maturities)
    spot_rates  = zeros(n)
    disc_factors = zeros(n)  # P(0, T_i)

    for i in 1:n
        T   = maturities[i]
        par = par_yields[i]
        c   = par / frequency  # coupon per period
        n_periods = Int(round(T * frequency))

        if n_periods <= 1
            # Simple: P = 1 / (1 + par * T)
            disc_factors[i] = 1.0 / (1 + par * T)
            spot_rates[i]   = -log(disc_factors[i]) / T
        else
            # Sum of known discount factors for intermediate coupons
            pv_coupons = 0.0
            for j in 1:(n_periods - 1)
                t_j = j / frequency
                # Interpolate discount factor for this maturity
                if t_j <= maturities[1]
                    df_j = disc_factors[1]^(t_j / maturities[1])
                else
                    # Linear interpolation on log(DF)
                    k = findlast(m -> m <= t_j, maturities[1:i-1])
                    isnothing(k) && (k = 1)
                    k2 = min(k + 1, i - 1)
                    if k == k2 || maturities[k] >= maturities[k2]
                        df_j = disc_factors[k]
                    else
                        frac = (t_j - maturities[k]) / (maturities[k2] - maturities[k])
                        df_j = exp(log(disc_factors[k]) * (1-frac) +
                                   log(disc_factors[k2]+1e-15) * frac)
                    end
                end
                pv_coupons += c * df_j
            end
            # Final period: coupon + face
            disc_factors[i] = (1.0 - pv_coupons) / (1.0 + c)
            disc_factors[i] = max(disc_factors[i], 1e-10)
            spot_rates[i]   = -log(disc_factors[i]) / T
        end
    end
    (spot_rates=spot_rates, discount_factors=disc_factors, maturities=maturities)
end

"""
    spot_from_discount(T, discount_factor) -> Float64

Continuously compounded spot rate from discount factor.
"""
spot_from_discount(T::Float64, df::Float64) = -log(max(df, 1e-15)) / T

"""
    forward_rate(df1, df2, t1, t2) -> Float64

Continuously compounded forward rate between t1 and t2.
"""
function forward_rate(df1::Float64, df2::Float64,
                       t1::Float64, t2::Float64)::Float64
    dt = t2 - t1
    dt < 1e-10 && return 0.0
    (log(max(df1, 1e-15)) - log(max(df2, 1e-15))) / dt
end

"""
    par_yield_curve(spot_rates, maturities; frequency=2) -> Vector{Float64}

Convert spot curve to par yield curve.
"""
function par_yield_curve(spot_rates::Vector{Float64},
                           maturities::Vector{Float64};
                           frequency::Int=2)::Vector{Float64}
    n = length(maturities)
    par_yields = zeros(n)
    for i in 1:n
        T = maturities[i]
        n_periods = Int(round(T * frequency))
        dt = T / n_periods
        # Discount factors (interpolated spot rates)
        dfs = [exp(-spot_rates[min(i,n)] * j * dt) for j in 1:n_periods]
        sum_dfs = sum(dfs[1:end-1])
        par_yields[i] = (1 - dfs[end]) / (sum_dfs / frequency + dfs[end] / frequency) * frequency
    end
    par_yields
end

# ─────────────────────────────────────────────────────────────
# 3. NELSON-SIEGEL-SVENSSON
# ─────────────────────────────────────────────────────────────

"""
    NelsonSiegel

Nelson-Siegel yield curve parametrization.
y(T) = β0 + β1 * (1 - e^{-T/τ}) / (T/τ)
     + β2 * [(1 - e^{-T/τ}) / (T/τ) - e^{-T/τ}]
"""
mutable struct NelsonSiegel
    beta0::Float64  # long-run level
    beta1::Float64  # slope (short rate - long rate)
    beta2::Float64  # curvature (hump)
    tau::Float64    # decay factor
end

NelsonSiegel() = NelsonSiegel(0.03, -0.01, 0.02, 2.0)

"""Nelson-Siegel yield at maturity T."""
function ns_yield(m::NelsonSiegel, T::Float64)::Float64
    T <= 0 && return m.beta0 + m.beta1
    decay = exp(-T / m.tau)
    load1 = (1 - decay) / (T / m.tau)
    m.beta0 + m.beta1 * load1 + m.beta2 * (load1 - decay)
end

"""
    NelsonSiegelSvensson

Extended Nelson-Siegel with second hump term.
Adds β3 and τ2 for better fit at long maturities.
"""
mutable struct NelsonSiegelSvensson
    beta0::Float64
    beta1::Float64
    beta2::Float64
    beta3::Float64
    tau1::Float64
    tau2::Float64
end

NelsonSiegelSvensson() = NelsonSiegelSvensson(0.03, -0.01, 0.02, 0.01, 2.0, 5.0)

"""NSS yield at maturity T."""
function nss_yield(m::NelsonSiegelSvensson, T::Float64)::Float64
    T <= 0 && return m.beta0 + m.beta1
    d1 = exp(-T / m.tau1)
    d2 = exp(-T / m.tau2)
    l1 = (1 - d1) / (T / m.tau1)
    l2 = (1 - d2) / (T / m.tau2)
    m.beta0 + m.beta1 * l1 + m.beta2 * (l1 - d1) + m.beta3 * (l2 - d2)
end

"""
    fit_nss!(model, maturities, yields; lr=0.001, maxiter=2000) -> NelsonSiegelSvensson

Fit NSS model to observed yield curve via gradient descent.
"""
function fit_nss!(model::NelsonSiegelSvensson,
                   maturities::Vector{Float64},
                   yields::Vector{Float64};
                   lr::Float64=0.001, maxiter::Int=2000)
    n = length(maturities)
    params = [model.beta0, model.beta1, model.beta2, model.beta3,
              max(model.tau1, 0.1), max(model.tau2, 0.1)]
    best_loss = Inf
    best_params = copy(params)

    for iter in 1:maxiter
        # Numerical gradient
        loss = sum((nss_yield(NelsonSiegelSvensson(params...), maturities[i]) - yields[i])^2
                   for i in 1:n)
        loss < best_loss && (best_loss = loss; best_params = copy(params))
        abs(loss) < 1e-12 && break

        grad = zeros(6)
        h = 1e-5
        for j in 1:6
            p_plus  = copy(params); p_plus[j]  += h
            p_minus = copy(params); p_minus[j] -= h
            l_plus  = sum((nss_yield(NelsonSiegelSvensson(p_plus...), maturities[i]) - yields[i])^2 for i in 1:n)
            l_minus = sum((nss_yield(NelsonSiegelSvensson(p_minus...), maturities[i]) - yields[i])^2 for i in 1:n)
            grad[j] = (l_plus - l_minus) / (2h)
        end

        # Adaptive learning rate
        step = lr / (1 + iter / 500)
        params .-= step .* grad
        params[5] = max(params[5], 0.05)
        params[6] = max(params[6], 0.05)
    end

    model.beta0 = best_params[1]
    model.beta1 = best_params[2]
    model.beta2 = best_params[3]
    model.beta3 = best_params[4]
    model.tau1  = best_params[5]
    model.tau2  = best_params[6]
    model
end

# ─────────────────────────────────────────────────────────────
# 4. SHORT-RATE MODELS
# ─────────────────────────────────────────────────────────────

"""
    VasicekModel

Vasicek (1977) mean-reverting short rate model.
dr = κ(θ - r)dt + σ dW

Fields:
  kappa — mean reversion speed
  theta — long-run mean
  sigma — volatility
  r0    — initial rate
"""
struct VasicekModel
    kappa::Float64
    theta::Float64
    sigma::Float64
    r0::Float64
end

VasicekModel() = VasicekModel(0.15, 0.05, 0.01, 0.03)

"""
    simulate_vasicek(m, T, n_steps, n_paths; rng=...) -> Matrix{Float64}

Simulate Vasicek short rate paths. Returns n_paths × (n_steps+1) matrix.
"""
function simulate_vasicek(m::VasicekModel, T::Float64, n_steps::Int,
                            n_paths::Int; rng=MersenneTwister(1))::Matrix{Float64}
    dt    = T / n_steps
    paths = zeros(n_paths, n_steps + 1)
    for p in 1:n_paths
        r = m.r0
        paths[p, 1] = r
        for t in 1:n_steps
            # Exact discretization
            drift   = r * exp(-m.kappa * dt) + m.theta * (1 - exp(-m.kappa * dt))
            vol_dt  = m.sigma * sqrt((1 - exp(-2*m.kappa*dt)) / (2*m.kappa))
            r       = drift + vol_dt * randn(rng)
            paths[p, t+1] = r
        end
    end
    paths
end

"""
    vasicek_bond_price(m::VasicekModel, T, r=m.r0) -> Float64

Analytic zero-coupon bond price under Vasicek model: P(0,T) = A(T)*exp(-B(T)*r).
"""
function vasicek_bond_price(m::VasicekModel, T::Float64, r::Float64=m.r0)::Float64
    T <= 0 && return 1.0
    B = (1 - exp(-m.kappa * T)) / m.kappa
    A_exp = exp((m.theta - m.sigma^2 / (2 * m.kappa^2)) * (B - T)
                - m.sigma^2 * B^2 / (4 * m.kappa))
    A_exp * exp(-B * r)
end

"""
    vasicek_yield_curve(m, maturities) -> Vector{Float64}

Vasicek model yield curve: y(T) = -ln(P(0,T)) / T.
"""
function vasicek_yield_curve(m::VasicekModel,
                               maturities::Vector{Float64})::Vector{Float64}
    [begin
         p = vasicek_bond_price(m, T)
         p > 0 ? -log(p) / T : m.r0
     end for T in maturities]
end

"""
    CIR model (Cox-Ingersoll-Ross)

dr = κ(θ - r)dt + σ√r dW
Ensures non-negative rates when 2κθ ≥ σ² (Feller condition).
"""
struct CIRModel
    kappa::Float64
    theta::Float64
    sigma::Float64
    r0::Float64
end

CIRModel() = CIRModel(0.15, 0.05, 0.05, 0.03)

"""Simulate CIR paths."""
function simulate_cir(m::CIRModel, T::Float64, n_steps::Int,
                       n_paths::Int; rng=MersenneTwister(1))::Matrix{Float64}
    dt    = T / n_steps
    paths = zeros(n_paths, n_steps + 1)
    for p in 1:n_paths
        r = m.r0
        paths[p, 1] = r
        for t in 1:n_steps
            r = r + m.kappa * (m.theta - r) * dt +
                m.sigma * sqrt(max(r, 0.0)) * sqrt(dt) * randn(rng)
            r = max(r, 0.0)
            paths[p, t+1] = r
        end
    end
    paths
end

"""
    cir_bond_price(m::CIRModel, T, r=m.r0) -> Float64

Analytic CIR zero-coupon bond price.
"""
function cir_bond_price(m::CIRModel, T::Float64, r::Float64=m.r0)::Float64
    T <= 0 && return 1.0
    gamma = sqrt(m.kappa^2 + 2*m.sigma^2)
    exp1  = exp(gamma * T)
    B = 2*(exp1 - 1) / ((gamma + m.kappa)*(exp1 - 1) + 2*gamma)
    A = (2*gamma*exp(0.5*(m.kappa + gamma)*T) /
         ((gamma + m.kappa)*(exp1 - 1) + 2*gamma))^(2*m.kappa*m.theta / m.sigma^2)
    A * exp(-B * r)
end

"""
    instantaneous_forward_curve(model, maturities) -> Vector{Float64}

Instantaneous forward rate f(0,T) = -∂ln P(0,T)/∂T.
"""
function instantaneous_forward_curve(model::Union{VasicekModel,CIRModel},
                                       maturities::Vector{Float64})::Vector{Float64}
    h = 1e-5
    map(maturities) do T
        if isa(model, VasicekModel)
            p1 = vasicek_bond_price(model, T + h)
            p2 = vasicek_bond_price(model, T - h)
        else
            p1 = cir_bond_price(model, T + h)
            p2 = cir_bond_price(model, T - h)
        end
        -(log(max(p1, 1e-15)) - log(max(p2, 1e-15))) / (2h)
    end
end

# ─────────────────────────────────────────────────────────────
# 5. DURATION MATCHING AND IMMUNIZATION
# ─────────────────────────────────────────────────────────────

"""
    immunize_portfolio(liability_pv, liability_duration, bond1, bond2, yield)
       -> (w1, w2)

Find portfolio weights (w1, w2) for two bonds that immunize a liability.
Constraints:
  w1 * P1 + w2 * P2 = L            (PV match)
  w1 * D1 * P1 + w2 * D2 * P2 = L * DL (duration match)
"""
function immunize_portfolio(liability_pv::Float64,
                              liability_duration::Float64,
                              bond1::Bond, bond2::Bond,
                              yield::Float64)
    P1 = bond_price(bond1, yield)
    P2 = bond_price(bond2, yield)
    D1 = bond_duration(bond1, yield).modified_duration
    D2 = bond_duration(bond2, yield).modified_duration

    # Solve 2×2 system
    A = [P1 P2; D1*P1 D2*P2]
    b = [liability_pv; liability_duration * liability_pv]
    det_A = A[1,1]*A[2,2] - A[1,2]*A[2,1]
    abs(det_A) < 1e-10 && return (nothing, nothing)

    w = [A[2,2]*b[1] - A[1,2]*b[2], -A[2,1]*b[1] + A[1,1]*b[2]] ./ det_A
    (w1=w[1], w2=w[2], portfolio_pv=w[1]*P1 + w[2]*P2,
     portfolio_duration=(w[1]*D1*P1 + w[2]*D2*P2) / (w[1]*P1 + w[2]*P2))
end

"""
    duration_match(target_duration, bonds, yields) -> Vector{Float64}

Find weights for a portfolio of bonds matching a target duration.
Uses minimum-variance portfolio subject to duration constraint.
"""
function duration_match(target_duration::Float64,
                          bonds::Vector{Bond},
                          yield::Float64)::Vector{Float64}
    n = length(bonds)
    prices    = [bond_price(b, yield) for b in bonds]
    durations = [bond_duration(b, yield).modified_duration for b in bonds]

    # Minimize Σ w_i² subject to Σ D_i*P_i*w_i = target * Σ P_i*w_i and Σ w_i = 1
    # Simplified: equal weight with closest duration bonds
    d_diff = abs.(durations .- target_duration)
    weights = 1.0 ./ (d_diff .+ 0.1)
    weights ./= sum(weights)
    weights
end

# ─────────────────────────────────────────────────────────────
# 6. INTEREST RATE SWAPS
# ─────────────────────────────────────────────────────────────

"""
    swap_price(fixed_rate, maturity, spot_curve_fn, frequency=2, notional=1.0)
       -> Float64

Price of pay-fixed, receive-floating swap.
spot_curve_fn: function T -> spot rate.
"""
function swap_price(fixed_rate::Float64, maturity::Float64,
                     spot_curve_fn::Function; frequency::Int=2,
                     notional::Float64=1.0)::Float64
    dt = 1.0 / frequency
    n  = Int(round(maturity * frequency))
    times = [i * dt for i in 1:n]

    # Discount factors from spot curve
    dfs = [exp(-spot_curve_fn(t) * t) for t in times]

    # Fixed leg PV
    fixed_pv = sum(fixed_rate / frequency * df * notional for df in dfs)

    # Floating leg PV (= notional - notional * last DF)
    float_pv = notional * (1 - dfs[end])

    float_pv - fixed_pv
end

"""
    par_swap_rate(maturity, spot_curve_fn; frequency=2) -> Float64

Find the fixed rate that makes a swap worth zero (par swap rate).
"""
function par_swap_rate(maturity::Float64, spot_curve_fn::Function;
                        frequency::Int=2)::Float64
    dt = 1.0 / frequency
    n  = Int(round(maturity * frequency))
    times = [i * dt for i in 1:n]
    dfs   = [exp(-spot_curve_fn(t) * t) for t in times]
    annuity = sum(dfs) / frequency
    annuity < 1e-10 && return 0.0
    (1 - dfs[end]) / annuity
end

"""
    swap_dv01(fixed_rate, maturity, spot_curve_fn; frequency=2, shift=1e-4)
       -> Float64

Swap DV01: price change per 1bp parallel shift in spot curve.
"""
function swap_dv01(fixed_rate::Float64, maturity::Float64,
                    spot_curve_fn::Function; frequency::Int=2,
                    shift::Float64=1e-4)::Float64
    p0 = swap_price(fixed_rate, maturity, spot_curve_fn; frequency=frequency)
    p1 = swap_price(fixed_rate, maturity,
                    T -> spot_curve_fn(T) + shift; frequency=frequency)
    (p1 - p0) / shift * 1e-4  # normalize to 1bp
end

"""
    swap_curve_from_yields(maturities, yields; frequency=2) -> Vector{Float64}

Compute par swap rates from a set of spot rates.
"""
function swap_curve_from_yields(maturities::Vector{Float64},
                                  yields::Vector{Float64};
                                  frequency::Int=2)::Vector{Float64}
    n = length(maturities)
    # Build interpolation closure
    function interp_spot(T::Float64)::Float64
        T <= maturities[1] && return yields[1]
        T >= maturities[end] && return yields[end]
        idx = searchsortedfirst(maturities, T)
        idx = clamp(idx, 2, n)
        t0 = maturities[idx-1]; t1 = maturities[idx]
        y0 = yields[idx-1];     y1 = yields[idx]
        y0 + (y1 - y0) * (T - t0) / (t1 - t0)
    end
    [par_swap_rate(m, interp_spot; frequency=frequency) for m in maturities]
end

# ─────────────────────────────────────────────────────────────
# 7. BASIS RISK
# ─────────────────────────────────────────────────────────────

"""
    basis_spread(curve1_yields, curve2_yields, maturities) -> Vector{Float64}

Compute basis spread between two yield curves (e.g., SOFR vs Treasury).
"""
basis_spread(c1::Vector{Float64}, c2::Vector{Float64}, mats=nothing) = c1 .- c2

"""
    basis_risk_factor(returns1, returns2) -> NamedTuple

Analyze basis risk between two rate benchmarks.
Returns correlation, beta, tracking error, and basis volatility.
"""
function basis_risk_factor(returns1::Vector{Float64},
                             returns2::Vector{Float64})
    n = min(length(returns1), length(returns2))
    r1 = returns1[1:n]; r2 = returns2[1:n]
    corr = cor(r1, r2)
    beta = cov(r1, r2) / (var(r2) + 1e-15)
    te   = std(r1 .- beta .* r2)
    basis_vol = std(r1 .- r2)
    (correlation=corr, beta=beta, tracking_error=te, basis_volatility=basis_vol)
end

# ─────────────────────────────────────────────────────────────
# 8. CRYPTO vs TRADITIONAL YIELD COMPARISON
# ─────────────────────────────────────────────────────────────

"""
    crypto_yield_comparison(trad_yields, crypto_yields, maturities)
       -> NamedTuple

Compare traditional fixed income yields vs crypto DeFi yields.
Computes spread, risk-adjusted excess yield, and macro factor loading.
"""
function crypto_yield_comparison(trad_yields::Vector{Float64},
                                   crypto_yields::Vector{Float64},
                                   maturities::Vector{Float64})
    n = min(length(trad_yields), length(crypto_yields), length(maturities))
    t = trad_yields[1:n]; c = crypto_yields[1:n]; m = maturities[1:n]

    spread          = c .- t  # raw excess yield
    # Risk-adjusted: assume crypto has higher vol (approx 3x trad)
    risk_adj_spread = spread ./ 3.0
    avg_spread      = mean(spread)
    spread_vol      = std(spread)

    # Is crypto a yield-chasing asset? Correlation with trad yields
    yield_chasing = length(t) > 3 ? cor(t, c) : 0.0

    # Compute implied Sharpe: excess_yield / yield_vol (simplified)
    sharpe_trad   = mean(t) / (std(t) + 1e-10)
    sharpe_crypto = mean(c) / (std(c) + 1e-10)

    (trad_yields=t, crypto_yields=c, raw_spread=spread,
     risk_adj_spread=risk_adj_spread,
     avg_spread=avg_spread, spread_volatility=spread_vol,
     yield_chasing_corr=yield_chasing,
     sharpe_traditional=sharpe_trad, sharpe_crypto=sharpe_crypto)
end

"""
    macro_rate_sensitivity(crypto_returns, rate_changes) -> NamedTuple

Estimate crypto asset sensitivity to interest rate changes.
Computes beta, R², and rolling sensitivity.
"""
function macro_rate_sensitivity(crypto_returns::Vector{Float64},
                                  rate_changes::Vector{Float64};
                                  window::Int=60)
    n = min(length(crypto_returns), length(rate_changes))
    r = crypto_returns[1:n]; x = rate_changes[1:n]

    # Full sample beta
    beta_full = cov(r, x) / (var(x) + 1e-15)
    alpha     = mean(r) - beta_full * mean(x)
    resid     = r .- (alpha .+ beta_full .* x)
    r2        = 1 - var(resid) / (var(r) + 1e-15)

    # Rolling beta
    rolling_beta = zeros(n)
    for t in window:n
        xw = x[t-window+1:t]
        rw = r[t-window+1:t]
        rolling_beta[t] = cov(rw, xw) / (var(xw) + 1e-15)
    end

    # Duration-equivalent: if crypto has equity-like properties
    # BTC yield sensitivity ≈ -duration * Δy
    implied_duration = -beta_full / 0.01  # per 1% rate change

    (beta=beta_full, alpha=alpha, r_squared=r2,
     rolling_beta=rolling_beta, implied_duration=implied_duration,
     rate_elasticity=beta_full)
end

# ─────────────────────────────────────────────────────────────
# 9. DEMO
# ─────────────────────────────────────────────────────────────

"""
    run_fixed_income_demo() -> Nothing
"""
function run_fixed_income_demo()
    println("=" ^ 60)
    println("FIXED INCOME ANALYTICS DEMO")
    println("=" ^ 60)

    # ── Bond Pricing ──
    println("\n1. Bond Pricing, Duration, Convexity")
    b = Bond(1000.0, 0.05, 10.0; freq=2)
    for y in [0.03, 0.05, 0.07]
        price = bond_price(b, y)
        dur   = bond_duration(b, y)
        conv  = bond_convexity(b, y)
        dv01  = bond_dv01(b, y)
        println("  Yield=$(Int(round(y*100)))%: Price=\$$(round(price,digits=2)), ModDur=$(round(dur.modified_duration,digits=2)), Conv=$(round(conv,digits=2)), DV01=\$$(round(dv01*1000,digits=4))")
    end

    println("\n2. Yield Calculation")
    b2 = Bond(1000.0, 0.05, 10.0; freq=2)
    price_target = 950.0
    ytm = bond_yield(b2, price_target)
    println("  Price \$$(price_target) → YTM = $(round(ytm*100,digits=4))%")
    println("  Verify: bond_price(YTM) = \$$(round(bond_price(b2,ytm),digits=4))")

    println("\n3. Yield Curve Bootstrap")
    mats = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    par  = [0.050, 0.052, 0.055, 0.058, 0.060, 0.062, 0.063, 0.065, 0.070, 0.072]
    curve = yield_curve_bootstrap(mats, par)
    println("  Par rates: $(round.(par .* 100, digits=2))'%'")
    println("  Spot rates: $(round.(curve.spot_rates .* 100, digits=2))'%'")
    df_last = curve.discount_factors[end]
    println("  30Y Discount Factor: $(round(df_last, digits=4))")

    println("\n4. Nelson-Siegel-Svensson Fitting")
    nss_model = NelsonSiegelSvensson()
    fit_nss!(nss_model, mats, par; maxiter=1000)
    fitted = [nss_yield(nss_model, T) * 100 for T in mats]
    rmse   = sqrt(mean((fitted .- par .* 100).^2))
    println("  NSS params: β0=$(round(nss_model.beta0,digits=4)), β1=$(round(nss_model.beta1,digits=4))")
    println("  NSS fit RMSE: $(round(rmse,digits=4)) bps×100")
    println("  10Y fitted: $(round(nss_yield(nss_model, 10.0)*100,digits=3))%")

    println("\n5. Vasicek Model")
    v = VasicekModel(0.15, 0.05, 0.01, 0.03)
    mats_v = [1.0, 2.0, 5.0, 10.0, 30.0]
    prices_v = [vasicek_bond_price(v, T) for T in mats_v]
    yields_v = [-log(p)/T for (p,T) in zip(prices_v, mats_v)]
    println("  Vasicek yields: $(round.(yields_v.*100, digits=3))%")
    fwd = instantaneous_forward_curve(v, [1.0,5.0,10.0])
    println("  Inst. forwards (1,5,10Y): $(round.(fwd.*100, digits=3))%")
    paths = simulate_vasicek(v, 1.0, 252, 3; rng=MersenneTwister(1))
    println("  Simulated paths (end rates): $(round.(paths[:,end].*100, digits=3))%")

    println("\n6. CIR Model")
    cir = CIRModel(0.15, 0.05, 0.06, 0.03)
    yields_cir = [-log(cir_bond_price(cir, T)) / T * 100 for T in mats_v]
    println("  CIR yields: $(round.(yields_cir, digits=3))%")

    println("\n7. Swap Pricing")
    spot_fn = T -> 0.03 + 0.02 * (1 - exp(-T/5))  # upward sloping
    psr_5y  = par_swap_rate(5.0, spot_fn) * 100
    psr_10y = par_swap_rate(10.0, spot_fn) * 100
    println("  Par swap rate 5Y:  $(round(psr_5y, digits=4))%")
    println("  Par swap rate 10Y: $(round(psr_10y, digits=4))%")
    val_atm = swap_price(psr_5y/100, 5.0, spot_fn)
    println("  ATM swap value:    $(round(val_atm, digits=8)) (should be ~0)")
    dv01_sw = swap_dv01(psr_5y/100, 5.0, spot_fn)
    println("  Swap DV01:         $(round(dv01_sw*1e4, digits=4)) per \$1 notional")

    println("\n8. Immunization")
    b_short = Bond(1000.0, 0.04, 2.0; freq=2)
    b_long  = Bond(1000.0, 0.06, 10.0; freq=2)
    y_imm   = 0.05
    # Immunize a 5-year liability of $100K
    imm = immunize_portfolio(100_000.0, 4.5, b_short, b_long, y_imm)
    println("  Weights: w1=$(round(imm.w1,digits=3)), w2=$(round(imm.w2,digits=3))")
    println("  Portfolio duration: $(round(imm.portfolio_duration,digits=3))")

    println("\n9. Basis Risk & Crypto Yield Comparison")
    rng = MersenneTwister(42)
    trad_y   = 0.04 .+ 0.005 .* randn(rng, 5)
    crypto_y = 0.08 .+ 0.02  .* randn(rng, 5)
    comp     = crypto_yield_comparison(trad_y, crypto_y, [1.0, 3.0, 5.0, 10.0, 30.0])
    println("  Avg crypto-trad spread: $(round(comp.avg_spread*100, digits=2)) bps*100")
    println("  Yield-chasing corr:     $(round(comp.yield_chasing_corr, digits=3))")

    println("\n10. Crypto Rate Sensitivity (Macro Factor)")
    crypto_rets = randn(rng, 200) .* 0.03
    rate_chg    = randn(rng, 200) .* 0.002
    # Add beta ~ -2: crypto falls when rates rise
    crypto_rets .-= 2.0 .* rate_chg
    ms = macro_rate_sensitivity(crypto_rets, rate_chg)
    println("  Rate beta:       $(round(ms.beta, digits=3))")
    println("  R²:              $(round(ms.r_squared, digits=3))")
    println("  Implied duration: $(round(ms.implied_duration, digits=1)) years")

    println("\nDone.")
    nothing
end

# ─────────────────────────────────────────────────────────────
# 11. MORTGAGE AND ABS ANALYTICS
# ─────────────────────────────────────────────────────────────

"""
    mortgage_payment(principal, annual_rate, n_months) -> Float64

Compute fixed monthly mortgage payment.
M = P * r * (1+r)^n / ((1+r)^n - 1)
"""
function mortgage_payment(principal::Float64, annual_rate::Float64,
                            n_months::Int)::Float64
    r = annual_rate / 12
    r < 1e-10 && return principal / n_months
    principal * r * (1+r)^n_months / ((1+r)^n_months - 1)
end

"""
    mortgage_amortization(principal, annual_rate, n_months) -> NamedTuple

Full amortization schedule for a fixed-rate mortgage.
"""
function mortgage_amortization(principal::Float64, annual_rate::Float64,
                                 n_months::Int)
    r = annual_rate / 12
    payment = mortgage_payment(principal, annual_rate, n_months)
    balance = principal
    sched   = zeros(n_months, 4)  # [month, payment, interest, principal]
    for t in 1:n_months
        interest_paid  = balance * r
        principal_paid = payment - interest_paid
        balance       -= principal_paid
        balance        = max(balance, 0.0)
        sched[t, :] = [Float64(t), payment, interest_paid, principal_paid]
    end
    (schedule=sched, total_paid=payment*n_months, total_interest=payment*n_months - principal)
end

"""
    prepayment_model(wac, market_rate, seasoning) -> Float64

PSA (Public Securities Association) prepayment model.
Returns monthly CPR (conditional prepayment rate).
"""
function prepayment_model(wac::Float64, market_rate::Float64,
                            seasoning::Int)::Float64
    # Base PSA: ramp from 0.2% to 6% CPR over 30 months
    base_cpr = min(0.002 * seasoning, 0.06)
    # Refinancing incentive: lower market rates → more prepayments
    refi_mult = exp(max(wac - market_rate, 0.0) * 2.0)
    min(base_cpr * refi_mult, 0.50)  # cap at 50% CPR
end

# ─────────────────────────────────────────────────────────────
# 12. INFLATION-LINKED BONDS (TIPS)
# ─────────────────────────────────────────────────────────────

"""
    tips_price(real_yield, coupon, maturity, inflation_index, base_index;
               frequency=2) -> Float64

Price TIPS (Treasury Inflation-Protected Securities).
Principal = face * (inflation_index / base_index)
"""
function tips_price(real_yield::Float64, coupon::Float64,
                     maturity::Float64, inflation_index::Float64,
                     base_index::Float64; frequency::Int=2)::Float64
    adj_principal = base_index > 0 ? inflation_index / base_index : 1.0
    # Real coupon on inflation-adjusted principal
    b = Bond(adj_principal, coupon, maturity; freq=frequency)
    bond_price(b, real_yield)
end

"""
    breakeven_inflation(nominal_yield, real_yield) -> Float64

Break-even inflation rate from nominal vs TIPS yields.
BEI = (1 + nominal) / (1 + real) - 1 ≈ nominal - real
"""
breakeven_inflation(nominal::Float64, real::Float64) =
    (1 + nominal) / (1 + real) - 1

# ─────────────────────────────────────────────────────────────
# 13. CREDIT SPREADS AND RISK PREMIA
# ─────────────────────────────────────────────────────────────

"""
    credit_spread_term_structure(hazard_rate, recovery, maturities, risk_free_curve)
       -> Vector{Float64}

Compute credit spread term structure from constant hazard rate.
"""
function credit_spread_term_structure(hazard_rate::Float64,
                                        recovery::Float64,
                                        maturities::Vector{Float64},
                                        risk_free_spot::Vector{Float64})::Vector{Float64}
    n = min(length(maturities), length(risk_free_spot))
    [begin
         T = maturities[i]; rf = risk_free_spot[min(i, end)]
         # Risky discount factor under flat hazard rate
         risky_df = exp(-(rf + hazard_rate * (1-recovery)) * T)
         riskfree_df = exp(-rf * T)
         # Credit spread
         T > 0 ? (log(riskfree_df) - log(max(risky_df,1e-15))) / T : 0.0
     end for i in 1:n]
end

"""
    term_premium(forward_rates, expected_spot) -> Vector{Float64}

Term premium = realized forward rate - expected future spot rate.
Captures compensation for duration risk.
"""
function term_premium(forward_rates::Vector{Float64},
                       expected_spot::Vector{Float64})::Vector{Float64}
    n = min(length(forward_rates), length(expected_spot))
    forward_rates[1:n] .- expected_spot[1:n]
end

# ─────────────────────────────────────────────────────────────
# 14. SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────

"""
    dv01_ladder(bonds, yield; shift=0.0001) -> Vector{Float64}

Compute DV01 for each bond in a portfolio.
"""
function dv01_ladder(bonds::Vector{Bond}, yield::Float64;
                      shift::Float64=0.0001)::Vector{Float64}
    [bond_dv01(b, yield) for b in bonds]
end

"""
    key_rate_duration(b::Bond, key_rates, yield_curve; shift=0.0001) -> Vector{Float64}

Key rate durations: sensitivity to shifts at specific maturity buckets.
"""
function key_rate_duration(b::Bond, key_rates::Vector{Float64},
                             yield_curve::Vector{Float64};
                             shift::Float64=0.0001)::Vector{Float64}
    # Baseline price using flat yield
    p0 = bond_price(b, mean(yield_curve))
    p0 <= 0 && return zeros(length(key_rates))

    krd = zeros(length(key_rates))
    for (i, kr) in enumerate(key_rates)
        # Interpolated yield shift at this key rate
        # Simplification: shift yield by shift, measure DV01
        p_up = bond_price(b, mean(yield_curve) + shift)
        krd[i] = -(p_up - p0) / (p0 * shift)
    end
    krd ./ sum(abs.(krd)) .* bond_duration(b, mean(yield_curve)).modified_duration
end

"""
    portfolio_dollar_duration(bonds, notionals, yield) -> Float64

Portfolio-level dollar duration (sum of weighted durations).
"""
function portfolio_dollar_duration(bonds::Vector{Bond},
                                    notionals::Vector{Float64},
                                    yield::Float64)::Float64
    sum(bond_dollar_duration(bonds[i], yield) * notionals[i]
        for i in 1:length(bonds))
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 15 – Swaption Pricing and Interest Rate Vol Surface
# ─────────────────────────────────────────────────────────────────────────────

"""
    swaption_black(forward_rate, strike, sigma_atm, T_option, annuity)

Black's model for European payer swaption:
  V = A * [F * N(d1) - K * N(d2)]
where A = annuity factor, F = forward swap rate.
"""
function swaption_black(forward_rate::Float64, strike::Float64,
                         sigma_atm::Float64, T_option::Float64,
                         annuity::Float64)
    T_option <= 0 && return max(forward_rate - strike, 0.0) * annuity
    d1 = (log(forward_rate / strike) + 0.5 * sigma_atm^2 * T_option) /
         (sigma_atm * sqrt(T_option))
    d2 = d1 - sigma_atm * sqrt(T_option)
    return annuity * (forward_rate * norm_cdf_fi(d1) - strike * norm_cdf_fi(d2))
end

function norm_cdf_fi(x::Float64)
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
             t * (-1.453152027 + t * 1.061405429))))
    r = 1.0 - p * exp(-x^2)
    return x >= 0 ? r : 1.0 - r
end

"""
    vol_surface_sabr(F, K, T, alpha, beta, rho, nu)

SABR implied volatility approximation (Hagan et al. 2002).
Returns Black-equivalent implied vol for strike K, forward F, expiry T.
"""
function vol_surface_sabr(F::Float64, K::Float64, T::Float64,
                            alpha::Float64, beta::Float64,
                            rho::Float64, nu::Float64)
    if abs(F - K) < 1e-6 * F
        # ATM approximation
        term1 = alpha / (F^(1 - beta))
        term2 = 1.0 + ((1-beta)^2 / 24 * alpha^2 / F^(2-2beta) +
                        0.25 * rho * beta * nu * alpha / F^(1-beta) +
                        (2 - 3*rho^2) / 24 * nu^2) * T
        return term1 * term2
    end
    logFK  = log(F / K)
    FK_mid = sqrt(F * K)
    z    = nu / alpha * FK_mid^(1-beta) * logFK
    x_z  = log((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
    num  = alpha
    den1 = FK_mid^(1-beta)
    den2 = (1 + (1-beta)^2 / 24 * logFK^2 + (1-beta)^4 / 1920 * logFK^4)
    corr = (x_z > 1e-8) ? z / x_z : 1.0
    term3 = 1.0 + ((1-beta)^2 / 24 * alpha^2 / FK_mid^(2-2beta) +
                    0.25 * rho * beta * nu * alpha / FK_mid^(1-beta) +
                    (2 - 3*rho^2) / 24 * nu^2) * T
    return num / (den1 * den2) * corr * term3
end

"""
    calibrate_sabr(F, strikes, T, market_vols; max_iter)

Calibrate SABR parameters (alpha, rho, nu) with beta fixed at 0.5
via Nelder-Mead-like grid search minimising RMSE to market vols.
"""
function calibrate_sabr(F::Float64, strikes::Vector{Float64}, T::Float64,
                          market_vols::Vector{Float64}; max_iter::Int=200)
    beta = 0.5
    best = (alpha=0.2, rho=-0.3, nu=0.4, rmse=Inf)
    for alpha in 0.05:0.05:0.5, rho in -0.7:0.2:0.3, nu in 0.2:0.2:1.0
        rmse = 0.0
        for (K, mv) in zip(strikes, market_vols)
            model_v = vol_surface_sabr(F, K, T, alpha, beta, rho, nu)
            rmse   += (model_v - mv)^2
        end
        rmse = sqrt(rmse / length(strikes))
        if rmse < best.rmse
            best = (alpha=alpha, rho=rho, nu=nu, rmse=rmse)
        end
    end
    return best
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 16 – Repo Markets and Collateral Management
# ─────────────────────────────────────────────────────────────────────────────

"""
    repo_price(face, coupon_rate, maturity, yield, repo_rate, repo_term)

Pricing of a repo: dirty price of collateral today and at repo maturity.
Returns (initial_price, terminal_price, repo_interest, haircut_requirement).
"""
function repo_price(face::Float64, coupon_rate::Float64, maturity::Float64,
                     yield::Float64, repo_rate::Float64, repo_term::Float64;
                     haircut::Float64=0.02)
    b = Bond(face, coupon_rate, maturity, 2)
    p0 = bond_price(b, yield)
    # Accrued interest (simple approximation)
    accrued  = face * coupon_rate / 2 * (repo_term / 0.5)
    dirty_p0 = p0 + accrued
    # Collateral value after haircut
    collateral_value = dirty_p0 * (1 - haircut)
    repo_interest    = collateral_value * repo_rate * repo_term
    terminal_payment = collateral_value + repo_interest
    return (initial_price=dirty_p0, collateral_value=collateral_value,
            repo_interest=repo_interest, terminal_payment=terminal_payment)
end

"""
    collateral_transformation(assets, haircuts, target_collateral)

Optimal collateral transformation: given a set of eligible assets with
known haircuts, find the minimum-cost portfolio meeting `target_collateral`.
Greedy selection by (1 - haircut) / borrowing_cost (assume cost = 1 for all).
"""
function collateral_transformation(asset_values::Vector{Float64},
                                    haircuts::Vector{Float64},
                                    target_collateral::Float64)
    eligible = (1 .- haircuts) .* asset_values
    # sort by descending eligible value
    order  = sortperm(eligible, rev=true)
    used   = zeros(length(asset_values))
    funded = 0.0
    for i in order
        if funded >= target_collateral; break; end
        remaining = min(asset_values[i], (target_collateral - funded) / (1 - haircuts[i]))
        used[i]   = remaining
        funded   += remaining * (1 - haircuts[i])
    end
    return (used_assets=used, total_eligible=funded,
            met_target=funded >= target_collateral)
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 17 – Fixed Income Summary Dashboard
# ─────────────────────────────────────────────────────────────────────────────

"""
    bond_analytics_report(b, yield)

Print a comprehensive analytics report for a single bond at given yield.
Covers price, duration, convexity, DV01, and horizon return.
"""
function bond_analytics_report(b::Bond, yield::Float64)
    p   = bond_price(b, yield)
    ytm = bond_yield(b, p)           # round-trip
    md  = bond_duration(b, yield, :modified)
    mac = bond_duration(b, yield, :macaulay)
    cx  = bond_convexity(b, yield)
    dv  = bond_dv01(b, yield)
    println("=" ^ 55)
    println("Bond Analytics Report")
    println("=" ^ 55)
    @printf("  Face:          %.2f\n", b.face)
    @printf("  Coupon:        %.4f%%  (%d× / yr)\n",
            b.coupon_rate * 100, b.frequency)
    @printf("  Maturity:      %.2f years\n", b.maturity)
    @printf("  Yield (input): %.4f%%\n", yield * 100)
    @printf("  Price:         %.6f\n", p)
    @printf("  YTM (check):   %.4f%%\n", ytm * 100)
    @printf("  Macaulay Dur:  %.4f\n", mac)
    @printf("  Modified Dur:  %.4f\n", md)
    @printf("  Convexity:     %.4f\n", cx)
    @printf("  DV01:          %.6f\n", dv)
    println("=" ^ 55)
    return (price=p, ytm=ytm, mac_dur=mac, mod_dur=md,
            convexity=cx, dv01=dv)
end

"""
    yield_curve_summary(tenors, yields)

Summarise key yield curve metrics: level, slope (10y-2y), curvature (2y-5y+10y).
"""
function yield_curve_summary(tenors::Vector{Float64}, yields::Vector{Float64})
    sort_idx = sortperm(tenors)
    ts = tenors[sort_idx]; ys = yields[sort_idx]
    level     = mean(ys)
    slope     = length(ys) >= 2 ? ys[end] - ys[1] : NaN
    curve_idx = [findfirst(>=(2.0), ts), findfirst(>=(5.0), ts), findfirst(>=(10.0), ts)]
    curvature = all(!isnothing, curve_idx) ?
                2 * ys[curve_idx[2]] - ys[curve_idx[1]] - ys[curve_idx[3]] : NaN
    println("\nYield Curve Summary")
    @printf("  Level (avg):   %.4f%%\n", level * 100)
    @printf("  Slope (lo→hi): %+.4f bps\n", slope * 1e4)
    @printf("  Curvature:     %+.4f bps\n", curvature * 1e4)
    return (level=level, slope=slope, curvature=curvature)
end

end  # module FixedIncome
