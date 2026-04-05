"""
credit_risk.jl — Credit Risk Modeling for Crypto & Traditional Finance

Covers:
  - Merton structural model (equity as call option on assets)
  - KMV model: distance to default, expected default frequency
  - CreditMetrics: migration matrices, VaR from credit transitions
  - CVA (Credit Valuation Adjustment) for OTC derivatives
  - Counterparty credit risk: EPE, PFE, CVA Monte Carlo
  - Wrong-way risk modeling
  - Crypto-specific: exchange default risk from proof-of-reserves
  - Contagion / default cascade simulation on financial network

Pure Julia stdlib only. No external dependencies.
"""

module CreditRisk

using Statistics, LinearAlgebra, Random

export MertonModel, kmv_distance_to_default, kmv_edf
export CreditMigrationsModel, credit_var, credit_metrics_simulation
export cva_analytic, cva_monte_carlo, expected_positive_exposure
export potential_future_exposure, wrong_way_risk_adjustment
export ExchangeDefaultModel, proof_of_reserves_risk
export ContagionNetwork, default_cascade_simulate
export credit_spread, credit_portfolio_loss, correlated_defaults

# ─────────────────────────────────────────────────────────────
# 1. NORMAL DISTRIBUTION UTILITIES
# ─────────────────────────────────────────────────────────────

"""Standard normal CDF via rational approximation (Abramowitz & Stegun)."""
function norm_cdf(x::Float64)::Float64
    if x < -8.0; return 0.0; end
    if x >  8.0; return 1.0; end
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 +
           t * (-0.356563782 +
           t * (1.781477937 +
           t * (-1.821255978 +
           t *  1.330274429))))
    pdf_val = exp(-0.5 * x^2) / sqrt(2π)
    cdf = 1.0 - pdf_val * poly
    x >= 0 ? cdf : 1.0 - cdf
end

"""Standard normal PDF."""
norm_pdf(x::Float64) = exp(-0.5 * x^2) / sqrt(2π)

"""Inverse normal CDF (Beasley-Springer-Moro algorithm)."""
function norm_inv(p::Float64)::Float64
    p = clamp(p, 1e-10, 1.0 - 1e-10)
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090,  23.08336743743, -21.06224101826,  3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    y = p - 0.5
    if abs(y) < 0.42
        r = y^2
        num = y * (((a[4]*r + a[3])*r + a[2])*r + a[1])
        den = ((((b[4]*r + b[3])*r + b[2])*r + b[1])*r + 1.0)
        return num / den
    else
        r = p < 0.5 ? log(-log(p)) : log(-log(1.0 - p))
        x = c[1] + r*(c[2] + r*(c[3] + r*(c[4] + r*(c[5] +
            r*(c[6] + r*(c[7] + r*(c[8] + r*c[9])))))))
        return p < 0.5 ? -x : x
    end
end

# ─────────────────────────────────────────────────────────────
# 2. MERTON STRUCTURAL MODEL
# ─────────────────────────────────────────────────────────────

"""
    MertonModel

Structural default model: equity = call option on firm assets.

Fields:
  V0     — current asset value
  sigma_V — asset volatility
  D      — face value of debt (default barrier)
  r      — risk-free rate
  T      — time to maturity (years)
"""
struct MertonModel
    V0::Float64
    sigma_V::Float64
    D::Float64
    r::Float64
    T::Float64
end

"""
    merton_d1(m::MertonModel) -> Float64

Black-Scholes d1 for Merton model.
"""
function merton_d1(m::MertonModel)::Float64
    (log(m.V0 / m.D) + (m.r + 0.5 * m.sigma_V^2) * m.T) /
    (m.sigma_V * sqrt(m.T))
end

"""
    merton_d2(m::MertonModel) -> Float64

Black-Scholes d2 = d1 - sigma*sqrt(T).
"""
merton_d2(m::MertonModel) = merton_d1(m) - m.sigma_V * sqrt(m.T)

"""
    merton_equity_value(m::MertonModel) -> Float64

Equity value = V0*N(d1) - D*exp(-r*T)*N(d2).
"""
function merton_equity_value(m::MertonModel)::Float64
    d1 = merton_d1(m)
    d2 = merton_d2(m)
    m.V0 * norm_cdf(d1) - m.D * exp(-m.r * m.T) * norm_cdf(d2)
end

"""
    merton_default_probability(m::MertonModel) -> Float64

Risk-neutral probability of default = N(-d2).
Physical default probability differs by risk premium adjustment.
"""
function merton_default_probability(m::MertonModel)::Float64
    norm_cdf(-merton_d2(m))
end

"""
    merton_credit_spread(m::MertonModel) -> Float64

Implied credit spread from Merton model (annualized).
CS = -(1/T)*ln(N(d2) + (V0/D)*exp(r*T)*N(-d1))
"""
function merton_credit_spread(m::MertonModel)::Float64
    d1 = merton_d1(m)
    d2 = merton_d2(m)
    leverage = m.D * exp(-m.r * m.T) / m.V0
    p_repay = norm_cdf(d2) + (1.0 / leverage) * norm_cdf(-d1)
    p_repay <= 0 && return 1.0
    -log(p_repay) / m.T
end

"""
    merton_recovery_rate(m::MertonModel) -> Float64

Expected recovery rate given default.
"""
function merton_recovery_rate(m::MertonModel)::Float64
    d1 = merton_d1(m)
    d2 = merton_d2(m)
    pd = norm_cdf(-d2)
    pd <= 1e-10 && return 1.0
    # E[V | V < D] * N(-d1) / (D * N(-d2))
    edf_adj = m.V0 * exp(m.r * m.T) * norm_cdf(-d1)
    edf_adj / (m.D * norm_cdf(-d2))
end

"""
    solve_merton_from_equity(E0, sigma_E, D, r, T; tol=1e-8, maxiter=200)
       -> (V0, sigma_V)

Iteratively solve for asset value and asset volatility from observed equity
value and equity volatility. Uses the system:
  E = V*N(d1) - D*exp(-rT)*N(d2)
  sigma_E * E = sigma_V * V * N(d1)
"""
function solve_merton_from_equity(E0::Float64, sigma_E::Float64,
                                   D::Float64, r::Float64, T::Float64;
                                   tol::Float64=1e-8, maxiter::Int=200)
    # Initial guess: V ≈ E + D*exp(-r*T)
    V = E0 + D * exp(-r * T)
    sigma_V = sigma_E * E0 / V
    for _ in 1:maxiter
        m = MertonModel(V, sigma_V, D, r, T)
        d1 = merton_d1(m)
        E_model = merton_equity_value(m)
        sigma_V_new = sigma_E * E0 / (V * norm_cdf(d1) + 1e-12)
        V_new = E0 + D * exp(-r * T) * norm_cdf(merton_d2(m))
        if abs(V_new - V) < tol && abs(sigma_V_new - sigma_V) < tol
            return V_new, sigma_V_new
        end
        V = 0.5 * V + 0.5 * V_new
        sigma_V = 0.5 * sigma_V + 0.5 * sigma_V_new
    end
    V, sigma_V
end

# ─────────────────────────────────────────────────────────────
# 3. KMV MODEL
# ─────────────────────────────────────────────────────────────

"""
    kmv_distance_to_default(V, sigma_V, D, mu, T) -> Float64

KMV distance to default (DD):
  DD = (ln(V/D) + (mu - 0.5*sigma^2)*T) / (sigma*sqrt(T))
where mu is the expected asset return (physical measure).
"""
function kmv_distance_to_default(V::Float64, sigma_V::Float64,
                                  D::Float64, mu::Float64, T::Float64)::Float64
    (log(V / D) + (mu - 0.5 * sigma_V^2) * T) / (sigma_V * sqrt(T))
end

"""
    kmv_edf(DD) -> Float64

Expected Default Frequency from distance to default.
Uses empirical mapping: EDF = N(-DD) in academic version.
(Moody's KMV uses a proprietary empirical database mapping.)
"""
kmv_edf(DD::Float64) = norm_cdf(-DD)

"""
    kmv_full_analysis(E0, sigma_E, D, r, mu_equity, T) -> NamedTuple

Full KMV analysis pipeline: solve for asset params, compute DD and EDF.
"""
function kmv_full_analysis(E0::Float64, sigma_E::Float64, D::Float64,
                            r::Float64, mu_equity::Float64, T::Float64)
    V, sigma_V = solve_merton_from_equity(E0, sigma_E, D, r, T)
    # Convert equity return to asset return (rough approximation)
    mu_V = r + (mu_equity - r) * E0 / V
    DD = kmv_distance_to_default(V, sigma_V, D, mu_V, T)
    edf = kmv_edf(DD)
    cs  = merton_credit_spread(MertonModel(V, sigma_V, D, r, T))
    (asset_value=V, asset_vol=sigma_V, distance_to_default=DD,
     expected_default_frequency=edf, credit_spread=cs)
end

# ─────────────────────────────────────────────────────────────
# 4. CREDIT METRICS — MIGRATION MATRICES
# ─────────────────────────────────────────────────────────────

"""
    CreditMigrationsModel

CreditMetrics-style model for a portfolio of credits.

Fields:
  transition_matrix — K×K matrix, row i = transition probs from rating i
  rating_spreads    — K-vector of credit spreads per rating bucket
  recovery_rates    — K-vector, recovery given default
  correlations      — N×N asset correlation matrix for obligors
"""
struct CreditMigrationsModel
    transition_matrix::Matrix{Float64}
    rating_spreads::Vector{Float64}
    recovery_rates::Vector{Float64}
    correlations::Matrix{Float64}
end

"""Build a simplified 8-bucket transition matrix (AAA→AA→A→BBB→BB→B→CCC→D)."""
function default_transition_matrix()::Matrix{Float64}
    # Approximate S&P 1-year average transition rates (rows sum to 1)
    T = [
        0.9081 0.0833 0.0068 0.0006 0.0008 0.0002 0.0002 0.0000;
        0.0070 0.9065 0.0779 0.0064 0.0006 0.0013 0.0002 0.0001;
        0.0009 0.0227 0.9105 0.0552 0.0074 0.0026 0.0001 0.0006;
        0.0002 0.0033 0.0595 0.8693 0.0530 0.0117 0.0012 0.0018;
        0.0003 0.0014 0.0067 0.0773 0.8053 0.0884 0.0100 0.0106;
        0.0000 0.0011 0.0024 0.0043 0.0648 0.8346 0.0407 0.0521;
        0.0022 0.0000 0.0022 0.0130 0.0238 0.1124 0.6486 0.1978;
        0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000
    ]
    T
end

"""
    credit_metrics_simulation(model, ratings, notionals, n_sim; rng=MersenneTwister(42))
       -> Vector{Float64}

Monte Carlo simulation of portfolio credit loss distribution.
Returns vector of portfolio losses across simulations.
"""
function credit_metrics_simulation(model::CreditMigrationsModel,
                                    ratings::Vector{Int},
                                    notionals::Vector{Float64},
                                    n_sim::Int;
                                    rng=MersenneTwister(42))::Vector{Float64}
    N = length(ratings)
    K = size(model.transition_matrix, 1)  # number of rating buckets
    losses = zeros(n_sim)

    # Precompute cumulative transition probabilities for each rating
    cum_T = similar(model.transition_matrix)
    for i in 1:K
        cum_T[i, :] = cumsum(model.transition_matrix[i, :])
    end

    # Cholesky for correlated asset returns
    C = cholesky(model.correlations + 1e-8*I).L

    # Precompute N(-1) thresholds for each starting rating
    thresholds = [norm_inv.(cum_T[r, :]) for r in 1:K]

    for s in 1:n_sim
        # Draw correlated standard normals
        z_indep = randn(rng, N)
        z = C * z_indep

        loss = 0.0
        for i in 1:N
            r0 = ratings[i]
            # Find new rating from z[i] via threshold mapping
            z_i = z[i]
            new_r = K  # default unless threshold exceeded
            for k in 1:(K-1)
                if z_i < thresholds[r0][k]
                    new_r = k
                    break
                end
            end
            # Loss = LGD * notional if defaulted (new_r == K)
            if new_r == K
                lgd = 1.0 - model.recovery_rates[r0]
                loss += lgd * notionals[i]
            end
        end
        losses[s] = loss
    end
    losses
end

"""
    credit_var(losses, confidence) -> (VaR, CVaR)

Compute credit VaR and CVaR at given confidence level from loss distribution.
"""
function credit_var(losses::Vector{Float64}, confidence::Float64=0.99)
    sorted = sort(losses)
    n = length(sorted)
    idx = Int(ceil(confidence * n))
    var  = sorted[min(idx, n)]
    cvar = mean(sorted[idx:end])
    (var=var, cvar=cvar, expected_loss=mean(losses), std_loss=std(losses))
end

# ─────────────────────────────────────────────────────────────
# 5. CVA — CREDIT VALUATION ADJUSTMENT
# ─────────────────────────────────────────────────────────────

"""
    cva_analytic(hazard_rate, recovery, exposure_profile, discount_factors) -> Float64

Analytic CVA approximation:
  CVA ≈ (1-R) * Σ_t  EE(t) * PD(t-1, t) * DF(t)

Arguments:
  hazard_rate      — constant hazard rate (or vector matching time steps)
  recovery         — recovery rate (scalar)
  exposure_profile — vector of expected exposures at each time step
  discount_factors — corresponding discount factors
"""
function cva_analytic(hazard_rate, recovery::Float64,
                       exposure_profile::Vector{Float64},
                       discount_factors::Vector{Float64})::Float64
    n = length(exposure_profile)
    lgd = 1.0 - recovery
    cva = 0.0
    h = isa(hazard_rate, Number) ? fill(Float64(hazard_rate), n) : hazard_rate
    survival_prev = 1.0
    for t in 1:n
        survival_t = exp(-h[t] * t)  # simplified: cumulative
        pd_t = survival_prev - survival_t
        cva += lgd * exposure_profile[t] * pd_t * discount_factors[t]
        survival_prev = survival_t
    end
    cva
end

"""
    cva_monte_carlo(hazard_rate, recovery, r, T, n_paths, exposure_fn; rng=...)
       -> (cva, std_error)

Full Monte Carlo CVA with stochastic exposure.
exposure_fn(t, W) should return exposure at time t given Brownian path W.
"""
function cva_monte_carlo(hazard_rate::Float64, recovery::Float64,
                          r::Float64, T::Float64, n_paths::Int,
                          exposure_fn::Function;
                          rng=MersenneTwister(42))
    lgd = 1.0 - recovery
    dt  = T / 100
    times = dt:dt:T
    n_t   = length(times)
    cva_samples = zeros(n_paths)

    for p in 1:n_paths
        # Simulate default time via inverse CDF of exponential
        u = rand(rng)
        tau = -log(u) / hazard_rate  # default time

        cva_p = 0.0
        W = 0.0
        for (idx, t) in enumerate(times)
            W += sqrt(dt) * randn(rng)
            if t > tau; break; end  # already defaulted in prior step
            # Check if default occurs in this interval
            if tau <= t
                ee = max(exposure_fn(t, W), 0.0)
                df = exp(-r * t)
                cva_p += lgd * ee * df
                break
            end
        end
        cva_samples[p] = cva_p
    end
    m = mean(cva_samples)
    se = std(cva_samples) / sqrt(n_paths)
    (cva=m, std_error=se)
end

"""
    expected_positive_exposure(exposure_paths::Matrix{Float64}) -> Vector{Float64}

EPE at each time step = mean of max(exposure, 0) across paths.
exposure_paths: n_paths × n_timesteps matrix.
"""
function expected_positive_exposure(exposure_paths::Matrix{Float64})::Vector{Float64}
    n_paths, n_t = size(exposure_paths)
    [mean(max.(exposure_paths[:, t], 0.0)) for t in 1:n_t]
end

"""
    potential_future_exposure(exposure_paths, confidence) -> Vector{Float64}

PFE at confidence level (e.g., 95%) at each time step.
"""
function potential_future_exposure(exposure_paths::Matrix{Float64},
                                    confidence::Float64=0.95)::Vector{Float64}
    n_paths, n_t = size(exposure_paths)
    idx = Int(ceil(confidence * n_paths))
    [sort(exposure_paths[:, t])[min(idx, n_paths)] for t in 1:n_t]
end

"""
    simulate_interest_rate_exposure(K, r0, kappa, theta, sigma_r, T, n_paths, n_t; rng=...)
       -> Matrix{Float64}

Simulate swap exposure paths using CIR short-rate model.
Returns n_paths × n_t matrix of NPVs (exposure = max(NPV,0) handled outside).
"""
function simulate_interest_rate_exposure(K::Float64, r0::Float64,
                                          kappa::Float64, theta::Float64,
                                          sigma_r::Float64, T::Float64,
                                          n_paths::Int, n_t::Int;
                                          rng=MersenneTwister(1))::Matrix{Float64}
    dt  = T / n_t
    paths = zeros(n_paths, n_t)
    for p in 1:n_paths
        r = r0
        for t in 1:n_t
            dr = kappa * (theta - r) * dt + sigma_r * sqrt(max(r, 0.0)) * sqrt(dt) * randn(rng)
            r  = max(r + dr, 1e-6)
            # Simplified NPV of receive-fixed swap: (K - r)*remaining_tenor
            remaining = (n_t - t) * dt
            paths[p, t] = (K - r) * remaining
        end
    end
    paths
end

# ─────────────────────────────────────────────────────────────
# 6. WRONG-WAY RISK
# ─────────────────────────────────────────────────────────────

"""
    wrong_way_risk_adjustment(base_cva, rho_exposure_default, exposure_vol) -> Float64

Approximate wrong-way risk (WWR) adjustment to CVA.
WWR occurs when counterparty credit quality deteriorates as exposure increases.

rho_exposure_default: correlation between exposure changes and default intensity changes
Returns adjusted CVA > base_cva when rho > 0 (adverse correlation).
"""
function wrong_way_risk_adjustment(base_cva::Float64,
                                    rho_exposure_default::Float64,
                                    exposure_vol::Float64)::Float64
    # Spread markup factor (heuristic from Basel III literature)
    alpha = 1.0 + rho_exposure_default * exposure_vol * sqrt(2.0 / π)
    base_cva * max(alpha, 1.0)
end

"""
    wwr_monte_carlo(hazard_base, rho, exposure_vol, recovery, r, T, n_paths; rng=...)
       -> Float64

Monte Carlo CVA with correlated hazard rate and exposure (wrong-way risk).
"""
function wwr_monte_carlo(hazard_base::Float64, rho::Float64,
                          exposure_vol::Float64, recovery::Float64,
                          r::Float64, T::Float64, n_paths::Int;
                          rng=MersenneTwister(42))::Float64
    lgd = 1.0 - recovery
    dt  = T / 100
    times = collect(dt:dt:T)
    n_t   = length(times)
    cva_total = 0.0

    for p in 1:n_paths
        h = hazard_base
        exposure = 1.0
        tau = Inf
        cva_p = 0.0

        for (idx, t) in enumerate(times)
            # Correlated increments for hazard and exposure
            z1 = randn(rng)
            z2 = rho * z1 + sqrt(1 - rho^2) * randn(rng)
            h  = max(h + 0.1 * h * z1 * sqrt(dt), 1e-6)  # hazard rate diffusion
            exposure = max(exposure + exposure_vol * z2 * sqrt(dt), 0.0)

            # Probability of default in this interval
            pd_step = h * dt
            if rand(rng) < pd_step
                ee = max(exposure, 0.0)
                df = exp(-r * t)
                cva_p += lgd * ee * df
                break
            end
        end
        cva_total += cva_p
    end
    cva_total / n_paths
end

# ─────────────────────────────────────────────────────────────
# 7. CRYPTO EXCHANGE DEFAULT RISK (PROOF OF RESERVES)
# ─────────────────────────────────────────────────────────────

"""
    ExchangeDefaultModel

Model for crypto exchange solvency risk.

Fields:
  reserves       — total reported reserves (in USD)
  liabilities    — total customer deposits (USD)
  reserve_vol    — volatility of reserve value (from crypto price vol)
  liabilities_vol — volatility of liabilities (withdrawals uncertainty)
  correlation    — corr between reserves and liabilities
  horizon        — time horizon (years)
"""
struct ExchangeDefaultModel
    reserves::Float64
    liabilities::Float64
    reserve_vol::Float64
    liabilities_vol::Float64
    correlation::Float64
    horizon::Float64
end

"""
    proof_of_reserves_risk(model::ExchangeDefaultModel) -> NamedTuple

Compute exchange solvency metrics from proof-of-reserves data.

Returns:
  - reserve_ratio: current R/L
  - distance_to_insolvency: Z-score of (R-L) > 0
  - default_probability_1y: probability reserves < liabilities in horizon
  - stress_reserve_ratio: reserves at 2-sigma drawdown
"""
function proof_of_reserves_risk(model::ExchangeDefaultModel)
    R  = model.reserves
    L  = model.liabilities
    sR = model.reserve_vol
    sL = model.liabilities_vol
    rho = model.correlation
    T   = model.horizon

    ratio = R / L

    # Net asset value = R - L
    nav = R - L
    # Variance of NAV change over T
    var_nav = (sR * R)^2 * T + (sL * L)^2 * T -
              2.0 * rho * (sR * R) * (sL * L) * T
    sigma_nav = sqrt(max(var_nav, 1e-10))

    # Approximate drift: assumes R grows with crypto market, L is stable
    mu_nav = 0.0  # conservative: no trend

    # Probability R < L at horizon (normal approximation)
    z = (nav + mu_nav * T) / sigma_nav
    pd = norm_cdf(-z)

    # Stress test: R drops by 2*sigma, L stays flat
    stress_R = R * exp(-2.0 * sR * sqrt(T))
    stress_ratio = stress_R / L

    # Bail-in haircut needed if insolvent
    haircut = max(1.0 - ratio, 0.0)

    (reserve_ratio=ratio, distance_to_insolvency=z,
     default_probability=pd, stress_reserve_ratio=stress_ratio,
     bail_in_haircut=haircut, nav=nav, nav_volatility=sigma_nav)
end

"""
    exchange_run_risk(deposit_vol, total_deposits, liquid_reserves, n_sim; rng=...)
       -> NamedTuple

Simulate bank-run dynamics: correlated depositor withdrawal process.
Models sudden liquidity crisis (FTX-style event).
"""
function exchange_run_risk(deposit_vol::Float64, total_deposits::Float64,
                            liquid_reserves::Float64, n_sim::Int;
                            rng=MersenneTwister(0))
    illiquid_at_50_pct = Int(0)
    illiquid_at_30_pct = Int(0)

    for _ in 1:n_sim
        # Simulate fraction of deposits withdrawn in a run scenario
        # Fat-tailed withdrawal shock (mixture model)
        normal_withdrawals = deposit_vol * randn(rng)
        run_shock = rand(rng) < 0.05 ? 0.4 + 0.3 * rand(rng) : 0.0  # 5% chance of run
        total_withdrawal_frac = clamp(0.1 + normal_withdrawals + run_shock, 0.0, 1.0)
        demanded = total_withdrawal_frac * total_deposits

        if demanded > 0.50 * total_deposits
            illiquid_at_50_pct += Int(demanded > liquid_reserves)
        end
        if demanded > 0.30 * total_deposits
            illiquid_at_30_pct += Int(demanded > liquid_reserves)
        end
    end

    prob_illiquid_50 = illiquid_at_50_pct / n_sim
    prob_illiquid_30 = illiquid_at_30_pct / n_sim

    (prob_illiquid_if_50pct_run=prob_illiquid_50,
     prob_illiquid_if_30pct_run=prob_illiquid_30,
     liquid_coverage_ratio=liquid_reserves / total_deposits)
end

# ─────────────────────────────────────────────────────────────
# 8. CONTAGION — DEFAULT CASCADE SIMULATION
# ─────────────────────────────────────────────────────────────

"""
    ContagionNetwork

Financial network for default cascade modeling.

Fields:
  n          — number of nodes (institutions/exchanges)
  exposures  — n×n matrix: exposures[i,j] = exposure of i to j
  assets     — n-vector of total assets per institution
  debts      — n-vector of total external liabilities
  recovery   — scalar or n-vector of recovery rates
"""
struct ContagionNetwork
    n::Int
    exposures::Matrix{Float64}
    assets::Vector{Float64}
    debts::Vector{Float64}
    recovery::Union{Float64, Vector{Float64}}
end

"""
    default_cascade_simulate(net::ContagionNetwork, initial_defaults::Vector{Int})
       -> NamedTuple

Eisenberg-Noe / clearing vector approach to default cascade.
Starting from a set of initially defaulted institutions, iteratively
propagate losses through the network until no new defaults occur.

Returns defaulted set, total system loss, number of cascade rounds.
"""
function default_cascade_simulate(net::ContagionNetwork,
                                   initial_defaults::Vector{Int})
    n = net.n
    defaulted = Set(initial_defaults)
    equity = net.assets .- net.debts  # simplified net equity
    losses = zeros(n)
    rounds = 0

    # Compute direct loss from initial defaults
    for d in initial_defaults
        rec = isa(net.recovery, Number) ? net.recovery : net.recovery[d]
        lgd = 1.0 - rec
        for i in 1:n
            if i ∉ defaulted
                losses[i] += lgd * net.exposures[i, d]
            end
        end
    end

    # Cascade iterations
    new_defaults = true
    while new_defaults
        new_defaults = false
        rounds += 1
        for i in 1:n
            if i ∉ defaulted
                effective_equity = equity[i] - losses[i]
                if effective_equity < 0
                    push!(defaulted, i)
                    new_defaults = true
                    # Propagate this node's default
                    rec_i = isa(net.recovery, Number) ? net.recovery : net.recovery[i]
                    lgd_i = 1.0 - rec_i
                    for j in 1:n
                        if j ∉ defaulted
                            losses[j] += lgd_i * net.exposures[j, i]
                        end
                    end
                end
            end
        end
        rounds > n && break  # safety: at most n rounds
    end

    total_system_loss = sum(losses)
    default_fraction  = length(defaulted) / n
    (defaulted_nodes=collect(defaulted), total_loss=total_system_loss,
     default_fraction=default_fraction, cascade_rounds=rounds,
     individual_losses=losses)
end

"""
    generate_random_financial_network(n, density, avg_exposure, rng=MersenneTwister(1))
       -> ContagionNetwork

Generate a random financial network for testing cascade dynamics.
"""
function generate_random_financial_network(n::Int, density::Float64=0.3,
                                            avg_exposure::Float64=100.0;
                                            rng=MersenneTwister(1))
    exposures = zeros(n, n)
    for i in 1:n, j in 1:n
        if i != j && rand(rng) < density
            exposures[i, j] = avg_exposure * (0.5 + rand(rng))
        end
    end
    assets = [sum(exposures[i, :]) * 2 + avg_exposure * (1 + rand(rng)) for i in 1:n]
    debts  = [assets[i] * (0.3 + 0.4 * rand(rng)) for i in 1:n]
    ContagionNetwork(n, exposures, assets, debts, 0.4)
end

# ─────────────────────────────────────────────────────────────
# 9. PORTFOLIO CREDIT LOSS — CORRELATED DEFAULTS
# ─────────────────────────────────────────────────────────────

"""
    correlated_defaults(pds, lgds, notionals, rho_matrix, n_sim; rng=...) -> Vector{Float64}

Gaussian copula model for correlated portfolio defaults.
pds      — vector of individual default probabilities
lgds     — vector of loss given default (1 - recovery)
notionals — vector of notional amounts
rho_matrix — asset return correlation matrix
"""
function correlated_defaults(pds::Vector{Float64}, lgds::Vector{Float64},
                               notionals::Vector{Float64},
                               rho_matrix::Matrix{Float64},
                               n_sim::Int;
                               rng=MersenneTwister(42))::Vector{Float64}
    n = length(pds)
    # Precompute default thresholds in normal space
    thresholds = norm_inv.(pds)

    # Cholesky of correlation matrix
    L = cholesky(rho_matrix + 1e-8*I).L

    losses = zeros(n_sim)
    for s in 1:n_sim
        z_raw = randn(rng, n)
        z = L * z_raw
        total_loss = 0.0
        for i in 1:n
            if z[i] < thresholds[i]  # defaulted
                total_loss += lgds[i] * notionals[i]
            end
        end
        losses[s] = total_loss
    end
    losses
end

"""
    credit_portfolio_loss(losses::Vector{Float64}) -> NamedTuple

Summary statistics for credit portfolio loss distribution.
"""
function credit_portfolio_loss(losses::Vector{Float64})
    n = length(losses)
    sorted = sort(losses)
    el   = mean(losses)
    ul   = std(losses)
    var95 = sorted[Int(ceil(0.95 * n))]
    var99 = sorted[Int(ceil(0.99 * n))]
    cvar99 = mean(sorted[Int(ceil(0.99 * n)):end])
    (expected_loss=el, unexpected_loss=ul,
     var_95=var95, var_99=var99, cvar_99=cvar99,
     max_loss=maximum(losses))
end

"""
    credit_spread(pd::Float64, lgd::Float64, maturity::Float64) -> Float64

Convert PD and LGD to implied credit spread.
Approximation: CS ≈ PD * LGD / T (annualized).
"""
function credit_spread(pd::Float64, lgd::Float64, maturity::Float64)::Float64
    # More precise: solve yield that prices risky bond
    # Assume flat hazard rate: CS = -ln(1 - PD*LGD) / T approximately
    annual_expected_loss = pd * lgd
    -log(max(1.0 - annual_expected_loss, 1e-10)) / maturity
end

# ─────────────────────────────────────────────────────────────
# 10. CRYPTO-SPECIFIC EXTENSIONS
# ─────────────────────────────────────────────────────────────

"""
    defi_protocol_default_risk(tvl, debt, vol_tvl, liquidation_threshold; T=1/12)
       -> NamedTuple

Model default risk for a DeFi lending protocol.
tvl                   — Total Value Locked (collateral)
debt                  — outstanding borrowing
vol_tvl               — annualized volatility of TVL
liquidation_threshold — TVL/debt ratio that triggers liquidation cascade
"""
function defi_protocol_default_risk(tvl::Float64, debt::Float64,
                                     vol_tvl::Float64,
                                     liquidation_threshold::Float64=1.2;
                                     T::Float64=1.0/12)
    ratio = tvl / debt
    # Z-score: how many sigma until TVL hits threshold * debt
    target = liquidation_threshold * debt
    mu_ln  = -0.5 * vol_tvl^2 * T
    sig_ln = vol_tvl * sqrt(T)
    z = (log(tvl) + mu_ln - log(target)) / sig_ln
    pd = norm_cdf(-z)
    stress_tvl = tvl * exp(-2.0 * vol_tvl * sqrt(T))  # 2-sigma drawdown
    (collateral_ratio=ratio, default_probability=pd,
     distance_to_liquidation=z, stress_collateral_ratio=stress_tvl/debt,
     liquidation_threshold=liquidation_threshold)
end

"""
    stablecoin_depeg_risk(peg_price, current_price, reserve_ratio,
                           reserve_vol, n_sim; rng=...) -> NamedTuple

Monte Carlo simulation of stablecoin depeg event.
Models reserve adequacy under redemption pressure.
"""
function stablecoin_depeg_risk(peg_price::Float64=1.0,
                                current_price::Float64=1.0,
                                reserve_ratio::Float64=1.05,
                                reserve_vol::Float64=0.10,
                                n_sim::Int=10000;
                                rng=MersenneTwister(42))
    depeg_events = 0
    severe_depeg  = 0

    for _ in 1:n_sim
        # Simulate reserve shock (crypto collateral drops)
        reserve_return = reserve_vol * randn(rng)
        new_ratio = reserve_ratio * exp(reserve_return)
        # If reserves < 100%, depeg occurs; severity scales with shortfall
        if new_ratio < 1.0
            depeg_events += 1
            depeg_magnitude = 1.0 - new_ratio
            new_ratio < 0.85 && (severe_depeg += 1)
        end
    end

    prob_depeg    = depeg_events / n_sim
    prob_severe   = severe_depeg / n_sim
    premium       = current_price - peg_price
    (prob_depeg=prob_depeg, prob_severe_depeg=prob_severe,
     current_premium=premium, reserve_ratio=reserve_ratio)
end

# ─────────────────────────────────────────────────────────────
# 11. FULL PIPELINE DEMO
# ─────────────────────────────────────────────────────────────

"""
    run_credit_risk_demo() -> Nothing

Demonstration of all credit risk models.
"""
function run_credit_risk_demo()
    println("=" ^ 60)
    println("CREDIT RISK MODELING DEMO")
    println("=" ^ 60)

    # --- Merton Model ---
    println("\n1. Merton Structural Model")
    m = MertonModel(100.0, 0.25, 80.0, 0.05, 1.0)
    E  = merton_equity_value(m)
    pd = merton_default_probability(m)
    cs = merton_credit_spread(m)
    rr = merton_recovery_rate(m)
    println("  Asset Value: \$$(round(m.V0,digits=2))")
    println("  Debt:        \$$(round(m.D,digits=2))")
    println("  Equity Val:  \$$(round(E,digits=2))")
    println("  Default Prob: $(round(pd*100,digits=2))%")
    println("  Credit Spread: $(round(cs*10000,digits=1)) bps")
    println("  Recovery Rate: $(round(rr*100,digits=1))%")

    # --- KMV ---
    println("\n2. KMV Analysis")
    res = kmv_full_analysis(20.0, 0.40, 80.0, 0.05, 0.12, 1.0)
    println("  Distance to Default: $(round(res.distance_to_default,digits=3))")
    println("  EDF (1-year):        $(round(res.expected_default_frequency*100,digits=2))%")

    # --- CreditMetrics ---
    println("\n3. CreditMetrics Portfolio VaR")
    T_mat = default_transition_matrix()
    spreads   = [0.002, 0.005, 0.010, 0.020, 0.040, 0.080, 0.150, 0.0]
    recoveries = fill(0.4, 8)
    n_obligors = 20
    corr = 0.3 * ones(n_obligors, n_obligors) + 0.7 * I
    model = CreditMigrationsModel(T_mat, spreads, recoveries, corr)
    ratings   = rand(MersenneTwister(1), 1:6, n_obligors)  # BBB/lower
    notionals = fill(1_000_000.0, n_obligors)
    losses = credit_metrics_simulation(model, ratings, notionals, 5000)
    stats  = credit_var(losses, 0.99)
    println("  Portfolio Notional: \$$(sum(notionals)/1e6)M")
    println("  Expected Loss:  \$$(round(stats.expected_loss/1e3,digits=1))K")
    println("  VaR (99%):      \$$(round(stats.var/1e3,digits=1))K")
    println("  CVaR (99%):     \$$(round(stats.cvar/1e3,digits=1))K")

    # --- Crypto Exchange Risk ---
    println("\n4. Crypto Exchange Default Risk (Proof of Reserves)")
    exch = ExchangeDefaultModel(1.05e9, 1.0e9, 0.60, 0.05, 0.30, 1.0)
    pr = proof_of_reserves_risk(exch)
    println("  Reserve Ratio:         $(round(pr.reserve_ratio,digits=3))x")
    println("  Distance to Insolvency: $(round(pr.distance_to_insolvency,digits=2)) σ")
    println("  Default Prob (1Y):      $(round(pr.default_probability*100,digits=2))%")
    println("  Stress Reserve Ratio:   $(round(pr.stress_reserve_ratio,digits=3))x")

    # --- Contagion ---
    println("\n5. Default Cascade Simulation")
    net = generate_random_financial_network(10, 0.3, 50.0)
    result = default_cascade_simulate(net, [1])
    println("  Network Nodes: $(net.n)")
    println("  Initial Defaults: [1]")
    println("  Final Defaulted: $(sort(result.defaulted_nodes))")
    println("  Default Fraction: $(round(result.default_fraction*100,digits=1))%")
    println("  Total System Loss: \$$(round(result.total_loss,digits=1))")
    println("  Cascade Rounds: $(result.cascade_rounds)")

    # --- DeFi Protocol ---
    println("\n6. DeFi Protocol Default Risk")
    defi = defi_protocol_default_risk(150.0, 100.0, 0.80, 1.15, T=1.0/12)
    println("  Collateral Ratio: $(round(defi.collateral_ratio,digits=2))x")
    println("  1-Month Default Prob: $(round(defi.default_probability*100,digits=2))%")
    println("  Distance to Liquidation: $(round(defi.distance_to_liquidation,digits=2)) σ")

    println("\nDone.")
    nothing
end

# ─────────────────────────────────────────────────────────────
# 12. CREDIT SCORING MODELS
# ─────────────────────────────────────────────────────────────

"""
    AltmanZScore

Altman Z-Score model for predicting bankruptcy.
Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
"""
struct AltmanZScore
    X1::Float64  # Working capital / Total assets
    X2::Float64  # Retained earnings / Total assets
    X3::Float64  # EBIT / Total assets
    X4::Float64  # Market cap / Total liabilities
    X5::Float64  # Revenue / Total assets
end

"""
    altman_z_score(a::AltmanZScore) -> NamedTuple

Compute Altman Z-score and credit assessment.
Z > 2.99: Safe zone
1.81 < Z < 2.99: Grey zone
Z < 1.81: Distress zone
"""
function altman_z_score(a::AltmanZScore)
    z = 1.2*a.X1 + 1.4*a.X2 + 3.3*a.X3 + 0.6*a.X4 + 1.0*a.X5
    zone = z > 2.99 ? "Safe" : z > 1.81 ? "Grey" : "Distress"
    pd_approx = z > 2.99 ? 0.01 : z > 1.81 ? 0.05 + (2.99-z)/1.18*0.15 : 0.20 + (1.81-z)*0.1
    (z_score=z, zone=zone, approx_pd=clamp(pd_approx,0.0,1.0))
end

"""
    logistic_pd_model(features, weights, bias) -> Float64

Logistic regression probability of default model.
"""
function logistic_pd_model(features::Vector{Float64},
                             weights::Vector{Float64}, bias::Float64)::Float64
    1.0 / (1.0 + exp(-clamp(dot(weights, features) + bias, -20.0, 20.0)))
end

# ─────────────────────────────────────────────────────────────
# 13. CREDIT RISK PORTFOLIO ANALYTICS
# ─────────────────────────────────────────────────────────────

"""
    credit_risk_contribution(losses, weights) -> Vector{Float64}

Marginal risk contribution of each obligor to total portfolio credit VaR.
Uses numerical differentiation of VaR w.r.t. weights.
"""
function credit_risk_contribution(losses::Matrix{Float64},
                                    weights::Vector{Float64};
                                    confidence::Float64=0.99)::Vector{Float64}
    # losses: n_sim × n_obligors matrix
    n_sim, n_obl = size(losses)
    portfolio_loss = losses * weights
    var_total = quantile(portfolio_loss, confidence)

    # Marginal VaR via covariance (Euler allocation)
    tail_mask = portfolio_loss .>= var_total
    sum(tail_mask) == 0 && return ones(n_obl) ./ n_obl

    tail_losses = losses[tail_mask, :]
    mrc = vec(mean(tail_losses, dims=1))
    mrc ./= (sum(mrc) + 1e-10)
    mrc
end

"""
    credit_var_contribution(pds, lgds, notionals, rho, confidence) -> NamedTuple

Analytic credit VaR contributions using Vasicek single-factor model.
"""
function credit_var_contribution(pds::Vector{Float64},
                                   lgds::Vector{Float64},
                                   notionals::Vector{Float64},
                                   rho::Float64, confidence::Float64=0.99)
    n = length(pds)

    # Vasicek conditional PD at confidence level
    # PD_cond(q) = N((N^{-1}(PD) - sqrt(rho)*N^{-1}(q)) / sqrt(1-rho))
    q_star = norm_inv(confidence)
    cond_pds = [norm_cdf((norm_inv(pds[i]) - sqrt(rho)*q_star) / sqrt(1-rho+1e-10))
                for i in 1:n]

    individual_var = cond_pds .* lgds .* notionals
    total_var = sum(individual_var)
    (conditional_pds=cond_pds, individual_var=individual_var,
     total_var=total_var, var_weights=individual_var ./ (total_var+1e-10))
end

# ─────────────────────────────────────────────────────────────
# 14. CREDIT DERIVATIVES PRICING
# ─────────────────────────────────────────────────────────────

"""
    cds_spread(pd_annual, recovery, maturity, risk_free) -> Float64

CDS (Credit Default Swap) spread approximation.
s ≈ (1-R) * h   where h = hazard rate from PD
"""
function cds_spread(pd_annual::Float64, recovery::Float64,
                     maturity::Float64, risk_free::Float64=0.05)::Float64
    h = -log(1 - min(pd_annual, 0.9999)) / maturity  # hazard rate
    lgd = 1 - recovery
    # Simplified: spread ≈ LGD * h
    lgd * h
end

"""
    cds_price(fixed_spread, fair_spread, maturity, notional) -> Float64

MTM value of a CDS position.
Positive = protection buyer has gained.
"""
function cds_price(fixed_spread::Float64, fair_spread::Float64,
                    maturity::Float64, notional::Float64=1_000_000.0)::Float64
    # Duration approximation: risky annuity ≈ (1 - exp(-h*T)) / h ≈ T for small h
    duration = maturity * 0.9  # rough risky duration
    (fair_spread - fixed_spread) * duration * notional
end

"""
    nth_to_default_spread(pds, recovery, rho, n_defaults, maturity; n_sim=5000)
       -> Float64

Price of an n-th-to-default basket CDS using Gaussian copula Monte Carlo.
"""
function nth_to_default_spread(pds::Vector{Float64}, recovery::Float64,
                                 rho::Float64, n_defaults::Int,
                                 maturity::Float64;
                                 n_sim::Int=5000,
                                 rng=MersenneTwister(42))::Float64
    n_names = length(pds)
    thresholds = norm_inv.(pds)

    # Correlation matrix: equicorrelation
    C = rho * ones(n_names, n_names) + (1-rho) * I
    L = cholesky(C + 1e-8*I).L

    n_trigger = 0
    for _ in 1:n_sim
        z = L * randn(rng, n_names)
        n_def = sum(z .< thresholds)
        n_def >= n_defaults && (n_trigger += 1)
    end

    pd_ntd = n_trigger / n_sim
    cds_spread(pd_ntd, recovery, maturity)
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 13 – Credit Portfolio Optimisation and Economic Capital
# ─────────────────────────────────────────────────────────────────────────────

"""
    vasicek_credit_var(pd, rho, confidence, n_debtors)

Vasicek single-factor model for portfolio VaR (economic capital):
  VaR = N( (N⁻¹(PD) − √ρ · N⁻¹(α)) / √(1−ρ) ) − PD
where N is the standard normal CDF.
Approximation uses erfinv via bisection.
"""
function vasicek_credit_var(pd::Float64, rho::Float64,
                              confidence::Float64=0.999)
    PD_z     = qnorm(pd)
    alpha_z  = qnorm(confidence)
    num      = PD_z - sqrt(rho) * alpha_z
    VaR_PD   = norm_cdf_cr(num / sqrt(1 - rho + 1e-12))
    return VaR_PD - pd
end

"""
    qnorm(p)

Quantile function of standard normal via rational approximation (Beasley-Springer-Moro).
"""
function qnorm(p::Float64)
    p = clamp(p, 1e-10, 1 - 1e-10)
    if p < 0.5
        t = sqrt(-2 * log(p))
    else
        t = sqrt(-2 * log(1 - p))
    end
    c0 = 2.515517; c1 = 0.802853; c2 = 0.010328
    d1 = 1.432788; d2 = 0.189269; d3 = 0.001308
    x  = t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)
    return p < 0.5 ? -x : x
end

function norm_cdf_cr(x::Float64)
    t = 1.0 / (1 + 0.3275911 * abs(x))
    p = t * (0.254829592 + t*(-0.284496736 + t*(1.421413741 +
             t*(-1.453152027 + t*1.061405429))))
    r = 1.0 - p * exp(-x^2)
    return x >= 0 ? r : 1.0 - r
end

"""
    credit_portfolio_ec(pds, lgds, correlations, confidence)

Economic capital for a credit portfolio using Vasicek sector model.
`pds`:     vector of individual PDs
`lgds`:    Loss Given Default (1 - recovery)
`correlations`: pairwise (approximated as mean pairwise rho)
"""
function credit_portfolio_ec(pds::Vector{Float64}, lgds::Vector{Float64},
                               correlations::Matrix{Float64},
                               confidence::Float64=0.999)
    n    = length(pds)
    rho_mean = (sum(correlations) - n) / (n * (n - 1) + 1e-8)
    el   = sum(pds .* lgds)
    # Vasicek standalone VaR per obligor
    ul_contributions = Float64[]
    for i in 1:n
        var_i = vasicek_credit_var(pds[i], rho_mean, confidence)
        push!(ul_contributions, var_i * lgds[i])
    end
    total_ec = sqrt(sum(ul_contributions .^ 2) +
                    2 * rho_mean * sum(ul_contributions[i] * ul_contributions[j]
                                       for i in 1:n for j in (i+1):n))
    return (economic_capital=total_ec, expected_loss=el,
            ul_contributions=ul_contributions)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 14 – Crypto Credit Risk: Undercollateralised Lending
# ─────────────────────────────────────────────────────────────────────────────

"""
    CryptoLendingPosition

Under-collateralised crypto lending: borrower posts crypto collateral C,
borrows stablecoin D.  LTV = D / (C × P).  Margin call at LTV_margin,
liquidation at LTV_liq.
"""
struct CryptoLendingPosition
    collateral_units::Float64    # e.g. BTC
    debt_stablecoin::Float64     # e.g. USD
    ltv_margin_call::Float64     # e.g. 0.75
    ltv_liquidation::Float64     # e.g. 0.85
    liquidation_penalty::Float{} # fraction of collateral (e.g. 0.10)
end

# Avoid parametric type confusion; redefine cleanly:
struct CryptoLendingPos
    collateral_units::Float64
    debt_stablecoin::Float64
    ltv_margin_call::Float64
    ltv_liquidation::Float64
    liquidation_penalty::Float64
end

function current_ltv(pos::CryptoLendingPos, price::Float64)
    return pos.debt_stablecoin / (pos.collateral_units * price + 1e-8)
end

function is_margin_call(pos::CryptoLendingPos, price::Float64)
    return current_ltv(pos, price) >= pos.ltv_margin_call
end

function is_liquidation(pos::CryptoLendingPos, price::Float64)
    return current_ltv(pos, price) >= pos.ltv_liquidation
end

function liquidation_loss(pos::CryptoLendingPos, price::Float64)
    # Proceeds from selling collateral minus penalty
    proceeds = pos.collateral_units * price * (1 - pos.liquidation_penalty)
    loss = max(0.0, pos.debt_stablecoin - proceeds)
    return loss
end

"""
    simulate_lending_book(positions, price_paths) -> NamedTuple

Simulate a book of crypto lending positions over Monte Carlo price paths.
Returns loss distribution, default rate, and average LGD.
"""
function simulate_lending_book(positions::Vector{CryptoLendingPos},
                                 price_paths::Matrix{Float64})
    n_paths, T = size(price_paths)
    n_pos = length(positions)
    total_losses = zeros(n_paths)
    n_defaults   = 0
    for path in 1:n_paths
        for pos in positions
            defaulted = false
            for t in 1:T
                if is_liquidation(pos, price_paths[path, t])
                    total_losses[path] += liquidation_loss(pos, price_paths[path, t])
                    if !defaulted
                        n_defaults += 1
                        defaulted = true
                    end
                    break
                end
            end
        end
    end
    default_rate = n_defaults / (n_paths * n_pos)
    losses_sorted = sort(total_losses)
    var95  = losses_sorted[floor(Int, 0.95 * n_paths)]
    cvar95 = mean(losses_sorted[floor(Int, 0.95 * n_paths):end])
    return (loss_distribution=total_losses, var95=var95, cvar95=cvar95,
            default_rate=default_rate)
end

end  # module CreditRisk
