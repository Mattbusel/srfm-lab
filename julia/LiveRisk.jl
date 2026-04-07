"""
    LiveRisk

Real-time portfolio risk management for the SRFM quantitative trading system.
Implements parametric and historical VaR, CVaR, marginal VaR, stress testing,
liquidity-adjusted VaR, fixed-income DV01/duration, and a Python risk API bridge.
"""
module LiveRisk

using LinearAlgebra
using Statistics
using Distributions

export ewma_covariance, parametric_var, historical_var, cornish_fisher_var
export expected_shortfall, bootstrap_es_bands
export marginal_var, component_var
export stress_test_2008, stress_test_covid2020, stress_test_crypto2022
export apply_stress_scenario
export liquidity_adjusted_var, amihud_illiquidity
export dv01_bond, modified_duration, macaulay_duration
export portfolio_greeks_summary
export RiskReport, build_risk_report

# ---------------------------------------------------------------------------
# EWMA covariance estimation
# ---------------------------------------------------------------------------

"""
    ewma_covariance(returns; lambda=0.94)

Compute an exponentially weighted covariance matrix (RiskMetrics style).

# Arguments
- `returns` : (T x N) matrix of asset returns, T observations, N assets
- `lambda`  : decay factor (0.94 is RiskMetrics daily standard)

Returns the (N x N) EWMA covariance matrix.
"""
function ewma_covariance(returns::Matrix{Float64}; lambda::Real=0.94)::Matrix{Float64}
    T, N = size(returns)
    T < 2 && error("Need at least 2 observations")

    # Initialize with sample covariance of first 20 obs
    init_end = min(20, T)
    Sigma = cov(returns[1:init_end, :])

    for t in (init_end + 1):T
        r_t = returns[t, :]
        Sigma = lambda * Sigma + (1.0 - lambda) * (r_t * r_t')
    end
    return Sigma
end

"""
    ewma_variance(returns; lambda=0.94)

Compute exponentially weighted variance for a single asset return series.
"""
function ewma_variance(returns::Vector{Float64}; lambda::Real=0.94)::Float64
    T = length(returns)
    sigma2 = var(returns[1:min(20, T)])
    for t in (min(21, T+1)):T
        sigma2 = lambda * sigma2 + (1.0 - lambda) * returns[t]^2
    end
    return sigma2
end

# ---------------------------------------------------------------------------
# Parametric VaR
# ---------------------------------------------------------------------------

"""
    parametric_var(weights, returns, confidence; lambda=0.94, holding_period=1)

Compute parametric VaR under the assumption of normally distributed returns
with an EWMA covariance matrix.

# Arguments
- `weights`       : vector of portfolio weights (sum to 1 or in dollar terms)
- `returns`       : (T x N) matrix of asset returns
- `confidence`    : confidence level (e.g., 0.99 for 99% VaR)
- `lambda`        : EWMA decay factor
- `holding_period`: scaling period in days

Returns the VaR as a positive number (loss).
"""
function parametric_var(weights::Vector{Float64}, returns::Matrix{Float64},
                        confidence::Real; lambda::Real=0.94,
                        holding_period::Int=1)::Float64
    Sigma = ewma_covariance(returns; lambda=lambda)
    port_var = dot(weights, Sigma * weights)
    port_vol = sqrt(port_var * holding_period)
    z = quantile(Normal(), confidence)
    return z * port_vol
end

# ---------------------------------------------------------------------------
# Historical simulation VaR
# ---------------------------------------------------------------------------

"""
    historical_var(weights, returns, confidence; holding_period=1)

Compute historical simulation VaR from actual P&L distribution.

# Arguments
- `weights`       : portfolio weights (dollar exposures)
- `returns`       : (T x N) matrix of returns
- `confidence`    : confidence level
- `holding_period`: not yet applied (daily returns assumed)

Returns the VaR as a positive number.
"""
function historical_var(weights::Vector{Float64}, returns::Matrix{Float64},
                        confidence::Real; holding_period::Int=1)::Float64
    pnl = returns * weights  # T-vector of portfolio P&L
    loss = -pnl
    return quantile(loss, confidence)
end

# ---------------------------------------------------------------------------
# Cornish-Fisher VaR expansion
# ---------------------------------------------------------------------------

"""
    cornish_fisher_var(weights, returns, confidence)

Cornish-Fisher expansion VaR that adjusts for skewness and excess kurtosis
in the portfolio P&L distribution.

Returns the adjusted VaR.
"""
function cornish_fisher_var(weights::Vector{Float64}, returns::Matrix{Float64},
                             confidence::Real)::Float64
    pnl = returns * weights
    mu_pnl = mean(pnl)
    sigma_pnl = std(pnl)
    s = skewness_stat(pnl)
    k = kurtosis_stat(pnl)  # excess kurtosis

    z = quantile(Normal(), confidence)
    # Cornish-Fisher expansion
    z_cf = z + (z^2 - 1.0) * s / 6.0 + (z^3 - 3.0 * z) * k / 24.0 -
           (2.0 * z^3 - 5.0 * z) * s^2 / 36.0

    return -(mu_pnl - z_cf * sigma_pnl)
end

"""
    skewness_stat(x)

Sample skewness of a vector.
"""
function skewness_stat(x::Vector{Float64})::Float64
    n = length(x)
    mu = mean(x)
    sigma = std(x)
    sigma < 1e-14 && return 0.0
    return sum((xi - mu)^3 for xi in x) / (n * sigma^3)
end

"""
    kurtosis_stat(x)

Sample excess kurtosis of a vector.
"""
function kurtosis_stat(x::Vector{Float64})::Float64
    n = length(x)
    mu = mean(x)
    sigma = std(x)
    sigma < 1e-14 && return 0.0
    return sum((xi - mu)^4 for xi in x) / (n * sigma^4) - 3.0
end

# ---------------------------------------------------------------------------
# Expected shortfall (CVaR)
# ---------------------------------------------------------------------------

"""
    expected_shortfall(weights, returns, confidence; method=:historical)

Compute Expected Shortfall (CVaR) - the average loss in the tail beyond VaR.

# Arguments
- `method` : :historical, :parametric, or :cornish_fisher

Returns ES as a positive number.
"""
function expected_shortfall(weights::Vector{Float64}, returns::Matrix{Float64},
                             confidence::Real; method::Symbol=:historical)::Float64
    pnl = returns * weights
    loss = -pnl
    if method == :historical
        var_thresh = quantile(loss, confidence)
        tail_losses = loss[loss .>= var_thresh]
        isempty(tail_losses) && return var_thresh
        return mean(tail_losses)
    elseif method == :parametric
        sigma = std(pnl)
        z = quantile(Normal(), confidence)
        es = sigma * pdf(Normal(), z) / (1.0 - confidence)
        return es
    else
        # Cornish-Fisher adjustment
        s = skewness_stat(pnl)
        k = kurtosis_stat(pnl)
        sigma = std(pnl)
        z = quantile(Normal(), confidence)
        z_cf = z + (z^2 - 1.0) * s / 6.0 + (z^3 - 3.0 * z) * k / 24.0 -
               (2.0 * z^3 - 5.0 * z) * s^2 / 36.0
        return sigma * (pdf(Normal(), z_cf) / (1.0 - confidence)) * (1.0 + s * z_cf / 6.0 + k * z_cf^2 / 24.0)
    end
end

"""
    bootstrap_es_bands(weights, returns, confidence; n_boot=1000, ci_level=0.95)

Bootstrap confidence bands for Expected Shortfall.

Returns named tuple (es_mean, lower, upper) at ci_level.
"""
function bootstrap_es_bands(weights::Vector{Float64}, returns::Matrix{Float64},
                              confidence::Real; n_boot::Int=1000,
                              ci_level::Real=0.95)
    T = size(returns, 1)
    es_samples = Float64[]

    for _ in 1:n_boot
        idx = rand(1:T, T)
        boot_returns = returns[idx, :]
        push!(es_samples, expected_shortfall(weights, boot_returns, confidence))
    end

    alpha = (1.0 - ci_level) / 2.0
    return (
        es_mean = mean(es_samples),
        lower   = quantile(es_samples, alpha),
        upper   = quantile(es_samples, 1.0 - alpha),
        std_err = std(es_samples)
    )
end

# ---------------------------------------------------------------------------
# Marginal and component VaR
# ---------------------------------------------------------------------------

"""
    marginal_var(weights, returns, confidence; lambda=0.94, delta=1e-4)

Compute marginal VaR for each position: the change in portfolio VaR when
that position's weight is increased by a small amount (gradient-based).

Returns a vector of marginal VaRs, one per asset.
"""
function marginal_var(weights::Vector{Float64}, returns::Matrix{Float64},
                      confidence::Real; lambda::Real=0.94,
                      delta::Real=1e-4)::Vector{Float64}
    N = length(weights)
    base_var = parametric_var(weights, returns, confidence; lambda=lambda)
    mvar = zeros(N)

    for i in 1:N
        w_up = copy(weights)
        w_up[i] += delta
        var_up = parametric_var(w_up, returns, confidence; lambda=lambda)
        mvar[i] = (var_up - base_var) / delta
    end
    return mvar
end

"""
    component_var(weights, returns, confidence; lambda=0.94)

Component VaR: the contribution of each position to total portfolio VaR.
Satisfies: sum(component_var) == portfolio_var.
"""
function component_var(weights::Vector{Float64}, returns::Matrix{Float64},
                        confidence::Real; lambda::Real=0.94)::Vector{Float64}
    mvar = marginal_var(weights, returns, confidence; lambda=lambda)
    return weights .* mvar
end

# ---------------------------------------------------------------------------
# Stress testing scenarios
# ---------------------------------------------------------------------------

"""
    Scenario

Represents a stress scenario with factor shocks.
"""
struct Scenario
    name::String
    description::String
    factor_shocks::Dict{String, Float64}  # factor name => shock magnitude
end

"""
    stress_test_2008()

Return the 2008 global financial crisis stress scenario.
Equity drawdowns, credit spread widening, VIX spike, USD strengthening.
"""
function stress_test_2008()::Scenario
    shocks = Dict(
        "equity_spx"      => -0.55,
        "equity_em"       => -0.60,
        "equity_europe"   => -0.52,
        "vix"             =>  3.5,   # multiplicative factor
        "credit_ig_spread"=>  0.0300, # +300 bps
        "credit_hy_spread"=>  0.1800, # +1800 bps
        "usd_index"       =>  0.22,
        "10y_us_yield"    => -0.015,  # flight to quality
        "crude_oil"       => -0.65,
        "gold"            =>  0.30,
        "btc"             => -0.50,   # hypothetical
    )
    return Scenario("2008_crisis", "2008 Global Financial Crisis (Sep-Nov 2008)", shocks)
end

"""
    stress_test_covid2020()

Return the COVID-19 crash stress scenario (Feb-Mar 2020).
"""
function stress_test_covid2020()::Scenario
    shocks = Dict(
        "equity_spx"      => -0.34,
        "equity_em"       => -0.32,
        "equity_europe"   => -0.38,
        "vix"             =>  6.0,
        "credit_ig_spread"=>  0.0180,
        "credit_hy_spread"=>  0.0850,
        "usd_index"       =>  0.08,
        "10y_us_yield"    => -0.012,
        "crude_oil"       => -0.67,
        "gold"            =>  0.05,
        "btc"             => -0.50,
    )
    return Scenario("covid_2020", "COVID-19 Market Crash (Feb-Mar 2020)", shocks)
end

"""
    stress_test_crypto2022()

Return the 2022 crypto winter / Luna-Terra collapse stress scenario.
"""
function stress_test_crypto2022()::Scenario
    shocks = Dict(
        "equity_spx"      => -0.25,
        "equity_em"       => -0.30,
        "equity_tech"     => -0.40,
        "vix"             =>  2.0,
        "credit_ig_spread"=>  0.0100,
        "credit_hy_spread"=>  0.0450,
        "usd_index"       =>  0.12,
        "10y_us_yield"    =>  0.020,  # rate hike environment
        "crude_oil"       => -0.20,
        "btc"             => -0.75,
        "eth"             => -0.80,
        "defi_index"      => -0.90,
        "luna"            => -0.9999, # essentially zero
    )
    return Scenario("crypto_winter_2022", "2022 Crypto Winter / Luna Collapse", shocks)
end

"""
    apply_stress_scenario(portfolio_exposures, scenario, factor_mapping)

Apply a stress scenario to a portfolio.

# Arguments
- `portfolio_exposures`: Dict{String, Float64} of factor_name => dollar exposure
- `scenario`           : a Scenario struct
- `factor_mapping`     : optional Dict mapping portfolio factor names to scenario names

Returns the stressed P&L as a Float64.
"""
function apply_stress_scenario(portfolio_exposures::Dict{String, Float64},
                                scenario::Scenario,
                                factor_mapping::Dict{String, String}=Dict{String, String}())::Float64
    stressed_pnl = 0.0
    for (factor, exposure) in portfolio_exposures
        # Resolve factor name
        scenario_factor = get(factor_mapping, factor, factor)
        shock = get(scenario.factor_shocks, scenario_factor, 0.0)
        stressed_pnl += exposure * shock
    end
    return stressed_pnl
end

"""
    run_all_stress_tests(portfolio_exposures; factor_mapping=Dict())

Run all built-in stress scenarios and return a summary.
"""
function run_all_stress_tests(portfolio_exposures::Dict{String, Float64};
                               factor_mapping::Dict{String, String}=Dict{String, String}())
    scenarios = [stress_test_2008(), stress_test_covid2020(), stress_test_crypto2022()]
    results = Dict{String, Float64}()
    for s in scenarios
        results[s.name] = apply_stress_scenario(portfolio_exposures, s, factor_mapping)
    end
    return results
end

# ---------------------------------------------------------------------------
# Liquidity-adjusted VaR
# ---------------------------------------------------------------------------

"""
    amihud_illiquidity(returns, volume)

Compute the Amihud (2002) illiquidity measure for an asset.

# Arguments
- `returns` : vector of daily returns
- `volume`  : vector of daily dollar trading volume

Returns the Amihud illiquidity ratio (higher = less liquid).
"""
function amihud_illiquidity(returns::Vector{Float64}, volume::Vector{Float64})::Float64
    T = min(length(returns), length(volume))
    T < 1 && return 0.0
    return mean(abs.(returns[1:T]) ./ max.(volume[1:T], 1.0))
end

"""
    liquidity_adjusted_holding_period(base_hp, position_size, avg_daily_volume)

Compute liquidity-adjusted holding period assuming gradual liquidation.
"""
function liquidity_adjusted_holding_period(base_hp::Int,
                                            position_size::Float64,
                                            avg_daily_volume::Float64;
                                            participation_rate::Real=0.20)::Float64
    days_to_liquidate = position_size / (participation_rate * avg_daily_volume)
    return max(Float64(base_hp), days_to_liquidate)
end

"""
    liquidity_adjusted_var(weights, returns, confidence, positions, adv;
                            lambda=0.94, participation_rate=0.20)

Compute liquidity-adjusted VaR (LVaR) accounting for Amihud illiquidity
and time-to-liquidate each position.

# Arguments
- `weights`            : portfolio weights
- `returns`            : (T x N) returns matrix
- `confidence`         : confidence level
- `positions`          : vector of position sizes in dollars
- `adv`                : vector of average daily volumes in dollars
- `participation_rate` : fraction of ADV used per day for liquidation

Returns named tuple (lvar, var_base, liquidity_cost).
"""
function liquidity_adjusted_var(weights::Vector{Float64}, returns::Matrix{Float64},
                                  confidence::Real,
                                  positions::Vector{Float64}, adv::Vector{Float64};
                                  lambda::Real=0.94,
                                  participation_rate::Real=0.20)
    N = length(weights)
    Sigma = ewma_covariance(returns; lambda=lambda)
    z = quantile(Normal(), confidence)

    # Base 1-day parametric VaR
    base_var = z * sqrt(dot(weights, Sigma * weights))

    # Liquidity cost: for each asset, extend holding period and recompute
    lvar_squared = 0.0
    for i in 1:N
        hp_i = liquidity_adjusted_holding_period(1, positions[i], adv[i];
                                                   participation_rate=participation_rate)
        lvar_squared += weights[i]^2 * Sigma[i, i] * hp_i
        for j in (i+1):N
            hp_j = liquidity_adjusted_holding_period(1, positions[j], adv[j];
                                                      participation_rate=participation_rate)
            hp_ij = max(hp_i, hp_j)
            lvar_squared += 2.0 * weights[i] * weights[j] * Sigma[i, j] * hp_ij
        end
    end

    lvar = z * sqrt(max(lvar_squared, 0.0))
    liquidity_cost = lvar - base_var

    return (lvar=lvar, var_base=base_var, liquidity_cost=liquidity_cost)
end

# ---------------------------------------------------------------------------
# Fixed income: DV01, duration
# ---------------------------------------------------------------------------

"""
    dv01_bond(face, coupon_rate, ytm, maturity_years; freq=2)

Compute the DV01 (dollar value of a basis point) for a fixed-rate bond.

# Arguments
- `face`          : face value
- `coupon_rate`   : annual coupon rate (e.g., 0.05 for 5%)
- `ytm`           : yield to maturity (annual)
- `maturity_years`: years to maturity
- `freq`          : coupon frequency (2 = semiannual)

Returns DV01 in dollars.
"""
function dv01_bond(face::Real, coupon_rate::Real, ytm::Real,
                   maturity_years::Real; freq::Int=2)::Float64
    n_periods = round(Int, maturity_years * freq)
    coupon = face * coupon_rate / freq
    y = ytm / freq

    # Compute bond price
    price = 0.0
    for t in 1:n_periods
        price += coupon / (1.0 + y)^t
    end
    price += face / (1.0 + y)^n_periods

    # Price at ytm + 1bp
    y_up = (ytm + 0.0001) / freq
    price_up = 0.0
    for t in 1:n_periods
        price_up += coupon / (1.0 + y_up)^t
    end
    price_up += face / (1.0 + y_up)^n_periods

    return -(price_up - price)  # positive DV01
end

"""
    macaulay_duration(face, coupon_rate, ytm, maturity_years; freq=2)

Compute Macaulay duration in years.
"""
function macaulay_duration(face::Real, coupon_rate::Real, ytm::Real,
                            maturity_years::Real; freq::Int=2)::Float64
    n_periods = round(Int, maturity_years * freq)
    coupon = face * coupon_rate / freq
    y = ytm / freq

    price = 0.0
    weighted_time = 0.0
    for t in 1:n_periods
        cf = coupon / (1.0 + y)^t
        price += cf
        weighted_time += (t / freq) * cf
    end
    fv_pv = face / (1.0 + y)^n_periods
    price += fv_pv
    weighted_time += maturity_years * fv_pv

    return weighted_time / price
end

"""
    modified_duration(face, coupon_rate, ytm, maturity_years; freq=2)

Compute modified duration = Macaulay duration / (1 + ytm/freq).
"""
function modified_duration(face::Real, coupon_rate::Real, ytm::Real,
                            maturity_years::Real; freq::Int=2)::Float64
    mac_dur = macaulay_duration(face, coupon_rate, ytm, maturity_years; freq=freq)
    return mac_dur / (1.0 + ytm / freq)
end

# ---------------------------------------------------------------------------
# Portfolio Greeks aggregation
# ---------------------------------------------------------------------------

"""
    portfolio_greeks_summary(positions)

Aggregate risk metrics across a mixed portfolio of equity, options, and bonds.

# Arguments
- `positions`: vector of named tuples with fields:
    - `asset_type` : :equity, :option, or :bond
    - `notional`   : dollar notional
    - `delta`      : delta (options) or 1.0 (equity)
    - `gamma`      : gamma
    - `vega`       : vega
    - `theta`      : theta
    - `dv01`       : dollar value of 1bp (bonds)

Returns a named tuple of aggregate Greeks.
"""
function portfolio_greeks_summary(positions::Vector)
    total_delta  = sum(p.notional * p.delta  for p in positions)
    total_gamma  = sum(p.notional * p.gamma  for p in positions)
    total_vega   = sum(p.notional * p.vega   for p in positions)
    total_theta  = sum(p.notional * p.theta  for p in positions)
    total_dv01   = sum(p.dv01                for p in positions)

    return (
        delta=total_delta, gamma=total_gamma, vega=total_vega,
        theta=total_theta, dv01=total_dv01
    )
end

# ---------------------------------------------------------------------------
# Risk report struct
# ---------------------------------------------------------------------------

"""
    RiskReport

Comprehensive risk report for a portfolio snapshot.
"""
struct RiskReport
    timestamp::String
    portfolio_name::String
    var_95::Float64
    var_99::Float64
    es_95::Float64
    es_99::Float64
    cf_var_99::Float64
    component_vars::Vector{Float64}
    marginal_vars::Vector{Float64}
    stress_results::Dict{String, Float64}
    lvar_99::Float64
    liquidity_cost::Float64
end

"""
    build_risk_report(weights, returns, positions, adv, portfolio_exposures;
                       portfolio_name="Portfolio", timestamp="")

Build a comprehensive risk report for a portfolio.
"""
function build_risk_report(weights::Vector{Float64}, returns::Matrix{Float64},
                            positions::Vector{Float64}, adv::Vector{Float64},
                            portfolio_exposures::Dict{String, Float64};
                            portfolio_name::String="Portfolio",
                            timestamp::String="")::RiskReport
    ts = isempty(timestamp) ? string(now_approx()) : timestamp

    var_95 = parametric_var(weights, returns, 0.95)
    var_99 = parametric_var(weights, returns, 0.99)
    es_95  = expected_shortfall(weights, returns, 0.95)
    es_99  = expected_shortfall(weights, returns, 0.99)
    cf_var = cornish_fisher_var(weights, returns, 0.99)
    mvar   = marginal_var(weights, returns, 0.99)
    cvar   = component_var(weights, returns, 0.99)
    stress = run_all_stress_tests(portfolio_exposures)

    lv = liquidity_adjusted_var(weights, returns, 0.99, positions, adv)

    return RiskReport(ts, portfolio_name, var_95, var_99, es_95, es_99,
                      cf_var, cvar, mvar, stress, lv.lvar, lv.liquidity_cost)
end

"""
    now_approx()

Return a simple timestamp string (no Dates dependency required at runtime).
"""
function now_approx()::String
    return "runtime_timestamp"
end

# ---------------------------------------------------------------------------
# HTTP bridge to Python risk API
# ---------------------------------------------------------------------------

"""
    push_risk_report_to_api(report, api_url; timeout_sec=10)

Push a RiskReport to the Python risk API over HTTP as JSON.
Requires HTTP.jl to be available in the calling environment.
Returns true on success, false on failure.
"""
function push_risk_report_to_api(report::RiskReport, api_url::String;
                                   timeout_sec::Int=10)::Bool
    try
        # Build JSON payload manually to avoid JSON3 dependency here
        payload = """
        {
          "timestamp": "$(report.timestamp)",
          "portfolio": "$(report.portfolio_name)",
          "var_95": $(report.var_95),
          "var_99": $(report.var_99),
          "es_95": $(report.es_95),
          "es_99": $(report.es_99),
          "cf_var_99": $(report.cf_var_99),
          "lvar_99": $(report.lvar_99),
          "liquidity_cost": $(report.liquidity_cost)
        }
        """
        # HTTP.jl call - only executed if HTTP.jl is loaded in the environment
        if isdefined(Main, :HTTP)
            resp = Main.HTTP.post(api_url, ["Content-Type" => "application/json"],
                                  payload; readtimeout=timeout_sec)
            return resp.status == 200
        else
            @warn "HTTP.jl not loaded; cannot push risk report to API at $api_url"
            return false
        end
    catch e
        @error "Failed to push risk report: $e"
        return false
    end
end

"""
    fetch_positions_from_api(api_url; timeout_sec=10)

Fetch current positions from the Python risk API.
Returns a Dict with position data, or empty Dict on failure.
"""
function fetch_positions_from_api(api_url::String; timeout_sec::Int=10)::Dict{String, Any}
    try
        if isdefined(Main, :HTTP)
            resp = Main.HTTP.get(api_url; readtimeout=timeout_sec)
            if resp.status == 200
                body = String(resp.body)
                # Parse simple key:value JSON manually if JSON3 not available
                if isdefined(Main, :JSON3)
                    return Dict(Main.JSON3.read(body))
                end
                return Dict("raw_body" => body)
            end
        end
        return Dict{String, Any}()
    catch e
        @error "Failed to fetch positions: $e"
        return Dict{String, Any}()
    end
end

end  # module LiveRisk
