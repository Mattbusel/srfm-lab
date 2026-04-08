module CreditRiskModels

using LinearAlgebra
using Statistics
using Random

# ============================================================================
# SECTION 1: Merton Structural Model
# ============================================================================

"""
    MertonModel

Merton (1974) structural model: equity = call option on firm assets.
"""
struct MertonModel
    asset_value::Float64
    asset_vol::Float64
    debt_face::Float64
    risk_free::Float64
    maturity::Float64
end

"""
    normal_cdf(x)

Standard normal CDF via rational approximation (Abramowitz & Stegun 26.2.17).
"""
function normal_cdf(x::Float64)::Float64
    if x < -8.0
        return 0.0
    elseif x > 8.0
        return 1.0
    end
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    pdf_val = exp(-0.5 * x * x) / sqrt(2.0 * pi)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    cdf_val = 1.0 - pdf_val * poly
    return x >= 0.0 ? cdf_val : 1.0 - cdf_val
end

"""
    normal_pdf(x)

Standard normal PDF.
"""
function normal_pdf(x::Float64)::Float64
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)
end

"""
    normal_inv(p)

Inverse normal CDF via rational approximation (Beasley-Springer-Moro).
"""
function normal_inv(p::Float64)::Float64
    if p <= 0.0
        return -10.0
    elseif p >= 1.0
        return 10.0
    end
    a = [-3.969683028665376e1, 2.209460984245205e2,
         -2.759285104469687e2, 1.383577518672690e2,
         -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2,
         -1.556989798598866e2, 6.680131188771972e1,
         -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1,
         -2.400758277161838e0, -2.549732539343734e0,
          4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1,
         2.445134137142996e0, 3.754408661907416e0]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low
        q = sqrt(-2.0 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    elseif p <= p_high
        q = p - 0.5
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6]) * q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    end
end

"""
    bs_call(S, K, r, sigma, T)

Black-Scholes call price.
"""
function bs_call(S::Float64, K::Float64, r::Float64, sigma::Float64, T::Float64)::Float64
    if T <= 0.0
        return max(S - K, 0.0)
    end
    d1 = (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
end

"""
    bs_call_delta(S, K, r, sigma, T)

Black-Scholes call delta.
"""
function bs_call_delta(S::Float64, K::Float64, r::Float64, sigma::Float64, T::Float64)::Float64
    if T <= 0.0
        return S > K ? 1.0 : 0.0
    end
    d1 = (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    return normal_cdf(d1)
end

"""
    bs_call_vega(S, K, r, sigma, T)

Black-Scholes call vega.
"""
function bs_call_vega(S::Float64, K::Float64, r::Float64, sigma::Float64, T::Float64)::Float64
    if T <= 0.0
        return 0.0
    end
    d1 = (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    return S * normal_pdf(d1) * sqrt(T)
end

"""
    merton_equity_value(V, sigma_V, D, r, T)

Equity value under Merton model: E = BS_Call(V, D, r, sigma_V, T).
"""
function merton_equity_value(V::Float64, sigma_V::Float64, D::Float64,
                              r::Float64, T::Float64)::Float64
    return bs_call(V, D, r, sigma_V, T)
end

"""
    merton_equity_vol(V, sigma_V, D, r, T, E)

Implied equity volatility: sigma_E = (V/E) * N(d1) * sigma_V.
"""
function merton_equity_vol(V::Float64, sigma_V::Float64, D::Float64,
                            r::Float64, T::Float64, E::Float64)::Float64
    d1 = (log(V / D) + (r + 0.5 * sigma_V^2) * T) / (sigma_V * sqrt(T))
    return (V / E) * normal_cdf(d1) * sigma_V
end

"""
    merton_solve(equity_value, equity_vol, debt_face, risk_free, maturity;
                 max_iter=200, tol=1e-10)

Iterative solver for (V, sigma_V) from observed (E, sigma_E).
Uses simultaneous Newton iteration on the two Merton equations.
"""
function merton_solve(equity_value::Float64, equity_vol::Float64,
                      debt_face::Float64, risk_free::Float64, maturity::Float64;
                      max_iter::Int=200, tol::Float64=1e-10)
    E = equity_value
    sigma_E = equity_vol
    D = debt_face
    r = risk_free
    T = maturity

    # Initial guess
    V = E + D * exp(-r * T)
    sigma_V = sigma_E * E / V

    for iter in 1:max_iter
        # Equation 1: E - BS_Call(V, D, r, sigma_V, T) = 0
        E_model = bs_call(V, D, r, sigma_V, T)
        d1 = (log(V / D) + (r + 0.5 * sigma_V^2) * T) / (sigma_V * sqrt(T))
        Nd1 = normal_cdf(d1)

        # Equation 2: sigma_E * E - V * N(d1) * sigma_V = 0
        f1 = E_model - E
        f2 = V * Nd1 * sigma_V - sigma_E * E

        # Jacobian
        dE_dV = Nd1
        dE_dsigma = bs_call_vega(V, D, r, sigma_V, T)

        nd1 = normal_pdf(d1)
        dd1_dV = 1.0 / (V * sigma_V * sqrt(T))
        dd1_dsigma = -(log(V / D) + (r - 0.5 * sigma_V^2) * T) / (sigma_V^2 * sqrt(T))

        df2_dV = Nd1 * sigma_V + V * nd1 * dd1_dV * sigma_V
        df2_dsigma = V * nd1 * dd1_dsigma * sigma_V + V * Nd1

        # Newton step
        det_J = dE_dV * df2_dsigma - dE_dsigma * df2_dV
        if abs(det_J) < 1e-15
            break
        end

        dV = (df2_dsigma * f1 - dE_dsigma * f2) / det_J
        dsigma = (dE_dV * f2 - df2_dV * f1) / det_J

        V -= dV
        sigma_V -= dsigma

        V = max(V, E + 0.01)
        sigma_V = max(sigma_V, 0.001)

        if abs(dV) + abs(dsigma) < tol
            break
        end
    end

    return MertonModel(V, sigma_V, D, r, T)
end

"""
    merton_default_probability(m::MertonModel)

Physical default probability: P(V_T < D).
"""
function merton_default_probability(m::MertonModel)::Float64
    d2 = (log(m.asset_value / m.debt_face) +
          (m.risk_free - 0.5 * m.asset_vol^2) * m.maturity) /
         (m.asset_vol * sqrt(m.maturity))
    return normal_cdf(-d2)
end

"""
    merton_debt_value(m::MertonModel)

Debt value: D*exp(-rT) - Put(V, D).
"""
function merton_debt_value(m::MertonModel)::Float64
    E = merton_equity_value(m.asset_value, m.asset_vol, m.debt_face,
                             m.risk_free, m.maturity)
    return m.asset_value - E
end

"""
    merton_credit_spread(m::MertonModel)

Credit spread implied by Merton model.
"""
function merton_credit_spread(m::MertonModel)::Float64
    D_risky = merton_debt_value(m)
    D_face = m.debt_face
    T = m.maturity
    if D_risky <= 0.0 || T <= 0.0
        return 0.0
    end
    yield_risky = -log(D_risky / D_face) / T
    return yield_risky - m.risk_free
end

"""
    merton_recovery_rate(m::MertonModel)

Expected recovery rate conditional on default.
"""
function merton_recovery_rate(m::MertonModel)::Float64
    pd = merton_default_probability(m)
    if pd < 1e-15
        return 1.0
    end
    V = m.asset_value
    sigma_V = m.asset_vol
    D = m.debt_face
    r = m.risk_free
    T = m.maturity

    d1 = (log(V / D) + (r + 0.5 * sigma_V^2) * T) / (sigma_V * sqrt(T))
    d2 = d1 - sigma_V * sqrt(T)

    expected_loss = D * exp(-r * T) * normal_cdf(-d2) - V * normal_cdf(-d1)
    lgd = expected_loss / (D * exp(-r * T) * pd)
    return 1.0 - lgd
end

# ============================================================================
# SECTION 2: KMV Distance-to-Default
# ============================================================================

"""
    KMVResult

Result of KMV model computation.
"""
struct KMVResult
    asset_value::Float64
    asset_vol::Float64
    distance_to_default::Float64
    default_probability::Float64
    default_point::Float64
end

"""
    kmv_default_point(short_term_debt, long_term_debt)

KMV default point: STD + 0.5 * LTD.
"""
function kmv_default_point(short_term_debt::Float64, long_term_debt::Float64)::Float64
    return short_term_debt + 0.5 * long_term_debt
end

"""
    kmv_distance_to_default(V, sigma_V, dp, mu, T)

Distance to default: DD = (V - DP) / (V * sigma_V).
With drift: DD = (ln(V/DP) + (mu - 0.5*sigma_V^2)*T) / (sigma_V*sqrt(T)).
"""
function kmv_distance_to_default(V::Float64, sigma_V::Float64, dp::Float64,
                                  mu::Float64, T::Float64)::Float64
    if V <= 0.0 || dp <= 0.0 || sigma_V <= 0.0 || T <= 0.0
        return 0.0
    end
    return (log(V / dp) + (mu - 0.5 * sigma_V^2) * T) / (sigma_V * sqrt(T))
end

"""
    kmv_solve(equity_value, equity_vol, short_term_debt, long_term_debt,
              risk_free, maturity; max_iter=500, tol=1e-10)

Full KMV iterative procedure.
"""
function kmv_solve(equity_value::Float64, equity_vol::Float64,
                   short_term_debt::Float64, long_term_debt::Float64,
                   risk_free::Float64, maturity::Float64;
                   max_iter::Int=500, tol::Float64=1e-10)::KMVResult
    dp = kmv_default_point(short_term_debt, long_term_debt)
    total_debt = short_term_debt + long_term_debt

    m = merton_solve(equity_value, equity_vol, total_debt, risk_free, maturity;
                     max_iter=max_iter, tol=tol)

    dd = kmv_distance_to_default(m.asset_value, m.asset_vol, dp, risk_free, maturity)
    pd = normal_cdf(-dd)

    return KMVResult(m.asset_value, m.asset_vol, dd, pd, dp)
end

"""
    kmv_edf_term_structure(equity_value, equity_vol, short_term_debt, long_term_debt,
                           risk_free, horizons)

Term structure of Expected Default Frequencies.
"""
function kmv_edf_term_structure(equity_value::Float64, equity_vol::Float64,
                                short_term_debt::Float64, long_term_debt::Float64,
                                risk_free::Float64, horizons::Vector{Float64})
    dp = kmv_default_point(short_term_debt, long_term_debt)
    total_debt = short_term_debt + long_term_debt

    n = length(horizons)
    dd_vec = Vector{Float64}(undef, n)
    edf_vec = Vector{Float64}(undef, n)

    for i in 1:n
        T = horizons[i]
        m = merton_solve(equity_value, equity_vol, total_debt, risk_free, T;
                         max_iter=300, tol=1e-10)
        dd_vec[i] = kmv_distance_to_default(m.asset_value, m.asset_vol, dp, risk_free, T)
        edf_vec[i] = normal_cdf(-dd_vec[i])
    end

    return dd_vec, edf_vec
end

"""
    kmv_empirical_edf(dd, dd_to_edf_map)

Map DD to empirical EDF using interpolation on historical data.
dd_to_edf_map is a sorted Nx2 matrix: col1=DD, col2=EDF.
"""
function kmv_empirical_edf(dd::Float64, dd_to_edf_map::Matrix{Float64})::Float64
    n = size(dd_to_edf_map, 1)
    if n == 0
        return normal_cdf(-dd)
    end

    if dd <= dd_to_edf_map[1, 1]
        return dd_to_edf_map[1, 2]
    end
    if dd >= dd_to_edf_map[n, 1]
        return dd_to_edf_map[n, 2]
    end

    for i in 1:(n-1)
        if dd_to_edf_map[i, 1] <= dd <= dd_to_edf_map[i+1, 1]
            w = (dd - dd_to_edf_map[i, 1]) / (dd_to_edf_map[i+1, 1] - dd_to_edf_map[i, 1])
            return (1.0 - w) * dd_to_edf_map[i, 2] + w * dd_to_edf_map[i+1, 2]
        end
    end

    return normal_cdf(-dd)
end

# ============================================================================
# SECTION 3: Jarrow-Turnbull Reduced Form Model
# ============================================================================

"""
    HazardRateCurve

Piecewise constant hazard rate curve.
"""
struct HazardRateCurve
    times::Vector{Float64}      # Tenor points
    hazard_rates::Vector{Float64}  # Piecewise constant hazard rates
end

"""
    survival_probability(hc::HazardRateCurve, t)

Survival probability Q(t) = exp(-integral h(s) ds from 0 to t).
"""
function survival_probability(hc::HazardRateCurve, t::Float64)::Float64
    if t <= 0.0
        return 1.0
    end
    integral = 0.0
    prev_t = 0.0
    for i in 1:length(hc.times)
        if t <= hc.times[i]
            integral += hc.hazard_rates[i] * (t - prev_t)
            return exp(-integral)
        end
        integral += hc.hazard_rates[i] * (hc.times[i] - prev_t)
        prev_t = hc.times[i]
    end
    # Flat extrapolation
    if length(hc.hazard_rates) > 0
        integral += hc.hazard_rates[end] * (t - prev_t)
    end
    return exp(-integral)
end

"""
    default_probability_interval(hc::HazardRateCurve, t1, t2)

Default probability in interval [t1, t2]: Q(t1) - Q(t2).
"""
function default_probability_interval(hc::HazardRateCurve, t1::Float64, t2::Float64)::Float64
    return survival_probability(hc, t1) - survival_probability(hc, t2)
end

"""
    hazard_rate_at(hc::HazardRateCurve, t)

Hazard rate at time t.
"""
function hazard_rate_at(hc::HazardRateCurve, t::Float64)::Float64
    for i in 1:length(hc.times)
        if t <= hc.times[i]
            return hc.hazard_rates[i]
        end
    end
    return length(hc.hazard_rates) > 0 ? hc.hazard_rates[end] : 0.0
end

"""
    bootstrap_hazard_rates(cds_tenors, cds_spreads, risk_free_rates, recovery_rate)

Bootstrap hazard rates from CDS par spreads.
Uses iterative stripping: for each tenor, solve for hazard rate that
reprices the CDS to par.
"""
function bootstrap_hazard_rates(cds_tenors::Vector{Float64},
                                 cds_spreads::Vector{Float64},
                                 risk_free_rates::Vector{Float64},
                                 recovery_rate::Float64)::HazardRateCurve
    n = length(cds_tenors)
    hazard_rates = Vector{Float64}(undef, n)
    dt = 0.25  # Quarterly payment frequency

    for i in 1:n
        T = cds_tenors[i]
        spread = cds_spreads[i]
        r = risk_free_rates[i]

        # Build partial curve with known hazard rates
        h_lo = 1e-6
        h_hi = 2.0

        for bisect_iter in 1:100
            h_mid = 0.5 * (h_lo + h_hi)
            hazard_rates[i] = h_mid

            hc = HazardRateCurve(cds_tenors[1:i], hazard_rates[1:i])

            # Price CDS
            premium_leg = 0.0
            default_leg = 0.0
            t_prev = 0.0
            t_curr = dt

            while t_curr <= T + 1e-10
                t_actual = min(t_curr, T)
                df = exp(-r * t_actual)
                q_prev = survival_probability(hc, t_prev)
                q_curr = survival_probability(hc, t_actual)

                # Premium leg: spread * dt * Q(t) * df
                premium_leg += spread * (t_actual - t_prev) * q_curr * df

                # Accrual on default
                premium_leg += spread * 0.5 * (t_actual - t_prev) * (q_prev - q_curr) * df

                # Default leg: (1-R) * (Q(t_prev) - Q(t)) * df
                default_leg += (1.0 - recovery_rate) * (q_prev - q_curr) * df

                t_prev = t_actual
                t_curr += dt
                if t_actual >= T
                    break
                end
            end

            diff = premium_leg - default_leg
            if abs(diff) < 1e-12
                break
            end
            if diff > 0
                h_lo = h_mid
            else
                h_hi = h_mid
            end
        end
    end

    return HazardRateCurve(cds_tenors, hazard_rates)
end

"""
    risky_bond_price(face, coupon_rate, coupon_freq, maturity, risk_free,
                     hc::HazardRateCurve, recovery_rate)

Price a risky bond using the Jarrow-Turnbull framework.
"""
function risky_bond_price(face::Float64, coupon_rate::Float64, coupon_freq::Int,
                          maturity::Float64, risk_free::Float64,
                          hc::HazardRateCurve, recovery_rate::Float64)::Float64
    dt = 1.0 / coupon_freq
    coupon = face * coupon_rate * dt
    price = 0.0
    t_prev = 0.0
    t = dt

    while t <= maturity + 1e-10
        t_actual = min(t, maturity)
        df = exp(-risk_free * t_actual)
        q = survival_probability(hc, t_actual)
        q_prev = survival_probability(hc, t_prev)

        # Coupon payment if surviving
        price += coupon * q * df

        # Recovery on default in this interval
        price += face * recovery_rate * (q_prev - q) * df

        t_prev = t_actual
        t += dt
        if t_actual >= maturity
            break
        end
    end

    # Principal at maturity if surviving
    price += face * survival_probability(hc, maturity) * exp(-risk_free * maturity)

    return price
end

"""
    risky_bond_spread(face, coupon_rate, coupon_freq, maturity, risk_free,
                      hc::HazardRateCurve, recovery_rate)

Z-spread of a risky bond.
"""
function risky_bond_spread(face::Float64, coupon_rate::Float64, coupon_freq::Int,
                           maturity::Float64, risk_free::Float64,
                           hc::HazardRateCurve, recovery_rate::Float64)::Float64
    risky_price = risky_bond_price(face, coupon_rate, coupon_freq, maturity, risk_free,
                                    hc, recovery_rate)

    # Solve for z-spread
    z_lo = -0.05
    z_hi = 0.50
    for _ in 1:100
        z_mid = 0.5 * (z_lo + z_hi)
        dt = 1.0 / coupon_freq
        coupon = face * coupon_rate * dt
        pv = 0.0
        t = dt
        while t <= maturity + 1e-10
            t_actual = min(t, maturity)
            pv += coupon * exp(-(risk_free + z_mid) * t_actual)
            t += dt
            if t_actual >= maturity
                break
            end
        end
        pv += face * exp(-(risk_free + z_mid) * maturity)

        if pv > risky_price
            z_lo = z_mid
        else
            z_hi = z_mid
        end
    end
    return 0.5 * (z_lo + z_hi)
end

# ============================================================================
# SECTION 4: CDS Pricing
# ============================================================================

"""
    CDSPricer

CDS pricing engine with full analytics.
"""
struct CDSPricer
    notional::Float64
    maturity::Float64
    recovery_rate::Float64
    premium_freq::Int      # Quarterly = 4
    hazard_curve::HazardRateCurve
    risk_free_rate::Float64
end

"""
    cds_par_spread(cp::CDSPricer)

Par spread that makes CDS MTM = 0.
"""
function cds_par_spread(cp::CDSPricer)::Float64
    dt = 1.0 / cp.premium_freq
    T = cp.maturity
    r = cp.risk_free_rate
    hc = cp.hazard_curve
    R = cp.recovery_rate

    risky_annuity = 0.0
    default_leg = 0.0
    t_prev = 0.0
    t = dt

    while t <= T + 1e-10
        t_actual = min(t, T)
        df = exp(-r * t_actual)
        q = survival_probability(hc, t_actual)
        q_prev = survival_probability(hc, t_prev)

        risky_annuity += dt * q * df
        risky_annuity += 0.5 * dt * (q_prev - q) * df  # Accrual

        default_leg += (1.0 - R) * (q_prev - q) * df

        t_prev = t_actual
        t += dt
        if t_actual >= T
            break
        end
    end

    if risky_annuity < 1e-15
        return 0.0
    end
    return default_leg / risky_annuity
end

"""
    cds_mtm(cp::CDSPricer, contract_spread)

Mark-to-market value of existing CDS position (protection buyer).
"""
function cds_mtm(cp::CDSPricer, contract_spread::Float64)::Float64
    par = cds_par_spread(cp)
    dt = 1.0 / cp.premium_freq
    T = cp.maturity
    r = cp.risk_free_rate
    hc = cp.hazard_curve

    risky_annuity = 0.0
    t_prev = 0.0
    t = dt
    while t <= T + 1e-10
        t_actual = min(t, T)
        df = exp(-r * t_actual)
        q = survival_probability(hc, t_actual)
        q_prev = survival_probability(hc, t_prev)
        risky_annuity += dt * q * df
        risky_annuity += 0.5 * dt * (q_prev - q) * df
        t_prev = t_actual
        t += dt
        if t_actual >= T
            break
        end
    end

    return cp.notional * (par - contract_spread) * risky_annuity
end

"""
    cds_risky_duration(cp::CDSPricer)

Risky DV01: change in CDS MTM for 1bp spread change.
"""
function cds_risky_duration(cp::CDSPricer)::Float64
    dt = 1.0 / cp.premium_freq
    T = cp.maturity
    r = cp.risk_free_rate
    hc = cp.hazard_curve

    risky_annuity = 0.0
    t_prev = 0.0
    t = dt
    while t <= T + 1e-10
        t_actual = min(t, T)
        df = exp(-r * t_actual)
        q = survival_probability(hc, t_actual)
        q_prev = survival_probability(hc, t_prev)
        risky_annuity += dt * q * df
        risky_annuity += 0.5 * dt * (q_prev - q) * df
        t_prev = t_actual
        t += dt
        if t_actual >= T
            break
        end
    end

    return cp.notional * risky_annuity * 0.0001
end

"""
    cds_survival_curve(cp::CDSPricer, times)

Survival probability curve at given time points.
"""
function cds_survival_curve(cp::CDSPricer, times::Vector{Float64})::Vector{Float64}
    return [survival_probability(cp.hazard_curve, t) for t in times]
end

"""
    cds_forward_spread(cp::CDSPricer, t1, t2)

Forward CDS spread for protection in [t1, t2].
"""
function cds_forward_spread(cp::CDSPricer, t1::Float64, t2::Float64)::Float64
    r = cp.risk_free_rate
    hc = cp.hazard_curve
    R = cp.recovery_rate
    dt = 0.25

    risky_annuity = 0.0
    default_leg = 0.0
    t_prev = t1
    t = t1 + dt

    while t <= t2 + 1e-10
        t_actual = min(t, t2)
        df = exp(-r * t_actual)
        q = survival_probability(hc, t_actual)
        q_prev = survival_probability(hc, t_prev)

        risky_annuity += (t_actual - t_prev) * q * df
        default_leg += (1.0 - R) * (q_prev - q) * df

        t_prev = t_actual
        t += dt
        if t_actual >= t2
            break
        end
    end

    if risky_annuity < 1e-15
        return 0.0
    end
    return default_leg / risky_annuity
end

# ============================================================================
# SECTION 5: CDO Tranching - Gaussian Copula
# ============================================================================

"""
    GaussianCopulaCDO

Gaussian copula CDO pricing (Li 2000).
"""
struct GaussianCopulaCDO
    num_names::Int
    notional_per_name::Float64
    default_probs::Vector{Float64}    # Individual default probabilities
    recovery_rates::Vector{Float64}   # Individual recovery rates
    correlation::Float64              # Uniform pairwise correlation
    maturity::Float64
    risk_free::Float64
end

"""
    conditional_default_prob(p, rho, z)

Conditional default probability given systematic factor z.
P(default | Z=z) = Phi((Phi^{-1}(p) - sqrt(rho)*z) / sqrt(1-rho))
"""
function conditional_default_prob(p::Float64, rho::Float64, z::Float64)::Float64
    if rho <= 0.0
        return p
    end
    if rho >= 1.0
        return z < normal_inv(p) ? 1.0 : 0.0
    end
    threshold = normal_inv(p)
    return normal_cdf((threshold - sqrt(rho) * z) / sqrt(1.0 - rho))
end

"""
    gauss_hermite_nodes(n)

Gauss-Hermite quadrature nodes and weights for numerical integration.
"""
function gauss_hermite_nodes(n::Int)
    # Golub-Welsch algorithm using tridiagonal eigenvalue problem
    # For Hermite polynomials: beta_i = sqrt(i/2)
    diag_vals = zeros(n)
    off_diag = [sqrt(i / 2.0) for i in 1:(n-1)]

    # Build tridiagonal matrix
    T = zeros(n, n)
    for i in 1:n
        T[i, i] = diag_vals[i]
    end
    for i in 1:(n-1)
        T[i, i+1] = off_diag[i]
        T[i+1, i] = off_diag[i]
    end

    eig = eigen(Symmetric(T))
    nodes = eig.values
    weights = [eig.vectors[1, i]^2 * sqrt(pi) for i in 1:n]

    return nodes, weights
end

"""
    cdo_tranche_expected_loss(cdo::GaussianCopulaCDO, attach, detach, num_quad)

Expected tranche loss using Gaussian copula with Gauss-Hermite quadrature.
"""
function cdo_tranche_expected_loss(cdo::GaussianCopulaCDO, attach::Float64,
                                    detach::Float64; num_quad::Int=40)::Float64
    N = cdo.num_names
    nodes, weights = gauss_hermite_nodes(num_quad)

    total_notional = N * cdo.notional_per_name
    tranche_notional = (detach - attach) * total_notional

    expected_tranche_loss = 0.0

    for q in 1:num_quad
        z = nodes[q] * sqrt(2.0)  # Transform from Hermite to standard normal
        w = weights[q] / sqrt(pi)

        # Conditional on Z=z, compute expected portfolio loss
        cond_expected_loss = 0.0
        cond_var_loss = 0.0

        for i in 1:N
            p_i = conditional_default_prob(cdo.default_probs[i], cdo.correlation, z)
            lgd_i = cdo.notional_per_name * (1.0 - cdo.recovery_rates[i])
            cond_expected_loss += p_i * lgd_i
            cond_var_loss += p_i * (1.0 - p_i) * lgd_i^2
        end

        # Large homogeneous pool approximation for tranche loss
        # Use recursive method for heterogeneous portfolio
        cond_loss_frac = cond_expected_loss / total_notional

        # Approximate loss distribution as beta distribution
        if cond_var_loss > 1e-15 && cond_expected_loss > 1e-15
            loss_frac_var = cond_var_loss / (total_notional^2)
            mu_l = cond_loss_frac
            sigma_l = sqrt(loss_frac_var)

            # Expected tranche loss via integration over loss distribution
            # For LHP: use beta approximation
            alpha_b = mu_l * (mu_l * (1.0 - mu_l) / max(loss_frac_var, 1e-15) - 1.0)
            beta_b = (1.0 - mu_l) * (mu_l * (1.0 - mu_l) / max(loss_frac_var, 1e-15) - 1.0)

            alpha_b = max(alpha_b, 0.01)
            beta_b = max(beta_b, 0.01)

            # Numerical integration over loss fraction
            num_loss_pts = 100
            dl = 1.0 / num_loss_pts
            tranche_loss_given_z = 0.0

            for k in 1:num_loss_pts
                l = (k - 0.5) * dl
                # Beta density
                if l > 0.0 && l < 1.0
                    log_beta_pdf = (alpha_b - 1.0) * log(l) + (beta_b - 1.0) * log(1.0 - l) +
                                   lgamma(alpha_b + beta_b) - lgamma(alpha_b) - lgamma(beta_b)
                    beta_pdf = exp(log_beta_pdf)
                else
                    beta_pdf = 0.0
                end

                # Tranche loss function
                if l <= attach
                    tl = 0.0
                elseif l >= detach
                    tl = detach - attach
                else
                    tl = l - attach
                end

                tranche_loss_given_z += tl * beta_pdf * dl
            end

            expected_tranche_loss += w * tranche_loss_given_z
        else
            # Degenerate case
            if cond_loss_frac <= attach
                expected_tranche_loss += w * 0.0
            elseif cond_loss_frac >= detach
                expected_tranche_loss += w * (detach - attach)
            else
                expected_tranche_loss += w * (cond_loss_frac - attach)
            end
        end
    end

    return expected_tranche_loss * total_notional
end

"""
    cdo_tranche_spread(cdo::GaussianCopulaCDO, attach, detach; num_quad=40)

Fair spread for a CDO tranche.
"""
function cdo_tranche_spread(cdo::GaussianCopulaCDO, attach::Float64,
                             detach::Float64; num_quad::Int=40)::Float64
    total_notional = cdo.num_names * cdo.notional_per_name
    tranche_notional = (detach - attach) * total_notional
    T = cdo.maturity
    r = cdo.risk_free

    # Approximate by computing expected loss at maturity
    el = cdo_tranche_expected_loss(cdo, attach, detach; num_quad=num_quad)

    # Protection leg PV (simplified)
    protection_pv = el * exp(-r * T * 0.5)  # Mid-period discounting

    # Premium leg PV
    dt = 0.25
    premium_pv_per_bp = 0.0
    t = dt
    while t <= T + 1e-10
        t_actual = min(t, T)
        df = exp(-r * t_actual)
        # Outstanding tranche notional (approximate)
        frac_remaining = 1.0 - el / tranche_notional * (t_actual / T)
        frac_remaining = max(frac_remaining, 0.0)
        premium_pv_per_bp += dt * tranche_notional * frac_remaining * df
        t += dt
        if t_actual >= T
            break
        end
    end

    if premium_pv_per_bp < 1e-15
        return 0.0
    end

    return protection_pv / premium_pv_per_bp
end

"""
    base_correlation_solve(cdo::GaussianCopulaCDO, detach, market_spread;
                           tol=1e-6, max_iter=100)

Solve for base correlation that reprices equity tranche [0, detach] to market.
"""
function base_correlation_solve(cdo::GaussianCopulaCDO, detach::Float64,
                                 market_spread::Float64;
                                 tol::Float64=1e-6, max_iter::Int=100)::Float64
    rho_lo = 0.001
    rho_hi = 0.999

    for _ in 1:max_iter
        rho_mid = 0.5 * (rho_lo + rho_hi)
        cdo_test = GaussianCopulaCDO(cdo.num_names, cdo.notional_per_name,
                                      cdo.default_probs, cdo.recovery_rates,
                                      rho_mid, cdo.maturity, cdo.risk_free)
        model_spread = cdo_tranche_spread(cdo_test, 0.0, detach)

        if abs(model_spread - market_spread) < tol
            return rho_mid
        end

        if model_spread > market_spread
            rho_hi = rho_mid
        else
            rho_lo = rho_mid
        end
    end

    return 0.5 * (rho_lo + rho_hi)
end

"""
    cdo_recursive_loss_distribution(default_probs_cond, loss_units, max_loss_units)

Recursive algorithm for exact loss distribution conditional on Z.
Andersen-Sidenius-Basu (2003) tower method.
"""
function cdo_recursive_loss_distribution(default_probs_cond::Vector{Float64},
                                          loss_units::Vector{Int},
                                          max_loss_units::Int)
    n = length(default_probs_cond)
    # prob[k+1] = P(L = k loss units) after processing i names
    prob = zeros(max_loss_units + 1)
    prob[1] = 1.0  # L=0 initially

    for i in 1:n
        p_i = default_probs_cond[i]
        l_i = loss_units[i]
        new_prob = zeros(max_loss_units + 1)

        for k in 0:max_loss_units
            if prob[k+1] > 0.0
                # No default of name i
                new_prob[k+1] += (1.0 - p_i) * prob[k+1]
                # Default of name i
                k_new = k + l_i
                if k_new <= max_loss_units
                    new_prob[k_new+1] += p_i * prob[k+1]
                end
            end
        end
        prob = new_prob
    end

    return prob
end

# ============================================================================
# SECTION 6: Credit Migration Matrix
# ============================================================================

"""
    CreditMigrationMatrix

Transition probability matrix for credit ratings.
"""
struct CreditMigrationMatrix
    ratings::Vector{String}
    transition_matrix::Matrix{Float64}  # (from_rating, to_rating) annual
end

"""
    create_sp_transition_matrix()

Standard & Poor's historical transition matrix (simplified).
Ratings: AAA, AA, A, BBB, BB, B, CCC, D
"""
function create_sp_transition_matrix()::CreditMigrationMatrix
    ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
    n = length(ratings)
    T = zeros(n, n)

    # AAA row
    T[1, :] = [0.9081, 0.0833, 0.0068, 0.0006, 0.0012, 0.0000, 0.0000, 0.0000]
    # AA row
    T[2, :] = [0.0070, 0.9065, 0.0779, 0.0064, 0.0006, 0.0014, 0.0002, 0.0000]
    # A row
    T[3, :] = [0.0009, 0.0227, 0.9105, 0.0552, 0.0074, 0.0026, 0.0001, 0.0006]
    # BBB row
    T[4, :] = [0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0117, 0.0012, 0.0018]
    # BB row
    T[5, :] = [0.0003, 0.0014, 0.0067, 0.0773, 0.8053, 0.0884, 0.0100, 0.0106]
    # B row
    T[6, :] = [0.0000, 0.0011, 0.0024, 0.0043, 0.0648, 0.8346, 0.0407, 0.0521]
    # CCC row
    T[7, :] = [0.0022, 0.0000, 0.0022, 0.0130, 0.0238, 0.1124, 0.6486, 0.1978]
    # D row (absorbing)
    T[8, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    return CreditMigrationMatrix(ratings, T)
end

"""
    multi_year_transition(cm::CreditMigrationMatrix, years)

Multi-year transition matrix: T^n.
"""
function multi_year_transition(cm::CreditMigrationMatrix, years::Int)::Matrix{Float64}
    result = copy(cm.transition_matrix)
    for _ in 2:years
        result = result * cm.transition_matrix
    end
    return result
end

"""
    generator_matrix(cm::CreditMigrationMatrix)

Extract generator (intensity) matrix Q such that T = exp(Q).
Uses eigendecomposition: Q = V * diag(log(lambda)) * V^{-1}.
"""
function generator_matrix(cm::CreditMigrationMatrix)::Matrix{Float64}
    T = cm.transition_matrix
    n = size(T, 1)

    eig = eigen(T)
    lambdas = eig.values
    V = eig.vectors

    # Take log of eigenvalues (handle complex)
    log_lambdas = zeros(ComplexF64, n)
    for i in 1:n
        if real(lambdas[i]) > 0
            log_lambdas[i] = log(Complex(lambdas[i]))
        else
            log_lambdas[i] = log(Complex(abs(lambdas[i]))) + im * pi
        end
    end

    Q_complex = V * Diagonal(log_lambdas) * inv(V)
    Q = real.(Q_complex)

    # Adjust to ensure valid generator: off-diagonal >= 0, rows sum to 0
    for i in 1:n
        for j in 1:n
            if i != j
                Q[i, j] = max(Q[i, j], 0.0)
            end
        end
        Q[i, i] = -sum(Q[i, k] for k in 1:n if k != i)
    end

    return Q
end

"""
    continuous_time_transition(cm::CreditMigrationMatrix, t)

Transition matrix at arbitrary time: P(t) = exp(Q*t).
"""
function continuous_time_transition(cm::CreditMigrationMatrix, t::Float64)::Matrix{Float64}
    Q = generator_matrix(cm)
    Qt = Q * t
    # Matrix exponential via Padé approximation
    return matrix_exponential(Qt)
end

"""
    matrix_exponential(A; order=13)

Padé approximation for matrix exponential using scaling and squaring.
"""
function matrix_exponential(A::Matrix{Float64}; order::Int=13)::Matrix{Float64}
    n = size(A, 1)
    norm_A = opnorm(A, Inf)

    # Scaling
    s = max(0, ceil(Int, log2(norm_A / 5.4)))
    A_scaled = A / 2^s

    # Padé coefficients for order 13
    c = [1.0, 0.5, 0.12, 1.833333333333333e-2, 1.992753623188406e-3,
         1.630434782608696e-4, 1.035196687370600e-5, 5.175983436853000e-7,
         2.043151356652500e-8, 6.306022705717595e-10, 1.483770048404140e-11,
         2.529153491597966e-13, 2.810170546219963e-15, 1.544049750670309e-17]

    I_n = Matrix{Float64}(I, n, n)
    A2 = A_scaled * A_scaled
    A4 = A2 * A2
    A6 = A2 * A4

    U = A_scaled * (A6 * (c[14] * A6 + c[12] * A4 + c[10] * A2) +
                     c[8] * A6 + c[6] * A4 + c[4] * A2 + c[2] * I_n)
    V = A6 * (c[13] * A6 + c[11] * A4 + c[9] * A2) +
        c[7] * A6 + c[5] * A4 + c[3] * A2 + c[1] * I_n

    F = (V - U) \ (V + U)

    # Squaring
    for _ in 1:s
        F = F * F
    end

    return F
end

"""
    credit_migration_var(cm::CreditMigrationMatrix, current_rating_idx,
                         bond_values, confidence; horizon=1)

Credit migration VaR: value change distribution from rating transitions.
bond_values[j] = bond value if rating changes to j.
"""
function credit_migration_var(cm::CreditMigrationMatrix, current_rating_idx::Int,
                               bond_values::Vector{Float64}, confidence::Float64;
                               horizon::Int=1)::Float64
    T_h = horizon == 1 ? cm.transition_matrix : multi_year_transition(cm, horizon)
    n = length(bond_values)
    current_value = bond_values[current_rating_idx]

    # Value changes and probabilities
    changes = [(bond_values[j] - current_value, T_h[current_rating_idx, j]) for j in 1:n]
    sort!(changes, by=x -> x[1])

    # Find quantile
    cum_prob = 0.0
    alpha = 1.0 - confidence
    for (dv, p) in changes
        cum_prob += p
        if cum_prob >= alpha
            return -dv
        end
    end
    return -(changes[1][1])
end

"""
    rating_momentum(cm::CreditMigrationMatrix, rating_idx, horizon)

Expected rating drift over horizon years.
"""
function rating_momentum(cm::CreditMigrationMatrix, rating_idx::Int, horizon::Int)::Float64
    T_h = horizon == 1 ? cm.transition_matrix : multi_year_transition(cm, horizon)
    n = size(T_h, 1)
    expected_rating = sum(j * T_h[rating_idx, j] for j in 1:n)
    return expected_rating - rating_idx
end

# ============================================================================
# SECTION 7: Portfolio Credit VaR (Monte Carlo)
# ============================================================================

"""
    PortfolioCreditRisk

Portfolio of credit exposures for VaR calculation.
"""
struct PortfolioCreditRisk
    num_obligors::Int
    exposures::Vector{Float64}           # Exposure at default
    default_probs::Vector{Float64}       # 1-year PD
    recovery_rates::Vector{Float64}      # Recovery rates
    correlation_matrix::Matrix{Float64}  # Asset correlation
    sector_loadings::Matrix{Float64}     # N x K factor loadings
end

"""
    portfolio_credit_var_mc(pcr::PortfolioCreditRisk; num_sims=100000,
                            confidence=0.99, seed=42)

Monte Carlo simulation for portfolio credit VaR with correlated defaults.
Uses single-factor or multi-factor Gaussian copula.
"""
function portfolio_credit_var_mc(pcr::PortfolioCreditRisk;
                                  num_sims::Int=100000,
                                  confidence::Float64=0.99,
                                  seed::Int=42)
    rng = Random.MersenneTwister(seed)
    N = pcr.num_obligors
    K = size(pcr.sector_loadings, 2)

    # Precompute default thresholds
    thresholds = [normal_inv(pcr.default_probs[i]) for i in 1:N]

    # Cholesky of factor correlation (assume identity for simplicity)
    # Each obligor: X_i = sum_k w_ik * Z_k + sqrt(1 - sum w_ik^2) * eps_i
    idio_var = zeros(N)
    for i in 1:N
        systematic_var = sum(pcr.sector_loadings[i, k]^2 for k in 1:K)
        idio_var[i] = sqrt(max(1.0 - systematic_var, 0.0))
    end

    losses = Vector{Float64}(undef, num_sims)

    for sim in 1:num_sims
        # Generate systematic factors
        Z = randn(rng, K)

        # Generate idiosyncratic shocks and compute defaults
        loss = 0.0
        for i in 1:N
            systematic = sum(pcr.sector_loadings[i, k] * Z[k] for k in 1:K)
            eps_i = randn(rng)
            X_i = systematic + idio_var[i] * eps_i

            if X_i < thresholds[i]
                loss += pcr.exposures[i] * (1.0 - pcr.recovery_rates[i])
            end
        end
        losses[sim] = loss
    end

    sort!(losses)

    var_idx = ceil(Int, confidence * num_sims)
    var_idx = min(var_idx, num_sims)
    credit_var = losses[var_idx]

    expected_loss = mean(losses)
    unexpected_loss = credit_var - expected_loss

    # ES = mean of losses beyond VaR
    tail_start = var_idx
    es = mean(losses[tail_start:end])

    return (var=credit_var, expected_loss=expected_loss,
            unexpected_loss=unexpected_loss, es=es,
            loss_distribution=losses)
end

"""
    portfolio_credit_var_analytic(pcr::PortfolioCreditRisk; confidence=0.99)

Vasicek single-factor analytic approximation for credit VaR.
"""
function portfolio_credit_var_analytic(pcr::PortfolioCreditRisk;
                                       confidence::Float64=0.99)::Float64
    total_loss = 0.0
    for i in 1:pcr.num_obligors
        rho_i = sum(pcr.sector_loadings[i, :]'.^2)
        rho_i = min(rho_i, 0.999)
        pd_i = pcr.default_probs[i]
        lgd_i = 1.0 - pcr.recovery_rates[i]
        ead_i = pcr.exposures[i]

        # Vasicek formula
        conditional_pd = normal_cdf((normal_inv(pd_i) + sqrt(rho_i) * normal_inv(confidence)) /
                                     sqrt(1.0 - rho_i))
        total_loss += ead_i * lgd_i * conditional_pd
    end
    return total_loss
end

"""
    granularity_adjustment(pcr::PortfolioCreditRisk; confidence=0.99)

Gordy (2003) granularity adjustment for portfolio concentration.
"""
function granularity_adjustment(pcr::PortfolioCreditRisk; confidence::Float64=0.99)::Float64
    N = pcr.num_obligors
    K = size(pcr.sector_loadings, 2)

    z_alpha = normal_inv(confidence)

    hhi_ead = sum((pcr.exposures[i] / sum(pcr.exposures))^2 for i in 1:N)

    ga = 0.0
    for i in 1:N
        rho_i = sum(pcr.sector_loadings[i, k]^2 for k in 1:K)
        rho_i = min(rho_i, 0.999)
        pd_i = pcr.default_probs[i]
        lgd_i = 1.0 - pcr.recovery_rates[i]
        ead_i = pcr.exposures[i]
        w_i = ead_i / sum(pcr.exposures)

        threshold = normal_inv(pd_i)
        cond_pd = normal_cdf((threshold + sqrt(rho_i) * z_alpha) / sqrt(1.0 - rho_i))

        # Second moment contribution
        ga += w_i^2 * lgd_i^2 * cond_pd * (1.0 - cond_pd)
    end

    return 0.5 * ga * sum(pcr.exposures)
end

"""
    sector_concentration_risk(pcr::PortfolioCreditRisk)

Herfindahl index by sector and total concentration risk.
"""
function sector_concentration_risk(pcr::PortfolioCreditRisk)
    N = pcr.num_obligors
    K = size(pcr.sector_loadings, 2)
    total_exposure = sum(pcr.exposures)

    # Single-name concentration (HHI)
    hhi_name = sum((pcr.exposures[i] / total_exposure)^2 for i in 1:N)

    # Sector concentration
    sector_exposures = zeros(K)
    for i in 1:N
        for k in 1:K
            sector_exposures[k] += pcr.exposures[i] * abs(pcr.sector_loadings[i, k])
        end
    end
    total_sector = sum(sector_exposures)
    hhi_sector = total_sector > 0.0 ?
        sum((sector_exposures[k] / total_sector)^2 for k in 1:K) : 1.0

    return (hhi_name=hhi_name, hhi_sector=hhi_sector,
            sector_exposures=sector_exposures)
end

# ============================================================================
# SECTION 8: Loss Given Default Models
# ============================================================================

"""
    lgd_beta_distribution(mean_lgd, std_lgd, x)

Beta distribution PDF for LGD modeling.
"""
function lgd_beta_distribution(mean_lgd::Float64, std_lgd::Float64, x::Float64)::Float64
    if x <= 0.0 || x >= 1.0
        return 0.0
    end
    v = std_lgd^2
    alpha = mean_lgd * (mean_lgd * (1.0 - mean_lgd) / v - 1.0)
    beta_param = (1.0 - mean_lgd) * (mean_lgd * (1.0 - mean_lgd) / v - 1.0)

    if alpha <= 0.0 || beta_param <= 0.0
        return 0.0
    end

    log_pdf = (alpha - 1.0) * log(x) + (beta_param - 1.0) * log(1.0 - x) +
              lgamma(alpha + beta_param) - lgamma(alpha) - lgamma(beta_param)
    return exp(log_pdf)
end

"""
    lgd_sample_beta(mean_lgd, std_lgd, n; rng=Random.GLOBAL_RNG)

Sample from beta-distributed LGD.
Uses rejection sampling.
"""
function lgd_sample_beta(mean_lgd::Float64, std_lgd::Float64, n::Int;
                          rng::AbstractRNG=Random.GLOBAL_RNG)::Vector{Float64}
    v = std_lgd^2
    alpha = mean_lgd * (mean_lgd * (1.0 - mean_lgd) / v - 1.0)
    beta_param = (1.0 - mean_lgd) * (mean_lgd * (1.0 - mean_lgd) / v - 1.0)

    alpha = max(alpha, 0.01)
    beta_param = max(beta_param, 0.01)

    # Gamma-based sampling for beta distribution
    samples = Vector{Float64}(undef, n)
    for i in 1:n
        # Generate gamma(alpha) and gamma(beta) using Marsaglia-Tsang
        g1 = _sample_gamma(alpha, rng)
        g2 = _sample_gamma(beta_param, rng)
        samples[i] = g1 / (g1 + g2)
    end
    return samples
end

"""
    _sample_gamma(shape, rng)

Sample from Gamma(shape, 1) using Marsaglia-Tsang method.
"""
function _sample_gamma(shape::Float64, rng::AbstractRNG)::Float64
    if shape < 1.0
        return _sample_gamma(shape + 1.0, rng) * rand(rng)^(1.0 / shape)
    end

    d = shape - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)

    while true
        x = randn(rng)
        v = (1.0 + c * x)^3
        if v > 0.0
            u = rand(rng)
            if u < 1.0 - 0.0331 * x^4 ||
               log(u) < 0.5 * x^2 + d * (1.0 - v + log(v))
                return d * v
            end
        end
    end
end

"""
    lgd_seniority_adjustment(base_lgd, seniority_class)

Adjust LGD based on debt seniority.
Seniority: 1=Senior Secured, 2=Senior Unsecured, 3=Subordinated, 4=Junior Sub.
"""
function lgd_seniority_adjustment(base_lgd::Float64, seniority_class::Int)::Float64
    # Historical average recovery rates by seniority (Moody's)
    recovery_map = Dict(
        1 => 0.535,  # Senior Secured
        2 => 0.368,  # Senior Unsecured
        3 => 0.283,  # Subordinated
        4 => 0.155   # Junior Subordinated
    )

    base_recovery = 1.0 - base_lgd
    target_recovery = get(recovery_map, seniority_class, 0.368)
    ratio = target_recovery / max(base_recovery, 0.01)
    adjusted_recovery = base_recovery * ratio
    adjusted_recovery = clamp(adjusted_recovery, 0.0, 1.0)
    return 1.0 - adjusted_recovery
end

"""
    lgd_downturn_adjustment(base_lgd, current_gdp_growth, long_run_gdp)

Downturn LGD: adjust base LGD for economic conditions.
"""
function lgd_downturn_adjustment(base_lgd::Float64, current_gdp_growth::Float64,
                                  long_run_gdp::Float64)::Float64
    # Frye-Jacobs model: LGD increases in downturns
    stress_factor = max(0.0, long_run_gdp - current_gdp_growth) / max(long_run_gdp, 0.01)
    downturn_lgd = base_lgd + 0.08 * stress_factor + 0.12 * stress_factor^2
    return clamp(downturn_lgd, 0.0, 1.0)
end

"""
    collateral_adjusted_lgd(ead, collateral_value, haircut, seniority)

LGD adjusted for collateral.
"""
function collateral_adjusted_lgd(ead::Float64, collateral_value::Float64,
                                  haircut::Float64, seniority::Int)::Float64
    effective_collateral = collateral_value * (1.0 - haircut)
    uncovered = max(ead - effective_collateral, 0.0)

    if ead <= 0.0
        return 0.0
    end

    base_lgd = uncovered / ead
    return lgd_seniority_adjustment(base_lgd, seniority)
end

# ============================================================================
# SECTION 9: Credit Scoring (Logistic Regression)
# ============================================================================

"""
    LogisticScorecard

Logistic regression-based credit scorecard.
"""
struct LogisticScorecard
    coefficients::Vector{Float64}
    intercept::Float64
    feature_names::Vector{String}
    woe_bins::Vector{Vector{Tuple{Float64, Float64, Float64}}}  # (lo, hi, woe) per feature
    base_score::Float64
    pdo::Float64
    base_odds::Float64
end

"""
    sigmoid(x)

Logistic sigmoid function.
"""
function sigmoid(x::Float64)::Float64
    if x > 500.0
        return 1.0
    elseif x < -500.0
        return 0.0
    end
    return 1.0 / (1.0 + exp(-x))
end

"""
    logistic_regression_fit(X, y; max_iter=100, lr=0.01, l2_reg=0.001, tol=1e-8)

Fit logistic regression via IRLS (Iteratively Reweighted Least Squares).
X: n x p feature matrix, y: n-vector of 0/1 labels.
"""
function logistic_regression_fit(X::Matrix{Float64}, y::Vector{Float64};
                                  max_iter::Int=100, lr::Float64=0.01,
                                  l2_reg::Float64=0.001, tol::Float64=1e-8)
    n, p = size(X)

    # Add intercept
    X_aug = hcat(ones(n), X)
    p_aug = p + 1
    beta = zeros(p_aug)

    for iter in 1:max_iter
        # Compute probabilities
        eta = X_aug * beta
        mu = [sigmoid(eta[i]) for i in 1:n]

        # Diagonal weight matrix W = mu * (1 - mu)
        w = [mu[i] * (1.0 - mu[i]) for i in 1:n]
        w = max.(w, 1e-10)

        # Working response
        z = eta + (y - mu) ./ w

        # IRLS update: beta = (X'WX + lambda*I)^{-1} X'Wz
        W = Diagonal(w)
        XtWX = X_aug' * W * X_aug + l2_reg * I
        XtWz = X_aug' * (W * z)

        beta_new = XtWX \ XtWz

        if norm(beta_new - beta) < tol
            beta = beta_new
            break
        end
        beta = beta_new
    end

    intercept = beta[1]
    coefficients = beta[2:end]

    return intercept, coefficients
end

"""
    compute_woe_iv(feature, target, num_bins)

Weight of Evidence and Information Value for a single feature.
"""
function compute_woe_iv(feature::Vector{Float64}, target::Vector{Float64},
                        num_bins::Int)
    n = length(feature)
    sorted_idx = sortperm(feature)

    bin_size = max(1, n ÷ num_bins)
    bins = Vector{Tuple{Float64, Float64, Float64}}()
    total_good = sum(1.0 .- target)
    total_bad = sum(target)

    iv = 0.0

    for b in 1:num_bins
        start_idx = (b - 1) * bin_size + 1
        end_idx = b == num_bins ? n : b * bin_size
        if start_idx > n
            break
        end

        bin_indices = sorted_idx[start_idx:end_idx]
        lo = feature[sorted_idx[start_idx]]
        hi = feature[sorted_idx[end_idx]]

        goods = sum(1.0 - target[i] for i in bin_indices)
        bads = sum(target[i] for i in bin_indices)

        dist_good = max(goods / total_good, 1e-10)
        dist_bad = max(bads / total_bad, 1e-10)

        woe = log(dist_good / dist_bad)
        iv += (dist_good - dist_bad) * woe

        push!(bins, (lo, hi, woe))
    end

    return bins, iv
end

"""
    build_scorecard(X, y, feature_names; num_woe_bins=10, base_score=600.0,
                    pdo=20.0, base_odds=50.0)

Build complete credit scorecard.
"""
function build_scorecard(X::Matrix{Float64}, y::Vector{Float64},
                         feature_names::Vector{String};
                         num_woe_bins::Int=10, base_score::Float64=600.0,
                         pdo::Float64=20.0, base_odds::Float64=50.0)::LogisticScorecard
    n, p = size(X)

    # Compute WoE transformation for each feature
    woe_bins_all = Vector{Vector{Tuple{Float64, Float64, Float64}}}()
    X_woe = zeros(n, p)

    for j in 1:p
        bins, iv = compute_woe_iv(X[:, j], y, num_woe_bins)
        push!(woe_bins_all, bins)

        # Transform feature to WoE values
        for i in 1:n
            val = X[i, j]
            woe_val = 0.0
            for (lo, hi, woe) in bins
                if val >= lo && val <= hi
                    woe_val = woe
                    break
                end
            end
            X_woe[i, j] = woe_val
        end
    end

    # Fit logistic regression on WoE-transformed features
    intercept, coefficients = logistic_regression_fit(X_woe, y)

    return LogisticScorecard(coefficients, intercept, feature_names, woe_bins_all,
                              base_score, pdo, base_odds)
end

"""
    scorecard_predict_proba(sc::LogisticScorecard, x)

Predict default probability for a single observation.
"""
function scorecard_predict_proba(sc::LogisticScorecard, x::Vector{Float64})::Float64
    p = length(sc.coefficients)
    logit = sc.intercept

    for j in 1:p
        val = x[j]
        woe_val = 0.0
        for (lo, hi, woe) in sc.woe_bins[j]
            if val >= lo && val <= hi
                woe_val = woe
                break
            end
        end
        logit += sc.coefficients[j] * woe_val
    end

    return sigmoid(logit)
end

"""
    scorecard_score(sc::LogisticScorecard, x)

Convert probability to credit score.
"""
function scorecard_score(sc::LogisticScorecard, x::Vector{Float64})::Float64
    prob = scorecard_predict_proba(sc, x)
    odds = (1.0 - prob) / max(prob, 1e-10)
    factor = sc.pdo / log(2.0)
    offset = sc.base_score - factor * log(sc.base_odds)
    return offset + factor * log(odds)
end

"""
    ks_statistic(scores, labels)

Kolmogorov-Smirnov statistic for scorecard validation.
"""
function ks_statistic(scores::Vector{Float64}, labels::Vector{Float64})::Float64
    n = length(scores)
    sorted_idx = sortperm(scores)

    total_good = sum(1.0 .- labels)
    total_bad = sum(labels)

    cum_good = 0.0
    cum_bad = 0.0
    max_ks = 0.0

    for i in 1:n
        idx = sorted_idx[i]
        if labels[idx] < 0.5
            cum_good += 1.0
        else
            cum_bad += 1.0
        end
        ks = abs(cum_good / total_good - cum_bad / total_bad)
        max_ks = max(max_ks, ks)
    end

    return max_ks
end

"""
    gini_coefficient(scores, labels)

Gini coefficient (2*AUC - 1) for scorecard discrimination.
"""
function gini_coefficient(scores::Vector{Float64}, labels::Vector{Float64})::Float64
    n = length(scores)
    sorted_idx = sortperm(scores, rev=true)

    total_good = sum(1.0 .- labels)
    total_bad = sum(labels)

    cum_bad = 0.0
    cum_good = 0.0
    auc = 0.0

    for i in 1:n
        idx = sorted_idx[i]
        if labels[idx] > 0.5
            cum_bad += 1.0
        else
            cum_good += 1.0
            auc += cum_bad
        end
    end

    auc /= (total_good * total_bad)
    return 2.0 * auc - 1.0
end

"""
    population_stability_index(expected, actual, num_bins)

PSI: measure of score distribution shift over time.
"""
function population_stability_index(expected::Vector{Float64}, actual::Vector{Float64},
                                     num_bins::Int)::Float64
    # Bin the expected distribution
    min_val = min(minimum(expected), minimum(actual))
    max_val = max(maximum(expected), maximum(actual))
    bin_width = (max_val - min_val) / num_bins

    psi = 0.0
    for b in 1:num_bins
        lo = min_val + (b - 1) * bin_width
        hi = min_val + b * bin_width

        exp_frac = max(count(x -> lo <= x < hi, expected) / length(expected), 1e-10)
        act_frac = max(count(x -> lo <= x < hi, actual) / length(actual), 1e-10)

        psi += (act_frac - exp_frac) * log(act_frac / exp_frac)
    end

    return psi
end

# ============================================================================
# SECTION 10: Altman Z-Score
# ============================================================================

"""
    altman_z_score(working_capital, total_assets, retained_earnings,
                   ebit, market_equity, total_liabilities, sales)

Original Altman Z-Score for manufacturing firms.
Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
"""
function altman_z_score(working_capital::Float64, total_assets::Float64,
                        retained_earnings::Float64, ebit::Float64,
                        market_equity::Float64, total_liabilities::Float64,
                        sales::Float64)::Float64
    if total_assets <= 0.0
        return 0.0
    end
    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = ebit / total_assets
    X4 = market_equity / max(total_liabilities, 1e-10)
    X5 = sales / total_assets

    return 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5
end

"""
    altman_z_score_classification(z)

Classify based on Z-Score zones.
"""
function altman_z_score_classification(z::Float64)::String
    if z > 2.99
        return "Safe"
    elseif z > 1.81
        return "Grey"
    else
        return "Distress"
    end
end

"""
    altman_z_prime(working_capital, total_assets, retained_earnings, ebit,
                   book_equity, total_liabilities, sales)

Z'-Score for private firms (book equity instead of market equity).
"""
function altman_z_prime(working_capital::Float64, total_assets::Float64,
                        retained_earnings::Float64, ebit::Float64,
                        book_equity::Float64, total_liabilities::Float64,
                        sales::Float64)::Float64
    if total_assets <= 0.0
        return 0.0
    end
    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = ebit / total_assets
    X4 = book_equity / max(total_liabilities, 1e-10)
    X5 = sales / total_assets

    return 0.717 * X1 + 0.847 * X2 + 3.107 * X3 + 0.420 * X4 + 0.998 * X5
end

"""
    altman_z_double_prime(working_capital, total_assets, retained_earnings, ebit,
                          book_equity, total_liabilities)

Z''-Score for non-manufacturing and emerging market firms.
Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
"""
function altman_z_double_prime(working_capital::Float64, total_assets::Float64,
                                retained_earnings::Float64, ebit::Float64,
                                book_equity::Float64, total_liabilities::Float64)::Float64
    if total_assets <= 0.0
        return 0.0
    end
    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = ebit / total_assets
    X4 = book_equity / max(total_liabilities, 1e-10)

    return 6.56 * X1 + 3.26 * X2 + 6.72 * X3 + 1.05 * X4
end

"""
    ohlson_o_score(total_assets, total_liabilities, working_capital,
                   current_liabilities, current_assets, net_income,
                   funds_from_operations, gnp_deflator)

Ohlson (1980) O-Score for bankruptcy prediction.
"""
function ohlson_o_score(total_assets::Float64, total_liabilities::Float64,
                        working_capital::Float64, current_liabilities::Float64,
                        current_assets::Float64, net_income::Float64,
                        funds_from_operations::Float64, gnp_deflator::Float64)::Float64
    log_ta = log(total_assets / gnp_deflator)
    tlta = total_liabilities / total_assets
    wcta = working_capital / total_assets
    clca = current_liabilities / max(current_assets, 1e-10)
    ni_flag = total_liabilities > total_assets ? 1.0 : 0.0
    nita = net_income / total_assets
    ffota = funds_from_operations / total_liabilities
    ni_change_flag = net_income < 0.0 ? 1.0 : 0.0

    o = -1.32 - 0.407 * log_ta + 6.03 * tlta - 1.43 * wcta + 0.0757 * clca -
        1.72 * ni_flag - 2.37 * nita - 1.83 * ffota + 0.285 * ni_change_flag

    return o
end

# ============================================================================
# SECTION 11: Credit Spread Decomposition
# ============================================================================

"""
    credit_spread_decomposition(observed_spread, pd, lgd, risk_free, maturity,
                                bid_ask_spread, funding_spread)

Decompose observed credit spread into components.
Spread = Default Component + Liquidity Premium + Risk Premium
"""
function credit_spread_decomposition(observed_spread::Float64, pd::Float64,
                                      lgd::Float64, risk_free::Float64,
                                      maturity::Float64, bid_ask_spread::Float64,
                                      funding_spread::Float64)
    # Default component: approximate as pd * lgd / (1 - pd)
    default_component = pd * lgd / max(1.0 - pd, 1e-10)

    # Liquidity premium: function of bid-ask and funding
    liquidity_premium = 0.5 * bid_ask_spread + 0.3 * funding_spread

    # Risk premium: residual
    risk_premium = observed_spread - default_component - liquidity_premium
    risk_premium = max(risk_premium, 0.0)

    # Decomposition percentages
    total = default_component + liquidity_premium + risk_premium
    if total < 1e-15
        return (default_component=0.0, liquidity_premium=0.0, risk_premium=0.0,
                default_pct=0.0, liquidity_pct=0.0, risk_pct=0.0)
    end

    return (default_component=default_component,
            liquidity_premium=liquidity_premium,
            risk_premium=risk_premium,
            default_pct=default_component / total,
            liquidity_pct=liquidity_premium / total,
            risk_pct=risk_premium / total)
end

"""
    credit_spread_option_adjusted(spread, embedded_option_value, duration)

Option-Adjusted Spread (OAS).
"""
function credit_spread_option_adjusted(spread::Float64, embedded_option_value::Float64,
                                        duration::Float64)::Float64
    return spread - embedded_option_value / max(duration, 0.001)
end

"""
    credit_spread_term_structure(pds, lgds, risk_free_curve, tenors)

Term structure of credit spreads from PD and LGD curves.
"""
function credit_spread_term_structure(pds::Vector{Float64}, lgds::Vector{Float64},
                                      risk_free_curve::Vector{Float64},
                                      tenors::Vector{Float64})::Vector{Float64}
    n = length(tenors)
    spreads = Vector{Float64}(undef, n)

    for i in 1:n
        # Hazard rate implied by cumulative PD
        if pds[i] < 1.0 - 1e-15
            h = -log(1.0 - pds[i]) / tenors[i]
        else
            h = 10.0
        end
        spreads[i] = h * lgds[i]
    end

    return spreads
end

# ============================================================================
# SECTION 12: Counterparty Credit Risk (CVA/DVA/FVA)
# ============================================================================

"""
    CVAEngine

Counterparty Credit Risk computation engine.
"""
struct CVAEngine
    exposure_times::Vector{Float64}
    expected_exposures::Vector{Float64}      # EE(t)
    counterparty_hazard::HazardRateCurve
    own_hazard::HazardRateCurve
    recovery_rate_cpty::Float64
    recovery_rate_own::Float64
    risk_free_rate::Float64
    funding_spread::Float64
end

"""
    compute_cva(eng::CVAEngine)

Unilateral CVA: CVA = (1-R) * integral EE(t) * h(t) * Q(t) * df(t) dt.
"""
function compute_cva(eng::CVAEngine)::Float64
    cva = 0.0
    n = length(eng.exposure_times)

    for i in 2:n
        t = eng.exposure_times[i]
        t_prev = eng.exposure_times[i-1]
        dt = t - t_prev

        ee_mid = 0.5 * (eng.expected_exposures[i] + eng.expected_exposures[i-1])
        q_mid = survival_probability(eng.counterparty_hazard, 0.5 * (t + t_prev))
        h_mid = hazard_rate_at(eng.counterparty_hazard, 0.5 * (t + t_prev))
        df = exp(-eng.risk_free_rate * 0.5 * (t + t_prev))

        cva += (1.0 - eng.recovery_rate_cpty) * ee_mid * h_mid * q_mid * df * dt
    end

    return cva
end

"""
    compute_dva(eng::CVAEngine)

Bilateral DVA: DVA = (1-R_own) * integral NEE(t) * h_own(t) * Q_own(t) * df(t) dt.
"""
function compute_dva(eng::CVAEngine)::Float64
    dva = 0.0
    n = length(eng.exposure_times)

    for i in 2:n
        t = eng.exposure_times[i]
        t_prev = eng.exposure_times[i-1]
        dt = t - t_prev

        # Negative expected exposure (what we owe)
        nee_mid = 0.5 * (eng.expected_exposures[i] + eng.expected_exposures[i-1])
        nee_mid = max(-nee_mid, 0.0)  # Flip sign for DVA

        q_own = survival_probability(eng.own_hazard, 0.5 * (t + t_prev))
        h_own = hazard_rate_at(eng.own_hazard, 0.5 * (t + t_prev))
        df = exp(-eng.risk_free_rate * 0.5 * (t + t_prev))

        dva += (1.0 - eng.recovery_rate_own) * nee_mid * h_own * q_own * df * dt
    end

    return dva
end

"""
    compute_fva(eng::CVAEngine)

Funding Value Adjustment.
FVA = integral (EE(t) * s_f * Q(t) * df(t)) dt
"""
function compute_fva(eng::CVAEngine)::Float64
    fva = 0.0
    n = length(eng.exposure_times)

    for i in 2:n
        t = eng.exposure_times[i]
        t_prev = eng.exposure_times[i-1]
        dt = t - t_prev

        ee_mid = 0.5 * (eng.expected_exposures[i] + eng.expected_exposures[i-1])
        q = survival_probability(eng.counterparty_hazard, 0.5 * (t + t_prev))
        df = exp(-eng.risk_free_rate * 0.5 * (t + t_prev))

        fva += eng.funding_spread * ee_mid * q * df * dt
    end

    return fva
end

"""
    compute_bilateral_cva(eng::CVAEngine)

Bilateral CVA = CVA - DVA.
"""
function compute_bilateral_cva(eng::CVAEngine)::Float64
    return compute_cva(eng) - compute_dva(eng)
end

"""
    exposure_simulation_irs(notional, fixed_rate, float_rate_vol, maturity,
                            risk_free, num_paths, num_steps; seed=42)

Simulate exposure profiles for an interest rate swap.
Returns (times, expected_exposure, potential_future_exposure_97.5).
"""
function exposure_simulation_irs(notional::Float64, fixed_rate::Float64,
                                  float_rate_vol::Float64, maturity::Float64,
                                  risk_free::Float64, num_paths::Int,
                                  num_steps::Int; seed::Int=42)
    rng = Random.MersenneTwister(seed)
    dt = maturity / num_steps
    times = [i * dt for i in 0:num_steps]

    exposures = zeros(num_paths, num_steps + 1)

    for path in 1:num_paths
        r = risk_free
        for step in 1:num_steps
            t = step * dt
            dr = float_rate_vol * sqrt(dt) * randn(rng)
            r += dr

            # Swap value: remaining fixed payments - floating
            remaining = maturity - t
            if remaining > 0
                # Simplified: swap value proportional to rate difference * remaining annuity
                annuity = (1.0 - exp(-r * remaining)) / max(r, 1e-10)
                swap_val = notional * (r - fixed_rate) * annuity
                exposures[path, step+1] = max(swap_val, 0.0)
            end
        end
    end

    ee = vec(mean(exposures, dims=1))
    pfe = [length(exposures[:, i]) > 0 ?
           sort(exposures[:, i])[ceil(Int, 0.975 * num_paths)] : 0.0
           for i in 1:(num_steps+1)]

    return times, ee, pfe
end

"""
    expected_positive_exposure(times, ee)

EPE: time-averaged expected exposure.
"""
function expected_positive_exposure(times::Vector{Float64},
                                     ee::Vector{Float64})::Float64
    if length(times) < 2
        return 0.0
    end
    T = times[end]
    integral = 0.0
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        integral += 0.5 * (ee[i] + ee[i-1]) * dt
    end
    return integral / T
end

"""
    effective_expected_exposure(times, ee)

Effective EE: non-decreasing profile.
"""
function effective_expected_exposure(times::Vector{Float64},
                                      ee::Vector{Float64})::Vector{Float64}
    n = length(ee)
    eff_ee = copy(ee)
    for i in 2:n
        eff_ee[i] = max(eff_ee[i], eff_ee[i-1])
    end
    return eff_ee
end

# ============================================================================
# SECTION 13: Wrong-Way Risk
# ============================================================================

"""
    wrong_way_risk_adjustment(base_cva, exposure_default_correlation,
                               exposure_vol, hazard_rate)

Adjust CVA for wrong-way risk: positive correlation between exposure
and counterparty default.
Uses Hull-White (2012) alpha-factor approach.
"""
function wrong_way_risk_adjustment(base_cva::Float64, exposure_default_corr::Float64,
                                    exposure_vol::Float64, hazard_rate::Float64)::Float64
    # Alpha factor: multiplicative adjustment
    alpha = 1.0 + exposure_default_corr * exposure_vol / max(hazard_rate, 1e-10)
    alpha = max(alpha, 0.5)  # Floor
    alpha = min(alpha, 3.0)  # Cap
    return base_cva * alpha
end

"""
    wrong_way_risk_mc(exposure_paths, default_times, recovery_rate, risk_free;
                      exposure_default_corr=0.3)

Monte Carlo CVA with wrong-way risk via correlated exposure and default.
exposure_paths: N_paths x N_steps matrix
default_times: N_paths vector (Inf if no default)
"""
function wrong_way_risk_mc(exposure_paths::Matrix{Float64},
                           default_times::Vector{Float64},
                           recovery_rate::Float64, risk_free::Float64;
                           exposure_default_corr::Float64=0.3)::Float64
    num_paths = size(exposure_paths, 1)
    num_steps = size(exposure_paths, 2)
    maturity = 5.0  # Assume 5Y
    dt = maturity / (num_steps - 1)

    cva = 0.0
    for path in 1:num_paths
        tau = default_times[path]
        if tau < maturity && tau > 0.0
            step = min(ceil(Int, tau / dt), num_steps)
            exposure_at_default = exposure_paths[path, step]
            df = exp(-risk_free * tau)
            cva += (1.0 - recovery_rate) * exposure_at_default * df
        end
    end

    return cva / num_paths
end

"""
    generate_correlated_default_exposure(num_paths, num_steps, maturity,
                                          hazard_rate, exposure_vol, correlation;
                                          seed=42)

Generate correlated exposure paths and default times for wrong-way risk analysis.
"""
function generate_correlated_default_exposure(num_paths::Int, num_steps::Int,
                                               maturity::Float64, hazard_rate::Float64,
                                               exposure_vol::Float64, correlation::Float64;
                                               seed::Int=42)
    rng = Random.MersenneTwister(seed)
    dt = maturity / num_steps

    exposure_paths = zeros(num_paths, num_steps + 1)
    default_times = fill(Inf, num_paths)

    for path in 1:num_paths
        # Correlated Brownian motions for exposure and default intensity
        exposure = 1.0
        survived = true

        for step in 1:num_steps
            z1 = randn(rng)
            z2 = randn(rng)
            w_exp = z1
            w_def = correlation * z1 + sqrt(1.0 - correlation^2) * z2

            # Exposure evolution (GBM)
            exposure *= exp(-0.5 * exposure_vol^2 * dt + exposure_vol * sqrt(dt) * w_exp)
            exposure_paths[path, step+1] = max(exposure, 0.0)

            # Default check (intensity model with stochastic intensity)
            lambda_t = hazard_rate * exp(0.3 * w_def * sqrt(dt))
            if survived && rand(rng) < lambda_t * dt
                default_times[path] = step * dt
                survived = false
            end
        end
    end

    return exposure_paths, default_times
end

# ============================================================================
# SECTION 14: Credit Contagion on Interbank Networks
# ============================================================================

"""
    InterbankNetwork

Interbank network for contagion modeling.
"""
struct InterbankNetwork
    num_banks::Int
    assets::Vector{Float64}            # Total assets
    liabilities::Vector{Float64}       # Total liabilities
    capital::Vector{Float64}           # Equity capital
    exposure_matrix::Matrix{Float64}   # L[i,j] = bank i's exposure to bank j
end

"""
    create_interbank_network(num_banks, avg_assets, capital_ratio, connectivity;
                              seed=42)

Generate a random interbank network.
"""
function create_interbank_network(num_banks::Int, avg_assets::Float64,
                                   capital_ratio::Float64, connectivity::Float64;
                                   seed::Int=42)::InterbankNetwork
    rng = Random.MersenneTwister(seed)
    assets = avg_assets .* (0.5 .+ rand(rng, num_banks))
    capital = capital_ratio .* assets
    liabilities = assets .- capital

    # Generate exposure matrix
    exposure_matrix = zeros(num_banks, num_banks)
    for i in 1:num_banks
        for j in 1:num_banks
            if i != j && rand(rng) < connectivity
                # Exposure proportional to assets
                exposure_matrix[i, j] = 0.05 * assets[i] * rand(rng)
            end
        end
    end

    return InterbankNetwork(num_banks, assets, liabilities, capital, exposure_matrix)
end

"""
    eisenberg_noe_clearing(network::InterbankNetwork, external_assets)

Eisenberg-Noe (2001) clearing payments algorithm.
Finds fixed-point payment vector in interbank network.
"""
function eisenberg_noe_clearing(network::InterbankNetwork,
                                 external_assets::Vector{Float64})
    N = network.num_banks
    L = network.exposure_matrix

    # Total obligations
    p_bar = vec(sum(L, dims=2))  # p_bar[i] = total outgoing obligations

    # Relative liability matrix
    Pi = zeros(N, N)
    for i in 1:N
        if p_bar[i] > 0
            for j in 1:N
                Pi[i, j] = L[i, j] / p_bar[i]
            end
        end
    end

    # Fixed-point iteration
    p = copy(p_bar)  # Start with full payment
    for iter in 1:1000
        p_new = zeros(N)
        for i in 1:N
            # Assets = external + incoming payments
            incoming = sum(Pi[j, i] * p[j] for j in 1:N if j != i)
            total_assets_i = external_assets[i] + incoming

            # Payment = min(obligations, assets)
            p_new[i] = min(p_bar[i], max(total_assets_i, 0.0))
        end

        if norm(p_new - p) < 1e-10
            p = p_new
            break
        end
        p = p_new
    end

    # Determine defaults
    defaults = [p[i] < p_bar[i] - 1e-8 for i in 1:N]

    return (payments=p, obligations=p_bar, defaults=defaults)
end

"""
    credit_contagion_cascade(network::InterbankNetwork, initial_defaults;
                              loss_given_default=0.6)

Simulate default cascade from initial bank failures.
"""
function credit_contagion_cascade(network::InterbankNetwork,
                                   initial_defaults::Vector{Int};
                                   loss_given_default::Float64=0.6)
    N = network.num_banks
    defaulted = falses(N)
    capital_remaining = copy(network.capital)

    # Initial defaults
    for i in initial_defaults
        defaulted[i] = true
    end

    cascade_rounds = Vector{Vector{Int}}()
    push!(cascade_rounds, copy(initial_defaults))

    while true
        new_defaults = Int[]

        for i in 1:N
            if !defaulted[i]
                # Loss from defaulted counterparties
                loss = 0.0
                for j in 1:N
                    if defaulted[j]
                        loss += network.exposure_matrix[i, j] * loss_given_default
                    end
                end

                capital_remaining[i] = network.capital[i] - loss
                if capital_remaining[i] < 0.0
                    defaulted[i] = true
                    push!(new_defaults, i)
                end
            end
        end

        if isempty(new_defaults)
            break
        end
        push!(cascade_rounds, new_defaults)
    end

    total_defaults = sum(defaulted)
    total_losses = sum(max(network.capital[i] - capital_remaining[i], 0.0)
                       for i in 1:N)

    return (defaulted=defaulted, cascade_rounds=cascade_rounds,
            total_defaults=total_defaults, total_losses=total_losses,
            capital_remaining=capital_remaining)
end

"""
    debtrank(network::InterbankNetwork, initial_shocks::Vector{Float64})

DebtRank (Battiston et al., 2012) systemic impact measure.
initial_shocks[i] = fraction of equity lost for bank i.
"""
function debtrank(network::InterbankNetwork, initial_shocks::Vector{Float64})
    N = network.num_banks
    h = copy(initial_shocks)  # Distress level
    h = clamp.(h, 0.0, 1.0)

    # Leverage-weighted exposure matrix
    W = zeros(N, N)
    for i in 1:N
        for j in 1:N
            if network.capital[i] > 0
                W[j, i] = min(network.exposure_matrix[j, i] / network.capital[j], 1.0)
            end
        end
    end

    state = ones(Int, N)  # 1=undistressed, 2=distressed, 3=inactive
    for i in 1:N
        if h[i] > 0.0
            state[i] = 2
        end
    end

    for round in 1:N
        h_new = copy(h)
        any_change = false

        for i in 1:N
            if state[i] == 1
                # Contagion from distressed neighbors
                stress = 0.0
                for j in 1:N
                    if state[j] == 2
                        stress += W[i, j] * h[j]
                    end
                end

                if stress > 0.0
                    h_new[i] = min(h[i] + stress, 1.0)
                    state[i] = 2
                    any_change = true
                end
            end
        end

        # Move distressed to inactive
        for i in 1:N
            if state[i] == 2 && h_new[i] >= h[i] && round > 1
                state[i] = 3
            end
        end

        h = h_new
        if !any_change
            break
        end
    end

    # DebtRank = sum of economic value lost
    total_assets = sum(network.assets)
    debt_rank = sum(h[i] * network.assets[i] for i in 1:N) / total_assets

    return (debtrank=debt_rank, distress_levels=h, states=state)
end

"""
    systemic_importance(network::InterbankNetwork)

Compute systemic importance of each bank via individual DebtRank.
"""
function systemic_importance(network::InterbankNetwork)::Vector{Float64}
    N = network.num_banks
    importance = Vector{Float64}(undef, N)

    for i in 1:N
        shock = zeros(N)
        shock[i] = 1.0
        result = debtrank(network, shock)
        importance[i] = result.debtrank
    end

    return importance
end

"""
    contagion_threshold(network::InterbankNetwork; num_samples=100, seed=42)

Estimate critical connectivity threshold for contagion via percolation analysis.
"""
function contagion_threshold(network::InterbankNetwork;
                              num_samples::Int=100, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    N = network.num_banks

    cascade_sizes = Vector{Float64}(undef, num_samples)

    for s in 1:num_samples
        # Random single-bank shock
        target = rand(rng, 1:N)
        result = credit_contagion_cascade(network, [target])
        cascade_sizes[s] = result.total_defaults / N
    end

    avg_cascade = mean(cascade_sizes)
    max_cascade = maximum(cascade_sizes)

    # Connectivity
    total_possible = N * (N - 1)
    actual_links = sum(network.exposure_matrix .> 0.0)
    connectivity = actual_links / total_possible

    return (avg_cascade_fraction=avg_cascade, max_cascade_fraction=max_cascade,
            connectivity=connectivity)
end

# ============================================================================
# SECTION 15: Additional Credit Risk Utilities
# ============================================================================

"""
    probability_of_default_from_spread(spread, recovery_rate)

Approximate PD from credit spread: PD ~ spread / (1 - R).
"""
function probability_of_default_from_spread(spread::Float64,
                                             recovery_rate::Float64)::Float64
    return spread / max(1.0 - recovery_rate, 0.01)
end

"""
    implied_hazard_rate(spread, recovery_rate)

Implied constant hazard rate from spread.
"""
function implied_hazard_rate(spread::Float64, recovery_rate::Float64)::Float64
    return spread / max(1.0 - recovery_rate, 0.01)
end

"""
    cumulative_default_prob(hazard_rate, T)

Cumulative default probability: 1 - exp(-h*T).
"""
function cumulative_default_prob(hazard_rate::Float64, T::Float64)::Float64
    return 1.0 - exp(-hazard_rate * T)
end

"""
    marginal_default_prob(hazard_rate, t1, t2)

Marginal (forward) default probability in [t1, t2].
"""
function marginal_default_prob(hazard_rate::Float64, t1::Float64, t2::Float64)::Float64
    return exp(-hazard_rate * t1) - exp(-hazard_rate * t2)
end

"""
    conditional_default_prob_forward(pd_cum_t1, pd_cum_t2)

Conditional default probability: P(default in [t1,t2] | survive to t1).
"""
function conditional_default_prob_forward(pd_cum_t1::Float64,
                                           pd_cum_t2::Float64)::Float64
    surv_t1 = 1.0 - pd_cum_t1
    if surv_t1 < 1e-15
        return 1.0
    end
    return (pd_cum_t2 - pd_cum_t1) / surv_t1
end

"""
    expected_loss_portfolio(exposures, pds, lgds)

Expected loss of a portfolio: EL = sum(EAD * PD * LGD).
"""
function expected_loss_portfolio(exposures::Vector{Float64},
                                  pds::Vector{Float64},
                                  lgds::Vector{Float64})::Float64
    return sum(exposures[i] * pds[i] * lgds[i] for i in 1:length(exposures))
end

"""
    unexpected_loss_portfolio(exposures, pds, lgds, correlations)

Unexpected loss with pairwise correlations.
UL = sqrt(sum_ij w_i * w_j * UL_i * UL_j * rho_ij)
"""
function unexpected_loss_portfolio(exposures::Vector{Float64},
                                    pds::Vector{Float64},
                                    lgds::Vector{Float64},
                                    correlations::Matrix{Float64})::Float64
    n = length(exposures)
    total = sum(exposures)

    var_sum = 0.0
    for i in 1:n
        ul_i = exposures[i] * lgds[i] * sqrt(pds[i] * (1.0 - pds[i]))
        for j in 1:n
            ul_j = exposures[j] * lgds[j] * sqrt(pds[j] * (1.0 - pds[j]))
            var_sum += ul_i * ul_j * correlations[i, j]
        end
    end

    return sqrt(max(var_sum, 0.0))
end

"""
    risk_contribution(exposures, pds, lgds, correlations)

Marginal risk contribution of each obligor.
"""
function risk_contribution(exposures::Vector{Float64},
                           pds::Vector{Float64},
                           lgds::Vector{Float64},
                           correlations::Matrix{Float64})::Vector{Float64}
    n = length(exposures)
    ul_total = unexpected_loss_portfolio(exposures, pds, lgds, correlations)

    if ul_total < 1e-15
        return zeros(n)
    end

    rc = Vector{Float64}(undef, n)
    for i in 1:n
        ul_i = exposures[i] * lgds[i] * sqrt(pds[i] * (1.0 - pds[i]))
        cov_sum = 0.0
        for j in 1:n
            ul_j = exposures[j] * lgds[j] * sqrt(pds[j] * (1.0 - pds[j]))
            cov_sum += ul_j * correlations[i, j]
        end
        rc[i] = ul_i * cov_sum / ul_total
    end

    return rc
end

"""
    economic_capital(exposures, pds, lgds, correlations; confidence=0.999)

Economic capital: VaR(confidence) - EL.
Uses Vasicek approximation with average correlation.
"""
function economic_capital(exposures::Vector{Float64},
                          pds::Vector{Float64},
                          lgds::Vector{Float64},
                          correlations::Matrix{Float64};
                          confidence::Float64=0.999)::Float64
    n = length(exposures)
    el = expected_loss_portfolio(exposures, pds, lgds)

    # Average correlation
    rho_avg = 0.0
    count = 0
    for i in 1:n
        for j in (i+1):n
            rho_avg += correlations[i, j]
            count += 1
        end
    end
    rho_avg = count > 0 ? rho_avg / count : 0.0
    rho_avg = clamp(rho_avg, 0.0, 0.999)

    # Vasicek VaR
    var_total = 0.0
    z_alpha = normal_inv(confidence)
    for i in 1:n
        cond_pd = normal_cdf((normal_inv(pds[i]) + sqrt(rho_avg) * z_alpha) /
                              sqrt(1.0 - rho_avg))
        var_total += exposures[i] * lgds[i] * cond_pd
    end

    return var_total - el
end

"""
    credit_var_contribution(exposures, pds, lgds, correlations;
                            confidence=0.999, epsilon=0.01)

Marginal VaR contribution via finite differences.
"""
function credit_var_contribution(exposures::Vector{Float64},
                                  pds::Vector{Float64},
                                  lgds::Vector{Float64},
                                  correlations::Matrix{Float64};
                                  confidence::Float64=0.999,
                                  epsilon::Float64=0.01)::Vector{Float64}
    n = length(exposures)
    base_ec = economic_capital(exposures, pds, lgds, correlations; confidence=confidence)

    contributions = Vector{Float64}(undef, n)
    for i in 1:n
        exp_bump = copy(exposures)
        exp_bump[i] *= (1.0 + epsilon)
        ec_bump = economic_capital(exp_bump, pds, lgds, correlations; confidence=confidence)
        contributions[i] = (ec_bump - base_ec) / (exposures[i] * epsilon)
    end

    return contributions
end

"""
    stressed_pd(base_pd, stress_factor, asset_correlation)

Stressed PD under adverse scenario.
"""
function stressed_pd(base_pd::Float64, stress_factor::Float64,
                     asset_correlation::Float64)::Float64
    rho = clamp(asset_correlation, 0.0, 0.999)
    threshold = normal_inv(base_pd)
    stressed = normal_cdf((threshold - sqrt(rho) * stress_factor) / sqrt(1.0 - rho))
    return stressed
end

"""
    transition_matrix_from_generators(Q, dt)

Compute transition probability matrix from generator for time step dt.
"""
function transition_matrix_from_generators(Q::Matrix{Float64}, dt::Float64)::Matrix{Float64}
    return matrix_exponential(Q * dt)
end

"""
    through_the_cycle_pd(point_in_time_pds, macro_factor, cycle_sensitivity)

Convert PIT PDs to TTC PDs by removing cyclical component.
"""
function through_the_cycle_pd(point_in_time_pds::Vector{Float64},
                               macro_factor::Float64,
                               cycle_sensitivity::Float64)::Vector{Float64}
    n = length(point_in_time_pds)
    ttc = Vector{Float64}(undef, n)
    for i in 1:n
        logit_pit = log(point_in_time_pds[i] / max(1.0 - point_in_time_pds[i], 1e-15))
        logit_ttc = logit_pit - cycle_sensitivity * macro_factor
        ttc[i] = sigmoid(logit_ttc)
    end
    return ttc
end

"""
    credit_portfolio_optimization(exposures, pds, lgds, correlations,
                                   max_exposure_pct, target_return;
                                   risk_appetite=0.05)

Optimize credit portfolio allocation subject to concentration limits.
Simple gradient-based approach.
"""
function credit_portfolio_optimization(exposures::Vector{Float64},
                                        pds::Vector{Float64},
                                        lgds::Vector{Float64},
                                        correlations::Matrix{Float64},
                                        max_exposure_pct::Float64,
                                        target_return::Float64;
                                        risk_appetite::Float64=0.05)
    n = length(exposures)
    total = sum(exposures)
    weights = exposures / total

    # Expected return proxy: spread - EL
    spreads = [pds[i] * lgds[i] * 1.5 for i in 1:n]  # Approximate spread
    returns = spreads .- pds .* lgds

    # Gradient descent on risk-adjusted return
    lr = 0.001
    for iter in 1:500
        # Portfolio risk
        risk = 0.0
        for i in 1:n
            for j in 1:n
                ul_i = weights[i] * lgds[i] * sqrt(pds[i])
                ul_j = weights[j] * lgds[j] * sqrt(pds[j])
                risk += ul_i * ul_j * correlations[i, j]
            end
        end
        risk = sqrt(max(risk, 0.0))

        # Portfolio return
        port_return = dot(weights, returns)

        # Gradient of Sharpe-like ratio
        for i in 1:n
            grad_return = returns[i]
            grad_risk = 0.0
            if risk > 1e-10
                for j in 1:n
                    ul_j = weights[j] * lgds[j] * sqrt(pds[j])
                    grad_risk += lgds[i] * sqrt(pds[i]) * ul_j * correlations[i, j]
                end
                grad_risk /= risk
            end

            weights[i] += lr * (grad_return - risk_appetite * grad_risk)
        end

        # Project onto constraints
        weights = max.(weights, 0.0)
        weights = min.(weights, max_exposure_pct)
        s = sum(weights)
        if s > 0
            weights ./= s
        end
    end

    optimal_exposures = weights * total
    return (weights=weights, exposures=optimal_exposures)
end

"""
    credit_valuation_adjustment_swaption(notional, strike, expiry, swap_tenor,
                                          vol, hazard_rate, recovery, risk_free)

CVA on a swaption using semi-analytical approach.
"""
function credit_valuation_adjustment_swaption(notional::Float64, strike::Float64,
                                               expiry::Float64, swap_tenor::Float64,
                                               vol::Float64, hazard_rate::Float64,
                                               recovery::Float64,
                                               risk_free::Float64)::Float64
    # Swaption value as function of rate
    annuity = (1.0 - exp(-risk_free * swap_tenor)) / max(risk_free, 1e-10)
    d1 = (log(risk_free / strike) + 0.5 * vol^2 * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)
    swaption_val = notional * annuity * (risk_free * normal_cdf(d1) - strike * normal_cdf(d2))

    # Expected exposure profile (hump-shaped)
    dt = 0.25
    num_steps = ceil(Int, (expiry + swap_tenor) / dt)
    cva = 0.0

    for step in 1:num_steps
        t = step * dt
        if t > expiry + swap_tenor
            break
        end

        # EE at time t
        if t <= expiry
            # Before expiry: value is swaption-like
            remaining_expiry = expiry - t
            if remaining_expiry > 0.01
                d1_t = (log(risk_free / strike) + 0.5 * vol^2 * remaining_expiry) /
                       (vol * sqrt(remaining_expiry))
                d2_t = d1_t - vol * sqrt(remaining_expiry)
                ee_t = notional * annuity * (risk_free * normal_cdf(d1_t) -
                                              strike * normal_cdf(d2_t))
            else
                ee_t = swaption_val * 0.5
            end
        else
            # After expiry: swap value declines
            remaining = expiry + swap_tenor - t
            ee_t = swaption_val * max(remaining / swap_tenor, 0.0) * 0.5
        end

        ee_t = max(ee_t, 0.0)

        # Default probability in this interval
        q_prev = exp(-hazard_rate * (t - dt))
        q_curr = exp(-hazard_rate * t)
        dp = q_prev - q_curr
        df = exp(-risk_free * t)

        cva += (1.0 - recovery) * ee_t * dp * df
    end

    return cva
end

"""
    netting_benefit(gross_exposures, netting_sets)

Compute netting benefit across netting sets.
netting_sets: Vector of vectors, each containing indices in same netting set.
"""
function netting_benefit(gross_exposures::Vector{Float64},
                          netting_sets::Vector{Vector{Int}})
    gross_total = sum(max.(gross_exposures, 0.0))

    net_total = 0.0
    for ns in netting_sets
        net_exposure = sum(gross_exposures[i] for i in ns)
        net_total += max(net_exposure, 0.0)
    end

    benefit = gross_total > 0.0 ? 1.0 - net_total / gross_total : 0.0
    return (gross=gross_total, net=net_total, benefit=benefit)
end

"""
    collateral_margining(exposure, initial_margin, variation_margin,
                          margin_period_of_risk, volatility)

Compute collateralized exposure accounting for margin period of risk.
"""
function collateral_margining(exposure::Float64, initial_margin::Float64,
                               variation_margin::Float64,
                               margin_period_of_risk::Float64,
                               volatility::Float64)::Float64
    # Exposure during MPOR
    mpor_exposure = exposure * volatility * sqrt(margin_period_of_risk / 252.0)
    net_exposure = max(exposure - initial_margin - variation_margin + mpor_exposure, 0.0)
    return net_exposure
end

"""
    incremental_cva(base_portfolio_cva, new_trade_exposure, counterparty_pd,
                     recovery, risk_free, maturity)

Incremental CVA from adding a new trade.
"""
function incremental_cva(base_portfolio_cva::Float64, new_trade_exposure::Float64,
                          counterparty_pd::Float64, recovery::Float64,
                          risk_free::Float64, maturity::Float64)::Float64
    hazard = -log(1.0 - counterparty_pd) / maturity
    standalone_cva = (1.0 - recovery) * new_trade_exposure *
                     (1.0 - exp(-hazard * maturity)) * exp(-risk_free * maturity * 0.5)

    # Incremental = standalone + netting/diversification adjustment
    # Simplified: assume 70% of standalone due to netting
    return 0.7 * standalone_cva
end

"""
    stress_test_credit_portfolio(exposures, pds, lgds, correlations,
                                  pd_stress_mult, lgd_stress_mult, corr_stress_add)

Stress test credit portfolio under adverse scenario.
"""
function stress_test_credit_portfolio(exposures::Vector{Float64},
                                       pds::Vector{Float64},
                                       lgds::Vector{Float64},
                                       correlations::Matrix{Float64},
                                       pd_stress_mult::Float64,
                                       lgd_stress_mult::Float64,
                                       corr_stress_add::Float64)
    n = length(exposures)

    stressed_pds = min.(pds * pd_stress_mult, 1.0)
    stressed_lgds = min.(lgds * lgd_stress_mult, 1.0)
    stressed_corrs = copy(correlations)
    for i in 1:n
        for j in 1:n
            if i != j
                stressed_corrs[i, j] = min(correlations[i, j] + corr_stress_add, 0.999)
            end
        end
    end

    base_el = expected_loss_portfolio(exposures, pds, lgds)
    stressed_el = expected_loss_portfolio(exposures, stressed_pds, stressed_lgds)

    base_ul = unexpected_loss_portfolio(exposures, pds, lgds, correlations)
    stressed_ul = unexpected_loss_portfolio(exposures, stressed_pds, stressed_lgds, stressed_corrs)

    base_ec = economic_capital(exposures, pds, lgds, correlations)
    stressed_ec = economic_capital(exposures, stressed_pds, stressed_lgds, stressed_corrs)

    return (base_el=base_el, stressed_el=stressed_el,
            base_ul=base_ul, stressed_ul=stressed_ul,
            base_ec=base_ec, stressed_ec=stressed_ec,
            el_increase=stressed_el / max(base_el, 1e-15) - 1.0,
            ec_increase=stressed_ec / max(base_ec, 1e-15) - 1.0)
end

"""
    exposure_at_default_distribution(committed, drawn, ccf_mean, ccf_vol, n_sims;
                                      seed=42)

Simulate EAD distribution for revolving credit facilities.
EAD = Drawn + CCF * (Committed - Drawn)
"""
function exposure_at_default_distribution(committed::Float64, drawn::Float64,
                                           ccf_mean::Float64, ccf_vol::Float64,
                                           n_sims::Int; seed::Int=42)
    rng = Random.MersenneTwister(seed)
    undrawn = committed - drawn

    eads = Vector{Float64}(undef, n_sims)
    for i in 1:n_sims
        # CCF from beta distribution
        ccf = ccf_mean + ccf_vol * randn(rng)
        ccf = clamp(ccf, 0.0, 1.0)
        eads[i] = drawn + ccf * undrawn
    end

    return (mean_ead=mean(eads), std_ead=std(eads),
            p75_ead=sort(eads)[ceil(Int, 0.75 * n_sims)],
            p95_ead=sort(eads)[ceil(Int, 0.95 * n_sims)],
            distribution=eads)
end

"""
    regulatory_capital_irb(pd, lgd, ead, maturity, asset_correlation)

Basel IRB formula for regulatory capital.
K = LGD * [N((1-R)^{-0.5} * G(PD) + (R/(1-R))^{0.5} * G(0.999)) - PD] * MA
"""
function regulatory_capital_irb(pd::Float64, lgd::Float64, ead::Float64,
                                 maturity::Float64, asset_correlation::Float64)::Float64
    R = clamp(asset_correlation, 0.001, 0.999)
    pd_adj = clamp(pd, 1e-10, 0.999)

    # Conditional PD at 99.9% confidence
    cond_pd = normal_cdf((normal_inv(pd_adj) + sqrt(R) * normal_inv(0.999)) / sqrt(1.0 - R))

    # Capital requirement before maturity adjustment
    K_base = lgd * (cond_pd - pd_adj)

    # Maturity adjustment
    b = (0.11852 - 0.05478 * log(pd_adj))^2
    MA = (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)

    K = K_base * MA
    return K * ead
end

"""
    expected_shortfall_credit(losses, confidence)

Expected shortfall (CVaR) from loss distribution.
"""
function expected_shortfall_credit(losses::Vector{Float64},
                                    confidence::Float64)::Float64
    sorted = sort(losses)
    n = length(sorted)
    cutoff = ceil(Int, confidence * n)
    if cutoff >= n
        return sorted[end]
    end
    return mean(sorted[cutoff:end])
end

"""
    loss_given_default_regression(features, lgd_observed; l2_reg=0.01)

Tobit regression for LGD modeling (censored at 0 and 1).
Simplified: linear regression with clamping.
"""
function loss_given_default_regression(features::Matrix{Float64},
                                        lgd_observed::Vector{Float64};
                                        l2_reg::Float64=0.01)
    n, p = size(features)
    X = hcat(ones(n), features)
    p_aug = p + 1

    # Ridge regression
    XtX = X' * X + l2_reg * I
    Xty = X' * lgd_observed
    beta = XtX \ Xty

    # Predictions
    predicted = X * beta
    predicted = clamp.(predicted, 0.0, 1.0)

    # R-squared
    ss_res = sum((lgd_observed - predicted).^2)
    ss_tot = sum((lgd_observed .- mean(lgd_observed)).^2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-15)

    return (coefficients=beta, r_squared=r_squared, predicted=predicted)
end

"""
    vintage_analysis(origination_periods, default_periods, observation_period)

Vintage analysis: default rates by origination cohort.
"""
function vintage_analysis(origination_periods::Vector{Int},
                           default_periods::Vector{Int},
                           observation_period::Int)
    max_orig = maximum(origination_periods)
    min_orig = minimum(origination_periods)
    num_vintages = max_orig - min_orig + 1

    vintage_counts = zeros(Int, num_vintages)
    vintage_defaults = zeros(Int, num_vintages, observation_period)

    for i in 1:length(origination_periods)
        v = origination_periods[i] - min_orig + 1
        vintage_counts[v] += 1

        if default_periods[i] > 0
            age_at_default = default_periods[i] - origination_periods[i]
            if 1 <= age_at_default <= observation_period
                vintage_defaults[v, age_at_default] += 1
            end
        end
    end

    # Cumulative default rates
    cum_default_rates = zeros(num_vintages, observation_period)
    for v in 1:num_vintages
        if vintage_counts[v] > 0
            for t in 1:observation_period
                cum_default_rates[v, t] = sum(vintage_defaults[v, 1:t]) / vintage_counts[v]
            end
        end
    end

    return (vintage_counts=vintage_counts, vintage_defaults=vintage_defaults,
            cumulative_rates=cum_default_rates)
end

"""
    concentration_risk_hhi(exposures, sector_assignments, num_sectors)

Herfindahl-Hirschman Index for concentration risk by sector.
"""
function concentration_risk_hhi(exposures::Vector{Float64},
                                 sector_assignments::Vector{Int},
                                 num_sectors::Int)::Float64
    total = sum(exposures)
    sector_totals = zeros(num_sectors)
    for i in 1:length(exposures)
        s = sector_assignments[i]
        if 1 <= s <= num_sectors
            sector_totals[s] += exposures[i]
        end
    end

    hhi = sum((sector_totals[s] / total)^2 for s in 1:num_sectors)
    return hhi
end

"""
    ifrs9_staging(pd_origination, pd_current, threshold_mult)

IFRS 9 staging logic: determine stage based on PD deterioration.
"""
function ifrs9_staging(pd_origination::Float64, pd_current::Float64,
                       threshold_mult::Float64)::Int
    if pd_current > 0.9
        return 3  # Credit-impaired
    elseif pd_current > pd_origination * threshold_mult
        return 2  # Significant increase in credit risk
    else
        return 1  # Performing
    end
end

"""
    ifrs9_ecl(ead, pd_12m, pd_lifetime, lgd, stage, maturity)

IFRS 9 Expected Credit Loss calculation.
Stage 1: 12-month ECL, Stage 2/3: lifetime ECL.
"""
function ifrs9_ecl(ead::Float64, pd_12m::Float64, pd_lifetime::Float64,
                   lgd::Float64, stage::Int, maturity::Float64)::Float64
    if stage == 1
        return ead * pd_12m * lgd
    else
        return ead * pd_lifetime * lgd
    end
end

"""
    credit_risk_capital_summary(exposures, pds, lgds, maturities, correlations)

Comprehensive capital summary.
"""
function credit_risk_capital_summary(exposures::Vector{Float64},
                                      pds::Vector{Float64},
                                      lgds::Vector{Float64},
                                      maturities::Vector{Float64},
                                      correlations::Matrix{Float64})
    n = length(exposures)
    total_ead = sum(exposures)
    el = expected_loss_portfolio(exposures, pds, lgds)
    ul = unexpected_loss_portfolio(exposures, pds, lgds, correlations)
    ec = economic_capital(exposures, pds, lgds, correlations)

    # IRB capital
    irb_capital = sum(regulatory_capital_irb(pds[i], lgds[i], exposures[i],
                       maturities[i], 0.15) for i in 1:n)

    # Risk contributions
    rc = risk_contribution(exposures, pds, lgds, correlations)

    return (total_ead=total_ead, expected_loss=el, unexpected_loss=ul,
            economic_capital=ec, irb_capital=irb_capital,
            risk_contributions=rc,
            el_pct=el / total_ead * 100,
            ec_pct=ec / total_ead * 100)
end

end # module CreditRiskModels
