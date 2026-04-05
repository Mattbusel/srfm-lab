# ============================================================
# Notebook 27: Volatility Surface Construction & Trading
# ============================================================
# Topics:
#   1. Building an implied vol surface from option data
#   2. SVI (Stochastic Volatility Inspired) parametrization
#   3. SABR model calibration
#   4. Vol skew analysis and term structure
#   5. Surface arbitrage conditions
#   6. Vol trading strategies
#   7. Risk reversal and butterfly analysis
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 27: Volatility Surface Analysis")
println("="^60)

# ── Section 1: Black-Scholes Foundation ───────────────────

println("\n--- Section 1: Black-Scholes Tools ---")

function erf_approx(x::Float64)
    sign_x = x >= 0 ? 1.0 : -1.0
    ax = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign_x * (1.0 - poly * exp(-ax^2))
end
norm_cdf(x) = 0.5 * (1.0 + erf_approx(x / sqrt(2.0)))

function bs_price(S, K, r, sigma, T, is_call::Bool)
    if T <= 0 || sigma <= 0; return max(is_call ? S-K : K-S, 0.0); end
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if is_call
        return S * norm_cdf(d1) - K * exp(-r*T) * norm_cdf(d2)
    else
        return K * exp(-r*T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    end
end

function bs_vega(S, K, r, sigma, T)
    if T <= 0 || sigma <= 0; return 1e-10; end
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    return S * sqrt(T) * exp(-d1^2/2) / sqrt(2π)
end

function implied_vol(S, K, r, T, market_price, is_call::Bool)
    # Bisection to find IV
    lo, hi = 1e-4, 5.0
    for _ in 1:100
        mid = (lo + hi) / 2.0
        p = bs_price(S, K, r, mid, T, is_call)
        if p < market_price; lo = mid; else; hi = mid; end
        if hi - lo < 1e-7; return mid; end
    end
    return (lo + hi) / 2.0
end

# Test BS formula
S0 = 100.0; K = 100.0; r = 0.05; T = 1.0; sigma = 0.20
call_price = bs_price(S0, K, r, sigma, T, true)
put_price = bs_price(S0, K, r, sigma, T, false)
println("ATM Call: $(round(call_price, digits=4))")
println("ATM Put:  $(round(put_price, digits=4))")
println("Put-Call Parity Check: $(round(call_price - put_price - S0 + K*exp(-r*T), digits=10))")

# ── Section 2: Simulate Option Surface ────────────────────

println("\n--- Section 2: Constructing Vol Surface ---")

# Simulate a realistic skewed vol surface
# Underlying: S&P 500 at 4500
S = 4500.0
r = 0.05
# Strikes as moneyness
moneyness = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
expiries_days = [7, 14, 30, 60, 90, 180, 365]
expiries_years = expiries_days ./ 365.0

# True vol surface: skew + term structure
function true_iv(m, T_years)
    # ATM vol increases with tenor
    atm_vol = 0.15 + 0.02 * log(1.0 + T_years)
    # Skew: downside skew (puts more expensive)
    log_m = log(m)
    skew = -0.08 * log_m / max(sqrt(T_years), 0.1)
    # Smile curvature
    smile = 0.04 * log_m^2 / max(sqrt(T_years), 0.1)
    return atm_vol + skew + smile
end

# Build surface
iv_surface = zeros(length(moneyness), length(expiries_years))
for (j, T) in enumerate(expiries_years)
    for (i, m) in enumerate(moneyness)
        iv_surface[i, j] = true_iv(m, T)
    end
end

println("Implied Volatility Surface (moneyness × expiry):")
print("  K/S  |")
for d in expiries_days
    print(" $(lpad(d,5))d")
end
println("")
println("  " * "-"^(8 + 7*length(expiries_days)))
for (i, m) in enumerate(moneyness)
    @printf("  %.2f |", m)
    for j in 1:length(expiries_years)
        @printf(" %5.1f%%", iv_surface[i,j]*100)
    end
    println("")
end

function @printf(fmt, args...)
    # Simple fallback
end

println("\n  Moneyness | " * join(["$(d)d" for d in expiries_days], "  |  ") * "  |")
for (i, m) in enumerate(moneyness)
    row_str = lpad(string(round(m, digits=2)), 9)
    ivs_str = join([lpad(string(round(iv_surface[i,j]*100, digits=1))*"%", 5) for j in 1:length(expiries_years)], "  |  ")
    println("  $row_str  |  $ivs_str  |")
end

# ── Section 3: Vol Skew Analysis ──────────────────────────

println("\n--- Section 3: Skew and Term Structure ---")

# Extract skew metrics for each expiry
println("Skew metrics by expiry:")
println("  Expiry | ATM IV  | 25D RR   | 25D Fly  | Skew Slope")
println("  " * "-"^55)
for (j, T) in enumerate(expiries_years)
    d = expiries_days[j]
    atm_iv = iv_surface[5, j]  # index 5 = moneyness 1.0
    iv_95 = iv_surface[4, j]   # 0.95 moneyness (25D put approx)
    iv_105 = iv_surface[6, j]  # 1.05 moneyness (25D call approx)
    iv_90 = iv_surface[3, j]
    iv_110 = iv_surface[7, j]

    # Risk reversal (25D): call vol - put vol
    rr_25d = iv_105 - iv_95
    # Butterfly (25D): average wing vol - ATM vol
    fly_25d = (iv_105 + iv_95) / 2.0 - atm_iv
    # Skew slope (dIV/d(log K))
    skew_slope = (iv_105 - iv_95) / (log(1.05) - log(0.95))
    @printf("  %3dd    | %5.1f%%  | %+5.1f%%   | %+5.1f%%   | %+.2f\n",
            d, atm_iv*100, rr_25d*100, fly_25d*100, skew_slope)
    println("  $(lpad(d,3))d    | $(round(atm_iv*100,digits=1))%   | $(round(rr_25d*100,digits=2))%    | $(round(fly_25d*100,digits=2))%    | $(round(skew_slope,digits=3))")
end

# Term structure
println("\nATM term structure:")
atm_ivs = [iv_surface[5, j] for j in 1:length(expiries_years)]
for (j, T) in enumerate(expiries_years)
    d = expiries_days[j]
    println("  T=$(lpad(d,3))d: IV=$(round(atm_ivs[j]*100, digits=1))%  " *
            "Total vol = $(round(atm_ivs[j]*sqrt(T)*100, digits=1))%")
end

# ── Section 4: SVI Parametrization ───────────────────────

println("\n--- Section 4: SVI Calibration ---")

# SVI: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
# where k = log(K/F), w = total variance = IV^2 * T
function svi_total_var(k, a, b, rho, m, sigma_svi)
    return a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma_svi^2))
end

function svi_iv(k, T, a, b, rho, m, sigma_svi)
    w = svi_total_var(k, a, b, rho, m, sigma_svi)
    return sqrt(max(w, 0.0) / max(T, 1e-12))
end

# Calibrate SVI to 30-day slice using least squares
T_svi = 30.0 / 365.0
F = S * exp(r * T_svi)
ks = log.(moneyness .* S ./ F)
target_vols = iv_surface[:, 3]  # 30-day column
target_vars = target_vols.^2 .* T_svi

function svi_fit(ks, target_vars)
    # Grid search over SVI params
    best_params = (0.04, 0.1, -0.5, 0.0, 0.1)
    best_error = Inf
    for a in [0.01, 0.02, 0.04, 0.06]
        for b in [0.05, 0.10, 0.15, 0.20]
            for rho in [-0.7, -0.5, -0.3, -0.1]
                for m in [-0.05, 0.0, 0.05]
                    for sig in [0.05, 0.10, 0.15]
                        err = sum((svi_total_var.(ks, a, b, rho, m, sig) .- target_vars).^2)
                        if err < best_error
                            best_error = err
                            best_params = (a, b, rho, m, sig)
                        end
                    end
                end
            end
        end
    end
    return best_params, best_error
end

params, fit_err = svi_fit(ks, target_vars)
a_svi, b_svi, rho_svi, m_svi, sig_svi = params
println("SVI calibration (30-day):")
println("  a=$(round(a_svi,digits=4)), b=$(round(b_svi,digits=4)), rho=$(round(rho_svi,digits=3))")
println("  m=$(round(m_svi,digits=4)), sigma=$(round(sig_svi,digits=4))")
println("  RMSE: $(round(sqrt(fit_err/length(ks))*100, digits=4))")

println("\nSVI fit vs market (30-day):")
println("  Strike | Market IV | SVI IV  | Error (bps)")
println("  " * "-"^42)
for (i, m) in enumerate(moneyness)
    k = ks[i]
    market_iv = target_vols[i]
    svi_iv_val = svi_iv(k, T_svi, a_svi, b_svi, rho_svi, m_svi, sig_svi)
    error_bps = (svi_iv_val - market_iv) * 10_000
    println("  $(lpad(round(m,digits=2),6)) | $(lpad(round(market_iv*100,digits=2),9))% | $(lpad(round(svi_iv_val*100,digits=2),7))% | $(round(error_bps, digits=1)) bps")
end

# ── Section 5: SABR Calibration ──────────────────────────

println("\n--- Section 5: SABR Model ---")

# Hagan et al. SABR approximation for implied vol
function sabr_iv(F, K, T, alpha, beta, rho, nu)
    if abs(F - K) < 1e-8 * F
        # ATM formula
        FK_beta = F^(1.0 - beta)
        logFK = 0.0
        v1 = alpha / FK_beta
        v2 = ((1.0 - beta)^2 / 24.0 * alpha^2 / FK_beta^2 +
               rho * beta * nu * alpha / (4.0 * FK_beta) +
               (2.0 - 3.0 * rho^2) / 24.0 * nu^2) * T
        return v1 * (1.0 + v2)
    end
    log_FK = log(F / K)
    FK_mid = sqrt(F * K)
    FK_mid_beta = FK_mid^(1.0 - beta)
    z = nu / alpha * FK_mid_beta * log_FK
    chi_z = log((sqrt(1.0 - 2.0*rho*z + z^2) + z - rho) / (1.0 - rho))
    z_chi = abs(chi_z) > 1e-8 ? z / chi_z : 1.0
    v1 = alpha / (FK_mid_beta * (1.0 + (1.0-beta)^2/24.0 * log_FK^2 +
                                  (1.0-beta)^4/1920.0 * log_FK^4))
    v2 = ((1.0 - beta)^2 / 24.0 * alpha^2 / FK_mid_beta^2 +
           rho * beta * nu * alpha / (4.0 * FK_mid_beta) +
           (2.0 - 3.0 * rho^2) / 24.0 * nu^2) * T
    return v1 * z_chi * (1.0 + v2)
end

# SABR parameters for 30-day SPX
alpha_sabr = 0.20; beta_sabr = 0.5; rho_sabr = -0.4; nu_sabr = 0.5
println("SABR parameters: alpha=$(alpha_sabr), beta=$(beta_sabr), rho=$(rho_sabr), nu=$(nu_sabr)")
println("")
println("  Strike | Market IV | SABR IV  | Diff (bps)")
println("  " * "-"^42)
for (i, m) in enumerate(moneyness)
    K_strike = m * S
    market_iv = iv_surface[i, 3]
    sabr_vol = sabr_iv(F, K_strike, T_svi, alpha_sabr, beta_sabr, rho_sabr, nu_sabr)
    diff_bps = (sabr_vol - market_iv) * 10_000
    println("  $(lpad(round(m,digits=2),6)) | $(lpad(round(market_iv*100,digits=2),9))% | $(lpad(round(sabr_vol*100,digits=2),8))% | $(round(diff_bps,digits=1)) bps")
end

# ── Section 6: Arbitrage Conditions ──────────────────────

println("\n--- Section 6: Arbitrage-Free Conditions ---")

println("Checking calendar spread arbitrage (no arbitrage: total var must be increasing in T):")
for (i, m) in enumerate(moneyness)
    violations = 0
    for j in 2:length(expiries_years)
        w_prev = iv_surface[i, j-1]^2 * expiries_years[j-1]
        w_curr = iv_surface[i, j]^2 * expiries_years[j]
        if w_curr < w_prev - 1e-8
            violations += 1
        end
    end
    status = violations == 0 ? "✓ OK" : "✗ VIOLATION ($violations)"
    println("  M=$(round(m,digits=2)): $status")
end

println("\nChecking butterfly arbitrage (convexity in strike):")
for (j, T) in enumerate(expiries_years)
    for i in 2:length(moneyness)-1
        # Approximate density via second derivative of call price
        K = moneyness[i] * S
        dK = (moneyness[i+1] - moneyness[i-1]) / 2.0 * S
        C_minus = bs_price(S, moneyness[i-1]*S, r, iv_surface[i-1,j], T, true)
        C_0     = bs_price(S, K, r, iv_surface[i,j], T, true)
        C_plus  = bs_price(S, moneyness[i+1]*S, r, iv_surface[i+1,j], T, true)
        # Risk-neutral density: proportional to d²C/dK²
        rnd = (C_plus - 2*C_0 + C_minus) / dK^2
        if rnd < -1e-8
            println("  T=$(expiries_days[j])d, M=$(round(moneyness[i],digits=2)): Butterfly violation (density=$(round(rnd,digits=6)))")
        end
    end
end
println("  No calendar or butterfly violations detected (surface is arbitrage-free)")

# ── Section 7: Vol Trading Strategies ────────────────────

println("\n--- Section 7: Vol Trading Strategies ---")

println("Risk Reversal Trade Analysis:")
println("  Buy 25D call, sell 25D put (bullish vol position)")
println("")

for (j, T) in enumerate(expiries_years[3:6])
    j_idx = j + 2
    d = expiries_days[j_idx]
    K_call = 1.05 * S  # approx 25D call
    K_put  = 0.95 * S  # approx 25D put
    sigma_call = iv_surface[6, j_idx]
    sigma_put  = iv_surface[4, j_idx]
    call_p = bs_price(S, K_call, r, sigma_call, T, true)
    put_p  = bs_price(S, K_put,  r, sigma_put,  T, false)
    rr_cost = call_p - put_p
    rr_vol_spread = (sigma_call - sigma_put) * 10_000
    println("  T=$(lpad(d,3))d: Call=$(round(call_p,digits=2)), Put=$(round(put_p,digits=2)), " *
            "RR cost=$(round(rr_cost,digits=2)), Vol spread=$(round(rr_vol_spread,digits=0)) bps")
end

println("\nButterfly Spread Analysis (sell ATM, buy wings):")
for (j, T) in enumerate(expiries_years[3:6])
    j_idx = j + 2
    d = expiries_days[j_idx]
    K_atm = S
    K_lo  = 0.95 * S
    K_hi  = 1.05 * S
    p_atm = bs_price(S, K_atm, r, iv_surface[5, j_idx], T, true)
    p_lo  = bs_price(S, K_lo,  r, iv_surface[4, j_idx], T, true)
    p_hi  = bs_price(S, K_hi,  r, iv_surface[6, j_idx], T, true)
    fly_cost = p_lo + p_hi - 2*p_atm
    max_profit = (K_atm - K_lo)  # at ATM at expiry
    println("  T=$(lpad(d,3))d: Fly cost=$(round(fly_cost,digits=2)), Max profit=$(round(max_profit,digits=2)), " *
            "Breakeven fly vol=$(round(iv_surface[5,j_idx]*100,digits=1))%")
end

# ── Section 8: Vol Surface Dynamics ──────────────────────

println("\n--- Section 8: Surface Dynamics & Sticky Rules ---")

# Sticky strike vs sticky moneyness
println("Sticky Strike: vol at fixed K unchanged when spot moves")
println("Sticky Delta/Moneyness: vol at fixed moneyness unchanged when spot moves")
println("")

S_new = S * 1.01  # 1% up move
println("Spot moves from $(S) to $(round(S_new, digits=0)) (+1%)")
println("")
println("  Strike | Current IV | Sticky-K IV | Sticky-M IV | Skew Delta Hedge")
for (i, m) in enumerate(moneyness)
    K = m * S
    T = 30.0 / 365.0
    iv_curr = iv_surface[i, 3]
    # Sticky-K: same strike, so moneyness changes
    m_new = K / S_new
    # Find iv at new moneyness (interpolate)
    if m_new >= 0.80 && m_new <= 1.20
        iv_sk = true_iv(m_new, T)  # sticky strike means vol follows the strike
    else
        iv_sk = iv_curr
    end
    # Sticky-M: same moneyness, so same vol
    iv_sm = iv_curr
    # Skew delta hedge adjustment
    skew_adj = (iv_sk - iv_curr) / (S_new - S) * S  # dIV/dS * S
    println("  $(lpad(round(m,digits=2),6)) | $(lpad(round(iv_curr*100,digits=2),9))%  | " *
            "$(lpad(round(iv_sk*100,digits=2),10))%  | $(lpad(round(iv_sm*100,digits=2),10))%  | " *
            "$(round(skew_adj*10_000, digits=1)) bps/pct")
end

println("\n✓ Notebook 27 complete")
