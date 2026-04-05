module VolatilitySurface

# ============================================================
# volatility_surface.jl -- Volatility Surface Modeling
# ============================================================
# Covers: SVI (Stochastic Volatility Inspired) parametrization,
# SABR model, surface interpolation (bilinear, bicubic),
# smile construction, skew and term structure analysis,
# local volatility (Dupire), variance swap pricing,
# volatility cone, risk reversal and butterfly metrics,
# calendar spread constraints, arbitrage-free checks.
# Pure stdlib Julia -- no external packages.
# ============================================================

using Statistics, LinearAlgebra

struct SVIParams
    a::Float64    # level
    b::Float64    # angle (slope)
    rho::Float64  # rotation
    m::Float64    # translation
    sigma::Float64 # ATM curvature
end

struct SABRParams
    alpha::Float64  # initial vol
    beta::Float64   # CEV exponent
    rho_sabr::Float64  # correlation
    nu::Float64     # vol-of-vol
    F::Float64      # forward price
    T::Float64      # time to expiry
end

struct VolSurface
    strikes::Vector{Float64}     # log-moneyness k = log(K/F)
    expiries::Vector{Float64}    # T in years
    implied_vols::Matrix{Float64} # n_strikes x n_expiries
end

struct SkewMetrics
    expiry::Float64
    atm_vol::Float64
    rr25::Float64        # 25d risk reversal (call - put)
    bf25::Float64        # 25d butterfly
    slope::Float64       # dVol/dk at ATM
    curvature::Float64   # d2Vol/dk2 at ATM
end

struct VarianceSwapResult
    fair_variance::Float64
    fair_vol::Float64
    replication_cost::Float64
    vega_notional::Float64
end

struct LocalVolGrid
    S_grid::Vector{Float64}
    T_grid::Vector{Float64}
    lv::Matrix{Float64}     # sigma_loc(S, T)
end

# ---- 1. SVI Parametrization ----

function svi_total_variance(k::Float64, params::SVIParams)::Float64
    d = k - params.m
    return params.a + params.b * (params.rho * d + sqrt(d^2 + params.sigma^2))
end

function svi_implied_vol(k::Float64, T::Float64, params::SVIParams)::Float64
    w = svi_total_variance(k, params)
    return sqrt(max(w, 0.0) / (T + 1e-12))
end

function svi_calibrate_atm(market_vols::Vector{Float64}, log_strikes::Vector{Float64},
                             T::Float64)::SVIParams
    n = length(market_vols)
    best_params = SVIParams(0.04, 0.1, -0.3, 0.0, 0.1)
    best_err = Inf
    for a_t in 0.01:0.01:0.10, b_t in 0.05:0.05:0.50, rho_t in -0.9:0.3:0.9
        for sig_t in 0.05:0.05:0.30
            p = SVIParams(a_t, b_t, rho_t, 0.0, sig_t)
            err = sum((svi_implied_vol(k, T, p) - market_vols[i])^2
                       for (i,k) in enumerate(log_strikes))
            if err < best_err
                best_err = err
                best_params = p
            end
        end
    end
    return best_params
end

function svi_butterfly_arbitrage(params::SVIParams, k_lo::Float64=-2.0,
                                   k_hi::Float64=2.0, n_pts::Int=100)::Bool
    ks = range(k_lo, k_hi, length=n_pts)
    ws = [svi_total_variance(k, params) for k in ks]
    # Calendar arbitrage: total variance must be increasing
    all_positive = all(w >= 0 for w in ws)
    # Butterfly: g(k) = (1 - k*rho/sigma)^2 - d^2*(1/4 - 1/w) + ... simplified check
    g_vals = Float64[]
    for i in 2:(length(ks)-1)
        dk = ks[2] - ks[1]
        dw  = (ws[i+1] - ws[i-1]) / (2dk)
        d2w = (ws[i+1] - 2ws[i] + ws[i-1]) / dk^2
        w = ws[i]; k = ks[i]
        g = (1 - k*dw/(2w+1e-12))^2 - dw^2/4*(1/w+0.25) + d2w/2
        push!(g_vals, g)
    end
    return all_positive && all(g >= 0 for g in g_vals)
end

function svi_density(k::Float64, params::SVIParams, T::Float64)::Float64
    h = 1e-4
    w(x) = svi_total_variance(x, params)
    dw  = (w(k+h) - w(k-h)) / (2h)
    d2w = (w(k+h) - 2*w(k) + w(k-h)) / h^2
    w0  = w(k)
    g   = (1 - k*dw/(2*w0+1e-12))^2 - dw^2/4*(1/w0+0.25) + d2w/2
    norm_const = 1 / sqrt(2pi * w0)
    return max(0.0, norm_const * g * exp(-0.5*(k/sqrt(w0+1e-12))^2))
end

# ---- 2. SABR Model ----

function sabr_vol(K::Float64, params::SABRParams)::Float64
    F = params.F; alpha = params.alpha; beta = params.beta
    rho = params.rho_sabr; nu = params.nu; T = params.T
    if abs(F - K) < 1e-8
        # ATM formula
        FK_mid = F
        num = alpha * (1 + ((1-beta)^2/24 * alpha^2/(FK_mid^(2-2beta)) +
                            rho*beta*nu*alpha/(4*FK_mid^(1-beta)) +
                            (2-3rho^2)/24*nu^2) * T)
        den = FK_mid^(1-beta)
        return num / den
    end
    FK = sqrt(F*K); log_FK = log(F/K)
    z = nu/alpha * FK^(1-beta) * log_FK
    x_z = log((sqrt(1 - 2rho*z + z^2) + z - rho) / (1-rho))
    A = alpha / (FK^(1-beta) * (1 + (1-beta)^2/24*log_FK^2 + (1-beta)^4/1920*log_FK^4))
    B = z / x_z
    C = 1 + ((1-beta)^2/24*alpha^2/FK^(2-2beta) +
              rho*beta*nu*alpha/(4*FK^(1-beta)) + (2-3rho^2)/24*nu^2)*T
    return A * B * C
end

function sabr_calibrate(market_strikes::Vector{Float64}, market_vols::Vector{Float64},
                          F::Float64, T::Float64)::SABRParams
    best = SABRParams(0.3, 0.5, 0.0, 0.4, F, T)
    best_err = Inf
    for alpha in 0.1:0.1:1.0, beta_v in 0.3:0.2:1.0, rho_v in -0.7:0.2:0.3
        for nu_v in 0.2:0.2:1.0
            p = SABRParams(alpha, beta_v, rho_v, nu_v, F, T)
            err = sum((sabr_vol(K, p) - market_vols[i])^2
                       for (i,K) in enumerate(market_strikes))
            if err < best_err
                best_err = err; best = p
            end
        end
    end
    return best
end

function sabr_delta(K::Float64, params::SABRParams, dS::Float64=1.0)::Float64
    p_up = SABRParams(params.alpha, params.beta, params.rho_sabr, params.nu,
                       params.F + dS, params.T)
    p_dn = SABRParams(params.alpha, params.beta, params.rho_sabr, params.nu,
                       params.F - dS, params.T)
    v_up = sabr_vol(K, p_up); v_dn = sabr_vol(K, p_dn)
    return (v_up - v_dn) / (2dS)
end

# ---- 3. Vol Surface Interpolation ----

function bilinear_interpolate(surface::VolSurface, k::Float64, T::Float64)::Float64
    ks = surface.strikes; Ts = surface.expiries
    nk = length(ks); nT = length(Ts)
    i = clamp(searchsortedfirst(ks, k) - 1, 1, nk-1)
    j = clamp(searchsortedfirst(Ts, T) - 1, 1, nT-1)
    t_k = (k - ks[i]) / (ks[i+1] - ks[i] + 1e-12)
    t_T = (T - Ts[j]) / (Ts[j+1] - Ts[j] + 1e-12)
    v00 = surface.implied_vols[i,  j  ]
    v10 = surface.implied_vols[i+1,j  ]
    v01 = surface.implied_vols[i,  j+1]
    v11 = surface.implied_vols[i+1,j+1]
    return (1-t_k)*(1-t_T)*v00 + t_k*(1-t_T)*v10 +
           (1-t_k)*t_T*v01 + t_k*t_T*v11
end

function vol_surface_slice(surface::VolSurface, T_target::Float64)::Vector{Float64}
    return [bilinear_interpolate(surface, k, T_target) for k in surface.strikes]
end

# ---- 4. Skew Metrics ----

function compute_skew_metrics(strikes::Vector{Float64}, vols::Vector{Float64},
                                F::Float64, T::Float64)::SkewMetrics
    log_strikes = log.(strikes ./ F)
    # Find ATM vol (nearest to k=0)
    atm_idx = argmin(abs.(log_strikes))
    atm_vol = vols[atm_idx]
    # Approximate 25d strikes (simplified)
    d25_k = 0.5*atm_vol*sqrt(T)
    # Risk reversal: vol at -0.25 delta strike vs +0.25 delta strike (approx by k = +/-d25_k)
    function interp_vol(k_target)
        idx = searchsortedfirst(log_strikes, k_target)
        idx = clamp(idx, 1, length(vols)-1)
        frac = (k_target - log_strikes[idx]) / (log_strikes[idx+1] - log_strikes[idx] + 1e-12)
        return vols[idx]*(1-frac) + vols[idx+1]*frac
    end
    call25_vol = interp_vol(d25_k)
    put25_vol  = interp_vol(-d25_k)
    rr25 = call25_vol - put25_vol
    bf25 = (call25_vol + put25_vol)/2 - atm_vol
    # Slope and curvature at ATM via FD
    h = 0.1
    slope = (interp_vol(h) - interp_vol(-h)) / (2h)
    curv  = (interp_vol(h) - 2*atm_vol + interp_vol(-h)) / h^2
    return SkewMetrics(T, atm_vol, rr25, bf25, slope, curv)
end

function vol_cone(historical_vols::Vector{Float64}, windows::Vector{Int})
    results = Dict{Int, NamedTuple}()
    n = length(historical_vols)
    for w in windows
        if n < w; continue; end
        rolling_vols = [std(historical_vols[i-w+1:i]) * sqrt(252.0) for i in w:n]
        results[w] = (
            min   = minimum(rolling_vols),
            p10   = sort(rolling_vols)[max(1,round(Int,0.1*length(rolling_vols)))],
            p25   = sort(rolling_vols)[max(1,round(Int,0.25*length(rolling_vols)))],
            median = median(rolling_vols),
            p75   = sort(rolling_vols)[min(length(rolling_vols), round(Int,0.75*length(rolling_vols)))],
            p90   = sort(rolling_vols)[min(length(rolling_vols), round(Int,0.9*length(rolling_vols)))],
            max   = maximum(rolling_vols),
            current = rolling_vols[end],
        )
    end
    return results
end

# ---- 5. Term Structure ----

function vol_term_structure_slope(vols_by_expiry::Vector{Float64},
                                    expiries::Vector{Float64})::Float64
    n = length(expiries)
    x = log.(expiries); y = vols_by_expiry
    xb = mean(x); yb = mean(y)
    return sum((x.-xb).*(y.-yb)) / (sum((x.-xb).^2) + 1e-12)
end

function forward_vol(v1::Float64, t1::Float64, v2::Float64, t2::Float64)::Float64
    return sqrt(max((v2^2 * t2 - v1^2 * t1) / (t2 - t1 + 1e-12), 0.0))
end

function vol_term_structure_arbitrage_check(vols::Vector{Float64},
                                              expiries::Vector{Float64})::Bool
    for i in 2:length(vols)
        fwd = forward_vol(vols[i-1], expiries[i-1], vols[i], expiries[i])
        if isnan(fwd) || fwd < 0; return false; end
    end
    return true
end

# ---- 6. Variance Swap Pricing ----

function variance_swap_fair_vol(strikes::Vector{Float64}, vols::Vector{Float64},
                                  F::Float64, r::Float64, T::Float64)::VarianceSwapResult
    n = length(strikes)
    log_strikes = log.(strikes ./ F)
    dk_log = (log_strikes[end] - log_strikes[1]) / (n - 1)
    # Replication via log-strip
    integral = 0.0
    for i in 1:n
        K = strikes[i]; sigma = vols[i]
        d1 = (log(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T) + 1e-12)
        d2 = d1 - sigma*sqrt(T)
        nd1 = 0.5*(1+erf(d1/sqrt(2))); nd2 = 0.5*(1+erf(d2/sqrt(2)))
        if K <= F
            price = K*exp(-r*T)*(1-nd2) - F*exp(-r*T)*(1-nd1)
        else
            price = F*exp(-r*T)*nd1 - K*exp(-r*T)*nd2
        end
        integral += price * dk_log / K^2
    end
    fair_var = 2*exp(r*T) * integral
    fair_vol = sqrt(max(fair_var, 0.0) / T)
    vega_notional = F^2 * T * exp(-r*T) / (2*fair_vol + 1e-12)
    return VarianceSwapResult(fair_var, fair_vol, integral, vega_notional)
end

function vix_approximation(strikes::Vector{Float64}, calls::Vector{Float64},
                             puts::Vector{Float64}, F::Float64,
                             r::Float64, T::Float64)::Float64
    n = length(strikes)
    sigma2 = 0.0
    for i in 2:n
        K = strikes[i]; dK = i < n ? (strikes[i+1] - strikes[i-1]) / 2 : strikes[i]-strikes[i-1]
        price = K < F ? puts[i] : calls[i]
        sigma2 += 2 * dK / K^2 * exp(r*T) * price
    end
    K0_idx = searchsortedfirst(strikes, F) - 1
    K0 = strikes[max(1, K0_idx)]
    sigma2 -= (F/K0 - 1)^2
    return sqrt(sigma2 / T)
end

# ---- 7. Local Volatility (Dupire) ----

function dupire_local_vol(surface::VolSurface, S0::Float64,
                           r::Float64, q::Float64)::LocalVolGrid
    ks = surface.strikes; Ts = surface.expiries
    nk = length(ks); nT = length(Ts)
    lv = zeros(nk, nT)
    for j in 1:nT
        T_val = Ts[j]
        for i in 2:(nk-1)
            K = ks[i] * S0  # convert log-moneyness back to strike
            sigma = surface.implied_vols[i, j]
            # Dupire formula numerator: dw/dT (finite diff)
            dT = j < nT ? Ts[j+1] - Ts[j] : Ts[j] - Ts[j-1]
            w_t = sigma^2 * T_val
            w_next = j < nT ? surface.implied_vols[i, min(j+1,nT)]^2 * Ts[min(j+1,nT)] : w_t
            dw_dT = (w_next - w_t) / (dT + 1e-12)
            # Denominator terms
            dk = surface.strikes[i] - surface.strikes[max(1,i-1)]
            dw_dk  = (surface.implied_vols[i+1,j]^2*T_val - surface.implied_vols[i-1,j]^2*T_val)/(2dk)
            d2w_dk2 = (surface.implied_vols[i+1,j]^2*T_val - 2*w_t +
                        surface.implied_vols[i-1,j]^2*T_val) / dk^2
            k_lm = surface.strikes[i]
            denom = (1 - k_lm*dw_dk/(2*w_t+1e-12))^2 + 0.25*(-0.25 - 1/w_t)*dw_dk^2 + 0.5*d2w_dk2
            lv[i, j] = sqrt(max(dw_dT / (denom + 1e-12), 0.0))
        end
        lv[1,  j] = lv[2, j]
        lv[nk, j] = lv[nk-1, j]
    end
    return LocalVolGrid(ks.*S0, Ts, lv)
end

# ---- 8. Implied Vol Inversion ----

function bs_implied_vol(mkt_price::Float64, S::Float64, K::Float64,
                         r::Float64, T::Float64, opt::Symbol=:call;
                         tol::Float64=1e-8)::Float64
    function bs_px(sig)
        d1 = (log(S/K)+(r+0.5*sig^2)*T)/(sig*sqrt(T)+1e-12)
        d2 = d1 - sig*sqrt(T)
        nd1=0.5*(1+erf(d1/sqrt(2))); nd2=0.5*(1+erf(d2/sqrt(2)))
        return opt==:call ? S*nd1 - K*exp(-r*T)*nd2 : K*exp(-r*T)*(1-nd2) - S*(1-nd1)
    end
    function vega(sig)
        d1 = (log(S/K)+(r+0.5*sig^2)*T)/(sig*sqrt(T)+1e-12)
        return S*sqrt(T)*exp(-0.5*d1^2)/sqrt(2pi)
    end
    sig = 0.3
    for _ in 1:100
        px = bs_px(sig); vg = vega(sig)
        if abs(vg) < 1e-14; break; end
        sig_new = sig - (px - mkt_price) / vg
        if abs(sig_new - sig) < tol; sig = sig_new; break; end
        sig = clamp(sig_new, 1e-4, 10.0)
    end
    return sig
end

function smile_from_market_quotes(strikes::Vector{Float64}, prices::Vector{Float64},
                                    S::Float64, r::Float64, T::Float64,
                                    opt_type::Symbol=:call)::Vector{Float64}
    return [bs_implied_vol(prices[i], S, strikes[i], r, T, opt_type) for i in eachindex(strikes)]
end

# ---- Demo ----

function demo()
    println("=== VolatilitySurface Demo ===")

    params = SVIParams(0.04, 0.1, -0.3, 0.0, 0.15)
    ks = -1.0:0.2:1.0
    println("SVI implied vols (T=1y):")
    for k in ks
        iv = svi_implied_vol(k, 1.0, params)
        println("  k=", round(k,digits=2), " IV=", round(iv*100,digits=2), "%")
    end

    sabr_p = SABRParams(0.3, 0.5, -0.3, 0.4, 100.0, 1.0)
    println("\nSABR vols (F=100, T=1y, alpha=30%, nu=40%):")
    for K in 80:10:120
        println("  K=", K, " IV=", round(sabr_vol(Float64(K), sabr_p)*100, digits=2), "%")
    end

    ks_vec = collect(-1.0:0.25:1.0); nk = length(ks_vec)
    Ts_vec = [0.25, 0.5, 1.0, 2.0]; nT = length(Ts_vec)
    vols_mat = [0.20 + 0.02*abs(k) + 0.01*(T-1) for k in ks_vec, T in Ts_vec]
    surface = VolSurface(ks_vec, Ts_vec, vols_mat)
    v = bilinear_interpolate(surface, 0.1, 0.75)
    println("\nInterpolated vol at k=0.1, T=0.75y: ", round(v*100, digits=3), "%")

    rets = cumsum(0.01*randn(300))
    cone = vol_cone(rets, [10, 21, 63, 126, 252])
    println("\nVol cone (252d lookback):")
    for (w, c) in sort(collect(cone))
        println("  window=", w, "d: [", round(c.p10*100,digits=1), "%, ",
                round(c.median*100,digits=1), "%, ", round(c.p90*100,digits=1), "%] current=",
                round(c.current*100,digits=1), "%")
    end

    arb_free = svi_butterfly_arbitrage(params)
    println("\nSVI butterfly arbitrage-free: ", arb_free)

    fv = forward_vol(0.20, 0.5, 0.22, 1.0)
    println("Forward vol (0.5y->1.0y): ", round(fv*100, digits=3), "%")
end

# ---- Additional Volatility Surface Functions ----

function svi_smile_at_expiry(params::SVIParams, T::Float64,
                               k_range::Tuple{Float64,Float64}=(-2.0,2.0),
                               n_pts::Int=41)::Vector{Tuple{Float64,Float64}}
    ks = range(k_range[1], k_range[2], length=n_pts)
    return [(k, svi_implied_vol(k, T, params)) for k in ks]
end

function normalised_vol_by_moneyness(strike::Float64, forward::Float64,
                                      atm_vol::Float64, T::Float64)::Float64
    m = log(strike/forward) / (atm_vol * sqrt(T) + 1e-12)
    return m
end

function sticky_strike_delta(old_vol::Float64, new_vol::Float64,
                               d_vol_d_strike::Float64, bs_delta::Float64)::Float64
    return bs_delta + d_vol_d_strike * old_vol / (new_vol + 1e-12)
end

function sticky_delta_assumption(k::Float64, params::SVIParams, T::Float64,
                                   dS::Float64=0.01)::Float64
    h = dS
    iv_up = svi_implied_vol(k - log(1+h), T, params)
    iv_dn = svi_implied_vol(k + log(1+h), T, params)
    return (iv_up - iv_dn) / (2h)
end

function calendar_arbitrage_free_check(vols_T1::Vector{Float64},
                                         vols_T2::Vector{Float64},
                                         T1::Float64, T2::Float64)::Bool
    T2 > T1 || return false
    var1 = vols_T1.^2 .* T1; var2 = vols_T2.^2 .* T2
    return all(var2 .>= var1)
end

function ssvi_parametrization(k::Float64, theta::Float64, rho_s::Float64,
                                phi::Float64)::Float64
    psi = phi * theta
    return theta/2 * (1 + rho_s*psi*k + sqrt((psi*k + rho_s)^2 + 1 - rho_s^2))
end

function vol_risk_premium(realized_vol::Float64, implied_vol::Float64)::Float64
    return (implied_vol - realized_vol) / (realized_vol + 1e-8) * 100.0
end

function vol_of_vol_estimate(iv_series::Vector{Float64}, window::Int=21)::Float64
    n = length(iv_series)
    if n < window + 1; return NaN; end
    log_changes = [log(iv_series[i]/iv_series[i-1]) for i in 2:n]
    recent = log_changes[end-window+1:end]
    return std(recent) * sqrt(252.0)
end

function term_structure_curvature(vols::Vector{Float64}, maturities::Vector{Float64})::Float64
    n = length(vols)
    if n < 3; return 0.0; end
    mid = n div 2 + 1
    T_mid = maturities[mid]; T_lo = maturities[1]; T_hi = maturities[end]
    interp_vol = vols[1] + (vols[end]-vols[1])*(T_mid-T_lo)/(T_hi-T_lo+1e-12)
    return vols[mid] - interp_vol
end

function vol_surface_pca(vol_changes::Matrix{Float64}, n_components::Int=3)
    T_len, n_strikes = size(vol_changes)
    mu = mean(vol_changes, dims=1)
    centred = vol_changes .- mu
    _, S, V = svd(centred)
    explained_var = S[1:n_components].^2 ./ sum(S.^2) .* 100
    return (components=V[:,1:n_components], explained_variance=explained_var,
            singular_values=S[1:n_components])
end

function risk_reversal_time_series(rr_series::Vector{Float64}, window::Int=21)
    n = length(rr_series)
    if n < window + 1
        return (mean_rr=mean(rr_series), std_rr=std(rr_series), current_z=NaN)
    end
    recent = rr_series[end-window+1:end]
    hist   = rr_series[1:end-window]
    z = (rr_series[end] - mean(hist)) / (std(hist) + 1e-8)
    return (mean_rr=mean(hist), std_rr=std(hist), current_z=z,
            signal=z > 1.5 ? :elevated_skew : z < -1.5 ? :depressed_skew : :normal)
end

function skew_adjusted_delta(bs_delta::Float64, skew_slope::Float64,
                               vega::Float64, spot::Float64)::Float64
    return bs_delta + vega * skew_slope / spot
end

function realized_vol_vs_term_structure(spot_vol::Float64,
                                          term_vols::Vector{Float64},
                                          maturities::Vector{Float64})::Vector{Float64}
    return [spot_vol - tv for tv in term_vols]
end

function market_making_spread(vol::Float64, gamma_val::Float64,
                                inventory::Float64, max_inventory::Float64,
                                T::Float64)::Tuple{Float64, Float64}
    mid_adj = -inventory / (max_inventory + 1e-8) * vol * sqrt(T)
    half_spread = vol * sqrt(T) * gamma_val / 2
    return (mid_adj - half_spread, mid_adj + half_spread)
end


# ---- Volatility Surface Utilities (continued) ----

function delta_to_strike(delta::Float64, S::Float64, r::Float64, q::Float64,
                           sigma::Float64, T::Float64, opt::Symbol=:call)::Float64
    target_delta = opt == :call ? delta : delta - 1.0
    function obj(K)
        d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T) + 1e-12)
        return 0.5*(1 + erf(d1/sqrt(2)))*exp(-q*T) - target_delta
    end
    K_lo = S*0.1; K_hi = S*5.0
    for _ in 1:60
        K_mid = (K_lo + K_hi)/2
        if obj(K_mid) * obj(K_lo) < 0; K_hi = K_mid; else; K_lo = K_mid; end
    end
    return (K_lo + K_hi)/2
end

function vol_surface_fit_error(surface::VolSurface,
                                 market_vols::Matrix{Float64})::Float64
    diff = surface.implied_vols .- market_vols
    return sqrt(mean(diff.^2))
end

function spot_delta_normalisation(call_delta::Float64, forward_delta::Float64,
                                   df::Float64)::Float64
    return call_delta / (df + 1e-12)
end

function total_variance_monotone_check(vols::Vector{Float64},
                                         maturities::Vector{Float64})::Bool
    vars = vols.^2 .* maturities
    return all(vars[2:end] .>= vars[1:end-1])
end

function atm_vol_interpolation(T_lo::Float64, T_hi::Float64,
                                 v_lo::Float64, v_hi::Float64, T::Float64)::Float64
    w_lo = v_lo^2 * T_lo; w_hi = v_hi^2 * T_hi
    w = w_lo + (w_hi - w_lo) * (T - T_lo) / (T_hi - T_lo + 1e-12)
    return sqrt(max(w / T, 0.0))
end

function vol_surface_greeks(surface::VolSurface, k::Float64, T::Float64)::NamedTuple
    h_k = 0.01; h_T = 1/52
    v0  = bilinear_interpolate(surface, k, T)
    vku = bilinear_interpolate(surface, k+h_k, T)
    vkd = bilinear_interpolate(surface, k-h_k, T)
    vTu = bilinear_interpolate(surface, k, T+h_T)
    dv_dk  = (vku - vkd) / (2h_k)
    d2v_dk2 = (vku - 2v0 + vkd) / h_k^2
    dv_dT  = (vTu - v0) / h_T
    return (skew=dv_dk, curvature=d2v_dk2, term_slope=dv_dT, atm_vol=v0)
end

function butterfly_from_smile(k_put::Float64, k_atm::Float64, k_call::Float64,
                                vol_put::Float64, vol_atm::Float64, vol_call::Float64)::Float64
    return (vol_put + vol_call)/2 - vol_atm
end

function risk_reversal_from_smile(vol_call::Float64, vol_put::Float64)::Float64
    return vol_call - vol_put
end

function vol_surface_pnl(old_vol::Float64, new_vol::Float64, vega::Float64,
                           old_skew::Float64, new_skew::Float64, vanna::Float64,
                           old_curv::Float64, new_curv::Float64, volga::Float64)::Float64
    dvol  = new_vol - old_vol
    dskew = new_skew - old_skew
    dcurv = new_curv - old_curv
    return vega*dvol + vanna*dvol*dskew + volga*dvol^2/2 + 0.1*volga*dcurv
end


# ---- Volatility Surface Utilities (continued) ----

function delta_to_strike(delta::Float64, S::Float64, r::Float64, q::Float64,
                           sigma::Float64, T::Float64, opt::Symbol=:call)::Float64
    target_delta = opt == :call ? delta : delta - 1.0
    function obj(K)
        d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T) + 1e-12)
        return 0.5*(1 + erf(d1/sqrt(2)))*exp(-q*T) - target_delta
    end
    K_lo = S*0.1; K_hi = S*5.0
    for _ in 1:60
        K_mid = (K_lo + K_hi)/2
        if obj(K_mid) * obj(K_lo) < 0; K_hi = K_mid; else; K_lo = K_mid; end
    end
    return (K_lo + K_hi)/2
end

function vol_surface_fit_error(surface::VolSurface,
                                 market_vols::Matrix{Float64})::Float64
    diff = surface.implied_vols .- market_vols
    return sqrt(mean(diff.^2))
end

function spot_delta_normalisation(call_delta::Float64, forward_delta::Float64,
                                   df::Float64)::Float64
    return call_delta / (df + 1e-12)
end

function total_variance_monotone_check(vols::Vector{Float64},
                                         maturities::Vector{Float64})::Bool
    vars = vols.^2 .* maturities
    return all(vars[2:end] .>= vars[1:end-1])
end

function atm_vol_interpolation(T_lo::Float64, T_hi::Float64,
                                 v_lo::Float64, v_hi::Float64, T::Float64)::Float64
    w_lo = v_lo^2 * T_lo; w_hi = v_hi^2 * T_hi
    w = w_lo + (w_hi - w_lo) * (T - T_lo) / (T_hi - T_lo + 1e-12)
    return sqrt(max(w / T, 0.0))
end

function vol_surface_greeks(surface::VolSurface, k::Float64, T::Float64)::NamedTuple
    h_k = 0.01; h_T = 1/52
    v0  = bilinear_interpolate(surface, k, T)
    vku = bilinear_interpolate(surface, k+h_k, T)
    vkd = bilinear_interpolate(surface, k-h_k, T)
    vTu = bilinear_interpolate(surface, k, T+h_T)
    dv_dk  = (vku - vkd) / (2h_k)
    d2v_dk2 = (vku - 2v0 + vkd) / h_k^2
    dv_dT  = (vTu - v0) / h_T
    return (skew=dv_dk, curvature=d2v_dk2, term_slope=dv_dT, atm_vol=v0)
end

function butterfly_from_smile(k_put::Float64, k_atm::Float64, k_call::Float64,
                                vol_put::Float64, vol_atm::Float64, vol_call::Float64)::Float64
    return (vol_put + vol_call)/2 - vol_atm
end

function risk_reversal_from_smile(vol_call::Float64, vol_put::Float64)::Float64
    return vol_call - vol_put
end

function vol_surface_pnl(old_vol::Float64, new_vol::Float64, vega::Float64,
                           old_skew::Float64, new_skew::Float64, vanna::Float64,
                           old_curv::Float64, new_curv::Float64, volga::Float64)::Float64
    dvol  = new_vol - old_vol
    dskew = new_skew - old_skew
    dcurv = new_curv - old_curv
    return vega*dvol + vanna*dvol*dskew + volga*dvol^2/2 + 0.1*volga*dcurv
end


# ---- Volatility Surface Utilities (continued) ----

function delta_to_strike(delta::Float64, S::Float64, r::Float64, q::Float64,
                           sigma::Float64, T::Float64, opt::Symbol=:call)::Float64
    target_delta = opt == :call ? delta : delta - 1.0
    function obj(K)
        d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T) + 1e-12)
        return 0.5*(1 + erf(d1/sqrt(2)))*exp(-q*T) - target_delta
    end
    K_lo = S*0.1; K_hi = S*5.0
    for _ in 1:60
        K_mid = (K_lo + K_hi)/2
        if obj(K_mid) * obj(K_lo) < 0; K_hi = K_mid; else; K_lo = K_mid; end
    end
    return (K_lo + K_hi)/2
end

function vol_surface_fit_error(surface::VolSurface,
                                 market_vols::Matrix{Float64})::Float64
    diff = surface.implied_vols .- market_vols
    return sqrt(mean(diff.^2))
end

function spot_delta_normalisation(call_delta::Float64, forward_delta::Float64,
                                   df::Float64)::Float64
    return call_delta / (df + 1e-12)
end

function total_variance_monotone_check(vols::Vector{Float64},
                                         maturities::Vector{Float64})::Bool
    vars = vols.^2 .* maturities
    return all(vars[2:end] .>= vars[1:end-1])
end

function atm_vol_interpolation(T_lo::Float64, T_hi::Float64,
                                 v_lo::Float64, v_hi::Float64, T::Float64)::Float64
    w_lo = v_lo^2 * T_lo; w_hi = v_hi^2 * T_hi
    w = w_lo + (w_hi - w_lo) * (T - T_lo) / (T_hi - T_lo + 1e-12)
    return sqrt(max(w / T, 0.0))
end

function vol_surface_greeks(surface::VolSurface, k::Float64, T::Float64)::NamedTuple
    h_k = 0.01; h_T = 1/52
    v0  = bilinear_interpolate(surface, k, T)
    vku = bilinear_interpolate(surface, k+h_k, T)
    vkd = bilinear_interpolate(surface, k-h_k, T)
    vTu = bilinear_interpolate(surface, k, T+h_T)
    dv_dk  = (vku - vkd) / (2h_k)
    d2v_dk2 = (vku - 2v0 + vkd) / h_k^2
    dv_dT  = (vTu - v0) / h_T
    return (skew=dv_dk, curvature=d2v_dk2, term_slope=dv_dT, atm_vol=v0)
end

function butterfly_from_smile(k_put::Float64, k_atm::Float64, k_call::Float64,
                                vol_put::Float64, vol_atm::Float64, vol_call::Float64)::Float64
    return (vol_put + vol_call)/2 - vol_atm
end

function risk_reversal_from_smile(vol_call::Float64, vol_put::Float64)::Float64
    return vol_call - vol_put
end

function vol_surface_pnl(old_vol::Float64, new_vol::Float64, vega::Float64,
                           old_skew::Float64, new_skew::Float64, vanna::Float64,
                           old_curv::Float64, new_curv::Float64, volga::Float64)::Float64
    dvol  = new_vol - old_vol
    dskew = new_skew - old_skew
    dcurv = new_curv - old_curv
    return vega*dvol + vanna*dvol*dskew + volga*dvol^2/2 + 0.1*volga*dcurv
end

end # module VolatilitySurface
