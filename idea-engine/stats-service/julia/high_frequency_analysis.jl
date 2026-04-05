# =============================================================================
# high_frequency_analysis.jl — High-Frequency Trading Statistics
# =============================================================================
# Provides rigorous estimators for realised volatility, noise correction,
# microstructure analysis, and Hawkes processes in pure Julia.
#
# Key algorithms:
#   - Zhang-Mykland-Aït-Sahalia (2005) Two-Scales Realised Variance (TSRV)
#   - Pre-averaging estimator (Jacod et al. 2009)
#   - Realised Kernel with Parzen kernel (Barndorff-Nielsen et al. 2008)
#   - Diurnal adjustment via Fourier Flexible Form (Andersen-Bollerslev 1997)
#   - Hawkes process MLE and branching ratio
#   - Queue imbalance → mid-price impact (Cont-Kukanov 2013 style)
#   - Optimal sampling via signature plot
#   - Bid-ask bounce correction (Roll 1984)
#
# Julia ≥ 1.10 | No external packages (stdlib only)
# =============================================================================

module HighFrequencyAnalysis

using Statistics
using LinearAlgebra

export realized_variance, bipower_variation, two_scales_rv, pre_averaging_rv
export realized_kernel, optimal_sampling_frequency, signature_plot
export diurnal_adjustment, fourier_flexible_form
export fit_hawkes, simulate_hawkes, hawkes_branching_ratio
export queue_imbalance_signal, order_book_impact
export roll_spread_estimator, bid_ask_bounce_correction
export jump_test_bns, jump_test_lee_mykland
export trade_intensity_profile, diurnal_vol_factor

# =============================================================================
# SECTION 1: REALISED VARIANCE AND NOISE-ROBUST ESTIMATORS
# =============================================================================

"""
    realized_variance(log_prices; sampling=:all) -> Float64

Standard realised variance estimator: sum of squared log-returns.

Σ r²ᵢ where rᵢ = log(Pᵢ) - log(Pᵢ₋₁)

At very high frequencies, market microstructure noise inflates this estimator.
Use two_scales_rv or pre_averaging_rv for noise-robust estimates.

# Arguments
- `log_prices`: vector of log prices at observed times
- `sampling`: `:all` for all ticks, `:n` for every-nth-tick subsampling

# Returns
- Realised variance (daily units if prices are daily-sampled)
"""
function realized_variance(log_prices::Vector{Float64};
                            subsampling_skip::Int=1)::Float64
    n = length(log_prices)
    n < 2 && return 0.0

    rv = 0.0
    count = 0
    i = 1
    while i + subsampling_skip <= n
        r = log_prices[i + subsampling_skip] - log_prices[i]
        rv += r^2
        i += subsampling_skip
        count += 1
    end
    return rv
end

"""
    bipower_variation(log_prices) -> Float64

Realised bipower variation of Barndorff-Nielsen and Shephard (2004).
BPV = (π/2) * Σ |rᵢ| |rᵢ₋₁|

BPV is robust to finite-activity jumps: RV - BPV estimates jump component.
The ratio (RV - BPV)/RV estimates the fraction of variance from jumps.
"""
function bipower_variation(log_prices::Vector{Float64})::Float64
    n = length(log_prices)
    n < 3 && return 0.0

    returns = diff(log_prices)
    m = length(returns)

    bpv = 0.0
    for i in 2:m
        bpv += abs(returns[i]) * abs(returns[i-1])
    end
    return (π / 2) * bpv
end

"""
    two_scales_rv(log_prices; K=nothing) -> NamedTuple

Two-Scales Realised Variance (TSRV) of Zhang, Mykland, Aït-Sahalia (2005).

Decomposes RV into a "slow" scale (sparse sampling, lower noise contamination)
and a "fast" scale (all ticks, maximum noise), then combines:

    TSRV = RV_slow - (n_slow/n_fast) * RV_fast

where n_slow, n_fast are the number of returns at each scale.

This is the first consistent estimator for the quadratic variation of a
semimartingale observed with i.i.d. additive noise.

# Arguments
- `log_prices`: tick-by-tick log prices
- `K`: averaging lag (default: cbrt(n) as in ZMA 2005)

# Returns
- NamedTuple with fields: tsrv, rv_slow, rv_fast, noise_variance, optimal_K
"""
function two_scales_rv(log_prices::Vector{Float64};
                        K::Union{Int,Nothing}=nothing)

    n = length(log_prices)
    n < 10 && return (tsrv=0.0, rv_slow=0.0, rv_fast=0.0,
                       noise_variance=0.0, optimal_K=1)

    # Optimal K from ZMA (2005): K* = (n * n_bar)^(1/3) where n_bar is mean
    # In practice K* ≈ n^(1/3) gives good finite-sample performance
    K_opt = K === nothing ? max(1, round(Int, cbrt(n))) : K

    # Fast scale: all tick returns (full sample)
    rv_fast = realized_variance(log_prices; subsampling_skip=1)

    # Slow scale: average over K subgrids
    rv_subgrid = zeros(K_opt)
    for k in 1:K_opt
        # Subgrid k: prices at indices k, k+K, k+2K, ...
        subgrid_idx = collect(k:K_opt:n)
        if length(subgrid_idx) < 2
            rv_subgrid[k] = 0.0
        else
            rv_subgrid[k] = realized_variance(log_prices[subgrid_idx]; subsampling_skip=1)
        end
    end
    rv_slow = mean(rv_subgrid)

    # Number of returns in slow and fast scales
    n_slow = (n - K_opt) / K_opt  # approximate
    n_fast = n - 1.0

    # TSRV with finite-sample correction
    correction = n_slow / n_fast
    tsrv = rv_slow - correction * rv_fast

    # Noise variance estimate: E[ε²] ≈ RV_fast / (2n)
    noise_var = rv_fast / (2 * n_fast)

    return (tsrv=max(tsrv, 0.0), rv_slow=rv_slow, rv_fast=rv_fast,
             noise_variance=noise_var, optimal_K=K_opt)
end

"""
    pre_averaging_rv(log_prices; theta=0.5, g_func=nothing) -> NamedTuple

Pre-averaging estimator for noisy high-frequency data (Jacod et al. 2009).

Pre-averaging smooths out the noise by replacing each price with a local
weighted average before computing squared returns. The key-weight function
g(x) on [0,1] is typically triangular: g(x) = min(x, 1-x).

    P̄ᵢ = Σⱼ g(j/kₙ) * ΔPᵢ₊ⱼ  (pre-averaged increment)
    PAV = (1/θ√n) * Σᵢ P̄ᵢ² - (1/2θ²n) * Σᵢ ΔPᵢ²

where kₙ = θ√n is the pre-averaging window.

# Arguments
- `log_prices`: tick prices
- `theta`: window parameter θ, typically 0.5 (window = θ√n)
- `g_func`: weight function on [0,1], default triangular

# Returns
- NamedTuple: pav (pre-avg variance), kn (window used), noise_var
"""
function pre_averaging_rv(log_prices::Vector{Float64};
                            theta::Float64=0.5,
                            g_func::Union{Function,Nothing}=nothing)

    n = length(log_prices)
    n < 10 && return (pav=0.0, kn=1, noise_var=0.0)

    # Window length kₙ = ⌊θ√n⌋
    kn = max(2, floor(Int, theta * sqrt(n)))

    # Default weight function: triangular g(x) = min(x, 1-x)
    g = g_func === nothing ? (x -> min(x, 1.0 - x)) : g_func

    # Precompute g weights: g(j/kn) for j = 1, ..., kn-1
    g_weights = [g(j / kn) for j in 1:(kn-1)]
    psi1 = sum(g_weights)       # Σ g(j/kn)
    psi2 = sum(g_weights .^ 2)  # Σ g²(j/kn)

    # Log returns
    dP = diff(log_prices)
    m = length(dP)

    # Pre-averaged increments: Ȳᵢ = Σⱼ g(j/kn) * dP[i+j]
    num_avg = m - kn + 1
    num_avg < 1 && return (pav=0.0, kn=kn, noise_var=0.0)

    Y_avg = zeros(num_avg)
    for i in 1:num_avg
        for (jdx, j) in enumerate(1:(kn-1))
            if i + j - 1 <= m
                Y_avg[i] += g_weights[jdx] * dP[i + j - 1]
            end
        end
    end

    # Pre-averaged variance
    sum_Y2 = sum(Y_avg .^ 2)
    sum_dP2 = sum(dP .^ 2)  # for noise correction

    # Noise variance estimate: σ²_ε ≈ (1/2n) * Σ dP²
    noise_var = sum_dP2 / (2 * m)

    # PAV formula with noise correction
    psi2_kn = psi2 / kn
    pav = (1.0 / (kn * psi2_kn)) * sum_Y2 - (psi1^2 / psi2) * noise_var

    return (pav=max(pav, 0.0), kn=kn, noise_var=noise_var)
end

"""
    realized_kernel(log_prices; kernel=:parzen, H=nothing) -> NamedTuple

Realised Kernel estimator (Barndorff-Nielsen et al. 2008).

The RK estimator applies a kernel to the autocovariances of returns:
    RK = Σₕ₌₋H^H k(h/H) * γ̂ₕ

where γ̂ₕ = Σᵢ rᵢ rᵢ₋ₕ is the h-th order autocovariance.

The Parzen kernel k(x) has excellent asymptotic properties:
    k(x) = 1 - 6x² + 6|x|³   for |x| ≤ 0.5
    k(x) = 2(1-|x|)³          for 0.5 < |x| ≤ 1
    k(x) = 0                   otherwise

Bandwidth H* = c_k * ξ^(4/5) * n^(3/5) (from Barndorff-Nielsen et al.)

# Arguments
- `log_prices`: tick log prices
- `kernel`: :parzen (default), :bartlett, :quadratic_spectral, :tukey_hanning
- `H`: bandwidth (auto-selected if nothing)

# Returns
- NamedTuple: rk, H, autocovariances
"""
function realized_kernel(log_prices::Vector{Float64};
                          kernel::Symbol=:parzen,
                          H::Union{Int,Nothing}=nothing)

    n = length(log_prices)
    n < 4 && return (rk=0.0, H=0, autocovariances=Float64[])

    returns = diff(log_prices)
    m = length(returns)

    # Kernel functions
    function parzen_kernel(x::Float64)::Float64
        ax = abs(x)
        ax > 1.0 && return 0.0
        ax <= 0.5 && return 1.0 - 6ax^2 + 6ax^3
        return 2.0 * (1.0 - ax)^3
    end

    function bartlett_kernel(x::Float64)::Float64
        ax = abs(x)
        ax > 1.0 ? 0.0 : 1.0 - ax
    end

    function tukey_hanning_kernel(x::Float64)::Float64
        abs(x) > 1.0 ? 0.0 : 0.5 * (1.0 + cos(π * x))
    end

    function qs_kernel(x::Float64)::Float64
        if abs(x) < 1e-10
            return 1.0
        end
        z = 6π * x / 5
        return 3.0 / z^2 * (sin(z)/z - cos(z))
    end

    kfunc = kernel == :parzen           ? parzen_kernel :
            kernel == :bartlett         ? bartlett_kernel :
            kernel == :tukey_hanning    ? tukey_hanning_kernel :
                                          qs_kernel

    # Autocovariances γ̂ₕ = (1/n) * Σᵢ rᵢ rᵢ₋ₕ
    H_use = H === nothing ? max(1, round(Int, cbrt(m))) : H
    H_use = min(H_use, m - 1)

    acovs = zeros(H_use + 1)
    for h in 0:H_use
        s = 0.0
        for i in (h+1):m
            s += returns[i] * returns[i - h]
        end
        acovs[h+1] = s
    end

    # RK = γ̂₀ + 2 * Σₕ₌₁ᴴ k(h/H) * γ̂ₕ
    rk = acovs[1]  # h=0: k(0)=1
    for h in 1:H_use
        rk += 2.0 * kfunc(h / H_use) * acovs[h+1]
    end

    return (rk=max(rk, 0.0), H=H_use, autocovariances=acovs)
end

# =============================================================================
# SECTION 2: OPTIMAL SAMPLING AND SIGNATURE PLOT
# =============================================================================

"""
    signature_plot(log_prices; max_lag=100) -> NamedTuple

Compute the signature plot: RV as a function of sampling frequency.

At very high frequencies, market microstructure noise inflates RV.
At very low frequencies, discretization error inflates variance.
The optimal sampling frequency is where the signature plot is flattest.

Plot RV(Δt) vs Δt and find the plateau region.

# Returns
- NamedTuple: lags, rv_values, optimal_lag, noise_variance, signal_variance
"""
function signature_plot(log_prices::Vector{Float64}; max_lag::Int=100)
    n = length(log_prices)
    max_lag = min(max_lag, n ÷ 4)
    max_lag < 2 && return (lags=Int[], rv_values=Float64[],
                             optimal_lag=1, noise_variance=0.0, signal_variance=0.0)

    lags = collect(1:max_lag)
    rv_values = zeros(max_lag)

    for (i, lag) in enumerate(lags)
        rv_values[i] = realized_variance(log_prices; subsampling_skip=lag)
        # Scale to be comparable: RV per unit of time
        num_returns = (n - 1) ÷ lag
        rv_values[i] /= num_returns
    end

    # Noise variance from Roll (1984): E[RV(1)] ≈ σ² + 2σ²_ε
    # From lag-1 autocov: Cov(r_t, r_{t-1}) ≈ -σ²_ε
    returns = diff(log_prices)
    m = length(returns)
    if m > 2
        acov1 = sum(returns[2:end] .* returns[1:end-1]) / (m - 1)
        noise_var = max(0.0, -acov1)
    else
        noise_var = 0.0
    end

    # Signal variance: approximately the plateau value of signature plot
    # Find the lag where RV stabilizes (smallest coefficient of variation)
    window = max(3, max_lag ÷ 5)
    best_cv = Inf
    optimal_lag = lags[end ÷ 2]
    for i in window:(max_lag - window)
        segment = rv_values[max(1,i-window):min(max_lag,i+window)]
        μ = mean(segment)
        σ = std(segment)
        cv = μ > 0 ? σ / μ : Inf
        if cv < best_cv
            best_cv = cv
            optimal_lag = lags[i]
        end
    end

    signal_var = rv_values[optimal_lag]

    return (lags=lags, rv_values=rv_values, optimal_lag=optimal_lag,
             noise_variance=noise_var, signal_variance=signal_var)
end

"""
    optimal_sampling_frequency(log_prices, times) -> NamedTuple

Estimate the optimal sampling frequency for RV estimation using the
MSE-minimizing criterion from Bandi & Russell (2006).

MSE of RV = Bias² + Variance
  Bias = 2n * σ²_ε (noise)
  Variance ∝ IQ / n (integrated quarticity)

Optimal n* = (4 * IQ / (4 * σ⁴_ε))^(1/3)

# Arguments
- `log_prices`: tick log prices
- `times`: observation times in seconds

# Returns
- NamedTuple: optimal_seconds, noise_var, iq_estimate, recommended_n
"""
function optimal_sampling_frequency(log_prices::Vector{Float64},
                                     times::Vector{Float64})
    n = length(log_prices)
    n < 10 && return (optimal_seconds=60.0, noise_var=0.0,
                       iq_estimate=0.0, recommended_n=n)

    returns = diff(log_prices)
    dt = diff(times)
    m = length(returns)

    # Noise variance from negative autocorrelation
    acov1 = 0.0
    for i in 2:m
        acov1 += returns[i] * returns[i-1]
    end
    acov1 /= (m - 1)
    noise_var = max(0.0, -acov1)

    # Integrated quarticity: Σ r⁴ / (3 * dt) scaled
    rv4 = sum(returns .^ 4)
    iq_estimate = rv4 / (3 * mean(dt) * m)

    # Optimal sampling: n* = (IQ / (4*σ⁴_ε))^(1/3)
    if noise_var > 1e-15
        optimal_n = (iq_estimate / (4 * noise_var^2))^(1/3)
    else
        optimal_n = n / 10.0  # fallback: subsample to 10% of ticks
    end

    total_time = times[end] - times[1]
    optimal_seconds = max(1.0, total_time / optimal_n)

    return (optimal_seconds=optimal_seconds, noise_var=noise_var,
             iq_estimate=iq_estimate, recommended_n=round(Int, optimal_n))
end

# =============================================================================
# SECTION 3: DIURNAL ADJUSTMENT
# =============================================================================

"""
    fourier_flexible_form(time_of_day, returns; n_harmonics=5) -> NamedTuple

Estimate the intraday diurnal pattern using the Fourier Flexible Form of
Andersen and Bollerslev (1997, 1998).

The seasonal component s(τ) at time-of-day τ ∈ [0, 1] is:
    s(τ) = exp(δ₀ + Σₖ [δₖ sin(2πkτ) + φₖ cos(2πkτ)])

Fit by regressing |r_t|² on Fourier basis functions.

# Arguments
- `time_of_day`: fraction of trading day [0,1] for each observation
- `returns`: corresponding log returns
- `n_harmonics`: number of Fourier harmonics (default 5)

# Returns
- NamedTuple: coefficients, predict (function), r_squared
"""
function fourier_flexible_form(time_of_day::Vector{Float64},
                                 returns::Vector{Float64};
                                 n_harmonics::Int=5)

    n = length(returns)
    @assert length(time_of_day) == n "time_of_day and returns must match"

    # Response: log(r² + ε) to handle zeros
    eps_floor = 1e-10
    y = log.(returns .^ 2 .+ eps_floor)

    # Design matrix: [1, sin(2πτ), cos(2πτ), sin(4πτ), cos(4πτ), ...]
    n_params = 1 + 2 * n_harmonics
    X = ones(n, n_params)
    for k in 1:n_harmonics
        X[:, 2k]   = sin.(2π * k * time_of_day)
        X[:, 2k+1] = cos.(2π * k * time_of_day)
    end

    # OLS: β = (X'X)⁻¹ X'y
    XtX = X' * X
    Xty = X' * y

    # Regularised solve for stability
    beta = try
        (XtX + 1e-8 * I) \ Xty
    catch
        zeros(n_params)
    end

    # R-squared
    y_hat = X * beta
    ss_res = sum((y - y_hat) .^ 2)
    ss_tot = sum((y .- mean(y)) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0

    # Predict function: s(τ) as a multiplicative seasonal factor
    function predict(tau::Float64)::Float64
        x = ones(n_params)
        for k in 1:n_harmonics
            x[2k]   = sin(2π * k * tau)
            x[2k+1] = cos(2π * k * tau)
        end
        exp(dot(beta, x))
    end

    function predict_vec(taus::Vector{Float64})::Vector{Float64}
        [predict(τ) for τ in taus]
    end

    return (coefficients=beta, predict=predict_vec, r_squared=r2,
             n_harmonics=n_harmonics)
end

"""
    diurnal_adjustment(returns, time_of_day; n_harmonics=5) -> NamedTuple

Remove intraday seasonality from returns to get diurnally-adjusted series.

Method:
  1. Fit Fourier Flexible Form to estimate s(τ)
  2. Scale: r̃_t = r_t / sqrt(s(τ_t))
  3. Adjusted series has uniform intraday variance

# Returns
- NamedTuple: adjusted_returns, seasonal_factors, fit
"""
function diurnal_adjustment(returns::Vector{Float64},
                              time_of_day::Vector{Float64};
                              n_harmonics::Int=5)

    n = length(returns)
    n < 20 && return (adjusted_returns=returns, seasonal_factors=ones(n), fit=nothing)

    fit = fourier_flexible_form(time_of_day, returns; n_harmonics=n_harmonics)

    # Seasonal factors at each observation time
    seasonal = fit.predict(time_of_day)

    # Normalize: divide by sqrt of seasonal factor so variance is unit
    seasonal_norm = seasonal ./ mean(seasonal)
    adjusted = returns ./ sqrt.(max.(seasonal_norm, 1e-10))

    return (adjusted_returns=adjusted, seasonal_factors=seasonal_norm, fit=fit)
end

"""
    trade_intensity_profile(timestamps, session_start, session_end; n_bins=78) -> NamedTuple

Compute the average number of trades per time bin throughout the trading day.
Typical result: U-shaped pattern (high at open/close, low at midday).

# Arguments
- `timestamps`: trade arrival times (Unix seconds or minutes)
- `session_start`, `session_end`: start/end of trading session
- `n_bins`: number of time bins (default 78 = 5-min bins in 6.5-hr day)

# Returns
- NamedTuple: bin_centers, avg_intensity, normalized_intensity
"""
function trade_intensity_profile(timestamps::Vector{Float64},
                                   session_start::Float64,
                                   session_end::Float64;
                                   n_bins::Int=78)

    session_len = session_end - session_start
    bin_width = session_len / n_bins

    counts = zeros(n_bins)
    for t in timestamps
        frac = (t - session_start) / session_len
        if 0.0 <= frac < 1.0
            bin_idx = floor(Int, frac * n_bins) + 1
            bin_idx = clamp(bin_idx, 1, n_bins)
            counts[bin_idx] += 1.0
        end
    end

    bin_centers = [session_start + (i - 0.5) * bin_width for i in 1:n_bins]
    avg_intensity = counts ./ bin_width  # trades per unit time
    total = sum(avg_intensity)
    normalized = total > 0 ? avg_intensity ./ mean(avg_intensity) : ones(n_bins)

    return (bin_centers=bin_centers, avg_intensity=avg_intensity,
             normalized_intensity=normalized)
end

"""
    diurnal_vol_factor(time_of_day, seasonal_profile) -> Float64

Given a time-of-day fraction and a seasonal profile (from diurnal_adjustment),
return the multiplicative volatility factor for that time.
"""
function diurnal_vol_factor(time_of_day::Float64,
                              seasonal_profile::Function)::Float64
    sqrt(max(seasonal_profile(time_of_day), 1e-10))
end

# =============================================================================
# SECTION 4: HAWKES PROCESS
# =============================================================================

"""
    fit_hawkes(event_times; T=nothing, max_iter=200) -> NamedTuple

Fit a univariate Hawkes process via MLE.

The Hawkes process has intensity:
    λ(t) = μ + α * Σ_{tᵢ < t} exp(-β(t - tᵢ))

Parameters: μ (baseline), α (jump), β (decay). Branching ratio n = α/β < 1.

Log-likelihood (Ozaki 1979):
    ℓ = -∫₀ᵀ λ(t)dt + Σᵢ log λ(tᵢ)

Uses coordinate descent / gradient optimization.

# Arguments
- `event_times`: sorted vector of event arrival times
- `T`: end of observation window (default: last event time)
- `max_iter`: maximum EM/gradient iterations

# Returns
- NamedTuple: mu, alpha, beta, branching_ratio, loglik, converged
"""
function fit_hawkes(event_times::Vector{Float64};
                     T::Union{Float64,Nothing}=nothing,
                     max_iter::Int=200)

    isempty(event_times) && return (mu=0.0, alpha=0.0, beta=1.0,
                                      branching_ratio=0.0, loglik=-Inf, converged=false)

    sort_times = sort(event_times)
    n = length(sort_times)
    T_end = T === nothing ? sort_times[end] : T

    # Evaluate log-likelihood for given parameters
    function hawkes_loglik(params::Vector{Float64})::Float64
        mu, alpha, beta = params
        (mu <= 0 || alpha <= 0 || beta <= 0 || alpha >= beta) && return -1e15

        # Compute ∫₀ᵀ λ(t) dt = μT + α/β * Σᵢ (1 - exp(-β(T-tᵢ)))
        integral = mu * T_end
        for t in sort_times
            integral += (alpha / beta) * (1.0 - exp(-beta * (T_end - t)))
        end

        # Compute Σᵢ log λ(tᵢ)
        log_sum = 0.0
        for i in 1:n
            lam_i = mu
            for j in 1:(i-1)
                lam_i += alpha * exp(-beta * (sort_times[i] - sort_times[j]))
            end
            lam_i <= 0 && return -1e15
            log_sum += log(lam_i)
        end

        return -integral + log_sum
    end

    # Efficient O(n) recursion for log-likelihood using the Ogata recursion
    function hawkes_loglik_fast(mu::Float64, alpha::Float64, beta::Float64)::Float64
        (mu <= 0 || alpha <= 0 || beta <= 0 || alpha >= beta) && return -1e15

        log_sum = 0.0
        A = 0.0  # recursive sum: Σ_{j<i} exp(-β(tᵢ - tⱼ))

        integral = mu * T_end

        for i in 1:n
            if i > 1
                dt = sort_times[i] - sort_times[i-1]
                A = exp(-beta * dt) * (1.0 + A)
            end
            lam_i = mu + alpha * A
            lam_i <= 0 && return -1e15
            log_sum += log(lam_i)
        end

        # Integral term: μT + α/β * Σᵢ (1 - exp(-β(T-tᵢ)))
        for t in sort_times
            integral += (alpha / beta) * (1.0 - exp(-beta * (T_end - t)))
        end

        return -integral + log_sum
    end

    # Initial parameter guess
    rate = n / T_end
    mu0  = rate * 0.3
    beta0 = 5.0 * rate
    alpha0 = 0.5 * beta0

    # Gradient-free coordinate search (Nelder-Mead style simplex)
    best_ll = hawkes_loglik_fast(mu0, alpha0, beta0)
    best_mu, best_alpha, best_beta = mu0, alpha0, beta0

    # Grid search to get good starting point
    for mu_try in [rate * f for f in [0.1, 0.3, 0.5]]
        for alpha_frac in [0.2, 0.4, 0.6, 0.8]
            for beta_mult in [1.0, 5.0, 10.0, 20.0]
                beta_try = beta_mult * rate
                alpha_try = alpha_frac * beta_try
                ll = hawkes_loglik_fast(mu_try, alpha_try, beta_try)
                if ll > best_ll
                    best_ll = ll
                    best_mu, best_alpha, best_beta = mu_try, alpha_try, beta_try
                end
            end
        end
    end

    # Coordinate descent refinement
    converged = false
    step = 0.1
    for iter in 1:max_iter
        improved = false
        for (param_idx, direction) in [(1,1),(1,-1),(2,1),(2,-1),(3,1),(3,-1)]
            params = [best_mu, best_alpha, best_beta]
            params[param_idx] *= (1.0 + direction * step)
            if all(params .> 0) && params[2] < params[3]
                ll = hawkes_loglik_fast(params[1], params[2], params[3])
                if ll > best_ll
                    best_ll = ll
                    best_mu, best_alpha, best_beta = params[1], params[2], params[3]
                    improved = true
                end
            end
        end
        if !improved
            step *= 0.5
            step < 1e-8 && (converged = true; break)
        end
    end

    branching = best_alpha / best_beta

    return (mu=best_mu, alpha=best_alpha, beta=best_beta,
             branching_ratio=branching, loglik=best_ll, converged=converged)
end

"""
    hawkes_branching_ratio(event_times; T=nothing) -> Float64

Convenience function: fit Hawkes and return branching ratio n = α/β.

Branching ratio < 1: stationary process (required for stability)
Branching ratio ≈ 0: near-Poisson (no self-excitement)
Branching ratio ≈ 0.9: highly self-exciting (microstructure noise, HFT feedback)
"""
function hawkes_branching_ratio(event_times::Vector{Float64};
                                  T::Union{Float64,Nothing}=nothing)::Float64
    fit = fit_hawkes(event_times; T=T)
    return fit.branching_ratio
end

"""
    simulate_hawkes(mu, alpha, beta, T; seed=42) -> Vector{Float64}

Simulate a univariate Hawkes process via Ogata's modified thinning algorithm.

# Arguments
- `mu`: baseline intensity
- `alpha`: jump size
- `beta`: decay rate
- `T`: simulation horizon
- `seed`: random seed

# Returns
- Sorted vector of event arrival times in [0, T]
"""
function simulate_hawkes(mu::Float64, alpha::Float64, beta::Float64,
                          T::Float64; seed::Int=42)::Vector{Float64}

    alpha >= beta && error("Branching ratio α/β must be < 1 for stationarity")

    rng_state = seed  # simple LCG for determinism
    function next_rand()::Float64
        rng_state = (1664525 * rng_state + 1013904223) % (2^32)
        rng_state / 2^32
    end

    events = Float64[]
    t = 0.0
    lambda_bar = mu  # current upper bound on intensity

    while t < T
        # Draw candidate from exponential(lambda_bar)
        u = next_rand()
        u <= 0.0 && (u = 1e-10)
        dt = -log(u) / lambda_bar
        t += dt
        t > T && break

        # Compute actual intensity at t
        lambda_t = mu
        for s in events
            lambda_t += alpha * exp(-beta * (t - s))
        end

        # Accept/reject
        u2 = next_rand()
        if u2 <= lambda_t / lambda_bar
            push!(events, t)
            lambda_bar = lambda_t + alpha
        else
            lambda_bar = lambda_t
        end

        lambda_bar = max(lambda_bar, mu)
    end

    return events
end

# =============================================================================
# SECTION 5: ORDER BOOK AND MARKET IMPACT
# =============================================================================

"""
    queue_imbalance_signal(bid_sizes, ask_sizes) -> Vector{Float64}

Compute order book queue imbalance at each observation:
    OI = (V_bid - V_ask) / (V_bid + V_ask)

Cont, Kukanov, Stoikov (2014): OI is a leading indicator of short-term
mid-price changes. Positive OI → price likely to rise.

# Arguments
- `bid_sizes`: vector of total bid-side volume at best bid
- `ask_sizes`: vector of total ask-side volume at best ask

# Returns
- Imbalance signal ∈ [-1, 1]
"""
function queue_imbalance_signal(bid_sizes::Vector{Float64},
                                  ask_sizes::Vector{Float64})::Vector{Float64}
    n = length(bid_sizes)
    @assert length(ask_sizes) == n "bid and ask sizes must match"

    imbalance = zeros(n)
    for i in 1:n
        total = bid_sizes[i] + ask_sizes[i]
        if total > 0
            imbalance[i] = (bid_sizes[i] - ask_sizes[i]) / total
        end
    end
    return imbalance
end

"""
    order_book_impact(imbalance, mid_price_changes; lag=1) -> NamedTuple

Regress future mid-price changes on current order book imbalance.

Model: Δm_{t+lag} = a + b * OI_t + ε_t

# Returns
- NamedTuple: alpha, beta, r_squared, t_stat
"""
function order_book_impact(imbalance::Vector{Float64},
                             mid_price_changes::Vector{Float64};
                             lag::Int=1)

    n = length(imbalance)
    n_lag = n - lag
    n_lag < 5 && return (alpha=0.0, beta=0.0, r_squared=0.0, t_stat=0.0)

    X = imbalance[1:n_lag]
    y = mid_price_changes[(1+lag):n]

    x_mean = mean(X)
    y_mean = mean(y)

    sxx = sum((X .- x_mean) .^ 2)
    sxy = sum((X .- x_mean) .* (y .- y_mean))

    beta = sxx > 0 ? sxy / sxx : 0.0
    alpha = y_mean - beta * x_mean

    y_hat = alpha .+ beta .* X
    residuals = y .- y_hat
    ss_res = sum(residuals .^ 2)
    ss_tot = sum((y .- y_mean) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0

    # t-statistic for beta
    if n_lag > 2 && sxx > 0
        s2 = ss_res / (n_lag - 2)
        se_beta = sqrt(s2 / sxx)
        t_stat = se_beta > 0 ? beta / se_beta : 0.0
    else
        t_stat = 0.0
    end

    return (alpha=alpha, beta=beta, r_squared=r2, t_stat=t_stat)
end

# =============================================================================
# SECTION 6: BID-ASK SPREAD AND BOUNCE CORRECTION
# =============================================================================

"""
    roll_spread_estimator(log_prices) -> NamedTuple

Roll (1984) implied spread estimator from transaction prices.

If Pₜ = mₜ + qₜ * s/2 where qₜ ∈ {-1,+1} is trade direction,
then Cov(ΔPₜ, ΔPₜ₋₁) = -s²/4, so:

    ŝ = 2 * sqrt(max(0, -Cov(ΔPₜ, ΔPₜ₋₁)))

# Returns
- NamedTuple: spread, autocovariance, bounce_component
"""
function roll_spread_estimator(log_prices::Vector{Float64})

    n = length(log_prices)
    n < 3 && return (spread=0.0, autocovariance=0.0, bounce_component=0.0)

    returns = diff(log_prices)
    m = length(returns)

    r_mean = mean(returns)
    acov1 = sum((returns[2:end] .- r_mean) .* (returns[1:end-1] .- r_mean)) / (m - 1)

    spread = 2.0 * sqrt(max(0.0, -acov1))
    bounce_component = -acov1  # = s²/4

    return (spread=spread, autocovariance=acov1, bounce_component=bounce_component)
end

"""
    bid_ask_bounce_correction(log_prices; method=:roll) -> NamedTuple

Correct the realized variance for bid-ask bounce contamination.

Under Roll's model:
    RV_true = RV_observed + 2n * Cov(Δrₜ, Δrₜ₋₁)
            = RV_observed - n * s²/2

where s is the effective spread and n is the number of observations.

# Arguments
- `log_prices`: observed transaction log prices
- `method`: :roll (default) or :hasbrouck

# Returns
- NamedTuple: rv_corrected, rv_raw, spread_estimate, correction_term
"""
function bid_ask_bounce_correction(log_prices::Vector{Float64};
                                    method::Symbol=:roll)

    n = length(log_prices)
    n < 3 && return (rv_corrected=0.0, rv_raw=0.0,
                      spread_estimate=0.0, correction_term=0.0)

    rv_raw = realized_variance(log_prices)
    roll = roll_spread_estimator(log_prices)

    # Correction: add back 2(n-1) times the (negative) autocovariance
    m = n - 1  # number of returns
    correction = -2.0 * (m - 1) * roll.autocovariance  # positive if bounce
    rv_corrected = max(0.0, rv_raw - correction)

    return (rv_corrected=rv_corrected, rv_raw=rv_raw,
             spread_estimate=roll.spread, correction_term=correction)
end

# =============================================================================
# SECTION 7: JUMP TESTS
# =============================================================================

"""
    jump_test_bns(log_prices; significance=0.05) -> NamedTuple

Barndorff-Nielsen & Shephard (2006) ratio-based jump test.

Test statistic: J = (RV - BPV) / (sqrt(n) * ω * sqrt(BPV²))
where ω² = (π²/4 + π - 5) (tripower quarticity factor)

Under H₀ (no jumps): J → N(0,1)

# Returns
- NamedTuple: statistic, p_value, has_jump, rv, bpv, jump_component
"""
function jump_test_bns(log_prices::Vector{Float64};
                        significance::Float64=0.05)

    n = length(log_prices)
    n < 5 && return (statistic=0.0, p_value=1.0, has_jump=false,
                      rv=0.0, bpv=0.0, jump_component=0.0)

    rv = realized_variance(log_prices)
    bpv = bipower_variation(log_prices)

    # Tripower quarticity for variance of BPV
    returns = abs.(diff(log_prices))
    m = length(returns)
    mu_1 = sqrt(2/π)  # E[|Z|] for Z ~ N(0,1)

    tpq = 0.0
    for i in 3:m
        tpq += returns[i]^(4/3) * returns[i-1]^(4/3) * returns[i-2]^(4/3)
    end
    tpq *= (m / (m-2)) * mu_1^(-3)

    # BNS test statistic
    omega_sq = (π^2 / 4 + π - 5)
    var_stat = omega_sq * max(tpq, 1e-20) / bpv^2
    stat_num = (rv - bpv) / rv

    if var_stat > 0 && m > 0
        test_stat = stat_num / sqrt(var_stat / m)
    else
        test_stat = 0.0
    end

    # One-sided p-value (testing RV > BPV)
    p_val = 1.0 - _normal_cdf(test_stat)

    return (statistic=test_stat, p_value=p_val, has_jump=(p_val < significance),
             rv=rv, bpv=bpv, jump_component=max(rv - bpv, 0.0))
end

"""
    jump_test_lee_mykland(log_prices, times; significance=0.05) -> NamedTuple

Lee & Mykland (2008) test for individual jump times.

For each return rₜ, test statistic:
    L(t) = rₜ / σ̂(t)

where σ̂(t) is a local volatility estimate. Under H₀: L(t) ~ |N(0,1)|.

Critical value based on extreme value distribution.

# Returns
- NamedTuple: jump_times, jump_sizes, test_statistics, threshold
"""
function jump_test_lee_mykland(log_prices::Vector{Float64},
                                 times::Vector{Float64};
                                 significance::Float64=0.05)

    n = length(log_prices)
    n < 5 && return (jump_times=Float64[], jump_sizes=Float64[],
                      test_statistics=Float64[], threshold=0.0)

    returns = diff(log_prices)
    m = length(returns)

    # Local volatility window: K = ceil(sqrt(m))
    K = max(3, ceil(Int, sqrt(m)))

    # For each return, estimate local BPV over window [i-K, i+K]
    local_vol = zeros(m)
    for i in 1:m
        i_start = max(1, i - K)
        i_end   = min(m, i + K)
        window_ret = returns[i_start:i_end]
        if length(window_ret) > 2
            bpv_w = 0.0
            for j in 2:length(window_ret)
                bpv_w += abs(window_ret[j]) * abs(window_ret[j-1])
            end
            bpv_w *= (π / 2)
            local_vol[i] = sqrt(bpv_w / (length(window_ret) - 1))
        else
            local_vol[i] = std(returns)
        end
    end

    # Test statistics |L(t)| = |r_t| / σ̂(t)
    L = zeros(m)
    for i in 1:m
        L[i] = local_vol[i] > 0 ? abs(returns[i]) / local_vol[i] : 0.0
    end

    # Extreme value critical value (Gumbel distribution)
    beta_m = (2 * log(m))^0.5 - (log(π) + log(log(m))) / (2 * (2 * log(m))^0.5)
    c_m = 1.0 / (2 * log(m))^0.5
    threshold = beta_m - c_m * log(-log(1 - significance))

    jump_indices = findall(L .> threshold)
    jump_times = isempty(jump_indices) ? Float64[] : times[jump_indices .+ 1]
    jump_sizes = isempty(jump_indices) ? Float64[] : returns[jump_indices]

    return (jump_times=jump_times, jump_sizes=jump_sizes,
             test_statistics=L, threshold=threshold)
end

# =============================================================================
# SECTION 8: ADVERSE SELECTION AND TRADE CLASSIFICATION
# =============================================================================

"""
    tick_rule_classify(prices) -> Vector{Int}

Classify trades as buyer-initiated (+1) or seller-initiated (-1) using
the Lee-Ready tick rule: if price up → buy; if price down → sell; else same.
"""
function tick_rule_classify(prices::Vector{Float64})::Vector{Int}
    n = length(prices)
    signs = zeros(Int, n)
    last_sign = 1
    for i in 2:n
        dp = prices[i] - prices[i-1]
        if dp > 0
            signs[i] = 1
            last_sign = 1
        elseif dp < 0
            signs[i] = -1
            last_sign = -1
        else
            signs[i] = last_sign  # zero-tick: use last
        end
    end
    signs[1] = signs[2] != 0 ? signs[2] : 1
    return signs
end

"""
    kyle_lambda(price_changes, signed_volumes) -> NamedTuple

Estimate Kyle's lambda: price impact coefficient from regression
    ΔP = λ * Q + ε

where Q is signed order flow (buyer volume - seller volume).
Higher λ → less liquid, prices move more per unit of order flow.

# Returns
- NamedTuple: lambda, intercept, r_squared, t_stat
"""
function kyle_lambda(price_changes::Vector{Float64},
                      signed_volumes::Vector{Float64})

    n = length(price_changes)
    @assert length(signed_volumes) == n "lengths must match"
    n < 5 && return (lambda=0.0, intercept=0.0, r_squared=0.0, t_stat=0.0)

    x = signed_volumes
    y = price_changes

    x_mean = mean(x)
    y_mean = mean(y)

    sxx = sum((x .- x_mean) .^ 2)
    sxy = sum((x .- x_mean) .* (y .- y_mean))

    lam = sxx > 0 ? sxy / sxx : 0.0
    intercept = y_mean - lam * x_mean

    y_hat = intercept .+ lam .* x
    ss_res = sum((y .- y_hat) .^ 2)
    ss_tot = sum((y .- y_mean) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0

    se = (n > 2 && sxx > 0) ? sqrt((ss_res/(n-2)) / sxx) : 1e-10
    t_stat = se > 0 ? lam / se : 0.0

    return (lambda=lam, intercept=intercept, r_squared=r2, t_stat=t_stat)
end

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

"""Normal CDF approximation (Abramowitz & Stegun 26.2.17)."""
function _normal_cdf(x::Float64)::Float64
    x >= 8.0  && return 1.0
    x <= -8.0 && return 0.0

    # Rational approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 +
           t * (-0.356563782 +
           t * (1.781477937 +
           t * (-1.821255978 +
           t * 1.330274429))))
    phi = exp(-0.5 * x^2) / sqrt(2π)
    cdf = 1.0 - phi * poly
    return x >= 0 ? cdf : 1.0 - cdf
end

end # module HighFrequencyAnalysis
