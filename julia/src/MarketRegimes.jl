"""
MarketRegimes — Advanced regime detection for crypto markets.

Covers:
  - Markov-switching GARCH (MS-GARCH): volatility regimes
  - Threshold VAR for multi-asset regime modelling
  - Momentum vs mean-reversion regime classifier
  - Regime persistence (survival analysis)
  - Transition matrix estimation with uncertainty
  - Out-of-sample regime prediction accuracy
  - Portfolio adaptation by regime (sizing, instrument selection)
  - Regime agreement score: when HMM, on-chain, and macro agree
"""
module MarketRegimes

using LinearAlgebra
using Statistics
using Random

export HMMRegime, fit_hmm_regimes, viterbi_regimes, hmm_filter
export MSGARCHModel, fit_ms_garch, forecast_ms_garch
export RegimeClassifier, train_regime_classifier, classify_regime
export regime_persistence, transition_matrix_uncertainty
export ThresholdVAR, fit_tvar, tvar_regimes
export momentum_mr_classifier, regime_signals
export portfolio_by_regime, regime_sizing_rule
export RegimeAgreementScore, compute_regime_agreement
export regime_prediction_accuracy, regime_backtest

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Hidden Markov Model Regime Detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    HMMRegime

Hidden Markov Model for regime detection.
K states, Gaussian emissions: x_t | s_t ~ N(μ_{s_t}, σ²_{s_t}).
Parameters: transition matrix A, means μ, variances σ², initial probs π.
"""
struct HMMRegime
    K::Int                     # number of states
    A::Matrix{Float64}         # K × K transition matrix
    mu::Vector{Float64}        # K emission means
    sigma::Vector{Float64}     # K emission std devs
    pi::Vector{Float64}        # K initial state probabilities
    ll::Float64                # log-likelihood at convergence
end

"""
    fit_hmm_regimes(observations, K; max_iter, tol, n_restarts, rng) -> HMMRegime

Fit a K-state Gaussian HMM to a time series via Baum-Welch (EM algorithm).
Multiple random restarts help avoid local optima.
"""
function fit_hmm_regimes(obs::Vector{Float64}, K::Int=2;
                           max_iter::Int=200, tol::Float64=1e-6,
                           n_restarts::Int=5,
                           rng::AbstractRNG=Random.default_rng())::HMMRegime
    n = length(obs)
    best_ll = -Inf
    best_hmm = nothing

    for restart in 1:n_restarts
        hmm = _random_hmm_init(obs, K, rng)
        hmm = _baum_welch(obs, hmm; max_iter=max_iter, tol=tol)
        if hmm.ll > best_ll
            best_ll = hmm.ll
            best_hmm = hmm
        end
    end

    return best_hmm
end

function _random_hmm_init(obs::Vector{Float64}, K::Int,
                            rng::AbstractRNG)::HMMRegime
    n = length(obs)
    # Initialise with random K-means-like assignment
    mu_init = sort(obs)[round.(Int, range(div(n,K), n - div(n,K), length=K))]
    sig_init = fill(std(obs), K)
    A_init   = (ones(K, K) .* 0.1 .+ I * 0.9 * (1 - 0.1*(K-1)))
    # Normalise rows
    for i in 1:K
        A_init[i, :] ./= sum(A_init[i, :])
    end
    pi_init = fill(1/K, K)
    return HMMRegime(K, A_init, mu_init, sig_init, pi_init, -Inf)
end

function _baum_welch(obs::Vector{Float64}, hmm::HMMRegime;
                      max_iter::Int=200, tol::Float64=1e-6)::HMMRegime
    n = length(obs)
    K = hmm.K
    mu    = copy(hmm.mu)
    sigma = copy(hmm.sigma)
    A     = copy(hmm.A)
    pi_v  = copy(hmm.pi)

    prev_ll = -Inf

    for iter in 1:max_iter
        # ── E-step: forward-backward ──
        alpha_mat = zeros(n, K)  # forward probs
        beta_mat  = ones(n, K)   # backward probs

        # Forward
        for k in 1:K
            alpha_mat[1, k] = pi_v[k] * _gauss_pdf(obs[1], mu[k], sigma[k])
        end
        scale = sum(alpha_mat[1, :])
        scale > 1e-300 && (alpha_mat[1, :] ./= scale)

        for t in 2:n
            for j in 1:K
                alpha_mat[t, j] = sum(alpha_mat[t-1, k] * A[k, j] for k in 1:K) *
                                   _gauss_pdf(obs[t], mu[j], sigma[j])
            end
            sc = sum(alpha_mat[t, :])
            sc > 1e-300 && (alpha_mat[t, :] ./= sc)
        end

        # Backward (scaled)
        for t in (n-1):-1:1
            for i in 1:K
                beta_mat[t, i] = sum(A[i, j] * _gauss_pdf(obs[t+1], mu[j], sigma[j]) *
                                      beta_mat[t+1, j] for j in 1:K)
            end
            sc = sum(beta_mat[t, :])
            sc > 1e-300 && (beta_mat[t, :] ./= sc)
        end

        # Gammas and xis
        gamma = alpha_mat .* beta_mat
        for t in 1:n
            s = sum(gamma[t, :])
            s > 1e-300 && (gamma[t, :] ./= s)
        end

        # ── M-step ──
        # Update pi
        pi_v = gamma[1, :]
        pi_v ./= max(sum(pi_v), 1e-300)

        # Update A
        for i in 1:K
            for j in 1:K
                num = sum(alpha_mat[t-1, i] * A[i, j] *
                          _gauss_pdf(obs[t], mu[j], sigma[j]) * beta_mat[t, j]
                          for t in 2:n)
                den = sum(gamma[t, i] for t in 1:(n-1))
                A[i, j] = max(num, 1e-10) / max(den, 1e-10)
            end
            A[i, :] ./= max(sum(A[i, :]), 1e-10)
        end

        # Update emissions
        for k in 1:K
            gk   = gamma[:, k]
            sum_gk = max(sum(gk), 1e-10)
            mu[k]    = dot(gk, obs) / sum_gk
            sigma[k] = sqrt(dot(gk, (obs .- mu[k]).^2) / sum_gk)
            sigma[k] = max(sigma[k], 1e-4)
        end

        # Log-likelihood
        ll = sum(log(max(sum(alpha_mat[n, :]), 1e-300)))

        abs(ll - prev_ll) < tol && break
        prev_ll = ll
    end

    # Sort states by mean (convention: state 1 = lowest mean)
    order = sortperm(mu)
    return HMMRegime(K, A[order, order], mu[order], sigma[order], pi_v[order], prev_ll)
end

function _gauss_pdf(x::Float64, mu::Float64, sigma::Float64)::Float64
    sigma < 1e-10 && return x == mu ? 1.0 : 0.0
    return exp(-0.5 * ((x - mu)/sigma)^2) / (sqrt(2π) * sigma)
end

"""
    viterbi_regimes(obs, hmm) -> Vector{Int}

Compute most likely state sequence via Viterbi algorithm.
Returns integer state labels 1..K.
"""
function viterbi_regimes(obs::Vector{Float64}, hmm::HMMRegime)::Vector{Int}
    n = length(obs)
    K = hmm.K

    log_delta = fill(-Inf, n, K)
    psi       = zeros(Int, n, K)

    for k in 1:K
        log_delta[1, k] = log(max(hmm.pi[k], 1e-300)) +
                           log(max(_gauss_pdf(obs[1], hmm.mu[k], hmm.sigma[k]), 1e-300))
    end

    for t in 2:n
        for j in 1:K
            best_val = -Inf
            best_k   = 1
            for k in 1:K
                val = log_delta[t-1, k] + log(max(hmm.A[k, j], 1e-300))
                if val > best_val
                    best_val = val
                    best_k   = k
                end
            end
            log_delta[t, j] = best_val + log(max(_gauss_pdf(obs[t], hmm.mu[j], hmm.sigma[j]), 1e-300))
            psi[t, j] = best_k
        end
    end

    # Backtrack
    states = zeros(Int, n)
    states[n] = argmax(log_delta[n, :])
    for t in (n-1):-1:1
        states[t] = psi[t+1, states[t+1]]
    end
    return states
end

"""
    hmm_filter(obs, hmm) -> Matrix{Float64}

Run the HMM forward filter, returning n × K matrix of filtered state probabilities
(P(s_t = k | obs_1:t)).
"""
function hmm_filter(obs::Vector{Float64}, hmm::HMMRegime)::Matrix{Float64}
    n = length(obs)
    K = hmm.K
    alpha = zeros(n, K)

    for k in 1:K
        alpha[1, k] = hmm.pi[k] * _gauss_pdf(obs[1], hmm.mu[k], hmm.sigma[k])
    end
    sc = sum(alpha[1, :])
    sc > 1e-300 && (alpha[1, :] ./= sc)

    for t in 2:n
        for j in 1:K
            alpha[t, j] = sum(alpha[t-1, k] * hmm.A[k, j] for k in 1:K) *
                           _gauss_pdf(obs[t], hmm.mu[j], hmm.sigma[j])
        end
        sc = sum(alpha[t, :])
        sc > 1e-300 && (alpha[t, :] ./= sc)
    end
    return alpha
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Markov-Switching GARCH
# ─────────────────────────────────────────────────────────────────────────────

"""
    MSGARCHModel

Markov-Switching GARCH model with K volatility regimes.
In each regime k: h_t = ω_k + α_k * ε²_{t-1} + β_k * h_{t-1}.
Regime transitions follow a K-state Markov chain.
"""
struct MSGARCHModel
    K::Int
    omega::Vector{Float64}
    alpha::Vector{Float64}
    beta::Vector{Float64}
    A::Matrix{Float64}        # transition matrix
    pi::Vector{Float64}       # stationary probabilities
    unconditional_vol::Vector{Float64}
end

"""
    fit_ms_garch(returns, K; max_iter) -> MSGARCHModel

Fit MS-GARCH with K regimes. Uses a simplified EM-style procedure:
1. Classify observations into K regimes by volatility clustering
2. Fit GARCH(1,1) in each regime
3. Estimate transition probabilities from regime sequence
"""
function fit_ms_garch(returns::Vector{Float64}, K::Int=2;
                       max_iter::Int=30)::MSGARCHModel
    n = length(returns)

    # Step 1: Initial regime assignment via rolling vol k-means
    window = 20
    roll_vol = zeros(n)
    for t in window:n
        roll_vol[t] = std(returns[(t-window+1):t])
    end
    roll_vol[1:(window-1)] .= mean(roll_vol[window:end])

    # Sort volatility and assign to K equal-frequency bins
    sorted_idx = sortperm(roll_vol)
    regime_assign = zeros(Int, n)
    bin_size = div(n, K)
    for k in 1:K
        lo = (k-1) * bin_size + 1
        hi = k == K ? n : k * bin_size
        regime_assign[sorted_idx[lo:hi]] .= k
    end

    # Step 2: Fit GARCH in each regime
    omegas = zeros(K)
    alphas = zeros(K)
    betas  = zeros(K)

    for k in 1:K
        idx_k = findall(==(k), regime_assign)
        length(idx_k) < 10 && continue
        r_k = returns[idx_k]
        # Simple MoM GARCH estimates
        var_k = var(r_k)
        # Rough persistence from ACF of r²
        r2 = r_k.^2
        acf1 = length(r2) > 2 ? cor(r2[1:end-1], r2[2:end]) : 0.3
        acf1 = clamp(acf1, 0.0, 0.99)

        alphas[k] = max(0.05, acf1 * 0.3)
        betas[k]  = max(0.10, acf1 * 0.7)
        alphas[k] + betas[k] >= 1 && (alphas[k] = 0.08; betas[k] = 0.88)
        omegas[k] = var_k * (1 - alphas[k] - betas[k])
        omegas[k] = max(omegas[k], 1e-7)
    end

    # Step 3: Estimate transition matrix
    A = ones(K, K) ./ K
    for t in 2:n
        i = regime_assign[t-1]
        j = regime_assign[t]
        A[i, j] += 1
    end
    for i in 1:K
        A[i, :] ./= max(sum(A[i, :]), 1e-10)
    end

    # Stationary distribution
    pi_stat = _stationary_dist(A)

    uncond_vol = sqrt.(omegas ./ max.(1 .- alphas .- betas, 0.001))

    return MSGARCHModel(K, omegas, alphas, betas, A, pi_stat, uncond_vol)
end

function _stationary_dist(A::Matrix{Float64})::Vector{Float64}
    K = size(A, 1)
    # Solve π = π A with sum(π) = 1
    # Equivalent: (A' - I) π = 0 with constraint
    M = (A' - I)
    M = vcat(M, ones(1, K))
    b = vcat(zeros(K), [1.0])
    try
        pi = M \ b
        return clamp.(pi, 0.0, 1.0)
    catch
        return fill(1/K, K)
    end
end

"""
    forecast_ms_garch(model, returns; h) -> Matrix{Float64}

Forecast h-step-ahead conditional variance for each regime.
Returns K × h matrix of variance forecasts.
"""
function forecast_ms_garch(model::MSGARCHModel, returns::Vector{Float64};
                             h::Int=5)::Matrix{Float64}
    K = model.K
    n = length(returns)
    forecasts = zeros(K, h)

    # Compute current conditional variance in each regime
    h0 = zeros(K)
    for k in 1:K
        # Use last 50 observations to initialise variance
        n_init = min(50, n)
        h_t = model.omega[k] / max(1 - model.alpha[k] - model.beta[k], 0.001)
        for t in (n - n_init + 1):n
            h_t = model.omega[k] + model.alpha[k] * returns[t]^2 + model.beta[k] * h_t
            h_t = max(h_t, 1e-12)
        end
        h0[k] = h_t
    end

    # Forecast forward
    for k in 1:K
        h_t = h0[k]
        unc = model.omega[k] / max(1 - model.alpha[k] - model.beta[k], 0.001)
        for step in 1:h
            persistence = model.alpha[k] + model.beta[k]
            # E[h_{t+step} | t] = unc + persistence^step * (h_t - unc)
            forecasts[k, step] = unc + persistence^step * (h_t - unc)
            forecasts[k, step] = max(forecasts[k, step], 1e-12)
        end
    end
    return forecasts
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Regime Persistence (Survival Analysis)
# ─────────────────────────────────────────────────────────────────────────────

"""
    regime_persistence(regime_sequence) -> NamedTuple

Analyse how long regimes persist.
Returns per-state durations, mean persistence, and Kaplan-Meier estimates.
"""
function regime_persistence(regime_sequence::Vector{Int})::NamedTuple
    n = length(regime_sequence)
    n < 2 && return (durations=Dict{Int,Vector{Int}}(), mean_duration=Dict{Int,Float64}())

    K = maximum(regime_sequence)
    durations = Dict(k => Int[] for k in 1:K)

    current_state = regime_sequence[1]
    current_dur   = 1

    for t in 2:n
        if regime_sequence[t] == current_state
            current_dur += 1
        else
            push!(durations[current_state], current_dur)
            current_state = regime_sequence[t]
            current_dur   = 1
        end
    end
    push!(durations[current_state], current_dur)

    mean_dur = Dict(k => isempty(durations[k]) ? 0.0 : mean(Float64.(durations[k]))
                    for k in 1:K)
    median_dur = Dict(k => isempty(durations[k]) ? 0.0 :
                      Float64(sort(durations[k])[div(end,2)+1])
                      for k in 1:K)

    # Kaplan-Meier survival function for each regime
    km_estimates = Dict{Int,Vector{Float64}}()
    for k in 1:K
        durs_k = sort(durations[k])
        isempty(durs_k) && continue
        max_dur = maximum(durs_k)
        survival = ones(max_dur)
        at_risk  = length(durs_k)
        for d in 1:max_dur
            deaths = sum(durs_k .== d)
            survival[d] = d == 1 ? 1.0 - deaths/max(at_risk, 1) :
                           survival[d-1] * (1 - deaths/max(at_risk, 1))
            at_risk -= deaths
        end
        km_estimates[k] = survival
    end

    return (durations=durations, mean_duration=mean_dur, median_duration=median_dur,
            km_survival=km_estimates)
end

"""
    transition_matrix_uncertainty(regime_sequence, K; n_bootstrap) -> NamedTuple

Estimate transition matrix with bootstrap confidence intervals.
"""
function transition_matrix_uncertainty(regime_sequence::Vector{Int}, K::Int;
                                        n_bootstrap::Int=200,
                                        rng::AbstractRNG=Random.default_rng())::NamedTuple
    n = length(regime_sequence)

    # Point estimate
    A_hat = zeros(K, K)
    for t in 2:n
        i = clamp(regime_sequence[t-1], 1, K)
        j = clamp(regime_sequence[t],   1, K)
        A_hat[i, j] += 1
    end
    for i in 1:K
        A_hat[i, :] ./= max(sum(A_hat[i, :]), 1e-10)
    end

    # Bootstrap CIs
    A_boot = zeros(n_bootstrap, K, K)
    for b in 1:n_bootstrap
        boot_idx = [1; rand(rng, 2:n, n-1)]
        boot_seq = regime_sequence[boot_idx]
        A_b = zeros(K, K)
        for t in 2:n
            i = clamp(boot_seq[t-1], 1, K)
            j = clamp(boot_seq[t],   1, K)
            A_b[i, j] += 1
        end
        for i in 1:K
            A_b[i, :] ./= max(sum(A_b[i, :]), 1e-10)
        end
        A_boot[b, :, :] = A_b
    end

    A_lo  = [quantile_v(A_boot[:, i, j], 0.025) for i in 1:K, j in 1:K]
    A_hi  = [quantile_v(A_boot[:, i, j], 0.975) for i in 1:K, j in 1:K]
    A_std = [std(A_boot[:, i, j]) for i in 1:K, j in 1:K]

    return (A=A_hat, A_lower=A_lo, A_upper=A_hi, A_std=A_std)
end

function quantile_v(x::Vector{Float64}, p::Float64)::Float64
    s = sort(x)
    n = length(s)
    idx = clamp(p * n, 1.0, Float64(n))
    lo = floor(Int, idx)
    hi = min(ceil(Int, idx), n)
    lo == hi && return s[lo]
    return s[lo] + (idx - lo) * (s[hi] - s[lo])
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Momentum vs Mean-Reversion Regime Classifier
# ─────────────────────────────────────────────────────────────────────────────

"""
    RegimeClassifier

Momentum vs mean-reversion classifier based on rolling statistics.
Uses Hurst exponent, autocorrelation, and variance ratio as features.
"""
struct RegimeClassifier
    hurst_threshold::Float64    # H < threshold → MR, H > threshold → momentum
    acf_threshold::Float64      # ACF > threshold → momentum (positive autocorr)
    vr_threshold::Float64       # VR > threshold → momentum
    window::Int
end

"""
    train_regime_classifier(; hurst_threshold, acf_threshold, vr_threshold, window) -> RegimeClassifier

Create a regime classifier with specified thresholds.
Thresholds can be estimated from historical data.
"""
function train_regime_classifier(; hurst_threshold::Float64=0.52,
                                   acf_threshold::Float64=0.05,
                                   vr_threshold::Float64=1.05,
                                   window::Int=60)::RegimeClassifier
    return RegimeClassifier(hurst_threshold, acf_threshold, vr_threshold, window)
end

"""
    classify_regime(returns, clf; t) -> Symbol

Classify the current regime at time t.
Returns :Momentum, :MeanReversion, or :Random.
"""
function classify_regime(returns::Vector{Float64}, clf::RegimeClassifier;
                           t::Int=length(returns))::Symbol
    t < clf.window + 1 && return :Random
    w = returns[max(1, t-clf.window+1):t]

    # Hurst exponent estimate
    H = _hurst_rs(w)

    # First-order autocorrelation
    acf1 = length(w) > 2 ? cor(w[1:end-1], w[2:end]) : 0.0

    # Variance ratio at lag 5
    vr5 = _variance_ratio(w, 5)

    # Vote-based classification
    momentum_votes = 0
    mr_votes       = 0

    H > clf.hurst_threshold  && (momentum_votes += 1)
    H < 1 - clf.hurst_threshold && (mr_votes += 1)
    acf1 > clf.acf_threshold  && (momentum_votes += 1)
    acf1 < -clf.acf_threshold && (mr_votes += 1)
    vr5 > clf.vr_threshold    && (momentum_votes += 1)
    vr5 < 1/clf.vr_threshold  && (mr_votes += 1)

    if momentum_votes >= 2
        return :Momentum
    elseif mr_votes >= 2
        return :MeanReversion
    else
        return :Random
    end
end

function _hurst_rs(x::Vector{Float64})::Float64
    n = length(x)
    n < 20 && return 0.5
    sizes = [8, 16, 32]
    sizes = filter(s -> s < n/2, sizes)
    isempty(sizes) && return 0.5

    log_rs_vals   = Float64[]
    log_size_vals = Float64[]

    for m in sizes
        rs_block = Float64[]
        n_blocks = div(n, m)
        for b in 0:(n_blocks-1)
            block = x[(b*m+1):((b+1)*m)]
            dev   = cumsum(block .- mean(block))
            r     = maximum(dev) - minimum(dev)
            s     = std(block)
            s < 1e-10 && continue
            push!(rs_block, r/s)
        end
        isempty(rs_block) && continue
        push!(log_rs_vals,   log(mean(rs_block)))
        push!(log_size_vals, log(Float64(m)))
    end

    length(log_size_vals) < 2 && return 0.5
    h_bar = mean(log_size_vals)
    r_bar = mean(log_rs_vals)
    slope = sum((log_size_vals .- h_bar) .* (log_rs_vals .- r_bar)) /
            sum((log_size_vals .- h_bar).^2)
    return clamp(slope, 0.0, 1.0)
end

function _variance_ratio(x::Vector{Float64}, q::Int)::Float64
    n = length(x)
    n < 2*q && return 1.0
    mu   = mean(x)
    var1 = mean((x .- mu).^2)
    var1 < 1e-12 && return 1.0
    ret_q = [sum(x[t:(t+q-1)]) for t in 1:(n-q+1)]
    var_q = mean((ret_q .- q*mu).^2)
    return var_q / (q * var1)
end

"""
    regime_signals(returns; window) -> NamedTuple

Compute rolling regime indicators over a window.
Returns time series of H, ACF1, VR5, and regime classification.
"""
function regime_signals(returns::Vector{Float64}; window::Int=60)::NamedTuple
    n = length(returns)
    H_series   = fill(NaN, n)
    acf_series = fill(NaN, n)
    vr_series  = fill(NaN, n)
    regime_seq = fill(:Unknown, n)

    clf = train_regime_classifier(; window=window)

    for t in window:n
        w = returns[(t-window+1):t]
        H_series[t]   = _hurst_rs(w)
        acf_series[t] = length(w) > 2 ? cor(w[1:end-1], w[2:end]) : 0.0
        vr_series[t]  = _variance_ratio(w, 5)
        regime_seq[t] = classify_regime(returns, clf; t=t)
    end

    return (H=H_series, acf1=acf_series, vr5=vr_series, regime=regime_seq)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Threshold VAR
# ─────────────────────────────────────────────────────────────────────────────

"""
    ThresholdVAR

Threshold Vector Autoregression model.
In regime 1 (threshold variable < γ): r_t = A1 * r_{t-1} + ε
In regime 2 (threshold variable ≥ γ): r_t = A2 * r_{t-1} + ε
"""
struct ThresholdVAR
    A1::Matrix{Float64}   # VAR coefficients in low regime
    A2::Matrix{Float64}   # VAR coefficients in high regime
    gamma::Float64        # threshold value
    threshold_var::Int    # index of the threshold variable
    n_assets::Int
end

"""
    fit_tvar(returns_matrix, threshold_variable; gamma_quantile, lag) -> ThresholdVAR

Fit a 2-regime Threshold VAR.
threshold_variable: column index used as the switching variable.
gamma_quantile: threshold set at this quantile of the threshold variable.
"""
function fit_tvar(R::Matrix{Float64}, threshold_var::Int=1;
                   gamma_quantile::Float64=0.50, lag::Int=1)::ThresholdVAR
    n, d = size(R)
    n < lag + 10 && error("Insufficient data for TVAR estimation")

    # Threshold
    tv = R[:, threshold_var]
    gamma = quantile_v(sort(tv), gamma_quantile)

    # Build lagged data
    Y = R[(lag+1):end, :]
    X = R[1:(end-lag), :]
    n_eff = size(Y, 1)

    # Regime assignment
    tv_lagged = tv[1:(end-lag)]
    mask1 = tv_lagged .< gamma
    mask2 = .!mask1

    # Fit OLS VAR in each regime
    function ols_var(Y_reg, X_reg)
        n_r, d_r = size(Y_reg)
        n_r < d_r + 2 && return zeros(d_r, d_r)
        X_aug = hcat(ones(n_r), X_reg)
        return (X_aug' * X_aug) \ (X_aug' * Y_reg)
    end

    Y1 = Y[mask1, :]
    X1 = X[mask1, :]
    Y2 = Y[mask2, :]
    X2 = X[mask2, :]

    coef1 = size(Y1, 1) > d + 2 ? ols_var(Y1, X1) : zeros(d+1, d)
    coef2 = size(Y2, 1) > d + 2 ? ols_var(Y2, X2) : zeros(d+1, d)

    # Extract slope matrices (excluding intercept row)
    A1 = coef1[2:end, :]'
    A2 = coef2[2:end, :]'

    return ThresholdVAR(A1, A2, gamma, threshold_var, d)
end

"""
    tvar_regimes(R, model; lag) -> Vector{Int}

Classify each observation in R into regime 1 or 2 using the TVAR model.
"""
function tvar_regimes(R::Matrix{Float64}, model::ThresholdVAR; lag::Int=1)::Vector{Int}
    n = size(R, 1)
    tv_lagged = R[1:(end-lag), model.threshold_var]
    regimes   = [v < model.gamma ? 1 : 2 for v in tv_lagged]
    # Pad beginning
    return vcat(fill(1, lag), regimes)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Portfolio Adaptation by Regime
# ─────────────────────────────────────────────────────────────────────────────

"""
    portfolio_by_regime(current_regime, base_weights; regime_multipliers) -> Vector{Float64}

Adjust portfolio weights based on the current market regime.
regime_multipliers: Dict mapping regime symbol to per-asset scaling vector.
"""
function portfolio_by_regime(current_regime::Symbol,
                               base_weights::Vector{Float64};
                               regime_multipliers::Dict{Symbol,Vector{Float64}}=Dict())::Vector{Float64}
    haskey(regime_multipliers, current_regime) || return base_weights
    mults = regime_multipliers[current_regime]
    length(mults) != length(base_weights) && return base_weights
    new_w = base_weights .* mults
    total = sum(abs.(new_w))
    total < 1e-10 && return base_weights
    return new_w ./ total  # normalise to sum-to-one
end

"""
    regime_sizing_rule(regime, base_size; rules) -> Float64

Apply a sizing rule based on the current regime.
rules: Dict{Symbol, Float64} mapping regime → size multiplier.
"""
function regime_sizing_rule(regime::Symbol, base_size::Float64;
                              rules::Dict{Symbol,Float64}=Dict(
                                  :Bull   => 1.0,
                                  :Bear   => 0.5,
                                  :Stress => 0.25,
                                  :Neutral => 0.75))::Float64
    mult = get(rules, regime, 1.0)
    return base_size * mult
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Regime Agreement Score
# ─────────────────────────────────────────────────────────────────────────────

"""
    RegimeAgreementScore

Score measuring consensus across multiple regime models.
A high agreement score means multiple independent methods agree on regime,
which historically corresponds to stronger and more reliable signals.
"""
struct RegimeAgreementScore
    models::Vector{Symbol}         # names of models included
    weights::Vector{Float64}       # model weights
    score::Float64                 # current agreement score [0, 1]
    majority_regime::Symbol        # the agreed-upon regime
    confidence::Float64            # confidence in the majority
end

"""
    compute_regime_agreement(model_regimes, model_weights, model_names) -> RegimeAgreementScore

Compute a weighted agreement score across multiple regime signals.
model_regimes: Vector of regime predictions (Symbols) from each model.
model_weights: Vector of weights for each model.
"""
function compute_regime_agreement(model_regimes::Vector{Symbol},
                                    model_weights::Vector{Float64},
                                    model_names::Vector{Symbol})::RegimeAgreementScore
    n_models = length(model_regimes)
    @assert n_models == length(model_weights)

    # Normalise weights
    w = model_weights ./ max(sum(model_weights), 1e-10)

    # Vote for majority regime
    regime_votes = Dict{Symbol,Float64}()
    for (i, r) in enumerate(model_regimes)
        regime_votes[r] = get(regime_votes, r, 0.0) + w[i]
    end

    majority_regime = argmax(regime_votes)
    majority_weight = regime_votes[majority_regime]

    # Agreement score: weighted fraction agreeing with majority
    confidence = majority_weight  # [0, 1]

    return RegimeAgreementScore(
        model_names, w, majority_weight, majority_regime, confidence)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Out-of-Sample Regime Prediction Accuracy
# ─────────────────────────────────────────────────────────────────────────────

"""
    regime_prediction_accuracy(predicted_regimes, true_regimes) -> NamedTuple

Compute accuracy metrics for regime prediction.
true_regimes: ground-truth regime sequence.
predicted_regimes: model-predicted regime sequence (1-step-ahead).
"""
function regime_prediction_accuracy(predicted::Vector{Int},
                                     true_regimes::Vector{Int})::NamedTuple
    n = min(length(predicted), length(true_regimes))
    correct = sum(predicted[1:n] .== true_regimes[1:n])
    accuracy = correct / n

    K = maximum(vcat(predicted, true_regimes))
    # Per-state accuracy
    per_state = Dict{Int,Float64}()
    for k in 1:K
        idx_k = findall(==(k), true_regimes[1:n])
        isempty(idx_k) && continue
        per_state[k] = mean(predicted[idx_k] .== k)
    end

    # Confusion matrix
    C = zeros(K, K)
    for i in 1:n
        ti = clamp(true_regimes[i], 1, K)
        pi = clamp(predicted[i], 1, K)
        C[ti, pi] += 1
    end
    for k in 1:K
        C[k, :] ./= max(sum(C[k, :]), 1e-10)
    end

    return (accuracy=accuracy, per_state_accuracy=per_state, confusion_matrix=C, n=n)
end

"""
    regime_backtest(returns, regimes, regime_strategies; cost) -> NamedTuple

Backtest a regime-conditional strategy.
regime_strategies: Dict{Int, Float64} = regime → position size.
"""
function regime_backtest(returns::Vector{Float64}, regimes::Vector{Int},
                           regime_strategies::Dict{Int,Float64};
                           cost::Float64=0.001)::NamedTuple
    n = min(length(returns), length(regimes))
    equity = ones(n + 1)
    prev_size = 0.0
    total_tc  = 0.0
    sizes     = Float64[]

    for t in 1:n
        r_t = clamp(regimes[t], 1, maximum(keys(regime_strategies)))
        f   = get(regime_strategies, r_t, 0.0)
        tc  = abs(f - prev_size) * cost
        net_ret = f * returns[t] - tc
        equity[t+1] = equity[t] * exp(net_ret)
        total_tc   += tc
        prev_size   = f
        push!(sizes, f)
    end

    rets_strat = diff(log.(equity))
    m  = mean(rets_strat)
    s  = std(rets_strat)
    sr = s > 1e-10 ? m / s * sqrt(252) : 0.0

    # Max drawdown
    mdd = 0.0
    peak = equity[1]
    for e in equity
        e > peak && (peak = e)
        dd = (peak - e) / peak
        dd > mdd && (mdd = dd)
    end

    cagr = (equity[end]^(252/n) - 1) * 100

    return (equity=equity, sharpe=sr, max_dd=mdd, cagr=cagr,
            total_tc=total_tc, avg_size=mean(sizes))
end

end  # module MarketRegimes
