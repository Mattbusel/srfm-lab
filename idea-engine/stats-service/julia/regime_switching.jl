"""
regime_switching.jl

Hamilton Markov Switching Model for crypto market regime detection.
  - 2-state and 3-state hidden Markov models (HMM)
  - EM algorithm (Baum-Welch) for parameter estimation
  - Viterbi algorithm for most-likely state sequence
  - Forward (filtered) and forward-backward (smoothed) state probabilities
  - Application to BTC returns: BULL / BEAR / CHOPPY regime identification
  - Regime forecasting: next-state probability distribution
  - Cross-model comparison: HMM vs on-chain vs macro regimes
"""

using Statistics
using LinearAlgebra
using JSON3
using Dates

# ─── Data Structures ─────────────────────────────────────────────────────────

"""Parameters for a single regime (Gaussian emission distribution)."""
struct RegimeParams
    mean::Float64
    std::Float64
end

"""Full HMM parameter set."""
mutable struct HMMParams
    n_states::Int
    transition::Matrix{Float64}   # (n_states × n_states) row-stochastic
    emission::Vector{RegimeParams}
    initial::Vector{Float64}      # initial state distribution (sums to 1)
end

"""Result from HMM fitting."""
struct HMMResult
    params::HMMParams
    log_likelihood::Float64
    n_iterations::Int
    converged::Bool
    filtered_probs::Matrix{Float64}    # T × n_states
    smoothed_probs::Matrix{Float64}    # T × n_states
    viterbi_states::Vector{Int}        # most likely state sequence
    regime_labels::Vector{String}      # human-readable labels
    forecast::Vector{Float64}          # probability over next state
end

# ─── Gaussian PDF ─────────────────────────────────────────────────────────────

"""Log-density of Normal(μ, σ) evaluated at x."""
@inline function lognorm(x::Float64, μ::Float64, σ::Float64)::Float64
    σ_safe = max(σ, 1e-8)
    -0.5 * log(2π) - log(σ_safe) - 0.5 * ((x - μ) / σ_safe)^2
end

@inline function norm_pdf(x::Float64, μ::Float64, σ::Float64)::Float64
    exp(lognorm(x, μ, σ))
end

# ─── HMM Initialization ───────────────────────────────────────────────────────

"""
    init_hmm_params(returns, n_states; seed_method=:kmeans_like)

Initialize HMM parameters using a simple quantile-based method.
States are ordered from low mean (bear) to high mean (bull).
"""
function init_hmm_params(returns::Vector{Float64}, n_states::Int)::HMMParams
    n = length(returns)
    sorted = sort(returns)
    chunk_size = n ÷ n_states

    emission = RegimeParams[]
    for k in 1:n_states
        start_idx = (k-1) * chunk_size + 1
        end_idx = k == n_states ? n : k * chunk_size
        chunk = sorted[start_idx:end_idx]
        push!(emission, RegimeParams(mean(chunk), max(std(chunk), 1e-4)))
    end

    # Transition: slight persistence (0.9 stay, spread remainder equally)
    persist = 0.88
    T = fill((1.0 - persist) / (n_states - 1), n_states, n_states)
    for k in 1:n_states
        T[k, k] = persist
    end

    init_dist = fill(1.0 / n_states, n_states)

    return HMMParams(n_states, T, emission, init_dist)
end

# ─── Forward Algorithm (Filtering) ────────────────────────────────────────────

"""
    forward_pass(returns, params)

Hamilton filter (forward algorithm) for HMM.
Computes filtered state probabilities P(s_t | y_{1:t}) and log-likelihood.

Returns (alpha, log_likelihood) where:
  alpha[t, k] = P(s_t=k, y_{1:t}) (unnormalized)
  scaled for numerical stability using the log-sum-exp trick.
"""
function forward_pass(
    returns::Vector{Float64},
    params::HMMParams
)::Tuple{Matrix{Float64}, Float64}
    T = length(returns)
    K = params.n_states
    alpha = zeros(T, K)     # scaled forward probabilities
    log_ll = 0.0

    # t = 1
    for k in 1:K
        alpha[1, k] = params.initial[k] *
            norm_pdf(returns[1], params.emission[k].mean, params.emission[k].std)
    end
    c1 = sum(alpha[1, :])
    c1 < 1e-300 && (c1 = 1e-300)
    alpha[1, :] ./= c1
    log_ll += log(c1)

    # t = 2..T
    for t in 2:T
        for k in 1:K
            alpha[t, k] = sum(alpha[t-1, j] * params.transition[j, k] for j in 1:K) *
                norm_pdf(returns[t], params.emission[k].mean, params.emission[k].std)
        end
        ct = sum(alpha[t, :])
        ct < 1e-300 && (ct = 1e-300)
        alpha[t, :] ./= ct
        log_ll += log(ct)
    end

    return (alpha, log_ll)
end

# ─── Backward Algorithm ───────────────────────────────────────────────────────

"""
    backward_pass(returns, params, alpha_scaling)

Compute scaled backward probabilities β_t(k) for the forward-backward algorithm.
"""
function backward_pass(
    returns::Vector{Float64},
    params::HMMParams
)::Matrix{Float64}
    T = length(returns)
    K = params.n_states
    beta = ones(T, K)

    for t in T-1:-1:1
        for j in 1:K
            beta[t, j] = sum(
                params.transition[j, k] *
                norm_pdf(returns[t+1], params.emission[k].mean, params.emission[k].std) *
                beta[t+1, k]
                for k in 1:K
            )
        end
        ct = maximum(beta[t, :])
        ct < 1e-300 && (ct = 1e-300)
        beta[t, :] ./= ct
    end

    return beta
end

# ─── Forward-Backward (Smoothing) ─────────────────────────────────────────────

"""
    forward_backward(returns, params)

Compute smoothed state probabilities P(s_t=k | y_{1:T}) for all t.
Also compute the pair marginals xi[t, j, k] = P(s_t=j, s_{t+1}=k | y_{1:T}).

Returns (gamma, xi) where gamma is T×K smoothed probs.
"""
function forward_backward(
    returns::Vector{Float64},
    params::HMMParams
)::Tuple{Matrix{Float64}, Array{Float64, 3}}
    T = length(returns)
    K = params.n_states

    alpha, _ = forward_pass(returns, params)
    beta = backward_pass(returns, params)

    # Smoothed single marginals: γ_t(k) ∝ α_t(k) · β_t(k)
    gamma = alpha .* beta
    row_sums = sum(gamma, dims=2)
    gamma ./= max.(row_sums, 1e-300)

    # Pair marginals: ξ_t(j,k) ∝ α_t(j) · A_{jk} · b_k(y_{t+1}) · β_{t+1}(k)
    xi = zeros(T-1, K, K)
    for t in 1:T-1
        for j in 1:K, k in 1:K
            xi[t, j, k] = alpha[t, j] *
                params.transition[j, k] *
                norm_pdf(returns[t+1], params.emission[k].mean, params.emission[k].std) *
                beta[t+1, k]
        end
        s = sum(xi[t, :, :])
        s < 1e-300 && (s = 1e-300)
        xi[t, :, :] ./= s
    end

    return (gamma, xi)
end

# ─── EM Algorithm (Baum-Welch) ────────────────────────────────────────────────

"""
    em_step!(params, returns, gamma, xi)

Single M-step of the Baum-Welch EM algorithm. Updates params in-place.
"""
function em_step!(params::HMMParams, returns::Vector{Float64}, gamma::Matrix{Float64}, xi::Array{Float64, 3})
    T = length(returns)
    K = params.n_states

    # Update initial distribution
    params.initial .= gamma[1, :]

    # Update transition matrix
    for j in 1:K
        denom = sum(xi[:, j, :])
        denom < 1e-300 && (denom = 1e-300)
        for k in 1:K
            params.transition[j, k] = sum(xi[:, j, k]) / denom
        end
    end

    # Update emission parameters (Gaussian)
    for k in 1:K
        gamma_k_sum = sum(gamma[:, k])
        gamma_k_sum < 1e-300 && (gamma_k_sum = 1e-300)
        new_mean = sum(gamma[t, k] * returns[t] for t in 1:T) / gamma_k_sum
        new_var = sum(gamma[t, k] * (returns[t] - new_mean)^2 for t in 1:T) / gamma_k_sum
        params.emission[k] = RegimeParams(new_mean, max(sqrt(new_var), 1e-4))
    end

    # Ensure transition rows sum to 1 (numerical cleanup)
    for j in 1:K
        row_sum = sum(params.transition[j, :])
        row_sum < 1e-300 && (row_sum = 1.0)
        params.transition[j, :] ./= row_sum
    end
end

"""
    fit_hmm(returns, n_states; max_iter=200, tol=1e-5)

Fit an HMM to the return series using the Baum-Welch EM algorithm.
Returns fitted HMMParams and log-likelihood history.
"""
function fit_hmm(
    returns::Vector{Float64},
    n_states::Int;
    max_iter::Int = 200,
    tol::Float64 = 1e-5
)::Tuple{HMMParams, Vector{Float64}, Bool}
    params = init_hmm_params(returns, n_states)
    ll_history = Float64[]
    converged = false

    for iter in 1:max_iter
        # E-step
        gamma, xi = forward_backward(returns, params)
        _, ll = forward_pass(returns, params)
        push!(ll_history, ll)

        # Convergence check
        if length(ll_history) >= 2
            improvement = abs(ll_history[end] - ll_history[end-1])
            if improvement < tol
                converged = true
                break
            end
        end

        # M-step
        em_step!(params, returns, gamma, xi)
    end

    return (params, ll_history, converged)
end

# ─── Viterbi Algorithm ────────────────────────────────────────────────────────

"""
    viterbi(returns, params)

Find the most likely state sequence using the Viterbi algorithm.
Returns a vector of state indices (1-indexed).
"""
function viterbi(returns::Vector{Float64}, params::HMMParams)::Vector{Int}
    T = length(returns)
    K = params.n_states

    # Log probabilities for numerical stability
    log_trans = log.(max.(params.transition, 1e-300))
    log_init  = log.(max.(params.initial, 1e-300))

    delta = fill(-Inf, T, K)
    psi   = zeros(Int, T, K)

    # Initialization
    for k in 1:K
        delta[1, k] = log_init[k] +
            lognorm(returns[1], params.emission[k].mean, params.emission[k].std)
    end

    # Recursion
    for t in 2:T
        for k in 1:K
            # Find best predecessor
            scores = [delta[t-1, j] + log_trans[j, k] for j in 1:K]
            best_j = argmax(scores)
            delta[t, k] = scores[best_j] +
                lognorm(returns[t], params.emission[k].mean, params.emission[k].std)
            psi[t, k] = best_j
        end
    end

    # Backtrack
    states = zeros(Int, T)
    states[T] = argmax(delta[T, :])
    for t in T-1:-1:1
        states[t] = psi[t+1, states[t+1]]
    end

    return states
end

# ─── Regime Labeling ─────────────────────────────────────────────────────────

"""
    label_regimes(params, n_states)

Assign human-readable labels to states based on their mean returns.
For n_states=2: BULL / BEAR
For n_states=3: BULL / CHOPPY / BEAR
"""
function label_regimes(params::HMMParams)::Vector{String}
    K = params.n_states
    means = [params.emission[k].mean for k in 1:K]
    ranked = sortperm(means)   # ascending: lowest mean = most bearish

    if K == 2
        labels = Vector{String}(undef, K)
        labels[ranked[1]] = "BEAR"
        labels[ranked[2]] = "BULL"
        return labels
    elseif K == 3
        labels = Vector{String}(undef, K)
        labels[ranked[1]] = "BEAR"
        labels[ranked[2]] = "CHOPPY"
        labels[ranked[3]] = "BULL"
        return labels
    else
        return ["REGIME_$k" for k in 1:K]
    end
end

# ─── Regime Forecasting ───────────────────────────────────────────────────────

"""
    forecast_next_state(current_probs, transition)

Compute the probability distribution over the next state given current
filtered state probabilities and the transition matrix.

current_probs: P(s_t=k | y_{1:t}) — the filtered distribution at time t
Returns: P(s_{t+1}=k | y_{1:t}) for each state k
"""
function forecast_next_state(
    current_probs::Vector{Float64},
    transition::Matrix{Float64}
)::Vector{Float64}
    next = transition' * current_probs
    s = sum(next)
    s < 1e-10 && return fill(1.0/length(next), length(next))
    return next ./ s
end

"""
    forecast_n_steps(current_probs, transition, n)

Forecast state probabilities n steps ahead.
"""
function forecast_n_steps(
    current_probs::Vector{Float64},
    transition::Matrix{Float64},
    n::Int
)::Matrix{Float64}
    K = length(current_probs)
    out = zeros(n, K)
    probs = copy(current_probs)
    for i in 1:n
        probs = forecast_next_state(probs, transition)
        out[i, :] = probs
    end
    return out
end

"""
    stationary_distribution(transition; tol=1e-10, max_iter=1000)

Compute the stationary distribution π of a Markov chain by power iteration:
π = π · A^∞
"""
function stationary_distribution(
    transition::Matrix{Float64};
    tol::Float64=1e-10,
    max_iter::Int=1000
)::Vector{Float64}
    K = size(transition, 1)
    pi = fill(1.0/K, K)
    for _ in 1:max_iter
        new_pi = transition' * pi
        new_pi ./= sum(new_pi)
        maximum(abs.(new_pi .- pi)) < tol && return new_pi
        pi = new_pi
    end
    return pi
end

# ─── Expected Duration ────────────────────────────────────────────────────────

"""
    expected_durations(transition)

For each regime, compute the expected number of periods before switching.
For a Markov chain: E[duration in state k] = 1 / (1 - A_{kk})
"""
function expected_durations(transition::Matrix{Float64})::Vector{Float64}
    K = size(transition, 1)
    [1.0 / max(1.0 - transition[k, k], 1e-10) for k in 1:K]
end

# ─── Full Fit Pipeline ────────────────────────────────────────────────────────

"""
    fit_regime_model(returns, n_states; max_iter=200, tol=1e-5)

Fit a complete HMM regime model and return an HMMResult with all outputs.
"""
function fit_regime_model(
    returns::Vector{Float64},
    n_states::Int;
    max_iter::Int = 200,
    tol::Float64 = 1e-5
)::HMMResult
    T = length(returns)

    params, ll_history, converged = fit_hmm(returns, n_states; max_iter, tol)

    # Compute all state probabilities
    alpha, log_ll = forward_pass(returns, params)
    gamma, _ = forward_backward(returns, params)
    viterbi_states = viterbi(returns, params)
    labels = label_regimes(params)

    # Forecast from last filtered state
    last_filtered = alpha[end, :]
    forecast = forecast_next_state(last_filtered, params.transition)

    return HMMResult(
        params,
        log_ll,
        length(ll_history),
        converged,
        alpha,       # filtered
        gamma,       # smoothed
        viterbi_states,
        labels,
        forecast,
    )
end

# ─── BIC / AIC Model Selection ────────────────────────────────────────────────

"""
    hmm_bic(log_ll, n_obs, n_states)

Compute Bayesian Information Criterion for model selection.
Number of free parameters for K-state Gaussian HMM:
  - K*(K-1) transition parameters (each row has K-1 free)
  - 2*K emission parameters (mean + std per state)
  - (K-1) initial distribution parameters
"""
function hmm_bic(log_ll::Float64, n_obs::Int, n_states::Int)::Float64
    K = n_states
    n_params = K*(K-1) + 2*K + (K-1)
    return -2 * log_ll + n_params * log(n_obs)
end

function hmm_aic(log_ll::Float64, n_states::Int)::Float64
    K = n_states
    n_params = K*(K-1) + 2*K + (K-1)
    return -2 * log_ll + 2 * n_params
end

"""
    select_n_states(returns; candidates=[2,3,4], criterion=:bic)

Select the optimal number of states using BIC or AIC.
"""
function select_n_states(
    returns::Vector{Float64};
    candidates::Vector{Int} = [2, 3, 4],
    criterion::Symbol = :bic
)::Int
    T = length(returns)
    best_score = Inf
    best_k = candidates[1]

    for K in candidates
        _, ll_history, _ = fit_hmm(returns, K)
        _, ll = forward_pass(returns, init_hmm_params(returns, K))
        score = criterion == :bic ? hmm_bic(ll_history[end], T, K) : hmm_aic(ll_history[end], K)
        if score < best_score
            best_score = score
            best_k = K
        end
    end
    return best_k
end

# ─── Regime Agreement Analysis ────────────────────────────────────────────────

"""
    regime_agreement(hmm_states, onchain_states, macro_states, n_states)

Compare three regime series and compute agreement statistics.

Returns a NamedTuple with:
  - pairwise_agreement: 3 agreement rates (hmm vs onchain, hmm vs macro, onchain vs macro)
  - confusion matrices
  - cohen_kappa: Cohen's kappa for each pair
"""
function regime_agreement(
    hmm_states::Vector{Int},
    onchain_states::Vector{Int},
    macro_states::Vector{Int}
)::NamedTuple
    T = length(hmm_states)
    length(onchain_states) == T || throw(ArgumentError("lengths must match"))
    length(macro_states) == T || throw(ArgumentError("lengths must match"))

    function agreement_rate(a, b)
        sum(a[i] == b[i] for i in 1:T) / T
    end

    function cohen_kappa(a, b)
        n = length(a)
        states = sort(unique(vcat(a, b)))
        K = length(states)
        # Observed agreement
        po = sum(a[i] == b[i] for i in 1:n) / n
        # Expected agreement
        pe = sum((sum(a .== s) / n) * (sum(b .== s) / n) for s in states)
        pe >= 1 - 1e-10 && return 1.0
        (po - pe) / (1.0 - pe)
    end

    return (
        hmm_vs_onchain_agreement = agreement_rate(hmm_states, onchain_states),
        hmm_vs_macro_agreement   = agreement_rate(hmm_states, macro_states),
        onchain_vs_macro_agreement = agreement_rate(onchain_states, macro_states),
        hmm_vs_onchain_kappa     = cohen_kappa(hmm_states, onchain_states),
        hmm_vs_macro_kappa       = cohen_kappa(hmm_states, macro_states),
        onchain_vs_macro_kappa   = cohen_kappa(onchain_states, macro_states),
    )
end

# ─── Result Serialization ─────────────────────────────────────────────────────

"""
    result_to_dict(result, returns, dates)

Convert HMMResult to a Dict suitable for JSON export.
dates: optional vector of date strings aligned with returns.
"""
function result_to_dict(
    result::HMMResult,
    returns::Vector{Float64},
    dates::Union{Nothing, Vector{String}} = nothing
)::Dict
    T = length(returns)
    K = result.params.n_states

    durations = expected_durations(result.params.transition)
    stationary = stationary_distribution(result.params.transition)

    state_summary = [
        Dict(
            "state"            => k,
            "label"            => result.regime_labels[k],
            "mean_return"      => result.params.emission[k].mean,
            "std_return"       => result.params.emission[k].std,
            "expected_duration"=> durations[k],
            "stationary_prob"  => stationary[k],
        )
        for k in 1:K
    ]

    timeline = [
        Dict(
            "t"              => t,
            "date"           => isnothing(dates) ? nothing : dates[t],
            "return"         => returns[t],
            "viterbi_state"  => result.viterbi_states[t],
            "viterbi_label"  => result.regime_labels[result.viterbi_states[t]],
            "filtered_probs" => result.filtered_probs[t, :],
            "smoothed_probs" => result.smoothed_probs[t, :],
        )
        for t in 1:T
    ]

    Dict(
        "n_states"       => K,
        "log_likelihood" => result.log_likelihood,
        "n_iterations"   => result.n_iterations,
        "converged"      => result.converged,
        "states"         => state_summary,
        "forecast"       => [
            Dict("state" => k, "label" => result.regime_labels[k], "prob" => result.forecast[k])
            for k in 1:K
        ],
        "timeline"       => timeline,
        "transition_matrix" => [result.params.transition[i, j] for i in 1:K, j in 1:K],
    )
end

# ─── Synthetic Data ───────────────────────────────────────────────────────────

"""
    synthetic_btc_returns(n; regime_len=60, seed=42)

Generate synthetic BTC-like return series with regime switches for testing.
"""
function synthetic_btc_returns(n::Int; regime_len::Int=60)::Vector{Float64}
    # True regime parameters
    true_params = [
        (mean=0.002,  std=0.025),  # BULL
        (mean=-0.001, std=0.018),  # CHOPPY
        (mean=-0.008, std=0.045),  # BEAR
    ]
    returns = Float64[]
    regime = 1
    count = 0

    for i in 1:n
        p = true_params[regime]
        push!(returns, p.mean + p.std * randn())
        count += 1
        if count >= regime_len
            # Regime switch
            regime = rand(1:3)
            count = 0
        end
    end
    return returns
end

# ─── Demo / Entry Point ───────────────────────────────────────────────────────

function run_regime_switching_demo()
    @info "Running regime switching model demo..."

    returns = synthetic_btc_returns(500; regime_len=80)
    @info "Generated $(length(returns)) synthetic BTC returns"

    # 2-state model
    @info "Fitting 2-state HMM..."
    result2 = fit_regime_model(returns, 2; max_iter=200)
    @info "  Converged: $(result2.converged), LL: $(round(result2.log_likelihood, digits=2))"
    @info "  States: $(join(result2.regime_labels, " / "))"
    @info "  Forecast: " * join(
        ["$(result2.regime_labels[k])=$(round(result2.forecast[k], digits=3))" for k in 1:2], ", "
    )

    # 3-state model
    @info "Fitting 3-state HMM..."
    result3 = fit_regime_model(returns, 3; max_iter=200)
    @info "  Converged: $(result3.converged), LL: $(round(result3.log_likelihood, digits=2))"
    @info "  States: $(join(result3.regime_labels, " / "))"
    @info "  Forecast: " * join(
        ["$(result3.regime_labels[k])=$(round(result3.forecast[k], digits=3))" for k in 1:3], ", "
    )

    # Model selection
    T = length(returns)
    bic2 = hmm_bic(result2.log_likelihood, T, 2)
    bic3 = hmm_bic(result3.log_likelihood, T, 3)
    @info "  BIC 2-state: $(round(bic2, digits=1)), BIC 3-state: $(round(bic3, digits=1))"
    @info "  Preferred model (BIC): $(bic2 < bic3 ? "2-state" : "3-state")"

    # Export
    outfile = joinpath(@__DIR__, "regime_switching_results.json")
    open(outfile, "w") do io
        JSON3.write(io, Dict(
            "model_2state" => result_to_dict(result2, returns),
            "model_3state" => result_to_dict(result3, returns),
            "bic" => Dict("k2" => bic2, "k3" => bic3),
        ))
    end
    @info "Results written to $outfile"

    return (result2, result3)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_regime_switching_demo()
end
