"""
    MLSignals

Machine learning signal generation and calibration for the SRFM quantitative
trading system. Implements Gaussian Process regression, Bayesian Ridge,
Kalman filter extraction, Hidden Markov Models for regime detection,
information-theoretic feature selection, and ensemble probability calibration.
"""
module MLSignals

using LinearAlgebra
using Statistics
using Distributions

export gp_sq_exp_kernel, gp_matern52_kernel, gp_predict, gp_log_marginal_likelihood
export optimize_gp_hyperparams
export bayesian_ridge_fit, bayesian_ridge_predict
export kalman_filter, kalman_smoother, kalman_em
export HMMModel, hmm_baum_welch, hmm_viterbi, hmm_predict_state
export mutual_information, conditional_mutual_information, mrmr_feature_selection
export isotonic_regression, platt_scaling, calibrate_probabilities

# ---------------------------------------------------------------------------
# Gaussian Process regression
# ---------------------------------------------------------------------------

"""
    gp_sq_exp_kernel(x1, x2, length_scale, signal_var)

Squared exponential (RBF) covariance kernel.
k(x1, x2) = signal_var * exp(-||x1-x2||^2 / (2 * length_scale^2))
"""
function gp_sq_exp_kernel(x1::AbstractVector, x2::AbstractVector,
                           length_scale::Real, signal_var::Real)::Float64
    diff = x1 .- x2
    return signal_var * exp(-dot(diff, diff) / (2.0 * length_scale^2))
end

"""
    gp_matern52_kernel(x1, x2, length_scale, signal_var)

Matern 5/2 covariance kernel.
k(x1, x2) = signal_var * (1 + sqrt5*r + 5*r^2/3) * exp(-sqrt5*r)
where r = ||x1-x2|| / length_scale
"""
function gp_matern52_kernel(x1::AbstractVector, x2::AbstractVector,
                              length_scale::Real, signal_var::Real)::Float64
    diff = x1 .- x2
    r = sqrt(dot(diff, diff)) / length_scale
    sqrt5r = sqrt(5.0) * r
    return signal_var * (1.0 + sqrt5r + 5.0 * r^2 / 3.0) * exp(-sqrt5r)
end

"""
    build_kernel_matrix(X, length_scale, signal_var, noise_var; kernel=:sqexp)

Build the kernel (covariance) matrix for a set of training inputs X.
X is (N x D) matrix; returns (N x N) matrix.
"""
function build_kernel_matrix(X::Matrix{Float64}, length_scale::Real, signal_var::Real,
                               noise_var::Real; kernel::Symbol=:sqexp)::Matrix{Float64}
    N = size(X, 1)
    K = zeros(N, N)
    kfunc = kernel == :matern52 ? gp_matern52_kernel : gp_sq_exp_kernel
    for i in 1:N
        for j in i:N
            k_val = kfunc(X[i, :], X[j, :], length_scale, signal_var)
            K[i, j] = k_val
            K[j, i] = k_val
        end
        K[i, i] += noise_var
    end
    return K
end

"""
    build_cross_kernel_matrix(X_star, X, length_scale, signal_var; kernel=:sqexp)

Build the cross-covariance matrix K(X_star, X) for prediction.
"""
function build_cross_kernel_matrix(X_star::Matrix{Float64}, X::Matrix{Float64},
                                    length_scale::Real, signal_var::Real;
                                    kernel::Symbol=:sqexp)::Matrix{Float64}
    N_star = size(X_star, 1)
    N = size(X, 1)
    K_star = zeros(N_star, N)
    kfunc = kernel == :matern52 ? gp_matern52_kernel : gp_sq_exp_kernel
    for i in 1:N_star
        for j in 1:N
            K_star[i, j] = kfunc(X_star[i, :], X[j, :], length_scale, signal_var)
        end
    end
    return K_star
end

"""
    gp_predict(X_train, y_train, X_test, length_scale, signal_var, noise_var;
               kernel=:sqexp)

Gaussian Process posterior mean and variance predictions.

# Arguments
- `X_train`      : (N x D) training inputs
- `y_train`      : (N,) training targets
- `X_test`       : (M x D) test inputs
- `length_scale` : kernel length scale
- `signal_var`   : kernel signal variance
- `noise_var`    : observation noise variance

Returns named tuple (mean, variance) each of length M.
"""
function gp_predict(X_train::Matrix{Float64}, y_train::Vector{Float64},
                    X_test::Matrix{Float64}, length_scale::Real,
                    signal_var::Real, noise_var::Real; kernel::Symbol=:sqexp)
    K = build_kernel_matrix(X_train, length_scale, signal_var, noise_var; kernel=kernel)
    K_star = build_cross_kernel_matrix(X_test, X_train, length_scale, signal_var; kernel=kernel)

    # Cholesky for stability
    K_chol = cholesky(Symmetric(K + 1e-10 * I))
    alpha = K_chol \ y_train

    mu_star = K_star * alpha

    # Predictive variance
    v = K_chol.L \ K_star'
    N_test = size(X_test, 1)
    var_star = zeros(N_test)
    for i in 1:N_test
        k_ss = gp_sq_exp_kernel(X_test[i, :], X_test[i, :], length_scale, signal_var)
        if kernel == :matern52
            k_ss = gp_matern52_kernel(X_test[i, :], X_test[i, :], length_scale, signal_var)
        end
        var_star[i] = max(k_ss + noise_var - dot(v[:, i], v[:, i]), 0.0)
    end

    return (mean=mu_star, variance=var_star)
end

"""
    gp_log_marginal_likelihood(X_train, y_train, length_scale, signal_var, noise_var;
                                kernel=:sqexp)

Compute the log marginal likelihood for GP hyperparameter optimization.
log p(y | X, theta) = -0.5 * y^T K^-1 y - 0.5 * log|K| - N/2 * log(2pi)
"""
function gp_log_marginal_likelihood(X_train::Matrix{Float64}, y_train::Vector{Float64},
                                     length_scale::Real, signal_var::Real, noise_var::Real;
                                     kernel::Symbol=:sqexp)::Float64
    length_scale > 0 || return -Inf
    signal_var > 0   || return -Inf
    noise_var > 0    || return -Inf

    K = build_kernel_matrix(X_train, length_scale, signal_var, noise_var; kernel=kernel)
    N = length(y_train)

    try
        K_chol = cholesky(Symmetric(K + 1e-10 * I))
        alpha = K_chol \ y_train
        log_det = 2.0 * sum(log(K_chol.L[i, i]) for i in 1:N)
        return -0.5 * dot(y_train, alpha) - 0.5 * log_det - 0.5 * N * log(2.0 * pi)
    catch
        return -Inf
    end
end

"""
    optimize_gp_hyperparams(X_train, y_train; kernel=:sqexp, n_restarts=5)

Optimize GP hyperparameters (length_scale, signal_var, noise_var) by maximizing
the log marginal likelihood using coordinate ascent with multiple restarts.

Returns named tuple (length_scale, signal_var, noise_var, lml).
"""
function optimize_gp_hyperparams(X_train::Matrix{Float64}, y_train::Vector{Float64};
                                   kernel::Symbol=:sqexp, n_restarts::Int=5)
    best_lml = -Inf
    best_params = (1.0, 1.0, 0.1)

    for _ in 1:n_restarts
        # Random initialization
        ls0 = exp(randn() * 1.5)
        sv0 = exp(randn() * 1.0)
        nv0 = exp(randn() * 1.0 - 2.0)
        params = [ls0, sv0, nv0]

        # Simple coordinate descent
        step = [0.5, 0.5, 0.1]
        best_local = gp_log_marginal_likelihood(X_train, y_train, params...; kernel=kernel)

        for iter in 1:200
            improved = false
            for j in 1:3
                for sign in [-1.0, 1.0]
                    trial = copy(params)
                    trial[j] = exp(log(params[j]) + sign * step[j])
                    trial[j] > 0 || continue
                    lml = gp_log_marginal_likelihood(X_train, y_train, trial...; kernel=kernel)
                    if lml > best_local
                        best_local = lml
                        params = trial
                        improved = true
                    end
                end
            end
            if !improved
                step .*= 0.6
                all(step .< 1e-6) && break
            end
        end

        if best_local > best_lml
            best_lml = best_local
            best_params = (params[1], params[2], params[3])
        end
    end

    return (length_scale=best_params[1], signal_var=best_params[2],
            noise_var=best_params[3], lml=best_lml)
end

# ---------------------------------------------------------------------------
# Bayesian Ridge regression
# ---------------------------------------------------------------------------

"""
    bayesian_ridge_fit(X, y; alpha_prior=1e-6, lambda_prior=1e-6, max_iter=300, tol=1e-4)

Fit a Bayesian Ridge regression model using the analytical posterior.
The model is y ~ N(X*w, alpha^{-1}*I) with prior w ~ N(0, lambda^{-1}*I).

# Arguments
- `alpha_prior`  : initial noise precision
- `lambda_prior` : initial weight precision (regularization)

Returns named tuple (weights, weight_cov, alpha, lambda, evidence).
"""
function bayesian_ridge_fit(X::Matrix{Float64}, y::Vector{Float64};
                             alpha_prior::Real=1e-6, lambda_prior::Real=1e-6,
                             max_iter::Int=300, tol::Real=1e-4)
    N, D = size(X)
    alpha = Float64(alpha_prior)
    lambda = Float64(lambda_prior)

    # Precompute SVD for efficiency
    U, S_svd, Vt = svd(X)
    S2 = S_svd .^ 2

    for iter in 1:max_iter
        # Posterior covariance: Sigma_w = (lambda*I + alpha*X'X)^{-1}
        eigenvalues = alpha .* S2 .+ lambda
        Sigma_w_diag = 1.0 ./ eigenvalues  # diagonal in SVD basis

        # Posterior mean
        mu_w = alpha * (Vt' * (Sigma_w_diag .* (Vt * (X' * y))))

        # Effective degrees of freedom
        gamma = sum(alpha .* S2_i ./ (alpha .* S2_i .+ lambda) for S2_i in S2)

        # Update hyperparameters (type-II ML / evidence optimization)
        alpha_new = (N - gamma) / max(sum((y - X * mu_w) .^ 2), 1e-10)
        lambda_new = gamma / max(dot(mu_w, mu_w), 1e-10)

        if abs(alpha_new - alpha) / (abs(alpha) + 1e-10) < tol &&
           abs(lambda_new - lambda) / (abs(lambda) + 1e-10) < tol
            alpha = alpha_new
            lambda = lambda_new
            break
        end
        alpha = alpha_new
        lambda = lambda_new
    end

    # Final posterior
    eigenvalues = alpha .* S2 .+ lambda
    Sigma_w_svd = Vt' * Diagonal(1.0 ./ eigenvalues) * Vt
    mu_w = alpha * Sigma_w_svd * (X' * y)

    # Evidence (log marginal likelihood)
    log_evidence = 0.5 * (D * log(lambda) + N * log(alpha)
                          - alpha * sum((y - X * mu_w).^2)
                          - lambda * dot(mu_w, mu_w)
                          - sum(log.(eigenvalues))
                          - N * log(2.0 * pi))

    return (weights=mu_w, weight_cov=Sigma_w_svd, alpha=alpha, lambda=lambda,
            evidence=log_evidence)
end

"""
    bayesian_ridge_predict(X_test, model)

Make predictions with a fitted Bayesian Ridge model.
Returns named tuple (mean, variance) for each test point.
"""
function bayesian_ridge_predict(X_test::Matrix{Float64}, model)
    mu = X_test * model.weights
    # Predictive variance: sigma_n^2 + x^T * Sigma_w * x
    noise_var = 1.0 / model.alpha
    pred_vars = [noise_var + dot(X_test[i, :], model.weight_cov * X_test[i, :])
                 for i in 1:size(X_test, 1)]
    return (mean=mu, variance=pred_vars)
end

# ---------------------------------------------------------------------------
# Kalman filter for signal extraction
# ---------------------------------------------------------------------------

"""
    kalman_filter(y, F, H, Q, R, x0, P0)

Run the linear Gaussian Kalman filter.

# Model
- State:       x_t = F * x_{t-1} + w_t,  w_t ~ N(0, Q)
- Observation: y_t = H * x_t + v_t,       v_t ~ N(0, R)

# Arguments
- `y`  : (T x m) observation matrix
- `F`  : (n x n) state transition matrix
- `H`  : (m x n) observation matrix
- `Q`  : (n x n) process noise covariance
- `R`  : (m x m) observation noise covariance
- `x0` : (n,) initial state mean
- `P0` : (n x n) initial state covariance

Returns named tuple (filtered_states, filtered_covs, innovations, S_mats, log_likelihood).
"""
function kalman_filter(y::Matrix{Float64}, F::Matrix{Float64}, H::Matrix{Float64},
                        Q::Matrix{Float64}, R::Matrix{Float64},
                        x0::Vector{Float64}, P0::Matrix{Float64})
    T, m = size(y)
    n = length(x0)

    x_filt = zeros(T, n)
    P_filt = Array{Matrix{Float64}}(undef, T)
    innovations = zeros(T, m)
    S_mats = Array{Matrix{Float64}}(undef, T)
    log_lik = 0.0

    x_pred = copy(x0)
    P_pred = copy(P0)

    for t in 1:T
        # Innovation
        innov = y[t, :] - H * x_pred
        S = H * P_pred * H' + R
        innovations[t, :] = innov
        S_mats[t] = S

        # Kalman gain
        K_gain = P_pred * H' / S

        # Update
        x_filt[t, :] = x_pred + K_gain * innov
        P_filt[t] = (I(n) - K_gain * H) * P_pred

        # Log-likelihood contribution
        try
            S_chol = cholesky(Symmetric(S))
            log_lik -= 0.5 * (m * log(2.0 * pi) + 2.0 * sum(log.(diag(S_chol.L)))
                               + dot(innov, S \ innov))
        catch
            log_lik -= 1e6
        end

        # Predict
        if t < T
            x_pred = F * x_filt[t, :]
            P_pred = F * P_filt[t] * F' + Q
        end
    end

    return (filtered_states=x_filt, filtered_covs=P_filt,
            innovations=innovations, S_mats=S_mats, log_likelihood=log_lik)
end

"""
    kalman_smoother(filtered_states, filtered_covs, F, Q)

Run the Rauch-Tung-Striebel smoother on Kalman filter output.
Returns (smoothed_states, smoothed_covs).
"""
function kalman_smoother(filtered_states::Matrix{Float64},
                          filtered_covs::Vector{Matrix{Float64}},
                          F::Matrix{Float64}, Q::Matrix{Float64})
    T, n = size(filtered_states)
    x_smooth = copy(filtered_states)
    P_smooth = deepcopy(filtered_covs)

    for t in (T - 1):-1:1
        P_pred = F * filtered_covs[t] * F' + Q
        L = filtered_covs[t] * F' / P_pred  # Smoother gain
        x_smooth[t, :] = filtered_states[t, :] + L * (x_smooth[t+1, :] - F * filtered_states[t, :])
        P_smooth[t] = filtered_covs[t] + L * (P_smooth[t+1] - P_pred) * L'
    end

    return (smoothed_states=x_smooth, smoothed_covs=P_smooth)
end

"""
    kalman_em(y; n_iter=50, local_linear=true)

EM algorithm for estimating Kalman filter parameters (F, H, Q, R, x0, P0)
from observed data using the local linear trend model.

Returns named tuple (F, H, Q, R, x0, P0, log_likelihoods).
"""
function kalman_em(y::Vector{Float64}; n_iter::Int=50, local_linear::Bool=true)
    T = length(y)
    Y = reshape(y, T, 1)  # (T x 1)

    if local_linear
        # State: [level, trend]
        n = 2; m = 1
        F = [1.0 1.0; 0.0 1.0]
        H = [1.0 0.0]
        Q = [0.1 0.0; 0.0 0.01]
        R = [var(y)][:, :]
        x0 = [y[1], 0.0]
        P0 = [1.0 0.0; 0.0 0.1]
    else
        # AR(1) state
        n = 1; m = 1
        F = [0.9][:, :]
        H = [1.0][:, :]
        Q = [0.1][:, :]
        R = [var(y) * 0.1][:, :]
        x0 = [y[1]]
        P0 = [1.0][:, :]
    end

    log_likelihoods = Float64[]

    for iter in 1:n_iter
        # E-step
        kf = kalman_filter(Y, F, H, Q, R, x0, P0)
        push!(log_likelihoods, kf.log_likelihood)

        ks = kalman_smoother(kf.filtered_states, kf.filtered_covs, F, Q)
        xs = ks.smoothed_states
        Ps = ks.smoothed_covs

        # M-step: update Q and R
        # E[x_t * x_t'] sum
        Pxx = zeros(n, n)
        Pxx1 = zeros(n, n)
        for t in 1:T
            Pxx += Ps[t] + (xs[t, :] * xs[t, :]')
        end
        for t in 2:T
            # Cross covariance: E[x_t * x_{t-1}'] needs Kalman gain - simplified here
            Pxx1 += Ps[min(t, T)] + (xs[t, :] * xs[t-1, :]')
        end

        # Update R
        R_new = zeros(m, m)
        for t in 1:T
            resid = Y[t, :] - H * xs[t, :]
            R_new += resid * resid' + H * Ps[t] * H'
        end
        R = R_new ./ T

        # Update Q
        Q_new = zeros(n, n)
        for t in 2:T
            diff = xs[t, :] - F * xs[t-1, :]
            Q_new += diff * diff' + Ps[t] + F * Ps[t-1] * F'
        end
        Q = Q_new ./ (T - 1)

        # Update x0, P0
        x0 = xs[1, :]
        P0 = Ps[1]

        # Check convergence
        if iter > 2 && abs(log_likelihoods[end] - log_likelihoods[end-1]) < 1e-6
            break
        end
    end

    return (F=F, H=H, Q=Q, R=R, x0=x0, P0=P0, log_likelihoods=log_likelihoods)
end

# ---------------------------------------------------------------------------
# Hidden Markov Model for regime detection
# ---------------------------------------------------------------------------

"""
    HMMModel

Gaussian HMM with K states, each state having a Gaussian emission distribution.
"""
mutable struct HMMModel
    K::Int                   # number of states
    pi_init::Vector{Float64}  # initial state distribution
    A::Matrix{Float64}       # (K x K) transition matrix
    mu::Vector{Float64}      # emission means
    sigma::Vector{Float64}   # emission standard deviations
end

"""
    HMMModel(K)

Initialize an HMM with K states using random parameters.
"""
function HMMModel(K::Int)::HMMModel
    pi_init = ones(K) ./ K
    A = ones(K, K) ./ K
    mu = randn(K)
    sigma = ones(K)
    return HMMModel(K, pi_init, A, mu, sigma)
end

"""
    hmm_emission_log_prob(model, x, k)

Log probability of observation x under state k.
"""
function hmm_emission_log_prob(model::HMMModel, x::Real, k::Int)::Float64
    return logpdf(Normal(model.mu[k], model.sigma[k]), x)
end

"""
    hmm_forward(model, obs)

Compute the forward (alpha) probabilities in log-scale.
Returns log_alpha matrix of shape (T x K).
"""
function hmm_forward(model::HMMModel, obs::Vector{Float64})::Matrix{Float64}
    T = length(obs)
    K = model.K
    log_alpha = zeros(T, K)

    # Initialize
    for k in 1:K
        log_alpha[1, k] = log(model.pi_init[k] + 1e-300) +
                           hmm_emission_log_prob(model, obs[1], k)
    end

    # Recursion
    log_A = log.(model.A .+ 1e-300)
    for t in 2:T
        for k in 1:K
            # log-sum-exp trick
            vals = [log_alpha[t-1, j] + log_A[j, k] for j in 1:K]
            log_alpha[t, k] = logsumexp(vals) + hmm_emission_log_prob(model, obs[t], k)
        end
    end

    return log_alpha
end

"""
    hmm_backward(model, obs)

Compute the backward (beta) probabilities in log-scale.
Returns log_beta matrix of shape (T x K).
"""
function hmm_backward(model::HMMModel, obs::Vector{Float64})::Matrix{Float64}
    T = length(obs)
    K = model.K
    log_beta = zeros(T, K)
    log_A = log.(model.A .+ 1e-300)

    for t in (T-1):-1:1
        for k in 1:K
            vals = [log_A[k, j] + hmm_emission_log_prob(model, obs[t+1], j) + log_beta[t+1, j]
                    for j in 1:K]
            log_beta[t, k] = logsumexp(vals)
        end
    end
    return log_beta
end

"""
    logsumexp(x)

Numerically stable log-sum-exp.
"""
function logsumexp(x::Vector{Float64})::Float64
    max_x = maximum(x)
    isinf(max_x) && return -Inf
    return max_x + log(sum(exp(xi - max_x) for xi in x))
end

"""
    hmm_baum_welch(obs, K; n_iter=100, tol=1e-6, n_restarts=3)

Baum-Welch EM algorithm to fit a K-state Gaussian HMM to observations.
Runs n_restarts random initializations and returns the best model.

# Arguments
- `obs`       : vector of observations
- `K`         : number of hidden states
- `n_iter`    : max EM iterations per restart
- `tol`       : log-likelihood convergence tolerance
- `n_restarts`: number of random initializations

Returns the fitted HMMModel.
"""
function hmm_baum_welch(obs::Vector{Float64}, K::Int=3;
                         n_iter::Int=100, tol::Real=1e-6,
                         n_restarts::Int=3)::HMMModel
    T = length(obs)
    obs_min, obs_max = minimum(obs), maximum(obs)
    best_model = HMMModel(K)
    best_llik = -Inf

    for restart in 1:n_restarts
        model = HMMModel(K)
        # Initialize mu spread across data range
        model.mu = collect(range(obs_min, obs_max, length=K)) .+ 0.1 .* randn(K)
        model.sigma = fill(std(obs), K)
        # Dirichlet-like random init for A
        for i in 1:K
            row = abs.(randn(K)) .+ 0.1
            model.A[i, :] = row ./ sum(row)
        end

        prev_llik = -Inf

        for iter in 1:n_iter
            # E-step: forward-backward
            log_alpha = hmm_forward(model, obs)
            log_beta  = hmm_backward(model, obs)

            # Log-likelihood
            llik = logsumexp(log_alpha[T, :])

            # Gamma: state posterior
            log_gamma = log_alpha .+ log_beta
            for t in 1:T
                log_gamma[t, :] .-= logsumexp(log_gamma[t, :])
            end
            gamma = exp.(log_gamma)

            # Xi: joint state posterior (T-1 x K x K)
            log_A = log.(model.A .+ 1e-300)
            new_A = zeros(K, K)
            for t in 1:(T-1)
                for i in 1:K
                    for j in 1:K
                        log_xi = log_alpha[t, i] + log_A[i, j] +
                                  hmm_emission_log_prob(model, obs[t+1], j) +
                                  log_beta[t+1, j]
                        new_A[i, j] += exp(log_xi - llik)
                    end
                end
            end

            # M-step: update parameters
            # Transition matrix
            for i in 1:K
                row_sum = sum(new_A[i, :])
                model.A[i, :] = row_sum > 1e-10 ? new_A[i, :] ./ row_sum : ones(K) ./ K
            end

            # Emission parameters
            gamma_sum = sum(gamma, dims=1)[1, :]
            for k in 1:K
                gk = gamma_sum[k]
                if gk > 1e-10
                    model.mu[k] = dot(gamma[:, k], obs) / gk
                    model.sigma[k] = max(sqrt(dot(gamma[:, k], (obs .- model.mu[k]).^2) / gk), 1e-4)
                end
            end

            # Initial state
            model.pi_init = gamma[1, :]

            if abs(llik - prev_llik) < tol
                break
            end
            prev_llik = llik
        end

        final_llik = logsumexp(hmm_forward(model, obs)[T, :])
        if final_llik > best_llik
            best_llik = final_llik
            best_model = deepcopy(model)
        end
    end

    # Sort states by mean (ascending) for interpretability: bear / sideways / bull
    order = sortperm(best_model.mu)
    best_model.mu = best_model.mu[order]
    best_model.sigma = best_model.sigma[order]
    best_model.pi_init = best_model.pi_init[order]
    best_model.A = best_model.A[order, order]

    return best_model
end

"""
    hmm_viterbi(model, obs)

Viterbi algorithm: find the most likely state sequence.
Returns a vector of state indices (1-indexed).
"""
function hmm_viterbi(model::HMMModel, obs::Vector{Float64})::Vector{Int}
    T = length(obs)
    K = model.K
    log_A = log.(model.A .+ 1e-300)

    log_delta = zeros(T, K)
    psi = zeros(Int, T, K)

    # Initialize
    for k in 1:K
        log_delta[1, k] = log(model.pi_init[k] + 1e-300) +
                           hmm_emission_log_prob(model, obs[1], k)
    end

    # Recursion
    for t in 2:T
        for k in 1:K
            vals = [log_delta[t-1, j] + log_A[j, k] for j in 1:K]
            best_j = argmax(vals)
            log_delta[t, k] = vals[best_j] + hmm_emission_log_prob(model, obs[t], k)
            psi[t, k] = best_j
        end
    end

    # Backtrack
    states = zeros(Int, T)
    states[T] = argmax(log_delta[T, :])
    for t in (T-1):-1:1
        states[t] = psi[t+1, states[t+1]]
    end

    return states
end

"""
    hmm_predict_state(model, obs)

Predict the most likely current state and return state probabilities.
Returns named tuple (state_sequence, state_probs, regime_labels).
"""
function hmm_predict_state(model::HMMModel, obs::Vector{Float64})
    states = hmm_viterbi(model, obs)
    log_alpha = hmm_forward(model, obs)
    T = length(obs)
    log_probs = log_alpha[T, :]
    log_probs .-= logsumexp(log_probs)
    probs = exp.(log_probs)

    # Label states: lowest mean = bear, middle = sideways, highest = bull
    K = model.K
    labels = K == 3 ? ["bear", "sideways", "bull"] :
             K == 2 ? ["bear", "bull"] :
             ["state_$k" for k in 1:K]

    return (state_sequence=states, state_probs=probs, regime_labels=labels)
end

# ---------------------------------------------------------------------------
# Information-theoretic feature selection
# ---------------------------------------------------------------------------

"""
    discretize(x, n_bins)

Discretize a continuous vector into n_bins equal-width bins.
Returns integer bin indices (1-indexed).
"""
function discretize(x::Vector{Float64}, n_bins::Int=10)::Vector{Int}
    lo, hi = minimum(x), maximum(x)
    if lo == hi
        return ones(Int, length(x))
    end
    edges = range(lo, hi, length=n_bins + 1)
    bins = [clamp(searchsortedfirst(collect(edges), xi) - 1, 1, n_bins) for xi in x]
    return bins
end

"""
    mutual_information(x, y; n_bins=10)

Estimate mutual information I(X;Y) using histogram discretization.
"""
function mutual_information(x::Vector{Float64}, y::Vector{Float64};
                              n_bins::Int=10)::Float64
    N = min(length(x), length(y))
    xd = discretize(x[1:N], n_bins)
    yd = discretize(y[1:N], n_bins)

    # Joint histogram
    joint = zeros(n_bins, n_bins)
    for i in 1:N
        joint[xd[i], yd[i]] += 1.0
    end
    joint ./= N

    px = sum(joint, dims=2)[:, 1]
    py = sum(joint, dims=1)[1, :]

    mi = 0.0
    for i in 1:n_bins
        for j in 1:n_bins
            pij = joint[i, j]
            pij > 1e-12 && px[i] > 1e-12 && py[j] > 1e-12 || continue
            mi += pij * log(pij / (px[i] * py[j]))
        end
    end
    return max(mi, 0.0)
end

"""
    conditional_mutual_information(x, y, z; n_bins=10)

Estimate conditional mutual information I(X;Y|Z).
"""
function conditional_mutual_information(x::Vector{Float64}, y::Vector{Float64},
                                         z::Vector{Float64}; n_bins::Int=10)::Float64
    N = minimum(length.([x, y, z]))
    xd = discretize(x[1:N], n_bins)
    yd = discretize(y[1:N], n_bins)
    zd = discretize(z[1:N], n_bins)

    # I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    # Use joint counts
    cmi = 0.0
    for zbin in 1:n_bins
        z_mask = zd .== zbin
        pz = sum(z_mask) / N
        pz < 1e-12 && continue
        cmi += pz * mutual_information(Float64.(xd[z_mask]), Float64.(yd[z_mask]);
                                        n_bins=n_bins)
    end
    return max(cmi, 0.0)
end

"""
    mrmr_feature_selection(X, y, n_features; n_bins=10)

Minimum Redundancy Maximum Relevance (mRMR) feature selection.
Selects `n_features` features from X (N x D matrix) that maximize
relevance to y while minimizing redundancy among selected features.

Returns vector of selected feature column indices (1-indexed).
"""
function mrmr_feature_selection(X::Matrix{Float64}, y::Vector{Float64},
                                  n_features::Int; n_bins::Int=10)::Vector{Int}
    N, D = size(X)
    n_select = min(n_features, D)

    # Compute relevance: MI(Xi; y) for all features
    relevance = [mutual_information(X[:, j], y; n_bins=n_bins) for j in 1:D]

    selected = Int[]
    remaining = collect(1:D)

    for iter in 1:n_select
        if iter == 1
            best_j = argmax(relevance)
            push!(selected, best_j)
            filter!(x -> x != best_j, remaining)
            continue
        end

        # mRMR score: relevance - avg redundancy with selected
        best_score = -Inf
        best_j = remaining[1]
        for j in remaining
            red = mean(mutual_information(X[:, j], X[:, s]; n_bins=n_bins) for s in selected)
            score = relevance[j] - red
            if score > best_score
                best_score = score
                best_j = j
            end
        end
        push!(selected, best_j)
        filter!(x -> x != best_j, remaining)
    end

    return selected
end

# ---------------------------------------------------------------------------
# Ensemble calibration
# ---------------------------------------------------------------------------

"""
    isotonic_regression(y_pred, y_true)

Pool Adjacent Violators (PAV) algorithm for isotonic regression.
Used for probability calibration.

Returns the isotonic-calibrated probabilities for y_pred values.
"""
function isotonic_regression(y_pred::Vector{Float64}, y_true::Vector{Float64})::Vector{Float64}
    N = length(y_pred)
    @assert length(y_true) == N

    # Sort by predicted value
    order = sortperm(y_pred)
    y_sorted = y_true[order]

    # PAV algorithm
    calibrated = copy(y_sorted)
    # Merge blocks that violate monotonicity
    changed = true
    while changed
        changed = false
        i = 1
        while i < length(calibrated)
            if calibrated[i] > calibrated[i+1]
                # Merge i and i+1 with mean
                new_val = 0.5 * (calibrated[i] + calibrated[i+1])
                calibrated[i] = new_val
                calibrated[i+1] = new_val
                changed = true
            end
            i += 1
        end
    end

    # Map back to original order
    result = zeros(N)
    result[order] = calibrated
    return clamp.(result, 0.0, 1.0)
end

"""
    platt_scaling(scores, labels; n_iter=1000, lr=0.01)

Platt scaling for calibrating classifier probability outputs.
Fits a logistic regression: P(y=1|f) = 1 / (1 + exp(A*f + B))

# Arguments
- `scores` : raw classifier scores
- `labels` : binary labels (0 or 1)

Returns named tuple (A, B) parameters.
"""
function platt_scaling(scores::Vector{Float64}, labels::Vector{Float64};
                        n_iter::Int=1000, lr::Real=0.01)
    N = length(scores)
    A = 0.0
    B = log((sum(labels) + 1.0) / (N - sum(labels) + 1.0))

    for iter in 1:n_iter
        grad_A = 0.0
        grad_B = 0.0
        loss = 0.0

        for i in 1:N
            logit = A * scores[i] + B
            p = 1.0 / (1.0 + exp(-clamp(logit, -50.0, 50.0)))
            err = p - labels[i]
            grad_A += err * scores[i]
            grad_B += err
            loss -= labels[i] * log(max(p, 1e-12)) + (1.0 - labels[i]) * log(max(1.0 - p, 1e-12))
        end

        A -= lr / N * grad_A
        B -= lr / N * grad_B
    end

    return (A=A, B=B)
end

"""
    calibrate_probabilities(scores, labels, X_test; method=:platt)

Calibrate raw classifier scores to well-calibrated probabilities.

# Arguments
- `scores`  : training set raw scores
- `labels`  : training binary labels
- `X_test`  : test set raw scores
- `method`  : :platt or :isotonic

Returns calibrated probabilities for X_test.
"""
function calibrate_probabilities(scores::Vector{Float64}, labels::Vector{Float64},
                                   X_test::Vector{Float64}; method::Symbol=:platt)::Vector{Float64}
    if method == :platt
        params = platt_scaling(scores, labels)
        return [1.0 / (1.0 + exp(-clamp(params.A * s + params.B, -50.0, 50.0))) for s in X_test]
    else
        # Isotonic: fit on training data, interpolate for test
        cal = isotonic_regression(scores, labels)
        # Linear interpolation for test points
        order = sortperm(scores)
        scores_sorted = scores[order]
        cal_sorted = cal[order]
        result = zeros(length(X_test))
        for (i, s) in enumerate(X_test)
            idx = searchsortedfirst(scores_sorted, s)
            if idx <= 1
                result[i] = cal_sorted[1]
            elseif idx > length(scores_sorted)
                result[i] = cal_sorted[end]
            else
                t = (s - scores_sorted[idx-1]) / (scores_sorted[idx] - scores_sorted[idx-1] + 1e-14)
                result[i] = (1.0 - t) * cal_sorted[idx-1] + t * cal_sorted[idx]
            end
        end
        return clamp.(result, 0.0, 1.0)
    end
end

end  # module MLSignals
