"""
OnlineLearning.jl
==================
Online learning module for adaptive trading and portfolio management.

Exports:
  AdaGrad          — adaptive gradient optimizer
  Adam             — Adam optimizer
  RMSprop          — RMS propagation optimizer
  FTRLProximal     — Follow-the-Regularized-Leader with L1/L2
  HedgeAlgorithm   — exponential weights for expert aggregation
  LinUCB           — linear upper confidence bound for contextual bandits
  UniversalPortfolio — Cover's universal portfolio algorithm
  PAMR             — Passive Aggressive Mean Reversion
  ADWIN            — adaptive windowing for concept drift detection
  OnlineEnsemble   — weighted combination of online learners
"""
module OnlineLearning

using Statistics, LinearAlgebra, Random

export AdaGrad, Adam, RMSprop, FTRLProximal, HedgeAlgorithm, LinUCB,
       UniversalPortfolio, PAMR, ADWIN, OnlineEnsemble,
       update!, predict, reset!, weights, portfolio_return

# ─────────────────────────────────────────────────────────────────────────────
# 1. ADAGRAD OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

"""
    AdaGrad

Adaptive gradient optimizer. Accumulates squared gradients to
adaptively scale learning rates per parameter.
"""
mutable struct AdaGrad
    eta      ::Float64           # global learning rate
    eps      ::Float64           # numerical stability
    theta    ::Vector{Float64}   # parameters
    G_sum    ::Vector{Float64}   # accumulated squared gradients
    t        ::Int               # step count
end

function AdaGrad(n_params::Int; eta::Float64=0.01, eps::Float64=1e-8)
    AdaGrad(eta, eps, zeros(n_params), zeros(n_params), 0)
end

"""
    update!(ag, grad) → Vector{Float64}

One AdaGrad update step. Returns updated parameters.
"""
function update!(ag::AdaGrad, grad::Vector{Float64})
    ag.G_sum .+= grad.^2
    lr = ag.eta ./ (sqrt.(ag.G_sum) .+ ag.eps)
    ag.theta .-= lr .* grad
    ag.t += 1
    return ag.theta
end

function reset!(ag::AdaGrad)
    fill!(ag.theta, 0.0)
    fill!(ag.G_sum, 0.0)
    ag.t = 0
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. ADAM OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

"""
    Adam

Adam: Adaptive Moment Estimation.
Combines adaptive learning rates (AdaGrad) with momentum (RMSprop).
"""
mutable struct Adam
    eta      ::Float64
    beta1    ::Float64
    beta2    ::Float64
    eps      ::Float64
    theta    ::Vector{Float64}
    m        ::Vector{Float64}   # first moment estimate
    v        ::Vector{Float64}   # second moment estimate
    t        ::Int
end

function Adam(n_params::Int; eta::Float64=0.001, beta1::Float64=0.9,
              beta2::Float64=0.999, eps::Float64=1e-8)
    Adam(eta, beta1, beta2, eps, zeros(n_params), zeros(n_params), zeros(n_params), 0)
end

function update!(adam::Adam, grad::Vector{Float64})
    adam.t += 1
    adam.m .= adam.beta1 .* adam.m .+ (1 - adam.beta1) .* grad
    adam.v .= adam.beta2 .* adam.v .+ (1 - adam.beta2) .* grad.^2

    # Bias correction
    m_hat = adam.m ./ (1 - adam.beta1^adam.t)
    v_hat = adam.v ./ (1 - adam.beta2^adam.t)

    adam.theta .-= adam.eta .* m_hat ./ (sqrt.(v_hat) .+ adam.eps)
    return adam.theta
end

function reset!(adam::Adam)
    fill!(adam.theta, 0.0)
    fill!(adam.m, 0.0)
    fill!(adam.v, 0.0)
    adam.t = 0
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. RMSPROP
# ─────────────────────────────────────────────────────────────────────────────

"""
    RMSprop

Root Mean Square Propagation: adaptive learning rate via exponential
moving average of squared gradients.
"""
mutable struct RMSprop
    eta      ::Float64
    rho      ::Float64           # decay rate for EMA
    eps      ::Float64
    theta    ::Vector{Float64}
    v        ::Vector{Float64}   # running avg of squared gradients
    t        ::Int
end

function RMSprop(n_params::Int; eta::Float64=0.001, rho::Float64=0.9, eps::Float64=1e-8)
    RMSprop(eta, rho, eps, zeros(n_params), ones(n_params), 0)
end

function update!(rms::RMSprop, grad::Vector{Float64})
    rms.v     .= rms.rho .* rms.v .+ (1 - rms.rho) .* grad.^2
    rms.theta .-= rms.eta .* grad ./ (sqrt.(rms.v) .+ rms.eps)
    rms.t += 1
    return rms.theta
end

function reset!(rms::RMSprop)
    fill!(rms.theta, 0.0)
    fill!(rms.v, 1.0)
    rms.t = 0
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. FTRL-PROXIMAL
# ─────────────────────────────────────────────────────────────────────────────

"""
    FTRLProximal

Follow-the-Regularized-Leader with L1 (Lasso) and L2 (Ridge) regularization.
Particularly effective for sparse online learning.
"""
mutable struct FTRLProximal
    alpha    ::Float64           # learning rate
    beta     ::Float64           # per-coordinate regularization
    lambda1  ::Float64           # L1 regularization
    lambda2  ::Float64           # L2 regularization
    theta    ::Vector{Float64}   # accumulated gradient (z in FTRL)
    n        ::Vector{Float64}   # accumulated squared gradients
    w        ::Vector{Float64}   # current weights (computed from z, n)
    t        ::Int
end

function FTRLProximal(n_params::Int; alpha::Float64=0.1, beta::Float64=1.0,
                       lambda1::Float64=0.01, lambda2::Float64=0.01)
    FTRLProximal(alpha, beta, lambda1, lambda2,
                  zeros(n_params), zeros(n_params), zeros(n_params), 0)
end

"""
    _compute_w!(ftrl)

Compute current weight vector from accumulated statistics.
"""
function _compute_w!(ftrl::FTRLProximal)
    for i in eachindex(ftrl.w)
        z_i = ftrl.theta[i]
        n_i = ftrl.n[i]
        if abs(z_i) <= ftrl.lambda1
            ftrl.w[i] = 0.0   # L1 sparsification
        else
            sign_z = sign(z_i)
            ftrl.w[i] = -(z_i - sign_z * ftrl.lambda1) /
                         ((ftrl.beta + sqrt(n_i)) / ftrl.alpha + ftrl.lambda2)
        end
    end
end

function update!(ftrl::FTRLProximal, grad::Vector{Float64})
    ftrl.t += 1
    sigma  = (sqrt.(ftrl.n .+ grad.^2) .- sqrt.(ftrl.n)) ./ ftrl.alpha
    ftrl.theta .+= grad .- sigma .* ftrl.w
    ftrl.n     .+= grad.^2
    _compute_w!(ftrl)
    return ftrl.w
end

function predict(ftrl::FTRLProximal, x::Vector{Float64})
    return dot(ftrl.w, x)
end

weights(ftrl::FTRLProximal) = copy(ftrl.w)

# ─────────────────────────────────────────────────────────────────────────────
# 5. HEDGE ALGORITHM (EXPONENTIAL WEIGHTS)
# ─────────────────────────────────────────────────────────────────────────────

"""
    HedgeAlgorithm

Multiplicative weights / Hedge algorithm for expert aggregation.
Maintains a distribution over N experts, updates with exponential weights.
"""
mutable struct HedgeAlgorithm
    n_experts ::Int
    eta       ::Float64          # learning rate (higher = faster adaptation)
    w         ::Vector{Float64}  # expert weights (unnormalized)
    p         ::Vector{Float64}  # expert probabilities (normalized)
    cumulative_loss::Vector{Float64}
    t         ::Int
end

function HedgeAlgorithm(n_experts::Int; eta::Float64=0.1)
    w = ones(n_experts)
    p = w ./ sum(w)
    HedgeAlgorithm(n_experts, eta, w, p, zeros(n_experts), 0)
end

"""
    update!(hedge, losses)

Update expert weights given realized losses for each expert.
`losses`: Vector of length n_experts with losses ∈ [0,1].
"""
function update!(hedge::HedgeAlgorithm, losses::Vector{Float64})
    length(losses) == hedge.n_experts || error("losses must have length n_experts")
    hedge.t += 1
    hedge.cumulative_loss .+= losses
    hedge.w .*= exp.(-hedge.eta .* losses)
    total_w   = sum(hedge.w)
    if total_w < 1e-30
        hedge.w .= 1.0 / hedge.n_experts
    else
        hedge.w ./= total_w
    end
    hedge.p .= hedge.w ./ sum(hedge.w)
    return hedge.p
end

"""
    predict(hedge, expert_predictions) → Float64

Weighted combination of expert predictions.
"""
function predict(hedge::HedgeAlgorithm, expert_preds::Vector{Float64})
    return dot(hedge.p, expert_preds)
end

weights(hedge::HedgeAlgorithm) = copy(hedge.p)

"""
    best_expert(hedge) → Int

Return index of the currently highest-weight expert.
"""
best_expert(hedge::HedgeAlgorithm) = argmax(hedge.p)

# ─────────────────────────────────────────────────────────────────────────────
# 6. LINUCB (LINEAR UPPER CONFIDENCE BOUND)
# ─────────────────────────────────────────────────────────────────────────────

"""
    LinUCB

Linear Upper Confidence Bound for contextual bandit problems.
Maintains a linear model per arm (action).
"""
mutable struct LinUCB
    n_arms    ::Int
    n_features::Int
    alpha     ::Float64          # exploration parameter
    A         ::Vector{Matrix{Float64}}   # A[k] = X'X + I per arm
    b         ::Vector{Vector{Float64}}   # b[k] = X'r per arm
    theta     ::Vector{Vector{Float64}}   # linear weights per arm
    t         ::Int
    arm_counts::Vector{Int}
end

function LinUCB(n_arms::Int, n_features::Int; alpha::Float64=1.0)
    A = [I(n_features) * 1.0 for _ in 1:n_arms]
    b = [zeros(n_features) for _ in 1:n_arms]
    θ = [zeros(n_features) for _ in 1:n_arms]
    LinUCB(n_arms, n_features, alpha, A, b, θ, 0, zeros(Int, n_arms))
end

"""
    select_arm(linucb, context) → Int

Select arm with highest UCB score given context vector.
"""
function select_arm(linucb::LinUCB, context::Vector{Float64})
    length(context) == linucb.n_features || error("Context dimension mismatch")
    ucb_scores = zeros(linucb.n_arms)
    for k in 1:linucb.n_arms
        A_inv    = inv(linucb.A[k])
        linucb.theta[k] .= A_inv * linucb.b[k]
        mu_k     = dot(linucb.theta[k], context)
        sigma_k  = sqrt(max(dot(context, A_inv * context), 0.0))
        ucb_scores[k] = mu_k + linucb.alpha * sigma_k
    end
    return argmax(ucb_scores)
end

"""
    update!(linucb, arm, context, reward)

Update LinUCB model for the chosen arm with observed reward.
"""
function update!(linucb::LinUCB, arm::Int, context::Vector{Float64}, reward::Float64)
    linucb.A[arm] .+= context * context'
    linucb.b[arm] .+= reward .* context
    linucb.arm_counts[arm] += 1
    linucb.t += 1
    # Recompute theta for this arm
    linucb.theta[arm] .= inv(linucb.A[arm]) * linucb.b[arm]
    return linucb
end

function predict(linucb::LinUCB, arm::Int, context::Vector{Float64})
    return dot(linucb.theta[arm], context)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. UNIVERSAL PORTFOLIO (COVER 1991)
# ─────────────────────────────────────────────────────────────────────────────

"""
    UniversalPortfolio

Cover's universal portfolio: theoretically optimal wealth-relative growth
strategy. Updates portfolio weights as the performance-weighted average
over all constant rebalanced portfolios (CRPs).

For n assets, the update rule (simplified by sampling):
  w_{t+1} ∝ ∫ b * W_t(b) db
  where W_t(b) = product of period returns under CRP b
"""
mutable struct UniversalPortfolio
    n         ::Int
    n_samples ::Int               # number of CRP samples for approximation
    portfolios::Matrix{Float64}   # n_samples × n portfolio samples (on simplex)
    wealth    ::Vector{Float64}   # cumulative wealth of each sampled portfolio
    w         ::Vector{Float64}   # current universal portfolio weights
    t         ::Int
end

function UniversalPortfolio(n::Int; n_samples::Int=500,
                              rng::AbstractRNG=MersenneTwister(42))
    # Sample portfolios uniformly on simplex (Dirichlet(1,...,1))
    portfolios = _sample_simplex(n_samples, n, rng)
    wealth     = ones(n_samples)
    w          = ones(n) / n
    UniversalPortfolio(n, n_samples, portfolios, wealth, w, 0)
end

function _sample_simplex(m::Int, n::Int, rng::AbstractRNG)
    # Each row is a point on the (n-1)-simplex
    X = -log.(rand(rng, m, n) .+ 1e-10)
    return X ./ sum(X, dims=2)
end

"""
    update!(up, returns) → Vector{Float64}

Update universal portfolio given period returns (gross, e.g. 1+r_t).
Returns new portfolio weights for next period.
"""
function update!(up::UniversalPortfolio, returns::Vector{Float64})
    length(returns) == up.n || error("returns must have length n")
    up.t += 1

    # Update wealth of each sampled portfolio
    for k in 1:up.n_samples
        port_ret = dot(up.portfolios[k,:], returns)
        up.wealth[k] *= max(port_ret, 1e-10)
    end

    # Universal portfolio: wealth-weighted average of sampled portfolios
    total_w = sum(up.wealth)
    if total_w < 1e-20
        up.w .= 1.0 / up.n
    else
        up.w .= vec(sum(up.portfolios .* (up.wealth ./ total_w), dims=1))
        up.w .= max.(up.w, 1e-8)
        up.w ./= sum(up.w)
    end

    return up.w
end

"""
    portfolio_return(up, returns) → Float64

Compute period return under current universal portfolio weights.
"""
portfolio_return(up::UniversalPortfolio, returns::Vector{Float64}) =
    dot(up.w, returns)

weights(up::UniversalPortfolio) = copy(up.w)

# ─────────────────────────────────────────────────────────────────────────────
# 8. PAMR (PASSIVE AGGRESSIVE MEAN REVERSION)
# ─────────────────────────────────────────────────────────────────────────────

"""
    PAMR

Passive Aggressive Mean Reversion for online portfolio selection.
Exploits the mean-reversion property of relative asset prices.

Reference: Li et al., "PAMR: Passive Aggressive Mean Reversion Strategy"
"""
mutable struct PAMR
    n         ::Int
    epsilon   ::Float64          # aggressiveness parameter
    C         ::Float64          # regularization (PAMR-1 and PAMR-2)
    variant   ::Symbol           # :PAMR, :PAMR1, :PAMR2
    w         ::Vector{Float64}  # current portfolio weights
    t         ::Int
    returns_history ::Matrix{Float64}  # circular buffer of past returns
    hist_size ::Int
end

function PAMR(n::Int; epsilon::Float64=0.5, C::Float64=500.0, variant::Symbol=:PAMR1)
    w = ones(n) / n
    PAMR(n, epsilon, C, variant, w, 0, zeros(0,0), 0)
end

"""
    update!(pamr, x_t) → Vector{Float64}

Update PAMR portfolio given price relatives x_t = p_t / p_{t-1}.
Returns new portfolio weights.
"""
function update!(pamr::PAMR, x_t::Vector{Float64})
    length(x_t) == pamr.n || error("x_t must have length n")
    pamr.t += 1

    # Predicted return of current portfolio
    x_bar = mean(x_t)
    x_tilde = x_t .- x_bar  # mean-centered price relative

    loss_t = max(dot(pamr.w, x_t) - pamr.epsilon, 0.0)

    # Passive-aggressive update
    denom = sum(x_tilde.^2) + 1e-10
    if pamr.variant == :PAMR
        tau = loss_t / denom
    elseif pamr.variant == :PAMR1
        tau = min(pamr.C, loss_t / denom)
    else  # PAMR2
        tau = loss_t / (denom + 1 / (2 * pamr.C))
    end

    w_new = pamr.w - tau .* x_tilde

    # Project onto simplex
    w_new = _project_simplex(w_new)
    pamr.w .= w_new

    return pamr.w
end

"""
    _project_simplex(v) → Vector{Float64}

Euclidean projection onto the probability simplex.
O(n log n) algorithm (sort-based).
"""
function _project_simplex(v::Vector{Float64})
    n   = length(v)
    u   = sort(v, rev=true)
    rho = 0
    cumsum_u = 0.0
    for j in 1:n
        cumsum_u += u[j]
        if u[j] - (cumsum_u - 1) / j > 0
            rho = j
        end
    end
    theta = (sum(u[1:rho]) - 1) / rho
    return max.(v .- theta, 0.0)
end

weights(pamr::PAMR) = copy(pamr.w)

# ─────────────────────────────────────────────────────────────────────────────
# 9. ADWIN (ADAPTIVE WINDOWING)
# ─────────────────────────────────────────────────────────────────────────────

"""
    ADWIN

Adaptive Windowing for concept drift detection.
Maintains a variable-length window of recent data.
Detects change when two sub-windows have statistically different means.

Reference: Bifet & Gavalda (2007)
"""
mutable struct ADWIN
    delta     ::Float64          # confidence parameter (p-value like)
    window    ::Vector{Float64}  # current data window
    total     ::Float64          # sum of all elements in window
    variance  ::Float64          # estimate of variance
    drift_detected::Bool
    last_drift_idx::Int
    n_drifts  ::Int
end

function ADWIN(; delta::Float64=0.002)
    ADWIN(delta, Float64[], 0.0, 0.0, false, 0, 0)
end

"""
    update!(adwin, x) → Bool

Add new observation x to ADWIN window.
Returns `true` if concept drift detected.
"""
function update!(adwin::ADWIN, x::Float64)
    push!(adwin.window, x)
    adwin.total += x
    adwin.drift_detected = false

    n = length(adwin.window)
    n < 2 && return false

    # Check for drift by comparing all possible split points
    # Simplified: check a subset of splits for efficiency
    best_eps = 0.0
    split_point = -1

    # Check splits at powers of 2 from the end (ADWIN bucket approximation)
    check_points = unique([max(1, n - 2^k) for k in 0:floor(Int, log2(n))])
    push!(check_points, n ÷ 2)

    for i in check_points
        1 <= i < n || continue
        n0 = i; n1 = n - i
        (n0 < 1 || n1 < 1) && continue

        mu0 = mean(adwin.window[1:n0])
        mu1 = mean(adwin.window[n0+1:end])
        delta_mean = abs(mu1 - mu0)

        # Hoeffding-like bound
        m_star = 1.0 / (1.0/n0 + 1.0/n1)
        eps_cut = sqrt(log(4.0 * n / adwin.delta) / (2 * m_star))

        if delta_mean > eps_cut && delta_mean > best_eps
            best_eps    = delta_mean
            split_point = n0
        end
    end

    if split_point > 0
        # Remove the stale portion (before split)
        adwin.total   = sum(adwin.window[split_point+1:end])
        adwin.window  = adwin.window[split_point+1:end]
        adwin.drift_detected = true
        adwin.last_drift_idx += split_point
        adwin.n_drifts += 1
        return true
    end

    return false
end

"""
    current_estimate(adwin) → (mean, std, n)

Return current mean, std, and window size.
"""
function current_estimate(adwin::ADWIN)
    isempty(adwin.window) && return (NaN, NaN, 0)
    w = adwin.window
    return (mean(w), std(w), length(w))
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. ONLINE ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

"""
    OnlineEnsemble

Weighted combination of online learners.
Uses Hedge-style exponential weight updates based on realized loss.
"""
mutable struct OnlineEnsemble
    n_learners ::Int
    eta        ::Float64
    weights    ::Vector{Float64}
    learners   ::Vector{Any}       # any online learner with predict() interface
    cumulative_loss::Vector{Float64}
    t          ::Int
end

function OnlineEnsemble(learners::Vector; eta::Float64=0.1)
    n = length(learners)
    OnlineEnsemble(n, eta, ones(n) ./ n, learners, zeros(n), 0)
end

"""
    predict(ens, x) → Float64

Ensemble prediction: weighted combination of learner predictions.
"""
function predict(ens::OnlineEnsemble, x)
    preds = [predict(ens.learners[k], x) for k in 1:ens.n_learners]
    return dot(ens.weights, preds)
end

"""
    update!(ens, x, y_true) → Nothing

Update ensemble weights based on realized prediction errors.
"""
function update!(ens::OnlineEnsemble, x, y_true::Float64)
    ens.t += 1
    preds  = [predict(ens.learners[k], x) for k in 1:ens.n_learners]
    losses = [(p - y_true)^2 for p in preds]  # squared error loss

    # Normalize losses to [0,1]
    max_l = maximum(losses)
    norm_losses = max_l > 0 ? losses ./ max_l : losses

    # Hedge weight update
    ens.weights .*= exp.(-ens.eta .* norm_losses)
    total = sum(ens.weights)
    ens.weights ./= max(total, 1e-30)

    ens.cumulative_loss .+= losses
    return ens
end

weights(ens::OnlineEnsemble) = copy(ens.weights)

best_learner(ens::OnlineEnsemble) = argmax(ens.weights)

# ─────────────────────────────────────────────────────────────────────────────
# 11. ONLINE GRADIENT DESCENT PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

"""
    OGDPortfolio

Online Gradient Descent for portfolio optimization.
Minimizes cumulative negative log-return (surrogate for Sharpe).
"""
mutable struct OGDPortfolio
    n         ::Int
    eta       ::Float64
    w         ::Vector{Float64}
    t         ::Int
    returns_log::Vector{Float64}   # log of realized portfolio returns
end

function OGDPortfolio(n::Int; eta::Float64=0.01)
    OGDPortfolio(n, eta, ones(n)/n, 0, Float64[])
end

function update!(ogd::OGDPortfolio, price_relatives::Vector{Float64})
    ogd.t += 1
    port_r = dot(ogd.w, price_relatives)
    push!(ogd.returns_log, log(max(port_r, 1e-10)))

    # Gradient of -log(w'x) w.r.t. w
    grad = -price_relatives ./ (port_r + 1e-10)

    # Gradient descent step
    w_new = ogd.w .- ogd.eta .* grad
    ogd.w .= _project_simplex(w_new)
    return ogd.w
end

weights(ogd::OGDPortfolio) = copy(ogd.w)

# ─────────────────────────────────────────────────────────────────────────────
# 12. SEQUENTIAL BAYESIAN UPDATE
# ─────────────────────────────────────────────────────────────────────────────

"""
    BayesianOnlineRegression

Online Bayesian linear regression with Gaussian prior.
Maintains posterior distribution over weights as Normal(mu, Sigma).
"""
mutable struct BayesianOnlineRegression
    n         ::Int
    mu        ::Vector{Float64}   # posterior mean
    Sigma     ::Matrix{Float64}   # posterior covariance
    sigma2    ::Float64           # noise variance (observation noise)
    t         ::Int
end

function BayesianOnlineRegression(n::Int; prior_var::Float64=1.0, sigma2::Float64=0.01)
    BayesianOnlineRegression(n, zeros(n), prior_var * I(n) * 1.0, sigma2, 0)
end

"""
    update!(bor, x, y)

Bayesian update given new observation (x, y) where y = w'x + ε.
"""
function update!(bor::BayesianOnlineRegression, x::Vector{Float64}, y::Float64)
    bor.t += 1
    # Kalman filter update (same as Bayesian linear regression)
    Sx  = bor.Sigma * x
    K   = Sx ./ (dot(x, Sx) + bor.sigma2)   # Kalman gain
    bor.mu    .+= K .* (y - dot(x, bor.mu))
    bor.Sigma .-= K * Sx'
    # Ensure symmetry
    bor.Sigma  .= (bor.Sigma + bor.Sigma') ./ 2
    return bor
end

function predict(bor::BayesianOnlineRegression, x::Vector{Float64})
    mu_pred  = dot(bor.mu, x)
    var_pred = dot(x, bor.Sigma * x) + bor.sigma2
    return (mean=mu_pred, variance=var_pred, std=sqrt(var_pred))
end

# ─────────────────────────────────────────────────────────────────────────────
# 13. EXPONENTIAL SMOOTHING / ETS
# ─────────────────────────────────────────────────────────────────────────────

"""
    ExponentialSmoothing

Simple exponential smoothing for online forecasting.
Supports Simple (SES), Holt's, and Holt-Winters variants.
"""
mutable struct ExponentialSmoothing
    alpha    ::Float64   # level smoothing
    beta     ::Float64   # trend smoothing (0 → no trend)
    L        ::Float64   # level
    B        ::Float64   # trend
    t        ::Int
    history  ::Vector{Float64}
end

function ExponentialSmoothing(; alpha::Float64=0.2, beta::Float64=0.0)
    ExponentialSmoothing(alpha, beta, 0.0, 0.0, 0, Float64[])
end

function update!(es::ExponentialSmoothing, y::Float64)
    es.t += 1
    push!(es.history, y)

    if es.t == 1
        es.L = y; es.B = 0.0
        return y
    end

    L_prev = es.L; B_prev = es.B
    es.L   = es.alpha * y + (1 - es.alpha) * (L_prev + B_prev)
    if es.beta > 0
        es.B = es.beta * (es.L - L_prev) + (1 - es.beta) * B_prev
    end
    return es.L + es.B
end

function forecast(es::ExponentialSmoothing, h::Int)
    return es.L + h * es.B
end

# ─────────────────────────────────────────────────────────────────────────────
# 14. REPLAY BUFFER FOR BATCH ONLINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

"""
    ReplayBuffer

Fixed-size circular buffer for experience replay.
"""
mutable struct ReplayBuffer{T}
    capacity ::Int
    buffer   ::Vector{T}
    pos      ::Int
    full     ::Bool
end

function ReplayBuffer{T}(capacity::Int) where T
    ReplayBuffer{T}(capacity, Vector{T}(undef, capacity), 1, false)
end

function push_sample!(rb::ReplayBuffer{T}, x::T) where T
    rb.buffer[rb.pos] = x
    rb.pos = mod1(rb.pos + 1, rb.capacity)
    if rb.pos == 1; rb.full = true; end
end

function sample(rb::ReplayBuffer, n::Int; rng::AbstractRNG=MersenneTwister(42))
    size = rb.full ? rb.capacity : rb.pos - 1
    size < 1 && return []
    idx = rand(rng, 1:size, min(n, size))
    return rb.buffer[idx]
end

Base.length(rb::ReplayBuffer) = rb.full ? rb.capacity : rb.pos - 1

end  # module OnlineLearning
