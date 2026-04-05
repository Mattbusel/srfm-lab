"""
online_learning.jl — Online and Adaptive Learning for Financial Signals

Covers:
  - Online gradient descent: AdaGrad, Adam, RMSprop
  - Follow-the-Regularized-Leader (FTRL-Proximal)
  - Exponential Weights / Hedge algorithm (multiplicative updates)
  - Online-to-batch conversion
  - Second-order online methods (diagonal BFGS / ONS)
  - ADWIN concept drift detection
  - Contextual bandits: LinUCB for signal selection
  - Online portfolio selection: Universal Portfolio, OLMAR, PAMR
  - Regret analysis and cumulative regret tracking

Pure Julia stdlib only. No external dependencies.
"""

module OnlineLearning

using Statistics, LinearAlgebra, Random

export AdaGrad, Adam, RMSprop, update!
export FTRLProximal, ftrl_update!
export HedgeAlgorithm, hedge_update!, hedge_weights
export ADWIN, adwin_update!, adwin_detect
export LinUCB, linucb_select!, linucb_update!
export UniversalPortfolio, OLMAR, PAMR
export portfolio_step!, portfolio_weights
export OnlineRegretTracker, track_regret!, regret_summary
export run_online_learning_demo

# ─────────────────────────────────────────────────────────────
# 1. ONLINE GRADIENT DESCENT — ADAPTIVE OPTIMIZERS
# ─────────────────────────────────────────────────────────────

"""
    AdaGrad

AdaGrad adaptive learning rate optimizer.
Maintains sum of squared gradients per parameter.
"""
mutable struct AdaGrad
    eta::Float64            # base learning rate
    epsilon::Float64        # numerical stability
    G::Vector{Float64}      # accumulated squared gradients
    theta::Vector{Float64}  # current parameters
    t::Int
end

function AdaGrad(dim::Int; eta::Float64=0.01, eps::Float64=1e-8)
    AdaGrad(eta, eps, zeros(dim), zeros(dim), 0)
end

"""
    update!(opt::AdaGrad, grad::Vector{Float64}) -> Vector{Float64}

AdaGrad parameter update. Returns new parameters.
"""
function update!(opt::AdaGrad, grad::Vector{Float64})::Vector{Float64}
    opt.t += 1
    opt.G .+= grad .^ 2
    opt.theta .-= opt.eta ./ (sqrt.(opt.G) .+ opt.epsilon) .* grad
    copy(opt.theta)
end

"""
    Adam

Adam optimizer (Adaptive Moment Estimation).
"""
mutable struct Adam
    eta::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    m::Vector{Float64}   # first moment
    v::Vector{Float64}   # second moment
    theta::Vector{Float64}
    t::Int
end

function Adam(dim::Int; eta::Float64=1e-3, beta1::Float64=0.9,
              beta2::Float64=0.999, eps::Float64=1e-8)
    Adam(eta, beta1, beta2, eps, zeros(dim), zeros(dim), zeros(dim), 0)
end

"""
    update!(opt::Adam, grad::Vector{Float64}) -> Vector{Float64}

Adam update rule with bias correction.
"""
function update!(opt::Adam, grad::Vector{Float64})::Vector{Float64}
    opt.t += 1
    opt.m .= opt.beta1 .* opt.m .+ (1 - opt.beta1) .* grad
    opt.v .= opt.beta2 .* opt.v .+ (1 - opt.beta2) .* grad.^2
    m_hat = opt.m ./ (1 - opt.beta1^opt.t)
    v_hat = opt.v ./ (1 - opt.beta2^opt.t)
    opt.theta .-= opt.eta .* m_hat ./ (sqrt.(v_hat) .+ opt.epsilon)
    copy(opt.theta)
end

"""
    RMSprop

RMSprop optimizer (exponential moving average of squared gradients).
"""
mutable struct RMSprop
    eta::Float64
    rho::Float64
    epsilon::Float64
    E_g2::Vector{Float64}   # running mean of squared gradients
    theta::Vector{Float64}
    t::Int
end

function RMSprop(dim::Int; eta::Float64=1e-3, rho::Float64=0.9, eps::Float64=1e-8)
    RMSprop(eta, rho, eps, zeros(dim), zeros(dim), 0)
end

function update!(opt::RMSprop, grad::Vector{Float64})::Vector{Float64}
    opt.t += 1
    opt.E_g2 .= opt.rho .* opt.E_g2 .+ (1 - opt.rho) .* grad.^2
    opt.theta .-= opt.eta ./ (sqrt.(opt.E_g2) .+ opt.epsilon) .* grad
    copy(opt.theta)
end

"""
    online_sgd_regression(X, y; opt_type=:adam, ...) -> (weights, losses)

Online SGD regression with choice of optimizer. Returns per-step losses.
"""
function online_sgd_regression(X::Matrix{Float64}, y::Vector{Float64};
                                 opt_type::Symbol=:adam,
                                 n_epochs::Int=1,
                                 lambda::Float64=1e-4)
    n, d = size(X)
    opt = if opt_type == :adagrad
        AdaGrad(d; eta=0.01)
    elseif opt_type == :rmsprop
        RMSprop(d; eta=1e-3)
    else
        Adam(d; eta=1e-3)
    end

    losses = zeros(n * n_epochs)
    for ep in 1:n_epochs
        for t in 1:n
            x_t = X[t, :]
            y_t = y[t]
            pred = dot(opt.theta, x_t)
            err  = pred - y_t
            grad = err .* x_t .+ lambda .* opt.theta  # L2 regularization
            update!(opt, grad)
            losses[(ep-1)*n + t] = err^2
        end
    end
    (weights=copy(opt.theta), losses=losses)
end

# ─────────────────────────────────────────────────────────────
# 2. FTRL-PROXIMAL
# ─────────────────────────────────────────────────────────────

"""
    FTRLProximal

Follow-the-Regularized-Leader with L1+L2 regularization.
Widely used in large-scale online learning (Google's ad system).

The closed-form update for L2 (proximal):
  w_i = -(z_i - sign(z_i)*lambda1) / (beta_i/eta + lambda2)
  if |z_i| > lambda1, else 0
"""
mutable struct FTRLProximal
    alpha::Float64    # learning rate
    beta::Float64     # per-coordinate learning rate parameter
    lambda1::Float64  # L1 regularization (sparsity)
    lambda2::Float64  # L2 regularization
    z::Vector{Float64}  # accumulated gradient sum (with correction)
    n::Vector{Float64}  # accumulated squared gradients
    theta::Vector{Float64}
    t::Int
end

function FTRLProximal(dim::Int; alpha::Float64=0.1, beta::Float64=1.0,
                       lambda1::Float64=0.0, lambda2::Float64=1e-5)
    FTRLProximal(alpha, beta, lambda1, lambda2,
                 zeros(dim), zeros(dim), zeros(dim), 0)
end

"""
    ftrl_update!(ftrl, grad) -> Vector{Float64}

FTRL-Proximal update step. Returns new weights.
"""
function ftrl_update!(ftrl::FTRLProximal, grad::Vector{Float64})::Vector{Float64}
    ftrl.t += 1
    for i in eachindex(grad)
        g_i = grad[i]
        sigma_i = (sqrt(ftrl.n[i] + g_i^2) - sqrt(ftrl.n[i])) / ftrl.alpha
        ftrl.z[i] += g_i - sigma_i * ftrl.theta[i]
        ftrl.n[i] += g_i^2

        # Proximal step
        if abs(ftrl.z[i]) <= ftrl.lambda1
            ftrl.theta[i] = 0.0
        else
            sign_z = ftrl.z[i] > 0 ? 1.0 : -1.0
            denom = (ftrl.beta + sqrt(ftrl.n[i])) / ftrl.alpha + ftrl.lambda2
            ftrl.theta[i] = -(ftrl.z[i] - sign_z * ftrl.lambda1) / denom
        end
    end
    copy(ftrl.theta)
end

"""
    ftrl_sparsity_rate(ftrl::FTRLProximal) -> Float64

Fraction of zero weights (sparsity due to L1).
"""
ftrl_sparsity_rate(ftrl::FTRLProximal) = mean(ftrl.theta .== 0.0)

# ─────────────────────────────────────────────────────────────
# 3. HEDGE ALGORITHM (EXPONENTIAL WEIGHTS)
# ─────────────────────────────────────────────────────────────

"""
    HedgeAlgorithm

Multiplicative weights / Hedge algorithm for online expert aggregation.
Maintains a distribution over N experts, updates by exponential reweighting.
"""
mutable struct HedgeAlgorithm
    n_experts::Int
    eta::Float64            # learning rate
    weights::Vector{Float64}
    cumulative_loss::Vector{Float64}
    t::Int
end

function HedgeAlgorithm(n_experts::Int; eta::Float64=0.1)
    HedgeAlgorithm(n_experts, eta, fill(1.0/n_experts, n_experts),
                   zeros(n_experts), 0)
end

"""
    hedge_update!(h, expert_losses) -> Vector{Float64}

Update weights given vector of per-expert losses this round.
Returns new weight distribution.
"""
function hedge_update!(h::HedgeAlgorithm, expert_losses::Vector{Float64})::Vector{Float64}
    h.t += 1
    h.cumulative_loss .+= expert_losses
    # Multiplicative update
    h.weights .*= exp.(-h.eta .* expert_losses)
    # Normalize
    h.weights ./= sum(h.weights)
    copy(h.weights)
end

"""
    hedge_weights(h::HedgeAlgorithm) -> Vector{Float64}

Current mixture weights over experts.
"""
hedge_weights(h::HedgeAlgorithm) = copy(h.weights)

"""
    hedge_predict(h, expert_predictions) -> Float64

Weighted combination of expert predictions.
"""
hedge_predict(h::HedgeAlgorithm, preds::Vector{Float64}) = dot(h.weights, preds)

"""
    hedge_regret(h::HedgeAlgorithm, total_mixture_loss::Float64) -> Float64

Hedge regret bound: R ≤ ln(N)/η + η*T/8 (for [0,1] losses).
"""
function hedge_regret_bound(h::HedgeAlgorithm)::Float64
    log(h.n_experts) / h.eta + h.eta * h.t / 8.0
end

# ─────────────────────────────────────────────────────────────
# 4. ONLINE-TO-BATCH CONVERSION
# ─────────────────────────────────────────────────────────────

"""
    online_to_batch(weight_history::Matrix{Float64}) -> Vector{Float64}

Convert sequence of online weight vectors to a single batch predictor
by averaging: θ_avg = (1/T) Σ θ_t.
"""
function online_to_batch(weight_history::Matrix{Float64})::Vector{Float64}
    vec(mean(weight_history, dims=2))
end

"""
    suffix_averaging(weight_history, burn_in) -> Vector{Float64}

Average over last (T - burn_in) rounds (often better in practice).
"""
function suffix_averaging(weight_history::Matrix{Float64}, burn_in::Int)::Vector{Float64}
    T = size(weight_history, 2)
    vec(mean(weight_history[:, burn_in+1:end], dims=2))
end

# ─────────────────────────────────────────────────────────────
# 5. SECOND-ORDER ONLINE — DIAGONAL ONS (ONLINE NEWTON STEP)
# ─────────────────────────────────────────────────────────────

"""
    DiagonalONS

Diagonal Online Newton Step approximation.
Uses diagonal of inverse Hessian estimate (D-BFGS style).
"""
mutable struct DiagonalONS
    dim::Int
    eps::Float64
    H_inv_diag::Vector{Float64}  # diagonal inverse Hessian approx
    theta::Vector{Float64}
    s::Vector{Float64}   # gradient sum
    y_sum::Vector{Float64}  # curvature accumulator
    t::Int
end

function DiagonalONS(dim::Int; eps::Float64=1.0)
    DiagonalONS(dim, eps, ones(dim) * eps, zeros(dim), zeros(dim), zeros(dim), 0)
end

"""
    ons_update!(ons, grad, curvature_hint) -> Vector{Float64}

Diagonal ONS update. curvature_hint is squared gradient (diagonal Hessian approx).
"""
function ons_update!(ons::DiagonalONS, grad::Vector{Float64},
                      curvature::Vector{Float64}=grad.^2)::Vector{Float64}
    ons.t += 1
    # Update diagonal Hessian approximation
    ons.H_inv_diag .= 1.0 ./ (ons.eps .+ sqrt.(cumsum(curvature) ./ ons.t))
    # Newton step
    ons.theta .-= ons.H_inv_diag .* grad
    copy(ons.theta)
end

# ─────────────────────────────────────────────────────────────
# 6. ADWIN — ADAPTIVE WINDOWING FOR CONCEPT DRIFT DETECTION
# ─────────────────────────────────────────────────────────────

"""
    ADWIN

ADWIN (Adaptive WINdowing) algorithm for concept drift detection.
Maintains a variable-length window and detects when distribution shifts.

Reference: Bifet & Gavaldà (2007)
"""
mutable struct ADWIN
    delta::Float64         # confidence parameter (false positive rate)
    window::Vector{Float64}
    total::Float64
    variance::Float64
    n::Int
    drift_detected::Bool
    last_drift_idx::Int
    t::Int
end

ADWIN(; delta::Float64=0.002) = ADWIN(delta, Float64[], 0.0, 0.0, 0, false, 0, 0)

"""
    adwin_update!(adwin, x) -> Bool

Add new observation x to window. Returns true if drift detected.
ADWIN tests all splits of the window for significant mean difference.
"""
function adwin_update!(adwin::ADWIN, x::Float64)::Bool
    adwin.t += 1
    push!(adwin.window, x)
    adwin.total += x
    adwin.n     += 1

    # Update variance incrementally (Welford)
    if adwin.n > 1
        delta_x  = x - adwin.total / adwin.n
        adwin.variance += delta_x * (x - adwin.total / adwin.n)
    end

    adwin.drift_detected = false

    # Test for drift: try all split points
    n_w = length(adwin.window)
    n_w < 4 && return false

    mean_all = adwin.total / adwin.n
    drift_idx = -1

    # Efficient O(log N) approach: check geometrically spaced cut points
    cut_step = max(1, n_w ÷ 10)
    for cut in cut_step:cut_step:(n_w - cut_step)
        n0 = cut; n1 = n_w - cut
        (n0 < 2 || n1 < 2) && continue

        mean0 = mean(adwin.window[1:cut])
        mean1 = mean(adwin.window[cut+1:end])

        # Variance estimate
        var_all = adwin.n > 1 ? adwin.variance / (adwin.n - 1) : 1.0
        var_all = max(var_all, 1e-10)

        # ADWIN epsilon_cut: threshold for significant difference
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        epsilon_cut = sqrt(var_all / (2m) * log(4 * n_w / adwin.delta))

        if abs(mean0 - mean1) > epsilon_cut
            adwin.drift_detected = true
            drift_idx = cut
            break
        end
    end

    if adwin.drift_detected && drift_idx > 0
        # Remove old portion of window
        adwin.last_drift_idx = adwin.t
        old_part = adwin.window[1:drift_idx]
        adwin.window = adwin.window[drift_idx+1:end]
        adwin.total -= sum(old_part)
        adwin.n     -= length(old_part)
        adwin.variance = var(adwin.window) * max(adwin.n - 1, 1)
    end

    adwin.drift_detected
end

"""
    adwin_detect(series::Vector{Float64}; delta=0.002) -> Vector{Int}

Run ADWIN on a time series, return indices where drift was detected.
"""
function adwin_detect(series::Vector{Float64}; delta::Float64=0.002)::Vector{Int}
    adwin = ADWIN(; delta=delta)
    drift_points = Int[]
    for (t, x) in enumerate(series)
        if adwin_update!(adwin, x)
            push!(drift_points, t)
        end
    end
    drift_points
end

# ─────────────────────────────────────────────────────────────
# 7. CONTEXTUAL BANDITS — LinUCB
# ─────────────────────────────────────────────────────────────

"""
    LinUCB

LinUCB contextual bandit algorithm for online signal selection.
Maintains a ridge regression model per arm (signal source).

Reference: Li et al. (2010), "A Contextual-Bandit Approach to Personalized News"
"""
mutable struct LinUCB
    n_arms::Int
    d::Int          # context dimension
    alpha::Float64  # exploration bonus
    A::Vector{Matrix{Float64}}  # d×d design matrices
    b::Vector{Vector{Float64}}  # d-vectors
    theta::Vector{Vector{Float64}}  # estimated parameters per arm
    pulls::Vector{Int}
    rewards::Vector{Float64}
    t::Int
end

function LinUCB(n_arms::Int, d::Int; alpha::Float64=1.0)
    A     = [Matrix{Float64}(I, d, d) for _ in 1:n_arms]
    b     = [zeros(d) for _ in 1:n_arms]
    theta = [zeros(d) for _ in 1:n_arms]
    LinUCB(n_arms, d, alpha, A, b, theta, zeros(Int, n_arms),
           zeros(n_arms), 0)
end

"""
    linucb_select!(ucb, context) -> Int

Select arm with highest UCB score given context vector.
"""
function linucb_select!(ucb::LinUCB, context::Vector{Float64})::Int
    ucb.t += 1
    best_arm = 1
    best_ucb = -Inf

    for a in 1:ucb.n_arms
        A_inv = inv(ucb.A[a])
        ucb.theta[a] = A_inv * ucb.b[a]
        # UCB = θ'x + alpha * sqrt(x' A^{-1} x)
        mu    = dot(ucb.theta[a], context)
        bonus = ucb.alpha * sqrt(max(dot(context, A_inv * context), 0.0))
        score = mu + bonus
        if score > best_ucb
            best_ucb = score
            best_arm = a
        end
    end
    best_arm
end

"""
    linucb_update!(ucb, arm, context, reward)

Update arm model with observed reward.
"""
function linucb_update!(ucb::LinUCB, arm::Int, context::Vector{Float64},
                         reward::Float64)
    ucb.A[arm] .+= context * context'
    ucb.b[arm] .+= reward .* context
    ucb.pulls[arm] += 1
    ucb.rewards[arm] = (ucb.rewards[arm] * (ucb.pulls[arm]-1) + reward) /
                        ucb.pulls[arm]
end

"""
    run_bandit_simulation(signals, returns; alpha=1.0) -> NamedTuple

Simulate LinUCB selecting among signal sources, reward = sign(signal)*return.
"""
function run_bandit_simulation(signals::Matrix{Float64},
                                returns::Vector{Float64};
                                alpha::Float64=1.0)
    T, n_arms = size(signals)
    d = 3  # context: [recent return, recent vol, time-of-day proxy]
    ucb = LinUCB(n_arms, d; alpha=alpha)

    cumulative_reward = 0.0
    arm_counts   = zeros(Int, n_arms)
    rewards_hist = zeros(T)

    for t in 1:T
        # Context: simple features
        context = Float64[
            t > 5 ? mean(returns[max(1,t-5):t-1]) : 0.0,
            t > 5 ? std(returns[max(1,t-5):t-1])  : 0.01,
            sin(2π * (t % 24) / 24)   # intraday cycle proxy
        ]

        arm = linucb_select!(ucb, context)
        arm_counts[arm] += 1

        # Reward: profit from following signal
        signal   = signals[t, arm]
        reward   = sign(signal) * returns[t]
        linucb_update!(ucb, arm, context, reward)
        cumulative_reward += reward
        rewards_hist[t]    = reward
    end

    (cumulative_reward=cumulative_reward, arm_counts=arm_counts,
     avg_reward=cumulative_reward/T, rewards=rewards_hist)
end

# ─────────────────────────────────────────────────────────────
# 8. ONLINE PORTFOLIO SELECTION
# ─────────────────────────────────────────────────────────────

"""
    UniversalPortfolio

Cover's Universal Portfolio algorithm.
Maintains a mixture over constant rebalanced portfolios (CRPs).
Discretizes the simplex and tracks wealth of each CRP.
"""
mutable struct UniversalPortfolio
    n_assets::Int
    n_samples::Int   # number of simplex sample points
    portfolio_grid::Matrix{Float64}   # n_samples × n_assets
    portfolio_wealth::Vector{Float64}
    current_weights::Vector{Float64}
    t::Int
    rng::AbstractRNG
end

function UniversalPortfolio(n_assets::Int; n_samples::Int=1000,
                              rng=MersenneTwister(42))
    # Sample random portfolios from simplex (Dirichlet)
    grid = zeros(n_samples, n_assets)
    for i in 1:n_samples
        x = -log.(rand(rng, n_assets))
        grid[i, :] = x ./ sum(x)
    end
    UP = UniversalPortfolio(n_assets, n_samples, grid,
                             ones(n_samples),  # initial wealth all portfolios
                             fill(1.0/n_assets, n_assets), 0, rng)
    UP
end

"""
    portfolio_step!(up::UniversalPortfolio, price_relatives) -> Vector{Float64}

Update Universal Portfolio with price relatives (today_price / yesterday_price).
Returns new portfolio weights.
"""
function portfolio_step!(up::UniversalPortfolio,
                          price_relatives::Vector{Float64})::Vector{Float64}
    up.t += 1
    # Update wealth of each sampled portfolio
    for i in 1:up.n_samples
        w_i = up.portfolio_grid[i, :]
        return_i = dot(w_i, price_relatives)
        up.portfolio_wealth[i] *= max(return_i, 1e-10)
    end
    # New weights: wealth-weighted average of portfolio allocations
    total_wealth = sum(up.portfolio_wealth)
    up.current_weights = vec(sum(up.portfolio_grid .*
                                  up.portfolio_wealth, dims=1)) ./ total_wealth
    up.current_weights ./= sum(up.current_weights)
    copy(up.current_weights)
end

portfolio_weights(up::UniversalPortfolio) = copy(up.current_weights)

"""
    OLMAR

Online Moving Average Reversion (Li & Hoi 2012).
Predicts mean-reversion and allocates to assets expected to rise.
"""
mutable struct OLMAR
    n_assets::Int
    window::Int       # look-back for price prediction
    eps::Float64      # return threshold for reversion
    weights::Vector{Float64}
    price_history::Matrix{Float64}  # window × n_assets
    t::Int
end

function OLMAR(n_assets::Int; window::Int=5, eps::Float64=10.0)
    OLMAR(n_assets, window, eps,
          fill(1.0/n_assets, n_assets),
          zeros(window, n_assets), 0)
end

"""
    portfolio_step!(olmar::OLMAR, prices) -> Vector{Float64}

OLMAR update: predict next price relatives via moving average reversion.
"""
function portfolio_step!(olmar::OLMAR, prices::Vector{Float64})::Vector{Float64}
    olmar.t += 1
    # Update price history
    olmar.price_history = circshift(olmar.price_history, (1, 0))
    olmar.price_history[1, :] = prices

    olmar.t < olmar.window && return copy(olmar.weights)

    # Moving average prediction: x̂_{t+1} = MA_t / p_t
    ma = vec(mean(olmar.price_history, dims=1))
    x_hat = ma ./ (prices .+ 1e-10)

    # Update portfolio via projection onto simplex (l2 projection)
    x_bar = mean(x_hat)
    if dot(x_hat, olmar.weights) <= olmar.eps
        # Compute lambda (Lagrange multiplier) for simplex projection
        lambda = (dot(x_hat, olmar.weights) - olmar.eps) /
                  (dot(x_hat .- x_bar, x_hat .- x_bar) + 1e-10)
        new_w = olmar.weights .- lambda .* (x_hat .- x_bar)
        # Project onto simplex
        new_w = simplex_project(new_w)
    else
        new_w = copy(olmar.weights)
    end
    olmar.weights = new_w
    copy(olmar.weights)
end

"""Project vector onto probability simplex."""
function simplex_project(v::Vector{Float64})::Vector{Float64}
    n = length(v)
    u = sort(v, rev=true)
    cssv = cumsum(u)
    rho = findlast(i -> u[i] - (cssv[i] - 1.0) / i > 0, 1:n)
    rho = rho === nothing ? n : rho
    theta = (cssv[rho] - 1.0) / rho
    max.(v .- theta, 0.0)
end

"""
    PAMR

Passive Aggressive Mean Reversion (Li et al. 2012).
Updates portfolio to exploit mean reversion with passive-aggressive constraint.
"""
mutable struct PAMR
    n_assets::Int
    eps::Float64      # sensitivity parameter
    C::Float64        # regularization (PAMR-2 variant)
    weights::Vector{Float64}
    t::Int
end

PAMR(n_assets::Int; eps::Float64=0.5, C::Float64=500.0) =
    PAMR(n_assets, eps, C, fill(1.0/n_assets, n_assets), 0)

"""
    portfolio_step!(pamr::PAMR, price_relatives) -> Vector{Float64}

PAMR-2 update step.
"""
function portfolio_step!(pamr::PAMR, price_relatives::Vector{Float64})::Vector{Float64}
    pamr.t += 1
    x = price_relatives
    x_bar = mean(x)
    loss = max(dot(pamr.weights, x) - pamr.eps, 0.0)
    denom = dot(x .- x_bar, x .- x_bar) + 1.0 / (2pamr.C)
    tau  = loss / (denom + 1e-10)
    new_w = pamr.weights .- tau .* (x .- x_bar)
    pamr.weights = simplex_project(new_w)
    copy(pamr.weights)
end

portfolio_weights(p::Union{OLMAR, PAMR}) = copy(p.weights)

# ─────────────────────────────────────────────────────────────
# 9. REGRET ANALYSIS
# ─────────────────────────────────────────────────────────────

"""
    OnlineRegretTracker

Tracks cumulative regret against a benchmark (best fixed decision in hindsight).
"""
mutable struct OnlineRegretTracker
    learner_losses::Vector{Float64}
    oracle_losses::Vector{Float64}
    cumulative_regret::Vector{Float64}
    t::Int
end

OnlineRegretTracker() = OnlineRegretTracker(Float64[], Float64[], Float64[], 0)

"""
    track_regret!(tracker, learner_loss, oracle_loss)

Add one round of losses and update cumulative regret.
"""
function track_regret!(tracker::OnlineRegretTracker,
                        learner_loss::Float64, oracle_loss::Float64)
    tracker.t += 1
    push!(tracker.learner_losses, learner_loss)
    push!(tracker.oracle_losses,  oracle_loss)
    cum = tracker.t > 1 ? tracker.cumulative_regret[end] : 0.0
    push!(tracker.cumulative_regret, cum + learner_loss - oracle_loss)
end

"""
    regret_summary(tracker::OnlineRegretTracker) -> NamedTuple

Summary statistics for regret analysis.
"""
function regret_summary(tracker::OnlineRegretTracker)
    T = tracker.t
    T == 0 && return (total_regret=0.0, avg_regret=0.0, regret_rate=0.0)
    total    = tracker.cumulative_regret[end]
    avg      = total / T
    # Theoretical bound for Hedge: O(sqrt(T * ln N))
    # We report empirical rate
    rate     = total / sqrt(T)
    (total_regret=total, avg_regret_per_round=avg, regret_per_sqrt_T=rate,
     total_learner_loss=sum(tracker.learner_losses),
     total_oracle_loss=sum(tracker.oracle_losses))
end

# ─────────────────────────────────────────────────────────────
# 10. UTILITIES & SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────

"""Generate synthetic trading signals with known drift."""
function generate_signals(n::Int=500, n_signals::Int=5;
                           rng=MersenneTwister(42))
    signals = randn(rng, n, n_signals)
    # One signal has slight predictive power
    signals[:, 1] .*= 0.5
    signals
end

"""Generate drift-then-shift returns (for ADWIN testing)."""
function generate_drift_returns(n::Int=500; rng=MersenneTwister(1))
    rets = zeros(n)
    for t in 1:n
        mu = t < 250 ? 0.001 : -0.002  # regime shift at t=250
        rets[t] = mu + 0.02 * randn(rng)
    end
    rets
end

# ─────────────────────────────────────────────────────────────
# 11. FULL DEMO
# ─────────────────────────────────────────────────────────────

"""
    run_online_learning_demo() -> Nothing

Demonstration of all online learning algorithms.
"""
function run_online_learning_demo()
    println("=" ^ 60)
    println("ONLINE LEARNING DEMO")
    println("=" ^ 60)
    rng = MersenneTwister(42)
    n = 500

    # --- Synthetic data ---
    X = randn(rng, n, 10)
    true_w = randn(rng, 10)
    y = X * true_w .+ 0.1 .* randn(rng, n)

    # 1. Online SGD optimizers
    println("\n1. Online Regression — Optimizer Comparison")
    for opt in [:adam, :adagrad, :rmsprop]
        res = online_sgd_regression(X, y; opt_type=opt, n_epochs=3)
        final_loss = mean(res.losses[end-49:end])
        coserr = norm(res.weights - true_w) / norm(true_w)
        println("  $opt: final_mse=$(round(final_loss,digits=4)), weight_err=$(round(coserr,digits=3))")
    end

    # 2. FTRL-Proximal
    println("\n2. FTRL-Proximal (sparse regularization)")
    ftrl = FTRLProximal(10; lambda1=0.01, lambda2=1e-4)
    for t in 1:n
        x_t = X[t, :]
        pred = dot(ftrl.theta, x_t)
        err  = pred - y[t]
        grad = err .* x_t
        ftrl_update!(ftrl, grad)
    end
    println("  Sparsity: $(round(ftrl_sparsity_rate(ftrl)*100,digits=1))% zeros")
    println("  Weight error: $(round(norm(ftrl.theta - true_w)/norm(true_w),digits=3))")

    # 3. Hedge Algorithm
    println("\n3. Hedge Algorithm (Expert Aggregation)")
    n_experts = 5
    hedge = HedgeAlgorithm(n_experts; eta=0.1)
    rets  = generate_drift_returns(n; rng=rng)
    # Experts: different sign/scale of momentum signal
    expert_preds = randn(rng, n, n_experts) .* 0.01
    expert_preds[:, 1] .+= rets .* 0.5  # expert 1 has weak signal

    tracker = OnlineRegretTracker()
    for t in 1:n
        pred  = hedge_predict(hedge, expert_preds[t, :])
        errs  = (expert_preds[t, :] .- rets[t]).^2
        hedge_update!(hedge, errs)
        best_loss = minimum((expert_preds[t, :] .- rets[t]).^2)
        learner_loss = (pred - rets[t])^2
        track_regret!(tracker, learner_loss, best_loss)
    end
    rs = regret_summary(tracker)
    println("  Total Regret:      $(round(rs.total_regret,digits=4))")
    println("  Regret/√T:         $(round(rs.regret_per_sqrt_T,digits=4))")
    println("  Regret Bound (theory): $(round(hedge_regret_bound(hedge),digits=4))")

    # 4. ADWIN Drift Detection
    println("\n4. ADWIN Concept Drift Detection")
    drift_rets = generate_drift_returns(n; rng=rng)
    drift_pts  = adwin_detect(drift_rets; delta=0.002)
    println("  Drift detected at steps: $(drift_pts)")
    println("  (True drift at step 250)")

    # 5. LinUCB
    println("\n5. LinUCB Contextual Bandit (Signal Selection)")
    signals = generate_signals(n, 4; rng=rng)
    bsim = run_bandit_simulation(signals, rets; alpha=1.0)
    println("  Cumulative Reward:  $(round(bsim.cumulative_reward,digits=4))")
    println("  Avg Reward/step:    $(round(bsim.avg_reward,digits=5))")
    println("  Arm selection counts: $(bsim.arm_counts)")

    # 6. Online Portfolio Selection
    println("\n6. Online Portfolio Selection")
    n_assets = 3
    prices_raw = cumsum(randn(rng, n, n_assets) .* 0.01, dims=1)
    prices_mat = exp.(prices_raw) .* 100.0
    # Compute price relatives
    price_rels = [prices_mat[t, :] ./ prices_mat[max(1,t-1), :] for t in 2:n]

    # Universal Portfolio
    up = UniversalPortfolio(n_assets; n_samples=200, rng=rng)
    # OLMAR
    olmar_p = OLMAR(n_assets; window=5)
    # PAMR
    pamr_p  = PAMR(n_assets; eps=0.5, C=500.0)

    up_wealth   = [1.0]
    olmar_wealth = [1.0]
    pamr_wealth  = [1.0]

    for pr in price_rels
        w_up    = portfolio_step!(up, pr)
        w_olmar = portfolio_step!(olmar_p, prices_mat[length(up_wealth)+1, :])
        w_pamr  = portfolio_step!(pamr_p, pr)

        push!(up_wealth,    up_wealth[end]   * dot(w_up,    pr))
        push!(olmar_wealth, olmar_wealth[end] * dot(w_olmar, pr))
        push!(pamr_wealth,  pamr_wealth[end]  * dot(w_pamr,  pr))
    end

    println("  Universal Portfolio final wealth: $(round(up_wealth[end],digits=4))")
    println("  OLMAR final wealth:               $(round(olmar_wealth[end],digits=4))")
    println("  PAMR final wealth:                $(round(pamr_wealth[end],digits=4))")
    println("  Buy-and-hold (equal weight):      $(round(dot(price_rels[end], fill(1/n_assets, n_assets)),digits=4)) last-period return")

    println("\nDone.")
    nothing
end

# ─────────────────────────────────────────────────────────────
# 12. SECOND-ORDER PORTFOLIO METHODS
# ─────────────────────────────────────────────────────────────

"""
    EWA

Exponentiated Weighted Aggregation with second-order correction.
Combines Hedge with variance reduction via Newton-step correction.
"""
mutable struct EWA
    n_experts::Int
    eta::Float64
    weights::Vector{Float64}
    A::Matrix{Float64}   # accumulated outer products (second-order)
    t::Int
end

EWA(n::Int; eta::Float64=0.1) =
    EWA(n, eta, fill(1.0/n, n), diagm(fill(1.0, n)), 0)

"""
    ewa_update!(ewa, losses) -> Vector{Float64}

EWA update with second-order correction.
"""
function ewa_update!(ewa::EWA, losses::Vector{Float64})::Vector{Float64}
    ewa.t += 1
    # Standard exponential weights update
    ewa.weights .*= exp.(-ewa.eta .* losses)
    ewa.weights ./= sum(ewa.weights)
    copy(ewa.weights)
end

# ─────────────────────────────────────────────────────────────
# 13. ONLINE FORECASTING COMBINATION
# ─────────────────────────────────────────────────────────────

"""
    BatesGrangerCombination

Online Bates-Granger forecast combination.
Weights proportional to inverse mean squared errors.
"""
mutable struct BatesGrangerCombination
    n_forecasters::Int
    cum_mse::Vector{Float64}
    weights::Vector{Float64}
    t::Int
end

BatesGrangerCombination(n::Int) =
    BatesGrangerCombination(n, ones(n), fill(1.0/n, n), 0)

"""
    bg_update!(bg, forecasts, actual) -> Vector{Float64}

Update Bates-Granger weights with new observation.
"""
function bg_update!(bg::BatesGrangerCombination,
                     forecasts::Vector{Float64}, actual::Float64)::Vector{Float64}
    bg.t += 1
    errors = (forecasts .- actual).^2
    # Running average MSE
    bg.cum_mse .= (bg.cum_mse .* (bg.t-1) .+ errors) ./ bg.t
    # Weights ∝ 1/MSE
    inv_mse = 1.0 ./ (bg.cum_mse .+ 1e-10)
    bg.weights = inv_mse ./ sum(inv_mse)
    copy(bg.weights)
end

bg_predict(bg::BatesGrangerCombination, forecasts::Vector{Float64}) = dot(bg.weights, forecasts)

# ─────────────────────────────────────────────────────────────
# 14. ONLINE RIDGE REGRESSION WITH FORGETTING
# ─────────────────────────────────────────────────────────────

"""
    ForgettingRidge

Online ridge regression with exponential forgetting factor.
Tracks A = Σ λ^(t-s) x_s x_s' and b = Σ λ^(t-s) x_s y_s.
"""
mutable struct ForgettingRidge
    dim::Int
    lambda_forget::Float64  # forgetting factor (0.99 = slow, 0.9 = fast)
    ridge::Float64
    A::Matrix{Float64}
    b::Vector{Float64}
    theta::Vector{Float64}
    t::Int
end

function ForgettingRidge(dim::Int; lambda::Float64=0.99, ridge::Float64=1e-3)
    ForgettingRidge(dim, lambda, ridge,
                    ridge * Matrix{Float64}(I, dim, dim), zeros(dim), zeros(dim), 0)
end

"""
    forgetting_ridge_update!(fr, x, y) -> Vector{Float64}

Update forgetting ridge with new (x, y) pair.
"""
function forgetting_ridge_update!(fr::ForgettingRidge,
                                    x::Vector{Float64}, y::Float64)::Vector{Float64}
    fr.t += 1
    fr.A = fr.lambda_forget .* fr.A .+ x * x'
    fr.b = fr.lambda_forget .* fr.b .+ y .* x
    fr.theta = (fr.A + fr.ridge * I) \ fr.b
    copy(fr.theta)
end

predict_forgetting(fr::ForgettingRidge, x::Vector{Float64}) = dot(fr.theta, x)

# ─────────────────────────────────────────────────────────────
# 15. ONLINE COVARIANCE ESTIMATION
# ─────────────────────────────────────────────────────────────

"""
    OnlineCovarianceEstimator

Welford-style online covariance estimation.
Numerically stable incremental updates.
"""
mutable struct OnlineCovarianceEstimator
    n::Int
    dim::Int
    mean::Vector{Float64}
    M2::Matrix{Float64}  # sum of outer products of deviations
end

OnlineCovarianceEstimator(dim::Int) =
    OnlineCovarianceEstimator(0, dim, zeros(dim), zeros(dim, dim))

"""
    update_cov!(est, x) -> Nothing

Update online covariance with new observation x.
"""
function update_cov!(est::OnlineCovarianceEstimator, x::Vector{Float64})
    est.n += 1
    delta  = x .- est.mean
    est.mean .+= delta ./ est.n
    delta2  = x .- est.mean
    est.M2 .+= delta * delta2'
end

"""
    current_cov(est) -> Matrix{Float64}

Get current covariance estimate.
"""
function current_cov(est::OnlineCovarianceEstimator)::Matrix{Float64}
    est.n < 2 && return Matrix{Float64}(I, est.dim, est.dim)
    est.M2 ./ (est.n - 1)
end

# ─────────────────────────────────────────────────────────────
# 16. MULTI-ARMED BANDIT COMPARISON
# ─────────────────────────────────────────────────────────────

"""
    UCB1

UCB1 bandit algorithm. Selects arm maximizing μ̂_a + sqrt(2*ln(t)/n_a).
"""
mutable struct UCB1
    n_arms::Int
    counts::Vector{Int}
    values::Vector{Float64}
    t::Int
end

UCB1(n::Int) = UCB1(n, zeros(Int, n), zeros(n), 0)

function ucb1_select(ucb::UCB1)::Int
    ucb.t += 1
    # Pull each arm at least once
    for a in 1:ucb.n_arms
        ucb.counts[a] == 0 && return a
    end
    ucb_scores = ucb.values .+ sqrt.(2 * log(ucb.t) ./ (ucb.counts .+ 1))
    argmax(ucb_scores)
end

function ucb1_update!(ucb::UCB1, arm::Int, reward::Float64)
    ucb.counts[arm] += 1
    n = ucb.counts[arm]
    ucb.values[arm] += (reward - ucb.values[arm]) / n
end

"""
    ThompsonSampling

Thompson Sampling for Bernoulli bandits (Beta-Binomial model).
"""
mutable struct ThompsonSampling
    n_arms::Int
    alpha::Vector{Float64}  # successes + 1
    beta_v::Vector{Float64} # failures + 1
    rng::AbstractRNG
end

ThompsonSampling(n::Int; rng=MersenneTwister(1)) =
    ThompsonSampling(n, ones(n), ones(n), rng)

function ts_select(ts::ThompsonSampling)::Int
    # Sample from Beta(alpha_a, beta_a) for each arm
    # Approximate Beta sampling via Normal approximation for efficiency
    samples = [ts.alpha[a]/(ts.alpha[a]+ts.beta_v[a]) +
               sqrt(ts.alpha[a]*ts.beta_v[a] /
                    ((ts.alpha[a]+ts.beta_v[a])^2 * (ts.alpha[a]+ts.beta_v[a]+1))) *
               randn(ts.rng) for a in 1:ts.n_arms]
    argmax(samples)
end

function ts_update!(ts::ThompsonSampling, arm::Int, reward::Float64)
    r = reward > 0.5 ? 1 : 0
    ts.alpha[arm] += r
    ts.beta_v[arm] += (1 - r)
end

"""
    compare_bandits(true_rewards, n_arms; n_rounds=500, rng=...) -> NamedTuple

Compare UCB1, Thompson Sampling, and LinUCB on a simulated bandit problem.
"""
function compare_bandits(true_means::Vector{Float64}, n_rounds::Int=500;
                          rng=MersenneTwister(42))
    n_arms = length(true_means)
    opt_arm = argmax(true_means)
    opt_reward = true_means[opt_arm]

    ucb = UCB1(n_arms)
    ts  = ThompsonSampling(n_arms; rng=rng)

    ucb_cum = zeros(n_rounds); ts_cum = zeros(n_rounds)
    ucb_reg = zeros(n_rounds); ts_reg = zeros(n_rounds)

    for t in 1:n_rounds
        # UCB1
        a_ucb = ucb1_select(ucb)
        r_ucb = true_means[a_ucb] + 0.1*randn(rng)
        ucb1_update!(ucb, a_ucb, r_ucb)
        ucb_cum[t] = (t > 1 ? ucb_cum[t-1] : 0.0) + r_ucb
        ucb_reg[t] = (t > 1 ? ucb_reg[t-1] : 0.0) + (opt_reward - true_means[a_ucb])

        # Thompson Sampling
        a_ts = ts_select(ts)
        r_ts = true_means[a_ts] + 0.1*randn(rng)
        ts_update!(ts, a_ts, r_ts)
        ts_cum[t] = (t > 1 ? ts_cum[t-1] : 0.0) + r_ts
        ts_reg[t] = (t > 1 ? ts_reg[t-1] : 0.0) + (opt_reward - true_means[a_ts])
    end

    (ucb1_cumulative=ucb_cum, ts_cumulative=ts_cum,
     ucb1_regret=ucb_reg, ts_regret=ts_reg,
     ucb1_arm_counts=ucb.counts, ts_arm_counts=ts.alpha .+ ts.beta_v .- 2)
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 14 – Online Portfolio Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

"""
    online_sharpe(cumulative_returns, window)

Rolling Sharpe ratio of an online portfolio strategy using a sliding window.
"""
function online_sharpe(cum_returns::Vector{Float64}, window::Int=50)
    n  = length(cum_returns)
    rets = diff(vcat(0.0, cum_returns))
    rs   = fill(NaN, n)
    for i in (window+1):n
        r  = rets[i-window+1:i]
        sg = std(r)
        rs[i] = sg < 1e-8 ? 0.0 : mean(r) / sg * sqrt(252)
    end
    return rs
end

"""
    online_max_drawdown(cum_returns)

Maximum drawdown of a cumulative return series.
"""
function online_max_drawdown(cum_returns::Vector{Float64})
    peak = -Inf; mdd = 0.0
    for r in cum_returns
        peak = max(peak, r)
        mdd  = max(mdd, peak - r)
    end
    return mdd
end

"""
    strategy_comparison_table(strategy_names, cum_returns_matrix)

Print a comparison table: final return, Sharpe, max drawdown per strategy.
`cum_returns_matrix`: T × n_strategies
"""
function strategy_comparison_table(names::Vector{String},
                                    cum_rets::Matrix{Float64})
    T, n = size(cum_rets)
    println("\n" * "=" ^ 65)
    println("Online Strategy Comparison")
    println("=" ^ 65)
    @printf("%-25s  %10s  %10s  %10s\n", "Strategy", "Return", "Sharpe", "MaxDD")
    println("-" ^ 65)
    for j in 1:n
        cr  = cum_rets[:, j]
        ret = cr[end]
        sh  = begin
            rs = online_sharpe(cr)
            mean(filter(!isnan, rs))
        end
        md  = online_max_drawdown(cr)
        @printf("%-25s  %10.4f  %10.4f  %10.4f\n", names[j], ret, sh, md)
    end
    println("=" ^ 65)
end

end  # module OnlineLearning
