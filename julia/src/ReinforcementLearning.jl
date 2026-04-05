"""
ReinforcementLearning.jl — Composable RL Module for Crypto Trading

Exports:
  - TradingEnvironment: standard state/action/reward interface
  - QNetwork, PolicyNetwork, ValueNetwork: matrix-based neural network structs
  - Replay buffer, batch sampling, gradient updates
  - Training infrastructure: multiple algorithm support
  - Evaluation: Sharpe, max drawdown, trade count
  - Hyperparameter search (grid + random)
  - Policy visualization: action distribution by market state

Pure Julia stdlib only.
"""
module ReinforcementLearning

using Statistics, LinearAlgebra, Random

export TradingEnvironment, reset!, env_step!
export QNetwork, PolicyNetwork, ValueNetwork
export forward_pass, backward_pass!, network_copy!, network_update!
export ReplayBuffer, add_experience!, sample_batch
export DQNTrainer, PPOTrainer, SACTrainer
export train_dqn!, train_reinforce!, train_a2c!
export HyperparamSearch, grid_search, random_search
export evaluate_policy, compute_sharpe, compute_max_drawdown
export policy_action_distribution, feature_importance
export run_rl_module_demo

# ─────────────────────────────────────────────────────────────
# 1. ENVIRONMENT INTERFACE
# ─────────────────────────────────────────────────────────────

"""
    TradingEnvironment

Standardized crypto trading environment.
State: [return, vol_5, vol_20, rsi, macd, bb_signal, volume_ratio, position, drawdown]
Actions: 0 = flat/sell, 1 = hold, 2 = buy long
Reward: risk-adjusted PnL increment
"""
mutable struct TradingEnvironment
    prices::Vector{Float64}
    volumes::Vector{Float64}
    log_returns::Vector{Float64}
    n::Int
    t::Int
    window::Int
    position::Float64
    portfolio::Float64
    peak::Float64
    tc::Float64
    done::Bool
    state_dim::Int
    n_actions::Int
    portfolio_history::Vector{Float64}
    action_history::Vector{Int}
    reward_history::Vector{Float64}
end

function TradingEnvironment(prices::Vector{Float64};
                             volumes::Union{Vector{Float64},Nothing}=nothing,
                             tc::Float64=0.001, window::Int=20,
                             capital::Float64=10_000.0)
    lr = [0.0; diff(log.(max.(prices, 1e-10)))]
    vols = isnothing(volumes) ? fill(1.0, length(prices)) : volumes
    n = length(prices)
    TradingEnvironment(prices, vols, lr, n, window+1, window,
                       0.0, capital, capital, tc, false, 9, 3,
                       Float64[], Int[], Float64[])
end

"""
    reset!(env) -> Vector{Float64}

Reset environment, return initial state vector.
"""
function reset!(env::TradingEnvironment)::Vector{Float64}
    env.t         = env.window + 1
    env.position  = 0.0
    env.portfolio = 10_000.0
    env.peak      = 10_000.0
    env.done      = false
    empty!(env.portfolio_history)
    empty!(env.action_history)
    empty!(env.reward_history)
    push!(env.portfolio_history, 10_000.0)
    _compute_state(env)
end

function _ema(data::AbstractVector{Float64}, alpha::Float64)::Float64
    e = data[1]
    for x in data[2:end]; e = alpha*x + (1-alpha)*e; end
    e
end

function _rsi(returns::AbstractVector{Float64})::Float64
    gains  = max.(returns, 0.0)
    losses = max.(-returns, 0.0)
    ag = mean(gains); al = mean(losses)
    al < 1e-10 && return 100.0
    100.0 - 100.0 / (1.0 + ag/al)
end

function _bollinger_signal(prices::AbstractVector{Float64})::Float64
    n = length(prices)
    n < 2 && return 0.0
    mu = mean(prices); sig = std(prices)
    sig < 1e-10 && return 0.0
    clamp((prices[end] - mu) / sig, -3.0, 3.0)
end

function _compute_state(env::TradingEnvironment)::Vector{Float64}
    t = env.t; w = env.window
    ri = env.log_returns[max(1, t-w):t]
    pi = env.prices[max(1, t-w):t]
    vi = env.volumes[max(1, t-w):t]

    ret       = env.log_returns[t]
    vol5      = std(env.log_returns[max(1,t-5):t])
    vol20     = std(ri)
    rsi       = _rsi(ri) / 100.0 - 0.5
    macd      = (length(pi) >= 12) ?
                (_ema(pi, 2.0/13) - _ema(pi, 2.0/27)) / (std(pi)+1e-10) : 0.0
    bb        = _bollinger_signal(pi)
    vol_ratio = (vi[end] / (mean(vi)+1e-10)) - 1.0
    position  = env.position
    drawdown  = -max((env.peak - env.portfolio) / (env.peak+1e-10), 0.0)

    clamp.(Float64[ret, vol5, vol20, rsi, macd, bb, vol_ratio, position, drawdown],
           -5.0, 5.0)
end

"""
    env_step!(env, action) -> (next_state, reward, done, info)
"""
function env_step!(env::TradingEnvironment, action::Int)
    env.done && return _compute_state(env), 0.0, true, (pnl=0.0, portfolio=env.portfolio)

    desired = Float64(action - 1)  # -1, 0, +1
    trade   = desired - env.position
    cost    = abs(trade) * env.tc * env.portfolio
    env.position = desired

    env.t += 1
    if env.t >= env.n; env.done = true; end

    ret = env.done ? 0.0 : env.log_returns[env.t]
    pnl = env.position * ret * env.portfolio - cost
    env.portfolio = max(env.portfolio + pnl, 1.0)
    env.peak = max(env.peak, env.portfolio)

    push!(env.portfolio_history, env.portfolio)
    push!(env.action_history, action)

    # Reward: log return adjusted for drawdown
    reward = pnl / (env.portfolio + 1e-10)
    dd = (env.peak - env.portfolio) / (env.peak + 1e-10)
    reward -= 0.3 * dd
    push!(env.reward_history, reward)

    ns = _compute_state(env)
    ns, reward, env.done, (pnl=pnl, portfolio=env.portfolio)
end

# ─────────────────────────────────────────────────────────────
# 2. NEURAL NETWORK STRUCTS
# ─────────────────────────────────────────────────────────────

"""
    QNetwork

Q-value network: outputs Q(s, a) for all actions simultaneously.
Architecture: [in_dim → h1 → h2 → n_actions]
"""
mutable struct QNetwork
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
    W3::Matrix{Float64}; b3::Vector{Float64}
    in_dim::Int; h1::Int; h2::Int; n_actions::Int
end

function QNetwork(in_dim::Int, h1::Int, h2::Int, n_actions::Int;
                   rng=MersenneTwister(42))
    QNetwork(
        randn(rng, h1, in_dim) .* sqrt(2.0/in_dim), zeros(h1),
        randn(rng, h2, h1)     .* sqrt(2.0/h1),     zeros(h2),
        randn(rng, n_actions, h2) .* sqrt(2.0/h2),  zeros(n_actions),
        in_dim, h1, h2, n_actions
    )
end

"""
    PolicyNetwork

Policy network: outputs action log-probabilities.
"""
mutable struct PolicyNetwork
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
    W3::Matrix{Float64}; b3::Vector{Float64}
    in_dim::Int; hidden::Int; n_actions::Int
end

function PolicyNetwork(in_dim::Int, hidden::Int, n_actions::Int;
                        rng=MersenneTwister(42))
    PolicyNetwork(
        randn(rng, hidden, in_dim) .* sqrt(2.0/in_dim), zeros(hidden),
        randn(rng, hidden, hidden) .* sqrt(2.0/hidden), zeros(hidden),
        randn(rng, n_actions, hidden) .* sqrt(2.0/hidden), zeros(n_actions),
        in_dim, hidden, n_actions
    )
end

"""
    ValueNetwork

State-value network V(s): scalar output.
"""
mutable struct ValueNetwork
    W1::Matrix{Float64}; b1::Vector{Float64}
    W2::Matrix{Float64}; b2::Vector{Float64}
    W3::Matrix{Float64}; b3::Vector{Float64}
    in_dim::Int; hidden::Int
end

function ValueNetwork(in_dim::Int, hidden::Int; rng=MersenneTwister(42))
    ValueNetwork(
        randn(rng, hidden, in_dim) .* sqrt(2.0/in_dim), zeros(hidden),
        randn(rng, hidden, hidden) .* sqrt(2.0/hidden), zeros(hidden),
        randn(rng, 1, hidden)      .* sqrt(2.0/hidden), zeros(1),
        in_dim, hidden
    )
end

# ─────────────────────────────────────────────────────────────
# 3. FORWARD/BACKWARD PASSES
# ─────────────────────────────────────────────────────────────

relu(x) = max.(x, 0.0)
relu(x::Float64) = max(x, 0.0)

function softmax(x::Vector{Float64})::Vector{Float64}
    e = exp.(x .- maximum(x)); e ./ sum(e)
end

"""
    forward_pass(net::QNetwork, x) -> (output, cache)
"""
function forward_pass(net::QNetwork, x::Vector{Float64})
    z1 = net.W1 * x .+ net.b1; a1 = relu(z1)
    z2 = net.W2 * a1 .+ net.b2; a2 = relu(z2)
    z3 = net.W3 * a2 .+ net.b3
    z3, (x, a1, a2, z1, z2)
end

function forward_pass(net::PolicyNetwork, x::Vector{Float64})
    z1 = net.W1 * x .+ net.b1; a1 = relu(z1)
    z2 = net.W2 * a1 .+ net.b2; a2 = relu(z2)
    z3 = net.W3 * a2 .+ net.b3
    softmax(z3), (x, a1, a2, z1, z2, z3)
end

function forward_pass(net::ValueNetwork, x::Vector{Float64})
    z1 = net.W1 * x .+ net.b1; a1 = relu(z1)
    z2 = net.W2 * a1 .+ net.b2; a2 = relu(z2)
    (net.W3 * a2 .+ net.b3)[1], (x, a1, a2)
end

"""
    backward_pass!(net::QNetwork, cache, dL_dout, lr)

Backpropagate gradient through QNetwork and update weights.
"""
function backward_pass!(net::QNetwork, cache, dL_dout::Vector{Float64},
                         lr::Float64)
    x, a1, a2, z1, z2 = cache
    # Layer 3
    net.W3 .-= lr * dL_dout * a2'
    net.b3 .-= lr * dL_dout
    # Layer 2
    da2 = net.W3' * dL_dout .* (a2 .> 0)
    net.W2 .-= lr * da2 * a1'
    net.b2 .-= lr * da2
    # Layer 1
    da1 = net.W2' * da2 .* (a1 .> 0)
    net.W1 .-= lr * da1 * x'
    net.b1 .-= lr * da1
end

function backward_pass!(net::PolicyNetwork, cache, dL_dlogits::Vector{Float64},
                         lr::Float64)
    x, a1, a2, z1, z2, z3 = cache
    net.W3 .-= lr * dL_dlogits * a2'
    net.b3 .-= lr * dL_dlogits
    da2 = net.W3' * dL_dlogits .* (a2 .> 0)
    net.W2 .-= lr * da2 * a1'
    net.b2 .-= lr * da2
    da1 = net.W2' * da2 .* (a1 .> 0)
    net.W1 .-= lr * da1 * x'
    net.b1 .-= lr * da1
end

function backward_pass!(net::ValueNetwork, cache, dL_dv::Float64, lr::Float64)
    x, a1, a2 = cache
    dz3 = [dL_dv]
    net.W3 .-= lr * dz3 * a2'
    net.b3 .-= lr * dz3
    da2 = net.W3' * dz3 .* (a2 .> 0)
    net.W2 .-= lr * da2 * a1'
    net.b2 .-= lr * da2
    da1 = net.W2' * da2 .* (a1 .> 0)
    net.W1 .-= lr * da1 * x'
    net.b1 .-= lr * da1
end

"""
    network_copy!(dst, src)

Copy all weights from src to dst (same type).
"""
function network_copy!(dst::QNetwork, src::QNetwork)
    dst.W1 .= src.W1; dst.b1 .= src.b1
    dst.W2 .= src.W2; dst.b2 .= src.b2
    dst.W3 .= src.W3; dst.b3 .= src.b3
end

"""Soft update: dst ← tau*src + (1-tau)*dst."""
function network_update!(dst::QNetwork, src::QNetwork, tau::Float64=0.005)
    dst.W1 .= tau*src.W1 .+ (1-tau)*dst.W1
    dst.b1 .= tau*src.b1 .+ (1-tau)*dst.b1
    dst.W2 .= tau*src.W2 .+ (1-tau)*dst.W2
    dst.b2 .= tau*src.b2 .+ (1-tau)*dst.b2
    dst.W3 .= tau*src.W3 .+ (1-tau)*dst.W3
    dst.b3 .= tau*src.b3 .+ (1-tau)*dst.b3
end

# ─────────────────────────────────────────────────────────────
# 4. REPLAY BUFFER
# ─────────────────────────────────────────────────────────────

"""
    ReplayBuffer

Priority-capable circular experience replay buffer.
"""
mutable struct ReplayBuffer
    states::Matrix{Float64}
    actions::Vector{Int}
    rewards::Vector{Float64}
    next_states::Matrix{Float64}
    dones::Vector{Bool}
    cap::Int; sz::Int; ptr::Int; dim::Int
    priorities::Vector{Float64}  # for prioritized replay
end

function ReplayBuffer(cap::Int, dim::Int)
    ReplayBuffer(zeros(dim,cap), zeros(Int,cap), zeros(cap),
                 zeros(dim,cap), zeros(Bool,cap),
                 cap, 0, 1, dim, ones(cap))
end

function add_experience!(buf::ReplayBuffer, s, a, r, ns, d; priority::Float64=1.0)
    buf.states[:,buf.ptr]      = s
    buf.actions[buf.ptr]       = a
    buf.rewards[buf.ptr]       = r
    buf.next_states[:,buf.ptr] = ns
    buf.dones[buf.ptr]         = d
    buf.priorities[buf.ptr]    = priority
    buf.sz  = min(buf.sz + 1, buf.cap)
    buf.ptr = (buf.ptr % buf.cap) + 1
end

function sample_batch(buf::ReplayBuffer, n::Int; rng=MersenneTwister(1),
                       prioritized::Bool=false)
    n = min(n, buf.sz)
    if prioritized
        p = buf.priorities[1:buf.sz]; p ./= sum(p)
        idx = [begin u=rand(rng); sum(cumsum(p) .< u)+1 end for _ in 1:n]
        idx = clamp.(idx, 1, buf.sz)
    else
        idx = rand(rng, 1:buf.sz, n)
    end
    (s=buf.states[:,idx], a=buf.actions[idx], r=buf.rewards[idx],
     ns=buf.next_states[:,idx], d=buf.dones[idx], idx=idx)
end

# ─────────────────────────────────────────────────────────────
# 5. TRAINERS
# ─────────────────────────────────────────────────────────────

"""
    DQNTrainer

Complete DQN training system with target network and replay.
"""
mutable struct DQNTrainer
    q_net::QNetwork
    target::QNetwork
    buf::ReplayBuffer
    gamma::Float64; lr::Float64; batch::Int
    eps::Float64; eps_min::Float64; eps_decay::Float64
    update_every::Int; step::Int
    rng::AbstractRNG
    losses::Vector{Float64}
end

function DQNTrainer(state_dim::Int; n_actions=3, hidden=64,
                     buf_cap=20000, gamma=0.95, lr=5e-4,
                     batch=64, rng=MersenneTwister(42))
    q  = QNetwork(state_dim, hidden, hidden, n_actions; rng=rng)
    tg = QNetwork(state_dim, hidden, hidden, n_actions; rng=rng)
    network_copy!(tg, q)
    DQNTrainer(q, tg, ReplayBuffer(buf_cap, state_dim),
               gamma, lr, batch, 1.0, 0.02, 0.997,
               100, 0, rng, Float64[])
end

function select_action(t::DQNTrainer, s::Vector{Float64})::Int
    if rand(t.rng) < t.eps
        return rand(t.rng, 0:t.q_net.n_actions-1)
    end
    q, _ = forward_pass(t.q_net, s)
    argmax(q) - 1
end

function train_step!(t::DQNTrainer)::Float64
    t.buf.sz < t.batch && return 0.0
    b    = sample_batch(t.buf, t.batch; rng=t.rng)
    loss = 0.0
    for i in 1:t.batch
        s  = b.s[:, i]; a = b.a[i]; r = b.r[i]; ns = b.ns[:,i]; d = b.d[i]
        q_vals, cache = forward_pass(t.q_net, s)
        q_next, _     = forward_pass(t.target, ns)
        target_val    = r + (d ? 0.0 : t.gamma * maximum(q_next))
        err           = q_vals[a+1] - target_val
        loss         += err^2
        dL = zeros(t.q_net.n_actions); dL[a+1] = 2*err / t.batch
        backward_pass!(t.q_net, cache, dL, t.lr)
    end
    t.step += 1
    t.eps   = max(t.eps * t.eps_decay, t.eps_min)
    t.step % t.update_every == 0 && network_update!(t.target, t.q_net, 0.01)
    push!(t.losses, loss / t.batch)
    loss / t.batch
end

"""
    train_dqn!(trainer, env, n_episodes; verbose=true) -> Vector{Float64}

Train DQN agent on environment. Returns episode rewards.
"""
function train_dqn!(trainer::DQNTrainer, env::TradingEnvironment,
                     n_episodes::Int; verbose::Bool=true)::Vector{Float64}
    ep_rewards = zeros(n_episodes)
    for ep in 1:n_episodes
        s = reset!(env); total_r = 0.0
        while !env.done
            a  = select_action(trainer, s)
            ns, r, d, _ = env_step!(env, a)
            add_experience!(trainer.buf, s, a, r, ns, d)
            train_step!(trainer)
            total_r += r; s = ns
        end
        ep_rewards[ep] = total_r
        if verbose && ep % max(1, n_episodes÷10) == 0
            avg = mean(ep_rewards[max(1,ep-9):ep])
            fp  = isempty(env.portfolio_history) ? 10000.0 : env.portfolio_history[end]
            println("  DQN Ep $ep | AvgR=$(round(avg,digits=4)) | Port=\$$(round(fp,digits=0)) | ε=$(round(trainer.eps,digits=3))")
        end
    end
    ep_rewards
end

# ─────────────────────────────────────────────────────────────
# 6. A2C TRAINER
# ─────────────────────────────────────────────────────────────

"""
    A2CTrainer

Advantage Actor-Critic training system.
"""
mutable struct A2CTrainer
    actor::PolicyNetwork
    critic::ValueNetwork
    gamma::Float64; lr_actor::Float64; lr_critic::Float64
    rng::AbstractRNG; losses::Vector{Float64}
end

function A2CTrainer(state_dim::Int; n_actions=3, hidden=64,
                     gamma=0.95, lr_a=3e-4, lr_c=1e-3, rng=MersenneTwister(42))
    A2CTrainer(PolicyNetwork(state_dim, hidden, n_actions; rng=rng),
               ValueNetwork(state_dim, hidden; rng=rng),
               gamma, lr_a, lr_c, rng, Float64[])
end

function select_action(t::A2CTrainer, s::Vector{Float64})::Int
    probs, _ = forward_pass(t.actor, s)
    u = rand(t.rng); cs = 0.0
    for (i, p) in enumerate(probs)
        cs += p; u <= cs && return i-1
    end
    length(probs) - 1
end

function train_a2c_step!(t::A2CTrainer, s, a, r, ns, done)
    v_cur,  cache_c  = forward_pass(t.critic, s)
    v_next, _        = forward_pass(t.critic, ns)
    adv = r + (done ? 0.0 : t.gamma * v_next) - v_cur
    # Critic update
    backward_pass!(t.critic, cache_c, -2*adv, t.lr_critic)
    # Actor update
    probs, cache_a = forward_pass(t.actor, s)
    ai = a + 1
    dL = copy(probs); dL[ai] -= 1.0; dL .*= -adv
    backward_pass!(t.actor, cache_a, dL, t.lr_actor)
    adv^2
end

function train_a2c!(trainer::A2CTrainer, env::TradingEnvironment,
                     n_episodes::Int; verbose::Bool=true)::Vector{Float64}
    ep_rewards = zeros(n_episodes)
    for ep in 1:n_episodes
        s = reset!(env); total_r = 0.0
        while !env.done
            a  = select_action(trainer, s)
            ns, r, d, _ = env_step!(env, a)
            train_a2c_step!(trainer, s, a, r, ns, d)
            total_r += r; s = ns
        end
        ep_rewards[ep] = total_r
        if verbose && ep % max(1, n_episodes÷10) == 0
            println("  A2C Ep $ep | AvgR=$(round(mean(ep_rewards[max(1,ep-9):ep]),digits=4))")
        end
    end
    ep_rewards
end

# ─────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────

"""
    compute_sharpe(returns; rf=0.0, annualize=252) -> Float64
"""
function compute_sharpe(returns::Vector{Float64}; rf::Float64=0.0,
                          annualize::Int=252)::Float64
    isempty(returns) && return 0.0
    excess = returns .- rf / annualize
    s = std(excess)
    s < 1e-10 && return 0.0
    mean(excess) / s * sqrt(annualize)
end

"""
    compute_max_drawdown(portfolio_values) -> Float64
"""
function compute_max_drawdown(pvs::Vector{Float64})::Float64
    isempty(pvs) && return 0.0
    peak = pvs[1]; mdd = 0.0
    for v in pvs
        peak = max(peak, v)
        mdd  = max(mdd, (peak - v) / (peak + 1e-10))
    end
    mdd
end

"""
    evaluate_policy(agent, env; n_runs=3) -> NamedTuple

Evaluate a trained agent (DQN or A2C) with no exploration.
"""
function evaluate_policy(agent, env::TradingEnvironment; n_runs::Int=3)
    all_returns = Float64[]
    all_portfolios = Float64[]

    saved_eps = isa(agent, DQNTrainer) ? agent.eps : NaN
    isa(agent, DQNTrainer) && (agent.eps = 0.0)

    for _ in 1:n_runs
        s = reset!(env)
        while !env.done
            a  = select_action(agent, s)
            s, _, _, _ = env_step!(env, a)
        end
        append!(all_returns, env.reward_history)
        isempty(env.portfolio_history) || push!(all_portfolios, env.portfolio_history[end])
    end

    isa(agent, DQNTrainer) && (agent.eps = saved_eps)

    final_val = isempty(all_portfolios) ? 10_000.0 : mean(all_portfolios)
    sharpe = compute_sharpe(all_returns)
    mdd    = compute_max_drawdown(env.portfolio_history)
    (sharpe=sharpe, max_drawdown=mdd, final_value=final_val,
     total_return=(final_val - 10_000.0)/10_000.0,
     n_trades=sum(abs.(diff(env.action_history; dims=1)) .> 0; init=0))
end

# ─────────────────────────────────────────────────────────────
# 8. HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────

"""
    HyperparamSearch

Container for hyperparameter search configurations.
"""
struct HyperparamSearch
    param_grid::Dict{Symbol, Vector}
    n_random::Int
    n_episodes::Int
    metric::Symbol   # :sharpe, :total_return, :max_drawdown
end

HyperparamSearch(; param_grid=Dict(), n_random=20, n_episodes=30, metric=:sharpe) =
    HyperparamSearch(param_grid, n_random, n_episodes, metric)

"""
    grid_search(prices, hps::HyperparamSearch; rng=...) -> NamedTuple

Grid search over hyperparameter combinations.
"""
function grid_search(prices::Vector{Float64}, hps::HyperparamSearch;
                      rng=MersenneTwister(42))
    keys_list = collect(keys(hps.param_grid))
    vals_list = [hps.param_grid[k] for k in keys_list]

    # Generate all combinations
    combos = [Dict(keys_list[i] => v[i] for (i,v) in enumerate(combo))
              for combo in Iterators.product(vals_list...)]

    results = map(combos) do config
        lr     = get(config, :lr, 5e-4)
        hidden = get(config, :hidden, 64)
        gamma  = get(config, :gamma, 0.95)
        trainer = DQNTrainer(9; n_actions=3, hidden=hidden, gamma=gamma, lr=lr, rng=rng)
        env     = TradingEnvironment(prices)
        train_dqn!(trainer, env, hps.n_episodes; verbose=false)
        met = evaluate_policy(trainer, TradingEnvironment(prices)).sharpe
        (config=config, metric=met)
    end

    best_idx = argmax([r.metric for r in results])
    (results=results, best=results[best_idx])
end

"""
    random_search(prices, param_ranges, n_trials; rng=...) -> NamedTuple

Random hyperparameter search.
param_ranges: Dict of :param => (min, max) pairs.
"""
function random_search(prices::Vector{Float64},
                        param_ranges::Dict,
                        n_trials::Int=10;
                        n_episodes::Int=20,
                        rng=MersenneTwister(42))
    results = map(1:n_trials) do trial
        config = Dict{Symbol,Float64}()
        for (k, (lo, hi)) in param_ranges
            config[k] = lo + rand(rng) * (hi - lo)
        end
        lr     = get(config, :lr, 5e-4)
        hidden = max(16, Int(round(get(config, :hidden, 64.0))))
        gamma  = clamp(get(config, :gamma, 0.95), 0.8, 0.999)

        trainer = DQNTrainer(9; hidden=hidden, gamma=gamma, lr=lr, rng=rng)
        env     = TradingEnvironment(prices)
        train_dqn!(trainer, env, n_episodes; verbose=false)
        met = evaluate_policy(trainer, TradingEnvironment(prices)).sharpe
        (config=config, metric=met, trial=trial)
    end

    best_idx = argmax([r.metric for r in results])
    (results=results, best=results[best_idx])
end

# ─────────────────────────────────────────────────────────────
# 9. POLICY VISUALIZATION & ANALYSIS
# ─────────────────────────────────────────────────────────────

"""
    policy_action_distribution(agent, env) -> Matrix{Float64}

Compute distribution of actions across different market states.
Returns n_states × n_actions frequency matrix by discretizing state space.
"""
function policy_action_distribution(agent, env::TradingEnvironment)
    s = reset!(env)
    state_dim = length(s)
    action_counts = Dict{Int, Vector{Int}}()  # market regime → action counts

    saved_eps = isa(agent, DQNTrainer) ? agent.eps : NaN
    isa(agent, DQNTrainer) && (agent.eps = 0.0)

    while !env.done
        a = select_action(agent, s)
        # Regime: sign of state[1] (return) and state[3] (vol)
        regime = Int(s[1] > 0) * 2 + Int(s[3] > median([0.01]))
        cnts = get!(action_counts, regime, zeros(Int, 3))
        cnts[a+1] += 1
        s, _, _, _ = env_step!(env, a)
    end

    isa(agent, DQNTrainer) && (agent.eps = saved_eps)
    action_counts
end

"""
    feature_importance(agent::DQNTrainer, states::Matrix{Float64}) -> Vector{Float64}

Estimate feature importance via perturbation analysis.
"""
function feature_importance(agent::DQNTrainer, states::Matrix{Float64})::Vector{Float64}
    n, d = size(states)
    base_q = [maximum(forward_pass(agent.q_net, states[i,:])[1]) for i in 1:n]
    importance = zeros(d)
    for j in 1:d
        perturbed = copy(states)
        perturbed[:, j] .= 0.0
        pert_q = [maximum(forward_pass(agent.q_net, perturbed[i,:])[1]) for i in 1:n]
        importance[j] = mean(abs.(base_q .- pert_q))
    end
    importance ./ (sum(importance) + 1e-10)
end

# ─────────────────────────────────────────────────────────────
# 10. DEMO
# ─────────────────────────────────────────────────────────────

"""Synthetic regime-switching prices."""
function _gen_prices(n::Int=500; rng=MersenneTwister(42))
    prices = [10_000.0]
    regime = 1
    for t in 2:n
        rand(rng) < 0.03 && (regime = 3 - regime)
        mu = regime == 1 ? 0.0005 : -0.0003
        sig = regime == 1 ? 0.015  : 0.025
        push!(prices, prices[end] * exp(mu + sig*randn(rng)))
    end
    prices
end

"""
    run_rl_module_demo() -> Nothing
"""
function run_rl_module_demo()
    println("=" ^ 60)
    println("REINFORCEMENT LEARNING MODULE DEMO")
    println("=" ^ 60)

    rng = MersenneTwister(42)
    prices = _gen_prices(600; rng=rng)
    train_prices = prices[1:400]; test_prices = prices[400:end]
    state_names = ["return","vol_5","vol_20","rsi","macd","bollinger","vol_ratio","position","drawdown"]

    println("\n1. DQN Training")
    dqn = DQNTrainer(9; hidden=32, rng=MersenneTwister(1))
    train_env = TradingEnvironment(train_prices)
    train_dqn!(dqn, train_env, 25; verbose=true)
    test_env = TradingEnvironment(test_prices)
    eval_dqn = evaluate_policy(dqn, test_env)
    println("  Test Sharpe:     $(round(eval_dqn.sharpe,digits=3))")
    println("  Test Max DD:     $(round(eval_dqn.max_drawdown*100,digits=2))%")
    println("  Test Return:     $(round(eval_dqn.total_return*100,digits=2))%")
    println("  Epsilon (final): $(round(dqn.eps,digits=3))")

    println("\n2. A2C Training")
    a2c = A2CTrainer(9; hidden=32, rng=MersenneTwister(2))
    train_a2c!(a2c, TradingEnvironment(train_prices), 25; verbose=false)
    eval_a2c = evaluate_policy(a2c, TradingEnvironment(test_prices))
    println("  Test Sharpe:  $(round(eval_a2c.sharpe,digits=3))")
    println("  Test Return:  $(round(eval_a2c.total_return*100,digits=2))%")

    println("\n3. Feature Importance (DQN)")
    states_matrix = zeros(50, 9)
    s = reset!(TradingEnvironment(train_prices))
    states_matrix[1, :] = s
    tmp_env = TradingEnvironment(train_prices); reset!(tmp_env)
    for i in 2:50
        a = select_action(dqn, tmp_env.done ? s : _compute_state(tmp_env))
        ns, _, _, _ = env_step!(tmp_env, a)
        states_matrix[i, :] = ns
    end
    imp = feature_importance(dqn, states_matrix)
    sorted_idx = sortperm(imp, rev=true)
    println("  Top 3 features:")
    for i in 1:3
        println("    $(state_names[sorted_idx[i]]): $(round(imp[sorted_idx[i]]*100,digits=1))%")
    end

    println("\n4. Random Hyperparameter Search")
    param_ranges = Dict(
        :lr     => (1e-4, 1e-2),
        :gamma  => (0.90, 0.99),
        :hidden => (16.0, 64.0)
    )
    rs = random_search(train_prices, param_ranges, 5; n_episodes=10, rng=rng)
    println("  Best Sharpe: $(round(rs.best.metric,digits=3))")
    println("  Best config: $(rs.best.config)")

    println("\nDone.")
    nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 – Double DQN and Dueling Architecture
# ─────────────────────────────────────────────────────────────────────────────

"""
    double_dqn_target(q_online, q_target, next_state, reward, done, gamma)

Double-DQN target: online network selects the action, target network evaluates.
Reduces maximisation bias versus vanilla DQN.
"""
function double_dqn_target(q_online::QNetwork, q_target::QNetwork,
                            next_state::Vector{Float64},
                            reward::Float64, done::Bool,
                            gamma::Float64=0.99)
    done && return reward
    q_sel  = forward_pass(q_online, next_state)
    best_a = argmax(q_sel)
    q_eval = forward_pass(q_target, next_state)
    return reward + gamma * q_eval[best_a]
end

"""
    DuelingQNetwork

Shared encoder → value stream V(s) + advantage stream A(s,a).
Q(s,a) = V(s) + A(s,a) − mean_a A(s,a).
"""
mutable struct DuelingQNetwork
    W_enc::Matrix{Float64}; b_enc::Vector{Float64}
    W_val::Matrix{Float64}; b_val::Vector{Float64}
    W_adv::Matrix{Float64}; b_adv::Vector{Float64}
    n_actions::Int
end

function DuelingQNetwork(input_dim::Int, hidden_dim::Int, n_actions::Int)
    sc = sqrt(2.0 / input_dim)
    DuelingQNetwork(
        randn(hidden_dim, input_dim) .* sc, zeros(hidden_dim),
        randn(1, hidden_dim) .* sc,         zeros(1),
        randn(n_actions, hidden_dim) .* sc, zeros(n_actions),
        n_actions)
end

function forward_dueling(net::DuelingQNetwork, s::Vector{Float64})
    h = max.(0.0, net.W_enc * s .+ net.b_enc)
    V = (net.W_val * h .+ net.b_val)[1]
    A = net.W_adv * h .+ net.b_adv
    return V .+ (A .- mean(A))
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 10 – Prioritised Experience Replay (PER)
# ─────────────────────────────────────────────────────────────────────────────

"""
    PrioritisedReplayBuffer

Sampling probability ∝ |δ|^α.  Importance-sampling weights correct for bias.
Fields: alpha (priority exponent), beta (IS exponent, annealed toward 1).
"""
mutable struct PrioritisedReplayBuffer
    capacity::Int
    states::Matrix{Float64}; actions::Vector{Int}
    rewards::Vector{Float64}; next_states::Matrix{Float64}
    dones::Vector{Bool}; priorities::Vector{Float64}
    ptr::Int; size::Int
    alpha::Float64; beta::Float64; eps_p::Float64
end

function PrioritisedReplayBuffer(capacity::Int, state_dim::Int;
                                  alpha::Float64=0.6, beta::Float64=0.4)
    PrioritisedReplayBuffer(
        capacity,
        zeros(state_dim, capacity), zeros(Int, capacity),
        zeros(capacity), zeros(state_dim, capacity),
        zeros(Bool, capacity), ones(capacity),
        1, 0, alpha, beta, 1e-6)
end

function per_push!(buf::PrioritisedReplayBuffer,
                   s, a, r, s2, done, td_err=1.0)
    p = (abs(td_err) + buf.eps_p) ^ buf.alpha
    buf.states[:, buf.ptr] = s;  buf.actions[buf.ptr]        = a
    buf.rewards[buf.ptr]   = r;  buf.next_states[:, buf.ptr] = s2
    buf.dones[buf.ptr]     = done; buf.priorities[buf.ptr]   = p
    buf.ptr  = mod1(buf.ptr + 1, buf.capacity)
    buf.size = min(buf.size + 1, buf.capacity)
end

function per_sample(buf::PrioritisedReplayBuffer, batch::Int)
    probs = buf.priorities[1:buf.size]
    probs ./= sum(probs)
    idxs = Int[]
    for _ in 1:batch
        u = rand(); cp = 0.0; chosen = 1
        for (i, p) in enumerate(probs)
            cp += p
            if u <= cp; chosen = i; break; end
        end
        push!(idxs, chosen)
    end
    is_w = (buf.size .* probs[idxs]) .^ (-buf.beta)
    is_w ./= maximum(is_w)
    return idxs, is_w
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 11 – Soft Actor-Critic (discrete-action variant)
# ─────────────────────────────────────────────────────────────────────────────

"""
    SACAgent

Discrete SAC: two Q-networks (clipped double-Q), policy network (softmax),
fixed entropy temperature alpha_ent.
"""
mutable struct SACAgent
    q1::QNetwork; q2::QNetwork
    q1_t::QNetwork; q2_t::QNetwork
    policy::PolicyNetwork
    alpha_ent::Float64; gamma::Float64; tau::Float64; lr::Float64
    replay::ReplayBuffer; n_actions::Int
end

function SACAgent(state_dim::Int, n_actions::Int;
                  hidden_dim::Int=64, alpha_ent::Float64=0.2,
                  gamma::Float64=0.99, tau::Float64=0.005,
                  lr::Float64=3e-4, buf::Int=10_000)
    q1 = QNetwork(state_dim, hidden_dim, n_actions)
    q2 = QNetwork(state_dim, hidden_dim, n_actions)
    SACAgent(q1, q2, deepcopy(q1), deepcopy(q2),
             PolicyNetwork(state_dim, hidden_dim, n_actions),
             alpha_ent, gamma, tau, lr,
             ReplayBuffer(buf, state_dim, n_actions), n_actions)
end

function sac_policy_probs(agent::SACAgent, state::Vector{Float64})
    logits = forward_pass(agent.policy, state)
    ex = exp.(logits .- maximum(logits))
    return ex ./ sum(ex)
end

function sac_select_action(agent::SACAgent, state::Vector{Float64};
                            deterministic::Bool=false)
    probs = sac_policy_probs(agent, state)
    deterministic && return argmax(probs)
    u = rand(); cp = 0.0
    for (i, p) in enumerate(probs)
        cp += p; if u <= cp; return i; end
    end
    return agent.n_actions
end

function sac_entropy(probs::Vector{Float64})
    return -sum(p * log(p + 1e-8) for p in probs)
end

function sac_value_target(agent::SACAgent, ns::Vector{Float64},
                           reward::Float64, done::Bool)
    done && return reward
    probs  = sac_policy_probs(agent, ns)
    H      = sac_entropy(probs)
    q1_t   = forward_pass(agent.q1_t, ns)
    q2_t   = forward_pass(agent.q2_t, ns)
    V_next = sum(probs .* min.(q1_t, q2_t)) + agent.alpha_ent * H
    return reward + agent.gamma * V_next
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 12 – Curriculum Learning and Adaptive Episode Length
# ─────────────────────────────────────────────────────────────────────────────

"""
    CurriculumScheduler

Three-phase difficulty ramp:
  Phase 1 — low vol, short episodes
  Phase 2 — mixed vol, medium episodes
  Phase 3 — full difficulty, long episodes
"""
mutable struct CurriculumScheduler
    total_episodes::Int; completed::Int
    thresholds::Vector{Float64}
    ep_lengths::Vector{Int}
    vol_mults::Vector{Float64}
end

function CurriculumScheduler(total::Int;
        thresholds=[0.33, 0.66, 1.0],
        ep_lengths=[50, 100, 200],
        vol_mults=[0.5, 1.0, 1.5])
    CurriculumScheduler(total, 0, thresholds, ep_lengths, vol_mults)
end

function current_phase(sched::CurriculumScheduler)
    frac = sched.completed / max(1, sched.total_episodes)
    for (i, t) in enumerate(sched.thresholds)
        frac <= t && return i
    end
    return length(sched.thresholds)
end

curriculum_episode_length(s::CurriculumScheduler) =
    s.ep_lengths[current_phase(s)]

curriculum_vol_mult(s::CurriculumScheduler) =
    s.vol_mults[current_phase(s)]

advance_curriculum!(s::CurriculumScheduler) = (s.completed += 1; nothing)

# ─────────────────────────────────────────────────────────────────────────────
# Section 13 – Reward Normalisation and Intrinsic Curiosity
# ─────────────────────────────────────────────────────────────────────────────

"""
    RunningNormalizer

Online Welford normaliser; clips extreme values to ±clip.
"""
mutable struct RunningNormalizer
    n::Int; mu::Float64; M2::Float64; clip::Float64
end
RunningNormalizer(; clip=10.0) = RunningNormalizer(0, 0.0, 0.0, clip)

function normalize!(rn::RunningNormalizer, x::Float64)
    rn.n += 1
    d = x - rn.mu; rn.mu += d / rn.n
    rn.M2 += d * (x - rn.mu)
    var = rn.n < 2 ? 1.0 : rn.M2 / (rn.n - 1)
    return clamp((x - rn.mu) / sqrt(var + 1e-8), -rn.clip, rn.clip)
end

"""
    IntrinsicCuriosityModule

Linear forward model predicts next state; prediction error = intrinsic reward.
"""
mutable struct IntrinsicCuriosityModule
    W::Matrix{Float64}; lr::Float64
    state_dim::Int; n_actions::Int; beta::Float64
end

function IntrinsicCuriosityModule(state_dim::Int, n_actions::Int;
                                   lr::Float64=1e-3, beta::Float64=0.1)
    IntrinsicCuriosityModule(
        randn(state_dim, state_dim + n_actions) .* 0.01,
        lr, state_dim, n_actions, beta)
end

function icm_reward!(icm::IntrinsicCuriosityModule,
                     s::Vector{Float64}, a::Int, ns::Vector{Float64})
    oh = zeros(icm.n_actions); oh[a] = 1.0
    x  = vcat(s, oh)
    pred = icm.W * x
    err  = ns .- pred
    intrinsic = 0.5 * sum(err.^2)
    icm.W .+= icm.lr .* err * x'          # gradient ascent on −loss
    return icm.beta * intrinsic
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 14 – Policy Ensemble with Majority-Vote Execution
# ─────────────────────────────────────────────────────────────────────────────

"""
    PolicyEnsemble

Collection of independently trained Q-networks.  Action selection uses
majority vote; uncertainty is the inter-member Q-value std.
"""
struct PolicyEnsemble
    networks::Vector{QNetwork}
    n_actions::Int
end

function ensemble_action(ens::PolicyEnsemble, state::Vector{Float64})
    votes = zeros(Int, ens.n_actions)
    for net in ens.networks
        votes[argmax(forward_pass(net, state))] += 1
    end
    return argmax(votes)
end

function ensemble_q_mean(ens::PolicyEnsemble, state::Vector{Float64})
    qs = [forward_pass(net, state) for net in ens.networks]
    return mean(qs)
end

function ensemble_q_uncertainty(ens::PolicyEnsemble, state::Vector{Float64})
    qs  = [forward_pass(net, state) for net in ens.networks]
    mu  = mean(qs)
    var = mean([(q .- mu).^2 for q in qs])
    return sqrt.(var)
end

"""
    train_bootstrap_ensemble(prices, n_members; kwargs...) -> PolicyEnsemble

Train `n_members` DQN agents on block-bootstrap resamples of `prices`.
"""
function train_bootstrap_ensemble(prices::Vector{Float64}, n_members::Int;
                                   n_episodes::Int=30, hidden_dim::Int=32)
    T = length(prices); n_actions = 3; state_dim = 9
    networks = QNetwork[]
    for _ in 1:n_members
        block = max(5, T ÷ 20)
        samp  = Float64[]
        while length(samp) < T
            i = rand(1:(T-block+1))
            append!(samp, prices[i:min(i+block-1, T)])
        end
        samp = samp[1:T]
        net = QNetwork(state_dim, hidden_dim, n_actions)
        tgt = deepcopy(net)
        buf = ReplayBuffer(2000, state_dim, n_actions)
        env = TradingEnvironment(samp)
        for ep in 1:n_episodes
            reset_environment!(env)
            s = get_state(env); eps = 0.5 * (1 - ep/n_episodes)
            for _ in 1:env.n_steps
                is_terminal(env) && break
                a  = rand() < eps ? rand(1:n_actions) : argmax(forward_pass(net, s))
                r, done = step_environment!(env, a)
                ns = get_state(env)
                push_experience!(buf, s, a, r, ns, done)
                s  = ns
                if buf.size >= 32
                    idxs, _ = sample_batch(buf, 32)
                    for idx in idxs
                        si  = buf.states[:, idx]
                        nsi = buf.next_states[:, idx]
                        ri  = buf.rewards[idx]; di = buf.dones[idx]
                        ai  = buf.actions[idx]
                        tgt_val = double_dqn_target(net, tgt, nsi, ri, di)
                        q_pred  = forward_pass(net, si)
                        target_q = copy(q_pred); target_q[ai] = tgt_val
                        backward_pass!(net, si, target_q; lr=1e-3)
                    end
                end
            end
            if mod(ep, 5) == 0
                network_update!(tgt, net; tau=1.0)
            end
        end
        push!(networks, net)
    end
    return PolicyEnsemble(networks, n_actions)
end

end  # module ReinforcementLearning
