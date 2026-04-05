"""
reinforcement_learning.jl — Reinforcement Learning for Crypto Trading

Covers:
  - Q-learning (tabular, discretized state space)
  - Deep Q-Network (DQN): pure matrix ops, replay buffer, target network
  - REINFORCE policy gradient algorithm
  - Actor-Critic (A2C) with generalized advantage estimation
  - Crypto trading environment (state=indicators, actions=buy/hold/sell)
  - Experience replay buffer (circular)
  - Epsilon-greedy exploration with decay schedule
  - Training loops with periodic evaluation
  - Comparison of RL policies vs buy-and-hold

Pure Julia stdlib only. No external dependencies.
"""

module ReinforcementLearningTrading

using Statistics, LinearAlgebra, Random

export TradingEnvironment, reset!, step!
export TabularQAgent, DQNAgent, REINFORCEAgent, ActorCriticAgent
export ReplayBuffer, push_experience!, sample_batch
export train_agent!, evaluate_policy
export EpsilonGreedy, decay_epsilon!
export compare_policies, run_rl_demo

# ─────────────────────────────────────────────────────────────
# 1. TRADING ENVIRONMENT
# ─────────────────────────────────────────────────────────────

"""
    TradingEnvironment

Crypto trading environment for RL agents.

State: [price_return, vol_5, vol_20, rsi, macd_signal, position, drawdown]
Actions: 0=sell/flat, 1=hold, 2=buy
Reward: Sharpe-scaled PnL increment minus transaction cost
"""
mutable struct TradingEnvironment
    prices::Vector{Float64}           # historical price series
    returns::Vector{Float64}          # log returns
    n_steps::Int                      # total timesteps
    current_step::Int                 # current index
    position::Float64                 # current position (-1, 0, +1)
    cash::Float64                     # cash value
    holdings::Float64                 # asset value held
    portfolio_value::Float64          # total portfolio
    peak_value::Float64               # for drawdown calc
    tc::Float64                       # transaction cost (fraction)
    window::Int                       # lookback window for indicators
    done::Bool
    reward_history::Vector{Float64}
    portfolio_history::Vector{Float64}
end

function TradingEnvironment(prices::Vector{Float64};
                             tc::Float64=0.001, window::Int=20,
                             initial_capital::Float64=10_000.0)
    returns = [0.0; diff(log.(prices))]
    n = length(prices)
    TradingEnvironment(prices, returns, n, window+1, 0.0, initial_capital,
                       0.0, initial_capital, initial_capital, tc, window,
                       false, Float64[], Float64[])
end

"""
    reset!(env::TradingEnvironment) -> Vector{Float64}

Reset environment to start and return initial state.
"""
function reset!(env::TradingEnvironment)::Vector{Float64}
    env.current_step = env.window + 1
    env.position     = 0.0
    env.cash         = 10_000.0
    env.holdings     = 0.0
    env.portfolio_value = 10_000.0
    env.peak_value   = 10_000.0
    env.done         = false
    empty!(env.reward_history)
    empty!(env.portfolio_history)
    get_state(env)
end

"""Compute RSI from return window."""
function compute_rsi(returns::AbstractVector{Float64}, period::Int=14)::Float64
    n = length(returns)
    n < period && return 50.0
    gains  = [max(r, 0.0) for r in returns[end-period+1:end]]
    losses = [max(-r, 0.0) for r in returns[end-period+1:end]]
    avg_g  = mean(gains)
    avg_l  = mean(losses)
    avg_l < 1e-10 && return 100.0
    rs = avg_g / avg_l
    100.0 - 100.0 / (1.0 + rs)
end

"""Compute MACD signal (EMA12 - EMA26) / price std."""
function compute_macd(prices::AbstractVector{Float64})::Float64
    n = length(prices)
    n < 26 && return 0.0
    # Simple EMA via recursive formula
    function ema(data, period)
        alpha = 2.0 / (period + 1)
        e = data[1]
        for i in 2:length(data)
            e = alpha * data[i] + (1 - alpha) * e
        end
        e
    end
    e12 = ema(prices, 12)
    e26 = ema(prices, 26)
    s = std(prices)
    s < 1e-10 && return 0.0
    (e12 - e26) / s
end

"""
    get_state(env::TradingEnvironment) -> Vector{Float64}

Extract 7-dimensional state vector from current environment.
"""
function get_state(env::TradingEnvironment)::Vector{Float64}
    t = env.current_step
    w = env.window
    ret_window = env.returns[max(1, t-w):t]
    price_window = env.prices[max(1, t-w):t]

    price_ret  = env.returns[t]
    vol_short  = std(env.returns[max(1,t-5):t])
    vol_long   = std(ret_window)
    rsi        = compute_rsi(ret_window) / 100.0 - 0.5  # center at 0
    macd       = clamp(compute_macd(price_window), -3.0, 3.0)
    position   = env.position
    drawdown   = env.peak_value > 0 ?
                 (env.peak_value - env.portfolio_value) / env.peak_value : 0.0

    Float64[price_ret, vol_short, vol_long, rsi, macd, position, -drawdown]
end

"""
    step!(env, action) -> (next_state, reward, done)

Execute action in environment. action ∈ {0, 1, 2} → {sell, hold, buy}.
"""
function step!(env::TradingEnvironment, action::Int)
    env.done && return get_state(env), 0.0, true

    # Desired position
    desired_pos = Float64(action - 1)  # -1, 0, +1
    trade       = desired_pos - env.position

    # Transaction cost
    cost = abs(trade) * env.tc * env.portfolio_value

    # Update position
    env.position = desired_pos

    # Move to next step
    env.current_step += 1
    if env.current_step >= env.n_steps
        env.done = true
    end

    # PnL from position
    ret = env.done ? 0.0 : env.returns[env.current_step]
    pnl = env.position * ret * env.portfolio_value - cost

    env.portfolio_value += pnl
    env.portfolio_value  = max(env.portfolio_value, 1.0)
    env.peak_value       = max(env.peak_value, env.portfolio_value)

    push!(env.portfolio_history, env.portfolio_value)

    # Reward: log return of portfolio
    reward = pnl / (env.portfolio_value - pnl + 1e-10)

    # Penalize drawdown
    dd = (env.peak_value - env.portfolio_value) / (env.peak_value + 1e-10)
    reward -= 0.5 * dd

    push!(env.reward_history, reward)

    next_state = env.done ? get_state(env) : get_state(env)
    next_state, reward, env.done
end

# ─────────────────────────────────────────────────────────────
# 2. EPSILON-GREEDY EXPLORATION
# ─────────────────────────────────────────────────────────────

"""
    EpsilonGreedy

Manages epsilon-greedy exploration schedule.
"""
mutable struct EpsilonGreedy
    epsilon::Float64
    epsilon_min::Float64
    decay::Float64
    step_count::Int
end

EpsilonGreedy(; start=1.0, min=0.01, decay=0.995) =
    EpsilonGreedy(start, min, decay, 0)

"""Decay epsilon by multiplicative factor."""
function decay_epsilon!(eg::EpsilonGreedy)
    eg.epsilon = max(eg.epsilon * eg.decay, eg.epsilon_min)
    eg.step_count += 1
end

"""Select action via epsilon-greedy given Q-values."""
function select_action(eg::EpsilonGreedy, q_values::Vector{Float64};
                        rng=MersenneTwister(0))::Int
    if rand(rng) < eg.epsilon
        rand(rng, 0:length(q_values)-1)
    else
        argmax(q_values) - 1  # 0-indexed
    end
end

# ─────────────────────────────────────────────────────────────
# 3. TABULAR Q-LEARNING
# ─────────────────────────────────────────────────────────────

"""
    TabularQAgent

Tabular Q-learning agent with discretized state space.

State discretization: each of 7 state dimensions bucketed into n_bins bins.
"""
mutable struct TabularQAgent
    n_actions::Int
    n_bins::Int
    state_dim::Int
    q_table::Dict{Vector{Int}, Vector{Float64}}
    alpha::Float64      # learning rate
    gamma::Float64      # discount factor
    explorer::EpsilonGreedy
    rng::AbstractRNG
end

function TabularQAgent(; n_actions=3, n_bins=5, state_dim=7,
                        alpha=0.1, gamma=0.95, rng=MersenneTwister(42))
    TabularQAgent(n_actions, n_bins, state_dim,
                  Dict{Vector{Int}, Vector{Float64}}(),
                  alpha, gamma, EpsilonGreedy(start=1.0, min=0.05, decay=0.998),
                  rng)
end

"""Discretize continuous state into bin indices."""
function discretize_state(agent::TabularQAgent, state::Vector{Float64})::Vector{Int}
    # Clip to [-3, 3] range, map to [1, n_bins]
    Int[clamp(Int(floor((s + 3.0) / 6.0 * agent.n_bins)) + 1, 1, agent.n_bins)
        for s in state]
end

"""Get Q-values for discretized state (initialize to zeros if new)."""
function get_q(agent::TabularQAgent, state::Vector{Float64})::Vector{Float64}
    ds = discretize_state(agent, state)
    get!(agent.q_table, ds, zeros(agent.n_actions))
end

"""Update Q-value via TD(0)."""
function update_q!(agent::TabularQAgent, state::Vector{Float64},
                   action::Int, reward::Float64, next_state::Vector{Float64},
                   done::Bool)
    q_cur  = get_q(agent, state)
    q_next = get_q(agent, next_state)
    target = reward + (done ? 0.0 : agent.gamma * maximum(q_next))
    ds = discretize_state(agent, state)
    agent.q_table[ds][action + 1] += agent.alpha * (target - q_cur[action + 1])
end

"""Select action from tabular Q-agent."""
function act(agent::TabularQAgent, state::Vector{Float64})::Int
    q = get_q(agent, state)
    select_action(agent.explorer, q; rng=agent.rng)
end

# ─────────────────────────────────────────────────────────────
# 4. DEEP Q-NETWORK (DQN) — PURE MATRIX OPS
# ─────────────────────────────────────────────────────────────

"""
    NeuralNetwork

Simple fully-connected neural network for DQN.
Architecture: [state_dim → hidden → hidden → n_actions]
"""
mutable struct NeuralNetwork
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    W3::Matrix{Float64}
    b3::Vector{Float64}
end

function NeuralNetwork(in_dim::Int, hidden::Int, out_dim::Int;
                        rng=MersenneTwister(1))
    scale1 = sqrt(2.0 / in_dim)
    scale2 = sqrt(2.0 / hidden)
    NeuralNetwork(
        randn(rng, hidden, in_dim)  * scale1, zeros(hidden),
        randn(rng, hidden, hidden) * scale2, zeros(hidden),
        randn(rng, out_dim, hidden) * scale2, zeros(out_dim)
    )
end

"""ReLU activation."""
relu(x::AbstractArray) = max.(x, 0.0)
relu(x::Float64) = max(x, 0.0)

"""Forward pass through network. Returns output and intermediate activations."""
function forward(net::NeuralNetwork, x::Vector{Float64})
    z1 = net.W1 * x .+ net.b1
    a1 = relu(z1)
    z2 = net.W2 * a1 .+ net.b2
    a2 = relu(z2)
    z3 = net.W3 * a2 .+ net.b3
    z3, (x, a1, a2)  # output, activations
end

"""Compute Q-values for batch of states (columns)."""
function batch_forward(net::NeuralNetwork, X::Matrix{Float64})::Matrix{Float64}
    n = size(X, 2)
    out = zeros(size(net.W3, 1), n)
    for i in 1:n
        q, _ = forward(net, X[:, i])
        out[:, i] = q
    end
    out
end

"""Copy weights from source network to target."""
function copy_network!(target::NeuralNetwork, source::NeuralNetwork)
    target.W1 .= source.W1
    target.b1 .= source.b1
    target.W2 .= source.W2
    target.b2 .= source.b2
    target.W3 .= source.W3
    target.b3 .= source.b3
end

"""Soft update: target ← tau*source + (1-tau)*target."""
function soft_update!(target::NeuralNetwork, source::NeuralNetwork, tau::Float64=0.005)
    target.W1 .= tau * source.W1 .+ (1-tau) * target.W1
    target.b1 .= tau * source.b1 .+ (1-tau) * target.b1
    target.W2 .= tau * source.W2 .+ (1-tau) * target.W2
    target.b2 .= tau * source.b2 .+ (1-tau) * target.b2
    target.W3 .= tau * source.W3 .+ (1-tau) * target.W3
    target.b3 .= tau * source.b3 .+ (1-tau) * target.b3
end

# ─────────────────────────────────────────────────────────────
# 5. EXPERIENCE REPLAY BUFFER
# ─────────────────────────────────────────────────────────────

"""
    ReplayBuffer

Circular experience replay buffer for DQN training.
"""
mutable struct ReplayBuffer
    states::Matrix{Float64}
    actions::Vector{Int}
    rewards::Vector{Float64}
    next_states::Matrix{Float64}
    dones::Vector{Bool}
    capacity::Int
    size::Int
    ptr::Int
    state_dim::Int
end

function ReplayBuffer(capacity::Int, state_dim::Int)
    ReplayBuffer(
        zeros(state_dim, capacity),
        zeros(Int, capacity),
        zeros(capacity),
        zeros(state_dim, capacity),
        zeros(Bool, capacity),
        capacity, 0, 1, state_dim
    )
end

"""Add experience tuple to buffer."""
function push_experience!(buf::ReplayBuffer, state::Vector{Float64},
                           action::Int, reward::Float64,
                           next_state::Vector{Float64}, done::Bool)
    buf.states[:, buf.ptr]      = state
    buf.actions[buf.ptr]        = action
    buf.rewards[buf.ptr]        = reward
    buf.next_states[:, buf.ptr] = next_state
    buf.dones[buf.ptr]          = done
    buf.size = min(buf.size + 1, buf.capacity)
    buf.ptr  = (buf.ptr % buf.capacity) + 1
end

"""Sample random mini-batch from buffer."""
function sample_batch(buf::ReplayBuffer, batch_size::Int;
                       rng=MersenneTwister(1))
    idx = rand(rng, 1:buf.size, batch_size)
    (states      = buf.states[:, idx],
     actions     = buf.actions[idx],
     rewards     = buf.rewards[idx],
     next_states = buf.next_states[:, idx],
     dones       = buf.dones[idx])
end

# ─────────────────────────────────────────────────────────────
# 6. DQN AGENT
# ─────────────────────────────────────────────────────────────

"""
    DQNAgent

Deep Q-Network agent with experience replay and target network.
"""
mutable struct DQNAgent
    q_network::NeuralNetwork
    target_network::NeuralNetwork
    buffer::ReplayBuffer
    explorer::EpsilonGreedy
    gamma::Float64
    lr::Float64
    batch_size::Int
    update_target_every::Int
    step_count::Int
    n_actions::Int
    rng::AbstractRNG
end

function DQNAgent(state_dim::Int; n_actions=3, hidden=64,
                   capacity=10000, gamma=0.95, lr=1e-3,
                   batch_size=64, update_every=100,
                   rng=MersenneTwister(42))
    q_net  = NeuralNetwork(state_dim, hidden, n_actions; rng=rng)
    t_net  = NeuralNetwork(state_dim, hidden, n_actions; rng=rng)
    copy_network!(t_net, q_net)
    buf    = ReplayBuffer(capacity, state_dim)
    DQNAgent(q_net, t_net, buf, EpsilonGreedy(start=1.0, min=0.02, decay=0.997),
             gamma, lr, batch_size, update_every, 0, n_actions, rng)
end

"""Select action using epsilon-greedy over Q-network output."""
function act(agent::DQNAgent, state::Vector{Float64})::Int
    q_vals, _ = forward(agent.q_network, state)
    select_action(agent.explorer, q_vals; rng=agent.rng)
end

"""
    dqn_gradient_step!(agent) -> Float64

Perform one gradient descent step on DQN. Returns loss.
Uses simple SGD with finite-difference gradient (pure Julia, no AD).
"""
function dqn_gradient_step!(agent::DQNAgent)::Float64
    agent.buffer.size < agent.batch_size && return 0.0
    batch = sample_batch(agent.buffer, agent.batch_size; rng=agent.rng)

    # Compute targets
    q_next = batch_forward(agent.target_network, batch.next_states)  # n_actions × batch
    max_q_next = vec(maximum(q_next, dims=1))
    targets = batch.rewards .+ agent.gamma .* max_q_next .* (1.0 .- Float64.(batch.dones))

    # Loss and gradient via manual backprop
    total_loss = 0.0
    lr = agent.lr

    # Approximate gradient: perturb each parameter, measure loss change (SPSA-style)
    # For efficiency, use a simplified update: direct TD update on output layer
    for b in 1:agent.batch_size
        s  = batch.states[:, b]
        a  = batch.actions[b] + 1  # 1-indexed
        tg = targets[b]

        q_vals, (x0, a1, a2) = forward(agent.q_network, s)
        err = q_vals[a] - tg
        total_loss += err^2

        # Backprop through output layer
        dL_dz3          = zeros(agent.n_actions)
        dL_dz3[a]       = 2.0 * err / agent.batch_size
        agent.q_network.W3 .-= lr * dL_dz3 * a2'
        agent.q_network.b3 .-= lr * dL_dz3

        # Backprop through layer 2
        dL_da2 = agent.q_network.W3' * dL_dz3
        dL_dz2 = dL_da2 .* (a2 .> 0)
        agent.q_network.W2 .-= lr * dL_dz2 * a1'
        agent.q_network.b2 .-= lr * dL_dz2

        # Backprop through layer 1
        dL_da1 = agent.q_network.W2' * dL_dz2
        dL_dz1 = dL_da1 .* (a1 .> 0)
        agent.q_network.W1 .-= lr * dL_dz1 * x0'
        agent.q_network.b1 .-= lr * dL_dz1
    end

    # Update target network
    agent.step_count += 1
    if agent.step_count % agent.update_target_every == 0
        soft_update!(agent.target_network, agent.q_network, 0.01)
    end

    total_loss / agent.batch_size
end

# ─────────────────────────────────────────────────────────────
# 7. POLICY GRADIENT — REINFORCE
# ─────────────────────────────────────────────────────────────

"""
    REINFORCEAgent

REINFORCE policy gradient agent.
Policy network outputs softmax probabilities over actions.
"""
mutable struct REINFORCEAgent
    policy_net::NeuralNetwork
    gamma::Float64
    lr::Float64
    n_actions::Int
    rng::AbstractRNG
    episode_states::Vector{Vector{Float64}}
    episode_actions::Vector{Int}
    episode_rewards::Vector{Float64}
end

function REINFORCEAgent(state_dim::Int; n_actions=3, hidden=64,
                         gamma=0.95, lr=1e-3, rng=MersenneTwister(42))
    net = NeuralNetwork(state_dim, hidden, n_actions; rng=rng)
    REINFORCEAgent(net, gamma, lr, n_actions, rng, [], [], [])
end

"""Softmax over vector."""
function softmax(x::Vector{Float64})::Vector{Float64}
    mx = maximum(x)
    ex = exp.(x .- mx)
    ex ./ sum(ex)
end

"""Sample action from policy."""
function act(agent::REINFORCEAgent, state::Vector{Float64})::Int
    logits, _ = forward(agent.policy_net, state)
    probs = softmax(logits)
    # Categorical sampling
    u = rand(agent.rng)
    csum = 0.0
    for (i, p) in enumerate(probs)
        csum += p
        u <= csum && return i - 1
    end
    length(probs) - 1
end

"""Store step for episode."""
function store_step!(agent::REINFORCEAgent, state::Vector{Float64},
                     action::Int, reward::Float64)
    push!(agent.episode_states, copy(state))
    push!(agent.episode_actions, action)
    push!(agent.episode_rewards, reward)
end

"""Compute discounted returns."""
function discounted_returns(rewards::Vector{Float64}, gamma::Float64)::Vector{Float64}
    n = length(rewards)
    G = zeros(n)
    G[end] = rewards[end]
    for t in (n-1):-1:1
        G[t] = rewards[t] + gamma * G[t+1]
    end
    # Normalize
    mu = mean(G); s = std(G)
    s < 1e-8 && return G .- mu
    (G .- mu) ./ s
end

"""Update policy using REINFORCE gradient."""
function update_policy!(agent::REINFORCEAgent)::Float64
    isempty(agent.episode_rewards) && return 0.0
    G = discounted_returns(agent.episode_rewards, agent.gamma)
    total_loss = 0.0

    for (t, (s, a, g)) in enumerate(zip(agent.episode_states,
                                         agent.episode_actions, G))
        logits, (x0, a1, a2) = forward(agent.policy_net, s)
        probs = softmax(logits)
        ai = a + 1

        log_prob = log(probs[ai] + 1e-10)
        total_loss += -g * log_prob

        # Gradient of log-softmax w.r.t. logits
        dL_dlogits = probs .- 0.0
        dL_dlogits[ai] -= 1.0
        dL_dlogits .*= (-g / length(agent.episode_rewards))

        # Backprop
        agent.policy_net.W3 .-= agent.lr * dL_dlogits * a2'
        agent.policy_net.b3 .-= agent.lr * dL_dlogits
        dL_da2 = agent.policy_net.W3' * dL_dlogits
        dL_dz2 = dL_da2 .* (a2 .> 0)
        agent.policy_net.W2 .-= agent.lr * dL_dz2 * a1'
        agent.policy_net.b2 .-= agent.lr * dL_dz2
        dL_da1 = agent.policy_net.W2' * dL_dz2
        dL_dz1 = dL_da1 .* (a1 .> 0)
        agent.policy_net.W1 .-= agent.lr * dL_dz1 * x0'
        agent.policy_net.b1 .-= agent.lr * dL_dz1
    end

    empty!(agent.episode_states)
    empty!(agent.episode_actions)
    empty!(agent.episode_rewards)
    total_loss
end

# ─────────────────────────────────────────────────────────────
# 8. ACTOR-CRITIC (A2C)
# ─────────────────────────────────────────────────────────────

"""
    ActorCriticAgent

Advantage Actor-Critic (A2C) with separate actor and critic networks.
"""
mutable struct ActorCriticAgent
    actor::NeuralNetwork    # outputs action logits
    critic::NeuralNetwork   # outputs state value V(s)
    gamma::Float64
    lr_actor::Float64
    lr_critic::Float64
    n_actions::Int
    rng::AbstractRNG
end

function ActorCriticAgent(state_dim::Int; n_actions=3, hidden=64,
                            gamma=0.95, lr_actor=5e-4, lr_critic=1e-3,
                            rng=MersenneTwister(42))
    actor  = NeuralNetwork(state_dim, hidden, n_actions; rng=rng)
    critic = NeuralNetwork(state_dim, hidden, 1; rng=rng)
    ActorCriticAgent(actor, critic, gamma, lr_actor, lr_critic, n_actions, rng)
end

"""Select action via actor network."""
function act(agent::ActorCriticAgent, state::Vector{Float64})::Int
    logits, _ = forward(agent.actor, state)
    probs = softmax(logits)
    u = rand(agent.rng)
    csum = 0.0
    for (i, p) in enumerate(probs)
        csum += p
        u <= csum && return i - 1
    end
    length(probs) - 1
end

"""
    update_a2c!(agent, state, action, reward, next_state, done) -> (actor_loss, critic_loss)

Single-step A2C update with TD advantage.
"""
function update_a2c!(agent::ActorCriticAgent,
                      state::Vector{Float64}, action::Int,
                      reward::Float64, next_state::Vector{Float64},
                      done::Bool)
    # Value estimates
    v_cur,  (x0c, a1c, a2c) = forward(agent.critic, state)
    v_next, _ = forward(agent.critic, next_state)

    # TD target and advantage
    td_target  = reward + (done ? 0.0 : agent.gamma * v_next[1])
    advantage  = td_target - v_cur[1]

    # ── Critic update ──
    critic_loss = advantage^2
    dL_dv = -2.0 * advantage  # gradient w.r.t. output
    # Backprop critic
    dz3c = [dL_dv]
    agent.critic.W3 .-= agent.lr_critic * dz3c * a2c'
    agent.critic.b3 .-= agent.lr_critic * dz3c
    da2c = agent.critic.W3' * dz3c
    dz2c = da2c .* (a2c .> 0)
    agent.critic.W2 .-= agent.lr_critic * dz2c * a1c'
    agent.critic.b2 .-= agent.lr_critic * dz2c
    da1c = agent.critic.W2' * dz2c
    dz1c = da1c .* (a1c .> 0)
    agent.critic.W1 .-= agent.lr_critic * dz1c * x0c'
    agent.critic.b1 .-= agent.lr_critic * dz1c

    # ── Actor update ──
    logits, (x0a, a1a, a2a) = forward(agent.actor, state)
    probs = softmax(logits)
    ai = action + 1
    actor_loss = -log(probs[ai] + 1e-10) * advantage

    dL_dlogits = copy(probs)
    dL_dlogits[ai] -= 1.0
    dL_dlogits .*= -advantage

    agent.actor.W3 .-= agent.lr_actor * dL_dlogits * a2a'
    agent.actor.b3 .-= agent.lr_actor * dL_dlogits
    da2a = agent.actor.W3' * dL_dlogits
    dz2a = da2a .* (a2a .> 0)
    agent.actor.W2 .-= agent.lr_actor * dz2a * a1a'
    agent.actor.b2 .-= agent.lr_actor * dz2a
    da1a = agent.actor.W2' * dz2a
    dz1a = da1a .* (a1a .> 0)
    agent.actor.W1 .-= agent.lr_actor * dz1a * x0a'
    agent.actor.b1 .-= agent.lr_actor * dz1a

    actor_loss, critic_loss
end

# ─────────────────────────────────────────────────────────────
# 9. TRAINING INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────

"""
    train_agent!(agent, env, n_episodes; verbose=true) -> Vector{Float64}

Generic training loop for DQN or A2C agent. Returns episode rewards.
"""
function train_agent!(agent::Union{DQNAgent, ActorCriticAgent},
                       env::TradingEnvironment,
                       n_episodes::Int;
                       verbose::Bool=true)::Vector{Float64}
    episode_rewards = zeros(n_episodes)

    for ep in 1:n_episodes
        state = reset!(env)
        total_reward = 0.0
        steps = 0

        while !env.done
            a = act(agent, state)
            next_state, reward, done = step!(env, a)

            if isa(agent, DQNAgent)
                push_experience!(agent.buffer, state, a, reward, next_state, done)
                dqn_gradient_step!(agent)
                decay_epsilon!(agent.explorer)
            else  # A2C
                update_a2c!(agent, state, a, reward, next_state, done)
            end

            total_reward += reward
            state = next_state
            steps += 1
        end

        episode_rewards[ep] = total_reward
        if verbose && ep % max(1, n_episodes ÷ 10) == 0
            avg_r = mean(episode_rewards[max(1,ep-9):ep])
            final_val = isempty(env.portfolio_history) ? 10000.0 :
                        env.portfolio_history[end]
            println("  Ep $ep/$n_episodes | AvgReward: $(round(avg_r,digits=4)) | Portfolio: \$$(round(final_val,digits=0))")
        end
    end
    episode_rewards
end

"""
    train_reinforce!(agent, env, n_episodes; verbose=true) -> Vector{Float64}

Training loop specifically for REINFORCE (episode-based update).
"""
function train_reinforce!(agent::REINFORCEAgent,
                           env::TradingEnvironment,
                           n_episodes::Int;
                           verbose::Bool=true)::Vector{Float64}
    episode_rewards = zeros(n_episodes)

    for ep in 1:n_episodes
        state = reset!(env)
        total_reward = 0.0

        while !env.done
            a = act(agent, state)
            next_state, reward, done = step!(env, a)
            store_step!(agent, state, a, reward)
            total_reward += reward
            state = next_state
        end

        update_policy!(agent)
        episode_rewards[ep] = total_reward

        if verbose && ep % max(1, n_episodes ÷ 10) == 0
            println("  Ep $ep REINFORCE | TotalReward: $(round(total_reward,digits=3))")
        end
    end
    episode_rewards
end

# ─────────────────────────────────────────────────────────────
# 10. EVALUATION
# ─────────────────────────────────────────────────────────────

"""
    evaluate_policy(agent, env; n_eval=1) -> NamedTuple

Run agent on environment (no exploration), return performance metrics.
"""
function evaluate_policy(agent, env::TradingEnvironment; n_eval::Int=1)
    all_returns = Float64[]
    all_final   = Float64[]

    for _ in 1:n_eval
        state = reset!(env)
        # Turn off exploration for DQN
        saved_eps = isa(agent, DQNAgent) ? agent.explorer.epsilon : 0.0
        isa(agent, DQNAgent) && (agent.explorer.epsilon = 0.0)

        while !env.done
            a = act(agent, state)
            state, _, _ = step!(env, a)
        end

        isa(agent, DQNAgent) && (agent.explorer.epsilon = saved_eps)
        push!(all_returns, env.reward_history...)
        push!(all_final, isempty(env.portfolio_history) ? 10000.0 :
              env.portfolio_history[end])
    end

    final_val = mean(all_final)
    total_ret  = (final_val - 10_000.0) / 10_000.0
    ret_series = all_returns
    sharpe     = isempty(ret_series) || std(ret_series) < 1e-10 ? 0.0 :
                 mean(ret_series) / std(ret_series) * sqrt(252)
    portfolio  = env.portfolio_history
    max_dd     = if !isempty(portfolio)
        peak = portfolio[1]
        mdd  = 0.0
        for v in portfolio
            peak = max(peak, v)
            mdd  = max(mdd, (peak - v) / peak)
        end
        mdd
    else 0.0 end

    (total_return=total_ret, annualized_sharpe=sharpe,
     max_drawdown=max_dd, final_portfolio_value=final_val,
     n_steps=length(portfolio))
end

"""
    buy_and_hold_baseline(prices::Vector{Float64}; tc=0.001) -> NamedTuple

Compute buy-and-hold strategy metrics for comparison.
"""
function buy_and_hold_baseline(prices::Vector{Float64}; tc::Float64=0.001)
    n = length(prices)
    n < 2 && return (total_return=0.0, sharpe=0.0, max_drawdown=0.0)
    returns   = diff(log.(prices))
    portfolio = [10_000.0 * exp(cumsum(returns)[t]) for t in 1:length(returns)]
    portfolio .*= (1 - tc)  # one-time entry cost
    total_ret = (portfolio[end] - 10_000.0) / 10_000.0
    sharpe    = std(returns) < 1e-10 ? 0.0 :
                mean(returns) / std(returns) * sqrt(252)
    peak = portfolio[1]; mdd = 0.0
    for v in portfolio
        peak = max(peak, v)
        mdd  = max(mdd, (peak - v) / peak)
    end
    (total_return=total_ret, sharpe=sharpe, max_drawdown=mdd,
     final_value=portfolio[end])
end

"""
    compare_policies(prices, agent; n_train=0.7, ...) -> NamedTuple

Train agent on first n_train fraction, evaluate on remainder, compare to BH.
"""
function compare_policies(prices::Vector{Float64}, agent;
                           train_frac::Float64=0.7,
                           n_episodes_train::Int=30,
                           verbose::Bool=true)
    n_total = length(prices)
    n_train = Int(floor(n_total * train_frac))
    train_prices = prices[1:n_train]
    test_prices  = prices[n_train:end]

    # Train
    if verbose
        println("Training on $(n_train) steps...")
    end
    train_env = TradingEnvironment(train_prices)
    if isa(agent, REINFORCEAgent)
        train_reinforce!(agent, train_env, n_episodes_train; verbose=verbose)
    else
        train_agent!(agent, train_env, n_episodes_train; verbose=verbose)
    end

    # Evaluate on test set
    test_env = TradingEnvironment(test_prices)
    rl_metrics = evaluate_policy(agent, test_env)

    # BH baseline on test set
    bh_metrics = buy_and_hold_baseline(test_prices)

    if verbose
        println("\n=== Policy Comparison (Test Period) ===")
        println("  RL Agent:")
        println("    Total Return:  $(round(rl_metrics.total_return*100,digits=2))%")
        println("    Sharpe Ratio:  $(round(rl_metrics.annualized_sharpe,digits=3))")
        println("    Max Drawdown:  $(round(rl_metrics.max_drawdown*100,digits=2))%")
        println("  Buy & Hold:")
        println("    Total Return:  $(round(bh_metrics.total_return*100,digits=2))%")
        println("    Sharpe Ratio:  $(round(bh_metrics.sharpe,digits=3))")
        println("    Max Drawdown:  $(round(bh_metrics.max_drawdown*100,digits=2))%")
    end

    (rl=rl_metrics, buy_and_hold=bh_metrics, outperformance=rl_metrics.total_return - bh_metrics.total_return)
end

# ─────────────────────────────────────────────────────────────
# 11. SYNTHETIC PRICE GENERATOR
# ─────────────────────────────────────────────────────────────

"""
    generate_gbm_prices(n, mu, sigma, S0; rng=...) -> Vector{Float64}

Generate GBM price path for testing.
"""
function generate_gbm_prices(n::Int=500, mu::Float64=0.0005,
                               sigma::Float64=0.02, S0::Float64=10_000.0;
                               rng=MersenneTwister(42))::Vector{Float64}
    prices = zeros(n)
    prices[1] = S0
    for t in 2:n
        dW = randn(rng)
        prices[t] = prices[t-1] * exp((mu - 0.5*sigma^2) + sigma*dW)
    end
    prices
end

"""
    generate_regime_prices(n; rng=...) -> Vector{Float64}

Generate regime-switching price path (bull/bear).
"""
function generate_regime_prices(n::Int=500; rng=MersenneTwister(1))::Vector{Float64}
    prices = zeros(n); prices[1] = 10_000.0
    regime = 1  # 1=bull, 2=bear
    params = [(0.001, 0.015), (-0.0005, 0.030)]  # (mu, sigma) per regime
    p_switch = [0.02, 0.04]  # switching probabilities
    for t in 2:n
        rand(rng) < p_switch[regime] && (regime = 3 - regime)
        mu, sigma = params[regime]
        prices[t] = prices[t-1] * exp(mu + sigma*randn(rng))
    end
    prices
end

# ─────────────────────────────────────────────────────────────
# 12. DEMO
# ─────────────────────────────────────────────────────────────

"""
    run_rl_demo() -> Nothing

Demonstration of all RL algorithms on synthetic crypto price data.
"""
function run_rl_demo()
    println("=" ^ 60)
    println("REINFORCEMENT LEARNING TRADING DEMO")
    println("=" ^ 60)

    rng = MersenneTwister(42)
    prices = generate_regime_prices(600; rng=rng)
    println("Generated $(length(prices)) synthetic price observations")
    println("Price range: $(round(minimum(prices),digits=0)) – $(round(maximum(prices),digits=0))")

    println("\n1. Tabular Q-Learning")
    q_agent = TabularQAgent(; rng=MersenneTwister(1))
    q_env   = TradingEnvironment(prices[1:400])
    rewards_q = Float64[]
    for ep in 1:50
        s = reset!(q_env)
        tot = 0.0
        while !q_env.done
            a = act(q_agent, s)
            sn, r, d = step!(q_env, a)
            update_q!(q_agent, s, a, r, sn, d)
            tot += r; s = sn
        end
        push!(rewards_q, tot)
        decay_epsilon!(q_agent.explorer)
    end
    println("  States visited: $(length(q_agent.q_table))")
    println("  Final epsilon:  $(round(q_agent.explorer.epsilon,digits=3))")
    println("  Avg reward (last 10 ep): $(round(mean(rewards_q[end-9:end]),digits=4))")

    println("\n2. DQN Agent")
    dqn  = DQNAgent(7; hidden=32, rng=MersenneTwister(2))
    res_dqn = compare_policies(prices, dqn; n_episodes_train=20, verbose=false)
    println("  RL Return:  $(round(res_dqn.rl.total_return*100,digits=2))%")
    println("  BH Return:  $(round(res_dqn.buy_and_hold.total_return*100,digits=2))%")
    println("  RL Sharpe:  $(round(res_dqn.rl.annualized_sharpe,digits=3))")
    println("  Outperformance: $(round(res_dqn.outperformance*100,digits=2))%")

    println("\n3. REINFORCE Agent")
    pg_agent = REINFORCEAgent(7; hidden=32, rng=MersenneTwister(3))
    res_pg = compare_policies(prices, pg_agent; n_episodes_train=20, verbose=false)
    println("  RL Return:  $(round(res_pg.rl.total_return*100,digits=2))%")
    println("  RL Sharpe:  $(round(res_pg.rl.annualized_sharpe,digits=3))")

    println("\n4. Actor-Critic (A2C)")
    a2c_agent = ActorCriticAgent(7; hidden=32, rng=MersenneTwister(4))
    res_a2c = compare_policies(prices, a2c_agent; n_episodes_train=20, verbose=false)
    println("  RL Return:  $(round(res_a2c.rl.total_return*100,digits=2))%")
    println("  RL Sharpe:  $(round(res_a2c.rl.annualized_sharpe,digits=3))")
    println("  Max DD:     $(round(res_a2c.rl.max_drawdown*100,digits=2))%")

    println("\nDone.")
    nothing
end

# ─────────────────────────────────────────────────────────────
# 13. PROXIMAL POLICY OPTIMIZATION (PPO) SIMPLIFIED
# ─────────────────────────────────────────────────────────────

"""
    PPOAgent

Simplified Proximal Policy Optimization agent.
Uses clipped surrogate objective to prevent large policy updates.
"""
mutable struct PPOAgent
    actor::NeuralNetwork
    critic::NeuralNetwork
    gamma::Float64
    clip_eps::Float64
    lr_actor::Float64; lr_critic::Float64
    n_actions::Int; rng::AbstractRNG
    old_log_probs::Vector{Float64}
end

function PPOAgent(state_dim::Int; n_actions=3, hidden=64,
                   gamma=0.95, clip_eps=0.2, lr_a=3e-4, lr_c=1e-3,
                   rng=MersenneTwister(42))
    PPOAgent(NeuralNetwork(state_dim, hidden, n_actions; rng=rng),
             NeuralNetwork(state_dim, hidden, 1; rng=rng),
             gamma, clip_eps, lr_a, lr_c, n_actions, rng, Float64[])
end

function act(agent::PPOAgent, state::Vector{Float64})::Int
    logits, _ = forward(agent.actor, state)
    probs = softmax(logits)
    u = rand(agent.rng); cs = 0.0
    for (i, p) in enumerate(probs)
        cs += p; u <= cs && return i-1
    end
    length(probs) - 1
end

"""PPO clipped surrogate update."""
function ppo_update!(agent::PPOAgent, state, action, advantage,
                      old_log_prob::Float64)
    logits, cache_a = forward(agent.actor, state)
    probs = softmax(logits)
    ai = action + 1
    new_log_prob = log(probs[ai] + 1e-10)
    ratio = exp(new_log_prob - old_log_prob)
    # Clipped surrogate
    clipped = clamp(ratio, 1-agent.clip_eps, 1+agent.clip_eps)
    loss_actor = -min(ratio * advantage, clipped * advantage)
    # Gradient of clipped loss (simplified)
    if ratio < clipped
        dL = copy(probs); dL[ai] -= 1.0; dL .*= -advantage / (probs[ai]+1e-10)
        backward_pass!(agent.actor, cache_a, dL, agent.lr_actor)
    end
    # Critic update
    v, cache_c = forward(agent.critic, state)
    backward_pass!(agent.critic, cache_c, Float64(-2*advantage), agent.lr_critic)
    loss_actor
end

# ─────────────────────────────────────────────────────────────
# 14. MULTI-ASSET RL ENVIRONMENT
# ─────────────────────────────────────────────────────────────

"""
    MultiAssetEnv

RL environment for trading a portfolio of N assets.
Actions: weight vector allocation (discretized to N^2 grid or continuous).
"""
mutable struct MultiAssetEnv
    prices::Matrix{Float64}  # T × N
    n_assets::Int; n::Int; t::Int
    weights::Vector{Float64}
    portfolio::Float64; peak::Float64
    tc::Float64; window::Int; done::Bool
    portfolio_history::Vector{Float64}
end

function MultiAssetEnv(prices::Matrix{Float64}; tc=0.001, window=20)
    n, k = size(prices)
    MultiAssetEnv(prices, k, n, window+1,
                  fill(1.0/k, k), 10_000.0, 10_000.0, tc, window, false, [10_000.0])
end

function reset!(env::MultiAssetEnv)::Vector{Float64}
    env.t = env.window + 1; env.portfolio = 10_000.0; env.peak = 10_000.0
    env.weights = fill(1.0/env.n_assets, env.n_assets); env.done = false
    empty!(env.portfolio_history); push!(env.portfolio_history, 10_000.0)
    _compute_multi_state(env)
end

function _compute_multi_state(env::MultiAssetEnv)::Vector{Float64}
    t = env.t; w = env.window
    state = Float64[]
    for i in 1:env.n_assets
        seg = [t > s ? log(env.prices[t, i]/env.prices[t-s, i]) : 0.0
               for s in [1,5,20]]
        append!(state, seg)
        push!(state, env.weights[i])
    end
    push!(state, -(env.peak - env.portfolio) / (env.peak + 1e-10))
    clamp.(state, -3.0, 3.0)
end

function env_step!(env::MultiAssetEnv, action_weights::Vector{Float64})
    env.done && return _compute_multi_state(env), 0.0, true
    new_w = max.(action_weights, 0.0)
    s = sum(new_w); s > 0 && (new_w ./= s)
    trade_cost = sum(abs.(new_w .- env.weights)) * env.tc
    env.weights = new_w
    env.t += 1
    env.t >= env.n && (env.done = true)
    rets = env.done ? zeros(env.n_assets) :
           [log(env.prices[env.t,i]/env.prices[env.t-1,i]) for i in 1:env.n_assets]
    port_ret = dot(env.weights, rets) - trade_cost
    env.portfolio = max(env.portfolio * exp(port_ret), 1.0)
    env.peak = max(env.peak, env.portfolio)
    push!(env.portfolio_history, env.portfolio)
    reward = port_ret - 0.3*(env.peak - env.portfolio)/(env.peak+1e-10)
    _compute_multi_state(env), reward, env.done
end

# ─────────────────────────────────────────────────────────────
# 15. REWARD SHAPING
# ─────────────────────────────────────────────────────────────

"""
    SharpeReward

Reward function that approximates Sharpe ratio increment.
Penalizes volatility as well as rewarding returns.
"""
mutable struct SharpeReward
    window::Int
    returns_buffer::Vector{Float64}
end

SharpeReward(window::Int=20) = SharpeReward(window, Float64[])

function compute_reward(sr::SharpeReward, portfolio_return::Float64)::Float64
    push!(sr.returns_buffer, portfolio_return)
    length(sr.returns_buffer) > sr.window && popfirst!(sr.returns_buffer)
    length(sr.returns_buffer) < 3 && return portfolio_return
    mu = mean(sr.returns_buffer); sg = std(sr.returns_buffer)
    sg < 1e-8 && return mu
    mu / sg  # Sharpe-like reward
end

"""
    CalmarReward

Reward proportional to Calmar ratio (return / max drawdown).
"""
mutable struct CalmarReward
    returns::Vector{Float64}
    portfolio_vals::Vector{Float64}
end

CalmarReward() = CalmarReward(Float64[], [10_000.0])

function compute_reward(cr::CalmarReward, ret::Float64, portfolio::Float64)::Float64
    push!(cr.returns, ret); push!(cr.portfolio_vals, portfolio)
    peak = maximum(cr.portfolio_vals)
    mdd  = (peak - portfolio) / (peak + 1e-10)
    mdd < 1e-6 && return ret
    ret / (mdd + 1e-4)
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 15 – Evaluation Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_sortino(returns, target_return)

Sortino ratio: excess return divided by downside deviation.
Only penalises downside volatility below `target_return`.
"""
function compute_sortino(returns::Vector{Float64},
                          target_return::Float64=0.0)
    excess = mean(returns) - target_return
    downside = sqrt(mean(min.(returns .- target_return, 0.0).^2))
    return downside < 1e-8 ? 0.0 : excess / downside
end

"""
    compute_omega_ratio(returns, threshold)

Omega ratio: probability-weighted gains above threshold divided by
probability-weighted losses below threshold.
"""
function compute_omega_ratio(returns::Vector{Float64},
                              threshold::Float64=0.0)
    gains  = sum(max.(returns .- threshold, 0.0))
    losses = sum(max.(threshold .- returns, 0.0))
    return losses < 1e-8 ? Inf : gains / losses
end

"""
    rolling_sharpe(returns, window, rf)

Rolling Sharpe ratio computed over a sliding `window` of returns.
"""
function rolling_sharpe(returns::Vector{Float64},
                          window::Int=252, rf::Float64=0.0)
    n = length(returns)
    rs = fill(NaN, n)
    for i in (window+1):n
        r = returns[i-window+1:i]
        mu = mean(r) - rf / window
        sg = std(r)
        rs[i] = sg < 1e-8 ? 0.0 : mu / sg * sqrt(window)
    end
    return rs
end

end  # module ReinforcementLearningTrading
