"""
RL Trainer: manages the full training loop for all RL agents.

Features:
- Vectorized environment support
- Curriculum learning (easy → hard markets)
- Regime-conditioned training
- Checkpoint save/load
- TensorBoard logging
- Multi-agent support (PPO, SAC, DQN, Transformer)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

from .environment import (
    TradingEnv, VecTradingEnv, RegimeTradingEnv,
    TradingConfig, Instrument, make_vec_env, generate_synthetic_data,
)
from .models.ppo import PPOAgent, PPOConfig, RolloutBuffer
from .models.sac import SACAgent, SACConfig
from .models.dqn import DQNAgent, DQNConfig, ActionDiscretizer


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # General
    agent_type: str = "ppo"          # "ppo" | "sac" | "dqn" | "transformer"
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    eval_freq: int = 10_000          # steps between evaluations
    eval_episodes: int = 5
    save_freq: int = 50_000
    log_dir: str = "./logs/rl"
    checkpoint_dir: str = "./checkpoints/rl"
    seed: int = 42

    # Curriculum
    use_curriculum: bool = True
    curriculum_thresholds: List[float] = field(default_factory=lambda: [0.5, 1.5])
    # Level up when rolling mean Sharpe > threshold[level]
    curriculum_window: int = 100

    # Regime training
    use_regime_training: bool = False
    regime_sampling_weights: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.1, 0.1])
    # Probability of sampling each regime: [bull, bear, sideways, high_vol, low_vol]

    # Logging
    log_interval: int = 100
    verbose: int = 1

    # Early stopping
    use_early_stopping: bool = True
    patience: int = 50              # evaluations without improvement
    min_improvement: float = 0.05

    # Normalization
    normalize_obs: bool = True
    normalize_rewards: bool = True
    reward_clip: float = 10.0


# ---------------------------------------------------------------------------
# Running statistics for online normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Online estimate of mean and variance (Welford's algorithm)."""

    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch = np.asarray(x, dtype=np.float64)
        if batch.ndim == len(self.mean.shape):
            batch = batch[np.newaxis]
        batch_mean = batch.mean(axis=0)
        batch_var  = batch.var(axis=0)
        batch_count = batch.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2  = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean  = new_mean
        self.var   = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        normed = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normed, -clip, clip).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * np.sqrt(self.var) + self.mean


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent: Any,
    env_fn: Callable,
    n_episodes: int = 5,
    deterministic: bool = True,
    obs_normalizer: Optional[RunningMeanStd] = None,
    agent_type: str = "ppo",
) -> Dict[str, float]:
    """Run n_episodes and return aggregated metrics."""
    env = env_fn()
    episode_rewards = []
    episode_sharpes = []
    episode_drawdowns = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if obs_normalizer:
            obs = obs_normalizer.normalize(obs)

        ep_reward = 0.0
        ep_len = 0
        done = False

        if agent_type == "ppo":
            agent.reset_lstm()
        elif agent_type == "transformer":
            agent.reset_sequence()

        while not done:
            if agent_type == "ppo":
                action, _, _ = agent.collect_action(obs)
            elif agent_type == "sac":
                action = agent.select_action(obs, deterministic=True)
            elif agent_type == "dqn":
                action_idx = agent.select_action(obs, deterministic=True)
                action = np.array([action_idx], dtype=np.float32)  # env handles decoding
            elif agent_type == "transformer":
                action, _, _ = agent.collect_action(obs, deterministic=True)
            else:
                action = env.action_space.sample() if hasattr(env, "action_space") else np.zeros(env.act_dim)

            obs, reward, terminated, truncated, info = env.step(action)
            if obs_normalizer:
                obs = obs_normalizer.normalize(obs)

            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

        stats = env.get_episode_stats()
        episode_sharpes.append(stats.get("sharpe", 0.0))
        episode_drawdowns.append(stats.get("max_drawdown", 0.0))

    env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_sharpe": float(np.mean(episode_sharpes)),
        "mean_drawdown": float(np.mean(episode_drawdowns)),
        "mean_length": float(np.mean(episode_lengths)),
    }


# ---------------------------------------------------------------------------
# Curriculum manager
# ---------------------------------------------------------------------------

class CurriculumManager:
    """Manages curriculum progression based on rolling evaluation metrics."""

    def __init__(
        self,
        thresholds: List[float],
        window: int = 100,
        metric: str = "mean_sharpe",
    ):
        self.thresholds = thresholds
        self.window = window
        self.metric = metric
        self._level = 0
        self._metric_history: List[float] = []

    @property
    def level(self) -> int:
        return self._level

    def update(self, metrics: Dict[str, float]) -> bool:
        """Update curriculum level. Returns True if level changed."""
        val = metrics.get(self.metric, 0.0)
        self._metric_history.append(val)

        if len(self._metric_history) >= self.window:
            rolling_mean = float(np.mean(self._metric_history[-self.window:]))
            if self._level < len(self.thresholds) and rolling_mean > self.thresholds[self._level]:
                self._level += 1
                return True
        return False

    def should_level_up(self, rolling_mean: float) -> bool:
        if self._level >= len(self.thresholds):
            return False
        return rolling_mean > self.thresholds[self._level]


# ---------------------------------------------------------------------------
# PPO Training loop
# ---------------------------------------------------------------------------

def train_ppo(
    data: Dict,
    config: TrainConfig,
    ppo_config: PPOConfig,
    env_config: Optional[TradingConfig] = None,
) -> PPOAgent:
    """Full PPO training loop with curriculum and logging."""

    rng = np.random.default_rng(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    env_config = env_config or TradingConfig(curriculum_level=0)

    # Create vectorized envs
    vec_env = make_vec_env(
        data, n_envs=config.n_envs,
        config=env_config,
        regime_aware=config.use_regime_training,
        seeds=list(range(config.n_envs)),
    )

    # Infer dims
    obs_dim = vec_env.obs_dim
    act_dim = vec_env.act_dim
    ppo_config.obs_dim = obs_dim
    ppo_config.act_dim = act_dim

    agent = PPOAgent(ppo_config)
    agent.reset_lstm(config.n_envs)

    # Normalizers
    obs_norm  = RunningMeanStd(shape=(obs_dim,)) if config.normalize_obs else None
    rew_norm  = RunningMeanStd() if config.normalize_rewards else None

    # TensorBoard
    writer = None
    if TB_AVAILABLE:
        writer = SummaryWriter(log_dir=config.log_dir)

    # Curriculum
    curriculum = CurriculumManager(config.curriculum_thresholds, config.curriculum_window) \
        if config.use_curriculum else None

    # Rollout storage (per env)
    rollout_data: Dict[str, List] = {
        "obs": [], "actions": [], "rewards": [], "dones": [],
        "values": [], "log_probs": [],
    }

    obs = vec_env.reset()
    if obs_norm:
        obs_norm.update(obs)
        obs = obs_norm.normalize(obs)

    total_steps = 0
    n_updates = 0
    episode_infos: List[Dict] = []
    eval_results: List[Dict] = []
    best_sharpe = -float("inf")
    no_improve_count = 0

    t_start = time.time()
    print(f"Starting PPO training: {config.total_timesteps:,} steps, {config.n_envs} envs")

    # Build per-env rollout buffers
    per_env_buffers = [
        RolloutBuffer(
            buffer_size=ppo_config.rollout_steps // config.n_envs,
            obs_dim=obs_dim,
            act_dim=act_dim,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            device=agent.device,
        )
        for _ in range(config.n_envs)
    ]

    steps_per_collection = ppo_config.rollout_steps // config.n_envs

    while total_steps < config.total_timesteps:
        # Collect rollout
        for _ in range(steps_per_collection):
            # Get action per env
            all_actions = np.zeros((config.n_envs, act_dim), dtype=np.float32)
            all_log_probs = np.zeros(config.n_envs, dtype=np.float32)
            all_values = np.zeros(config.n_envs, dtype=np.float32)

            for env_i in range(config.n_envs):
                obs_i = obs[env_i]
                action_i, lp_i, val_i = agent.collect_action(obs_i)
                all_actions[env_i]   = action_i
                all_log_probs[env_i] = lp_i
                all_values[env_i]    = val_i

            # Step envs
            next_obs, rewards, dones, infos = vec_env.step(all_actions)

            if obs_norm:
                obs_norm.update(next_obs)
                next_obs_normed = obs_norm.normalize(next_obs)
            else:
                next_obs_normed = next_obs

            # Process rewards
            if rew_norm:
                rew_norm.update(rewards)
                rewards_normed = np.clip(rewards / (np.sqrt(rew_norm.var) + 1e-8), -config.reward_clip, config.reward_clip)
            else:
                rewards_normed = rewards

            # Store in per-env buffers
            for env_i in range(config.n_envs):
                per_env_buffers[env_i].add(
                    obs[env_i], all_actions[env_i],
                    float(rewards_normed[env_i]), bool(dones[env_i]),
                    float(all_values[env_i]), float(all_log_probs[env_i]),
                )

            # Collect episode info
            for info in infos:
                if info:
                    episode_infos.append(info)

            obs = next_obs_normed
            total_steps += config.n_envs

        # Compute GAE and update
        for env_i in range(config.n_envs):
            last_val_i, _, _ = agent.collect_action(obs[env_i])
            last_val_scalar  = float(agent.network.get_value(
                __import__("torch").FloatTensor(obs[env_i]).unsqueeze(0).to(agent.device)
            ).item())
            per_env_buffers[env_i].compute_gae(last_val_scalar, False)

        # Merge buffers and update agent
        merged_buf = _merge_rollout_buffers(per_env_buffers, ppo_config)
        agent.buffer = merged_buf
        update_metrics = agent.update(obs[0], False)
        n_updates += 1

        # Reset per-env buffers
        for b in per_env_buffers:
            b.reset()

        # Logging
        if total_steps % config.log_interval == 0 and episode_infos:
            recent = episode_infos[-100:]
            mean_reward = float(np.mean([i.get("total_reward", 0) for i in recent]))
            mean_pnl    = float(np.mean([i.get("total_pnl", 0) for i in recent]))

            if config.verbose >= 1:
                elapsed = time.time() - t_start
                fps = total_steps / max(elapsed, 1)
                print(
                    f"Steps: {total_steps:8,} | FPS: {fps:6.0f} | "
                    f"Reward: {mean_reward:+.3f} | PnL: {mean_pnl:+,.0f} | "
                    f"Updates: {n_updates} | Curriculum: {curriculum.level if curriculum else 0}"
                )

            if writer:
                writer.add_scalar("train/mean_reward", mean_reward, total_steps)
                writer.add_scalar("train/mean_pnl",    mean_pnl,    total_steps)
                for k, v in update_metrics.items():
                    writer.add_scalar(f"ppo/{k}", v, total_steps)

        # Periodic evaluation
        if total_steps % config.eval_freq == 0:
            eval_env_fn = lambda: make_vec_env(
                data, n_envs=1, config=env_config
            ).envs[0]

            eval_metrics = evaluate_agent(
                agent, eval_env_fn, config.eval_episodes,
                obs_normalizer=obs_norm, agent_type="ppo"
            )
            eval_results.append({"step": total_steps, **eval_metrics})

            if config.verbose >= 1:
                print(f"  [EVAL] Sharpe: {eval_metrics['mean_sharpe']:+.3f} | "
                      f"Drawdown: {eval_metrics['mean_drawdown']:.2%} | "
                      f"Reward: {eval_metrics['mean_reward']:+.3f}")

            if writer:
                for k, v in eval_metrics.items():
                    writer.add_scalar(f"eval/{k}", v, total_steps)

            # Curriculum update
            if curriculum:
                changed = curriculum.update(eval_metrics)
                if changed:
                    env_config.curriculum_level = curriculum.level
                    print(f"  [CURRICULUM] Leveled up to {curriculum.level}")
                    vec_env = make_vec_env(
                        data, n_envs=config.n_envs, config=env_config,
                        regime_aware=config.use_regime_training,
                    )
                    obs = vec_env.reset()
                    if obs_norm:
                        obs = obs_norm.normalize(obs)

            # Save best
            sharpe = eval_metrics["mean_sharpe"]
            if sharpe > best_sharpe + config.min_improvement:
                best_sharpe = sharpe
                no_improve_count = 0
                best_path = os.path.join(config.checkpoint_dir, "best_ppo.pt")
                agent.save(best_path)
            else:
                no_improve_count += 1

            # Early stopping
            if config.use_early_stopping and no_improve_count >= config.patience:
                print(f"Early stopping after {no_improve_count} evals without improvement.")
                break

        # Periodic checkpoint
        if total_steps % config.save_freq == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"ppo_step_{total_steps}.pt")
            agent.save(ckpt_path)

    # Final save
    final_path = os.path.join(config.checkpoint_dir, "ppo_final.pt")
    agent.save(final_path)

    if writer:
        writer.close()

    vec_env.close()
    print(f"Training complete. Best Sharpe: {best_sharpe:.3f}")
    _save_training_log(eval_results, os.path.join(config.log_dir, "eval_log.json"))

    return agent


def _merge_rollout_buffers(buffers: List[RolloutBuffer], config: PPOConfig) -> RolloutBuffer:
    """Merge multiple per-env rollout buffers into one."""
    import torch
    merged = RolloutBuffer(
        buffer_size=sum(b.buffer_size for b in buffers),
        obs_dim=buffers[0].obs_dim,
        act_dim=buffers[0].act_dim,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        device=buffers[0].device,
    )
    ptr = 0
    total = merged.buffer_size
    for b in buffers:
        sz = b.buffer_size
        merged.obs[ptr:ptr+sz]        = b.obs
        merged.actions[ptr:ptr+sz]    = b.actions
        merged.rewards[ptr:ptr+sz]    = b.rewards
        merged.dones[ptr:ptr+sz]      = b.dones
        merged.values[ptr:ptr+sz]     = b.values
        merged.log_probs[ptr:ptr+sz]  = b.log_probs
        merged.returns[ptr:ptr+sz]    = b.returns
        merged.advantages[ptr:ptr+sz] = b.advantages
        ptr += sz
    merged._ptr = total
    merged._full = True
    return merged


# ---------------------------------------------------------------------------
# SAC Training loop
# ---------------------------------------------------------------------------

def train_sac(
    data: Dict,
    config: TrainConfig,
    sac_config: SACConfig,
    env_config: Optional[TradingConfig] = None,
) -> SACAgent:
    """Full SAC training loop."""

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    env_config = env_config or TradingConfig()

    from .environment import make_trading_env
    env = make_trading_env(data, config=env_config)
    eval_env_fn = lambda: make_trading_env(data, config=env_config)

    sac_config.obs_dim = env.obs_dim
    sac_config.act_dim = env.act_dim
    agent = SACAgent(sac_config)

    obs_norm = RunningMeanStd(shape=(env.obs_dim,)) if config.normalize_obs else None

    writer = None
    if TB_AVAILABLE:
        writer = SummaryWriter(log_dir=config.log_dir + "_sac")

    obs, _ = env.reset()
    if obs_norm:
        obs_norm.update(obs)
        obs = obs_norm.normalize(obs)

    total_steps = 0
    ep_rewards: List[float] = []
    ep_reward = 0.0
    best_sharpe = -float("inf")
    no_improve_count = 0
    eval_results = []

    t_start = time.time()
    print(f"Starting SAC training: {config.total_timesteps:,} steps")

    while total_steps < config.total_timesteps:
        if total_steps < sac_config.warmup_steps:
            action = np.random.uniform(-1, 1, env.act_dim).astype(np.float32)
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if obs_norm:
            obs_norm.update(next_obs)
            next_obs_n = obs_norm.normalize(next_obs)
        else:
            next_obs_n = next_obs

        agent.observe(obs, action, reward, next_obs_n, done)
        obs = next_obs_n
        ep_reward += reward
        total_steps += 1

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
            if obs_norm:
                obs_norm.update(obs)
                obs = obs_norm.normalize(obs)

        if agent.should_update():
            for _ in range(sac_config.updates_per_step):
                m = agent.update()

        if total_steps % config.log_interval == 0 and ep_rewards:
            mean_ep = float(np.mean(ep_rewards[-20:]))
            fps = total_steps / max(time.time() - t_start, 1)
            if config.verbose >= 1:
                print(f"Steps: {total_steps:8,} | FPS: {fps:5.0f} | "
                      f"Ep Reward: {mean_ep:+.3f} | Alpha: {agent.alpha.item():.4f}")
            if writer:
                writer.add_scalar("train/ep_reward", mean_ep, total_steps)
                writer.add_scalar("train/alpha", agent.alpha.item(), total_steps)

        if total_steps % config.eval_freq == 0:
            eval_metrics = evaluate_agent(agent, eval_env_fn, config.eval_episodes,
                                          obs_normalizer=obs_norm, agent_type="sac")
            eval_results.append({"step": total_steps, **eval_metrics})

            if config.verbose >= 1:
                print(f"  [EVAL] Sharpe: {eval_metrics['mean_sharpe']:+.3f}")

            sharpe = eval_metrics["mean_sharpe"]
            if sharpe > best_sharpe + config.min_improvement:
                best_sharpe = sharpe
                no_improve_count = 0
                agent.save(os.path.join(config.checkpoint_dir, "best_sac.pt"))
            else:
                no_improve_count += 1

            if config.use_early_stopping and no_improve_count >= config.patience:
                print("Early stopping.")
                break

        if total_steps % config.save_freq == 0:
            agent.save(os.path.join(config.checkpoint_dir, f"sac_step_{total_steps}.pt"))

    agent.save(os.path.join(config.checkpoint_dir, "sac_final.pt"))
    if writer:
        writer.close()
    env.close()
    _save_training_log(eval_results, os.path.join(config.log_dir, "sac_eval_log.json"))
    print(f"SAC training complete. Best Sharpe: {best_sharpe:.3f}")
    return agent


# ---------------------------------------------------------------------------
# DQN Training loop
# ---------------------------------------------------------------------------

def train_dqn(
    data: Dict,
    config: TrainConfig,
    dqn_config: DQNConfig,
    env_config: Optional[TradingConfig] = None,
    discretizer: Optional[ActionDiscretizer] = None,
) -> DQNAgent:
    """Full DQN training loop."""

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    env_config = env_config or TradingConfig()

    from .environment import make_trading_env
    env = make_trading_env(data, config=env_config)

    n_assets = env.n_instruments
    if discretizer is None:
        levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
        discretizer = ActionDiscretizer(n_assets, levels, mode="independent")

    dqn_config.obs_dim = env.obs_dim
    dqn_config.n_actions = discretizer.n_actions
    agent = DQNAgent(dqn_config)

    obs_norm = RunningMeanStd(shape=(env.obs_dim,)) if config.normalize_obs else None
    writer = None
    if TB_AVAILABLE:
        writer = SummaryWriter(log_dir=config.log_dir + "_dqn")

    obs, _ = env.reset()
    if obs_norm:
        obs_norm.update(obs)
        obs = obs_norm.normalize(obs)

    total_steps = 0
    ep_rewards = []
    ep_reward = 0.0
    best_sharpe = -float("inf")
    no_improve_count = 0
    eval_results = []

    t_start = time.time()
    print(f"Starting DQN training: {config.total_timesteps:,} steps")

    while total_steps < config.total_timesteps:
        action_idx = agent.select_action(obs)
        action = discretizer.decode(action_idx)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if obs_norm:
            obs_norm.update(next_obs)
            next_obs_n = obs_norm.normalize(next_obs)
        else:
            next_obs_n = next_obs

        agent.observe(obs, action_idx, reward, next_obs_n, done)
        obs = next_obs_n
        ep_reward += reward
        total_steps += 1

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
            if obs_norm:
                obs_norm.update(obs)
                obs = obs_norm.normalize(obs)

        if agent.should_update():
            m = agent.update()

        if total_steps % config.log_interval == 0 and ep_rewards:
            mean_ep = float(np.mean(ep_rewards[-20:]))
            fps = total_steps / max(time.time() - t_start, 1)
            if config.verbose >= 1:
                print(f"Steps: {total_steps:8,} | FPS: {fps:5.0f} | "
                      f"Ep Reward: {mean_ep:+.3f} | Eps: {agent._epsilon:.3f}")

        if total_steps % config.eval_freq == 0:
            eval_env_fn = lambda: make_trading_env(data, config=env_config)
            # For DQN eval, wrap select_action to also decode
            class DQNEvalWrapper:
                def __init__(self, ag, disc):
                    self.ag = ag
                    self.disc = disc
                def select_action(self, obs, deterministic=True):
                    idx = self.ag.select_action(obs, deterministic=deterministic)
                    return self.disc.decode(idx)

            wrapped = DQNEvalWrapper(agent, discretizer)
            eval_metrics = evaluate_agent(
                wrapped, eval_env_fn, config.eval_episodes,
                obs_normalizer=obs_norm, agent_type="sac"  # reuse sac path
            )
            eval_results.append({"step": total_steps, **eval_metrics})

            if config.verbose >= 1:
                print(f"  [EVAL] Sharpe: {eval_metrics['mean_sharpe']:+.3f}")

            sharpe = eval_metrics["mean_sharpe"]
            if sharpe > best_sharpe + config.min_improvement:
                best_sharpe = sharpe
                no_improve_count = 0
                agent.save(os.path.join(config.checkpoint_dir, "best_dqn.pt"))
            else:
                no_improve_count += 1

            if config.use_early_stopping and no_improve_count >= config.patience:
                print("Early stopping.")
                break

        if total_steps % config.save_freq == 0:
            agent.save(os.path.join(config.checkpoint_dir, f"dqn_step_{total_steps}.pt"))

    agent.save(os.path.join(config.checkpoint_dir, "dqn_final.pt"))
    if writer:
        writer.close()
    env.close()
    _save_training_log(eval_results, os.path.join(config.log_dir, "dqn_eval_log.json"))
    print(f"DQN training complete. Best Sharpe: {best_sharpe:.3f}")
    return agent


# ---------------------------------------------------------------------------
# Regime-conditioned training
# ---------------------------------------------------------------------------

class RegimeConditionedTrainer:
    """
    Trains agent on specific market regimes, cycling through them.
    Used to ensure the agent learns diverse market conditions.
    """

    REGIMES = ["bull", "bear", "sideways", "high_vol", "low_vol"]

    def __init__(
        self,
        agent: Any,
        data: Dict,
        env_config: TradingConfig,
        config: TrainConfig,
        agent_type: str = "ppo",
    ):
        self.agent = agent
        self.data  = data
        self.env_config = env_config
        self.config = config
        self.agent_type = agent_type
        self._regime_counts = {r: 0 for r in self.REGIMES}
        self._regime_rewards: Dict[str, List[float]] = {r: [] for r in self.REGIMES}

    def train_on_regime(self, regime_idx: int, n_steps: int) -> Dict[str, float]:
        """Train agent on a specific regime for n_steps."""
        from .environment import RegimeTradingEnv
        env = RegimeTradingEnv(self.data, config=self.env_config)
        obs, info = env.reset(regime=regime_idx)

        regime_name = self.REGIMES[regime_idx]
        total_reward = 0.0
        steps = 0

        if self.agent_type == "ppo":
            self.agent.reset_lstm()

        while steps < n_steps:
            if self.agent_type == "ppo":
                action, lp, val = self.agent.collect_action(obs)
            elif self.agent_type == "sac":
                action = self.agent.select_action(obs)
            else:
                action = np.zeros(env.act_dim)

            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                obs, info = env.reset(regime=regime_idx)
                if self.agent_type == "ppo":
                    self.agent.reset_lstm()

        self._regime_counts[regime_name] += steps
        self._regime_rewards[regime_name].append(total_reward / max(n_steps, 1))
        env.close()

        return {
            "regime": regime_name,
            "steps": steps,
            "mean_reward": float(total_reward / max(n_steps, 1)),
        }

    def cycle_train(self, total_steps: int, steps_per_regime: int = 1000) -> None:
        """Cycle through all regimes during training."""
        steps_done = 0
        while steps_done < total_steps:
            for regime_idx in range(len(self.REGIMES)):
                result = self.train_on_regime(regime_idx, steps_per_regime)
                steps_done += steps_per_regime
                if steps_done >= total_steps:
                    break

        print("Regime cycle training complete.")
        for r, rewards in self._regime_rewards.items():
            if rewards:
                print(f"  {r}: mean reward = {np.mean(rewards):.4f}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_training_log(data: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_agent(path: str, agent_type: str, config: Any) -> Any:
    """Load a saved agent from checkpoint."""
    if agent_type == "ppo":
        agent = PPOAgent(config)
        agent.load(path)
    elif agent_type == "sac":
        agent = SACAgent(config)
        agent.load(path)
    elif agent_type == "dqn":
        agent = DQNAgent(config)
        agent.load(path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing trainer...")

    data = generate_synthetic_data(n_assets=2, n_days=500)

    env_config = TradingConfig(
        max_episode_steps=50,
        window_size=20,
        reward_type="sharpe",
    )

    ppo_cfg = PPOConfig(
        lstm_hidden=64,
        lstm_layers=1,
        mlp_hidden=[128, 64],
        rollout_steps=64,
        n_epochs=2,
        minibatch_size=16,
        device="cpu",
    )

    train_cfg = TrainConfig(
        total_timesteps=500,
        n_envs=2,
        eval_freq=200,
        eval_episodes=2,
        save_freq=500,
        log_dir="/tmp/rl_test_logs",
        checkpoint_dir="/tmp/rl_test_ckpts",
        use_curriculum=False,
        use_early_stopping=False,
        normalize_obs=False,
        normalize_rewards=False,
        verbose=1,
    )

    agent = train_ppo(data, train_cfg, ppo_cfg, env_config)
    print(f"Trainer self-test passed. Agent train step: {agent.train_step}")
