"""
research/agent_training/trainer.py

Agent training orchestration with curriculum learning, early stopping,
checkpoint management, and training curve plotting.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from research.agent_training.environment import (
    TradingEnvironment,
    EnvironmentConfig,
    EpisodeStats,
    episode_stats,
    difficulty_schedule,
)
from research.agent_training.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    EpisodeBuffer,
    Batch,
)
from research.agent_training.agents import (
    DQNAgent,
    DDQNAgent,
    D3QNAgent,
    TD3Agent,
    PPOAgent,
    EnsembleAgent,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """All hyperparameters for AgentTrainer."""

    n_episodes: int = 1000
    eval_every: int = 50
    eval_episodes: int = 20
    batch_size: int = 64
    buffer_capacity: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005
    target_update_freq: int = 200         # hard update every N steps (DQN-style)
    use_soft_update: bool = True          # polyak instead of hard update
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    grad_clip: float = 10.0
    use_per: bool = False                 # Prioritized Experience Replay
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    max_steps_per_episode: int = 2000
    warmup_steps: int = 1000             # steps before training starts
    checkpoint_every: int = 100          # save checkpoint every N episodes
    checkpoint_dir: str = "checkpoints"
    early_stop_patience: int = 50        # episodes without OOS improvement
    early_stop_min_delta: float = 0.001
    use_curriculum: bool = True
    curriculum_start_difficulty: float = 0.2
    curriculum_end_difficulty: float = 1.0
    use_ppo: bool = False                # switch to on-policy PPO mode
    ppo_rollout_steps: int = 2048
    ppo_epochs: int = 10
    ppo_batch_size: int = 64
    gae_lambda: float = 0.95
    seed: Optional[int] = None
    verbose: bool = True


@dataclass
class EvalResult:
    """Result of evaluating an agent for N episodes."""

    mean_return: float
    std_return: float
    mean_sharpe: float
    mean_max_drawdown: float
    mean_n_trades: float
    episode_returns: list[float]
    episode_sharpes: list[float]
    episode_drawdowns: list[float]


@dataclass
class TrainingResult:
    """Aggregated result of a full training run."""

    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    eval_returns: list[float] = field(default_factory=list)
    eval_episodes: list[int] = field(default_factory=list)
    eval_results: list[EvalResult] = field(default_factory=list)
    best_eval_return: float = -math.inf
    best_episode: int = 0
    best_weights_path: str = ""
    total_steps: int = 0
    training_time_s: float = 0.0
    early_stopped: bool = False
    final_epsilon: float = 0.0
    config: Optional[TrainingConfig] = None


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------


def curriculum_schedule(
    episode: int,
    max_episodes: int,
    start_difficulty: float = 0.2,
    end_difficulty: float = 1.0,
) -> float:
    """
    Compute difficulty in [start_difficulty, end_difficulty].

    Uses square-root ramp so agents spend more time in easier settings early.
    """
    t = float(episode) / max(1, max_episodes)
    diff = start_difficulty + (end_difficulty - start_difficulty) * (t ** 0.5)
    return float(np.clip(diff, start_difficulty, end_difficulty))


def _apply_difficulty(env: TradingEnvironment, difficulty: float) -> None:
    """
    Adjust environment config based on difficulty.

    At low difficulty:
      - Reduced transaction cost
      - Shorter episodes (easier to get positive reward)
      - No drawdown penalty
    """
    cfg = env.cfg
    cfg.transaction_cost = 0.0001 + difficulty * 0.0002
    cfg.max_steps = max(200, int(difficulty * cfg.max_steps))
    cfg.penalty_drawdown = difficulty * 0.15
    cfg.slippage_bps = difficulty * 1.0


# ---------------------------------------------------------------------------
# AgentTrainer
# ---------------------------------------------------------------------------


class AgentTrainer:
    """
    Orchestrates agent training against a TradingEnvironment.

    Supports:
    - Off-policy agents (DQN, DDQN, D3QN, TD3, Ensemble) with replay buffer
    - On-policy PPO with episodic rollout buffer
    - Curriculum learning with difficulty scheduling
    - Early stopping on OOS eval performance
    - Checkpoint saving every N episodes
    - Training curve plotting (requires matplotlib)

    Args:
        agent  : Any trained agent (DQNAgent, DDQNAgent, D3QNAgent,
                 TD3Agent, PPOAgent, EnsembleAgent).
        env    : TradingEnvironment for training.
        config : TrainingConfig.
        eval_env : Optional separate environment for evaluation.
                   If None, training env is reused with a fixed start.
    """

    def __init__(
        self,
        agent: Any,
        env: TradingEnvironment,
        config: TrainingConfig,
        eval_env: Optional[TradingEnvironment] = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.cfg = config
        self.eval_env = eval_env if eval_env is not None else env
        self._is_ppo = isinstance(agent, PPOAgent)

        if config.seed is not None:
            np.random.seed(config.seed)
            env.seed(config.seed)

        # Build replay buffer
        if not self._is_ppo:
            obs_dim = env.obs_dim
            action_dim = 1
            if config.use_per:
                self._buffer: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(
                    config.buffer_capacity, obs_dim, action_dim,
                    alpha=config.per_alpha, beta_start=config.per_beta_start,
                )
            else:
                self._buffer = ReplayBuffer(config.buffer_capacity, obs_dim, action_dim)
        else:
            self._ep_buffer = EpisodeBuffer(
                env.obs_dim, 1,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                max_size=config.ppo_rollout_steps,
            )

        # Checkpoint dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self._global_step = 0
        self._result = TrainingResult(config=config)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        n_episodes: Optional[int] = None,
        eval_every: Optional[int] = None,
    ) -> TrainingResult:
        """
        Train the agent for n_episodes episodes.

        Args:
            n_episodes : Override config.n_episodes.
            eval_every : Override config.eval_every.

        Returns:
            TrainingResult with all metrics.
        """
        cfg = self.cfg
        n_eps = n_episodes or cfg.n_episodes
        eval_freq = eval_every or cfg.eval_every

        start_time = time.time()
        best_eval_return = -math.inf
        patience_counter = 0

        for episode in range(n_eps):
            # Curriculum difficulty
            if cfg.use_curriculum:
                diff = curriculum_schedule(
                    episode, n_eps,
                    cfg.curriculum_start_difficulty,
                    cfg.curriculum_end_difficulty,
                )
                _apply_difficulty(self.env, diff)
            else:
                diff = 1.0

            # Run one episode
            if self._is_ppo:
                ep_return, ep_len, ep_loss = self._ppo_episode()
            else:
                ep_return, ep_len, ep_loss = self._off_policy_episode()

            self._result.episode_returns.append(ep_return)
            self._result.episode_lengths.append(ep_len)
            if ep_loss is not None:
                self._result.losses.append(ep_loss)

            # Periodic evaluation
            if (episode + 1) % eval_freq == 0:
                eval_res = self.evaluate(cfg.eval_episodes)
                self._result.eval_results.append(eval_res)
                self._result.eval_returns.append(eval_res.mean_return)
                self._result.eval_episodes.append(episode + 1)

                if cfg.verbose:
                    self._log(episode, ep_return, ep_loss, eval_res, diff)

                # Early stopping
                if eval_res.mean_return > best_eval_return + cfg.early_stop_min_delta:
                    best_eval_return = eval_res.mean_return
                    patience_counter = 0
                    self._result.best_eval_return = best_eval_return
                    self._result.best_episode = episode + 1
                    # Save best weights
                    path = os.path.join(cfg.checkpoint_dir, "best_agent")
                    self._save_agent(path)
                    self._result.best_weights_path = path
                else:
                    patience_counter += 1

                if patience_counter >= cfg.early_stop_patience:
                    if cfg.verbose:
                        print(f"Early stopping at episode {episode + 1}.")
                    self._result.early_stopped = True
                    break
            elif cfg.verbose and (episode + 1) % max(1, eval_freq // 5) == 0:
                print(
                    f"Ep {episode+1:5d} | ret={ep_return:.4f} "
                    f"| loss={ep_loss:.5f if ep_loss else 0:.5f}"
                )

            # Checkpoint
            if (episode + 1) % cfg.checkpoint_every == 0:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"ckpt_ep{episode+1}")
                self._save_agent(ckpt_path)

        self._result.total_steps = self._global_step
        self._result.training_time_s = time.time() - start_time

        # Final epsilon
        if hasattr(self.agent, "epsilon"):
            self._result.final_epsilon = float(self.agent.epsilon)
        elif hasattr(self.agent, "d3qn"):
            self._result.final_epsilon = float(self.agent.d3qn.epsilon)

        return self._result

    # ------------------------------------------------------------------
    # Off-policy episode
    # ------------------------------------------------------------------

    def _off_policy_episode(self) -> tuple[float, int, Optional[float]]:
        obs = self.env.reset()
        ep_return = 0.0
        ep_len = 0
        total_loss = 0.0
        n_updates = 0
        done = False
        cfg = self.cfg

        while not done:
            # Select action
            if isinstance(self.agent, EnsembleAgent):
                action = self.agent.act_explore(obs)
            elif hasattr(self.agent, "act"):
                action = self.agent.act(obs)
            else:
                action = 0.0

            next_obs, reward, done, info = self.env.step(action)
            self._buffer.push(obs, action, reward, next_obs, done)
            self._global_step += 1
            ep_return += reward
            ep_len += 1
            obs = next_obs

            # Training step
            if (
                self._global_step >= cfg.warmup_steps
                and len(self._buffer) >= cfg.batch_size
            ):
                batch = self._buffer.sample(cfg.batch_size)
                if isinstance(self.agent, EnsembleAgent):
                    losses = self.agent.train_all(batch)
                    loss = float(np.mean(list(losses.values())))
                else:
                    loss = self.agent.train_step(batch)

                # Update PER priorities
                if cfg.use_per and isinstance(self._buffer, PrioritizedReplayBuffer):
                    if hasattr(self.agent, "_last_td_errors"):
                        self._buffer.update_priorities(
                            batch.indices, self.agent._last_td_errors
                        )

                total_loss += loss
                n_updates += 1

            # Target network update
            if cfg.use_soft_update:
                if hasattr(self.agent, "soft_update_target"):
                    self.agent.soft_update_target(cfg.tau)
                elif hasattr(self.agent, "soft_update_targets"):
                    self.agent.soft_update_targets(cfg.tau)
            elif self._global_step % cfg.target_update_freq == 0:
                if hasattr(self.agent, "hard_update_target"):
                    self.agent.hard_update_target()

        avg_loss = total_loss / max(1, n_updates)
        return ep_return, ep_len, avg_loss

    # ------------------------------------------------------------------
    # PPO on-policy episode
    # ------------------------------------------------------------------

    def _ppo_episode(self) -> tuple[float, int, Optional[float]]:
        assert isinstance(self.agent, PPOAgent)
        obs = self.env.reset()
        ep_return = 0.0
        ep_len = 0
        done = False
        self._ep_buffer.clear()
        cfg = self.cfg

        while not done and ep_len < cfg.ppo_rollout_steps:
            action, log_prob = self.agent.act(obs)
            value = self.agent.get_value(obs)
            next_obs, reward, done, info = self.env.step(action)
            self._ep_buffer.push(obs, np.array([action]), reward, next_obs, done, log_prob, value)
            self._global_step += 1
            ep_return += reward
            ep_len += 1
            obs = next_obs

        # Train on collected rollout
        last_val = 0.0 if done else self.agent.get_value(obs)
        batches = self._ep_buffer.sample_batches(
            cfg.ppo_batch_size, last_value=last_val, seed=cfg.seed
        )

        epoch_losses = []
        for _ in range(cfg.ppo_epochs):
            for b in batches:
                losses = self.agent.train_epoch(b, n_epochs=1, batch_size=cfg.ppo_batch_size)
                epoch_losses.append(losses["total"])

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        return ep_return, ep_len, avg_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int = 20) -> EvalResult:
        """
        Evaluate the agent for n_episodes episodes (no exploration).

        Returns:
            EvalResult with aggregated metrics.
        """
        returns = []
        sharpes = []
        drawdowns = []
        n_trades_list = []
        env = self.eval_env

        for ep_idx in range(n_episodes):
            obs = env.reset()
            transitions = []
            done = False
            while not done:
                if isinstance(self.agent, (PPOAgent,)):
                    action = self.agent.act_greedy(obs)
                elif isinstance(self.agent, EnsembleAgent):
                    action = self.agent.act(obs)
                elif hasattr(self.agent, "act_greedy"):
                    action = self.agent.act_greedy(obs)
                else:
                    action = 0.0

                next_obs, reward, done, info = env.step(action)
                transitions.append({
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "equity": info.get("equity", 0.0),
                    "info": info,
                })
                obs = next_obs

            stats = episode_stats(transitions)
            returns.append(stats.total_return)
            sharpes.append(stats.sharpe_ratio)
            drawdowns.append(stats.max_drawdown)
            n_trades_list.append(stats.n_trades)

        return EvalResult(
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            mean_sharpe=float(np.mean(sharpes)),
            mean_max_drawdown=float(np.mean(drawdowns)),
            mean_n_trades=float(np.mean(n_trades_list)),
            episode_returns=returns,
            episode_sharpes=sharpes,
            episode_drawdowns=drawdowns,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _save_agent(self, path: str) -> None:
        if hasattr(self.agent, "save"):
            self.agent.save(path)

    def _log(
        self,
        episode: int,
        ep_return: float,
        ep_loss: Optional[float],
        eval_res: EvalResult,
        difficulty: float,
    ) -> None:
        print(
            f"[Ep {episode+1:5d}] "
            f"train_ret={ep_return:.4f} | "
            f"loss={ep_loss:.5f if ep_loss is not None else 0:.5f} | "
            f"eval_ret={eval_res.mean_return:.4f}±{eval_res.std_return:.4f} | "
            f"sharpe={eval_res.mean_sharpe:.3f} | "
            f"dd={eval_res.mean_max_drawdown:.3f} | "
            f"diff={difficulty:.2f}"
        )

    def plot_training_curves(
        self,
        result: Optional[TrainingResult] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training and evaluation return curves, loss, and drawdown.

        Args:
            result    : TrainingResult to plot (defaults to self._result).
            save_path : If provided, save the figure to this path.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Cannot plot training curves.")
            return

        r = result if result is not None else self._result

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Agent Training Curves", fontsize=14)

        # Episode returns
        ax = axes[0, 0]
        ax.plot(r.episode_returns, alpha=0.4, color="steelblue", label="Episode Return")
        if len(r.episode_returns) > 20:
            smooth = _moving_average(np.array(r.episode_returns), 20)
            ax.plot(smooth, color="steelblue", linewidth=2, label="MA(20)")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_title("Training Episode Returns")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.legend(fontsize=8)

        # Eval returns
        ax = axes[0, 1]
        if r.eval_episodes:
            ax.plot(r.eval_episodes, r.eval_returns, marker="o", color="darkorange", label="Eval Return")
            ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
            if r.best_episode:
                ax.axvline(r.best_episode, color="green", linestyle="--", alpha=0.7, label="Best")
        ax.set_title("Evaluation Returns")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Return")
        ax.legend(fontsize=8)

        # Training loss
        ax = axes[1, 0]
        if r.losses:
            ax.plot(r.losses, alpha=0.5, color="crimson", label="Loss")
            if len(r.losses) > 20:
                smooth = _moving_average(np.array(r.losses), 20)
                ax.plot(smooth, color="crimson", linewidth=2, label="MA(20)")
        ax.set_title("Training Loss")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

        # Eval sharpe and drawdown
        ax = axes[1, 1]
        if r.eval_results:
            sharpes = [e.mean_sharpe for e in r.eval_results]
            dds = [e.mean_max_drawdown for e in r.eval_results]
            ax.plot(r.eval_episodes, sharpes, marker="^", color="teal", label="Mean Sharpe")
            ax2 = ax.twinx()
            ax2.plot(r.eval_episodes, dds, marker="v", color="coral", label="Mean MaxDD")
            ax2.set_ylabel("MaxDrawdown", color="coral")
        ax.set_title("Eval Sharpe & Drawdown")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Sharpe", color="teal")
        ax.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average via convolution."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def quick_train(
    agent: Any,
    price_df,
    features: np.ndarray,
    n_episodes: int = 200,
    batch_size: int = 64,
    verbose: bool = True,
) -> TrainingResult:
    """
    Convenience function: build env + trainer and run training.

    Args:
        agent      : Any RL agent.
        price_df   : pd.DataFrame with 'close' column.
        features   : np.ndarray of shape (T, F).
        n_episodes : Number of training episodes.
        batch_size : Batch size for off-policy updates.
        verbose    : Print progress.

    Returns:
        TrainingResult.
    """
    env = TradingEnvironment(price_df, features)
    cfg = TrainingConfig(
        n_episodes=n_episodes,
        batch_size=batch_size,
        verbose=verbose,
    )
    trainer = AgentTrainer(agent, env, cfg)
    return trainer.train()
