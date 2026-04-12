"""
MAPPOTrainer — Centralized Training / Decentralized Execution for MAPPO.

Coordinates training of all MAPPO agents:
  - Rollout collection across all agents simultaneously
  - Centralized critic updates using global state
  - Decentralized actor updates using local observations
  - Multi-GPU support via DataParallel
  - TensorBoard logging
  - Checkpoint management
  - Early stopping on Nash equilibrium convergence
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore

from hyper_agent.env_compat import MarketEnvironment, MultiAgentTradingEnv
from hyper_agent.agents.mappo_agent import MAPPOAgent


# ============================================================
# Rollout Collector
# ============================================================

class RolloutCollector:
    """
    Collects N-step rollouts from the environment for all agents.

    Stores trajectories in per-agent PPORolloutBuffers.
    At end of rollout, computes GAE advantages for each agent.
    """

    def __init__(
        self,
        env:        MultiAgentTradingEnv,
        agents:     Dict[str, MAPPOAgent],
        rollout_len: int = 256,
    ) -> None:
        self.env         = env
        self.agents      = agents
        self.rollout_len = rollout_len

        self._current_obs:    Optional[Dict[str, np.ndarray]] = None
        self._current_global: Optional[np.ndarray]            = None
        self._episode_rewards: Dict[str, float] = {a: 0.0 for a in agents}
        self._episode_lens:   deque = deque(maxlen=100)
        self._ep_rewards_buf: deque = deque(maxlen=100)

    def reset(self) -> None:
        obs, _ = self.env.reset()
        self._current_obs    = obs
        self._current_global = self.env.get_global_state()
        self._episode_rewards = {a: 0.0 for a in self.agents}

    def collect(self) -> Dict[str, int]:
        """
        Collect rollout_len steps in environment.
        Returns dict with step counts per agent.
        """
        if self._current_obs is None:
            self.reset()

        step_counts = defaultdict(int)

        for t in range(self.rollout_len):
            actions_dict: Dict[str, np.ndarray] = {}
            log_probs:    Dict[str, float]       = {}
            values:       Dict[str, float]       = {}

            global_obs = self._current_global

            # Each agent selects action
            for aid, agent in self.agents.items():
                obs        = self._current_obs[aid]
                act, lp, _ = agent.act(obs, deterministic=False)
                val         = agent.compute_value(global_obs)

                actions_dict[aid] = act
                log_probs[aid]    = lp
                values[aid]       = val

            # Step environment
            next_obs, rewards, terminated, truncated, info = self.env.step(actions_dict)
            global_next = self.env.get_global_state()

            # Decode actions into dir/size for storage
            for aid, agent in self.agents.items():
                obs     = self._current_obs[aid]
                act     = actions_dict[aid]
                logits  = act[:3]
                probs   = np.exp(logits - logits.max())
                probs  /= probs.sum()
                act_dir = int(np.argmax(probs))
                act_sz  = float(np.clip(act[3], 0.0, 1.0))
                reward  = rewards.get(aid, 0.0)
                done    = terminated.get(aid, False) or truncated.get(aid, False)

                agent.store_rollout(
                    obs        = obs,
                    global_obs = global_obs,
                    act_dir    = act_dir,
                    act_sz     = act_sz,
                    log_prob   = log_probs[aid],
                    reward     = reward,
                    value      = values[aid],
                    done       = done,
                )
                self._episode_rewards[aid] += reward
                step_counts[aid] += 1

            self._current_obs    = next_obs
            self._current_global = global_next

            # Reset if done
            if any(terminated.values()) or any(truncated.values()):
                self._ep_rewards_buf.append(
                    float(np.mean(list(self._episode_rewards.values())))
                )
                self._episode_lens.append(t + 1)
                self.reset()

        # Compute GAE for all agents
        for aid, agent in self.agents.items():
            agent.finish_rollout(self._current_global)

        return dict(step_counts)

    def mean_episode_reward(self) -> float:
        if not self._ep_rewards_buf:
            return 0.0
        return float(np.mean(self._ep_rewards_buf))

    def mean_episode_len(self) -> float:
        if not self._episode_lens:
            return 0.0
        return float(np.mean(self._episode_lens))


# ============================================================
# MAPPOTrainer
# ============================================================

class MAPPOTrainer:
    """
    Centralized trainer for MAPPO agents.

    Training loop:
      1. Collect rollouts with all agents acting simultaneously
      2. Compute GAE advantages for each agent
      3. PPO update for each agent (actor + critic)
      4. Log metrics to TensorBoard
      5. Checkpoint every K episodes
      6. Check Nash convergence for early stopping

    Supports:
      - Multi-GPU via DataParallel (if available)
      - Mixed agent types in same environment
      - Periodic evaluation in deterministic mode
    """

    def __init__(
        self,
        env:              MultiAgentTradingEnv,
        agents:           Dict[str, MAPPOAgent],
        rollout_len:      int   = 256,
        n_total_steps:    int   = 2_000_000,
        eval_every:       int   = 50,
        checkpoint_every: int   = 100,
        checkpoint_dir:   str   = "./checkpoints",
        log_dir:          str   = "./logs",
        nash_eps:         float = 1e-3,
        nash_window:      int   = 20,
        use_tensorboard:  bool  = True,
        device:           str   = "cpu",
    ) -> None:
        self.env              = env
        self.agents           = agents
        self.rollout_len      = rollout_len
        self.n_total_steps    = n_total_steps
        self.eval_every       = eval_every
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir   = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.nash_eps         = nash_eps
        self.nash_window      = nash_window
        self.device           = device

        self.collector = RolloutCollector(env, agents, rollout_len)

        # TensorBoard
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=log_dir)

        # Tracking
        self.global_step      = 0
        self.episode_count    = 0
        self._policy_params:  Dict[str, np.ndarray] = {
            aid: agent.get_policy_params()
            for aid, agent in agents.items()
        }
        self._policy_changes: deque = deque(maxlen=nash_window)
        self._train_stats:    List[Dict] = []

        # Multi-GPU setup
        self._setup_multigpu()

    def _setup_multigpu(self) -> None:
        if torch.cuda.device_count() > 1:
            for aid, agent in self.agents.items():
                agent.actor  = nn.DataParallel(agent.actor)
                agent.critic = nn.DataParallel(agent.critic)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns final training statistics.
        """
        self.collector.reset()
        start_time   = time.time()
        best_reward  = float("-inf")

        while self.global_step < self.n_total_steps:
            # Collect rollout
            step_counts = self.collector.collect()
            self.global_step += sum(step_counts.values())
            self.episode_count += 1

            # Update all agents
            update_stats = self._update_all_agents()

            # Track Nash convergence
            nash_converged = self._check_nash_convergence()

            # Logging
            if self.episode_count % 10 == 0:
                self._log_stats(update_stats)

            # Evaluation
            if self.episode_count % self.eval_every == 0:
                eval_reward = self._evaluate()
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self._save_best()

            # Checkpoint
            if self.episode_count % self.checkpoint_every == 0:
                self._checkpoint()

            # Early stopping: Nash equilibrium
            if nash_converged:
                print(f"Nash equilibrium converged at step {self.global_step}. Stopping.")
                break

        elapsed = time.time() - start_time
        print(f"Training complete: {self.global_step:,} steps in {elapsed:.1f}s")
        return self._final_stats()

    def _update_all_agents(self) -> Dict[str, Dict[str, float]]:
        """Run PPO update for each agent. Returns per-agent stats."""
        all_stats = {}
        for aid, agent in self.agents.items():
            stats = agent.update()
            if stats:
                all_stats[aid] = stats
        return all_stats

    def _check_nash_convergence(self) -> bool:
        """
        Check if policies have converged (≈ Nash equilibrium).

        Convergence criterion: mean policy change over last window < epsilon.
        """
        changes = []
        for aid, agent in self.agents.items():
            old_params = self._policy_params[aid]
            new_params = agent.get_policy_params()
            change     = float(np.linalg.norm(new_params - old_params))
            changes.append(change)
            self._policy_params[aid] = new_params

        mean_change = float(np.mean(changes))
        self._policy_changes.append(mean_change)

        if len(self._policy_changes) < self.nash_window:
            return False
        return float(np.mean(self._policy_changes)) < self.nash_eps

    def _evaluate(self, n_episodes: int = 3) -> float:
        """Run deterministic evaluation. Returns mean episode reward."""
        total_rewards = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_rewards = {a: 0.0 for a in self.agents}
            done = False
            steps = 0

            while not done and steps < self.env.env.max_steps:
                actions = {}
                for aid, agent in self.agents.items():
                    act, _, _ = agent.act(obs[aid], deterministic=True)
                    actions[aid] = act

                next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
                for aid in self.agents:
                    ep_rewards[aid] += rewards.get(aid, 0.0)

                obs   = next_obs
                done  = terminated.get("__all__", False) or truncated.get("__all__", False)
                steps += 1

            total_rewards.append(float(np.mean(list(ep_rewards.values()))))

        mean_reward = float(np.mean(total_rewards))
        if self.writer:
            self.writer.add_scalar("eval/mean_reward", mean_reward, self.global_step)
        return mean_reward

    def _log_stats(self, update_stats: Dict) -> None:
        if not self.writer:
            return

        ep_reward = self.collector.mean_episode_reward()
        ep_len    = self.collector.mean_episode_len()

        self.writer.add_scalar("train/mean_episode_reward", ep_reward, self.global_step)
        self.writer.add_scalar("train/mean_episode_len",    ep_len,    self.global_step)

        # Per-agent stats
        for aid, stats in update_stats.items():
            for k, v in stats.items():
                self.writer.add_scalar(f"agent/{aid}/{k}", v, self.global_step)

        # Nash convergence metric
        if self._policy_changes:
            self.writer.add_scalar(
                "train/mean_policy_change",
                float(np.mean(self._policy_changes)),
                self.global_step,
            )

    def _checkpoint(self) -> None:
        """Save checkpoint for all agents."""
        for aid, agent in self.agents.items():
            path = self.checkpoint_dir / f"{aid}_ep{self.episode_count}.pt"
            agent.save_checkpoint(str(path), self.episode_count)

    def _save_best(self) -> None:
        """Save best checkpoint."""
        for aid, agent in self.agents.items():
            path = self.checkpoint_dir / f"{aid}_best.pt"
            agent.save_checkpoint(str(path), self.episode_count)

    def _final_stats(self) -> Dict[str, Any]:
        return {
            "global_steps":    self.global_step,
            "episodes":        self.episode_count,
            "mean_ep_reward":  self.collector.mean_episode_reward(),
            "mean_ep_len":     self.collector.mean_episode_len(),
            "policy_changes":  list(self._policy_changes),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def load_checkpoints(self, checkpoint_dir: Optional[str] = None) -> None:
        """Load latest checkpoints for all agents."""
        ckpt_dir = Path(checkpoint_dir or self.checkpoint_dir)
        for aid, agent in self.agents.items():
            best_path = ckpt_dir / f"{aid}_best.pt"
            if best_path.exists():
                ep = agent.load_checkpoint(str(best_path))
                print(f"Loaded {aid} checkpoint from episode {ep}")

    def get_policy_snapshot(self) -> Dict[str, np.ndarray]:
        """Return snapshot of all current policy parameters."""
        return {
            aid: agent.get_policy_params().copy()
            for aid, agent in self.agents.items()
        }

    def policy_distance_from_snapshot(
        self, snapshot: Dict[str, np.ndarray]
    ) -> float:
        """Measure L2 distance of current policies from a prior snapshot."""
        distances = []
        for aid, agent in self.agents.items():
            if aid in snapshot:
                curr   = agent.get_policy_params()
                prev   = snapshot[aid]
                dist   = float(np.linalg.norm(curr - prev))
                distances.append(dist)
        return float(np.mean(distances)) if distances else 0.0

    def per_agent_rewards(self) -> Dict[str, float]:
        """Return recent average reward per agent."""
        return {
            aid: float(np.mean(agent._episode_rewards) if agent._episode_rewards else 0.0)
            for aid, agent in self.agents.items()
        }

    def close(self) -> None:
        if self.writer:
            self.writer.close()


# ============================================================
# Factory: build MAPPO trainer from env
# ============================================================

def build_mappo_trainer(
    env:              MultiAgentTradingEnv,
    obs_dim:          int,
    hidden_dim:       int   = 64,
    lr:               float = 3e-4,
    rollout_len:      int   = 256,
    n_total_steps:    int   = 1_000_000,
    checkpoint_dir:   str   = "./checkpoints",
    log_dir:          str   = "./logs/mappo",
    device:           str   = "cpu",
) -> MAPPOTrainer:
    """
    Build a MAPPOTrainer with one MAPPOAgent per env agent.
    """
    agent_ids      = env.agent_ids
    global_dim     = obs_dim * len(agent_ids)

    agents: Dict[str, MAPPOAgent] = {}
    for aid in agent_ids:
        agents[aid] = MAPPOAgent(
            agent_id         = aid,
            obs_dim          = obs_dim,
            global_state_dim = global_dim,
            hidden_dim       = hidden_dim,
            lr               = lr,
            rollout_len      = rollout_len,
            device           = device,
        )

    return MAPPOTrainer(
        env             = env,
        agents          = agents,
        rollout_len     = rollout_len,
        n_total_steps   = n_total_steps,
        checkpoint_dir  = checkpoint_dir,
        log_dir         = log_dir,
        device          = device,
    )
