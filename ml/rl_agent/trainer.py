"""
ml/rl_agent/trainer.py -- Training orchestrator for DQN and PPO RL exit policy agents.

Provides RLTrainer which:
  - Trains DQN and PPO agents on the TradingEnvironment
  - Evaluates agents on held-out episodes
  - Exports trained DQN to config/rl_exit_qtable.json (compatible with
    RLExitPolicy in tools/live_trader_alpaca.py)
  - Compares DQN vs PPO head-to-head

All I/O uses standard library. No external ML dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.rl_agent.environment import (
    TradingEnvironment,
    TradingState,
    TradeEpisodeGenerator,
    HOLD,
    PARTIAL_EXIT,
    FULL_EXIT,
    MAX_BARS,
    N_FEATURES,
    N_ACTIONS,
)
from ml.rl_agent.q_network import DQNAgent, BATCH_SIZE
from ml.rl_agent.ppo_agent import PPOAgent, Episode

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QTABLE_PATH = _REPO_ROOT / "config" / "rl_exit_qtable.json"
DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "config" / "rl_checkpoints"

N_BINS = 5   # discretization bins -- matches RLExitPolicy._N_BINS


# ---------------------------------------------------------------------------
# Training config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Hyperparameters and settings for RL training.

    Attributes
    ----------
    dqn_episodes : int
        Number of training episodes for DQN.
    ppo_episodes : int
        Number of training episodes for PPO.
    eval_episodes : int
        Number of episodes for evaluation.
    save_every : int
        Save DQN checkpoint every N episodes.
    log_every : int
        Print training log every N episodes.
    dqn_train_steps_per_episode : int
        Number of gradient steps per episode for DQN.
    ppo_episodes_per_update : int
        Number of episodes to collect before a PPO update.
    seed : int
    checkpoint_dir : str
    """

    dqn_episodes: int = 2000
    ppo_episodes: int = 2000
    eval_episodes: int = 200
    save_every: int = 500
    log_every: int = 100
    dqn_train_steps_per_episode: int = 4
    ppo_episodes_per_update: int = 8
    seed: int = 42
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Evaluation metrics for an RL agent."""

    avg_pnl: float = 0.0
    avg_hold_bars: float = 0.0
    exit_at_target_rate: float = 0.0   # fraction of episodes ending with positive PnL
    stop_loss_hit_rate: float = 0.0
    voluntary_exit_rate: float = 0.0   # fraction ended by FULL_EXIT (not forced)
    partial_exit_rate: float = 0.0     # fraction of episodes that used PARTIAL_EXIT
    vs_hold_pnl: float = 0.0           # avg PnL vs. passive hold-to-end strategy
    n_episodes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_pnl": round(self.avg_pnl, 6),
            "avg_hold_bars": round(self.avg_hold_bars, 2),
            "exit_at_target_rate": round(self.exit_at_target_rate, 4),
            "stop_loss_hit_rate": round(self.stop_loss_hit_rate, 4),
            "voluntary_exit_rate": round(self.voluntary_exit_rate, 4),
            "partial_exit_rate": round(self.partial_exit_rate, 4),
            "vs_hold_pnl": round(self.vs_hold_pnl, 6),
            "n_episodes": self.n_episodes,
        }


# ---------------------------------------------------------------------------
# RLTrainer
# ---------------------------------------------------------------------------

class RLTrainer:
    """
    Training orchestrator for DQN and PPO exit policy agents.

    Parameters
    ----------
    config : TrainConfig, optional
    seed : int, optional
        Overrides config.seed if provided.
    """

    def __init__(
        self,
        config: Optional[TrainConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = config or TrainConfig()
        if seed is not None:
            self.cfg.seed = seed

        self.rng = np.random.default_rng(self.cfg.seed)
        self.ep_gen = TradeEpisodeGenerator(seed=self.cfg.seed)

        # Separate generators for train and eval splits
        self.train_env = TradingEnvironment(
            episode_generator=TradeEpisodeGenerator(seed=self.cfg.seed),
            seed=self.cfg.seed,
        )
        self.eval_env = TradingEnvironment(
            episode_generator=TradeEpisodeGenerator(seed=self.cfg.seed + 9999),
            seed=self.cfg.seed + 9999,
        )

        # Ensure checkpoint dir exists
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DQN Training
    # ------------------------------------------------------------------

    def train_dqn(
        self,
        n_episodes: Optional[int] = None,
        save_every: Optional[int] = None,
    ) -> DQNAgent:
        """
        Train a DQN agent.

        Parameters
        ----------
        n_episodes : int, optional
            Overrides config.dqn_episodes.
        save_every : int, optional
            Overrides config.save_every.

        Returns
        -------
        DQNAgent
            Trained DQN agent.
        """
        n_episodes = n_episodes or self.cfg.dqn_episodes
        save_every = save_every or self.cfg.save_every

        agent = DQNAgent(seed=self.cfg.seed)
        env = self.train_env

        log.info("Starting DQN training: %d episodes", n_episodes)
        t0 = time.time()

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []

        for ep_idx in range(n_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0

            while not done:
                action = agent.act(obs, explore=True)
                next_obs, reward, done, info = env.step(action)
                agent.store(obs, action, reward, next_obs, done)

                # Gradient steps
                for _ in range(self.cfg.dqn_train_steps_per_episode):
                    agent.train_step()

                ep_reward += reward
                ep_len += 1
                obs = next_obs

            agent.episode_rewards.append(ep_reward)
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)

            if (ep_idx + 1) % self.cfg.log_every == 0:
                recent_rewards = episode_rewards[-self.cfg.log_every:]
                elapsed = time.time() - t0
                log.info(
                    "DQN ep %d/%d | eps=%.3f | mean_reward=%.2f | "
                    "mean_len=%.1f | steps=%d | %.1fs elapsed",
                    ep_idx + 1,
                    n_episodes,
                    agent.epsilon,
                    float(np.mean(recent_rewards)),
                    float(np.mean(episode_lengths[-self.cfg.log_every:])),
                    agent._step,
                    elapsed,
                )

            if (ep_idx + 1) % save_every == 0:
                ckpt_path = os.path.join(
                    self.cfg.checkpoint_dir, f"dqn_ep{ep_idx+1}"
                )
                agent.save(ckpt_path)
                log.info("DQN checkpoint saved: %s", ckpt_path)

        log.info(
            "DQN training complete. Total steps: %d, final eps: %.4f",
            agent._step, agent.epsilon,
        )
        return agent

    # ------------------------------------------------------------------
    # PPO Training
    # ------------------------------------------------------------------

    def train_ppo(
        self,
        n_episodes: Optional[int] = None,
    ) -> PPOAgent:
        """
        Train a PPO agent.

        Parameters
        ----------
        n_episodes : int, optional
            Overrides config.ppo_episodes.

        Returns
        -------
        PPOAgent
            Trained PPO agent.
        """
        n_episodes = n_episodes or self.cfg.ppo_episodes
        per_update = self.cfg.ppo_episodes_per_update

        agent = PPOAgent(seed=self.cfg.seed)
        env = self.train_env

        log.info("Starting PPO training: %d episodes", n_episodes)
        t0 = time.time()

        ep_idx = 0
        while ep_idx < n_episodes:
            batch_episodes: List[Episode] = []
            batch_count = min(per_update, n_episodes - ep_idx)

            for _ in range(batch_count):
                ep = agent.collect_episode(env)
                batch_episodes.append(ep)
                ep_idx += 1

            update_info = agent.update(batch_episodes)

            if ep_idx % self.cfg.log_every < per_update:
                recent = agent.episode_returns[-self.cfg.log_every:]
                elapsed = time.time() - t0
                log.info(
                    "PPO ep %d/%d | mean_return=%.2f | "
                    "policy_loss=%.4f | value_loss=%.4f | entropy=%.4f | %.1fs",
                    ep_idx,
                    n_episodes,
                    float(np.mean(recent)) if recent else 0.0,
                    update_info["policy_loss"],
                    update_info["value_loss"],
                    update_info["entropy"],
                    elapsed,
                )

        log.info("PPO training complete. Total episodes: %d", ep_idx)
        return agent

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        agent,
        n_episodes: Optional[int] = None,
        use_deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate an agent on held-out episodes.

        Works with both DQNAgent and PPOAgent (duck-typed via act() method).

        Parameters
        ----------
        agent : DQNAgent or PPOAgent
        n_episodes : int, optional
        use_deterministic : bool
            For PPO: use argmax action. For DQN: explore=False.

        Returns
        -------
        dict -- EvalResult metrics
        """
        n_episodes = n_episodes or self.cfg.eval_episodes
        env = self.eval_env

        pnl_list: List[float] = []
        hold_bars_list: List[float] = []
        stop_loss_hits: int = 0
        voluntary_exits: int = 0
        partial_exit_episodes: int = 0
        hold_pnl_list: List[float] = []

        is_dqn = isinstance(agent, DQNAgent)
        is_ppo = isinstance(agent, PPOAgent)

        for _ in range(n_episodes):
            episode = self.eval_env.episode_gen.random_episode()
            obs = env.reset(trade_episode=episode)
            done = False
            ep_len = 0
            used_partial = False
            terminal_type = "voluntary"
            info: Dict[str, Any] = {}

            while not done:
                if is_dqn:
                    action = agent.act(obs, explore=False)
                elif is_ppo:
                    action = agent.act(obs, deterministic=use_deterministic)
                else:
                    action = int(agent.act(obs))

                if action == PARTIAL_EXIT:
                    used_partial = True

                obs, reward, done, info = env.step(action)
                ep_len += 1

                if done:
                    if info.get("terminal") == "stop_loss":
                        terminal_type = "stop_loss"
                        stop_loss_hits += 1
                    elif info.get("terminal") == "max_bars":
                        terminal_type = "max_bars"
                    else:
                        voluntary_exits += 1

            pnl = info.get("episode_pnl", 0.0)
            pnl_list.append(float(pnl))
            hold_bars_list.append(float(ep_len))

            if used_partial:
                partial_exit_episodes += 1

            # Passive hold comparison: hold to the end of the episode
            hold_pnl = _passive_hold_pnl(episode)
            hold_pnl_list.append(hold_pnl)

        avg_pnl = float(np.mean(pnl_list))
        avg_hold = float(np.mean(hold_bars_list))
        avg_hold_pnl = float(np.mean(hold_pnl_list))

        result = EvalResult(
            avg_pnl=avg_pnl,
            avg_hold_bars=avg_hold,
            exit_at_target_rate=float(np.mean([p > 0 for p in pnl_list])),
            stop_loss_hit_rate=stop_loss_hits / n_episodes,
            voluntary_exit_rate=voluntary_exits / n_episodes,
            partial_exit_rate=partial_exit_episodes / n_episodes,
            vs_hold_pnl=avg_pnl - avg_hold_pnl,
            n_episodes=n_episodes,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Q-table export for RLExitPolicy compatibility
    # ------------------------------------------------------------------

    def export_qtable(
        self,
        agent: DQNAgent,
        path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Export a trained DQN as a discrete Q-table compatible with
        RLExitPolicy in tools/live_trader_alpaca.py.

        Evaluates the DQN at all 5^5 = 3125 discrete state combinations.
        Each key maps to [Q(HOLD), Q(EXIT)] (2-element list) to match the
        existing JSON schema read by RLExitPolicy.

        The 5 features discretized are (matching RLExitPolicy._state_key):
          f0: pnl_pct (scaled x2 to [-1,1] at +-50%)
          f1: bars_held normalized to [0,1], then to [-1,1]
          f2: bh_mass [0,1] -> [-1,1]
          f3: bh_active binary (0 or 1)
          f4: atr_ratio centered at 1.0

        Parameters
        ----------
        agent : DQNAgent
        path : str, optional
            Write path for JSON output. Defaults to config/rl_exit_qtable.json.

        Returns
        -------
        dict -- the Q-table as {state_key: [q_hold, q_exit]}
        """
        if path is None:
            path = str(DEFAULT_QTABLE_PATH)

        qtable: Dict[str, List[float]] = {}

        def _bin_centers(lo: float, hi: float, n: int = N_BINS) -> np.ndarray:
            edges = np.linspace(lo, hi, n + 1)
            return (edges[:-1] + edges[1:]) / 2.0

        # Bin center values for each of the 5 key features
        # These represent the continuous values at the center of each bin.
        pnl_centers = _bin_centers(-1.0, 1.0) / 2.0        # pnl in [-0.4, 0.4]
        bars_centers = (_bin_centers(-1.0, 1.0) + 1.0) / 2.0  # bars_norm in [0.1, 0.9]
        bh_mass_centers = (_bin_centers(-1.0, 1.0) + 1.0) / 2.0
        atr_centers = _bin_centers(-1.0, 1.0) + 1.0         # atr in [0.0, 2.0]

        # For f3 we use the full 5-bin sweep but map to actual 0/1 values
        # This produces 5^4 * 5 = 3125 total entries (5^5) as specified.
        # For bh_active continuous bins: 0..4 -> map threshold > 2 = active
        bh_active_bin_vals = [0.0, 0.0, 0.0, 1.0, 1.0]  # bins 0-2 = inactive, 3-4 = active

        count = 0
        for b0 in range(N_BINS):
            pnl = float(pnl_centers[b0])
            for b1 in range(N_BINS):
                bars_norm = float(bars_centers[b1])
                for b2 in range(N_BINS):
                    bh_mass = float(bh_mass_centers[b2])
                    for b3 in range(N_BINS):
                        bh_active = bh_active_bin_vals[b3]
                        for b4 in range(N_BINS):
                            atr = float(atr_centers[b4])

                            state = np.array([
                                float(np.clip(pnl, -1.0, 1.0)),
                                float(bars_norm),
                                float(bh_mass),
                                float(bh_active),
                                float(np.clip(atr, 0.0, 5.0)),
                                0.5,   # hurst_h neutral
                                0.0,   # nav_omega neutral
                                0.5,   # vol_percentile neutral
                                0.0,   # tod_sin
                                1.0,   # tod_cos
                            ], dtype=np.float32)

                            q_values = agent.online.predict(state)  # (3,)

                            # Collapse 3 actions to 2 for RLExitPolicy compatibility
                            q_hold = float(q_values[HOLD])
                            q_exit = float(max(q_values[PARTIAL_EXIT], q_values[FULL_EXIT]))

                            key = f"{b0},{b1},{b2},{b3},{b4}"
                            qtable[key] = [round(q_hold, 6), round(q_exit, 6)]
                            count += 1

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(qtable, separators=(",", ":")))
        log.info("Q-table exported: %d states -> %s", count, out_path)

        return qtable

    # ------------------------------------------------------------------
    # Head-to-head agent comparison
    # ------------------------------------------------------------------

    def compare_agents(
        self,
        dqn: DQNAgent,
        ppo: PPOAgent,
        n_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compare DQN and PPO agents on the same held-out episodes.

        Both agents are evaluated on identical episode sequences by using
        the same random seed for the evaluation environment.

        Parameters
        ----------
        dqn : DQNAgent
        ppo : PPOAgent
        n_episodes : int, optional

        Returns
        -------
        dict with "dqn", "ppo", and "comparison" sub-dicts
        """
        n_episodes = n_episodes or self.cfg.eval_episodes

        # Evaluate DQN
        dqn_results = self.evaluate(dqn, n_episodes=n_episodes)

        # Reset eval env seed for fair comparison
        self.eval_env = TradingEnvironment(
            episode_generator=TradeEpisodeGenerator(seed=self.cfg.seed + 9999),
            seed=self.cfg.seed + 9999,
        )
        ppo_results = self.evaluate(ppo, n_episodes=n_episodes)

        comparison = {
            "avg_pnl_delta": round(ppo_results["avg_pnl"] - dqn_results["avg_pnl"], 6),
            "avg_hold_bars_delta": round(
                ppo_results["avg_hold_bars"] - dqn_results["avg_hold_bars"], 2
            ),
            "voluntary_exit_rate_delta": round(
                ppo_results["voluntary_exit_rate"] - dqn_results["voluntary_exit_rate"], 4
            ),
            "stop_loss_delta": round(
                ppo_results["stop_loss_hit_rate"] - dqn_results["stop_loss_hit_rate"], 4
            ),
            "vs_hold_pnl_delta": round(
                ppo_results["vs_hold_pnl"] - dqn_results["vs_hold_pnl"], 6
            ),
            "winner": (
                "ppo" if ppo_results["avg_pnl"] > dqn_results["avg_pnl"] else "dqn"
            ),
        }

        return {
            "dqn": dqn_results,
            "ppo": ppo_results,
            "comparison": comparison,
        }

    # ------------------------------------------------------------------
    # Convenience: train both and export
    # ------------------------------------------------------------------

    def train_and_export(self) -> Dict[str, Any]:
        """
        Train both DQN and PPO, evaluate, compare, and export Q-table.

        Returns
        -------
        dict with training summary, evaluation results, and comparison
        """
        t0 = time.time()
        log.info("=== SRFM RL Training Pipeline ===")

        dqn_agent = self.train_dqn()
        ppo_agent = self.train_ppo()

        comparison = self.compare_agents(dqn_agent, ppo_agent)
        qtable = self.export_qtable(dqn_agent)

        elapsed = time.time() - t0
        log.info("Full training pipeline complete in %.1fs", elapsed)

        return {
            "dqn_eval": comparison["dqn"],
            "ppo_eval": comparison["ppo"],
            "comparison": comparison["comparison"],
            "qtable_states": len(qtable),
            "qtable_path": str(DEFAULT_QTABLE_PATH),
            "elapsed_seconds": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passive_hold_pnl(episode) -> float:
    """
    Compute the PnL of holding the full position until the last bar.

    Parameters
    ----------
    episode : TradeEpisode

    Returns
    -------
    float -- (last_price - entry_price) / entry_price
    """
    entry = float(episode.prices[0])
    last = float(episode.prices[-1])
    if entry == 0:
        return 0.0
    return (last - entry) / entry
