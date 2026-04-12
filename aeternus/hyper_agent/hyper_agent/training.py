"""
training.py — Training orchestration for Hyper-Agent MARL.

Implements:
- Episode rollout with centralized training / decentralized execution (CTDE)
- Curriculum learning (progressive difficulty)
- Self-play (agent vs own past versions)
- League training (diverse opponent pool)
- Multi-algorithm support (MAPPO, QMIX, COMA, MFG)
- Parallel environment rollout
- Training metrics and logging
"""

from __future__ import annotations

import os
import math
import copy
import logging
import time
import collections
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .environment import MultiAssetTradingEnv, VecTradingEnv, make_env
from .agents.base_agent import BaseAgent, Transition
from .agents.mappo_agent import MAPPOAgent, MAPPOPopulation, MARolloutBuffer
from .population import AgentPopulation, ScriptedAgent
from .emergence import EmergenceAnalyzer
from .replay_buffer import MultiAgentReplayBuffer, EpisodeReplayBuffer

logger = logging.getLogger(__name__)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

class TrainingConfig:
    """Configuration for the training loop."""

    def __init__(
        self,
        # Environment
        num_assets: int = 4,
        num_agents: int = 8,
        max_steps_per_episode: int = 500,
        num_envs: int = 1,

        # Training
        total_timesteps: int = 1_000_000,
        rollout_length: int = 200,
        num_epochs: int = 10,
        mini_batch_size: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,

        # Algorithm
        algorithm: str = "mappo",  # "mappo", "qmix", "coma", "mfg", "sac"
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,

        # Curriculum
        use_curriculum: bool = True,
        curriculum_stages: int = 3,

        # Self-play
        use_self_play: bool = False,
        self_play_interval: int = 500,

        # Logging
        log_interval: int = 10,
        save_interval: int = 100,
        eval_interval: int = 50,
        save_dir: str = "checkpoints",

        # Misc
        device: str = "cpu",
        seed: int = 42,
    ):
        self.num_assets = num_assets
        self.num_agents = num_agents
        self.max_steps_per_episode = max_steps_per_episode
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.rollout_length = rollout_length
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.algorithm = algorithm
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.use_curriculum = use_curriculum
        self.curriculum_stages = curriculum_stages
        self.use_self_play = use_self_play
        self.self_play_interval = self_play_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.device = device
        self.seed = seed


# ---------------------------------------------------------------------------
# Curriculum learning
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Progressive curriculum for MARL training.

    Stages:
    1. Single asset, few agents, simple reward
    2. Multi-asset, more agents, full reward
    3. Full complexity with adversarial agents
    """

    def __init__(
        self,
        num_stages: int = 3,
        advancement_threshold: float = 0.8,
        min_episodes_per_stage: int = 100,
    ):
        self.num_stages = num_stages
        self.advancement_threshold = advancement_threshold
        self.min_episodes_per_stage = min_episodes_per_stage

        self._stage = 0
        self._stage_episodes = 0
        self._stage_rewards: List[float] = []
        self._stage_history: List[Dict] = []

    @property
    def stage(self) -> int:
        return self._stage

    def get_stage_config(self) -> Dict:
        """Return environment config for current stage."""
        base = {
            "num_assets": 1,
            "num_agents": 2,
            "max_steps": 200,
            "enable_flash_crash": False,
            "enable_circuit_breaker": False,
        }

        if self._stage >= 1:
            base["num_assets"] = 2
            base["num_agents"] = 4
            base["max_steps"] = 400
            base["enable_circuit_breaker"] = True

        if self._stage >= 2:
            base["num_assets"] = 4
            base["num_agents"] = 8
            base["max_steps"] = 500
            base["enable_flash_crash"] = True

        return base

    def should_advance(self) -> bool:
        if self._stage >= self.num_stages - 1:
            return False
        if self._stage_episodes < self.min_episodes_per_stage:
            return False
        if len(self._stage_rewards) < 20:
            return False
        mean_reward = float(np.mean(self._stage_rewards[-20:]))
        return mean_reward > self.advancement_threshold

    def record_episode(self, mean_reward: float) -> bool:
        """Record episode result. Returns True if stage advanced."""
        self._stage_episodes += 1
        self._stage_rewards.append(mean_reward)

        if self.should_advance():
            self._stage_history.append({
                "stage": self._stage,
                "episodes": self._stage_episodes,
                "final_mean_reward": float(np.mean(self._stage_rewards[-20:])),
            })
            self._stage += 1
            self._stage_episodes = 0
            self._stage_rewards.clear()
            logger.info(f"Curriculum advanced to stage {self._stage}")
            return True
        return False


# ---------------------------------------------------------------------------
# Self-play league
# ---------------------------------------------------------------------------

class SelfPlayLeague:
    """
    Maintains a league of agent checkpoints for diverse self-play training.
    """

    def __init__(
        self,
        max_checkpoints: int = 20,
        selection_strategy: str = "priority",  # "uniform", "priority", "pfsp"
    ):
        self.max_checkpoints = max_checkpoints
        self.selection_strategy = selection_strategy
        self._checkpoints: List[Dict] = []
        self._win_rates: List[float] = []
        self._step = 0

    def add_checkpoint(
        self,
        agent_states: List[Dict],
        step: int,
        performance: float = 0.0,
    ) -> None:
        """Add current agent states to the league."""
        checkpoint = {
            "states": agent_states,
            "step": step,
            "performance": performance,
            "added_at": time.time(),
        }
        if len(self._checkpoints) < self.max_checkpoints:
            self._checkpoints.append(checkpoint)
            self._win_rates.append(0.5)
        else:
            # Replace lowest performance checkpoint
            min_idx = int(np.argmin([c["performance"] for c in self._checkpoints]))
            self._checkpoints[min_idx] = checkpoint
            self._win_rates[min_idx] = 0.5

    def sample_opponent(self) -> Optional[Dict]:
        """Sample an opponent checkpoint."""
        if not self._checkpoints:
            return None

        if self.selection_strategy == "uniform":
            idx = np.random.randint(len(self._checkpoints))
        elif self.selection_strategy == "priority":
            # Sample proportional to (1 - win_rate): choose harder opponents
            difficulties = [max(0.01, 1.0 - wr) for wr in self._win_rates]
            total = sum(difficulties)
            probs = [d / total for d in difficulties]
            idx = int(np.random.choice(len(self._checkpoints), p=probs))
        elif self.selection_strategy == "pfsp":
            # Prioritized fictitious self-play
            win_rates = np.array(self._win_rates)
            priorities = np.abs(win_rates - 0.5)
            total = priorities.sum() + EPS
            probs = priorities / total
            idx = int(np.random.choice(len(self._checkpoints), p=probs))
        else:
            idx = np.random.randint(len(self._checkpoints))

        return self._checkpoints[idx]

    def update_win_rate(self, checkpoint_idx: int, won: bool) -> None:
        """Update win rate against a checkpoint."""
        if 0 <= checkpoint_idx < len(self._win_rates):
            self._win_rates[checkpoint_idx] = (
                0.9 * self._win_rates[checkpoint_idx] + 0.1 * float(won)
            )

    def __len__(self) -> int:
        return len(self._checkpoints)


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

class EpisodeRunner:
    """
    Collects episode rollouts from a multi-agent environment.
    Supports both on-policy (for MAPPO) and off-policy (for SAC/QMIX) collection.
    """

    def __init__(
        self,
        env: MultiAssetTradingEnv,
        agents: List[BaseAgent],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        use_global_state: bool = True,
    ):
        self.env = env
        self.agents = agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_global_state = use_global_state
        self.num_agents = len(agents)

        self._episode_count = 0
        self._total_steps = 0

    def collect_rollout(
        self,
        rollout_buffer: MARolloutBuffer,
        num_steps: int,
        reset_env: bool = False,
        deterministic: bool = False,
    ) -> Dict[str, float]:
        """
        Collect num_steps steps into rollout_buffer.
        Returns episode statistics.
        """
        if reset_env:
            self.env.reset()
            for agent in self.agents:
                if hasattr(agent, "reset_hidden"):
                    agent.reset_hidden()

        obs_list = self.env.get_all_observations()
        global_state = self.env.get_state() if self.use_global_state else None

        episode_rewards = [0.0] * self.num_agents
        episode_lengths = [0] * self.num_agents
        stats: Dict[str, Any] = collections.defaultdict(list)

        steps_collected = 0
        while steps_collected < num_steps:
            # Collect actions
            actions = []
            log_probs = []
            values = []

            for i, agent in enumerate(self.agents):
                if hasattr(agent, "select_action_with_value"):
                    a, lp, v = agent.select_action_with_value(
                        obs_list[i],
                        global_state if global_state is not None else obs_list[i],
                        deterministic=deterministic,
                    )
                else:
                    a, lp, v = agent.select_action(obs_list[i], deterministic=deterministic)
                actions.append(a)
                log_probs.append(lp)
                values.append(v)

            # Step environment
            next_obs_list, rewards, terminated, truncated, infos = self.env._marl_step(actions)
            dones = [t or tr for t, tr in zip(terminated, truncated)]

            next_global_state = self.env.get_state() if self.use_global_state else None

            # Store in rollout buffer
            rollout_buffer.add_step(
                obs_list=obs_list,
                actions=actions,
                rewards=rewards,
                dones=dones,
                log_probs=log_probs,
                values=values,
                global_state=global_state,
            )

            # Accumulate stats
            for i in range(self.num_agents):
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1

            obs_list = next_obs_list
            global_state = next_global_state
            steps_collected += 1
            self._total_steps += 1

            if any(dones):
                stats["episode_reward"].append(float(np.mean(episode_rewards)))
                stats["episode_length"].append(float(np.mean(episode_lengths)))
                episode_rewards = [0.0] * self.num_agents
                episode_lengths = [0] * self.num_agents
                self._episode_count += 1
                self.env.reset()
                obs_list = self.env.get_all_observations()
                global_state = self.env.get_state() if self.use_global_state else None
                for agent in self.agents:
                    if hasattr(agent, "reset_hidden"):
                        agent.reset_hidden()

        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in stats.items()
        }

    def collect_episode(
        self,
        max_steps: Optional[int] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], List[List[Transition]]]:
        """
        Collect a complete episode.
        Returns (episode_info, per_agent_transitions).
        """
        max_steps = max_steps or self.env.max_steps
        self.env.reset()
        for agent in self.agents:
            if hasattr(agent, "reset_hidden"):
                agent.reset_hidden()

        obs_list = self.env.get_all_observations()
        global_state = self.env.get_state() if self.use_global_state else None

        agent_transitions: List[List[Transition]] = [[] for _ in range(self.num_agents)]
        total_rewards = [0.0] * self.num_agents
        step = 0

        while step < max_steps:
            actions, log_probs, values = [], [], []
            for i, agent in enumerate(self.agents):
                if hasattr(agent, "select_action_with_value"):
                    a, lp, v = agent.select_action_with_value(
                        obs_list[i],
                        global_state if global_state is not None else obs_list[i],
                        deterministic=deterministic,
                    )
                else:
                    a, lp, v = agent.select_action(obs_list[i], deterministic=deterministic)
                actions.append(a)
                log_probs.append(lp)
                values.append(v)

            next_obs_list, rewards, terminated, truncated, infos = self.env._marl_step(actions)
            dones = [t or tr for t, tr in zip(terminated, truncated)]
            next_global_state = self.env.get_state() if self.use_global_state else None

            for i in range(self.num_agents):
                t = Transition(
                    obs=obs_list[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_obs=next_obs_list[i],
                    done=dones[i],
                    log_prob=log_probs[i],
                    value=values[i],
                    global_state=global_state,
                )
                agent_transitions[i].append(t)
                total_rewards[i] += rewards[i]

            obs_list = next_obs_list
            global_state = next_global_state
            step += 1

            if all(dones):
                break

        self._episode_count += 1
        self._total_steps += step

        episode_info = {
            "episode": self._episode_count,
            "length": step,
            "total_rewards": total_rewards,
            "mean_reward": float(np.mean(total_rewards)),
            "infos": self.env.get_all_infos(),
        }
        return episode_info, agent_transitions


# ---------------------------------------------------------------------------
# MAPPO trainer
# ---------------------------------------------------------------------------

class MAPPOTrainer:
    """
    Full MAPPO training loop with CTDE.
    """

    def __init__(
        self,
        config: TrainingConfig,
        env: Optional[MultiAssetTradingEnv] = None,
        agents: Optional[List[MAPPOAgent]] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # Environment
        env_config = {
            "num_assets": config.num_assets,
            "num_agents": config.num_agents,
            "max_steps": config.max_steps_per_episode,
        }
        self.env = env or make_env(env_config)
        self.eval_env = make_env(env_config)

        obs_dim = self.env.obs_dim
        action_dim = self.env.action_dim
        state_dim = self.env.state_dim

        # Agents
        if agents is not None:
            self.agents = agents
        else:
            self.agents = [
                MAPPOAgent(
                    agent_id=i,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    state_dim=state_dim,
                    num_agents=config.num_agents,
                    lr_actor=config.lr_actor,
                    lr_critic=config.lr_critic,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    ppo_epochs=config.num_epochs,
                    mini_batch_size=config.mini_batch_size,
                    device=config.device,
                )
                for i in range(config.num_agents)
            ]

        self.rollout_buffer = MARolloutBuffer(
            num_agents=config.num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            max_size=config.rollout_length,
        )

        self.runner = EpisodeRunner(
            env=self.env,
            agents=self.agents,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # Curriculum
        self.curriculum = CurriculumScheduler(config.curriculum_stages) if config.use_curriculum else None

        # Self-play league
        self.league = SelfPlayLeague() if config.use_self_play else None

        # Metrics
        self._metrics: Dict[str, List[float]] = collections.defaultdict(list)
        self._total_steps = 0
        self._episodes = 0

        os.makedirs(config.save_dir, exist_ok=True)
        logger.info(f"MAPPOTrainer initialized: {config.num_agents} agents, {config.num_assets} assets")

    def train(self, total_timesteps: Optional[int] = None) -> Dict[str, List[float]]:
        """Main training loop."""
        total_ts = total_timesteps or self.config.total_timesteps
        logger.info(f"Starting training for {total_ts} timesteps")

        start_time = time.time()
        iteration = 0

        while self._total_steps < total_ts:
            # Collect rollout
            rollout_stats = self.runner.collect_rollout(
                self.rollout_buffer,
                num_steps=self.config.rollout_length,
                reset_env=(iteration == 0),
            )

            self._total_steps += self.config.rollout_length
            self._episodes = self.runner._episode_count

            # Compute advantages
            last_values = []
            for i, agent in enumerate(self.agents):
                global_state = self.env.get_state()
                lv = agent.get_value(global_state) if hasattr(agent, "get_value") else 0.0
                last_values.append(lv)

            adv_list, ret_list = self.rollout_buffer.compute_advantages_and_returns(
                last_values=last_values,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

            # Update agents
            all_metrics = []
            for i, agent in enumerate(self.agents):
                # Prepare batch
                obs_np = np.array(self.rollout_buffer.obs[i], dtype=np.float32)
                acts_np = np.array(self.rollout_buffer.actions[i], dtype=np.float32)
                old_lp_np = np.array(self.rollout_buffer.log_probs[i], dtype=np.float32)
                vals_np = np.array(self.rollout_buffer.values[i], dtype=np.float32)

                states_np = (
                    np.array(self.rollout_buffer.global_states, dtype=np.float32)
                    if self.rollout_buffer.global_states
                    else obs_np
                )

                batch = {
                    "obs": obs_np,
                    "actions": acts_np,
                    "log_probs": old_lp_np,
                    "advantages": adv_list[i],
                    "returns": ret_list[i],
                    "global_states": states_np,
                    "old_values": vals_np,
                }

                metrics = agent.update(batch)
                if metrics:
                    all_metrics.append(metrics)

            self.rollout_buffer.clear()

            # Log
            if iteration % self.config.log_interval == 0 and all_metrics:
                self._log_iteration(iteration, rollout_stats, all_metrics, start_time)

            # Save
            if iteration % self.config.save_interval == 0:
                self._save_checkpoint(iteration)

            # Evaluate
            if iteration % self.config.eval_interval == 0:
                eval_rewards = self._evaluate(n_episodes=3)
                self._metrics["eval_reward"].append(float(np.mean(eval_rewards)))

            # Curriculum
            if self.curriculum and rollout_stats.get("episode_reward") is not None:
                advanced = self.curriculum.record_episode(rollout_stats["episode_reward"])
                if advanced:
                    new_config = self.curriculum.get_stage_config()
                    logger.info(f"Curriculum stage config: {new_config}")

            # Self-play
            if self.league and iteration % self.config.self_play_interval == 0:
                agent_states = [
                    {
                        "actor": agent.actor.state_dict(),
                        "encoder": agent.encoder.state_dict(),
                    }
                    for agent in self.agents
                ]
                self.league.add_checkpoint(
                    agent_states, self._total_steps,
                    performance=self._metrics["eval_reward"][-1] if self._metrics["eval_reward"] else 0.0,
                )

            iteration += 1

        logger.info(f"Training complete: {self._total_steps} steps, {self._episodes} episodes")
        return dict(self._metrics)

    def _log_iteration(
        self,
        iteration: int,
        rollout_stats: Dict,
        all_metrics: List[Dict],
        start_time: float,
    ) -> None:
        elapsed = time.time() - start_time
        fps = self._total_steps / (elapsed + EPS)

        mean_actor_loss = float(np.mean([m.get("actor_loss", 0) for m in all_metrics]))
        mean_critic_loss = float(np.mean([m.get("critic_loss", 0) for m in all_metrics]))
        mean_entropy = float(np.mean([m.get("entropy", 0) for m in all_metrics]))

        ep_rew = rollout_stats.get("episode_reward", 0.0)

        self._metrics["actor_loss"].append(mean_actor_loss)
        self._metrics["critic_loss"].append(mean_critic_loss)
        self._metrics["entropy"].append(mean_entropy)
        self._metrics["episode_reward"].append(float(ep_rew))
        self._metrics["fps"].append(fps)

        logger.info(
            f"Iter {iteration} | Steps {self._total_steps} | "
            f"EP Rew {ep_rew:.3f} | "
            f"AL {mean_actor_loss:.4f} | CL {mean_critic_loss:.4f} | "
            f"Ent {mean_entropy:.4f} | FPS {fps:.0f}"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        save_dir = self.config.save_dir
        os.makedirs(save_dir, exist_ok=True)
        for agent in self.agents:
            path = os.path.join(save_dir, f"agent_{agent.agent_id}_iter{iteration}.pt")
            try:
                agent.save(path)
            except Exception as e:
                logger.warning(f"Failed to save agent {agent.agent_id}: {e}")

    def _evaluate(self, n_episodes: int = 5) -> List[float]:
        """Evaluate agents in eval environment."""
        episode_rewards = []
        for _ in range(n_episodes):
            self.eval_env.reset()
            obs_list = self.eval_env.get_all_observations()
            total_rewards = [0.0] * self.config.num_agents
            for step in range(self.config.max_steps_per_episode):
                actions = []
                for i, agent in enumerate(self.agents):
                    a, _, _ = agent.select_action(obs_list[i], deterministic=True)
                    actions.append(a)
                obs_list, rewards, terminated, truncated, _ = self.eval_env._marl_step(actions)
                for i in range(self.config.num_agents):
                    total_rewards[i] += rewards[i]
                if any(t or tr for t, tr in zip(terminated, truncated)):
                    break
            episode_rewards.append(float(np.mean(total_rewards)))
        return episode_rewards


# ---------------------------------------------------------------------------
# Training factory
# ---------------------------------------------------------------------------

def create_trainer(config: TrainingConfig) -> MAPPOTrainer:
    """Create trainer based on algorithm config."""
    if config.algorithm == "mappo":
        return MAPPOTrainer(config)
    elif config.algorithm in ("qmix", "coma", "mfg", "sac"):
        # MAPPO as fallback for now; can be extended
        logger.warning(f"Algorithm {config.algorithm} defaulting to MAPPO trainer structure")
        return MAPPOTrainer(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")


__all__ = [
    "TrainingConfig",
    "CurriculumScheduler",
    "SelfPlayLeague",
    "EpisodeRunner",
    "MAPPOTrainer",
    "create_trainer",
]
