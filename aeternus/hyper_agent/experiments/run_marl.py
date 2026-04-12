"""
run_marl.py — Full MARL experiment runner for Hyper-Agent.

Usage:
    python experiments/run_marl.py [--config CONFIG] [--seed SEED] [--device DEVICE]

Runs a full multi-agent reinforcement learning experiment using MAPPO with
centralized training and decentralized execution (CTDE).
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import logging
import argparse
import collections
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hyper_agent.environment import MultiAssetTradingEnv, make_env
from hyper_agent.agents.mappo_agent import MAPPOAgent, MAPPOPopulation
from hyper_agent.training import TrainingConfig, MAPPOTrainer, create_trainer
from hyper_agent.population import AgentPopulation, EvolutionaryDynamics
from hyper_agent.emergence import EmergenceAnalyzer
from hyper_agent.replay_buffer import MultiAgentReplayBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MARL experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_assets", type=int, default=4)
    parser.add_argument("--num_agents", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    parser.add_argument("--rollout_length", type=int, default=200)
    parser.add_argument("--lr_actor", type=float, default=5e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--use_curriculum", action="store_true", default=False)
    parser.add_argument("--use_self_play", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="checkpoints/marl")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--algorithm", type=str, default="mappo")
    parser.add_argument("--share_parameters", action="store_true", default=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--critic_hidden_dim", type=int, default=512)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def setup_environment(args: argparse.Namespace) -> MultiAssetTradingEnv:
    """Create and configure the trading environment."""
    env_config = {
        "num_assets": args.num_assets,
        "num_agents": args.num_agents,
        "max_steps": args.max_steps_per_episode,
        "seed": args.seed,
        "enable_circuit_breaker": True,
        "enable_flash_crash": True,
    }
    env = MultiAssetTradingEnv(**env_config)
    logger.info(
        f"Environment created: {args.num_assets} assets, {args.num_agents} agents, "
        f"obs_dim={env.obs_dim}, action_dim={env.action_dim}, state_dim={env.state_dim}"
    )
    return env


def setup_agents(
    args: argparse.Namespace,
    env: MultiAssetTradingEnv,
) -> List[MAPPOAgent]:
    """Create MAPPO agents."""
    agents = [
        MAPPOAgent(
            agent_id=i,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            state_dim=env.state_dim,
            num_agents=args.num_agents,
            hidden_dim=args.hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ppo_epochs=args.num_epochs,
            mini_batch_size=args.mini_batch_size,
            device=args.device,
            seed=args.seed + i,
        )
        for i in range(args.num_agents)
    ]

    if args.share_parameters and args.num_agents > 1:
        # Share encoder and actor parameters
        ref = agents[0]
        for agent in agents[1:]:
            agent.encoder.load_state_dict(ref.encoder.state_dict())
            agent.actor.load_state_dict(ref.actor.state_dict())
            agent.centralized_critic.load_state_dict(ref.centralized_critic.state_dict())
        logger.info("Parameter sharing enabled across agents")

    logger.info(f"Created {len(agents)} MAPPO agents")
    return agents


# ---------------------------------------------------------------------------
# Training with emergence tracking
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full MARL experiment."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 70)
    logger.info("Hyper-Agent MARL Experiment")
    logger.info(f"Config: {vars(args)}")
    logger.info("=" * 70)

    # Setup
    env = setup_environment(args)
    agents = setup_agents(args, env)

    # Emergence analyzer
    emergence = EmergenceAnalyzer(
        num_assets=args.num_assets,
        num_agents=args.num_agents,
        action_dim=env.action_dim,
    )

    # Training config
    config = TrainingConfig(
        num_assets=args.num_assets,
        num_agents=args.num_agents,
        max_steps_per_episode=args.max_steps_per_episode,
        total_timesteps=args.total_timesteps,
        rollout_length=args.rollout_length,
        num_epochs=args.num_epochs,
        mini_batch_size=args.mini_batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        algorithm=args.algorithm,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        use_curriculum=args.use_curriculum,
        use_self_play=args.use_self_play,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
    )

    # Trainer
    trainer = MAPPOTrainer(config, env=env, agents=agents)

    # Run training
    start_time = time.time()
    training_metrics = trainer.train(total_timesteps=args.total_timesteps)
    elapsed = time.time() - start_time

    logger.info(f"Training completed in {elapsed:.1f}s")
    logger.info(f"Total steps: {trainer._total_steps}")
    logger.info(f"Total episodes: {trainer._episodes}")

    # Final evaluation
    final_rewards = trainer._evaluate(n_episodes=10)
    logger.info(f"Final evaluation: mean={np.mean(final_rewards):.4f}, std={np.std(final_rewards):.4f}")

    # Emergence report
    emergence_report = emergence.full_report()
    logger.info(f"Market health score: {emergence.get_market_health_score():.3f}")

    # Save results
    results = {
        "config": vars(args),
        "training_metrics": {k: v[-10:] for k, v in training_metrics.items()},
        "final_eval_mean": float(np.mean(final_rewards)),
        "final_eval_std": float(np.std(final_rewards)),
        "total_steps": trainer._total_steps,
        "total_episodes": trainer._episodes,
        "training_time": elapsed,
        "market_health": emergence.get_market_health_score(),
    }

    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Per-agent performance analysis
# ---------------------------------------------------------------------------

def analyze_agent_performance(
    agents: List[MAPPOAgent],
    env: MultiAssetTradingEnv,
    n_episodes: int = 20,
) -> Dict[str, Any]:
    """Detailed per-agent performance analysis."""
    per_agent_stats: Dict[int, Dict] = {i: collections.defaultdict(list) for i in range(len(agents))}

    for ep in range(n_episodes):
        env.reset()
        obs_list = env.get_all_observations()

        for step in range(env.max_steps):
            actions = []
            for i, agent in enumerate(agents):
                a, _, _ = agent.select_action(obs_list[i], deterministic=True)
                actions.append(a)

            obs_list, rewards, terminated, truncated, infos = env._marl_step(actions)

            for i in range(len(agents)):
                per_agent_stats[i]["reward"].append(rewards[i])
                per_agent_stats[i]["equity"].append(infos[i]["equity"])

            if any(t or tr for t, tr in zip(terminated, truncated)):
                break

    result = {}
    for i, stats in per_agent_stats.items():
        rew = np.array(stats["reward"])
        eq = np.array(stats["equity"])
        result[f"agent_{i}"] = {
            "mean_reward": float(np.mean(rew)),
            "total_reward": float(np.sum(rew)),
            "sharpe": float(np.mean(rew) / (np.std(rew) + 1e-8) * np.sqrt(252)),
            "final_equity": float(np.mean(eq[-1:])) if len(eq) > 0 else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    results = run_experiment(args)

    logger.info("\n=== Final Results ===")
    logger.info(f"Mean eval reward: {results['final_eval_mean']:.4f}")
    logger.info(f"Market health: {results['market_health']:.3f}")
    logger.info(f"Training time: {results['training_time']:.1f}s")


if __name__ == "__main__":
    main()
