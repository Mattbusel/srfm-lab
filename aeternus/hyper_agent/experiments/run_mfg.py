"""
run_mfg.py — Mean Field Game experiment for Hyper-Agent.

Runs MFG equilibrium computation with a population of mean field agents.
Tracks convergence toward Nash equilibrium via fixed-point iteration.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
import collections
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hyper_agent.environment import MultiAssetTradingEnv, make_env
from hyper_agent.agents.mean_field_agent import (
    MeanFieldAgent, PopulationMeanFieldTracker, MFGEquilibriumSolver
)
from hyper_agent.agents.base_agent import Transition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MFG experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_assets", type=int, default=2)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--episode_len", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mf_update_interval", type=int, default=10)
    parser.add_argument("--equilibrium_check_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="checkpoints/mfg")
    return parser.parse_args()


def setup_mfg_agents(args: argparse.Namespace, env: MultiAssetTradingEnv) -> List[MeanFieldAgent]:
    agents = [
        MeanFieldAgent(
            agent_id=i,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            num_agents=args.num_agents,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed + i,
        )
        for i in range(args.num_agents)
    ]
    logger.info(f"Created {len(agents)} MFG agents")
    return agents


def run_mfg_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 70)
    logger.info("Mean Field Game Experiment")
    logger.info(f"Agents: {args.num_agents}, Assets: {args.num_assets}")
    logger.info("=" * 70)

    env = MultiAssetTradingEnv(
        num_assets=args.num_assets,
        num_agents=args.num_agents,
        max_steps=args.episode_len,
        seed=args.seed,
    )

    agents = setup_mfg_agents(args, env)

    tracker = PopulationMeanFieldTracker(
        num_agents=args.num_agents,
        action_dim=env.action_dim,
        obs_dim=env.obs_dim,
    )

    solver = MFGEquilibriumSolver(
        agents=agents,
        tracker=tracker,
        convergence_tol=1e-3,
        max_iterations=200,
    )

    # Training metrics
    metrics: Dict[str, List[float]] = collections.defaultdict(list)
    total_steps = 0
    episode = 0
    convergence_history: List[float] = []
    converged = False

    logger.info("Starting MFG training...")
    start_time = time.time()

    while total_steps < args.total_steps:
        # Episode rollout
        env.reset()
        obs_list = env.get_all_observations()

        ep_rewards = [0.0] * args.num_agents
        ep_actions: List[List[np.ndarray]] = [[] for _ in range(args.num_agents)]
        ep_obs: List[List[np.ndarray]] = [[] for _ in range(args.num_agents)]

        for step in range(args.episode_len):
            # Get actions
            actions = []
            for i, agent in enumerate(agents):
                a, lp, _ = agent.select_action(obs_list[i])
                actions.append(a)
                ep_actions[i].append(a.copy())
                ep_obs[i].append(obs_list[i].copy())

            # Step
            next_obs_list, rewards, terminated, truncated, infos = env._marl_step(actions)
            dones = [t or tr for t, tr in zip(terminated, truncated)]

            # Store transitions
            for i, agent in enumerate(agents):
                t = Transition(
                    obs=obs_list[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_obs=next_obs_list[i],
                    done=dones[i],
                )
                agent.observe(t)
                ep_rewards[i] += rewards[i]

            # Update mean field periodically
            if step % args.mf_update_interval == 0:
                all_current_obs = [obs_list[i] for i in range(args.num_agents)]
                conv, delta = solver.step(all_current_obs, actions)
                convergence_history.append(delta)
                if conv and not converged:
                    converged = True
                    logger.info(f"MFG equilibrium converged at step {total_steps + step}!")

            obs_list = next_obs_list
            if all(dones):
                break

        # Update agents
        update_metrics = []
        for agent in agents:
            m = agent.update()
            if m:
                update_metrics.append(m)

        total_steps += args.episode_len
        episode += 1

        mean_reward = float(np.mean(ep_rewards))
        metrics["mean_reward"].append(mean_reward)

        if update_metrics:
            metrics["actor_loss"].append(
                float(np.mean([m.get("actor_loss", 0) for m in update_metrics]))
            )
            metrics["critic_loss"].append(
                float(np.mean([m.get("critic_loss", 0) for m in update_metrics]))
            )

        # Equilibrium gap
        if episode % args.equilibrium_check_interval == 0:
            gaps = [agent.get_equilibrium_gap() for agent in agents]
            valid_gaps = [g for g in gaps if not (isinstance(g, float) and g != g)]
            if valid_gaps:
                mean_gap = float(np.mean(valid_gaps))
                metrics["equilibrium_gap"].append(mean_gap)
                logger.info(
                    f"Episode {episode} | Steps {total_steps} | "
                    f"Reward {mean_reward:.4f} | Gap {mean_gap:.4f} | "
                    f"Converged: {converged}"
                )

    elapsed = time.time() - start_time
    logger.info(f"Training complete: {elapsed:.1f}s, {episode} episodes")

    # Check final equilibrium
    final_gaps = [agent.get_equilibrium_gap() for agent in agents]
    valid = [g for g in final_gaps if isinstance(g, float) and not (g != g)]
    final_gap = float(np.mean(valid)) if valid else float("nan")

    results = {
        "config": vars(args),
        "converged": converged,
        "final_equilibrium_gap": final_gap,
        "convergence_steps": len(convergence_history),
        "final_mean_reward": float(np.mean(metrics["mean_reward"][-20:])) if metrics["mean_reward"] else 0.0,
        "training_time": elapsed,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "mfg_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results: converged={converged}, eq_gap={final_gap:.4f}")
    return results


def main() -> None:
    args = parse_args()
    run_mfg_experiment(args)


if __name__ == "__main__":
    main()
