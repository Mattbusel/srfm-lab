"""
experiments/run_curriculum.py
==============================
Curriculum learning experiment for the Hyper-Agent MARL ecosystem.

Trains a single agent through a sequence of progressively harder
market scenarios using the automated curriculum learning system.
Integrates:
  - CurriculumTrainingManager
  - Population-based training for hyperparameter search
  - Domain randomisation
  - ZPD-based scenario selection
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curriculum MARL training")
    p.add_argument("--n-episodes", type=int, default=3000)
    p.add_argument("--episode-len", type=int, default=500)
    p.add_argument("--n-synthetic", type=int, default=5000)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--pbt-population", type=int, default=4)
    p.add_argument("--domain-randomise", action="store_true", default=True)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="./curriculum_checkpoints")
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--success-threshold", type=float, default=5.0,
                    help="Episode return above this = success")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class CurriculumPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        layers: List[nn.Module] = [nn.Linear(obs_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.LayerNorm(hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple:
        h = self.backbone(obs)
        mean = torch.tanh(self.actor(h))
        return mean, self.log_std.exp(), self.critic(h).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, val = self.forward(obs)
        if deterministic:
            return mean.clamp(-1, 1), val
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        lp = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), lp

    def evaluate(self, obs, action):
        mean, std, val = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        lp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        return lp, ent, val


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_episode(policy: CurriculumPolicy, env,
                     max_steps: int, device: str,
                     gamma: float = 0.99, lam: float = 0.95,
                     ) -> Dict[str, Any]:
    policy.eval()
    obs, _ = env.reset()
    done = False
    total_return = 0.0
    steps = 0

    obs_list, act_list, rew_list, done_list, lp_list, val_list = [], [], [], [], [], []

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    while not done and steps < max_steps:
        with torch.no_grad():
            action, lp = policy.act(obs_t)
            _, _, val = policy.forward(obs_t)

        action_np = action.cpu().numpy()[0]
        obs_next, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        obs_list.append(obs_t.squeeze(0))
        act_list.append(action.squeeze(0))
        rew_list.append(reward)
        done_list.append(float(done))
        lp_list.append(lp.squeeze())
        val_list.append(val.squeeze())

        total_return += reward
        steps += 1
        obs = obs_next
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    T = len(rew_list)
    if T == 0:
        return {"total_return": 0.0, "steps": 0}

    observations = torch.stack(obs_list)
    actions = torch.stack(act_list)
    rewards = torch.tensor(rew_list, dtype=torch.float32)
    dones = torch.tensor(done_list, dtype=torch.float32)
    log_probs = torch.stack(lp_list)
    values = torch.stack(val_list)

    # GAE
    advantages = torch.zeros(T)
    last_gae = 0.0
    vals_ext = torch.cat([values, torch.zeros(1)])
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * vals_ext[t + 1] * (1 - dones[t]) - values[t]
        last_gae = float(delta) + gamma * lam * (1 - float(dones[t])) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return {
        "observations": observations,
        "actions": actions,
        "log_probs": log_probs,
        "values": values,
        "advantages": advantages,
        "returns": returns,
        "total_return": total_return,
        "steps": steps,
    }


def ppo_update(policy: CurriculumPolicy, opt: optim.Optimizer,
               data: Dict, args: argparse.Namespace, device: str) -> float:
    if "observations" not in data:
        return 0.0

    obs = data["observations"].to(device)
    acts = data["actions"].to(device)
    old_lps = data["log_probs"].to(device).detach()
    advs = data["advantages"].to(device)
    rets = data["returns"].to(device)

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    T = obs.shape[0]

    policy.train()
    total_loss = 0.0
    n_batches = 0
    batch_size = min(64, T)

    for _ in range(4):  # n_epochs
        idx = torch.randperm(T)
        for start in range(0, T, batch_size):
            b_idx = idx[start:start + batch_size]
            new_lp, ent, val = policy.evaluate(obs[b_idx], acts[b_idx])
            ratio = torch.exp(new_lp - old_lps[b_idx])
            clip_adv = torch.min(
                ratio * advs[b_idx],
                ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * advs[b_idx]
            )
            policy_loss = -clip_adv.mean()
            value_loss = torch.nn.functional.mse_loss(val, rets[b_idx])
            entropy_loss = -ent.mean()

            loss = policy_loss + 0.5 * value_loss + args.entropy_coef * entropy_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(policy: CurriculumPolicy, make_env_fn,
                     n_episodes: int = 10, device: str = "cpu") -> Dict[str, float]:
    policy.eval()
    returns = []
    for ep in range(n_episodes):
        env = make_env_fn(seed=ep)
        obs, _ = env.reset()
        total_return = 0.0
        done = False
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        for _ in range(2000):
            with torch.no_grad():
                action, _ = policy.act(obs_t, deterministic=True)
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action_np)
            total_return += reward
            done = terminated or truncated
            if done:
                break
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        returns.append(total_return)

    arr = np.array(returns)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "sharpe": float(arr.mean() / (arr.std() + 1e-9)),
    }


# ---------------------------------------------------------------------------
# Main curriculum training loop
# ---------------------------------------------------------------------------

def run_curriculum_training(args: argparse.Namespace) -> List[Dict]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Try importing modules
    try:
        from hyper_agent.chronos_env_bridge import ChronosLOBEnv
        from hyper_agent.curriculum_learning import (
            CurriculumTrainingManager,
            build_default_curriculum,
            DomainRandomiser,
        )

        def make_env(seed: int = 0, episode_len: int = None, scenario: str = None):
            return ChronosLOBEnv(
                n_synthetic=args.n_synthetic,
                episode_len=episode_len or args.episode_len,
                scenario=scenario,
                seed=seed,
            )

        sample_env = make_env(seed=0)
        obs_dim = sample_env.observation_space.shape[0]
        act_dim = sample_env.action_space.shape[0]

        tasks = build_default_curriculum()
        curriculum_mgr = CurriculumTrainingManager(
            agent_ids=["agent_0"],
            tasks=tasks,
            pbt_population_size=args.pbt_population,
            domain_randomise=args.domain_randomise,
            seed=args.seed,
        )
        domain_rng = DomainRandomiser(seed=args.seed)
        have_curriculum = True
    except Exception as exc:
        logger.warning("Curriculum import failed (%s), using plain training.", exc)
        obs_dim, act_dim = 64, 10
        tasks = []
        curriculum_mgr = None
        domain_rng = None
        have_curriculum = False

        def make_env(seed: int = 0, **kwargs):
            class _Env:
                observation_space = type("S", (), {"shape": (obs_dim,)})()
                action_space = type("S", (), {
                    "shape": (act_dim,),
                    "sample": lambda s: np.random.randn(act_dim).astype(np.float32)
                })()
                def reset(self, seed=None, options=None):
                    return np.zeros(obs_dim, np.float32), {}
                def step(self, action):
                    return np.zeros(obs_dim, np.float32), float(np.random.randn()), False, False, {}
            return _Env()

    # Build policy
    policy = CurriculumPolicy(obs_dim, act_dim, args.hidden_dim, args.n_layers).to(device)
    optimiser = optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.n_episodes)

    logger.info("Curriculum training | obs_dim=%d act_dim=%d device=%s", obs_dim, act_dim, device)
    if tasks:
        logger.info("Tasks: %s", [t.task_id for t in tasks])

    # Training state
    metrics_history: List[Dict] = []
    recent_returns: deque = deque(maxlen=100)
    start_time = time.time()

    for episode in range(1, args.n_episodes + 1):
        # Get task from curriculum
        if have_curriculum and curriculum_mgr is not None:
            task, domain = curriculum_mgr.get_next_task("agent_0")
            env_kwargs: Dict[str, Any] = {
                "seed": int(np.random.randint(0, 2**31)),
                "episode_len": min(task.env_kwargs.get("episode_len", args.episode_len),
                                   args.episode_len),
                "scenario": task.scenario_type if task.scenario_type != "mixed" else None,
            }
        else:
            task = None
            env_kwargs = {"seed": int(np.random.randint(0, 2**31))}

        # Collect episode
        try:
            env = make_env(**env_kwargs)
            data = collect_episode(
                policy, env,
                max_steps=args.rollout_steps,
                device=device,
                gamma=args.gamma,
                lam=args.gae_lambda,
            )
        except Exception as exc:
            logger.debug("Episode collection failed: %s", exc)
            data = {"total_return": 0.0, "steps": 0}

        # PPO update
        try:
            loss = ppo_update(policy, optimiser, data, args, device)
        except Exception as exc:
            logger.debug("PPO update failed: %s", exc)
            loss = 0.0

        scheduler.step()

        ep_return = float(data.get("total_return", 0.0))
        recent_returns.append(ep_return)
        success = ep_return > args.success_threshold

        # Report to curriculum manager
        if have_curriculum and curriculum_mgr is not None and task is not None:
            curriculum_info = curriculum_mgr.on_episode_end(
                "agent_0", task.task_id, ep_return, success
            )
        else:
            curriculum_info = {}

        metrics = {
            "episode": episode,
            "return": ep_return,
            "loss": loss,
            "success": success,
            "task": task.task_id if task else "default",
            "lr": float(optimiser.param_groups[0]["lr"]),
        }
        metrics_history.append(metrics)

        if episode % args.log_interval == 0:
            recent_mean = float(np.mean(recent_returns))
            logger.info(
                "Ep %5d | task=%-25s | ret=%.3f | mean100=%.3f | loss=%.4f",
                episode,
                metrics["task"],
                ep_return,
                recent_mean,
                loss,
            )
            if curriculum_mgr is not None:
                prog = curriculum_mgr.global_progress()
                logger.info("  Curriculum competency: %.3f", prog)

        # Checkpoint
        if episode % 500 == 0:
            ckpt_path = checkpoint_dir / f"policy_ep{episode}.pt"
            torch.save({
                "episode": episode,
                "policy_state": policy.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "metrics": metrics,
            }, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

    # Final evaluation
    logger.info("Running final evaluation...")
    try:
        eval_results = evaluate_policy(policy, lambda seed=0: make_env(seed=seed),
                                        n_episodes=20, device=device)
        logger.info("Final eval: %s", eval_results)
    except Exception as exc:
        logger.warning("Eval failed: %s", exc)
        eval_results = {}

    # Summary
    all_returns = [m["return"] for m in metrics_history]
    logger.info(
        "Training complete | Episodes=%d | Mean return=%.3f | Elapsed=%.0fs",
        args.n_episodes,
        float(np.mean(all_returns[-100:])),
        time.time() - start_time,
    )

    return metrics_history


# ---------------------------------------------------------------------------
# Curriculum analysis utilities
# ---------------------------------------------------------------------------

def analyse_curriculum_progress(metrics_history: List[Dict]) -> Dict[str, Any]:
    """Analyse how performance evolved over curriculum tasks."""
    if not metrics_history:
        return {}

    returns = np.array([m["return"] for m in metrics_history])
    tasks = [m["task"] for m in metrics_history]

    # Returns per task
    task_returns: Dict[str, List[float]] = {}
    for m in metrics_history:
        task_returns.setdefault(m["task"], []).append(m["return"])

    task_stats = {
        task: {
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "n": len(r),
        }
        for task, r in task_returns.items()
    }

    # Progression (rolling mean returns)
    window = min(100, len(returns))
    rolling_mean = np.convolve(returns, np.ones(window) / window, mode="valid")

    return {
        "total_episodes": len(metrics_history),
        "overall_mean_return": float(returns.mean()),
        "final_100_mean_return": float(returns[-100:].mean()),
        "task_stats": task_stats,
        "progression_slope": float(np.polyfit(np.arange(len(rolling_mean)),
                                               rolling_mean, 1)[0]),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = get_args()
    logger.info("Starting curriculum training with config: %s", vars(args))

    metrics = run_curriculum_training(args)
    analysis = analyse_curriculum_progress(metrics)

    logger.info("=" * 60)
    logger.info("CURRICULUM TRAINING SUMMARY")
    logger.info("=" * 60)
    for key, val in analysis.items():
        if key != "task_stats":
            logger.info("  %s: %s", key, val)
    if "task_stats" in analysis:
        logger.info("Task performance breakdown:")
        for task_name, stats in analysis["task_stats"].items():
            logger.info("  %-30s mean=%.3f std=%.3f n=%d",
                        task_name, stats["mean"], stats["std"], stats["n"])
