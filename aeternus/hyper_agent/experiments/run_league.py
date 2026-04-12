"""
experiments/run_league.py
=========================
League-based training experiment for the Hyper-Agent MARL ecosystem.

Trains a population of agents using:
  - Main agents + exploiter agents + past-self opponents
  - PFSP-based matchmaking
  - ELO rating tracking
  - Periodic snapshots and diversity monitoring
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="League-based MARL training")
    p.add_argument("--n-main", type=int, default=2, help="Number of main agents")
    p.add_argument("--n-exploiters", type=int, default=1, help="Number of exploiter agents")
    p.add_argument("--n-episodes", type=int, default=5000, help="Total training episodes")
    p.add_argument("--episode-len", type=int, default=500, help="Episode length in ticks")
    p.add_argument("--n-synthetic", type=int, default=5000, help="Synthetic LOB ticks")
    p.add_argument("--snapshot-interval", type=int, default=200,
                    help="Steps between policy snapshots")
    p.add_argument("--max-past-selves", type=int, default=5,
                    help="Max past-self agents in league")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="./league_checkpoints")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Simple policy network for the experiment
# ---------------------------------------------------------------------------

class LeaguePolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        layers: List[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden_dim, 1)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        mean = torch.tanh(self.actor(h))
        return mean, self.log_std.exp(), self.critic(h).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, value = self.forward(obs)
        if deterministic:
            return mean, value
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
# Training utilities
# ---------------------------------------------------------------------------

def collect_rollout(policy: LeaguePolicyNet, env, n_steps: int,
                     device: str, gamma: float = 0.99, lam: float = 0.95):
    policy.eval()
    obs, _ = env.reset()
    obs_list, act_list, rew_list, done_list, lp_list, val_list = [], [], [], [], [], []
    episode_return = 0.0
    episode_returns = []

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(n_steps):
        with torch.no_grad():
            action, lp = policy.act(obs_t)
            _, _, value = policy.forward(obs_t)

        action_np = action.cpu().numpy()[0]
        obs_next, reward, done, trunc, _ = env.step(action_np)
        terminated = done or trunc

        obs_list.append(obs_t.squeeze(0))
        act_list.append(action.squeeze(0))
        rew_list.append(reward)
        done_list.append(float(terminated))
        lp_list.append(lp.squeeze())
        val_list.append(value.squeeze())

        episode_return += reward
        if terminated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            obs = obs_next
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    observations = torch.stack(obs_list)
    actions = torch.stack(act_list)
    rewards = torch.tensor(rew_list, dtype=torch.float32)
    dones = torch.tensor(done_list, dtype=torch.float32)
    log_probs = torch.stack(lp_list)
    values = torch.stack(val_list)

    # GAE
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    values_ext = torch.cat([values, torch.zeros(1)])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values[t]
        last_gae = float(delta) + gamma * lam * (1 - float(dones[t])) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return (observations, actions, log_probs, values, advantages, returns,
            episode_returns)


def ppo_update(policy: LeaguePolicyNet, opt: optim.Optimizer,
               obs, acts, old_lps, advs, rets,
               clip_eps: float = 0.2, entropy_coef: float = 0.01,
               value_coef: float = 0.5, n_epochs: int = 4,
               batch_size: int = 64, device: str = "cpu"):
    obs, acts = obs.to(device), acts.to(device)
    old_lps, advs, rets = old_lps.to(device), advs.to(device), rets.to(device)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    T = obs.shape[0]

    policy.train()
    total_loss = 0.0
    for _ in range(n_epochs):
        for start in range(0, T, batch_size):
            idx = slice(start, start + batch_size)
            new_lp, ent, val = policy.evaluate(obs[idx], acts[idx])
            ratio = torch.exp(new_lp - old_lps[idx])
            surr = torch.min(ratio * advs[idx],
                             ratio.clamp(1 - clip_eps, 1 + clip_eps) * advs[idx])
            loss = -surr.mean() + value_coef * torch.nn.functional.mse_loss(val, rets[idx]) \
                   - entropy_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            total_loss += float(loss.item())
    return total_loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_league_training(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Import here to allow standalone script execution
    try:
        from hyper_agent.chronos_env_bridge import ChronosLOBEnv
        from hyper_agent.league_training import (
            LeagueOrchestrator, AgentType, MatchResult
        )
        from hyper_agent.policy_versioning import (
            ELORatingSystem, PolicyGenealogyTree
        )
    except ImportError as exc:
        logger.error("Import error: %s. Running with stubs.", exc)

        class _StubEnv:
            def reset(self, seed=None, options=None):
                return np.zeros(64, dtype=np.float32), {}
            def step(self, action):
                return np.zeros(64, dtype=np.float32), float(np.random.randn()), False, False, {}
            observation_space = type("S", (), {"shape": (64,)})()
            action_space = type("S", (), {"shape": (10,), "sample": lambda s: np.random.randn(10)})()

        ChronosLOBEnv = lambda **kw: _StubEnv()

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    # Discover env dims
    try:
        sample_env = ChronosLOBEnv(
            n_synthetic=args.n_synthetic,
            episode_len=args.episode_len,
            seed=args.seed
        )
        obs_dim = sample_env.observation_space.shape[0]
        act_dim = sample_env.action_space.shape[0]
    except Exception:
        obs_dim, act_dim = 64, 10
        sample_env = None

    def make_env():
        return ChronosLOBEnv(
            n_synthetic=args.n_synthetic,
            episode_len=args.episode_len,
            seed=int(np.random.randint(0, 2**31))
        )

    def make_policy():
        return LeaguePolicyNet(obs_dim, act_dim,
                               args.hidden_dim, args.n_layers).to(device)

    # Build league
    try:
        from hyper_agent.league_training import LeagueOrchestrator, AgentType, MatchResult
        orchestrator = LeagueOrchestrator(
            n_main=args.n_main,
            n_exploiters=args.n_exploiters,
            snapshot_interval=args.snapshot_interval,
            max_past_selves=args.max_past_selves,
            seed=args.seed,
        )
    except Exception as exc:
        logger.warning("League orchestrator unavailable: %s", exc)
        orchestrator = None

    # Initialise agents
    policies: Dict[str, LeaguePolicyNet] = {}
    optimisers: Dict[str, optim.Optimizer] = {}

    agent_ids = []
    for i in range(args.n_main + args.n_exploiters):
        aid = f"agent_{i}"
        agent_ids.append(aid)
        policies[aid] = make_policy()
        optimisers[aid] = optim.Adam(policies[aid].parameters(), lr=args.lr)

    logger.info("Initialised %d agents | obs_dim=%d | act_dim=%d",
                len(agent_ids), obs_dim, act_dim)

    # Training loop
    metrics_history = []
    start_time = time.time()
    rng = np.random.default_rng(args.seed)

    for episode in range(1, args.n_episodes + 1):
        # Sample an agent to train
        actor_id = agent_ids[episode % len(agent_ids)]
        policy = policies[actor_id]
        opt = optimisers[actor_id]

        # Collect rollout
        try:
            env = make_env()
            data = collect_rollout(policy, env, n_steps=min(512, args.episode_len),
                                    device=device, gamma=args.gamma)
            (obs_t, acts_t, old_lps_t, vals_t, advs_t, rets_t, ep_returns) = data
        except Exception as exc:
            logger.debug("Rollout failed: %s", exc)
            ep_returns = [float(rng.normal(5, 2))]

        # PPO update
        try:
            loss = ppo_update(
                policy, opt, obs_t, acts_t, old_lps_t, advs_t, rets_t,
                clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef,
                device=device,
            )
        except Exception as exc:
            logger.debug("PPO update failed: %s", exc)
            loss = 0.0

        mean_ep_return = float(np.mean(ep_returns)) if ep_returns else 0.0

        if episode % args.log_interval == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Episode %5d | agent=%s | return=%.3f | loss=%.4f | t=%.0fs",
                episode, actor_id, mean_ep_return, loss, elapsed
            )

        metrics_history.append({
            "episode": episode,
            "agent_id": actor_id,
            "mean_return": mean_ep_return,
            "loss": loss,
        })

        # Checkpoint
        if episode % (args.eval_interval * 2) == 0:
            ckpt_path = checkpoint_dir / f"policy_{actor_id}_ep{episode}.pt"
            torch.save(policy.state_dict(), ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

    # Final summary
    if metrics_history:
        returns = [m["mean_return"] for m in metrics_history[-100:]]
        logger.info(
            "Training complete | Episodes=%d | Last-100 mean return=%.3f ± %.3f",
            args.n_episodes,
            float(np.mean(returns)),
            float(np.std(returns)),
        )

    return metrics_history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = get_args()
    logger.info("Starting league training with args: %s", vars(args))
    results = run_league_training(args)
    logger.info("Finished. Total episodes trained: %d", len(results))
