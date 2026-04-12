"""
policy_versioning.py
====================
Policy version management for the Hyper-Agent MARL ecosystem.

Provides:
  - PolicyCheckpoint: serialisable policy snapshot with metadata
  - PolicyRegistry: versioned storage with CRUD + search
  - PolicyGenealogyTree: parent-child relationships between policy versions
  - PolicyEvaluator: A/B testing framework between policy versions
  - ELORatingSystem: ELO-based competitive ranking for policies
  - LeagueTrainer: self-play + historical opponents training loop
  - BestResponseOracle: compute exact best responses to a fixed opponent
  - PolicyDistiller: knowledge distillation from large to small network
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import pickle
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy network primitives
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """Generic actor-critic policy network."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers: List[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = torch.tanh(self.actor_mean(h))
        log_std = self.actor_log_std.clamp(-5, 2)
        value = self.critic(h).squeeze(-1)
        return mean, log_std.exp(), value

    def act(self, obs: torch.Tensor,
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(obs)
        if deterministic:
            return mean, value
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob

    def evaluate(self, obs: torch.Tensor,
                 action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SmallPolicyNetwork(PolicyNetwork):
    """Compact student network for distillation."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__(obs_dim, act_dim, hidden_dim=64, n_layers=2)


# ---------------------------------------------------------------------------
# Checkpoint data structure
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PolicyCheckpoint:
    """Serialisable snapshot of a policy at a point in training."""
    policy_id: str
    version: int
    parent_id: Optional[str]
    agent_role: str
    obs_dim: int
    act_dim: int
    hidden_dim: int
    n_layers: int
    state_dict: Dict[str, Any]           # torch state dict
    optimizer_state: Optional[Dict]
    training_steps: int
    training_episodes: int
    metrics: Dict[str, float]            # mean_return, sharpe, win_rate, …
    elo_rating: float = 1000.0
    created_at: float = dataclasses.field(default_factory=time.time)
    tags: List[str] = dataclasses.field(default_factory=list)
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Hash of state_dict for identity comparison."""
        blob = pickle.dumps({k: v.cpu().numpy().tobytes()
                             if isinstance(v, torch.Tensor) else v
                             for k, v in self.state_dict.items()})
        return hashlib.md5(blob).hexdigest()[:12]

    def to_network(self, device: str = "cpu") -> PolicyNetwork:
        net = PolicyNetwork(self.obs_dim, self.act_dim,
                            self.hidden_dim, self.n_layers).to(device)
        state = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                 for k, v in self.state_dict.items()}
        net.load_state_dict(state)
        return net

    def summary(self) -> str:
        m = self.metrics
        return (f"PolicyCheckpoint(id={self.policy_id[:8]}, v={self.version}, "
                f"role={self.agent_role}, elo={self.elo_rating:.1f}, "
                f"ret={m.get('mean_return', 0):.3f}, steps={self.training_steps})")


def checkpoint_from_network(
    network: PolicyNetwork,
    policy_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    agent_role: str = "generic",
    training_steps: int = 0,
    training_episodes: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    optimizer: Optional[optim.Optimizer] = None,
    tags: Optional[List[str]] = None,
) -> PolicyCheckpoint:
    state_dict = {k: v.cpu().detach().numpy()
                  for k, v in network.state_dict().items()}
    opt_state = optimizer.state_dict() if optimizer is not None else None
    return PolicyCheckpoint(
        policy_id=policy_id or str(uuid.uuid4()),
        version=0,
        parent_id=parent_id,
        agent_role=agent_role,
        obs_dim=network.obs_dim,
        act_dim=network.act_dim,
        hidden_dim=network.hidden_dim,
        n_layers=network.n_layers,
        state_dict=state_dict,
        optimizer_state=opt_state,
        training_steps=training_steps,
        training_episodes=training_episodes,
        metrics=metrics or {},
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------

class PolicyRegistry:
    """
    Persistent policy registry backed by a directory of pickle files.
    Supports tagging, searching, and version incrementing.
    """

    def __init__(self, root_dir: Union[str, pathlib.Path] = "./policy_registry"):
        self.root_dir = pathlib.Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, List[PolicyCheckpoint]] = defaultdict(list)
        self._load_index()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, checkpoint: PolicyCheckpoint) -> str:
        """Persist checkpoint and update index."""
        existing = self._index.get(checkpoint.policy_id, [])
        checkpoint.version = len(existing)
        checkpoint.created_at = time.time()

        path = self._checkpoint_path(checkpoint)
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

        self._index[checkpoint.policy_id].append(checkpoint)
        logger.info("Saved %s", checkpoint.summary())
        return str(path)

    def load(self, policy_id: str,
             version: int = -1) -> Optional[PolicyCheckpoint]:
        """Load a specific version (default: latest)."""
        versions = self._index.get(policy_id, [])
        if not versions:
            # Try loading from disk
            self._load_index()
            versions = self._index.get(policy_id, [])
        if not versions:
            return None
        idx = version if version >= 0 else len(versions) - 1
        if idx >= len(versions):
            return None
        return versions[idx]

    def delete(self, policy_id: str, version: int = -1) -> bool:
        versions = self._index.get(policy_id, [])
        if not versions:
            return False
        idx = version if version >= 0 else len(versions) - 1
        if idx >= len(versions):
            return False
        ckpt = versions[idx]
        path = self._checkpoint_path(ckpt)
        if path.exists():
            path.unlink()
        del versions[idx]
        return True

    def list_all(self) -> List[PolicyCheckpoint]:
        result = []
        for versions in self._index.values():
            if versions:
                result.append(versions[-1])   # latest version only
        return sorted(result, key=lambda c: c.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self,
               role: Optional[str] = None,
               tags: Optional[List[str]] = None,
               min_elo: float = 0.0,
               min_return: float = -float("inf"),
               top_k: int = 10) -> List[PolicyCheckpoint]:
        candidates = self.list_all()
        if role:
            candidates = [c for c in candidates if c.agent_role == role]
        if tags:
            candidates = [c for c in candidates
                          if any(t in c.tags for t in tags)]
        candidates = [c for c in candidates if c.elo_rating >= min_elo]
        candidates = [c for c in candidates
                      if c.metrics.get("mean_return", -float("inf")) >= min_return]
        candidates.sort(key=lambda c: c.elo_rating, reverse=True)
        return candidates[:top_k]

    def best_policy(self, role: Optional[str] = None) -> Optional[PolicyCheckpoint]:
        results = self.search(role=role, top_k=1)
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _checkpoint_path(self, checkpoint: PolicyCheckpoint) -> pathlib.Path:
        role_dir = self.root_dir / checkpoint.agent_role
        role_dir.mkdir(parents=True, exist_ok=True)
        return role_dir / f"{checkpoint.policy_id}_v{checkpoint.version}.pkl"

    def _load_index(self) -> None:
        for pkl_path in self.root_dir.rglob("*.pkl"):
            try:
                with open(pkl_path, "rb") as f:
                    ckpt: PolicyCheckpoint = pickle.load(f)
                stored = self._index[ckpt.policy_id]
                if not any(c.version == ckpt.version for c in stored):
                    stored.append(ckpt)
                    stored.sort(key=lambda c: c.version)
            except Exception as exc:
                logger.debug("Failed to load %s: %s", pkl_path, exc)


# ---------------------------------------------------------------------------
# Genealogy tree
# ---------------------------------------------------------------------------

class PolicyGenealogyTree:
    """
    Tracks parent-child relationships between policy versions.
    Allows tracing lineage, finding best descendant, etc.
    """

    def __init__(self):
        self._nodes: Dict[str, PolicyCheckpoint] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, checkpoint: PolicyCheckpoint) -> None:
        self._nodes[checkpoint.policy_id] = checkpoint
        if checkpoint.parent_id and checkpoint.parent_id in self._nodes:
            self._children[checkpoint.parent_id].append(checkpoint.policy_id)

    def children(self, policy_id: str) -> List[PolicyCheckpoint]:
        return [self._nodes[c] for c in self._children.get(policy_id, [])
                if c in self._nodes]

    def ancestors(self, policy_id: str) -> List[PolicyCheckpoint]:
        result = []
        current = self._nodes.get(policy_id)
        while current and current.parent_id:
            parent = self._nodes.get(current.parent_id)
            if parent is None:
                break
            result.append(parent)
            current = parent
        return result

    def best_in_subtree(self, root_id: str,
                         metric: str = "elo_rating") -> Optional[PolicyCheckpoint]:
        """Find the best-performing node in a subtree."""
        subtree = self._collect_subtree(root_id)
        if not subtree:
            return None
        return max(subtree, key=lambda c: c.metrics.get(metric, c.elo_rating))

    def _collect_subtree(self, root_id: str) -> List[PolicyCheckpoint]:
        result = []
        queue = [root_id]
        while queue:
            current_id = queue.pop(0)
            if current_id in self._nodes:
                result.append(self._nodes[current_id])
                queue.extend(self._children.get(current_id, []))
        return result

    def visualise(self) -> str:
        """Return ASCII tree representation."""
        roots = [pid for pid in self._nodes if self._nodes[pid].parent_id is None]
        lines = []
        for root in roots:
            lines.extend(self._render_node(root, "", True))
        return "\n".join(lines)

    def _render_node(self, policy_id: str,
                      prefix: str, is_last: bool) -> List[str]:
        connector = "└─ " if is_last else "├─ "
        node = self._nodes[policy_id]
        line = prefix + connector + f"{policy_id[:8]} (elo={node.elo_rating:.0f})"
        lines = [line]
        children_ids = self._children.get(policy_id, [])
        for i, child_id in enumerate(children_ids):
            extension = "   " if is_last else "│  "
            lines.extend(self._render_node(
                child_id, prefix + extension, i == len(children_ids) - 1
            ))
        return lines


# ---------------------------------------------------------------------------
# A/B testing framework
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ABTestResult:
    policy_a_id: str
    policy_b_id: str
    n_trials: int
    a_wins: int
    b_wins: int
    draws: int
    a_mean_return: float
    b_mean_return: float
    p_value: float
    winner: Optional[str]
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)


class PolicyABTester:
    """
    Runs head-to-head A/B tests between two policy versions.
    Uses paired t-test for statistical significance.
    """

    def __init__(self, eval_env_fn: Callable,
                 n_trials: int = 100,
                 significance_level: float = 0.05):
        self.eval_env_fn = eval_env_fn
        self.n_trials = n_trials
        self.significance_level = significance_level

    def run_test(self, checkpoint_a: PolicyCheckpoint,
                 checkpoint_b: PolicyCheckpoint,
                 device: str = "cpu") -> ABTestResult:
        returns_a = self._evaluate_policy(checkpoint_a, device)
        returns_b = self._evaluate_policy(checkpoint_b, device)

        a_mean = float(np.mean(returns_a))
        b_mean = float(np.mean(returns_b))

        # Paired t-test approximation
        diffs = np.array(returns_a) - np.array(returns_b)
        t_stat = float(diffs.mean() / (diffs.std() + 1e-9) * math.sqrt(len(diffs)))
        # Approximate p-value using normal distribution
        from scipy import stats as scipy_stats
        try:
            p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(t_stat))))
        except Exception:
            p_value = 1.0 if abs(t_stat) < 2.0 else 0.05

        a_wins = int(np.sum(np.array(returns_a) > np.array(returns_b)))
        b_wins = int(np.sum(np.array(returns_b) > np.array(returns_a)))
        draws = self.n_trials - a_wins - b_wins

        if p_value < self.significance_level:
            winner = checkpoint_a.policy_id if a_mean > b_mean else checkpoint_b.policy_id
        else:
            winner = None

        return ABTestResult(
            policy_a_id=checkpoint_a.policy_id,
            policy_b_id=checkpoint_b.policy_id,
            n_trials=self.n_trials,
            a_wins=a_wins,
            b_wins=b_wins,
            draws=draws,
            a_mean_return=a_mean,
            b_mean_return=b_mean,
            p_value=p_value,
            winner=winner,
        )

    def _evaluate_policy(self, checkpoint: PolicyCheckpoint,
                          device: str) -> List[float]:
        network = checkpoint.to_network(device)
        network.eval()
        returns = []
        for trial in range(self.n_trials):
            env = self.eval_env_fn(seed=trial)
            obs, _ = env.reset(seed=trial)
            total_return = 0.0
            done = False
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            while not done:
                with torch.no_grad():
                    action, _ = network.act(obs_t, deterministic=True)
                action_np = action.cpu().numpy()[0]
                obs, reward, terminated, truncated, _ = env.step(action_np)
                total_return += reward
                done = terminated or truncated
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            returns.append(total_return)
        return returns


# ---------------------------------------------------------------------------
# ELO rating system
# ---------------------------------------------------------------------------

class ELORatingSystem:
    """
    ELO-based competitive rating for policies.
    Tracks all pairwise match outcomes and updates ratings after each match.
    """

    def __init__(self, initial_rating: float = 1000.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: Dict[str, float] = {}
        self._match_history: List[Dict] = []
        self._wins: Dict[str, int] = defaultdict(int)
        self._losses: Dict[str, int] = defaultdict(int)
        self._draws: Dict[str, int] = defaultdict(int)

    def register(self, policy_id: str,
                  initial: Optional[float] = None) -> None:
        if policy_id not in self._ratings:
            self._ratings[policy_id] = initial or self.initial_rating

    def get_rating(self, policy_id: str) -> float:
        return self._ratings.get(policy_id, self.initial_rating)

    def record_match(self, winner_id: str, loser_id: str,
                     draw: bool = False) -> Tuple[float, float]:
        self.register(winner_id)
        self.register(loser_id)

        r_w = self._ratings[winner_id]
        r_l = self._ratings[loser_id]

        exp_w = 1.0 / (1.0 + 10 ** ((r_l - r_w) / 400))
        exp_l = 1.0 - exp_w

        if draw:
            score_w, score_l = 0.5, 0.5
            self._draws[winner_id] += 1
            self._draws[loser_id] += 1
        else:
            score_w, score_l = 1.0, 0.0
            self._wins[winner_id] += 1
            self._losses[loser_id] += 1

        new_r_w = r_w + self.k_factor * (score_w - exp_w)
        new_r_l = r_l + self.k_factor * (score_l - exp_l)

        self._ratings[winner_id] = new_r_w
        self._ratings[loser_id] = new_r_l

        self._match_history.append({
            "winner": winner_id,
            "loser": loser_id,
            "draw": draw,
            "winner_rating_before": r_w,
            "loser_rating_before": r_l,
            "winner_rating_after": new_r_w,
            "loser_rating_after": new_r_l,
        })

        return new_r_w, new_r_l

    def leaderboard(self, top_k: int = 20) -> List[Dict]:
        sorted_ids = sorted(self._ratings.keys(),
                            key=lambda pid: self._ratings[pid], reverse=True)
        return [
            {
                "rank": i + 1,
                "policy_id": pid,
                "rating": self._ratings[pid],
                "wins": self._wins[pid],
                "losses": self._losses[pid],
                "draws": self._draws[pid],
            }
            for i, pid in enumerate(sorted_ids[:top_k])
        ]

    def expected_win_probability(self, a_id: str, b_id: str) -> float:
        r_a = self.get_rating(a_id)
        r_b = self.get_rating(b_id)
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))

    def rating_uncertainty(self, policy_id: str) -> float:
        """Higher uncertainty for policies with fewer matches."""
        n_games = (self._wins[policy_id] + self._losses[policy_id] +
                   self._draws[policy_id])
        return max(0.0, 1.0 - math.log1p(n_games) / 10.0)


# ---------------------------------------------------------------------------
# League training
# ---------------------------------------------------------------------------

class LeaguePolicy:
    """Wrapper around a checkpoint + ELO metadata for use in the league."""

    def __init__(self, checkpoint: PolicyCheckpoint,
                 policy_type: str = "main"):
        self.checkpoint = checkpoint
        self.policy_type = policy_type  # "main", "exploiter", "past_self"
        self.n_matches: int = 0
        self.is_frozen: bool = (policy_type == "past_self")

    @property
    def policy_id(self) -> str:
        return self.checkpoint.policy_id

    @property
    def elo(self) -> float:
        return self.checkpoint.elo_rating


class LeagueTrainer:
    """
    Implements the league-based training scheme:
      - Main agents: trained vs all opponents
      - Exploiter agents: trained specifically against main agents
      - Past-self agents: frozen snapshots of main agent history

    Uses Prioritised Fictitious Self-Play (PFSP) for opponent sampling.
    """

    def __init__(self,
                 main_policy_fn: Callable[[], PolicyNetwork],
                 n_main: int = 2,
                 n_exploiters: int = 2,
                 n_past_snapshots: int = 10,
                 snapshot_interval: int = 500,
                 pfsp_temperature: float = 1.0,
                 elo_system: Optional[ELORatingSystem] = None):
        self.main_policy_fn = main_policy_fn
        self.n_main = n_main
        self.n_exploiters = n_exploiters
        self.n_past_snapshots = n_past_snapshots
        self.snapshot_interval = snapshot_interval
        self.pfsp_temperature = pfsp_temperature
        self.elo = elo_system or ELORatingSystem()
        self._rng = np.random.default_rng()

        self._main_agents: List[LeaguePolicy] = []
        self._exploiters: List[LeaguePolicy] = []
        self._past_selves: List[LeaguePolicy] = []

        self._step_count: int = 0
        self._match_results: List[Dict] = []

    def initialise(self) -> None:
        for i in range(self.n_main):
            net = self.main_policy_fn()
            pid = f"main_{i}_{uuid.uuid4().hex[:6]}"
            ckpt = checkpoint_from_network(net, policy_id=pid,
                                            agent_role="main",
                                            tags=["main"])
            lp = LeaguePolicy(ckpt, "main")
            self._main_agents.append(lp)
            self.elo.register(pid, 1200.0)

        for i in range(self.n_exploiters):
            net = self.main_policy_fn()
            pid = f"exploiter_{i}_{uuid.uuid4().hex[:6]}"
            ckpt = checkpoint_from_network(net, policy_id=pid,
                                            agent_role="exploiter",
                                            tags=["exploiter"])
            lp = LeaguePolicy(ckpt, "exploiter")
            self._exploiters.append(lp)
            self.elo.register(pid, 1000.0)

    def sample_opponent(self, actor: LeaguePolicy) -> LeaguePolicy:
        """PFSP: sample opponent weighted by win probability of actor."""
        all_opponents = (
            self._main_agents + self._exploiters + self._past_selves
        )
        all_opponents = [op for op in all_opponents if op.policy_id != actor.policy_id]
        if not all_opponents:
            return actor

        # PFSP weighting: prefer opponents with win prob near 0.5
        weights = []
        for op in all_opponents:
            p_win = self.elo.expected_win_probability(actor.policy_id, op.policy_id)
            weight = math.exp(-self.pfsp_temperature * (p_win - 0.5) ** 2)
            weights.append(weight)

        weights_arr = np.array(weights)
        probs = weights_arr / weights_arr.sum()
        idx = int(self._rng.choice(len(all_opponents), p=probs))
        return all_opponents[idx]

    def record_match_outcome(self, actor_id: str, opponent_id: str,
                              actor_won: bool) -> None:
        winner = actor_id if actor_won else opponent_id
        loser = opponent_id if actor_won else actor_id
        self.elo.record_match(winner, loser)
        self._match_results.append({
            "actor": actor_id,
            "opponent": opponent_id,
            "actor_won": actor_won,
            "step": self._step_count,
        })

    def maybe_snapshot(self, main_agent_idx: int,
                        network: PolicyNetwork) -> Optional[LeaguePolicy]:
        """Snapshot main agent if snapshot interval reached."""
        self._step_count += 1
        if self._step_count % self.snapshot_interval != 0:
            return None

        main_lp = self._main_agents[main_agent_idx]
        pid = f"past_self_{main_lp.policy_id[:8]}_{self._step_count}"
        ckpt = checkpoint_from_network(network, policy_id=pid,
                                        parent_id=main_lp.policy_id,
                                        agent_role="past_self",
                                        training_steps=self._step_count,
                                        tags=["past_self"])
        ckpt.elo_rating = self.elo.get_rating(main_lp.policy_id)
        snapshot_lp = LeaguePolicy(ckpt, "past_self")

        if len(self._past_selves) >= self.n_past_snapshots:
            self._past_selves.pop(0)
        self._past_selves.append(snapshot_lp)
        self.elo.register(pid, ckpt.elo_rating)

        logger.info("Snapshot saved: %s (elo=%.1f)", pid, ckpt.elo_rating)
        return snapshot_lp

    def diversity_metric(self) -> float:
        """Average pairwise ELO difference across league (higher = more diverse)."""
        all_pols = self._main_agents + self._exploiters + self._past_selves
        if len(all_pols) < 2:
            return 0.0
        elos = [self.elo.get_rating(lp.policy_id) for lp in all_pols]
        diffs = []
        for i in range(len(elos)):
            for j in range(i + 1, len(elos)):
                diffs.append(abs(elos[i] - elos[j]))
        return float(np.mean(diffs)) if diffs else 0.0

    @property
    def leaderboard(self) -> List[Dict]:
        return self.elo.leaderboard()

    def all_policies(self) -> List[LeaguePolicy]:
        return self._main_agents + self._exploiters + self._past_selves


# ---------------------------------------------------------------------------
# Best-response oracle
# ---------------------------------------------------------------------------

class BestResponseOracle:
    """
    Computes the approximate best-response policy against a fixed opponent.
    Uses gradient-based training against a frozen opponent.
    """

    def __init__(self,
                 env_fn: Callable,
                 n_training_steps: int = 10_000,
                 lr: float = 3e-4,
                 device: str = "cpu"):
        self.env_fn = env_fn
        self.n_training_steps = n_training_steps
        self.lr = lr
        self.device = device

    def compute(self, opponent_checkpoint: PolicyCheckpoint,
                obs_dim: int, act_dim: int) -> PolicyCheckpoint:
        """Train a best-response policy against `opponent_checkpoint`."""
        opponent = opponent_checkpoint.to_network(self.device)
        opponent.eval()
        for p in opponent.parameters():
            p.requires_grad_(False)

        br_policy = PolicyNetwork(obs_dim, act_dim).to(self.device)
        optimiser = optim.Adam(br_policy.parameters(), lr=self.lr)

        env = self.env_fn()
        total_returns = deque(maxlen=100)
        step = 0

        while step < self.n_training_steps:
            obs, _ = env.reset()
            episode_data = []
            total_return = 0.0
            done = False

            while not done and len(episode_data) < 2000:
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
                action, log_prob = br_policy.act(obs_t)
                action_np = action.cpu().numpy()[0]
                obs_next, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                episode_data.append((obs_t, action, log_prob, reward))
                obs = obs_next
                total_return += reward
                step += 1

            # Simple REINFORCE update
            G = 0.0
            returns = []
            for _, _, _, r in reversed(episode_data):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for i, (obs_t, action, log_prob, _) in enumerate(episode_data):
                lp, ent, _ = br_policy.evaluate(obs_t, action)
                loss = loss + (-lp * returns_t[i] - 0.01 * ent)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(br_policy.parameters(), 0.5)
            optimiser.step()
            total_returns.append(total_return)

        mean_return = float(np.mean(total_returns)) if total_returns else 0.0
        br_id = f"br_{opponent_checkpoint.policy_id[:8]}_{uuid.uuid4().hex[:6]}"
        return checkpoint_from_network(
            br_policy,
            policy_id=br_id,
            parent_id=opponent_checkpoint.policy_id,
            agent_role="best_response",
            training_steps=self.n_training_steps,
            metrics={"mean_return": mean_return},
            tags=["best_response"],
        )


# ---------------------------------------------------------------------------
# Policy distiller
# ---------------------------------------------------------------------------

class PolicyDistiller:
    """
    Knowledge distillation: train a small student network to mimic
    the action distribution of a large teacher network.
    """

    def __init__(self,
                 env_fn: Callable,
                 n_distil_steps: int = 50_000,
                 lr: float = 1e-3,
                 temperature: float = 2.0,
                 device: str = "cpu"):
        self.env_fn = env_fn
        self.n_distil_steps = n_distil_steps
        self.lr = lr
        self.temperature = temperature
        self.device = device

    def distil(self, teacher_checkpoint: PolicyCheckpoint) -> PolicyCheckpoint:
        """Distil teacher into a SmallPolicyNetwork student."""
        teacher = teacher_checkpoint.to_network(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        student = SmallPolicyNetwork(
            teacher_checkpoint.obs_dim,
            teacher_checkpoint.act_dim,
        ).to(self.device)
        optimiser = optim.Adam(student.parameters(), lr=self.lr)

        env = self.env_fn()
        obs_buffer: List[torch.Tensor] = []
        step = 0

        # Collect observations from teacher rollout
        obs, _ = env.reset()
        done = False
        while step < self.n_distil_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            obs_buffer.append(obs_t)
            with torch.no_grad():
                action, _ = teacher.act(obs_t, deterministic=False)
            obs, _, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            if done:
                obs, _ = env.reset()
                done = False
            step += 1

            # Train in mini-batches
            if len(obs_buffer) >= 256:
                batch = torch.cat(obs_buffer[:256], dim=0)
                obs_buffer = obs_buffer[128:]

                with torch.no_grad():
                    t_mean, t_std, _ = teacher.forward(batch)

                s_mean, s_std, _ = student.forward(batch)

                # KL divergence between Gaussian action distributions
                t_dist = torch.distributions.Normal(t_mean, t_std * self.temperature)
                s_dist = torch.distributions.Normal(s_mean, s_std)
                kl_loss = torch.distributions.kl_divergence(t_dist, s_dist).mean()

                optimiser.zero_grad()
                kl_loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimiser.step()

        student_id = f"student_{teacher_checkpoint.policy_id[:8]}_{uuid.uuid4().hex[:6]}"
        ckpt = checkpoint_from_network(
            student,
            policy_id=student_id,
            parent_id=teacher_checkpoint.policy_id,
            agent_role="distilled",
            training_steps=step,
            metrics={"param_count": student.parameter_count()},
            tags=["distilled", "small"],
        )
        logger.info("Distilled teacher (%.1fK params) -> student (%.1fK params)",
                    teacher.parameter_count() / 1000,
                    student.parameter_count() / 1000)
        return ckpt


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== policy_versioning.py smoke test ===")

    # Create network
    net = PolicyNetwork(obs_dim=64, act_dim=10)
    print(f"Network params: {net.parameter_count():,}")

    # Checkpoint
    ckpt = checkpoint_from_network(
        net,
        agent_role="market_maker",
        training_steps=1000,
        metrics={"mean_return": 25.3, "sharpe": 1.2},
        tags=["test"],
    )
    print(f"Checkpoint: {ckpt.summary()}")
    print(f"Fingerprint: {ckpt.fingerprint}")

    # Restore from checkpoint
    restored = ckpt.to_network()
    obs = torch.randn(1, 64)
    action, lp = restored.act(obs)
    print(f"Action shape: {action.shape}, log_prob shape: {lp.shape}")

    # ELO system
    elo = ELORatingSystem()
    elo.register("pol_a", 1000.0)
    elo.register("pol_b", 1000.0)
    for i in range(20):
        winner = "pol_a" if i % 3 != 0 else "pol_b"
        loser = "pol_b" if winner == "pol_a" else "pol_a"
        elo.record_match(winner, loser)
    board = elo.leaderboard()
    print(f"ELO leaderboard: {board}")

    # Genealogy tree
    tree = PolicyGenealogyTree()
    ckpt2 = checkpoint_from_network(
        net, policy_id="child_1", parent_id=ckpt.policy_id,
        agent_role="market_maker"
    )
    tree.add_node(ckpt)
    tree.add_node(ckpt2)
    print(f"Tree:\n{tree.visualise()}")

    print("\nAll smoke tests passed.")
