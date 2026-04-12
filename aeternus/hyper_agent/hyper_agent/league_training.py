"""
league_training.py
==================
League-based training system for the Hyper-Agent MARL ecosystem.

Implements:
  - Main agents (trained against all opponents)
  - Exploiter agents (specialised against main agents)
  - Past-self agents (frozen historical snapshots)
  - Matchmaking (PFSP + ELO-weighted)
  - PFSP (Prioritised Fictitious Self-Play)
  - Agent selection probabilities
  - Diversity metrics
  - League management and monitoring
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import logging
import math
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent roles
# ---------------------------------------------------------------------------

class AgentType(enum.Enum):
    MAIN = "main"
    EXPLOITER = "exploiter"
    PAST_SELF = "past_self"
    EVALUATOR = "evaluator"


@dataclasses.dataclass
class LeagueAgent:
    """A single agent in the league."""
    agent_id: str
    agent_type: AgentType
    policy_id: str
    elo_rating: float = 1000.0
    n_matches: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_draws: int = 0
    is_frozen: bool = False
    created_at: float = dataclasses.field(default_factory=time.time)
    parent_id: Optional[str] = None
    generation: int = 0
    role_tags: List[str] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        total = self.n_matches
        if total == 0:
            return 0.5
        return self.n_wins / total

    @property
    def match_history_str(self) -> str:
        return f"W{self.n_wins}/L{self.n_losses}/D{self.n_draws}"

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "policy_id": self.policy_id,
            "elo_rating": self.elo_rating,
            "n_matches": self.n_matches,
            "win_rate": self.win_rate,
            "generation": self.generation,
        }


# ---------------------------------------------------------------------------
# ELO engine (standalone, efficient)
# ---------------------------------------------------------------------------

class ELOEngine:
    def __init__(self, k_factor: float = 32.0, initial: float = 1000.0):
        self.k = k_factor
        self.initial = initial
        self._ratings: Dict[str, float] = {}

    def register(self, agent_id: str, rating: Optional[float] = None) -> None:
        if agent_id not in self._ratings:
            self._ratings[agent_id] = rating or self.initial

    def expected_win(self, a: str, b: str) -> float:
        ra = self._ratings.get(a, self.initial)
        rb = self._ratings.get(b, self.initial)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def update(self, winner: str, loser: str, draw: bool = False) -> Tuple[float, float]:
        self.register(winner)
        self.register(loser)
        exp_w = self.expected_win(winner, loser)
        exp_l = 1 - exp_w
        s_w, s_l = (0.5, 0.5) if draw else (1.0, 0.0)
        new_w = self._ratings[winner] + self.k * (s_w - exp_w)
        new_l = self._ratings[loser] + self.k * (s_l - exp_l)
        self._ratings[winner] = new_w
        self._ratings[loser] = new_l
        return new_w, new_l

    def get(self, agent_id: str) -> float:
        return self._ratings.get(agent_id, self.initial)

    def leaderboard(self, top_k: int = 20) -> List[Tuple[str, float]]:
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# PFSP matchmaking
# ---------------------------------------------------------------------------

class PFSPMatchmaker:
    """
    Prioritised Fictitious Self-Play matchmaking.
    Samples opponent based on how close their win probability is to 0.5.
    """

    def __init__(self, temperature: float = 1.0, seed: int = 0):
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)

    def sample_opponent(self,
                         actor: LeagueAgent,
                         candidates: List[LeagueAgent],
                         elo: ELOEngine) -> Optional[LeagueAgent]:
        if not candidates:
            return None
        candidates = [c for c in candidates if c.agent_id != actor.agent_id]
        if not candidates:
            return None

        weights = []
        for cand in candidates:
            p_win = elo.expected_win(actor.agent_id, cand.agent_id)
            w = math.exp(-self.temperature * (p_win - 0.5) ** 2)
            weights.append(w)

        weights_arr = np.array(weights)
        probs = weights_arr / weights_arr.sum()
        idx = int(self._rng.choice(len(candidates), p=probs))
        return candidates[idx]

    def sample_batch(self,
                      actor: LeagueAgent,
                      candidates: List[LeagueAgent],
                      elo: ELOEngine,
                      n: int = 4) -> List[LeagueAgent]:
        if not candidates:
            return []
        candidates_filtered = [c for c in candidates if c.agent_id != actor.agent_id]
        if not candidates_filtered:
            return []

        weights = []
        for cand in candidates_filtered:
            p_win = elo.expected_win(actor.agent_id, cand.agent_id)
            w = math.exp(-self.temperature * (p_win - 0.5) ** 2)
            weights.append(w)

        weights_arr = np.array(weights)
        probs = weights_arr / weights_arr.sum()
        n_sample = min(n, len(candidates_filtered))
        idxs = self._rng.choice(len(candidates_filtered), size=n_sample,
                                  replace=False, p=probs)
        return [candidates_filtered[i] for i in idxs]


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MatchResult:
    actor_id: str
    opponent_id: str
    actor_type: AgentType
    opponent_type: AgentType
    actor_return: float
    opponent_return: float
    n_steps: int
    scenario: str = "normal"
    timestamp: float = dataclasses.field(default_factory=time.time)

    @property
    def actor_won(self) -> bool:
        return self.actor_return > self.opponent_return

    @property
    def is_draw(self) -> bool:
        return abs(self.actor_return - self.opponent_return) < 0.01


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

class DiversityCalculator:
    """
    Measures diversity of the league via behavioural fingerprinting.
    Uses policy outputs on a fixed set of canonical observations as fingerprints.
    """

    def __init__(self, n_canonical_obs: int = 100,
                 obs_dim: int = 64, device: str = "cpu"):
        self.device = device
        self._rng = np.random.default_rng(42)
        self._canonical_obs = torch.tensor(
            self._rng.standard_normal((n_canonical_obs, obs_dim)),
            dtype=torch.float32, device=device
        )
        self._fingerprints: Dict[str, np.ndarray] = {}

    def compute_fingerprint(self, policy: nn.Module, agent_id: str) -> np.ndarray:
        policy.eval()
        with torch.no_grad():
            actions, _ = policy.act(self._canonical_obs, deterministic=True)
        fp = actions.cpu().numpy().flatten()
        self._fingerprints[agent_id] = fp
        return fp

    def pairwise_distance(self, id_a: str, id_b: str) -> float:
        fp_a = self._fingerprints.get(id_a)
        fp_b = self._fingerprints.get(id_b)
        if fp_a is None or fp_b is None:
            return 0.0
        return float(np.linalg.norm(fp_a - fp_b))

    def mean_pairwise_diversity(self) -> float:
        ids = list(self._fingerprints.keys())
        if len(ids) < 2:
            return 0.0
        dists = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dists.append(self.pairwise_distance(ids[i], ids[j]))
        return float(np.mean(dists)) if dists else 0.0

    def nearest_neighbour_diversity(self) -> float:
        """Average distance to nearest neighbour (higher = more diverse)."""
        ids = list(self._fingerprints.keys())
        if len(ids) < 2:
            return 0.0
        nn_dists = []
        for i in range(len(ids)):
            dists = [self.pairwise_distance(ids[i], ids[j])
                     for j in range(len(ids)) if j != i]
            nn_dists.append(min(dists))
        return float(np.mean(nn_dists))


# ---------------------------------------------------------------------------
# League snapshot manager
# ---------------------------------------------------------------------------

class LeagueSnapshotManager:
    """
    Manages historical policy snapshots in the league.
    Decides when to snapshot and handles snapshot pruning.
    """

    def __init__(self,
                 max_snapshots: int = 10,
                 snapshot_interval_steps: int = 500,
                 min_elo_for_snapshot: float = 900.0):
        self.max_snapshots = max_snapshots
        self.snapshot_interval = snapshot_interval_steps
        self.min_elo = min_elo_for_snapshot
        self._step_since_last_snapshot: Dict[str, int] = defaultdict(int)
        self._snapshots: List[LeagueAgent] = []

    def should_snapshot(self, agent: LeagueAgent, step: int) -> bool:
        steps = self._step_since_last_snapshot[agent.agent_id] + 1
        self._step_since_last_snapshot[agent.agent_id] = steps
        return (steps >= self.snapshot_interval and
                agent.elo_rating >= self.min_elo)

    def add_snapshot(self, snapshot: LeagueAgent) -> None:
        self._step_since_last_snapshot[snapshot.parent_id or ""] = 0
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_snapshots:
            # Prune lowest-ELO snapshot
            self._snapshots.sort(key=lambda a: a.elo_rating, reverse=True)
            self._snapshots = self._snapshots[:self.max_snapshots]

    @property
    def snapshots(self) -> List[LeagueAgent]:
        return list(self._snapshots)


# ---------------------------------------------------------------------------
# League training orchestrator
# ---------------------------------------------------------------------------

class LeagueOrchestrator:
    """
    Full league training orchestrator.
    Manages the lifecycle of all agents, matchmaking, and outcome recording.
    """

    def __init__(self,
                 n_main: int = 2,
                 n_exploiters: int = 2,
                 pfsp_temperature: float = 1.0,
                 snapshot_interval: int = 500,
                 max_past_selves: int = 10,
                 diversity_threshold: float = 50.0,
                 elo_k_factor: float = 32.0,
                 seed: int = 0):
        self.n_main = n_main
        self.n_exploiters = n_exploiters
        self.pfsp_temperature = pfsp_temperature
        self.diversity_threshold = diversity_threshold
        self._rng = np.random.default_rng(seed)

        self.elo = ELOEngine(k_factor=elo_k_factor)
        self.matchmaker = PFSPMatchmaker(pfsp_temperature, seed)
        self.snapshot_mgr = LeagueSnapshotManager(
            max_snapshots=max_past_selves,
            snapshot_interval_steps=snapshot_interval,
        )
        self.diversity_calc = DiversityCalculator()

        self._main_agents: List[LeagueAgent] = []
        self._exploiters: List[LeagueAgent] = []
        self._match_history: List[MatchResult] = []
        self._step: int = 0

    def register_agent(self, agent_type: AgentType,
                         policy_id: str,
                         initial_elo: float = 1000.0,
                         parent_id: Optional[str] = None,
                         generation: int = 0) -> LeagueAgent:
        agent_id = f"{agent_type.value}_{uuid.uuid4().hex[:8]}"
        agent = LeagueAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            policy_id=policy_id,
            elo_rating=initial_elo,
            parent_id=parent_id,
            generation=generation,
        )
        self.elo.register(agent_id, initial_elo)

        if agent_type == AgentType.MAIN:
            self._main_agents.append(agent)
        elif agent_type == AgentType.EXPLOITER:
            self._exploiters.append(agent)

        logger.info("Registered %s agent: %s (elo=%.0f)", agent_type.value, agent_id, initial_elo)
        return agent

    def get_opponent(self, actor: LeagueAgent) -> Optional[LeagueAgent]:
        """Select opponent using PFSP."""
        all_opponents = self.all_agents
        return self.matchmaker.sample_opponent(actor, all_opponents, self.elo)

    def get_opponents_batch(self, actor: LeagueAgent, n: int = 4) -> List[LeagueAgent]:
        return self.matchmaker.sample_batch(actor, self.all_agents, self.elo, n)

    def record_match(self, result: MatchResult) -> None:
        """Record match outcome and update ELO ratings."""
        self._match_history.append(result)
        self._step += 1

        # Update ELO
        if result.is_draw:
            winner_id, loser_id = result.actor_id, result.opponent_id
            r_w, r_l = self.elo.update(winner_id, loser_id, draw=True)
        elif result.actor_won:
            r_w, r_l = self.elo.update(result.actor_id, result.opponent_id)
        else:
            r_w, r_l = self.elo.update(result.opponent_id, result.actor_id)

        # Update agent structs
        for agent in self.all_agents:
            if agent.agent_id == result.actor_id:
                agent.elo_rating = self.elo.get(agent.agent_id)
                agent.n_matches += 1
                if result.actor_won:
                    agent.n_wins += 1
                elif result.is_draw:
                    agent.n_draws += 1
                else:
                    agent.n_losses += 1
            elif agent.agent_id == result.opponent_id:
                agent.elo_rating = self.elo.get(agent.agent_id)
                agent.n_matches += 1
                if not result.actor_won and not result.is_draw:
                    agent.n_wins += 1
                elif result.is_draw:
                    agent.n_draws += 1
                else:
                    agent.n_losses += 1

    def maybe_snapshot_main(self, main_agent: LeagueAgent,
                              policy: nn.Module) -> Optional[LeagueAgent]:
        if not self.snapshot_mgr.should_snapshot(main_agent, self._step):
            return None

        snapshot_id = f"past_{main_agent.agent_id[:8]}_{self._step}"
        snapshot = LeagueAgent(
            agent_id=snapshot_id,
            agent_type=AgentType.PAST_SELF,
            policy_id=main_agent.policy_id + f"_snap_{self._step}",
            elo_rating=main_agent.elo_rating,
            is_frozen=True,
            parent_id=main_agent.agent_id,
            generation=main_agent.generation,
        )
        self.elo.register(snapshot_id, snapshot.elo_rating)
        self.snapshot_mgr.add_snapshot(snapshot)
        logger.info("Snapshot: %s (elo=%.0f)", snapshot_id, snapshot.elo_rating)
        return snapshot

    def update_fingerprint(self, agent: LeagueAgent, policy: nn.Module) -> None:
        self.diversity_calc.compute_fingerprint(policy, agent.agent_id)

    @property
    def all_agents(self) -> List[LeagueAgent]:
        return (self._main_agents + self._exploiters +
                self.snapshot_mgr.snapshots)

    @property
    def active_agents(self) -> List[LeagueAgent]:
        return [a for a in self.all_agents if not a.is_frozen]

    def leaderboard(self, top_k: int = 20) -> List[Dict[str, Any]]:
        agents = sorted(self.all_agents, key=lambda a: a.elo_rating, reverse=True)
        return [a.to_dict() for a in agents[:top_k]]

    def diversity_report(self) -> Dict[str, float]:
        return {
            "mean_pairwise_diversity": self.diversity_calc.mean_pairwise_diversity(),
            "nearest_neighbour_diversity": self.diversity_calc.nearest_neighbour_diversity(),
            "n_agents": len(self.all_agents),
            "elo_spread": self._elo_spread(),
        }

    def _elo_spread(self) -> float:
        elos = [a.elo_rating for a in self.all_agents]
        if len(elos) < 2:
            return 0.0
        return float(max(elos) - min(elos))

    def match_statistics(self) -> Dict[str, Any]:
        if not self._match_history:
            return {}
        returns = [m.actor_return for m in self._match_history]
        return {
            "total_matches": len(self._match_history),
            "mean_actor_return": float(np.mean(returns)),
            "std_actor_return": float(np.std(returns)),
            "win_rate_main": self._win_rate_for_type(AgentType.MAIN),
            "win_rate_exploiters": self._win_rate_for_type(AgentType.EXPLOITER),
        }

    def _win_rate_for_type(self, agent_type: AgentType) -> float:
        matches = [m for m in self._match_history
                   if m.actor_type == agent_type]
        if not matches:
            return 0.5
        return float(np.mean([m.actor_won for m in matches]))

    def full_summary(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "leaderboard": self.leaderboard(5),
            "diversity": self.diversity_report(),
            "match_stats": self.match_statistics(),
        }


# ---------------------------------------------------------------------------
# Training loop adapter
# ---------------------------------------------------------------------------

class LeagueTrainingLoop:
    """
    High-level training loop that integrates LeagueOrchestrator
    with actual policy training.
    """

    def __init__(self,
                 orchestrator: LeagueOrchestrator,
                 make_env_fn: Callable,
                 policy_factory: Callable,
                 n_steps_per_match: int = 2000,
                 n_eval_episodes: int = 5,
                 device: str = "cpu"):
        self.orchestrator = orchestrator
        self.make_env_fn = make_env_fn
        self.policy_factory = policy_factory
        self.n_steps_per_match = n_steps_per_match
        self.n_eval_episodes = n_eval_episodes
        self.device = device
        self._policies: Dict[str, nn.Module] = {}
        self._optimisers: Dict[str, optim.Optimizer] = {}

    def register_policies(self, agents: List[LeagueAgent]) -> None:
        for agent in agents:
            if agent.policy_id not in self._policies:
                pol = self.policy_factory().to(self.device)
                self._policies[agent.policy_id] = pol
                if not agent.is_frozen:
                    self._optimisers[agent.policy_id] = optim.Adam(
                        pol.parameters(), lr=3e-4
                    )

    def run_match(self, actor: LeagueAgent,
                   opponent: LeagueAgent) -> MatchResult:
        """Simulate a match between two agents."""
        actor_policy = self._policies.get(actor.policy_id)
        opp_policy = self._policies.get(opponent.policy_id)

        if actor_policy is None or opp_policy is None:
            return MatchResult(
                actor_id=actor.agent_id,
                opponent_id=opponent.agent_id,
                actor_type=actor.agent_type,
                opponent_type=opponent.agent_type,
                actor_return=0.0,
                opponent_return=0.0,
                n_steps=0,
            )

        env = self.make_env_fn()
        obs, _ = env.reset()
        total_return = 0.0

        actor_policy.eval()
        opp_policy.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        for step in range(self.n_steps_per_match):
            with torch.no_grad():
                action, _ = actor_policy.act(obs_t, deterministic=False)
            action_np = action.cpu().numpy()[0]
            obs, reward, done, trunc, _ = env.step(action_np)
            total_return += reward
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if done or trunc:
                break

        return MatchResult(
            actor_id=actor.agent_id,
            opponent_id=opponent.agent_id,
            actor_type=actor.agent_type,
            opponent_type=opponent.agent_type,
            actor_return=total_return,
            opponent_return=-total_return * 0.5,  # simplified
            n_steps=step + 1,
        )

    def train_step(self, actor: LeagueAgent,
                    replay_buffer: Optional[Any] = None) -> Dict[str, float]:
        """One gradient update for the actor policy."""
        policy = self._policies.get(actor.policy_id)
        opt = self._optimisers.get(actor.policy_id)
        if policy is None or opt is None or actor.is_frozen:
            return {}

        # Simplified REINFORCE gradient step
        policy.train()
        env = self.make_env_fn()
        obs, _ = env.reset()
        trajectory = []
        total_return = 0.0

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        for _ in range(min(500, self.n_steps_per_match)):
            action, log_prob = policy.act(obs_t)
            action_np = action.cpu().numpy()[0]
            obs, reward, done, trunc, _ = env.step(action_np)
            trajectory.append((log_prob, reward))
            total_return += reward
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if done or trunc:
                break

        # Compute returns
        G = 0.0
        returns = []
        for _, r in reversed(trajectory):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        policy_loss = torch.tensor(0.0, device=self.device)
        for i, (log_prob, _) in enumerate(trajectory):
            policy_loss = policy_loss + (-log_prob * returns_t[i])

        opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "episode_return": total_return,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== league_training.py smoke test ===")

    orch = LeagueOrchestrator(n_main=2, n_exploiters=1, seed=0)

    # Register agents
    main0 = orch.register_agent(AgentType.MAIN, "policy_main_0", 1100.0)
    main1 = orch.register_agent(AgentType.MAIN, "policy_main_1", 950.0)
    exp0 = orch.register_agent(AgentType.EXPLOITER, "policy_exp_0", 1000.0)

    print(f"All agents: {[a.agent_id for a in orch.all_agents]}")

    # Simulate matches
    rng = np.random.default_rng(42)
    for ep in range(50):
        actor = rng.choice(orch.active_agents)
        opponent = orch.get_opponent(actor)
        if opponent is None:
            continue
        actor_return = float(rng.normal(10, 5))
        opp_return = float(rng.normal(8, 5))
        result = MatchResult(
            actor_id=actor.agent_id,
            opponent_id=opponent.agent_id,
            actor_type=actor.agent_type,
            opponent_type=opponent.agent_type,
            actor_return=actor_return,
            opponent_return=opp_return,
            n_steps=500,
        )
        orch.record_match(result)

    print("\nLeaderboard:")
    for entry in orch.leaderboard():
        print(f"  {entry['agent_id']:30s} elo={entry['elo_rating']:.1f} "
              f"wins={entry.get('n_wins', 'N/A')}")

    print("\nDiversity report:", orch.diversity_report())
    print("Match stats:", orch.match_statistics())

    # Test PFSP
    pfsp = PFSPMatchmaker(temperature=1.0, seed=0)
    elo = ELOEngine()
    for agent in orch.all_agents:
        elo.register(agent.agent_id, agent.elo_rating)
    opp = pfsp.sample_opponent(main0, orch.all_agents, elo)
    print(f"\nPFSP selected opponent: {opp.agent_id if opp else None}")

    print("\nAll smoke tests passed.")
