"""
Multi-Agent BH Consensus Protocol (T4-3)
Byzantine Fault Tolerant weighted voting across specialized trading agents.

Agents:
  1. PhysicsAgent: BH physics (mass, curvature, geodesic)
  2. StatAgent: OU mean reversion, GARCH, Hurst
  3. MLAgent: XGBoost score
  4. RLAgent: Q-value from RL exit policy
  5. MacroAgent: correlation, event calendar, on-chain
  6. AdversarialAgent: always argues against (Devil's Advocate)

Consensus: weighted vote; trade executes when weighted sum > BH_FORM analog threshold.
Agent weights updated by rolling Sharpe contribution (T1-4 output).
"""
import logging
import math
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class AgentVote:
    agent_name: str
    conviction: float     # signed float: +1.0 = strong buy, -1.0 = strong sell, 0 = neutral
    confidence: float     # [0, 1] — how confident this agent is
    reason: str = ""      # human-readable explanation

@dataclass
class ConsensusConfig:
    entry_threshold: float = 0.35    # weighted vote must exceed this to enter
    exit_threshold: float = -0.20   # weighted vote below this triggers exit
    adversarial_weight: float = 0.25 # adversarial agent always gets this weight floor
    min_agents_voting: int = 3       # minimum agents with non-zero conviction required
    weight_update_alpha: float = 0.05  # EMA for rolling weight updates

class MultiAgentConsensus:
    """
    Collects votes from all agents and computes weighted consensus.

    Agent weights are initialized equal and updated based on rolling trade attribution.

    Usage:
        consensus = MultiAgentConsensus()

        votes = [
            AgentVote("physics", conviction=0.8, confidence=0.9, reason="BH mass=2.3"),
            AgentVote("stat", conviction=0.3, confidence=0.6, reason="Hurst=0.62"),
            AgentVote("ml", conviction=0.5, confidence=0.7, reason="XGB score=0.4"),
            AgentVote("rl", conviction=0.2, confidence=0.5, reason="Q_hold>Q_exit"),
            AgentVote("macro", conviction=0.4, confidence=0.6, reason="BTC lead"),
            AgentVote("adversarial", conviction=-0.6, confidence=0.8, reason="high curvature"),
        ]

        result = consensus.vote(votes)
        if result["enter"]:
            size_scale = result["size_scale"]
    """

    AGENT_NAMES = ["physics", "stat", "ml", "rl", "macro", "adversarial"]

    def __init__(self, cfg: ConsensusConfig = None):
        self.cfg = cfg or ConsensusConfig()
        # Equal initial weights, adversarial gets its floor
        self._weights: dict[str, float] = {
            "physics": 1.0, "stat": 0.8, "ml": 0.7,
            "rl": 0.6, "macro": 0.6, "adversarial": 0.5,
        }
        # Rolling Sharpe contribution per agent (for weight updates)
        self._agent_pnl: dict[str, deque] = {
            name: deque(maxlen=100) for name in self.AGENT_NAMES
        }
        self._last_votes: dict[str, AgentVote] = {}
        self._trade_count: int = 0

    def vote(self, votes: list[AgentVote]) -> dict:
        """
        Compute consensus from a list of agent votes.

        Returns:
          enter: bool
          exit: bool
          weighted_conviction: float — the consensus score
          size_scale: float — scales position size by conviction strength
          dominant_agent: str — agent with highest weighted contribution
          explanation: str — human-readable trade rationale
        """
        if not votes:
            return self._null_result()

        self._last_votes = {v.agent_name: v for v in votes}

        # Ensure adversarial agent has minimum weight
        adv_weight = max(
            self._weights.get("adversarial", 0.5),
            self.cfg.adversarial_weight
        )

        total_weight = 0.0
        weighted_sum = 0.0
        agent_contributions: dict[str, float] = {}
        active_agents = 0

        for vote in votes:
            w = self._weights.get(vote.agent_name, 0.5)
            if vote.agent_name == "adversarial":
                w = adv_weight

            # Scale weight by agent's confidence
            effective_weight = w * vote.confidence
            contribution = effective_weight * vote.conviction

            weighted_sum += contribution
            total_weight += effective_weight
            agent_contributions[vote.agent_name] = contribution

            if abs(vote.conviction) > 0.05:
                active_agents += 1

        if total_weight < 1e-12:
            return self._null_result()

        normalized_conviction = weighted_sum / total_weight

        # Entry/exit decisions
        enough_agents = active_agents >= self.cfg.min_agents_voting
        enter = normalized_conviction > self.cfg.entry_threshold and enough_agents
        exit_ = normalized_conviction < self.cfg.exit_threshold and enough_agents

        # Size scale: conviction strength → position scaling
        if enter:
            # 1.0 at threshold, scales up with conviction
            size_scale = 1.0 + max(0.0, (normalized_conviction - self.cfg.entry_threshold) / 0.3)
            size_scale = min(2.0, size_scale)
        else:
            size_scale = 0.0

        # Dominant agent
        dominant = max(agent_contributions, key=lambda k: abs(agent_contributions[k]), default="")

        # Explanation
        vote_strs = [f"{v.agent_name}={v.conviction:+.2f}({v.reason})" for v in votes]
        explanation = f"consensus={normalized_conviction:+.3f} [{', '.join(vote_strs)}]"

        if enter:
            log.info("Consensus ENTER: %s", explanation)
        elif exit_:
            log.info("Consensus EXIT: %s", explanation)

        return {
            "enter": enter,
            "exit": exit_,
            "weighted_conviction": normalized_conviction,
            "size_scale": size_scale,
            "dominant_agent": dominant,
            "agent_contributions": agent_contributions,
            "explanation": explanation,
        }

    def _null_result(self) -> dict:
        return {
            "enter": False, "exit": False, "weighted_conviction": 0.0,
            "size_scale": 0.0, "dominant_agent": "", "agent_contributions": {},
            "explanation": "no votes",
        }

    def update_agent_weight_from_pnl(self, agent_name: str, trade_pnl_pct: float):
        """
        Update agent weight based on trade outcome.
        Call this after each completed trade, attributing P&L to the dominant agent.

        Weights updated via rolling Sharpe: higher Sharpe → higher weight.
        """
        if agent_name not in self._agent_pnl:
            return
        self._agent_pnl[agent_name].append(trade_pnl_pct)

        # Compute rolling Sharpe for this agent
        returns = list(self._agent_pnl[agent_name])
        if len(returns) < 5:
            return

        mean_r = sum(returns) / len(returns)
        std_r = (sum((r - mean_r)**2 for r in returns) / len(returns)) ** 0.5
        sharpe = mean_r / (std_r + 1e-6) * (252 ** 0.5)  # annualized

        # Target weight: proportional to positive Sharpe, floor at 0.1
        target_weight = max(0.10, min(2.0, 0.5 + sharpe * 0.3))
        if agent_name == "adversarial":
            target_weight = max(self.cfg.adversarial_weight, target_weight)

        # EMA update
        current = self._weights.get(agent_name, 0.5)
        self._weights[agent_name] = (
            (1 - self.cfg.weight_update_alpha) * current +
            self.cfg.weight_update_alpha * target_weight
        )
