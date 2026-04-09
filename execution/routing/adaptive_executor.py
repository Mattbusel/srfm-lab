"""
Adaptive Execution Engine: regime-aware strategy selection + TCA feedback loop.

Implements three Gemma-proposed execution improvements:
1. Regime-aware strategy selector: market state -> optimal execution algo
2. TCA feedback loop: post-trade slippage feeds back into routing decisions
3. Dark pool learning: adjust fill rate expectations from realized performance

Integrates with:
  - execution/routing/smart_router.py
  - execution/order_management/ (TWAP, VWAP engines)
  - execution/tca/ (transaction cost analysis)
"""

from __future__ import annotations
import math
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Market Regime for Execution
# ---------------------------------------------------------------------------

class ExecutionRegime:
    """Three regimes relevant to execution quality."""
    STABLE = "stable"           # low vol, tight spreads, good liquidity
    VOLATILE = "volatile"       # high vol, moderate spreads
    LIQUIDITY_CRUNCH = "crunch" # wide spreads, thin books


@dataclass
class MarketConditions:
    """Current market conditions for a single instrument."""
    symbol: str
    spread_bps: float = 5.0
    daily_vol: float = 0.15
    adv: float = 1e6           # average daily volume in notional
    book_depth_ratio: float = 1.0  # current depth vs average (1.0 = normal)
    recent_trade_intensity: float = 1.0  # trades per minute vs average


def classify_execution_regime(conditions: MarketConditions) -> str:
    """Classify market conditions into execution regime."""
    if conditions.spread_bps > 30 or conditions.book_depth_ratio < 0.3:
        return ExecutionRegime.LIQUIDITY_CRUNCH
    elif conditions.daily_vol > 0.03 or conditions.spread_bps > 15:
        return ExecutionRegime.VOLATILE
    else:
        return ExecutionRegime.STABLE


# ---------------------------------------------------------------------------
# Execution Strategy Definitions
# ---------------------------------------------------------------------------

@dataclass
class ExecutionStrategy:
    """Recommended execution approach."""
    algorithm: str       # market / limit / twap / vwap / is / iceberg
    venue: str           # lit / dark / mixed
    urgency: float       # 0-1 (0=patient, 1=urgent)
    participation_limit: float  # max % of volume
    time_horizon_bars: int
    expected_cost_bps: float
    reason: str


# Strategy lookup: (regime, urgency_level, size_level) -> strategy
STRATEGY_TABLE = {
    # Stable market
    (ExecutionRegime.STABLE, "low", "small"):    ("limit", "lit", 0.2, 0.02, 1),
    (ExecutionRegime.STABLE, "low", "medium"):   ("twap", "mixed", 0.3, 0.03, 3),
    (ExecutionRegime.STABLE, "low", "large"):    ("vwap", "mixed", 0.4, 0.05, 6),
    (ExecutionRegime.STABLE, "high", "small"):   ("market", "lit", 0.9, 0.05, 1),
    (ExecutionRegime.STABLE, "high", "medium"):  ("is", "lit", 0.8, 0.05, 2),
    (ExecutionRegime.STABLE, "high", "large"):   ("twap", "lit", 0.7, 0.08, 3),

    # Volatile market
    (ExecutionRegime.VOLATILE, "low", "small"):  ("limit", "dark", 0.2, 0.01, 2),
    (ExecutionRegime.VOLATILE, "low", "medium"): ("twap", "dark", 0.3, 0.02, 5),
    (ExecutionRegime.VOLATILE, "low", "large"):  ("vwap", "mixed", 0.4, 0.03, 8),
    (ExecutionRegime.VOLATILE, "high", "small"): ("market", "lit", 0.8, 0.10, 1),
    (ExecutionRegime.VOLATILE, "high", "medium"):("twap", "lit", 0.7, 0.05, 3),
    (ExecutionRegime.VOLATILE, "high", "large"): ("iceberg", "lit", 0.6, 0.05, 5),

    # Liquidity crunch
    (ExecutionRegime.LIQUIDITY_CRUNCH, "low", "small"):  ("limit", "dark", 0.1, 0.01, 5),
    (ExecutionRegime.LIQUIDITY_CRUNCH, "low", "medium"): ("twap", "dark", 0.2, 0.02, 10),
    (ExecutionRegime.LIQUIDITY_CRUNCH, "low", "large"):  ("vwap", "dark", 0.3, 0.02, 15),
    (ExecutionRegime.LIQUIDITY_CRUNCH, "high", "small"): ("limit", "lit", 0.5, 0.05, 2),
    (ExecutionRegime.LIQUIDITY_CRUNCH, "high", "medium"):("iceberg", "mixed", 0.5, 0.03, 5),
    (ExecutionRegime.LIQUIDITY_CRUNCH, "high", "large"): ("twap", "mixed", 0.4, 0.03, 10),
}


class AdaptiveStrategySelector:
    """
    Select execution strategy based on market conditions, order size, and urgency.
    Replaces fixed-threshold logic with regime-aware multi-dimensional lookup.
    """

    def __init__(self):
        self._tca_adjustments: Dict[Tuple[str, str], float] = {}  # (algo, venue) -> cost multiplier

    def select(
        self,
        conditions: MarketConditions,
        order_notional: float,
        urgency: float = 0.5,
    ) -> ExecutionStrategy:
        regime = classify_execution_regime(conditions)

        # Size classification
        participation = order_notional / max(conditions.adv, 1e-10)
        if participation < 0.01:
            size_level = "small"
        elif participation < 0.05:
            size_level = "medium"
        else:
            size_level = "large"

        # Urgency classification
        urgency_level = "high" if urgency > 0.6 else "low"

        # Lookup
        key = (regime, urgency_level, size_level)
        algo, venue, urg, part_limit, horizon = STRATEGY_TABLE.get(
            key, ("twap", "lit", 0.5, 0.05, 3)
        )

        # Expected cost (base + spread + impact)
        base_cost = conditions.spread_bps / 2
        impact_cost = 10 * math.sqrt(participation) * 100  # sqrt model, in bps
        expected_cost = base_cost + impact_cost

        # Apply TCA adjustment if we have historical data
        tca_key = (algo, venue)
        if tca_key in self._tca_adjustments:
            expected_cost *= self._tca_adjustments[tca_key]

        return ExecutionStrategy(
            algorithm=algo,
            venue=venue,
            urgency=urg,
            participation_limit=part_limit,
            time_horizon_bars=horizon,
            expected_cost_bps=expected_cost,
            reason=f"Regime={regime}, Size={size_level}, Urgency={urgency_level}",
        )

    def apply_tca_adjustment(self, algo: str, venue: str, multiplier: float) -> None:
        """Apply TCA-derived cost adjustment."""
        self._tca_adjustments[(algo, venue)] = max(0.5, min(2.0, multiplier))


# ---------------------------------------------------------------------------
# 2. TCA Feedback Loop
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRecord:
    """Record of a completed execution for TCA analysis."""
    symbol: str
    algorithm: str
    venue: str
    side: str
    notional: float
    estimated_cost_bps: float
    actual_cost_bps: float
    slippage_bps: float        # arrival price vs execution price
    fill_rate: float           # 0-1, fraction filled
    execution_time_seconds: float
    market_vol_at_execution: float
    spread_at_execution_bps: float
    timestamp: float = 0.0


class TCAFeedbackEngine:
    """
    Closed-loop TCA: feed realized slippage back into routing decisions.

    Tracks slippage error (actual - estimated) per (algorithm, venue) pair.
    If a venue consistently underperforms expectations, reduce its priority.
    If a venue outperforms, increase its priority.
    """

    def __init__(self, window: int = 100, update_interval: int = 20):
        self.window = window
        self.update_interval = update_interval
        self._records: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=window))
        self._count = 0
        self._adjustments: Dict[Tuple[str, str], float] = {}

    def record(self, execution: ExecutionRecord) -> None:
        """Record a completed execution."""
        key = (execution.algorithm, execution.venue)
        self._records[key].append(execution)
        self._count += 1

        # Auto-update adjustments periodically
        if self._count % self.update_interval == 0:
            self.recompute_adjustments()

    def recompute_adjustments(self) -> Dict[Tuple[str, str], float]:
        """
        Recompute cost adjustment multipliers from recent execution data.

        If actual_cost > estimated_cost consistently: multiplier > 1.0 (penalize)
        If actual_cost < estimated_cost consistently: multiplier < 1.0 (reward)
        """
        for key, records in self._records.items():
            if len(records) < 5:
                continue

            errors = [r.actual_cost_bps - r.estimated_cost_bps for r in records]
            mean_error = sum(errors) / len(errors)

            # Convert error to multiplier
            # +5 bps mean error -> 1.1x multiplier (penalize by 10%)
            # -5 bps mean error -> 0.9x multiplier (reward by 10%)
            multiplier = 1.0 + mean_error / 50.0
            multiplier = max(0.5, min(2.0, multiplier))

            self._adjustments[key] = multiplier

        return dict(self._adjustments)

    def get_adjustment(self, algorithm: str, venue: str) -> float:
        """Get the current cost adjustment multiplier for an algo/venue pair."""
        return self._adjustments.get((algorithm, venue), 1.0)

    def get_report(self) -> Dict:
        """TCA summary report."""
        report = {}
        for key, records in self._records.items():
            if not records:
                continue
            errors = [r.actual_cost_bps - r.estimated_cost_bps for r in records]
            report[f"{key[0]}:{key[1]}"] = {
                "n_trades": len(records),
                "mean_slippage_error_bps": sum(errors) / len(errors),
                "adjustment_multiplier": self._adjustments.get(key, 1.0),
                "avg_fill_rate": sum(r.fill_rate for r in records) / len(records),
                "avg_execution_time_s": sum(r.execution_time_seconds for r in records) / len(records),
            }
        return report


# ---------------------------------------------------------------------------
# 3. Dark Pool Learning
# ---------------------------------------------------------------------------

@dataclass
class DarkPoolStats:
    """Rolling statistics for a dark pool venue."""
    venue_name: str
    fill_rate_history: deque = field(default_factory=lambda: deque(maxlen=200))
    adverse_selection_history: deque = field(default_factory=lambda: deque(maxlen=200))
    information_leakage_events: int = 0
    estimated_fill_probability: float = 0.4
    toxicity_score: float = 0.0  # 0=safe, 1=toxic


class DarkPoolLearner:
    """
    Learn dark pool fill rates and toxicity from historical fills.

    Adjusts routing to prefer dark pools with:
      - High fill rates (more liquidity)
      - Low adverse selection (less front-running)
      - Low information leakage (price doesn't move against us after fill)
    """

    def __init__(self, venues: List[str] = None):
        self.venues = venues or ["dark_a", "dark_b", "midpoint_cross"]
        self.stats: Dict[str, DarkPoolStats] = {
            v: DarkPoolStats(venue_name=v) for v in self.venues
        }

    def record_attempt(self, venue: str, filled: bool, fill_fraction: float,
                       post_fill_adverse_move_bps: float) -> None:
        """Record a dark pool interaction."""
        if venue not in self.stats:
            self.stats[venue] = DarkPoolStats(venue_name=venue)

        s = self.stats[venue]
        s.fill_rate_history.append(1.0 if filled else 0.0)
        if filled:
            s.adverse_selection_history.append(post_fill_adverse_move_bps)
            if post_fill_adverse_move_bps > 5.0:  # >5 bps adverse = leakage event
                s.information_leakage_events += 1

        # Update estimates
        if s.fill_rate_history:
            s.estimated_fill_probability = sum(s.fill_rate_history) / len(s.fill_rate_history)

        if s.adverse_selection_history:
            avg_adverse = sum(s.adverse_selection_history) / len(s.adverse_selection_history)
            s.toxicity_score = min(1.0, avg_adverse / 10.0)  # 10 bps adverse = fully toxic

    def rank_venues(self) -> List[Tuple[str, float]]:
        """
        Rank dark pool venues by desirability.
        Score = fill_rate * (1 - toxicity).
        Higher = better.
        """
        rankings = []
        for venue, stats in self.stats.items():
            score = stats.estimated_fill_probability * (1 - stats.toxicity_score)
            rankings.append((venue, score))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def should_use_dark(self, venue: str, min_fill_prob: float = 0.1,
                         max_toxicity: float = 0.7) -> bool:
        """Should we route to this dark pool?"""
        s = self.stats.get(venue)
        if s is None:
            return True  # no data, try it
        return s.estimated_fill_probability >= min_fill_prob and s.toxicity_score <= max_toxicity

    def get_report(self) -> Dict:
        return {
            venue: {
                "fill_prob": s.estimated_fill_probability,
                "toxicity": s.toxicity_score,
                "leakage_events": s.information_leakage_events,
                "n_observations": len(s.fill_rate_history),
            }
            for venue, s in self.stats.items()
        }


# ---------------------------------------------------------------------------
# Unified Adaptive Executor
# ---------------------------------------------------------------------------

class AdaptiveExecutor:
    """
    Unified adaptive execution engine combining:
    1. Regime-aware strategy selection
    2. TCA feedback loop
    3. Dark pool learning

    Usage:
      executor = AdaptiveExecutor()
      strategy = executor.select_strategy(conditions, notional, urgency)
      # ... execute the order ...
      executor.record_execution(execution_record)
    """

    def __init__(self, dark_venues: List[str] = None):
        self.selector = AdaptiveStrategySelector()
        self.tca = TCAFeedbackEngine()
        self.dark_pool = DarkPoolLearner(dark_venues)

    def select_strategy(
        self,
        conditions: MarketConditions,
        order_notional: float,
        urgency: float = 0.5,
    ) -> ExecutionStrategy:
        """Select optimal execution strategy given current conditions."""
        # Get base strategy from regime-aware selector
        strategy = self.selector.select(conditions, order_notional, urgency)

        # Apply TCA adjustments
        tca_adj = self.tca.get_adjustment(strategy.algorithm, strategy.venue)
        strategy.expected_cost_bps *= tca_adj

        # Dark pool override: if recommended dark but venue is toxic, switch to lit
        if strategy.venue in ("dark", "mixed"):
            rankings = self.dark_pool.rank_venues()
            if rankings and rankings[0][1] < 0.1:
                # All dark pools are bad, go lit
                strategy.venue = "lit"
                strategy.reason += " (dark pools toxic, switched to lit)"

        return strategy

    def record_execution(self, record: ExecutionRecord) -> None:
        """Record a completed execution for learning."""
        self.tca.record(record)

    def record_dark_attempt(self, venue: str, filled: bool, fill_frac: float,
                             adverse_bps: float) -> None:
        """Record a dark pool attempt for learning."""
        self.dark_pool.record_attempt(venue, filled, fill_frac, adverse_bps)

    def get_full_report(self) -> Dict:
        """Combined report across all components."""
        return {
            "tca": self.tca.get_report(),
            "dark_pools": self.dark_pool.get_report(),
            "dark_pool_rankings": self.dark_pool.rank_venues(),
            "tca_adjustments": dict(self.tca._adjustments),
        }
