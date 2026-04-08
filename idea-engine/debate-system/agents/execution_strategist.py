"""
debate-system/agents/execution_strategist.py

ExecutionStrategist -- evaluates trade execution feasibility and recommends
optimal execution strategy for the debate system.

Execution dimensions analysed:
  - Market impact cost estimation (Almgren-Chriss model)
  - Optimal execution horizon given urgency and alpha decay
  - Venue selection: dark pool vs lit exchange routing
  - Intraday timing patterns (U-shape volume, spread dynamics)
  - Urgency assessment: how quickly must the trade be executed
  - Slippage estimation from order size, ADV, and bid-ask spread
  - Execution risk: probability of adverse price movement during execution
  - Position sizing recommendation based on liquidity constraints

Execution strategies evaluated:
  - TWAP  (Time-Weighted Average Price)
  - VWAP  (Volume-Weighted Average Price)
  - IS    (Implementation Shortfall / Arrival Price)
  - Opportunistic / Liquidity-Seeking
  - Iceberg / Reserve
  - Close / MOC (Market on Close)

Output: ExecutionVerdict with strategy, expected_cost_bps, risk_score, sizing,
        and a full cost breakdown.

Expected market_data keys
-------------------------
order_size_usd        : float  -- notional order size in USD
adv_usd               : float  -- 20-day average daily volume in USD
adv_shares            : float  -- 20-day average daily shares traded
bid_ask_spread_bps    : float  -- current bid-ask spread in basis points
daily_volatility      : float  -- annualized daily volatility (decimal, e.g., 0.25)
intraday_volume_curve : np.ndarray, optional  -- 78 half-hour buckets normalised
current_price         : float  -- current mid price
urgency               : str    -- "low" | "medium" | "high" | "critical"
alpha_decay_halflife  : float, optional  -- hours until alpha halves
hypothesis_direction  : str    -- "long" | "short"
asset_class           : str    -- "equity" | "crypto" | "fx" | "commodity"
market_cap_usd        : float, optional  -- market cap for equity
dark_pool_available   : bool   -- whether dark pools are accessible
exchange_hours_left   : float, optional  -- hours until market close
portfolio_nav         : float, optional  -- total portfolio NAV for sizing
max_position_pct      : float, optional  -- max single-position % of NAV
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote
from hypothesis.types import Hypothesis


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExecutionStrategy(str, Enum):
    TWAP           = "twap"
    VWAP           = "vwap"
    IS             = "implementation_shortfall"
    OPPORTUNISTIC  = "opportunistic"
    ICEBERG        = "iceberg"
    MOC            = "market_on_close"


class Urgency(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class VenueType(str, Enum):
    LIT        = "lit"
    DARK       = "dark"
    MIXED      = "mixed"
    DARK_FIRST = "dark_first"


class ExecutionRiskLevel(str, Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    EXTREME  = "extreme"


# ---------------------------------------------------------------------------
# ExecutionVerdict
# ---------------------------------------------------------------------------

@dataclass
class ExecutionVerdict:
    """
    Rich output from ExecutionStrategist.evaluate().
    """
    recommended_strategy: ExecutionStrategy
    alternative_strategy: ExecutionStrategy | None
    expected_cost_bps: float              # total expected execution cost
    market_impact_bps: float              # price impact component
    spread_cost_bps: float                # half-spread component
    timing_cost_bps: float                # delay/opportunity cost
    risk_score: float                     # 0-1, probability of adverse move
    risk_level: ExecutionRiskLevel
    recommended_horizon_minutes: int      # how long to execute
    venue_recommendation: VenueType
    dark_pool_pct: float                  # % to route to dark pools
    recommended_size_usd: float           # position size recommendation
    recommended_size_pct_nav: float       # as % of portfolio NAV
    adv_participation_rate: float         # order as fraction of ADV
    num_slices: int                       # number of child orders
    reasoning: list[str]
    warnings: list[str]
    cost_breakdown: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_strategy": self.recommended_strategy.value,
            "alternative_strategy": (
                self.alternative_strategy.value
                if self.alternative_strategy else None
            ),
            "expected_cost_bps": round(self.expected_cost_bps, 2),
            "market_impact_bps": round(self.market_impact_bps, 2),
            "spread_cost_bps": round(self.spread_cost_bps, 2),
            "timing_cost_bps": round(self.timing_cost_bps, 2),
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "recommended_horizon_minutes": self.recommended_horizon_minutes,
            "venue_recommendation": self.venue_recommendation.value,
            "dark_pool_pct": round(self.dark_pool_pct, 2),
            "recommended_size_usd": round(self.recommended_size_usd, 2),
            "recommended_size_pct_nav": round(self.recommended_size_pct_nav, 4),
            "adv_participation_rate": round(self.adv_participation_rate, 4),
            "num_slices": self.num_slices,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
            "cost_breakdown": {
                k: round(v, 2) for k, v in self.cost_breakdown.items()
            },
        }


# ---------------------------------------------------------------------------
# Almgren-Chriss impact model parameters
# ---------------------------------------------------------------------------

@dataclass
class ImpactModelParams:
    """
    Simplified Almgren-Chriss temporary + permanent impact parameters.

    permanent_impact = eta * sigma * (Q / V)^delta
    temporary_impact = gamma * sigma * (Q / (V * T))^kappa

    where Q = order quantity, V = daily volume, T = execution time (days),
    sigma = daily volatility.
    """
    eta: float = 0.142        # permanent impact coefficient
    delta: float = 0.60       # permanent impact exponent
    gamma: float = 0.314      # temporary impact coefficient
    kappa: float = 0.60       # temporary impact exponent
    spread_fraction: float = 0.5  # fraction of spread paid (crossing)


# ---------------------------------------------------------------------------
# Urgency multipliers
# ---------------------------------------------------------------------------

_URGENCY_HORIZON_SCALE: dict[str, float] = {
    "low":      1.0,
    "medium":   0.6,
    "high":     0.3,
    "critical": 0.1,
}

_URGENCY_PARTICIPATION_CAP: dict[str, float] = {
    "low":      0.05,   # max 5% of ADV
    "medium":   0.10,
    "high":     0.20,
    "critical": 0.40,
}


# ---------------------------------------------------------------------------
# Intraday volume U-curve (default if not provided)
# ---------------------------------------------------------------------------

def _default_intraday_volume_curve() -> np.ndarray:
    """
    Generates a stylized U-shaped intraday volume curve.
    13 half-hour buckets for a 6.5-hour trading day.
    """
    buckets = 13
    x = np.linspace(0, 1, buckets)
    # U-shape: high at open and close, low at midday
    curve = 1.0 + 1.5 * (4.0 * (x - 0.5) ** 2)
    curve /= curve.sum()
    return curve


# ---------------------------------------------------------------------------
# ExecutionStrategist
# ---------------------------------------------------------------------------

class ExecutionStrategist(BaseAnalyst):
    """
    Execution strategy agent for the multi-agent debate system.

    Estimates execution costs, recommends optimal strategy and horizon,
    assesses execution risk, and provides position sizing guidance
    based on liquidity constraints.
    """

    def __init__(
        self,
        name: str = "ExecutionStrategist",
        initial_credibility: float = 0.5,
        impact_params: ImpactModelParams | None = None,
        max_cost_bps_threshold: float = 50.0,
        alpha: float = 5.0,
        beta: float = 5.0,
    ) -> None:
        super().__init__(
            name=name,
            specialization="execution_strategy",
            initial_credibility=initial_credibility,
            alpha=alpha,
            beta=beta,
        )
        self._impact = impact_params or ImpactModelParams()
        self._max_cost_threshold = max_cost_bps_threshold
        self._execution_history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public: full evaluate -> ExecutionVerdict
    # ------------------------------------------------------------------

    def evaluate(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> ExecutionVerdict:
        """Run the full execution analysis pipeline."""
        order_size = float(market_data.get("order_size_usd", 0.0))
        adv = float(market_data.get("adv_usd", 1.0))
        spread_bps = float(market_data.get("bid_ask_spread_bps", 5.0))
        vol = float(market_data.get("daily_volatility", 0.25))
        price = float(market_data.get("current_price", 100.0))
        urgency_str = str(market_data.get("urgency", "medium"))
        dark_ok = bool(market_data.get("dark_pool_available", True))
        alpha_hl = market_data.get("alpha_decay_halflife")
        hours_left = market_data.get("exchange_hours_left")
        nav = market_data.get("portfolio_nav")
        max_pos_pct = float(market_data.get("max_position_pct", 0.05))

        reasoning: list[str] = []
        warnings: list[str] = []

        # 1. Participation rate
        participation = order_size / adv if adv > 0 else 1.0
        reasoning.append(f"Participation rate: {participation:.2%} of ADV")

        if participation > 0.25:
            warnings.append(
                f"Order is {participation:.0%} of ADV — significant market impact expected"
            )

        # 2. Market impact estimation (Almgren-Chriss simplified)
        horizon_days = self._compute_optimal_horizon(
            participation, urgency_str, alpha_hl, hours_left,
        )
        horizon_minutes = int(horizon_days * 6.5 * 60)  # trading day hours

        permanent_impact = self._permanent_impact_bps(participation, vol)
        temporary_impact = self._temporary_impact_bps(
            participation, vol, horizon_days,
        )
        spread_cost = spread_bps * self._impact.spread_fraction
        timing_cost = self._timing_cost_bps(urgency_str, vol, horizon_days)

        total_cost = permanent_impact + temporary_impact + spread_cost + timing_cost

        reasoning.append(
            f"Impact model: perm={permanent_impact:.1f}bp, "
            f"temp={temporary_impact:.1f}bp, spread={spread_cost:.1f}bp, "
            f"timing={timing_cost:.1f}bp"
        )

        # 3. Strategy selection
        strategy, alt_strategy = self._select_strategy(
            participation, urgency_str, alpha_hl, horizon_minutes,
            total_cost, vol, hours_left, market_data,
        )
        reasoning.append(f"Recommended: {strategy.value}")
        if alt_strategy:
            reasoning.append(f"Alternative: {alt_strategy.value}")

        # 4. Venue selection
        venue, dark_pct = self._select_venue(
            participation, spread_bps, dark_ok, strategy,
        )
        reasoning.append(f"Venue: {venue.value} (dark={dark_pct:.0%})")

        # 5. Execution risk
        risk_score = self._execution_risk(
            participation, vol, horizon_days, urgency_str,
        )
        risk_level = self._risk_level(risk_score)
        if risk_level in (ExecutionRiskLevel.HIGH, ExecutionRiskLevel.EXTREME):
            warnings.append(
                f"Execution risk is {risk_level.value}: "
                f"consider reducing size or extending horizon"
            )

        # 6. Position sizing
        rec_size, rec_pct = self._position_sizing(
            order_size, adv, vol, participation, nav, max_pos_pct,
        )
        if rec_size < order_size:
            warnings.append(
                f"Recommended size ({rec_size:,.0f} USD) is less than "
                f"requested ({order_size:,.0f} USD) due to liquidity constraints"
            )

        # 7. Slicing
        num_slices = self._compute_slices(rec_size, adv, horizon_minutes, strategy)

        # 8. Cost threshold check
        if total_cost > self._max_cost_threshold:
            warnings.append(
                f"Total expected cost ({total_cost:.1f}bp) exceeds "
                f"threshold ({self._max_cost_threshold:.0f}bp)"
            )

        cost_breakdown = {
            "permanent_impact_bps": permanent_impact,
            "temporary_impact_bps": temporary_impact,
            "spread_cost_bps": spread_cost,
            "timing_cost_bps": timing_cost,
            "total_cost_bps": total_cost,
        }

        return ExecutionVerdict(
            recommended_strategy=strategy,
            alternative_strategy=alt_strategy,
            expected_cost_bps=total_cost,
            market_impact_bps=permanent_impact + temporary_impact,
            spread_cost_bps=spread_cost,
            timing_cost_bps=timing_cost,
            risk_score=risk_score,
            risk_level=risk_level,
            recommended_horizon_minutes=max(1, horizon_minutes),
            venue_recommendation=venue,
            dark_pool_pct=dark_pct,
            recommended_size_usd=rec_size,
            recommended_size_pct_nav=rec_pct,
            adv_participation_rate=participation,
            num_slices=num_slices,
            reasoning=reasoning,
            warnings=warnings,
            cost_breakdown=cost_breakdown,
        )

    # ------------------------------------------------------------------
    # BaseAnalyst interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> AnalystVerdict:
        """Wrap evaluate() into an AnalystVerdict for the DebateChamber."""
        ev = self.evaluate(hypothesis, market_data)

        # High execution cost or risk -> vote AGAINST
        if ev.expected_cost_bps > self._max_cost_threshold:
            vote = Vote.AGAINST
            confidence = min(1.0, ev.expected_cost_bps / (2 * self._max_cost_threshold))
            reasoning = (
                f"Execution cost too high ({ev.expected_cost_bps:.1f}bp > "
                f"{self._max_cost_threshold:.0f}bp threshold). "
                f"Strategy: {ev.recommended_strategy.value}"
            )
        elif ev.risk_level == ExecutionRiskLevel.EXTREME:
            vote = Vote.AGAINST
            confidence = ev.risk_score
            reasoning = (
                f"Extreme execution risk (score={ev.risk_score:.2f}). "
                f"Participation {ev.adv_participation_rate:.1%} of ADV."
            )
        elif ev.risk_level == ExecutionRiskLevel.HIGH:
            vote = Vote.FOR
            confidence = 0.3
            reasoning = (
                f"Executable but high risk. Cost={ev.expected_cost_bps:.1f}bp, "
                f"horizon={ev.recommended_horizon_minutes}min, "
                f"{ev.recommended_strategy.value}"
            )
        else:
            vote = Vote.FOR
            confidence = max(0.4, 1.0 - ev.expected_cost_bps / self._max_cost_threshold)
            reasoning = (
                f"Execution feasible. Cost={ev.expected_cost_bps:.1f}bp, "
                f"horizon={ev.recommended_horizon_minutes}min, "
                f"{ev.recommended_strategy.value}, "
                f"venue={ev.venue_recommendation.value}"
            )

        return self._make_verdict(
            vote=vote,
            confidence=confidence,
            reasoning=reasoning,
            key_concerns=ev.warnings[:5],
        )

    # ------------------------------------------------------------------
    # Impact model internals
    # ------------------------------------------------------------------

    def _permanent_impact_bps(
        self, participation: float, vol: float,
    ) -> float:
        """Permanent price impact in basis points."""
        if participation <= 0:
            return 0.0
        daily_vol_bps = vol / math.sqrt(252) * 10_000
        impact = (
            self._impact.eta
            * daily_vol_bps
            * (participation ** self._impact.delta)
        )
        return max(0.0, impact)

    def _temporary_impact_bps(
        self, participation: float, vol: float, horizon_days: float,
    ) -> float:
        """Temporary price impact in basis points."""
        if participation <= 0 or horizon_days <= 0:
            return 0.0
        daily_vol_bps = vol / math.sqrt(252) * 10_000
        trading_rate = participation / horizon_days
        impact = (
            self._impact.gamma
            * daily_vol_bps
            * (trading_rate ** self._impact.kappa)
        )
        return max(0.0, impact)

    def _timing_cost_bps(
        self, urgency: str, vol: float, horizon_days: float,
    ) -> float:
        """
        Opportunity cost of delayed execution.
        Higher urgency -> lower timing cost (we execute fast).
        But higher vol -> higher timing cost per unit of delay.
        """
        if horizon_days <= 0:
            return 0.0
        daily_vol_bps = vol / math.sqrt(252) * 10_000
        # Cost of waiting: vol-scaled, proportional to horizon
        delay_cost = daily_vol_bps * math.sqrt(horizon_days) * 0.1
        urgency_discount = {
            "critical": 0.1,
            "high": 0.3,
            "medium": 0.6,
            "low": 1.0,
        }
        return delay_cost * urgency_discount.get(urgency, 0.6)

    # ------------------------------------------------------------------
    # Optimal horizon
    # ------------------------------------------------------------------

    def _compute_optimal_horizon(
        self,
        participation: float,
        urgency: str,
        alpha_halflife: float | None,
        hours_left: float | None,
    ) -> float:
        """
        Compute optimal execution horizon in trading days.
        Balances impact (wants slow) vs alpha decay (wants fast) vs urgency.
        """
        # Base horizon from participation rate
        if participation < 0.01:
            base_hours = 0.5
        elif participation < 0.05:
            base_hours = 2.0
        elif participation < 0.10:
            base_hours = 4.0
        elif participation < 0.25:
            base_hours = 6.5  # full day
        else:
            base_hours = 13.0  # two days

        # Urgency scaling
        scale = _URGENCY_HORIZON_SCALE.get(urgency, 0.6)
        horizon_hours = base_hours * scale

        # Alpha decay constraint: don't execute longer than 2x alpha halflife
        if alpha_halflife is not None and alpha_halflife > 0:
            max_hours = alpha_halflife * 2.0
            horizon_hours = min(horizon_hours, max_hours)

        # Can't exceed hours left in session
        if hours_left is not None and hours_left > 0:
            horizon_hours = min(horizon_hours, hours_left)

        # Floor at 5 minutes (0.013 hours)
        horizon_hours = max(0.013, horizon_hours)

        return horizon_hours / 6.5  # convert to trading days

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_strategy(
        self,
        participation: float,
        urgency: str,
        alpha_halflife: float | None,
        horizon_minutes: int,
        total_cost: float,
        vol: float,
        hours_left: float | None,
        md: dict,
    ) -> tuple[ExecutionStrategy, ExecutionStrategy | None]:
        """
        Select primary and alternative execution strategy.
        """
        # Critical urgency -> IS (minimize shortfall)
        if urgency == "critical":
            return ExecutionStrategy.IS, ExecutionStrategy.VWAP

        # Very small order -> just cross the spread
        if participation < 0.005:
            return ExecutionStrategy.IS, None

        # Large order with no urgency -> TWAP to minimize impact
        if participation > 0.15 and urgency == "low":
            return ExecutionStrategy.TWAP, ExecutionStrategy.ICEBERG

        # Fast alpha decay -> IS
        if alpha_halflife is not None and alpha_halflife < 1.0:
            return ExecutionStrategy.IS, ExecutionStrategy.VWAP

        # Close to market close -> MOC if hours_left < 1
        if hours_left is not None and hours_left < 1.0 and participation < 0.10:
            return ExecutionStrategy.MOC, ExecutionStrategy.VWAP

        # Medium participation -> VWAP (blend with natural volume)
        if 0.03 < participation < 0.15:
            return ExecutionStrategy.VWAP, ExecutionStrategy.OPPORTUNISTIC

        # Low participation with medium urgency -> opportunistic
        if participation < 0.03 and urgency in ("low", "medium"):
            return ExecutionStrategy.OPPORTUNISTIC, ExecutionStrategy.TWAP

        # Default: VWAP
        return ExecutionStrategy.VWAP, ExecutionStrategy.TWAP

    # ------------------------------------------------------------------
    # Venue selection
    # ------------------------------------------------------------------

    def _select_venue(
        self,
        participation: float,
        spread_bps: float,
        dark_available: bool,
        strategy: ExecutionStrategy,
    ) -> tuple[VenueType, float]:
        """
        Decide lit vs dark venue routing.
        Dark pools reduce information leakage but have execution uncertainty.
        """
        if not dark_available:
            return VenueType.LIT, 0.0

        # Large orders benefit most from dark pools
        if participation > 0.10:
            return VenueType.DARK_FIRST, 0.70

        # Wide spreads -> try dark for midpoint execution
        if spread_bps > 10.0:
            return VenueType.DARK_FIRST, 0.60

        # Medium orders -> mixed routing
        if participation > 0.03:
            return VenueType.MIXED, 0.40

        # Small orders -> lit is fine
        return VenueType.LIT, 0.0

    # ------------------------------------------------------------------
    # Execution risk
    # ------------------------------------------------------------------

    def _execution_risk(
        self,
        participation: float,
        vol: float,
        horizon_days: float,
        urgency: str,
    ) -> float:
        """
        Probability score (0-1) of adverse price movement during execution.
        Combines participation risk, volatility risk, and duration risk.
        """
        # Participation risk: large orders are more likely to move the market
        part_risk = min(1.0, participation / 0.30)

        # Volatility risk: higher vol -> more execution uncertainty
        daily_vol = vol / math.sqrt(252)
        vol_risk = min(1.0, daily_vol / 0.05)  # 5% daily vol = max risk

        # Duration risk: longer horizon -> more exposure to adverse moves
        dur_risk = min(1.0, horizon_days / 3.0)

        # Urgency penalty: critical urgency means we can't wait for better fills
        urgency_penalty = {
            "low": 0.0,
            "medium": 0.05,
            "high": 0.15,
            "critical": 0.30,
        }.get(urgency, 0.05)

        risk = (
            part_risk * 0.35
            + vol_risk * 0.30
            + dur_risk * 0.20
            + urgency_penalty
        )
        return min(1.0, max(0.0, risk))

    @staticmethod
    def _risk_level(risk_score: float) -> ExecutionRiskLevel:
        if risk_score > 0.75:
            return ExecutionRiskLevel.EXTREME
        if risk_score > 0.50:
            return ExecutionRiskLevel.HIGH
        if risk_score > 0.25:
            return ExecutionRiskLevel.MODERATE
        return ExecutionRiskLevel.LOW

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _position_sizing(
        self,
        requested_size: float,
        adv: float,
        vol: float,
        participation: float,
        nav: float | None,
        max_pos_pct: float,
    ) -> tuple[float, float]:
        """
        Recommend position size that respects liquidity and risk constraints.

        Returns (recommended_size_usd, recommended_pct_of_nav).
        """
        # Liquidity constraint: don't exceed 15% of ADV per day
        max_daily_liquidity = adv * 0.15
        # If we can spread over multiple days, allow more
        execution_days = max(1.0, participation / 0.15)
        liquidity_cap = max_daily_liquidity * min(execution_days, 5.0)

        rec_size = min(requested_size, liquidity_cap)

        # Volatility-based sizing: higher vol -> smaller position
        if vol > 0:
            daily_vol = vol / math.sqrt(252)
            # Target max 2% daily P&L at 1-sigma
            if nav is not None and nav > 0:
                vol_cap = (0.02 * nav) / daily_vol
                rec_size = min(rec_size, vol_cap)

        # NAV constraint
        rec_pct = 0.0
        if nav is not None and nav > 0:
            nav_cap = nav * max_pos_pct
            rec_size = min(rec_size, nav_cap)
            rec_pct = rec_size / nav
        else:
            rec_pct = 0.0

        return max(0.0, rec_size), rec_pct

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_slices(
        size_usd: float,
        adv: float,
        horizon_minutes: int,
        strategy: ExecutionStrategy,
    ) -> int:
        """Compute number of child order slices."""
        if strategy == ExecutionStrategy.IS:
            # IS tends to front-load, fewer slices
            base_slices = max(1, horizon_minutes // 10)
        elif strategy == ExecutionStrategy.TWAP:
            # TWAP: uniform slices
            base_slices = max(1, horizon_minutes // 5)
        elif strategy == ExecutionStrategy.VWAP:
            # VWAP: volume-proportional, moderate slicing
            base_slices = max(1, horizon_minutes // 5)
        elif strategy == ExecutionStrategy.ICEBERG:
            # Iceberg: many small slices
            base_slices = max(5, horizon_minutes // 3)
        elif strategy == ExecutionStrategy.MOC:
            # MOC: single order
            return 1
        else:
            base_slices = max(1, horizon_minutes // 8)

        # Scale down for very small orders
        if adv > 0:
            participation = size_usd / adv
            if participation < 0.01:
                base_slices = max(1, base_slices // 3)

        return min(500, max(1, base_slices))

    # ------------------------------------------------------------------
    # History tracking
    # ------------------------------------------------------------------

    def record_execution(
        self,
        predicted_cost_bps: float,
        actual_cost_bps: float,
        strategy_used: str,
    ) -> None:
        """Record an execution outcome for model calibration."""
        self._execution_history.append({
            "predicted": predicted_cost_bps,
            "actual": actual_cost_bps,
            "strategy": strategy_used,
            "error": actual_cost_bps - predicted_cost_bps,
        })
        # Keep last 200 executions
        if len(self._execution_history) > 200:
            self._execution_history = self._execution_history[-200:]

        # Update credibility based on prediction accuracy
        error_pct = abs(actual_cost_bps - predicted_cost_bps) / max(
            1.0, predicted_cost_bps,
        )
        self.update_credibility(error_pct < 0.30)

    @property
    def prediction_bias(self) -> float | None:
        """Mean prediction error: positive = underestimates cost."""
        if not self._execution_history:
            return None
        errors = [h["error"] for h in self._execution_history]
        return float(np.mean(errors))

    @property
    def prediction_mae(self) -> float | None:
        """Mean absolute error of cost predictions."""
        if not self._execution_history:
            return None
        errors = [abs(h["error"]) for h in self._execution_history]
        return float(np.mean(errors))
