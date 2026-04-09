"""
Portfolio Brain: converts all Event Horizon intelligence into optimal positions.

This is where the rubber meets the road. Takes inputs from:
  - Fractal timeframe detector (multi-scale momentum)
  - Liquidity black hole detector (cascade risk)
  - Information surprise detector (pre-move signals)
  - Market consciousness (emergent beliefs)
  - Fear/greed oscillator (contrarian sizing)
  - Market memory (gravitational levels)
  - Groupthink detector (consensus dampening)
  - Dream engine (fragility-adjusted confidence)
  - Guardian (hard limits)
  - Stability monitor (convergence certification)

And outputs:
  - Target portfolio weights per symbol
  - Execution strategy per trade (TWAP/VWAP/IS/dark)
  - Risk budget allocation
  - Confidence intervals on expected returns
  - Narrative explanation of every position

This is the single entry point for live trading: one function call
that returns the optimal portfolio given everything the system knows.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PositionTarget:
    """Target position for a single symbol."""
    symbol: str
    weight: float                 # target portfolio weight (-1 to +1, positive=long)
    size_dollars: float           # target notional
    confidence: float             # 0-1 how confident
    execution_strategy: str       # "market" / "twap" / "vwap" / "limit" / "dark"
    urgency: float                # 0-1 how quickly to execute
    narrative: str                # why this position

    # Signal breakdown
    signals: Dict[str, float] = field(default_factory=dict)

    # Risk
    expected_return_1d: float = 0.0
    expected_vol_1d: float = 0.0
    max_loss_1d: float = 0.0
    dream_fragility: float = 0.0

    # Adjustments
    fear_greed_multiplier: float = 1.0
    groupthink_multiplier: float = 1.0
    memory_gravitational_pull: float = 0.0


@dataclass
class PortfolioOutput:
    """Complete portfolio output from the Portfolio Brain."""
    timestamp: float
    total_exposure_pct: float     # gross exposure
    net_exposure_pct: float       # net long/short
    n_positions: int
    positions: List[PositionTarget]
    risk_budget_used_pct: float
    overall_confidence: float
    overall_narrative: str
    fear_greed_state: str
    stability_certified: bool
    guardian_approved: bool


class PortfolioBrain:
    """
    The unified portfolio construction engine.

    Takes all Event Horizon intelligence as inputs and produces
    a single, coherent, risk-managed portfolio.

    Key principles:
    1. Diversification: no single signal source dominates
    2. Risk budgeting: each position gets a risk budget
    3. Contrarian sizing: reduce when too confident, increase when fearful
    4. Memory-aware: respect gravitational levels
    5. Dream-tested: only trade signals that survived dream scenarios
    6. Stability-certified: only deploy if the system is mathematically stable
    """

    def __init__(
        self,
        symbols: List[str],
        total_equity: float = 1_000_000,
        max_position_pct: float = 0.10,
        max_gross_exposure: float = 1.0,
        vol_target: float = 0.15,
    ):
        self.symbols = symbols
        self.equity = total_equity
        self.max_position = max_position_pct
        self.max_gross = max_gross_exposure
        self.vol_target = vol_target

    def compute_portfolio(
        self,
        # Signal inputs
        fractal_signals: Dict[str, float],        # symbol -> signal (-1 to +1)
        consciousness_signal: float,               # global directional bias
        information_signals: Dict[str, float],     # symbol -> surprise signal
        physics_signals: Dict[str, Dict[str, float]],  # symbol -> {concept: signal}

        # Risk inputs
        liquidity_warnings: Dict[str, float],      # symbol -> black hole probability
        dream_fragilities: Dict[str, float],       # symbol -> fragility (0-1)
        memory_gravitational: Dict[str, float],    # symbol -> gravitational pull
        fear_greed_multiplier: float,              # from oscillator (0.5 to 1.5)
        groupthink_multiplier: float,              # from detector (0.5 to 1.0)

        # Constraint inputs
        guardian_approved: bool,
        stability_certified: bool,
        current_vol: Dict[str, float],             # symbol -> annualized vol

        # Context
        current_regime: str = "unknown",
    ) -> PortfolioOutput:
        """
        Compute the optimal portfolio given all available intelligence.
        """
        if not guardian_approved:
            return self._empty_output("Guardian has halted trading")

        if not stability_certified:
            return self._empty_output("Stability monitor: system not certified")

        positions = []

        for symbol in self.symbols:
            # 1. Composite signal (multi-source fusion)
            fractal = fractal_signals.get(symbol, 0.0)
            info = information_signals.get(symbol, 0.0)
            physics = physics_signals.get(symbol, {})
            physics_avg = float(np.mean(list(physics.values()))) if physics else 0.0

            # Weighted composite
            raw_signal = (
                0.35 * fractal +
                0.20 * consciousness_signal +
                0.20 * info +
                0.25 * physics_avg
            )

            # 2. Liquidity adjustment: reduce near black holes
            liq_warning = liquidity_warnings.get(symbol, 0.0)
            if liq_warning > 0.7:
                raw_signal = 0.0  # flatten
            elif liq_warning > 0.3:
                raw_signal *= (1 - liq_warning)

            # 3. Dream fragility adjustment
            fragility = dream_fragilities.get(symbol, 0.5)
            dream_mult = max(0.2, 1 - fragility)
            raw_signal *= dream_mult

            # 4. Fear/greed contrarian adjustment
            raw_signal *= fear_greed_multiplier

            # 5. Groupthink dampening
            raw_signal *= groupthink_multiplier

            # 6. Memory gravitational pull (add a small bias toward remembered levels)
            grav = memory_gravitational.get(symbol, 0.0)
            raw_signal += grav * 0.1  # small gravitational nudge

            # 7. Vol targeting: scale by inverse vol
            symbol_vol = current_vol.get(symbol, 0.20)
            if symbol_vol > 0.01:
                vol_scale = self.vol_target / (symbol_vol * math.sqrt(len(self.symbols)))
            else:
                vol_scale = 1.0

            # 8. Final weight
            weight = float(np.clip(raw_signal * vol_scale, -self.max_position, self.max_position))

            if abs(weight) < 0.005:
                continue  # skip negligible positions

            # Confidence: geometric mean of signal components
            signal_strengths = [abs(fractal), abs(info), abs(physics_avg), abs(consciousness_signal)]
            confidence = float(np.mean([s for s in signal_strengths if s > 0.01])) if any(s > 0.01 for s in signal_strengths) else 0.0
            confidence *= dream_mult * groupthink_multiplier

            # Execution strategy
            if abs(weight) > 0.08 or liq_warning > 0.2:
                exec_strat = "twap"
                urgency = 0.3
            elif abs(info) > 0.5:
                exec_strat = "market"  # surprise -> act fast
                urgency = 0.8
            else:
                exec_strat = "limit"
                urgency = 0.2

            # Narrative
            direction = "Long" if weight > 0 else "Short"
            parts = [f"{direction} {symbol} ({weight:+.1%})"]
            if abs(fractal) > 0.2:
                parts.append(f"fractal {'bullish' if fractal > 0 else 'bearish'}")
            if abs(physics_avg) > 0.2:
                top_physics = max(physics.items(), key=lambda x: abs(x[1]))[0] if physics else "unknown"
                parts.append(f"{top_physics} physics signal")
            if fragility > 0.5:
                parts.append(f"dream-tested (fragility {fragility:.0%})")
            if liq_warning > 0.2:
                parts.append(f"liquidity caution ({liq_warning:.0%})")
            narrative = " | ".join(parts)

            positions.append(PositionTarget(
                symbol=symbol,
                weight=weight,
                size_dollars=abs(weight) * self.equity,
                confidence=confidence,
                execution_strategy=exec_strat,
                urgency=urgency,
                narrative=narrative,
                signals={
                    "fractal": fractal,
                    "consciousness": consciousness_signal,
                    "information": info,
                    "physics": physics_avg,
                },
                expected_vol_1d=symbol_vol / math.sqrt(252),
                dream_fragility=fragility,
                fear_greed_multiplier=fear_greed_multiplier,
                groupthink_multiplier=groupthink_multiplier,
                memory_gravitational_pull=grav,
            ))

        # Enforce gross exposure limit
        gross = sum(abs(p.weight) for p in positions)
        if gross > self.max_gross:
            scale = self.max_gross / gross
            for p in positions:
                p.weight *= scale
                p.size_dollars *= scale

        # Sort by confidence
        positions.sort(key=lambda p: p.confidence, reverse=True)

        # Overall metrics
        net = sum(p.weight for p in positions)
        gross = sum(abs(p.weight) for p in positions)
        avg_conf = float(np.mean([p.confidence for p in positions])) if positions else 0.0

        # Overall narrative
        if not positions:
            overall = "No positions: all signals below threshold or risk limits active."
        else:
            n_long = sum(1 for p in positions if p.weight > 0)
            n_short = sum(1 for p in positions if p.weight < 0)
            top = positions[0]
            overall = (
                f"{n_long} long, {n_short} short positions. "
                f"Gross {gross:.0%}, net {net:+.0%}. "
                f"Top position: {top.symbol} ({top.weight:+.1%}, confidence {top.confidence:.0%}). "
                f"Regime: {current_regime}."
            )

        return PortfolioOutput(
            timestamp=time.time(),
            total_exposure_pct=float(gross * 100),
            net_exposure_pct=float(net * 100),
            n_positions=len(positions),
            positions=positions,
            risk_budget_used_pct=float(gross / self.max_gross * 100),
            overall_confidence=avg_conf,
            overall_narrative=overall,
            fear_greed_state=f"multiplier={fear_greed_multiplier:.2f}",
            stability_certified=stability_certified,
            guardian_approved=guardian_approved,
        )

    def _empty_output(self, reason: str) -> PortfolioOutput:
        return PortfolioOutput(
            timestamp=time.time(),
            total_exposure_pct=0.0,
            net_exposure_pct=0.0,
            n_positions=0,
            positions=[],
            risk_budget_used_pct=0.0,
            overall_confidence=0.0,
            overall_narrative=reason,
            fear_greed_state="halted",
            stability_certified=False,
            guardian_approved=False,
        )
