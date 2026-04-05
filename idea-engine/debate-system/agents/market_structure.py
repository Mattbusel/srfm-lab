"""
debate-system/agents/market_structure.py

MarketStructureAnalyst: evaluates whether a pattern has a plausible
microeconomic causal mechanism, or whether it is mere correlation.

Philosophy
----------
Pure statistical significance without a causal story is just noise that
survived the sample.  This analyst asks: WHY would this pattern exist?
Is there a structural reason that makes it likely to persist?

Domain knowledge encoded
------------------------
Hour-of-day patterns:
  - Hour 0-1 UTC  : Asian liquidity gap — thin books, wide spreads, easy
                    to push price.  Entry signals here frequently get bad
                    fills.  Blocking entry is structurally justified.
  - Hour 8-9 UTC  : European open — liquidity surge, often sharp moves.
  - Hour 13-14 UTC: US open — highest volume, most efficient prices.
  - Hour 22-23 UTC: End-of-day CME settlement — thin, unreliable.

SOL/high-inflation assets:
  - High token emission → constant sell pressure from validators/stakers.
  - VC unlock cliffs create predictable overhead.
  - These create structural headwinds, not just bad recent luck.

Causal scoring rubric
---------------------
STRONG causal story  → +2 points
PLAUSIBLE but thin   → +1 point
AMBIGUOUS            → 0 points
LIKELY SPURIOUS      → -1 point
DEFINITELY SPURIOUS  → -2 points

Score >= 1: FOR | Score 0: ABSTAIN | Score <= -1: AGAINST
"""

from __future__ import annotations

from typing import Any

from hypothesis.types import Hypothesis, HypothesisType
from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote


# ---------------------------------------------------------------------------
# Domain knowledge constants
# ---------------------------------------------------------------------------

# UTC hours with documented structural microstructure disadvantage
THIN_BOOK_HOURS = {0, 1, 22, 23}

# Assets with well-documented structural selling pressure
HIGH_INFLATION_ASSETS = {"SOL", "NEAR", "APT", "SUI", "SEI", "INJ", "TIA"}
VC_UNLOCK_RISK = {"SOL", "APT", "SUI", "ARB", "OP", "IMX"}

# Hypothesis description keywords that map to known causal mechanisms
CAUSAL_KEYWORDS = {
    "liquidity gap": "Asian session liquidity gap creates structurally wide spreads",
    "spread": "Spread widening is directly measurable and causally linked to fill quality",
    "inflation": "Token inflation creates constant circulating supply pressure",
    "vc unlock": "VC unlock events create predictable large-order flow",
    "settlement": "Futures settlement creates predictable hedging demand",
    "funding rate": "Funding rate rebalancing creates predictable directional flows",
    "open interest": "OI changes reflect leveraged exposure build-up with known liquidation cascades",
    "volume profile": "Volume clustering at known levels reflects institutional order flow",
}

# Keywords that suggest purely data-mined correlation
SPURIOUS_KEYWORDS = {
    "arbitrary", "random", "happened to", "coincidence",
    "magic number", "parameter search",
}


class MarketStructureAnalyst(BaseAnalyst):
    """
    Evaluates whether a trading pattern has a plausible microeconomic
    causal mechanism versus being a data artefact.
    """

    def __init__(self) -> None:
        super().__init__(
            name="MarketStructureAnalyst",
            specialization=(
                "Microeconomic causality, exchange microstructure, "
                "token economics, liquidity analysis, causal mechanism validation"
            ),
        )

    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> AnalystVerdict:
        """
        Assess causal plausibility of a hypothesis.

        Expected market_data keys
        -------------------------
        entry_hours      : list[int]  - UTC hours hypothesis applies to
        instruments      : list[str]  - symbols involved
        pattern_type     : str        - 'entry_timing' | 'asset_filter' | etc.
        causal_tag       : str | None - optional explicit causal tag from miner
        """
        desc = (hypothesis.description or "").lower()
        params = hypothesis.parameters or {}
        instruments: list[str] = market_data.get(
            "instruments", params.get("instruments", [])
        )
        entry_hours: list[int] = market_data.get(
            "entry_hours",
            self._extract_hours(params)
        )
        causal_tag: str = (market_data.get("causal_tag") or "").lower()

        concerns: list[str] = []
        causal_score = 0
        causal_stories: list[str] = []

        # --- Entry hour causal analysis ----------------------------------
        if entry_hours:
            thin_hours = [h for h in entry_hours if h in THIN_BOOK_HOURS]
            if thin_hours:
                causal_score += 2
                causal_stories.append(
                    f"Hours {thin_hours} UTC fall in the Asian liquidity gap "
                    f"(00:00-02:00 UTC) where order books are thin, effective "
                    f"spreads 2-4× wider, and adverse selection risk is highest. "
                    f"Structural microstructure basis is strong."
                )
            else:
                causal_score += 0
                concerns.append(
                    f"Entry hours {entry_hours} do not correspond to known "
                    f"structural microstructure disadvantage windows. "
                    f"Pattern may be data-specific."
                )

        # --- Asset-specific causal analysis ------------------------------
        if instruments:
            inflation_assets = [s for s in instruments if s.upper() in HIGH_INFLATION_ASSETS]
            unlock_assets = [s for s in instruments if s.upper() in VC_UNLOCK_RISK]

            if inflation_assets and hypothesis.type in (
                HypothesisType.REGIME_FILTER,
                HypothesisType.ENTRY_TIMING,
            ):
                causal_score += 2
                causal_stories.append(
                    f"{inflation_assets} have structurally high token inflation "
                    f"rates, creating constant sell pressure from validator/staker "
                    f"rewards. Poor win-rate is consistent with this mechanism."
                )

            if unlock_assets:
                causal_score += 1
                causal_stories.append(
                    f"{unlock_assets} are subject to VC/team unlock schedules "
                    f"(cliff and linear). Selling pressure from large unlocks is "
                    f"predictable and structurally documented."
                )

        # --- Keyword causal analysis ------------------------------------
        combined_text = desc + " " + causal_tag
        for keyword, explanation in CAUSAL_KEYWORDS.items():
            if keyword in combined_text:
                causal_score += 1
                causal_stories.append(f"Causal keyword '{keyword}': {explanation}")
                break   # one keyword match is sufficient bonus

        for keyword in SPURIOUS_KEYWORDS:
            if keyword in combined_text:
                causal_score -= 1
                concerns.append(
                    f"Description contains spurious-signal keyword '{keyword}', "
                    f"suggesting the pattern may have been found by exhaustive "
                    f"parameter search rather than hypothesis-driven research."
                )
                break

        # --- Correlation-without-mechanism penalty ----------------------
        has_causal_story = bool(causal_stories) or causal_score > 0
        if not has_causal_story:
            causal_score -= 1
            concerns.append(
                "No causal mechanism identified. Pattern appears to be "
                "correlation without economic explanation. High risk of "
                "regime change destroying the edge."
            )

        # --- Verdict -----------------------------------------------------
        if causal_score >= 2:
            return self._make_verdict(
                Vote.FOR,
                confidence=min(0.90, 0.60 + 0.10 * causal_score),
                reasoning=(
                    f"Strong causal story (score={causal_score}): "
                    + " | ".join(causal_stories[:2])
                ),
                key_concerns=concerns,
            )
        elif causal_score == 1:
            return self._make_verdict(
                Vote.FOR,
                confidence=0.60,
                reasoning=(
                    f"Plausible but thin causal story (score={causal_score}). "
                    + (causal_stories[0] if causal_stories else "")
                ),
                key_concerns=concerns,
            )
        elif causal_score == 0:
            return self._make_verdict(
                Vote.ABSTAIN,
                confidence=0.50,
                reasoning=(
                    "Causal mechanism is ambiguous. Cannot confirm or deny "
                    "microeconomic basis. Other agents should decide."
                ),
                key_concerns=concerns,
            )
        else:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=min(0.90, 0.55 + 0.15 * abs(causal_score)),
                reasoning=(
                    f"Negative causal score ({causal_score}): pattern likely "
                    f"spurious. "
                    + (concerns[0] if concerns else "")
                ),
                key_concerns=concerns,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_hours(self, params: dict[str, Any]) -> list[int]:
        """Try to extract relevant hours from hypothesis parameters."""
        hours: list[int] = []
        start = params.get("entry_hour_start")
        end = params.get("entry_hour_end")
        blocked = params.get("blocked_hours", [])
        if start is not None and end is not None:
            hours = list(range(int(start), int(end) + 1))
        elif blocked:
            hours = [int(h) for h in blocked]
        return hours
