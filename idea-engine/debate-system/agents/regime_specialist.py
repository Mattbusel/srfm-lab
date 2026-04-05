"""
debate-system/agents/regime_specialist.py

RegimeSpecialist: checks whether a hypothesis is regime-conditional.

Financial rationale
-------------------
Many patterns discovered in backtests are period-specific artefacts.
A momentum strategy discovered in 2020-2021 bull market data will fail
spectacularly in a bear market.  The regime specialist explicitly tests
whether the edge persists across all market conditions.

Regime classification approach
-------------------------------
The BTC return quintile method splits market days into 5 equal groups
based on BTC daily return.  A robust hypothesis should show positive
expected value in at least 3 of the 5 quintiles (especially the middle
three which represent "normal" markets), not just in the top quintile.

This directly addresses the common failure mode of strategies that only
work in trending bull markets.

Evaluation criteria
-------------------
VOTES FOR (high confidence) if:
  - Pattern PnL positive in >= 4/5 quintiles
  - Pattern PnL positive in ALL: bear, neutral, bull regimes
  - Sharpe > 0 in both high-vol and low-vol regimes

VOTES AGAINST if:
  - Pattern only positive in top 1 quintile (pure bull-market strategy)
  - Pattern negative in current regime (as of analysis date)
  - Strong regime dependency with no hedge mechanism

VOTES ABSTAIN if:
  - Insufficient data to split into quintiles (n < 100 per quintile)
"""

from __future__ import annotations

from typing import Any

from hypothesis.types import Hypothesis
from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote


class RegimeSpecialist(BaseAnalyst):
    """
    Tests whether hypothesis performance is stable across market regimes.
    """

    MIN_QUINTILE_N = 20       # minimum trades per quintile for reliable stats
    POSITIVE_QUINTILE_THRESHOLD = 3   # must be positive in at least this many
    HIGH_CONFIDENCE_THRESHOLD = 4     # positive in this many = strong FOR

    def __init__(self) -> None:
        super().__init__(
            name="RegimeSpecialist",
            specialization=(
                "Regime-conditional performance, BTC return quintile analysis, "
                "bull/bear/high-vol/low-vol regime splits, current-regime assessment"
            ),
        )

    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> AnalystVerdict:
        """
        Evaluate regime stability of the hypothesis.

        Expected market_data keys
        -------------------------
        quintile_pnls        : list[float] (len=5) - mean PnL per BTC return quintile
                               index 0 = worst BTC days, index 4 = best BTC days
        quintile_ns          : list[int]   (len=5) - observations per quintile
        bear_market_sharpe   : float | None
        bull_market_sharpe   : float | None
        high_vol_sharpe      : float | None
        low_vol_sharpe       : float | None
        current_regime       : str  - 'bull' | 'bear' | 'neutral' | 'high_vol' | 'low_vol'
        current_regime_pnl   : float | None - hypothesis perf in current regime
        """
        quintile_pnls: list[float] = list(market_data.get("quintile_pnls", []))
        quintile_ns: list[int] = list(market_data.get("quintile_ns", []))
        bear_sharpe: float | None = market_data.get("bear_market_sharpe")
        bull_sharpe: float | None = market_data.get("bull_market_sharpe")
        high_vol_sharpe: float | None = market_data.get("high_vol_sharpe")
        low_vol_sharpe: float | None = market_data.get("low_vol_sharpe")
        current_regime: str = market_data.get("current_regime", "neutral")
        current_pnl: float | None = market_data.get("current_regime_pnl")

        concerns: list[str] = []

        # --- Data sufficiency check -------------------------------------
        if len(quintile_pnls) < 5 or len(quintile_ns) < 5:
            return self._make_verdict(
                Vote.ABSTAIN,
                confidence=0.70,
                reasoning=(
                    "Insufficient data to perform BTC return quintile split. "
                    "Need 5 quintile PnL values. Regime analysis cannot be completed."
                ),
                key_concerns=["Provide quintile_pnls[5] and quintile_ns[5]"],
            )

        min_quintile_n = min(quintile_ns)
        if min_quintile_n < self.MIN_QUINTILE_N:
            concerns.append(
                f"Thinnest quintile has only {min_quintile_n} observations "
                f"(minimum: {self.MIN_QUINTILE_N}). Quintile statistics are noisy."
            )

        # --- Quintile analysis ------------------------------------------
        positive_quintiles = sum(1 for p in quintile_pnls if p > 0)
        worst_quintile_pnl = quintile_pnls[0]    # worst BTC days
        best_quintile_pnl = quintile_pnls[4]      # best BTC days
        middle_quintiles_positive = all(p > 0 for p in quintile_pnls[1:4])

        # Critical failure: works ONLY in top quintile
        if positive_quintiles == 1 and best_quintile_pnl > 0:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.92,
                reasoning=(
                    f"BULL-ONLY PATTERN: positive PnL in only 1 of 5 BTC return "
                    f"quintiles (the best 20% of market days). This strategy is "
                    f"a leveraged long-BTC exposure disguised as an alpha strategy. "
                    f"Will fail catastrophically in bear market."
                ),
                key_concerns=[
                    "Positive only in top quintile — bull-market beta trap",
                    f"Worst-quintile PnL = {worst_quintile_pnl:.4f}",
                    "Not a diversifying strategy",
                ],
            )

        # --- Current regime concern -------------------------------------
        if current_pnl is not None and current_pnl < 0:
            concerns.append(
                f"Pattern shows negative PnL ({current_pnl:.4f}) in current "
                f"regime '{current_regime}'. Deploying now carries elevated risk."
            )
            if current_regime in ("bear", "high_vol"):
                concerns.append(
                    "Current regime is the most adversarial type for most crypto "
                    "strategies. Timing risk is high."
                )

        # --- Sharpe across broad regimes --------------------------------
        regime_sharpes = {
            "bear": bear_sharpe,
            "bull": bull_sharpe,
            "high_vol": high_vol_sharpe,
            "low_vol": low_vol_sharpe,
        }
        negative_regimes = [
            name for name, s in regime_sharpes.items()
            if s is not None and s < 0
        ]
        if negative_regimes:
            concerns.append(
                f"Negative Sharpe in regime(s): {negative_regimes}. "
                f"Strategy degrades in these conditions."
            )

        # --- Verdict based on quintile breadth --------------------------
        if positive_quintiles >= self.HIGH_CONFIDENCE_THRESHOLD and middle_quintiles_positive:
            confidence = 0.80 + 0.04 * (positive_quintiles - self.HIGH_CONFIDENCE_THRESHOLD)
            return self._make_verdict(
                Vote.FOR,
                confidence=min(0.93, confidence),
                reasoning=(
                    f"Regime-robust: positive in {positive_quintiles}/5 BTC "
                    f"return quintiles including all middle quintiles. "
                    f"Pattern survives bear, neutral, and bull market conditions."
                ),
                key_concerns=concerns,
            )
        elif positive_quintiles >= self.POSITIVE_QUINTILE_THRESHOLD:
            return self._make_verdict(
                Vote.FOR,
                confidence=0.60,
                reasoning=(
                    f"Moderately regime-robust: positive in {positive_quintiles}/5 "
                    f"quintiles. Not all regimes covered — some regime-conditional "
                    f"risk remains."
                ),
                key_concerns=concerns,
            )
        elif positive_quintiles == 2:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.70,
                reasoning=(
                    f"Regime-fragile: positive in only {positive_quintiles}/5 "
                    f"quintiles. High probability of regime-specific artefact."
                ),
                key_concerns=concerns + [
                    "Regime conditional strategy — consider explicit regime filter"
                ],
            )
        else:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.85,
                reasoning=(
                    f"Severely regime-conditional: positive in {positive_quintiles}/5 "
                    f"quintiles. This pattern is not a robust trading edge."
                ),
                key_concerns=concerns,
            )
