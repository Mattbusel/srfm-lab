"""
debate-system/agents/statistician.py

StatisticalAnalyst: evaluates hypotheses through the lens of classical and
modern statistical rigor.

Decision rules
--------------
VOTES AGAINST if any hard gate fails:
  - n < 50          (too few observations for reliable inference)
  - p > 0.05        (not significant even before MTC)
  - effect_size (Cohen's d) < 0.1 (effect too small to trade)

VOTES FOR (strong) if:
  - p < 0.01 after Bonferroni correction
  - n > 500
  - Effect is consistent across sub-periods (CV < 0.5)

Otherwise ABSTAIN or soft FOR with reduced confidence.

Multiple-testing correction
---------------------------
IAE generates many hypotheses from the same dataset, so every p-value
is Bonferroni-corrected by the number of simultaneous tests. The number
of tests is passed in market_data['n_tests'] (default 20 if missing).

Data-snooping concerns
----------------------
The statistician also flags look-ahead bias indicators supplied by the
data pipeline (market_data['has_lookahead']) and warns about overfitting
when the number of parameters in the hypothesis is high relative to n.
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis.types import Hypothesis
from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote


class StatisticalAnalyst(BaseAnalyst):
    """
    Specialist in sample statistics, significance testing, and multiple
    testing corrections.  Acts as the first quantitative gate in the
    debate chamber.
    """

    # Hard rejection thresholds
    MIN_N = 50
    STRICT_N = 100
    STRONG_N = 500
    MAX_P_VALUE = 0.05
    STRONG_P_VALUE = 0.01
    MIN_EFFECT_SIZE = 0.1          # Cohen's d
    MEANINGFUL_EFFECT = 0.2        # Cohen's d  (conventional "small" effect)
    MAX_SUB_PERIOD_CV = 0.5        # coefficient of variation across sub-periods

    def __init__(self) -> None:
        super().__init__(
            name="StatisticalAnalyst",
            specialization=(
                "Sample size adequacy, p-value significance, multiple testing "
                "correction, effect size (Cohen's d), data snooping detection"
            ),
        )

    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> AnalystVerdict:
        """
        Run statistical battery on the hypothesis evidence.

        Expected market_data keys
        -------------------------
        n_samples        : int   - number of trade / bar observations
        p_value          : float - raw (uncorrected) p-value
        effect_size      : float - Cohen's d or equivalent
        n_tests          : int   - number of simultaneous tests (for MTC)
        sub_period_pnls  : list[float] - period-level performance (for stability)
        has_lookahead    : bool  - flag from data pipeline
        n_hypothesis_params : int - number of free parameters tested
        """
        n = int(market_data.get("n_samples", 0))
        raw_p = float(market_data.get("p_value", 1.0))
        effect = float(market_data.get("effect_size", 0.0))
        n_tests = int(market_data.get("n_tests", 20))
        sub_periods = list(market_data.get("sub_period_pnls", []))
        has_lookahead = bool(market_data.get("has_lookahead", False))
        n_params = int(market_data.get("n_hypothesis_params", 1))

        concerns: list[str] = []

        # --- Multiple testing correction (Bonferroni) --------------------
        corrected_p = min(1.0, raw_p * n_tests)

        # --- Hard rejections ---------------------------------------------
        if n < self.MIN_N:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.95,
                reasoning=(
                    f"Sample size n={n} is below minimum threshold of "
                    f"{self.MIN_N}. No inference is reliable with this few "
                    f"observations."
                ),
                key_concerns=[
                    f"n={n} < {self.MIN_N}: insufficient sample",
                    "Collect more data before re-evaluating",
                ],
            )

        if corrected_p > self.MAX_P_VALUE:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.90,
                reasoning=(
                    f"Corrected p-value {corrected_p:.4f} (raw {raw_p:.4f}, "
                    f"Bonferroni × {n_tests}) exceeds α=0.05. The hypothesis "
                    f"is not statistically significant after accounting for "
                    f"multiple testing."
                ),
                key_concerns=[
                    f"p_corrected={corrected_p:.4f} > 0.05",
                    f"Raw p={raw_p:.4f} with {n_tests} simultaneous tests",
                    "Multiple comparisons inflate false-positive risk",
                ],
            )

        if effect < self.MIN_EFFECT_SIZE:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.85,
                reasoning=(
                    f"Effect size (Cohen's d={effect:.3f}) is below the minimum "
                    f"tradeable threshold of {self.MIN_EFFECT_SIZE}. Even if "
                    f"statistically real, the effect is too small to survive "
                    f"transaction costs."
                ),
                key_concerns=[
                    f"Cohen's d={effect:.3f} < {self.MIN_EFFECT_SIZE}: sub-tradeable",
                    "Effect may be wiped out by spread, slippage, funding",
                ],
            )

        # --- Conditional concerns ----------------------------------------
        if has_lookahead:
            concerns.append(
                "Data pipeline flagged potential look-ahead bias — "
                "results must be reproduced on strict walk-forward data"
            )

        if n < self.STRICT_N:
            concerns.append(
                f"n={n} is above minimum but below {self.STRICT_N}; "
                f"confidence intervals will be wide"
            )

        over_parameterised = n_params > 0 and (n / n_params) < 20
        if over_parameterised:
            concerns.append(
                f"n/params ratio = {n / n_params:.1f} < 20; high risk of "
                f"in-sample curve-fitting"
            )
            concerns.append(
                "Data snooping concern: more parameters than the 1:20 rule allows"
            )

        # --- Sub-period consistency --------------------------------------
        cv = self._coefficient_of_variation(sub_periods)
        if cv is not None and cv > self.MAX_SUB_PERIOD_CV:
            concerns.append(
                f"Sub-period performance CV={cv:.2f} > {self.MAX_SUB_PERIOD_CV}: "
                f"effect is inconsistent across time periods"
            )

        # --- Positive verdict --------------------------------------------
        is_strong = (
            corrected_p < self.STRONG_P_VALUE
            and n >= self.STRONG_N
            and effect >= self.MEANINGFUL_EFFECT
            and (cv is None or cv <= self.MAX_SUB_PERIOD_CV)
            and not has_lookahead
        )

        if is_strong:
            confidence = min(0.95, 0.7 + 0.1 * math.log10(n / self.STRONG_N + 1))
            vote = Vote.FOR
            reasoning = (
                f"Strong statistical evidence: n={n}, "
                f"p_corrected={corrected_p:.4f}, Cohen's d={effect:.3f}. "
                f"Effect holds across sub-periods (CV={cv:.2f if cv else 'N/A'}). "
                f"Pattern survives Bonferroni correction across {n_tests} tests."
            )
        else:
            # Nominally significant — let other agents decide
            confidence = 0.5 + 0.2 * (self.MAX_P_VALUE - corrected_p) / self.MAX_P_VALUE
            vote = Vote.FOR
            reasoning = (
                f"Nominally significant: n={n}, p_corrected={corrected_p:.4f}, "
                f"Cohen's d={effect:.3f}. Meets minimum thresholds but not "
                f"the strong criteria. Key concerns attached."
            )

        if concerns:
            reasoning += f" | Concerns: {'; '.join(concerns[:2])}"

        return self._make_verdict(vote, confidence, reasoning, concerns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coefficient_of_variation(
        self, values: list[float]
    ) -> float | None:
        """Return |std/mean| for a list of numbers, or None if < 2 items."""
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        if abs(mean) < 1e-12:
            return None
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance) / abs(mean)
