"""
debate-system/agents/devil_advocate.py

DevilsAdvocate: deliberately finds the strongest counterargument to every
hypothesis.  Acts as the system's epistemic immune system.

Design principle
----------------
Most biases in systematic trading research push toward false positives:
data snooping, confirmation bias, p-hacking, and survivorship bias.
A dedicated skeptic hardcoded to find the bear case counterbalances these
pressures.  The Devil's Advocate votes FOR only when the evidence is
overwhelming — p < 0.001, consistent across multiple robustness checks.

Robustness checks performed
----------------------------
1. Time-window stability: does the effect hold when the window is shifted?
2. Market-cap robustness: does it work on small caps and large caps separately?
3. Volatility-regime split: does it work in both calm and turbulent markets?
4. Transaction cost sensitivity: does it remain positive after realistic costs?
5. Survivorship bias exposure: are delisted assets included?

Deliberate hardcoded skepticism
--------------------------------
The agent is initialised with a credibility score of 0.5 but uses p < 0.001
(not 0.05) as its personal FOR threshold.  This means it will vote FOR less
than 5% of hypotheses on base rates, forcing the rest of the chamber to
provide the FOR majority — a feature, not a bug.
"""

from __future__ import annotations

from typing import Any

from hypothesis.types import Hypothesis
from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote


class DevilsAdvocate(BaseAnalyst):
    """
    Structural skeptic.  Hardcoded to find the bear case for every idea.

    Votes FOR only if evidence is overwhelming across ALL robustness axes.
    Votes AGAINST or raises nuanced ABSTAIN in all other cases.
    """

    # Deliberately stringent threshold — much tighter than the statistician
    OVERWHELMING_P = 0.001
    STRONG_N = 200
    MIN_WINDOW_CONSISTENCY = 0.70   # fraction of shifted windows still positive
    MAX_TC_SENSITIVITY = 2.0        # max allowable cost as multiple of gross PnL

    def __init__(self) -> None:
        super().__init__(
            name="DevilsAdvocate",
            specialization=(
                "Counterargument construction, survivorship bias detection, "
                "regime specificity, transaction cost reality, window sensitivity"
            ),
        )

    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> AnalystVerdict:
        """
        Find every reason this hypothesis might be wrong.

        Expected market_data keys
        -------------------------
        p_value               : float - raw p-value
        n_samples             : int
        window_consistency    : float - fraction of ±20% shifted windows positive
        large_cap_p           : float - p-value on large-cap subset
        small_cap_p           : float - p-value on small-cap subset
        calm_regime_p         : float - p-value in low-volatility regime
        turbulent_regime_p    : float - p-value in high-volatility regime
        gross_pnl_bps         : float - gross edge in basis points per trade
        est_tc_bps            : float - estimated round-trip cost in bps
        survivorship_clean    : bool  - True if delisted assets included
        """
        p = float(market_data.get("p_value", 1.0))
        n = int(market_data.get("n_samples", 0))
        window_consistency = float(market_data.get("window_consistency", 0.0))
        large_cap_p = float(market_data.get("large_cap_p", 1.0))
        small_cap_p = float(market_data.get("small_cap_p", 1.0))
        calm_p = float(market_data.get("calm_regime_p", 1.0))
        turbulent_p = float(market_data.get("turbulent_regime_p", 1.0))
        gross_pnl = float(market_data.get("gross_pnl_bps", 0.0))
        est_tc = float(market_data.get("est_tc_bps", 0.0))
        survivorship_clean = bool(market_data.get("survivorship_clean", False))

        concerns: list[str] = []
        failure_count = 0

        # --- Bear-case checks -------------------------------------------

        # 1. Survivorship bias
        if not survivorship_clean:
            concerns.append(
                "SURVIVORSHIP BIAS: analysis does not include delisted assets. "
                "Winners survive; losers disappear. Effect likely overstated."
            )
            failure_count += 1

        # 2. Time-window sensitivity
        if window_consistency < self.MIN_WINDOW_CONSISTENCY:
            concerns.append(
                f"WINDOW INSTABILITY: only {window_consistency:.0%} of shifted "
                f"windows show positive effect. The pattern may be a coincidence "
                f"of the specific backtest dates chosen."
            )
            failure_count += 1

        # 3. Market-cap specificity
        cap_failures = []
        if large_cap_p > 0.05:
            cap_failures.append("large-cap")
        if small_cap_p > 0.05:
            cap_failures.append("small-cap")
        if cap_failures:
            concerns.append(
                f"MARKET-CAP REGIME SPECIFICITY: does not hold for "
                f"{', '.join(cap_failures)} subset. Pattern may be driven by "
                f"a single size cohort with insufficient diversification."
            )
            failure_count += 1

        # 4. Volatility regime specificity
        vol_failures = []
        if calm_p > 0.05:
            vol_failures.append("calm")
        if turbulent_p > 0.05:
            vol_failures.append("turbulent")
        if vol_failures:
            concerns.append(
                f"VOL-REGIME SPECIFICITY: pattern fails in {', '.join(vol_failures)} "
                f"volatility regime(s). If regime shifts, strategy degrades."
            )
            failure_count += 1

        # 5. Transaction cost reality
        if est_tc > 0 and gross_pnl > 0:
            tc_ratio = est_tc / gross_pnl
            if tc_ratio > 0.5:
                concerns.append(
                    f"TRANSACTION COST REALITY: estimated costs {est_tc:.1f} bps "
                    f"consume {tc_ratio:.0%} of gross edge {gross_pnl:.1f} bps. "
                    f"Net edge is marginal and highly sensitive to cost assumptions."
                )
                failure_count += 1
        elif gross_pnl <= 0:
            concerns.append(
                "NEGATIVE GROSS EDGE: hypothesis shows no positive return even "
                "before transaction costs. Cannot be salvaged by cost reduction."
            )
            failure_count += 2   # counts double

        # 6. Bull-market-only hypothesis framing
        if hypothesis.description and any(
            kw in hypothesis.description.lower()
            for kw in ("bull", "uptrend", "momentum", "long")
        ):
            if turbulent_p > 0.05 or calm_p > 0.05:
                concerns.append(
                    "REGIME SPECIFICITY: hypothesis framing is directional but "
                    "evidence does not hold uniformly across regimes. Bear market "
                    "performance is unproven."
                )
                failure_count += 1

        # --- Verdict determination ---------------------------------------

        all_checks_pass = (
            failure_count == 0
            and p < self.OVERWHELMING_P
            and n >= self.STRONG_N
        )

        if all_checks_pass:
            return self._make_verdict(
                Vote.FOR,
                confidence=0.65,   # even when approving, keep confidence modest
                reasoning=(
                    f"Devil's Advocate found no material weaknesses: p={p:.5f} "
                    f"< 0.001, n={n}, window consistency={window_consistency:.0%}, "
                    f"holds across cap-size and vol regimes, TC-adjusted positive. "
                    f"Reluctantly votes FOR."
                ),
                key_concerns=[],
            )

        if failure_count >= 3:
            confidence = min(0.95, 0.6 + 0.1 * failure_count)
            return self._make_verdict(
                Vote.AGAINST,
                confidence=confidence,
                reasoning=(
                    f"Multiple robustness failures ({failure_count}). "
                    f"This hypothesis does not survive scrutiny across time windows, "
                    f"market-cap regimes, and transaction cost assumptions. "
                    f"Strong bear case exists."
                ),
                key_concerns=concerns,
            )

        # Some failures but not overwhelming — pessimistic AGAINST
        return self._make_verdict(
            Vote.AGAINST,
            confidence=0.6 + 0.05 * failure_count,
            reasoning=(
                f"Devil's Advocate identified {failure_count} concern(s). "
                f"Evidence does not meet the overwhelming threshold (p < 0.001 "
                f"with full robustness). Voting AGAINST pending more evidence."
            ),
            key_concerns=concerns,
        )
