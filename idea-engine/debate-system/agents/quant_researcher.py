"""
debate-system/agents/quant_researcher.py

QuantResearcher: independently reproduces and validates a statistical finding
using different methodology than the original discovery pipeline.

Role in the debate
------------------
The primary statistical pipeline (stats-service) uses one set of methods.
This agent deliberately uses alternative methods to test if the finding is
robust to methodology choice.  If the effect vanishes with a different test,
it was likely a function of the specific methodology, not the underlying market.

Validation methods
------------------
1. Bootstrap confidence intervals (10,000 samples with replacement)
   — Does the bootstrap CI for the mean include zero?

2. Permutation / randomisation test
   — Shuffle timestamps 1,000 times. What fraction of permutations yields
     a higher test statistic? This is the exact p-value under H0: no effect.

3. Out-of-sample test (holdout 20%)
   — Fit on first 80%, test on last 20% (time-ordered split, not random).
   — Report: IS Sharpe, OOS Sharpe, IS/OOS ratio.
   — Ratio < 0.5 suggests overfitting.

4. Walk-forward validation
   — Re-estimate over rolling windows, report % windows with positive return.

Key metric: IS/OOS ratio
------------------------
< 0.5 : severe overfitting — AGAINST
0.5–0.8 : moderate degradation — concerns
> 0.8 : robust generalisation — supports FOR
"""

from __future__ import annotations

import math
import random
from typing import Any

from hypothesis.types import Hypothesis
from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote


class QuantResearcher(BaseAnalyst):
    """
    Independent quant reproducing findings with alternative methodology.
    """

    BOOTSTRAP_N = 10_000
    PERMUTATION_N = 1_000
    OOS_RATIO_FLOOR = 0.50           # IS/OOS ratio below this = overfit
    OOS_RATIO_GOOD = 0.80
    MIN_PERMUTATION_PVALUE = 0.05
    MIN_WF_POSITIVE_FRACTION = 0.60  # walk-forward: % positive windows

    def __init__(self) -> None:
        super().__init__(
            name="QuantResearcher",
            specialization=(
                "Bootstrap confidence intervals, permutation tests, "
                "out-of-sample validation, walk-forward analysis, "
                "in-sample / out-of-sample ratio"
            ),
        )

    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> AnalystVerdict:
        """
        Independently validate the hypothesis.

        Expected market_data keys
        -------------------------
        trade_returns       : list[float] - per-trade returns (full sample)
        is_sharpe           : float       - in-sample Sharpe
        oos_sharpe          : float       - out-of-sample Sharpe
        permutation_pvalue  : float | None - permutation test p-value
        wf_positive_frac    : float | None - walk-forward % positive windows
        bootstrap_ci_lower  : float | None - pre-computed bootstrap CI lower bound
        bootstrap_ci_upper  : float | None - pre-computed bootstrap CI upper bound
        n_wf_windows        : int | None  - number of walk-forward windows
        """
        trade_returns: list[float] = list(market_data.get("trade_returns", []))
        is_sharpe: float = float(market_data.get("is_sharpe", 0.0))
        oos_sharpe: float = float(market_data.get("oos_sharpe", 0.0))
        perm_p: float | None = market_data.get("permutation_pvalue")
        wf_pos_frac: float | None = market_data.get("wf_positive_frac")
        boot_lower: float | None = market_data.get("bootstrap_ci_lower")
        boot_upper: float | None = market_data.get("bootstrap_ci_upper")

        concerns: list[str] = []
        passed_checks: list[str] = []
        failed_checks: list[str] = []

        # --- Compute bootstrap CI if not pre-supplied -------------------
        if boot_lower is None and len(trade_returns) >= 30:
            boot_lower, boot_upper = self._bootstrap_mean_ci(
                trade_returns, n_samples=self.BOOTSTRAP_N
            )

        # --- Bootstrap CI check -----------------------------------------
        if boot_lower is not None and boot_upper is not None:
            if boot_lower > 0:
                passed_checks.append(
                    f"Bootstrap 95% CI [{boot_lower:.5f}, {boot_upper:.5f}] "
                    f"entirely positive — strong evidence of non-zero mean."
                )
            elif boot_upper < 0:
                failed_checks.append(
                    f"Bootstrap 95% CI [{boot_lower:.5f}, {boot_upper:.5f}] "
                    f"entirely negative — effect does not exist."
                )
                concerns.append(
                    f"Bootstrap CI entirely negative: [{boot_lower:.5f}, {boot_upper:.5f}]"
                )
            else:
                concerns.append(
                    f"Bootstrap CI [{boot_lower:.5f}, {boot_upper:.5f}] straddles "
                    f"zero — mean return not reliably different from zero."
                )
                failed_checks.append("Bootstrap CI includes zero")

        # --- Permutation test check -------------------------------------
        if perm_p is not None:
            if perm_p < self.MIN_PERMUTATION_PVALUE:
                passed_checks.append(
                    f"Permutation test p={perm_p:.4f} < 0.05 — effect is "
                    f"unlikely under the null hypothesis of no temporal structure."
                )
            else:
                failed_checks.append(
                    f"Permutation test p={perm_p:.4f} ≥ 0.05 — cannot reject "
                    f"null hypothesis using exact permutation method."
                )
                concerns.append(f"Permutation test fails: p={perm_p:.4f}")
        elif len(trade_returns) >= 30:
            # Compute permutation p-value from raw returns
            perm_p = self._permutation_pvalue(trade_returns, n_perm=self.PERMUTATION_N)
            if perm_p < self.MIN_PERMUTATION_PVALUE:
                passed_checks.append(
                    f"Computed permutation test p={perm_p:.4f} — passes."
                )
            else:
                failed_checks.append(f"Computed permutation p={perm_p:.4f} — fails.")
                concerns.append(f"Permutation test p={perm_p:.4f} >= 0.05")

        # --- IS/OOS ratio check -----------------------------------------
        oos_ratio: float | None = None
        if is_sharpe > 0.01:   # avoid division by near-zero
            oos_ratio = oos_sharpe / is_sharpe
            if oos_ratio >= self.OOS_RATIO_GOOD:
                passed_checks.append(
                    f"IS/OOS Sharpe ratio = {oos_ratio:.2f} (IS={is_sharpe:.2f}, "
                    f"OOS={oos_sharpe:.2f}) — minimal overfitting."
                )
            elif oos_ratio >= self.OOS_RATIO_FLOOR:
                concerns.append(
                    f"Moderate overfitting: IS/OOS ratio = {oos_ratio:.2f}. "
                    f"OOS Sharpe ({oos_sharpe:.2f}) is {1-oos_ratio:.0%} below IS ({is_sharpe:.2f})."
                )
            else:
                failed_checks.append(
                    f"SEVERE OVERFITTING: IS/OOS ratio = {oos_ratio:.2f}. "
                    f"OOS Sharpe ({oos_sharpe:.2f}) collapses vs IS ({is_sharpe:.2f}). "
                    f"Strategy does not generalise."
                )
                concerns.append(f"IS/OOS ratio {oos_ratio:.2f} < {self.OOS_RATIO_FLOOR}")
        elif is_sharpe <= 0:
            failed_checks.append(f"In-sample Sharpe is non-positive ({is_sharpe:.3f}).")
            concerns.append("Non-positive in-sample Sharpe")

        # --- Walk-forward check -----------------------------------------
        if wf_pos_frac is not None:
            if wf_pos_frac >= self.MIN_WF_POSITIVE_FRACTION:
                passed_checks.append(
                    f"Walk-forward validation: {wf_pos_frac:.0%} of windows positive."
                )
            else:
                failed_checks.append(
                    f"Walk-forward: only {wf_pos_frac:.0%} of windows positive "
                    f"(threshold: {self.MIN_WF_POSITIVE_FRACTION:.0%})."
                )
                concerns.append(f"Walk-forward positive fraction {wf_pos_frac:.0%} too low")

        # --- Verdict determination ---------------------------------------
        n_pass = len(passed_checks)
        n_fail = len(failed_checks)

        if n_fail == 0 and n_pass >= 2:
            return self._make_verdict(
                Vote.FOR,
                confidence=min(0.90, 0.65 + 0.08 * n_pass),
                reasoning=(
                    f"Independent validation passed {n_pass} checks: "
                    + "; ".join(passed_checks[:2])
                ),
                key_concerns=concerns,
            )
        elif n_fail == 0 and n_pass == 1:
            return self._make_verdict(
                Vote.FOR,
                confidence=0.60,
                reasoning=(
                    f"Passed {n_pass} independent check(s), none failed. "
                    f"Limited validation data available."
                ),
                key_concerns=concerns,
            )
        elif n_fail >= 2:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.75 + 0.05 * n_fail,
                reasoning=(
                    f"Independent validation failed {n_fail} check(s): "
                    + "; ".join(failed_checks[:2])
                ),
                key_concerns=concerns,
            )
        else:
            return self._make_verdict(
                Vote.ABSTAIN,
                confidence=0.55,
                reasoning=(
                    f"Mixed results: {n_pass} passed, {n_fail} failed. "
                    f"Cannot confidently confirm or deny hypothesis."
                ),
                key_concerns=concerns,
            )

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    def _bootstrap_mean_ci(
        self,
        returns: list[float],
        n_samples: int = 10_000,
        ci: float = 0.95,
    ) -> tuple[float, float]:
        """
        Non-parametric bootstrap 95% CI for the mean of `returns`.
        Samples with replacement; returns (lower, upper) bounds.
        """
        n = len(returns)
        rng = random.Random(42)   # deterministic seed for reproducibility
        bootstrap_means: list[float] = []
        for _ in range(n_samples):
            sample = [rng.choice(returns) for _ in range(n)]
            bootstrap_means.append(sum(sample) / n)
        bootstrap_means.sort()
        alpha = (1.0 - ci) / 2.0
        lo_idx = int(alpha * n_samples)
        hi_idx = int((1.0 - alpha) * n_samples)
        return bootstrap_means[lo_idx], bootstrap_means[hi_idx]

    def _permutation_pvalue(
        self,
        returns: list[float],
        n_perm: int = 1_000,
    ) -> float:
        """
        Exact permutation test: how often does a random shuffle produce
        a mean >= observed mean?  Returns p-value (fraction of permutations
        at least as extreme as observed).
        """
        observed_mean = sum(returns) / len(returns)
        rng = random.Random(99)
        extreme_count = 0
        shuffled = list(returns)
        for _ in range(n_perm):
            rng.shuffle(shuffled)
            perm_mean = sum(shuffled) / len(shuffled)
            if perm_mean >= observed_mean:
                extreme_count += 1
        return extreme_count / n_perm
