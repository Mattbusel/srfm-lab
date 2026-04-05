"""
significance_tester.py
----------------------
Statistical significance testing for A/B experiment results.

Tests implemented
-----------------
1. Two-sample Welch t-test on daily returns
2. Mann-Whitney U test (non-parametric, distribution-free)
3. Bootstrap permutation test (10,000 shuffles)
4. Effect size: Cohen's d

Decision rule (all three must hold to declare a winner):
  * p < 0.05  (any of the three tests)
  * Cohen's d > 0.2 (practical significance)
  * Minimum 50 trades per variant

Multiple-testing correction
---------------------------
When evaluating N simultaneous tests, Bonferroni threshold = 0.05 / N.
Pass n_concurrent_tests to SignificanceTester to activate correction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignificanceResult:
    """
    Full significance test result for one A/B comparison.

    Attributes
    ----------
    winner          : "A" | "B" | "INCONCLUSIVE"
    p_ttest         : p-value from Welch t-test
    p_mannwhitney   : p-value from Mann-Whitney U
    p_bootstrap     : p-value from permutation test
    cohens_d        : effect size (signed, positive = B is better)
    min_p           : minimum of the three p-values
    threshold       : significance threshold used (alpha, Bonferroni-corrected)
    sufficient_data : True if both variants have >= min_trades trades
    reason          : human-readable explanation of the decision
    """
    winner: str
    p_ttest: float
    p_mannwhitney: float
    p_bootstrap: float
    cohens_d: float
    min_p: float
    threshold: float
    sufficient_data: bool
    reason: str

    def is_significant(self) -> bool:
        return (
            self.sufficient_data
            and self.min_p < self.threshold
            and abs(self.cohens_d) > 0.2
        )

    def __str__(self) -> str:
        return (
            f"SignificanceResult(winner={self.winner}, "
            f"min_p={self.min_p:.4f}, d={self.cohens_d:.3f}, "
            f"significant={self.is_significant()})"
        )


# ---------------------------------------------------------------------------
# Core tester
# ---------------------------------------------------------------------------

class SignificanceTester:
    """
    Runs significance tests comparing two sets of daily returns / trade P&Ls.

    Parameters
    ----------
    min_trades          : minimum trades per variant to allow a decision
    alpha               : base significance level (default 0.05)
    min_effect_size     : minimum Cohen's d for practical significance (default 0.2)
    n_concurrent_tests  : number of simultaneous experiments (for Bonferroni)
    n_bootstrap         : number of permutation iterations (default 10_000)
    """

    def __init__(
        self,
        min_trades: int = 50,
        alpha: float = 0.05,
        min_effect_size: float = 0.2,
        n_concurrent_tests: int = 1,
        n_bootstrap: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.min_trades         = min_trades
        self.alpha              = alpha
        self.min_effect_size    = min_effect_size
        self.n_bootstrap        = n_bootstrap
        self.threshold          = alpha / n_concurrent_tests  # Bonferroni
        self._rng               = random.Random(seed)
        self._np_rng            = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def test(
        self,
        returns_a: Sequence[float],
        returns_b: Sequence[float],
        trades_a: int | None = None,
        trades_b: int | None = None,
    ) -> SignificanceResult:
        """
        Compare variant A (control) vs variant B (challenger).

        Parameters
        ----------
        returns_a  : daily returns for variant A
        returns_b  : daily returns for variant B
        trades_a   : number of trades for A (falls back to len(returns_a))
        trades_b   : number of trades for B (falls back to len(returns_b))
        """
        n_a = trades_a if trades_a is not None else len(returns_a)
        n_b = trades_b if trades_b is not None else len(returns_b)
        sufficient = n_a >= self.min_trades and n_b >= self.min_trades

        arr_a = np.asarray(returns_a, dtype=float)
        arr_b = np.asarray(returns_b, dtype=float)

        if len(arr_a) < 2 or len(arr_b) < 2:
            return SignificanceResult(
                winner="INCONCLUSIVE",
                p_ttest=1.0,
                p_mannwhitney=1.0,
                p_bootstrap=1.0,
                cohens_d=0.0,
                min_p=1.0,
                threshold=self.threshold,
                sufficient_data=False,
                reason="Insufficient data for any test",
            )

        p_t   = self._ttest(arr_a, arr_b)
        p_mw  = self._mannwhitney(arr_a, arr_b)
        p_bs  = self._bootstrap(arr_a, arr_b)
        d     = self._cohens_d(arr_a, arr_b)
        min_p = min(p_t, p_mw, p_bs)

        winner, reason = self._decide(min_p, d, sufficient, arr_a, arr_b)

        return SignificanceResult(
            winner=winner,
            p_ttest=p_t,
            p_mannwhitney=p_mw,
            p_bootstrap=p_bs,
            cohens_d=d,
            min_p=min_p,
            threshold=self.threshold,
            sufficient_data=sufficient,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def _ttest(self, a: np.ndarray, b: np.ndarray) -> float:
        """Welch's t-test (unequal variance). Returns two-sided p-value."""
        n_a, n_b = len(a), len(b)
        mean_diff = float(np.mean(b) - np.mean(a))
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))

        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se < 1e-15:
            return 1.0

        t_stat = mean_diff / se

        # Welch-Satterthwaite degrees of freedom
        denom = ((var_a / n_a) ** 2 / (n_a - 1)) + ((var_b / n_b) ** 2 / (n_b - 1))
        if denom < 1e-15:
            df = float(n_a + n_b - 2)
        else:
            df = (var_a / n_a + var_b / n_b) ** 2 / denom

        # Approximate p-value via t-distribution CDF (Abramowitz & Stegun)
        return 2.0 * self._t_cdf_upper(abs(t_stat), df)

    def _mannwhitney(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Mann-Whitney U statistic approximated with normal approximation.
        Suitable for n > ~20 per group.
        """
        n_a, n_b = len(a), len(b)
        # Rank all combined
        combined = np.concatenate([a, b])
        ranks = self._rank(combined)
        rank_a = ranks[:n_a]

        U_a = float(np.sum(rank_a)) - n_a * (n_a + 1) / 2
        U_b = n_a * n_b - U_a

        U = min(U_a, U_b)
        mu_u = n_a * n_b / 2
        sigma_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
        if sigma_u < 1e-9:
            return 1.0

        z = (U - mu_u) / sigma_u
        # Two-sided p via standard normal CDF approximation
        return 2.0 * self._normal_cdf_upper(abs(z))

    def _bootstrap(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Permutation test: shuffle labels 10,000 times, compute p-value for
        observed mean difference being in the tails.
        """
        observed_diff = float(np.mean(b) - np.mean(a))
        combined = np.concatenate([a, b])
        n_a = len(a)
        count_extreme = 0

        for _ in range(self.n_bootstrap):
            self._np_rng.shuffle(combined)
            diff = float(np.mean(combined[n_a:]) - np.mean(combined[:n_a]))
            if abs(diff) >= abs(observed_diff):
                count_extreme += 1

        return (count_extreme + 1) / (self.n_bootstrap + 1)  # +1 for continuity

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cohen's d = (mean_B - mean_A) / pooled_std
        Positive value means B > A.
        """
        mean_diff = float(np.mean(b) - np.mean(a))
        pooled_var = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2
        if pooled_var < 1e-20:
            return 0.0
        return mean_diff / math.sqrt(pooled_var)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide(
        self,
        min_p: float,
        d: float,
        sufficient: bool,
        a: np.ndarray,
        b: np.ndarray,
    ) -> tuple[str, str]:
        if not sufficient:
            return "INCONCLUSIVE", f"Need >= {self.min_trades} trades per variant"

        if min_p >= self.threshold:
            return (
                "INCONCLUSIVE",
                f"p={min_p:.4f} >= threshold={self.threshold:.4f} — not significant",
            )

        if abs(d) < self.min_effect_size:
            return (
                "INCONCLUSIVE",
                f"p={min_p:.4f} significant but |d|={abs(d):.3f} < {self.min_effect_size} — negligible effect",
            )

        if d > 0:
            return "B", f"B wins: p={min_p:.4f}, d={d:.3f} (B mean={np.mean(b):.5f} vs A mean={np.mean(a):.5f})"
        else:
            return "A", f"A wins: p={min_p:.4f}, d={d:.3f} (A mean={np.mean(a):.5f} vs B mean={np.mean(b):.5f})"

    # ------------------------------------------------------------------
    # Math helpers (no scipy dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _rank(arr: np.ndarray) -> np.ndarray:
        """Average-rank (handles ties)."""
        n = len(arr)
        order = np.argsort(arr, kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        # Handle ties: average the ranks
        i = 0
        while i < n:
            j = i + 1
            while j < n and arr[order[j]] == arr[order[i]]:
                j += 1
            if j > i + 1:
                avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2
                ranks[order[i:j]] = avg
            i = j
        return ranks

    @staticmethod
    def _normal_cdf_upper(z: float) -> float:
        """P(Z > z) for standard normal, rational approximation (Abramowitz & Stegun 26.2.17)."""
        if z < 0:
            return 1.0 - SignificanceTester._normal_cdf_upper(-z)
        p = 1.0
        t = 1.0 / (1.0 + 0.2316419 * z)
        poly = t * (0.319381530
               + t * (-0.356563782
               + t * (1.781477937
               + t * (-1.821255978
               + t * 1.330274429))))
        p = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
        return 1.0 - p

    @staticmethod
    def _t_cdf_upper(t: float, df: float) -> float:
        """P(T > t) for t-distribution via incomplete beta function approximation."""
        x = df / (df + t * t)
        # Regularized incomplete beta I_x(df/2, 1/2) using continued fraction
        # For our purposes, a simple normal approximation suffices for large df
        if df > 30:
            return SignificanceTester._normal_cdf_upper(t)
        # Use the relation to the beta distribution CDF
        a_param = df / 2.0
        b_param = 0.5
        ibeta = SignificanceTester._reg_incomplete_beta(x, a_param, b_param)
        return ibeta / 2.0

    @staticmethod
    def _reg_incomplete_beta(x: float, a: float, b: float) -> float:
        """Regularized incomplete beta function via continued fraction (Lentz)."""
        if x < 0 or x > 1:
            return 0.0
        if x == 0:
            return 0.0
        if x == 1:
            return 1.0
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
        # CF via modified Lentz
        tiny = 1e-30
        f = tiny
        C = f
        D = 0.0
        for m in range(200):
            for j in (0, 1):
                if m == 0 and j == 0:
                    d = 1.0
                elif j == 0:
                    d = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
                else:
                    d = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
                D = 1.0 + d * D
                if abs(D) < tiny:
                    D = tiny
                D = 1.0 / D
                C = 1.0 + d / C
                if abs(C) < tiny:
                    C = tiny
                f *= C * D
                if abs(C * D - 1.0) < 1e-8:
                    return front * f
        return front * f
