"""
research/validation/performance_persistence.py -- performance persistence analysis.

Tests whether strategy or signal performance is persistent (skill) vs episodic (luck).
Key question: does a signal that worked in one period keep working in the next?

Methods:
  - Contingency table / chi-square test for winner/loser persistence
  - Spearman rank IC between periods
  - Information ratio stability (variance of rolling IR)
  - Regime-conditional performance F-test
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContingencyResult:
    """Result of 2x2 winner/loser persistence contingency table."""
    wins_persist: int    # WW: winner in t1 and t2
    wins_switch: int     # WL: winner in t1, loser in t2
    loses_persist: int   # LL: loser in t1 and t2
    loses_switch: int    # LW: loser in t1, winner in t2
    chi2: float          # Chi-square test for independence (H0: no persistence)
    p_value: float
    percent_persist: float  # (WW + LL) / total: fraction that persisted
    odds_ratio: float       # (WW * LL) / (WL * LW): odds of persistence
    reject_no_persistence: bool


@dataclass
class IRStabilityResult:
    """Result of information ratio stability analysis."""
    rolling_irs: List[float]
    mean_ir: float
    std_ir: float
    cv_ir: float          # Coefficient of variation = std / |mean|
    is_stable: bool       # CV < 1.0 suggests stable signal (rough heuristic)
    min_ir: float
    max_ir: float
    pct_positive: float   # Fraction of periods with positive IR
    # Stability score: 1.0 = perfectly stable, 0.0 = highly variable
    stability_score: float


@dataclass
class RegimePersistenceResult:
    """Result of regime-conditional performance test."""
    regime_means: Dict[str, float]   # Mean return per regime
    regime_stds: Dict[str, float]    # Std return per regime
    regime_ns: Dict[str, int]        # Number of periods per regime
    f_stat: float                    # F-stat for between-regime mean differences
    p_value: float                   # H0: equal means across regimes
    is_consistent: bool              # True if performance is similar across regimes
    best_regime: str
    worst_regime: str
    regime_sharpes: Dict[str, float]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PerformancePersistenceAnalyzer:
    """
    Analyzes whether strategy performance is skill-based (persistent) or luck-based (episodic).

    The key insight is that skill should produce:
      1. Winner persistence: good signals stay good across non-overlapping periods
      2. Stable IR: information ratio should have low variance relative to its mean
      3. Cross-regime consistency: performance should hold across market regimes
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Contingency Table
    # ------------------------------------------------------------------

    def contingency_table(
        self,
        scores_t1: pd.Series,
        scores_t2: pd.Series,
        threshold: float = 0.0,
        use_median_split: bool = True,
    ) -> ContingencyResult:
        """
        Build 2x2 winner/loser persistence table and test for persistence.

        Units are classified as "winners" or "losers" in each period based on
        whether their score exceeds the threshold. The chi-square test checks
        whether winners in t1 are disproportionately winners in t2.

        Parameters
        ----------
        scores_t1 : pd.Series
            Scores in period 1 (e.g., IC, Sharpe, raw return).
        scores_t2 : pd.Series
            Scores in period 2 (same units as scores_t1, must be aligned).
        threshold : float
            Classification threshold. Scores above threshold = winner.
            Ignored if use_median_split=True.
        use_median_split : bool
            If True, use the median as the threshold for each period separately.
            This ensures exactly 50/50 winners/losers per period.

        Returns
        -------
        ContingencyResult
        """
        # Align
        both = pd.concat([scores_t1, scores_t2], axis=1).dropna()
        if len(both) < 4:
            raise ValueError(f"Need at least 4 paired observations, got {len(both)}")

        s1 = both.iloc[:, 0].values.astype(float)
        s2 = both.iloc[:, 1].values.astype(float)

        if use_median_split:
            t1_thresh = float(np.median(s1))
            t2_thresh = float(np.median(s2))
        else:
            t1_thresh = threshold
            t2_thresh = threshold

        w1 = s1 > t1_thresh  # winners in period 1
        w2 = s2 > t2_thresh  # winners in period 2

        ww = int(np.sum(w1 & w2))     # winner -> winner
        wl = int(np.sum(w1 & ~w2))    # winner -> loser
        lw = int(np.sum(~w1 & w2))    # loser -> winner
        ll = int(np.sum(~w1 & ~w2))   # loser -> loser

        total = ww + wl + lw + ll
        if total == 0:
            raise ValueError("No valid pairs in contingency table")

        # Chi-square test for independence (2x2 contingency table)
        contingency_matrix = np.array([[ww, wl], [lw, ll]], dtype=float)
        if np.any(contingency_matrix < 5):
            # Use Fisher's exact test for small samples
            _, p_val = stats.fisher_exact(contingency_matrix)
            # Approximate chi2 from Fisher p
            chi2 = float(stats.chi2.ppf(1 - p_val, df=1)) if p_val < 1.0 else 0.0
        else:
            chi2_result = stats.chi2_contingency(contingency_matrix, correction=False)
            chi2 = float(chi2_result[0])
            p_val = float(chi2_result[1])

        pct_persist = (ww + ll) / total if total > 0 else 0.0
        # Odds ratio: (WW * LL) / (WL * LW), undefined if any cell is 0
        if wl > 0 and lw > 0:
            odds_ratio = (ww * ll) / (wl * lw)
        elif wl == 0 and lw == 0:
            odds_ratio = float("inf")
        else:
            odds_ratio = float(ww * ll) / max(float(wl * lw), 1e-10)

        return ContingencyResult(
            wins_persist=ww,
            wins_switch=wl,
            loses_persist=ll,
            loses_switch=lw,
            chi2=chi2,
            p_value=p_val,
            percent_persist=pct_persist,
            odds_ratio=float(odds_ratio),
            reject_no_persistence=p_val < self.alpha,
        )

    # ------------------------------------------------------------------
    # Spearman Rank Correlation
    # ------------------------------------------------------------------

    def spearman_rank_correlation(
        self,
        scores_t1: pd.Series,
        scores_t2: pd.Series,
    ) -> float:
        """
        Compute Spearman rank IC between scores in two non-overlapping periods.

        A high rank IC (> 0.3) between periods suggests the signal is persistent.
        A near-zero rank IC suggests luck dominates.

        Parameters
        ----------
        scores_t1 : pd.Series
            Scores in period 1.
        scores_t2 : pd.Series
            Scores in period 2 (aligned with scores_t1).

        Returns
        -------
        float
            Spearman rank correlation coefficient (-1 to 1).
        """
        both = pd.concat([scores_t1, scores_t2], axis=1).dropna()
        if len(both) < 4:
            raise ValueError(f"Need at least 4 paired observations, got {len(both)}")

        s1 = both.iloc[:, 0].values.astype(float)
        s2 = both.iloc[:, 1].values.astype(float)

        corr, _ = stats.spearmanr(s1, s2)
        return float(corr)

    # ------------------------------------------------------------------
    # Information Ratio Stability
    # ------------------------------------------------------------------

    def information_ratio_stability(
        self,
        ir_series: List[float],
        window: int = 12,
    ) -> IRStabilityResult:
        """
        Analyze whether the information ratio is stable or highly variable.

        A stable IR (low CV) suggests genuine skill. A variable IR (high CV)
        suggests the strategy's performance is sensitive to period selection
        (possibly lucky timing or regime-dependent).

        Key heuristic from Grinold & Kahn:
          CV < 0.5: very stable (consistent skill)
          0.5 <= CV < 1.0: moderately stable
          CV >= 1.0: unstable (luck or regime-dependent)

        Parameters
        ----------
        ir_series : List[float]
            Time series of information ratio observations
            (e.g., quarterly or monthly IRs).
        window : int
            Rolling window for computing rolling mean/std of IR.

        Returns
        -------
        IRStabilityResult
        """
        arr = np.array([x for x in ir_series if np.isfinite(x)])
        if len(arr) < 4:
            raise ValueError(f"Need at least 4 finite IR observations, got {len(arr)}")

        mean_ir = float(np.mean(arr))
        std_ir = float(np.std(arr, ddof=1))
        cv_ir = std_ir / abs(mean_ir) if abs(mean_ir) > 1e-10 else float("inf")

        min_ir = float(np.min(arr))
        max_ir = float(np.max(arr))
        pct_positive = float(np.mean(arr > 0))

        # Rolling IR (rolling mean / rolling std, using sub-windows)
        rolling_irs: List[float] = []
        for i in range(window, len(arr) + 1):
            window_data = arr[i - window: i]
            w_mean = float(np.mean(window_data))
            w_std = float(np.std(window_data, ddof=1))
            roll_ir = w_mean / w_std if w_std > 1e-10 else 0.0
            rolling_irs.append(roll_ir)

        # Stability score: 1 - tanh(CV) maps [0, inf) to (0, 1]
        # 0 CV -> score 1.0 (perfectly stable)
        # CV = 1 -> score ~0.24 (unstable)
        if np.isfinite(cv_ir):
            stability_score = float(1.0 - np.tanh(cv_ir))
        else:
            stability_score = 0.0

        is_stable = cv_ir < 1.0 and mean_ir > 0

        return IRStabilityResult(
            rolling_irs=rolling_irs,
            mean_ir=mean_ir,
            std_ir=std_ir,
            cv_ir=float(cv_ir),
            is_stable=is_stable,
            min_ir=min_ir,
            max_ir=max_ir,
            pct_positive=pct_positive,
            stability_score=stability_score,
        )

    # ------------------------------------------------------------------
    # Regime Persistence Test
    # ------------------------------------------------------------------

    def regime_persistence_test(
        self,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> RegimePersistenceResult:
        """
        Test whether strategy returns are consistent across market regimes.

        Uses one-way ANOVA (F-test) to test H0: equal mean returns across regimes.
        Rejection means returns differ significantly by regime (regime-dependent).
        Non-rejection is consistent with (but does not prove) regime-independent performance.

        Parameters
        ----------
        returns : pd.Series
            Strategy return or score time series.
        regimes : pd.Series
            Regime labels (categorical). Must be aligned with returns.
            Example values: "bull", "bear", "neutral" or integer labels.

        Returns
        -------
        RegimePersistenceResult
        """
        both = pd.concat([returns, regimes], axis=1).dropna()
        if len(both) < 10:
            raise ValueError(f"Need at least 10 observations, got {len(both)}")

        r_vals = both.iloc[:, 0].values.astype(float)
        regime_labels = both.iloc[:, 1].values

        unique_regimes = np.unique(regime_labels)
        if len(unique_regimes) < 2:
            raise ValueError("Need at least 2 distinct regimes for F-test")

        groups: Dict[str, np.ndarray] = {}
        for reg in unique_regimes:
            mask = regime_labels == reg
            groups[str(reg)] = r_vals[mask]

        regime_means: Dict[str, float] = {}
        regime_stds: Dict[str, float] = {}
        regime_ns: Dict[str, int] = {}
        regime_sharpes: Dict[str, float] = {}

        for reg_name, group in groups.items():
            n_group = len(group)
            mu = float(np.mean(group))
            sigma = float(np.std(group, ddof=1)) if n_group > 1 else 0.0
            regime_means[reg_name] = mu
            regime_stds[reg_name] = sigma
            regime_ns[reg_name] = n_group
            regime_sharpes[reg_name] = mu / sigma if sigma > 1e-10 else 0.0

        # One-way ANOVA F-test
        group_arrays = [g for g in groups.values() if len(g) >= 2]
        if len(group_arrays) < 2:
            raise ValueError("At least 2 regimes must have >= 2 observations each for F-test")

        f_stat, p_value = stats.f_oneway(*group_arrays)
        f_stat = float(f_stat)
        p_value = float(p_value)

        # Performance is "consistent" if we cannot reject equal means
        is_consistent = p_value >= self.alpha

        best_regime = max(regime_sharpes, key=lambda k: regime_sharpes[k])
        worst_regime = min(regime_sharpes, key=lambda k: regime_sharpes[k])

        return RegimePersistenceResult(
            regime_means=regime_means,
            regime_stds=regime_stds,
            regime_ns=regime_ns,
            f_stat=f_stat,
            p_value=p_value,
            is_consistent=is_consistent,
            best_regime=best_regime,
            worst_regime=worst_regime,
            regime_sharpes=regime_sharpes,
        )

    # ------------------------------------------------------------------
    # Rolling period persistence summary
    # ------------------------------------------------------------------

    def rolling_period_persistence(
        self,
        scores: pd.Series,
        period_length: int = 63,
        min_periods: int = 3,
    ) -> Dict[str, object]:
        """
        Test persistence by splitting scores into non-overlapping periods and
        testing all adjacent period pairs.

        Parameters
        ----------
        scores : pd.Series
            Time series of signal scores (daily or other frequency).
        period_length : int
            Length of each non-overlapping period.
        min_periods : int
            Minimum number of periods required.

        Returns
        -------
        dict with keys:
          period_scores: List[float] -- mean score per period
          contingency: ContingencyResult -- based on adjacent period comparisons
          spearman_corrs: List[float] -- rank IC between all adjacent pairs
          mean_spearman: float
          persistence_signal: bool -- True if evidence of significant persistence
        """
        arr = np.asarray(scores, dtype=float)
        valid = arr[np.isfinite(arr)]
        n = len(valid)

        n_periods = n // period_length
        if n_periods < min_periods:
            raise ValueError(
                f"Only {n_periods} periods (need {min_periods}). "
                f"Reduce period_length or provide more data."
            )

        period_scores: List[float] = []
        for i in range(n_periods):
            chunk = valid[i * period_length: (i + 1) * period_length]
            period_scores.append(float(np.mean(chunk)))

        # Test adjacent period persistence
        s1 = pd.Series(period_scores[:-1])
        s2 = pd.Series(period_scores[1:])

        contingency = self.contingency_table(s1, s2)
        spearman_corr = self.spearman_rank_correlation(s1, s2)

        # Pairwise Spearman between all adjacent pairs (using raw scores, not means)
        spearman_corrs: List[float] = []
        for i in range(n_periods - 1):
            chunk1 = valid[i * period_length: (i + 1) * period_length]
            chunk2 = valid[(i + 1) * period_length: (i + 2) * period_length]
            min_len = min(len(chunk1), len(chunk2))
            if min_len >= 5:
                c, _ = stats.spearmanr(chunk1[:min_len], chunk2[:min_len])
                if np.isfinite(c):
                    spearman_corrs.append(float(c))

        mean_spearman = float(np.mean(spearman_corrs)) if spearman_corrs else 0.0

        # Overall persistence signal: contingency test rejects AND mean spearman > 0.1
        persistence_signal = contingency.reject_no_persistence and mean_spearman > 0.1

        return {
            "period_scores": period_scores,
            "contingency": contingency,
            "spearman_corrs": spearman_corrs,
            "mean_spearman": mean_spearman,
            "overall_spearman": spearman_corr,
            "persistence_signal": persistence_signal,
        }
