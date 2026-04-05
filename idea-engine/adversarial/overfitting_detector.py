"""
overfitting_detector.py
=======================
Detect overfitting in IAE hypotheses using walk-forward validation and
Combinatorial Purged Cross-Validation (CPCV).

Motivation
----------
The IAE system generates hypotheses by optimising parameters on
historical data.  Even with out-of-sample walk-forward testing, there is
a risk that:
1. The in-sample period "leaks" information into the out-of-sample period
   (look-ahead bias).
2. Multiple testing inflates apparent Sharpe ratios
   (the "overfitting via backtest" problem -- De Prado 2018).
3. Short samples have high variance, making real improvements
   indistinguishable from noise.

Methods
-------
**Walk-forward validation**
    For each IAE hypothesis with an associated P&L series:
    1. Compute in-sample Sharpe improvement vs baseline.
    2. Compute out-of-sample Sharpe improvement on the next time block.
    3. Flag if in-sample improvement > 2x out-of-sample improvement.

**Combinatorial Purged Cross-Validation (CPCV)**
    Split the sample into k groups.  For each combination of k-2 groups
    as "training" and the remaining 2 as "test", compute Sharpe.
    Aggregate to get a distribution of Sharpe estimates unbiased by
    backtest overfitting.  See De Prado (2018) Advances in Financial
    Machine Learning, Ch. 12.

    CPCV controls for:
    - Lookahead: purge gap between train and test.
    - Embargo: exclude trades near the train/test boundary.

Usage::

    detector = OverfittingDetector(pnl_series=pnl_arr, n_splits=6)
    report   = detector.run()
    print(report.summary())
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """
    Annualised Sharpe ratio from a series of per-trade returns.

    Parameters
    ----------
    returns          : 1-D array of fractional returns.
    periods_per_year : number of returns per year (default 252 daily).

    Returns
    -------
    Sharpe ratio (NaN if std is 0).
    """
    if len(returns) < 2:
        return float("nan")
    mu  = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return float("nan")
    return mu / std * math.sqrt(periods_per_year)


def _sharpe_improvement(
    baseline: np.ndarray,
    hypothesis: np.ndarray,
    periods_per_year: float = 252.0,
) -> float:
    """
    Improvement in Sharpe from *baseline* to *hypothesis*.

    Returns Sharpe(hypothesis) - Sharpe(baseline).
    """
    return _sharpe(hypothesis, periods_per_year) - _sharpe(baseline, periods_per_year)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class HypothesisValidation:
    """Walk-forward validation result for a single hypothesis."""

    hypothesis_id:       str
    in_sample_sharpe:    float
    out_of_sample_sharpe: float
    sharpe_ratio:        float    # in-sample / out-of-sample improvement ratio
    is_overfit:          bool
    n_in_sample:         int
    n_out_of_sample:     int

    def flag_reason(self) -> str:
        if self.is_overfit:
            return (
                f"IS improvement {self.in_sample_sharpe:+.3f} is "
                f"{self.sharpe_ratio:.1f}x the OOS improvement "
                f"{self.out_of_sample_sharpe:+.3f}"
            )
        return "OK"


@dataclass
class CPCVResult:
    """
    Result of Combinatorial Purged Cross-Validation.

    Attributes
    ----------
    n_splits            : number of data splits.
    n_combinations      : number of train/test combinations evaluated.
    sharpe_distribution : array of Sharpe estimates across combinations.
    mean_sharpe         : mean Sharpe across combinations.
    std_sharpe          : std of Sharpe across combinations.
    sharpe_p25          : 25th-percentile Sharpe (conservative estimate).
    sharpe_p75          : 75th-percentile Sharpe.
    is_robust           : True if p25 > 0 (strategy is profitable in 75%+ of folds).
    probability_overfit : P(IS Sharpe > OOS Sharpe) across all folds.
    """

    n_splits:            int
    n_combinations:      int
    sharpe_distribution: np.ndarray
    mean_sharpe:         float
    std_sharpe:          float
    sharpe_p25:          float
    sharpe_p75:          float
    is_robust:           bool
    probability_overfit: float


@dataclass
class OverfittingReport:
    """
    Full overfitting detection report.

    Attributes
    ----------
    hypothesis_validations: walk-forward results per hypothesis.
    cpcv_result           : CPCV result for the main P&L series.
    n_overfit             : number of overfitted hypotheses.
    overall_overfit_score : fraction of hypotheses that are overfit.
    recommendation        : plain-English recommendation.
    """

    hypothesis_validations: List[HypothesisValidation]
    cpcv_result:            CPCVResult
    n_overfit:              int
    overall_overfit_score:  float
    recommendation:         str

    def summary(self) -> str:
        lines = [
            "=== OVERFITTING DETECTION REPORT ===",
            f"Hypotheses tested: {len(self.hypothesis_validations)}",
            f"Overfit hypotheses: {self.n_overfit} ({self.overall_overfit_score:.1%})",
            "",
            "CPCV Results:",
            f"  Mean Sharpe:  {self.cpcv_result.mean_sharpe:+.3f}",
            f"  Std Sharpe:   {self.cpcv_result.std_sharpe:.3f}",
            f"  P25 Sharpe:   {self.cpcv_result.sharpe_p25:+.3f}",
            f"  Robust:       {'YES' if self.cpcv_result.is_robust else 'NO'}",
            f"  P(overfit):   {self.cpcv_result.probability_overfit:.2%}",
            "",
            f"Recommendation: {self.recommendation}",
        ]
        if self.hypothesis_validations:
            lines.append("")
            lines.append("Walk-forward per hypothesis:")
            for v in self.hypothesis_validations:
                flag = "OVERFIT" if v.is_overfit else "ok"
                lines.append(
                    f"  [{flag}] {v.hypothesis_id}: "
                    f"IS={v.in_sample_sharpe:+.3f} OOS={v.out_of_sample_sharpe:+.3f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OverfittingDetector
# ---------------------------------------------------------------------------

class OverfittingDetector:
    """
    Detect overfitting in backtested hypotheses.

    Parameters
    ----------
    pnl_series         : array of per-trade P&L (the main strategy P&L).
    n_splits           : number of splits for CPCV (default 6).
    n_test_splits      : number of test splits per CPCV combination (default 2).
    purge_frac         : fraction of splits to purge at the train/test boundary.
    embargo_frac       : fraction of splits to embargo after the test period.
    overfit_ratio      : in-sample / OOS improvement ratio threshold for flagging.
    periods_per_year   : for Sharpe annualisation.
    """

    def __init__(
        self,
        pnl_series:       np.ndarray,
        n_splits:         int   = 6,
        n_test_splits:    int   = 2,
        purge_frac:       float = 0.10,
        embargo_frac:     float = 0.05,
        overfit_ratio:    float = 2.0,
        periods_per_year: float = 252.0,
    ):
        self.pnl_series       = np.asarray(pnl_series, dtype=float)
        self.n_splits         = n_splits
        self.n_test_splits    = n_test_splits
        self.purge_frac       = purge_frac
        self.embargo_frac     = embargo_frac
        self.overfit_ratio    = overfit_ratio
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # CPCV
    # ------------------------------------------------------------------

    def _split_indices(self) -> List[Tuple[int, int]]:
        """
        Divide pnl_series into n_splits contiguous blocks.

        Returns list of (start, end) index pairs.
        """
        n  = len(self.pnl_series)
        sz = n // self.n_splits
        groups = []
        for i in range(self.n_splits):
            lo = i * sz
            hi = (i + 1) * sz if i < self.n_splits - 1 else n
            groups.append((lo, hi))
        return groups

    def run_cpcv(self) -> CPCVResult:
        """
        Combinatorial Purged Cross-Validation.

        For each combination of *n_test_splits* groups chosen as test,
        the remaining groups are used as training (with purge/embargo
        applied at the boundary).

        Returns
        -------
        CPCVResult.
        """
        groups  = self._split_indices()
        n_obs   = len(self.pnl_series)
        k       = self.n_splits
        t       = self.n_test_splits
        combos  = list(itertools.combinations(range(k), t))

        purge_n   = max(1, int(self.purge_frac * (n_obs // k)))
        embargo_n = max(1, int(self.embargo_frac * (n_obs // k)))

        sharpes: List[float] = []
        is_better_than_oos:  List[bool] = []

        for test_idx in combos:
            train_idx = [i for i in range(k) if i not in test_idx]

            # Build training set with purge/embargo
            train_returns = []
            for i in train_idx:
                lo, hi = groups[i]
                # Purge: exclude trades near the test boundary
                left_test_nearest  = min(test_idx, key=lambda j: abs(j - i))
                lo_purged = lo
                hi_purged = hi
                if left_test_nearest > i:
                    hi_purged = max(lo, hi - purge_n)
                elif left_test_nearest < i:
                    lo_purged = min(hi, lo + purge_n)
                chunk = self.pnl_series[lo_purged:hi_purged]
                if len(chunk) > 0:
                    train_returns.append(chunk)

            # Build test set with embargo
            test_returns = []
            for i in test_idx:
                lo, hi = groups[i]
                lo_emb = min(hi, lo + embargo_n)
                chunk  = self.pnl_series[lo_emb:hi]
                if len(chunk) > 0:
                    test_returns.append(chunk)

            train_flat = np.concatenate(train_returns) if train_returns else np.array([])
            test_flat  = np.concatenate(test_returns)  if test_returns  else np.array([])

            if len(train_flat) < 5 or len(test_flat) < 5:
                continue

            is_sharpe  = _sharpe(train_flat, self.periods_per_year)
            oos_sharpe = _sharpe(test_flat,  self.periods_per_year)

            if math.isfinite(oos_sharpe):
                sharpes.append(oos_sharpe)
            if math.isfinite(is_sharpe) and math.isfinite(oos_sharpe):
                is_better_than_oos.append(is_sharpe > oos_sharpe)

        if not sharpes:
            sharpes = [float("nan")]

        sharpe_arr = np.array(sharpes)
        p_overfit  = float(np.mean(is_better_than_oos)) if is_better_than_oos else float("nan")

        return CPCVResult(
            n_splits=k,
            n_combinations=len(combos),
            sharpe_distribution=sharpe_arr,
            mean_sharpe=float(np.nanmean(sharpe_arr)),
            std_sharpe=float(np.nanstd(sharpe_arr)),
            sharpe_p25=float(np.nanpercentile(sharpe_arr, 25)),
            sharpe_p75=float(np.nanpercentile(sharpe_arr, 75)),
            is_robust=bool(np.nanpercentile(sharpe_arr, 25) > 0),
            probability_overfit=p_overfit,
        )

    # ------------------------------------------------------------------
    # Walk-forward per hypothesis
    # ------------------------------------------------------------------

    def validate_hypothesis(
        self,
        hypothesis_id: str,
        baseline_is: np.ndarray,
        hypothesis_is: np.ndarray,
        baseline_oos: np.ndarray,
        hypothesis_oos: np.ndarray,
    ) -> HypothesisValidation:
        """
        Walk-forward validation for a single hypothesis.

        Parameters
        ----------
        hypothesis_id  : unique identifier.
        baseline_is    : baseline P&L on the in-sample period.
        hypothesis_is  : hypothesis P&L on the in-sample period.
        baseline_oos   : baseline P&L on the out-of-sample period.
        hypothesis_oos : hypothesis P&L on the out-of-sample period.

        Returns
        -------
        HypothesisValidation.
        """
        is_improvement  = _sharpe_improvement(baseline_is, hypothesis_is,
                                               self.periods_per_year)
        oos_improvement = _sharpe_improvement(baseline_oos, hypothesis_oos,
                                              self.periods_per_year)

        # Ratio of IS to OOS improvement
        if abs(oos_improvement) < 1e-6:
            ratio = float("inf") if abs(is_improvement) > 0 else 1.0
        else:
            ratio = is_improvement / oos_improvement

        is_overfit = (
            math.isfinite(ratio)
            and ratio > self.overfit_ratio
            and is_improvement > 0
        )

        return HypothesisValidation(
            hypothesis_id=hypothesis_id,
            in_sample_sharpe=is_improvement,
            out_of_sample_sharpe=oos_improvement,
            sharpe_ratio=ratio if math.isfinite(ratio) else 999.0,
            is_overfit=is_overfit,
            n_in_sample=len(hypothesis_is),
            n_out_of_sample=len(hypothesis_oos),
        )

    def validate_all(
        self,
        hypothesis_pnl: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> List[HypothesisValidation]:
        """
        Validate multiple hypotheses.

        Parameters
        ----------
        hypothesis_pnl : dict mapping hypothesis_id to
                         (baseline_is, hypothesis_is, baseline_oos, hypothesis_oos).

        Returns
        -------
        List of HypothesisValidation.
        """
        return [
            self.validate_hypothesis(hid, *arrays)
            for hid, arrays in hypothesis_pnl.items()
        ]

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def run(
        self,
        hypothesis_pnl: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
    ) -> OverfittingReport:
        """
        Run the complete overfitting detection suite.

        Parameters
        ----------
        hypothesis_pnl : optional per-hypothesis P&L dict (see validate_all).
                         If None, only CPCV is run.

        Returns
        -------
        OverfittingReport.
        """
        cpcv = self.run_cpcv()

        validations: List[HypothesisValidation] = []
        if hypothesis_pnl:
            validations = self.validate_all(hypothesis_pnl)

        n_overfit = sum(1 for v in validations if v.is_overfit)
        score     = n_overfit / max(len(validations), 1)

        # Build recommendation
        rec_parts = []
        if not cpcv.is_robust:
            rec_parts.append(
                "CPCV indicates strategy is NOT robust across time periods "
                "(P25 Sharpe < 0). Consider simplifying the parameter set."
            )
        if cpcv.probability_overfit > 0.50:
            rec_parts.append(
                f"P(IS Sharpe > OOS Sharpe) = {cpcv.probability_overfit:.1%}: "
                "significant overfitting detected. Use stricter regularisation."
            )
        if score > 0.30:
            rec_parts.append(
                f"{n_overfit}/{len(validations)} hypotheses show in-sample/out-of-sample "
                f"improvement ratio > {self.overfit_ratio}x. "
                "Recommend tightening the acceptance threshold for new hypotheses."
            )
        if not rec_parts:
            rec_parts.append(
                "No significant overfitting detected. "
                "Continue monitoring as new hypotheses are added."
            )

        report = OverfittingReport(
            hypothesis_validations=validations,
            cpcv_result=cpcv,
            n_overfit=n_overfit,
            overall_overfit_score=score,
            recommendation=" ".join(rec_parts),
        )
        logger.info(report.summary())
        return report
