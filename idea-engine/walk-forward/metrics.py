"""
walk-forward/metrics.py
────────────────────────
Walk-forward specific performance metrics.

All metrics are computed from a list of fold result dicts (as returned by
WalkForwardEngine._run_fold) or from parallel IS/OOS arrays.

Key metrics
-----------
  efficiency_ratio  — OOS Sharpe / IS Sharpe (mean across folds); ideal ≈ 1.0
  degradation       — fractional decay OOS vs IS; lower is better
  stability         — variance of OOS Sharpe; lower = more stable
  consistency       — fraction of folds with positive OOS Sharpe
  robustness_score  — composite 0-1 score combining all of the above
  overfitting_score — Bailey-Lopez inspired penalty for parameter-rich models
  min_track_record  — minimum track record length at target confidence

Usage
-----
    from metrics import WFAMetrics, compute_wfa_metrics

    metrics = compute_wfa_metrics(fold_results)
    print(metrics.robustness_score)          # 0.72
    print(metrics.verdict_hint())            # 'ADOPT'
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── WFAMetrics dataclass ──────────────────────────────────────────────────────

@dataclass
class WFAMetrics:
    """
    Aggregated walk-forward analysis metrics across all folds.

    Attributes
    ----------
    n_folds            : number of completed folds
    efficiency_ratio   : mean(OOS Sharpe) / mean(IS Sharpe); ideally near 1.0
    degradation        : (mean_IS_sharpe - mean_OOS_sharpe) / abs(mean_IS_sharpe)
    stability          : 1 / (1 + std(OOS Sharpe)); higher = more stable
    consistency        : fraction of folds with OOS Sharpe > 0
    robustness_score   : composite 0-1 score
    mean_is_sharpe     : mean Sharpe over IS folds
    mean_oos_sharpe    : mean Sharpe over OOS folds
    std_oos_sharpe     : std dev of OOS Sharpe across folds
    mean_is_maxdd      : mean max drawdown over IS folds (negative fraction)
    mean_oos_maxdd     : mean max drawdown over OOS folds
    dd_ratio           : oos_maxdd / is_maxdd (lower = less drawdown amplification)
    trade_decay        : OOS trades / IS trades per bar (proxy for activity decay)
    fold_verdicts      : per-fold pass/fail flags
    raw_fold_data      : original fold result dicts
    """
    n_folds:           int                      = 0
    efficiency_ratio:  float                    = 0.0
    degradation:       float                    = 0.0
    stability:         float                    = 0.0
    consistency:       float                    = 0.0
    robustness_score:  float                    = 0.0
    mean_is_sharpe:    float                    = 0.0
    mean_oos_sharpe:   float                    = 0.0
    std_oos_sharpe:    float                    = 0.0
    mean_is_maxdd:     float                    = 0.0
    mean_oos_maxdd:    float                    = 0.0
    dd_ratio:          float                    = 1.0
    trade_decay:       float                    = 1.0
    fold_verdicts:     List[bool]               = field(default_factory=list)
    raw_fold_data:     List[Dict[str, Any]]     = field(default_factory=list)

    # ── Verdict helpers ────────────────────────────────────────────────

    def verdict_hint(
        self,
        adopt_efficiency: float = 0.60,
        adopt_stability: float  = 0.70,
        adopt_degradation: float = 0.40,
    ) -> str:
        """
        Return a non-binding verdict hint based on threshold comparison.

        Full decision logic lives in WalkForwardEngine.verdict().
        This method is for quick interactive inspection.

        Returns
        -------
        'ADOPT' | 'REJECT' | 'RETEST'
        """
        if (
            self.efficiency_ratio > adopt_efficiency
            and self.stability    > adopt_stability
            and self.degradation  < adopt_degradation
        ):
            return "ADOPT"

        # Borderline: any two criteria pass
        passes = sum([
            self.efficiency_ratio > adopt_efficiency * 0.85,
            self.stability        > adopt_stability  * 0.85,
            self.degradation      < adopt_degradation * 1.25,
        ])
        if passes >= 2:
            return "RETEST"

        return "REJECT"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (excludes raw_fold_data for brevity)."""
        return {
            "n_folds":          self.n_folds,
            "efficiency_ratio": round(self.efficiency_ratio, 4),
            "degradation":      round(self.degradation, 4),
            "stability":        round(self.stability, 4),
            "consistency":      round(self.consistency, 4),
            "robustness_score": round(self.robustness_score, 4),
            "mean_is_sharpe":   round(self.mean_is_sharpe, 4),
            "mean_oos_sharpe":  round(self.mean_oos_sharpe, 4),
            "std_oos_sharpe":   round(self.std_oos_sharpe, 4),
            "mean_is_maxdd":    round(self.mean_is_maxdd, 4),
            "mean_oos_maxdd":   round(self.mean_oos_maxdd, 4),
            "dd_ratio":         round(self.dd_ratio, 4),
            "trade_decay":      round(self.trade_decay, 4),
        }

    def __repr__(self) -> str:
        return (
            f"WFAMetrics(folds={self.n_folds}, "
            f"eff={self.efficiency_ratio:.3f}, "
            f"deg={self.degradation:.3f}, "
            f"stab={self.stability:.3f}, "
            f"robust={self.robustness_score:.3f}, "
            f"hint={self.verdict_hint()!r})"
        )


# ── Core metric functions ─────────────────────────────────────────────────────

def efficiency_ratio(fold_results: List[Dict[str, Any]]) -> float:
    """
    Compute OOS/IS Sharpe efficiency ratio across folds.

    For each fold, the fold-level efficiency = oos_sharpe / is_sharpe.
    Final value = mean across folds with valid (non-zero IS) entries.

    Parameters
    ----------
    fold_results : list of dicts with keys 'is_sharpe' and 'oos_sharpe'

    Returns
    -------
    float — mean efficiency ratio (0.0 if no valid folds)
    """
    ratios: List[float] = []
    for fr in fold_results:
        is_s  = fr.get("is_sharpe")
        oos_s = fr.get("oos_sharpe")
        if is_s is None or oos_s is None:
            continue
        if abs(is_s) < 1e-8:
            continue
        ratios.append(oos_s / is_s)

    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def degradation_score(fold_results: List[Dict[str, Any]]) -> float:
    """
    Compute performance degradation: how much does Sharpe decay OOS vs IS?

    degradation = (mean_IS - mean_OOS) / abs(mean_IS)
    A value of 0.0 means no degradation; 1.0 means complete loss of performance.
    Negative values mean OOS outperforms IS (rare but possible).

    Parameters
    ----------
    fold_results : list of dicts with keys 'is_sharpe' and 'oos_sharpe'

    Returns
    -------
    float — degradation fraction
    """
    is_sharpes:  List[float] = []
    oos_sharpes: List[float] = []
    for fr in fold_results:
        is_s  = fr.get("is_sharpe")
        oos_s = fr.get("oos_sharpe")
        if is_s is None or oos_s is None:
            continue
        is_sharpes.append(float(is_s))
        oos_sharpes.append(float(oos_s))

    if not is_sharpes:
        return 1.0   # worst-case if no data

    mean_is  = float(np.mean(is_sharpes))
    mean_oos = float(np.mean(oos_sharpes))

    if abs(mean_is) < 1e-8:
        return 1.0

    return float((mean_is - mean_oos) / abs(mean_is))


def stability_score(fold_results: List[Dict[str, Any]]) -> float:
    """
    Compute OOS Sharpe stability across folds.

    stability = 1 / (1 + std(OOS Sharpe))
    Range: (0, 1].  Higher = more stable (lower variance).

    Parameters
    ----------
    fold_results : list of dicts with key 'oos_sharpe'

    Returns
    -------
    float — stability in (0, 1]
    """
    oos_sharpes: List[float] = [
        float(fr["oos_sharpe"])
        for fr in fold_results
        if fr.get("oos_sharpe") is not None
    ]
    if len(oos_sharpes) < 2:
        return 0.5   # neutral if insufficient data

    std = float(np.std(oos_sharpes, ddof=1))
    return float(1.0 / (1.0 + std))


def consistency_score(fold_results: List[Dict[str, Any]]) -> float:
    """
    Fraction of folds where OOS Sharpe is positive.

    Parameters
    ----------
    fold_results : list of dicts with key 'oos_sharpe'

    Returns
    -------
    float — fraction in [0, 1]
    """
    oos_sharpes: List[float] = [
        float(fr["oos_sharpe"])
        for fr in fold_results
        if fr.get("oos_sharpe") is not None
    ]
    if not oos_sharpes:
        return 0.0
    positive = sum(1 for s in oos_sharpes if s > 0)
    return positive / len(oos_sharpes)


def drawdown_ratio(fold_results: List[Dict[str, Any]]) -> float:
    """
    Ratio of mean OOS max-drawdown to mean IS max-drawdown.

    A ratio > 1.0 means drawdowns are worse OOS (typical).
    Uses absolute values so sign convention doesn't matter.

    Parameters
    ----------
    fold_results : list of dicts with keys 'is_maxdd' and 'oos_maxdd'

    Returns
    -------
    float — dd_ratio (1.0 if insufficient data)
    """
    is_dds:  List[float] = []
    oos_dds: List[float] = []
    for fr in fold_results:
        is_dd  = fr.get("is_maxdd")
        oos_dd = fr.get("oos_maxdd")
        if is_dd is None or oos_dd is None:
            continue
        is_dds.append(abs(float(is_dd)))
        oos_dds.append(abs(float(oos_dd)))

    if not is_dds or np.mean(is_dds) < 1e-8:
        return 1.0

    return float(np.mean(oos_dds) / np.mean(is_dds))


def trade_decay(fold_results: List[Dict[str, Any]]) -> float:
    """
    Compute trade activity decay (OOS trades per bar vs IS trades per bar).

    A value of 1.0 means no activity change; < 1.0 means fewer OOS trades
    (possible regime change); > 1.0 means more OOS activity.

    Parameters
    ----------
    fold_results : list of dicts with keys 'is_trades', 'oos_trades',
                   'is_bars', 'oos_bars'

    Returns
    -------
    float — trade decay ratio
    """
    is_rates:  List[float] = []
    oos_rates: List[float] = []
    for fr in fold_results:
        is_t  = fr.get("is_trades")
        oos_t = fr.get("oos_trades")
        is_b  = fr.get("is_bars",  1)
        oos_b = fr.get("oos_bars", 1)
        if is_t is None or oos_t is None:
            continue
        is_b  = max(1, int(is_b))
        oos_b = max(1, int(oos_b))
        is_rates.append(float(is_t) / is_b)
        oos_rates.append(float(oos_t) / oos_b)

    if not is_rates or np.mean(is_rates) < 1e-8:
        return 1.0

    return float(np.mean(oos_rates) / np.mean(is_rates))


def robustness_score(
    eff: float,
    deg: float,
    stab: float,
    cons: float,
    dd_rat: float = 1.0,
) -> float:
    """
    Compute composite robustness score in [0, 1].

    Weights:
      efficiency  40%
      stability   25%
      consistency 20%
      degradation 15% (inverted — lower degradation → higher score)

    Drawdown ratio is used as a multiplier penalty (capped at 1.0).

    Parameters
    ----------
    eff     : efficiency_ratio (0–1+ ideally near 1.0)
    deg     : degradation_score (lower is better)
    stab    : stability_score (0–1, higher is better)
    cons    : consistency_score (0–1)
    dd_rat  : drawdown_ratio (1.0 = no change OOS)

    Returns
    -------
    float in [0, 1]
    """
    # Clamp inputs to reasonable ranges
    eff_c  = max(0.0, min(eff,  2.0))
    stab_c = max(0.0, min(stab, 1.0))
    cons_c = max(0.0, min(cons, 1.0))
    deg_c  = max(0.0, min(deg,  2.0))   # 2.0 = complete OOS degradation

    eff_score  = min(1.0, eff_c)                   # cap at 1.0
    deg_score  = max(0.0, 1.0 - deg_c)             # invert
    stab_score = stab_c
    cons_score = cons_c

    raw = (
        0.40 * eff_score  +
        0.25 * stab_score +
        0.20 * cons_score +
        0.15 * deg_score
    )

    # DD ratio penalty: dd_ratio = 2.0 means drawdowns doubled OOS → 0.9x
    dd_penalty = max(0.5, 1.0 - 0.1 * max(0.0, dd_rat - 1.0))
    raw *= dd_penalty

    return float(min(1.0, max(0.0, raw)))


# ── Scatter data for dashboard ────────────────────────────────────────────────

def oos_sharpe_vs_is_sharpe(
    fold_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build scatter data for an OOS-vs-IS Sharpe plot.

    Parameters
    ----------
    fold_results : list of fold result dicts

    Returns
    -------
    List of dicts: [{'fold': int, 'is_sharpe': float, 'oos_sharpe': float,
                     'efficiency': float, 'label': str}, ...]
    """
    points: List[Dict[str, Any]] = []
    for fr in fold_results:
        is_s  = fr.get("is_sharpe")
        oos_s = fr.get("oos_sharpe")
        if is_s is None or oos_s is None:
            continue
        is_s  = float(is_s)
        oos_s = float(oos_s)
        eff   = (oos_s / is_s) if abs(is_s) > 1e-8 else 0.0
        fold_num = fr.get("fold_number", 0)
        points.append({
            "fold":       fold_num,
            "is_sharpe":  round(is_s, 4),
            "oos_sharpe": round(oos_s, 4),
            "efficiency": round(eff, 4),
            "label":      f"F{fold_num}",
        })
    return points


# ── Parameter sensitivity ─────────────────────────────────────────────────────

def parameter_sensitivity_across_folds(
    fold_results: List[Dict[str, Any]],
    param_name: str,
) -> Dict[str, Any]:
    """
    Analyse how OOS performance varies with a specific parameter value across folds.

    Useful for detecting over-optimised parameters that only work in a narrow
    value range.

    Parameters
    ----------
    fold_results : list of fold result dicts (must include 'params' sub-dict)
    param_name   : name of the parameter to analyse

    Returns
    -------
    dict with keys:
      param_name, values, oos_sharpes, correlation, is_sensitive
    """
    values:      List[float] = []
    oos_sharpes: List[float] = []

    for fr in fold_results:
        params = fr.get("params") or fr.get("param_delta", {})
        v = params.get(param_name) if isinstance(params, dict) else None
        oos_s = fr.get("oos_sharpe")
        if v is None or oos_s is None:
            continue
        try:
            values.append(float(v))
            oos_sharpes.append(float(oos_s))
        except (TypeError, ValueError):
            continue

    if len(values) < 3:
        return {
            "param_name":   param_name,
            "values":       values,
            "oos_sharpes":  oos_sharpes,
            "correlation":  None,
            "is_sensitive": False,
            "note":         "insufficient data",
        }

    corr = float(np.corrcoef(values, oos_sharpes)[0, 1])
    is_sensitive = abs(corr) > 0.5   # high correlation = sensitivity

    return {
        "param_name":   param_name,
        "values":       [round(v, 6) for v in values],
        "oos_sharpes":  [round(s, 4) for s in oos_sharpes],
        "correlation":  round(corr, 4),
        "is_sensitive": is_sensitive,
        "note":         "high param sensitivity — may be over-fit" if is_sensitive else "stable",
    }


# ── Regime-conditional efficiency ─────────────────────────────────────────────

def regime_conditional_efficiency(
    fold_results: List[Dict[str, Any]],
    regime_labels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute efficiency ratio split by market regime (BULL / BEAR / NEUTRAL).

    Parameters
    ----------
    fold_results   : list of fold result dicts
    regime_labels  : list of regime label strings, one per fold (same length)

    Returns
    -------
    dict mapping regime → {'mean_oos_sharpe': float, 'n_folds': int,
                            'efficiency': float}
    """
    if len(regime_labels) != len(fold_results):
        logger.warning(
            "regime_labels length (%d) != fold_results length (%d).",
            len(regime_labels), len(fold_results),
        )
        regime_labels = list(regime_labels) + ["UNKNOWN"] * max(
            0, len(fold_results) - len(regime_labels)
        )

    buckets: Dict[str, List[Tuple[float, float]]] = {}
    for fr, reg in zip(fold_results, regime_labels):
        is_s  = fr.get("is_sharpe")
        oos_s = fr.get("oos_sharpe")
        if is_s is None or oos_s is None:
            continue
        if reg not in buckets:
            buckets[reg] = []
        buckets[reg].append((float(is_s), float(oos_s)))

    result: Dict[str, Dict[str, float]] = {}
    for reg, pairs in buckets.items():
        is_vals  = [p[0] for p in pairs]
        oos_vals = [p[1] for p in pairs]
        mean_is  = float(np.mean(is_vals))
        mean_oos = float(np.mean(oos_vals))
        eff = (mean_oos / mean_is) if abs(mean_is) > 1e-8 else 0.0
        result[reg] = {
            "mean_is_sharpe":  round(mean_is, 4),
            "mean_oos_sharpe": round(mean_oos, 4),
            "efficiency":      round(eff, 4),
            "n_folds":         len(pairs),
        }
    return result


# ── Overfitting score ─────────────────────────────────────────────────────────

def overfitting_score(
    is_sharpe: float,
    oos_sharpe: float,
    n_params: int,
    n_trades: int = 100,
) -> float:
    """
    Compute an overfitting score inspired by Bailey-Lopez (2014).

    The penalisation increases with the number of parameters and decreases
    with the number of trades (degrees of freedom).

    Score > 0.5 indicates likely overfitting.

    Formula (simplified):
        t_stat  = sharpe * sqrt(n_trades)
        penalty = n_params / sqrt(n_trades)
        score   = penalty / (1 + penalty)
        adjusted_IS_sharpe = is_sharpe - penalty
        overfitting = max(0, adjusted_IS_sharpe - oos_sharpe) / abs(is_sharpe)

    Parameters
    ----------
    is_sharpe  : annualised IS Sharpe ratio
    oos_sharpe : annualised OOS Sharpe ratio
    n_params   : number of free parameters in the hypothesis
    n_trades   : total number of trades (proxy for degrees of freedom)

    Returns
    -------
    float in [0, 1] — higher = more likely overfit
    """
    n_trades = max(1, n_trades)
    n_params = max(0, n_params)

    penalty = n_params / math.sqrt(n_trades)
    adjusted_is = is_sharpe - penalty

    gap = max(0.0, adjusted_is - oos_sharpe)
    if abs(is_sharpe) < 1e-8:
        return float(min(1.0, penalty))

    raw_score = gap / abs(is_sharpe)
    # Add a direct penalty term for many params relative to trades
    param_penalty = min(0.5, penalty / 4.0)

    return float(min(1.0, max(0.0, raw_score + param_penalty)))


# ── Minimum track record length ───────────────────────────────────────────────

def minimum_track_record_length(
    sharpe: float,
    target_confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the minimum track record length (in years) needed to confirm a
    given annualised Sharpe ratio at the specified confidence level.

    Based on Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier."

    Parameters
    ----------
    sharpe            : annualised Sharpe ratio
    target_confidence : required confidence level (default 0.95)
    skewness          : return distribution skewness (0 = normal)
    kurtosis          : return distribution kurtosis (3 = normal)
    periods_per_year  : bars per year for annualisation

    Returns
    -------
    float — minimum number of years of track record required
    """
    if sharpe <= 0:
        return float("inf")

    # Convert annualised Sharpe to per-period Sharpe
    sr_period = sharpe / math.sqrt(periods_per_year)

    # Adjust variance for non-normality (Cornish-Fisher)
    # var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurtosis-1)/4 * SR^2) / T
    # We want z_alpha = SR_period * sqrt(T) / sqrt(adjustment)
    # => T = (z_alpha / SR_period)^2 * adjustment
    from scipy.stats import norm
    z_alpha = norm.ppf(target_confidence)

    # Non-normality correction factor
    correction = 1.0 + 0.5 * sr_period**2 - skewness * sr_period + (kurtosis - 3) / 4.0 * sr_period**2
    correction = max(correction, 0.01)   # clamp

    n_periods = correction * (z_alpha / sr_period) ** 2
    n_years   = n_periods / periods_per_year

    return float(max(0.0, n_years))


# ── Master compute function ───────────────────────────────────────────────────

def compute_wfa_metrics(
    fold_results: List[Dict[str, Any]],
) -> WFAMetrics:
    """
    Compute all WFA metrics from a list of fold result dicts.

    This is the primary entry point used by WalkForwardEngine.

    Parameters
    ----------
    fold_results : list of dicts, each with keys:
        fold_number, is_sharpe, oos_sharpe, is_maxdd, oos_maxdd,
        is_trades, oos_trades, is_bars (optional), oos_bars (optional)

    Returns
    -------
    WFAMetrics — fully populated
    """
    if not fold_results:
        return WFAMetrics(n_folds=0)

    eff  = efficiency_ratio(fold_results)
    deg  = degradation_score(fold_results)
    stab = stability_score(fold_results)
    cons = consistency_score(fold_results)
    dd_r = drawdown_ratio(fold_results)
    td   = trade_decay(fold_results)

    robust = robustness_score(eff, deg, stab, cons, dd_r)

    # Mean IS/OOS Sharpe
    is_sharpes  = [float(fr["is_sharpe"])  for fr in fold_results if fr.get("is_sharpe")  is not None]
    oos_sharpes = [float(fr["oos_sharpe"]) for fr in fold_results if fr.get("oos_sharpe") is not None]
    is_dds      = [float(fr["is_maxdd"])   for fr in fold_results if fr.get("is_maxdd")   is not None]
    oos_dds     = [float(fr["oos_maxdd"])  for fr in fold_results if fr.get("oos_maxdd")  is not None]

    mean_is  = float(np.mean(is_sharpes))  if is_sharpes  else 0.0
    mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    std_oos  = float(np.std(oos_sharpes, ddof=1)) if len(oos_sharpes) >= 2 else 0.0
    mean_is_dd  = float(np.mean(is_dds))  if is_dds  else 0.0
    mean_oos_dd = float(np.mean(oos_dds)) if oos_dds else 0.0

    fold_verdicts = [
        (fr.get("oos_sharpe") or 0.0) > 0.0
        for fr in fold_results
        if fr.get("oos_sharpe") is not None
    ]

    return WFAMetrics(
        n_folds          = len(fold_results),
        efficiency_ratio = eff,
        degradation      = deg,
        stability        = stab,
        consistency      = cons,
        robustness_score = robust,
        mean_is_sharpe   = mean_is,
        mean_oos_sharpe  = mean_oos,
        std_oos_sharpe   = std_oos,
        mean_is_maxdd    = mean_is_dd,
        mean_oos_maxdd   = mean_oos_dd,
        dd_ratio         = dd_r,
        trade_decay      = td,
        fold_verdicts    = fold_verdicts,
        raw_fold_data    = fold_results,
    )
