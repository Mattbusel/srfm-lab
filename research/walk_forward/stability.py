"""
research/walk_forward/stability.py
────────────────────────────────────
Parameter stability and robustness analysis for walk-forward results.

Provides:
  • StabilityAnalyzer: rolling metrics and parameter drift analysis
  • robustness_test: perturbation-based sensitivity profiling
  • Structural break tests: Chow test, CUSUM test
  • Rolling performance series: Sharpe, win rate, profit factor
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from .engine import WFResult, FoldResult
from .metrics import (
    sharpe_ratio,
    win_rate,
    profit_factor,
    max_drawdown,
    PerformanceStats,
    compute_performance_stats,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StabilityReport:
    """
    Report summarising parameter stability across walk-forward folds.

    Attributes
    ----------
    param_drift         : per-parameter drift fraction (fraction of folds NOT
                          selecting the most common value).
    most_common_params  : most frequently selected params across folds.
    sharpe_dispersion   : std dev of OOS Sharpe across folds.
    sharpe_cv           : coefficient of variation of OOS Sharpe.
    fold_sharpes        : OOS Sharpe per fold.
    fold_params         : best params per fold.
    stability_score     : overall stability score in [0, 1].
                          1.0 = perfectly stable (same params every fold).
    is_oos_ratio        : mean IS Sharpe / mean OOS Sharpe.
    regime_coverage     : dict of regime → fraction of folds where regime appeared.
    """
    param_drift:        Dict[str, float]
    most_common_params: Dict[str, Any]
    sharpe_dispersion:  float
    sharpe_cv:          float
    fold_sharpes:       List[float]
    fold_params:        List[Dict[str, Any]]
    stability_score:    float
    is_oos_ratio:       float
    regime_coverage:    Dict[str, float]


@dataclass
class PerturbationResult:
    """Result for a single parameter perturbation."""
    param_name:      str
    base_value:      Any
    perturbed_value: Any
    perturbation_pct: float   # e.g. 0.10 for +10%
    base_score:      float
    perturbed_score: float
    score_delta:     float    # perturbed_score - base_score
    score_delta_pct: float    # percentage change in score


@dataclass
class RobustnessResult:
    """
    Result from a parameter robustness (sensitivity) test.

    Attributes
    ----------
    base_params         : the parameter set that was perturbed.
    base_score          : score at the base parameters.
    perturbations       : all PerturbationResult objects.
    worst_degradation   : worst (most negative) score delta across all perturbations.
    mean_degradation    : mean score delta across all perturbations.
    robustness_score    : fraction of perturbations with score > 0
                          (score didn't degrade). Higher is better.
    sensitivity_by_param: dict of param → mean |score_delta| (sensitivity).
    """
    base_params:          Dict[str, Any]
    base_score:           float
    perturbations:        List[PerturbationResult]
    worst_degradation:    float
    mean_degradation:     float
    robustness_score:     float
    sensitivity_by_param: Dict[str, float]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert perturbation results to a DataFrame."""
        rows = []
        for p in self.perturbations:
            rows.append({
                "param_name":      p.param_name,
                "base_value":      p.base_value,
                "perturbed_value": p.perturbed_value,
                "perturbation_pct": p.perturbation_pct * 100,
                "base_score":      p.base_score,
                "perturbed_score": p.perturbed_score,
                "score_delta":     p.score_delta,
                "score_delta_pct": p.score_delta_pct * 100,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Structural break tests
# ─────────────────────────────────────────────────────────────────────────────

def chow_test(
    series1: Union[np.ndarray, pd.Series, List[float]],
    series2: Union[np.ndarray, pd.Series, List[float]],
) -> Tuple[float, float]:
    """
    Chow structural break test between two sub-series.

    Tests whether two sub-periods of a time series have the same mean (and
    variance) under OLS. The null hypothesis is that there is no structural
    break (i.e. the same model fits both periods).

    F-statistic = [(RSS_pooled - RSS1 - RSS2) / k] / [(RSS1 + RSS2) / (n - 2k)]

    where k = number of regression parameters = 1 (intercept only), and
    n = total observations.

    Parameters
    ----------
    series1 : first sub-period values (e.g. OOS Sharpes from folds 1–3).
    series2 : second sub-period values (e.g. OOS Sharpes from folds 4–6).

    Returns
    -------
    (F_statistic, p_value) under F(k, n - 2k) distribution.

    References
    ----------
    Chow, G.C. (1960). Tests of Equality Between Sets of Coefficients in Two
    Linear Regressions. Econometrica.
    """
    s1 = np.asarray(series1, dtype=np.float64)
    s2 = np.asarray(series2, dtype=np.float64)
    s1 = s1[np.isfinite(s1)]
    s2 = s2[np.isfinite(s2)]

    if len(s1) < 3 or len(s2) < 3:
        return (np.nan, np.nan)

    def _ols_rss(x: np.ndarray) -> Tuple[float, float]:
        """Fit constant OLS and return RSS and coefficient."""
        n   = len(x)
        mu  = float(np.mean(x))
        rss = float(np.sum((x - mu) ** 2))
        return rss, mu

    rss1, mu1 = _ols_rss(s1)
    rss2, mu2 = _ols_rss(s2)

    # Pooled: fit single mean to all data
    s_all       = np.concatenate([s1, s2])
    rss_all, _  = _ols_rss(s_all)

    n   = len(s_all)
    k   = 1   # number of parameters (intercept only)

    numerator   = (rss_all - rss1 - rss2) / k
    denominator = (rss1 + rss2) / max(n - 2 * k, 1)

    if denominator < 1e-12:
        return (0.0, 1.0)

    F      = numerator / denominator
    p_val  = 1.0 - float(stats.f.cdf(F, dfn=k, dfd=max(n - 2 * k, 1)))
    return (float(F), p_val)


def cusum_test(
    series: Union[np.ndarray, pd.Series, List[float]],
    significance: float = 0.05,
) -> Tuple[float, float, Optional[int]]:
    """
    CUSUM (Cumulative Sum) structural break test.

    Detects whether the cumulative sum of a series exceeds critical boundaries,
    indicating a structural break in the mean.

    Parameters
    ----------
    series       : 1-D time series (e.g. rolling Sharpe values).
    significance : significance level for critical value (default 0.05).

    Returns
    -------
    (statistic, critical_value, break_point) where:
      - statistic     : maximum absolute CUSUM value.
      - critical_value: threshold at `significance` level.
      - break_point   : index of maximum CUSUM (potential break location).
                        None if no break detected.

    References
    ----------
    Brown, R.L., Durbin, J. & Evans, J.M. (1975). Techniques for Testing
    the Constancy of Regression Relationships Over Time. JRSS-B.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 10:
        return (np.nan, np.nan, None)

    mean  = np.mean(x)
    std   = np.std(x, ddof=1)

    if std < 1e-12:
        return (0.0, 1.0, None)

    # Cumulative sum of standardised residuals
    cusum       = np.cumsum((x - mean) / std)
    cusum_abs   = np.abs(cusum)
    statistic   = float(np.max(cusum_abs))
    break_idx   = int(np.argmax(cusum_abs))

    # Critical value from Brownian bridge approximation
    # For significance=0.05: c ≈ 1.36 * sqrt(n)
    cv_table = {0.01: 1.63, 0.05: 1.36, 0.10: 1.22}
    cv_factor = cv_table.get(significance, 1.36)
    critical  = cv_factor * math.sqrt(n)

    break_point = break_idx if statistic > critical else None

    return (statistic, critical, break_point)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling performance series
# ─────────────────────────────────────────────────────────────────────────────

def rolling_sharpe(
    returns: Union[np.ndarray, pd.Series],
    window:  int = 60,
) -> pd.Series:
    """
    Compute rolling annualized Sharpe ratio.

    Parameters
    ----------
    returns : 1-D period returns.
    window  : rolling window size in bars.

    Returns
    -------
    pd.Series of rolling Sharpe values.
    """
    if isinstance(returns, pd.Series):
        r = returns.reset_index(drop=True)
    else:
        r = pd.Series(np.asarray(returns, dtype=np.float64))

    result = pd.Series(np.nan, index=r.index)

    for i in range(window - 1, len(r)):
        chunk = r.iloc[i - window + 1:i + 1].to_numpy()
        result.iloc[i] = sharpe_ratio(chunk)

    return result


def rolling_win_rate(
    pnl:    Union[np.ndarray, pd.Series],
    window: int = 60,
) -> pd.Series:
    """
    Compute rolling win rate over a sliding window.

    Parameters
    ----------
    pnl    : 1-D per-trade P&L series.
    window : rolling window size.

    Returns
    -------
    pd.Series of rolling win rates in [0, 1].
    """
    if isinstance(pnl, pd.Series):
        p = pnl.reset_index(drop=True)
    else:
        p = pd.Series(np.asarray(pnl, dtype=np.float64))

    result = pd.Series(np.nan, index=p.index)

    for i in range(window - 1, len(p)):
        chunk = p.iloc[i - window + 1:i + 1].to_numpy()
        result.iloc[i] = win_rate(chunk)

    return result


def rolling_profit_factor(
    pnl:    Union[np.ndarray, pd.Series],
    window: int = 60,
) -> pd.Series:
    """
    Compute rolling profit factor over a sliding window.

    Parameters
    ----------
    pnl    : 1-D per-trade P&L series.
    window : rolling window size.

    Returns
    -------
    pd.Series of rolling profit factor values (clipped at 10 for display).
    """
    if isinstance(pnl, pd.Series):
        p = pnl.reset_index(drop=True)
    else:
        p = pd.Series(np.asarray(pnl, dtype=np.float64))

    result = pd.Series(np.nan, index=p.index)

    for i in range(window - 1, len(p)):
        chunk = p.iloc[i - window + 1:i + 1].to_numpy()
        pf    = profit_factor(chunk)
        result.iloc[i] = min(pf, 10.0)  # cap for display

    return result


# ─────────────────────────────────────────────────────────────────────────────
# StabilityAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class StabilityAnalyzer:
    """
    Analyse parameter stability and strategy robustness from walk-forward results.

    Parameters
    ----------
    starting_equity : initial equity for performance calculations.

    Examples
    --------
    >>> analyzer = StabilityAnalyzer()
    >>> report = analyzer.parameter_stability(wf_result)
    >>> print(f"Stability score: {report.stability_score:.2f}")
    """

    def __init__(self, starting_equity: float = 100_000.0) -> None:
        self.starting_equity = starting_equity

    # ------------------------------------------------------------------
    # parameter_stability
    # ------------------------------------------------------------------

    def parameter_stability(self, wf_result: WFResult) -> StabilityReport:
        """
        Analyse parameter stability across walk-forward folds.

        Measures:
        1. Per-parameter drift: fraction of folds NOT using the modal value.
        2. OOS Sharpe dispersion: std of OOS Sharpe across folds.
        3. Stability score: weighted combination of drift and consistency.

        Parameters
        ----------
        wf_result : WFResult from WalkForwardEngine.run().

        Returns
        -------
        StabilityReport

        Notes
        -----
        A stability_score near 1.0 indicates the same parameters were selected
        consistently across folds — a prerequisite for trust in OOS results.
        """
        successful = [fr for fr in wf_result.fold_results if fr.success]
        if not successful:
            return StabilityReport(
                param_drift={}, most_common_params={},
                sharpe_dispersion=np.nan, sharpe_cv=np.nan,
                fold_sharpes=[], fold_params=[],
                stability_score=0.0, is_oos_ratio=np.nan,
                regime_coverage={},
            )

        fold_params    = [fr.params for fr in successful]
        fold_sharpes   = [fr.oos_sharpe for fr in successful]

        # Get all param names
        all_param_names: set = set()
        for p in fold_params:
            all_param_names.update(p.keys())

        # Per-parameter: compute most common value and drift fraction
        param_drift:        Dict[str, float]   = {}
        most_common_params: Dict[str, Any]     = {}

        for param in sorted(all_param_names):
            values = [p.get(param) for p in fold_params if param in p]
            if not values:
                param_drift[param] = 0.0
                continue

            from collections import Counter
            counts     = Counter(str(v) for v in values)
            modal_val  = counts.most_common(1)[0][0]
            modal_count = counts.most_common(1)[0][1]
            drift_frac  = 1.0 - modal_count / len(values)

            param_drift[param] = float(drift_frac)

            # Recover original typed value for most common
            for p in fold_params:
                if str(p.get(param)) == modal_val:
                    most_common_params[param] = p[param]
                    break

        # Sharpe dispersion
        sharpe_arr = np.array(fold_sharpes)
        sharpe_arr = sharpe_arr[np.isfinite(sharpe_arr)]

        if len(sharpe_arr) > 1:
            sharpe_std = float(np.std(sharpe_arr, ddof=1))
            sharpe_mean = float(np.mean(sharpe_arr))
            sharpe_cv   = sharpe_std / (abs(sharpe_mean) + 1e-8)
        else:
            sharpe_std = 0.0
            sharpe_cv  = 0.0

        # Stability score: fraction of params with low drift × Sharpe consistency
        drift_scores = [1.0 - d for d in param_drift.values()]
        param_score  = float(np.mean(drift_scores)) if drift_scores else 1.0

        # CV score: low CV = high consistency (cap at 1)
        cv_score     = float(np.clip(1.0 - min(sharpe_cv, 2.0) / 2.0, 0.0, 1.0))
        stability    = 0.7 * param_score + 0.3 * cv_score

        # IS/OOS ratio
        is_sharpes  = [fr.is_stats.sharpe for fr in successful]
        oos_sharpes = [fr.oos_sharpe      for fr in successful]
        mean_is  = float(np.mean(is_sharpes))  if is_sharpes  else 0.0
        mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        is_oos_ratio = mean_is / (mean_oos + 1e-8) if abs(mean_oos) > 1e-8 else np.inf

        # Regime coverage across folds
        regime_counts: Dict[str, int] = {}
        for fr in successful:
            if fr.split and fr.split.metadata.get("regime_distribution"):
                for reg in fr.split.metadata["regime_distribution"]:
                    regime_counts[reg] = regime_counts.get(reg, 0) + 1

        n_folds = len(successful)
        regime_coverage = {
            reg: count / n_folds
            for reg, count in regime_counts.items()
        }

        return StabilityReport(
            param_drift         = param_drift,
            most_common_params  = most_common_params,
            sharpe_dispersion   = sharpe_std,
            sharpe_cv           = sharpe_cv,
            fold_sharpes        = fold_sharpes,
            fold_params         = fold_params,
            stability_score     = float(stability),
            is_oos_ratio        = float(is_oos_ratio),
            regime_coverage     = regime_coverage,
        )

    # ------------------------------------------------------------------
    # robustness_test
    # ------------------------------------------------------------------

    def robustness_test(
        self,
        strategy_fn:     Callable,
        best_params:     Dict[str, Any],
        trades:          pd.DataFrame,
        splitter:        List,   # List[WFSplit]
        n_perturbations: int   = 50,
        perturbation_pcts: Optional[List[float]] = None,
        metric:          str   = "sharpe",
    ) -> RobustnessResult:
        """
        Test strategy robustness by perturbing each parameter.

        For each numeric parameter, evaluates strategy performance at:
        ±10%, ±20%, ±50% of the base value.

        This reveals how sensitive the strategy is to parameter choices and
        whether the IS optimum is sharp (fragile) or broad (robust).

        Parameters
        ----------
        strategy_fn      : callable(train_trades, params) → List[dict].
        best_params      : base parameter dict.
        trades           : full trades DataFrame.
        splitter         : list of WFSplit objects.
        n_perturbations  : max perturbations per parameter.
        perturbation_pcts: list of fractional perturbation amounts.
                           Default: [±0.10, ±0.20, ±0.50].
        metric           : scoring metric.

        Returns
        -------
        RobustnessResult

        Examples
        --------
        >>> result = analyzer.robustness_test(my_fn, best_params, trades, splits)
        >>> print(f"Robustness score: {result.robustness_score:.2f}")
        >>> print(result.to_dataframe().sort_values('score_delta'))
        """
        from .optimizer import _call_and_score

        if perturbation_pcts is None:
            perturbation_pcts = [-0.50, -0.20, -0.10, +0.10, +0.20, +0.50]

        # Compute base score
        try:
            base_score, _ = _call_and_score(
                strategy_fn, best_params, splitter, trades, metric, self.starting_equity
            )
        except Exception as e:
            logger.error("robustness_test: base evaluation failed: %s", e)
            return RobustnessResult(
                base_params={}, base_score=np.nan, perturbations=[],
                worst_degradation=np.nan, mean_degradation=np.nan,
                robustness_score=0.0, sensitivity_by_param={},
            )

        perturbations: List[PerturbationResult] = []

        for param, base_val in best_params.items():
            if not isinstance(base_val, (int, float)):
                continue  # skip non-numeric params

            for pct in perturbation_pcts:
                perturbed_val = base_val * (1.0 + pct)

                # Preserve integer type
                if isinstance(base_val, int):
                    perturbed_val = max(1, int(round(perturbed_val)))

                perturbed_params = {**best_params, param: perturbed_val}

                try:
                    p_score, _ = _call_and_score(
                        strategy_fn, perturbed_params, splitter, trades,
                        metric, self.starting_equity
                    )
                except Exception:
                    p_score = np.nan

                delta = float(p_score - base_score) if (np.isfinite(p_score) and np.isfinite(base_score)) else np.nan
                delta_pct = delta / (abs(base_score) + 1e-8) if np.isfinite(delta) else np.nan

                perturbations.append(PerturbationResult(
                    param_name       = param,
                    base_value       = base_val,
                    perturbed_value  = perturbed_val,
                    perturbation_pct = pct,
                    base_score       = float(base_score),
                    perturbed_score  = float(p_score) if np.isfinite(p_score) else np.nan,
                    score_delta      = float(delta) if np.isfinite(delta) else np.nan,
                    score_delta_pct  = float(delta_pct) if np.isfinite(delta_pct) else np.nan,
                ))

        # Aggregate statistics
        valid_deltas = [p.score_delta for p in perturbations if np.isfinite(p.score_delta)]

        worst      = float(min(valid_deltas)) if valid_deltas else np.nan
        mean_delta = float(np.mean(valid_deltas)) if valid_deltas else np.nan

        # Robustness score: fraction of perturbations where score didn't degrade > 10%
        non_degrading = sum(
            1 for p in perturbations
            if np.isfinite(p.score_delta) and p.score_delta >= -0.1 * abs(base_score)
        )
        robustness_score = non_degrading / max(1, len(perturbations))

        # Sensitivity by param: mean |delta| per parameter
        sensitivity: Dict[str, float] = {}
        for param in best_params:
            param_deltas = [
                abs(p.score_delta) for p in perturbations
                if p.param_name == param and np.isfinite(p.score_delta)
            ]
            sensitivity[param] = float(np.mean(param_deltas)) if param_deltas else 0.0

        logger.info(
            "robustness_test: %d perturbations | worst_delta=%.4f | robustness=%.2f",
            len(perturbations), worst if np.isfinite(worst) else -999, robustness_score,
        )

        return RobustnessResult(
            base_params          = best_params,
            base_score           = float(base_score),
            perturbations        = perturbations,
            worst_degradation    = worst,
            mean_degradation     = mean_delta,
            robustness_score     = robustness_score,
            sensitivity_by_param = sensitivity,
        )

    # ------------------------------------------------------------------
    # plot_stability_dashboard
    # ------------------------------------------------------------------

    def plot_stability_dashboard(
        self,
        wf_result:  WFResult,
        save_path:  Optional[str] = None,
        show:       bool = True,
    ) -> None:
        """
        Plot a comprehensive stability dashboard for a WFResult.

        Includes:
        - Fold-by-fold IS vs OOS Sharpe bar chart
        - Rolling OOS Sharpe (if sufficient data)
        - Parameter selection frequency per fold (heatmap)
        - OOS equity curve with fold boundaries

        Parameters
        ----------
        wf_result : WFResult from WalkForwardEngine.run().
        save_path : optional path to save the figure.
        show      : if True, display interactively.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            logger.warning("matplotlib not available — skipping stability dashboard")
            return

        successful = [fr for fr in wf_result.fold_results if fr.success]
        if not successful:
            logger.warning("No successful folds to plot")
            return

        fig = plt.figure(figsize=(16, 12))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_sharpe  = fig.add_subplot(gs[0, 0])
        ax_equity  = fig.add_subplot(gs[0, 1])
        ax_params  = fig.add_subplot(gs[1, 0])
        ax_rolling = fig.add_subplot(gs[1, 1])

        fold_ids   = [fr.fold_id        for fr in successful]
        is_sharpes = [fr.is_stats.sharpe for fr in successful]
        oos_sharpes = [fr.oos_sharpe     for fr in successful]

        # ── IS vs OOS Sharpe bars ──────────────────────────────────────────
        x = np.arange(len(fold_ids))
        w = 0.35
        bars_is  = ax_sharpe.bar(x - w/2, is_sharpes,  w, label="IS Sharpe",  color="#3498db", alpha=0.8)
        bars_oos = ax_sharpe.bar(x + w/2, oos_sharpes, w, label="OOS Sharpe", color="#e74c3c", alpha=0.8)
        ax_sharpe.axhline(0, color="black", linewidth=0.8)
        ax_sharpe.set_xticks(x); ax_sharpe.set_xticklabels([f"F{fid}" for fid in fold_ids])
        ax_sharpe.set_ylabel("Sharpe Ratio")
        ax_sharpe.set_title("IS vs OOS Sharpe by Fold")
        ax_sharpe.legend()
        ax_sharpe.grid(True, axis="y", alpha=0.3)

        # ── OOS Equity Curve ──────────────────────────────────────────────
        equity = self.starting_equity
        eq_curve = [equity]
        fold_boundaries = [0]

        for fr in sorted(successful, key=lambda f: f.fold_id):
            for trade in fr.oos_trades:
                pnl = float(trade.get("pnl", 0.0)) if isinstance(trade, dict) else getattr(trade, "pnl", 0.0)
                equity += pnl
                eq_curve.append(equity)
            fold_boundaries.append(len(eq_curve) - 1)

        eq_series = pd.Series(eq_curve)
        color     = "#2ecc71" if eq_series.iloc[-1] > self.starting_equity else "#e74c3c"
        ax_equity.plot(eq_series.index, eq_series.values, color=color, linewidth=1.5)

        # Mark fold boundaries
        for boundary in fold_boundaries[1:-1]:
            ax_equity.axvline(boundary, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

        ax_equity.axhline(self.starting_equity, color="black", linestyle="--",
                          linewidth=0.8, alpha=0.5, label="Start equity")
        ax_equity.set_ylabel("Equity ($)")
        ax_equity.set_title("Combined OOS Equity Curve")
        ax_equity.legend(fontsize=8)
        ax_equity.grid(True, alpha=0.3)

        # ── Parameter Selection Heatmap ───────────────────────────────────
        all_param_names = sorted(set().union(*[set(fr.params.keys()) for fr in successful]))
        n_folds  = len(successful)
        n_params = len(all_param_names)

        if n_params > 0:
            param_matrix = np.zeros((n_params, n_folds))
            for fi, fr in enumerate(sorted(successful, key=lambda f: f.fold_id)):
                for pi, param in enumerate(all_param_names):
                    val = fr.params.get(param, np.nan)
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        param_matrix[pi, fi] = float(val)

            # Normalise rows to [0, 1] for colour scale
            for pi in range(n_params):
                row     = param_matrix[pi]
                row_min = row.min()
                row_max = row.max()
                if row_max - row_min > 1e-10:
                    param_matrix[pi] = (row - row_min) / (row_max - row_min)

            im = ax_params.imshow(param_matrix, aspect="auto", cmap="viridis",
                                  interpolation="nearest")
            ax_params.set_yticks(range(n_params))
            ax_params.set_yticklabels(all_param_names, fontsize=8)
            ax_params.set_xticks(range(n_folds))
            ax_params.set_xticklabels([f"F{fr.fold_id}" for fr in sorted(successful, key=lambda f: f.fold_id)])
            ax_params.set_title("Parameter Values per Fold (normalised)")
            plt.colorbar(im, ax=ax_params, label="Normalised value")

        # ── Rolling Sharpe ────────────────────────────────────────────────
        all_pnl: List[float] = []
        for fr in sorted(successful, key=lambda f: f.fold_id):
            for trade in fr.oos_trades:
                pnl = float(trade.get("pnl", 0.0)) if isinstance(trade, dict) else 0.0
                pos = float(trade.get("dollar_pos", 1.0)) if isinstance(trade, dict) else 1.0
                if abs(pos) > 1e-6:
                    all_pnl.append(pnl / pos)
                else:
                    all_pnl.append(0.0)

        if len(all_pnl) >= 30:
            window = min(60, len(all_pnl) // 3)
            roll_sr = rolling_sharpe(pd.Series(all_pnl), window=window)
            ax_rolling.plot(roll_sr.index, roll_sr.values, color="#9b59b6", linewidth=1.5)
            ax_rolling.axhline(0, color="black", linewidth=0.8)
            ax_rolling.fill_between(
                roll_sr.index, roll_sr.values, 0,
                where=(roll_sr.values >= 0), alpha=0.2, color="green",
            )
            ax_rolling.fill_between(
                roll_sr.index, roll_sr.values, 0,
                where=(roll_sr.values < 0), alpha=0.2, color="red",
            )
            ax_rolling.set_ylabel("Rolling Sharpe")
            ax_rolling.set_title(f"Rolling {window}-Trade Sharpe (OOS)")
            ax_rolling.grid(True, alpha=0.3)
        else:
            ax_rolling.text(0.5, 0.5, "Insufficient OOS data\nfor rolling Sharpe",
                            ha="center", va="center", transform=ax_rolling.transAxes)
            ax_rolling.set_title("Rolling Sharpe (OOS)")

        # Global title
        fig.suptitle(
            f"Walk-Forward Stability Dashboard  |  "
            f"OOS Sharpe: {wf_result.oos_sharpe:.3f}  |  "
            f"Param Stability: {wf_result.param_stability_score:.1%}  |  "
            f"Folds: {wf_result.n_folds}",
            fontsize=12, y=1.01,
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved stability dashboard: %s", save_path)
        if show:
            plt.show()
        plt.close(fig)
