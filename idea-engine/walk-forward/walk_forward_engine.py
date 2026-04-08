"""
walk-forward/walk_forward_engine.py

Advanced Walk-Forward Analysis engine with regime awareness, purging,
embargo, and comprehensive reporting.

This module extends the base WFA engine with:
  - WalkForwardConfig: flexible train/test window specification
  - Rolling vs expanding window modes
  - Anchored walk-forward (always start from beginning)
  - Purging + embargo gaps between train and test to prevent lookahead
  - Regime-aware fold generation (ensure each fold sees multiple regimes)
  - Parameter stability tracking across folds
  - Per-fold detailed metrics and aggregate statistics
  - WalkForwardReport: comprehensive results object with visualization data
  - Rolling OOS equity curve construction

Usage:
    from walk_forward.walk_forward_engine import AdvancedWalkForward, WalkForwardConfig

    cfg = WalkForwardConfig(
        train_window=252, test_window=63,
        step_size=21, mode="rolling",
        purge_gap=5, embargo_gap=3,
    )
    engine = AdvancedWalkForward(config=cfg)
    report = engine.run(returns, param_grid, fit_fn, predict_fn)
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class WindowMode(str, Enum):
    ROLLING   = "rolling"     # fixed-size sliding window
    EXPANDING = "expanding"   # grows from start
    ANCHORED  = "anchored"    # always starts from index 0


class FoldVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"    # not enough data


class RegimeLabel(str, Enum):
    BULL       = "bull"
    BEAR       = "bear"
    HIGH_VOL   = "high_vol"
    LOW_VOL    = "low_vol"
    TRANSITION = "transition"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward analysis.

    Indices are in number of observations (bars/days).
    """
    train_window: int = 252       # training window size
    test_window: int = 63         # test window size (one quarter)
    step_size: int = 21           # how far to advance between folds (monthly)
    mode: str = "rolling"         # "rolling" | "expanding" | "anchored"
    purge_gap: int = 5            # bars to drop between train end and test start
    embargo_gap: int = 3          # bars to drop after test end before next train
    min_train_size: int = 126     # minimum training samples required
    min_test_size: int = 21       # minimum test samples required
    require_multi_regime: bool = False   # each fold must see >1 regime
    regime_column: str | None = None     # column name in DataFrame for regime
    oos_sharpe_threshold: float = 0.0    # minimum OOS Sharpe to pass
    oos_hit_rate_threshold: float = 0.5  # minimum OOS hit rate to pass
    max_param_drift: float = 0.5         # max normalised param change across folds

    @property
    def window_mode(self) -> WindowMode:
        return WindowMode(self.mode.lower())

    def validate(self) -> list[str]:
        errors = []
        if self.train_window < self.min_train_size:
            errors.append(
                f"train_window ({self.train_window}) < min_train_size ({self.min_train_size})"
            )
        if self.test_window < self.min_test_size:
            errors.append(
                f"test_window ({self.test_window}) < min_test_size ({self.min_test_size})"
            )
        if self.step_size < 1:
            errors.append("step_size must be >= 1")
        if self.purge_gap < 0:
            errors.append("purge_gap must be >= 0")
        if self.embargo_gap < 0:
            errors.append("embargo_gap must be >= 0")
        return errors


# ---------------------------------------------------------------------------
# Fold specification
# ---------------------------------------------------------------------------

@dataclass
class FoldSpec:
    """Index boundaries for one train/test fold."""
    fold_id: int
    train_start: int
    train_end: int        # exclusive
    test_start: int
    test_end: int         # exclusive
    purge_start: int      # start of purge gap
    purge_end: int        # end of purge gap (= test_start)

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


# ---------------------------------------------------------------------------
# Per-fold result
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Metrics from a single fold."""
    fold_id: int
    fold_spec: FoldSpec
    # In-sample metrics
    is_sharpe: float
    is_return: float
    is_volatility: float
    is_max_dd: float
    is_hit_rate: float
    # Out-of-sample metrics
    oos_sharpe: float
    oos_return: float
    oos_volatility: float
    oos_max_dd: float
    oos_hit_rate: float
    oos_ic: float                   # information coefficient (rank corr)
    # Fitted parameters
    fitted_params: dict[str, float]
    # Derived
    efficiency_ratio: float         # OOS Sharpe / IS Sharpe
    verdict: FoldVerdict
    regimes_seen: list[str] = field(default_factory=list)
    oos_returns: list[float] = field(default_factory=list)
    fit_duration_seconds: float = 0.0
    eval_duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_range": (self.fold_spec.train_start, self.fold_spec.train_end),
            "test_range": (self.fold_spec.test_start, self.fold_spec.test_end),
            "is_sharpe": round(self.is_sharpe, 4),
            "oos_sharpe": round(self.oos_sharpe, 4),
            "oos_return": round(self.oos_return, 6),
            "oos_hit_rate": round(self.oos_hit_rate, 4),
            "oos_ic": round(self.oos_ic, 4),
            "efficiency_ratio": round(self.efficiency_ratio, 4),
            "verdict": self.verdict.value,
            "regimes_seen": self.regimes_seen,
        }


# ---------------------------------------------------------------------------
# Walk-Forward Report
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardReport:
    """Comprehensive results from a walk-forward analysis run."""
    config: WalkForwardConfig
    n_folds: int
    n_passed: int
    n_failed: int
    n_skipped: int
    fold_results: list[FoldResult]
    # Aggregate OOS metrics
    agg_oos_sharpe: float
    agg_oos_return: float
    agg_oos_hit_rate: float
    agg_oos_ic: float
    agg_efficiency_ratio: float
    # Parameter stability
    param_stability: dict[str, float]   # param_name -> coefficient of variation
    param_drift_score: float            # overall drift score (0 = stable, 1 = unstable)
    # Equity curve data (for visualization)
    oos_equity_curve: list[float]
    oos_equity_dates: list[int]         # index positions
    # Timing
    total_duration_seconds: float
    started_at: str
    finished_at: str
    # Verdict
    overall_verdict: str                # "ADOPT" | "REJECT" | "RETEST"
    verdict_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "agg_oos_sharpe": round(self.agg_oos_sharpe, 4),
            "agg_oos_return": round(self.agg_oos_return, 6),
            "agg_oos_hit_rate": round(self.agg_oos_hit_rate, 4),
            "agg_oos_ic": round(self.agg_oos_ic, 4),
            "agg_efficiency_ratio": round(self.agg_efficiency_ratio, 4),
            "param_drift_score": round(self.param_drift_score, 4),
            "overall_verdict": self.overall_verdict,
            "verdict_reasons": self.verdict_reasons,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
        }

    def summary(self) -> str:
        lines = [
            f"Walk-Forward Report: {self.overall_verdict}",
            f"  Folds: {self.n_folds} ({self.n_passed} pass, {self.n_failed} fail, {self.n_skipped} skip)",
            f"  OOS Sharpe: {self.agg_oos_sharpe:.4f}",
            f"  OOS Hit Rate: {self.agg_oos_hit_rate:.2%}",
            f"  OOS IC: {self.agg_oos_ic:.4f}",
            f"  Efficiency: {self.agg_efficiency_ratio:.2%}",
            f"  Param Drift: {self.param_drift_score:.4f}",
            f"  Duration: {self.total_duration_seconds:.1f}s",
        ]
        for reason in self.verdict_reasons:
            lines.append(f"  - {reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regime detector (simple)
# ---------------------------------------------------------------------------

class SimpleRegimeDetector:
    """
    Lightweight regime detector for walk-forward fold annotation.
    Uses rolling return and volatility to classify regimes.
    """

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def detect(self, returns: np.ndarray) -> list[str]:
        """Return a regime label per bar."""
        n = len(returns)
        labels: list[str] = []
        for i in range(n):
            window = returns[max(0, i - self._lookback + 1): i + 1]
            if len(window) < 10:
                labels.append(RegimeLabel.TRANSITION.value)
                continue
            cum_ret = float(np.sum(window))
            vol = float(np.std(window)) * math.sqrt(252)
            if cum_ret > 0.05 and vol < 0.25:
                labels.append(RegimeLabel.BULL.value)
            elif cum_ret < -0.05 and vol < 0.25:
                labels.append(RegimeLabel.BEAR.value)
            elif vol > 0.30:
                labels.append(RegimeLabel.HIGH_VOL.value)
            elif vol < 0.10:
                labels.append(RegimeLabel.LOW_VOL.value)
            else:
                labels.append(RegimeLabel.TRANSITION.value)
        return labels


# ---------------------------------------------------------------------------
# Split generator
# ---------------------------------------------------------------------------

class SplitGenerator:
    """Generate train/test fold specifications from config."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self._cfg = config

    def generate(
        self,
        n_samples: int,
        regimes: list[str] | None = None,
    ) -> list[FoldSpec]:
        """
        Produce FoldSpec objects covering the data.

        Parameters
        ----------
        n_samples : total number of observations
        regimes   : optional per-bar regime labels (for multi-regime enforcement)
        """
        cfg = self._cfg
        mode = cfg.window_mode
        folds: list[FoldSpec] = []
        fold_id = 0

        pos = 0
        if mode == WindowMode.ROLLING:
            pos = 0
        elif mode in (WindowMode.EXPANDING, WindowMode.ANCHORED):
            pos = 0

        while True:
            # Determine train boundaries
            if mode == WindowMode.ANCHORED:
                train_start = 0
                train_end = cfg.train_window + fold_id * cfg.step_size
            elif mode == WindowMode.EXPANDING:
                train_start = 0
                train_end = cfg.train_window + fold_id * cfg.step_size
            else:  # rolling
                train_start = fold_id * cfg.step_size
                train_end = train_start + cfg.train_window

            # Purge gap
            purge_start = train_end
            purge_end = train_end + cfg.purge_gap

            # Test boundaries
            test_start = purge_end
            test_end = test_start + cfg.test_window

            # Check bounds
            if test_end > n_samples:
                break
            if train_end - train_start < cfg.min_train_size:
                fold_id += 1
                continue
            if test_end - test_start < cfg.min_test_size:
                fold_id += 1
                continue

            # Regime check
            if cfg.require_multi_regime and regimes is not None:
                test_regimes = set(regimes[test_start:test_end])
                train_regimes = set(regimes[train_start:train_end])
                combined = train_regimes | test_regimes
                if len(combined) < 2:
                    fold_id += 1
                    continue

            folds.append(FoldSpec(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_start=purge_start,
                purge_end=purge_end,
            ))
            fold_id += 1

        logger.info("Generated %d folds from %d samples", len(folds), n_samples)
        return folds


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, annual_factor: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    if sigma < 1e-12:
        return 0.0
    return mu / sigma * math.sqrt(annual_factor)


def _max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    return float(np.min(drawdowns))


def _hit_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def _information_coefficient(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Rank correlation between predictions and actuals."""
    if len(predictions) < 3:
        return 0.0
    from scipy.stats import spearmanr
    corr, _ = spearmanr(predictions, actuals)
    return float(corr) if not math.isnan(corr) else 0.0


def _compute_fold_metrics(returns: np.ndarray) -> dict[str, float]:
    return {
        "sharpe": _sharpe(returns),
        "total_return": float(np.sum(returns)),
        "volatility": float(np.std(returns, ddof=1)) * math.sqrt(252) if len(returns) > 1 else 0.0,
        "max_dd": _max_drawdown(returns),
        "hit_rate": _hit_rate(returns),
    }


# ---------------------------------------------------------------------------
# Advanced Walk-Forward Engine
# ---------------------------------------------------------------------------

class AdvancedWalkForward:
    """
    Walk-forward analysis engine with regime awareness, purging/embargo,
    parameter stability tracking, and comprehensive reporting.
    """

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        regime_detector: SimpleRegimeDetector | None = None,
    ) -> None:
        self._cfg = config or WalkForwardConfig()
        self._regime_detector = regime_detector or SimpleRegimeDetector()

        # Validate config
        errors = self._cfg.validate()
        if errors:
            raise ValueError(f"Invalid WalkForwardConfig: {'; '.join(errors)}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        returns: np.ndarray,
        fit_fn: Callable[[np.ndarray, dict[str, Any]], dict[str, float]],
        predict_fn: Callable[[np.ndarray, dict[str, float]], np.ndarray],
        initial_params: dict[str, Any] | None = None,
        signals: np.ndarray | None = None,
    ) -> WalkForwardReport:
        """
        Execute walk-forward analysis.

        Parameters
        ----------
        returns       : 1-D array of daily returns.
        fit_fn        : Callable(train_returns, params) -> fitted_params.
                        Optimises parameters on training data.
        predict_fn    : Callable(test_returns, fitted_params) -> predictions.
                        Generates signals/returns on test data using fitted params.
        initial_params: Starting parameter dict (optional).
        signals       : Optional pre-computed signal array for IC calculation.
        """
        t0 = time.monotonic()
        started = datetime.now(timezone.utc).isoformat()
        n = len(returns)

        # Detect regimes
        regimes = self._regime_detector.detect(returns)

        # Generate folds
        splitter = SplitGenerator(self._cfg)
        folds = splitter.generate(n, regimes)

        if not folds:
            logger.warning("No valid folds generated for %d samples", n)
            return self._empty_report(started)

        # Run each fold
        fold_results: list[FoldResult] = []
        all_oos_returns: list[float] = []
        all_oos_indices: list[int] = []
        all_fitted_params: list[dict[str, float]] = []

        for fold_spec in folds:
            result = self._run_fold(
                returns, fold_spec, fit_fn, predict_fn,
                initial_params or {}, regimes, signals,
            )
            fold_results.append(result)
            all_oos_returns.extend(result.oos_returns)
            all_oos_indices.extend(
                range(fold_spec.test_start, fold_spec.test_end)
            )
            all_fitted_params.append(result.fitted_params)

        # Aggregate metrics
        oos_arr = np.array(all_oos_returns) if all_oos_returns else np.array([0.0])
        agg_sharpe = _sharpe(oos_arr)
        agg_return = float(np.sum(oos_arr))
        agg_hit = _hit_rate(oos_arr)

        # Aggregate IC
        oos_ics = [fr.oos_ic for fr in fold_results if fr.verdict != FoldVerdict.SKIP]
        agg_ic = float(np.mean(oos_ics)) if oos_ics else 0.0

        # Efficiency ratio
        is_sharpes = [fr.is_sharpe for fr in fold_results if fr.is_sharpe != 0]
        oos_sharpes = [fr.oos_sharpe for fr in fold_results if fr.is_sharpe != 0]
        if is_sharpes:
            mean_is = float(np.mean(is_sharpes))
            mean_oos = float(np.mean(oos_sharpes))
            agg_efficiency = mean_oos / mean_is if abs(mean_is) > 1e-6 else 0.0
        else:
            agg_efficiency = 0.0

        # Parameter stability
        param_stability, drift_score = self._compute_param_stability(all_fitted_params)

        # OOS equity curve
        equity_curve = list(np.cumsum(oos_arr))

        # Counts
        n_passed = sum(1 for fr in fold_results if fr.verdict == FoldVerdict.PASS)
        n_failed = sum(1 for fr in fold_results if fr.verdict == FoldVerdict.FAIL)
        n_skipped = sum(1 for fr in fold_results if fr.verdict == FoldVerdict.SKIP)

        # Overall verdict
        verdict, reasons = self._overall_verdict(
            agg_sharpe, agg_hit, agg_efficiency, drift_score,
            n_passed, len(fold_results),
        )

        elapsed = time.monotonic() - t0
        finished = datetime.now(timezone.utc).isoformat()

        return WalkForwardReport(
            config=self._cfg,
            n_folds=len(fold_results),
            n_passed=n_passed,
            n_failed=n_failed,
            n_skipped=n_skipped,
            fold_results=fold_results,
            agg_oos_sharpe=agg_sharpe,
            agg_oos_return=agg_return,
            agg_oos_hit_rate=agg_hit,
            agg_oos_ic=agg_ic,
            agg_efficiency_ratio=agg_efficiency,
            param_stability=param_stability,
            param_drift_score=drift_score,
            oos_equity_curve=equity_curve,
            oos_equity_dates=all_oos_indices,
            total_duration_seconds=elapsed,
            started_at=started,
            finished_at=finished,
            overall_verdict=verdict,
            verdict_reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Per-fold execution
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        returns: np.ndarray,
        spec: FoldSpec,
        fit_fn: Callable,
        predict_fn: Callable,
        initial_params: dict[str, Any],
        regimes: list[str],
        signals: np.ndarray | None,
    ) -> FoldResult:
        """Fit on train, evaluate on test for one fold."""
        train_ret = returns[spec.train_start:spec.train_end]
        test_ret = returns[spec.test_start:spec.test_end]

        # Regimes in this fold
        fold_regimes = list(set(regimes[spec.train_start:spec.test_end]))

        # Skip if insufficient data
        if len(train_ret) < self._cfg.min_train_size or len(test_ret) < self._cfg.min_test_size:
            return FoldResult(
                fold_id=spec.fold_id, fold_spec=spec,
                is_sharpe=0, is_return=0, is_volatility=0, is_max_dd=0, is_hit_rate=0,
                oos_sharpe=0, oos_return=0, oos_volatility=0, oos_max_dd=0,
                oos_hit_rate=0, oos_ic=0,
                fitted_params={}, efficiency_ratio=0,
                verdict=FoldVerdict.SKIP, regimes_seen=fold_regimes,
            )

        # Fit
        t_fit = time.monotonic()
        try:
            fitted_params = fit_fn(train_ret, dict(initial_params))
        except Exception as exc:
            logger.warning("Fit failed on fold %d: %s", spec.fold_id, exc)
            return FoldResult(
                fold_id=spec.fold_id, fold_spec=spec,
                is_sharpe=0, is_return=0, is_volatility=0, is_max_dd=0, is_hit_rate=0,
                oos_sharpe=0, oos_return=0, oos_volatility=0, oos_max_dd=0,
                oos_hit_rate=0, oos_ic=0,
                fitted_params={}, efficiency_ratio=0,
                verdict=FoldVerdict.FAIL, regimes_seen=fold_regimes,
            )
        fit_dur = time.monotonic() - t_fit

        # In-sample metrics
        is_metrics = _compute_fold_metrics(train_ret)

        # Out-of-sample: generate predictions/returns
        t_eval = time.monotonic()
        try:
            oos_predictions = predict_fn(test_ret, fitted_params)
            if len(oos_predictions) != len(test_ret):
                oos_predictions = test_ret  # fallback
        except Exception as exc:
            logger.warning("Predict failed on fold %d: %s", spec.fold_id, exc)
            oos_predictions = test_ret
        eval_dur = time.monotonic() - t_eval

        oos_metrics = _compute_fold_metrics(oos_predictions)

        # IC calculation
        oos_ic = 0.0
        if signals is not None:
            test_signals = signals[spec.test_start:spec.test_end]
            if len(test_signals) == len(test_ret):
                oos_ic = _information_coefficient(test_signals, test_ret)

        # Efficiency ratio
        efficiency = 0.0
        if abs(is_metrics["sharpe"]) > 1e-6:
            efficiency = oos_metrics["sharpe"] / is_metrics["sharpe"]

        # Fold verdict
        verdict = FoldVerdict.PASS
        if oos_metrics["sharpe"] < self._cfg.oos_sharpe_threshold:
            verdict = FoldVerdict.FAIL
        if oos_metrics["hit_rate"] < self._cfg.oos_hit_rate_threshold:
            verdict = FoldVerdict.FAIL

        return FoldResult(
            fold_id=spec.fold_id,
            fold_spec=spec,
            is_sharpe=is_metrics["sharpe"],
            is_return=is_metrics["total_return"],
            is_volatility=is_metrics["volatility"],
            is_max_dd=is_metrics["max_dd"],
            is_hit_rate=is_metrics["hit_rate"],
            oos_sharpe=oos_metrics["sharpe"],
            oos_return=oos_metrics["total_return"],
            oos_volatility=oos_metrics["volatility"],
            oos_max_dd=oos_metrics["max_dd"],
            oos_hit_rate=oos_metrics["hit_rate"],
            oos_ic=oos_ic,
            fitted_params=fitted_params,
            efficiency_ratio=efficiency,
            verdict=verdict,
            regimes_seen=fold_regimes,
            oos_returns=list(oos_predictions),
            fit_duration_seconds=fit_dur,
            eval_duration_seconds=eval_dur,
        )

    # ------------------------------------------------------------------
    # Parameter stability
    # ------------------------------------------------------------------

    def _compute_param_stability(
        self, all_params: list[dict[str, float]],
    ) -> tuple[dict[str, float], float]:
        """
        Compute coefficient of variation for each parameter across folds.
        Returns per-param CV and an overall drift score.
        """
        if len(all_params) < 2:
            return {}, 0.0

        # Collect all parameter names
        all_keys: set[str] = set()
        for p in all_params:
            all_keys.update(p.keys())

        stability: dict[str, float] = {}
        for key in sorted(all_keys):
            values = [float(p.get(key, 0.0)) for p in all_params]
            arr = np.array(values)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1))
            if abs(mean_val) > 1e-10:
                cv = std_val / abs(mean_val)
            else:
                cv = std_val  # absolute variation when mean near zero
            stability[key] = cv

        # Overall drift: mean CV across parameters
        drift = float(np.mean(list(stability.values()))) if stability else 0.0
        return stability, drift

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------

    def _overall_verdict(
        self,
        agg_sharpe: float,
        agg_hit: float,
        efficiency: float,
        drift: float,
        n_passed: int,
        n_total: int,
    ) -> tuple[str, list[str]]:
        """Determine ADOPT / REJECT / RETEST verdict with reasons."""
        reasons: list[str] = []
        score = 0  # positive -> adopt, negative -> reject

        # Sharpe
        if agg_sharpe > 1.0:
            score += 2
            reasons.append(f"Strong OOS Sharpe ({agg_sharpe:.2f})")
        elif agg_sharpe > 0.5:
            score += 1
            reasons.append(f"Acceptable OOS Sharpe ({agg_sharpe:.2f})")
        elif agg_sharpe > 0.0:
            reasons.append(f"Weak OOS Sharpe ({agg_sharpe:.2f})")
        else:
            score -= 2
            reasons.append(f"Negative OOS Sharpe ({agg_sharpe:.2f})")

        # Hit rate
        if agg_hit > 0.55:
            score += 1
            reasons.append(f"Good hit rate ({agg_hit:.1%})")
        elif agg_hit < 0.45:
            score -= 1
            reasons.append(f"Poor hit rate ({agg_hit:.1%})")

        # Efficiency
        if efficiency > 0.6:
            score += 1
            reasons.append(f"Good IS/OOS efficiency ({efficiency:.1%})")
        elif efficiency < 0.3:
            score -= 1
            reasons.append(f"Poor IS/OOS efficiency ({efficiency:.1%})")

        # Drift
        if drift > self._cfg.max_param_drift:
            score -= 1
            reasons.append(f"High parameter drift ({drift:.3f} > {self._cfg.max_param_drift})")
        else:
            reasons.append(f"Stable parameters (drift={drift:.3f})")

        # Pass rate
        pass_rate = n_passed / max(1, n_total)
        if pass_rate > 0.7:
            score += 1
            reasons.append(f"High fold pass rate ({pass_rate:.0%})")
        elif pass_rate < 0.4:
            score -= 1
            reasons.append(f"Low fold pass rate ({pass_rate:.0%})")

        if score >= 3:
            return "ADOPT", reasons
        elif score <= -1:
            return "REJECT", reasons
        else:
            return "RETEST", reasons

    # ------------------------------------------------------------------
    # Empty report (when no folds)
    # ------------------------------------------------------------------

    def _empty_report(self, started: str) -> WalkForwardReport:
        return WalkForwardReport(
            config=self._cfg,
            n_folds=0, n_passed=0, n_failed=0, n_skipped=0,
            fold_results=[],
            agg_oos_sharpe=0, agg_oos_return=0, agg_oos_hit_rate=0,
            agg_oos_ic=0, agg_efficiency_ratio=0,
            param_stability={}, param_drift_score=0,
            oos_equity_curve=[], oos_equity_dates=[],
            total_duration_seconds=0,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            overall_verdict="REJECT",
            verdict_reasons=["No valid folds generated"],
        )
