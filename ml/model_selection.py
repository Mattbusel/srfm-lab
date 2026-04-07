"""
ml/model_selection.py
Model selection and validation for live trading ML models.

Implements time-series aware cross-validation, Bayesian hyperparameter
search, walk-forward analysis, and model drift detection.

No em dashes. Uses numpy, scipy, pandas.
"""

from __future__ import annotations

import math
import sqlite3
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm


# ---------------------------------------------------------------------------
# PurgedCVSplit
# ---------------------------------------------------------------------------

class PurgedCVSplit:
    """
    Time-series cross-validation with purging and embargo.

    Purging: removes training samples whose label period overlaps with
    the test period (prevents leakage when labels span multiple bars).

    Embargo: removes the N bars immediately after each training fold
    to prevent leakage from serial correlation.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    embargo_pct : float
        Fraction of training samples to embargo (e.g. 0.01 = 1%).
    purge_pct : float
        Fraction of training samples to purge before test fold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_indices, test_indices) tuples.
        Train always precedes test in time. Test indices are non-overlapping.
        """
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        embargo_n = max(1, int(n * self.embargo_pct))
        purge_n = max(1, int(fold_size * self.purge_pct))

        for fold in range(self.n_splits):
            # Test set: fold-th segment starting from the middle
            test_start = (fold + 1) * fold_size
            test_end = min(test_start + fold_size, n)
            test_idx = np.arange(test_start, test_end)

            # Training set: all data before test, minus purge window and embargo
            train_end = test_start - embargo_n
            # Purge: remove samples near end of training (overlap with test labels)
            train_end = max(0, train_end - purge_n)
            train_idx = np.arange(0, train_end)

            if len(train_idx) < 2 or len(test_idx) < 2:
                continue

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV
# ---------------------------------------------------------------------------

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (Lopez de Prado, 2018).

    Generates all C(N, k) combinations of k test folds from N total folds.
    Provides multiple backtest paths for variance estimation.

    Each combination produces a unique train/test split with proper
    purging and embargo to prevent lookahead bias.
    """

    def __init__(
        self,
        n_folds: int = 6,
        n_test_folds: int = 2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
    ) -> None:
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate all C(n_folds, n_test_folds) (train, test) index pairs.
        """
        n = len(X)
        fold_size = n // self.n_folds
        embargo_n = max(1, int(n * self.embargo_pct))
        purge_n = max(1, int(fold_size * self.purge_pct))

        # Define fold boundaries
        folds = []
        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else n
            folds.append((start, end))

        # Generate all combinations of n_test_folds from n_folds
        for test_fold_indices in combinations(range(self.n_folds), self.n_test_folds):
            # Collect test indices
            test_idx_list = []
            for fi in test_fold_indices:
                test_idx_list.extend(range(folds[fi][0], folds[fi][1]))
            test_idx = np.array(sorted(test_idx_list))

            # Build training indices: all indices not in test, minus embargo/purge
            test_set = set(test_idx.tolist())
            min_test = min(test_idx)
            max_test = max(test_idx)

            train_idx_list = []
            for i in range(n):
                if i in test_set:
                    continue
                # Embargo: skip bars immediately before test period
                if max_test - embargo_n <= i < min_test:
                    continue
                # Purge: skip bars at the transition boundary
                if min_test - purge_n <= i < min_test:
                    continue
                # Embargo after test
                if max_test <= i < max_test + embargo_n:
                    continue
                train_idx_list.append(i)

            train_idx = np.array(train_idx_list)
            if len(train_idx) < 10 or len(test_idx) < 5:
                continue

            yield train_idx, test_idx

    @property
    def n_combinations(self) -> int:
        from math import comb
        return comb(self.n_folds, self.n_test_folds)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_sharpe(returns: np.ndarray, annualize: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns))
    if std < 1e-10:
        return 0.0
    return float(np.mean(returns)) / std * math.sqrt(annualize)


def _compute_max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / (running_max + 1e-10)
    return float(np.min(drawdowns))


def _compute_icir(
    signals: np.ndarray, returns: np.ndarray, window: int = 20
) -> float:
    if len(signals) < window + 5:
        return 0.0
    from scipy.stats import spearmanr
    ics = []
    for i in range(window, len(signals)):
        ic, _ = spearmanr(signals[i - window: i], returns[i - window: i])
        if not math.isnan(ic):
            ics.append(ic)
    if len(ics) < 3:
        return 0.0
    arr = np.array(ics)
    std = arr.std()
    if std < 1e-10:
        return 0.0
    return float(arr.mean() / std)


@dataclass
class ModelEvalResult:
    model_name: str
    sharpe: float
    max_drawdown: float
    icir: float
    accuracy: float
    sharpe_ci: Tuple[float, float] = (0.0, 0.0)
    icir_ci: Tuple[float, float] = (0.0, 0.0)
    n_test_samples: int = 0


# ---------------------------------------------------------------------------
# ModelSelector
# ---------------------------------------------------------------------------

class ModelSelector:
    """
    Selects the best model from candidates using CPCV or purged k-fold.

    For each candidate model:
    1. Runs CPCV splits on historical data.
    2. Trains model on each training fold and evaluates on test fold.
    3. Aggregates OOS Sharpe, max drawdown, and ICIR.
    4. Computes confidence intervals via bootstrap.
    5. Returns the best model by Sharpe (with optional constraints).
    """

    def __init__(
        self,
        cv_strategy: str = "purged",
        n_splits: int = 5,
        n_bootstrap: int = 200,
        min_sharpe: float = 0.0,
        embargo_pct: float = 0.01,
    ) -> None:
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.n_bootstrap = n_bootstrap
        self.min_sharpe = min_sharpe
        self.embargo_pct = embargo_pct

    def _get_cv(self) -> Any:
        if self.cv_strategy == "cpcv":
            return CombinatorialPurgedCV(
                n_folds=self.n_splits, n_test_folds=2, embargo_pct=self.embargo_pct
            )
        return PurgedCVSplit(n_splits=self.n_splits, embargo_pct=self.embargo_pct)

    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "model",
    ) -> ModelEvalResult:
        """
        Evaluate a single model via cross-validation.
        Model must have fit_one(x, y) and predict(x) methods.
        y: continuous returns (not binary).
        """
        cv = self._get_cv()
        oos_signals: List[float] = []
        oos_returns: List[float] = []
        oos_pnl: List[float] = []

        for train_idx, test_idx in cv.split(X, y):
            model.reset()

            # Train
            for i in train_idx:
                model.fit_one(X[i], 1.0 if y[i] > 0 else 0.0)

            # Test
            for i in test_idx:
                pred = model.predict(X[i])
                oos_signals.append(pred)
                oos_returns.append(float(y[i]))
                oos_pnl.append(pred * float(y[i]))

        if len(oos_pnl) < 5:
            return ModelEvalResult(
                model_name=model_name,
                sharpe=0.0, max_drawdown=0.0,
                icir=0.0, accuracy=0.5,
                n_test_samples=0,
            )

        pnl_arr = np.array(oos_pnl)
        sig_arr = np.array(oos_signals)
        ret_arr = np.array(oos_returns)

        sharpe = _compute_sharpe(pnl_arr)
        max_dd = _compute_max_drawdown(pnl_arr)
        icir = _compute_icir(sig_arr, ret_arr)
        accuracy = float(np.mean(np.sign(sig_arr) == np.sign(ret_arr)))

        # Bootstrap confidence intervals
        sharpe_samples = []
        icir_samples = []
        for _ in range(self.n_bootstrap):
            idx = np.random.randint(0, len(pnl_arr), size=len(pnl_arr))
            sharpe_samples.append(_compute_sharpe(pnl_arr[idx]))
            icir_samples.append(_compute_icir(sig_arr[idx], ret_arr[idx]))

        sharpe_ci = (
            float(np.percentile(sharpe_samples, 5)),
            float(np.percentile(sharpe_samples, 95)),
        )
        icir_ci = (
            float(np.percentile(icir_samples, 5)),
            float(np.percentile(icir_samples, 95)),
        )

        return ModelEvalResult(
            model_name=model_name,
            sharpe=sharpe,
            max_drawdown=max_dd,
            icir=icir,
            accuracy=accuracy,
            sharpe_ci=sharpe_ci,
            icir_ci=icir_ci,
            n_test_samples=len(oos_pnl),
        )

    def select(
        self,
        candidates: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[str, Any, List[ModelEvalResult]]:
        """
        Evaluate all candidate models and return the best one.
        Returns (best_name, best_model, all_results).
        """
        results = []
        for name, model in candidates.items():
            result = self.evaluate_model(model, X, y, model_name=name)
            results.append(result)

        # Sort by Sharpe, filtering by minimum constraint
        valid = [r for r in results if r.sharpe >= self.min_sharpe]
        if not valid:
            valid = results

        best = max(valid, key=lambda r: r.sharpe)
        return best.model_name, candidates[best.model_name], results


# ---------------------------------------------------------------------------
# HyperparameterSearch
# ---------------------------------------------------------------------------

@dataclass
class _GPState:
    """Simple GP surrogate state for Bayesian optimization."""
    X_obs: List[np.ndarray] = field(default_factory=list)
    y_obs: List[float] = field(default_factory=list)


class HyperparameterSearch:
    """
    Bayesian hyperparameter optimization with a simple GP surrogate
    and Upper Confidence Bound (UCB) acquisition function.

    Parameters
    ----------
    param_grid : dict
        Parameter grid, mapping param name -> (low, high) range or list of values.
    n_trials : int
        Number of evaluations.
    kappa : float
        UCB exploration parameter.
    n_random_init : int
        Number of random initialization trials before GP kicks in.
    """

    def __init__(
        self,
        param_grid: Dict[str, Any],
        n_trials: int = 20,
        kappa: float = 2.0,
        n_random_init: int = 5,
        n_workers: int = 1,
    ) -> None:
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.kappa = kappa
        self.n_random_init = n_random_init
        self.n_workers = n_workers
        self._param_names = list(param_grid.keys())
        self._gp = _GPState()

    def _encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameter dict to a normalized [0,1]^d vector."""
        vec = []
        for name in self._param_names:
            val = params[name]
            spec = self.param_grid[name]
            if isinstance(spec, (list, tuple)) and len(spec) == 2 and not isinstance(spec[0], str):
                lo, hi = float(spec[0]), float(spec[1])
                vec.append((float(val) - lo) / (hi - lo + 1e-10))
            elif isinstance(spec, list):
                idx = spec.index(val) if val in spec else 0
                vec.append(float(idx) / max(1, len(spec) - 1))
            else:
                vec.append(0.5)
        return np.array(vec)

    def _sample_random(self) -> Dict[str, Any]:
        params = {}
        for name, spec in self.param_grid.items():
            if isinstance(spec, list):
                if isinstance(spec[0], str):
                    params[name] = np.random.choice(spec)
                else:
                    params[name] = float(np.random.choice(spec))
            elif isinstance(spec, tuple) and len(spec) == 2:
                lo, hi = spec
                params[name] = float(np.random.uniform(lo, hi))
            else:
                params[name] = spec
        return params

    def _gp_ucb(self, x: np.ndarray) -> float:
        """UCB acquisition: mu + kappa * sigma."""
        if len(self._gp.X_obs) < 2:
            return float(np.random.randn())
        X = np.array(self._gp.X_obs)
        y = np.array(self._gp.y_obs)

        # Simple RBF kernel
        def rbf(a: np.ndarray, b: np.ndarray, ls: float = 0.3) -> float:
            return float(math.exp(-np.sum((a - b) ** 2) / (2 * ls ** 2)))

        k_xx = np.array([[rbf(X[i], X[j]) for j in range(len(X))] for i in range(len(X))])
        k_xx += 1e-4 * np.eye(len(X))
        k_xs = np.array([rbf(X[i], x) for i in range(len(X))])

        try:
            K_inv_y = np.linalg.solve(k_xx, y)
            K_inv_k = np.linalg.solve(k_xx, k_xs)
            mu = float(k_xs @ K_inv_y)
            var = max(0.0, 1.0 - float(k_xs @ K_inv_k))
            return mu + self.kappa * math.sqrt(var)
        except np.linalg.LinAlgError:
            return float(np.random.randn())

    def _suggest_next(self) -> Dict[str, Any]:
        """Suggest the next parameter set to evaluate."""
        if len(self._gp.X_obs) < self.n_random_init:
            return self._sample_random()

        # Grid search over random candidates, pick best UCB
        best_ucb = -float("inf")
        best_params = self._sample_random()
        for _ in range(50):
            candidate = self._sample_random()
            x = self._encode_params(candidate)
            ucb = self._gp_ucb(x)
            if ucb > best_ucb:
                best_ucb = ucb
                best_params = candidate

        return best_params

    def search(
        self,
        objective: Callable[[Dict[str, Any]], float],
    ) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
        """
        Run Bayesian optimization.
        objective: function that takes params dict and returns a score (higher is better).
        Returns (best_params, best_score, all_trials).
        """
        history: List[Tuple[Dict[str, Any], float]] = []
        best_params = self._sample_random()
        best_score = -float("inf")

        for trial in range(self.n_trials):
            params = self._suggest_next()
            try:
                score = float(objective(params))
            except Exception:
                score = -float("inf")

            history.append((params, score))
            x = self._encode_params(params)
            self._gp.X_obs.append(x)
            self._gp.y_obs.append(score)

            if score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score, history


# ---------------------------------------------------------------------------
# WalkForwardValidator
# ---------------------------------------------------------------------------

@dataclass
class WFAResult:
    is_sharpe: float       # in-sample Sharpe
    oos_sharpe: float      # out-of-sample Sharpe
    wfa_efficiency: float  # OOS Sharpe / IS Sharpe
    n_periods: int
    period_results: List[Dict[str, float]]


class WalkForwardValidator:
    """
    Rolling window walk-forward analysis.

    1. Train on first train_window bars.
    2. Evaluate on next test_window bars.
    3. Roll window forward by step_size.
    4. Aggregate OOS performance.
    5. Compute WFA efficiency = OOS Sharpe / IS Sharpe.
    """

    def __init__(
        self,
        train_window: int = 200,
        test_window: int = 50,
        step_size: int = 25,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> WFAResult:
        """
        Run WFA on the given model and data.
        Model must have fit_one(x, label) and predict(x) methods.
        y: continuous returns.
        """
        n = len(X)
        period_results = []
        all_oos_pnl = []
        all_is_pnl = []

        start = 0
        while start + self.train_window + self.test_window <= n:
            train_idx = np.arange(start, start + self.train_window)
            test_idx = np.arange(
                start + self.train_window,
                min(start + self.train_window + self.test_window, n),
            )

            model.reset()

            # IS training and evaluation
            is_pnl = []
            for i in train_idx:
                pred = model.predict(X[i])
                is_pnl.append(pred * float(y[i]))
                model.fit_one(X[i], 1.0 if y[i] > 0 else 0.0)

            # OOS evaluation (no training)
            oos_pnl = []
            for i in test_idx:
                pred = model.predict(X[i])
                oos_pnl.append(pred * float(y[i]))

            is_sharpe = _compute_sharpe(np.array(is_pnl))
            oos_sharpe = _compute_sharpe(np.array(oos_pnl))

            period_results.append({
                "start": int(start),
                "train_end": int(start + self.train_window),
                "test_end": int(start + self.train_window + len(test_idx)),
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
            })

            all_is_pnl.extend(is_pnl)
            all_oos_pnl.extend(oos_pnl)
            start += self.step_size

        if not period_results:
            return WFAResult(
                is_sharpe=0.0, oos_sharpe=0.0,
                wfa_efficiency=0.0, n_periods=0,
                period_results=[],
            )

        is_sharpe_total = _compute_sharpe(np.array(all_is_pnl))
        oos_sharpe_total = _compute_sharpe(np.array(all_oos_pnl))
        wfa_eff = (
            oos_sharpe_total / (abs(is_sharpe_total) + 1e-10)
            if abs(is_sharpe_total) > 1e-10
            else 0.0
        )

        return WFAResult(
            is_sharpe=is_sharpe_total,
            oos_sharpe=oos_sharpe_total,
            wfa_efficiency=float(wfa_eff),
            n_periods=len(period_results),
            period_results=period_results,
        )


# ---------------------------------------------------------------------------
# ModelDriftDetector
# ---------------------------------------------------------------------------

@dataclass
class DriftEvent:
    timestamp: float
    model_name: str
    drift_type: str
    cusum_stat: float
    n_observations: int


class ModelDriftDetector:
    """
    Monitors model performance degradation via CUSUM test.

    CUSUM detects a shift in the mean of the performance series.
    When drift is detected, logs to SQLite and signals for retraining.

    Parameters
    ----------
    model_name : str
        Name of model being monitored.
    threshold : float
        CUSUM threshold h (drift fires when CUSUM statistic exceeds this).
    slack : float
        Allowable drift magnitude k (half the minimum shift to detect).
    db_path : str
        SQLite database path for logging.
    min_samples : int
        Minimum samples before drift detection is active.
    """

    def __init__(
        self,
        model_name: str = "model",
        threshold: float = 5.0,
        slack: float = 0.5,
        db_path: str = "model_performance.db",
        min_samples: int = 30,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.slack = slack
        self.db_path = db_path
        self.min_samples = min_samples

        self._cusum_pos = 0.0  # upper CUSUM (detects upward shift in loss)
        self._cusum_neg = 0.0  # lower CUSUM (detects downward shift)
        self._n = 0
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._warmup_buf: deque = deque(maxlen=min_samples)
        self._drift_events: List[DriftEvent] = []
        self._last_drift_n = 0
        self._init_db()

    def _init_db(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS drift_events (
                    ts REAL,
                    model_name TEXT,
                    drift_type TEXT,
                    cusum_stat REAL,
                    n_observations INTEGER
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def update(self, performance_metric: float) -> bool:
        """
        Update CUSUM with new performance observation.
        Returns True if drift is detected, False otherwise.
        performance_metric: e.g. rolling accuracy or PnL.
        Higher is better (drift = sustained decrease).
        """
        self._n += 1
        self._warmup_buf.append(performance_metric)

        # Calibrate baseline from first min_samples observations
        if len(self._warmup_buf) == self.min_samples and self._baseline_mean is None:
            arr = np.array(list(self._warmup_buf))
            self._baseline_mean = float(arr.mean())
            self._baseline_std = max(float(arr.std()), 1e-4)

        if self._baseline_mean is None or self._n < self.min_samples:
            return False

        # Standardize
        z = (performance_metric - self._baseline_mean) / self._baseline_std

        # CUSUM update (detect downward shift = degradation)
        self._cusum_pos = max(0.0, self._cusum_pos + z - self.slack)
        self._cusum_neg = max(0.0, self._cusum_neg - z - self.slack)

        # Check threshold
        drift_detected = False
        if self._cusum_neg > self.threshold and (self._n - self._last_drift_n) > self.min_samples:
            event = DriftEvent(
                timestamp=time.time(),
                model_name=self.model_name,
                drift_type="DEGRADATION",
                cusum_stat=self._cusum_neg,
                n_observations=self._n,
            )
            self._drift_events.append(event)
            self._log_drift(event)
            self._cusum_neg = 0.0
            self._cusum_pos = 0.0
            self._last_drift_n = self._n
            drift_detected = True

        if self._cusum_pos > self.threshold * 2 and (self._n - self._last_drift_n) > self.min_samples:
            # Large positive shift (regime improvement) - reset baseline
            arr = np.array(list(self._warmup_buf)[-self.min_samples:])
            self._baseline_mean = float(arr.mean())
            self._baseline_std = max(float(arr.std()), 1e-4)
            self._cusum_pos = 0.0

        return drift_detected

    def _log_drift(self, event: DriftEvent) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO drift_events VALUES (?,?,?,?,?)",
                (
                    event.timestamp,
                    event.model_name,
                    event.drift_type,
                    event.cusum_stat,
                    event.n_observations,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def reset(self) -> None:
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._baseline_mean = None
        self._baseline_std = None
        self._warmup_buf.clear()
        self._n = 0

    @property
    def drift_count(self) -> int:
        return len(self._drift_events)

    @property
    def cusum_stat(self) -> float:
        return self._cusum_neg

    def get_recent_events(self, n: int = 10) -> List[DriftEvent]:
        return self._drift_events[-n:]

    def should_retrain(self) -> bool:
        """Returns True if retraining is recommended."""
        if not self._drift_events:
            return False
        last_event = self._drift_events[-1]
        return (self._n - self._last_drift_n) < self.min_samples * 2
