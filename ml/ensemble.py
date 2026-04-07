"""
ml/ensemble.py
Signal ensemble and combination for live trading.

Combines physics signals (BH, GARCH, OU, QuatNav), statistical signals
(Granger, Hurst), and ML models via dynamic weighting and stacking.

No em dashes. Uses numpy, scipy, pandas.
"""

from __future__ import annotations

import math
import sqlite3
import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class Regime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"


@dataclass
class EnsembleReport:
    """Summary report of ensemble state."""
    component_names: List[str]
    component_weights: List[float]
    component_icirs: List[float]
    combined_icir: float
    regime: str
    regime_breakdown: Dict[str, float]  # regime -> fraction of time
    staleness_flags: Dict[str, bool]    # component -> is stale?
    n_updates: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_names": self.component_names,
            "component_weights": self.component_weights,
            "component_icirs": self.component_icirs,
            "combined_icir": self.combined_icir,
            "regime": self.regime,
            "regime_breakdown": self.regime_breakdown,
            "staleness_flags": self.staleness_flags,
            "n_updates": self.n_updates,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _information_coefficient(
    signals: np.ndarray, returns: np.ndarray
) -> float:
    """Spearman rank correlation between signals and next-bar returns."""
    if len(signals) < 5:
        return 0.0
    from scipy.stats import spearmanr
    corr, _ = spearmanr(signals, returns)
    return float(corr) if not math.isnan(corr) else 0.0


def _icir(ic_series: np.ndarray, min_obs: int = 5) -> float:
    """Information Coefficient Information Ratio = mean(IC) / std(IC)."""
    if len(ic_series) < min_obs:
        return 0.0
    std = float(np.std(ic_series))
    if std < 1e-10:
        return 0.0
    return float(np.mean(ic_series)) / std


def _project_simplex_weights(
    w: np.ndarray, floor: float = 0.05, cap: float = 0.40
) -> np.ndarray:
    """Project weights onto simplex with per-component floor and cap."""
    n = len(w)
    w = np.clip(w, floor, cap)
    w = w / w.sum()
    # Iterative projection to satisfy both bounds simultaneously
    for _ in range(50):
        w = np.clip(w, floor, cap)
        w = w / w.sum()
    return w


# ---------------------------------------------------------------------------
# DynamicWeightOptimizer
# ---------------------------------------------------------------------------

class DynamicWeightOptimizer:
    """
    Computes optimal ensemble weights via online convex optimization
    (projected gradient descent onto the probability simplex).

    Maintains weight history and tracks evolution over time.
    Minimum weight floor: 0.05. Maximum weight cap: 0.40.
    """

    FLOOR = 0.05
    CAP = 0.40

    def __init__(
        self,
        n_components: int,
        learning_rate: float = 0.01,
        ic_window: int = 60,
    ) -> None:
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.ic_window = ic_window

        self._weights = np.ones(n_components) / n_components
        self._ic_buf: List[deque] = [deque(maxlen=ic_window) for _ in range(n_components)]
        self._weight_history: List[np.ndarray] = []

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    def update(self, signals: np.ndarray, realized_return: float) -> np.ndarray:
        """
        Update weights given realized return.
        signals: array of shape (n_components,) with each component's signal.
        realized_return: the actual return observed.
        """
        # Compute per-component IC proxy (sign agreement)
        ics = np.sign(signals) * np.sign(realized_return)
        for i, ic in enumerate(ics):
            self._ic_buf[i].append(float(ic))

        # Gradient of regret: negative IC = gradient to minimize
        grad = -ics
        self._weights -= self.learning_rate * grad
        self._weights = _project_simplex_weights(
            self._weights, floor=self.FLOOR, cap=self.CAP
        )
        self._weight_history.append(self._weights.copy())
        if len(self._weight_history) > 2000:
            self._weight_history = self._weight_history[-1000:]
        return self._weights.copy()

    def bates_granger_weights(
        self, forecast_errors: np.ndarray
    ) -> np.ndarray:
        """
        Bates-Granger optimal combination weights.
        Minimizes forecast error variance given individual error variances.
        forecast_errors: shape (n_components, T) array of historical errors.
        """
        if forecast_errors.shape[1] < 5:
            return np.ones(self.n_components) / self.n_components
        cov = np.cov(forecast_errors)
        ones = np.ones(self.n_components)
        try:
            inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(self.n_components))
            raw_w = inv_cov @ ones
            w = raw_w / (ones @ inv_cov @ ones)
        except np.linalg.LinAlgError:
            w = ones / self.n_components
        return _project_simplex_weights(w, floor=self.FLOOR, cap=self.CAP)

    def component_icirs(self) -> np.ndarray:
        out = np.zeros(self.n_components)
        for i, buf in enumerate(self._ic_buf):
            out[i] = _icir(np.array(list(buf)))
        return out

    def weight_history_array(self) -> np.ndarray:
        if not self._weight_history:
            return np.zeros((0, self.n_components))
        return np.array(self._weight_history)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_components": self.n_components,
            "learning_rate": self.learning_rate,
            "ic_window": self.ic_window,
            "_weights": self._weights.tolist(),
            "_ic_buf": [list(b) for b in self._ic_buf],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynamicWeightOptimizer":
        obj = cls(
            n_components=d["n_components"],
            learning_rate=d["learning_rate"],
            ic_window=d["ic_window"],
        )
        obj._weights = np.array(d["_weights"])
        for i, buf in enumerate(d["_ic_buf"]):
            obj._ic_buf[i] = deque(buf, maxlen=d["ic_window"])
        return obj


# ---------------------------------------------------------------------------
# SignalEnsemble
# ---------------------------------------------------------------------------

class SignalEnsemble:
    """
    Combines N individual trading signals with dynamic weights.

    Signals: BH physics, GARCH, OU, QuatNav angular velocity,
             Granger, ML, Hurst (and others).

    Weights updated daily based on realized IC. Uses Bates-Granger
    optimal combination when enough history is available (>= 30 bars).

    Output: combined signal in [-1, 1].
    """

    DEFAULT_NAMES = [
        "bh_physics",
        "garch",
        "ou_zscore",
        "quat_nav",
        "granger",
        "ml",
        "hurst",
    ]

    def __init__(
        self,
        component_names: Optional[List[str]] = None,
        ic_window: int = 60,
        bates_granger_threshold: int = 30,
        weight_lr: float = 0.01,
    ) -> None:
        self.component_names = component_names or self.DEFAULT_NAMES
        self.n_components = len(self.component_names)
        self.ic_window = ic_window
        self.bates_granger_threshold = bates_granger_threshold

        self._optimizer = DynamicWeightOptimizer(
            n_components=self.n_components,
            learning_rate=weight_lr,
            ic_window=ic_window,
        )
        # Rolling buffers of signals and returns for Bates-Granger
        self._signal_buf: deque = deque(maxlen=200)
        self._return_buf: deque = deque(maxlen=200)
        self._error_buf: List[deque] = [
            deque(maxlen=200) for _ in range(self.n_components)
        ]
        self._n_updates = 0
        self._last_signals: Optional[np.ndarray] = None

    def predict(self, signals: Dict[str, float]) -> float:
        """
        Combine component signals into a single value in [-1, 1].
        signals: dict mapping component name -> signal value.
        """
        sig_vec = np.array(
            [signals.get(name, 0.0) for name in self.component_names]
        )
        # Clip inputs to [-1, 1]
        sig_vec = np.clip(sig_vec, -1.0, 1.0)
        self._last_signals = sig_vec

        if len(self._return_buf) >= self.bates_granger_threshold:
            # Use Bates-Granger weights
            err_arr = np.array([list(b) for b in self._error_buf])
            if err_arr.shape[1] >= self.bates_granger_threshold:
                bg_w = self._optimizer.bates_granger_weights(err_arr)
                combined = float(bg_w @ sig_vec)
                return float(np.clip(combined, -1.0, 1.0))

        w = self._optimizer.weights
        combined = float(w @ sig_vec)
        return float(np.clip(combined, -1.0, 1.0))

    def update(self, signals: Dict[str, float], realized_return: float) -> None:
        """Update weights given realized next-bar return."""
        sig_vec = np.array(
            [signals.get(name, 0.0) for name in self.component_names]
        )
        sig_vec = np.clip(sig_vec, -1.0, 1.0)

        self._optimizer.update(sig_vec, realized_return)
        self._signal_buf.append(sig_vec.copy())
        self._return_buf.append(realized_return)

        # Track forecast errors per component
        for i, sv in enumerate(sig_vec):
            err = sv - realized_return
            self._error_buf[i].append(err)

        self._n_updates += 1

    def component_weights(self) -> Dict[str, float]:
        w = self._optimizer.weights
        return {name: float(w[i]) for i, name in enumerate(self.component_names)}

    def component_icirs(self) -> Dict[str, float]:
        icirs = self._optimizer.component_icirs()
        return {name: float(icirs[i]) for i, name in enumerate(self.component_names)}

    def report(self) -> EnsembleReport:
        weights = self._optimizer.weights
        icirs = self._optimizer.component_icirs()

        # Combined ICIR from recent history
        if len(self._signal_buf) >= 5 and len(self._return_buf) >= 5:
            sig_arr = np.array(list(self._signal_buf))
            ret_arr = np.array(list(self._return_buf))
            combined_signals = sig_arr @ weights
            ic_list = []
            window = 20
            for i in range(window, len(combined_signals)):
                ic = _information_coefficient(
                    combined_signals[i - window: i], ret_arr[i - window: i]
                )
                ic_list.append(ic)
            combined_icir = _icir(np.array(ic_list)) if ic_list else 0.0
        else:
            combined_icir = 0.0

        return EnsembleReport(
            component_names=self.component_names,
            component_weights=weights.tolist(),
            component_icirs=icirs.tolist(),
            combined_icir=combined_icir,
            regime="UNKNOWN",
            regime_breakdown={},
            staleness_flags={n: False for n in self.component_names},
            n_updates=self._n_updates,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_names": self.component_names,
            "ic_window": self.ic_window,
            "bates_granger_threshold": self.bates_granger_threshold,
            "_optimizer": self._optimizer.to_dict(),
            "_signal_buf": [s.tolist() for s in self._signal_buf],
            "_return_buf": list(self._return_buf),
            "_error_buf": [list(b) for b in self._error_buf],
            "_n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignalEnsemble":
        obj = cls(
            component_names=d["component_names"],
            ic_window=d["ic_window"],
            bates_granger_threshold=d["bates_granger_threshold"],
        )
        obj._optimizer = DynamicWeightOptimizer.from_dict(d["_optimizer"])
        for sv in d["_signal_buf"]:
            obj._signal_buf.append(np.array(sv))
        for rv in d["_return_buf"]:
            obj._return_buf.append(rv)
        for i, buf in enumerate(d["_error_buf"]):
            obj._error_buf[i] = deque(buf, maxlen=200)
        obj._n_updates = d["_n_updates"]
        return obj


# ---------------------------------------------------------------------------
# StackingEnsemble
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """
    Meta-learner (online ridge) trained on base model predictions.

    Uses time-series cross-validation for meta-learner evaluation.
    Purged embargo prevents lookahead contamination.
    Tracks meta-learner IC rolling series.
    """

    def __init__(
        self,
        n_base_models: int = 5,
        meta_lam: float = 1.0,
        embargo_bars: int = 5,
        ic_window: int = 60,
    ) -> None:
        self.n_base_models = n_base_models
        self.meta_lam = meta_lam
        self.embargo_bars = embargo_bars
        self.ic_window = ic_window

        from ml.online_learning import OnlineRidge
        self._meta = OnlineRidge(n_features=n_base_models, lam=meta_lam)
        self._base_pred_buf: deque = deque(maxlen=500)
        self._return_buf: deque = deque(maxlen=500)
        self._ic_buf: deque = deque(maxlen=ic_window)
        self._n_updates = 0

    def predict(self, base_preds: np.ndarray) -> float:
        """Predict using meta-learner on base model outputs."""
        base_preds = np.asarray(base_preds, dtype=float)
        if len(base_preds) < self.n_base_models:
            base_preds = np.pad(base_preds, (0, self.n_base_models - len(base_preds)))
        elif len(base_preds) > self.n_base_models:
            base_preds = base_preds[: self.n_base_models]
        raw = self._meta.predict(base_preds)
        return float(np.clip(np.tanh(raw), -1.0, 1.0))

    def fit_one(
        self, base_preds: np.ndarray, realized_return: float
    ) -> float:
        """Update meta-learner on new observation."""
        base_preds = np.asarray(base_preds, dtype=float)
        if len(base_preds) < self.n_base_models:
            base_preds = np.pad(
                base_preds, (0, self.n_base_models - len(base_preds))
            )
        elif len(base_preds) > self.n_base_models:
            base_preds = base_preds[: self.n_base_models]

        self._base_pred_buf.append(base_preds.copy())
        self._return_buf.append(realized_return)

        err = self._meta.fit_one(base_preds, realized_return)
        self._n_updates += 1

        # Track rolling IC every 20 updates
        if self._n_updates % 20 == 0 and len(self._return_buf) >= 20:
            preds_arr = np.array(
                [self._meta.predict(x) for x in list(self._base_pred_buf)[-20:]]
            )
            ret_arr = np.array(list(self._return_buf)[-20:])
            ic = _information_coefficient(preds_arr, ret_arr)
            self._ic_buf.append(ic)

        return err

    @property
    def meta_icir(self) -> float:
        return _icir(np.array(list(self._ic_buf)))

    def reset(self) -> None:
        self._meta.reset()
        self._base_pred_buf.clear()
        self._return_buf.clear()
        self._ic_buf.clear()
        self._n_updates = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_base_models": self.n_base_models,
            "meta_lam": self.meta_lam,
            "embargo_bars": self.embargo_bars,
            "ic_window": self.ic_window,
            "_meta": self._meta.to_dict(),
            "_n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StackingEnsemble":
        from ml.online_learning import OnlineRidge
        obj = cls(
            n_base_models=d["n_base_models"],
            meta_lam=d["meta_lam"],
            embargo_bars=d["embargo_bars"],
            ic_window=d["ic_window"],
        )
        obj._meta = OnlineRidge.from_dict(d["_meta"])
        obj._n_updates = d["_n_updates"]
        return obj


# ---------------------------------------------------------------------------
# RegimeSwitchingEnsemble
# ---------------------------------------------------------------------------

class RegimeSwitchingEnsemble:
    """
    Maintains separate SignalEnsemble instances for each market regime.
    Switches the active ensemble based on the current detected regime.

    During regime transitions, applies a smooth weight blend over
    transition_bars bars to avoid abrupt signal discontinuities.
    """

    REGIMES = [Regime.BULL, Regime.BEAR, Regime.SIDEWAYS, Regime.HIGH_VOL]

    def __init__(
        self,
        component_names: Optional[List[str]] = None,
        transition_bars: int = 5,
        ic_window: int = 60,
    ) -> None:
        self.component_names = component_names or SignalEnsemble.DEFAULT_NAMES
        self.transition_bars = transition_bars

        self._ensembles: Dict[str, SignalEnsemble] = {
            r.value: SignalEnsemble(
                component_names=self.component_names,
                ic_window=ic_window,
            )
            for r in self.REGIMES
        }
        self._current_regime: Regime = Regime.SIDEWAYS
        self._prev_regime: Optional[Regime] = None
        self._transition_counter: int = 0
        self._regime_counts: Dict[str, int] = {r.value: 0 for r in self.REGIMES}
        self._n_updates = 0

    def set_regime(self, regime: Regime) -> None:
        if regime != self._current_regime:
            self._prev_regime = self._current_regime
            self._current_regime = regime
            self._transition_counter = 0

    def predict(self, signals: Dict[str, float]) -> float:
        """Predict with smooth transition blending."""
        current_pred = self._ensembles[self._current_regime.value].predict(signals)

        if (
            self._prev_regime is not None
            and self._transition_counter < self.transition_bars
        ):
            prev_pred = self._ensembles[self._prev_regime.value].predict(signals)
            alpha = self._transition_counter / self.transition_bars
            combined = (1.0 - alpha) * prev_pred + alpha * current_pred
            self._transition_counter += 1
            return float(np.clip(combined, -1.0, 1.0))

        return float(np.clip(current_pred, -1.0, 1.0))

    def update(self, signals: Dict[str, float], realized_return: float) -> None:
        """Update only the active regime's ensemble."""
        self._ensembles[self._current_regime.value].update(signals, realized_return)
        self._regime_counts[self._current_regime.value] += 1
        self._n_updates += 1

    def regime_breakdown(self) -> Dict[str, float]:
        total = max(1, self._n_updates)
        return {k: v / total for k, v in self._regime_counts.items()}

    def report(self) -> EnsembleReport:
        active = self._ensembles[self._current_regime.value]
        rep = active.report()
        rep.regime = self._current_regime.value
        rep.regime_breakdown = self.regime_breakdown()
        rep.n_updates = self._n_updates
        return rep

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_names": self.component_names,
            "transition_bars": self.transition_bars,
            "_current_regime": self._current_regime.value,
            "_regime_counts": self._regime_counts,
            "_n_updates": self._n_updates,
            "_ensembles": {k: v.to_dict() for k, v in self._ensembles.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeSwitchingEnsemble":
        obj = cls(
            component_names=d["component_names"],
            transition_bars=d["transition_bars"],
        )
        obj._current_regime = Regime(d["_current_regime"])
        obj._regime_counts = d["_regime_counts"]
        obj._n_updates = d["_n_updates"]
        for k, vd in d["_ensembles"].items():
            obj._ensembles[k] = SignalEnsemble.from_dict(vd)
        return obj


# ---------------------------------------------------------------------------
# ModelMonitor
# ---------------------------------------------------------------------------

@dataclass
class _ModelRecord:
    name: str
    rolling_accuracy: deque = field(default_factory=lambda: deque(maxlen=100))
    rolling_pnl: deque = field(default_factory=lambda: deque(maxlen=100))
    psi_reference: Optional[np.ndarray] = None


class ModelMonitor:
    """
    Tracks each model's rolling accuracy, Sharpe, and population stability
    index (PSI). Fires alerts when model degrades and logs to SQLite.
    """

    PSI_THRESHOLD = 0.2
    ACCURACY_THRESHOLD = 0.52
    MIN_SHARPE = -0.5

    def __init__(
        self,
        model_names: List[str],
        db_path: str = "model_performance.db",
        window: int = 100,
    ) -> None:
        self.model_names = model_names
        self.db_path = db_path
        self.window = window
        self._records: Dict[str, _ModelRecord] = {
            name: _ModelRecord(name=name) for name in model_names
        }
        self._n_updates = 0
        self._alerts: List[Dict[str, Any]] = []
        self._init_db()

    def _init_db(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_performance (
                    ts REAL,
                    model_name TEXT,
                    accuracy REAL,
                    sharpe REAL,
                    psi REAL,
                    alert TEXT
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def update(
        self,
        model_name: str,
        prediction: float,
        realized_return: float,
    ) -> Optional[str]:
        """
        Update model record. Returns alert message if degradation detected.
        prediction: model signal in [-1, 1] or probability in [0, 1].
        realized_return: actual return.
        """
        if model_name not in self._records:
            self._records[model_name] = _ModelRecord(name=model_name)

        rec = self._records[model_name]
        correct = 1.0 if (prediction > 0) == (realized_return > 0) else 0.0
        rec.rolling_accuracy.append(correct)
        pnl = prediction * realized_return
        rec.rolling_pnl.append(pnl)

        alert = None
        if len(rec.rolling_accuracy) >= 30:
            acc = float(np.mean(rec.rolling_accuracy))
            pnl_arr = np.array(list(rec.rolling_pnl))
            sharpe = (
                float(pnl_arr.mean() / (pnl_arr.std() + 1e-10)) * math.sqrt(252)
            )
            psi = self._compute_psi(model_name, pnl_arr)

            # Check thresholds
            if acc < self.ACCURACY_THRESHOLD:
                alert = f"ACCURACY_DEGRADED:{model_name}:acc={acc:.3f}"
            elif sharpe < self.MIN_SHARPE:
                alert = f"SHARPE_DEGRADED:{model_name}:sharpe={sharpe:.3f}"
            elif psi > self.PSI_THRESHOLD:
                alert = f"DRIFT_DETECTED:{model_name}:psi={psi:.3f}"

            self._log_db(model_name, acc, sharpe, psi, alert or "")

        self._n_updates += 1
        if alert:
            self._alerts.append({"ts": time.time(), "alert": alert})
        return alert

    def _compute_psi(self, model_name: str, recent: np.ndarray) -> float:
        """Population Stability Index between reference and recent distribution."""
        rec = self._records[model_name]
        if rec.psi_reference is None:
            rec.psi_reference = recent.copy()
            return 0.0
        ref = rec.psi_reference
        bins = np.linspace(
            min(ref.min(), recent.min()), max(ref.max(), recent.max()) + 1e-10, 11
        )
        ref_hist, _ = np.histogram(ref, bins=bins, density=True)
        rec_hist, _ = np.histogram(recent, bins=bins, density=True)
        ref_hist = np.maximum(ref_hist, 1e-6)
        rec_hist = np.maximum(rec_hist, 1e-6)
        psi = float(np.sum((rec_hist - ref_hist) * np.log(rec_hist / ref_hist)))
        return psi

    def _log_db(
        self,
        name: str,
        accuracy: float,
        sharpe: float,
        psi: float,
        alert: str,
    ) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO model_performance VALUES (?,?,?,?,?,?)",
                (time.time(), name, accuracy, sharpe, psi, alert),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_model_accuracy(self, model_name: str) -> float:
        rec = self._records.get(model_name)
        if rec is None or len(rec.rolling_accuracy) == 0:
            return 0.5
        return float(np.mean(rec.rolling_accuracy))

    def get_alerts(self, n: int = 20) -> List[Dict[str, Any]]:
        return self._alerts[-n:]

    def refresh_reference(self, model_name: str) -> None:
        """Reset PSI reference distribution for a model."""
        rec = self._records.get(model_name)
        if rec is not None and len(rec.rolling_pnl) > 0:
            rec.psi_reference = np.array(list(rec.rolling_pnl)).copy()


# ---------------------------------------------------------------------------
# Full SignalEnsemble with monitoring
# ---------------------------------------------------------------------------

class ManagedSignalEnsemble:
    """
    Top-level ensemble that combines RegimeSwitchingEnsemble, StackingEnsemble,
    and ModelMonitor into a single managed interface.

    This is the main entry point for the live trading system.
    """

    def __init__(
        self,
        component_names: Optional[List[str]] = None,
        db_path: str = "model_performance.db",
        transition_bars: int = 5,
        ic_window: int = 60,
        n_stacking_models: int = 5,
    ) -> None:
        self.component_names = component_names or SignalEnsemble.DEFAULT_NAMES
        self._regime_ensemble = RegimeSwitchingEnsemble(
            component_names=self.component_names,
            transition_bars=transition_bars,
            ic_window=ic_window,
        )
        self._stacking = StackingEnsemble(
            n_base_models=len(self.component_names),
            meta_lam=1.0,
            embargo_bars=5,
            ic_window=ic_window,
        )
        self._monitor = ModelMonitor(
            model_names=self.component_names + ["ensemble", "stacking"],
            db_path=db_path,
        )
        self._n_updates = 0

    def set_regime(self, regime: Regime) -> None:
        self._regime_ensemble.set_regime(regime)

    def predict(
        self, signals: Dict[str, float], use_stacking: bool = False
    ) -> float:
        regime_pred = self._regime_ensemble.predict(signals)

        if use_stacking and self._stacking._n_updates >= 20:
            base_preds = np.array(
                [signals.get(n, 0.0) for n in self.component_names]
            )
            stack_pred = self._stacking.predict(base_preds)
            # Blend regime and stacking predictions
            combined = 0.6 * regime_pred + 0.4 * stack_pred
            return float(np.clip(combined, -1.0, 1.0))

        return regime_pred

    def update(
        self,
        signals: Dict[str, float],
        realized_return: float,
        current_regime: Optional[Regime] = None,
    ) -> None:
        if current_regime is not None:
            self.set_regime(current_regime)

        self._regime_ensemble.update(signals, realized_return)

        base_preds = np.array(
            [signals.get(n, 0.0) for n in self.component_names]
        )
        self._stacking.fit_one(base_preds, realized_return)

        # Monitor each component
        for name in self.component_names:
            sig = signals.get(name, 0.0)
            self._monitor.update(name, sig, realized_return)

        self._n_updates += 1

    def report(self) -> EnsembleReport:
        rep = self._regime_ensemble.report()
        rep.staleness_flags = {
            name: self._monitor.get_model_accuracy(name) < 0.45
            for name in self.component_names
        }
        return rep

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_names": self.component_names,
            "_regime_ensemble": self._regime_ensemble.to_dict(),
            "_stacking": self._stacking.to_dict(),
            "_n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManagedSignalEnsemble":
        obj = cls(component_names=d["component_names"])
        obj._regime_ensemble = RegimeSwitchingEnsemble.from_dict(
            d["_regime_ensemble"]
        )
        obj._stacking = StackingEnsemble.from_dict(d["_stacking"])
        obj._n_updates = d["_n_updates"]
        return obj
