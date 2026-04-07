"""
ml/ml_signal_live.py
Live integration of ML ensemble into the trading system.

Wraps ForgettingEnsemble + FeaturePipeline for live use.
Reads from execution/live_trades.db and nav_state table to build
training labels (next-bar return), trains incrementally, exports
signal values to config/ml_signal_state.json.

No em dashes. Uses numpy, pandas, sqlite3.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.online_learning import ForgettingEnsemble, make_default_ensemble
from ml.feature_engineering import FeaturePipeline, RawFeatures


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_STATE_PATH = str(
    Path(__file__).parent.parent / "config" / "ml_signal_state.json"
)
DEFAULT_DB_PATH = str(
    Path(__file__).parent.parent / "execution" / "live_trades.db"
)
ACCURACY_ALERT_THRESHOLD = 0.52
ACCURACY_WINDOW = 100
MIN_WARMUP_BARS = 30


# ---------------------------------------------------------------------------
# LiveMLSignal
# ---------------------------------------------------------------------------

class LiveMLSignal:
    """
    Wraps ForgettingEnsemble + FeaturePipeline for live per-symbol signal.

    Usage:
        signal = LiveMLSignal()
        # on each new bar:
        features = signal.extract_features(bar_data)
        sig = signal.predict(sym, features)
        # after observing next bar return:
        signal.update(sym, features, next_return)

    Signal in [-1, 1]: positive = bullish, negative = bearish.
    """

    def __init__(
        self,
        n_features: int = 200,
        warmup_bars: int = MIN_WARMUP_BARS,
        accuracy_window: int = ACCURACY_WINDOW,
        accuracy_threshold: float = ACCURACY_ALERT_THRESHOLD,
    ) -> None:
        self.n_features = n_features
        self.warmup_bars = warmup_bars
        self.accuracy_window = accuracy_window
        self.accuracy_threshold = accuracy_threshold

        # Per-symbol state
        self._ensembles: Dict[str, ForgettingEnsemble] = {}
        self._pipelines: Dict[str, FeaturePipeline] = {}
        self._bar_counts: Dict[str, int] = {}
        self._accuracy_bufs: Dict[str, deque] = {}
        self._last_features: Dict[str, Optional[np.ndarray]] = {}
        self._alerts: List[Dict[str, Any]] = []

    def _ensure(self, sym: str) -> None:
        if sym not in self._ensembles:
            self._ensembles[sym] = make_default_ensemble(n_features=self.n_features)
            self._pipelines[sym] = FeaturePipeline()
            self._bar_counts[sym] = 0
            self._accuracy_bufs[sym] = deque(maxlen=self.accuracy_window)
            self._last_features[sym] = None

    def extract_features(self, sym: str, raw: RawFeatures) -> np.ndarray:
        """
        Transform raw bar features through the feature pipeline.
        Returns a 200-dim normalized feature vector.
        """
        self._ensure(sym)
        return self._pipelines[sym].transform_one(raw)

    def predict(self, sym: str, features: np.ndarray) -> float:
        """
        Return signal in [-1, 1].
        Returns 0.0 before warmup is complete.
        """
        self._ensure(sym)
        if self._bar_counts[sym] < self.warmup_bars:
            return 0.0
        self._last_features[sym] = features
        return float(self._ensembles[sym].predict(features))

    def update(
        self,
        sym: str,
        features: np.ndarray,
        realized_return: float,
        y_label: Optional[float] = None,
    ) -> None:
        """
        Update the ensemble for sym with observed return.
        y_label: 1.0 if return > 0, else 0.0. Auto-computed if None.
        """
        self._ensure(sym)
        label = y_label if y_label is not None else (1.0 if realized_return > 0 else 0.0)

        # Track accuracy before updating
        if self._bar_counts[sym] >= self.warmup_bars:
            pred = self._ensembles[sym].predict(features)
            correct = 1.0 if (pred > 0) == (realized_return > 0) else 0.0
            self._accuracy_bufs[sym].append(correct)
            self._check_accuracy_alert(sym)

        self._ensembles[sym].fit_one(features, label)
        self._bar_counts[sym] += 1

    def _check_accuracy_alert(self, sym: str) -> None:
        buf = self._accuracy_bufs[sym]
        if len(buf) >= self.accuracy_window:
            acc = float(np.mean(buf))
            if acc < self.accuracy_threshold:
                alert = {
                    "ts": time.time(),
                    "sym": sym,
                    "accuracy": acc,
                    "message": f"ML accuracy for {sym} dropped to {acc:.3f} (threshold {self.accuracy_threshold})",
                }
                self._alerts.append(alert)

    def get_accuracy(self, sym: str) -> float:
        buf = self._accuracy_bufs.get(sym, deque())
        if not buf:
            return 0.5
        return float(np.mean(buf))

    def get_recent_alerts(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._alerts[-n:]

    def get_model_weights(self, sym: str) -> Optional[np.ndarray]:
        if sym not in self._ensembles:
            return None
        return self._ensembles[sym].get_weights()

    def is_warmed_up(self, sym: str) -> bool:
        return self._bar_counts.get(sym, 0) >= self.warmup_bars

    def reset(self, sym: str) -> None:
        if sym in self._ensembles:
            self._ensembles[sym].reset()
            self._bar_counts[sym] = 0
            self._accuracy_bufs[sym].clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "warmup_bars": self.warmup_bars,
            "accuracy_window": self.accuracy_window,
            "accuracy_threshold": self.accuracy_threshold,
            "_bar_counts": self._bar_counts,
            "_ensembles": {
                sym: ens.to_dict()
                for sym, ens in self._ensembles.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LiveMLSignal":
        obj = cls(
            n_features=d["n_features"],
            warmup_bars=d["warmup_bars"],
            accuracy_window=d["accuracy_window"],
            accuracy_threshold=d["accuracy_threshold"],
        )
        for sym, count in d["_bar_counts"].items():
            obj._ensure(sym)
            obj._bar_counts[sym] = count
        for sym, ens_d in d["_ensembles"].items():
            obj._ensure(sym)
            obj._ensembles[sym] = ForgettingEnsemble.from_dict(ens_d)
        return obj

    def save(self, path: str = DEFAULT_STATE_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_STATE_PATH) -> "LiveMLSignal":
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# MLSignalBridge
# ---------------------------------------------------------------------------

@dataclass
class _TradeRecord:
    sym: str
    ts: float
    close: float
    volume: float
    ret_1: float


class MLSignalBridge:
    """
    Reads trade data from execution/live_trades.db and nav_state table
    to construct training labels (next-bar return).

    Trains LiveMLSignal models incrementally on historical data,
    then exports signal values to config/ml_signal_state.json.

    Typical usage:
        bridge = MLSignalBridge()
        bridge.load_history()
        bridge.train_incremental()
        signals = bridge.export_signals()
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        state_path: str = DEFAULT_STATE_PATH,
        lookback_days: int = 90,
        symbols: Optional[List[str]] = None,
    ) -> None:
        self.db_path = db_path
        self.state_path = state_path
        self.lookback_days = lookback_days
        self.symbols = symbols or []
        self._signal = LiveMLSignal()
        self._records: Dict[str, List[_TradeRecord]] = {}
        self._last_signals: Dict[str, float] = {}

    def load_history(self) -> int:
        """
        Load historical bar data from SQLite.
        Returns number of records loaded.
        """
        if not os.path.exists(self.db_path):
            return 0

        total = 0
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_ts = time.time() - self.lookback_days * 86400

            # Try to load from trades table
            try:
                df = pd.read_sql_query(
                    "SELECT sym, ts, close, volume FROM trades WHERE ts >= ? ORDER BY ts ASC",
                    conn,
                    params=(cutoff_ts,),
                )
                for sym, grp in df.groupby("sym"):
                    if self.symbols and sym not in self.symbols:
                        continue
                    grp = grp.sort_values("ts").reset_index(drop=True)
                    grp["ret_1"] = grp["close"].pct_change().fillna(0.0)
                    records = []
                    for _, row in grp.iterrows():
                        records.append(_TradeRecord(
                            sym=str(sym),
                            ts=float(row["ts"]),
                            close=float(row["close"]),
                            volume=float(row.get("volume", 0.0)),
                            ret_1=float(row["ret_1"]),
                        ))
                    self._records[str(sym)] = records
                    total += len(records)
            except Exception:
                pass

            # Try nav_state table
            try:
                df_nav = pd.read_sql_query(
                    "SELECT * FROM nav_state WHERE ts >= ? ORDER BY ts ASC",
                    conn,
                    params=(cutoff_ts,),
                )
                # Parse nav_state columns if available
                if not df_nav.empty and "sym" in df_nav.columns:
                    for sym, grp in df_nav.groupby("sym"):
                        if self.symbols and sym not in self.symbols:
                            continue
                        grp = grp.sort_values("ts").reset_index(drop=True)
                        if str(sym) not in self._records:
                            self._records[str(sym)] = []
            except Exception:
                pass

            conn.close()
        except Exception:
            pass

        return total

    def _build_raw_features(
        self,
        record: _TradeRecord,
        garch_vol: float = 0.01,
        ou_zscore: float = 0.0,
        hurst: float = 0.5,
    ) -> RawFeatures:
        """Build a RawFeatures object from a trade record."""
        import datetime
        dt = datetime.datetime.fromtimestamp(record.ts)
        return RawFeatures(
            close=record.close,
            open=record.close * (1.0 - record.ret_1 * 0.5),
            high=record.close * (1.0 + abs(record.ret_1) * 0.5),
            low=record.close * (1.0 - abs(record.ret_1) * 0.5),
            volume=record.volume,
            ret_1=record.ret_1,
            garch_vol=garch_vol,
            ou_zscore=ou_zscore,
            hurst=hurst,
            hour=dt.hour,
            day_of_week=dt.weekday(),
            day_of_month=dt.day,
            month=dt.month,
        )

    def train_incremental(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Train models incrementally on loaded history.
        Returns dict of sym -> n_training_samples.
        """
        syms = symbols or list(self._records.keys())
        trained = {}

        for sym in syms:
            records = self._records.get(sym, [])
            if len(records) < 10:
                continue

            n_trained = 0
            garch_vol_est = 0.01
            alpha_garch = 0.9

            for i in range(1, len(records)):
                rec = records[i]
                prev_rec = records[i - 1]

                # Simple GARCH(1,1) volatility estimate
                garch_vol_est = math.sqrt(
                    0.01 * rec.ret_1 ** 2
                    + alpha_garch * garch_vol_est ** 2
                )
                garch_vol_est = max(garch_vol_est, 1e-5)

                raw = self._build_raw_features(rec, garch_vol=garch_vol_est)
                features = self._signal.extract_features(sym, raw)

                # Label: direction of next-bar return
                if i + 1 < len(records):
                    next_ret = records[i + 1].ret_1
                    self._signal.update(sym, features, next_ret)
                    n_trained += 1

            trained[sym] = n_trained

        return trained

    def predict_current(
        self,
        sym: str,
        raw: RawFeatures,
    ) -> float:
        """
        Predict current signal for sym given the latest bar features.
        Returns signal in [-1, 1].
        """
        features = self._signal.extract_features(sym, raw)
        sig = self._signal.predict(sym, features)
        self._last_signals[sym] = sig
        return sig

    def export_signals(self) -> Dict[str, float]:
        """
        Export current signal values to config/ml_signal_state.json.
        Returns the signal dict.
        """
        output = {
            "ts": time.time(),
            "signals": self._last_signals.copy(),
            "model_accuracy": {
                sym: self._signal.get_accuracy(sym)
                for sym in self._last_signals
            },
            "n_bars": {
                sym: self._signal._bar_counts.get(sym, 0)
                for sym in self._last_signals
            },
            "alerts": self._signal.get_recent_alerts(5),
        }

        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(output, f, indent=2)
        except Exception:
            pass

        return self._last_signals.copy()

    def save_model_state(self) -> None:
        """Persist LiveMLSignal to disk."""
        self._signal.save(self.state_path.replace(".json", "_model.json"))

    def load_model_state(self) -> bool:
        """Load persisted LiveMLSignal from disk. Returns True if successful."""
        model_path = self.state_path.replace(".json", "_model.json")
        if not os.path.exists(model_path):
            return False
        try:
            self._signal = LiveMLSignal.load(model_path)
            return True
        except Exception:
            return False

    def run_daily_update(
        self,
        sym: str,
        raw: RawFeatures,
        realized_return: Optional[float] = None,
    ) -> float:
        """
        Convenience method for the live trader to call once per bar.

        1. Extract features.
        2. Get signal prediction.
        3. If realized_return provided, update model.
        4. Export state.

        Returns signal in [-1, 1].
        """
        features = self._signal.extract_features(sym, raw)
        sig = self._signal.predict(sym, features)
        self._last_signals[sym] = sig

        if realized_return is not None:
            self._signal.update(sym, features, realized_return)

        return sig

    @property
    def live_signal(self) -> LiveMLSignal:
        return self._signal

    def monitoring_summary(self) -> Dict[str, Any]:
        """Return monitoring summary for all tracked symbols."""
        syms = list(self._signal._bar_counts.keys())
        return {
            "symbols": syms,
            "bar_counts": {s: self._signal._bar_counts.get(s, 0) for s in syms},
            "accuracies": {s: self._signal.get_accuracy(s) for s in syms},
            "warmed_up": {s: self._signal.is_warmed_up(s) for s in syms},
            "signals": self._last_signals.copy(),
            "alerts": self._signal.get_recent_alerts(10),
            "n_alerts_total": len(self._signal._alerts),
        }
