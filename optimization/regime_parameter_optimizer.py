"""
optimization/regime_parameter_optimizer.py
============================================
Per-regime parameter optimization for the SRFM trading system.

Maintains separate parameter sets for each detected market regime:
  Bull / Bear / Sideways / HighVol

Uses Optuna to optimize parameters within each regime's historical bars,
stores results in SQLite with full version history, and provides a
LiveRegimeSwitcher that loads the appropriate parameters at runtime.

Classes:
  RegimeType             -- enum for the four supported regimes
  RegimeDataSplitter     -- groups bars by regime label
  PerRegimeObjective     -- Optuna objective restricted to one regime
  RegimeParameterOptimizer -- top-level orchestrator
  RegimeParamStore       -- SQLite persistence with version history
  LiveRegimeSwitcher     -- runtime parameter loader keyed by current regime
  ConsistencyChecker     -- validates parameters don't diverge too much across regimes

Requires: numpy, pandas, optuna (optional but recommended), sqlite3
"""

from __future__ import annotations

import copy
import json
import logging
import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Optuna import
# ---------------------------------------------------------------------------

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    _OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None  # type: ignore[assignment]
    _OPTUNA = False
    logger.debug("Optuna not available -- using random search fallback")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path("regime_params.db")
_CONSISTENCY_THRESHOLD = 0.50   -- parameters must not differ by more than 50%
_MIN_BARS_FOR_OPTIM = 200       -- skip regime if fewer bars available
_DEFAULT_N_TRIALS = 50


# ---------------------------------------------------------------------------
# Regime type
# ---------------------------------------------------------------------------

class RegimeType(str, Enum):
    BULL = "Bull"
    BEAR = "Bear"
    SIDEWAYS = "Sideways"
    HIGH_VOL = "HighVol"

    @classmethod
    def from_label(cls, label: Any) -> "RegimeType":
        """Parse a regime label string (case-insensitive) to RegimeType."""
        s = str(label).strip()
        mapping = {
            "bull": cls.BULL,
            "trending_up": cls.BULL,
            "bear": cls.BEAR,
            "trending_down": cls.BEAR,
            "sideways": cls.SIDEWAYS,
            "ranging": cls.SIDEWAYS,
            "flat": cls.SIDEWAYS,
            "highvol": cls.HIGH_VOL,
            "high_vol": cls.HIGH_VOL,
            "volatile": cls.HIGH_VOL,
            "crisis": cls.HIGH_VOL,
        }
        return mapping.get(s.lower(), cls.SIDEWAYS)

    @classmethod
    def all_regimes(cls) -> List["RegimeType"]:
        return [cls.BULL, cls.BEAR, cls.SIDEWAYS, cls.HIGH_VOL]


# ---------------------------------------------------------------------------
# RegimeDataSplitter
# ---------------------------------------------------------------------------

class RegimeDataSplitter:
    """
    Splits a bar DataFrame into per-regime subsets.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV bar data. Must have a DatetimeIndex or a 'timestamp' column.
    regime_labels : pd.Series or list
        Regime label for each bar (same length as bars).
        Labels can be strings like "Bull", "Bear", etc.
    """

    def __init__(self, bars: pd.DataFrame, regime_labels: Any):
        self.bars = bars.copy()
        if isinstance(regime_labels, pd.Series):
            labels = regime_labels.values
        else:
            labels = np.asarray(regime_labels)
        if len(labels) != len(bars):
            raise ValueError(
                f"regime_labels length {len(labels)} != bars length {len(bars)}"
            )
        self.regime_labels = labels
        self._build_index()

    def _build_index(self) -> None:
        """Build mapping from RegimeType to integer row indices."""
        self._regime_indices: Dict[RegimeType, np.ndarray] = {}
        for regime in RegimeType.all_regimes():
            mask = np.array([
                RegimeType.from_label(lbl) == regime
                for lbl in self.regime_labels
            ])
            self._regime_indices[regime] = np.where(mask)[0]

    def get_bars(self, regime: RegimeType) -> pd.DataFrame:
        """Return bars belonging to the given regime."""
        idx = self._regime_indices.get(regime, np.array([], dtype=int))
        return self.bars.iloc[idx].copy()

    def regime_counts(self) -> Dict[str, int]:
        """Return count of bars per regime."""
        return {r.value: int(len(self._regime_indices[r])) for r in RegimeType.all_regimes()}

    def regime_fractions(self) -> Dict[str, float]:
        """Return fraction of bars per regime."""
        total = len(self.bars)
        if total == 0:
            return {r.value: 0.0 for r in RegimeType.all_regimes()}
        return {r.value: len(self._regime_indices[r]) / total for r in RegimeType.all_regimes()}

    def get_contiguous_windows(self, regime: RegimeType) -> List[pd.DataFrame]:
        """
        Return list of contiguous bar sub-DataFrames for the given regime.

        Useful for walk-forward analysis where you want regime episodes
        rather than a single shuffled pool.
        """
        idx = self._regime_indices.get(regime, np.array([], dtype=int))
        if len(idx) == 0:
            return []
        windows: List[pd.DataFrame] = []
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i - prev > 1:  -- gap in indices means new contiguous window
                windows.append(self.bars.iloc[start:prev + 1].copy())
                start = i
            prev = i
        windows.append(self.bars.iloc[start:prev + 1].copy())
        return windows


# ---------------------------------------------------------------------------
# PerRegimeObjective
# ---------------------------------------------------------------------------

class PerRegimeObjective:
    """
    Wraps a backtest function to only evaluate on bars from one regime.

    The backtest_fn signature must be:
      backtest_fn(bars: pd.DataFrame, params: dict) -> float

    where the returned float is the metric to maximize (e.g. Sharpe).
    """

    def __init__(
        self,
        backtest_fn: Callable[[pd.DataFrame, Dict[str, Any]], float],
        regime_bars: pd.DataFrame,
        param_space: Dict[str, Tuple[float, float]],
        min_trades: int = 5,
    ):
        self.backtest_fn = backtest_fn
        self.regime_bars = regime_bars
        self.param_space = param_space
        self.min_trades = min_trades

    def __call__(self, trial: Any) -> float:
        """Optuna objective -- suggest params, run backtest, return score."""
        params: Dict[str, Any] = {}
        for name, spec in self.param_space.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, tuple) and len(spec) >= 2:
                lo, hi = float(spec[0]), float(spec[1])
                log_scale = len(spec) == 3 and spec[2] is True
                if isinstance(spec[0], int) and isinstance(spec[1], int) and not log_scale:
                    params[name] = trial.suggest_int(name, int(lo), int(hi))
                else:
                    params[name] = trial.suggest_float(name, lo, hi, log=log_scale)
        try:
            score = self.backtest_fn(self.regime_bars, params)
            return float(score) if score is not None else -10.0
        except Exception as exc:
            logger.debug("Backtest error: %s", exc)
            return -10.0

    def random_search(self, n_trials: int = 50, seed: int = 42) -> Tuple[Dict[str, Any], float]:
        """
        Fallback random search when Optuna is not available.

        Returns (best_params, best_score).
        """
        rng = np.random.default_rng(seed)
        best_score = -_INF
        best_params: Dict[str, Any] = {}

        for _ in range(n_trials):
            params: Dict[str, Any] = {}
            for name, spec in self.param_space.items():
                if isinstance(spec, list):
                    params[name] = spec[rng.integers(0, len(spec))]
                elif isinstance(spec, tuple):
                    lo, hi = float(spec[0]), float(spec[1])
                    params[name] = float(rng.uniform(lo, hi))
            try:
                score = self.backtest_fn(self.regime_bars, params)
                score = float(score) if score is not None else -10.0
            except Exception:
                score = -10.0
            if score > best_score:
                best_score = score
                best_params = copy.deepcopy(params)

        return best_params, best_score


_INF = float("inf")


# ---------------------------------------------------------------------------
# RegimeParameterOptimizer
# ---------------------------------------------------------------------------

class RegimeParameterOptimizer:
    """
    Separate parameter sets for Bull/Bear/Sideways/HighVol regimes.

    Uses historical bars labeled by regime to optimize within each regime.

    Parameters
    ----------
    param_space : dict
        {name: (min, max)} or {name: [choices]}.
    backtest_fn : Callable
        backtest_fn(bars, params) -> float (score to maximize).
    n_trials : int
        Optuna trials per regime.
    store : RegimeParamStore, optional
        If provided, persist results automatically.
    """

    def __init__(
        self,
        param_space: Dict[str, Any],
        backtest_fn: Callable[[pd.DataFrame, Dict[str, Any]], float],
        n_trials: int = _DEFAULT_N_TRIALS,
        store: Optional["RegimeParamStore"] = None,
        seed: int = 42,
    ):
        self.param_space = param_space
        self.backtest_fn = backtest_fn
        self.n_trials = n_trials
        self.store = store
        self.seed = seed
        self._results: Dict[RegimeType, Dict[str, Any]] = {}
        self._scores: Dict[RegimeType, float] = {}

    def optimize_regime_params(
        self,
        regime: RegimeType,
        bars: pd.DataFrame,
        param_space: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run Optuna (or random search) within regime bars.

        Returns the best parameter dict for this regime.
        """
        ps = param_space or self.param_space
        if len(bars) < _MIN_BARS_FOR_OPTIM:
            logger.warning(
                "Regime %s has only %d bars (< %d) -- skipping optimization",
                regime.value, len(bars), _MIN_BARS_FOR_OPTIM,
            )
            return {}

        objective = PerRegimeObjective(
            backtest_fn=self.backtest_fn,
            regime_bars=bars,
            param_space=ps,
        )

        if _OPTUNA:
            sampler = TPESampler(seed=self.seed)
            pruner = MedianPruner(n_startup_trials=5)
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                study_name=f"regime_{regime.value}_{int(time.time())}",
            )
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
        else:
            best_params, best_score = objective.random_search(
                n_trials=self.n_trials, seed=self.seed
            )

        logger.info(
            "Regime %s | best score=%.4f | params=%s",
            regime.value, best_score, best_params,
        )
        self._results[regime] = best_params
        self._scores[regime] = best_score

        if self.store is not None:
            self.store.save(regime, best_params, score=best_score)

        return best_params

    def optimize_all_regimes(
        self,
        bars: pd.DataFrame,
        regime_labels: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize parameters for all regimes.

        Parameters
        ----------
        bars : pd.DataFrame
            Full historical bar data.
        regime_labels : array-like
            Regime label for each bar.

        Returns
        -------
        dict
            {regime_name: best_params} for each regime.
        """
        splitter = RegimeDataSplitter(bars, regime_labels)
        counts = splitter.regime_counts()
        logger.info("Regime bar counts: %s", counts)

        for regime in RegimeType.all_regimes():
            regime_bars = splitter.get_bars(regime)
            self.optimize_regime_params(regime, regime_bars)

        return {r.value: p for r, p in self._results.items()}

    def get_params(self, regime: RegimeType) -> Optional[Dict[str, Any]]:
        """Return best params for a regime (from memory or store)."""
        if regime in self._results:
            return self._results[regime]
        if self.store is not None:
            return self.store.load_latest(regime)
        return None

    def scores(self) -> Dict[str, float]:
        """Return optimization scores per regime."""
        return {r.value: s for r, s in self._scores.items()}


# ---------------------------------------------------------------------------
# RegimeParamStore -- SQLite persistence
# ---------------------------------------------------------------------------

class RegimeParamStore:
    """
    SQLite storage for regime parameter sets with full version history.

    Schema:
      regime_params(id, regime, params_json, score, version, created_at)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_params (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime      TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    score       REAL,
                    version     INTEGER NOT NULL DEFAULT 1,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_regime_version
                ON regime_params (regime, version DESC)
            """)

    def save(
        self,
        regime: RegimeType,
        params: Dict[str, Any],
        score: Optional[float] = None,
    ) -> int:
        """Save a new parameter version. Returns the new row id."""
        version = self._next_version(regime)
        created_at = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO regime_params (regime, params_json, score, version, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (regime.value, json.dumps(params), score, version, created_at),
            )
            return cursor.lastrowid

    def _next_version(self, regime: RegimeType) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(version) as v FROM regime_params WHERE regime = ?",
                (regime.value,),
            ).fetchone()
            return (row["v"] or 0) + 1

    def load_latest(self, regime: RegimeType) -> Optional[Dict[str, Any]]:
        """Load the most recent parameter set for a regime."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT params_json FROM regime_params
                WHERE regime = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (regime.value,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["params_json"])

    def load_version(self, regime: RegimeType, version: int) -> Optional[Dict[str, Any]]:
        """Load a specific version of parameters for a regime."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT params_json FROM regime_params
                WHERE regime = ? AND version = ?
                """,
                (regime.value, version),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["params_json"])

    def load_all_versions(self, regime: RegimeType) -> pd.DataFrame:
        """Return all versions for a regime as a DataFrame."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT version, score, created_at, params_json
                FROM regime_params
                WHERE regime = ?
                ORDER BY version ASC
                """,
                (regime.value,),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        records = []
        for row in rows:
            params = json.loads(row["params_json"])
            records.append({
                "version": row["version"],
                "score": row["score"],
                "created_at": row["created_at"],
                **{f"param_{k}": v for k, v in params.items()},
            })
        return pd.DataFrame(records)

    def load_all_latest(self) -> Dict[str, Dict[str, Any]]:
        """Load the latest parameter set for all regimes."""
        result: Dict[str, Dict[str, Any]] = {}
        for regime in RegimeType.all_regimes():
            params = self.load_latest(regime)
            if params is not None:
                result[regime.value] = params
        return result

    def delete_old_versions(self, regime: RegimeType, keep: int = 10) -> int:
        """Delete all but the most recent `keep` versions. Returns deleted count."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id FROM regime_params
                WHERE regime = ?
                ORDER BY version DESC
                """,
                (regime.value,),
            ).fetchall()
        if len(rows) <= keep:
            return 0
        ids_to_delete = [r["id"] for r in rows[keep:]]
        with self._conn() as conn:
            conn.executemany(
                "DELETE FROM regime_params WHERE id = ?",
                [(i,) for i in ids_to_delete],
            )
        return len(ids_to_delete)


# ---------------------------------------------------------------------------
# LiveRegimeSwitcher
# ---------------------------------------------------------------------------

class LiveRegimeSwitcher:
    """
    Given the current regime, loads and returns the appropriate parameters.

    Supports both in-memory cache (from an optimizer run) and SQLite-backed
    persistence via RegimeParamStore.

    Falls back to `default_params` if no regime-specific params are available.
    """

    def __init__(
        self,
        store: Optional[RegimeParamStore] = None,
        default_params: Optional[Dict[str, Any]] = None,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.store = store
        self.default_params = default_params or {}
        self._cache: Dict[str, Dict[str, Any]] = cache or {}

        # -- pre-warm cache from store
        if self.store is not None and not self._cache:
            self._cache = self.store.load_all_latest()

    def get_params(self, regime: RegimeType) -> Dict[str, Any]:
        """
        Return the parameter set for the given regime.

        Lookup order:
          1. In-memory cache
          2. RegimeParamStore (if configured)
          3. default_params
        """
        params = self._cache.get(regime.value)
        if params:
            return copy.deepcopy(params)

        if self.store is not None:
            params = self.store.load_latest(regime)
            if params:
                self._cache[regime.value] = params
                return copy.deepcopy(params)

        logger.debug(
            "No params found for regime %s -- using defaults", regime.value
        )
        return copy.deepcopy(self.default_params)

    def get_params_from_label(self, label: Any) -> Dict[str, Any]:
        """Convenience wrapper: parse label string, then get params."""
        regime = RegimeType.from_label(label)
        return self.get_params(regime)

    def update_cache(self, regime: RegimeType, params: Dict[str, Any]) -> None:
        """Update in-memory cache (does NOT write to store)."""
        self._cache[regime.value] = copy.deepcopy(params)

    def refresh_from_store(self) -> None:
        """Re-load all latest params from the store into the cache."""
        if self.store is not None:
            self._cache = self.store.load_all_latest()


# ---------------------------------------------------------------------------
# ConsistencyChecker
# ---------------------------------------------------------------------------

class ConsistencyChecker:
    """
    Validates that regime parameter sets don't diverge too much.

    For each shared numeric parameter, checks that the ratio of the max value
    to the min value across regimes does not exceed (1 + threshold).

    Default threshold: 50% -- parameters must not differ by more than 50%.
    """

    def __init__(self, threshold: float = _CONSISTENCY_THRESHOLD):
        self.threshold = threshold

    def check(
        self,
        regime_params: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check consistency across regime parameter sets.

        Parameters
        ----------
        regime_params : dict
            {regime_name: {param: value}} for each regime.

        Returns
        -------
        dict with keys:
          passed : bool
          violations : list of {param, regime_a, regime_b, val_a, val_b, ratio}
          warnings : list of string messages
        """
        if len(regime_params) < 2:
            return {"passed": True, "violations": [], "warnings": []}

        # -- collect all param names that appear in at least two regimes
        all_params: Dict[str, List[Tuple[str, float]]] = {}
        for regime_name, params in regime_params.items():
            for k, v in params.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    all_params.setdefault(k, []).append((regime_name, float(v)))

        violations = []
        warnings = []

        for param, regime_vals in all_params.items():
            if len(regime_vals) < 2:
                continue
            values = [v for _, v in regime_vals]
            max_val = max(abs(v) for v in values)
            min_val = min(abs(v) for v in values)

            if max_val < 1e-10:
                continue

            ratio = max_val / max(min_val, 1e-10)
            if ratio > (1.0 + self.threshold):
                # -- find the two extreme regimes
                max_regime = max(regime_vals, key=lambda x: abs(x[1]))
                min_regime = min(regime_vals, key=lambda x: abs(x[1]))
                violation = {
                    "param": param,
                    "regime_max": max_regime[0],
                    "regime_min": min_regime[0],
                    "val_max": max_regime[1],
                    "val_min": min_regime[1],
                    "ratio": ratio,
                    "threshold": 1.0 + self.threshold,
                }
                violations.append(violation)
                warnings.append(
                    f"Param '{param}': {max_regime[0]}={max_regime[1]:.4f} vs "
                    f"{min_regime[0]}={min_regime[1]:.4f} (ratio={ratio:.2f}, "
                    f"limit={1.0 + self.threshold:.2f})"
                )

        passed = len(violations) == 0
        if not passed:
            logger.warning("ConsistencyChecker: %d violations:\n%s",
                           len(violations), "\n".join(warnings))
        return {
            "passed": passed,
            "violations": violations,
            "warnings": warnings,
        }

    def enforce(
        self,
        regime_params: Dict[str, Dict[str, Any]],
        reference_regime: str = RegimeType.BULL.value,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Clip parameter values so no regime deviates more than `threshold` from reference.

        Returns a corrected copy of regime_params.
        """
        result = copy.deepcopy(regime_params)
        ref = result.get(reference_regime, {})
        if not ref:
            return result

        for regime_name, params in result.items():
            if regime_name == reference_regime:
                continue
            for k, ref_val in ref.items():
                if k not in params:
                    continue
                if not isinstance(ref_val, (int, float)) or isinstance(ref_val, bool):
                    continue
                v = float(params[k])
                r = float(ref_val)
                lo = r * (1.0 - self.threshold) if r > 0 else r * (1.0 + self.threshold)
                hi = r * (1.0 + self.threshold) if r > 0 else r * (1.0 - self.threshold)
                params[k] = float(np.clip(v, lo, hi))

        return result
