"""
execution/risk/correlation_monitor.py
======================================
Real-time correlation matrix monitoring for the SRFM Lab portfolio.

Provides:
    CorrelationMatrix   -- incremental EWMA update with Ledoit-Wolf shrinkage,
                           stress-regime detection, and PCA decomposition
    ConcentrationRisk   -- HHI index and effective-N computation
    CorrelationMonitor  -- orchestrates snapshots to SQLite every hour

Stress regime is defined as: average pairwise correlation > 0.6 threshold
(matching CORR_STRESS_THRESHOLD from live_trader_alpaca.py).
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

log = logging.getLogger("execution.risk.correlation_monitor")

_DB_PATH = Path(__file__).parents[2] / "execution" / "live_trades.db"

EWMA_LAMBDA = 0.94
STRESS_CORR_THRESHOLD = 0.60
SNAPSHOT_INTERVAL_SECS = 3600   # 1 hour
PCA_N_COMPONENTS = 3


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PairCorrelation:
    """Rolling correlation between two symbols."""
    sym_a: str
    sym_b: str
    correlation: float
    period_days: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CorrSnapshot:
    """Full correlation matrix state at one point in time."""
    symbols: List[str]
    matrix: np.ndarray              # shape (n, n)
    avg_correlation: float
    is_stress_regime: bool
    pca_explained: np.ndarray       # fraction of variance per top-3 components
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------------------------------

class CorrelationMatrix:
    """
    Maintains an N x N correlation (and covariance) matrix, updated
    incrementally as new daily returns arrive.

    EWMA updating:
        cov_ij(t) = lambda * cov_ij(t-1) + (1 - lambda) * r_i(t) * r_j(t)

    After building the EWMA covariance, Ledoit-Wolf shrinkage is applied
    to improve conditioning for small samples.

    Stress regime detection: average off-diagonal correlation > threshold.
    PCA: top-3 eigenvectors and their variance-explained fractions.
    """

    def __init__(
        self,
        symbols: List[str],
        ewma_lambda: float = EWMA_LAMBDA,
        stress_threshold: float = STRESS_CORR_THRESHOLD,
        history_window: int = 252,
    ) -> None:
        self.symbols = list(symbols)
        self.n = len(symbols)
        self.lam = ewma_lambda
        self.stress_threshold = stress_threshold
        self.history_window = history_window
        self._sym_idx: Dict[str, int] = {s: i for i, s in enumerate(symbols)}
        # EWMA covariance state
        self._cov_ewma = np.eye(self.n) * 1e-6
        # Rolling return history for Ledoit-Wolf
        self._return_history: List[np.ndarray] = []
        self._n_updates: int = 0

    def add_symbol(self, symbol: str) -> None:
        """Extend the matrix to include a new symbol."""
        if symbol in self._sym_idx:
            return
        idx = self.n
        self.symbols.append(symbol)
        self._sym_idx[symbol] = idx
        self.n += 1
        # Expand EWMA covariance matrix
        new_cov = np.zeros((self.n, self.n))
        new_cov[:idx, :idx] = self._cov_ewma
        new_cov[idx, idx] = 1e-6
        self._cov_ewma = new_cov

    def update(self, returns: Dict[str, float]) -> None:
        """
        Ingest one day's returns and update EWMA covariance.

        Parameters
        ----------
        returns : dict symbol -> fractional return
        """
        r = np.zeros(self.n)
        for sym, ret in returns.items():
            if sym in self._sym_idx:
                r[self._sym_idx[sym]] = ret
        outer = np.outer(r, r)
        self._cov_ewma = self.lam * self._cov_ewma + (1 - self.lam) * outer
        self._return_history.append(r.copy())
        if len(self._return_history) > self.history_window:
            self._return_history.pop(0)
        self._n_updates += 1

    def _ledoit_wolf_cov(self) -> np.ndarray:
        """
        Apply Ledoit-Wolf shrinkage to the return history.

        Falls back to EWMA covariance if insufficient data.
        """
        if len(self._return_history) < max(10, self.n):
            return self._cov_ewma.copy()
        X = np.array(self._return_history)  # (T, n)
        try:
            lw = LedoitWolf(assume_centered=False)
            lw.fit(X)
            return lw.covariance_
        except Exception as exc:
            log.debug("LedoitWolf failed: %s", exc)
            return self._cov_ewma.copy()

    def correlation_matrix(self, use_ledoit_wolf: bool = True) -> np.ndarray:
        """
        Return the N x N correlation matrix.

        Parameters
        ----------
        use_ledoit_wolf : if True and enough history exists, apply LW shrinkage
        """
        if use_ledoit_wolf and len(self._return_history) >= max(10, self.n):
            cov = self._ledoit_wolf_cov()
        else:
            cov = self._cov_ewma.copy()

        # Standardise to correlation
        std = np.sqrt(np.diag(cov))
        std = np.where(std < 1e-10, 1.0, std)
        corr = cov / np.outer(std, std)
        # Clip to [-1, 1] for numerical safety
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def average_correlation(self) -> float:
        """Average of all off-diagonal absolute correlations."""
        corr = self.correlation_matrix()
        mask = ~np.eye(self.n, dtype=bool)
        if not mask.any():
            return 0.0
        return float(np.abs(corr[mask]).mean())

    def is_stress_regime(self) -> bool:
        """True if avg pairwise correlation exceeds stress_threshold."""
        return self.average_correlation() > self.stress_threshold

    def pca_explained_variance(self, n_components: int = PCA_N_COMPONENTS) -> np.ndarray:
        """
        Fraction of variance explained by the top n_components eigenvectors.

        Returns array of length n_components (or fewer if n < n_components).
        """
        corr = self.correlation_matrix()
        try:
            eigvals = np.linalg.eigvalsh(corr)
            eigvals = np.sort(eigvals)[::-1]
            eigvals = np.clip(eigvals, 0.0, None)
            total = eigvals.sum()
            if total <= 0:
                return np.zeros(min(n_components, self.n))
            n = min(n_components, len(eigvals))
            return eigvals[:n] / total
        except np.linalg.LinAlgError as exc:
            log.debug("PCA eigvalsh failed: %s", exc)
            return np.zeros(min(n_components, self.n))

    def pairwise_correlations(self) -> List[PairCorrelation]:
        """Return all unique pairwise PairCorrelation objects."""
        corr = self.correlation_matrix()
        ts = datetime.now(timezone.utc)
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append(PairCorrelation(
                    sym_a=self.symbols[i],
                    sym_b=self.symbols[j],
                    correlation=float(corr[i, j]),
                    period_days=min(self._n_updates, self.history_window),
                    timestamp=ts,
                ))
        return pairs

    def snapshot(self) -> CorrSnapshot:
        """Build a CorrSnapshot from current state."""
        corr = self.correlation_matrix()
        avg_corr = self.average_correlation()
        pca_exp = self.pca_explained_variance()
        return CorrSnapshot(
            symbols=list(self.symbols),
            matrix=corr,
            avg_correlation=avg_corr,
            is_stress_regime=avg_corr > self.stress_threshold,
            pca_explained=pca_exp,
        )


# ---------------------------------------------------------------------------
# Concentration Risk
# ---------------------------------------------------------------------------

class ConcentrationRisk:
    """
    Herfindahl-Hirschman Index (HHI) and effective-N for position sizing.

    HHI = sum(w_i^2) where w_i = abs(notional_i) / total_abs_notional
    Effective N = 1 / HHI (number of equally-weighted positions with same HHI)
    """

    def __init__(self) -> None:
        self._weights: Dict[str, float] = {}

    def update(self, position_notionals: Dict[str, float]) -> None:
        """
        Update with current position notionals.

        Parameters
        ----------
        position_notionals : dict symbol -> signed notional in USD
        """
        total = sum(abs(v) for v in position_notionals.values())
        if total <= 0:
            self._weights = {}
            return
        self._weights = {sym: abs(v) / total for sym, v in position_notionals.items()}

    @property
    def hhi(self) -> float:
        """Herfindahl-Hirschman Index in [0, 1]. Higher = more concentrated."""
        if not self._weights:
            return 0.0
        return float(sum(w ** 2 for w in self._weights.values()))

    @property
    def effective_n(self) -> float:
        """
        Effective number of independent positions.
        Range: 1 (fully concentrated) to n (perfectly diversified).
        """
        h = self.hhi
        if h <= 0:
            return 0.0
        return 1.0 / h

    def largest_positions(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return top_n symbols by weight, descending."""
        sorted_w = sorted(self._weights.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_w[:top_n]

    def is_over_concentrated(self, threshold_hhi: float = 0.25) -> bool:
        """
        True if HHI exceeds threshold.
        Threshold 0.25 corresponds to effective N < 4 (moderately concentrated).
        """
        return self.hhi > threshold_hhi

    def report(self) -> Dict:
        return {
            "hhi": round(self.hhi, 6),
            "effective_n": round(self.effective_n, 2),
            "n_positions": len(self._weights),
            "largest": [
                {"symbol": s, "weight": round(w, 4)}
                for s, w in self.largest_positions()
            ],
        }


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_CREATE_CORR_TABLE = """
CREATE TABLE IF NOT EXISTS correlation_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    symbols_json    TEXT    NOT NULL,
    matrix_json     TEXT    NOT NULL,
    avg_correlation REAL,
    is_stress       INTEGER,
    pca_explained_json TEXT,
    hhi             REAL,
    effective_n     REAL
);
"""


# ---------------------------------------------------------------------------
# Correlation Monitor
# ---------------------------------------------------------------------------

class CorrelationMonitor:
    """
    Orchestrates CorrelationMatrix and ConcentrationRisk updates.

    Behaviour:
    - Accepts per-bar return dicts via update()
    - Writes a snapshot to SQLite every SNAPSHOT_INTERVAL_SECS seconds
    - Emits a warning when the stress regime is newly detected
    """

    def __init__(
        self,
        symbols: List[str],
        db_path: Path = _DB_PATH,
        ewma_lambda: float = EWMA_LAMBDA,
        stress_threshold: float = STRESS_CORR_THRESHOLD,
        snapshot_interval_secs: float = SNAPSHOT_INTERVAL_SECS,
    ) -> None:
        self.db_path = db_path
        self.snapshot_interval = snapshot_interval_secs
        self.corr_matrix = CorrelationMatrix(
            symbols=symbols,
            ewma_lambda=ewma_lambda,
            stress_threshold=stress_threshold,
        )
        self.concentration = ConcentrationRisk()
        self._last_snapshot_time: float = 0.0
        self._was_stress: bool = False
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(_CREATE_CORR_TABLE)
                conn.commit()
        except sqlite3.Error as exc:
            log.error("Failed to create correlation_snapshots table: %s", exc)

    def update(
        self,
        returns: Dict[str, float],
        position_notionals: Optional[Dict[str, float]] = None,
    ) -> CorrSnapshot:
        """
        Ingest one period's returns and optionally update concentration.

        Parameters
        ----------
        returns             : dict symbol -> fractional daily return
        position_notionals  : dict symbol -> signed notional in USD (optional)

        Returns
        -------
        CorrSnapshot with current state.
        """
        self.corr_matrix.update(returns)
        if position_notionals:
            self.concentration.update(position_notionals)

        snap = self.corr_matrix.snapshot()

        # Stress regime transition warning
        if snap.is_stress_regime and not self._was_stress:
            log.warning(
                "CorrelationMonitor: STRESS REGIME detected. "
                "Avg correlation=%.3f > threshold=%.3f",
                snap.avg_correlation,
                self.corr_matrix.stress_threshold,
            )
        elif not snap.is_stress_regime and self._was_stress:
            log.info(
                "CorrelationMonitor: stress regime ended. Avg corr=%.3f",
                snap.avg_correlation,
            )
        self._was_stress = snap.is_stress_regime

        # Periodic snapshot to database
        now = time.monotonic()
        if now - self._last_snapshot_time >= self.snapshot_interval:
            self._persist(snap)
            self._last_snapshot_time = now

        return snap

    def _persist(self, snap: CorrSnapshot) -> None:
        import json as _json
        ts = snap.timestamp.isoformat()
        sym_json = _json.dumps(snap.symbols)
        mat_json = _json.dumps(snap.matrix.tolist())
        pca_json = _json.dumps(snap.pca_explained.tolist())
        hhi = self.concentration.hhi
        eff_n = self.concentration.effective_n
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO correlation_snapshots
                       (timestamp, symbols_json, matrix_json, avg_correlation,
                        is_stress, pca_explained_json, hhi, effective_n)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (ts, sym_json, mat_json, snap.avg_correlation,
                     int(snap.is_stress_regime), pca_json, hhi, eff_n),
                )
                conn.commit()
        except sqlite3.Error as exc:
            log.error("Failed to persist correlation snapshot: %s", exc)

    def latest_snapshot(self) -> Optional[CorrSnapshot]:
        """Return the most recent persisted snapshot from the database."""
        import json as _json
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """SELECT timestamp, symbols_json, matrix_json, avg_correlation,
                              is_stress, pca_explained_json
                       FROM correlation_snapshots ORDER BY id DESC LIMIT 1"""
                ).fetchone()
        except Exception as exc:
            log.error("latest_snapshot DB read failed: %s", exc)
            return None
        if row is None:
            return None
        ts = datetime.fromisoformat(row[0])
        symbols = _json.loads(row[1])
        matrix = np.array(_json.loads(row[2]))
        avg_corr = float(row[3])
        is_stress = bool(row[4])
        pca_exp = np.array(_json.loads(row[5]))
        return CorrSnapshot(
            symbols=symbols,
            matrix=matrix,
            avg_correlation=avg_corr,
            is_stress_regime=is_stress,
            pca_explained=pca_exp,
            timestamp=ts,
        )

    def correlation_json(self) -> Dict:
        """Return correlation matrix as a JSON-serialisable dict."""
        snap = self.corr_matrix.snapshot()
        conc = self.concentration.report()
        return {
            "timestamp": snap.timestamp.isoformat(),
            "symbols": snap.symbols,
            "matrix": snap.matrix.tolist(),
            "avg_correlation": round(snap.avg_correlation, 6),
            "is_stress_regime": snap.is_stress_regime,
            "pca_explained": snap.pca_explained.tolist(),
            "concentration": conc,
        }

    def add_symbol(self, symbol: str) -> None:
        """Add a new symbol to the matrix (no history available for it yet)."""
        self.corr_matrix.add_symbol(symbol)
