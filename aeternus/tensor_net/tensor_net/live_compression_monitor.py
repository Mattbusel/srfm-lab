"""
live_compression_monitor.py — Live compression quality monitoring for AETERNUS.

Provides:
  - Reconstruction error tracking in real-time (rolling average)
  - Fidelity alert: if reconstruction error > threshold, increase bond dimension
  - Compression ratio dashboard per module output type
  - Regime-aware monitoring: track quality degradation during market stress
  - Auto-healing: detect rank collapse (singular values near zero), trigger recompression
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Deque, Dict, List, Optional, Tuple, Union
)

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reconstruction error metrics
# ---------------------------------------------------------------------------

class ErrorMetric(Enum):
    """Which norm to use when computing reconstruction error."""
    FROBENIUS    = auto()   # ||A - A_hat||_F / ||A||_F
    RELATIVE_L2  = auto()   # same as Frobenius for matrices
    MAX_ABS      = auto()   # max(|A - A_hat|)
    MEAN_ABS     = auto()   # mean(|A - A_hat|)
    COSINE_SIM   = auto()   # 1 - cosine_similarity(A.ravel(), A_hat.ravel())


def reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    metric: ErrorMetric = ErrorMetric.FROBENIUS,
) -> float:
    """Compute reconstruction error between *original* and *reconstructed*."""
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    if metric == ErrorMetric.FROBENIUS or metric == ErrorMetric.RELATIVE_L2:
        denom = np.linalg.norm(original) + 1e-12
        return float(np.linalg.norm(diff) / denom)
    elif metric == ErrorMetric.MAX_ABS:
        return float(np.max(np.abs(diff)))
    elif metric == ErrorMetric.MEAN_ABS:
        return float(np.mean(np.abs(diff)))
    elif metric == ErrorMetric.COSINE_SIM:
        a = original.ravel().astype(np.float64)
        b = reconstructed.ravel().astype(np.float64)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return float(1.0 - cos)
    raise ValueError(f"Unknown metric: {metric}")


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """CR = original_bytes / compressed_bytes."""
    return original_bytes / (compressed_bytes + 1)


# ---------------------------------------------------------------------------
# Rolling statistics tracker
# ---------------------------------------------------------------------------

class RollingStats:
    """Maintains a fixed-length rolling window of scalar measurements."""

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._buf: Deque[float] = deque(maxlen=window)

    def push(self, value: float) -> None:
        self._buf.append(value)

    @property
    def count(self) -> int:
        return len(self._buf)

    @property
    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return float(np.mean(list(self._buf)))

    @property
    def std(self) -> float:
        if len(self._buf) < 2:
            return 0.0
        return float(np.std(list(self._buf), ddof=1))

    @property
    def max(self) -> float:
        if not self._buf:
            return 0.0
        return float(max(self._buf))

    @property
    def min(self) -> float:
        if not self._buf:
            return 0.0
        return float(min(self._buf))

    @property
    def latest(self) -> Optional[float]:
        if not self._buf:
            return None
        return self._buf[-1]

    def percentile(self, p: float) -> float:
        if not self._buf:
            return 0.0
        return float(np.percentile(list(self._buf), p))

    def reset(self) -> None:
        self._buf.clear()


# ---------------------------------------------------------------------------
# Fidelity alert
# ---------------------------------------------------------------------------

class AlertSeverity(Enum):
    INFO    = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass
class FidelityAlert:
    schema_name: str
    tick_id: int
    error_value: float
    threshold: float
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.perf_counter)
    action_taken: str = ""


class FidelityAlertSystem:
    """
    Raises fidelity alerts when reconstruction error exceeds thresholds,
    and optionally triggers a bond-dimension increase callback.
    """

    def __init__(
        self,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.15,
        on_warning: Optional[Callable[[FidelityAlert], None]] = None,
        on_critical: Optional[Callable[[FidelityAlert], None]] = None,
    ) -> None:
        self._warn_thr = warning_threshold
        self._crit_thr = critical_threshold
        self._on_warning = on_warning
        self._on_critical = on_critical
        self._history: List[FidelityAlert] = []
        self._lock = threading.Lock()

    def check(
        self,
        schema_name: str,
        tick_id: int,
        error: float,
    ) -> Optional[FidelityAlert]:
        """
        Check *error* against thresholds.
        Returns a FidelityAlert if a threshold is exceeded, else None.
        """
        if error >= self._crit_thr:
            sev = AlertSeverity.CRITICAL
            msg = (
                f"CRITICAL: reconstruction error {error:.4f} >= {self._crit_thr:.4f} "
                f"for schema '{schema_name}' at tick {tick_id}."
            )
        elif error >= self._warn_thr:
            sev = AlertSeverity.WARNING
            msg = (
                f"WARNING: reconstruction error {error:.4f} >= {self._warn_thr:.4f} "
                f"for schema '{schema_name}' at tick {tick_id}."
            )
        else:
            return None

        alert = FidelityAlert(
            schema_name=schema_name,
            tick_id=tick_id,
            error_value=error,
            threshold=self._crit_thr if sev == AlertSeverity.CRITICAL else self._warn_thr,
            severity=sev,
            message=msg,
        )
        with self._lock:
            self._history.append(alert)

        if sev == AlertSeverity.CRITICAL:
            logger.error(msg)
            if self._on_critical:
                self._on_critical(alert)
        else:
            logger.warning(msg)
            if self._on_warning:
                self._on_warning(alert)
        return alert

    def recent_alerts(
        self,
        n: int = 20,
        severity: Optional[AlertSeverity] = None,
    ) -> List[FidelityAlert]:
        with self._lock:
            history = list(self._history)
        if severity:
            history = [a for a in history if a.severity == severity]
        return history[-n:]

    def alert_rate(self, last_n: int = 100) -> float:
        with self._lock:
            recent = self._history[-last_n:]
        return len(recent) / last_n if recent else 0.0

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


# ---------------------------------------------------------------------------
# Rank collapse detector
# ---------------------------------------------------------------------------

@dataclass
class RankCollapseEvent:
    schema_name: str
    tick_id: int
    n_collapsed_modes: int
    singular_value_summary: List[float]
    action: str = "recompression_triggered"
    timestamp: float = field(default_factory=time.perf_counter)


def detect_rank_collapse(
    singular_values: np.ndarray,
    *,
    near_zero_tol: float = 1e-6,
    collapse_fraction: float = 0.5,
) -> Tuple[bool, int]:
    """
    Detect rank collapse in a list/array of singular values.

    Parameters
    ----------
    singular_values:
        1D array of singular values (should be sorted descending).
    near_zero_tol:
        Values below this are considered collapsed.
    collapse_fraction:
        If more than this fraction are collapsed, flag as rank collapse.

    Returns
    -------
    (is_collapsed, n_collapsed)
    """
    sv = np.asarray(singular_values, dtype=np.float64)
    n_collapsed = int(np.sum(sv < near_zero_tol))
    fraction = n_collapsed / (len(sv) + 1e-12)
    return fraction > collapse_fraction, n_collapsed


class RankCollapseDetector:
    """
    Monitor TT-core singular values and trigger recompression on collapse.
    """

    def __init__(
        self,
        near_zero_tol: float = 1e-6,
        collapse_fraction: float = 0.5,
        on_collapse: Optional[Callable[[RankCollapseEvent], None]] = None,
    ) -> None:
        self._tol = near_zero_tol
        self._collapse_frac = collapse_fraction
        self._on_collapse = on_collapse
        self._events: List[RankCollapseEvent] = []
        self._lock = threading.Lock()

    def check_mode(
        self,
        schema_name: str,
        tick_id: int,
        singular_values: np.ndarray,
        mode_idx: int = 0,
    ) -> Optional[RankCollapseEvent]:
        """Check singular values for a single TT mode. Returns event if collapsed."""
        collapsed, n_col = detect_rank_collapse(
            singular_values,
            near_zero_tol=self._tol,
            collapse_fraction=self._collapse_frac,
        )
        if not collapsed:
            return None
        sv_summary = sorted(singular_values.tolist(), reverse=True)[:10]
        event = RankCollapseEvent(
            schema_name=schema_name,
            tick_id=tick_id,
            n_collapsed_modes=n_col,
            singular_value_summary=sv_summary,
        )
        with self._lock:
            self._events.append(event)
        logger.warning(
            "Rank collapse detected in '%s' at tick %d: %d/%d singular values near zero.",
            schema_name, tick_id, n_col, len(singular_values),
        )
        if self._on_collapse:
            self._on_collapse(event)
        return event

    def recent_events(self, n: int = 20) -> List[RankCollapseEvent]:
        with self._lock:
            return list(self._events[-n:])

    def n_collapses(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Compression ratio tracker per module
# ---------------------------------------------------------------------------

@dataclass
class CompressionRatioEntry:
    schema_name: str
    tick_id: int
    original_bytes: int
    compressed_bytes: int
    ratio: float
    timestamp: float = field(default_factory=time.perf_counter)


class CompressionRatioTracker:
    """
    Track per-schema compression ratios over time.
    """

    def __init__(self, window: int = 200) -> None:
        self._window = window
        self._history: Dict[str, Deque[CompressionRatioEntry]] = {}
        self._lock = threading.Lock()

    def record(
        self,
        schema_name: str,
        tick_id: int,
        original_bytes: int,
        compressed_bytes: int,
    ) -> float:
        cr = compression_ratio(original_bytes, compressed_bytes)
        entry = CompressionRatioEntry(
            schema_name=schema_name,
            tick_id=tick_id,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            ratio=cr,
        )
        with self._lock:
            if schema_name not in self._history:
                self._history[schema_name] = deque(maxlen=self._window)
            self._history[schema_name].append(entry)
        return cr

    def current_ratio(self, schema_name: str) -> Optional[float]:
        with self._lock:
            hist = self._history.get(schema_name)
            if not hist:
                return None
            return hist[-1].ratio

    def mean_ratio(self, schema_name: str) -> Optional[float]:
        with self._lock:
            hist = self._history.get(schema_name)
            if not hist:
                return None
            return float(np.mean([e.ratio for e in hist]))

    def dashboard(self) -> str:
        lines = ["Compression Ratio Dashboard", "-" * 50]
        with self._lock:
            for schema_name, hist in sorted(self._history.items()):
                if not hist:
                    continue
                ratios = [e.ratio for e in hist]
                lines.append(
                    f"  {schema_name:<35s}  "
                    f"current={hist[-1].ratio:6.1f}x  "
                    f"mean={float(np.mean(ratios)):6.1f}x  "
                    f"min={float(np.min(ratios)):6.1f}x  "
                    f"max={float(np.max(ratios)):6.1f}x  "
                    f"n={len(hist)}"
                )
        return "\n".join(lines)

    def all_schemas(self) -> List[str]:
        with self._lock:
            return list(self._history.keys())


# ---------------------------------------------------------------------------
# Regime-aware monitoring
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    NORMAL  = auto()
    STRESS  = auto()
    CRISIS  = auto()
    UNKNOWN = auto()


def infer_market_regime(
    chronos_data: np.ndarray,
    stress_vol_threshold: float = 0.03,
    crisis_vol_threshold: float = 0.08,
) -> MarketRegime:
    """
    Infer market regime from ChronosOutput data (N, T, 6).

    Uses rolling realized vol (spread column) as a proxy for stress.
    """
    if chronos_data.ndim != 3 or chronos_data.shape[2] < 4:
        return MarketRegime.UNKNOWN
    spread = chronos_data[:, :, 3]  # (N, T) spread feature
    mean_spread = float(np.mean(spread))
    if mean_spread > crisis_vol_threshold:
        return MarketRegime.CRISIS
    if mean_spread > stress_vol_threshold:
        return MarketRegime.STRESS
    return MarketRegime.NORMAL


@dataclass
class RegimeQualityRecord:
    regime: MarketRegime
    schema_name: str
    tick_id: int
    error: float
    compression_ratio: float
    timestamp: float = field(default_factory=time.perf_counter)


class RegimeAwareMonitor:
    """
    Tracks compression quality per market regime.
    Detects if quality degrades specifically during stress / crisis.
    """

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._records: Deque[RegimeQualityRecord] = deque(maxlen=window * 3)
        self._current_regime: MarketRegime = MarketRegime.UNKNOWN
        self._lock = threading.Lock()

    def update_regime(self, regime: MarketRegime) -> None:
        with self._lock:
            if regime != self._current_regime:
                logger.info(
                    "Market regime changed: %s -> %s",
                    self._current_regime.name, regime.name,
                )
            self._current_regime = regime

    def record(
        self,
        schema_name: str,
        tick_id: int,
        error: float,
        cr: float,
    ) -> None:
        rec = RegimeQualityRecord(
            regime=self._current_regime,
            schema_name=schema_name,
            tick_id=tick_id,
            error=error,
            compression_ratio=cr,
        )
        with self._lock:
            self._records.append(rec)

    def quality_by_regime(
        self,
        schema_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Return mean error and CR per regime, optionally filtered by schema.
        """
        with self._lock:
            records = list(self._records)
        if schema_name:
            records = [r for r in records if r.schema_name == schema_name]
        result: Dict[str, Dict[str, float]] = {}
        for regime in MarketRegime:
            regime_recs = [r for r in records if r.regime == regime]
            if not regime_recs:
                continue
            errors = [r.error for r in regime_recs]
            crs    = [r.compression_ratio for r in regime_recs]
            result[regime.name] = {
                "mean_error": float(np.mean(errors)),
                "max_error":  float(np.max(errors)),
                "mean_cr":    float(np.mean(crs)),
                "n":          len(regime_recs),
            }
        return result

    def degradation_detected(
        self,
        schema_name: str,
        stress_error_ratio: float = 1.5,
    ) -> bool:
        """
        Return True if error during STRESS/CRISIS is >= stress_error_ratio * NORMAL error.
        """
        by_regime = self.quality_by_regime(schema_name)
        normal_err = by_regime.get(MarketRegime.NORMAL.name, {}).get("mean_error")
        stress_err = by_regime.get(MarketRegime.STRESS.name, {}).get("mean_error")
        crisis_err = by_regime.get(MarketRegime.CRISIS.name, {}).get("mean_error")
        if normal_err is None or normal_err < 1e-10:
            return False
        for err in [stress_err, crisis_err]:
            if err is not None and err >= stress_error_ratio * normal_err:
                return True
        return False

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime


# ---------------------------------------------------------------------------
# Auto-healing bond-dimension manager
# ---------------------------------------------------------------------------

@dataclass
class BondDimUpdate:
    schema_name: str
    tick_id: int
    old_bond_dim: int
    new_bond_dim: int
    reason: str
    timestamp: float = field(default_factory=time.perf_counter)


class AutoHealingBondManager:
    """
    Monitors reconstruction error and rank collapse; automatically adjusts
    bond dimension to maintain target fidelity.

    Parameters
    ----------
    initial_bond_dim:
        Starting bond dimension.
    min_bond_dim / max_bond_dim:
        Bounds on the adaptive bond dimension.
    error_threshold_up:
        Increase bond dim when error > this.
    error_threshold_down:
        Decrease bond dim when error < this (after sufficient good ticks).
    step_up / step_down:
        How much to increase / decrease bond dim per adjustment.
    cooldown_ticks:
        Minimum ticks between adjustments.
    on_bond_change:
        Callback invoked when bond dim changes.
    """

    def __init__(
        self,
        initial_bond_dim: int = 8,
        min_bond_dim: int = 4,
        max_bond_dim: int = 64,
        error_threshold_up: float = 0.05,
        error_threshold_down: float = 0.01,
        step_up: int = 4,
        step_down: int = 2,
        cooldown_ticks: int = 20,
        on_bond_change: Optional[Callable[[BondDimUpdate], None]] = None,
    ) -> None:
        self._bond_dim = initial_bond_dim
        self._min = min_bond_dim
        self._max = max_bond_dim
        self._up_thr = error_threshold_up
        self._down_thr = error_threshold_down
        self._step_up = step_up
        self._step_down = step_down
        self._cooldown = cooldown_ticks
        self._on_change = on_bond_change
        self._last_change_tick: int = -cooldown_ticks
        self._consecutive_good: int = 0
        self._required_good: int = 10  # ticks of low error before reducing
        self._updates: List[BondDimUpdate] = []
        self._lock = threading.Lock()

        # Per-schema error stats
        self._error_stats: Dict[str, RollingStats] = {}

    @property
    def bond_dim(self) -> int:
        return self._bond_dim

    def update(
        self,
        schema_name: str,
        tick_id: int,
        error: float,
    ) -> Optional[BondDimUpdate]:
        """
        Feed a new reconstruction error; may trigger bond dim change.
        Returns BondDimUpdate if a change was made, else None.
        """
        with self._lock:
            stats = self._error_stats.setdefault(schema_name, RollingStats(window=20))
            stats.push(error)
            rolling_error = stats.mean

            in_cooldown = (tick_id - self._last_change_tick) < self._cooldown
            if in_cooldown:
                return None

            old_dim = self._bond_dim
            reason = ""

            if rolling_error > self._up_thr and self._bond_dim < self._max:
                self._bond_dim = min(self._bond_dim + self._step_up, self._max)
                self._consecutive_good = 0
                reason = f"error {rolling_error:.4f} > threshold {self._up_thr}"
            elif rolling_error < self._down_thr and self._bond_dim > self._min:
                self._consecutive_good += 1
                if self._consecutive_good >= self._required_good:
                    self._bond_dim = max(self._bond_dim - self._step_down, self._min)
                    self._consecutive_good = 0
                    reason = f"error {rolling_error:.4f} < threshold {self._down_thr} sustained"
            else:
                if rolling_error < self._down_thr:
                    self._consecutive_good += 1
                else:
                    self._consecutive_good = 0
                return None

            if self._bond_dim == old_dim:
                return None

            update = BondDimUpdate(
                schema_name=schema_name,
                tick_id=tick_id,
                old_bond_dim=old_dim,
                new_bond_dim=self._bond_dim,
                reason=reason,
            )
            self._updates.append(update)
            self._last_change_tick = tick_id
            logger.info(
                "Bond dim for '%s' changed: %d -> %d at tick %d. Reason: %s",
                schema_name, old_dim, self._bond_dim, tick_id, reason,
            )
            if self._on_change:
                self._on_change(update)
            return update

    def force_bond_dim(self, new_dim: int, schema_name: str = "all", tick_id: int = 0) -> BondDimUpdate:
        """Force-set bond dimension regardless of cooldown."""
        with self._lock:
            old = self._bond_dim
            self._bond_dim = np.clip(new_dim, self._min, self._max)
            update = BondDimUpdate(
                schema_name=schema_name,
                tick_id=tick_id,
                old_bond_dim=old,
                new_bond_dim=self._bond_dim,
                reason="forced",
            )
            self._updates.append(update)
        return update

    def update_history(self) -> List[BondDimUpdate]:
        return list(self._updates)

    def error_stats(self, schema_name: str) -> Optional[RollingStats]:
        return self._error_stats.get(schema_name)


# ---------------------------------------------------------------------------
# Unified live monitor
# ---------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    """Configuration for LiveCompressionMonitor."""
    rolling_window: int = 100
    error_metric: ErrorMetric = ErrorMetric.FROBENIUS
    fidelity_warning_threshold: float = 0.05
    fidelity_critical_threshold: float = 0.15
    rank_collapse_tol: float = 1e-6
    rank_collapse_fraction: float = 0.5
    stress_vol_threshold: float = 0.03
    crisis_vol_threshold: float = 0.08
    auto_heal: bool = True
    initial_bond_dim: int = 8
    min_bond_dim: int = 4
    max_bond_dim: int = 64
    snapshot_interval_ticks: int = 50


class LiveCompressionMonitor:
    """
    Unified live compression quality monitor.

    Integrates:
      - Rolling reconstruction error tracking
      - Fidelity alerting
      - Compression ratio dashboard
      - Regime-aware quality monitoring
      - Rank collapse detection
      - Auto-healing bond dimension management
    """

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        on_fidelity_alert: Optional[Callable[[FidelityAlert], None]] = None,
        on_rank_collapse: Optional[Callable[[RankCollapseEvent], None]] = None,
        on_bond_change: Optional[Callable[[BondDimUpdate], None]] = None,
    ) -> None:
        self._cfg = config or MonitorConfig()

        # Per-schema error stats
        self._error_stats: Dict[str, RollingStats] = {}

        # Sub-systems
        self._alert_system = FidelityAlertSystem(
            warning_threshold=self._cfg.fidelity_warning_threshold,
            critical_threshold=self._cfg.fidelity_critical_threshold,
            on_warning=on_fidelity_alert,
            on_critical=on_fidelity_alert,
        )
        self._rank_detector = RankCollapseDetector(
            near_zero_tol=self._cfg.rank_collapse_tol,
            collapse_fraction=self._cfg.rank_collapse_fraction,
            on_collapse=on_rank_collapse,
        )
        self._cr_tracker = CompressionRatioTracker(window=self._cfg.rolling_window)
        self._regime_monitor = RegimeAwareMonitor(window=self._cfg.rolling_window)
        self._bond_manager = AutoHealingBondManager(
            initial_bond_dim=self._cfg.initial_bond_dim,
            min_bond_dim=self._cfg.min_bond_dim,
            max_bond_dim=self._cfg.max_bond_dim,
            on_bond_change=on_bond_change,
        ) if self._cfg.auto_heal else None

        self._tick_id: int = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Primary update interface
    # ------------------------------------------------------------------ #

    def record_compression(
        self,
        schema_name: str,
        original: np.ndarray,
        reconstructed: np.ndarray,
        compressed_bytes: int,
        singular_values: Optional[np.ndarray] = None,
        tick_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Record one compression event and run all checks.

        Parameters
        ----------
        schema_name:
            UTR schema name of the compressed tensor.
        original:
            Original tensor.
        reconstructed:
            Reconstructed (decompressed) tensor.
        compressed_bytes:
            Size of compressed representation in bytes.
        singular_values:
            Optional: singular values of the current TT core (for rank collapse).
        tick_id:
            Pipeline tick; uses internal counter if None.

        Returns
        -------
        Dict with 'error', 'compression_ratio', 'alert', 'rank_collapse_event',
        'bond_update'.
        """
        t_id = tick_id if tick_id is not None else self._tick_id
        with self._lock:
            self._tick_id = max(self._tick_id, t_id)

        # Error
        error = reconstruction_error(original, reconstructed, self._cfg.error_metric)
        stats = self._error_stats.setdefault(schema_name, RollingStats(self._cfg.rolling_window))
        stats.push(error)

        # Fidelity alert
        alert = self._alert_system.check(schema_name, t_id, error)

        # Compression ratio
        cr = self._cr_tracker.record(
            schema_name, t_id, original.nbytes, compressed_bytes
        )

        # Regime-aware record
        self._regime_monitor.record(schema_name, t_id, error, cr)

        # Rank collapse
        rc_event = None
        if singular_values is not None:
            rc_event = self._rank_detector.check_mode(schema_name, t_id, singular_values)

        # Auto-heal bond dim
        bond_update = None
        if self._bond_manager is not None and self._cfg.auto_heal:
            bond_update = self._bond_manager.update(schema_name, t_id, error)

        return {
            "error": error,
            "rolling_mean_error": stats.mean,
            "compression_ratio": cr,
            "alert": alert,
            "rank_collapse_event": rc_event,
            "bond_update": bond_update,
        }

    def update_market_regime(self, chronos_data: np.ndarray) -> MarketRegime:
        """Infer and update the current market regime from ChronosOutput data."""
        regime = infer_market_regime(
            chronos_data,
            stress_vol_threshold=self._cfg.stress_vol_threshold,
            crisis_vol_threshold=self._cfg.crisis_vol_threshold,
        )
        self._regime_monitor.update_regime(regime)
        return regime

    def advance_tick(self) -> int:
        with self._lock:
            self._tick_id += 1
            return self._tick_id

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def current_bond_dim(self) -> Optional[int]:
        return self._bond_manager.bond_dim if self._bond_manager else None

    @property
    def current_regime(self) -> MarketRegime:
        return self._regime_monitor.current_regime

    def get_error_stats(self, schema_name: str) -> Optional[RollingStats]:
        return self._error_stats.get(schema_name)

    def recent_alerts(self, n: int = 20) -> List[FidelityAlert]:
        return self._alert_system.recent_alerts(n)

    def recent_rank_collapses(self, n: int = 20) -> List[RankCollapseEvent]:
        return self._rank_detector.recent_events(n)

    def compression_ratio_dashboard(self) -> str:
        return self._cr_tracker.dashboard()

    def quality_by_regime(self, schema_name: Optional[str] = None) -> Dict[str, Any]:
        return self._regime_monitor.quality_by_regime(schema_name)

    def degradation_detected(self, schema_name: str) -> bool:
        return self._regime_monitor.degradation_detected(schema_name)

    def bond_update_history(self) -> List[BondDimUpdate]:
        if self._bond_manager is None:
            return []
        return self._bond_manager.update_history()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines = ["LiveCompressionMonitor Summary", "=" * 60]

        lines.append(f"\nCurrent tick: {self._tick_id}")
        lines.append(f"Market regime: {self.current_regime.name}")
        if self._bond_manager:
            lines.append(f"Current bond dim: {self._bond_manager.bond_dim}")

        lines.append("\nPer-schema rolling error (mean ± std):")
        for name, stats in sorted(self._error_stats.items()):
            lines.append(
                f"  {name:<35s}  {stats.mean:.4f} ± {stats.std:.4f}  "
                f"(max={stats.max:.4f}, n={stats.count})"
            )

        lines.append("")
        lines.append(self._cr_tracker.dashboard())

        n_alerts = len(self._alert_system.recent_alerts(1000))
        n_collapses = self._rank_detector.n_collapses()
        n_bond_changes = len(self.bond_update_history())
        lines.append(f"\nAlerts raised: {n_alerts}")
        lines.append(f"Rank collapse events: {n_collapses}")
        lines.append(f"Bond dim changes: {n_bond_changes}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Error metrics
    "ErrorMetric",
    "reconstruction_error",
    "compression_ratio",
    # Rolling stats
    "RollingStats",
    # Fidelity alerting
    "AlertSeverity",
    "FidelityAlert",
    "FidelityAlertSystem",
    # Rank collapse
    "detect_rank_collapse",
    "RankCollapseEvent",
    "RankCollapseDetector",
    # Compression ratio
    "CompressionRatioEntry",
    "CompressionRatioTracker",
    # Regime monitoring
    "MarketRegime",
    "infer_market_regime",
    "RegimeQualityRecord",
    "RegimeAwareMonitor",
    # Auto-healing
    "BondDimUpdate",
    "AutoHealingBondManager",
    # Unified monitor
    "MonitorConfig",
    "LiveCompressionMonitor",
]
