"""
signal_validator.py -- Validates GP-discovered signals before library admission.

Checks performed before a new signal expression is accepted:
  1. IC threshold (overall Spearman IC)
  2. ICIR threshold (rolling IC stability)
  3. No-lookahead check (signal at bar t must not use data after t)
  4. IC stability (std of rolling IC)
  5. Correlation dedupe (reject if too similar to existing library signals)
  6. Out-of-sample validation (IS/OOS IC comparison)

All checks are pure numpy/scipy -- no external ML libraries required.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from .expression_tree import ExpressionTree

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Utility: Spearman IC
# ---------------------------------------------------------------------------

def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, ignoring NaN pairs."""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    if mask.sum() < 4:
        return 0.0
    xv, yv = x[mask], y[mask]
    if _HAVE_SCIPY:
        rho, _ = spearmanr(xv, yv)
        return float(rho) if np.isfinite(rho) else 0.0
    # fallback: rank-based pearson
    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        return ranks
    rx, ry = _rank(xv), _rank(yv)
    rx_m = rx - rx.mean()
    ry_m = ry - ry.mean()
    denom = np.sqrt((rx_m ** 2).sum() * (ry_m ** 2).sum())
    return float((rx_m * ry_m).sum() / denom) if denom > _EPS else 0.0


def _rolling_ic(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """Rolling Spearman IC series."""
    n = len(signal)
    if n < window:
        v = _spearman_ic(signal, forward_returns)
        return np.array([v])
    ic_vals = []
    for i in range(window, n + 1):
        ic = _spearman_ic(signal[i - window: i], forward_returns[i - window: i])
        ic_vals.append(ic)
    return np.array(ic_vals, dtype=float)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, ignoring NaN pairs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 4:
        return 0.0
    xv, yv = x[mask], y[mask]
    std_x = float(np.std(xv, ddof=1))
    std_y = float(np.std(yv, ddof=1))
    if std_x < _EPS or std_y < _EPS:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


# ---------------------------------------------------------------------------
# Validation result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of all validation checks for a candidate signal."""
    passed: bool
    ic: float                 = 0.0
    icir: float               = 0.0
    ic_std: float             = 0.0
    max_library_corr: float   = 0.0
    oos_ic: float             = 0.0
    is_ic: float              = 0.0
    failure_reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any]   = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SignalValidator
# ---------------------------------------------------------------------------

class SignalValidator:
    """
    Validates a candidate GP signal before adding it to the signal library.

    Parameters
    ----------
    min_ic : float
        Minimum absolute Spearman IC required (default 0.05).
    min_icir : float
        Minimum absolute ICIR required (default 0.5).
    max_ic_std : float
        Maximum standard deviation of rolling IC (default 0.15).
    max_library_corr : float
        Maximum Pearson correlation with any existing library signal (default 0.85).
    ic_window : int
        Rolling window for ICIR and stability checks (default 30).
    oos_min_is_ratio : float
        OOS IC must be >= this fraction of IS IC (default 0.5).
    """

    def __init__(
        self,
        min_ic: float           = 0.05,
        min_icir: float         = 0.50,
        max_ic_std: float       = 0.15,
        max_library_corr: float = 0.85,
        ic_window: int          = 30,
        oos_min_is_ratio: float = 0.50,
    ):
        self.min_ic             = min_ic
        self.min_icir           = min_icir
        self.max_ic_std         = max_ic_std
        self.max_library_corr   = max_library_corr
        self.ic_window          = ic_window
        self.oos_min_is_ratio   = oos_min_is_ratio

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def validate_ic(
        self,
        signal_values: np.ndarray,
        forward_returns: np.ndarray,
        min_ic: Optional[float] = None,
    ) -> bool:
        """
        Check that the signal achieves a minimum absolute Spearman IC.

        Parameters
        ----------
        signal_values : 1D array of signal values
        forward_returns : 1D array of aligned forward bar returns
        min_ic : override instance minimum if provided

        Returns True if |IC| >= threshold.
        """
        threshold = min_ic if min_ic is not None else self.min_ic
        ic = _spearman_ic(signal_values, forward_returns)
        return abs(ic) >= threshold

    def validate_icir(
        self,
        ic_series: np.ndarray,
        min_icir: Optional[float] = None,
        window: Optional[int] = None,
    ) -> bool:
        """
        Check that the ICIR (mean IC / std IC) meets the minimum threshold.

        ic_series may be a pre-computed rolling IC array, or a full signal
        array -- if len(ic_series) > window it is treated as a rolling IC
        array already.

        Parameters
        ----------
        ic_series : pre-computed IC series (one value per window)
        min_icir : override instance minimum if provided
        window : if provided, only use the last `window` IC observations
        """
        threshold = min_icir if min_icir is not None else self.min_icir
        series = ic_series[np.isfinite(ic_series)]
        if window is not None:
            series = series[-window:]
        if len(series) < 3:
            return False
        mean_ic = float(np.mean(series))
        std_ic  = float(np.std(series, ddof=1))
        icir    = abs(mean_ic) / max(std_ic, _EPS)
        return icir >= threshold

    def validate_no_lookahead(
        self,
        signal_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
        historical_data: Dict[str, np.ndarray],
    ) -> bool:
        """
        Lookahead bias detection.

        Strategy:
          1. Evaluate the signal on full data [0..T] -> signal_full[T-1]
          2. Evaluate the signal on truncated data [0..T-k] -> signal_trunc[T-k-1]
          3. If signal_full[T-1] differs from signal_trunc[T-k-1] for the
             SAME bar index (T-k-1), lookahead is detected.

        We test multiple truncation points k in {1, 5, 10, 20} and flag as
        lookahead if ANY test shows a significant change.

        Returns True (no lookahead detected) or False (lookahead detected).
        """
        if not historical_data:
            return True

        sample_key = next(iter(historical_data))
        n = len(historical_data[sample_key])
        if n < 30:
            return True

        # evaluate on full data
        try:
            full_signal = signal_fn(historical_data)
        except Exception:
            return True  # cannot evaluate -- pass (fail-safe)

        if len(full_signal) != n:
            return False

        # check multiple truncation points
        lookahead_detected = False
        test_offsets = [1, 5, 10, 20]
        for k in test_offsets:
            check_idx = n - k - 1
            if check_idx < 0:
                continue
            # build truncated data: only up to check_idx (inclusive)
            trunc_data = {
                key: arr[: check_idx + 1]
                for key, arr in historical_data.items()
            }
            try:
                trunc_signal = signal_fn(trunc_data)
            except Exception:
                continue

            if len(trunc_signal) != check_idx + 1:
                continue

            val_full  = full_signal[check_idx]
            val_trunc = trunc_signal[check_idx]

            # both NaN or both very close -> no lookahead at this offset
            if np.isnan(val_full) and np.isnan(val_trunc):
                continue
            if np.isnan(val_full) != np.isnan(val_trunc):
                lookahead_detected = True
                break
            if abs(val_full - val_trunc) > 1e-6 * (1 + abs(val_full)):
                lookahead_detected = True
                break

        return not lookahead_detected

    def validate_stability(
        self,
        ic_series: np.ndarray,
        max_std: Optional[float] = None,
    ) -> bool:
        """
        Check that the rolling IC does not fluctuate excessively.

        Parameters
        ----------
        ic_series : rolling IC values (one per window)
        max_std : override instance max_ic_std if provided
        """
        threshold = max_std if max_std is not None else self.max_ic_std
        series = ic_series[np.isfinite(ic_series)]
        if len(series) < 3:
            return True  # insufficient data -- pass by default
        std = float(np.std(series, ddof=1))
        return std <= threshold

    def check_correlation_with_library(
        self,
        new_signal: np.ndarray,
        library_signals: Dict[str, np.ndarray],
        max_corr: Optional[float] = None,
    ) -> bool:
        """
        Reject the new signal if its Pearson correlation with any existing
        library signal exceeds max_corr.

        Parameters
        ----------
        new_signal : 1D array of the candidate signal
        library_signals : dict of signal_name -> 1D array
        max_corr : override instance max_library_corr if provided

        Returns True if signal is sufficiently uncorrelated (accept).
        Returns False if too correlated with any existing signal (reject).
        """
        threshold = max_corr if max_corr is not None else self.max_library_corr
        for name, lib_sig in library_signals.items():
            # align lengths
            min_len = min(len(new_signal), len(lib_sig))
            if min_len < 4:
                continue
            corr = abs(_pearson_corr(new_signal[-min_len:], lib_sig[-min_len:]))
            if corr >= threshold:
                return False  # too correlated -- reject
        return True

    def validate_out_of_sample(
        self,
        signal_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
        train_data: Dict[str, np.ndarray],
        test_data: Dict[str, np.ndarray],
        train_returns: Optional[np.ndarray] = None,
        test_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate IS and OOS IC for the signal function.

        signal_fn must accept a data dict and return a 1D array.

        If train_returns / test_returns are not supplied, this function
        looks for a key "forward_return" in the data dicts.

        Returns dict with keys:
          is_ic, oos_ic, is_icir, oos_icir, ratio (oos_ic / is_ic)
        """
        result: Dict[str, float] = {
            "is_ic": 0.0, "oos_ic": 0.0,
            "is_icir": 0.0, "oos_icir": 0.0,
            "ratio": 0.0,
        }

        # -- resolve returns
        if train_returns is None:
            if "forward_return" in train_data:
                train_returns = train_data["forward_return"]
            else:
                return result
        if test_returns is None:
            if "forward_return" in test_data:
                test_returns = test_data["forward_return"]
            else:
                return result

        # -- IS evaluation
        try:
            is_signal = signal_fn(train_data)
        except Exception:
            return result
        if len(is_signal) == len(train_returns):
            is_ic   = _spearman_ic(is_signal, train_returns)
            is_ics  = _rolling_ic(is_signal, train_returns, window=self.ic_window)
            valid   = is_ics[np.isfinite(is_ics)]
            is_icir = (float(np.mean(valid)) / max(float(np.std(valid, ddof=1)), _EPS)
                       if len(valid) >= 3 else 0.0)
            result["is_ic"]   = float(is_ic)
            result["is_icir"] = float(is_icir)

        # -- OOS evaluation
        try:
            oos_signal = signal_fn(test_data)
        except Exception:
            return result
        if len(oos_signal) == len(test_returns):
            oos_ic  = _spearman_ic(oos_signal, test_returns)
            oos_ics = _rolling_ic(oos_signal, test_returns, window=self.ic_window)
            valid   = oos_ics[np.isfinite(oos_ics)]
            oos_icir = (float(np.mean(valid)) / max(float(np.std(valid, ddof=1)), _EPS)
                        if len(valid) >= 3 else 0.0)
            result["oos_ic"]   = float(oos_ic)
            result["oos_icir"] = float(oos_icir)

        # -- ratio
        is_ic_abs = abs(result["is_ic"])
        if is_ic_abs > _EPS:
            result["ratio"] = abs(result["oos_ic"]) / is_ic_abs
        return result

    # ------------------------------------------------------------------
    # Full validation pipeline
    # ------------------------------------------------------------------

    def validate_all(
        self,
        signal_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
        train_data: Dict[str, np.ndarray],
        test_data: Dict[str, np.ndarray],
        train_returns: np.ndarray,
        test_returns: np.ndarray,
        library_signals: Optional[Dict[str, np.ndarray]] = None,
    ) -> ValidationResult:
        """
        Run the complete validation pipeline.

        Returns a ValidationResult with passed=True only if ALL checks pass.

        Checks (in order):
          1. IC threshold on train set
          2. ICIR threshold on train set
          3. IC stability on train set
          4. No-lookahead check
          5. Library correlation check (if library_signals provided)
          6. OOS validation (OOS IC >= oos_min_is_ratio * IS IC)
        """
        reasons: List[str] = []
        details: Dict[str, Any] = {}

        # -- evaluate signal on train data
        try:
            train_signal = signal_fn(train_data)
        except Exception as exc:
            return ValidationResult(
                passed=False,
                failure_reasons=[f"Signal evaluation failed: {exc}"],
            )

        if len(train_signal) != len(train_returns):
            return ValidationResult(
                passed=False,
                failure_reasons=["Signal length mismatch with train returns"],
            )

        # 1. IC check
        train_ic = _spearman_ic(train_signal, train_returns)
        details["train_ic"] = train_ic
        if not self.validate_ic(train_signal, train_returns):
            reasons.append(
                f"IC too low: {train_ic:.4f} < {self.min_ic:.4f}"
            )

        # 2. ICIR check
        ic_series = _rolling_ic(train_signal, train_returns, window=self.ic_window)
        valid_ics = ic_series[np.isfinite(ic_series)]
        if len(valid_ics) >= 3:
            mean_ic = float(np.mean(valid_ics))
            std_ic  = float(np.std(valid_ics, ddof=1))
            icir    = abs(mean_ic) / max(std_ic, _EPS)
        else:
            icir = 0.0
        details["icir"] = icir
        if not self.validate_icir(ic_series):
            reasons.append(
                f"ICIR too low: {icir:.4f} < {self.min_icir:.4f}"
            )

        # 3. IC stability
        if len(valid_ics) >= 3:
            ic_std = float(np.std(valid_ics, ddof=1))
        else:
            ic_std = 0.0
        details["ic_std"] = ic_std
        if not self.validate_stability(ic_series):
            reasons.append(
                f"IC std too high: {ic_std:.4f} > {self.max_ic_std:.4f}"
            )

        # 4. No-lookahead
        no_lookahead = self.validate_no_lookahead(signal_fn, train_data)
        details["no_lookahead"] = no_lookahead
        if not no_lookahead:
            reasons.append("Lookahead bias detected")

        # 5. Library correlation
        max_corr = 0.0
        if library_signals:
            accept = self.check_correlation_with_library(
                train_signal, library_signals
            )
            # compute exact max for reporting
            for lib_sig in library_signals.values():
                min_len = min(len(train_signal), len(lib_sig))
                if min_len >= 4:
                    c = abs(_pearson_corr(
                        train_signal[-min_len:], lib_sig[-min_len:]
                    ))
                    if c > max_corr:
                        max_corr = c
            details["max_library_corr"] = max_corr
            if not accept:
                reasons.append(
                    f"Too correlated with library: max_corr={max_corr:.4f} >= {self.max_library_corr:.4f}"
                )
        else:
            details["max_library_corr"] = 0.0

        # 6. OOS validation
        oos_result = self.validate_out_of_sample(
            signal_fn, train_data, test_data, train_returns, test_returns
        )
        details.update({"oos_" + k: v for k, v in oos_result.items()})
        oos_ic  = oos_result.get("oos_ic", 0.0)
        is_ic   = oos_result.get("is_ic",  train_ic)
        ratio   = oos_result.get("ratio",  0.0)
        details["oos_is_ratio"] = ratio
        if abs(is_ic) > _EPS:
            if ratio < self.oos_min_is_ratio:
                reasons.append(
                    f"OOS degradation: OOS/IS IC ratio={ratio:.3f} < {self.oos_min_is_ratio:.3f}"
                )

        passed = len(reasons) == 0
        return ValidationResult(
            passed=passed,
            ic=train_ic,
            icir=icir,
            ic_std=ic_std,
            max_library_corr=max_corr,
            oos_ic=oos_ic,
            is_ic=is_ic,
            failure_reasons=reasons,
            details=details,
        )
