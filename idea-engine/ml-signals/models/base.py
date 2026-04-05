"""
models/base.py
==============
Abstract contract every ML signal must satisfy, plus the SignalMetrics
dataclass used throughout the evaluation pipeline.

Financial rationale
-------------------
All learned signals in SRFM are treated as *information sources* whose
value is measured through Information Coefficient (IC) – the
Spearman rank-correlation between the predicted score and the realised
forward return.  A model earns weight in the ensemble proportional to
its rolling IC.  SignalMetrics packages the four quantities that
matter most for a systematic crypto strategy:

    ic                 – mean Spearman IC over the evaluation period
    icir               – IC / std(IC) – analogous to a signal Sharpe ratio
    sharpe_contribution– incremental Sharpe when this signal is added to
                         the BH baseline (difference in portfolio Sharpe)
    max_drawdown_from_signal – worst peak-to-trough of cumulative P&L
                               driven by this signal alone
"""

from __future__ import annotations

import abc
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SignalMetrics
# ---------------------------------------------------------------------------

@dataclass
class SignalMetrics:
    """Aggregated performance statistics for one ML signal.

    Parameters
    ----------
    ic : float
        Mean Spearman rank-correlation between signal and forward return
        over the evaluation window.  Industry benchmark: IC > 0.05 is
        considered useful for low-frequency strategies.
    icir : float
        IC Information Ratio = mean(IC) / std(IC).  A value > 0.5
        suggests the signal is consistent, not just occasionally lucky.
    sharpe_contribution : float
        Incremental Sharpe ratio added to the portfolio when this signal
        is included.  Computed as Sharpe(BH + signal) - Sharpe(BH alone).
    max_drawdown_from_signal : float
        Largest peak-to-trough decline in the *signal-only* P&L stream
        (i.e., position = sign(signal), returns = forward_return).
        Expressed as a fraction, e.g. -0.15 means -15 %.
    n_predictions : int
        Number of prediction samples used to compute the metrics.
    eval_start : str, optional
        ISO date string for the start of the evaluation period.
    eval_end : str, optional
        ISO date string for the end of the evaluation period.
    extra : dict
        Arbitrary extra metadata (model name, hyper-parameters, etc.).
    """

    ic: float = 0.0
    icir: float = 0.0
    sharpe_contribution: float = 0.0
    max_drawdown_from_signal: float = 0.0
    n_predictions: int = 0
    eval_start: Optional[str] = None
    eval_end: Optional[str] = None
    extra: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def t_stat(self) -> float:
        """t-statistic for H0: IC == 0, assuming i.i.d. IC observations."""
        if self.n_predictions < 2 or self.icir == 0.0:
            return 0.0
        return self.icir * np.sqrt(self.n_predictions)

    def is_useful(self, min_ic: float = 0.02, min_icir: float = 0.3) -> bool:
        """Return True if the signal meets minimum usefulness thresholds."""
        return self.ic >= min_ic and self.icir >= min_icir

    def to_dict(self) -> Dict:
        """Serialise to plain dict for logging / JSON export."""
        return {
            "ic": round(self.ic, 6),
            "icir": round(self.icir, 6),
            "sharpe_contribution": round(self.sharpe_contribution, 6),
            "max_drawdown_from_signal": round(self.max_drawdown_from_signal, 6),
            "n_predictions": self.n_predictions,
            "eval_start": self.eval_start,
            "eval_end": self.eval_end,
            "t_stat": round(self.t_stat, 4),
            **self.extra,
        }

    def __str__(self) -> str:
        return (
            f"SignalMetrics(IC={self.ic:.4f}, ICIR={self.icir:.4f}, "
            f"Sharpe contrib={self.sharpe_contribution:.4f}, "
            f"MaxDD={self.max_drawdown_from_signal:.2%}, "
            f"n={self.n_predictions})"
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class MLSignal(abc.ABC):
    """Abstract base class for all SRFM ML signal models.

    Subclasses must implement :meth:`fit`, :meth:`predict`,
    :meth:`save`, :meth:`load`, and :meth:`feature_importance`.

    The fit/predict interface deliberately mirrors sklearn's transformer
    convention while adding financial-specific behaviour:

    * ``fit`` receives the full historical DataFrame including feature
      columns, target columns, and timestamp index.  The model is
      responsible for selecting the columns it needs.
    * ``predict`` returns a *single float* in [-1, +1]:
        +1  → maximum long signal
        -1  → maximum short signal
         0  → no position recommended
    * ``feature_importance`` returns a dict mapping feature name to
      importance score (higher = more important).  Used for reporting
      and drift detection.
    """

    def __init__(self, name: str = "MLSignal") -> None:
        self.name = name
        self._is_fitted: bool = False
        self._metrics: Optional[SignalMetrics] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame) -> "MLSignal":
        """Train the model on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Time-indexed DataFrame.  Must contain feature columns and
            at least one target column named ``target`` (forward return).
            Additional target columns (``target_sign``, etc.) may be
            present and used by specific subclasses.

        Returns
        -------
        self
            Fitted model instance (for method chaining).
        """

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> float:
        """Generate a signal score for the most recent bar.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame.  The model uses the *last* row (or last
            ``seq_len`` rows for sequence models) to produce the signal.

        Returns
        -------
        float
            Score in [-1, +1].  Positive = bullish, negative = bearish.
        """

    @abc.abstractmethod
    def save(self, path: pathlib.Path) -> None:
        """Persist model weights / parameters to ``path``."""

    @abc.abstractmethod
    def load(self, path: pathlib.Path) -> "MLSignal":
        """Restore model weights / parameters from ``path``.

        Returns
        -------
        self
        """

    @abc.abstractmethod
    def feature_importance(self) -> Dict[str, float]:
        """Return a dict of {feature_name: importance_score}.

        Importance scores should sum to 1.0 (or be normalised so that
        the most important feature equals 1.0 – document which
        convention the subclass uses).
        """

    # ------------------------------------------------------------------
    # Concrete helpers shared by all subclasses
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def metrics(self) -> Optional[SignalMetrics]:
        """Most recent evaluation metrics, set after calling evaluate()."""
        return self._metrics

    def evaluate(
        self,
        predictions: np.ndarray,
        forward_returns: np.ndarray,
    ) -> SignalMetrics:
        """Compute SignalMetrics from aligned prediction / return arrays.

        Parameters
        ----------
        predictions : np.ndarray, shape (N,)
            Model signal scores.
        forward_returns : np.ndarray, shape (N,)
            Realised forward returns aligned with predictions.

        Returns
        -------
        SignalMetrics
        """
        from scipy.stats import spearmanr

        n = len(predictions)
        if n < 10:
            return SignalMetrics(n_predictions=n)

        # Rolling IC (20-bar windows) to get ICIR
        window = min(20, n // 2)
        ics = []
        for i in range(window, n):
            p = predictions[i - window : i]
            r = forward_returns[i - window : i]
            if np.std(p) < 1e-8 or np.std(r) < 1e-8:
                continue
            ic_val, _ = spearmanr(p, r)
            if not np.isnan(ic_val):
                ics.append(ic_val)

        ics_arr = np.array(ics)
        ic_mean = float(np.mean(ics_arr)) if len(ics_arr) else 0.0
        ic_std = float(np.std(ics_arr)) if len(ics_arr) > 1 else 1.0
        icir = ic_mean / (ic_std + 1e-9)

        # Signal-only P&L: position = sign(prediction), no tx costs
        position = np.sign(predictions)
        pnl = position * forward_returns
        cum_pnl = np.cumprod(1.0 + np.clip(pnl, -0.5, 0.5))
        rolling_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl / rolling_max - 1.0
        max_dd = float(np.min(drawdowns))

        self._metrics = SignalMetrics(
            ic=ic_mean,
            icir=icir,
            sharpe_contribution=0.0,  # filled by ensemble / backtester
            max_drawdown_from_signal=max_dd,
            n_predictions=n,
        )
        return self._metrics

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted yet.  Call fit() first."
            )
