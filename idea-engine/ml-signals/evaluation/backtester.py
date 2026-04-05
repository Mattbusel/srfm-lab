"""
evaluation/backtester.py
=========================
Backtest an ML signal as a standalone strategy and compare it to the
BH (Black-Hole physics) baseline.

Financial rationale
-------------------
A backtest simulates what would have happened if we had traded the signal
in the past.  The primary metrics are:

IC (Information Coefficient)
    Spearman rank-correlation between signal and next-bar return.
    Measures *predictive power* without any assumption about position
    sizing.  IC is independent of transaction costs.

ICIR (IC Information Ratio)
    mean(IC) / std(IC) over a rolling window.  Rewards *consistency*:
    a signal with IC=0.05 every day is worth more than one with
    IC=0.15 half the time and IC=-0.05 the other half.

Sharpe Ratio
    annualised return / annualised volatility of signal-driven P&L.
    Position = sign(signal), return = next-bar return × position.
    Annualisation factor: sqrt(bars_per_year).

Maximum Drawdown
    Largest peak-to-trough decline in cumulative P&L.

Calmar Ratio
    annualised_return / |max_drawdown|.  Higher = better risk-adjusted.

BH Baseline
    The physics-based signal from the BH engine, evaluated with the same
    metrics.  We compute incremental Sharpe:
        sharpe_contribution = Sharpe(ML only) - Sharpe(BH only)
    and joint Sharpe:
        Sharpe(0.5 × ML + 0.5 × BH)
    to quantify the diversification benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..models.base import MLSignal, SignalMetrics


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Complete backtest metrics for one signal."""
    signal_name:   str
    n_bars:        int
    ic:            float
    icir:          float
    sharpe:        float
    calmar:        float
    max_drawdown:  float
    annual_return: float
    hit_rate:      float        # fraction of bars with correct direction
    pnl_series:    np.ndarray   # cumulative P&L
    signal_series: np.ndarray
    return_series: np.ndarray
    extra:         Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal_name":   self.signal_name,
            "n_bars":        self.n_bars,
            "ic":            round(self.ic, 6),
            "icir":          round(self.icir, 6),
            "sharpe":        round(self.sharpe, 4),
            "calmar":        round(self.calmar, 4),
            "max_drawdown":  round(self.max_drawdown, 4),
            "annual_return": round(self.annual_return, 4),
            "hit_rate":      round(self.hit_rate, 4),
            **self.extra,
        }

    def __str__(self) -> str:
        return (
            f"Backtest({self.signal_name}): "
            f"IC={self.ic:.4f} ICIR={self.icir:.4f} "
            f"Sharpe={self.sharpe:.2f} MaxDD={self.max_drawdown:.2%} "
            f"Calmar={self.calmar:.2f} HitRate={self.hit_rate:.2%}"
        )


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Simulate ML signal strategy on historical data.

    Parameters
    ----------
    bars_per_year : int
        Number of bars in a year.  Use 252 for daily, 8760 for hourly.
    transaction_cost : float
        One-way transaction cost as a fraction of trade value.
        e.g. 0.001 = 10 bps per trade.
    signal_threshold : float
        Minimum |signal| to enter a position.  Below this → flat.
    ic_window : int
        Rolling window for computing ICIR.
    """

    def __init__(
        self,
        bars_per_year:    int   = 252,
        transaction_cost: float = 0.001,
        signal_threshold: float = 0.1,
        ic_window:        int   = 20,
    ) -> None:
        self.bars_per_year    = bars_per_year
        self.transaction_cost = transaction_cost
        self.signal_threshold = signal_threshold
        self.ic_window        = ic_window

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def run(
        self,
        signal:       np.ndarray,
        returns:      np.ndarray,
        signal_name:  str = "ML Signal",
    ) -> BacktestResult:
        """Simulate trading the ``signal`` against realised ``returns``.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Signal scores in [-1, +1].
        returns : np.ndarray, shape (N,)
            Forward returns aligned with signal.
        signal_name : str

        Returns
        -------
        BacktestResult
        """
        n = min(len(signal), len(returns))
        sig = np.array(signal[:n], dtype=np.float64)
        ret = np.array(returns[:n], dtype=np.float64)

        # Position: threshold filter, then sign
        position = np.where(np.abs(sig) >= self.signal_threshold,
                             np.sign(sig), 0.0)

        # Transaction costs: cost on every position change
        pos_change  = np.abs(np.diff(np.concatenate([[0], position])))
        tc_per_bar  = pos_change * self.transaction_cost
        gross_pnl   = position * ret
        net_pnl     = gross_pnl - tc_per_bar

        # Cumulative P&L
        cum_pnl = np.cumprod(1.0 + np.clip(net_pnl, -0.5, 0.5))

        # IC
        ic_arr = self._rolling_ic(sig, ret)
        ic_mean = float(np.nanmean(ic_arr))
        ic_std  = float(np.nanstd(ic_arr))
        icir    = ic_mean / (ic_std + 1e-9)

        # Drawdown
        rolling_max = np.maximum.accumulate(cum_pnl)
        drawdowns   = cum_pnl / rolling_max - 1.0
        max_dd      = float(np.min(drawdowns))

        # Annualised return
        total_ret   = float(cum_pnl[-1]) - 1.0
        n_years     = n / self.bars_per_year
        ann_ret     = float((1.0 + total_ret) ** (1.0 / max(n_years, 1e-3)) - 1.0)

        # Sharpe
        ann_vol  = float(np.std(net_pnl) * np.sqrt(self.bars_per_year))
        sharpe   = ann_ret / (ann_vol + 1e-9)

        # Calmar
        calmar = ann_ret / (abs(max_dd) + 1e-9)

        # Hit rate (direction correctness when in position)
        in_pos = np.abs(position) > 0
        if in_pos.sum() > 0:
            hit_rate = float(np.mean(np.sign(position[in_pos]) == np.sign(ret[in_pos])))
        else:
            hit_rate = 0.5

        return BacktestResult(
            signal_name   = signal_name,
            n_bars        = n,
            ic            = ic_mean,
            icir          = icir,
            sharpe        = sharpe,
            calmar        = calmar,
            max_drawdown  = max_dd,
            annual_return = ann_ret,
            hit_rate      = hit_rate,
            pnl_series    = cum_pnl,
            signal_series = sig,
            return_series = ret,
        )

    # ------------------------------------------------------------------
    # Comparison: ML vs BH baseline
    # ------------------------------------------------------------------

    def compare(
        self,
        ml_signal:  np.ndarray,
        bh_signal:  np.ndarray,
        returns:    np.ndarray,
        blend:      float = 0.5,
    ) -> Dict[str, BacktestResult]:
        """Compare ML signal to BH baseline.

        Parameters
        ----------
        ml_signal : np.ndarray
        bh_signal : np.ndarray  BH physics signal, aligned with ml_signal
        returns   : np.ndarray  realised forward returns
        blend     : float       weight of ML signal in combined signal

        Returns
        -------
        dict with keys: 'ml', 'bh', 'combined'
        """
        n = min(len(ml_signal), len(bh_signal), len(returns))

        ml_res   = self.run(ml_signal[:n], returns[:n], "ML Signal")
        bh_res   = self.run(bh_signal[:n], returns[:n], "BH Baseline")

        combined = blend * ml_signal[:n] + (1 - blend) * bh_signal[:n]
        comb_res = self.run(combined, returns[:n], "Combined (ML+BH)")

        # Sharpe contribution
        ml_res.extra["sharpe_contribution"] = ml_res.sharpe - bh_res.sharpe
        ml_res.extra["combined_sharpe"]     = comb_res.sharpe

        return {"ml": ml_res, "bh": bh_res, "combined": comb_res}

    # ------------------------------------------------------------------
    # Model evaluation wrapper
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model:        MLSignal,
        df:           pd.DataFrame,
        bh_signal_col: Optional[str] = "bh_signal",
    ) -> Dict[str, BacktestResult]:
        """Generate predictions from ``model`` on ``df`` and backtest.

        Uses expanding-window prediction to avoid lookahead.
        """
        preds = []
        for i in range(len(df)):
            try:
                p = model.predict(df.iloc[:i + 1])
            except Exception:
                p = 0.0
            preds.append(float(p))

        preds   = np.array(preds)
        returns = df["target"].values if "target" in df.columns else np.zeros(len(df))

        if bh_signal_col and bh_signal_col in df.columns:
            bh_sig = df[bh_signal_col].values.astype(float)
            return self.compare(preds, bh_sig, returns)
        else:
            ml_res = self.run(preds, returns, model.name)
            return {"ml": ml_res}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rolling_ic(self, signal: np.ndarray, returns: np.ndarray) -> np.ndarray:
        n   = len(signal)
        ics = np.full(n, np.nan)
        for i in range(self.ic_window, n):
            s = signal[i - self.ic_window : i]
            r = returns[i - self.ic_window : i]
            if np.std(s) < 1e-9 or np.std(r) < 1e-9:
                continue
            ic, _ = spearmanr(s, r)
            if not np.isnan(ic):
                ics[i] = ic
        return ics

    @staticmethod
    def metrics_table(results: List[BacktestResult]) -> pd.DataFrame:
        """Convert a list of results to a comparison DataFrame."""
        return pd.DataFrame([r.to_dict() for r in results]).set_index("signal_name")
