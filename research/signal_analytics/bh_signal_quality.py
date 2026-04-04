"""
research/signal_analytics/bh_signal_quality.py
===============================================
BH-specific signal quality diagnostics.

BH signal variables:
  tf_score       : 0–7  (timeframe alignment count)
  mass           : 0–2  (conviction weight)
  ATR            : Average True Range (raw)
  ensemble_signal: −1 to +1 (composite directional score)
  delta_score    : tf_score × mass × ATR / vol²

This module measures:
  - Activation quality: what fraction of BH activations are profitable
  - IC of each BH signal component vs realised return
  - Mass threshold sweep: optimal entry threshold
  - TF-score analysis: performance by timeframe combination
  - Ensemble signal calibration (does signal magnitude predict return magnitude?)
  - ATR normalisation quality
  - Cross-instrument signal correlation

Usage example
-------------
>>> analyzer = BHSignalAnalyzer()
>>> report = analyzer.activation_quality(trades)
>>> sweep = analyzer.mass_threshold_sweep(trades, [0.5, 1.0, 1.5, 2.0])
>>> analyzer.plot_signal_quality_dashboard(report, save_path="results/bh_dashboard.png")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

from research.signal_analytics.ic_framework import ICCalculator


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PerformanceStats:
    """Basic performance statistics for a trade subset."""
    n_trades: int
    win_rate: float
    mean_return: float
    mean_pnl: float
    total_pnl: float
    sharpe: float
    max_drawdown: float
    avg_hold_bars: float
    ic_vs_return: float   # IC of signal vs normalised return


@dataclass
class ActivationQualityReport:
    """Comprehensive BH activation quality report."""
    total_activations: int
    profitable_fraction: float
    ic_tf_score: float
    ic_mass: float
    ic_delta_score: float
    ic_ensemble: float
    ic_atr: float
    mean_return_on_activation: float
    mean_pnl_on_activation: float
    total_pnl: float
    win_rate: float
    sharpe: float
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float
    long_ic: float         # IC of ensemble for long trades
    short_ic: float        # IC of ensemble for short trades


@dataclass
class TFScoreAnalysis:
    """Performance breakdown by tf_score value."""
    scores: List[int]
    n_trades: List[int]
    win_rates: List[float]
    mean_returns: List[float]
    mean_pnl: List[float]
    ics: List[float]              # IC of (ensemble or delta) at each tf_score
    best_score: int
    worst_score: int


@dataclass
class EnsembleQuality:
    """Ensemble signal quality metrics."""
    ic: float
    icir: float
    calibration_r2: float         # R² of |ensemble| vs |return|
    calibration_slope: float
    long_ic: float
    short_ic: float
    magnitude_ic: float           # IC of |ensemble| vs |return|
    direction_accuracy: float     # fraction of correct direction calls


@dataclass
class ATRQuality:
    """ATR normalisation quality metrics."""
    raw_atr_ic: float             # IC of raw ATR vs return
    normalised_ic: float          # IC of delta_score (ATR-normalised) vs return
    normalisation_improvement: float  # normalised_ic - raw_atr_ic
    optimal_atr_lookback: int     # best lookback period (if sweep provided)


# ---------------------------------------------------------------------------
# BHSignalAnalyzer
# ---------------------------------------------------------------------------

class BHSignalAnalyzer:
    """BH-specific signal quality diagnostics engine.

    Parameters
    ----------
    signal_col      : default signal column name for IC computation
    return_col      : column containing trade P&L
    dollar_pos_col  : column for position size (used to normalise returns)
    """

    def __init__(
        self,
        signal_col: str = "ensemble_signal",
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
    ) -> None:
        self.signal_col = signal_col
        self.return_col = return_col
        self.dollar_pos_col = dollar_pos_col
        self._ic_calc = ICCalculator()

    # ------------------------------------------------------------------ #
    # Normalised returns helper
    # ------------------------------------------------------------------ #

    def _normalised_returns(self, trades: pd.DataFrame) -> pd.Series:
        """Return P&L normalised by |dollar_pos|, or raw pnl if unavailable."""
        if self.dollar_pos_col in trades.columns:
            pos = trades[self.dollar_pos_col].abs().replace(0, np.nan)
            return (trades[self.return_col] / pos).rename("norm_return")
        return trades[self.return_col].rename("norm_return")

    # ------------------------------------------------------------------ #
    # Activation quality
    # ------------------------------------------------------------------ #

    def activation_quality(self, trades: pd.DataFrame) -> ActivationQualityReport:
        """Compute BH activation quality metrics.

        Measures:
          - What fraction of activations lead to profitable trades
          - IC of each BH signal component vs realised return
          - Long/short breakdown

        Parameters
        ----------
        trades : DataFrame with columns including pnl, tf_score, mass,
                 delta_score (if available), ensemble_signal (if available),
                 ATR (if available), dollar_pos (if available)

        Returns
        -------
        ActivationQualityReport
        """
        df = trades.copy()
        ret = self._normalised_returns(df)
        df["_ret"] = ret

        n_total = len(df)
        total_pnl = float(df[self.return_col].sum())
        profitable = float((df["_ret"] > 0).mean())
        mean_ret = float(df["_ret"].mean())
        mean_pnl = float(df[self.return_col].mean())
        win_rate = float((df[self.return_col] > 0).mean())

        # Sharpe (annualised, assuming each trade is independent)
        r_arr = df["_ret"].dropna().values
        sharpe = float(np.mean(r_arr) / np.std(r_arr, ddof=1) * np.sqrt(252)) if len(r_arr) > 1 else float("nan")

        # IC for each signal component
        ic_tf_score = self._ic_calc.ic_from_trades(df, "tf_score") if "tf_score" in df.columns else float("nan")
        ic_mass = self._ic_calc.ic_from_trades(df, "mass") if "mass" in df.columns else float("nan")
        ic_delta = self._ic_calc.ic_from_trades(df, "delta_score") if "delta_score" in df.columns else float("nan")
        ic_ens = self._ic_calc.ic_from_trades(df, self.signal_col) if self.signal_col in df.columns else float("nan")
        ic_atr = self._ic_calc.ic_from_trades(df, "ATR") if "ATR" in df.columns else float("nan")

        # Long/short breakdown
        long_ic, short_ic = float("nan"), float("nan")
        long_trades, short_trades = 0, 0
        long_wr, short_wr = float("nan"), float("nan")

        if self.signal_col in df.columns:
            long_mask = df[self.signal_col] > 0
            short_mask = df[self.signal_col] < 0
            long_df = df[long_mask]
            short_df = df[short_mask]
            long_trades = len(long_df)
            short_trades = len(short_df)
            if long_trades >= 3:
                long_ic = self._ic_calc.ic_from_trades(long_df, self.signal_col)
                long_wr = float((long_df[self.return_col] > 0).mean())
            if short_trades >= 3:
                short_ic = self._ic_calc.ic_from_trades(short_df, self.signal_col)
                short_wr = float((short_df[self.return_col] > 0).mean())

        return ActivationQualityReport(
            total_activations=n_total,
            profitable_fraction=profitable,
            ic_tf_score=ic_tf_score,
            ic_mass=ic_mass,
            ic_delta_score=ic_delta,
            ic_ensemble=ic_ens,
            ic_atr=ic_atr,
            mean_return_on_activation=mean_ret,
            mean_pnl_on_activation=mean_pnl,
            total_pnl=total_pnl,
            win_rate=win_rate,
            sharpe=sharpe,
            long_trades=long_trades,
            short_trades=short_trades,
            long_win_rate=long_wr,
            short_win_rate=short_wr,
            long_ic=long_ic,
            short_ic=short_ic,
        )

    # ------------------------------------------------------------------ #
    # Mass threshold sweep
    # ------------------------------------------------------------------ #

    def mass_threshold_sweep(
        self,
        trades: pd.DataFrame,
        mass_thresholds: Optional[List[float]] = None,
    ) -> Dict[float, PerformanceStats]:
        """Performance statistics for trades filtered by mass threshold.

        For each threshold t, only trades with mass ≥ t are included.

        Parameters
        ----------
        trades          : trade DataFrame with 'mass' column
        mass_thresholds : list of thresholds to test

        Returns
        -------
        Dict[threshold → PerformanceStats]
        """
        if "mass" not in trades.columns:
            raise ValueError("trades must have 'mass' column for mass_threshold_sweep")

        if mass_thresholds is None:
            mass_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

        results: dict[float, PerformanceStats] = {}
        for threshold in mass_thresholds:
            subset = trades[trades["mass"] >= threshold].copy()
            if len(subset) < 3:
                results[threshold] = PerformanceStats(
                    n_trades=len(subset),
                    win_rate=float("nan"),
                    mean_return=float("nan"),
                    mean_pnl=float("nan"),
                    total_pnl=float("nan"),
                    sharpe=float("nan"),
                    max_drawdown=float("nan"),
                    avg_hold_bars=float("nan"),
                    ic_vs_return=float("nan"),
                )
                continue
            results[threshold] = self._compute_performance_stats(subset)
        return results

    def _compute_performance_stats(self, trades: pd.DataFrame) -> PerformanceStats:
        """Compute performance stats for a subset of trades."""
        ret = self._normalised_returns(trades)
        pnl = trades[self.return_col]

        r_arr = ret.dropna().values
        sharpe = float(np.mean(r_arr) / np.std(r_arr, ddof=1) * np.sqrt(252)) if len(r_arr) > 1 else float("nan")

        # Max drawdown of cumulative pnl
        cum_pnl = pnl.cumsum()
        rolling_max = cum_pnl.cummax()
        dd = cum_pnl - rolling_max
        max_dd = float(dd.min())

        avg_hold = float(trades["hold_bars"].mean()) if "hold_bars" in trades.columns else float("nan")

        ic = float("nan")
        if self.signal_col in trades.columns:
            ic = self._ic_calc.ic_from_trades(trades, self.signal_col)

        return PerformanceStats(
            n_trades=len(trades),
            win_rate=float((pnl > 0).mean()),
            mean_return=float(ret.mean()),
            mean_pnl=float(pnl.mean()),
            total_pnl=float(pnl.sum()),
            sharpe=sharpe,
            max_drawdown=max_dd,
            avg_hold_bars=avg_hold,
            ic_vs_return=ic,
        )

    # ------------------------------------------------------------------ #
    # TF-score analysis
    # ------------------------------------------------------------------ #

    def tf_score_analysis(self, trades: pd.DataFrame) -> TFScoreAnalysis:
        """Performance breakdown by tf_score (0–7).

        Parameters
        ----------
        trades : trade DataFrame with 'tf_score' column

        Returns
        -------
        TFScoreAnalysis
        """
        if "tf_score" not in trades.columns:
            raise ValueError("trades must have 'tf_score' column")

        scores, n_trades, win_rates, mean_rets, mean_pnls, ics = [], [], [], [], [], []
        ret = self._normalised_returns(trades)
        trades = trades.copy()
        trades["_ret"] = ret

        for score in range(8):  # 0 to 7
            subset = trades[trades["tf_score"] == score]
            if len(subset) == 0:
                continue
            scores.append(score)
            n_trades.append(len(subset))
            win_rates.append(float((subset[self.return_col] > 0).mean()))
            mean_rets.append(float(subset["_ret"].mean()))
            mean_pnls.append(float(subset[self.return_col].mean()))

            ic = float("nan")
            if self.signal_col in subset.columns and len(subset) >= 3:
                ic = self._ic_calc.ic_from_trades(subset, self.signal_col)
            ics.append(ic)

        best_score = scores[int(np.argmax(mean_rets))] if mean_rets else -1
        worst_score = scores[int(np.argmin(mean_rets))] if mean_rets else -1

        return TFScoreAnalysis(
            scores=scores,
            n_trades=n_trades,
            win_rates=win_rates,
            mean_returns=mean_rets,
            mean_pnl=mean_pnls,
            ics=ics,
            best_score=best_score,
            worst_score=worst_score,
        )

    # ------------------------------------------------------------------ #
    # Ensemble signal quality
    # ------------------------------------------------------------------ #

    def ensemble_signal_quality(self, trades: pd.DataFrame) -> EnsembleQuality:
        """Measure quality of the ensemble_signal (−1 to +1).

        Computes:
          - IC (Spearman correlation with normalised return)
          - ICIR
          - Calibration: does signal magnitude predict return magnitude?
          - Direction accuracy
          - Long vs short IC

        Parameters
        ----------
        trades : trade DataFrame with ensemble_signal column

        Returns
        -------
        EnsembleQuality
        """
        if self.signal_col not in trades.columns:
            raise ValueError(f"trades must have '{self.signal_col}' column")

        df = trades.copy()
        df["_ret"] = self._normalised_returns(df)
        df = df[[self.signal_col, "_ret"]].dropna()

        if len(df) < 5:
            return EnsembleQuality(
                ic=float("nan"), icir=float("nan"), calibration_r2=float("nan"),
                calibration_slope=float("nan"), long_ic=float("nan"),
                short_ic=float("nan"), magnitude_ic=float("nan"),
                direction_accuracy=float("nan"),
            )

        sig = df[self.signal_col].values
        ret = df["_ret"].values

        # Overall IC
        r_ic, _ = stats.spearmanr(sig, ret)
        ic = float(r_ic)

        # ICIR — approximate using signal as a single "period" IC
        # We compute rolling ICs if there are enough observations
        ic_arr = np.array([ic])  # degenerate case for single cross-section
        icir_val = float("nan")
        if len(df) >= 20:
            # Compute IC in windows of 20 to get a distribution
            window_ics: list[float] = []
            for i in range(0, len(df) - 19, 10):
                w_sig = sig[i : i + 20]
                w_ret = ret[i : i + 20]
                r_w, _ = stats.spearmanr(w_sig, w_ret)
                window_ics.append(float(r_w))
            if len(window_ics) >= 2:
                ic_s = pd.Series(window_ics)
                icir_val = self._ic_calc.icir(ic_s)

        # Calibration: |signal| vs |return|
        abs_sig = np.abs(sig)
        abs_ret = np.abs(ret)
        r_calib, _ = stats.pearsonr(abs_sig, abs_ret)
        calib_r2 = float(r_calib**2)

        # Calibration slope (OLS: |ret| ~ slope × |sig|)
        if abs_sig.std() > 0:
            slope = float(np.cov(abs_sig, abs_ret)[0, 1] / np.var(abs_sig, ddof=1))
        else:
            slope = float("nan")

        # Magnitude IC (Spearman of |sig| vs |ret|)
        r_mag, _ = stats.spearmanr(abs_sig, abs_ret)
        magnitude_ic = float(r_mag)

        # Direction accuracy
        pred_dir = np.sign(sig)
        actual_dir = np.sign(ret)
        direction_acc = float(np.mean(pred_dir == actual_dir))

        # Long/short IC
        long_mask = sig > 0
        short_mask = sig < 0
        long_ic = float("nan")
        short_ic = float("nan")
        if long_mask.sum() >= 3:
            r_l, _ = stats.spearmanr(sig[long_mask], ret[long_mask])
            long_ic = float(r_l)
        if short_mask.sum() >= 3:
            r_s, _ = stats.spearmanr(sig[short_mask], ret[short_mask])
            short_ic = float(r_s)

        return EnsembleQuality(
            ic=ic,
            icir=icir_val,
            calibration_r2=calib_r2,
            calibration_slope=slope,
            long_ic=long_ic,
            short_ic=short_ic,
            magnitude_ic=magnitude_ic,
            direction_accuracy=direction_acc,
        )

    # ------------------------------------------------------------------ #
    # ATR normalisation quality
    # ------------------------------------------------------------------ #

    def atr_normalization_quality(
        self,
        trades: pd.DataFrame,
        atr_lookbacks: Optional[List[int]] = None,
    ) -> ATRQuality:
        """Measure whether ATR normalisation improves signal IC.

        Compares:
          - IC of raw ATR vs return
          - IC of delta_score (ATR-normalised) vs return

        Parameters
        ----------
        trades       : trade DataFrame
        atr_lookbacks: list of lookback periods (not used directly unless
                       ATR columns named ATR_N are present)

        Returns
        -------
        ATRQuality
        """
        df = trades.copy()
        df["_ret"] = self._normalised_returns(df)

        raw_atr_ic = float("nan")
        if "ATR" in df.columns:
            raw_atr_ic = self._ic_calc.ic_from_trades(df, "ATR")

        normalised_ic = float("nan")
        if "delta_score" in df.columns:
            normalised_ic = self._ic_calc.ic_from_trades(df, "delta_score")
        elif all(c in df.columns for c in ["tf_score", "mass", "ATR"]):
            # Approximate delta_score = tf_score × mass × ATR
            df["_delta_approx"] = df["tf_score"] * df["mass"] * df["ATR"]
            normalised_ic = self._ic_calc.ic_from_trades(df, "_delta_approx")

        improvement = (
            normalised_ic - raw_atr_ic
            if not np.isnan(normalised_ic) and not np.isnan(raw_atr_ic)
            else float("nan")
        )

        # Sweep over ATR lookback columns if present
        best_lookback = -1
        if atr_lookbacks:
            best_ic = float("-inf")
            for lb in atr_lookbacks:
                col = f"ATR_{lb}"
                if col in df.columns:
                    ic = self._ic_calc.ic_from_trades(df, col)
                    if not np.isnan(ic) and ic > best_ic:
                        best_ic = ic
                        best_lookback = lb

        return ATRQuality(
            raw_atr_ic=raw_atr_ic,
            normalised_ic=normalised_ic,
            normalisation_improvement=improvement,
            optimal_atr_lookback=best_lookback,
        )

    # ------------------------------------------------------------------ #
    # Cross-instrument signal correlation
    # ------------------------------------------------------------------ #

    def cross_instrument_signal_correlation(
        self,
        trades: pd.DataFrame,
        signal_col: Optional[str] = None,
        min_obs: int = 10,
    ) -> pd.DataFrame:
        """Compute correlation matrix of signal values across instruments.

        Trades are pivoted to (date/index × sym) and the Pearson correlation
        of signals is computed across all instrument pairs.

        Parameters
        ----------
        trades     : trade DataFrame with 'sym' column and signal column
        signal_col : signal column (defaults to self.signal_col)
        min_obs    : minimum number of shared observations to compute corr

        Returns
        -------
        pd.DataFrame — symmetric correlation matrix (instruments × instruments)
        """
        sig_col = signal_col or self.signal_col
        if "sym" not in trades.columns or sig_col not in trades.columns:
            raise ValueError("trades must have 'sym' and signal columns")

        # Pivot: one row per trade index, columns = sym
        # Use a reset index as row identifier
        df = trades[[sig_col, "sym"]].copy()
        df["_trade_idx"] = range(len(df))

        pivot = df.pivot_table(
            index="_trade_idx", columns="sym", values=sig_col, aggfunc="mean"
        )
        # Correlation matrix
        corr = pivot.corr(min_periods=min_obs)
        return corr

    # ------------------------------------------------------------------ #
    # Signal quality by regime
    # ------------------------------------------------------------------ #

    def signal_quality_by_regime(
        self,
        trades: pd.DataFrame,
        signal_col: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Break down signal IC and win-rate by regime label.

        Parameters
        ----------
        trades     : trade DataFrame with 'regime' column
        signal_col : signal column (defaults to self.signal_col)

        Returns
        -------
        Dict[regime_label → {'ic': float, 'win_rate': float, 'n_trades': int}]
        """
        if "regime" not in trades.columns:
            return {}

        sig_col = signal_col or self.signal_col
        df = trades.copy()
        df["_ret"] = self._normalised_returns(df)

        result: dict[str, dict[str, float]] = {}
        for regime in df["regime"].unique():
            sub = df[df["regime"] == regime]
            ic = float("nan")
            if sig_col in sub.columns and len(sub) >= 3:
                ic = self._ic_calc.ic_from_trades(sub, sig_col)
            result[str(regime)] = {
                "ic": ic,
                "win_rate": float((sub[self.return_col] > 0).mean()),
                "n_trades": len(sub),
                "mean_pnl": float(sub[self.return_col].mean()),
            }
        return result

    # ------------------------------------------------------------------ #
    # Signal stability over time
    # ------------------------------------------------------------------ #

    def signal_stability_over_time(
        self,
        trades: pd.DataFrame,
        signal_col: Optional[str] = None,
        n_periods: int = 4,
    ) -> pd.DataFrame:
        """Split trades into time periods and compute IC per period.

        Parameters
        ----------
        trades    : trade DataFrame with exit_time or DateTime index
        signal_col: signal column
        n_periods : number of equal-length time periods

        Returns
        -------
        pd.DataFrame with columns: period, n_trades, ic, win_rate, sharpe
        """
        sig_col = signal_col or self.signal_col
        df = trades.copy()
        df["_ret"] = self._normalised_returns(df)

        # Ensure sorted by time
        if "exit_time" in df.columns:
            df = df.sort_values("exit_time")
        else:
            df = df.sort_index()

        period_size = max(1, len(df) // n_periods)
        records: list[dict] = []

        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else len(df)
            sub = df.iloc[start:end]

            ic = float("nan")
            if sig_col in sub.columns and len(sub) >= 3:
                ic = self._ic_calc.ic_from_trades(sub, sig_col)

            r_arr = sub["_ret"].dropna().values
            sharpe = float("nan")
            if len(r_arr) > 1 and r_arr.std(ddof=1) > 0:
                sharpe = float(r_arr.mean() / r_arr.std(ddof=1) * np.sqrt(252))

            records.append({
                "period": i + 1,
                "start_idx": start,
                "end_idx": end,
                "n_trades": len(sub),
                "ic": ic,
                "win_rate": float((sub[self.return_col] > 0).mean()),
                "sharpe": sharpe,
                "mean_pnl": float(sub[self.return_col].mean()),
            })

        return pd.DataFrame(records).set_index("period")

    # ------------------------------------------------------------------ #
    # Comprehensive summary
    # ------------------------------------------------------------------ #

    def signal_summary_table(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Produce a summary table of all BH signal component ICs.

        Returns
        -------
        pd.DataFrame with index=signal_component, columns=[IC, n_obs, significant]
        """
        signal_cols = ["tf_score", "mass", "delta_score", self.signal_col, "ATR", "ensemble_signal"]
        records: list[dict] = []

        df = trades.copy()
        df["_ret"] = self._normalised_returns(df)

        for col in signal_cols:
            if col not in df.columns:
                continue
            sub = df[[col, "_ret"]].dropna()
            n = len(sub)
            ic = float("nan")
            t_stat = float("nan")
            p_val = float("nan")

            if n >= 3:
                r, p = stats.spearmanr(sub[col], sub["_ret"])
                ic = float(r)
                p_val = float(p)
                # Approximate t-stat
                if n > 2:
                    t_stat = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic**2, 1e-10))

            records.append({
                "signal": col,
                "ic": ic,
                "t_stat": t_stat,
                "p_value": p_val,
                "n_obs": n,
                "significant_5pct": abs(t_stat) > 1.96 if not np.isnan(t_stat) else False,
            })

        return pd.DataFrame(records).set_index("signal")

    # ------------------------------------------------------------------ #
    # Dashboard visualisation
    # ------------------------------------------------------------------ #

    def plot_signal_quality_dashboard(
        self,
        quality_report: ActivationQualityReport,
        trades: Optional[pd.DataFrame] = None,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Four-panel BH signal quality dashboard.

        Panels:
          1. IC bar chart for each signal component
          2. Win-rate and trade counts (long vs short)
          3. Mass threshold sweep (if trades provided)
          4. TF-score performance (if trades provided)

        Parameters
        ----------
        quality_report : ActivationQualityReport from activation_quality()
        trades         : original trades DataFrame (for detailed sub-plots)
        save_path      : optional save path
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # Panel 1: IC bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        ic_labels = ["tf_score", "mass", "delta_score", "ensemble", "ATR"]
        ic_values = [
            quality_report.ic_tf_score,
            quality_report.ic_mass,
            quality_report.ic_delta_score,
            quality_report.ic_ensemble,
            quality_report.ic_atr,
        ]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ic_values]
        ax1.barh(ic_labels, ic_values, color=colors, alpha=0.8)
        ax1.axvline(0, color="black", linewidth=0.8)
        ax1.set_xlabel("IC (Spearman)")
        ax1.set_title("Signal Component ICs")

        # Panel 2: Long/Short win rates
        ax2 = fig.add_subplot(gs[0, 1])
        categories = ["All", "Long", "Short"]
        win_rates_vals = [
            quality_report.win_rate,
            quality_report.long_win_rate,
            quality_report.short_win_rate,
        ]
        win_rates_clean = [v if not np.isnan(v) else 0 for v in win_rates_vals]
        ax2.bar(categories, win_rates_clean, color=["#3498db", "#2ecc71", "#e74c3c"], alpha=0.8)
        ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Win Rate")
        ax2.set_title(
            f"Win Rates  |  L={quality_report.long_trades}  S={quality_report.short_trades}"
        )

        # Panel 3: Activation summary text
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis("off")
        summary_text = "\n".join([
            f"Total Activations: {quality_report.total_activations}",
            f"Profitable Fraction: {quality_report.profitable_fraction:.1%}",
            f"Total PnL: ${quality_report.total_pnl:,.0f}",
            f"Mean PnL/Trade: ${quality_report.mean_pnl_on_activation:,.2f}",
            f"Sharpe: {quality_report.sharpe:.2f}",
            f"IC Ensemble: {quality_report.ic_ensemble:.4f}",
            f"IC TF-score: {quality_report.ic_tf_score:.4f}",
            f"IC Mass: {quality_report.ic_mass:.4f}",
            f"Long IC: {quality_report.long_ic:.4f}",
            f"Short IC: {quality_report.short_ic:.4f}",
        ])
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8})
        ax3.set_title("Summary Statistics")

        # Panel 4: Mass threshold sweep (if trades provided)
        ax4 = fig.add_subplot(gs[1, 0])
        if trades is not None and "mass" in trades.columns:
            thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            sweep = self.mass_threshold_sweep(trades, thresholds)
            t_vals = sorted(sweep.keys())
            wr_vals = [sweep[t].win_rate for t in t_vals]
            n_vals = [sweep[t].n_trades for t in t_vals]
            ax4.plot(t_vals, wr_vals, "o-", color="#3498db", linewidth=1.5, label="Win Rate")
            ax4_twin = ax4.twinx()
            ax4_twin.bar(t_vals, n_vals, color="#bdc3c7", alpha=0.4, label="N trades")
            ax4_twin.set_ylabel("N Trades", color="gray")
            ax4.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
            ax4.set_xlabel("Min Mass Threshold")
            ax4.set_ylabel("Win Rate")
            ax4.set_title("Mass Threshold Sweep")
            ax4.legend(loc="upper left", fontsize=8)
        else:
            ax4.text(0.5, 0.5, "No mass data", ha="center", va="center", transform=ax4.transAxes)
            ax4.set_title("Mass Threshold Sweep")

        # Panel 5: TF-score analysis (if trades provided)
        ax5 = fig.add_subplot(gs[1, 1])
        if trades is not None and "tf_score" in trades.columns:
            tf_analysis = self.tf_score_analysis(trades)
            colors_tf = ["#2ecc71" if v >= 0 else "#e74c3c" for v in tf_analysis.mean_returns]
            ax5.bar(tf_analysis.scores, tf_analysis.mean_returns, color=colors_tf, alpha=0.8)
            ax5.axhline(0, color="black", linewidth=0.8)
            ax5.set_xlabel("TF Score")
            ax5.set_ylabel("Mean Return")
            ax5.set_title(f"Performance by TF Score\nBest={tf_analysis.best_score}")
        else:
            ax5.text(0.5, 0.5, "No tf_score data", ha="center", va="center",
                     transform=ax5.transAxes)
            ax5.set_title("TF Score Analysis")

        # Panel 6: Regime breakdown (if trades provided)
        ax6 = fig.add_subplot(gs[1, 2])
        if trades is not None and "regime" in trades.columns:
            regime_stats = self.signal_quality_by_regime(trades)
            regimes = list(regime_stats.keys())
            regime_ics = [regime_stats[r]["ic"] for r in regimes]
            colors_reg = ["#2ecc71" if v >= 0 else "#e74c3c" for v in regime_ics]
            ax6.bar(regimes, regime_ics, color=colors_reg, alpha=0.8)
            ax6.axhline(0, color="black", linewidth=0.8)
            ax6.set_xlabel("Regime")
            ax6.set_ylabel("IC")
            ax6.set_title("Signal IC by Regime")
            ax6.tick_params(axis="x", rotation=30)
        else:
            ax6.text(0.5, 0.5, "No regime data", ha="center", va="center",
                     transform=ax6.transAxes)
            ax6.set_title("IC by Regime")

        fig.suptitle("BH Signal Quality Dashboard", fontsize=14, y=1.01)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_ensemble_calibration(
        self,
        trades: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Scatter plot of |ensemble_signal| vs |return| with regression line."""
        if self.signal_col not in trades.columns:
            raise ValueError(f"trades must have '{self.signal_col}' column")

        df = trades.copy()
        df["_ret"] = self._normalised_returns(df)
        df = df[[self.signal_col, "_ret"]].dropna()

        abs_sig = df[self.signal_col].abs().values
        abs_ret = df["_ret"].abs().values

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(abs_sig, abs_ret, alpha=0.3, s=15, color="#3498db", label="Trades")

        # Regression line
        slope, intercept, r_val, p_val, se = stats.linregress(abs_sig, abs_ret)
        x_line = np.linspace(abs_sig.min(), abs_sig.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
                label=f"Fit: slope={slope:.4f}  R²={r_val**2:.3f}")

        ax.set_xlabel(f"|{self.signal_col}|")
        ax.set_ylabel("|Normalised Return|")
        ax.set_title("Ensemble Signal Calibration\n(Does signal magnitude predict return magnitude?)")
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # Delta score sweep (tf_score * mass * ATR / vol^2)
    # ------------------------------------------------------------------ #

    def delta_score_threshold_sweep(
        self,
        trades: pd.DataFrame,
        thresholds: Optional[List[float]] = None,
        delta_col: str = "delta_score",
    ) -> Dict[float, PerformanceStats]:
        """Sweep over delta_score thresholds, same logic as mass_threshold_sweep.

        Parameters
        ----------
        trades     : trade DataFrame with delta_score column
        thresholds : list of minimum delta_score values to test
        delta_col  : column name for delta score

        Returns
        -------
        Dict[threshold -> PerformanceStats]
        """
        if delta_col not in trades.columns:
            # Try to compute it
            if all(c in trades.columns for c in ["tf_score", "mass", "ATR"]):
                trades = trades.copy()
                if "vol" in trades.columns:
                    vol2 = trades["vol"] ** 2
                    vol2 = vol2.replace(0, np.nan)
                    trades[delta_col] = trades["tf_score"] * trades["mass"] * trades["ATR"] / vol2
                else:
                    trades[delta_col] = trades["tf_score"] * trades["mass"] * trades["ATR"]
            else:
                raise ValueError(f"trades must have '{delta_col}' or tf_score/mass/ATR columns")

        if thresholds is None:
            abs_delta = trades[delta_col].abs()
            thresholds = list(np.quantile(abs_delta.dropna(), [0, 0.1, 0.25, 0.5, 0.75, 0.9]))

        results: dict[float, PerformanceStats] = {}
        for thr in thresholds:
            subset = trades[trades[delta_col].abs() >= thr].copy()
            if len(subset) < 3:
                results[thr] = PerformanceStats(
                    n_trades=len(subset), win_rate=float("nan"), mean_return=float("nan"),
                    mean_pnl=float("nan"), total_pnl=float("nan"), sharpe=float("nan"),
                    max_drawdown=float("nan"), avg_hold_bars=float("nan"), ic_vs_return=float("nan"),
                )
                continue
            results[thr] = self._compute_performance_stats(subset)
        return results

    # ------------------------------------------------------------------ #
    # BH signal persistence (consecutive activation analysis)
    # ------------------------------------------------------------------ #

    def signal_persistence_analysis(
        self,
        trades: pd.DataFrame,
        signal_col: Optional[str] = None,
        min_streak: int = 2,
    ) -> Dict[str, float]:
        """Analyse whether consecutive same-direction signals perform better.

        A streak of same-direction signals (e.g. 3 consecutive long signals)
        may indicate a stronger regime alignment and higher expected IC.

        Parameters
        ----------
        trades      : trade records with sym column (sorted by time)
        signal_col  : signal column (defaults to self.signal_col)
        min_streak  : minimum streak length to analyse

        Returns
        -------
        Dict with keys: mean_ret_streak, mean_ret_no_streak,
                        ic_streak, ic_no_streak, n_streak, n_no_streak
        """
        sig_col = signal_col or self.signal_col
        if sig_col not in trades.columns:
            return {}

        df = trades.copy()
        df["_ret"] = self._normalised_returns(df)
        df = df.sort_index()

        # Compute consecutive same-direction run-length per symbol
        if "sym" in df.columns:
            streak_flags: list[bool] = []
            for sym in df["sym"].unique():
                sym_df = df[df["sym"] == sym].sort_index()
                dir_series = np.sign(sym_df[sig_col].values)
                flags: list[bool] = [False]
                streak = 1
                for i in range(1, len(dir_series)):
                    if dir_series[i] == dir_series[i - 1]:
                        streak += 1
                    else:
                        streak = 1
                    flags.append(streak >= min_streak)
                streak_flags.extend(flags)
            df["_streak"] = streak_flags
        else:
            dir_series = np.sign(df[sig_col].values)
            flags = [False]
            streak = 1
            for i in range(1, len(dir_series)):
                if dir_series[i] == dir_series[i - 1]:
                    streak += 1
                else:
                    streak = 1
                flags.append(streak >= min_streak)
            df["_streak"] = flags

        streak_df = df[df["_streak"]]
        no_streak_df = df[~df["_streak"]]

        def safe_ic(sub: pd.DataFrame) -> float:
            s = sub[[sig_col, "_ret"]].dropna()
            if len(s) < 3:
                return float("nan")
            r, _ = stats.spearmanr(s[sig_col], s["_ret"])
            return float(r)

        return {
            "mean_ret_streak": float(streak_df["_ret"].mean()) if len(streak_df) > 0 else float("nan"),
            "mean_ret_no_streak": float(no_streak_df["_ret"].mean()) if len(no_streak_df) > 0 else float("nan"),
            "ic_streak": safe_ic(streak_df),
            "ic_no_streak": safe_ic(no_streak_df),
            "n_streak": len(streak_df),
            "n_no_streak": len(no_streak_df),
            "win_rate_streak": float((streak_df[self.return_col] > 0).mean()) if len(streak_df) > 0 else float("nan"),
            "win_rate_no_streak": float((no_streak_df[self.return_col] > 0).mean()) if len(no_streak_df) > 0 else float("nan"),
        }

    # ------------------------------------------------------------------ #
    # Hold-time distribution by signal strength
    # ------------------------------------------------------------------ #

    def hold_time_distribution(
        self,
        trades: pd.DataFrame,
        signal_col: Optional[str] = None,
        n_buckets: int = 5,
    ) -> pd.DataFrame:
        """Distribution of hold times by signal strength bucket.

        Parameters
        ----------
        trades    : trade records with hold_bars
        signal_col: signal column
        n_buckets : number of signal-strength buckets

        Returns
        -------
        pd.DataFrame[bucket x (mean_hold, median_hold, std_hold, n_trades)]
        """
        sig_col = signal_col or self.signal_col
        if "hold_bars" not in trades.columns:
            raise ValueError("trades must have 'hold_bars' column")

        df = trades[[sig_col, "hold_bars"]].dropna()
        if len(df) < n_buckets * 3:
            return pd.DataFrame()

        df["bucket"] = pd.qcut(df[sig_col].abs(), q=n_buckets, labels=False, duplicates="drop")
        records: list[dict] = []
        for q in sorted(df["bucket"].dropna().unique()):
            sub = df[df["bucket"] == q]["hold_bars"].values
            records.append({
                "bucket": f"S{int(q)+1}",
                "mean_hold": float(np.mean(sub)),
                "median_hold": float(np.median(sub)),
                "std_hold": float(np.std(sub, ddof=1)),
                "n_trades": len(sub),
            })
        return pd.DataFrame(records).set_index("bucket")

    # ------------------------------------------------------------------ #
    # Signal correlation over time (rolling pairwise)
    # ------------------------------------------------------------------ #

    def rolling_signal_correlation(
        self,
        trades: pd.DataFrame,
        signal_col_a: str,
        signal_col_b: str,
        window: int = 60,
    ) -> pd.Series:
        """Rolling Spearman correlation between two signal columns.

        Useful for detecting regime-dependent signal convergence.

        Parameters
        ----------
        trades       : trade records (sorted by time)
        signal_col_a : first signal column
        signal_col_b : second signal column
        window       : rolling window size

        Returns
        -------
        pd.Series of rolling correlation values
        """
        df = trades[[signal_col_a, signal_col_b]].dropna().copy()
        n = len(df)
        if n < window + 1:
            return pd.Series(dtype=float)

        corr_vals: list[float] = []
        for i in range(window - 1, n):
            w = df.iloc[i - window + 1 : i + 1]
            r, _ = stats.spearmanr(w[signal_col_a], w[signal_col_b])
            corr_vals.append(float(r))

        return pd.Series(corr_vals, index=df.index[window - 1 :], name=f"corr_{signal_col_a}_{signal_col_b}")

    # ------------------------------------------------------------------ #
    # BH Score composite quality metric
    # ------------------------------------------------------------------ #

    def bh_composite_quality_score(
        self,
        activation_report: ActivationQualityReport,
        weight_ic: float = 0.35,
        weight_win_rate: float = 0.25,
        weight_sharpe: float = 0.25,
        weight_consistency: float = 0.15,
    ) -> float:
        """Compute a single composite BH signal quality score (0-100).

        Combines IC, win rate, Sharpe, and long/short consistency into
        a single interpretable score.

        Parameters
        ----------
        activation_report : ActivationQualityReport from activation_quality()
        weight_ic         : weight for IC component (IC in [-1,1] normalised to [0,1])
        weight_win_rate   : weight for win rate component
        weight_sharpe     : weight for Sharpe component (capped at 3)
        weight_consistency: weight for long/short IC consistency

        Returns
        -------
        float in [0, 100]
        """
        # Normalise IC: -1 -> 0, +1 -> 100
        ic = activation_report.ic_ensemble
        ic_score = (ic + 1) / 2 * 100 if not np.isnan(ic) else 50.0

        # Win rate: 0 -> 0, 1 -> 100
        wr_score = activation_report.win_rate * 100

        # Sharpe: cap at 3, normalise to 0-100
        sharpe = activation_report.sharpe
        sharpe_score = min(max(sharpe, -3), 3) / 3 * 50 + 50 if not np.isnan(sharpe) else 50.0

        # Long/short consistency: how similar are long_ic and short_ic
        long_ic = activation_report.long_ic
        short_ic = activation_report.short_ic
        if not (np.isnan(long_ic) or np.isnan(short_ic)):
            consistency = 1 - abs(long_ic - short_ic)  # 1 when identical
            consistency_score = max(0, consistency) * 100
        else:
            consistency_score = 50.0

        total = (
            weight_ic * ic_score
            + weight_win_rate * wr_score
            + weight_sharpe * sharpe_score
            + weight_consistency * consistency_score
        )
        return float(total)

    # ------------------------------------------------------------------ #
    # Full BH quality report as DataFrame
    # ------------------------------------------------------------------ #

    def full_quality_report_df(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate a comprehensive BH quality report as a flat DataFrame.

        Combines activation quality, TF-score, ensemble calibration,
        and ATR normalisation metrics.

        Parameters
        ----------
        trades : trade records

        Returns
        -------
        pd.DataFrame[metric x value]
        """
        records: list[dict] = []

        # Activation quality
        aq = self.activation_quality(trades)
        records.append({"category": "activation", "metric": "n_activations", "value": aq.total_activations})
        records.append({"category": "activation", "metric": "profitable_fraction", "value": aq.profitable_fraction})
        records.append({"category": "activation", "metric": "win_rate", "value": aq.win_rate})
        records.append({"category": "activation", "metric": "sharpe", "value": aq.sharpe})
        records.append({"category": "activation", "metric": "total_pnl", "value": aq.total_pnl})
        records.append({"category": "ic", "metric": "ic_ensemble", "value": aq.ic_ensemble})
        records.append({"category": "ic", "metric": "ic_tf_score", "value": aq.ic_tf_score})
        records.append({"category": "ic", "metric": "ic_mass", "value": aq.ic_mass})
        records.append({"category": "ic", "metric": "ic_delta_score", "value": aq.ic_delta_score})
        records.append({"category": "ic", "metric": "ic_atr", "value": aq.ic_atr})
        records.append({"category": "directional", "metric": "long_ic", "value": aq.long_ic})
        records.append({"category": "directional", "metric": "short_ic", "value": aq.short_ic})
        records.append({"category": "directional", "metric": "long_win_rate", "value": aq.long_win_rate})
        records.append({"category": "directional", "metric": "short_win_rate", "value": aq.short_win_rate})

        # Ensemble quality
        if self.signal_col in trades.columns:
            try:
                eq = self.ensemble_signal_quality(trades)
                records.append({"category": "ensemble", "metric": "calibration_r2", "value": eq.calibration_r2})
                records.append({"category": "ensemble", "metric": "direction_accuracy", "value": eq.direction_accuracy})
                records.append({"category": "ensemble", "metric": "magnitude_ic", "value": eq.magnitude_ic})
                records.append({"category": "ensemble", "metric": "icir", "value": eq.icir})
            except Exception:
                pass

        # ATR quality
        try:
            atr_q = self.atr_normalization_quality(trades)
            records.append({"category": "atr", "metric": "raw_atr_ic", "value": atr_q.raw_atr_ic})
            records.append({"category": "atr", "metric": "normalised_ic", "value": atr_q.normalised_ic})
            records.append({"category": "atr", "metric": "normalisation_improvement", "value": atr_q.normalisation_improvement})
        except Exception:
            pass

        # Composite score
        composite = self.bh_composite_quality_score(aq)
        records.append({"category": "composite", "metric": "quality_score_0_100", "value": composite})

        return pd.DataFrame(records)
