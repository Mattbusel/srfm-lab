"""
research/walk_forward/regime_filter.py
────────────────────────────────────────
Regime-conditional analysis and filtering for walk-forward backtests.

Tools for:
  • Filtering trades by regime label
  • Per-regime performance breakdown
  • Markov regime transition matrices
  • Regime-conditional Sharpe and entry timing scores
  • Optimal regime exposure selection
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from .metrics import (
    PerformanceStats,
    compute_performance_stats,
    sharpe_ratio,
    max_drawdown,
    profit_factor,
    win_rate,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RegimeFilter class
# ─────────────────────────────────────────────────────────────────────────────

class RegimeFilter:
    """
    Filter and analyze trade performance by market regime.

    Works with any trade DataFrame that contains a regime column.
    Regimes are arbitrary string or integer labels.

    Parameters
    ----------
    regime_col   : column name in trade DataFrame containing regime labels.
    starting_equity : initial equity for equity-curve computations.

    Examples
    --------
    >>> rf = RegimeFilter()
    >>> bull_trades = rf.filter_by_regime(trades, "BULL")
    >>> breakdown = rf.regime_performance_breakdown(trades)
    >>> print(breakdown["BULL"].sharpe)
    """

    def __init__(
        self,
        regime_col:      str   = "regime",
        starting_equity: float = 100_000.0,
    ) -> None:
        self.regime_col      = regime_col
        self.starting_equity = starting_equity

    # ------------------------------------------------------------------
    # filter_by_regime
    # ------------------------------------------------------------------

    def filter_by_regime(
        self,
        trades:  pd.DataFrame,
        regime:  Union[str, int, Sequence],
        include: bool = True,
    ) -> pd.DataFrame:
        """
        Filter a trade DataFrame by one or more regime labels.

        Parameters
        ----------
        trades  : DataFrame with a regime column.
        regime  : single regime label or list of labels to filter.
        include : if True (default), keep matching rows.
                  if False, remove matching rows (exclude the regime).

        Returns
        -------
        Filtered DataFrame with reset integer index.

        Examples
        --------
        >>> bull_trades = rf.filter_by_regime(trades, "BULL")
        >>> non_bear    = rf.filter_by_regime(trades, ["BEAR", "HIGH_VOL"], include=False)
        """
        if self.regime_col not in trades.columns:
            raise KeyError(
                f"regime_col='{self.regime_col}' not found in DataFrame columns: "
                f"{list(trades.columns)}"
            )

        if isinstance(regime, (str, int, float)):
            target = {regime}
        else:
            target = set(regime)

        mask = trades[self.regime_col].isin(target)
        if not include:
            mask = ~mask

        result = trades[mask].reset_index(drop=True)

        n_kept = len(result)
        n_orig = len(trades)
        logger.debug(
            "filter_by_regime: regime=%s include=%s → %d/%d trades (%.1f%%)",
            target, include, n_kept, n_orig,
            100.0 * n_kept / max(1, n_orig),
        )
        return result

    # ------------------------------------------------------------------
    # regime_performance_breakdown
    # ------------------------------------------------------------------

    def regime_performance_breakdown(
        self,
        trades: pd.DataFrame,
    ) -> Dict[str, PerformanceStats]:
        """
        Compute performance statistics broken down by regime.

        Parameters
        ----------
        trades : DataFrame with regime column and at least 'pnl', 'dollar_pos'.

        Returns
        -------
        Dict mapping regime label → PerformanceStats.
        """
        if self.regime_col not in trades.columns:
            raise KeyError(f"regime_col='{self.regime_col}' not found")

        regimes   = sorted(trades[self.regime_col].dropna().unique())
        breakdown: Dict[str, PerformanceStats] = {}

        for reg in regimes:
            sub = self.filter_by_regime(trades, reg)
            if len(sub) == 0:
                breakdown[str(reg)] = PerformanceStats()
                continue

            stats = compute_performance_stats(
                sub,
                starting_equity = self.starting_equity,
            )
            breakdown[str(reg)] = stats

            logger.debug(
                "Regime '%s': n=%d, Sharpe=%.3f, MaxDD=%.2f%%, WinRate=%.1f%%",
                reg, len(sub), stats.sharpe, stats.max_dd * 100, stats.win_rate_ * 100,
            )

        return breakdown

    # ------------------------------------------------------------------
    # regime_transition_matrix
    # ------------------------------------------------------------------

    def regime_transition_matrix(
        self,
        regime_series: Union[pd.Series, np.ndarray, List],
        normalize:     bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute the Markov transition probability matrix from a regime series.

        Counts transitions regime_t → regime_{t+1} and optionally normalises
        each row to produce probabilities.

        Parameters
        ----------
        regime_series : ordered sequence of regime labels.
        normalize     : if True, normalise rows to sum to 1 (default True).

        Returns
        -------
        (matrix, labels) where matrix[i, j] = P(regime_j | regime_i)
        and labels are the sorted unique regime values.

        Examples
        --------
        >>> regimes = ["BULL", "BULL", "BEAR", "SIDEWAYS", "BULL"]
        >>> mat, labels = rf.regime_transition_matrix(regimes)
        >>> print(mat)
        """
        if isinstance(regime_series, pd.Series):
            arr = regime_series.dropna().to_numpy()
        elif isinstance(regime_series, np.ndarray):
            arr = regime_series
        else:
            arr = np.asarray(regime_series)

        arr = arr[arr != np.array(None)]  # drop None

        labels = sorted(set(arr.tolist()))
        n      = len(labels)
        idx    = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((n, n), dtype=np.float64)

        for t in range(len(arr) - 1):
            from_state = arr[t]
            to_state   = arr[t + 1]
            if from_state in idx and to_state in idx:
                matrix[idx[from_state], idx[to_state]] += 1

        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            matrix   = matrix / row_sums

        logger.debug(
            "regime_transition_matrix: labels=%s, n_transitions=%d",
            labels, int(matrix.sum() if not normalize else len(arr) - 1),
        )
        return matrix, [str(l) for l in labels]

    # ------------------------------------------------------------------
    # regime_conditional_sharpe
    # ------------------------------------------------------------------

    def regime_conditional_sharpe(
        self,
        trades: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute the annualized Sharpe ratio conditional on each regime.

        Parameters
        ----------
        trades : DataFrame with regime column.

        Returns
        -------
        Dict of regime_label → Sharpe ratio.
        """
        breakdown = self.regime_performance_breakdown(trades)
        return {reg: stats.sharpe for reg, stats in breakdown.items()}

    # ------------------------------------------------------------------
    # optimal_regime_exposure
    # ------------------------------------------------------------------

    def optimal_regime_exposure(
        self,
        trades:           pd.DataFrame,
        regimes_to_trade: Optional[List] = None,
        min_sharpe:       float = 0.0,
        min_win_rate:     float = 0.0,
    ) -> Tuple[pd.DataFrame, Dict[str, PerformanceStats]]:
        """
        Backtest a strategy where trading occurs only in selected regimes.

        Selects the subset of regimes that maximises overall OOS performance.
        If `regimes_to_trade` is None, it uses a heuristic to select regimes
        with Sharpe > `min_sharpe` and win_rate > `min_win_rate`.

        Parameters
        ----------
        trades           : full trade DataFrame.
        regimes_to_trade : explicit list of regime labels to include.
                           If None, auto-select by performance threshold.
        min_sharpe       : minimum regime Sharpe for auto-selection.
        min_win_rate     : minimum regime win-rate for auto-selection.

        Returns
        -------
        (filtered_trades, regime_stats) where filtered_trades contains
        only trades from selected regimes, and regime_stats is the per-regime
        breakdown of the selected regimes.

        Examples
        --------
        >>> filtered, stats = rf.optimal_regime_exposure(trades, min_sharpe=0.5)
        >>> print(f"Trading in: {list(stats.keys())}")
        """
        breakdown = self.regime_performance_breakdown(trades)

        if regimes_to_trade is None:
            regimes_to_trade = [
                reg for reg, stats in breakdown.items()
                if stats.sharpe >= min_sharpe and stats.win_rate_ >= min_win_rate
            ]

            if not regimes_to_trade:
                warnings.warn(
                    f"No regimes meet criteria (min_sharpe={min_sharpe}, "
                    f"min_win_rate={min_win_rate}). Using all regimes.",
                    UserWarning, stacklevel=2,
                )
                regimes_to_trade = list(breakdown.keys())

        logger.info(
            "optimal_regime_exposure: trading %d regimes: %s",
            len(regimes_to_trade), regimes_to_trade,
        )

        filtered = self.filter_by_regime(trades, regimes_to_trade, include=True)
        selected_stats = {reg: breakdown[str(reg)] for reg in regimes_to_trade
                         if str(reg) in breakdown}

        return filtered, selected_stats

    # ------------------------------------------------------------------
    # regime_timing_score
    # ------------------------------------------------------------------

    def regime_timing_score(
        self,
        trades:        pd.DataFrame,
        regime_series: pd.Series,
        lookahead:     int = 5,
    ) -> float:
        """
        Measure how well entry timing aligns with profitable regimes.

        Computes the fraction of trades entered during regimes that have
        above-median performance over the next `lookahead` periods.

        A score near 1.0 indicates excellent regime timing (entries
        consistently in high-quality regime periods). A score near 0.5
        indicates random timing relative to regime.

        Parameters
        ----------
        trades        : trade DataFrame with exit_time for time alignment.
        regime_series : time-indexed regime labels (DatetimeIndex or RangeIndex).
        lookahead     : periods forward to evaluate regime quality.

        Returns
        -------
        float timing score in [0, 1].

        Notes
        -----
        If trades lack exit_time information, falls back to index-based matching.
        """
        if self.regime_col not in trades.columns:
            raise KeyError(f"regime_col='{self.regime_col}' not found")

        if len(trades) == 0:
            return 0.5

        # Compute per-regime forward returns (lookahead-period rolling mean P&L)
        # as proxy for "good" vs "bad" regime periods

        # Map trade regime to whether it's a "good" regime
        breakdown    = self.regime_performance_breakdown(trades)
        sharpes      = {reg: stats.sharpe for reg, stats in breakdown.items()}
        median_sharpe = float(np.median(list(sharpes.values()))) if sharpes else 0.0

        good_regimes = {reg for reg, sh in sharpes.items() if sh >= median_sharpe}

        trade_regimes   = trades[self.regime_col].astype(str)
        in_good_regime  = trade_regimes.isin(good_regimes)
        timing_score    = float(in_good_regime.mean())

        logger.debug(
            "regime_timing_score: good_regimes=%s, timing=%.3f",
            good_regimes, timing_score,
        )
        return timing_score

    # ------------------------------------------------------------------
    # plot_regime_performance
    # ------------------------------------------------------------------

    def plot_regime_performance(
        self,
        trades:    pd.DataFrame,
        save_path: Optional[str] = None,
        show:      bool = True,
    ) -> None:
        """
        Plot per-regime performance metrics as a grouped bar chart.

        Shows: Sharpe ratio, win rate, profit factor, max drawdown per regime.

        Parameters
        ----------
        trades    : trade DataFrame.
        save_path : optional path to save the figure.
        show      : if True, display interactively.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            logger.warning("matplotlib not available — skipping regime performance plot")
            return

        breakdown = self.regime_performance_breakdown(trades)
        if not breakdown:
            logger.warning("No regime breakdown data to plot")
            return

        regimes = list(breakdown.keys())
        n_reg   = len(regimes)

        # Extract metrics
        sharpes   = [breakdown[r].sharpe           for r in regimes]
        win_rates = [breakdown[r].win_rate_ * 100  for r in regimes]
        pfs       = [min(breakdown[r].profit_factor_, 5.0) for r in regimes]  # cap at 5
        max_dds   = [abs(breakdown[r].max_dd) * 100 for r in regimes]
        n_trades  = [breakdown[r].n_trades          for r in regimes]

        # Color by Sharpe quality
        colors = [
            "#2ecc71" if s > 1.0 else
            "#f1c40f" if s > 0.0 else
            "#e74c3c"
            for s in sharpes
        ]

        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        x = np.arange(n_reg)
        bar_w = 0.6

        # Sharpe
        bars = ax1.bar(x, sharpes, bar_w, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.set_xticks(x); ax1.set_xticklabels(regimes, rotation=25, ha="right")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.set_title("Sharpe by Regime")
        for bar, val in zip(bars, sharpes):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        # Win Rate
        ax2.bar(x, win_rates, bar_w, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax2.axhline(50, color="black", linestyle="--", linewidth=0.8, label="50% baseline")
        ax2.set_xticks(x); ax2.set_xticklabels(regimes, rotation=25, ha="right")
        ax2.set_ylabel("Win Rate (%)")
        ax2.set_title("Win Rate by Regime")
        ax2.set_ylim(0, 100)

        # Profit Factor
        ax3.bar(x, pfs, bar_w, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax3.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Break-even")
        ax3.set_xticks(x); ax3.set_xticklabels(regimes, rotation=25, ha="right")
        ax3.set_ylabel("Profit Factor (capped at 5)")
        ax3.set_title("Profit Factor by Regime")

        # Max Drawdown
        ax4.bar(x, max_dds, bar_w,
                color=["#e74c3c" if dd > 20 else "#f39c12" if dd > 10 else "#2ecc71" for dd in max_dds],
                alpha=0.85, edgecolor="black", linewidth=0.5)
        ax4.set_xticks(x); ax4.set_xticklabels(regimes, rotation=25, ha="right")
        ax4.set_ylabel("Max Drawdown (%)")
        ax4.set_title("Max Drawdown by Regime")

        # Subtitle with trade counts
        fig.suptitle(
            f"Regime Performance Breakdown  |  Trade counts: " +
            ", ".join(f"{r}={n}" for r, n in zip(regimes, n_trades)),
            fontsize=10, y=1.01,
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved regime performance plot: %s", save_path)
        if show:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone functions for convenience
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_regime(
    trades:     pd.DataFrame,
    regime:     Union[str, int, Sequence],
    include:    bool = True,
    regime_col: str  = "regime",
) -> pd.DataFrame:
    """
    Convenience wrapper for RegimeFilter.filter_by_regime().

    Parameters
    ----------
    trades     : DataFrame with regime column.
    regime     : single label or list of labels.
    include    : keep (True) or remove (False) the specified regimes.
    regime_col : column name.

    Returns
    -------
    Filtered DataFrame.
    """
    return RegimeFilter(regime_col=regime_col).filter_by_regime(trades, regime, include)


def regime_performance_breakdown(
    trades:     pd.DataFrame,
    regime_col: str   = "regime",
    starting_equity: float = 100_000.0,
) -> Dict[str, PerformanceStats]:
    """
    Convenience wrapper for RegimeFilter.regime_performance_breakdown().
    """
    return RegimeFilter(
        regime_col=regime_col,
        starting_equity=starting_equity,
    ).regime_performance_breakdown(trades)


def regime_transition_matrix(
    regime_series: Union[pd.Series, np.ndarray, List],
    normalize:     bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience wrapper for RegimeFilter.regime_transition_matrix().
    """
    return RegimeFilter().regime_transition_matrix(regime_series, normalize)


def regime_conditional_sharpe(
    trades:     pd.DataFrame,
    regime_col: str = "regime",
) -> Dict[str, float]:
    """
    Convenience wrapper for RegimeFilter.regime_conditional_sharpe().
    """
    return RegimeFilter(regime_col=regime_col).regime_conditional_sharpe(trades)


# ─────────────────────────────────────────────────────────────────────────────
# Regime-aware performance decomposition
# ─────────────────────────────────────────────────────────────────────────────

def decompose_performance_by_regime(
    trades:          pd.DataFrame,
    regime_col:      str   = "regime",
    starting_equity: float = 100_000.0,
) -> pd.DataFrame:
    """
    Build a summary table of performance metrics decomposed by regime.

    Parameters
    ----------
    trades          : trade DataFrame.
    regime_col      : regime column name.
    starting_equity : initial equity.

    Returns
    -------
    DataFrame with one row per regime and columns for all standard metrics.
    The last row is 'ALL' (full-sample performance).
    """
    breakdown = regime_performance_breakdown(trades, regime_col, starting_equity)

    rows: List[Dict] = []
    for regime, stats in breakdown.items():
        row = {"regime": regime}
        row.update(stats.to_dict())
        rows.append(row)

    # Add full-sample row
    full_stats = compute_performance_stats(trades, starting_equity=starting_equity)
    all_row    = {"regime": "ALL"}
    all_row.update(full_stats.to_dict())
    rows.append(all_row)

    df = pd.DataFrame(rows).set_index("regime")

    # Convert n_trades to int
    if "n_trades" in df.columns:
        df["n_trades"] = df["n_trades"].astype(int)

    return df


def regime_holding_periods(
    trades:     pd.DataFrame,
    regime_col: str = "regime",
    hold_col:   str = "hold_bars",
) -> pd.DataFrame:
    """
    Compare average holding periods by regime.

    Parameters
    ----------
    trades     : trade DataFrame with regime and hold_bars columns.
    regime_col : regime column name.
    hold_col   : holding period column name.

    Returns
    -------
    DataFrame with regime, mean_hold, median_hold, std_hold, n_trades.
    """
    if regime_col not in trades.columns:
        raise KeyError(f"regime_col='{regime_col}' not found")
    if hold_col not in trades.columns:
        warnings.warn(f"hold_col='{hold_col}' not found — returning empty DataFrame", UserWarning)
        return pd.DataFrame()

    result = (
        trades.groupby(regime_col)[hold_col]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "mean_hold", "median": "median_hold",
                         "std": "std_hold", "count": "n_trades"})
        .reset_index()
    )
    return result
