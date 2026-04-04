"""
research/signal_analytics/quantile_analysis.py
===============================================
Quintile (quantile) portfolio analysis.

Provides:
  - Quintile return spread (Q5 − Q1)
  - IC per quantile bin
  - Quintile turnover
  - Monotonicity score (how monotone is the signal-return relationship)
  - Hit-rate by quantile
  - Bar chart and cumulative equity curve visualisations

Usage example
-------------
>>> qa = QuantileAnalyzer(n_quantiles=5)
>>> result = qa.compute_quintile_returns(signal, forward_returns)
>>> qa.plot_quintile_bar(result, save_path="results/quintile.png")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QuantileResult:
    """Output of quintile portfolio analysis."""
    n_quantiles: int
    quantile_labels: List[str]
    mean_returns: List[float]          # average forward return per quantile
    median_returns: List[float]
    std_returns: List[float]
    hit_rates: List[float]             # fraction of positive returns per Q
    n_obs: List[int]                   # number of observations per Q
    spread: float                      # Q_top − Q_bottom mean return
    spread_t_stat: float               # t-stat of Q5-Q1 spread
    spread_p_value: float              # p-value of spread
    monotonicity_score: float          # 0→1, how monotone the quintile returns are
    signal_col: str = "signal"

    @property
    def long_return(self) -> float:
        return self.mean_returns[-1]

    @property
    def short_return(self) -> float:
        return self.mean_returns[0]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "quantile": self.quantile_labels,
            "mean_return": self.mean_returns,
            "median_return": self.median_returns,
            "std_return": self.std_returns,
            "hit_rate": self.hit_rates,
            "n_obs": self.n_obs,
        }).set_index("quantile")


@dataclass
class QuintileCumulativeResult:
    """Cumulative return paths per quantile."""
    cumulative_returns: pd.DataFrame  # time × quantile
    n_quantiles: int


# ---------------------------------------------------------------------------
# QuantileAnalyzer
# ---------------------------------------------------------------------------

class QuantileAnalyzer:
    """Quintile portfolio analysis engine.

    Parameters
    ----------
    n_quantiles : number of equally-sized quantile buckets (default 5)
    """

    def __init__(self, n_quantiles: int = 5) -> None:
        self.n_quantiles = n_quantiles

    # ------------------------------------------------------------------ #
    # Core quintile analysis
    # ------------------------------------------------------------------ #

    def compute_quintile_returns(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: Optional[int] = None,
        signal_col: str = "signal",
    ) -> QuantileResult:
        """Sort observations into signal quantiles and compute return statistics.

        Parameters
        ----------
        signal          : signal values (cross-sectional or pooled)
        forward_returns : corresponding forward returns
        n_quantiles     : number of quantile buckets (default self.n_quantiles)
        signal_col      : name label for the signal

        Returns
        -------
        QuantileResult with per-quantile statistics
        """
        nq = n_quantiles or self.n_quantiles
        df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1).dropna()
        if len(df) < nq * 3:
            raise ValueError(f"Insufficient observations ({len(df)}) for {nq} quantiles.")

        # Assign quantile labels using equal-frequency buckets
        try:
            df["q"] = pd.qcut(df["sig"], q=nq, labels=False, duplicates="drop")
        except Exception:
            df["q"] = pd.cut(df["sig"], bins=nq, labels=False)

        labels: list[str] = []
        means, medians, stds, hits, ns = [], [], [], [], []
        quantile_ret_vecs: list[np.ndarray] = []

        for q in sorted(df["q"].dropna().unique()):
            bucket_ret = df.loc[df["q"] == q, "ret"].values
            labels.append(f"Q{int(q)+1}")
            means.append(float(np.mean(bucket_ret)))
            medians.append(float(np.median(bucket_ret)))
            stds.append(float(np.std(bucket_ret, ddof=1)))
            hits.append(float(np.mean(bucket_ret > 0)))
            ns.append(len(bucket_ret))
            quantile_ret_vecs.append(bucket_ret)

        # Spread: Q_top − Q_bottom
        if len(quantile_ret_vecs) >= 2:
            top_vec = quantile_ret_vecs[-1]
            bot_vec = quantile_ret_vecs[0]
            spread_vec = np.concatenate([top_vec, -bot_vec])
            t_stat, p_val = stats.ttest_ind(top_vec, bot_vec, equal_var=False)
            spread = means[-1] - means[0]
        else:
            spread, t_stat, p_val = 0.0, float("nan"), float("nan")

        mono_score = self._monotonicity_score(means)

        return QuantileResult(
            n_quantiles=nq,
            quantile_labels=labels,
            mean_returns=means,
            median_returns=medians,
            std_returns=stds,
            hit_rates=hits,
            n_obs=ns,
            spread=float(spread),
            spread_t_stat=float(t_stat),
            spread_p_value=float(p_val),
            monotonicity_score=mono_score,
            signal_col=signal_col,
        )

    def _monotonicity_score(self, values: list[float]) -> float:
        """Measure how monotonically increasing values are.

        Score = fraction of consecutive pairs where v[i] < v[i+1].
        A perfectly monotone ascending series scores 1.0.

        Parameters
        ----------
        values : list of float values

        Returns
        -------
        float in [0, 1]
        """
        if len(values) < 2:
            return float("nan")
        n_pairs = len(values) - 1
        n_correct = sum(1 for i in range(n_pairs) if values[i] <= values[i + 1])
        return float(n_correct / n_pairs)

    # ------------------------------------------------------------------ #
    # Panel quintile analysis (time-series of cross-sections)
    # ------------------------------------------------------------------ #

    def panel_quintile_returns(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        n_quantiles: Optional[int] = None,
    ) -> QuantileResult:
        """Pool all time-period observations and run quintile analysis.

        Parameters
        ----------
        signal_df  : DataFrame[time × assets]
        returns_df : DataFrame[time × assets]
        n_quantiles: number of buckets

        Returns
        -------
        QuantileResult on pooled data
        """
        cols = signal_df.columns.intersection(returns_df.columns)
        idx = signal_df.index.intersection(returns_df.index)

        sig_flat = signal_df.loc[idx, cols].stack()
        ret_flat = returns_df.loc[idx, cols].stack()
        return self.compute_quintile_returns(
            sig_flat, ret_flat, n_quantiles=n_quantiles, signal_col="pooled_signal"
        )

    # ------------------------------------------------------------------ #
    # IC by quantile
    # ------------------------------------------------------------------ #

    def information_coefficient_by_quintile(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: Optional[int] = None,
    ) -> Dict[str, float]:
        """Within-bucket Spearman IC per quantile.

        This measures whether the signal retains predictive power within
        each signal-strength bucket.

        Parameters
        ----------
        signal          : cross-sectional signal
        forward_returns : corresponding returns
        n_quantiles     : number of buckets

        Returns
        -------
        Dict[quantile_label → IC]
        """
        nq = n_quantiles or self.n_quantiles
        df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1).dropna()
        if len(df) < nq * 3:
            return {}

        df["q"] = pd.qcut(df["sig"], q=nq, labels=False, duplicates="drop")
        result: dict[str, float] = {}
        for q in sorted(df["q"].dropna().unique()):
            bucket = df[df["q"] == q]
            if len(bucket) < 3:
                result[f"Q{int(q)+1}"] = float("nan")
                continue
            r, _ = stats.spearmanr(bucket["sig"], bucket["ret"])
            result[f"Q{int(q)+1}"] = float(r)
        return result

    # ------------------------------------------------------------------ #
    # Turnover per quantile
    # ------------------------------------------------------------------ #

    def quintile_turnover(
        self,
        signal_df: pd.DataFrame,
        n_quantiles: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute signal turnover (fraction entering/leaving each quintile per bar).

        Parameters
        ----------
        signal_df  : DataFrame[time × assets]
        n_quantiles: number of buckets

        Returns
        -------
        Dict[quantile_label → avg_daily_turnover]
        """
        nq = n_quantiles or self.n_quantiles
        # Assign quintile rank at each time step
        ranked = signal_df.rank(axis=1, pct=True)

        # Bin into quintiles
        def _assign_q(row: pd.Series) -> pd.Series:
            return pd.cut(row, bins=nq, labels=False)

        q_df = ranked.apply(_assign_q, axis=1)

        turnover_per_q: dict[str, list[float]] = {f"Q{i+1}": [] for i in range(nq)}

        for t in range(1, len(q_df)):
            prev = q_df.iloc[t - 1]
            curr = q_df.iloc[t]
            for q in range(nq):
                prev_members = set((prev == q).index[prev == q].tolist())
                curr_members = set((curr == q).index[curr == q].tolist())
                if len(prev_members) == 0:
                    turnover_per_q[f"Q{q+1}"].append(0.0)
                    continue
                entered = len(curr_members - prev_members)
                exited = len(prev_members - curr_members)
                to = (entered + exited) / (2 * len(prev_members))
                turnover_per_q[f"Q{q+1}"].append(to)

        return {q: float(np.nanmean(v)) for q, v in turnover_per_q.items()}

    # ------------------------------------------------------------------ #
    # Monotonicity score (standalone)
    # ------------------------------------------------------------------ #

    def factor_monotonicity_score(
        self,
        quintile_returns: QuantileResult,
    ) -> float:
        """Return the monotonicity score of quintile mean returns.

        1.0 = perfectly monotone ascending (Q1 < Q2 < ... < Q_n).
        0.0 = perfectly monotone descending.
        0.5 = random.

        Parameters
        ----------
        quintile_returns : QuantileResult from compute_quintile_returns()

        Returns
        -------
        float in [0, 1]
        """
        return quintile_returns.monotonicity_score

    # ------------------------------------------------------------------ #
    # Hit-rate by quantile
    # ------------------------------------------------------------------ #

    def hit_rate_by_quantile(
        self,
        signal: pd.Series,
        direction: pd.Series,
        n_quantiles: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute fraction of correct directional predictions per quantile.

        Parameters
        ----------
        signal    : numeric signal (positive = long, negative = short)
        direction : actual subsequent direction (+1/-1 or binary 1/0)
        n_quantiles: number of quantile buckets

        Returns
        -------
        Dict[quantile_label → hit_rate]
        """
        nq = n_quantiles or self.n_quantiles
        df = pd.concat({"sig": signal, "dir": direction}, axis=1).dropna()
        if len(df) < nq * 3:
            return {}

        # Convert signal to prediction: sign of signal
        df["pred"] = np.sign(df["sig"])
        df["hit"] = (df["pred"] == df["dir"]).astype(float)
        df["q"] = pd.qcut(df["sig"], q=nq, labels=False, duplicates="drop")

        result: dict[str, float] = {}
        for q in sorted(df["q"].dropna().unique()):
            bucket = df[df["q"] == q]
            result[f"Q{int(q)+1}"] = float(bucket["hit"].mean())
        return result

    # ------------------------------------------------------------------ #
    # Cumulative returns by quantile
    # ------------------------------------------------------------------ #

    def compute_cumulative_quintile_returns(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        n_quantiles: Optional[int] = None,
    ) -> QuintileCumulativeResult:
        """Compute cumulative equity curves for each quintile portfolio.

        At each time step, assets are sorted into quantile buckets;
        each bucket's equal-weighted return is tracked.

        Parameters
        ----------
        signal_df  : DataFrame[time × assets] — signal at time t
        returns_df : DataFrame[time × assets] — 1-bar forward return from t

        Returns
        -------
        QuintileCumulativeResult
        """
        nq = n_quantiles or self.n_quantiles
        cols = signal_df.columns.intersection(returns_df.columns)
        idx = signal_df.index.intersection(returns_df.index)
        sig = signal_df.loc[idx, cols]
        ret = returns_df.loc[idx, cols]

        q_labels = [f"Q{i+1}" for i in range(nq)]
        q_returns_over_time: dict[str, list[float]] = {q: [] for q in q_labels}

        for t in idx:
            s_t = sig.loc[t].dropna()
            r_t = ret.loc[t].dropna()
            common = s_t.index.intersection(r_t.index)
            if len(common) < nq:
                for q in q_labels:
                    q_returns_over_time[q].append(0.0)
                continue

            ranks = s_t.loc[common].rank(pct=True)
            for i in range(nq):
                lo = i / nq
                hi = (i + 1) / nq
                if i == nq - 1:
                    mask = ranks >= lo
                else:
                    mask = (ranks >= lo) & (ranks < hi)
                bucket_ret = r_t.loc[common[mask]]
                q_returns_over_time[q_labels[i]].append(
                    float(bucket_ret.mean()) if len(bucket_ret) > 0 else 0.0
                )

        q_df = pd.DataFrame(q_returns_over_time, index=idx)
        cum_ret = (1 + q_df).cumprod()

        return QuintileCumulativeResult(
            cumulative_returns=cum_ret,
            n_quantiles=nq,
        )

    # ------------------------------------------------------------------ #
    # Sharpe by quantile
    # ------------------------------------------------------------------ #

    def sharpe_by_quantile(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        n_quantiles: Optional[int] = None,
        bars_per_year: int = 252,
    ) -> Dict[str, float]:
        """Annualised Sharpe ratio for each quantile portfolio.

        Parameters
        ----------
        signal_df     : DataFrame[time × assets]
        returns_df    : DataFrame[time × assets]
        n_quantiles   : number of buckets
        bars_per_year : annualisation factor

        Returns
        -------
        Dict[quantile → Sharpe]
        """
        cum_result = self.compute_cumulative_quintile_returns(signal_df, returns_df, n_quantiles)
        nq = n_quantiles or self.n_quantiles
        q_labels = [f"Q{i+1}" for i in range(nq)]

        # Compute period returns from cumulative
        period_ret = cum_result.cumulative_returns.pct_change().dropna()
        sharpes: dict[str, float] = {}
        for q in q_labels:
            if q not in period_ret.columns:
                sharpes[q] = float("nan")
                continue
            r = period_ret[q].dropna()
            if len(r) < 2 or r.std(ddof=1) == 0:
                sharpes[q] = float("nan")
            else:
                sharpes[q] = float(r.mean() / r.std(ddof=1) * np.sqrt(bars_per_year))
        return sharpes

    # ------------------------------------------------------------------ #
    # Visualisations
    # ------------------------------------------------------------------ #

    def plot_quintile_bar(
        self,
        quintile_result: QuantileResult,
        save_path: Optional[str | Path] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Classic bar chart of mean return per quantile.

        Parameters
        ----------
        quintile_result : QuantileResult from compute_quintile_returns()
        save_path       : optional save path
        title           : figure title

        Returns
        -------
        matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        q_labels = quintile_result.quantile_labels
        means = quintile_result.mean_returns
        stds = quintile_result.std_returns
        ns = quintile_result.n_obs

        # Mean return bar chart
        ax = axes[0]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in means]
        bars = ax.bar(q_labels, means, color=colors, alpha=0.8, edgecolor="white")
        yerr = [s / np.sqrt(n) if n > 0 else 0 for s, n in zip(stds, ns)]
        ax.errorbar(
            range(len(q_labels)), means, yerr=yerr,
            fmt="none", color="black", capsize=4, linewidth=1,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Signal Quantile")
        ax.set_ylabel("Mean Forward Return")

        spread_str = (
            f"Spread={quintile_result.spread:.4f}  "
            f"t={quintile_result.spread_t_stat:.2f}  "
            f"p={quintile_result.spread_p_value:.4f}"
        )
        mono_str = f"Monotonicity={quintile_result.monotonicity_score:.2f}"
        if title is not None:
            ax.set_title(f"{title}\n{spread_str}  |  {mono_str}")
        else:
            ax.set_title(f"Quintile Returns: {quintile_result.signal_col}\n{spread_str}  |  {mono_str}")

        # Hit-rate bar chart
        ax2 = axes[1]
        ax2.bar(q_labels, quintile_result.hit_rates, color="#3498db", alpha=0.8, edgecolor="white")
        ax2.axhline(0.5, color="black", linewidth=1, linestyle="--", label="50% base")
        ax2.set_xlabel("Signal Quantile")
        ax2.set_ylabel("Hit Rate (fraction positive)")
        ax2.set_ylim(0, 1)
        ax2.set_title("Hit Rate by Quantile")
        ax2.legend()

        fig.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_quantile_cumulative_returns(
        self,
        quintile_result: QuintileCumulativeResult,
        save_path: Optional[str | Path] = None,
        title: str = "Quantile Cumulative Returns",
    ) -> plt.Figure:
        """Cumulative equity curves per quantile on one plot.

        Parameters
        ----------
        quintile_result : QuintileCumulativeResult from compute_cumulative_quintile_returns()
        save_path       : optional save path
        title           : figure title

        Returns
        -------
        matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        palette = plt.cm.RdYlGn(np.linspace(0.05, 0.95, quintile_result.n_quantiles))

        for i, col in enumerate(quintile_result.cumulative_returns.columns):
            series = quintile_result.cumulative_returns[col]
            ax.plot(series.index, series.values, color=palette[i], linewidth=1.2, label=col)

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (1 = start)")
        ax.set_title(title)
        ax.legend(loc="upper left")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_quintile_summary_grid(
        self,
        quintile_result: QuantileResult,
        cum_result: Optional[QuintileCumulativeResult] = None,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Four-panel summary grid: mean return, hit rate, std, n_obs."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        q_labels = quintile_result.quantile_labels

        # Mean return
        ax = axes[0, 0]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in quintile_result.mean_returns]
        ax.bar(q_labels, quintile_result.mean_returns, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Mean Return per Quintile")
        ax.set_ylabel("Mean Return")

        # Hit rate
        ax2 = axes[0, 1]
        ax2.bar(q_labels, quintile_result.hit_rates, color="#3498db", alpha=0.8)
        ax2.axhline(0.5, color="red", linestyle="--", linewidth=1)
        ax2.set_ylim(0, 1)
        ax2.set_title("Hit Rate per Quintile")
        ax2.set_ylabel("Fraction Positive")

        # Std deviation
        ax3 = axes[1, 0]
        ax3.bar(q_labels, quintile_result.std_returns, color="#9b59b6", alpha=0.8)
        ax3.set_title("Return Std per Quintile")
        ax3.set_ylabel("Std Return")

        # N obs
        ax4 = axes[1, 1]
        ax4.bar(q_labels, quintile_result.n_obs, color="#e67e22", alpha=0.8)
        ax4.set_title("Observations per Quintile")
        ax4.set_ylabel("Count")

        fig.suptitle(
            f"Quintile Analysis: {quintile_result.signal_col}  |  "
            f"Spread={quintile_result.spread:.4f}  |  "
            f"Monotonicity={quintile_result.monotonicity_score:.2f}",
            fontsize=11,
        )
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # Utility: quantile spread time-series
    # ------------------------------------------------------------------ #

    def quantile_spread_timeseries(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        n_quantiles: Optional[int] = None,
    ) -> pd.Series:
        """Compute Q_top − Q_bottom spread return at each time step.

        Parameters
        ----------
        signal_df  : DataFrame[time × assets]
        returns_df : DataFrame[time × assets]
        n_quantiles: number of buckets

        Returns
        -------
        pd.Series[time → spread]
        """
        nq = n_quantiles or self.n_quantiles
        cols = signal_df.columns.intersection(returns_df.columns)
        idx = signal_df.index.intersection(returns_df.index)
        sig = signal_df.loc[idx, cols]
        ret = returns_df.loc[idx, cols]

        spreads: list[float] = []
        for t in idx:
            s_t = sig.loc[t].dropna()
            r_t = ret.loc[t].dropna()
            common = s_t.index.intersection(r_t.index)
            if len(common) < nq * 2:
                spreads.append(float("nan"))
                continue
            ranks = s_t.loc[common].rank(pct=True)
            top = r_t.loc[common[ranks >= (nq - 1) / nq]]
            bot = r_t.loc[common[ranks < 1 / nq]]
            if len(top) > 0 and len(bot) > 0:
                spreads.append(float(top.mean() - bot.mean()))
            else:
                spreads.append(float("nan"))

        return pd.Series(spreads, index=idx, name="q_spread")

    # ------------------------------------------------------------------ #
    # Double-sort quantile analysis
    # ------------------------------------------------------------------ #

    def double_sort_quintile_returns(
        self,
        signal1: pd.Series,
        signal2: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """Double-sort by two signals: compute mean return in each (Q1,Q2) cell.

        Parameters
        ----------
        signal1         : first sort signal
        signal2         : second sort signal
        forward_returns : forward returns
        n_quantiles     : number of buckets per signal

        Returns
        -------
        pd.DataFrame[Q1_bucket x Q2_bucket] of mean returns
        """
        df = pd.concat({"s1": signal1, "s2": signal2, "ret": forward_returns}, axis=1).dropna()
        if len(df) < n_quantiles ** 2 * 3:
            raise ValueError(f"Insufficient observations for {n_quantiles}x{n_quantiles} double sort.")

        df["q1"] = pd.qcut(df["s1"], q=n_quantiles, labels=False, duplicates="drop")
        df["q2"] = pd.qcut(df["s2"], q=n_quantiles, labels=False, duplicates="drop")

        table = df.groupby(["q1", "q2"])["ret"].mean().unstack()
        table.index = [f"Q1_{int(i)+1}" for i in table.index]
        table.columns = [f"Q2_{int(j)+1}" for j in table.columns]
        return table

    # ------------------------------------------------------------------ #
    # Information ratio by quantile
    # ------------------------------------------------------------------ #

    def information_ratio_by_quantile(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: Optional[int] = None,
        bars_per_year: int = 252,
    ) -> Dict[str, float]:
        """Annualised information ratio (mean_ret / std_ret * sqrt(N)) per quantile.

        Parameters
        ----------
        signal          : signal values
        forward_returns : corresponding returns
        n_quantiles     : number of buckets
        bars_per_year   : annualisation factor

        Returns
        -------
        Dict[quantile_label -> annualised_IR]
        """
        nq = n_quantiles or self.n_quantiles
        df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1).dropna()
        if len(df) < nq * 3:
            return {}

        df["q"] = pd.qcut(df["sig"], q=nq, labels=False, duplicates="drop")
        result: dict[str, float] = {}
        for q in sorted(df["q"].dropna().unique()):
            bucket_ret = df.loc[df["q"] == q, "ret"].values
            if len(bucket_ret) < 2 or np.std(bucket_ret, ddof=1) == 0:
                result[f"Q{int(q)+1}"] = float("nan")
            else:
                ir = np.mean(bucket_ret) / np.std(bucket_ret, ddof=1) * np.sqrt(bars_per_year)
                result[f"Q{int(q)+1}"] = float(ir)
        return result

    # ------------------------------------------------------------------ #
    # Long-short portfolio stats
    # ------------------------------------------------------------------ #

    def long_short_portfolio_stats(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: Optional[int] = None,
        bars_per_year: int = 252,
    ) -> Dict[str, float]:
        """Statistics for a long top-quintile / short bottom-quintile portfolio.

        Parameters
        ----------
        signal          : signal values
        forward_returns : forward returns
        n_quantiles     : number of quantile buckets
        bars_per_year   : annualisation factor for Sharpe

        Returns
        -------
        Dict with keys: mean_spread, std_spread, sharpe, hit_rate, n_obs
        """
        nq = n_quantiles or self.n_quantiles
        df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1).dropna()
        if len(df) < nq * 3:
            return {}

        df["q"] = pd.qcut(df["sig"], q=nq, labels=False, duplicates="drop")
        q_max = int(df["q"].max())
        q_min = int(df["q"].min())

        top = df[df["q"] == q_max]["ret"].values
        bot = df[df["q"] == q_min]["ret"].values

        if len(top) == 0 or len(bot) == 0:
            return {}

        # Long-short daily returns (assuming equal allocation)
        n_pairs = min(len(top), len(bot))
        ls_ret = top[:n_pairs] - bot[:n_pairs]

        mean_sp = float(np.mean(ls_ret))
        std_sp = float(np.std(ls_ret, ddof=1))
        sharpe = (mean_sp / std_sp * np.sqrt(bars_per_year)) if std_sp > 0 else float("nan")
        hit_rate = float(np.mean(ls_ret > 0))

        return {
            "mean_spread": float(np.mean(top) - np.mean(bot)),
            "std_spread": std_sp,
            "sharpe": sharpe,
            "hit_rate": hit_rate,
            "n_pairs": n_pairs,
            "long_mean": float(np.mean(top)),
            "short_mean": float(np.mean(bot)),
        }

    # ------------------------------------------------------------------ #
    # Quantile transition matrix
    # ------------------------------------------------------------------ #

    def quantile_transition_matrix(
        self,
        signal_df: pd.DataFrame,
        n_quantiles: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute quantile transition matrix (from Q_t to Q_{t+1}).

        Shows how often assets migrate between quantiles from one period to
        the next — a measure of signal persistence.

        Parameters
        ----------
        signal_df  : DataFrame[time x assets]
        n_quantiles: number of buckets

        Returns
        -------
        pd.DataFrame[Q_from x Q_to] — transition probability matrix
        """
        nq = n_quantiles or self.n_quantiles
        transitions: np.ndarray = np.zeros((nq, nq))

        for t in range(1, len(signal_df)):
            prev = signal_df.iloc[t - 1].dropna()
            curr = signal_df.iloc[t].dropna()
            common = prev.index.intersection(curr.index)
            if len(common) < nq:
                continue
            prev_q = pd.qcut(prev.loc[common], q=nq, labels=False, duplicates="drop")
            curr_q = pd.qcut(curr.loc[common], q=nq, labels=False, duplicates="drop")
            for asset in common:
                pq = prev_q.get(asset, float("nan"))
                cq = curr_q.get(asset, float("nan"))
                if not (np.isnan(pq) or np.isnan(cq)):
                    transitions[int(pq), int(cq)] += 1

        # Normalise rows
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        trans_prob = transitions / row_sums

        labels = [f"Q{i+1}" for i in range(nq)]
        return pd.DataFrame(trans_prob, index=[f"From Q{i+1}" for i in range(nq)],
                            columns=[f"To Q{i+1}" for i in range(nq)])

    # ------------------------------------------------------------------ #
    # Conditional quintile analysis (conditioned on regime)
    # ------------------------------------------------------------------ #

    def quintile_returns_by_regime(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        regime: pd.Series,
        n_quantiles: Optional[int] = None,
    ) -> Dict[str, QuantileResult]:
        """Compute quintile returns separately for each regime label.

        Parameters
        ----------
        signal          : signal values
        forward_returns : forward returns
        regime          : regime labels (same index as signal)
        n_quantiles     : number of quantile buckets

        Returns
        -------
        Dict[regime_label -> QuantileResult]
        """
        df = pd.concat({"sig": signal, "ret": forward_returns, "reg": regime}, axis=1).dropna()
        result: dict[str, QuantileResult] = {}

        for reg_label in df["reg"].unique():
            sub = df[df["reg"] == reg_label]
            if len(sub) < (n_quantiles or self.n_quantiles) * 3:
                continue
            try:
                qr = self.compute_quintile_returns(
                    sub["sig"], sub["ret"],
                    n_quantiles=n_quantiles,
                    signal_col=f"{signal.name}_{reg_label}",
                )
                result[str(reg_label)] = qr
            except Exception:
                continue
        return result

    # ------------------------------------------------------------------ #
    # Plot transition matrix
    # ------------------------------------------------------------------ #

    def plot_transition_matrix(
        self,
        trans_matrix: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Heatmap of the quantile transition probability matrix.

        Parameters
        ----------
        trans_matrix : output of quantile_transition_matrix()
        save_path    : optional save path
        """
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            trans_matrix, ax=ax, vmin=0, vmax=1,
            cmap="Blues", annot=True, fmt=".2f",
            linewidths=0.5, cbar_kws={"label": "Transition Probability"},
        )
        ax.set_title("Quantile Transition Matrix")
        ax.set_xlabel("To Quantile (t+1)")
        ax.set_ylabel("From Quantile (t)")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # Spread return distribution
    # ------------------------------------------------------------------ #

    def plot_spread_distribution(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: Optional[int] = None,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Plot distribution of Q_top and Q_bottom returns on the same axis.

        Parameters
        ----------
        signal          : signal values
        forward_returns : forward returns
        n_quantiles     : number of buckets
        save_path       : optional save path
        """
        nq = n_quantiles or self.n_quantiles
        df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1).dropna()
        df["q"] = pd.qcut(df["sig"], q=nq, labels=False, duplicates="drop")

        q_max = int(df["q"].max())
        q_min = int(df["q"].min())

        top_ret = df[df["q"] == q_max]["ret"].values
        bot_ret = df[df["q"] == q_min]["ret"].values

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(top_ret, bins=30, density=True, alpha=0.6, color="#2ecc71",
                label=f"Q{nq} (Top)  mean={np.mean(top_ret):.4f}")
        ax.hist(bot_ret, bins=30, density=True, alpha=0.6, color="#e74c3c",
                label=f"Q1 (Bottom)  mean={np.mean(bot_ret):.4f}")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Forward Return")
        ax.set_ylabel("Density")
        ax.set_title(f"Return Distributions: Q{nq} vs Q1")
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------
    # Extended quintile diagnostics
    # ------------------------------------------------------------------

    def quintile_drawdown_profile(
        self,
        signal: "pd.Series",
        forward_returns: "pd.Series",
        n_quantiles: int = 5,
    ) -> "pd.DataFrame":
        """
        Compute maximum drawdown for each quantile portfolio assuming a
        simple buy-and-hold of the quintile bucket.

        Parameters
        ----------
        signal : pd.Series
            Signal values aligned with forward_returns.
        forward_returns : pd.Series
            Per-bar forward returns.
        n_quantiles : int
            Number of quantile bins.

        Returns
        -------
        pd.DataFrame with columns [quantile, mean_return, vol, sharpe,
                                    max_drawdown, calmar, skewness, kurtosis].
        """
        import numpy as np
        import pandas as pd

        df = pd.DataFrame({"signal": signal, "ret": forward_returns}).dropna()
        df["quantile"] = pd.qcut(df["signal"], q=n_quantiles,
                                  labels=[f"Q{i+1}" for i in range(n_quantiles)],
                                  duplicates="drop")

        rows: list[dict] = []
        for q_label, grp in df.groupby("quantile", observed=False):
            ret = grp["ret"].values.astype(float)
            if len(ret) < 2:
                continue
            cumret = np.cumprod(1.0 + ret) - 1.0
            running_max = np.maximum.accumulate(cumret + 1.0)
            dd = (cumret + 1.0) / running_max - 1.0
            max_dd = float(dd.min())

            mean_r = float(np.mean(ret))
            std_r = float(np.std(ret, ddof=1))
            sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else np.nan
            calmar = abs(mean_r * 252) / abs(max_dd) if max_dd != 0 else np.nan

            from scipy import stats as _stats
            sk = float(_stats.skew(ret))
            ku = float(_stats.kurtosis(ret))

            rows.append({
                "quantile": str(q_label),
                "mean_return": mean_r,
                "vol": std_r,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "calmar": calmar,
                "skewness": sk,
                "kurtosis": ku,
                "n_obs": len(ret),
            })

        return pd.DataFrame(rows)

    def quintile_significance_test(
        self,
        signal: "pd.Series",
        forward_returns: "pd.Series",
        n_quantiles: int = 5,
    ) -> "pd.DataFrame":
        """
        Test whether each quantile's mean return is statistically different
        from zero using a two-sided t-test, and whether adjacent quantile
        means are statistically different (pairwise t-test).

        Parameters
        ----------
        signal : pd.Series
            Signal values.
        forward_returns : pd.Series
            Forward return values.
        n_quantiles : int
            Number of quantile bins.

        Returns
        -------
        pd.DataFrame with columns [quantile, mean_return, t_stat, p_value,
                                    significant_vs_zero, vs_next_q_p_value].
        """
        import pandas as pd
        from scipy import stats as _stats

        df = pd.DataFrame({"signal": signal, "ret": forward_returns}).dropna()
        df["quantile"] = pd.qcut(df["signal"], q=n_quantiles,
                                  labels=[f"Q{i+1}" for i in range(n_quantiles)],
                                  duplicates="drop")

        quantile_rets: dict[str, list[float]] = {}
        for q_label, grp in df.groupby("quantile", observed=False):
            quantile_rets[str(q_label)] = grp["ret"].tolist()

        rows: list[dict] = []
        q_labels = list(quantile_rets.keys())
        for i, q_label in enumerate(q_labels):
            ret = quantile_rets[q_label]
            if len(ret) < 3:
                continue
            t_stat, p_val = _stats.ttest_1samp(ret, 0.0)
            sig = p_val < 0.05

            # Compare with next quantile
            next_pval = np.nan
            if i + 1 < len(q_labels):
                next_ret = quantile_rets[q_labels[i + 1]]
                if len(next_ret) >= 3:
                    _, next_pval = _stats.ttest_ind(ret, next_ret, equal_var=False)

            rows.append({
                "quantile": q_label,
                "mean_return": float(np.mean(ret)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant_vs_zero": bool(sig),
                "vs_next_q_p_value": float(next_pval),
                "n_obs": len(ret),
            })

        return pd.DataFrame(rows)

    def quintile_regime_consistency(
        self,
        signal: "pd.Series",
        forward_returns: "pd.Series",
        regime_labels: "pd.Series",
        n_quantiles: int = 5,
    ) -> "pd.DataFrame":
        """
        Test whether the Q5-Q1 spread is consistent across regimes.
        Computes the spread (top minus bottom quintile mean return) within
        each regime and returns a summary.

        Parameters
        ----------
        signal : pd.Series
            Signal values.
        forward_returns : pd.Series
            Forward returns.
        regime_labels : pd.Series
            Discrete regime labels aligned with signal/returns.
        n_quantiles : int
            Number of quantile bins.

        Returns
        -------
        pd.DataFrame with columns [regime, q_top_mean, q_bot_mean, spread,
                                    n_obs, spread_t_stat].
        """
        import pandas as pd
        from scipy import stats as _stats

        df = pd.DataFrame({
            "signal": signal,
            "ret": forward_returns,
            "regime": regime_labels,
        }).dropna()

        rows: list[dict] = []
        for regime_name, grp in df.groupby("regime"):
            if len(grp) < n_quantiles * 5:
                continue
            try:
                grp["quantile"] = pd.qcut(
                    grp["signal"], q=n_quantiles,
                    labels=[f"Q{i+1}" for i in range(n_quantiles)],
                    duplicates="drop",
                )
            except Exception:
                continue

            q_groups = grp.groupby("quantile", observed=False)["ret"]
            q_means = q_groups.mean()
            q_labels = q_means.index.tolist()
            if len(q_labels) < 2:
                continue

            top_ret = grp[grp["quantile"] == q_labels[-1]]["ret"].values
            bot_ret = grp[grp["quantile"] == q_labels[0]]["ret"].values

            if len(top_ret) < 3 or len(bot_ret) < 3:
                continue

            spread = float(top_ret.mean() - bot_ret.mean())
            # t-stat of spread via Welch test
            _, pval = _stats.ttest_ind(top_ret, bot_ret, equal_var=False)
            t_stat = spread / (np.sqrt(top_ret.var(ddof=1) / len(top_ret) +
                                       bot_ret.var(ddof=1) / len(bot_ret))) if len(top_ret) > 1 else np.nan

            rows.append({
                "regime": str(regime_name),
                "q_top_mean": float(top_ret.mean()),
                "q_bot_mean": float(bot_ret.mean()),
                "spread": spread,
                "n_obs": len(grp),
                "spread_t_stat": t_stat,
                "spread_p_value": float(pval),
            })

        return pd.DataFrame(rows)
