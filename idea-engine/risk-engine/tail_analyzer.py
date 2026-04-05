"""
risk-engine/tail_analyzer.py

Tail risk analysis: extreme-event detection, tail-ratio metrics,
risk-adjusted performance ratios, tail dependence, and crisis alpha.

All methods return plain Python floats or pandas DataFrames so results
are easy to pass downstream into the hypothesis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TailEvent:
    """
    A single extreme return event with surrounding context.

    Attributes
    ----------
    timestamp : str
        ISO-8601 timestamp of the extreme observation.
    return_value : float
        The raw return on this bar.
    z_score : float
        How many standard deviations from the mean.
    pct_rank : float
        Empirical percentile rank (0 = worst).
    pre_context : list[float]
        Returns for the ``context_window`` bars preceding the event.
    post_context : list[float]
        Returns for the ``context_window`` bars following the event.
    regime : str
        Optional regime label at the time of the event.
    """

    timestamp: str
    return_value: float
    z_score: float
    pct_rank: float
    pre_context: list[float] = field(default_factory=list)
    post_context: list[float] = field(default_factory=list)
    regime: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "return_value": self.return_value,
            "z_score": self.z_score,
            "pct_rank": self.pct_rank,
            "pre_context": self.pre_context,
            "post_context": self.post_context,
            "regime": self.regime,
        }


# ---------------------------------------------------------------------------
# Core TailAnalyzer
# ---------------------------------------------------------------------------


class TailAnalyzer:
    """
    Tail risk and risk-adjusted performance metrics.

    Parameters
    ----------
    context_window : int
        Number of bars before/after an extreme event to include as context
        when calling :meth:`extreme_events`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> returns = pd.Series(rng.normal(0.0003, 0.013, 500))
    >>> equity  = (1 + returns).cumprod()
    >>> ta = TailAnalyzer()
    >>> print(f"Sortino: {ta.sortino_ratio(returns):.3f}")
    >>> print(f"Calmar:  {ta.calmar_ratio(equity):.3f}")
    """

    def __init__(self, context_window: int = 5) -> None:
        self.context_window = context_window

    # ------------------------------------------------------------------
    # Risk-adjusted performance ratios
    # ------------------------------------------------------------------

    def tail_ratio(self, returns: pd.Series) -> float:
        """
        Tail ratio: 95th-percentile gain divided by the absolute value of
        the 5th-percentile loss.

        A value > 1 indicates that large gains are bigger than large losses,
        suggesting positive skew in the tails.

        Parameters
        ----------
        returns : pd.Series
            Return series.

        Returns
        -------
        float
            Tail ratio.  Returns inf when the loss tail is zero.
        """
        arr = returns.dropna().values
        p95 = float(np.percentile(arr, 95))
        p05 = float(np.percentile(arr, 5))
        if p05 >= 0:
            return np.inf
        return float(p95 / abs(p05))

    def omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Omega ratio: probability-weighted ratio of gains above the threshold
        to losses below it.

        Ω(L) = ∫_{L}^{∞} [1 - F(r)] dr  /  ∫_{-∞}^{L} F(r) dr

        Approximated as the mean of gains above L divided by mean of losses below L.

        Parameters
        ----------
        returns : pd.Series
            Return series.
        threshold : float
            Minimum acceptable return (MAR).

        Returns
        -------
        float
            Omega ratio.
        """
        arr = returns.dropna().values
        gains = arr[arr > threshold] - threshold
        losses = threshold - arr[arr <= threshold]

        if losses.sum() == 0:
            return np.inf
        return float(gains.sum() / losses.sum())

    def sortino_ratio(
        self,
        returns: pd.Series,
        target: float = 0.0,
        annualisation: int = 252,
    ) -> float:
        """
        Sortino ratio using downside deviation as the risk denominator.

        S = (mean_return - target) / downside_deviation * sqrt(annualisation)

        Parameters
        ----------
        returns : pd.Series
            Return series (per-bar, not annualised).
        target : float
            Minimum acceptable return per bar.
        annualisation : int
            Number of bars per year for annualisation.

        Returns
        -------
        float
            Annualised Sortino ratio.
        """
        arr = returns.dropna().values
        excess = arr - target
        downside = excess[excess < 0]
        if len(downside) == 0:
            return np.inf
        downside_std = float(np.sqrt(np.mean(downside**2)))
        if downside_std == 0:
            return np.inf
        mean_excess = float(np.mean(excess))
        return float(mean_excess / downside_std * np.sqrt(annualisation))

    def calmar_ratio(self, equity_curve: pd.Series) -> float:
        """
        Calmar ratio: annualised CAGR divided by maximum drawdown.

        Parameters
        ----------
        equity_curve : pd.Series
            Cumulative equity (e.g. starting at 1.0), indexed by date.

        Returns
        -------
        float
            Calmar ratio.  Returns inf when max drawdown is zero.
        """
        arr = equity_curve.dropna().values
        if len(arr) < 2:
            return 0.0

        total_return = arr[-1] / arr[0] - 1.0
        n_bars = len(arr)
        # Estimate CAGR assuming 252 bars per year
        cagr = (arr[-1] / arr[0]) ** (252 / n_bars) - 1.0

        max_dd = self._max_drawdown(arr)
        if max_dd == 0:
            return np.inf
        return float(cagr / max_dd)

    def pain_ratio(self, equity_curve: pd.Series) -> float:
        """
        Pain ratio: CAGR divided by the Pain Index (average drawdown from peak).

        The Pain Index is the average depth of all underwater periods, giving
        a measure of the investor's average "pain" throughout the series.

        Parameters
        ----------
        equity_curve : pd.Series
            Cumulative equity indexed by date.

        Returns
        -------
        float
            Pain ratio.
        """
        arr = equity_curve.dropna().values
        if len(arr) < 2:
            return 0.0

        rolling_peak = np.maximum.accumulate(arr)
        drawdowns = (rolling_peak - arr) / rolling_peak
        pain_index = float(np.mean(drawdowns))

        if pain_index == 0:
            return np.inf

        cagr = (arr[-1] / arr[0]) ** (252 / len(arr)) - 1.0
        return float(cagr / pain_index)

    def ulcer_index(self, equity_curve: pd.Series) -> float:
        """
        Ulcer Index: RMS of percentage drawdowns from rolling peak.

        UI = sqrt( mean( drawdown_pct² ) )

        A lower Ulcer Index indicates smoother equity growth.

        Parameters
        ----------
        equity_curve : pd.Series
            Cumulative equity.

        Returns
        -------
        float
            Ulcer Index (0 to 1 scale, where 0 = no drawdown).
        """
        arr = equity_curve.dropna().values
        if len(arr) < 2:
            return 0.0
        rolling_peak = np.maximum.accumulate(arr)
        drawdown_pct = (arr - rolling_peak) / rolling_peak  # negative values
        ulcer = float(np.sqrt(np.mean(drawdown_pct**2)))
        return ulcer

    def martin_ratio(
        self, equity_curve: pd.Series, rf: float = 0.0, annualisation: int = 252
    ) -> float:
        """
        Martin ratio (Ulcer Performance Index): annualised excess return
        divided by Ulcer Index.

        Parameters
        ----------
        equity_curve : pd.Series
            Cumulative equity.
        rf : float
            Risk-free rate per bar.
        annualisation : int
            Bars per year.

        Returns
        -------
        float
            Martin ratio.
        """
        arr = equity_curve.dropna().values
        returns = np.diff(arr) / arr[:-1]
        excess = float(np.mean(returns) - rf)
        ui = self.ulcer_index(equity_curve)
        if ui == 0:
            return np.inf
        return float(excess * np.sqrt(annualisation) / ui)

    # ------------------------------------------------------------------
    # Tail event detection
    # ------------------------------------------------------------------

    def extreme_events(
        self,
        returns: pd.Series,
        threshold: float = 3.0,
        regime_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Identify extreme return events — observations beyond ``threshold``
        standard deviations from the mean — and collect surrounding context.

        Parameters
        ----------
        returns : pd.Series
            Return series.  A DatetimeIndex produces useful timestamps.
        threshold : float
            Z-score magnitude cutoff.  Only the left (loss) tail is flagged
            by default; pass a negative value to capture gains instead.
        regime_series : pd.Series, optional
            Regime labels aligned with ``returns``.

        Returns
        -------
        pd.DataFrame
            One row per tail event with columns: timestamp, return_value,
            z_score, pct_rank, pre_context, post_context, regime.
        """
        arr = returns.dropna().values
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))

        if sigma == 0:
            return pd.DataFrame()

        z_scores = (arr - mu) / sigma
        # Only flag negative extremes (large losses)
        mask = z_scores < -abs(threshold)
        indices = np.where(mask)[0]

        idx = returns.dropna().index
        has_regime = regime_series is not None

        events: list[dict[str, Any]] = []
        for i in indices:
            pre_start = max(0, i - self.context_window)
            post_end = min(len(arr), i + self.context_window + 1)

            ts = (
                str(idx[i].isoformat()) if hasattr(idx[i], "isoformat") else str(idx[i])
            )
            regime = (
                str(regime_series.iloc[i])
                if has_regime and i < len(regime_series)
                else "unknown"
            )
            pct_rank = float(stats.percentileofscore(arr, arr[i]) / 100)

            events.append(
                TailEvent(
                    timestamp=ts,
                    return_value=float(arr[i]),
                    z_score=float(z_scores[i]),
                    pct_rank=pct_rank,
                    pre_context=arr[pre_start:i].tolist(),
                    post_context=arr[i + 1 : post_end].tolist(),
                    regime=regime,
                ).to_dict()
            )

        if not events:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "return_value",
                    "z_score",
                    "pct_rank",
                    "pre_context",
                    "post_context",
                    "regime",
                ]
            )
        return pd.DataFrame(events)

    # ------------------------------------------------------------------
    # Tail dependence
    # ------------------------------------------------------------------

    def tail_dependence_matrix(
        self,
        returns_df: pd.DataFrame,
        quantile: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compute pairwise lower and upper tail dependence coefficients.

        The empirical lower tail dependence coefficient between assets i and j
        is estimated as:

            λ_L(i,j) = P(X_i ≤ F_i⁻¹(q)  |  X_j ≤ F_j⁻¹(q))

        and similarly for the upper tail (using 1-q).

        Parameters
        ----------
        returns_df : pd.DataFrame
            DataFrame where each column is an asset return series.
        quantile : float
            Tail quantile to use (default 0.05 = lower 5 %).

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame with levels ('lower', 'upper') providing
            separate coefficient matrices.  Access via
            ``result.loc['lower']`` and ``result.loc['upper']``.
        """
        cols = list(returns_df.columns)
        n = len(cols)
        clean = returns_df.dropna()

        lower = np.zeros((n, n))
        upper = np.zeros((n, n))

        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                if i == j:
                    lower[i, j] = 1.0
                    upper[i, j] = 1.0
                    continue

                xi = clean[ci].values
                xj = clean[cj].values

                qi_lower = np.quantile(xi, quantile)
                qj_lower = np.quantile(xj, quantile)
                qi_upper = np.quantile(xi, 1 - quantile)
                qj_upper = np.quantile(xj, 1 - quantile)

                joint_lower = np.mean((xi <= qi_lower) & (xj <= qj_lower))
                marginal_j_lower = np.mean(xj <= qj_lower)
                lower[i, j] = (
                    joint_lower / marginal_j_lower if marginal_j_lower > 0 else 0.0
                )

                joint_upper = np.mean((xi >= qi_upper) & (xj >= qj_upper))
                marginal_j_upper = np.mean(xj >= qj_upper)
                upper[i, j] = (
                    joint_upper / marginal_j_upper if marginal_j_upper > 0 else 0.0
                )

        lower_df = pd.DataFrame(lower, index=cols, columns=cols)
        upper_df = pd.DataFrame(upper, index=cols, columns=cols)

        result = pd.concat(
            [lower_df, upper_df],
            keys=["lower", "upper"],
        )
        return result

    # ------------------------------------------------------------------
    # Crisis alpha
    # ------------------------------------------------------------------

    def crisis_alpha(
        self,
        returns: pd.Series,
        crisis_periods: list[tuple[str, str]],
        benchmark: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Measure strategy performance conditional on market-stress periods.

        Parameters
        ----------
        returns : pd.Series
            Strategy daily returns, DatetimeIndex required.
        crisis_periods : list of (start, end) tuples
            ISO-8601 date strings defining stress windows, e.g.
            ``[("2020-02-20", "2020-03-23"), ("2008-09-15", "2009-03-09")]``.
        benchmark : pd.Series, optional
            Benchmark returns aligned with strategy returns.  When provided,
            alpha is computed as excess return over the benchmark.

        Returns
        -------
        pd.DataFrame
            Columns: period_start, period_end, strategy_return, benchmark_return,
            alpha, sharpe_in_crisis, n_bars.
        """
        rows: list[dict[str, Any]] = []

        for start_str, end_str in crisis_periods:
            try:
                mask = (returns.index >= start_str) & (returns.index <= end_str)
                slice_ = returns[mask]
                if len(slice_) == 0:
                    continue

                strat_ret = float((1 + slice_).prod() - 1)

                if benchmark is not None:
                    bm_slice = benchmark[mask]
                    bm_ret = float((1 + bm_slice).prod() - 1)
                else:
                    bm_ret = np.nan

                alpha = strat_ret - (bm_ret if not np.isnan(bm_ret) else 0.0)
                sigma_crisis = float(slice_.std()) * np.sqrt(252)
                sharpe_crisis = (
                    float(slice_.mean() * 252 / sigma_crisis) if sigma_crisis > 0 else 0.0
                )

                rows.append(
                    {
                        "period_start": start_str,
                        "period_end": end_str,
                        "strategy_return": strat_ret,
                        "benchmark_return": bm_ret,
                        "alpha": alpha,
                        "sharpe_in_crisis": sharpe_crisis,
                        "n_bars": len(slice_),
                    }
                )
            except Exception:
                continue

        if not rows:
            return pd.DataFrame(
                columns=[
                    "period_start",
                    "period_end",
                    "strategy_return",
                    "benchmark_return",
                    "alpha",
                    "sharpe_in_crisis",
                    "n_bars",
                ]
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(arr: np.ndarray) -> float:
        """Maximum drawdown from peak as a positive fraction."""
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / peak
        return float(np.max(dd))
