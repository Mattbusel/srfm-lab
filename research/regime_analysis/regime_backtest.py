"""
Regime-conditional backtesting utilities.

Implements:
- Slice a backtest by detected regime
- Regime-conditional performance attribution
- Regime transition heatmap
- Drawdown analysis per regime
- Strategy switching based on regime classification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared stats helper
# ---------------------------------------------------------------------------

def _stats(returns: pd.Series, freq: int = 252) -> Dict:
    r = returns.dropna()
    if len(r) == 0:
        return {k: np.nan for k in [
            "total_return", "cagr", "sharpe", "sortino",
            "max_drawdown", "calmar", "win_rate",
        ]}
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    n_years = len(r) / freq
    cagr = float((1 + total) ** (1 / max(n_years, 1e-6)) - 1)
    sr = float(r.mean() / (r.std() + 1e-12) * np.sqrt(freq))
    neg = r[r < 0]
    sortino = float(r.mean() / (neg.std() + 1e-12) * np.sqrt(freq))
    roll_max = eq.cummax()
    dd = (eq - roll_max) / (roll_max + 1e-12)
    mdd = float(dd.min())
    calmar = cagr / (abs(mdd) + 1e-12)
    wins = (r > 0).sum()
    losses = (r < 0).sum()
    win_rate = float(wins / max(wins + losses, 1))
    return {
        "total_return": total, "cagr": cagr, "sharpe": sr, "sortino": sortino,
        "max_drawdown": mdd, "calmar": calmar, "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# Regime Backtest Slicer
# ---------------------------------------------------------------------------

class RegimeBacktest:
    """
    Slice a strategy backtest by regime to evaluate conditional performance.

    Parameters
    ----------
    freq : int
        Number of trading periods per year (252 for daily).
    """

    def __init__(self, freq: int = 252) -> None:
        self.freq = freq

    def slice_by_regime(
        self,
        strategy_returns: pd.Series,
        regime_series: pd.Series,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split strategy returns by regime.

        Parameters
        ----------
        strategy_returns : pd.Series
            Daily strategy return series.
        regime_series : pd.Series
            Regime labels aligned to same index.

        Returns
        -------
        dict mapping regime_label -> pd.Series of returns during that regime.
        """
        combined = pd.concat(
            [strategy_returns.rename("ret"), regime_series.rename("regime")],
            axis=1,
        ).dropna()

        slices = {}
        for regime in sorted(combined["regime"].unique()):
            slices[str(regime)] = combined[combined["regime"] == regime]["ret"]

        return slices

    def conditional_performance(
        self,
        strategy_returns: pd.Series,
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute performance statistics for each regime.

        Returns
        -------
        pd.DataFrame
            Performance statistics (rows = regimes).
        """
        slices = self.slice_by_regime(strategy_returns, regime_series)
        rows = []
        for regime, ret_slice in slices.items():
            s = _stats(ret_slice, self.freq)
            rows.append({
                "regime": regime,
                "n_obs": len(ret_slice),
                "freq_pct": round(len(ret_slice) / len(strategy_returns.dropna()), 4),
                **{k: round(v, 4) for k, v in s.items()},
            })
        return pd.DataFrame(rows).set_index("regime")

    def regime_equity_curves(
        self,
        strategy_returns: pd.Series,
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute a synthetic equity curve for each regime by concatenating
        returns only during that regime.

        Returns
        -------
        pd.DataFrame
            Equity curves per regime (columns), NaN outside their regime.
        """
        combined = pd.concat(
            [strategy_returns.rename("ret"), regime_series.rename("regime")],
            axis=1,
        ).dropna()

        regimes = sorted(combined["regime"].unique())
        eq_df = pd.DataFrame(np.nan, index=combined.index,
                             columns=[str(r) for r in regimes])

        for regime in regimes:
            mask = combined["regime"] == regime
            regime_rets = combined.loc[mask, "ret"]
            eq = (1 + regime_rets).cumprod()
            eq_df.loc[mask, str(regime)] = eq.values

        return eq_df

    def drawdown_by_regime(
        self,
        strategy_returns: pd.Series,
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute max drawdown statistics per regime.

        Returns
        -------
        pd.DataFrame
            Regime, max_drawdown, avg_drawdown, n_drawdowns, avg_recovery_days.
        """
        slices = self.slice_by_regime(strategy_returns, regime_series)
        rows = []
        for regime, ret_slice in slices.items():
            if len(ret_slice) < 2:
                continue
            eq = (1 + ret_slice).cumprod()
            roll_max = eq.cummax()
            dd = (eq - roll_max) / (roll_max + 1e-12)

            # Identify drawdown episodes
            in_dd = dd < 0
            dd_starts = []
            dd_depths = []
            recovery_days = []
            in_episode = False
            start_idx = None
            max_dd_in_ep = 0.0

            for i, (dt, val) in enumerate(dd.items()):
                if val < 0 and not in_episode:
                    in_episode = True
                    start_idx = i
                    max_dd_in_ep = val
                elif val < 0 and in_episode:
                    max_dd_in_ep = min(max_dd_in_ep, val)
                elif val >= 0 and in_episode:
                    in_episode = False
                    dd_starts.append(start_idx)
                    dd_depths.append(max_dd_in_ep)
                    recovery_days.append(i - start_idx)
                    max_dd_in_ep = 0.0

            if in_episode:
                dd_starts.append(start_idx)
                dd_depths.append(max_dd_in_ep)

            rows.append({
                "regime": regime,
                "max_drawdown": round(float(dd.min()), 4),
                "avg_drawdown": round(float(np.mean(dd_depths)) if dd_depths else 0.0, 4),
                "n_drawdown_episodes": len(dd_depths),
                "avg_recovery_days": round(float(np.mean(recovery_days)) if recovery_days else 0.0, 2),
                "time_in_drawdown_pct": round(float(in_dd.mean()), 4),
            })

        return pd.DataFrame(rows).set_index("regime")

    def transition_heatmap(
        self,
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute the empirical transition frequency matrix between regimes.

        Element [i, j] = fraction of times regime i is followed by regime j.

        Returns
        -------
        pd.DataFrame
            (n_regimes x n_regimes) transition frequency matrix.
        """
        clean = regime_series.dropna()
        regimes = sorted(clean.unique())
        n = len(regimes)
        regime_to_idx = {r: i for i, r in enumerate(regimes)}

        counts = np.zeros((n, n))
        prev = None
        for r in clean:
            if prev is not None:
                counts[regime_to_idx[prev], regime_to_idx[r]] += 1
            prev = r

        # Row-normalize
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        freq = counts / row_sums

        regime_names = [str(r) for r in regimes]
        return pd.DataFrame(
            freq.round(4),
            index=[f"from_{r}" for r in regime_names],
            columns=[f"to_{r}" for r in regime_names],
        )

    def regime_duration_stats(self, regime_series: pd.Series) -> pd.DataFrame:
        """
        Statistics on run lengths (duration) per regime.

        Returns
        -------
        pd.DataFrame
            Regime, mean_duration, std_duration, max_duration, n_episodes.
        """
        clean = regime_series.dropna()
        regimes = sorted(clean.unique())
        rows = []
        for regime in regimes:
            mask = clean == regime
            runs = []
            current_run = 0
            for v in mask:
                if v:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                        current_run = 0
            if current_run > 0:
                runs.append(current_run)

            if runs:
                rows.append({
                    "regime": str(regime),
                    "n_episodes": len(runs),
                    "mean_duration": round(float(np.mean(runs)), 2),
                    "std_duration": round(float(np.std(runs)), 2),
                    "max_duration": int(np.max(runs)),
                    "min_duration": int(np.min(runs)),
                    "total_periods": int(np.sum(runs)),
                    "freq_pct": round(float(mask.mean()), 4),
                })

        return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Regime-Switching Strategy
# ---------------------------------------------------------------------------

class RegimeSwitchingStrategy:
    """
    Combine multiple strategies by switching based on regime.

    Each strategy provides a daily return series.  The combined strategy
    uses the return of the designated strategy for each regime.

    Parameters
    ----------
    strategy_returns : dict
        {strategy_name: pd.Series} daily strategy returns.
    regime_strategy_map : dict
        {regime_label: strategy_name} mapping.
    blend : bool
        If True, blend strategies using regime posterior probabilities
        instead of hard switching.
    regime_probs : pd.DataFrame or None
        (dates x regimes) posterior probabilities for soft blending.
    """

    def __init__(
        self,
        strategy_returns: Dict[str, pd.Series],
        regime_strategy_map: Dict[str, str],
        blend: bool = False,
        regime_probs: Optional[pd.DataFrame] = None,
    ) -> None:
        self.strategy_returns = strategy_returns
        self.regime_strategy_map = regime_strategy_map
        self.blend = blend
        self.regime_probs = regime_probs

    def compute_portfolio_returns(
        self, regime_series: pd.Series
    ) -> pd.Series:
        """
        Compute portfolio returns under regime switching.

        Parameters
        ----------
        regime_series : pd.Series
            Regime label for each date.

        Returns
        -------
        pd.Series
            Combined daily return series.
        """
        if self.blend and self.regime_probs is not None:
            return self._soft_blend()
        return self._hard_switch(regime_series)

    def _hard_switch(self, regime_series: pd.Series) -> pd.Series:
        """Hard regime switching: use one strategy per regime."""
        # Build combined return series
        all_dates = sorted(set.union(*[
            set(s.index) for s in self.strategy_returns.values()
        ]))
        idx = pd.DatetimeIndex(all_dates)
        port_rets = pd.Series(np.nan, index=idx)

        regime_aligned = regime_series.reindex(idx, method="ffill")

        for date in idx:
            regime = regime_aligned.get(date, None)
            if regime is None or pd.isna(regime):
                continue
            strategy_name = self.regime_strategy_map.get(str(regime))
            if strategy_name is None:
                continue
            strat_ret = self.strategy_returns.get(strategy_name)
            if strat_ret is not None and date in strat_ret.index:
                port_rets[date] = strat_ret[date]

        return port_rets.dropna()

    def _soft_blend(self) -> pd.Series:
        """Soft blending: weight strategies by regime posterior probabilities."""
        probs = self.regime_probs
        # Map regime columns to strategies
        port_rets = pd.Series(0.0, index=probs.index)

        for regime_col in probs.columns:
            strategy_name = self.regime_strategy_map.get(str(regime_col))
            if strategy_name is None:
                continue
            strat_ret = self.strategy_returns.get(strategy_name)
            if strat_ret is None:
                continue
            aligned = strat_ret.reindex(probs.index).fillna(0)
            weight = probs[regime_col]
            port_rets += weight * aligned

        return port_rets

    def backtest(
        self,
        regime_series: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Full backtest with performance comparison.

        Parameters
        ----------
        regime_series : pd.Series
        benchmark : pd.Series or None
            Benchmark return series for comparison.

        Returns
        -------
        pd.DataFrame
            Performance metrics for switching strategy and benchmark.
        """
        port_rets = self.compute_portfolio_returns(regime_series)
        rows = []

        s = _stats(port_rets)
        rows.append({"strategy": "regime_switching", **{k: round(v, 4) for k, v in s.items()}})

        if benchmark is not None:
            sb = _stats(benchmark.reindex(port_rets.index).dropna())
            rows.append({"strategy": "benchmark", **{k: round(v, 4) for k, v in sb.items()}})

        # Individual strategies
        for name, ret_series in self.strategy_returns.items():
            si = _stats(ret_series.reindex(port_rets.index).dropna())
            rows.append({"strategy": name, **{k: round(v, 4) for k, v in si.items()}})

        return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# Correlation of regime-conditional returns
# ---------------------------------------------------------------------------

def regime_correlation_matrix(
    strategy_returns_dict: Dict[str, pd.Series],
    regime_series: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """
    Compute correlation matrix of strategies for each regime.

    Parameters
    ----------
    strategy_returns_dict : dict
        {name: pd.Series} strategy returns.
    regime_series : pd.Series
        Regime labels.

    Returns
    -------
    dict
        {regime_label: pd.DataFrame} correlation matrices.
    """
    combined = pd.concat(strategy_returns_dict, axis=1)
    combined["regime"] = regime_series
    combined = combined.dropna(subset=["regime"])

    result = {}
    for regime in sorted(combined["regime"].unique()):
        subset = combined[combined["regime"] == regime].drop(columns="regime")
        result[str(regime)] = subset.corr().round(4)

    return result


def rolling_regime_sharpe(
    strategy_returns: pd.Series,
    regime_series: pd.Series,
    window: int = 63,
    freq: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe ratio within each regime on a rolling basis.

    Parameters
    ----------
    strategy_returns : pd.Series
    regime_series : pd.Series
    window : int
        Rolling window length.
    freq : int

    Returns
    -------
    pd.DataFrame
        (dates x regimes) rolling Sharpe within each regime.
    """
    combined = pd.concat(
        [strategy_returns.rename("ret"), regime_series.rename("regime")],
        axis=1,
    ).dropna()
    regimes = sorted(combined["regime"].unique())
    result = pd.DataFrame(np.nan, index=combined.index,
                          columns=[str(r) for r in regimes])

    for regime in regimes:
        mask = combined["regime"] == regime
        regime_rets = combined.loc[mask, "ret"]

        for i in range(window, len(regime_rets)):
            window_rets = regime_rets.iloc[i - window:i]
            sharpe = window_rets.mean() / (window_rets.std() + 1e-12) * np.sqrt(freq)
            dt = regime_rets.index[i]
            result.loc[dt, str(regime)] = round(sharpe, 4)

    return result
