"""
research/signal_analytics/portfolio_signal.py
==============================================
Portfolio-level signal analytics.

Provides:
  - Aggregate signal strength and diversification benefit
  - Signal concentration (Herfindahl-Hirschman Index)
  - Cross-signal correlation impact on net IC
  - Signal capacity: at what AUM does alpha decay?
  - Turnover-cost impact on net IC

Usage example
-------------
>>> psa = PortfolioSignalAnalyzer()
>>> agg = psa.aggregate_signal_strength(delta_scores)
>>> conc = psa.signal_concentration(delta_scores)
>>> capacity = psa.signal_capacity(signal_series, returns_series)
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
class AggregateSignalResult:
    """Aggregate portfolio signal metrics."""
    total_signal: float             # sum of |delta_score| across positions
    mean_signal: float              # mean |delta_score|
    weighted_signal: float          # dollar-weighted signal
    n_active_positions: int
    diversification_ratio: float    # std(sum) / sum(std) — >1 means diversification
    long_signal: float              # total long signal
    short_signal: float             # total short signal
    net_signal: float               # long - |short|


@dataclass
class CapacityResult:
    """Signal capacity estimation results."""
    estimated_capacity_usd: float
    decay_onset_usd: float          # AUM at which IC starts declining
    decay_rate_per_usd: float       # rate of IC decay per additional USD
    current_ic: float
    capacity_ic: float              # IC at estimated capacity
    market_impact_slope: float      # linear slope of IC vs AUM


# ---------------------------------------------------------------------------
# PortfolioSignalAnalyzer
# ---------------------------------------------------------------------------

class PortfolioSignalAnalyzer:
    """Portfolio-level signal aggregation and diagnostics.

    Parameters
    ----------
    signal_col    : column name for delta_score or equivalent
    dollar_pos_col: column for position size
    return_col    : column for trade P&L
    """

    def __init__(
        self,
        signal_col: str = "delta_score",
        dollar_pos_col: str = "dollar_pos",
        return_col: str = "pnl",
    ) -> None:
        self.signal_col = signal_col
        self.dollar_pos_col = dollar_pos_col
        self.return_col = return_col

    # ------------------------------------------------------------------ #
    # Aggregate signal strength
    # ------------------------------------------------------------------ #

    def aggregate_signal_strength(
        self,
        delta_scores: pd.DataFrame,
        positions: Optional[pd.Series] = None,
    ) -> AggregateSignalResult:
        """Compute portfolio-level signal aggregation metrics.

        Parameters
        ----------
        delta_scores : DataFrame[time * assets] of delta_score values
                       OR pd.Series for a single time period
        positions    : optional pd.Series of dollar positions for weighting

        Returns
        -------
        AggregateSignalResult
        """
        if isinstance(delta_scores, pd.Series):
            # Single period
            scores = delta_scores.dropna()
            total = float(scores.abs().sum())
            mean_s = float(scores.abs().mean())
            long_s = float(scores[scores > 0].sum())
            short_s = float(scores[scores < 0].sum())
            n_active = int((scores != 0).sum())

            if positions is not None:
                pos_aligned = positions.loc[scores.index].fillna(0)
                weights = pos_aligned.abs() / pos_aligned.abs().sum()
                weighted = float((scores * weights).sum())
            else:
                weighted = float(scores.mean())

            # Diversification ratio: for a single period use std of sum / sum of std
            # as a single number: 1.0 (no meaningful diversification measure without time)
            div_ratio = 1.0

            return AggregateSignalResult(
                total_signal=total,
                mean_signal=mean_s,
                weighted_signal=weighted,
                n_active_positions=n_active,
                diversification_ratio=div_ratio,
                long_signal=long_s,
                short_signal=short_s,
                net_signal=long_s + short_s,
            )

        # Panel case: DataFrame[time * assets]
        abs_scores = delta_scores.abs()
        total_per_t = abs_scores.sum(axis=1)
        mean_total = float(total_per_t.mean())
        mean_signal = float(abs_scores.mean().mean())
        n_active = int((delta_scores != 0).any().sum())

        long_signal = float(delta_scores[delta_scores > 0].sum().sum())
        short_signal = float(delta_scores[delta_scores < 0].sum().sum())
        net_signal = long_signal + short_signal

        # Diversification ratio: sum std(asset_i) / std(sum asset_i)
        sum_std = float(delta_scores.std().sum())
        portfolio_std = float(delta_scores.sum(axis=1).std())
        div_ratio = sum_std / portfolio_std if portfolio_std > 0 else float("nan")

        if positions is not None:
            dollar_signals = delta_scores.multiply(positions.abs(), axis=1).sum(axis=1)
            pos_total = positions.abs().sum()
            weighted = float(dollar_signals.mean() / pos_total) if pos_total > 0 else float("nan")
        else:
            weighted = float(delta_scores.mean().mean())

        return AggregateSignalResult(
            total_signal=mean_total,
            mean_signal=mean_signal,
            weighted_signal=weighted,
            n_active_positions=n_active,
            diversification_ratio=div_ratio,
            long_signal=long_signal,
            short_signal=short_signal,
            net_signal=net_signal,
        )

    # ------------------------------------------------------------------ #
    # Signal concentration
    # ------------------------------------------------------------------ #

    def signal_concentration(
        self,
        delta_scores: pd.DataFrame | pd.Series,
    ) -> float:
        """Compute Herfindahl-Hirschman Index (HHI) of signal concentration.

        HHI = sum (w_i^2) where w_i = |delta_score_i| / sum |delta_score_j|

        HHI = 1/n for equal-weight (minimum concentration)
        HHI -> 1   for fully concentrated signal

        Parameters
        ----------
        delta_scores : signal values (Series for single period, DataFrame for panel)

        Returns
        -------
        float HHI in [1/n, 1]
        """
        if isinstance(delta_scores, pd.DataFrame):
            # Mean HHI over time
            hhi_vals: list[float] = []
            for _, row in delta_scores.iterrows():
                h = self._hhi(row.dropna())
                hhi_vals.append(h)
            return float(np.nanmean(hhi_vals))
        return self._hhi(delta_scores.dropna())

    @staticmethod
    def _hhi(values: pd.Series) -> float:
        """HHI for a single cross-section."""
        abs_vals = values.abs()
        total = abs_vals.sum()
        if total == 0:
            return float("nan")
        weights = abs_vals / total
        return float((weights**2).sum())

    # ------------------------------------------------------------------ #
    # Cross-signal correlation impact
    # ------------------------------------------------------------------ #

    def cross_signal_correlation_impact(
        self,
        delta_scores: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """Measure how cross-asset signal correlation affects portfolio IC.

        Computes:
          - Individual asset ICs
          - Portfolio (equal-weight sum) IC
          - Diversification benefit: portfolio_IC / mean(individual_IC)

        Parameters
        ----------
        delta_scores : DataFrame[time * assets]
        returns      : DataFrame[time * assets]

        Returns
        -------
        Dict with keys: portfolio_ic, mean_individual_ic, diversification_benefit,
                        signal_corr_mean, signal_corr_max
        """
        cols = delta_scores.columns.intersection(returns.columns)
        idx = delta_scores.index.intersection(returns.index)
        sig = delta_scores.loc[idx, cols]
        ret = returns.loc[idx, cols]

        # Individual ICs
        individual_ics: list[float] = []
        for col in cols:
            s = sig[col].dropna()
            r = ret[col].dropna()
            common = s.index.intersection(r.index)
            if len(common) < 3:
                continue
            r_ic, _ = stats.spearmanr(s.loc[common], r.loc[common])
            individual_ics.append(float(r_ic))

        mean_indiv_ic = float(np.nanmean(individual_ics)) if individual_ics else float("nan")

        # Portfolio IC: average signal and average return
        port_sig = sig.mean(axis=1).dropna()
        port_ret = ret.mean(axis=1).dropna()
        common_idx = port_sig.index.intersection(port_ret.index)
        if len(common_idx) >= 3:
            r_port, _ = stats.spearmanr(port_sig.loc[common_idx], port_ret.loc[common_idx])
            port_ic = float(r_port)
        else:
            port_ic = float("nan")

        div_benefit = (
            port_ic / mean_indiv_ic
            if not np.isnan(port_ic) and mean_indiv_ic != 0
            else float("nan")
        )

        # Signal correlation statistics
        sig_corr = sig.corr()
        upper_tri = sig_corr.values[np.triu_indices_from(sig_corr.values, k=1)]
        mean_corr = float(np.nanmean(upper_tri)) if len(upper_tri) > 0 else float("nan")
        max_corr = float(np.nanmax(np.abs(upper_tri))) if len(upper_tri) > 0 else float("nan")

        return {
            "portfolio_ic": port_ic,
            "mean_individual_ic": mean_indiv_ic,
            "diversification_benefit": div_benefit,
            "signal_corr_mean": mean_corr,
            "signal_corr_max": max_corr,
            "n_assets": len(cols),
        }

    # ------------------------------------------------------------------ #
    # Signal capacity
    # ------------------------------------------------------------------ #

    def signal_capacity(
        self,
        signal_series: pd.Series,
        returns_series: pd.Series,
        max_capacity: float = 1e7,
        n_bins: int = 10,
        dollar_vol_col: Optional[str] = None,
    ) -> CapacityResult:
        """Estimate AUM at which signal IC starts to decay.

        Method: split observations by implied dollar volume (position size)
        and measure IC within each bucket.  Fit a linear decay model.

        Parameters
        ----------
        signal_series  : signal values (1 per trade or period)
        returns_series : forward returns
        max_capacity   : maximum AUM to consider (USD)
        n_bins         : number of AUM bins
        dollar_vol_col : if given, use this external dollar-volume series

        Returns
        -------
        CapacityResult
        """
        df = pd.concat({"sig": signal_series, "ret": returns_series}, axis=1).dropna()
        if len(df) < n_bins * 3:
            return CapacityResult(
                estimated_capacity_usd=float("nan"),
                decay_onset_usd=float("nan"),
                decay_rate_per_usd=float("nan"),
                current_ic=float("nan"),
                capacity_ic=float("nan"),
                market_impact_slope=float("nan"),
            )

        # Current (overall) IC
        r_ic, _ = stats.spearmanr(df["sig"], df["ret"])
        current_ic = float(r_ic)

        # Simulate IC at different hypothetical AUM levels by
        # subsetting observations with signal > AUM-dependent threshold
        # (weaker signals are dropped as AUM makes them uneconomical)
        aum_levels = np.linspace(0, max_capacity, n_bins + 1)[1:]
        ics_at_aum: list[float] = []

        for aum in aum_levels:
            # Filter: keep only trades where |signal| exceeds a threshold
            # proportional to AUM (market impact model: threshold ∝ AUM)
            threshold = (aum / max_capacity) * df["sig"].abs().quantile(0.9)
            subset = df[df["sig"].abs() >= threshold]
            if len(subset) < 5:
                ics_at_aum.append(float("nan"))
                continue
            r_sub, _ = stats.spearmanr(subset["sig"], subset["ret"])
            ics_at_aum.append(float(r_sub))

        ics_arr = np.array(ics_at_aum)
        valid_mask = ~np.isnan(ics_arr)

        if valid_mask.sum() < 3:
            return CapacityResult(
                estimated_capacity_usd=float("nan"),
                decay_onset_usd=float("nan"),
                decay_rate_per_usd=float("nan"),
                current_ic=current_ic,
                capacity_ic=float("nan"),
                market_impact_slope=float("nan"),
            )

        # Linear fit: IC(AUM) = IC_0 - slope * AUM
        aum_valid = aum_levels[valid_mask]
        ic_valid = ics_arr[valid_mask]
        slope, intercept, r_val, p_val, se = stats.linregress(aum_valid, ic_valid)

        # Estimated capacity: AUM where IC drops to 50% of current IC
        if slope < 0 and current_ic != 0:
            capacity_usd = (intercept - 0.5 * current_ic) / (-slope)
            decay_onset = (intercept - 0.9 * current_ic) / (-slope)
        else:
            capacity_usd = float("inf")
            decay_onset = float("inf")

        capacity_ic = float(intercept + slope * min(capacity_usd, max_capacity))

        return CapacityResult(
            estimated_capacity_usd=float(max(0, capacity_usd)),
            decay_onset_usd=float(max(0, decay_onset)),
            decay_rate_per_usd=float(-slope) if slope < 0 else 0.0,
            current_ic=current_ic,
            capacity_ic=capacity_ic,
            market_impact_slope=float(slope),
        )

    # ------------------------------------------------------------------ #
    # Turnover cost impact
    # ------------------------------------------------------------------ #

    def turnover_cost_impact(
        self,
        trades: pd.DataFrame,
        transaction_cost: float = 0.0002,
        signal_col: Optional[str] = None,
        return_col: Optional[str] = None,
        dollar_pos_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute gross and net IC after deducting transaction costs.

        Parameters
        ----------
        trades           : trade DataFrame
        transaction_cost : one-way cost fraction
        signal_col       : override signal column
        return_col       : override return column
        dollar_pos_col   : override position column

        Returns
        -------
        Dict with gross_ic, net_ic, cost_drag, round_trip_cost,
              mean_turnover, n_trades
        """
        sig_col = signal_col or self.signal_col
        ret_col = return_col or self.return_col
        pos_col = dollar_pos_col or self.dollar_pos_col

        df = trades.copy()
        if pos_col in df.columns:
            pos = df[pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[ret_col] / pos
        else:
            df["_ret"] = df[ret_col]

        sub = df[[sig_col, "_ret"]].dropna()
        if len(sub) < 3:
            return {"gross_ic": float("nan"), "net_ic": float("nan")}

        r_ic, _ = stats.spearmanr(sub[sig_col], sub["_ret"])
        gross_ic = float(r_ic)

        # Estimate turnover from consecutive signal changes
        if sig_col in trades.columns and "sym" in trades.columns:
            # Per-symbol consecutive delta
            turns: list[float] = []
            for sym in trades["sym"].unique():
                sym_trades = trades[trades["sym"] == sym].sort_index()
                if sig_col in sym_trades.columns:
                    delta = sym_trades[sig_col].diff().abs()
                    turns.extend(delta.dropna().tolist())
            mean_turnover = float(np.mean(turns)) if turns else 1.0
        else:
            mean_turnover = 1.0

        round_trip_cost = 2 * transaction_cost
        cost_drag = round_trip_cost * mean_turnover
        net_ic = gross_ic - cost_drag / max(abs(gross_ic), 1e-8)

        return {
            "gross_ic": gross_ic,
            "net_ic": net_ic,
            "cost_drag": cost_drag,
            "round_trip_cost": round_trip_cost,
            "mean_turnover": mean_turnover,
            "n_trades": len(sub),
        }

    # ------------------------------------------------------------------ #
    # Portfolio signal time-series
    # ------------------------------------------------------------------ #

    def portfolio_signal_timeseries(
        self,
        delta_scores: pd.DataFrame,
        weights: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute portfolio-level signal statistics over time.

        Parameters
        ----------
        delta_scores : DataFrame[time * assets]
        weights      : optional weight matrix; defaults to equal-weight

        Returns
        -------
        DataFrame with columns: total_signal, n_active, hhi, long, short, net
        """
        records: list[dict] = []
        for t in delta_scores.index:
            row = delta_scores.loc[t].dropna()
            if len(row) == 0:
                records.append({
                    "total_signal": 0.0, "n_active": 0, "hhi": float("nan"),
                    "long": 0.0, "short": 0.0, "net": 0.0,
                })
                continue

            total = float(row.abs().sum())
            n_active = int((row != 0).sum())
            hhi = self._hhi(row)
            long_s = float(row[row > 0].sum())
            short_s = float(row[row < 0].sum())

            records.append({
                "total_signal": total,
                "n_active": n_active,
                "hhi": hhi,
                "long": long_s,
                "short": short_s,
                "net": long_s + short_s,
            })

        return pd.DataFrame(records, index=delta_scores.index)

    # ------------------------------------------------------------------ #
    # Visualisations
    # ------------------------------------------------------------------ #

    def plot_signal_concentration(
        self,
        delta_scores: pd.DataFrame,
        save_path: Optional[str | Path] = None,
        title: str = "Portfolio Signal Concentration",
    ) -> plt.Figure:
        """Time-series of HHI signal concentration."""
        ts = self.portfolio_signal_timeseries(delta_scores)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax = axes[0]
        ax.plot(ts.index, ts["total_signal"], color="#3498db", linewidth=1.0)
        ax.set_ylabel("Total |Signal|")
        ax.set_title(title)

        ax2 = axes[1]
        ax2.plot(ts.index, ts["hhi"], color="#e74c3c", linewidth=1.0, label="HHI")
        n_cols = delta_scores.shape[1]
        ax2.axhline(1 / n_cols, color="gray", linestyle="--", linewidth=0.8,
                    label=f"Equal-weight (1/n={1/n_cols:.3f})")
        ax2.set_ylabel("HHI")
        ax2.set_xlabel("Date")
        ax2.legend()

        fig.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_capacity_curve(
        self,
        capacity_result: CapacityResult,
        max_capacity: float = 1e7,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Plot IC vs AUM capacity curve."""
        fig, ax = plt.subplots(figsize=(9, 5))
        aum_range = np.linspace(0, max_capacity, 300)
        slope = capacity_result.market_impact_slope
        ic0 = capacity_result.current_ic
        # Reconstruct: IC(AUM) ~= ic0 + slope * AUM (slope < 0 for decay)
        ic_curve = ic0 + slope * aum_range
        ic_curve = np.clip(ic_curve, 0, None)

        ax.plot(aum_range / 1e6, ic_curve, color="#3498db", linewidth=2, label="IC(AUM)")
        ax.axhline(ic0 * 0.5, color="#e74c3c", linestyle="--", linewidth=1,
                   label=f"50% IC = {ic0*0.5:.4f}")

        if not np.isnan(capacity_result.estimated_capacity_usd):
            ax.axvline(
                capacity_result.estimated_capacity_usd / 1e6,
                color="#e74c3c", linestyle="--", linewidth=1,
                label=f"Capacity ~${capacity_result.estimated_capacity_usd/1e6:.1f}M",
            )

        ax.set_xlabel("AUM ($M)")
        ax.set_ylabel("IC")
        ax.set_title("Signal Capacity Curve")
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_long_short_signal_balance(
        self,
        delta_scores: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Stacked area chart of long and short signal over time."""
        ts = self.portfolio_signal_timeseries(delta_scores)
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.fill_between(ts.index, ts["long"], 0, alpha=0.6, color="#2ecc71", label="Long Signal")
        ax.fill_between(ts.index, ts["short"], 0, alpha=0.6, color="#e74c3c", label="Short Signal")
        ax.plot(ts.index, ts["net"], color="black", linewidth=1.0, label="Net Signal")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Aggregate Signal")
        ax.set_title("Long/Short Signal Balance Over Time")
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # Signal diversification score
    # ------------------------------------------------------------------ #

    def signal_diversification_score(
        self,
        delta_scores: pd.DataFrame,
    ) -> float:
        """Effective number of independent signals (diversification ratio^2).

        Based on the portfolio theory concept:
          DR = sum(std_i) / std(sum_i)
          Effective_N = DR^2

        Parameters
        ----------
        delta_scores : DataFrame[time x assets]

        Returns
        -------
        float effective number of independent signals
        """
        stds = delta_scores.std()
        sum_std = float(stds.sum())
        portfolio_std = float(delta_scores.sum(axis=1).std())
        if portfolio_std == 0:
            return float("nan")
        dr = sum_std / portfolio_std
        return float(dr ** 2)

    # ------------------------------------------------------------------ #
    # Marginal signal contribution
    # ------------------------------------------------------------------ #

    def marginal_signal_contribution(
        self,
        delta_scores: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """IC contribution of each asset's signal to the portfolio IC.

        Measures: partial correlation of each signal with portfolio return,
        controlling for aggregate signal.

        Parameters
        ----------
        delta_scores : DataFrame[time x assets]
        returns      : DataFrame[time x assets]

        Returns
        -------
        pd.Series[asset -> marginal IC contribution]
        """
        cols = delta_scores.columns.intersection(returns.columns)
        idx = delta_scores.index.intersection(returns.index)
        sig = delta_scores.loc[idx, cols]
        ret = returns.loc[idx, cols]

        port_sig = sig.mean(axis=1).dropna()
        port_ret = ret.mean(axis=1).dropna()
        common_idx = port_sig.index.intersection(port_ret.index)

        result: dict[str, float] = {}
        for col in cols:
            s_col = sig[col].loc[common_idx].fillna(0)
            r_col = ret[col].loc[common_idx].fillna(0)

            r_ind, _ = stats.spearmanr(s_col, port_ret.loc[common_idx])
            r_port, _ = stats.spearmanr(port_sig.loc[common_idx], port_ret.loc[common_idx])
            result[col] = float(r_ind) - float(r_port)

        return pd.Series(result, name="marginal_IC_contribution").sort_values(ascending=False)

    # ------------------------------------------------------------------ #
    # Rolling portfolio IC
    # ------------------------------------------------------------------ #

    def rolling_portfolio_ic(
        self,
        delta_scores: pd.DataFrame,
        returns: pd.DataFrame,
        window: int = 60,
        method: str = "spearman",
    ) -> pd.Series:
        """Rolling cross-asset IC of the aggregate portfolio signal.

        Parameters
        ----------
        delta_scores : DataFrame[time x assets]
        returns      : DataFrame[time x assets]
        window       : rolling window in time periods
        method       : correlation method

        Returns
        -------
        pd.Series[time -> rolling IC]
        """
        cols = delta_scores.columns.intersection(returns.columns)
        idx = delta_scores.index.intersection(returns.index)
        sig = delta_scores.loc[idx, cols]
        ret = returns.loc[idx, cols]

        ic_vals: list[float] = []
        for i in range(window - 1, len(idx)):
            w_sig = sig.iloc[i - window + 1 : i + 1].values.flatten()
            w_ret = ret.iloc[i - window + 1 : i + 1].values.flatten()
            mask = ~(np.isnan(w_sig) | np.isnan(w_ret))
            if mask.sum() < 3:
                ic_vals.append(float("nan"))
                continue
            if method == "spearman":
                r, _ = stats.spearmanr(w_sig[mask], w_ret[mask])
            else:
                r, _ = stats.pearsonr(w_sig[mask], w_ret[mask])
            ic_vals.append(float(r))

        return pd.Series(ic_vals, index=idx[window - 1 :], name="portfolio_rolling_ic")

    # ------------------------------------------------------------------ #
    # Signal timing quality
    # ------------------------------------------------------------------ #

    def signal_timing_quality(
        self,
        trades: pd.DataFrame,
        signal_col: Optional[str] = None,
        return_col: Optional[str] = None,
        dollar_pos_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """Measure how well signal timing aligns with return timing.

        Parameters
        ----------
        trades        : trade records
        signal_col    : signal column override
        return_col    : return column override
        dollar_pos_col: position column override

        Returns
        -------
        Dict with keys: rank_corr, hit_rate_direction, top_quartile_ret, bottom_quartile_ret
        """
        sig_col = signal_col or self.signal_col
        ret_col = return_col or self.return_col
        pos_col = dollar_pos_col or self.dollar_pos_col

        df = trades.copy()
        if pos_col in df.columns:
            pos = df[pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[ret_col] / pos
        else:
            df["_ret"] = df[ret_col]

        sub = df[[sig_col, "_ret"]].dropna()
        if len(sub) < 10:
            return {}

        sig = sub[sig_col].values
        ret = sub["_ret"].values

        r, _ = stats.spearmanr(sig, ret)
        hit_rate = float(np.mean(np.sign(sig) == np.sign(ret)))

        q75 = np.quantile(sig, 0.75)
        q25 = np.quantile(sig, 0.25)
        top_ret = float(np.mean(ret[sig >= q75]))
        bot_ret = float(np.mean(ret[sig <= q25]))

        return {
            "rank_corr": float(r),
            "hit_rate_direction": hit_rate,
            "top_quartile_ret": top_ret,
            "bottom_quartile_ret": bot_ret,
            "timing_spread": top_ret - bot_ret,
            "n_obs": len(sub),
        }

    # ------------------------------------------------------------------ #
    # Plot marginal signal contributions
    # ------------------------------------------------------------------ #

    def plot_marginal_signal_contributions(
        self,
        marginal_ic: pd.Series,
        save_path: Optional[str | Path] = None,
        title: str = "Marginal IC Contribution per Asset",
    ) -> plt.Figure:
        """Horizontal bar chart of marginal IC contributions.

        Parameters
        ----------
        marginal_ic : output of marginal_signal_contribution()
        save_path   : optional save path
        title       : figure title
        """
        fig, ax = plt.subplots(figsize=(8, max(4, len(marginal_ic) * 0.4)))
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in marginal_ic.values]
        ax.barh(range(len(marginal_ic)), marginal_ic.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(marginal_ic)))
        ax.set_yticklabels(marginal_ic.index.tolist(), fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Marginal IC Contribution")
        ax.set_title(title)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------
    # Extended portfolio-signal analytics
    # ------------------------------------------------------------------

    def signal_pnl_attribution(
        self,
        trades: pd.DataFrame,
        signal_cols: list[str] | None = None,
        pnl_col: str = "pnl",
    ) -> pd.DataFrame:
        """
        Attribute realised PnL to each signal using OLS with signal values
        as regressors and trade PnL as the dependent variable.

        The OLS coefficient gamma_k approximates the marginal dollar PnL
        per unit of signal strength.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade log with pnl and signal columns.
        signal_cols : list of str, optional
            Signal column names (default: common BH columns).
        pnl_col : str
            PnL column name.

        Returns
        -------
        pd.DataFrame with columns [signal, gamma, std_error, t_stat,
                                    pnl_attributed, pct_of_total_pnl, r_squared].
        """
        default_signals = ["delta_score", "ensemble_signal", "tf_score", "mass"]
        scols = signal_cols or [s for s in default_signals if s in trades.columns]
        if not scols or pnl_col not in trades.columns:
            raise KeyError("Signal columns or PnL column missing.")

        sub = trades[scols + [pnl_col]].dropna()
        if len(sub) < len(scols) + 2:
            raise ValueError("Insufficient observations for PnL attribution OLS.")

        X = sub[scols].values.astype(float)
        y = sub[pnl_col].values.astype(float)
        X_c = np.column_stack([np.ones(len(X)), X])

        try:
            coef, _, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"OLS failed: {exc}") from exc

        fitted = X_c @ coef
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        n, p = X_c.shape
        dof = max(n - p, 1)
        residuals = y - fitted
        mse = float(np.sum(residuals ** 2) / dof)
        try:
            cov = mse * np.linalg.inv(X_c.T @ X_c)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(p, np.nan)

        total_pnl = float(y.sum())
        rows: list[dict] = []
        for k, sname in enumerate(scols):
            gamma = float(coef[k + 1])
            t_stat = gamma / se[k + 1] if np.isfinite(se[k + 1]) and se[k + 1] > 0 else np.nan
            attributed = float(gamma * sub[sname].sum())
            rows.append({
                "signal": sname,
                "gamma": gamma,
                "std_error": float(se[k + 1]),
                "t_stat": t_stat,
                "pnl_attributed": attributed,
                "pct_of_total_pnl": attributed / total_pnl if total_pnl != 0 else np.nan,
                "r_squared": r2,
            })

        return pd.DataFrame(rows)

    def signal_turnover_profile(
        self,
        trades: pd.DataFrame,
        signal_cols: list[str] | None = None,
        time_col: str = "exit_time",
    ) -> pd.DataFrame:
        """
        Estimate the turnover profile for each signal: fraction of bars on
        which the signal changes sign, and implied holding period.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade log sorted by time.
        signal_cols : list of str, optional
            Signal columns to analyse.
        time_col : str
            Datetime column for sorting.

        Returns
        -------
        pd.DataFrame with columns [signal, mean_turnover, std_turnover,
                                    annualised_turnover, holding_period_estimate, n_bars].
        """
        default_signals = ["delta_score", "ensemble_signal", "tf_score", "mass"]
        scols = signal_cols or [s for s in default_signals if s in trades.columns]
        if not scols:
            raise KeyError("No signal columns found.")

        df = trades.copy()
        if time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)

        rows: list[dict] = []
        for sname in scols:
            if sname not in df.columns:
                continue
            sig = df[sname].dropna().values.astype(float)
            if len(sig) < 4:
                continue
            sign_changes = np.diff(np.sign(sig)) != 0
            turnover_rate = float(sign_changes.mean())
            hold_est = 1.0 / turnover_rate if turnover_rate > 0 else np.inf
            annual_to = turnover_rate * 252
            rows.append({
                "signal": sname,
                "mean_turnover": turnover_rate,
                "std_turnover": float(sign_changes.std()),
                "annualised_turnover": annual_to,
                "holding_period_estimate": hold_est,
                "n_bars": len(sig),
            })

        return pd.DataFrame(rows)

    def signal_size_distribution(
        self,
        trades: pd.DataFrame,
        signal_col: str = "delta_score",
        n_bins: int = 20,
    ) -> pd.DataFrame:
        """
        Analyse the distribution of signal magnitude and its relationship
        to trade PnL. Bins signal absolute value into quantiles and computes
        mean/std PnL, hit-rate, and avg hold bars per bin.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade log.
        signal_col : str
            Signal column.
        n_bins : int
            Number of magnitude bins.

        Returns
        -------
        pd.DataFrame with columns [bin_label, signal_mean, signal_std,
                                    mean_pnl, std_pnl, hit_rate, avg_hold_bars, n_obs].
        """
        if signal_col not in trades.columns:
            raise KeyError(f"Column '{signal_col}' not found.")

        extra = [c for c in ["pnl", "hold_bars"] if c in trades.columns]
        df = trades[[signal_col] + extra].dropna(subset=[signal_col]).copy()
        df["_abs_sig"] = df[signal_col].abs()

        try:
            df["_bin"] = pd.qcut(df["_abs_sig"], q=n_bins, duplicates="drop")
        except ValueError:
            df["_bin"] = pd.cut(df["_abs_sig"], bins=n_bins)

        rows: list[dict] = []
        for bin_label, grp in df.groupby("_bin", observed=False):
            row: dict = {
                "bin_label": str(bin_label),
                "signal_mean": float(grp[signal_col].mean()),
                "signal_std": float(grp[signal_col].std()),
                "n_obs": len(grp),
            }
            if "pnl" in grp.columns:
                row["mean_pnl"] = float(grp["pnl"].mean())
                row["std_pnl"] = float(grp["pnl"].std())
                row["hit_rate"] = float((grp["pnl"] > 0).mean())
            if "hold_bars" in grp.columns:
                row["avg_hold_bars"] = float(grp["hold_bars"].mean())
            rows.append(row)

        return pd.DataFrame(rows)

    def top_bottom_signal_analysis(
        self,
        trades: pd.DataFrame,
        signal_col: str = "delta_score",
        pnl_col: str = "pnl",
        top_pct: float = 0.2,
    ) -> dict:
        """
        Compare top-quintile vs bottom-quintile signal trades by PnL, hit
        rate and average holding period.

        Parameters
        ----------
        trades : pd.DataFrame
        signal_col : str
        pnl_col : str
        top_pct : float
            Fraction of trades classified as top/bottom (default 20%).

        Returns
        -------
        dict with keys 'top', 'bottom', 'spread' each containing summary stats.
        """
        hold_cols = ["hold_bars"] if "hold_bars" in trades.columns else []
        df = trades[[signal_col, pnl_col] + hold_cols].dropna()
        k = max(1, int(len(df) * top_pct))

        df_sorted = df.sort_values(signal_col)
        bottom = df_sorted.iloc[:k]
        top = df_sorted.iloc[-k:]

        def _stats(sub: pd.DataFrame) -> dict:
            pnl = sub[pnl_col]
            out: dict = {
                "n": len(sub),
                "mean_pnl": float(pnl.mean()),
                "std_pnl": float(pnl.std()),
                "hit_rate": float((pnl > 0).mean()),
                "total_pnl": float(pnl.sum()),
                "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else np.nan,
            }
            if "hold_bars" in sub.columns:
                out["avg_hold_bars"] = float(sub["hold_bars"].mean())
            return out

        top_stats = _stats(top)
        bot_stats = _stats(bottom)
        spread: dict = {
            "mean_pnl_spread": top_stats["mean_pnl"] - bot_stats["mean_pnl"],
            "hit_rate_spread": top_stats["hit_rate"] - bot_stats["hit_rate"],
            "total_pnl_spread": top_stats["total_pnl"] - bot_stats["total_pnl"],
        }
        return {"top": top_stats, "bottom": bot_stats, "spread": spread}

    def plot_signal_pnl_attribution(
        self,
        attribution_df: pd.DataFrame,
        save_path=None,
        title: str = "Signal PnL Attribution",
    ):
        """
        Horizontal waterfall bar chart of PnL attributed to each signal.

        Parameters
        ----------
        attribution_df : output of signal_pnl_attribution()
        save_path      : optional save path
        title          : chart title

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = attribution_df.sort_values("pnl_attributed", ascending=True)
        signals = df["signal"].tolist()
        vals = df["pnl_attributed"].values

        fig, ax = plt.subplots(figsize=(8, max(4, len(signals) * 0.5)))
        colors = ["#27ae60" if v >= 0 else "#c0392b" for v in vals]
        ax.barh(signals, vals, color=colors, alpha=0.75, edgecolor="black")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Attributed PnL ($)")
        ax.set_ylabel("Signal")
        r2 = df["r_squared"].iloc[0] if "r_squared" in df.columns else np.nan
        if np.isfinite(r2):
            ax.text(0.98, 0.02, f"R^2={r2:.3f}", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=9, color="grey")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_signal_size_distribution(
        self,
        size_df: pd.DataFrame,
        save_path=None,
        title: str = "Signal Size vs PnL Distribution",
    ):
        """
        Dual-axis bar/line chart of mean PnL and hit-rate across signal
        magnitude bins.

        Parameters
        ----------
        size_df   : output of signal_size_distribution()
        save_path : optional save path
        title     : chart title

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        x = range(len(size_df))
        labels = [str(b)[:12] for b in size_df["bin_label"]]

        if "mean_pnl" in size_df.columns:
            colors = ["#27ae60" if v >= 0 else "#c0392b" for v in size_df["mean_pnl"]]
            ax1.bar(x, size_df["mean_pnl"], color=colors, alpha=0.6, label="Mean PnL")
            ax1.set_ylabel("Mean PnL ($)")

        if "hit_rate" in size_df.columns:
            ax2.plot(x, size_df["hit_rate"], color="#e67e22", linewidth=2,
                     marker="o", markersize=4, label="Hit Rate")
            ax2.axhline(0.5, color="#e67e22", linewidth=0.8, linestyle="--", alpha=0.5)
            ax2.set_ylabel("Hit Rate")
            ax2.set_ylim(0, 1)

        ax1.set_xticks(list(x))
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax1.set_xlabel("Signal Magnitude Bin")
        ax1.set_title(title, fontsize=13, fontweight="bold")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

        fig.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig
