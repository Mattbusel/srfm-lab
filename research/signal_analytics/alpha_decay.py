"""
research/signal_analytics/alpha_decay.py
=========================================
Alpha decay and optimal holding period analysis.

Provides:
  - Half-life estimation via exponential fit to IC decay curve
  - Auto-correlation-based half-life (price-return persistence)
  - Optimal holding period maximising net IC after transaction costs
  - Rolling half-life timeline
  - Turnover statistics
  - Transaction-cost-adjusted IC

Usage example
-------------
>>> analyzer = AlphaDecayAnalyzer()
>>> decay_model = analyzer.signal_decay_model(ic_decay_result)
>>> opt_hold = analyzer.optimal_holding_period(ic_decay_result, transaction_cost=0.0002)
>>> analyzer.plot_alpha_decay(ic_decay_result, save_path="results/decay.png")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, minimize_scalar

from research.signal_analytics.ic_framework import ICCalculator, ICDecayResult


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DecayModel:
    """Fitted exponential alpha decay model: IC(h) = IC0 * exp(-lambda_ * h)."""
    ic_at_zero: float       # IC0
    decay_rate: float       # lambda_ (per bar)
    half_life: float        # ln(2)/lambda_ in bars
    r_squared: float        # goodness-of-fit
    model_type: str = "exponential"

    def ic_at_horizon(self, h: float) -> float:
        """Evaluate model IC at horizon *h*."""
        return self.ic_at_zero * np.exp(-self.decay_rate * h)

    def horizon_for_ic(self, target_ic: float) -> float:
        """Return horizon at which model IC equals *target_ic*."""
        if self.ic_at_zero == 0:
            return float("nan")
        ratio = target_ic / self.ic_at_zero
        if ratio <= 0 or self.decay_rate == 0:
            return float("inf")
        return -np.log(ratio) / self.decay_rate


@dataclass
class TurnoverStats:
    """Turnover statistics for a given signal and holding period."""
    holding_period: int          # bars
    daily_turnover: float        # fraction of portfolio replaced per bar
    weekly_turnover: float       # fraction per 5-bar week
    monthly_turnover: float      # fraction per 21-bar month
    signal_autocorr_lag1: float  # lag-1 autocorrelation of signal
    signal_autocorr_full: pd.Series  # full ACF up to holding_period
    avg_holding_bars: float      # average realised holding period
    round_trip_cost: float       # cost_per_trade * daily_turnover


# ---------------------------------------------------------------------------
# AlphaDecayAnalyzer
# ---------------------------------------------------------------------------

class AlphaDecayAnalyzer:
    """Comprehensive alpha decay diagnostics.

    Parameters
    ----------
    default_cost : default transaction cost per one-way trade (fraction)
    """

    def __init__(self, default_cost: float = 0.0002) -> None:
        self.default_cost = default_cost
        self._ic_calc = ICCalculator()

    # ------------------------------------------------------------------ #
    # Half-life from IC decay
    # ------------------------------------------------------------------ #

    def compute_signal_halflife(self, ic_decay_result: ICDecayResult) -> float:
        """Extract half-life from a fitted ICDecayResult.

        Returns
        -------
        float — bars until IC halves (from fitted exponential)
        """
        if not np.isnan(ic_decay_result.half_life):
            return float(ic_decay_result.half_life)
        # Fallback: numerical search
        return self._numerical_halflife(
            ic_decay_result.horizons,
            ic_decay_result.ic_values,
        )

    def _numerical_halflife(
        self,
        horizons: List[int],
        ic_values: List[float],
    ) -> float:
        """Find h such that IC(h) = 0.5 * IC(1) via linear interpolation."""
        h = np.array(horizons, dtype=float)
        ic = np.array(ic_values, dtype=float)
        mask = ~np.isnan(ic)
        if mask.sum() < 2:
            return float("nan")
        h_clean, ic_clean = h[mask], ic[mask]
        if len(ic_clean) == 0 or ic_clean[0] == 0:
            return float("nan")
        target = 0.5 * ic_clean[0]
        # Find crossing
        for i in range(1, len(ic_clean)):
            if (ic_clean[i] - target) * (ic_clean[i - 1] - target) <= 0:
                # Linear interpolation
                frac = (target - ic_clean[i - 1]) / (ic_clean[i] - ic_clean[i - 1])
                return float(h_clean[i - 1] + frac * (h_clean[i] - h_clean[i - 1]))
        return float("inf")

    # ------------------------------------------------------------------ #
    # Auto-correlation half-life
    # ------------------------------------------------------------------ #

    def auto_correlation_halflife(
        self,
        returns: pd.Series,
        max_lag: int = 60,
    ) -> float:
        """Estimate return persistence half-life via AR(1) autocorrelation.

        Fits: ACF(h) ~= rho^h -> half-life = ln(0.5)/ln(rho)

        Parameters
        ----------
        returns : pd.Series of return values
        max_lag : maximum lag to consider

        Returns
        -------
        float — half-life in bars
        """
        clean = returns.dropna().values
        n = len(clean)
        if n < max_lag + 5:
            return float("nan")

        # Compute autocorrelation at each lag
        mean_r = clean.mean()
        var_r = np.var(clean, ddof=1)
        if var_r == 0:
            return float("nan")

        acf_vals: list[float] = []
        for lag in range(1, max_lag + 1):
            cov = np.mean((clean[: n - lag] - mean_r) * (clean[lag:] - mean_r))
            acf_vals.append(cov / var_r)

        # Fit AR(1) model: acf[h] ~= rho^h
        lags = np.arange(1, max_lag + 1, dtype=float)
        acf_arr = np.array(acf_vals)
        mask = ~np.isnan(acf_arr) & (acf_arr > 0)
        if mask.sum() < 3:
            return float("nan")

        try:
            def _ar1_model(h: np.ndarray, rho: float) -> np.ndarray:
                return rho**h

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(_ar1_model, lags[mask], acf_arr[mask], p0=[0.5], bounds=(0, 1))
            rho = float(popt[0])
            if rho <= 0 or rho >= 1:
                return float("nan")
            return float(np.log(0.5) / np.log(rho))
        except Exception:
            # Fallback: lag-1 autocorrelation only
            rho1 = float(acf_vals[0])
            if rho1 <= 0 or rho1 >= 1:
                return float("nan")
            return float(np.log(0.5) / np.log(rho1))

    # ------------------------------------------------------------------ #
    # Optimal holding period
    # ------------------------------------------------------------------ #

    def optimal_holding_period(
        self,
        ic_decay_result: ICDecayResult,
        transaction_cost: float = 0.0002,
        turnover_per_bar: float = 1.0,
    ) -> int:
        """Find holding period that maximises net IC after transaction costs.

        Objective:  IC(h) * sqrth - 2 * cost / IC(h)   (Grinold-Kahn framework)

        More precisely we maximise:
            net_IC(h) = IC(h) * sqrth - cost_drag(h)
        where:
            cost_drag(h) = 2 * transaction_cost * turnover_per_bar

        Parameters
        ----------
        ic_decay_result  : ICDecayResult from ic_decay()
        transaction_cost : one-way cost as fraction (default 0.0002 = 2bps)
        turnover_per_bar : fraction of portfolio turned over per bar

        Returns
        -------
        int — optimal holding period in bars
        """
        horizons = np.array(ic_decay_result.horizons, dtype=float)
        ic_vals = np.array(ic_decay_result.ic_values, dtype=float)

        # If we have a fitted model, use it for a smooth objective
        if not np.isnan(ic_decay_result.decay_rate) and not np.isnan(ic_decay_result.ic_at_zero):
            ic0 = ic_decay_result.ic_at_zero
            lam = ic_decay_result.decay_rate

            def neg_net_ic(h: float) -> float:
                if h <= 0:
                    return 1e10
                ic_h = ic0 * np.exp(-lam * h)
                # Net benefit: IC sqrth minus round-trip cost amortised over holding period
                round_trip_cost = 2 * transaction_cost * turnover_per_bar
                net = ic_h * np.sqrt(h) - round_trip_cost / max(ic_h, 1e-8)
                return -net

            result = minimize_scalar(neg_net_ic, bounds=(1, max(horizons)), method="bounded")
            return max(1, int(round(result.x)))

        # Fallback: evaluate over discrete horizons
        mask = ~np.isnan(ic_vals)
        if mask.sum() == 0:
            return 1
        net_ics: list[Tuple[float, int]] = []
        for h, ic in zip(horizons[mask], ic_vals[mask]):
            round_trip_cost = 2 * transaction_cost * turnover_per_bar
            if abs(ic) < 1e-10:
                continue
            net = ic * np.sqrt(h) - round_trip_cost / abs(ic)
            net_ics.append((net, int(h)))

        if not net_ics:
            return 1
        return max(net_ics, key=lambda x: x[0])[1]

    # ------------------------------------------------------------------ #
    # Decay model fitting
    # ------------------------------------------------------------------ #

    def signal_decay_model(
        self,
        ic_decay_result: ICDecayResult,
    ) -> DecayModel:
        """Fit IC(h) = IC0 * exp(-lambda_ * h) and return model parameters.

        Parameters
        ----------
        ic_decay_result : ICDecayResult with horizons and ic_values

        Returns
        -------
        DecayModel with decay_rate lambda_, half_life, ic_at_zero, r_squared
        """
        # Use the already-fitted values if available
        if not np.isnan(ic_decay_result.decay_rate):
            return DecayModel(
                ic_at_zero=ic_decay_result.ic_at_zero,
                decay_rate=ic_decay_result.decay_rate,
                half_life=ic_decay_result.half_life,
                r_squared=ic_decay_result.r_squared,
                model_type="exponential",
            )

        # Attempt fresh fit
        h = np.array(ic_decay_result.horizons, dtype=float)
        ic = np.array(ic_decay_result.ic_values, dtype=float)
        mask = ~np.isnan(ic)
        if mask.sum() < 3:
            return DecayModel(
                ic_at_zero=float("nan"),
                decay_rate=float("nan"),
                half_life=float("nan"),
                r_squared=float("nan"),
            )

        def _exp(h_: np.ndarray, ic0: float, lam: float) -> np.ndarray:
            return ic0 * np.exp(-lam * h_)

        try:
            popt, _ = curve_fit(_exp, h[mask], ic[mask], p0=[ic[mask][0], 0.1], maxfev=5000)
            ic0, lam = popt
            pred = _exp(h[mask], ic0, lam)
            ss_res = np.sum((ic[mask] - pred) ** 2)
            ss_tot = np.sum((ic[mask] - ic[mask].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            hl = np.log(2) / lam if lam > 0 else float("inf")
            return DecayModel(
                ic_at_zero=float(ic0),
                decay_rate=float(lam),
                half_life=float(hl),
                r_squared=float(r2),
            )
        except Exception:
            return DecayModel(
                ic_at_zero=float("nan"),
                decay_rate=float("nan"),
                half_life=float("nan"),
                r_squared=float("nan"),
            )

    # ------------------------------------------------------------------ #
    # Rolling half-life
    # ------------------------------------------------------------------ #

    def rolling_halflife(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        window: int = 120,
        max_horizon: int = 20,
        method: str = "spearman",
    ) -> pd.Series:
        """Compute rolling half-life of IC decay over time.

        For each rolling window of *window* bars, compute the IC decay curve
        and extract its half-life.

        Parameters
        ----------
        signal_df    : DataFrame[time * assets] signals
        returns_df   : DataFrame[time * assets] returns
        window       : rolling window length in bars
        max_horizon  : maximum decay horizon
        method       : IC correlation method

        Returns
        -------
        pd.Series[time -> half_life_in_bars]
        """
        cols = signal_df.columns.intersection(returns_df.columns)
        idx = signal_df.index.intersection(returns_df.index)
        sig = signal_df.loc[idx, cols]
        ret = returns_df.loc[idx, cols]

        hl_vals: list[float] = []
        hl_idx: list = []

        calc = ICCalculator()
        for i in range(window - 1, len(idx)):
            window_idx = idx[i - window + 1 : i + 1]
            sig_w = sig.loc[window_idx]
            ret_w = ret.loc[window_idx]

            try:
                decay_result = calc.ic_decay(sig_w, ret_w, max_horizon=max_horizon, method=method)
                hl = self.compute_signal_halflife(decay_result)
            except Exception:
                hl = float("nan")

            hl_vals.append(hl)
            hl_idx.append(idx[i])

        return pd.Series(hl_vals, index=hl_idx, name="rolling_halflife_bars")

    # ------------------------------------------------------------------ #
    # Turnover analysis
    # ------------------------------------------------------------------ #

    def turnover_analysis(
        self,
        signal_df: pd.DataFrame,
        holding_period: int,
        cost_per_trade: float = 0.0002,
    ) -> TurnoverStats:
        """Compute signal turnover statistics.

        Turnover is measured as the average fraction of the signal portfolio
        that changes between rebalances.

        Parameters
        ----------
        signal_df      : DataFrame[time * assets] of signal values
        holding_period : target holding period (bars) — determines rebalance freq
        cost_per_trade : one-way transaction cost fraction

        Returns
        -------
        TurnoverStats
        """
        # Rank signals each period
        ranked = signal_df.rank(axis=1, pct=True)  # Rank as percentile

        # Compute turnover: sum of absolute changes in rank
        delta = ranked.diff().abs()
        daily_turnover = float(delta.mean().mean())  # avg per bar per asset

        # Compute signal autocorrelation
        acf_series = self._signal_acf(signal_df, max_lag=max(holding_period, 20))
        acf_lag1 = float(acf_series.iloc[0]) if len(acf_series) > 0 else float("nan")

        avg_holding = 1.0 / daily_turnover if daily_turnover > 0 else float("inf")

        return TurnoverStats(
            holding_period=holding_period,
            daily_turnover=daily_turnover,
            weekly_turnover=daily_turnover * 5,
            monthly_turnover=daily_turnover * 21,
            signal_autocorr_lag1=acf_lag1,
            signal_autocorr_full=acf_series,
            avg_holding_bars=avg_holding,
            round_trip_cost=2 * cost_per_trade * daily_turnover,
        )

    def _signal_acf(self, signal_df: pd.DataFrame, max_lag: int) -> pd.Series:
        """Compute ACF of signal time-series (averaged across assets)."""
        acf_per_asset: list[np.ndarray] = []
        for col in signal_df.columns:
            s = signal_df[col].dropna().values
            n = len(s)
            if n < max_lag + 5:
                continue
            mean_s = s.mean()
            var_s = np.var(s, ddof=1)
            if var_s == 0:
                continue
            acfs = []
            for lag in range(1, max_lag + 1):
                cov = np.mean((s[: n - lag] - mean_s) * (s[lag:] - mean_s))
                acfs.append(cov / var_s)
            acf_per_asset.append(np.array(acfs))

        if not acf_per_asset:
            return pd.Series(dtype=float)

        mean_acf = np.nanmean(acf_per_asset, axis=0)
        return pd.Series(mean_acf, index=range(1, max_lag + 1), name="signal_acf")

    # ------------------------------------------------------------------ #
    # Transaction-cost adjusted IC
    # ------------------------------------------------------------------ #

    def transaction_cost_adjusted_ic(
        self,
        ic_series: pd.Series,
        turnover: float,
        cost_per_trade: float = 0.0002,
    ) -> float:
        """Compute IC adjusted for transaction cost drag.

        adj_IC = mean(IC) - 2 * cost_per_trade * turnover / std(IC)

        Parameters
        ----------
        ic_series    : time-series of IC values
        turnover     : daily turnover fraction (from TurnoverStats.daily_turnover)
        cost_per_trade: one-way cost fraction

        Returns
        -------
        float — transaction-cost-adjusted IC
        """
        clean = ic_series.dropna()
        if len(clean) < 2:
            return float("nan")
        mean_ic = float(clean.mean())
        std_ic = float(clean.std(ddof=1))
        if std_ic == 0:
            return float("nan")
        round_trip_cost = 2 * cost_per_trade * turnover
        adj_ic = mean_ic - round_trip_cost / std_ic
        return float(adj_ic)

    # ------------------------------------------------------------------ #
    # Half-life from trades
    # ------------------------------------------------------------------ #

    def halflife_from_trades(
        self,
        trades: pd.DataFrame,
        signal_col: str = "ensemble_signal",
        hold_col: str = "hold_bars",
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        max_horizon: int = 20,
    ) -> float:
        """Estimate half-life from a flat trades DataFrame.

        Delegates to ICCalculator.ic_decay_from_trades() then extracts HL.

        Parameters
        ----------
        trades       : flat trade records DataFrame
        signal_col   : signal column
        hold_col     : column for hold duration in bars
        return_col   : P&L column
        max_horizon  : maximum horizon to test

        Returns
        -------
        float half-life in bars
        """
        calc = ICCalculator()
        decay_result = calc.ic_decay_from_trades(
            trades,
            signal_col=signal_col,
            return_col=return_col,
            hold_col=hold_col,
            dollar_pos_col=dollar_pos_col,
            max_horizon=max_horizon,
        )
        return self.compute_signal_halflife(decay_result)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def decay_summary(
        self,
        ic_decay_result: ICDecayResult,
        transaction_cost: float = 0.0002,
    ) -> Dict[str, float]:
        """Return a dict summarising alpha decay diagnostics.

        Keys: half_life, decay_rate, ic_at_zero, r_squared,
              optimal_holding, peak_ic, peak_horizon
        """
        decay_model = self.signal_decay_model(ic_decay_result)
        opt_hold = self.optimal_holding_period(ic_decay_result, transaction_cost)
        return {
            "half_life_bars": decay_model.half_life,
            "decay_rate": decay_model.decay_rate,
            "ic_at_zero": decay_model.ic_at_zero,
            "model_r_squared": decay_model.r_squared,
            "optimal_holding_bars": float(opt_hold),
            "peak_ic": ic_decay_result.peak_ic,
            "peak_horizon": float(ic_decay_result.peak_horizon),
        }

    # ------------------------------------------------------------------ #
    # Visualisations
    # ------------------------------------------------------------------ #

    def plot_alpha_decay(
        self,
        decay_result: ICDecayResult,
        save_path: Optional[str | Path] = None,
        transaction_cost: float = 0.0002,
        title: str = "Alpha Decay Analysis",
    ) -> plt.Figure:
        """Multi-panel alpha decay figure.

        Panel 1: IC vs horizon (bar chart) with fitted decay curve
        Panel 2: Net IC (after transaction costs) vs horizon
        Panel 3: Cumulative IC sum vs horizon

        Parameters
        ----------
        decay_result     : ICDecayResult
        save_path        : optional path to save PNG
        transaction_cost : one-way cost used for net IC panel
        title            : figure title
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        h_arr = np.array(decay_result.horizons, dtype=float)
        ic_arr = np.array(decay_result.ic_values, dtype=float)

        # Panel 1: IC decay
        ax = axes[0]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ic_arr]
        ax.bar(h_arr, ic_arr, color=colors, alpha=0.75)
        if not np.isnan(decay_result.decay_rate):
            h_fine = np.linspace(1, h_arr[-1], 300)
            ic_fit = decay_result.ic_at_zero * np.exp(-decay_result.decay_rate * h_fine)
            ax.plot(h_fine, ic_fit, "k--", linewidth=1.5,
                    label=f"lambda_={decay_result.decay_rate:.3f}\nHL={decay_result.half_life:.1f}b")
            ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Horizon (bars)")
        ax.set_ylabel("IC")
        ax.set_title("IC Decay")

        # Panel 2: Net IC after costs
        ax2 = axes[1]
        opt_h = self.optimal_holding_period(decay_result, transaction_cost)
        net_ic = ic_arr * np.sqrt(h_arr) - 2 * transaction_cost
        net_ic_clean = np.where(np.isnan(net_ic), 0, net_ic)
        ax2.plot(h_arr, net_ic_clean, color="#9b59b6", linewidth=1.5, marker="o", markersize=3)
        ax2.axvline(opt_h, color="#e74c3c", linestyle="--", linewidth=1.2,
                    label=f"Optimal h={opt_h}")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Horizon (bars)")
        ax2.set_ylabel("IC * sqrth - cost")
        ax2.set_title("Net IC After Costs")
        ax2.legend(fontsize=8)

        # Panel 3: Cumulative IC
        ax3 = axes[2]
        cum_ic = np.nancumsum(ic_arr)
        ax3.plot(h_arr, cum_ic, color="#e67e22", linewidth=1.5, marker="o", markersize=3)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_xlabel("Horizon (bars)")
        ax3.set_ylabel("Cumulative IC")
        ax3.set_title("Cumulative IC")

        fig.suptitle(title, fontsize=12, y=1.01)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_halflife_timeline(
        self,
        rolling_halflife: pd.Series,
        save_path: Optional[str | Path] = None,
        title: str = "Rolling Alpha Half-Life",
    ) -> plt.Figure:
        """Time-series of rolling half-life estimates.

        Parameters
        ----------
        rolling_halflife : pd.Series[time -> half-life in bars]
        save_path        : optional save path
        title            : figure title
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        clean = rolling_halflife.replace([np.inf, -np.inf], np.nan).dropna()

        ax.plot(clean.index, clean.values, color="#3498db", linewidth=1.0, label="Half-life")

        mu = clean.mean()
        sigma = clean.std(ddof=1)
        ax.axhline(mu, color="#e67e22", linestyle="--", linewidth=1.2,
                   label=f"Mean={mu:.1f} bars")
        ax.fill_between(clean.index, mu - sigma, mu + sigma, alpha=0.15, color="#e67e22")

        ax.set_xlabel("Date")
        ax.set_ylabel("Half-life (bars)")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_turnover_breakdown(
        self,
        turnover_stats: TurnoverStats,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Bar chart of daily/weekly/monthly turnover and signal ACF."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Turnover bars
        categories = ["Daily", "Weekly", "Monthly"]
        values = [
            turnover_stats.daily_turnover,
            turnover_stats.weekly_turnover,
            turnover_stats.monthly_turnover,
        ]
        ax1.bar(categories, values, color=["#3498db", "#2ecc71", "#e67e22"], alpha=0.8)
        ax1.set_ylabel("Turnover (fraction)")
        ax1.set_title(
            f"Signal Turnover  |  AvgHold={turnover_stats.avg_holding_bars:.1f} bars\n"
            f"RoundTrip Cost={turnover_stats.round_trip_cost:.4%}/bar"
        )

        # Signal ACF
        acf = turnover_stats.signal_autocorr_full
        if len(acf) > 0:
            ax2.bar(acf.index, acf.values, color="#9b59b6", alpha=0.7)
            ax2.axhline(0, color="black", linewidth=0.8)
            ax2.axhline(1.96 / np.sqrt(len(acf)), color="gray", linestyle="--", linewidth=0.8)
            ax2.axhline(-1.96 / np.sqrt(len(acf)), color="gray", linestyle="--", linewidth=0.8)
            ax2.set_xlabel("Lag (bars)")
            ax2.set_ylabel("ACF")
            ax2.set_title(f"Signal ACF  |  Lag-1={turnover_stats.signal_autocorr_lag1:.3f}")

        fig.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # Holding-period optimisation: sensitivity analysis
    # ------------------------------------------------------------------ #

    def holding_period_cost_sensitivity(
        self,
        ic_decay_result: ICDecayResult,
        cost_range: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Optimal holding period as a function of transaction cost.

        Parameters
        ----------
        ic_decay_result : ICDecayResult
        cost_range      : list of one-way costs to test (default log-spaced)

        Returns
        -------
        pd.DataFrame[cost -> optimal_holding_bars]
        """
        if cost_range is None:
            cost_range = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]

        records: list[dict] = []
        for cost in cost_range:
            opt_h = self.optimal_holding_period(ic_decay_result, transaction_cost=cost)
            records.append({"cost": cost, "optimal_holding_bars": opt_h})
        return pd.DataFrame(records).set_index("cost")

    # ------------------------------------------------------------------ #
    # Multi-signal decay comparison
    # ------------------------------------------------------------------ #

    def compare_signal_halflives(
        self,
        trades: pd.DataFrame,
        signal_cols: List[str],
        hold_col: str = "hold_bars",
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        max_horizon: int = 20,
    ) -> pd.Series:
        """Compare half-lives across multiple signal columns.

        Parameters
        ----------
        trades      : trade records
        signal_cols : list of signal column names
        hold_col    : column for hold duration
        return_col  : P&L column
        dollar_pos_col: position column
        max_horizon : max decay horizon

        Returns
        -------
        pd.Series[signal_col -> half_life_bars]
        """
        result: dict[str, float] = {}
        calc = ICCalculator()
        for col in signal_cols:
            if col not in trades.columns:
                result[col] = float("nan")
                continue
            try:
                decay = calc.ic_decay_from_trades(
                    trades, signal_col=col, return_col=return_col,
                    hold_col=hold_col, dollar_pos_col=dollar_pos_col,
                    max_horizon=max_horizon,
                )
                result[col] = self.compute_signal_halflife(decay)
            except Exception:
                result[col] = float("nan")
        return pd.Series(result, name="half_life_bars")

    # ------------------------------------------------------------------ #
    # Net IC after costs — full curve
    # ------------------------------------------------------------------ #

    def net_ic_curve(
        self,
        ic_decay_result: ICDecayResult,
        transaction_cost: float = 0.0002,
        turnover_per_bar: float = 1.0,
    ) -> pd.Series:
        """Compute net IC at each horizon after transaction cost adjustment.

        net_IC(h) = IC(h) * sqrt(h) - 2 * cost * turnover_per_bar

        Parameters
        ----------
        ic_decay_result  : ICDecayResult
        transaction_cost : one-way cost fraction
        turnover_per_bar : fraction of portfolio turned per bar

        Returns
        -------
        pd.Series[horizon -> net_IC]
        """
        horizons = np.array(ic_decay_result.horizons, dtype=float)
        ic_vals = np.array(ic_decay_result.ic_values, dtype=float)
        round_trip = 2 * transaction_cost * turnover_per_bar
        net = ic_vals * np.sqrt(horizons) - round_trip
        return pd.Series(net, index=ic_decay_result.horizons, name="net_IC")

    # ------------------------------------------------------------------ #
    # Decay model forecast
    # ------------------------------------------------------------------ #

    def forecast_ic(
        self,
        decay_model: DecayModel,
        horizons: Optional[List[int]] = None,
    ) -> pd.Series:
        """Forecast IC at future horizons using the fitted decay model.

        Parameters
        ----------
        decay_model : DecayModel from signal_decay_model()
        horizons    : list of horizons to forecast (default 1..30)

        Returns
        -------
        pd.Series[horizon -> forecast_IC]
        """
        if horizons is None:
            horizons = list(range(1, 31))
        values = [decay_model.ic_at_horizon(float(h)) for h in horizons]
        return pd.Series(values, index=horizons, name="forecast_IC")

    # ------------------------------------------------------------------ #
    # AR(p) decay model
    # ------------------------------------------------------------------ #

    def ar_decay_model(
        self,
        ic_decay_result: ICDecayResult,
        p: int = 2,
    ) -> Dict[str, float]:
        """Fit an AR(p) model to the IC decay curve.

        More flexible than exponential: captures oscillations or
        non-monotone decay patterns.

        Parameters
        ----------
        ic_decay_result : ICDecayResult
        p               : AR order

        Returns
        -------
        Dict with keys: coefficients (list), r_squared, aic, bic
        """
        ic = np.array(ic_decay_result.ic_values, dtype=float)
        mask = ~np.isnan(ic)
        if mask.sum() < p + 2:
            return {"coefficients": [], "r_squared": float("nan"), "aic": float("nan"), "bic": float("nan")}

        ic_clean = ic[mask]
        n = len(ic_clean)

        # Build design matrix for AR(p)
        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        for t in range(p, n):
            X_rows.append(ic_clean[t - p : t][::-1])
            y_rows.append(ic_clean[t])

        X = np.array(X_rows)
        y = np.array(y_rows)
        X_int = np.column_stack([np.ones(len(y)), X])

        beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
        y_hat = X_int @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # AIC / BIC
        k = p + 1
        n_eff = len(y)
        sigma2 = ss_res / n_eff
        ll = -n_eff / 2 * (np.log(2 * np.pi * sigma2) + 1)
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(n_eff)

        return {
            "coefficients": beta[1:].tolist(),
            "intercept": float(beta[0]),
            "r_squared": float(r2),
            "aic": float(aic),
            "bic": float(bic),
        }

    # ------------------------------------------------------------------ #
    # Decay stability across sub-periods
    # ------------------------------------------------------------------ #

    def decay_stability_across_periods(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        hold_col: str = "hold_bars",
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        max_horizon: int = 20,
        n_periods: int = 4,
    ) -> pd.DataFrame:
        """Estimate decay parameters in each sub-period.

        Tests whether alpha decay is stable over time.

        Parameters
        ----------
        trades      : trade records with exit_time
        signal_col  : signal column
        n_periods   : number of equal-length time periods

        Returns
        -------
        pd.DataFrame[period x (half_life, decay_rate, ic_at_zero, r_squared)]
        """
        df = trades.copy()
        if "exit_time" in df.columns:
            df = df.sort_values("exit_time")

        period_size = max(1, len(df) // n_periods)
        calc = ICCalculator()
        records: list[dict] = []

        for i in range(n_periods):
            sub = df.iloc[i * period_size : (i + 1) * period_size]
            try:
                decay = calc.ic_decay_from_trades(
                    sub, signal_col=signal_col, return_col=return_col,
                    hold_col=hold_col, dollar_pos_col=dollar_pos_col,
                    max_horizon=max_horizon,
                )
                model = self.signal_decay_model(decay)
                records.append({
                    "period": i + 1,
                    "n_trades": len(sub),
                    "half_life": model.half_life,
                    "decay_rate": model.decay_rate,
                    "ic_at_zero": model.ic_at_zero,
                    "r_squared": model.r_squared,
                })
            except Exception:
                records.append({
                    "period": i + 1, "n_trades": len(sub),
                    "half_life": float("nan"), "decay_rate": float("nan"),
                    "ic_at_zero": float("nan"), "r_squared": float("nan"),
                })

        return pd.DataFrame(records).set_index("period")

    # ------------------------------------------------------------------ #
    # Plot decay stability
    # ------------------------------------------------------------------ #

    def plot_decay_stability(
        self,
        stability_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Plot half-life and decay-rate stability across sub-periods.

        Parameters
        ----------
        stability_df : output of decay_stability_across_periods()
        save_path    : optional save path
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        hl = stability_df["half_life"].replace([float("inf")], float("nan")).dropna()
        ax1.bar(hl.index, hl.values, color="#3498db", alpha=0.8)
        ax1.axhline(hl.mean(), color="#e74c3c", linestyle="--", linewidth=1.2,
                    label=f"Mean={hl.mean():.1f}")
        ax1.set_xlabel("Period")
        ax1.set_ylabel("Half-life (bars)")
        ax1.set_title("Half-life by Sub-period")
        ax1.legend()

        ax2 = axes[1]
        dr = stability_df["decay_rate"].dropna()
        ax2.bar(dr.index, dr.values, color="#e74c3c", alpha=0.8)
        ax2.axhline(dr.mean(), color="#3498db", linestyle="--", linewidth=1.2,
                    label=f"Mean={dr.mean():.4f}")
        ax2.set_xlabel("Period")
        ax2.set_ylabel("Decay Rate")
        ax2.set_title("Decay Rate by Sub-period")
        ax2.legend()

        fig.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------
    # Extended decay analytics
    # ------------------------------------------------------------------

    def grinold_kahn_optimal_period(
        self,
        ic_series: "pd.Series",
        cost_per_bar: float = 0.0002,
        max_horizon: int = 60,
    ) -> dict:
        """
        Estimate the optimal holding period using the Grinold-Kahn framework:
            net_value(h) = IC0 * exp(-lambda * h) * sqrt(h) - cost_per_bar * h

        The optimal h* maximises net_value(h).

        Parameters
        ----------
        ic_series : pd.Series
            Time series of rolling IC values.
        cost_per_bar : float
            Transaction cost per holding period bar.
        max_horizon : int
            Maximum horizon to search.

        Returns
        -------
        dict with keys: ic0, decay_lambda, halflife, optimal_h,
                        net_value_at_optimal, h_range, net_value_curve.
        """
        from scipy.optimize import curve_fit

        arr = np.asarray(ic_series.dropna(), dtype=float)
        if len(arr) < 5:
            raise ValueError("Insufficient IC data for Grinold-Kahn optimisation.")

        ic0 = float(arr.mean())
        if ic0 == 0:
            return {"ic0": 0.0, "decay_lambda": 0.0, "halflife": np.inf,
                    "optimal_h": 1, "net_value_at_optimal": 0.0,
                    "h_range": [], "net_value_curve": []}

        lags = np.arange(1, min(len(arr) // 2, max_horizon + 1))
        auto_corrs: list[float] = []
        for lag in lags:
            if lag < len(arr):
                rho = float(np.corrcoef(arr[:-lag], arr[lag:])[0, 1])
                auto_corrs.append(rho if np.isfinite(rho) else 0.0)
            else:
                break

        lags = lags[:len(auto_corrs)]
        ac = np.array(auto_corrs)

        try:
            def exp_model(h, lam):
                return np.exp(-lam * h)
            popt, _ = curve_fit(exp_model, lags, ac, p0=[0.1],
                                maxfev=500, bounds=(0, np.inf))
            decay_lambda = float(popt[0])
        except Exception:
            decay_lambda = 0.05

        h_range = np.arange(1, max_horizon + 1)
        net_values = (ic0 * np.exp(-decay_lambda * h_range) * np.sqrt(h_range)
                      - cost_per_bar * h_range)
        optimal_idx = int(np.argmax(net_values))
        optimal_h = int(h_range[optimal_idx])

        return {
            "ic0": ic0,
            "decay_lambda": decay_lambda,
            "halflife": float(np.log(2) / decay_lambda) if decay_lambda > 0 else np.inf,
            "optimal_h": optimal_h,
            "net_value_at_optimal": float(net_values[optimal_idx]),
            "h_range": h_range.tolist(),
            "net_value_curve": net_values.tolist(),
        }

    def signal_persistence_matrix(
        self,
        trades: "pd.DataFrame",
        signal_col: str = "delta_score",
        lags: list[int] | None = None,
    ) -> "pd.DataFrame":
        """
        Autocorrelation of the signal at multiple lags to characterise
        signal memory / persistence.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade log sorted by exit_time.
        signal_col : str
            Signal column name.
        lags : list of int, optional
            Lags to compute (default [1, 2, 5, 10, 20]).

        Returns
        -------
        pd.DataFrame with columns [lag, autocorr, p_value, halflife_est].
        """
        if lags is None:
            lags = [1, 2, 5, 10, 20]

        df = trades.copy()
        if "exit_time" in df.columns:
            df = df.sort_values("exit_time")
        sig = df[signal_col].dropna().values.astype(float)
        n = len(sig)

        from scipy import stats as _stats
        rows: list[dict] = []
        for lag in lags:
            if lag >= n:
                continue
            rho, pval = _stats.pearsonr(sig[:-lag], sig[lag:])
            if rho > 0:
                lam = -np.log(rho) / lag
                hl = float(np.log(2) / lam) if lam > 0 else np.inf
            else:
                hl = np.nan
            rows.append({
                "lag": lag,
                "autocorr": float(rho),
                "p_value": float(pval),
                "halflife_est": hl if np.isfinite(hl) else np.nan,
            })

        import pandas as pd
        return pd.DataFrame(rows)

    def ic_decay_with_confidence(
        self,
        ic_series: "pd.Series",
        max_horizon: int = 20,
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
        seed: int = 42,
    ) -> "pd.DataFrame":
        """
        Compute IC decay curve with bootstrap confidence intervals.

        Parameters
        ----------
        ic_series : pd.Series
            Rolling IC time series.
        max_horizon : int
            Maximum horizon.
        n_bootstrap : int
            Bootstrap resamples.
        ci_level : float
            Confidence level.
        seed : int
            RNG seed.

        Returns
        -------
        pd.DataFrame with columns [horizon, ic_mean, ci_lower, ci_upper, ic_std].
        """
        import pandas as pd
        from scipy.stats import spearmanr

        rng = np.random.default_rng(seed)
        arr = np.asarray(ic_series.dropna(), dtype=float)
        n = len(arr)
        alpha = 1.0 - ci_level

        rows: list[dict] = []
        for h in range(1, min(max_horizon + 1, n // 2)):
            x = arr[:-h]
            y = arr[h:]
            if len(x) < 5:
                break
            rho, _ = spearmanr(x, y)

            boot_rhos: list[float] = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(x), size=len(x))
                br, _ = spearmanr(x[idx], y[idx])
                if np.isfinite(br):
                    boot_rhos.append(float(br))

            if boot_rhos:
                ci_lo = float(np.percentile(boot_rhos, alpha / 2 * 100))
                ci_hi = float(np.percentile(boot_rhos, (1 - alpha / 2) * 100))
                ic_std = float(np.std(boot_rhos))
            else:
                ci_lo = ci_hi = ic_std = np.nan

            rows.append({
                "horizon": h,
                "ic_mean": float(rho),
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "ic_std": ic_std,
            })

        return pd.DataFrame(rows)

    def plot_grinold_kahn_curve(
        self,
        gk_result: dict,
        save_path=None,
        title: str = "Grinold-Kahn Optimal Holding Period",
    ):
        """
        Plot the Grinold-Kahn net-value curve highlighting the optimal period.

        Parameters
        ----------
        gk_result : output of grinold_kahn_optimal_period()
        save_path : optional save path
        title     : chart title

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        h_range = gk_result.get("h_range", [])
        nv = gk_result.get("net_value_curve", [])
        opt_h = gk_result.get("optimal_h", 1)
        ic0 = gk_result.get("ic0", 0.0)
        decay_lam = gk_result.get("decay_lambda", 0.0)
        hl = gk_result.get("halflife", np.inf)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(h_range, nv, color="#2980b9", linewidth=2, label="Net value(h)")
        ax.axvline(opt_h, color="#e74c3c", linewidth=1.5, linestyle="--",
                   label=f"Optimal h*={opt_h}")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

        ic_curve = ic0 * np.exp(-decay_lam * np.array(h_range))
        ax.plot(h_range, ic_curve, color="#27ae60", linewidth=1.5,
                linestyle="--", label=f"IC decay (half-life={hl:.1f})")

        ax.set_xlabel("Holding Period (bars)")
        ax.set_ylabel("Net Value / IC")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_ic_decay_with_ci(
        self,
        decay_ci_df: "pd.DataFrame",
        save_path=None,
        title: str = "IC Decay with Bootstrap CI",
    ):
        """
        Plot IC decay curve with shaded bootstrap confidence band.

        Parameters
        ----------
        decay_ci_df : output of ic_decay_with_confidence()
        save_path   : optional save path
        title       : chart title

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        df = decay_ci_df
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df["horizon"], df["ic_mean"], color="#2980b9", linewidth=2, label="IC mean")
        if "ci_lower" in df.columns and "ci_upper" in df.columns:
            ax.fill_between(df["horizon"], df["ci_lower"], df["ci_upper"],
                            alpha=0.3, color="#2980b9", label="95% CI")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Horizon (bars)")
        ax.set_ylabel("IC")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig
