"""
Event-driven macro strategies: FOMC, NFP, CPI.

Implements:
- FOMCDrift    : post-FOMC announcement drift (equity/bond reaction)
- NFPMomentum  : non-farm payrolls surprise momentum
- CPIRegime    : CPI-driven inflation regime switching
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    n_trades: int
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    extra: Dict = field(default_factory=dict)


def _stats(returns: pd.Series, freq: int = 252) -> Dict:
    r = returns.dropna()
    if len(r) == 0:
        return {k: np.nan for k in [
            "total_return", "cagr", "sharpe", "sortino",
            "max_drawdown", "calmar", "win_rate", "profit_factor",
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
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    pf = float(gross_profit / (gross_loss + 1e-12))
    return {
        "total_return": total, "cagr": cagr, "sharpe": sr, "sortino": sortino,
        "max_drawdown": mdd, "calmar": calmar, "win_rate": win_rate,
        "profit_factor": pf,
    }


def _build_equity(port_returns: pd.Series) -> pd.Series:
    return (1 + port_returns.dropna()).cumprod()


# ---------------------------------------------------------------------------
# FOMCDrift
# ---------------------------------------------------------------------------

class FOMCDrift:
    """
    Post-FOMC announcement drift strategy.

    Historically, equity markets exhibit a drift in the direction of the
    initial reaction to FOMC statements.  This strategy enters a position
    in the direction of the intraday FOMC reaction and holds for
    drift_period trading days.

    Parameters
    ----------
    drift_period : int
        Trading days to hold after the FOMC announcement.
    entry_delay : int
        Days after announcement to wait before entering (0 = same day close).
    min_reaction : float
        Minimum absolute intraday price reaction to trade.
    use_vix_filter : bool
        If True, skip trade when VIX is above vix_threshold (high uncertainty).
    vix_threshold : float
        VIX level above which trades are skipped.
    fade_after : int
        Days after which position begins to fade (linear decay to zero by
        drift_period).
    """

    def __init__(
        self,
        drift_period: int = 5,
        entry_delay: int = 0,
        min_reaction: float = 0.003,
        use_vix_filter: bool = True,
        vix_threshold: float = 30.0,
        fade_after: int = 3,
    ) -> None:
        self.drift_period = drift_period
        self.entry_delay = entry_delay
        self.min_reaction = min_reaction
        self.use_vix_filter = use_vix_filter
        self.vix_threshold = vix_threshold
        self.fade_after = fade_after

    def compute_fomc_reaction(
        self,
        price: pd.Series,
        fomc_dates: pd.DatetimeIndex,
        intraday_open: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Estimate the intraday FOMC reaction.

        If intraday_open is provided, reaction = (close - open) / open on the
        FOMC date.  Otherwise uses (close_t - close_{t-1}) / close_{t-1}.

        Returns
        -------
        pd.Series
            Reaction values indexed by FOMC date.
        """
        reactions = {}
        daily_ret = price.pct_change()
        for dt in fomc_dates:
            if dt not in price.index:
                # Find nearest date
                future = price.index[price.index >= dt]
                if len(future) == 0:
                    continue
                dt = future[0]
            if intraday_open is not None and dt in intraday_open.index:
                reaction = price.loc[dt] / intraday_open.loc[dt] - 1
            else:
                if dt in daily_ret.index:
                    reaction = float(daily_ret.loc[dt])
                else:
                    continue
            reactions[dt] = reaction
        return pd.Series(reactions, name="fomc_reaction")

    def generate_signals(
        self,
        price: pd.Series,
        fomc_dates: pd.DatetimeIndex,
        vix: Optional[pd.Series] = None,
        intraday_open: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Build a daily position signal based on FOMC reactions.

        Returns
        -------
        pd.Series
            Position size in [-1, 1].
        """
        reactions = self.compute_fomc_reaction(price, fomc_dates, intraday_open)
        signal = pd.Series(0.0, index=price.index)

        for ann_dt, reaction in reactions.items():
            if abs(reaction) < self.min_reaction:
                continue

            # VIX filter
            if self.use_vix_filter and vix is not None and ann_dt in vix.index:
                if float(vix.loc[ann_dt]) > self.vix_threshold:
                    continue

            direction = np.sign(reaction)
            magnitude = min(abs(reaction) / self.min_reaction, 3.0)

            future = price.index[price.index >= ann_dt]
            if len(future) <= self.entry_delay:
                continue
            entry_dt = future[self.entry_delay]
            entry_idx = price.index.get_loc(entry_dt)
            end_idx = min(entry_idx + self.drift_period, len(price.index))

            for j in range(entry_idx, end_idx):
                days_in = j - entry_idx
                if days_in >= self.fade_after:
                    fade = 1.0 - (days_in - self.fade_after) / max(
                        self.drift_period - self.fade_after, 1
                    )
                    fade = max(fade, 0.0)
                else:
                    fade = 1.0
                new_val = direction * magnitude * fade
                if abs(new_val) > abs(signal.iloc[j]):
                    signal.iloc[j] = new_val

        # Clip to [-1, 1]
        signal = signal.clip(-1, 1)
        return signal

    def backtest(
        self,
        price: pd.Series,
        fomc_dates: pd.DatetimeIndex,
        vix: Optional[pd.Series] = None,
        intraday_open: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Full backtest of FOMC drift strategy."""
        daily_ret = price.pct_change()
        signal = self.generate_signals(price, fomc_dates, vix, intraday_open)
        # Position taken at day t, return realized at t+1
        port_returns = (signal.shift(1) * daily_ret).dropna()
        eq = _build_equity(port_returns)
        s = _stats(port_returns)

        reactions = self.compute_fomc_reaction(price, fomc_dates, intraday_open)
        n_trades = int((reactions.abs() >= self.min_reaction).sum())

        return BacktestResult(
            total_return=s["total_return"],
            cagr=s["cagr"],
            sharpe=s["sharpe"],
            sortino=s["sortino"],
            max_drawdown=s["max_drawdown"],
            calmar=s["calmar"],
            win_rate=s["win_rate"],
            profit_factor=s["profit_factor"],
            n_trades=n_trades,
            equity_curve=eq,
            returns=port_returns,
            extra={"reactions": reactions, "signal": signal},
        )

    def event_study(
        self,
        price: pd.Series,
        fomc_dates: pd.DatetimeIndex,
        window_before: int = 3,
        window_after: int = 10,
        intraday_open: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute average cumulative returns around FOMC dates, split by
        direction of initial reaction.

        Returns
        -------
        pd.DataFrame
            Columns: 'positive_reaction', 'negative_reaction', 'all_events'
            Index: relative day (-window_before to +window_after).
        """
        reactions = self.compute_fomc_reaction(price, fomc_dates, intraday_open)
        daily_ret = price.pct_change()

        days = list(range(-window_before, window_after + 1))
        pos_cumrets: List[List[float]] = []
        neg_cumrets: List[List[float]] = []

        for ann_dt, reaction in reactions.items():
            if ann_dt not in price.index:
                continue
            ann_idx = price.index.get_loc(ann_dt)
            window_start = ann_idx - window_before
            window_end = ann_idx + window_after + 1

            if window_start < 0 or window_end > len(price.index):
                continue

            ret_window = daily_ret.iloc[window_start:window_end].values
            cum = np.cumprod(1 + ret_window) - 1
            # Re-index so that FOMC day = 0 corresponds to element window_before
            base_val = cum[window_before - 1] if window_before > 0 else 0.0
            cum_adj = cum - base_val

            if reaction > 0:
                pos_cumrets.append(list(cum_adj))
            else:
                neg_cumrets.append(list(cum_adj))

        result_df = pd.DataFrame(index=days)
        if pos_cumrets:
            result_df["positive_reaction"] = np.mean(pos_cumrets, axis=0)
        if neg_cumrets:
            result_df["negative_reaction"] = np.mean(neg_cumrets, axis=0)

        all_cumrets = pos_cumrets + neg_cumrets
        if all_cumrets:
            result_df["all_events"] = np.mean(all_cumrets, axis=0)

        return result_df

    def fomc_statistics(
        self,
        price: pd.Series,
        fomc_dates: pd.DatetimeIndex,
        intraday_open: Optional[pd.Series] = None,
    ) -> Dict:
        """Summary statistics of FOMC reactions and subsequent drift."""
        reactions = self.compute_fomc_reaction(price, fomc_dates, intraday_open)
        daily_ret = price.pct_change()

        drift_returns = []
        for ann_dt, reaction in reactions.items():
            if ann_dt not in price.index:
                continue
            ann_idx = price.index.get_loc(ann_dt)
            end_idx = min(ann_idx + self.drift_period + 1, len(price.index))
            drift_ret = (
                price.iloc[end_idx - 1] / price.iloc[ann_idx] - 1
                if end_idx > ann_idx else np.nan
            )
            drift_returns.append(np.sign(reaction) * drift_ret)

        drift_arr = np.array([d for d in drift_returns if not np.isnan(d)])
        return {
            "n_events": len(reactions),
            "mean_reaction": round(reactions.mean(), 4),
            "std_reaction": round(reactions.std(), 4),
            "pct_positive": round((reactions > 0).mean(), 4),
            "mean_drift": round(drift_arr.mean(), 4) if len(drift_arr) > 0 else np.nan,
            "drift_hit_rate": round((drift_arr > 0).mean(), 4) if len(drift_arr) > 0 else np.nan,
        }


# ---------------------------------------------------------------------------
# NFPMomentum
# ---------------------------------------------------------------------------

class NFPMomentum:
    """
    Non-Farm Payrolls surprise momentum strategy.

    The NFP surprise (actual - consensus) drives USD and equity momentum.
    A positive surprise (strong employment) → long equities / long USD.
    A negative surprise → short equities / short USD.

    Parameters
    ----------
    holding_period : int
        Trading days to hold after NFP release.
    min_surprise_std : float
        Minimum std of NFP surprise series to compute z-score.
    z_threshold : float
        Minimum |z-score| of surprise to trade.
    lookback : int
        Rolling window (months) to estimate surprise mean and std.
    momentum_window : int
        Number of prior NFP surprises to compute momentum signal.
    blend_momentum : bool
        If True, blend point-in-time surprise with momentum of recent surprises.
    momentum_weight : float
        Weight on momentum component (1 - momentum_weight on spot surprise).
    """

    def __init__(
        self,
        holding_period: int = 10,
        min_surprise_std: float = 10.0,
        z_threshold: float = 0.5,
        lookback: int = 12,
        momentum_window: int = 3,
        blend_momentum: bool = True,
        momentum_weight: float = 0.3,
    ) -> None:
        self.holding_period = holding_period
        self.min_surprise_std = min_surprise_std
        self.z_threshold = z_threshold
        self.lookback = lookback
        self.momentum_window = momentum_window
        self.blend_momentum = blend_momentum
        self.momentum_weight = momentum_weight

    def compute_surprise(
        self,
        actual_nfp: pd.Series,
        consensus_nfp: pd.Series,
    ) -> pd.Series:
        """
        Compute standardized NFP surprise.

        Parameters
        ----------
        actual_nfp, consensus_nfp : pd.Series
            Monthly NFP series indexed by release date.

        Returns
        -------
        pd.Series
            Z-score of NFP surprise.
        """
        raw = actual_nfp - consensus_nfp
        rolling_std = raw.rolling(self.lookback, min_periods=3).std()
        rolling_std = rolling_std.clip(lower=self.min_surprise_std)
        rolling_mean = raw.rolling(self.lookback, min_periods=3).mean()
        z = (raw - rolling_mean) / rolling_std
        return z

    def compute_momentum(self, surprise_z: pd.Series) -> pd.Series:
        """Rolling mean of recent NFP surprises (momentum component)."""
        return surprise_z.rolling(self.momentum_window, min_periods=1).mean()

    def generate_signals(
        self,
        price: pd.Series,
        actual_nfp: pd.Series,
        consensus_nfp: pd.Series,
    ) -> pd.Series:
        """
        Build daily position signal from NFP surprises.

        Returns
        -------
        pd.Series
            Position in [-1, 1] indexed on price.index.
        """
        surprise_z = self.compute_surprise(actual_nfp, consensus_nfp)
        momentum_z = self.compute_momentum(surprise_z)

        if self.blend_momentum:
            combined_z = (
                (1 - self.momentum_weight) * surprise_z
                + self.momentum_weight * momentum_z
            )
        else:
            combined_z = surprise_z

        signal = pd.Series(0.0, index=price.index)

        for nfp_dt, z_val in combined_z.items():
            if abs(z_val) < self.z_threshold:
                continue

            # Find nearest trading day on or after release date
            future = price.index[price.index >= nfp_dt]
            if len(future) == 0:
                continue
            entry_dt = future[0]
            entry_idx = price.index.get_loc(entry_dt)
            end_idx = min(entry_idx + self.holding_period, len(price.index))

            direction = np.sign(z_val)
            position_size = min(abs(z_val) / 2.0, 1.0)

            for j in range(entry_idx, end_idx):
                fade = 1.0 - (j - entry_idx) / self.holding_period
                new_val = direction * position_size * fade
                if abs(new_val) > abs(signal.iloc[j]):
                    signal.iloc[j] = new_val

        return signal.clip(-1, 1)

    def backtest(
        self,
        price: pd.Series,
        actual_nfp: pd.Series,
        consensus_nfp: pd.Series,
    ) -> BacktestResult:
        """Full backtest of NFP momentum strategy."""
        daily_ret = price.pct_change()
        signal = self.generate_signals(price, actual_nfp, consensus_nfp)
        port_returns = (signal.shift(1) * daily_ret).dropna()
        eq = _build_equity(port_returns)
        s = _stats(port_returns)

        surprise_z = self.compute_surprise(actual_nfp, consensus_nfp)
        n_trades = int((surprise_z.abs() >= self.z_threshold).sum())

        return BacktestResult(
            total_return=s["total_return"],
            cagr=s["cagr"],
            sharpe=s["sharpe"],
            sortino=s["sortino"],
            max_drawdown=s["max_drawdown"],
            calmar=s["calmar"],
            win_rate=s["win_rate"],
            profit_factor=s["profit_factor"],
            n_trades=n_trades,
            equity_curve=eq,
            returns=port_returns,
            extra={"surprise_z": surprise_z, "signal": signal},
        )

    def nfp_statistics(
        self,
        actual_nfp: pd.Series,
        consensus_nfp: pd.Series,
        price: Optional[pd.Series] = None,
    ) -> Dict:
        """Summary statistics of NFP surprises and market impact."""
        surprise_z = self.compute_surprise(actual_nfp, consensus_nfp)
        raw_surprise = actual_nfp - consensus_nfp

        stats: Dict = {
            "n_releases": len(surprise_z),
            "mean_surprise_k": round(float(raw_surprise.mean()), 0),
            "std_surprise_k": round(float(raw_surprise.std()), 0),
            "pct_positive": round(float((raw_surprise > 0).mean()), 4),
            "pct_above_threshold": round(
                float((surprise_z.abs() >= self.z_threshold).mean()), 4
            ),
            "mean_abs_z": round(float(surprise_z.abs().mean()), 4),
        }

        if price is not None:
            daily_ret = price.pct_change()
            same_day_rets = []
            for dt in surprise_z.index:
                future = price.index[price.index >= dt]
                if len(future) > 0:
                    close_dt = future[0]
                    if close_dt in daily_ret.index:
                        same_day_rets.append(
                            (np.sign(surprise_z.loc[dt]), float(daily_ret.loc[close_dt]))
                        )
            if same_day_rets:
                directions, rets = zip(*same_day_rets)
                aligned = [d * r for d, r in zip(directions, rets)]
                stats["avg_aligned_day1_ret"] = round(np.mean(aligned), 4)
                stats["day1_hit_rate"] = round((np.array(aligned) > 0).mean(), 4)

        return stats

    def rolling_signal_ic(
        self,
        price: pd.Series,
        actual_nfp: pd.Series,
        consensus_nfp: pd.Series,
        ic_window: int = 12,
    ) -> pd.Series:
        """
        Rolling IC of the NFP surprise signal against subsequent returns.

        Returns
        -------
        pd.Series
            Monthly IC values.
        """
        from scipy.stats import spearmanr

        surprise_z = self.compute_surprise(actual_nfp, consensus_nfp)
        # Compute realized return over holding_period after each release
        fwd_rets = []
        for dt in surprise_z.index:
            future = price.index[price.index >= dt]
            if len(future) <= self.holding_period:
                fwd_rets.append(np.nan)
                continue
            ret = price.iloc[price.index.get_loc(future[0]) + self.holding_period] / \
                  price.loc[future[0]] - 1
            fwd_rets.append(ret)

        fwd_ret_series = pd.Series(fwd_rets, index=surprise_z.index)
        combined = pd.concat([surprise_z, fwd_ret_series], axis=1).dropna()
        combined.columns = ["z", "ret"]

        ic_vals = []
        ic_idx = []
        for i in range(ic_window, len(combined)):
            window = combined.iloc[i - ic_window:i]
            rho, _ = spearmanr(window["z"], window["ret"])
            ic_vals.append(rho)
            ic_idx.append(combined.index[i])

        return pd.Series(ic_vals, index=ic_idx, name="nfp_ic")


# ---------------------------------------------------------------------------
# CPIRegime
# ---------------------------------------------------------------------------

class CPIRegime:
    """
    CPI-driven inflation regime switching strategy.

    Classifies the economy into:
      - DEFLATIONARY : CPI trend < low_threshold
      - LOW_INFLATION : low_threshold <= trend <= high_threshold
      - HIGH_INFLATION: trend > high_threshold

    Each regime has preferred asset class tilts.

    Parameters
    ----------
    low_threshold : float
        Annualized CPI YoY below which regime = DEFLATIONARY.
    high_threshold : float
        Annualized CPI YoY above which regime = HIGH_INFLATION.
    smoothing_window : int
        Rolling window to smooth CPI trend signal.
    momentum_lookback : int
        Months to measure CPI momentum (acceleration/deceleration).
    use_momentum : bool
        If True, adjust regime signal by CPI momentum direction.
    min_regime_duration : int
        Minimum months in a regime before switching (avoids whipsawing).
    """

    DEFLATION = "DEFLATIONARY"
    LOW = "LOW_INFLATION"
    HIGH = "HIGH_INFLATION"

    def __init__(
        self,
        low_threshold: float = 0.02,
        high_threshold: float = 0.04,
        smoothing_window: int = 3,
        momentum_lookback: int = 6,
        use_momentum: bool = True,
        min_regime_duration: int = 2,
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.smoothing_window = smoothing_window
        self.momentum_lookback = momentum_lookback
        self.use_momentum = use_momentum
        self.min_regime_duration = min_regime_duration

    def compute_regime(
        self,
        cpi_yoy: pd.Series,
    ) -> pd.Series:
        """
        Classify each date into an inflation regime.

        Parameters
        ----------
        cpi_yoy : pd.Series
            Month-over-year CPI change (decimal, e.g. 0.03 = 3%).

        Returns
        -------
        pd.Series
            Regime labels.
        """
        smoothed = cpi_yoy.rolling(self.smoothing_window, min_periods=1).mean()

        if self.use_momentum:
            momentum = smoothed.diff(self.momentum_lookback)
        else:
            momentum = pd.Series(0.0, index=smoothed.index)

        raw_regime = []
        for i, (dt, val) in enumerate(smoothed.items()):
            if np.isnan(val):
                raw_regime.append(np.nan)
                continue
            mom = float(momentum.iloc[i]) if not np.isnan(momentum.iloc[i]) else 0.0
            # Adjust effective threshold by momentum
            eff_low = self.low_threshold - 0.005 * np.sign(mom)
            eff_high = self.high_threshold - 0.005 * np.sign(mom)

            if val < eff_low:
                raw_regime.append(self.DEFLATION)
            elif val > eff_high:
                raw_regime.append(self.HIGH)
            else:
                raw_regime.append(self.LOW)

        regime = pd.Series(raw_regime, index=smoothed.index, name="regime")

        # Apply minimum duration filter
        if self.min_regime_duration > 1:
            regime = self._apply_min_duration(regime)

        return regime

    def _apply_min_duration(self, regime: pd.Series) -> pd.Series:
        """Prevent regime switches lasting fewer than min_regime_duration periods."""
        values = regime.values.copy()
        i = 0
        while i < len(values):
            if pd.isna(values[i]):
                i += 1
                continue
            # Find run length
            j = i
            while j < len(values) and values[j] == values[i]:
                j += 1
            run_len = j - i
            if run_len < self.min_regime_duration and i > 0:
                # Replace with previous regime
                prev = values[i - 1]
                values[i:j] = prev
            i = j
        return pd.Series(values, index=regime.index, name=regime.name)

    def compute_signals(
        self,
        price_df: pd.DataFrame,
        cpi_yoy: pd.Series,
        asset_regime_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Build daily portfolio weights based on inflation regime.

        Parameters
        ----------
        price_df : pd.DataFrame
            Daily prices for each asset (dates x tickers).
        cpi_yoy : pd.Series
            Monthly CPI YoY (aligned to end-of-month dates).
        asset_regime_map : dict of {ticker: {regime: weight}}, optional
            Specifies portfolio weight per asset per regime.
            Defaults to a stock/bond/TIPS/commodity tilt:
              - HIGH_INFLATION: commodities +, bonds -
              - DEFLATIONARY: bonds +, equities -
              - LOW_INFLATION: balanced

        Returns
        -------
        pd.DataFrame
            Daily portfolio weights (dates x tickers).
        """
        regime = self.compute_regime(cpi_yoy)
        # Forward-fill monthly regime to daily
        regime_daily = regime.reindex(price_df.index, method="ffill")

        if asset_regime_map is None:
            # Default: assign +1/0/-1 weight per asset based on regime
            # (generic; user should supply real asset_regime_map in practice)
            asset_regime_map = {
                col: {
                    self.DEFLATION: -0.5,
                    self.LOW: 0.0,
                    self.HIGH: 0.5,
                }
                for col in price_df.columns
            }

        weights = pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
        for dt in price_df.index:
            r = regime_daily.loc[dt] if dt in regime_daily.index else np.nan
            if pd.isna(r):
                continue
            for col in price_df.columns:
                if col in asset_regime_map and r in asset_regime_map[col]:
                    weights.loc[dt, col] = asset_regime_map[col][r]

        # Normalize so |weights| sum to 1 per row
        row_sum = weights.abs().sum(axis=1).replace(0, np.nan)
        weights = weights.div(row_sum, axis=0).fillna(0)
        return weights

    def backtest(
        self,
        price_df: pd.DataFrame,
        cpi_yoy: pd.Series,
        asset_regime_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> BacktestResult:
        """Full backtest of CPI regime strategy."""
        fwd_returns = price_df.pct_change().shift(-1)
        weights = self.compute_signals(price_df, cpi_yoy, asset_regime_map)
        port_returns = (weights * fwd_returns).sum(axis=1).dropna()
        eq = _build_equity(port_returns)
        s = _stats(port_returns)

        regime = self.compute_regime(cpi_yoy)
        n_switches = int((regime != regime.shift(1)).sum())

        return BacktestResult(
            total_return=s["total_return"],
            cagr=s["cagr"],
            sharpe=s["sharpe"],
            sortino=s["sortino"],
            max_drawdown=s["max_drawdown"],
            calmar=s["calmar"],
            win_rate=s["win_rate"],
            profit_factor=s["profit_factor"],
            n_trades=n_switches,
            equity_curve=eq,
            returns=port_returns,
            extra={"regime": regime, "weights": weights},
        )

    def regime_statistics(self, cpi_yoy: pd.Series) -> pd.DataFrame:
        """Summary statistics of regime durations and frequencies."""
        regime = self.compute_regime(cpi_yoy)
        regime_clean = regime.dropna()
        rows = []
        for r in [self.DEFLATION, self.LOW, self.HIGH]:
            mask = regime_clean == r
            freq = mask.mean()
            # Count runs
            runs = (mask & ~mask.shift(1, fill_value=False)).sum()
            run_lengths = []
            in_run = False
            length = 0
            for v in mask:
                if v:
                    in_run = True
                    length += 1
                elif in_run:
                    run_lengths.append(length)
                    in_run = False
                    length = 0
            if in_run:
                run_lengths.append(length)
            avg_len = np.mean(run_lengths) if run_lengths else 0.0
            rows.append({
                "regime": r,
                "frequency": round(freq, 4),
                "n_episodes": int(runs),
                "avg_duration_months": round(avg_len, 2),
            })
        return pd.DataFrame(rows).set_index("regime")

    def cpi_momentum_signal(
        self, cpi_yoy: pd.Series, price: pd.Series
    ) -> pd.Series:
        """
        Additional signal: CPI momentum (acceleration / deceleration).

        Rising inflation (positive momentum) → risk-off signal.
        Falling inflation → risk-on signal.

        Returns
        -------
        pd.Series
            Daily signal in [-1, 1] on price.index.
        """
        momentum = (
            cpi_yoy.rolling(self.smoothing_window, min_periods=1)
            .mean()
            .diff(self.momentum_lookback)
        )
        # Normalize
        mom_std = momentum.rolling(24, min_periods=6).std().clip(lower=1e-6)
        mom_z = (momentum / mom_std).clip(-3, 3)
        # Invert: rising CPI → short signal
        signal_monthly = (-mom_z / 3.0).clip(-1, 1)
        # Forward-fill to daily
        signal_daily = signal_monthly.reindex(price.index, method="ffill").fillna(0)
        return signal_daily

    def inflation_regime_returns(
        self,
        price_df: pd.DataFrame,
        cpi_yoy: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute average returns for each asset in each inflation regime.

        Returns
        -------
        pd.DataFrame
            (regimes x assets) mean daily return table.
        """
        regime = self.compute_regime(cpi_yoy)
        regime_daily = regime.reindex(price_df.index, method="ffill")
        daily_rets = price_df.pct_change()

        rows = []
        for r in [self.DEFLATION, self.LOW, self.HIGH]:
            mask = regime_daily == r
            avg_rets = daily_rets[mask].mean() * 252  # annualized
            row = {"regime": r}
            row.update(avg_rets.round(4).to_dict())
            rows.append(row)

        return pd.DataFrame(rows).set_index("regime")
