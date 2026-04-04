"""
Event-driven earnings strategies.

Implements:
- EarningsMomentum   : buy stocks with consecutive positive surprises
- EarningsSurprise   : standardized unexpected earnings (SUE) signal
- EarningsReversal   : post-earnings announcement drift fade after 60d
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Shared data structures
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
    """Compute standard performance statistics from a return series."""
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


# ---------------------------------------------------------------------------
# EarningsSurprise (SUE)
# ---------------------------------------------------------------------------

class EarningsSurprise:
    """
    Standardized Unexpected Earnings (SUE) momentum strategy.

    SUE_t = (actual_eps_t - expected_eps_t) / std(surprise_t[-lookback:])

    A long/short portfolio is built by going long top-n and short bottom-n
    stocks ranked by SUE.

    Parameters
    ----------
    lookback : int
        Rolling window (in quarters) to estimate surprise volatility.
    n_long : int
        Number of long positions.
    n_short : int
        Number of short positions.
    holding_period : int
        Trading days to hold a position after earnings announcement.
    min_surprise_std : float
        Minimum surprise std to produce a signal (avoids division by zero
        for companies with flat consensus).
    decay_halflife : int
        Exponential decay half-life (days) for signal staleness.
    """

    def __init__(
        self,
        lookback: int = 8,
        n_long: int = 10,
        n_short: int = 10,
        holding_period: int = 21,
        min_surprise_std: float = 0.01,
        decay_halflife: int = 30,
    ) -> None:
        self.lookback = lookback
        self.n_long = n_long
        self.n_short = n_short
        self.holding_period = holding_period
        self.min_surprise_std = min_surprise_std
        self.decay_halflife = decay_halflife

    def compute_sue(
        self,
        actual_eps: pd.DataFrame,
        expected_eps: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute SUE for each ticker over time.

        Parameters
        ----------
        actual_eps : pd.DataFrame
            (dates x tickers) actual EPS; NaN on non-announcement dates.
        expected_eps : pd.DataFrame
            (dates x tickers) analyst consensus EPS estimate.

        Returns
        -------
        pd.DataFrame
            SUE values; NaN where no announcement.
        """
        raw_surprise = actual_eps - expected_eps
        # Rolling std of surprises (quarterly, so window = lookback quarters)
        surprise_std = (
            raw_surprise.rolling(self.lookback, min_periods=2).std()
            .clip(lower=self.min_surprise_std)
        )
        sue = raw_surprise / surprise_std
        return sue

    def compute_signals(
        self,
        sue: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert SUE announcements into daily signals by forward-filling
        the most recent SUE for holding_period days, with exponential decay.

        Parameters
        ----------
        sue : pd.DataFrame
            (announcement dates x tickers) SUE values.
        price_df : pd.DataFrame
            Daily price dataframe (dates x tickers) — used for index.

        Returns
        -------
        pd.DataFrame
            Daily signal values (dates x tickers).
        """
        daily = pd.DataFrame(index=price_df.index, columns=sue.columns, dtype=float)

        decay = np.log(2) / self.decay_halflife

        for ticker in sue.columns:
            ann_dates = sue[ticker].dropna()
            signal_arr = np.full(len(price_df.index), np.nan)

            for ann_dt, sue_val in ann_dates.items():
                try:
                    start_idx = price_df.index.get_loc(ann_dt)
                except KeyError:
                    # Find nearest future date
                    future = price_df.index[price_df.index >= ann_dt]
                    if len(future) == 0:
                        continue
                    start_idx = price_df.index.get_loc(future[0])

                end_idx = min(start_idx + self.holding_period, len(price_df.index))
                for j in range(start_idx, end_idx):
                    days_elapsed = j - start_idx
                    decayed = sue_val * np.exp(-decay * days_elapsed)
                    # Most recent announcement takes precedence
                    if np.isnan(signal_arr[j]) or abs(decayed) > abs(signal_arr[j]):
                        signal_arr[j] = decayed

            daily[ticker] = signal_arr

        return daily

    def generate_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Build portfolio weights: equal-weight long top-n, short bottom-n.

        Parameters
        ----------
        signals : pd.DataFrame
            Daily signal values.

        Returns
        -------
        pd.DataFrame
            Equal-weight long/short portfolio weights.
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for date in signals.index:
            row = signals.loc[date].dropna()
            if len(row) < self.n_long + self.n_short:
                continue
            ranked = row.rank(ascending=False)
            long_pos = ranked[ranked <= self.n_long].index
            short_pos = ranked[ranked > len(row) - self.n_short].index
            if len(long_pos) > 0:
                weights.loc[date, long_pos] = 1.0 / len(long_pos)
            if len(short_pos) > 0:
                weights.loc[date, short_pos] = -1.0 / len(short_pos)
        return weights

    def backtest(
        self,
        price_df: pd.DataFrame,
        actual_eps: pd.DataFrame,
        expected_eps: pd.DataFrame,
    ) -> BacktestResult:
        """
        Full backtest of the SUE strategy.

        Parameters
        ----------
        price_df : pd.DataFrame
            Daily close prices (dates x tickers).
        actual_eps : pd.DataFrame
            Actual EPS announcements.
        expected_eps : pd.DataFrame
            Consensus EPS estimates.

        Returns
        -------
        BacktestResult
        """
        fwd_returns = price_df.pct_change().shift(-1)
        sue = self.compute_sue(actual_eps, expected_eps)
        signals = self.compute_signals(sue, price_df)
        weights = self.generate_weights(signals)

        port_returns = (weights * fwd_returns).sum(axis=1)
        port_returns = port_returns.dropna()

        eq = (1 + port_returns).cumprod()
        s = _stats(port_returns)
        n_trades = int((weights.diff().abs() > 0.001).sum().sum())

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
            extra={"sue": sue, "signals": signals, "weights": weights},
        )

    def sue_statistics(
        self,
        actual_eps: pd.DataFrame,
        expected_eps: pd.DataFrame,
    ) -> pd.DataFrame:
        """Summary statistics of SUE distribution per ticker."""
        sue = self.compute_sue(actual_eps, expected_eps)
        rows = []
        for col in sue.columns:
            s = sue[col].dropna()
            if len(s) == 0:
                continue
            rows.append({
                "ticker": col,
                "n_announcements": len(s),
                "mean_sue": round(s.mean(), 4),
                "std_sue": round(s.std(), 4),
                "pct_positive": round((s > 0).mean(), 4),
                "skew": round(float(s.skew()), 4),
                "max_sue": round(s.max(), 4),
                "min_sue": round(s.min(), 4),
            })
        return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------------------
# EarningsMomentum
# ---------------------------------------------------------------------------

class EarningsMomentum:
    """
    Buy stocks with a streak of consecutive positive earnings surprises;
    short stocks with consecutive negative surprises.

    The signal strength is proportional to the length of the streak and
    the magnitude of the most recent SUE.

    Parameters
    ----------
    min_streak : int
        Minimum consecutive same-direction surprises to trigger a signal.
    max_streak : int
        Cap streak length for signal scaling.
    holding_period : int
        Days to hold the position.
    n_long : int
        Number of long positions.
    n_short : int
        Number of short positions.
    sue_threshold : float
        Minimum absolute SUE magnitude to count toward a streak.
    """

    def __init__(
        self,
        min_streak: int = 3,
        max_streak: int = 8,
        holding_period: int = 63,
        n_long: int = 10,
        n_short: int = 10,
        sue_threshold: float = 0.5,
    ) -> None:
        self.min_streak = min_streak
        self.max_streak = max_streak
        self.holding_period = holding_period
        self.n_long = n_long
        self.n_short = n_short
        self.sue_threshold = sue_threshold

    def compute_streaks(
        self,
        actual_eps: pd.DataFrame,
        expected_eps: pd.DataFrame,
        lookback: int = 8,
        min_surprise_std: float = 0.01,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute consecutive beat/miss streaks and latest SUE per ticker.

        Returns
        -------
        streaks : pd.DataFrame
            Number of consecutive same-direction surprises (positive = beats).
        sue_df : pd.DataFrame
            Standardized unexpected earnings.
        """
        raw_surprise = actual_eps - expected_eps
        surprise_std = (
            raw_surprise.rolling(lookback, min_periods=2).std()
            .clip(lower=min_surprise_std)
        )
        sue_df = raw_surprise / surprise_std

        streaks = pd.DataFrame(0, index=sue_df.index, columns=sue_df.columns)

        for col in sue_df.columns:
            s = sue_df[col].dropna()
            if len(s) == 0:
                continue
            streak_vals = []
            current_streak = 0
            for val in s:
                if abs(val) < self.sue_threshold:
                    current_streak = 0
                elif val > 0:
                    current_streak = current_streak + 1 if current_streak >= 0 else 1
                else:
                    current_streak = current_streak - 1 if current_streak <= 0 else -1
                streak_vals.append(current_streak)
            streak_series = pd.Series(streak_vals, index=s.index)
            streaks[col] = streak_series

        return streaks, sue_df

    def compute_signals(
        self,
        streaks: pd.DataFrame,
        sue_df: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Map streak + SUE information onto a daily signal.

        Signal = streak_length * sue_magnitude if |streak| >= min_streak, else 0.
        """
        daily = pd.DataFrame(0.0, index=price_df.index, columns=sue_df.columns)

        for ticker in sue_df.columns:
            ann_dates = sue_df[ticker].dropna().index
            for ann_dt in ann_dates:
                if ann_dt not in streaks.index:
                    continue
                streak_val = streaks.loc[ann_dt, ticker]
                sue_val = sue_df.loc[ann_dt, ticker]

                if abs(streak_val) < self.min_streak:
                    continue

                # Scale signal
                effective_streak = min(abs(streak_val), self.max_streak)
                signal_val = np.sign(streak_val) * effective_streak * abs(sue_val)

                # Find daily index position
                future_dates = price_df.index[price_df.index >= ann_dt]
                if len(future_dates) == 0:
                    continue
                start_idx = price_df.index.get_loc(future_dates[0])
                end_idx = min(start_idx + self.holding_period, len(price_df.index))

                for j in range(start_idx, end_idx):
                    fade = 1.0 - (j - start_idx) / self.holding_period
                    new_val = signal_val * fade
                    if abs(new_val) > abs(daily.iloc[j][ticker]):
                        daily.iloc[j, daily.columns.get_loc(ticker)] = new_val

        return daily

    def generate_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for date in signals.index:
            row = signals.loc[date]
            nonzero = row[row != 0].dropna()
            if len(nonzero) == 0:
                continue
            ranked = nonzero.rank(ascending=False)
            long_pos = ranked[ranked <= self.n_long].index
            short_pos = ranked[ranked > max(len(nonzero) - self.n_short, 0)].index
            if len(long_pos) > 0:
                weights.loc[date, long_pos] = 1.0 / len(long_pos)
            if len(short_pos) > 0:
                weights.loc[date, short_pos] = -1.0 / len(short_pos)
        return weights

    def backtest(
        self,
        price_df: pd.DataFrame,
        actual_eps: pd.DataFrame,
        expected_eps: pd.DataFrame,
    ) -> BacktestResult:
        """Full backtest of the earnings momentum strategy."""
        fwd_returns = price_df.pct_change().shift(-1)
        streaks, sue_df = self.compute_streaks(actual_eps, expected_eps)
        signals = self.compute_signals(streaks, sue_df, price_df)
        weights = self.generate_weights(signals)

        port_returns = (weights * fwd_returns).sum(axis=1).dropna()
        eq = (1 + port_returns).cumprod()
        s = _stats(port_returns)
        n_trades = int((weights.diff().abs() > 0.001).sum().sum())

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
            extra={"streaks": streaks, "signals": signals},
        )

    def streak_statistics(
        self,
        actual_eps: pd.DataFrame,
        expected_eps: pd.DataFrame,
    ) -> pd.DataFrame:
        """Summary of streak distribution per ticker."""
        streaks, _ = self.compute_streaks(actual_eps, expected_eps)
        rows = []
        for col in streaks.columns:
            s = streaks[col].dropna()
            ann = s[s != 0]
            if len(ann) == 0:
                continue
            rows.append({
                "ticker": col,
                "n_announcements": len(ann),
                "mean_streak": round(ann.mean(), 2),
                "max_streak": int(ann.max()),
                "min_streak": int(ann.min()),
                "pct_long_streak": round((ann >= self.min_streak).mean(), 4),
            })
        return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------------------
# EarningsReversal
# ---------------------------------------------------------------------------

class EarningsReversal:
    """
    Post-earnings announcement drift (PEAD) fade strategy.

    After the initial PEAD holding period (drift_period days), the alpha
    of an earnings announcement typically decays.  This strategy fades the
    initial direction after drift_period days, betting on mean reversion.

    Optionally combines SUE signal with price reaction to identify cases
    where the market overreacted.

    Parameters
    ----------
    drift_period : int
        Days after announcement to wait before entering reversal trade.
    hold_period : int
        Days to hold the reversal trade.
    min_abs_sue : float
        Minimum |SUE| to trade (filter weak events).
    price_reaction_threshold : float
        Minimum absolute price reaction on announcement day to qualify.
    n_long : int
        Number of long positions (stocks that dropped sharply).
    n_short : int
        Number of short positions (stocks that rose sharply).
    use_price_reaction : bool
        Weight signal by initial price reaction magnitude.
    """

    def __init__(
        self,
        drift_period: int = 60,
        hold_period: int = 42,
        min_abs_sue: float = 1.0,
        price_reaction_threshold: float = 0.03,
        n_long: int = 10,
        n_short: int = 10,
        use_price_reaction: bool = True,
    ) -> None:
        self.drift_period = drift_period
        self.hold_period = hold_period
        self.min_abs_sue = min_abs_sue
        self.price_reaction_threshold = price_reaction_threshold
        self.n_long = n_long
        self.n_short = n_short
        self.use_price_reaction = use_price_reaction

    def compute_price_reaction(
        self,
        price_df: pd.DataFrame,
        announcement_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the abnormal price return on/around announcement date.

        Parameters
        ----------
        price_df : pd.DataFrame
            Daily close prices.
        announcement_df : pd.DataFrame
            Boolean/1.0 mask indicating announcement dates.

        Returns
        -------
        pd.DataFrame
            Price reaction (return from day before to day after announcement).
        """
        daily_ret = price_df.pct_change()
        reaction = pd.DataFrame(np.nan, index=announcement_df.index,
                                columns=announcement_df.columns)
        for col in announcement_df.columns:
            ann_dates = announcement_df[col].dropna()
            ann_dates = ann_dates[ann_dates != 0].index
            for dt in ann_dates:
                if dt not in price_df.index:
                    continue
                idx = price_df.index.get_loc(dt)
                if idx < 1 or idx + 1 >= len(price_df.index):
                    continue
                # 3-day window: t-1 to t+1
                r = price_df[col].iloc[idx + 1] / price_df[col].iloc[idx - 1] - 1
                reaction.loc[dt, col] = r
        return reaction

    def compute_signals(
        self,
        actual_eps: pd.DataFrame,
        expected_eps: pd.DataFrame,
        price_df: pd.DataFrame,
        announcement_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build reversal signals: -sign(initial_reaction) entered at drift_period.

        Parameters
        ----------
        actual_eps, expected_eps : pd.DataFrame
            EPS data.
        price_df : pd.DataFrame
            Daily prices.
        announcement_df : pd.DataFrame, optional
            1.0 on announcement dates, else NaN.  If None, uses
            actual_eps.notna().astype(float).

        Returns
        -------
        pd.DataFrame
            Daily reversal signal values.
        """
        if announcement_df is None:
            announcement_df = actual_eps.notna().astype(float).replace(0, np.nan)

        raw_surprise = actual_eps - expected_eps
        surprise_std = raw_surprise.rolling(8, min_periods=2).std().clip(lower=0.01)
        sue_df = raw_surprise / surprise_std

        price_reaction = self.compute_price_reaction(price_df, announcement_df)
        daily = pd.DataFrame(0.0, index=price_df.index, columns=actual_eps.columns)

        for col in actual_eps.columns:
            ann_dates = actual_eps[col].dropna().index
            for ann_dt in ann_dates:
                # Filter by SUE magnitude
                if ann_dt not in sue_df.index:
                    continue
                sue_val = sue_df.loc[ann_dt, col] if col in sue_df.columns else np.nan
                if np.isnan(sue_val) or abs(sue_val) < self.min_abs_sue:
                    continue

                # Get initial price reaction
                reaction_val = (
                    price_reaction.loc[ann_dt, col]
                    if ann_dt in price_reaction.index else np.nan
                )
                if np.isnan(reaction_val):
                    continue
                if abs(reaction_val) < self.price_reaction_threshold:
                    continue

                # Reversal signal direction: fade the reaction
                direction = -np.sign(reaction_val)
                magnitude = abs(reaction_val) if self.use_price_reaction else 1.0

                # Find entry date (drift_period after announcement)
                future = price_df.index[price_df.index >= ann_dt]
                if len(future) <= self.drift_period:
                    continue
                entry_dt = future[self.drift_period]
                entry_idx = price_df.index.get_loc(entry_dt)
                end_idx = min(entry_idx + self.hold_period, len(price_df.index))

                col_idx = daily.columns.get_loc(col)
                for j in range(entry_idx, end_idx):
                    fade = 1.0 - (j - entry_idx) / self.hold_period
                    new_val = direction * magnitude * fade
                    if abs(new_val) > abs(daily.iloc[j, col_idx]):
                        daily.iloc[j, col_idx] = new_val

        return daily

    def generate_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for date in signals.index:
            row = signals.loc[date]
            nonzero = row[row != 0].dropna()
            if len(nonzero) == 0:
                continue
            ranked = nonzero.rank(ascending=False)
            long_pos = ranked[ranked <= self.n_long].index
            short_pos = ranked[ranked > max(len(nonzero) - self.n_short, 0)].index
            if len(long_pos) > 0:
                weights.loc[date, long_pos] = 1.0 / len(long_pos)
            if len(short_pos) > 0:
                weights.loc[date, short_pos] = -1.0 / len(short_pos)
        return weights

    def backtest(
        self,
        price_df: pd.DataFrame,
        actual_eps: pd.DataFrame,
        expected_eps: pd.DataFrame,
        announcement_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Full backtest of earnings reversal strategy."""
        fwd_returns = price_df.pct_change().shift(-1)
        signals = self.compute_signals(
            actual_eps, expected_eps, price_df, announcement_df
        )
        weights = self.generate_weights(signals)

        port_returns = (weights * fwd_returns).sum(axis=1).dropna()
        eq = (1 + port_returns).cumprod()
        s = _stats(port_returns)
        n_trades = int((weights.diff().abs() > 0.001).sum().sum())

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
            extra={"signals": signals},
        )

    def reversal_quality_report(
        self,
        price_df: pd.DataFrame,
        actual_eps: pd.DataFrame,
        expected_eps: pd.DataFrame,
        announcement_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        For each ticker, report how many reversals were identified and the
        average subsequent return during the hold period.
        """
        if announcement_df is None:
            announcement_df = actual_eps.notna().astype(float).replace(0, np.nan)

        raw_surprise = actual_eps - expected_eps
        surprise_std = raw_surprise.rolling(8, min_periods=2).std().clip(lower=0.01)
        sue_df = raw_surprise / surprise_std
        price_reaction = self.compute_price_reaction(price_df, announcement_df)
        fwd_ret = price_df.pct_change()

        rows = []
        for col in actual_eps.columns:
            ann_dates = actual_eps[col].dropna().index
            results = []
            for ann_dt in ann_dates:
                if ann_dt not in sue_df.index:
                    continue
                sue_val = sue_df.loc[ann_dt, col] if col in sue_df.columns else np.nan
                if np.isnan(sue_val) or abs(sue_val) < self.min_abs_sue:
                    continue
                reaction_val = (
                    price_reaction.loc[ann_dt, col]
                    if ann_dt in price_reaction.index else np.nan
                )
                if np.isnan(reaction_val) or abs(reaction_val) < self.price_reaction_threshold:
                    continue

                direction = -np.sign(reaction_val)
                future = price_df.index[price_df.index >= ann_dt]
                if len(future) <= self.drift_period + self.hold_period:
                    continue
                entry_dt = future[self.drift_period]
                exit_dt = future[min(self.drift_period + self.hold_period, len(future) - 1)]
                if entry_dt in fwd_ret.index and exit_dt in price_df.index and entry_dt in price_df.index:
                    ret = price_df.loc[exit_dt, col] / price_df.loc[entry_dt, col] - 1
                    results.append(direction * ret)

            if results:
                arr = np.array(results)
                rows.append({
                    "ticker": col,
                    "n_trades": len(arr),
                    "mean_return": round(arr.mean(), 4),
                    "hit_rate": round((arr > 0).mean(), 4),
                    "t_stat": round(
                        float(arr.mean() / (arr.std() / np.sqrt(len(arr)) + 1e-12)), 4
                    ),
                })

        return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
