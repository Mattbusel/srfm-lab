"""
variance_premium.py — Variance risk premium harvesting strategies.

The variance risk premium (VRP) is the compensation for bearing variance risk.
It equals: E[realized_var] - implied_var ≈ negative on average
(sellers of variance earn a premium because buyers are risk-averse).
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class BacktestResult:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    n_trades: int = 0
    avg_vrp: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"VRP={self.avg_vrp:.4f}")


def _equity_stats(ec: np.ndarray) -> dict:
    n = len(ec)
    tot = ec[-1] / ec[0] - 1
    cagr = (ec[-1] / ec[0]) ** (1 / max(1, n / 252)) - 1
    r = np.diff(ec) / (ec[:-1] + 1e-9)
    r = np.concatenate([[0], r])
    std = r.std()
    sh = r.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down = r[r < 0]
    sortino = r.mean() / (np.std(down) + 1e-9) * math.sqrt(252)
    pk = np.maximum.accumulate(ec)
    dd = (ec - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return dict(total_return=tot, cagr=cagr, sharpe=sh, sortino=sortino,
                max_drawdown=mdd, calmar=calmar)


# ─────────────────────────────────────────────────────────────────────────────
# 1. VariancePremiumCapture
# ─────────────────────────────────────────────────────────────────────────────

class VariancePremiumCapture:
    """
    Variance Risk Premium (VRP) capture strategy.

    The VRP is defined as: VRP = implied_vol^2 - realized_vol^2
    (in variance terms), or VRP_vol = implied_vol - realized_vol.

    A persistent positive VRP means options are consistently overpriced
    relative to subsequent realized vol → sell volatility.

    This strategy:
    1. Sells variance/options when implied vol is high relative to realized
    2. Goes neutral or buys when IV is cheap
    3. Sizes position by the magnitude of the VRP signal

    Parameters
    ----------
    realized_window    : window for realized vol (default 21)
    implied_vol_proxy  : use VIX or compute from option chain
    signal_smoothing   : smooth the VRP signal (default 5)
    threshold          : minimum VRP to trade (default 0.01 = 1 vol point)
    max_position       : max short vol position (default 1.0)
    """

    def __init__(
        self,
        realized_window: int = 21,
        signal_smoothing: int = 5,
        threshold: float = 0.01,
        max_position: float = 1.0,
    ):
        self.realized_window = realized_window
        self.signal_smoothing = signal_smoothing
        self.threshold = threshold
        self.max_position = max_position

    def compute_realized_vol(self, price_series: pd.Series) -> pd.Series:
        """Annualized realized volatility (close-to-close)."""
        log_ret = np.log(price_series / price_series.shift(1))
        rv = log_ret.rolling(self.realized_window, min_periods=max(3, self.realized_window // 3)).std()
        return rv * math.sqrt(252)

    def compute_vrp(
        self,
        realized_vol: pd.Series,
        implied_vol_proxy: pd.Series,
    ) -> pd.Series:
        """
        Compute the annualized VRP = implied_vol - realized_vol.

        Positive VRP = IV > RV = overpriced options → sell vol.
        """
        vrp = implied_vol_proxy - realized_vol
        if self.signal_smoothing > 1:
            vrp = vrp.ewm(span=self.signal_smoothing, adjust=False).mean()
        return vrp

    def generate_signals(
        self,
        realized_vol: pd.Series,
        implied_vol_proxy: pd.Series,
    ) -> pd.Series:
        """
        Continuous position: negative = short vol, positive = long vol.
        Position scaled by VRP magnitude.
        """
        vrp = self.compute_vrp(realized_vol, implied_vol_proxy)

        # Position: short vol when VRP high, long vol when VRP negative
        # Signal: short vol = bearish on realized vol = long on equity (proxy)
        position = pd.Series(0.0, index=vrp.index)

        for i in range(1, len(vrp)):
            v = float(vrp.iloc[i])
            if np.isnan(v):
                continue
            if v > self.threshold:
                # Short vol: scale by VRP / threshold → max position at 2x threshold
                pos = min(self.max_position, v / (self.threshold * 2))
                position.iloc[i] = pos
            elif v < -self.threshold:
                pos = max(-self.max_position, v / (self.threshold * 2))
                position.iloc[i] = pos
            else:
                position.iloc[i] = 0.0

        return position

    def backtest(
        self,
        realized_vol: pd.Series,
        implied_vol_proxy: pd.Series,
        underlying_price: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        """
        Backtest the VRP strategy on the underlying as a proxy for vol exposure.
        Selling vol (short vol) ≈ long underlying in calm markets.
        """
        signal = self.generate_signals(realized_vol, implied_vol_proxy)
        vrp = self.compute_vrp(realized_vol, implied_vol_proxy)

        close = underlying_price.values
        sig = signal.values
        n = len(close)

        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        pos = 0.0

        for i in range(1, n):
            s = float(sig[i - 1]) if not np.isnan(sig[i - 1]) else 0.0
            bar_ret = (close[i] - close[i - 1]) / (close[i - 1] + 1e-9)
            turnover = abs(s - pos) * commission_pct
            equity *= (1 + s * bar_ret - turnover)
            if abs(bar_ret * s) > 1e-9:
                trades.append(s * bar_ret)
            ec[i] = equity
            pos = s

        s_stats = _equity_stats(ec)
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]

        return BacktestResult(
            **s_stats,
            n_trades=len(trades),
            avg_vrp=float(vrp.dropna().mean()),
            equity_curve=pd.Series(ec, index=underlying_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=underlying_price.index[1:]),
            signals=signal,
            params={"realized_window": self.realized_window, "threshold": self.threshold},
        )

    def carry_analysis(
        self,
        realized_vol: pd.Series,
        implied_vol_proxy: pd.Series,
    ) -> dict:
        """Analyze the VRP carry statistics."""
        vrp = self.compute_vrp(realized_vol, implied_vol_proxy)
        vrp_clean = vrp.dropna()
        return {
            "mean_vrp_annualized": float(vrp_clean.mean()),
            "std_vrp": float(vrp_clean.std()),
            "pct_positive": float((vrp_clean > 0).mean()),
            "sharpe_vrp": float(vrp_clean.mean() / (vrp_clean.std() + 1e-9) * math.sqrt(252)),
            "max_vrp": float(vrp_clean.max()),
            "min_vrp": float(vrp_clean.min()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. VIXFuturesRolldown
# ─────────────────────────────────────────────────────────────────────────────

class VIXFuturesRolldown:
    """
    VIX Futures Roll-Down / Term Structure Carry Strategy.

    The VIX futures curve is almost always in contango (futures > spot VIX).
    Shorting VIX futures (or long SVXY/short VXX) harvests this premium
    as futures roll down to spot.

    Expected P&L from roll: (Futures - Spot) / (days_to_expiry / 365)

    Parameters
    ----------
    min_contango_pct    : minimum contango premium to enter (default 0.05 = 5%)
    exit_contango_pct   : minimum contango to stay in trade (default 0.01)
    backwardation_exit  : exit if backwardation exceeds this level (default 0.05)
    vol_filter_window   : VIX moving average for trend filter (default 30)
    max_vix_level       : refuse entry if VIX > this level (default 35)
    """

    def __init__(
        self,
        min_contango_pct: float = 0.05,
        exit_contango_pct: float = 0.01,
        backwardation_exit: float = 0.05,
        vol_filter_window: int = 30,
        max_vix_level: float = 35.0,
    ):
        self.min_contango_pct = min_contango_pct
        self.exit_contango_pct = exit_contango_pct
        self.backwardation_exit = backwardation_exit
        self.vol_filter_window = vol_filter_window
        self.max_vix_level = max_vix_level

    def compute_term_structure(
        self,
        vix_spot: pd.Series,
        vix_futures: pd.Series,
        roll_days: pd.Series = None,
    ) -> pd.Series:
        """
        Compute the VIX term structure slope (contango/backwardation).

        term_structure = (vix_futures - vix_spot) / vix_spot
        Positive = contango (normal state). Negative = backwardation (fear).
        """
        ts = (vix_futures - vix_spot) / (vix_spot + 1e-9)
        if roll_days is not None:
            # Annualize by remaining term
            ts = ts / (roll_days / 30)
        return ts

    def generate_signals(
        self,
        vix_spot: pd.Series,
        vix_futures: pd.Series,
        roll_days: pd.Series = None,
    ) -> pd.Series:
        """
        Short VIX futures (or long inverse VIX ETP proxy) when in strong contango.
        Signal: -1 = short volatility, 0 = flat.
        """
        ts = self.compute_term_structure(vix_spot, vix_futures, roll_days)
        vix_ma = vix_spot.ewm(span=self.vol_filter_window, adjust=False).mean()

        signal = pd.Series(0.0, index=vix_spot.index)
        position = 0

        for i in range(self.vol_filter_window, len(ts)):
            ts_val = float(ts.iloc[i])
            vix_val = float(vix_spot.iloc[i])
            if np.isnan(ts_val) or np.isnan(vix_val):
                continue

            # Entry conditions: strong contango AND VIX not too high
            if position == 0:
                if (ts_val > self.min_contango_pct and
                        vix_val < self.max_vix_level):
                    position = -1  # short vol

            # Exit conditions
            elif position == -1:
                if (ts_val < self.exit_contango_pct or
                        ts_val < -self.backwardation_exit or
                        vix_val > self.max_vix_level):
                    position = 0

            signal.iloc[i] = float(position)

        signal.iloc[:self.vol_filter_window] = np.nan
        return signal

    def backtest(
        self,
        vix_spot: pd.Series,
        vix_futures: pd.Series,
        inverse_vix_etf: pd.Series,
        roll_days: pd.Series = None,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        """
        Backtest using inverse VIX ETF (SVXY proxy) as trading vehicle.
        Signal: -1 (short vol) → go LONG inverse VIX ETF.
        """
        signal = self.generate_signals(vix_spot, vix_futures, roll_days)
        ts = self.compute_term_structure(vix_spot, vix_futures, roll_days)

        # For inverse VIX ETF: short vol = long position
        adj_signal = -signal  # invert: -1 short vol → +1 long inverse ETP

        close = inverse_vix_etf.values
        sig = adj_signal.values
        n = len(close)

        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        pos = 0.0

        for i in range(1, n):
            s = float(sig[i - 1]) if not np.isnan(sig[i - 1]) else 0.0
            bar_ret = (close[i] - close[i - 1]) / (close[i - 1] + 1e-9)
            cost = abs(s - pos) * commission_pct
            equity *= (1 + s * bar_ret - cost)
            if abs(bar_ret * s) > 1e-9:
                trades.append(s * bar_ret)
            ec[i] = equity
            pos = s

        s_stats = _equity_stats(ec)
        wins = [t for t in trades if t > 0]

        return BacktestResult(
            **s_stats,
            n_trades=len(trades),
            avg_vrp=float(ts.dropna().mean()),
            equity_curve=pd.Series(ec, index=inverse_vix_etf.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=inverse_vix_etf.index[1:]),
            signals=signal,
            params={"min_contango_pct": self.min_contango_pct,
                    "max_vix_level": self.max_vix_level},
        )

    def roll_yield_statistics(
        self,
        vix_spot: pd.Series,
        vix_futures: pd.Series,
        roll_days: pd.Series = None,
    ) -> dict:
        """Summary statistics of the VIX roll yield."""
        ts = self.compute_term_structure(vix_spot, vix_futures, roll_days)
        ts_clean = ts.dropna()
        return {
            "mean_contango": float(ts_clean.mean()),
            "std_contango": float(ts_clean.std()),
            "pct_contango": float((ts_clean > 0).mean()),
            "pct_backwardation": float((ts_clean < 0).mean()),
            "max_contango": float(ts_clean.max()),
            "max_backwardation": float(ts_clean.min()),
            "sharpe_roll": float(ts_clean.mean() / (ts_clean.std() + 1e-9) * math.sqrt(252)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. VolatilitySurfaceArbitrage
# ─────────────────────────────────────────────────────────────────────────────

class VolatilitySurfaceArbitrage:
    """
    Volatility surface arbitrage: exploit mispricings in the vol surface.

    Strategies:
    1. Calendar spread arb: front month vol too high vs back month
    2. Butterfly spread arb: middle strike vol misaligned with wings
    3. Skew trade: put skew mean reversion

    The vol surface is represented as a grid:
        rows = strikes (or deltas), columns = expiries

    Parameters
    ----------
    min_calendar_spread : minimum vol diff (ann.) to trade calendar (default 0.02)
    min_butterfly_width : minimum butterfly width to trade (default 0.01)
    smoothing_window    : window for surface smoothing (default 5)
    """

    def __init__(
        self,
        min_calendar_spread: float = 0.02,
        min_butterfly_width: float = 0.01,
        smoothing_window: int = 5,
    ):
        self.min_calendar_spread = min_calendar_spread
        self.min_butterfly_width = min_butterfly_width
        self.smoothing_window = smoothing_window

    def detect_calendar_arb(
        self,
        surface_grid: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect calendar spread arbitrage opportunities.

        surface_grid: rows = strikes, columns = expiry dates (as strings or dates)
        Values = implied volatilities (annualized, e.g., 0.25 = 25%)

        Calendar arb: IV at short expiry > IV at long expiry by > threshold.
        (calendar spread arb: buy far, sell near)

        Returns DataFrame of opportunities.
        """
        strikes = surface_grid.index
        expiries = surface_grid.columns
        opportunities = []

        for strike in strikes:
            for j in range(len(expiries) - 1):
                near_exp = expiries[j]
                far_exp = expiries[j + 1]
                near_iv = float(surface_grid.loc[strike, near_exp])
                far_iv = float(surface_grid.loc[strike, far_exp])

                if np.isnan(near_iv) or np.isnan(far_iv):
                    continue

                spread = near_iv - far_iv
                if spread > self.min_calendar_spread:
                    opportunities.append({
                        "strike": strike,
                        "near_expiry": near_exp,
                        "far_expiry": far_exp,
                        "near_iv": near_iv,
                        "far_iv": far_iv,
                        "spread": spread,
                        "trade": "buy far, sell near",
                    })

        return pd.DataFrame(opportunities) if opportunities else pd.DataFrame()

    def detect_butterfly_arb(
        self,
        surface_grid: pd.DataFrame,
        expiry: str = None,
    ) -> pd.DataFrame:
        """
        Detect butterfly spread arbitrage.

        Butterfly arb: middle strike vol < (lower_strike_vol + upper_strike_vol) / 2.
        A positive butterfly means convexity is violated — arb exists.

        Parameters
        ----------
        surface_grid : rows = strikes, columns = expiries
        expiry       : which expiry to examine (default: first column)
        """
        if expiry is None:
            expiry = surface_grid.columns[0]

        vol_slice = surface_grid[expiry].dropna().sort_index()
        strikes = vol_slice.index.values
        vols = vol_slice.values
        opportunities = []

        for i in range(1, len(strikes) - 1):
            k_low = strikes[i - 1]
            k_mid = strikes[i]
            k_high = strikes[i + 1]
            v_low = vols[i - 1]
            v_mid = vols[i]
            v_high = vols[i + 1]

            # Linear interpolation of vol at k_mid from wings
            interp_vol = v_low + (v_high - v_low) * (k_mid - k_low) / (k_high - k_low)
            butterfly_width = interp_vol - v_mid

            if butterfly_width < -self.min_butterfly_width:
                opportunities.append({
                    "expiry": expiry,
                    "k_low": k_low,
                    "k_mid": k_mid,
                    "k_high": k_high,
                    "v_low": v_low,
                    "v_mid": v_mid,
                    "v_high": v_high,
                    "butterfly_width": butterfly_width,
                    "trade": "buy butterfly (long wings, short body)",
                })

        return pd.DataFrame(opportunities) if opportunities else pd.DataFrame()

    def skew_signal(self, surface_grid: pd.DataFrame, expiry: str = None) -> float:
        """
        Compute the skew signal for a given expiry.
        Skew = (25-delta put vol) - (25-delta call vol).
        Positive skew = put premium.
        """
        if expiry is None:
            expiry = surface_grid.columns[0]
        vol_slice = surface_grid[expiry].sort_index()
        if len(vol_slice) < 3:
            return 0.0
        # Assume strikes are normalized: <1 = put side, >1 = call side
        put_strikes = vol_slice.index[vol_slice.index < 1.0]
        call_strikes = vol_slice.index[vol_slice.index > 1.0]
        if len(put_strikes) == 0 or len(call_strikes) == 0:
            return 0.0
        put_vol = float(vol_slice.loc[put_strikes].iloc[-1])
        call_vol = float(vol_slice.loc[call_strikes].iloc[0])
        return put_vol - call_vol

    def backtest_calendar_strategy(
        self,
        surface_time_series: List[pd.DataFrame],
        dates: pd.DatetimeIndex,
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        """
        Backtest calendar spread arbitrage over time.

        surface_time_series: list of surface_grid DataFrames, one per date.
        Returns cumulative P&L assuming each arb earns (spread - threshold) * scale.
        """
        n = len(dates)
        ec = np.full(n, initial_equity, dtype=float)
        equity = initial_equity
        trades = []
        signals_list = []

        for i, surf in enumerate(surface_time_series[:n]):
            arbs = self.detect_calendar_arb(surf)
            if len(arbs) > 0:
                # Estimate daily P&L from each arb (simplified)
                daily_pnl = float(arbs["spread"].sum() - self.min_calendar_spread * len(arbs))
                daily_pnl *= 0.01  # scale to equity
                equity *= (1 + daily_pnl)
                trades.append(daily_pnl)
                signals_list.append(1.0)
            else:
                signals_list.append(0.0)
            ec[i] = equity

        s_stats = _equity_stats(ec)
        return BacktestResult(
            **s_stats,
            n_trades=len(trades),
            avg_vrp=float(np.mean(trades)) if trades else 0.0,
            equity_curve=pd.Series(ec, index=dates),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=dates[1:]),
            signals=pd.Series(signals_list, index=dates),
            params={"min_calendar_spread": self.min_calendar_spread},
        )

    def no_arbitrage_check(self, surface_grid: pd.DataFrame) -> dict:
        """
        Check the vol surface for no-arbitrage conditions.

        Returns dict with True/False for each condition:
        - calendar_arbitrage_free: no calendar arb violations
        - butterfly_arbitrage_free: no butterfly arb violations
        - n_calendar_violations: count of calendar arb violations
        - n_butterfly_violations: count of butterfly violations
        """
        cal_arbs = self.detect_calendar_arb(surface_grid)
        n_cal = len(cal_arbs)

        n_butterfly = 0
        for exp in surface_grid.columns:
            bf = self.detect_butterfly_arb(surface_grid, exp)
            n_butterfly += len(bf)

        return {
            "calendar_arbitrage_free": n_cal == 0,
            "butterfly_arbitrage_free": n_butterfly == 0,
            "n_calendar_violations": n_cal,
            "n_butterfly_violations": n_butterfly,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    underlying = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)
    rv = pd.Series(np.abs(rng.normal(0.18, 0.04, n)), index=idx)
    iv = rv + rng.normal(0.04, 0.02, n)  # VRP ~ 4% positive on average

    # VRP capture
    vpc = VariancePremiumCapture(realized_window=21, threshold=0.02)
    res1 = vpc.backtest(rv, iv, underlying)
    print("VRP Capture:", res1.summary())
    print("Carry analysis:", vpc.carry_analysis(rv, iv))

    # VIX futures rolldown
    vix = pd.Series(np.abs(rng.normal(18, 4, n)), index=idx).clip(lower=10)
    vix_fut = vix * (1 + rng.normal(0.05, 0.02, n))  # mostly in contango
    inv_vix = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n)), index=idx)

    vfr = VIXFuturesRolldown(min_contango_pct=0.03, max_vix_level=30)
    res2 = vfr.backtest(vix, vix_fut, inv_vix)
    print("VIX Rolldown:", res2.summary())
    print("Roll stats:", vfr.roll_yield_statistics(vix, vix_fut))

    # Vol surface arb
    strikes = [0.8, 0.9, 1.0, 1.1, 1.2]
    surface = pd.DataFrame(
        rng.uniform(0.15, 0.35, (5, 3)),
        index=strikes,
        columns=["1M", "3M", "6M"],
    )
    # Force some calendar arb
    surface.loc[1.0, "1M"] = 0.30
    surface.loc[1.0, "3M"] = 0.22  # near > far → arb!

    vsa = VolatilitySurfaceArbitrage(min_calendar_spread=0.05)
    print("\nCalendar arbs:", vsa.detect_calendar_arb(surface))
    print("Butterfly arbs:", vsa.detect_butterfly_arb(surface, "1M"))
    print("No-arb check:", vsa.no_arbitrage_check(surface))
