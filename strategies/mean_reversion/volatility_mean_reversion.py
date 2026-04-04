"""
volatility_mean_reversion.py — Volatility mean reversion strategies.

VIX reverts to long-run mean after spikes.
Realized vs. implied volatility spread mean-reverts.
Options skew (put/call) oscillates around structural levels.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class BacktestResult:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade_return: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    vol_series: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"Trades={self.n_trades}")


def _stats_from_equity(ec: np.ndarray, trades: list) -> dict:
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
    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x <= 0]
    wr = len(wins) / len(trades) if trades else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return dict(total_return=tot, cagr=cagr, sharpe=sh, sortino=sortino,
                max_drawdown=mdd, calmar=calmar, win_rate=wr, profit_factor=pf,
                n_trades=len(trades), avg_trade_return=float(np.mean(trades)) if trades else 0.0)


def _backtest_signal(close_arr, sig_arr, initial=1_000_000, cost=0.0002):
    n = len(close_arr)
    equity = initial
    ec = np.full(n, initial, dtype=float)
    trades = []
    pos = 0.0
    ep = None
    for i in range(1, n):
        s = float(sig_arr[i - 1]) if not np.isnan(sig_arr[i - 1]) else 0.0
        if s != pos:
            if ep is not None and pos != 0:
                ret = pos * ((close_arr[i] - ep) / ep - cost * 2)
                trades.append(ret)
            pos = s
            ep = close_arr[i] if s != 0 else None
        if pos != 0:
            equity *= (1 + pos * (close_arr[i] - close_arr[i - 1]) / (close_arr[i - 1] + 1e-9))
        ec[i] = equity
    return ec, trades


# ─────────────────────────────────────────────────────────────────────────────
# 1. VIXMeanReversion
# ─────────────────────────────────────────────────────────────────────────────

class VIXMeanReversion:
    """
    VIX Mean Reversion Strategy.

    The VIX (CBOE Volatility Index) tends to spike during market fear events
    and then revert to its long-run mean (~18-20). This strategy:

    1. Sells volatility (buys equities/sells VIX) when VIX is elevated
    2. Buys volatility when VIX is abnormally low

    The z-score is computed against a rolling mean/std of the VIX.

    Parameters
    ----------
    entry_z  : z-score threshold to enter (default 1.5)
    exit_z   : z-score to exit (default 0.0)
    lookback : window for mean/std estimation (default 252)
    """

    def __init__(self, entry_z: float = 1.5, exit_z: float = 0.0, lookback: int = 252):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback

    def compute_vix_zscore(self, vix: pd.Series) -> pd.Series:
        """
        Compute z-score of VIX relative to rolling mean/std.
        Uses log-VIX for more symmetric distribution.
        """
        log_vix = np.log(vix.clip(lower=1.0))
        rolling_mean = log_vix.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        rolling_std = log_vix.rolling(self.lookback, min_periods=self.lookback // 2).std()
        z = (log_vix - rolling_mean) / (rolling_std + 1e-9)
        return z

    def vix_percentile(self, vix: pd.Series) -> pd.Series:
        """Rolling percentile rank of VIX (more robust than z-score)."""
        def rank_pct(x):
            return scipy_stats.percentileofscore(x, x[-1]) / 100.0
        return vix.rolling(self.lookback, min_periods=self.lookback // 2).apply(rank_pct, raw=True)

    def generate_signals(
        self,
        vix: pd.Series,
        underlying_returns: pd.Series = None,
    ) -> pd.Series:
        """
        Generate trading signals.

        Signal on UNDERLYING (e.g., S&P 500):
        +1 = long underlying (VIX spike = buy dip)
        -1 = short underlying (VIX too low = reduce exposure)
         0 = neutral

        Parameters
        ----------
        vix                 : VIX index series
        underlying_returns  : optional underlying return series (unused but kept for interface)
        """
        z = self.compute_vix_zscore(vix)
        signal = pd.Series(0.0, index=vix.index)
        position = 0

        for i in range(self.lookback, len(z)):
            zi = z.iloc[i]
            if np.isnan(zi):
                continue
            if position == 0:
                if zi > self.entry_z:  # VIX spike: buy underlying
                    position = 1
                elif zi < -self.entry_z:  # VIX too low: go short/hedge
                    position = -1
            elif position == 1:
                if zi < self.exit_z:
                    position = 0
            elif position == -1:
                if zi > -self.exit_z:
                    position = 0
            signal.iloc[i] = float(position)

        signal.iloc[:self.lookback] = np.nan
        return signal

    def backtest(
        self,
        vix: pd.Series,
        underlying_price: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        signal = self.generate_signals(vix)
        z = self.compute_vix_zscore(vix)

        ec, trades = _backtest_signal(underlying_price.values, signal.values, initial_equity, commission_pct)
        s = _stats_from_equity(ec, trades)

        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=underlying_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=underlying_price.index[1:]),
            signals=signal,
            vol_series=z,
            params={"entry_z": self.entry_z, "exit_z": self.exit_z, "lookback": self.lookback},
        )

    def vix_regime(self, vix: pd.Series) -> pd.Series:
        """
        Classify VIX into regimes.
        Returns: "LOW" (<15), "NORMAL" (15-25), "HIGH" (25-35), "EXTREME" (>35)
        """
        regimes = pd.Series("NORMAL", index=vix.index)
        regimes[vix < 15] = "LOW"
        regimes[(vix >= 15) & (vix < 25)] = "NORMAL"
        regimes[(vix >= 25) & (vix < 35)] = "HIGH"
        regimes[vix >= 35] = "EXTREME"
        return regimes

    def term_structure_signal(self, vix_spot: pd.Series, vix_3m: pd.Series) -> pd.Series:
        """
        Signal from VIX term structure (slope).
        Backwardation (spot > 3M) → fear/sell equities.
        Contango (spot < 3M) → calm/buy equities.
        """
        slope = (vix_3m - vix_spot) / (vix_spot + 1e-9)
        # Positive slope = contango = bullish; negative = backwardation = bearish
        return np.sign(slope)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VolatilityArbitrage
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityArbitrage:
    """
    Realized vs. Implied Volatility Arbitrage.

    The volatility risk premium (VRP) = implied_vol - realized_vol.
    The VRP is typically positive (IV > RV) because options buyers pay
    a premium for insurance/protection.

    Strategy:
    - Sell options (short vega) when IV >> RV (large VRP → expensive options)
    - Buy options (long vega) when RV >> IV (small/negative VRP → cheap options)

    Implemented as a position on the underlying with vol-targeting.

    Parameters
    ----------
    realized_window : window for realized vol estimation (default 21)
    threshold       : minimum VRP to enter a short vol position (default 0.02)
    long_threshold  : RV-IV gap to enter long vol position (default -0.05)
    vol_target      : target portfolio vol for sizing (default 0.10)
    """

    def __init__(
        self,
        realized_window: int = 21,
        threshold: float = 0.02,
        long_threshold: float = -0.05,
        vol_target: float = 0.10,
    ):
        self.realized_window = realized_window
        self.threshold = threshold
        self.long_threshold = long_threshold
        self.vol_target = vol_target

    def compute_vrp(self, realized_vol: pd.Series, implied_vol: pd.Series) -> pd.Series:
        """
        Volatility risk premium = implied_vol - realized_vol.
        Both should be in annualized terms (e.g., 0.20 = 20%).
        """
        return implied_vol - realized_vol

    def estimate_realized_vol(
        self,
        price_series: pd.Series,
        method: str = "close_to_close",
    ) -> pd.Series:
        """
        Estimate realized volatility using various methods.

        Parameters
        ----------
        price_series : close prices
        method       : "close_to_close", "parkinson", "garman_klass", or "yang_zhang"
        """
        if method == "close_to_close":
            returns = price_series.pct_change()
            return returns.rolling(self.realized_window, min_periods=5).std() * math.sqrt(252)

        elif method == "parkinson":
            # Requires high/low — approximation using 1% range
            log_hl = np.log(price_series * 1.01 / (price_series * 0.99)).rolling(
                self.realized_window, min_periods=5).mean()
            return (log_hl / (4 * math.log(2)) * math.sqrt(252)).apply(math.sqrt)

        return price_series.pct_change().rolling(self.realized_window, min_periods=5).std() * math.sqrt(252)

    def generate_signals(
        self,
        realized_vol: pd.Series,
        implied_vol: pd.Series,
    ) -> pd.Series:
        """
        Generate trading signals based on volatility risk premium.

        Signal on underlying (not options directly):
        +1 = long underlying (low VRP = calm, momentum works)
        -1 = short underlying / hedge (high VRP = fear, sell)
         0 = neutral
        """
        vrp = self.compute_vrp(realized_vol, implied_vol)
        signal = pd.Series(0.0, index=realized_vol.index)

        for i in range(1, len(vrp)):
            v = float(vrp.iloc[i])
            if np.isnan(v):
                continue
            if v > self.threshold:  # IV much > RV → expensive → short vol
                signal.iloc[i] = -1.0  # bearish on underlying (high fear)
            elif v < self.long_threshold:  # RV > IV → cheap options → long vol
                signal.iloc[i] = 1.0   # bullish long vol
            else:
                signal.iloc[i] = 0.0

        return signal

    def vol_adjusted_position(
        self,
        realized_vol: pd.Series,
        implied_vol: pd.Series,
        underlying_price: pd.Series,
    ) -> pd.Series:
        """
        Size positions to hit vol_target, scaled by the VRP signal.
        Returns continuous position size.
        """
        vrp = self.compute_vrp(realized_vol, implied_vol)
        iv = implied_vol.replace(0, np.nan).fillna(method="ffill")
        # Position = (target_vol / IV) * sign(VRP)
        pos = pd.Series(0.0, index=underlying_price.index)
        strong_vrp = vrp.abs() > self.threshold / 2
        pos[strong_vrp] = (self.vol_target / iv[strong_vrp]) * np.sign(vrp[strong_vrp]) * -1
        # Normalize max position to 1
        pos = pos.clip(-2.0, 2.0)
        return pos

    def backtest(
        self,
        realized_vol: pd.Series,
        implied_vol: pd.Series,
        underlying_price: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        signal = self.generate_signals(realized_vol, implied_vol)
        vrp = self.compute_vrp(realized_vol, implied_vol)

        ec, trades = _backtest_signal(underlying_price.values, signal.values, initial_equity, commission_pct)
        s = _stats_from_equity(ec, trades)

        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=underlying_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=underlying_price.index[1:]),
            signals=signal,
            vol_series=vrp,
            params={"realized_window": self.realized_window, "threshold": self.threshold},
        )

    def vrp_statistics(self, realized_vol: pd.Series, implied_vol: pd.Series) -> dict:
        """Summary statistics of the volatility risk premium."""
        vrp = self.compute_vrp(realized_vol, implied_vol)
        return {
            "mean_vrp": float(vrp.mean()),
            "std_vrp": float(vrp.std()),
            "sharpe_vrp": float(vrp.mean() / (vrp.std() + 1e-9) * math.sqrt(252)),
            "pct_positive": float((vrp > 0).mean()),
            "mean_iv": float(implied_vol.mean()),
            "mean_rv": float(realized_vol.mean()),
            "iv_rv_ratio": float((implied_vol / (realized_vol + 1e-9)).mean()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SkewTrading
# ─────────────────────────────────────────────────────────────────────────────

class SkewTrading:
    """
    Options Skew Mean Reversion Strategy.

    The put/call skew (or SKEW index) measures the market's pricing of
    tail risk. When put implied vol significantly exceeds call implied vol
    (high negative skew), the market is pricing in fear/crash risk.
    This tends to be mean-reverting.

    Signal logic:
    - High positive skew (calls > puts): market expects upside → fade, go short
    - High negative skew (puts > calls): market expects crash → fade, go long

    Parameters
    ----------
    lookback   : rolling window for skew normalization (default 63)
    threshold  : z-score threshold for signal (default 1.5)
    exit_z     : z-score to exit (default 0.3)
    """

    def __init__(self, lookback: int = 63, threshold: float = 1.5, exit_z: float = 0.3):
        self.lookback = lookback
        self.threshold = threshold
        self.exit_z = exit_z

    def compute_skew_zscore(self, put_call_skew: pd.Series) -> pd.Series:
        """
        Compute z-score of the put-call skew.

        put_call_skew: can be:
        - Ratio: put_IV / call_IV (typically > 1 meaning puts are more expensive)
        - Slope: -25delta_put_IV + 25delta_call_IV (negative = skew toward puts)
        - CBOE SKEW index value

        Returns z-score relative to rolling mean/std.
        """
        rolling_mean = put_call_skew.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        rolling_std = put_call_skew.rolling(self.lookback, min_periods=self.lookback // 2).std()
        return (put_call_skew - rolling_mean) / (rolling_std + 1e-9)

    def generate_signals(self, put_call_skew: pd.Series) -> pd.Series:
        """
        Generate signals based on skew z-score.

        Convention: high skew (puts expensive) → mean reversion → go long (contrarian).
        Low skew (calls expensive or low fear) → go short.
        """
        z = self.compute_skew_zscore(put_call_skew)
        signal = pd.Series(0.0, index=put_call_skew.index)
        position = 0

        for i in range(self.lookback, len(z)):
            zi = z.iloc[i]
            if np.isnan(zi):
                continue
            if position == 0:
                if zi > self.threshold:  # skew extremely high → mean revert → long
                    position = 1
                elif zi < -self.threshold:  # skew extremely low → complacency → short
                    position = -1
            elif position == 1:
                if zi < self.exit_z:
                    position = 0
            elif position == -1:
                if zi > -self.exit_z:
                    position = 0
            signal.iloc[i] = float(position)

        signal.iloc[:self.lookback] = np.nan
        return signal

    def put_call_ratio_signal(
        self,
        put_volume: pd.Series,
        call_volume: pd.Series,
        smoothing: int = 5,
    ) -> pd.Series:
        """
        Generate signal from put/call volume ratio.
        High P/C ratio = fear → contrarian buy signal.
        Low P/C ratio = complacency → contrarian sell signal.
        """
        pc_ratio = put_volume / (call_volume + 1e-9)
        if smoothing > 1:
            pc_ratio = pc_ratio.ewm(span=smoothing, adjust=False).mean()
        return self.generate_signals(pc_ratio)

    def backtest(
        self,
        put_call_skew: pd.Series,
        underlying_price: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        signal = self.generate_signals(put_call_skew)
        z = self.compute_skew_zscore(put_call_skew)

        ec, trades = _backtest_signal(underlying_price.values, signal.values, initial_equity, commission_pct)
        s = _stats_from_equity(ec, trades)

        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=underlying_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=underlying_price.index[1:]),
            signals=signal,
            vol_series=z,
            params={"lookback": self.lookback, "threshold": self.threshold},
        )

    def skew_regime_analysis(
        self,
        put_call_skew: pd.Series,
        underlying_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Analyze future returns conditional on current skew z-score regime.

        Returns DataFrame showing avg/std/sharpe of returns following
        different skew regimes.
        """
        z = self.compute_skew_zscore(put_call_skew)
        forward_ret = underlying_returns.shift(-5).rolling(5).sum()

        regimes = pd.cut(z, bins=[-np.inf, -2, -1, 0, 1, 2, np.inf],
                         labels=["Extreme Low", "Low", "Neutral-", "Neutral+", "High", "Extreme High"])

        results = []
        for regime in regimes.cat.categories:
            mask = regimes == regime
            fwd = forward_ret[mask].dropna()
            if len(fwd) < 10:
                continue
            results.append({
                "regime": regime,
                "n_obs": len(fwd),
                "mean_5d_return": float(fwd.mean()),
                "std_5d_return": float(fwd.std()),
                "hit_rate": float((fwd > 0).mean()),
                "t_stat": float(scipy_stats.ttest_1samp(fwd, 0).statistic),
            })

        return pd.DataFrame(results).set_index("regime") if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1500
    idx = pd.date_range("2018-01-01", periods=n, freq="D")

    # Simulated VIX (mean-reverting around 18)
    vix = pd.Series(18.0 + np.cumsum(rng.normal(0, 0.3, n) - 0.05 * (np.arange(n) % 20 - 10)), index=idx)
    vix = vix.clip(lower=10.0)

    underlying = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)

    vmr = VIXMeanReversion(entry_z=1.5, exit_z=0.0, lookback=252)
    res1 = vmr.backtest(vix, underlying)
    print("VIX MeanRev:", res1.summary())

    # Realized vs implied vol
    rv = pd.Series(np.abs(rng.normal(0.18, 0.05, n)), index=idx)
    iv = rv + rng.normal(0.03, 0.02, n)  # IV typically above RV by VRP

    va = VolatilityArbitrage(realized_window=21, threshold=0.04)
    res2 = va.backtest(rv, iv, underlying)
    print("Vol Arb:", res2.summary())
    print("VRP stats:", va.vrp_statistics(rv, iv))

    # Skew trading
    skew = pd.Series(1.05 + rng.normal(0, 0.1, n), index=idx)  # put/call skew ratio
    st = SkewTrading(lookback=63, threshold=1.5)
    res3 = st.backtest(skew, underlying)
    print("Skew Trading:", res3.summary())
