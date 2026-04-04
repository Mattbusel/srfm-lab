"""
crypto/crypto_momentum.py — Crypto-specific momentum and breadth signals.

BTC dominance cycle, market breadth, on-chain whale accumulation proxy.
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
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    indicator_series: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"Trades={self.n_trades}")


def _stats(ec: np.ndarray) -> dict:
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


def _bt(close, signal, initial=1_000_000, cost=0.001):
    n = len(close)
    equity = initial
    ec = np.full(n, initial, dtype=float)
    trades = []
    pos = 0.0
    ep = None
    for i in range(1, n):
        s = float(signal[i - 1]) if not np.isnan(signal[i - 1]) else 0.0
        if s != pos:
            if ep is not None and pos != 0:
                trades.append(pos * ((close[i] - ep) / ep - cost * 2))
            pos = s
            ep = close[i] if s != 0 else None
        if pos != 0:
            equity *= (1 + pos * (close[i] - close[i-1]) / (close[i-1] + 1e-9))
        ec[i] = equity
    return ec, trades


# ─────────────────────────────────────────────────────────────────────────────
# 1. BitcoinDominance
# ─────────────────────────────────────────────────────────────────────────────

class BitcoinDominance:
    """
    Bitcoin Dominance Cycle Strategy.

    Bitcoin dominance = BTC market cap / Total crypto market cap.

    Interpretation:
    - Rising BTC dominance: money flows into BTC, alts underperform (risk-off)
    - Falling BTC dominance: alt season, altcoins outperform BTC

    Strategy:
    - When BTC dominance is falling: go long altcoins vs BTC (alt season)
    - When BTC dominance is rising: rotate to BTC or go defensive

    Parameters
    ----------
    threshold      : dominance change threshold to trigger signal (default 0.02 = 2%)
    lookback       : window for trend detection (default 30)
    smoothing      : smoothing for dominance series (default 7)
    """

    def __init__(
        self,
        threshold: float = 0.02,
        lookback: int = 30,
        smoothing: int = 7,
    ):
        self.threshold = threshold
        self.lookback = lookback
        self.smoothing = smoothing

    def smooth_dominance(self, btc_dom_series: pd.Series) -> pd.Series:
        """Apply exponential smoothing to dominance series."""
        return btc_dom_series.ewm(span=self.smoothing, adjust=False).mean()

    def dominance_trend(self, btc_dom_series: pd.Series) -> pd.Series:
        """
        Compute the trend in BTC dominance.
        Positive = rising dominance. Negative = falling.
        """
        smooth_dom = self.smooth_dominance(btc_dom_series)
        trend = smooth_dom - smooth_dom.shift(self.lookback)
        return trend

    def generate_signals(
        self,
        btc_dom_series: pd.Series,
        target: str = "altcoins",
    ) -> pd.Series:
        """
        Generate signals for either 'altcoins' or 'btc' trading vehicle.

        If target='altcoins': +1 during alt season (falling BTC dom)
        If target='btc': +1 when BTC dominance rising (BTC strength)

        Parameters
        ----------
        btc_dom_series : BTC dominance series (fraction, e.g., 0.45 = 45%)
        target         : 'altcoins' or 'btc'
        """
        trend = self.dominance_trend(btc_dom_series)
        signal = pd.Series(0.0, index=btc_dom_series.index)
        position = 0

        for i in range(self.lookback, len(trend)):
            t = float(trend.iloc[i])
            if np.isnan(t):
                continue

            if target == "altcoins":
                # Alt season: falling BTC dom → go long alts
                if position == 0 and t < -self.threshold:
                    position = 1
                elif position == 1 and t > self.threshold * 0.5:
                    position = 0
            else:  # BTC
                # BTC strength: rising dominance → go long BTC
                if position == 0 and t > self.threshold:
                    position = 1
                elif position == 1 and t < -self.threshold * 0.5:
                    position = 0

            signal.iloc[i] = float(position)

        signal.iloc[:self.lookback] = np.nan
        return signal

    def backtest(
        self,
        btc_dom_series: pd.Series,
        price_series: pd.Series,
        target: str = "altcoins",
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        signal = self.generate_signals(btc_dom_series, target)
        trend = self.dominance_trend(btc_dom_series)

        ec, trades = _bt(price_series.values, signal.values, initial_equity)
        s = _stats(ec)

        return BacktestResult(
            **s, n_trades=len(trades),
            equity_curve=pd.Series(ec, index=price_series.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=price_series.index[1:]),
            signals=signal, indicator_series=trend,
            params={"threshold": self.threshold, "lookback": self.lookback, "target": target},
        )

    def alt_season_indicator(self, btc_dom_series: pd.Series) -> pd.Series:
        """
        Binary alt season indicator.
        Returns 1 (alt season) or 0 (BTC season) based on dominance trend.
        """
        trend = self.dominance_trend(btc_dom_series)
        return (trend < -self.threshold).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CryptoBreadth
# ─────────────────────────────────────────────────────────────────────────────

class CryptoBreadth:
    """
    Crypto Market Breadth Indicator.

    Market breadth = number of coins trading above their N-day MA.
    High breadth = broad-based rally (healthy bull). Low breadth = weak.

    Parameters
    ----------
    ma_period          : MA period for coin-level filter (default 50)
    bullish_threshold  : breadth level for bullish signal (default 0.6 = 60%)
    bearish_threshold  : breadth level for bearish signal (default 0.4)
    smoothing          : smoothing window for breadth (default 5)
    """

    def __init__(
        self,
        ma_period: int = 50,
        bullish_threshold: float = 0.6,
        bearish_threshold: float = 0.4,
        smoothing: int = 5,
    ):
        self.ma_period = ma_period
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.smoothing = smoothing

    def compute_breadth(self, universe_prices: pd.DataFrame) -> pd.Series:
        """
        Compute fraction of coins above their N-day MA.

        Parameters
        ----------
        universe_prices : DataFrame, columns = coins, rows = dates
        """
        ma = universe_prices.rolling(self.ma_period, min_periods=self.ma_period // 2).mean()
        above_ma = (universe_prices > ma).astype(float)
        breadth = above_ma.mean(axis=1)

        if self.smoothing > 1:
            breadth = breadth.ewm(span=self.smoothing, adjust=False).mean()

        return breadth

    def compute_breadth_momentum(
        self,
        breadth: pd.Series,
        momentum_window: int = 20,
    ) -> pd.Series:
        """
        Breadth momentum: rate of change in breadth.
        Rising breadth + > threshold = strong bull signal.
        """
        return breadth - breadth.shift(momentum_window)

    def generate_signals(self, n_coins_above_ma: pd.Series) -> pd.Series:
        """
        Generate signals from breadth series.

        n_coins_above_ma: fraction of coins above MA (0 to 1)
        or a pre-computed breadth series.

        +1 = bullish breadth (majority of coins above MA)
        -1 = bearish breadth (minority above MA)
        0  = neutral
        """
        # Handle if given as raw number (not fraction)
        if n_coins_above_ma.max() > 1.5:
            # Normalize by max
            breadth = n_coins_above_ma / n_coins_above_ma.max()
        else:
            breadth = n_coins_above_ma

        signal = pd.Series(0.0, index=breadth.index)
        position = 0

        for i in range(1, len(breadth)):
            b = float(breadth.iloc[i])
            if np.isnan(b):
                continue
            if position == 0:
                if b > self.bullish_threshold:
                    position = 1
                elif b < self.bearish_threshold:
                    position = -1
            elif position == 1:
                if b < self.bearish_threshold:
                    position = 0
            elif position == -1:
                if b > self.bullish_threshold:
                    position = 0
            signal.iloc[i] = float(position)

        return signal

    def backtest(
        self,
        universe_prices: pd.DataFrame,
        benchmark_price: pd.Series,
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        """
        Backtest breadth strategy on benchmark price.

        Parameters
        ----------
        universe_prices : prices of all coins in universe
        benchmark_price : price of asset to trade on signals (e.g., BTC or ETH)
        """
        breadth = self.compute_breadth(universe_prices)
        signal = self.generate_signals(breadth)
        signal = signal.reindex(benchmark_price.index).fillna(0)

        ec, trades = _bt(benchmark_price.values, signal.values, initial_equity)
        s = _stats(ec)

        return BacktestResult(
            **s, n_trades=len(trades),
            equity_curve=pd.Series(ec, index=benchmark_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=benchmark_price.index[1:]),
            signals=signal, indicator_series=breadth.reindex(benchmark_price.index),
            params={"ma_period": self.ma_period, "bullish_threshold": self.bullish_threshold},
        )

    def breadth_statistics(self, universe_prices: pd.DataFrame) -> dict:
        """Summary statistics of breadth."""
        breadth = self.compute_breadth(universe_prices)
        return {
            "mean_breadth": float(breadth.mean()),
            "current_breadth": float(breadth.iloc[-1]),
            "pct_time_bullish": float((breadth > self.bullish_threshold).mean()),
            "pct_time_bearish": float((breadth < self.bearish_threshold).mean()),
            "breadth_trend_20d": float((breadth.iloc[-1] - breadth.iloc[-20]) if len(breadth) >= 20 else 0.0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. WhaleWatcher
# ─────────────────────────────────────────────────────────────────────────────

class WhaleWatcher:
    """
    On-chain Whale Accumulation Proxy.

    Large transactions on-chain (whale movements) can signal accumulation
    or distribution. High whale activity with rising prices = accumulation.
    High whale activity with falling prices = distribution.

    Parameters
    ----------
    threshold        : large transaction threshold (default 1e6 USD)
    lookback         : window for signal normalization (default 30)
    smoothing        : smoothing for transaction series (default 7)
    """

    def __init__(
        self,
        threshold: float = 1_000_000,
        lookback: int = 30,
        smoothing: int = 7,
    ):
        self.threshold = threshold
        self.lookback = lookback
        self.smoothing = smoothing

    def compute_whale_index(
        self,
        large_tx_series: pd.Series,
        price: pd.Series,
    ) -> pd.Series:
        """
        Compute whale accumulation index.

        Whale Index = (large_tx_volume / total_volume) * price_momentum

        High whale index with positive momentum = accumulation signal.

        Parameters
        ----------
        large_tx_series : USD value of large transactions per day
        price           : price series
        """
        # Smooth large tx volume
        smooth_tx = large_tx_series.ewm(span=self.smoothing, adjust=False).mean()

        # Price momentum
        price_mom = price.pct_change(self.lookback).fillna(0)

        # Whale-price alignment index
        whale_index = np.sign(smooth_tx) * price_mom

        return whale_index

    def accumulation_score(
        self,
        large_tx_series: pd.Series,
        price: pd.Series,
    ) -> pd.Series:
        """
        Compute accumulation score:
        - +1: whale buying (large inflows + price rising)
        - -1: whale selling (large outflows + price falling)
        - 0: no signal
        """
        smooth_tx = large_tx_series.ewm(span=self.smoothing, adjust=False).mean()
        # Rolling z-score of tx volume
        tx_mean = smooth_tx.rolling(self.lookback, min_periods=5).mean()
        tx_std = smooth_tx.rolling(self.lookback, min_periods=5).std()
        tx_z = (smooth_tx - tx_mean) / (tx_std + 1e-9)

        # Price trend
        price_trend = np.sign(price - price.rolling(self.lookback, min_periods=5).mean())

        # Accumulation: high tx + rising price
        score = tx_z * price_trend
        return score

    def generate_signals(
        self,
        large_tx_series: pd.Series,
        price: pd.Series,
    ) -> pd.Series:
        """
        Signal: +1 whale accumulation (buy), -1 distribution (sell), 0 neutral.
        """
        score = self.accumulation_score(large_tx_series, price)

        # Normalize score
        score_mean = score.rolling(self.lookback * 2, min_periods=10).mean()
        score_std = score.rolling(self.lookback * 2, min_periods=10).std()
        z = (score - score_mean) / (score_std + 1e-9)

        signal = pd.Series(0.0, index=large_tx_series.index)
        signal[z > 1.0] = 1.0   # strong accumulation
        signal[z < -1.0] = -1.0  # strong distribution
        signal.iloc[:self.lookback * 2] = np.nan
        return signal

    def backtest(
        self,
        large_tx_series: pd.Series,
        price: pd.Series,
        initial_equity: float = 1_000_000,
    ) -> BacktestResult:
        signal = self.generate_signals(large_tx_series, price)
        score = self.accumulation_score(large_tx_series, price)

        ec, trades = _bt(price.values, signal.values, initial_equity)
        s = _stats(ec)

        return BacktestResult(
            **s, n_trades=len(trades),
            equity_curve=pd.Series(ec, index=price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=price.index[1:]),
            signals=signal, indicator_series=score,
            params={"threshold": self.threshold, "lookback": self.lookback},
        )

    def whale_alert_summary(
        self,
        large_tx_series: pd.Series,
        price: pd.Series,
        n_recent: int = 30,
    ) -> dict:
        """Summary of recent whale activity."""
        recent_tx = large_tx_series.tail(n_recent)
        recent_price = price.tail(n_recent)
        score = self.accumulation_score(large_tx_series, price)

        return {
            "mean_large_tx_30d": float(recent_tx.mean()),
            "std_large_tx_30d": float(recent_tx.std()),
            "current_score": float(score.iloc[-1]) if len(score) > 0 else 0.0,
            "price_trend_30d": float((price.iloc[-1] - price.iloc[-min(30, len(price))]) / price.iloc[-min(30, len(price))]),
            "signal": "ACCUMULATION" if (score.iloc[-1] > 0.5 if len(score) > 0 else False) else
                      "DISTRIBUTION" if (score.iloc[-1] < -0.5 if len(score) > 0 else False) else "NEUTRAL",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 730
    idx = pd.date_range("2022-01-01", periods=n, freq="D")

    # BTC price
    btc = pd.Series(40000.0 * np.cumprod(1 + rng.normal(0.001, 0.04, n)), index=idx)

    # Alt coin price (more volatile)
    alts = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.002, 0.06, n)), index=idx)

    # BTC dominance series
    btc_dom = pd.Series(0.45 + np.cumsum(rng.normal(0, 0.003, n)), index=idx).clip(0.3, 0.7)

    # Bitcoin Dominance Strategy
    bd = BitcoinDominance(threshold=0.02, lookback=30)
    res1 = bd.backtest(btc_dom, alts, target="altcoins")
    print("Bitcoin Dominance (alt season):", res1.summary())

    # Crypto Breadth
    coins = pd.DataFrame({
        f"coin_{i}": 100.0 * np.cumprod(1 + rng.normal(0.001 + i*0.0001, 0.05, n))
        for i in range(20)
    }, index=idx)
    breadth = CryptoBreadth(ma_period=50, bullish_threshold=0.6)
    res2 = breadth.backtest(coins, btc)
    print("Crypto Breadth:", res2.summary())
    print("Breadth stats:", breadth.breadth_statistics(coins))

    # Whale Watcher
    large_tx = pd.Series(rng.exponential(scale=500_000, size=n), index=idx)
    # Correlate with price somewhat
    large_tx = large_tx + btc.pct_change().fillna(0) * 1e6
    ww = WhaleWatcher(threshold=1e6, lookback=30)
    res3 = ww.backtest(large_tx, btc)
    print("Whale Watcher:", res3.summary())
    print("Whale alert:", ww.whale_alert_summary(large_tx, btc))
