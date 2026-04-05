"""
idea-engine/signal-library/macro.py
=====================================
6 macro / on-chain proxy signals for the SRFM Idea Engine Signal Library.

All signals work on OHLCV DataFrames. On-chain metrics are proxied from
price history when raw on-chain data is unavailable.

Signals
-------
1.  MayerMultiple        — price / 200-day MA (Bitcoin valuation indicator)
2.  StockToFlowDeviation — deviation from the S2F power-law price model
3.  PiCycleTop           — 111-day MA crossing 2× the 350-day MA (BTC top)
4.  NUPLProxy            — unrealised profit/loss proxy from price vs long MA
5.  PuellMultiple        — miner revenue proxy (using vol as difficulty proxy)
6.  FearGreedProxy       — composite fear/greed from vol + momentum + breadth
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Dict

import numpy as np
import pandas as pd

from .base import Signal, SignalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. MayerMultiple
# ---------------------------------------------------------------------------

class MayerMultiple(Signal):
    """
    Mayer Multiple: price / 200-day (or N-bar) simple moving average.

    Historical BTC interpretations:
        > 2.4   historically overvalued (potential top)
        1.0-2.4 fair value zone
        < 1.0   historically undervalued (accumulation zone)

    Returns the raw multiple (unbounded above 0).
    """

    name:        str = "mayer_multiple"
    category:    str = "macro"
    lookback:    int = 200
    signal_type: str = "continuous"

    def __init__(self, ma_period: int = 200) -> None:
        self.ma_period = ma_period
        self.lookback  = ma_period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close  = df["Close"]
        ma     = close.rolling(self.ma_period, min_periods=self.ma_period // 2).mean()
        result = close / ma.replace(0.0, np.nan)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 2. StockToFlowDeviation
# ---------------------------------------------------------------------------

class StockToFlowDeviation(Signal):
    """
    Stock-to-Flow model deviation.

    The S2F model predicts Bitcoin's value based on its scarcity (stock / flow
    where flow = newly mined supply). Since we don't have direct halving data
    in an OHLCV frame, we model the S2F price as a power-law function of time:

        S2F_price(t) = a * t^b

    where t is the bar index (a proxy for block count since genesis) and
    a, b are empirical constants derived from BTC history.

    Default constants (a=0.18, b=3.3) approximate the original PlanB S2F model
    in log-log space, calibrated to produce prices in the range of BTC history
    when t represents approximate days since genesis.

    Returns: (log(price) - log(s2f_model_price)) — positive = above model.
    """

    name:        str = "s2f_deviation"
    category:    str = "macro"
    lookback:    int = 1
    signal_type: str = "continuous"

    # Approximate PlanB S2F log-log coefficients
    # log(price) = a + b * log(s2f_ratio)
    # We model S2F ratio as proportional to t (time since genesis in days)
    _A = 0.18       # intercept coefficient
    _B = 3.3        # power exponent

    def __init__(
        self,
        genesis_date:  Optional[str] = "2009-01-03",
        a: float = 0.18,
        b: float = 3.3,
    ) -> None:
        self.genesis_date = pd.Timestamp(genesis_date) if genesis_date else None
        self.a = a
        self.b = b

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        if isinstance(df.index, pd.DatetimeIndex) and self.genesis_date is not None:
            days_since_genesis = (df.index - self.genesis_date).days.clip(lower=1)
            t = days_since_genesis.values.astype(float)
        else:
            # Fall back to sequential bar index
            t = np.arange(1, len(df) + 1, dtype=float)

        # S2F model price (power-law in day count)
        # Calibrated so S2F_price at ~day 4000 ≈ 10k USD scale
        s2f_price = self.a * (t ** self.b)
        s2f_price = np.clip(s2f_price, 1e-6, None)

        log_deviation = np.log(close.values) - np.log(s2f_price)
        result = pd.Series(log_deviation, index=df.index, name=self.name)
        return result


# ---------------------------------------------------------------------------
# 3. PiCycleTop
# ---------------------------------------------------------------------------

class PiCycleTop(Signal):
    """
    Pi Cycle Top indicator (Glassnode / Philip Swift).

    Signal = 111-day SMA - 2 × 350-day SMA

    When the 111-day MA crosses above the 350×2 MA from below, it has
    historically marked BTC cycle tops (within days).

    Returns the raw difference (in price units).
        < 0  safe zone (111-day below 2× 350-day)
        ≈ 0  approaching top signal
        > 0  top signal triggered (historically very rare)
    """

    name:        str = "pi_cycle_top"
    category:    str = "macro"
    lookback:    int = 350
    signal_type: str = "continuous"

    def __init__(self, short_period: int = 111, long_period: int = 350) -> None:
        self.short_period = short_period
        self.long_period  = long_period
        self.lookback     = long_period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close      = df["Close"]
        ma_short   = close.rolling(self.short_period,
                                   min_periods=self.short_period // 2).mean()
        ma_long    = close.rolling(self.long_period,
                                   min_periods=self.long_period // 2).mean()
        result     = ma_short - 2.0 * ma_long
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 4. NUPLProxy
# ---------------------------------------------------------------------------

class NUPLProxy(Signal):
    """
    Net Unrealised Profit/Loss (NUPL) proxy.

    True NUPL requires UTXO-level on-chain data. This proxy approximates it
    using the ratio of current price to a long-term moving average (a proxy
    for average cost basis of long-term holders):

        NUPL_proxy = (price - ma_long) / price

    Range: approximately (-∞, 1)
        > 0.75   euphoria/greed (historically bubble territory)
        0.50-0.75 belief
        0.25-0.50 optimism
        0-0.25   hope
        < 0      capitulation / fear

    Returns raw value (not clamped).
    """

    name:        str = "nupl_proxy"
    category:    str = "macro"
    lookback:    int = 365
    signal_type: str = "continuous"

    def __init__(self, ma_period: int = 365) -> None:
        self.ma_period = ma_period
        self.lookback  = ma_period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close  = df["Close"]
        ma     = close.rolling(self.ma_period, min_periods=self.ma_period // 2).mean()
        result = (close - ma) / close.replace(0.0, np.nan)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 5. PuellMultiple
# ---------------------------------------------------------------------------

class PuellMultiple(Signal):
    """
    Puell Multiple proxy.

    The true Puell Multiple = daily miner issuance USD value / 365-day MA
    of daily issuance. This requires on-chain data.

    Proxy: we use the ratio of short-term realised volatility to its long-term
    mean as a stand-in for "miner revenue pressure" — high short-term vol
    means miners are either accumulating or selling into strength.

    Alternative: if ``daily_issuance_series`` is provided (USD value per bar),
    computes the true Puell-style ratio.

    Returns:
        > 4.0   historically overvalued (miners selling heavily)
        0.5-1.0 accumulation zone
    """

    name:        str = "puell_multiple"
    category:    str = "macro"
    lookback:    int = 365
    signal_type: str = "continuous"

    def __init__(
        self,
        short_period:           int                    = 30,
        long_period:            int                    = 365,
        daily_issuance_series:  Optional[pd.Series]   = None,
    ) -> None:
        self.short_period          = short_period
        self.long_period           = long_period
        self.daily_issuance_series = daily_issuance_series
        self.lookback              = long_period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if self.daily_issuance_series is not None:
            iso   = self.daily_issuance_series.reindex(df.index).ffill()
            ma    = iso.rolling(self.long_period, min_periods=self.long_period // 2).mean()
            result = iso / ma.replace(0.0, np.nan)
        else:
            # Proxy: close-to-close vol ratio
            log_ret    = np.log(df["Close"] / df["Close"].shift(1))
            short_vol  = log_ret.rolling(self.short_period, min_periods=2).std()
            long_vol   = log_ret.rolling(self.long_period,  min_periods=2).std()
            result     = short_vol / long_vol.replace(0.0, np.nan)

        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 6. FearGreedProxy
# ---------------------------------------------------------------------------

class FearGreedProxy(Signal):
    """
    Composite Fear & Greed index proxy (0 = extreme fear, 100 = extreme greed).

    Combines three sub-components:
        1. Volatility component:  low vol → greed; high vol → fear
        2. Momentum component:    strong uptrend → greed; downtrend → fear
        3. Breadth component:     RSI proxy for market sentiment

    Each component is normalised to 0-100, then averaged.
    """

    name:        str = "fear_greed_proxy"
    category:    str = "macro"
    lookback:    int = 90
    signal_type: str = "continuous"

    def __init__(
        self,
        vol_window:  int = 30,
        mom_window:  int = 90,
        rsi_period:  int = 14,
        vol_rank_w:  int = 252,
    ) -> None:
        self.vol_window  = vol_window
        self.mom_window  = mom_window
        self.rsi_period  = rsi_period
        self.vol_rank_w  = vol_rank_w
        self.lookback    = max(vol_window, mom_window, rsi_period, vol_rank_w)

    def _percentile_rank(self, series: pd.Series, window: int) -> pd.Series:
        def _rank(arr: np.ndarray) -> float:
            if len(arr) <= 1:
                return 50.0
            return float(np.sum(arr[:-1] < arr[-1]) / (len(arr) - 1) * 100)
        return series.rolling(window, min_periods=2).apply(_rank, raw=True)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close   = df["Close"]
        log_ret = np.log(close / close.shift(1))

        # ── Volatility component (low vol → high greed)
        rvol       = log_ret.rolling(self.vol_window, min_periods=2).std()
        vol_rank   = self._percentile_rank(rvol, self.vol_rank_w).fillna(50.0)
        vol_score  = 100.0 - vol_rank  # invert: low vol = high greed

        # ── Momentum component (price above long MA → greed)
        ma          = close.rolling(self.mom_window, min_periods=2).mean()
        pct_vs_ma   = (close - ma) / ma.replace(0.0, np.nan) * 100.0
        mom_rank    = self._percentile_rank(pct_vs_ma, self.vol_rank_w).fillna(50.0)
        mom_score   = mom_rank

        # ── RSI component (high RSI = greed)
        delta    = close.diff()
        gain     = delta.clip(lower=0.0)
        loss     = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period,
                            adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period,
                            adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi_val  = 100.0 - (100.0 / (1.0 + rs))
        rsi_score = rsi_val.fillna(50.0)  # RSI is already 0-100

        # ── Composite (equal weight)
        composite = (vol_score + mom_score + rsi_score) / 3.0
        composite = composite.clip(0.0, 100.0)
        composite.name = self.name
        return composite
