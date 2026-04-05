"""
idea-engine/signal-library/cross_asset.py
==========================================
8 cross-asset signals for the SRFM Idea Engine Signal Library.

All signals accept a primary OHLCV DataFrame (df) and optional companion
DataFrames / Series for multi-asset computations.

Signals
-------
1.  BTCDominance         — BTC dominance proxy from price ratios
2.  BTCLead              — BTC price change leading alt price change by N bars
3.  AltSeasonIndex       — fraction of alts outperforming BTC over N days
4.  CorrelationBreakdown — pairwise correlation dropping below threshold
5.  BetaAdjusted         — asset beta-adjusted return vs BTC
6.  FlightToQuality      — alts underperforming BTC (risk-off signal)
7.  CryptoGlobalCap      — total market cap trend signal
8.  DefiVsCex            — DeFi vs CEX relative momentum
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import Signal, SignalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. BTCDominance
# ---------------------------------------------------------------------------

class BTCDominance(Signal):
    """
    BTC Dominance proxy: BTC's share of total combined market cap.

    Given a list of alt close prices, computes:
        dominance = BTC_price / sum(all_prices)

    Returns the dominance ratio (0 to 1). Rising dominance → alts underperforming.

    Parameters
    ----------
    alt_series_dict : dict[str, pd.Series]
        Mapping of symbol → close price Series. Must include "BTC".
        If None, falls back to returning NaN series.
    btc_key : str
        The key in alt_series_dict that represents BTC.
    """

    name:        str = "btc_dominance"
    category:    str = "cross_asset"
    lookback:    int = 1
    signal_type: str = "continuous"

    def __init__(
        self,
        alt_series_dict: Optional[Dict[str, pd.Series]] = None,
        btc_key: str = "BTC",
    ) -> None:
        self.alt_series_dict = alt_series_dict or {}
        self.btc_key         = btc_key

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not self.alt_series_dict or self.btc_key not in self.alt_series_dict:
            logger.warning(
                "BTCDominance: alt_series_dict missing or BTC key absent. "
                "Returning NaN series."
            )
            return pd.Series(np.nan, index=df.index, name=self.name)

        btc_price  = self.alt_series_dict[self.btc_key].reindex(df.index).ffill()
        total      = pd.Series(0.0, index=df.index)
        for sym, series in self.alt_series_dict.items():
            aligned = series.reindex(df.index).ffill().fillna(0.0)
            total   = total + aligned

        result = btc_price / total.replace(0.0, np.nan)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 2. BTCLead
# ---------------------------------------------------------------------------

class BTCLead(Signal):
    """
    BTC Lead Signal: BTC price change lagged N bars as predictor of
    the alt's price change.

    Returns the lagged BTC N-bar log return — positive when BTC was rising N
    bars ago (which may predict the alt rising now).

    Parameters
    ----------
    btc_close : pd.Series
        BTC close prices aligned to df.index.
    lag : int
        Number of bars to lag BTC returns.
    period : int
        Return period (bars).
    """

    name:        str = "btc_lead"
    category:    str = "cross_asset"
    lookback:    int = 10
    signal_type: str = "continuous"

    def __init__(
        self,
        btc_close: Optional[pd.Series] = None,
        lag:       int                  = 4,
        period:    int                  = 4,
    ) -> None:
        self.btc_close = btc_close
        self.lag       = lag
        self.period    = period
        self.lookback  = lag + period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if self.btc_close is None:
            logger.warning("BTCLead: btc_close not provided. Returning NaN.")
            return pd.Series(np.nan, index=df.index, name=self.name)

        btc  = self.btc_close.reindex(df.index).ffill()
        ret  = np.log(btc / btc.shift(self.period))
        # Lag: BTC return from N bars ago
        lagged = ret.shift(self.lag)
        lagged.name = self.name
        return lagged


# ---------------------------------------------------------------------------
# 3. AltSeasonIndex
# ---------------------------------------------------------------------------

class AltSeasonIndex(Signal):
    """
    Alt Season Index: fraction of alts outperforming BTC over N bars.

    Returns a value in [0, 1]:
        > 0.75  alt season (most alts beating BTC)
        < 0.25  BTC dominance (most alts underperforming)
        ~0.5    neutral

    Parameters
    ----------
    alt_series_dict : dict[str, pd.Series]
        Close price series for each asset including BTC.
    btc_key : str
        Key in alt_series_dict for BTC.
    period : int
        Return lookback window (bars).
    """

    name:        str = "alt_season_index"
    category:    str = "cross_asset"
    lookback:    int = 30
    signal_type: str = "continuous"

    def __init__(
        self,
        alt_series_dict: Optional[Dict[str, pd.Series]] = None,
        btc_key: str = "BTC",
        period:  int = 30,
    ) -> None:
        self.alt_series_dict = alt_series_dict or {}
        self.btc_key         = btc_key
        self.period          = period
        self.lookback        = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not self.alt_series_dict or self.btc_key not in self.alt_series_dict:
            return pd.Series(np.nan, index=df.index, name=self.name)

        btc      = self.alt_series_dict[self.btc_key].reindex(df.index).ffill()
        btc_ret  = np.log(btc / btc.shift(self.period))

        alt_rets = []
        for sym, series in self.alt_series_dict.items():
            if sym == self.btc_key:
                continue
            aligned = series.reindex(df.index).ffill()
            alt_rets.append(np.log(aligned / aligned.shift(self.period)))

        if not alt_rets:
            return pd.Series(np.nan, index=df.index, name=self.name)

        alt_ret_df  = pd.concat(alt_rets, axis=1)
        # Fraction of alts outperforming BTC row-wise
        outperform  = (alt_ret_df.gt(btc_ret, axis=0)).sum(axis=1) / alt_ret_df.shape[1]
        outperform.name = self.name
        return outperform


# ---------------------------------------------------------------------------
# 4. CorrelationBreakdown
# ---------------------------------------------------------------------------

class CorrelationBreakdown(Signal):
    """
    Correlation Breakdown: rolling pairwise correlation between the primary
    asset and a reference asset (BTC by default).

    Returns the rolling correlation. Low/negative values signal
    de-correlation (portfolio diversification is improving; risk-off regime).

    Parameters
    ----------
    reference_close : pd.Series
        Close prices of the reference asset (e.g. BTC).
    window : int
        Rolling correlation window (bars).
    threshold : float
        If provided, returns a binary signal (1 if corr < threshold).
    """

    name:        str = "correlation_breakdown"
    category:    str = "cross_asset"
    lookback:    int = 60
    signal_type: str = "continuous"

    def __init__(
        self,
        reference_close: Optional[pd.Series] = None,
        window:          int                  = 60,
        threshold:       Optional[float]      = None,
    ) -> None:
        self.reference_close = reference_close
        self.window          = window
        self.threshold       = threshold
        self.lookback        = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if self.reference_close is None:
            logger.warning("CorrelationBreakdown: reference_close not provided.")
            return pd.Series(np.nan, index=df.index, name=self.name)

        asset_ret = np.log(df["Close"] / df["Close"].shift(1))
        ref       = self.reference_close.reindex(df.index).ffill()
        ref_ret   = np.log(ref / ref.shift(1))

        corr = asset_ret.rolling(self.window, min_periods=self.window // 2).corr(ref_ret)

        if self.threshold is not None:
            result = (corr < self.threshold).astype(float)
        else:
            result = corr

        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 5. BetaAdjusted
# ---------------------------------------------------------------------------

class BetaAdjusted(Signal):
    """
    Beta-adjusted excess return vs BTC.

    Computes rolling beta of the asset vs BTC, then:
        alpha = asset_return - beta * btc_return

    Positive alpha → asset outperforming BTC on a risk-adjusted basis.
    Negative alpha → asset underperforming.

    Parameters
    ----------
    btc_close : pd.Series
        BTC close prices.
    window : int
        Rolling window for beta estimation and return computation.
    """

    name:        str = "beta_adjusted"
    category:    str = "cross_asset"
    lookback:    int = 60
    signal_type: str = "continuous"

    def __init__(
        self,
        btc_close: Optional[pd.Series] = None,
        window:    int                  = 60,
    ) -> None:
        self.btc_close = btc_close
        self.window    = window
        self.lookback  = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if self.btc_close is None:
            return pd.Series(np.nan, index=df.index, name=self.name)

        asset_ret = np.log(df["Close"] / df["Close"].shift(1))
        btc       = self.btc_close.reindex(df.index).ffill()
        btc_ret   = np.log(btc / btc.shift(1))

        # Rolling beta = cov(asset, btc) / var(btc)
        roll_cov = asset_ret.rolling(self.window, min_periods=self.window // 2).cov(btc_ret)
        roll_var = btc_ret.rolling(self.window, min_periods=self.window // 2).var()
        beta     = roll_cov / roll_var.replace(0.0, np.nan)

        alpha    = asset_ret - beta * btc_ret
        alpha.name = self.name
        return alpha


# ---------------------------------------------------------------------------
# 6. FlightToQuality
# ---------------------------------------------------------------------------

class FlightToQuality(Signal):
    """
    Flight-to-quality signal: alts significantly underperforming BTC.

    A positive signal (risk-off) is produced when the average alt underperforms
    BTC by more than a threshold over the window.

    Returns continuous value: (BTC_return - mean_alt_return) over the window.
    Positive → BTC outperforming alts (de-risking, flight to BTC quality).
    Negative → alts outperforming (risk-on, alt season).

    Parameters
    ----------
    alt_series_dict : dict[str, pd.Series]
    btc_key : str
    period : int
    """

    name:        str = "flight_to_quality"
    category:    str = "cross_asset"
    lookback:    int = 30
    signal_type: str = "continuous"

    def __init__(
        self,
        alt_series_dict: Optional[Dict[str, pd.Series]] = None,
        btc_key: str = "BTC",
        period:  int = 30,
    ) -> None:
        self.alt_series_dict = alt_series_dict or {}
        self.btc_key         = btc_key
        self.period          = period
        self.lookback        = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not self.alt_series_dict or self.btc_key not in self.alt_series_dict:
            return pd.Series(np.nan, index=df.index, name=self.name)

        btc      = self.alt_series_dict[self.btc_key].reindex(df.index).ffill()
        btc_ret  = np.log(btc / btc.shift(self.period))

        alt_rets: List[pd.Series] = []
        for sym, series in self.alt_series_dict.items():
            if sym == self.btc_key:
                continue
            aligned = series.reindex(df.index).ffill()
            alt_rets.append(np.log(aligned / aligned.shift(self.period)))

        if not alt_rets:
            return pd.Series(np.nan, index=df.index, name=self.name)

        mean_alt_ret = pd.concat(alt_rets, axis=1).mean(axis=1)
        result       = btc_ret - mean_alt_ret
        result.name  = self.name
        return result


# ---------------------------------------------------------------------------
# 7. CryptoGlobalCap
# ---------------------------------------------------------------------------

class CryptoGlobalCap(Signal):
    """
    Crypto Global Market Cap trend signal.

    Computes the momentum of the summed market caps (or price proxies) across
    all assets provided. A rising total cap is bullish for the space.

    Returns N-bar log-return of total cap (rolling EMA for smoothness).

    Parameters
    ----------
    cap_series_dict : dict[str, pd.Series]
        Close price (or cap) series for each asset.
    period : int
        Return period.
    smooth_span : int
        EMA span for smoothing the total cap before computing returns.
    """

    name:        str = "crypto_global_cap"
    category:    str = "cross_asset"
    lookback:    int = 30
    signal_type: str = "continuous"

    def __init__(
        self,
        cap_series_dict: Optional[Dict[str, pd.Series]] = None,
        period:          int = 20,
        smooth_span:     int = 5,
    ) -> None:
        self.cap_series_dict = cap_series_dict or {}
        self.period          = period
        self.smooth_span     = smooth_span
        self.lookback        = period + smooth_span

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not self.cap_series_dict:
            return pd.Series(np.nan, index=df.index, name=self.name)

        total = pd.Series(0.0, index=df.index)
        for sym, series in self.cap_series_dict.items():
            aligned = series.reindex(df.index).ffill().fillna(0.0)
            total   = total + aligned

        # Smooth then compute return
        smoothed = total.ewm(span=self.smooth_span, min_periods=1, adjust=False).mean()
        result   = np.log(smoothed / smoothed.shift(self.period))
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 8. DefiVsCex
# ---------------------------------------------------------------------------

class DefiVsCex(Signal):
    """
    DeFi vs CEX token relative momentum.

    Compares the average return of DeFi tokens to CEX tokens over N bars.
    Positive → DeFi outperforming (risk appetite, on-chain activity up).
    Negative → CEX outperforming (risk-off, centralised flows dominating).

    Parameters
    ----------
    defi_series_dict : dict[str, pd.Series]
        Close prices of DeFi tokens (UNI, AAVE, CRV, etc.).
    cex_series_dict : dict[str, pd.Series]
        Close prices of CEX tokens (BNB, OKB, CRO, etc.).
    period : int
        Return period.
    """

    name:        str = "defi_vs_cex"
    category:    str = "cross_asset"
    lookback:    int = 30
    signal_type: str = "continuous"

    def __init__(
        self,
        defi_series_dict: Optional[Dict[str, pd.Series]] = None,
        cex_series_dict:  Optional[Dict[str, pd.Series]] = None,
        period:           int = 30,
    ) -> None:
        self.defi_series_dict = defi_series_dict or {}
        self.cex_series_dict  = cex_series_dict  or {}
        self.period           = period
        self.lookback         = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def _avg_ret(series_dict: Dict[str, pd.Series]) -> Optional[pd.Series]:
            rets = []
            for sym, s in series_dict.items():
                aligned = s.reindex(df.index).ffill()
                rets.append(np.log(aligned / aligned.shift(self.period)))
            if not rets:
                return None
            return pd.concat(rets, axis=1).mean(axis=1)

        defi_ret = _avg_ret(self.defi_series_dict)
        cex_ret  = _avg_ret(self.cex_series_dict)

        if defi_ret is None or cex_ret is None:
            logger.warning("DefiVsCex: insufficient data. Returning NaN.")
            return pd.Series(np.nan, index=df.index, name=self.name)

        result      = defi_ret - cex_ret
        result.name = self.name
        return result
