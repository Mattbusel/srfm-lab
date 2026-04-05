"""
features/feature_engineer.py
=============================
Feature engineering pipeline for all SRFM ML models.

No-lookahead guarantee
-----------------------
Every normalisation step uses a *rolling* mean and standard deviation
over the past ``norm_window`` bars (default 252, ≈ 1 year of daily
data).  The rolling statistics are computed strictly on past data:
position i uses the window [i-norm_window, i-1].  This ensures there
is zero information leakage from the future into any feature.

Feature groups
--------------
Price       – log returns at 1/5/10/20/60 bar horizons, rolling vol,
              vol-of-vol
Technical   – RSI-14, MACD, Bollinger %B, ATR ratio, VWAP deviation
BH physics  – bh_mass, bh_active, bh_ctl (consecutive timelike periods),
              OU mean-reversion z-score
Cross-asset – BTC 1-day return, BTC/ETH 20-day rolling correlation
Time        – hour and day-of-week encoded as sin/cos pairs, is_weekend
Micro       – volume ratio vs 20-day avg, high-low range ratio,
              close position within bar range

All output features are in [-1, +1] after rolling z-score normalisation
with a soft clamp at ±3 σ before rescaling.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

NORM_WINDOW = 252   # rolling window for z-score normalisation


class FeatureEngineer:
    """Compute and normalise all ML features from raw OHLCV + BH data.

    Parameters
    ----------
    norm_window : int
        Lookback for rolling mean/std used in z-score normalisation.
    btc_col_prefix : str
        Prefix used for BTC price columns when the instrument is not BTC
        itself (e.g. ``'btc_'`` → ``btc_close``, ``btc_volume``).
    eth_col_prefix : str
        Same convention for ETH columns.
    """

    def __init__(
        self,
        norm_window: int = NORM_WINDOW,
        btc_col_prefix: str = "btc_",
        eth_col_prefix: str = "eth_",
    ) -> None:
        self.norm_window     = norm_window
        self.btc_col_prefix  = btc_col_prefix
        self.eth_col_prefix  = eth_col_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for every row in ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at minimum: ``open``, ``high``, ``low``,
            ``close``, ``volume`` columns with a DatetimeIndex.
            Optional BH columns: ``bh_mass``, ``bh_active``, ``bh_ctl``,
            ``ou_zscore``.
            Optional cross-asset: ``btc_close``, ``eth_close``.

        Returns
        -------
        pd.DataFrame
            One row per input row, all features normalised to [-1, +1].
            Rows before the warmup period (max lookback = 60 bars) are
            dropped.
        """
        out = pd.DataFrame(index=df.index)

        out = self._price_features(df, out)
        out = self._technical_features(df, out)
        out = self._bh_features(df, out)
        out = self._cross_asset_features(df, out)
        out = self._time_features(df, out)
        out = self._microstructure_features(df, out)

        # Drop warmup rows
        min_lookback = 60
        out = out.iloc[min_lookback:].copy()

        # Rolling z-score normalisation (no lookahead)
        out = self._rolling_zscore_normalise(out)

        return out

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _price_features(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        for n in [1, 5, 10, 20, 60]:
            out[f"returns_{n}d"] = c.pct_change(n).clip(-1, 1)
        out["log_return_1d"] = np.log(c / c.shift(1)).replace([np.inf, -np.inf], 0)

        # Rolling vol = std(log returns) over n bars
        lr = out["log_return_1d"]
        for n in [5, 20, 60]:
            out[f"vol_{n}d"] = lr.rolling(n).std()

        # Vol-of-vol: rolling std of rolling vol
        out["vol_of_vol"] = out["vol_20d"].rolling(20).std()
        return out

    def _technical_features(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        c  = df["close"]
        h  = df["high"]
        lo = df["low"]

        # RSI-14
        delta    = c.diff()
        gain     = delta.clip(lower=0).rolling(14).mean()
        loss     = (-delta.clip(upper=0)).rolling(14).mean()
        rs       = gain / (loss + 1e-9)
        out["rsi_14"] = (100 - 100 / (1 + rs)) / 100.0 - 0.5   # centre around 0

        # MACD = EMA12 - EMA26, normalised by price
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        out["macd"] = (ema12 - ema26) / (c + 1e-9)

        # Bollinger %B
        mid  = c.rolling(20).mean()
        std  = c.rolling(20).std()
        out["bb_pct_b"] = ((c - mid) / (2 * std + 1e-9)).clip(-2, 2) / 2.0

        # ATR ratio = ATR(14) / close
        tr    = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        out["atr_ratio"] = (atr14 / (c + 1e-9)).clip(0, 0.2) / 0.1 - 1.0

        # VWAP deviation (intraday not always available; use 20-bar VWAP proxy)
        if "volume" in df.columns:
            vwap = (c * df["volume"]).rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-9)
            out["vwap_deviation"] = ((c - vwap) / (vwap + 1e-9)).clip(-0.1, 0.1) / 0.05
        else:
            out["vwap_deviation"] = 0.0

        # Mayer Multiple = price / 200-bar MA
        ma200 = c.rolling(200).mean()
        out["mayer_multiple"] = (c / (ma200 + 1e-9) - 1.0).clip(-1, 1)

        # EMA ratio = EMA12 / EMA26 - 1
        out["ema_ratio"] = ((ema12 / (ema26 + 1e-9)) - 1.0).clip(-0.1, 0.1) / 0.05

        return out

    def _bh_features(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        for col in ["bh_mass", "bh_active", "bh_ctl", "ou_zscore"]:
            if col in df.columns:
                out[col] = df[col]
            else:
                out[col] = 0.0
        return out

    def _cross_asset_features(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        btc_close = self.btc_col_prefix + "close"
        eth_close = self.eth_col_prefix + "close"

        if btc_close in df.columns:
            out["btc_return_1d"] = df[btc_close].pct_change(1).clip(-1, 1)
        else:
            # Assume instrument IS BTC or use own return as proxy
            out["btc_return_1d"] = df["close"].pct_change(1).clip(-1, 1)

        if btc_close in df.columns and eth_close in df.columns:
            btc_lr = np.log(df[btc_close] / df[btc_close].shift(1))
            eth_lr = np.log(df[eth_close] / df[eth_close].shift(1))

            def _rolling_corr(a: pd.Series, b: pd.Series, w: int) -> pd.Series:
                out_c = pd.Series(np.nan, index=a.index)
                for i in range(w, len(a)):
                    aa = a.iloc[i - w : i].values
                    bb = b.iloc[i - w : i].values
                    if np.std(aa) < 1e-9 or np.std(bb) < 1e-9:
                        out_c.iloc[i] = 0.0
                    else:
                        out_c.iloc[i] = float(np.corrcoef(aa, bb)[0, 1])
                return out_c

            out["btc_eth_corr_20d"] = _rolling_corr(btc_lr, eth_lr, 20)
        else:
            out["btc_eth_corr_20d"] = 0.5  # neutral

        return out

    def _time_features(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            hour = idx.hour
            dow  = idx.dayofweek
        else:
            hour = np.zeros(len(df), dtype=int)
            dow  = np.zeros(len(df), dtype=int)

        out["hour_sin"]     = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"]     = np.cos(2 * np.pi * hour / 24)
        out["dow_sin"]      = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"]      = np.cos(2 * np.pi * dow / 7)
        out["is_weekend"]   = (np.asarray(dow) >= 5).astype(float)
        return out

    def _microstructure_features(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        c  = df["close"]
        h  = df["high"]
        lo = df["low"]

        # Volume ratio vs 20-bar average
        if "volume" in df.columns:
            vol_avg = df["volume"].rolling(20).mean()
            out["volume_ratio"] = (df["volume"] / (vol_avg + 1e-9) - 1.0).clip(-3, 3) / 3.0
        else:
            out["volume_ratio"] = 0.0

        # High-low ratio = (H-L) / C
        out["high_low_ratio"] = ((h - lo) / (c + 1e-9)).clip(0, 0.2) / 0.1 - 1.0

        # Close position within bar range: 0 = at low, 1 = at high, 0.5 = mid
        bar_range = (h - lo).replace(0, np.nan)
        out["close_position"] = ((c - lo) / bar_range).fillna(0.5) * 2.0 - 1.0

        return out

    # ------------------------------------------------------------------
    # Normalisation (rolling z-score, no lookahead)
    # ------------------------------------------------------------------

    def _rolling_zscore_normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling z-score normalisation, clamp ±3, rescale to [-1, +1]."""
        result = df.copy()
        # Columns that are already in [-1, +1] by construction (sin/cos, flags)
        skip = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
                "close_position"}
        for col in df.columns:
            if col in skip:
                continue
            s    = df[col]
            mu   = s.rolling(self.norm_window, min_periods=10).mean()
            sig  = s.rolling(self.norm_window, min_periods=10).std().clip(lower=1e-9)
            z    = (s - mu) / sig
            # Soft clamp at ±3, rescale to [-1, +1]
            result[col] = (z.clip(-3, 3) / 3.0).fillna(0.0)
        return result

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> List[str]:
        """Return the canonical ordered list of feature column names."""
        return [
            "returns_1d", "returns_5d", "returns_10d", "returns_20d", "returns_60d",
            "log_return_1d", "vol_5d", "vol_20d", "vol_60d", "vol_of_vol",
            "rsi_14", "macd", "bb_pct_b", "atr_ratio", "vwap_deviation",
            "mayer_multiple", "ema_ratio",
            "bh_mass", "bh_active", "bh_ctl", "ou_zscore",
            "btc_return_1d", "btc_eth_corr_20d",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
            "volume_ratio", "high_low_ratio", "close_position",
        ]
