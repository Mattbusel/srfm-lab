"""
regime-oracle/feature_builder.py
──────────────────────────────────
Feature engineering for regime classification.

The RegimeFeatureBuilder transforms raw OHLCV data (plus optional BH-mass
and breadth signals) into a normalised feature dict suitable for the
RegimeOracle classifier.

Features computed
-----------------
  ema_ratio_50_200  : EMA(50) / EMA(200) — trend state
  ema_ratio_20_50   : EMA(20) / EMA(50)  — short-term trend
  vol_percentile_30d: rolling vol percentile over 30-day window
  vol_percentile_90d: rolling vol percentile over 90-day window
  momentum_5d       : 5-bar return
  momentum_20d      : 20-bar return
  momentum_60d      : 60-bar return
  breadth_50d       : fraction of assets above 50-day EMA (if multi-asset)
  drawdown_from_ath : current drawdown from all-time-high
  bh_mass_mean      : mean BH mass across instruments (if available)
  bh_mass_max       : max BH mass (if available)
  atr_percentile    : ATR as rolling percentile
  skewness_20d      : 20-bar return skewness
  kurtosis_20d      : 20-bar return kurtosis

Usage
-----
    builder = RegimeFeatureBuilder()
    features = builder.build_features(ohlcv_df, symbol="BTC")
    normed   = builder.normalize_features(features)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

logger = logging.getLogger(__name__)

# Minimum bars needed before emitting features (warmup)
_WARMUP_BARS = 210   # 200-bar EMA needs ~200 bars minimum


class RegimeFeatureBuilder:
    """
    Builds regime classification features from OHLCV data.

    Parameters
    ----------
    ema_short       : short EMA period (default 20)
    ema_mid         : mid EMA period (default 50)
    ema_long        : long EMA period (default 200)
    vol_short_days  : short vol window in bars (default 30)
    vol_long_days   : long vol window in bars (default 90)
    atr_period      : ATR period (default 14)
    momentum_windows: list of momentum look-back periods (default [5, 20, 60])
    skew_window     : window for skewness/kurtosis (default 20)
    """

    def __init__(
        self,
        ema_short:        int        = 20,
        ema_mid:          int        = 50,
        ema_long:         int        = 200,
        vol_short_days:   int        = 30,
        vol_long_days:    int        = 90,
        atr_period:       int        = 14,
        momentum_windows: List[int]  = None,
        skew_window:      int        = 20,
    ) -> None:
        self.ema_short       = ema_short
        self.ema_mid         = ema_mid
        self.ema_long        = ema_long
        self.vol_short_days  = vol_short_days
        self.vol_long_days   = vol_long_days
        self.atr_period      = atr_period
        self.momentum_windows = momentum_windows or [5, 20, 60]
        self.skew_window     = skew_window

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def build_features(
        self,
        ohlcv_df: pd.DataFrame,
        symbol:   str = "BTC",
        bh_mass_series: Optional[pd.Series] = None,
        breadth_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Build a feature dict from OHLCV data.

        Parameters
        ----------
        ohlcv_df       : DataFrame with columns [open, high, low, close, volume].
                         Must have a DatetimeIndex (or similar index).
                         Column names are case-insensitive.
        symbol         : instrument symbol (metadata only)
        bh_mass_series : optional Series of BH mass values (same index as ohlcv_df)
        breadth_series : optional Series of breadth values (fraction above 50EMA)

        Returns
        -------
        dict mapping feature name → float value (or NaN if insufficient data)
        """
        df = self._normalise_columns(ohlcv_df)

        n = len(df)
        if n < _WARMUP_BARS:
            logger.warning(
                "Only %d bars available — some features will be NaN (need %d for warmup).",
                n, _WARMUP_BARS,
            )

        close  = df["close"]
        high   = df["high"]   if "high"   in df.columns else close
        low    = df["low"]    if "low"    in df.columns else close

        features: Dict[str, Any] = {"symbol": symbol, "n_bars": n}

        # ── EMA ratios ─────────────────────────────────────────────────
        ema20  = close.ewm(span=self.ema_short, adjust=False).mean()
        ema50  = close.ewm(span=self.ema_mid,   adjust=False).mean()
        ema200 = close.ewm(span=self.ema_long,  adjust=False).mean()

        features["ema_ratio_50_200"] = self._last(ema50  / ema200.replace(0, np.nan))
        features["ema_ratio_20_50"]  = self._last(ema20  / ema50.replace(0, np.nan))

        # ── Volatility ─────────────────────────────────────────────────
        returns = close.pct_change()
        vol_daily = returns.rolling(window=24, min_periods=12).std()   # hourly bars

        short_w = self.vol_short_days * 24   # convert days → hourly bars
        long_w  = self.vol_long_days  * 24

        features["vol_percentile_30d"] = self._rolling_percentile(vol_daily, short_w)
        features["vol_percentile_90d"] = self._rolling_percentile(vol_daily, long_w)

        # ── Momentum ───────────────────────────────────────────────────
        for w in self.momentum_windows:
            bars = w * 24   # days → hourly bars
            if n > bars:
                mom = float(close.iloc[-1] / close.iloc[-(bars + 1)] - 1.0)
            else:
                mom = float("nan")
            features[f"momentum_{w}d"] = mom

        # ── ATR percentile ─────────────────────────────────────────────
        atr = self._compute_atr(high, low, close, self.atr_period * 24)
        features["atr_percentile"] = self._rolling_percentile(atr, short_w)

        # ── Drawdown from ATH ──────────────────────────────────────────
        ath = close.cummax()
        dd  = (close - ath) / ath.replace(0, np.nan)
        features["drawdown_from_ath"] = self._last(dd)

        # ── Skewness / Kurtosis ────────────────────────────────────────
        sk_w = self.skew_window * 24
        recent_returns = returns.dropna().iloc[-sk_w:] if n > sk_w else returns.dropna()
        if len(recent_returns) >= 4:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features["skewness_20d"] = float(scipy_skew(recent_returns))
                features["kurtosis_20d"] = float(scipy_kurtosis(recent_returns, fisher=True))
        else:
            features["skewness_20d"] = float("nan")
            features["kurtosis_20d"] = float("nan")

        # ── Breadth ────────────────────────────────────────────────────
        if breadth_series is not None and len(breadth_series) > 0:
            features["breadth_50d"] = self._last(breadth_series)
        else:
            # Estimate breadth from single instrument (proxy: close vs ema50)
            above_ema50 = float((close > ema50).rolling(50 * 24, min_periods=10).mean().iloc[-1]) \
                if n > 0 else float("nan")
            features["breadth_50d"] = above_ema50

        # ── BH mass ────────────────────────────────────────────────────
        if bh_mass_series is not None and len(bh_mass_series) > 0:
            features["bh_mass_mean"] = float(bh_mass_series.mean())
            features["bh_mass_max"]  = float(bh_mass_series.max())
        else:
            features["bh_mass_mean"] = float("nan")
            features["bh_mass_max"]  = float("nan")

        return features

    # ------------------------------------------------------------------
    # Rolling feature computation
    # ------------------------------------------------------------------

    def build_features_rolling(
        self,
        ohlcv_df: pd.DataFrame,
        symbol:   str = "BTC",
        bh_mass_series: Optional[pd.Series] = None,
        breadth_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute features for every bar in the DataFrame (rolling computation).

        Used for back-labelling historical regimes.

        Parameters
        ----------
        ohlcv_df        : full OHLCV history
        symbol          : instrument symbol (metadata only)
        bh_mass_series  : optional BH mass series aligned with ohlcv_df
        breadth_series  : optional breadth series

        Returns
        -------
        pd.DataFrame of features, one row per bar (NaN during warmup)
        """
        df = self._normalise_columns(ohlcv_df)
        n  = len(df)
        close  = df["close"]
        high   = df["high"]  if "high"  in df.columns else close
        low    = df["low"]   if "low"   in df.columns else close

        out = pd.DataFrame(index=df.index)

        # EMA ratios
        ema20  = close.ewm(span=self.ema_short, adjust=False).mean()
        ema50  = close.ewm(span=self.ema_mid,   adjust=False).mean()
        ema200 = close.ewm(span=self.ema_long,  adjust=False).mean()
        out["ema_ratio_50_200"] = (ema50  / ema200.replace(0, np.nan)).values
        out["ema_ratio_20_50"]  = (ema20  / ema50.replace(0,  np.nan)).values

        # Volatility
        returns  = close.pct_change()
        vol_d    = returns.rolling(window=24, min_periods=6).std()
        short_w  = self.vol_short_days * 24
        long_w   = self.vol_long_days  * 24
        out["vol_percentile_30d"] = self._rolling_percentile_series(vol_d, short_w)
        out["vol_percentile_90d"] = self._rolling_percentile_series(vol_d, long_w)

        # Momentum
        for w in self.momentum_windows:
            bars = w * 24
            out[f"momentum_{w}d"] = close.pct_change(periods=bars)

        # ATR
        atr = self._compute_atr(high, low, close, self.atr_period * 24)
        out["atr_percentile"] = self._rolling_percentile_series(atr, short_w)

        # Drawdown from ATH
        ath = close.cummax()
        out["drawdown_from_ath"] = (close - ath) / ath.replace(0, np.nan)

        # Skewness / Kurtosis (rolling)
        sk_w = self.skew_window * 24
        out["skewness_20d"] = returns.rolling(sk_w, min_periods=20).apply(
            lambda x: float(scipy_skew(x)) if len(x) >= 4 else float("nan"),
            raw=True,
        )
        out["kurtosis_20d"] = returns.rolling(sk_w, min_periods=20).apply(
            lambda x: float(scipy_kurtosis(x, fisher=True)) if len(x) >= 4 else float("nan"),
            raw=True,
        )

        # Breadth
        if breadth_series is not None:
            out["breadth_50d"] = breadth_series.reindex(df.index).values
        else:
            out["breadth_50d"] = (close > ema50).rolling(50 * 24, min_periods=10).mean().values

        # BH mass
        if bh_mass_series is not None:
            out["bh_mass_mean"] = bh_mass_series.rolling(24, min_periods=1).mean().reindex(df.index).values
            out["bh_mass_max"]  = bh_mass_series.rolling(24, min_periods=1).max().reindex(df.index).values
        else:
            out["bh_mass_mean"] = float("nan")
            out["bh_mass_max"]  = float("nan")

        return out

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize_features(
        self,
        features: Dict[str, Any],
        clip_std: float = 3.0,
    ) -> Dict[str, float]:
        """
        Normalise a feature dict so numeric features are in [0, 1].

        Strategy per feature:
          ema_ratio_*       → clipped to [0.5, 1.5] then min-max scaled
          vol_percentile_*  → already in [0, 1]
          momentum_*        → sigmoid transform
          breadth_50d       → already in [0, 1]
          drawdown_from_ath → negated and clipped to [0, 1] (0 = no dd, 1 = full)
          bh_mass_*         → clipped to [0, 4] then / 4
          atr_percentile    → already in [0, 1]
          skewness_20d      → sigmoid
          kurtosis_20d      → tanh scaled

        Parameters
        ----------
        features : raw feature dict from build_features
        clip_std : number of stds for sigmoid clipping (not currently used)

        Returns
        -------
        dict mapping feature name → float in [0, 1]
        """
        normed: Dict[str, float] = {}

        def _safe(v: Any, default: float = 0.5) -> float:
            try:
                f = float(v)
                return default if (f != f) else f   # NaN → default
            except (TypeError, ValueError):
                return default

        # EMA ratios: ratio near 1.0 = trend neutral; >1 = bullish; <1 = bearish
        # Scale [0.5, 2.0] → [0, 1]
        for k in ("ema_ratio_50_200", "ema_ratio_20_50"):
            v = _safe(features.get(k), 1.0)
            normed[k] = float(np.clip((v - 0.5) / 1.5, 0.0, 1.0))

        # Vol percentiles already in [0, 1]
        for k in ("vol_percentile_30d", "vol_percentile_90d", "atr_percentile"):
            normed[k] = float(np.clip(_safe(features.get(k), 0.5), 0.0, 1.0))

        # Momentum: sigmoid transform centred at 0
        for w in self.momentum_windows:
            key = f"momentum_{w}d"
            v = _safe(features.get(key), 0.0)
            normed[key] = float(1.0 / (1.0 + np.exp(-v * 10)))   # sigmoid(10x)

        # Breadth: [0, 1] already
        normed["breadth_50d"] = float(np.clip(_safe(features.get("breadth_50d"), 0.5), 0.0, 1.0))

        # Drawdown: negate (dd is negative fraction); clamp to [0, 1]
        dd = _safe(features.get("drawdown_from_ath"), 0.0)
        normed["drawdown_from_ath"] = float(np.clip(-dd, 0.0, 1.0))

        # BH mass: scale to [0, 1] with max ≈ 4
        for k in ("bh_mass_mean", "bh_mass_max"):
            v = _safe(features.get(k), 1.0)
            normed[k] = float(np.clip(v / 4.0, 0.0, 1.0))

        # Skewness: sigmoid(-skew * 0.5) so negative skew → higher (bearish signal)
        sk = _safe(features.get("skewness_20d"), 0.0)
        normed["skewness_20d"] = float(1.0 / (1.0 + np.exp(sk * 0.5)))

        # Kurtosis: tanh scaled; high kurtosis → fat tails → crisis risk
        kurt = _safe(features.get("kurtosis_20d"), 0.0)
        normed["kurtosis_20d"] = float(np.clip(np.tanh(kurt / 4.0) * 0.5 + 0.5, 0.0, 1.0))

        return normed

    def normalize_features_df(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalisation to a feature DataFrame row-by-row.

        Parameters
        ----------
        feature_df : DataFrame of raw features from build_features_rolling

        Returns
        -------
        pd.DataFrame of normalised features
        """
        normed_rows: List[Dict[str, float]] = []
        for _, row in feature_df.iterrows():
            normed_rows.append(self.normalize_features(row.to_dict()))
        return pd.DataFrame(normed_rows, index=feature_df.index)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names to lowercase."""
        rename = {c: c.lower() for c in df.columns}
        return df.rename(columns=rename)

    @staticmethod
    def _last(series: pd.Series, default: float = float("nan")) -> float:
        """Return the last non-NaN value of a series, or default."""
        s = series.dropna()
        if len(s) == 0:
            return default
        return float(s.iloc[-1])

    @staticmethod
    def _compute_atr(
        high:   pd.Series,
        low:    pd.Series,
        close:  pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute Average True Range (ATR).

        Parameters
        ----------
        high, low, close : OHLCV columns
        period           : rolling window size (bars)
        """
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _rolling_percentile(
        series: pd.Series,
        window: int,
    ) -> float:
        """
        Compute the current value's percentile within a rolling window.

        Returns float in [0, 1].
        """
        s = series.dropna()
        if len(s) < 2:
            return 0.5
        current = float(s.iloc[-1])
        window_data = s.iloc[-window:] if len(s) >= window else s
        pct = float((window_data < current).sum() / len(window_data))
        return pct

    @staticmethod
    def _rolling_percentile_series(
        series: pd.Series,
        window: int,
    ) -> pd.Series:
        """
        Compute rolling percentile rank for every bar.

        Returns pd.Series in [0, 1].
        """
        def _rank(x: np.ndarray) -> float:
            if len(x) < 2 or np.isnan(x[-1]):
                return float("nan")
            return float((x[:-1] < x[-1]).sum() / (len(x) - 1))

        return series.rolling(window=window, min_periods=max(10, window // 10)).apply(
            _rank, raw=True
        )
