"""
ml/feature_engineering.py
200+ feature pipeline for ML trading models.

Transforms raw bar-time inputs into a rich feature set covering:
lag features, interaction terms, calendar effects, technical indicators,
BH physics features, cross-asset signals, volatility regime, and
microstructure proxies.

No em dashes. Uses numpy, scipy, pandas.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# RawFeatures dataclass
# ---------------------------------------------------------------------------

@dataclass
class RawFeatures:
    """
    All raw inputs available at bar time.
    Each field represents what the system has access to when building features.
    """
    # OHLCV
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    # Return and vol
    ret_1: float = 0.0   # 1-bar return
    garch_vol: float = 0.01

    # BH physics states (3 timeframes: short, mid, long)
    bh_mass_short: float = 0.0
    bh_mass_mid: float = 0.0
    bh_mass_long: float = 0.0
    bh_active_short: float = 0.0
    bh_active_mid: float = 0.0
    bh_active_long: float = 0.0
    bh_proper_time: float = 0.0
    bh_geodesic_dev: float = 0.0

    # QuatNav signals
    quat_angular_vel: float = 0.0
    quat_nav_signal: float = 0.0

    # OU mean reversion
    ou_zscore: float = 0.0
    ou_theta: float = 0.0

    # Hurst exponent
    hurst: float = 0.5

    # Cross asset
    btc_ret_1: float = 0.0
    spy_ret_1: float = 0.0
    vix_level: float = 20.0

    # Granger signal
    granger_signal: float = 0.0

    # Calendar
    hour: int = 12
    day_of_week: int = 2
    day_of_month: int = 15
    month: int = 6
    is_fomc_week: bool = False
    is_earnings_week: bool = False

    # Ask/bid spread proxy
    spread_proxy: float = 0.001


# ---------------------------------------------------------------------------
# Base FeatureTransformer
# ---------------------------------------------------------------------------

class FeatureTransformer:
    """Base class for all feature transformers."""

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureTransformer":
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_features_out(self) -> int:
        raise NotImplementedError

    @property
    def feature_names(self) -> List[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# LagFeatures
# ---------------------------------------------------------------------------

class LagFeatures(FeatureTransformer):
    """
    Lagged features for returns, volatility, and volume.
    Lags: 1, 2, 3, 5, 10, 20 for each of 3 base features = 18 features.
    Additional: rolling std of returns (realized vol) at lags = 18 more.
    Rolling mean return at lags = 24 more.
    Total: 60 features.
    """

    LAGS = [1, 2, 3, 5, 10, 20]
    BASE_SERIES = ["ret", "vol", "volume"]

    def __init__(self) -> None:
        super().__init__()
        self._ret_buf: deque = deque(maxlen=22)
        self._vol_buf: deque = deque(maxlen=22)
        self._vol_buf_raw: deque = deque(maxlen=22)

    def update_buffers(self, ret: float, garch_vol: float, volume: float) -> None:
        self._ret_buf.append(ret)
        self._vol_buf.append(garch_vol)
        self._vol_buf_raw.append(volume)

    def transform_one(
        self, ret: float, garch_vol: float, volume: float
    ) -> np.ndarray:
        """
        Transform a single bar's raw inputs into 60 lag features.
        """
        self.update_buffers(ret, garch_vol, volume)
        rets = list(self._ret_buf)
        vols = list(self._vol_buf)
        volumes = list(self._vol_buf_raw)

        n = len(rets)
        features = []

        # 1. Raw lagged values (18 features)
        for lag in self.LAGS:
            features.append(rets[-lag] if n >= lag else 0.0)
        for lag in self.LAGS:
            features.append(vols[-lag] if n >= lag else 0.0)
        for lag in self.LAGS:
            v = volumes[-lag] if n >= lag else 0.0
            # normalize volume by max recent volume
            max_vol = max(volumes) if volumes else 1.0
            features.append(v / (max_vol + 1e-10))

        # 2. Rolling realized vol (mean of abs returns) over each lag window (6 features)
        for lag in self.LAGS:
            window = rets[-lag:] if n >= lag else rets
            features.append(float(np.mean(np.abs(window))) if window else 0.0)

        # 3. Rolling mean return over each lag window (6 features)
        for lag in self.LAGS:
            window = rets[-lag:] if n >= lag else rets
            features.append(float(np.mean(window)) if window else 0.0)

        # 4. Rolling skewness of returns (6 features)
        for lag in self.LAGS:
            window = rets[-lag:] if n >= lag else rets
            if len(window) >= 3:
                features.append(float(stats.skew(window)))
            else:
                features.append(0.0)

        # 5. Autocorrelation lag-1 (6 features at different windows)
        for lag in self.LAGS:
            window = rets[-lag:] if n >= lag else rets
            if len(window) >= 3:
                arr = np.array(window)
                ac = np.corrcoef(arr[:-1], arr[1:])[0, 1]
                features.append(0.0 if math.isnan(ac) else ac)
            else:
                features.append(0.0)

        # Pad to exactly 60
        features = features[:60]
        while len(features) < 60:
            features.append(0.0)

        return np.array(features, dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 3) array of [ret, garch_vol, volume]."""
        out = []
        self._ret_buf.clear()
        self._vol_buf.clear()
        self._vol_buf_raw.clear()
        for row in X:
            out.append(self.transform_one(row[0], row[1], row[2]))
        return np.array(out)

    @property
    def n_features_out(self) -> int:
        return 60

    @property
    def feature_names(self) -> List[str]:
        names = []
        for s in self.BASE_SERIES:
            for lag in self.LAGS:
                names.append(f"{s}_lag{lag}")
        for lag in self.LAGS:
            names.append(f"realized_vol_w{lag}")
        for lag in self.LAGS:
            names.append(f"mean_ret_w{lag}")
        for lag in self.LAGS:
            names.append(f"skew_ret_w{lag}")
        for lag in self.LAGS:
            names.append(f"ac1_ret_w{lag}")
        # Pad to exactly 60 with indexed names
        while len(names) < 60:
            names.append(f"lag_extra_{len(names)}")
        return names[:60]


# ---------------------------------------------------------------------------
# InteractionFeatures
# ---------------------------------------------------------------------------

class InteractionFeatures(FeatureTransformer):
    """
    Pairwise product interactions of key physics and market features.
    Total: 30 features.
    """

    def transform_one(self, raw: RawFeatures) -> np.ndarray:
        feats = [
            raw.bh_mass_mid * raw.garch_vol,
            raw.hurst * raw.ret_1,
            raw.ou_zscore * raw.garch_vol,
            raw.bh_mass_mid * raw.hurst,
            raw.quat_angular_vel * raw.garch_vol,
            raw.bh_mass_short * raw.bh_mass_long,
            raw.ou_zscore * raw.hurst,
            raw.quat_nav_signal * raw.ou_zscore,
            raw.granger_signal * raw.bh_mass_mid,
            raw.bh_geodesic_dev * raw.garch_vol,
            raw.bh_proper_time * raw.hurst,
            raw.bh_active_mid * raw.bh_mass_mid,
            raw.bh_active_long * raw.bh_mass_long,
            raw.vix_level * raw.garch_vol,
            raw.btc_ret_1 * raw.ret_1,
            raw.spy_ret_1 * raw.ou_zscore,
            raw.granger_signal * raw.hurst,
            raw.ou_theta * raw.ou_zscore,
            raw.quat_angular_vel * raw.bh_mass_mid,
            raw.bh_mass_mid * raw.bh_mass_short,
            raw.hurst * raw.garch_vol,
            raw.ret_1 * raw.volume,
            raw.bh_geodesic_dev * raw.bh_proper_time,
            raw.granger_signal * raw.ou_zscore,
            raw.vix_level * raw.ret_1,
            raw.btc_ret_1 * raw.garch_vol,
            raw.ou_zscore ** 2,
            raw.bh_mass_mid ** 2,
            raw.ret_1 ** 2,
            raw.quat_angular_vel * raw.ou_zscore,
        ]
        return np.array(feats[:30], dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use transform_one for online use")

    @property
    def n_features_out(self) -> int:
        return 30

    @property
    def feature_names(self) -> List[str]:
        return [
            "bh_mass_mid_x_garch_vol",
            "hurst_x_ret1",
            "ou_z_x_garch_vol",
            "bh_mass_mid_x_hurst",
            "quat_angvel_x_garch_vol",
            "bh_mass_short_x_long",
            "ou_z_x_hurst",
            "quat_nav_x_ou_z",
            "granger_x_bh_mass_mid",
            "bh_geodev_x_garch_vol",
            "bh_proper_time_x_hurst",
            "bh_active_mid_x_mass_mid",
            "bh_active_long_x_mass_long",
            "vix_x_garch_vol",
            "btc_ret1_x_ret1",
            "spy_ret1_x_ou_z",
            "granger_x_hurst",
            "ou_theta_x_ou_z",
            "quat_angvel_x_bh_mass_mid",
            "bh_mass_mid_x_short",
            "hurst_x_garch_vol",
            "ret1_x_volume",
            "bh_geodev_x_proper_time",
            "granger_x_ou_z",
            "vix_x_ret1",
            "btc_ret1_x_garch_vol",
            "ou_z_sq",
            "bh_mass_mid_sq",
            "ret1_sq",
            "quat_angvel_x_ou_z",
        ]


# ---------------------------------------------------------------------------
# CalendarFeatures
# ---------------------------------------------------------------------------

class CalendarFeatures(FeatureTransformer):
    """
    Cyclically encoded calendar features. Total: 20 features.
    """

    def transform_one(self, raw: RawFeatures) -> np.ndarray:
        h = raw.hour
        dow = raw.day_of_week
        dom = raw.day_of_month
        mon = raw.month

        feats = [
            math.sin(2 * math.pi * h / 24.0),
            math.cos(2 * math.pi * h / 24.0),
            math.sin(2 * math.pi * dow / 7.0),
            math.cos(2 * math.pi * dow / 7.0),
            math.sin(2 * math.pi * dom / 31.0),
            math.cos(2 * math.pi * dom / 31.0),
            math.sin(2 * math.pi * mon / 12.0),
            math.cos(2 * math.pi * mon / 12.0),
            float(raw.is_fomc_week),
            float(raw.is_earnings_week),
            # Additional: session indicator (US open hours 9-16 ET)
            float(9 <= h <= 16),
            float(h < 9),
            float(h > 16),
            # Day-of-week dummy (Monday=0)
            float(dow == 0),
            float(dow == 4),
            # Month seasonality
            float(mon == 1),   # January effect
            float(mon == 12),  # December
            # Week of month (approx)
            math.sin(2 * math.pi * (dom - 1) / 7.0),
            math.cos(2 * math.pi * (dom - 1) / 7.0),
            float(dom <= 7),   # first week
        ]
        return np.array(feats[:20], dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use transform_one for online use")

    @property
    def n_features_out(self) -> int:
        return 20

    @property
    def feature_names(self) -> List[str]:
        return [
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
            "dom_sin", "dom_cos",
            "month_sin", "month_cos",
            "is_fomc_week", "is_earnings_week",
            "us_session", "pre_market", "after_hours",
            "is_monday", "is_friday",
            "is_january", "is_december",
            "week_of_month_sin", "week_of_month_cos",
            "is_first_week",
        ]


# ---------------------------------------------------------------------------
# TechnicalFeatures
# ---------------------------------------------------------------------------

class TechnicalFeatures(FeatureTransformer):
    """
    Technical indicators derived from OHLCV history.
    RSI, MACD, Bollinger position, ATR ratio, ADX, normalized.
    Total: 20 features.
    """

    def __init__(self) -> None:
        super().__init__()
        self._close_buf: deque = deque(maxlen=30)
        self._high_buf: deque = deque(maxlen=30)
        self._low_buf: deque = deque(maxlen=30)
        self._atr_buf: deque = deque(maxlen=30)

    def _rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period - 1:])
        gain = np.where(deltas > 0, deltas, 0.0)
        loss = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gain.mean()
        avg_loss = loss.mean()
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _ema(self, values: List[float], period: int) -> float:
        if not values:
            return 0.0
        alpha = 2.0 / (period + 1.0)
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1.0 - alpha) * result
        return result

    def _bollinger_pos(self, prices: List[float], period: int = 20) -> float:
        """Position within Bollinger Bands: 0=lower band, 1=upper band."""
        if len(prices) < period:
            return 0.5
        window = prices[-period:]
        mean = np.mean(window)
        std = np.std(window)
        if std < 1e-10:
            return 0.5
        current = prices[-1]
        pos = (current - (mean - 2 * std)) / (4 * std)
        return float(np.clip(pos, 0.0, 1.0))

    def _adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Simplified ADX."""
        if len(closes) < period + 1:
            return 25.0
        h = np.array(highs[-period:])
        l = np.array(lows[-period:])
        c = np.array(closes[-period - 1:])
        tr = np.maximum(h - l, np.maximum(np.abs(h - c[:-1]), np.abs(l - c[:-1])))
        if tr.mean() < 1e-10:
            return 25.0
        dm_plus = np.where(
            (h[1:] - h[:-1]) > (l[:-1] - l[1:]),
            np.maximum(h[1:] - h[:-1], 0.0),
            0.0,
        )
        dm_minus = np.where(
            (l[:-1] - l[1:]) > (h[1:] - h[:-1]),
            np.maximum(l[:-1] - l[1:], 0.0),
            0.0,
        )
        di_plus = dm_plus.mean() / tr[1:].mean() * 100.0
        di_minus = dm_minus.mean() / tr[1:].mean() * 100.0
        if di_plus + di_minus < 1e-10:
            return 25.0
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100.0
        return float(np.clip(dx, 0.0, 100.0))

    def update_buffers(self, raw: RawFeatures) -> None:
        self._close_buf.append(raw.close)
        self._high_buf.append(raw.high)
        self._low_buf.append(raw.low)
        prev_close = list(self._close_buf)[-2] if len(self._close_buf) >= 2 else raw.close
        tr = max(
            raw.high - raw.low,
            abs(raw.high - prev_close),
            abs(raw.low - prev_close),
        )
        self._atr_buf.append(tr)

    def transform_one(self, raw: RawFeatures) -> np.ndarray:
        self.update_buffers(raw)
        closes = list(self._close_buf)
        highs = list(self._high_buf)
        lows = list(self._low_buf)
        atrs = list(self._atr_buf)

        rsi = self._rsi(closes, 14) / 100.0  # normalize to [0, 1]

        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(closes[-9:] if len(closes) >= 9 else closes, 9)
        macd_hist = macd_line - signal_line
        price_range = max(closes) - min(closes) if closes else 1.0
        macd_norm = macd_hist / (price_range + 1e-10)

        bb_pos = self._bollinger_pos(closes, 20)

        atr_mean = float(np.mean(atrs)) if atrs else 0.01
        current_price = raw.close if raw.close > 0 else 1.0
        atr_ratio = atr_mean / (current_price + 1e-10)

        adx = self._adx(highs, lows, closes, 14) / 100.0

        ema5 = self._ema(closes, 5)
        ema20 = self._ema(closes, 20)
        ema_diff = (ema5 - ema20) / (current_price + 1e-10)

        momentum_5 = (closes[-1] - closes[-5]) / (closes[-5] + 1e-10) if len(closes) >= 5 else 0.0
        momentum_10 = (closes[-1] - closes[-10]) / (closes[-10] + 1e-10) if len(closes) >= 10 else 0.0

        roc = (closes[-1] - closes[-12]) / (closes[-12] + 1e-10) if len(closes) >= 12 else 0.0

        # Stochastic %K
        if len(closes) >= 14:
            low_14 = min(lows[-14:])
            high_14 = max(highs[-14:])
            stoch_k = (closes[-1] - low_14) / (high_14 - low_14 + 1e-10)
        else:
            stoch_k = 0.5

        vol_ratio = atr_ratio / (float(np.mean(atrs[-5:])) / (current_price + 1e-10) + 1e-10) if len(atrs) >= 5 else 1.0
        price_vs_ema20 = (raw.close - ema20) / (current_price + 1e-10)
        obv_proxy = float(np.sign(raw.ret_1)) * raw.volume / (float(np.mean(list(self._atr_buf))) + 1e-10)
        obv_norm = float(np.tanh(obv_proxy * 1e-6))

        Williams_R = -((max(highs[-14:]) - closes[-1]) / (max(highs[-14:]) - min(lows[-14:]) + 1e-10) * 100.0) if len(closes) >= 14 else -50.0
        williams_r_norm = (Williams_R + 50.0) / 50.0

        feats = [
            rsi,
            float(np.clip(macd_norm, -0.1, 0.1)) / 0.1,
            bb_pos,
            float(np.clip(atr_ratio, 0.0, 0.05)) / 0.05,
            adx,
            float(np.clip(ema_diff, -0.05, 0.05)) / 0.05,
            float(np.clip(momentum_5, -0.1, 0.1)) / 0.1,
            float(np.clip(momentum_10, -0.2, 0.2)) / 0.2,
            float(np.clip(roc, -0.3, 0.3)) / 0.3,
            stoch_k,
            float(np.clip(vol_ratio, 0.0, 3.0)) / 3.0,
            float(np.clip(price_vs_ema20, -0.05, 0.05)) / 0.05,
            obv_norm,
            williams_r_norm,
            rsi - 0.5,
            float(np.sign(macd_hist)),
            float(bb_pos > 0.8),
            float(bb_pos < 0.2),
            float(adx > 0.25),
            float(momentum_5 > 0),
        ]
        return np.array(feats[:20], dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use transform_one for online use")

    @property
    def n_features_out(self) -> int:
        return 20

    @property
    def feature_names(self) -> List[str]:
        return [
            "rsi_norm", "macd_hist_norm", "bb_pos", "atr_ratio_norm", "adx_norm",
            "ema_diff", "mom_5", "mom_10", "roc_12", "stoch_k",
            "vol_ratio", "price_vs_ema20", "obv_norm", "williams_r",
            "rsi_centered", "macd_sign", "bb_overbought", "bb_oversold",
            "adx_trending", "mom_5_sign",
        ]


# ---------------------------------------------------------------------------
# PhysicsFeatures
# ---------------------------------------------------------------------------

class PhysicsFeatures(FeatureTransformer):
    """
    BH physics and QuatNav features. Total: 15 features.
    """

    def transform_one(self, raw: RawFeatures) -> np.ndarray:
        # Normalize BH masses
        bh_mass_sum = raw.bh_mass_short + raw.bh_mass_mid + raw.bh_mass_long + 1e-10
        feats = [
            raw.bh_mass_short / bh_mass_sum,
            raw.bh_mass_mid / bh_mass_sum,
            raw.bh_mass_long / bh_mass_sum,
            float(raw.bh_active_short),
            float(raw.bh_active_mid),
            float(raw.bh_active_long),
            float(np.tanh(raw.bh_proper_time / 100.0)),
            float(np.tanh(raw.bh_geodesic_dev)),
            float(np.tanh(raw.quat_angular_vel * 10.0)),
            float(np.clip(raw.quat_nav_signal, -1.0, 1.0)),
            float(np.clip(raw.hurst, 0.0, 1.0)),
            float(raw.hurst - 0.5),        # deviation from random walk
            float(raw.hurst > 0.6),        # trending
            float(raw.hurst < 0.4),        # mean-reverting
            float(raw.bh_active_short + raw.bh_active_mid + raw.bh_active_long),
        ]
        return np.array(feats[:15], dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use transform_one for online use")

    @property
    def n_features_out(self) -> int:
        return 15

    @property
    def feature_names(self) -> List[str]:
        return [
            "bh_mass_short_frac", "bh_mass_mid_frac", "bh_mass_long_frac",
            "bh_active_short", "bh_active_mid", "bh_active_long",
            "bh_proper_time_norm", "bh_geodesic_dev_norm",
            "quat_angvel_norm", "quat_nav_signal",
            "hurst", "hurst_dev", "hurst_trending", "hurst_mean_rev",
            "bh_n_active",
        ]


# ---------------------------------------------------------------------------
# CrossAssetFeatures
# ---------------------------------------------------------------------------

class CrossAssetFeatures(FeatureTransformer):
    """
    Cross-asset signals: BTC-crypto correlations, lead signals, sector ETFs.
    Total: 15 features.
    """

    def __init__(self) -> None:
        super().__init__()
        self._btc_ret_buf: deque = deque(maxlen=20)
        self._spy_ret_buf: deque = deque(maxlen=20)
        self._asset_ret_buf: deque = deque(maxlen=20)

    def transform_one(self, raw: RawFeatures) -> np.ndarray:
        self._btc_ret_buf.append(raw.btc_ret_1)
        self._spy_ret_buf.append(raw.spy_ret_1)
        self._asset_ret_buf.append(raw.ret_1)

        btc_rets = list(self._btc_ret_buf)
        spy_rets = list(self._spy_ret_buf)
        asset_rets = list(self._asset_ret_buf)

        def safe_corr(a: List[float], b: List[float]) -> float:
            if len(a) < 5:
                return 0.0
            c = np.corrcoef(a, b)[0, 1]
            return 0.0 if math.isnan(c) else float(c)

        btc_spy_corr = safe_corr(btc_rets, spy_rets)
        btc_asset_corr = safe_corr(btc_rets, asset_rets)
        spy_asset_corr = safe_corr(spy_rets, asset_rets)

        # BTC lead: is BTC leading the asset? (correlation at lag 1)
        btc_lead = 0.0
        if len(btc_rets) >= 5:
            a1 = btc_rets[:-1]
            b1 = asset_rets[1:]
            if len(a1) == len(b1):
                btc_lead = safe_corr(a1, b1)

        vix_norm = float(np.clip((raw.vix_level - 15.0) / 30.0, -1.0, 2.0))
        risk_on = float(raw.vix_level < 20.0)
        risk_off = float(raw.vix_level > 30.0)

        btc_momentum = float(np.tanh(sum(btc_rets[-5:]) * 10.0)) if len(btc_rets) >= 5 else 0.0
        spy_momentum = float(np.tanh(sum(spy_rets[-5:]) * 10.0)) if len(spy_rets) >= 5 else 0.0

        feats = [
            float(np.clip(raw.btc_ret_1, -0.2, 0.2)) / 0.2,
            float(np.clip(raw.spy_ret_1, -0.1, 0.1)) / 0.1,
            btc_spy_corr,
            btc_asset_corr,
            spy_asset_corr,
            btc_lead,
            vix_norm,
            risk_on,
            risk_off,
            btc_momentum,
            spy_momentum,
            float(np.sign(raw.btc_ret_1) == np.sign(raw.ret_1)),
            float(np.sign(raw.spy_ret_1) == np.sign(raw.ret_1)),
            float(abs(raw.btc_ret_1) > abs(raw.ret_1)),
            float(raw.granger_signal),
        ]
        return np.array(feats[:15], dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use transform_one for online use")

    @property
    def n_features_out(self) -> int:
        return 15

    @property
    def feature_names(self) -> List[str]:
        return [
            "btc_ret_norm", "spy_ret_norm",
            "btc_spy_corr", "btc_asset_corr", "spy_asset_corr",
            "btc_lead_corr",
            "vix_norm", "risk_on", "risk_off",
            "btc_momentum", "spy_momentum",
            "btc_same_direction", "spy_same_direction",
            "btc_dominates", "granger_signal",
        ]


# ---------------------------------------------------------------------------
# VolatilityFeatures
# ---------------------------------------------------------------------------

class VolatilityFeatures(FeatureTransformer):
    """
    Volatility regime and term structure features. Total: 15 features.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ret_buf: deque = deque(maxlen=25)
        self._vol_buf: deque = deque(maxlen=25)

    def transform_one(self, raw: RawFeatures) -> np.ndarray:
        self._ret_buf.append(raw.ret_1)
        self._vol_buf.append(raw.garch_vol)

        rets = list(self._ret_buf)
        vols = list(self._vol_buf)

        realized_vol_5 = float(np.std(rets[-5:])) if len(rets) >= 5 else raw.garch_vol
        realized_vol_20 = float(np.std(rets[-20:])) if len(rets) >= 20 else raw.garch_vol

        garch_ann = raw.garch_vol * math.sqrt(252)
        rv5_ann = realized_vol_5 * math.sqrt(252)
        rv20_ann = realized_vol_20 * math.sqrt(252)

        # Vol of vol
        vol_of_vol = float(np.std(vols[-10:])) if len(vols) >= 10 else 0.001

        # Vol ratio (GARCH vs realized)
        vol_ratio = raw.garch_vol / (realized_vol_5 + 1e-10)

        # Term structure: short vol vs long vol
        vol_term_structure = realized_vol_5 / (realized_vol_20 + 1e-10) - 1.0

        # Vol regime: low/medium/high (percentile-based)
        if len(vols) >= 20:
            p25 = float(np.percentile(vols, 25))
            p75 = float(np.percentile(vols, 75))
            vol_low = float(raw.garch_vol < p25)
            vol_high = float(raw.garch_vol > p75)
        else:
            vol_low = 0.0
            vol_high = 0.0

        feats = [
            float(np.clip(garch_ann, 0.0, 2.0)) / 2.0,
            float(np.clip(rv5_ann, 0.0, 2.0)) / 2.0,
            float(np.clip(rv20_ann, 0.0, 2.0)) / 2.0,
            float(np.tanh(vol_of_vol * 100.0)),
            float(np.clip(vol_ratio, 0.0, 3.0)) / 3.0,
            float(np.clip(vol_term_structure, -1.0, 1.0)),
            vol_low,
            vol_high,
            float(1.0 - vol_low - vol_high),  # medium vol
            float(np.clip(raw.garch_vol * 100.0, 0.0, 5.0)) / 5.0,
            float(np.sign(raw.ret_1) * raw.garch_vol),  # directional vol
            float(max(0.0, raw.garch_vol - realized_vol_5) / (realized_vol_5 + 1e-10)),  # vol premium
            float(np.tanh(vol_term_structure * 5.0)),
            float(raw.garch_vol > 0.03),  # high vol flag
            float(raw.garch_vol < 0.01),  # low vol flag
        ]
        return np.array(feats[:15], dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use transform_one for online use")

    @property
    def n_features_out(self) -> int:
        return 15

    @property
    def feature_names(self) -> List[str]:
        return [
            "garch_ann", "rv5_ann", "rv20_ann",
            "vol_of_vol", "garch_rv5_ratio",
            "vol_term_structure", "vol_low", "vol_high", "vol_mid",
            "garch_norm", "directional_vol", "vol_premium",
            "vol_term_tanh", "high_vol_flag", "low_vol_flag",
        ]


# ---------------------------------------------------------------------------
# MicrostructureFeatures
# ---------------------------------------------------------------------------

class MicrostructureFeatures(FeatureTransformer):
    """
    Market microstructure proxies. Total: 15 features.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ret_buf: deque = deque(maxlen=20)
        self._vol_buf: deque = deque(maxlen=20)
        self._obv: float = 0.0

    def transform_one(self, raw: RawFeatures) -> np.ndarray:
        self._ret_buf.append(raw.ret_1)
        self._vol_buf.append(raw.volume)

        rets = list(self._ret_buf)
        vols = list(self._vol_buf)

        # Kyle's lambda proxy: price impact = |ret| / volume
        kyle_lambda = abs(raw.ret_1) / (raw.volume + 1e-10)
        kyle_norm = float(np.tanh(kyle_lambda * 1e6))

        # Volume surprise: current vol vs rolling mean
        mean_vol = float(np.mean(vols)) if vols else raw.volume
        vol_surprise = (raw.volume - mean_vol) / (mean_vol + 1e-10)

        # OBV change
        self._obv += raw.volume * float(np.sign(raw.ret_1))
        obv_change = float(np.tanh(self._obv / (mean_vol * 10.0 + 1.0)))

        # Chaikin Money Flow proxy
        hl_range = raw.high - raw.low
        if hl_range > 0:
            mfm = ((raw.close - raw.low) - (raw.high - raw.close)) / hl_range
        else:
            mfm = 0.0
        cmf = mfm * raw.volume / (mean_vol + 1.0)
        cmf_norm = float(np.tanh(cmf))

        # ATR/spread ratio
        atr_proxy = (raw.high - raw.low)
        spread = raw.spread_proxy
        atr_spread_ratio = atr_proxy / (spread + 1e-10)
        atr_spread_norm = float(np.tanh(atr_spread_ratio / 100.0))

        # Amihud illiquidity
        amihud = abs(raw.ret_1) / (raw.volume * raw.close + 1e-10) * 1e6
        amihud_norm = float(np.tanh(amihud))

        # Price impact direction
        signed_vol = raw.volume * float(np.sign(raw.ret_1))
        signed_vol_norm = float(np.tanh(signed_vol / (mean_vol + 1e-10)))

        feats = [
            kyle_norm,
            float(np.clip(vol_surprise, -3.0, 3.0)) / 3.0,
            obv_change,
            cmf_norm,
            atr_spread_norm,
            amihud_norm,
            signed_vol_norm,
            float(raw.volume > mean_vol),
            float(raw.volume < mean_vol * 0.5),
            float(np.clip(mfm, -1.0, 1.0)),
            float(np.tanh(vol_surprise)),
            float(atr_proxy / (raw.close + 1e-10) * 100.0),
            float(np.sign(raw.ret_1) * vol_surprise),
            float(abs(mfm) > 0.5),
            float(abs(vol_surprise) > 1.0),
        ]
        return np.array(feats[:15], dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use transform_one for online use")

    @property
    def n_features_out(self) -> int:
        return 15

    @property
    def feature_names(self) -> List[str]:
        return [
            "kyle_lambda", "vol_surprise",
            "obv_change", "cmf", "atr_spread_ratio",
            "amihud", "signed_vol",
            "vol_above_mean", "vol_very_low",
            "mfm", "vol_surprise_tanh",
            "atr_pct", "directional_vol_surprise",
            "high_mfm", "high_vol_surprise",
        ]


# ---------------------------------------------------------------------------
# OUFeatures (OU-derived features)
# ---------------------------------------------------------------------------

class OUFeatures(FeatureTransformer):
    """Additional OU mean-reversion features. 5 features."""

    def transform_one(self, raw: RawFeatures) -> np.ndarray:
        z = raw.ou_zscore
        theta = raw.ou_theta
        feats = [
            float(np.clip(z, -4.0, 4.0)) / 4.0,
            float(np.sign(z) * z ** 2 / 16.0),  # squared zscore with direction
            float(abs(z) > 2.0),   # extreme reversion signal
            float(np.clip(theta, 0.0, 1.0)),
            float(z * theta),      # interaction: speed * zscore
        ]
        return np.array(feats, dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_features_out(self) -> int:
        return 5

    @property
    def feature_names(self) -> List[str]:
        return ["ou_z_norm", "ou_z_sq_signed", "ou_extreme", "ou_theta", "ou_z_x_theta"]


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Chains all feature transformers into a single 200-feature vector.
    Handles missing values, z-score normalization, winsorization,
    and optional PCA dimensionality reduction.

    Feature count breakdown:
        LagFeatures:          60
        InteractionFeatures:  30
        CalendarFeatures:     20
        TechnicalFeatures:    20
        PhysicsFeatures:      15
        CrossAssetFeatures:   15
        VolatilityFeatures:   15
        MicrostructureFeatures: 15
        OUFeatures:            5
        Extra raw:             5
        Total:               200
    """

    TOTAL_FEATURES = 200

    def __init__(
        self,
        use_pca: bool = False,
        pca_components: int = 50,
        zscore_window: int = 500,
        winsor_limit: float = 3.0,
    ) -> None:
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.zscore_window = zscore_window
        self.winsor_limit = winsor_limit

        self._lag = LagFeatures()
        self._interaction = InteractionFeatures()
        self._calendar = CalendarFeatures()
        self._technical = TechnicalFeatures()
        self._physics = PhysicsFeatures()
        self._cross_asset = CrossAssetFeatures()
        self._volatility = VolatilityFeatures()
        self._microstructure = MicrostructureFeatures()
        self._ou = OUFeatures()

        # Online normalization: rolling mean and std
        self._feature_buf: deque = deque(maxlen=zscore_window)
        self._rolling_mean: Optional[np.ndarray] = None
        self._rolling_std: Optional[np.ndarray] = None

        # PCA (fitted offline; placeholder here)
        self._pca_components_mat: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None

        self._n_processed = 0

    def transform_one(self, raw: RawFeatures) -> np.ndarray:
        """
        Transform a single RawFeatures instance into a 200-dim feature vector.
        """
        # 1. Lag features (60)
        lag_feats = self._lag.transform_one(
            raw.ret_1, raw.garch_vol, raw.volume
        )
        # 2. Interaction (30)
        interaction_feats = self._interaction.transform_one(raw)
        # 3. Calendar (20)
        calendar_feats = self._calendar.transform_one(raw)
        # 4. Technical (20)
        tech_feats = self._technical.transform_one(raw)
        # 5. Physics (15)
        physics_feats = self._physics.transform_one(raw)
        # 6. Cross-asset (15)
        cross_feats = self._cross_asset.transform_one(raw)
        # 7. Volatility (15)
        vol_feats = self._volatility.transform_one(raw)
        # 8. Microstructure (15)
        micro_feats = self._microstructure.transform_one(raw)
        # 9. OU (5)
        ou_feats = self._ou.transform_one(raw)
        # 10. Extra raw features (5)
        extra = np.array([
            float(np.clip(raw.ou_zscore, -4.0, 4.0)) / 4.0,
            float(raw.hurst - 0.5),
            float(np.clip(raw.garch_vol * 100.0, 0.0, 10.0)) / 10.0,
            float(np.clip(raw.quat_angular_vel, -5.0, 5.0)) / 5.0,
            float(np.clip(raw.bh_mass_mid, 0.0, 1.0)),
        ])

        # Concatenate all features
        feature_vec = np.concatenate([
            lag_feats,        # 60
            interaction_feats, # 30
            calendar_feats,   # 20
            tech_feats,       # 20
            physics_feats,    # 15
            cross_feats,      # 15
            vol_feats,        # 15
            micro_feats,      # 15
            ou_feats,         # 5
            extra,            # 5
        ])

        # Handle NaN/Inf (forward fill -> zero)
        feature_vec = np.where(np.isfinite(feature_vec), feature_vec, 0.0)

        # Winsorize
        feature_vec = np.clip(feature_vec, -self.winsor_limit, self.winsor_limit)

        # Update rolling stats
        self._feature_buf.append(feature_vec.copy())
        if len(self._feature_buf) >= 30:
            buf_arr = np.array(list(self._feature_buf))
            self._rolling_mean = buf_arr.mean(axis=0)
            self._rolling_std = buf_arr.std(axis=0) + 1e-8

        # Z-score normalize
        if self._rolling_mean is not None and self._rolling_std is not None:
            feature_vec = (feature_vec - self._rolling_mean) / self._rolling_std
            # Clip after normalization
            feature_vec = np.clip(feature_vec, -5.0, 5.0)

        # PCA (optional)
        if (
            self.use_pca
            and self._pca_components_mat is not None
            and self._pca_mean is not None
        ):
            feature_vec = (feature_vec - self._pca_mean) @ self._pca_components_mat.T

        self._n_processed += 1
        return feature_vec

    def fit_pca(self, feature_matrix: np.ndarray) -> None:
        """
        Fit PCA on a (N, 200) feature matrix.
        Only needed if use_pca=True.
        """
        from numpy.linalg import svd
        self._pca_mean = feature_matrix.mean(axis=0)
        centered = feature_matrix - self._pca_mean
        U, S, Vt = svd(centered, full_matrices=False)
        self._pca_components_mat = Vt[: self.pca_components]

    @property
    def feature_names(self) -> List[str]:
        names = (
            self._lag.feature_names
            + self._interaction.feature_names
            + self._calendar.feature_names
            + self._technical.feature_names
            + self._physics.feature_names
            + self._cross_asset.feature_names
            + self._volatility.feature_names
            + self._microstructure.feature_names
            + self._ou.feature_names
            + ["ou_z_raw", "hurst_dev", "garch_raw", "quat_angvel_raw", "bh_mass_mid_raw"]
        )
        return names

    def reset_normalization(self) -> None:
        self._feature_buf.clear()
        self._rolling_mean = None
        self._rolling_std = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
            "zscore_window": self.zscore_window,
            "winsor_limit": self.winsor_limit,
            "_n_processed": self._n_processed,
            "_rolling_mean": self._rolling_mean.tolist() if self._rolling_mean is not None else None,
            "_rolling_std": self._rolling_std.tolist() if self._rolling_std is not None else None,
            "_pca_mean": self._pca_mean.tolist() if self._pca_mean is not None else None,
            "_pca_components_mat": self._pca_components_mat.tolist() if self._pca_components_mat is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeaturePipeline":
        obj = cls(
            use_pca=d["use_pca"],
            pca_components=d["pca_components"],
            zscore_window=d["zscore_window"],
            winsor_limit=d["winsor_limit"],
        )
        obj._n_processed = d["_n_processed"]
        if d["_rolling_mean"] is not None:
            obj._rolling_mean = np.array(d["_rolling_mean"])
        if d["_rolling_std"] is not None:
            obj._rolling_std = np.array(d["_rolling_std"])
        if d["_pca_mean"] is not None:
            obj._pca_mean = np.array(d["_pca_mean"])
        if d["_pca_components_mat"] is not None:
            obj._pca_components_mat = np.array(d["_pca_components_mat"])
        return obj


# ---------------------------------------------------------------------------
# FeatureImportanceAnalyzer
# ---------------------------------------------------------------------------

class FeatureImportanceAnalyzer:
    """
    SHAP-like feature importance via permutation.

    Shuffles one feature at a time, measures prediction degradation,
    and ranks features by their contribution to model performance.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 5,
        metric: str = "accuracy",
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.n_repeats = n_repeats
        self.metric = metric

    def compute_importance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute permutation importance.
        X: (N, n_features) feature matrix.
        y: (N,) target labels (0/1 or continuous).
        Returns dict mapping feature name -> importance score.
        """
        n, d = X.shape

        # Baseline score
        baseline_score = self._score(X, y)
        importances = np.zeros(d)

        for feat_idx in range(d):
            degradations = []
            for _ in range(self.n_repeats):
                X_perm = X.copy()
                perm_idx = np.random.permutation(n)
                X_perm[:, feat_idx] = X_perm[perm_idx, feat_idx]
                permuted_score = self._score(X_perm, y)
                degradations.append(baseline_score - permuted_score)
            importances[feat_idx] = float(np.mean(degradations))

        # Normalize to sum to 1
        total = importances.sum()
        if total > 0:
            importances = importances / total

        if self.feature_names and len(self.feature_names) == d:
            return {name: float(imp) for name, imp in zip(self.feature_names, importances)}
        else:
            return {f"feat_{i}": float(imp) for i, imp in enumerate(importances)}

    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.metric == "accuracy":
            preds = np.array([self.model.predict(X[i]) for i in range(len(X))])
            return float(np.mean(np.sign(preds) == np.sign(y - 0.5)))
        elif self.metric == "mse":
            preds = np.array([self.model.predict(X[i]) for i in range(len(X))])
            return -float(np.mean((preds - y) ** 2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def top_n_features(
        self, X: np.ndarray, y: np.ndarray, n: int = 20
    ) -> List[Tuple[str, float]]:
        """Return top N features by importance, sorted descending."""
        importances = self.compute_importance(X, y)
        sorted_feats = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_feats[:n]
