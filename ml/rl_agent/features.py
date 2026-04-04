"""
Feature engineering for the RL trading agent.

Covers:
  - Normalized return features
  - Buy-and-Hold (BH) state encoding
  - Order book proxy features
  - Macro/cross-asset features
  - Rolling statistical moments
  - Regime indicators
  - Calendar / time features
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Controls which feature groups are computed."""
    returns_windows: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 60])
    vol_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 28])
    bb_periods: List[int] = field(default_factory=lambda: [10, 20])
    macd_params: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [(12, 26, 9), (5, 13, 4)]
    )
    atr_periods: List[int] = field(default_factory=lambda: [7, 14])
    use_bh_encoding: bool = True
    use_order_book: bool = True
    use_macro: bool = False
    use_calendar: bool = True
    use_regime: bool = True
    normalize: bool = True
    clip_zscore: float = 5.0


# ---------------------------------------------------------------------------
# Basic price transforms
# ---------------------------------------------------------------------------

def log_returns(prices: np.ndarray, lag: int = 1) -> np.ndarray:
    """Compute log returns with given lag. Returns array of same length (NaN-padded)."""
    out = np.full_like(prices, np.nan, dtype=np.float64)
    out[lag:] = np.log(prices[lag:] / (prices[:-lag] + 1e-12))
    return out


def simple_returns(prices: np.ndarray, lag: int = 1) -> np.ndarray:
    """Simple (arithmetic) returns."""
    out = np.full_like(prices, np.nan, dtype=np.float64)
    out[lag:] = (prices[lag:] - prices[:-lag]) / (prices[:-lag] + 1e-12)
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean via cumsum."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    cumsum = np.cumsum(np.where(np.isnan(x), 0, x))
    count  = np.cumsum(~np.isnan(x))
    for i in range(window - 1, len(x)):
        n = count[i] - (count[i - window] if i >= window else 0)
        s = cumsum[i] - (cumsum[i - window] if i >= window else 0)
        out[i] = s / max(n, 1)
    return out


def rolling_std(x: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Rolling standard deviation."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(window - 1, len(x)):
        w = x[i - window + 1: i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) >= 2:
            out[i] = float(valid.std(ddof=ddof))
    return out


def rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Z-score relative to rolling window."""
    mu = rolling_mean(x, window)
    sigma = rolling_std(x, window)
    return (x - mu) / (sigma + 1e-12)


def rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    from collections import deque
    out = np.full_like(x, np.nan, dtype=np.float64)
    q: deque = deque()
    for i, val in enumerate(x):
        while q and x[q[-1]] >= val:
            q.pop()
        q.append(i)
        if q[0] <= i - window:
            q.popleft()
        if i >= window - 1:
            out[i] = x[q[0]]
    return out


def rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    from collections import deque
    out = np.full_like(x, np.nan, dtype=np.float64)
    q: deque = deque()
    for i, val in enumerate(x):
        while q and x[q[-1]] <= val:
            q.pop()
        q.append(i)
        if q[0] <= i - window:
            q.popleft()
        if i >= window - 1:
            out[i] = x[q[0]]
    return out


def exponential_weights(n: int, halflife: float) -> np.ndarray:
    """Exponential decay weights summing to 1."""
    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    weights = (1 - alpha) ** np.arange(n - 1, -1, -1)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Return features
# ---------------------------------------------------------------------------

def compute_return_features(
    closes: np.ndarray,
    windows: List[int],
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute multi-horizon return features.
    Returns matrix of shape (T, len(windows) * 3) with:
    [log_ret, sign, zscore] for each window.
    """
    T = len(closes)
    cols = []

    for w in windows:
        lr = log_returns(closes, w)
        sign = np.sign(lr)
        zs = rolling_zscore(lr, max(w * 2, 20))
        cols.extend([lr, sign, zs])

    out = np.column_stack(cols)  # (T, n_features)
    out = np.nan_to_num(out, nan=0.0)
    if normalize:
        out = np.clip(out, -5.0, 5.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Volatility features
# ---------------------------------------------------------------------------

def compute_volatility_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    windows: List[int],
) -> np.ndarray:
    """
    Realized vol, intraday range, Parkinson estimator, Yang-Zhang,
    and EWMA vol for each window.
    Returns (T, n_features).
    """
    T = len(closes)
    lr = log_returns(closes)
    lr = np.nan_to_num(lr)

    cols = []
    for w in windows:
        # Realized vol
        rv = rolling_std(lr, w)

        # Parkinson high-low estimator
        hl = np.log(highs / (lows + 1e-12))
        park = rolling_mean(hl ** 2, w)
        park = np.sqrt(park / (4 * np.log(2) + 1e-12))

        # EWMA vol
        ewma_var = np.full(T, np.nan)
        if T > 0:
            ewma_var[0] = lr[0] ** 2
            alpha_ewma = 2.0 / (w + 1)
            for i in range(1, T):
                ewma_var[i] = alpha_ewma * lr[i] ** 2 + (1 - alpha_ewma) * ewma_var[i - 1]
        ewma_vol = np.sqrt(np.abs(ewma_var))

        # Vol of vol
        vol_of_vol = rolling_std(rv, w)

        # Vol ratio (current / long-term)
        long_rv = rolling_std(lr, min(w * 4, T // 2 + 1))
        vol_ratio = rv / (long_rv + 1e-12)

        cols.extend([rv, park, ewma_vol, vol_of_vol, vol_ratio])

    out = np.column_stack(cols)
    out = np.nan_to_num(out, nan=0.0)
    out = np.clip(out, -5.0, 5.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Momentum & mean-reversion features
# ---------------------------------------------------------------------------

def compute_momentum_features(
    closes: np.ndarray,
    volumes: np.ndarray,
    windows: List[int],
) -> np.ndarray:
    """
    RSI, Stochastic, MFI, CMF, Aroon, Williams %R for each window.
    Returns (T, n_features).
    """
    T = len(closes)
    cols = []

    for w in windows:
        # RSI
        delta = np.diff(closes, prepend=closes[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = rolling_mean(gain, w)
        avg_loss = rolling_mean(loss, w)
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 1.0 - 1.0 / (1.0 + rs)
        cols.append(rsi)

        # Williams %R
        highest = rolling_max(closes, w)
        lowest  = rolling_min(closes, w)
        williams_r = (highest - closes) / (highest - lowest + 1e-12)
        cols.append(williams_r)

        # Stochastic %K
        stoch_k = (closes - lowest) / (highest - lowest + 1e-12)
        stoch_d = rolling_mean(stoch_k, 3)
        cols.extend([stoch_k, stoch_d])

        # Price position in channel
        channel_pos = (closes - lowest) / (highest - lowest + 1e-12)
        cols.append(channel_pos)

    out = np.column_stack(cols)
    out = np.nan_to_num(out, nan=0.5)  # neutral default
    out = np.clip(out, -5.0, 5.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Technical indicator features
# ---------------------------------------------------------------------------

def compute_rsi_features(prices: np.ndarray, periods: List[int]) -> np.ndarray:
    """RSI at multiple periods."""
    cols = []
    for p in periods:
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.full(len(prices), np.nan)
        avg_loss = np.full(len(prices), np.nan)

        # SMA for first period
        if len(prices) >= p:
            avg_gain[p - 1] = gain[:p].mean()
            avg_loss[p - 1] = loss[:p].mean()
            for i in range(p, len(prices)):
                avg_gain[i] = (avg_gain[i - 1] * (p - 1) + gain[i]) / p
                avg_loss[i] = (avg_loss[i - 1] * (p - 1) + loss[i]) / p

        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 1.0 - 1.0 / (1.0 + rs)
        rsi = np.nan_to_num(rsi, nan=0.5)

        # Overbought/oversold signals
        overbought = (rsi > 0.7).astype(np.float32)
        oversold   = (rsi < 0.3).astype(np.float32)
        rsi_mom    = np.diff(rsi, prepend=rsi[0])

        cols.extend([rsi, overbought, oversold, rsi_mom])

    return np.column_stack(cols).astype(np.float32)


def compute_bollinger_features(
    prices: np.ndarray,
    periods: List[int],
    n_std: float = 2.0,
) -> np.ndarray:
    """Bollinger band features at multiple periods."""
    cols = []
    for p in periods:
        mu    = rolling_mean(prices, p)
        sigma = rolling_std(prices, p)
        upper = mu + n_std * sigma
        lower = mu - n_std * sigma
        width = (upper - lower) / (mu + 1e-12)
        pos   = (prices - lower) / (upper - lower + 1e-12)  # 0=lower, 1=upper
        dev   = (prices - mu) / (sigma + 1e-12)             # z-score within band

        # Squeeze detection
        width_ma = rolling_mean(width, p)
        squeeze  = (width < width_ma * 0.8).astype(np.float32)

        cols.extend([pos, dev, width, squeeze])

    out = np.column_stack(cols)
    out = np.nan_to_num(out, nan=0.0)
    out = np.clip(out, -5.0, 5.0)
    return out.astype(np.float32)


def compute_macd_features(
    prices: np.ndarray,
    params: List[Tuple[int, int, int]],
) -> np.ndarray:
    """MACD features for multiple (fast, slow, signal) configurations."""
    cols = []
    price_scale = np.abs(prices).mean() + 1e-12

    for fast, slow, sig in params:
        ser = pd.Series(prices)
        ema_fast = ser.ewm(span=fast, adjust=False).mean().values
        ema_slow = ser.ewm(span=slow, adjust=False).mean().values
        macd_line = ema_fast - ema_slow
        sig_line  = pd.Series(macd_line).ewm(span=sig, adjust=False).mean().values
        histogram  = macd_line - sig_line

        # Normalize
        macd_norm = macd_line / price_scale
        sig_norm  = sig_line / price_scale
        hist_norm = histogram / price_scale

        # Crossover signals
        cross_up   = ((macd_line > sig_line) & (np.roll(macd_line, 1) <= np.roll(sig_line, 1))).astype(np.float32)
        cross_down = ((macd_line < sig_line) & (np.roll(macd_line, 1) >= np.roll(sig_line, 1))).astype(np.float32)

        cols.extend([macd_norm, sig_norm, hist_norm, cross_up, cross_down])

    out = np.column_stack(cols)
    out = np.nan_to_num(out, nan=0.0)
    out = np.clip(out, -5.0, 5.0)
    return out.astype(np.float32)


def compute_atr_features(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    periods: List[int],
) -> np.ndarray:
    """ATR at multiple periods, normalized by close."""
    cols = []
    for p in periods:
        tr = np.zeros(len(closes))
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)
        tr[0] = highs[0] - lows[0]

        atr = rolling_mean(tr, p)
        atr_norm = atr / (closes + 1e-12)

        # ATR percentile rank
        atr_rank = np.zeros(len(closes))
        for i in range(p, len(closes)):
            w = atr_norm[i - p: i]
            atr_rank[i] = float((w < atr_norm[i]).mean())

        cols.extend([atr_norm, atr_rank])

    out = np.column_stack(cols)
    out = np.nan_to_num(out, nan=0.0)
    out = np.clip(out, -5.0, 5.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Buy-and-Hold state encoding
# ---------------------------------------------------------------------------

class BHStateEncoder:
    """
    Encodes buy-and-hold baseline information as features for the RL agent.

    The "BH form" captures:
    - BH cumulative returns vs agent
    - Rolling alpha (excess return)
    - Information ratio
    - BH drawdown
    - Relative position sizing signals
    """

    def __init__(
        self,
        n_assets: int,
        initial_capital: float = 100_000.0,
        windows: Optional[List[int]] = None,
    ):
        self.n_assets = n_assets
        self.initial_capital = initial_capital
        self.windows = windows or [5, 20, 60]
        self._bh_returns: List[np.ndarray] = []
        self._agent_returns: List[float] = []
        self._bh_entry_prices: Optional[np.ndarray] = None

    def reset(self, initial_prices: np.ndarray) -> None:
        self._bh_entry_prices = initial_prices.copy()
        self._bh_returns = []
        self._agent_returns = []

    def update(self, current_prices: np.ndarray, agent_step_return: float) -> None:
        if self._bh_entry_prices is None:
            return
        price_rets = (current_prices - self._bh_entry_prices) / (self._bh_entry_prices + 1e-12)
        bh_ret = float(price_rets.mean())  # equal-weight BH
        self._bh_returns.append(np.array(price_rets, dtype=np.float64))
        self._agent_returns.append(agent_step_return)

    def encode(self) -> np.ndarray:
        """Return feature vector of length n_assets * n_windows * 4 + global_features."""
        if not self._bh_returns:
            return np.zeros(self._feature_dim(), dtype=np.float32)

        bh_arr = np.array([r.mean() for r in self._bh_returns])  # (t,)
        ag_arr = np.array(self._agent_returns)

        features = []

        # Global features
        bh_cum = float(np.cumprod(1 + bh_arr)[-1]) - 1.0
        ag_cum = float(np.cumprod(1 + ag_arr)[-1]) - 1.0
        features.append(bh_cum)
        features.append(ag_cum)
        features.append(ag_cum - bh_cum)  # total alpha

        # BH drawdown
        bh_wealth = np.cumprod(1 + bh_arr)
        bh_peak   = np.maximum.accumulate(bh_wealth)
        bh_dd     = float((bh_wealth[-1] / bh_peak[-1]) - 1.0)
        features.append(bh_dd)

        # Per-window rolling features
        for w in self.windows:
            if len(bh_arr) >= w:
                bh_w = bh_arr[-w:]
                ag_w = ag_arr[-w:]
                excess = ag_w - bh_w
                alpha   = float(excess.mean())
                te      = float(excess.std() + 1e-12)
                ir      = alpha / te
                bh_vol  = float(bh_w.std() + 1e-12)
                features.extend([alpha, te, ir, bh_vol])
            else:
                features.extend([0.0, 1.0, 0.0, 0.01])

        # Per-asset BH returns (last window)
        w = self.windows[-1]
        for i in range(self.n_assets):
            if len(self._bh_returns) >= w:
                asset_rets = np.array([r[i] for r in self._bh_returns[-w:]])
                features.append(float(asset_rets.mean()))
                features.append(float(asset_rets.std() + 1e-12))
                features.append(float(np.cumprod(1 + asset_rets)[-1]) - 1.0)
            else:
                features.extend([0.0, 0.01, 0.0])

        return np.array(features, dtype=np.float32)

    def _feature_dim(self) -> int:
        return 4 + len(self.windows) * 4 + self.n_assets * 3


# ---------------------------------------------------------------------------
# Order book proxy features (from OHLCV)
# ---------------------------------------------------------------------------

def compute_order_book_proxy_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    windows: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Proxy features for market microstructure / order book from OHLCV.
    Includes: VWAP deviation, volume profile, imbalance, liquidity.
    """
    windows = windows or [5, 20]
    T = len(closes)
    cols = []

    # Typical price & VWAP
    typical = (highs + lows + closes) / 3.0
    for w in windows:
        vwap = rolling_mean(typical * volumes, w) / (rolling_mean(volumes, w) + 1e-12)
        vwap_dev = (closes - vwap) / (vwap + 1e-12)
        cols.append(vwap_dev)

        # Volume-weighted price momentum
        vw_mom = rolling_mean(np.diff(closes, prepend=closes[0]) * volumes, w) / \
                 (rolling_mean(volumes, w) + 1e-12) / (closes + 1e-12)
        cols.append(vw_mom)

    # Intraday range (spread proxy)
    spread_proxy = (highs - lows) / (closes + 1e-12)
    cols.append(spread_proxy)

    # Candle body / wick ratio
    body = np.abs(closes - opens) / (highs - lows + 1e-12)
    upper_wick = (highs - np.maximum(opens, closes)) / (highs - lows + 1e-12)
    lower_wick = (np.minimum(opens, closes) - lows) / (highs - lows + 1e-12)
    cols.extend([body, upper_wick, lower_wick])

    # Price-volume correlation (short window)
    for w in windows:
        lr = np.diff(np.log(closes + 1e-12), prepend=0.0)
        pv_corr = np.full(T, np.nan)
        for i in range(w - 1, T):
            r_w = lr[i - w + 1: i + 1]
            v_w = volumes[i - w + 1: i + 1]
            if r_w.std() > 1e-12 and v_w.std() > 1e-12:
                pv_corr[i] = float(np.corrcoef(r_w, v_w)[0, 1])
            else:
                pv_corr[i] = 0.0
        cols.append(pv_corr)

    # Volume surge
    vol_ma = rolling_mean(volumes, 20)
    vol_surge = volumes / (vol_ma + 1e-12)
    cols.append(vol_surge)

    # Amihud illiquidity
    abs_ret = np.abs(np.diff(np.log(closes + 1e-12), prepend=0.0))
    dollar_vol = volumes * closes
    amihud = abs_ret / (dollar_vol + 1e-12)
    amihud_ma = rolling_mean(amihud, 20)
    cols.append(rolling_zscore(amihud_ma, 60))

    out = np.column_stack(cols)
    out = np.nan_to_num(out, nan=0.0)
    out = np.clip(out, -5.0, 5.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Macro / cross-asset features
# ---------------------------------------------------------------------------

def compute_cross_asset_features(
    price_matrix: np.ndarray,    # (T, n_assets)
    windows: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Cross-asset correlation, beta, lead-lag, and return dispersion features.
    """
    windows = windows or [20, 60]
    T, n = price_matrix.shape
    cols = []

    log_rets = np.diff(np.log(price_matrix + 1e-12), axis=0, prepend=np.log(price_matrix[:1] + 1e-12))

    for w in windows:
        # Rolling correlation matrix (upper triangle)
        for i in range(n):
            for j in range(i + 1, n):
                corr = np.full(T, np.nan)
                for t in range(w - 1, T):
                    ri = log_rets[t - w + 1: t + 1, i]
                    rj = log_rets[t - w + 1: t + 1, j]
                    if ri.std() > 1e-12 and rj.std() > 1e-12:
                        corr[t] = float(np.corrcoef(ri, rj)[0, 1])
                    else:
                        corr[t] = 0.0
                cols.append(corr)

        # Cross-sectional return dispersion
        dispersion = np.full(T, np.nan)
        for t in range(w - 1, T):
            ret_w = log_rets[t - w + 1: t + 1, :].mean(axis=0)  # (n,)
            dispersion[t] = float(ret_w.std())
        cols.append(dispersion)

        # Return of each asset vs cross-sectional mean
        for i in range(n):
            xsec_alpha = np.full(T, np.nan)
            for t in range(w - 1, T):
                ret_w = log_rets[t - w + 1: t + 1, :]
                mean_ret = ret_w.mean(axis=1)
                xsec_alpha[t] = float(ret_w[:, i].mean() - mean_ret.mean())
            cols.append(xsec_alpha)

    out = np.column_stack(cols)
    out = np.nan_to_num(out, nan=0.0)
    out = np.clip(out, -5.0, 5.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Calendar / time features
# ---------------------------------------------------------------------------

def compute_calendar_features(
    dates: Union[pd.DatetimeIndex, List],
    T: int,
) -> np.ndarray:
    """
    Calendar features: day-of-week, month, quarter, year-fraction,
    day-of-month, week-of-year, month-end flag.
    """
    if dates is None or len(dates) == 0:
        # Return synthetic time features
        t = np.linspace(0, 1, T)
        dow = np.sin(2 * np.pi * t * 5)
        month = np.sin(2 * np.pi * t * 12)
        return np.column_stack([t, dow, month]).astype(np.float32)

    dti = pd.DatetimeIndex(dates)
    T_actual = len(dti)

    dow    = dti.dayofweek / 4.0         # 0=Mon, 1=Fri, normalized
    month  = (dti.month - 1) / 11.0
    dom    = (dti.day - 1) / 30.0
    yfrac  = (dti.dayofyear - 1) / 365.0
    quarter = (dti.quarter - 1) / 3.0

    # Cyclical encoding
    dow_sin    = np.sin(2 * np.pi * dow)
    dow_cos    = np.cos(2 * np.pi * dow)
    month_sin  = np.sin(2 * np.pi * month)
    month_cos  = np.cos(2 * np.pi * month)
    yfrac_sin  = np.sin(2 * np.pi * yfrac)
    yfrac_cos  = np.cos(2 * np.pi * yfrac)

    # Month-end / month-start
    month_end   = (dti + pd.offsets.BDay(1)).month != dti.month
    month_start = (dti - pd.offsets.BDay(1)).month != dti.month

    out = np.column_stack([
        dow_sin, dow_cos,
        month_sin, month_cos,
        yfrac_sin, yfrac_cos,
        quarter,
        month_end.astype(np.float32),
        month_start.astype(np.float32),
    ])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Regime detection features
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Detect market regimes using:
    - Hidden Markov Model proxy (threshold-based)
    - Trend strength (ADX)
    - Volatility regime
    """

    REGIMES = {
        "trending_up": 0,
        "trending_down": 1,
        "ranging": 2,
        "high_vol": 3,
        "low_vol": 4,
    }

    def __init__(self, adx_period: int = 14, vol_window: int = 20):
        self.adx_period = adx_period
        self.vol_window = vol_window

    def compute_adx(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ADX, +DI, -DI."""
        T = len(closes)
        tr   = np.zeros(T)
        plus_dm  = np.zeros(T)
        minus_dm = np.zeros(T)

        for i in range(1, T):
            tr[i]   = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            up_move   = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            plus_dm[i]  = up_move   if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        p = self.adx_period
        tr_smooth    = rolling_mean(tr, p)
        pdm_smooth   = rolling_mean(plus_dm, p)
        mdm_smooth   = rolling_mean(minus_dm, p)

        plus_di  = 100.0 * pdm_smooth / (tr_smooth + 1e-12)
        minus_di = 100.0 * mdm_smooth / (tr_smooth + 1e-12)
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
        adx = rolling_mean(dx, p)

        return adx, plus_di, minus_di

    def compute_regime_features(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        """
        Returns (T, n_regime_features) array.
        Features: adx_norm, plus_di_norm, minus_di_norm, vol_regime,
                  regime_onehot (5), trend_strength, mean_reversion_signal.
        """
        adx, plus_di, minus_di = self.compute_adx(highs, lows, closes)

        lr = log_returns(closes)
        lr = np.nan_to_num(lr)
        vol = rolling_std(lr, self.vol_window)
        vol_ma = rolling_mean(vol, self.vol_window * 2)
        vol_ratio = vol / (vol_ma + 1e-12)

        # Regime classification
        T = len(closes)
        regime_labels = np.zeros(T, dtype=np.int32)
        for t in range(T):
            adx_t   = float(adx[t]) if not np.isnan(adx[t]) else 0.0
            pdi_t   = float(plus_di[t]) if not np.isnan(plus_di[t]) else 0.0
            mdi_t   = float(minus_di[t]) if not np.isnan(minus_di[t]) else 0.0
            vr_t    = float(vol_ratio[t]) if not np.isnan(vol_ratio[t]) else 1.0

            if vr_t > 1.5:
                regime_labels[t] = 3  # high_vol
            elif vr_t < 0.6:
                regime_labels[t] = 4  # low_vol
            elif adx_t > 25 and pdi_t > mdi_t:
                regime_labels[t] = 0  # trending_up
            elif adx_t > 25 and mdi_t > pdi_t:
                regime_labels[t] = 1  # trending_down
            else:
                regime_labels[t] = 2  # ranging

        # One-hot encode
        regime_onehot = np.zeros((T, 5), dtype=np.float32)
        for t in range(T):
            regime_onehot[t, regime_labels[t]] = 1.0

        # Mean reversion signal (Hurst exponent proxy)
        mr_signal = np.zeros(T)
        w = self.vol_window
        for t in range(w, T):
            ret_w = lr[t - w: t]
            autocorr = float(pd.Series(ret_w).autocorr(lag=1))
            mr_signal[t] = -autocorr  # positive = mean-reverting

        adx_norm = adx / 100.0
        pdi_norm = plus_di / 100.0
        mdi_norm = minus_di / 100.0

        out = np.column_stack([
            adx_norm, pdi_norm, mdi_norm,
            vol_ratio,
            regime_onehot,
            mr_signal,
        ])
        out = np.nan_to_num(out, nan=0.0)
        out = np.clip(out, -5.0, 5.0)
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def compute_rolling_statistics(
    returns: np.ndarray,
    windows: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Skewness, kurtosis, autocorrelation, Hurst exponent, and quantile features
    over rolling windows.
    """
    windows = windows or [20, 60]
    T = len(returns)
    cols = []

    for w in windows:
        skew  = np.full(T, np.nan)
        kurt  = np.full(T, np.nan)
        ac1   = np.full(T, np.nan)
        q10   = np.full(T, np.nan)
        q90   = np.full(T, np.nan)
        hurst = np.full(T, np.nan)

        for t in range(w - 1, T):
            r = returns[t - w + 1: t + 1]
            if r.std() > 1e-12:
                m3 = float(((r - r.mean()) ** 3).mean()) / (r.std() ** 3 + 1e-12)
                m4 = float(((r - r.mean()) ** 4).mean()) / (r.std() ** 4 + 1e-12) - 3.0
                skew[t] = m3
                kurt[t] = m4
            q10[t] = float(np.percentile(r, 10))
            q90[t] = float(np.percentile(r, 90))

            # AR(1) autocorrelation
            if len(r) >= 3:
                ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
                ac1[t] = ac if not np.isnan(ac) else 0.0

            # Simple Hurst estimate via variance ratios
            if w >= 10:
                half = w // 2
                var_half = float(r[:half].var() + 1e-12)
                var_full = float(r.var() + 1e-12)
                hurst[t] = float(np.log(var_full / var_half) / np.log(2)) * 0.5

        cols.extend([skew, kurt, ac1, q10, q90, hurst])

    out = np.column_stack(cols)
    out = np.nan_to_num(out, nan=0.0)
    out = np.clip(out, -5.0, 5.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Main FeatureEngineer class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Orchestrates all feature computation for a set of instruments.

    Usage:
        fe = FeatureEngineer(config)
        features_df = fe.compute(data_dict)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._regime_detector = RegimeDetector()

    def compute(
        self,
        data: Dict[str, pd.DataFrame],
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for each instrument.

        Args:
            data: dict mapping symbol -> DataFrame with OHLCV columns
            dates: optional DatetimeIndex for calendar features

        Returns:
            dict mapping symbol -> feature array (T, n_features)
        """
        result: Dict[str, np.ndarray] = {}

        # Compute cross-asset features if multiple instruments
        symbols = list(data.keys())
        n = len(symbols)
        T = min(len(df) for df in data.values())

        price_matrix = np.stack([
            data[s]["close"].values[-T:] for s in symbols
        ], axis=1)  # (T, n)

        cross_feats = None
        if self.config.use_macro and n > 1:
            cross_feats = compute_cross_asset_features(price_matrix)

        for sym in symbols:
            df = data[sym].copy()
            df.columns = [c.lower() for c in df.columns]
            opens   = df["open"].values.astype(np.float64)[-T:]
            highs   = df["high"].values.astype(np.float64)[-T:]
            lows    = df["low"].values.astype(np.float64)[-T:]
            closes  = df["close"].values.astype(np.float64)[-T:]
            volumes = df["volume"].values.astype(np.float64)[-T:]

            feature_parts = []

            # Return features
            ret_feats = compute_return_features(
                closes, self.config.returns_windows, self.config.normalize
            )
            feature_parts.append(ret_feats)

            # Volatility features
            vol_feats = compute_volatility_features(
                closes, highs, lows, self.config.vol_windows
            )
            feature_parts.append(vol_feats)

            # Momentum features
            mom_feats = compute_momentum_features(
                closes, volumes, self.config.vol_windows
            )
            feature_parts.append(mom_feats)

            # RSI
            rsi_feats = compute_rsi_features(closes, self.config.rsi_periods)
            feature_parts.append(rsi_feats)

            # Bollinger bands
            bb_feats = compute_bollinger_features(closes, self.config.bb_periods)
            feature_parts.append(bb_feats)

            # MACD
            macd_feats = compute_macd_features(closes, self.config.macd_params)
            feature_parts.append(macd_feats)

            # ATR
            atr_feats = compute_atr_features(highs, lows, closes, self.config.atr_periods)
            feature_parts.append(atr_feats)

            # Order book proxy
            if self.config.use_order_book:
                ob_feats = compute_order_book_proxy_features(opens, highs, lows, closes, volumes)
                feature_parts.append(ob_feats)

            # Regime features
            if self.config.use_regime:
                regime_feats = self._regime_detector.compute_regime_features(highs, lows, closes)
                feature_parts.append(regime_feats)

            # Rolling statistics
            lr = np.diff(np.log(closes + 1e-12), prepend=0.0)
            roll_stats = compute_rolling_statistics(lr)
            feature_parts.append(roll_stats)

            # Calendar features
            if self.config.use_calendar:
                sym_dates = dates if dates is not None else None
                cal_feats = compute_calendar_features(sym_dates, T)
                if cal_feats.shape[0] != T:
                    cal_feats = np.resize(cal_feats, (T, cal_feats.shape[1]))
                feature_parts.append(cal_feats)

            # Cross-asset features
            if cross_feats is not None:
                feature_parts.append(cross_feats[:T])

            # Concatenate
            combined = np.concatenate([f[:T] for f in feature_parts], axis=1)
            combined = np.nan_to_num(combined, nan=0.0, posinf=5.0, neginf=-5.0)
            if self.config.normalize:
                combined = np.clip(combined, -self.config.clip_zscore, self.config.clip_zscore)
            result[sym] = combined.astype(np.float32)

        return result

    def feature_names(self, n_assets: int = 1) -> List[str]:
        """Return list of feature names (for interpretation)."""
        names = []
        windows = self.config.returns_windows
        for w in windows:
            names += [f"log_ret_{w}", f"sign_ret_{w}", f"zscore_ret_{w}"]

        for w in self.config.vol_windows:
            names += [f"rv_{w}", f"parkinson_{w}", f"ewma_vol_{w}", f"vol_of_vol_{w}", f"vol_ratio_{w}"]

        for w in self.config.vol_windows:
            names += [f"rsi_{w}", f"williams_r_{w}", f"stoch_k_{w}", f"stoch_d_{w}", f"channel_pos_{w}"]

        for p in self.config.rsi_periods:
            names += [f"rsi{p}", f"ob{p}", f"os{p}", f"rsi_mom{p}"]

        for p in self.config.bb_periods:
            names += [f"bb_pos_{p}", f"bb_dev_{p}", f"bb_width_{p}", f"bb_squeeze_{p}"]

        for (f, s, sig) in self.config.macd_params:
            names += [f"macd_{f}_{s}", f"macd_sig_{f}_{s}", f"macd_hist_{f}_{s}",
                      f"macd_cup_{f}_{s}", f"macd_cdown_{f}_{s}"]

        for p in self.config.atr_periods:
            names += [f"atr_{p}", f"atr_rank_{p}"]

        if self.config.use_order_book:
            for w in [5, 20]:
                names += [f"vwap_dev_{w}", f"vw_mom_{w}"]
            names += ["spread_proxy", "body", "upper_wick", "lower_wick"]
            for w in [5, 20]:
                names += [f"pv_corr_{w}"]
            names += ["vol_surge", "amihud"]

        if self.config.use_regime:
            names += ["adx", "plus_di", "minus_di", "vol_ratio",
                      "regime_trend_up", "regime_trend_down", "regime_ranging",
                      "regime_high_vol", "regime_low_vol", "mr_signal"]

        for w in [20, 60]:
            names += [f"skew_{w}", f"kurt_{w}", f"ac1_{w}", f"q10_{w}", f"q90_{w}", f"hurst_{w}"]

        if self.config.use_calendar:
            names += ["dow_sin", "dow_cos", "month_sin", "month_cos",
                      "yfrac_sin", "yfrac_cos", "quarter", "month_end", "month_start"]

        return names


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

class OnlineNormalizer:
    """
    Welford's online algorithm for running mean/variance normalization.
    Used to normalize observations during RL training.
    """

    def __init__(self, shape: Tuple[int, ...], clip: float = 5.0, eps: float = 1e-8):
        self.shape = shape
        self.clip = clip
        self.eps = eps
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with batch of observations."""
        batch = np.asarray(x, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[np.newaxis, :]
        for xi in batch:
            self.count += 1
            delta = xi - self.mean
            self.mean += delta / self.count
            delta2 = xi - self.mean
            self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running stats."""
        normed = (x - self.mean) / (np.sqrt(self.var) + self.eps)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def denormalize(self, x_normed: np.ndarray) -> np.ndarray:
        return x_normed * (np.sqrt(self.var) + self.eps) + self.mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, var=self.var, count=np.array([self.count]))

    def load(self, path: str) -> None:
        data = np.load(path)
        self.mean  = data["mean"]
        self.var   = data["var"]
        self.count = int(data["count"][0])


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    T = 300

    # Synthetic OHLCV
    closes  = np.cumprod(1 + np.random.randn(T) * 0.01) * 100
    highs   = closes * (1 + np.abs(np.random.randn(T) * 0.005))
    lows    = closes * (1 - np.abs(np.random.randn(T) * 0.005))
    opens   = closes * (1 + np.random.randn(T) * 0.002)
    volumes = np.exp(np.random.randn(T) + 8)

    data = {
        "AAPL": pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}),
        "MSFT": pd.DataFrame({"open": opens * 1.1, "high": highs * 1.1, "low": lows * 1.1, "close": closes * 1.1, "volume": volumes}),
    }

    fe = FeatureEngineer()
    features = fe.compute(data)
    for sym, arr in features.items():
        print(f"{sym}: shape {arr.shape}, NaN: {np.isnan(arr).sum()}, range [{arr.min():.2f}, {arr.max():.2f}]")

    print("Feature engineering self-test passed.")
