"""
AETERNUS Real-Time Execution Layer (RTEL)
market_data_utils.py — Market Data Preprocessing and Feature Engineering

Utilities for converting raw LOB data into ML-ready feature matrices.
Used by Lumina, TensorNet, and OmniGraph Python modules.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

MAX_ASSETS     = 512
MAX_LOB_LEVELS = 10


# ---------------------------------------------------------------------------
# LOB feature extraction
# ---------------------------------------------------------------------------

def extract_lob_features(bids: List[Tuple[float, float]],
                          asks: List[Tuple[float, float]],
                          n_levels: int = 5) -> np.ndarray:
    """
    Extract standardized feature vector from LOB price/size levels.
    Returns float32 array of shape [4 * n_levels + 10].

    Features:
      [0:n]   bid prices (normalized by mid)
      [n:2n]  bid sizes (log-scaled)
      [2n:3n] ask prices (normalized by mid)
      [3n:4n] ask sizes (log-scaled)
      [-10:]  mid, spread, imbalance, vwap_bid, vwap_ask,
              log_bid_depth, log_ask_depth, price_pressure,
              level_ratio, microprice
    """
    n = n_levels
    feats = np.zeros(4 * n + 10, dtype=np.float32)

    if not bids or not asks:
        return feats

    mid   = (bids[0][0] + asks[0][0]) * 0.5
    if mid < 1e-10:
        return feats

    # Bid prices/sizes
    for i, (p, s) in enumerate(bids[:n]):
        feats[i]     = (p - mid) / mid  # negative for bids
        feats[n + i] = math.log1p(max(0.0, s))

    # Ask prices/sizes
    for i, (p, s) in enumerate(asks[:n]):
        feats[2*n + i] = (p - mid) / mid  # positive for asks
        feats[3*n + i] = math.log1p(max(0.0, s))

    # Derived features
    spread     = asks[0][0] - bids[0][0]
    bid_depth  = sum(s for _, s in bids[:n])
    ask_depth  = sum(s for _, s in asks[:n])
    total_depth= bid_depth + ask_depth

    imbalance  = (bid_depth - ask_depth) / (total_depth + 1e-10)
    vwap_bid   = sum(p*s for p,s in bids[:n]) / (bid_depth + 1e-10)
    vwap_ask   = sum(p*s for p,s in asks[:n]) / (ask_depth + 1e-10)

    # Micro-price
    bs, as_ = bids[0][1], asks[0][1]
    microprice = (bids[0][0] * as_ + asks[0][0] * bs) / (bs + as_ + 1e-10)

    # Price pressure (volume-weighted)
    bid_vol = sum(p*s for p,s in bids[:n])
    ask_vol = sum(p*s for p,s in asks[:n])
    price_pressure = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)

    off = 4 * n
    feats[off + 0] = float(mid)
    feats[off + 1] = float(spread / mid)
    feats[off + 2] = float(imbalance)
    feats[off + 3] = float((vwap_bid - mid) / mid)
    feats[off + 4] = float((vwap_ask - mid) / mid)
    feats[off + 5] = float(math.log1p(bid_depth))
    feats[off + 6] = float(math.log1p(ask_depth))
    feats[off + 7] = float(price_pressure)
    feats[off + 8] = float(min(len(bids), n) / (min(len(asks), n) + 1e-10))
    feats[off + 9] = float((microprice - mid) / (spread + 1e-10))

    return feats


def lob_to_feature_matrix(lob_snaps: List, n_assets: int,
                            n_levels: int = 5) -> np.ndarray:
    """
    Convert a list of LobSnapshot objects to a [n_assets × feature_dim] matrix.
    """
    from .shm_reader import LobSnapshot
    feat_dim = 4 * n_levels + 10
    matrix = np.zeros((n_assets, feat_dim), dtype=np.float32)
    for snap in lob_snaps:
        ai = getattr(snap, "asset_id", 0)
        if ai < n_assets:
            matrix[ai] = extract_lob_features(
                snap.bids[:n_levels], snap.asks[:n_levels], n_levels)
    return matrix


# ---------------------------------------------------------------------------
# Rolling feature windows
# ---------------------------------------------------------------------------

class RollingWindow:
    """Fixed-size rolling window of numpy arrays."""

    def __init__(self, shape: Tuple[int, ...], maxlen: int = 60,
                 dtype: np.dtype = np.float32):
        self.shape  = shape
        self.maxlen = maxlen
        self.dtype  = dtype
        self._data  = np.zeros((maxlen,) + shape, dtype=dtype)
        self._head  = 0
        self._count = 0

    def append(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=self.dtype).reshape(self.shape)
        self._data[self._head % self.maxlen] = arr
        self._head  += 1
        self._count  = min(self._count + 1, self.maxlen)

    def get(self, n: Optional[int] = None) -> np.ndarray:
        """Get last n items as [n × *shape] array (oldest first)."""
        n = min(n or self.maxlen, self._count)
        if n == 0:
            return np.zeros((0,) + self.shape, dtype=self.dtype)
        indices = [(self._head - n + i) % self.maxlen for i in range(n)]
        return self._data[indices]

    def latest(self) -> Optional[np.ndarray]:
        if self._count == 0:
            return None
        return self._data[(self._head - 1) % self.maxlen].copy()

    def count(self) -> int:
        return self._count

    def full(self) -> bool:
        return self._count >= self.maxlen


class MultiAssetRollingFeatures:
    """Maintains rolling LOB feature windows for multiple assets."""

    def __init__(self, n_assets: int, n_levels: int = 5, window: int = 60):
        self.n_assets = n_assets
        self.n_levels = n_levels
        self.window   = window
        feat_dim = 4 * n_levels + 10
        self._windows: Dict[int, RollingWindow] = {
            i: RollingWindow((feat_dim,), window) for i in range(n_assets)
        }

    def update(self, asset_id: int, bids: List, asks: List) -> None:
        if asset_id >= self.n_assets:
            return
        feats = extract_lob_features(bids, asks, self.n_levels)
        self._windows[asset_id].append(feats)

    def get_window(self, asset_id: int, n: Optional[int] = None) -> np.ndarray:
        w = self._windows.get(asset_id)
        if w is None:
            return np.zeros((0,), dtype=np.float32)
        return w.get(n)

    def current_matrix(self) -> np.ndarray:
        """Current features for all assets: [n_assets × feat_dim]."""
        rows = []
        for i in range(self.n_assets):
            lat = self._windows[i].latest()
            if lat is not None:
                rows.append(lat)
            else:
                rows.append(np.zeros(4 * self.n_levels + 10, dtype=np.float32))
        return np.stack(rows)


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------

def parkinson_vol(highs: np.ndarray, lows: np.ndarray) -> float:
    """Parkinson (high-low) volatility estimator."""
    if len(highs) == 0:
        return 0.0
    log_hl = np.log(highs / (lows + 1e-10)) ** 2
    return float(np.sqrt(np.mean(log_hl) / (4 * math.log(2))))


def garman_klass_vol(opens: np.ndarray, highs: np.ndarray,
                     lows: np.ndarray, closes: np.ndarray) -> float:
    """Garman-Klass volatility estimator."""
    if len(opens) == 0:
        return 0.0
    log_hl = np.log(highs / (lows + 1e-10)) ** 2
    log_co = np.log(closes / (opens + 1e-10)) ** 2
    gk = 0.5 * log_hl - (2 * math.log(2) - 1) * log_co
    return float(np.sqrt(np.mean(gk)))


def realized_vol(returns: np.ndarray, annualize: bool = True,
                  freq_per_year: float = 252 * 390) -> float:
    """Realized volatility from log returns."""
    if len(returns) == 0:
        return 0.0
    rv = float(np.sqrt(np.sum(returns ** 2)))
    if annualize:
        rv *= math.sqrt(freq_per_year / len(returns))
    return rv


def bipower_variation(returns: np.ndarray) -> float:
    """Barndorff-Nielsen & Shephard bipower variation (jump-robust vol)."""
    if len(returns) < 2:
        return 0.0
    bv = (math.pi / 2) * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))
    return float(math.sqrt(bv / len(returns)))


def vol_of_vol(vols: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling volatility of volatility."""
    if len(vols) < 2:
        return np.zeros_like(vols)
    vv = np.zeros_like(vols)
    for i in range(len(vols)):
        start = max(0, i - window + 1)
        chunk = vols[start:i+1]
        if len(chunk) > 1:
            vv[i] = float(np.std(chunk))
    return vv


# ---------------------------------------------------------------------------
# Returns and risk metrics
# ---------------------------------------------------------------------------

def log_returns(prices: np.ndarray) -> np.ndarray:
    return np.log(prices[1:] / (prices[:-1] + 1e-10))


def simple_returns(prices: np.ndarray) -> np.ndarray:
    return (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-10)


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0,
                 annualize: bool = True, periods_per_year: float = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / periods_per_year
    mean   = float(excess.mean())
    std    = float(excess.std())
    sr     = mean / (std + 1e-10)
    if annualize:
        sr *= math.sqrt(periods_per_year)
    return sr


def sortino_ratio(returns: np.ndarray, rf: float = 0.0,
                  annualize: bool = True, periods_per_year: float = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess    = returns - rf / periods_per_year
    downside  = excess[excess < 0]
    down_std  = float(downside.std()) if len(downside) > 0 else 1e-10
    sr        = float(excess.mean()) / down_std
    if annualize:
        sr *= math.sqrt(periods_per_year)
    return sr


def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd   = peak - cum
    return float(dd.max())


def calmar_ratio(returns: np.ndarray, periods_per_year: float = 252) -> float:
    pnl = np.cumsum(returns)
    mdd = max_drawdown(np.diff(pnl, prepend=0))
    ann_return = float(returns.mean()) * periods_per_year
    return ann_return / (mdd + 1e-10)


def var_historical(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Value at Risk (historical simulation)."""
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar_historical(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Expected Shortfall / CVaR (historical simulation)."""
    var = var_historical(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


# ---------------------------------------------------------------------------
# Order sizing utilities
# ---------------------------------------------------------------------------

def kelly_fraction(mu: float, sigma: float, leverage_cap: float = 2.0) -> float:
    """Full Kelly fraction = mu / sigma^2, capped at leverage_cap."""
    if sigma < 1e-10:
        return 0.0
    return min(leverage_cap, max(-leverage_cap, mu / (sigma ** 2)))


def half_kelly(mu: float, sigma: float) -> float:
    """Half-Kelly (more conservative sizing)."""
    return kelly_fraction(mu, sigma) * 0.5


def position_size_volatility_target(signal: float, vol: float,
                                     vol_target: float = 0.01) -> float:
    """Scale position to target portfolio volatility."""
    if vol < 1e-10:
        return 0.0
    return signal * vol_target / vol


def portfolio_weights_mean_variance(mu: np.ndarray, cov: np.ndarray,
                                     risk_aversion: float = 1.0,
                                     max_position: float = 1.0) -> np.ndarray:
    """Mean-variance optimal weights w = (1/gamma) * Sigma^-1 * mu."""
    n = len(mu)
    try:
        cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(n))
        w = (1.0 / risk_aversion) * cov_inv @ mu
        # Normalize and clip
        w = np.clip(w, -max_position, max_position)
        norm = np.sum(np.abs(w))
        if norm > max_position * n:
            w = w / norm * max_position * n
    except np.linalg.LinAlgError:
        w = np.zeros(n)
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Correlation and covariance utilities
# ---------------------------------------------------------------------------

def rolling_corr_matrix(returns_matrix: np.ndarray,
                          window: int = 60) -> np.ndarray:
    """
    Compute rolling correlation matrix from [T × N] returns.
    Returns [N × N] correlation matrix using last `window` observations.
    """
    T, N = returns_matrix.shape
    start = max(0, T - window)
    r = returns_matrix[start:]
    if len(r) < 2:
        return np.eye(N)
    return np.corrcoef(r.T)


def shrinkage_cov(sample_cov: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage covariance estimator."""
    n = sample_cov.shape[0]
    target = np.trace(sample_cov) / n * np.eye(n)
    return (1 - shrinkage) * sample_cov + shrinkage * target


def exponential_weighted_cov(returns: np.ndarray, alpha: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance (RiskMetrics style)."""
    T, N = returns.shape
    cov = np.eye(N) * 0.01
    for t in range(T):
        r = returns[t:t+1].T
        cov = alpha * cov + (1 - alpha) * (r @ r.T)
    return cov


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    if len(prices) < window:
        return np.full_like(prices, prices.mean() if len(prices) > 0 else 0.0)
    out = np.zeros_like(prices)
    for i in range(len(prices)):
        start = max(0, i - window + 1)
        out[i] = prices[start:i+1].mean()
    return out


def ema_array(prices: np.ndarray, window: int) -> np.ndarray:
    """Exponential moving average array."""
    alpha = 2.0 / (window + 1)
    out = np.zeros_like(prices, dtype=np.float64)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i-1]
    return out.astype(prices.dtype)


def rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    out = np.full(len(prices), 50.0)
    if len(deltas) < window:
        return out

    avg_gain = gains[:window].mean()
    avg_loss = losses[:window].mean()

    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        if avg_loss < 1e-10:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def bollinger_bands(prices: np.ndarray, window: int = 20,
                     n_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (upper, middle, lower) Bollinger Bands."""
    middle = sma(prices, window)
    std_arr = np.zeros_like(prices)
    for i in range(len(prices)):
        start = max(0, i - window + 1)
        std_arr[i] = prices[start:i+1].std()
    upper = middle + n_std * std_arr
    lower = middle - n_std * std_arr
    return upper, middle, lower


def macd(prices: np.ndarray, fast: int = 12, slow: int = 26,
         signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram."""
    ema_fast = ema_array(prices, fast)
    ema_slow = ema_array(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema_array(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(highs: np.ndarray, lows: np.ndarray,
        closes: np.ndarray, window: int = 14) -> np.ndarray:
    """Average True Range."""
    T = len(highs)
    tr = np.zeros(T)
    tr[0] = highs[0] - lows[0]
    for i in range(1, T):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1]))
    return ema_array(tr, window)


# ---------------------------------------------------------------------------
# Information coefficient (IC) analysis
# ---------------------------------------------------------------------------

def compute_ic(forecasts: np.ndarray, actuals: np.ndarray,
               method: str = "pearson") -> float:
    """Compute Information Coefficient between forecasts and actuals."""
    if len(forecasts) < 2:
        return 0.0
    if method == "pearson":
        corr = np.corrcoef(forecasts.flatten(), actuals.flatten())
        return float(corr[0, 1]) if not np.isnan(corr[0, 1]) else 0.0
    elif method == "spearman":
        from scipy.stats import spearmanr
        result = spearmanr(forecasts.flatten(), actuals.flatten())
        return float(result.correlation) if not np.isnan(result.correlation) else 0.0
    return 0.0


def compute_icir(ic_series: np.ndarray) -> float:
    """IC Information Ratio = mean(IC) / std(IC)."""
    if len(ic_series) < 2:
        return 0.0
    return float(np.mean(ic_series)) / (float(np.std(ic_series)) + 1e-10)


def hit_rate(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Fraction of correct directional predictions."""
    if len(forecasts) == 0:
        return 0.0
    correct = np.sign(forecasts) == np.sign(actuals)
    return float(correct.mean())


# ---------------------------------------------------------------------------
# DataNormalizer — standardization and normalization
# ---------------------------------------------------------------------------

class DataNormalizer:
    """Online (streaming) normalizer using running mean and variance."""

    def __init__(self, n_features: int, eps: float = 1e-6, clip: float = 5.0):
        self.n = n_features
        self.eps = eps
        self.clip = clip
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._var  = np.ones(n_features, dtype=np.float64)
        self._count = 0

    def update(self, x: np.ndarray) -> None:
        """Update running statistics (Welford's online algorithm)."""
        x = np.asarray(x, dtype=np.float64).flatten()[:self.n]
        self._count += 1
        delta = x - self._mean[:len(x)]
        self._mean[:len(x)] += delta / self._count
        delta2 = x - self._mean[:len(x)]
        self._var[:len(x)] += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mean = self._mean.astype(np.float32)
        std  = np.sqrt(self._var / max(1, self._count - 1) + self.eps).astype(np.float32)
        out  = (x - mean) / std
        return np.clip(out, -self.clip, self.clip)

    def fit_batch(self, data: np.ndarray) -> None:
        """Fit to a batch of data [T × n_features]."""
        self._mean = data.mean(axis=0).astype(np.float64)
        self._var  = data.var(axis=0).astype(np.float64)
        self._count= len(data)

    def transform_batch(self, data: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._var / max(1, self._count - 1) + self.eps)
        out = (data - self._mean) / std
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self._mean.tolist(), "var": self._var.tolist(),
                "count": self._count}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self._mean  = np.array(d["mean"])
        self._var   = np.array(d["var"])
        self._count = d["count"]
