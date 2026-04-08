"""
feature_engineering.py — Comprehensive feature engineering for ML signal generation.

Covers:
  - Price features: multi-lookback returns
  - Vol features: realized vol, vol-of-vol, vol ratio
  - Technical indicators: RSI, MACD, Bollinger %B, ATR, ADX
  - Microstructure: bid-ask spread proxy, Amihud illiquidity
  - Statistical: skewness, kurtosis, autocorrelation, Hurst exponent
  - Regime: HMM-based bull/bear/sideways probabilities
  - Cross-asset correlations
  - Calendar effects
  - Normalization: rolling z-score, rank transform
  - Feature selection: rolling IC filter, mutual information threshold
  - FeatureMatrix: builds and maintains rolling feature matrix
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import digamma


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_div(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-12, a / b, fill)
    return result


def _rolling_window(x: np.ndarray, window: int) -> np.ndarray:
    """Return 2D array of shape (T - window + 1, window) via stride tricks."""
    n = len(x)
    if n < window:
        return np.empty((0, window))
    shape = (n - window + 1, window)
    strides = (x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


# ---------------------------------------------------------------------------
# 1. Price Features
# ---------------------------------------------------------------------------

RETURN_LOOKBACKS = [1, 5, 10, 21, 63, 126, 252]


def price_features(closes: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute log returns at multiple horizons.
    Output arrays are same length as closes (NaN-padded at start).
    """
    T = len(closes)
    log_p = np.log(np.maximum(closes, 1e-10))
    feats: Dict[str, np.ndarray] = {}

    for lb in RETURN_LOOKBACKS:
        ret = np.full(T, np.nan)
        ret[lb:] = log_p[lb:] - log_p[:-lb]
        feats[f"ret_{lb}d"] = ret

    # Momentum score: sign-weighted combination
    momentum = np.zeros(T)
    weights = np.array([1, 1, 2, 3, 5, 8, 13], dtype=float)
    weights /= weights.sum()
    for i, lb in enumerate(RETURN_LOOKBACKS):
        r = feats[f"ret_{lb}d"]
        mask = ~np.isnan(r)
        momentum[mask] += weights[i] * r[mask]
    feats["momentum_composite"] = momentum

    # Price-to-moving-average ratios
    for w in [20, 50, 200]:
        ma = np.full(T, np.nan)
        for t in range(w - 1, T):
            ma[t] = np.mean(closes[t - w + 1:t + 1])
        ratio = _safe_div(closes - ma, ma)
        feats[f"price_ma_ratio_{w}d"] = ratio

    return feats


# ---------------------------------------------------------------------------
# 2. Volatility Features
# ---------------------------------------------------------------------------

VOL_WINDOWS = [5, 10, 21, 63]


def vol_features(closes: np.ndarray, highs: Optional[np.ndarray] = None,
                  lows: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    T = len(closes)
    log_rets = np.full(T, np.nan)
    log_rets[1:] = np.diff(np.log(np.maximum(closes, 1e-10)))
    feats: Dict[str, np.ndarray] = {}

    for w in VOL_WINDOWS:
        rv = np.full(T, np.nan)
        for t in range(w, T):
            rv[t] = float(np.std(log_rets[t - w + 1:t + 1], ddof=1)) * math.sqrt(252)
        feats[f"realized_vol_{w}d"] = rv

    # Vol-of-vol: std of rolling realized vol
    rv_21 = feats["realized_vol_21d"]
    vov = np.full(T, np.nan)
    vov_window = 21
    for t in range(vov_window * 2, T):
        window = rv_21[t - vov_window + 1:t + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 5:
            vov[t] = float(np.std(valid, ddof=1))
    feats["vol_of_vol"] = vov

    # Vol ratio: short-term / long-term
    feats["vol_ratio_5_21"] = _safe_div(feats["realized_vol_5d"], feats["realized_vol_21d"])
    feats["vol_ratio_21_63"] = _safe_div(feats["realized_vol_21d"], feats["realized_vol_63d"])

    # Parkinson estimator (uses high/low if available)
    if highs is not None and lows is not None:
        pk = np.full(T, np.nan)
        for t in range(21, T):
            h = highs[t - 21 + 1:t + 1]
            l = lows[t - 21 + 1:t + 1]
            hl_sq = (np.log(h / np.maximum(l, 1e-10))) ** 2
            pk[t] = math.sqrt(np.mean(hl_sq) / (4 * math.log(2))) * math.sqrt(252)
        feats["parkinson_vol_21d"] = pk

    return feats


# ---------------------------------------------------------------------------
# 3. Technical Indicators
# ---------------------------------------------------------------------------

def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    T = len(closes)
    result = np.full(T, np.nan)
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    for t in range(period, T):
        avg_gain = float(np.mean(gains[t - period:t]))
        avg_loss = float(np.mean(losses[t - period:t]))
        if avg_loss < 1e-10:
            result[t] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[t] = 100.0 - 100.0 / (1.0 + rs)

    return result


def macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal_window: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (macd_line, signal_line)."""
    T = len(closes)

    def ema(arr: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        out = np.full(T, np.nan)
        out[0] = arr[0]
        for i in range(1, T):
            if np.isnan(out[i - 1]):
                out[i] = arr[i]
            else:
                out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(np.where(np.isnan(macd_line), 0.0, macd_line), signal_window)
    return macd_line, signal_line


def bollinger_pctb(closes: np.ndarray, window: int = 20, n_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (%B, bandwidth)."""
    T = len(closes)
    pctb = np.full(T, np.nan)
    bw = np.full(T, np.nan)
    for t in range(window - 1, T):
        w = closes[t - window + 1:t + 1]
        mu = float(np.mean(w))
        sigma = float(np.std(w, ddof=1))
        upper = mu + n_std * sigma
        lower = mu - n_std * sigma
        if upper - lower > 1e-10:
            pctb[t] = (closes[t] - lower) / (upper - lower)
        bw[t] = (upper - lower) / max(mu, 1e-10)
    return pctb, bw


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    T = len(closes)
    tr = np.full(T, np.nan)
    tr[0] = highs[0] - lows[0]
    for t in range(1, T):
        tr[t] = max(
            highs[t] - lows[t],
            abs(highs[t] - closes[t - 1]),
            abs(lows[t] - closes[t - 1]),
        )
    atr_arr = np.full(T, np.nan)
    if T >= period:
        atr_arr[period - 1] = float(np.mean(tr[:period]))
        alpha = 1.0 / period
        for t in range(period, T):
            atr_arr[t] = alpha * tr[t] + (1 - alpha) * atr_arr[t - 1]
    return atr_arr


def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    T = len(closes)
    plus_dm = np.zeros(T)
    minus_dm = np.zeros(T)
    for t in range(1, T):
        up_move = highs[t] - highs[t - 1]
        down_move = lows[t - 1] - lows[t]
        plus_dm[t] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[t] = down_move if down_move > up_move and down_move > 0 else 0.0

    atr_arr = atr(highs, lows, closes, period)
    plus_di = 100.0 * _safe_div(
        np.convolve(plus_dm, np.ones(period) / period, mode="same"),
        np.maximum(atr_arr, 1e-10)
    )
    minus_di = 100.0 * _safe_div(
        np.convolve(minus_dm, np.ones(period) / period, mode="same"),
        np.maximum(atr_arr, 1e-10)
    )
    di_sum = plus_di + minus_di
    dx = 100.0 * _safe_div(np.abs(plus_di - minus_di), np.maximum(di_sum, 1e-10))
    adx_arr = np.full(T, np.nan)
    if T >= 2 * period:
        adx_arr[2 * period - 1] = float(np.mean(dx[period:2 * period]))
        for t in range(2 * period, T):
            adx_arr[t] = ((period - 1) * adx_arr[t - 1] + dx[t]) / period
    return adx_arr


def technical_features(
    closes: np.ndarray,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    feats: Dict[str, np.ndarray] = {}
    feats["rsi_14"] = rsi(closes, 14)
    feats["rsi_28"] = rsi(closes, 28)
    macd_l, macd_s = macd(closes)
    feats["macd_line"] = macd_l
    feats["macd_signal"] = macd_s
    feats["macd_hist"] = macd_l - macd_s
    pctb, bw = bollinger_pctb(closes)
    feats["bollinger_pctb"] = pctb
    feats["bollinger_bw"] = bw

    if highs is not None and lows is not None:
        feats["atr_14"] = atr(highs, lows, closes)
        feats["adx_14"] = adx(highs, lows, closes)

    return feats


# ---------------------------------------------------------------------------
# 4. Microstructure Features
# ---------------------------------------------------------------------------

def microstructure_features(
    closes: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    T = len(closes)
    feats: Dict[str, np.ndarray] = {}

    # Bid-ask spread proxy: Roll (1984) estimator
    # cov(delta_p_t, delta_p_{t-1}) = -c^2, spread = 2c
    log_rets = np.full(T, np.nan)
    log_rets[1:] = np.diff(np.log(np.maximum(closes, 1e-10)))
    roll_spread = np.full(T, np.nan)
    window = 21
    for t in range(window, T):
        r = log_rets[t - window + 1:t + 1]
        r = r[~np.isnan(r)]
        if len(r) >= 4:
            cov = float(np.cov(r[1:], r[:-1])[0, 1])
            roll_spread[t] = 2.0 * math.sqrt(abs(cov)) if cov < 0 else 0.0
    feats["roll_spread_proxy"] = roll_spread

    # Amihud illiquidity: |ret| / volume
    if volumes is not None:
        abs_ret = np.abs(log_rets)
        illiq = _safe_div(abs_ret, np.maximum(volumes, 1.0))
        amihud = np.full(T, np.nan)
        for t in range(window, T):
            window_illiq = illiq[t - window + 1:t + 1]
            valid = window_illiq[~np.isnan(window_illiq)]
            if len(valid) > 0:
                amihud[t] = float(np.mean(valid))
        feats["amihud_illiquidity"] = amihud
        feats["volume_ratio"] = np.full(T, np.nan)
        for t in range(window, T):
            avg_vol = float(np.mean(volumes[t - window + 1:t]))
            feats["volume_ratio"][t] = volumes[t] / max(avg_vol, 1.0)

    # High-low range as spread proxy
    if highs is not None and lows is not None:
        hl_spread = _safe_div(highs - lows, np.maximum(closes, 1e-10))
        feats["hl_spread_pct"] = hl_spread

    return feats


# ---------------------------------------------------------------------------
# 5. Statistical Features
# ---------------------------------------------------------------------------

def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    """Estimate Hurst exponent via R/S analysis."""
    n = len(ts)
    if n < max_lag * 2:
        return 0.5
    lags = range(2, max_lag)
    rs_vals = []
    for lag in lags:
        chunks = n // lag
        if chunks < 2:
            continue
        rs_list = []
        for i in range(chunks):
            chunk = ts[i * lag:(i + 1) * lag]
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            dev = np.cumsum(chunk - mean_c)
            r = float(dev.max() - dev.min())
            s = float(np.std(chunk, ddof=1))
            if s > 1e-10:
                rs_list.append(r / s)
        if rs_list:
            rs_vals.append(np.mean(rs_list))
        else:
            rs_vals.append(np.nan)

    valid = [(math.log(lag), math.log(rs)) for lag, rs in zip(lags, rs_vals)
             if not math.isnan(rs) and rs > 0]
    if len(valid) < 3:
        return 0.5
    x_arr = np.array([v[0] for v in valid])
    y_arr = np.array([v[1] for v in valid])
    slope, _, _, _, _ = stats.linregress(x_arr, y_arr)
    return float(np.clip(slope, 0.0, 1.0))


def statistical_features(closes: np.ndarray, window: int = 63) -> Dict[str, np.ndarray]:
    T = len(closes)
    log_rets = np.full(T, np.nan)
    log_rets[1:] = np.diff(np.log(np.maximum(closes, 1e-10)))
    feats: Dict[str, np.ndarray] = {}

    skew_arr = np.full(T, np.nan)
    kurt_arr = np.full(T, np.nan)
    ac1_arr = np.full(T, np.nan)
    hurst_arr = np.full(T, np.nan)

    for t in range(window, T):
        r = log_rets[t - window + 1:t + 1]
        r = r[~np.isnan(r)]
        if len(r) < 10:
            continue
        skew_arr[t] = float(stats.skew(r))
        kurt_arr[t] = float(stats.kurtosis(r))
        if len(r) > 2:
            ac1_arr[t] = float(np.corrcoef(r[1:], r[:-1])[0, 1]) if len(r) > 2 else 0.0
        hurst_arr[t] = hurst_exponent(r, max_lag=min(20, len(r) // 4))

    feats["skewness_63d"] = skew_arr
    feats["kurtosis_63d"] = kurt_arr
    feats["autocorr_lag1_63d"] = ac1_arr
    feats["hurst_63d"] = hurst_arr

    return feats


# ---------------------------------------------------------------------------
# 6. Regime Features (HMM-based)
# ---------------------------------------------------------------------------

def _hmm_viterbi_2state(returns: np.ndarray) -> np.ndarray:
    """
    Minimal 2-state HMM (bull/bear) via Viterbi.
    States: 0=bear (low mean, high vol), 1=bull (high mean, low vol).
    Uses Gaussian emission. Parameters estimated via simple moment matching.
    """
    T = len(returns)
    if T < 20:
        return np.zeros(T, dtype=int)

    mid = T // 2
    sorted_r = np.sort(returns)
    bear_data = sorted_r[:T // 2]
    bull_data = sorted_r[T // 2:]
    mu = np.array([float(np.mean(bear_data)), float(np.mean(bull_data))])
    sigma = np.array([max(float(np.std(bear_data)), 1e-5), max(float(np.std(bull_data)), 1e-5)])

    A = np.array([[0.95, 0.05], [0.05, 0.95]])  # transition matrix
    pi = np.array([0.5, 0.5])

    def emission(r: float, state: int) -> float:
        return stats.norm.pdf(r, mu[state], sigma[state]) + 1e-300

    log_delta = np.full((T, 2), -np.inf)
    psi = np.zeros((T, 2), dtype=int)

    for s in range(2):
        log_delta[0, s] = math.log(pi[s]) + math.log(emission(returns[0], s))

    for t in range(1, T):
        for s in range(2):
            candidates = [log_delta[t - 1, s_prev] + math.log(A[s_prev, s]) for s_prev in range(2)]
            best_prev = int(np.argmax(candidates))
            log_delta[t, s] = candidates[best_prev] + math.log(emission(returns[t], s))
            psi[t, s] = best_prev

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(log_delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


def regime_features(closes: np.ndarray, window: int = 63) -> Dict[str, np.ndarray]:
    T = len(closes)
    log_rets = np.full(T, np.nan)
    log_rets[1:] = np.diff(np.log(np.maximum(closes, 1e-10)))
    feats: Dict[str, np.ndarray] = {}

    bull_prob = np.full(T, np.nan)
    regime = np.full(T, np.nan)

    for t in range(window, T):
        r = log_rets[t - window + 1:t + 1]
        r = r[~np.isnan(r)]
        if len(r) < 20:
            continue
        path = _hmm_viterbi_2state(r)
        regime[t] = float(path[-1])
        bull_prob[t] = float(np.mean(path))

    feats["hmm_regime"] = regime
    feats["hmm_bull_prob"] = bull_prob

    # Trend regime: MA cross
    fast_ma = np.full(T, np.nan)
    slow_ma = np.full(T, np.nan)
    for t in range(20, T):
        fast_ma[t] = float(np.mean(closes[t - 20 + 1:t + 1]))
    for t in range(63, T):
        slow_ma[t] = float(np.mean(closes[t - 63 + 1:t + 1]))
    feats["trend_regime"] = np.where(
        ~np.isnan(fast_ma) & ~np.isnan(slow_ma),
        (fast_ma > slow_ma).astype(float),
        np.nan,
    )

    return feats


# ---------------------------------------------------------------------------
# 7. Cross-Asset Features
# ---------------------------------------------------------------------------

def cross_asset_features(
    target_returns: np.ndarray,
    btc_returns: Optional[np.ndarray] = None,
    sp500_returns: Optional[np.ndarray] = None,
    gold_returns: Optional[np.ndarray] = None,
    window: int = 21,
) -> Dict[str, np.ndarray]:
    T = len(target_returns)
    feats: Dict[str, np.ndarray] = {}
    proxies = {
        "btc_corr": btc_returns,
        "sp500_corr": sp500_returns,
        "gold_corr": gold_returns,
    }

    for name, proxy in proxies.items():
        if proxy is None or len(proxy) != T:
            feats[name] = np.full(T, np.nan)
            feats[name.replace("corr", "beta")] = np.full(T, np.nan)
            continue
        corr_arr = np.full(T, np.nan)
        beta_arr = np.full(T, np.nan)
        for t in range(window, T):
            x = proxy[t - window + 1:t + 1]
            y = target_returns[t - window + 1:t + 1]
            valid = ~(np.isnan(x) | np.isnan(y))
            xv, yv = x[valid], y[valid]
            if len(xv) >= 5:
                corr_arr[t] = float(np.corrcoef(xv, yv)[0, 1])
                var_x = float(np.var(xv))
                beta_arr[t] = float(np.cov(xv, yv)[0, 1]) / max(var_x, 1e-10)
        feats[name] = corr_arr
        feats[name.replace("corr", "beta")] = beta_arr

    return feats


# ---------------------------------------------------------------------------
# 8. Calendar Features
# ---------------------------------------------------------------------------

def calendar_features(timestamps: np.ndarray) -> Dict[str, np.ndarray]:
    """
    timestamps: array of Unix timestamps (seconds) or datetime64.
    Returns sin/cos encodings for day-of-week, month, hour.
    """
    try:
        dt = timestamps.astype("datetime64[s]").astype(object)
    except (AttributeError, TypeError):
        dt = np.array([0] * len(timestamps))

    T = len(timestamps)
    feats: Dict[str, np.ndarray] = {}

    try:
        import datetime
        dow = np.array([d.weekday() if hasattr(d, "weekday") else 0 for d in dt], dtype=float)
        month = np.array([d.month if hasattr(d, "month") else 1 for d in dt], dtype=float)
        hour = np.array([d.hour if hasattr(d, "hour") else 0 for d in dt], dtype=float)
    except Exception:
        dow = np.zeros(T)
        month = np.ones(T)
        hour = np.zeros(T)

    feats["dow_sin"] = np.sin(2 * math.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * math.pi * dow / 7)
    feats["month_sin"] = np.sin(2 * math.pi * (month - 1) / 12)
    feats["month_cos"] = np.cos(2 * math.pi * (month - 1) / 12)
    feats["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * math.pi * hour / 24)

    # End-of-month indicator (last 3 trading days)
    feats["is_month_end"] = (dow >= 4).astype(float)  # proxy

    return feats


# ---------------------------------------------------------------------------
# 9. Feature Normalization
# ---------------------------------------------------------------------------

def rolling_zscore(arr: np.ndarray, window: int = 63) -> np.ndarray:
    T = len(arr)
    result = np.full(T, np.nan)
    for t in range(window, T):
        w = arr[t - window + 1:t + 1]
        valid = w[~np.isnan(w)]
        if len(valid) < 5:
            continue
        mu = float(np.mean(valid))
        sigma = float(np.std(valid, ddof=1))
        if sigma > 1e-10:
            result[t] = (arr[t] - mu) / sigma
    return result


def rank_transform(arr: np.ndarray) -> np.ndarray:
    """
    Cross-sectional rank transform.
    Ranks values in arr (already cross-sectional), maps to [-1, 1].
    For time-series usage: rank of arr[t] within arr[t-window:t].
    """
    T = len(arr)
    result = np.full(T, np.nan)
    valid_mask = ~np.isnan(arr)
    valid_vals = arr[valid_mask]
    if len(valid_vals) == 0:
        return result
    ranks = stats.rankdata(valid_vals, method="average")
    n = len(ranks)
    normalized = (ranks - 1) / max(n - 1, 1) * 2 - 1
    result[valid_mask] = normalized
    return result


def normalize_features(
    feature_matrix: np.ndarray,         # shape (T, F)
    method: str = "zscore",
    window: int = 63,
) -> np.ndarray:
    """Normalize each feature column independently."""
    T, F = feature_matrix.shape
    out = np.full_like(feature_matrix, np.nan)
    for f in range(F):
        if method == "zscore":
            out[:, f] = rolling_zscore(feature_matrix[:, f], window)
        elif method == "rank":
            out[:, f] = rank_transform(feature_matrix[:, f])
        elif method == "minmax":
            col = feature_matrix[:, f]
            valid = col[~np.isnan(col)]
            if len(valid) > 1:
                lo, hi = valid.min(), valid.max()
                out[:, f] = (col - lo) / max(hi - lo, 1e-10)
        else:
            out[:, f] = feature_matrix[:, f]
    return out


# ---------------------------------------------------------------------------
# 10. Feature Selection
# ---------------------------------------------------------------------------

def rolling_ic(
    features: np.ndarray,       # shape (T,)
    forward_returns: np.ndarray,  # shape (T,)
    window: int = 63,
) -> np.ndarray:
    """Rolling Information Coefficient (Spearman rank correlation)."""
    T = len(features)
    ic = np.full(T, np.nan)
    for t in range(window, T):
        f = features[t - window + 1:t + 1]
        r = forward_returns[t - window + 1:t + 1]
        valid = ~(np.isnan(f) | np.isnan(r))
        if valid.sum() >= 10:
            ic[t], _ = stats.spearmanr(f[valid], r[valid])
    return ic


def _mutual_information_knn(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    KNN-based mutual information estimator (Kraskov et al. 2004).
    Both x and y should be 1D arrays of equal length.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]
    n = len(xv)
    if n < k + 2:
        return 0.0

    joint = np.column_stack([xv, yv])
    # Normalize to [0,1]
    joint -= joint.min(axis=0)
    joint /= (joint.max(axis=0) + 1e-10)

    # For each point, find kth nearest neighbor in joint space
    mi = 0.0
    for i in range(n):
        dists = np.linalg.norm(joint - joint[i], axis=1, ord=np.inf)
        dists[i] = np.inf
        sorted_d = np.sort(dists)
        eps = sorted_d[k - 1]
        nx = np.sum(np.abs(xv - xv[i]) < eps) - 1
        ny = np.sum(np.abs(yv - yv[i]) < eps) - 1
        mi += digamma(k) - (1.0 / k) + digamma(n) - digamma(max(nx, 1)) - digamma(max(ny, 1))

    return max(float(mi / n), 0.0)


def select_features_by_ic(
    feature_matrix: np.ndarray,    # shape (T, F)
    forward_returns: np.ndarray,   # shape (T,)
    ic_threshold: float = 0.03,
    window: int = 63,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select features where mean |IC| > ic_threshold.
    Returns (selected_matrix, selected_indices).
    """
    F = feature_matrix.shape[1]
    selected = []
    for f in range(F):
        ic = rolling_ic(feature_matrix[:, f], forward_returns, window)
        valid_ic = ic[~np.isnan(ic)]
        if len(valid_ic) > 0 and float(np.mean(np.abs(valid_ic))) >= ic_threshold:
            selected.append(f)
    idx = np.array(selected, dtype=int)
    if len(idx) == 0:
        return np.empty((len(feature_matrix), 0)), idx
    return feature_matrix[:, idx], idx


# ---------------------------------------------------------------------------
# FeatureMatrix: Main Interface
# ---------------------------------------------------------------------------

@dataclass
class FeatureMatrixConfig:
    include_price: bool = True
    include_vol: bool = True
    include_technical: bool = True
    include_microstructure: bool = True
    include_statistical: bool = True
    include_regime: bool = True
    include_cross_asset: bool = False
    include_calendar: bool = False
    normalization: str = "zscore"     # "zscore", "rank", "minmax", "none"
    norm_window: int = 63
    ic_filter: bool = False
    ic_threshold: float = 0.03
    ic_window: int = 63


class FeatureMatrix:
    """
    Builds and maintains a rolling feature matrix from OHLCV data.
    Call build() to compute all features. Call get_latest() to get the
    most recent feature vector for inference.
    """
    def __init__(self, cfg: FeatureMatrixConfig):
        self.cfg = cfg
        self.feature_names: List[str] = []
        self.raw_matrix: Optional[np.ndarray] = None
        self.normalized_matrix: Optional[np.ndarray] = None
        self.selected_indices: Optional[np.ndarray] = None

    def build(
        self,
        closes: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        cross_assets: Optional[Dict[str, np.ndarray]] = None,
        forward_returns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        cfg = self.cfg
        all_feats: Dict[str, np.ndarray] = {}

        if cfg.include_price:
            all_feats.update(price_features(closes))

        if cfg.include_vol:
            all_feats.update(vol_features(closes, highs, lows))

        if cfg.include_technical:
            all_feats.update(technical_features(closes, highs, lows))

        if cfg.include_microstructure:
            all_feats.update(microstructure_features(closes, volumes, highs, lows))

        if cfg.include_statistical:
            all_feats.update(statistical_features(closes))

        if cfg.include_regime:
            all_feats.update(regime_features(closes))

        if cfg.include_cross_asset and cross_assets is not None:
            log_rets = np.full(len(closes), np.nan)
            log_rets[1:] = np.diff(np.log(np.maximum(closes, 1e-10)))
            ca = cross_asset_features(
                log_rets,
                btc_returns=cross_assets.get("btc"),
                sp500_returns=cross_assets.get("sp500"),
                gold_returns=cross_assets.get("gold"),
            )
            all_feats.update(ca)

        if cfg.include_calendar and timestamps is not None:
            all_feats.update(calendar_features(timestamps))

        # Stack into matrix
        self.feature_names = sorted(all_feats.keys())
        T = len(closes)
        mat = np.full((T, len(self.feature_names)), np.nan)
        for i, name in enumerate(self.feature_names):
            arr = all_feats[name]
            n = min(len(arr), T)
            mat[:n, i] = arr[:n]

        self.raw_matrix = mat

        # Normalize
        if cfg.normalization != "none":
            self.normalized_matrix = normalize_features(mat, cfg.normalization, cfg.norm_window)
        else:
            self.normalized_matrix = mat.copy()

        # Feature selection via IC
        if cfg.ic_filter and forward_returns is not None:
            sel_mat, sel_idx = select_features_by_ic(
                self.normalized_matrix, forward_returns, cfg.ic_threshold, cfg.ic_window
            )
            self.selected_indices = sel_idx
            return sel_mat
        else:
            self.selected_indices = np.arange(len(self.feature_names))
            return self.normalized_matrix

    def get_latest(self) -> Optional[np.ndarray]:
        """Return the last row of the normalized (and optionally filtered) matrix."""
        if self.normalized_matrix is None:
            return None
        row = self.normalized_matrix[-1]
        if self.selected_indices is not None:
            row = row[self.selected_indices]
        # Replace NaN with 0
        return np.where(np.isnan(row), 0.0, row)

    def get_feature_names(self) -> List[str]:
        if self.selected_indices is not None:
            return [self.feature_names[i] for i in self.selected_indices]
        return self.feature_names

    def describe(self) -> Dict:
        if self.normalized_matrix is None:
            return {}
        mat = self.normalized_matrix
        return {
            "n_timesteps": mat.shape[0],
            "n_features_raw": len(self.feature_names),
            "n_features_selected": len(self.selected_indices) if self.selected_indices is not None else len(self.feature_names),
            "nan_fraction": float(np.mean(np.isnan(mat))),
            "feature_names": self.get_feature_names(),
        }
