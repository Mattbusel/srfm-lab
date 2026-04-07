"""
research/signal_analytics/signal_library.py
============================================
100+ signal implementations for the SRFM-Lab alpha engine.

Signal signature:
    signal(prices: pd.Series, volume: pd.Series = None, **params) -> pd.Series

All signals:
- Return a pd.Series aligned to the input index.
- Use real mathematical formulas (no stubs).
- Produce NaN during warmup periods; stable values after.
- Accept keyword parameters with sensible defaults.

Categories:
- MOMENTUM (20 signals)
- MEAN_REVERSION (20 signals)
- VOLATILITY (15 signals)
- MICROSTRUCTURE (15 signals)
- PHYSICS / SRFM-specific (15 signals)
- TECHNICAL (15 signals)

Dependencies: numpy, pandas, scipy
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log_returns(prices: pd.Series) -> pd.Series:
    """Log return series."""
    return np.log(prices.replace(0.0, float("nan"))).diff()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=2).std()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window, min_periods=window // 2).mean()
    s = series.rolling(window, min_periods=window // 2).std()
    return (series - m) / s.replace(0.0, float("nan"))


def _rank_norm(series: pd.Series) -> pd.Series:
    """Rank normalize to [-1, 1]."""
    ranked = series.rank(pct=True)
    return 2 * ranked - 1


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


def _winsorize(series: pd.Series, limits: float = 0.01) -> pd.Series:
    lo = series.quantile(limits)
    hi = series.quantile(1 - limits)
    return series.clip(lo, hi)


def _hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 100) -> float:
    """
    Compute Hurst exponent via R/S analysis on a series.
    Returns value in [0, 1]. H>0.5 = trending, H<0.5 = mean-reverting.
    """
    ts = series.dropna().values
    if len(ts) < max_lag:
        return float("nan")
    lags = range(min_lag, min(max_lag, len(ts) // 4))
    rs_vals = []
    for lag in lags:
        chunks = [ts[i : i + lag] for i in range(0, len(ts) - lag, lag)]
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            deviations = np.cumsum(chunk - mean)
            R = deviations.max() - deviations.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append((lag, np.mean(rs_chunk)))
    if len(rs_vals) < 3:
        return float("nan")
    log_lags = np.log([x[0] for x in rs_vals])
    log_rs = np.log([x[1] for x in rs_vals])
    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
    return float(slope)


# ===========================================================================
# MOMENTUM SIGNALS (20)
# ===========================================================================


def mom_1d(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """1-day price momentum (log return)."""
    return _log_returns(prices)


def mom_5d(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """5-day log return."""
    lp = np.log(prices.replace(0.0, float("nan")))
    return lp - lp.shift(5)


def mom_20d(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """20-day log return."""
    lp = np.log(prices.replace(0.0, float("nan")))
    return lp - lp.shift(20)


def mom_60d(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """60-day log return."""
    lp = np.log(prices.replace(0.0, float("nan")))
    return lp - lp.shift(60)


def mom_252d(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """252-day (annual) log return."""
    lp = np.log(prices.replace(0.0, float("nan")))
    return lp - lp.shift(252)


def mom_sharpe(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Risk-adjusted momentum: rolling mean return / rolling std (Sharpe momentum).
    Uses log returns.
    """
    lr = _log_returns(prices)
    mu = lr.rolling(window, min_periods=window // 2).mean()
    sigma = lr.rolling(window, min_periods=window // 2).std()
    return mu / sigma.replace(0.0, float("nan"))


def mom_acceleration(prices: pd.Series, volume: pd.Series = None, window: int = 10, **kwargs) -> pd.Series:
    """
    Momentum acceleration: second derivative of log price.
    (momentum - lagged momentum)
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    mom = lp - lp.shift(window)
    return mom - mom.shift(window)


def mom_52w_high(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """52-week high ratio: price / rolling-252-bar high."""
    high_252 = prices.rolling(252, min_periods=60).max()
    return prices / high_252.replace(0.0, float("nan"))


def mom_crash_protection(prices: pd.Series, volume: pd.Series = None, window: int = 20, vol_window: int = 60, **kwargs) -> pd.Series:
    """
    Momentum crash protection.
    mom_20d * (1 - vol_ratio) when recent vol is elevated.
    vol_ratio = recent_vol / long_vol.
    Clips the protection factor to [0, 1].
    """
    lr = _log_returns(prices)
    recent_vol = lr.rolling(window, min_periods=window // 2).std()
    long_vol = lr.rolling(vol_window, min_periods=vol_window // 2).std()
    vol_ratio = recent_vol / long_vol.replace(0.0, float("nan"))
    protection = (1.0 - vol_ratio).clip(0.0, 1.0)
    lp = np.log(prices.replace(0.0, float("nan")))
    raw_mom = lp - lp.shift(20)
    return raw_mom * protection


def mom_ts_moskowitz(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """
    Time-series momentum (Moskowitz et al. 2012).
    sign(12m_return) * |12m_return|
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    ret_12m = lp - lp.shift(252)
    return np.sign(ret_12m) * ret_12m.abs()


def mom_cs_rank(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Cross-sectional momentum rank proxy.
    Rank of the 20-day return within the rolling distribution of past 252-day returns.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    ret_20 = lp - lp.shift(window)
    return ret_20.rolling(252, min_periods=60).rank(pct=True) * 2 - 1


def mom_seasonality(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """
    Momentum adjusted for calendar seasonality.
    Uses same-month returns from 12 months ago as seasonal adjustment.
    Seasonal return = price[t] / price[t-252] - 1 minus price[t-21]/price[t-273] - 1.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    mom_12m = lp - lp.shift(252)
    seasonal = lp.shift(21) - lp.shift(273)
    return mom_12m - seasonal


def mom_dual(prices: pd.Series, volume: pd.Series = None, rf_rate: float = 0.02 / 252, **kwargs) -> pd.Series:
    """
    Dual momentum: max(price_mom, rf_rate) > 0 -> 1, else -1.
    Returns binary signal.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    ret_12m = lp - lp.shift(252)
    return pd.Series(
        np.where(ret_12m > rf_rate * 252, 1.0, -1.0),
        index=prices.index,
    )


def mom_absolute(prices: pd.Series, volume: pd.Series = None, window: int = 252, **kwargs) -> pd.Series:
    """Absolute momentum: 1 if 12m return > 0, else -1."""
    lp = np.log(prices.replace(0.0, float("nan")))
    ret = lp - lp.shift(window)
    return ret.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))


def mom_intermediate(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """Intermediate-term momentum: 2-12 months (skips last month reversal)."""
    lp = np.log(prices.replace(0.0, float("nan")))
    return (lp.shift(21) - lp.shift(252))


def mom_short_reversal(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """Short-term reversal: negative of 1-month return."""
    lp = np.log(prices.replace(0.0, float("nan")))
    return -(lp - lp.shift(21))


def mom_end_of_month(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """
    Monthly rebalancing signal: returns 1 on the last trading day of a month
    (end-of-month effect), else 0.
    """
    idx = prices.index
    if hasattr(idx, "to_period"):
        month_end = idx.to_series().groupby(idx.to_period("M")).transform("last")
        signal = (idx.to_series() == month_end).astype(float)
    else:
        signal = pd.Series(0.0, index=idx)
    signal.index = idx
    return signal


def mom_gap(prices: pd.Series, volume: pd.Series = None, window: int = 14, **kwargs) -> pd.Series:
    """
    Gap signal: overnight gap ratio normalized by ATR.
    Gap = close[t] / close[t-1] - 1 used as proxy (requires open; uses close here).
    Normalized by rolling ATR.
    """
    lr = _log_returns(prices)
    atr_proxy = lr.abs().ewm(span=window, adjust=False).mean()
    return lr / atr_proxy.replace(0.0, float("nan"))


def mom_volume_weighted(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Volume-weighted momentum.
    Sum of (daily_return * volume) / total_volume over window.
    """
    lr = _log_returns(prices)
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    vol_norm = volume / volume.rolling(window, min_periods=1).mean()
    weighted = lr * vol_norm
    return weighted.rolling(window, min_periods=window // 2).sum()


def mom_up_down_volume(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """Up/down volume ratio: sum(volume on up days) / sum(volume on down days)."""
    lr = _log_returns(prices)
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    up_vol = volume.where(lr > 0, 0.0).rolling(window, min_periods=window // 2).sum()
    dn_vol = volume.where(lr <= 0, 0.0).rolling(window, min_periods=window // 2).sum()
    return (up_vol / dn_vol.replace(0.0, float("nan"))) - 1.0


def mom_tick_proxy(prices: pd.Series, volume: pd.Series = None, window: int = 14, **kwargs) -> pd.Series:
    """
    Tick momentum proxy: sign of return / ATR.
    Approximates the tick imbalance signal using closing prices.
    """
    lr = _log_returns(prices)
    atr_proxy = lr.abs().rolling(window, min_periods=window // 2).mean()
    return lr / atr_proxy.replace(0.0, float("nan"))


def mom_price_accel(prices: pd.Series, volume: pd.Series = None, window: int = 5, **kwargs) -> pd.Series:
    """Price acceleration: 2nd derivative of log price."""
    lp = np.log(prices.replace(0.0, float("nan")))
    first_deriv = lp.diff()
    second_deriv = first_deriv.diff()
    return second_deriv.rolling(window, min_periods=1).mean()


def mom_multi_tf_composite(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """
    Multi-timeframe momentum composite.
    Equal-weight average of standardized 5d, 20d, 60d, 252d log returns.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    returns = {}
    for w in [5, 20, 60, 252]:
        r = lp - lp.shift(w)
        std = r.rolling(252, min_periods=60).std()
        returns[w] = r / std.replace(0.0, float("nan"))
    df = pd.DataFrame(returns)
    return df.mean(axis=1)


# ===========================================================================
# MEAN REVERSION SIGNALS (20)
# ===========================================================================


def mr_zscore_10(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """Z-score mean reversion, 10-bar window."""
    return -_zscore(prices, 10)


def mr_zscore_20(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """Z-score mean reversion, 20-bar window."""
    return -_zscore(prices, 20)


def mr_zscore_50(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """Z-score mean reversion, 50-bar window."""
    return -_zscore(prices, 50)


def mr_bollinger_position(prices: pd.Series, volume: pd.Series = None, window: int = 20, n_std: float = 2.0, **kwargs) -> pd.Series:
    """
    Bollinger Band position: (price - mid) / bandwidth.
    Negative = mean reverting signal (price above band -> short).
    """
    mid = prices.rolling(window, min_periods=window // 2).mean()
    sigma = prices.rolling(window, min_periods=window // 2).std()
    bandwidth = 2 * n_std * sigma
    position = (prices - mid) / bandwidth.replace(0.0, float("nan"))
    return -position  # invert: high position -> negative signal


def mr_rsi(prices: pd.Series, volume: pd.Series = None, window: int = 14, **kwargs) -> pd.Series:
    """
    RSI-based mean reversion signal.
    RSI bounded [0, 100]. Signal: -(RSI - 50) / 50.
    Overbought (RSI > 70) -> negative signal.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=window, adjust=False).mean()
    rs = gain / loss.replace(0.0, float("nan"))
    rsi = 100 - 100 / (1 + rs)
    return -(rsi - 50.0) / 50.0


def mr_linreg_residual(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Linear regression residual mean reversion.
    Fit OLS on log(price) vs time index over rolling window;
    return the most recent residual (normalized).
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    residuals = pd.Series(index=prices.index, dtype=float)
    for i in range(window, len(lp) + 1):
        chunk = lp.iloc[i - window : i].dropna()
        if len(chunk) < window // 2:
            continue
        x = np.arange(len(chunk), dtype=float)
        slope, intercept, _, _, _ = stats.linregress(x, chunk.values)
        fitted = slope * x[-1] + intercept
        res = chunk.iloc[-1] - fitted
        residuals.iloc[i - 1] = res
    std = residuals.rolling(window, min_periods=window // 2).std()
    return -(residuals / std.replace(0.0, float("nan")))


def mr_sma_deviation(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """Moving average deviation: -(price / SMA - 1)."""
    sma = prices.rolling(window, min_periods=window // 2).mean()
    dev = prices / sma.replace(0.0, float("nan")) - 1.0
    return -dev


def mr_kalman_residual(prices: pd.Series, volume: pd.Series = None, process_var: float = 1e-4, obs_var: float = 1e-2, **kwargs) -> pd.Series:
    """
    Kalman filter residual (price minus Kalman estimate).
    Uses a simple constant-level Kalman filter.
    """
    p = prices.values.astype(float)
    n = len(p)
    x_est = np.full(n, float("nan"))
    P = 1.0
    x = p[0] if np.isfinite(p[0]) else 0.0

    for i in range(n):
        if not np.isfinite(p[i]):
            x_est[i] = x
            continue
        # Predict
        P = P + process_var
        # Update
        K = P / (P + obs_var)
        x = x + K * (p[i] - x)
        P = (1 - K) * P
        x_est[i] = x

    residuals = pd.Series(p - x_est, index=prices.index)
    std = residuals.rolling(20, min_periods=5).std()
    return -(residuals / std.replace(0.0, float("nan")))


def mr_pairs_ratio_zscore(prices: pd.Series, volume: pd.Series = None, prices2: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Pairs ratio z-score.
    If prices2 is provided: z-score of log(prices / prices2).
    Otherwise z-score of log(prices).
    """
    if prices2 is not None and len(prices2) == len(prices):
        ratio = np.log((prices / prices2.replace(0.0, float("nan"))).replace(0.0, float("nan")))
    else:
        ratio = np.log(prices.replace(0.0, float("nan")))
    return -_zscore(ratio, window)


def mr_ou_weighted(prices: pd.Series, volume: pd.Series = None, window: int = 60, **kwargs) -> pd.Series:
    """
    Half-life weighted mean reversion (OU model).
    Fits an AR(1) to log prices, computes the mean-reversion signal
    weighted by the speed of mean reversion (kappa).
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    signal = pd.Series(index=prices.index, dtype=float)
    for i in range(window, len(lp) + 1):
        chunk = lp.iloc[i - window : i].dropna()
        if len(chunk) < 10:
            continue
        y = chunk.values
        x = y[:-1]
        y1 = y[1:]
        if np.std(x) < 1e-10:
            continue
        slope, intercept, _, _, _ = stats.linregress(x, y1)
        kappa = -np.log(max(abs(slope), 1e-10))
        mu = intercept / (1 - slope + 1e-10)
        dev = y[-1] - mu
        std = np.std(y1 - (slope * x + intercept))
        if std > 0:
            signal.iloc[i - 1] = -dev / std * kappa
    return signal


def mr_vwap_deviation(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """Absolute deviation from VWAP (negative for mean reversion)."""
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    cum_pv = (prices * volume).rolling(window, min_periods=1).sum()
    cum_vol = volume.rolling(window, min_periods=1).sum()
    vwap = cum_pv / cum_vol.replace(0.0, float("nan"))
    dev = (prices - vwap) / vwap.replace(0.0, float("nan"))
    return -dev


def mr_price_oscillator(prices: pd.Series, volume: pd.Series = None, fast: int = 10, slow: int = 30, **kwargs) -> pd.Series:
    """Price oscillator: EMA(fast) - EMA(slow), normalized by EMA(slow). Inverted."""
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    osc = (ema_fast - ema_slow) / ema_slow.replace(0.0, float("nan"))
    return -osc


def mr_dpo(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Detrended Price Oscillator.
    DPO = price - SMA shifted back by (window/2 + 1).
    """
    offset = window // 2 + 1
    sma_shifted = prices.rolling(window, min_periods=window // 2).mean().shift(offset)
    dpo = prices - sma_shifted
    std = dpo.rolling(window, min_periods=window // 2).std()
    return -(dpo / std.replace(0.0, float("nan")))


def mr_cci(prices: pd.Series, volume: pd.Series = None, window: int = 20, constant: float = 0.015, **kwargs) -> pd.Series:
    """
    Commodity Channel Index.
    CCI = (price - SMA(price)) / (constant * mean_abs_deviation).
    """
    sma = prices.rolling(window, min_periods=window // 2).mean()
    mad = prices.rolling(window, min_periods=window // 2).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci = (prices - sma) / (constant * mad.replace(0.0, float("nan")))
    return -(cci / 100.0).clip(-3, 3)


def mr_williams_r(prices: pd.Series, volume: pd.Series = None, window: int = 14, **kwargs) -> pd.Series:
    """
    Williams %R.
    %R = (highest_high - close) / (highest_high - lowest_low) * -100.
    Bounded [-100, 0]. Inverted for mean-reversion signal.
    """
    highest = prices.rolling(window, min_periods=window // 2).max()
    lowest = prices.rolling(window, min_periods=window // 2).min()
    wr = (highest - prices) / (highest - lowest).replace(0.0, float("nan")) * (-100.0)
    # Normalize to [-1, 1]: very low %R (near -100) -> buy signal
    return (wr + 50.0) / 50.0


def mr_stochastic_k(prices: pd.Series, volume: pd.Series = None, window: int = 14, **kwargs) -> pd.Series:
    """
    Stochastic oscillator %K.
    %K = (close - lowest_low) / (highest_high - lowest_low) * 100.
    Bounded [0, 100]. Inverted and normalized.
    """
    lowest = prices.rolling(window, min_periods=window // 2).min()
    highest = prices.rolling(window, min_periods=window // 2).max()
    k = (prices - lowest) / (highest - lowest).replace(0.0, float("nan")) * 100.0
    return -((k - 50.0) / 50.0)


def mr_chande_momentum(prices: pd.Series, volume: pd.Series = None, window: int = 14, **kwargs) -> pd.Series:
    """
    Chande Momentum Oscillator.
    CMO = (sum_up - sum_down) / (sum_up + sum_down) * 100.
    """
    diff = prices.diff()
    up = diff.where(diff > 0, 0.0).rolling(window, min_periods=window // 2).sum()
    down = (-diff.where(diff < 0, 0.0)).rolling(window, min_periods=window // 2).sum()
    total = (up + down).replace(0.0, float("nan"))
    cmo = (up - down) / total * 100.0
    return -(cmo / 100.0)


def mr_roc_mean_rev(prices: pd.Series, volume: pd.Series = None, window: int = 10, **kwargs) -> pd.Series:
    """Rate of change with mean reversion filter: -(ROC - EMA(ROC))."""
    roc = prices / prices.shift(window).replace(0.0, float("nan")) - 1.0
    ema_roc = _ema(roc, window * 2)
    return -(roc - ema_roc)


def mr_price_channel(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Price channel mean reversion.
    Position within N-bar range: (price - low) / (high - low).
    Inverted (high in range = mean reversion sell signal).
    """
    low = prices.rolling(window, min_periods=window // 2).min()
    high = prices.rolling(window, min_periods=window // 2).max()
    pos = (prices - low) / (high - low).replace(0.0, float("nan"))
    return -(pos - 0.5) * 2.0


def mr_log_autoregression(prices: pd.Series, volume: pd.Series = None, window: int = 20, lags: int = 1, **kwargs) -> pd.Series:
    """
    Log-price autoregression residual.
    Regress log(price[t]) on log(price[t-1]) rolling; return standardized residual.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    residuals = pd.Series(index=prices.index, dtype=float)
    for i in range(window, len(lp) + 1):
        chunk = lp.iloc[i - window : i].dropna()
        if len(chunk) < 5:
            continue
        y = chunk.iloc[lags:].values
        x = chunk.iloc[:-lags].values
        if np.std(x) < 1e-10:
            continue
        slope, intercept, _, _, _ = stats.linregress(x, y)
        residual = y[-1] - (slope * x[-1] + intercept)
        residuals.iloc[i - 1] = residual
    std = residuals.rolling(window, min_periods=window // 2).std()
    return -(residuals / std.replace(0.0, float("nan")))


def mr_hurst_adjusted(prices: pd.Series, volume: pd.Series = None, window: int = 60, hurst_window: int = 100, threshold: float = 0.45, **kwargs) -> pd.Series:
    """
    Mean reversion signal active only when Hurst exponent < threshold.
    H < 0.45 -> mean reverting regime -> apply z-score signal.
    """
    base_signal = mr_zscore_20(prices)
    hurst_vals = pd.Series(index=prices.index, dtype=float)
    for i in range(hurst_window, len(prices) + 1, max(1, hurst_window // 10)):
        chunk = prices.iloc[max(0, i - hurst_window) : i]
        h = _hurst_exponent(chunk, min_lag=2, max_lag=min(50, len(chunk) // 4))
        hurst_vals.iloc[i - 1] = h
    hurst_vals = hurst_vals.ffill()
    active = (hurst_vals < threshold).astype(float)
    return base_signal * active


# ===========================================================================
# VOLATILITY SIGNALS (15)
# ===========================================================================


def vol_ewma_forecast(prices: pd.Series, volume: pd.Series = None, span: int = 20, **kwargs) -> pd.Series:
    """GARCH proxy: EWMA volatility forecast (annualized)."""
    lr = _log_returns(prices)
    ewma_var = lr.pow(2).ewm(span=span, adjust=False).mean()
    return np.sqrt(ewma_var * 252)


def vol_realized_5d(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """5-day realized volatility (annualized)."""
    lr = _log_returns(prices)
    return lr.rolling(5, min_periods=2).std() * np.sqrt(252)


def vol_realized_20d(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """20-day realized volatility (annualized)."""
    lr = _log_returns(prices)
    return lr.rolling(20, min_periods=5).std() * np.sqrt(252)


def vol_realized_60d(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """60-day realized volatility (annualized)."""
    lr = _log_returns(prices)
    return lr.rolling(60, min_periods=15).std() * np.sqrt(252)


def vol_of_vol(prices: pd.Series, volume: pd.Series = None, inner: int = 5, outer: int = 20, **kwargs) -> pd.Series:
    """Volatility of volatility: std of daily realized vol over outer window."""
    lr = _log_returns(prices)
    daily_vol = lr.rolling(inner, min_periods=2).std()
    return daily_vol.rolling(outer, min_periods=outer // 2).std()


def vol_regime(prices: pd.Series, volume: pd.Series = None, window: int = 20, long_window: int = 252, **kwargs) -> pd.Series:
    """
    Volatility regime signal: recent_vol / long_vol - 1.
    Positive = high vol regime.
    """
    lr = _log_returns(prices)
    recent = lr.rolling(window, min_periods=window // 2).std()
    long = lr.rolling(long_window, min_periods=long_window // 4).std()
    return recent / long.replace(0.0, float("nan")) - 1.0


def vol_atr_percentile(prices: pd.Series, volume: pd.Series = None, window: int = 20, percentile_window: int = 252, **kwargs) -> pd.Series:
    """
    ATR percentile: where today's ATR sits in the N-bar distribution.
    Uses rolling percentile rank.
    """
    lr = _log_returns(prices).abs()
    atr = lr.ewm(span=window, adjust=False).mean()
    pct = atr.rolling(percentile_window, min_periods=60).rank(pct=True)
    return pct


def vol_skew_proxy(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Volatility skew proxy: upside_vol / downside_vol - 1.
    upside_vol = std of positive returns, downside_vol = std of negative returns.
    """
    lr = _log_returns(prices)
    up = lr.where(lr > 0, float("nan")).rolling(window, min_periods=5).std()
    dn = lr.where(lr < 0, float("nan")).rolling(window, min_periods=5).std()
    return up / dn.replace(0.0, float("nan")) - 1.0


def vol_term_structure(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """Vol term structure proxy: 5d realized vol / 60d realized vol."""
    v5 = vol_realized_5d(prices)
    v60 = vol_realized_60d(prices)
    return v5 / v60.replace(0.0, float("nan")) - 1.0


def vol_parkinson(prices: pd.Series, volume: pd.Series = None, window: int = 20,
                  high: pd.Series = None, low: pd.Series = None, **kwargs) -> pd.Series:
    """
    Parkinson volatility estimator.
    Uses high-low range. If high/low not provided, approximates from close.
    sigma^2 = 1/(4*n*ln2) * sum(ln(H/L)^2)
    """
    if high is None or low is None:
        # Approximate using rolling max/min of close
        high = prices.rolling(2, min_periods=1).max()
        low = prices.rolling(2, min_periods=1).min()
    log_hl = np.log((high / low.replace(0.0, float("nan"))).replace(0.0, float("nan")))
    factor = 1.0 / (4.0 * math.log(2))
    var = (log_hl ** 2 * factor).rolling(window, min_periods=window // 2).mean()
    return np.sqrt(var * 252)


def vol_garman_klass(prices: pd.Series, volume: pd.Series = None, window: int = 20,
                     high: pd.Series = None, low: pd.Series = None, open_: pd.Series = None, **kwargs) -> pd.Series:
    """
    Garman-Klass volatility estimator.
    sigma^2 = 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2
    Falls back to Parkinson if open not provided.
    """
    if high is None or low is None:
        return vol_parkinson(prices, window=window)
    log_hl = np.log((high / low.replace(0.0, float("nan"))).replace(0.0, float("nan")))
    if open_ is not None and not open_.empty:
        log_co = np.log((prices / open_.replace(0.0, float("nan"))).replace(0.0, float("nan")))
        gk = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2
    else:
        gk = 0.5 * log_hl ** 2
    var = gk.rolling(window, min_periods=window // 2).mean()
    return np.sqrt(var.clip(0) * 252)


def vol_yang_zhang(prices: pd.Series, volume: pd.Series = None, window: int = 20,
                   high: pd.Series = None, low: pd.Series = None, open_: pd.Series = None, **kwargs) -> pd.Series:
    """
    Yang-Zhang volatility estimator.
    Combines overnight, open-to-close, and Rogers-Satchell components.
    Falls back to close-based EWMA when OHLC not available.
    """
    if high is None or low is None or open_ is None:
        return vol_ewma_forecast(prices, span=window)
    log_ho = np.log(high / open_.replace(0.0, float("nan")))
    log_lo = np.log(low / open_.replace(0.0, float("nan")))
    log_co = np.log(prices / open_.replace(0.0, float("nan")))
    log_oc = np.log(open_ / prices.shift(1).replace(0.0, float("nan")))

    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
    sigma_o = log_oc.rolling(window, min_periods=window // 2).var()
    sigma_c = log_co.rolling(window, min_periods=window // 2).var()
    rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window, min_periods=window // 2).mean()
    var = sigma_o + k * sigma_c + (1 - k) * rs
    return np.sqrt(var.clip(0) * 252)


def vol_rogers_satchell(prices: pd.Series, volume: pd.Series = None, window: int = 20,
                        high: pd.Series = None, low: pd.Series = None, open_: pd.Series = None, **kwargs) -> pd.Series:
    """
    Rogers-Satchell volatility estimator.
    sigma^2 = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    """
    if high is None or low is None or open_ is None:
        return vol_realized_20d(prices)
    log_hc = np.log(high / prices.replace(0.0, float("nan")))
    log_ho = np.log(high / open_.replace(0.0, float("nan")))
    log_lc = np.log(low / prices.replace(0.0, float("nan")))
    log_lo = np.log(low / open_.replace(0.0, float("nan")))
    rs = (log_hc * log_ho + log_lc * log_lo).rolling(window, min_periods=window // 2).mean()
    return np.sqrt(rs.clip(0) * 252)


def vol_arch_signal(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    ARCH effect (volatility clustering) signal.
    Ljung-Box-like: correlation of squared returns over window.
    High clustering -> high value.
    """
    lr = _log_returns(prices)
    sq = lr ** 2
    # Use autocorrelation of squared returns at lag 1 as clustering measure
    sq_mean = sq.rolling(window, min_periods=window // 2).mean()
    sq_var = sq.rolling(window, min_periods=window // 2).var()
    sq_lag = sq.shift(1)
    cov = ((sq - sq_mean) * (sq_lag - sq_lag.rolling(window, min_periods=window // 2).mean())).rolling(window, min_periods=window // 2).mean()
    autocorr = cov / sq_var.replace(0.0, float("nan"))
    return autocorr.clip(-1, 1)


def vol_normalized_range(prices: pd.Series, volume: pd.Series = None, window: int = 14, **kwargs) -> pd.Series:
    """Normalized range: (rolling_high - rolling_low) / close."""
    high = prices.rolling(window, min_periods=window // 2).max()
    low = prices.rolling(window, min_periods=window // 2).min()
    return (high - low) / prices.replace(0.0, float("nan"))


def vol_hist_vs_implied(prices: pd.Series, volume: pd.Series = None, window_short: int = 5, window_long: int = 20, **kwargs) -> pd.Series:
    """
    Historical vs implied vol spread proxy.
    Uses ratio of short-term to long-term ATR (normalized range) as proxy.
    """
    atr_short = vol_realized_5d(prices)
    atr_long = vol_realized_20d(prices)
    return atr_short / atr_long.replace(0.0, float("nan")) - 1.0


# ===========================================================================
# MICROSTRUCTURE SIGNALS (15)
# ===========================================================================


def ms_volume_surprise(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """Volume surprise: volume / 20d average volume - 1."""
    if volume is None:
        return pd.Series(0.0, index=prices.index)
    avg_vol = volume.rolling(window, min_periods=window // 2).mean()
    return volume / avg_vol.replace(0.0, float("nan")) - 1.0


def ms_vpt(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """Volume Price Trend (VPT): cumulative sum of volume * pct_change."""
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    pct = prices.pct_change().fillna(0.0)
    vpt = (volume * pct).cumsum()
    # Normalize
    std = vpt.rolling(20, min_periods=5).std()
    mean = vpt.rolling(20, min_periods=5).mean()
    return (vpt - mean) / std.replace(0.0, float("nan"))


def ms_obv_normalized(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """On-Balance Volume normalized by rolling std."""
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    direction = np.sign(prices.diff().fillna(0.0))
    obv = (direction * volume).cumsum()
    std = obv.rolling(window, min_periods=window // 2).std()
    mean = obv.rolling(window, min_periods=window // 2).mean()
    return (obv - mean) / std.replace(0.0, float("nan"))


def ms_cmf(prices: pd.Series, volume: pd.Series = None, window: int = 20,
           high: pd.Series = None, low: pd.Series = None, **kwargs) -> pd.Series:
    """
    Chaikin Money Flow.
    CMF = sum(MFV) / sum(volume) over window.
    MFV = ((close - low) - (high - close)) / (high - low) * volume.
    """
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    if high is None:
        high = prices.rolling(2, min_periods=1).max()
    if low is None:
        low = prices.rolling(2, min_periods=1).min()
    hl_range = (high - low).replace(0.0, float("nan"))
    mf_multiplier = ((prices - low) - (high - prices)) / hl_range
    mfv = mf_multiplier * volume
    cmf = mfv.rolling(window, min_periods=window // 2).sum() / volume.rolling(window, min_periods=window // 2).sum().replace(0.0, float("nan"))
    return cmf.clip(-1, 1)


def ms_adl(prices: pd.Series, volume: pd.Series = None,
           high: pd.Series = None, low: pd.Series = None, **kwargs) -> pd.Series:
    """
    Accumulation/Distribution Line.
    ADL = cumsum(CLV * volume). CLV = ((close - low) - (high - close)) / (high - low).
    """
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    if high is None:
        high = prices.rolling(2, min_periods=1).max()
    if low is None:
        low = prices.rolling(2, min_periods=1).min()
    hl_range = (high - low).replace(0.0, float("nan"))
    clv = ((prices - low) - (high - prices)) / hl_range
    adl = (clv * volume).cumsum()
    # Normalize
    adl_mean = adl.rolling(20, min_periods=5).mean()
    adl_std = adl.rolling(20, min_periods=5).std()
    return (adl - adl_mean) / adl_std.replace(0.0, float("nan"))


def ms_force_index(prices: pd.Series, volume: pd.Series = None, window: int = 13, **kwargs) -> pd.Series:
    """Force Index = EMA(price_change * volume)."""
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    fi = prices.diff() * volume
    fi_ema = _ema(fi, window)
    std = fi_ema.rolling(20, min_periods=5).std()
    return fi_ema / std.replace(0.0, float("nan"))


def ms_emv(prices: pd.Series, volume: pd.Series = None, window: int = 14,
           high: pd.Series = None, low: pd.Series = None, **kwargs) -> pd.Series:
    """
    Ease of Movement.
    EMV = ((mid - prev_mid) / (volume / (high - low))).
    """
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    if high is None:
        high = prices.rolling(2, min_periods=1).max()
    if low is None:
        low = prices.rolling(2, min_periods=1).min()
    mid = (high + low) / 2.0
    mid_move = mid - mid.shift(1)
    box_ratio = volume / ((high - low).replace(0.0, float("nan")) * 10000)
    emv = mid_move / box_ratio.replace(0.0, float("nan"))
    emv_ma = emv.rolling(window, min_periods=window // 2).mean()
    std = emv_ma.rolling(20, min_periods=5).std()
    return emv_ma / std.replace(0.0, float("nan"))


def ms_volume_oscillator(prices: pd.Series, volume: pd.Series = None, fast: int = 5, slow: int = 20, **kwargs) -> pd.Series:
    """Volume oscillator: (fast_ema_vol - slow_ema_vol) / slow_ema_vol."""
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    fast_ema = _ema(volume, fast)
    slow_ema = _ema(volume, slow)
    return (fast_ema - slow_ema) / slow_ema.replace(0.0, float("nan"))


def ms_mfi(prices: pd.Series, volume: pd.Series = None, window: int = 14,
           high: pd.Series = None, low: pd.Series = None, **kwargs) -> pd.Series:
    """
    Money Flow Index.
    MFI = 100 - 100/(1 + money_flow_ratio).
    Bounded [0, 100].
    """
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    if high is None:
        high = prices
    if low is None:
        low = prices
    tp = (high + low + prices) / 3.0
    mf = tp * volume
    direction = tp.diff()
    pos_mf = mf.where(direction > 0, 0.0).rolling(window, min_periods=window // 2).sum()
    neg_mf = mf.where(direction <= 0, 0.0).rolling(window, min_periods=window // 2).sum()
    mfr = pos_mf / neg_mf.replace(0.0, float("nan"))
    mfi = 100.0 - 100.0 / (1.0 + mfr)
    return (mfi - 50.0) / 50.0


def ms_nvi(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """
    Negative Volume Index.
    Cumulative price return only on days where volume < previous day's volume.
    """
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    lr = _log_returns(prices).fillna(0.0)
    vol_down = volume < volume.shift(1)
    nvi = lr.where(vol_down, 0.0).cumsum()
    nvi_mean = nvi.rolling(20, min_periods=5).mean()
    nvi_std = nvi.rolling(20, min_periods=5).std()
    return (nvi - nvi_mean) / nvi_std.replace(0.0, float("nan"))


def ms_pvi(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """
    Positive Volume Index.
    Cumulative price return only on days where volume > previous day's volume.
    """
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    lr = _log_returns(prices).fillna(0.0)
    vol_up = volume > volume.shift(1)
    pvi = lr.where(vol_up, 0.0).cumsum()
    pvi_mean = pvi.rolling(20, min_periods=5).mean()
    pvi_std = pvi.rolling(20, min_periods=5).std()
    return (pvi - pvi_mean) / pvi_std.replace(0.0, float("nan"))


def ms_pvt(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """
    Price Volume Trend (PVT): like OBV but uses pct_change not sign.
    PVT += volume * (close - prev_close) / prev_close.
    """
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    pct = prices.pct_change().fillna(0.0)
    pvt = (volume * pct).cumsum()
    pvt_mean = pvt.rolling(20, min_periods=5).mean()
    pvt_std = pvt.rolling(20, min_periods=5).std()
    return (pvt - pvt_mean) / pvt_std.replace(0.0, float("nan"))


def ms_volume_momentum(prices: pd.Series, volume: pd.Series = None, window: int = 10, **kwargs) -> pd.Series:
    """Volume momentum: volume / volume.shift(window) - 1."""
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    return volume / volume.shift(window).replace(0.0, float("nan")) - 1.0


def ms_large_trade(prices: pd.Series, volume: pd.Series = None, window: int = 20, n_std: float = 2.0, **kwargs) -> pd.Series:
    """
    Large trade indicator: volume spike detection.
    +1 if volume > mean + n_std * std and price went up.
    -1 if volume > mean + n_std * std and price went down.
    0 otherwise.
    """
    if volume is None:
        return pd.Series(0.0, index=prices.index)
    mean_vol = volume.rolling(window, min_periods=window // 2).mean()
    std_vol = volume.rolling(window, min_periods=window // 2).std()
    spike = volume > (mean_vol + n_std * std_vol)
    direction = np.sign(prices.diff())
    return (spike.astype(float) * direction).fillna(0.0)


def ms_kyle_lambda(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Kyle's lambda proxy: |return| / sqrt(volume).
    Measures price impact per unit of volume flow.
    Normalized by rolling mean.
    """
    if volume is None:
        volume = pd.Series(1.0, index=prices.index)
    lr = _log_returns(prices).abs()
    sqrt_vol = np.sqrt(volume.replace(0.0, float("nan")))
    lam = lr / sqrt_vol
    lam_mean = lam.rolling(window, min_periods=window // 2).mean()
    lam_std = lam.rolling(window, min_periods=window // 2).std()
    return (lam - lam_mean) / lam_std.replace(0.0, float("nan"))


# ===========================================================================
# PHYSICS / SRFM-SPECIFIC SIGNALS (15)
# ===========================================================================


def phys_bh_mass(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    BH mass signal proxy.
    Mass accumulation = cumulative absolute curvature of log price path.
    Normalized by rolling mean.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    d1 = lp.diff()
    d2 = d1.diff()
    mass_acc = d2.abs().rolling(window, min_periods=window // 2).sum()
    mean_m = mass_acc.rolling(window * 5, min_periods=window).mean()
    std_m = mass_acc.rolling(window * 5, min_periods=window).std()
    return (mass_acc - mean_m) / std_m.replace(0.0, float("nan"))


def phys_proper_time(prices: pd.Series, volume: pd.Series = None, **kwargs) -> pd.Series:
    """
    Proper time accumulation.
    ds^2 = dt^2 - (dx/dt)^2. Proper time increment = sqrt(max(ds^2, 0)).
    Cumulative proper time normalized.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    dx_dt = lp.diff()
    ds2 = 1.0 - dx_dt ** 2  # simplified metric: g=diag(1, -1)
    d_tau = np.sqrt(ds2.clip(0.0))
    tau = d_tau.cumsum()
    tau_mean = tau.rolling(60, min_periods=10).mean()
    tau_std = tau.rolling(60, min_periods=10).std()
    return (tau - tau_mean) / tau_std.replace(0.0, float("nan"))


def phys_timelike_fraction(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Timelike bar fraction: fraction of bars where ds^2 > 0 in rolling window.
    Timelike bar: |dx/dt| < 1 (speed of light = 1 in natural units).
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    dx_dt = lp.diff().abs()
    # Use normalized dx_dt: standardize to make "c=1" meaningful
    norm_dx = dx_dt / dx_dt.rolling(252, min_periods=60).mean().replace(0.0, float("nan"))
    timelike = (norm_dx < 1.0).astype(float)
    return timelike.rolling(window, min_periods=window // 2).mean()


def phys_ds2_trend(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    ds^2 trend: is the spacetime interval increasing or decreasing?
    Positive = interval expanding (trending market).
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    dx_dt = lp.diff()
    ds2 = 1.0 - dx_dt ** 2
    ds2_sma = ds2.rolling(window, min_periods=window // 2).mean()
    ds2_trend = ds2_sma - ds2_sma.shift(window)
    return ds2_trend


def phys_bh_formation_rate(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    BH formation rate: rate of mass accumulation.
    d(mass)/dt approximated by rolling change in curvature.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    curvature = lp.diff().diff().abs()
    mass = curvature.rolling(window, min_periods=window // 2).sum()
    rate = mass - mass.shift(window)
    std = rate.rolling(window * 3, min_periods=window).std()
    return rate / std.replace(0.0, float("nan"))


def phys_geodesic_deviation(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Geodesic deviation proxy.
    Measures divergence of nearby price paths (Jacobi field proxy).
    Uses rolling std of differences between close and linear interpolation.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    # Geodesic = linear path; deviation = residual from rolling linear regression
    residuals = pd.Series(index=prices.index, dtype=float)
    for i in range(window, len(lp) + 1):
        chunk = lp.iloc[i - window : i].dropna()
        if len(chunk) < window // 2:
            continue
        x = np.arange(len(chunk), dtype=float)
        slope, intercept, _, _, _ = stats.linregress(x, chunk.values)
        geodesic = slope * x + intercept
        deviation = np.std(chunk.values - geodesic)
        residuals.iloc[i - 1] = deviation
    mean_r = residuals.rolling(window * 3, min_periods=window).mean()
    std_r = residuals.rolling(window * 3, min_periods=window).std()
    return (residuals - mean_r) / std_r.replace(0.0, float("nan"))


def phys_angular_velocity(prices: pd.Series, volume: pd.Series = None, window: int = 10, **kwargs) -> pd.Series:
    """
    Angular velocity proxy (from quaternion nav analogy).
    Approximated as the rate of change of the direction of price movement.
    omega = d(theta)/dt where theta = atan2(price_diff, time_diff).
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    d1 = lp.diff()
    d2 = d1.diff()
    # Approximate angular velocity: |d2| / (1 + d1^2)^(3/2) (curvature formula)
    curvature = d2.abs() / (1.0 + d1 ** 2) ** 1.5
    return curvature.rolling(window, min_periods=window // 2).mean()


def phys_hurst_signal(prices: pd.Series, volume: pd.Series = None, hurst_window: int = 100, **kwargs) -> pd.Series:
    """
    Hurst exponent signal.
    H > 0.5 -> trending (+1), H < 0.5 -> mean-reverting (-1).
    Computed in rolling batches for efficiency.
    """
    hurst_vals = pd.Series(index=prices.index, dtype=float)
    step = max(1, hurst_window // 10)
    for i in range(hurst_window, len(prices) + 1, step):
        chunk = prices.iloc[max(0, i - hurst_window) : i]
        h = _hurst_exponent(chunk)
        if math.isfinite(h):
            hurst_vals.iloc[i - 1] = h

    hurst_vals = hurst_vals.ffill()
    return (hurst_vals - 0.5) * 2.0  # map [0,1] to [-1, 1] centered at 0.5


def phys_fractal_dimension(prices: pd.Series, volume: pd.Series = None, window: int = 50, **kwargs) -> pd.Series:
    """
    Fractal dimension signal.
    D = 2 - H (Hurst). D ~ 1.5 = random walk; D < 1.5 = trending.
    """
    hurst_sig = phys_hurst_signal(prices, hurst_window=window)
    # Convert back to D = 2 - H
    H = hurst_sig / 2.0 + 0.5  # undo the mapping
    D = 2.0 - H
    return D


def phys_hawking_temperature(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Hawking temperature proxy.
    T_H = hbar * c^3 / (8 * pi * G * M).
    As mass (curvature) increases, T_H decreases.
    Uses: T_proxy = 1 / (mass_proxy + epsilon).
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    curvature = lp.diff().diff().abs()
    mass = curvature.rolling(window, min_periods=window // 2).sum() + 1e-8
    temp = 1.0 / mass
    temp_mean = temp.rolling(window * 5, min_periods=window).mean()
    temp_std = temp.rolling(window * 5, min_periods=window).std()
    return (temp - temp_mean) / temp_std.replace(0.0, float("nan"))


def phys_grav_lensing(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Gravitational lensing signal: mass * direction.
    High mass (curvature) amplifies the directional signal.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    d1 = lp.diff()
    curvature = d1.diff().abs()
    mass = curvature.rolling(window, min_periods=window // 2).sum()
    direction = d1.rolling(window, min_periods=window // 2).mean()
    lensed = mass * direction
    std = lensed.rolling(window * 5, min_periods=window).std()
    return lensed / std.replace(0.0, float("nan"))


def phys_phase_transition(prices: pd.Series, volume: pd.Series = None, mass_threshold_pct: float = 0.9, window: int = 60, **kwargs) -> pd.Series:
    """
    Phase transition detector: mass crossing a percentile threshold.
    Returns +1 just after mass crosses threshold (regime change), 0 otherwise.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    curvature = lp.diff().diff().abs()
    mass = curvature.rolling(20, min_periods=5).sum()
    threshold = mass.rolling(window, min_periods=window // 4).quantile(mass_threshold_pct)
    above = (mass > threshold).astype(float)
    transition = above.diff().abs()  # 1 at transition points
    return transition.rolling(5, min_periods=1).mean()  # smooth slightly


def phys_causal_info_ratio(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Causal information ratio: fraction of timelike bars over window.
    Same as timelike_fraction but returned as-is (maps to [0, 1]).
    """
    return phys_timelike_fraction(prices, volume=volume, window=window)


def phys_regime_velocity(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Regime velocity: rate of change of the Hurst exponent.
    High velocity = rapid regime transitions.
    """
    hurst_sig = phys_hurst_signal(prices, hurst_window=max(50, window))
    H = hurst_sig / 2.0 + 0.5
    H_smooth = H.rolling(window, min_periods=window // 2).mean()
    velocity = H_smooth.diff()
    std = velocity.rolling(window * 3, min_periods=window).std()
    return velocity / std.replace(0.0, float("nan"))


def phys_curvature_proxy(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Spacetime curvature proxy.
    Ricci scalar proxy = average of squared second derivatives of log price.
    """
    lp = np.log(prices.replace(0.0, float("nan")))
    d2 = lp.diff().diff()
    ricci = d2 ** 2
    ricci_mean = ricci.rolling(window, min_periods=window // 2).mean()
    ricci_long = ricci.rolling(window * 10, min_periods=window).mean()
    ricci_std = ricci.rolling(window * 10, min_periods=window).std()
    return (ricci_mean - ricci_long) / ricci_std.replace(0.0, float("nan"))


# ===========================================================================
# TECHNICAL SIGNALS (15)
# ===========================================================================


def tech_macd_histogram(prices: pd.Series, volume: pd.Series = None, fast: int = 12, slow: int = 26, signal_span: int = 9, **kwargs) -> pd.Series:
    """MACD histogram: MACD line - signal line."""
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal_span)
    histogram = macd_line - signal_line
    std = histogram.rolling(60, min_periods=20).std()
    return histogram / std.replace(0.0, float("nan"))


def tech_adx(prices: pd.Series, volume: pd.Series = None, window: int = 14,
             high: pd.Series = None, low: pd.Series = None, **kwargs) -> pd.Series:
    """
    ADX trend strength (0-100). Positive = trending.
    Uses ATR and directional movement.
    """
    if high is None:
        high = prices
    if low is None:
        low = prices

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = prices.shift(1)

    dm_pos = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0).clip(0)
    dm_neg = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0).clip(0)

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(span=window, adjust=False).mean()

    di_pos = 100 * _ema(dm_pos, window) / atr_w.replace(0.0, float("nan"))
    di_neg = 100 * _ema(dm_neg, window) / atr_w.replace(0.0, float("nan"))

    dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0.0, float("nan"))
    adx = _ema(dx, window)
    return adx / 100.0


def tech_aroon_oscillator(prices: pd.Series, volume: pd.Series = None, window: int = 25, **kwargs) -> pd.Series:
    """
    Aroon Oscillator: Aroon Up - Aroon Down. Bounded [-100, 100].
    """
    aroon_up = prices.rolling(window + 1, min_periods=window // 2).apply(
        lambda x: (np.argmax(x) / (window)) * 100, raw=True
    )
    aroon_down = prices.rolling(window + 1, min_periods=window // 2).apply(
        lambda x: (np.argmin(x) / (window)) * 100, raw=True
    )
    return (aroon_up - aroon_down) / 100.0


def tech_cci(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Commodity Channel Index (duplicate in Technical for completeness).
    See mr_cci for formula. Returns normalized version.
    """
    return mr_cci(prices, window=window)


def tech_keltner_position(prices: pd.Series, volume: pd.Series = None, window: int = 20, atr_mult: float = 2.0, **kwargs) -> pd.Series:
    """
    Keltner Channel position.
    (price - midline) / (atr_mult * ATR).
    """
    ema_mid = _ema(prices, window)
    lr = _log_returns(prices).abs()
    atr_proxy = lr.ewm(span=window, adjust=False).mean() * prices
    upper = ema_mid + atr_mult * atr_proxy
    lower = ema_mid - atr_mult * atr_proxy
    band = upper - lower
    pos = (prices - ema_mid) / band.replace(0.0, float("nan")) * 2.0
    return pos.clip(-2, 2)


def tech_donchian_breakout(prices: pd.Series, volume: pd.Series = None, window: int = 20, **kwargs) -> pd.Series:
    """
    Donchian channel breakout.
    +1 if price >= window high, -1 if price <= window low, 0 otherwise.
    """
    high = prices.rolling(window, min_periods=window // 2).max()
    low = prices.rolling(window, min_periods=window // 2).min()
    sig = pd.Series(0.0, index=prices.index)
    sig[prices >= high] = 1.0
    sig[prices <= low] = -1.0
    return sig


def tech_ichimoku_cloud(prices: pd.Series, volume: pd.Series = None,
                        tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, **kwargs) -> pd.Series:
    """
    Ichimoku cloud position.
    Signal = (price - cloud_midpoint) / cloud_bandwidth.
    """
    def mid_line(s: pd.Series, w: int) -> pd.Series:
        return (s.rolling(w, min_periods=w // 2).max() + s.rolling(w, min_periods=w // 2).min()) / 2.0

    tenkan_sen = mid_line(prices, tenkan)
    kijun_sen = mid_line(prices, kijun)
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    senkou_b_line = mid_line(prices, senkou_b).shift(kijun)
    cloud_top = pd.concat([senkou_a, senkou_b_line], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b_line], axis=1).min(axis=1)
    cloud_mid = (cloud_top + cloud_bot) / 2.0
    bandwidth = (cloud_top - cloud_bot).replace(0.0, float("nan"))
    return ((prices - cloud_mid) / bandwidth).clip(-3, 3)


def tech_psar(prices: pd.Series, volume: pd.Series = None, af_start: float = 0.02, af_max: float = 0.2, **kwargs) -> pd.Series:
    """
    Parabolic SAR signal.
    +1 if price > PSAR (bullish), -1 if price < PSAR (bearish).
    """
    psar = prices.copy().astype(float)
    bull = True
    af = af_start
    ep = float(prices.iloc[0])
    sar = float(prices.iloc[0])

    signals = np.zeros(len(prices))
    p = prices.values.astype(float)

    for i in range(1, len(p)):
        prev_sar = sar
        if bull:
            sar = prev_sar + af * (ep - prev_sar)
            sar = min(sar, p[i - 1], p[max(i - 2, 0)])
            if p[i] < sar:
                bull = False
                sar = ep
                ep = p[i]
                af = af_start
            else:
                if p[i] > ep:
                    ep = p[i]
                    af = min(af + af_start, af_max)
        else:
            sar = prev_sar + af * (ep - prev_sar)
            sar = max(sar, p[i - 1], p[max(i - 2, 0)])
            if p[i] > sar:
                bull = True
                sar = ep
                ep = p[i]
                af = af_start
            else:
                if p[i] < ep:
                    ep = p[i]
                    af = min(af + af_start, af_max)
        signals[i] = 1.0 if bull else -1.0

    return pd.Series(signals, index=prices.index, dtype=float)


def tech_elder_ray(prices: pd.Series, volume: pd.Series = None, window: int = 13,
                   high: pd.Series = None, low: pd.Series = None, **kwargs) -> pd.Series:
    """
    Elder Ray: Bull Power = High - EMA; Bear Power = Low - EMA.
    Signal = (Bull Power + Bear Power) normalized by ATR.
    """
    ema = _ema(prices, window)
    h = high if high is not None else prices
    l = low if low is not None else prices
    bull_power = h - ema
    bear_power = l - ema
    net = bull_power + bear_power
    lr = _log_returns(prices).abs()
    atr_proxy = lr.ewm(span=window, adjust=False).mean() * prices
    return net / atr_proxy.replace(0.0, float("nan"))


def tech_vortex(prices: pd.Series, volume: pd.Series = None, window: int = 14,
                high: pd.Series = None, low: pd.Series = None, **kwargs) -> pd.Series:
    """
    Vortex Indicator: VI+ - VI-.
    VI+ = sum(|High - prev_Low|) / ATR_sum.
    VI- = sum(|Low - prev_High|) / ATR_sum.
    """
    h = high if high is not None else prices
    l = low if low is not None else prices
    prev_close = prices.shift(1)
    prev_high = h.shift(1)
    prev_low = l.shift(1)

    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr_sum = tr.rolling(window, min_periods=window // 2).sum()

    vm_pos = (h - prev_low).abs().rolling(window, min_periods=window // 2).sum()
    vm_neg = (l - prev_high).abs().rolling(window, min_periods=window // 2).sum()

    vi_pos = vm_pos / atr_sum.replace(0.0, float("nan"))
    vi_neg = vm_neg / atr_sum.replace(0.0, float("nan"))
    return (vi_pos - vi_neg).clip(-2, 2)


def tech_chande_kroll(prices: pd.Series, volume: pd.Series = None, p: int = 10, q: int = 9, x: int = 3, **kwargs) -> pd.Series:
    """
    Chande Kroll Stop.
    Signal = (price - stop_level) / price.
    """
    lr = _log_returns(prices).abs()
    atr = lr.ewm(span=p, adjust=False).mean() * prices

    first_high = prices.rolling(p, min_periods=p // 2).max() - x * atr
    first_low = prices.rolling(p, min_periods=p // 2).min() + x * atr
    stop_short = first_high.rolling(q, min_periods=q // 2).max()
    stop_long = first_low.rolling(q, min_periods=q // 2).min()

    # Signal: how far price is above/below the stop
    signal = (prices - (stop_short + stop_long) / 2.0) / prices.replace(0.0, float("nan"))
    return signal.clip(-1, 1)


def tech_supertrend(prices: pd.Series, volume: pd.Series = None, window: int = 10, multiplier: float = 3.0,
                    high: pd.Series = None, low: pd.Series = None, **kwargs) -> pd.Series:
    """
    SuperTrend indicator.
    Returns +1 if price above SuperTrend line, -1 if below.
    """
    h = high if high is not None else prices
    l = low if low is not None else prices
    hl_mid = (h + l) / 2.0
    lr = _log_returns(prices).abs()
    atr = lr.ewm(span=window, adjust=False).mean() * prices

    upper_band = hl_mid + multiplier * atr
    lower_band = hl_mid - multiplier * atr

    supertrend = pd.Series(index=prices.index, dtype=float)
    direction = pd.Series(index=prices.index, dtype=float)

    p_arr = prices.values.astype(float)
    ub = upper_band.values.astype(float)
    lb = lower_band.values.astype(float)
    st = np.full(len(p_arr), float("nan"))
    d = np.ones(len(p_arr))

    for i in range(1, len(p_arr)):
        if not (np.isfinite(ub[i]) and np.isfinite(lb[i])):
            st[i] = st[i - 1] if i > 0 else float("nan")
            d[i] = d[i - 1]
            continue

        if d[i - 1] == 1:
            st[i] = max(lb[i], st[i - 1]) if np.isfinite(st[i - 1]) else lb[i]
            if p_arr[i] < st[i]:
                d[i] = -1
                st[i] = ub[i]
            else:
                d[i] = 1
        else:
            st[i] = min(ub[i], st[i - 1]) if np.isfinite(st[i - 1]) else ub[i]
            if p_arr[i] > st[i]:
                d[i] = 1
                st[i] = lb[i]
            else:
                d[i] = -1

    return pd.Series(d, index=prices.index, dtype=float)


def tech_trix(prices: pd.Series, volume: pd.Series = None, window: int = 15, **kwargs) -> pd.Series:
    """
    Triple Exponential (TRIX): 1-period ROC of triple EMA.
    Trend-following oscillator.
    """
    ema1 = _ema(prices, window)
    ema2 = _ema(ema1, window)
    ema3 = _ema(ema2, window)
    trix = ema3.pct_change() * 100.0
    std = trix.rolling(60, min_periods=20).std()
    return trix / std.replace(0.0, float("nan"))


def tech_mass_index(prices: pd.Series, volume: pd.Series = None,
                    high: pd.Series = None, low: pd.Series = None,
                    fast: int = 9, slow: int = 25, **kwargs) -> pd.Series:
    """
    Mass Index.
    MI = sum(EMA9(H-L) / EMA9(EMA9(H-L))).
    Identifies trend reversals when MI crosses 27 then falls below 26.5.
    """
    h = high if high is not None else prices
    l = low if low is not None else prices
    hl_range = h - l
    ema1 = _ema(hl_range, fast)
    ema2 = _ema(ema1, fast)
    ratio = ema1 / ema2.replace(0.0, float("nan"))
    mi = ratio.rolling(slow, min_periods=slow // 2).sum()
    mi_mean = mi.rolling(60, min_periods=20).mean()
    mi_std = mi.rolling(60, min_periods=20).std()
    return (mi - mi_mean) / mi_std.replace(0.0, float("nan"))


def tech_ulcer_index(prices: pd.Series, volume: pd.Series = None, window: int = 14, **kwargs) -> pd.Series:
    """
    Ulcer Index: measures downside risk/volatility.
    UI = sqrt(mean(R^2)) where R = (price - rolling_max) / rolling_max * 100.
    Lower values are generally better. Normalized as signal.
    """
    rolling_max = prices.rolling(window, min_periods=window // 2).max()
    drawdown_pct = (prices - rolling_max) / rolling_max.replace(0.0, float("nan")) * 100.0
    ui = np.sqrt((drawdown_pct ** 2).rolling(window, min_periods=window // 2).mean())
    # Negative signal: high ulcer = high drawdown risk = bearish
    ui_mean = ui.rolling(60, min_periods=20).mean()
    ui_std = ui.rolling(60, min_periods=20).std()
    return -((ui - ui_mean) / ui_std.replace(0.0, float("nan")))


# ===========================================================================
# Signal registry: maps name -> function
# ===========================================================================

SIGNAL_REGISTRY = {
    # Momentum
    "mom_1d": mom_1d,
    "mom_5d": mom_5d,
    "mom_20d": mom_20d,
    "mom_60d": mom_60d,
    "mom_252d": mom_252d,
    "mom_sharpe": mom_sharpe,
    "mom_acceleration": mom_acceleration,
    "mom_52w_high": mom_52w_high,
    "mom_crash_protection": mom_crash_protection,
    "mom_ts_moskowitz": mom_ts_moskowitz,
    "mom_cs_rank": mom_cs_rank,
    "mom_seasonality": mom_seasonality,
    "mom_dual": mom_dual,
    "mom_absolute": mom_absolute,
    "mom_intermediate": mom_intermediate,
    "mom_short_reversal": mom_short_reversal,
    "mom_end_of_month": mom_end_of_month,
    "mom_gap": mom_gap,
    "mom_volume_weighted": mom_volume_weighted,
    "mom_up_down_volume": mom_up_down_volume,
    "mom_tick_proxy": mom_tick_proxy,
    "mom_price_accel": mom_price_accel,
    "mom_multi_tf_composite": mom_multi_tf_composite,
    # Mean Reversion
    "mr_zscore_10": mr_zscore_10,
    "mr_zscore_20": mr_zscore_20,
    "mr_zscore_50": mr_zscore_50,
    "mr_bollinger_position": mr_bollinger_position,
    "mr_rsi": mr_rsi,
    "mr_linreg_residual": mr_linreg_residual,
    "mr_sma_deviation": mr_sma_deviation,
    "mr_kalman_residual": mr_kalman_residual,
    "mr_pairs_ratio_zscore": mr_pairs_ratio_zscore,
    "mr_ou_weighted": mr_ou_weighted,
    "mr_vwap_deviation": mr_vwap_deviation,
    "mr_price_oscillator": mr_price_oscillator,
    "mr_dpo": mr_dpo,
    "mr_cci": mr_cci,
    "mr_williams_r": mr_williams_r,
    "mr_stochastic_k": mr_stochastic_k,
    "mr_chande_momentum": mr_chande_momentum,
    "mr_roc_mean_rev": mr_roc_mean_rev,
    "mr_price_channel": mr_price_channel,
    "mr_log_autoregression": mr_log_autoregression,
    "mr_hurst_adjusted": mr_hurst_adjusted,
    # Volatility
    "vol_ewma_forecast": vol_ewma_forecast,
    "vol_realized_5d": vol_realized_5d,
    "vol_realized_20d": vol_realized_20d,
    "vol_realized_60d": vol_realized_60d,
    "vol_of_vol": vol_of_vol,
    "vol_regime": vol_regime,
    "vol_atr_percentile": vol_atr_percentile,
    "vol_skew_proxy": vol_skew_proxy,
    "vol_term_structure": vol_term_structure,
    "vol_parkinson": vol_parkinson,
    "vol_garman_klass": vol_garman_klass,
    "vol_yang_zhang": vol_yang_zhang,
    "vol_rogers_satchell": vol_rogers_satchell,
    "vol_arch_signal": vol_arch_signal,
    "vol_normalized_range": vol_normalized_range,
    "vol_hist_vs_implied": vol_hist_vs_implied,
    # Microstructure
    "ms_volume_surprise": ms_volume_surprise,
    "ms_vpt": ms_vpt,
    "ms_obv_normalized": ms_obv_normalized,
    "ms_cmf": ms_cmf,
    "ms_adl": ms_adl,
    "ms_force_index": ms_force_index,
    "ms_emv": ms_emv,
    "ms_volume_oscillator": ms_volume_oscillator,
    "ms_mfi": ms_mfi,
    "ms_nvi": ms_nvi,
    "ms_pvi": ms_pvi,
    "ms_pvt": ms_pvt,
    "ms_volume_momentum": ms_volume_momentum,
    "ms_large_trade": ms_large_trade,
    "ms_kyle_lambda": ms_kyle_lambda,
    # Physics
    "phys_bh_mass": phys_bh_mass,
    "phys_proper_time": phys_proper_time,
    "phys_timelike_fraction": phys_timelike_fraction,
    "phys_ds2_trend": phys_ds2_trend,
    "phys_bh_formation_rate": phys_bh_formation_rate,
    "phys_geodesic_deviation": phys_geodesic_deviation,
    "phys_angular_velocity": phys_angular_velocity,
    "phys_hurst_signal": phys_hurst_signal,
    "phys_fractal_dimension": phys_fractal_dimension,
    "phys_hawking_temperature": phys_hawking_temperature,
    "phys_grav_lensing": phys_grav_lensing,
    "phys_phase_transition": phys_phase_transition,
    "phys_causal_info_ratio": phys_causal_info_ratio,
    "phys_regime_velocity": phys_regime_velocity,
    "phys_curvature_proxy": phys_curvature_proxy,
    # Technical
    "tech_macd_histogram": tech_macd_histogram,
    "tech_adx": tech_adx,
    "tech_aroon_oscillator": tech_aroon_oscillator,
    "tech_cci": tech_cci,
    "tech_keltner_position": tech_keltner_position,
    "tech_donchian_breakout": tech_donchian_breakout,
    "tech_ichimoku_cloud": tech_ichimoku_cloud,
    "tech_psar": tech_psar,
    "tech_elder_ray": tech_elder_ray,
    "tech_vortex": tech_vortex,
    "tech_chande_kroll": tech_chande_kroll,
    "tech_supertrend": tech_supertrend,
    "tech_trix": tech_trix,
    "tech_mass_index": tech_mass_index,
    "tech_ulcer_index": tech_ulcer_index,
}

# Category mapping
SIGNAL_CATEGORIES = {
    "momentum": [k for k in SIGNAL_REGISTRY if k.startswith("mom_")],
    "mean_reversion": [k for k in SIGNAL_REGISTRY if k.startswith("mr_")],
    "volatility": [k for k in SIGNAL_REGISTRY if k.startswith("vol_")],
    "microstructure": [k for k in SIGNAL_REGISTRY if k.startswith("ms_")],
    "physics": [k for k in SIGNAL_REGISTRY if k.startswith("phys_")],
    "technical": [k for k in SIGNAL_REGISTRY if k.startswith("tech_")],
}


# ===========================================================================
# Signal utility functions (post-processing and analysis helpers)
# ===========================================================================


def standardize_signal(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score standardize a signal.
    Clips output to [-3, 3] to control extreme values.
    """
    mu = series.rolling(window, min_periods=window // 2).mean()
    sigma = series.rolling(window, min_periods=window // 2).std()
    z = (series - mu) / sigma.replace(0.0, float("nan"))
    return z.clip(-3.0, 3.0)


def rank_signal(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling percentile rank of signal values.
    Returns values in [0, 1].
    """
    return series.rolling(window, min_periods=window // 4).rank(pct=True)


def winsorize_signal(series: pd.Series, limits: float = 0.02) -> pd.Series:
    """
    Winsorize a signal at given quantile limits.
    Applied over the full series (not rolling).
    """
    lo = series.quantile(limits)
    hi = series.quantile(1 - limits)
    return series.clip(lo, hi)


def neutralize_signal(
    signal: pd.Series,
    factor: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Neutralize a signal with respect to a factor (e.g., remove beta exposure).
    Uses rolling OLS: residual of signal ~ factor.
    """
    residuals = pd.Series(index=signal.index, dtype=float)
    combined = pd.concat([signal.rename("sig"), factor.rename("fac")], axis=1).dropna()

    for i in range(window, len(combined) + 1):
        chunk = combined.iloc[i - window : i]
        if len(chunk) < window // 2:
            continue
        x = chunk["fac"].values
        y = chunk["sig"].values
        if np.std(x) < 1e-10:
            residuals.iloc[i - 1] = 0.0
            continue
        slope, intercept, _, _, _ = stats.linregress(x, y)
        residuals.iloc[i - 1] = y[-1] - (slope * x[-1] + intercept)

    return residuals


def smooth_signal(series: pd.Series, method: str = "ema", span: int = 3) -> pd.Series:
    """
    Smooth a signal to reduce noise.

    method: 'ema' | 'sma' | 'savgol'
    """
    if method == "ema":
        return series.ewm(span=span, adjust=False).mean()
    elif method == "sma":
        return series.rolling(span, min_periods=1).mean()
    else:
        # Fallback to EMA
        return series.ewm(span=span, adjust=False).mean()


def lag_signal(series: pd.Series, lags: int = 1) -> pd.Series:
    """Lag a signal by N periods."""
    return series.shift(lags)


def signal_to_weights(
    signal: pd.Series,
    method: str = "long_short",
    cap: float = 0.1,
) -> pd.Series:
    """
    Convert a signal to portfolio weights.

    method:
    - 'long_short': positive signal -> long, negative -> short, normalized
    - 'long_only': only positive signals, normalized
    - 'rank': rank-based weights
    """
    if method == "rank":
        ranked = signal.rank(pct=True) - 0.5
        denom = ranked.abs().sum()
        return ranked / denom if denom > 1e-10 else ranked
    elif method == "long_only":
        pos = signal.clip(0.0, None)
        total = pos.sum()
        return pos / total if total > 1e-10 else pos
    else:  # long_short
        # Z-score then normalize to unit absolute sum
        z = (signal - signal.mean()) / signal.std().replace(0.0, 1.0)
        total = z.abs().sum()
        return (z / total).clip(-cap, cap) if total > 1e-10 else z


def compute_signal_autocorrelation(
    series: pd.Series,
    max_lag: int = 10,
) -> Dict[int, float]:
    """
    Compute autocorrelation of a signal at multiple lags.
    Returns dict of lag -> autocorrelation.
    """
    clean = series.dropna()
    result: Dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        if len(clean) > lag:
            corr = clean.autocorr(lag=lag)
            result[lag] = float(corr) if math.isfinite(corr) else float("nan")
        else:
            result[lag] = float("nan")
    return result


def signal_persistence(series: pd.Series, threshold: float = 0.0) -> float:
    """
    Measure signal persistence: fraction of consecutive periods
    where the signal stays on the same side of threshold.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return float("nan")
    sides = (clean > threshold).astype(int)
    same = (sides == sides.shift(1)).sum()
    return float(same / (len(sides) - 1))


def signal_sharpe_ratio(
    signal: pd.Series,
    prices: pd.Series,
    holding_period: int = 1,
    annualize: bool = True,
) -> float:
    """
    Compute Sharpe ratio of a long-short strategy that follows the signal.
    Long when signal > 0, short when signal < 0.
    """
    log_prices = np.log(prices.replace(0.0, float("nan")))
    fwd = log_prices.shift(-holding_period) - log_prices
    direction = np.sign(signal.reindex(fwd.index))
    pnl = direction * fwd
    clean = pnl.dropna()
    if clean.std() < 1e-10 or len(clean) < 10:
        return float("nan")
    sr = clean.mean() / clean.std()
    if annualize:
        sr *= math.sqrt(252 / holding_period)
    return float(sr)


def cross_sectional_zscore(
    signal_df: pd.DataFrame,
    clip: float = 3.0,
) -> pd.DataFrame:
    """
    Cross-sectional z-score: for each date (row), z-score across all symbols (columns).
    """
    mu = signal_df.mean(axis=1)
    sigma = signal_df.std(axis=1)
    z = signal_df.subtract(mu, axis=0).divide(sigma.replace(0.0, float("nan")), axis=0)
    return z.clip(-clip, clip)


def signal_decay_factor(
    series: pd.Series,
    half_life: float,
) -> pd.Series:
    """
    Apply exponential time decay to a signal.
    Older observations are weighted by exp(-lambda * t).
    Returns EWMA with span derived from half_life.
    """
    lam = math.log(2) / max(half_life, 1e-4)
    # EWMA span s = 2/alpha - 1 where alpha = 1 - exp(-lambda)
    alpha = 1.0 - math.exp(-lam)
    span = max(2.0 / alpha - 1.0, 1.0)
    return series.ewm(span=span, adjust=False).mean()


def combine_signals_by_factor(
    signals: Dict[str, pd.Series],
    weights: Dict[str, float],
) -> pd.Series:
    """
    Weighted combination of multiple signals (as pd.Series).
    Weights need not sum to 1; they will be normalized.
    """
    total_weight = sum(abs(w) for w in weights.values() if math.isfinite(w))
    if total_weight < 1e-10:
        return pd.Series(dtype=float)

    result: Optional[pd.Series] = None
    for name, series in signals.items():
        w = weights.get(name, 0.0)
        if not math.isfinite(w) or abs(w) < 1e-10:
            continue
        z = standardize_signal(series)
        if result is None:
            result = z * (w / total_weight)
        else:
            result = result.add(z * (w / total_weight), fill_value=0.0)

    return result if result is not None else pd.Series(dtype=float)


# ===========================================================================
# OHLCV helper: generate synthetic test data
# ===========================================================================


def generate_synthetic_ohlcv(
    n_bars: int = 500,
    start_price: float = 100.0,
    drift: float = 0.0002,
    vol: float = 0.012,
    seed: int = 42,
) -> Dict[str, pd.Series]:
    """
    Generate synthetic OHLCV data for testing.

    Returns dict with keys: close, open, high, low, volume.
    All series share the same DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")

    log_returns = rng.normal(drift, vol, n_bars)
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    close = pd.Series(np.exp(log_prices), index=idx)

    # Open: close[t-1] plus a small overnight gap
    open_arr = np.exp(log_prices - log_returns + rng.normal(0, vol * 0.3, n_bars))
    open_ = pd.Series(open_arr, index=idx)

    # High / Low: add intraday noise
    intraday_range = np.abs(rng.normal(0, vol * 0.5, n_bars))
    high = pd.Series(np.maximum(close.values, open_.values) * (1 + intraday_range), index=idx)
    low = pd.Series(np.minimum(close.values, open_.values) * (1 - intraday_range), index=idx)

    # Volume: log-normal
    volume = pd.Series(rng.lognormal(mean=15, sigma=0.5, size=n_bars), index=idx)

    return {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
    }


# ===========================================================================
# Batch signal computation
# ===========================================================================


def compute_all_signals(
    prices: pd.Series,
    volume: Optional[pd.Series] = None,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    open_: Optional[pd.Series] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute all signals (or a subset by category) and return as DataFrame.

    Returns DataFrame indexed by prices.index, with one column per signal.
    """
    target_signals = []
    if categories:
        for cat in categories:
            target_signals.extend(SIGNAL_CATEGORIES.get(cat, []))
    else:
        target_signals = list(SIGNAL_REGISTRY.keys())

    results = {}
    for name in target_signals:
        func = SIGNAL_REGISTRY[name]
        try:
            kwargs = {}
            if high is not None:
                kwargs["high"] = high
            if low is not None:
                kwargs["low"] = low
            if open_ is not None:
                kwargs["open_"] = open_
            series = func(prices, volume=volume, **kwargs)
            results[name] = series
        except Exception:
            results[name] = pd.Series(float("nan"), index=prices.index)

    return pd.DataFrame(results, index=prices.index)


def signal_ic_table(
    prices: pd.Series,
    volume: Optional[pd.Series] = None,
    horizons: List[int] = (1, 5, 20),
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute IC for all signals at multiple horizons.

    Returns DataFrame with columns: signal_name, horizon, ic, t_stat, n_obs.
    """
    from research.signal_analytics.alpha_engine import ICCalculator

    calc = ICCalculator(min_obs=20)
    log_prices = np.log(prices.replace(0.0, float("nan")))
    signal_df = compute_all_signals(prices, volume=volume, categories=categories)

    rows = []
    for h in horizons:
        fwd = log_prices.shift(-h) - log_prices
        for col in signal_df.columns:
            sig = signal_df[col].dropna()
            if sig.empty:
                continue
            ic = calc.compute_ic(sig, fwd)
            n = len(sig.dropna().index.intersection(fwd.dropna().index))
            if math.isfinite(ic) and n > 1:
                denom = max(1 - ic ** 2, 1e-10)
                t_stat = ic * math.sqrt(max(n - 2, 0) / denom)
            else:
                t_stat = float("nan")
            rows.append({
                "signal_name": col,
                "horizon": h,
                "ic": ic,
                "t_stat": t_stat,
                "n_obs": n,
            })

    return pd.DataFrame(rows)


# Type alias for function signature
from typing import Callable as _Callable, Dict as _Dict, List as _List  # noqa: E402
SignalFunction = _Callable[..., pd.Series]
