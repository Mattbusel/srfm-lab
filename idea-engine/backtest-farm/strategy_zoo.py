"""
strategy_zoo.py — Complete library of 30+ strategy implementations for the backtest farm.

Every strategy follows the signature:
    strategy(returns: np.ndarray, **params) -> np.ndarray

where `returns` is a 1-D array of period returns and the output is a signal
array of the same length with values in [-1, 1] (position sizing).

Dependencies: numpy, scipy (only).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import welch
from scipy.optimize import minimize_scalar
from scipy.special import factorial as _factorial


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _validate_1d(arr: np.ndarray, name: str = "returns") -> np.ndarray:
    """Ensure *arr* is a finite 1-D float64 array."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Simple rolling mean with NaN padding at the front."""
    out = np.full_like(x, np.nan)
    cs = np.cumsum(x)
    out[w - 1:] = (cs[w - 1:] - np.concatenate([[0.0], cs[:-w]])) / w
    return out


def _rolling_std(x: np.ndarray, w: int, ddof: int = 1) -> np.ndarray:
    """Rolling standard deviation (NaN-padded)."""
    out = np.full_like(x, np.nan)
    for i in range(w - 1, len(x)):
        out[i] = np.std(x[i - w + 1: i + 1], ddof=ddof)
    return out


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling sum (NaN-padded)."""
    out = np.full_like(x, np.nan)
    cs = np.cumsum(x)
    out[w - 1:] = cs[w - 1:] - np.concatenate([[0.0], cs[:-w]])
    return out


def _rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(w - 1, len(x)):
        out[i] = np.max(x[i - w + 1: i + 1])
    return out


def _rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(w - 1, len(x)):
        out[i] = np.min(x[i - w + 1: i + 1])
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (in-place, NaN-safe start)."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _clamp(sig: np.ndarray) -> np.ndarray:
    """Clamp signal to [-1, 1] and replace NaN with 0."""
    sig = np.clip(sig, -1.0, 1.0)
    sig[np.isnan(sig)] = 0.0
    return sig


def _prices_from_returns(returns: np.ndarray) -> np.ndarray:
    """Reconstruct a synthetic price series from returns (start = 1)."""
    return np.cumprod(1.0 + returns)


def _true_range_from_returns(returns: np.ndarray) -> np.ndarray:
    """Approximate true range from return series (|r| * price)."""
    prices = _prices_from_returns(returns)
    tr = np.abs(returns) * prices
    tr[0] = 0.0
    return tr


def _atr(returns: np.ndarray, period: int) -> np.ndarray:
    """Average true range approximation from returns."""
    tr = _true_range_from_returns(returns)
    return _ema(tr, period)


def _permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Permutation entropy of a time series segment."""
    n = len(x)
    n_perms = 0
    perm_counts: dict[tuple, int] = {}
    for i in range(n - (order - 1) * delay):
        motif = tuple(np.argsort(x[i: i + order * delay: delay]))
        perm_counts[motif] = perm_counts.get(motif, 0) + 1
        n_perms += 1
    if n_perms == 0:
        return 0.0
    probs = np.array(list(perm_counts.values()), dtype=np.float64) / n_perms
    return -np.sum(probs * np.log2(probs + 1e-15))


def _hurst_rs(x: np.ndarray) -> float:
    """Rescaled range Hurst exponent estimate."""
    n = len(x)
    if n < 20:
        return 0.5
    max_k = min(n // 2, 200)
    sizes = []
    rs_vals = []
    for k in [int(s) for s in np.logspace(np.log10(10), np.log10(max_k), 8)]:
        if k < 10 or k > n:
            continue
        rs_list = []
        for start in range(0, n - k + 1, k):
            segment = x[start: start + k]
            m = np.mean(segment)
            y = np.cumsum(segment - m)
            r = np.max(y) - np.min(y)
            s = np.std(segment, ddof=1)
            if s > 1e-15:
                rs_list.append(r / s)
        if rs_list:
            sizes.append(k)
            rs_vals.append(np.mean(rs_list))
    if len(sizes) < 2:
        return 0.5
    log_n = np.log(sizes)
    log_rs = np.log(np.array(rs_vals) + 1e-15)
    slope, _, _, _, _ = sp_stats.linregress(log_n, log_rs)
    return float(np.clip(slope, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Strategy 1: momentum_12_1
# ---------------------------------------------------------------------------

def momentum_12_1(
    returns: np.ndarray,
    *,
    lookback: int = 252,
    skip: int = 21,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Classic 12-month momentum with 1-month skip (Jegadeesh & Titman style).

    Computes cumulative return over *lookback* days, skipping the most recent
    *skip* days.  Signal is +1 if past return > threshold, -1 if < -threshold,
    else 0.

    Parameters
    ----------
    returns : 1-D array of period returns.
    lookback : total lookback window in periods (default 252 ~ 12 months).
    skip : number of recent periods to skip (default 21 ~ 1 month).
    threshold : minimum absolute return to generate a signal.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    if lookback < 1 or skip < 0 or lookback <= skip:
        raise ValueError("Require lookback > skip >= 0")
    sig = np.zeros(n, dtype=np.float64)
    cum = np.log1p(returns)
    cum_sum = np.cumsum(cum)

    for i in range(lookback, n):
        end_idx = i - skip
        start_idx = i - lookback
        if end_idx <= start_idx:
            continue
        past_ret = cum_sum[end_idx] - cum_sum[start_idx]
        if past_ret > threshold:
            sig[i] = 1.0
        elif past_ret < -threshold:
            sig[i] = -1.0
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 2: momentum_3m
# ---------------------------------------------------------------------------

def momentum_3m(
    returns: np.ndarray,
    *,
    lookback: int = 63,
    vol_scale: bool = True,
) -> np.ndarray:
    """
    3-month momentum, optionally scaled by inverse realised volatility.

    Parameters
    ----------
    returns : 1-D array of period returns.
    lookback : window for momentum calculation (default 63 ~ 3 months).
    vol_scale : if True, divide raw momentum score by rolling vol so that
        the signal is volatility-normalised.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    sig = np.zeros(n, dtype=np.float64)
    cum = np.cumsum(np.log1p(returns))
    rvol = _rolling_std(returns, lookback)

    for i in range(lookback, n):
        mom = cum[i] - cum[i - lookback]
        if vol_scale and rvol[i] > 1e-12:
            mom = mom / (rvol[i] * np.sqrt(lookback))
        sig[i] = np.tanh(mom)
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 3: mean_reversion_z
# ---------------------------------------------------------------------------

def mean_reversion_z(
    returns: np.ndarray,
    *,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> np.ndarray:
    """
    Z-score mean reversion.  Build a cumulative-return proxy, z-score it
    against its own rolling mean/std, and fade extremes.

    Parameters
    ----------
    returns : 1-D array of period returns.
    window : lookback for rolling statistics.
    entry_z : absolute z-score to enter a position (short above, long below).
    exit_z : absolute z-score to flatten.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    if window < 2:
        raise ValueError("window must be >= 2")
    if entry_z <= exit_z:
        raise ValueError("entry_z must be > exit_z")

    prices = _prices_from_returns(returns)
    rm = _rolling_mean(prices, window)
    rs = _rolling_std(prices, window)
    sig = np.zeros(n, dtype=np.float64)
    pos = 0.0

    for i in range(window, n):
        if rs[i] < 1e-15:
            sig[i] = pos
            continue
        z = (prices[i] - rm[i]) / rs[i]
        if z > entry_z:
            pos = -1.0
        elif z < -entry_z:
            pos = 1.0
        elif abs(z) < exit_z:
            pos = 0.0
        sig[i] = pos
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 4: mean_reversion_ou
# ---------------------------------------------------------------------------

def mean_reversion_ou(
    returns: np.ndarray,
    *,
    halflife_est: int = 30,
    entry_sigma: float = 1.5,
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck calibrated mean reversion.

    Estimate OU parameters from recent data (rolling window = 4 * halflife_est),
    then trade when deviations exceed *entry_sigma* equilibrium standard
    deviations.

    Parameters
    ----------
    returns : 1-D returns.
    halflife_est : assumed OU half-life for calibration window sizing.
    entry_sigma : number of OU-equilibrium sigmas for entry.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    prices = _prices_from_returns(returns)
    log_p = np.log(prices + 1e-15)
    window = max(4 * halflife_est, 60)
    sig = np.zeros(n, dtype=np.float64)
    pos = 0.0

    for i in range(window, n):
        seg = log_p[i - window: i + 1]
        y = seg[1:]
        x = seg[:-1]
        if np.std(x) < 1e-15:
            sig[i] = pos
            continue
        slope, intercept, _, _, _ = sp_stats.linregress(x, y)
        theta = -np.log(max(abs(slope), 1e-8))
        mu = intercept / max(1.0 - slope, 1e-8)
        residuals = y - (slope * x + intercept)
        sigma_eq = np.std(residuals) / np.sqrt(max(2.0 * theta, 1e-8))

        deviation = (log_p[i] - mu) / max(sigma_eq, 1e-12)
        if deviation > entry_sigma:
            pos = -1.0
        elif deviation < -entry_sigma:
            pos = 1.0
        elif abs(deviation) < entry_sigma * 0.3:
            pos = 0.0
        sig[i] = pos
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 5: breakout_donchian
# ---------------------------------------------------------------------------

def breakout_donchian(
    returns: np.ndarray,
    *,
    window: int = 55,
    atr_mult_for_stop: float = 2.0,
) -> np.ndarray:
    """
    Donchian channel breakout (turtle-style).

    Go long when price exceeds rolling high, short when below rolling low.
    An ATR-based trailing stop is applied.

    Parameters
    ----------
    returns : 1-D returns.
    window : Donchian channel lookback.
    atr_mult_for_stop : ATR multiplier for trailing stop-loss.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    prices = _prices_from_returns(returns)
    high = _rolling_max(prices, window)
    low = _rolling_min(prices, window)
    atr_arr = _atr(returns, window)

    sig = np.zeros(n, dtype=np.float64)
    pos = 0.0
    trail_stop = 0.0

    for i in range(window, n):
        p = prices[i]
        a = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0

        if pos == 0.0:
            if p >= high[i - 1]:
                pos = 1.0
                trail_stop = p - atr_mult_for_stop * a
            elif p <= low[i - 1]:
                pos = -1.0
                trail_stop = p + atr_mult_for_stop * a
        elif pos == 1.0:
            trail_stop = max(trail_stop, p - atr_mult_for_stop * a)
            if p < trail_stop:
                pos = 0.0
        elif pos == -1.0:
            trail_stop = min(trail_stop, p + atr_mult_for_stop * a)
            if p > trail_stop:
                pos = 0.0
        sig[i] = pos
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 6: breakout_bollinger
# ---------------------------------------------------------------------------

def breakout_bollinger(
    returns: np.ndarray,
    *,
    window: int = 20,
    n_std: float = 2.0,
    close_at_mid: bool = True,
) -> np.ndarray:
    """
    Bollinger Band breakout.

    Enter long on upper-band break, short on lower-band break.
    Optionally flatten when price reverts to the middle band.

    Parameters
    ----------
    returns : 1-D returns.
    window : Bollinger lookback.
    n_std : number of standard deviations for band width.
    close_at_mid : if True, flatten when price crosses back to the MA.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    prices = _prices_from_returns(returns)
    rm = _rolling_mean(prices, window)
    rs = _rolling_std(prices, window)
    sig = np.zeros(n, dtype=np.float64)
    pos = 0.0

    for i in range(window, n):
        upper = rm[i] + n_std * rs[i]
        lower = rm[i] - n_std * rs[i]
        p = prices[i]
        if p > upper:
            pos = 1.0
        elif p < lower:
            pos = -1.0
        elif close_at_mid:
            if pos == 1.0 and p < rm[i]:
                pos = 0.0
            elif pos == -1.0 and p > rm[i]:
                pos = 0.0
        sig[i] = pos
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 7: rsi_reversal
# ---------------------------------------------------------------------------

def rsi_reversal(
    returns: np.ndarray,
    *,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> np.ndarray:
    """
    RSI-based reversal strategy.

    Compute Wilder-style RSI from returns.  Go long when RSI < oversold,
    short when RSI > overbought, flatten otherwise.

    Parameters
    ----------
    returns : 1-D returns.
    period : RSI period.
    oversold : long entry threshold.
    overbought : short entry threshold.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    if period < 1:
        raise ValueError("period must be >= 1")
    if not (0.0 < oversold < overbought < 100.0):
        raise ValueError("Need 0 < oversold < overbought < 100")

    gains = np.maximum(returns, 0.0)
    losses = np.abs(np.minimum(returns, 0.0))

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    sig = np.zeros(n, dtype=np.float64)
    pos = 0.0
    for i in range(period + 1, n):
        if avg_loss[i] < 1e-15:
            rsi = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi = 100.0 - 100.0 / (1.0 + rs)

        if rsi < oversold:
            pos = 1.0
        elif rsi > overbought:
            pos = -1.0
        elif 40.0 < rsi < 60.0:
            pos = 0.0
        sig[i] = pos
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 8: macd_crossover
# ---------------------------------------------------------------------------

def macd_crossover(
    returns: np.ndarray,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> np.ndarray:
    """
    MACD crossover.

    Compute MACD line from fast/slow EMA of cumulative returns, then a signal
    line as EMA of the MACD.  Go long when MACD > signal, short otherwise.

    Parameters
    ----------
    returns : 1-D returns.
    fast : fast EMA span.
    slow : slow EMA span.
    signal : signal EMA span.
    """
    returns = _validate_1d(returns)
    if fast >= slow:
        raise ValueError("fast must be < slow")
    prices = _prices_from_returns(returns)
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)

    sig = np.zeros(len(returns), dtype=np.float64)
    for i in range(slow, len(returns)):
        diff = macd_line[i] - signal_line[i]
        sig[i] = np.tanh(diff / (np.std(macd_line[slow:i + 1]) + 1e-12))
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 9: dual_ma_cross
# ---------------------------------------------------------------------------

def dual_ma_cross(
    returns: np.ndarray,
    *,
    fast_window: int = 20,
    slow_window: int = 60,
) -> np.ndarray:
    """
    Dual moving average crossover.

    Long when fast MA > slow MA, short otherwise.

    Parameters
    ----------
    returns : 1-D returns.
    fast_window : fast SMA window.
    slow_window : slow SMA window.
    """
    returns = _validate_1d(returns)
    if fast_window >= slow_window:
        raise ValueError("fast_window must be < slow_window")
    prices = _prices_from_returns(returns)
    ma_fast = _rolling_mean(prices, fast_window)
    ma_slow = _rolling_mean(prices, slow_window)

    sig = np.zeros(len(returns), dtype=np.float64)
    for i in range(slow_window, len(returns)):
        if ma_fast[i] > ma_slow[i]:
            sig[i] = 1.0
        else:
            sig[i] = -1.0
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 10: triple_ma
# ---------------------------------------------------------------------------

def triple_ma(
    returns: np.ndarray,
    *,
    fast: int = 10,
    medium: int = 30,
    slow: int = 60,
) -> np.ndarray:
    """
    Triple moving average filter.

    Full long only when fast > medium > slow.  Full short only when
    fast < medium < slow.  Half signals for partial alignment.

    Parameters
    ----------
    returns : 1-D returns.
    fast : fast SMA window.
    medium : medium SMA window.
    slow : slow SMA window.
    """
    returns = _validate_1d(returns)
    if not (fast < medium < slow):
        raise ValueError("Require fast < medium < slow")
    prices = _prices_from_returns(returns)
    mf = _rolling_mean(prices, fast)
    mm = _rolling_mean(prices, medium)
    ms = _rolling_mean(prices, slow)

    sig = np.zeros(len(returns), dtype=np.float64)
    for i in range(slow, len(returns)):
        if mf[i] > mm[i] > ms[i]:
            sig[i] = 1.0
        elif mf[i] < mm[i] < ms[i]:
            sig[i] = -1.0
        elif mf[i] > mm[i]:
            sig[i] = 0.5
        elif mf[i] < mm[i]:
            sig[i] = -0.5
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 11: trend_following_adx
# ---------------------------------------------------------------------------

def trend_following_adx(
    returns: np.ndarray,
    *,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
) -> np.ndarray:
    """
    ADX trend filter.

    Approximate +DI / -DI from returns.  When ADX > threshold and +DI > -DI,
    go long; when ADX > threshold and -DI > +DI, go short.  Flat otherwise.

    Parameters
    ----------
    returns : 1-D returns.
    adx_period : ADX smoothing period.
    adx_threshold : minimum ADX to take a position.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    prices = _prices_from_returns(returns)

    # Approximate directional moves from consecutive returns
    up_move = np.zeros(n)
    down_move = np.zeros(n)
    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            up_move[i] = diff
        else:
            down_move[i] = -diff

    tr = _true_range_from_returns(returns)
    smooth_tr = _ema(tr, adx_period)
    smooth_up = _ema(up_move, adx_period)
    smooth_down = _ema(down_move, adx_period)

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    for i in range(adx_period, n):
        if smooth_tr[i] > 1e-15:
            plus_di[i] = 100.0 * smooth_up[i] / smooth_tr[i]
            minus_di[i] = 100.0 * smooth_down[i] / smooth_tr[i]
        denom = plus_di[i] + minus_di[i]
        if denom > 1e-15:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom

    adx = _ema(dx, adx_period)
    sig = np.zeros(n, dtype=np.float64)
    for i in range(2 * adx_period, n):
        if adx[i] > adx_threshold:
            if plus_di[i] > minus_di[i]:
                sig[i] = 1.0
            else:
                sig[i] = -1.0
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 12: volatility_breakout
# ---------------------------------------------------------------------------

def volatility_breakout(
    returns: np.ndarray,
    *,
    vol_window: int = 20,
    expansion_mult: float = 1.5,
) -> np.ndarray:
    """
    Volatility expansion breakout.

    When current vol exceeds *expansion_mult* x trailing average vol,
    trade in the direction of the recent move.

    Parameters
    ----------
    returns : 1-D returns.
    vol_window : window for measuring volatility.
    expansion_mult : multiplier triggering breakout.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    rvol = _rolling_std(returns, vol_window)
    avg_vol = _rolling_mean(rvol, vol_window * 2)
    rmean = _rolling_mean(returns, vol_window)

    sig = np.zeros(n, dtype=np.float64)
    for i in range(vol_window * 3, n):
        if np.isnan(avg_vol[i]) or avg_vol[i] < 1e-15:
            continue
        if rvol[i] > expansion_mult * avg_vol[i]:
            direction = np.sign(rmean[i])
            sig[i] = direction
        else:
            sig[i] = 0.0
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 13: carry_trade
# ---------------------------------------------------------------------------

def carry_trade(
    returns: np.ndarray,
    *,
    lookback: int = 60,
    min_carry: float = 0.0002,
) -> np.ndarray:
    """
    Positive drift carry.

    Estimate carry as rolling mean of returns (drift proxy).  Go long
    when drift exceeds *min_carry*, short when below negative threshold.

    Parameters
    ----------
    returns : 1-D returns.
    lookback : window for drift estimation.
    min_carry : minimum daily drift to take position.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    drift = _rolling_mean(returns, lookback)
    vol = _rolling_std(returns, lookback)

    sig = np.zeros(n, dtype=np.float64)
    for i in range(lookback, n):
        if np.isnan(drift[i]) or np.isnan(vol[i]):
            continue
        if vol[i] < 1e-15:
            continue
        carry_score = drift[i] / vol[i]
        if drift[i] > min_carry:
            sig[i] = min(carry_score * 10.0, 1.0)
        elif drift[i] < -min_carry:
            sig[i] = max(carry_score * 10.0, -1.0)
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 14: vol_targeting
# ---------------------------------------------------------------------------

def vol_targeting(
    returns: np.ndarray,
    *,
    target_vol: float = 0.10,
    vol_window: int = 60,
) -> np.ndarray:
    """
    Inverse volatility sizing (vol targeting).

    Position size = target_vol / realised_vol.  Direction follows recent
    momentum sign.  This is a sizing overlay rather than a pure alpha signal.

    Parameters
    ----------
    returns : 1-D returns.
    target_vol : annualised target volatility.
    vol_window : window for vol estimation.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    rvol = _rolling_std(returns, vol_window)
    ann_factor = np.sqrt(252.0)
    cum = np.cumsum(np.log1p(returns))

    sig = np.zeros(n, dtype=np.float64)
    for i in range(vol_window, n):
        ann_vol = rvol[i] * ann_factor
        if ann_vol < 1e-12:
            continue
        size = target_vol / ann_vol
        direction = np.sign(cum[i] - cum[max(i - vol_window, 0)])
        sig[i] = direction * min(size, 1.0)
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 15: pairs_reversion
# ---------------------------------------------------------------------------

def pairs_reversion(
    returns: np.ndarray,
    *,
    spread_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> np.ndarray:
    """
    Self-pairs z-score reversion.

    Uses the spread between short-term and long-term cumulative returns
    as a synthetic spread, then applies z-score mean reversion.

    Parameters
    ----------
    returns : 1-D returns.
    spread_window : lookback for spread statistics.
    entry_z : z-score to enter.
    exit_z : z-score to exit.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    if spread_window < 2:
        raise ValueError("spread_window must be >= 2")
    prices = _prices_from_returns(returns)
    ma_short = _rolling_mean(prices, max(spread_window // 4, 2))
    ma_long = _rolling_mean(prices, spread_window)
    spread = ma_short - ma_long

    rm = _rolling_mean(spread, spread_window)
    rs = _rolling_std(spread, spread_window)
    sig = np.zeros(n, dtype=np.float64)
    pos = 0.0

    for i in range(spread_window * 2, n):
        if np.isnan(rm[i]) or np.isnan(rs[i]) or rs[i] < 1e-15:
            sig[i] = pos
            continue
        z = (spread[i] - rm[i]) / rs[i]
        if z > entry_z:
            pos = -1.0
        elif z < -entry_z:
            pos = 1.0
        elif abs(z) < exit_z:
            pos = 0.0
        sig[i] = pos
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 16: entropy_regime
# ---------------------------------------------------------------------------

def entropy_regime(
    returns: np.ndarray,
    *,
    order: int = 3,
    delay: int = 1,
    threshold: float = 0.7,
) -> np.ndarray:
    """
    Permutation entropy regime detection.

    High entropy => random => no position.  Low entropy => predictable =>
    trade momentum.

    Parameters
    ----------
    returns : 1-D returns.
    order : permutation order (typically 3-7).
    delay : embedding delay.
    threshold : normalised entropy threshold (0-1).  Below = trade.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    max_entropy = np.log2(float(np.math.factorial(order)))
    window = max(order * delay * 10, 50)
    cum = np.cumsum(np.log1p(returns))
    sig = np.zeros(n, dtype=np.float64)

    for i in range(window, n):
        seg = returns[i - window: i]
        pe = _permutation_entropy(seg, order, delay)
        norm_pe = pe / max_entropy if max_entropy > 0 else 1.0

        if norm_pe < threshold:
            mom = cum[i] - cum[i - window // 2]
            sig[i] = np.sign(mom)
        else:
            sig[i] = 0.0
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 17: hurst_adaptive
# ---------------------------------------------------------------------------

def hurst_adaptive(
    returns: np.ndarray,
    *,
    hurst_window: int = 100,
    trending_h: float = 0.6,
    reverting_h: float = 0.4,
) -> np.ndarray:
    """
    Hurst exponent adaptive strategy.

    H > trending_h => momentum mode.  H < reverting_h => mean reversion mode.
    In between => flat.

    Parameters
    ----------
    returns : 1-D returns.
    hurst_window : window for Hurst estimation.
    trending_h : Hurst threshold for trending regime.
    reverting_h : Hurst threshold for reverting regime.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    prices = _prices_from_returns(returns)
    cum = np.cumsum(np.log1p(returns))
    rm = _rolling_mean(prices, hurst_window)
    rs = _rolling_std(prices, hurst_window)
    sig = np.zeros(n, dtype=np.float64)

    step = max(hurst_window // 10, 1)
    cached_h = 0.5
    for i in range(hurst_window, n):
        if (i - hurst_window) % step == 0:
            seg = returns[i - hurst_window: i]
            cached_h = _hurst_rs(seg)

        mom = cum[i] - cum[max(i - hurst_window // 2, 0)]
        if cached_h > trending_h:
            sig[i] = np.sign(mom)
        elif cached_h < reverting_h:
            if rs[i] > 1e-15:
                z = (prices[i] - rm[i]) / rs[i]
                sig[i] = -np.tanh(z)
            else:
                sig[i] = 0.0
        else:
            sig[i] = 0.0
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 18: bh_physics_mass
# ---------------------------------------------------------------------------

def bh_physics_mass(
    returns: np.ndarray,
    *,
    bh_form: float = 1.0,
    bh_decay: float = 0.95,
    ctl_min: float = 0.1,
) -> np.ndarray:
    """
    Black-hole mass accumulation model.

    Treat absolute returns as mass accretion.  Accumulated mass above a
    critical threshold triggers a signal in the direction of recent drift.
    Mass decays each period.

    Parameters
    ----------
    returns : 1-D returns.
    bh_form : formation rate multiplier for mass accretion.
    bh_decay : per-period mass decay factor.
    ctl_min : minimum accumulated mass to generate signal.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    mass = np.zeros(n, dtype=np.float64)
    drift = np.zeros(n, dtype=np.float64)
    sig = np.zeros(n, dtype=np.float64)

    alpha = 0.05
    for i in range(1, n):
        accretion = bh_form * abs(returns[i])
        mass[i] = bh_decay * mass[i - 1] + accretion
        drift[i] = (1.0 - alpha) * drift[i - 1] + alpha * returns[i]

        if mass[i] > ctl_min:
            hawking_factor = 1.0 / (1.0 + mass[i])
            sig[i] = np.sign(drift[i]) * (1.0 - hawking_factor)
        else:
            sig[i] = 0.0
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 19: fractal_coherence
# ---------------------------------------------------------------------------

def fractal_coherence(
    returns: np.ndarray,
    *,
    scales: list[int] | None = None,
    coherence_threshold: float = 0.6,
) -> np.ndarray:
    """
    Multi-scale coherence.

    Compute momentum at multiple time scales.  When they agree (coherence
    is high), trade in that direction.

    Parameters
    ----------
    returns : 1-D returns.
    scales : list of lookback windows.  Default [5, 10, 21, 63].
    coherence_threshold : fraction of scales that must agree.
    """
    returns = _validate_1d(returns)
    if scales is None:
        scales = [5, 10, 21, 63]
    n = len(returns)
    cum = np.cumsum(np.log1p(returns))
    sig = np.zeros(n, dtype=np.float64)
    max_scale = max(scales)

    for i in range(max_scale, n):
        directions = []
        for s in scales:
            mom = cum[i] - cum[i - s]
            directions.append(np.sign(mom))
        avg_dir = np.mean(directions)
        coherence = abs(avg_dir)
        if coherence >= coherence_threshold:
            sig[i] = np.sign(avg_dir)
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 20: info_surprise
# ---------------------------------------------------------------------------

def info_surprise(
    returns: np.ndarray,
    *,
    entropy_window: int = 50,
    spike_threshold: float = 2.0,
) -> np.ndarray:
    """
    Entropy spike detection.

    Compare recent permutation entropy to a trailing baseline.  A spike
    (increase) suggests a regime change — fade the recent move.

    Parameters
    ----------
    returns : 1-D returns.
    entropy_window : window for entropy computation.
    spike_threshold : z-score of entropy above baseline to trigger.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    order = 3
    entropies = np.full(n, np.nan)
    baseline_window = entropy_window * 3

    for i in range(entropy_window, n):
        seg = returns[i - entropy_window: i]
        entropies[i] = _permutation_entropy(seg, order, 1)

    sig = np.zeros(n, dtype=np.float64)
    for i in range(baseline_window, n):
        base_seg = entropies[i - baseline_window: i]
        valid = base_seg[~np.isnan(base_seg)]
        if len(valid) < 10:
            continue
        mu, sd = np.mean(valid), np.std(valid)
        if sd < 1e-12:
            continue
        z = (entropies[i] - mu) / sd
        if z > spike_threshold:
            recent_ret = np.sum(returns[i - 5: i])
            sig[i] = -np.sign(recent_ret)
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 21: liquidity_fade
# ---------------------------------------------------------------------------

def liquidity_fade(
    returns: np.ndarray,
    *,
    spread_threshold: float = 2.0,
    recovery_window: int = 10,
) -> np.ndarray:
    """
    Fade liquidity events.

    Detect large absolute returns (proxy for illiquidity) and bet on mean
    reversion over *recovery_window* periods.

    Parameters
    ----------
    returns : 1-D returns.
    spread_threshold : z-score of |return| to identify liquidity event.
    recovery_window : how long to hold the fade position.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    window = 60
    abs_ret = np.abs(returns)
    rm = _rolling_mean(abs_ret, window)
    rs = _rolling_std(abs_ret, window)
    sig = np.zeros(n, dtype=np.float64)
    hold_until = 0

    for i in range(window, n):
        if rs[i] < 1e-15:
            continue
        z = (abs_ret[i] - rm[i]) / rs[i]
        if z > spread_threshold and i > hold_until:
            sig[i] = -np.sign(returns[i])
            hold_until = i + recovery_window
        elif i <= hold_until and i > 0:
            sig[i] = sig[i - 1]
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 22: whale_follow
# ---------------------------------------------------------------------------

def whale_follow(
    returns: np.ndarray,
    *,
    size_threshold: float = 2.5,
    persistence: int = 5,
) -> np.ndarray:
    """
    Follow large order flow (approximated from outsized returns).

    When a return exceeds *size_threshold* standard deviations, follow it
    for *persistence* periods (assumes informed flow).

    Parameters
    ----------
    returns : 1-D returns.
    size_threshold : z-score to identify whale activity.
    persistence : how many periods to hold the follow position.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    window = 60
    rm = _rolling_mean(returns, window)
    rs = _rolling_std(returns, window)
    sig = np.zeros(n, dtype=np.float64)
    hold_until = 0
    hold_dir = 0.0

    for i in range(window, n):
        if rs[i] < 1e-15:
            continue
        z = (returns[i] - rm[i]) / rs[i]
        if abs(z) > size_threshold and i >= hold_until:
            hold_dir = np.sign(z)
            hold_until = i + persistence
        if i < hold_until:
            sig[i] = hold_dir
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 23: calendar_effect
# ---------------------------------------------------------------------------

def calendar_effect(
    returns: np.ndarray,
    *,
    boost_days: list[int] | None = None,
    reduce_days: list[int] | None = None,
) -> np.ndarray:
    """
    Calendar effect (day-of-week / month-end proxy).

    Use modular arithmetic on indices as a proxy for day-of-week.
    Boost position on *boost_days*, reduce on *reduce_days*.

    Parameters
    ----------
    returns : 1-D returns.
    boost_days : index-mod-5 days to boost (default [0, 4] = Mon, Fri).
    reduce_days : index-mod-5 days to reduce (default [2] = Wed).
    """
    returns = _validate_1d(returns)
    n = len(returns)
    if boost_days is None:
        boost_days = [0, 4]
    if reduce_days is None:
        reduce_days = [2]

    cum = np.cumsum(np.log1p(returns))
    trend_window = 20
    sig = np.zeros(n, dtype=np.float64)

    for i in range(trend_window, n):
        base_direction = np.sign(cum[i] - cum[i - trend_window])
        day = i % 5
        if day in boost_days:
            sig[i] = base_direction * 1.0
        elif day in reduce_days:
            sig[i] = base_direction * 0.25
        else:
            sig[i] = base_direction * 0.5
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 24: overnight_gap
# ---------------------------------------------------------------------------

def overnight_gap(
    returns: np.ndarray,
    *,
    gap_threshold: float = 1.5,
    fade_or_follow: str = "fade",
) -> np.ndarray:
    """
    Trade overnight gaps (approximated).

    Detect outsized first-period moves and either fade or follow them.

    Parameters
    ----------
    returns : 1-D returns.
    gap_threshold : z-score to identify gap.
    fade_or_follow : 'fade' to mean-revert the gap, 'follow' to ride it.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    if fade_or_follow not in ("fade", "follow"):
        raise ValueError("fade_or_follow must be 'fade' or 'follow'")
    window = 30
    rm = _rolling_mean(returns, window)
    rs = _rolling_std(returns, window)
    sig = np.zeros(n, dtype=np.float64)
    mult = -1.0 if fade_or_follow == "fade" else 1.0

    for i in range(window, n):
        if rs[i] < 1e-15:
            continue
        z = (returns[i] - rm[i]) / rs[i]
        if abs(z) > gap_threshold:
            sig[i] = mult * np.sign(z)
        elif i > 0:
            sig[i] = sig[i - 1] * 0.8
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 25: vwap_reversion
# ---------------------------------------------------------------------------

def vwap_reversion(
    returns: np.ndarray,
    *,
    vwap_window: int = 20,
    deviation_threshold: float = 1.5,
) -> np.ndarray:
    """
    Revert to VWAP (approximated as volume-weighted average price proxy).

    Uses absolute-return weighting as a volume proxy.

    Parameters
    ----------
    returns : 1-D returns.
    vwap_window : window for VWAP calculation.
    deviation_threshold : number of std devs from VWAP to trade.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    prices = _prices_from_returns(returns)
    vol_proxy = np.abs(returns) + 1e-10

    sig = np.zeros(n, dtype=np.float64)
    pos = 0.0

    for i in range(vwap_window, n):
        seg_p = prices[i - vwap_window: i + 1]
        seg_v = vol_proxy[i - vwap_window: i + 1]
        vwap = np.sum(seg_p * seg_v) / np.sum(seg_v)
        std_p = np.std(seg_p)
        if std_p < 1e-15:
            sig[i] = pos
            continue
        dev = (prices[i] - vwap) / std_p
        if dev > deviation_threshold:
            pos = -1.0
        elif dev < -deviation_threshold:
            pos = 1.0
        elif abs(dev) < 0.5:
            pos = 0.0
        sig[i] = pos
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 26: order_flow_imbalance
# ---------------------------------------------------------------------------

def order_flow_imbalance(
    returns: np.ndarray,
    *,
    imbalance_window: int = 20,
    threshold: float = 0.6,
) -> np.ndarray:
    """
    Trade imbalanced order flow (proxied from return asymmetry).

    Count positive vs negative returns in a rolling window.  When the
    fraction exceeds *threshold*, follow the dominant direction.

    Parameters
    ----------
    returns : 1-D returns.
    imbalance_window : rolling window to assess imbalance.
    threshold : fraction of same-sign returns to trigger signal.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    positive = (returns > 0).astype(np.float64)
    frac_pos = _rolling_mean(positive, imbalance_window)
    vol_weighted = _rolling_mean(returns * np.abs(returns), imbalance_window)

    sig = np.zeros(n, dtype=np.float64)
    for i in range(imbalance_window, n):
        if np.isnan(frac_pos[i]):
            continue
        if frac_pos[i] > threshold:
            sig[i] = min((frac_pos[i] - 0.5) * 4.0, 1.0)
        elif frac_pos[i] < (1.0 - threshold):
            sig[i] = max((frac_pos[i] - 0.5) * 4.0, -1.0)
        else:
            sig[i] = np.sign(vol_weighted[i]) * 0.3 if not np.isnan(vol_weighted[i]) else 0.0
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 27: correlation_regime
# ---------------------------------------------------------------------------

def correlation_regime(
    returns: np.ndarray,
    *,
    corr_window: int = 60,
    herding_threshold: float = 0.7,
) -> np.ndarray:
    """
    Trade correlation regime transitions.

    Compute autocorrelation as a herding proxy.  High autocorrelation =>
    herding => momentum.  Low => dispersion => mean reversion.

    Parameters
    ----------
    returns : 1-D returns.
    corr_window : rolling window for autocorrelation.
    herding_threshold : autocorrelation level triggering momentum mode.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    cum = np.cumsum(np.log1p(returns))
    prices = _prices_from_returns(returns)
    rm = _rolling_mean(prices, corr_window)
    rs = _rolling_std(prices, corr_window)
    sig = np.zeros(n, dtype=np.float64)

    for i in range(corr_window + 1, n):
        seg = returns[i - corr_window: i]
        x = seg[:-1]
        y = seg[1:]
        sx, sy = np.std(x), np.std(y)
        if sx < 1e-15 or sy < 1e-15:
            continue
        autocorr = np.corrcoef(x, y)[0, 1]

        if abs(autocorr) > herding_threshold:
            mom = cum[i] - cum[i - corr_window // 2]
            sig[i] = np.sign(mom)
        elif abs(autocorr) < (1.0 - herding_threshold):
            if rs[i] > 1e-15:
                z = (prices[i] - rm[i]) / rs[i]
                sig[i] = -np.tanh(z)
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 28: gamma_exposure
# ---------------------------------------------------------------------------

def gamma_exposure(
    returns: np.ndarray,
    *,
    pin_distance: float = 0.02,
    amplify_distance: float = 0.05,
) -> np.ndarray:
    """
    Trade GEX levels (gamma exposure proxy).

    Identify round-number price levels.  Near pin levels, expect mean
    reversion (dealer hedging pins price).  Far from pin => amplification.

    Parameters
    ----------
    returns : 1-D returns.
    pin_distance : relative distance to pin level triggering reversion.
    amplify_distance : relative distance beyond which moves amplify.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    prices = _prices_from_returns(returns)
    sig = np.zeros(n, dtype=np.float64)
    window = 20

    for i in range(window, n):
        p = prices[i]
        magnitude = 10 ** np.floor(np.log10(max(p, 0.01)))
        round_level = np.round(p / magnitude) * magnitude
        rel_dist = abs(p - round_level) / max(p, 1e-15)

        recent_mom = np.sum(returns[i - 5: i])
        if rel_dist < pin_distance:
            sig[i] = -np.sign(p - round_level) * 0.8
        elif rel_dist > amplify_distance:
            sig[i] = np.sign(recent_mom)
        else:
            sig[i] = np.sign(recent_mom) * 0.3
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 29: funding_rate
# ---------------------------------------------------------------------------

def funding_rate(
    returns: np.ndarray,
    *,
    rate_threshold: float = 0.0005,
    lookback: int = 30,
) -> np.ndarray:
    """
    Crypto funding rate carry (proxy).

    Use rolling mean return as a proxy for funding rate.  Positive drift
    implies longs pay shorts — carry favours shorts and vice versa.

    Parameters
    ----------
    returns : 1-D returns.
    rate_threshold : minimum funding rate proxy to trade.
    lookback : window for funding rate estimation.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    drift = _rolling_mean(returns, lookback)
    vol = _rolling_std(returns, lookback)
    sig = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        if np.isnan(drift[i]) or np.isnan(vol[i]):
            continue
        funding_proxy = drift[i]
        if abs(funding_proxy) > rate_threshold:
            carry_signal = -np.sign(funding_proxy)
            if vol[i] > 1e-15:
                sharpe_weight = min(abs(funding_proxy) / vol[i] * np.sqrt(252), 3.0) / 3.0
            else:
                sharpe_weight = 0.5
            sig[i] = carry_signal * sharpe_weight
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy 30: combined_momentum_reversion
# ---------------------------------------------------------------------------

def combined_momentum_reversion(
    returns: np.ndarray,
    *,
    regime_window: int = 60,
    mom_weight: float = 0.6,
    rev_weight: float = 0.4,
) -> np.ndarray:
    """
    Regime-switching between momentum and mean reversion.

    Estimate whether the market is trending or reverting using the Hurst
    exponent, then blend momentum and reversion signals accordingly.

    Parameters
    ----------
    returns : 1-D returns.
    regime_window : window for regime detection.
    mom_weight : base weight for momentum when trending.
    rev_weight : base weight for reversion when reverting.
    """
    returns = _validate_1d(returns)
    n = len(returns)
    prices = _prices_from_returns(returns)
    cum = np.cumsum(np.log1p(returns))
    rm = _rolling_mean(prices, regime_window)
    rs = _rolling_std(prices, regime_window)
    sig = np.zeros(n, dtype=np.float64)

    step = max(regime_window // 5, 1)
    cached_h = 0.5
    for i in range(regime_window, n):
        if (i - regime_window) % step == 0:
            seg = returns[i - regime_window: i]
            cached_h = _hurst_rs(seg)

        mom_sig = np.tanh((cum[i] - cum[max(i - regime_window // 2, 0)]) * 10.0)

        rev_sig = 0.0
        if rs[i] > 1e-15:
            z = (prices[i] - rm[i]) / rs[i]
            rev_sig = -np.tanh(z)

        if cached_h > 0.55:
            w_mom = mom_weight + (cached_h - 0.5) * 2.0 * (1.0 - mom_weight)
            w_rev = 1.0 - w_mom
        elif cached_h < 0.45:
            w_rev = rev_weight + (0.5 - cached_h) * 2.0 * (1.0 - rev_weight)
            w_mom = 1.0 - w_rev
        else:
            w_mom = 0.5
            w_rev = 0.5

        sig[i] = w_mom * mom_sig + w_rev * rev_sig
    return _clamp(sig)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, callable] = {
    "momentum_12_1": momentum_12_1,
    "momentum_3m": momentum_3m,
    "mean_reversion_z": mean_reversion_z,
    "mean_reversion_ou": mean_reversion_ou,
    "breakout_donchian": breakout_donchian,
    "breakout_bollinger": breakout_bollinger,
    "rsi_reversal": rsi_reversal,
    "macd_crossover": macd_crossover,
    "dual_ma_cross": dual_ma_cross,
    "triple_ma": triple_ma,
    "trend_following_adx": trend_following_adx,
    "volatility_breakout": volatility_breakout,
    "carry_trade": carry_trade,
    "vol_targeting": vol_targeting,
    "pairs_reversion": pairs_reversion,
    "entropy_regime": entropy_regime,
    "hurst_adaptive": hurst_adaptive,
    "bh_physics_mass": bh_physics_mass,
    "fractal_coherence": fractal_coherence,
    "info_surprise": info_surprise,
    "liquidity_fade": liquidity_fade,
    "whale_follow": whale_follow,
    "calendar_effect": calendar_effect,
    "overnight_gap": overnight_gap,
    "vwap_reversion": vwap_reversion,
    "order_flow_imbalance": order_flow_imbalance,
    "correlation_regime": correlation_regime,
    "gamma_exposure": gamma_exposure,
    "funding_rate": funding_rate,
    "combined_momentum_reversion": combined_momentum_reversion,
}


def list_strategies() -> list[str]:
    """Return the names of all registered strategies."""
    return list(STRATEGY_REGISTRY.keys())


def get_strategy(name: str) -> callable:
    """Look up a strategy function by name."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list_strategies()}")
    return STRATEGY_REGISTRY[name]


def run_strategy(name: str, returns: np.ndarray, **params) -> np.ndarray:
    """Look up and run a strategy by name."""
    fn = get_strategy(name)
    return fn(returns, **params)


def default_params(name: str) -> dict:
    """Return the default parameters for a strategy (from its signature)."""
    import inspect
    fn = get_strategy(name)
    sig = inspect.signature(fn)
    defaults = {}
    for pname, param in sig.parameters.items():
        if pname == "returns":
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[pname] = param.default
    return defaults
