"""
fast_indicators — High-performance technical indicators and BH physics.

Attempts to import the compiled C extension. If unavailable, falls back
to pure numpy implementations that are API-compatible.

Usage:
    from fast_indicators import ema, rsi, macd, atr, bollinger, adx
    from fast_indicators import stochastic, vwap, obv, bh_series, bh_backtest
    from fast_indicators import USING_C  # True if C extension is loaded
"""

from __future__ import annotations

import math
import warnings
import numpy as np
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Attempt C extension import
# ─────────────────────────────────────────────────────────────────────────────

try:
    from . import fast_indicators as _c_ext
    USING_C = True
except ImportError:
    _c_ext = None
    USING_C = False
    warnings.warn(
        "fast_indicators C extension not found. "
        "Using numpy fallback (slower). Run `python setup.py build_ext --inplace`.",
        ImportWarning,
        stacklevel=2,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helper: bytes → numpy array
# ─────────────────────────────────────────────────────────────────────────────

def _to_f64(buf: bytes) -> np.ndarray:
    """Convert raw C bytes (float64) to a numpy array."""
    return np.frombuffer(buf, dtype=np.float64).copy()


def _to_i32(buf: bytes) -> np.ndarray:
    """Convert raw C bytes (int32) to a numpy array."""
    return np.frombuffer(buf, dtype=np.int32).copy()


def _c64(arr) -> np.ndarray:
    """Ensure contiguous float64 C-order array."""
    return np.ascontiguousarray(arr, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA — Exponential Moving Average
# ─────────────────────────────────────────────────────────────────────────────

def ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average.

    Parameters
    ----------
    close  : 1-D float64 array of close prices
    period : EMA period

    Returns
    -------
    np.ndarray, same length as close. NaN for warm-up bars.
    """
    close = _c64(close)
    if USING_C:
        return _to_f64(_c_ext._ema(close, period))

    # Numpy fallback
    n = len(close)
    out = np.full(n, np.nan)
    if n < period:
        return out
    k = 2.0 / (period + 1.0)
    seed = np.mean(close[:period])
    out[period - 1] = seed
    val = seed
    for i in range(period, n):
        val = close[i] * k + val * (1.0 - k)
        out[i] = val
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. SMA — Simple Moving Average
# ─────────────────────────────────────────────────────────────────────────────

def sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average using sliding window sum."""
    close = _c64(close)
    if USING_C:
        return _to_f64(_c_ext._sma(close, period))

    n = len(close)
    out = np.full(n, np.nan)
    if n < period:
        return out
    window_sum = np.sum(close[:period])
    out[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += close[i] - close[i - period]
        out[i] = window_sum / period
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. WMA — Weighted Moving Average
# ─────────────────────────────────────────────────────────────────────────────

def wma(close: np.ndarray, period: int) -> np.ndarray:
    """Linearly-weighted moving average."""
    close = _c64(close)
    if USING_C:
        return _to_f64(_c_ext._wma(close, period))

    n = len(close)
    out = np.full(n, np.nan)
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = weights.sum()
    for i in range(period - 1, n):
        out[i] = np.dot(close[i - period + 1 : i + 1], weights) / denom
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. HMA — Hull Moving Average
# ─────────────────────────────────────────────────────────────────────────────

def hma(close: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average: wma(2*wma(n/2) - wma(n), sqrt(n))."""
    close = _c64(close)
    if USING_C:
        return _to_f64(_c_ext._hma(close, period))

    half_p = period // 2
    sqrt_p = int(math.sqrt(period))
    diff = 2 * wma(close, half_p) - wma(close, period)
    return wma(diff, sqrt_p)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RSI — Relative Strength Index
# ─────────────────────────────────────────────────────────────────────────────

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI using Wilder's smoothed averages.

    Parameters
    ----------
    close  : 1-D float64 close prices
    period : RSI period (default 14)

    Returns
    -------
    RSI values [0, 100]. NaN for first `period` bars.
    """
    close = _c64(close)
    if USING_C:
        return _to_f64(_c_ext._rsi(close, period))

    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out

    diffs = np.diff(close)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss < 1e-10:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    inv_p = 1.0 / period
    for i in range(period + 1, n):
        avg_gain = avg_gain * (1.0 - inv_p) + gains[i - 1] * inv_p
        avg_loss = avg_loss * (1.0 - inv_p) + losses[i - 1] * inv_p
        if avg_loss < 1e-10:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. MACD
# ─────────────────────────────────────────────────────────────────────────────

def macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD line, signal line, histogram.

    Returns
    -------
    (macd_line, signal_line, histogram) — each same length as close
    """
    close = _c64(close)
    if USING_C:
        m_b, s_b, h_b = _c_ext._macd(close, fast, slow, signal)
        return _to_f64(m_b), _to_f64(s_b), _to_f64(h_b)

    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow

    # Signal: EMA of MACD
    valid_mask = ~np.isnan(macd_line)
    signal_line = np.full(len(close), np.nan)
    hist = np.full(len(close), np.nan)

    first_valid = np.argmax(valid_mask)
    if first_valid + signal <= len(close):
        sig_arr = ema(macd_line[first_valid:], signal)
        n_sig = len(sig_arr)
        signal_line[first_valid: first_valid + n_sig] = sig_arr
        valid_both = ~np.isnan(macd_line) & ~np.isnan(signal_line)
        hist[valid_both] = macd_line[valid_both] - signal_line[valid_both]

    return macd_line, signal_line, hist


# ─────────────────────────────────────────────────────────────────────────────
# 7. ATR — Average True Range
# ─────────────────────────────────────────────────────────────────────────────

def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range (Wilder's method)."""
    high = _c64(high); low = _c64(low); close = _c64(close)
    if USING_C:
        return _to_f64(_c_ext._atr(high, low, close, period))

    n = len(close)
    out = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))

    if n < period:
        return out
    seed = np.mean(tr[:period])
    out[period - 1] = seed
    inv_p = 1.0 / period
    val = seed
    for i in range(period, n):
        val = val * (1.0 - inv_p) + tr[i] * inv_p
        out[i] = val
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

def bollinger(
    close: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.

    Returns
    -------
    (upper, middle, lower) — each same length as close
    """
    close = _c64(close)
    if USING_C:
        u_b, m_b, l_b = _c_ext._bollinger(close, period, num_std)
        return _to_f64(u_b), _to_f64(m_b), _to_f64(l_b)

    n = len(close)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = close[i - period + 1: i + 1]
        m = np.mean(window)
        s = np.std(window, ddof=0)
        middle[i] = m
        upper[i]  = m + num_std * s
        lower[i]  = m - num_std * s
    return upper, middle, lower


# ─────────────────────────────────────────────────────────────────────────────
# 9. ADX — Average Directional Index
# ─────────────────────────────────────────────────────────────────────────────

def adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Directional Index (trend strength 0-100)."""
    high = _c64(high); low = _c64(low); close = _c64(close)
    if USING_C:
        return _to_f64(_c_ext._adx(high, low, close, period))

    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    # True range + directional movements
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        up   = high[i]   - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i]  = up   if (up > down and up > 0)   else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))

    # Wilder smooth
    inv_p = 1.0 / period
    sp = np.sum(plus_dm[1: period + 1])
    sm = np.sum(minus_dm[1: period + 1])
    st = np.sum(tr[1: period + 1])

    dx_series = []
    for i in range(period + 1, n):
        sp = sp * (1.0 - inv_p) + plus_dm[i]
        sm = sm * (1.0 - inv_p) + minus_dm[i]
        st = st * (1.0 - inv_p) + tr[i]
        pdi = sp / max(st, 1e-10) * 100.0
        mdi = sm / max(st, 1e-10) * 100.0
        dx  = abs(pdi - mdi) / max(pdi + mdi, 1e-10) * 100.0
        dx_series.append(dx)

    if len(dx_series) < period:
        return out

    adx_val = np.mean(dx_series[:period])
    out_start = 2 * period
    if out_start < n:
        out[out_start] = adx_val
    for k in range(period, len(dx_series)):
        adx_val = adx_val * (1.0 - inv_p) + dx_series[k] * inv_p
        idx = out_start + k - period + 1
        if idx < n:
            out[idx] = adx_val
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 10. Stochastic Oscillator
# ─────────────────────────────────────────────────────────────────────────────

def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic %K and %D."""
    high = _c64(high); low = _c64(low); close = _c64(close)
    if USING_C:
        k_b, d_b = _c_ext._stochastic(high, low, close, k_period, d_period)
        return _to_f64(k_b), _to_f64(d_b)

    n = len(close)
    k_out = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        lo = np.min(low[i - k_period + 1: i + 1])
        hi = np.max(high[i - k_period + 1: i + 1])
        rng = hi - lo
        k_out[i] = (close[i] - lo) / rng * 100.0 if rng > 1e-10 else 50.0
    d_out = sma(k_out, d_period)
    return k_out, d_out


# ─────────────────────────────────────────────────────────────────────────────
# 11. VWAP
# ─────────────────────────────────────────────────────────────────────────────

def vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Cumulative VWAP using typical price."""
    high = _c64(high); low = _c64(low); close = _c64(close); volume = _c64(volume)
    if USING_C:
        return _to_f64(_c_ext._vwap(high, low, close, volume))

    tp = (high + low + close) / 3.0
    vol = np.where(volume > 0, volume, 0.0)
    cum_tpv = np.cumsum(tp * vol)
    cum_v   = np.cumsum(vol)
    return np.where(cum_v > 1e-10, cum_tpv / cum_v, tp)


# ─────────────────────────────────────────────────────────────────────────────
# 12. OBV
# ─────────────────────────────────────────────────────────────────────────────

def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    close = _c64(close); volume = _c64(volume)
    if USING_C:
        return _to_f64(_c_ext._obv(close, volume))

    n = len(close)
    out = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 13. CCI
# ─────────────────────────────────────────────────────────────────────────────

def cci(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Commodity Channel Index."""
    high = _c64(high); low = _c64(low); close = _c64(close)
    if USING_C:
        return _to_f64(_c_ext._cci(high, low, close, period))

    n = len(close)
    out = np.full(n, np.nan)
    tp = (high + low + close) / 3.0
    for i in range(period - 1, n):
        tp_win = tp[i - period + 1: i + 1]
        m = np.mean(tp_win)
        mad = np.mean(np.abs(tp_win - m))
        out[i] = (tp[i] - m) / (0.015 * mad) if mad > 1e-10 else 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 14. ROC — Rate of Change
# ─────────────────────────────────────────────────────────────────────────────

def roc(close: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of Change percentage."""
    close = _c64(close)
    if USING_C:
        return _to_f64(_c_ext._roc(close, period))

    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period, n):
        prev = close[i - period]
        out[i] = (close[i] - prev) / prev * 100.0 if prev > 1e-10 else np.nan
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 15. BH Series
# ─────────────────────────────────────────────────────────────────────────────

def bh_series(
    closes: np.ndarray,
    cf: float = 0.003,
    bh_form: float = 0.20,
    bh_decay: float = 0.97,
    bh_collapse: float = 0.08,
    ctl_req: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BH physics over a price series.

    Returns
    -------
    (masses, active, ctl) — all same length as closes
      masses : float64 BH mass per bar
      active : int32  (0/1) BH active flag
      ctl    : int32  consecutive timelike count
    """
    closes = _c64(closes)
    if USING_C:
        m_b, a_b, c_b = _c_ext._bh_series(closes, cf, bh_form, bh_decay, bh_collapse, ctl_req)
        return _to_f64(m_b), _to_i32(a_b), _to_i32(c_b)

    # Numpy fallback
    n = len(closes)
    masses = np.zeros(n)
    active = np.zeros(n, dtype=np.int32)
    ctl_arr = np.zeros(n, dtype=np.int32)

    mass = 0.0; act = 0; ctl = 0
    prev = closes[0]

    for i in range(1, n):
        beta = abs(math.log(closes[i] / prev)) if prev > 0 else 0.0
        is_tl = beta < cf

        ctl = ctl + 1 if is_tl else 0

        dm = cf * 0.5 if is_tl else (beta - cf) * 2.0
        mass = mass * bh_decay + dm

        if not act:
            if mass >= bh_form and ctl >= ctl_req:
                act = 1
        else:
            if mass < bh_collapse:
                act = 0; mass = 0.0; ctl = 0

        masses[i] = mass
        active[i] = act
        ctl_arr[i] = ctl
        prev = closes[i]

    return masses, active, ctl_arr


# ─────────────────────────────────────────────────────────────────────────────
# 16. BH Backtest
# ─────────────────────────────────────────────────────────────────────────────

def bh_backtest(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    cf: float = 0.003,
    bh_form: float = 0.20,
    bh_decay: float = 0.97,
    bh_collapse: float = 0.08,
    ctl_req: int = 3,
    long_only: bool = False,
    commission: float = 0.0004,
    slippage: float = 0.0001,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Full BH backtest over price series.

    Returns
    -------
    (equity_curve, positions, n_trades)
      equity_curve : float64 array
      positions    : int32 array (+1, 0, -1)
      n_trades     : int
    """
    closes = _c64(closes); highs = _c64(highs); lows = _c64(lows)

    if USING_C:
        eq_b, pos_b, n_trades = _c_ext._bh_backtest(
            closes, highs, lows,
            cf, bh_form, bh_decay, bh_collapse,
            ctl_req, int(long_only), commission, slippage
        )
        return _to_f64(eq_b), _to_i32(pos_b), int(n_trades)

    # Numpy fallback
    n = len(closes)
    equity = np.ones(n)
    positions = np.zeros(n, dtype=np.int32)

    masses, active, ctl_arr = bh_series(
        closes, cf, bh_form, bh_decay, bh_collapse, ctl_req
    )

    cost = commission + slippage
    pos = 0
    entry_px = 0.0
    entry_dir = 0
    eq = 1.0
    prev_active = 0

    for i in range(1, n):
        newly_activated = int(active[i]) == 1 and int(active[i - 1]) == 0

        if pos != 0:
            bar_ret = math.log(closes[i] / closes[i - 1]) * pos
            eq *= math.exp(bar_ret)

        equity[i] = eq
        positions[i] = pos

        if pos == 0 and newly_activated:
            # Determine direction from recent price action
            direction = 1 if closes[i] > closes[i - 1] else -1
            if direction == 1:
                pos = 1
            elif not long_only:
                pos = -1

            if pos != 0:
                entry_px  = closes[i] * (1 + cost * pos)
                entry_dir = pos
                eq       *= (1 - cost)
                equity[i] = eq

        if pos != 0 and int(active[i]) == 0 and prev_active == 1:
            exit_px = closes[i] * (1 - cost * pos)
            eq *= (1 - cost)
            equity[i] = eq
            pos = 0

        prev_active = int(active[i])
        positions[i] = pos

    n_trades = sum(
        1 for i in range(1, n)
        if int(active[i]) == 0 and int(active[i - 1]) == 1
    )

    return equity, positions, n_trades


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: all indicators in one batch
# ─────────────────────────────────────────────────────────────────────────────

def all_indicators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all indicators in one call. Returns a dict of name → array.

    Useful for feature engineering.
    """
    if volume is None:
        volume = np.ones(len(close))

    return {
        "ema_12":        ema(close, 12),
        "ema_26":        ema(close, 26),
        "ema_50":        ema(close, 50),
        "ema_200":       ema(close, 200),
        "sma_20":        sma(close, 20),
        "sma_50":        sma(close, 50),
        "wma_14":        wma(close, 14),
        "hma_20":        hma(close, 20),
        "rsi_14":        rsi(close, 14),
        "rsi_9":         rsi(close, 9),
        "macd_line":     macd(close)[0],
        "macd_signal":   macd(close)[1],
        "macd_hist":     macd(close)[2],
        "atr_14":        atr(high, low, close, 14),
        "bb_upper":      bollinger(close, 20, 2.0)[0],
        "bb_middle":     bollinger(close, 20, 2.0)[1],
        "bb_lower":      bollinger(close, 20, 2.0)[2],
        "adx_14":        adx(high, low, close, 14),
        "stoch_k":       stochastic(high, low, close)[0],
        "stoch_d":       stochastic(high, low, close)[1],
        "vwap":          vwap(high, low, close, volume),
        "obv":           obv(close, volume),
        "cci_20":        cci(high, low, close, 20),
        "roc_10":        roc(close, 10),
    }


__all__ = [
    "ema", "sma", "wma", "hma", "rsi", "macd", "atr", "bollinger",
    "adx", "stochastic", "vwap", "obv", "cci", "roc",
    "bh_series", "bh_backtest", "all_indicators",
    "USING_C",
]
