"""
signals/order_flow_signal.py

Sophisticated order flow analysis signal generator.

Computes:
  - Cumulative delta (volume delta running sum)
  - Delta divergence (price up but delta down = bearish)
  - Footprint POC (point of control from per-bar volume profile)
  - Delta imbalance ratio per bar
  - Stacked imbalances (3+ consecutive bars same side)
  - CVD trend signal (regression slope of cumulative volume delta)
  - Exhaustion detection (high volume + small price move = absorption)
  - Delta-weighted RSI
  - Composite recommendation

All arrays are numpy-based.  The public entry point is compute_order_flow_signal().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OFDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ExhaustionType(str, Enum):
    BUYING_EXHAUSTION = "buying_exhaustion"
    SELLING_EXHAUSTION = "selling_exhaustion"
    NONE = "none"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrderFlowSignal:
    """
    Complete order flow analysis output for a price/volume series.

    All scalar values refer to the most recent bar unless documented otherwise.
    Arrays have length == n_bars.
    """

    # --- raw components ---
    cumulative_delta: np.ndarray          # running sum of (buy_vol - sell_vol)
    delta_per_bar: np.ndarray             # buy_vol[i] - sell_vol[i]
    buy_volume: np.ndarray
    sell_volume: np.ndarray

    # --- divergence ---
    delta_divergence: float               # +1 bullish div, -1 bearish div, 0 none
    divergence_bars: int                  # how many bars back divergence started
    divergence_strength: float            # normalized 0-1

    # --- footprint POC ---
    poc_price: float                      # volume-weighted POC of last N bars
    poc_distance_pct: float               # % distance from current price to POC
    poc_side: OFDirection                 # is price above or below POC?

    # --- imbalance ---
    delta_imbalance_ratio: np.ndarray     # abs(delta) / total_volume per bar
    current_imbalance: float              # most recent bar
    stacked_imbalances: int               # count of consecutive same-side bars
    stacked_direction: OFDirection        # direction of stack

    # --- CVD trend ---
    cvd_slope: float                      # regression slope (normalized)
    cvd_acceleration: float               # second derivative (change in slope)
    cvd_trend_signal: float               # -1 to +1

    # --- exhaustion ---
    exhaustion: ExhaustionType
    exhaustion_score: float               # 0-1 intensity
    absorption_bars: list[int]            # bar indices where absorption detected

    # --- delta-weighted RSI ---
    dw_rsi: float                         # 0-100, delta-weighted RSI
    dw_rsi_signal: float                  # overbought/oversold: +1 OS, -1 OB

    # --- composite ---
    composite_score: float                # -1 (strong bear) to +1 (strong bull)
    recommendation: OFDirection
    confidence: float                     # 0-1
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_buy_sell_volume(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate buy and sell volume per bar using the Tick Rule approximation.

    buy_ratio = (close - low) / (high - low + 1e-12)
    buy_vol   = volume * buy_ratio
    sell_vol  = volume * (1 - buy_ratio)

    This is the standard proxy when trade-level data is unavailable.
    """
    bar_range = high - low
    safe_range = np.where(bar_range < 1e-12, 1e-12, bar_range)
    buy_ratio = np.clip((close - low) / safe_range, 0.0, 1.0)
    buy_vol = volume * buy_ratio
    sell_vol = volume * (1.0 - buy_ratio)
    return buy_vol, sell_vol


def _cumulative_delta(buy_vol: np.ndarray, sell_vol: np.ndarray) -> np.ndarray:
    """Running sum of (buy - sell) volume."""
    return np.cumsum(buy_vol - sell_vol)


def _detect_delta_divergence(
    prices: np.ndarray,
    cvd: np.ndarray,
    lookback: int = 20,
) -> tuple[float, int, float]:
    """
    Detect divergence between price direction and CVD direction.

    Returns (divergence_signal, bars_back, strength):
      +1 = bullish divergence (price down, CVD up)
      -1 = bearish divergence (price up, CVD down)
       0 = no divergence
    """
    n = min(len(prices), lookback)
    if n < 5:
        return 0.0, 0, 0.0

    p = prices[-n:]
    c = cvd[-n:]

    price_slope, _, price_r, _, _ = sp_stats.linregress(np.arange(n), p)
    cvd_slope, _, cvd_r, _, _ = sp_stats.linregress(np.arange(n), c)

    # Normalize slopes by their standard deviations
    p_std = np.std(p) + 1e-12
    c_std = np.std(c) + 1e-12
    norm_price_slope = price_slope / p_std
    norm_cvd_slope = cvd_slope / c_std

    # Divergence: slopes have opposite signs
    diverging = (norm_price_slope * norm_cvd_slope) < 0
    if not diverging:
        return 0.0, 0, 0.0

    strength = min(1.0, abs(norm_price_slope - norm_cvd_slope) / 2.0)
    direction = -1.0 if norm_price_slope > 0 else 1.0  # price up, cvd down = bearish
    return direction, n, strength


def _footprint_poc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 50,
    n_levels: int = 100,
) -> tuple[float, float, OFDirection]:
    """
    Volume profile POC over last `lookback` bars.

    Distributes each bar's volume uniformly across its high-low range into
    `n_levels` price buckets, then finds the bucket with highest accumulated volume.
    """
    h = high[-lookback:]
    l = low[-lookback:]
    v = volume[-lookback:]

    price_min = float(np.min(l))
    price_max = float(np.max(h))
    if price_max <= price_min:
        poc = float(close[-1])
        return poc, 0.0, OFDirection.NEUTRAL

    bucket_width = (price_max - price_min) / n_levels
    buckets = np.zeros(n_levels, dtype=np.float64)

    for i in range(len(h)):
        bar_min = l[i]
        bar_max = h[i]
        bar_vol = v[i]
        lo_idx = int((bar_min - price_min) / bucket_width)
        hi_idx = int((bar_max - price_min) / bucket_width)
        lo_idx = max(0, min(lo_idx, n_levels - 1))
        hi_idx = max(0, min(hi_idx, n_levels - 1))
        n_touched = hi_idx - lo_idx + 1
        vol_per_level = bar_vol / max(n_touched, 1)
        buckets[lo_idx: hi_idx + 1] += vol_per_level

    poc_idx = int(np.argmax(buckets))
    poc_price = price_min + (poc_idx + 0.5) * bucket_width
    current_price = float(close[-1])
    distance_pct = (current_price - poc_price) / (poc_price + 1e-12) * 100.0
    side = OFDirection.BULLISH if current_price > poc_price else OFDirection.BEARISH
    return poc_price, distance_pct, side


def _stacked_imbalances(
    delta_per_bar: np.ndarray,
    min_stack: int = 3,
) -> tuple[int, OFDirection]:
    """
    Count consecutive bars with the same delta sign at the tail of the series.

    Returns (count, direction).  If count < min_stack, direction = NEUTRAL.
    """
    n = len(delta_per_bar)
    if n == 0:
        return 0, OFDirection.NEUTRAL

    last_sign = np.sign(delta_per_bar[-1])
    if last_sign == 0:
        return 0, OFDirection.NEUTRAL

    count = 0
    for i in range(n - 1, -1, -1):
        if np.sign(delta_per_bar[i]) == last_sign:
            count += 1
        else:
            break

    if count < min_stack:
        return count, OFDirection.NEUTRAL

    direction = OFDirection.BULLISH if last_sign > 0 else OFDirection.BEARISH
    return count, direction


def _cvd_trend(
    cvd: np.ndarray,
    lookback: int = 30,
) -> tuple[float, float, float]:
    """
    Compute CVD trend via linear regression slope.

    Returns (normalized_slope, acceleration, signal).
    Signal in [-1, +1].
    """
    n = min(len(cvd), lookback)
    if n < 4:
        return 0.0, 0.0, 0.0

    c = cvd[-n:]
    x = np.arange(n, dtype=np.float64)
    slope, intercept, r, p, se = sp_stats.linregress(x, c)

    # Normalize slope by total CVD range
    cvd_range = np.ptp(c) + 1e-12
    norm_slope = float(np.clip(slope * n / cvd_range, -1.0, 1.0))

    # Acceleration: compare slope of first half vs second half
    half = n // 2
    if half >= 2:
        s1, *_ = sp_stats.linregress(x[:half], c[:half])
        s2, *_ = sp_stats.linregress(x[half:], c[half:])
        accel = float(np.clip((s2 - s1) / (cvd_range / n + 1e-12), -1.0, 1.0))
    else:
        accel = 0.0

    return norm_slope, accel, norm_slope


def _detect_exhaustion(
    close: np.ndarray,
    volume: np.ndarray,
    delta_per_bar: np.ndarray,
    vol_threshold_pct: float = 80.0,
    move_threshold_pct: float = 20.0,
    lookback: int = 50,
) -> tuple[ExhaustionType, float, list[int]]:
    """
    Detect absorption/exhaustion: high volume bar with small price move.

    High volume is defined as > vol_threshold_pct percentile.
    Small move is < move_threshold_pct percentile of absolute bar returns.
    """
    n = min(len(close), lookback)
    vols = volume[-n:]
    moves = np.abs(np.diff(close[-n - 1:])) if len(close) > n else np.abs(np.diff(close))

    if len(moves) < 4:
        return ExhaustionType.NONE, 0.0, []

    vol_thresh = np.percentile(vols[:-1] if len(vols) > len(moves) else vols, vol_threshold_pct)
    move_thresh = np.percentile(moves, move_threshold_pct)

    absorption_indices: list[int] = []
    for i in range(len(moves)):
        bar_vol = vols[i] if i < len(vols) else 0.0
        bar_move = moves[i]
        if bar_vol > vol_thresh and bar_move < move_thresh:
            absorption_indices.append(len(close) - n + i)

    if not absorption_indices:
        return ExhaustionType.NONE, 0.0, []

    # Most recent absorption bar
    recent_idx = absorption_indices[-1]
    # Direction of exhaustion = delta sign at that bar
    abs_bar_delta = delta_per_bar[min(recent_idx, len(delta_per_bar) - 1)]
    ex_type = (ExhaustionType.BUYING_EXHAUSTION if abs_bar_delta > 0
               else ExhaustionType.SELLING_EXHAUSTION)

    # Score: proportion of lookback bars that are absorbing
    score = float(np.clip(len(absorption_indices) / max(n * 0.1, 1), 0.0, 1.0))
    return ex_type, score, absorption_indices


def _delta_weighted_rsi(
    close: np.ndarray,
    delta_per_bar: np.ndarray,
    period: int = 14,
) -> tuple[float, float]:
    """
    RSI where each bar's gain/loss is weighted by the absolute delta magnitude.

    Bars with strong delta agreement (strong buy delta on up-close) get more weight.
    Returns (dw_rsi, signal) where signal = +1 (oversold), -1 (overbought), 0 (neutral).
    """
    n = min(len(close), period * 3)
    if n < period + 1:
        return 50.0, 0.0

    p = close[-n:]
    d = np.abs(delta_per_bar[-n:])
    # Normalize delta weights
    d_sum = np.sum(d) + 1e-12
    d_w = d / d_sum * len(d)  # scale so average weight = 1

    returns = np.diff(p)
    weights = d_w[1:]  # align with returns

    gains = np.where(returns > 0, returns * weights, 0.0)
    losses = np.where(returns < 0, -returns * weights, 0.0)

    # Wilder smoothing over last `period` bars
    if len(gains) < period:
        return 50.0, 0.0

    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))

    if avg_loss < 1e-12:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    signal = 0.0
    if rsi < 30:
        signal = 1.0  # oversold = bullish
    elif rsi > 70:
        signal = -1.0  # overbought = bearish

    return rsi, signal


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_order_flow_signal(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    buy_volume: Optional[np.ndarray] = None,
    sell_volume: Optional[np.ndarray] = None,
    lookback_divergence: int = 20,
    lookback_cvd: int = 30,
    lookback_poc: int = 50,
    lookback_exhaustion: int = 50,
    rsi_period: int = 14,
    min_stack: int = 3,
) -> OrderFlowSignal:
    """
    Compute the full OrderFlowSignal from OHLCV bars.

    Parameters
    ----------
    open_, high, low, close, volume : np.ndarray
        Bar-level OHLCV data, oldest-first.
    buy_volume, sell_volume : optional
        If provided (e.g., from footprint/DOM data), used directly.
        Otherwise estimated from bar structure.
    """
    if len(close) < 5:
        raise ValueError("Need at least 5 bars to compute OrderFlowSignal.")

    # --- buy/sell split ---
    if buy_volume is None or sell_volume is None:
        bvol, svol = _estimate_buy_sell_volume(open_, high, low, close, volume)
    else:
        bvol = buy_volume.astype(np.float64)
        svol = sell_volume.astype(np.float64)

    delta = bvol - svol
    cvd = _cumulative_delta(bvol, svol)

    # --- imbalance ratio ---
    total_vol = bvol + svol + 1e-12
    imbalance_ratio = np.abs(delta) / total_vol
    current_imbalance = float(imbalance_ratio[-1])

    # --- stacked imbalances ---
    stack_count, stack_dir = _stacked_imbalances(delta, min_stack=min_stack)

    # --- divergence ---
    div_signal, div_bars, div_strength = _detect_delta_divergence(
        close, cvd, lookback=lookback_divergence
    )

    # --- footprint POC ---
    poc_price, poc_dist_pct, poc_side = _footprint_poc(
        high, low, close, volume, lookback=lookback_poc
    )

    # --- CVD trend ---
    cvd_slope, cvd_accel, cvd_trend = _cvd_trend(cvd, lookback=lookback_cvd)

    # --- exhaustion ---
    ex_type, ex_score, abs_bars = _detect_exhaustion(
        close, volume, delta, lookback=lookback_exhaustion
    )

    # --- delta-weighted RSI ---
    dw_rsi, dw_rsi_sig = _delta_weighted_rsi(close, delta, period=rsi_period)

    # --- composite scoring ---
    # Weights: CVD trend (0.25), divergence (0.20), stack (0.20), RSI (0.15),
    #          POC position (0.10), exhaustion (0.10)
    warnings: list[str] = []

    cvd_component = cvd_trend * 0.25

    div_component = float(div_signal) * div_strength * 0.20

    # Stack: normalized by length (cap at 7)
    stack_val = 0.0
    if stack_dir == OFDirection.BULLISH:
        stack_val = min(stack_count / 7.0, 1.0)
    elif stack_dir == OFDirection.BEARISH:
        stack_val = -min(stack_count / 7.0, 1.0)
    stack_component = stack_val * 0.20

    rsi_component = dw_rsi_sig * 0.15

    poc_component = float(np.clip(poc_dist_pct / 5.0, -1.0, 1.0)) * 0.10

    ex_component = 0.0
    if ex_type == ExhaustionType.BUYING_EXHAUSTION:
        ex_component = -ex_score * 0.10  # bearish signal
    elif ex_type == ExhaustionType.SELLING_EXHAUSTION:
        ex_component = ex_score * 0.10

    composite = float(np.clip(
        cvd_component + div_component + stack_component
        + rsi_component + poc_component + ex_component,
        -1.0, 1.0,
    ))

    # Confidence: derived from agreement of sub-signals
    sub_signs = [
        np.sign(cvd_component),
        np.sign(div_component) if div_component != 0 else 0,
        np.sign(stack_component) if stack_component != 0 else 0,
        np.sign(rsi_component) if rsi_component != 0 else 0,
        np.sign(poc_component) if poc_component != 0 else 0,
    ]
    composite_sign = np.sign(composite)
    agreements = sum(1 for s in sub_signs if s == composite_sign and s != 0)
    total_active = sum(1 for s in sub_signs if s != 0)
    confidence = float(agreements / max(total_active, 1))

    if abs(composite) < 0.1:
        recommendation = OFDirection.NEUTRAL
    elif composite > 0:
        recommendation = OFDirection.BULLISH
    else:
        recommendation = OFDirection.BEARISH

    if ex_type != ExhaustionType.NONE:
        warnings.append(f"{ex_type.value} detected — high volume absorption near {close[-1]:.4f}")
    if abs(div_signal) > 0 and div_strength > 0.5:
        warnings.append(f"Strong delta divergence ({div_signal:+.0f}) over {div_bars} bars")
    if stack_count >= 5:
        warnings.append(f"Extreme stacked imbalances: {stack_count} bars ({stack_dir.value})")

    return OrderFlowSignal(
        cumulative_delta=cvd,
        delta_per_bar=delta,
        buy_volume=bvol,
        sell_volume=svol,
        delta_divergence=float(div_signal),
        divergence_bars=div_bars,
        divergence_strength=div_strength,
        poc_price=poc_price,
        poc_distance_pct=poc_dist_pct,
        poc_side=poc_side,
        delta_imbalance_ratio=imbalance_ratio,
        current_imbalance=current_imbalance,
        stacked_imbalances=stack_count,
        stacked_direction=stack_dir,
        cvd_slope=cvd_slope,
        cvd_acceleration=cvd_accel,
        cvd_trend_signal=cvd_trend,
        exhaustion=ex_type,
        exhaustion_score=ex_score,
        absorption_bars=abs_bars,
        dw_rsi=dw_rsi,
        dw_rsi_signal=dw_rsi_sig,
        composite_score=composite,
        recommendation=recommendation,
        confidence=confidence,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 200
    prices = np.cumsum(rng.normal(0, 1, n)) + 100.0
    prices = np.clip(prices, 10, None)
    volume = rng.uniform(1000, 5000, n)
    high = prices + rng.uniform(0, 0.5, n)
    low = prices - rng.uniform(0, 0.5, n)
    open_ = prices + rng.normal(0, 0.2, n)

    sig = compute_order_flow_signal(open_, high, low, prices, volume)
    print(f"Recommendation : {sig.recommendation.value}")
    print(f"Composite score: {sig.composite_score:+.4f}")
    print(f"Confidence     : {sig.confidence:.3f}")
    print(f"CVD slope      : {sig.cvd_slope:+.4f}")
    print(f"DW-RSI         : {sig.dw_rsi:.2f}")
    print(f"Stacked        : {sig.stacked_imbalances} bars ({sig.stacked_direction.value})")
    print(f"POC            : {sig.poc_price:.4f} ({sig.poc_distance_pct:+.2f}%)")
    print(f"Exhaustion     : {sig.exhaustion.value} ({sig.exhaustion_score:.3f})")
    if sig.warnings:
        print("Warnings:")
        for w in sig.warnings:
            print(f"  - {w}")
