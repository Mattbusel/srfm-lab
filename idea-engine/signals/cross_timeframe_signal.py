"""
Cross-timeframe signal synthesis — aligns signals across multiple timeframes.

Multi-timeframe alignment is one of the most robust signal filters:
  - When HTF trend, MTF momentum, and LTF entry all align → high conviction
  - Divergence between timeframes → reduce size or skip trade
  - Regime detection at each TF → full market picture

Implements:
  - Multi-TF trend alignment score
  - TF cascade: top-down from HTF to entry
  - Divergence score between TFs
  - TF-weighted composite signal
  - Timeframe volatility normalization
  - TF momentum spillover
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimeframeConfig:
    name: str
    multiplier: int          # relative to base TF (e.g., base=15m, HTF=96=1d)
    trend_lookback: int      # bars for trend
    momentum_lookback: int   # bars for momentum
    weight: float            # signal weight (HTF > MTF > LTF)


DEFAULT_TF_CONFIG = [
    TimeframeConfig("LTF", 1,   20, 5,  0.20),    # 15m
    TimeframeConfig("MTF", 4,   40, 10, 0.30),    # 1h
    TimeframeConfig("HTF", 16,  80, 20, 0.50),    # 4h
]


def resample_to_tf(prices: np.ndarray, multiplier: int) -> np.ndarray:
    """Resample prices to higher timeframe (take close of each bar group)."""
    n = len(prices) // multiplier
    if n == 0:
        return prices[-1:]
    return np.array([prices[i * multiplier: (i + 1) * multiplier][-1] for i in range(n)])


def trend_signal_at_tf(
    prices_tf: np.ndarray,
    lookback: int,
) -> float:
    """
    Trend signal at a given TF: positive = uptrend, negative = downtrend.
    Uses regression slope normalized by volatility.
    """
    n = min(len(prices_tf), lookback)
    if n < 3:
        return 0.0
    p = prices_tf[-n:]
    t = np.arange(n)
    slope = float(np.polyfit(t, p, 1)[0])
    vol = float(p.std())
    return float(math.tanh(slope / max(vol, 1e-10) * n * 0.1))


def momentum_signal_at_tf(
    prices_tf: np.ndarray,
    lookback: int,
) -> float:
    """Rate-of-change momentum at a given timeframe."""
    n = min(len(prices_tf), lookback + 1)
    if n < 2:
        return 0.0
    roc = float((prices_tf[-1] - prices_tf[-n]) / max(abs(prices_tf[-n]), 1e-10))
    return float(math.tanh(roc * 5))


def mean_reversion_signal_at_tf(
    prices_tf: np.ndarray,
    lookback: int,
) -> float:
    """Z-score based mean reversion signal at a given timeframe."""
    n = min(len(prices_tf), lookback)
    if n < 5:
        return 0.0
    sub = prices_tf[-n:]
    z = float((prices_tf[-1] - sub.mean()) / max(sub.std(), 1e-10))
    return float(-math.tanh(z))  # negative: fade extremes


@dataclass
class MultiTFSignal:
    alignment_score: float         # -1 to +1 (all agree → ±1)
    divergence_score: float        # 0 to 1 (0 = aligned, 1 = max divergence)
    composite: float               # final combined signal
    htf_bias: float                # HTF directional bias
    mtf_momentum: float            # MTF momentum
    ltf_entry: float               # LTF entry quality
    conviction: float              # overall conviction
    tf_signals: dict[str, float]   # signal per timeframe
    recommendation: str            # ENTER_LONG, ENTER_SHORT, WAIT, EXIT


def multi_timeframe_signal(
    prices: np.ndarray,
    tf_configs: Optional[list[TimeframeConfig]] = None,
    signal_type: str = "trend",
) -> MultiTFSignal:
    """
    Compute multi-timeframe aligned signal.

    signal_type: "trend", "mean_reversion", "momentum"
    """
    if tf_configs is None:
        tf_configs = DEFAULT_TF_CONFIG

    T = len(prices)
    tf_signals = {}
    total_weight = sum(c.weight for c in tf_configs)

    for cfg in tf_configs:
        prices_tf = resample_to_tf(prices, cfg.multiplier)
        if len(prices_tf) < 3:
            tf_signals[cfg.name] = 0.0
            continue

        if signal_type == "trend":
            sig = trend_signal_at_tf(prices_tf, cfg.trend_lookback)
        elif signal_type == "mean_reversion":
            sig = mean_reversion_signal_at_tf(prices_tf, cfg.momentum_lookback)
        elif signal_type == "momentum":
            sig = momentum_signal_at_tf(prices_tf, cfg.momentum_lookback)
        else:
            sig = trend_signal_at_tf(prices_tf, cfg.trend_lookback)

        tf_signals[cfg.name] = float(sig)

    signals = np.array(list(tf_signals.values()))
    weights = np.array([c.weight / total_weight for c in tf_configs])

    # Composite: weighted average
    composite = float(np.sum(signals * weights))

    # Alignment: how much do they agree?
    if len(signals) > 0 and signals.std() > 1e-10:
        # Agreement = correlation across timeframes
        signs = np.sign(signals[signals != 0])
        if len(signs) > 0:
            alignment = float(signs.mean())  # +1 = all up, -1 = all down, 0 = mixed
        else:
            alignment = 0.0
    else:
        alignment = float(np.sign(composite))

    # Divergence: std of normalized signals
    divergence = float(signals.std())

    # Extract individual TF signals
    htf_name = tf_configs[-1].name if tf_configs else "HTF"
    mtf_name = tf_configs[len(tf_configs)//2].name if len(tf_configs) > 1 else "MTF"
    ltf_name = tf_configs[0].name if tf_configs else "LTF"

    htf_bias = tf_signals.get(htf_name, 0.0)
    mtf_momentum = tf_signals.get(mtf_name, 0.0)
    ltf_entry = tf_signals.get(ltf_name, 0.0)

    # Conviction: alignment * magnitude
    conviction = float(abs(alignment) * abs(composite))

    # Recommendation
    if abs(composite) < 0.15 or divergence > 0.6:
        rec = "WAIT"
    elif composite > 0.3 and alignment > 0.3 and htf_bias > 0:
        rec = "ENTER_LONG"
    elif composite < -0.3 and alignment < -0.3 and htf_bias < 0:
        rec = "ENTER_SHORT"
    elif abs(composite) < 0.1:
        rec = "EXIT"
    else:
        rec = "WAIT"

    return MultiTFSignal(
        alignment_score=alignment,
        divergence_score=divergence,
        composite=composite,
        htf_bias=htf_bias,
        mtf_momentum=mtf_momentum,
        ltf_entry=ltf_entry,
        conviction=conviction,
        tf_signals=tf_signals,
        recommendation=rec,
    )


def tf_momentum_spillover(
    prices: np.ndarray,
    spillover_lag: int = 5,
    tf_multipliers: Optional[list[int]] = None,
) -> dict:
    """
    Detect momentum spillover between timeframes.
    HTF momentum spills to LTF with lag → predict LTF direction.
    """
    if tf_multipliers is None:
        tf_multipliers = [1, 4, 16]

    T = len(prices)
    spillover_signals = {}

    htf_prices = resample_to_tf(prices, tf_multipliers[-1])
    htf_returns = np.diff(np.log(htf_prices + 1e-10))

    for mult in tf_multipliers[:-1]:
        ltf_prices = resample_to_tf(prices, mult)
        ltf_returns = np.diff(np.log(ltf_prices + 1e-10))

        # Ratio of lengths
        ratio = tf_multipliers[-1] // mult
        n = min(len(htf_returns), len(ltf_returns) // ratio)

        if n < spillover_lag + 2:
            spillover_signals[mult] = 0.0
            continue

        # Lagged correlation: htf[t-lag] → ltf[t]
        htf_recent = htf_returns[-n:]
        ltf_chunks = np.array([
            ltf_returns[i * ratio: (i + 1) * ratio].mean()
            for i in range(n)
        ])

        if len(ltf_chunks) > spillover_lag:
            corr = float(np.corrcoef(
                htf_recent[-len(ltf_chunks) + spillover_lag:],
                ltf_chunks[spillover_lag:]
            )[0, 1])
            current_htf = float(htf_returns[-spillover_lag:].mean())
            spillover_signals[mult] = float(corr * math.tanh(current_htf * 10))

    return {
        "spillover_signals": spillover_signals,
        "htf_momentum": float(htf_returns[-5:].mean()) if len(htf_returns) >= 5 else 0.0,
        "spillover_strength": float(abs(np.mean(list(spillover_signals.values())))),
    }


def tf_convergence_entry(
    prices: np.ndarray,
    tf_configs: Optional[list[TimeframeConfig]] = None,
    required_alignment: float = 0.6,
) -> dict:
    """
    Find optimal entry timing using TF convergence.
    Returns entry signal when multiple TF signals align at same time.
    """
    if tf_configs is None:
        tf_configs = DEFAULT_TF_CONFIG

    trend_signal = multi_timeframe_signal(prices, tf_configs, "trend")
    mom_signal = multi_timeframe_signal(prices, tf_configs, "momentum")

    # Best entry: trend and momentum agree, divergence low
    trend_agree = abs(trend_signal.alignment_score) >= required_alignment
    mom_agree = abs(mom_signal.alignment_score) >= required_alignment
    direction_match = float(np.sign(trend_signal.composite) == np.sign(mom_signal.composite))

    entry_quality = float(
        (abs(trend_signal.composite) + abs(mom_signal.composite)) / 2
        * direction_match
        * (1 - trend_signal.divergence_score * 0.5)
    )

    direction = float(np.sign(trend_signal.composite + mom_signal.composite))

    return {
        "entry_quality": float(min(entry_quality, 1.0)),
        "direction": direction,
        "trend_aligned": trend_agree,
        "momentum_aligned": mom_agree,
        "should_enter": bool(entry_quality > 0.35 and trend_agree),
        "htf_trend": trend_signal.htf_bias,
        "ltf_entry": trend_signal.ltf_entry,
        "conviction": float(trend_signal.conviction * mom_signal.conviction * direction_match),
    }
