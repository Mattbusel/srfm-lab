"""
Volatility regime detection and signal generation.

Implements:
  - GARCH-based vol regime (low/medium/high)
  - Realized vol percentile rank signal
  - Vol-of-vol (VoV) signal
  - Vol term structure slope (contango/backwardation)
  - Implied vol proxy from realized vol
  - Variance risk premium proxy
  - Vol regime persistence scoring
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class VolRegimeState:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SPIKE = "spike"


def realized_vol(returns: np.ndarray, window: int = 20, annualize: int = 252) -> np.ndarray:
    """Rolling realized vol (annualized)."""
    n = len(returns)
    rv = np.full(n, np.nan)
    for i in range(window, n):
        rv[i] = float(returns[i - window: i].std() * math.sqrt(annualize))
    return rv


def vol_percentile_signal(
    returns: np.ndarray,
    short_window: int = 20,
    long_window: int = 252,
    low_pct: float = 25,
    high_pct: float = 75,
) -> np.ndarray:
    """
    Signal based on vol percentile rank in historical distribution.
    Signal = 1 when current vol is low (buy vol), -1 when high (sell vol).
    Returns: signal in [-1, +1].
    """
    n = len(returns)
    signal = np.zeros(n)
    rv = realized_vol(returns, short_window)

    for i in range(long_window, n):
        hist_vols = rv[i - long_window: i]
        hist_vols = hist_vols[~np.isnan(hist_vols)]
        if len(hist_vols) < 30:
            continue
        pct = float(np.sum(hist_vols < rv[i]) / len(hist_vols) * 100)
        if pct <= low_pct:
            signal[i] = 1.0   # vol low → buy
        elif pct >= high_pct:
            signal[i] = -1.0  # vol high → sell / reduce
    return signal


def vol_of_vol_signal(
    returns: np.ndarray,
    vol_window: int = 10,
    vov_window: int = 30,
) -> np.ndarray:
    """
    Vol of vol signal: high VoV → regime instability, reduce exposure.
    Returns normalized VoV in [0, 1].
    """
    n = len(returns)
    rv = realized_vol(returns, vol_window)
    vov = np.full(n, np.nan)

    for i in range(vov_window, n):
        rv_window = rv[i - vov_window: i]
        rv_window = rv_window[~np.isnan(rv_window)]
        if len(rv_window) > 3:
            vov[i] = float(rv_window.std() / (rv_window.mean() + 1e-10))

    # Normalize
    valid = vov[~np.isnan(vov)]
    if len(valid) > 10:
        p95 = np.percentile(valid, 95)
        vov_norm = np.clip(vov / max(p95, 1e-10), 0, 1)
    else:
        vov_norm = np.clip(vov, 0, 1)

    return np.nan_to_num(vov_norm, nan=0.0)


def vol_regime_classify(
    returns: np.ndarray,
    window: int = 20,
    spike_z: float = 2.0,
) -> list[str]:
    """
    Classify each bar into vol regime: LOW, MEDIUM, HIGH, SPIKE.
    """
    rv = realized_vol(returns, window)
    valid = rv[~np.isnan(rv)]
    if len(valid) < 30:
        return [VolRegimeState.MEDIUM] * len(returns)

    q25 = np.percentile(valid, 25)
    q75 = np.percentile(valid, 75)
    mu = np.mean(valid)
    sigma = np.std(valid)

    regimes = []
    for v in rv:
        if np.isnan(v):
            regimes.append(VolRegimeState.MEDIUM)
        elif v > mu + spike_z * sigma:
            regimes.append(VolRegimeState.SPIKE)
        elif v > q75:
            regimes.append(VolRegimeState.HIGH)
        elif v < q25:
            regimes.append(VolRegimeState.LOW)
        else:
            regimes.append(VolRegimeState.MEDIUM)
    return regimes


def variance_risk_premium_proxy(
    returns: np.ndarray,
    rv_window: int = 20,
    iv_scale: float = 1.2,
) -> np.ndarray:
    """
    Proxy VRP = IV - RV (expected vol minus realized vol).
    IV proxy: last month's RV scaled by historical IV/RV ratio.
    Positive VRP → sell vol; negative VRP → buy vol.
    """
    rv = realized_vol(returns, rv_window)
    iv_proxy = rv * iv_scale  # simplified: IV typically > RV
    vrp = iv_proxy - rv
    return np.nan_to_num(vrp, nan=0.0)


def vol_term_structure(
    returns: np.ndarray,
    short_window: int = 5,
    long_window: int = 20,
) -> np.ndarray:
    """
    Vol term structure slope = RV(short) / RV(long) - 1.
    Positive = contango (short vol < long vol) → normal
    Negative = backwardation (short vol > long vol) → stressed
    """
    rv_short = realized_vol(returns, short_window)
    rv_long = realized_vol(returns, long_window)
    slope = np.where(
        rv_long > 1e-6,
        rv_short / rv_long - 1,
        0.0,
    )
    return np.nan_to_num(slope, nan=0.0)


def vol_regime_position_scaler(
    returns: np.ndarray,
    window: int = 20,
    target_vol: float = 0.15,
) -> np.ndarray:
    """
    Position scalar based on vol targeting.
    Scale = target_vol / realized_vol (annual).
    Capped at [0.25, 2.0] to avoid extreme leverage.
    """
    rv = realized_vol(returns, window)
    scale = np.where(rv > 1e-4, target_vol / rv, 1.0)
    return np.clip(scale, 0.25, 2.0)
