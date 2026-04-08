"""
Cross-asset momentum and spillover signals.

Captures momentum transmission across asset classes and within crypto:
  - Time-series momentum (TSMOM) per asset
  - Cross-sectional momentum ranking
  - Momentum spillover (lead-lag detection)
  - Idiosyncratic vs systematic momentum decomposition
  - Momentum crash risk (crowding indicator)
  - Reversal signals (overbought/oversold cross-asset)
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def tsmom_signal(
    prices: np.ndarray,
    lookback: int = 60,
    vol_scale: bool = True,
) -> float:
    """
    Time-series momentum signal: sign(past_return) / recent_vol.
    Returns risk-adjusted position size [-1, +1].
    """
    if len(prices) < lookback + 1:
        return 0.0
    ret_lb = (prices[-1] / prices[-lookback] - 1)
    if vol_scale:
        r = np.diff(np.log(prices[-lookback:]))
        sigma = float(r.std() * (252 ** 0.5)) + 1e-6
        return float(np.sign(ret_lb) * min(abs(ret_lb) / sigma, 1.0))
    return float(np.clip(ret_lb * 10, -1, 1))


def cross_sectional_momentum(
    prices: dict[str, np.ndarray],
    lookback: int = 20,
    n_long: int = 3,
    n_short: int = 3,
) -> dict[str, float]:
    """
    Cross-sectional momentum: rank assets by past return, long top / short bottom.
    Returns dict of signals in [-1, +1].
    """
    returns = {}
    for sym, px in prices.items():
        if len(px) >= lookback + 1:
            returns[sym] = px[-1] / px[-lookback] - 1
        else:
            returns[sym] = 0.0

    sorted_syms = sorted(returns, key=lambda s: returns[s])
    n = len(sorted_syms)
    signals = {sym: 0.0 for sym in prices}

    for i, sym in enumerate(sorted_syms):
        if i < n_short:
            signals[sym] = -1.0
        elif i >= n - n_long:
            signals[sym] = 1.0

    return signals


def momentum_spillover_signal(
    source_returns: np.ndarray,
    target_returns: np.ndarray,
    lag: int = 1,
    window: int = 30,
) -> np.ndarray:
    """
    Detect momentum spillover: source leads target by `lag` bars.
    Returns rolling cross-correlation as signal (-1 to +1).
    """
    n = len(target_returns)
    signal = np.zeros(n)
    for i in range(window + lag, n):
        s = source_returns[i - window - lag: i - lag]
        t = target_returns[i - window: i]
        corr = float(np.corrcoef(s, t)[0, 1]) if len(s) == len(t) else 0.0
        signal[i] = corr
    return signal


def idiosyncratic_momentum(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Idiosyncratic momentum = cumulative residual after removing market factor.
    Assets with high idio-momentum tend to continue independently of market.
    """
    n = len(asset_returns)
    idio = np.zeros(n)
    for i in range(window, n):
        r_a = asset_returns[i - window: i]
        r_m = market_returns[i - window: i]
        # OLS beta
        cov = np.cov(r_a, r_m)
        beta = cov[0, 1] / max(cov[1, 1], 1e-10)
        residuals = r_a - beta * r_m
        idio[i] = float(residuals.sum())
    return idio


def momentum_crash_risk(
    returns_matrix: np.ndarray,
    lookback: int = 60,
    crowding_threshold: float = 0.7,
) -> dict:
    """
    Estimate momentum crash risk from cross-sectional correlation of returns.
    High average correlation → crowded momentum trade → crash risk.

    returns_matrix: shape (T, N)
    """
    if returns_matrix.shape[0] < lookback:
        return {"crash_risk": 0.0, "avg_correlation": 0.0, "crowded": False}

    recent = returns_matrix[-lookback:]
    C = np.corrcoef(recent.T)
    np.fill_diagonal(C, 0.0)
    n = C.shape[0]
    avg_corr = float(C.sum() / max(n * (n - 1), 1))

    return {
        "crash_risk": float(max(avg_corr - crowding_threshold, 0) / (1 - crowding_threshold)),
        "avg_correlation": avg_corr,
        "crowded": bool(avg_corr > crowding_threshold),
        "n_assets": n,
    }
