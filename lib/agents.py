"""
agents.py — LARSA agent signal functions, extracted verbatim.

These are NOT neural networks — they are fixed linear/nonlinear signal
combiners with hand-tuned weights, wrapped in tanh.  The "D3QN/DDQN/TD3QN"
names reflect the design intent (each captures different market dynamics)
rather than actual deep-learning implementations.

All functions are pure (no state, no LEAN dependency).
mu is the gravitational lensing amplification from GravitationalLens.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple
from srfm_core import MarketRegime

# ─── Feature index shorthands ─────────────────────────────────────────────────
# Match features.py F_* constants for readability
_RSI  = 0; _MACD = 1; _MSIG = 2; _MHST = 3; _MOM = 4; _ROC = 5
_ATPCT= 6; _BBW  = 7; _BBP  = 8; _BBD  = 9; _STD = 10
_D12  = 11;_D26  = 12;_D50  = 13;_D200 = 14;_EX  = 15;_EX2 = 16;_ADX = 17
_VOLR = 18;_OBS  = 19;_LR   = 21;_R3   = 22;_R10 = 23


# ─────────────────────────────────────────────────────────────────────────────
# Individual agents
# ─────────────────────────────────────────────────────────────────────────────

def agent_d3qn(f: np.ndarray, mu: float, rc: float) -> Tuple[float, float]:
    """
    D3QN agent — trend + momentum focus.
    Primary inputs: EX (EMA spread), D12, D200, MACD, momentum.
    RSI used as contrarian modifier.
    ATR used as volatility damper.

    Returns (signal, confidence) both in [-1, 1].
    """
    s = f[_EX]*0.25 + f[_D12]*0.15 + f[_D200]*0.20 + f[_MACD]*0.15 + f[_MOM]*0.10
    if f[_RSI] < 0.30:
        s += 0.10
    elif f[_RSI] > 0.70:
        s -= 0.10
    s *= max(0.3, 1.0 - f[_ATPCT] * 2)
    s  = float(np.tanh(s * mu))
    return s, float(np.clip(abs(s) * rc, 0, 1))


def agent_ddqn(f: np.ndarray, mu: float, rc: float) -> Tuple[float, float]:
    """
    DDQN agent — alignment + momentum composite.
    Primary inputs: sign-alignment of MACD/histogram/mom/roc,
    D26, EX, volume, open-price momentum, log returns.
    RSI contrarian.

    Returns (signal, confidence).
    """
    aln = float(sum([np.sign(f[_MACD]), np.sign(f[_MHST]),
                     np.sign(f[_MOM]),  np.sign(f[_ROC])]))
    s = aln*0.12 + f[_D26]*0.12 + f[_EX]*0.10
    s *= 1.0 + float(np.clip(f[_VOLR], -0.5, 0.5))
    # f[20] = 0.0 (reserved), so f[20]*0.10 = 0; kept for formula fidelity
    s += 0.0*0.10 + f[_LR]*0.10 + f[_R3]*0.08 + f[_R10]*0.08
    if f[_RSI] < 0.25:
        s += 0.08
    elif f[_RSI] > 0.75:
        s -= 0.08
    s = float(np.tanh(s * mu))
    return s, float(np.clip(abs(s) * rc, 0, 1))


def agent_td3qn(f: np.ndarray, mu: float, rc: float, ht: float) -> Tuple[float, float]:
    """
    TD3QN agent — mean-reversion + volatility-aware.
    Primary inputs: BBP (contrarian), BBW (regime filter), BBD,
    RSI contrarian, ATR damper, Hawking temperature.
    Std deviation used as confidence suppressor.

    Returns (signal, confidence).
    """
    s = -(f[_BBP] - 0.5) * 0.40
    if f[_BBW] < 0.02:
        s += f[_EX] * 0.20
    elif f[_BBW] > 0.08:
        s -= f[_BBD] * 0.15
    s += (0.5 - f[_RSI]) * 0.25
    if f[_ATPCT] > 0.03:
        s *= 0.60
    s += f[_D200] * 0.08
    if ht > 1.5:
        s -= 0.15
    elif ht < -1.5:
        s += 0.10
    sc = 1.0 - float(np.clip(f[_STD], 0, 0.8))
    s  = float(np.tanh(s * mu))
    return s, float(np.clip(abs(s) * rc * sc, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

REGIME_WEIGHTS = {
    MarketRegime.BULL:           np.array([0.40, 0.35, 0.25]),
    MarketRegime.BEAR:           np.array([0.25, 0.40, 0.35]),
    MarketRegime.SIDEWAYS:       np.array([0.30, 0.30, 0.40]),
    MarketRegime.HIGH_VOLATILITY: np.array([0.25, 0.35, 0.40]),
}


def ensemble(
    f:       np.ndarray,
    mu:      float,
    rc:      float,
    ht:      float,
    beta:    float,
    regime:  MarketRegime,
    geo_slope: float,
    geo_dev:   float,
    rapidity:  float,
) -> Tuple[float, float, Tuple[float, float, float]]:
    """
    Combine D3QN + DDQN + TD3QN with regime-weighted confidence blending.

    From LARSA FutureInstrument.ensemble():
    1. Compute raw signals from each agent.
    2. Apply Lorentz boost / damping based on beta:
       - beta < 1 (TIMELIKE): γ = 1/√(1−β²), boost up to 2×
       - beta >= 1 (SPACELIKE): damp by 1/β
    3. Apply geodesic corrections:
       - D3QN boosted if slope > 0 and dev < 0
       - TD3QN adjusted by geo_dev * -0.3
       - DDQN halved if geo_slope sign disagrees
       - D3QN adjusted by rapidity * 0.10
    4. Weighted combine using regime weights × (confidence + 0.1)

    Returns (action, confidence, (s1,s2,s3)).
    """
    s1, c1 = agent_d3qn(f, mu, rc)
    s2, c2 = agent_ddqn(f, mu, rc)
    s3, c3 = agent_td3qn(f, mu, rc, ht)

    # Lorentz boost / damping
    if beta < 1.0:
        g = min(1.0 / np.sqrt(max(1e-9, 1.0 - beta * beta)), 2.0)
        s1 *= g; s2 *= g; s3 *= g
    else:
        d = 1.0 / (beta + 1e-9)
        s1 *= d; s2 *= d; s3 *= d

    # Geodesic corrections
    if geo_slope > 0 and geo_dev < 0:
        s1 += 0.10
    s3  += geo_dev * -0.3
    if np.sign(geo_slope) != np.sign(s2):
        s2 *= 0.5
    s1 += rapidity * 0.10

    sigs = np.array([s1, s2, s3])
    cons = np.array([c1, c2, c3])

    w  = REGIME_WEIGHTS[regime].copy() * (cons + 0.1)
    w /= w.sum()

    action     = float(np.dot(w, sigs))
    confidence = float(np.dot(w, cons))
    return action, confidence, (float(s1), float(s2), float(s3))


# ─────────────────────────────────────────────────────────────────────────────
# Sizer
# ─────────────────────────────────────────────────────────────────────────────

def size_position(
    f:       np.ndarray,
    action:  float,
    conf:    float,
    rm:      float,        # risk multiplier 0.0 or 0.5 or 1.0
    regime:  MarketRegime,
    mu:      float,
    rc:      float,
    ht:      float,
    tl_window: list,
) -> float:
    """
    Compute position target from LARSA FutureInstrument.size().
    Returns signed position fraction (pre-kill-condition clipping).
    """
    if rm == 0.0:
        return 0.0

    sign = float(np.sign(action)) if action != 0 else 0.0
    mag  = abs(action)

    if regime in (MarketRegime.BULL, MarketRegime.BEAR):
        rw = mag * conf * rm * 1.5
        if tl_window:
            tlf = sum(tl_window) / len(tl_window)
            rw *= max(0.3, min(1.5, tlf / 0.7))
        return sign * rw

    if regime == MarketRegime.SIDEWAYS:
        s3, c3 = agent_td3qn(f, mu, rc, ht)
        return sign * max(abs(s3) * c3 * rm * 1.25, mag * conf * rm * 1.0)

    if regime == MarketRegime.HIGH_VOLATILITY:
        return sign * mag * conf * rm * 1.25

    return 0.0
