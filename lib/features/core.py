"""
features.py — 31-element feature vector extracted from LARSA compute_features().

Index map (matches LARSA exactly):
  [0]  rsi / 100
  [1]  MACD normalized (tanh)
  [2]  MACD signal normalized
  [3]  MACD histogram normalized
  [4]  Momentum normalized
  [5]  ROC normalized
  [6]  ATR% × 10
  [7]  Bollinger Band width
  [8]  Bollinger Band %B (clipped 0-1)
  [9]  BB deviation (tanh)
  [10] Std deviation normalized
  [11] Price - EMA12 (ATR-normalized, tanh)
  [12] Price - EMA26
  [13] Price - EMA50
  [14] Price - EMA200
  [15] EMA12 - EMA26 (spread)
  [16] EMA12 - EMA50
  [17] ADX / 100
  [18] Volume ratio (tanh)
  [19] Open-price momentum (tanh)
  [20] 0.0  (reserved)
  [21] 1-bar log return × 100 (tanh)
  [22] 3-bar log return × 50 (tanh)
  [23] 10-bar log return × 20 (tanh)
  [24] Regime one-hot: BULL
  [25] Regime one-hot: BEAR
  [26] Regime one-hot: SIDEWAYS
  [27] Regime one-hot: HIGH_VOLATILITY
  [28] beta (current bar) / 3.0, clipped 0-1
  [29] TIMELIKE fraction (rolling window)
  [30] Hawking temperature normalized (tanh(ht/4))

TIMELIKE weight cw = 1.0 if TIMELIKE else 0.3.
All features clipped to [-3, 3].
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List
from srfm_core import MarketRegime


def compute_features(
    # Prices & indicators
    close:       float,
    rsi:         float,          # raw 0-100
    macd_val:    float,
    macd_sig:    float,
    macd_hist:   float,
    mom:         float,          # momentum (price diff over N bars)
    roc:         float,          # rate of change %
    atr:         float,
    bb_upper:    float,
    bb_lower:    float,
    bb_middle:   float,
    std:         float,
    ema12:       float,
    ema26:       float,
    ema50:       float,
    ema200:      float,
    adx:         float,
    # Rolling window data (lists, index 0 = newest)
    volume_window:  List[float],  # len >= 1
    open_window:    List[float],  # len >= 5 for obs
    close_window:   List[float],  # len >= 10 for r10
    tl_window:      List[float],  # 1.0 / 0.0 per bar
    beta_window:    List[float],  # beta per bar
    # State
    bit:     str,           # "TIMELIKE" | "SPACELIKE"
    regime:  MarketRegime,
    ht:      float,         # Hawking temperature
) -> Optional[np.ndarray]:
    """
    Returns 31-element float32 feature vector or None if not ready.
    Exact reproduction of LARSA FutureInstrument.compute_features().
    """
    if close <= 0 or atr <= 0:
        return None

    cw = 1.0 if bit == "TIMELIKE" else 0.3

    # [0] RSI
    feat_rsi = rsi / 100.0

    # [1-3] MACD family
    mn  = float(np.tanh(macd_val  / (close * 0.01 + 1e-9)) * cw)
    msn = float(np.tanh(macd_sig  / (close * 0.01 + 1e-9)) * cw)
    mhn = float(np.tanh(macd_hist / (close * 0.01 + 1e-9)) * cw)

    # [4-5] Momentum / ROC
    momnorm = float(np.tanh(mom / (close * 0.05 + 1e-9)) * cw)
    rocnorm = float(np.tanh(roc / 10.0) * cw)

    # [6] ATR%
    atr_pct = atr / close

    # [7-9] Bollinger
    bbw = (bb_upper - bb_lower) / (bb_middle + 1e-9)
    bbp = float(np.clip((close - bb_lower) / (bb_upper - bb_lower + 1e-9), 0, 1))
    bbd = float(np.tanh((close - bb_middle) / (atr + 1e-9)))

    # [10] Std
    stn = float(np.tanh(std / (close * 0.02 + 1e-9)))

    # [11-16] EMA distances & spreads
    d12  = float(np.tanh((close - ema12)  / (atr + 1e-9)))
    d26  = float(np.tanh((close - ema26)  / (atr + 1e-9)))
    d50  = float(np.tanh((close - ema50)  / (atr + 1e-9)))
    d200 = float(np.tanh((close - ema200) / (atr + 1e-9)))
    ex   = float(np.tanh((ema12 - ema26)  / (atr + 1e-9)))
    ex2  = float(np.tanh((ema12 - ema50)  / (atr + 1e-9)))

    # [17] ADX
    adxn = adx / 100.0

    # [18] Volume ratio
    volr = 0.0
    if len(volume_window) >= 1:
        va = np.array(volume_window[:20])
        volr = float(np.tanh((va[0] / (va.mean() + 1e-9)) - 1.0))

    # [19] Open momentum (obs)
    obs = 0.0
    if len(open_window) >= 5:
        ro  = np.array(open_window[:5])
        obs = float(np.tanh((ro[0] - ro[4]) / (abs(ro[4]) + 1e-9)))

    # [20] Reserved
    reserved = 0.0

    # [21-23] Log returns
    lr  = float(np.tanh(np.log(close_window[0]  / (close_window[1]  + 1e-9) + 1e-9) * 100)) if len(close_window) >= 2  else 0.0
    r3  = float(np.tanh(np.log(close_window[0]  / (close_window[2]  + 1e-9) + 1e-9) * 50))  if len(close_window) >= 3  else 0.0
    r10 = float(np.tanh(np.log(close_window[0]  / (close_window[9]  + 1e-9) + 1e-9) * 20))  if len(close_window) >= 10 else 0.0

    # [24-27] Regime one-hot
    roh = np.zeros(4)
    roh[int(regime)] = 1.0

    # [28] Beta (current bar)
    bn = float(np.clip(beta_window[0] / 3.0, 0, 1)) if len(beta_window) > 0 else 0.5

    # [29] TIMELIKE fraction
    tlf = 0.5
    if len(tl_window) > 0:
        tlf = float(sum(tl_window) / len(tl_window))

    # [30] Hawking temp
    hn = float(np.tanh(ht / 4.0))

    f = np.array([
        feat_rsi, mn, msn, mhn, momnorm, rocnorm,
        atr_pct * 10, bbw, bbp, bbd, stn,
        d12, d26, d50, d200, ex, ex2, adxn,
        volr, obs, reserved, lr, r3, r10,
        *roh,
        bn, tlf, hn,
    ], dtype=np.float32)

    return np.clip(f, -3.0, 3.0)


# ─── Feature index constants ─────────────────────────────────────────────────
# Use these to index into the feature vector by name.

F_RSI       = 0
F_MACD      = 1
F_MACD_SIG  = 2
F_MACD_HIST = 3
F_MOM       = 4
F_ROC       = 5
F_ATR_PCT   = 6
F_BBW       = 7
F_BBP       = 8
F_BBD       = 9
F_STD       = 10
F_D12       = 11
F_D26       = 12
F_D50       = 13
F_D200      = 14
F_EX        = 15
F_EX2       = 16
F_ADX       = 17
F_VOLR      = 18
F_OBS       = 19
F_RESERVED  = 20
F_LR        = 21
F_R3        = 22
F_R10       = 23
F_BULL      = 24
F_BEAR      = 25
F_SIDEWAYS  = 26
F_HIGHVOL   = 27
F_BETA      = 28
F_TLF       = 29
F_HT        = 30

FEATURE_NAMES = [
    "rsi", "macd", "macd_sig", "macd_hist", "mom", "roc",
    "atr_pct", "bbw", "bbp", "bbd", "std",
    "d12", "d26", "d50", "d200", "ex", "ex2", "adx",
    "volr", "obs", "reserved", "lr", "r3", "r10",
    "bull", "bear", "sideways", "highvol",
    "beta", "tlf", "ht",
]
assert len(FEATURE_NAMES) == 31
