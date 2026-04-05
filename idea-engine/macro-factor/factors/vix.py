"""
macro-factor/factors/vix.py
────────────────────────────
VIX Fear Index Factor.

Financial rationale
───────────────────
VIX (CBOE Volatility Index) measures the 30-day implied volatility of the S&P500
options market.  It is widely known as the "fear gauge":

  VIX > 30 → elevated fear → flight to safety → crypto historically sells off
    (crypto has been increasingly correlated with equities since 2020)
  VIX > 40 → crisis level → maximum risk-off → crypto dumps hard
  VIX < 15 → complacency → risk-on environment → crypto typically rallies
  VIX < 12 → extreme complacency → often contrarian warning (too calm)

VIX Spike Rate of Change:
  A sudden VIX spike (>50% in 5 days) is a severe risk-off signal even if the
  absolute level seems manageable — it signals a shift in market regime.

VIX Percentile (rolling 252-day):
  Contextualises the absolute VIX level within the recent regime.
  High percentile = currently fearful vs recent history = risk-off.

Ticker: VIXY (ProShares VIX Short-Term Futures ETF) or ^VIX (spot VIX).
We prefer ^VIX (spot) since VIXY has roll decay that distorts levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_VIX_TICKER          = "^VIX"
_VIX_SPIKE_WINDOW    = 5    # days for spike detection
_VIX_PERCENTILE_WIN  = 252  # rolling window for percentile
_LOOKBACK_DAYS       = 400

# VIX regime thresholds
_VIX_CALM       = 15.0
_VIX_ELEVATED   = 25.0
_VIX_FEAR       = 30.0
_VIX_CRISIS     = 40.0


@dataclass
class VIXResult:
    vix_current: float
    vix_5d_change_pct: float    # % change over 5 days
    vix_percentile_252d: float  # percentile within trailing year [0, 1]
    vix_regime: str             # "CALM" | "ELEVATED" | "FEAR" | "CRISIS"
    is_spike: bool              # VIX jumped >50% in 5 days
    signal: float               # [-1, +1]: +1 = calm (bullish for crypto), -1 = fear
    computed_at: str


def fetch_vix(days: int = _LOOKBACK_DAYS) -> pd.Series:
    """Download VIX from yfinance (^VIX), with simulated fallback."""
    try:
        hist = yf.Ticker(_VIX_TICKER).history(period=f"{days}d", interval="1d", auto_adjust=False)
        if not hist.empty:
            logger.info("VIX: fetched %d bars from yfinance", len(hist))
            return hist["Close"].dropna()
    except Exception as exc:
        logger.warning("VIX: yfinance failed: %s", exc)

    # Fallback: simulate VIX as a mean-reverting jump-diffusion process
    logger.info("VIX: using simulated fallback")
    rng    = np.random.default_rng(seed=55)
    n      = days
    dates  = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
    vix    = [20.0]
    mean   = 20.0
    kappa  = 0.15   # mean reversion speed
    sigma  = 2.5    # volatility of volatility
    for _ in range(n - 1):
        v = vix[-1]
        dv = kappa * (mean - v) + sigma * rng.normal()
        # Occasional jumps (simulate fear events)
        if rng.random() < 0.02:
            dv += rng.uniform(5, 15)
        vix.append(max(9.0, v + dv))
    return pd.Series(vix, index=dates, name="VIX")


def compute_vix(lookback_days: int = _LOOKBACK_DAYS) -> VIXResult:
    """Compute VIX-based macro signals.

    Returns
    -------
    VIXResult with current VIX, spike detection, percentile, and signal.
    """
    vix = fetch_vix(lookback_days)
    if len(vix) < 20:
        raise ValueError("Insufficient VIX data")

    current = float(vix.iloc[-1])

    # 5-day change
    if len(vix) > _VIX_SPIKE_WINDOW:
        past = float(vix.iloc[-(_VIX_SPIKE_WINDOW + 1)])
        change_5d = (current - past) / past if past != 0 else 0.0
    else:
        change_5d = 0.0

    # Spike detection: >50% jump in 5 days
    is_spike = change_5d > 0.50

    # Rolling percentile (252d)
    window = min(_VIX_PERCENTILE_WIN, len(vix))
    recent = vix.iloc[-window:]
    percentile = float((recent < current).mean())

    # Regime classification
    if current >= _VIX_CRISIS:
        regime = "CRISIS"
    elif current >= _VIX_FEAR:
        regime = "FEAR"
    elif current >= _VIX_ELEVATED:
        regime = "ELEVATED"
    else:
        regime = "CALM"

    # Signal mapping: high VIX = risk-off = negative for crypto
    level_signal = _vix_level_to_signal(current)
    spike_penalty = -0.30 if is_spike else 0.0
    percentile_signal = -((percentile - 0.5) * 1.2)  # high percentile = -ve signal

    signal = float(np.clip(
        0.5 * level_signal + 0.3 * percentile_signal + 0.2 * (-np.clip(change_5d / 0.3, -1.0, 1.0)) + spike_penalty,
        -1.0, 1.0,
    ))

    return VIXResult(
        vix_current=round(current, 2),
        vix_5d_change_pct=round(change_5d, 4),
        vix_percentile_252d=round(percentile, 4),
        vix_regime=regime,
        is_spike=is_spike,
        signal=round(signal, 4),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _vix_level_to_signal(vix: float) -> float:
    """Map absolute VIX level to [-1, +1].  High VIX → negative signal."""
    if vix >= 50:   return -1.0
    if vix >= 40:   return -0.85
    if vix >= 30:   return -0.60
    if vix >= 25:   return -0.30
    if vix >= 20:   return -0.10
    if vix >= 15:   return  0.20
    if vix >= 12:   return  0.50
    return  0.30   # below 12 = contrarian warning (too calm)


def vix_summary(result: VIXResult) -> str:
    spike_str = " [SPIKE]" if result.is_spike else ""
    direction = "RISK-ON" if result.signal > 0 else "RISK-OFF"
    return (
        f"VIX={result.vix_current:.1f} ({result.vix_regime}) "
        f"5d={result.vix_5d_change_pct:+.1%} pct252={result.vix_percentile_252d:.0%} "
        f"— {direction} signal={result.signal:+.2f}{spike_str}"
    )
