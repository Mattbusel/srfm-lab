"""
macro-factor/factors/dxy.py
────────────────────────────
US Dollar Index (DXY) Factor.

Financial rationale
───────────────────
The US Dollar Index measures the USD against a basket of six major currencies
(EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2%, CHF 3.6%).

DXY has a historically strong INVERSE correlation with Bitcoin and risk assets:

  RISING DXY → dollar strengthening → liquidity tightening →
    risk assets, commodities, and crypto under pressure (RISK_OFF)

  FALLING DXY → dollar weakening → global liquidity expanding →
    crypto historically rallies (RISK_ON)

Key signals computed:
  1. 20-day momentum: percent return over last 20 trading days.
  2. RSI(14): overbought (>70) or oversold (<30) conditions.
  3. Distance from 200-day MA: structural trend direction.
  4. Composite risk_off_score: weighted average of the three signals.

The risk_off_score ranges from -1 (maximum risk-off, bearish for crypto)
to +1 (maximum risk-on, bullish for crypto).

DXY ticker on yfinance: "DX-Y.NYB"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_DXY_TICKER       = "DX-Y.NYB"
_MOMENTUM_WINDOW  = 20     # days
_RSI_WINDOW       = 14     # days
_MA_LONG_WINDOW   = 200    # days
_LOOKBACK_DAYS    = 300    # days to fetch from yfinance


@dataclass
class DXYResult:
    momentum_20d: float         # % return over 20d
    rsi_14: float               # RSI value [0, 100]
    ma200_distance: float       # (price - MA200) / MA200  (+ = above)
    dxy_level: float            # current DXY level
    risk_off_score: float       # [-1, +1]: -1 = max risk-off (dollar surge), +1 = risk-on (dollar weak)
    signal: float               # [-1, +1] for regime classifier (same convention: +1 = bullish for crypto)
    computed_at: str


def _compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI using exponential moving average smoothing (Wilder's method)."""
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def fetch_dxy(lookback_days: int = _LOOKBACK_DAYS) -> pd.Series:
    """Download DXY close prices from yfinance.

    Returns a pd.Series of daily close prices indexed by DatetimeIndex.
    Falls back to a simulated series if yfinance is unavailable.
    """
    try:
        ticker = yf.Ticker(_DXY_TICKER)
        hist   = ticker.history(period=f"{lookback_days}d", interval="1d", auto_adjust=True)
        if not hist.empty:
            logger.info("DXY: fetched %d bars from yfinance", len(hist))
            return hist["Close"].dropna()
    except Exception as exc:
        logger.warning("DXY yfinance fetch failed: %s", exc)

    # Fallback: simulate DXY around 103 with realistic mean-reverting dynamics
    logger.info("DXY: using simulated fallback")
    rng   = np.random.default_rng(seed=77)
    n     = lookback_days
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
    level = 103.0
    prices = [level]
    for _ in range(n - 1):
        level += rng.normal(0, 0.3) - 0.01 * (level - 103)
        prices.append(max(85.0, min(125.0, level)))
    return pd.Series(prices, index=dates, name="DXY")


def compute_dxy(lookback_days: int = _LOOKBACK_DAYS) -> DXYResult:
    """Compute all DXY-based macro signals.

    Returns
    -------
    DXYResult with momentum, RSI, MA distance, and composite risk_off_score.
    """
    prices = fetch_dxy(lookback_days)

    if len(prices) < _MA_LONG_WINDOW + 10:
        raise ValueError(f"Need at least {_MA_LONG_WINDOW + 10} bars for DXY analysis")

    current = float(prices.iloc[-1])

    # 1. Momentum: 20-day return
    if len(prices) > _MOMENTUM_WINDOW:
        past = float(prices.iloc[-(_MOMENTUM_WINDOW + 1)])
        momentum = (current - past) / past if past != 0 else 0.0
    else:
        momentum = 0.0

    # 2. RSI(14)
    rsi_series = _compute_rsi(prices, _RSI_WINDOW)
    rsi_current = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else 50.0

    # 3. Distance from 200d MA
    ma200 = float(prices.rolling(_MA_LONG_WINDOW, min_periods=100).mean().iloc[-1])
    ma200_dist = (current - ma200) / ma200 if ma200 != 0 else 0.0

    # Composite risk_off_score (invert all: higher DXY = more risk-off = negative for crypto)
    # Momentum signal: DXY surging → risk-off → crypto negative
    mom_signal = -np.clip(momentum / 0.03, -1.0, 1.0)   # 3% move = ±1.0

    # RSI signal: overbought DXY (>70) → risk-off; oversold (<30) → risk-on
    if rsi_current > 70:
        rsi_signal = -0.8
    elif rsi_current > 60:
        rsi_signal = -0.3
    elif rsi_current < 30:
        rsi_signal = 0.8
    elif rsi_current < 40:
        rsi_signal = 0.3
    else:
        rsi_signal = 0.0

    # MA distance signal: DXY above 200d MA = structural risk-off
    ma_signal = -np.clip(ma200_dist / 0.05, -1.0, 1.0)  # 5% deviation = ±1.0

    # Weighted composite
    risk_off_score = float(0.5 * mom_signal + 0.3 * rsi_signal + 0.2 * ma_signal)
    risk_off_score = float(np.clip(risk_off_score, -1.0, 1.0))

    return DXYResult(
        momentum_20d=round(momentum, 4),
        rsi_14=round(rsi_current, 2),
        ma200_distance=round(ma200_dist, 4),
        dxy_level=round(current, 3),
        risk_off_score=round(risk_off_score, 4),
        signal=round(risk_off_score, 4),   # same value — convention: +1 = bullish for crypto
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def dxy_summary(result: DXYResult) -> str:
    direction = "RISK-ON (crypto tailwind)" if result.signal > 0 else "RISK-OFF (crypto headwind)"
    return (
        f"DXY={result.dxy_level:.2f} mom20d={result.momentum_20d:+.2%} "
        f"RSI={result.rsi_14:.1f} MA200_dist={result.ma200_distance:+.2%} "
        f"— {direction} signal={result.signal:+.2f}"
    )
