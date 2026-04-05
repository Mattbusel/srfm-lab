"""
macro-factor/factors/rates.py
──────────────────────────────
Interest Rate Factors — Yield Curve Analysis.

Financial rationale
───────────────────
Interest rates are among the most powerful macro forces on all risk assets.

  RISING rates → discount rate increases → future cash flows worth less →
    growth/risk assets devalued → crypto headwind (especially in rate-hike cycles)

  FALLING rates → liquidity increasing → risk-on → crypto tailwind

Yield Curve Slope (2y10y spread):
  Normal (positive slope): long rates > short rates → growth expectations →
    risk-on environment.
  Inverted (negative slope): short rates > long rates → recession fear →
    risk-off → crypto historically dumps or stagnates.
  The 2s10s inversion has preceded every US recession since the 1980s.

We proxy rates using Treasury ETFs:
  TLT = iShares 20+ Year Treasury Bond ETF (long-duration rates proxy)
  SHY = iShares 1-3 Year Treasury Bond ETF (short-duration rates proxy)

Yield curve slope proxy: TLT / SHY ratio.
  Rising ratio → long rates falling faster than short → curve steepening → risk-on
  Falling ratio → curve flattening/inverting → risk-off

Additional signals:
  TLT momentum (20d): falling TLT = rising long rates = headwind
  TLT vs 200d MA: structural rate trend
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_TLT_TICKER      = "TLT"
_SHY_TICKER      = "SHY"
_MOMENTUM_WINDOW = 20
_MA_LONG_WINDOW  = 200
_LOOKBACK_DAYS   = 300


@dataclass
class RatesResult:
    tlt_price: float
    shy_price: float
    yield_curve_ratio: float        # TLT / SHY  (higher = steeper = risk-on)
    yield_curve_slope_pct: float    # 20d % change in TLT/SHY ratio
    curve_regime: str               # "STEEP" | "FLAT" | "INVERTED"
    tlt_momentum_20d: float         # TLT 20d return
    tlt_ma200_distance: float       # (TLT - MA200) / MA200
    signal: float                   # [-1, +1]: +1 = rates bullish for crypto
    computed_at: str


def _fetch_etf(ticker: str, days: int) -> pd.Series:
    """Download ETF close prices, with simulated fallback."""
    try:
        hist = yf.Ticker(ticker).history(period=f"{days}d", interval="1d", auto_adjust=True)
        if not hist.empty:
            logger.info("Rates: fetched %s (%d bars)", ticker, len(hist))
            return hist["Close"].dropna()
    except Exception as exc:
        logger.warning("Rates: yfinance failed for %s: %s", ticker, exc)

    # Fallback simulation
    rng = np.random.default_rng(seed=hash(ticker) % 10000)
    n   = days
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
    base  = 95.0 if ticker == "TLT" else 82.0
    prices = [base]
    for _ in range(n - 1):
        prices.append(max(40.0, prices[-1] * (1 + rng.normal(0, 0.006))))
    return pd.Series(prices, index=dates, name=ticker)


def compute_rates(lookback_days: int = _LOOKBACK_DAYS) -> RatesResult:
    """Compute yield curve and rate-momentum signals.

    Returns
    -------
    RatesResult with yield curve regime, TLT signals, and composite signal.
    """
    tlt = _fetch_etf(_TLT_TICKER, lookback_days)
    shy = _fetch_etf(_SHY_TICKER, lookback_days)

    # Align series on common dates
    common = tlt.index.intersection(shy.index)
    if len(common) < 60:
        raise ValueError("Insufficient data for rates analysis")
    tlt = tlt.loc[common]
    shy = shy.loc[common]

    current_tlt = float(tlt.iloc[-1])
    current_shy = float(shy.iloc[-1])

    # Yield curve ratio (TLT / SHY)
    curve = tlt / shy
    current_curve = float(curve.iloc[-1])

    # 20d slope change
    if len(curve) > _MOMENTUM_WINDOW:
        past_curve = float(curve.iloc[-(_MOMENTUM_WINDOW + 1)])
        curve_slope_pct = (current_curve - past_curve) / past_curve if past_curve != 0 else 0.0
    else:
        curve_slope_pct = 0.0

    # Classify curve regime based on 52-week percentile
    year_window = min(252, len(curve))
    curve_pct   = float((curve.iloc[-year_window:] < current_curve).mean())
    if curve_pct > 0.60:
        curve_regime = "STEEP"
    elif curve_pct > 0.30:
        curve_regime = "FLAT"
    else:
        curve_regime = "INVERTED"

    # TLT momentum
    if len(tlt) > _MOMENTUM_WINDOW:
        tlt_mom = (current_tlt - float(tlt.iloc[-(_MOMENTUM_WINDOW + 1)])) / float(tlt.iloc[-(_MOMENTUM_WINDOW + 1)])
    else:
        tlt_mom = 0.0

    # TLT vs 200d MA
    ma200 = float(tlt.rolling(_MA_LONG_WINDOW, min_periods=100).mean().iloc[-1])
    tlt_ma_dist = (current_tlt - ma200) / ma200 if ma200 != 0 else 0.0

    # Composite signal
    # Steepening curve / rising TLT = rates falling = risk-on = bullish for crypto
    curve_signal  = np.clip(curve_slope_pct / 0.02, -1.0, 1.0)   # 2% curve steepening = ±1
    tlt_mom_sig   = np.clip(tlt_mom / 0.05, -1.0, 1.0)             # 5% TLT move = ±1
    ma_signal     = np.clip(tlt_ma_dist / 0.08, -1.0, 1.0)         # 8% above MA = ±1

    # Regime adjustment: inverted curve = structural headwind
    regime_adj = {"STEEP": 0.15, "FLAT": 0.0, "INVERTED": -0.25}[curve_regime]

    signal = float(np.clip(
        0.4 * curve_signal + 0.4 * tlt_mom_sig + 0.2 * ma_signal + regime_adj,
        -1.0, 1.0,
    ))

    return RatesResult(
        tlt_price=round(current_tlt, 2),
        shy_price=round(current_shy, 2),
        yield_curve_ratio=round(current_curve, 4),
        yield_curve_slope_pct=round(curve_slope_pct, 4),
        curve_regime=curve_regime,
        tlt_momentum_20d=round(tlt_mom, 4),
        tlt_ma200_distance=round(tlt_ma_dist, 4),
        signal=round(signal, 4),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def rates_summary(result: RatesResult) -> str:
    direction = "RISK-ON" if result.signal > 0 else "RISK-OFF"
    return (
        f"Rates curve={result.curve_regime} TLT={result.tlt_price:.1f} "
        f"mom20d={result.tlt_momentum_20d:+.2%} MA200={result.tlt_ma200_distance:+.2%} "
        f"— {direction} signal={result.signal:+.2f}"
    )
