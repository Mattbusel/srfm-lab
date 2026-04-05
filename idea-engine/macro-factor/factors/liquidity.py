"""
macro-factor/factors/liquidity.py
───────────────────────────────────
Global Liquidity Factor — M2 Money Supply and Fed Balance Sheet.

Financial rationale
───────────────────
Crypto is fundamentally a liquidity-driven asset class.  When central banks
expand the money supply and their balance sheets:
  1. Nominal prices of all assets rise (inflation of asset values).
  2. Investors search for yield in riskier assets → crypto benefits.
  3. Dollar weakens → BTC (priced in USD) rises mechanically.

The key metric is Global M2 Money Supply growth rate.  Research (Raoul Pal,
CrossBorderCapital) shows crypto prices have ~90% correlation with global M2
with an approximate 3-month lag:
  Rising M2 now → expect crypto rally in ~3 months.
  Contracting M2 → expect crypto correction in ~3 months.

Data sources:
  1. FRED API (free, no key required for public endpoints):
     M2 (M2NS): US M2 money supply, NSA, monthly
     WALCL: Fed balance sheet total assets (weekly)
  2. Fallback: simulated time series with realistic growth dynamics.

The 3-month lag is implemented by using the M2 growth rate from 90 days ago
as the primary signal for current crypto positioning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_FRED_BASE           = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_M2_SERIES           = "M2NS"         # US M2 Money Supply (NSA, monthly, billions)
_FED_BS_SERIES       = "WALCL"        # Fed balance sheet total assets (weekly, millions)
_LAG_DAYS            = 90             # 3-month lag for M2 → crypto effect
_MOMENTUM_WINDOW     = 90             # 3-month growth rate
_LOOKBACK_MONTHS     = 36


@dataclass
class LiquidityResult:
    m2_current_bn: float             # current M2 in billions USD
    m2_growth_3m: float              # 3-month growth rate
    m2_growth_lagged: float          # 3-month growth rate from 90 days ago (primary signal)
    m2_yoy_growth: float             # year-over-year growth rate
    fed_bs_current_bn: float         # Fed balance sheet in billions USD
    fed_bs_growth_3m: float          # 3-month Fed BS growth rate
    liquidity_score: float           # composite liquidity metric [-1, +1]
    signal: float                    # [-1, +1] with 3-month lag applied
    computed_at: str


def _fetch_fred(series_id: str) -> pd.Series:
    """Fetch a FRED data series as a pd.Series indexed by date."""
    url = f"{_FRED_BASE}?id={series_id}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
        df = df.set_index("DATE")[series_id].replace(".", np.nan).dropna().astype(float)
        logger.info("FRED: fetched %s (%d rows)", series_id, len(df))
        return df
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return pd.Series(dtype=float)


def _simulate_m2(months: int = 48) -> pd.Series:
    """Simulate US M2 money supply growth from known trend.

    Calibrated to approximate US M2 dynamics:
      - Long-run mean growth ~7% annualised (pre-COVID)
      - COVID era: ~25% annualised
      - 2022 QT: slight contraction
      - 2023-2024: re-acceleration ~5-8% annualised
    """
    rng   = np.random.default_rng(seed=33)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=months, freq="MS")
    m2    = [21_000.0]  # approximate 2022 US M2 in billions
    monthly_drift = 0.005  # ~6% annualised growth
    sigma = 0.003

    for i in range(months - 1):
        growth = monthly_drift + rng.normal(0, sigma)
        # Add a structural deceleration in 2022 (months 12-20 ago)
        age = months - 1 - i
        if 12 <= age <= 20:
            growth -= 0.008  # QT contraction
        m2.append(m2[-1] * (1 + growth))
    return pd.Series(m2, index=dates, name="M2NS")


def _simulate_fed_bs(months: int = 48) -> pd.Series:
    """Simulate Fed balance sheet assets (in billions)."""
    rng   = np.random.default_rng(seed=44)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=months * 4, freq="W")
    bs    = [8_500.0]  # approximate Fed BS in billions (post-COVID peak)
    for i in range(len(dates) - 1):
        # Mean revert toward 7000 (QT period trend)
        target = 7_500
        bs.append(max(4000, bs[-1] + 0.002 * (target - bs[-1]) + rng.normal(0, 30)))
    return pd.Series(bs, index=dates, name="WALCL_bn")


def compute_liquidity() -> LiquidityResult:
    """Compute global liquidity signals from M2 and Fed balance sheet data.

    Returns
    -------
    LiquidityResult with current levels, growth rates, and a lagged signal.
    """
    # --- Fetch M2 ---
    m2_raw = _fetch_fred(_M2_SERIES)
    if len(m2_raw) < 24:
        logger.info("Liquidity: using simulated M2")
        m2_raw = _simulate_m2(48)

    # --- Fetch Fed Balance Sheet ---
    fed_raw = _fetch_fred(_FED_BS_SERIES)
    if len(fed_raw) < 24:
        logger.info("Liquidity: using simulated Fed BS")
        fed_raw = _simulate_fed_bs(48)
        # Convert weekly millions to billions
        fed_raw = fed_raw / 1000.0

    # --- M2 growth rates ---
    m2 = m2_raw.sort_index()
    current_m2 = float(m2.iloc[-1])

    def growth_rate(series: pd.Series, periods: int) -> float:
        if len(series) <= periods:
            return 0.0
        past = float(series.iloc[-(periods + 1)])
        return (float(series.iloc[-1]) - past) / past if past != 0 else 0.0

    # M2 is monthly; 3 periods = 3 months
    m2_3m    = growth_rate(m2, 3)
    m2_yoy   = growth_rate(m2, 12)

    # Lagged signal: use M2 growth rate from 3 months ago
    if len(m2) >= 6:
        m2_3m_ago = float(m2.iloc[-4])
        m2_6m_ago = float(m2.iloc[-7]) if len(m2) >= 7 else m2_3m_ago
        m2_lagged = (m2_3m_ago - m2_6m_ago) / m2_6m_ago if m2_6m_ago != 0 else 0.0
    else:
        m2_lagged = m2_3m

    # --- Fed BS growth ---
    fed_bs = fed_raw.sort_index()
    # Resample to monthly for consistency
    fed_monthly = fed_bs.resample("MS").last().dropna()
    current_fed = float(fed_monthly.iloc[-1]) if not fed_monthly.empty else 7_500.0
    fed_3m = growth_rate(fed_monthly, 3)

    # --- Composite liquidity score ---
    # M2 lagged (primary): normalise — 10% annualised = strong bull = +1
    m2_signal_lagged  = float(np.clip(m2_lagged / 0.08, -1.0, 1.0))  # monthly: 8% lag
    m2_signal_current = float(np.clip(m2_3m     / 0.05, -1.0, 1.0))  # 5% 3m growth = +1
    fed_signal        = float(np.clip(fed_3m    / 0.05, -1.0, 1.0))

    liquidity_score = float(np.clip(
        0.50 * m2_signal_lagged + 0.30 * m2_signal_current + 0.20 * fed_signal,
        -1.0, 1.0,
    ))

    return LiquidityResult(
        m2_current_bn=round(current_m2, 0),
        m2_growth_3m=round(m2_3m, 4),
        m2_growth_lagged=round(m2_lagged, 4),
        m2_yoy_growth=round(m2_yoy, 4),
        fed_bs_current_bn=round(current_fed, 0),
        fed_bs_growth_3m=round(fed_3m, 4),
        liquidity_score=round(liquidity_score, 4),
        signal=round(liquidity_score, 4),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def liquidity_summary(result: LiquidityResult) -> str:
    direction = "EXPANDING (crypto tailwind)" if result.signal > 0 else "CONTRACTING (crypto headwind)"
    return (
        f"M2=${result.m2_current_bn:,.0f}B 3m={result.m2_growth_3m:+.2%} "
        f"lagged={result.m2_growth_lagged:+.2%} YoY={result.m2_yoy_growth:+.2%} "
        f"FedBS=${result.fed_bs_current_bn:,.0f}B — {direction} signal={result.signal:+.2f}"
    )
