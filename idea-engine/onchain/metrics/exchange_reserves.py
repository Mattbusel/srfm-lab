"""
onchain/metrics/exchange_reserves.py
──────────────────────────────────────
Exchange Reserve Tracking.

Financial rationale
───────────────────
Exchange reserves measure the total BTC (or ETH) held in known exchange wallets.
This is one of the most reliable on-chain leading indicators:

  DECLINING reserves → coins leaving exchanges → being moved to cold storage
    → HODLers accumulating → BULLISH (supply shock incoming)

  RISING reserves → coins entering exchanges → being prepared for sale
    → BEARISH (sell pressure building)

The 30-day rate of change (RoC) captures the momentum of this flow.
Extreme outflows (RoC < -5%) combined with low absolute reserve levels have
historically preceded significant bull runs (e.g., Q4 2020, Q1 2021).

Data source priority
────────────────────
1. CoinMetrics Community API (SplyExNtv metric — supply held on exchanges)
2. Simulated model: reserves follow a mean-reverting process with
   price-correlated flows (rising price → outflows as HODLers buy; falling
   price → inflows as leveraged longs liquidate and sellers move to exchanges).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CM_BASE = "https://community-api.coinmetrics.io/v4"
_LOOKBACK_DAYS = 365


@dataclass
class ExchangeReserveResult:
    symbol: str
    reserve_coins: float          # current total coins on exchanges
    reserve_usd: float            # current USD value of reserves
    reserve_pct_supply: float     # % of total supply on exchanges
    roc_7d: float                 # 7-day rate of change (fractional)
    roc_30d: float                # 30-day rate of change (fractional)
    roc_90d: float                # 90-day rate of change (fractional)
    signal: float                 # [-1, +1]
    source: str
    computed_at: str


def _fetch_exchange_reserves(asset: str, days: int = 400) -> Optional[pd.Series]:
    """Fetch on-exchange supply from CoinMetrics (SplyExNtv metric)."""
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        f"{_CM_BASE}/timeseries/asset-metrics"
        f"?assets={asset}&metrics=SplyExNtv"
        f"&start_time={start}&end_time={end}&page_size=10000"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time")["SplyExNtv"].astype(float)
    except Exception as exc:
        logger.warning("CoinMetrics SplyExNtv failed: %s", exc)
        return None


def _simulate_reserves(
    price_series: pd.Series,
    circulating_supply: float,
    initial_pct: float = 0.12,  # ~12% of BTC supply on exchanges historically
) -> pd.Series:
    """Simulate exchange reserves from price history.

    Model:
      - Base reserve level mean-reverts around initial_pct * supply.
      - Daily flows proportional to negative of price return:
          price up 1% → outflows ≈ 0.05% of reserve (accumulation)
          price down 1% → inflows ≈ 0.07% of reserve (fear / liquidation)
      - AR(1) mean reversion keeps reserves realistic.
    """
    rng = np.random.default_rng(seed=99)
    prices = price_series.values.astype(float)
    n = len(prices)
    returns = np.diff(np.log(prices + 1e-9), prepend=0.0)

    reserves = np.zeros(n)
    target = initial_pct * circulating_supply
    reserves[0] = target

    phi = 0.98       # slow mean reversion
    noise_sigma = 0.0015 * target

    for i in range(1, n):
        # Price-correlated flow: up day → outflows, down day → inflows
        ret = returns[i]
        flow = -ret * 0.25 * reserves[i - 1]   # sign: negative return → positive inflow

        # Mean reversion
        reversion = phi * (reserves[i - 1] - target)
        noise     = rng.normal(0, noise_sigma)

        reserves[i] = max(
            0.01 * circulating_supply,
            reserves[i - 1] + flow - reversion * 0.005 + noise,
        )

    return pd.Series(reserves, index=price_series.index, name="exchange_reserves")


def compute_exchange_reserves(
    price_series: pd.Series,
    circulating_supply: float = 19_700_000.0,
    symbol: str = "BTC-USD",
) -> ExchangeReserveResult:
    """Compute exchange reserve metrics for the given price history.

    Parameters
    ----------
    price_series:
        Daily close prices, DatetimeIndex.
    circulating_supply:
        Total circulating supply (BTC: ~19.7M).
    symbol:
        Ticker label.
    """
    if price_series.empty or len(price_series) < 60:
        raise ValueError("Need at least 60 price bars")

    price_series = price_series.dropna().sort_index()
    asset = "btc" if "BTC" in symbol.upper() else "eth"

    reserve_series = _fetch_exchange_reserves(asset, days=_LOOKBACK_DAYS + 30)
    source = "simulated"
    if reserve_series is not None and len(reserve_series) >= 30:
        source = "coinmetrics"
        logger.info("ExchangeReserves: CoinMetrics data (%d rows)", len(reserve_series))
    else:
        logger.info("ExchangeReserves: using simulated model")
        reserve_series = _simulate_reserves(price_series, circulating_supply)

    reserve_series = reserve_series.dropna()

    current_reserve = float(reserve_series.iloc[-1])
    current_price   = float(price_series.iloc[-1])
    reserve_usd     = current_reserve * current_price
    reserve_pct     = current_reserve / circulating_supply

    # Rate of change: (current - past) / past
    def roc(series: pd.Series, days: int) -> float:
        if len(series) <= days:
            return 0.0
        past = float(series.iloc[-(days + 1)])
        if past == 0:
            return 0.0
        return (float(series.iloc[-1]) - past) / past

    roc_7d  = roc(reserve_series, 7)
    roc_30d = roc(reserve_series, 30)
    roc_90d = roc(reserve_series, 90)

    signal = _reserves_to_signal(roc_7d, roc_30d, roc_90d, reserve_pct)

    return ExchangeReserveResult(
        symbol=symbol,
        reserve_coins=round(current_reserve, 0),
        reserve_usd=round(reserve_usd, 0),
        reserve_pct_supply=round(reserve_pct, 4),
        roc_7d=round(roc_7d, 4),
        roc_30d=round(roc_30d, 4),
        roc_90d=round(roc_90d, 4),
        signal=signal,
        source=source,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _reserves_to_signal(
    roc_7d: float,
    roc_30d: float,
    roc_90d: float,
    reserve_pct: float,
) -> float:
    """Map reserve rate-of-change to [-1, +1] signal.

    Declining reserves → bullish (supply leaving exchanges).
    Rising reserves → bearish (coins being prepared for sale).
    """
    # Weighted composite of multiple timeframes
    score = -(roc_7d * 0.3 + roc_30d * 0.5 + roc_90d * 0.2)

    # Absolute level modifier: very low reserves → structural supply shock
    if reserve_pct < 0.08:
        score += 0.2
    elif reserve_pct < 0.10:
        score += 0.1
    elif reserve_pct > 0.18:
        score -= 0.15

    # Scale: a 1% 30d decline ≈ 0.5 signal → scale to get ±1 at ±2% RoC
    score = score * 25.0  # 2% decline → score ≈ +0.5
    return float(np.clip(score, -1.0, 1.0))


def reserves_summary(result: ExchangeReserveResult) -> str:
    direction = "BULLISH" if result.signal > 0 else "BEARISH" if result.signal < 0 else "NEUTRAL"
    return (
        f"ExchangeReserves={result.reserve_coins:,.0f} BTC ({result.reserve_pct_supply:.1%} of supply) "
        f"30d RoC={result.roc_30d:+.2%} — {direction} signal={result.signal:+.2f} [{result.source}]"
    )
