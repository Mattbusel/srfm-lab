"""
onchain/metrics/mvrv.py
────────────────────────
MVRV Z-Score — Market Value to Realized Value.

Financial rationale
───────────────────
Realized Value prices each coin at the USD value when it last moved on-chain,
summing to a network-wide "cost basis".  When Market Cap >> Realized Cap,
holders are sitting on large unrealised profits and are more likely to sell —
historically MVRV > 3.7 has coincided with every major cycle top.

MVRV < 1.0 means the average coin is at a loss; holders have capitulated and
selling pressure is exhausted — a classic long-term accumulation zone.

The Z-Score normalises the raw ratio across its full history so we can compare
across cycles of very different absolute price levels.

    Z = (MVRV_ratio - mean(MVRV_ratio)) / std(MVRV_ratio)

Data source priority
────────────────────
1. CoinMetrics Community API  (free, no key, BTC/ETH)
2. Simulated model from price history using a convolution-based realized-value
   approximation (exponential-decay weighted average of historical prices).

Signal thresholds
─────────────────
  MVRV ratio > 7   → overheated   → sell zone  → signal  = -1.0
  MVRV ratio 3–7   → caution      →             → signal  = -0.5
  MVRV ratio 1–3   → fair value   →             → signal   = 0.0
  MVRV ratio < 1   → undervalued  → buy zone   → signal  = +1.0
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

# CoinMetrics community API base URL (no API key required for community tier)
_CM_BASE = "https://community-api.coinmetrics.io/v4"

# Decay half-life for the realized-value simulation model (days).
# Coins that last moved 180 days ago contribute half as much to realized cap
# as coins that moved today.  This is a simplification of the UTXO model.
_REALIZED_VALUE_HALFLIFE_DAYS = 180

# Lookback window (days) for Z-score normalisation.
_ZSCORE_WINDOW = 365 * 4  # ~4 years captures at least one full cycle


@dataclass
class MVRVResult:
    symbol: str
    mvrv_ratio: float
    mvrv_zscore: float
    market_cap_usd: float
    realized_cap_usd: float
    signal: float           # [-1, +1]
    source: str             # "coinmetrics" | "simulated"
    computed_at: str        # ISO 8601


def _fetch_coinmetrics(asset: str, metric: str, days: int = 730) -> Optional[pd.Series]:
    """Pull a single metric time-series from the CoinMetrics community API.

    Returns a pd.Series indexed by date or None on any network/parse error.
    """
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = pd.Timestamp.now() - pd.Timedelta(days=days)
    start_str = start.strftime("%Y-%m-%d")
    url = (
        f"{_CM_BASE}/timeseries/asset-metrics"
        f"?assets={asset}&metrics={metric}"
        f"&start_time={start_str}&end_time={end}&page_size=10000"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")[metric].astype(float)
        return df
    except Exception as exc:
        logger.warning("CoinMetrics fetch failed for %s/%s: %s", asset, metric, exc)
        return None


def _simulate_realized_cap(price_series: pd.Series) -> pd.Series:
    """Approximate realized cap from price history using exponential decay.

    Each day's price is weighted by an exponentially decaying "age" factor.
    This simulates the on-chain UTXO age distribution: recently moved coins
    dominate realized cap while old dormant coins contribute their historical
    cost basis.

    The result is a rough but directionally correct realized cap proxy.
    """
    n = len(price_series)
    prices = price_series.values.astype(float)

    # Exponential decay weights — older prices decay toward historical level
    alpha = np.log(2) / _REALIZED_VALUE_HALFLIFE_DAYS

    realized = np.zeros(n)
    for i in range(n):
        weights = np.exp(-alpha * np.arange(i + 1)[::-1])
        weights /= weights.sum()
        realized[i] = np.dot(prices[: i + 1], weights)

    return pd.Series(realized, index=price_series.index, name="realized_price")


def compute_mvrv(
    price_series: pd.Series,
    circulating_supply: float = 19_700_000.0,  # approximate BTC supply
    symbol: str = "BTC-USD",
) -> MVRVResult:
    """Compute MVRV Z-Score for the given price series.

    Parameters
    ----------
    price_series:
        Daily close prices indexed by DatetimeIndex.
    circulating_supply:
        Approximate circulating supply (coins) used to convert price → cap.
        For BTC use ~19.7M; for ETH use ~120M.
    symbol:
        Human-readable label for the result.

    Returns
    -------
    MVRVResult with ratio, Z-score, and a directional signal in [-1, +1].
    """
    if price_series.empty or len(price_series) < 30:
        raise ValueError("price_series must contain at least 30 data points")

    price_series = price_series.dropna().sort_index()

    # --- Attempt live CoinMetrics data first ---
    asset = "btc" if "BTC" in symbol.upper() else "eth"
    cm_mvrv = _fetch_coinmetrics(asset, "CapMVRVCur", days=_ZSCORE_WINDOW + 30)
    source = "simulated"

    if cm_mvrv is not None and len(cm_mvrv) >= 30:
        mvrv_series = cm_mvrv.dropna()
        source = "coinmetrics"
        logger.info("MVRV: using CoinMetrics data (%d rows)", len(mvrv_series))
    else:
        logger.info("MVRV: falling back to simulated realized-cap model")
        realized_price = _simulate_realized_cap(price_series)
        mvrv_series = (price_series / realized_price).dropna()
        mvrv_series.name = "mvrv_ratio"

    # Z-score normalisation over rolling window
    window = min(_ZSCORE_WINDOW, len(mvrv_series))
    rolling_mean = mvrv_series.rolling(window, min_periods=60).mean()
    rolling_std = mvrv_series.rolling(window, min_periods=60).std()
    zscore_series = (mvrv_series - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore_series = zscore_series.dropna()

    current_ratio = float(mvrv_series.iloc[-1])
    current_zscore = float(zscore_series.iloc[-1]) if not zscore_series.empty else 0.0

    # Market cap and realized cap (approximations for display)
    current_price = float(price_series.iloc[-1])
    market_cap = current_price * circulating_supply
    realized_cap = (current_price / current_ratio) * circulating_supply if current_ratio != 0 else market_cap

    # Signal mapping — uses raw ratio for interpretability
    signal = _ratio_to_signal(current_ratio)

    return MVRVResult(
        symbol=symbol,
        mvrv_ratio=round(current_ratio, 4),
        mvrv_zscore=round(current_zscore, 4),
        market_cap_usd=round(market_cap, 0),
        realized_cap_usd=round(realized_cap, 0),
        signal=signal,
        source=source,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _ratio_to_signal(ratio: float) -> float:
    """Map MVRV ratio to a continuous [-1, +1] signal.

    Breakpoints are calibrated to historical BTC cycle extremes:
      < 0.5  → extreme capitulation  → +1.0
      0.5–1  → undervalued           → +0.7
      1–2    → fair value lower band → +0.2
      2–3    → fair value upper band → -0.1
      3–5    → caution zone          → -0.5
      5–7    → overheated            → -0.8
      > 7    → extreme greed         → -1.0
    """
    if ratio < 0.5:
        return 1.0
    if ratio < 1.0:
        return 0.7
    if ratio < 2.0:
        return 0.2
    if ratio < 3.0:
        return -0.1
    if ratio < 5.0:
        return -0.5
    if ratio < 7.0:
        return -0.8
    return -1.0


def mvrv_summary(result: MVRVResult) -> str:
    """Return a human-readable one-line summary suitable for hypothesis descriptions."""
    direction = "BULLISH" if result.signal > 0 else "BEARISH" if result.signal < 0 else "NEUTRAL"
    return (
        f"MVRV Z-Score={result.mvrv_zscore:.2f} (ratio={result.mvrv_ratio:.2f}) "
        f"— {direction} signal={result.signal:+.1f} [{result.source}]"
    )
