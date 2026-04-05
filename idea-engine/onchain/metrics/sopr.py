"""
onchain/metrics/sopr.py
────────────────────────
SOPR — Spent Output Profit Ratio.

Financial rationale
───────────────────
Each time a Bitcoin UTXO is spent, we can compare the USD value at the time of
spending to the USD value when it was created (i.e., last moved).  SOPR is the
ratio of these two values aggregated across all UTXOs spent in a day:

    SOPR = Sum(USD value at spend time) / Sum(USD value at creation time)

  SOPR > 1.0 → on average, coins moving today were acquired at a lower price
               than today → sellers are in profit → potential sell pressure
  SOPR = 1.0 → break-even for movers — often a support/resistance level
  SOPR < 1.0 → coins are being sold at a loss → capitulation →
               historically a powerful buy signal as weak hands exit

Adjusted SOPR (aSOPR) excludes same-day UTXO re-spends (short-term noise).

Simulation model
────────────────
We cannot directly observe UTXO creation prices without a full node.  Instead
we approximate SOPR from price history:

  1. Compute a rolling "average cost basis" using a decay-weighted average of
     recent prices (representing the coins likely to be spent on a given day).
  2. SOPR ≈ current_price / cost_basis
  3. Add mean-reverting AR(1) noise to simulate microstructure variation.

The simulation captures the key directional behavior: SOPR > 1 in bull markets,
SOPR < 1 during crashes, and SOPR crossing 1 as a regime change signal.
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

_CM_BASE              = "https://community-api.coinmetrics.io/v4"
_SOPR_SMOOTH_WINDOW   = 14     # days for smoothed SOPR (reduce noise)
_COST_BASIS_HALFLIFE  = 90     # days for simulated cost-basis decay


@dataclass
class SOPRResult:
    symbol: str
    sopr_raw: float          # point-in-time SOPR
    sopr_smooth: float       # 14-day SMA of SOPR
    sopr_zscore: float       # Z-score vs 1-year rolling window
    signal: float            # [-1, +1]
    is_capitulation: bool    # SOPR < 1 for 3+ consecutive days
    source: str
    computed_at: str


def _fetch_sopr_coinmetrics(asset: str, days: int = 400) -> Optional[pd.Series]:
    """Attempt to fetch SOPR (SoprAdjUsd) from CoinMetrics community API."""
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        f"{_CM_BASE}/timeseries/asset-metrics"
        f"?assets={asset}&metrics=SoprAdjUsd"
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
        return df.set_index("time")["SoprAdjUsd"].astype(float)
    except Exception as exc:
        logger.warning("CoinMetrics SOPR fetch failed: %s", exc)
        return None


def _simulate_sopr(price_series: pd.Series) -> pd.Series:
    """Simulate SOPR from price history using decay-weighted cost basis.

    Algorithm:
      cost_basis[t] = exponentially weighted average of price[-halflife:]
      sopr_raw[t]   = price[t] / cost_basis[t]  + AR(1) noise

    The halflife of 90 days captures the typical UTXO age distribution
    for coins likely to be spent on any given day.  The noise term (σ≈0.05)
    adds realistic day-to-day variation while preserving the price-driven
    directional signal.
    """
    rng = np.random.default_rng(seed=7)
    prices = price_series.values.astype(float)
    n = len(prices)
    alpha = np.log(2) / _COST_BASIS_HALFLIFE

    cost_basis = np.zeros(n)
    for i in range(n):
        lookback = min(i + 1, _COST_BASIS_HALFLIFE * 3)
        w = np.exp(-alpha * np.arange(lookback)[::-1])
        w /= w.sum()
        cost_basis[i] = np.dot(prices[i - lookback + 1: i + 1], w)

    sopr_base = prices / np.where(cost_basis > 0, cost_basis, np.nan)

    # AR(1) multiplicative noise
    noise = np.zeros(n)
    phi, sigma = 0.7, 0.04
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, sigma)
    sopr_sim = sopr_base * np.exp(noise)

    return pd.Series(sopr_sim, index=price_series.index, name="sopr")


def compute_sopr(
    price_series: pd.Series,
    symbol: str = "BTC-USD",
) -> SOPRResult:
    """Compute SOPR for the given price history.

    Parameters
    ----------
    price_series:
        Daily close prices, DatetimeIndex, at least 120 bars.
    symbol:
        Ticker label (BTC-USD or ETH-USD).

    Returns
    -------
    SOPRResult with raw, smoothed, Z-score, capitulation flag, and signal.
    """
    if price_series.empty or len(price_series) < 120:
        raise ValueError("Need at least 120 price bars for SOPR computation")

    price_series = price_series.dropna().sort_index()
    asset = "btc" if "BTC" in symbol.upper() else "eth"

    sopr_series = _fetch_sopr_coinmetrics(asset, days=400)
    source = "simulated"
    if sopr_series is not None and len(sopr_series) >= 60:
        source = "coinmetrics"
        logger.info("SOPR: CoinMetrics data (%d rows)", len(sopr_series))
    else:
        logger.info("SOPR: falling back to simulated model")
        sopr_series = _simulate_sopr(price_series)

    sopr_series = sopr_series.replace([np.inf, -np.inf], np.nan).dropna()

    # Smoothed SOPR (14d SMA)
    sopr_smooth = sopr_series.rolling(_SOPR_SMOOTH_WINDOW, min_periods=5).mean()

    # Z-score normalisation (1-year rolling)
    window = min(365, len(sopr_series))
    rm = sopr_series.rolling(window, min_periods=30).mean()
    rs = sopr_series.rolling(window, min_periods=30).std()
    zscore = ((sopr_series - rm) / rs.replace(0, np.nan)).dropna()

    current_raw    = float(sopr_series.iloc[-1])
    current_smooth = float(sopr_smooth.dropna().iloc[-1]) if not sopr_smooth.dropna().empty else current_raw
    current_z      = float(zscore.iloc[-1]) if not zscore.empty else 0.0

    # Capitulation: SOPR < 1 for 3+ consecutive days
    recent = sopr_series.iloc[-5:]
    consec_below = int((recent < 1.0).astype(int).rolling(3, min_periods=3).sum().max() or 0)
    is_capitulation = consec_below >= 3

    signal = _sopr_to_signal(current_smooth, is_capitulation)

    return SOPRResult(
        symbol=symbol,
        sopr_raw=round(current_raw, 4),
        sopr_smooth=round(current_smooth, 4),
        sopr_zscore=round(current_z, 4),
        signal=signal,
        is_capitulation=is_capitulation,
        source=source,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _sopr_to_signal(sopr_smooth: float, is_capitulation: bool) -> float:
    """Map smoothed SOPR to [-1, +1].

    Capitulation bonus: when SOPR < 1 for extended period, conviction is higher
    that selling is exhausted and a reversal is imminent.
    """
    if sopr_smooth > 1.3:
        base = -1.0
    elif sopr_smooth > 1.15:
        base = -0.6
    elif sopr_smooth > 1.05:
        base = -0.2
    elif sopr_smooth > 0.97:
        base = 0.1
    elif sopr_smooth > 0.90:
        base = 0.5
    else:
        base = 0.8

    # Capitulation amplifies the buy signal
    if is_capitulation and sopr_smooth < 1.0:
        base = min(1.0, base + 0.2)

    return round(base, 2)


def sopr_summary(result: SOPRResult) -> str:
    cap_flag = " [CAPITULATION]" if result.is_capitulation else ""
    direction = "BULLISH" if result.signal > 0 else "BEARISH" if result.signal < 0 else "NEUTRAL"
    return (
        f"SOPR={result.sopr_smooth:.3f} (raw={result.sopr_raw:.3f}, "
        f"Z={result.sopr_zscore:.2f}) — {direction} signal={result.signal:+.1f}"
        f"{cap_flag} [{result.source}]"
    )
