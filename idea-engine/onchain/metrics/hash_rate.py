"""
onchain/metrics/hash_rate.py
─────────────────────────────
Bitcoin Hash Rate and Difficulty Analysis.

Financial rationale
───────────────────
Hash rate measures the total computational power securing the Bitcoin network.
It is a proxy for miner sentiment and network health:

  RISING hash rate → miners are expanding operations → confident in future
    price → bullish for network security and miner selling pressure
  FALLING hash rate → miner capitulation → less profitable to mine →
    some miners shut down → short-term selling pressure from miner HODL sales

The HASH RATE RIBBON is the most powerful signal in this module:
  Ribbon = 30d SMA / 60d SMA of hash rate
  When the ribbon compresses (crosses below 1) → miner capitulation
  When the ribbon re-expands (crosses above 1 after compression) →
    historically one of the most reliable BTC accumulation signals
  This is analogous to a "death cross / golden cross" for hash rate.

Difficulty adjustment (every ~2016 blocks, ~2 weeks):
  Difficulty increase = competitive environment, miner confidence.
  Difficulty decrease = stress event, some miners offline (capitulation).

Data source priority
────────────────────
1. CoinMetrics Community API (HashRate metric)
2. Simulated: hash rate as a lagged, smoothed function of price with
   mean-reverting noise (miners respond to price with ~30-90d lag).
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

_CM_BASE            = "https://community-api.coinmetrics.io/v4"
_RIBBON_FAST        = 30   # days for fast SMA of hash rate ribbon
_RIBBON_SLOW        = 60   # days for slow SMA
_DIFFICULTY_PERIOD  = 14   # approximate days between difficulty adjustments


@dataclass
class HashRateResult:
    symbol: str
    hash_rate_current: float     # EH/s
    hash_rate_sma30: float       # 30d SMA
    hash_rate_sma60: float       # 60d SMA
    ribbon_ratio: float          # sma30 / sma60  (>1 = expanding, <1 = compressed)
    ribbon_crossover: str        # "golden" | "death" | "none"
    difficulty_change_14d: float # approx 14d difficulty change (%)
    is_capitulation: bool        # ribbon compressed for 5+ consecutive days
    signal: float                # [-1, +1]
    source: str
    computed_at: str


def _fetch_hash_rate(days: int = 400) -> Optional[pd.Series]:
    """Fetch BTC hash rate from CoinMetrics (HashRate metric, EH/s)."""
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        f"{_CM_BASE}/timeseries/asset-metrics"
        f"?assets=btc&metrics=HashRate"
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
        return df.set_index("time")["HashRate"].astype(float)
    except Exception as exc:
        logger.warning("CoinMetrics HashRate fetch failed: %s", exc)
        return None


def _simulate_hash_rate(price_series: pd.Series) -> pd.Series:
    """Simulate hash rate from price history.

    Model:
      - Miners respond to price with a 60-day lag (capex investment cycle).
      - Base hash rate grows at ~50% annualised (secular trend of ASICs).
      - Hash rate mean-reverts toward a lagged price-driven target.
      - AR(1) noise adds realistic day-to-day variation.

    Units: output is in arbitrary "hash rate units" proportional to EH/s.
    The absolute scale doesn't matter for signal generation.
    """
    rng = np.random.default_rng(seed=21)
    prices = price_series.values.astype(float)
    n = len(prices)

    # Secular growth trend
    growth_daily = (1.50 ** (1 / 365)) - 1
    trend = np.array([1.0 * (1 + growth_daily) ** i for i in range(n)])

    # Lagged price signal (60d lag, normalised)
    price_lag60 = pd.Series(prices).shift(60).fillna(method="bfill").values
    price_norm = price_lag60 / price_lag60.mean()

    # Target hash rate = trend * price_driven_factor
    target = trend * (price_norm ** 0.4)  # sub-linear response

    # AR(1) around target
    hash_sim = np.zeros(n)
    hash_sim[0] = target[0]
    phi, sigma = 0.92, 0.03

    for i in range(1, n):
        mean_reversion = target[i] - hash_sim[i - 1]
        noise = rng.normal(0, sigma * hash_sim[i - 1])
        hash_sim[i] = max(0.01, hash_sim[i - 1] + 0.08 * mean_reversion + noise)

    # Scale to realistic EH/s range (current BTC ~600 EH/s as of 2024)
    scale = 600.0 / hash_sim[-1]
    hash_sim *= scale

    return pd.Series(hash_sim, index=price_series.index, name="hash_rate_ehs")


def compute_hash_rate(
    price_series: pd.Series,
    symbol: str = "BTC-USD",
) -> HashRateResult:
    """Compute hash rate ribbon and difficulty signals.

    Parameters
    ----------
    price_series:
        Daily BTC close prices, DatetimeIndex, at least 120 bars.
    symbol:
        Must be BTC-USD (hash rate is Bitcoin-specific).
    """
    if "ETH" in symbol.upper():
        raise ValueError("Hash rate analysis is Bitcoin-specific (BTC-USD only)")
    if price_series.empty or len(price_series) < _RIBBON_SLOW + 10:
        raise ValueError(f"Need at least {_RIBBON_SLOW + 10} price bars")

    price_series = price_series.dropna().sort_index()

    hr_series = _fetch_hash_rate(days=400)
    source = "simulated"
    if hr_series is not None and len(hr_series) >= _RIBBON_SLOW + 10:
        source = "coinmetrics"
        logger.info("HashRate: CoinMetrics data (%d rows)", len(hr_series))
    else:
        logger.info("HashRate: using simulated model")
        hr_series = _simulate_hash_rate(price_series)

    hr_series = hr_series.dropna()

    sma_fast = hr_series.rolling(_RIBBON_FAST, min_periods=10).mean()
    sma_slow = hr_series.rolling(_RIBBON_SLOW, min_periods=20).mean()
    ribbon   = (sma_fast / sma_slow.replace(0, np.nan)).dropna()

    current_hr   = float(hr_series.iloc[-1])
    current_fast = float(sma_fast.dropna().iloc[-1])
    current_slow = float(sma_slow.dropna().iloc[-1])
    current_rib  = float(ribbon.iloc[-1]) if not ribbon.empty else 1.0

    # Detect crossover in last 5 days
    crossover = "none"
    if len(ribbon) >= 6:
        prev_rib = float(ribbon.iloc[-6])
        if prev_rib < 1.0 and current_rib >= 1.0:
            crossover = "golden"   # ribbon re-expanding after compression
        elif prev_rib >= 1.0 and current_rib < 1.0:
            crossover = "death"    # ribbon compressing (miner stress)

    # Capitulation: ribbon < 1 for 5+ consecutive days
    recent_ribbon = ribbon.iloc[-10:] if len(ribbon) >= 10 else ribbon
    cap_run = int((recent_ribbon < 1.0).astype(int).rolling(5, min_periods=5).sum().max() or 0)
    is_cap  = cap_run >= 5

    # Difficulty change proxy: 14d % change in hash rate (hash rate drives difficulty)
    diff_change = 0.0
    if len(hr_series) > _DIFFICULTY_PERIOD:
        past = float(hr_series.iloc[-(_DIFFICULTY_PERIOD + 1)])
        if past > 0:
            diff_change = (current_hr - past) / past

    signal = _hash_rate_to_signal(current_rib, crossover, is_cap)

    return HashRateResult(
        symbol=symbol,
        hash_rate_current=round(current_hr, 2),
        hash_rate_sma30=round(current_fast, 2),
        hash_rate_sma60=round(current_slow, 2),
        ribbon_ratio=round(current_rib, 4),
        ribbon_crossover=crossover,
        difficulty_change_14d=round(diff_change, 4),
        is_capitulation=is_cap,
        signal=signal,
        source=source,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _hash_rate_to_signal(ribbon_ratio: float, crossover: str, is_cap: bool) -> float:
    """Map hash rate ribbon to [-1, +1] signal.

    The golden crossover after miner capitulation is historically a very
    strong accumulation signal — we assign it maximum bullish weight.
    """
    # Base signal from ribbon level
    if ribbon_ratio > 1.05:
        base = 0.3    # expanding — mild bullish (miner confidence)
    elif ribbon_ratio > 1.00:
        base = 0.1
    elif ribbon_ratio > 0.95:
        base = -0.1   # slight compression
    elif ribbon_ratio > 0.90:
        base = -0.3
    else:
        base = -0.5   # severe compression — capitulation in progress

    # Crossover overrides
    if crossover == "golden":
        base = 0.9    # historically very strong buy signal
    elif crossover == "death":
        base = -0.7   # miner capitulation beginning

    # Capitulation zone: if we're in it, the market is near a bottom
    if is_cap and crossover != "golden":
        base = max(base, -0.4)   # don't go more negative — near inflection

    return round(float(np.clip(base, -1.0, 1.0)), 2)


def hash_rate_summary(result: HashRateResult) -> str:
    cap_str  = " [CAPITULATION]"  if result.is_capitulation else ""
    cross_str = f" [{result.ribbon_crossover.upper()} CROSS]" if result.ribbon_crossover != "none" else ""
    direction = "BULLISH" if result.signal > 0 else "BEARISH" if result.signal < 0 else "NEUTRAL"
    return (
        f"HashRate={result.hash_rate_current:.1f} EH/s ribbon={result.ribbon_ratio:.3f} "
        f"diff14d={result.difficulty_change_14d:+.2%} — {direction} signal={result.signal:+.2f}"
        f"{cap_str}{cross_str} [{result.source}]"
    )
