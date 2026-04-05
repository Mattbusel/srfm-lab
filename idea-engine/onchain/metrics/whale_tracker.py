"""
onchain/metrics/whale_tracker.py
──────────────────────────────────
Whale Wallet Movement Tracker.

Financial rationale
───────────────────
"Whales" are large Bitcoin holders whose movements have outsized impact on
supply/demand dynamics.  A single 10,000 BTC transaction can move markets if
it hits an exchange order book.

Key classifications:
  Exchange inflow  (large tx → exchange address):  DISTRIBUTION → bearish
  Exchange outflow (large tx ← exchange address):  ACCUMULATION → bullish
  OTC / wallet-to-wallet (large tx, no exchange):  NEUTRAL / accumulation

Net Whale Flow (7d, 30d):
  Positive → net accumulation (whales buying / withdrawing from exchanges)
  Negative → net distribution (whales selling / depositing to exchanges)

Simulation model
────────────────
Without a full node we cannot observe individual UTXOs.  We simulate whale
activity as follows:

  1. Whale transaction count is Poisson-distributed, with intensity proportional
     to rolling 7d volatility (whales more active during volatile periods).
  2. Each transaction is classified as accumulation / distribution with
     probability conditioned on recent price trend:
       - Falling price → higher probability of accumulation (whales buy dips)
       - Rising price (late stage) → higher probability of distribution
  3. Transaction size is log-normally distributed around a mean of 500 BTC,
     clipped to [100, 10000] BTC (threshold for "large transaction").
  4. Net flow = sum(accumulation txs) - sum(distribution txs) in BTC/day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_WHALE_TX_BTC = 100.0    # minimum BTC to classify as whale transaction
_LOOKBACK_DAYS    = 365


@dataclass
class WhaleTransaction:
    date: str
    btc_amount: float
    tx_type: str          # "accumulation" | "distribution" | "neutral"
    price_usd: float
    usd_value: float


@dataclass
class WhaleTrackerResult:
    symbol: str
    net_flow_7d: float          # net BTC whale flow (+ = accumulation)
    net_flow_30d: float
    whale_tx_count_7d: int
    whale_tx_count_30d: int
    dominant_regime: str        # "ACCUMULATION" | "DISTRIBUTION" | "NEUTRAL"
    large_tx_alert: bool        # unusually large single tx in last 3 days
    signal: float               # [-1, +1]
    recent_transactions: List[WhaleTransaction] = field(default_factory=list)
    computed_at: str = ""


def _simulate_whale_activity(price_series: pd.Series) -> pd.DataFrame:
    """Generate simulated daily whale transaction activity from price history.

    Returns a DataFrame with columns:
      date, btc_accumulated, btc_distributed, tx_count_accum, tx_count_distrib
    """
    rng = np.random.default_rng(seed=13)
    prices = price_series.values.astype(float)
    n = len(prices)
    log_ret = np.diff(np.log(prices + 1e-9), prepend=0.0)

    # Rolling 7d price momentum (sign tells us bull/bear short-term)
    momentum = pd.Series(log_ret).rolling(7, min_periods=2).sum().fillna(0).values

    # Rolling 7d volatility (drives whale activity intensity)
    vol_7d = pd.Series(log_ret).rolling(7, min_periods=2).std().fillna(0.02).values

    rows = []
    for i in range(n):
        # Poisson intensity: more transactions during high vol
        intensity = max(0.5, vol_7d[i] * 50)  # ~1-5 whale txs/day on high vol
        n_txs = rng.poisson(intensity)

        # Probability of accumulation vs distribution conditioned on momentum
        # Falling market → whales more likely to accumulate (buy the dip)
        # Rising market (late) → whales more likely to distribute
        mom = momentum[i]
        if mom < -0.10:
            p_accum = 0.70
        elif mom < 0:
            p_accum = 0.55
        elif mom < 0.10:
            p_accum = 0.45
        else:
            p_accum = 0.35  # late bull market → whale distribution

        btc_accum   = 0.0
        btc_distrib = 0.0
        n_accum   = 0
        n_distrib = 0

        for _ in range(n_txs):
            # Log-normal tx size, clipped to whale range
            size = np.clip(rng.lognormal(mean=np.log(300), sigma=0.8), _MIN_WHALE_TX_BTC, 10_000)
            if rng.random() < p_accum:
                btc_accum += size
                n_accum   += 1
            else:
                btc_distrib += size
                n_distrib   += 1

        rows.append({
            "date":       price_series.index[i],
            "price":      prices[i],
            "btc_accum":  btc_accum,
            "btc_distrib": btc_distrib,
            "n_accum":    n_accum,
            "n_distrib":  n_distrib,
        })

    return pd.DataFrame(rows).set_index("date")


def compute_whale_tracker(
    price_series: pd.Series,
    symbol: str = "BTC-USD",
) -> WhaleTrackerResult:
    """Compute whale flow metrics from price history simulation.

    Parameters
    ----------
    price_series:
        Daily close prices, DatetimeIndex, at least 60 bars.
    symbol:
        Ticker label.
    """
    if price_series.empty or len(price_series) < 60:
        raise ValueError("Need at least 60 price bars for whale tracker")

    price_series = price_series.dropna().sort_index()
    logger.info("WhaleTracker: simulating whale activity (%d bars)", len(price_series))

    df = _simulate_whale_activity(price_series)

    # Net flow = accumulation - distribution (BTC)
    df["net_flow"] = df["btc_accum"] - df["btc_distrib"]
    df["tx_count"] = df["n_accum"] + df["n_distrib"]

    def rolling_sum(col: str, days: int) -> float:
        return float(df[col].iloc[-days:].sum()) if len(df) >= days else float(df[col].sum())

    net_7d  = rolling_sum("net_flow", 7)
    net_30d = rolling_sum("net_flow", 30)
    tx_7d   = int(rolling_sum("tx_count", 7))
    tx_30d  = int(rolling_sum("tx_count", 30))

    # Large tx alert: any single day with >5000 BTC total whale flow in last 3d
    recent_3d   = df.iloc[-3:]
    large_alert = bool((recent_3d["btc_accum"] + recent_3d["btc_distrib"]).max() > 5_000)

    # Dominant regime
    if net_30d > 500:
        regime = "ACCUMULATION"
    elif net_30d < -500:
        regime = "DISTRIBUTION"
    else:
        regime = "NEUTRAL"

    signal = _whale_flow_to_signal(net_7d, net_30d)

    # Build recent transactions list (last 7 trading days, one per day)
    recent_txs: List[WhaleTransaction] = []
    for date, row in df.iloc[-7:].iterrows():
        if row["n_accum"] + row["n_distrib"] == 0:
            continue
        dominant = "accumulation" if row["btc_accum"] >= row["btc_distrib"] else "distribution"
        recent_txs.append(WhaleTransaction(
            date=str(date.date()),
            btc_amount=round(max(row["btc_accum"], row["btc_distrib"]), 1),
            tx_type=dominant,
            price_usd=round(float(row["price"]), 0),
            usd_value=round(max(row["btc_accum"], row["btc_distrib"]) * float(row["price"]), 0),
        ))

    return WhaleTrackerResult(
        symbol=symbol,
        net_flow_7d=round(net_7d, 1),
        net_flow_30d=round(net_30d, 1),
        whale_tx_count_7d=tx_7d,
        whale_tx_count_30d=tx_30d,
        dominant_regime=regime,
        large_tx_alert=large_alert,
        signal=signal,
        recent_transactions=recent_txs,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _whale_flow_to_signal(net_7d: float, net_30d: float) -> float:
    """Map net whale BTC flow to [-1, +1] signal.

    Scale: ±5000 BTC net flow over 30 days → ±1.0 signal.
    Short-term (7d) gets 40% weight, medium-term (30d) gets 60%.
    """
    score_7d  = np.clip(net_7d  / 2_000, -1.0, 1.0)
    score_30d = np.clip(net_30d / 5_000, -1.0, 1.0)
    return float(np.clip(0.4 * score_7d + 0.6 * score_30d, -1.0, 1.0))


def whale_summary(result: WhaleTrackerResult) -> str:
    alert_str = " [LARGE TX ALERT]" if result.large_tx_alert else ""
    return (
        f"WhaleFlow 7d={result.net_flow_7d:+,.0f} BTC, 30d={result.net_flow_30d:+,.0f} BTC "
        f"({result.dominant_regime}) signal={result.signal:+.2f}{alert_str}"
    )
