"""
Order flow toxicity and adverse selection signals.

Implements:
  - VPIN (Volume-synchronized Probability of Informed Trading)
  - Kyle's lambda (price impact coefficient)
  - Amihud illiquidity ratio
  - Roll spread estimator
  - Hasbrouck information share
  - Toxic flow detection (asymmetric information events)
  - Order flow imbalance (OFI) signal
  - Trade direction classification (tick rule, Lee-Ready)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── VPIN ──────────────────────────────────────────────────────────────────────

def vpin(
    prices: np.ndarray,
    volumes: np.ndarray,
    n_buckets: int = 50,
    bucket_size: Optional[float] = None,
) -> float:
    """
    VPIN: Volume-synchronized Probability of Informed Trading.
    Easley, Lopez de Prado, O'Hara (2012).

    Estimates fraction of volume that is informed order flow.
    High VPIN → toxic flow → widen spreads / reduce exposure.
    """
    if len(prices) < n_buckets + 1:
        return 0.5

    returns = np.diff(np.log(prices))
    # Classify each bar as buy (up) or sell (down) volume
    buy_vol = np.where(returns >= 0, volumes[1:], 0.0)
    sell_vol = np.where(returns < 0, volumes[1:], 0.0)

    if bucket_size is None:
        total_vol = volumes[1:].sum()
        bucket_size = total_vol / n_buckets

    # Aggregate into volume buckets
    cum_vol = np.cumsum(volumes[1:])
    bucket_imbalances = []

    current_bucket = 0
    cum_buy = 0.0
    cum_sell = 0.0

    for i, (bv, sv, cv) in enumerate(zip(buy_vol, sell_vol, cum_vol)):
        cum_buy += bv
        cum_sell += sv
        bucket_bound = (current_bucket + 1) * bucket_size
        while cum_vol[i] >= bucket_bound:
            excess = cum_vol[i] - bucket_bound
            if volumes[i + 1] > 0:
                frac = min(excess / volumes[i + 1], 1.0)
                cum_buy -= bv * frac
                cum_sell -= sv * frac
            bucket_imbalances.append(abs(cum_buy - cum_sell) / (cum_buy + cum_sell + 1e-10))
            cum_buy = bv * frac if volumes[i + 1] > 0 else 0
            cum_sell = sv * frac if volumes[i + 1] > 0 else 0
            current_bucket += 1
            bucket_bound = (current_bucket + 1) * bucket_size

    if not bucket_imbalances:
        return 0.5

    # VPIN = average bucket imbalance over last n_buckets
    recent = bucket_imbalances[-n_buckets:]
    return float(np.mean(recent))


# ── Kyle's lambda ─────────────────────────────────────────────────────────────

def kyle_lambda(
    price_changes: np.ndarray,
    signed_volumes: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """
    Kyle's lambda: price impact per unit of order flow.
    Estimated via OLS: delta_p = lambda * signed_volume + epsilon

    signed_volume = (buy_vol - sell_vol)

    High lambda → illiquid, high impact → avoid large trades.
    """
    n = len(price_changes)
    lambdas = np.full(n, np.nan)

    for i in range(window, n):
        dp = price_changes[i - window: i]
        sv = signed_volumes[i - window: i]
        if sv.std() < 1e-10:
            continue
        # OLS
        cov = np.cov(dp, sv)
        lam = cov[0, 1] / max(cov[1, 1], 1e-10)
        lambdas[i] = float(lam)

    return lambdas


# ── Amihud illiquidity ────────────────────────────────────────────────────────

def amihud_illiquidity(
    returns: np.ndarray,
    volumes: np.ndarray,
    window: int = 20,
    scale: float = 1e6,
) -> np.ndarray:
    """
    Amihud (2002) illiquidity ratio: |r| / dollar_volume.
    Scaled by `scale` for readability.
    Higher value → less liquid → wider effective spread.
    """
    n = len(returns)
    illiq = np.full(n, np.nan)

    for i in range(window, n):
        r = returns[i - window: i]
        v = volumes[i - window: i]
        dv = v  # assume volumes are in dollar terms already
        ratio = np.abs(r) / (dv + 1e-10)
        illiq[i] = float(ratio.mean() * scale)

    return illiq


# ── Roll spread ───────────────────────────────────────────────────────────────

def roll_spread(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Roll (1984) spread estimator: 2 * sqrt(-Cov(dp_t, dp_{t-1})).
    Measures effective bid-ask spread from price changes.
    """
    n = len(prices)
    dp = np.diff(np.log(prices))
    spreads = np.full(n, np.nan)

    for i in range(window + 1, n):
        d1 = dp[i - window: i]
        d2 = dp[i - window - 1: i - 1]
        cov = float(np.cov(d1, d2)[0, 1])
        if cov < 0:
            spreads[i] = 2 * math.sqrt(-cov)
        else:
            spreads[i] = 0.0

    return spreads


# ── Order flow imbalance ──────────────────────────────────────────────────────

def order_flow_imbalance(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    mid_prices: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Order Flow Imbalance (OFI): measure of pressure in limit order book.
    OFI = delta(bid_qty) - delta(ask_qty)
    Higher OFI → more buy pressure → price likely to increase.
    """
    n = len(mid_prices)
    ofi = np.zeros(n)

    for i in range(1, n):
        # Bid changes: positive if bid queue grew
        delta_bid = (bid_sizes[i] - bid_sizes[i - 1]) if mid_prices[i] >= mid_prices[i - 1] else -bid_sizes[i]
        # Ask changes: positive if ask queue grew
        delta_ask = (ask_sizes[i] - ask_sizes[i - 1]) if mid_prices[i] <= mid_prices[i - 1] else -ask_sizes[i]
        ofi[i] = float(delta_bid - delta_ask)

    # Rolling normalized OFI
    ofi_smooth = np.convolve(ofi, np.ones(window) / window, mode="same")
    return ofi_smooth


# ── Toxic flow detector ───────────────────────────────────────────────────────

class ToxicFlowDetector:
    """
    Real-time toxic flow detection.
    Toxic flow = informed trading → adverse selection → expected loss to market makers.
    Signal: reduce position when toxic flow probability is high.
    """

    def __init__(
        self,
        vpin_window: int = 30,
        vpin_threshold: float = 0.7,
        kyle_window: int = 50,
        amihud_window: int = 20,
    ):
        self._vpin_window = vpin_window
        self._vpin_threshold = vpin_threshold
        self._kyle_window = kyle_window
        self._amihud_window = amihud_window

        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._returns: list[float] = []

    def update(self, price: float, volume: float) -> None:
        self._prices.append(price)
        self._volumes.append(volume)
        if len(self._prices) > 1:
            self._returns.append(math.log(price / self._prices[-2]))

    def signal(self) -> dict:
        if len(self._prices) < max(self._vpin_window, self._kyle_window) + 5:
            return {"toxic": False, "vpin": 0.5, "urgency": 0.0}

        prices = np.array(self._prices)
        volumes = np.array(self._volumes)
        returns = np.array(self._returns)

        # VPIN
        vpin_val = vpin(prices[-self._vpin_window * 2:], volumes[-self._vpin_window * 2:],
                        n_buckets=self._vpin_window)

        # Amihud
        amihud = amihud_illiquidity(returns, volumes[1:], self._amihud_window)
        amihud_recent = float(amihud[~np.isnan(amihud)][-1]) if len(amihud[~np.isnan(amihud)]) > 0 else 0.0

        # Kyle lambda proxy via recent price impact
        signed_vol = np.sign(returns[-self._kyle_window:]) * volumes[1:][-self._kyle_window:]
        kyle_lam = kyle_lambda(returns[-self._kyle_window:], signed_vol, window=self._kyle_window)
        kyle_recent = float(kyle_lam[~np.isnan(kyle_lam)][-1]) if len(kyle_lam[~np.isnan(kyle_lam)]) > 0 else 0.0

        toxic = bool(vpin_val > self._vpin_threshold)
        urgency = float((vpin_val - 0.5) / 0.5) if vpin_val > 0.5 else 0.0

        return {
            "toxic": toxic,
            "vpin": float(vpin_val),
            "amihud": float(amihud_recent),
            "kyle_lambda": float(kyle_recent),
            "urgency": float(np.clip(urgency, 0, 1)),
            "position_scale": float(max(1.0 - urgency, 0.2)),
        }
