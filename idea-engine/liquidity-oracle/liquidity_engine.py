"""
Liquidity Oracle — real-time liquidity assessment and prediction engine.

Implements:
  - Multi-dimensional liquidity score (spread, depth, resilience, impact)
  - Liquidity regime detection: normal → thinning → drought → crisis
  - Intraday liquidity forecasting: predict spread/depth by time-of-day
  - Cross-asset liquidity contagion: when one market dries up, which follows?
  - Optimal execution timing: when is liquidity best?
  - Liquidity-adjusted returns: penalize illiquid positions
  - Dark pool liquidity estimation
  - Market maker inventory proxy
  - Liquidity premium estimation (Amihud, Pastor-Stambaugh)
  - Alert system: early warning of liquidity deterioration
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Liquidity Score ───────────────────────────────────────────────────────────

@dataclass
class LiquiditySnapshot:
    """Point-in-time liquidity measurement for an asset."""
    timestamp: float = 0.0
    bid_ask_spread_bps: float = 0.0
    depth_at_touch: float = 0.0       # shares/notional at best bid+ask
    depth_10bps: float = 0.0          # cumulative depth within 10bps of mid
    adv_20d: float = 0.0              # 20-day average daily volume
    price_impact_1pct_bps: float = 0.0  # impact of 1% ADV trade
    resilience_seconds: float = 0.0    # time to recover depth after large trade
    dark_pool_fill_rate: float = 0.0   # dark pool fill probability (0-1)
    maker_inventory_proxy: float = 0.0  # inferred MM inventory direction (-1 to +1)


@dataclass
class LiquidityScore:
    """Composite liquidity score and regime."""
    overall_score: float          # 0-1 (1 = most liquid)
    spread_score: float
    depth_score: float
    resilience_score: float
    impact_score: float
    volume_score: float
    regime: str                   # normal / thinning / drought / crisis
    percentile_rank: float        # vs own history
    z_score: float                # vs own history (negative = deteriorating)
    alert_level: str              # green / yellow / orange / red


class LiquidityScorer:
    """Score and track asset liquidity over time."""

    def __init__(
        self,
        history_window: int = 252,
        crisis_threshold: float = 0.25,
        drought_threshold: float = 0.40,
        thinning_threshold: float = 0.60,
    ):
        self.history_window = history_window
        self.crisis_threshold = crisis_threshold
        self.drought_threshold = drought_threshold
        self.thinning_threshold = thinning_threshold
        self._score_history: list[float] = []

    def score(self, snapshot: LiquiditySnapshot) -> LiquidityScore:
        """Compute composite liquidity score from snapshot."""
        # Individual components (higher = better)
        spread_score = float(max(1 - snapshot.bid_ask_spread_bps / 50, 0))
        depth_score = float(min(math.log10(max(snapshot.depth_10bps, 1)) / 7, 1.0))
        volume_score = float(min(math.log10(max(snapshot.adv_20d, 1)) / 9, 1.0))
        impact_score = float(max(1 - snapshot.price_impact_1pct_bps / 100, 0))
        resilience_score = float(max(1 - snapshot.resilience_seconds / 60, 0))

        # Composite
        overall = float(
            0.25 * spread_score +
            0.20 * depth_score +
            0.20 * volume_score +
            0.20 * impact_score +
            0.15 * resilience_score
        )

        # Update history
        self._score_history.append(overall)
        if len(self._score_history) > self.history_window:
            self._score_history = self._score_history[-self.history_window:]

        # Percentile rank vs own history
        history = np.array(self._score_history)
        percentile = float(np.mean(history <= overall))
        z_score = float((overall - history.mean()) / max(history.std(), 1e-10))

        # Regime classification
        if overall < self.crisis_threshold:
            regime = "crisis"
            alert = "red"
        elif overall < self.drought_threshold:
            regime = "drought"
            alert = "orange"
        elif overall < self.thinning_threshold:
            regime = "thinning"
            alert = "yellow"
        else:
            regime = "normal"
            alert = "green"

        return LiquidityScore(
            overall_score=overall,
            spread_score=spread_score,
            depth_score=depth_score,
            resilience_score=resilience_score,
            impact_score=impact_score,
            volume_score=volume_score,
            regime=regime,
            percentile_rank=percentile,
            z_score=z_score,
            alert_level=alert,
        )


# ── Liquidity Premium Estimation ──────────────────────────────────────────────

def amihud_illiquidity(
    returns: np.ndarray,
    volumes: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """
    Amihud (2002) illiquidity ratio: |r| / volume.
    Higher = more illiquid.
    """
    n = len(returns)
    illiq = np.zeros(n)
    for t in range(window, n):
        r = np.abs(returns[t - window: t])
        v = volumes[t - window: t] + 1e-10
        illiq[t] = float(np.mean(r / v))
    return illiq


def pastor_stambaugh_liquidity(
    returns: np.ndarray,
    signed_volumes: np.ndarray,  # positive = buy, negative = sell
    window: int = 63,
) -> np.ndarray:
    """
    Pastor-Stambaugh (2003) liquidity measure: sensitivity of returns to signed volume.
    gamma_t: regression of r_{t+1} on sign(v_t) * |v_t|
    Negative gamma = low liquidity (price reversal after volume).
    """
    n = len(returns)
    gamma = np.zeros(n)
    for t in range(window + 1, n):
        y = returns[t - window + 1: t + 1]      # forward returns
        x = signed_volumes[t - window: t]        # lagged signed volume
        if x.std() > 1e-10 and y.std() > 1e-10:
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            gamma[t] = float(np.dot(x_centered, y_centered) / (np.dot(x_centered, x_centered) + 1e-10))
    return gamma


def roll_spread_estimate(
    price_changes: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """Rolling Roll (1984) spread estimator."""
    n = len(price_changes)
    spread = np.zeros(n)
    for t in range(window + 1, n):
        dp = price_changes[t - window: t]
        cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
        spread[t] = 2 * math.sqrt(-cov) if cov < 0 else 0.0
    return spread


def liquidity_adjusted_returns(
    returns: np.ndarray,
    illiquidity: np.ndarray,
    penalty_coefficient: float = 0.5,
) -> np.ndarray:
    """
    Adjust returns for illiquidity.
    r_adj = r - penalty * illiquidity
    """
    # Normalize illiquidity to [0, 1]
    illiq_norm = illiquidity / (np.percentile(illiquidity[illiquidity > 0], 95) + 1e-10)
    illiq_norm = np.clip(illiq_norm, 0, 1)
    return returns - penalty_coefficient * illiq_norm * np.abs(returns)


# ── Intraday Liquidity Pattern ────────────────────────────────────────────────

class IntradayLiquidityModel:
    """Model intraday liquidity patterns (U-shape, J-shape)."""

    def __init__(self, n_buckets: int = 78):  # 5-min buckets in 6.5hr trading day
        self.n_buckets = n_buckets
        self._spread_by_bucket: list[list[float]] = [[] for _ in range(n_buckets)]
        self._volume_by_bucket: list[list[float]] = [[] for _ in range(n_buckets)]

    def add_observation(self, bucket: int, spread: float, volume: float) -> None:
        if 0 <= bucket < self.n_buckets:
            self._spread_by_bucket[bucket].append(spread)
            self._volume_by_bucket[bucket].append(volume)

    def get_pattern(self) -> dict:
        """Get average intraday liquidity pattern."""
        avg_spread = np.zeros(self.n_buckets)
        avg_volume = np.zeros(self.n_buckets)

        for i in range(self.n_buckets):
            if self._spread_by_bucket[i]:
                avg_spread[i] = float(np.mean(self._spread_by_bucket[i]))
            if self._volume_by_bucket[i]:
                avg_volume[i] = float(np.mean(self._volume_by_bucket[i]))

        # Best execution window: lowest spread + highest volume
        if avg_spread.sum() > 0 and avg_volume.sum() > 0:
            spread_norm = avg_spread / max(avg_spread.max(), 1e-10)
            volume_norm = avg_volume / max(avg_volume.max(), 1e-10)
            liquidity_score = (1 - spread_norm) * 0.5 + volume_norm * 0.5
            best_bucket = int(np.argmax(liquidity_score))
            worst_bucket = int(np.argmin(liquidity_score))
        else:
            best_bucket = self.n_buckets // 2  # mid-day default
            worst_bucket = 0  # open default

        return {
            "avg_spread_by_bucket": avg_spread.tolist(),
            "avg_volume_by_bucket": avg_volume.tolist(),
            "best_execution_bucket": best_bucket,
            "worst_execution_bucket": worst_bucket,
            "spread_range_bps": float(avg_spread.max() - avg_spread.min()) if avg_spread.sum() > 0 else 0.0,
        }

    def forecast_spread(self, bucket: int, n_ahead: int = 1) -> float:
        """Forecast spread for a future bucket."""
        target = (bucket + n_ahead) % self.n_buckets
        if self._spread_by_bucket[target]:
            return float(np.mean(self._spread_by_bucket[target]))
        return 0.0


# ── Cross-Asset Liquidity Contagion ───────────────────────────────────────────

def liquidity_contagion_matrix(
    liquidity_scores: np.ndarray,   # (T, N) liquidity scores over time
    lag: int = 1,
) -> dict:
    """
    Estimate liquidity contagion: does deterioration in asset i predict
    deterioration in asset j?
    """
    T, N = liquidity_scores.shape
    if T < lag + 10:
        return {"contagion_matrix": np.zeros((N, N)), "avg_contagion": 0.0}

    # Liquidity changes
    dliq = np.diff(liquidity_scores, axis=0)

    # Cross-correlation at lag
    contagion = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                x = dliq[:-lag, i]
                y = dliq[lag:, j]
                if x.std() > 1e-10 and y.std() > 1e-10:
                    contagion[i, j] = float(np.corrcoef(x, y)[0, 1])

    # Average contagion strength
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    avg = float(np.abs(contagion[mask]).mean())

    # Most contagious asset
    outgoing = np.abs(contagion).sum(axis=1)
    most_contagious = int(np.argmax(outgoing))

    # Most vulnerable asset
    incoming = np.abs(contagion).sum(axis=0)
    most_vulnerable = int(np.argmax(incoming))

    return {
        "contagion_matrix": contagion,
        "avg_contagion": avg,
        "most_contagious_asset": most_contagious,
        "most_vulnerable_asset": most_vulnerable,
        "outgoing_contagion": outgoing.tolist(),
        "incoming_vulnerability": incoming.tolist(),
    }


# ── Market Maker Inventory Estimation ─────────────────────────────────────────

def estimate_mm_inventory(
    trade_prices: np.ndarray,
    trade_directions: np.ndarray,  # +1 buy, -1 sell (from aggressor's perspective)
    trade_sizes: np.ndarray,
    window: int = 100,
) -> dict:
    """
    Estimate market maker inventory from trade flow.
    MM takes the opposite side: buys = MM selling, sells = MM buying.
    """
    n = len(trade_prices)
    if n < 10:
        return {"inventory_proxy": 0.0, "inventory_pressure": "neutral"}

    # Cumulative MM inventory (opposite of aggressor flow)
    mm_flow = -trade_directions * trade_sizes
    mm_inventory = np.cumsum(mm_flow)

    # Recent inventory
    recent = mm_inventory[-min(window, n):]
    current_inv = float(recent[-1])
    avg_inv = float(recent.mean())
    inv_std = float(recent.std() + 1e-10)

    # Normalized inventory proxy
    inv_z = float((current_inv - avg_inv) / inv_std)

    # When MM is long (positive inventory), they want to sell
    # This creates selling pressure → bearish signal
    if inv_z > 1.5:
        pressure = "mm_long_selling_pressure"
    elif inv_z < -1.5:
        pressure = "mm_short_buying_pressure"
    else:
        pressure = "neutral"

    return {
        "inventory_proxy": float(current_inv),
        "inventory_z_score": float(inv_z),
        "inventory_pressure": pressure,
        "signal": float(-np.tanh(inv_z / 2)),  # negative because MM wants to unwind
        "mm_inventory_series": mm_inventory[-20:].tolist(),
    }


# ── Dark Pool Analytics ───────────────────────────────────────────────────────

def dark_pool_analytics(
    dark_volume: np.ndarray,
    lit_volume: np.ndarray,
    dark_fill_sizes: np.ndarray,
    price_changes: np.ndarray,
) -> dict:
    """Analyze dark pool activity for liquidity insights."""
    n = len(dark_volume)
    if n < 5:
        return {"dark_pct": 0.0}

    total = dark_volume + lit_volume + 1e-10
    dark_pct = dark_volume / total

    avg_dark_pct = float(dark_pct.mean())

    # Dark pool activity as signal
    # High dark pool % = institutional activity = potential for large moves
    recent_dark = float(dark_pct[-5:].mean()) if n >= 5 else avg_dark_pct
    dark_z = float((recent_dark - avg_dark_pct) / max(float(dark_pct.std()), 1e-10))

    # Fill size analysis
    avg_fill = float(dark_fill_sizes.mean()) if len(dark_fill_sizes) > 0 else 0.0
    large_fill_frac = float(np.mean(dark_fill_sizes > avg_fill * 3)) if len(dark_fill_sizes) > 0 else 0.0

    # Dark pool activity vs subsequent price moves
    if n > 5:
        lead_lag = float(np.corrcoef(dark_pct[:-1], np.abs(price_changes[1:]))[0, 1])
    else:
        lead_lag = 0.0

    return {
        "avg_dark_pool_pct": avg_dark_pct,
        "recent_dark_pool_pct": recent_dark,
        "dark_z_score": dark_z,
        "avg_dark_fill_size": avg_fill,
        "large_fill_fraction": large_fill_frac,
        "dark_predicts_vol": float(lead_lag),
        "institutional_activity": "elevated" if dark_z > 1.5 else "normal" if dark_z > -1 else "low",
    }


# ── Liquidity Oracle Engine ──────────────────────────────────────────────────

class LiquidityOracle:
    """
    Master liquidity engine: combines all modules for comprehensive
    liquidity monitoring and prediction.
    """

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.scorers = [LiquidityScorer() for _ in range(n_assets)]
        self.intraday_models = [IntradayLiquidityModel() for _ in range(n_assets)]
        self._liquidity_history: list[np.ndarray] = []

    def update(
        self,
        snapshots: list[LiquiditySnapshot],
    ) -> list[LiquidityScore]:
        """Update all asset liquidity scores."""
        scores = []
        score_values = np.zeros(self.n_assets)
        for i, snap in enumerate(snapshots):
            if i < self.n_assets:
                score = self.scorers[i].score(snap)
                scores.append(score)
                score_values[i] = score.overall_score
        self._liquidity_history.append(score_values)
        return scores

    def market_liquidity_index(self) -> dict:
        """Aggregate market-wide liquidity index."""
        if not self._liquidity_history:
            return {"index": 0.5, "regime": "normal"}

        current = self._liquidity_history[-1]
        avg = float(current.mean())
        min_score = float(current.min())
        n_stressed = int(np.sum(current < 0.3))

        # Historical comparison
        if len(self._liquidity_history) > 20:
            history = np.array(self._liquidity_history[-252:])
            hist_avg = history.mean(axis=1)
            percentile = float(np.mean(hist_avg <= avg))
            z = float((avg - hist_avg.mean()) / max(hist_avg.std(), 1e-10))
        else:
            percentile = 0.5
            z = 0.0

        if avg < 0.25:
            regime = "crisis"
        elif avg < 0.40:
            regime = "stressed"
        elif avg < 0.55:
            regime = "below_normal"
        else:
            regime = "normal"

        return {
            "index": avg,
            "min_asset_score": min_score,
            "n_stressed_assets": n_stressed,
            "percentile": percentile,
            "z_score": z,
            "regime": regime,
            "n_assets": self.n_assets,
        }

    def contagion_risk(self) -> dict:
        """Assess liquidity contagion risk."""
        if len(self._liquidity_history) < 20:
            return {"avg_contagion": 0.0}
        scores = np.array(self._liquidity_history[-min(len(self._liquidity_history), 252):])
        return liquidity_contagion_matrix(scores, lag=1)

    def optimal_execution_window(self, asset_idx: int) -> dict:
        """Get best execution timing for an asset."""
        if asset_idx < len(self.intraday_models):
            return self.intraday_models[asset_idx].get_pattern()
        return {}

    def liquidity_alert(self) -> list[dict]:
        """Generate liquidity alerts."""
        alerts = []
        if not self._liquidity_history:
            return alerts

        current = self._liquidity_history[-1]
        for i in range(self.n_assets):
            if current[i] < 0.25:
                alerts.append({
                    "asset_idx": i,
                    "score": float(current[i]),
                    "level": "critical",
                    "message": f"Asset {i}: liquidity crisis (score={current[i]:.2f})",
                })
            elif current[i] < 0.40:
                alerts.append({
                    "asset_idx": i,
                    "score": float(current[i]),
                    "level": "warning",
                    "message": f"Asset {i}: liquidity deterioration (score={current[i]:.2f})",
                })

        # Cross-asset alert: if many assets stressed simultaneously
        n_stressed = int(np.sum(current < 0.40))
        if n_stressed > self.n_assets * 0.3:
            alerts.append({
                "asset_idx": -1,
                "score": float(current.mean()),
                "level": "systemic",
                "message": f"Systemic liquidity stress: {n_stressed}/{self.n_assets} assets below threshold",
            })

        return alerts
