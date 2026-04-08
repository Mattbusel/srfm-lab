"""
Liquidation cascade detection and modeling.

Implements:
  - Leverage-adjusted liquidation threshold estimation
  - Cascade propagation model (contagion)
  - Liquidation heatmap construction
  - Cascade risk index
  - Reflexivity detector (price → liquidations → price feedback)
  - Recovery time estimator post-cascade
  - Long/short squeeze signals
  - Open interest depletion signal
  - Cross-exchange liquidation aggregation
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Liquidation Threshold Estimation ─────────────────────────────────────────

@dataclass
class LeverageBook:
    """Simulated distribution of leveraged positions."""
    price: float               # current price
    leverage_levels: np.ndarray   # e.g., [5, 10, 20, 50, 100]
    longs_usd: np.ndarray         # open long interest per leverage level ($)
    shorts_usd: np.ndarray        # open short interest per leverage level ($)

    def long_liq_price(self, leverage: float) -> float:
        """Price at which long position is liquidated (assuming entry = current)."""
        return self.price * (1 - 1 / leverage + 0.005)  # +0.5% maintenance margin

    def short_liq_price(self, leverage: float) -> float:
        """Price at which short position is liquidated."""
        return self.price * (1 + 1 / leverage - 0.005)


def liquidation_heatmap(
    book: LeverageBook,
    price_range_pct: float = 0.25,
    n_bins: int = 100,
) -> dict:
    """
    Build liquidation heatmap: USD value of liquidations at each price level.

    Returns dict with price levels and cumulative liq amounts.
    """
    p_min = book.price * (1 - price_range_pct)
    p_max = book.price * (1 + price_range_pct)
    price_grid = np.linspace(p_min, p_max, n_bins)

    liq_longs = np.zeros(n_bins)   # long liquidations triggered at each price
    liq_shorts = np.zeros(n_bins)  # short liquidations triggered at each price

    for i, lev in enumerate(book.leverage_levels):
        liq_long_price = book.long_liq_price(lev)
        liq_short_price = book.short_liq_price(lev)

        # Find nearest bin
        long_bin = np.argmin(np.abs(price_grid - liq_long_price))
        short_bin = np.argmin(np.abs(price_grid - liq_short_price))

        if 0 <= long_bin < n_bins:
            liq_longs[long_bin] += book.longs_usd[i]
        if 0 <= short_bin < n_bins:
            liq_shorts[short_bin] += book.shorts_usd[i]

    # Cumulative: if price moves down to level X, total longs liquidated
    cum_liq_longs = np.cumsum(liq_longs[::-1])[::-1]   # longs liquidated as price falls
    cum_liq_shorts = np.cumsum(liq_shorts)               # shorts liquidated as price rises

    return {
        "price_grid": price_grid,
        "liq_longs": liq_longs,
        "liq_shorts": liq_shorts,
        "cum_liq_longs_down": cum_liq_longs,
        "cum_liq_shorts_up": cum_liq_shorts,
        "max_liq_long_price": float(price_grid[np.argmax(liq_longs)]),
        "max_liq_short_price": float(price_grid[np.argmax(liq_shorts)]),
        "total_long_risk_usd": float(cum_liq_longs[0]),
        "total_short_risk_usd": float(cum_liq_shorts[-1]),
    }


# ── Cascade Propagation Model ─────────────────────────────────────────────────

@dataclass
class CascadeParams:
    price_impact_per_million: float = 0.001   # price move per $1M liquidated
    market_depth_usd: float = 50_000_000      # total market depth ($)
    recovery_halflife: float = 10.0           # periods for market to absorb impact


def simulate_cascade(
    initial_price: float,
    direction: int,           # -1 (down, long liq), +1 (up, short liq)
    liq_heatmap: dict,
    cascade_params: CascadeParams,
    max_steps: int = 50,
) -> dict:
    """
    Simulate liquidation cascade given an initial price shock.
    Models feedback: liq → price impact → more liq.
    """
    price = initial_price
    price_path = [price]
    liq_volumes = []
    total_liquidated = 0.0

    price_grid = liq_heatmap["price_grid"]
    if direction == -1:
        liq_profile = liq_heatmap["liq_longs"]
    else:
        liq_profile = liq_heatmap["liq_shorts"]

    triggered = np.zeros_like(liq_profile, dtype=bool)

    for step in range(max_steps):
        # Find new liquidations triggered at current price
        if direction == -1:
            newly_triggered = (price_grid <= price) & ~triggered
        else:
            newly_triggered = (price_grid >= price) & ~triggered

        liq_usd = float(liq_profile[newly_triggered].sum())
        triggered |= newly_triggered

        if liq_usd < 100_000:  # < $100K, cascade dies
            break

        # Price impact
        price_impact = direction * liq_usd * cascade_params.price_impact_per_million / 1_000_000
        price += price_impact
        price_path.append(price)
        liq_volumes.append(liq_usd)
        total_liquidated += liq_usd

    price_path = np.array(price_path)
    max_drawdown = float((initial_price - price_path.min()) / initial_price) if direction == -1 else \
                   float((price_path.max() - initial_price) / initial_price)

    return {
        "price_path": price_path,
        "liq_volumes": liq_volumes,
        "total_liquidated_usd": float(total_liquidated),
        "max_price_move_pct": float(abs(price_path[-1] - initial_price) / initial_price),
        "max_drawdown_pct": max_drawdown,
        "n_cascade_steps": len(liq_volumes),
        "cascade_exhausted": len(liq_volumes) < max_steps,
    }


# ── Cascade Risk Index ─────────────────────────────────────────────────────────

def cascade_risk_index(
    prices: np.ndarray,
    volumes: np.ndarray,
    open_interest: np.ndarray,
    funding_rate: Optional[np.ndarray] = None,
    window: int = 20,
) -> np.ndarray:
    """
    Composite cascade risk index (0-1 scale, higher = more risk).
    Combines OI/volume ratio, funding extremes, vol, and price momentum.
    """
    T = min(len(prices), len(volumes), len(open_interest))
    risk = np.zeros(T)

    for i in range(window, T):
        # Component 1: OI/Volume ratio (high ratio = crowded, cascade-prone)
        oi_vol = open_interest[i] / max(volumes[i - window: i].mean(), 1.0)
        oi_score = float(min(oi_vol / 3.0, 1.0))

        # Component 2: Vol spike (recent vol vs baseline)
        ret = np.diff(np.log(prices[i - window: i + 1]))
        vol_recent = ret[-5:].std() if len(ret) >= 5 else ret.std()
        vol_base = ret.std()
        vol_score = float(min(vol_recent / (vol_base + 1e-8) / 3.0, 1.0))

        # Component 3: Price momentum (trending = higher cascade risk)
        if len(prices[i - window: i]) >= window:
            trend = (prices[i] - prices[i - window]) / prices[i - window]
            momentum_score = float(min(abs(trend) * 10, 1.0))
        else:
            momentum_score = 0.0

        # Component 4: Funding extremes
        funding_score = 0.0
        if funding_rate is not None and i < len(funding_rate):
            fr = funding_rate[max(0, i - 10): i + 1]
            funding_score = float(min(abs(fr.mean()) / 0.01, 1.0))

        # Weighted composite
        risk[i] = 0.35 * oi_score + 0.30 * vol_score + 0.20 * momentum_score + 0.15 * funding_score

    return risk


# ── Reflexivity Detector ──────────────────────────────────────────────────────

def reflexivity_detector(
    prices: np.ndarray,
    liq_volumes: np.ndarray,
    lag: int = 1,
    window: int = 50,
) -> dict:
    """
    Detects reflexive feedback loops: price moves → liquidations → more price moves.
    Uses cross-correlation and Granger-causality proxy.
    """
    T = min(len(prices), len(liq_volumes))
    returns = np.diff(np.log(prices[:T]))
    liqs = liq_volumes[1:T]

    n = len(returns)
    if n < window + lag:
        return {"reflexivity_score": 0.0, "is_reflexive": False}

    # Cross-correlations
    xcorr_liq_to_ret = []   # liq at t-lag predicts return at t
    xcorr_ret_to_liq = []   # return at t-lag predicts liq at t

    for w_start in range(0, n - window, window // 2):
        w_end = w_start + window
        r_w = returns[w_start: w_end]
        l_w = liqs[w_start: w_end]

        if len(r_w) <= lag:
            continue

        c1 = float(np.corrcoef(l_w[:-lag], r_w[lag:])[0, 1])  # liq → ret
        c2 = float(np.corrcoef(r_w[:-lag], l_w[lag:])[0, 1])  # ret → liq

        xcorr_liq_to_ret.append(abs(c1))
        xcorr_ret_to_liq.append(abs(c2))

    if not xcorr_liq_to_ret:
        return {"reflexivity_score": 0.0, "is_reflexive": False}

    # Reflexivity = both directions are correlated
    forward_corr = float(np.mean(xcorr_liq_to_ret))
    backward_corr = float(np.mean(xcorr_ret_to_liq))
    reflexivity = float(math.sqrt(forward_corr * backward_corr))

    return {
        "reflexivity_score": reflexivity,
        "liq_causes_returns": forward_corr,
        "returns_cause_liq": backward_corr,
        "is_reflexive": reflexivity > 0.2,
        "feedback_loop_strength": float(
            "strong" if reflexivity > 0.4
            else "moderate" if reflexivity > 0.2
            else "weak"
        ) if False else (
            "strong" if reflexivity > 0.4 else "moderate" if reflexivity > 0.2 else "weak"
        ),
    }


# ── Long/Short Squeeze Signal ─────────────────────────────────────────────────

def squeeze_signal(
    prices: np.ndarray,
    open_interest: np.ndarray,
    funding_rate: Optional[np.ndarray] = None,
    volume: Optional[np.ndarray] = None,
    window: int = 20,
) -> dict:
    """
    Detect long squeeze and short squeeze conditions.

    Long squeeze: price falling fast + high long OI + positive funding
    Short squeeze: price rising fast + high short OI + negative funding
    """
    T = min(len(prices), len(open_interest))
    returns = np.diff(np.log(prices[:T]))

    long_squeeze_score = np.zeros(T - 1)
    short_squeeze_score = np.zeros(T - 1)

    for i in range(window, T - 1):
        # Recent momentum
        recent_ret = float(np.sum(returns[i - min(5, window): i]))

        # OI vs recent average
        oi_ratio = float(open_interest[i] / (open_interest[i - window: i].mean() + 1))
        oi_elevated = max(oi_ratio - 1.0, 0.0)  # 0 if normal, positive if elevated

        # Funding component
        funding_component = 0.0
        if funding_rate is not None and i < len(funding_rate):
            funding_component = float(funding_rate[i])

        # Volume spike
        vol_spike = 0.0
        if volume is not None and i < len(volume):
            vol_ratio = float(volume[i] / (volume[i - window: i].mean() + 1))
            vol_spike = max(vol_ratio - 1.5, 0.0)

        # Long squeeze: price down + OI elevated + positive funding
        if recent_ret < 0:
            long_squeeze_score[i] = (
                abs(recent_ret) * 20  # price impact
                + oi_elevated
                + max(funding_component * 100, 0)
                + vol_spike * 0.5
            )

        # Short squeeze: price up + OI elevated + negative funding
        if recent_ret > 0:
            short_squeeze_score[i] = (
                recent_ret * 20
                + oi_elevated
                + max(-funding_component * 100, 0)
                + vol_spike * 0.5
            )

    long_squeeze_score = np.clip(long_squeeze_score, 0, 1)
    short_squeeze_score = np.clip(short_squeeze_score, 0, 1)

    return {
        "long_squeeze_score": long_squeeze_score,
        "short_squeeze_score": short_squeeze_score,
        "current_long_squeeze": float(long_squeeze_score[-1]),
        "current_short_squeeze": float(short_squeeze_score[-1]),
        "long_squeeze_alert": bool(long_squeeze_score[-1] > 0.5),
        "short_squeeze_alert": bool(short_squeeze_score[-1] > 0.5),
    }


# ── Open Interest Depletion Signal ────────────────────────────────────────────

def oi_depletion_signal(
    open_interest: np.ndarray,
    prices: np.ndarray,
    window: int = 20,
) -> dict:
    """
    Signal based on OI depletion relative to price move.
    Sharp OI drop + large price move = cascade happened, position cleared.
    OI drop + small price move = organic deleveraging (healthier).
    """
    T = min(len(open_interest), len(prices))
    oi_changes = np.diff(open_interest[:T])
    price_changes = np.diff(np.log(prices[:T]))

    depletion_score = np.zeros(T - 1)

    for i in range(window, T - 1):
        # Normalized OI change
        oi_pct = oi_changes[i] / (open_interest[i] + 1)
        # Was there a big drop?
        if oi_pct < -0.02:  # >2% OI drop
            price_move = abs(price_changes[i])
            # Large price move with OI drop = cascade/forced unwind
            depletion_score[i] = abs(oi_pct) * (1 + price_move * 50)

    depletion_score = np.minimum(depletion_score, 1.0)

    # Post-cascade opportunity: OI depleted + volatility declining
    recent_oi_depletion = float(depletion_score[-window:].max()) if len(depletion_score) >= window else 0
    recent_vol = float(np.abs(price_changes[-window:]).std()) if len(price_changes) >= window else 0
    baseline_vol = float(np.abs(price_changes).std())

    post_cascade_recovery = bool(
        recent_oi_depletion > 0.3
        and recent_vol < baseline_vol
        and len(price_changes) > window
    )

    return {
        "depletion_score": depletion_score,
        "recent_max_depletion": recent_oi_depletion,
        "post_cascade_recovery_signal": post_cascade_recovery,
        "oi_trend": float(np.polyfit(np.arange(min(window, T-1)), open_interest[-(min(window, T-1)):], 1)[0]),
    }
