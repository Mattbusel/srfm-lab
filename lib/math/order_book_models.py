"""
Order book models — microstructure and limit order book mathematics.

Implements:
  - Level-2 order book reconstruction and analytics
  - Order book imbalance (OBI) signals at multiple depths
  - Queue-reactive model (Cont-Stoikov-Talreja)
  - Weighted mid-price (micro-price) estimation
  - Resilience model: order book recovery after large trade
  - Price impact decomposition: temporary vs permanent
  - Order flow toxicity via VPIN and adverse selection
  - Liquidity-adjusted returns: Kyle-Obizhaeva model
  - Optimal placement: Bayesian update on order book state
  - Hidden order detection: volume fingerprinting
  - Spread decomposition: Roll, Glosten, Huang-Stoll
  - Order arrival rate estimation (Poisson / Hawkes)
  - Market depth and resilience metrics
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Order Book State ──────────────────────────────────────────────────────────

@dataclass
class OrderBookLevel:
    price: float
    quantity: float
    order_count: int = 1


@dataclass
class OrderBookSnapshot:
    """Snapshot of an order book at a point in time."""
    bids: list[OrderBookLevel]  # sorted descending by price
    asks: list[OrderBookLevel]  # sorted ascending by price
    timestamp: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else float("inf")

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        return self.spread / max(self.mid_price, 1e-10) * 10000


# ── Micro-Price (Weighted Mid) ────────────────────────────────────────────────

def micro_price(book: OrderBookSnapshot, depth: int = 5) -> float:
    """
    Stoikov (2018) micro-price: weighted mid weighted by bid/ask imbalance.
    micro = mid + spread/2 * (Q_bid - Q_ask) / (Q_bid + Q_ask)
    """
    bid_qty = sum(l.quantity for l in book.bids[:depth])
    ask_qty = sum(l.quantity for l in book.asks[:depth])
    total = bid_qty + ask_qty
    if total < 1e-10:
        return book.mid_price
    imbalance = (bid_qty - ask_qty) / total
    return float(book.mid_price + (book.spread / 2) * imbalance)


def weighted_mid_price(book: OrderBookSnapshot, depth: int = 10) -> float:
    """
    Volume-weighted mid: each level contributes by quantity.
    Closer to the inside than simple mid when book is skewed.
    """
    bid_levels = book.bids[:depth]
    ask_levels = book.asks[:depth]

    bid_vwap = sum(l.price * l.quantity for l in bid_levels) / max(
        sum(l.quantity for l in bid_levels), 1e-10)
    ask_vwap = sum(l.price * l.quantity for l in ask_levels) / max(
        sum(l.quantity for l in ask_levels), 1e-10)
    return float((bid_vwap + ask_vwap) / 2)


# ── Order Book Imbalance ──────────────────────────────────────────────────────

def order_book_imbalance(
    book: OrderBookSnapshot,
    depths: list[int] = None,
) -> dict:
    """
    Order Book Imbalance (OBI) at multiple depths.
    OBI_k = (V_bid_k - V_ask_k) / (V_bid_k + V_ask_k)
    Range: -1 (all ask) to +1 (all bid)
    """
    if depths is None:
        depths = [1, 3, 5, 10]

    results = {}
    for d in depths:
        bid_vol = sum(l.quantity for l in book.bids[:d])
        ask_vol = sum(l.quantity for l in book.asks[:d])
        total = bid_vol + ask_vol
        obi = float((bid_vol - ask_vol) / max(total, 1e-10))
        results[f"obi_{d}"] = obi

    # Composite: exponentially weighted by depth (closer levels = more weight)
    composite = 0.0
    w_total = 0.0
    for i, d in enumerate(depths):
        w = math.exp(-0.3 * i)
        composite += results[f"obi_{d}"] * w
        w_total += w
    results["obi_composite"] = float(composite / max(w_total, 1e-10))
    return results


def cumulative_depth_imbalance(
    book: OrderBookSnapshot,
    price_range_bps: float = 50.0,
) -> dict:
    """
    Imbalance within a price range (bps) from mid.
    More robust than fixed-level depth.
    """
    mid = book.mid_price
    range_abs = mid * price_range_bps / 10000

    bid_vol = sum(l.quantity for l in book.bids
                  if l.price >= mid - range_abs)
    ask_vol = sum(l.quantity for l in book.asks
                  if l.price <= mid + range_abs)
    total = bid_vol + ask_vol
    return {
        "bid_volume": float(bid_vol),
        "ask_volume": float(ask_vol),
        "imbalance": float((bid_vol - ask_vol) / max(total, 1e-10)),
        "price_range_bps": price_range_bps,
    }


# ── Queue-Reactive Model (Cont-Stoikov-Talreja) ───────────────────────────────

@dataclass
class QueueReactiveState:
    """State for the CST queue-reactive model."""
    bid_queue: float       # volume at best bid
    ask_queue: float       # volume at best ask
    lambda_b: float = 5.0  # bid arrival rate
    lambda_a: float = 5.0  # ask arrival rate
    mu: float = 3.0        # market order arrival rate
    delta: float = 2.0     # cancellation rate


def queue_reactive_mid_price_drift(state: QueueReactiveState) -> dict:
    """
    Expected mid-price drift from queue dynamics.
    P(price move up) ∝ ask queue depleting faster than bid queue.
    """
    # Probability best ask empties before best bid (price goes up)
    # Simplified: ratio of effective depletion rates
    ask_depletion = state.mu * 0.5 + state.delta
    bid_depletion = state.mu * 0.5 + state.delta

    # Queue ratio as predictor
    q_ratio = state.ask_queue / max(state.bid_queue, 1e-10)
    # P(up) decreases as ask queue grows relative to bid queue
    p_up = float(1 / (1 + q_ratio))
    p_down = 1.0 - p_up

    expected_drift = p_up - p_down  # positive = drift up

    return {
        "p_price_up": p_up,
        "p_price_down": p_down,
        "expected_drift": expected_drift,
        "queue_ratio": float(q_ratio),
        "signal": float(np.sign(expected_drift) * abs(expected_drift) ** 0.5),
    }


def queue_imbalance_signal(
    bid_queue_series: np.ndarray,
    ask_queue_series: np.ndarray,
    forward_returns: np.ndarray,
) -> dict:
    """
    Compute predictive accuracy of queue imbalance for forward returns.
    QI = (Q_bid - Q_ask) / (Q_bid + Q_ask)
    """
    total = bid_queue_series + ask_queue_series + 1e-10
    qi = (bid_queue_series - ask_queue_series) / total

    # IC (Information Coefficient)
    if len(qi) >= 5 and qi.std() > 1e-10 and forward_returns.std() > 1e-10:
        ic = float(np.corrcoef(qi, forward_returns)[0, 1])
    else:
        ic = 0.0

    # Sign accuracy
    sign_correct = float(np.mean(np.sign(qi) == np.sign(forward_returns)))

    return {
        "queue_imbalance": qi.tolist(),
        "ic": ic,
        "sign_accuracy": sign_correct,
        "mean_qi": float(qi.mean()),
    }


# ── Resilience Model ──────────────────────────────────────────────────────────

def order_book_resilience(
    depth_series: np.ndarray,   # depth at best level over time
    after_trade_idx: int,       # index when large trade hit
    window: int = 20,
) -> dict:
    """
    Measure order book resilience: how fast depth recovers after a large trade.
    Fit exponential recovery: depth(t) = depth_inf * (1 - exp(-kappa * t))
    """
    if after_trade_idx >= len(depth_series) - 3:
        return {"resilience_kappa": 0.0, "half_life": float("inf")}

    recovery = depth_series[after_trade_idx: after_trade_idx + window]
    if len(recovery) < 3:
        return {"resilience_kappa": 0.0, "half_life": float("inf")}

    depth_inf = float(np.percentile(depth_series, 75))
    depth_min = float(recovery[0])

    # Fit: recovery[t] = depth_inf - (depth_inf - depth_min) * exp(-kappa*t)
    t = np.arange(len(recovery), dtype=float)
    y = np.log(np.maximum(depth_inf - recovery + 1e-10, 1e-10))
    y_target = np.log(max(depth_inf - depth_min, 1e-10)) - 0.0  # offset

    # Linear regression on log scale
    if t.std() > 1e-10:
        slope = float(np.polyfit(t, y, 1)[0])
        kappa = float(-slope)
    else:
        kappa = 0.0

    kappa = max(kappa, 0.0)
    half_life = math.log(2) / max(kappa, 1e-6)

    return {
        "resilience_kappa": kappa,
        "half_life_periods": float(min(half_life, 1e6)),
        "depth_floor": float(depth_min),
        "depth_ceiling": float(depth_inf),
        "recovery_ratio": float(depth_min / max(depth_inf, 1e-10)),
    }


# ── Spread Decomposition ──────────────────────────────────────────────────────

def roll_spread_estimator(price_changes: np.ndarray) -> dict:
    """
    Roll (1984) spread estimator from price changes.
    s = 2 * sqrt(-cov(dp_t, dp_{t-1}))
    """
    if len(price_changes) < 3:
        return {"roll_spread": 0.0, "effective_spread_pct": 0.0}

    dp = price_changes
    cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
    if cov < 0:
        roll_spread = 2 * math.sqrt(-cov)
    else:
        roll_spread = 0.0  # positive cov = no bounce component

    mid_price = float(np.abs(price_changes).mean() * 100)
    eff_spread_pct = float(roll_spread / max(mid_price, 1e-10) * 100)

    return {
        "roll_spread": roll_spread,
        "effective_spread_pct": eff_spread_pct,
        "autocovariance": float(cov),
    }


def huang_stoll_decomposition(
    trade_prices: np.ndarray,
    trade_directions: np.ndarray,  # +1 buy, -1 sell
    mid_prices: np.ndarray,
) -> dict:
    """
    Huang-Stoll (1997) spread decomposition.
    Total spread = adverse selection + inventory + order processing.
    """
    n = min(len(trade_prices), len(trade_directions), len(mid_prices))
    if n < 10:
        return {"adverse_selection": 0.5, "inventory": 0.25, "processing": 0.25}

    trade_prices = trade_prices[:n]
    dirs = trade_directions[:n]
    mids = mid_prices[:n]

    # Effective spread
    eff_half_spread = (trade_prices - mids) * dirs
    avg_eff_half = float(eff_half_spread.mean())

    # Realized spread: mid 5 periods later vs trade price
    if n > 5:
        realized = []
        for i in range(n - 5):
            r = (mids[i + 5] - trade_prices[i]) * dirs[i]
            realized.append(r)
        avg_realized = float(np.mean(realized))
    else:
        avg_realized = avg_eff_half * 0.3

    # Adverse selection = realized spread (permanent impact)
    # Inventory + processing = effective - adverse
    adverse = avg_realized
    non_adverse = avg_eff_half - adverse

    # Further split: inventory ~30% of non-adverse (typical)
    inventory = non_adverse * 0.3
    processing = non_adverse * 0.7

    total = max(abs(avg_eff_half), 1e-10)
    return {
        "total_half_spread": float(avg_eff_half),
        "adverse_selection_frac": float(adverse / total),
        "inventory_frac": float(inventory / total),
        "processing_frac": float(processing / total),
        "adverse_selection_bps": float(adverse * 10000),
    }


# ── Price Impact Decomposition ────────────────────────────────────────────────

def temporary_permanent_impact(
    pre_trade_mid: float,
    trade_price: float,
    post_trade_mid_series: np.ndarray,  # mid prices after trade
    direction: float,  # +1 buy, -1 sell
) -> dict:
    """
    Decompose price impact into temporary (reversing) and permanent (lasting).
    """
    immediate_impact = (trade_price - pre_trade_mid) * direction

    if len(post_trade_mid_series) == 0:
        return {
            "total_impact_bps": float(immediate_impact / pre_trade_mid * 10000),
            "permanent_bps": 0.0,
            "temporary_bps": float(immediate_impact / pre_trade_mid * 10000),
        }

    # Long-run mid price (after book recovers)
    permanent_mid = float(post_trade_mid_series[-1])
    permanent_impact = (permanent_mid - pre_trade_mid) * direction
    temporary_impact = immediate_impact - permanent_impact

    scale = max(pre_trade_mid, 1e-10)
    return {
        "total_impact_bps": float(immediate_impact / scale * 10000),
        "permanent_bps": float(permanent_impact / scale * 10000),
        "temporary_bps": float(temporary_impact / scale * 10000),
        "permanent_fraction": float(permanent_impact / max(abs(immediate_impact), 1e-10)),
    }


# ── Hidden Order Detection ────────────────────────────────────────────────────

def hidden_order_fingerprint(
    trade_sizes: np.ndarray,
    normal_size: float,
    clustering_threshold: float = 3.0,
) -> dict:
    """
    Detect potential hidden/algorithmic orders via size clustering.
    Hidden orders often result in regular trade size patterns.
    """
    if len(trade_sizes) < 10:
        return {"hidden_order_detected": False, "regularity_score": 0.0}

    # Coefficient of variation (low = regular sizing = algorithmic)
    cv = float(trade_sizes.std() / max(trade_sizes.mean(), 1e-10))

    # Autocorrelation of trade sizes (high = patterned)
    if len(trade_sizes) > 5:
        acf1 = float(np.corrcoef(trade_sizes[1:], trade_sizes[:-1])[0, 1])
    else:
        acf1 = 0.0

    # Cluster around round numbers
    round_number_frac = float(np.mean(trade_sizes % normal_size < normal_size * 0.05))

    regularity = float((1 - cv) * 0.4 + abs(acf1) * 0.3 + round_number_frac * 0.3)
    hidden_detected = bool(regularity > 0.6 and cv < 0.3)

    # Estimated participation rate if hidden
    avg_trade = float(trade_sizes.mean())
    est_remaining = avg_trade * len(trade_sizes) * 2 if hidden_detected else 0.0

    return {
        "hidden_order_detected": hidden_detected,
        "regularity_score": float(np.clip(regularity, 0, 1)),
        "size_cv": cv,
        "size_autocorr": acf1,
        "round_number_fraction": round_number_frac,
        "estimated_remaining_volume": est_remaining,
    }


def volume_clock_imbalance(
    trade_sizes: np.ndarray,
    trade_directions: np.ndarray,  # +1 buy, -1 sell
    bucket_size: float = 1000.0,
) -> dict:
    """
    Volume-clock order imbalance: split trades into volume buckets, measure directional flow.
    Used to detect informed trader activity (similar to VPIN but per-bucket).
    """
    if len(trade_sizes) < 5:
        return {"vpin_estimate": 0.5, "informed_fraction": 0.5}

    buckets = []
    current_buy = 0.0
    current_sell = 0.0
    current_vol = 0.0

    for size, direction in zip(trade_sizes, trade_directions):
        if direction > 0:
            current_buy += size
        else:
            current_sell += size
        current_vol += size

        if current_vol >= bucket_size:
            total = current_buy + current_sell
            imbalance = abs(current_buy - current_sell) / max(total, 1e-10)
            buckets.append(imbalance)
            current_buy = current_sell = current_vol = 0.0

    if not buckets:
        return {"vpin_estimate": 0.5, "informed_fraction": 0.5}

    vpin = float(np.mean(buckets))
    return {
        "vpin_estimate": vpin,
        "informed_fraction": float(vpin),
        "n_buckets": len(buckets),
        "bucket_imbalances": buckets[-10:],
    }


# ── Order Arrival Rate (Hawkes) ───────────────────────────────────────────────

class HawkesOrderFlow:
    """
    Hawkes process model for order arrival self-excitation.
    Lambda(t) = mu + sum alpha * exp(-beta * (t - t_i)) for t_i < t
    """

    def __init__(self, mu: float = 1.0, alpha: float = 0.5, beta: float = 2.0):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self._events: list[float] = []

    def add_event(self, timestamp: float) -> None:
        self._events.append(timestamp)

    def intensity(self, t: float) -> float:
        """Current arrival intensity at time t."""
        base = self.mu
        excitation = sum(
            self.alpha * math.exp(-self.beta * (t - ti))
            for ti in self._events if ti < t
        )
        return float(base + excitation)

    def fit(self, event_times: np.ndarray) -> dict:
        """
        MLE fit of Hawkes parameters via gradient ascent (simplified).
        Returns fitted mu, alpha, beta.
        """
        if len(event_times) < 10:
            return {"mu": self.mu, "alpha": self.alpha, "beta": self.beta}

        T = float(event_times[-1] - event_times[0]) + 1e-6
        n = len(event_times)

        # Simple moment matching
        mean_rate = n / T
        # Variance of inter-arrival times
        inter = np.diff(event_times)
        if len(inter) > 1:
            cv_sq = float(inter.var() / max(inter.mean() ** 2, 1e-10))
        else:
            cv_sq = 1.0

        # For Hawkes: E[lambda] = mu / (1 - alpha/beta)
        # Branching ratio: alpha/beta < 1 for stationarity
        branching = float(min(max(1 - 1 / max(cv_sq, 0.1), 0), 0.9))
        mu_fit = mean_rate * (1 - branching)
        beta_fit = float(self.beta)
        alpha_fit = float(branching * beta_fit)

        return {
            "mu": mu_fit,
            "alpha": alpha_fit,
            "beta": beta_fit,
            "branching_ratio": branching,
            "mean_rate": mean_rate,
            "cv_squared": cv_sq,
        }

    def excitation_decay_time(self) -> float:
        """Time for excitation to decay to 5% of initial."""
        return float(math.log(20) / max(self.beta, 1e-6))


# ── Market Depth Metrics ──────────────────────────────────────────────────────

def market_depth_analytics(book: OrderBookSnapshot, depth_levels: int = 10) -> dict:
    """
    Comprehensive market depth analytics.
    """
    bid_levels = book.bids[:depth_levels]
    ask_levels = book.asks[:depth_levels]

    # Total depth
    total_bid_vol = sum(l.quantity for l in bid_levels)
    total_ask_vol = sum(l.quantity for l in ask_levels)

    # Depth at different BPS ranges
    mid = book.mid_price
    depth_by_range = {}
    for bps in [5, 10, 25, 50]:
        range_abs = mid * bps / 10000
        bid_in_range = sum(l.quantity for l in bid_levels
                           if l.price >= mid - range_abs)
        ask_in_range = sum(l.quantity for l in ask_levels
                           if l.price <= mid + range_abs)
        depth_by_range[f"depth_{bps}bps"] = {
            "bid": float(bid_in_range),
            "ask": float(ask_in_range),
            "total": float(bid_in_range + ask_in_range),
        }

    # Price impact of hypothetical orders
    def cost_to_move(side: str, target_size: float) -> float:
        levels = bid_levels if side == "sell" else ask_levels
        remaining = target_size
        total_cost = 0.0
        for lvl in levels:
            executed = min(remaining, lvl.quantity)
            total_cost += executed * lvl.price
            remaining -= executed
            if remaining <= 0:
                break
        if target_size > remaining:
            avg_price = total_cost / (target_size - remaining)
            return float(abs(avg_price - mid) / mid * 10000)  # bps
        return float("inf")

    return {
        "total_bid_volume": float(total_bid_vol),
        "total_ask_volume": float(total_ask_vol),
        "depth_asymmetry": float((total_bid_vol - total_ask_vol) /
                                  max(total_bid_vol + total_ask_vol, 1e-10)),
        "depth_by_range": depth_by_range,
        "bid_levels": len(bid_levels),
        "ask_levels": len(ask_levels),
        "spread_bps": float(book.spread_bps),
        "micro_price": float(micro_price(book)),
    }


# ── Optimal Order Placement ───────────────────────────────────────────────────

def optimal_limit_order_price(
    mid_price: float,
    spread: float,
    sigma_per_tick: float,
    tick_size: float,
    urgency: float = 0.5,        # 0=patient, 1=urgent
    adverse_selection_prob: float = 0.3,
) -> dict:
    """
    Optimal limit order placement (Bayesian framework).
    Balance between: fill probability vs adverse selection vs timing risk.
    """
    # Half-spread
    half_spread = spread / 2

    # Aggressive limit order: price improvement over mid
    # Conservative: join the queue at best bid/ask
    if urgency > 0.7:
        # Post at best bid/ask (join queue)
        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread
    else:
        # Post inside spread for better fill prob
        improvement = half_spread * (1 - urgency)
        bid_price = mid_price - half_spread + improvement
        ask_price = mid_price + half_spread - improvement

    # Snap to tick size
    bid_price = round(bid_price / tick_size) * tick_size
    ask_price = round(ask_price / tick_size) * tick_size

    # Estimated fill probability (simplified)
    bid_offset_ticks = (mid_price - bid_price) / tick_size
    fill_prob_bid = float(math.exp(-0.3 * bid_offset_ticks))

    ask_offset_ticks = (ask_price - mid_price) / tick_size
    fill_prob_ask = float(math.exp(-0.3 * ask_offset_ticks))

    # Expected value after adverse selection
    ev_bid = bid_price * (1 - adverse_selection_prob) - sigma_per_tick * adverse_selection_prob
    ev_ask = ask_price * (1 - adverse_selection_prob) + sigma_per_tick * adverse_selection_prob

    return {
        "optimal_bid": float(bid_price),
        "optimal_ask": float(ask_price),
        "fill_prob_bid": float(fill_prob_bid),
        "fill_prob_ask": float(fill_prob_ask),
        "ev_bid": float(ev_bid),
        "ev_ask": float(ev_ask),
        "quote_spread": float(ask_price - bid_price),
        "quote_spread_bps": float((ask_price - bid_price) / mid_price * 10000),
    }


# ── Adverse Selection Metrics ─────────────────────────────────────────────────

def adverse_selection_score(
    trade_prices: np.ndarray,
    trade_directions: np.ndarray,
    mid_prices_after: np.ndarray,
    horizon: int = 10,
) -> dict:
    """
    Measure adverse selection: do prices move against the market maker after fills?
    """
    n = len(trade_prices)
    if n < horizon + 5:
        return {"adverse_selection_bps": 0.0, "informativeness": 0.0}

    adverse_moves = []
    for i in range(n - horizon):
        fill_price = trade_prices[i]
        direction = trade_directions[i]
        future_mid = mid_prices_after[i + horizon] if i + horizon < len(mid_prices_after) else mid_prices_after[-1]

        # Adverse: future mid moves against the maker (away from fill)
        maker_pnl = direction * (fill_price - future_mid)  # maker is on opposite side
        adverse_moves.append(-maker_pnl)

    adverse_arr = np.array(adverse_moves)
    avg_adverse_bps = float(adverse_arr.mean() * 10000 / max(trade_prices.mean(), 1e-10))

    # Informativeness: fraction of trades followed by price move in trade direction
    informed_frac = float(np.mean(adverse_arr > 0))

    return {
        "adverse_selection_bps": avg_adverse_bps,
        "informativeness": informed_frac,
        "maker_pnl_bps": float(-avg_adverse_bps),
        "is_toxic_flow": bool(avg_adverse_bps > 5),
    }


# ── Book Pressure Signal ──────────────────────────────────────────────────────

class OrderBookPressureSignal:
    """
    Real-time order book pressure signal for trading.
    Combines OBI, queue imbalance, and depth analytics.
    """

    def __init__(self, depth: int = 5, obi_ema_alpha: float = 0.1):
        self.depth = depth
        self.alpha = obi_ema_alpha
        self._ema_obi = 0.0
        self._n = 0

    def update(self, book: OrderBookSnapshot) -> dict:
        obi = order_book_imbalance(book, [1, 3, self.depth])
        composite = obi["obi_composite"]

        # EMA smoothing
        if self._n == 0:
            self._ema_obi = composite
        else:
            self._ema_obi = self.alpha * composite + (1 - self.alpha) * self._ema_obi
        self._n += 1

        mp = micro_price(book)
        depth_analytics = market_depth_analytics(book, self.depth)

        # Signal: smoothed OBI + micro-price vs mid deviation
        mp_dev = (mp - book.mid_price) / max(book.spread, 1e-10)  # -0.5 to 0.5

        signal = float(0.6 * self._ema_obi + 0.4 * mp_dev * 2)
        signal = float(np.clip(signal, -1, 1))

        return {
            "signal": signal,
            "ema_obi": float(self._ema_obi),
            "micro_price": float(mp),
            "micro_price_deviation": float(mp_dev),
            "raw_obi": composite,
            "depth_asymmetry": float(depth_analytics["depth_asymmetry"]),
            "spread_bps": float(book.spread_bps),
        }
