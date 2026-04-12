"""
market_microstructure_analysis.py
==================================
Post-hoc microstructure analysis of trained MARL agents.

Analyses:
  - Bid-ask spread evolution (time-series + distribution)
  - Price impact of agent trades (temporary & permanent)
  - Adverse selection analysis (fill quality vs subsequent price move)
  - Inventory turnover statistics
  - P&L attribution (alpha vs execution quality)
  - Comparison to Avellaneda-Stoikov theoretical optimal
  - Emergent market maker behaviour detection
  - Agent interaction patterns (who trades with whom)
  - Order flow toxicity (VPIN)
  - Market quality metrics (depth, resilience)
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TradeRecord:
    """Record of a single fill for microstructure analysis."""
    timestamp: int
    agent_id: str
    side: int              # +1 buy, -1 sell
    price: float
    size: float
    mid_price_at_fill: float
    mid_price_5s_later: float = 0.0    # for adverse selection
    mid_price_10s_later: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    vwap_at_fill: float = 0.0
    spread_at_fill: float = 0.0
    inventory_before: float = 0.0
    inventory_after: float = 0.0
    order_type: str = "market"   # "market" or "limit"
    episode_id: int = 0

    @property
    def implementation_shortfall(self) -> float:
        """Cost vs mid at time of decision."""
        return self.side * (self.price - self.mid_price_at_fill) * self.size

    @property
    def adverse_selection_5(self) -> float:
        """Price move in fill direction within 5 ticks: negative = adverse."""
        return self.side * (self.mid_price_5s_later - self.mid_price_at_fill)

    @property
    def passive_fill(self) -> bool:
        return self.order_type == "limit"


@dataclasses.dataclass
class EpisodeRecord:
    """Complete record of one agent's episode for analysis."""
    agent_id: str
    episode_id: int
    trades: List[TradeRecord]
    prices: np.ndarray           # mid prices, shape (T,)
    spreads: np.ndarray          # shape (T,)
    inventories: np.ndarray      # shape (T,)
    pnl_series: np.ndarray       # shape (T,)
    bid_depths: np.ndarray       # shape (T, depth)
    ask_depths: np.ndarray       # shape (T, depth)
    timestamps: np.ndarray       # shape (T,)

    @property
    def n_ticks(self) -> int:
        return len(self.prices)

    @property
    def total_pnl(self) -> float:
        return float(self.pnl_series[-1]) if len(self.pnl_series) > 0 else 0.0

    @property
    def total_volume(self) -> float:
        return sum(abs(t.size) for t in self.trades)


# ---------------------------------------------------------------------------
# Spread analyser
# ---------------------------------------------------------------------------

class SpreadAnalyser:
    """Analysis of bid-ask spread dynamics."""

    def analyse(self, records: List[EpisodeRecord]) -> Dict[str, Any]:
        all_spreads = np.concatenate([r.spreads for r in records if len(r.spreads) > 0])
        if len(all_spreads) == 0:
            return {}

        return {
            "mean_spread": float(all_spreads.mean()),
            "median_spread": float(np.median(all_spreads)),
            "std_spread": float(all_spreads.std()),
            "p5_spread": float(np.percentile(all_spreads, 5)),
            "p95_spread": float(np.percentile(all_spreads, 95)),
            "spread_autocorr_1": float(
                np.corrcoef(all_spreads[:-1], all_spreads[1:])[0, 1]
                if len(all_spreads) > 2 else 0.0
            ),
            "fraction_tight_spread": float(
                (all_spreads < np.median(all_spreads)).mean()
            ),
        }

    def spread_time_series(self, record: EpisodeRecord,
                            window: int = 50) -> np.ndarray:
        """Rolling mean spread."""
        spreads = record.spreads
        if len(spreads) < window:
            return spreads
        return np.convolve(spreads, np.ones(window) / window, mode="valid")

    def spread_distribution_by_regime(
        self, records: List[EpisodeRecord],
        volatility_threshold: float = 0.003,
    ) -> Dict[str, np.ndarray]:
        low_vol_spreads = []
        high_vol_spreads = []
        for r in records:
            if len(r.prices) < 2:
                continue
            rets = np.diff(r.prices) / (r.prices[:-1] + 1e-9)
            vol = float(rets.std())
            if vol < volatility_threshold:
                low_vol_spreads.extend(r.spreads.tolist())
            else:
                high_vol_spreads.extend(r.spreads.tolist())
        return {
            "low_vol": np.array(low_vol_spreads),
            "high_vol": np.array(high_vol_spreads),
        }


# ---------------------------------------------------------------------------
# Price impact analyser
# ---------------------------------------------------------------------------

class PriceImpactAnalyser:
    """
    Estimates price impact of agent trades.
    Separates temporary (intraday reverting) from permanent impact.
    """

    def analyse_impact(self, trades: List[TradeRecord],
                        prices: np.ndarray,
                        window: int = 20) -> Dict[str, Any]:
        if not trades:
            return {"mean_temporary_impact": 0.0, "mean_permanent_impact": 0.0}

        temp_impacts = []
        perm_impacts = []

        for trade in trades:
            t = trade.timestamp
            if t < 0 or t >= len(prices):
                continue

            pre_price = prices[max(0, t - 5):t].mean() if t > 5 else prices[t]
            post_short = prices[t:min(len(prices), t + window // 2)].mean()
            post_long = prices[t:min(len(prices), t + window)].mean()

            # Temporary: price reverts
            temp = trade.side * (post_short - trade.price)
            perm = trade.side * (post_long - pre_price)

            temp_impacts.append(temp)
            perm_impacts.append(perm)

        return {
            "mean_temporary_impact": float(np.mean(temp_impacts)) if temp_impacts else 0.0,
            "mean_permanent_impact": float(np.mean(perm_impacts)) if perm_impacts else 0.0,
            "std_temporary_impact": float(np.std(temp_impacts)) if temp_impacts else 0.0,
            "price_impact_coefficient": self._estimate_kyle_lambda(trades, prices),
        }

    def _estimate_kyle_lambda(self, trades: List[TradeRecord],
                               prices: np.ndarray) -> float:
        """Estimate Kyle's lambda: price impact per unit volume."""
        if len(trades) < 10:
            return 0.0

        order_flow = np.array([t.side * t.size for t in trades if t.timestamp < len(prices)])
        price_changes = np.array([
            prices[min(t.timestamp + 1, len(prices) - 1)] - prices[t.timestamp]
            for t in trades if t.timestamp < len(prices)
        ])

        if len(order_flow) < 2 or order_flow.std() < 1e-10:
            return 0.0

        # OLS: delta_p = lambda * q + epsilon
        q_std = order_flow.std() + 1e-9
        p_std = price_changes.std() + 1e-9
        lam = float(np.cov(order_flow, price_changes)[0, 1] / (order_flow.var() + 1e-9))
        return lam


# ---------------------------------------------------------------------------
# Adverse selection analyser
# ---------------------------------------------------------------------------

class AdverseSelectionAnalyser:
    """
    Measures adverse selection experienced by market makers.
    High adverse selection = informed flow taking liquidity.
    """

    def analyse(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        passive = [t for t in trades if t.passive_fill]
        aggressive = [t for t in trades if not t.passive_fill]

        passive_as = [t.adverse_selection_5 for t in passive]
        aggressive_as = [t.adverse_selection_5 for t in aggressive]

        return {
            "mean_adverse_selection_passive": float(np.mean(passive_as)) if passive_as else 0.0,
            "mean_adverse_selection_aggressive": float(np.mean(aggressive_as)) if aggressive_as else 0.0,
            "n_passive_fills": len(passive),
            "n_aggressive_fills": len(aggressive),
            "passive_fill_rate": len(passive) / max(len(trades), 1),
            "toxic_order_flow_fraction": self._toxicity(passive),
        }

    def _toxicity(self, passive_trades: List[TradeRecord]) -> float:
        """Fraction of passive fills with adverse selection (price moved against MM)."""
        if not passive_trades:
            return 0.0
        toxic = [t for t in passive_trades if t.adverse_selection_5 < 0]
        return len(toxic) / len(passive_trades)

    def compute_vpin(self, trades: List[TradeRecord],
                     bucket_size: float = 100.0) -> float:
        """
        VPIN (Volume-Synchronised Probability of Informed Trading).
        Estimates probability that a given volume bucket is from informed traders.
        """
        if not trades:
            return 0.0

        # Accumulate volume buckets
        buy_vol_buckets = []
        sell_vol_buckets = []
        curr_buy = 0.0
        curr_sell = 0.0
        curr_vol = 0.0

        for trade in trades:
            if trade.side > 0:
                curr_buy += trade.size
            else:
                curr_sell += trade.size
            curr_vol += trade.size

            if curr_vol >= bucket_size:
                buy_vol_buckets.append(curr_buy)
                sell_vol_buckets.append(curr_sell)
                curr_buy = curr_sell = curr_vol = 0.0

        if not buy_vol_buckets:
            return 0.0

        n_buckets = min(50, len(buy_vol_buckets))
        imbalances = [abs(b - s) / (b + s + 1e-9)
                      for b, s in zip(buy_vol_buckets[-n_buckets:],
                                      sell_vol_buckets[-n_buckets:])]
        return float(np.mean(imbalances))


# ---------------------------------------------------------------------------
# Inventory turnover
# ---------------------------------------------------------------------------

class InventoryAnalyser:
    """Analyses inventory management and turnover."""

    def analyse(self, record: EpisodeRecord) -> Dict[str, Any]:
        inv = record.inventories
        trades = record.trades

        if len(inv) == 0:
            return {}

        # Inventory stats
        max_long = float(np.maximum(inv, 0).max())
        max_short = float(abs(np.minimum(inv, 0).min()))
        mean_abs_inv = float(np.abs(inv).mean())

        # Turnover: total volume / mean abs inventory
        total_vol = sum(t.size for t in trades)
        turnover = total_vol / max(mean_abs_inv, 1.0)

        # Time at zero inventory
        t_flat = float((np.abs(inv) < 0.5).mean())

        # Inventory half-life (how long it takes to reduce by half)
        half_life = self._estimate_half_life(inv)

        # Mean reversion of inventory
        if len(inv) > 2:
            inv_ar = float(np.corrcoef(inv[:-1], inv[1:])[0, 1])
        else:
            inv_ar = 0.0

        return {
            "max_long_position": max_long,
            "max_short_position": max_short,
            "mean_abs_inventory": mean_abs_inv,
            "inventory_turnover": turnover,
            "time_at_flat": t_flat,
            "inventory_half_life": half_life,
            "inventory_autocorrelation": inv_ar,
            "inventory_std": float(inv.std()),
        }

    def _estimate_half_life(self, inv: np.ndarray) -> float:
        """Ornstein-Uhlenbeck half-life of inventory."""
        if len(inv) < 10:
            return float("inf")
        delta_inv = np.diff(inv)
        lagged = inv[:-1]
        if lagged.std() < 1e-9:
            return float("inf")
        # AR(1) coefficient
        rho = float(np.cov(delta_inv, lagged)[0, 1] / (lagged.var() + 1e-9))
        if rho >= 0:
            return float("inf")
        return float(-math.log(2) / math.log(1 + rho))


# ---------------------------------------------------------------------------
# P&L attribution
# ---------------------------------------------------------------------------

class PnLAttributor:
    """
    Decomposes P&L into:
      - Alpha (directional from price prediction)
      - Market making (spread capture)
      - Execution cost (slippage + impact)
      - Inventory holding (mark-to-market PnL on position)
    """

    def attribute(self, record: EpisodeRecord) -> Dict[str, float]:
        trades = record.trades
        prices = record.prices

        if len(trades) == 0 or len(prices) == 0:
            return {}

        spread_capture = 0.0
        slippage_cost = 0.0
        alpha_pnl = 0.0

        for trade in trades:
            t = trade.timestamp
            if t >= len(prices):
                continue

            # Spread capture (passive fills earn half spread)
            if trade.passive_fill and trade.spread_at_fill > 0:
                spread_capture += trade.spread_at_fill / 2 * trade.size

            # Slippage cost
            slippage_cost += trade.slippage * trade.size

            # Alpha: directional prediction (simplified: sign of trade vs subsequent return)
            future_t = min(t + 10, len(prices) - 1)
            future_return = (prices[future_t] - prices[t]) / (prices[t] + 1e-9)
            alpha_pnl += trade.side * future_return * prices[t] * trade.size * 0.1

        # Inventory holding PnL
        inv_pnl = float(record.pnl_series[-1]) - spread_capture - alpha_pnl + slippage_cost \
            if len(record.pnl_series) > 0 else 0.0

        return {
            "total_pnl": float(record.pnl_series[-1]) if len(record.pnl_series) > 0 else 0.0,
            "spread_capture": spread_capture,
            "slippage_cost": -slippage_cost,
            "alpha_pnl": alpha_pnl,
            "inventory_holding_pnl": inv_pnl,
        }


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov optimal comparison
# ---------------------------------------------------------------------------

class AvellanedaStoikovBenchmark:
    """
    Computes the Avellaneda-Stoikov (AS) theoretical optimal quotes
    and compares them to agent behaviour.

    AS model:
      reservation price: r = mid - q * gamma * sigma^2 * (T - t)
      optimal spread: delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)
    """

    def __init__(self, gamma: float = 0.1, k: float = 1.5, sigma: float = 0.0002):
        self.gamma = gamma
        self.k = k
        self.sigma = sigma

    def optimal_quotes(self, mid: float, inventory: float,
                        time_remaining: float) -> Tuple[float, float]:
        """Returns (optimal_bid, optimal_ask)."""
        reservation = mid - inventory * self.gamma * (self.sigma ** 2) * time_remaining
        spread = (self.gamma * (self.sigma ** 2) * time_remaining +
                  (2 / self.gamma) * math.log(1 + self.gamma / self.k))
        spread = max(spread, 0.001)
        return reservation - spread / 2, reservation + spread / 2

    def compare_to_agent(self, record: EpisodeRecord) -> Dict[str, float]:
        T = record.n_ticks
        deviations_bid = []
        deviations_ask = []

        for i, (price, inv) in enumerate(zip(record.prices, record.inventories)):
            time_remaining = (T - i) / T
            opt_bid, opt_ask = self.optimal_quotes(price, inv, time_remaining)

            if len(record.bid_depths) > i and len(record.ask_depths) > i:
                agent_bid = float(record.bid_depths[i, 0]) if record.bid_depths.shape[1] > 0 else price
                agent_ask = float(record.ask_depths[i, 0]) if record.ask_depths.shape[1] > 0 else price
                deviations_bid.append(agent_bid - opt_bid)
                deviations_ask.append(agent_ask - opt_ask)

        return {
            "mean_bid_deviation": float(np.mean(deviations_bid)) if deviations_bid else 0.0,
            "mean_ask_deviation": float(np.mean(deviations_ask)) if deviations_ask else 0.0,
            "std_bid_deviation": float(np.std(deviations_bid)) if deviations_bid else 0.0,
            "tracking_error": float(np.sqrt(
                np.mean([d ** 2 for d in deviations_bid + deviations_ask])
            )) if (deviations_bid or deviations_ask) else 0.0,
        }


# ---------------------------------------------------------------------------
# Market maker behaviour detector
# ---------------------------------------------------------------------------

class MarketMakerDetector:
    """
    Detects whether an agent has learned market-maker behaviour:
      - Posts on both sides of the book simultaneously
      - Captures spread on passive fills
      - Keeps inventory near zero
      - High fill rate on limit orders
    """

    def classify_behaviour(self, record: EpisodeRecord,
                            threshold: float = 0.6) -> Dict[str, Any]:
        trades = record.trades
        if len(trades) == 0:
            return {"is_market_maker": False, "score": 0.0}

        # Signal 1: inventory mean reversion
        inv = record.inventories
        inv_mean = float(np.abs(inv).mean())
        inv_score = max(0.0, 1.0 - inv_mean / 200.0)

        # Signal 2: passive fill rate
        passive_rate = sum(1 for t in trades if t.passive_fill) / len(trades)

        # Signal 3: spread capture (positive = earned spread)
        spread_earned = sum(
            t.spread_at_fill / 2 * t.size for t in trades if t.passive_fill
        )
        spread_score = min(1.0, max(0.0, spread_earned / max(len(trades), 1) / 0.1))

        # Signal 4: alternating buy/sell pattern
        if len(trades) > 2:
            sides = [t.side for t in trades]
            alternations = sum(1 for i in range(1, len(sides)) if sides[i] != sides[i - 1])
            alternation_rate = alternations / (len(sides) - 1)
        else:
            alternation_rate = 0.0

        # Composite score
        mm_score = (0.3 * inv_score + 0.3 * passive_rate +
                    0.2 * spread_score + 0.2 * alternation_rate)

        return {
            "is_market_maker": mm_score >= threshold,
            "score": mm_score,
            "inventory_score": inv_score,
            "passive_fill_rate": passive_rate,
            "spread_capture_score": spread_score,
            "alternation_rate": alternation_rate,
        }

    def classify_momentum(self, record: EpisodeRecord) -> float:
        """Score for momentum-following behaviour [0,1]."""
        trades = record.trades
        prices = record.prices
        if len(trades) < 5 or len(prices) < 10:
            return 0.0

        aligned = 0
        for trade in trades:
            t = trade.timestamp
            if t < 5 or t >= len(prices):
                continue
            recent_return = prices[t] - prices[t - 5]
            if trade.side * recent_return > 0:
                aligned += 1

        return aligned / len(trades)


# ---------------------------------------------------------------------------
# Multi-agent interaction analyser
# ---------------------------------------------------------------------------

class AgentInteractionAnalyser:
    """
    Analyses interactions between multiple agents:
    who trades with whom, complementary vs competing strategies.
    """

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids

    def compute_correlation_matrix(
        self, records_by_agent: Dict[str, EpisodeRecord]
    ) -> np.ndarray:
        """Correlation matrix of agent P&L time series."""
        n = len(self.agent_ids)
        mat = np.eye(n)
        for i, aid_i in enumerate(self.agent_ids):
            for j, aid_j in enumerate(self.agent_ids):
                if i >= j:
                    continue
                r_i = records_by_agent.get(aid_i)
                r_j = records_by_agent.get(aid_j)
                if r_i and r_j and len(r_i.pnl_series) > 5:
                    n_common = min(len(r_i.pnl_series), len(r_j.pnl_series))
                    pnl_i = r_i.pnl_series[:n_common]
                    pnl_j = r_j.pnl_series[:n_common]
                    ret_i = np.diff(pnl_i)
                    ret_j = np.diff(pnl_j)
                    if ret_i.std() > 1e-9 and ret_j.std() > 1e-9:
                        corr = float(np.corrcoef(ret_i, ret_j)[0, 1])
                        mat[i, j] = corr
                        mat[j, i] = corr
        return mat

    def detect_herding(self, records: Dict[str, EpisodeRecord],
                        threshold: float = 0.7) -> bool:
        """Detect herding: many agents moving in the same direction."""
        mat = self.compute_correlation_matrix(records)
        n = len(self.agent_ids)
        upper = mat[np.triu_indices(n, k=1)]
        return float(upper.mean()) > threshold if len(upper) > 0 else False


# ---------------------------------------------------------------------------
# Full microstructure report
# ---------------------------------------------------------------------------

class MicrostructureReport:
    """
    Generates a comprehensive microstructure report from episode records.
    """

    def __init__(self, agent_ids: Optional[List[str]] = None):
        self.agent_ids = agent_ids or []
        self._spread_analyser = SpreadAnalyser()
        self._impact_analyser = PriceImpactAnalyser()
        self._adverse_sel = AdverseSelectionAnalyser()
        self._inv_analyser = InventoryAnalyser()
        self._pnl_attr = PnLAttributor()
        self._as_bench = AvellanedaStoikovBenchmark()
        self._mm_detector = MarketMakerDetector()
        if agent_ids:
            self._interaction = AgentInteractionAnalyser(agent_ids)
        else:
            self._interaction = None

    def generate(self, records: List[EpisodeRecord]) -> Dict[str, Any]:
        if not records:
            return {"error": "No records provided"}

        report: Dict[str, Any] = {
            "n_episodes": len(records),
            "n_trades_total": sum(len(r.trades) for r in records),
        }

        # Spread analysis
        report["spread"] = self._spread_analyser.analyse(records)

        # Price impact (use trades from all records)
        all_trades = [t for r in records for t in r.trades]
        all_prices = np.concatenate([r.prices for r in records if len(r.prices) > 0])
        report["price_impact"] = self._impact_analyser.analyse_impact(all_trades, all_prices)

        # Adverse selection
        report["adverse_selection"] = self._adverse_sel.analyse(all_trades)
        report["vpin"] = self._adverse_sel.compute_vpin(all_trades)

        # Inventory (per record average)
        inv_analyses = [self._inv_analyser.analyse(r) for r in records]
        report["inventory"] = {
            k: float(np.mean([a[k] for a in inv_analyses if k in a]))
            for k in (inv_analyses[0].keys() if inv_analyses else [])
        }

        # P&L attribution
        pnl_attrs = [self._pnl_attr.attribute(r) for r in records]
        report["pnl_attribution"] = {
            k: float(np.mean([a.get(k, 0.0) for a in pnl_attrs]))
            for k in ["total_pnl", "spread_capture", "slippage_cost",
                       "alpha_pnl", "inventory_holding_pnl"]
        }

        # Avellaneda-Stoikov comparison
        as_comps = [self._as_bench.compare_to_agent(r) for r in records]
        if as_comps:
            first = as_comps[0]
            report["avellaneda_stoikov"] = {
                k: float(np.mean([a.get(k, 0.0) for a in as_comps]))
                for k in first.keys()
            }

        # Behaviour classification
        mm_scores = [self._mm_detector.classify_behaviour(r) for r in records]
        report["behaviour"] = {
            "mean_mm_score": float(np.mean([s["score"] for s in mm_scores])),
            "fraction_market_maker": float(np.mean([s["is_market_maker"] for s in mm_scores])),
            "mean_alternation_rate": float(
                np.mean([s.get("alternation_rate", 0.0) for s in mm_scores])
            ),
        }

        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        print("=" * 60)
        print("MARKET MICROSTRUCTURE REPORT")
        print("=" * 60)
        for section, data in report.items():
            print(f"\n[{section.upper()}]")
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, float):
                        print(f"  {key:40s}: {val:+.6f}")
                    else:
                        print(f"  {key:40s}: {val}")
            else:
                print(f"  {data}")


# ---------------------------------------------------------------------------
# Market quality index
# ---------------------------------------------------------------------------

class MarketQualityIndex:
    """
    Composite index of market quality metrics, inspired by academic literature.
    Combines spread, depth, resilience, and impact into a single score.
    """

    def __init__(self,
                 spread_weight: float = 0.3,
                 depth_weight: float = 0.3,
                 resilience_weight: float = 0.2,
                 impact_weight: float = 0.2):
        self.weights = {
            "spread": spread_weight,
            "depth": depth_weight,
            "resilience": resilience_weight,
            "impact": impact_weight,
        }

    def compute(self, records: List[EpisodeRecord]) -> Dict[str, float]:
        if not records:
            return {"mqi": 0.0}

        spreads = np.concatenate([r.spreads for r in records if len(r.spreads) > 0])
        spread_score = max(0.0, 1.0 - float(spreads.mean()) / 0.1)

        depths = np.concatenate([r.bid_depths.sum(axis=1) + r.ask_depths.sum(axis=1)
                                  for r in records if r.bid_depths.shape[0] > 0])
        depth_score = min(1.0, float(depths.mean()) / 1000.0) if len(depths) > 0 else 0.5

        resilience_scores = []
        for r in records:
            if len(r.spreads) > 20:
                s_arr = r.spreads
                autocorr = float(np.corrcoef(s_arr[:-1], s_arr[1:])[0, 1])
                resilience_scores.append(max(0.0, 1.0 - autocorr))
        resilience_score = float(np.mean(resilience_scores)) if resilience_scores else 0.5

        # Impact: lower = better quality
        impact_analyser = PriceImpactAnalyser()
        all_trades = [t for r in records for t in r.trades]
        all_prices = np.concatenate([r.prices for r in records if len(r.prices) > 0])
        if all_trades and len(all_prices) > 0:
            impact_data = impact_analyser.analyse_impact(all_trades, all_prices)
            impact_score = max(0.0, 1.0 - abs(impact_data.get("price_impact_coefficient", 0)) * 1e5)
        else:
            impact_score = 0.5

        mqi = (self.weights["spread"] * spread_score +
               self.weights["depth"] * depth_score +
               self.weights["resilience"] * resilience_score +
               self.weights["impact"] * impact_score)

        return {
            "mqi": float(mqi),
            "spread_score": spread_score,
            "depth_score": depth_score,
            "resilience_score": resilience_score,
            "impact_score": impact_score,
        }


# ---------------------------------------------------------------------------
# Synthetic episode generator (for testing)
# ---------------------------------------------------------------------------

def generate_synthetic_episode(
    agent_id: str = "agent_0",
    n_ticks: int = 500,
    n_trades: int = 50,
    seed: int = 0,
) -> EpisodeRecord:
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.0002, n_ticks)))
    spreads = np.maximum(0.02, rng.normal(0.03, 0.005, n_ticks))
    inventories = np.cumsum(rng.normal(0, 1.0, n_ticks))
    pnl = inventories * prices / 1000 + rng.cumsum(rng.normal(0, 0.1, n_ticks))
    bid_depths = rng.exponential(10, (n_ticks, 5)).astype(np.float32)
    ask_depths = rng.exponential(10, (n_ticks, 5)).astype(np.float32)

    trades = []
    for i in range(n_trades):
        t = int(rng.integers(5, n_ticks - 5))
        side = int(rng.choice([-1, 1]))
        trades.append(TradeRecord(
            timestamp=t,
            agent_id=agent_id,
            side=side,
            price=prices[t] + side * spreads[t] / 2,
            size=float(rng.exponential(5)),
            mid_price_at_fill=prices[t],
            mid_price_5s_later=prices[min(t + 5, n_ticks - 1)],
            mid_price_10s_later=prices[min(t + 10, n_ticks - 1)],
            slippage=float(rng.exponential(0.005)),
            spread_at_fill=spreads[t],
            inventory_before=inventories[max(0, t - 1)],
            inventory_after=inventories[t],
            order_type=rng.choice(["limit", "market"]),
        ))

    return EpisodeRecord(
        agent_id=agent_id,
        episode_id=seed,
        trades=trades,
        prices=prices,
        spreads=spreads,
        inventories=inventories,
        pnl_series=pnl,
        bid_depths=bid_depths,
        ask_depths=ask_depths,
        timestamps=np.arange(n_ticks, dtype=float),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== market_microstructure_analysis.py smoke test ===")

    records = [generate_synthetic_episode("agent_0", seed=i) for i in range(10)]

    reporter = MicrostructureReport(agent_ids=["agent_0"])
    report = reporter.generate(records)
    reporter.print_report(report)

    mqi = MarketQualityIndex()
    quality = mqi.compute(records)
    print(f"\nMarket Quality Index: {quality}")

    print("\nAll smoke tests passed.")
