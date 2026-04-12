"""
emergence.py — Market emergence analysis for Hyper-Agent MARL.

Implements:
- Market impact measurement (Kyle's lambda, Amihud illiquidity)
- Price discovery metrics (information share, contribution ratio)
- Liquidity emergence analysis (spread dynamics, depth resilience)
- Flash crash detection (volatility regimes, price cliff detection)
- Nash equilibrium approximation (exploitability, regret bounds)
- Market microstructure breakdown detection
"""

from __future__ import annotations

import math
import logging
import collections
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Market impact measurement
# ---------------------------------------------------------------------------

class MarketImpactAnalyzer:
    """
    Measures market impact of trades on prices.

    Metrics:
    - Kyle's lambda (permanent impact)
    - Amihud illiquidity ratio
    - Temporary vs permanent impact decomposition
    - Square root market impact model
    """

    def __init__(
        self,
        lookback: int = 100,
        decay_factor: float = 0.95,
    ):
        self.lookback = lookback
        self.decay_factor = decay_factor

        self._price_history: collections.deque = collections.deque(maxlen=lookback)
        self._volume_history: collections.deque = collections.deque(maxlen=lookback)
        self._return_history: collections.deque = collections.deque(maxlen=lookback)
        self._trade_sign_history: collections.deque = collections.deque(maxlen=lookback)

    def update(self, price: float, volume: float, trade_sign: int) -> None:
        """Add a trade observation. trade_sign: +1 for buy, -1 for sell."""
        if self._price_history:
            ret = (price - list(self._price_history)[-1]) / (list(self._price_history)[-1] + EPS)
            self._return_history.append(float(ret))
        else:
            self._return_history.append(0.0)

        self._price_history.append(float(price))
        self._volume_history.append(float(volume))
        self._trade_sign_history.append(int(trade_sign))

    def kyle_lambda(self) -> float:
        """
        Estimate Kyle's lambda (price impact coefficient).
        Regression: delta_price = lambda * order_flow + epsilon
        """
        if len(self._return_history) < 20:
            return 0.0

        returns = np.array(list(self._return_history))
        volumes = np.array(list(self._volume_history)[1:len(returns) + 1])
        signs = np.array(list(self._trade_sign_history)[1:len(returns) + 1])

        if len(volumes) != len(returns) or len(signs) != len(returns):
            return 0.0

        order_flow = volumes * signs

        # OLS regression
        var_of = float(np.var(order_flow))
        if var_of < EPS:
            return 0.0

        cov = float(np.cov(order_flow, returns)[0, 1])
        return cov / (var_of + EPS)

    def amihud_illiquidity(self) -> float:
        """
        Amihud illiquidity ratio: |r_t| / volume_t
        Higher value = less liquid.
        """
        if not self._return_history or not self._volume_history:
            return 0.0

        returns = np.array(list(self._return_history))
        volumes = np.array(list(self._volume_history))
        n = min(len(returns), len(volumes))
        if n == 0:
            return 0.0
        return float(np.mean(np.abs(returns[:n]) / (volumes[:n] + EPS)))

    def price_impact_curve(self, trade_sizes: np.ndarray) -> np.ndarray:
        """
        Estimate price impact as function of trade size.
        Uses square root market impact model: impact = sigma * sqrt(Q/ADV)
        """
        sigma = float(np.std(list(self._return_history))) if self._return_history else 0.01
        adv = float(np.mean(list(self._volume_history))) if self._volume_history else 100.0
        return sigma * np.sqrt(trade_sizes / (adv + EPS))

    def temporary_vs_permanent_impact(self) -> Dict[str, float]:
        """
        Decompose impact into temporary and permanent components.
        Uses Hasbrouck (1991) VAR approach.
        """
        if len(self._return_history) < 30:
            return {"temporary": 0.0, "permanent": 0.0, "ratio": 0.5}

        returns = np.array(list(self._return_history))
        # Autocorrelation at lag 1 (negative = mean reversion = temporary impact)
        if len(returns) < 2:
            return {"temporary": 0.0, "permanent": 0.0, "ratio": 0.5}

        ac1 = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        # Permanent: returns persist; Temporary: returns revert
        permanent_frac = max(0.0, min(1.0, 0.5 + 0.5 * (-ac1)))
        temporary_frac = 1.0 - permanent_frac

        return {
            "temporary": temporary_frac,
            "permanent": permanent_frac,
            "ratio": temporary_frac / (permanent_frac + EPS),
        }

    def summary(self) -> Dict[str, float]:
        return {
            "kyle_lambda": self.kyle_lambda(),
            "amihud_illiquidity": self.amihud_illiquidity(),
            **self.temporary_vs_permanent_impact(),
        }


# ---------------------------------------------------------------------------
# Price discovery metrics
# ---------------------------------------------------------------------------

class PriceDiscoveryAnalyzer:
    """
    Measures the efficiency of price discovery.

    Metrics:
    - Variance ratio test (Cochrane 1988)
    - Autocorrelation of returns
    - Information content of order flow
    - Price informativeness
    """

    def __init__(self, lookback: int = 200):
        self.lookback = lookback
        self._price_history: collections.deque = collections.deque(maxlen=lookback)
        self._fundamental_history: collections.deque = collections.deque(maxlen=lookback)

    def update(self, price: float, fundamental: Optional[float] = None) -> None:
        self._price_history.append(float(price))
        if fundamental is not None:
            self._fundamental_history.append(float(fundamental))

    def variance_ratio(self, k: int = 5) -> float:
        """
        Variance ratio test for random walk.
        VR = Var(k-period return) / (k * Var(1-period return))
        VR = 1 for random walk, <1 for mean reversion, >1 for momentum.
        """
        if len(self._price_history) < k * 3:
            return 1.0

        prices = np.array(list(self._price_history))
        log_p = np.log(prices + EPS)
        returns_1 = np.diff(log_p)
        if len(returns_1) < k:
            return 1.0
        returns_k = log_p[k:] - log_p[:-k]

        var_1 = float(np.var(returns_1)) + EPS
        var_k = float(np.var(returns_k)) + EPS
        return var_k / (k * var_1)

    def return_autocorrelation(self, max_lag: int = 10) -> Dict[int, float]:
        """Compute autocorrelation at multiple lags."""
        if len(self._price_history) < max_lag + 2:
            return {}
        prices = np.array(list(self._price_history))
        returns = np.diff(np.log(prices + EPS))
        result = {}
        for lag in range(1, max_lag + 1):
            if len(returns) > lag:
                ac = float(np.corrcoef(returns[:-lag], returns[lag:])[0, 1])
                result[lag] = ac
        return result

    def price_vs_fundamental_gap(self) -> float:
        """Measure deviation of price from fundamental value."""
        if not self._price_history or not self._fundamental_history:
            return float("nan")
        prices = np.array(list(self._price_history))
        fundamentals = np.array(list(self._fundamental_history))
        n = min(len(prices), len(fundamentals))
        if n == 0:
            return float("nan")
        gaps = np.abs(prices[:n] - fundamentals[:n]) / (fundamentals[:n] + EPS)
        return float(np.mean(gaps))

    def price_efficiency_score(self) -> float:
        """
        Composite efficiency score in [0, 1].
        1 = fully efficient (random walk, no gaps from fundamental).
        """
        vr = self.variance_ratio()
        ac = self.return_autocorrelation(3)

        # Score based on variance ratio being close to 1
        vr_score = max(0.0, 1.0 - abs(vr - 1.0))

        # Score based on low autocorrelation
        if ac:
            mean_ac = float(np.mean(list(ac.values())))
            ac_score = max(0.0, 1.0 - abs(mean_ac) * 2)
        else:
            ac_score = 0.5

        # Score based on fundamental gap
        gap = self.price_vs_fundamental_gap()
        if not math.isnan(gap):
            gap_score = max(0.0, 1.0 - gap * 10)
        else:
            gap_score = 0.5

        return float((vr_score + ac_score + gap_score) / 3.0)


# ---------------------------------------------------------------------------
# Liquidity emergence
# ---------------------------------------------------------------------------

class LiquidityEmergenceTracker:
    """
    Tracks how market liquidity emerges from agent interactions.
    """

    def __init__(self, lookback: int = 500):
        self.lookback = lookback
        self._spread_history: collections.deque = collections.deque(maxlen=lookback)
        self._depth_history: collections.deque = collections.deque(maxlen=lookback)
        self._volume_history: collections.deque = collections.deque(maxlen=lookback)
        self._trade_count_history: collections.deque = collections.deque(maxlen=lookback)

    def update(
        self,
        spread: float,
        total_depth: float,
        volume: float,
        trade_count: int,
    ) -> None:
        self._spread_history.append(float(spread) if not math.isnan(spread) else 0.0)
        self._depth_history.append(float(total_depth))
        self._volume_history.append(float(volume))
        self._trade_count_history.append(int(trade_count))

    def spread_stability(self) -> float:
        """How stable is the spread over time? (lower CV = more stable)"""
        if len(self._spread_history) < 10:
            return 0.0
        spreads = np.array(list(self._spread_history))
        valid = spreads[spreads > 0]
        if len(valid) == 0:
            return 0.0
        cv = float(np.std(valid) / (np.mean(valid) + EPS))
        return float(max(0.0, 1.0 - cv))

    def depth_resilience(self) -> float:
        """Measure how quickly depth recovers after shock (autocorrelation)."""
        if len(self._depth_history) < 20:
            return 0.5
        depths = np.array(list(self._depth_history))
        if len(depths) > 1:
            ac = float(np.corrcoef(depths[:-1], depths[1:])[0, 1])
            return float(max(0.0, ac))
        return 0.5

    def liquidity_provision_rate(self) -> float:
        """Average depth per unit time."""
        if not self._depth_history:
            return 0.0
        return float(np.mean(list(self._depth_history)))

    def market_activity(self) -> Dict[str, float]:
        if not self._volume_history:
            return {}
        return {
            "mean_volume": float(np.mean(list(self._volume_history))),
            "mean_trades": float(np.mean(list(self._trade_count_history))),
            "volume_variance": float(np.var(list(self._volume_history))),
        }

    def summary(self) -> Dict[str, float]:
        return {
            "spread_stability": self.spread_stability(),
            "depth_resilience": self.depth_resilience(),
            "liquidity_provision": self.liquidity_provision_rate(),
            **self.market_activity(),
        }


# ---------------------------------------------------------------------------
# Flash crash detection
# ---------------------------------------------------------------------------

class FlashCrashDetector:
    """
    Detects flash crash events and market microstructure breakdowns.

    Uses:
    - Price velocity thresholding
    - Volume-price divergence
    - Bid-ask spread explosion
    - Order book imbalance extremes
    """

    def __init__(
        self,
        price_velocity_threshold: float = 0.05,  # 5% per step
        spread_explosion_factor: float = 5.0,      # 5x normal spread
        imbalance_threshold: float = 0.9,
        detection_window: int = 10,
        cooldown: int = 20,
    ):
        self.price_velocity_threshold = price_velocity_threshold
        self.spread_explosion_factor = spread_explosion_factor
        self.imbalance_threshold = imbalance_threshold
        self.detection_window = detection_window
        self.cooldown = cooldown

        self._price_history: collections.deque = collections.deque(maxlen=detection_window * 2)
        self._spread_history: collections.deque = collections.deque(maxlen=detection_window * 2)
        self._imbalance_history: collections.deque = collections.deque(maxlen=detection_window)
        self._normal_spread: float = 0.002
        self._last_crash_step: int = -1000
        self._crash_events: List[Dict] = []

    def update(
        self,
        price: float,
        spread: float,
        imbalance: float,
        volume: float,
        step: int,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Update detector with new market data.
        Returns (is_crash, crash_event) tuple.
        """
        self._price_history.append(float(price))
        spread_val = float(spread) if not math.isnan(spread) else 0.0
        self._spread_history.append(spread_val)
        self._imbalance_history.append(float(imbalance))

        # Update normal spread estimate
        if len(self._spread_history) > 20:
            recent_spreads = np.array(list(self._spread_history))
            self._normal_spread = float(np.percentile(recent_spreads, 25))

        # Check cooldown
        if step - self._last_crash_step < self.cooldown:
            return False, None

        # Detection conditions
        is_crash = False
        crash_type = None

        # 1. Price velocity check
        if len(self._price_history) >= self.detection_window:
            prices = np.array(list(self._price_history)[-self.detection_window:])
            max_ret = float(np.max(np.abs(np.diff(prices) / (prices[:-1] + EPS))))
            if max_ret > self.price_velocity_threshold:
                is_crash = True
                crash_type = "price_velocity"

        # 2. Spread explosion
        if (
            self._normal_spread > EPS
            and spread_val > self._normal_spread * self.spread_explosion_factor
        ):
            is_crash = True
            crash_type = crash_type or "spread_explosion"

        # 3. Order imbalance extreme
        if abs(imbalance) > self.imbalance_threshold:
            is_crash = True
            crash_type = crash_type or "imbalance_extreme"

        if is_crash:
            self._last_crash_step = step
            event = {
                "step": step,
                "type": crash_type,
                "price": price,
                "spread": spread_val,
                "imbalance": imbalance,
                "volume": volume,
            }
            self._crash_events.append(event)
            logger.warning(f"Flash crash detected at step {step}: {crash_type}")
            return True, event

        return False, None

    def get_crash_history(self) -> List[Dict]:
        return self._crash_events.copy()

    def crash_frequency(self, window: int = 1000) -> float:
        """Crashes per step in recent window."""
        if not self._crash_events:
            return 0.0
        recent = [e for e in self._crash_events if e["step"] > (self._last_crash_step - window)]
        return len(recent) / max(window, 1)


# ---------------------------------------------------------------------------
# Nash equilibrium approximation
# ---------------------------------------------------------------------------

class NashEquilibriumApproximator:
    """
    Approximates Nash equilibrium quality in the multi-agent system.

    Metrics:
    - Exploitability (distance from Nash)
    - Individual regret bounds
    - Correlated equilibrium measure
    - Strategy convergence
    """

    def __init__(self, num_agents: int, action_dim: int, lookback: int = 500):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.lookback = lookback

        self._agent_rewards: Dict[int, collections.deque] = {
            i: collections.deque(maxlen=lookback)
            for i in range(num_agents)
        }
        self._agent_actions: Dict[int, collections.deque] = {
            i: collections.deque(maxlen=lookback)
            for i in range(num_agents)
        }
        self._best_response_rewards: Dict[int, collections.deque] = {
            i: collections.deque(maxlen=lookback)
            for i in range(num_agents)
        }

    def update(
        self,
        rewards: List[float],
        actions: List[np.ndarray],
        best_response_rewards: Optional[List[float]] = None,
    ) -> None:
        for i in range(self.num_agents):
            self._agent_rewards[i].append(float(rewards[i]))
            self._agent_actions[i].append(actions[i].copy())
            if best_response_rewards is not None and i < len(best_response_rewards):
                self._best_response_rewards[i].append(float(best_response_rewards[i]))

    def individual_regret(self, agent_id: int) -> float:
        """
        Compute individual regret for agent_id.
        Regret = max_a E[r | a] - E[r | pi]
        """
        actual_rewards = list(self._agent_rewards[agent_id])
        br_rewards = list(self._best_response_rewards[agent_id])

        if not actual_rewards or not br_rewards:
            return 0.0

        n = min(len(actual_rewards), len(br_rewards))
        actual_mean = float(np.mean(actual_rewards[-n:]))
        br_mean = float(np.mean(br_rewards[-n:]))
        return max(0.0, br_mean - actual_mean)

    def average_regret(self) -> float:
        """Average regret across all agents."""
        if not any(self._best_response_rewards[i] for i in range(self.num_agents)):
            return float("nan")
        regrets = [self.individual_regret(i) for i in range(self.num_agents)]
        return float(np.mean([r for r in regrets if not math.isnan(r)]))

    def action_entropy(self, agent_id: int) -> float:
        """Entropy of agent's action distribution (higher = more mixed strategy)."""
        actions = list(self._agent_actions[agent_id])
        if not actions:
            return 0.0
        action_arr = np.array(actions)
        # Marginal entropy for each action dimension
        entropies = []
        for d in range(self.action_dim):
            hist, _ = np.histogram(action_arr[:, d], bins=20, range=(-1, 1))
            hist = hist / (hist.sum() + EPS)
            ent = -np.sum(hist[hist > 0] * np.log(hist[hist > 0] + EPS))
            entropies.append(float(ent))
        return float(np.mean(entropies))

    def strategy_convergence(self, agent_id: int, window: int = 50) -> float:
        """
        Measure how much agent's strategy has converged.
        Returns variance of recent actions (lower = more converged).
        """
        actions = list(self._agent_actions[agent_id])
        if len(actions) < window:
            return float("nan")
        recent = np.array(actions[-window:])
        return float(np.mean(np.var(recent, axis=0)))

    def equilibrium_gap(self) -> float:
        """
        Overall measure of distance from Nash equilibrium.
        Uses average regret as proxy.
        """
        regret = self.average_regret()
        if math.isnan(regret):
            return float("nan")
        return regret

    def summary(self) -> Dict[str, Any]:
        return {
            "average_regret": self.average_regret(),
            "per_agent_regret": [self.individual_regret(i) for i in range(self.num_agents)],
            "action_entropy": [self.action_entropy(i) for i in range(self.num_agents)],
            "strategy_convergence": [
                self.strategy_convergence(i) for i in range(self.num_agents)
            ],
            "equilibrium_gap": self.equilibrium_gap(),
        }


# ---------------------------------------------------------------------------
# Emergence analyzer (aggregate)
# ---------------------------------------------------------------------------

class EmergenceAnalyzer:
    """
    Top-level emergence analysis module.
    Aggregates all microstructure metrics and detects emergent phenomena.
    """

    def __init__(
        self,
        num_assets: int,
        num_agents: int,
        action_dim: int,
    ):
        self.num_assets = num_assets
        self.num_agents = num_agents

        self.impact_analyzers = {
            i: MarketImpactAnalyzer() for i in range(num_assets)
        }
        self.price_discovery = {
            i: PriceDiscoveryAnalyzer() for i in range(num_assets)
        }
        self.liquidity_trackers = {
            i: LiquidityEmergenceTracker() for i in range(num_assets)
        }
        self.flash_detectors = {
            i: FlashCrashDetector() for i in range(num_assets)
        }
        self.nash_approx = NashEquilibriumApproximator(num_agents, action_dim)

        self._step = 0
        self._emergence_events: List[Dict] = []

    def update(
        self,
        prices: np.ndarray,
        spreads: np.ndarray,
        imbalances: np.ndarray,
        volumes: np.ndarray,
        trade_counts: np.ndarray,
        total_depths: np.ndarray,
        agent_rewards: List[float],
        agent_actions: List[np.ndarray],
        fundamentals: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Update all analyzers with current market state."""
        self._step += 1
        events = []

        for i in range(self.num_assets):
            # Price discovery
            self.price_discovery[i].update(
                float(prices[i]),
                float(fundamentals[i]) if fundamentals is not None else None,
            )

            # Liquidity
            self.liquidity_trackers[i].update(
                float(spreads[i]) if not math.isnan(spreads[i]) else 0.0,
                float(total_depths[i]),
                float(volumes[i]),
                int(trade_counts[i]),
            )

            # Flash crash
            is_crash, crash_event = self.flash_detectors[i].check(
                float(prices[i]),
                float(spreads[i]) if not math.isnan(spreads[i]) else 0.0,
                float(imbalances[i]),
                float(volumes[i]),
                self._step,
            ) if hasattr(self.flash_detectors[i], "check") else (False, None)

            if is_crash and crash_event:
                events.append({"asset": i, **crash_event})
                self._emergence_events.append({"asset": i, **crash_event})

        # Nash equilibrium
        self.nash_approx.update(agent_rewards, agent_actions)

        return {
            "step": self._step,
            "events": events,
            "num_events": len(events),
        }

    def full_report(self) -> Dict[str, Any]:
        """Generate comprehensive emergence report."""
        report: Dict[str, Any] = {}

        for i in range(self.num_assets):
            asset_report = {
                "market_impact": self.impact_analyzers[i].summary(),
                "price_discovery": {
                    "efficiency": self.price_discovery[i].price_efficiency_score(),
                    "variance_ratio": self.price_discovery[i].variance_ratio(),
                    "fundamental_gap": self.price_discovery[i].price_vs_fundamental_gap(),
                },
                "liquidity": self.liquidity_trackers[i].summary(),
                "flash_crashes": len(self.flash_detectors[i].get_crash_history()),
            }
            report[f"asset_{i}"] = asset_report

        report["nash"] = self.nash_approx.summary()
        report["total_emergence_events"] = len(self._emergence_events)
        report["step"] = self._step

        return report

    def get_market_health_score(self) -> float:
        """
        Composite market health score in [0, 1].
        1 = perfectly healthy market.
        """
        scores = []

        for i in range(self.num_assets):
            # Efficiency
            eff = self.price_discovery[i].price_efficiency_score()
            scores.append(eff)

            # Liquidity stability
            liq = self.liquidity_trackers[i].spread_stability()
            scores.append(liq)

            # No crashes
            crash_freq = self.flash_detectors[i].crash_frequency(window=max(self._step, 1))
            crash_score = max(0.0, 1.0 - crash_freq * 100)
            scores.append(crash_score)

        # Nash convergence
        regret = self.nash_approx.average_regret()
        if not math.isnan(regret):
            nash_score = max(0.0, 1.0 - regret)
            scores.append(nash_score)

        return float(np.mean(scores)) if scores else 0.5


# Add check method to FlashCrashDetector for cleaner API
def _detector_check(self, price, spread, imbalance, volume, step):
    return self.update(price, spread, imbalance, volume, step)

FlashCrashDetector.check = _detector_check


__all__ = [
    "MarketImpactAnalyzer",
    "PriceDiscoveryAnalyzer",
    "LiquidityEmergenceTracker",
    "FlashCrashDetector",
    "NashEquilibriumApproximator",
    "EmergenceAnalyzer",
]
