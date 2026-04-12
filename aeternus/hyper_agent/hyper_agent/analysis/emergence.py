"""
EmergenceAnalyzer — Detects emergent macro properties from micro agent behaviors.

Metrics:
  - PriceDiscovery:      how quickly agents converge to true asset value
  - LiquidityEmergence:  bid-ask spread as function of MM competition
  - InformationCascade:  how information propagates through agent network
  - HerdingIndex:        Lakonishok-style correlated agent behavior
  - FlashCrashDetector:  self-reinforcing crash detection
  - NashCheck:           is current policy profile approximately Nash?
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks


# ============================================================
# Price Discovery Metric
# ============================================================

class PriceDiscovery:
    """
    Measures how quickly market prices incorporate new information.

    Method:
      1. Track deviation of observed mid from estimated "true" fundamental
      2. Fit exponential decay to deviation: |p(t) - f| ~ e^{-λt}
      3. Speed of discovery = decay rate λ

    True fundamental is approximated as EMA of prices with very slow decay.
    """

    def __init__(self, fundamental_ema: float = 0.01) -> None:
        self.fundamental_ema = fundamental_ema
        self._fundamental_est: Optional[float] = None
        self._deviations: List[float] = []
        self._prices:     List[float] = []

    def update(self, mid_price: float) -> None:
        if self._fundamental_est is None:
            self._fundamental_est = mid_price
        else:
            self._fundamental_est = (
                (1 - self.fundamental_ema) * self._fundamental_est
                + self.fundamental_ema * mid_price
            )
        deviation = abs(mid_price - self._fundamental_est)
        self._deviations.append(deviation)
        self._prices.append(mid_price)

    def discovery_rate(self) -> float:
        """
        Estimate price discovery speed as exponential decay rate.
        Returns λ; higher = faster discovery.
        """
        n = len(self._deviations)
        if n < 10:
            return 0.0
        dev   = np.array(self._deviations[-50:])
        times = np.arange(len(dev), dtype=np.float64)

        # Fit log(dev) = log(dev_0) - λ*t
        dev_safe = dev + 1e-8
        log_dev  = np.log(dev_safe)
        mask     = np.isfinite(log_dev)
        if mask.sum() < 5:
            return 0.0
        try:
            slope, _, r, _, _ = stats.linregress(times[mask], log_dev[mask])
            return float(max(-slope, 0.0))
        except Exception:
            return 0.0

    def forecast_error(self) -> float:
        """Mean absolute deviation from fundamental estimate."""
        if not self._deviations:
            return 0.0
        return float(np.mean(self._deviations[-100:]))

    def reset(self) -> None:
        self._fundamental_est = None
        self._deviations.clear()
        self._prices.clear()


# ============================================================
# Liquidity Emergence
# ============================================================

class LiquidityEmergence:
    """
    Tracks how bid-ask spread emerges from competition between MM agents.

    Expectation: spread decreases with more MMs (Cournot competition).
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._spread_history: deque = deque(maxlen=window)
        self._n_mm_history:   deque = deque(maxlen=window)

    def update(self, spread: float, n_active_mms: int) -> None:
        self._spread_history.append(spread)
        self._n_mm_history.append(n_active_mms)

    def spread_vs_mms(self) -> Dict[int, float]:
        """
        Return dict: n_mms → mean_spread for each observed MM count.
        """
        result: Dict[int, List[float]] = {}
        for sp, n in zip(self._spread_history, self._n_mm_history):
            result.setdefault(n, []).append(sp)
        return {n: float(np.mean(v)) for n, v in result.items()}

    def competition_elasticity(self) -> float:
        """
        d(log spread) / d(log n_mms).
        Negative = more MMs reduce spread (as theory predicts).
        """
        n_sp = self.spread_vs_mms()
        if len(n_sp) < 2:
            return 0.0
        ns  = np.array(sorted(n_sp.keys()), dtype=np.float64)
        sps = np.array([n_sp[n] for n in ns], dtype=np.float64)
        if (ns <= 0).any() or (sps <= 0).any():
            return 0.0
        try:
            slope, _, _, _, _ = stats.linregress(np.log(ns), np.log(sps))
            return float(slope)
        except Exception:
            return 0.0

    def current_spread(self) -> float:
        if not self._spread_history:
            return 0.0
        return float(np.mean(self._spread_history))


# ============================================================
# Information Cascade
# ============================================================

class InformationCascade:
    """
    Tracks how trading signals propagate through the agent network.

    Measures the network effect: does one agent's signal cause
    correlated behavior in connected neighbors?
    """

    def __init__(self, agent_ids: List[str]) -> None:
        self.agent_ids = agent_ids
        # action_history: agent → recent direction sequence
        self._action_hist: Dict[str, deque] = {
            a: deque(maxlen=50) for a in agent_ids
        }

    def update(self, agent_actions: Dict[str, int]) -> None:
        for aid, action in agent_actions.items():
            if aid in self._action_hist:
                self._action_hist[aid].append(action)

    def cascade_probability(
        self,
        source_id: str,
        neighbor_ids: List[str],
        lag: int = 1,
    ) -> float:
        """
        P(neighbor copies source direction at lag t).

        Higher = stronger cascade effect.
        """
        src_hist = list(self._action_hist.get(source_id, []))
        if len(src_hist) < lag + 5:
            return 0.0

        cascade_scores = []
        for nb_id in neighbor_ids:
            nb_hist = list(self._action_hist.get(nb_id, []))
            if len(nb_hist) < lag + 5:
                continue
            # Align and count matches
            src_t = src_hist[:-lag] if lag > 0 else src_hist
            nb_t  = nb_hist[lag:]   if lag > 0 else nb_hist
            min_n = min(len(src_t), len(nb_t))
            if min_n < 3:
                continue
            matches = sum(
                1 for s, n in zip(src_t[-min_n:], nb_t[-min_n:]) if s == n
            )
            cascade_scores.append(matches / min_n)

        return float(np.mean(cascade_scores)) if cascade_scores else 0.0

    def network_synchrony(self) -> float:
        """
        Mean pairwise action correlation across all agent pairs.
        High synchrony → herding or coordination.
        """
        agents_with_hist = [
            a for a in self.agent_ids
            if len(self._action_hist[a]) >= 10
        ]
        if len(agents_with_hist) < 2:
            return 0.0

        correlations = []
        for i in range(len(agents_with_hist)):
            for j in range(i + 1, min(i + 5, len(agents_with_hist))):
                a1 = np.array(list(self._action_hist[agents_with_hist[i]]))
                a2 = np.array(list(self._action_hist[agents_with_hist[j]]))
                min_n = min(len(a1), len(a2))
                if min_n < 5:
                    continue
                try:
                    r, _ = stats.pearsonr(a1[-min_n:], a2[-min_n:])
                    if np.isfinite(r):
                        correlations.append(abs(r))
                except Exception:
                    continue

        return float(np.mean(correlations)) if correlations else 0.0


# ============================================================
# Herding Index
# ============================================================

class HerdingIndex:
    """
    Lakonishok, Shleifer, and Vishny (1992) herding measure.

    H = |fraction_buying - p| - E[|fraction_buying - p|]

    where p = E[fraction_buying] over all agents.

    High H → agents are correlated in their trading direction.
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._frac_buying: deque = deque(maxlen=window)
        self._herding_hist: deque = deque(maxlen=window)

    def update(self, agent_actions: Dict[str, int]) -> float:
        """
        Compute and record herding index for this timestep.

        Returns current herding index.
        """
        actions = list(agent_actions.values())
        n = len(actions)
        if n < 2:
            return 0.0

        n_buy  = sum(1 for a in actions if a > 0)
        f_buy  = n_buy / n
        self._frac_buying.append(f_buy)

        if len(self._frac_buying) < 5:
            return 0.0

        p   = float(np.mean(self._frac_buying))
        # Expected deviation under null (no herding): binomial variance
        exp_dev = math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        h       = abs(f_buy - p) - exp_dev
        self._herding_hist.append(h)
        return float(max(h, 0.0))

    def mean_herding(self) -> float:
        if not self._herding_hist:
            return 0.0
        return float(np.mean([max(h, 0) for h in self._herding_hist]))

    def is_herding(self, threshold: float = 0.1) -> bool:
        return self.mean_herding() > threshold


# ============================================================
# Flash Crash Detector
# ============================================================

class FlashCrashDetector:
    """
    Detects flash crashes: sudden rapid price drops followed by partial recovery.

    Criteria:
      1. Price drops > N standard deviations in < K bars
      2. Drop is at least threshold% of reference price
      3. Followed by partial recovery within window
    """

    def __init__(
        self,
        drop_threshold_std:   float = 4.0,   # standard deviations
        drop_pct:             float = 0.02,  # min % drop
        crash_window:         int   = 5,     # bars for crash
        recovery_window:      int   = 20,    # bars for recovery check
        recovery_fraction:    float = 0.3,   # partial recovery needed
        vol_window:           int   = 100,
    ) -> None:
        self.drop_threshold_std = drop_threshold_std
        self.drop_pct           = drop_pct
        self.crash_window       = crash_window
        self.recovery_window    = recovery_window
        self.recovery_fraction  = recovery_fraction

        self._prices: deque = deque(maxlen=vol_window + recovery_window + crash_window + 10)
        self._crashes: List[Dict] = []
        self._vol_window = vol_window

    def update(self, price: float) -> bool:
        """
        Update with new price. Returns True if a flash crash was detected.
        """
        self._prices.append(price)

        if len(self._prices) < self._vol_window + self.crash_window + 5:
            return False

        prices = np.array(list(self._prices))
        n      = len(prices)

        # Estimate rolling volatility
        returns = np.diff(np.log(prices[-self._vol_window:] + 1e-8))
        vol     = float(np.std(returns)) + 1e-8

        # Check for rapid drop in last crash_window bars
        recent = prices[-self.crash_window - 1:]
        ref    = recent[0]
        drop   = (ref - recent.min()) / ref
        drop_std = drop / vol

        if drop > self.drop_pct and drop_std > self.drop_threshold_std:
            # Verify it's not already in crashes list (avoid duplicates)
            crash_bar = n
            if not self._crashes or self._crashes[-1]["bar"] < crash_bar - self.recovery_window:
                self._crashes.append({
                    "bar":           crash_bar,
                    "ref_price":     ref,
                    "trough_price":  recent.min(),
                    "drop_pct":      drop,
                    "drop_std":      drop_std,
                    "recovered":     False,
                })
                return True

        # Check recovery for most recent crash
        if self._crashes and not self._crashes[-1]["recovered"]:
            last = self._crashes[-1]
            if n - last["bar"] <= self.recovery_window:
                recovery = (prices[-1] - last["trough_price"]) / (
                    last["ref_price"] - last["trough_price"] + 1e-8
                )
                if recovery >= self.recovery_fraction:
                    self._crashes[-1]["recovered"] = True
                    self._crashes[-1]["recovery_pct"] = float(recovery)

        return False

    def n_crashes(self) -> int:
        return len(self._crashes)

    def crash_log(self) -> List[Dict]:
        return list(self._crashes)

    def is_in_crash(self) -> bool:
        """True if a crash started recently and no recovery yet."""
        if not self._crashes:
            return False
        last = self._crashes[-1]
        n    = len(self._prices)
        return (not last["recovered"] and
                n - last["bar"] <= self.recovery_window)


# ============================================================
# Nash Equilibrium Checker
# ============================================================

class NashChecker:
    """
    Approximate Nash equilibrium check for current policy profile.

    For each agent, estimate whether they could benefit from a unilateral
    deviation by checking if their current policy is approximately
    a best response to the others' policies.

    Approximation: compute finite-difference gradient of expected reward
    w.r.t. agent's policy parameters. Near Nash → gradient ≈ 0.
    """

    def __init__(
        self,
        n_agents:     int,
        window:       int   = 50,
        nash_eps:     float = 0.01,
    ) -> None:
        self.n_agents = n_agents
        self.window   = window
        self.nash_eps = nash_eps
        self._reward_hist: Dict[str, deque] = {}

    def update(self, rewards: Dict[str, float]) -> None:
        for aid, r in rewards.items():
            if aid not in self._reward_hist:
                self._reward_hist[aid] = deque(maxlen=self.window)
            self._reward_hist[aid].append(r)

    def is_approximate_nash(self) -> bool:
        """
        Check if rewards are stable (no systematic improving deviations).

        Heuristic: all agents' reward trends ≈ 0 → no one is actively improving.
        """
        trends = []
        for hist in self._reward_hist.values():
            if len(hist) < 10:
                return False
            y     = np.array(list(hist))
            x     = np.arange(len(y))
            if y.std() < 1e-8:
                trends.append(0.0)
                continue
            try:
                slope, _, _, _, _ = stats.linregress(x, y)
                trends.append(abs(slope))
            except Exception:
                pass

        if not trends:
            return False
        return float(np.max(trends)) < self.nash_eps

    def nash_residual(self) -> float:
        """
        Max absolute reward trend across all agents.
        Near 0 → approximate Nash equilibrium.
        """
        trends = []
        for hist in self._reward_hist.values():
            if len(hist) < 5:
                continue
            y = np.array(list(hist))
            x = np.arange(len(y))
            try:
                slope, _, _, _, _ = stats.linregress(x, y)
                trends.append(abs(slope))
            except Exception:
                pass
        return float(np.max(trends)) if trends else 0.0


# ============================================================
# EmergenceAnalyzer (composite)
# ============================================================

class EmergenceAnalyzer:
    """
    Composite emergence analyzer.

    Aggregates all emergence metrics and provides a unified interface
    for the training loop and experiment scripts.
    """

    def __init__(
        self,
        agent_ids: List[str],
        neighbor_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.agent_ids    = agent_ids
        self.neighbor_map = neighbor_map or {}

        self.price_discovery   = PriceDiscovery()
        self.liquidity         = LiquidityEmergence()
        self.cascade           = InformationCascade(agent_ids)
        self.herding           = HerdingIndex()
        self.flash_crash       = FlashCrashDetector()
        self.nash_checker      = NashChecker(len(agent_ids))

        self._step = 0
        self._history: List[Dict] = []

    def step(
        self,
        mid_price:       float,
        spread:          float,
        agent_actions:   Dict[str, int],
        agent_rewards:   Dict[str, float],
        n_active_mms:    int = 1,
    ) -> Dict[str, Any]:
        """
        Update all analyzers with one timestep of data.

        Returns dict of current emergence metrics.
        """
        self._step += 1

        self.price_discovery.update(mid_price)
        self.liquidity.update(spread, n_active_mms)
        self.cascade.update(agent_actions)
        h     = self.herding.update(agent_actions)
        crash = self.flash_crash.update(mid_price)
        self.nash_checker.update(agent_rewards)

        metrics = {
            "step":               self._step,
            "discovery_rate":     self.price_discovery.discovery_rate(),
            "forecast_error":     self.price_discovery.forecast_error(),
            "current_spread":     self.liquidity.current_spread(),
            "competition_elasticity": self.liquidity.competition_elasticity(),
            "network_synchrony":  self.cascade.network_synchrony(),
            "herding_index":      h,
            "mean_herding":       self.herding.mean_herding(),
            "is_herding":         self.herding.is_herding(),
            "flash_crash":        crash,
            "n_crashes":          self.flash_crash.n_crashes(),
            "in_crash":           self.flash_crash.is_in_crash(),
            "approx_nash":        self.nash_checker.is_approximate_nash(),
            "nash_residual":      self.nash_checker.nash_residual(),
        }

        if len(self._history) % 50 == 0:  # store every 50 steps to save memory
            self._history.append(metrics)

        return metrics

    def full_report(self) -> Dict[str, Any]:
        """Return full analysis summary."""
        return {
            "total_steps":         self._step,
            "discovery_rate":      self.price_discovery.discovery_rate(),
            "spread_vs_mms":       self.liquidity.spread_vs_mms(),
            "competition_elasticity": self.liquidity.competition_elasticity(),
            "mean_herding":        self.herding.mean_herding(),
            "network_synchrony":   self.cascade.network_synchrony(),
            "n_flash_crashes":     self.flash_crash.n_crashes(),
            "crash_log":           self.flash_crash.crash_log(),
            "approx_nash":         self.nash_checker.is_approximate_nash(),
            "nash_residual":       self.nash_checker.nash_residual(),
        }

    def reset(self) -> None:
        self.price_discovery.reset()
        self.herding     = HerdingIndex()
        self.flash_crash = FlashCrashDetector()
        self._step       = 0
        self._history.clear()
