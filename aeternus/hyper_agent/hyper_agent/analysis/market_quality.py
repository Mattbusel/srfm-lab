"""
MarketQualityMetrics — Measures of market health produced by agent population.

Metrics:
  - EffectiveSpread:      volume-weighted average effective spread
  - PriceImpact:         Kyle lambda from agent trades
  - MarketDepth:         quoted volume within N ticks of mid
  - InformationalEfficiency: how quickly prices incorporate information
  - Resilience:          recovery speed after order flow shock
  - Comparative baseline: MARL vs single-agent vs no-agents
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats


# ============================================================
# Effective Spread
# ============================================================

class EffectiveSpread:
    """
    Volume-weighted average effective spread (VWAS).

    Effective spread = 2 * |exec_price - mid_price_before_trade|

    Captures actual transaction costs; more informative than quoted spread
    because it accounts for order flow timing effects.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._spreads: deque = deque(maxlen=window)
        self._volumes: deque = deque(maxlen=window)

    def update(
        self, exec_price: float, mid_price: float, volume: float
    ) -> float:
        """Update with a trade. Returns current effective half-spread."""
        eff_half_spread = abs(exec_price - mid_price)
        self._spreads.append(eff_half_spread)
        self._volumes.append(max(volume, 1e-8))
        return eff_half_spread

    def vwas(self) -> float:
        """Volume-weighted average effective spread."""
        if not self._spreads:
            return 0.0
        spreads = np.array(list(self._spreads))
        volumes = np.array(list(self._volumes))
        return float(np.average(spreads, weights=volumes) * 2.0)

    def simple_mean(self) -> float:
        if not self._spreads:
            return 0.0
        return float(np.mean(self._spreads) * 2.0)


# ============================================================
# Price Impact (Kyle Lambda)
# ============================================================

class PriceImpactEstimator:
    """
    Estimates Kyle's lambda — the price impact coefficient.

    Model: Δp = λ * order_flow + noise
    λ = Cov(Δp, Q) / Var(Q)

    where Q = net signed order flow.

    High λ → price is sensitive to order flow (illiquid market).
    Low λ  → market absorbs order flow without large price moves.
    """

    def __init__(self, window: int = 200) -> None:
        self.window   = window
        self._price_changes: deque = deque(maxlen=window)
        self._order_flows:   deque = deque(maxlen=window)

    def update(self, price_change: float, net_order_flow: float) -> None:
        self._price_changes.append(price_change)
        self._order_flows.append(net_order_flow)

    def kyle_lambda(self) -> float:
        """
        OLS estimate of λ = Cov(Δp, Q) / Var(Q).
        """
        if len(self._price_changes) < 20:
            return 0.0

        dp = np.array(list(self._price_changes))
        q  = np.array(list(self._order_flows))

        var_q = np.var(q)
        if var_q < 1e-12:
            return 0.0

        return float(np.cov(dp, q)[0, 1] / var_q)

    def r_squared(self) -> float:
        """R² of price impact regression."""
        if len(self._price_changes) < 20:
            return 0.0
        dp = np.array(list(self._price_changes))
        q  = np.array(list(self._order_flows))
        try:
            _, _, r, _, _ = stats.linregress(q, dp)
            return float(r ** 2)
        except Exception:
            return 0.0


# ============================================================
# Market Depth
# ============================================================

class MarketDepth:
    """
    Tracks total quoted volume within N ticks of mid price.

    Shallow market → small quoted volume → large price impact per unit.
    """

    def __init__(self, n_ticks: int = 5, tick_size: float = 0.01) -> None:
        self.n_ticks   = n_ticks
        self.tick_size = tick_size
        self._depth_hist: deque = deque(maxlen=100)

    def update(
        self,
        mid_price:  float,
        bid_volume: float,
        ask_volume: float,
    ) -> float:
        """Total depth = bid volume + ask volume within n_ticks."""
        depth = bid_volume + ask_volume
        self._depth_hist.append(depth)
        return depth

    def mean_depth(self) -> float:
        if not self._depth_hist:
            return 0.0
        return float(np.mean(self._depth_hist))

    def depth_imbalance(self) -> Optional[float]:
        """
        Recent trend in depth.
        Positive → depth increasing (improving liquidity).
        """
        if len(self._depth_hist) < 10:
            return None
        d = np.array(list(self._depth_hist))
        x = np.arange(len(d), dtype=np.float64)
        try:
            slope, _, _, _, _ = stats.linregress(x, d)
            return float(slope)
        except Exception:
            return None


# ============================================================
# Informational Efficiency
# ============================================================

class InformationalEfficiency:
    """
    Measures how quickly prices incorporate new information.

    Methods:
      1. Autocorrelation test: efficient prices → near-zero serial correlation
      2. Variance ratio test: Var(k-period returns) / k * Var(1-period returns) ≈ 1
      3. Return predictability: OLS R² of return on lagged returns
    """

    def __init__(self, window: int = 200) -> None:
        self.window  = window
        self._prices: deque = deque(maxlen=window)

    def update(self, price: float) -> None:
        self._prices.append(price)

    def autocorrelation(self, lag: int = 1) -> float:
        """
        First-order autocorrelation of log-returns.
        Near 0 → informationally efficient.
        """
        if len(self._prices) < lag + 10:
            return 0.0
        prices  = np.array(list(self._prices))
        returns = np.diff(np.log(prices + 1e-8))
        if len(returns) < lag + 5:
            return 0.0
        try:
            r, _ = stats.pearsonr(returns[:-lag], returns[lag:])
            return float(r) if np.isfinite(r) else 0.0
        except Exception:
            return 0.0

    def variance_ratio(self, k: int = 5) -> float:
        """
        Variance ratio VR(k) = Var(k-period return) / (k * Var(1-period return)).
        VR = 1 → random walk (efficient).
        VR > 1 → positive autocorrelation (momentum).
        VR < 1 → mean reversion.
        """
        if len(self._prices) < k * 3:
            return 1.0
        prices = np.array(list(self._prices))
        ret1   = np.diff(np.log(prices + 1e-8))
        var1   = np.var(ret1)
        if var1 < 1e-12:
            return 1.0

        # k-period returns
        retk   = np.log(prices[k:] + 1e-8) - np.log(prices[:-k] + 1e-8)
        vark   = np.var(retk)
        return float(vark / (k * var1 + 1e-12))

    def efficiency_score(self) -> float:
        """
        Combined efficiency score in [0, 1].
        1 = perfectly efficient; 0 = highly inefficient.
        """
        ac1  = abs(self.autocorrelation(1))
        vr   = abs(self.variance_ratio(5) - 1.0)
        return float(max(0.0, 1.0 - ac1 - 0.5 * vr))


# ============================================================
# Resilience
# ============================================================

class ResilienceMeasure:
    """
    Speed of spread / price recovery after an order flow shock.

    Measures time for spread to return to pre-shock level
    and price to revert toward pre-shock mid.
    """

    def __init__(
        self,
        shock_threshold: float = 3.0,  # z-score for shock detection
        window:          int   = 100,
        half_life_cap:   int   = 50,
    ) -> None:
        self.shock_threshold = shock_threshold
        self.window          = window
        self.half_life_cap   = half_life_cap

        self._spread_hist: deque = deque(maxlen=window)
        self._price_hist:  deque = deque(maxlen=window)
        self._shock_log:   List[Dict] = []
        self._in_shock:    bool  = False
        self._shock_start: int   = 0
        self._shock_spread: float = 0.0
        self._baseline_spread: float = 0.0
        self._t:           int   = 0

    def update(self, spread: float, mid_price: float) -> bool:
        """Returns True if a shock was detected."""
        self._t += 1
        self._spread_hist.append(spread)
        self._price_hist.append(mid_price)

        if len(self._spread_hist) < 20:
            return False

        spreads = np.array(list(self._spread_hist))
        mu      = spreads[:-5].mean()
        sigma   = spreads[:-5].std() + 1e-8
        z       = (spread - mu) / sigma

        if not self._in_shock and z > self.shock_threshold:
            self._in_shock        = True
            self._shock_start     = self._t
            self._shock_spread    = spread
            self._baseline_spread = mu
            return True

        if self._in_shock:
            # Check recovery: spread returned to within 1 sigma of baseline
            if spread < self._baseline_spread + sigma:
                recovery_time = self._t - self._shock_start
                self._shock_log.append({
                    "start":         self._shock_start,
                    "peak_spread":   self._shock_spread,
                    "baseline":      self._baseline_spread,
                    "recovery_bars": recovery_time,
                })
                self._in_shock = False

        return False

    def mean_recovery_bars(self) -> float:
        if not self._shock_log:
            return float(self.half_life_cap)
        return float(np.mean([r["recovery_bars"] for r in self._shock_log]))

    def n_shocks(self) -> int:
        return len(self._shock_log)

    def resilience_score(self) -> float:
        """
        Resilience score in [0, 1].
        High → fast recovery (low mean_recovery_bars).
        """
        mean_rec = self.mean_recovery_bars()
        return float(max(0.0, 1.0 - mean_rec / self.half_life_cap))


# ============================================================
# Comparative Baseline
# ============================================================

class BaselineComparison:
    """
    Compares MARL market quality against baselines.

    Baselines:
      - No agents (Brownian motion only)
      - Single random agent
      - Single market maker
    """

    def __init__(self) -> None:
        self._marl_metrics: List[Dict[str, float]] = []
        self._baseline_metrics: Dict[str, List[Dict[str, float]]] = {
            "no_agents":    [],
            "single_agent": [],
            "single_mm":    [],
        }

    def record_marl(self, metrics: Dict[str, float]) -> None:
        self._marl_metrics.append(metrics)

    def record_baseline(self, baseline: str, metrics: Dict[str, float]) -> None:
        if baseline in self._baseline_metrics:
            self._baseline_metrics[baseline].append(metrics)

    def compare(self, metric_key: str) -> Dict[str, float]:
        """
        Compare MARL vs baselines on a specific metric.
        Returns dict: {condition → mean metric value}.
        """
        result = {}
        if self._marl_metrics:
            vals = [m.get(metric_key, 0.0) for m in self._marl_metrics]
            result["marl"] = float(np.mean(vals))

        for name, records in self._baseline_metrics.items():
            if records:
                vals = [m.get(metric_key, 0.0) for m in records]
                result[name] = float(np.mean(vals))

        return result

    def improvement_over_baseline(
        self, metric_key: str, baseline: str = "no_agents", higher_is_better: bool = True
    ) -> float:
        """
        Percentage improvement of MARL over baseline.
        Positive → MARL is better.
        """
        comparison = self.compare(metric_key)
        marl_val   = comparison.get("marl", 0.0)
        base_val   = comparison.get(baseline, 0.0)
        if abs(base_val) < 1e-8:
            return 0.0
        diff = marl_val - base_val
        return float(diff / abs(base_val) * 100.0 * (1 if higher_is_better else -1))


# ============================================================
# MarketQualityMetrics (composite)
# ============================================================

class MarketQualityMetrics:
    """
    Composite market quality measurement system.
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        self.eff_spread   = EffectiveSpread()
        self.price_impact = PriceImpactEstimator()
        self.depth        = MarketDepth(tick_size=tick_size)
        self.efficiency   = InformationalEfficiency()
        self.resilience   = ResilienceMeasure()
        self.baseline     = BaselineComparison()
        self._step        = 0

    def step(
        self,
        mid_price:      float,
        exec_price:     float,
        spread:         float,
        volume:         float,
        net_order_flow: float,
        bid_volume:     float = 0.0,
        ask_volume:     float = 0.0,
        prev_price:     Optional[float] = None,
    ) -> Dict[str, float]:
        """Update all metrics. Returns dict of current quality measures."""
        self._step += 1

        self.eff_spread.update(exec_price, mid_price, volume)
        self.efficiency.update(mid_price)
        self.resilience.update(spread, mid_price)
        self.depth.update(mid_price, bid_volume, ask_volume)

        if prev_price is not None and prev_price > 0:
            price_change = math.log(mid_price / prev_price)
            self.price_impact.update(price_change, net_order_flow)

        metrics = {
            "step":                self._step,
            "vwas":                self.eff_spread.vwas(),
            "kyle_lambda":         self.price_impact.kyle_lambda(),
            "price_impact_r2":     self.price_impact.r_squared(),
            "mean_depth":          self.depth.mean_depth(),
            "autocorr_1":          self.efficiency.autocorrelation(1),
            "variance_ratio_5":    self.efficiency.variance_ratio(5),
            "efficiency_score":    self.efficiency.efficiency_score(),
            "resilience_score":    self.resilience.resilience_score(),
            "mean_recovery_bars":  self.resilience.mean_recovery_bars(),
            "n_shocks":            float(self.resilience.n_shocks()),
        }
        return metrics

    def full_report(self) -> Dict[str, Any]:
        return {
            "total_steps":       self._step,
            "vwas":              self.eff_spread.vwas(),
            "kyle_lambda":       self.price_impact.kyle_lambda(),
            "mean_depth":        self.depth.mean_depth(),
            "efficiency_score":  self.efficiency.efficiency_score(),
            "autocorr_1":        self.efficiency.autocorrelation(1),
            "variance_ratio_5":  self.efficiency.variance_ratio(5),
            "resilience_score":  self.resilience.resilience_score(),
            "mean_recovery":     self.resilience.mean_recovery_bars(),
            "n_shocks":          self.resilience.n_shocks(),
        }

    def record_marl_snapshot(self) -> None:
        """Record current metrics as MARL observation for baseline comparison."""
        self.baseline.record_marl(self.full_report())

    def record_baseline_snapshot(self, baseline: str) -> None:
        self.baseline.record_baseline(baseline, self.full_report())

    def marl_improvement(self, metric: str = "efficiency_score") -> float:
        return self.baseline.improvement_over_baseline(metric)
