"""
correlation_attack.py
=====================
Test strategy robustness to correlation breakdown.

The strategy's ``corr_factor`` parameter controls how much inter-asset
correlation affects position sizing.  A sudden spike in realized
correlation (all coins dumping simultaneously) can violate the
diversification assumptions and cause large correlated losses.

Analyses performed
------------------
1. **Correlation spike test**
   Simulate 30 days where realized correlation jumps to 0.95.
   Compute: P&L distribution, max drawdown, blowup probability.

2. **Assumed vs realized divergence**
   What if our assumed correlation is wrong by 2x?
   (e.g., we assume 0.25 but true correlation is 0.50.)
   Compute: P&L impact of the misspecification.

3. **Critical correlation level**
   Find the minimum realized correlation at which the portfolio blows up
   (defined as exceeding the blowup_threshold).
   Binary search over correlation levels [0, 1].

4. **Minimum safe correlation assumption**
   The lowest corr_factor value such that, even if realized correlation
   is 50% higher, the blowup probability stays below a safety threshold.

Usage::

    attacker = CorrelationAttacker()
    result   = attacker.run()
    print(result.summary())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

logger = logging.getLogger(__name__)

# Default simulation parameters
DEFAULT_N_PATHS          = 1000
DEFAULT_BASELINE_STD     = 0.018
DEFAULT_TRADES_PER_DAY   = 3.5
DEFAULT_BLOWUP_THRESHOLD = -0.20  # -20% total P&L
DEFAULT_BLOWUP_SAFETY_P  = 0.05   # blowup probability must stay below 5%


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CorrelationScenario:
    """P&L statistics for a single realized-correlation scenario."""

    realized_corr:     float
    assumed_corr:      float
    pnl_median:        float
    pnl_p5:            float
    max_dd_median:     float
    blowup_probability: float
    n_paths:           int


@dataclass
class CorrelationResult:
    """
    Full result of the correlation attack analysis.

    Attributes
    ----------
    spike_result           : scenario with corr=0.95 (correlation spike).
    divergence_scenarios   : scenarios for various assumed vs realized mismatches.
    critical_corr          : realized correlation at which blowup prob exceeds threshold.
    min_safe_assumed_corr  : minimum corr_factor to maintain safety margin.
    current_corr_factor    : the strategy's current corr_factor value.
    """

    spike_result:           CorrelationScenario
    divergence_scenarios:   List[CorrelationScenario]
    critical_corr:          float
    min_safe_assumed_corr:  float
    current_corr_factor:    float = 0.25

    def summary(self) -> str:
        lines = [
            "=== CORRELATION ATTACK REPORT ===",
            f"Current corr_factor: {self.current_corr_factor:.2f}",
            "",
            "Correlation spike (realized=0.95, 30 days):",
            f"  Median P&L:        {self.spike_result.pnl_median:+.4f}",
            f"  5th-pct P&L:       {self.spike_result.pnl_p5:+.4f}",
            f"  Median max drawdown: {self.spike_result.max_dd_median:+.2%}",
            f"  Blowup probability: {self.spike_result.blowup_probability:.2%}",
            "",
            f"Critical correlation (blowup threshold): {self.critical_corr:.3f}",
            f"  -> Portfolio blows up when realized corr > {self.critical_corr:.3f}",
            "",
            f"Minimum safe assumed corr: {self.min_safe_assumed_corr:.3f}",
            "  -> Strategy must assume at least this level of correlation",
            "     to stay within the blowup probability safety margin.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate_corr_path(
    assumed_corr:   float,
    realized_corr:  float,
    n_days:         int,
    trades_per_day: float,
    baseline_std:   float,
    mean_return:    float,
    blowup_threshold: float,
    rng:            np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Simulate one equity path with a given assumed vs realized correlation.

    The position size is scaled inversely with *assumed_corr* (higher
    assumed correlation = smaller positions = more conservative).
    The actual P&L is driven by *realized_corr* (true correlation).

    Returns
    -------
    (total_pnl, max_drawdown, hit_blowup)
    """
    n_trades = max(1, int(rng.poisson(trades_per_day * n_days)))

    # Position size: inversely proportional to assumed correlation
    # Baseline position size = 1.0 at assumed_corr = 0.25
    pos_size = 0.25 / max(assumed_corr, 0.01)
    pos_size = min(pos_size, 4.0)  # cap at 4x

    # Common market factor (drives realized correlation)
    market_factor = rng.normal(mean_return, baseline_std * realized_corr, n_trades)
    idio_factor   = rng.normal(0, baseline_std * math.sqrt(1 - realized_corr ** 2), n_trades)
    raw_returns   = market_factor + idio_factor

    # Scale by position size
    trade_pnls = raw_returns * pos_size
    equity     = np.concatenate([[0.0], np.cumsum(trade_pnls)])

    total_pnl  = float(equity[-1])
    peak       = np.maximum.accumulate(equity)
    dd         = float((equity - peak).min())
    hit_blowup = 1.0 if total_pnl <= blowup_threshold else 0.0

    return total_pnl, dd, hit_blowup


def _run_scenario(
    assumed_corr:    float,
    realized_corr:   float,
    n_paths:         int,
    n_days:          int,
    trades_per_day:  float,
    baseline_std:    float,
    mean_return:     float,
    blowup_threshold: float,
    seed:            int,
) -> CorrelationScenario:
    """Run *n_paths* simulations for a single (assumed, realized) pair."""
    totals  = np.zeros(n_paths)
    dds     = np.zeros(n_paths)
    blowups = np.zeros(n_paths)

    for i in range(n_paths):
        rng = np.random.default_rng(seed + i * 17)
        pnl, dd, blow = _simulate_corr_path(
            assumed_corr=assumed_corr,
            realized_corr=realized_corr,
            n_days=n_days,
            trades_per_day=trades_per_day,
            baseline_std=baseline_std,
            mean_return=mean_return,
            blowup_threshold=blowup_threshold,
            rng=rng,
        )
        totals[i]  = pnl
        dds[i]     = dd
        blowups[i] = blow

    return CorrelationScenario(
        realized_corr=realized_corr,
        assumed_corr=assumed_corr,
        pnl_median=float(np.median(totals)),
        pnl_p5=float(np.percentile(totals, 5)),
        max_dd_median=float(np.median(dds)),
        blowup_probability=float(np.mean(blowups)),
        n_paths=n_paths,
    )


# ---------------------------------------------------------------------------
# CorrelationAttacker
# ---------------------------------------------------------------------------

class CorrelationAttacker:
    """
    Test strategy robustness to correlation spikes and misspecification.

    Parameters
    ----------
    current_corr_factor  : strategy's current corr_factor parameter.
    n_paths              : Monte Carlo paths per scenario.
    n_days_spike         : duration of the correlation spike (days).
    baseline_std         : per-trade P&L std under normal correlation.
    mean_return          : average per-trade return (slightly positive).
    blowup_threshold     : total P&L at which a path is considered a blowup.
    blowup_safety_prob   : maximum acceptable blowup probability.
    seed                 : base random seed.
    """

    def __init__(
        self,
        current_corr_factor: float = 0.25,
        n_paths:             int   = DEFAULT_N_PATHS,
        n_days_spike:        int   = 30,
        baseline_std:        float = DEFAULT_BASELINE_STD,
        mean_return:         float = 0.003,
        blowup_threshold:    float = DEFAULT_BLOWUP_THRESHOLD,
        blowup_safety_prob:  float = DEFAULT_BLOWUP_SAFETY_P,
        seed:                int   = 42,
    ):
        self.current_corr_factor = current_corr_factor
        self.n_paths             = n_paths
        self.n_days_spike        = n_days_spike
        self.baseline_std        = baseline_std
        self.mean_return         = mean_return
        self.blowup_threshold    = blowup_threshold
        self.blowup_safety_prob  = blowup_safety_prob
        self.seed                = seed

    def _scenario(self, assumed: float, realized: float) -> CorrelationScenario:
        return _run_scenario(
            assumed_corr=assumed,
            realized_corr=realized,
            n_paths=self.n_paths,
            n_days=self.n_days_spike,
            trades_per_day=DEFAULT_TRADES_PER_DAY,
            baseline_std=self.baseline_std,
            mean_return=self.mean_return,
            blowup_threshold=self.blowup_threshold,
            seed=self.seed,
        )

    def correlation_spike_test(self) -> CorrelationScenario:
        """
        Simulate 30 days with realized correlation = 0.95.

        The strategy still assumes *current_corr_factor* for sizing.
        """
        logger.info("Running correlation spike test (realized=0.95, days=%d).", self.n_days_spike)
        return self._scenario(
            assumed=self.current_corr_factor,
            realized=0.95,
        )

    def divergence_test(
        self, multipliers: Optional[List[float]] = None
    ) -> List[CorrelationScenario]:
        """
        Test scenarios where realized correlation is *multiplier* times assumed.

        Parameters
        ----------
        multipliers : list of realized/assumed ratios to test.
                      Default: [1.0, 1.5, 2.0, 3.0, 4.0].

        Returns
        -------
        List of CorrelationScenario objects.
        """
        if multipliers is None:
            multipliers = [1.0, 1.5, 2.0, 3.0, 4.0]

        scenarios = []
        for mult in multipliers:
            realized = min(self.current_corr_factor * mult, 0.99)
            logger.info(
                "Divergence test: assumed=%.3f, realized=%.3f (x%.1f)",
                self.current_corr_factor, realized, mult,
            )
            sc = self._scenario(assumed=self.current_corr_factor, realized=realized)
            scenarios.append(sc)
        return scenarios

    def find_critical_correlation(self) -> float:
        """
        Binary search for the realized correlation level at which
        blowup probability first exceeds *blowup_safety_prob*.

        Returns
        -------
        critical_corr : float in [0, 1].
        """
        logger.info("Searching for critical correlation level...")

        def blowup_prob_at(realized: float) -> float:
            sc = self._scenario(assumed=self.current_corr_factor, realized=realized)
            return sc.blowup_probability

        # Quick scan first
        corr_grid = np.linspace(0.10, 0.99, 20)
        probs = [blowup_prob_at(c) for c in corr_grid]

        # Find where it crosses the safety threshold
        for i in range(len(probs) - 1):
            if probs[i] < self.blowup_safety_prob <= probs[i + 1]:
                # Bisect in this interval
                lo, hi = corr_grid[i], corr_grid[i + 1]
                for _ in range(10):
                    mid = (lo + hi) / 2
                    if blowup_prob_at(mid) < self.blowup_safety_prob:
                        lo = mid
                    else:
                        hi = mid
                crit = (lo + hi) / 2
                logger.info("Critical correlation: %.4f", crit)
                return float(crit)

        # If blowup never exceeds threshold in [0.1, 0.99]
        logger.info("Blowup probability never exceeds threshold; critical_corr = 1.0")
        return 1.0

    def find_min_safe_assumed_corr(self) -> float:
        """
        Find the minimum assumed corr_factor such that, even if realized
        correlation is 50% higher, blowup probability stays below threshold.

        Returns
        -------
        min_safe_assumed_corr : float.
        """
        logger.info("Searching for minimum safe assumed corr_factor...")

        def blowup_given_assumed(assumed: float) -> float:
            realized = min(assumed * 1.5, 0.99)
            sc = self._scenario(assumed=assumed, realized=realized)
            return sc.blowup_probability

        assumed_grid = np.linspace(0.05, 0.80, 30)
        for a in assumed_grid:
            prob = blowup_given_assumed(a)
            if prob <= self.blowup_safety_prob:
                logger.info("Min safe assumed corr: %.4f", a)
                return float(a)

        logger.warning("Could not find safe assumed corr; returning 0.80")
        return 0.80

    def run(self) -> CorrelationResult:
        """
        Execute the full correlation attack suite.

        Returns
        -------
        CorrelationResult with all sub-analysis results.
        """
        spike       = self.correlation_spike_test()
        divergence  = self.divergence_test()
        critical    = self.find_critical_correlation()
        min_safe    = self.find_min_safe_assumed_corr()

        result = CorrelationResult(
            spike_result=spike,
            divergence_scenarios=divergence,
            critical_corr=critical,
            min_safe_assumed_corr=min_safe,
            current_corr_factor=self.current_corr_factor,
        )
        logger.info(result.summary())
        return result

    def divergence_dataframe(self, scenarios: List[CorrelationScenario]) -> pd.DataFrame:
        """Return divergence scenarios as a DataFrame."""
        return pd.DataFrame([{
            "assumed_corr":       s.assumed_corr,
            "realized_corr":      s.realized_corr,
            "ratio":              s.realized_corr / max(s.assumed_corr, 1e-6),
            "median_pnl":         s.pnl_median,
            "p5_pnl":             s.pnl_p5,
            "median_max_dd":      s.max_dd_median,
            "blowup_probability": s.blowup_probability,
        } for s in scenarios])
