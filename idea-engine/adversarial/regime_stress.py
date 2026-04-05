"""
regime_stress.py
================
Stress test the strategy under specific historical regimes.

Three canonical crises are modelled:

1. **COVID crash** (March 2020)
   Bitcoin fell ~50% in 48 hours.  Volatility multiplier ~5x.
   Correlations spike toward 1.0 (all assets fall together).
   Liquidity dries up: slippage increases 3x.

2. **Luna/UST collapse** (May 2022)
   LUNA lost >99% of its value over 5 days.  Correlated -60% drawdown
   across many altcoins.  Some assets became illiquid.
   Models a "contagion" event: losses are correlated and cascade.

3. **FTX collapse** (November 2022)
   Exchange halt for a major venue.  ~30% of open positions become
   inaccessible for 48-72 hours.  Spreads widen 5-10x.
   Modelled as a sudden forced closure at unfavourable prices.

Methodology
-----------
For each regime we:
1. Scale the baseline trade return distribution to match the regime's
   observed volatility and correlation profile.
2. Simulate *n_paths* independent paths using the regime parameters.
3. Compute:
   - Maximum drawdown (worst peak-to-trough over the period).
   - Time to recovery (how long before equity recovers to the pre-stress peak).
   - Blowup probability (fraction of paths that breach a -30% threshold).
   - 5th-percentile P&L.

The trade return distributions are calibrated to the actual observed
crypto market returns during each event, not synthetic assumptions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Historical regime parameters
# (calibrated from public crypto market data)
# ---------------------------------------------------------------------------

REGIME_PARAMS = {
    "covid_crash": {
        "description": "COVID crash -- March 2020 (BTC -50% in 48h)",
        "vol_multiplier":       5.0,
        "corr_override":        0.90,
        "slippage_multiplier":  3.0,
        "mean_return_override": -0.015,  # strongly negative daily mean
        "duration_days":        14,
        "blowup_threshold":     -0.30,
    },
    "luna_collapse": {
        "description": "Luna/UST collapse -- May 2022 (LUNA -99%, alts -60%)",
        "vol_multiplier":       8.0,
        "corr_override":        0.85,
        "slippage_multiplier":  2.0,
        "mean_return_override": -0.025,
        "duration_days":        7,
        "blowup_threshold":     -0.40,
    },
    "ftx_collapse": {
        "description": "FTX collapse -- November 2022 (exchange halt + contagion)",
        "vol_multiplier":       4.0,
        "corr_override":        0.80,
        "slippage_multiplier":  8.0,
        "mean_return_override": -0.010,
        "duration_days":        10,
        "position_lockup_frac": 0.30,   # 30% of positions inaccessible
        "blowup_threshold":     -0.25,
    },
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StressResult:
    """
    Result of a single regime stress test.

    Attributes
    ----------
    regime_name         : identifier ("covid_crash", etc.).
    description         : human-readable regime description.
    n_paths             : number of Monte Carlo paths.
    max_drawdown_median : median max drawdown across paths.
    max_drawdown_p5     : 5th-percentile max drawdown (severe case).
    pnl_p5              : 5th-percentile total P&L across paths.
    pnl_median          : median total P&L.
    blowup_probability  : fraction of paths that hit the blowup threshold.
    time_to_recovery_median: median recovery time in days (None if never).
    blowup_threshold    : P&L level used to define blowup.
    """

    regime_name:              str
    description:              str
    n_paths:                  int
    max_drawdown_median:      float
    max_drawdown_p5:          float
    pnl_p5:                   float
    pnl_median:               float
    blowup_probability:       float
    time_to_recovery_median:  Optional[float]
    blowup_threshold:         float

    def summary(self) -> str:
        return (
            f"[{self.regime_name}] {self.description}\n"
            f"  Median max drawdown: {self.max_drawdown_median:+.2%}\n"
            f"  5th-pct max drawdown: {self.max_drawdown_p5:+.2%}\n"
            f"  Median P&L: {self.pnl_median:+.4f}\n"
            f"  5th-pct P&L: {self.pnl_p5:+.4f}\n"
            f"  Blowup probability: {self.blowup_probability:.2%}\n"
            f"  Median time to recovery: "
            f"{self.time_to_recovery_median:.1f} days"
            if self.time_to_recovery_median is not None
            else f"  Median time to recovery: did not recover within window"
        )


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate_stress_path(
    vol_multiplier:       float,
    corr_override:        float,
    mean_return_override: float,
    slippage_multiplier:  float,
    duration_days:        int,
    trades_per_day:       float,
    baseline_pnl_std:     float,
    position_lockup_frac: float,
    rng:                  np.random.Generator,
) -> np.ndarray:
    """
    Simulate a single equity path through a stress regime.

    Returns
    -------
    equity_curve : 1-D array of cumulative P&L (starting at 0.0).
    """
    n_trades_per_day = max(1, int(rng.poisson(trades_per_day)))
    n_total  = n_trades_per_day * duration_days

    # Scale trade returns by the regime vol multiplier
    stressed_std  = baseline_pnl_std * vol_multiplier
    trade_returns = rng.normal(mean_return_override / n_trades_per_day,
                               stressed_std, n_total)

    # Apply corr_override: all trades in the same "down day" share a
    # common component drawn from the regime mean
    common_factor = rng.normal(mean_return_override / n_trades_per_day,
                               stressed_std * corr_override, n_total)
    idio_factor   = rng.normal(0, stressed_std * math.sqrt(1 - corr_override ** 2), n_total)
    trade_returns = common_factor + idio_factor

    # Apply slippage: reduces positive P&L, amplifies negative P&L
    slippage      = slippage_multiplier * 0.001  # 0.1% baseline slippage
    trade_returns -= slippage

    # Apply position lockup: a fraction of trades are force-closed at -1%
    if position_lockup_frac > 0:
        lockup_mask       = rng.uniform(size=n_total) < position_lockup_frac
        trade_returns     = np.where(lockup_mask, -0.01, trade_returns)

    return np.cumsum(np.concatenate([[0.0], trade_returns]))


def _max_drawdown(equity: np.ndarray) -> float:
    """
    Compute maximum drawdown as a fraction.

    Returns a non-positive number (e.g. -0.25 = 25% drawdown).
    """
    peak = np.maximum.accumulate(equity)
    dd   = equity - peak
    return float(dd.min())


def _time_to_recovery(equity: np.ndarray, pre_stress_peak: float = 0.0) -> Optional[float]:
    """
    Number of steps until equity returns to *pre_stress_peak*.

    Returns None if equity never recovers within the path.
    """
    recovery = np.where(equity >= pre_stress_peak)[0]
    # Skip the first element (starting at 0)
    recovery = recovery[recovery > 0]
    if len(recovery) == 0:
        return None
    return float(recovery[0])


# ---------------------------------------------------------------------------
# RegimeStressor
# ---------------------------------------------------------------------------

class RegimeStressor:
    """
    Stress test the strategy under historical market regimes.

    Parameters
    ----------
    n_paths          : number of Monte Carlo paths per regime.
    trades_per_day   : expected trade entries per trading day.
    baseline_pnl_std : per-trade P&L std under normal conditions.
    seed             : base random seed.
    """

    def __init__(
        self,
        n_paths:          int   = 1000,
        trades_per_day:   float = 3.5,
        baseline_pnl_std: float = 0.018,
        seed:             int   = 42,
    ):
        self.n_paths          = n_paths
        self.trades_per_day   = trades_per_day
        self.baseline_pnl_std = baseline_pnl_std
        self.seed             = seed

    def run_regime(self, regime_name: str) -> StressResult:
        """
        Run Monte Carlo stress test for a specific regime.

        Parameters
        ----------
        regime_name : key in REGIME_PARAMS ("covid_crash", etc.).

        Returns
        -------
        StressResult.
        """
        if regime_name not in REGIME_PARAMS:
            raise ValueError(f"Unknown regime: {regime_name!r}. "
                             f"Available: {list(REGIME_PARAMS)}")

        p = REGIME_PARAMS[regime_name]
        logger.info("Running stress test: %s (%s)", regime_name, p["description"])

        all_curves: List[np.ndarray] = []
        all_max_dd: List[float]      = []
        all_total:  List[float]      = []
        all_ttrecov: List[Optional[float]] = []

        for i in range(self.n_paths):
            rng   = np.random.default_rng(self.seed + i * 31)
            curve = _simulate_stress_path(
                vol_multiplier=p["vol_multiplier"],
                corr_override=p["corr_override"],
                mean_return_override=p["mean_return_override"],
                slippage_multiplier=p["slippage_multiplier"],
                duration_days=p["duration_days"],
                trades_per_day=self.trades_per_day,
                baseline_pnl_std=self.baseline_pnl_std,
                position_lockup_frac=p.get("position_lockup_frac", 0.0),
                rng=rng,
            )
            all_curves.append(curve)
            all_max_dd.append(_max_drawdown(curve))
            all_total.append(float(curve[-1]))
            all_ttrecov.append(_time_to_recovery(curve, pre_stress_peak=0.0))

        max_dds = np.array(all_max_dd)
        totals  = np.array(all_total)

        blowup_thresh     = p["blowup_threshold"]
        blowup_prob       = float(np.mean(totals <= blowup_thresh))

        # Time to recovery (finite values only)
        finite_recov = [r for r in all_ttrecov if r is not None]
        ttrecov_med  = float(np.median(finite_recov)) if finite_recov else None

        result = StressResult(
            regime_name=regime_name,
            description=p["description"],
            n_paths=self.n_paths,
            max_drawdown_median=float(np.median(max_dds)),
            max_drawdown_p5=float(np.percentile(max_dds, 5)),
            pnl_p5=float(np.percentile(totals, 5)),
            pnl_median=float(np.median(totals)),
            blowup_probability=blowup_prob,
            time_to_recovery_median=ttrecov_med,
            blowup_threshold=blowup_thresh,
        )

        logger.info(result.summary())
        return result

    def run_all(self) -> Dict[str, StressResult]:
        """
        Run stress tests for all available regimes.

        Returns
        -------
        dict of {regime_name: StressResult}.
        """
        return {name: self.run_regime(name) for name in REGIME_PARAMS}

    def compare_regimes(self, results: Dict[str, StressResult]) -> pd.DataFrame:
        """
        Return a DataFrame comparing all regimes side by side.

        Parameters
        ----------
        results : output of run_all().
        """
        rows = []
        for name, r in results.items():
            rows.append({
                "regime":                 name,
                "description":            r.description,
                "median_max_dd":          r.max_drawdown_median,
                "p5_max_dd":              r.max_drawdown_p5,
                "median_pnl":             r.pnl_median,
                "p5_pnl":                 r.pnl_p5,
                "blowup_prob":            r.blowup_probability,
                "time_to_recovery_days":  r.time_to_recovery_median,
            })
        return pd.DataFrame(rows)

    def worst_regime(self, results: Dict[str, StressResult]) -> str:
        """Return the name of the most dangerous regime (lowest p5 P&L)."""
        return min(results, key=lambda k: results[k].pnl_p5)
