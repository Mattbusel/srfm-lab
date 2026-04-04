"""
research/regime_lab/simulator.py
==================================
Regime-aware Monte Carlo simulator.

Extends spacetime/engine/mc.py with full Markov-regime dynamics and
Black-Hole (BH) activation tracking per regime.

Classes
-------
RegimeMCSim    — Monte Carlo engine with regime transitions
RegimeMCResult — result dataclass

Functions
---------
compute_regime_conditional_var(result) -> Dict[str, float]
tail_risk_attribution(result) -> pd.DataFrame
plot_regime_mc_paths(result, n_show=100, save_path=None)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BULL     = "BULL"
BEAR     = "BEAR"
SIDEWAYS = "SIDEWAYS"
HIGH_VOL = "HIGH_VOL"
REGIMES  = (BULL, BEAR, SIDEWAYS, HIGH_VOL)

# Default per-regime trade parameters (mu, sigma, win_rate per trade)
DEFAULT_REGIME_TRADE_PARAMS: Dict[str, Dict[str, float]] = {
    BULL:     {"mu": 0.015,  "sigma": 0.040, "win_rate": 0.58, "trades_per_month": 12.0},
    BEAR:     {"mu": -0.005, "sigma": 0.060, "win_rate": 0.42, "trades_per_month": 8.0},
    SIDEWAYS: {"mu": 0.008,  "sigma": 0.025, "win_rate": 0.52, "trades_per_month": 6.0},
    HIGH_VOL: {"mu": 0.020,  "sigma": 0.090, "win_rate": 0.50, "trades_per_month": 14.0},
}

DEFAULT_TRANSITION_MATRIX: np.ndarray = np.array([
    [0.97,  0.01,  0.015, 0.005],
    [0.01,  0.96,  0.020, 0.010],
    [0.02,  0.02,  0.950, 0.010],
    [0.05,  0.05,  0.050, 0.850],
], dtype=float)

# BH activation rates per regime (fraction of bars where BH fires)
DEFAULT_BH_ACTIVATION: Dict[str, float] = {
    BULL:     0.12,
    BEAR:     0.04,
    SIDEWAYS: 0.06,
    HIGH_VOL: 0.08,
}


# ===========================================================================
# 1. RegimeMCResult
# ===========================================================================

@dataclass
class RegimeMCResult:
    """Full result from a regime-aware Monte Carlo simulation."""
    # Aggregate path arrays
    final_equities:   np.ndarray   # (n_sims,)
    equity_paths:     np.ndarray   # (n_sims, n_months + 1) or (n_sims, n_bars + 1)
    max_drawdowns:    np.ndarray   # (n_sims,)
    blowup_flags:     np.ndarray   # (n_sims,) bool

    # Per-simulation regime stats
    regime_fractions: np.ndarray   # (n_sims, K) — time spent per regime
    regime_pnl_share: np.ndarray   # (n_sims, K) — P&L attributed to each regime

    # BH stats
    bh_activations:   np.ndarray   # (n_sims,) total BH firing count
    bh_per_regime:    np.ndarray   # (n_sims, K) BH firings per regime

    # Summary scalars
    n_sims:            int
    n_months:          int
    starting_equity:   float
    blowup_rate:       float
    median_equity:     float
    mean_equity:       float
    pct_5:             float
    pct_25:            float
    pct_75:            float
    pct_95:            float
    max_dd_mean:       float
    max_dd_p95:        float

    # Per-regime statistics
    regime_stats:      Dict[str, Dict[str, float]] = field(default_factory=dict)
    # regime → {mean_equity_share, pnl_share, bh_rate, time_pct}

    blowup_threshold:  float = 0.10

    def to_summary_df(self) -> pd.DataFrame:
        rows = []
        for r, stats in self.regime_stats.items():
            rows.append({"regime": r, **stats})
        return pd.DataFrame(rows)

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.final_equities, p))

    def sharpe_ratio(self, annualise: bool = True) -> float:
        rets = (self.final_equities - self.starting_equity) / self.starting_equity
        if rets.std() == 0:
            return 0.0
        sr = rets.mean() / rets.std()
        return sr * (12 ** 0.5) if annualise else sr


# ===========================================================================
# 2. RegimeMCSim
# ===========================================================================

class RegimeMCSim:
    """
    Regime-aware Monte Carlo simulator.

    For each simulation path:
      1. The regime evolves as a Markov chain.
      2. In each period the number of trades is drawn from the regime's rate.
      3. Trade returns are drawn from N(mu_k, sigma_k^2).
      4. BH activations are drawn from a Bernoulli with regime-specific probability.
      5. BH activation multiplies the trade return by (1 + bh_boost) with some probability.

    Parameters
    ----------
    regime_params       : dict regime → {mu, sigma, win_rate, trades_per_month}
    transition_matrix   : (K, K) row-stochastic
    n_sims              : number of MC paths (default 10_000)
    bh_activation_rates : dict regime → P(BH fires per bar)
    bh_return_boost     : multiplicative return boost when BH fires
    serial_corr         : AR(1) loss serial correlation
    blowup_threshold    : equity below this fraction of start = blowup
    """

    def __init__(self,
                 regime_params: Optional[Dict[str, Dict[str, float]]] = None,
                 transition_matrix: Optional[np.ndarray] = None,
                 n_sims: int = 10_000,
                 bh_activation_rates: Optional[Dict[str, float]] = None,
                 bh_return_boost: float = 0.40,
                 serial_corr: float = 0.20,
                 blowup_threshold: float = 0.10):
        self.regime_params       = regime_params or DEFAULT_REGIME_TRADE_PARAMS.copy()
        self.bh_activation_rates = bh_activation_rates or DEFAULT_BH_ACTIVATION.copy()
        self.bh_return_boost     = bh_return_boost
        self.serial_corr         = serial_corr
        self.blowup_threshold    = blowup_threshold
        self.n_sims              = n_sims

        if transition_matrix is None:
            self.transition_matrix = DEFAULT_TRANSITION_MATRIX.copy()
        else:
            tm = np.asarray(transition_matrix, dtype=float)
            self.transition_matrix = tm / tm.sum(axis=1, keepdims=True)

        self._K        = len(REGIMES)
        self._regime_list = list(REGIMES)
        self._r2i      = {r: i for i, r in enumerate(REGIMES)}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _stationary(self) -> np.ndarray:
        P = self.transition_matrix[:self._K, :self._K]
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        v   = np.abs(eigvecs[:, idx].real)
        return v / v.sum()

    def _regime_params(self, regime: str) -> Dict[str, float]:
        return self.regime_params.get(regime, DEFAULT_REGIME_TRADE_PARAMS[SIDEWAYS])

    # ------------------------------------------------------------------ #
    # Core simulation
    # ------------------------------------------------------------------ #

    def run(self, starting_equity: float = 1_000_000.0,
            n_months: int = 12,
            seed: Optional[int] = None) -> RegimeMCResult:
        """
        Run the Monte Carlo simulation.

        Parameters
        ----------
        starting_equity : initial account value
        n_months        : simulation horizon in months
        seed            : RNG seed for reproducibility

        Returns
        -------
        RegimeMCResult
        """
        rng = np.random.default_rng(seed)
        K   = self._K

        # Pre-allocate output arrays
        final_equities  = np.zeros(self.n_sims)
        max_drawdowns   = np.zeros(self.n_sims)
        blowup_flags    = np.zeros(self.n_sims, dtype=bool)
        regime_fracs    = np.zeros((self.n_sims, K))
        regime_pnl_sh   = np.zeros((self.n_sims, K))
        bh_total        = np.zeros(self.n_sims, dtype=int)
        bh_per_regime   = np.zeros((self.n_sims, K), dtype=int)

        # Store full equity paths (month-end snapshots)
        equity_paths = np.zeros((self.n_sims, n_months + 1))
        equity_paths[:, 0] = starting_equity

        pi0 = self._stationary()
        P   = self.transition_matrix[:K, :K]

        for sim in range(self.n_sims):
            eq            = starting_equity
            peak_eq       = starting_equity
            max_dd        = 0.0
            blowup        = False
            regime_counts = np.zeros(K, dtype=int)
            regime_pnl    = np.zeros(K)
            bh_counts_sim = np.zeros(K, dtype=int)

            # Starting regime
            cur_regime = int(rng.choice(K, p=pi0))

            # AR(1) loss state
            prev_loss_factor = 0.0

            for month in range(n_months):
                params         = self._regime_params(self._regime_list[cur_regime])
                mu             = params["mu"]
                sigma          = params["sigma"]
                n_trades_float = params["trades_per_month"]
                n_trades       = max(1, int(rng.poisson(n_trades_float)))

                month_pnl = 0.0

                for _ in range(n_trades):
                    # AR(1) serial correlation adjustment
                    ar_adj = self.serial_corr * prev_loss_factor
                    z      = float(rng.standard_normal())
                    ret    = mu + sigma * z + ar_adj

                    # BH activation
                    bh_rate  = self.bh_activation_rates.get(self._regime_list[cur_regime], 0.08)
                    bh_fired = float(rng.random()) < bh_rate
                    if bh_fired:
                        ret += self.bh_return_boost * abs(sigma)
                        bh_counts_sim[cur_regime] += 1

                    trade_pnl = eq * ret * 0.05   # 5% position size
                    month_pnl += trade_pnl
                    prev_loss_factor = ret if ret < 0 else 0.0

                regime_counts[cur_regime] += 1
                regime_pnl[cur_regime]    += month_pnl

                eq += month_pnl
                eq  = max(eq, 0.0)

                if eq > peak_eq:
                    peak_eq = eq
                dd = (peak_eq - eq) / peak_eq if peak_eq > 0 else 0.0
                max_dd = max(max_dd, dd)

                equity_paths[sim, month + 1] = eq

                if eq < starting_equity * self.blowup_threshold:
                    blowup = True
                    # Fill remaining months at floor
                    equity_paths[sim, month + 2:] = eq
                    break

                # Regime transition
                row = P[cur_regime]
                cur_regime = int(rng.choice(K, p=row))

            final_equities[sim]     = eq
            max_drawdowns[sim]      = max_dd
            blowup_flags[sim]       = blowup
            regime_fracs[sim]       = regime_counts / max(regime_counts.sum(), 1)
            regime_pnl_sh[sim]      = regime_pnl
            bh_total[sim]           = int(bh_counts_sim.sum())
            bh_per_regime[sim]      = bh_counts_sim

        # Build per-regime aggregate stats
        regime_stats: Dict[str, Dict[str, float]] = {}
        for i, r in enumerate(REGIMES):
            mean_pnl_share = float(np.mean(regime_pnl_sh[:, i]))
            mean_time_pct  = float(np.mean(regime_fracs[:, i]) * 100)
            mean_bh_rate   = float(np.mean(bh_per_regime[:, i] /
                                            np.maximum(bh_total, 1)))
            regime_stats[r] = {
                "mean_pnl_share":    round(mean_pnl_share, 2),
                "mean_time_pct":     round(mean_time_pct, 2),
                "mean_bh_fire_rate": round(mean_bh_rate, 4),
            }

        return RegimeMCResult(
            final_equities=final_equities,
            equity_paths=equity_paths,
            max_drawdowns=max_drawdowns,
            blowup_flags=blowup_flags,
            regime_fractions=regime_fracs,
            regime_pnl_share=regime_pnl_sh,
            bh_activations=bh_total,
            bh_per_regime=bh_per_regime,
            n_sims=self.n_sims,
            n_months=n_months,
            starting_equity=starting_equity,
            blowup_rate=float(blowup_flags.mean()),
            median_equity=float(np.median(final_equities)),
            mean_equity=float(np.mean(final_equities)),
            pct_5=float(np.percentile(final_equities, 5)),
            pct_25=float(np.percentile(final_equities, 25)),
            pct_75=float(np.percentile(final_equities, 75)),
            pct_95=float(np.percentile(final_equities, 95)),
            max_dd_mean=float(np.mean(max_drawdowns)),
            max_dd_p95=float(np.percentile(max_drawdowns, 95)),
            regime_stats=regime_stats,
            blowup_threshold=self.blowup_threshold,
        )

    # ------------------------------------------------------------------ #
    # BH-physics simulation
    # ------------------------------------------------------------------ #

    def run_with_bh_physics(self,
                             closes_history: np.ndarray | pd.Series,
                             starting_equity: float = 1_000_000.0,
                             n_months: int = 12,
                             timeframe_weights: Optional[Dict[str, float]] = None,
                             seed: Optional[int] = None) -> RegimeMCResult:
        """
        Simulate with proper BH physics.

        BH activation follows: delta_score = tf × mass × ATR

        At each simulated bar a BH fires if delta_score > threshold (0.5).
        The ATR is drawn from the empirical ATR distribution of the
        historical closes series.

        Parameters
        ----------
        closes_history     : historical close prices (used to estimate ATR dist)
        starting_equity    : initial equity
        n_months           : simulation horizon
        timeframe_weights  : dict tf_name → weight multiplier (default all 1.0)
        seed               : RNG seed

        Returns
        -------
        RegimeMCResult
        """
        from research.regime_lab.detector import _atr as _compute_atr

        closes = np.asarray(closes_history, dtype=float)
        atr    = _compute_atr(closes, closes, closes, period=14)
        atr_pct = atr / (closes + 1e-10)
        atr_dist = atr_pct[~np.isnan(atr_pct)]

        if timeframe_weights is None:
            timeframe_weights = {"daily": 1.0, "hourly": 0.5, "15m": 0.25}

        # Compute composite timeframe factor
        tf_total = sum(timeframe_weights.values())

        rng = np.random.default_rng(seed)
        K   = self._K

        final_equities  = np.zeros(self.n_sims)
        max_drawdowns   = np.zeros(self.n_sims)
        blowup_flags    = np.zeros(self.n_sims, dtype=bool)
        regime_fracs    = np.zeros((self.n_sims, K))
        regime_pnl_sh   = np.zeros((self.n_sims, K))
        bh_total        = np.zeros(self.n_sims, dtype=int)
        bh_per_regime   = np.zeros((self.n_sims, K), dtype=int)
        equity_paths    = np.zeros((self.n_sims, n_months + 1))
        equity_paths[:, 0] = starting_equity

        pi0 = self._stationary()
        P   = self.transition_matrix[:K, :K]

        n_bars_per_month = 21  # trading days

        for sim in range(self.n_sims):
            eq            = starting_equity
            peak_eq       = starting_equity
            max_dd        = 0.0
            blowup        = False
            regime_counts = np.zeros(K, dtype=int)
            regime_pnl    = np.zeros(K)
            bh_counts_sim = np.zeros(K, dtype=int)

            cur_regime    = int(rng.choice(K, p=pi0))
            prev_loss_fac = 0.0

            for month in range(n_months):
                month_pnl = 0.0
                for bar in range(n_bars_per_month):
                    regime_name = self._regime_list[cur_regime]
                    params      = self._regime_params(regime_name)
                    mu          = params["mu"]
                    sigma       = params["sigma"]

                    # BH physics: delta_score = tf × mass × ATR
                    atr_sample = float(rng.choice(atr_dist)) if len(atr_dist) > 0 else 0.01
                    mass       = float(rng.exponential(1.0))  # mass ~ Exp(1) on TIMELIKE bar
                    delta_score = tf_total * mass * atr_sample

                    bh_fired = delta_score > 0.5

                    # Generate trade if regime triggers entry
                    entry_prob = params.get("trades_per_month", 10) / n_bars_per_month
                    if float(rng.random()) < entry_prob:
                        ar_adj    = self.serial_corr * prev_loss_fac
                        z         = float(rng.standard_normal())
                        ret       = mu + sigma * z + ar_adj

                        if bh_fired:
                            ret += self.bh_return_boost * delta_score * 0.1
                            bh_counts_sim[cur_regime] += 1

                        trade_pnl = eq * ret * 0.05
                        month_pnl += trade_pnl
                        prev_loss_fac = ret if ret < 0 else 0.0

                    # Bar-level regime transition
                    row = P[cur_regime]
                    cur_regime = int(rng.choice(K, p=row))

                regime_counts[cur_regime] += 1
                regime_pnl[cur_regime]    += month_pnl

                eq += month_pnl
                eq  = max(eq, 0.0)

                if eq > peak_eq:
                    peak_eq = eq
                dd = (peak_eq - eq) / peak_eq if peak_eq > 0 else 0.0
                max_dd = max(max_dd, dd)

                equity_paths[sim, month + 1] = eq

                if eq < starting_equity * self.blowup_threshold:
                    blowup = True
                    equity_paths[sim, month + 2:] = eq
                    break

            final_equities[sim]   = eq
            max_drawdowns[sim]    = max_dd
            blowup_flags[sim]     = blowup
            regime_fracs[sim]     = regime_counts / max(regime_counts.sum(), 1)
            regime_pnl_sh[sim]    = regime_pnl
            bh_total[sim]         = int(bh_counts_sim.sum())
            bh_per_regime[sim]    = bh_counts_sim

        regime_stats: Dict[str, Dict[str, float]] = {}
        for i, r in enumerate(REGIMES):
            regime_stats[r] = {
                "mean_pnl_share":    round(float(np.mean(regime_pnl_sh[:, i])), 2),
                "mean_time_pct":     round(float(np.mean(regime_fracs[:, i]) * 100), 2),
                "mean_bh_fire_rate": round(float(np.mean(bh_per_regime[:, i] /
                                                           np.maximum(bh_total, 1))), 4),
            }

        return RegimeMCResult(
            final_equities=final_equities,
            equity_paths=equity_paths,
            max_drawdowns=max_drawdowns,
            blowup_flags=blowup_flags,
            regime_fractions=regime_fracs,
            regime_pnl_share=regime_pnl_sh,
            bh_activations=bh_total,
            bh_per_regime=bh_per_regime,
            n_sims=self.n_sims,
            n_months=n_months,
            starting_equity=starting_equity,
            blowup_rate=float(blowup_flags.mean()),
            median_equity=float(np.median(final_equities)),
            mean_equity=float(np.mean(final_equities)),
            pct_5=float(np.percentile(final_equities, 5)),
            pct_25=float(np.percentile(final_equities, 25)),
            pct_75=float(np.percentile(final_equities, 75)),
            pct_95=float(np.percentile(final_equities, 95)),
            max_dd_mean=float(np.mean(max_drawdowns)),
            max_dd_p95=float(np.percentile(max_drawdowns, 95)),
            regime_stats=regime_stats,
            blowup_threshold=self.blowup_threshold,
        )


# ===========================================================================
# 3. compute_regime_conditional_var
# ===========================================================================

def compute_regime_conditional_var(result: RegimeMCResult,
                                    confidence: float = 0.95
                                    ) -> Dict[str, float]:
    """
    Compute Conditional Value-at-Risk (CVaR) per regime.

    For each regime, we identify the simulations where that regime
    dominated (time_fraction > 0.5) and compute CVaR within that subset.

    Parameters
    ----------
    result     : RegimeMCResult
    confidence : CVaR confidence level (default 0.95)

    Returns
    -------
    Dict regime → CVaR (as positive loss fraction)
    """
    K    = len(REGIMES)
    cvar: Dict[str, float] = {}

    for i, r in enumerate(REGIMES):
        # Simulations dominated by this regime
        dominant_mask = result.regime_fractions[:, i] > 0.4
        if dominant_mask.sum() < 10:
            cvar[r] = 0.0
            continue

        rets = (result.final_equities[dominant_mask] - result.starting_equity) / result.starting_equity
        tail_pct = 1 - confidence
        threshold = np.percentile(rets, tail_pct * 100)
        tail_rets = rets[rets <= threshold]
        cvar[r] = float(-np.mean(tail_rets)) if len(tail_rets) > 0 else 0.0

    return cvar


# ===========================================================================
# 4. tail_risk_attribution
# ===========================================================================

def tail_risk_attribution(result: RegimeMCResult,
                           tail_pct: float = 5.0) -> pd.DataFrame:
    """
    Determine which regimes drive the worst simulation outcomes.

    Selects the bottom *tail_pct* percent of simulations by final equity,
    then analyses what fraction of time was spent in each regime for those
    bad paths vs the full distribution.

    Parameters
    ----------
    result   : RegimeMCResult
    tail_pct : bottom X% considered "tail" (default 5)

    Returns
    -------
    pd.DataFrame with regime × {tail_avg_time, full_avg_time, overweight}
    """
    n_tail = max(1, int(result.n_sims * tail_pct / 100))
    worst_idx = np.argsort(result.final_equities)[:n_tail]

    rows = []
    for i, r in enumerate(REGIMES):
        tail_time = float(np.mean(result.regime_fractions[worst_idx, i]))
        full_time  = float(np.mean(result.regime_fractions[:, i]))
        overweight = (tail_time / full_time - 1.0) if full_time > 0 else 0.0
        # BH attribution
        tail_bh    = float(np.mean(result.bh_per_regime[worst_idx, i]))
        full_bh    = float(np.mean(result.bh_per_regime[:, i]))

        rows.append({
            "regime":         r,
            "tail_time_pct":  round(tail_time * 100, 2),
            "full_time_pct":  round(full_time  * 100, 2),
            "overweight_pct": round(overweight * 100, 2),
            "tail_avg_bh":    round(tail_bh, 2),
            "full_avg_bh":    round(full_bh, 2),
        })

    df = pd.DataFrame(rows)
    df.sort_values("overweight_pct", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ===========================================================================
# 5. plot_regime_mc_paths
# ===========================================================================

def plot_regime_mc_paths(result: RegimeMCResult,
                          n_show: int = 100,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (14, 8)) -> Any:
    """
    Plot Monte Carlo equity paths with percentile fan.

    Parameters
    ----------
    result    : RegimeMCResult
    n_show    : number of individual paths to overlay (default 100)
    save_path : optional PNG save path
    figsize   : matplotlib figsize

    Returns
    -------
    matplotlib Figure or None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed.")
        return None

    paths = result.equity_paths
    n_sims, n_steps = paths.shape
    x = np.arange(n_steps)

    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    # Individual paths (random subset)
    rng  = np.random.default_rng(42)
    show_idx = rng.choice(n_sims, size=min(n_show, n_sims), replace=False)
    for idx in show_idx:
        ax1.plot(x, paths[idx], color="#90CAF9", alpha=0.3, linewidth=0.5)

    # Percentile fan
    pct_5   = np.percentile(paths, 5,  axis=0)
    pct_25  = np.percentile(paths, 25, axis=0)
    pct_50  = np.percentile(paths, 50, axis=0)
    pct_75  = np.percentile(paths, 75, axis=0)
    pct_95  = np.percentile(paths, 95, axis=0)

    ax1.fill_between(x, pct_5,  pct_95, alpha=0.15, color="#1565C0", label="5-95 %ile")
    ax1.fill_between(x, pct_25, pct_75, alpha=0.25, color="#1565C0", label="25-75 %ile")
    ax1.plot(x, pct_50, color="#0D47A1", linewidth=2.0, label="Median", zorder=5)
    ax1.axhline(result.starting_equity, color="black", linewidth=1.0,
                linestyle="--", alpha=0.5, label="Start equity")

    ax1.set_ylabel("Portfolio Equity ($)", fontsize=11)
    ax1.set_title(f"Regime-Aware Monte Carlo — {result.n_sims:,} Paths × {result.n_months} Months",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_xlim(0, n_steps - 1)
    ax1.grid(alpha=0.3)

    # Annotation box
    textstr = (f"Blowup rate: {result.blowup_rate*100:.1f}%\n"
               f"Median equity: ${result.median_equity:,.0f}\n"
               f"5th pct: ${result.pct_5:,.0f}\n"
               f"Max DD (p95): {result.max_dd_p95*100:.1f}%")
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top",
             bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7})

    # Drawdown distribution
    ax2.hist(result.max_drawdowns * 100, bins=50, color="#EF5350", edgecolor="black",
             linewidth=0.4, density=True)
    ax2.axvline(result.max_dd_p95 * 100, color="darkred", linewidth=1.5,
                linestyle="--", label=f"p95 DD = {result.max_dd_p95*100:.1f}%")
    ax2.set_xlabel("Max Drawdown (%)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("Max Drawdown Distribution", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("MC paths plot saved to %s", save_path)

    return fig


# ===========================================================================
# 6. Regime-scenario batch runner
# ===========================================================================

def run_regime_scenario_grid(
        starting_equity: float = 1_000_000.0,
        n_months: int = 12,
        n_sims: int = 1_000,
        regimes_override: Optional[List[str]] = None,
        seed: int = 42) -> pd.DataFrame:
    """
    Run the simulator once for each dominant-regime scenario and compare results.

    Creates a custom transition matrix that forces 80% time in each regime.

    Parameters
    ----------
    starting_equity : initial equity
    n_months        : months per path
    n_sims          : simulations per scenario
    regimes_override: subset of regimes to test (default all four)
    seed            : RNG seed

    Returns
    -------
    pd.DataFrame with one row per regime scenario
    """
    target_regimes = regimes_override or list(REGIMES)
    K = len(REGIMES)
    rows = []

    for dominant in target_regimes:
        idx = list(REGIMES).index(dominant)
        # Build transition matrix biased toward *dominant*
        P = np.ones((K, K)) * 0.05 / (K - 1)
        for i in range(K):
            P[i, i]   = 0.80
            others    = [j for j in range(K) if j != i]
            for j in others:
                P[i, j] = 0.20 / len(others)
        # Override: all rows heavily transition into *dominant*
        for i in range(K):
            if i != idx:
                P[i] = np.ones(K) * 0.05 / (K - 1)
                P[i, i]   = 0.15
                P[i, idx] = 0.80

        sim = RegimeMCSim(transition_matrix=P, n_sims=n_sims)
        res = sim.run(starting_equity=starting_equity, n_months=n_months,
                      seed=seed + list(REGIMES).index(dominant))

        rows.append({
            "dominant_regime":   dominant,
            "blowup_rate_pct":   round(res.blowup_rate * 100, 2),
            "median_equity":     round(res.median_equity, 0),
            "pct_5_equity":      round(res.pct_5, 0),
            "pct_95_equity":     round(res.pct_95, 0),
            "max_dd_p95_pct":    round(res.max_dd_p95 * 100, 2),
            "sharpe":            round(res.sharpe_ratio(), 3),
        })

    return pd.DataFrame(rows)
