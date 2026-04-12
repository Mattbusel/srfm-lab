"""
Experiment: Population Stability

Run populations with varying MM/momentum/noise ratios and measure stability.
Plot phase diagram: (MM_fraction, momentum_fraction) → market_volatility.
"""

from __future__ import annotations

import sys
import os
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyper_agent.env_compat import make_env
from hyper_agent.agents.noise_trader import NoiseTrader
from hyper_agent.agents.momentum_agent import MomentumAgent
from hyper_agent.agents.market_maker_agent import MarketMakerAgent
from hyper_agent.analysis.stability import StabilityAnalyzer
from hyper_agent.analysis.market_quality import MarketQualityMetrics


# ============================================================
# Configuration
# ============================================================

TOTAL_AGENTS  = 20
N_EPISODES    = 30
STEPS_PER_EP  = 200
OBS_DIM       = 23
SEED          = 42

# Grid: fraction of MMs and momentum agents (rest = noise)
MM_FRACS   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MOM_FRACS  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


# ============================================================
# Single run
# ============================================================

def run_single_config(
    mm_frac:   float,
    mom_frac:  float,
    n_episodes: int = N_EPISODES,
    steps:     int = STEPS_PER_EP,
    seed:      int = SEED,
) -> Dict[str, float]:
    """Run one (mm_frac, mom_frac) configuration and return stability metrics."""
    n_mm   = max(1, int(TOTAL_AGENTS * mm_frac))
    n_mom  = max(1, int(TOTAL_AGENTS * mom_frac))
    n_arb  = 0
    n_noise = max(0, TOTAL_AGENTS - n_mm - n_mom)

    try:
        env = make_env(
            n_market_makers = n_mm,
            n_momentum      = n_mom,
            n_arbitrage     = n_arb,
            n_noise         = n_noise,
            max_steps       = steps,
            seed            = seed,
        )
    except Exception:
        return {"volatility": 1.0, "lyapunov": 0.0, "mean_spread": 0.1}

    agent_ids  = env.agent_ids
    stability  = StabilityAnalyzer(agent_ids)
    mq         = MarketQualityMetrics()

    all_vols: List[float] = []
    all_spreads: List[float] = []
    prev_price: Optional[float] = None

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for t in range(steps):
            actions = {}
            for aid in agent_ids:
                # Random actions for this sweep (no learning — we want structure only)
                actions[aid] = np.array(
                    [np.random.uniform(-1, 1) for _ in range(3)] + [np.random.uniform(0, 1)],
                    dtype=np.float32,
                )
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            if info:
                fi         = next(iter(info.values()))
                mid        = fi.get("mid_price", 100.0)
                spread     = fi.get("spread", 0.02)
                volume     = fi.get("volume", 1.0)
                agent_acts = {a: 1 for a in agent_ids}  # placeholder
                stability.step(mid, agent_acts, mm_fraction=mm_frac, mom_fraction=mom_frac)
                mq.step(
                    mid_price      = mid,
                    exec_price     = fi.get("exec_price", mid),
                    spread         = spread,
                    volume         = volume,
                    net_order_flow = 0.0,
                    prev_price     = prev_price,
                )
                if prev_price is not None and prev_price > 0:
                    ret = abs(np.log(mid / prev_price))
                    all_vols.append(ret)
                all_spreads.append(spread)
                prev_price = mid

            obs = next_obs
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

    return {
        "volatility":   float(np.mean(all_vols))   if all_vols   else 0.01,
        "mean_spread":  float(np.mean(all_spreads)) if all_spreads else 0.05,
        "lyapunov":     stability.lyapunov.estimate_lle(),
        "efficiency":   mq.efficiency.efficiency_score(),
    }


# ============================================================
# Phase diagram sweep
# ============================================================

def run_phase_diagram_sweep() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep (MM_frac, mom_frac) grid and return volatility surface.

    Returns:
        mm_grid:  (n_mm, n_mom) array of MM fractions
        mom_grid: (n_mm, n_mom) array of momentum fractions
        vol_grid: (n_mm, n_mom) array of volatilities
    """
    n_mm  = len(MM_FRACS)
    n_mom = len(MOM_FRACS)

    vol_grid = np.zeros((n_mm, n_mom))
    lle_grid = np.zeros((n_mm, n_mom))

    for i, mm_f in enumerate(MM_FRACS):
        for j, mom_f in enumerate(MOM_FRACS):
            if mm_f + mom_f > 1.0:
                vol_grid[i, j] = np.nan
                lle_grid[i, j] = np.nan
                continue
            print(f"  mm_frac={mm_f:.1f}, mom_frac={mom_f:.1f} ...", end=" ", flush=True)
            result = run_single_config(mm_f, mom_f)
            vol_grid[i, j] = result["volatility"]
            lle_grid[i, j] = result["lyapunov"]
            print(f"vol={result['volatility']:.4f}")

    mm_grid  = np.array(MM_FRACS)
    mom_grid = np.array(MOM_FRACS)
    return mm_grid, mom_grid, vol_grid, lle_grid


# ============================================================
# Plotting
# ============================================================

def plot_phase_diagram(
    mm_grid:  np.ndarray,
    mom_grid: np.ndarray,
    vol_grid: np.ndarray,
    lle_grid: np.ndarray,
    output_dir: str = ".",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    MM, MOM = np.meshgrid(mm_grid, mom_grid, indexing="ij")

    # Volatility surface
    ax = axes[0]
    vg = np.nan_to_num(vol_grid, nan=0.0)
    c  = ax.contourf(MM, MOM, vg, levels=20, cmap="RdYlGn_r")
    fig.colorbar(c, ax=ax, label="Volatility")
    ax.set_xlabel("Market Maker Fraction")
    ax.set_ylabel("Momentum Fraction")
    ax.set_title("Phase Diagram: Market Volatility")
    ax.contour(MM, MOM, vg, levels=5, colors="black", alpha=0.3)

    # Lyapunov surface
    ax = axes[1]
    lg = np.nan_to_num(lle_grid, nan=0.0)
    c2 = ax.contourf(MM, MOM, lg, levels=20, cmap="RdBu_r")
    fig.colorbar(c2, ax=ax, label="Lyapunov Exponent")
    ax.axhspan(0, 0, color="k", linestyle="--")
    ax.set_xlabel("Market Maker Fraction")
    ax.set_ylabel("Momentum Fraction")
    ax.set_title("Phase Diagram: Lyapunov Exponent (+ = unstable)")
    ax.contour(MM, MOM, lg, levels=[0.0], colors="red")

    plt.suptitle("AETERNUS Hyper-Agent — Stability Phase Diagram")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_phase_diagram.png"), dpi=120)
    plt.close()
    print(f"Saved stability_phase_diagram.png to {output_dir}")


def plot_volatility_vs_mm(
    mm_fracs: List[float],
    results_by_mm: List[float],
    output_dir: str = ".",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mm_fracs, results_by_mm, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Market Maker Fraction")
    ax.set_ylabel("Mean Volatility")
    ax.set_title("Volatility vs Market Maker Population Fraction")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "volatility_vs_mm_fraction.png"), dpi=120)
    plt.close()
    print(f"Saved volatility_vs_mm_fraction.png to {output_dir}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== AETERNUS: Experiment — Market Stability Phase Diagram ===")
    print(f"Grid: {len(MM_FRACS)} x {len(MOM_FRACS)} configurations")
    print(f"Each config: {N_EPISODES} episodes x {STEPS_PER_EP} steps\n")

    mm_grid, mom_grid, vol_grid, lle_grid = run_phase_diagram_sweep()

    # Summary stats
    valid_vols = vol_grid[~np.isnan(vol_grid)]
    print(f"\n=== Summary ===")
    print(f"Mean volatility: {np.mean(valid_vols):.4f}")
    print(f"Min  volatility: {np.min(valid_vols):.4f}")
    print(f"Max  volatility: {np.max(valid_vols):.4f}")

    # Find most stable configuration
    best_idx = np.unravel_index(np.nanargmin(vol_grid), vol_grid.shape)
    best_mm  = MM_FRACS[best_idx[0]]
    best_mom = MOM_FRACS[best_idx[1]]
    print(f"Most stable config: MM={best_mm:.1f}, Mom={best_mom:.1f} "
          f"(vol={vol_grid[best_idx]:.4f})")

    # 1D slice: vol vs MM fraction (fixed mom=0.2)
    if len(MOM_FRACS) > 1:
        mid_mom_idx = len(MOM_FRACS) // 2
        vols_vs_mm  = vol_grid[:, mid_mom_idx].tolist()
        plot_volatility_vs_mm(MM_FRACS, vols_vs_mm, output_dir)

    plot_phase_diagram(mm_grid, mom_grid, vol_grid, lle_grid, output_dir)

    print("\nExperiment complete.")
    return {
        "vol_grid": vol_grid.tolist(),
        "lle_grid": lle_grid.tolist(),
        "best_config": {"mm_frac": best_mm, "mom_frac": best_mom},
    }


if __name__ == "__main__":
    main()
