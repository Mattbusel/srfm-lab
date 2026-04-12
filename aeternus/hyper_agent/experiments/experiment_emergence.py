"""
Experiment: Emergence

Show that market bid-ask spread emerges from competition between MM agents.
Demonstrate price discovery improves with more informed agents.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyper_agent.env_compat import make_env
from hyper_agent.analysis.emergence import EmergenceAnalyzer
from hyper_agent.analysis.market_quality import MarketQualityMetrics


# ============================================================
# Config
# ============================================================

N_EPISODES = 20
STEPS_EP   = 300
SEED       = 42
OBS_DIM    = 23

# Vary number of MM agents
MM_COUNTS  = [0, 1, 2, 3, 5, 8, 10]
# Vary fraction of "informed" (momentum/arb) agents
INFORMED_FRACS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]


# ============================================================
# Run with N market makers
# ============================================================

def run_with_n_mms(
    n_mms:       int,
    n_noise:     int = 10,
    n_episodes:  int = N_EPISODES,
    steps:       int = STEPS_EP,
    seed:        int = SEED,
) -> Dict[str, List[float]]:
    """
    Run N_episodes episodes with given number of market makers.
    Returns time-series of spreads and prices.
    """
    n_mms   = max(0, n_mms)
    n_noise = max(1, n_noise)

    env = make_env(
        n_market_makers = n_mms,
        n_momentum      = 0,
        n_arbitrage     = 0,
        n_noise         = n_noise,
        max_steps       = steps,
        seed             = seed,
    )
    agent_ids = env.agent_ids
    emergence = EmergenceAnalyzer(agent_ids)
    mq        = MarketQualityMetrics()

    spreads   : List[float] = []
    prices    : List[float] = []
    disc_rates: List[float] = []
    prev_price: Optional[float] = None

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for t in range(steps):
            actions = {
                aid: np.array(
                    [np.random.uniform(-0.5, 0.5) for _ in range(3)] + [0.3],
                    dtype=np.float32,
                )
                for aid in agent_ids
            }
            _, rewards, terminated, truncated, info = env.step(actions)

            if info:
                fi     = next(iter(info.values()))
                mid    = fi.get("mid_price", 100.0)
                spread = fi.get("spread", 0.02)
                volume = fi.get("volume", 1.0)
                n_active_mms = sum(1 for a in agent_ids if a.startswith("mm_"))

                agent_acts = {a: int(np.sign(np.random.randn())) for a in agent_ids}
                metrics    = emergence.step(mid, spread, agent_acts, rewards, n_active_mms)
                mq_m       = mq.step(
                    mid, fi.get("exec_price", mid), spread, volume, 0.0,
                    prev_price=prev_price
                )

                spreads.append(spread)
                prices.append(mid)
                disc_rates.append(metrics["discovery_rate"])
                prev_price = mid

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

    return {
        "spreads":     spreads,
        "prices":      prices,
        "disc_rates":  disc_rates,
        "mean_spread": float(np.mean(spreads)) if spreads else 0.05,
        "discovery":   float(np.mean(disc_rates)) if disc_rates else 0.0,
    }


# ============================================================
# Spread emergence sweep
# ============================================================

def sweep_spread_vs_mms() -> Dict[int, float]:
    """Compute mean spread for each MM count."""
    results = {}
    for n_mm in MM_COUNTS:
        print(f"  n_mm={n_mm} ...", end=" ", flush=True)
        r = run_with_n_mms(n_mm)
        results[n_mm] = r["mean_spread"]
        print(f"mean_spread={r['mean_spread']:.5f}")
    return results


# ============================================================
# Plotting
# ============================================================

def plot_spread_vs_mms(
    spread_by_mm: Dict[int, float],
    output_dir: str = ".",
) -> None:
    n_mms   = sorted(spread_by_mm.keys())
    spreads = [spread_by_mm[n] for n in n_mms]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_mms, spreads, "o-", color="green", linewidth=2.5, markersize=8)
    ax.fill_between(n_mms, 0, spreads, alpha=0.15, color="green")
    ax.set_xlabel("Number of Market Makers")
    ax.set_ylabel("Mean Bid-Ask Spread")
    ax.set_title("Emergent Spread: Competition Among Market Makers")
    ax.grid(True, alpha=0.3)

    # Annotate theoretical curve (Cournot: spread ~ 1/N)
    if max(n_mms) > 0:
        n_arr = np.linspace(max(0.5, min(n_mms)), max(n_mms), 100)
        ref   = spreads[1] if spreads[1] > 0 else 0.05
        cournot = ref * (n_mms[1] if n_mms[1] > 0 else 1) / n_arr
        ax.plot(n_arr, cournot, "--", color="gray", label="Cournot 1/N", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spread_emergence.png"), dpi=120)
    plt.close()
    print(f"Saved spread_emergence.png")


def plot_price_discovery_timeseries(
    n_mm_list: List[int],
    output_dir: str = ".",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_mm_list)))

    for idx, n_mm in enumerate(n_mm_list[:4]):
        r     = run_with_n_mms(n_mm, n_episodes=5, steps=200)
        t     = np.arange(len(r["spreads"]))
        label = f"{n_mm} MMs"
        axes[0].plot(t, r["spreads"], alpha=0.7, label=label, color=colors[idx])
        axes[1].plot(t, r["disc_rates"], alpha=0.7, label=label, color=colors[idx])

    axes[0].set_title("Spread Over Time by MM Count")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Bid-Ask Spread")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Price Discovery Rate by MM Count")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Discovery Rate λ")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("AETERNUS: Emergence of Market Quality from Agent Competition")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "price_discovery_timeseries.png"), dpi=120)
    plt.close()
    print(f"Saved price_discovery_timeseries.png")


# ============================================================
# Main
# ============================================================

def main() -> None:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== AETERNUS: Experiment — Market Emergence ===\n")

    print("[1] Sweeping spread vs number of market makers...")
    spread_by_mm = sweep_spread_vs_mms()

    print("\n[2] Plotting spread emergence curve...")
    plot_spread_vs_mms(spread_by_mm, output_dir)

    print("\n[3] Plotting price discovery time-series...")
    plot_price_discovery_timeseries([0, 2, 5, 10], output_dir)

    print("\n=== Results Summary ===")
    for n_mm, spread in sorted(spread_by_mm.items()):
        pct_reduction = 0.0
        if n_mm > 0 and 0 in spread_by_mm and spread_by_mm[0] > 0:
            pct_reduction = (spread_by_mm[0] - spread) / spread_by_mm[0] * 100
        print(f"  n_mm={n_mm:3d}: spread={spread:.5f}  "
              f"(reduction vs 0 MMs: {pct_reduction:.1f}%)")

    print("\nEmergence experiment complete.")


if __name__ == "__main__":
    main()
