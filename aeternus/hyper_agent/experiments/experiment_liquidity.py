"""
Experiment: Liquidity

Vary number of market makers, plot bid-ask spread vs MM count.
Show optimal MM population size.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyper_agent.env_compat import make_env
from hyper_agent.analysis.market_quality import MarketQualityMetrics


# ============================================================
# Config
# ============================================================

N_EPISODES   = 25
STEPS_EP     = 200
SEED         = 42
N_NOISE      = 8
MM_COUNTS    = list(range(0, 16))  # 0 to 15 market makers


# ============================================================
# Single configuration run
# ============================================================

def measure_liquidity(
    n_mms:      int,
    n_noise:    int = N_NOISE,
    n_episodes: int = N_EPISODES,
    steps:      int = STEPS_EP,
    seed:       int = SEED,
) -> Dict[str, float]:
    """
    Measure liquidity metrics for a given MM count.
    """
    env = make_env(
        n_market_makers = max(0, n_mms),
        n_momentum      = 0,
        n_arbitrage     = 0,
        n_noise         = max(1, n_noise),
        max_steps       = steps,
        seed             = seed,
    )
    agent_ids = env.agent_ids
    mq        = MarketQualityMetrics()

    spreads   : List[float] = []
    vwas_vals : List[float] = []
    depths    : List[float] = []
    impacts   : List[float] = []
    prev_price: Optional[float] = None
    all_flows : List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_vol = 0.0
        for t in range(steps):
            # Use random actions to stress-test liquidity without learning
            actions = {
                aid: np.array(
                    [np.random.uniform(-1, 1),
                     np.random.uniform(-1, 1),
                     np.random.uniform(-1, 1),
                     np.random.uniform(0, 0.5)],
                    dtype=np.float32,
                )
                for aid in agent_ids
            }
            _, rewards, terminated, truncated, info = env.step(actions)

            if info:
                fi     = next(iter(info.values()))
                mid    = fi.get("mid_price", 100.0)
                spread = fi.get("spread", 0.02)
                exec_p = fi.get("exec_price", mid)
                volume = fi.get("volume", 1.0)
                impact = fi.get("price_impact", 0.0)
                bid_v  = env.env.lob.total_bid_volume() if hasattr(env.env, "lob") else 0.0
                ask_v  = env.env.lob.total_ask_volume() if hasattr(env.env, "lob") else 0.0

                all_flows.append(impact)
                mq_metrics = mq.step(
                    mid_price      = mid,
                    exec_price     = exec_p,
                    spread         = spread,
                    volume         = volume,
                    net_order_flow = impact,
                    bid_volume     = bid_v,
                    ask_volume     = ask_v,
                    prev_price     = prev_price,
                )
                spreads.append(spread)
                vwas_vals.append(mq_metrics["vwas"])
                depths.append(mq_metrics["mean_depth"])
                impacts.append(abs(impact))
                ep_vol += volume
                prev_price = mid

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

    return {
        "n_mms":         n_mms,
        "mean_spread":   float(np.mean(spreads))   if spreads   else 0.05,
        "std_spread":    float(np.std(spreads))    if spreads   else 0.0,
        "vwas":          float(np.mean(vwas_vals)) if vwas_vals else 0.05,
        "mean_depth":    float(np.mean(depths))    if depths    else 0.0,
        "kyle_lambda":   float(mq.price_impact.kyle_lambda()),
        "efficiency":    float(mq.efficiency.efficiency_score()),
        "resilience":    float(mq.resilience.resilience_score()),
    }


# ============================================================
# Sweep
# ============================================================

def sweep_liquidity() -> List[Dict]:
    results = []
    for n_mm in MM_COUNTS:
        print(f"  n_mm={n_mm:2d} ...", end=" ", flush=True)
        r = measure_liquidity(n_mm)
        results.append(r)
        print(f"spread={r['mean_spread']:.5f} vwas={r['vwas']:.5f} "
              f"depth={r['mean_depth']:.2f}")
    return results


# ============================================================
# Fitting
# ============================================================

def power_law_fit(n: np.ndarray, a: float, b: float) -> np.ndarray:
    """f(n) = a * n^b  (expect b < 0)"""
    return a * np.power(np.maximum(n, 0.01), b)


def fit_spread_curve(
    n_mms: np.ndarray, spreads: np.ndarray
) -> Tuple[float, float]:
    """Fit power law to spread vs MM count. Returns (a, b)."""
    try:
        popt, _ = curve_fit(
            power_law_fit, n_mms[n_mms > 0], spreads[n_mms > 0],
            p0=[0.1, -0.5], maxfev=1000,
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        return 0.1, -0.5


# ============================================================
# Plotting
# ============================================================

def plot_liquidity_results(
    results:    List[Dict],
    output_dir: str = ".",
) -> None:
    n_arr    = np.array([r["n_mms"]       for r in results])
    sp_arr   = np.array([r["mean_spread"] for r in results])
    sp_std   = np.array([r["std_spread"]  for r in results])
    vwas_arr = np.array([r["vwas"]        for r in results])
    dep_arr  = np.array([r["mean_depth"]  for r in results])
    eff_arr  = np.array([r["efficiency"]  for r in results])
    ky_arr   = np.array([r["kyle_lambda"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Spread vs MM count
    ax = axes[0, 0]
    ax.plot(n_arr, sp_arr, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.fill_between(n_arr, sp_arr - sp_std, sp_arr + sp_std, alpha=0.2, color="steelblue")

    # Power law fit
    valid  = n_arr > 0
    if valid.sum() > 2:
        a, b = fit_spread_curve(n_arr, sp_arr)
        n_fit = np.linspace(1, max(n_arr), 100)
        ax.plot(n_fit, power_law_fit(n_fit, a, b), "--r",
                label=f"fit: {a:.3f}×N^{b:.2f}", alpha=0.8)
        ax.legend(fontsize=9)

    ax.set_xlabel("Number of Market Makers")
    ax.set_ylabel("Mean Bid-Ask Spread")
    ax.set_title("Spread vs Market Maker Count")
    ax.grid(True, alpha=0.3)
    ax.axhline(sp_arr[0] if len(sp_arr) > 0 else 0.05, color="red",
               linestyle=":", alpha=0.5, label="0 MMs baseline")

    # 2. Market depth
    ax = axes[0, 1]
    ax.bar(n_arr, dep_arr, color="mediumseagreen", alpha=0.8)
    ax.set_xlabel("Number of Market Makers")
    ax.set_ylabel("Mean Market Depth (quoted volume)")
    ax.set_title("Market Depth vs MM Count")
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Informational efficiency
    ax = axes[1, 0]
    ax.plot(n_arr, eff_arr, "s-", color="darkorange", linewidth=2)
    ax.set_xlabel("Number of Market Makers")
    ax.set_ylabel("Efficiency Score [0,1]")
    ax.set_title("Informational Efficiency vs MM Count")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # 4. Kyle lambda (price impact)
    ax = axes[1, 1]
    ax.plot(n_arr, ky_arr, "D-", color="purple", linewidth=2)
    ax.set_xlabel("Number of Market Makers")
    ax.set_ylabel("Kyle λ (Price Impact)")
    ax.set_title("Price Impact vs MM Count")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "AETERNUS Hyper-Agent — Liquidity vs Market Maker Population",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "liquidity_vs_mm.png"), dpi=120)
    plt.close()
    print(f"Saved liquidity_vs_mm.png")


def find_optimal_mm_count(results: List[Dict]) -> int:
    """
    Identify optimal MM count as elbow point in spread vs MM curve.

    Uses the point of maximum marginal spread reduction.
    """
    if len(results) < 3:
        return 1
    spreads = np.array([r["mean_spread"] for r in results])
    n_mms   = [r["n_mms"] for r in results]

    # Marginal reduction
    if len(spreads) < 2:
        return 1
    marginal = np.abs(np.diff(spreads))
    # Elbow: where marginal reduction is maximized (most bang per MM)
    idx     = int(np.argmax(marginal)) + 1
    return n_mms[min(idx, len(n_mms) - 1)]


# ============================================================
# Main
# ============================================================

def main() -> None:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== AETERNUS: Experiment — Liquidity ===\n")
    print(f"Sweeping MM counts: {MM_COUNTS}")
    print(f"N episodes: {N_EPISODES}, steps/episode: {STEPS_EP}\n")

    results = sweep_liquidity()

    optimal_n = find_optimal_mm_count(results)
    print(f"\n=== Analysis ===")
    print(f"Optimal MM count (elbow method): {optimal_n}")

    # Print sorted summary
    print(f"\n{'N_MM':>5} | {'Spread':>10} | {'VWAS':>10} | {'Depth':>8} | {'Eff':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['n_mms']:>5} | {r['mean_spread']:>10.5f} | "
              f"{r['vwas']:>10.5f} | {r['mean_depth']:>8.2f} | "
              f"{r['efficiency']:>8.4f}")

    plot_liquidity_results(results, output_dir)

    print("\nLiquidity experiment complete.")
    return results


if __name__ == "__main__":
    main()
