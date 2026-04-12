"""
Experiment: Crisis

Inject crisis, observe how agent population responds.
Does herding behavior emerge? Do MMs withdraw? Does price discovery break?
Compare with and without communication between agents.
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

from hyper_agent.env_compat import make_env, MultiAgentTradingEnv
from hyper_agent.analysis.emergence import EmergenceAnalyzer, FlashCrashDetector
from hyper_agent.analysis.stability import StabilityAnalyzer
from hyper_agent.analysis.market_quality import MarketQualityMetrics


# ============================================================
# Config
# ============================================================

N_STEPS       = 500
CRISIS_STEP   = 200
N_MARKET_MAKERS = 5
N_MOMENTUM    = 5
N_NOISE       = 10
SEED          = 42


# ============================================================
# Single crisis run
# ============================================================

def run_crisis_experiment(
    with_crisis:     bool   = True,
    crisis_step:     int    = CRISIS_STEP,
    crisis_vol_mult: float  = 5.0,
    n_mms:           int    = N_MARKET_MAKERS,
    n_mom:           int    = N_MOMENTUM,
    n_noise:         int    = N_NOISE,
    steps:           int    = N_STEPS,
    seed:            int    = SEED,
) -> Dict:
    """
    Run one episode with or without crisis injection.

    Returns detailed time-series of market quality metrics.
    """
    env = make_env(
        n_market_makers       = n_mms,
        n_momentum            = n_mom,
        n_arbitrage           = 2,
        n_noise               = n_noise,
        max_steps             = steps,
        crisis_step           = crisis_step if with_crisis else None,
        crisis_vol_multiplier = crisis_vol_mult,
        seed                  = seed,
    )
    agent_ids = env.agent_ids
    emergence = EmergenceAnalyzer(agent_ids)
    stability = StabilityAnalyzer(agent_ids)
    mq        = MarketQualityMetrics()

    # Time series to record
    ts: Dict[str, List] = {
        "prices":       [],
        "spreads":      [],
        "herding":      [],
        "volatility":   [],
        "mm_activity":  [],  # fraction of MM agents trading
        "buy_pressure": [],
        "in_crisis":    [],
        "discovery":    [],
        "efficiency":   [],
        "n_crashes":    [],
    }

    prev_price: Optional[float] = None

    obs, _ = env.reset()
    for t in range(steps):
        actions = {}
        for aid in agent_ids:
            # Noise-like random actions to isolate structural effects
            logits = np.random.uniform(-2, 2, size=3).astype(np.float32)
            size   = np.random.uniform(0, 1)
            actions[aid] = np.array([*logits, size], dtype=np.float32)

        next_obs, rewards, terminated, truncated, info = env.step(actions)

        if info:
            fi       = next(iter(info.values()))
            mid      = fi.get("mid_price", 100.0)
            spread   = fi.get("spread", 0.02)
            exec_p   = fi.get("exec_price", mid)
            volume   = fi.get("volume", 1.0)
            in_crisis= fi.get("in_crisis", False)

            # Agent actions (decode from action arrays)
            agent_acts: Dict[str, int] = {}
            for aid, act in actions.items():
                logits_ = act[:3]
                dir_    = int(np.argmax(logits_)) - 1
                agent_acts[aid] = dir_

            # MM activity: fraction of MM agents with non-zero actions
            mm_agents = [a for a in agent_ids if a.startswith("mm_")]
            mm_active = sum(
                1 for a in mm_agents if abs(agent_acts.get(a, 0)) > 0
            )
            mm_act_frac = mm_active / max(len(mm_agents), 1)

            # Emergence metrics
            em = emergence.step(mid, spread, agent_acts, rewards,
                                n_active_mms=len(mm_agents))
            st = stability.step(mid, agent_acts)

            # Market quality
            mq_m = mq.step(
                mid_price      = mid,
                exec_price     = exec_p,
                spread         = spread,
                volume         = volume,
                net_order_flow = fi.get("price_impact", 0.0),
                prev_price     = prev_price,
            )

            ts["prices"].append(mid)
            ts["spreads"].append(spread)
            ts["herding"].append(em["herding_index"])
            ts["volatility"].append(st["volatility"])
            ts["mm_activity"].append(mm_act_frac)
            ts["buy_pressure"].append(st["buy_pressure"])
            ts["in_crisis"].append(1.0 if in_crisis else 0.0)
            ts["discovery"].append(em["discovery_rate"])
            ts["efficiency"].append(mq_m["efficiency_score"])
            ts["n_crashes"].append(em["n_crashes"])
            prev_price = mid

        obs = next_obs
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            break

    # Summary stats: pre-crisis vs during-crisis
    crisis_start = crisis_step if with_crisis else len(ts["prices"])
    pre   = slice(0, crisis_start)
    post  = slice(crisis_start, len(ts["prices"]))

    def mean_slice(lst: List, s: slice) -> float:
        sub = lst[s]
        return float(np.mean(sub)) if sub else 0.0

    summary = {
        "with_crisis":         with_crisis,
        "pre_crisis_spread":   mean_slice(ts["spreads"],    pre),
        "post_crisis_spread":  mean_slice(ts["spreads"],    post),
        "pre_crisis_herding":  mean_slice(ts["herding"],    pre),
        "post_crisis_herding": mean_slice(ts["herding"],    post),
        "pre_crisis_mm_act":   mean_slice(ts["mm_activity"],pre),
        "post_crisis_mm_act":  mean_slice(ts["mm_activity"],post),
        "pre_disc_rate":       mean_slice(ts["discovery"],  pre),
        "post_disc_rate":      mean_slice(ts["discovery"],  post),
        "n_flash_crashes":     ts["n_crashes"][-1] if ts["n_crashes"] else 0,
        "timeseries":          ts,
    }
    return summary


# ============================================================
# Plotting
# ============================================================

def plot_crisis_comparison(
    crisis_run:     Dict,
    no_crisis_run:  Dict,
    crisis_step:    int   = CRISIS_STEP,
    output_dir:     str   = ".",
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    t_ax = np.arange(len(crisis_run["timeseries"]["prices"]))

    def plot_metric(ax, key, label, ylabel):
        c_vals = crisis_run["timeseries"][key]
        nc_vals = no_crisis_run["timeseries"].get(key, [0.0] * len(t_ax))
        ax.plot(t_ax[:len(c_vals)], c_vals,  color="crimson",  label="With Crisis",    alpha=0.9)
        ax.plot(t_ax[:len(nc_vals)], nc_vals, color="steelblue", label="No Crisis",      alpha=0.7)
        if len(crisis_run["timeseries"].get("in_crisis", [])) > 0:
            ic = np.array(crisis_run["timeseries"]["in_crisis"])
            ax.fill_between(t_ax[:len(ic)], 0, ic * max(max(c_vals, default=0), 0.01),
                            alpha=0.15, color="red", label="Crisis Zone")
        ax.axvline(crisis_step, color="red", linestyle="--", alpha=0.5, label="Crisis Start")
        ax.set_title(label)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plot_metric(axes[0, 0], "prices",      "Mid Price",                   "Price")
    plot_metric(axes[0, 1], "spreads",     "Bid-Ask Spread",              "Spread")
    plot_metric(axes[1, 0], "herding",     "Herding Index",               "H-Index")
    plot_metric(axes[1, 1], "mm_activity", "Market Maker Activity",       "MM Active Fraction")
    plot_metric(axes[2, 0], "discovery",   "Price Discovery Rate",        "Discovery λ")
    plot_metric(axes[2, 1], "efficiency",  "Informational Efficiency",    "Score [0,1]")

    for ax in axes.flat:
        ax.set_xlabel("Step")

    plt.suptitle("AETERNUS: Crisis Injection — Market Response Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "crisis_analysis.png"), dpi=120)
    plt.close()
    print("Saved crisis_analysis.png")


def print_crisis_summary(crisis_run: Dict, no_crisis_run: Dict) -> None:
    print("\n=== Crisis Impact Summary ===")
    metrics = [
        ("Spread",         "pre_crisis_spread",   "post_crisis_spread",   True),
        ("Herding",        "pre_crisis_herding",  "post_crisis_herding",  True),
        ("MM Activity",    "pre_crisis_mm_act",   "post_crisis_mm_act",   False),
        ("Discovery Rate", "pre_disc_rate",        "post_disc_rate",       False),
    ]
    print(f"{'Metric':>20} | {'Pre-Crisis':>12} | {'Post-Crisis':>12} | {'Change':>12}")
    print("-" * 65)
    for label, pre_k, post_k, higher_bad in metrics:
        pre  = crisis_run.get(pre_k, 0.0)
        post = crisis_run.get(post_k, 0.0)
        chg  = (post - pre) / (abs(pre) + 1e-8) * 100
        sign = "↑" if chg > 0 else "↓"
        print(f"{label:>20} | {pre:>12.5f} | {post:>12.5f} | {sign}{abs(chg):>10.1f}%")

    print(f"\nFlash crashes detected: {crisis_run.get('n_flash_crashes', 0)}")
    print(f"No-crisis flash crashes: {no_crisis_run.get('n_flash_crashes', 0)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== AETERNUS: Experiment — Crisis Injection ===\n")

    print("[1] Running with crisis injection...")
    crisis_run = run_crisis_experiment(with_crisis=True)

    print("[2] Running baseline (no crisis)...")
    no_crisis_run = run_crisis_experiment(with_crisis=False)

    print("[3] Plotting comparison...")
    plot_crisis_comparison(crisis_run, no_crisis_run, CRISIS_STEP, output_dir)

    print_crisis_summary(crisis_run, no_crisis_run)

    # Additional analysis: spread amplification
    pre_sp  = crisis_run["pre_crisis_spread"]
    post_sp = crisis_run["post_crisis_spread"]
    if pre_sp > 0:
        amp = post_sp / pre_sp
        print(f"\nSpread amplification during crisis: {amp:.2f}x")
        if amp > 2.0:
            print("  → SIGNIFICANT spread widening detected")
        if amp > 5.0:
            print("  → SEVERE liquidity impairment!")

    # Herding during crisis
    pre_h  = crisis_run["pre_crisis_herding"]
    post_h = crisis_run["post_crisis_herding"]
    if pre_h >= 0:
        print(f"Herding during crisis: {post_h:.4f} (vs {pre_h:.4f} pre-crisis)")
        if post_h > pre_h * 2:
            print("  → HERDING BEHAVIOR DETECTED during crisis")

    print("\nCrisis experiment complete.")


if __name__ == "__main__":
    main()
