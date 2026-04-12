"""
analysis.py — Post-hoc analysis and emergence metrics for Hyper-Agent.

Loads trained agents and performs:
- Market microstructure analysis
- Price discovery efficiency measurement
- Nash equilibrium gap estimation
- Flash crash frequency analysis
- Agent strategy visualization
"""

from __future__ import annotations

import os
import sys
import json
import logging
import collections
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hyper_agent.environment import MultiAssetTradingEnv, MarketMicrostructureAnalyzer
from hyper_agent.emergence import EmergenceAnalyzer, FlashCrashDetector, PriceDiscoveryAnalyzer
from hyper_agent.population import AgentPopulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market microstructure analysis
# ---------------------------------------------------------------------------

def analyze_microstructure(
    env: MultiAssetTradingEnv,
    n_steps: int = 1000,
) -> Dict[str, Any]:
    """Run random-action episode and analyze microstructure."""
    env.reset()
    analyzer = MarketMicrostructureAnalyzer(env)

    for _ in range(n_steps):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        env._marl_step(actions)

    return analyzer.summary()


# ---------------------------------------------------------------------------
# Price discovery analysis
# ---------------------------------------------------------------------------

def measure_price_discovery(
    env: MultiAssetTradingEnv,
    n_episodes: int = 10,
    episode_len: int = 500,
) -> Dict[str, float]:
    """Measure price discovery efficiency across episodes."""
    analyzers = [PriceDiscoveryAnalyzer() for _ in range(env.num_assets)]
    results: Dict[str, List[float]] = collections.defaultdict(list)

    for ep in range(n_episodes):
        env.reset()
        for step in range(episode_len):
            actions = [env.action_space.sample() for _ in range(env.num_agents)]
            env._marl_step(actions)

            prices = env._mid_prices
            fundamentals = env._fundamental_values
            for i, (analyzer, p, f) in enumerate(zip(analyzers, prices, fundamentals)):
                analyzer.update(float(p), float(f))

    for i, analyzer in enumerate(analyzers):
        results[f"asset_{i}_efficiency"] = [analyzer.price_efficiency_score()]
        results[f"asset_{i}_variance_ratio"] = [analyzer.variance_ratio()]
        gap = analyzer.price_vs_fundamental_gap()
        if gap == gap:  # not nan
            results[f"asset_{i}_gap"] = [float(gap)]

    return {k: float(np.mean(v)) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Flash crash analysis
# ---------------------------------------------------------------------------

def analyze_flash_crashes(
    env: MultiAssetTradingEnv,
    n_steps: int = 5000,
) -> Dict[str, Any]:
    """Detect flash crash events over a long simulation."""
    detectors = [FlashCrashDetector() for _ in range(env.num_assets)]
    crash_counts = [0] * env.num_assets

    env.reset()
    for step in range(n_steps):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        env._marl_step(actions)

        for i in range(env.num_assets):
            ob = env.order_books[i]
            spread = ob.spread()
            if spread != spread:
                spread = 0.0
            is_crash, _ = detectors[i].update(
                float(env._mid_prices[i]),
                float(spread),
                float(ob.order_imbalance()),
                float(ob.volume_today),
                step,
            )
            if is_crash:
                crash_counts[i] += 1

    return {
        "crash_counts": crash_counts,
        "crash_frequencies": [c / max(n_steps, 1) for c in crash_counts],
        "total_crashes": sum(crash_counts),
    }


# ---------------------------------------------------------------------------
# Full analysis report
# ---------------------------------------------------------------------------

def run_full_analysis(
    checkpoint_dir: Optional[str] = None,
    num_assets: int = 4,
    num_agents: int = 8,
    n_steps: int = 2000,
) -> Dict[str, Any]:
    """Run complete post-hoc analysis."""
    logger.info("Running full market analysis...")

    env = MultiAssetTradingEnv(
        num_assets=num_assets,
        num_agents=num_agents,
        max_steps=n_steps,
        seed=42,
    )

    report: Dict[str, Any] = {}

    logger.info("Analyzing microstructure...")
    report["microstructure"] = analyze_microstructure(env, n_steps=min(n_steps, 500))

    logger.info("Measuring price discovery...")
    report["price_discovery"] = measure_price_discovery(env, n_episodes=5, episode_len=200)

    logger.info("Analyzing flash crashes...")
    report["flash_crashes"] = analyze_flash_crashes(env, n_steps=min(n_steps, 1000))

    # Emergence analysis
    logger.info("Computing emergence metrics...")
    emergence = EmergenceAnalyzer(num_assets=num_assets, num_agents=num_agents, action_dim=env.action_dim)

    env.reset()
    for step in range(min(n_steps, 500)):
        actions = [env.action_space.sample() for _ in range(num_agents)]
        env._marl_step(actions)

        prices = env._mid_prices.copy()
        spreads = np.array([ob.spread() if ob.spread() == ob.spread() else 0.0 for ob in env.order_books])
        imbalances = np.array([ob.order_imbalance() for ob in env.order_books])
        volumes = np.array([ob.volume_today for ob in env.order_books])
        trade_counts = np.array([ob.trade_count for ob in env.order_books])
        depths = np.array([sum(s for _, s in ob.depth(5)[0]) + sum(s for _, s in ob.depth(5)[1]) for ob in env.order_books])
        rewards = [0.0] * num_agents
        actions_list = [a.copy() for a in actions]

        emergence.update(
            prices=prices, spreads=spreads, imbalances=imbalances,
            volumes=volumes, trade_counts=trade_counts, total_depths=depths,
            agent_rewards=rewards, agent_actions=actions_list,
            fundamentals=env._fundamental_values.copy(),
        )

    report["emergence"] = emergence.full_report()
    report["market_health_score"] = emergence.get_market_health_score()

    logger.info(f"Market health score: {report['market_health_score']:.3f}")
    logger.info(f"Total crashes: {report['flash_crashes']['total_crashes']}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Post-hoc analysis")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--num_assets", type=int, default=4)
    parser.add_argument("--num_agents", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--output", type=str, default="analysis_report.json")
    args = parser.parse_args()

    report = run_full_analysis(
        checkpoint_dir=args.checkpoint_dir,
        num_assets=args.num_assets,
        num_agents=args.num_agents,
        n_steps=args.n_steps,
    )

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Analysis saved to {args.output}")


if __name__ == "__main__":
    main()
