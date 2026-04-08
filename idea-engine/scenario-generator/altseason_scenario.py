"""
Altcoin season scenario generator — simulates altcoin rotation and outperformance cycles.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AltseasonScenario:
    name: str
    description: str
    btc_path: np.ndarray
    eth_path: np.ndarray
    alt_large_cap_path: np.ndarray
    alt_small_cap_path: np.ndarray
    dominance_path: np.ndarray     # BTC dominance (0-1)
    volume_path: np.ndarray        # Total crypto volume multiplier
    tags: list[str] = field(default_factory=list)


def classic_altseason_scenario(
    btc_move: float = 0.30,
    eth_multiplier: float = 2.5,
    alt_multiplier: float = 5.0,
    small_cap_multiplier: float = 10.0,
    duration_steps: int = 180,
    rng: Optional[np.random.Generator] = None,
) -> AltseasonScenario:
    """
    Classic altseason: BTC pumps first, then capital rotates to alts.
    Phase 1: BTC dominance rises
    Phase 2: ETH leads, BTC dominance flat
    Phase 3: Alt rotation, BTC dominance falls
    Phase 4: Small caps explode, then full correction
    """
    rng = rng or np.random.default_rng(42)
    n = duration_steps
    t = np.linspace(0, 1, n)

    # Phase boundaries
    p1_end = int(n * 0.25)
    p2_end = int(n * 0.50)
    p3_end = int(n * 0.75)

    def sigmoid(x, center, sharpness=10):
        return 1 / (1 + np.exp(-sharpness * (x - center)))

    # BTC: leads early, then laggard
    btc_return = btc_move * sigmoid(t, 0.1) * (1 - 0.3 * sigmoid(t, 0.5))
    btc_path = np.exp(btc_return + rng.normal(0, 0.015, n))

    # ETH: follows BTC then outperforms
    eth_return = btc_move * eth_multiplier * sigmoid(t, 0.25) * (1 - 0.2 * sigmoid(t, 0.7))
    eth_path = np.exp(eth_return + rng.normal(0, 0.025, n))

    # Large cap alts: explosive in mid-cycle
    alt_return = btc_move * alt_multiplier * sigmoid(t, 0.40) * (1 - 0.4 * sigmoid(t, 0.75))
    alt_large_cap_path = np.exp(alt_return + rng.normal(0, 0.04, n))

    # Small caps: last to pump, most volatile
    small_return = btc_move * small_cap_multiplier * sigmoid(t, 0.60) * (1 - 0.7 * sigmoid(t, 0.82))
    alt_small_cap_path = np.exp(small_return + rng.normal(0, 0.07, n))

    # BTC dominance: rises then falls
    dominance_path = 0.55 + 0.1 * sigmoid(t, 0.15) - 0.25 * sigmoid(t, 0.40)
    dominance_path = np.clip(dominance_path + rng.normal(0, 0.01, n), 0.25, 0.75)

    # Volume: grows throughout
    volume_path = 1.0 + 5.0 * sigmoid(t, 0.5) + rng.exponential(0.3, n)

    return AltseasonScenario(
        name="classic_altseason",
        description=f"Classic altseason: BTC+{btc_move:.0%}, ETH+{eth_multiplier:.0f}x, alts+{alt_multiplier:.0f}x",
        btc_path=btc_path,
        eth_path=eth_path,
        alt_large_cap_path=alt_large_cap_path,
        alt_small_cap_path=alt_small_cap_path,
        dominance_path=dominance_path,
        volume_path=volume_path,
        tags=["altseason", "rotation", "cycle"],
    )


def eth_led_cycle_scenario(
    duration_steps: int = 120,
    rng: Optional[np.random.Generator] = None,
) -> AltseasonScenario:
    """
    ETH-led cycle: ETH ecosystem tokens outperform before BTC confirmation.
    Typical: DeFi summer, NFT season.
    """
    rng = rng or np.random.default_rng(42)
    n = duration_steps
    t = np.linspace(0, 1, n)

    def sigmoid(x, center, s=10):
        return 1 / (1 + np.exp(-s * (x - center)))

    eth_path = np.exp(1.5 * sigmoid(t, 0.2) + rng.normal(0, 0.03, n))
    btc_path = np.exp(0.4 * sigmoid(t, 0.4) + rng.normal(0, 0.015, n))
    alt_large_cap_path = np.exp(3.0 * sigmoid(t, 0.35) * (1 - 0.5 * sigmoid(t, 0.75)) + rng.normal(0, 0.05, n))
    alt_small_cap_path = np.exp(6.0 * sigmoid(t, 0.50) * (1 - 0.8 * sigmoid(t, 0.80)) + rng.normal(0, 0.08, n))
    dominance_path = np.clip(0.50 - 0.20 * sigmoid(t, 0.30) + rng.normal(0, 0.01, n), 0.25, 0.65)
    volume_path = 1 + 4 * sigmoid(t, 0.40) + rng.exponential(0.2, n)

    return AltseasonScenario(
        name="eth_led_cycle",
        description="ETH-led cycle: ETH ecosystem leads before BTC confirms",
        btc_path=btc_path,
        eth_path=eth_path,
        alt_large_cap_path=alt_large_cap_path,
        alt_small_cap_path=alt_small_cap_path,
        dominance_path=dominance_path,
        volume_path=volume_path,
        tags=["eth_season", "defi", "rotation"],
    )


def failed_altseason_scenario(
    duration_steps: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> AltseasonScenario:
    """
    Failed altseason: capital rotates to alts briefly, then BTC dominance reasserts.
    Alts dump harder than BTC. Common in bear market rallies.
    """
    rng = rng or np.random.default_rng(42)
    n = duration_steps
    t = np.linspace(0, 1, n)

    def sigmoid(x, center, s=10):
        return 1 / (1 + np.exp(-s * (x - center)))

    btc_path = np.exp(0.15 * sigmoid(t, 0.2) - 0.1 * sigmoid(t, 0.6) + rng.normal(0, 0.015, n))
    eth_path = np.exp(0.3 * sigmoid(t, 0.25) - 0.35 * sigmoid(t, 0.55) + rng.normal(0, 0.025, n))
    alt_large_cap_path = np.exp(0.4 * sigmoid(t, 0.30) - 0.55 * sigmoid(t, 0.50) + rng.normal(0, 0.04, n))
    alt_small_cap_path = np.exp(0.6 * sigmoid(t, 0.35) - 0.80 * sigmoid(t, 0.48) + rng.normal(0, 0.06, n))
    dominance_path = np.clip(0.45 - 0.05 * sigmoid(t, 0.30) + 0.15 * sigmoid(t, 0.55) + rng.normal(0, 0.01, n), 0.35, 0.70)
    volume_path = 1 + 2 * sigmoid(t, 0.25) * (1 - 0.8 * sigmoid(t, 0.55)) + rng.exponential(0.2, n)

    return AltseasonScenario(
        name="failed_altseason",
        description="Failed altseason: brief rotation reversed by BTC dominance",
        btc_path=btc_path,
        eth_path=eth_path,
        alt_large_cap_path=alt_large_cap_path,
        alt_small_cap_path=alt_small_cap_path,
        dominance_path=dominance_path,
        volume_path=volume_path,
        tags=["failed_alt", "btc_dominance", "bear_trap"],
    )


def compute_rotation_signal(scenario: AltseasonScenario) -> dict:
    """
    Compute rotation signal from altseason scenario data.
    Returns timing signals for each phase.
    """
    n = len(scenario.btc_path)
    # BTC relative to alts
    alt_btc_ratio = scenario.alt_large_cap_path / scenario.btc_path
    eth_btc_ratio = scenario.eth_path / scenario.btc_path

    # Rotation score: when alts outperform BTC
    alt_returns = np.diff(np.log(scenario.alt_large_cap_path))
    btc_returns = np.diff(np.log(scenario.btc_path))
    rotation_score = np.cumsum(alt_returns - btc_returns)

    # Phase detection
    dominance = scenario.dominance_path
    dom_falling = np.diff(dominance) < 0
    phase = np.where(dom_falling, "rotation", "accumulation")

    return {
        "alt_btc_ratio": alt_btc_ratio,
        "eth_btc_ratio": eth_btc_ratio,
        "rotation_score": rotation_score,
        "altseason_intensity": float((alt_btc_ratio[-1] - 1.0) / (alt_btc_ratio.std() + 1e-10)),
        "btc_dominance_trend": float(np.mean(np.diff(dominance[-20:]))),
        "is_altseason": bool(dominance[-1] < 0.45 and float(np.mean(np.diff(dominance[-10:]))) < 0),
    }


def get_altseason_scenarios(rng: Optional[np.random.Generator] = None) -> dict[str, AltseasonScenario]:
    """Return all altseason scenario types."""
    rng = rng or np.random.default_rng(42)
    return {
        "classic_altseason": classic_altseason_scenario(rng=rng),
        "mega_altseason": classic_altseason_scenario(
            btc_move=0.50, eth_multiplier=4.0, alt_multiplier=10.0, small_cap_multiplier=25.0, rng=rng
        ),
        "eth_led": eth_led_cycle_scenario(rng=rng),
        "failed_altseason": failed_altseason_scenario(rng=rng),
    }
