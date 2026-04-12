"""
CurriculumScheduler — Progressively harder training scenarios.

Stages:
  1: 2 agents, no crisis, high liquidity
  2: 10 agents, mild volatility
  3: 50 agents, regime switches
  4: 100+ agents, crisis injection
  5: adversarial agents (manipulation attempts)

Automatic progression: advance when current stage performance plateaus.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from hyper_agent.env_compat import MultiAgentTradingEnv, make_env


# ============================================================
# Stage definitions
# ============================================================

STAGE_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "name":            "Basics",
        "description":     "2 agents, no crisis, high liquidity",
        "n_market_makers": 1,
        "n_momentum":      1,
        "n_arbitrage":     0,
        "n_noise":         0,
        "max_steps":       200,
        "crisis_step":     None,
        "impact_coeff":    0.02,   # low impact (high liquidity)
        "mean_reversion":  0.2,
        "target_reward":   0.5,    # advance when this is reached
        "min_episodes":    50,     # minimum episodes before advancing
    },
    2: {
        "name":            "Small Population",
        "description":     "10 agents, mild volatility",
        "n_market_makers": 3,
        "n_momentum":      3,
        "n_arbitrage":     2,
        "n_noise":         2,
        "max_steps":       300,
        "crisis_step":     None,
        "impact_coeff":    0.04,
        "mean_reversion":  0.15,
        "target_reward":   0.3,
        "min_episodes":    100,
    },
    3: {
        "name":            "Regime Switches",
        "description":     "50 agents, regime switches",
        "n_market_makers": 5,
        "n_momentum":      5,
        "n_arbitrage":     3,
        "n_noise":         7,  # total ~20
        "max_steps":       400,
        "crisis_step":     200,
        "impact_coeff":    0.05,
        "mean_reversion":  0.1,
        "target_reward":   0.1,
        "min_episodes":    150,
    },
    4: {
        "name":            "Large Population",
        "description":     "100+ agents, crisis injection",
        "n_market_makers": 10,
        "n_momentum":      15,
        "n_arbitrage":     5,
        "n_noise":         20,
        "max_steps":       500,
        "crisis_step":     250,
        "crisis_duration": 75,
        "impact_coeff":    0.05,
        "mean_reversion":  0.08,
        "target_reward":   0.05,
        "min_episodes":    200,
    },
    5: {
        "name":            "Adversarial",
        "description":     "Adversarial agents attempt manipulation",
        "n_market_makers": 10,
        "n_momentum":      20,
        "n_arbitrage":     5,
        "n_noise":         30,
        "max_steps":       600,
        "crisis_step":     300,
        "crisis_duration": 100,
        "crisis_vol_multiplier": 8.0,
        "impact_coeff":    0.06,
        "mean_reversion":  0.05,
        "target_reward":   None,    # no automatic advancement from stage 5
        "min_episodes":    300,
    },
}


# ============================================================
# Performance plateau detector
# ============================================================

class PlateauDetector:
    """
    Detects when training performance has plateaued.

    Uses linear regression slope on recent reward history.
    Plateau = slope close to 0 over a window.
    """

    def __init__(
        self,
        window:    int   = 30,
        threshold: float = 0.001,
        min_n:     int   = 20,
    ) -> None:
        self.window    = window
        self.threshold = threshold
        self.min_n     = min_n
        self._rewards: deque = deque(maxlen=window)

    def update(self, reward: float) -> None:
        self._rewards.append(reward)

    def is_plateau(self) -> bool:
        if len(self._rewards) < self.min_n:
            return False
        y   = np.array(list(self._rewards))
        x   = np.arange(len(y), dtype=np.float64)
        # OLS slope
        n   = len(y)
        m   = (n * (x * y).sum() - x.sum() * y.sum()) / (
            n * (x**2).sum() - x.sum()**2 + 1e-12
        )
        return abs(m) < self.threshold

    def trend(self) -> float:
        if len(self._rewards) < 3:
            return 0.0
        y = np.array(list(self._rewards))
        x = np.arange(len(y), dtype=np.float64)
        n = len(y)
        return float(
            (n * (x * y).sum() - x.sum() * y.sum())
            / (n * (x**2).sum() - x.sum()**2 + 1e-12)
        )

    def reset(self) -> None:
        self._rewards.clear()


# ============================================================
# CurriculumScheduler
# ============================================================

class CurriculumScheduler:
    """
    Manages curriculum progression for MARL training.

    Automatically advances stages based on:
      1. Performance plateau detection
      2. Target reward reached
      3. Minimum episode count met

    Also supports:
      - Manual stage override
      - Custom per-stage callbacks
      - Stage history tracking
    """

    def __init__(
        self,
        initial_stage:     int   = 1,
        plateau_window:    int   = 30,
        plateau_threshold: float = 0.002,
        min_episodes_per_stage: Optional[int] = None,
        auto_advance:      bool  = True,
        seed:              int   = 42,
        callbacks:         Optional[Dict[int, Callable]] = None,
    ) -> None:
        self.current_stage       = initial_stage
        self.max_stage           = max(STAGE_CONFIGS.keys())
        self.auto_advance        = auto_advance
        self.seed                = seed
        self.callbacks           = callbacks or {}
        self._min_ep_override    = min_episodes_per_stage

        self.plateau_detector    = PlateauDetector(plateau_window, plateau_threshold)

        # Per-stage tracking
        self._stage_episodes:   Dict[int, int]   = {s: 0 for s in STAGE_CONFIGS}
        self._stage_rewards:    Dict[int, List]  = {s: [] for s in STAGE_CONFIGS}
        self._stage_entry_time: Dict[int, int]   = {initial_stage: 0}
        self.total_episodes     = 0
        self.stage_history:     List[Tuple[int, int]] = [(initial_stage, 0)]

        # Current environment (lazily constructed)
        self._env: Optional[MultiAgentTradingEnv] = None

    # ------------------------------------------------------------------
    # Environment construction
    # ------------------------------------------------------------------

    def build_env(self, stage: Optional[int] = None) -> MultiAgentTradingEnv:
        """Build environment for given stage (default: current stage)."""
        s   = stage or self.current_stage
        cfg = STAGE_CONFIGS[s].copy()

        env_kwargs = {
            "n_market_makers":      cfg["n_market_makers"],
            "n_momentum":           cfg["n_momentum"],
            "n_arbitrage":          cfg["n_arbitrage"],
            "n_noise":              cfg["n_noise"],
            "max_steps":            cfg["max_steps"],
            "crisis_step":          cfg.get("crisis_step"),
            "seed":                 self.seed,
            "impact_coeff":         cfg.get("impact_coeff", 0.05),
            "mean_reversion":       cfg.get("mean_reversion", 0.1),
        }
        if "crisis_duration" in cfg:
            env_kwargs["crisis_duration"] = cfg["crisis_duration"]
        if "crisis_vol_multiplier" in cfg:
            env_kwargs["crisis_vol_multiplier"] = cfg["crisis_vol_multiplier"]

        self._env = make_env(**env_kwargs)
        return self._env

    def current_env(self) -> MultiAgentTradingEnv:
        if self._env is None:
            return self.build_env()
        return self._env

    # ------------------------------------------------------------------
    # Episode recording and advancement
    # ------------------------------------------------------------------

    def record_episode(self, mean_reward: float) -> bool:
        """
        Record episode performance and check if stage should advance.

        Returns True if stage was advanced.
        """
        self._stage_episodes[self.current_stage] += 1
        self._stage_rewards[self.current_stage].append(mean_reward)
        self.total_episodes += 1
        self.plateau_detector.update(mean_reward)

        if not self.auto_advance:
            return False
        if self.current_stage >= self.max_stage:
            return False

        # Check advancement criteria
        if self._should_advance():
            self._advance_stage()
            return True
        return False

    def _should_advance(self) -> bool:
        cfg       = STAGE_CONFIGS[self.current_stage]
        min_ep    = self._min_ep_override or cfg["min_episodes"]
        n_ep      = self._stage_episodes[self.current_stage]

        if n_ep < min_ep:
            return False

        # Criterion 1: performance plateau
        if self.plateau_detector.is_plateau():
            return True

        # Criterion 2: target reward reached
        target = cfg.get("target_reward")
        if target is not None:
            recent = self._stage_rewards[self.current_stage][-20:]
            if recent and float(np.mean(recent)) >= target:
                return True

        return False

    def _advance_stage(self) -> None:
        old_stage         = self.current_stage
        self.current_stage += 1
        self._stage_entry_time[self.current_stage] = self.total_episodes
        self.stage_history.append((self.current_stage, self.total_episodes))
        self.plateau_detector.reset()

        # Rebuild environment for new stage
        self._env = None
        self.build_env()

        # Run callback if registered
        cb = self.callbacks.get(self.current_stage)
        if cb is not None:
            cb(old_stage, self.current_stage)

        print(
            f"[Curriculum] Advanced: Stage {old_stage} → {self.current_stage} "
            f"({STAGE_CONFIGS[self.current_stage]['name']}) "
            f"at episode {self.total_episodes}"
        )

    # ------------------------------------------------------------------
    # Manual control
    # ------------------------------------------------------------------

    def set_stage(self, stage: int, rebuild_env: bool = True) -> None:
        """Manually set training stage."""
        assert stage in STAGE_CONFIGS, f"Unknown stage: {stage}"
        self.current_stage = stage
        self.plateau_detector.reset()
        if rebuild_env:
            self._env = None
            self.build_env()

    def force_advance(self) -> bool:
        """Force advance to next stage."""
        if self.current_stage < self.max_stage:
            self._advance_stage()
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def stage_config(self) -> Dict[str, Any]:
        return STAGE_CONFIGS[self.current_stage].copy()

    @property
    def stage_name(self) -> str:
        return STAGE_CONFIGS[self.current_stage]["name"]

    def n_agents_current(self) -> int:
        cfg = STAGE_CONFIGS[self.current_stage]
        return (cfg["n_market_makers"] + cfg["n_momentum"]
                + cfg["n_arbitrage"] + cfg["n_noise"])

    def episodes_in_stage(self) -> int:
        return self._stage_episodes.get(self.current_stage, 0)

    def stage_mean_reward(self, last_n: int = 50) -> float:
        rewards = self._stage_rewards.get(self.current_stage, [])
        if not rewards:
            return 0.0
        return float(np.mean(rewards[-last_n:]))

    def performance_trend(self) -> float:
        return self.plateau_detector.trend()

    def curriculum_summary(self) -> Dict[str, Any]:
        return {
            "current_stage":      self.current_stage,
            "stage_name":         self.stage_name,
            "total_episodes":     self.total_episodes,
            "episodes_in_stage":  self.episodes_in_stage(),
            "stage_mean_reward":  self.stage_mean_reward(),
            "trend":              self.performance_trend(),
            "is_plateau":         self.plateau_detector.is_plateau(),
            "stage_history":      self.stage_history,
            "n_agents":           self.n_agents_current(),
        }

    def stage_report(self) -> str:
        lines = [
            f"=== Curriculum Status ===",
            f"Stage {self.current_stage}/{self.max_stage}: {self.stage_name}",
            f"Total episodes: {self.total_episodes}",
            f"Episodes in stage: {self.episodes_in_stage()}",
            f"Recent mean reward: {self.stage_mean_reward():.4f}",
            f"Performance trend: {self.performance_trend():.6f}",
            f"Plateau detected: {self.plateau_detector.is_plateau()}",
        ]
        return "\n".join(lines)
