"""
env_compat.py — Compatibility shim wrapping MultiAssetTradingEnv with
string agent IDs and dict-keyed observations/rewards.

All training code imports `from hyper_agent.env_compat import make_env`
(or uses the alias in __init__.py).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hyper_agent.environment import MultiAssetTradingEnv


class DictMultiAgentEnv(gym.Env):
    """
    Wraps MultiAssetTradingEnv to expose a string-keyed dict interface.

    obs  → Dict[str, np.ndarray]
    acts → Dict[str, np.ndarray]  (accepted; int lists also OK internally)
    rew  → Dict[str, float]
    """

    metadata = {"render_modes": ["human"], "name": "DictMultiAgentEnv-v1"}

    def __init__(
        self,
        n_agents:       int   = 8,
        max_steps:      int   = 500,
        agent_prefix:   str   = "agent",
        seed:           int   = 42,
        crisis_step:    Optional[int] = None,
        **env_kwargs,
    ) -> None:
        super().__init__()
        self._n_agents    = n_agents
        self._seed        = seed
        self._agent_prefix = agent_prefix
        self.agent_ids    = [f"{agent_prefix}_{i}" for i in range(n_agents)]
        self._id_to_idx   = {aid: i for i, aid in enumerate(self.agent_ids)}

        # Build inner environment
        self._inner = MultiAssetTradingEnv(
            num_agents = n_agents,
            max_steps  = max_steps,
            seed       = seed,
            marl_mode  = True,
            **env_kwargs,
        )
        self._crisis_step = crisis_step
        self._step_count  = 0

        # Extract obs/action dims from inner env
        inner_obs   = self._inner.observation_space
        inner_act   = self._inner.action_space
        obs_shape   = inner_obs.shape
        act_shape   = inner_act.shape

        self.observation_spaces: Dict[str, spaces.Box] = {
            aid: spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
            )
            for aid in self.agent_ids
        }
        self.action_spaces: Dict[str, spaces.Box] = {
            aid: spaces.Box(
                low=inner_act.low, high=inner_act.high,
                shape=act_shape, dtype=np.float32
            )
            for aid in self.agent_ids
        }
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space      = spaces.Dict(self.action_spaces)

        self._obs_dim = int(np.prod(obs_shape))
        self._act_dim = int(np.prod(act_shape))
        self._last_obs_list: List[np.ndarray] = []

    @property
    def obs_shape(self) -> int:
        return self._obs_dim

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        obs_arr, info = self._inner.reset(seed=seed)
        # inner reset returns (per_agent_obs_list, info_dict)
        # If it returns a single array, split it
        if isinstance(obs_arr, np.ndarray) and obs_arr.ndim == 1:
            # Single-agent mode; tile
            obs_list = [obs_arr] * self._n_agents
        elif isinstance(obs_arr, list):
            obs_list = obs_arr
        else:
            obs_list = [obs_arr] * self._n_agents

        self._last_obs_list = obs_list
        self._step_count    = 0

        obs_dict  = {aid: obs_list[i].astype(np.float32)
                     for i, aid in enumerate(self.agent_ids)}
        info_dict = {aid: (info if isinstance(info, dict) else {})
                     for aid in self.agent_ids}
        return obs_dict, info_dict

    def step(
        self,
        action_dict: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        self._step_count += 1

        # Build action list in agent order
        act_list = [
            action_dict.get(aid, np.zeros(self._act_dim, dtype=np.float32))
            for aid in self.agent_ids
        ]
        # inner step
        result = self._inner.step(act_list)

        if len(result) == 5:
            obs_out, rew_out, term_out, trunc_out, info_out = result
        else:
            obs_out, rew_out, term_out, info_out = result
            trunc_out = [False] * self._n_agents

        # Normalize outputs to lists
        if isinstance(obs_out, np.ndarray) and obs_out.ndim == 1:
            obs_list = [obs_out] * self._n_agents
        elif isinstance(obs_out, list):
            obs_list = obs_out
        else:
            obs_list = [obs_out] * self._n_agents

        if isinstance(rew_out, (int, float)):
            rew_list = [float(rew_out)] * self._n_agents
        elif isinstance(rew_out, np.ndarray):
            rew_list = rew_out.tolist()
        else:
            rew_list = list(rew_out)

        if isinstance(term_out, bool):
            term_list = [term_out] * self._n_agents
        else:
            term_list = [bool(t) for t in term_out]

        if isinstance(trunc_out, bool):
            trunc_list = [trunc_out] * self._n_agents
        else:
            trunc_list = [bool(t) for t in trunc_out]

        # Package as dicts
        obs_d  = {aid: obs_list[i].astype(np.float32)
                  for i, aid in enumerate(self.agent_ids)}
        rew_d  = {aid: float(rew_list[i]) for i, aid in enumerate(self.agent_ids)}
        term_d = {aid: term_list[i]       for i, aid in enumerate(self.agent_ids)}
        trunc_d= {aid: trunc_list[i]      for i, aid in enumerate(self.agent_ids)}
        info_d = {aid: (info_out if isinstance(info_out, dict) else {})
                  for aid in self.agent_ids}

        # RLlib-style __all__ flag
        all_done = all(term_list) or all(trunc_list)
        term_d["__all__"]  = all_done
        trunc_d["__all__"] = all_done

        return obs_d, rew_d, term_d, trunc_d, info_d

    def render(self, mode: str = "human") -> None:
        return self._inner.render()

    def get_global_state(self) -> np.ndarray:
        """Concatenated observations for centralized critic."""
        if self._last_obs_list:
            return np.concatenate([o.ravel() for o in self._last_obs_list]).astype(np.float32)
        return np.zeros(self._obs_dim * self._n_agents, dtype=np.float32)

    def get_price_history(self) -> np.ndarray:
        """Return mid-price history from inner env if available."""
        inner = self._inner
        if hasattr(inner, "order_books") and inner.order_books:
            ob    = inner.order_books[0]
            if hasattr(ob, "_trade_history"):
                return np.array([t.price for t in ob._trade_history], dtype=np.float32)
        return np.zeros(10, dtype=np.float32)

    @property
    def env(self) -> "DictMultiAgentEnv":
        """Self-reference for code that does env.env.max_steps"""
        return self

    @property
    def max_steps(self) -> int:
        return self._inner.max_steps

    @property
    def lob(self):
        """Compatibility: expose first order book as lob."""
        if hasattr(self._inner, "order_books") and self._inner.order_books:
            return self._inner.order_books[0]
        return None


# ============================================================
# Factory functions matching original API
# ============================================================

def make_env(
    n_market_makers:        int   = 5,
    n_momentum:             int   = 5,
    n_arbitrage:            int   = 3,
    n_noise:                int   = 10,
    max_steps:              int   = 500,
    crisis_step:            Optional[int] = None,
    crisis_vol_multiplier:  float = 5.0,
    crisis_duration:        int   = 50,
    impact_coeff:           float = 0.05,
    mean_reversion:         float = 0.1,
    seed:                   int   = 42,
    **kwargs,
) -> DictMultiAgentEnv:
    """
    Compatibility factory: builds a DictMultiAgentEnv with the same
    keyword argument signature as the original make_env.
    """
    n_agents = n_market_makers + n_momentum + n_arbitrage + n_noise
    n_agents = max(n_agents, 1)

    return DictMultiAgentEnv(
        n_agents    = n_agents,
        max_steps   = max_steps,
        seed        = seed,
        crisis_step = crisis_step,
    )


def make_curriculum_env(stage: int, seed: int = 42) -> DictMultiAgentEnv:
    configs = {
        1: dict(n_market_makers=1, n_momentum=1, n_noise=0, max_steps=200),
        2: dict(n_market_makers=3, n_momentum=3, n_noise=4, max_steps=300),
        3: dict(n_market_makers=5, n_momentum=5, n_noise=10, max_steps=400, crisis_step=200),
        4: dict(n_market_makers=10, n_momentum=15, n_noise=25, max_steps=500, crisis_step=250),
        5: dict(n_market_makers=10, n_momentum=20, n_noise=35, max_steps=600, crisis_step=300),
    }
    cfg = configs.get(stage, configs[1])
    return make_env(seed=seed, **cfg)


# Aliases
MultiAgentTradingEnv = DictMultiAgentEnv
MarketEnvironment    = DictMultiAgentEnv
