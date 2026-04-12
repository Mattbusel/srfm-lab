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


class LimitOrderBook:
    """Compatibility LOB for tests; standalone implementation."""

    def __init__(self, mid_price: float = 100.0,
                 tick_size: float = 0.01, depth: int = 5) -> None:
        self._init_price = mid_price
        self.tick_size   = tick_size
        self.depth       = depth
        self.mid_price   = mid_price
        self.volume_this_bar: float = 0.0
        self._spread     = tick_size * 2

    def best_bid(self) -> float:
        return self.mid_price - self._spread / 2

    def best_ask(self) -> float:
        return self.mid_price + self._spread / 2

    def spread(self) -> float:
        return self._spread

    def submit_orders(self, net_direction: float = 0.0) -> Dict:
        impact = net_direction * 0.01
        self.mid_price = max(self.mid_price + impact, 1e-3)
        self.volume_this_bar += abs(net_direction)
        return {
            "exec_price": self.mid_price,
            "impact":     impact,
            "spread":     self._spread,
            "volume":     self.volume_this_bar,
        }

    def reset(self, price: Optional[float] = None) -> None:
        self.mid_price = price if price is not None else self._init_price
        self.volume_this_bar = 0.0


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
        agent_ids:      Optional[List[str]] = None,
        **env_kwargs,
    ) -> None:
        # Support passing a config dict or agent_ids list as first positional arg
        if isinstance(n_agents, dict):
            cfg = n_agents
            agent_ids  = cfg.get("agent_ids", agent_ids)
            max_steps  = cfg.get("max_steps", max_steps)
            seed       = cfg.get("seed", seed)
            crisis_step= cfg.get("crisis_step", crisis_step)
            n_agents   = len(agent_ids) if agent_ids else cfg.get("n_agents", 8)
        elif isinstance(n_agents, (list, tuple)):
            agent_ids = list(n_agents)
            n_agents  = len(agent_ids)

        super().__init__()

        if agent_ids is not None:
            self.agent_ids = list(agent_ids)
            n_agents = len(self.agent_ids)
        else:
            self.agent_ids = [f"{agent_prefix}_{i}" for i in range(n_agents)]

        self._n_agents    = n_agents
        self._seed        = seed
        self._agent_prefix = agent_prefix
        self._id_to_idx   = {aid: i for i, aid in enumerate(self.agent_ids)}
        # Runtime state
        self.positions:     Dict[str, float] = {aid: 0.0 for aid in self.agent_ids}
        self.current_vol:   float = 1.0
        self.max_position:  float = 10.0

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
        self.positions      = {aid: 0.0 for aid in self.agent_ids}
        self.current_vol    = 1.0

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

        # Build action list in agent order; pad/truncate to inner env's expected dim
        act_list = []
        for aid in self.agent_ids:
            a = action_dict.get(aid, np.zeros(self._act_dim, dtype=np.float32))
            a = np.asarray(a, dtype=np.float32).ravel()
            if len(a) < self._act_dim:
                padded = np.zeros(self._act_dim, dtype=np.float32)
                padded[:len(a)] = a
                a = padded
            elif len(a) > self._act_dim:
                a = a[:self._act_dim]
            act_list.append(a)
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

        # RLlib-style __all__ flag (only in trunc to avoid breaking set comparisons on term)
        all_done = all(term_list) or all(trunc_list)
        trunc_d["__all__"] = all_done
        term_d["__all__"]  = all_done  # keep for compatibility; some tests check term.get("__all__")

        # Update positions from action dict (index 2 = long strength)
        for aid in self.agent_ids:
            if aid in action_dict:
                a = action_dict[aid]
                # softmax of first 3 → direction; index 2 = buy, 0 = sell
                logits = np.array(a[:3], dtype=np.float64)
                probs  = np.exp(logits - logits.max())
                probs /= probs.sum()
                size   = float(a[3]) if len(a) > 3 else 0.5
                net    = (probs[2] - probs[0]) * size
                self.positions[aid] = float(
                    np.clip(self.positions.get(aid, 0.0) + net,
                            -self.max_position, self.max_position)
                )
        # Estimate volatility from rewards; spike on crisis
        rews = [v for k, v in rew_d.items() if k != "__all__"]
        vol_from_rewards = float(abs(np.std(rews)) * 10.0) if len(rews) > 1 else 0.0
        self.current_vol = max(vol_from_rewards, self.current_vol * 0.99)
        if self._crisis_step is not None and self._step_count >= self._crisis_step:
            self.current_vol = max(self.current_vol, 5.0)

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

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _decode_actions(self, action_dict: Dict[str, np.ndarray]) -> Dict:
        decoded = {}
        for aid, a in action_dict.items():
            a = np.asarray(a, dtype=np.float32).ravel()
            probs = self._softmax(a[:3])
            size  = float(a[3]) if len(a) > 3 else 0.5
            direction = int(np.argmax(probs)) - 1  # -1, 0, +1
            decoded[aid] = {
                "direction":   direction,
                "size":        size,
                "probs":       probs,
                "signed_size": direction * size,
            }
        return decoded

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
    n_arbitrage:            int   = 0,
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
