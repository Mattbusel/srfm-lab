"""
NoiseTrader — Random agent providing realistic background noise.

Features:
  - Poisson arrival of random orders
  - Parameterized aggression (fraction market vs limit orders)
  - Log-normal order sizes
  - Populates market with realistic microstructure noise
  - No learning — purely stochastic but configurable
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from hyper_agent.agents._base_compat import BaseAgent


class NoiseTrader(BaseAgent):
    """
    Stochastic noise trader.

    Does NOT learn; acts as background noise in the market.
    Parameters control the statistical properties of its order flow.
    """

    AGENT_TYPE = "noise"

    def __init__(
        self,
        agent_id:         str,
        obs_dim:          int,
        poisson_lambda:   float = 0.7,    # mean orders per step
        aggression:       float = 0.5,    # fraction of market orders
        size_mu:          float = 0.5,    # log-normal mean for order size
        size_sigma:       float = 0.3,    # log-normal std for order size
        directional_bias: float = 0.0,    # bias toward buy (+) or sell (-)
        seed:             int   = 42,
        device:           str   = "cpu",
    ) -> None:
        super().__init__(
            agent_id   = agent_id,
            obs_dim    = obs_dim,
            hidden_dim = 16,  # minimal; not used for learning
            device     = device,
        )
        self.poisson_lambda   = poisson_lambda
        self.aggression       = aggression
        self.size_mu          = size_mu
        self.size_sigma       = size_sigma
        self.directional_bias = np.clip(directional_bias, -0.5, 0.5)
        self.rng              = np.random.default_rng(seed)

        # Direction probabilities: [short, flat, long]
        p_long  = 1.0 / 3.0 + self.directional_bias
        p_short = 1.0 / 3.0 - self.directional_bias
        p_flat  = 1.0 - p_long - p_short
        p_long  = max(p_long, 0.0)
        p_short = max(p_short, 0.0)
        p_flat  = max(p_flat, 0.0)
        total   = p_short + p_flat + p_long
        self._direction_probs = np.array(
            [p_short / total, p_flat / total, p_long / total], dtype=np.float32
        )

        # Order log
        self.order_log: List[Dict] = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> str:
        return self.AGENT_TYPE

    def act(
        self,
        obs:           np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Sample a random action.

        Number of orders arrives as Poisson; size is log-normal.
        Returns standard action format for environment compatibility.
        """
        # Decide whether to trade this step
        n_orders = self.rng.poisson(self.poisson_lambda)

        if n_orders == 0:
            # No trade
            logits = np.array([-1.0, 5.0, -1.0], dtype=np.float32)
            size   = 0.0
            action = np.array([*logits, size], dtype=np.float32)
            self.order_log.append({"direction": 0, "size": 0.0})
            return action, 0.0, 0.0

        # Sample direction
        direction_idx = int(self.rng.choice(3, p=self._direction_probs))
        direction     = direction_idx - 1  # -1, 0, +1

        # Sample size: log-normal, clipped to [0, 1]
        raw_size = float(self.rng.lognormal(self.size_mu, self.size_sigma))
        size     = float(np.clip(raw_size / 10.0, 0.0, 1.0))

        # Aggression: market vs limit (affects how quickly filled)
        is_market = self.rng.random() < self.aggression
        if not is_market:
            size *= 0.5  # limit orders: smaller effective size

        logits = np.array([-5.0, -5.0, -5.0], dtype=np.float32)
        logits[direction_idx] = 5.0

        action   = np.array([*logits, size], dtype=np.float32)
        log_prob = float(np.log(self._direction_probs[direction_idx] + 1e-8))

        self.order_log.append({
            "direction":  direction,
            "size":       size,
            "is_market":  is_market,
            "n_orders":   n_orders,
        })
        if len(self.order_log) > 1000:
            self.order_log.pop(0)

        return action, log_prob, 0.0

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        """Noise traders do not learn; update is a no-op."""
        return {}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def set_directional_bias(self, bias: float) -> None:
        """
        Adjust directional bias (positive = bullish, negative = bearish).
        Useful for simulating informed noise or panic selling.
        """
        self.directional_bias = np.clip(bias, -0.49, 0.49)
        p_long  = 1.0 / 3.0 + self.directional_bias
        p_short = 1.0 / 3.0 - self.directional_bias
        p_flat  = max(1.0 - p_long - p_short, 0.0)
        total   = p_short + p_flat + p_long
        self._direction_probs = np.array(
            [p_short / total, p_flat / total, p_long / total], dtype=np.float32
        )

    def set_aggression(self, aggression: float) -> None:
        self.aggression = float(np.clip(aggression, 0.0, 1.0))

    def volume_stats(self) -> Dict[str, float]:
        """Summarize order flow statistics."""
        if not self.order_log:
            return {}
        sizes     = [o["size"] for o in self.order_log]
        dirs      = [o["direction"] for o in self.order_log]
        buy_frac  = sum(1 for d in dirs if d > 0) / max(len(dirs), 1)
        return {
            "mean_size":   float(np.mean(sizes)),
            "std_size":    float(np.std(sizes)),
            "buy_fraction": buy_frac,
            "activity_rate": float(sum(1 for s in sizes if s > 0)) / max(len(sizes), 1),
        }

    def clone_with_params(
        self,
        poisson_lambda:   Optional[float] = None,
        aggression:       Optional[float] = None,
        directional_bias: Optional[float] = None,
        new_id:           Optional[str]   = None,
    ) -> "NoiseTrader":
        """Return a new NoiseTrader with (optionally) modified parameters."""
        return NoiseTrader(
            agent_id         = new_id or f"{self.agent_id}_clone",
            obs_dim          = self.obs_dim,
            poisson_lambda   = poisson_lambda   or self.poisson_lambda,
            aggression       = aggression       or self.aggression,
            size_mu          = self.size_mu,
            size_sigma       = self.size_sigma,
            directional_bias = directional_bias if directional_bias is not None
                               else self.directional_bias,
            device           = str(self.device),
        )
