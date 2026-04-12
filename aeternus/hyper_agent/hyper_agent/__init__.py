"""
AETERNUS Project — Module 6: Hyper-Agent
Multi-Agent Reinforcement Learning ecosystem for financial markets.

Agents compete, collude, and respond to one another in a simulated
limit order book environment with configurable crisis injection.
"""

from hyper_agent.env_compat import (
    DictMultiAgentEnv, make_env, make_curriculum_env,
    MultiAgentTradingEnv, MarketEnvironment,
)
from hyper_agent.environment import MultiAssetTradingEnv

__version__ = "0.1.0"
__all__ = [
    "MultiAssetTradingEnv",
    "DictMultiAgentEnv",
    "MarketEnvironment",
    "MultiAgentTradingEnv",
    "make_env",
    "make_curriculum_env",
]
