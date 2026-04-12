"""Agent implementations for the Hyper-Agent MARL ecosystem."""

from hyper_agent.agents.base_agent import (
    BaseAgent, ObservationEncoder, RunningMeanStd as StandardNorm,
    EpisodeBuffer as Memory, layer_init,
)
from hyper_agent.agents.market_maker_agent import MarketMakerAgent
from hyper_agent.agents.momentum_agent import MomentumAgent
from hyper_agent.agents.arbitrage_agent import ArbitrageAgent
from hyper_agent.agents.noise_trader import NoiseTrader

# Try importing agents that depend on upgraded base
try:
    from hyper_agent.agents.mappo_agent import MAPPOAgent
except ImportError:
    MAPPOAgent = None  # type: ignore

try:
    from hyper_agent.agents.mean_field_agent import MeanFieldAgent
except ImportError:
    MeanFieldAgent = None  # type: ignore

__all__ = [
    "BaseAgent",
    "ObservationEncoder",
    "StandardNorm",
    "Memory",
    "MAPPOAgent",
    "MeanFieldAgent",
    "MarketMakerAgent",
    "MomentumAgent",
    "ArbitrageAgent",
    "NoiseTrader",
]
