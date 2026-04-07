"""
ml/rl_agent -- Reinforcement learning agent training framework for SRFM exit policy.

Provides DQN and PPO agents trained in an OpenAI Gym-compatible trading
environment. Trained agents can be exported to config/rl_exit_qtable.json
for live-trading consumption by RLExitPolicy in tools/live_trader_alpaca.py.
"""

from ml.rl_agent.environment import TradingEnvironment, TradingState, TradeEpisodeGenerator
from ml.rl_agent.q_network import QNetwork, ReplayBuffer, DQNAgent
from ml.rl_agent.ppo_agent import PolicyNetwork, PPOAgent
from ml.rl_agent.trainer import RLTrainer

__all__ = [
    "TradingEnvironment",
    "TradingState",
    "TradeEpisodeGenerator",
    "QNetwork",
    "ReplayBuffer",
    "DQNAgent",
    "PolicyNetwork",
    "PPOAgent",
    "RLTrainer",
]
