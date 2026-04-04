"""
research/agent_training/__init__.py

Agent training environment for SRFM-Lab.
Provides Gym-compatible environments, replay buffers, pure-numpy networks,
agent implementations, training orchestration, hyperparameter search,
and evaluation utilities for the D3QN / DDQN / TD3QN ensemble.
"""

from __future__ import annotations

from research.agent_training.environment import (
    TradingEnvironment,
    MultiInstrumentEnvironment,
    EnvironmentConfig,
    RewardShaper,
    EpisodeStats,
    episode_stats,
)
from research.agent_training.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    EpisodeBuffer,
    HindsightReplayBuffer,
    Batch,
)
from research.agent_training.networks import (
    Linear,
    ReLU,
    Tanh,
    Sigmoid,
    GELU,
    LayerNorm,
    Dropout,
    Sequential,
    DuelingHead,
    ActorNetwork,
    CriticNetwork,
    TransformerBlock,
    LSTMCell,
)
from research.agent_training.agents import (
    DQNAgent,
    DDQNAgent,
    D3QNAgent,
    TD3Agent,
    PPOAgent,
    EnsembleAgent,
)
from research.agent_training.trainer import (
    AgentTrainer,
    TrainingConfig,
    TrainingResult,
    EvalResult,
)
from research.agent_training.hyperopt import (
    AgentHyperSearch,
    HyperSearchResult,
    CVResult,
)
from research.agent_training.evaluation import (
    AgentEvaluator,
)

__all__ = [
    "TradingEnvironment",
    "MultiInstrumentEnvironment",
    "EnvironmentConfig",
    "RewardShaper",
    "EpisodeStats",
    "episode_stats",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "EpisodeBuffer",
    "HindsightReplayBuffer",
    "Batch",
    "Linear",
    "ReLU",
    "Tanh",
    "Sigmoid",
    "GELU",
    "LayerNorm",
    "Dropout",
    "Sequential",
    "DuelingHead",
    "ActorNetwork",
    "CriticNetwork",
    "TransformerBlock",
    "LSTMCell",
    "DQNAgent",
    "DDQNAgent",
    "D3QNAgent",
    "TD3Agent",
    "PPOAgent",
    "EnsembleAgent",
    "AgentTrainer",
    "TrainingConfig",
    "TrainingResult",
    "EvalResult",
    "AgentHyperSearch",
    "HyperSearchResult",
    "CVResult",
    "AgentEvaluator",
]

__version__ = "0.1.0"
