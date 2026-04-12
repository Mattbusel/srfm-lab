"""Communication infrastructure for Hyper-Agent MARL."""

from hyper_agent.communication.message_passing import (
    AgentCommunicationGraph,
    MessagePassing,
    AttentionAggregation,
    CommunicationPolicy,
)
from hyper_agent.communication.consensus_protocol import (
    ConsensusProtocol,
    CredibilityTracker,
    WeightedVoting,
)

__all__ = [
    "AgentCommunicationGraph",
    "MessagePassing",
    "AttentionAggregation",
    "CommunicationPolicy",
    "ConsensusProtocol",
    "CredibilityTracker",
    "WeightedVoting",
]
