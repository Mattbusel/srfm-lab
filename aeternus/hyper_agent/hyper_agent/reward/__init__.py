"""Reward shaping and credit assignment for Hyper-Agent MARL."""

from hyper_agent.reward.reward_shaping import (
    IndividualReward,
    TeamReward,
    MarketQualityReward,
    AdversarialPenalty,
    PotentialBasedShaping,
    CuriosityBonus,
    CompositeReward,
)
from hyper_agent.reward.credit_assignment import (
    COMAAdvantage,
    QMIXMixer,
    VDNMixer,
)

__all__ = [
    "IndividualReward",
    "TeamReward",
    "MarketQualityReward",
    "AdversarialPenalty",
    "PotentialBasedShaping",
    "CuriosityBonus",
    "CompositeReward",
    "COMAAdvantage",
    "QMIXMixer",
    "VDNMixer",
]
