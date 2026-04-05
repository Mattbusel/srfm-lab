"""debate-system/agents/__init__.py"""
from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote
from debate_system.agents.statistician import StatisticalAnalyst
from debate_system.agents.devil_advocate import DevilsAdvocate
from debate_system.agents.market_structure import MarketStructureAnalyst
from debate_system.agents.regime_specialist import RegimeSpecialist
from debate_system.agents.risk_manager import RiskManagementAnalyst
from debate_system.agents.quant_researcher import QuantResearcher

__all__ = [
    "BaseAnalyst", "AnalystVerdict", "Vote",
    "StatisticalAnalyst", "DevilsAdvocate", "MarketStructureAnalyst",
    "RegimeSpecialist", "RiskManagementAnalyst", "QuantResearcher",
]
