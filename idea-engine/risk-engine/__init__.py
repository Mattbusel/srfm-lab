"""
risk-engine/__init__.py

Public API for the Risk Engine subsystem of the Idea Automation Engine.

Provides Value at Risk calculations, tail risk analysis, correlation/concentration
risk measurement, and active drawdown control — each capable of generating
risk-reduction Hypothesis objects that feed back into the hypothesis pipeline.
"""

from risk_engine.var_calculator import VaRCalculator, VaRBacktestResult
from risk_engine.tail_analyzer import TailAnalyzer
from risk_engine.correlation_risk import CorrelationRiskAnalyzer
from risk_engine.drawdown_controller import DrawdownController, DrawdownEvent

__all__ = [
    "VaRCalculator",
    "VaRBacktestResult",
    "TailAnalyzer",
    "CorrelationRiskAnalyzer",
    "DrawdownController",
    "DrawdownEvent",
]
