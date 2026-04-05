"""
portfolio-optimizer/__init__.py

Public API for the Portfolio Optimizer subsystem of the Idea Automation Engine.

Provides mean-variance, Black-Litterman, Hierarchical Risk Parity,
and multi-objective optimisation; a rebalancing engine; and genome-portfolio
allocation across multiple trading strategies.
"""

from portfolio_optimizer.optimizer import PortfolioOptimizer
from portfolio_optimizer.hrp import HierarchicalRiskParity
from portfolio_optimizer.rebalancer import Rebalancer, RebalanceSchedule
from portfolio_optimizer.multi_objective import (
    MultiObjectivePortfolioOptimizer,
    ParetoPortfolio,
)

__all__ = [
    "PortfolioOptimizer",
    "HierarchicalRiskParity",
    "Rebalancer",
    "RebalanceSchedule",
    "MultiObjectivePortfolioOptimizer",
    "ParetoPortfolio",
]
