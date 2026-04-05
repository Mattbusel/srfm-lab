"""Simulation sub-package: paper trading simulator and parallel variant screener."""

from .paper_simulator import PaperSimulator, SimulationResult
from .parallel_simulator import ParallelSimulator, VariantResult

__all__ = [
    "PaperSimulator", "SimulationResult",
    "ParallelSimulator", "VariantResult",
]
