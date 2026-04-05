"""
experiment-tracker
==================
Tracks every experiment — hypothesis tests, genome runs, counterfactuals —
through their full lifecycle with provenance, reproducibility, and result
comparison.

Inspired by MLflow's experiment-tracking API, but entirely SQLite-backed
with no external service dependencies.

Submodules
----------
tracker   — ExperimentTracker: start/log/end experiments, search, compare
lineage   — ExperimentLineage: parent–child chains and provenance reports
reporter  — ExperimentReporter: Markdown tables, parameter importance, curves
"""

from __future__ import annotations

from .lineage import ExperimentLineage
from .reporter import ExperimentReporter
from .tracker import ExperimentRecord, ExperimentTracker

__all__ = [
    "ExperimentTracker",
    "ExperimentRecord",
    "ExperimentLineage",
    "ExperimentReporter",
]
