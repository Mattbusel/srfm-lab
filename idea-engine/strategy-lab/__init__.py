"""
strategy-lab: Strategy versioning and A/B testing framework for the SRFM trading lab.

Tracks strategy versions like git tracks code, runs A/B experiments on paper accounts
simultaneously, and promotes winning variants.

Sub-packages
------------
versioning   -- Version control for strategy parameters (StrategyVersion, VersionStore, diffs)
experiments  -- A/B test runner, significance testing, promotion logic, shadow trading
champion     -- Champion tracker, performance monitoring, degradation detection
simulation   -- High-fidelity paper simulator, parallel variant screening
reporting    -- Experiment reports and strategy lineage visualisation

Quick start
-----------
>>> from strategy_lab.versioning.version_store import VersionStore
>>> from strategy_lab.versioning.strategy_version import StrategyVersion, VersionStatus
>>> store = VersionStore()
>>> v = StrategyVersion.new(parameters={"min_hold_bars": 6}, description="baseline")
>>> store.save(v)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("strategy-lab")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "versioning",
    "experiments",
    "champion",
    "simulation",
    "reporting",
]
