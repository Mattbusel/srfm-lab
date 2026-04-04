"""
research/regime_lab/__init__.py
================================
Regime Simulation Lab — public API surface.

Modules
-------
detector     — HMM, rolling-vol, trend, changepoint, and ensemble regime detectors
generator    — Markov, GARCH, Heston, jump-diffusion, bootstrap scenario generators
stress       — 20 historical stress scenarios + StressTester framework
transition   — Transition-matrix analytics and regime duration statistics
simulator    — Regime-aware Monte Carlo engine (extends spacetime/engine/mc.py)
calibration  — MLE/Bayes calibration of regime parameters to history
report       — HTML / console regime-lab report
cli          — Click CLI entry-points

Regime constants
----------------
BULL, BEAR, SIDEWAYS, HIGH_VOL  — the four canonical market states
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "REGIMES",
    "BULL",
    "BEAR",
    "SIDEWAYS",
    "HIGH_VOL",
]

# Canonical regime identifiers shared across all sub-modules.
BULL      = "BULL"
BEAR      = "BEAR"
SIDEWAYS  = "SIDEWAYS"
HIGH_VOL  = "HIGH_VOL"

REGIMES: tuple[str, ...] = (BULL, BEAR, SIDEWAYS, HIGH_VOL)
