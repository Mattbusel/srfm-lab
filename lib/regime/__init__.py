"""Regime detection.

``lib/regime.py`` used to sit next to this package, and the package (which
Python prefers) shadowed it, so ``from regime import RegimeDetector`` failed.
The detector now lives in ``detector.py`` and is re-exported here.
"""

from .detector import RegimeDetector

__all__ = ["RegimeDetector"]
