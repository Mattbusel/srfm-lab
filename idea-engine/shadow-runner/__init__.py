"""
Shadow Runner — silently runs genome variants in paper mode alongside live.

Submodules:
    runner      — ShadowRunner: load shadows, feed bars, maintain virtual state
    shadow_state — ShadowState, ShadowPhysics, ShadowGARCH, ShadowOU
    comparator  — ShadowComparator: rank by alpha, promote winners
"""

from .runner import ShadowRunner
from .shadow_state import ShadowState, ShadowPhysics, ShadowGARCH, ShadowOU
from .comparator import ShadowComparator

__all__ = [
    "ShadowRunner",
    "ShadowState",
    "ShadowPhysics",
    "ShadowGARCH",
    "ShadowOU",
    "ShadowComparator",
]
