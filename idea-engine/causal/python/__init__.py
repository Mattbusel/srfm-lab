"""
causal/python/__init__.py

Causal graph subsystem for the Idea Automation Engine.
Distinguishes correlation from causation using Granger tests and DAG construction.
"""

from causal.python.feature_extractor import CausalFeatureExtractor
from causal.python.granger.granger_tests import GrangerTester, GrangerResult
from causal.python.granger.granger_graph import CausalGraph
from causal.python.hypothesis_converter import CausalHypothesisConverter
from causal.python.visualization import CausalVisualizer

__all__ = [
    "CausalFeatureExtractor",
    "GrangerTester",
    "GrangerResult",
    "CausalGraph",
    "CausalHypothesisConverter",
    "CausalVisualizer",
]
