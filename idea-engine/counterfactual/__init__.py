"""
Counterfactual Oracle — asks "what would have happened with different parameters?"

Submodules:
    oracle           — CounterfactualOracle: run, compare, rank parameter variants
    parameter_space  — ParameterSpace: LHS, neighborhood, gradient, steepest-ascent
    sensitivity      — SensitivityAnalyzer: Sobol indices, tornado, interaction matrix
"""

from .oracle import CounterfactualOracle
from .parameter_space import ParameterSpace
from .sensitivity import SensitivityAnalyzer

__all__ = ["CounterfactualOracle", "ParameterSpace", "SensitivityAnalyzer"]
