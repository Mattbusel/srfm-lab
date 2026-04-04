"""
research/param_explorer
=======================
Comprehensive parameter space exploration toolkit for the srfm-lab trading
research platform.

Modules
-------
space           : ParamSpec / ParamSpace / BHParamSpace / LiveTraderParamSpace
sensitivity     : OAT, Sobol, Morris screening, gradient / Hessian analysis
landscape       : 2-D scans, basin detection, robustness scoring
bayesian_opt    : GP-based Bayesian optimisation (single- and multi-objective)
regime_sensitivity: How optimal parameters shift across market regimes
visualization   : Publication-quality dashboards (matplotlib + optional plotly)
cli             : Click command-line interface

Typical usage
-------------
>>> from research.param_explorer.space import BHParamSpace
>>> from research.param_explorer.bayesian_opt import BayesianOptimizer, AcquisitionFunction
>>> space = BHParamSpace()
>>> def obj(params): ...
>>> opt = BayesianOptimizer(space, obj, acquisition=AcquisitionFunction.EI)
>>> result = opt.run(n_iter=50)
>>> print(result.best_params, result.best_score)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "srfm-lab"

from research.param_explorer.space import (
    ParamSpec,
    ParamSpace,
    BHParamSpace,
    LiveTraderParamSpace,
    ParamType,
    sensitivity_indices,
    correlation_heatmap,
)
from research.param_explorer.sensitivity import (
    SensitivityAnalyzer,
    OATResult,
    SobolResult,
    MorrisResult,
)
from research.param_explorer.landscape import (
    ObjectiveLandscape,
    LandscapeGrid,
    Basin,
)
from research.param_explorer.bayesian_opt import (
    BayesianOptimizer,
    MOBayesianOptimizer,
    BayesOptResult,
    AcquisitionFunction,
    GPSurrogate,
)
from research.param_explorer.regime_sensitivity import (
    RegimeSensitivityAnalyzer,
)
from research.param_explorer.visualization import (
    ParamExplorerDashboard,
    create_sensitivity_dashboard,
    create_landscape_dashboard,
    create_bayesian_opt_dashboard,
    interactive_param_explorer,
)

__all__ = [
    # space
    "ParamSpec",
    "ParamSpace",
    "BHParamSpace",
    "LiveTraderParamSpace",
    "ParamType",
    "sensitivity_indices",
    "correlation_heatmap",
    # sensitivity
    "SensitivityAnalyzer",
    "OATResult",
    "SobolResult",
    "MorrisResult",
    # landscape
    "ObjectiveLandscape",
    "LandscapeGrid",
    "Basin",
    # bayesian_opt
    "BayesianOptimizer",
    "MOBayesianOptimizer",
    "BayesOptResult",
    "AcquisitionFunction",
    "GPSurrogate",
    # regime_sensitivity
    "RegimeSensitivityAnalyzer",
    # visualization
    "ParamExplorerDashboard",
    "create_sensitivity_dashboard",
    "create_landscape_dashboard",
    "create_bayesian_opt_dashboard",
    "interactive_param_explorer",
]
