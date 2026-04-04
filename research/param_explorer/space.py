"""
research/param_explorer/space.py
=================================
Parameter space definitions, sampling strategies, and first-pass sensitivity
helpers (correlation heatmap + Sobol-index wrapper).

Classes
-------
ParamType       : Enum of continuous / discrete / categorical
ParamSpec       : Dataclass describing a single parameter dimension
ParamSpace      : Collection of ParamSpec with sampling / transform helpers
BHParamSpace    : Concrete subclass for the Black-Hole (BH) engine parameters
LiveTraderParamSpace : Concrete subclass for live-trader tuning constants

Stand-alone helpers
-------------------
sensitivity_indices   : Fast Saltelli Sobol indices over any ParamSpace
correlation_heatmap   : Spearman correlation grid between params and objective
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr
from scipy.stats.qmc import Sobol, LatinHypercube, scale as qmc_scale

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ParamType
# ---------------------------------------------------------------------------

class ParamType(str, Enum):
    """Enumeration of supported parameter types."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


# ---------------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """
    Specification of a single parameter dimension.

    Attributes
    ----------
    name : str
        Human-readable identifier (must be unique within a ParamSpace).
    low : float
        Lower bound (for continuous/discrete).  Ignored for categorical.
    high : float
        Upper bound (for continuous/discrete).  Ignored for categorical.
    param_type : ParamType
        Nature of the parameter.
    log_scale : bool
        If True, sampling and transforms are performed in log space so that
        the parameter is distributed log-uniformly.  Only meaningful for
        continuous parameters where ``low > 0``.
    categories : list[Any]
        Ordered category values for categorical parameters.
    default : float | Any
        Default / baseline value used in OAT experiments.
    description : str
        Optional human-readable description.

    Notes
    -----
    For discrete parameters, ``low`` and ``high`` are inclusive integer
    boundaries.  Sampled values are rounded to the nearest integer.
    """

    name: str
    low: float = 0.0
    high: float = 1.0
    param_type: ParamType = ParamType.CONTINUOUS
    log_scale: bool = False
    categories: List[Any] = field(default_factory=list)
    default: Any = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.param_type == ParamType.CATEGORICAL:
            if not self.categories:
                raise ValueError(
                    f"ParamSpec '{self.name}': categories must be non-empty for categorical type."
                )
            self.low = 0.0
            self.high = float(len(self.categories) - 1)
            if self.default is None:
                self.default = self.categories[0]
        else:
            if self.low >= self.high:
                raise ValueError(
                    f"ParamSpec '{self.name}': low ({self.low}) must be < high ({self.high})."
                )
            if self.log_scale and self.low <= 0:
                raise ValueError(
                    f"ParamSpec '{self.name}': log_scale requires low > 0."
                )
            if self.default is None:
                self.default = (self.low + self.high) / 2.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unit_to_value(self, u: float) -> Any:
        """
        Map a unit-interval sample *u* ∈ [0, 1] → actual parameter value.
        """
        if self.param_type == ParamType.CATEGORICAL:
            idx = int(round(u * (len(self.categories) - 1)))
            idx = max(0, min(idx, len(self.categories) - 1))
            return self.categories[idx]

        if self.log_scale:
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            value = math.exp(log_low + u * (log_high - log_low))
        else:
            value = self.low + u * (self.high - self.low)

        if self.param_type == ParamType.DISCRETE:
            value = int(round(value))
            value = max(int(self.low), min(int(self.high), value))

        return value

    def _value_to_unit(self, value: Any) -> float:
        """
        Map an actual parameter *value* → unit-interval ∈ [0, 1].
        """
        if self.param_type == ParamType.CATEGORICAL:
            try:
                idx = self.categories.index(value)
            except ValueError:
                raise ValueError(
                    f"ParamSpec '{self.name}': value {value!r} not in categories."
                )
            return idx / max(len(self.categories) - 1, 1)

        if self.log_scale:
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            return (math.log(float(value)) - log_low) / (log_high - log_low)
        else:
            return (float(value) - self.low) / (self.high - self.low)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def sample_uniform_1d(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw *n* samples uniformly from this parameter's range.

        Returns
        -------
        np.ndarray of shape (n,) with actual (decoded) values.
        """
        u = rng.uniform(0.0, 1.0, size=n)
        return np.array([self._unit_to_value(ui) for ui in u])

    def clip(self, value: float) -> float:
        """Clip a raw value to [low, high]."""
        return float(np.clip(value, self.low, self.high))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["param_type"] = self.param_type.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParamSpec":
        d = dict(d)
        d["param_type"] = ParamType(d["param_type"])
        return cls(**d)

    def __repr__(self) -> str:
        if self.param_type == ParamType.CATEGORICAL:
            return (
                f"ParamSpec(name={self.name!r}, type=categorical, "
                f"categories={self.categories})"
            )
        scale = "log" if self.log_scale else "linear"
        return (
            f"ParamSpec(name={self.name!r}, [{self.low}, {self.high}], "
            f"type={self.param_type.value}, scale={scale})"
        )


# ---------------------------------------------------------------------------
# ParamSpace
# ---------------------------------------------------------------------------

class ParamSpace:
    """
    A collection of :class:`ParamSpec` objects that together define a
    multi-dimensional search space.

    Sampling methods return arrays in *unit hypercube* form and real-valued
    parameter dictionaries via ``to_params``.

    Parameters
    ----------
    specs : list[ParamSpec]
        Parameter specifications; names must be unique.
    name : str
        Optional label for this space (used in plot titles).
    """

    def __init__(
        self,
        specs: List[ParamSpec],
        name: str = "ParamSpace",
    ) -> None:
        self._validate_specs(specs)
        self.specs: List[ParamSpec] = list(specs)
        self.name = name
        self._name_to_idx: Dict[str, int] = {s.name: i for i, s in enumerate(specs)}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_specs(specs: List[ParamSpec]) -> None:
        seen: set[str] = set()
        for s in specs:
            if s.name in seen:
                raise ValueError(f"Duplicate param name: {s.name!r}")
            seen.add(s.name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_dims(self) -> int:
        """Number of parameter dimensions."""
        return len(self.specs)

    @property
    def names(self) -> List[str]:
        """Ordered list of parameter names."""
        return [s.name for s in self.specs]

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """List of (low, high) bounds in actual space (not unit space)."""
        return [(s.low, s.high) for s in self.specs]

    @property
    def unit_bounds(self) -> List[Tuple[float, float]]:
        """All bounds are (0, 1) in unit space."""
        return [(0.0, 1.0)] * self.n_dims

    @property
    def defaults(self) -> Dict[str, Any]:
        """Default parameter values as a dict."""
        return {s.name: s.default for s in self.specs}

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def to_params(self, array_row: np.ndarray) -> Dict[str, Any]:
        """
        Decode a row from unit hypercube into a parameter dict.

        Parameters
        ----------
        array_row : np.ndarray of shape (d,)

        Returns
        -------
        dict mapping param_name → decoded value
        """
        if len(array_row) != self.n_dims:
            raise ValueError(
                f"Expected array of length {self.n_dims}, got {len(array_row)}."
            )
        return {
            s.name: s._unit_to_value(float(array_row[i]))
            for i, s in enumerate(self.specs)
        }

    def from_params(self, param_dict: Dict[str, Any]) -> np.ndarray:
        """
        Encode a parameter dict into a unit-hypercube row.

        Parameters
        ----------
        param_dict : dict

        Returns
        -------
        np.ndarray of shape (d,)
        """
        row = np.zeros(self.n_dims)
        for i, s in enumerate(self.specs):
            if s.name not in param_dict:
                raise KeyError(f"Missing parameter {s.name!r} in param_dict.")
            row[i] = s._value_to_unit(param_dict[s.name])
        return row

    def decode_matrix(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode an (n, d) unit matrix into a list of n parameter dicts.
        """
        return [self.to_params(X[i]) for i in range(X.shape[0])]

    def encode_params_list(self, param_list: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode a list of parameter dicts into an (n, d) unit matrix.
        """
        return np.vstack([self.from_params(p) for p in param_list])

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_uniform(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Draw *n* samples from a uniform distribution over the unit hypercube.

        Returns
        -------
        np.ndarray of shape (n, d) in unit space.
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, size=(n, self.n_dims))

    def sample_sobol(self, n: int, seed: Optional[int] = 0) -> np.ndarray:
        """
        Quasi-random Sobol sequence sample.

        Parameters
        ----------
        n : int
            Number of samples.  Internally rounded up to next power-of-2 for
            optimal Sobol properties; extra rows are trimmed.
        seed : int | None

        Returns
        -------
        np.ndarray of shape (n, d) in unit space.
        """
        # Sobol works best with power-of-2 sizes; we overshoot and trim.
        n_pow2 = int(2 ** math.ceil(math.log2(max(n, 2))))
        sampler = Sobol(d=self.n_dims, scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = sampler.random(n_pow2)
        return samples[:n]

    def sample_latin_hypercube(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Latin Hypercube sample for better space-filling than uniform random.

        Returns
        -------
        np.ndarray of shape (n, d) in unit space.
        """
        sampler = LatinHypercube(d=self.n_dims, seed=seed)
        return sampler.random(n)

    def sample_grid(self, n_per_dim: int) -> np.ndarray:
        """
        Full factorial grid over the unit hypercube.

        Warning: scales as n_per_dim ** n_dims.  Use only for low-d spaces.

        Returns
        -------
        np.ndarray of shape (n_per_dim**d, d).
        """
        axes = [np.linspace(0.0, 1.0, n_per_dim)] * self.n_dims
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])

    # ------------------------------------------------------------------
    # Add / remove specs
    # ------------------------------------------------------------------

    def add(self, spec: ParamSpec) -> None:
        """Append a new :class:`ParamSpec` to this space."""
        if spec.name in self._name_to_idx:
            raise ValueError(f"Parameter {spec.name!r} already exists.")
        self._name_to_idx[spec.name] = len(self.specs)
        self.specs.append(spec)

    def remove(self, name: str) -> None:
        """Remove a parameter by name."""
        if name not in self._name_to_idx:
            raise KeyError(f"Parameter {name!r} not found.")
        idx = self._name_to_idx.pop(name)
        self.specs.pop(idx)
        # Rebuild index
        self._name_to_idx = {s.name: i for i, s in enumerate(self.specs)}

    def subset(self, names: List[str]) -> "ParamSpace":
        """Return a new ParamSpace containing only the named parameters."""
        specs = [self.specs[self._name_to_idx[n]] for n in names]
        return ParamSpace(specs, name=f"{self.name}[subset]")

    def __getitem__(self, name: str) -> ParamSpec:
        return self.specs[self._name_to_idx[name]]

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_idx

    def __len__(self) -> int:
        return self.n_dims

    def __repr__(self) -> str:
        lines = [f"ParamSpace(name={self.name!r}, n_dims={self.n_dims}):"]
        for s in self.specs:
            lines.append(f"  {s}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "specs": [s.to_dict() for s in self.specs]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParamSpace":
        specs = [ParamSpec.from_dict(s) for s in d["specs"]]
        return cls(specs, name=d.get("name", "ParamSpace"))

    def save_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "ParamSpace":
        d = json.loads(Path(path).read_text())
        return cls.from_dict(d)

    # ------------------------------------------------------------------
    # Random perturbation helpers
    # ------------------------------------------------------------------

    def perturb(
        self,
        param_dict: Dict[str, Any],
        scale: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Randomly perturb a parameter dict in unit space by ±``scale``.

        Parameters
        ----------
        param_dict : dict
        scale : float
            Standard deviation of perturbation in unit space.
        rng : np.random.Generator | None

        Returns
        -------
        dict with perturbed (clipped) parameter values.
        """
        if rng is None:
            rng = np.random.default_rng()
        row = self.from_params(param_dict)
        noise = rng.normal(0.0, scale, size=self.n_dims)
        row_noisy = np.clip(row + noise, 0.0, 1.0)
        return self.to_params(row_noisy)

    def linspace_1d(
        self, name: str, n: int, base_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Vary parameter ``name`` over its full range in *n* steps while
        holding all others at their default (or ``base_params``) values.

        Returns a list of n parameter dicts.
        """
        if base_params is None:
            base_params = self.defaults
        spec = self[name]
        values = np.linspace(spec.low, spec.high, n)
        if spec.log_scale:
            values = np.exp(
                np.linspace(math.log(spec.low), math.log(spec.high), n)
            )
        result = []
        for v in values:
            p = dict(base_params)
            p[name] = v
            result.append(p)
        return result


# ---------------------------------------------------------------------------
# BHParamSpace
# ---------------------------------------------------------------------------

class BHParamSpace(ParamSpace):
    """
    Concrete :class:`ParamSpace` for Black-Hole (BH) engine parameters.

    Parameters
    ----------
    cf_range : tuple
        (low, high) for the ``cf`` cost-fraction parameter.
    bh_form_range : tuple
        (low, high) for the BH formation threshold.
    bh_collapse_range : tuple
        (low, high) for the BH collapse trigger.
    bh_decay_range : tuple
        (low, high) for the momentum decay factor.
    include_derived : bool
        If True, include additional derived / secondary BH parameters.
    """

    # Canonical defaults from spec
    CF_DEFAULT: float = 0.002
    BH_FORM_DEFAULT: float = 1.6
    BH_COLLAPSE_DEFAULT: float = 0.8
    BH_DECAY_DEFAULT: float = 0.95

    def __init__(
        self,
        cf_range: Tuple[float, float] = (0.001, 0.003),
        bh_form_range: Tuple[float, float] = (1.2, 2.0),
        bh_collapse_range: Tuple[float, float] = (0.6, 1.0),
        bh_decay_range: Tuple[float, float] = (0.90, 0.99),
        include_derived: bool = True,
    ) -> None:
        specs = [
            ParamSpec(
                name="cf",
                low=cf_range[0],
                high=cf_range[1],
                param_type=ParamType.CONTINUOUS,
                log_scale=True,
                default=self.CF_DEFAULT,
                description="Cost fraction for BH engine signal scaling.",
            ),
            ParamSpec(
                name="bh_form",
                low=bh_form_range[0],
                high=bh_form_range[1],
                param_type=ParamType.CONTINUOUS,
                log_scale=False,
                default=self.BH_FORM_DEFAULT,
                description="BH formation threshold multiplier.",
            ),
            ParamSpec(
                name="bh_collapse",
                low=bh_collapse_range[0],
                high=bh_collapse_range[1],
                param_type=ParamType.CONTINUOUS,
                log_scale=False,
                default=self.BH_COLLAPSE_DEFAULT,
                description="BH collapse trigger ratio.",
            ),
            ParamSpec(
                name="bh_decay",
                low=bh_decay_range[0],
                high=bh_decay_range[1],
                param_type=ParamType.CONTINUOUS,
                log_scale=False,
                default=self.BH_DECAY_DEFAULT,
                description="Momentum decay factor α ∈ (0,1).",
            ),
        ]
        if include_derived:
            specs += [
                ParamSpec(
                    name="bh_radius_scale",
                    low=0.5,
                    high=3.0,
                    param_type=ParamType.CONTINUOUS,
                    log_scale=False,
                    default=1.0,
                    description="Schwarzschild radius scaling coefficient.",
                ),
                ParamSpec(
                    name="bh_event_horizon",
                    low=0.05,
                    high=0.50,
                    param_type=ParamType.CONTINUOUS,
                    log_scale=False,
                    default=0.20,
                    description="Event horizon fraction of BH radius.",
                ),
                ParamSpec(
                    name="bh_hawking_temp",
                    low=0.01,
                    high=0.50,
                    param_type=ParamType.CONTINUOUS,
                    log_scale=True,
                    default=0.10,
                    description="Hawking temperature analogue for noise injection.",
                ),
                ParamSpec(
                    name="bh_accretion_rate",
                    low=0.001,
                    high=0.10,
                    param_type=ParamType.CONTINUOUS,
                    log_scale=True,
                    default=0.02,
                    description="Rate at which BH absorbs position.",
                ),
            ]
        super().__init__(specs, name="BHParamSpace")

    def default_params(self) -> Dict[str, Any]:
        """Return the canonical BH parameter set."""
        return self.defaults


# ---------------------------------------------------------------------------
# LiveTraderParamSpace
# ---------------------------------------------------------------------------

class LiveTraderParamSpace(ParamSpace):
    """
    Concrete :class:`ParamSpace` for live-trader tuning constants.

    Parameters
    ----------
    delta_max_frac_range : tuple
        (low, high) for DELTA_MAX_FRAC.
    min_trade_frac_range : tuple
        (low, high) for MIN_TRADE_FRAC.
    min_hold_range : tuple
        (low, high) for MIN_HOLD (discrete steps).
    include_ensemble_weights : bool
        If True, add D3QN / DDQN / TD3QN ensemble weight parameters.
    """

    DELTA_MAX_FRAC_DEFAULT: float = 0.7
    MIN_TRADE_FRAC_DEFAULT: float = 0.02
    MIN_HOLD_DEFAULT: int = 3

    def __init__(
        self,
        delta_max_frac_range: Tuple[float, float] = (0.5, 0.9),
        min_trade_frac_range: Tuple[float, float] = (0.01, 0.05),
        min_hold_range: Tuple[float, float] = (1.0, 5.0),
        include_ensemble_weights: bool = True,
    ) -> None:
        specs = [
            ParamSpec(
                name="DELTA_MAX_FRAC",
                low=delta_max_frac_range[0],
                high=delta_max_frac_range[1],
                param_type=ParamType.CONTINUOUS,
                log_scale=False,
                default=self.DELTA_MAX_FRAC_DEFAULT,
                description="Maximum allowed position delta as fraction of capital.",
            ),
            ParamSpec(
                name="MIN_TRADE_FRAC",
                low=min_trade_frac_range[0],
                high=min_trade_frac_range[1],
                param_type=ParamType.CONTINUOUS,
                log_scale=True,
                default=self.MIN_TRADE_FRAC_DEFAULT,
                description="Minimum trade size as fraction of capital.",
            ),
            ParamSpec(
                name="MIN_HOLD",
                low=min_hold_range[0],
                high=min_hold_range[1],
                param_type=ParamType.DISCRETE,
                log_scale=False,
                default=self.MIN_HOLD_DEFAULT,
                description="Minimum holding period in bars.",
            ),
        ]
        if include_ensemble_weights:
            # Ensemble weights must sum to 1; we parameterise via Dirichlet
            # simplex projections (3 free weights for D3QN / DDQN / TD3QN).
            specs += [
                ParamSpec(
                    name="w_d3qn",
                    low=0.0,
                    high=1.0,
                    param_type=ParamType.CONTINUOUS,
                    log_scale=False,
                    default=0.334,
                    description="D3QN ensemble weight (normalised post-sampling).",
                ),
                ParamSpec(
                    name="w_ddqn",
                    low=0.0,
                    high=1.0,
                    param_type=ParamType.CONTINUOUS,
                    log_scale=False,
                    default=0.333,
                    description="DDQN ensemble weight (normalised post-sampling).",
                ),
                ParamSpec(
                    name="w_td3qn",
                    low=0.0,
                    high=1.0,
                    param_type=ParamType.CONTINUOUS,
                    log_scale=False,
                    default=0.333,
                    description="TD3QN ensemble weight (normalised post-sampling).",
                ),
                ParamSpec(
                    name="ensemble_temp",
                    low=0.1,
                    high=5.0,
                    param_type=ParamType.CONTINUOUS,
                    log_scale=True,
                    default=1.0,
                    description="Softmax temperature for ensemble blending.",
                ),
            ]
        super().__init__(specs, name="LiveTraderParamSpace")

    def normalise_weights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise ensemble weights so they sum to 1.

        Returns a copy of *params* with w_d3qn / w_ddqn / w_td3qn normalised.
        """
        params = dict(params)
        weight_keys = ["w_d3qn", "w_ddqn", "w_td3qn"]
        if not all(k in params for k in weight_keys):
            return params
        total = sum(params[k] for k in weight_keys)
        if total <= 0:
            for k in weight_keys:
                params[k] = 1.0 / 3.0
        else:
            for k in weight_keys:
                params[k] = params[k] / total
        return params


# ---------------------------------------------------------------------------
# CombinedParamSpace helper
# ---------------------------------------------------------------------------

class CombinedParamSpace(ParamSpace):
    """
    Merge multiple :class:`ParamSpace` instances into one flat space.

    Useful for joint optimisation of BH engine + live trader parameters.
    """

    def __init__(self, spaces: List[ParamSpace], name: str = "CombinedParamSpace") -> None:
        all_specs: List[ParamSpec] = []
        seen: set[str] = set()
        for space in spaces:
            for spec in space.specs:
                if spec.name in seen:
                    logger.warning(
                        "Duplicate parameter %r when merging spaces; skipping.", spec.name
                    )
                    continue
                all_specs.append(spec)
                seen.add(spec.name)
        super().__init__(all_specs, name=name)


# ---------------------------------------------------------------------------
# Stand-alone sensitivity helpers
# ---------------------------------------------------------------------------

def sensitivity_indices(
    space: ParamSpace,
    objective_fn: Callable[[Dict[str, Any]], float],
    n_samples: int = 1000,
    seed: int = 42,
    conf_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute first-order (Si) and total-order (STi) Sobol sensitivity indices
    using the Saltelli (2010) estimator.

    The function requires 2*(d+1)*n_samples evaluations of ``objective_fn``.
    For expensive objectives, reduce *n_samples* or use
    :func:`research.param_explorer.sensitivity.morris_screening` instead.

    Parameters
    ----------
    space : ParamSpace
    objective_fn : callable
        Maps a parameter dict → scalar float.
    n_samples : int
        Base sample count N.  Total evaluations = N*(2d+2).
    seed : int
    conf_level : float
        Confidence level for bootstrap confidence intervals.

    Returns
    -------
    dict with keys:
        ``Si``   : dict[param_name → first-order index]
        ``STi``  : dict[param_name → total-order index]
        ``Si_conf``  : dict[param_name → bootstrap half-width]
        ``STi_conf`` : dict[param_name → bootstrap half-width]
        ``var_y``    : total output variance
        ``n_evals``  : total number of objective evaluations
    """
    d = space.n_dims
    rng = np.random.default_rng(seed)

    # Build A and B matrices (n_samples × d) in unit space
    A = space.sample_sobol(n_samples, seed=seed)
    B = space.sample_sobol(n_samples, seed=seed + 1)

    # Evaluate A and B
    y_A = np.array([objective_fn(space.to_params(A[i])) for i in range(n_samples)])
    y_B = np.array([objective_fn(space.to_params(B[i])) for i in range(n_samples)])

    # Build A_B matrices: replace column j of A with column j of B
    y_AB: Dict[int, np.ndarray] = {}
    for j in range(d):
        A_Bj = A.copy()
        A_Bj[:, j] = B[:, j]
        y_AB[j] = np.array(
            [objective_fn(space.to_params(A_Bj[i])) for i in range(n_samples)]
        )

    # Saltelli 2010 estimators
    f0_sq = np.mean(y_A) * np.mean(y_B)
    var_y = np.var(np.concatenate([y_A, y_B]), ddof=1)
    if var_y < 1e-30:
        logger.warning("Total output variance is near zero; sensitivity indices meaningless.")
        var_y = 1e-30

    Si: Dict[str, float] = {}
    STi: Dict[str, float] = {}
    Si_conf: Dict[str, float] = {}
    STi_conf: Dict[str, float] = {}

    # Bootstrap for confidence intervals
    n_boot = 200
    boot_idx = rng.integers(0, n_samples, size=(n_boot, n_samples))

    for j, spec in enumerate(space.specs):
        yj = y_AB[j]

        # First-order: Si = (1/N) Σ y_B(y_AB_j - y_A) / Var(Y)
        si_vals = y_B * (yj - y_A) / var_y
        si_mean = float(np.mean(si_vals))

        # Total-order: STi = (1/2N) Σ (y_A - y_AB_j)^2 / Var(Y)
        sti_vals = (y_A - yj) ** 2 / (2.0 * var_y)
        sti_mean = float(np.mean(sti_vals))

        Si[spec.name] = si_mean
        STi[spec.name] = sti_mean

        # Bootstrap CI
        si_boot = np.array([np.mean(si_vals[boot_idx[b]]) for b in range(n_boot)])
        sti_boot = np.array([np.mean(sti_vals[boot_idx[b]]) for b in range(n_boot)])
        alpha = 1.0 - conf_level
        Si_conf[spec.name] = float(
            np.percentile(si_boot, 100 * (1 - alpha / 2))
            - np.percentile(si_boot, 100 * (alpha / 2))
        ) / 2.0
        STi_conf[spec.name] = float(
            np.percentile(sti_boot, 100 * (1 - alpha / 2))
            - np.percentile(sti_boot, 100 * (alpha / 2))
        ) / 2.0

    n_evals = n_samples * (d + 2)
    return {
        "Si": Si,
        "STi": STi,
        "Si_conf": Si_conf,
        "STi_conf": STi_conf,
        "var_y": float(var_y),
        "n_evals": n_evals,
    }


def correlation_heatmap(
    space: ParamSpace,
    objective_fn: Callable[[Dict[str, Any]], float],
    n_samples: int = 500,
    seed: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """
    Compute pairwise Spearman correlations between parameters and the
    objective, then render a heatmap.

    Parameters
    ----------
    space : ParamSpace
    objective_fn : callable
    n_samples : int
    seed : int
    save_path : str | Path | None
        Where to save the figure.  ``None`` = do not save.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    X = space.sample_latin_hypercube(n_samples, seed=seed)
    param_list = space.decode_matrix(X)
    y = np.array([objective_fn(p) for p in param_list])

    # Build (n × d+1) matrix of unit-space param values + objective
    data = np.column_stack([X, y])
    col_names = space.names + ["objective"]
    n_cols = data.shape[1]

    corr_matrix = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                rho, _ = spearmanr(data[:, i], data[:, j])
                corr_matrix[i, j] = float(rho)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_cols))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(col_names, fontsize=8)
    ax.set_title(f"Spearman Correlation Heatmap — {space.name}\n(n={n_samples} LHS samples)")

    # Annotate cells
    for i in range(n_cols):
        for j in range(n_cols):
            ax.text(
                j, i, f"{corr_matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=6,
                color="white" if abs(corr_matrix[i, j]) > 0.6 else "black",
            )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Correlation heatmap saved to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Utility: pretty-print space summary
# ---------------------------------------------------------------------------

def summarise_space(space: ParamSpace) -> str:
    """Return a human-readable string summary of the parameter space."""
    lines = [
        f"{'─'*60}",
        f"  ParamSpace: {space.name}  ({space.n_dims} dimensions)",
        f"{'─'*60}",
        f"  {'Name':<25} {'Type':<12} {'Low':>10} {'High':>10} {'Default':>10}",
        f"  {'─'*25} {'─'*12} {'─'*10} {'─'*10} {'─'*10}",
    ]
    for s in space.specs:
        if s.param_type == ParamType.CATEGORICAL:
            lines.append(
                f"  {s.name:<25} {'categorical':<12} "
                f"{'—':>10} {'—':>10} {str(s.default):>10}"
            )
        else:
            scale_tag = " (log)" if s.log_scale else ""
            lines.append(
                f"  {s.name:<25} {s.param_type.value + scale_tag:<12} "
                f"{s.low:>10.4g} {s.high:>10.4g} {str(s.default):>10}"
            )
    lines.append(f"{'─'*60}")
    return "\n".join(lines)
