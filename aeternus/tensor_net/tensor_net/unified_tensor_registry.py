"""
unified_tensor_registry.py — Unified Tensor Registry (UTR) for Project AETERNUS.

Defines canonical tensor schemas for all inter-module data exchange:
  - ChronosOutput: market tick data from Chronos module
  - NeuroSDEState: hidden state from Neural SDE module
  - OmniGraphAdjacency: sparse adjacency from OmniGraph module
  - LuminaPrediction: predictions from Lumina module
  - HyperAgentAction: actions from HyperAgent module

Provides:
  - Schema registration / lookup
  - Validation (dtype, shape, NaN/Inf, version)
  - Broadcasting rules between schemas
  - Conversion utilities (ChronosOutput -> TT input, adjacency -> edge_index)
  - Backward-compatible schema versioning
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Sequence,
    Set, Tuple, Type, Union
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

UTR_VERSION = "1.4.0"
SCHEMA_COMPAT_TABLE: Dict[str, List[str]] = {
    "1.4.0": ["1.3.0", "1.2.0"],
    "1.3.0": ["1.2.0", "1.1.0"],
    "1.2.0": ["1.1.0"],
    "1.1.0": [],
}


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP: Dict[str, np.dtype] = {
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "int8":    np.dtype("int8"),
    "int16":   np.dtype("int16"),
    "int32":   np.dtype("int32"),
    "int64":   np.dtype("int64"),
    "bool":    np.dtype("bool"),
}


def _resolve_dtype(dtype: Union[str, np.dtype]) -> np.dtype:
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, str):
        if dtype in _DTYPE_MAP:
            return _DTYPE_MAP[dtype]
        return np.dtype(dtype)
    raise TypeError(f"Cannot resolve dtype from {type(dtype)}: {dtype!r}")


# ---------------------------------------------------------------------------
# Shape specification
# ---------------------------------------------------------------------------

class DimSpec:
    """A single dimension in a tensor shape — can be fixed, symbolic, or wildcard."""

    WILDCARD = -1  # matches any size

    def __init__(self, value: Union[int, str]):
        if isinstance(value, int):
            self._int = value
            self._sym: Optional[str] = None
        elif isinstance(value, str):
            self._int = self.WILDCARD
            self._sym = value
        else:
            raise TypeError(f"DimSpec value must be int or str, got {type(value)}")

    @property
    def is_fixed(self) -> bool:
        return self._int != self.WILDCARD and self._sym is None

    @property
    def is_symbolic(self) -> bool:
        return self._sym is not None

    @property
    def is_wildcard(self) -> bool:
        return self._int == self.WILDCARD and self._sym is None

    def matches(self, size: int, bindings: Optional[Dict[str, int]] = None) -> bool:
        if self.is_fixed:
            return self._int == size
        if self.is_wildcard:
            return True
        if self.is_symbolic:
            if bindings is None:
                return True  # can't check without bindings
            sym_val = bindings.get(self._sym)  # type: ignore[arg-type]
            if sym_val is None:
                bindings[self._sym] = size  # type: ignore[index]
                return True
            return sym_val == size
        return False

    def __repr__(self) -> str:
        if self.is_fixed:
            return str(self._int)
        if self.is_wildcard:
            return "*"
        return f"<{self._sym}>"


class ShapeSpec:
    """Tensor shape specification supporting fixed dims, symbolic dims, and wildcards."""

    def __init__(self, *dims: Union[int, str]):
        self._dims: Tuple[DimSpec, ...] = tuple(DimSpec(d) for d in dims)

    @property
    def ndim(self) -> int:
        return len(self._dims)

    def matches(self, shape: Tuple[int, ...]) -> Tuple[bool, str]:
        if len(shape) != self.ndim:
            return False, f"ndim mismatch: expected {self.ndim}, got {len(shape)}"
        bindings: Dict[str, int] = {}
        for i, (spec, size) in enumerate(zip(self._dims, shape)):
            if not spec.matches(size, bindings):
                return False, (
                    f"dim[{i}] mismatch: spec={spec!r}, actual={size}, "
                    f"bindings={bindings}"
                )
        return True, ""

    def __repr__(self) -> str:
        inner = ", ".join(repr(d) for d in self._dims)
        return f"ShapeSpec({inner})"


# ---------------------------------------------------------------------------
# Value range / constraint specifications
# ---------------------------------------------------------------------------

@dataclass
class RangeConstraint:
    low: Optional[float] = None
    high: Optional[float] = None
    allow_nan: bool = False
    allow_inf: bool = False

    def check(self, arr: np.ndarray) -> List[str]:
        errors: List[str] = []
        if not self.allow_nan and np.any(np.isnan(arr)):
            n_nan = int(np.sum(np.isnan(arr)))
            errors.append(f"contains {n_nan} NaN values")
        if not self.allow_inf and np.any(np.isinf(arr)):
            n_inf = int(np.sum(np.isinf(arr)))
            errors.append(f"contains {n_inf} Inf values")
        finite = arr[np.isfinite(arr)]
        if len(finite) > 0:
            if self.low is not None and float(finite.min()) < self.low:
                errors.append(
                    f"min value {float(finite.min()):.6g} < constraint low {self.low}"
                )
            if self.high is not None and float(finite.max()) > self.high:
                errors.append(
                    f"max value {float(finite.max()):.6g} > constraint high {self.high}"
                )
        return errors


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

@dataclass
class TensorSchema:
    """
    Full specification for a named tensor type in the AETERNUS pipeline.

    Attributes
    ----------
    name:
        Canonical schema name, e.g. "ChronosOutput".
    shape_spec:
        Shape specification (fixed, symbolic, or wildcard dims).
    dtype:
        Required NumPy dtype.
    feature_names:
        Optional ordered list of feature/channel names for the last axis.
    range_constraint:
        Optional numeric range / NaN / Inf policy.
    sparse:
        True if this is a sparse tensor (CSR adjacency etc.).
    version:
        Schema version string (semver).
    description:
        Human-readable description.
    tags:
        Arbitrary string tags for filtering / routing.
    """

    name: str
    shape_spec: ShapeSpec
    dtype: np.dtype
    feature_names: Optional[List[str]] = None
    range_constraint: Optional[RangeConstraint] = None
    sparse: bool = False
    version: str = UTR_VERSION
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(
        self,
        arr: np.ndarray,
        *,
        check_values: bool = True,
        strict_dtype: bool = True,
    ) -> "ValidationResult":
        errors: List[str] = []
        warnings_list: List[str] = []

        # dtype check
        if strict_dtype:
            if arr.dtype != self.dtype:
                errors.append(
                    f"dtype mismatch: expected {self.dtype}, got {arr.dtype}"
                )
        else:
            if not np.can_cast(arr.dtype, self.dtype):
                warnings_list.append(
                    f"dtype {arr.dtype} cannot be safely cast to {self.dtype}"
                )

        # shape check
        ok, msg = self.shape_spec.matches(arr.shape)
        if not ok:
            errors.append(f"shape error: {msg}")

        # value range / NaN / Inf
        if check_values and self.range_constraint is not None:
            errors.extend(self.range_constraint.check(arr))

        return ValidationResult(
            schema_name=self.name,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings_list,
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        dims = []
        for d in self.shape_spec._dims:
            if d.is_fixed:
                dims.append(d._int)
            elif d.is_wildcard:
                dims.append("*")
            else:
                dims.append(d._sym)
        return {
            "name": self.name,
            "shape": dims,
            "dtype": str(self.dtype),
            "feature_names": self.feature_names,
            "sparse": self.sparse,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorSchema":
        shape_spec = ShapeSpec(*d["shape"])
        dtype = _resolve_dtype(d["dtype"])
        rc = None
        if "range_constraint" in d and d["range_constraint"]:
            rc = RangeConstraint(**d["range_constraint"])
        return cls(
            name=d["name"],
            shape_spec=shape_spec,
            dtype=dtype,
            feature_names=d.get("feature_names"),
            range_constraint=rc,
            sparse=d.get("sparse", False),
            version=d.get("version", UTR_VERSION),
            description=d.get("description", ""),
            tags=d.get("tags", []),
        )


@dataclass
class ValidationResult:
    schema_name: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            detail = "; ".join(self.errors)
            raise ValueError(
                f"Schema '{self.schema_name}' validation failed: {detail}"
            )

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"ValidationResult({self.schema_name}, {status}"]
        if self.errors:
            parts.append(f", errors={self.errors}")
        if self.warnings:
            parts.append(f", warnings={self.warnings}")
        parts.append(")")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Built-in canonical schemas
# ---------------------------------------------------------------------------

#: ChronosOutput: (N_assets, T_ticks, 6) float32
CHRONOS_OUTPUT_SCHEMA = TensorSchema(
    name="ChronosOutput",
    shape_spec=ShapeSpec("N_assets", "T_ticks", 6),
    dtype=np.dtype("float32"),
    feature_names=["bid", "ask", "mid", "spread", "volume", "imbalance"],
    range_constraint=RangeConstraint(low=0.0, allow_nan=False, allow_inf=False),
    version=UTR_VERSION,
    description=(
        "Raw market data from the Chronos module. Each tick contains bid, ask, "
        "mid price, bid-ask spread, volume, and order-book imbalance."
    ),
    tags=["market_data", "chronos", "raw"],
)

#: NeuroSDEState: (N_assets, latent_dim) float32
NEURO_SDE_STATE_SCHEMA = TensorSchema(
    name="NeuroSDEState",
    shape_spec=ShapeSpec("N_assets", "latent_dim"),
    dtype=np.dtype("float32"),
    range_constraint=RangeConstraint(allow_nan=False, allow_inf=False),
    version=UTR_VERSION,
    description=(
        "Current hidden state of the Neural SDE module. Encodes latent "
        "stochastic dynamics for each asset."
    ),
    tags=["latent_state", "neuro_sde"],
)

#: OmniGraphAdjacency: sparse CSR (N_assets x N_assets) float32
OMNI_GRAPH_ADJACENCY_SCHEMA = TensorSchema(
    name="OmniGraphAdjacency",
    shape_spec=ShapeSpec("N_assets", "N_assets"),
    dtype=np.dtype("float32"),
    sparse=True,
    range_constraint=RangeConstraint(low=0.0, high=1.0, allow_nan=False, allow_inf=False),
    version=UTR_VERSION,
    description=(
        "Sparse adjacency matrix from OmniGraph module. CSR format, "
        "values represent edge weights in [0, 1]."
    ),
    tags=["graph", "omni_graph", "adjacency", "sparse"],
)

#: LuminaPrediction: (N_assets, horizon, 3) float32
LUMINA_PREDICTION_SCHEMA = TensorSchema(
    name="LuminaPrediction",
    shape_spec=ShapeSpec("N_assets", "horizon", 3),
    dtype=np.dtype("float32"),
    feature_names=["direction_prob", "vol_forecast", "regime_logits"],
    range_constraint=RangeConstraint(allow_nan=False, allow_inf=False),
    version=UTR_VERSION,
    description=(
        "Predictions from the Lumina forecasting module. For each asset "
        "and horizon step: directional probability, volatility forecast, "
        "and regime logits."
    ),
    tags=["prediction", "lumina", "forecast"],
)

#: HyperAgentAction: (N_agents, action_dim) float32
HYPER_AGENT_ACTION_SCHEMA = TensorSchema(
    name="HyperAgentAction",
    shape_spec=ShapeSpec("N_agents", "action_dim"),
    dtype=np.dtype("float32"),
    range_constraint=RangeConstraint(allow_nan=False, allow_inf=False),
    version=UTR_VERSION,
    description=(
        "Action tensor output from HyperAgent module. Each agent produces "
        "a vector of continuous or discretized actions."
    ),
    tags=["action", "hyper_agent", "rl"],
)

# List of all built-in schemas (order matters for registry initialization)
BUILTIN_SCHEMAS: List[TensorSchema] = [
    CHRONOS_OUTPUT_SCHEMA,
    NEURO_SDE_STATE_SCHEMA,
    OMNI_GRAPH_ADJACENCY_SCHEMA,
    LUMINA_PREDICTION_SCHEMA,
    HYPER_AGENT_ACTION_SCHEMA,
]


# ---------------------------------------------------------------------------
# Broadcasting rules
# ---------------------------------------------------------------------------

@dataclass
class BroadcastRule:
    """Defines how tensor A can be broadcast/adapted to match schema B."""
    source_schema: str
    target_schema: str
    description: str
    transform: Callable[[np.ndarray], np.ndarray]


def _chronos_to_neuro_sde_broadcast(arr: np.ndarray) -> np.ndarray:
    """
    ChronosOutput (N, T, 6) -> NeuroSDEState (N, T*6) via reshape.
    Treat last-tick features concatenated as a flat latent vector.
    """
    N, T, F = arr.shape
    return arr[:, -1, :].reshape(N, F)  # use last tick, shape (N, 6)


def _chronos_to_lumina_broadcast(arr: np.ndarray) -> np.ndarray:
    """
    ChronosOutput (N, T, 6) -> partial LuminaPrediction input.
    Returns (N, 1, 3) using mid, vol-proxy, imbalance as logit proxy.
    """
    mid = arr[:, -1:, 2:3]
    vol = arr[:, -1:, 4:5]
    imb = arr[:, -1:, 5:6]
    return np.concatenate([mid, vol, imb], axis=-1).astype(np.float32)


_BROADCAST_RULES: List[BroadcastRule] = [
    BroadcastRule(
        source_schema="ChronosOutput",
        target_schema="NeuroSDEState",
        description="Use last-tick features as flat latent vector",
        transform=_chronos_to_neuro_sde_broadcast,
    ),
    BroadcastRule(
        source_schema="ChronosOutput",
        target_schema="LuminaPrediction",
        description="Use mid/vol/imbalance from last tick as proxy prediction",
        transform=_chronos_to_lumina_broadcast,
    ),
]


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def chronos_to_tt_input(
    arr: np.ndarray,
    *,
    reshape_as: Tuple[int, ...] = (-1,),
) -> np.ndarray:
    """
    Convert a ChronosOutput tensor to an input suitable for TT decomposition.

    ChronosOutput shape: (N_assets, T_ticks, 6)
    Returns a reshaped float32 array.

    Parameters
    ----------
    arr:
        ChronosOutput array with shape (N, T, 6).
    reshape_as:
        Target shape per asset. Default (-1,) flattens to (N, T*6).
        Pass e.g. (T, 2, 3) for a structured TT input.

    Returns
    -------
    np.ndarray of shape (N, *reshape_as) or (N, T*6) if reshape_as==(-1,).
    """
    CHRONOS_OUTPUT_SCHEMA.validate(arr).raise_if_failed()
    N, T, F = arr.shape
    if reshape_as == (-1,):
        return arr.reshape(N, T * F).astype(np.float32)
    target_size = int(np.prod([s for s in reshape_as if s != -1]))
    if T * F % target_size != 0:
        raise ValueError(
            f"Cannot reshape (N, T={T}, F={F}) -> (N, {reshape_as}): "
            f"T*F={T*F} not divisible by target_size={target_size}"
        )
    return arr.reshape((N,) + reshape_as).astype(np.float32)


def omni_graph_to_edge_index(
    adj: np.ndarray,
    *,
    threshold: float = 0.0,
    add_self_loops: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a dense OmniGraphAdjacency matrix to COO edge_index format.

    Parameters
    ----------
    adj:
        Dense adjacency matrix (N_assets, N_assets), float32.
    threshold:
        Edges with weight <= threshold are excluded.
    add_self_loops:
        If True, add self-loops (i -> i) with weight 1.0.

    Returns
    -------
    edge_index: np.ndarray of shape (2, E), int64, rows/cols of non-zero edges.
    edge_weight: np.ndarray of shape (E,), float32, edge weights.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square 2D, got shape {adj.shape}")
    mask = adj > threshold
    if add_self_loops:
        np.fill_diagonal(mask, True)
        np.fill_diagonal(adj, np.maximum(adj.diagonal(), 1.0))
    rows, cols = np.where(mask)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_weight = adj[rows, cols].astype(np.float32)
    return edge_index, edge_weight


def sparse_csr_to_edge_index(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert CSR sparse matrix components to COO edge_index.

    Parameters
    ----------
    data: non-zero values.
    indices: column indices.
    indptr: row pointer array.
    shape: (n_rows, n_cols).

    Returns
    -------
    edge_index: (2, E) int64
    edge_weight: (E,) float32
    """
    n_rows = shape[0]
    rows = np.repeat(np.arange(n_rows, dtype=np.int64),
                     np.diff(indptr.astype(np.int64)))
    cols = indices.astype(np.int64)
    edge_index = np.stack([rows, cols], axis=0)
    edge_weight = data.astype(np.float32)
    return edge_index, edge_weight


def lumina_to_signal_map(
    pred: np.ndarray,
    asset_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convert a LuminaPrediction tensor to a human-readable signal map.

    Parameters
    ----------
    pred:
        LuminaPrediction array, shape (N_assets, horizon, 3).
    asset_ids:
        Optional list of asset identifier strings.

    Returns
    -------
    dict mapping asset_id -> {"direction_prob": [...], "vol_forecast": [...],
                               "regime_logits": [...]}
    """
    LUMINA_PREDICTION_SCHEMA.validate(pred).raise_if_failed()
    N, H, _ = pred.shape
    if asset_ids is None:
        asset_ids = [f"asset_{i}" for i in range(N)]
    result: Dict[str, Any] = {}
    for i, aid in enumerate(asset_ids):
        result[aid] = {
            "direction_prob": pred[i, :, 0].tolist(),
            "vol_forecast": pred[i, :, 1].tolist(),
            "regime_logits": pred[i, :, 2].tolist(),
        }
    return result


# ---------------------------------------------------------------------------
# Schema Registry
# ---------------------------------------------------------------------------

class UTRRegistryError(Exception):
    """Raised for registry-level errors."""


class SchemaVersionError(UTRRegistryError):
    """Raised when an incompatible schema version is used."""


class UnifiedTensorRegistry:
    """
    Central registry for AETERNUS tensor schemas.

    Thread-safety: this class is NOT thread-safe for concurrent registration.
    Concurrent reads (validate, lookup) are safe for typical CPython usage.

    Usage
    -----
    >>> reg = UnifiedTensorRegistry()
    >>> reg.register(MY_SCHEMA)
    >>> schema = reg.lookup("MySchema")
    >>> result = reg.validate("MySchema", my_array)
    """

    _instance: Optional["UnifiedTensorRegistry"] = None

    def __init__(self, *, auto_register_builtins: bool = True) -> None:
        self._schemas: Dict[str, TensorSchema] = {}
        self._version_history: Dict[str, List[str]] = {}  # name -> list[version]
        self._broadcast_rules: Dict[Tuple[str, str], BroadcastRule] = {}
        self._stats: Dict[str, Dict[str, int]] = {}  # name -> {validates, hits, misses}

        if auto_register_builtins:
            for schema in BUILTIN_SCHEMAS:
                self.register(schema)
            for rule in _BROADCAST_RULES:
                self.register_broadcast_rule(rule)

    # ------------------------------------------------------------------ #
    # Singleton accessor
    # ------------------------------------------------------------------ #

    @classmethod
    def global_registry(cls) -> "UnifiedTensorRegistry":
        """Return (or lazily create) the process-wide singleton registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(
        self,
        schema: TensorSchema,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a schema. Raises if name already exists unless overwrite=True."""
        name = schema.name
        if name in self._schemas and not overwrite:
            existing_ver = self._schemas[name].version
            if existing_ver == schema.version:
                logger.debug("Schema '%s' v%s already registered, skipping.", name, schema.version)
                return
            raise UTRRegistryError(
                f"Schema '{name}' already registered at version {existing_ver}. "
                f"Use overwrite=True or register under a new name."
            )
        self._schemas[name] = schema
        hist = self._version_history.setdefault(name, [])
        if schema.version not in hist:
            hist.append(schema.version)
        self._stats[name] = {"validates": 0, "passes": 0, "failures": 0}
        logger.debug("Registered schema '%s' v%s.", name, schema.version)

    def deregister(self, name: str) -> None:
        """Remove a schema from the registry."""
        if name not in self._schemas:
            raise UTRRegistryError(f"Schema '{name}' not found in registry.")
        del self._schemas[name]
        del self._stats[name]
        logger.debug("Deregistered schema '%s'.", name)

    def register_broadcast_rule(self, rule: BroadcastRule) -> None:
        """Register a broadcasting rule between two schemas."""
        key = (rule.source_schema, rule.target_schema)
        self._broadcast_rules[key] = rule
        logger.debug(
            "Registered broadcast rule %s -> %s.", rule.source_schema, rule.target_schema
        )

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def lookup(self, name: str) -> TensorSchema:
        """Return the schema for *name*. Raises KeyError if not found."""
        if name not in self._schemas:
            available = sorted(self._schemas.keys())
            raise KeyError(
                f"Schema '{name}' not in registry. Available: {available}"
            )
        return self._schemas[name]

    def list_schemas(self, *, tags: Optional[List[str]] = None) -> List[str]:
        """List all registered schema names, optionally filtered by tags."""
        if tags is None:
            return sorted(self._schemas.keys())
        result = []
        for name, schema in self._schemas.items():
            if any(t in schema.tags for t in tags):
                result.append(name)
        return sorted(result)

    def has_schema(self, name: str) -> bool:
        return name in self._schemas

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(
        self,
        schema_name: str,
        arr: np.ndarray,
        *,
        check_values: bool = True,
        strict_dtype: bool = True,
        raise_on_failure: bool = False,
    ) -> ValidationResult:
        """Validate *arr* against a named schema."""
        schema = self.lookup(schema_name)
        result = schema.validate(arr, check_values=check_values, strict_dtype=strict_dtype)

        stats = self._stats[schema_name]
        stats["validates"] += 1
        if result.passed:
            stats["passes"] += 1
        else:
            stats["failures"] += 1

        if raise_on_failure:
            result.raise_if_failed()
        return result

    def validate_batch(
        self,
        tensors: Dict[str, np.ndarray],
        *,
        raise_on_first_failure: bool = False,
    ) -> Dict[str, ValidationResult]:
        """Validate multiple named tensors in one call."""
        results: Dict[str, ValidationResult] = {}
        for name, arr in tensors.items():
            r = self.validate(name, arr)
            results[name] = r
            if raise_on_first_failure and not r.passed:
                r.raise_if_failed()
        return results

    # ------------------------------------------------------------------ #
    # Version compatibility
    # ------------------------------------------------------------------ #

    def check_version_compat(
        self,
        schema_name: str,
        producer_version: str,
    ) -> bool:
        """
        Return True if a tensor produced at *producer_version* is compatible
        with the currently registered schema version.
        """
        schema = self.lookup(schema_name)
        current_ver = schema.version
        if current_ver == producer_version:
            return True
        compat = SCHEMA_COMPAT_TABLE.get(current_ver, [])
        return producer_version in compat

    def migrate_tensor(
        self,
        schema_name: str,
        arr: np.ndarray,
        from_version: str,
    ) -> np.ndarray:
        """
        Attempt to migrate *arr* from an older schema version to the current one.
        Currently handles dtype promotion and last-dim padding/truncation.
        """
        schema = self.lookup(schema_name)
        if not self.check_version_compat(schema_name, from_version):
            raise SchemaVersionError(
                f"Schema '{schema_name}': version {from_version} is not compatible "
                f"with current {schema.version}"
            )
        # Cast dtype if needed
        if arr.dtype != schema.dtype:
            logger.info(
                "Migrating '%s': casting dtype %s -> %s.",
                schema_name, arr.dtype, schema.dtype
            )
            arr = arr.astype(schema.dtype)
        return arr

    # ------------------------------------------------------------------ #
    # Broadcasting
    # ------------------------------------------------------------------ #

    def can_broadcast(self, source: str, target: str) -> bool:
        return (source, target) in self._broadcast_rules

    def broadcast(
        self,
        source_schema: str,
        target_schema: str,
        arr: np.ndarray,
    ) -> np.ndarray:
        """Apply a registered broadcast rule to *arr*."""
        key = (source_schema, target_schema)
        if key not in self._broadcast_rules:
            raise UTRRegistryError(
                f"No broadcast rule registered for {source_schema} -> {target_schema}."
            )
        rule = self._broadcast_rules[key]
        return rule.transform(arr)

    # ------------------------------------------------------------------ #
    # Conversion helpers (delegate to module-level functions)
    # ------------------------------------------------------------------ #

    def chronos_to_tt_input(
        self,
        arr: np.ndarray,
        reshape_as: Tuple[int, ...] = (-1,),
    ) -> np.ndarray:
        return chronos_to_tt_input(arr, reshape_as=reshape_as)

    def omni_graph_to_edge_index(
        self,
        adj: np.ndarray,
        threshold: float = 0.0,
        add_self_loops: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return omni_graph_to_edge_index(adj, threshold=threshold, add_self_loops=add_self_loops)

    # ------------------------------------------------------------------ #
    # Statistics & diagnostics
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Dict[str, int]]:
        """Return per-schema validation statistics."""
        return {k: dict(v) for k, v in self._stats.items()}

    def reset_stats(self) -> None:
        for name in self._stats:
            self._stats[name] = {"validates": 0, "passes": 0, "failures": 0}

    def summary(self) -> str:
        lines = [f"UnifiedTensorRegistry  (UTR v{UTR_VERSION})", "-" * 50]
        for name in sorted(self._schemas):
            schema = self._schemas[name]
            st = self._stats.get(name, {})
            lines.append(
                f"  {name:<30s}  v{schema.version}  "
                f"validates={st.get('validates', 0)}  "
                f"pass={st.get('passes', 0)}  fail={st.get('failures', 0)}"
            )
        lines.append(f"\n  Broadcast rules: {len(self._broadcast_rules)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_json(self) -> str:
        data = {
            "utr_version": UTR_VERSION,
            "schemas": [s.to_dict() for s in self._schemas.values()],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "UnifiedTensorRegistry":
        data = json.loads(json_str)
        reg = cls(auto_register_builtins=False)
        for schema_dict in data.get("schemas", []):
            reg.register(TensorSchema.from_dict(schema_dict))
        return reg

    # ------------------------------------------------------------------ #
    # Context manager (scoped registry)
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "UnifiedTensorRegistry":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Tensor envelope — wraps an ndarray with its schema metadata
# ---------------------------------------------------------------------------

@dataclass
class TensorEnvelope:
    """
    A named, versioned tensor with metadata.

    Attributes
    ----------
    schema_name:
        Name of the UTR schema this tensor conforms to.
    data:
        The actual NumPy array.
    producer:
        Name of the module that produced this tensor.
    timestamp_ns:
        Wall-clock nanoseconds at creation.
    tick_id:
        Pipeline tick counter.
    schema_version:
        Schema version at production time.
    metadata:
        Arbitrary key-value annotations.
    """

    schema_name: str
    data: np.ndarray
    producer: str = "unknown"
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    tick_id: int = 0
    schema_version: str = UTR_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        registry: Optional[UnifiedTensorRegistry] = None,
        *,
        raise_on_failure: bool = True,
    ) -> ValidationResult:
        if registry is None:
            registry = UnifiedTensorRegistry.global_registry()
        return registry.validate(
            self.schema_name,
            self.data,
            raise_on_failure=raise_on_failure,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    def age_us(self) -> float:
        """Microseconds elapsed since this envelope was created."""
        return (time.time_ns() - self.timestamp_ns) / 1_000.0

    def fingerprint(self) -> str:
        """SHA-256 digest of the raw data bytes + schema_name."""
        h = hashlib.sha256()
        h.update(self.schema_name.encode())
        h.update(self.data.tobytes())
        return h.hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"TensorEnvelope(schema={self.schema_name!r}, "
            f"shape={self.shape}, dtype={self.dtype}, "
            f"producer={self.producer!r}, tick={self.tick_id})"
        )


# ---------------------------------------------------------------------------
# Schema-aware allocation helpers
# ---------------------------------------------------------------------------

def allocate_chronos_buffer(
    n_assets: int,
    t_ticks: int,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Pre-allocate a ChronosOutput buffer of shape (n_assets, t_ticks, 6)."""
    return np.full((n_assets, t_ticks, 6), fill_value, dtype=np.float32)


def allocate_neuro_sde_buffer(
    n_assets: int,
    latent_dim: int,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    return np.full((n_assets, latent_dim), fill_value, dtype=np.float32)


def allocate_omni_graph_buffer(
    n_assets: int,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    return np.full((n_assets, n_assets), fill_value, dtype=np.float32)


def allocate_lumina_buffer(
    n_assets: int,
    horizon: int,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    return np.full((n_assets, horizon, 3), fill_value, dtype=np.float32)


def allocate_hyper_agent_buffer(
    n_agents: int,
    action_dim: int,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    return np.full((n_agents, action_dim), fill_value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Envelope factory helpers
# ---------------------------------------------------------------------------

def make_chronos_envelope(
    data: np.ndarray,
    *,
    producer: str = "Chronos",
    tick_id: int = 0,
    validate: bool = True,
    registry: Optional[UnifiedTensorRegistry] = None,
) -> TensorEnvelope:
    env = TensorEnvelope(
        schema_name="ChronosOutput",
        data=data.astype(np.float32),
        producer=producer,
        tick_id=tick_id,
    )
    if validate:
        env.validate(registry, raise_on_failure=True)
    return env


def make_neuro_sde_envelope(
    data: np.ndarray,
    *,
    producer: str = "NeuroSDE",
    tick_id: int = 0,
    validate: bool = True,
    registry: Optional[UnifiedTensorRegistry] = None,
) -> TensorEnvelope:
    env = TensorEnvelope(
        schema_name="NeuroSDEState",
        data=data.astype(np.float32),
        producer=producer,
        tick_id=tick_id,
    )
    if validate:
        env.validate(registry, raise_on_failure=True)
    return env


def make_lumina_envelope(
    data: np.ndarray,
    *,
    producer: str = "Lumina",
    tick_id: int = 0,
    validate: bool = True,
    registry: Optional[UnifiedTensorRegistry] = None,
) -> TensorEnvelope:
    env = TensorEnvelope(
        schema_name="LuminaPrediction",
        data=data.astype(np.float32),
        producer=producer,
        tick_id=tick_id,
    )
    if validate:
        env.validate(registry, raise_on_failure=True)
    return env


def make_omni_graph_envelope(
    data: np.ndarray,
    *,
    producer: str = "OmniGraph",
    tick_id: int = 0,
    validate: bool = True,
    registry: Optional[UnifiedTensorRegistry] = None,
) -> TensorEnvelope:
    env = TensorEnvelope(
        schema_name="OmniGraphAdjacency",
        data=data.astype(np.float32),
        producer=producer,
        tick_id=tick_id,
    )
    if validate:
        env.validate(registry, raise_on_failure=True)
    return env


def make_hyper_agent_envelope(
    data: np.ndarray,
    *,
    producer: str = "HyperAgent",
    tick_id: int = 0,
    validate: bool = True,
    registry: Optional[UnifiedTensorRegistry] = None,
) -> TensorEnvelope:
    env = TensorEnvelope(
        schema_name="HyperAgentAction",
        data=data.astype(np.float32),
        producer=producer,
        tick_id=tick_id,
    )
    if validate:
        env.validate(registry, raise_on_failure=True)
    return env


# ---------------------------------------------------------------------------
# Schema evolution helpers
# ---------------------------------------------------------------------------

class SchemaEvolutionManager:
    """
    Manages backward-compatible schema evolution.

    Supports:
      - Adding new optional features (pads with zeros for older producers)
      - Removing deprecated features (drops trailing dims)
      - Renaming features (remapping via index)
    """

    def __init__(self, registry: Optional[UnifiedTensorRegistry] = None) -> None:
        self._registry = registry or UnifiedTensorRegistry.global_registry()
        self._migrations: Dict[Tuple[str, str, str], Callable[[np.ndarray], np.ndarray]] = {}

    def register_migration(
        self,
        schema_name: str,
        from_version: str,
        to_version: str,
        fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        key = (schema_name, from_version, to_version)
        self._migrations[key] = fn
        logger.debug(
            "Registered migration for '%s': %s -> %s", schema_name, from_version, to_version
        )

    def migrate(
        self,
        schema_name: str,
        arr: np.ndarray,
        from_version: str,
    ) -> np.ndarray:
        schema = self._registry.lookup(schema_name)
        to_version = schema.version
        if from_version == to_version:
            return arr
        key = (schema_name, from_version, to_version)
        if key in self._migrations:
            return self._migrations[key](arr)
        # Try dtype cast as fallback
        return self._registry.migrate_tensor(schema_name, arr, from_version)

    def pad_feature_dim(
        self,
        arr: np.ndarray,
        current_n_features: int,
        target_n_features: int,
    ) -> np.ndarray:
        """Pad last dimension from current_n_features to target_n_features with zeros."""
        if current_n_features == target_n_features:
            return arr
        if current_n_features > target_n_features:
            return arr[..., :target_n_features]
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_n_features - current_n_features)]
        return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


# ---------------------------------------------------------------------------
# Diagnostic / introspection utilities
# ---------------------------------------------------------------------------

def infer_schema(arr: np.ndarray, registry: Optional[UnifiedTensorRegistry] = None) -> List[str]:
    """
    Return a list of schema names in *registry* that *arr* could conform to,
    based on shape and dtype alone (no value checks).
    """
    if registry is None:
        registry = UnifiedTensorRegistry.global_registry()
    candidates = []
    for name in registry.list_schemas():
        schema = registry.lookup(name)
        if schema.sparse:
            continue
        if arr.dtype != schema.dtype:
            if not np.can_cast(arr.dtype, schema.dtype):
                continue
        ok, _ = schema.shape_spec.matches(arr.shape)
        if ok:
            candidates.append(name)
    return candidates


def describe_tensor(arr: np.ndarray) -> Dict[str, Any]:
    """Return a summary dict of array statistics."""
    finite = arr[np.isfinite(arr)] if arr.size > 0 else arr.ravel()
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "nbytes": arr.nbytes,
        "n_nan": int(np.sum(np.isnan(arr))),
        "n_inf": int(np.sum(np.isinf(arr))),
        "min": float(finite.min()) if len(finite) > 0 else None,
        "max": float(finite.max()) if len(finite) > 0 else None,
        "mean": float(finite.mean()) if len(finite) > 0 else None,
        "std": float(finite.std()) if len(finite) > 0 else None,
        "sparsity": float(np.sum(arr == 0) / arr.size) if arr.size > 0 else None,
    }


def assert_schema_compat(
    source_schema: str,
    target_schema: str,
    registry: Optional[UnifiedTensorRegistry] = None,
) -> None:
    """
    Raise UTRRegistryError if there is no direct or broadcast path from
    source_schema to target_schema.
    """
    if registry is None:
        registry = UnifiedTensorRegistry.global_registry()
    if source_schema == target_schema:
        return
    if registry.can_broadcast(source_schema, target_schema):
        return
    raise UTRRegistryError(
        f"No compatibility path from '{source_schema}' to '{target_schema}'. "
        f"Register a BroadcastRule to enable this conversion."
    )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "UTR_VERSION",
    "SCHEMA_COMPAT_TABLE",
    # Spec helpers
    "DimSpec",
    "ShapeSpec",
    "RangeConstraint",
    # Schema + result
    "TensorSchema",
    "ValidationResult",
    # Built-in canonical schemas
    "CHRONOS_OUTPUT_SCHEMA",
    "NEURO_SDE_STATE_SCHEMA",
    "OMNI_GRAPH_ADJACENCY_SCHEMA",
    "LUMINA_PREDICTION_SCHEMA",
    "HYPER_AGENT_ACTION_SCHEMA",
    "BUILTIN_SCHEMAS",
    # Broadcasting
    "BroadcastRule",
    # Conversion utilities
    "chronos_to_tt_input",
    "omni_graph_to_edge_index",
    "sparse_csr_to_edge_index",
    "lumina_to_signal_map",
    # Registry
    "UTRRegistryError",
    "SchemaVersionError",
    "UnifiedTensorRegistry",
    # Envelope
    "TensorEnvelope",
    # Allocation helpers
    "allocate_chronos_buffer",
    "allocate_neuro_sde_buffer",
    "allocate_omni_graph_buffer",
    "allocate_lumina_buffer",
    "allocate_hyper_agent_buffer",
    # Envelope factories
    "make_chronos_envelope",
    "make_neuro_sde_envelope",
    "make_lumina_envelope",
    "make_omni_graph_envelope",
    "make_hyper_agent_envelope",
    # Evolution
    "SchemaEvolutionManager",
    # Diagnostics
    "infer_schema",
    "describe_tensor",
    "assert_schema_compat",
]
