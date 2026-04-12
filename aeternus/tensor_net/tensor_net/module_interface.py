"""
module_interface.py — Standardized module interface definitions for AETERNUS.

Provides:
  - AETERNUSModule: abstract base class for all pipeline modules
  - GlobalState: dataclass holding references to all current UTR tensors
  - ModuleOutput: typed output envelope with timing metadata
  - Pipeline: chain modules with schema compatibility checking
  - Lazy evaluation: only compute output if downstream consumer is ready
  - Mock implementations of all 6 module interfaces for integration testing
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generator, Iterator, List, Optional,
    Sequence, Set, Tuple, Type, Union
)

import numpy as np

from .unified_tensor_registry import (
    TensorEnvelope,
    UnifiedTensorRegistry,
    ValidationResult,
    UTR_VERSION,
    allocate_chronos_buffer,
    allocate_hyper_agent_buffer,
    allocate_lumina_buffer,
    allocate_neuro_sde_buffer,
    allocate_omni_graph_buffer,
    make_chronos_envelope,
    make_hyper_agent_envelope,
    make_lumina_envelope,
    make_neuro_sde_envelope,
    make_omni_graph_envelope,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

@dataclass
class GlobalState:
    """
    Snapshot of all inter-module tensor data at a single pipeline tick.

    Each field holds a TensorEnvelope (or None if not yet produced).
    Modules read from this state and produce ModuleOutput that updates
    the next GlobalState.
    """

    tick_id: int = 0
    timestamp_ns: int = field(default_factory=time.time_ns)

    # Per-module tensor slots
    chronos_output:       Optional[TensorEnvelope] = None
    neuro_sde_state:      Optional[TensorEnvelope] = None
    omni_graph_adjacency: Optional[TensorEnvelope] = None
    lumina_prediction:    Optional[TensorEnvelope] = None
    hyper_agent_action:   Optional[TensorEnvelope] = None

    # Arbitrary extra tensors (for extensibility)
    extras: Dict[str, TensorEnvelope] = field(default_factory=dict)

    # Pipeline-level metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_us(self) -> float:
        """Microseconds since this state was created."""
        return (time.time_ns() - self.timestamp_ns) / 1_000.0

    def has_all(self, *schema_names: str) -> bool:
        """Return True if all named schemas have non-None values."""
        _map = {
            "ChronosOutput":       self.chronos_output,
            "NeuroSDEState":       self.neuro_sde_state,
            "OmniGraphAdjacency":  self.omni_graph_adjacency,
            "LuminaPrediction":    self.lumina_prediction,
            "HyperAgentAction":    self.hyper_agent_action,
        }
        for name in schema_names:
            val = _map.get(name) or self.extras.get(name)
            if val is None:
                return False
        return True

    def get_tensor(self, schema_name: str) -> Optional[TensorEnvelope]:
        """Unified accessor by schema name."""
        _map: Dict[str, Optional[TensorEnvelope]] = {
            "ChronosOutput":       self.chronos_output,
            "NeuroSDEState":       self.neuro_sde_state,
            "OmniGraphAdjacency":  self.omni_graph_adjacency,
            "LuminaPrediction":    self.lumina_prediction,
            "HyperAgentAction":    self.hyper_agent_action,
        }
        return _map.get(schema_name) or self.extras.get(schema_name)

    def set_tensor(self, schema_name: str, env: TensorEnvelope) -> None:
        """Unified mutator by schema name."""
        _map = {
            "ChronosOutput":       "chronos_output",
            "NeuroSDEState":       "neuro_sde_state",
            "OmniGraphAdjacency":  "omni_graph_adjacency",
            "LuminaPrediction":    "lumina_prediction",
            "HyperAgentAction":    "hyper_agent_action",
        }
        attr = _map.get(schema_name)
        if attr:
            setattr(self, attr, env)
        else:
            self.extras[schema_name] = env

    def clone(self) -> "GlobalState":
        """Shallow clone (envelopes are shared, not deep-copied)."""
        return GlobalState(
            tick_id=self.tick_id,
            timestamp_ns=self.timestamp_ns,
            chronos_output=self.chronos_output,
            neuro_sde_state=self.neuro_sde_state,
            omni_graph_adjacency=self.omni_graph_adjacency,
            lumina_prediction=self.lumina_prediction,
            hyper_agent_action=self.hyper_agent_action,
            extras=dict(self.extras),
            metadata=dict(self.metadata),
        )


# ---------------------------------------------------------------------------
# Module output
# ---------------------------------------------------------------------------

@dataclass
class ModuleOutput:
    """
    Typed output produced by an AETERNUSModule.

    Attributes
    ----------
    module_name:
        Name of the producing module.
    output_schema:
        UTR schema name of the primary output tensor.
    envelope:
        The output TensorEnvelope.
    latency_us:
        Wall-clock microseconds consumed by the forward pass.
    tick_id:
        Pipeline tick at which this output was produced.
    success:
        False if the module encountered an error.
    error_msg:
        Error description if success is False.
    aux_outputs:
        Additional named outputs (schema_name -> envelope).
    """

    module_name: str
    output_schema: str
    envelope: Optional[TensorEnvelope]
    latency_us: float = 0.0
    tick_id: int = 0
    success: bool = True
    error_msg: str = ""
    aux_outputs: Dict[str, TensorEnvelope] = field(default_factory=dict)

    def apply_to_state(self, state: GlobalState) -> None:
        """Write this output into a GlobalState."""
        if self.envelope is not None:
            state.set_tensor(self.output_schema, self.envelope)
        for name, env in self.aux_outputs.items():
            state.set_tensor(name, env)

    def __bool__(self) -> bool:
        return self.success


# ---------------------------------------------------------------------------
# Module base class
# ---------------------------------------------------------------------------

class ModuleStatus(Enum):
    UNINITIALIZED = auto()
    READY = auto()
    RUNNING = auto()
    ERROR = auto()
    DISABLED = auto()


@dataclass
class ModuleMetrics:
    """Running statistics for a module."""
    n_calls: int = 0
    n_errors: int = 0
    total_latency_us: float = 0.0
    min_latency_us: float = float("inf")
    max_latency_us: float = 0.0

    def update(self, latency_us: float, success: bool) -> None:
        self.n_calls += 1
        if not success:
            self.n_errors += 1
        self.total_latency_us += latency_us
        self.min_latency_us = min(self.min_latency_us, latency_us)
        self.max_latency_us = max(self.max_latency_us, latency_us)

    @property
    def mean_latency_us(self) -> float:
        if self.n_calls == 0:
            return 0.0
        return self.total_latency_us / self.n_calls

    @property
    def error_rate(self) -> float:
        if self.n_calls == 0:
            return 0.0
        return self.n_errors / self.n_calls


class AETERNUSModule(ABC):
    """
    Abstract base class for all AETERNUS pipeline modules.

    Subclasses must implement:
      - forward(state: GlobalState) -> ModuleOutput
      - required_inputs (property) -> List[str]
      - output_schema (property) -> str

    The base class handles:
      - Timing and metrics collection
      - Input availability checking
      - Error catching and reporting
      - Schema compatibility checking
    """

    def __init__(
        self,
        name: str,
        registry: Optional[UnifiedTensorRegistry] = None,
    ) -> None:
        self._name = name
        self._registry = registry or UnifiedTensorRegistry.global_registry()
        self._status = ModuleStatus.UNINITIALIZED
        self._metrics = ModuleMetrics()
        self._enabled = True
        self._downstream_ready_fn: Optional[Callable[[], bool]] = None

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def required_inputs(self) -> List[str]:
        """Schema names this module requires from GlobalState."""

    @property
    @abstractmethod
    def output_schema(self) -> str:
        """Primary UTR schema name this module produces."""

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Optional initialization hook (called once before first forward)."""
        self._status = ModuleStatus.READY

    def teardown(self) -> None:
        """Optional cleanup hook."""
        pass

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _forward_impl(self, state: GlobalState) -> ModuleOutput:
        """Module-specific forward logic. Called by forward()."""

    def forward(self, state: GlobalState) -> ModuleOutput:
        """
        Public forward pass with timing, error handling, and lazy evaluation.
        """
        if not self._enabled:
            return ModuleOutput(
                module_name=self._name,
                output_schema=self.output_schema,
                envelope=None,
                success=True,
                tick_id=state.tick_id,
                error_msg="module disabled",
            )

        # Lazy evaluation: skip if downstream not ready
        if self._downstream_ready_fn is not None and not self._downstream_ready_fn():
            return ModuleOutput(
                module_name=self._name,
                output_schema=self.output_schema,
                envelope=None,
                success=True,
                tick_id=state.tick_id,
                error_msg="downstream not ready (lazy skip)",
            )

        # Check required inputs
        for schema in self.required_inputs:
            if state.get_tensor(schema) is None:
                return ModuleOutput(
                    module_name=self._name,
                    output_schema=self.output_schema,
                    envelope=None,
                    success=False,
                    tick_id=state.tick_id,
                    error_msg=f"missing required input: {schema}",
                )

        t0 = time.perf_counter_ns()
        try:
            self._status = ModuleStatus.RUNNING
            result = self._forward_impl(state)
        except Exception as exc:
            latency_us = (time.perf_counter_ns() - t0) / 1_000.0
            self._status = ModuleStatus.ERROR
            self._metrics.update(latency_us, success=False)
            logger.exception("Module '%s' raised exception: %s", self._name, exc)
            return ModuleOutput(
                module_name=self._name,
                output_schema=self.output_schema,
                envelope=None,
                success=False,
                latency_us=latency_us,
                tick_id=state.tick_id,
                error_msg=str(exc),
            )

        latency_us = (time.perf_counter_ns() - t0) / 1_000.0
        result.latency_us = latency_us
        result.tick_id = state.tick_id
        self._status = ModuleStatus.READY
        self._metrics.update(latency_us, success=result.success)
        return result

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    def set_downstream_ready_fn(self, fn: Callable[[], bool]) -> None:
        """Register a callable that returns True when downstream is ready."""
        self._downstream_ready_fn = fn

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def status(self) -> ModuleStatus:
        return self._status

    @property
    def metrics(self) -> ModuleMetrics:
        return self._metrics

    def reset_metrics(self) -> None:
        self._metrics = ModuleMetrics()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self._name!r}, "
            f"status={self._status.name}, calls={self._metrics.n_calls})"
        )


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------

class SchemaCompatibilityError(Exception):
    """Raised when two modules in a pipeline have incompatible schemas."""


class PipelineStage:
    """Wrapper around a module with its position in the pipeline."""

    def __init__(
        self,
        module: AETERNUSModule,
        stage_idx: int,
    ) -> None:
        self.module = module
        self.stage_idx = stage_idx
        self.last_output: Optional[ModuleOutput] = None

    def run(self, state: GlobalState) -> Tuple[ModuleOutput, GlobalState]:
        out = self.module.forward(state)
        self.last_output = out
        if out.success and out.envelope is not None:
            out.apply_to_state(state)
        return out, state


class Pipeline:
    """
    Ordered sequence of AETERNUSModules with schema compatibility checking.

    Usage
    -----
    >>> pipeline = Pipeline()
    >>> pipeline.add(chronos_mock)
    >>> pipeline.add(neuro_sde_mock)
    >>> state = pipeline.run(GlobalState(tick_id=1))
    """

    def __init__(
        self,
        name: str = "AETERNUSPipeline",
        registry: Optional[UnifiedTensorRegistry] = None,
    ) -> None:
        self._name = name
        self._registry = registry or UnifiedTensorRegistry.global_registry()
        self._stages: List[PipelineStage] = []
        self._compiled = False

    def add(self, module: AETERNUSModule) -> "Pipeline":
        """Append a module to the pipeline. Returns self for chaining."""
        stage = PipelineStage(module, len(self._stages))
        self._stages.append(stage)
        self._compiled = False
        return self

    def compile(self) -> None:
        """
        Check schema compatibility across all stages.
        Verifies each module's required_inputs are satisfied by previous modules
        or the initial GlobalState.
        """
        available_schemas: Set[str] = set()
        for stage in self._stages:
            mod = stage.module
            for req in mod.required_inputs:
                if req not in available_schemas:
                    raise SchemaCompatibilityError(
                        f"Module '{mod.name}' requires schema '{req}', "
                        f"but it is not produced by any earlier module. "
                        f"Available so far: {sorted(available_schemas)}"
                    )
            available_schemas.add(mod.output_schema)
        self._compiled = True
        logger.info(
            "Pipeline '%s' compiled with %d stages.", self._name, len(self._stages)
        )

    def run(
        self,
        initial_state: GlobalState,
        *,
        stop_on_error: bool = False,
    ) -> Tuple[GlobalState, List[ModuleOutput]]:
        """
        Run the full pipeline on *initial_state*.

        Returns
        -------
        (final_state, outputs_per_stage)
        """
        state = initial_state.clone()
        outputs: List[ModuleOutput] = []
        for stage in self._stages:
            out, state = stage.run(state)
            outputs.append(out)
            if not out.success and stop_on_error:
                logger.warning(
                    "Pipeline stopping at stage %d ('%s'): %s",
                    stage.stage_idx, stage.module.name, out.error_msg,
                )
                break
        return state, outputs

    def run_generator(
        self,
        states: Iterator[GlobalState],
        *,
        stop_on_error: bool = False,
    ) -> Generator[Tuple[GlobalState, List[ModuleOutput]], None, None]:
        """Yield (final_state, outputs) for each state in *states*."""
        for state in states:
            yield self.run(state, stop_on_error=stop_on_error)

    def initialize_all(self) -> None:
        """Call initialize() on all modules."""
        for stage in self._stages:
            stage.module.initialize()

    def teardown_all(self) -> None:
        """Call teardown() on all modules."""
        for stage in self._stages:
            stage.module.teardown()

    def summary(self) -> str:
        lines = [f"Pipeline '{self._name}' ({len(self._stages)} stages)"]
        for stage in self._stages:
            mod = stage.module
            m = mod.metrics
            lines.append(
                f"  [{stage.stage_idx}] {mod.name:<30s}  "
                f"calls={m.n_calls}  "
                f"mean_lat={m.mean_latency_us:.1f}us  "
                f"errors={m.n_errors}"
            )
        return "\n".join(lines)

    @property
    def n_stages(self) -> int:
        return len(self._stages)

    def __len__(self) -> int:
        return len(self._stages)


# ---------------------------------------------------------------------------
# Mock implementations of all 6 module interfaces
# ---------------------------------------------------------------------------

class MockChronosModule(AETERNUSModule):
    """
    Mock Chronos module.
    Produces synthetic ChronosOutput tensors from Gaussian noise.
    """

    def __init__(
        self,
        n_assets: int = 10,
        t_ticks: int = 64,
        seed: Optional[int] = 42,
        registry: Optional[UnifiedTensorRegistry] = None,
    ) -> None:
        super().__init__("MockChronos", registry)
        self._n_assets = n_assets
        self._t_ticks = t_ticks
        self._rng = np.random.default_rng(seed)

    @property
    def required_inputs(self) -> List[str]:
        return []  # Chronos is a source module

    @property
    def output_schema(self) -> str:
        return "ChronosOutput"

    def _forward_impl(self, state: GlobalState) -> ModuleOutput:
        # Generate plausible market data: prices ~100, volume ~1e6, imbalance in [-1,1]
        base_prices = self._rng.uniform(50.0, 200.0, (self._n_assets, 1))
        price_noise = self._rng.normal(0.0, 0.5, (self._n_assets, self._t_ticks))
        prices = np.clip(base_prices + np.cumsum(price_noise, axis=1), 1.0, None)

        spread = self._rng.uniform(0.01, 0.1, (self._n_assets, self._t_ticks))
        bid = (prices - spread / 2).astype(np.float32)
        ask = (prices + spread / 2).astype(np.float32)
        mid = prices.astype(np.float32)
        vol = self._rng.uniform(1e4, 1e6, (self._n_assets, self._t_ticks)).astype(np.float32)
        imb = self._rng.uniform(-1.0, 1.0, (self._n_assets, self._t_ticks)).astype(np.float32)

        data = np.stack([bid, ask, mid, spread.astype(np.float32), vol, imb], axis=-1)

        env = TensorEnvelope(
            schema_name="ChronosOutput",
            data=data,
            producer=self._name,
            tick_id=state.tick_id,
        )
        return ModuleOutput(
            module_name=self._name,
            output_schema="ChronosOutput",
            envelope=env,
            success=True,
        )


class MockNeuroSDEModule(AETERNUSModule):
    """
    Mock Neural SDE module.
    Produces NeuroSDEState from ChronosOutput via random projection.
    """

    def __init__(
        self,
        n_assets: int = 10,
        latent_dim: int = 32,
        seed: Optional[int] = 43,
        registry: Optional[UnifiedTensorRegistry] = None,
    ) -> None:
        super().__init__("MockNeuroSDE", registry)
        self._n_assets = n_assets
        self._latent_dim = latent_dim
        self._rng = np.random.default_rng(seed)
        # Random projection matrix: (6, latent_dim)
        self._proj: Optional[np.ndarray] = None

    def initialize(self) -> None:
        super().initialize()
        self._proj = self._rng.normal(0.0, 0.1, (6, self._latent_dim)).astype(np.float32)

    @property
    def required_inputs(self) -> List[str]:
        return ["ChronosOutput"]

    @property
    def output_schema(self) -> str:
        return "NeuroSDEState"

    def _forward_impl(self, state: GlobalState) -> ModuleOutput:
        chronos_env = state.get_tensor("ChronosOutput")
        assert chronos_env is not None
        arr = chronos_env.data  # (N, T, 6)
        # Use last tick: (N, 6) @ (6, latent_dim) = (N, latent_dim)
        last_tick = arr[:, -1, :]  # (N, 6)
        if self._proj is None:
            self._proj = self._rng.normal(0.0, 0.1, (6, self._latent_dim)).astype(np.float32)
        latent = (last_tick @ self._proj).astype(np.float32)

        env = TensorEnvelope(
            schema_name="NeuroSDEState",
            data=latent,
            producer=self._name,
            tick_id=state.tick_id,
        )
        return ModuleOutput(
            module_name=self._name,
            output_schema="NeuroSDEState",
            envelope=env,
            success=True,
        )


class MockOmniGraphModule(AETERNUSModule):
    """
    Mock OmniGraph module.
    Produces OmniGraphAdjacency from NeuroSDEState via cosine similarity.
    """

    def __init__(
        self,
        n_assets: int = 10,
        registry: Optional[UnifiedTensorRegistry] = None,
    ) -> None:
        super().__init__("MockOmniGraph", registry)
        self._n_assets = n_assets

    @property
    def required_inputs(self) -> List[str]:
        return ["NeuroSDEState"]

    @property
    def output_schema(self) -> str:
        return "OmniGraphAdjacency"

    def _forward_impl(self, state: GlobalState) -> ModuleOutput:
        sde_env = state.get_tensor("NeuroSDEState")
        assert sde_env is not None
        Z = sde_env.data  # (N, latent_dim)
        # Cosine similarity matrix
        norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        Z_norm = Z / norms
        adj = (Z_norm @ Z_norm.T).astype(np.float32)
        # Clip to [0, 1] and zero diagonal
        adj = np.clip(adj, 0.0, 1.0)
        np.fill_diagonal(adj, 0.0)

        env = TensorEnvelope(
            schema_name="OmniGraphAdjacency",
            data=adj,
            producer=self._name,
            tick_id=state.tick_id,
        )
        return ModuleOutput(
            module_name=self._name,
            output_schema="OmniGraphAdjacency",
            envelope=env,
            success=True,
        )


class MockLuminaModule(AETERNUSModule):
    """
    Mock Lumina forecasting module.
    Produces LuminaPrediction from ChronosOutput.
    """

    def __init__(
        self,
        n_assets: int = 10,
        horizon: int = 5,
        seed: Optional[int] = 44,
        registry: Optional[UnifiedTensorRegistry] = None,
    ) -> None:
        super().__init__("MockLumina", registry)
        self._n_assets = n_assets
        self._horizon = horizon
        self._rng = np.random.default_rng(seed)

    @property
    def required_inputs(self) -> List[str]:
        return ["ChronosOutput"]

    @property
    def output_schema(self) -> str:
        return "LuminaPrediction"

    def _forward_impl(self, state: GlobalState) -> ModuleOutput:
        # direction_prob in [0, 1], vol_forecast > 0, regime_logits unconstrained
        dir_prob = self._rng.uniform(0.0, 1.0, (self._n_assets, self._horizon, 1)).astype(np.float32)
        vol_fore = self._rng.uniform(0.001, 0.05, (self._n_assets, self._horizon, 1)).astype(np.float32)
        reg_logits = self._rng.normal(0.0, 1.0, (self._n_assets, self._horizon, 1)).astype(np.float32)
        pred = np.concatenate([dir_prob, vol_fore, reg_logits], axis=-1)

        env = TensorEnvelope(
            schema_name="LuminaPrediction",
            data=pred,
            producer=self._name,
            tick_id=state.tick_id,
        )
        return ModuleOutput(
            module_name=self._name,
            output_schema="LuminaPrediction",
            envelope=env,
            success=True,
        )


class MockHyperAgentModule(AETERNUSModule):
    """
    Mock HyperAgent RL module.
    Produces HyperAgentAction from LuminaPrediction and OmniGraphAdjacency.
    """

    def __init__(
        self,
        n_agents: int = 4,
        action_dim: int = 10,
        seed: Optional[int] = 45,
        registry: Optional[UnifiedTensorRegistry] = None,
    ) -> None:
        super().__init__("MockHyperAgent", registry)
        self._n_agents = n_agents
        self._action_dim = action_dim
        self._rng = np.random.default_rng(seed)

    @property
    def required_inputs(self) -> List[str]:
        return ["LuminaPrediction", "OmniGraphAdjacency"]

    @property
    def output_schema(self) -> str:
        return "HyperAgentAction"

    def _forward_impl(self, state: GlobalState) -> ModuleOutput:
        # Random action with small tanh non-linearity to bound to (-1, 1)
        raw = self._rng.normal(0.0, 0.3, (self._n_agents, self._action_dim)).astype(np.float32)
        actions = np.tanh(raw)

        env = TensorEnvelope(
            schema_name="HyperAgentAction",
            data=actions,
            producer=self._name,
            tick_id=state.tick_id,
        )
        return ModuleOutput(
            module_name=self._name,
            output_schema="HyperAgentAction",
            envelope=env,
            success=True,
        )


class MockRiskModule(AETERNUSModule):
    """
    Mock Risk / Portfolio module.
    Reads HyperAgentAction and LuminaPrediction; produces a risk score tensor.
    This is a 6th module for completeness.
    """

    def __init__(
        self,
        n_assets: int = 10,
        seed: Optional[int] = 46,
        registry: Optional[UnifiedTensorRegistry] = None,
    ) -> None:
        super().__init__("MockRisk", registry)
        self._n_assets = n_assets
        self._rng = np.random.default_rng(seed)

    @property
    def required_inputs(self) -> List[str]:
        return ["HyperAgentAction", "LuminaPrediction"]

    @property
    def output_schema(self) -> str:
        return "RiskScore"

    def initialize(self) -> None:
        super().initialize()
        # Register a custom RiskScore schema if not present
        reg = self._registry
        if not reg.has_schema("RiskScore"):
            from .unified_tensor_registry import TensorSchema, ShapeSpec, RangeConstraint
            risk_schema = TensorSchema(
                name="RiskScore",
                shape_spec=ShapeSpec("N_assets", 3),
                dtype=np.dtype("float32"),
                feature_names=["var_95", "cvar_95", "drawdown_prob"],
                range_constraint=RangeConstraint(low=0.0, high=1.0),
                version=UTR_VERSION,
                description="Per-asset risk scores from the Risk module.",
                tags=["risk", "portfolio"],
            )
            reg.register(risk_schema)

    def _forward_impl(self, state: GlobalState) -> ModuleOutput:
        # Dummy risk scores derived from lumina vol forecast
        lumina_env = state.get_tensor("LuminaPrediction")
        assert lumina_env is not None
        vol_fore = lumina_env.data[:, 0, 1]  # (N,)
        var_95 = np.clip(vol_fore * 1.645, 0.0, 1.0).astype(np.float32)
        cvar_95 = np.clip(vol_fore * 2.063, 0.0, 1.0).astype(np.float32)
        dd_prob = self._rng.uniform(0.0, 0.3, self._n_assets).astype(np.float32)
        risk = np.stack([var_95, cvar_95, dd_prob], axis=-1)

        env = TensorEnvelope(
            schema_name="RiskScore",
            data=risk,
            producer=self._name,
            tick_id=state.tick_id,
        )
        return ModuleOutput(
            module_name=self._name,
            output_schema="RiskScore",
            envelope=env,
            success=True,
        )


# ---------------------------------------------------------------------------
# Pre-built pipeline factory
# ---------------------------------------------------------------------------

def build_mock_pipeline(
    n_assets: int = 10,
    n_agents: int = 4,
    t_ticks: int = 64,
    latent_dim: int = 32,
    horizon: int = 5,
    action_dim: int = 10,
    seed: int = 0,
    registry: Optional[UnifiedTensorRegistry] = None,
) -> Pipeline:
    """
    Build a complete mock AETERNUS pipeline with all 6 modules.

    ChronosOutput -> NeuroSDEState -> OmniGraphAdjacency
                  -> LuminaPrediction
                                   -> HyperAgentAction
                                                     -> RiskScore
    """
    reg = registry or UnifiedTensorRegistry.global_registry()

    chronos  = MockChronosModule(n_assets, t_ticks, seed=seed, registry=reg)
    neuro    = MockNeuroSDEModule(n_assets, latent_dim, seed=seed+1, registry=reg)
    graph    = MockOmniGraphModule(n_assets, registry=reg)
    lumina   = MockLuminaModule(n_assets, horizon, seed=seed+2, registry=reg)
    agent    = MockHyperAgentModule(n_agents, action_dim, seed=seed+3, registry=reg)
    risk     = MockRiskModule(n_assets, seed=seed+4, registry=reg)

    # Initialize risk to register its custom schema
    risk.initialize()

    pipeline = Pipeline(name="MockAETERNUSPipeline", registry=reg)
    (pipeline
        .add(chronos)
        .add(neuro)
        .add(graph)
        .add(lumina)
        .add(agent)
        .add(risk))

    pipeline.initialize_all()
    return pipeline


# ---------------------------------------------------------------------------
# Module decorator for functional-style modules
# ---------------------------------------------------------------------------

def aeternus_module(
    name: str,
    required_inputs: List[str],
    output_schema: str,
    registry: Optional[UnifiedTensorRegistry] = None,
) -> Callable[[Callable[[GlobalState], ModuleOutput]], AETERNUSModule]:
    """
    Decorator that wraps a plain function into an AETERNUSModule.

    Usage
    -----
    >>> @aeternus_module("MyModule", ["ChronosOutput"], "NeuroSDEState")
    ... def my_module(state: GlobalState) -> ModuleOutput:
    ...     ...
    """
    def decorator(fn: Callable[[GlobalState], ModuleOutput]) -> AETERNUSModule:
        class _WrappedModule(AETERNUSModule):
            @property
            def required_inputs(self) -> List[str]:
                return required_inputs

            @property
            def output_schema(self) -> str:
                return output_schema

            def _forward_impl(self, state: GlobalState) -> ModuleOutput:
                return fn(state)

        mod = _WrappedModule(name, registry)
        mod.__doc__ = fn.__doc__
        return mod

    return decorator  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Integration test runner
# ---------------------------------------------------------------------------

def run_integration_test(
    n_ticks: int = 10,
    n_assets: int = 10,
    n_agents: int = 4,
    verbose: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Run a complete integration test of the mock pipeline.

    Returns a dict with per-tick results and aggregate metrics.
    """
    pipeline = build_mock_pipeline(
        n_assets=n_assets,
        n_agents=n_agents,
        seed=seed,
    )

    results = []
    for tick in range(n_ticks):
        state = GlobalState(tick_id=tick)
        final_state, outputs = pipeline.run(state)
        tick_result = {
            "tick_id": tick,
            "outputs": {
                o.module_name: {
                    "success": o.success,
                    "latency_us": o.latency_us,
                    "schema": o.output_schema,
                    "shape": o.envelope.shape if o.envelope else None,
                }
                for o in outputs
            },
            "all_success": all(o.success for o in outputs),
        }
        results.append(tick_result)
        if verbose:
            n_ok = sum(1 for o in outputs if o.success)
            logger.info("Tick %d: %d/%d modules succeeded.", tick, n_ok, len(outputs))

    if verbose:
        print(pipeline.summary())

    return {
        "n_ticks": n_ticks,
        "n_stages": pipeline.n_stages,
        "tick_results": results,
        "all_passed": all(r["all_success"] for r in results),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # State / output
    "GlobalState",
    "ModuleOutput",
    "ModuleStatus",
    "ModuleMetrics",
    # Base class
    "AETERNUSModule",
    # Pipeline
    "SchemaCompatibilityError",
    "PipelineStage",
    "Pipeline",
    # Mock modules
    "MockChronosModule",
    "MockNeuroSDEModule",
    "MockOmniGraphModule",
    "MockLuminaModule",
    "MockHyperAgentModule",
    "MockRiskModule",
    # Factory
    "build_mock_pipeline",
    # Decorator
    "aeternus_module",
    # Test runner
    "run_integration_test",
]
