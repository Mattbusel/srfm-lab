"""
test_utr.py — Tests for Unified Tensor Registry (UTR) schema validation and conversions.
"""

from __future__ import annotations

import json
import pytest
import numpy as np

from tensor_net.unified_tensor_registry import (
    # Spec
    DimSpec,
    ShapeSpec,
    RangeConstraint,
    # Schema
    TensorSchema,
    ValidationResult,
    # Built-ins
    CHRONOS_OUTPUT_SCHEMA,
    NEURO_SDE_STATE_SCHEMA,
    OMNI_GRAPH_ADJACENCY_SCHEMA,
    LUMINA_PREDICTION_SCHEMA,
    HYPER_AGENT_ACTION_SCHEMA,
    BUILTIN_SCHEMAS,
    # Registry
    UTRRegistryError,
    SchemaVersionError,
    UnifiedTensorRegistry,
    # Envelope
    TensorEnvelope,
    # Allocators
    allocate_chronos_buffer,
    allocate_neuro_sde_buffer,
    allocate_omni_graph_buffer,
    allocate_lumina_buffer,
    allocate_hyper_agent_buffer,
    # Factories
    make_chronos_envelope,
    make_neuro_sde_envelope,
    make_lumina_envelope,
    make_omni_graph_envelope,
    make_hyper_agent_envelope,
    # Conversions
    chronos_to_tt_input,
    omni_graph_to_edge_index,
    sparse_csr_to_edge_index,
    lumina_to_signal_map,
    # Evolution
    SchemaEvolutionManager,
    # Diagnostics
    infer_schema,
    describe_tensor,
    assert_schema_compat,
    UTR_VERSION,
    BroadcastRule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return UnifiedTensorRegistry()


@pytest.fixture
def chronos_arr():
    """Valid ChronosOutput tensor (2 assets, 10 ticks, 6 features)."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.01, 100.0, (2, 10, 6)).astype(np.float32)
    return arr


@pytest.fixture
def neuro_arr():
    rng = np.random.default_rng(1)
    return rng.normal(0.0, 1.0, (5, 16)).astype(np.float32)


@pytest.fixture
def adj_arr():
    rng = np.random.default_rng(2)
    a = rng.uniform(0.0, 1.0, (4, 4)).astype(np.float32)
    np.fill_diagonal(a, 0.0)
    return a


@pytest.fixture
def lumina_arr():
    rng = np.random.default_rng(3)
    arr = rng.uniform(0.0, 1.0, (3, 5, 3)).astype(np.float32)
    return arr


@pytest.fixture
def agent_arr():
    rng = np.random.default_rng(4)
    return rng.normal(0.0, 0.5, (2, 8)).astype(np.float32)


# ---------------------------------------------------------------------------
# DimSpec tests
# ---------------------------------------------------------------------------

class TestDimSpec:
    def test_fixed(self):
        d = DimSpec(6)
        assert d.is_fixed
        assert not d.is_symbolic
        assert not d.is_wildcard
        assert d.matches(6)
        assert not d.matches(7)

    def test_wildcard(self):
        d = DimSpec(-1)
        assert d.is_wildcard
        assert d.matches(0)
        assert d.matches(9999)

    def test_symbolic_first_binding(self):
        d = DimSpec("N")
        bindings: dict = {}
        assert d.matches(5, bindings)
        assert bindings["N"] == 5

    def test_symbolic_second_binding_match(self):
        d = DimSpec("N")
        bindings = {"N": 5}
        assert d.matches(5, bindings)

    def test_symbolic_second_binding_mismatch(self):
        d = DimSpec("N")
        bindings = {"N": 5}
        assert not d.matches(6, bindings)

    def test_repr_fixed(self):
        assert repr(DimSpec(3)) == "3"

    def test_repr_wildcard(self):
        assert repr(DimSpec(-1)) == "*"

    def test_repr_symbolic(self):
        assert repr(DimSpec("K")) == "<K>"


# ---------------------------------------------------------------------------
# ShapeSpec tests
# ---------------------------------------------------------------------------

class TestShapeSpec:
    def test_fixed_match(self):
        s = ShapeSpec(2, 10, 6)
        ok, msg = s.matches((2, 10, 6))
        assert ok

    def test_fixed_mismatch_ndim(self):
        s = ShapeSpec(2, 10, 6)
        ok, msg = s.matches((2, 10))
        assert not ok
        assert "ndim" in msg

    def test_fixed_mismatch_dim(self):
        s = ShapeSpec(2, 10, 6)
        ok, msg = s.matches((2, 10, 5))
        assert not ok
        assert "dim[2]" in msg

    def test_symbolic_match(self):
        s = ShapeSpec("N", "T", 6)
        ok, msg = s.matches((5, 20, 6))
        assert ok

    def test_symbolic_same_symbol_mismatch(self):
        s = ShapeSpec("N", "N")
        ok, _ = s.matches((3, 4))
        assert not ok

    def test_wildcard_any(self):
        s = ShapeSpec(-1, -1, 6)
        ok, _ = s.matches((100, 200, 6))
        assert ok


# ---------------------------------------------------------------------------
# RangeConstraint tests
# ---------------------------------------------------------------------------

class TestRangeConstraint:
    def test_no_nan_fail(self):
        rc = RangeConstraint(allow_nan=False)
        arr = np.array([1.0, float("nan")])
        errors = rc.check(arr)
        assert any("NaN" in e for e in errors)

    def test_no_inf_fail(self):
        rc = RangeConstraint(allow_inf=False)
        arr = np.array([1.0, float("inf")])
        errors = rc.check(arr)
        assert any("Inf" in e for e in errors)

    def test_range_low_fail(self):
        rc = RangeConstraint(low=0.0)
        arr = np.array([-0.01, 1.0])
        errors = rc.check(arr)
        assert any("min" in e for e in errors)

    def test_range_high_fail(self):
        rc = RangeConstraint(high=1.0)
        arr = np.array([0.5, 1.01])
        errors = rc.check(arr)
        assert any("max" in e for e in errors)

    def test_valid_passes(self):
        rc = RangeConstraint(low=0.0, high=10.0)
        arr = np.array([0.0, 5.0, 10.0])
        errors = rc.check(arr)
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# TensorSchema validation tests
# ---------------------------------------------------------------------------

class TestTensorSchemaValidation:
    def test_chronos_valid(self, chronos_arr):
        result = CHRONOS_OUTPUT_SCHEMA.validate(chronos_arr)
        assert result.passed, result.errors

    def test_chronos_wrong_dtype(self, chronos_arr):
        arr64 = chronos_arr.astype(np.float64)
        result = CHRONOS_OUTPUT_SCHEMA.validate(arr64)
        assert not result.passed
        assert any("dtype" in e for e in result.errors)

    def test_chronos_wrong_ndim(self):
        arr = np.ones((5, 6), dtype=np.float32)
        result = CHRONOS_OUTPUT_SCHEMA.validate(arr)
        assert not result.passed

    def test_chronos_wrong_feature_dim(self):
        arr = np.ones((2, 10, 5), dtype=np.float32)
        result = CHRONOS_OUTPUT_SCHEMA.validate(arr)
        assert not result.passed

    def test_chronos_nan_fails(self, chronos_arr):
        chronos_arr[0, 0, 0] = float("nan")
        result = CHRONOS_OUTPUT_SCHEMA.validate(chronos_arr)
        assert not result.passed

    def test_neuro_sde_valid(self, neuro_arr):
        result = NEURO_SDE_STATE_SCHEMA.validate(neuro_arr)
        assert result.passed

    def test_omni_graph_valid(self, adj_arr):
        result = OMNI_GRAPH_ADJACENCY_SCHEMA.validate(adj_arr)
        assert result.passed

    def test_omni_graph_out_of_range(self, adj_arr):
        adj_arr[0, 1] = 1.5
        result = OMNI_GRAPH_ADJACENCY_SCHEMA.validate(adj_arr)
        assert not result.passed

    def test_lumina_valid(self, lumina_arr):
        result = LUMINA_PREDICTION_SCHEMA.validate(lumina_arr)
        assert result.passed

    def test_hyper_agent_valid(self, agent_arr):
        result = HYPER_AGENT_ACTION_SCHEMA.validate(agent_arr)
        assert result.passed

    def test_raise_if_failed(self):
        arr = np.ones((2, 10, 5), dtype=np.float32)
        result = CHRONOS_OUTPUT_SCHEMA.validate(arr)
        with pytest.raises(ValueError):
            result.raise_if_failed()

    def test_validation_result_bool(self, chronos_arr):
        result = CHRONOS_OUTPUT_SCHEMA.validate(chronos_arr)
        assert bool(result) is True


# ---------------------------------------------------------------------------
# UnifiedTensorRegistry tests
# ---------------------------------------------------------------------------

class TestUnifiedTensorRegistry:
    def test_builtin_schemas_registered(self, registry):
        for schema in BUILTIN_SCHEMAS:
            assert registry.has_schema(schema.name)

    def test_lookup_existing(self, registry):
        s = registry.lookup("ChronosOutput")
        assert s.name == "ChronosOutput"

    def test_lookup_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.lookup("NonExistentSchema")

    def test_list_schemas(self, registry):
        names = registry.list_schemas()
        assert "ChronosOutput" in names
        assert "LuminaPrediction" in names

    def test_list_schemas_by_tag(self, registry):
        names = registry.list_schemas(tags=["graph"])
        assert "OmniGraphAdjacency" in names
        assert "ChronosOutput" not in names

    def test_register_duplicate_same_version_no_overwrite(self, registry):
        # Should silently skip (no exception) for same version
        schema = registry.lookup("ChronosOutput")
        registry.register(schema)  # should not raise

    def test_register_duplicate_different_version_raises(self, registry):
        from dataclasses import replace as dc_replace
        schema = registry.lookup("ChronosOutput")
        new_schema = TensorSchema(
            name="ChronosOutput",
            shape_spec=ShapeSpec("N", "T", 6),
            dtype=np.dtype("float32"),
            version="9.9.9",
        )
        with pytest.raises(UTRRegistryError):
            registry.register(new_schema)

    def test_register_overwrite(self, registry):
        new_schema = TensorSchema(
            name="ChronosOutput",
            shape_spec=ShapeSpec("N", "T", 6),
            dtype=np.dtype("float32"),
            version="9.9.9",
        )
        registry.register(new_schema, overwrite=True)
        assert registry.lookup("ChronosOutput").version == "9.9.9"

    def test_deregister(self, registry):
        registry.deregister("HyperAgentAction")
        assert not registry.has_schema("HyperAgentAction")

    def test_deregister_missing_raises(self, registry):
        with pytest.raises(UTRRegistryError):
            registry.deregister("Ghost")

    def test_validate_via_registry(self, registry, chronos_arr):
        result = registry.validate("ChronosOutput", chronos_arr)
        assert result.passed

    def test_validate_fail_raise(self, registry):
        bad = np.ones((2, 10, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            registry.validate("ChronosOutput", bad, raise_on_failure=True)

    def test_validate_batch(self, registry, chronos_arr, neuro_arr):
        results = registry.validate_batch({
            "ChronosOutput": chronos_arr,
            "NeuroSDEState": neuro_arr,
        })
        assert results["ChronosOutput"].passed
        assert results["NeuroSDEState"].passed

    def test_stats_accumulate(self, registry, chronos_arr):
        for _ in range(3):
            registry.validate("ChronosOutput", chronos_arr)
        stats = registry.stats()
        assert stats["ChronosOutput"]["validates"] == 3
        assert stats["ChronosOutput"]["passes"] == 3

    def test_reset_stats(self, registry, chronos_arr):
        registry.validate("ChronosOutput", chronos_arr)
        registry.reset_stats()
        assert registry.stats()["ChronosOutput"]["validates"] == 0

    def test_can_broadcast(self, registry):
        assert registry.can_broadcast("ChronosOutput", "NeuroSDEState")
        assert not registry.can_broadcast("LuminaPrediction", "HyperAgentAction")

    def test_broadcast_chronos_to_neuro(self, registry, chronos_arr):
        out = registry.broadcast("ChronosOutput", "NeuroSDEState", chronos_arr)
        assert out.shape == (chronos_arr.shape[0], 6)
        assert out.dtype == np.float32

    def test_version_compat_same(self, registry):
        assert registry.check_version_compat("ChronosOutput", UTR_VERSION)

    def test_version_compat_older(self, registry):
        # 1.3.0 is in compat table for 1.4.0
        assert registry.check_version_compat("ChronosOutput", "1.3.0")

    def test_version_compat_incompatible(self, registry):
        assert not registry.check_version_compat("ChronosOutput", "0.1.0")

    def test_summary_runs(self, registry):
        s = registry.summary()
        assert "UTR" in s

    def test_to_json_from_json(self, registry):
        js = registry.to_json()
        reg2 = UnifiedTensorRegistry.from_json(js)
        assert reg2.has_schema("ChronosOutput")

    def test_global_registry_singleton(self):
        r1 = UnifiedTensorRegistry.global_registry()
        r2 = UnifiedTensorRegistry.global_registry()
        assert r1 is r2


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

class TestConversions:
    def test_chronos_to_tt_input_flat(self, chronos_arr):
        out = chronos_to_tt_input(chronos_arr)
        N, T, F = chronos_arr.shape
        assert out.shape == (N, T * F)
        assert out.dtype == np.float32

    def test_chronos_to_tt_input_custom_shape(self, chronos_arr):
        N, T, F = chronos_arr.shape
        out = chronos_to_tt_input(chronos_arr, reshape_as=(T * F,))
        assert out.shape == (N, T * F)

    def test_chronos_to_tt_input_bad_reshape_raises(self, chronos_arr):
        with pytest.raises(ValueError):
            chronos_to_tt_input(chronos_arr, reshape_as=(7,))  # 10*6=60 not divisible by 7

    def test_omni_graph_to_edge_index(self, adj_arr):
        ei, ew = omni_graph_to_edge_index(adj_arr, threshold=0.0)
        assert ei.shape[0] == 2
        assert ew.dtype == np.float32
        assert len(ew) == ei.shape[1]

    def test_omni_graph_to_edge_index_threshold(self, adj_arr):
        # Set all but one edge to zero
        arr = np.zeros_like(adj_arr)
        arr[0, 1] = 0.8
        ei, ew = omni_graph_to_edge_index(arr, threshold=0.5)
        assert ei.shape[1] == 1
        assert ew[0] == pytest.approx(0.8, abs=1e-5)

    def test_omni_graph_to_edge_index_self_loops(self, adj_arr):
        ei, ew = omni_graph_to_edge_index(adj_arr, add_self_loops=True)
        # All diagonal entries should now be present
        diag_mask = ei[0] == ei[1]
        assert diag_mask.sum() == adj_arr.shape[0]

    def test_sparse_csr_to_edge_index(self):
        data = np.array([0.5, 0.3, 0.7], dtype=np.float32)
        indices = np.array([1, 0, 2], dtype=np.int32)
        indptr = np.array([0, 1, 2, 3], dtype=np.int32)
        ei, ew = sparse_csr_to_edge_index(data, indices, indptr, shape=(3, 3))
        assert ei.shape == (2, 3)
        assert ew.tolist() == pytest.approx([0.5, 0.3, 0.7], abs=1e-5)

    def test_lumina_to_signal_map(self, lumina_arr):
        asset_ids = ["AAPL", "GOOG", "MSFT"]
        sig = lumina_to_signal_map(lumina_arr, asset_ids)
        assert set(sig.keys()) == set(asset_ids)
        assert "direction_prob" in sig["AAPL"]
        assert len(sig["AAPL"]["direction_prob"]) == lumina_arr.shape[1]


# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

class TestAllocationHelpers:
    def test_allocate_chronos(self):
        buf = allocate_chronos_buffer(5, 32)
        assert buf.shape == (5, 32, 6)
        assert buf.dtype == np.float32
        assert np.all(buf == 0.0)

    def test_allocate_neuro_sde(self):
        buf = allocate_neuro_sde_buffer(5, 16)
        assert buf.shape == (5, 16)

    def test_allocate_omni_graph(self):
        buf = allocate_omni_graph_buffer(4)
        assert buf.shape == (4, 4)

    def test_allocate_lumina(self):
        buf = allocate_lumina_buffer(3, 5)
        assert buf.shape == (3, 5, 3)

    def test_allocate_hyper_agent(self):
        buf = allocate_hyper_agent_buffer(2, 8)
        assert buf.shape == (2, 8)


# ---------------------------------------------------------------------------
# Envelope factories
# ---------------------------------------------------------------------------

class TestEnvelopeFactories:
    def test_make_chronos_envelope_valid(self, chronos_arr):
        env = make_chronos_envelope(chronos_arr, tick_id=7)
        assert env.schema_name == "ChronosOutput"
        assert env.tick_id == 7
        assert env.data.dtype == np.float32

    def test_make_neuro_sde_envelope_valid(self, neuro_arr):
        env = make_neuro_sde_envelope(neuro_arr)
        assert env.schema_name == "NeuroSDEState"

    def test_make_lumina_envelope_valid(self, lumina_arr):
        env = make_lumina_envelope(lumina_arr)
        assert env.schema_name == "LuminaPrediction"

    def test_envelope_fingerprint_unique(self, chronos_arr):
        env1 = make_chronos_envelope(chronos_arr)
        arr2 = chronos_arr.copy()
        arr2[0, 0, 0] += 1.0
        env2 = make_chronos_envelope(arr2)
        assert env1.fingerprint() != env2.fingerprint()

    def test_envelope_age_us(self, chronos_arr):
        import time
        env = make_chronos_envelope(chronos_arr)
        time.sleep(0.001)
        assert env.age_us() > 0.0


# ---------------------------------------------------------------------------
# Schema evolution
# ---------------------------------------------------------------------------

class TestSchemaEvolution:
    def test_pad_feature_dim_extend(self):
        mgr = SchemaEvolutionManager()
        arr = np.ones((3, 4), dtype=np.float32)
        out = mgr.pad_feature_dim(arr, 4, 6)
        assert out.shape == (3, 6)
        assert np.all(out[:, 4:] == 0.0)

    def test_pad_feature_dim_truncate(self):
        mgr = SchemaEvolutionManager()
        arr = np.ones((3, 8), dtype=np.float32)
        out = mgr.pad_feature_dim(arr, 8, 6)
        assert out.shape == (3, 6)

    def test_migrate_tensor_dtype_cast(self, registry, chronos_arr):
        arr64 = chronos_arr.astype(np.float64)
        out = registry.migrate_tensor("ChronosOutput", arr64, UTR_VERSION)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_infer_schema_chronos(self, chronos_arr):
        candidates = infer_schema(chronos_arr)
        assert "ChronosOutput" in candidates

    def test_describe_tensor(self, chronos_arr):
        info = describe_tensor(chronos_arr)
        assert "shape" in info
        assert "mean" in info
        assert info["n_nan"] == 0

    def test_assert_schema_compat_same(self, registry):
        assert_schema_compat("ChronosOutput", "ChronosOutput", registry)

    def test_assert_schema_compat_broadcast(self, registry):
        assert_schema_compat("ChronosOutput", "NeuroSDEState", registry)

    def test_assert_schema_compat_no_path_raises(self, registry):
        from tensor_net.unified_tensor_registry import UTRRegistryError
        with pytest.raises(UTRRegistryError):
            assert_schema_compat("LuminaPrediction", "HyperAgentAction", registry)
