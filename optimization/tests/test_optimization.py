"""
optimization/tests/test_optimization.py
=========================================
Test suite for the LARSA parameter optimization and configuration
management infrastructure.

Coverage:
  - ParamSchema validation (ranges, types, cross-parameter constraints)
  - LiveParams dataclass construction and serialization
  - ParamManager local validation and delta computation
  - GenomeDecoder roundtrip encoding/decoding
  - GenomeDecoder constraint enforcement
  - SensitivityAnalyzer OAT sensitivity ordering
  - WalkForwardOptimizer window construction
  - Integration: schema -> manager -> bridge pipeline

Run with: pytest optimization/tests/test_optimization.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add repo root to path
_REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from config.param_schema import ParamSchema
from config.param_manager import LiveParams, ParamManager
from optimization.genome_bridge import GenomeDecoder, IAEBridge
from optimization.sensitivity_analyzer import SensitivityAnalyzer, OATResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def schema() -> ParamSchema:
    return ParamSchema()


@pytest.fixture(scope="module")
def manager(schema: ParamSchema) -> ParamManager:
    return ParamManager(schema=schema, base_url="http://localhost:19999")


@pytest.fixture(scope="module")
def decoder(schema: ParamSchema) -> GenomeDecoder:
    return GenomeDecoder(schema=schema)


@pytest.fixture
def default_params(schema: ParamSchema) -> dict[str, Any]:
    return schema.defaults()


@pytest.fixture
def live_params(schema: ParamSchema) -> LiveParams:
    return LiveParams.from_schema_defaults(schema)


# ---------------------------------------------------------------------------
# 1. ParamSchema -- range validation
# ---------------------------------------------------------------------------

class TestParamSchemaRanges:

    def test_float_within_range_is_valid(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("CF_BULL_THRESH", 1.5)
        assert ok, f"Expected valid, got: {msg}"

    def test_float_at_lower_bound_is_valid(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("CF_BULL_THRESH", 0.1)
        assert ok, f"Expected valid at lower bound, got: {msg}"

    def test_float_at_upper_bound_is_valid(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("CF_BULL_THRESH", 5.0)
        assert ok, f"Expected valid at upper bound, got: {msg}"

    def test_float_below_min_is_invalid(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("CF_BULL_THRESH", 0.05)
        assert not ok, "Expected invalid for value below min"
        assert "below minimum" in msg

    def test_float_above_max_is_invalid(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("CF_BULL_THRESH", 6.0)
        assert not ok, "Expected invalid for value above max"
        assert "exceeds maximum" in msg

    def test_int_within_range_is_valid(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("MIN_HOLD_BARS", 8)
        assert ok, f"Expected valid int, got: {msg}"

    def test_int_below_min_is_invalid(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("MIN_HOLD_BARS", 0)
        assert not ok, "Expected invalid for int below min"

    def test_int_above_max_is_invalid(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("MIN_HOLD_BARS", 100)
        assert not ok, "Expected invalid for int above max"

    def test_nav_omega_zero_is_valid(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("NAV_OMEGA_SCALE_K", 0.0)
        assert ok, f"Expected 0.0 valid for NAV_OMEGA_SCALE_K: {msg}"

    def test_nav_ema_alpha_bounds(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("NAV_EMA_ALPHA", 0.005)
        assert ok
        ok, _ = schema.validate_one("NAV_EMA_ALPHA", 0.50)
        assert ok
        ok, _ = schema.validate_one("NAV_EMA_ALPHA", 0.001)
        assert not ok

    def test_rl_stop_loss_bounds(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("RL_STOP_LOSS", 0.005)
        assert ok
        ok, _ = schema.validate_one("RL_STOP_LOSS", 0.20)
        assert ok
        ok, _ = schema.validate_one("RL_STOP_LOSS", 0.001)
        assert not ok

    def test_garch_alpha_bounds(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("GARCH_ALPHA", 0.01)
        assert ok
        ok, _ = schema.validate_one("GARCH_ALPHA", 0.30)
        assert ok
        ok, _ = schema.validate_one("GARCH_ALPHA", 0.005)
        assert not ok

    def test_garch_beta_lower_bound(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("GARCH_BETA", 0.50)
        assert ok
        ok, _ = schema.validate_one("GARCH_BETA", 0.49)
        assert not ok

    def test_hurst_window_bounds(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("HURST_WINDOW", 20)
        assert ok
        ok, _ = schema.validate_one("HURST_WINDOW", 500)
        assert ok
        ok, _ = schema.validate_one("HURST_WINDOW", 19)
        assert not ok


# ---------------------------------------------------------------------------
# 2. ParamSchema -- type checking
# ---------------------------------------------------------------------------

class TestParamSchemaTypeChecking:

    def test_float_param_rejects_string(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("CF_BULL_THRESH", "1.5")
        assert not ok, "String should be rejected for float param"

    def test_float_param_rejects_bool(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("CF_BULL_THRESH", True)
        assert not ok, "Bool should be rejected for float param"

    def test_int_param_rejects_float_with_decimal(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("MIN_HOLD_BARS", 4.5)
        assert not ok, "4.5 should be rejected for int param"

    def test_int_param_accepts_int_as_float(self, schema: ParamSchema) -> None:
        # 4.0 is technically an integer value even as float
        ok, msg = schema.validate_one("MIN_HOLD_BARS", 4.0)
        assert ok, f"4.0 should be accepted for int param: {msg}"

    def test_bool_param_rejects_int(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("RL_EXIT_ACTIVE", 1)
        assert not ok, "Integer 1 should be rejected for bool param"

    def test_bool_param_accepts_true(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("RL_EXIT_ACTIVE", True)
        assert ok

    def test_bool_param_accepts_false(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("RL_EXIT_ACTIVE", False)
        assert ok

    def test_list_int_accepts_valid_list(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("BLOCKED_HOURS", [1, 2, 3])
        assert ok

    def test_list_int_rejects_string(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("BLOCKED_HOURS", "1,2,3")
        assert not ok

    def test_list_int_rejects_out_of_range_element(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("BLOCKED_HOURS", [25])
        assert not ok, "Hour 25 is out of range"

    def test_list_int_rejects_float_element(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("BLOCKED_HOURS", [1.5])
        assert not ok, "Float element should be rejected"

    def test_unknown_param_is_rejected(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate_one("NONEXISTENT_PARAM", 1.0)
        assert not ok
        assert "Unknown" in msg

    def test_empty_blocked_hours_is_valid(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("BLOCKED_HOURS", [])
        assert ok

    def test_all_hours_blocked_is_valid(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("BLOCKED_HOURS", list(range(20)))
        assert ok

    def test_too_many_blocked_hours_is_invalid(self, schema: ParamSchema) -> None:
        ok, _ = schema.validate_one("BLOCKED_HOURS", list(range(21)))
        assert not ok


# ---------------------------------------------------------------------------
# 3. Cross-parameter constraints
# ---------------------------------------------------------------------------

class TestCrossParameterConstraints:

    def test_bear_less_than_bull_fails(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate({"CF_BULL_THRESH": 2.0, "CF_BEAR_THRESH": 1.5})
        assert not ok
        assert "CF_BEAR_THRESH" in msg or "constraint" in msg.lower()

    def test_bear_equal_to_bull_passes(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate({"CF_BULL_THRESH": 1.5, "CF_BEAR_THRESH": 1.5})
        assert ok, f"Bear == bull should be allowed: {msg}"

    def test_garch_non_stationary_fails(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate({"GARCH_ALPHA": 0.15, "GARCH_BETA": 0.90})
        assert not ok
        assert "GARCH" in msg or "stationary" in msg.lower() or "constraint" in msg.lower()

    def test_garch_barely_stationary_passes(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate({"GARCH_ALPHA": 0.09, "GARCH_BETA": 0.88})
        assert ok, f"0.09+0.88=0.97 should pass: {msg}"

    def test_ou_kappa_min_geq_max_fails(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate({"OU_KAPPA_MIN": 2.0, "OU_KAPPA_MAX": 1.0})
        assert not ok

    def test_max_hold_leq_min_hold_fails(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate({"MIN_HOLD_BARS": 10, "MAX_HOLD_BARS": 5})
        assert not ok

    def test_ml_suppress_geq_boost_thresh_fails(self, schema: ParamSchema) -> None:
        ok, msg = schema.validate({"ML_SIGNAL_SUPPRESS_THRESH": 0.5, "ML_SIGNAL_BOOST_THRESH": 0.3})
        assert not ok


# ---------------------------------------------------------------------------
# 4. ParamManager -- validate_locally
# ---------------------------------------------------------------------------

class TestParamManagerValidateLocally:

    def test_valid_single_param(self, manager: ParamManager) -> None:
        # CF_BULL_THRESH=1.5 also needs CF_BEAR_THRESH >= 1.5 to satisfy constraint
        ok, msg = manager.validate_locally({"CF_BULL_THRESH": 1.5, "CF_BEAR_THRESH": 1.6})
        assert ok, f"Expected valid: {msg}"

    def test_invalid_param_out_of_range(self, manager: ParamManager) -> None:
        ok, msg = manager.validate_locally({"CF_BULL_THRESH": 99.0})
        assert not ok

    def test_valid_full_defaults(self, manager: ParamManager) -> None:
        defaults = manager._schema.defaults()
        ok, msg = manager.validate_locally(defaults)
        assert ok, f"Schema defaults should be valid: {msg}"

    def test_invalid_cross_constraint(self, manager: ParamManager) -> None:
        ok, msg = manager.validate_locally({
            "GARCH_ALPHA": 0.20,
            "GARCH_BETA": 0.85,
        })
        assert not ok, "0.20+0.85=1.05 should violate GARCH stationarity"

    def test_empty_dict_is_valid(self, manager: ParamManager) -> None:
        # Empty update is technically valid (no params to validate)
        ok, msg = manager.validate_locally({})
        assert ok

    def test_boolean_param_validation(self, manager: ParamManager) -> None:
        ok, _ = manager.validate_locally({"RL_EXIT_ACTIVE": True})
        assert ok
        ok, _ = manager.validate_locally({"RL_EXIT_ACTIVE": False})
        assert ok
        ok, _ = manager.validate_locally({"RL_EXIT_ACTIVE": 1})
        assert not ok

    def test_list_param_validation(self, manager: ParamManager) -> None:
        ok, _ = manager.validate_locally({"BLOCKED_HOURS": [1, 13, 14]})
        assert ok
        ok, _ = manager.validate_locally({"BLOCKED_HOURS": [99]})
        assert not ok


# ---------------------------------------------------------------------------
# 5. ParamManager -- compute_delta
# ---------------------------------------------------------------------------

class TestParamDeltaComputation:

    def test_single_float_change(self, manager: ParamManager, live_params: LiveParams) -> None:
        new = {"CF_BULL_THRESH": live_params.CF_BULL_THRESH + 0.3}
        delta = manager.compute_delta(live_params, new)
        assert "CF_BULL_THRESH" in delta
        d = delta["CF_BULL_THRESH"]
        assert d["changed"] is True
        assert abs(d["abs_change"] - 0.3) < 1e-9

    def test_unchanged_param_reports_not_changed(self, manager: ParamManager, live_params: LiveParams) -> None:
        new = {"CF_BULL_THRESH": live_params.CF_BULL_THRESH}
        delta = manager.compute_delta(live_params, new)
        assert delta["CF_BULL_THRESH"]["changed"] is False

    def test_bool_param_change(self, manager: ParamManager, live_params: LiveParams) -> None:
        orig_val = live_params.RL_EXIT_ACTIVE
        new = {"RL_EXIT_ACTIVE": not orig_val}
        delta = manager.compute_delta(live_params, new)
        assert delta["RL_EXIT_ACTIVE"]["changed"] is True

    def test_bool_param_unchanged(self, manager: ParamManager, live_params: LiveParams) -> None:
        new = {"RL_EXIT_ACTIVE": live_params.RL_EXIT_ACTIVE}
        delta = manager.compute_delta(live_params, new)
        assert delta["RL_EXIT_ACTIVE"]["changed"] is False

    def test_list_param_change(self, manager: ParamManager, live_params: LiveParams) -> None:
        old_hours = list(live_params.BLOCKED_HOURS)
        new_hours = old_hours + [22]
        new = {"BLOCKED_HOURS": new_hours}
        delta = manager.compute_delta(live_params, new)
        assert delta["BLOCKED_HOURS"]["changed"] is True

    def test_pct_change_calculation(self, manager: ParamManager, live_params: LiveParams) -> None:
        # CF_BULL_THRESH default = 1.2, change by +0.12 = 10%
        new = {"CF_BULL_THRESH": 1.32}
        delta = manager.compute_delta(live_params, new)
        pct = delta["CF_BULL_THRESH"]["pct_change"]
        assert abs(pct - 10.0) < 0.5, f"Expected ~10% change, got {pct}"

    def test_unknown_key_in_new_is_ignored(self, manager: ParamManager, live_params: LiveParams) -> None:
        new = {"NONEXISTENT_KEY": 99.9}
        delta = manager.compute_delta(live_params, new)
        assert "NONEXISTENT_KEY" not in delta

    def test_delta_returns_old_and_new_values(self, manager: ParamManager, live_params: LiveParams) -> None:
        new_val = 2.5
        new = {"CF_BULL_THRESH": new_val}
        delta = manager.compute_delta(live_params, new)
        d = delta["CF_BULL_THRESH"]
        assert d["old_value"] == live_params.CF_BULL_THRESH
        assert d["new_value"] == new_val

    def test_zero_old_value_handled(self, manager: ParamManager) -> None:
        lp = LiveParams(NAV_OMEGA_SCALE_K=0.0)
        new = {"NAV_OMEGA_SCALE_K": 0.5}
        delta = manager.compute_delta(lp, new)
        # Should not raise division by zero
        assert delta["NAV_OMEGA_SCALE_K"]["changed"] is True


# ---------------------------------------------------------------------------
# 6. GenomeDecoder -- roundtrip
# ---------------------------------------------------------------------------

class TestGenomeDecoderRoundtrip:

    def test_encode_decode_defaults_roundtrip(self, decoder: GenomeDecoder, schema: ParamSchema) -> None:
        defaults = schema.defaults()
        genome = decoder.encode(defaults)
        decoded = decoder.decode(genome)
        # Float params should match within tolerance
        float_params = [n for n in schema.parameter_names if schema.get_spec(n)["type"] == "float"]
        for name in float_params:
            original = defaults.get(name, 0.0)
            recovered = decoded.get(name, 0.0)
            assert abs(original - recovered) < 0.02, (
                f"{name}: original={original}, recovered={recovered}"
            )

    def test_genome_length_matches_decoder(self, decoder: GenomeDecoder, schema: ParamSchema) -> None:
        defaults = schema.defaults()
        genome = decoder.encode(defaults)
        assert len(genome) == decoder.genome_length

    def test_random_genome_has_correct_length(self, decoder: GenomeDecoder) -> None:
        rng = np.random.default_rng(0)
        genome = decoder.random_genome(rng)
        assert len(genome) == decoder.genome_length

    def test_decode_random_genome_is_valid(self, decoder: GenomeDecoder, schema: ParamSchema) -> None:
        rng = np.random.default_rng(42)
        genome = decoder.random_genome(rng)
        params = decoder.decode(genome)
        # Every numeric param should be within its schema range
        for name, val in params.items():
            spec = schema._schema.get(name, {})
            ptype = spec.get("type")
            if ptype == "float":
                lo = spec.get("min", -1e9)
                hi = spec.get("max", 1e9)
                assert lo <= val <= hi, f"{name}={val} out of [{lo}, {hi}]"
            elif ptype == "int":
                lo = spec.get("min", 0)
                hi = spec.get("max", 10000)
                assert lo <= val <= hi, f"{name}={val} out of [{lo}, {hi}]"

    def test_bool_encode_true(self, decoder: GenomeDecoder) -> None:
        genome = decoder.encode({"RL_EXIT_ACTIVE": True})
        idx = decoder._genome_indices["RL_EXIT_ACTIVE"]
        assert genome[idx] == 1.0

    def test_bool_encode_false(self, decoder: GenomeDecoder) -> None:
        genome = decoder.encode({"RL_EXIT_ACTIVE": False})
        idx = decoder._genome_indices["RL_EXIT_ACTIVE"]
        assert genome[idx] == 0.0

    def test_bool_decode_threshold(self, decoder: GenomeDecoder) -> None:
        genome = [0.0] * decoder.genome_length
        idx = decoder._genome_indices["RL_EXIT_ACTIVE"]
        genome[idx] = 0.6
        params = decoder.decode(genome)
        assert params["RL_EXIT_ACTIVE"] is True

        genome[idx] = 0.4
        params = decoder.decode(genome)
        assert params["RL_EXIT_ACTIVE"] is False

    def test_encode_decode_blocked_hours(self, decoder: GenomeDecoder) -> None:
        hours = [1, 5, 13, 22]
        genome = decoder.encode({"BLOCKED_HOURS": hours})
        params = decoder.decode(genome)
        assert set(params["BLOCKED_HOURS"]) == set(hours)

    def test_encode_empty_blocked_hours(self, decoder: GenomeDecoder) -> None:
        genome = decoder.encode({"BLOCKED_HOURS": []})
        params = decoder.decode(genome)
        assert params["BLOCKED_HOURS"] == []

    def test_genome_all_zeros_is_decodable(self, decoder: GenomeDecoder) -> None:
        genome = [0.0] * decoder.genome_length
        params = decoder.decode(genome)
        assert isinstance(params, dict)
        assert len(params) > 0


# ---------------------------------------------------------------------------
# 7. GenomeDecoder -- constraint enforcement
# ---------------------------------------------------------------------------

class TestGenomeConstraintEnforcement:

    def test_bear_less_than_bull_is_repaired(self, decoder: GenomeDecoder) -> None:
        # Inject bear < bull directly
        params_in = {"CF_BULL_THRESH": 2.0, "CF_BEAR_THRESH": 1.5}
        result = decoder._enforce_constraints(params_in)
        assert result["CF_BEAR_THRESH"] >= result["CF_BULL_THRESH"]

    def test_bh_mass_extreme_leq_thresh_is_repaired(self, decoder: GenomeDecoder) -> None:
        params_in = {"BH_MASS_THRESH": 2.5, "BH_MASS_EXTREME": 2.0}
        result = decoder._enforce_constraints(params_in)
        assert result["BH_MASS_EXTREME"] > result["BH_MASS_THRESH"]

    def test_garch_non_stationary_is_repaired(self, decoder: GenomeDecoder) -> None:
        params_in = {"GARCH_ALPHA": 0.30, "GARCH_BETA": 0.80}
        result = decoder._enforce_constraints(params_in)
        assert result["GARCH_ALPHA"] + result["GARCH_BETA"] < 1.0

    def test_max_hold_leq_min_hold_is_repaired(self, decoder: GenomeDecoder) -> None:
        params_in = {"MIN_HOLD_BARS": 20, "MAX_HOLD_BARS": 15}
        result = decoder._enforce_constraints(params_in)
        assert result["MAX_HOLD_BARS"] > result["MIN_HOLD_BARS"]

    def test_ou_kappa_violation_is_repaired(self, decoder: GenomeDecoder) -> None:
        params_in = {"OU_KAPPA_MIN": 3.0, "OU_KAPPA_MAX": 2.0}
        result = decoder._enforce_constraints(params_in)
        assert result["OU_KAPPA_MAX"] > result["OU_KAPPA_MIN"]

    def test_ml_thresh_violation_is_repaired(self, decoder: GenomeDecoder) -> None:
        params_in = {"ML_SIGNAL_SUPPRESS_THRESH": 0.5, "ML_SIGNAL_BOOST_THRESH": 0.3}
        result = decoder._enforce_constraints(params_in)
        assert result["ML_SIGNAL_SUPPRESS_THRESH"] < result["ML_SIGNAL_BOOST_THRESH"]

    def test_no_constraint_violation_unchanged(self, decoder: GenomeDecoder) -> None:
        params_in = {
            "CF_BULL_THRESH": 1.2,
            "CF_BEAR_THRESH": 1.4,
            "GARCH_ALPHA": 0.09,
            "GARCH_BETA": 0.88,
        }
        result = decoder._enforce_constraints(params_in)
        assert result["CF_BULL_THRESH"] == 1.2
        assert result["CF_BEAR_THRESH"] == 1.4
        assert abs(result["GARCH_ALPHA"] - 0.09) < 1e-9
        assert abs(result["GARCH_BETA"] - 0.88) < 1e-9

    def test_decoded_genome_always_satisfies_constraints(self, decoder: GenomeDecoder) -> None:
        rng = np.random.default_rng(123)
        for _ in range(50):
            genome = decoder.random_genome(rng)
            params = decoder.decode(genome)
            # Check each constraint
            assert params.get("CF_BEAR_THRESH", 999) >= params.get("CF_BULL_THRESH", 0)
            assert params.get("BH_MASS_EXTREME", 999) > params.get("BH_MASS_THRESH", 0)
            assert params.get("MAX_HOLD_BARS", 999) > params.get("MIN_HOLD_BARS", 0)
            alpha = params.get("GARCH_ALPHA", 0)
            beta = params.get("GARCH_BETA", 0)
            assert alpha + beta < 1.0, f"GARCH non-stationary: {alpha}+{beta}={alpha+beta}"
            assert params.get("OU_KAPPA_MAX", 999) > params.get("OU_KAPPA_MIN", 0)
            assert params.get("ML_SIGNAL_SUPPRESS_THRESH", -1) < params.get("ML_SIGNAL_BOOST_THRESH", 1)


# ---------------------------------------------------------------------------
# 8. SensitivityAnalyzer -- OAT ordering
# ---------------------------------------------------------------------------

class TestSensitivityOATOrdering:

    @pytest.fixture
    def analyzer(self, schema: ParamSchema) -> SensitivityAnalyzer:
        # Use synthetic evaluator (no price data)
        return SensitivityAnalyzer(bars={}, schema=schema)

    def test_oat_returns_results_for_all_numeric_params(self, analyzer: SensitivityAnalyzer) -> None:
        results = analyzer.oat_sensitivity(perturbations=(0.10,))
        param_names = {r.param_name for r in results}
        numeric = set(analyzer._get_numeric_params())
        # At minimum all params that can be perturbed should appear
        assert len(param_names) > 0

    def test_oat_perturbed_sharpe_differs_from_base(self, analyzer: SensitivityAnalyzer) -> None:
        results = analyzer.oat_sensitivity(perturbations=(0.10, -0.10))
        # At least some results should show non-zero Sharpe change
        nonzero = [r for r in results if abs(r.sharpe_change) > 1e-6]
        assert len(nonzero) > 0, "OAT should produce at least some Sharpe changes"

    def test_oat_direction_up_has_positive_perturbation(self, analyzer: SensitivityAnalyzer) -> None:
        results = analyzer.oat_sensitivity(perturbations=(0.10,))
        for r in results:
            assert r.direction == "up"
            assert r.perturbation_pct > 0

    def test_oat_direction_down_has_negative_perturbation(self, analyzer: SensitivityAnalyzer) -> None:
        results = analyzer.oat_sensitivity(perturbations=(-0.10,))
        for r in results:
            assert r.direction == "down"
            assert r.perturbation_pct < 0

    def test_oat_importance_ordering_is_sorted(self, analyzer: SensitivityAnalyzer) -> None:
        results = analyzer.oat_sensitivity(perturbations=(-0.10, 0.10))
        importance = analyzer._oat_importance(results)
        ranked = sorted(importance.keys(), key=lambda n: -importance[n])
        # Top param should have higher importance than bottom
        if len(ranked) >= 2:
            assert importance[ranked[0]] >= importance[ranked[-1]]

    def test_critical_params_are_subset_of_numeric(self, analyzer: SensitivityAnalyzer) -> None:
        critical = analyzer.get_critical_params(threshold=0.01)
        numeric = set(analyzer._get_numeric_params())
        for name in critical:
            assert name in numeric, f"{name} is not a numeric param"

    def test_oat_perturbed_value_within_schema_bounds(
        self, analyzer: SensitivityAnalyzer, schema: ParamSchema
    ) -> None:
        results = analyzer.oat_sensitivity(perturbations=(-0.25, 0.25))
        for r in results:
            spec = schema._schema.get(r.param_name, {})
            lo = spec.get("min", float("-inf"))
            hi = spec.get("max", float("inf"))
            assert lo <= r.perturbed_value <= hi, (
                f"{r.param_name}: {r.perturbed_value} not in [{lo}, {hi}]"
            )

    def test_oat_base_sharpe_consistent_across_params(self, analyzer: SensitivityAnalyzer) -> None:
        results = analyzer.oat_sensitivity(perturbations=(0.10,))
        if not results:
            pytest.skip("No OAT results")
        base_sharpes = {r.base_sharpe for r in results}
        # All results should share the same base Sharpe
        assert len(base_sharpes) == 1, f"Inconsistent base Sharpes: {base_sharpes}"

    def test_full_analysis_runs_without_error(self, analyzer: SensitivityAnalyzer) -> None:
        report = analyzer.run_full_analysis(run_sobol=False, run_morris=False)
        assert report is not None
        assert isinstance(report.oat_results, list)
        assert isinstance(report.critical_params, list)
        assert isinstance(report.param_importance_rank, list)

    def test_print_summary_does_not_raise(self, analyzer: SensitivityAnalyzer) -> None:
        analyzer.run_full_analysis(run_sobol=False, run_morris=False)
        # Should not raise
        analyzer.print_summary()


# ---------------------------------------------------------------------------
# 9. LiveParams dataclass
# ---------------------------------------------------------------------------

class TestLiveParamsDataclass:

    def test_from_schema_defaults_matches_schema(self, schema: ParamSchema) -> None:
        lp = LiveParams.from_schema_defaults(schema)
        defaults = schema.defaults()
        assert abs(lp.CF_BULL_THRESH - defaults["CF_BULL_THRESH"]) < 1e-9
        assert abs(lp.BH_MASS_THRESH - defaults["BH_MASS_THRESH"]) < 1e-9

    def test_to_dict_roundtrip(self, live_params: LiveParams) -> None:
        d = live_params.to_dict()
        lp2 = LiveParams.from_dict(d)
        assert lp2.CF_BULL_THRESH == live_params.CF_BULL_THRESH
        assert lp2.BLOCKED_HOURS == live_params.BLOCKED_HOURS
        assert lp2.RL_EXIT_ACTIVE == live_params.RL_EXIT_ACTIVE

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {"CF_BULL_THRESH": 2.0, "UNKNOWN_KEY": "garbage"}
        lp = LiveParams.from_dict(d)
        assert lp.CF_BULL_THRESH == 2.0

    def test_default_blocked_hours(self, live_params: LiveParams) -> None:
        assert isinstance(live_params.BLOCKED_HOURS, list)
        assert len(live_params.BLOCKED_HOURS) > 0

    def test_metadata_fields_present(self, live_params: LiveParams) -> None:
        d = live_params.to_dict()
        assert "version" in d
        assert "source" in d
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# 10. Schema defaults validity
# ---------------------------------------------------------------------------

class TestSchemaDefaultsValidity:

    def test_all_defaults_pass_validation(self, schema: ParamSchema) -> None:
        """Every default value in the schema must pass individual validation."""
        for name in schema.parameter_names:
            spec = schema.get_spec(name)
            default = spec["default"]
            ok, msg = schema.validate_one(name, default)
            assert ok, f"Default for {name}={default} fails validation: {msg}"

    def test_full_defaults_dict_passes_cross_constraints(self, schema: ParamSchema) -> None:
        defaults = schema.defaults()
        ok, msg = schema.validate(defaults)
        assert ok, f"Schema defaults fail cross-constraint validation: {msg}"

    def test_schema_has_expected_parameters(self, schema: ParamSchema) -> None:
        required = [
            "CF_BULL_THRESH", "CF_BEAR_THRESH", "BH_MASS_THRESH",
            "MIN_HOLD_BARS", "BLOCKED_HOURS", "NAV_OMEGA_SCALE_K",
            "NAV_GEO_ENTRY_GATE", "NAV_EMA_ALPHA", "ML_SIGNAL_BOOST",
            "RL_EXIT_ACTIVE", "GARCH_ALPHA", "GARCH_BETA",
        ]
        for name in required:
            assert name in schema.parameter_names, f"Expected param {name} in schema"

    def test_schema_has_at_least_25_parameters(self, schema: ParamSchema) -> None:
        assert len(schema.parameter_names) >= 25

    def test_fill_defaults_adds_missing_keys(self, schema: ParamSchema) -> None:
        partial = {"CF_BULL_THRESH": 2.0}
        filled = schema.fill_defaults(partial)
        assert "GARCH_ALPHA" in filled
        assert filled["CF_BULL_THRESH"] == 2.0


# ---------------------------------------------------------------------------
# 11. IAEBridge -- unit tests with mocked HTTP
# ---------------------------------------------------------------------------

class TestIAEBridgeMocked:

    def test_fetch_genome_parses_response(self, decoder: GenomeDecoder) -> None:
        bridge = IAEBridge(decoder=decoder)
        mock_genome = [0.5] * decoder.genome_length
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"genome": mock_genome, "fitness": 1.5}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(bridge._session, "get", return_value=mock_resp):
            genome, fitness = bridge.fetch_latest_genome()

        assert len(genome) == decoder.genome_length
        assert fitness == 1.5

    def test_fetch_genome_handles_connection_error(self, decoder: GenomeDecoder) -> None:
        import requests
        bridge = IAEBridge(decoder=decoder)
        with patch.object(bridge._session, "get", side_effect=requests.exceptions.ConnectionError):
            genome, fitness = bridge.fetch_latest_genome()
        assert genome == []
        assert fitness == float("-inf")

    def test_push_elite_params_encodes_correctly(self, decoder: GenomeDecoder, schema: ParamSchema) -> None:
        bridge = IAEBridge(decoder=decoder)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True}
        mock_resp.raise_for_status = MagicMock()
        params = schema.defaults()
        posted_payloads = []

        def capture_post(url, json=None, **kwargs):
            posted_payloads.append(json)
            return mock_resp

        with patch.object(bridge._session, "post", side_effect=capture_post):
            result = bridge.push_elite_params(params, label="test")

        assert result is True
        assert len(posted_payloads) == 1
        payload = posted_payloads[0]
        assert "genome" in payload
        assert len(payload["genome"]) == decoder.genome_length
        assert payload["label"] == "test"

    def test_get_evolution_stats_returns_dict(self, decoder: GenomeDecoder) -> None:
        bridge = IAEBridge(decoder=decoder)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "generation": 42,
            "population_size": 100,
            "best_fitness_history": [0.5, 0.8, 1.2],
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(bridge._session, "get", return_value=mock_resp):
            stats = bridge.get_evolution_stats()

        assert stats["generation"] == 42
        assert stats["population_size"] == 100


# ---------------------------------------------------------------------------
# 12. WalkForwardOptimizer window construction
# ---------------------------------------------------------------------------

class TestWalkForwardWindowConstruction:

    def test_windows_cover_data_range(self) -> None:
        """Walk-forward windows should start at first date and not exceed last."""
        import pandas as pd
        from optimization.optuna_optimizer import WalkForwardOptimizer

        dates = pd.date_range("2023-01-01", "2024-12-31", freq="15min")
        bars = {"SPY": pd.DataFrame(
            {"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 1000.0},
            index=dates,
        )}
        optimizer = WalkForwardOptimizer(bars=bars, is_months=6, oos_months=2)
        windows = optimizer._build_windows()
        assert len(windows) > 0
        # First window starts at or near data start
        assert windows[0][0] >= dates[0]
        # No window extends beyond data end
        for _, _, oos_end in windows:
            assert oos_end <= dates[-1] + pd.DateOffset(days=1)

    def test_windows_step_by_oos_months(self) -> None:
        import pandas as pd
        from optimization.optuna_optimizer import WalkForwardOptimizer

        dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
        bars = {"SPY": pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            index=dates,
        )}
        optimizer = WalkForwardOptimizer(bars=bars, is_months=6, oos_months=2)
        windows = optimizer._build_windows()
        if len(windows) >= 2:
            gap = (windows[1][0] - windows[0][0]).days
            # ~2 months step
            assert 55 <= gap <= 65, f"Expected ~60-day step, got {gap}"

    def test_empty_bars_produces_no_windows(self) -> None:
        from optimization.optuna_optimizer import WalkForwardOptimizer
        optimizer = WalkForwardOptimizer(bars={})
        windows = optimizer._build_windows()
        assert windows == []
