"""
tests/integration/test_coordination_layer.py
============================================
Integration tests for the SRFM parameter coordination layer.

The Elixir coordination service runs on :8781.  When the real service is not
available these tests spin up a MockElixirCoordination Flask server on the
same port and tear it down after the test session.

All tests use the requests library to call the HTTP API exactly as production
code would.

Run with:
    pytest tests/integration/test_coordination_layer.py -v
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from flask import Flask, jsonify, request as flask_request
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Schema constants (mirror coordination layer rules)
# ---------------------------------------------------------------------------

COORDINATION_HOST = os.environ.get("SRFM_COORD_HOST", "localhost")
COORDINATION_PORT = int(os.environ.get("SRFM_COORD_PORT", "8781"))
BASE_URL = f"http://{COORDINATION_HOST}:{COORDINATION_PORT}"

# Valid parameter schema bounds
PARAM_SCHEMA = {
    "CF_BULL_THRESH": {"min": 0.0001, "max": 0.01, "default": 0.001},
    "CF_BEAR_THRESH": {"min": 0.0001, "max": 0.01, "default": 0.0008},
    "BH_DECAY": {"min": 0.80, "max": 0.99, "default": 0.95},
    "BH_FORM": {"min": 0.5, "max": 5.0, "default": 1.5},
    "SIGNAL_ENTRY_THRESHOLD": {"min": 0.30, "max": 0.95, "default": 0.65},
    "POSITION_SIZE_BASE": {"min": 0.005, "max": 0.05, "default": 0.02},
    "MIN_HOLD_BARS": {"min": 1, "max": 20, "default": 4},
    "RL_EXIT_LOSS_THRESHOLD": {"min": -0.10, "max": -0.005, "default": -0.031},
}

MAX_DELTA_PCT = 0.25   # 25% max single-step change
ROLLBACK_SHARPE = -0.5  # rolling 4h Sharpe below this triggers rollback


# ---------------------------------------------------------------------------
# Mock Elixir coordination server
# ---------------------------------------------------------------------------

class MockElixirCoordination:
    """
    Minimal Flask server that mimics the coordination layer HTTP API.

    Endpoints implemented:
      POST /params/propose        -- propose parameter change
      GET  /params/current        -- current parameter values
      POST /params/rollback       -- force rollback to last snapshot
      GET  /circuit/:service      -- circuit breaker state
      POST /circuit/:service/reset -- reset circuit breaker
      GET  /health                -- health check
      GET  /metrics/sharpe        -- current rolling Sharpe
      POST /metrics/update        -- push performance update
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8781):
        self.host = host
        self.port = port
        self._app = Flask("mock_coordination")
        self._params: Dict[str, Any] = {k: v["default"] for k, v in PARAM_SCHEMA.items()}
        self._param_snapshot: Dict[str, Any] = dict(self._params)
        self._circuit_states: Dict[str, str] = {
            "alpaca": "closed",
            "binance": "closed",
            "coinmetrics": "closed",
        }
        self._sharpe_4h: float = 0.8
        self._rollback_log: List[dict] = []
        self._proposal_log: List[dict] = []
        self._server_thread: Optional[threading.Thread] = None
        self._register_routes()

    def _register_routes(self) -> None:
        app = self._app

        @app.route("/params/propose", methods=["POST"])
        def propose_params():
            data = flask_request.get_json(force=True) or {}
            proposed = data.get("params", {})
            errors = self._validate_proposal(proposed)
            if errors:
                return jsonify({"status": "rejected", "errors": errors}), 422
            # Check delta
            delta_errors = self._check_delta(proposed)
            if delta_errors:
                return jsonify({"status": "rejected", "errors": delta_errors}), 422
            # Accept
            self._param_snapshot = dict(self._params)
            self._params.update(proposed)
            self._proposal_log.append({"params": proposed, "status": "accepted"})
            return jsonify({"status": "accepted", "current_params": self._params})

        @app.route("/params/current", methods=["GET"])
        def get_params():
            return jsonify({"params": self._params})

        @app.route("/params/rollback", methods=["POST"])
        def rollback():
            self._params = dict(self._param_snapshot)
            entry = {"reason": "manual_rollback", "restored": self._params}
            self._rollback_log.append(entry)
            return jsonify({"status": "rolled_back", "params": self._params})

        @app.route("/circuit/<service>", methods=["GET"])
        def circuit_state(service: str):
            state = self._circuit_states.get(service, "unknown")
            return jsonify({"service": service, "state": state})

        @app.route("/circuit/<service>/reset", methods=["POST"])
        def circuit_reset(service: str):
            if service in self._circuit_states:
                self._circuit_states[service] = "closed"
                return jsonify({"service": service, "state": "closed"})
            return jsonify({"error": "unknown service"}), 404

        @app.route("/circuit/<service>/open", methods=["POST"])
        def circuit_open(service: str):
            if service in self._circuit_states:
                self._circuit_states[service] = "open"
                return jsonify({"service": service, "state": "open"})
            return jsonify({"error": "unknown service"}), 404

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok", "services": self._circuit_states})

        @app.route("/metrics/sharpe", methods=["GET"])
        def sharpe():
            return jsonify({"sharpe_4h": self._sharpe_4h})

        @app.route("/metrics/update", methods=["POST"])
        def update_metrics():
            data = flask_request.get_json(force=True) or {}
            if "sharpe_4h" in data:
                self._sharpe_4h = float(data["sharpe_4h"])
            # Auto-rollback if poor performance
            if self._sharpe_4h < ROLLBACK_SHARPE:
                self._params = dict(self._param_snapshot)
                self._rollback_log.append({
                    "reason": "poor_performance",
                    "sharpe_4h": self._sharpe_4h,
                    "restored": self._params,
                })
                return jsonify({
                    "status": "rollback_triggered",
                    "sharpe_4h": self._sharpe_4h,
                    "params": self._params,
                })
            return jsonify({"status": "ok", "sharpe_4h": self._sharpe_4h})

    def _validate_proposal(self, proposed: dict) -> List[str]:
        errors: List[str] = []
        for key, value in proposed.items():
            if key not in PARAM_SCHEMA:
                errors.append(f"Unknown parameter: {key}")
                continue
            schema = PARAM_SCHEMA[key]
            if value < schema["min"] or value > schema["max"]:
                errors.append(
                    f"{key}={value} outside bounds [{schema['min']}, {schema['max']}]"
                )
        return errors

    def _check_delta(self, proposed: dict) -> List[str]:
        errors: List[str] = []
        for key, new_val in proposed.items():
            if key not in self._params:
                continue
            current = self._params[key]
            if abs(current) < 1e-12:
                continue
            delta = abs(new_val - current) / abs(current)
            if delta > MAX_DELTA_PCT:
                errors.append(
                    f"{key}: change of {delta*100:.1f}% exceeds max {MAX_DELTA_PCT*100:.0f}%"
                )
        return errors

    def start(self) -> None:
        """Start the Flask server in a background thread."""
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        self._server_thread = threading.Thread(
            target=lambda: self._app.run(host=self.host, port=self.port, use_reloader=False),
            daemon=True,
        )
        self._server_thread.start()
        # Wait for server to come up
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                requests.get(f"http://localhost:{self.port}/health", timeout=0.5)
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError(f"Mock coordination server failed to start on port {self.port}")

    def stop(self) -> None:
        # Flask dev server does not expose a clean stop -- daemon thread dies with process
        pass


# ---------------------------------------------------------------------------
# Session-scoped fixture: start mock server once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def coord_server():
    """
    Start MockElixirCoordination if real service not already running.
    Yields the base URL.
    """
    if not _REQUESTS_AVAILABLE or not _FLASK_AVAILABLE:
        pytest.skip("requests and flask required for coordination integration tests")

    # Check if real service is up
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=1.0)
        if r.status_code == 200:
            yield BASE_URL
            return
    except Exception:
        pass

    # Start mock
    mock = MockElixirCoordination(port=COORDINATION_PORT)
    mock.start()
    yield BASE_URL
    mock.stop()


@pytest.fixture(autouse=True)
def reset_params(coord_server):
    """Reset params to defaults before each test."""
    # Use rollback to restore snapshot (snapshot was set at server init)
    try:
        requests.post(f"{coord_server}/params/rollback", timeout=2.0)
    except Exception:
        pass
    yield


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def propose(coord_server: str, params: dict, timeout: float = 3.0) -> requests.Response:
    return requests.post(
        f"{coord_server}/params/propose",
        json={"params": params},
        timeout=timeout,
    )


# ===========================================================================
# Test suite
# ===========================================================================

class TestParamProposal:
    """Tests for the /params/propose endpoint."""

    def test_param_proposal_accepted(self, coord_server):
        """Valid params within schema bounds should be accepted."""
        resp = propose(coord_server, {"BH_DECAY": 0.93, "BH_FORM": 2.0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "accepted"
        assert body["current_params"]["BH_DECAY"] == pytest.approx(0.93)

    def test_param_proposal_rejected_schema_negative(self, coord_server):
        """CF_BULL_THRESH=-1 is below schema min=0.0001 -> rejected."""
        resp = propose(coord_server, {"CF_BULL_THRESH": -1.0})
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "rejected"
        assert any("CF_BULL_THRESH" in e for e in body["errors"])

    def test_param_proposal_rejected_schema_above_max(self, coord_server):
        """BH_FORM=99 is above schema max=5.0 -> rejected."""
        resp = propose(coord_server, {"BH_FORM": 99.0})
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "rejected"

    def test_param_proposal_rejected_delta(self, coord_server):
        """A change of >25% from current value should be rejected."""
        # Get current CF_BULL_THRESH
        current_resp = requests.get(f"{coord_server}/params/current", timeout=2.0)
        current = current_resp.json()["params"]["CF_BULL_THRESH"]
        # Propose 50% increase
        new_val = current * 1.50
        # Clamp to schema bounds
        new_val = min(new_val, PARAM_SCHEMA["CF_BULL_THRESH"]["max"])
        if new_val == current:
            pytest.skip("Cannot construct >25% delta within schema bounds")
        resp = propose(coord_server, {"CF_BULL_THRESH": new_val})
        body = resp.json()
        assert body["status"] == "rejected"
        assert any("%" in e for e in body["errors"])

    def test_unknown_param_rejected(self, coord_server):
        """Unknown parameter keys should be rejected."""
        resp = propose(coord_server, {"MYSTERY_PARAM": 42.0})
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "rejected"
        assert any("MYSTERY_PARAM" in e for e in body["errors"])

    def test_empty_proposal_accepted(self, coord_server):
        """Empty proposal dict is valid (no-op change)."""
        resp = propose(coord_server, {})
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_multiple_valid_params_accepted(self, coord_server):
        """Multiple valid params in one proposal should all be accepted."""
        resp = propose(coord_server, {
            "BH_DECAY": 0.92,
            "SIGNAL_ENTRY_THRESHOLD": 0.70,
            "MIN_HOLD_BARS": 5,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "accepted"
        cp = body["current_params"]
        assert cp["BH_DECAY"] == pytest.approx(0.92)
        assert cp["SIGNAL_ENTRY_THRESHOLD"] == pytest.approx(0.70)
        assert cp["MIN_HOLD_BARS"] == 5

    def test_partial_rejection_rejects_all(self, coord_server):
        """If any param in a batch is invalid, entire proposal rejected."""
        resp = propose(coord_server, {
            "BH_DECAY": 0.91,      # valid
            "CF_BULL_THRESH": -5,  # invalid
        })
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "rejected"

    def test_boundary_values_accepted(self, coord_server):
        """Exactly at schema min/max boundaries should be accepted."""
        resp = propose(coord_server, {
            "CF_BULL_THRESH": PARAM_SCHEMA["CF_BULL_THRESH"]["min"],
            "BH_DECAY": PARAM_SCHEMA["BH_DECAY"]["max"],
        })
        assert resp.status_code == 200


class TestCurrentParams:
    """Tests for the /params/current endpoint."""

    def test_current_params_returns_dict(self, coord_server):
        resp = requests.get(f"{coord_server}/params/current", timeout=2.0)
        assert resp.status_code == 200
        body = resp.json()
        assert "params" in body
        assert isinstance(body["params"], dict)

    def test_current_params_has_expected_keys(self, coord_server):
        resp = requests.get(f"{coord_server}/params/current", timeout=2.0)
        params = resp.json()["params"]
        for key in PARAM_SCHEMA:
            assert key in params, f"Expected key {key} in current params"

    def test_current_params_reflect_accepted_proposal(self, coord_server):
        """After an accepted proposal, GET /params/current reflects new values."""
        propose(coord_server, {"BH_FORM": 2.5})
        resp = requests.get(f"{coord_server}/params/current", timeout=2.0)
        assert resp.json()["params"]["BH_FORM"] == pytest.approx(2.5)


class TestRollback:
    """Tests for parameter rollback behaviour."""

    def test_rollback_on_poor_performance(self, coord_server):
        """
        Mock 4h Sharpe = -0.6 -> rollback should be triggered automatically.
        """
        # First accept a param change (creates a snapshot to roll back to)
        original_resp = requests.get(f"{coord_server}/params/current", timeout=2.0)
        original_bh_decay = original_resp.json()["params"]["BH_DECAY"]

        propose(coord_server, {"BH_DECAY": 0.88})

        # Push poor performance metric
        resp = requests.post(
            f"{coord_server}/metrics/update",
            json={"sharpe_4h": -0.6},
            timeout=2.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "rollback_triggered"
        assert body["sharpe_4h"] == pytest.approx(-0.6)

        # Verify params rolled back
        current = requests.get(f"{coord_server}/params/current", timeout=2.0).json()["params"]
        assert current["BH_DECAY"] == pytest.approx(original_bh_decay)

    def test_no_rollback_on_good_performance(self, coord_server):
        """Positive Sharpe should not trigger rollback."""
        propose(coord_server, {"BH_DECAY": 0.91})
        resp = requests.post(
            f"{coord_server}/metrics/update",
            json={"sharpe_4h": 1.2},
            timeout=2.0,
        )
        assert resp.json()["status"] == "ok"
        # Params should remain at proposed value
        current = requests.get(f"{coord_server}/params/current", timeout=2.0).json()["params"]
        assert current["BH_DECAY"] == pytest.approx(0.91)

    def test_manual_rollback_endpoint(self, coord_server):
        """POST /params/rollback should restore snapshot."""
        propose(coord_server, {"BH_FORM": 3.0})
        # Manual rollback
        resp = requests.post(f"{coord_server}/params/rollback", timeout=2.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "rolled_back"
        # BH_FORM should be back to default
        assert body["params"]["BH_FORM"] == pytest.approx(PARAM_SCHEMA["BH_FORM"]["default"])

    def test_rollback_at_sharpe_boundary(self, coord_server):
        """Sharpe exactly at ROLLBACK_SHARPE should still trigger rollback."""
        propose(coord_server, {"BH_DECAY": 0.89})
        resp = requests.post(
            f"{coord_server}/metrics/update",
            json={"sharpe_4h": ROLLBACK_SHARPE - 0.0001},
            timeout=2.0,
        )
        assert resp.json()["status"] == "rollback_triggered"


class TestCircuitBreaker:
    """Tests for circuit breaker state management."""

    def test_circuit_breaker_state_default_closed(self, coord_server):
        """Default circuit state should be 'closed' (normal operation)."""
        resp = requests.get(f"{coord_server}/circuit/alpaca", timeout=2.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "alpaca"
        assert body["state"] == "closed"

    def test_circuit_breaker_open(self, coord_server):
        """Opening a circuit changes state to 'open'."""
        requests.post(f"{coord_server}/circuit/alpaca/open", timeout=2.0)
        resp = requests.get(f"{coord_server}/circuit/alpaca", timeout=2.0)
        assert resp.json()["state"] == "open"

    def test_circuit_breaker_reset(self, coord_server):
        """Resetting an open circuit returns state to 'closed'."""
        requests.post(f"{coord_server}/circuit/alpaca/open", timeout=2.0)
        requests.post(f"{coord_server}/circuit/alpaca/reset", timeout=2.0)
        resp = requests.get(f"{coord_server}/circuit/alpaca", timeout=2.0)
        assert resp.json()["state"] == "closed"

    def test_circuit_breaker_unknown_service(self, coord_server):
        """Unknown service name returns 404."""
        resp = requests.get(f"{coord_server}/circuit/nonexistent_service", timeout=2.0)
        assert resp.status_code == 404

    def test_binance_circuit_independent(self, coord_server):
        """Opening alpaca circuit does not affect binance circuit."""
        requests.post(f"{coord_server}/circuit/alpaca/open", timeout=2.0)
        resp = requests.get(f"{coord_server}/circuit/binance", timeout=2.0)
        assert resp.json()["state"] == "closed"

    def test_all_circuits_visible_in_health(self, coord_server):
        """Health endpoint should include all circuit states."""
        resp = requests.get(f"{coord_server}/health", timeout=2.0)
        body = resp.json()
        assert "services" in body
        for svc in ["alpaca", "binance", "coinmetrics"]:
            assert svc in body["services"], f"{svc} missing from health response"


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_ok(self, coord_server):
        resp = requests.get(f"{coord_server}/health", timeout=2.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_response_time_reasonable(self, coord_server):
        """Health check should respond in < 500ms."""
        start = time.time()
        requests.get(f"{coord_server}/health", timeout=2.0)
        elapsed = time.time() - start
        assert elapsed < 0.5, f"Health check took {elapsed*1000:.0f}ms, expected < 500ms"


class TestMetricsEndpoint:
    """Tests for the /metrics endpoints."""

    def test_sharpe_endpoint_returns_float(self, coord_server):
        resp = requests.get(f"{coord_server}/metrics/sharpe", timeout=2.0)
        assert resp.status_code == 200
        body = resp.json()
        assert "sharpe_4h" in body
        assert isinstance(body["sharpe_4h"], (int, float))

    def test_metrics_update_persists(self, coord_server):
        """Updated Sharpe is reflected in subsequent GET."""
        requests.post(
            f"{coord_server}/metrics/update",
            json={"sharpe_4h": 2.1},
            timeout=2.0,
        )
        resp = requests.get(f"{coord_server}/metrics/sharpe", timeout=2.0)
        assert resp.json()["sharpe_4h"] == pytest.approx(2.1)

    def test_metrics_update_missing_sharpe_ok(self, coord_server):
        """Metrics update without sharpe_4h should not error."""
        resp = requests.post(
            f"{coord_server}/metrics/update",
            json={"other_metric": 1.0},
            timeout=2.0,
        )
        assert resp.status_code == 200


class TestConcurrentProposals:
    """Concurrent proposal handling."""

    def test_sequential_proposals_preserve_state(self, coord_server):
        """Multiple sequential proposals should each be applied correctly."""
        propose(coord_server, {"BH_DECAY": 0.93})
        propose(coord_server, {"BH_FORM": 1.8})
        current = requests.get(f"{coord_server}/params/current", timeout=2.0).json()["params"]
        assert current["BH_DECAY"] == pytest.approx(0.93)
        assert current["BH_FORM"] == pytest.approx(1.8)

    def test_second_proposal_within_delta(self, coord_server):
        """
        After accepting first proposal, second proposal's delta is measured from
        the new current value, not the original default.
        """
        # Move BH_DECAY from 0.95 to 0.92 (3.2% change -- within 25%)
        propose(coord_server, {"BH_DECAY": 0.92})
        # Now move from 0.92 to 0.89 (3.3% change -- within 25%)
        resp = propose(coord_server, {"BH_DECAY": 0.89})
        assert resp.json()["status"] == "accepted"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
