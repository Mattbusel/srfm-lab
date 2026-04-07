# infra/deployment/tests/test_deployment.py -- unit tests for the deployment infrastructure
from __future__ import annotations

import json
import threading
import time
import unittest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# -- module imports ----------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from infra.deployment.health_checker import (
    HealthDegradationDetector,
    HealthStatus,
    ServiceEndpoint,
    ServiceHealth,
    ServiceHealthChecker,
)
from infra.deployment.process_manager import (
    ProcessManager,
    ProcessStatus,
    RestartPolicy,
    ServiceDefinition,
    _process_alive,
    _read_pid,
    _write_pid,
    _rotate_log_if_needed,
)
from infra.deployment.config_sync import (
    ConfigChange,
    ConfigDiff,
    ConfigSync,
    SyncResult,
    validate_srfm_config,
)
from infra.deployment.runbook_executor import (
    RunbookDef,
    RunbookExecutor,
    RunbookResult,
    RunbookStep,
)
from infra.deployment.deployment_manager import (
    DeploymentManager,
    DeploymentRecord,
    DeployStrategy,
    VersionManager,
)


# ===========================================================================
# Helper: fake HTTP server simulation via monkeypatching
# ===========================================================================

class _FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ===========================================================================
# 1. HealthStatus enum
# ===========================================================================

class TestHealthStatus(unittest.TestCase):
    def test_all_values_exist(self):
        expected = {"healthy", "degraded", "unhealthy", "unreachable", "unknown"}
        actual = {s.value for s in HealthStatus}
        self.assertEqual(expected, actual)


# ===========================================================================
# 2. ServiceHealth data class
# ===========================================================================

class TestServiceHealth(unittest.TestCase):
    def test_is_ok_only_for_healthy(self):
        def make(status):
            return ServiceHealth(
                name="svc",
                status=status,
                latency_ms=10.0,
                last_check=datetime.utcnow(),
            )

        self.assertTrue(make(HealthStatus.HEALTHY).is_ok())
        self.assertFalse(make(HealthStatus.DEGRADED).is_ok())
        self.assertFalse(make(HealthStatus.UNHEALTHY).is_ok())
        self.assertFalse(make(HealthStatus.UNREACHABLE).is_ok())
        self.assertFalse(make(HealthStatus.UNKNOWN).is_ok())

    def test_summary_contains_name_and_status(self):
        h = ServiceHealth(
            name="risk-api",
            status=HealthStatus.UNHEALTHY,
            latency_ms=123.4,
            last_check=datetime.utcnow(),
            error="timeout",
        )
        s = h.summary()
        self.assertIn("risk-api", s)
        self.assertIn("UNHEALTHY", s)
        self.assertIn("timeout", s)


# ===========================================================================
# 3. ServiceHealthChecker -- HTTP check via mock
# ===========================================================================

class TestServiceHealthChecker(unittest.TestCase):
    def _make_checker(self, status_code: int, body: Dict) -> ServiceHealthChecker:
        endpoints = [
            ServiceEndpoint(name="test-svc", url="http://localhost:9999/health")
        ]
        checker = ServiceHealthChecker(services=endpoints)
        return checker

    def test_healthy_200_json(self):
        body = json.dumps({"status": "ok"}).encode()
        endpoints = [ServiceEndpoint(name="test-svc", url="http://localhost:9999/health")]
        checker = ServiceHealthChecker(services=endpoints)

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _FakeResponse(200, body)
            result = checker.check_service("test-svc")

        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertEqual(result.name, "test-svc")

    def test_unhealthy_500(self):
        body = b""
        endpoints = [ServiceEndpoint(name="test-svc", url="http://localhost:9999/health")]
        checker = ServiceHealthChecker(services=endpoints)

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _FakeResponse(500, body)
            result = checker.check_service("test-svc")

        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        self.assertIn("500", result.error or "")

    def test_unreachable_on_url_error(self):
        import urllib.error
        endpoints = [ServiceEndpoint(name="test-svc", url="http://localhost:9999/health")]
        checker = ServiceHealthChecker(services=endpoints)

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = checker.check_service("test-svc")

        self.assertEqual(result.status, HealthStatus.UNREACHABLE)

    def test_unknown_service_returns_unknown_status(self):
        checker = ServiceHealthChecker(services=[])
        result = checker.check_service("nonexistent")
        self.assertEqual(result.status, HealthStatus.UNKNOWN)

    def test_health_report_contains_service_name(self):
        body = json.dumps({"status": "ok"}).encode()
        endpoints = [ServiceEndpoint(name="my-service", url="http://localhost:9999/health")]
        checker = ServiceHealthChecker(services=endpoints)

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _FakeResponse(200, body)
            checker.check_all()

        report = checker.health_report()
        self.assertIn("my-service", report)
        self.assertIn("HEALTHY", report)

    def test_degraded_status_from_json_body(self):
        body = json.dumps({"status": "degraded"}).encode()
        endpoints = [ServiceEndpoint(name="test-svc", url="http://localhost:9999/health")]
        checker = ServiceHealthChecker(services=endpoints)

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _FakeResponse(200, body)
            result = checker.check_service("test-svc")

        self.assertEqual(result.status, HealthStatus.DEGRADED)


# ===========================================================================
# 4. HealthDegradationDetector
# ===========================================================================

class TestHealthDegradationDetector(unittest.TestCase):
    def _make_health(self, name: str, ok: bool, latency: float = 10.0) -> ServiceHealth:
        status = HealthStatus.HEALTHY if ok else HealthStatus.UNREACHABLE
        return ServiceHealth(name=name, status=status, latency_ms=latency, last_check=datetime.utcnow())

    def test_failure_rate_alert_fires_after_threshold(self):
        alerts = []
        detector = HealthDegradationDetector(alert_callback=alerts.append)

        for _ in range(4):
            detector.record(self._make_health("svc", ok=False))

        self.assertTrue(any(a.alert_type == "failure_rate" for a in alerts))

    def test_no_alert_below_threshold(self):
        alerts = []
        detector = HealthDegradationDetector(alert_callback=alerts.append)

        # 2 failures -- below threshold of 3
        for _ in range(2):
            detector.record(self._make_health("svc", ok=False))
        for _ in range(5):
            detector.record(self._make_health("svc", ok=True))

        failure_alerts = [a for a in alerts if a.alert_type == "failure_rate"]
        self.assertEqual(len(failure_alerts), 0)

    def test_window_summary(self):
        detector = HealthDegradationDetector()
        for i in range(5):
            detector.record(self._make_health("svc", ok=(i % 2 == 0), latency=float(i * 10)))

        summary = detector.window_summary("svc")
        self.assertEqual(summary["size"], 5)
        self.assertIn("failure_count", summary)
        self.assertIn("mean_latency_ms", summary)


# ===========================================================================
# 5. ProcessManager -- status
# ===========================================================================

class TestProcessManager(unittest.TestCase):
    def test_status_unknown_service(self):
        pm = ProcessManager()
        status = pm.status("nonexistent")
        self.assertFalse(status.running)
        self.assertIsNone(status.pid)

    def test_register_and_status_all(self):
        pm = ProcessManager()
        defn = ServiceDefinition(
            name="test-svc",
            command=["python", "-c", "import time; time.sleep(100)"],
            cwd="/tmp",
        )
        pm.register(defn)
        all_statuses = pm.status_all()
        self.assertIn("test-svc", all_statuses)
        st = all_statuses["test-svc"]
        self.assertFalse(st.running)  # not started yet

    def test_tail_logs_unknown_service(self):
        pm = ProcessManager()
        lines = pm.tail_logs("no-such-service")
        self.assertEqual(len(lines), 1)
        self.assertIn("Unknown service", lines[0])

    def test_process_status_summary(self):
        ps = ProcessStatus(
            name="live-trader",
            pid=12345,
            running=True,
            uptime_s=3600.0,
            restarts=2,
            last_exit_code=None,
        )
        s = ps.summary()
        self.assertIn("RUNNING", s)
        self.assertIn("live-trader", s)
        self.assertIn("12345", s)


# ===========================================================================
# 6. ConfigDiff
# ===========================================================================

class TestConfigDiff(unittest.TestCase):
    def setUp(self):
        self.differ = ConfigDiff()

    def test_no_changes(self):
        cfg = {"a": 1, "b": {"c": 2}}
        changes = self.differ.diff(cfg, cfg)
        self.assertEqual(changes, [])

    def test_added_key(self):
        old = {"a": 1}
        new = {"a": 1, "b": 2}
        changes = self.differ.diff(old, new)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].change_type, "add")
        self.assertEqual(changes[0].path, "b")
        self.assertEqual(changes[0].new_value, 2)

    def test_deleted_key(self):
        old = {"a": 1, "b": 2}
        new = {"a": 1}
        changes = self.differ.diff(old, new)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].change_type, "delete")
        self.assertEqual(changes[0].path, "b")

    def test_modified_value(self):
        old = {"a": {"x": 1}}
        new = {"a": {"x": 99}}
        changes = self.differ.diff(old, new)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].change_type, "modify")
        self.assertEqual(changes[0].path, "a.x")
        self.assertEqual(changes[0].old_value, 1)
        self.assertEqual(changes[0].new_value, 99)

    def test_nested_diff(self):
        old = {"a": {"b": {"c": 1}}}
        new = {"a": {"b": {"c": 1, "d": 2}}}
        changes = self.differ.diff(old, new)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].path, "a.b.d")
        self.assertEqual(changes[0].change_type, "add")


# ===========================================================================
# 7. validate_srfm_config
# ===========================================================================

class TestConfigValidation(unittest.TestCase):
    def _valid_config(self) -> Dict:
        return {
            "services": {"live-trader": {}},
            "trading": {
                "max_position_usd": 100_000,
                "max_daily_loss_usd": 5_000,
            },
            "risk": {
                "max_leverage": 3.0,
                "halt_on_loss_pct": 0.05,
            },
        }

    def test_valid_config_no_errors(self):
        errors = validate_srfm_config(self._valid_config())
        self.assertEqual(errors, [])

    def test_missing_top_level_key(self):
        cfg = self._valid_config()
        del cfg["services"]
        errors = validate_srfm_config(cfg)
        self.assertTrue(any("services" in e for e in errors))

    def test_invalid_leverage(self):
        cfg = self._valid_config()
        cfg["risk"]["max_leverage"] = -1
        errors = validate_srfm_config(cfg)
        self.assertTrue(any("max_leverage" in e for e in errors))

    def test_invalid_halt_pct_above_1(self):
        cfg = self._valid_config()
        cfg["risk"]["halt_on_loss_pct"] = 1.5
        errors = validate_srfm_config(cfg)
        self.assertTrue(any("halt_on_loss_pct" in e for e in errors))


# ===========================================================================
# 8. RunbookExecutor -- validation
# ===========================================================================

class TestRunbookValidation(unittest.TestCase):
    def test_builtin_runbooks_validate_cleanly(self):
        executor = RunbookExecutor()
        for name in ["emergency_stop", "daily_startup", "post_update_validation", "incident_diagnostic"]:
            errors = executor.validate(name)
            self.assertEqual(errors, [], msg=f"Runbook '{name}' has validation errors: {errors}")

    def test_invalid_action(self):
        rb = RunbookDef(
            name="bad_rb",
            description="test",
            steps=[
                RunbookStep(name="step1", action="nonexistent_action", params={"url": "http://x"})
            ],
        )
        executor = RunbookExecutor(extra_runbooks={"bad_rb": rb})
        errors = executor.validate("bad_rb")
        self.assertTrue(any("nonexistent_action" in e for e in errors))

    def test_missing_url_param(self):
        rb = RunbookDef(
            name="bad_rb2",
            description="test",
            steps=[RunbookStep(name="step1", action="http_get", params={})],
        )
        executor = RunbookExecutor(extra_runbooks={"bad_rb2": rb})
        errors = executor.validate("bad_rb2")
        self.assertTrue(any("url" in e for e in errors))

    def test_unknown_runbook(self):
        executor = RunbookExecutor()
        errors = executor.validate("no_such_runbook")
        self.assertEqual(len(errors), 1)
        self.assertIn("not found", errors[0])


# ===========================================================================
# 9. RunbookExecutor -- execution of simple steps
# ===========================================================================

class TestRunbookExecution(unittest.TestCase):
    def test_log_message_step_succeeds(self):
        rb = RunbookDef(
            name="test_rb",
            description="test",
            steps=[
                RunbookStep(
                    name="say_hello",
                    action="log_message",
                    params={"message": "hello from test"},
                )
            ],
        )
        executor = RunbookExecutor(extra_runbooks={"test_rb": rb})
        result = executor.run("test_rb")
        self.assertTrue(result.success)
        self.assertEqual(result.steps_completed, 1)
        self.assertEqual(result.steps_total, 1)
        self.assertTrue(any("hello from test" in line for line in result.log))

    def test_wait_seconds_step(self):
        rb = RunbookDef(
            name="wait_rb",
            description="test",
            steps=[
                RunbookStep(name="short_wait", action="wait_seconds", params={"seconds": 0.05})
            ],
        )
        executor = RunbookExecutor(extra_runbooks={"wait_rb": rb})
        t_start = time.monotonic()
        result = executor.run("wait_rb")
        elapsed = time.monotonic() - t_start
        self.assertTrue(result.success)
        self.assertGreaterEqual(elapsed, 0.05)

    def test_abort_on_failure(self):
        rb = RunbookDef(
            name="abort_rb",
            description="test",
            steps=[
                RunbookStep(
                    name="fail_step",
                    action="http_get",
                    params={"url": "http://localhost:19999/unreachable"},
                    timeout_s=1.0,
                    on_failure="abort",
                ),
                RunbookStep(
                    name="never_reached",
                    action="log_message",
                    params={"message": "should not appear"},
                ),
            ],
        )
        executor = RunbookExecutor(extra_runbooks={"abort_rb": rb})
        result = executor.run("abort_rb")
        self.assertFalse(result.success)
        self.assertEqual(result.steps_completed, 0)
        # The second step log message should not appear
        self.assertFalse(any("should not appear" in l for l in result.log))

    def test_continue_on_failure(self):
        rb = RunbookDef(
            name="continue_rb",
            description="test",
            steps=[
                RunbookStep(
                    name="fail_step",
                    action="http_get",
                    params={"url": "http://localhost:19999/unreachable"},
                    timeout_s=1.0,
                    on_failure="continue",
                ),
                RunbookStep(
                    name="continue_step",
                    action="log_message",
                    params={"message": "reached after failure"},
                ),
            ],
        )
        executor = RunbookExecutor(extra_runbooks={"continue_rb": rb})
        result = executor.run("continue_rb")
        # Overall success even though step 1 failed (only completed steps count)
        self.assertTrue(any("reached after failure" in l for l in result.log))
        self.assertEqual(result.steps_total, 2)

    def test_unknown_runbook_returns_failure(self):
        executor = RunbookExecutor()
        result = executor.run("does_not_exist")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_list_runbooks_includes_builtins(self):
        executor = RunbookExecutor()
        names = {rb.name for rb in executor.list_runbooks()}
        self.assertIn("emergency_stop", names)
        self.assertIn("daily_startup", names)
        self.assertIn("post_update_validation", names)
        self.assertIn("incident_diagnostic", names)


# ===========================================================================
# 10. VersionManager
# ===========================================================================

class TestVersionManager(unittest.TestCase):
    def _make_vm(self) -> VersionManager:
        import tempfile, os
        tmp = tempfile.mktemp(suffix=".db")
        return VersionManager(db_path=tmp)

    def test_default_active_version_is_unknown(self):
        vm = self._make_vm()
        self.assertEqual(vm.get_active_version(), "unknown")

    def test_set_and_get_active_version(self):
        vm = self._make_vm()
        vm.set_active_version("1.2.3")
        self.assertEqual(vm.get_active_version(), "1.2.3")

    def test_save_and_load_record(self):
        vm = self._make_vm()
        rec = DeploymentRecord(
            id="deploy-001",
            version="1.0.0",
            strategy="rolling",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            status="complete",
            services_deployed=["svc-a", "svc-b"],
        )
        vm.save_record(rec)
        history = vm.load_history(n=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].id, "deploy-001")
        self.assertEqual(history[0].services_deployed, ["svc-a", "svc-b"])

    def test_rollback_target_returns_previous_version(self):
        vm = self._make_vm()
        for i, ver in enumerate(["1.0.0", "1.1.0", "1.2.0"]):
            rec = DeploymentRecord(
                id=f"d{i}",
                version=ver,
                strategy="rolling",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                status="complete",
            )
            vm.save_record(rec)
        vm.set_active_version("1.2.0")
        target = vm.get_rollback_target()
        self.assertEqual(target, "1.1.0")


if __name__ == "__main__":
    unittest.main()
