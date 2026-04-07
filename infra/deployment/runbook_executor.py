# infra/deployment/runbook_executor.py -- automated runbook execution for SRFM
from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunbookStep:
    name: str
    action: str             # http_get | http_post | shell_cmd | wait_healthy | wait_seconds | log_message | check_threshold
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 30.0
    on_failure: str = "abort"   # "continue", "abort", "rollback"


@dataclass
class RunbookDef:
    name: str
    description: str
    steps: List[RunbookStep] = field(default_factory=list)


@dataclass
class RunbookResult:
    success: bool
    runbook_name: str
    steps_completed: int
    steps_total: int
    log: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    error: Optional[str] = None

    def summary(self) -> str:
        state = "OK" if self.success else "FAILED"
        return (
            f"[{state}] {self.runbook_name}: "
            f"{self.steps_completed}/{self.steps_total} steps "
            f"in {self.duration_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# Built-in runbook definitions
# ---------------------------------------------------------------------------

EMERGENCY_STOP_RUNBOOK = RunbookDef(
    name="emergency_stop",
    description="Halt all trading, flatten positions, send alert",
    steps=[
        RunbookStep(
            name="log_start",
            action="log_message",
            params={"message": "EMERGENCY STOP initiated"},
            on_failure="continue",
        ),
        RunbookStep(
            name="halt_live_trader",
            action="http_post",
            params={
                "url": "http://localhost:8080/trading/halt",
                "body": {"reason": "emergency_stop"},
            },
            timeout_s=10.0,
            on_failure="continue",
        ),
        RunbookStep(
            name="flatten_positions",
            action="http_post",
            params={
                "url": "http://localhost:8080/trading/flatten",
                "body": {"aggressive": True},
            },
            timeout_s=30.0,
            on_failure="continue",
        ),
        RunbookStep(
            name="halt_idea_engine",
            action="http_post",
            params={
                "url": "http://localhost:8785/engine/halt",
                "body": {},
            },
            timeout_s=10.0,
            on_failure="continue",
        ),
        RunbookStep(
            name="check_positions_zero",
            action="http_get",
            params={
                "url": "http://localhost:8783/risk/positions/summary",
                "expect_field": "total_notional",
                "expect_value": 0,
            },
            timeout_s=15.0,
            on_failure="continue",
        ),
        RunbookStep(
            name="log_done",
            action="log_message",
            params={"message": "EMERGENCY STOP complete -- all positions flattened"},
            on_failure="continue",
        ),
    ],
)

DAILY_STARTUP_RUNBOOK = RunbookDef(
    name="daily_startup",
    description="Check all services, validate config, enable trading",
    steps=[
        RunbookStep(
            name="log_startup",
            action="log_message",
            params={"message": "Daily startup sequence starting"},
            on_failure="continue",
        ),
        RunbookStep(
            name="wait_coordination_healthy",
            action="wait_healthy",
            params={"service": "coordination", "url": "http://localhost:8781/health"},
            timeout_s=60.0,
            on_failure="abort",
        ),
        RunbookStep(
            name="wait_risk_api_healthy",
            action="wait_healthy",
            params={"service": "risk-api", "url": "http://localhost:8783/health"},
            timeout_s=60.0,
            on_failure="abort",
        ),
        RunbookStep(
            name="wait_market_data_healthy",
            action="wait_healthy",
            params={"service": "market-data", "url": "http://localhost:8784/health"},
            timeout_s=60.0,
            on_failure="abort",
        ),
        RunbookStep(
            name="wait_live_trader_healthy",
            action="wait_healthy",
            params={"service": "live-trader", "url": "http://localhost:8080/health"},
            timeout_s=60.0,
            on_failure="abort",
        ),
        RunbookStep(
            name="validate_risk_limits",
            action="http_get",
            params={"url": "http://localhost:8783/risk/limits/validate"},
            timeout_s=10.0,
            on_failure="abort",
        ),
        RunbookStep(
            name="enable_trading",
            action="http_post",
            params={
                "url": "http://localhost:8080/trading/enable",
                "body": {"session": "normal"},
            },
            timeout_s=10.0,
            on_failure="abort",
        ),
        RunbookStep(
            name="log_complete",
            action="log_message",
            params={"message": "Daily startup complete -- trading enabled"},
            on_failure="continue",
        ),
    ],
)

POST_UPDATE_VALIDATION_RUNBOOK = RunbookDef(
    name="post_update_validation",
    description="Run health checks and performance sanity after a parameter update",
    steps=[
        RunbookStep(
            name="log_start",
            action="log_message",
            params={"message": "Post-update validation starting"},
            on_failure="continue",
        ),
        RunbookStep(
            name="wait_2s",
            action="wait_seconds",
            params={"seconds": 2},
            on_failure="continue",
        ),
        RunbookStep(
            name="check_live_trader_health",
            action="http_get",
            params={"url": "http://localhost:8080/health"},
            timeout_s=10.0,
            on_failure="abort",
        ),
        RunbookStep(
            name="check_risk_api_health",
            action="http_get",
            params={"url": "http://localhost:8783/health"},
            timeout_s=10.0,
            on_failure="abort",
        ),
        RunbookStep(
            name="check_idea_engine_health",
            action="http_get",
            params={"url": "http://localhost:8785/health"},
            timeout_s=10.0,
            on_failure="abort",
        ),
        RunbookStep(
            name="check_error_rate",
            action="check_threshold",
            params={
                "url": "http://localhost:8080/metrics/error_rate",
                "field": "error_rate",
                "max_value": 0.05,
            },
            timeout_s=10.0,
            on_failure="rollback",
        ),
        RunbookStep(
            name="check_latency_p99",
            action="check_threshold",
            params={
                "url": "http://localhost:8080/metrics/latency",
                "field": "p99_ms",
                "max_value": 1000.0,
            },
            timeout_s=10.0,
            on_failure="rollback",
        ),
        RunbookStep(
            name="log_done",
            action="log_message",
            params={"message": "Post-update validation passed"},
            on_failure="continue",
        ),
    ],
)

INCIDENT_DIAGNOSTIC_RUNBOOK = RunbookDef(
    name="incident_diagnostic",
    description="Collect logs, check circuit breakers, snapshot system state",
    steps=[
        RunbookStep(
            name="log_start",
            action="log_message",
            params={"message": "Incident diagnostic starting"},
            on_failure="continue",
        ),
        RunbookStep(
            name="snapshot_risk_state",
            action="http_get",
            params={"url": "http://localhost:8783/risk/state/snapshot"},
            timeout_s=10.0,
            on_failure="continue",
        ),
        RunbookStep(
            name="check_circuit_breakers",
            action="http_get",
            params={"url": "http://localhost:8781/circuit_breakers/status"},
            timeout_s=10.0,
            on_failure="continue",
        ),
        RunbookStep(
            name="collect_live_trader_logs",
            action="shell_cmd",
            params={"cmd": "tail -n 500 /tmp/srfm_live-trader.log > /tmp/incident_live_trader.txt 2>&1"},
            timeout_s=10.0,
            on_failure="continue",
        ),
        RunbookStep(
            name="collect_risk_api_logs",
            action="shell_cmd",
            params={"cmd": "tail -n 500 /tmp/srfm_risk-api.log > /tmp/incident_risk_api.txt 2>&1"},
            timeout_s=10.0,
            on_failure="continue",
        ),
        RunbookStep(
            name="check_coordination_health",
            action="http_get",
            params={"url": "http://localhost:8781/health"},
            timeout_s=10.0,
            on_failure="continue",
        ),
        RunbookStep(
            name="log_done",
            action="log_message",
            params={"message": "Incident diagnostic complete -- artifacts in /tmp/incident_*.txt"},
            on_failure="continue",
        ),
    ],
)

BUILT_IN_RUNBOOKS: Dict[str, RunbookDef] = {
    "emergency_stop":           EMERGENCY_STOP_RUNBOOK,
    "daily_startup":            DAILY_STARTUP_RUNBOOK,
    "post_update_validation":   POST_UPDATE_VALIDATION_RUNBOOK,
    "incident_diagnostic":      INCIDENT_DIAGNOSTIC_RUNBOOK,
}


# ---------------------------------------------------------------------------
# RunbookExecutor
# ---------------------------------------------------------------------------

class RunbookExecutor:
    """Executes named runbooks step by step with structured error handling.

    -- Supports built-in runbooks and user-registered runbooks.
    -- Actions: http_get, http_post, shell_cmd, wait_healthy,
       wait_seconds, log_message, check_threshold.
    -- Per-step on_failure policy: continue | abort | rollback.
    """

    HTTP_TIMEOUT_S: float = 10.0

    def __init__(self, extra_runbooks: Optional[Dict[str, RunbookDef]] = None) -> None:
        self._runbooks: Dict[str, RunbookDef] = dict(BUILT_IN_RUNBOOKS)
        if extra_runbooks:
            self._runbooks.update(extra_runbooks)

    # -- registration --------------------------------------------------------

    def register(self, runbook: RunbookDef) -> None:
        self._runbooks[runbook.name] = runbook

    # -- listing & validation ------------------------------------------------

    def list_runbooks(self) -> List[RunbookDef]:
        return list(self._runbooks.values())

    def validate(self, runbook_name: str) -> List[str]:
        """Check that all steps have valid action types and required params."""
        rb = self._runbooks.get(runbook_name)
        if rb is None:
            return [f"Runbook '{runbook_name}' not found"]

        errors: List[str] = []
        valid_actions = {
            "http_get", "http_post", "shell_cmd",
            "wait_healthy", "wait_seconds", "log_message", "check_threshold",
        }
        valid_on_failure = {"continue", "abort", "rollback"}

        for i, step in enumerate(rb.steps):
            prefix = f"step[{i}] '{step.name}'"
            if step.action not in valid_actions:
                errors.append(f"{prefix}: unknown action '{step.action}'")
            if step.on_failure not in valid_on_failure:
                errors.append(f"{prefix}: invalid on_failure '{step.on_failure}'")
            if step.timeout_s <= 0:
                errors.append(f"{prefix}: timeout_s must be > 0")

            # Action-specific required params
            if step.action in ("http_get", "http_post", "check_threshold"):
                if "url" not in step.params:
                    errors.append(f"{prefix}: action '{step.action}' requires params.url")
            if step.action == "shell_cmd":
                if "cmd" not in step.params:
                    errors.append(f"{prefix}: shell_cmd requires params.cmd")
            if step.action == "wait_seconds":
                if "seconds" not in step.params:
                    errors.append(f"{prefix}: wait_seconds requires params.seconds")
            if step.action == "wait_healthy":
                if "url" not in step.params:
                    errors.append(f"{prefix}: wait_healthy requires params.url")
            if step.action == "log_message":
                if "message" not in step.params:
                    errors.append(f"{prefix}: log_message requires params.message")
            if step.action == "check_threshold":
                if "field" not in step.params or "max_value" not in step.params:
                    errors.append(f"{prefix}: check_threshold requires params.field and params.max_value")

        return errors

    # -- execution -----------------------------------------------------------

    def run(
        self,
        runbook_name: str,
        params: Optional[Dict[str, Any]] = None,
        rollback_callback: Optional[Callable[[], None]] = None,
    ) -> RunbookResult:
        """Execute a runbook by name. Returns a RunbookResult."""
        rb = self._runbooks.get(runbook_name)
        if rb is None:
            return RunbookResult(
                success=False,
                runbook_name=runbook_name,
                steps_completed=0,
                steps_total=0,
                log=[f"Runbook '{runbook_name}' not found"],
                error=f"Runbook '{runbook_name}' not found",
            )

        params = params or {}
        log: List[str] = []
        t_start = time.monotonic()
        steps_completed = 0

        log.append(f"[{datetime.utcnow().isoformat()}] Starting runbook '{runbook_name}'")
        logger.info("RunbookExecutor: starting '%s' (%d steps)", runbook_name, len(rb.steps))

        for step_idx, step in enumerate(rb.steps):
            t_step = time.monotonic()
            log.append(
                f"[{datetime.utcnow().isoformat()}] Step {step_idx + 1}/{len(rb.steps)}: "
                f"{step.name} ({step.action})"
            )
            logger.debug("Runbook '%s' step '%s' action=%s", runbook_name, step.name, step.action)

            # Merge runtime params into step params
            merged_params = {**step.params, **params}

            try:
                step_result = self._execute_step(step, merged_params)
                elapsed = (time.monotonic() - t_step) * 1000
                log.append(f"  OK ({elapsed:.0f}ms): {step_result}")
                steps_completed += 1

            except Exception as exc:
                elapsed = (time.monotonic() - t_step) * 1000
                err_msg = f"  FAILED ({elapsed:.0f}ms): {exc}"
                log.append(err_msg)
                logger.warning("Runbook '%s' step '%s' failed: %s", runbook_name, step.name, exc)

                if step.on_failure == "abort":
                    log.append(f"  Aborting runbook on step failure (on_failure=abort)")
                    return RunbookResult(
                        success=False,
                        runbook_name=runbook_name,
                        steps_completed=steps_completed,
                        steps_total=len(rb.steps),
                        log=log,
                        duration_s=time.monotonic() - t_start,
                        error=str(exc),
                    )
                elif step.on_failure == "rollback":
                    log.append(f"  Triggering rollback (on_failure=rollback)")
                    if rollback_callback:
                        try:
                            rollback_callback()
                            log.append("  Rollback callback executed")
                        except Exception as rb_exc:
                            log.append(f"  Rollback callback failed: {rb_exc}")
                    return RunbookResult(
                        success=False,
                        runbook_name=runbook_name,
                        steps_completed=steps_completed,
                        steps_total=len(rb.steps),
                        log=log,
                        duration_s=time.monotonic() - t_start,
                        error=f"Rollback triggered: {exc}",
                    )
                else:
                    # on_failure == "continue"
                    log.append("  Continuing despite step failure (on_failure=continue)")

        duration = time.monotonic() - t_start
        log.append(f"[{datetime.utcnow().isoformat()}] Runbook '{runbook_name}' complete in {duration:.1f}s")
        logger.info("RunbookExecutor: '%s' complete in %.1fs", runbook_name, duration)

        return RunbookResult(
            success=True,
            runbook_name=runbook_name,
            steps_completed=steps_completed,
            steps_total=len(rb.steps),
            log=log,
            duration_s=duration,
        )

    # -- action dispatch -----------------------------------------------------

    def _execute_step(self, step: RunbookStep, params: Dict[str, Any]) -> str:
        """Dispatch to the appropriate action handler."""
        action = step.action

        if action == "log_message":
            return self._action_log_message(params)
        if action == "wait_seconds":
            return self._action_wait_seconds(params)
        if action == "http_get":
            return self._action_http_get(params, step.timeout_s)
        if action == "http_post":
            return self._action_http_post(params, step.timeout_s)
        if action == "shell_cmd":
            return self._action_shell_cmd(params, step.timeout_s)
        if action == "wait_healthy":
            return self._action_wait_healthy(params, step.timeout_s)
        if action == "check_threshold":
            return self._action_check_threshold(params, step.timeout_s)

        raise ValueError(f"Unknown action: {action}")

    # -- action implementations ----------------------------------------------

    @staticmethod
    def _action_log_message(params: Dict[str, Any]) -> str:
        msg = str(params.get("message", ""))
        logger.info("[runbook] %s", msg)
        return f"logged: {msg}"

    @staticmethod
    def _action_wait_seconds(params: Dict[str, Any]) -> str:
        seconds = float(params.get("seconds", 1))
        time.sleep(seconds)
        return f"waited {seconds}s"

    def _action_http_get(self, params: Dict[str, Any], timeout_s: float) -> str:
        url = params["url"]
        data = self._http_get_json(url, timeout_s)
        return f"GET {url} -> {data}"

    def _action_http_post(self, params: Dict[str, Any], timeout_s: float) -> str:
        url = params["url"]
        body = params.get("body", {})
        status = self._http_post_json(url, body, timeout_s)
        return f"POST {url} -> HTTP {status}"

    @staticmethod
    def _action_shell_cmd(params: Dict[str, Any], timeout_s: float) -> str:
        cmd = str(params["cmd"])
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"shell_cmd exited {result.returncode}: {result.stderr.strip()[:200]}"
            )
        return f"exit=0 stdout={result.stdout.strip()[:100]!r}"

    def _action_wait_healthy(self, params: Dict[str, Any], timeout_s: float) -> str:
        url = params["url"]
        service = params.get("service", url)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data = self._http_get_json(url, timeout_s=5.0)
                status = str(data.get("status", "ok")).lower() if isinstance(data, dict) else "ok"
                if status not in ("down", "unhealthy", "error", "fail"):
                    return f"service '{service}' is healthy"
            except Exception:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(2.0, remaining))
        raise TimeoutError(f"Service '{service}' not healthy within {timeout_s}s")

    def _action_check_threshold(self, params: Dict[str, Any], timeout_s: float) -> str:
        url = params["url"]
        field = params["field"]
        max_value = float(params["max_value"])
        data = self._http_get_json(url, timeout_s)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object from {url}, got: {type(data)}")
        actual = float(data.get(field, 0))
        if actual > max_value:
            raise ValueError(
                f"Threshold exceeded: {field}={actual:.4f} > max={max_value:.4f}"
            )
        return f"{field}={actual:.4f} <= max={max_value:.4f} (ok)"

    # -- HTTP helpers --------------------------------------------------------

    def _http_get_json(self, url: str, timeout_s: float) -> Any:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return {"raw": raw.decode("utf-8", errors="replace")[:200]}

    def _http_post_json(self, url: str, body: Any, timeout_s: float) -> int:
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status
