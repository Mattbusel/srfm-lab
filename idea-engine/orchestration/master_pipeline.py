"""
master_pipeline.py -- Master orchestration pipeline for the idea engine.

Pipeline configuration, stage execution, retry policies, dependency graphs,
circuit breakers, monitoring, scheduling, versioning, and A/B testing.

All heavy work uses numpy/scipy where relevant.  Framework is pure Python.
"""

from __future__ import annotations

import enum
import hashlib
import heapq
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


# ===================================================================
# 1.  Enums and status codes
# ===================================================================

class StageStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    CIRCUIT_OPEN = "circuit_open"
    TIMEOUT = "timeout"


class PipelineStatus(enum.Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class StageName(enum.Enum):
    DATA_COLLECTION = "data_collection"
    FEATURE_COMPUTATION = "feature_computation"
    SIGNAL_GENERATION = "signal_generation"
    IDEA_SYNTHESIS = "idea_synthesis"
    DEBATE = "debate"
    VALIDATION = "validation"
    RISK_CHECK = "risk_check"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    FEEDBACK = "feedback"


# ===================================================================
# 2.  Retry Policy
# ===================================================================

@dataclass
class RetryPolicy:
    """Exponential backoff retry configuration."""
    max_retries: int = 3
    initial_delay_sec: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_sec: float = 60.0
    jitter: bool = True
    retryable_exceptions: List[str] = field(default_factory=lambda: ["TimeoutError", "ConnectionError"])

    def compute_delay(self, attempt: int) -> float:
        delay = self.initial_delay_sec * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay_sec)
        if self.jitter:
            delay *= np.random.uniform(0.5, 1.5)
        return delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self.max_retries:
            return False
        exc_name = type(exception).__name__
        if self.retryable_exceptions and exc_name not in self.retryable_exceptions:
            return False
        return True


# ===================================================================
# 3.  Dead Letter Queue
# ===================================================================

@dataclass
class DeadLetterItem:
    stage_name: str
    timestamp: float
    attempt: int
    error: str
    input_data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeadLetterQueue:
    """Store failed items for later inspection / replay."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._items: List[DeadLetterItem] = []

    def add(self, item: DeadLetterItem) -> None:
        self._items.append(item)
        if len(self._items) > self.max_size:
            self._items.pop(0)

    def get_all(self) -> List[DeadLetterItem]:
        return list(self._items)

    def get_by_stage(self, stage_name: str) -> List[DeadLetterItem]:
        return [i for i in self._items if i.stage_name == stage_name]

    def clear(self) -> None:
        self._items.clear()

    @property
    def size(self) -> int:
        return len(self._items)

    def replay_items(self, stage_fn: Callable[[Any], Any]) -> List[Any]:
        results = []
        replayed = []
        for item in self._items:
            try:
                result = stage_fn(item.input_data)
                results.append(result)
                replayed.append(item)
            except Exception:
                pass
        for item in replayed:
            self._items.remove(item)
        return results


# ===================================================================
# 4.  Circuit Breaker
# ===================================================================

class CircuitState(enum.Enum):
    CLOSED = "closed"           # normal operation
    OPEN = "open"               # failing, reject calls
    HALF_OPEN = "half_open"     # testing if recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_sec: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """Per-stage circuit breaker for fault tolerance."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.cfg = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed > self.cfg.recovery_timeout_sec:
                self.state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.cfg.half_open_max_calls
        return False

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls += 1
            if self._success_count >= self.cfg.success_threshold:
                self.state = CircuitState.CLOSED
                self._failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self._failure_count = max(self._failure_count - 1, 0)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self._failure_count >= self.cfg.failure_threshold:
            self.state = CircuitState.OPEN

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0


# ===================================================================
# 5.  Stage Result
# ===================================================================

@dataclass
class StageResult:
    stage_name: str
    status: StageStatus
    duration_sec: float
    output: Any = None
    error: str = ""
    attempt: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == StageStatus.SUCCESS


# ===================================================================
# 6.  Stage Definition
# ===================================================================

@dataclass
class StageDefinition:
    """Definition of a pipeline stage."""
    name: str
    execute_fn: Callable[[Any], Any]
    timeout_sec: float = 300.0
    retry_policy: RetryPolicy | None = None
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    critical: bool = True       # if True, pipeline fails when stage fails
    parallelizable: bool = False
    circuit_breaker_config: CircuitBreakerConfig | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===================================================================
# 7.  Dependency Graph
# ===================================================================

class DependencyGraph:
    """DAG of stage dependencies with topological sort."""

    def __init__(self):
        self._edges: Dict[str, Set[str]] = defaultdict(set)
        self._nodes: Set[str] = set()

    def add_node(self, name: str) -> None:
        self._nodes.add(name)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """from_node must complete before to_node."""
        self._nodes.add(from_node)
        self._nodes.add(to_node)
        self._edges[to_node].add(from_node)

    def topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological ordering."""
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        adj: Dict[str, List[str]] = defaultdict(list)
        for node, deps in self._edges.items():
            in_degree.setdefault(node, 0)
            for dep in deps:
                adj[dep].append(node)
                in_degree[node] = in_degree.get(node, 0) + 1

        queue = [n for n in self._nodes if in_degree.get(n, 0) == 0]
        heapq.heapify(queue)
        result = []
        while queue:
            node = heapq.heappop(queue)
            result.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(queue, neighbor)

        if len(result) != len(self._nodes):
            raise ValueError("Cycle detected in dependency graph")
        return result

    def get_ready_stages(self, completed: Set[str]) -> List[str]:
        """Get stages whose dependencies are all completed."""
        ready = []
        for node in self._nodes:
            if node in completed:
                continue
            deps = self._edges.get(node, set())
            if deps.issubset(completed):
                ready.append(node)
        return sorted(ready)

    def get_dependencies(self, stage: str) -> Set[str]:
        return self._edges.get(stage, set())

    def validate(self) -> bool:
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False


# ===================================================================
# 8.  Pipeline Configuration
# ===================================================================

@dataclass
class PipelineConfig:
    name: str = "master_pipeline"
    version: str = "1.0.0"
    stages: List[StageDefinition] = field(default_factory=list)
    global_timeout_sec: float = 3600.0
    max_parallel: int = 4
    fail_fast: bool = False
    default_retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    dead_letter_queue_size: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: StageDefinition) -> None:
        self.stages.append(stage)

    def get_stage(self, name: str) -> StageDefinition | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def build_dependency_graph(self) -> DependencyGraph:
        graph = DependencyGraph()
        for stage in self.stages:
            graph.add_node(stage.name)
            for dep in stage.dependencies:
                graph.add_edge(dep, stage.name)
        return graph


# ===================================================================
# 9.  Pipeline Monitor
# ===================================================================

@dataclass
class StageMetrics:
    total_runs: int = 0
    successes: int = 0
    failures: int = 0
    total_duration_sec: float = 0.0
    max_duration_sec: float = 0.0
    min_duration_sec: float = float("inf")
    retry_count: int = 0
    circuit_opens: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.successes / self.total_runs

    @property
    def avg_duration_sec(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.total_duration_sec / self.total_runs


class PipelineMonitor:
    """Track pipeline execution metrics."""

    def __init__(self):
        self._stage_metrics: Dict[str, StageMetrics] = defaultdict(StageMetrics)
        self._pipeline_runs: List[Dict[str, Any]] = []
        self._current_run: Dict[str, Any] = {}

    def start_pipeline_run(self, run_id: str) -> None:
        self._current_run = {
            "run_id": run_id,
            "start_time": time.time(),
            "stage_results": [],
        }

    def record_stage_result(self, result: StageResult) -> None:
        metrics = self._stage_metrics[result.stage_name]
        metrics.total_runs += 1
        metrics.total_duration_sec += result.duration_sec
        metrics.max_duration_sec = max(metrics.max_duration_sec, result.duration_sec)
        metrics.min_duration_sec = min(metrics.min_duration_sec, result.duration_sec)
        if result.succeeded:
            metrics.successes += 1
        else:
            metrics.failures += 1
        if result.attempt > 1:
            metrics.retry_count += result.attempt - 1
        if result.status == StageStatus.CIRCUIT_OPEN:
            metrics.circuit_opens += 1
        if self._current_run:
            self._current_run["stage_results"].append({
                "name": result.stage_name,
                "status": result.status.value,
                "duration": result.duration_sec,
            })

    def end_pipeline_run(self, status: PipelineStatus) -> None:
        if self._current_run:
            self._current_run["end_time"] = time.time()
            self._current_run["status"] = status.value
            self._current_run["total_duration"] = (
                self._current_run["end_time"] - self._current_run["start_time"]
            )
            self._pipeline_runs.append(self._current_run)
            self._current_run = {}

    def get_stage_metrics(self, stage_name: str) -> StageMetrics:
        return self._stage_metrics[stage_name]

    def get_bottlenecks(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """Identify slowest stages."""
        stage_times = [
            (name, m.avg_duration_sec)
            for name, m in self._stage_metrics.items()
        ]
        stage_times.sort(key=lambda x: -x[1])
        return stage_times[:top_n]

    def get_failure_hotspots(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """Identify stages with highest failure rates."""
        stage_failures = [
            (name, 1.0 - m.success_rate)
            for name, m in self._stage_metrics.items()
            if m.total_runs > 0
        ]
        stage_failures.sort(key=lambda x: -x[1])
        return stage_failures[:top_n]

    def summary(self) -> Dict[str, Any]:
        total_runs = len(self._pipeline_runs)
        successes = sum(
            1 for r in self._pipeline_runs if r.get("status") == "completed"
        )
        avg_duration = (
            np.mean([r["total_duration"] for r in self._pipeline_runs])
            if self._pipeline_runs
            else 0.0
        )
        return {
            "total_pipeline_runs": total_runs,
            "success_rate": successes / max(total_runs, 1),
            "avg_duration_sec": float(avg_duration),
            "bottlenecks": self.get_bottlenecks(),
            "failure_hotspots": self.get_failure_hotspots(),
            "stage_count": len(self._stage_metrics),
        }


# ===================================================================
# 10. Pipeline Executor
# ===================================================================

class PipelineExecutor:
    """Execute pipeline stages respecting dependencies, retries, circuit breakers."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.monitor = PipelineMonitor()
        self.dlq = DeadLetterQueue(config.dead_letter_queue_size)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._results: Dict[str, StageResult] = {}
        self._dep_graph = config.build_dependency_graph()

        for stage in config.stages:
            cb_cfg = stage.circuit_breaker_config or CircuitBreakerConfig()
            self._circuit_breakers[stage.name] = CircuitBreaker(stage.name, cb_cfg)

    def run(self, initial_input: Any = None) -> Dict[str, StageResult]:
        """Execute the full pipeline."""
        run_id = str(uuid.uuid4())[:8]
        self.monitor.start_pipeline_run(run_id)
        self._results.clear()

        try:
            execution_order = self._dep_graph.topological_sort()
        except ValueError as e:
            self.monitor.end_pipeline_run(PipelineStatus.FAILED)
            return {"__error__": StageResult("__error__", StageStatus.FAILED, 0, error=str(e))}

        completed: Set[str] = set()
        pipeline_failed = False
        stage_input = initial_input

        for stage_name in execution_order:
            stage_def = self.config.get_stage(stage_name)
            if stage_def is None:
                continue
            if not stage_def.enabled:
                self._results[stage_name] = StageResult(
                    stage_name, StageStatus.SKIPPED, 0.0
                )
                completed.add(stage_name)
                continue

            # Check dependencies
            deps = self._dep_graph.get_dependencies(stage_name)
            deps_met = all(
                d in completed and self._results.get(d, StageResult(d, StageStatus.FAILED, 0)).succeeded
                for d in deps
            )
            if not deps_met:
                if stage_def.critical and self.config.fail_fast:
                    pipeline_failed = True
                    break
                self._results[stage_name] = StageResult(
                    stage_name, StageStatus.SKIPPED, 0.0,
                    error="Dependencies not met",
                )
                completed.add(stage_name)
                continue

            # Gather inputs from dependencies
            dep_outputs = {
                d: self._results[d].output
                for d in deps
                if d in self._results and self._results[d].output is not None
            }
            if dep_outputs:
                stage_input = dep_outputs
            elif initial_input is not None:
                stage_input = initial_input

            result = self._execute_stage(stage_def, stage_input)
            self._results[stage_name] = result
            self.monitor.record_stage_result(result)
            completed.add(stage_name)

            if not result.succeeded and stage_def.critical:
                if self.config.fail_fast:
                    pipeline_failed = True
                    break

        status = PipelineStatus.COMPLETED
        if pipeline_failed:
            status = PipelineStatus.FAILED
        elif any(
            not r.succeeded
            for r in self._results.values()
            if r.status != StageStatus.SKIPPED
        ):
            status = PipelineStatus.PARTIAL

        self.monitor.end_pipeline_run(status)
        return self._results

    def _execute_stage(
        self, stage: StageDefinition, input_data: Any
    ) -> StageResult:
        """Execute a single stage with retries and circuit breaker."""
        cb = self._circuit_breakers.get(stage.name)

        if cb and not cb.can_execute():
            return StageResult(
                stage.name, StageStatus.CIRCUIT_OPEN, 0.0,
                error="Circuit breaker is open",
            )

        retry_policy = stage.retry_policy or self.config.default_retry_policy
        last_error = ""

        for attempt in range(retry_policy.max_retries + 1):
            start = time.time()
            try:
                output = stage.execute_fn(input_data)
                duration = time.time() - start
                if cb:
                    cb.record_success()
                return StageResult(
                    stage.name, StageStatus.SUCCESS, duration,
                    output=output, attempt=attempt + 1,
                )
            except Exception as e:
                duration = time.time() - start
                last_error = f"{type(e).__name__}: {str(e)}"
                if cb:
                    cb.record_failure()
                if not retry_policy.should_retry(attempt, e):
                    break
                delay = retry_policy.compute_delay(attempt)
                # In production would sleep here; we skip for simulation
                # time.sleep(delay)

        # All retries exhausted
        self.dlq.add(DeadLetterItem(
            stage_name=stage.name,
            timestamp=time.time(),
            attempt=attempt + 1,
            error=last_error,
            input_data=input_data,
        ))
        return StageResult(
            stage.name, StageStatus.FAILED, duration,
            error=last_error, attempt=attempt + 1,
        )

    def get_results(self) -> Dict[str, StageResult]:
        return self._results

    def reset(self) -> None:
        self._results.clear()
        for cb in self._circuit_breakers.values():
            cb.reset()


# ===================================================================
# 11. Pipeline Report
# ===================================================================

@dataclass
class PipelineReport:
    """End-to-end pipeline execution report."""
    run_id: str
    status: PipelineStatus
    total_duration_sec: float
    stage_results: Dict[str, StageResult]
    stage_metrics: Dict[str, StageMetrics]
    bottlenecks: List[Tuple[str, float]]
    failure_hotspots: List[Tuple[str, float]]
    dlq_size: int
    recommendations: List[str]

    @classmethod
    def from_executor(cls, executor: PipelineExecutor, run_id: str = "") -> "PipelineReport":
        summary = executor.monitor.summary()
        results = executor.get_results()
        total_dur = sum(r.duration_sec for r in results.values())
        recommendations = []
        for name, rate in executor.monitor.get_failure_hotspots():
            if rate > 0.3:
                recommendations.append(
                    f"Stage '{name}' has {rate:.0%} failure rate -- consider increasing retries or adding fallback."
                )
        for name, dur in executor.monitor.get_bottlenecks():
            if dur > 60:
                recommendations.append(
                    f"Stage '{name}' averages {dur:.1f}s -- consider parallelization or caching."
                )
        if executor.dlq.size > 10:
            recommendations.append(
                f"DLQ has {executor.dlq.size} items -- review and replay or discard."
            )
        return cls(
            run_id=run_id,
            status=PipelineStatus(summary.get("status", "completed")) if "status" in summary else PipelineStatus.COMPLETED,
            total_duration_sec=total_dur,
            stage_results=results,
            stage_metrics={
                name: executor.monitor.get_stage_metrics(name)
                for name in [s.name for s in executor.config.stages]
            },
            bottlenecks=summary["bottlenecks"],
            failure_hotspots=summary["failure_hotspots"],
            dlq_size=executor.dlq.size,
            recommendations=recommendations,
        )


# ===================================================================
# 12. Scheduled Pipeline
# ===================================================================

@dataclass
class MarketHours:
    """Market hours configuration."""
    open_hour: int = 9
    open_minute: int = 30
    close_hour: int = 16
    close_minute: int = 0
    timezone: str = "US/Eastern"
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    def is_market_open(self, hour: int, minute: int, weekday: int) -> bool:
        if weekday not in self.trading_days:
            return False
        current = hour * 60 + minute
        open_t = self.open_hour * 60 + self.open_minute
        close_t = self.close_hour * 60 + self.close_minute
        return open_t <= current < close_t

    def minutes_to_close(self, hour: int, minute: int) -> int:
        close_t = self.close_hour * 60 + self.close_minute
        current = hour * 60 + minute
        return max(close_t - current, 0)


@dataclass
class ScheduleEntry:
    """Cron-like schedule for pipeline execution."""
    name: str
    cron_expression: str         # simplified: "HH:MM" or "*/N" (every N minutes)
    pipeline_config: PipelineConfig
    market_hours_only: bool = True
    enabled: bool = True
    last_run: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScheduledPipeline:
    """Cron-like scheduler with market hours awareness."""

    def __init__(self, market_hours: MarketHours | None = None):
        self.market_hours = market_hours or MarketHours()
        self.schedules: List[ScheduleEntry] = []
        self._executors: Dict[str, PipelineExecutor] = {}

    def add_schedule(self, entry: ScheduleEntry) -> None:
        self.schedules.append(entry)
        self._executors[entry.name] = PipelineExecutor(entry.pipeline_config)

    def check_due(self, current_time: float, hour: int, minute: int, weekday: int) -> List[str]:
        """Check which pipelines are due to run."""
        due = []
        for entry in self.schedules:
            if not entry.enabled:
                continue
            if entry.market_hours_only and not self.market_hours.is_market_open(hour, minute, weekday):
                continue
            interval = self._parse_interval(entry.cron_expression)
            if interval > 0:
                if current_time - entry.last_run >= interval:
                    due.append(entry.name)
            else:
                target_h, target_m = self._parse_time(entry.cron_expression)
                if hour == target_h and minute == target_m and current_time - entry.last_run > 60:
                    due.append(entry.name)
        return due

    def run_due(
        self, current_time: float, hour: int, minute: int, weekday: int,
        initial_input: Any = None,
    ) -> Dict[str, Dict[str, StageResult]]:
        """Run all due pipelines."""
        due = self.check_due(current_time, hour, minute, weekday)
        results = {}
        for name in due:
            entry = next(e for e in self.schedules if e.name == name)
            executor = self._executors[name]
            results[name] = executor.run(initial_input)
            entry.last_run = current_time
        return results

    def _parse_interval(self, cron: str) -> float:
        if cron.startswith("*/"):
            try:
                minutes = int(cron[2:])
                return minutes * 60.0
            except ValueError:
                return 0
        return 0

    def _parse_time(self, cron: str) -> Tuple[int, int]:
        if ":" in cron and not cron.startswith("*/"):
            parts = cron.split(":")
            return int(parts[0]), int(parts[1])
        return -1, -1


# ===================================================================
# 13. Pipeline Versioning
# ===================================================================

@dataclass
class PipelineVersion:
    version: str
    config_hash: str
    config: PipelineConfig
    created_at: float = field(default_factory=time.time)
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


class PipelineVersionManager:
    """Track pipeline configuration changes and support A/B testing."""

    def __init__(self):
        self._versions: Dict[str, PipelineVersion] = {}
        self._active_version: str = ""
        self._ab_tests: Dict[str, Dict[str, float]] = {}  # test_name -> {version: weight}

    def register_version(self, config: PipelineConfig, description: str = "") -> str:
        """Register a new pipeline version."""
        config_str = f"{config.name}_{config.version}_{len(config.stages)}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        version = PipelineVersion(
            version=config.version,
            config_hash=config_hash,
            config=config,
            description=description,
        )
        self._versions[config.version] = version
        if not self._active_version:
            self._active_version = config.version
        return config.version

    def get_version(self, version: str) -> PipelineVersion | None:
        return self._versions.get(version)

    def get_active(self) -> PipelineVersion | None:
        return self._versions.get(self._active_version)

    def set_active(self, version: str) -> None:
        if version in self._versions:
            self._active_version = version

    def setup_ab_test(
        self,
        test_name: str,
        version_weights: Dict[str, float],
    ) -> None:
        """Set up an A/B test between pipeline versions.
        version_weights: {version_id: probability_weight}."""
        total = sum(version_weights.values())
        self._ab_tests[test_name] = {
            v: w / total for v, w in version_weights.items()
        }

    def select_version_for_ab(self, test_name: str) -> str:
        """Select a version based on A/B test weights."""
        if test_name not in self._ab_tests:
            return self._active_version
        weights = self._ab_tests[test_name]
        versions = list(weights.keys())
        probs = list(weights.values())
        return str(np.random.choice(versions, p=probs))

    def record_metrics(self, version: str, metrics: Dict[str, float]) -> None:
        if version in self._versions:
            self._versions[version].metrics.update(metrics)

    def compare_versions(self, v1: str, v2: str) -> Dict[str, Any]:
        ver1 = self._versions.get(v1)
        ver2 = self._versions.get(v2)
        if not ver1 or not ver2:
            return {"error": "Version not found"}
        return {
            "v1": {"version": v1, "metrics": ver1.metrics},
            "v2": {"version": v2, "metrics": ver2.metrics},
            "stage_diff": {
                "v1_only": [s.name for s in ver1.config.stages if s.name not in [s2.name for s2 in ver2.config.stages]],
                "v2_only": [s.name for s in ver2.config.stages if s.name not in [s1.name for s1 in ver1.config.stages]],
            },
        }

    def list_versions(self) -> List[Dict[str, Any]]:
        return [
            {
                "version": v.version,
                "hash": v.config_hash,
                "created": v.created_at,
                "description": v.description,
                "active": v.version == self._active_version,
                "n_stages": len(v.config.stages),
            }
            for v in self._versions.values()
        ]


# ===================================================================
# 14. Default Pipeline Builders
# ===================================================================

def _noop_stage(name: str) -> Callable[[Any], Any]:
    def fn(input_data: Any) -> Any:
        return {"stage": name, "status": "ok", "input_type": type(input_data).__name__}
    return fn


def build_default_pipeline() -> PipelineConfig:
    """Build the default idea-engine pipeline with all 10 stages."""
    config = PipelineConfig(name="idea_engine", version="1.0.0")
    stages = [
        StageDefinition(
            name=StageName.DATA_COLLECTION.value,
            execute_fn=_noop_stage("data_collection"),
            timeout_sec=120,
            dependencies=[],
            critical=True,
        ),
        StageDefinition(
            name=StageName.FEATURE_COMPUTATION.value,
            execute_fn=_noop_stage("feature_computation"),
            timeout_sec=180,
            dependencies=["data_collection"],
            critical=True,
        ),
        StageDefinition(
            name=StageName.SIGNAL_GENERATION.value,
            execute_fn=_noop_stage("signal_generation"),
            timeout_sec=120,
            dependencies=["feature_computation"],
            critical=True,
        ),
        StageDefinition(
            name=StageName.IDEA_SYNTHESIS.value,
            execute_fn=_noop_stage("idea_synthesis"),
            timeout_sec=60,
            dependencies=["signal_generation"],
            critical=True,
        ),
        StageDefinition(
            name=StageName.DEBATE.value,
            execute_fn=_noop_stage("debate"),
            timeout_sec=90,
            dependencies=["idea_synthesis"],
            critical=False,
        ),
        StageDefinition(
            name=StageName.VALIDATION.value,
            execute_fn=_noop_stage("validation"),
            timeout_sec=120,
            dependencies=["debate"],
            critical=True,
        ),
        StageDefinition(
            name=StageName.RISK_CHECK.value,
            execute_fn=_noop_stage("risk_check"),
            timeout_sec=60,
            dependencies=["validation"],
            critical=True,
        ),
        StageDefinition(
            name=StageName.EXECUTION.value,
            execute_fn=_noop_stage("execution"),
            timeout_sec=30,
            dependencies=["risk_check"],
            critical=True,
        ),
        StageDefinition(
            name=StageName.MONITORING.value,
            execute_fn=_noop_stage("monitoring"),
            timeout_sec=60,
            dependencies=["execution"],
            critical=False,
        ),
        StageDefinition(
            name=StageName.FEEDBACK.value,
            execute_fn=_noop_stage("feedback"),
            timeout_sec=60,
            dependencies=["monitoring"],
            critical=False,
        ),
    ]
    for s in stages:
        config.add_stage(s)
    return config


def build_fast_pipeline() -> PipelineConfig:
    """Minimal pipeline for rapid iteration."""
    config = PipelineConfig(name="fast_pipeline", version="1.0.0-fast")
    for name in ["signal_generation", "risk_check", "execution"]:
        config.add_stage(StageDefinition(
            name=name,
            execute_fn=_noop_stage(name),
            timeout_sec=30,
            dependencies=[] if name == "signal_generation" else (
                ["signal_generation"] if name == "risk_check" else ["risk_check"]
            ),
        ))
    return config


def build_research_pipeline() -> PipelineConfig:
    """Research-focused pipeline with extended validation."""
    config = PipelineConfig(name="research_pipeline", version="1.0.0-research")
    stages = [
        ("data_collection", [], True),
        ("feature_computation", ["data_collection"], True),
        ("signal_generation", ["feature_computation"], True),
        ("validation", ["signal_generation"], True),
        ("debate", ["validation"], False),
        ("feedback", ["debate"], False),
    ]
    for name, deps, critical in stages:
        config.add_stage(StageDefinition(
            name=name,
            execute_fn=_noop_stage(name),
            timeout_sec=300,
            dependencies=deps,
            critical=critical,
        ))
    return config


# ===================================================================
# 15. Pipeline analytics
# ===================================================================

def analyze_pipeline_performance(
    monitor: PipelineMonitor,
) -> Dict[str, Any]:
    """Deep analysis of pipeline performance."""
    summary = monitor.summary()
    stage_data = {}
    for name in monitor._stage_metrics:
        m = monitor._stage_metrics[name]
        stage_data[name] = {
            "success_rate": m.success_rate,
            "avg_duration": m.avg_duration_sec,
            "total_runs": m.total_runs,
            "retry_rate": m.retry_count / max(m.total_runs, 1),
        }
    # Identify critical path
    durations = {name: d["avg_duration"] for name, d in stage_data.items()}
    critical_path = sorted(durations.items(), key=lambda x: -x[1])

    return {
        "summary": summary,
        "per_stage": stage_data,
        "critical_path": critical_path,
        "total_retries": sum(m.retry_count for m in monitor._stage_metrics.values()),
        "total_circuit_opens": sum(m.circuit_opens for m in monitor._stage_metrics.values()),
    }


def simulate_pipeline_load(
    config: PipelineConfig,
    n_runs: int = 100,
    failure_prob: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate pipeline execution under load with random failures."""
    rng = np.random.default_rng(seed)
    durations = []
    successes = 0
    stage_failures: Dict[str, int] = defaultdict(int)

    for run in range(n_runs):
        # Inject random failures
        modified_stages = []
        for stage in config.stages:
            def make_fn(s: StageDefinition, fail_p: float) -> Callable:
                original = s.execute_fn
                def fn(x: Any) -> Any:
                    if rng.random() < fail_p:
                        raise RuntimeError(f"Simulated failure in {s.name}")
                    return original(x)
                return fn
            new_stage = StageDefinition(
                name=stage.name,
                execute_fn=make_fn(stage, failure_prob),
                timeout_sec=stage.timeout_sec,
                retry_policy=stage.retry_policy,
                dependencies=stage.dependencies,
                enabled=stage.enabled,
                critical=stage.critical,
            )
            modified_stages.append(new_stage)

        test_config = PipelineConfig(
            name=config.name,
            version=config.version,
            stages=modified_stages,
            fail_fast=config.fail_fast,
            default_retry_policy=config.default_retry_policy,
        )
        executor = PipelineExecutor(test_config)
        start = time.time()
        results = executor.run({"run_number": run})
        duration = time.time() - start
        durations.append(duration)

        all_ok = all(
            r.succeeded or r.status == StageStatus.SKIPPED
            for r in results.values()
        )
        if all_ok:
            successes += 1
        for name, r in results.items():
            if not r.succeeded and r.status != StageStatus.SKIPPED:
                stage_failures[name] += 1

    return {
        "n_runs": n_runs,
        "success_rate": successes / n_runs,
        "avg_duration_sec": float(np.mean(durations)),
        "p95_duration_sec": float(np.percentile(durations, 95)),
        "stage_failure_counts": dict(stage_failures),
        "failure_prob": failure_prob,
    }


# ===================================================================
# 16. Utility: pipeline composition
# ===================================================================

def compose_pipelines(
    *configs: PipelineConfig,
) -> PipelineConfig:
    """Compose multiple pipelines into a single sequential pipeline."""
    combined = PipelineConfig(
        name="composed_" + "_".join(c.name for c in configs),
        version="1.0.0",
    )
    prev_last_stage = ""
    for i, cfg in enumerate(configs):
        prefix = f"p{i}_"
        for stage in cfg.stages:
            new_deps = [prefix + d for d in stage.dependencies]
            if not new_deps and prev_last_stage:
                new_deps = [prev_last_stage]
            new_name = prefix + stage.name
            combined.add_stage(StageDefinition(
                name=new_name,
                execute_fn=stage.execute_fn,
                timeout_sec=stage.timeout_sec,
                retry_policy=stage.retry_policy,
                dependencies=new_deps,
                enabled=stage.enabled,
                critical=stage.critical,
            ))
            prev_last_stage = new_name
    return combined


def branch_pipeline(
    config: PipelineConfig,
    branch_after: str,
    branch_stages: List[StageDefinition],
    branch_name: str = "branch",
) -> PipelineConfig:
    """Add a branch to the pipeline after a specific stage."""
    new_config = PipelineConfig(
        name=config.name,
        version=config.version,
        stages=list(config.stages),
    )
    for stage in branch_stages:
        new_stage = StageDefinition(
            name=f"{branch_name}_{stage.name}",
            execute_fn=stage.execute_fn,
            timeout_sec=stage.timeout_sec,
            dependencies=[branch_after] + [f"{branch_name}_{d}" for d in stage.dependencies],
            critical=stage.critical,
        )
        new_config.add_stage(new_stage)
    return new_config


# ===================================================================
# __all__
# ===================================================================

__all__ = [
    "StageStatus",
    "PipelineStatus",
    "StageName",
    "RetryPolicy",
    "DeadLetterQueue",
    "DeadLetterItem",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "StageResult",
    "StageDefinition",
    "DependencyGraph",
    "PipelineConfig",
    "PipelineMonitor",
    "StageMetrics",
    "PipelineExecutor",
    "PipelineReport",
    "MarketHours",
    "ScheduleEntry",
    "ScheduledPipeline",
    "PipelineVersion",
    "PipelineVersionManager",
    "build_default_pipeline",
    "build_fast_pipeline",
    "build_research_pipeline",
    "analyze_pipeline_performance",
    "simulate_pipeline_load",
    "compose_pipelines",
    "branch_pipeline",
]
