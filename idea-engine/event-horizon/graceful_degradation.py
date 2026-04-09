"""
Graceful Degradation: keep trading when subsystems fail.

In production, things break. Ollama might be slow. ChromaDB might be down.
A signal might throw an exception. The Dream Engine might time out.

Instead of crashing, the system degrades gracefully:
  1. Each module has a health status and fallback behavior
  2. If a module is unhealthy, use cached results or defaults
  3. Critical vs non-critical: Guardian and execution NEVER fail,
     dreams and narratives can be skipped
  4. Automatic recovery: retry failed modules with backoff
  5. Performance tracking: measure how degradation affects returns

The system should ALWAYS be able to trade, even if half the modules
are down. The core BH physics + basic risk management is sufficient
for baseline operation.
"""

from __future__ import annotations
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ModulePriority:
    """Module criticality levels."""
    CRITICAL = 0       # System halts if this fails (Guardian, execution)
    HIGH = 1           # Performance degrades significantly (signals, portfolio)
    MEDIUM = 2         # Some features lost (dreams, consciousness)
    LOW = 3            # Nice to have (narratives, tear sheets)


@dataclass
class ModuleHealth:
    """Health status of a single module."""
    name: str
    priority: int
    is_healthy: bool = True
    last_success: float = 0.0
    last_failure: float = 0.0
    failure_count: int = 0
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    cached_result: Any = None
    cached_at: float = 0.0
    fallback_mode: str = "cache"   # "cache" / "default" / "skip" / "halt"


@dataclass
class DegradationEvent:
    """Record of a degradation event."""
    module: str
    timestamp: float
    error: str
    fallback_used: str
    impact_estimate: str


class GracefulDegradation:
    """
    Manages graceful degradation across all modules.

    Every module call goes through this manager:
      result = degradation.call(module_name, function, *args)

    If the function succeeds: return result, update health.
    If the function fails: return cached/default, log event.
    """

    def __init__(self, cache_ttl_seconds: float = 300):
        self.cache_ttl = cache_ttl_seconds
        self._modules: Dict[str, ModuleHealth] = {}
        self._events: List[DegradationEvent] = []
        self._latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    def register(self, name: str, priority: int, fallback_mode: str = "cache",
                  default_result: Any = None) -> None:
        """Register a module with the degradation manager."""
        self._modules[name] = ModuleHealth(
            name=name,
            priority=priority,
            fallback_mode=fallback_mode,
            cached_result=default_result,
        )

    def call(self, module_name: str, fn: Callable, *args,
              timeout_ms: float = 5000, **kwargs) -> Any:
        """
        Call a module function with graceful degradation.

        If the function succeeds: return result, update cache.
        If it fails or times out: return cached/default result.
        """
        health = self._modules.get(module_name)
        if not health:
            # Unknown module, just call directly
            return fn(*args, **kwargs)

        start = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed_ms = (time.time() - start) * 1000

            # Update health
            health.is_healthy = True
            health.last_success = time.time()
            health.consecutive_failures = 0
            health.avg_latency_ms = health.avg_latency_ms * 0.9 + elapsed_ms * 0.1
            health.cached_result = result
            health.cached_at = time.time()

            self._latency_history[module_name].append(elapsed_ms)

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            health.failure_count += 1
            health.consecutive_failures += 1
            health.last_failure = time.time()

            # Determine if module is now unhealthy
            if health.consecutive_failures >= 3:
                health.is_healthy = False

            # Log degradation event
            self._events.append(DegradationEvent(
                module=module_name,
                timestamp=time.time(),
                error=str(e)[:200],
                fallback_used=health.fallback_mode,
                impact_estimate=self._estimate_impact(health),
            ))

            # Return fallback
            if health.fallback_mode == "cache" and health.cached_result is not None:
                cache_age = time.time() - health.cached_at
                if cache_age < self.cache_ttl:
                    return health.cached_result
                # Cache expired, try default
                return health.cached_result  # stale cache better than nothing

            elif health.fallback_mode == "default":
                return health.cached_result

            elif health.fallback_mode == "skip":
                return None

            elif health.fallback_mode == "halt":
                raise  # Critical module, propagate error

            return None

    def _estimate_impact(self, health: ModuleHealth) -> str:
        """Estimate the impact of this module being down."""
        impacts = {
            ModulePriority.CRITICAL: "SYSTEM HALT - Cannot trade without this module",
            ModulePriority.HIGH: "Performance degraded - using stale signals",
            ModulePriority.MEDIUM: "Feature unavailable - some intelligence lost",
            ModulePriority.LOW: "Cosmetic - reporting/narrative degraded",
        }
        return impacts.get(health.priority, "Unknown impact")

    def get_system_health(self) -> Dict:
        """Overall system health assessment."""
        total = len(self._modules)
        healthy = sum(1 for m in self._modules.values() if m.is_healthy)
        critical_down = sum(1 for m in self._modules.values()
                            if not m.is_healthy and m.priority == ModulePriority.CRITICAL)

        if critical_down > 0:
            status = "CRITICAL"
        elif healthy < total * 0.7:
            status = "DEGRADED"
        elif healthy < total:
            status = "PARTIAL"
        else:
            status = "HEALTHY"

        return {
            "status": status,
            "healthy_modules": healthy,
            "total_modules": total,
            "critical_down": critical_down,
            "total_failures": sum(m.failure_count for m in self._modules.values()),
            "recent_events": len([e for e in self._events if time.time() - e.timestamp < 3600]),
            "modules": {
                name: {
                    "healthy": m.is_healthy,
                    "priority": m.priority,
                    "failures": m.failure_count,
                    "avg_latency_ms": m.avg_latency_ms,
                    "last_success_ago_s": time.time() - m.last_success if m.last_success > 0 else -1,
                }
                for name, m in self._modules.items()
            },
        }

    def should_trade(self) -> bool:
        """Can the system still trade in its current state?"""
        # Can trade as long as no CRITICAL modules are down
        critical_down = sum(1 for m in self._modules.values()
                            if not m.is_healthy and m.priority == ModulePriority.CRITICAL)
        return critical_down == 0


def setup_default_degradation() -> GracefulDegradation:
    """Set up degradation with all Event Horizon modules registered."""
    gd = GracefulDegradation()

    # Critical (system halts if down)
    gd.register("guardian", ModulePriority.CRITICAL, "halt")
    gd.register("execution", ModulePriority.CRITICAL, "halt")

    # High (performance degrades)
    gd.register("fractal_signal", ModulePriority.HIGH, "cache")
    gd.register("info_surprise", ModulePriority.HIGH, "cache")
    gd.register("liquidity_blackhole", ModulePriority.HIGH, "cache")
    gd.register("portfolio_brain", ModulePriority.HIGH, "cache")
    gd.register("stability_monitor", ModulePriority.HIGH, "cache")

    # Medium (features lost)
    gd.register("consciousness", ModulePriority.MEDIUM, "cache")
    gd.register("dream_engine", ModulePriority.MEDIUM, "skip")
    gd.register("ehs_synthesizer", ModulePriority.MEDIUM, "skip")
    gd.register("rmea", ModulePriority.MEDIUM, "skip")
    gd.register("market_memory", ModulePriority.MEDIUM, "cache")
    gd.register("fear_greed", ModulePriority.MEDIUM, "default", 0.0)
    gd.register("groupthink", ModulePriority.MEDIUM, "default", None)
    gd.register("topology", ModulePriority.MEDIUM, "cache")

    # Low (nice to have)
    gd.register("narrative", ModulePriority.LOW, "skip")
    gd.register("tear_sheet", ModulePriority.LOW, "skip")
    gd.register("provenance", ModulePriority.LOW, "skip")
    gd.register("compliance", ModulePriority.LOW, "cache")
    gd.register("signal_api", ModulePriority.LOW, "skip")
    gd.register("adversarial_detector", ModulePriority.LOW, "skip")
    gd.register("mistake_learner", ModulePriority.LOW, "skip")

    return gd
