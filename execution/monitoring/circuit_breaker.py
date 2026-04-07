"""
Circuit breaker system for SRFM execution monitoring.
Protects downstream services (brokers, DB, coordination) from cascade failures.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"       # normal operation -- requests pass through
    OPEN = "OPEN"           # blocking -- requests rejected immediately
    HALF_OPEN = "HALF_OPEN" # probing -- limited requests allowed through


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""

    def __init__(self, name: str, opened_at: float, timeout_seconds: float) -> None:
        self.name = name
        self.opened_at = opened_at
        self.timeout_seconds = timeout_seconds
        remaining = max(0.0, (opened_at + timeout_seconds) - time.monotonic())
        super().__init__(
            f"Circuit '{name}' is OPEN -- retry in {remaining:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for a single circuit breaker."""

    failure_threshold: int = 5
    # number of failures in window before opening
    window_seconds: float = 60.0
    # rolling window for failure counting
    timeout_seconds: float = 120.0
    # how long to stay OPEN before probing
    probe_success_threshold: int = 2
    # consecutive successes in HALF_OPEN to close


class CircuitBreaker:
    """
    Three-state circuit breaker.

    CLOSED    -- all calls pass through; failures are counted in a rolling window.
    OPEN      -- all calls fail fast with CircuitOpenError; after timeout_seconds
                 the breaker transitions to HALF_OPEN.
    HALF_OPEN -- calls pass through as probes; probe_success_threshold consecutive
                 successes close the circuit; any failure re-opens it.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self._name = name
        self._config = config
        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()

        # rolling failure timestamps (monotonic)
        self._failure_times: deque = deque()
        # when we transitioned to OPEN (monotonic)
        self._opened_at: Optional[float] = None
        # consecutive successes while HALF_OPEN
        self._probe_successes: int = 0

        # lifetime stats
        self._total_calls: int = 0
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._total_rejected: int = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute fn through the circuit breaker."""
        with self._lock:
            self._maybe_transition_to_half_open()
            current_state = self._state

        if current_state == CircuitState.OPEN:
            with self._lock:
                self._total_rejected += 1
            raise CircuitOpenError(
                self._name,
                self._opened_at or time.monotonic(),
                self._config.timeout_seconds,
            )

        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except CircuitOpenError:
            raise
        except Exception as exc:
            self.record_failure(exc)
            raise

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._total_calls += 1
            self._total_successes += 1
            if self._state == CircuitState.HALF_OPEN:
                self._probe_successes += 1
                if self._probe_successes >= self._config.probe_success_threshold:
                    self._close()
                    logger.info(
                        "Circuit '%s' CLOSED after %d probe successes",
                        self._name,
                        self._config.probe_success_threshold,
                    )

    def record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            now = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # any failure in probe state re-opens the circuit
                self._open(now)
                logger.warning(
                    "Circuit '%s' re-OPENED during probe: %s",
                    self._name,
                    error,
                )
                return

            # CLOSED -- add failure timestamp and check threshold
            self._failure_times.append(now)
            self._evict_old_failures(now)

            if len(self._failure_times) >= self._config.failure_threshold:
                self._open(now)
                logger.error(
                    "Circuit '%s' OPENED after %d failures in %.0fs window",
                    self._name,
                    self._config.failure_threshold,
                    self._config.window_seconds,
                )

    def reset(self) -> None:
        """Manually force the circuit back to CLOSED state."""
        with self._lock:
            self._close()
            logger.info("Circuit '%s' manually RESET to CLOSED", self._name)

    def status(self) -> Dict[str, Any]:
        """Return a summary dict of current state and lifetime stats."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return {
                "name": self._name,
                "state": self._state.value,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "total_rejected": self._total_rejected,
                "recent_failures": len(self._failure_times),
                "probe_successes": self._probe_successes,
                "failure_threshold": self._config.failure_threshold,
                "window_seconds": self._config.window_seconds,
                "timeout_seconds": self._config.timeout_seconds,
            }

    # ------------------------------------------------------------------
    # Internal helpers -- must be called while holding self._lock
    # ------------------------------------------------------------------

    def _open(self, now: float) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = now
        self._probe_successes = 0

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._probe_successes = 0
        self._failure_times.clear()

    def _maybe_transition_to_half_open(self) -> None:
        """Check whether enough time has elapsed to move OPEN -> HALF_OPEN."""
        if self._state != CircuitState.OPEN:
            return
        if self._opened_at is None:
            return
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self._config.timeout_seconds:
            self._state = CircuitState.HALF_OPEN
            self._probe_successes = 0
            logger.info(
                "Circuit '%s' transitioned to HALF_OPEN after %.1fs",
                self._name,
                elapsed,
            )

    def _evict_old_failures(self, now: float) -> None:
        """Remove failure timestamps outside the rolling window."""
        cutoff = now - self._config.window_seconds
        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class CircuitBreakerRegistry:
    """
    Global registry of named circuit breakers.
    Provides a singleton access pattern and pre-configured defaults.
    """

    _instance: Optional["CircuitBreakerRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        self._init_defaults()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "CircuitBreakerRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Return existing breaker or create a new one with the given config."""
        with self._lock:
            if name not in self._breakers:
                cfg = config or CircuitBreakerConfig()
                self._breakers[name] = CircuitBreaker(name, cfg)
                logger.debug("CircuitBreaker '%s' created", name)
            return self._breakers[name]

    def status_all(self) -> Dict[str, str]:
        """Return mapping of name -> state string for every registered breaker."""
        with self._lock:
            result: Dict[str, str] = {}
            for name, cb in self._breakers.items():
                result[name] = cb.state.value
            return result

    def status_detailed(self) -> Dict[str, Dict[str, Any]]:
        """Return full status dict for every registered breaker."""
        with self._lock:
            return {name: cb.status() for name, cb in self._breakers.items()}

    def reset(self, name: str) -> None:
        """Manually close (reset) the named circuit."""
        with self._lock:
            if name in self._breakers:
                self._breakers[name].reset()
            else:
                logger.warning("reset() called for unknown circuit '%s'", name)

    def reset_all(self) -> None:
        """Manually close all registered circuits."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()
            logger.info("All circuits reset to CLOSED")

    def list_names(self) -> List[str]:
        with self._lock:
            return list(self._breakers.keys())

    def register(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register (or overwrite) a named circuit breaker with explicit config."""
        with self._lock:
            cb = CircuitBreaker(name, config)
            self._breakers[name] = cb
            return cb

    # ------------------------------------------------------------------
    # Pre-configured defaults
    # ------------------------------------------------------------------

    def _init_defaults(self) -> None:
        defaults: Dict[str, CircuitBreakerConfig] = {
            "alpaca_orders": CircuitBreakerConfig(
                failure_threshold=5,
                window_seconds=60.0,
                timeout_seconds=120.0,
                probe_success_threshold=2,
            ),
            "alpaca_data": CircuitBreakerConfig(
                failure_threshold=3,
                window_seconds=30.0,
                timeout_seconds=60.0,
                probe_success_threshold=2,
            ),
            "binance_orders": CircuitBreakerConfig(
                failure_threshold=5,
                window_seconds=60.0,
                timeout_seconds=120.0,
                probe_success_threshold=2,
            ),
            "database": CircuitBreakerConfig(
                failure_threshold=3,
                window_seconds=10.0,
                timeout_seconds=30.0,
                probe_success_threshold=2,
            ),
            "coordination": CircuitBreakerConfig(
                failure_threshold=3,
                window_seconds=60.0,
                timeout_seconds=180.0,
                probe_success_threshold=2,
            ),
        }
        for name, cfg in defaults.items():
            self._breakers[name] = CircuitBreaker(name, cfg)
            logger.debug("Default circuit breaker '%s' registered", name)


# ---------------------------------------------------------------------------
# FastAPI / Starlette middleware
# ---------------------------------------------------------------------------

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    class CircuitBreakerMiddleware(BaseHTTPMiddleware):
        """
        Starlette/FastAPI middleware that checks named circuit breakers before
        dispatching a request.  Returns HTTP 503 immediately when the relevant
        circuit is OPEN.

        Example usage:
            app.add_middleware(
                CircuitBreakerMiddleware,
                registry=CircuitBreakerRegistry.get_instance(),
                route_circuit_map={"/orders": "alpaca_orders"},
            )
        """

        def __init__(
            self,
            app: Any,
            registry: Optional[CircuitBreakerRegistry] = None,
            route_circuit_map: Optional[Dict[str, str]] = None,
        ) -> None:
            super().__init__(app)
            self._registry = registry or CircuitBreakerRegistry.get_instance()
            self._route_circuit_map: Dict[str, str] = route_circuit_map or {}

        async def dispatch(self, request: Request, call_next: Callable) -> Any:
            path = request.url.path
            circuit_name = self._resolve_circuit(path)

            if circuit_name is not None:
                cb = self._registry.get_or_create(circuit_name)
                if cb.state == CircuitState.OPEN:
                    logger.warning(
                        "Circuit '%s' is OPEN -- rejecting %s %s",
                        circuit_name,
                        request.method,
                        path,
                    )
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": "service_unavailable",
                            "circuit": circuit_name,
                            "message": (
                                f"Circuit '{circuit_name}' is OPEN"
                                " -- service temporarily unavailable"
                            ),
                        },
                    )

            return await call_next(request)

        def _resolve_circuit(self, path: str) -> Optional[str]:
            """Find the best matching circuit name for the request path."""
            if path in self._route_circuit_map:
                return self._route_circuit_map[path]
            for prefix, circuit in self._route_circuit_map.items():
                if path.startswith(prefix):
                    return circuit
            return None

except ImportError:
    class CircuitBreakerMiddleware:  # type: ignore[no-redef]
        """Stub -- install starlette/fastapi to enable CircuitBreakerMiddleware."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "CircuitBreakerMiddleware requires starlette. "
                "Install with: pip install fastapi"
            )


# ---------------------------------------------------------------------------
# Decorator helper
# ---------------------------------------------------------------------------

def circuit_protected(
    circuit_name: str,
    registry: Optional[CircuitBreakerRegistry] = None,
) -> Callable:
    """
    Function decorator that wraps a callable with a named circuit breaker.

    Example:
        @circuit_protected("alpaca_orders")
        def submit_order(order):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            reg = registry or CircuitBreakerRegistry.get_instance()
            cb = reg.get_or_create(circuit_name)
            return cb.call(fn, *args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Async-compatible circuit breaker
# ---------------------------------------------------------------------------

class AsyncCircuitBreaker:
    """
    Async-compatible circuit breaker.  Delegates state management to a
    synchronous CircuitBreaker so all state transitions are lock-protected.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self._inner = CircuitBreaker(name, config)

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def state(self) -> CircuitState:
        return self._inner.state

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a coroutine or sync callable through the circuit breaker."""
        import inspect

        if self._inner.state == CircuitState.OPEN:
            with self._inner._lock:
                self._inner._total_rejected += 1
            raise CircuitOpenError(
                self._inner.name,
                self._inner._opened_at or time.monotonic(),
                self._inner._config.timeout_seconds,
            )
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)
            self._inner.record_success()
            return result
        except CircuitOpenError:
            raise
        except Exception as exc:
            self._inner.record_failure(exc)
            raise

    def reset(self) -> None:
        self._inner.reset()

    def status(self) -> Dict[str, Any]:
        return self._inner.status()
