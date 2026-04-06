"""
tools/observability/tracing.py
================================
OpenTelemetry distributed tracing for the LARSA live trader.

Provides:
  - TracingMiddleware   — global tracer initialisation + context helpers
  - trace_span          — decorator that wraps a sync or async function in a span
  - trace_bar_handler   — convenience decorator for bar-processing functions

Environment variables
---------------------
OTEL_EXPORTER_OTLP_ENDPOINT
    gRPC endpoint for the OTLP exporter, e.g. ``http://localhost:4317``.
    When unset the tracer writes JSON to stdout (development default).

OTEL_SERVICE_NAME
    Service name reported to the collector.  Default: ``"larsa-live-trader"``.

OTEL_TRACES_SAMPLER_ARG
    Float in [0, 1] for tail-based sampling ratio.  Default: ``1.0``.

Usage::

    from tools.observability.tracing import TracingMiddleware, trace_span

    tracing = TracingMiddleware()
    tracing.start()

    @trace_span("compute_targets")
    def compute_targets(symbol, bars):
        ...

    # or in the bar handler:
    with tracing.span("bar_processing", symbol="BTC", timeframe="1h") as span:
        span.set_attribute("frac", 0.08)
        ...

    tracing.stop()

Dependencies:
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, TypeVar

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OpenTelemetry imports — degrade gracefully if missing.
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        SpanKind,
        StatusCode,
        set_span_in_context,
        use_span,
    )
    from opentelemetry.context import attach, detach, get_current
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    log.warning("opentelemetry packages not installed — tracing is a no-op")

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _OTLP_AVAILABLE = True
except ImportError:
    _OTLP_AVAILABLE = False

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Context variables for propagating span / symbol info to log enrichment
# ---------------------------------------------------------------------------

_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_trace_id", default=None
)
_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_span_id", default=None
)
_current_symbol: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_symbol", default=None
)


def get_trace_id() -> Optional[str]:
    """Return the active trace-id string, or None."""
    return _current_trace_id.get()


def get_span_id() -> Optional[str]:
    """Return the active span-id string, or None."""
    return _current_span_id.get()


def get_current_symbol() -> Optional[str]:
    """Return the symbol currently being processed, or None."""
    return _current_symbol.get()


# ---------------------------------------------------------------------------
# No-op span for when OTEL is unavailable
# ---------------------------------------------------------------------------

class _NoopSpan:
    """Drop-in replacement for an OpenTelemetry Span."""

    def set_attribute(self, _key: str, _value: Any) -> "_NoopSpan":
        return self

    def set_status(self, _status: Any, _description: str = "") -> "_NoopSpan":
        return self

    def record_exception(self, _exc: Exception) -> "_NoopSpan":
        return self

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


# ---------------------------------------------------------------------------
# TracingMiddleware
# ---------------------------------------------------------------------------

class TracingMiddleware:
    """
    Initialises the OpenTelemetry tracer and exposes helper methods for
    creating spans in the live trader.

    One instance should be created at process startup and shared (or accessed
    via ``get_tracing()``).
    """

    def __init__(
        self,
        service_name: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        sample_ratio: Optional[float] = None,
    ) -> None:
        self._service_name = (
            service_name
            or os.getenv("OTEL_SERVICE_NAME", "larsa-live-trader")
        )
        self._otlp_endpoint = (
            otlp_endpoint
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        )
        self._sample_ratio = float(
            sample_ratio
            or os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0")
        )
        self._tracer: Any = None
        self._provider: Any = None
        self._started = False

    # ----------------------------------------------------------------- start
    def start(self) -> None:
        """Initialise the tracer provider.  Call once at process start."""
        if self._started:
            return

        if not _OTEL_AVAILABLE:
            log.warning("TracingMiddleware.start(): OTEL not available — "
                        "all spans are no-ops.")
            self._started = True
            return

        resource = Resource(attributes={SERVICE_NAME: self._service_name})
        provider = TracerProvider(resource=resource)

        if self._otlp_endpoint and _OTLP_AVAILABLE:
            exporter = OTLPSpanExporter(endpoint=self._otlp_endpoint)
            processor = BatchSpanProcessor(exporter)
            log.info("TracingMiddleware: OTLP exporter → %s", self._otlp_endpoint)
        else:
            exporter = ConsoleSpanExporter()
            processor = SimpleSpanProcessor(exporter)
            log.info("TracingMiddleware: stdout JSON exporter (dev mode)")

        provider.add_span_processor(processor)
        otel_trace.set_tracer_provider(provider)

        self._provider = provider
        self._tracer = otel_trace.get_tracer(
            self._service_name, schema_url="https://opentelemetry.io/schemas/1.20.0"
        )
        self._started = True
        log.info("TracingMiddleware started (service=%s)", self._service_name)

    # ----------------------------------------------------------------- stop
    def stop(self) -> None:
        """Flush and shut down the span processor."""
        if not self._started:
            return
        if _OTEL_AVAILABLE and self._provider is not None:
            try:
                self._provider.shutdown()
            except Exception as exc:
                log.debug("TracingMiddleware.stop() error: %s", exc)
        self._started = False
        log.info("TracingMiddleware stopped")

    # ----------------------------------------------------------------- span context manager
    @contextmanager
    def span(
        self,
        name: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        direction: Optional[int] = None,
        frac: Optional[float] = None,
        notional: Optional[float] = None,
        **extra_attrs: Any,
    ) -> Generator[Any, None, None]:
        """
        Context manager that creates (and ends) an OTel span.

        Common span names used in the live trader:
            - ``bar_processing``
            - ``order_submission``
            - ``fill_handling``
            - ``bootstrap``
            - ``signal_computation``

        Parameters
        ----------
        name:
            Span name.
        symbol, timeframe, direction, frac, notional:
            Standard LARSA attributes applied as span attributes.
        **extra_attrs:
            Any additional attributes to set on the span.

        Yields
        ------
        The active span object (or ``_NoopSpan`` when OTEL is unavailable).
        """
        if not _OTEL_AVAILABLE or self._tracer is None:
            noop = _NoopSpan()
            # Still update context vars for log enrichment
            sym_token = _current_symbol.set(symbol) if symbol else None
            try:
                yield noop
            finally:
                if sym_token is not None:
                    _current_symbol.reset(sym_token)
            return

        with self._tracer.start_as_current_span(name) as otel_span:
            # Set standard LARSA attributes
            if symbol:
                otel_span.set_attribute("larsa.symbol", symbol.upper())
            if timeframe:
                otel_span.set_attribute("larsa.timeframe", timeframe)
            if direction is not None:
                otel_span.set_attribute("larsa.direction", direction)
            if frac is not None:
                otel_span.set_attribute("larsa.frac", round(frac, 6))
            if notional is not None:
                otel_span.set_attribute("larsa.notional_usd", round(notional, 2))
            for k, v in extra_attrs.items():
                otel_span.set_attribute(k, v)

            # Propagate IDs to context vars for structured logging
            ctx = otel_span.get_span_context()
            trace_tok = _current_trace_id.set(
                format(ctx.trace_id, "032x") if ctx.is_valid else None
            )
            span_tok = _current_span_id.set(
                format(ctx.span_id, "016x") if ctx.is_valid else None
            )
            sym_token = _current_symbol.set(symbol) if symbol else None

            try:
                yield otel_span
            except Exception as exc:
                otel_span.record_exception(exc)
                otel_span.set_status(StatusCode.ERROR, str(exc))
                raise
            finally:
                _current_trace_id.reset(trace_tok)
                _current_span_id.reset(span_tok)
                if sym_token is not None:
                    _current_symbol.reset(sym_token)

    # ----------------------------------------------------------------- named span helpers
    def bar_processing(self, symbol: str, timeframe: str) -> Any:
        """Return a context manager for a ``bar_processing`` span."""
        return self.span("bar_processing", symbol=symbol, timeframe=timeframe)

    def order_submission(self, symbol: str, side: str, notional: float) -> Any:
        """Return a context manager for an ``order_submission`` span."""
        return self.span(
            "order_submission", symbol=symbol,
            side=side, notional=notional
        )

    def fill_handling(self, symbol: str, side: str, fill_price: float) -> Any:
        """Return a context manager for a ``fill_handling`` span."""
        return self.span(
            "fill_handling", symbol=symbol,
            side=side, fill_price=fill_price
        )

    def bootstrap(self) -> Any:
        """Return a context manager for the ``bootstrap`` span."""
        return self.span("bootstrap")

    def signal_computation(self, symbol: str, timeframe: str) -> Any:
        """Return a context manager for a ``signal_computation`` span."""
        return self.span("signal_computation", symbol=symbol, timeframe=timeframe)

    # ----------------------------------------------------------------- context manager
    def __enter__(self) -> "TracingMiddleware":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"<TracingMiddleware service={self._service_name!r} "
            f"started={self._started}>"
        )


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def trace_span(
    name: Optional[str] = None,
    symbol_arg: Optional[str] = None,
    timeframe_arg: Optional[str] = None,
    *,
    middleware: Optional[TracingMiddleware] = None,
) -> Callable[[F], F]:
    """
    Decorator that wraps a sync or async function in an OTel span.

    Parameters
    ----------
    name:
        Span name.  Defaults to the decorated function's qualified name.
    symbol_arg:
        Name of the parameter in the decorated function that holds the symbol
        string.  If provided, the symbol will be extracted and set as a span
        attribute automatically.
    timeframe_arg:
        Name of the parameter that holds the timeframe string.
    middleware:
        Optional ``TracingMiddleware`` instance.  When ``None``, the module-
        level singleton returned by ``get_tracing()`` is used.

    Example::

        @trace_span("compute_targets", symbol_arg="symbol")
        def compute_targets(symbol: str, bars: np.ndarray) -> float:
            ...

        @trace_span("bar_handler")
        async def on_bar(bar: dict) -> None:
            ...
    """
    def decorator(fn: F) -> F:
        span_name = name or fn.__qualname__
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        def _extract(args, kwargs, arg_name: Optional[str]) -> Optional[str]:
            if arg_name is None:
                return None
            if arg_name in kwargs:
                return str(kwargs[arg_name])
            try:
                idx = param_names.index(arg_name)
                return str(args[idx]) if idx < len(args) else None
            except ValueError:
                return None

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            tm = middleware or get_tracing()
            sym = _extract(args, kwargs, symbol_arg)
            tf = _extract(args, kwargs, timeframe_arg)
            with tm.span(span_name, symbol=sym, timeframe=tf):
                return fn(*args, **kwargs)

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            tm = middleware or get_tracing()
            sym = _extract(args, kwargs, symbol_arg)
            tf = _extract(args, kwargs, timeframe_arg)
            with tm.span(span_name, symbol=sym, timeframe=tf):
                return await fn(*args, **kwargs)

        if inspect.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def trace_bar_handler(fn: F) -> F:
    """
    Convenience decorator for bar-processing coroutines.

    Assumes the function has a first positional argument that is a dict
    (the bar) containing at least ``"S"`` (symbol) and ``"tf"`` (timeframe)
    keys — matching the live_trader_alpaca bar format.

    Example::

        @trace_bar_handler
        async def _handle_bar(self, bar: dict) -> None:
            ...
    """
    span_name = f"bar_processing.{fn.__qualname__}"

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        bar: Optional[dict] = None
        for a in args:
            if isinstance(a, dict):
                bar = a
                break

        sym = bar.get("S") if bar else None
        tf = str(bar.get("tf", "")) if bar else None

        tm = get_tracing()
        with tm.span(span_name, symbol=sym, timeframe=tf):
            return await fn(*args, **kwargs)

    return wrapper  # type: ignore


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_tracing: Optional[TracingMiddleware] = None
_tracing_lock = threading.Lock()


def get_tracing() -> TracingMiddleware:
    """
    Return (and lazily create) the module-level singleton
    ``TracingMiddleware`` instance.
    """
    global _default_tracing
    if _default_tracing is None:
        with _tracing_lock:
            if _default_tracing is None:
                _default_tracing = TracingMiddleware()
    return _default_tracing
