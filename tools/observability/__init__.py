"""
tools/observability/__init__.py
================================
Observability suite for the LARSA live trading system.

Exports:
    PrometheusMetrics  - Prometheus /metrics server on :9090
    TracingMiddleware  - OpenTelemetry distributed tracing
    HealthServer       - /health, /ready, /live on :8799
    ProfilerDaemon     - Continuous cProfile + memory profiler
    LogStreamer        - Structured JSON log streaming + WebSocket
    DashboardAggregator - Real-time dashboard data + WebSocket push
"""

from .metrics import PrometheusMetrics
from .tracing import TracingMiddleware, trace_span, trace_bar_handler
from .health import HealthServer
from .profiler import ProfilerDaemon
from .log_streamer import LogStreamer
from .dashboard_data import DashboardAggregator

__all__ = [
    "PrometheusMetrics",
    "TracingMiddleware",
    "trace_span",
    "trace_bar_handler",
    "HealthServer",
    "ProfilerDaemon",
    "LogStreamer",
    "DashboardAggregator",
]

__version__ = "1.0.0"
