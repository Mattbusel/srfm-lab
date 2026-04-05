"""
execution/ — SRFM Trading Lab Execution Layer
==============================================
Separates order management, smart routing, transaction cost analysis,
monitoring, and audit concerns from the strategy layer.

Submodules
----------
oms/            Order Management System (order lifecycle, position tracking, risk)
routing/        Smart order routing, TWAP execution, Alpaca broker adapter
tca/            Transaction Cost Analysis and execution quality reporting
monitoring/     Live monitoring thread, circuit breakers, reconciliation
audit/          Immutable audit log and performance ledger

Typical wiring::

    from execution.oms.order_manager import OrderManager
    from execution.oms.risk_guard import RiskGuard
    from execution.routing.alpaca_adapter import AlpacaAdapter
    from execution.routing.smart_router import SmartRouter
    from execution.monitoring.circuit_breaker import CircuitBreaker
    from execution.audit.audit_log import AuditLog

    audit  = AuditLog()
    risk   = RiskGuard()
    broker = AlpacaAdapter(paper=True)
    router = SmartRouter(broker=broker)
    mgr    = OrderManager(router=router, risk=risk, audit=audit)
"""

from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "oms",
    "routing",
    "tca",
    "monitoring",
    "audit",
]

EXECUTION_LAYER_VERSION = "1.0.0"
