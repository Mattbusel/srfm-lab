"""
adapter_manager.py -- Manages multiple broker adapters with smart routing and failover.

AdapterManager routes orders to the appropriate adapter based on asset class
(crypto -> Binance, equity -> Alpaca, etc.) and handles failover when an adapter
goes unhealthy. AdapterHealthMonitor runs periodic checks and triggers failover
on consecutive failures.

Routing hierarchy:
  1. Explicit asset_class routing table
  2. Symbol-prefix heuristics (BTC-, ETH- -> crypto)
  3. Default adapter if no rule matches
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from .base_adapter import (
    AssetClass,
    BrokerAdapter,
    BrokerAdapterError,
    Fill,
    OrderRequest,
    OrderResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HEALTH_INTERVAL_S = 30.0
DEFAULT_FAILURE_THRESHOLD = 3  # consecutive failures to trigger failover
CRYPTO_SYMBOL_PREFIXES = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE", "MATIC", "LTC", "LINK", "DOT"}


# ---------------------------------------------------------------------------
# Health record
# ---------------------------------------------------------------------------


@dataclass
class AdapterHealth:
    """Health record for a single adapter.

    Fields
    ------
    name                 -- adapter name
    last_check_at        -- UTC timestamp of last health check
    is_healthy           -- True if last check passed
    consecutive_failures -- number of consecutive failed checks
    last_error           -- last error message (empty if healthy)
    """

    name: str
    last_check_at: Optional[datetime] = None
    is_healthy: bool = True
    consecutive_failures: int = 0
    last_error: str = ""

    def record_success(self) -> None:
        self.is_healthy = True
        self.consecutive_failures = 0
        self.last_error = ""
        self.last_check_at = datetime.now(timezone.utc)

    def record_failure(self, error: str) -> None:
        self.is_healthy = False
        self.consecutive_failures += 1
        self.last_error = error
        self.last_check_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Failover rule
# ---------------------------------------------------------------------------


@dataclass
class FailoverRule:
    """Defines a primary -> backup failover relationship.

    Fields
    ------
    primary_name  -- name of the primary adapter
    backup_name   -- name of the backup adapter to activate on failure
    active_backup -- True if currently routing to backup
    """

    primary_name: str
    backup_name: str
    active_backup: bool = False


# ---------------------------------------------------------------------------
# Adapter health monitor
# ---------------------------------------------------------------------------


class AdapterHealthMonitor:
    """Periodically checks all registered adapters and triggers failover on failure.

    Runs in a background asyncio task. Calls test_connection() on each adapter
    at the specified interval. After DEFAULT_FAILURE_THRESHOLD consecutive
    failures, triggers failover for that adapter.

    Parameters
    ----------
    manager           -- AdapterManager to notify on failover
    failure_threshold -- consecutive failures before failover
    """

    def __init__(
        self,
        manager: "AdapterManager",
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    ) -> None:
        self.manager = manager
        self.failure_threshold = failure_threshold
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False
        self._health: Dict[str, AdapterHealth] = {}

    def get_health(self, name: str) -> Optional[AdapterHealth]:
        """Return health record for the named adapter."""
        return self._health.get(name)

    def get_all_health(self) -> Dict[str, AdapterHealth]:
        """Return health records for all known adapters."""
        return dict(self._health)

    async def start_monitoring(self, interval_s: float = DEFAULT_HEALTH_INTERVAL_S) -> None:
        """Start the background monitoring loop.

        Parameters
        ----------
        interval_s -- seconds between health checks
        """
        if self._running:
            logger.warning("AdapterHealthMonitor already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop(interval_s))
        logger.info("AdapterHealthMonitor started (interval=%.1fs)", interval_s)

    async def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("AdapterHealthMonitor stopped")

    async def _monitor_loop(self, interval_s: float) -> None:
        """Main monitoring loop -- checks all adapters every interval_s seconds."""
        while self._running:
            await self._check_all_adapters()
            await asyncio.sleep(interval_s)

    async def _check_all_adapters(self) -> None:
        """Run health checks for all registered adapters concurrently."""
        adapters = self.manager.get_all_adapters()
        tasks = [self._check_adapter(name, adapter) for name, adapter in adapters.items()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_adapter(self, name: str, adapter: BrokerAdapter) -> None:
        """Check a single adapter and update its health record.

        Parameters
        ----------
        name    -- adapter name
        adapter -- adapter instance to check
        """
        if name not in self._health:
            self._health[name] = AdapterHealth(name=name)

        health = self._health[name]
        try:
            ok = await asyncio.wait_for(adapter.test_connection(), timeout=10.0)
            if ok:
                health.record_success()
                logger.debug("Health check PASS: %s", name)
            else:
                health.record_failure("test_connection() returned False")
                logger.warning("Health check FAIL: %s (False)", name)
        except Exception as exc:
            health.record_failure(str(exc))
            logger.warning("Health check FAIL: %s (%s)", name, exc)

        # Check failover threshold
        if health.consecutive_failures >= self.failure_threshold:
            self.manager._trigger_failover_if_needed(name)


# ---------------------------------------------------------------------------
# Adapter manager
# ---------------------------------------------------------------------------


class AdapterManager:
    """Manages multiple broker adapters and routes orders by asset class.

    Registration:
        manager.register("alpaca", alpaca_adapter)
        manager.register("binance", binance_adapter)

    Routing:
        manager.set_route(AssetClass.EQUITY, "alpaca")
        manager.set_route(AssetClass.CRYPTO, "binance")
        adapter = manager.route_order(order)

    Failover:
        manager.failover("alpaca", "paper")  -- paper is backup for alpaca

    Parameters
    ----------
    default_adapter_name -- name of adapter to use when no route matches
    """

    def __init__(self, default_adapter_name: Optional[str] = None) -> None:
        self._adapters: Dict[str, BrokerAdapter] = {}
        self._routes: Dict[AssetClass, str] = {}
        self._failover_rules: Dict[str, FailoverRule] = {}
        self._disabled: Set[str] = set()
        self._default_adapter_name = default_adapter_name
        self._monitor: Optional[AdapterHealthMonitor] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, adapter: BrokerAdapter) -> None:
        """Register an adapter under the given name.

        Parameters
        ----------
        name    -- unique identifier for this adapter
        adapter -- adapter instance
        """
        if name in self._adapters:
            logger.warning("Replacing existing adapter %s", name)
        self._adapters[name] = adapter
        logger.info("Registered adapter: %s (%s)", name, adapter.__class__.__name__)

    def unregister(self, name: str) -> None:
        """Remove a registered adapter.

        Parameters
        ----------
        name -- adapter name to remove
        """
        if name in self._adapters:
            del self._adapters[name]
            logger.info("Unregistered adapter: %s", name)

    def set_route(self, asset_class: AssetClass, adapter_name: str) -> None:
        """Set the default adapter for an asset class.

        Parameters
        ----------
        asset_class  -- AssetClass enum value
        adapter_name -- name of registered adapter to use
        """
        if adapter_name not in self._adapters:
            raise ValueError(f"Unknown adapter: {adapter_name!r} -- register it first")
        self._routes[asset_class] = adapter_name
        logger.info("Route set: %s -> %s", asset_class.value, adapter_name)

    def set_default(self, adapter_name: str) -> None:
        """Set the fallback adapter when no route matches.

        Parameters
        ----------
        adapter_name -- name of registered adapter
        """
        if adapter_name not in self._adapters:
            raise ValueError(f"Unknown adapter: {adapter_name!r}")
        self._default_adapter_name = adapter_name

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_adapter(self, name: str) -> BrokerAdapter:
        """Get an adapter by name.

        Raises KeyError if the adapter is not registered.

        Parameters
        ----------
        name -- adapter name
        """
        if name not in self._adapters:
            raise KeyError(f"No adapter registered with name {name!r}")
        return self._adapters[name]

    def get_active(self, asset_class: Optional[AssetClass] = None) -> BrokerAdapter:
        """Return the active adapter for the given asset class.

        Respects failover state -- if primary is disabled, returns backup.

        Parameters
        ----------
        asset_class -- optional asset class filter; uses default if None

        Returns
        -------
        Active BrokerAdapter.

        Raises
        ------
        RuntimeError -- if no adapter is configured for the asset class
        """
        if asset_class is not None and asset_class in self._routes:
            name = self._routes[asset_class]
        elif self._default_adapter_name:
            name = self._default_adapter_name
        else:
            raise RuntimeError(
                f"No adapter configured for asset_class={asset_class} and no default set"
            )

        # Check failover: if primary is in a failed state, use backup
        name = self._resolve_failover(name)
        if name in self._disabled:
            raise RuntimeError(f"Adapter {name!r} is disabled and no backup is available")

        return self._adapters[name]

    def _resolve_failover(self, name: str) -> str:
        """Walk the failover chain to find an active adapter name.

        Parameters
        ----------
        name -- starting adapter name

        Returns
        -------
        Name of the active (non-disabled) adapter.
        """
        visited: Set[str] = set()
        current = name
        while current in self._failover_rules:
            rule = self._failover_rules[current]
            if not rule.active_backup:
                break
            if rule.backup_name in visited:
                logger.error("Failover loop detected involving %s", rule.backup_name)
                break
            visited.add(current)
            current = rule.backup_name
        return current

    def get_all_adapters(self) -> Dict[str, BrokerAdapter]:
        """Return all registered adapters."""
        return dict(self._adapters)

    # ------------------------------------------------------------------
    # Order routing
    # ------------------------------------------------------------------

    def route_order(self, order: OrderRequest) -> BrokerAdapter:
        """Determine the correct adapter for an order.

        Routing logic (in priority order):
        1. Explicit asset_class route from self._routes
        2. Symbol-prefix heuristic (e.g. BTC- -> crypto)
        3. Default adapter

        Parameters
        ----------
        order -- the order to route

        Returns
        -------
        BrokerAdapter to send this order to.
        """
        # 1. Use order's explicit asset class if we have a route
        if order.asset_class in self._routes:
            name = self._routes[order.asset_class]
            name = self._resolve_failover(name)
            return self._adapters[name]

        # 2. Symbol-prefix heuristic
        inferred = self._infer_asset_class(order.symbol)
        if inferred is not None and inferred in self._routes:
            name = self._routes[inferred]
            name = self._resolve_failover(name)
            return self._adapters[name]

        # 3. Default
        if self._default_adapter_name:
            name = self._resolve_failover(self._default_adapter_name)
            return self._adapters[name]

        raise RuntimeError(
            f"No adapter found for order symbol={order.symbol} "
            f"asset_class={order.asset_class.value}"
        )

    def _infer_asset_class(self, symbol: str) -> Optional[AssetClass]:
        """Infer asset class from symbol format.

        Rules:
        - Symbols containing '-' where the base is a known crypto prefix -> CRYPTO
        - Otherwise -> EQUITY (conservative fallback)

        Parameters
        ----------
        symbol -- SRFM symbol string

        Returns
        -------
        AssetClass guess, or None if ambiguous.
        """
        if "-" in symbol:
            base = symbol.split("-")[0].upper()
            if base in CRYPTO_SYMBOL_PREFIXES:
                return AssetClass.CRYPTO
        return None

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def health_check_all(self) -> Dict[str, bool]:
        """Run test_connection() on all adapters concurrently.

        Returns
        -------
        Dict mapping adapter name to health bool.
        """
        results: Dict[str, bool] = {}

        async def check(name: str, adapter: BrokerAdapter) -> None:
            try:
                ok = await asyncio.wait_for(adapter.test_connection(), timeout=10.0)
                results[name] = ok
            except Exception as exc:
                logger.warning("Health check failed for %s: %s", name, exc)
                results[name] = False

        tasks = [check(n, a) for n, a in self._adapters.items()]
        if tasks:
            await asyncio.gather(*tasks)

        return results

    # ------------------------------------------------------------------
    # Failover management
    # ------------------------------------------------------------------

    def failover(self, primary: str, backup: str) -> None:
        """Register a failover rule: if primary fails, use backup.

        Parameters
        ----------
        primary -- name of the primary adapter
        backup  -- name of the backup adapter
        """
        if primary not in self._adapters:
            raise ValueError(f"Primary adapter {primary!r} not registered")
        if backup not in self._adapters:
            raise ValueError(f"Backup adapter {backup!r} not registered")

        self._failover_rules[primary] = FailoverRule(
            primary_name=primary,
            backup_name=backup,
        )
        logger.info("Failover rule registered: %s -> %s", primary, backup)

    def _trigger_failover_if_needed(self, name: str) -> None:
        """Activate failover for the named adapter if a rule exists.

        Called by AdapterHealthMonitor when consecutive_failures >= threshold.

        Parameters
        ----------
        name -- name of the failing adapter
        """
        if name not in self._failover_rules:
            return
        rule = self._failover_rules[name]
        if not rule.active_backup:
            rule.active_backup = True
            logger.warning(
                "FAILOVER TRIGGERED: %s -> %s",
                rule.primary_name,
                rule.backup_name,
            )

    def recover(self, name: str) -> None:
        """Manually deactivate failover and return to primary.

        Use this when the primary adapter has been restored.

        Parameters
        ----------
        name -- name of the primary adapter to restore
        """
        if name in self._failover_rules:
            self._failover_rules[name].active_backup = False
            logger.info("Adapter %s recovered -- failover deactivated", name)

    def disable_adapter(self, name: str) -> None:
        """Mark an adapter as disabled (requests will fail).

        Parameters
        ----------
        name -- adapter name to disable
        """
        self._disabled.add(name)
        logger.warning("Adapter %s disabled", name)

    def enable_adapter(self, name: str) -> None:
        """Re-enable a previously disabled adapter.

        Parameters
        ----------
        name -- adapter name to enable
        """
        self._disabled.discard(name)
        logger.info("Adapter %s enabled", name)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    async def start_health_monitoring(
        self,
        interval_s: float = DEFAULT_HEALTH_INTERVAL_S,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    ) -> None:
        """Start the background health monitor.

        Creates an AdapterHealthMonitor and starts its monitoring loop.

        Parameters
        ----------
        interval_s        -- seconds between health checks
        failure_threshold -- consecutive failures to trigger failover
        """
        self._monitor = AdapterHealthMonitor(self, failure_threshold=failure_threshold)
        await self._monitor.start_monitoring(interval_s)

    async def stop_health_monitoring(self) -> None:
        """Stop the background health monitor."""
        if self._monitor:
            await self._monitor.stop_monitoring()
            self._monitor = None

    def get_health_monitor(self) -> Optional[AdapterHealthMonitor]:
        """Return the active health monitor, or None if not started."""
        return self._monitor

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_adapters(self) -> List[str]:
        """Return sorted list of registered adapter names."""
        return sorted(self._adapters.keys())

    def __repr__(self) -> str:
        routes_str = ", ".join(f"{k.value}={v}" for k, v in self._routes.items())
        return (
            f"AdapterManager(adapters={self.list_adapters()}, "
            f"routes=[{routes_str}], "
            f"default={self._default_adapter_name!r})"
        )
