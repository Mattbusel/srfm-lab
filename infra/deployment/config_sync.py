# infra/deployment/config_sync.py -- configuration synchronization for SRFM services
from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
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
class SyncResult:
    success: bool
    services_updated: List[str] = field(default_factory=list)
    failures: Dict[str, str] = field(default_factory=dict)  # service_name -> error message

    def summary(self) -> str:
        ok = len(self.services_updated)
        fail = len(self.failures)
        return f"SyncResult: {ok} updated, {fail} failed, success={self.success}"


@dataclass
class ConfigChange:
    path: str           # dot-notation key path, e.g. "trading.max_position"
    old_value: Any
    new_value: Any
    change_type: str    # "add", "modify", "delete"

    def __str__(self) -> str:
        if self.change_type == "add":
            return f"ADD    {self.path} = {self.new_value!r}"
        if self.change_type == "delete":
            return f"DELETE {self.path} (was {self.old_value!r})"
        return f"MODIFY {self.path}: {self.old_value!r} -> {self.new_value!r}"


@dataclass
class VersionedConfig:
    version_hash: str
    timestamp: datetime
    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# ConfigDiff
# ---------------------------------------------------------------------------

class ConfigDiff:
    """Computes structural diffs between two config dicts."""

    def diff(self, old: Dict[str, Any], new: Dict[str, Any]) -> List[ConfigChange]:
        """Return a list of ConfigChange describing the delta from old to new."""
        changes: List[ConfigChange] = []
        self._recurse(old, new, prefix="", changes=changes)
        return changes

    def _recurse(
        self,
        old: Any,
        new: Any,
        prefix: str,
        changes: List[ConfigChange],
    ) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            all_keys = set(old) | set(new)
            for key in sorted(all_keys):
                child_path = f"{prefix}.{key}" if prefix else key
                if key not in old:
                    changes.append(
                        ConfigChange(
                            path=child_path,
                            old_value=None,
                            new_value=new[key],
                            change_type="add",
                        )
                    )
                elif key not in new:
                    changes.append(
                        ConfigChange(
                            path=child_path,
                            old_value=old[key],
                            new_value=None,
                            change_type="delete",
                        )
                    )
                else:
                    self._recurse(old[key], new[key], child_path, changes)
        else:
            if old != new:
                changes.append(
                    ConfigChange(
                        path=prefix,
                        old_value=old,
                        new_value=new,
                        change_type="modify",
                    )
                )

    def format_diff(self, changes: List[ConfigChange]) -> str:
        if not changes:
            return "(no changes)"
        return "\n".join(str(c) for c in changes)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL_KEYS = [
    "services",
    "trading",
    "risk",
]

_TRADING_REQUIRED = [
    "max_position_usd",
    "max_daily_loss_usd",
]

_RISK_REQUIRED = [
    "max_leverage",
    "halt_on_loss_pct",
]


def validate_srfm_config(config: Dict[str, Any]) -> List[str]:
    """Validate an SRFM config dict. Returns a list of error strings (empty = valid)."""
    errors: List[str] = []

    if not isinstance(config, dict):
        return ["Config must be a dict"]

    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            errors.append(f"Missing required top-level key: '{key}'")

    trading = config.get("trading", {})
    if isinstance(trading, dict):
        for key in _TRADING_REQUIRED:
            if key not in trading:
                errors.append(f"Missing trading.{key}")
            elif not isinstance(trading[key], (int, float)):
                errors.append(f"trading.{key} must be numeric")

    risk = config.get("risk", {})
    if isinstance(risk, dict):
        for key in _RISK_REQUIRED:
            if key not in risk:
                errors.append(f"Missing risk.{key}")

    max_lev = risk.get("max_leverage")
    if isinstance(max_lev, (int, float)) and max_lev <= 0:
        errors.append("risk.max_leverage must be > 0")

    halt_pct = risk.get("halt_on_loss_pct")
    if isinstance(halt_pct, (int, float)) and not (0 < halt_pct <= 1.0):
        errors.append("risk.halt_on_loss_pct must be in (0, 1]")

    return errors


# ---------------------------------------------------------------------------
# ConfigSync
# ---------------------------------------------------------------------------

# Default reload endpoints per service
_RELOAD_ENDPOINTS: Dict[str, str] = {
    "live-trader":        "http://localhost:8080/config/reload",
    "coordination":       "http://localhost:8781/config/reload",
    "risk-api":           "http://localhost:8783/config/reload",
    "market-data":        "http://localhost:8784/config/reload",
    "idea-engine":        "http://localhost:8785/config/reload",
    "dashboard-api":      "http://localhost:9091/config/reload",
    "metrics-collector":  "http://localhost:9090/config/reload",
}


class ConfigSync:
    """Synchronises configuration changes to all SRFM services.

    -- Keeps an in-memory history of versioned configs keyed by hash.
    -- Sends HTTP POST to each service's /config/reload endpoint.
    -- Supports dry-run (validate_only) mode.
    -- Polls a config directory for file changes.
    """

    HTTP_TIMEOUT_S: float = 5.0

    def __init__(
        self,
        reload_endpoints: Optional[Dict[str, str]] = None,
        http_timeout_s: float = 5.0,
    ) -> None:
        self._endpoints = reload_endpoints or dict(_RELOAD_ENDPOINTS)
        self._timeout = http_timeout_s
        self._history: List[VersionedConfig] = []
        self._current_hash: Optional[str] = None
        self._lock = threading.Lock()
        self._differ = ConfigDiff()

    # -- public API ----------------------------------------------------------

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate config and return list of error messages. Empty list = valid."""
        return validate_srfm_config(config)

    def sync_to_services(
        self,
        config: Dict[str, Any],
        validate_only: bool = False,
    ) -> SyncResult:
        """Push config to all registered services.

        -- If validate_only=True, only validates and returns without sending.
        """
        errors = self.validate_config(config)
        if errors:
            return SyncResult(
                success=False,
                failures={"validation": "; ".join(errors)},
            )

        version_hash = self._hash_config(config)
        versioned = VersionedConfig(
            version_hash=version_hash,
            timestamp=datetime.utcnow(),
            config=copy.deepcopy(config),
        )

        if validate_only:
            logger.info("Config validation passed (hash=%s, dry-run mode)", version_hash)
            return SyncResult(success=True)

        with self._lock:
            self._history.append(versioned)
            self._current_hash = version_hash

        logger.info("Syncing config hash=%s to %d services", version_hash, len(self._endpoints))

        updated: List[str] = []
        failures: Dict[str, str] = {}

        for service_name, url in self._endpoints.items():
            ok, err = self._post_reload(service_name, url, config, version_hash)
            if ok:
                updated.append(service_name)
            else:
                failures[service_name] = err or "unknown error"

        success = len(failures) == 0
        result = SyncResult(success=success, services_updated=updated, failures=failures)
        logger.info(result.summary())
        return result

    def hot_reload(self, service_name: str, new_config: Dict[str, Any]) -> bool:
        """Push config to a single named service."""
        url = self._endpoints.get(service_name)
        if url is None:
            logger.error("hot_reload: unknown service '%s'", service_name)
            return False
        version_hash = self._hash_config(new_config)
        ok, err = self._post_reload(service_name, url, new_config, version_hash)
        if not ok:
            logger.error("hot_reload failed for '%s': %s", service_name, err)
        return ok

    def revert_to(self, version_hash: str) -> bool:
        """Re-send a previously applied config identified by its hash."""
        with self._lock:
            target = next(
                (v for v in reversed(self._history) if v.version_hash == version_hash),
                None,
            )
        if target is None:
            logger.error("revert_to: version hash '%s' not found in history", version_hash)
            return False

        logger.info("Reverting to config version %s", version_hash)
        result = self.sync_to_services(target.config)
        return result.success

    def watch_for_changes(
        self,
        config_dir: str,
        callback: Callable[[str, Dict[str, Any]], None],
        poll_interval_s: float = 2.0,
        stop_event: Optional[threading.Event] = None,
    ) -> threading.Thread:
        """Poll config_dir for .json file changes and invoke callback(filename, config).

        -- Returns the watcher thread (daemon=True).
        -- Pass a threading.Event to stop_event to stop the watcher.
        """
        _stop = stop_event or threading.Event()

        def _watch_loop() -> None:
            last_mtime: Dict[str, float] = {}
            import os

            while not _stop.is_set():
                try:
                    if os.path.isdir(config_dir):
                        for fname in os.listdir(config_dir):
                            if not fname.endswith(".json"):
                                continue
                            full_path = os.path.join(config_dir, fname)
                            try:
                                mtime = os.path.getmtime(full_path)
                            except OSError:
                                continue
                            if last_mtime.get(full_path) != mtime:
                                last_mtime[full_path] = mtime
                                try:
                                    import json as _json
                                    with open(full_path, encoding="utf-8") as f:
                                        data = _json.load(f)
                                    logger.info("Config file changed: %s", fname)
                                    callback(fname, data)
                                except Exception as exc:
                                    logger.error("Error reading changed config %s: %s", fname, exc)
                except Exception:
                    logger.exception("Config watcher loop error")
                _stop.wait(timeout=poll_interval_s)

        t = threading.Thread(target=_watch_loop, name="ConfigWatcher", daemon=True)
        t.start()
        logger.info("Config watcher started on directory: %s", config_dir)
        return t

    def history(self) -> List[VersionedConfig]:
        with self._lock:
            return list(self._history)

    def current_hash(self) -> Optional[str]:
        with self._lock:
            return self._current_hash

    def diff_configs(
        self,
        old: Dict[str, Any],
        new: Dict[str, Any],
    ) -> List[ConfigChange]:
        return self._differ.diff(old, new)

    def format_diff(self, old: Dict[str, Any], new: Dict[str, Any]) -> str:
        changes = self.diff_configs(old, new)
        return self._differ.format_diff(changes)

    # -- internal ------------------------------------------------------------

    def _post_reload(
        self,
        service_name: str,
        url: str,
        config: Dict[str, Any],
        version_hash: str,
    ) -> tuple:
        """POST config payload to a service's reload endpoint."""
        payload = json.dumps({
            "version_hash": version_hash,
            "config": config,
            "timestamp": datetime.utcnow().isoformat(),
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Config-Hash": version_hash,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status = resp.status
            if 200 <= status < 300:
                logger.debug("Config reloaded on '%s' (HTTP %d)", service_name, status)
                return True, None
            return False, f"HTTP {status}"

        except urllib.error.HTTPError as exc:
            return False, f"HTTPError {exc.code}: {exc.reason}"
        except urllib.error.URLError as exc:
            return False, f"URLError: {exc.reason}"
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _hash_config(config: Dict[str, Any]) -> str:
        """Compute a stable SHA-256 hash of a config dict."""
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
