"""
config/param_manager.py
=======================
Parameter management system for the LARSA live trading system.

Interfaces with the Elixir coordination layer at :8781 for parameter
proposal, retrieval, history, and rollback. Validates all parameters
against the JSON schema before sending or applying them.

Classes:
  ParamSchema   -- loads param_schema.json, validates values
  LiveParams    -- dataclass holding current parameter values
  ParamManager  -- full coordination-layer client with polling
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "param_schema.json"
_COORD_BASE  = "http://localhost:8781"

# ---------------------------------------------------------------------------
# ParamSchema
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    """Raised when a parameter value fails schema validation."""


class ParamSchema:
    """
    Loads param_schema.json and validates parameter dicts against it.

    Responsibilities:
      - Type checking (float, int, bool, list_int)
      - Range checking (min/max for numerics, min/max length for lists)
      - Allowed-values checking for list elements
      - Cross-parameter constraint evaluation
      - Default extraction
    """

    def __init__(self, schema_path: Path = _SCHEMA_PATH) -> None:
        with open(schema_path, "r") as fh:
            raw = json.load(fh)
        self._schema: dict[str, dict] = raw["parameters"]
        self._constraints: list[dict] = raw.get("cross_parameter_constraints", [])
        self._groups: dict[str, dict] = raw.get("parameter_groups", {})
        logger.debug("ParamSchema loaded %d parameters from %s", len(self._schema), schema_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def parameter_names(self) -> list[str]:
        return list(self._schema.keys())

    def get_spec(self, name: str) -> dict:
        if name not in self._schema:
            raise KeyError(f"Unknown parameter: {name!r}")
        return dict(self._schema[name])

    def defaults(self) -> dict[str, Any]:
        """Return a dict of {name: default_value} for all parameters."""
        return {name: spec["default"] for name, spec in self._schema.items()}

    def validate_one(self, name: str, value: Any) -> tuple[bool, str]:
        """
        Validate a single parameter value against the schema.

        Returns (True, "") on success or (False, reason) on failure.
        """
        if name not in self._schema:
            return False, f"Unknown parameter name: {name!r}"

        spec = self._schema[name]
        ptype = spec["type"]

        # -- Type checking ------------------------------------------------
        if ptype == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"{name}: expected float, got {type(value).__name__}"
            value = float(value)
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and value < lo:
                return False, f"{name}={value} is below minimum {lo}"
            if hi is not None and value > hi:
                return False, f"{name}={value} exceeds maximum {hi}"

        elif ptype == "int":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"{name}: expected int, got {type(value).__name__}"
            if float(value) != int(value):
                return False, f"{name}: expected integer value, got {value}"
            value = int(value)
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and value < lo:
                return False, f"{name}={value} is below minimum {lo}"
            if hi is not None and value > hi:
                return False, f"{name}={value} exceeds maximum {hi}"

        elif ptype == "bool":
            if not isinstance(value, bool):
                return False, f"{name}: expected bool, got {type(value).__name__}"

        elif ptype == "list_int":
            if not isinstance(value, (list, tuple)):
                return False, f"{name}: expected list, got {type(value).__name__}"
            min_len = spec.get("min_length", 0)
            max_len = spec.get("max_length", 1000)
            if len(value) < min_len:
                return False, f"{name}: list has {len(value)} items, minimum is {min_len}"
            if len(value) > max_len:
                return False, f"{name}: list has {len(value)} items, maximum is {max_len}"
            allowed = spec.get("allowed_values")
            if allowed is not None:
                for item in value:
                    if not isinstance(item, int):
                        return False, f"{name}: list element {item!r} is not an int"
                    if item not in allowed:
                        return False, f"{name}: list element {item} not in allowed values"
        else:
            return False, f"{name}: unknown schema type {ptype!r}"

        return True, ""

    def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate a full parameter dict including cross-parameter constraints.

        Returns (True, "") on success or (False, first_failure_reason).
        """
        # -- Per-parameter validation -------------------------------------
        for name, value in params.items():
            ok, reason = self.validate_one(name, value)
            if not ok:
                return False, reason

        # -- Cross-parameter constraints ----------------------------------
        # Only evaluate a constraint when ALL involved params are present
        # in the submitted dict (or can be resolved from it + defaults).
        # Single-param proposals that don't touch a constraint's peers are
        # allowed through -- the coordination layer enforces the full set.
        merged = {**self.defaults(), **params}
        for constraint in self._constraints:
            expr = constraint["expression"]
            involved = constraint["params"]
            # Skip if none of the constraint's params are in the submitted dict
            if not any(k in params for k in involved):
                continue
            # Build a restricted eval namespace from merged values
            ns: dict[str, Any] = {k: merged[k] for k in involved if k in merged}
            try:
                result = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307
            except Exception as exc:
                return False, f"Constraint evaluation failed for {expr!r}: {exc}"
            if not result:
                return False, f"Cross-parameter constraint violated: {expr} -- {constraint['description']}"

        return True, ""

    def coerce(self, name: str, value: Any) -> Any:
        """
        Attempt to coerce value to the correct type for the given parameter.
        Returns coerced value or raises SchemaValidationError.
        """
        spec = self._schema.get(name)
        if spec is None:
            raise SchemaValidationError(f"Unknown parameter: {name!r}")
        ptype = spec["type"]
        try:
            if ptype == "float":
                return float(value)
            elif ptype == "int":
                return int(value)
            elif ptype == "bool":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            elif ptype == "list_int":
                return [int(v) for v in value]
        except (ValueError, TypeError) as exc:
            raise SchemaValidationError(f"Cannot coerce {name}={value!r} to {ptype}: {exc}") from exc
        return value

    def fill_defaults(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return params merged with schema defaults for any missing keys."""
        result = self.defaults()
        result.update(params)
        return result


# ---------------------------------------------------------------------------
# LiveParams dataclass
# ---------------------------------------------------------------------------

@dataclass
class LiveParams:
    """
    Holds current parameter values for the LARSA live trader.

    All fields map directly to constants used in tools/live_trader_alpaca.py.
    Generated from schema defaults; updated by ParamManager.load_current().
    """

    # -- Black-hole physics -----------------------------------------------
    CF_BULL_THRESH: float = 1.2
    CF_BEAR_THRESH: float = 1.4
    BH_MASS_THRESH: float = 1.92
    BH_MASS_EXTREME: float = 3.5

    # -- Execution constraints --------------------------------------------
    MIN_HOLD_BARS: int = 4
    MAX_HOLD_BARS: int = 96
    BLOCKED_HOURS: list = field(default_factory=lambda: [1, 13, 14, 15, 17, 18])

    # -- Risk parameters --------------------------------------------------
    BASE_RISK_PCT: float = 0.02
    MAX_RISK_PCT: float = 0.05
    VOL_TARGET: float = 0.90
    KELLY_FRACTION: float = 0.25

    # -- Quaternion navigation --------------------------------------------
    NAV_OMEGA_SCALE_K: float = 0.5
    NAV_GEO_ENTRY_GATE: float = 3.0
    NAV_EMA_ALPHA: float = 0.05

    # -- Regime detection -------------------------------------------------
    HURST_WINDOW: int = 100

    # -- Filter toggles ---------------------------------------------------
    EVENT_CAL_ACTIVE: bool = True
    GRANGER_BOOST_ACTIVE: bool = True

    # -- ML integration ---------------------------------------------------
    ML_SIGNAL_BOOST: float = 1.20
    ML_SIGNAL_BOOST_THRESH: float = 0.30
    ML_SIGNAL_SUPPRESS_THRESH: float = -0.30

    # -- RL exit policy ---------------------------------------------------
    RL_EXIT_ACTIVE: bool = True
    RL_STOP_LOSS: float = 0.04

    # -- GARCH vol model --------------------------------------------------
    GARCH_ALPHA: float = 0.09
    GARCH_BETA: float = 0.88

    # -- OU mean reversion ------------------------------------------------
    OU_KAPPA_MIN: float = 0.05
    OU_KAPPA_MAX: float = 2.0

    # -- Metadata (not sent to live trader) -------------------------------
    version: str = ""
    source: str = "default"
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LiveParams":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_schema_defaults(cls, schema: ParamSchema) -> "LiveParams":
        """Build a LiveParams instance entirely from schema defaults."""
        return cls.from_dict(schema.defaults())


# ---------------------------------------------------------------------------
# ParamManager
# ---------------------------------------------------------------------------

class ParamManager:
    """
    Full parameter management client for the LARSA coordination layer.

    Communicates with the Elixir coordination server at :8781 to load,
    propose, roll back, and track parameter changes. Also supports
    background polling that fires a callback whenever parameters change.

    Usage::

        schema  = ParamSchema()
        manager = ParamManager(schema)
        current = manager.load_current()
        ok, msg = manager.validate_locally({"CF_BULL_THRESH": 1.5})
        if ok:
            manager.propose_update({"CF_BULL_THRESH": 1.5}, source="manual")
    """

    def __init__(
        self,
        schema: Optional[ParamSchema] = None,
        base_url: str = _COORD_BASE,
        timeout: float = 10.0,
    ) -> None:
        self._schema = schema or ParamSchema()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_stop = threading.Event()
        self._last_known_params: Optional[dict] = None

    # ------------------------------------------------------------------
    # Core coordination-layer operations
    # ------------------------------------------------------------------

    def load_current(self) -> LiveParams:
        """
        Fetch current parameters from the coordination layer.

        GET :8781/params/current

        Returns a LiveParams dataclass populated from the server response.
        Falls back to schema defaults if the server is unreachable.
        """
        url = f"{self._base_url}/params/current"
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            params_data = data.get("params", data)
            live = LiveParams.from_dict(params_data)
            self._last_known_params = live.to_dict()
            logger.info("Loaded current params from coordination layer (version=%s)", live.version)
            return live
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Coordination layer at %s is unreachable -- returning schema defaults", self._base_url
            )
            return LiveParams.from_schema_defaults(self._schema)
        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching params from %s -- returning schema defaults", url)
            return LiveParams.from_schema_defaults(self._schema)
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error fetching params: %s", exc)
            return LiveParams.from_schema_defaults(self._schema)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Failed to parse params response: %s", exc)
            return LiveParams.from_schema_defaults(self._schema)

    def propose_update(self, new_params: dict[str, Any], source: str = "unknown") -> bool:
        """
        Propose a parameter update to the coordination layer.

        POST :8781/params/propose

        Validates locally first; if validation fails, logs the error
        and returns False without contacting the server.

        Args:
            new_params: dict of parameter_name -> new_value
            source: identifier for who is proposing (e.g. "optuna", "iae", "manual")

        Returns True if the coordination layer accepted the proposal.
        """
        ok, reason = self.validate_locally(new_params)
        if not ok:
            logger.error("Local validation failed for propose_update: %s", reason)
            return False

        url = f"{self._base_url}/params/propose"
        payload = {"params": new_params, "source": source}
        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            result = resp.json()
            accepted = result.get("accepted", True)
            if accepted:
                logger.info("Coordination layer accepted param update from source=%s", source)
            else:
                reason_srv = result.get("reason", "no reason provided")
                logger.warning("Coordination layer rejected update from %s: %s", source, reason_srv)
            return bool(accepted)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot reach coordination layer to propose update")
            return False
        except requests.exceptions.Timeout:
            logger.error("Timeout proposing update to coordination layer")
            return False
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error proposing update: %s", exc)
            return False
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse propose response: %s", exc)
            return False

    def get_history(self, n: int = 10) -> list[dict]:
        """
        Retrieve the last n parameter update records.

        GET :8781/params/history?n=<n>

        Returns a list of dicts, each with keys:
          timestamp, source, params, delta, accepted
        Returns empty list on error.
        """
        url = f"{self._base_url}/params/history"
        try:
            resp = self._session.get(url, params={"n": n}, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            history = data.get("history", data) if isinstance(data, dict) else data
            if not isinstance(history, list):
                logger.warning("Unexpected history response format: %s", type(history))
                return []
            return history[:n]
        except requests.exceptions.ConnectionError:
            logger.warning("Coordination layer unreachable for history request")
            return []
        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching param history")
            return []
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error fetching history: %s", exc)
            return []
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Failed to parse history response: %s", exc)
            return []

    def manual_rollback(self) -> bool:
        """
        Roll back to the previous accepted parameter set.

        POST :8781/params/rollback

        This is a destructive operation -- it undoes the last accepted
        proposal. Returns True if the rollback was confirmed by the server.
        """
        url = f"{self._base_url}/params/rollback"
        try:
            resp = self._session.post(url, json={}, timeout=self._timeout)
            resp.raise_for_status()
            result = resp.json()
            success = result.get("success", True)
            if success:
                logger.warning("Manual rollback executed successfully")
            else:
                logger.error("Rollback failed: %s", result.get("reason", "unknown"))
            return bool(success)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot reach coordination layer for rollback")
            return False
        except requests.exceptions.Timeout:
            logger.error("Timeout during rollback request")
            return False
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error during rollback: %s", exc)
            return False
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse rollback response: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Local validation and delta computation
    # ------------------------------------------------------------------

    def validate_locally(self, params: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate a parameter dict against the schema without contacting the server.

        Checks individual parameter types and ranges, then cross-parameter
        constraints. Returns (True, "") on success or (False, reason).
        """
        return self._schema.validate(params)

    def compute_delta(
        self, old: LiveParams, new: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """
        Compute per-parameter change amounts and percentages.

        Args:
            old: current LiveParams
            new: dict of proposed parameter values

        Returns a dict keyed by parameter name, each value containing:
          old_value, new_value, abs_change, pct_change, changed (bool)

        Only includes parameters that are present in new.
        """
        old_dict = old.to_dict()
        delta: dict[str, dict[str, Any]] = {}

        for name, new_val in new.items():
            if name not in old_dict:
                continue
            old_val = old_dict[name]
            spec = self._schema._schema.get(name, {})
            ptype = spec.get("type", "")

            if ptype in ("float", "int") and isinstance(old_val, (int, float)):
                abs_change = float(new_val) - float(old_val)
                if old_val != 0:
                    pct_change = abs_change / abs(float(old_val)) * 100.0
                else:
                    pct_change = float("inf") if abs_change != 0 else 0.0
                changed = abs(abs_change) > 1e-12
            elif ptype == "bool":
                abs_change = int(new_val) - int(old_val)
                pct_change = 0.0
                changed = bool(new_val) != bool(old_val)
            elif ptype == "list_int":
                old_set = set(old_val) if isinstance(old_val, list) else set()
                new_set = set(new_val) if isinstance(new_val, list) else set()
                added = new_set - old_set
                removed = old_set - new_set
                abs_change = len(added) - len(removed)
                pct_change = 0.0
                changed = old_set != new_set
            else:
                abs_change = 0
                pct_change = 0.0
                changed = old_val != new_val

            delta[name] = {
                "old_value": old_val,
                "new_value": new_val,
                "abs_change": abs_change,
                "pct_change": round(pct_change, 4),
                "changed": changed,
            }

        return delta

    # ------------------------------------------------------------------
    # Background watch
    # ------------------------------------------------------------------

    def watch_for_updates(
        self,
        callback: Callable[[LiveParams, dict], None],
        poll_interval: float = 60.0,
    ) -> None:
        """
        Start a background thread that polls the coordination layer and
        fires callback(new_params, delta) whenever parameters change.

        The thread runs until stop_watch() is called. If the coordination
        layer is unreachable the thread keeps retrying silently.

        Args:
            callback: function(new_params: LiveParams, delta: dict) -> None
            poll_interval: seconds between polls (default 60)
        """
        if self._watch_thread is not None and self._watch_thread.is_alive():
            logger.warning("watch_for_updates already running -- ignoring duplicate call")
            return

        self._watch_stop.clear()

        def _poll_loop() -> None:
            logger.info("ParamManager watcher started (interval=%.0fs)", poll_interval)
            while not self._watch_stop.is_set():
                try:
                    current = self.load_current()
                    current_dict = current.to_dict()

                    if self._last_known_params is not None:
                        # Check for any changes excluding metadata fields
                        _meta = {"version", "source", "timestamp"}
                        old_cmp = {k: v for k, v in self._last_known_params.items() if k not in _meta}
                        new_cmp = {k: v for k, v in current_dict.items() if k not in _meta}
                        if old_cmp != new_cmp:
                            old_lp = LiveParams.from_dict(self._last_known_params)
                            delta = self.compute_delta(old_lp, current_dict)
                            changed = {k: v for k, v in delta.items() if v["changed"]}
                            if changed:
                                logger.info(
                                    "Parameter change detected (%d params changed) -- invoking callback",
                                    len(changed),
                                )
                                try:
                                    callback(current, changed)
                                except Exception as cb_exc:
                                    logger.error("watch_for_updates callback raised: %s", cb_exc)

                    self._last_known_params = current_dict
                except Exception as exc:
                    logger.debug("Watcher poll error (will retry): %s", exc)

                self._watch_stop.wait(poll_interval)

            logger.info("ParamManager watcher stopped")

        self._watch_thread = threading.Thread(
            target=_poll_loop, daemon=True, name="param-watcher"
        )
        self._watch_thread.start()

    def stop_watch(self) -> None:
        """Stop the background polling thread."""
        self._watch_stop.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=5.0)
            self._watch_thread = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def apply_dict_to_live_trader(self, params: LiveParams) -> dict[str, Any]:
        """
        Return a dict mapping coordination-layer names to live-trader constant names.
        Useful for patching a running live_trader_alpaca module in tests.
        """
        schema = self._schema
        result: dict[str, Any] = {}
        d = params.to_dict()
        for name, value in d.items():
            if name in ("version", "source", "timestamp"):
                continue
            spec = schema._schema.get(name, {})
            live_name = spec.get("live_trader_name", name)
            result[live_name] = value
        return result

    def summarize_history(self, n: int = 10) -> str:
        """Return a human-readable summary of the last n parameter changes."""
        history = self.get_history(n)
        if not history:
            return "No history available."
        lines = []
        for i, record in enumerate(history):
            ts = record.get("timestamp", "?")
            src = record.get("source", "?")
            accepted = record.get("accepted", True)
            changed_params = {
                k: v for k, v in record.get("delta", {}).items()
                if isinstance(v, dict) and v.get("changed")
            }
            n_changed = len(changed_params)
            status = "accepted" if accepted else "rejected"
            lines.append(f"[{i+1}] {ts} | source={src} | {n_changed} params changed | {status}")
        return "\n".join(lines)

    def diff_from_defaults(self, params: Optional[LiveParams] = None) -> dict[str, Any]:
        """
        Return parameters that differ from schema defaults.

        Useful for understanding how far the current live config
        has drifted from the baseline.
        """
        if params is None:
            params = self.load_current()
        defaults = self._schema.defaults()
        current = params.to_dict()
        diffs: dict[str, Any] = {}
        for name, default_val in defaults.items():
            cur_val = current.get(name)
            if cur_val != default_val:
                diffs[name] = {"default": default_val, "current": cur_val}
        return diffs

    def bulk_validate(self, param_sets: list[dict]) -> list[tuple[bool, str]]:
        """
        Validate a list of parameter dicts, returning one (ok, reason) per dict.
        Useful for batch-validating Optuna trial results before submission.
        """
        return [self.validate_locally(p) for p in param_sets]

    def propose_batch(
        self, param_sets: list[dict], source: str = "batch"
    ) -> list[bool]:
        """
        Propose multiple parameter sets in sequence.
        Only the first accepted proposal is applied; subsequent ones
        go into the history queue on the coordination layer.
        Returns a list of booleans indicating acceptance.
        """
        results: list[bool] = []
        for params in param_sets:
            ok = self.propose_update(params, source=source)
            results.append(ok)
        return results

    def __repr__(self) -> str:
        return (
            f"ParamManager(base_url={self._base_url!r}, "
            f"n_params={len(self._schema.parameter_names)}, "
            f"watch_active={self._watch_thread is not None and self._watch_thread.is_alive()})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

def get_default_manager() -> ParamManager:
    """Return a ParamManager configured with the default schema and coordination URL."""
    return ParamManager(ParamSchema())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mgr = get_default_manager()
    current = mgr.load_current()
    print("Current params:", json.dumps(current.to_dict(), indent=2, default=str))
    diffs = mgr.diff_from_defaults(current)
    if diffs:
        print(f"\nDrift from defaults ({len(diffs)} params):")
        for name, info in diffs.items():
            print(f"  {name}: default={info['default']}  current={info['current']}")
    else:
        print("\nAll parameters are at schema defaults.")
