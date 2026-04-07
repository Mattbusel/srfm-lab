"""
lib/config_loader.py
====================
Dynamic configuration loader with hot-reload support for the LARSA live trader.

Loads parameters from config/live_params.json (written by the param bridge /
Optuna optimizer) and monitors the file for changes using watchdog. Thread-safe
throughout using threading.RLock.

Hot-reloadable parameters (safe to change mid-session):
  CF thresholds, BH mass threshold, size limits, NAV params, ML flags,
  GARCH coefficients, OU params, risk fractions, hold bar limits.

Non-hot-reloadable (require restart):
  INSTRUMENTS dict, DB path, API credentials, BLOCKED_HOURS list.

Usage:
    from lib.config_loader import LiveConfig

    cfg = LiveConfig()
    cfg.register_callback(on_config_change)
    val = cfg.get("BH_MASS_THRESH")
    snapshot = cfg.get_snapshot()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("config_loader")

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
_REPO_ROOT       = Path(__file__).parents[1]
_LIVE_PARAMS_PATH = _REPO_ROOT / "config" / "live_params.json"

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

@dataclass
class ConfigSchema:
    """Schema entry for a single configuration parameter."""
    param_name:     str
    type_str:       str               # "float" | "int" | "bool" | "list_int" | "str"
    min_val:        Optional[float]   # None for bool / str / list types
    max_val:        Optional[float]
    default:        Any
    hot_reloadable: bool
    description:    str = ""
    group:          str = ""


# ---------------------------------------------------------------------------
# Master schema -- derived from config/param_schema.json but self-contained
# so this module has zero external I/O dependency at import time.
# ---------------------------------------------------------------------------
_SCHEMA_LIST: list[ConfigSchema] = [
    # ── Black-hole signal parameters (hot-reloadable) ──────────────────────
    ConfigSchema("CF_BULL_THRESH",          "float",    0.1,   5.0,   1.2,    True,  "CF bull signal threshold",           "blackhole"),
    ConfigSchema("CF_BEAR_THRESH",          "float",    0.1,   5.0,   1.4,    True,  "CF bear signal threshold",           "blackhole"),
    ConfigSchema("BH_MASS_THRESH",          "float",    1.0,   4.0,   1.92,   True,  "BH mass formation threshold",        "blackhole"),
    ConfigSchema("BH_MASS_EXTREME",         "float",    2.0,   8.0,   3.5,    True,  "BH extreme mass threshold",          "blackhole"),
    ConfigSchema("BH_DECAY",                "float",    0.80,  0.999, 0.924,  True,  "BH mass per-bar decay multiplier",   "blackhole"),
    ConfigSchema("BH_COLLAPSE",             "float",    0.80,  0.999, 0.992,  True,  "BH mass collapse threshold",         "blackhole"),
    ConfigSchema("BH_CTL_MIN",              "int",      1,     20,    3,      True,  "Min consecutive calm bars for BH",   "blackhole"),

    # ── Execution parameters (partially hot-reloadable) ────────────────────
    ConfigSchema("MIN_HOLD_BARS",           "int",      1,     48,    16,     True,  "Min bars before exit signal acted on", "execution"),
    ConfigSchema("MAX_HOLD_BARS",           "int",      4,     480,   96,     True,  "Max bars before forced exit",          "execution"),
    # BLOCKED_HOURS is a list -- changing it requires care (non-hot)
    ConfigSchema("BLOCKED_HOURS",           "list_int", None,  None,  [1,13,14,15,17,18], False, "UTC hours blocked for new entries", "execution"),

    # ── Risk parameters (hot-reloadable) ───────────────────────────────────
    ConfigSchema("BASE_RISK_PCT",           "float",    0.005, 0.10,  0.02,   True,  "Base fraction of NAV risked per trade",  "risk"),
    ConfigSchema("MAX_RISK_PCT",            "float",    0.01,  0.25,  0.05,   True,  "Max daily loss tolerance fraction",      "risk"),
    ConfigSchema("VOL_TARGET",              "float",    0.05,  3.0,   0.90,   True,  "GARCH annualised vol target",            "risk"),
    ConfigSchema("KELLY_FRACTION",          "float",    0.05,  1.0,   0.25,   True,  "Fractional Kelly multiplier",            "risk"),
    ConfigSchema("CRYPTO_CAP_FRAC",         "float",    0.10,  0.80,  0.40,   True,  "Max portfolio fraction in crypto",       "risk"),
    ConfigSchema("EQUITY_CAP_FRAC",         "float",    0.05,  0.80,  0.20,   True,  "Max portfolio fraction per equity pos",  "risk"),
    ConfigSchema("MIN_TRADE_FRAC",          "float",    0.0005,0.05,  0.003,  True,  "Minimum order size as fraction of NAV",  "risk"),
    ConfigSchema("DD_HALT_PCT",             "float",    0.02,  0.30,  0.10,   True,  "Drawdown % that halts new entries",      "risk"),
    ConfigSchema("DD_RESUME_PCT",           "float",    0.01,  0.20,  0.05,   True,  "Recovery % required to resume trading",  "risk"),
    ConfigSchema("DAILY_RISK",              "float",    0.005, 0.20,  0.05,   True,  "Portfolio daily risk budget (DAILY_RISK)","risk"),

    # ── QuatNav parameters (hot-reloadable) ────────────────────────────────
    ConfigSchema("NAV_OMEGA_SCALE_K",       "float",    0.0,   5.0,   0.5,    True,  "Omega ratio size penalty factor",     "nav"),
    ConfigSchema("NAV_GEO_ENTRY_GATE",      "float",    1.0,   20.0,  3.0,    True,  "Geodesic deviation entry gate",       "nav"),
    ConfigSchema("NAV_EMA_ALPHA",           "float",    0.005, 0.50,  0.05,   True,  "EMA alpha for nav baselines",         "nav"),

    # ── GARCH model (hot-reloadable) ───────────────────────────────────────
    ConfigSchema("GARCH_ALPHA",             "float",    0.01,  0.30,  0.09,   True,  "GARCH alpha (recent return weight)",  "garch"),
    ConfigSchema("GARCH_BETA",              "float",    0.50,  0.97,  0.88,   True,  "GARCH beta (variance persistence)",   "garch"),
    ConfigSchema("GARCH_TARGET_VOL",        "float",    0.05,  3.0,   0.90,   True,  "Alias for VOL_TARGET (legacy name)",  "garch"),

    # ── OU mean-reversion (hot-reloadable) ─────────────────────────────────
    ConfigSchema("OU_KAPPA_MIN",            "float",    0.001, 1.0,   0.05,   True,  "Min OU mean-reversion speed",         "ou"),
    ConfigSchema("OU_KAPPA_MAX",            "float",    0.05,  10.0,  2.0,    True,  "Max OU mean-reversion speed",         "ou"),
    ConfigSchema("OU_FRAC",                 "float",    0.01,  0.50,  0.08,   True,  "OU size contribution fraction",       "ou"),

    # ── Regime detection (hot-reloadable) ──────────────────────────────────
    ConfigSchema("HURST_WINDOW",            "int",      20,    500,   100,    True,  "Hurst exponent rolling window",       "regime"),
    ConfigSchema("CORR_NORMAL",             "float",    0.05,  0.80,  0.25,   True,  "Normal correlation assumption",       "regime"),
    ConfigSchema("CORR_STRESS",             "float",    0.20,  0.99,  0.60,   True,  "Stress correlation assumption",       "regime"),
    ConfigSchema("CORR_STRESS_THRESHOLD",   "float",    0.20,  0.99,  0.60,   True,  "Threshold to trigger stress regime",  "regime"),

    # ── ML signal parameters (hot-reloadable) ──────────────────────────────
    ConfigSchema("ML_SIGNAL_BOOST",         "float",    1.0,   3.0,   1.20,   True,  "ML boost multiplier",                 "ml"),
    ConfigSchema("ML_SIGNAL_BOOST_THRESH",  "float",    0.0,   1.0,   0.30,   True,  "ML boost activation threshold",       "ml"),
    ConfigSchema("ML_SIGNAL_SUPPRESS_THRESH","float",  -1.0,   0.0,  -0.30,   True,  "ML suppression threshold (negative)", "ml"),

    # ── RL exit policy (hot-reloadable) ────────────────────────────────────
    ConfigSchema("RL_EXIT_ACTIVE",          "bool",     None,  None,  True,   True,  "Whether RL exit policy is enabled",   "rl"),
    ConfigSchema("RL_STOP_LOSS",            "float",    0.005, 0.20,  0.04,   True,  "RL backstop hard stop-loss fraction", "rl"),

    # ── Timing / session (hot-reloadable) ──────────────────────────────────
    ConfigSchema("EVENT_CAL_ACTIVE",        "bool",     None,  None,  True,   True,  "Event calendar filter toggle",        "filters"),
    ConfigSchema("GRANGER_BOOST_ACTIVE",    "bool",     None,  None,  True,   True,  "Granger causality boost toggle",      "filters"),
    ConfigSchema("HOUR_BOOST_MULTIPLIER",   "float",    1.0,   3.0,   1.25,   True,  "Multiplier applied during boost hours","execution"),
    ConfigSchema("WINNER_PROTECTION_PCT",   "float",    0.001, 0.05,  0.005,  True,  "Trailing stop protection for winners","risk"),
    ConfigSchema("STALE_15M_MOVE",          "float",    0.0005,0.05,  0.002,  True,  "Min move to consider bar non-stale",  "execution"),
    ConfigSchema("MAX_ORDER_NOTIONAL",      "float",    1000,  2e6,   195000, True,  "Max single order notional ($)",       "execution"),

    # ── Options overlay (hot-reloadable) ───────────────────────────────────
    ConfigSchema("OPT_TARGET_DTE",          "int",      7,     120,   35,     True,  "Target DTE when opening options",     "options"),
    ConfigSchema("OPT_ROLL_DTE",            "int",      1,     30,    7,      True,  "Roll/close DTE threshold",            "options"),
    ConfigSchema("OPT_NOTIONAL_FRAC",       "float",    0.001, 0.10,  0.015,  True,  "Option position size fraction of NAV","options"),
    ConfigSchema("OPT_MIN_TF",              "int",      0,     7,     2,      True,  "Min TF score to open option overlay", "options"),
    ConfigSchema("OPT_EXIT_TF",             "int",      0,     7,     0,      True,  "TF score at/below which option exits","options"),
    ConfigSchema("OPT_MAX_HOLD_BARS",       "int",      4,     480,   96,     True,  "Max bars for option positions",       "options"),

    # ── Non-hot-reloadable (require restart) ───────────────────────────────
    ConfigSchema("DB_PATH",                 "str",      None,  None,  str(_REPO_ROOT / "execution" / "live_trades.db"), False, "SQLite DB path",       "infra"),
    ConfigSchema("OVERRIDES_TTL_SECS",      "int",      30,    3600,  300,    False, "Override file re-read interval",     "infra"),
    ConfigSchema("STRATEGY_VERSION",        "str",      None,  None,  "larsa_v18", False, "Strategy version tag",          "infra"),
    ConfigSchema("ALPACA_API_KEY",          "str",      None,  None,  "",     False, "Alpaca API key",                     "credentials"),
    ConfigSchema("ALPACA_SECRET_KEY",       "str",      None,  None,  "",     False, "Alpaca secret key",                  "credentials"),
    ConfigSchema("ALPACA_PAPER",            "bool",     None,  None,  True,   False, "Whether to use Alpaca paper trading","credentials"),
]

# Build lookup dict
SCHEMA: dict[str, ConfigSchema] = {s.param_name: s for s in _SCHEMA_LIST}


# ---------------------------------------------------------------------------
# Cross-parameter constraint definitions
# ---------------------------------------------------------------------------
@dataclass
class CrossConstraint:
    description: str
    params:      list[str]
    # Returns None if OK, error message if violated
    check:       Callable[[dict[str, Any]], Optional[str]]


_CROSS_CONSTRAINTS: list[CrossConstraint] = [
    CrossConstraint(
        "CF_BEAR_THRESH >= CF_BULL_THRESH",
        ["CF_BEAR_THRESH", "CF_BULL_THRESH"],
        lambda d: None if d.get("CF_BEAR_THRESH", 1.4) >= d.get("CF_BULL_THRESH", 1.2)
                  else "CF_BEAR_THRESH must be >= CF_BULL_THRESH",
    ),
    CrossConstraint(
        "BH_MASS_EXTREME > BH_MASS_THRESH",
        ["BH_MASS_EXTREME", "BH_MASS_THRESH"],
        lambda d: None if d.get("BH_MASS_EXTREME", 3.5) > d.get("BH_MASS_THRESH", 1.92)
                  else "BH_MASS_EXTREME must be > BH_MASS_THRESH",
    ),
    CrossConstraint(
        "MAX_HOLD_BARS > MIN_HOLD_BARS",
        ["MAX_HOLD_BARS", "MIN_HOLD_BARS"],
        lambda d: None if d.get("MAX_HOLD_BARS", 96) > d.get("MIN_HOLD_BARS", 16)
                  else "MAX_HOLD_BARS must be > MIN_HOLD_BARS",
    ),
    CrossConstraint(
        "MAX_RISK_PCT >= BASE_RISK_PCT",
        ["MAX_RISK_PCT", "BASE_RISK_PCT"],
        lambda d: None if d.get("MAX_RISK_PCT", 0.05) >= d.get("BASE_RISK_PCT", 0.02)
                  else "MAX_RISK_PCT must be >= BASE_RISK_PCT",
    ),
    CrossConstraint(
        "GARCH_ALPHA + GARCH_BETA < 1.0 (stationarity)",
        ["GARCH_ALPHA", "GARCH_BETA"],
        lambda d: None if d.get("GARCH_ALPHA", 0.09) + d.get("GARCH_BETA", 0.88) < 1.0
                  else "GARCH_ALPHA + GARCH_BETA must be < 1.0 for stationarity",
    ),
    CrossConstraint(
        "OU_KAPPA_MIN < OU_KAPPA_MAX",
        ["OU_KAPPA_MIN", "OU_KAPPA_MAX"],
        lambda d: None if d.get("OU_KAPPA_MIN", 0.05) < d.get("OU_KAPPA_MAX", 2.0)
                  else "OU_KAPPA_MIN must be < OU_KAPPA_MAX",
    ),
    CrossConstraint(
        "ML_SIGNAL_SUPPRESS_THRESH < ML_SIGNAL_BOOST_THRESH",
        ["ML_SIGNAL_SUPPRESS_THRESH", "ML_SIGNAL_BOOST_THRESH"],
        lambda d: None if d.get("ML_SIGNAL_SUPPRESS_THRESH", -0.30) < d.get("ML_SIGNAL_BOOST_THRESH", 0.30)
                  else "ML_SIGNAL_SUPPRESS_THRESH must be < ML_SIGNAL_BOOST_THRESH",
    ),
    CrossConstraint(
        "DD_RESUME_PCT < DD_HALT_PCT",
        ["DD_RESUME_PCT", "DD_HALT_PCT"],
        lambda d: None if d.get("DD_RESUME_PCT", 0.05) < d.get("DD_HALT_PCT", 0.10)
                  else "DD_RESUME_PCT must be < DD_HALT_PCT",
    ),
    CrossConstraint(
        "OPT_EXIT_TF <= OPT_MIN_TF",
        ["OPT_EXIT_TF", "OPT_MIN_TF"],
        lambda d: None if d.get("OPT_EXIT_TF", 0) <= d.get("OPT_MIN_TF", 2)
                  else "OPT_EXIT_TF must be <= OPT_MIN_TF",
    ),
    CrossConstraint(
        "OPT_ROLL_DTE < OPT_TARGET_DTE",
        ["OPT_ROLL_DTE", "OPT_TARGET_DTE"],
        lambda d: None if d.get("OPT_ROLL_DTE", 7) < d.get("OPT_TARGET_DTE", 35)
                  else "OPT_ROLL_DTE must be < OPT_TARGET_DTE",
    ),
]


# ---------------------------------------------------------------------------
# ConfigDiff
# ---------------------------------------------------------------------------

@dataclass
class ParamChange:
    """Represents a single parameter change between two configs."""
    name:           str
    old_value:      Any
    new_value:      Any
    change_pct:     Optional[float]   # None for non-numeric / bools
    is_significant: bool              # True if abs change > 10%

    def __str__(self) -> str:
        pct_str = f" ({self.change_pct:+.1f}%)" if self.change_pct is not None else ""
        flag    = " [SIGNIFICANT]" if self.is_significant else ""
        return f"{self.name}: {self.old_value} -> {self.new_value}{pct_str}{flag}"


class ConfigDiff:
    """Computes the diff between two config snapshots."""

    SIGNIFICANT_THRESHOLD = 0.10   # 10%

    @classmethod
    def compute(cls, old_config: dict[str, Any], new_config: dict[str, Any]) -> list[ParamChange]:
        """Return list of ParamChange for all parameters that changed."""
        changes: list[ParamChange] = []
        all_keys = set(old_config) | set(new_config)
        for key in sorted(all_keys):
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            if old_val == new_val:
                continue
            change_pct    = cls._change_pct(old_val, new_val)
            is_significant = (
                change_pct is not None and abs(change_pct) > cls.SIGNIFICANT_THRESHOLD * 100
            )
            changes.append(ParamChange(
                name           = key,
                old_value      = old_val,
                new_value      = new_val,
                change_pct     = change_pct,
                is_significant = is_significant,
            ))
        return changes

    @staticmethod
    def _change_pct(old: Any, new: Any) -> Optional[float]:
        """Compute percentage change, returning None for non-numeric or bool values."""
        if old is None or new is None:
            return None
        # Booleans are subclasses of int in Python; treat them as non-numeric for pct
        if isinstance(old, bool) or isinstance(new, bool):
            return None
        try:
            old_f = float(old)
            new_f = float(new)
            if old_f == 0.0:
                return None
            return (new_f - old_f) / abs(old_f) * 100.0
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Watchdog file monitor (optional dependency)
# ---------------------------------------------------------------------------

def _make_file_handler(callback: Callable[[], None], path: Path):
    """Build a watchdog FileSystemEventHandler that fires callback on modify."""
    try:
        from watchdog.events import FileSystemEventHandler

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and Path(event.src_path).resolve() == path.resolve():
                    callback()

            def on_created(self, event):
                if not event.is_directory and Path(event.src_path).resolve() == path.resolve():
                    callback()

        return _Handler()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# LiveConfig
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when a config value fails schema or cross-constraint validation."""


class LiveConfig:
    """
    Thread-safe live configuration container with hot-reload support.

    Loads parameters from config/live_params.json (written by param bridge).
    Uses watchdog for filesystem monitoring; falls back to polling if not
    available.

    Thread safety: all reads and writes are protected by threading.RLock so
    the live trader can read config from multiple signal threads without
    data races.
    """

    POLL_INTERVAL_SECS = 15   # fallback polling interval when watchdog absent

    def __init__(
        self,
        config_path: Optional[Path] = None,
        watch: bool = True,
    ) -> None:
        self._path      = Path(config_path) if config_path else _LIVE_PARAMS_PATH
        self._lock      = threading.RLock()
        self._config:   dict[str, Any] = {}
        self._callbacks: list[Callable[[list[ParamChange]], None]] = []
        self._observer  = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_mtime: float = 0.0
        self._load_count: int   = 0

        # Initialise from defaults so the trader can run even if file missing
        self._config = self._defaults()

        # Load from file if it exists
        if self._path.exists():
            try:
                self._apply(self._read_file())
                log.info("config_loader: loaded %d params from %s", len(self._config), self._path)
            except Exception as exc:
                log.warning("config_loader: initial load failed, using defaults -- %s", exc)

        if watch:
            self._start_watch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, param_name: str, default: Any = None) -> Any:
        """Thread-safe parameter read. Returns default if param not found."""
        with self._lock:
            return self._config.get(param_name, default)

    def get_snapshot(self) -> dict[str, Any]:
        """Return a complete copy of the current config for audit trail."""
        with self._lock:
            return dict(self._config)

    def register_callback(self, fn: Callable[[list[ParamChange]], None]) -> None:
        """Register a callback invoked on successful hot-reload with the list of changes."""
        with self._lock:
            self._callbacks.append(fn)

    def reload(self) -> bool:
        """
        Attempt to reload config from disk.

        Returns True if a reload happened (file changed and was valid),
        False if no change or validation failed (old config retained).
        """
        try:
            raw = self._read_file()
        except FileNotFoundError:
            log.debug("config_loader: file not found on reload, skipping")
            return False
        except json.JSONDecodeError as exc:
            log.error("config_loader: JSON parse error during reload -- %s", exc)
            return False

        try:
            validated = self._validate(raw)
        except ConfigValidationError as exc:
            log.error("config_loader: validation failed, keeping current config -- %s", exc)
            return False

        with self._lock:
            old_snapshot = dict(self._config)
            self._apply_validated(validated)
            new_snapshot = dict(self._config)

        changes = ConfigDiff.compute(old_snapshot, new_snapshot)
        if not changes:
            return False

        self._load_count += 1
        self._log_changes(changes)

        for cb in list(self._callbacks):
            try:
                cb(changes)
            except Exception as exc:
                log.error("config_loader: callback error -- %s", exc)

        return True

    def stop(self) -> None:
        """Stop the file watcher / polling thread."""
        self._stop_event.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=3)
            except Exception:
                pass
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5)

    def get_schema(self) -> dict[str, ConfigSchema]:
        """Return the full parameter schema."""
        return dict(SCHEMA)

    def get_hot_reloadable_params(self) -> list[str]:
        """Return names of all hot-reloadable parameters."""
        return [s.param_name for s in _SCHEMA_LIST if s.hot_reloadable]

    def get_non_hot_reloadable_params(self) -> list[str]:
        """Return names of all non-hot-reloadable parameters."""
        return [s.param_name for s in _SCHEMA_LIST if not s.hot_reloadable]

    def load_count(self) -> int:
        """Number of successful hot-reloads since start."""
        return self._load_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _defaults(self) -> dict[str, Any]:
        """Build a defaults dict from the schema."""
        return {s.param_name: s.default for s in _SCHEMA_LIST}

    def _read_file(self) -> dict[str, Any]:
        """Read and JSON-parse the config file."""
        with open(self._path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Validate raw dict against schema. Returns merged config (defaults +
        validated overrides). Raises ConfigValidationError on failure.
        Only validates known schema keys -- unknown keys are ignored.
        """
        merged: dict[str, Any] = self._defaults()

        for key, value in raw.items():
            if key not in SCHEMA:
                log.debug("config_loader: unknown param '%s' ignored", key)
                continue

            schema = SCHEMA[key]

            # Type coercion and validation
            try:
                coerced = self._coerce(value, schema)
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(
                    f"param '{key}': type error -- {exc}"
                ) from exc

            # Range validation
            if schema.min_val is not None and isinstance(coerced, (int, float)):
                if coerced < schema.min_val:
                    raise ConfigValidationError(
                        f"param '{key}' value {coerced} < min {schema.min_val}"
                    )
            if schema.max_val is not None and isinstance(coerced, (int, float)):
                if coerced > schema.max_val:
                    raise ConfigValidationError(
                        f"param '{key}' value {coerced} > max {schema.max_val}"
                    )

            merged[key] = coerced

        # Cross-parameter constraints
        for constraint in _CROSS_CONSTRAINTS:
            error = constraint.check(merged)
            if error is not None:
                raise ConfigValidationError(
                    f"cross-constraint violated: {constraint.description} -- {error}"
                )

        return merged

    @staticmethod
    def _coerce(value: Any, schema: ConfigSchema) -> Any:
        """Coerce value to the type specified in schema."""
        t = schema.type_str
        if t == "float":
            return float(value)
        if t == "int":
            return int(value)
        if t == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        if t == "list_int":
            if isinstance(value, list):
                return [int(x) for x in value]
            raise ValueError(f"expected list, got {type(value).__name__}")
        if t == "str":
            return str(value)
        return value

    def _apply(self, raw: dict[str, Any]) -> None:
        """Validate and apply in one call (used during initial load, holds no lock)."""
        validated = self._validate(raw)
        self._apply_validated(validated)

    def _apply_validated(self, validated: dict[str, Any]) -> None:
        """Apply a pre-validated config dict (call under lock)."""
        self._config.update(validated)

    def _start_watch(self) -> None:
        """Start watchdog observer or polling fallback."""
        try:
            from watchdog.observers import Observer

            handler = _make_file_handler(self._on_file_event, self._path)
            if handler is None:
                raise ImportError("watchdog handler creation failed")

            observer = Observer()
            observer.schedule(handler, str(self._path.parent), recursive=False)
            observer.daemon = True
            observer.start()
            self._observer = observer
            log.info("config_loader: watchdog monitoring %s", self._path)
        except ImportError:
            log.info("config_loader: watchdog not available, using polling every %ds", self.POLL_INTERVAL_SECS)
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def _on_file_event(self) -> None:
        """Watchdog callback -- debounced with mtime check."""
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime <= self._last_mtime:
            return
        self._last_mtime = mtime
        # Small sleep to allow writer to finish flushing
        time.sleep(0.05)
        self.reload()

    def _poll_loop(self) -> None:
        """Fallback polling thread."""
        while not self._stop_event.wait(self.POLL_INTERVAL_SECS):
            try:
                if not self._path.exists():
                    continue
                mtime = self._path.stat().st_mtime
                if mtime > self._last_mtime:
                    self._last_mtime = mtime
                    self.reload()
            except Exception as exc:
                log.debug("config_loader: poll error -- %s", exc)

    @staticmethod
    def _log_changes(changes: list[ParamChange]) -> None:
        """Log each changed parameter at appropriate severity."""
        for c in changes:
            schema = SCHEMA.get(c.name)
            if schema and not schema.hot_reloadable:
                log.warning(
                    "config_loader: non-hot-reloadable param changed -- %s (restart needed)", c
                )
            elif c.is_significant:
                log.warning("config_loader: SIGNIFICANT change -- %s", c)
            else:
                log.info("config_loader: param updated -- %s", c)


# ---------------------------------------------------------------------------
# Convenience: write a default live_params.json if none exists
# ---------------------------------------------------------------------------

def write_default_params(path: Optional[Path] = None) -> Path:
    """
    Write a default live_params.json containing all schema defaults.
    Useful for first-time setup or test fixtures.
    """
    target = Path(path) if path else _LIVE_PARAMS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    defaults = {s.param_name: s.default for s in _SCHEMA_LIST if s.hot_reloadable}
    defaults["_written_at"] = datetime.now(timezone.utc).isoformat()
    defaults["_schema_version"] = "1.0.0"
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(defaults, fh, indent=2)
    log.info("config_loader: wrote default params to %s", target)
    return target


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_singleton: Optional[LiveConfig] = None
_singleton_lock = threading.Lock()


def get_live_config(config_path: Optional[Path] = None) -> LiveConfig:
    """Return (or create) the module-level LiveConfig singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = LiveConfig(config_path=config_path)
    return _singleton
