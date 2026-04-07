"""
bridge/iae_signal_bridge.py
============================
Bridge between the IAE (Idea-Arithmetic Engine) genome evolution system
(:8780) and the live signal library.

Architecture
------------
  IAESignalBridge
    - Polls :8780/patterns/confirmed every 10 minutes
    - Translates each confirmed IAE pattern into a SignalSpec
    - Validates spec against signal library schema
    - Deploys valid signals to the SQLite signal registry
    - Retires stale signals whose ICIR drops below 0.20

  SignalSpec (dataclass)
    - Portable description of a deployable signal
    - Contains signal_fn_str: an evaluable Python expression
    - Tracks source, regime requirements, minimum IC threshold

  SignalRegistry
    - SQLite-backed persistent store of deployed_signals table
    - Thread-safe reads and writes via threading.Lock

Usage::

    import asyncio
    from bridge.iae_signal_bridge import IAESignalBridge

    bridge = IAESignalBridge()
    asyncio.run(bridge.run())
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import aiohttp

log = logging.getLogger("bridge.iae_signal_bridge")

_REPO_ROOT = Path(__file__).parents[1]
_REGISTRY_DB = _REPO_ROOT / "execution" / "signal_registry.db"
_IAE_BASE = "http://localhost:8780"
_COORDINATION_BASE = "http://localhost:8781"

_POLL_INTERVAL_SECS = 600   # 10 minutes
_ICIR_RETIRE_THRESHOLD = 0.20
_ICIR_PROBATION_THRESHOLD = 0.30
_MAX_SIGNAL_AGE_DAYS = 30


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SignalSpec:
    """
    Portable specification for a deployable signal derived from an IAE pattern.

    Attributes
    ----------
    name : str
        Unique signal identifier (snake_case).
    signal_fn_str : str
        A Python expression string that can be eval()'d against a namespace
        containing ``prices`` (pd.Series), ``volume`` (pd.Series), and
        standard numeric imports.  Must return a pd.Series.
    required_regime : str
        Regime label required for this signal to be active.
        Use "any" for regime-agnostic signals.
    min_ic : float
        Minimum Information Coefficient this signal must sustain to remain
        active.  Signals below this are placed on probation.
    created_at : str
        ISO-8601 timestamp of creation.
    source : str
        Where this spec was derived from ("iae_genome", "manual", etc.).
    pattern_id : str
        Reference back to the IAE pattern that produced this signal.
    params : dict
        Additional pattern-specific parameters embedded in the signal fn.
    description : str
        Human-readable description of the signal logic.
    """

    name: str
    signal_fn_str: str
    required_regime: str = "any"
    min_ic: float = 0.02
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "iae_genome"
    pattern_id: str = ""
    params: dict = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "signal_fn_str":  self.signal_fn_str,
            "required_regime": self.required_regime,
            "min_ic":         self.min_ic,
            "created_at":     self.created_at,
            "source":         self.source,
            "pattern_id":     self.pattern_id,
            "params":         self.params,
            "description":    self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SignalSpec":
        return cls(
            name=d["name"],
            signal_fn_str=d["signal_fn_str"],
            required_regime=d.get("required_regime", "any"),
            min_ic=float(d.get("min_ic", 0.02)),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            source=d.get("source", "iae_genome"),
            pattern_id=d.get("pattern_id", ""),
            params=d.get("params", {}),
            description=d.get("description", ""),
        )


@dataclass
class DeployedSignalRecord:
    """Row in the deployed_signals table."""

    id: int
    name: str
    pattern_id: str
    deployed_at: str
    retired_at: Optional[str]
    current_icir: float
    status: str  # "active" | "probation" | "retired"
    signal_fn_str: str
    required_regime: str
    min_ic: float
    source: str


# ---------------------------------------------------------------------------
# Signal validation
# ---------------------------------------------------------------------------

# IAE pattern types that we know how to translate
_KNOWN_PATTERN_TYPES = {
    "momentum_burst",
    "mean_reversion_ou",
    "volatility_regime_shift",
    "microstructure_imbalance",
    "black_hole_formation",
    "centrifugal_force_threshold",
    "regime_transition",
    "cross_sectional_momentum",
    "carry_signal",
    "liquidity_stress",
}

# Required fields in an IAE pattern payload
_REQUIRED_PATTERN_FIELDS = {"pattern_id", "pattern_type", "confidence", "entry_conditions"}


def validate_signal_spec(spec: SignalSpec) -> list[str]:
    """
    Validate a SignalSpec against the signal library schema.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    if not spec.name or not spec.name.replace("_", "").isalnum():
        errors.append(f"Invalid signal name: '{spec.name}' -- must be alphanumeric+underscore")

    if not spec.signal_fn_str.strip():
        errors.append("signal_fn_str is empty")

    if not (0.0 <= spec.min_ic <= 1.0):
        errors.append(f"min_ic={spec.min_ic} out of range [0, 1]")

    # Syntax check the signal expression
    fn_errors = _check_expression_syntax(spec.signal_fn_str)
    if fn_errors:
        errors.extend(fn_errors)

    # Regime label whitelist
    valid_regimes = {
        "any", "trending", "mean_reverting", "volatile", "low_vol",
        "bull", "bear", "neutral", "crisis", "risk_on", "risk_off",
    }
    if spec.required_regime not in valid_regimes:
        errors.append(
            f"Unknown required_regime '{spec.required_regime}' -- "
            f"expected one of {sorted(valid_regimes)}"
        )

    return errors


def _check_expression_syntax(expr: str) -> list[str]:
    """
    Attempt to compile the signal expression for syntax errors.
    Returns list of error strings.
    """
    errors: list[str] = []
    try:
        compile(expr, "<signal_fn_str>", "eval")
    except SyntaxError as exc:
        errors.append(f"SyntaxError in signal_fn_str: {exc}")
    except Exception as exc:
        errors.append(f"Compile error in signal_fn_str: {exc}")
    return errors


# ---------------------------------------------------------------------------
# Pattern translation
# ---------------------------------------------------------------------------

class PatternTranslator:
    """
    Converts IAE confirmed patterns into SignalSpec objects.

    IAE patterns have the schema::

        {
          "pattern_id": "iae_20260406_abc123",
          "pattern_type": "momentum_burst",
          "confidence": 0.82,
          "entry_conditions": {
            "lookback": 20,
            "threshold": 1.5,
            "regime": "trending",
            ...
          },
          "regime_requirements": ["trending", "bull"],
          "fitness_score": 0.74,
          "generation": 42,
          "genome_id": "g42_xyz"
        }
    """

    def translate_pattern_to_signal(self, pattern: dict) -> SignalSpec | None:
        """
        Convert an IAE pattern dict into a SignalSpec.

        Returns None if the pattern cannot be translated.
        """
        pattern_type = pattern.get("pattern_type", "")
        if pattern_type not in _KNOWN_PATTERN_TYPES:
            log.warning(
                "PatternTranslator: unknown pattern_type '%s' -- skipping", pattern_type
            )
            return None

        handler = getattr(self, f"_translate_{pattern_type}", None)
        if handler is None:
            return self._translate_generic(pattern)
        return handler(pattern)

    # ------------------------------------------------------------------
    # Per-type translators
    # ------------------------------------------------------------------

    def _translate_momentum_burst(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        lookback = int(cond.get("lookback", 20))
        threshold = float(cond.get("threshold", 1.5))
        regime = _pick_regime(pattern)
        fn = (
            f"prices.pct_change({lookback}).rolling(3).mean() / "
            f"prices.pct_change({lookback}).rolling(20).std().replace(0, float('nan'))"
            f" - {threshold}"
        )
        return SignalSpec(
            name=f"iae_momentum_burst_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.03)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"lookback": lookback, "threshold": threshold},
            description=f"IAE momentum burst signal (lookback={lookback}, thresh={threshold})",
        )

    def _translate_mean_reversion_ou(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        half_life = float(cond.get("half_life", 10.0))
        z_thresh = float(cond.get("z_threshold", 1.5))
        span = max(2, int(half_life * 2))
        regime = _pick_regime(pattern)
        fn = (
            f"-(prices - prices.ewm(span={span}, adjust=False).mean()) / "
            f"prices.ewm(span={span}, adjust=False).std().replace(0, float('nan')) * "
            f"(abs(prices - prices.ewm(span={span}, adjust=False).mean()) > "
            f"prices.ewm(span={span}, adjust=False).std() * {z_thresh}).astype(float)"
        )
        return SignalSpec(
            name=f"iae_ou_reversion_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.02)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"half_life": half_life, "z_threshold": z_thresh},
            description=f"IAE OU mean reversion (half_life={half_life}, z={z_thresh})",
        )

    def _translate_volatility_regime_shift(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        short_vol = int(cond.get("short_vol_window", 5))
        long_vol = int(cond.get("long_vol_window", 20))
        regime = _pick_regime(pattern)
        fn = (
            f"prices.pct_change().rolling({short_vol}).std() / "
            f"prices.pct_change().rolling({long_vol}).std().replace(0, float('nan')) - 1.0"
        )
        return SignalSpec(
            name=f"iae_vol_shift_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.025)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"short_vol": short_vol, "long_vol": long_vol},
            description=f"IAE volatility regime shift ({short_vol}/{long_vol})",
        )

    def _translate_microstructure_imbalance(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        window = int(cond.get("window", 10))
        regime = _pick_regime(pattern)
        fn = (
            f"(volume - volume.rolling({window}).mean()) / "
            f"volume.rolling({window}).std().replace(0, float('nan')) * "
            f"prices.pct_change().apply(lambda x: 1 if x > 0 else -1)"
        )
        return SignalSpec(
            name=f"iae_microstructure_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.02)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"window": window},
            description=f"IAE microstructure imbalance (window={window})",
        )

    def _translate_black_hole_formation(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        atr_mult = float(cond.get("atr_multiplier", 2.0))
        atr_window = int(cond.get("atr_window", 14))
        regime = _pick_regime(pattern)
        fn = (
            f"(prices - prices.rolling({atr_window}).mean()).abs() / "
            f"(prices.diff().abs().rolling({atr_window}).mean() * {atr_mult}).replace(0, float('nan')) - 1.0"
        )
        return SignalSpec(
            name=f"iae_blackhole_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.04)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"atr_mult": atr_mult, "atr_window": atr_window},
            description=f"IAE black-hole formation (atr_mult={atr_mult})",
        )

    def _translate_centrifugal_force_threshold(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        cf_thresh = float(cond.get("cf_threshold", 1.5))
        momentum_window = int(cond.get("momentum_window", 12))
        regime = _pick_regime(pattern)
        fn = (
            f"prices.pct_change({momentum_window}).rolling(3).sum() * "
            f"(prices.pct_change({momentum_window}).abs() > {cf_thresh} * "
            f"prices.pct_change({momentum_window}).rolling(20).std()).astype(float)"
        )
        return SignalSpec(
            name=f"iae_cf_thresh_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.035)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"cf_threshold": cf_thresh, "momentum_window": momentum_window},
            description=f"IAE centrifugal force threshold (cf={cf_thresh})",
        )

    def _translate_regime_transition(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        window = int(cond.get("transition_window", 15))
        regime = _pick_regime(pattern)
        fn = (
            f"prices.pct_change().rolling({window}).mean() / "
            f"prices.pct_change().rolling({window * 3}).mean().replace(0, float('nan')) - 1.0"
        )
        return SignalSpec(
            name=f"iae_regime_trans_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.02)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"window": window},
            description=f"IAE regime transition detector (window={window})",
        )

    def _translate_cross_sectional_momentum(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        lookback = int(cond.get("lookback", 60))
        skip = int(cond.get("skip_recent", 5))
        regime = _pick_regime(pattern)
        fn = (
            f"prices.pct_change({lookback}).shift({skip})"
        )
        return SignalSpec(
            name=f"iae_xsec_mom_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.03)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"lookback": lookback, "skip_recent": skip},
            description=f"IAE cross-sectional momentum (lookback={lookback}, skip={skip})",
        )

    def _translate_carry_signal(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        carry_window = int(cond.get("carry_window", 30))
        regime = _pick_regime(pattern)
        fn = (
            f"(prices / prices.shift({carry_window}) - 1.0) - "
            f"prices.pct_change().rolling({carry_window}).mean()"
        )
        return SignalSpec(
            name=f"iae_carry_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.02)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"carry_window": carry_window},
            description=f"IAE carry signal (window={carry_window})",
        )

    def _translate_liquidity_stress(self, pattern: dict) -> SignalSpec:
        cond = pattern.get("entry_conditions", {})
        window = int(cond.get("window", 10))
        stress_thresh = float(cond.get("stress_threshold", 2.0))
        regime = _pick_regime(pattern)
        fn = (
            f"-prices.pct_change() * "
            f"(volume / volume.rolling({window}).mean().replace(0, float('nan')) < 1.0 / {stress_thresh}).astype(float)"
        )
        return SignalSpec(
            name=f"iae_liq_stress_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.025)) * 0.1,
            pattern_id=str(pattern.get("pattern_id", "")),
            params={"window": window, "stress_threshold": stress_thresh},
            description=f"IAE liquidity stress (window={window})",
        )

    def _translate_generic(self, pattern: dict) -> SignalSpec:
        """Fallback translator for unrecognised pattern types."""
        cond = pattern.get("entry_conditions", {})
        window = int(cond.get("lookback", cond.get("window", 20)))
        regime = _pick_regime(pattern)
        fn = f"prices.pct_change({window}).rolling(3).mean()"
        return SignalSpec(
            name=f"iae_generic_{pattern.get('pattern_id', 'unknown')[-6:]}",
            signal_fn_str=fn,
            required_regime=regime,
            min_ic=float(pattern.get("fitness_score", 0.02)) * 0.08,
            pattern_id=str(pattern.get("pattern_id", "")),
            params=cond,
            description=f"IAE generic signal (type={pattern.get('pattern_type')})",
        )


def _pick_regime(pattern: dict) -> str:
    """Extract regime requirement from pattern, defaulting to 'any'."""
    reqs = pattern.get("regime_requirements", [])
    if isinstance(reqs, list) and reqs:
        return str(reqs[0])
    return str(pattern.get("entry_conditions", {}).get("regime", "any"))


# ---------------------------------------------------------------------------
# SignalRegistry -- SQLite-backed store
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS deployed_signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    pattern_id   TEXT    NOT NULL,
    deployed_at  TEXT    NOT NULL,
    retired_at   TEXT,
    current_icir REAL    NOT NULL DEFAULT 0.0,
    status       TEXT    NOT NULL DEFAULT 'active',
    signal_fn_str TEXT   NOT NULL,
    required_regime TEXT NOT NULL DEFAULT 'any',
    min_ic       REAL    NOT NULL DEFAULT 0.02,
    source       TEXT    NOT NULL DEFAULT 'iae_genome',
    description  TEXT    NOT NULL DEFAULT '',
    params_json  TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_ds_status ON deployed_signals (status);
CREATE INDEX IF NOT EXISTS idx_ds_pattern ON deployed_signals (pattern_id);
"""


class SignalRegistry:
    """SQLite-backed registry of deployed signals."""

    def __init__(self, db_path: Path = _REGISTRY_DB) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.executescript(_DDL)

    def insert_signal(self, spec: SignalSpec) -> int:
        """Insert a new signal record. Returns the new row ID."""
        with self._lock:
            with self._conn() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO deployed_signals
                      (name, pattern_id, deployed_at, current_icir, status,
                       signal_fn_str, required_regime, min_ic, source, description, params_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        spec.name,
                        spec.pattern_id,
                        datetime.now(timezone.utc).isoformat(),
                        0.0,
                        "active",
                        spec.signal_fn_str,
                        spec.required_regime,
                        spec.min_ic,
                        spec.source,
                        spec.description,
                        json.dumps(spec.params),
                    ),
                )
                return cursor.lastrowid or -1

    def signal_exists(self, name: str) -> bool:
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT id FROM deployed_signals WHERE name = ?", (name,)
                ).fetchone()
                return row is not None

    def update_icir(self, name: str, icir: float) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE deployed_signals SET current_icir = ? WHERE name = ?",
                    (icir, name),
                )

    def retire_signal(self, name: str) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE deployed_signals SET status = 'retired', retired_at = ? WHERE name = ?",
                    (datetime.now(timezone.utc).isoformat(), name),
                )

    def get_active_signals(self) -> list[DeployedSignalRecord]:
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM deployed_signals WHERE status IN ('active', 'probation')"
                ).fetchall()
                return [self._row_to_record(r) for r in rows]

    def get_all_signals(self) -> list[DeployedSignalRecord]:
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute("SELECT * FROM deployed_signals").fetchall()
                return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DeployedSignalRecord:
        return DeployedSignalRecord(
            id=row["id"],
            name=row["name"],
            pattern_id=row["pattern_id"],
            deployed_at=row["deployed_at"],
            retired_at=row["retired_at"],
            current_icir=float(row["current_icir"]),
            status=row["status"],
            signal_fn_str=row["signal_fn_str"],
            required_regime=row["required_regime"],
            min_ic=float(row["min_ic"]),
            source=row["source"],
        )


# ---------------------------------------------------------------------------
# IAESignalBridge
# ---------------------------------------------------------------------------

class IAESignalBridge:
    """
    Polls the IAE genome evolution system for confirmed patterns and
    deploys them as signals into the live signal registry.

    Lifecycle
    ---------
    1. Every 10 minutes: GET :8780/patterns/confirmed
    2. For each new pattern: translate -> validate -> deploy to SQLite
    3. Periodically: poll signal decay monitor -> retire ICIR < 0.20 signals
    4. POST deployment notifications to coordination layer (:8781)
    """

    def __init__(
        self,
        iae_url: str = _IAE_BASE,
        coordination_url: str = _COORDINATION_BASE,
        db_path: Path = _REGISTRY_DB,
        poll_interval_secs: float = _POLL_INTERVAL_SECS,
    ) -> None:
        self._iae_url = iae_url
        self._coordination_url = coordination_url
        self._poll_interval = poll_interval_secs
        self._registry = SignalRegistry(db_path)
        self._translator = PatternTranslator()
        self._running = False
        self._seen_pattern_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the bridge until stopped."""
        self._running = True
        # Pre-load known pattern IDs to avoid re-deploying after restart
        for rec in self._registry.get_all_signals():
            if rec.pattern_id:
                self._seen_pattern_ids.add(rec.pattern_id)

        log.info(
            "IAESignalBridge: starting -- polling %s/patterns/confirmed every %.0fs",
            self._iae_url,
            self._poll_interval,
        )
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                asyncio.create_task(self._poll_patterns_loop(session), name="iae_pattern_poll"),
                asyncio.create_task(self._decay_monitor_loop(), name="signal_decay"),
            ]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self._running = False
                log.info("IAESignalBridge: stopped.")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Pattern polling
    # ------------------------------------------------------------------

    async def _poll_patterns_loop(self, session: aiohttp.ClientSession) -> None:
        while self._running:
            try:
                await self._fetch_and_process_patterns(session)
            except Exception as exc:
                log.error("IAESignalBridge: pattern poll error: %s", exc)
            await asyncio.sleep(self._poll_interval)

    async def _fetch_and_process_patterns(self, session: aiohttp.ClientSession) -> None:
        url = f"{self._iae_url}/patterns/confirmed"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    log.warning("IAESignalBridge: GET %s returned %d", url, resp.status)
                    return
                data = await resp.json()
        except aiohttp.ClientError as exc:
            log.warning("IAESignalBridge: fetch failed: %s", exc)
            return

        patterns: list[dict] = data if isinstance(data, list) else data.get("patterns", [])
        log.info("IAESignalBridge: received %d confirmed patterns", len(patterns))

        deployed_count = 0
        for pattern in patterns:
            if not self._validate_pattern_schema(pattern):
                continue
            pid = str(pattern.get("pattern_id", ""))
            if pid in self._seen_pattern_ids:
                continue
            spec = self._translator.translate_pattern_to_signal(pattern)
            if spec is None:
                continue
            deployed = await asyncio.get_event_loop().run_in_executor(
                None, self.deploy_signal, spec
            )
            if deployed:
                self._seen_pattern_ids.add(pid)
                deployed_count += 1
                await self._notify_coordination(session, spec)

        if deployed_count:
            log.info("IAESignalBridge: deployed %d new signal(s)", deployed_count)

    @staticmethod
    def _validate_pattern_schema(pattern: dict) -> bool:
        missing = _REQUIRED_PATTERN_FIELDS - set(pattern.keys())
        if missing:
            log.warning("IAESignalBridge: pattern missing fields %s -- skipping", missing)
            return False
        confidence = float(pattern.get("confidence", 0.0))
        if confidence < 0.50:
            log.debug(
                "IAESignalBridge: pattern %s confidence %.2f < 0.50 -- skipping",
                pattern.get("pattern_id"),
                confidence,
            )
            return False
        return True

    def deploy_signal(self, spec: SignalSpec) -> bool:
        """
        Add a SignalSpec to the active signal set in SQLite.

        Returns True if successfully deployed, False otherwise.
        """
        errors = validate_signal_spec(spec)
        if errors:
            log.warning(
                "IAESignalBridge: spec '%s' invalid -- %s", spec.name, errors
            )
            return False

        if self._registry.signal_exists(spec.name):
            log.debug("IAESignalBridge: signal '%s' already exists -- skipping", spec.name)
            return False

        try:
            row_id = self._registry.insert_signal(spec)
            log.info(
                "IAESignalBridge: deployed signal '%s' (id=%d) -- regime=%s min_ic=%.4f",
                spec.name,
                row_id,
                spec.required_regime,
                spec.min_ic,
            )
            return True
        except sqlite3.IntegrityError:
            log.debug("IAESignalBridge: signal '%s' already exists (race) -- skipping", spec.name)
            return False
        except Exception as exc:
            log.error("IAESignalBridge: deploy_signal failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Signal decay monitoring
    # ------------------------------------------------------------------

    async def _decay_monitor_loop(self) -> None:
        """
        Every 30 minutes, check active signals for ICIR decay and retire
        any that fall below the threshold.
        """
        decay_check_interval = 1800.0  # 30 minutes
        while self._running:
            await asyncio.sleep(decay_check_interval)
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.retire_stale_signals
                )
            except Exception as exc:
                log.error("IAESignalBridge: decay monitor error: %s", exc)

    def retire_stale_signals(self) -> list[str]:
        """
        Poll signal decay state and retire signals with ICIR < _ICIR_RETIRE_THRESHOLD.

        Reads ICIR from the signal_decay_monitor if available, else uses
        stored current_icir from the registry.

        Returns list of retired signal names.
        """
        active = self._registry.get_active_signals()
        retired_names: list[str] = []

        for rec in active:
            icir = rec.current_icir
            if icir == 0.0:
                # Zero ICIR can mean not yet computed -- skip
                continue
            if icir < _ICIR_RETIRE_THRESHOLD:
                log.warning(
                    "IAESignalBridge: retiring signal '%s' -- ICIR=%.4f < threshold=%.2f",
                    rec.name,
                    icir,
                    _ICIR_RETIRE_THRESHOLD,
                )
                self._registry.retire_signal(rec.name)
                retired_names.append(rec.name)
            elif icir < _ICIR_PROBATION_THRESHOLD:
                log.info(
                    "IAESignalBridge: signal '%s' on probation -- ICIR=%.4f",
                    rec.name,
                    icir,
                )

        if retired_names:
            log.info(
                "IAESignalBridge: retired %d signal(s): %s",
                len(retired_names),
                retired_names,
            )
        return retired_names

    def update_signal_icir(self, name: str, icir: float) -> None:
        """Update the ICIR for a deployed signal (called by external decay monitor)."""
        self._registry.update_icir(name, icir)

    # ------------------------------------------------------------------
    # Coordination notifications
    # ------------------------------------------------------------------

    async def _notify_coordination(
        self, session: aiohttp.ClientSession, spec: SignalSpec
    ) -> None:
        url = f"{self._coordination_url}/signals/deployed"
        payload = {
            "event": "signal_deployed",
            "signal": spec.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status not in (200, 201, 204):
                    log.debug(
                        "IAESignalBridge: coordination notify %s returned %d", url, resp.status
                    )
        except aiohttp.ClientError as exc:
            log.debug("IAESignalBridge: coordination notify failed: %s", exc)

    # ------------------------------------------------------------------
    # Read-only public API
    # ------------------------------------------------------------------

    def get_active_signals(self) -> list[DeployedSignalRecord]:
        return self._registry.get_active_signals()

    def get_all_signals(self) -> list[DeployedSignalRecord]:
        return self._registry.get_all_signals()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    bridge = IAESignalBridge()
    try:
        await bridge.run()
    except KeyboardInterrupt:
        bridge.stop()


if __name__ == "__main__":
    asyncio.run(_main())
