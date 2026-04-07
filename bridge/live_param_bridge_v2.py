"""
bridge/live_param_bridge_v2.py
==============================
Enhanced bi-directional parameter bridge for the SRFM live trading system.

Architecture
------------
  LiveParamBridgeV2
    - Polls :8781/params/current every poll_interval_secs (default 60)
    - On change: validates -> writes config/live_params.json atomically
    - Diffs params and logs what changed, by how much, from which source
    - Staleness watchdog: if no response for 5 minutes -> warn + revert
    - Rollback subscriber: SSE stream :8781/events?topic=rollback_triggered

  PerformanceReporter
    - Reads execution/live_trades.db every 15 minutes
    - Computes: equity_change, realized_pnl, sharpe_4h, win_rate_4h
    - POSTs metrics to :8781/performance/update and :9091/metrics/performance

  ConfigFileWriter
    - Thread-safe atomic write to config/live_params.json
    - Format matches live_trader_alpaca.py hot-reload contract

Usage::

    import asyncio
    from bridge.live_param_bridge_v2 import LiveParamBridgeV2, ParamBridgeConfig

    cfg = ParamBridgeConfig(
        param_file_path="config/live_params.json",
        coordination_url="http://localhost:8781",
        iae_url="http://localhost:8780",
    )
    bridge = LiveParamBridgeV2(cfg)
    asyncio.run(bridge.run())
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Deque, Optional

import aiohttp

log = logging.getLogger("bridge.live_param_bridge_v2")

_REPO_ROOT = Path(__file__).parents[1]
_DEFAULT_PARAM_FILE = _REPO_ROOT / "config" / "live_params.json"
_DB_PATH = _REPO_ROOT / "execution" / "live_trades.db"
_LOG_DIR = _REPO_ROOT / "logs"

# Coordination and observability endpoints
_COORDINATION_BASE = "http://localhost:8781"
_OBSERVABILITY_BASE = "http://localhost:9091"
_IAE_BASE = "http://localhost:8780"

# Staleness threshold -- if no response for this many seconds, warn and revert
_STALE_THRESHOLD_SECS = 300  # 5 minutes

# Param constraints (key -> (min, max)) -- kept in sync with live_param_bridge.py
_PARAM_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "CF_BULL_THRESH":           (0.1,   5.0),
    "CF_BEAR_THRESH":           (0.1,   5.0),
    "BH_FORM":                  (1.0,   4.0),
    "BH_MASS_EXTREME":          (2.0,   8.0),
    "MIN_HOLD_BARS":            (1,     200),
    "MAX_HOLD_BARS":            (1,     2000),
    "min_hold_bars":            (1,     200),
    "max_hold_bars":            (1,     2000),
    "min_correlation":          (-1.0,  1.0),
    "max_correlation":          (-1.0,  1.0),
    "position_size_pct":        (0.001, 0.5),
    "stop_loss_pct":            (0.001, 0.5),
    "take_profit_pct":          (0.001, 2.0),
    "regime_filter_threshold":  (0.0,   1.0),
    "GARCH_CLIP":               (0.5,   5.0),
    "OU_SPEED":                 (0.01,  2.0),
    "OU_VOL_MULT":              (0.1,   5.0),
    "KELLY_FRAC":               (0.01,  1.0),
    "MAX_POS_PCT":              (0.001, 0.5),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParamBridgeConfig:
    """Configuration for LiveParamBridgeV2."""

    param_file_path: str | Path = _DEFAULT_PARAM_FILE
    coordination_url: str = _COORDINATION_BASE
    iae_url: str = _IAE_BASE
    observability_url: str = _OBSERVABILITY_BASE
    poll_interval_secs: float = 60.0
    perf_report_interval_secs: float = 900.0       # 15 minutes
    stale_threshold_secs: float = _STALE_THRESHOLD_SECS
    db_path: str | Path = _DB_PATH
    log_dir: str | Path = _LOG_DIR
    # If True, actually write signal to live_trader reload socket
    signal_reload: bool = True
    # Path to live_trader.pid
    pid_file: str | Path = _REPO_ROOT / "tools" / "live_trader.pid"


@dataclass
class ParamDiff:
    """Record of a single parameter change."""

    key: str
    old_value: Any
    new_value: Any
    source: str
    timestamp: str

    def pct_change(self) -> float | None:
        """Return percent change if both values are numeric, else None."""
        try:
            old = float(self.old_value)
            new = float(self.new_value)
            if old == 0:
                return None
            return (new - old) / abs(old) * 100.0
        except (TypeError, ValueError):
            return None

    def to_dict(self) -> dict:
        pct = self.pct_change()
        return {
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "pct_change": round(pct, 4) if pct is not None else None,
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass
class PerformanceSnapshot:
    """4-hour performance window computed from live_trades.db."""

    window_start: str
    window_end: str
    trade_count: int
    equity_change: float
    realized_pnl: float
    sharpe_4h: float
    win_rate_4h: float
    avg_pnl_per_trade: float
    max_drawdown: float
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "window_start":       self.window_start,
            "window_end":         self.window_end,
            "trade_count":        self.trade_count,
            "equity_change":      round(self.equity_change, 6),
            "realized_pnl":       round(self.realized_pnl, 4),
            "sharpe_4h":          round(self.sharpe_4h, 6),
            "win_rate_4h":        round(self.win_rate_4h, 4),
            "avg_pnl_per_trade":  round(self.avg_pnl_per_trade, 4),
            "max_drawdown":       round(self.max_drawdown, 6),
            "computed_at":        self.computed_at,
            "strategy":           "larsa_v18",
        }


# ---------------------------------------------------------------------------
# ConfigFileWriter -- thread-safe atomic writes
# ---------------------------------------------------------------------------

class ConfigFileWriter:
    """
    Thread-safe writer for config/live_params.json.

    Writes atomically: serialise to a temp file in the same directory,
    then os.replace() -- which is guaranteed atomic on POSIX and
    near-atomic on Windows (single kernel call since Vista).

    Format produced::

        {
          "version": 7,
          "applied_at": "2026-04-06T12:00:00+00:00",
          "source": "coordination",
          "hypothesis_id": "",
          "params": { ... }
        }

    The live trader reads this schema on SIGUSR1 or its reload-socket signal.
    """

    def __init__(self, param_file: Path) -> None:
        self._path = param_file
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(
        self,
        params: dict[str, Any],
        version: int,
        source: str,
        hypothesis_id: str = "",
    ) -> bool:
        """
        Atomically write params to the config file.

        Returns True on success, False on IO error.
        """
        payload: dict[str, Any] = {
            "version":      version,
            "applied_at":   datetime.now(timezone.utc).isoformat(),
            "source":       source,
            "hypothesis_id": hypothesis_id,
            "strategy":     "larsa_v18",
            "params":       params,
        }
        with self._lock:
            return self._atomic_write(payload)

    def read(self) -> dict[str, Any]:
        """Read and return current config, or empty dict on error."""
        with self._lock:
            if not self._path.exists():
                return {}
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("ConfigFileWriter.read failed: %s", exc)
                return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _atomic_write(self, data: dict[str, Any]) -> bool:
        """Write to a sibling .tmp file then rename."""
        tmp_path = self._path.with_suffix(".json.tmp")
        try:
            serialised = json.dumps(data, indent=2, ensure_ascii=False)
            # Write to tmp in same directory -- ensures same filesystem
            tmp_path.write_text(serialised, encoding="utf-8")
            # os.replace is atomic on POSIX; on Windows it overwrites atomically
            os.replace(str(tmp_path), str(self._path))
            log.debug("ConfigFileWriter: wrote version %d to %s", data.get("version", -1), self._path)
            return True
        except Exception as exc:
            log.error("ConfigFileWriter: atomic write failed: %s", exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return False


# ---------------------------------------------------------------------------
# PerformanceReporter
# ---------------------------------------------------------------------------

class PerformanceReporter:
    """
    Reads live_trades.db, computes 4-hour performance metrics, and
    POSTs them to the coordination layer and observability dashboard.

    Metrics computed
    ----------------
    - equity_change    : total PnL over the last 4 hours
    - realized_pnl     : sum of closed trade PnL
    - sharpe_4h        : annualised Sharpe using 15-minute return bars
    - win_rate_4h      : fraction of winning trades
    - avg_pnl_per_trade: mean PnL per closed trade
    - max_drawdown     : running max drawdown over 4h window
    """

    def __init__(self, config: ParamBridgeConfig) -> None:
        self._cfg = config
        self._db_path = Path(config.db_path)
        self._coordination_url = config.coordination_url
        self._observability_url = config.observability_url
        self._last_snapshot: PerformanceSnapshot | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def report_once(self, session: aiohttp.ClientSession) -> PerformanceSnapshot | None:
        """Compute metrics and POST them. Returns snapshot or None on error."""
        snap = await asyncio.get_event_loop().run_in_executor(None, self._compute_snapshot)
        if snap is None:
            return None
        self._last_snapshot = snap
        await self._post_to_coordination(session, snap)
        await self._post_to_observability(session, snap)
        return snap

    # ------------------------------------------------------------------
    # DB computation (runs in thread executor to avoid blocking)
    # ------------------------------------------------------------------

    def _compute_snapshot(self) -> PerformanceSnapshot | None:
        if not self._db_path.exists():
            log.warning("PerformanceReporter: DB not found at %s", self._db_path)
            return None
        try:
            return self._query_and_compute()
        except Exception as exc:
            log.error("PerformanceReporter: compute failed: %s", exc)
            return None

    def _query_and_compute(self) -> PerformanceSnapshot | None:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=4)
        cutoff_iso = cutoff.isoformat()

        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM trades WHERE exit_time > ? ORDER BY exit_time ASC",
                (cutoff_iso,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            log.debug("PerformanceReporter: no trades in last 4h window")
            return PerformanceSnapshot(
                window_start=cutoff_iso,
                window_end=now.isoformat(),
                trade_count=0,
                equity_change=0.0,
                realized_pnl=0.0,
                sharpe_4h=0.0,
                win_rate_4h=0.0,
                avg_pnl_per_trade=0.0,
                max_drawdown=0.0,
            )

        pnls: list[float] = []
        for row in rows:
            # Try multiple column name variants used by different DB versions
            pnl_val = (
                _row_get(row, "pnl")
                or _row_get(row, "realized_pnl")
                or _row_get(row, "net_pnl")
                or 0.0
            )
            try:
                pnls.append(float(pnl_val))
            except (TypeError, ValueError):
                pnls.append(0.0)

        realized_pnl = sum(pnls)
        trade_count = len(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / trade_count if trade_count else 0.0
        avg_pnl = realized_pnl / trade_count if trade_count else 0.0

        # Sharpe: annualise 15-min returns
        # Approximate: 4h = 16 bars of 15m, annualisation ~ sqrt(16 * 365 * 24 / 4)
        sharpe = self._compute_sharpe(pnls)
        max_dd = self._compute_max_drawdown(pnls)

        return PerformanceSnapshot(
            window_start=cutoff_iso,
            window_end=now.isoformat(),
            trade_count=trade_count,
            equity_change=realized_pnl,
            realized_pnl=realized_pnl,
            sharpe_4h=sharpe,
            win_rate_4h=round(win_rate, 4),
            avg_pnl_per_trade=round(avg_pnl, 4),
            max_drawdown=round(max_dd, 6),
        )

    @staticmethod
    def _compute_sharpe(pnls: list[float]) -> float:
        """Annualised Sharpe ratio from per-trade PnL series."""
        if len(pnls) < 2:
            return 0.0
        n = len(pnls)
        mean = sum(pnls) / n
        variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0.0:
            return 0.0
        # Annualisation: 4h window -> scale by sqrt(365 * 6) (6 four-hour periods/day)
        annualisation = math.sqrt(365.0 * 6.0)
        return (mean / std) * annualisation

    @staticmethod
    def _compute_max_drawdown(pnls: list[float]) -> float:
        """Peak-to-trough drawdown on cumulative PnL series."""
        if not pnls:
            return 0.0
        equity: list[float] = []
        running = 0.0
        for p in pnls:
            running += p
            equity.append(running)
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / abs(peak) if peak != 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    # ------------------------------------------------------------------
    # HTTP posting
    # ------------------------------------------------------------------

    async def _post_to_coordination(
        self, session: aiohttp.ClientSession, snap: PerformanceSnapshot
    ) -> None:
        url = f"{self._coordination_url}/performance/update"
        try:
            async with session.post(
                url, json=snap.to_dict(), timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status not in (200, 201, 204):
                    log.warning(
                        "PerformanceReporter: coordination POST %s returned %d", url, resp.status
                    )
                else:
                    log.info(
                        "PerformanceReporter: posted to coordination -- sharpe=%.4f win_rate=%.2f",
                        snap.sharpe_4h,
                        snap.win_rate_4h,
                    )
        except aiohttp.ClientError as exc:
            log.warning("PerformanceReporter: coordination POST failed: %s", exc)

    async def _post_to_observability(
        self, session: aiohttp.ClientSession, snap: PerformanceSnapshot
    ) -> None:
        url = f"{self._observability_url}/metrics/performance"
        try:
            async with session.post(
                url, json=snap.to_dict(), timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status not in (200, 201, 204):
                    log.debug(
                        "PerformanceReporter: observability POST %s returned %d", url, resp.status
                    )
        except aiohttp.ClientError as exc:
            log.debug("PerformanceReporter: observability POST failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Param validation
# ---------------------------------------------------------------------------

def validate_params(params: dict[str, Any]) -> list[str]:
    """
    Validate param values against known constraints.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []
    for key, value in params.items():
        if key in _PARAM_CONSTRAINTS:
            lo, hi = _PARAM_CONSTRAINTS[key]
            try:
                fv = float(value)
                if not (lo <= fv <= hi):
                    errors.append(f"{key}={fv} out of range [{lo}, {hi}]")
            except (TypeError, ValueError):
                errors.append(f"{key}: non-numeric value '{value}'")

    # Cross-param sanity checks
    _check_pair(params, errors, "MIN_HOLD_BARS", "MAX_HOLD_BARS", operator="lt")
    _check_pair(params, errors, "min_hold_bars", "max_hold_bars", operator="lt")
    _check_pair(params, errors, "min_correlation", "max_correlation", operator="lte")
    _check_pair(params, errors, "CF_BULL_THRESH", "CF_BEAR_THRESH", operator="lte")
    _check_pair(params, errors, "BH_FORM", "BH_MASS_EXTREME", operator="lt")

    return errors


def _check_pair(
    params: dict,
    errors: list[str],
    key_a: str,
    key_b: str,
    operator: str,
) -> None:
    if key_a not in params or key_b not in params:
        return
    try:
        a = float(params[key_a])
        b = float(params[key_b])
        if operator == "lt" and not (a < b):
            errors.append(f"{key_a} must be < {key_b} (got {a} vs {b})")
        elif operator == "lte" and not (a <= b):
            errors.append(f"{key_a} must be <= {key_b} (got {a} vs {b})")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------

def compute_diff(
    old_params: dict[str, Any],
    new_params: dict[str, Any],
    source: str,
) -> list[ParamDiff]:
    """Return list of ParamDiff for every key that changed."""
    ts = datetime.now(timezone.utc).isoformat()
    diffs: list[ParamDiff] = []
    all_keys = set(old_params) | set(new_params)
    for key in sorted(all_keys):
        old_val = old_params.get(key)
        new_val = new_params.get(key)
        if old_val != new_val:
            diffs.append(ParamDiff(
                key=key,
                old_value=old_val,
                new_value=new_val,
                source=source,
                timestamp=ts,
            ))
    return diffs


def log_diff(diffs: list[ParamDiff], logger: logging.Logger = log) -> None:
    """Log each param diff at INFO level."""
    if not diffs:
        return
    logger.info("ParamBridge: %d param(s) changed:", len(diffs))
    for d in diffs:
        pct = d.pct_change()
        pct_str = f" ({pct:+.2f}%)" if pct is not None else ""
        logger.info(
            "  %s: %s -> %s%s  [source=%s]",
            d.key,
            d.old_value,
            d.new_value,
            pct_str,
            d.source,
        )


# ---------------------------------------------------------------------------
# DiffLogger -- persists diffs to a JSONL file
# ---------------------------------------------------------------------------

class DiffLogger:
    """Appends ParamDiff records to a JSONL file for audit trail."""

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / "param_diffs.jsonl"
        self._lock = threading.Lock()

    def append(self, diffs: list[ParamDiff]) -> None:
        if not diffs:
            return
        lines = [json.dumps(d.to_dict(), ensure_ascii=False) for d in diffs]
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
            except Exception as exc:
                log.warning("DiffLogger: write failed: %s", exc)


# ---------------------------------------------------------------------------
# LiveParamBridgeV2
# ---------------------------------------------------------------------------

class LiveParamBridgeV2:
    """
    Bi-directional parameter bridge between the SRFM coordination layer
    (:8781) and the live trader config file.

    Lifecycle
    ---------
    1. poll :8781/params/current every poll_interval_secs
    2. On change -> validate -> write config/live_params.json -> signal trader
    3. Every perf_report_interval_secs -> PerformanceReporter -> POST metrics
    4. Background task: subscribe to SSE for rollback events
    5. Staleness watchdog: if :8781 silent for 5 minutes -> revert + warn

    Thread model
    ------------
    All I/O is async (aiohttp).  The ConfigFileWriter uses a threading.Lock
    for the atomic write so it is safe to call from both async and sync code.
    """

    def __init__(self, config: ParamBridgeConfig | None = None) -> None:
        self._cfg = config or ParamBridgeConfig()
        self._param_file = Path(self._cfg.param_file_path)
        self._writer = ConfigFileWriter(self._param_file)
        self._reporter = PerformanceReporter(self._cfg)
        self._diff_logger = DiffLogger(Path(self._cfg.log_dir))

        self._current_params: dict[str, Any] = {}
        self._last_good_params: dict[str, Any] = {}
        self._current_version: int = 0
        self._last_response_time: float = time.monotonic()
        self._running = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run all bridge tasks concurrently until stopped."""
        self._running = True
        # Load existing params as baseline
        existing = self._writer.read()
        if existing:
            self._current_params = existing.get("params", {})
            self._last_good_params = dict(self._current_params)
            self._current_version = int(existing.get("version", 0))
            log.info(
                "LiveParamBridgeV2: loaded existing params (v%d) from %s",
                self._current_version,
                self._param_file,
            )

        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                asyncio.create_task(self._poll_params_loop(session), name="param_poll"),
                asyncio.create_task(self._perf_report_loop(session), name="perf_report"),
                asyncio.create_task(self._staleness_watchdog(), name="staleness_watchdog"),
                asyncio.create_task(self.watch_for_rollback(session), name="rollback_watch"),
            ]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self._running = False
                log.info("LiveParamBridgeV2: stopped.")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Param polling loop
    # ------------------------------------------------------------------

    async def _poll_params_loop(self, session: aiohttp.ClientSession) -> None:
        log.info(
            "LiveParamBridgeV2: polling %s/params/current every %.0fs",
            self._cfg.coordination_url,
            self._cfg.poll_interval_secs,
        )
        while self._running:
            await self._fetch_and_apply(session)
            await asyncio.sleep(self._cfg.poll_interval_secs)

    async def _fetch_and_apply(self, session: aiohttp.ClientSession) -> None:
        url = f"{self._cfg.coordination_url}/params/current"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    log.warning("LiveParamBridgeV2: GET %s returned %d", url, resp.status)
                    return
                data = await resp.json()
                self._last_response_time = time.monotonic()
        except aiohttp.ClientError as exc:
            log.warning("LiveParamBridgeV2: fetch failed: %s", exc)
            return
        except Exception as exc:
            log.error("LiveParamBridgeV2: unexpected fetch error: %s", exc)
            return

        await self._apply_params(data)

    async def _apply_params(self, data: dict[str, Any]) -> None:
        """Validate, diff, write, and signal if params changed."""
        remote_version = int(data.get("version", 0))
        remote_params: dict[str, Any] = data.get("params", data)  # flat or nested
        source: str = str(data.get("source", "coordination"))
        hypothesis_id: str = str(data.get("hypothesis_id", ""))

        # No change -- skip
        if remote_version > 0 and remote_version <= self._current_version:
            return
        if remote_params == self._current_params and remote_version == self._current_version:
            return

        errors = validate_params(remote_params)
        if errors:
            log.warning(
                "LiveParamBridgeV2: rejecting params v%d -- validation errors: %s",
                remote_version,
                errors,
            )
            return

        diffs = compute_diff(self._current_params, remote_params, source)
        if not diffs and remote_version == self._current_version:
            return

        log_diff(diffs)
        self._diff_logger.append(diffs)

        new_version = max(remote_version, self._current_version + 1)
        ok = self._writer.write(
            remote_params,
            version=new_version,
            source=source,
            hypothesis_id=hypothesis_id,
        )
        if ok:
            self._last_good_params = dict(self._current_params)
            self._current_params = remote_params
            self._current_version = new_version
            log.info(
                "LiveParamBridgeV2: applied v%d from %s (%d param(s) changed)",
                new_version,
                source,
                len(diffs),
            )
            if self._cfg.signal_reload:
                self._signal_trader()
        else:
            log.error("LiveParamBridgeV2: config write failed -- params NOT updated")

    # ------------------------------------------------------------------
    # Performance reporting loop
    # ------------------------------------------------------------------

    async def _perf_report_loop(self, session: aiohttp.ClientSession) -> None:
        log.info(
            "LiveParamBridgeV2: performance reporter every %.0fs",
            self._cfg.perf_report_interval_secs,
        )
        # Stagger startup -- wait one interval before first report
        await asyncio.sleep(self._cfg.perf_report_interval_secs)
        while self._running:
            try:
                snap = await self._reporter.report_once(session)
                if snap:
                    log.info(
                        "LiveParamBridgeV2: perf snapshot -- trades=%d sharpe=%.4f win_rate=%.2f pnl=%.2f",
                        snap.trade_count,
                        snap.sharpe_4h,
                        snap.win_rate_4h,
                        snap.realized_pnl,
                    )
            except Exception as exc:
                log.error("LiveParamBridgeV2: perf report failed: %s", exc)
            await asyncio.sleep(self._cfg.perf_report_interval_secs)

    # ------------------------------------------------------------------
    # Staleness watchdog
    # ------------------------------------------------------------------

    async def _staleness_watchdog(self) -> None:
        """
        Check every 60 seconds whether we have heard from the coordination layer
        within the last stale_threshold_secs.  If not, log a warning and revert
        to last-known-good params.
        """
        check_interval = 60.0
        while self._running:
            await asyncio.sleep(check_interval)
            elapsed = time.monotonic() - self._last_response_time
            if elapsed > self._cfg.stale_threshold_secs:
                log.warning(
                    "LiveParamBridgeV2: coordination layer STALE -- no response for %.0fs (threshold=%.0fs)",
                    elapsed,
                    self._cfg.stale_threshold_secs,
                )
                self._write_stale_warning(elapsed)
                if self._last_good_params:
                    log.warning(
                        "LiveParamBridgeV2: reverting to last-known-good params"
                    )
                    self._writer.write(
                        self._last_good_params,
                        version=self._current_version,
                        source="stale_revert",
                    )
                    if self._cfg.signal_reload:
                        self._signal_trader()

    def _write_stale_warning(self, elapsed: float) -> None:
        """Append a stale-alert entry to the log directory."""
        log_dir = Path(self._cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        alert_path = log_dir / "stale_alerts.jsonl"
        entry = json.dumps({
            "alert_type": "stale_coordination",
            "elapsed_secs": round(elapsed, 1),
            "threshold_secs": self._cfg.stale_threshold_secs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last_good_param_keys": list(self._last_good_params.keys()),
        })
        try:
            with alert_path.open("a", encoding="utf-8") as fh:
                fh.write(entry + "\n")
        except Exception as exc:
            log.debug("LiveParamBridgeV2: stale alert write failed: %s", exc)

    # ------------------------------------------------------------------
    # Rollback subscriber (SSE)
    # ------------------------------------------------------------------

    async def watch_for_rollback(self, session: aiohttp.ClientSession) -> None:
        """
        Subscribe to :8781/events?topic=rollback_triggered via SSE.

        On receiving a rollback event, log the event, send an alert, and
        apply the rollback parameters if provided in the event payload.
        """
        url = f"{self._cfg.coordination_url}/events?topic=rollback_triggered"
        log.info("LiveParamBridgeV2: subscribing to rollback events at %s", url)

        backoff = 5.0
        while self._running:
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=None, connect=10),
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    if resp.status != 200:
                        log.warning(
                            "LiveParamBridgeV2: SSE endpoint %s returned %d -- retrying in %.0fs",
                            url,
                            resp.status,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 120.0)
                        continue

                    backoff = 5.0  # reset on successful connect
                    log.info("LiveParamBridgeV2: SSE stream connected")
                    buffer: list[str] = []

                    async for line_bytes in resp.content:
                        if not self._running:
                            break
                        line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                        if line.startswith("data:"):
                            buffer.append(line[5:].strip())
                        elif line == "" and buffer:
                            payload_str = "\n".join(buffer)
                            buffer = []
                            await self._handle_rollback_event(payload_str)

            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning(
                    "LiveParamBridgeV2: SSE stream error: %s -- reconnecting in %.0fs",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 120.0)

    async def _handle_rollback_event(self, payload_str: str) -> None:
        """Process a rollback_triggered SSE event."""
        try:
            event = json.loads(payload_str)
        except json.JSONDecodeError:
            log.warning(
                "LiveParamBridgeV2: rollback event parse failed -- raw: %s", payload_str[:200]
            )
            return

        log.warning(
            "LiveParamBridgeV2: ROLLBACK triggered -- reason=%s initiated_by=%s",
            event.get("reason", "unknown"),
            event.get("initiated_by", "unknown"),
        )

        # Log to rollback audit file
        log_dir = Path(self._cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        audit_path = log_dir / "rollback_events.jsonl"
        event_with_ts = {**event, "received_at": datetime.now(timezone.utc).isoformat()}
        try:
            with audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event_with_ts) + "\n")
        except Exception as exc:
            log.warning("LiveParamBridgeV2: rollback audit write failed: %s", exc)

        # If rollback params provided, apply them
        rollback_params = event.get("rollback_params") or event.get("params")
        if rollback_params and isinstance(rollback_params, dict):
            errors = validate_params(rollback_params)
            if not errors:
                log.warning("LiveParamBridgeV2: applying rollback params")
                self._writer.write(
                    rollback_params,
                    version=self._current_version + 1,
                    source="rollback",
                    hypothesis_id=str(event.get("hypothesis_id", "")),
                )
                self._current_version += 1
                self._current_params = rollback_params
                if self._cfg.signal_reload:
                    self._signal_trader()
            else:
                log.error(
                    "LiveParamBridgeV2: rollback params invalid -- %s", errors
                )
        elif self._last_good_params:
            # Fall back to last known good
            log.warning(
                "LiveParamBridgeV2: no rollback params in event -- reverting to last-known-good"
            )
            self._writer.write(
                self._last_good_params,
                version=self._current_version + 1,
                source="rollback_revert",
            )
            self._current_version += 1
            self._current_params = self._last_good_params
            if self._cfg.signal_reload:
                self._signal_trader()

    # ------------------------------------------------------------------
    # Trader reload signalling
    # ------------------------------------------------------------------

    def _signal_trader(self) -> None:
        """Send SIGUSR1 to the live trader process (Unix) or skip on Windows."""
        pid_file = Path(self._cfg.pid_file)
        if not pid_file.exists():
            return
        try:
            pid = int(pid_file.read_text().strip())
        except Exception:
            return
        try:
            import signal as _sig
            os.kill(pid, _sig.SIGUSR1)
            log.info("LiveParamBridgeV2: sent SIGUSR1 to PID %d", pid)
        except (AttributeError, OSError):
            log.debug("LiveParamBridgeV2: SIGUSR1 not available (Windows) or PID %d gone", pid)

    # ------------------------------------------------------------------
    # Read-only access for external callers
    # ------------------------------------------------------------------

    def get_current_params(self) -> dict[str, Any]:
        return dict(self._current_params)

    def get_current_version(self) -> int:
        return self._current_version

    def get_last_response_time(self) -> float:
        return self._last_response_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_get(row: sqlite3.Row, key: str) -> Any:
    """Safely get a column value from a sqlite3.Row, return None if absent."""
    try:
        return row[key]
    except IndexError:
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    cfg = ParamBridgeConfig()
    bridge = LiveParamBridgeV2(cfg)
    try:
        await bridge.run()
    except KeyboardInterrupt:
        bridge.stop()


if __name__ == "__main__":
    asyncio.run(_main())
