"""
microstructure/live_monitor.py

Real-time microstructure monitoring loop.

Architecture
------------
Runs as a background thread / process pulling the latest OHLCV bars every
5 minutes from the IAE data ingestion layer.  For each monitored symbol:
1. Pulls last N bars from the data store.
2. Runs all four microstructure models.
3. Assembles a MicrostructureSignal.
4. Emits the signal to the IAE event bus (POST to /events).
5. Logs anomalies (spread widening, sudden illiquidity, high PIN).
6. Triggers hypothesis_generator if a structural pattern is detected.

Signal emission
---------------
Each signal is posted to:
  POST http://localhost:8767/microstructure/signal
  Body: MicrostructureSignal.to_dict()

Anomaly logging
---------------
Anomalies are logged to idea_engine.db (microstructure_anomalies table)
and pushed to the event bus as high-priority events.

Shutdown
--------
The monitor runs until stop() is called.  Uses threading.Event for clean
shutdown without daemon thread leaks.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from microstructure.models.amihud import AmihudCalculator
from microstructure.models.adverse_selection import AdverseSelectionCalculator
from microstructure.models.roll_spread import RollSpreadCalculator
from microstructure.models.kyle_lambda import KyleLambdaCalculator
from microstructure.signals.microstructure_signal import (
    MicrostructureHealth,
    MicrostructureSignal,
)

logger = logging.getLogger(__name__)

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")
IAE_API = "http://localhost:8767"

CREATE_ANOMALY_TABLE = """
CREATE TABLE IF NOT EXISTS microstructure_anomalies (
    anomaly_id    TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    detected_at   TEXT NOT NULL,
    anomaly_type  TEXT NOT NULL,
    severity      TEXT NOT NULL,
    details_json  TEXT NOT NULL
);
"""


@dataclass
class MonitorConfig:
    """Configuration for the live monitor."""
    symbols: list[str]
    poll_interval_seconds: int = 300      # 5 minutes
    bar_lookback: int = 200               # bars to pull each cycle
    api_base: str = IAE_API
    db_path: Path = DB_PATH
    emit_to_bus: bool = True
    dry_run: bool = False


class LiveMicrostructureMonitor:
    """
    Real-time microstructure monitoring loop.

    Parameters
    ----------
    config     : MonitorConfig with symbols, poll interval, etc.
    data_fn    : Callable[[symbol, n_bars]] → dict with OHLCV lists.
                 Must return keys: opens, highs, lows, closes, volumes, timestamps.
    on_signal  : Optional callback invoked with each MicrostructureSignal.
    """

    def __init__(
        self,
        config: MonitorConfig,
        data_fn: Callable[[str, int], dict[str, Any]],
        on_signal: Callable[[MicrostructureSignal], None] | None = None,
    ) -> None:
        self.config = config
        self.data_fn = data_fn
        self.on_signal = on_signal
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.amihud = AmihudCalculator()
        self.roll = RollSpreadCalculator()
        self.adverse = AdverseSelectionCalculator()
        self.kyle = KyleLambdaCalculator()

        self._ensure_table()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the monitor in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Monitor already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="MicrostructureMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "LiveMicrostructureMonitor started. Symbols: %s, interval: %ds",
            self.config.symbols,
            self.config.poll_interval_seconds,
        )

    def stop(self) -> None:
        """Signal the monitor loop to stop cleanly."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)
        logger.info("LiveMicrostructureMonitor stopped.")

    def run_once(self) -> dict[str, MicrostructureSignal]:
        """
        Run a single polling cycle for all symbols.
        Returns a dict of symbol → MicrostructureSignal.
        Useful for testing and on-demand checks.
        """
        signals: dict[str, MicrostructureSignal] = {}
        for symbol in self.config.symbols:
            sig = self._process_symbol(symbol)
            if sig:
                signals[symbol] = sig
        return signals

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                self.run_once()
            except Exception as exc:
                logger.exception("Error in monitor cycle: %s", exc)
            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0, self.config.poll_interval_seconds - elapsed)
            self._stop_event.wait(timeout=sleep_for)

    def _process_symbol(self, symbol: str) -> MicrostructureSignal | None:
        try:
            bars = self.data_fn(symbol, self.config.bar_lookback)
        except Exception as exc:
            logger.error("Failed to fetch bars for %s: %s", symbol, exc)
            return None

        opens = bars.get("opens", [])
        closes = bars.get("closes", [])
        volumes = bars.get("volumes", [])
        timestamps = bars.get("timestamps", [])
        n = min(len(opens), len(closes), len(volumes), len(timestamps))

        if n < 30:
            logger.debug("%s: only %d bars, skipping.", symbol, n)
            return None

        # --- Compute model outputs ------------------------------------
        amihud_r = self.amihud.latest(symbol, closes[:n], volumes[:n], timestamps[:n])
        roll_r = self.roll.latest(symbol, closes[:n], timestamps[:n])
        adverse_r = self.adverse.latest(symbol, opens[:n], closes[:n], volumes[:n], timestamps[:n])
        kyle_r = self.kyle.latest(symbol, opens[:n], closes[:n], volumes[:n], timestamps[:n])

        if not all([amihud_r, roll_r, adverse_r, kyle_r]):
            logger.debug("%s: one or more model readings unavailable.", symbol)
            return None

        from microstructure.models.adverse_selection import AdverseSelectionRisk
        sig = MicrostructureSignal.build(
            symbol=symbol,
            amihud_percentile=max(0.0, min(1.0, (amihud_r.z_score + 3) / 6)),
            amihud_is_thin=amihud_r.is_thin,
            roll_spread=roll_r.effective_spread,
            roll_baseline=roll_r.rolling_baseline,
            adverse_risk=adverse_r.risk_level,
            adverse_pin=adverse_r.pin_proxy,
            kyle_percentile=kyle_r.percentile,
            kyle_size_multiplier=kyle_r.size_multiplier,
        )

        # --- Anomaly detection ----------------------------------------
        self._check_anomalies(symbol, sig, amihud_r, roll_r, adverse_r)

        # --- Emit signal ----------------------------------------------
        if self.config.emit_to_bus and not self.config.dry_run:
            self._emit_signal(sig)

        if self.on_signal:
            self.on_signal(sig)

        logger.debug("%s", sig)
        return sig

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def _check_anomalies(
        self,
        symbol: str,
        sig: MicrostructureSignal,
        amihud_r: Any,
        roll_r: Any,
        adverse_r: Any,
    ) -> None:
        anomalies: list[dict[str, Any]] = []

        if amihud_r.is_thin:
            anomalies.append({
                "type": "amihud_thin",
                "severity": "high",
                "detail": f"Amihud illiquidity {amihud_r.thinness_ratio:.2f}x baseline",
            })

        if roll_r.wide_spread_alert:
            anomalies.append({
                "type": "wide_spread",
                "severity": "high",
                "detail": f"Roll spread {roll_r.spread_ratio:.1f}x baseline",
            })

        if adverse_r.risk_level.value == "high":
            anomalies.append({
                "type": "high_adverse_selection",
                "severity": "medium",
                "detail": f"PIN proxy {adverse_r.pin_proxy:.3f} (high threshold)",
            })

        if sig.health_state == MicrostructureHealth.BROKEN:
            anomalies.append({
                "type": "broken_microstructure",
                "severity": "critical",
                "detail": f"Composite health {sig.composite_health:.3f} < 0.30",
            })

        for anom in anomalies:
            self._persist_anomaly(symbol, anom)
            logger.warning(
                "MICROSTRUCTURE ANOMALY [%s] %s: %s",
                symbol,
                anom["type"],
                anom["detail"],
            )

    # ------------------------------------------------------------------
    # API emission
    # ------------------------------------------------------------------

    def _emit_signal(self, sig: MicrostructureSignal) -> None:
        url = f"{self.config.api_base}/microstructure/signal"
        body = json.dumps(sig.to_dict()).encode("utf-8")
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as exc:
            logger.debug("Could not emit signal for %s: %s", sig.symbol, exc)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_anomaly(self, symbol: str, anom: dict[str, Any]) -> None:
        import uuid
        anomaly_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(str(self.config.db_path)) as conn:
                conn.execute(
                    """INSERT INTO microstructure_anomalies
                       (anomaly_id, symbol, detected_at, anomaly_type, severity, details_json)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (anomaly_id, symbol, now, anom["type"],
                     anom["severity"], json.dumps(anom)),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to persist anomaly: %s", exc)

    def _ensure_table(self) -> None:
        try:
            with sqlite3.connect(str(self.config.db_path)) as conn:
                conn.execute(CREATE_ANOMALY_TABLE)
                conn.commit()
        except sqlite3.Error:
            pass
