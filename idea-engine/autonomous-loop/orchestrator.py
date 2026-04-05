"""
idea-engine/autonomous-loop/orchestrator.py

AutonomousOrchestrator: the main loop that runs everything.

Runs continuously, observing live performance, generating hypotheses,
debating them, backtesting winners, and automatically applying validated
improvements to the live strategy. No human required.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .signal_collector import SignalCollector, SystemSignal
from .pattern_miner import PatternMiner
from .hypothesis_queue import HypothesisQueue, HypothesisStage
from .backtest_bridge import BacktestBridge
from .parameter_applier import ParameterApplier
from .loop_monitor import LoopMonitor
from .reporting.cycle_report import CycleReporter
from .reporting.performance_attribution import PerformanceAttributor

logger = logging.getLogger(__name__)

_DEFAULT_CYCLE_INTERVAL = 3600  # 1 hour
_CIRCUIT_BREAKER_THRESHOLD = 3
_STATE_DB = Path(__file__).parent / "loop_state.db"
_REPORTS_DIR = Path(__file__).parent / "reports"


class AutonomousOrchestrator:
    """
    Main autonomous loop controller.

    Runs in an infinite async loop. Each cycle:
      1. Collects external signals (macro, on-chain, sentiment, derivatives)
      2. Mines patterns from recent live trade data
      3. Generates hypotheses from discovered patterns
      4. Debates hypotheses via the LLM debate chamber
      5. Backtests approved hypotheses
      6. Promotes validated improvements to live parameters
      7. Monitors live performance vs expectations
      8. Detects Bayesian drift in live results
      9. Writes a cycle summary report

    Safety: circuit breaker halts loop after 3 consecutive cycle failures.
    State is persisted to SQLite so restarts resume correctly.
    """

    def __init__(
        self,
        cycle_interval: int = _DEFAULT_CYCLE_INTERVAL,
        db_path: Path | str | None = None,
        reports_dir: Path | str | None = None,
    ) -> None:
        self.cycle_interval = cycle_interval
        self.db_path = Path(db_path) if db_path else _STATE_DB
        self.reports_dir = Path(reports_dir) if reports_dir else _REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # subsystems
        self.signal_collector = SignalCollector()
        self.pattern_miner = PatternMiner()
        self.hypothesis_queue = HypothesisQueue(db_path=self.db_path)
        self.backtest_bridge = BacktestBridge()
        self.parameter_applier = ParameterApplier()
        self.loop_monitor = LoopMonitor(db_path=self.db_path)
        self.cycle_reporter = CycleReporter(reports_dir=self.reports_dir)
        self.perf_attributor = PerformanceAttributor(db_path=self.db_path)

        # runtime state
        self._consecutive_failures = 0
        self._cycle_number = 0
        self._running = False
        self._last_signal: SystemSignal | None = None

        self._init_state_db()
        self._restore_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Infinite async loop. Runs one cycle every `cycle_interval` seconds.
        Stops on KeyboardInterrupt or if the circuit breaker trips.
        """
        self._running = True
        logger.info(
            "AutonomousOrchestrator starting. cycle_interval=%ds", self.cycle_interval
        )

        try:
            while self._running:
                cycle_start = datetime.now(timezone.utc)
                self._cycle_number += 1

                logger.info("=== Autonomous cycle #%d starting ===", self._cycle_number)

                try:
                    await self._run_cycle(cycle_start)
                    self._consecutive_failures = 0
                    self._persist_state()
                except Exception as exc:  # pylint: disable=broad-except
                    self._consecutive_failures += 1
                    logger.error(
                        "Cycle #%d FAILED (%d/%d): %s",
                        self._cycle_number,
                        self._consecutive_failures,
                        _CIRCUIT_BREAKER_THRESHOLD,
                        exc,
                        exc_info=True,
                    )
                    if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                        await self._trip_circuit_breaker()
                        return

                elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                sleep_for = max(0.0, self.cycle_interval - elapsed)
                logger.info(
                    "Cycle #%d done in %.1fs. Sleeping %.1fs.",
                    self._cycle_number,
                    elapsed,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)

        except KeyboardInterrupt:
            logger.info("Orchestrator interrupted by user.")
        finally:
            self._running = False
            self._persist_state()
            logger.info("AutonomousOrchestrator stopped after %d cycles.", self._cycle_number)

    def stop(self) -> None:
        """Gracefully stop the loop after the current cycle finishes."""
        logger.info("Stop requested — will halt after current cycle.")
        self._running = False

    # ------------------------------------------------------------------
    # Internal cycle execution
    # ------------------------------------------------------------------

    async def _run_cycle(self, cycle_start: datetime) -> None:
        """Execute one full autonomous cycle and build a summary report."""
        ctx: dict[str, Any] = {
            "cycle_number": self._cycle_number,
            "cycle_start": cycle_start.isoformat(),
        }

        # 1. Collect external signals
        signal = await self.collect_signals()
        ctx["signal"] = signal
        self._last_signal = signal
        logger.info("Signals collected: composite=%.3f", signal.composite_score)

        # 2. Mine patterns from live trade data
        patterns = await self.mine_patterns()
        ctx["patterns_found"] = len(patterns)
        logger.info("Patterns mined: %d", len(patterns))

        # 3. Generate hypotheses from patterns
        new_hypotheses = await self.generate_hypotheses(patterns)
        ctx["hypotheses_generated"] = len(new_hypotheses)
        logger.info("Hypotheses generated: %d", len(new_hypotheses))

        # 4. Debate hypotheses
        approved = await self.debate_hypotheses(new_hypotheses)
        ctx["hypotheses_approved"] = len(approved)
        logger.info("Hypotheses approved by debate: %d", len(approved))

        # 5. Backtest approved hypotheses
        validated = await self.backtest_approved(approved)
        ctx["hypotheses_validated"] = len(validated)
        logger.info("Hypotheses validated by backtest: %d", len(validated))

        # 6. Promote validated to live
        applied = await self.promote_validated(validated)
        ctx["parameters_applied"] = len(applied)
        logger.info("Parameter sets applied to live: %d", len(applied))

        # 7. Monitor live performance
        live_health = await self.monitor_live()
        ctx["live_health"] = live_health
        logger.info("Live health check: %s", live_health.get("status", "unknown"))

        # 8. Detect drift
        drift_alerts = await self.detect_drift()
        ctx["drift_alerts"] = drift_alerts
        if drift_alerts:
            logger.warning("Drift alerts: %s", drift_alerts)

        # 9. Write cycle report
        await self.generate_report(ctx)

        # Update loop monitor
        self.loop_monitor.record_cycle(
            cycle_number=self._cycle_number,
            elapsed_seconds=(datetime.now(timezone.utc) - cycle_start).total_seconds(),
            patterns_found=len(patterns),
            hypotheses_generated=len(new_hypotheses),
            drift_alerts=len(drift_alerts),
        )

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    async def collect_signals(self) -> SystemSignal:
        """Run all IAE data sources in parallel and return composite signal."""
        return await self.signal_collector.collect()

    async def mine_patterns(self):
        """Run IAE pattern miners on latest live trade data."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.pattern_miner.mine
        )

    async def generate_hypotheses(self, patterns) -> list:
        """Convert mined patterns to hypotheses via generators."""
        if not patterns:
            return []
        return await asyncio.get_event_loop().run_in_executor(
            None, self.hypothesis_queue.ingest_patterns, patterns
        )

    async def debate_hypotheses(self, hypotheses: list) -> list:
        """Run debate chamber on new hypotheses. Returns approved list."""
        if not hypotheses:
            return []
        return await asyncio.get_event_loop().run_in_executor(
            None, self.hypothesis_queue.run_debates
        )

    async def backtest_approved(self, approved: list) -> list:
        """Queue approved hypotheses for backtesting. Returns validated."""
        if not approved:
            return []
        validated = []
        for hyp in approved:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.backtest_bridge.run_for_hypothesis, hyp
            )
            if result and result.passed_validation:
                validated.append((hyp, result))
                self.hypothesis_queue.mark_validated(hyp.hypothesis_id)
        return validated

    async def promote_validated(self, validated: list) -> list:
        """Apply validated parameters to the live strategy."""
        applied = []
        for hyp, result in validated:
            success = await asyncio.get_event_loop().run_in_executor(
                None, self.parameter_applier.apply, hyp, result
            )
            if success:
                applied.append(hyp.hypothesis_id)
                self.hypothesis_queue.mark_applied(hyp.hypothesis_id)
                self.perf_attributor.record_application(hyp, result)
        return applied

    async def monitor_live(self) -> dict[str, Any]:
        """Check live performance vs backtest expectations."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.loop_monitor.check_live_performance
        )

    async def detect_drift(self) -> list[str]:
        """Run Bayesian drift detection on live results."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.loop_monitor.detect_drift
        )

    async def generate_report(self, ctx: dict[str, Any]) -> None:
        """Write cycle summary to loop/reports/."""
        await asyncio.get_event_loop().run_in_executor(
            None, self.cycle_reporter.write, ctx
        )

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    async def _trip_circuit_breaker(self) -> None:
        """Halt the loop and write an alert file."""
        self._running = False
        alert = {
            "event": "circuit_breaker_tripped",
            "cycle_number": self._cycle_number,
            "consecutive_failures": self._consecutive_failures,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        alert_path = self.reports_dir / "circuit_breaker_alert.json"
        import json
        alert_path.write_text(json.dumps(alert, indent=2))
        logger.critical(
            "CIRCUIT BREAKER TRIPPED after %d consecutive failures. "
            "Alert written to %s. Autonomous loop halted.",
            self._consecutive_failures,
            alert_path,
        )

    # ------------------------------------------------------------------
    # State persistence (SQLite)
    # ------------------------------------------------------------------

    def _init_state_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS loop_state (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _persist_state(self) -> None:
        import json
        state = {
            "cycle_number": self._cycle_number,
            "consecutive_failures": self._consecutive_failures,
            "last_persisted": datetime.now(timezone.utc).isoformat(),
        }
        with sqlite3.connect(self.db_path) as conn:
            for k, v in state.items():
                conn.execute(
                    "INSERT OR REPLACE INTO loop_state (key, value) VALUES (?, ?)",
                    (k, json.dumps(v)),
                )
            conn.commit()

    def _restore_state(self) -> None:
        import json
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT key, value FROM loop_state").fetchall()
            state = {k: json.loads(v) for k, v in rows}
            self._cycle_number = int(state.get("cycle_number", 0))
            self._consecutive_failures = int(state.get("consecutive_failures", 0))
            logger.info(
                "Restored loop state: cycle_number=%d, consecutive_failures=%d",
                self._cycle_number,
                self._consecutive_failures,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not restore loop state: %s", exc)
