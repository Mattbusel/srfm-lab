"""
idea-engine/ingestion/pipeline.py
──────────────────────────────────
IngestionPipeline: orchestrates loading → mining → filtering → persistence.

Usage
─────
    from idea_engine.ingestion.pipeline import IngestionPipeline

    pipe = IngestionPipeline()
    result = pipe.run()

    # Custom config
    pipe = IngestionPipeline(config={
        "loaders":  {"backtest": True, "live_trades": True, "walk_forward": False},
        "miners":   {"time_of_day": True, "regime_cluster": False, ...},
        "filter":   {"alpha": 0.05, "min_effect": 0.20, "n_resamples": 1000},
    })
    result = pipe.run()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_N_RESAMPLES,
    DB_PATH,
    MIN_EFFECT_SIZE,
    PIPELINE_DEFAULT_CONFIG,
)
from .loaders.backtest_loader import load_backtest
from .loaders.trade_loader import load_live_trades
from .loaders.walk_forward_loader import load_walk_forward
from .miners.drawdown_miner import DrawdownMiner
from .miners.mass_physics_miner import MassPhysicsMiner
from .miners.regime_cluster_miner import RegimeClusterMiner
from .miners.time_of_day_miner import TimeOfDayMiner
from .statistical_filters.bootstrap_filter import enrich_effect_sizes, filter_patterns
from .types import (
    BacktestResult,
    LiveTradeData,
    MinedPattern,
    PatternStatus,
    WalkForwardResult,
)

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Summary of a pipeline run."""
    run_ts:             str                         = ""
    duration_s:         float                       = 0.0
    n_patterns_raw:     int                         = 0
    n_patterns_confirmed: int                       = 0
    n_patterns_written: int                         = 0
    errors:             List[str]                   = field(default_factory=list)
    warnings:           List[str]                   = field(default_factory=list)
    backtest_loaded:    bool                        = False
    live_trades_loaded: bool                        = False
    walk_forward_loaded: bool                       = False
    confirmed_patterns: List[MinedPattern]          = field(default_factory=list)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) idea_engine.db.  Runs schema if tables missing."""
    from ..db.init import create_db
    conn = create_db(db_path=db_path)
    return conn


def _write_patterns(conn: sqlite3.Connection, patterns: List[MinedPattern]) -> int:
    """Insert confirmed patterns into the patterns table.  Returns rows inserted."""
    if not patterns:
        return 0
    insert_sql = """
        INSERT INTO patterns (
            source, miner, pattern_type, label, description,
            feature_json, window_start, window_end, instruments,
            sample_size, p_value, effect_size, effect_size_type,
            win_rate, avg_pnl, avg_pnl_baseline, sharpe, max_dd, profit_factor,
            confidence, status, tags
        ) VALUES (
            :source, :miner, :pattern_type, :label, :description,
            :feature_json, :window_start, :window_end, :instruments,
            :sample_size, :p_value, :effect_size, :effect_size_type,
            :win_rate, :avg_pnl, :avg_pnl_baseline, :sharpe, :max_dd, :profit_factor,
            :confidence, :status, :tags
        )
    """
    n = 0
    for pat in patterns:
        row = pat.to_db_dict()
        try:
            conn.execute(insert_sql, row)
            n += 1
        except Exception as exc:
            logger.warning("Failed to insert pattern '%s': %s", pat.label, exc)
    conn.commit()
    return n


def _emit_event(
    conn: sqlite3.Connection,
    event_type: str,
    payload: Optional[Dict] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[int] = None,
    severity: str = "info",
) -> None:
    try:
        conn.execute(
            "INSERT INTO event_log (event_type, entity_type, entity_id, payload_json, severity) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_type, entity_type, entity_id, json.dumps(payload or {}), severity),
        )
        conn.commit()
    except Exception as exc:
        logger.debug("Could not emit event %s: %s", event_type, exc)


# ── rich progress (optional) ──────────────────────────────────────────────────

def _try_rich():
    try:
        from rich.progress import (
            Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn,
        )
        from rich.console import Console
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=Console(stderr=True),
        )
    except ImportError:
        return None


class _SimpleProgress:
    """Fallback progress bar when rich is not available."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def add_task(self, desc, total=1): return 0
    def update(self, task, advance=1, description=None):
        if description:
            logger.info("[progress] %s", description)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates the full ingestion pipeline.

    Stages
    ──────
    1. Load data   (backtest / live / walk-forward)
    2. Run miners  (time-of-day / regime-cluster / bh-physics / drawdown)
    3. Filter      (stationary bootstrap + BH correction)
    4. Persist     (write confirmed patterns to idea_engine.db)
    5. Emit events (event_log)
    """

    def __init__(
        self,
        config:  Optional[Dict[str, Any]] = None,
        db_path: Path = DB_PATH,
    ):
        # Deep merge config over defaults
        self.config  = {**PIPELINE_DEFAULT_CONFIG}
        if config:
            for section, values in config.items():
                if section in self.config and isinstance(values, dict):
                    self.config[section] = {**self.config[section], **values}
                else:
                    self.config[section] = values
        self.db_path = db_path

    # ── loaders ──────────────────────────────────────────────────────────────

    def _load_data(
        self,
        result: PipelineResult,
    ) -> tuple[Optional[BacktestResult], Optional[LiveTradeData], Optional[WalkForwardResult]]:
        loader_cfg = self.config.get("loaders", {})
        bt, live, wf = None, None, None

        if loader_cfg.get("backtest", True):
            try:
                bt = load_backtest()
                result.backtest_loaded = True
                logger.info("Backtest loaded: %d trades", len(bt.trades_df) if bt.trades_df is not None else 0)
            except Exception as exc:
                msg = f"Backtest loader failed: {exc}"
                result.errors.append(msg)
                logger.error(msg)

        if loader_cfg.get("live_trades", True):
            try:
                live = load_live_trades()
                result.live_trades_loaded = True
                logger.info("Live trades loaded: %d trades", len(live.trades) if live.trades is not None else 0)
            except Exception as exc:
                msg = f"Live trade loader failed: {exc}"
                result.errors.append(msg)
                logger.error(msg)

        if loader_cfg.get("walk_forward", True):
            try:
                wf = load_walk_forward()
                result.walk_forward_loaded = True
                logger.info("Walk-forward loaded: %d folds", len(wf.folds))
            except Exception as exc:
                msg = f"Walk-forward loader failed: {exc}"
                result.errors.append(msg)
                logger.error(msg)

        return bt, live, wf

    # ── miners ────────────────────────────────────────────────────────────────

    def _run_miners(
        self,
        bt:   Optional[BacktestResult],
        live: Optional[LiveTradeData],
        wf:   Optional[WalkForwardResult],
        result: PipelineResult,
    ) -> List[MinedPattern]:
        miner_cfg = self.config.get("miners", {})
        patterns: List[MinedPattern] = []

        # Time-of-day miner (uses both backtest and live trades)
        if miner_cfg.get("time_of_day", True):
            tod_miner = TimeOfDayMiner()
            for label, df in [
                ("backtest", bt.trades_df if bt else None),
                ("live",     live.trades  if live else None),
            ]:
                if df is not None and not df.empty and "pnl" in df.columns:
                    try:
                        pats = tod_miner.mine(df)
                        for p in pats:
                            p.source = label
                        patterns.extend(pats)
                        logger.info("TimeOfDayMiner[%s]: %d pattern(s)", label, len(pats))
                    except Exception as exc:
                        msg = f"TimeOfDayMiner[{label}] failed: {exc}"
                        result.warnings.append(msg)
                        logger.warning(msg)

        # Regime cluster miner (needs live data with regime_log)
        if miner_cfg.get("regime_cluster", True) and live is not None:
            try:
                pats = RegimeClusterMiner(source="live").mine(live)
                patterns.extend(pats)
                logger.info("RegimeClusterMiner: %d pattern(s)", len(pats))
            except Exception as exc:
                msg = f"RegimeClusterMiner failed: {exc}\n{traceback.format_exc()}"
                result.warnings.append(msg)
                logger.warning(msg)

        # BH physics miner (needs live data)
        if miner_cfg.get("bh_physics", True) and live is not None:
            try:
                pats = MassPhysicsMiner(source="live").mine(live)
                patterns.extend(pats)
                logger.info("MassPhysicsMiner: %d pattern(s)", len(pats))
            except Exception as exc:
                msg = f"MassPhysicsMiner failed: {exc}\n{traceback.format_exc()}"
                result.warnings.append(msg)
                logger.warning(msg)

        # Drawdown miner (needs live data with equity series)
        if miner_cfg.get("drawdown", True) and live is not None:
            try:
                pats = DrawdownMiner(source="live").mine(live)
                patterns.extend(pats)
                logger.info("DrawdownMiner: %d pattern(s)", len(pats))
            except Exception as exc:
                msg = f"DrawdownMiner failed: {exc}\n{traceback.format_exc()}"
                result.warnings.append(msg)
                logger.warning(msg)

        return patterns

    # ── filter ────────────────────────────────────────────────────────────────

    def _filter(
        self,
        patterns: List[MinedPattern],
    ) -> List[MinedPattern]:
        filter_cfg  = self.config.get("filter", {})
        alpha       = float(filter_cfg.get("alpha",       BOOTSTRAP_ALPHA))
        min_effect  = float(filter_cfg.get("min_effect",  MIN_EFFECT_SIZE))
        n_resamples = int(filter_cfg.get("n_resamples",   BOOTSTRAP_N_RESAMPLES))

        enrich_effect_sizes(patterns)
        return filter_patterns(patterns, alpha=alpha, min_effect=min_effect, n_resamples=n_resamples)

    # ── public: run ───────────────────────────────────────────────────────────

    def run(self) -> PipelineResult:
        """
        Execute the full pipeline.

        Returns
        -------
        PipelineResult with summary statistics and the confirmed patterns.
        """
        from datetime import datetime, timezone
        start_time = time.monotonic()
        run_ts     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        result = PipelineResult(run_ts=run_ts)

        progress = _try_rich()
        ctx      = progress if progress else _SimpleProgress()

        with ctx as prog:
            n_stages = 4
            task = prog.add_task("Ingestion pipeline", total=n_stages) if progress else 0

            # ── Stage 1: Load data ──────────────────────────────────────────
            if progress:
                prog.update(task, description="[1/4] Loading data …")
            logger.info("=== Stage 1: Loading data ===")
            bt, live, wf = self._load_data(result)
            if progress:
                prog.update(task, advance=1)

            # ── Stage 2: Mine patterns ──────────────────────────────────────
            if progress:
                prog.update(task, description="[2/4] Mining patterns …")
            logger.info("=== Stage 2: Mining patterns ===")
            all_patterns = self._run_miners(bt, live, wf, result)
            result.n_patterns_raw = len(all_patterns)
            logger.info("Total raw patterns: %d", result.n_patterns_raw)
            if progress:
                prog.update(task, advance=1)

            # ── Stage 3: Statistical filter ─────────────────────────────────
            if progress:
                prog.update(task, description="[3/4] Filtering (bootstrap) …")
            logger.info("=== Stage 3: Statistical filtering ===")
            if all_patterns:
                all_patterns = self._filter(all_patterns)
            confirmed = [p for p in all_patterns if p.status == PatternStatus.CONFIRMED]
            result.n_patterns_confirmed = len(confirmed)
            result.confirmed_patterns   = confirmed
            logger.info("Confirmed patterns: %d / %d", len(confirmed), len(all_patterns))
            if progress:
                prog.update(task, advance=1)

            # ── Stage 4: Persist to DB ──────────────────────────────────────
            if progress:
                prog.update(task, description="[4/4] Writing to database …")
            logger.info("=== Stage 4: Persisting to %s ===", self.db_path)
            try:
                conn = _ensure_db(self.db_path)
                n_written = _write_patterns(conn, confirmed)
                result.n_patterns_written = n_written

                _emit_event(
                    conn,
                    "pipeline_run",
                    payload={
                        "run_ts":          run_ts,
                        "n_raw":           result.n_patterns_raw,
                        "n_confirmed":     result.n_patterns_confirmed,
                        "n_written":       n_written,
                        "errors":          result.errors,
                        "warnings":        result.warnings[:5],  # trim
                        "loaders":         {
                            "backtest":     result.backtest_loaded,
                            "live_trades":  result.live_trades_loaded,
                            "walk_forward": result.walk_forward_loaded,
                        },
                    },
                    severity="info" if not result.errors else "warning",
                )
                conn.close()
                logger.info("Wrote %d patterns to DB", n_written)
            except Exception as exc:
                msg = f"DB write failed: {exc}"
                result.errors.append(msg)
                logger.error(msg)

            if progress:
                prog.update(task, advance=1)

        result.duration_s = time.monotonic() - start_time
        logger.info(
            "Pipeline complete in %.2fs | raw=%d confirmed=%d written=%d errors=%d",
            result.duration_s,
            result.n_patterns_raw,
            result.n_patterns_confirmed,
            result.n_patterns_written,
            len(result.errors),
        )
        return result


# ── CLI entry point ───────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="IAE Ingestion Pipeline — mine patterns from backtest + live trade history",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db", default=str(DB_PATH), metavar="PATH",
        help="Path to idea_engine.db (SQLite)",
    )
    parser.add_argument(
        "--live-db", default=None, metavar="PATH",
        help="Path to live_trades.db (overrides config default)",
    )
    parser.add_argument(
        "--miners", default="",
        help=(
            "Comma-separated list of miners to enable "
            "(time_of_day, regime_cluster, bh_physics, drawdown). "
            "Default: all enabled."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run loaders and miners but do not write patterns to the database",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
        stream=sys.stderr,
    )

    # Build miner config overrides
    miner_cfg: Dict[str, bool] = {}
    if args.miners:
        _all_miners = {"time_of_day", "regime_cluster", "bh_physics", "drawdown"}
        _requested = {m.strip() for m in args.miners.split(",") if m.strip()}
        _unknown = _requested - _all_miners
        if _unknown:
            parser.error(f"Unknown miners: {_unknown}. Valid: {sorted(_all_miners)}")
        miner_cfg = {m: (m in _requested) for m in _all_miners}

    # Build pipeline config
    pipeline_config: Dict[str, Any] = {}
    if miner_cfg:
        pipeline_config["miners"] = miner_cfg

    # Override live trades DB path in config if provided
    if args.live_db:
        from . import config as _cfg
        _cfg.LIVE_TRADES_DB = Path(args.live_db)
        logger.info("Live trades DB overridden: %s", args.live_db)

    db_path = Path(args.db)

    if args.verbose:
        logger.debug("DB path: %s", db_path)
        logger.debug("Miners config: %s", miner_cfg or "all (defaults)")
        logger.debug("Dry run: %s", args.dry_run)

    pipe = IngestionPipeline(config=pipeline_config or None, db_path=db_path)

    if args.dry_run:
        # Run all stages up to (but not including) persistence
        from datetime import datetime, timezone as _tz
        _start = time.monotonic()
        _run_ts = datetime.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = PipelineResult(run_ts=_run_ts)
        bt, live, wf = pipe._load_data(result)
        all_patterns = pipe._run_miners(bt, live, wf, result)
        result.n_patterns_raw = len(all_patterns)
        confirmed = pipe._filter(all_patterns)
        result.n_patterns_confirmed = len([p for p in confirmed if p.status.value == "confirmed"])
        result.confirmed_patterns = confirmed
        result.duration_s = time.monotonic() - _start
        print(
            f"[dry-run] pipeline complete in {result.duration_s:.2f}s | "
            f"raw={result.n_patterns_raw} confirmed={result.n_patterns_confirmed} "
            f"written=0 (dry run) errors={len(result.errors)}"
        )
        sys.exit(0)

    result = pipe.run()

    # Summary to stdout
    print(
        f"Pipeline complete in {result.duration_s:.2f}s | "
        f"raw={result.n_patterns_raw} confirmed={result.n_patterns_confirmed} "
        f"written={result.n_patterns_written} errors={len(result.errors)}"
    )
    if result.errors:
        for err in result.errors:
            print(f"  [ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
