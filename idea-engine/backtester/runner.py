"""
BacktestRunner
==============
Programmatic interface for running crypto_backtest_mc.py as a subprocess.

Features:
  - Synchronous, async (callback-based), and batched (multiprocessing.Pool) execution.
  - Hypothesis-driven runs that load param_delta from the database.
  - Graceful error handling — subprocess failures return error BacktestResults,
    not exceptions.
  - Configurable timeout per run (default 120 s).
  - Results persisted to the ``sim_runs`` table in idea_engine.db.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable

from .param_manager import ParamManager, BASELINE_PARAMS
from .result_parser import ResultParser, BacktestResult, BacktestMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_ENV         = "IDEA_ENGINE_DB"
DEFAULT_DB_PATH     = Path(__file__).resolve().parents[2] / "idea_engine.db"
REPO_ROOT           = Path(__file__).resolve().parents[3]      # srfm-lab/
BACKTEST_MODULE     = "tools.crypto_backtest_mc"
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_N_WORKERS   = 4


# ---------------------------------------------------------------------------
# Worker function — must be module-level for pickling
# ---------------------------------------------------------------------------

def _run_worker(args: tuple) -> BacktestResult:
    """
    Subprocess worker that runs one backtest and returns a BacktestResult.

    ``args`` layout:
        (params, label, timeout_sec, repo_root, bars_per_year)
    """
    params, label, timeout_sec, repo_root, bars_per_year = args

    param_manager = ParamManager()
    result_parser = ResultParser(bars_per_year=bars_per_year)
    params_hash = param_manager.hash_params(params)

    stdout_txt = ""
    stderr_txt = ""
    trades_csv_path: Path | None = None
    params_file: Path | None = None
    t0 = time.perf_counter()

    try:
        # Write params to a temp JSON file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=str(repo_root)
        ) as pf:
            json.dump(params, pf)
            params_file = Path(pf.name)

        # Output CSV path
        trades_fd, trades_path_str = tempfile.mkstemp(
            suffix=".csv", dir=str(repo_root)
        )
        os.close(trades_fd)
        trades_csv_path = Path(trades_path_str)

        cmd = [
            sys.executable, "-m", BACKTEST_MODULE,
            "--params-file", str(params_file),
            "--output-csv",  str(trades_csv_path),
            "--quiet",
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        stdout_txt = proc.stdout or ""
        stderr_txt = proc.stderr or ""
        duration = time.perf_counter() - t0

        if proc.returncode != 0:
            logger.warning(
                "Backtest exited %d for params_hash=%s: %s",
                proc.returncode, params_hash, stderr_txt[:300],
            )
            return BacktestResult(
                params_hash=params_hash,
                params=params,
                metrics=BacktestMetrics(),
                label=label,
                stdout=stdout_txt,
                stderr=stderr_txt,
                duration_seconds=duration,
                status="error",
                error_message=f"exit code {proc.returncode}: {stderr_txt[:300]}",
            )

        result = result_parser.parse_all(
            stdout=stdout_txt,
            trades_csv_path=trades_csv_path if trades_csv_path.stat().st_size > 0 else None,
            params=params,
            params_hash=params_hash,
            label=label,
            duration_seconds=duration,
        )
        result.stderr = stderr_txt
        return result

    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - t0
        logger.warning("Backtest timed out after %ds (params_hash=%s).", timeout_sec, params_hash)
        return BacktestResult(
            params_hash=params_hash,
            params=params,
            metrics=BacktestMetrics(),
            label=label,
            stdout=stdout_txt,
            stderr=stderr_txt,
            duration_seconds=duration,
            status="timeout",
            error_message=f"Timed out after {timeout_sec}s",
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - t0
        logger.error("Backtest worker exception: %s", exc, exc_info=True)
        return BacktestResult(
            params_hash=params_hash,
            params=params,
            metrics=BacktestMetrics(),
            label=label,
            stdout=stdout_txt,
            stderr=stderr_txt,
            duration_seconds=duration,
            status="error",
            error_message=str(exc),
        )
    finally:
        for p in filter(None, [params_file, trades_csv_path]):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------

class BacktestRunner:
    """
    Runs backtests via subprocess and persists results to idea_engine.db.

    Parameters
    ----------
    db_path : path to idea_engine.db.
    timeout_sec : per-run subprocess timeout in seconds.
    n_workers : number of parallel workers for batch runs.
    bars_per_year : used by ResultParser for annualisation.
    repo_root : root of the srfm-lab repository (cwd for subprocess).
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        n_workers: int = DEFAULT_N_WORKERS,
        bars_per_year: int = ResultParser.__init__.__defaults__[0],  # type: ignore[index]
        repo_root: Path | str | None = None,
    ) -> None:
        self.db_path = Path(
            db_path or os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
        )
        self.timeout_sec = timeout_sec
        self.n_workers = n_workers
        self.bars_per_year = bars_per_year
        self.repo_root = Path(repo_root or REPO_ROOT)
        self._param_manager = ParamManager()
        self._result_parser = ResultParser(bars_per_year=bars_per_year)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Core run methods
    # ------------------------------------------------------------------

    def run(
        self,
        params: dict[str, Any],
        label: str | None = None,
        experiment_id: int | None = None,
        hypothesis_id: int | None = None,
    ) -> BacktestResult:
        """
        Run a single backtest synchronously.

        Parameters
        ----------
        params        : complete parameter dict.
        label         : optional human-readable label for logging/DB.
        experiment_id : ID of the parent experiment in the DB.
        hypothesis_id : ID of the associated hypothesis in the DB.

        Returns
        -------
        BacktestResult — never raises; failures are captured in result.status.
        """
        logger.info(
            "Starting backtest: label=%r params_hash=%s",
            label, self._param_manager.hash_params(params),
        )
        args = (params, label, self.timeout_sec, self.repo_root, self.bars_per_year)
        result = _run_worker(args)
        self.result_to_db(result, experiment_id=experiment_id, hypothesis_id=hypothesis_id)
        return result

    def run_async(
        self,
        params: dict[str, Any],
        callback: Callable[[BacktestResult], None],
        label: str | None = None,
        experiment_id: int | None = None,
        hypothesis_id: int | None = None,
    ) -> threading.Thread:
        """
        Run a backtest in a daemon thread and invoke *callback* when complete.

        Returns the Thread object (already started).
        """
        def _target() -> None:
            result = self.run(params, label=label,
                              experiment_id=experiment_id,
                              hypothesis_id=hypothesis_id)
            try:
                callback(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("run_async callback raised: %s", exc)

        t = threading.Thread(target=_target, daemon=True, name=f"bt-{label or 'run'}")
        t.start()
        return t

    def run_batch(
        self,
        param_list: list[dict[str, Any]],
        labels: list[str | None] | None = None,
        n_workers: int | None = None,
        experiment_id: int | None = None,
    ) -> list[BacktestResult]:
        """
        Run multiple backtests in parallel using multiprocessing.Pool.

        Parameters
        ----------
        param_list : list of complete param dicts to run.
        labels     : optional list of labels (same length as param_list).
        n_workers  : number of processes (overrides instance default).
        experiment_id : DB experiment ID for all runs in this batch.

        Returns
        -------
        list of BacktestResult in the same order as param_list.
        """
        workers = n_workers or self.n_workers
        n = len(param_list)
        if n == 0:
            return []

        if labels is None:
            labels = [None] * n
        elif len(labels) != n:
            raise ValueError("labels must be the same length as param_list")

        args_list = [
            (p, l, self.timeout_sec, self.repo_root, self.bars_per_year)
            for p, l in zip(param_list, labels)
        ]

        logger.info("Starting batch of %d backtests with %d workers.", n, workers)
        t0 = time.perf_counter()

        with Pool(processes=workers) as pool:
            results = pool.map(_run_worker, args_list)

        elapsed = time.perf_counter() - t0
        logger.info("Batch complete: %d runs in %.1fs.", n, elapsed)

        for result in results:
            self.result_to_db(result, experiment_id=experiment_id)

        return results

    def run_with_hypothesis(
        self,
        hypothesis_id: int,
        baseline_params: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """
        Load a hypothesis from the DB, apply its param_delta to *baseline_params*,
        and run the resulting backtest.

        Parameters
        ----------
        hypothesis_id   : row ID in the ``hypotheses`` table.
        baseline_params : base params to apply the delta to.
                          Defaults to BASELINE_PARAMS.
        """
        delta, label = self._load_hypothesis_delta(hypothesis_id)
        base = baseline_params or self._param_manager.get_baseline()
        params = self._param_manager.apply_delta(base, delta)

        return self.run(
            params,
            label=label or f"hypothesis_{hypothesis_id}",
            hypothesis_id=hypothesis_id,
        )

    # ------------------------------------------------------------------
    # Baseline params
    # ------------------------------------------------------------------

    def get_baseline_params(self) -> dict[str, Any]:
        """
        Return the current production baseline parameters.

        In future this could parse live_trader_alpaca.py constants dynamically;
        for now it delegates to ParamManager.
        """
        return self._param_manager.get_baseline()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def result_to_db(
        self,
        result: BacktestResult,
        experiment_id: int | None = None,
        hypothesis_id: int | None = None,
    ) -> int | None:
        """
        Persist *result* to the ``sim_runs`` table.

        Returns the new row ID, or None if the DB is not available.
        """
        if not self.db_path.exists():
            logger.debug("DB not found at %s; skipping persist.", self.db_path)
            return None

        m = result.metrics
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.execute(
                """
                INSERT INTO sim_runs
                    (params_hash, params_json, label, hypothesis_id, experiment_id,
                     sharpe, calmar, max_dd, total_return, win_rate, num_trades,
                     duration_seconds, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.params_hash,
                    json.dumps(result.params, sort_keys=True),
                    result.label,
                    hypothesis_id,
                    experiment_id,
                    m.sharpe,
                    m.calmar,
                    m.max_dd,
                    m.total_return,
                    m.win_rate,
                    m.num_trades,
                    result.duration_seconds,
                    result.status,
                ),
            )
            row_id = cur.lastrowid
            conn.commit()
            conn.close()
            logger.debug("Persisted backtest result: id=%d status=%s", row_id, result.status)
            return row_id
        except sqlite3.Error as exc:
            logger.warning("Failed to persist backtest result: %s", exc)
            return None

    def load_result(self, run_id: int) -> dict[str, Any] | None:
        """Load a previously persisted sim_run by its row ID."""
        if not self.db_path.exists():
            return None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sim_runs WHERE id = ?", (run_id,)
            ).fetchone()
            conn.close()
            return dict(row) if row else None
        except sqlite3.Error as exc:
            logger.warning("Failed to load result %d: %s", run_id, exc)
            return None

    def top_runs(
        self,
        metric: str = "sharpe",
        limit: int = 20,
        status: str = "completed",
    ) -> list[dict[str, Any]]:
        """Return the top *limit* sim_runs ordered by *metric* descending."""
        allowed = {"sharpe", "calmar", "total_return", "win_rate"}
        if metric not in allowed:
            raise ValueError(f"metric must be one of {allowed}")
        if not self.db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM sim_runs WHERE status = ? "
                f"ORDER BY {metric} DESC LIMIT ?",
                (status, limit),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.warning("Failed to query top runs: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hypothesis_delta(
        self, hypothesis_id: int
    ) -> tuple[dict[str, Any], str | None]:
        """Load param_delta JSON and title from the hypotheses table."""
        if not self.db_path.exists():
            logger.warning("DB not found; using empty delta for hypothesis %d.", hypothesis_id)
            return {}, None

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT title, param_delta FROM hypotheses WHERE id = ?",
                (hypothesis_id,),
            ).fetchone()
            conn.close()
            if row is None:
                logger.warning("Hypothesis %d not found in DB.", hypothesis_id)
                return {}, None

            delta_raw = row["param_delta"]
            delta = json.loads(delta_raw) if delta_raw else {}
            return delta, row["title"]
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            logger.warning("Failed to load hypothesis %d: %s", hypothesis_id, exc)
            return {}, None

    def _ensure_schema(self) -> None:
        sql_path = Path(__file__).parent / "schema_extension.sql"
        if not sql_path.exists() or not self.db_path.exists():
            return
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.executescript(sql_path.read_text(encoding="utf-8"))
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not apply backtester schema: %s", exc)

    # ------------------------------------------------------------------
    # Batch sweep helpers
    # ------------------------------------------------------------------

    def parameter_sweep(
        self,
        param_name: str,
        values: list[float],
        baseline: dict[str, Any] | None = None,
        n_workers: int | None = None,
        experiment_id: int | None = None,
    ) -> list[BacktestResult]:
        """
        Run a 1-D sweep over *param_name* at each of *values*.

        Returns results in the same order as *values*.
        """
        base = baseline or self._param_manager.get_baseline()
        param_list = [
            self._param_manager.apply_delta(base, {param_name: v})
            for v in values
        ]
        labels = [f"sweep_{param_name}={v}" for v in values]
        return self.run_batch(param_list, labels=labels,
                              n_workers=n_workers, experiment_id=experiment_id)

    def neighborhood_sweep(
        self,
        center: dict[str, Any] | None = None,
        step_pct: float = 0.05,
        n_workers: int | None = None,
        experiment_id: int | None = None,
    ) -> list[BacktestResult]:
        """
        Run all ±step_pct neighbors of *center* (one param at a time).

        Returns results for all neighbors.
        """
        base = center or self._param_manager.get_baseline()
        neighbors = self._param_manager.enumerate_neighbors(base, step_pct=step_pct)
        labels = [f"neighbor_{i}" for i in range(len(neighbors))]
        return self.run_batch(neighbors, labels=labels,
                              n_workers=n_workers, experiment_id=experiment_id)
