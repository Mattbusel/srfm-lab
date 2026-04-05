"""
run_r_analysis.py — Python orchestration wrapper for R and Julia analyses.

Invokes R scripts (via Rscript) and Julia scripts (via julia) as subprocesses,
parses JSON output, and stores results back into idea_engine.db.

All public functions return plain Python dicts suitable for JSON serialisation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE       = Path(__file__).resolve().parent
_R_DIR      = _HERE / "r"
_JULIA_DIR  = _HERE / "julia"
_OUTPUT_DIR = _HERE / "output"

_DB_PATH    = Path(
    os.environ.get("IDEA_ENGINE_DB", str(_HERE.parent / "db" / "idea_engine.db"))
)

_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tool availability checks
# ---------------------------------------------------------------------------

def check_r_available() -> bool:
    """Return True if Rscript is on PATH and executable."""
    return shutil.which("Rscript") is not None


def check_julia_available() -> bool:
    """Return True if julia is on PATH and executable."""
    return shutil.which("julia") is not None


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _open_db() -> sqlite3.Connection | None:
    """Open a connection to idea_engine.db; returns None if DB doesn't exist."""
    if not _DB_PATH.exists():
        logger.warning("DB not found at %s", _DB_PATH)
        return None
    return sqlite3.connect(str(_DB_PATH))


def _ensure_schema(con: sqlite3.Connection) -> None:
    """Create stats_reports / optimization_runs tables if absent."""
    schema_file = _HERE / "schema_extension.sql"
    if schema_file.exists():
        sql = schema_file.read_text(encoding="utf-8")
        # Split on semicolons and execute each statement
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    con.execute(stmt)
                except sqlite3.OperationalError:
                    pass
        con.commit()


def _store_stats_report(
    run_id: str,
    report_type: str,
    content: dict[str, Any],
) -> None:
    """Persist a stats report to idea_engine.db."""
    con = _open_db()
    if con is None:
        logger.warning("Cannot persist report — DB unavailable")
        return
    try:
        _ensure_schema(con)
        con.execute(
            """
            INSERT INTO stats_reports (run_id, report_type, content_json)
            VALUES (?, ?, ?)
            """,
            (run_id, report_type, json.dumps(content)),
        )
        con.commit()
        logger.info("Stored stats report type=%s for run=%s", report_type, run_id)
    finally:
        con.close()


def _store_optimization_run(
    method: str,
    param_bounds: dict[str, Any],
    best_params: dict[str, Any] | None,
    best_score: float | None,
    n_iterations: int | None,
) -> int | None:
    """Persist an optimization run to idea_engine.db. Returns inserted row id."""
    con = _open_db()
    if con is None:
        logger.warning("Cannot persist optimization run — DB unavailable")
        return None
    try:
        _ensure_schema(con)
        cur = con.execute(
            """
            INSERT INTO optimization_runs
                (method, param_bounds, best_params, best_score, n_iterations)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                method,
                json.dumps(param_bounds),
                json.dumps(best_params) if best_params is not None else None,
                best_score,
                n_iterations,
            ),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


# ---------------------------------------------------------------------------
# JSON output file parser
# ---------------------------------------------------------------------------

def _read_output_json(name: str) -> dict[str, Any] | None:
    """
    Read a JSON output file written by an R or Julia script.

    Tries the output directory first, then the current working directory.
    Returns None if the file is absent or malformed.
    """
    candidates = [
        _OUTPUT_DIR / f"{name}.json",
        Path.cwd() / f"{name}.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open(encoding="utf-8") as fh:
                    return json.load(fh)
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse %s: %s", path, exc)
    logger.debug("Output file %s.json not found", name)
    return None


# ---------------------------------------------------------------------------
# R analysis runner
# ---------------------------------------------------------------------------

def run_r_analysis(
    run_id: str,
    script: str = "analysis.R",
    timeout: int = 600,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Invoke an R analysis script via Rscript subprocess.

    The script is expected to write JSON files to the output directory
    (set via STATS_OUTPUT_DIR env var).

    Parameters
    ----------
    run_id:     Backtest run identifier (passed as first CLI arg to the script)
    script:     Filename within stats-service/r/  (default: "analysis.R")
    timeout:    Maximum wall-clock seconds to wait (default: 600)
    extra_env:  Additional environment variables to inject

    Returns
    -------
    dict with keys:
        - ``success``: bool
        - ``run_id``: str
        - ``script``: str
        - ``output_files``: list of JSON result dicts keyed by output name
        - ``stderr``: captured stderr (truncated to 4 KB)
        - ``elapsed_s``: wall-clock seconds
    """
    if not check_r_available():
        logger.warning("Rscript not found — skipping R analysis")
        return {
            "success": False,
            "run_id":  run_id,
            "script":  script,
            "error":   "Rscript not available",
        }

    script_path = _R_DIR / script
    if not script_path.exists():
        return {
            "success": False,
            "run_id":  run_id,
            "script":  script,
            "error":   f"Script not found: {script_path}",
        }

    env = {**os.environ}
    env["IDEA_ENGINE_DB"]   = str(_DB_PATH)
    env["STATS_OUTPUT_DIR"] = str(_OUTPUT_DIR)
    if extra_env:
        env.update(extra_env)

    cmd = ["Rscript", "--vanilla", str(script_path), run_id]
    logger.info("Running R script: %s (run_id=%s)", script, run_id)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(_HERE),
        )
        elapsed = time.monotonic() - t0
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "run_id":  run_id,
            "script":  script,
            "error":   f"R script timed out after {timeout}s",
        }
    except Exception as exc:
        return {
            "success": False,
            "run_id":  run_id,
            "script":  script,
            "error":   str(exc),
        }

    success = proc.returncode == 0
    if not success:
        logger.error(
            "R script %s failed (rc=%d):\n%s",
            script, proc.returncode,
            proc.stderr[:2000],
        )

    # Collect all JSON output files written during this run
    _EXPECTED_OUTPUTS: dict[str, list[str]] = {
        "analysis.R": [
            "regime_clustering",
            "bootstrap_sharpe",
            "factor_attribution",
            "rolling_correlation",
            "drawdown_decomposition",
            "parameter_sensitivity_anova",
            "white_reality_check",
            "multiple_hypothesis_correction",
        ],
        "walk_forward_analysis.R": [
            "walk_forward_test",
            "parameter_stability",
            "efficiency_ratio",
            "optimal_f",
            "monte_carlo_dominance",
        ],
        "reporting.R": [
            "tearsheet",
            f"report_{run_id}",
        ],
    }

    output_data: dict[str, Any] = {}
    for output_name in _EXPECTED_OUTPUTS.get(script, []):
        data = _read_output_json(output_name)
        if data is not None:
            output_data[output_name] = data

    # Persist main report to DB
    if output_data:
        _store_stats_report(run_id, script.replace(".R", ""), output_data)

    return {
        "success":      success,
        "run_id":       run_id,
        "script":       script,
        "output_files": output_data,
        "stderr":       proc.stderr[:4096],
        "stdout":       proc.stdout[:4096],
        "elapsed_s":    round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Julia optimizer runner
# ---------------------------------------------------------------------------

def run_julia_optimizer(
    param_bounds: dict[str, tuple[float, float]],
    method: str = "bayesian",
    n_iter: int = 100,
    timeout: int = 900,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Invoke the Julia optimizer as a subprocess.

    Writes a temporary driver script that calls the appropriate Julia module
    function with the given parameter bounds, then reads the JSON result.

    Parameters
    ----------
    param_bounds: dict mapping param name → (min, max)
    method:       "bayesian" | "grid" | "pareto"
    n_iter:       number of optimization iterations
    timeout:      maximum seconds to wait
    extra_env:    additional environment variables

    Returns
    -------
    dict with keys:
        - ``success``: bool
        - ``method``: str
        - ``best_params``: dict or None
        - ``best_score``: float or None
        - ``n_iterations``: int or None
        - ``history``: list of {params, score} dicts
        - ``elapsed_s``: float
    """
    if not check_julia_available():
        logger.warning("julia not found — skipping Julia optimizer")
        return {
            "success": False,
            "method":  method,
            "error":   "julia not available",
        }

    # Write a temporary driver script
    driver_path = _OUTPUT_DIR / "_driver_optimizer.jl"
    bounds_julia = "{" + ", ".join(
        f'"{k}" => ({v[0]}, {v[1]})'
        for k, v in param_bounds.items()
    ) + "}"

    driver_content = f"""
push!(LOAD_PATH, "{_JULIA_DIR.as_posix()}")
include("{(_JULIA_DIR / 'optimizer.jl').as_posix()}")
using .Optimizer
import JSON3

bounds = Dict{str}({bounds_julia})

result = Optimizer.optimize(
    Optimizer.objective_sharpe,
    bounds;
    n_iter = {n_iter}
)

out = Optimizer.result_to_json(result)
out_path = joinpath("{_OUTPUT_DIR.as_posix()}", "optimizer_result.json")
open(out_path, "w") do io
    write(io, out)
end
println("[julia] Done. best_score=", result.best_score)
"""
    # Fix the Dict type annotation in the generated script
    driver_content = driver_content.replace("{str}", "{String, Tuple{Float64,Float64}}")

    driver_path.write_text(driver_content, encoding="utf-8")

    env = {**os.environ}
    env["IDEA_ENGINE_DB"]   = str(_DB_PATH)
    env["STATS_OUTPUT_DIR"] = str(_OUTPUT_DIR)
    if extra_env:
        env.update(extra_env)

    cmd = ["julia", "--threads=auto", str(driver_path)]
    logger.info("Running Julia optimizer (method=%s, n_iter=%d)", method, n_iter)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(_HERE),
        )
        elapsed = time.monotonic() - t0
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "method":  method,
            "error":   f"Julia timed out after {timeout}s",
        }
    except Exception as exc:
        return {"success": False, "method": method, "error": str(exc)}

    success = proc.returncode == 0

    # Parse result
    result_data = _read_output_json("optimizer_result")

    best_params   = result_data.get("best_params")   if result_data else None
    best_score    = result_data.get("best_score")    if result_data else None
    n_iterations  = result_data.get("n_iterations")  if result_data else None
    history       = result_data.get("history", [])   if result_data else []

    # Persist to DB
    _store_optimization_run(method, param_bounds, best_params, best_score, n_iterations)

    # Clean up driver
    try:
        driver_path.unlink()
    except OSError:
        pass

    return {
        "success":     success,
        "method":      method,
        "best_params": best_params,
        "best_score":  best_score,
        "n_iterations": n_iterations,
        "history":     history,
        "stderr":      proc.stderr[:4096],
        "elapsed_s":   round(elapsed, 2),
    }


def run_julia_time_series(
    series: list[float],
    analyses: list[str] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """
    Run Julia time-series analyses on a numeric series.

    Parameters
    ----------
    series:    List of float values (e.g. daily returns or prices)
    analyses:  Subset of analyses to run.  Defaults to all:
               ["hurst", "fracdiff", "arima", "wavelet", "spectral"]
    timeout:   Max seconds

    Returns
    -------
    dict with analysis results keyed by analysis name
    """
    if not check_julia_available():
        return {"success": False, "error": "julia not available"}

    analyses = analyses or ["hurst", "fracdiff", "arima", "wavelet", "spectral"]

    series_julia = "[" + ", ".join(str(v) for v in series) + "]"
    out_path     = _OUTPUT_DIR / "ts_results.json"

    include_path = (_JULIA_DIR / "time_series.jl").as_posix()
    out_posix    = out_path.as_posix()

    driver = f"""
include("{include_path}")
using .TimeSeries
import JSON3, Statistics

ts = Float64.({series_julia})

results = Dict{{String, Any}}()

if "hurst" in {json.dumps(analyses)}
    h = TimeSeries.HurstExponent(ts)
    results["hurst"] = Dict("H" => h.H, "interpretation" => h.interpretation)
end

if "fracdiff" in {json.dumps(analyses)}
    d_opt = TimeSeries.find_stationary_d(ts)
    fd = TimeSeries.FractionalDifferencing(ts, d_opt)
    results["fracdiff"] = Dict("d" => d_opt, "n_output" => length(fd.differenced))
end

if "arima" in {json.dumps(analyses)}
    ar = TimeSeries.ARIMA_fit(ts)
    results["arima"] = Dict("p" => ar.p, "d" => ar.d, "q" => ar.q,
                             "bic" => ar.bic, "aic" => ar.aic)
end

if "wavelet" in {json.dumps(analyses)}
    wt = TimeSeries.WaveletDecomposition(ts; levels=4)
    results["wavelet"] = Dict("snr" => wt.snr, "n_levels" => wt.n_levels,
                               "energy_fraction" => collect(wt.energy_fraction))
end

if "spectral" in {json.dumps(analyses)}
    sd = TimeSeries.SpectralDensity(ts)
    results["spectral"] = Dict("dominant_cycles" => sd.dominant_cycles_days,
                                "n_samples" => sd.n_samples)
end

open("{out_posix}", "w") do io
    write(io, JSON3.write(results))
end
println("[ts] Done")
"""

    driver_path = _OUTPUT_DIR / "_driver_ts.jl"
    driver_path.write_text(driver, encoding="utf-8")

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["julia", "--threads=auto", str(driver_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(_HERE),
        )
        elapsed = time.monotonic() - t0
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timed out after {timeout}s"}
    finally:
        try: driver_path.unlink()
        except OSError: pass

    data = None
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    return {
        "success":  proc.returncode == 0,
        "results":  data or {},
        "elapsed_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run R or Julia analysis scripts for the IAE stats service"
    )
    sub = parser.add_subparsers(dest="command")

    r_cmd = sub.add_parser("r", help="Run an R analysis script")
    r_cmd.add_argument("run_id", help="Backtest run ID")
    r_cmd.add_argument("--script", default="analysis.R")
    r_cmd.add_argument("--timeout", type=int, default=600)

    j_cmd = sub.add_parser("julia", help="Run Julia optimizer")
    j_cmd.add_argument("--method", choices=["bayesian", "grid", "pareto"],
                        default="bayesian")
    j_cmd.add_argument("--n-iter", type=int, default=100)
    j_cmd.add_argument(
        "--bounds",
        help='JSON dict: {"param": [min, max], ...}',
        default='{"fast_period": [5, 50], "slow_period": [20, 200]}',
    )

    check_cmd = sub.add_parser("check", help="Check tool availability")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if args.command == "r":
        result = run_r_analysis(args.run_id, script=args.script, timeout=args.timeout)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "julia":
        bounds_raw = json.loads(args.bounds)
        bounds = {k: tuple(v) for k, v in bounds_raw.items()}
        result = run_julia_optimizer(bounds, method=args.method, n_iter=args.n_iter)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "check":
        r_ok = check_r_available()
        j_ok = check_julia_available()
        print(f"Rscript available : {'YES' if r_ok else 'NO'}")
        print(f"julia available   : {'YES' if j_ok else 'NO'}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
