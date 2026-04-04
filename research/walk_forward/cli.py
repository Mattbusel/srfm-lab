"""
research/walk_forward/cli.py
──────────────────────────────
Click CLI for the walk-forward analysis platform.

Commands:
  wf run      — run walk-forward analysis
  wf optimize — run parameter optimization
  wf cpcv     — run CPCV analysis
  wf report   — generate HTML report from saved results

Usage examples:
  wf run     --trades crypto_trades.csv --train 500 --test 100
  wf optimize --trades crypto_trades.csv --method sobol --n-iter 200 --output results/
  wf cpcv    --trades crypto_trades.csv --folds 6 --test-folds 2
  wf report  --results results/wf_result.json --output report/
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Bootstrap project path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "lib"), str(_PROJECT_ROOT / "spacetime")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import click
except ImportError:
    raise ImportError(
        "click is required for the CLI. Install with: pip install click"
    )

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level  = level,
        format = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_trades(trades_path: str) -> pd.DataFrame:
    """Load trades from CSV, Parquet, or JSON file."""
    path = Path(trades_path)
    if not path.exists():
        raise click.BadParameter(f"File not found: {trades_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(trades_path, parse_dates=True)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(trades_path)
    elif suffix == ".json":
        df = pd.read_json(trades_path)
    else:
        raise click.BadParameter(
            f"Unsupported file format: {suffix}. Use .csv, .parquet, or .json"
        )

    # Auto-parse time columns
    for col in ("exit_time", "entry_time", "timestamp"):
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    click.echo(f"  Loaded {len(df)} trades from {path.name}")
    click.echo(f"  Columns: {list(df.columns)}")
    return df


def _parse_param_grid(param_grid_str: Optional[str]) -> Optional[Dict[str, List]]:
    """Parse JSON param grid string from CLI."""
    if not param_grid_str:
        return None
    try:
        grid = json.loads(param_grid_str)
        if not isinstance(grid, dict):
            raise ValueError("param_grid must be a JSON object")
        # Ensure all values are lists
        for k, v in grid.items():
            if not isinstance(v, list):
                grid[k] = [v]
        return grid
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"Invalid JSON for param_grid: {e}")


def _default_param_grid() -> Dict[str, List]:
    """Default BH parameter grid."""
    return {
        "cf":          [0.001, 0.0015, 0.002, 0.0025, 0.003],
        "bh_form":     [1.2, 1.5, 1.8, 2.0],
        "bh_collapse": [0.7, 0.8, 0.9, 1.0],
        "bh_decay":    [0.90, 0.92, 0.95, 0.98],
    }


def _save_result_json(result: Any, path: str) -> None:
    """Serialize WFResult or CPCVResult to JSON."""
    from dataclasses import asdict

    def _serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            f = float(obj)
            return f if np.isfinite(f) else None
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serializable(v) for v in obj]
        return obj

    try:
        data = asdict(result)
        data_clean = _serializable(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_clean, f, indent=2, default=str)
        click.echo(f"  Saved result JSON: {path}")
    except Exception as e:
        click.echo(f"  Warning: could not save JSON result: {e}", err=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI group
# ─────────────────────────────────────────────────────────────────────────────

@click.group(
    name="wf",
    help="SRFM-Lab Walk-Forward Analysis + CPCV Platform\n\n"
         "Proper out-of-sample validation and anti-overfitting tools for\n"
         "strategy parameter selection using the Black Hole engine.",
)
@click.version_option(version="1.0.0", prog_name="wf")
def cli() -> None:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# wf run
# ─────────────────────────────────────────────────────────────────────────────

@cli.command(name="run")
@click.option("--trades",       "-t", required=True,  help="Path to trades CSV/Parquet/JSON")
@click.option("--output",       "-o", default="results/wf/", show_default=True,
              help="Output directory for results")
@click.option("--train",        default=500,  show_default=True,
              help="Training window size (number of trades)")
@click.option("--test",         default=100,  show_default=True,
              help="Test window size (number of trades)")
@click.option("--step",         default=100,  show_default=True,
              help="Step size between folds")
@click.option("--gap",          default=5,    show_default=True,
              help="Embargo gap between train and test windows")
@click.option("--mode",         default="rolling", show_default=True,
              type=click.Choice(["rolling", "expanding"]),
              help="Walk-forward mode: rolling or expanding window")
@click.option("--metric",       default="sharpe", show_default=True,
              type=click.Choice(["sharpe", "sortino", "calmar", "profit_factor", "win_rate"]),
              help="IS optimization metric")
@click.option("--param-grid",   default=None,
              help="JSON string for param grid, e.g. '{\"cf\": [0.001, 0.002]}'")
@click.option("--n-jobs",       default=-1, show_default=True,
              help="Parallel workers (-1 = all CPUs)")
@click.option("--equity",       default=100_000.0, show_default=True,
              help="Starting equity")
@click.option("--sym",          default="BTC", show_default=True,
              help="Instrument symbol")
@click.option("--verbose",      "-v", is_flag=True, help="Verbose logging")
@click.option("--no-report",    is_flag=True, help="Skip HTML report generation")
@click.option("--no-plots",     is_flag=True, help="Skip plot generation")
def run_command(
    trades, output, train, test, step, gap, mode, metric,
    param_grid, n_jobs, equity, sym, verbose, no_report, no_plots,
) -> None:
    """
    Run walk-forward analysis with IS parameter optimization.

    For each fold, performs a grid search over the parameter space on the
    training window, selects the best parameters, and evaluates them on the
    out-of-sample test window.

    \b
    Examples:
      wf run --trades btc_trades.csv --train 500 --test 100 --metric sharpe
      wf run --trades trades.csv --mode expanding --param-grid '{"cf": [0.001, 0.002]}'
    """
    _setup_logging(verbose)

    click.echo(click.style("\n📊 SRFM-Lab Walk-Forward Analysis", fg="cyan", bold=True))
    click.echo(f"  Trades:  {trades}")
    click.echo(f"  Mode:    {mode}")
    click.echo(f"  Train:   {train}, Test: {test}, Step: {step}, Gap: {gap}")
    click.echo(f"  Metric:  {metric}")
    click.echo(f"  Output:  {output}")

    # Load trades
    click.echo("\n→ Loading trades...")
    df = _load_trades(trades)

    # Build param grid
    pg = _parse_param_grid(param_grid) or _default_param_grid()
    n_combos = 1
    for v in pg.values():
        n_combos *= len(v)
    click.echo(f"\n→ Param grid: {n_combos} combinations")

    # Build splitter
    from .splits import walk_forward_splits, expanding_window_splits
    n = len(df)
    if mode == "rolling":
        splits = walk_forward_splits(n, train, test, step, gap)
    else:
        splits = expanding_window_splits(n, train, test, step, gap)

    click.echo(f"  {len(splits)} folds generated")

    if not splits:
        click.echo(click.style("  ERROR: No folds generated. Check train/test/step parameters.", fg="red"))
        sys.exit(1)

    # Build adapter
    from .backtest_adapter import BHStrategyAdapter
    adapter = BHStrategyAdapter(sym=sym, starting_equity=equity)

    # Run engine
    from .engine import WalkForwardEngine
    engine = WalkForwardEngine(
        strategy_fn     = adapter.run_bh_strategy,
        param_grid      = pg,
        n_jobs          = n_jobs,
        verbose         = verbose,
        metric          = metric,
        starting_equity = equity,
    )

    click.echo(f"\n→ Running {len(splits)} folds...")
    with click.progressbar(length=len(splits), label="  Progress") as bar:
        _fold_done = [0]
        orig_evaluate = engine._run_fold

        def _tracked_fold(trades_df, split):
            result = orig_evaluate(trades_df, split)
            _fold_done[0] += 1
            bar.update(1)
            return result

        engine._run_fold = _tracked_fold
        wf_result = engine.run(df, splits)

    # Summary
    click.echo(click.style("\n✓ Walk-Forward Complete", fg="green", bold=True))
    click.echo(f"  OOS Sharpe:     {wf_result.oos_sharpe:.4f}"
               f"  (95% CI: [{wf_result.sharpe_ci[0]:.3f}, {wf_result.sharpe_ci[1]:.3f}])")
    click.echo(f"  OOS CAGR:       {wf_result.oos_cagr:.1%}")
    click.echo(f"  OOS Max DD:     {wf_result.oos_max_dd:.1%}")
    click.echo(f"  Param Stability:{wf_result.param_stability_score:.1%}")
    click.echo(f"  Best Params:    {wf_result.best_params}")
    click.echo(f"  Elapsed:        {wf_result.total_elapsed_sec:.1f}s")

    # Save output
    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n→ Saving results to {output}...")

    # Save JSON
    _save_result_json(wf_result, str(out_path / "wf_result.json"))

    # Save fold summary CSV
    from .engine import is_oos_degradation_summary
    fold_df = is_oos_degradation_summary(wf_result)
    if not fold_df.empty:
        fold_df.to_csv(out_path / "fold_summary.csv", index=False)
        click.echo(f"  Saved fold_summary.csv")

    # Generate report
    if not no_report:
        click.echo("\n→ Generating HTML report...")
        from .report import generate_wf_report, to_console
        report = generate_wf_report(
            wf_result  = wf_result,
            output_dir = str(out_path / "report"),
            save_plots = not no_plots,
        )
        to_console(report)
        if report.html_path:
            click.echo(click.style(f"\n  Report: {report.html_path}", fg="cyan"))

    click.echo(click.style("\n✓ Done!", fg="green", bold=True))


# ─────────────────────────────────────────────────────────────────────────────
# wf optimize
# ─────────────────────────────────────────────────────────────────────────────

@cli.command(name="optimize")
@click.option("--trades",       "-t", required=True,  help="Path to trades CSV/Parquet/JSON")
@click.option("--output",       "-o", default="results/opt/", show_default=True,
              help="Output directory")
@click.option("--method",       "-m", default="sobol", show_default=True,
              type=click.Choice(["grid", "random", "sobol", "bayesian"]),
              help="Optimization method")
@click.option("--n-iter",       default=200, show_default=True,
              help="Number of optimization iterations (random/sobol/bayesian)")
@click.option("--n-init",       default=20,  show_default=True,
              help="Bayesian warm-up iterations")
@click.option("--metric",       default="sharpe", show_default=True,
              type=click.Choice(["sharpe", "sortino", "calmar", "profit_factor"]),
              help="Optimization objective metric")
@click.option("--train",        default=500,  show_default=True, help="IS train window")
@click.option("--test",         default=100,  show_default=True, help="OOS test window")
@click.option("--step",         default=100,  show_default=True, help="Fold step")
@click.option("--n-folds",      default=3,    show_default=True,
              help="Number of IS folds for optimization scoring")
@click.option("--equity",       default=100_000.0, show_default=True, help="Starting equity")
@click.option("--sym",          default="BTC", show_default=True, help="Instrument symbol")
@click.option("--seed",         default=42,   show_default=True, help="Random seed")
@click.option("--param-space",  default=None,
              help="JSON param space (for random/sobol/bayesian). "
                   "E.g. '{\"cf\": [0.001, 0.003, true]}'")
@click.option("--verbose",      "-v", is_flag=True, help="Verbose logging")
def optimize_command(
    trades, output, method, n_iter, n_init, metric, train, test,
    step, n_folds, equity, sym, seed, param_space, verbose,
) -> None:
    """
    Optimize BH strategy parameters using the selected method.

    Runs IS optimization on walk-forward splits and reports the best parameters
    with convergence curves and parameter importance analysis.

    \b
    Examples:
      wf optimize --trades trades.csv --method sobol --n-iter 200
      wf optimize --trades trades.csv --method bayesian --n-iter 50 --n-init 15
      wf optimize --trades trades.csv --method grid --metric profit_factor
    """
    _setup_logging(verbose)

    click.echo(click.style(f"\n🔍 Walk-Forward Optimization ({method.upper()})", fg="cyan", bold=True))
    click.echo(f"  Trades:  {trades}")
    click.echo(f"  Method:  {method}, N-iter: {n_iter}, Metric: {metric}")

    df = _load_trades(trades)

    # Build IS splits (use only first n_folds rolling folds for optimization)
    from .splits import walk_forward_splits
    n      = len(df)
    splits = walk_forward_splits(n, train, test, step, gap=5)[:n_folds]

    if not splits:
        click.echo(click.style("ERROR: No folds generated.", fg="red"))
        sys.exit(1)
    click.echo(f"  Using {len(splits)} IS folds for scoring")

    # Build adapter and param space
    from .backtest_adapter import BHStrategyAdapter
    from .optimizer import ParamOptimizer, build_param_space

    adapter = BHStrategyAdapter(sym=sym, starting_equity=equity)

    # Parse param space
    if param_space:
        try:
            ps_dict = json.loads(param_space)
        except json.JSONDecodeError as e:
            click.echo(click.style(f"ERROR: Invalid param_space JSON: {e}", fg="red"))
            sys.exit(1)
    else:
        ps_dict = adapter.get_sobol_param_space()

    click.echo(f"  Param space: {list(ps_dict.keys())}")

    # Run optimization
    opt = ParamOptimizer(
        strategy_fn     = adapter.run_bh_strategy,
        metric          = metric,
        starting_equity = equity,
        verbose         = verbose,
        seed            = seed,
    )

    click.echo(f"\n→ Running {method} optimization ({n_iter} evaluations)...")

    result = opt.optimize(
        method      = method,
        trades      = df,
        splitter    = splits,
        param_space = ps_dict if method in ("random", "sobol", "bayesian") else
                      {k: list(np.linspace(v[0], v[1], 5)) if isinstance(v, tuple) else v
                       for k, v in ps_dict.items()},
        n_iter      = n_iter,
        n_init      = n_init,
    )

    # Summary
    click.echo(click.style("\n✓ Optimization Complete", fg="green", bold=True))
    click.echo(f"  Best Score:  {result.best_score:.4f} ({metric})")
    click.echo(f"  Best Params: {result.best_params}")
    click.echo(f"  N Trials:    {result.n_trials}")
    click.echo(f"  Elapsed:     {result.elapsed_sec:.1f}s")

    click.echo("\n  Parameter Importance:")
    for param, importance in sorted(result.param_importance.items(), key=lambda x: -x[1]):
        bar_len = int(importance * 30)
        bar_str = "█" * bar_len + "░" * (30 - bar_len)
        click.echo(f"    {param:<15} {bar_str}  {importance:.3f}")

    # Save output
    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save trial results CSV
    trials_df = result.to_dataframe()
    trials_df.to_csv(out_path / "trials.csv", index=False)

    # Save best params JSON
    with open(out_path / "best_params.json", "w") as f:
        json.dump({
            "best_params":       result.best_params,
            "best_score":        result.best_score,
            "metric":            result.metric,
            "method":            result.method,
            "n_trials":          result.n_trials,
            "param_importance":  result.param_importance,
        }, f, indent=2)
    click.echo(f"  Saved best_params.json")

    # Generate plots
    from .optimizer import plot_convergence, plot_param_importance
    try:
        plot_convergence(result, save_path=str(out_path / "convergence.png"), show=False)
        plot_param_importance(result, save_path=str(out_path / "importance.png"), show=False)
        click.echo("  Saved convergence.png, importance.png")
    except Exception as e:
        click.echo(f"  Warning: plot generation failed: {e}", err=True)

    click.echo(click.style(f"\n✓ Results saved to: {output}", fg="green"))


# ─────────────────────────────────────────────────────────────────────────────
# wf cpcv
# ─────────────────────────────────────────────────────────────────────────────

@cli.command(name="cpcv")
@click.option("--trades",       "-t", required=True,  help="Path to trades CSV/Parquet/JSON")
@click.option("--output",       "-o", default="results/cpcv/", show_default=True,
              help="Output directory")
@click.option("--folds",        default=6,  show_default=True,
              help="Total number of CPCV groups (k)")
@click.option("--test-folds",   default=2,  show_default=True,
              help="Number of test groups per combination (k_test)")
@click.option("--purge",        default=5,  show_default=True,
              help="Purge bars at group boundaries")
@click.option("--embargo",      default=5,  show_default=True,
              help="Embargo bars after test groups")
@click.option("--metric",       default="sharpe", show_default=True,
              type=click.Choice(["sharpe", "sortino", "calmar", "profit_factor"]),
              help="IS optimization metric")
@click.option("--equity",       default=100_000.0, show_default=True, help="Starting equity")
@click.option("--sym",          default="BTC",     show_default=True, help="Symbol")
@click.option("--param-grid",   default=None,
              help="JSON param grid. Default: minimal BH grid")
@click.option("--verbose",      "-v", is_flag=True, help="Verbose logging")
def cpcv_command(
    trades, output, folds, test_folds, purge, embargo,
    metric, equity, sym, param_grid, verbose,
) -> None:
    """
    Run Combinatorial Purged Cross-Validation (CPCV) analysis.

    Evaluates all C(k, k_test) train/test combinations to produce:
      • Deflated Sharpe Ratio (DSR)
      • Probability of Backtest Overfitting (PBO)
      • Path-averaged OOS performance distribution

    \b
    Examples:
      wf cpcv --trades trades.csv --folds 6 --test-folds 2
      wf cpcv --trades trades.csv --folds 8 --test-folds 3 --purge 10
    """
    import math
    _setup_logging(verbose)

    n_combos = math.comb(folds, test_folds)
    click.echo(click.style("\n🔮 CPCV Analysis", fg="magenta", bold=True))
    click.echo(f"  Trades:      {trades}")
    click.echo(f"  k={folds}, k_test={test_folds} → C({folds},{test_folds}) = {n_combos} combinations")
    click.echo(f"  Purge: {purge}, Embargo: {embargo}")

    df = _load_trades(trades)

    # Build CPCV splitter
    from .splits import CPCVSplitter
    cpcv = CPCVSplitter(
        n_splits      = folds,
        n_test_splits = test_folds,
        purge         = purge,
        embargo       = embargo,
    )
    click.echo(f"\n{cpcv.summary()}")

    # Param grid
    pg = _parse_param_grid(param_grid)
    if pg is None:
        # Use a small grid for CPCV (full grid × n_combos can be slow)
        pg = {
            "cf":          [0.001, 0.002, 0.003],
            "bh_form":     [1.2, 1.5, 2.0],
            "bh_collapse": [0.8, 1.0],
            "bh_decay":    [0.92, 0.95, 0.98],
        }

    n_param_combos = 1
    for v in pg.values():
        n_param_combos *= len(v)

    click.echo(f"  Param combos: {n_param_combos}")
    click.echo(f"  Total evaluations: ~{n_combos * n_param_combos}")

    # Build adapter and engine
    from .backtest_adapter import BHStrategyAdapter
    from .engine import WalkForwardEngine

    adapter = BHStrategyAdapter(sym=sym, starting_equity=equity)
    engine  = WalkForwardEngine(
        strategy_fn     = adapter.run_bh_strategy,
        param_grid      = pg,
        n_jobs          = 1,  # CPCV is already parallelised at combination level
        verbose         = verbose,
        metric          = metric,
        starting_equity = equity,
    )

    click.echo(f"\n→ Running {n_combos} CPCV combinations...")
    cpcv_result = engine.run_cpcv(df, cpcv)

    # Results
    click.echo(click.style("\n✓ CPCV Complete", fg="green", bold=True))
    click.echo(f"  Combinations evaluated: {cpcv_result.n_combinations}")
    click.echo(f"  Complete paths:         {cpcv_result.n_paths}")
    click.echo(f"  Deflated Sharpe (DSR):  {cpcv_result.deflated_sharpe:.4f}")
    pbo = cpcv_result.probability_of_overfitting
    pbo_str = f"{pbo:.3f}" if isinstance(pbo, float) and not (pbo != pbo) else "N/A"
    click.echo(f"  PBO:                    {pbo_str}")
    click.echo(f"  PBO Interpretation:     {cpcv_result.pbo_interpretation}")
    click.echo(f"  Mean Path Sharpe:       {cpcv_result.mean_path_sharpe:.4f} ± {cpcv_result.std_path_sharpe:.4f}")

    if cpcv_result.oos_sharpes:
        oos_arr = np.array(cpcv_result.oos_sharpes)
        oos_arr = oos_arr[np.isfinite(oos_arr)]
        if len(oos_arr) > 0:
            click.echo(f"  OOS Sharpe distribution: "
                       f"mean={np.mean(oos_arr):.3f}, "
                       f"std={np.std(oos_arr):.3f}, "
                       f"min={np.min(oos_arr):.3f}, "
                       f"max={np.max(oos_arr):.3f}")

    # Save
    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)

    _save_result_json(cpcv_result, str(out_path / "cpcv_result.json"))

    # Summary CSV
    combo_summary = pd.DataFrame({
        "combo_id":  list(range(len(cpcv_result.is_sharpes))),
        "is_sharpe": cpcv_result.is_sharpes,
        "oos_sharpe": cpcv_result.oos_sharpes,
    })
    combo_summary.to_csv(out_path / "combo_summary.csv", index=False)

    # DSR interpretation
    dsr = cpcv_result.deflated_sharpe
    if np.isfinite(dsr):
        click.echo(f"\n  DSR Interpretation:")
        if dsr > 0.95:
            click.echo(click.style("  ✓ DSR > 0.95: High confidence in genuine edge", fg="green"))
        elif dsr > 0.75:
            click.echo(click.style("  ~ DSR > 0.75: Moderate confidence, some overfitting risk", fg="yellow"))
        else:
            click.echo(click.style("  ✗ DSR < 0.75: Low confidence, likely overfitted", fg="red"))

    click.echo(click.style(f"\n✓ CPCV results saved to: {output}", fg="green"))


# ─────────────────────────────────────────────────────────────────────────────
# wf report
# ─────────────────────────────────────────────────────────────────────────────

@cli.command(name="report")
@click.option("--trades",       "-t", required=True,  help="Path to trades CSV/Parquet/JSON")
@click.option("--output",       "-o", default="results/report/", show_default=True,
              help="Output directory for report")
@click.option("--train",        default=500,  show_default=True, help="Training window")
@click.option("--test",         default=100,  show_default=True, help="Test window")
@click.option("--step",         default=100,  show_default=True, help="Step size")
@click.option("--metric",       default="sharpe", show_default=True,
              type=click.Choice(["sharpe", "sortino", "calmar", "profit_factor"]))
@click.option("--equity",       default=100_000.0, show_default=True)
@click.option("--sym",          default="BTC",     show_default=True)
@click.option("--title",        default="Walk-Forward Analysis Report", show_default=True)
@click.option("--open",         "open_browser", is_flag=True,
              help="Open HTML report in browser after generation")
@click.option("--verbose",      "-v", is_flag=True)
def report_command(
    trades, output, train, test, step, metric, equity, sym, title, open_browser, verbose
) -> None:
    """
    Run walk-forward analysis and generate a comprehensive HTML report.

    Runs a full walk-forward analysis on the provided trades and produces
    an HTML report with all charts, fold breakdowns, and stability analysis.

    \b
    Examples:
      wf report --trades trades.csv --output reports/btc_wf/
      wf report --trades trades.csv --open
    """
    _setup_logging(verbose)

    click.echo(click.style("\n📄 Walk-Forward Report Generator", fg="blue", bold=True))

    df = _load_trades(trades)

    # Run walk-forward
    from .splits import walk_forward_splits
    from .backtest_adapter import BHStrategyAdapter
    from .engine import WalkForwardEngine
    from .report import generate_wf_report, to_console

    n      = len(df)
    splits = walk_forward_splits(n, train, test, step, gap=5)

    if not splits:
        click.echo(click.style("ERROR: No folds could be generated.", fg="red"))
        sys.exit(1)

    click.echo(f"  {len(splits)} folds  |  {n} trades total")

    adapter = BHStrategyAdapter(sym=sym, starting_equity=equity)
    engine  = WalkForwardEngine(
        strategy_fn     = adapter.run_bh_strategy,
        param_grid      = _default_param_grid(),
        metric          = metric,
        starting_equity = equity,
        verbose         = verbose,
    )

    click.echo(f"\n→ Running walk-forward analysis...")
    wf_result = engine.run(df, splits)

    click.echo(f"\n→ Generating report in {output}...")
    report = generate_wf_report(
        wf_result  = wf_result,
        output_dir = output,
        title      = title,
        save_plots = True,
        open_html  = open_browser,
    )

    to_console(report)

    click.echo(click.style(f"\n✓ Report: {report.html_path}", fg="cyan", bold=True))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
