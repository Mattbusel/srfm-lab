"""
research/param_explorer/cli.py
================================
Click command-line interface for the parameter space explorer.

Usage examples
--------------
# Run Sobol sensitivity analysis
python -m research.param_explorer.cli sensitivity \\
    --trades data/crypto_trades.csv \\
    --space bh \\
    --method sobol \\
    --n-samples 1024 \\
    --output results/sensitivity/

# 2-D landscape scan
python -m research.param_explorer.cli landscape \\
    --trades data/crypto_trades.csv \\
    --p1 cf --p2 bh_form \\
    --n1 25 --n2 25 \\
    --output results/landscape/

# Bayesian optimisation
python -m research.param_explorer.cli bayes \\
    --trades data/crypto_trades.csv \\
    --space bh \\
    --n-iter 100 \\
    --acquisition ei \\
    --output results/bayes/

# Full pipeline (all methods)
python -m research.param_explorer.cli full \\
    --trades data/crypto_trades.csv \\
    --space combined \\
    --output results/full/
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import click
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("param_explorer.cli")


# ---------------------------------------------------------------------------
# Lazy imports (avoid importing heavy ML/scipy at import-time)
# ---------------------------------------------------------------------------

def _import_space():
    from research.param_explorer.space import (
        BHParamSpace, LiveTraderParamSpace, CombinedParamSpace, ParamSpace
    )
    return BHParamSpace, LiveTraderParamSpace, CombinedParamSpace, ParamSpace


def _import_sensitivity():
    from research.param_explorer.sensitivity import SensitivityAnalyzer
    return SensitivityAnalyzer


def _import_landscape():
    from research.param_explorer.landscape import ObjectiveLandscape
    return ObjectiveLandscape


def _import_bayes():
    from research.param_explorer.bayesian_opt import (
        BayesianOptimizer, AcquisitionFunction, MOBayesianOptimizer
    )
    return BayesianOptimizer, AcquisitionFunction, MOBayesianOptimizer


def _import_viz():
    from research.param_explorer.visualization import (
        create_sensitivity_dashboard,
        create_landscape_dashboard,
        create_bayesian_opt_dashboard,
        interactive_param_explorer,
    )
    return (
        create_sensitivity_dashboard,
        create_landscape_dashboard,
        create_bayesian_opt_dashboard,
        interactive_param_explorer,
    )


# ---------------------------------------------------------------------------
# Objective function builders
# ---------------------------------------------------------------------------

def _build_default_objective(trades: pd.DataFrame) -> Callable[[Dict[str, Any]], float]:
    """
    Build a simple Sharpe-ratio-based objective from a trades DataFrame.

    The trades DataFrame is expected to have a 'pnl' column (profit/loss per
    trade) and optionally a 'cf' column (cost fraction applied).

    The returned objective function takes a params dict and returns the
    Sharpe ratio of the PnL series after applying cost and parameter-driven
    scaling.

    This is a demonstration objective; real usage should inject a proper
    backtester.
    """
    if "pnl" not in trades.columns:
        logger.warning("Trades DataFrame has no 'pnl' column; using random noise objective.")
        rng = np.random.default_rng(0)

        def _noisy_obj(params: Dict[str, Any]) -> float:
            seed_val = int(sum(hash(str(v)) for v in params.values()) % 2**31)
            rng2 = np.random.default_rng(seed_val)
            return float(rng2.normal(1.0, 0.3))

        return _noisy_obj

    pnl = trades["pnl"].values.astype(float)

    def _sharpe_objective(params: Dict[str, Any]) -> float:
        cf = float(params.get("cf", 0.002))
        bh_form = float(params.get("bh_form", 1.6))
        bh_collapse = float(params.get("bh_collapse", 0.8))
        bh_decay = float(params.get("bh_decay", 0.95))

        # Simulate BH engine effect: amplify signals above bh_form threshold
        # and attenuate positions below bh_collapse (toy model)
        signal_mask = np.abs(pnl) > np.std(pnl) * bh_form
        adj_pnl = np.where(signal_mask, pnl * bh_decay, pnl * bh_collapse)
        adj_pnl = adj_pnl - cf * np.abs(adj_pnl)

        # Apply ensemble weights if present
        w_d3qn = float(params.get("w_d3qn", 0.334))
        w_ddqn = float(params.get("w_ddqn", 0.333))
        w_td3qn = float(params.get("w_td3qn", 0.333))
        total_w = w_d3qn + w_ddqn + w_td3qn
        if total_w > 0:
            w_blend = (w_d3qn + w_ddqn + w_td3qn) / (3 * total_w)
        else:
            w_blend = 1.0
        adj_pnl = adj_pnl * w_blend

        mu = float(np.mean(adj_pnl))
        sigma = float(np.std(adj_pnl, ddof=1))
        if sigma < 1e-10:
            return 0.0
        sharpe = mu / sigma * np.sqrt(252)
        return float(sharpe)

    return _sharpe_objective


def _build_live_trader_objective(trades: pd.DataFrame) -> Callable[[Dict[str, Any]], float]:
    """
    Build an objective function for live-trader parameter tuning.
    """
    if "pnl" not in trades.columns:
        return _build_default_objective(trades)

    pnl = trades["pnl"].values.astype(float)

    def _lt_objective(params: Dict[str, Any]) -> float:
        delta_max = float(params.get("DELTA_MAX_FRAC", 0.7))
        min_trade_frac = float(params.get("MIN_TRADE_FRAC", 0.02))
        min_hold = int(params.get("MIN_HOLD", 3))

        # Filter out trades that don't meet minimum size
        trade_sizes = np.abs(pnl) / (np.abs(pnl).mean() + 1e-8)
        valid = trade_sizes >= min_trade_frac
        if valid.sum() < 5:
            return -10.0

        # Apply delta cap: clip very large trades
        capped_pnl = np.clip(pnl[valid], -delta_max * np.std(pnl), delta_max * np.std(pnl))

        # Simulate holding penalty: reduce noise for longer holds
        hold_factor = 1.0 - 0.05 / max(min_hold, 1)
        capped_pnl = capped_pnl * hold_factor

        mu = float(np.mean(capped_pnl))
        sigma = float(np.std(capped_pnl, ddof=1))
        if sigma < 1e-10:
            return 0.0
        return float(mu / sigma * np.sqrt(252))

    return _lt_objective


# ---------------------------------------------------------------------------
# Space factory
# ---------------------------------------------------------------------------

def _build_space(space_name: str):
    BHParamSpace, LiveTraderParamSpace, CombinedParamSpace, _ = _import_space()
    name = space_name.lower()
    if name == "bh":
        return BHParamSpace()
    elif name in ("lt", "livetrader", "live"):
        return LiveTraderParamSpace()
    elif name == "combined":
        return CombinedParamSpace([BHParamSpace(), LiveTraderParamSpace()])
    else:
        raise click.BadParameter(
            f"Unknown space: {space_name!r}. Use 'bh', 'lt', or 'combined'."
        )


# ---------------------------------------------------------------------------
# Common CLI options
# ---------------------------------------------------------------------------

def _common_options(f):
    f = click.option(
        "--trades", "-t",
        required=False, default=None,
        type=click.Path(exists=False),
        help="Path to trades CSV file.",
    )(f)
    f = click.option(
        "--space", "-s",
        default="bh",
        type=click.Choice(["bh", "lt", "combined"], case_sensitive=False),
        show_default=True,
        help="Parameter space to explore.",
    )(f)
    f = click.option(
        "--output", "-o",
        default="results/param_explorer/",
        type=click.Path(),
        show_default=True,
        help="Output directory.",
    )(f)
    f = click.option(
        "--seed", default=42, show_default=True, type=int, help="Random seed."
    )(f)
    f = click.option(
        "--verbose/--quiet", default=True, show_default=True,
        help="Verbose logging.",
    )(f)
    return f


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.1.0", prog_name="param_explorer")
def cli():
    """
    srfm-lab Parameter Space Explorer

    Explore the sensitivity, landscape, and Bayesian optimisation of
    BH engine and live-trader parameters.
    """
    pass


# ---------------------------------------------------------------------------
# sensitivity command
# ---------------------------------------------------------------------------

@cli.command("sensitivity")
@_common_options
@click.option(
    "--method", "-m",
    default="all",
    type=click.Choice(["oat", "sobol", "morris", "all"], case_sensitive=False),
    show_default=True,
    help="Sensitivity method(s) to run.",
)
@click.option("--n-samples", default=512, show_default=True, type=int,
              help="Sample count for Sobol analysis.")
@click.option("--n-trajectories", default=10, show_default=True, type=int,
              help="Trajectories for Morris screening.")
@click.option("--n-points", default=20, show_default=True, type=int,
              help="Evaluation points for OAT per parameter.")
@click.option("--export-csv/--no-export-csv", default=True, show_default=True,
              help="Export results to CSV.")
@click.option("--export-json/--no-export-json", default=True, show_default=True,
              help="Export results to JSON.")
def sensitivity_cmd(
    trades, space, output, seed, verbose, method, n_samples,
    n_trajectories, n_points, export_csv, export_json,
):
    """Run sensitivity analysis on the parameter space."""
    if verbose:
        click.echo(f"[sensitivity] space={space}, method={method}, seed={seed}")

    # Load trades
    trades_df = _load_trades(trades)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build space + objective
    param_space = _build_space(space)
    if space == "lt":
        obj_fn = _build_live_trader_objective(trades_df)
    else:
        obj_fn = _build_default_objective(trades_df)

    SensitivityAnalyzer = _import_sensitivity()
    analyzer = SensitivityAnalyzer(param_space, obj_fn, verbose=verbose)

    results = {}
    t0 = time.perf_counter()

    if method in ("oat", "all"):
        click.echo("  Running OAT…")
        oat = analyzer.run_oat(n_points=n_points)
        results["oat"] = oat
        if verbose:
            click.echo(f"  OAT done ({oat.total_evals} evals). Top-3:")
            for nm, rng in oat.top_k(3):
                click.echo(f"    {nm}: range={rng:.4g}")

    if method in ("morris", "all"):
        click.echo("  Running Morris…")
        morris = analyzer.run_morris(n_trajectories=n_trajectories)
        results["morris"] = morris
        if verbose:
            click.echo(f"  Morris done ({morris.total_evals} evals). Top-3:")
            for nm, val in morris.top_k(3):
                click.echo(f"    {nm}: μ*={val:.4g}")

    if method in ("sobol", "all"):
        click.echo(f"  Running Sobol (N={n_samples})…")
        sobol = analyzer.run_sobol(n_samples=n_samples, seed=seed)
        results["sobol"] = sobol
        if verbose:
            click.echo(f"  Sobol done ({sobol.total_evals} evals, Var(Y)={sobol.var_y:.4g}). Top-3:")
            for nm, val in sobol.top_k(3):
                click.echo(f"    {nm}: STi={val:.4f}")

    elapsed = time.perf_counter() - t0
    click.echo(f"  Total time: {elapsed:.1f}s")

    # Export
    if export_json:
        _export_sensitivity_json(results, out_dir / "sensitivity_results.json")
    if export_csv:
        _export_sensitivity_csv(results, out_dir)

    # Plots
    (
        create_sensitivity_dashboard,
        _,
        _,
        _,
    ) = _import_viz()
    try:
        fig = create_sensitivity_dashboard(results, save_dir=out_dir / "plots")
        import matplotlib.pyplot as plt
        plt.close(fig)
        click.echo(f"  Plots saved to {out_dir / 'plots'}")
    except Exception as exc:
        logger.warning("Could not create sensitivity dashboard: %s", exc)

    click.echo(f"[sensitivity] Done. Results in {out_dir}")


# ---------------------------------------------------------------------------
# landscape command
# ---------------------------------------------------------------------------

@cli.command("landscape")
@_common_options
@click.option("--p1", required=True, type=str, help="First parameter name.")
@click.option("--p2", required=True, type=str, help="Second parameter name.")
@click.option("--n1", default=20, show_default=True, type=int)
@click.option("--n2", default=20, show_default=True, type=int)
@click.option("--find-basins/--no-find-basins", default=True, show_default=True)
@click.option("--robustness-samples", default=100, show_default=True, type=int)
def landscape_cmd(
    trades, space, output, seed, verbose,
    p1, p2, n1, n2, find_basins, robustness_samples,
):
    """Scan the objective landscape for two parameters."""
    click.echo(f"[landscape] {p1} × {p2}, grid {n1}×{n2}")

    trades_df = _load_trades(trades)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    param_space = _build_space(space)
    if p1 not in param_space:
        raise click.BadParameter(f"Parameter '{p1}' not in space '{space}'.")
    if p2 not in param_space:
        raise click.BadParameter(f"Parameter '{p2}' not in space '{space}'.")

    obj_fn = _build_default_objective(trades_df)

    ObjectiveLandscape = _import_landscape()
    landscape = ObjectiveLandscape(param_space, obj_fn, maximise=True)

    t0 = time.perf_counter()
    grid = landscape.scan_2d(p1, p2, n1=n1, n2=n2)
    elapsed = time.perf_counter() - t0
    click.echo(f"  Grid scan complete in {elapsed:.1f}s. Best: {grid.global_best_value:.4g}")

    basins = []
    if find_basins:
        basins = landscape.find_basins(grid)
        click.echo(f"  Found {len(basins)} basins.")
        for k, b in enumerate(basins[:5]):
            click.echo(f"    B{k+1}: {b}")

    rough = landscape.roughness_index(grid)
    flat = landscape.flatness_score(grid)
    click.echo(f"  Roughness: {rough:.4g}, Flatness (10%): {flat:.1%}")

    best_full = dict(param_space.defaults)
    best_full.update(grid.global_best_params)
    rob = landscape.robustness_score(best_full, n_perturbations=robustness_samples, seed=seed)
    click.echo(f"  Robustness score: {rob:.4g}")

    # Export
    results_json = {
        "p1": p1, "p2": p2,
        "global_best_params": grid.global_best_params,
        "global_best_value": float(grid.global_best_value),
        "roughness": float(rough),
        "flatness_10pct": float(flat),
        "robustness": float(rob),
        "n_basins": len(basins),
        "basins": [
            {"p1": b.p1_center, "p2": b.p2_center, "value": b.peak_value,
             "area": b.area_fraction}
            for b in basins
        ],
    }
    json_path = out_dir / f"landscape_{p1}_{p2}.json"
    json_path.write_text(json.dumps(results_json, indent=2))
    click.echo(f"  Results JSON: {json_path}")

    # Plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    (_, create_landscape_dashboard, _, _) = _import_viz()
    try:
        import matplotlib.pyplot as plt
        fig = create_landscape_dashboard(grid, save_dir=plots_dir, basins=basins)
        plt.close(fig)
        click.echo(f"  Plots saved to {plots_dir}")
    except Exception as exc:
        logger.warning("Could not create landscape dashboard: %s", exc)

    click.echo(f"[landscape] Done. Results in {out_dir}")


# ---------------------------------------------------------------------------
# bayes command
# ---------------------------------------------------------------------------

@cli.command("bayes")
@_common_options
@click.option("--n-iter", default=50, show_default=True, type=int,
              help="BO iterations after initial design.")
@click.option("--n-init", default=10, show_default=True, type=int,
              help="Initial Sobol design size.")
@click.option(
    "--acquisition", "-a", "acq",
    default="ei",
    type=click.Choice(["ei", "ucb", "pi", "thompson"], case_sensitive=False),
    show_default=True,
    help="Acquisition function.",
)
@click.option("--export-csv/--no-export-csv", default=True, show_default=True)
@click.option("--export-json/--no-export-json", default=True, show_default=True)
@click.option("--multi-objective/--single-objective", default=False, show_default=True,
              help="Run multi-objective BO (Sharpe + max-drawdown).")
def bayes_cmd(
    trades, space, output, seed, verbose,
    n_iter, n_init, acq, export_csv, export_json, multi_objective,
):
    """Run Bayesian optimisation over the parameter space."""
    click.echo(f"[bayes] space={space}, n_init={n_init}, n_iter={n_iter}, acq={acq}")

    trades_df = _load_trades(trades)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    param_space = _build_space(space)
    obj_fn = _build_default_objective(trades_df)

    BayesianOptimizer, AcquisitionFunction, MOBayesianOptimizer = _import_bayes()
    acq_enum = AcquisitionFunction(acq)

    t0 = time.perf_counter()

    if multi_objective:
        # Second objective: negative drawdown (toy version)
        if "pnl" in trades_df.columns:
            pnl_arr = trades_df["pnl"].values.astype(float)
        else:
            pnl_arr = np.zeros(10)

        def _drawdown_obj(params: Dict[str, Any]) -> float:
            cf = float(params.get("cf", 0.002))
            adj = pnl_arr * (1 - cf)
            cum = np.cumsum(adj)
            if len(cum) == 0:
                return 0.0
            running_max = np.maximum.accumulate(cum)
            dd = float(np.min(running_max - cum))
            return -dd  # negative drawdown (higher = smaller drawdown)

        mo_opt = MOBayesianOptimizer(
            param_space=param_space,
            objective_fns=[obj_fn, _drawdown_obj],
            objective_names=["Sharpe", "neg_maxDD"],
            n_init=n_init,
            seed=seed,
        )
        mo_result = mo_opt.run(n_iter=n_iter, verbose=verbose)
        elapsed = time.perf_counter() - t0
        click.echo(f"  MO-BO done in {elapsed:.1f}s. Pareto front size: {len(mo_result.pareto_params)}")

        if export_json:
            json_path = out_dir / "mo_bayes_result.json"
            json_path.write_text(json.dumps(mo_result.to_dict(), indent=2))
            click.echo(f"  JSON: {json_path}")

        from research.param_explorer.bayesian_opt import plot_pareto_front
        import matplotlib.pyplot as plt
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig = plot_pareto_front(mo_result, save_path=plots_dir / "pareto_front.png")
        plt.close(fig)

    else:
        opt = BayesianOptimizer(
            param_space=param_space,
            objective_fn=obj_fn,
            acquisition=acq_enum,
            n_init=n_init,
            seed=seed,
        )

        def _cb(iteration: int, params: Dict[str, Any], score: float) -> None:
            if verbose and iteration % max(1, n_iter // 10) == 0:
                click.echo(f"  iter {iteration}/{n_iter}: score={score:.4g}")

        result = opt.run(n_iter=n_iter, verbose=False, callback=_cb)
        elapsed = time.perf_counter() - t0

        click.echo(f"  BO done in {elapsed:.1f}s. Best score: {result.best_score:.4g}")
        click.echo(f"  Best params:")
        for k, v in result.best_params.items():
            click.echo(f"    {k} = {_fmt_val(v)}")

        if export_json:
            json_path = out_dir / "bayes_result.json"
            json_path.write_text(json.dumps(result.to_dict(), indent=2))
            click.echo(f"  JSON: {json_path}")

        if export_csv:
            csv_path = out_dir / "bayes_history.csv"
            _export_bayes_csv(result, csv_path)
            click.echo(f"  CSV history: {csv_path}")

        (_, _, create_bayesian_opt_dashboard, _) = _import_viz()
        import matplotlib.pyplot as plt
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        try:
            fig = create_bayesian_opt_dashboard(result, save_dir=plots_dir)
            plt.close(fig)
            click.echo(f"  Plots saved to {plots_dir}")
        except Exception as exc:
            logger.warning("Could not create BO dashboard: %s", exc)

    click.echo(f"[bayes] Done. Results in {out_dir}")


# ---------------------------------------------------------------------------
# full command
# ---------------------------------------------------------------------------

@cli.command("full")
@_common_options
@click.option("--n-samples-sobol", default=512, show_default=True, type=int)
@click.option("--n-iter-bayes", default=50, show_default=True, type=int)
@click.option("--n-init-bayes", default=10, show_default=True, type=int)
@click.option("--scan-pairs/--no-scan-pairs", default=False, show_default=True,
              help="Run 2-D landscape for all parameter pairs (slow for large spaces).")
def full_cmd(
    trades, space, output, seed, verbose,
    n_samples_sobol, n_iter_bayes, n_init_bayes, scan_pairs,
):
    """Run the full analysis pipeline: sensitivity + landscape + Bayes."""
    click.echo("[full] Starting full pipeline…")

    trades_df = _load_trades(trades)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    param_space = _build_space(space)
    obj_fn = _build_default_objective(trades_df)

    summary: Dict[str, Any] = {
        "space": space,
        "n_params": param_space.n_dims,
        "param_names": param_space.names,
    }

    # --- 1. Sensitivity ---
    click.echo("[full] Step 1/3: Sensitivity analysis…")
    SensitivityAnalyzer = _import_sensitivity()
    analyzer = SensitivityAnalyzer(param_space, obj_fn, verbose=False)
    sens_results = analyzer.run_all(
        sobol_n_samples=n_samples_sobol,
        morris_n_trajectories=10,
        oat_n_points=20,
    )
    summary["sensitivity"] = {
        "sobol_top3": sens_results["sobol"].top_k(3),
        "morris_top3": sens_results["morris"].top_k(3),
        "oat_top3": sens_results["oat"].top_k(3),
    }

    (create_sens_dash, _, _, _) = _import_viz()
    try:
        import matplotlib.pyplot as plt
        fig = create_sens_dash(sens_results, save_dir=out_dir / "sensitivity")
        plt.close(fig)
    except Exception as exc:
        logger.warning("Sensitivity dashboard failed: %s", exc)

    click.echo(f"  Sensitivity done. Sobol top: {sens_results['sobol'].top_k(1)}")

    # --- 2. Landscape (first 2 params by importance) ---
    click.echo("[full] Step 2/3: Landscape scan…")
    ObjectiveLandscape = _import_landscape()
    top2 = [nm for nm, _ in sens_results["sobol"].top_k(2)]
    if len(top2) < 2:
        top2 = param_space.names[:2]

    landscape = ObjectiveLandscape(param_space, obj_fn)
    grid = landscape.scan_2d(top2[0], top2[1], n1=20, n2=20)
    basins = landscape.find_basins(grid)
    rough = landscape.roughness_index(grid)
    flat = landscape.flatness_score(grid)

    summary["landscape"] = {
        "p1": top2[0], "p2": top2[1],
        "best_value": float(grid.global_best_value),
        "roughness": float(rough),
        "flatness_10pct": float(flat),
        "n_basins": len(basins),
    }

    (_, create_land_dash, _, _) = _import_viz()
    try:
        import matplotlib.pyplot as plt
        fig = create_land_dash(grid, save_dir=out_dir / "landscape", basins=basins)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Landscape dashboard failed: %s", exc)

    click.echo(f"  Landscape done. Best={grid.global_best_value:.4g}, basins={len(basins)}")

    # --- 3. Bayesian opt ---
    click.echo(f"[full] Step 3/3: Bayesian optimisation (n_iter={n_iter_bayes})…")
    BayesianOptimizer, AcquisitionFunction, _ = _import_bayes()
    opt = BayesianOptimizer(
        param_space=param_space,
        objective_fn=obj_fn,
        acquisition=AcquisitionFunction.EI,
        n_init=n_init_bayes,
        seed=seed,
    )
    bayes_result = opt.run(n_iter=n_iter_bayes, verbose=verbose)
    summary["bayes_opt"] = {
        "best_score": float(bayes_result.best_score),
        "best_params": bayes_result.best_params,
        "n_evals": bayes_result.n_total_evals,
    }

    (_, _, create_bayes_dash, _) = _import_viz()
    try:
        import matplotlib.pyplot as plt
        fig = create_bayes_dash(bayes_result, save_dir=out_dir / "bayesopt")
        plt.close(fig)
    except Exception as exc:
        logger.warning("BO dashboard failed: %s", exc)

    click.echo(f"  BO done. Best score: {bayes_result.best_score:.4g}")

    # --- Save summary ---
    summary_path = out_dir / "full_pipeline_summary.json"
    summary_serialisable = _make_serialisable(summary)
    summary_path.write_text(json.dumps(summary_serialisable, indent=2))
    click.echo(f"\n[full] Pipeline complete. Summary at {summary_path}")

    # Print a brief human-readable summary
    click.echo("\n" + "=" * 50)
    click.echo(" Full Pipeline Summary")
    click.echo("=" * 50)
    click.echo(f"  Space: {space} ({param_space.n_dims} params)")
    click.echo(f"  Best score (BO): {bayes_result.best_score:.4g}")
    click.echo("  Best params:")
    for k, v in bayes_result.best_params.items():
        click.echo(f"    {k} = {_fmt_val(v)}")
    click.echo(f"  Top sensitive params:")
    for nm, val in sens_results["sobol"].top_k(3):
        click.echo(f"    {nm}: STi={val:.3f}")
    click.echo("=" * 50)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _load_trades(path: Optional[str]) -> pd.DataFrame:
    """Load trades CSV or return an empty DataFrame."""
    if path is None:
        logger.info("No trades file specified; using empty DataFrame.")
        return pd.DataFrame(columns=["pnl"])
    p = Path(path)
    if not p.exists():
        logger.warning("Trades file not found: %s; using empty DataFrame.", path)
        return pd.DataFrame(columns=["pnl"])
    try:
        df = pd.read_csv(p)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df
    except Exception as exc:
        logger.error("Failed to load trades: %s", exc)
        return pd.DataFrame(columns=["pnl"])


def _export_sensitivity_json(results: Dict[str, Any], path: Path) -> None:
    """Serialise sensitivity results to JSON."""
    out: Dict[str, Any] = {}
    if "oat" in results:
        out["oat"] = results["oat"].to_dict()
    if "sobol" in results:
        out["sobol"] = results["sobol"].to_dict()
    if "morris" in results:
        out["morris"] = results["morris"].to_dict()
    path.write_text(json.dumps(out, indent=2))
    logger.info("Sensitivity JSON saved to %s", path)


def _export_sensitivity_csv(results: Dict[str, Any], out_dir: Path) -> None:
    """Export sensitivity tables to CSV."""
    if "sobol" in results:
        sobol = results["sobol"]
        names = list(sobol.Si.keys())
        df = pd.DataFrame({
            "param": names,
            "Si": [sobol.Si[n] for n in names],
            "STi": [sobol.STi[n] for n in names],
            "Si_conf": [sobol.Si_conf[n] for n in names],
            "STi_conf": [sobol.STi_conf[n] for n in names],
        }).sort_values("STi", ascending=False)
        df.to_csv(out_dir / "sobol_indices.csv", index=False)

    if "morris" in results:
        morris = results["morris"]
        names = list(morris.mu_star.keys())
        df_m = pd.DataFrame({
            "param": names,
            "mu": [morris.mu[n] for n in names],
            "mu_star": [morris.mu_star[n] for n in names],
            "sigma": [morris.sigma[n] for n in names],
            "rank": [morris.sensitivity_rank[n] for n in names],
        }).sort_values("rank")
        df_m.to_csv(out_dir / "morris_effects.csv", index=False)

    if "oat" in results:
        oat = results["oat"]
        df_o = pd.DataFrame({
            "param": oat.param_names,
            "range": [oat.sensitivity_range[n] for n in oat.param_names],
            "std": [oat.sensitivity_std[n] for n in oat.param_names],
            "rank": [oat.sensitivity_rank[n] for n in oat.param_names],
        }).sort_values("rank")
        df_o.to_csv(out_dir / "oat_sensitivity.csv", index=False)


def _export_bayes_csv(result, path: Path) -> None:
    """Export BO history to CSV."""
    param_names = list(result.best_params.keys())
    rows = []
    for i, (params, score) in enumerate(zip(result.history_params, result.history_y)):
        row = {"iteration": i + 1, "score": float(score), "running_best": float(result.convergence[i])}
        row.update({k: _fmt_val(v) for k, v in params.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _fmt_val(v: Any) -> str:
    try:
        return f"{float(v):.6g}"
    except Exception:
        return str(v)


def _make_serialisable(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
