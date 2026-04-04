"""
research/regime_lab/cli.py
============================
Click CLI for the Regime Lab.

Commands
--------
regime detect    — detect regimes from a price CSV
regime simulate  — run regime-aware Monte Carlo
regime stress    — run all historical stress scenarios against a trade file
regime calibrate — calibrate regime model to historical data
regime report    — full pipeline: detect + stress + MC + report
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Ensure parent packages are on path when running as script
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    import click  # type: ignore
except ImportError:
    # Minimal stub so the module is importable without click installed
    class _FakeClick:  # type: ignore
        class group:
            def __call__(self, *a, **kw):
                def decorator(f):
                    return f
                return decorator
            @staticmethod
            def command(*a, **kw):
                def decorator(f):
                    return f
                return decorator
            @staticmethod
            def add_command(*a, **kw):
                pass

        command   = group
        option    = group
        argument  = group
        echo      = print
        style     = lambda s, **k: s
        Path      = str

        @staticmethod
        def pass_context(f):
            return f

    click = _FakeClick()  # type: ignore
    _HAS_CLICK = False
else:
    _HAS_CLICK = True

logger = logging.getLogger("regime_lab.cli")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s")


# ===========================================================================
# Helpers
# ===========================================================================

def _load_prices(path: str) -> np.ndarray:
    """Load a price CSV.  Accepts close/Close/price/Price column."""
    df = pd.read_csv(path)
    for col in ("close", "Close", "price", "Price", "adj_close", "Adj Close"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    # Fallback: last numeric column
    numeric = df.select_dtypes(include=[float, int]).columns.tolist()
    if not numeric:
        raise ValueError(f"Cannot find price column in {path}")
    return df[numeric[-1]].to_numpy(dtype=float)


def _load_trades(path: str) -> pd.DataFrame:
    """Load a trade CSV."""
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return pd.read_csv(path)


def _save_output(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    p = os.path.join(output_dir, filename)
    df.to_csv(p, index=False)
    return p


# ===========================================================================
# Main group
# ===========================================================================

if _HAS_CLICK:
    @click.group(name="regime")
    @click.option("--verbose", "-v", is_flag=True, default=False,
                  help="Enable DEBUG-level logging.")
    @click.pass_context
    def regime_group(ctx: click.Context, verbose: bool) -> None:
        """Regime Simulation Lab — detect, simulate, stress, calibrate, report."""
        ctx.ensure_object(dict)
        ctx.obj["verbose"] = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    # ===========================================================================
    # regime detect
    # ===========================================================================

    @regime_group.command("detect")
    @click.option("--prices",  "-p", required=True, type=click.Path(exists=True),
                  help="CSV file with close prices.")
    @click.option("--method",  "-m", default="rolling_vol",
                  type=click.Choice(["rolling_vol", "trend", "hmm", "changepoint", "ensemble"],
                                    case_sensitive=False),
                  help="Regime detection method (default: rolling_vol).")
    @click.option("--output", "-o", default="results/regime_lab/detected_regimes.csv",
                  help="Output CSV path.")
    @click.option("--min-duration", default=5, type=int,
                  help="Minimum regime episode length in bars (smoothing).")
    @click.pass_context
    def cmd_detect(ctx: click.Context, prices: str, method: str,
                   output: str, min_duration: int) -> None:
        """Detect market regimes from a price file."""
        click.echo(click.style(f"[detect] Loading prices from {prices}...", fg="cyan"))
        price_arr = _load_prices(prices)
        log_rets  = np.diff(np.log(np.where(price_arr > 0, price_arr, 1e-10)))
        log_rets  = np.concatenate([[0.0], log_rets])

        from research.regime_lab.detector import build_detector, smooth_regime_series

        det = build_detector(method)

        if method == "hmm":
            det.fit(log_rets)
            labels = det.decode_states(det.predict(log_rets))
        elif method in ("rolling_vol",):
            labels = det.detect(price_arr)
        elif method == "changepoint":
            labels = det.segment_regimes(log_rets)
        elif method == "trend":
            labels = det.detect(closes=price_arr)
        else:
            labels = det.predict(log_rets, prices=price_arr)

        if min_duration > 1:
            labels = smooth_regime_series(labels, min_duration=min_duration)

        df = pd.DataFrame({"regime": labels, "price": price_arr[:len(labels)]})
        # Regime distribution
        for r in ("BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"):
            pct = float(np.sum(labels == r)) / max(len(labels), 1) * 100
            click.echo(f"  {r:<12}  {pct:5.1f}%")

        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
        df.to_csv(output, index=True, index_label="bar")
        click.echo(click.style(f"[detect] Saved → {output}", fg="green"))

    # ===========================================================================
    # regime simulate
    # ===========================================================================

    @regime_group.command("simulate")
    @click.option("--starting-equity",  default=1_000_000.0,  type=float,
                  help="Starting portfolio equity (default 1,000,000).")
    @click.option("--months",           default=12,           type=int,
                  help="Simulation horizon in months (default 12).")
    @click.option("--n-sims",           default=10_000,       type=int,
                  help="Number of Monte Carlo paths (default 10,000).")
    @click.option("--prices",           default=None, type=click.Path(exists=True),
                  help="Optional price CSV for BH-physics mode.")
    @click.option("--output",           default="results/regime_lab/mc_result.csv",
                  help="Output CSV for simulation summary.")
    @click.option("--plot",             is_flag=True, default=False,
                  help="Save equity path chart.")
    @click.option("--seed",             default=42, type=int,
                  help="RNG seed.")
    @click.pass_context
    def cmd_simulate(ctx: click.Context, starting_equity: float, months: int,
                     n_sims: int, prices: Optional[str], output: str,
                     plot: bool, seed: int) -> None:
        """Run regime-aware Monte Carlo simulation."""
        click.echo(click.style(
            f"[simulate] {n_sims:,} paths × {months} months, equity={starting_equity:,.0f}",
            fg="cyan"))

        from research.regime_lab.simulator import RegimeMCSim, plot_regime_mc_paths

        sim = RegimeMCSim(n_sims=n_sims)

        if prices:
            p_arr = _load_prices(prices)
            result = sim.run_with_bh_physics(p_arr, starting_equity=starting_equity,
                                              n_months=months, seed=seed)
        else:
            result = sim.run(starting_equity=starting_equity, n_months=months, seed=seed)

        click.echo(f"  Blowup rate    : {result.blowup_rate*100:.2f}%")
        click.echo(f"  Median equity  : ${result.median_equity:,.0f}")
        click.echo(f"  p5 equity      : ${result.pct_5:,.0f}")
        click.echo(f"  p95 equity     : ${result.pct_95:,.0f}")
        click.echo(f"  Max DD (p95)   : {result.max_dd_p95*100:.1f}%")
        click.echo(f"  Sharpe         : {result.sharpe_ratio():.3f}")

        # Per-regime breakdown
        click.echo("\n  Per-regime breakdown:")
        for r, stats in result.regime_stats.items():
            click.echo(f"    {r:<12}  "
                       f"time={stats['mean_time_pct']:.1f}%  "
                       f"pnl_share=${stats['mean_pnl_share']:,.0f}  "
                       f"bh_rate={stats['mean_bh_fire_rate']:.4f}")

        # Save summary
        summary_df = result.to_summary_df()
        p = _save_output(summary_df, os.path.dirname(output), os.path.basename(output))
        click.echo(click.style(f"[simulate] Saved → {p}", fg="green"))

        if plot:
            chart_path = os.path.join(os.path.dirname(output), "mc_paths.png")
            plot_regime_mc_paths(result, save_path=chart_path)
            click.echo(click.style(f"[simulate] Chart  → {chart_path}", fg="green"))

    # ===========================================================================
    # regime stress
    # ===========================================================================

    @regime_group.command("stress")
    @click.option("--trades", "-t", required=True, type=click.Path(exists=True),
                  help="Trade CSV or JSON file.")
    @click.option("--output", "-o", default="results/stress/",
                  help="Output directory for stress results.")
    @click.option("--starting-equity", default=1_000_000.0, type=float,
                  help="Starting equity for blowup calculation.")
    @click.option("--asset-class", default=None,
                  type=click.Choice(["crypto", "equity", "rates", "mixed"]),
                  help="Filter scenarios by asset class.")
    @click.option("--plot", is_flag=True, default=False,
                  help="Save stress bar chart.")
    @click.pass_context
    def cmd_stress(ctx: click.Context, trades: str, output: str,
                   starting_equity: float, asset_class: Optional[str],
                   plot: bool) -> None:
        """Run all historical stress scenarios against a trade set."""
        click.echo(click.style(f"[stress] Loading trades from {trades}...", fg="cyan"))
        trade_df = _load_trades(trades)
        trade_list = trade_df.to_dict(orient="records")

        from research.regime_lab.stress import (
            StressTester, stress_report, plot_stress_results, scenario_by_asset_class)

        tester = StressTester(starting_equity=starting_equity)

        if asset_class:
            from research.regime_lab.stress import SCENARIO_CATALOGUE
            filtered = scenario_by_asset_class(SCENARIO_CATALOGUE, asset_class)
            results  = [tester.run_scenario(trade_list, sc) for sc in filtered]
        else:
            results = tester.run_all_scenarios(trade_list)

        df = stress_report(results)
        click.echo("\n  Top-5 worst scenarios:")
        cols = ["scenario", "peak_drawdown_pct", "pct_equity_impact", "blowup"]
        click.echo(df.head(5)[cols].to_string(index=False))

        blowups = [r.scenario.name for r in results if r.blowup]
        if blowups:
            click.echo(click.style(f"\n  BLOWUP scenarios: {blowups}", fg="red"))
        else:
            click.echo(click.style("\n  No blowup scenarios.", fg="green"))

        p = _save_output(df, output, "stress_results.csv")
        click.echo(click.style(f"[stress] Saved → {p}", fg="green"))

        if plot:
            chart_path = os.path.join(output, "stress_results.png")
            plot_stress_results(results, save_path=chart_path)
            click.echo(click.style(f"[stress] Chart → {chart_path}", fg="green"))

    # ===========================================================================
    # regime calibrate
    # ===========================================================================

    @regime_group.command("calibrate")
    @click.option("--prices", "-p", required=True, type=click.Path(exists=True),
                  help="Price CSV file.")
    @click.option("--trades", "-t", default=None, type=click.Path(exists=True),
                  help="Optional trade CSV for strategy calibration.")
    @click.option("--model", "-m", default="markov",
                  type=click.Choice(["markov", "garch", "heston", "jump", "bootstrap"],
                                    case_sensitive=False),
                  help="Generator model to calibrate (default: markov).")
    @click.option("--output", "-o", default="results/regime_lab/calibration.json",
                  help="Output JSON path.")
    @click.option("--window", default=252, type=int,
                  help="Rolling window for parameter estimates (default 252).")
    @click.pass_context
    def cmd_calibrate(ctx: click.Context, prices: str, trades: Optional[str],
                      model: str, output: str, window: int) -> None:
        """Calibrate regime model parameters to historical data."""
        click.echo(click.style(f"[calibrate] Model={model}, window={window}", fg="cyan"))
        price_arr = _load_prices(prices)

        from research.regime_lab.generator import calibrate_to_history
        from research.regime_lab.calibration import validate_calibration, rolling_calibration
        from research.regime_lab.detector import RollingVolRegimeDetector

        log_rets = np.diff(np.log(np.where(price_arr > 0, price_arr, 1e-10)))
        log_rets = np.concatenate([[0.0], log_rets])

        # Detect regimes
        det = RollingVolRegimeDetector()
        regime_labels = det.detect(price_arr)

        gen = calibrate_to_history(price_arr, model=model, regime_labels=regime_labels)

        # Validate
        gen_result = gen.generate(len(log_rets), seed=42) if hasattr(gen, "generate") else None
        cal_report = None
        if gen_result is not None:
            gen_rets = gen_result.returns if hasattr(gen_result, "returns") else log_rets
            cal_report = validate_calibration(gen_rets, log_rets)
            click.echo(f"  Calibration overall score: {cal_report.overall_score:.3f}")
            click.echo(f"  KS p-value              : {cal_report.ks_pvalue:.4f}")

        # Rolling calibration
        roll_df = rolling_calibration(price_arr, window=window)
        click.echo(f"  Rolling windows computed: {len(roll_df)}")

        # Save
        result_dict: dict = {
            "model":    model,
            "n_bars":   len(price_arr),
            "cal_report": cal_report.to_dict() if cal_report else {},
        }
        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
        with open(output, "w") as f:
            json.dump(result_dict, f, indent=2)
        click.echo(click.style(f"[calibrate] Saved → {output}", fg="green"))

        roll_path = output.replace(".json", "_rolling.csv")
        roll_df.to_csv(roll_path)
        click.echo(click.style(f"[calibrate] Rolling params → {roll_path}", fg="green"))

    # ===========================================================================
    # regime report
    # ===========================================================================

    @regime_group.command("report")
    @click.option("--prices",          "-p",  required=True, type=click.Path(exists=True),
                  help="Price CSV file.")
    @click.option("--trades",          "-t",  default=None,  type=click.Path(exists=True),
                  help="Optional trade CSV.")
    @click.option("--output",          "-o",  default="results/regime_lab/",
                  help="Output directory.")
    @click.option("--starting-equity", default=1_000_000.0, type=float)
    @click.option("--n-sims",          default=2_000,        type=int)
    @click.option("--months",          default=12,           type=int)
    @click.option("--detector",        default="rolling_vol",
                  type=click.Choice(["rolling_vol", "trend", "hmm", "changepoint", "ensemble"],
                                    case_sensitive=False))
    @click.option("--html",            is_flag=True, default=False,
                  help="Save an HTML report.")
    @click.option("--no-mc",           is_flag=True, default=False)
    @click.option("--no-stress",       is_flag=True, default=False)
    @click.option("--seed",            default=42, type=int)
    @click.pass_context
    def cmd_report(ctx: click.Context, prices: str, trades: Optional[str],
                   output: str, starting_equity: float, n_sims: int,
                   months: int, detector: str, html: bool,
                   no_mc: bool, no_stress: bool, seed: int) -> None:
        """Full regime-lab pipeline: detect + analyse + stress + MC + report."""
        click.echo(click.style("[report] Starting full pipeline...", fg="cyan"))

        price_arr  = _load_prices(prices)
        trade_df   = _load_trades(trades) if trades else pd.DataFrame()

        from research.regime_lab.report import generate_regime_report, to_html, to_console

        rep = generate_regime_report(
            trades=trade_df,
            price_history=price_arr,
            output_dir=output,
            starting_equity=starting_equity,
            run_mc=not no_mc,
            run_stress=not no_stress,
            n_mc_sims=n_sims,
            n_mc_months=months,
            mc_seed=seed,
            detector_method=detector,
        )

        to_console(rep)

        if html:
            html_path = os.path.join(output, "regime_lab_report.html")
            to_html(rep, html_path)
            click.echo(click.style(f"[report] HTML → {html_path}", fg="green"))

        click.echo(click.style("[report] Done.", fg="green"))

else:
    # If click is not installed, expose a minimal callable
    def regime_group(*args: object, **kwargs: object) -> None:  # type: ignore
        print("ERROR: 'click' package is required for the CLI. "
              "Install with:  pip install click")


# ===========================================================================
# Entry-point
# ===========================================================================

if __name__ == "__main__":
    if not _HAS_CLICK:
        print("ERROR: 'click' package not installed. Run:  pip install click")
        sys.exit(1)
    regime_group()
