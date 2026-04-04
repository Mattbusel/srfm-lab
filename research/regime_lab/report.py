"""
research/regime_lab/report.py
===============================
Regime Lab Report generation — HTML and console output.

Classes
-------
RegimeLabReport — dataclass aggregating all analysis outputs

Functions
---------
generate_regime_report(trades, price_history, output_dir) -> RegimeLabReport
to_html(report, path)
to_console(report)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BULL     = "BULL"
BEAR     = "BEAR"
SIDEWAYS = "SIDEWAYS"
HIGH_VOL = "HIGH_VOL"
REGIMES  = (BULL, BEAR, SIDEWAYS, HIGH_VOL)


# ===========================================================================
# 1. RegimeLabReport dataclass
# ===========================================================================

@dataclass
class RegimeLabReport:
    """Aggregated regime-lab analysis report."""

    # Metadata
    generated_at:    str = ""
    n_bars:          int = 0
    n_trades:        int = 0

    # Regime detection summary
    regime_labels:       Optional[np.ndarray] = None
    regime_distribution: Dict[str, float]      = field(default_factory=dict)
    # {regime: fraction of bars}

    # Transition analysis
    transition_matrix:   Optional[np.ndarray] = None
    stationary_dist:     Optional[np.ndarray] = None
    duration_stats_df:   Optional[pd.DataFrame] = None

    # Per-regime performance
    regime_perf_df:      Optional[pd.DataFrame] = None
    # columns: regime, n_trades, win_rate, mean_pnl, total_pnl, sharpe

    # Stress test
    stress_results_df:   Optional[pd.DataFrame] = None
    worst_case_summary:  Optional[Dict[str, Any]] = None

    # MC simulation summary
    mc_summary:          Optional[Dict[str, Any]] = None
    regime_mc_stats:     Optional[pd.DataFrame]   = None

    # Calibration
    calibration_report:  Optional[Dict[str, Any]] = None

    # File paths to generated artefacts
    artefacts:           Dict[str, str] = field(default_factory=dict)
    # {name: file_path}

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  REGIME LAB REPORT",
            "=" * 60,
            f"  Generated  : {self.generated_at}",
            f"  Price bars : {self.n_bars:,}",
            f"  Trades     : {self.n_trades:,}",
        ]
        if self.regime_distribution:
            lines.append("\n  REGIME DISTRIBUTION")
            lines.append("  " + "-" * 40)
            for r, frac in self.regime_distribution.items():
                lines.append(f"    {r:<12} {frac*100:5.1f}%")
        if self.regime_perf_df is not None and not self.regime_perf_df.empty:
            lines.append("\n  PER-REGIME PERFORMANCE")
            lines.append("  " + "-" * 40)
            lines.append("  " + self.regime_perf_df.to_string(index=False))
        if self.worst_case_summary:
            wc = self.worst_case_summary
            lines.append("\n  WORST-CASE STRESS")
            lines.append("  " + "-" * 40)
            lines.append(f"    Worst scenario : {wc.get('worst_scenario', 'N/A')}")
            lines.append(f"    Worst P&L      : ${wc.get('worst_pnl', 0):,.0f}")
            lines.append(f"    VaR (5%)       : ${wc.get('var_estimate', 0):,.0f}")
            lines.append(f"    CVaR (5%)      : ${wc.get('cvar_estimate', 0):,.0f}")
        if self.mc_summary:
            mc = self.mc_summary
            lines.append("\n  MONTE CARLO SUMMARY")
            lines.append("  " + "-" * 40)
            lines.append(f"    Blowup rate    : {mc.get('blowup_rate', 0)*100:.1f}%")
            lines.append(f"    Median equity  : ${mc.get('median_equity', 0):,.0f}")
            lines.append(f"    p5 equity      : ${mc.get('pct_5', 0):,.0f}")
            lines.append(f"    p95 equity     : ${mc.get('pct_95', 0):,.0f}")
            lines.append(f"    Max DD p95     : {mc.get('max_dd_p95', 0)*100:.1f}%")
        lines.append("=" * 60)
        return "\n".join(lines)


# ===========================================================================
# 2. generate_regime_report
# ===========================================================================

def generate_regime_report(
        trades: Any,
        price_history: np.ndarray | pd.Series,
        output_dir: str = "results/regime_lab",
        starting_equity: float = 1_000_000.0,
        run_mc: bool = True,
        run_stress: bool = True,
        run_calibration: bool = True,
        n_mc_sims: int = 2_000,
        n_mc_months: int = 12,
        mc_seed: int = 42,
        detector_method: str = "rolling_vol",
) -> RegimeLabReport:
    """
    Full regime-lab pipeline: detect → analyse → stress → MC → report.

    Parameters
    ----------
    trades          : trade records (list of dicts or pd.DataFrame)
    price_history   : 1-D price array
    output_dir      : directory to save charts and artefacts
    starting_equity : initial equity for stress/MC
    run_mc          : whether to run Monte Carlo simulation
    run_stress      : whether to run stress tests
    run_calibration : whether to run calibration validation
    n_mc_sims       : number of MC paths
    n_mc_months     : MC horizon
    mc_seed         : RNG seed
    detector_method : regime detector to use ('rolling_vol', 'trend', etc.)

    Returns
    -------
    RegimeLabReport
    """
    import datetime
    from research.regime_lab.stress import _extract_trades, _trade_pnl, _trade_regime

    os.makedirs(output_dir, exist_ok=True)

    prices    = np.asarray(price_history, dtype=float)
    log_rets  = np.diff(np.log(np.where(prices > 0, prices, 1e-10)))
    log_rets  = np.concatenate([[0.0], log_rets])
    T         = len(prices)
    trade_list = _extract_trades(trades)

    report = RegimeLabReport(
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        n_bars=T,
        n_trades=len(trade_list),
    )

    # ------------------------------------------------------------------ #
    # Step 1: Regime detection
    # ------------------------------------------------------------------ #
    logger.info("Detecting regimes with method='%s'...", detector_method)
    from research.regime_lab.detector import build_detector
    det     = build_detector(detector_method)

    if detector_method == "hmm":
        det.fit(log_rets)
        regime_labels = det.decode_states(det.predict(log_rets))
    elif detector_method in ("rolling_vol", "changepoint"):
        if hasattr(det, "detect"):
            regime_labels = det.detect(prices if detector_method == "rolling_vol" else log_rets)
        else:
            regime_labels = det.segment_regimes(log_rets)
    else:
        regime_labels = det.detect(prices)

    report.regime_labels = regime_labels

    # Distribution
    total = max(len(regime_labels), 1)
    report.regime_distribution = {
        r: round(float(np.sum(regime_labels == r)) / total, 4)
        for r in REGIMES
    }

    # ------------------------------------------------------------------ #
    # Step 2: Transition analysis
    # ------------------------------------------------------------------ #
    logger.info("Computing transition matrix...")
    from research.regime_lab.transition import (
        compute_transition_matrix,
        stationary_distribution,
        duration_stats_dataframe,
        plot_transition_matrix,
        plot_regime_timeline,
    )

    trans = compute_transition_matrix(regime_labels, smoothing=0.5)
    pi    = stationary_distribution(trans)
    dur_df = duration_stats_dataframe(regime_labels)

    report.transition_matrix  = trans
    report.stationary_dist    = pi
    report.duration_stats_df  = dur_df

    # Charts
    tm_path = os.path.join(output_dir, "transition_matrix.png")
    tl_path = os.path.join(output_dir, "regime_timeline.png")
    try:
        plot_transition_matrix(trans, save_path=tm_path)
        plot_regime_timeline(regime_labels, prices, save_path=tl_path)
        report.artefacts["transition_matrix_chart"] = tm_path
        report.artefacts["regime_timeline_chart"]   = tl_path
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Step 3: Per-regime trade performance
    # ------------------------------------------------------------------ #
    logger.info("Computing per-regime performance...")
    regime_perf = _compute_regime_performance(trade_list)
    report.regime_perf_df = regime_perf

    # ------------------------------------------------------------------ #
    # Step 4: Stress testing
    # ------------------------------------------------------------------ #
    if run_stress:
        logger.info("Running stress tests...")
        from research.regime_lab.stress import StressTester, stress_report, plot_stress_results
        tester = StressTester(starting_equity=starting_equity)
        results = tester.run_all_scenarios(trade_list)
        wc      = tester.worst_case_analysis(trade_list)

        report.stress_results_df = stress_report(results)
        report.worst_case_summary = {
            "worst_scenario": wc.worst_scenario,
            "worst_pnl":      wc.worst_pnl,
            "var_estimate":   wc.var_estimate,
            "cvar_estimate":  wc.cvar_estimate,
            "blowup_scenarios": wc.blowup_scenarios,
        }

        st_path = os.path.join(output_dir, "stress_results.png")
        try:
            plot_stress_results(results, save_path=st_path)
            report.artefacts["stress_chart"] = st_path
        except Exception as exc:
            logger.warning("Stress chart failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Step 5: Monte Carlo
    # ------------------------------------------------------------------ #
    if run_mc:
        logger.info("Running Monte Carlo (%d sims × %d months)...", n_mc_sims, n_mc_months)
        from research.regime_lab.simulator import RegimeMCSim, plot_regime_mc_paths
        from research.regime_lab.calibration import calibrate_regime_params, calibrate_transition_matrix

        try:
            regime_params = calibrate_regime_params(trade_list, prices,
                                                     regime_labels=regime_labels)
            sim_params = {}
            for r, p in regime_params.items():
                sim_params[r] = {
                    "mu":               p["mu"],
                    "sigma":            p["sigma"],
                    "win_rate":         p["win_rate"],
                    "trades_per_month": max(p["trades_per_month"], 2.0),
                }
        except Exception:
            sim_params = None

        mc_sim = RegimeMCSim(
            regime_params=sim_params,
            transition_matrix=trans,
            n_sims=n_mc_sims,
        )
        mc_res = mc_sim.run(starting_equity=starting_equity,
                             n_months=n_mc_months,
                             seed=mc_seed)

        report.mc_summary = {
            "blowup_rate":   mc_res.blowup_rate,
            "median_equity": mc_res.median_equity,
            "pct_5":         mc_res.pct_5,
            "pct_95":        mc_res.pct_95,
            "max_dd_p95":    mc_res.max_dd_p95,
            "sharpe":        mc_res.sharpe_ratio(),
        }
        report.regime_mc_stats = mc_res.to_summary_df()

        mc_path = os.path.join(output_dir, "mc_paths.png")
        try:
            plot_regime_mc_paths(mc_res, n_show=50, save_path=mc_path)
            report.artefacts["mc_paths_chart"] = mc_path
        except Exception as exc:
            logger.warning("MC chart failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Step 6: Calibration validation
    # ------------------------------------------------------------------ #
    if run_calibration:
        logger.info("Validating calibration...")
        try:
            from research.regime_lab.calibration import (
                validate_calibration,
                calibrate_to_history,
            )
            from research.regime_lab.generator import calibrate_to_history as gen_calibrate
            gen = gen_calibrate(prices, model="markov", regime_labels=regime_labels)
            gen_result = gen.generate(len(log_rets), seed=mc_seed)
            cal_report = validate_calibration(gen_result.returns, log_rets)
            report.calibration_report = cal_report.to_dict()
        except Exception as exc:
            logger.warning("Calibration validation failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Save CSV artefacts
    # ------------------------------------------------------------------ #
    if report.duration_stats_df is not None:
        p = os.path.join(output_dir, "duration_stats.csv")
        report.duration_stats_df.to_csv(p, index=False)
        report.artefacts["duration_stats"] = p

    if report.regime_perf_df is not None and not report.regime_perf_df.empty:
        p = os.path.join(output_dir, "regime_performance.csv")
        report.regime_perf_df.to_csv(p, index=False)
        report.artefacts["regime_performance"] = p

    if report.stress_results_df is not None and not report.stress_results_df.empty:
        p = os.path.join(output_dir, "stress_results.csv")
        report.stress_results_df.to_csv(p, index=False)
        report.artefacts["stress_results"] = p

    logger.info("Report complete. Artefacts: %s", list(report.artefacts))
    return report


# ===========================================================================
# 3. Per-regime performance helper
# ===========================================================================

def _compute_regime_performance(trade_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute per-regime win_rate, mean_pnl, total_pnl, sharpe."""
    from research.regime_lab.stress import _trade_pnl, _trade_regime

    regime_pnls: Dict[str, List[float]] = {r: [] for r in REGIMES}
    for trade in trade_list:
        r   = _trade_regime(trade)
        pnl = _trade_pnl(trade)
        regime_pnls[r].append(pnl)

    rows = []
    for r in REGIMES:
        pnls = regime_pnls[r]
        if not pnls:
            rows.append({"regime": r, "n_trades": 0, "win_rate": 0.0,
                          "mean_pnl": 0.0, "total_pnl": 0.0, "sharpe": 0.0})
            continue
        arr = np.array(pnls, dtype=float)
        s   = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sharpe = float(np.mean(arr) / s) if s > 0 else 0.0
        rows.append({
            "regime":    r,
            "n_trades":  len(arr),
            "win_rate":  round(float(np.mean(arr > 0)), 4),
            "mean_pnl":  round(float(np.mean(arr)), 2),
            "total_pnl": round(float(arr.sum()), 2),
            "sharpe":    round(sharpe, 4),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# 4. to_html
# ===========================================================================

def to_html(report: RegimeLabReport, path: str) -> None:
    """
    Render the report as a self-contained HTML file.

    Uses inline CSS and embedded base64 images.

    Parameters
    ----------
    report : RegimeLabReport
    path   : output HTML file path
    """
    import base64

    def _img_tag(img_path: str, title: str = "") -> str:
        if not os.path.exists(img_path):
            return f"<p><em>Chart not available: {os.path.basename(img_path)}</em></p>"
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (f'<figure><img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;border:1px solid #ccc;" alt="{title}">'
                f'<figcaption>{title}</figcaption></figure>')

    def _df_html(df: Optional[pd.DataFrame], title: str) -> str:
        if df is None or df.empty:
            return f"<h3>{title}</h3><p><em>No data.</em></p>"
        return (f"<h3>{title}</h3>"
                + df.to_html(index=False, border=0, classes="rtable"))

    sections: List[str] = []

    # Header
    sections.append(f"""
    <h1>Regime Lab Report</h1>
    <p><b>Generated:</b> {report.generated_at} &nbsp;|&nbsp;
       <b>Bars:</b> {report.n_bars:,} &nbsp;|&nbsp;
       <b>Trades:</b> {report.n_trades:,}</p>
    """)

    # Regime distribution
    if report.regime_distribution:
        rows = "".join(
            f"<tr><td>{r}</td><td>{v*100:.1f}%</td></tr>"
            for r, v in report.regime_distribution.items()
        )
        sections.append(f"<h2>Regime Distribution</h2>"
                         f"<table class='rtable'><tr><th>Regime</th><th>% Time</th></tr>"
                         f"{rows}</table>")

    # Charts
    for name, path_ in report.artefacts.items():
        if path_.endswith(".png"):
            sections.append(_img_tag(path_, name.replace("_", " ").title()))

    # Duration stats
    sections.append(_df_html(report.duration_stats_df, "Regime Duration Statistics"))

    # Regime performance
    sections.append(_df_html(report.regime_perf_df, "Per-Regime Trade Performance"))

    # Stress results
    sections.append(_df_html(report.stress_results_df, "Stress Test Results (worst first)"))

    # MC summary
    if report.mc_summary:
        mc = report.mc_summary
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in mc.items()
        )
        sections.append(f"<h2>Monte Carlo Summary</h2>"
                         f"<table class='rtable'><tr><th>Metric</th><th>Value</th></tr>"
                         f"{rows}</table>")

    sections.append(_df_html(report.regime_mc_stats, "Per-Regime MC Statistics"))

    # Calibration
    if report.calibration_report:
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in report.calibration_report.items()
        )
        sections.append(f"<h2>Calibration Validation</h2>"
                         f"<table class='rtable'><tr><th>Metric</th><th>Value</th></tr>"
                         f"{rows}</table>")

    css = """
    body { font-family: Arial, sans-serif; margin: 30px; background: #fafafa; color: #222; }
    h1   { color: #1a237e; }
    h2   { color: #283593; border-bottom: 2px solid #c5cae9; padding-bottom: 4px; }
    h3   { color: #3949ab; }
    .rtable { border-collapse: collapse; margin: 10px 0; font-size: 13px; }
    .rtable th, .rtable td { border: 1px solid #c5cae9; padding: 6px 12px; }
    .rtable th { background: #e8eaf6; font-weight: bold; }
    .rtable tr:hover { background: #f0f4ff; }
    figure { display: inline-block; margin: 10px; }
    figcaption { text-align: center; font-size: 11px; color: #555; }
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Regime Lab Report — {report.generated_at}</title>
<style>{css}</style>
</head>
<body>
{''.join(sections)}
</body>
</html>
"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML report written to %s", path)


# ===========================================================================
# 5. to_console
# ===========================================================================

def to_console(report: RegimeLabReport) -> None:
    """
    Print a human-readable summary of the report to stdout.

    Parameters
    ----------
    report : RegimeLabReport
    """
    print(str(report))

    if report.duration_stats_df is not None and not report.duration_stats_df.empty:
        print("\nRegime Duration Statistics:")
        print(report.duration_stats_df.to_string(index=False))

    if report.regime_perf_df is not None and not report.regime_perf_df.empty:
        print("\nPer-Regime Trade Performance:")
        print(report.regime_perf_df.to_string(index=False))

    if report.stress_results_df is not None and not report.stress_results_df.empty:
        print("\nStress Test Results (worst 5):")
        worst5 = report.stress_results_df.head(5)
        print(worst5[["scenario", "peak_drawdown_pct", "pct_equity_impact",
                       "blowup"]].to_string(index=False))

    if report.regime_mc_stats is not None and not report.regime_mc_stats.empty:
        print("\nMC Per-Regime Stats:")
        print(report.regime_mc_stats.to_string(index=False))

    if report.calibration_report:
        print(f"\nCalibration overall score: "
              f"{report.calibration_report.get('overall_score', 'N/A')}")

    if report.artefacts:
        print("\nSaved artefacts:")
        for name, path in report.artefacts.items():
            print(f"  {name:35s}  {path}")
