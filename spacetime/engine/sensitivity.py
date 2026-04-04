"""
sensitivity.py — Parameter sensitivity analyzer for Spacetime Arena.

Holds Geeky Orange Sheep config as baseline.
For each parameter: cf, bh_form, bh_decay, bh_collapse, ctl_req
  Perturbs by multipliers [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
  Runs full backtest at each perturbation.
  Records: CAGR, Sharpe, max drawdown, win_rate, profit_factor.

Output: robustness scores + fragility flags + edge summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bh_engine import GEEKY_DEFAULTS, INSTRUMENT_CONFIGS, run_backtest

logger = logging.getLogger(__name__)

PERTURBATION_MULTIPLIERS = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
FRAGILITY_THRESHOLD = 0.5    # if sharpe changes > 0.5 at ±10% → FRAGILE
SENSITIVITY_PARAMS  = ["cf", "bh_form", "bh_decay", "bh_collapse"]

METRICS = ["cagr", "sharpe", "max_drawdown", "win_rate", "profit_factor"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParamSensitivity:
    param:          str
    multipliers:    List[float]
    metric_values:  Dict[str, List[float]]   # {metric: [val_at_each_mult]}
    robustness:     float                    # std(sharpe) across perturbations
    fragile:        bool                     # sharpe changes >0.5 at ±10%


@dataclass
class SensitivityReport:
    sym:          str
    baseline:     Dict[str, float]           # baseline metric values
    params:       Dict[str, ParamSensitivity]
    edge_summary: str


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sensitivity(
    sym: str,
    df: pd.DataFrame,
    long_only: bool = True,
    base_params: Optional[Dict[str, Any]] = None,
) -> SensitivityReport:
    """
    Run parameter sensitivity analysis on the given OHLCV DataFrame.

    Parameters
    ----------
    sym         : instrument symbol
    df          : OHLCV DataFrame
    long_only   : backtest mode
    base_params : override baseline config (defaults to Geeky + instrument config)
    """
    # Build baseline params
    cfg = dict(INSTRUMENT_CONFIGS.get(sym.upper(), INSTRUMENT_CONFIGS["ES"]))
    cfg.update(GEEKY_DEFAULTS)
    if base_params:
        cfg.update(base_params)

    logger.info("Sensitivity analysis for %s, %d bars", sym, len(df))

    def run_one(params: Dict[str, Any]) -> Dict[str, float]:
        try:
            result = run_backtest(sym, df, long_only=long_only, params=params)
            s = result.stats
            return {
                "cagr":           s.get("cagr", 0.0),
                "sharpe":         s.get("sharpe", 0.0),
                "max_drawdown":   s.get("max_drawdown", 0.0),
                "win_rate":       s.get("win_rate", 0.0),
                "profit_factor":  min(s.get("profit_factor", 0.0), 10.0),
            }
        except Exception as e:
            logger.warning("Backtest failed with params %s: %s", params, e)
            return {m: 0.0 for m in METRICS}

    # Baseline run
    baseline_metrics = run_one(cfg)
    baseline_sharpe  = baseline_metrics["sharpe"]

    param_results: Dict[str, ParamSensitivity] = {}

    for param in SENSITIVITY_PARAMS:
        base_val = cfg.get(param, 1.0)
        mult_sharpes: List[float] = []
        metric_vals: Dict[str, List[float]] = {m: [] for m in METRICS}

        for mult in PERTURBATION_MULTIPLIERS:
            perturbed = dict(cfg)
            perturbed[param] = base_val * mult
            # Enforce physical constraints
            if param == "bh_decay":
                perturbed[param] = float(np.clip(perturbed[param], 0.5, 0.999))
            elif param in ("bh_form", "bh_collapse"):
                perturbed[param] = max(0.1, perturbed[param])
            elif param == "cf":
                perturbed[param] = max(1e-5, perturbed[param])

            metrics = run_one(perturbed)
            mult_sharpes.append(metrics["sharpe"])
            for m in METRICS:
                metric_vals[m].append(metrics[m])

        sharpe_arr = np.array(mult_sharpes)
        robustness = float(np.std(sharpe_arr))

        # Fragility: change at ±10% (multipliers 0.9 and 1.1 → indices 2, 4)
        sharpe_090 = mult_sharpes[2]
        sharpe_110 = mult_sharpes[4]
        delta_neg  = abs(baseline_sharpe - sharpe_090)
        delta_pos  = abs(baseline_sharpe - sharpe_110)
        fragile    = (delta_neg > FRAGILITY_THRESHOLD) or (delta_pos > FRAGILITY_THRESHOLD)

        param_results[param] = ParamSensitivity(
            param=param,
            multipliers=list(PERTURBATION_MULTIPLIERS),
            metric_values=metric_vals,
            robustness=robustness,
            fragile=fragile,
        )

    edge_summary = _build_edge_summary(sym, baseline_metrics, param_results)

    return SensitivityReport(
        sym=sym,
        baseline=baseline_metrics,
        params=param_results,
        edge_summary=edge_summary,
    )


def _build_edge_summary(
    sym: str,
    baseline: Dict[str, float],
    params: Dict[str, ParamSensitivity],
) -> str:
    lines = [
        f"SENSITIVITY EDGE SUMMARY — {sym}",
        "=" * 50,
        f"Baseline Sharpe:         {baseline.get('sharpe', 0):.3f}",
        f"Baseline CAGR:           {baseline.get('cagr', 0):.1%}",
        f"Baseline Max Drawdown:   {baseline.get('max_drawdown', 0):.1%}",
        f"Baseline Win Rate:       {baseline.get('win_rate', 0):.1%}",
        f"Baseline Profit Factor:  {baseline.get('profit_factor', 0):.2f}",
        "",
        "PARAMETER ROBUSTNESS:",
    ]

    robust_params = sorted(params.items(), key=lambda x: x[1].robustness)
    for param, ps in robust_params:
        flag = "FRAGILE ⚠" if ps.fragile else "ROBUST  ✓"
        lines.append(f"  {param:<15} robustness={ps.robustness:.3f}  {flag}")

    fragile_count = sum(1 for ps in params.values() if ps.fragile)
    lines.append("")
    if fragile_count == 0:
        lines.append("VERDICT: Strategy is robust — no parameters are fragile.")
    elif fragile_count <= 1:
        lines.append(f"VERDICT: Mostly robust — {fragile_count} parameter(s) show fragility at ±10%.")
    else:
        lines.append(f"VERDICT: CAUTION — {fragile_count} parameters are fragile at ±10% perturbation.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serialize for API
# ---------------------------------------------------------------------------

def sensitivity_to_dict(report: SensitivityReport) -> Dict[str, Any]:
    return {
        "sym":          report.sym,
        "baseline":     report.baseline,
        "edge_summary": report.edge_summary,
        "params": {
            param: {
                "multipliers":   ps.multipliers,
                "metric_values": ps.metric_values,
                "robustness":    ps.robustness,
                "fragile":       ps.fragile,
            }
            for param, ps in report.params.items()
        },
    }
