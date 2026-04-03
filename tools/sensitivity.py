"""
sensitivity.py — Multi-dimensional parameter sensitivity analysis.

For each parameter, sweeps 5 values around the current setting, runs
lean backtest, plots sensitivity surface, and flags overfitting risk.

Overfitting flag: if return drops by > 30% from optimal when parameter
changes by ±1 step, the strategy is too sensitive to that parameter.

Usage:
    python tools/sensitivity.py strategies/larsa-v1 --params CF,BH_FORM,BH_DECAY --resolution 5
    python tools/sensitivity.py strategies/larsa-v1 --params BH_FORM --resolution 7 --metric SharpeRatio

Output:
    results/larsa-v1/sensitivity/sensitivity_report.md
    results/larsa-v1/sensitivity/<param>_sensitivity.png
    results/larsa-v1/sensitivity/sensitivity_summary.csv
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --- Parameter extraction -----------------------------------------------------

def read_param(main_py: str, param: str) -> Optional[float]:
    """Read current value of a constant from main.py."""
    with open(main_py) as f:
        code = f.read()
    m = re.search(rf'^{re.escape(param)}\s*=\s*([0-9.e+-]+)', code, re.MULTILINE)
    if m:
        return float(m.group(1))
    return None


def patch_param(src: str, param: str, value: float, dst: str):
    with open(src) as f:
        code = f.read()
    new_code, n = re.subn(
        rf'^({re.escape(param)}\s*=\s*)[^\n#]+',
        rf'\g<1>{value!r}',
        code, flags=re.MULTILINE
    )
    if n == 0:
        print(f"  [WARN] '{param}' not found in {src}", file=sys.stderr)
    with open(dst, 'w') as f:
        f.write(new_code)


# --- Backtest runner ----------------------------------------------------------

def run_backtest(strategy_dir: str, output_dir: str) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        ['lean', 'backtest', strategy_dir, '--output', output_dir],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        return None
    for name in ['result.json', 'backtest-results.json']:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return p
    jsons = list(Path(output_dir).glob('*.json'))
    return str(jsons[0]) if jsons else None


def extract_metric(result_json: str, metric: str) -> float:
    try:
        with open(result_json) as f:
            data = json.load(f)
        stats = data.get('TotalPerformance', {}).get('PortfolioStatistics', {})
        v = stats.get(metric)
        if v is None:
            v = data.get('TotalPerformance', {}).get('TradeStatistics', {}).get(metric)
        if isinstance(v, str):
            v = v.replace('%', '').strip()
            v = float(v)
        return float(v) if v is not None else float('nan')
    except Exception:
        return float('nan')


# --- Sweep logic -------------------------------------------------------------

def build_values(center: float, resolution: int) -> List[float]:
    """
    Generate `resolution` values centred on `center`.
    Spacing = max(center * 0.15, 0.0001) per step.
    """
    n     = resolution // 2
    step  = max(abs(center) * 0.15, 0.0001)
    vals  = [center + (i - n) * step for i in range(resolution)]
    # Keep values positive for most parameters
    return [max(1e-6, v) for v in vals]


def run_sweep(
    strategy_dir: str,
    param:        str,
    values:       List[float],
    metric:       str,
    out_base:     str,
) -> List[Tuple[float, float]]:
    results = []
    src_main = os.path.join(strategy_dir, 'main.py')
    name     = Path(strategy_dir).name

    for val in values:
        label = f"{param}_{val:.6g}".replace('.', 'p')
        out   = os.path.join(out_base, label)
        print(f"  {param}={val:.6g} ...", end=' ', flush=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_s = os.path.join(tmp, 'strategy')
            shutil.copytree(strategy_dir, tmp_s)
            patch_param(src_main, param, val, os.path.join(tmp_s, 'main.py'))
            rp = run_backtest(tmp_s, out)
            score = extract_metric(rp, metric) if rp else float('nan')

        print(f"{metric}={score:.4f}" if not math.isnan(score) else "FAILED")
        results.append((val, score))

    return results


# --- Overfitting analysis ----------------------------------------------------

def overfitting_risk(values: List[float], scores: List[float], threshold: float = 0.30) -> dict:
    """
    Compute how fast the score drops off from the optimum.
    High sensitivity = overfit risk.
    """
    valid  = [(v, s) for v, s in zip(values, scores) if not math.isnan(s)]
    if len(valid) < 3:
        return {'risk': 'UNKNOWN', 'sensitivity': None}

    xs, ys     = zip(*valid)
    best_score = max(ys)
    if best_score <= 0:
        return {'risk': 'UNKNOWN', 'sensitivity': None}

    # Adjacent drop: max fractional drop between consecutive steps
    drops = [abs(ys[i] - ys[i-1]) / (abs(best_score) + 1e-9) for i in range(1, len(ys))]
    max_drop = max(drops)

    if max_drop > threshold:
        risk = 'HIGH — parameter cliff detected'
    elif max_drop > threshold * 0.5:
        risk = 'MEDIUM — moderate sensitivity'
    else:
        risk = 'LOW — robust to this parameter'

    return {'risk': risk, 'sensitivity': max_drop, 'best_score': best_score}


# --- Plot ---------------------------------------------------------------------

def plot_sensitivity(param: str, values: List[float], scores: List[float],
                     metric: str, name: str, path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    valid = [(v, s) for v, s in zip(values, scores) if not math.isnan(s)]
    if not valid:
        return
    xs, ys = zip(*valid)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, ys, 'o-', linewidth=2, markersize=7, color='steelblue')
    best_x = xs[ys.index(max(ys))]
    ax.axvline(best_x, color='green', linestyle='--', alpha=0.6, label=f'Best: {param}={best_x:.4g}')
    ax.set_title(f'Sensitivity: {name} / {param} -> {metric}', fontweight='bold')
    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"  PNG -> {path}")


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SRFM parameter sensitivity')
    parser.add_argument('strategy',        help='Strategy directory')
    parser.add_argument('--params',        default='BH_FORM', help='Comma-separated param names')
    parser.add_argument('--resolution',    type=int,   default=5,          help='Values per param')
    parser.add_argument('--metric',        default='SharpeRatio')
    parser.add_argument('--no-plot',       action='store_true')
    args   = parser.parse_args()

    params   = [p.strip() for p in args.params.split(',')]
    name     = Path(args.strategy).name
    src_main = os.path.join(args.strategy, 'main.py')
    out_base = os.path.join('results', name, 'sensitivity')
    os.makedirs(out_base, exist_ok=True)

    all_results: Dict[str, List[Tuple[float, float]]] = {}
    report_rows = []

    for param in params:
        print(f"\n[Param: {param}]")
        center = read_param(src_main, param)
        if center is None:
            print(f"  [SKIP] '{param}' not found in {src_main}")
            continue
        print(f"  Current value: {center}")
        values = build_values(center, args.resolution)
        print(f"  Testing: {[f'{v:.4g}' for v in values]}")

        results = run_sweep(args.strategy, param, values, args.metric, out_base)
        all_results[param] = results

        vals   = [r[0] for r in results]
        scores = [r[1] for r in results]
        risk   = overfitting_risk(vals, scores)

        print(f"  Overfitting risk: {risk['risk']}")

        for v, s in results:
            report_rows.append({
                'param': param, 'value': v, 'score': s,
                'metric': args.metric, 'risk': risk['risk'],
            })

        if not args.no_plot:
            plot_sensitivity(
                param, vals, scores, args.metric, name,
                os.path.join(out_base, f'{param}_sensitivity.png'),
            )

    # CSV
    csv_path = os.path.join(out_base, 'sensitivity_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        if report_rows:
            w = csv.DictWriter(f, fieldnames=report_rows[0].keys())
            w.writeheader(); w.writerows(report_rows)
    print(f"\nCSV -> {csv_path}")

    # Markdown report
    md_path = os.path.join(out_base, 'sensitivity_report.md')
    with open(md_path, 'w') as f:
        f.write(f"# Parameter Sensitivity Report — {name}\n\n")
        f.write(f"Metric: **{args.metric}**\n\n")
        for param, results in all_results.items():
            vals   = [r[0] for r in results]
            scores = [r[1] for r in results]
            risk   = overfitting_risk(vals, scores)
            f.write(f"## {param}\n\n")
            f.write(f"- Overfitting risk: **{risk['risk']}**\n")
            if risk.get('sensitivity') is not None:
                f.write(f"- Max adjacent drop: {risk['sensitivity']:.1%}\n")
            f.write(f"- Best {args.metric}: {risk.get('best_score', 'N/A'):.4f}\n\n")
            f.write(f"| Value | {args.metric} |\n|-------|--------|\n")
            for v, s in results:
                f.write(f"| {v:.4g} | {s:.4f if not math.isnan(s) else 'FAILED'} |\n")
            f.write("\n")
    print(f"MD  -> {md_path}")


if __name__ == '__main__':
    main()
