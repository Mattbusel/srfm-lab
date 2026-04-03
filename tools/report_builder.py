#!/usr/bin/env python
"""
report_builder.py — Generates a rich visual lab report at results/lab_report.md.

Usage:
    python tools/report_builder.py [--skip-primitives] [--out results/lab_report.md]
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo(path: str) -> str:
    return os.path.join(REPO_ROOT, path)


def fmt(val, decimals=2):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "—"
        return f"{val:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def md_table(headers, rows):
    col_widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        sr = [str(c) for c in row]
        for i, cell in enumerate(sr):
            col_widths[i] = max(col_widths[i], len(cell))
        str_rows.append(sr)

    def pad(s, w):
        return s.ljust(w)

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header = "| " + " | ".join(pad(h, col_widths[i]) for i, h in enumerate(headers)) + " |"
    lines = [header, sep]
    for row in str_rows:
        lines.append("| " + " | ".join(pad(cell, col_widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ASCII Beta Histogram
# ---------------------------------------------------------------------------

BETA_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 5.0, float("inf")]
BETA_LABELS = [
    "[0.0-0.2)",
    "[0.2-0.4)",
    "[0.4-0.6)",
    "[0.6-0.8)",
    "[0.8-1.0)",
    "[1.0-1.5)",
    "[1.5-2.0)",
    "[2.0-5.0)",
    "[5.0-∞) ",
]
BAR_CHARS = "█▓▒░"
MAX_BAR = 40


def build_beta_histogram(csv_path: str, cf: float) -> str:
    if not HAS_PANDAS or not os.path.exists(csv_path):
        return "_Beta histogram unavailable — pandas or CSV missing._\n"

    df = pd.read_csv(csv_path, parse_dates=["date"])
    delta = df["close"].diff().abs()
    beta = (delta / df["close"].shift(1)) / cf
    beta = beta.dropna()
    n_total = len(beta)

    counts = [0] * (len(BETA_BINS) - 1)
    for b in beta:
        for i in range(len(BETA_BINS) - 1):
            if BETA_BINS[i] <= b < BETA_BINS[i + 1]:
                counts[i] += 1
                break

    max_count = max(counts) if counts else 1

    lines = [
        f"```",
        f"Beta Distribution (CF={cf}, N={n_total:,} bars)",
        "━" * 55,
    ]

    for i, (label, count) in enumerate(zip(BETA_LABELS, counts)):
        pct_val = 100.0 * count / n_total if n_total else 0.0
        bar_len = int(MAX_BAR * count / max_count) if max_count else 0
        # Use filled chars for TIMELIKE, lighter for SPACELIKE
        bar_char = "█" if i < 5 else "░"
        bar = bar_char * bar_len
        btype = "TIMELIKE " if i < 5 else "SPACELIKE"

        if i == 5:
            lines.append("─" * 55 + "  ← lightcone boundary")

        threshold_note = "  ← β=1.0 threshold" if i == 4 else ""
        lines.append(f"{label}  {bar:<{MAX_BAR}}  {pct_val:5.1f}%  {btype}{threshold_note}")

    lines.append("```")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------

def build_executive_summary() -> str:
    lines = ["## Executive Summary\n"]

    # Load experiment data
    exp_path = repo("results/v2_experiments.json")
    best_exp_name = "—"
    best_exp_sharpe = 0.0
    best_exp_return = 0.0
    baseline_sharpe = 0.0
    baseline_return = 0.0

    if os.path.exists(exp_path):
        with open(exp_path) as f:
            exps = json.load(f)
        baseline = next((e for e in exps if e.get("exp") == "BASELINE"), None)
        if baseline:
            baseline_sharpe = baseline.get("arena_sharpe", 0)
            baseline_return = baseline.get("arena_return", 0)
        non_overfit = [e for e in exps if not e.get("overfit")]
        if non_overfit:
            best = max(non_overfit, key=lambda e: e.get("combined_score", 0))
            best_exp_name = best.get("exp", "?")
            best_exp_sharpe = best.get("arena_sharpe", 0)
            best_exp_return = best.get("arena_return", 0)

    # Load trade analysis
    analysis_path = repo("research/trade_analysis_data.json")
    qc_sharpe = "—"
    total_pnl = "—"
    n_trades = "—"

    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            data = json.load(f)
        s = data.get("summary", {})
        n_trades = str(s.get("n_trades", "—"))
        total_pnl = f"${s.get('total_pnl', 0):,.0f}"
        # approximate QC sharpe from equity curve if available
        equity_curve = data.get("equity_curve", [])
        if len(equity_curve) > 2:
            returns = []
            for i in range(1, len(equity_curve)):
                prev = equity_curve[i - 1][1]
                curr = equity_curve[i][1]
                if prev:
                    returns.append((curr - prev) / prev)
            if returns:
                import statistics
                avg_r = statistics.mean(returns)
                std_r = statistics.stdev(returns) if len(returns) > 1 else 1e-9
                qc_sharpe = f"{avg_r / std_r * (252 ** 0.5):.2f}" if std_r else "—"

    key_metrics = [
        ["Baseline Arena Sharpe", fmt(baseline_sharpe, 3), "arena backtester"],
        ["Baseline Arena Return%", fmt(baseline_return, 2) + "%", "arena backtester"],
        ["Best Experiment", best_exp_name, f"Sharpe {fmt(best_exp_sharpe, 3)}, Return {fmt(best_exp_return, 2)}%"],
        ["Total PnL (trade analysis)", total_pnl, f"{n_trades} trades"],
        ["QC Approx Sharpe", qc_sharpe, "from equity curve, annualized"],
    ]

    lines.append(md_table(["Metric", "Value", "Notes"], key_metrics))
    lines.append("")

    # Hypothesis bullets
    lines.append("### Hypothesis Status\n")
    lines.append("- **Convergence hypothesis:** Multi-instrument well clustering produces superior PnL density — "
                 "supported by convergence matrix (2+ instrument wells show elevated win rates).")
    lines.append("- **BEAR_FAST regime rejection:** High-volatility short-side entries in fast bear regimes "
                 "exhibit SPACELIKE beta inflation; these bars should be filtered or sized down.")
    lines.append("- **Arena vs QC gap:** Arena Sharpe (~0.4) vs QC Sharpe (~4) gap indicates CF miscalibration "
                 "and/or fill-assumption differences — not alpha decay. Recommend CF sweep.\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spacetime State Machine Diagram
# ---------------------------------------------------------------------------

SPACETIME_MERMAID = """\
## Spacetime Analysis

```mermaid
stateDiagram-v2
    [*] --> SIDEWAYS
    SIDEWAYS --> BULL : EMA12>EMA26>EMA50>EMA200
    SIDEWAYS --> BEAR : EMA200>EMA50>EMA26>EMA12
    BULL --> BH_FORMING : ctl>=3 AND mass>1.0
    BH_FORMING --> BH_ACTIVE : mass>1.5 AND ctl>=5
    BH_ACTIVE --> CONVERGENCE : bh_count>=2
    CONVERGENCE --> LIQUIDATE : geo_raw>2.0 OR weak_bars>=3
    BH_ACTIVE --> SIDEWAYS : SPACELIKE AND mass<bh_collapse
```

The SRFM state machine models market regimes as spacetime geometries. Bars with
beta < 1.0 are **TIMELIKE** (normal drift), while bars with beta >= 1.0 are
**SPACELIKE** (explosive move — the market has crossed a lightcone boundary).
Black Hole (BH) wells form when enough TIMELIKE CTL (causal time-like) bars
accumulate with sufficient mass. Convergence occurs when multiple instruments
simultaneously develop active wells.
"""


# ---------------------------------------------------------------------------
# Well Analysis
# ---------------------------------------------------------------------------

def build_well_analysis() -> str:
    lines = ["## Well Analysis\n"]
    wells_path = repo("results/wells_ES.csv")

    if not HAS_PANDAS or not os.path.exists(wells_path):
        lines.append("_Well data unavailable._\n")
        return "\n".join(lines)

    df = pd.read_csv(wells_path, parse_dates=["formed_at", "collapsed_at"])
    df = df.sort_values("mass_peak", ascending=False).head(10)

    # Mermaid timeline of top 10 wells
    lines.append("### Top 10 Wells by Mass (Mermaid Timeline)\n")
    lines.append("```mermaid")
    lines.append("gantt")
    lines.append("    title Top 10 BH Wells (ES)")
    lines.append("    dateFormat YYYY-MM-DD")
    lines.append("    axisFormat %Y-%m")
    for _, row in df.iterrows():
        formed = str(row["formed_at"])[:10]
        collapsed = str(row["collapsed_at"])[:10]
        if formed == collapsed:
            collapsed = formed  # gantt needs end >= start; same-day wells
        label = f"{row['direction']} m={row['mass_peak']:.2f}"
        lines.append(f"    {label} : {formed}, {collapsed}")
    lines.append("```\n")

    # ASCII well calendar — year x week grid
    lines.append("### Well Activity Calendar (year × week, █ = active week)\n")
    all_wells = pd.read_csv(repo("results/wells_ES.csv"), parse_dates=["formed_at"])
    all_wells["year"] = all_wells["formed_at"].dt.year
    all_wells["week"] = all_wells["formed_at"].dt.isocalendar().week

    years = sorted(all_wells["year"].unique())
    active = set(zip(all_wells["year"], all_wells["week"]))

    lines.append("```")
    lines.append("Year  " + "".join(str(w % 10) for w in range(1, 53)))
    lines.append("      " + "".join([str(w // 10) if w % 10 == 0 else " " for w in range(1, 53)]))
    for yr in years:
        row_str = ""
        for wk in range(1, 53):
            row_str += "█" if (yr, wk) in active else "░"
        lines.append(f"{yr}  {row_str}")
    lines.append("```\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment Results
# ---------------------------------------------------------------------------

def build_experiment_results() -> str:
    lines = ["## Experiment Results\n"]
    exp_path = repo("results/v2_experiments.json")

    if not os.path.exists(exp_path):
        lines.append("_Experiment data unavailable._\n")
        return "\n".join(lines)

    with open(exp_path) as f:
        exps = json.load(f)

    baseline = next((e for e in exps if e.get("exp") == "BASELINE"), None)
    base_sharpe = baseline["arena_sharpe"] if baseline else 0.0
    ranked = sorted(exps, key=lambda e: (e.get("combined_score", 0), e.get("arena_sharpe", 0)), reverse=True)

    rows = []
    for i, e in enumerate(ranked, 1):
        delta_s = e.get("arena_sharpe", 0) - base_sharpe
        marker = " ★" if i == 1 else ("  OVERFIT" if e.get("overfit") else "")
        rows.append([
            str(i),
            e.get("exp", "?") + marker,
            fmt(e.get("combined_score", 0), 3),
            fmt(e.get("arena_sharpe", 0), 3),
            f"{delta_s:+.3f}",
            fmt(e.get("arena_return", 0), 2) + "%",
            fmt(e.get("arena_dd", 0), 2) + "%",
        ])

    lines.append(md_table(
        ["#", "Experiment", "Score", "Sharpe", "ΔSharpe", "Return", "MaxDD"],
        rows,
    ))
    lines.append("")

    # ASCII bar chart of ΔSharpe per experiment
    lines.append("### ΔSharpe vs Baseline (ASCII)\n")
    lines.append("```")
    max_delta = max(abs(e.get("arena_sharpe", 0) - base_sharpe) for e in exps) or 1.0
    for e in ranked:
        delta_s = e.get("arena_sharpe", 0) - base_sharpe
        bar_len = int(abs(delta_s) / max_delta * 30)
        if delta_s >= 0:
            bar = "█" * bar_len
            line = f"{e['exp']:<22} +{delta_s:.3f} |{bar}"
        else:
            bar = "▒" * bar_len
            line = f"{e['exp']:<22} {delta_s:.3f} |{bar}"
        lines.append(line)
    lines.append("```\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Arena vs QC Gap Analysis
# ---------------------------------------------------------------------------

def build_gap_analysis() -> str:
    lines = ["## Arena vs QC Gap Analysis\n"]

    lines.append(
        "The arena backtester produces Sharpe ~0.4 while the QC equity curve implies "
        "a much higher annualized Sharpe (~4+). This gap arises from several sources:\n"
    )

    gap_rows = [
        ["Fill assumptions", "Arena uses mid-price fills; QC may use limit fills", "High"],
        ["CF calibration", "CF=0.005 may be too small, inflating beta and blocking trades", "High"],
        ["Slippage model", "Arena applies fixed slippage; QC uses tick-based", "Medium"],
        ["Bar aggregation", "Hourly bars may miss intrabar entries", "Medium"],
        ["Universe", "Arena runs on synthetic data; QC on real ES ticks", "Medium"],
        ["Risk sizing", "Arena fixed-size; QC may scale by well mass", "Low"],
    ]

    lines.append(md_table(["Factor", "Description", "Impact"], gap_rows))
    lines.append("")

    # CF calibration table
    lines.append("### CF Calibration Table\n")
    lines.append("Effect of CF on TIMELIKE fraction and expected trade count:\n")

    cf_rows = []
    for cf_test in [0.001, 0.002, 0.005, 0.010, 0.020, 0.050]:
        # Rough approximation: for NDX hourly, typical |Δclose/close| ~0.002
        # beta = typical_move / cf_test; TIMELIKE when beta < 1 → typical_move < cf_test
        typical_move = 0.0020
        if cf_test > typical_move:
            est_timelike = min(99.9, max(0.1, 100.0 * (1.0 - typical_move / cf_test)))
        else:
            est_timelike = 100.0 * cf_test / typical_move / 2
        cf_rows.append([f"{cf_test:.3f}", f"~{est_timelike:.0f}%", "Higher CF → more TIMELIKE bars → more entries"])

    lines.append(md_table(["CF Value", "Est. TIMELIKE%", "Effect"], cf_rows))
    lines.append("")

    lines.append("### Recommendations\n")
    lines.append("1. Run `primitive_builder.py --cf 0.010` and compare TIMELIKE fraction.")
    lines.append("2. Align arena slippage model with QC tick assumptions.")
    lines.append("3. Add fill-price logging to arena to validate vs QC fills.")
    lines.append("4. Test well mass threshold sensitivity (current: 1.0 form, 1.5 active).\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Next Experiments
# ---------------------------------------------------------------------------

NEXT_EXPERIMENTS = """\
## Next Experiments

1. **CF_SWEEP_0010** — Run primitive_builder with CF=0.010; re-run arena. Hypothesis: more TIMELIKE bars \
increase trade count and reduce false SPACELIKE filtering, raising Sharpe above 0.6.

2. **MASS_THRESHOLD_SENSITIVITY** — Sweep mass thresholds (form: 0.8–1.2, active: 1.2–2.0). \
Hypothesis: current threshold of 1.5 is too conservative for liquid instruments.

3. **CONVERGENCE_ONLY** — Trade only wells with 2+ instruments in convergence. \
Hypothesis: convergence matrix shows these wells have higher win rate; isolating them improves risk-adjusted returns.

4. **BEAR_REGIME_FILTER** — Skip new BH entries when regime == BEAR AND hawking_temp > 0.03. \
Hypothesis: BEAR_FAST bars are consistently SPACELIKE and generate losing trades.

5. **HAWKING_TEMP_SIZING** — Scale position size inversely with hawking_temp. \
Hypothesis: cooler wells (lower Hawking temperature) indicate stronger gravitational wells with less evaporation risk.

6. **ANNUAL_WALK_FORWARD** — Re-run walk-forward with annual refit windows instead of quarterly. \
Hypothesis: annual windows reduce overfitting while preserving enough signal for the optimizer.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report(skip_primitives: bool, out_path: str):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Optionally regenerate primitives
    primitives_path = repo("results/primitives.md")
    if not skip_primitives:
        primitive_script = repo("tools/primitive_builder.py")
        print("[report_builder] Running primitive_builder.py...")
        result = subprocess.run(
            [sys.executable, primitive_script],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  Warning: primitive_builder exited {result.returncode}")
            print(result.stderr[:500])
        else:
            print(result.stdout.strip())

    # Read primitives
    primitives_content = ""
    if os.path.exists(primitives_path):
        with open(primitives_path, encoding="utf-8") as f:
            primitives_content = f.read()
    else:
        primitives_content = "_Primitives file not found — run primitive_builder.py first._\n"

    # CSV path for beta histogram
    csv_path = repo("data/NDX_hourly_poly.csv")

    sections = []

    sections.append(f"# SRFM Lab Report — {today}\n")
    sections.append(f"> Generated: {now}\n")
    sections.append("---\n")

    sections.append(build_executive_summary())
    sections.append("---\n")

    sections.append("## SRFM Physics Primitives\n")
    sections.append(primitives_content)
    sections.append("")
    sections.append("### Beta Distribution Histogram\n")
    sections.append(build_beta_histogram(csv_path, cf=0.005))
    sections.append("---\n")

    sections.append(SPACETIME_MERMAID)
    sections.append("---\n")

    sections.append(build_well_analysis())
    sections.append("---\n")

    sections.append(build_experiment_results())
    sections.append("---\n")

    sections.append(build_gap_analysis())
    sections.append("---\n")

    sections.append(NEXT_EXPERIMENTS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    print(f"[report_builder] Written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRFM Report Builder")
    parser.add_argument("--skip-primitives", action="store_true",
                        help="Skip regenerating primitives.md")
    parser.add_argument("--out", default=repo("results/lab_report.md"),
                        help="Output markdown path")
    args = parser.parse_args()

    build_report(args.skip_primitives, args.out)
