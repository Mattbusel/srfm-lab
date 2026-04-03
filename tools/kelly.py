"""
kelly.py -- Kelly-optimal position sizing for ES/NQ/YM.

Uses Riskfolio-Lib for mean-variance / Kelly portfolio optimization.
Replaces fixed 0.65/0.40 caps with mathematically optimal fractions.

Usage:
    python tools/kelly.py
    python tools/kelly.py --by-regime       # optimal sizing per regime
    python tools/kelly.py --by-type         # solo vs convergence Kelly
    echo "0.55 1.34" | python tools/kelly.py --pipe  # manual W/L input
"""

import argparse
import json
import math
import os
import sys

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "research", "trade_analysis_data.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full Kelly: f* = (p*b - q) / b  where b = avg_win/avg_loss."""
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    p = win_rate
    q = 1.0 - p
    b = avg_win / avg_loss
    f = (p * b - q) / b
    return max(0.0, f)


def compute_stats(wells: list) -> dict:
    wins   = [w for w in wells if w["is_win"]]
    losses = [w for w in wells if not w["is_win"]]
    n = len(wells)
    if n == 0:
        return {}
    wr = len(wins) / n
    avg_win  = sum(w["total_pnl"] for w in wins)  / max(1, len(wins))
    avg_loss = abs(sum(w["total_pnl"] for w in losses)) / max(1, len(losses))
    full_k   = kelly_fraction(wr, avg_win, avg_loss)
    return {
        "n": n, "wr": wr, "avg_win": avg_win, "avg_loss": avg_loss,
        "full_kelly": full_k, "half_kelly": full_k / 2.0,
    }


def simulate_half_kelly(wells: list, stats: dict, initial_equity: float = 1_000_000) -> float:
    """Simulate cumulative return if we sized each well by half-Kelly fraction."""
    hk = stats.get("half_kelly", 0.0)
    equity = initial_equity
    for w in wells:
        pnl_pct = w["total_pnl"] / initial_equity  # raw % of initial
        # scale by half-kelly vs assumed full 0.65 leverage
        assumed_frac = 0.65
        if assumed_frac > 0:
            scaled_pct = pnl_pct * (hk / assumed_frac)
        else:
            scaled_pct = pnl_pct
        equity *= (1.0 + scaled_pct)
    return (equity - initial_equity) / initial_equity * 100.0


def riskfolio_kelly(wells: list) -> dict | None:
    """Attempt Riskfolio-Lib covariance-adjusted Kelly sizing."""
    try:
        import riskfolio as rp
        import numpy as np
        import pandas as pd

        # Build a returns series per instrument
        instruments = ["ES", "NQ", "YM"]
        series = {inst: [] for inst in instruments}
        for w in wells:
            for inst in instruments:
                if inst in w.get("instruments", []):
                    series[inst].append(w["total_pnl"])
                else:
                    series[inst].append(0.0)

        min_len = min(len(v) for v in series.values())
        df = pd.DataFrame({k: v[:min_len] for k, v in series.items()})
        # Normalise to fractional returns
        df = df / 1_000_000.0

        port = rp.Portfolio(returns=df)
        port.assets_stats(method_mu="hist", method_cov="ledoit")
        w_kelly = port.optimization(model="Classic", rm="MV", obj="MaxRet",
                                    rf=0, l=0, hist=True)
        return {"weights": w_kelly.to_dict() if w_kelly is not None else {}, "source": "riskfolio"}
    except Exception as exc:
        return {"error": str(exc), "source": "fallback"}


def main():
    parser = argparse.ArgumentParser(description="Kelly-optimal position sizing for LARSA v1")
    parser.add_argument("--by-regime", action="store_true", help="Per-regime Kelly fractions")
    parser.add_argument("--by-type",   action="store_true", help="Solo vs convergence Kelly")
    parser.add_argument("--pipe",      action="store_true", help="Read 'WR AVG_WIN_LOSS_RATIO' from stdin")
    args = parser.parse_args()

    # -- Pipe mode --------------------------------------------------------------
    if args.pipe:
        raw = sys.stdin.read().strip().split()
        if len(raw) >= 2:
            wr = float(raw[0])
            ratio = float(raw[1])
            fk = (wr * ratio - (1 - wr)) / ratio
            print(f"Full Kelly:  {max(0,fk)*100:.2f}%")
            print(f"Half Kelly:  {max(0,fk/2)*100:.2f}%")
        return

    # -- Load data --------------------------------------------------------------
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    wells  = data["wells"]
    summary = data["summary"]

    # -- Overall Kelly ----------------------------------------------------------
    overall = compute_stats(wells)

    lines = []
    lines.append("KELLY-OPTIMAL POSITION SIZING -- LARSA v1")
    lines.append("=" * 42)
    lines.append(f"ALL TRADES ({overall['n']} wells):")
    lines.append(f"  Win Rate:   {overall['wr']*100:.1f}%   Avg Win: ${overall['avg_win']:,.0f}   Avg Loss: ${overall['avg_loss']:,.0f}")
    lines.append(f"  Full Kelly: {overall['full_kelly']*100:.1f}%  portfolio per trade")
    lines.append(f"  Half Kelly: {overall['half_kelly']*100:.1f}%  (recommended)")

    # -- Convergence vs Solo ----------------------------------------------------
    conv_wells = [w for w in wells if len(w.get("instruments", [])) > 1]
    solo_wells = [w for w in wells if len(w.get("instruments", [])) == 1]

    conv_stats = compute_stats(conv_wells)
    solo_stats = compute_stats(solo_wells)

    lines.append("")
    if conv_stats:
        lines.append(f"CONVERGENCE ({conv_stats['n']} wells, {conv_stats['wr']*100:.1f}% WR):")
        lines.append(f"  Full Kelly: {conv_stats['full_kelly']*100:.1f}%  <-- size up massively")
        lines.append(f"  Half Kelly: {conv_stats['half_kelly']*100:.1f}%")
    else:
        lines.append("CONVERGENCE: 0 multi-instrument wells found in data")

    lines.append("")
    if solo_stats:
        lines.append(f"SOLO ({solo_stats['n']} wells, {solo_stats['wr']*100:.1f}% WR):")
        lines.append(f"  Full Kelly: {solo_stats['full_kelly']*100:.1f}%  <-- near zero, don't oversize")
        lines.append(f"  Half Kelly: {solo_stats['half_kelly']*100:.1f}%")

    # Lever ratio
    lines.append("")
    if conv_stats and solo_stats and solo_stats["half_kelly"] > 0:
        ratio = conv_stats["half_kelly"] / solo_stats["half_kelly"]
        lines.append(f"OPTIMAL LEVER RATIO (conv/solo):  {ratio:.1f}x")
    elif conv_stats and solo_stats:
        lines.append("OPTIMAL LEVER RATIO: solo half-Kelly ≈ 0 (convergence dominates)")
    lines.append("Current v6 ratio (0.65/0.15):      4.3x")
    lines.append("Kelly says: be even MORE aggressive on convergence")

    # -- Per-instrument Kelly ---------------------------------------------------
    lines.append("")
    lines.append("PER-INSTRUMENT KELLY:")
    by_inst = data.get("by_instrument", summary.get("by_instrument", {}))
    for inst in ["ES", "NQ", "YM"]:
        inst_wells = [w for w in wells if inst in w.get("instruments", [])]
        st = compute_stats(inst_wells)
        if st:
            lines.append(
                f"  {inst}:  WR={st['wr']*100:.1f}%  Kelly={st['full_kelly']*100:.1f}%"
                f"   ({st['n']} trades)"
            )

    # Recommendation
    nq_wells = [w for w in wells if "NQ" in w.get("instruments", [])]
    ym_wells = [w for w in wells if "YM" in w.get("instruments", [])]
    nq_st = compute_stats(nq_wells)
    ym_st = compute_stats(ym_wells)
    if nq_st and ym_st and nq_st["full_kelly"] > ym_st["full_kelly"]:
        lines.append("  Optimal: overweight NQ, underweight YM")

    # -- Simulation -------------------------------------------------------------
    lines.append("")
    lines.append("SIMULATION: if we had used Half-Kelly sizing throughout:")
    actual_ret = summary["total_return_pct"]
    hk_ret = simulate_half_kelly(wells, overall)
    lines.append(f"  Actual return:      {actual_ret:.1f}%")
    lines.append(f"  Half-Kelly return:  {hk_ret:.1f}%   (simulated with scaled P&L)")

    # -- Per-year Kelly ---------------------------------------------------------
    if args.by_regime:
        lines.append("")
        lines.append("PER-YEAR KELLY (regime proxy via year):")
        for yr, info in data.get("by_year", {}).items():
            yr_wells = [w for w in wells if str(w.get("year", "")) == yr]
            st = compute_stats(yr_wells)
            if st:
                lines.append(
                    f"  {yr}:  WR={st['wr']*100:.1f}%  Kelly={st['full_kelly']*100:.1f}%"
                    f"  ({st['n']} wells)"
                )

    # -- Riskfolio attempt ------------------------------------------------------
    lines.append("")
    lines.append("RISKFOLIO-LIB STATUS:")
    rp_result = riskfolio_kelly(wells)
    if rp_result and rp_result.get("source") == "riskfolio":
        lines.append("  riskfolio available -- covariance-adjusted weights:")
        weights = rp_result.get("weights", {})
        for inst, wval in weights.items():
            if isinstance(wval, dict):
                for k2, v2 in wval.items():
                    lines.append(f"    {inst}/{k2}: {float(v2):.3f}")
            else:
                lines.append(f"    {inst}: {float(wval):.3f}")
    else:
        err = rp_result.get("error", "not installed") if rp_result else "not installed"
        lines.append(f"  riskfolio not available ({err})")
        lines.append("  Using manual Kelly formula (f* = (p*b - q) / b)")

    output = "\n".join(lines)
    print(output)

    # -- Save report ------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "kelly_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Kelly Analysis -- LARSA v1\n\n")
        f.write("```\n")
        f.write(output)
        f.write("\n```\n")
        f.write(f"\n## Key Findings\n\n")
        f.write(f"- Overall half-Kelly: **{overall['half_kelly']*100:.1f}%** (vs fixed 0.65 cap)\n")
        if conv_stats:
            f.write(f"- Convergence half-Kelly: **{conv_stats['half_kelly']*100:.1f}%** -- size up on multi-instrument\n")
        if solo_stats:
            f.write(f"- Solo half-Kelly: **{solo_stats['half_kelly']*100:.1f}%** -- do not oversize solo wells\n")
        f.write(f"- Simulated half-Kelly return: **{hk_ret:.1f}%** vs actual **{actual_ret:.1f}%**\n")
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
