"""
regime_graph.py — Graphviz regime state machine visualizer for SRFM trading lab.

Usage:
    python tools/regime_graph.py [--csv results/regimes_ES.csv] [--out results/graphics/regime_states]
"""

import argparse
import os
import sys

import polars as pl


# ── Colour palette ────────────────────────────────────────────────────────────

NODE_COLORS = {
    "BULL":            "#2ecc71",
    "BEAR":            "#e74c3c",
    "SIDEWAYS":        "#f39c12",
    "HIGH_VOLATILITY": "#9b59b6",
}

NODE_LABELS = {
    "BULL":            "BULL",
    "BEAR":            "BEAR",
    "SIDEWAYS":        "SIDEWAYS",
    "HIGH_VOLATILITY": "HIGH_VOL",
}

EDGE_COLORS = {
    "BULL":            "#27ae60",
    "BEAR":            "#c0392b",
    "SIDEWAYS":        "#e67e22",
    "HIGH_VOLATILITY": "#8e44ad",
}

SELF_LOOP_COLOR = "#95a5a6"


# ── Core computation ──────────────────────────────────────────────────────────

def load_and_compute(csv_path: str):
    df = pl.read_csv(csv_path)

    # Normalise: HIGH_VOLATILITY may appear as HIGH_VOL in some CSVs
    df = df.with_columns(
        pl.col("regime")
        .str.replace("HIGH_VOL$", "HIGH_VOLATILITY")
        .alias("regime")
    )

    total_bars = len(df)

    # ── Regime time fractions ─────────────────────────────────────────────────
    regime_counts = (
        df.group_by("regime")
        .agg(pl.len().alias("bar_count"))
        .with_columns(
            (pl.col("bar_count") / total_bars * 100).round(1).alias("pct_time")
        )
    )

    # ── Transition matrix ─────────────────────────────────────────────────────
    transitions = df.with_columns(
        pl.col("regime").shift(-1).alias("next_regime")
    ).filter(pl.col("next_regime").is_not_null())

    transition_counts = (
        transitions
        .group_by(["regime", "next_regime"])
        .agg(pl.len().alias("count"))
    )

    total_per_regime = (
        transitions
        .group_by("regime")
        .agg(pl.len().alias("total"))
    )

    transition_probs = (
        transition_counts
        .join(total_per_regime, on="regime")
        .with_columns(
            (pl.col("count") / pl.col("total")).alias("prob")
        )
        .sort(["regime", "prob"], descending=[False, True])
    )

    # ── Run lengths (persistence) ─────────────────────────────────────────────
    runs = (
        df.with_columns(
            (pl.col("regime") != pl.col("regime").shift(1))
            .cast(pl.Int32)
            .cum_sum()
            .alias("run_id")
        )
        .group_by(["run_id", "regime"])
        .agg(pl.len().alias("run_length"))
    )

    avg_run = (
        runs
        .group_by("regime")
        .agg(pl.col("run_length").mean().alias("avg_run_length"))
        .with_columns(pl.col("avg_run_length").round(1))
    )

    return df, total_bars, regime_counts, transition_probs, avg_run


def best_pnl_transition(transition_probs: pl.DataFrame) -> tuple[str, str] | None:
    """
    We don't have per-transition P&L in the available data, so we use the
    highest-probability cross-regime transition TO BULL as the proxy for
    'best entry into bull regime'.  Returns (from_regime, to_regime) or None.
    """
    to_bull = (
        transition_probs
        .filter(
            (pl.col("next_regime") == "BULL") &
            (pl.col("regime") != "BULL")
        )
        .sort("prob", descending=True)
    )
    if len(to_bull) == 0:
        return None
    row = to_bull.row(0, named=True)
    return (row["regime"], row["next_regime"])


# ── DOT generation ────────────────────────────────────────────────────────────

def build_dot(
    regime_counts: pl.DataFrame,
    transition_probs: pl.DataFrame,
    total_bars: int,
    avg_run: pl.DataFrame,
    highlight_edge: tuple[str, str] | None,
) -> str:

    # Build lookup dicts
    pct_map = {r["regime"]: r["pct_time"] for r in regime_counts.iter_rows(named=True)}
    run_map = {r["regime"]: r["avg_run_length"] for r in avg_run.iter_rows(named=True)}
    all_regimes = sorted(pct_map.keys())

    lines = []
    lines.append('digraph RegimeStateMachine {')
    lines.append('    rankdir=LR;')
    lines.append('    bgcolor="#1a1a2e";')
    lines.append('    fontcolor=white;')
    lines.append('    node [style=filled, fontcolor=white, fontsize=14, fontname="Helvetica"];')
    lines.append('    edge [fontcolor=white, fontsize=11, fontname="Helvetica"];')
    lines.append(
        f'    label="SRFM Regime State Machine\\n'
        f'Measured from ES 2018-2024 (N={total_bars:,} bars)";'
    )
    lines.append('    labelloc=t;')
    lines.append('    fontsize=18;')
    lines.append('')

    # Nodes
    for regime in all_regimes:
        pct = pct_map.get(regime, 0.0)
        avg_rl = run_map.get(regime, 0.0)
        short_label = NODE_LABELS.get(regime, regime)
        color = NODE_COLORS.get(regime, "#888888")
        label = f'{short_label}\\n{pct:.1f}% of time\\navg run: {avg_rl:.1f} bars'
        lines.append(
            f'    {regime} [label="{label}", fillcolor="{color}", shape=ellipse];'
        )

    lines.append('')

    # Edges
    for row in transition_probs.iter_rows(named=True):
        src = row["regime"]
        dst = row["next_regime"]
        prob = row["prob"]
        count = row["count"]

        if prob < 0.01:
            continue

        is_self = src == dst
        penwidth = round(1 + prob * 8, 2)
        prob_label = f"{prob:.0%}"

        if is_self:
            color = SELF_LOOP_COLOR
            extra = ' constraint=false,'
        else:
            color = EDGE_COLORS.get(dst, "#888888")
            extra = ''

        # Highlight best entry to BULL
        style_attr = ''
        if highlight_edge and (src, dst) == highlight_edge and not is_self:
            color = '#FFD700'
            style_attr = ' style=bold,'

        lines.append(
            f'    {src} -> {dst} ['
            f'{extra}{style_attr}'
            f' label="{prob_label}",'
            f' penwidth={penwidth},'
            f' color="{color}",'
            f' tooltip="{count} transitions"];'
        )

    lines.append('}')
    return '\n'.join(lines)


# ── Markdown table ────────────────────────────────────────────────────────────

def build_markdown(transition_probs: pl.DataFrame, avg_run: pl.DataFrame) -> str:
    run_map = {r["regime"]: r["avg_run_length"] for r in avg_run.iter_rows(named=True)}

    rows = ["| From | → To | Prob | Count | Avg Run (bars) |",
            "|------|------|------|-------|----------------|"]

    for row in transition_probs.sort(["regime", "prob"], descending=[False, True]).iter_rows(named=True):
        src = row["regime"]
        dst = row["next_regime"]
        prob = row["prob"]
        count = row["count"]
        avg_rl = run_map.get(src, 0.0)
        rows.append(f"| {src} | {dst} | {prob:.1%} | {count:,} | {avg_rl:.1f} |")

    return "# Regime Transition Probabilities\n\n" + "\n".join(rows) + "\n"


# ── Console report ────────────────────────────────────────────────────────────

def print_report(transition_probs: pl.DataFrame, avg_run: pl.DataFrame, total_bars: int):
    run_map = {r["regime"]: r["avg_run_length"] for r in avg_run.iter_rows(named=True)}

    print("\nRegime Transition Statistics:")
    print("-" * 72)
    print(f"{'From':<16} {'-> To':<20} {'Prob':>7}  {'Count':>8}  {'Note'}")
    print("-" * 72)

    for row in transition_probs.sort(["regime", "prob"], descending=[False, True]).iter_rows(named=True):
        src = row["regime"]
        dst = row["next_regime"]
        prob = row["prob"]
        count = row["count"]
        note = "(self-loop)" if src == dst else ""
        print(f"{src:<16} -> {dst:<18} {prob:>7.1%}  {count:>8,}  {note}")

    print("-" * 72)

    # Most persistent
    self_loops = transition_probs.filter(pl.col("regime") == pl.col("next_regime"))
    if len(self_loops) > 0:
        best = self_loops.sort("prob", descending=True).row(0, named=True)
        print(f"\nMost persistent regime: {best['regime']} ({best['prob']:.1%} stay probability)")

    # Best entry to BULL
    to_bull = transition_probs.filter(
        (pl.col("next_regime") == "BULL") & (pl.col("regime") != "BULL")
    ).sort("prob", descending=True)
    if len(to_bull) > 0:
        b = to_bull.row(0, named=True)
        print(f"Best entry transition:  {b['regime']}->BULL (prob {b['prob']:.1%})")

    # Rarest non-trivial transition
    rare = (
        transition_probs
        .filter(pl.col("prob") > 0.001)
        .sort("prob")
    )
    if len(rare) > 0:
        r = rare.row(0, named=True)
        print(f"Rarest transition:      {r['regime']}->{r['next_regime']} ({r['prob']:.1%})")

    print()

    # Persistence table
    print(f"{'Regime':<20} {'Avg Run (bars)':>16}")
    print("-" * 38)
    for row in avg_run.sort("avg_run_length", descending=True).iter_rows(named=True):
        print(f"{row['regime']:<20} {row['avg_run_length']:>16.1f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SRFM Regime State Machine Visualizer")
    parser.add_argument(
        "--csv", default="results/regimes_ES.csv",
        help="Path to regimes CSV (columns: date, regime, confidence)"
    )
    parser.add_argument(
        "--out", default="results/graphics/regime_states",
        help="Output path stem (no extension)"
    )
    args = parser.parse_args()

    csv_path = args.csv
    out_stem = args.out

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)

    print(f"Loading {csv_path} ...")
    df, total_bars, regime_counts, transition_probs, avg_run = load_and_compute(csv_path)
    print(f"  {total_bars:,} bars loaded, {len(regime_counts)} regimes found.")

    highlight_edge = best_pnl_transition(transition_probs)
    if highlight_edge:
        print(f"  Highlighted best entry edge: {highlight_edge[0]}->{highlight_edge[1]} (gold)")

    # Console report
    print_report(transition_probs, avg_run, total_bars)

    # DOT file
    dot_src = build_dot(regime_counts, transition_probs, total_bars, avg_run, highlight_edge)
    dot_path = out_stem + ".dot"
    with open(dot_path, "w", encoding="utf-8") as fh:
        fh.write(dot_src)
    print(f"Written: {dot_path}")

    # Markdown table
    md_path = os.path.join(os.path.dirname(out_stem), "regime_transitions.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(build_markdown(transition_probs, avg_run))
    print(f"Written: {md_path}")

    # SVG via graphviz binary
    svg_path = out_stem + ".svg"
    rendered = False
    try:
        import graphviz
        src = graphviz.Source(dot_src, format="svg")
        # graphviz.Source.render writes <stem>.gv and <stem>.gv.svg by default;
        # use outfile kwarg if supported (graphviz >= 0.18), otherwise rename.
        try:
            src.render(outfile=svg_path, cleanup=True)
            rendered = True
        except TypeError:
            # Older graphviz package: render(filename, directory)
            import tempfile, shutil
            with tempfile.TemporaryDirectory() as tmp:
                src.render(filename="regime_states", directory=tmp, cleanup=True)
                candidate = os.path.join(tmp, "regime_states.svg")
                if not os.path.exists(candidate):
                    # Try .gv.svg naming
                    candidate = os.path.join(tmp, "regime_states.gv.svg")
                if os.path.exists(candidate):
                    shutil.copy(candidate, svg_path)
                    rendered = True
    except Exception as exc:
        print(f"  SVG render skipped ({exc})")
        print("  To render manually: dot -Tsvg regime_states.dot -o regime_states.svg")

    if rendered:
        print(f"Written: {svg_path}")
    else:
        print(f"NOTE: Graphviz binary not found. Install from https://graphviz.org/download/")
        print(f"      Then run: dot -Tsvg {dot_path} -o {svg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
