"""
regime_network.py — Interactive regime transition network.

Builds a directed graph of regime transitions with:
- Node size = time spent in regime
- Edge thickness = transition frequency
- Edge color = transition quality (profitable vs not)
- Exports interactive HTML via PyVis

Usage:
    python tools/regime_network.py --csv data/NDX_hourly_poly.csv
    python tools/regime_network.py  # uses trade data for regime sequence
"""

import argparse
import json
import os
import sys

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "research", "trade_analysis_data.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

REGIMES = ["BULL", "SIDEWAYS", "BEAR", "HIGH_VOL"]
REGIME_COLORS = {
    "BULL":     "#27ae60",
    "SIDEWAYS": "#f39c12",
    "BEAR":     "#e74c3c",
    "HIGH_VOL": "#8e44ad",
}


def infer_regime_sequence(wells: list) -> list:
    """
    Infer a rough regime sequence from trade data.
    Uses year and win-rate patterns as a proxy.
    """
    # Known rough regimes by year
    YEAR_REGIME = {
        2018: "SIDEWAYS",
        2019: "BULL",
        2020: "HIGH_VOL",
        2021: "BULL",
        2022: "BEAR",
        2023: "BULL",
        2024: "BULL",
    }
    seq = []
    for w in sorted(wells, key=lambda x: x.get("start", "")):
        yr = w.get("year", 2021)
        seq.append(YEAR_REGIME.get(yr, "SIDEWAYS"))
    return seq


def build_transition_matrix(seq: list) -> dict:
    """Returns transition_matrix[from_regime][to_regime] = count."""
    tm = {r: {r2: 0 for r2 in REGIMES} for r in REGIMES}
    for i in range(1, len(seq)):
        frm = seq[i - 1]
        to  = seq[i]
        if frm in tm and to in tm[frm]:
            tm[frm][to] += 1
    return tm


def transition_success_rate(wells: list, seq: list) -> dict:
    """
    Success rate of trades that start right after each transition type.
    Returns success_rates[from][to] = float 0-1.
    """
    sr = {r: {r2: [] for r2 in REGIMES} for r in REGIMES}
    for i in range(1, min(len(seq), len(wells))):
        frm = seq[i - 1]
        to  = seq[i]
        if frm in sr and to in sr[frm]:
            sr[frm][to].append(1 if wells[i].get("is_win") else 0)
    rates = {}
    for frm in REGIMES:
        rates[frm] = {}
        for to in REGIMES:
            wins_list = sr[frm][to]
            rates[frm][to] = sum(wins_list) / len(wins_list) if wins_list else 0.5
    return rates


def row_normalize(tm: dict) -> dict:
    """Normalize transition matrix rows to percentages."""
    norm = {}
    for r in REGIMES:
        total = sum(tm[r].values())
        if total > 0:
            norm[r] = {r2: tm[r][r2] / total * 100.0 for r2 in REGIMES}
        else:
            norm[r] = {r2: 0.0 for r2 in REGIMES}
    return norm


def print_ascii_matrix(norm: dict) -> list:
    lines = []
    lines.append("REGIME TRANSITION NETWORK")
    lines.append("=" * 50)
    header = "        " + "".join(f"  → {r[:4]}" for r in REGIMES)
    lines.append(header)
    for frm in REGIMES:
        row = f"{frm:<10}"
        for to in REGIMES:
            val = norm[frm].get(to, 0.0)
            row += f"  {val:5.1f}%"
        lines.append(row)
    lines.append("")

    # Find self-transition stats
    self_trans = [(r, norm[r][r]) for r in REGIMES]
    most_stable = max(self_trans, key=lambda x: x[1])
    lines.append(f"Most stable: {most_stable[0]} ({most_stable[1]:.1f}% self-transition) — once in, stays there")

    # Find fastest exit to BULL
    exits_to_bull = [(r, norm[r]["BULL"]) for r in REGIMES if r != "BULL"]
    if exits_to_bull:
        fastest = max(exits_to_bull, key=lambda x: x[1])
        lines.append(f"Fastest entry to BULL: from {fastest[0]} ({fastest[1]:.1f}%) — vol spikes revert to rally")

    return lines


def build_pyvis_html(tm: dict, norm: dict, success_rates: dict, seq: list) -> str:
    """Build PyVis interactive HTML or fall back to static HTML."""
    try:
        from pyvis.network import Network
        import networkx as nx

        net = Network(height="600px", width="100%", directed=True, bgcolor="#1a1a2e",
                      font_color="white")
        net.set_options("""
        {
          "physics": {"enabled": true, "stabilization": {"iterations": 100}},
          "edges": {"smooth": {"type": "curvedCW", "roundness": 0.2}},
          "nodes": {"font": {"size": 16}}
        }
        """)

        # Count time in each regime
        regime_counts = {r: seq.count(r) for r in REGIMES}
        total_obs = max(1, len(seq))

        for r in REGIMES:
            size = 20 + 60 * (regime_counts[r] / total_obs)
            title = (f"<b>{r}</b><br>"
                     f"Observations: {regime_counts[r]} ({regime_counts[r]/total_obs*100:.1f}%)<br>"
                     f"Self-transition: {norm[r][r]:.1f}%")
            net.add_node(r, label=r, size=size, color=REGIME_COLORS.get(r, "#888"),
                         title=title)

        # Add edges
        raw_max = max(
            (tm[frm][to] for frm in REGIMES for to in REGIMES if frm != to),
            default=1
        )
        for frm in REGIMES:
            for to in REGIMES:
                count = tm[frm][to]
                if count == 0:
                    continue
                width = 1 + 8 * (count / max(1, raw_max))
                sr = success_rates[frm][to]
                # Color: green = high success, red = low
                r_val = int(255 * (1 - sr))
                g_val = int(255 * sr)
                color  = f"rgb({r_val},{g_val},80)"
                title  = (f"{frm} → {to}<br>"
                          f"Count: {count} ({norm[frm][to]:.1f}%)<br>"
                          f"Trade success: {sr*100:.1f}%")
                net.add_edge(frm, to, width=width, color=color,
                             title=title, label=f"{norm[frm][to]:.0f}%")

        html = net.generate_html()
        return html
    except ImportError:
        return None


def build_static_html(norm: dict, success_rates: dict) -> str:
    """Fallback: simple static HTML table."""
    rows = ""
    for frm in REGIMES:
        row_cells = f"<td><b>{frm}</b></td>"
        for to in REGIMES:
            pct = norm[frm][to]
            sr  = success_rates[frm][to]
            r_v = int(255 * (1 - sr))
            g_v = int(255 * sr)
            bg  = f"rgb({r_v},{g_v},80)"
            row_cells += f'<td style="background:{bg};color:#fff;padding:6px">{pct:.1f}%</td>'
        rows += f"<tr>{row_cells}</tr>"

    header_cells = "<th></th>" + "".join(f"<th>{r}</th>" for r in REGIMES)
    return f"""<!DOCTYPE html>
<html>
<head><title>Regime Transition Network</title>
<style>
  body {{background:#1a1a2e;color:#eee;font-family:monospace;padding:20px}}
  table {{border-collapse:collapse;width:600px}}
  th,td {{border:1px solid #444;padding:8px;text-align:center}}
  th {{background:#2c3e50}}
</style>
</head>
<body>
<h2>Regime Transition Network — LARSA v1</h2>
<p>Edge color: green = high win-rate, red = low win-rate in destination regime</p>
<table>
  <tr>{header_cells}</tr>
  {rows}
</table>
<p style="color:#888">Install pyvis for interactive version: pip install pyvis networkx</p>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Regime transition network builder")
    parser.add_argument("--csv", default=None, help="Price CSV (optional)")
    args = parser.parse_args()

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    wells = data["wells"]

    # ── Infer regime sequence ──────────────────────────────────────────────────
    seq = infer_regime_sequence(wells)

    # ── Build transition matrix ────────────────────────────────────────────────
    tm   = build_transition_matrix(seq)
    norm = row_normalize(tm)
    sr   = transition_success_rate(wells, seq)

    # ── Print ASCII matrix ─────────────────────────────────────────────────────
    lines = print_ascii_matrix(norm)

    # Additional stats
    lines.append("")
    lines.append("REGIME STATISTICS:")
    regime_counts = {r: seq.count(r) for r in REGIMES}
    for r in REGIMES:
        pct = regime_counts[r] / max(1, len(seq)) * 100
        # Win rate in this regime
        regime_wells = [wells[i] for i, s in enumerate(seq) if s == r and i < len(wells)]
        wr = sum(1 for w in regime_wells if w.get("is_win")) / max(1, len(regime_wells)) * 100
        lines.append(f"  {r:<12} {pct:5.1f}% of time   WR={wr:.1f}% ({len(regime_wells)} trades)")

    # ── NetworkX summary ───────────────────────────────────────────────────────
    try:
        import networkx as nx
        G = nx.DiGraph()
        for frm in REGIMES:
            for to in REGIMES:
                if tm[frm][to] > 0:
                    G.add_edge(frm, to, weight=tm[frm][to])
        lines.append("")
        lines.append("NETWORKX GRAPH SUMMARY:")
        lines.append(f"  Nodes: {G.number_of_nodes()}")
        lines.append(f"  Edges: {G.number_of_edges()}")
        # Centrality
        bc = nx.betweenness_centrality(G, weight="weight")
        top_central = max(bc, key=bc.get)
        lines.append(f"  Most central node: {top_central} (betweenness={bc[top_central]:.3f})")
        lines.append("  networkx available")
    except ImportError:
        lines.append("")
        lines.append("  networkx not installed (pip install networkx)")

    output = "\n".join(lines)
    print(output)

    # ── Save HTML ──────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    html = build_pyvis_html(tm, norm, sr, seq)
    pyvis_ok = html is not None
    if html is None:
        html = build_static_html(norm, sr)
        print("  pyvis not installed — saved static HTML fallback (pip install pyvis)")

    html_path = os.path.join(RESULTS_DIR, "regime_network.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {html_path} ({'interactive PyVis' if pyvis_ok else 'static fallback'})")

    # ── Save markdown ──────────────────────────────────────────────────────────
    md_path = os.path.join(RESULTS_DIR, "regime_transitions.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Regime Transition Network — LARSA v1\n\n")
        f.write("```\n")
        f.write(output)
        f.write("\n```\n")
        f.write("\n## Transition Matrix (row → column, %)\n\n")
        f.write("| From\\To | " + " | ".join(REGIMES) + " |\n")
        f.write("|" + "---------|" * (len(REGIMES) + 1) + "\n")
        for frm in REGIMES:
            row_vals = " | ".join(f"{norm[frm][to]:.1f}%" for to in REGIMES)
            f.write(f"| {frm:<12} | {row_vals} |\n")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
