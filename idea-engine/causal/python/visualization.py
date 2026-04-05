"""
causal/python/visualization.py

CausalVisualizer: renders the causal DAG as an interactive HTML file.

Primary renderer: pyvis (interactive, network-style)
Fallback renderer: matplotlib + networkx (static PNG)

Output: idea-engine/causal/output/dag.html (and dag.png if matplotlib fallback)

Visual encoding:
    Node size    = number of outgoing causal edges (out-degree)
    Node colour  = feature category (mass / regime / calendar / volatility / outcome)
    Edge colour  = causal strength (effect_size), green = strong, yellow = moderate, red = weak
    Edge width   = 1 + 4 * effect_size
    Edge label   = "lag=N, p=X.XXX"
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("C:/Users/Matthew/srfm-lab/idea-engine/causal/output")

# ---------------------------------------------------------------------------
# Category colours for nodes
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "mass":     "#4CAF50",   # green
    "regime":   "#2196F3",   # blue
    "calendar": "#FF9800",   # orange
    "vol":      "#9C27B0",   # purple
    "outcome":  "#F44336",   # red
    "momentum": "#00BCD4",   # cyan
    "other":    "#9E9E9E",   # grey
}

OUTCOME_KEYWORDS = {"pnl", "win", "loss", "return", "profit", "drawdown", "trade"}
MASS_KEYWORDS = {"bh_mass", "mass", "btc_dominance"}
REGIME_KEYWORDS = {"tf_score", "regime", "cluster"}
CALENDAR_KEYWORDS = {"hour", "day_of_week", "day", "session"}
VOL_KEYWORDS = {"atr", "garch", "vol", "std", "variance"}
MOMENTUM_KEYWORDS = {"momentum", "ou_zscore", "zscore", "roc"}


def _categorize_node(name: str) -> str:
    n = name.lower()
    if any(k in n for k in OUTCOME_KEYWORDS):
        return "outcome"
    if any(k in n for k in MASS_KEYWORDS):
        return "mass"
    if any(k in n for k in REGIME_KEYWORDS):
        return "regime"
    if any(k in n for k in CALENDAR_KEYWORDS):
        return "calendar"
    if any(k in n for k in VOL_KEYWORDS):
        return "vol"
    if any(k in n for k in MOMENTUM_KEYWORDS):
        return "momentum"
    return "other"


def _edge_color(effect_size: float) -> str:
    """Green (strong) → yellow (moderate) → red (weak)."""
    if effect_size >= 0.3:
        return "#4CAF50"
    if effect_size >= 0.1:
        return "#FFC107"
    return "#F44336"


def _edge_width(effect_size: float) -> float:
    return 1.0 + 4.0 * min(effect_size, 1.0)


def _node_size_pyvis(out_degree: int, base: int = 20) -> int:
    return base + out_degree * 8


def _node_size_mpl(out_degree: int, base: float = 300.0) -> float:
    return base + out_degree * 120.0


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class CausalVisualizer:
    """
    Renders a CausalGraph to HTML using pyvis, with matplotlib fallback.

    Parameters
    ----------
    output_dir : directory to write output files
    height     : canvas height (px) for pyvis
    width      : canvas width (px / %) for pyvis
    """

    def __init__(
        self,
        output_dir: Path | str = OUTPUT_DIR,
        height: str = "750px",
        width: str = "100%",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.height = height
        self.width = width

    def render(
        self,
        graph: Any,  # CausalGraph or nx.DiGraph
        filename: str = "dag.html",
        title: str = "Causal DAG — Idea Automation Engine",
        show_edge_labels: bool = True,
    ) -> Path:
        """
        Render the graph to HTML (pyvis) or PNG (matplotlib fallback).
        Returns the path to the output file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accept either CausalGraph or raw nx.DiGraph
        import networkx as nx
        if hasattr(graph, "graph"):
            g: nx.DiGraph = graph.graph
        else:
            g = graph

        if g.number_of_nodes() == 0:
            log.warning("CausalVisualizer: empty graph, skipping render")
            return self.output_dir / filename

        try:
            return self._render_pyvis(g, filename, title, show_edge_labels)
        except ImportError:
            log.info("pyvis not available, falling back to matplotlib")
            return self._render_matplotlib(g, filename.replace(".html", ".png"), title)
        except Exception as exc:
            log.warning("pyvis render failed (%s), falling back to matplotlib", exc)
            return self._render_matplotlib(g, filename.replace(".html", ".png"), title)

    # ------------------------------------------------------------------
    # pyvis renderer
    # ------------------------------------------------------------------

    def _render_pyvis(
        self,
        g: Any,
        filename: str,
        title: str,
        show_edge_labels: bool,
    ) -> Path:
        from pyvis.network import Network

        net = Network(
            height=self.height,
            width=self.width,
            directed=True,
            notebook=False,
            heading=title,
        )

        # Physics options for cleaner layout
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 150,
              "springConstant": 0.05,
              "damping": 0.09
            },
            "stabilization": { "iterations": 200 }
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 } },
            "smooth": { "type": "dynamic" }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true
          }
        }
        """)

        # Add nodes
        for node in g.nodes:
            out_deg = g.out_degree(node)
            category = _categorize_node(node)
            color = CATEGORY_COLORS[category]
            size = _node_size_pyvis(out_deg)

            tooltip = (
                f"<b>{node}</b><br>"
                f"Category: {category}<br>"
                f"Out-degree (causal influence): {out_deg}<br>"
                f"In-degree (being caused): {g.in_degree(node)}"
            )

            net.add_node(
                node,
                label=node,
                title=tooltip,
                color=color,
                size=size,
                font={"size": 14, "color": "#ffffff" if category == "outcome" else "#222222"},
            )

        # Add edges
        for cause, effect, data in g.edges(data=True):
            eff_size = float(data.get("effect_size", 0.0))
            lag = int(data.get("lag", 1))
            p_val = float(data.get("p_value", 1.0))
            color = _edge_color(eff_size)
            width = _edge_width(eff_size)

            label = f"lag={lag}, p={p_val:.3f}" if show_edge_labels else ""
            tooltip = (
                f"{cause} → {effect}<br>"
                f"Lag: {lag} bars<br>"
                f"p-value: {p_val:.4f}<br>"
                f"Effect size: {eff_size:.4f}<br>"
                f"F-statistic: {data.get('f_statistic', 0.0):.3f}"
            )

            net.add_edge(
                cause, effect,
                title=tooltip,
                label=label,
                color=color,
                width=width,
                arrows="to",
            )

        out_path = self.output_dir / filename
        net.save_graph(str(out_path))
        log.info("Causal DAG saved to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # matplotlib fallback
    # ------------------------------------------------------------------

    def _render_matplotlib(
        self,
        g: Any,
        filename: str,
        title: str,
    ) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")

        # Layout
        try:
            pos = nx.spring_layout(g, k=2.0, seed=42)
        except Exception:
            pos = nx.random_layout(g, seed=42)

        # Node properties
        node_colors = [CATEGORY_COLORS[_categorize_node(n)] for n in g.nodes]
        node_sizes = [_node_size_mpl(g.out_degree(n)) for n in g.nodes]

        # Edge properties
        edge_colors = [
            _edge_color(g[u][v].get("effect_size", 0.0))
            for u, v in g.edges
        ]
        edge_widths = [
            _edge_width(g[u][v].get("effect_size", 0.0))
            for u, v in g.edges
        ]

        nx.draw_networkx_nodes(
            g, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
        )
        nx.draw_networkx_labels(
            g, pos, ax=ax,
            font_size=9,
            font_color="white",
            font_weight="bold",
        )
        nx.draw_networkx_edges(
            g, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            alpha=0.8,
        )

        # Edge labels: lag
        edge_labels = {
            (u, v): f"lag={g[u][v].get('lag', '?')}"
            for u, v in g.edges
        }
        nx.draw_networkx_edge_labels(
            g, pos, edge_labels=edge_labels, ax=ax, font_size=7
        )

        # Legend
        legend_patches = [
            mpatches.Patch(color=c, label=cat)
            for cat, c in CATEGORY_COLORS.items()
            if any(_categorize_node(n) == cat for n in g.nodes)
        ]
        ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.8)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Causal DAG (matplotlib) saved to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Legend / summary HTML
    # ------------------------------------------------------------------

    def render_summary_html(
        self,
        graph: Any,
        filename: str = "dag_summary.html",
    ) -> Path:
        """
        Generate a simple HTML table summarising the most significant edges.
        """
        import networkx as nx
        if hasattr(graph, "graph"):
            g: nx.DiGraph = graph.graph
        else:
            g = graph

        all_edges = sorted(
            [
                (u, v, d)
                for u, v, d in g.edges(data=True)
            ],
            key=lambda x: x[2].get("effect_size", 0.0),
            reverse=True,
        )

        rows_html = ""
        for cause, effect, data in all_edges[:50]:
            eff = data.get("effect_size", 0.0)
            lag = data.get("lag", "?")
            p = data.get("p_value", 1.0)
            bar_width = int(min(eff * 300, 200))
            rows_html += f"""
            <tr>
              <td>{cause}</td>
              <td>{effect}</td>
              <td>{lag}</td>
              <td>{p:.4f}</td>
              <td>{eff:.4f}</td>
              <td><div style="width:{bar_width}px;height:12px;
                  background:{_edge_color(eff)};border-radius:3px;"></div></td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Causal Edge Summary</title>
  <style>
    body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
    h1 {{ color: #4CAF50; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #16213e; color: #4CAF50; padding: 10px; text-align: left; }}
    td {{ padding: 8px; border-bottom: 1px solid #333; }}
    tr:hover td {{ background: #16213e; }}
  </style>
</head>
<body>
  <h1>Causal Edge Summary</h1>
  <p>{g.number_of_nodes()} nodes, {g.number_of_edges()} edges</p>
  <table>
    <tr>
      <th>Cause</th><th>Effect</th><th>Lag</th>
      <th>p-value</th><th>Effect Size</th><th>Strength</th>
    </tr>
    {rows_html}
  </table>
</body>
</html>"""

        out_path = self.output_dir / filename
        out_path.write_text(html, encoding="utf-8")
        log.info("Causal summary HTML saved to %s", out_path)
        return out_path
