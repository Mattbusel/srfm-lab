"""
visualizer.py — GenealogyVisualizer: render the genome family tree.

Produces:
  - Graphviz DOT strings (for static rendering with `dot -Tsvg`)
  - D3.js force-directed JSON (for the idea-dashboard frontend)
  - Fitness heatmap data (generation × island grid)
  - Mutation frequency analysis
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter, defaultdict
from typing import Any

from .tree import GenealogyTree, GenomeNode

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _fitness_colour(fitness: float | None,
                     vmin: float = -1.0,
                     vmax: float = 3.0) -> str:
    """
    Map a fitness value to a hex colour string.

    - Green  (#00c851) for high fitness  (≥ vmax)
    - Red    (#ff3547) for low fitness   (≤ vmin)
    - Yellow (#ffbb33) for mid range
    - Grey   (#aaaaaa) for None / unknown
    """
    if fitness is None:
        return "#aaaaaa"

    if vmax <= vmin:
        return "#aaaaaa"

    t = (fitness - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))

    if t < 0.5:
        # Red → Yellow
        r  = 255
        g  = int(t * 2 * 187)      # 0 → 187
        b  = int(t * 2 * 51)       # 0 → 51
    else:
        # Yellow → Green
        t2 = (t - 0.5) * 2
        r  = int((1 - t2) * 255)
        g  = int(187 + t2 * (200 - 187))
        b  = int(51  + t2 * (81  - 51))

    return f"#{r:02x}{g:02x}{b:02x}"


_HOF_COLOUR    = "#ffd700"  # gold
_ROOT_COLOUR   = "#6c8ebf"  # blue
_PRUNED_COLOUR = "#dddddd"


# ---------------------------------------------------------------------------
# GenealogyVisualizer
# ---------------------------------------------------------------------------


class GenealogyVisualizer:
    """
    Renders a GenealogyTree as various graph formats for display.

    Parameters
    ----------
    tree:
        A GenealogyTree instance (must already be populated).
    fitness_min:
        Minimum fitness for colour scaling (default -1.0).
    fitness_max:
        Maximum fitness for colour scaling (default 3.0).
    """

    def __init__(
        self,
        tree: GenealogyTree,
        fitness_min: float = -1.0,
        fitness_max: float = 3.0,
    ) -> None:
        self._tree        = tree
        self._fitness_min = fitness_min
        self._fitness_max = fitness_max

    # ------------------------------------------------------------------
    # Graphviz DOT
    # ------------------------------------------------------------------

    def to_dot(self, max_nodes: int = 100) -> str:
        """
        Produce a Graphviz DOT string representing the genealogy tree.

        Nodes are coloured by fitness (green=good, red=bad, gold=HoF).
        Edges point from parent to child.

        Parameters
        ----------
        max_nodes:
            Cap on number of nodes to include (selects highest-fitness subset).
        """
        all_nodes = self._tree.all_nodes()
        # Select top-N by fitness (keep HoF regardless)
        hof_ids = {n.genome_id for n in all_nodes if n.is_hall_of_fame}
        sorted_nodes = sorted(
            all_nodes,
            key=lambda n: n.fitness if n.fitness is not None else -999,
            reverse=True,
        )
        selected_ids: set[int] = set()
        for n in sorted_nodes:
            if len(selected_ids) >= max_nodes and n.genome_id not in hof_ids:
                break
            selected_ids.add(n.genome_id)
        selected_ids.update(hof_ids)

        node_map = {n.genome_id: n for n in all_nodes if n.genome_id in selected_ids}

        lines = [
            'digraph Genealogy {',
            '    graph [rankdir=TB, fontname="Helvetica", bgcolor="#1a1a2e"];',
            '    node  [shape=circle, style=filled, fontsize=9, fontcolor=white, '
            'fontname="Helvetica"];',
            '    edge  [color="#555566", arrowsize=0.7];',
        ]

        # Cluster by island
        islands = sorted({n.island for n in node_map.values()})
        for island in islands:
            island_nodes = [n for n in node_map.values() if n.island == island]
            lines.append(f'    subgraph "cluster_{island}" {{')
            lines.append(f'        label="{html.escape(island)}";')
            lines.append('        color="#444455"; fontcolor="#cccccc";')
            for n in island_nodes:
                colour = _HOF_COLOUR if n.is_hall_of_fame else _fitness_colour(
                    n.fitness, self._fitness_min, self._fitness_max
                )
                fitness_str = f"{n.fitness:.3f}" if n.fitness is not None else "N/A"
                label = f"G{n.generation}\\n{fitness_str}"
                lines.append(
                    f'        {n.genome_id} [label="{label}", '
                    f'fillcolor="{colour}", '
                    f'tooltip="genome={n.genome_id} island={n.island} "gen={n.generation}"];'
                )
            lines.append("    }")

        # Edges (only between selected nodes)
        all_edges = self._tree.all_edges()
        for child_id, parent_id in all_edges:
            if child_id in selected_ids and parent_id in selected_ids:
                lines.append(f"    {parent_id} -> {child_id};")

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # D3.js JSON
    # ------------------------------------------------------------------

    def to_d3_json(self, max_nodes: int = 100) -> dict[str, Any]:
        """
        Produce a JSON dict suitable for a D3.js force-directed graph.

        Schema::

            {
              "nodes": [
                {"id": 42, "label": "42", "fitness": 1.84,
                 "colour": "#00c851", "island": "alpha",
                 "generation": 5, "is_hof": false},
                ...
              ],
              "links": [
                {"source": 17, "target": 42},
                ...
              ]
            }
        """
        all_nodes = self._tree.all_nodes()
        sorted_nodes = sorted(
            all_nodes,
            key=lambda n: n.fitness if n.fitness is not None else -999,
            reverse=True,
        )
        hof_ids = {n.genome_id for n in all_nodes if n.is_hall_of_fame}

        selected: list[GenomeNode] = []
        selected_ids: set[int] = set()
        for n in sorted_nodes:
            if len(selected_ids) >= max_nodes and n.genome_id not in hof_ids:
                break
            selected.append(n)
            selected_ids.add(n.genome_id)
        for n in all_nodes:
            if n.genome_id in hof_ids and n.genome_id not in selected_ids:
                selected.append(n)
                selected_ids.add(n.genome_id)

        nodes_json = [
            {
                "id":         n.genome_id,
                "label":      str(n.genome_id),
                "fitness":    n.fitness,
                "colour":     _HOF_COLOUR if n.is_hall_of_fame else _fitness_colour(
                    n.fitness, self._fitness_min, self._fitness_max
                ),
                "island":     n.island,
                "generation": n.generation,
                "is_hof":     n.is_hall_of_fame,
                "mutation_ops": n.mutation_ops,
            }
            for n in selected
        ]

        all_edges = self._tree.all_edges()
        links_json = [
            {"source": parent_id, "target": child_id}
            for child_id, parent_id in all_edges
            if child_id in selected_ids and parent_id in selected_ids
        ]

        return {
            "nodes": nodes_json,
            "links": links_json,
            "metadata": {
                "total_nodes":   self._tree.size(),
                "shown_nodes":   len(nodes_json),
                "islands":       self._tree.islands(),
                "fitness_min":   self._fitness_min,
                "fitness_max":   self._fitness_max,
            },
        }

    def to_d3_json_str(self, max_nodes: int = 100) -> str:
        """Return D3 JSON as a string."""
        return json.dumps(self.to_d3_json(max_nodes), indent=2)

    # ------------------------------------------------------------------
    # Fitness heatmap
    # ------------------------------------------------------------------

    def fitness_heatmap_data(self) -> dict[str, Any]:
        """
        Build a 2-D grid of mean fitness values: rows = generation, cols = island.

        Returns a dict::

            {
              "islands":     ["alpha", "beta", ...],
              "generations": [0, 1, 2, ...],
              "grid":        [[mean_fitness_or_null, ...], ...],  # [gen][island_idx]
              "vmin":        float,
              "vmax":        float,
            }
        """
        summary = self._tree.generation_summary()
        if not summary:
            return {"islands": [], "generations": [], "grid": [], "vmin": 0, "vmax": 0}

        islands     = sorted({s["island"] for s in summary})
        generations = sorted({s["generation"] for s in summary})

        island_idx  = {isl: i for i, isl in enumerate(islands)}
        gen_idx     = {gen: i for i, gen in enumerate(generations)}

        grid: list[list[float | None]] = [
            [None] * len(islands) for _ in generations
        ]

        for row in summary:
            gi = gen_idx[row["generation"]]
            ii = island_idx[row["island"]]
            grid[gi][ii] = row["mean_fitness"]

        all_vals = [v for row in grid for v in row if v is not None]
        vmin     = min(all_vals) if all_vals else 0.0
        vmax     = max(all_vals) if all_vals else 1.0

        return {
            "islands":     islands,
            "generations": generations,
            "grid":        grid,
            "vmin":        vmin,
            "vmax":        vmax,
        }

    # ------------------------------------------------------------------
    # Mutation frequency
    # ------------------------------------------------------------------

    def mutation_frequency_data(self, top_k: int = 20) -> dict[str, Any]:
        """
        Analyse which mutation operations appear most frequently in the
        top-performing genomes.

        Parameters
        ----------
        top_k:
            Consider only the top-K genomes by fitness.

        Returns
        -------
        dict::

            {
              "top_performers":   [{"genome_id": ..., "fitness": ...,
                                    "mutation_ops": [...]}, ...],
              "mutation_counts":  {"crossover": 42, "mutate_threshold": 18, ...},
              "total_mutations":  int,
              "top_mutation":     str,
              "enrichment":       {"crossover": float, ...}  # count / total
            }
        """
        all_nodes = self._tree.all_nodes()
        sorted_nodes = sorted(
            [n for n in all_nodes if n.fitness is not None],
            key=lambda n: n.fitness,  # type: ignore[arg-type]
            reverse=True,
        )
        top_performers = sorted_nodes[:top_k]

        # Count mutations in top performers
        top_counts: Counter[str] = Counter()
        for n in top_performers:
            for op in n.mutation_ops:
                top_counts[op] += 1

        # Count mutations across ALL genomes
        all_counts: Counter[str] = Counter()
        for n in all_nodes:
            for op in n.mutation_ops:
                all_counts[op] += 1

        total = sum(all_counts.values()) or 1
        total_top = sum(top_counts.values()) or 1

        # Enrichment: (freq in top) / (freq overall)
        enrichment = {}
        for op, cnt in top_counts.items():
            base_rate = all_counts.get(op, 1) / total
            top_rate  = cnt / total_top
            enrichment[op] = round(top_rate / base_rate, 3) if base_rate > 0 else 0.0

        top_mutation = top_counts.most_common(1)[0][0] if top_counts else None

        return {
            "top_performers": [
                {
                    "genome_id":    n.genome_id,
                    "fitness":      n.fitness,
                    "mutation_ops": n.mutation_ops,
                }
                for n in top_performers
            ],
            "mutation_counts": dict(top_counts.most_common()),
            "total_mutations": sum(top_counts.values()),
            "top_mutation":    top_mutation,
            "enrichment":      enrichment,
        }

    # ------------------------------------------------------------------
    # Fitness arc chart data
    # ------------------------------------------------------------------

    def fitness_arc_chart_data(self, genome_id: int) -> dict[str, Any]:
        """
        Return Vega-Lite-compatible chart data for the fitness arc of a genome.

        Schema::

            {
              "values": [
                {"generation": 0, "fitness": 0.5, "genome_id": 1, "mutation": "root"},
                ...
              ]
            }
        """
        arc = self._tree.fitness_arc(genome_id)
        values = [
            {
                "generation":    step["generation"],
                "fitness":       step["fitness"],
                "genome_id":     step["genome_id"],
                "mutation":      ", ".join(step.get("mutation_ops") or []) or "root",
                "cumulative_improvement": step.get("cumulative_improvement"),
            }
            for step in arc
        ]
        return {"values": values, "genome_id": genome_id}
