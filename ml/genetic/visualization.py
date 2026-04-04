"""
Visualization for genetic algorithm optimization results.

Provides:
- Fitness landscape plots
- Pareto front animation (static frames and animation)
- Parameter convergence plots
- Genealogy tree
- Diversity metrics over time
- Strategy performance attribution
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Graceful imports for plotting dependencies
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .genome import StrategyGenome, GenomeFactory
from .population import PopulationStats, HallOfFame, DiversityMetrics
from .evolution import EvolutionResult


# ---------------------------------------------------------------------------
# ASCII fallback for environments without matplotlib
# ---------------------------------------------------------------------------

class ASCIIPlotter:
    """Simple ASCII art plotter for environments without matplotlib."""

    @staticmethod
    def bar_chart(values: List[float], labels: Optional[List[str]] = None,
                  title: str = "", width: int = 50) -> str:
        if not values:
            return f"{title}\n(no data)"
        max_val = max(abs(v) for v in values)
        if max_val == 0:
            max_val = 1.0
        lines = [title, ""]
        for i, v in enumerate(values):
            label = labels[i] if labels else f"[{i}]"
            bar_len = int(abs(v) / max_val * width)
            bar = "#" * bar_len
            sign = "+" if v >= 0 else "-"
            lines.append(f"{label:20s} | {sign}{bar:<{width}} {v:.4f}")
        return "\n".join(lines)

    @staticmethod
    def line_chart(series: List[float], title: str = "",
                   width: int = 60, height: int = 15) -> str:
        if not series:
            return f"{title}\n(no data)"
        min_v = min(series)
        max_v = max(series)
        span = max_v - min_v if max_v != min_v else 1.0

        chart = [[" "] * width for _ in range(height)]
        n = len(series)

        for col in range(width):
            idx = min(int(col * n / width), n - 1)
            val = series[idx]
            row = int((max_v - val) / span * (height - 1))
            row = max(0, min(height - 1, row))
            chart[row][col] = "*"

        lines = [f"{title} (min={min_v:.4f}, max={max_v:.4f})", ""]
        for row_idx, row in enumerate(chart):
            y_val = max_v - row_idx * span / max(height - 1, 1)
            lines.append(f"{y_val:8.4f} |" + "".join(row))
        lines.append(f"{'':9s}+" + "-" * width)
        lines.append(f"{'':9s}0{' ' * (width // 2 - 1)}{n}")
        return "\n".join(lines)

    @staticmethod
    def scatter_2d(xs: List[float], ys: List[float],
                   labels: Optional[List[str]] = None,
                   title: str = "", width: int = 60, height: int = 20) -> str:
        if not xs or not ys:
            return f"{title}\n(no data)"
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max_x - min_x if max_x != min_x else 1.0
        span_y = max_y - min_y if max_y != min_y else 1.0

        grid = [[" "] * width for _ in range(height)]
        for x, y in zip(xs, ys):
            col = int((x - min_x) / span_x * (width - 1))
            row = int((max_y - y) / span_y * (height - 1))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            grid[row][col] = "o"

        lines = [f"{title} ({len(xs)} points)", ""]
        for row_idx, row in enumerate(grid):
            y_val = max_y - row_idx * span_y / max(height - 1, 1)
            lines.append(f"{y_val:8.4f} |" + "".join(row))
        lines.append(f"{'':9s}+" + "-" * width)
        lines.append(f"        {min_x:.4f}{' ' * (width - 12)}{max_x:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------

@dataclass
class PlotConfig:
    output_dir: str = "./plots"
    dpi: int = 150
    figsize: Tuple[int, int] = (12, 8)
    style: str = "seaborn-v0_8-darkgrid"
    colormap: str = "viridis"
    format: str = "png"
    show_plots: bool = False
    font_size: int = 10
    title_size: int = 12


# ---------------------------------------------------------------------------
# Fitness landscape visualization
# ---------------------------------------------------------------------------

class FitnessLandscapeVisualizer:
    """
    Visualizes the fitness landscape by projecting high-dimensional
    parameter space onto 2D using PCA or random projections.
    """

    def __init__(self, config: PlotConfig) -> None:
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

    def _pca_projection(self, vectors: List[List[float]]) -> Tuple[List[float], List[float]]:
        """Simple 2D PCA projection (pure Python, no numpy)."""
        if not vectors or len(vectors[0]) < 2:
            return [0.0] * len(vectors), [0.0] * len(vectors)

        n = len(vectors)
        d = len(vectors[0])

        # Center
        means = [sum(v[j] for v in vectors) / n for j in range(d)]
        centered = [[v[j] - means[j] for j in range(d)] for v in vectors]

        # Compute first two principal components via power iteration
        def _matrix_vec_multiply(mat: List[List[float]], vec: List[float]) -> List[float]:
            result = [sum(mat[i][j] * vec[j] for j in range(len(vec)))
                      for i in range(len(mat))]
            return result

        def _covariance_mat_vec(data: List[List[float]], vec: List[float]) -> List[float]:
            """Compute covariance_matrix @ vec without forming full covariance matrix."""
            # (X^T X) v = X^T (X v), but we need (X X^T)... use data directly
            # Av = X^T (X v) / n
            n_d = len(data)
            d_v = len(vec)
            # X v
            projections = [sum(data[i][j] * vec[j] for j in range(d_v))
                           for i in range(n_d)]
            # X^T proj
            result = [sum(projections[i] * data[i][j] for i in range(n_d)) / n_d
                      for j in range(d_v)]
            return result

        def _power_iteration(data: List[List[float]], n_iter: int = 20) -> List[float]:
            rng = random.Random(42)
            d_dim = len(data[0])
            vec = [rng.gauss(0, 1) for _ in range(d_dim)]
            norm = math.sqrt(sum(x * x for x in vec))
            vec = [x / norm for x in vec]
            for _ in range(n_iter):
                vec = _covariance_mat_vec(data, vec)
                norm = math.sqrt(sum(x * x for x in vec))
                if norm < 1e-10:
                    break
                vec = [x / norm for x in vec]
            return vec

        pc1 = _power_iteration(centered)
        # Deflate
        pc1_dot = [sum(c[j] * pc1[j] for j in range(d)) for c in centered]
        deflated = [[c[j] - pc1_dot[i] * pc1[j]
                     for j in range(d)]
                    for i, c in enumerate(centered)]
        pc2 = _power_iteration(deflated)

        xs = [sum(c[j] * pc1[j] for j in range(d)) for c in centered]
        ys = [sum(c[j] * pc2[j] for j in range(d)) for c in centered]
        return xs, ys

    def plot_fitness_landscape(self, population: List[StrategyGenome],
                                generation: int = 0) -> Optional[str]:
        """
        Plot fitness landscape: 2D PCA projection colored by fitness.
        Returns path to saved plot, or ASCII art if matplotlib unavailable.
        """
        evaluated = [g for g in population if g.fitness is not None]
        if not evaluated:
            return None

        vectors = [g.chromosome.to_float_vector() for g in evaluated]
        fitnesses = [g.fitness for g in evaluated]

        xs, ys = self._pca_projection(vectors)

        if not HAS_MATPLOTLIB:
            ascii_plot = ASCIIPlotter.scatter_2d(
                xs, ys,
                title=f"Fitness Landscape (Gen {generation}) - ASCII",
            )
            print(ascii_plot)
            return None

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        scatter = ax.scatter(xs, ys, c=fitnesses, cmap=self.config.colormap,
                              alpha=0.7, s=40)
        plt.colorbar(scatter, ax=ax, label="Fitness")
        ax.set_title(f"Fitness Landscape - PCA Projection (Gen {generation})",
                     fontsize=self.config.title_size)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Annotate best
        best_idx = fitnesses.index(max(fitnesses))
        ax.annotate("Best", (xs[best_idx], ys[best_idx]),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))

        plt.tight_layout()
        path = os.path.join(self.config.output_dir,
                            f"fitness_landscape_gen{generation:04d}.{self.config.format}")
        plt.savefig(path)
        plt.close()
        return path

    def plot_fitness_heatmap(self, param1: str, param2: str,
                              population: List[StrategyGenome]) -> Optional[str]:
        """
        Plot 2D heatmap of fitness vs two parameters.
        """
        evaluated = [g for g in population
                     if g.fitness is not None
                     and g.chromosome.get(param1) is not None
                     and g.chromosome.get(param2) is not None]
        if not evaluated:
            return None

        xs = [float(g.chromosome[param1]) for g in evaluated]
        ys = [float(g.chromosome[param2]) for g in evaluated]
        zs = [g.fitness for g in evaluated]

        if not HAS_MATPLOTLIB:
            ascii_plot = ASCIIPlotter.scatter_2d(
                xs, ys, title=f"Fitness Heatmap: {param1} vs {param2}")
            print(ascii_plot)
            return None

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        sc = ax.scatter(xs, ys, c=zs, cmap=self.config.colormap,
                         alpha=0.8, s=50)
        plt.colorbar(sc, ax=ax, label="Fitness")
        ax.set_xlabel(param1, fontsize=self.config.font_size)
        ax.set_ylabel(param2, fontsize=self.config.font_size)
        ax.set_title(f"Fitness by {param1} vs {param2}",
                     fontsize=self.config.title_size)
        plt.tight_layout()
        path = os.path.join(self.config.output_dir,
                            f"fitness_heatmap_{param1}_{param2}.{self.config.format}")
        plt.savefig(path)
        plt.close()
        return path


# ---------------------------------------------------------------------------
# Pareto front visualization
# ---------------------------------------------------------------------------

class ParetoFrontVisualizer:
    """
    Visualizes Pareto fronts for multi-objective optimization.
    Supports static plots and animation over generations.
    """

    def __init__(self, config: PlotConfig) -> None:
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        self._frame_data: List[List[StrategyGenome]] = []

    def add_frame(self, front: List[StrategyGenome]) -> None:
        """Add a Pareto front snapshot (for animation)."""
        self._frame_data.append([g for g in front if g.objectives])

    def plot_pareto_front_2d(self, front: List[StrategyGenome],
                              obj_names: Optional[List[str]] = None,
                              generation: int = 0,
                              highlight_knee: bool = True) -> Optional[str]:
        """
        Plot a 2D Pareto front.
        """
        candidates = [g for g in front if g.objectives is not None
                       and len(g.objectives) >= 2]
        if not candidates:
            return None

        obj1 = [g.objectives[0] for g in candidates]
        obj2 = [g.objectives[1] for g in candidates]
        obj1_name = (obj_names[0] if obj_names else "Objective 1")
        obj2_name = (obj_names[1] if obj_names else "Objective 2")

        if not HAS_MATPLOTLIB:
            ascii_plot = ASCIIPlotter.scatter_2d(
                obj1, obj2,
                title=f"Pareto Front (Gen {generation}): {obj1_name} vs {obj2_name}")
            print(ascii_plot)
            return None

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        # Sort by first objective for connected line
        sorted_pts = sorted(zip(obj1, obj2), key=lambda x: x[0])
        sorted_x = [p[0] for p in sorted_pts]
        sorted_y = [p[1] for p in sorted_pts]

        ax.plot(sorted_x, sorted_y, "b-", alpha=0.3, linewidth=1)
        ax.scatter(obj1, obj2, color="steelblue", s=50, zorder=5, alpha=0.8)

        if highlight_knee and len(candidates) >= 3:
            from .fitness import ParetoAnalysis
            knee = ParetoAnalysis.knee_point(candidates)
            if knee and knee.objectives:
                ax.scatter([knee.objectives[0]], [knee.objectives[1]],
                            color="red", s=150, zorder=10,
                            marker="*", label="Knee point")
                ax.legend(fontsize=self.config.font_size)

        ax.set_xlabel(obj1_name, fontsize=self.config.font_size)
        ax.set_ylabel(obj2_name, fontsize=self.config.font_size)
        ax.set_title(f"Pareto Front (Generation {generation}, "
                     f"N={len(candidates)})",
                     fontsize=self.config.title_size)
        plt.tight_layout()
        path = os.path.join(self.config.output_dir,
                            f"pareto_front_gen{generation:04d}.{self.config.format}")
        plt.savefig(path)
        plt.close()
        return path

    def animate_pareto_front(self, obj_names: Optional[List[str]] = None,
                              output_filename: str = "pareto_animation.gif") -> Optional[str]:
        """
        Create an animated GIF of Pareto front evolution across recorded frames.
        Requires matplotlib with pillow backend.
        """
        if not HAS_MATPLOTLIB or not self._frame_data:
            return None

        try:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

            # Compute overall axis limits
            all_obj1 = [g.objectives[0] for frame in self._frame_data
                        for g in frame if g.objectives]
            all_obj2 = [g.objectives[1] for frame in self._frame_data
                        for g in frame if g.objectives]
            if not all_obj1:
                plt.close()
                return None

            x_margin = (max(all_obj1) - min(all_obj1)) * 0.1
            y_margin = (max(all_obj2) - min(all_obj2)) * 0.1
            x_lim = (min(all_obj1) - x_margin, max(all_obj1) + x_margin)
            y_lim = (min(all_obj2) - y_margin, max(all_obj2) + y_margin)

            scatter_plot = ax.scatter([], [], color="steelblue", s=50)
            title = ax.set_title("", fontsize=self.config.title_size)
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_xlabel(obj_names[0] if obj_names else "Obj 1",
                           fontsize=self.config.font_size)
            ax.set_ylabel(obj_names[1] if obj_names else "Obj 2",
                           fontsize=self.config.font_size)

            def _update(frame_idx: int):
                frame = self._frame_data[frame_idx]
                obj1 = [g.objectives[0] for g in frame if g.objectives]
                obj2 = [g.objectives[1] for g in frame if g.objectives]
                scatter_plot.set_offsets(list(zip(obj1, obj2)) if obj1 else [[0, 0]])
                title.set_text(f"Pareto Front - Generation {frame_idx} "
                               f"(N={len(obj1)})")
                return scatter_plot, title

            anim = animation.FuncAnimation(
                fig, _update, frames=len(self._frame_data),
                blit=False, interval=200)

            path = os.path.join(self.config.output_dir, output_filename)
            try:
                anim.save(path, writer="pillow", fps=5)
            except Exception:
                # Fall back to saving individual frames
                for i, frame in enumerate(self._frame_data):
                    self.plot_pareto_front_2d(frame, obj_names, generation=i)
                path = None

            plt.close()
            return path
        except Exception as e:
            print(f"[WARNING] Animation failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Parameter convergence visualization
# ---------------------------------------------------------------------------

class ParameterConvergenceVisualizer:
    """
    Tracks and visualizes how parameter distributions converge over generations.
    """

    def __init__(self, config: PlotConfig) -> None:
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        self._history: Dict[str, List[List[float]]] = defaultdict(list)
        self._generation_count = 0

    def record_generation(self, population: List[StrategyGenome]) -> None:
        """Record parameter values for the current generation."""
        if not population:
            return
        params = [g.chromosome.to_dict() for g in population]
        all_keys = set()
        for p in params:
            all_keys.update(k for k, v in p.items() if isinstance(v, (int, float, bool)))

        for key in all_keys:
            vals = []
            for p in params:
                v = p.get(key)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, bool):
                    vals.append(1.0 if v else 0.0)
            if vals:
                self._history[key].append(vals)

        self._generation_count += 1

    def plot_parameter_convergence(self, param_names: Optional[List[str]] = None,
                                    max_params: int = 9) -> Optional[str]:
        """
        Plot mean ± std of each parameter over generations.
        """
        if not self._history:
            return None

        keys = param_names or list(self._history.keys())[:max_params]
        generations = list(range(self._generation_count))

        if not HAS_MATPLOTLIB:
            for key in keys[:3]:
                vals_by_gen = self._history[key]
                if vals_by_gen:
                    means = [sum(v) / len(v) for v in vals_by_gen]
                    chart = ASCIIPlotter.line_chart(
                        means, title=f"Param convergence: {key}")
                    print(chart)
            return None

        n_params = len(keys)
        n_cols = min(3, n_params)
        n_rows = math.ceil(n_params / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(6 * n_cols, 4 * n_rows),
                                  dpi=self.config.dpi)
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [list(axes)]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for idx, key in enumerate(keys):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]
            vals_by_gen = self._history[key]
            if not vals_by_gen:
                continue

            means = [sum(v) / len(v) for v in vals_by_gen]
            stds = [math.sqrt(sum((x - means[i]) ** 2 for x in v) / max(len(v) - 1, 1))
                    for i, v in enumerate(vals_by_gen)]

            gen_range = list(range(len(means)))
            ax.plot(gen_range, means, "b-", linewidth=1.5, label="Mean")
            ax.fill_between(gen_range,
                              [m - s for m, s in zip(means, stds)],
                              [m + s for m, s in zip(means, stds)],
                              alpha=0.2, color="blue")
            ax.set_title(key, fontsize=self.config.font_size)
            ax.set_xlabel("Generation")

        # Hide unused axes
        for idx in range(len(keys), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].set_visible(False)

        plt.suptitle("Parameter Convergence Over Generations",
                     fontsize=self.config.title_size)
        plt.tight_layout()
        path = os.path.join(self.config.output_dir,
                            f"param_convergence.{self.config.format}")
        plt.savefig(path)
        plt.close()
        return path

    def plot_parameter_distribution(self, population: List[StrategyGenome],
                                     generation: int = 0,
                                     n_bins: int = 20) -> Optional[str]:
        """Plot histogram of each parameter's distribution."""
        if not population or not HAS_MATPLOTLIB:
            return None

        params = [g.chromosome.to_dict() for g in population]
        numeric_keys = [k for k in (params[0] if params else {}).keys()
                        if isinstance(params[0].get(k), (int, float))][:9]

        if not numeric_keys:
            return None

        n_cols = min(3, len(numeric_keys))
        n_rows = math.ceil(len(numeric_keys) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(6 * n_cols, 4 * n_rows),
                                  dpi=self.config.dpi)

        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [list(axes)]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for idx, key in enumerate(numeric_keys):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]
            vals = [float(p[key]) for p in params if isinstance(p.get(key), (int, float))]
            ax.hist(vals, bins=n_bins, color="steelblue", alpha=0.7)
            ax.set_title(key, fontsize=self.config.font_size)

        plt.suptitle(f"Parameter Distributions (Gen {generation})",
                     fontsize=self.config.title_size)
        plt.tight_layout()
        path = os.path.join(self.config.output_dir,
                            f"param_distributions_gen{generation:04d}.{self.config.format}")
        plt.savefig(path)
        plt.close()
        return path


# ---------------------------------------------------------------------------
# Genealogy tree visualization
# ---------------------------------------------------------------------------

@dataclass
class GenealogyNode:
    """Node in a genealogy tree."""
    genome_id: str
    generation: int
    fitness: Optional[float]
    parent_ids: List[str]
    creation_method: str
    children_ids: List[str] = field(default_factory=list)


class GenealogyTracker:
    """
    Tracks lineage of genomes across generations and builds a genealogy graph.
    """

    def __init__(self, max_nodes: int = 500) -> None:
        self.max_nodes = max_nodes
        self._nodes: Dict[str, GenealogyNode] = {}

    def record(self, genome: StrategyGenome) -> None:
        """Record a genome in the genealogy."""
        if len(self._nodes) >= self.max_nodes:
            return  # Limit memory
        gid = genome.metadata.genome_id
        if gid not in self._nodes:
            node = GenealogyNode(
                genome_id=gid,
                generation=genome.metadata.generation,
                fitness=genome.fitness,
                parent_ids=list(genome.metadata.parent_ids),
                creation_method=genome.metadata.creation_method,
            )
            self._nodes[gid] = node
            # Link children to parents
            for pid in genome.metadata.parent_ids:
                if pid in self._nodes:
                    self._nodes[pid].children_ids.append(gid)
        else:
            # Update fitness if evaluated
            if genome.fitness is not None:
                self._nodes[gid].fitness = genome.fitness

    def record_population(self, population: List[StrategyGenome]) -> None:
        for g in population:
            self.record(g)

    def get_lineage(self, genome_id: str, depth: int = 10) -> List[GenealogyNode]:
        """Get ancestry chain for a genome (up to depth)."""
        lineage = []
        current_id = genome_id
        for _ in range(depth):
            if current_id not in self._nodes:
                break
            node = self._nodes[current_id]
            lineage.append(node)
            if not node.parent_ids:
                break
            current_id = node.parent_ids[0]
        return lineage

    def plot_genealogy(self, root_id: Optional[str] = None,
                        max_depth: int = 5,
                        config: Optional[PlotConfig] = None) -> Optional[str]:
        """
        Plot genealogy tree for the best individual (or given root_id).
        """
        if config is None:
            config = PlotConfig()

        if root_id is None:
            evaluated = [(gid, n) for gid, n in self._nodes.items()
                         if n.fitness is not None]
            if not evaluated:
                return None
            root_id = max(evaluated, key=lambda x: x[1].fitness or float("-inf"))[0]  # type: ignore

        lineage = self.get_lineage(root_id, depth=max_depth)
        if not lineage:
            return None

        # ASCII representation
        lines = ["Genealogy Tree:", ""]
        for depth, node in enumerate(lineage):
            indent = "  " * depth
            method = node.creation_method
            fitness_str = f"{node.fitness:.4f}" if node.fitness is not None else "N/A"
            lines.append(f"{indent}[Gen {node.generation}] {node.genome_id} "
                          f"(fitness={fitness_str}, method={method})")

        print("\n".join(lines))

        if not HAS_MATPLOTLIB:
            return None

        # Matplotlib tree
        os.makedirs(config.output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.axis("off")

        y_positions = list(range(len(lineage)))
        x_center = 0.5

        for i, node in enumerate(lineage):
            y = 1.0 - i / max(len(lineage), 1)
            fitness_str = f"f={node.fitness:.4f}" if node.fitness is not None else "unevaluated"
            label = f"[{node.genome_id}]\nGen {node.generation}\n{fitness_str}"
            ax.text(x_center, y, label, ha="center", va="center",
                     fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="lightblue", edgecolor="gray"))
            if i > 0:
                prev_y = 1.0 - (i - 1) / max(len(lineage), 1)
                ax.annotate("", xy=(x_center, y + 0.04),
                              xytext=(x_center, prev_y - 0.04),
                              arrowprops=dict(arrowstyle="->", color="gray"))

        ax.set_title(f"Genealogy of {root_id[:8]}", fontsize=config.title_size)
        path = os.path.join(config.output_dir,
                            f"genealogy_{root_id[:8]}.{config.format}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path


# ---------------------------------------------------------------------------
# Evolution dashboard
# ---------------------------------------------------------------------------

class EvolutionDashboard:
    """
    Comprehensive visualization of an evolution run result.
    Generates multiple plots and an HTML summary.
    """

    def __init__(self, config: PlotConfig) -> None:
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

    def plot_fitness_history(self, stats_history: List[PopulationStats]) -> Optional[str]:
        """Plot best/mean/worst fitness over generations."""
        if not stats_history:
            return None

        gens = [s.generation for s in stats_history]
        bests = [s.best_fitness for s in stats_history]
        means = [s.mean_fitness for s in stats_history]
        worsts = [s.worst_fitness for s in stats_history]

        if not HAS_MATPLOTLIB:
            chart = ASCIIPlotter.line_chart(bests, title="Fitness History (Best)")
            print(chart)
            return None

        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)

        # Fitness over time
        ax = axes[0, 0]
        ax.plot(gens, bests, "g-", label="Best", linewidth=2)
        ax.plot(gens, means, "b-", label="Mean", linewidth=1.5)
        ax.fill_between(gens,
                          [m - s.std_fitness for m, s in zip(means, stats_history)],
                          [m + s.std_fitness for m, s in zip(means, stats_history)],
                          alpha=0.2, color="blue")
        ax.set_title("Fitness History", fontsize=self.config.title_size)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=self.config.font_size)

        # Diversity over time
        ax2 = axes[0, 1]
        diversities = [s.param_diversity for s in stats_history]
        ax2.plot(gens, diversities, "r-", linewidth=1.5)
        ax2.set_title("Population Diversity", fontsize=self.config.title_size)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Mean Pairwise Distance")

        # Unique genomes
        ax3 = axes[1, 0]
        unique_fracs = [s.unique_fingerprints / max(s.size, 1) for s in stats_history]
        ax3.plot(gens, unique_fracs, "m-", linewidth=1.5)
        ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
        ax3.set_title("Unique Genomes Fraction", fontsize=self.config.title_size)
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Fraction")
        ax3.set_ylim(0, 1.1)

        # Fitness range
        ax4 = axes[1, 1]
        fitness_ranges = [s.fitness_range for s in stats_history]
        ax4.plot(gens, fitness_ranges, "c-", linewidth=1.5)
        ax4.set_title("Fitness Range", fontsize=self.config.title_size)
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Best - Worst")

        plt.suptitle("Evolution Progress Dashboard", fontsize=self.config.title_size + 2)
        plt.tight_layout()
        path = os.path.join(self.config.output_dir, f"fitness_history.{self.config.format}")
        plt.savefig(path)
        plt.close()
        return path

    def plot_hall_of_fame(self, hof: HallOfFame,
                           top_k: int = 20) -> Optional[str]:
        """Plot bar chart of Hall of Fame fitness values."""
        entries = hof.top_k[:top_k]
        if not entries:
            return None

        fitnesses = [g.fitness for g in entries if g.fitness is not None]
        labels = [g.metadata.genome_id[:6] for g in entries if g.fitness is not None]

        if not HAS_MATPLOTLIB:
            chart = ASCIIPlotter.bar_chart(fitnesses, labels,
                                            title="Hall of Fame Top Individuals")
            print(chart)
            return None

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        colors = plt.cm.viridis([i / max(len(fitnesses) - 1, 1)
                                   for i in range(len(fitnesses))])
        bars = ax.bar(range(len(fitnesses)), fitnesses, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right",
                            fontsize=self.config.font_size - 2)
        ax.set_title("Hall of Fame - Top Individuals",
                      fontsize=self.config.title_size)
        ax.set_xlabel("Genome ID")
        ax.set_ylabel("Fitness")
        plt.tight_layout()
        path = os.path.join(self.config.output_dir, f"hall_of_fame.{self.config.format}")
        plt.savefig(path)
        plt.close()
        return path

    def plot_parameter_importance(self, hof: HallOfFame,
                                   all_genomes: Optional[List[StrategyGenome]] = None) -> Optional[str]:
        """
        Estimate parameter importance by correlation with fitness.
        """
        genomes = all_genomes or list(hof)
        evaluated = [g for g in genomes if g.fitness is not None]
        if len(evaluated) < 5:
            return None

        fitnesses = [g.fitness for g in evaluated]
        all_params = evaluated[0].chromosome.to_dict().keys()
        numeric_params = [k for k in all_params
                          if isinstance(evaluated[0].chromosome[k], (int, float))]

        correlations = {}
        for param in numeric_params:
            vals = [float(g.chromosome[param]) for g in evaluated]
            if len(set(vals)) < 2:
                continue
            # Pearson correlation with fitness
            n = len(vals)
            mean_v = sum(vals) / n
            mean_f = sum(fitnesses) / n
            cov = sum((v - mean_v) * (f - mean_f) for v, f in zip(vals, fitnesses)) / n
            std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / n + 1e-10)
            std_f = math.sqrt(sum((f - mean_f) ** 2 for f in fitnesses) / n + 1e-10)
            correlations[param] = abs(cov / (std_v * std_f))

        if not correlations:
            return None

        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        param_names = [p[0] for p in sorted_params[:15]]
        corr_vals = [p[1] for p in sorted_params[:15]]

        if not HAS_MATPLOTLIB:
            chart = ASCIIPlotter.bar_chart(corr_vals, param_names,
                                            title="Parameter Importance (|Correlation| with Fitness)")
            print(chart)
            return None

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        colors = ["#2ecc71" if v > 0.5 else "#3498db" if v > 0.3 else "#95a5a6"
                  for v in corr_vals]
        ax.barh(range(len(param_names)), corr_vals, color=colors)
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(param_names, fontsize=self.config.font_size)
        ax.set_xlabel("|Pearson Correlation| with Fitness",
                       fontsize=self.config.font_size)
        ax.set_title("Parameter Importance", fontsize=self.config.title_size)
        ax.set_xlim(0, 1)
        plt.tight_layout()
        path = os.path.join(self.config.output_dir,
                            f"parameter_importance.{self.config.format}")
        plt.savefig(path)
        plt.close()
        return path

    def generate_full_report(self, result: EvolutionResult,
                              population: Optional[List[StrategyGenome]] = None) -> Dict[str, Optional[str]]:
        """
        Generate all visualization plots for an evolution result.
        Returns dict of plot_name -> file_path.
        """
        paths = {}

        # Fitness history
        if result.stats_history:
            paths["fitness_history"] = self.plot_fitness_history(result.stats_history)

        # Hall of fame
        if result.hall_of_fame:
            paths["hall_of_fame"] = self.plot_hall_of_fame(result.hall_of_fame)
            paths["parameter_importance"] = self.plot_parameter_importance(result.hall_of_fame)

        # Pareto front (if multi-objective)
        if result.pareto_front:
            pareto_viz = ParetoFrontVisualizer(self.config)
            paths["pareto_front"] = pareto_viz.plot_pareto_front_2d(
                result.pareto_front, generation=result.n_generations_run)

        # Fitness landscape (final population)
        if population:
            landscape_viz = FitnessLandscapeVisualizer(self.config)
            paths["fitness_landscape"] = landscape_viz.plot_fitness_landscape(
                population, generation=result.n_generations_run)
            paths["param_distribution"] = ParameterConvergenceVisualizer(
                self.config).plot_parameter_distribution(population,
                                                          generation=result.n_generations_run)

        # Text summary
        summary_path = os.path.join(self.config.output_dir, "evolution_summary.txt")
        with open(summary_path, "w") as f:
            f.write(result.summary())
            if result.best_genome:
                f.write("\n\nBest genome parameters:\n")
                f.write(json.dumps(result.best_genome.chromosome.to_dict(), indent=2, default=str))
        paths["summary"] = summary_path

        return paths


# ---------------------------------------------------------------------------
# Convergence curve comparison (multiple runs)
# ---------------------------------------------------------------------------

def plot_multi_run_comparison(results: List["EvolutionResult"],
                               config: Optional[PlotConfig] = None,
                               labels: Optional[List[str]] = None) -> Optional[str]:
    """
    Plot fitness convergence curves for multiple independent GA runs.
    """
    if config is None:
        config = PlotConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    if not HAS_MATPLOTLIB:
        for i, result in enumerate(results):
            if result.stats_history:
                bests = [s.best_fitness for s in result.stats_history]
                label = labels[i] if labels else f"Run {i + 1}"
                chart = ASCIIPlotter.line_chart(bests, title=f"Run {label}")
                print(chart)
        return None

    fig, axes = plt.subplots(1, 2, figsize=config.figsize, dpi=config.dpi)

    colors = plt.cm.tab10(range(min(len(results), 10)))

    for i, result in enumerate(results):
        if not result.stats_history:
            continue
        gens = [s.generation for s in result.stats_history]
        bests = [s.best_fitness for s in result.stats_history]
        means = [s.mean_fitness for s in result.stats_history]
        label = labels[i] if labels else f"Run {i + 1}"
        color = colors[i % len(colors)]

        axes[0].plot(gens, bests, linewidth=1.5, label=label, color=color)
        axes[1].plot(gens, means, linewidth=1.5, label=label, color=color)

    axes[0].set_title("Best Fitness per Generation", fontsize=config.title_size)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best Fitness")
    axes[0].legend(fontsize=config.font_size - 2)

    axes[1].set_title("Mean Fitness per Generation", fontsize=config.title_size)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Mean Fitness")
    axes[1].legend(fontsize=config.font_size - 2)

    plt.suptitle("Multi-Run Evolution Comparison", fontsize=config.title_size + 2)
    plt.tight_layout()
    path = os.path.join(config.output_dir, f"multi_run_comparison.{config.format}")
    plt.savefig(path)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Visualization self-test ===")
    from .genome import GenomeFactory
    from .population import DiversityMetrics, PopulationStats

    factory = GenomeFactory("momentum", seed=42)
    pop = factory.create_population(50)
    for i, g in enumerate(pop):
        g.fitness = random.gauss(0.5, 0.3)
        g.objectives = [random.gauss(0.5, 0.2), random.gauss(0.3, 0.15)]

    config = PlotConfig(output_dir="./test_plots", show_plots=False)

    # Test ASCII plotter
    print("--- ASCII Bar Chart ---")
    chart = ASCIIPlotter.bar_chart(
        [0.5, 0.3, 0.8, 0.2, 0.6],
        ["alpha", "beta", "gamma", "delta", "epsilon"],
        title="Test Bar Chart",
    )
    print(chart)

    print("\n--- ASCII Line Chart ---")
    series = [math.sin(i * 0.3) * 0.5 + 0.5 for i in range(50)]
    chart2 = ASCIIPlotter.line_chart(series, title="Test Sine Wave")
    print(chart2)

    print("\n--- ASCII Scatter ---")
    xs = [random.gauss(0, 1) for _ in range(30)]
    ys = [random.gauss(0, 1) for _ in range(30)]
    scatter_chart = ASCIIPlotter.scatter_2d(xs, ys, title="Test Scatter")
    print(scatter_chart)

    # Test parameter convergence tracker
    print("\n--- Parameter convergence tracking ---")
    convergence_viz = ParameterConvergenceVisualizer(config)
    for gen in range(10):
        convergence_viz.record_generation(pop)
    print(f"Recorded {convergence_viz._generation_count} generations")

    # Test genealogy tracker
    print("\n--- Genealogy tracking ---")
    tracker = GenealogyTracker()
    tracker.record_population(pop)
    best_id = max(pop, key=lambda g: g.fitness).metadata.genome_id
    lineage = tracker.get_lineage(best_id)
    print(f"Lineage depth for best: {len(lineage)}")

    # Test fitness landscape
    print("\n--- Fitness landscape (ASCII fallback) ---")
    landscape_viz = FitnessLandscapeVisualizer(config)
    landscape_viz.plot_fitness_landscape(pop, generation=0)

    # Test Pareto front
    print("\n--- Pareto front (ASCII fallback) ---")
    pareto_viz = ParetoFrontVisualizer(config)
    pareto_viz.add_frame(pop[:10])
    pareto_viz.add_frame(pop[10:20])
    pareto_viz.plot_pareto_front_2d(pop[:10],
                                     obj_names=["Sharpe", "Calmar"],
                                     generation=5)

    print("\n--- Dashboard test ---")
    # Create synthetic evolution result
    stats_history = []
    for gen in range(20):
        stats_history.append(PopulationStats(
            generation=gen, size=50,
            best_fitness=0.3 + gen * 0.02 + random.gauss(0, 0.01),
            mean_fitness=0.1 + gen * 0.015,
            std_fitness=0.15,
            worst_fitness=-0.1,
            median_fitness=0.2,
            fitness_range=0.5,
            param_diversity=0.4 - gen * 0.01,
            unique_fingerprints=45,
            elite_count=5,
        ))

    hof = HallOfFame(50)
    hof.update(pop[:20])

    from .evolution import EvolutionResult
    synthetic_result = EvolutionResult(
        best_genome=pop[0],
        hall_of_fame=hof,
        pareto_front=pop[:10],
        stats_history=stats_history,
        n_generations_run=20,
        n_evaluations=1000,
        n_restarts=1,
        converged=False,
        convergence_reason="max_generations",
        elapsed_seconds=42.0,
        final_population=pop,
    )

    dashboard = EvolutionDashboard(config)
    paths = dashboard.generate_full_report(synthetic_result, population=pop)
    print(f"Generated {len(paths)} plots: {list(paths.keys())}")
    print(f"Summary saved to: {paths.get('summary')}")

    print("\nAll visualization tests passed.")
