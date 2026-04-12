"""
visualization.py — Visualization tools for TensorNet.

Provides plotting functions for MPS/TT structure, compression errors,
anomaly scores, and quantum kernel matrices.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Optional network visualization
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from tensor_net.mps import (
    MatrixProductState,
    mps_bond_entropies,
    mps_entanglement_spectrum,
)
from tensor_net.tensor_train import TensorTrain


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

TENSORNET_STYLE = {
    "figure.facecolor": "#0a0a0f",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#f0f6fc",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.5,
    "lines.linewidth": 1.5,
}

REGIME_COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"]
CRISIS_COLOR = "#f85149"
NORMAL_COLOR = "#3fb950"
ANOMALY_COLOR = "#ff7b72"


def apply_tensornet_style():
    """Apply SRFM-lab dark theme to matplotlib."""
    plt.rcParams.update(TENSORNET_STYLE)


def save_or_show(fig: plt.Figure, save_path: Optional[str] = None, dpi: int = 150):
    """Save figure to path or show interactively."""
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bond dimension plots
# ---------------------------------------------------------------------------

def plot_bond_dimensions(
    mps: MatrixProductState,
    title: str = "MPS Bond Dimensions",
    save_path: Optional[str] = None,
    show_max_bond: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Bar chart of bond dimensions across sites.

    Parameters
    ----------
    mps : MatrixProductState
    title : plot title
    save_path : if provided, save figure to this path
    show_max_bond : if provided, draw a horizontal line at this value
    """
    apply_tensornet_style()
    fig, ax = plt.subplots(figsize=figsize, facecolor=TENSORNET_STYLE["figure.facecolor"])
    ax.set_facecolor(TENSORNET_STYLE["axes.facecolor"])

    bond_dims = mps.bond_dims
    x = np.arange(len(bond_dims))
    bars = ax.bar(x, bond_dims, color="#58a6ff", alpha=0.85, edgecolor="#1f6feb", linewidth=0.7)

    # Annotate bars with values
    for bar, val in zip(bars, bond_dims):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(val),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#c9d1d9",
        )

    if show_max_bond is not None:
        ax.axhline(show_max_bond, color=CRISIS_COLOR, linestyle="--",
                   linewidth=1.2, label=f"max_bond={show_max_bond}")
        ax.legend(fontsize=9)

    ax.set_xlabel("Bond index (left=0)", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("Bond dimension χ", fontsize=10, color="#c9d1d9")
    ax.set_title(title, fontsize=12, color="#f0f6fc", pad=12)
    ax.set_xticks(x)
    ax.tick_params(colors="#8b949e")
    ax.grid(axis="y", alpha=0.3, color="#21262d")

    # Info text
    info = (
        f"n_sites={mps.n_sites}  "
        f"max_bond={mps.max_bond}  "
        f"n_params={mps.num_params():,}  "
        f"phys_dims={mps.phys_dims[:4]}{'...' if len(mps.phys_dims) > 4 else ''}"
    )
    fig.text(0.5, 0.01, info, ha="center", fontsize=8, color="#8b949e")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Entanglement spectrum
# ---------------------------------------------------------------------------

def plot_entanglement_spectrum(
    mps: MatrixProductState,
    bonds: Optional[List[int]] = None,
    title: str = "Entanglement Spectrum",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot eigenvalues of the reduced density matrix (entanglement spectrum) at each bond.

    For each bond, shows the sorted singular values on a log scale.
    """
    apply_tensornet_style()
    n = mps.n_sites

    if bonds is None:
        # Sample up to 6 bonds for readability
        if n <= 6:
            bonds = list(range(n - 1))
        else:
            bonds = list(range(0, n - 1, (n - 1) // 5 + 1))[:6]

    n_bonds = len(bonds)
    fig, axes = plt.subplots(1, n_bonds, figsize=figsize,
                              facecolor=TENSORNET_STYLE["figure.facecolor"])
    if n_bonds == 1:
        axes = [axes]

    for ax, bond in zip(axes, bonds):
        ax.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
        try:
            spectrum = np.array(mps_entanglement_spectrum(mps, bond))
        except Exception:
            spectrum = np.array([1.0])

        spectrum_sorted = np.sort(spectrum)[::-1]
        x = np.arange(1, len(spectrum_sorted) + 1)

        ax.semilogy(x, spectrum_sorted, "o-", color="#58a6ff",
                    markersize=5, linewidth=1.5)
        ax.fill_between(x, spectrum_sorted, alpha=0.2, color="#58a6ff")

        # Compute entropy
        s2 = spectrum_sorted ** 2
        s2 /= s2.sum() + 1e-12
        entropy = -np.sum(s2 * np.log(s2 + 1e-12))

        ax.set_title(f"Bond {bond}\nS={entropy:.3f}", fontsize=9, color="#f0f6fc")
        ax.set_xlabel("Index", fontsize=8)
        ax.set_ylabel("Singular value", fontsize=8)
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.grid(alpha=0.3, color="#21262d")

    fig.suptitle(title, fontsize=13, color="#f0f6fc", y=1.02)
    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Compression error vs ratio
# ---------------------------------------------------------------------------

def plot_compression_error_vs_ratio(
    results: Dict,
    title: str = "Compression Error vs Ratio",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 5),
    target_error: Optional[float] = None,
    show_pareto: bool = True,
) -> plt.Figure:
    """
    Plot the compression tradeoff curve: error vs compression ratio.

    Parameters
    ----------
    results : dict with keys 'bond_dims', 'errors', 'compression_ratios'
    target_error : if given, mark the minimum bond dim achieving this error
    """
    apply_tensornet_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    facecolor=TENSORNET_STYLE["figure.facecolor"])

    bond_dims = results.get("bond_dims", [])
    errors = results.get("errors", [])
    ratios = results.get("compression_ratios", [])

    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(bond_dims) - 1, 1)) for i in range(len(bond_dims))]

    # Plot 1: Error vs bond dimension
    ax1.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    ax1.plot(bond_dims, errors, "o-", color="#58a6ff", linewidth=2, markersize=7)
    for bd, err, c in zip(bond_dims, errors, colors):
        ax1.scatter([bd], [err], color=c, s=60, zorder=5)
        ax1.annotate(f"D={bd}\n{err:.3f}", (bd, err),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=7, color="#c9d1d9")

    if target_error is not None:
        ax1.axhline(target_error, color=CRISIS_COLOR, linestyle="--",
                    linewidth=1.2, label=f"Target error={target_error:.3f}")
        ax1.legend(fontsize=8)

    ax1.set_xlabel("Bond dimension D", fontsize=10)
    ax1.set_ylabel("Relative Frobenius error", fontsize=10)
    ax1.set_title("Error vs Bond Dimension", fontsize=11, color="#f0f6fc")
    ax1.tick_params(colors="#8b949e")
    ax1.grid(alpha=0.3, color="#21262d")

    # Plot 2: Error vs compression ratio (Pareto curve)
    ax2.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    ax2.plot(ratios, errors, "o-", color="#3fb950", linewidth=2, markersize=7)
    for r, err, bd, c in zip(ratios, errors, bond_dims, colors):
        ax2.scatter([r], [err], color=c, s=60, zorder=5)
        ax2.annotate(f"D={bd}", (r, err),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=7, color="#c9d1d9")

    ax2.set_xlabel("Compression ratio (×)", fontsize=10)
    ax2.set_ylabel("Relative Frobenius error", fontsize=10)
    ax2.set_title("Error vs Compression Ratio", fontsize=11, color="#f0f6fc")
    ax2.tick_params(colors="#8b949e")
    ax2.grid(alpha=0.3, color="#21262d")

    # Shade the "good" region
    if len(errors) > 0 and len(ratios) > 0:
        min_err = min(errors)
        ax2.axhline(min_err * 1.05, color=ANOMALY_COLOR, linestyle=":",
                    linewidth=1, alpha=0.6, label="+5% of min error")
        ax2.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, color="#f0f6fc", y=1.01)
    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Anomaly scores
# ---------------------------------------------------------------------------

def plot_anomaly_scores(
    scores: jnp.ndarray,
    returns: jnp.ndarray,
    timestamps: Optional[jnp.ndarray] = None,
    crisis_bars: Optional[List[int]] = None,
    regime_boundaries: Optional[List[int]] = None,
    title: str = "MPS Anomaly Detection",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 7),
    threshold: float = 2.5,
) -> plt.Figure:
    """
    Overlay anomaly scores on a return series with crisis markers.

    Parameters
    ----------
    scores : anomaly z-score series
    returns : return series of shape (T,) or (T, N)
    timestamps : time indices for scores
    crisis_bars : list of bar indices marking known crisis events
    regime_boundaries : list of bar indices marking regime changes
    threshold : z-score threshold for anomaly marking
    """
    apply_tensornet_style()
    scores = np.array(scores)
    returns = np.array(returns)

    if returns.ndim == 2:
        # Use portfolio return (equal weight)
        portfolio_returns = returns.mean(axis=1)
    else:
        portfolio_returns = returns

    if timestamps is None:
        timestamps = np.arange(len(scores))
    timestamps = np.array(timestamps)

    fig = plt.figure(figsize=figsize, facecolor=TENSORNET_STYLE["figure.facecolor"])
    gs = GridSpec(3, 1, height_ratios=[2, 1.5, 0.8], hspace=0.08, figure=fig)

    # Panel 1: Returns with crisis shading
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    t_ret = np.arange(len(portfolio_returns))
    ax1.plot(t_ret, portfolio_returns, color="#58a6ff", linewidth=0.8, alpha=0.9)
    ax1.fill_between(t_ret, portfolio_returns, 0,
                     where=portfolio_returns < 0, alpha=0.3, color=CRISIS_COLOR)
    ax1.fill_between(t_ret, portfolio_returns, 0,
                     where=portfolio_returns >= 0, alpha=0.2, color=NORMAL_COLOR)

    # Shade regime regions
    if regime_boundaries is not None:
        colors_reg = ["#1c2128", "#161b22", "#0d1117"]
        starts = [0] + list(regime_boundaries[:-1])
        ends = list(regime_boundaries)
        for s, e, c in zip(starts, ends, colors_reg):
            ax1.axvspan(s, e, alpha=0.2, color=REGIME_COLORS[starts.index(s) % len(REGIME_COLORS)])

    ax1.axhline(0, color="#30363d", linewidth=0.5)
    ax1.set_ylabel("Portfolio Return", fontsize=9)
    ax1.tick_params(colors="#8b949e", labelbottom=False)
    ax1.grid(alpha=0.2, color="#21262d")
    ax1.set_title(title, fontsize=13, color="#f0f6fc", pad=10)

    # Panel 2: Anomaly scores
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor(TENSORNET_STYLE["axes.facecolor"])

    # Color bars by anomaly level
    colors_score = np.where(scores > threshold, ANOMALY_COLOR,
                   np.where(scores > 1.5, "#ffa657", "#58a6ff"))
    ax2.bar(timestamps, scores, color=colors_score, width=1.0, alpha=0.85)
    ax2.axhline(threshold, color=CRISIS_COLOR, linestyle="--",
                linewidth=1.2, label=f"Threshold z={threshold}")
    ax2.axhline(0, color="#30363d", linewidth=0.5)
    ax2.set_ylabel("Anomaly z-score", fontsize=9)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.tick_params(colors="#8b949e", labelbottom=False)
    ax2.grid(alpha=0.2, color="#21262d")

    # Mark crisis bars with vertical lines
    if crisis_bars is not None:
        for cb in crisis_bars:
            ax1.axvline(cb, color=CRISIS_COLOR, linestyle=":", linewidth=1.5, alpha=0.8)
            ax2.axvline(cb, color=CRISIS_COLOR, linestyle=":", linewidth=1.5, alpha=0.8)
            ax2.annotate("CRISIS", (cb, ax2.get_ylim()[1] * 0.9),
                         fontsize=7, color=CRISIS_COLOR,
                         rotation=90, va="top", ha="right")

    # Panel 3: Anomaly binary (threshold crossing)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    is_anomaly = (scores > threshold).astype(float)
    ax3.fill_between(timestamps, is_anomaly, alpha=0.7, color=ANOMALY_COLOR, step="mid")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Normal", "Alert"], fontsize=8)
    ax3.set_xlabel("Bar", fontsize=9)
    ax3.tick_params(colors="#8b949e")
    ax3.grid(alpha=0.2, color="#21262d")

    # Add legend patches
    patches = [
        mpatches.Patch(color=NORMAL_COLOR, alpha=0.7, label="Normal"),
        mpatches.Patch(color=ANOMALY_COLOR, alpha=0.7, label=f"Alert (z>{threshold:.1f})"),
    ]
    if crisis_bars:
        patches.append(mpatches.Patch(color=CRISIS_COLOR, alpha=0.7, label="Known crisis"))
    ax3.legend(handles=patches, fontsize=8, loc="upper right")

    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# TT structure diagram
# ---------------------------------------------------------------------------

def plot_tt_structure(
    tt: TensorTrain,
    title: str = "Tensor Train Structure",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 3),
) -> plt.Figure:
    """
    Diagram of tensor train with bond dimensions labeled.
    Shows each core as a circle, bonds as labeled edges.
    """
    apply_tensornet_style()
    fig, ax = plt.subplots(figsize=figsize,
                           facecolor=TENSORNET_STYLE["figure.facecolor"])
    ax.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    ax.set_aspect("equal")

    n = tt.ndim
    ranks = tt.ranks

    # Draw cores as circles
    core_y = 0.5
    core_radius = 0.25
    spacing = 2.0
    x_positions = [i * spacing for i in range(n)]

    # Draw bond edges first
    for i in range(n):
        x = x_positions[i]

        # Left bond
        if i == 0:
            x_l = x - 0.8
            ax.plot([x_l, x - core_radius], [core_y, core_y],
                    color="#58a6ff", linewidth=2)
            ax.text(x_l - 0.05, core_y + 0.2, "1", fontsize=8, color="#8b949e", ha="center")
        else:
            x_prev = x_positions[i - 1]
            mid_x = (x_prev + x) / 2
            ax.plot([x_prev + core_radius, x - core_radius], [core_y, core_y],
                    color="#58a6ff", linewidth=2.5)
            bond_label = str(ranks[i])
            ax.text(mid_x, core_y + 0.25, f"χ={bond_label}", fontsize=8,
                    color="#d2a8ff", ha="center")

        # Right bond for last core
        if i == n - 1:
            x_r = x + 0.8
            ax.plot([x + core_radius, x_r], [core_y, core_y],
                    color="#58a6ff", linewidth=2)
            ax.text(x_r + 0.05, core_y + 0.2, "1", fontsize=8, color="#8b949e", ha="center")

        # Physical index (downward line)
        ax.plot([x, x], [core_y - core_radius, core_y - 0.7],
                color="#3fb950", linewidth=2)
        ax.text(x, core_y - 0.9, f"n={tt.shape[i]}", fontsize=7,
                color="#3fb950", ha="center")

    # Draw cores
    for i, x in enumerate(x_positions):
        core = tt.cores[i]
        circle = plt.Circle((x, core_y), core_radius,
                             color="#1f6feb", linewidth=2, fill=True,
                             facecolor="#0d419d", zorder=10)
        ax.add_patch(circle)
        # Core size label
        r_l, n_k, r_r = core.shape
        ax.text(x, core_y, f"G{i+1}", fontsize=8, color="white",
                ha="center", va="center", fontweight="bold", zorder=11)

    # Limits and labels
    ax.set_xlim(-1.5, (n - 1) * spacing + 1.5)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    ax.set_title(
        f"{title}\n"
        f"shape={tt.shape}  ranks={tt.ranks}  n_params={tt.n_params:,}",
        fontsize=11, color="#f0f6fc", pad=10
    )

    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Animated MPS evolution
# ---------------------------------------------------------------------------

def animate_mps_evolution(
    mps_list: List[MatrixProductState],
    fps: int = 5,
    save_path: Optional[str] = None,
    title: str = "MPS Evolution",
    figsize: Tuple[float, float] = (10, 4),
) -> animation.FuncAnimation:
    """
    Animated evolution of MPS bond dimensions over time.

    Parameters
    ----------
    mps_list : list of MPS at different time steps
    fps : frames per second
    save_path : if provided, save as GIF or MP4
    """
    apply_tensornet_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    facecolor=TENSORNET_STYLE["figure.facecolor"])

    def get_bond_dims(mps):
        return np.array(mps.bond_dims)

    def get_entropies(mps):
        try:
            return np.array(mps_bond_entropies(mps))
        except Exception:
            return np.zeros(mps.n_sites - 1)

    max_bond = max(m.max_bond for m in mps_list)
    n_sites = mps_list[0].n_sites

    def init():
        ax1.clear()
        ax2.clear()
        for ax in [ax1, ax2]:
            ax.set_facecolor(TENSORNET_STYLE["axes.facecolor"])

    def update(frame):
        mps = mps_list[frame]
        bonds = get_bond_dims(mps)
        entropies = get_entropies(mps)

        ax1.clear()
        ax1.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
        x1 = np.arange(len(bonds))
        ax1.bar(x1, bonds, color="#58a6ff", alpha=0.85)
        ax1.set_ylim(0, max_bond * 1.15)
        ax1.set_xlabel("Bond index", fontsize=9)
        ax1.set_ylabel("Bond dimension χ", fontsize=9)
        ax1.set_title(f"{title} — Step {frame}", fontsize=10, color="#f0f6fc")
        ax1.tick_params(colors="#8b949e")
        ax1.grid(axis="y", alpha=0.3)

        ax2.clear()
        ax2.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
        x2 = np.arange(len(entropies))
        ax2.plot(x2, entropies, "o-", color="#3fb950", linewidth=1.5, markersize=5)
        ax2.fill_between(x2, entropies, alpha=0.3, color="#3fb950")
        ax2.set_xlabel("Bond index", fontsize=9)
        ax2.set_ylabel("Entanglement entropy S", fontsize=9)
        ax2.set_title("Bond Entropy Profile", fontsize=10, color="#f0f6fc")
        ax2.tick_params(colors="#8b949e")
        ax2.grid(alpha=0.3)

        return ax1, ax2

    anim = animation.FuncAnimation(
        fig, update, frames=len(mps_list), init_func=init,
        interval=1000 // fps, blit=False
    )

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
    else:
        plt.show()

    return anim


# ---------------------------------------------------------------------------
# Quantum kernel matrix
# ---------------------------------------------------------------------------

def plot_quantum_kernel_matrix(
    K: jnp.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    title: str = "Quantum Kernel Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 7),
    show_dendrogram: bool = False,
) -> plt.Figure:
    """
    Kernel matrix heatmap with regime labels.

    Parameters
    ----------
    K : kernel matrix of shape (n, n)
    labels : array of shape (n,) with integer regime labels
    label_names : optional list of regime name strings
    title : plot title
    show_dendrogram : if True, use seaborn clustermap instead of heatmap
    """
    apply_tensornet_style()
    K_np = np.array(K)
    n = K_np.shape[0]

    if show_dendrogram and labels is not None:
        # Use clustermap
        fig = plt.figure(figsize=figsize, facecolor=TENSORNET_STYLE["figure.facecolor"])
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=figsize,
                               facecolor=TENSORNET_STYLE["figure.facecolor"])

    ax.set_facecolor(TENSORNET_STYLE["axes.facecolor"])

    # Plot heatmap
    im = ax.imshow(K_np, cmap="viridis", aspect="auto",
                   vmin=0, vmax=K_np.max())
    plt.colorbar(im, ax=ax, label="Kernel value")

    # Overlay regime boundaries if labels provided
    if labels is not None:
        labels_np = np.array(labels)
        n_regimes = labels_np.max() + 1

        # Sort by label to show block structure
        sort_idx = np.argsort(labels_np)
        K_sorted = K_np[sort_idx, :][:, sort_idx]
        im.set_data(K_sorted)

        # Draw regime block boundaries
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        boundaries = np.cumsum(counts)[:-1] - 0.5

        for b in boundaries:
            ax.axhline(b, color="white", linewidth=1.0, alpha=0.6)
            ax.axvline(b, color="white", linewidth=1.0, alpha=0.6)

        # Label regime blocks
        pos = 0
        for label, count in zip(unique_labels, counts):
            mid = pos + count / 2
            name = label_names[label] if label_names else f"R{label}"
            ax.text(mid, -2, name, ha="center", va="bottom",
                    fontsize=8, color=REGIME_COLORS[label % len(REGIME_COLORS)])
            ax.text(-2, mid, name, ha="right", va="center",
                    fontsize=8, color=REGIME_COLORS[label % len(REGIME_COLORS)])
            pos += count

    ax.set_title(title, fontsize=12, color="#f0f6fc", pad=12)
    ax.set_xlabel("Sample index", fontsize=10)
    ax.set_ylabel("Sample index", fontsize=10)
    ax.tick_params(colors="#8b949e")

    # Summary stats
    off_diag = K_np[np.eye(n) == 0]
    stats_text = (
        f"n={n}  min={K_np.min():.3f}  max={K_np.max():.3f}  "
        f"mean_off_diag={off_diag.mean():.3f}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=8, color="#8b949e")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    metrics_dict: Dict,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Plot training and validation loss curves, gradient norms, and bond entropy evolution.

    Parameters
    ----------
    metrics_dict : dict or TrainingMetrics object with loss/grad/entropy data
    """
    apply_tensornet_style()

    # Extract data
    if hasattr(metrics_dict, "train_losses"):
        train_losses = metrics_dict.train_losses
        val_losses = metrics_dict.val_losses
        grad_norms = metrics_dict.gradient_norms
        entropies = metrics_dict.bond_entropies
    else:
        train_losses = metrics_dict.get("train_losses", [])
        val_losses = metrics_dict.get("val_losses", [])
        grad_norms = metrics_dict.get("gradient_norms", [])
        entropies = metrics_dict.get("bond_entropies", [])

    fig, axes = plt.subplots(2, 2, figsize=figsize,
                              facecolor=TENSORNET_STYLE["figure.facecolor"])

    # Loss curves
    ax = axes[0, 0]
    ax.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    if train_losses:
        ax.semilogy(train_losses, color="#58a6ff", linewidth=1.5, label="Train")
    if val_losses:
        val_x = np.linspace(0, len(train_losses) - 1, len(val_losses))
        ax.semilogy(val_x, val_losses, color="#3fb950", linewidth=1.5,
                    linestyle="--", label="Val")
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("Training Loss", fontsize=10, color="#f0f6fc")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(colors="#8b949e")

    # Gradient norms
    ax = axes[0, 1]
    ax.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    if grad_norms:
        ax.semilogy(grad_norms, color="#ffa657", linewidth=1.2)
        ax.axhline(1.0, color="#30363d", linestyle="--", linewidth=0.8, label="norm=1")
        ax.legend(fontsize=8)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Gradient norm", fontsize=9)
    ax.set_title("Gradient Norms", fontsize=10, color="#f0f6fc")
    ax.grid(alpha=0.3)
    ax.tick_params(colors="#8b949e")

    # Entropy evolution
    ax = axes[1, 0]
    ax.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    if entropies:
        entropy_arr = np.array([np.array(e) for e in entropies])
        im = ax.imshow(entropy_arr.T, aspect="auto", cmap="plasma",
                       origin="lower", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Entropy S")
        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel("Bond index", fontsize=9)
        ax.set_title("Bond Entropy Evolution", fontsize=10, color="#f0f6fc")
    ax.tick_params(colors="#8b949e")

    # Loss improvement
    ax = axes[1, 1]
    ax.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    if len(train_losses) > 2:
        improvements = -np.diff(np.log(np.array(train_losses) + 1e-12))
        ax.plot(improvements, color="#d2a8ff", linewidth=1.0)
        ax.axhline(0, color="#30363d", linewidth=0.8, linestyle="--")
        ax.fill_between(np.arange(len(improvements)),
                        improvements, 0,
                        where=improvements > 0, alpha=0.3, color=NORMAL_COLOR)
        ax.fill_between(np.arange(len(improvements)),
                        improvements, 0,
                        where=improvements < 0, alpha=0.3, color=CRISIS_COLOR)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Loss improvement (log)", fontsize=9)
    ax.set_title("Per-Step Improvement", fontsize=10, color="#f0f6fc")
    ax.grid(alpha=0.3)
    ax.tick_params(colors="#8b949e")

    fig.suptitle(title, fontsize=13, color="#f0f6fc", y=1.01)
    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Compression summary dashboard
# ---------------------------------------------------------------------------

def plot_compression_dashboard(
    original: np.ndarray,
    reconstructed: np.ndarray,
    errors_by_bond: Dict,
    mps: MatrixProductState,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 9),
    title: str = "MPS Compression Dashboard",
) -> plt.Figure:
    """
    Full compression dashboard: original vs reconstructed + error analysis.
    """
    apply_tensornet_style()
    fig = plt.figure(figsize=figsize, facecolor=TENSORNET_STYLE["figure.facecolor"])
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Original matrix
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    im1 = ax1.imshow(original, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    ax1.set_title("Original", fontsize=10, color="#f0f6fc")
    ax1.tick_params(colors="#8b949e")

    # Reconstructed matrix
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    im2 = ax2.imshow(reconstructed, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    err = np.linalg.norm(original - reconstructed) / (np.linalg.norm(original) + 1e-12)
    ax2.set_title(f"Reconstructed (err={err:.4f})", fontsize=10, color="#f0f6fc")
    ax2.tick_params(colors="#8b949e")

    # Residual
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    residual = np.abs(original - reconstructed)
    im3 = ax3.imshow(residual, cmap="hot", aspect="auto")
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    ax3.set_title("|Residual|", fontsize=10, color="#f0f6fc")
    ax3.tick_params(colors="#8b949e")

    # Bond dims
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    bond_dims = mps.bond_dims
    ax4.bar(range(len(bond_dims)), bond_dims, color="#58a6ff", alpha=0.85)
    ax4.set_title("Bond Dimensions", fontsize=10, color="#f0f6fc")
    ax4.set_xlabel("Bond index", fontsize=9)
    ax4.set_ylabel("χ", fontsize=9)
    ax4.tick_params(colors="#8b949e")
    ax4.grid(axis="y", alpha=0.3)

    # Error vs bond dim
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    if errors_by_bond:
        bds = errors_by_bond.get("bond_dims", [])
        errs = errors_by_bond.get("errors", [])
        ax5.plot(bds, errs, "o-", color="#3fb950", linewidth=2)
        ax5.set_xlabel("Bond dim D", fontsize=9)
        ax5.set_ylabel("Relative error", fontsize=9)
        ax5.set_title("Error vs Bond Dim", fontsize=10, color="#f0f6fc")
        ax5.tick_params(colors="#8b949e")
        ax5.grid(alpha=0.3)

    # Compression stats
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(TENSORNET_STYLE["axes.facecolor"])
    ax6.axis("off")
    n_dense = original.size
    n_params = mps.num_params()
    ratio = n_dense / max(n_params, 1)
    stats = [
        f"Original size: {n_dense:,}",
        f"Compressed:    {n_params:,}",
        f"Ratio:         {ratio:.1f}×",
        f"Error:         {err:.4f}",
        f"Max bond χ:    {mps.max_bond}",
        f"n_sites:       {mps.n_sites}",
        f"phys_dims:     {mps.phys_dims[:3]}...",
    ]
    for j, s in enumerate(stats):
        ax6.text(0.05, 0.9 - j * 0.12, s, transform=ax6.transAxes,
                 fontsize=10, color="#c9d1d9", fontfamily="monospace")
    ax6.set_title("Statistics", fontsize=10, color="#f0f6fc")

    fig.suptitle(title, fontsize=13, color="#f0f6fc", y=1.01)
    save_or_show(fig, save_path)
    return fig
