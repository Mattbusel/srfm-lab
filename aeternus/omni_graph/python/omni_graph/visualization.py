"""
visualization.py
================
Graph visualization utilities for Omni-Graph financial networks.

Implements:
  - NetworkX / Plotly interactive graph rendering
  - Edge weight heatmaps (correlation matrix plots)
  - Temporal graph animations
  - Community detection visualization
  - Regime-colored graphs
  - Degree distribution and spectral plots
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

# Optional visualization dependencies
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.animation as animation
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not available; some visualizations disabled.")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

REGIME_COLORS = {
    0: "#2196F3",  # neutral — blue
    1: "#4CAF50",  # bull — green
    2: "#F44336",  # bear — red
    3: "#FF9800",  # crisis — orange
}

REGIME_NAMES = {
    0: "Neutral",
    1: "Bull",
    2: "Bear",
    3: "Crisis",
}

COMMUNITY_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62",
]


# ---------------------------------------------------------------------------
# Graph drawing utilities
# ---------------------------------------------------------------------------

def draw_financial_graph(
    edge_index: Tensor,
    num_nodes: int,
    edge_weights: Optional[Tensor] = None,
    node_labels: Optional[List[str]] = None,
    node_colors: Optional[List] = None,
    node_sizes: Optional[List[float]] = None,
    title: str = "Financial Graph",
    figsize: Tuple[int, int] = (12, 10),
    layout: str = "spring",
    ax: Optional[Any] = None,
    save_path: Optional[str] = None,
    return_fig: bool = False,
) -> Optional[Any]:
    """
    Draw a financial graph using networkx + matplotlib.

    Parameters
    ----------
    edge_index   : (2, E)
    num_nodes    : N
    edge_weights : (E,) optional — controls edge width/color
    node_labels  : list of node label strings
    node_colors  : list of colors per node
    node_sizes   : list of sizes per node
    title        : plot title
    figsize      : figure size
    layout       : networkx layout — spring | circular | kamada_kawai | spectral
    ax           : existing axes (optional)
    save_path    : save figure to file
    return_fig   : return figure object

    Returns
    -------
    fig if return_fig else None
    """
    if not HAS_MPL:
        warnings.warn("matplotlib required for draw_financial_graph")
        return None
    if not HAS_NX:
        warnings.warn("networkx required for draw_financial_graph")
        return None

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    E = edge_index.shape[1]
    for k in range(E):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        w = float(edge_weights[k]) if edge_weights is not None and k < len(edge_weights) else 1.0
        if i != j:
            G.add_edge(i, j, weight=abs(w), raw_weight=w)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=2.0 / math.sqrt(max(num_nodes, 1)))
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, weight="weight") if G.number_of_edges() > 0 else nx.circular_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G, weight="weight") if G.number_of_edges() > 0 else nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Edge properties
    edge_widths = []
    edge_colors = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        raw_w = data.get("raw_weight", w)
        edge_widths.append(1.0 + 3.0 * w)
        edge_colors.append("#2196F3" if raw_w >= 0 else "#F44336")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.6,
    )

    # Node properties
    nc = node_colors or ["#90CAF9"] * num_nodes
    ns = node_sizes or [300.0] * num_nodes

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=ns, alpha=0.9)

    if node_labels:
        label_dict = {i: node_labels[i] for i in range(min(len(node_labels), num_nodes))}
        nx.draw_networkx_labels(G, pos, labels=label_dict, ax=ax, font_size=8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if return_fig:
        return fig
    return None


def draw_regime_colored_graph(
    edge_index: Tensor,
    num_nodes: int,
    node_regimes: List[int],
    edge_weights: Optional[Tensor] = None,
    node_labels: Optional[List[str]] = None,
    title: str = "Regime-Colored Financial Graph",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Draw graph with nodes colored by market regime.
    """
    node_colors = [REGIME_COLORS.get(r, "#9E9E9E") for r in node_regimes]
    return draw_financial_graph(
        edge_index, num_nodes,
        edge_weights=edge_weights,
        node_labels=node_labels,
        node_colors=node_colors,
        title=title,
        figsize=figsize,
        save_path=save_path,
        return_fig=True,
    )


# ---------------------------------------------------------------------------
# Correlation heatmaps
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (10, 8),
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate: bool = False,
    save_path: Optional[str] = None,
    return_fig: bool = False,
) -> Optional[Any]:
    """
    Plot correlation matrix as a heatmap.

    Parameters
    ----------
    corr_matrix  : (N, N)
    asset_names  : list of asset name strings
    title        : plot title
    cmap         : matplotlib colormap
    annotate     : show correlation values in cells (small N only)
    """
    if not HAS_MPL:
        warnings.warn("matplotlib required for plot_correlation_heatmap")
        return None

    n = corr_matrix.shape[0]
    asset_names = asset_names or [f"A{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(asset_names, rotation=90, fontsize=max(4, 8 - n // 10))
    ax.set_yticklabels(asset_names, fontsize=max(4, 8 - n // 10))
    ax.set_title(title, fontsize=14, fontweight="bold")

    if annotate and n <= 20:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_edge_weight_distribution(
    edge_weights: Tensor,
    title: str = "Edge Weight Distribution",
    bins: int = 50,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """Plot histogram of edge weights."""
    if not HAS_MPL:
        return None

    w = edge_weights.detach().cpu().numpy().flatten()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(w, bins=bins, color="#2196F3", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Edge Weight")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution")

    axes[1].hist(np.abs(w), bins=bins, color="#FF9800", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("|Edge Weight|")
    axes[1].set_title("|Weight| Distribution")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Temporal graph animation
# ---------------------------------------------------------------------------

def animate_temporal_graph(
    edge_index_sequence: List[Tensor],
    num_nodes: int,
    edge_weight_sequence: Optional[List[Tensor]] = None,
    node_labels: Optional[List[str]] = None,
    node_regime_sequence: Optional[List[List[int]]] = None,
    fps: int = 2,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    title_prefix: str = "t=",
) -> Optional[Any]:
    """
    Create an animation of a temporal graph sequence.

    Parameters
    ----------
    edge_index_sequence    : list of (2, E_t) edge index tensors
    num_nodes              : N (fixed across all snapshots)
    edge_weight_sequence   : optional list of (E_t,) edge weights
    node_labels            : optional list of node label strings
    node_regime_sequence   : optional list of regime labels per node per snapshot
    fps                    : frames per second
    save_path              : path to save animation (.gif or .mp4)

    Returns
    -------
    FuncAnimation object or None
    """
    if not HAS_MPL or not HAS_NX:
        warnings.warn("matplotlib and networkx required for animate_temporal_graph")
        return None

    T = len(edge_index_sequence)

    # Compute a fixed layout from the first snapshot
    G0 = nx.Graph()
    G0.add_nodes_from(range(num_nodes))
    if edge_index_sequence[0].shape[1] > 0:
        for k in range(edge_index_sequence[0].shape[1]):
            i, j = int(edge_index_sequence[0][0, k]), int(edge_index_sequence[0][1, k])
            if i != j:
                G0.add_edge(i, j)
    pos = nx.spring_layout(G0, seed=42, k=2.0 / math.sqrt(max(num_nodes, 1)))

    fig, ax = plt.subplots(figsize=figsize)

    def _draw_frame(t: int) -> None:
        ax.clear()
        ei = edge_index_sequence[t]
        ew = edge_weight_sequence[t] if edge_weight_sequence else None
        regimes = node_regime_sequence[t] if node_regime_sequence else None

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            w = float(ew[k]) if ew is not None and k < len(ew) else 1.0
            if i != j:
                G.add_edge(i, j, weight=abs(w), raw_w=w)

        edge_ws = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        edge_cs = ["#2196F3" if G[u][v].get("raw_w", 1.0) >= 0 else "#F44336"
                   for u, v in G.edges()]

        nc = [REGIME_COLORS.get(r, "#90CAF9") for r in regimes] if regimes else ["#90CAF9"] * num_nodes

        nx.draw_networkx_edges(G, pos, ax=ax, width=[1 + 3 * w for w in edge_ws],
                               edge_color=edge_cs, alpha=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=300, alpha=0.9)
        if node_labels:
            labels = {i: node_labels[i] for i in range(min(len(node_labels), num_nodes))}
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

        n_edges = ei.shape[1]
        ax.set_title(f"{title_prefix}{t}  |  E={n_edges}", fontsize=12)
        ax.axis("off")

    anim = animation.FuncAnimation(fig, _draw_frame, frames=T, interval=1000 // fps)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, fps=fps, extra_args=["-vcodec", "libx264"])
        plt.close(fig)

    return anim


# ---------------------------------------------------------------------------
# Community visualization
# ---------------------------------------------------------------------------

def draw_community_graph(
    edge_index: Tensor,
    num_nodes: int,
    communities: List[List[int]],
    node_labels: Optional[List[str]] = None,
    edge_weights: Optional[Tensor] = None,
    title: str = "Community Structure",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Visualise graph with nodes colored by detected community.

    Parameters
    ----------
    communities : list of node lists (one per community)
    """
    if not HAS_MPL:
        return None

    node_colors = ["#9E9E9E"] * num_nodes
    for cidx, community in enumerate(communities):
        color = COMMUNITY_PALETTE[cidx % len(COMMUNITY_PALETTE)]
        for node in community:
            if 0 <= node < num_nodes:
                node_colors[node] = color

    return draw_financial_graph(
        edge_index, num_nodes,
        edge_weights=edge_weights,
        node_labels=node_labels,
        node_colors=node_colors,
        title=title,
        figsize=figsize,
        save_path=save_path,
        return_fig=True,
    )


def plot_community_matrix(
    communities: List[List[int]],
    corr_matrix: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Community Correlation Structure",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot correlation matrix reordered by community membership.
    """
    if not HAS_MPL:
        return None

    # Build reordering
    order = []
    for c in communities:
        order.extend(c)
    remaining = [i for i in range(corr_matrix.shape[0]) if i not in order]
    order.extend(remaining)

    reordered = corr_matrix[np.ix_(order, order)]
    names = [asset_names[i] for i in order] if asset_names else [f"A{i}" for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(reordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)

    n = len(order)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=max(4, 8 - n // 10))
    ax.set_yticklabels(names, fontsize=max(4, 8 - n // 10))
    ax.set_title(title, fontsize=14)

    # Draw community boundaries
    boundary = 0
    for c in communities[:-1]:
        boundary += len(c)
        ax.axhline(y=boundary - 0.5, color="black", linewidth=2)
        ax.axvline(x=boundary - 0.5, color="black", linewidth=2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Spectral plots
# ---------------------------------------------------------------------------

def plot_laplacian_spectrum(
    edge_index: Tensor,
    num_nodes: int,
    edge_weights: Optional[Tensor] = None,
    title: str = "Laplacian Eigenvalue Spectrum",
    figsize: Tuple[int, int] = (10, 5),
    n_eigenvalues: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot the eigenvalue spectrum of the graph Laplacian.
    """
    if not HAS_MPL:
        return None

    n = min(num_nodes, 500)
    A = np.zeros((n, n), dtype=np.float32)
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if 0 <= i < n and 0 <= j < n and i != j:
            w = float(edge_weights[k]) if edge_weights is not None and k < len(edge_weights) else 1.0
            A[i, j] = abs(w)
            A[j, i] = abs(w)

    D = np.diag(A.sum(axis=1))
    L = D - A

    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(eigvals)

    if n_eigenvalues is not None:
        eigvals = eigvals[:n_eigenvalues]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(eigvals, "o-", color="#2196F3", markersize=4)
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Eigenvalue Spectrum")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Fiedler value highlight
    if len(eigvals) > 1:
        fiedler = eigvals[1]
        axes[0].axhline(y=fiedler, color="#F44336", linestyle="--",
                        label=f"Fiedler λ₂={fiedler:.4f}")
        axes[0].legend(fontsize=8)

    axes[1].hist(eigvals, bins=30, color="#4CAF50", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Eigenvalue")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Eigenvalue Distribution")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_fiedler_series(
    fiedler_series: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    regime_series: Optional[np.ndarray] = None,
    title: str = "Fiedler Value (Market Connectivity / Liquidity Proxy)",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot Fiedler value over time, optionally colored by regime.
    """
    if not HAS_MPL:
        return None

    T = len(fiedler_series)
    xs = timestamps if timestamps is not None else np.arange(T)

    fig, ax = plt.subplots(figsize=figsize)

    if regime_series is not None:
        for t in range(len(xs) - 1):
            r = int(regime_series[t]) if t < len(regime_series) else 0
            color = REGIME_COLORS.get(r, "#9E9E9E")
            ax.fill_between(xs[t : t + 2], 0, fiedler_series[t : t + 2], alpha=0.15, color=color)

    ax.plot(xs, fiedler_series, color="#1A237E", linewidth=1.5, label="Fiedler λ₂")
    ax.axhline(y=0.01, color="#F44336", linestyle="--", alpha=0.7, label="Alert threshold (0.01)")

    ax.set_xlabel("Time")
    ax.set_ylabel("Fiedler Value")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Interactive Plotly graph
# ---------------------------------------------------------------------------

def plotly_financial_graph(
    edge_index: Tensor,
    num_nodes: int,
    edge_weights: Optional[Tensor] = None,
    node_labels: Optional[List[str]] = None,
    node_colors: Optional[List[str]] = None,
    node_sizes: Optional[List[float]] = None,
    title: str = "Financial Network",
    layout: str = "spring",
    return_html: bool = False,
) -> Optional[Any]:
    """
    Create an interactive Plotly graph.

    Parameters
    ----------
    return_html : if True, return HTML string instead of figure

    Returns
    -------
    plotly Figure or HTML string
    """
    if not HAS_PLOTLY:
        warnings.warn("plotly required for plotly_financial_graph")
        return None
    if not HAS_NX:
        warnings.warn("networkx required for plotly_financial_graph")
        return None

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        w = float(edge_weights[k]) if edge_weights is not None and k < len(edge_weights) else 1.0
        if i != j:
            G.add_edge(i, j, weight=abs(w), raw_w=w)

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=2.0 / math.sqrt(max(num_nodes, 1)))
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, weight="weight") if G.number_of_edges() > 0 else nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Edge traces
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data.get("raw_w", 1.0)
        color = "blue" if w >= 0 else "red"
        width = 1 + 3 * data.get("weight", 1.0)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none",
            showlegend=False,
        ))

    # Node trace
    node_x = [pos[i][0] for i in range(num_nodes)]
    node_y = [pos[i][1] for i in range(num_nodes)]
    labels = node_labels or [str(i) for i in range(num_nodes)]
    ncolors = node_colors or [f"#{hash(i) % 0xFFFFFF:06X}" for i in range(num_nodes)]
    nsizes = node_sizes or [15] * num_nodes

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=nsizes, color=ncolors, line=dict(width=1, color="white")),
        text=labels,
        textposition="top center",
        hoverinfo="text",
        hovertext=[f"{labels[i]}<br>Degree: {G.degree(i)}" for i in range(num_nodes)],
        showlegend=False,
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        ),
    )

    if return_html:
        return fig.to_html(full_html=False)
    return fig


# ---------------------------------------------------------------------------
# Node embedding visualization
# ---------------------------------------------------------------------------

def plot_node_embeddings(
    embeddings: Union[np.ndarray, Tensor],
    labels: Optional[List[int]] = None,
    node_names: Optional[List[str]] = None,
    title: str = "Node Embedding Space (PCA)",
    figsize: Tuple[int, int] = (10, 8),
    method: str = "pca",
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Visualize node embeddings in 2D via PCA or t-SNE.

    Parameters
    ----------
    embeddings : (N, D)
    labels     : optional regime or community labels per node
    method     : pca | tsne
    """
    if not HAS_MPL:
        return None

    if isinstance(embeddings, Tensor):
        emb = embeddings.detach().cpu().numpy()
    else:
        emb = embeddings

    N = emb.shape[0]

    # Dimensionality reduction
    if emb.shape[1] > 2:
        if method == "pca" and HAS_SKLEARN:
            reducer = PCA(n_components=2)
            emb_2d = reducer.fit_transform(emb)
        elif method == "tsne":
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, N - 1))
                emb_2d = reducer.fit_transform(emb)
            except ImportError:
                # Fall back to PCA
                if HAS_SKLEARN:
                    reducer = PCA(n_components=2)
                    emb_2d = reducer.fit_transform(emb)
                else:
                    emb_2d = emb[:, :2]
        else:
            emb_2d = emb[:, :2]
    else:
        emb_2d = emb

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = sorted(set(labels))
        for lbl in unique_labels:
            mask = [l == lbl for l in labels]
            pts = emb_2d[mask]
            color = REGIME_COLORS.get(lbl, COMMUNITY_PALETTE[lbl % len(COMMUNITY_PALETTE)])
            name = REGIME_NAMES.get(lbl, f"Class {lbl}")
            ax.scatter(pts[:, 0], pts[:, 1], c=color, label=name, s=80, alpha=0.8, edgecolors="white")

            if node_names:
                idx_list = [i for i, m in enumerate(mask) if m]
                for i in idx_list:
                    ax.annotate(node_names[i], emb_2d[i], fontsize=6, alpha=0.7)
        ax.legend(fontsize=9)
    else:
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c="#2196F3", s=80, alpha=0.8, edgecolors="white")
        if node_names:
            for i, name in enumerate(node_names[:N]):
                ax.annotate(name, emb_2d[i], fontsize=6, alpha=0.7)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Regime timeline
# ---------------------------------------------------------------------------

def plot_regime_timeline(
    regime_series: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    regime_probs: Optional[np.ndarray] = None,
    title: str = "Market Regime Timeline",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot market regime classification over time.

    Parameters
    ----------
    regime_series  : (T,) integer regime labels
    timestamps     : (T,) x-axis timestamps
    regime_probs   : (T, n_regimes) optional probability series
    """
    if not HAS_MPL:
        return None

    T = len(regime_series)
    xs = timestamps if timestamps is not None else np.arange(T)

    n_panels = 2 if regime_probs is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: regime as colored bands
    ax = axes[0]
    ax.set_ylabel("Regime")
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([REGIME_NAMES.get(i, f"R{i}") for i in range(4)])

    for t in range(len(xs) - 1):
        r = int(regime_series[t])
        color = REGIME_COLORS.get(r, "#9E9E9E")
        ax.axvspan(xs[t], xs[t + 1], ymin=r / 4, ymax=(r + 1) / 4, alpha=0.4, color=color)

    ax.step(xs, regime_series, where="post", color="#1A237E", linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 2: regime probability stacked area
    if regime_probs is not None:
        ax2 = axes[1]
        n_regimes = regime_probs.shape[1]
        cumulative = np.zeros(T)
        for r in range(n_regimes):
            p = regime_probs[:, r]
            color = REGIME_COLORS.get(r, "#9E9E9E")
            ax2.fill_between(xs, cumulative, cumulative + p, alpha=0.7, color=color,
                             label=REGIME_NAMES.get(r, f"R{r}"))
            cumulative += p
        ax2.set_ylabel("Regime Probability")
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Anomaly score plot
# ---------------------------------------------------------------------------

def plot_anomaly_scores(
    anomaly_scores: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    fiedler_series: Optional[np.ndarray] = None,
    threshold: float = 0.6,
    title: str = "Graph Anomaly Scores",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """Plot anomaly scores over time with threshold markers."""
    if not HAS_MPL:
        return None

    T = len(anomaly_scores)
    xs = timestamps if timestamps is not None else np.arange(T)

    n_panels = 2 if fiedler_series is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(xs, anomaly_scores, color="#F44336", linewidth=1.5, label="Anomaly Score")
    ax.axhline(y=threshold, color="#FF9800", linestyle="--", label=f"Threshold ({threshold})")
    alarm_mask = anomaly_scores > threshold
    if alarm_mask.any():
        ax.scatter(xs[alarm_mask], anomaly_scores[alarm_mask], c="red", zorder=5, s=30, label="Alarm")
    ax.fill_between(xs, 0, anomaly_scores, alpha=0.15, color="#F44336")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    if fiedler_series is not None:
        ax2 = axes[1]
        ax2.plot(xs[:len(fiedler_series)], fiedler_series, color="#1A237E", linewidth=1.5)
        ax2.set_ylabel("Fiedler Value")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(xs[:len(fiedler_series)], 0, fiedler_series, alpha=0.1, color="#1A237E")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Dashboard-style summary plot
# ---------------------------------------------------------------------------

def plot_graph_dashboard(
    edge_index: Tensor,
    num_nodes: int,
    corr_matrix: Optional[np.ndarray] = None,
    node_labels: Optional[List[str]] = None,
    edge_weights: Optional[Tensor] = None,
    node_regimes: Optional[List[int]] = None,
    title: str = "Graph Dashboard",
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Multi-panel dashboard combining graph view, heatmap, and degree distribution.
    """
    if not HAS_MPL:
        return None

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Graph
    ax1 = fig.add_subplot(gs[0, :2])
    if node_regimes:
        nc = [REGIME_COLORS.get(r, "#90CAF9") for r in node_regimes]
    else:
        nc = ["#90CAF9"] * num_nodes
    if HAS_NX:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if i != j:
                G.add_edge(i, j)
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos, ax=ax1, node_color=nc, node_size=200,
                         with_labels=True if num_nodes < 30 else False,
                         labels={i: (node_labels[i] if node_labels else str(i)) for i in range(num_nodes)},
                         font_size=7, edge_color="#BDBDBD", alpha=0.8)
    ax1.set_title("Graph Structure", fontsize=11)
    ax1.axis("off")

    # Panel 2: Correlation heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    if corr_matrix is not None:
        im = ax2.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax2, shrink=0.6)
        ax2.set_title("Correlation Matrix", fontsize=11)
        ax2.set_xticks([])
        ax2.set_yticks([])

    # Panel 3: Degree distribution
    ax3 = fig.add_subplot(gs[1, 0])
    degrees = np.zeros(num_nodes)
    for k in range(edge_index.shape[1]):
        i = int(edge_index[0, k])
        if 0 <= i < num_nodes:
            degrees[i] += 1
    ax3.hist(degrees, bins=20, color="#4CAF50", alpha=0.7, edgecolor="black")
    ax3.set_xlabel("Degree")
    ax3.set_ylabel("Count")
    ax3.set_title("Degree Distribution", fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Edge weight distribution
    ax4 = fig.add_subplot(gs[1, 1])
    if edge_weights is not None:
        w = edge_weights.detach().cpu().numpy().flatten()
        ax4.hist(w, bins=30, color="#2196F3", alpha=0.7, edgecolor="black")
        ax4.set_xlabel("Edge Weight")
        ax4.set_ylabel("Count")
        ax4.set_title("Edge Weight Distribution", fontsize=11)
        ax4.grid(True, alpha=0.3)

    # Panel 5: Stats text
    ax5 = fig.add_subplot(gs[1, 2])
    E = int(edge_index.shape[1])
    density = 2 * E / max(num_nodes * (num_nodes - 1), 1)
    stats_text = (
        f"Nodes: {num_nodes}\n"
        f"Edges: {E}\n"
        f"Density: {density:.4f}\n"
        f"Mean degree: {degrees.mean():.2f}\n"
        f"Max degree: {int(degrees.max())}\n"
    )
    if corr_matrix is not None:
        off_diag = corr_matrix[np.triu_indices(num_nodes, k=1)]
        stats_text += f"Mean |corr|: {np.abs(off_diag).mean():.4f}\n"
        stats_text += f"Std corr: {off_diag.std():.4f}\n"

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#ECEFF1", alpha=0.8))
    ax5.set_title("Graph Statistics", fontsize=11)
    ax5.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "REGIME_COLORS",
    "REGIME_NAMES",
    "COMMUNITY_PALETTE",
    "draw_financial_graph",
    "draw_regime_colored_graph",
    "plot_correlation_heatmap",
    "plot_edge_weight_distribution",
    "animate_temporal_graph",
    "draw_community_graph",
    "plot_community_matrix",
    "plot_laplacian_spectrum",
    "plot_fiedler_series",
    "plotly_financial_graph",
    "plot_node_embeddings",
    "plot_regime_timeline",
    "plot_anomaly_scores",
    "plot_graph_dashboard",
]
