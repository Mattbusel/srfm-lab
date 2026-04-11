"""
financial_graphs.py — Build financial graphs from return time series.

Functions:
    build_correlation_graph       Rolling correlation matrix -> PyG graph.
    build_granger_graph           Granger causality DAG.
    build_partial_correlation_graph  GLASSO partial correlation.
    build_transfer_entropy_graph  Transfer entropy directed graph.

Classes:
    GraphEvolution                Manages rolling window graph snapshots.

Visualization:
    visualize_graph               Static graph with curvature color coding.
    animate_graph_evolution       Animated evolution of the graph.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from torch_geometric.data import Data
import torch
import networkx as nx

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ── Utility ───────────────────────────────────────────────────────────────────

def returns_to_numpy(returns: Any) -> np.ndarray:
    """Coerce returns to (T, N) numpy array."""
    if isinstance(returns, pd.DataFrame):
        return returns.values.astype(float)
    elif isinstance(returns, np.ndarray):
        return returns.astype(float)
    else:
        return np.array(returns, dtype=float)


def correlation_matrix_to_graph(
    corr: np.ndarray,
    threshold: float,
    node_features: Optional[np.ndarray] = None,
    directed: bool = False,
) -> Data:
    """Convert a correlation matrix to a PyG graph.

    Args:
        corr:           (N, N) correlation matrix.
        threshold:      Minimum absolute correlation to include edge.
        node_features:  (N, F) optional node feature matrix.
        directed:       If True, keep directional info.
    Returns:
        PyG Data object.
    """
    n = corr.shape[0]
    if node_features is None:
        # Default node features: degree-like statistics from corr matrix
        node_features = np.column_stack([
            np.abs(corr).sum(axis=1) / n,            # strength
            (corr > threshold).sum(axis=1) / n,       # degree fraction
            corr.diagonal(),                           # self-correlation (1.0)
            np.abs(corr).mean(axis=1),                 # mean abs corr
        ])

    edges_src, edges_dst, edge_weights = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= threshold:
                edges_src.append(i)
                edges_dst.append(j)
                edge_weights.append(abs(corr[i, j]))
                if not directed:
                    edges_src.append(j)
                    edges_dst.append(i)
                    edge_weights.append(abs(corr[i, j]))

    if not edges_src:
        # Empty graph: connect everything at minimum weight
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1))
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    x = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


# ── build_correlation_graph ───────────────────────────────────────────────────

def build_correlation_graph(
    returns: Any,
    threshold: float = 0.3,
    method: str = "pearson",
    window: Optional[int] = None,
    min_periods: int = 20,
) -> Data:
    """Build a graph from rolling or full-sample correlation matrix.

    Args:
        returns:     (T, N) return matrix (DataFrame or numpy).
        threshold:   Minimum absolute correlation to include edge.
        method:      'pearson', 'spearman', or 'kendall'.
        window:      Rolling window size (None = full-sample).
        min_periods: Minimum non-NaN observations for rolling.
    Returns:
        PyG Data with node features and edge attributes.
    """
    R = returns_to_numpy(returns)
    T, N = R.shape

    if window is not None and window < T:
        # Use the last `window` periods
        R = R[-window:]

    # Handle NaN
    R = np.nan_to_num(R, nan=0.0)

    if method == "pearson":
        corr = np.corrcoef(R.T)
    elif method == "spearman":
        corr, _ = stats.spearmanr(R)
        if N == 2:
            corr = np.array([[1.0, corr], [corr, 1.0]])
    elif method == "kendall":
        corr = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    tau, _ = stats.kendalltau(R[:, i], R[:, j])
                    corr[i, j] = corr[j, i] = tau
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fix NaN in correlation matrix
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    # Node features: mean return, volatility, skewness, kurtosis
    node_features = np.column_stack([
        R.mean(axis=0),
        R.std(axis=0) + 1e-8,
        stats.skew(R, axis=0),
        stats.kurtosis(R, axis=0),
        np.abs(corr).mean(axis=1),   # mean absolute correlation (connectedness)
        (np.abs(corr) > threshold).sum(axis=1) / N,  # degree fraction
    ])
    node_features = np.nan_to_num(node_features, nan=0.0)

    return correlation_matrix_to_graph(corr, threshold, node_features)


# ── build_granger_graph ───────────────────────────────────────────────────────

def build_granger_graph(
    returns: Any,
    max_lag: int = 5,
    alpha: float = 0.05,
    n_permutations: int = 0,
) -> Data:
    """Build a directed Granger causality graph.

    For each ordered pair (i, j), tests whether past values of series i
    Granger-cause series j using an F-test. Significant causality creates
    a directed edge i -> j with weight = -log10(p-value).

    Args:
        returns:       (T, N) return matrix.
        max_lag:       Maximum lag to test.
        alpha:         Significance level.
        n_permutations: If > 0, use permutation test instead of F-test.
    Returns:
        PyG Data (directed graph).
    """
    R = returns_to_numpy(returns)
    T, N = R.shape
    R = np.nan_to_num(R, nan=0.0)

    edges_src, edges_dst, edge_weights = [], [], []

    for j in range(N):
        for i in range(N):
            if i == j:
                continue

            p_val = _granger_test(R[:, i], R[:, j], max_lag)
            if p_val < alpha:
                weight = float(-math.log10(p_val + 1e-300))
                edges_src.append(i)
                edges_dst.append(j)
                edge_weights.append(weight)

    # Node features: same as correlation graph
    node_features = np.column_stack([
        R.mean(axis=0),
        R.std(axis=0) + 1e-8,
        stats.skew(R, axis=0),
        stats.kurtosis(R, axis=0),
    ])
    node_features = np.nan_to_num(node_features, nan=0.0)

    if not edges_src:
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 1)),
            num_nodes=N,
        )

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    x = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)


def _granger_test(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    """Simple Granger causality F-test: does x Granger-cause y?
    Returns p-value.
    """
    T = len(y)
    if T < 2 * max_lag + 10:
        return 1.0

    # Restricted model: AR(max_lag) of y only
    # Unrestricted model: AR(max_lag) of y + max_lag lags of x
    Y = y[max_lag:]
    T_eff = len(Y)

    # Build lagged matrices
    X_r = np.column_stack([
        np.ones(T_eff),
        *[y[max_lag - k - 1:T - k - 1] for k in range(max_lag)]
    ])
    X_u = np.column_stack([
        X_r,
        *[x[max_lag - k - 1:T - k - 1] for k in range(max_lag)]
    ])

    try:
        # OLS estimates
        b_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
        b_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]

        rss_r = np.sum((Y - X_r @ b_r) ** 2)
        rss_u = np.sum((Y - X_u @ b_u) ** 2)

        k = max_lag
        n = T_eff
        df1 = k
        df2 = n - 2 * k - 1

        if df2 <= 0 or rss_u < 1e-14:
            return 1.0

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        if f_stat < 0:
            return 1.0

        p_val = 1.0 - stats.f.cdf(f_stat, df1, df2)
        return float(p_val)
    except np.linalg.LinAlgError:
        return 1.0


# ── build_partial_correlation_graph ──────────────────────────────────────────

def build_partial_correlation_graph(
    returns: Any,
    alpha: float = 0.1,
    cv: bool = False,
    threshold: float = 0.0,
) -> Data:
    """Build a partial correlation graph using GLASSO.

    The precision matrix from GLASSO encodes partial correlations: a non-zero
    entry (i, j) means asset i and j are conditionally dependent given all
    other assets.

    Args:
        returns:   (T, N) return matrix.
        alpha:     GLASSO regularization (larger = sparser graph).
        cv:        If True, use cross-validation to select alpha.
        threshold: Minimum |partial correlation| to include edge.
    Returns:
        PyG Data.
    """
    R = returns_to_numpy(returns)
    T, N = R.shape
    R = np.nan_to_num(R, nan=0.0)

    # Standardize
    R_std = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-8)

    if T < N + 5:
        # Not enough data: fall back to marginal correlation
        warnings.warn("Not enough data for GLASSO, falling back to correlation.")
        return build_correlation_graph(R, threshold=max(threshold, 0.1))

    try:
        if cv:
            model = GraphicalLassoCV(cv=3)
        else:
            model = GraphicalLasso(alpha=alpha)
        model.fit(R_std)
        precision = model.precision_
    except Exception as e:
        warnings.warn(f"GLASSO failed: {e}. Using correlation.")
        return build_correlation_graph(R, threshold=max(threshold, 0.1))

    # Convert precision to partial correlation
    # pcorr(i,j) = -precision(i,j) / sqrt(precision(i,i) * precision(j,j))
    partial_corr = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            denom = math.sqrt(abs(precision[i, i] * precision[j, j]))
            if denom > 1e-10:
                partial_corr[i, j] = -precision[i, j] / denom
    np.fill_diagonal(partial_corr, 1.0)

    # Node features
    node_features = np.column_stack([
        R.mean(axis=0),
        R.std(axis=0) + 1e-8,
        stats.skew(R, axis=0),
        np.diag(precision),
        np.abs(partial_corr).sum(axis=1) / N,
    ])
    node_features = np.nan_to_num(node_features, nan=0.0)

    return correlation_matrix_to_graph(partial_corr, threshold, node_features)


# ── build_transfer_entropy_graph ──────────────────────────────────────────────

def build_transfer_entropy_graph(
    returns: Any,
    bins: int = 10,
    lag: int = 1,
    alpha: float = 0.05,
    n_bootstrap: int = 50,
) -> Data:
    """Build a directed Transfer Entropy graph.

    TE(X -> Y) measures how much knowing past X reduces uncertainty about Y,
    above and beyond knowing past Y. Non-zero TE creates a directed edge X -> Y.

    Args:
        returns:     (T, N) return matrix.
        bins:        Number of bins for histogram-based entropy.
        lag:         Time lag.
        alpha:       Significance level for bootstrap test.
        n_bootstrap: Number of bootstrap permutations for significance test.
    Returns:
        PyG Data (directed graph).
    """
    R = returns_to_numpy(returns)
    T, N = R.shape
    R = np.nan_to_num(R, nan=0.0)

    # Discretize each time series
    def discretize(x: np.ndarray, n_bins: int) -> np.ndarray:
        bin_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-10
        return np.digitize(x, bin_edges[1:-1])

    R_disc = np.column_stack([discretize(R[:, i], bins) for i in range(N)])

    edges_src, edges_dst, edge_weights = [], [], []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            te = _transfer_entropy(R_disc[:, i], R_disc[:, j], lag)
            if te <= 0:
                continue

            # Bootstrap significance test
            if n_bootstrap > 0:
                te_null = []
                for _ in range(n_bootstrap):
                    perm = np.random.permutation(len(R_disc[:, i]))
                    te_null.append(_transfer_entropy(R_disc[perm, i], R_disc[:, j], lag))
                threshold = np.percentile(te_null, 100 * (1 - alpha))
                if te <= threshold:
                    continue

            edges_src.append(i)
            edges_dst.append(j)
            edge_weights.append(float(te))

    # Node features
    node_features = np.column_stack([
        R.mean(axis=0),
        R.std(axis=0) + 1e-8,
        stats.skew(R, axis=0),
        stats.kurtosis(R, axis=0),
    ])
    node_features = np.nan_to_num(node_features, nan=0.0)

    if not edges_src:
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 1)),
            num_nodes=N,
        )

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    x = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)


def _transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """Compute TE(X -> Y) via conditional entropy.

    TE(X -> Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
    """
    T = len(y)
    if T < lag + 2:
        return 0.0

    y_t = y[lag:]
    y_prev = y[:-lag]
    x_prev = x[:-lag]

    # H(Y_t | Y_{t-1})
    h_y_given_yprev = _conditional_entropy(y_t, y_prev)

    # H(Y_t | Y_{t-1}, X_{t-1})
    # Joint state of (Y_{t-1}, X_{t-1})
    joint_yx_prev = y_prev * 100 + x_prev  # simple combination for histogram
    h_y_given_yx = _conditional_entropy(y_t, joint_yx_prev)

    te = h_y_given_yprev - h_y_given_yx
    return max(0.0, float(te))


def _conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    """H(Y | X) = H(Y, X) - H(X)."""
    xy_counts = {}
    x_counts = {}
    T = len(y)

    for yi, xi in zip(y, x):
        xy_key = (int(yi), int(xi))
        xy_counts[xy_key] = xy_counts.get(xy_key, 0) + 1
        x_counts[int(xi)] = x_counts.get(int(xi), 0) + 1

    h_joint = -sum(v / T * math.log2(v / T + 1e-300) for v in xy_counts.values())
    h_x = -sum(v / T * math.log2(v / T + 1e-300) for v in x_counts.values())
    return h_joint - h_x


# ── GraphEvolution ────────────────────────────────────────────────────────────

class GraphEvolution:
    """Manages rolling window graph snapshots over a return time series.

    Builds a new graph snapshot at each step of a rolling window,
    accumulates them, and provides utilities for temporal analysis.

    Args:
        returns:         (T, N) return matrix (DataFrame or numpy).
        window:          Rolling window size.
        step:            Step between windows (1 = daily update).
        graph_type:      'correlation', 'granger', 'partial_correlation', or 'transfer_entropy'.
        threshold:       Minimum edge weight to include.
        asset_names:     Optional list of asset names.
        **kwargs:        Additional arguments passed to graph builder.
    """

    def __init__(
        self,
        returns: Any,
        window: int = 60,
        step: int = 5,
        graph_type: str = "correlation",
        threshold: float = 0.3,
        asset_names: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        self.returns = returns_to_numpy(returns)
        self.window = window
        self.step = step
        self.graph_type = graph_type
        self.threshold = threshold
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.returns.shape[1])]
        self.kwargs = kwargs
        self.T, self.N = self.returns.shape

        # Storage
        self.snapshots: List[Data] = []
        self.timestamps: List[int] = []
        self.edge_counts: List[int] = []
        self.mean_weights: List[float] = []

        # Pre-computed Ricci curvatures (from Rust, if available)
        self.ricci_curvatures: List[Optional[np.ndarray]] = []

    def build_snapshot(self, start: int, end: int) -> Data:
        """Build a graph from returns[start:end]."""
        R_window = self.returns[start:end]
        if self.graph_type == "correlation":
            return build_correlation_graph(R_window, self.threshold, **self.kwargs)
        elif self.graph_type == "granger":
            return build_granger_graph(R_window, **self.kwargs)
        elif self.graph_type == "partial_correlation":
            return build_partial_correlation_graph(R_window, **self.kwargs)
        elif self.graph_type == "transfer_entropy":
            return build_transfer_entropy_graph(R_window, **self.kwargs)
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")

    def run(self, verbose: bool = False) -> "GraphEvolution":
        """Build all rolling snapshots."""
        starts = range(0, self.T - self.window, self.step)
        if verbose:
            print(f"Building {len(starts)} graph snapshots "
                  f"(window={self.window}, step={self.step}, type={self.graph_type})")

        for start in starts:
            end = start + self.window
            snap = self.build_snapshot(start, end)
            self.snapshots.append(snap)
            self.timestamps.append(end)
            self.ricci_curvatures.append(None)

            # Track statistics
            n_edges = snap.edge_index.size(1)
            self.edge_counts.append(n_edges)
            if snap.edge_attr is not None and snap.edge_attr.size(0) > 0:
                self.mean_weights.append(float(snap.edge_attr[:, 0].mean()))
            else:
                self.mean_weights.append(0.0)

        if verbose:
            print(f"Built {len(self.snapshots)} snapshots. "
                  f"Avg edges: {np.mean(self.edge_counts):.1f}")
        return self

    def set_ricci_curvatures(self, ricci_list: List[Optional[np.ndarray]]) -> None:
        """Set pre-computed Ricci curvatures (from Rust engine) for all snapshots."""
        assert len(ricci_list) == len(self.snapshots), \
            f"Length mismatch: {len(ricci_list)} != {len(self.snapshots)}"
        self.ricci_curvatures = ricci_list

    def compute_edge_ages(self) -> List[torch.Tensor]:
        """Compute edge ages (steps since each edge first appeared) for each snapshot.

        Returns:
            List of (E,) age tensors, one per snapshot.
        """
        if not self.snapshots:
            return []

        edge_birth: Dict[Tuple[int, int], int] = {}
        age_tensors = []

        for t, snap in enumerate(self.snapshots):
            ei = snap.edge_index
            n_edges = ei.size(1)
            ages = torch.zeros(n_edges)
            for e_idx in range(n_edges):
                src = int(ei[0, e_idx])
                dst = int(ei[1, e_idx])
                key = (min(src, dst), max(src, dst))
                if key not in edge_birth:
                    edge_birth[key] = t
                ages[e_idx] = float(t - edge_birth[key])
            age_tensors.append(ages)

        return age_tensors

    def get_networkx_graph(self, t: int) -> nx.Graph:
        """Convert snapshot t to a NetworkX graph."""
        snap = self.snapshots[t]
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        ei = snap.edge_index.numpy()
        for e_idx in range(ei.shape[1]):
            src, dst = int(ei[0, e_idx]), int(ei[1, e_idx])
            if src != dst:
                w = float(snap.edge_attr[e_idx, 0]) if snap.edge_attr is not None else 1.0
                G.add_edge(src, dst, weight=w)

        # Add node labels
        for i, name in enumerate(self.asset_names):
            G.nodes[i]["name"] = name

        return G

    def summary_statistics(self) -> pd.DataFrame:
        """Return a DataFrame of snapshot statistics over time."""
        records = []
        for t, (snap, ts) in enumerate(zip(self.snapshots, self.timestamps)):
            n_edges = snap.edge_index.size(1)
            if snap.edge_attr is not None and snap.edge_attr.size(0) > 0:
                ew = snap.edge_attr[:, 0].numpy()
                ew_mean = float(ew.mean())
                ew_std = float(ew.std())
                ew_max = float(ew.max())
            else:
                ew_mean = ew_std = ew_max = 0.0

            density = n_edges / max(self.N * (self.N - 1), 1)

            rc = self.ricci_curvatures[t]
            rc_mean = float(np.mean(rc)) if rc is not None else None
            rc_min = float(np.min(rc)) if rc is not None else None

            records.append({
                "t": ts,
                "n_edges": n_edges,
                "density": density,
                "mean_weight": ew_mean,
                "std_weight": ew_std,
                "max_weight": ew_max,
                "ricci_mean": rc_mean,
                "ricci_min": rc_min,
            })
        return pd.DataFrame(records)


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_graph(
    data: Data,
    asset_names: Optional[List[str]] = None,
    ricci_curvatures: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """Plot a financial graph with optional Ricci curvature color coding.

    Args:
        data:             PyG Data object.
        asset_names:      Node labels.
        ricci_curvatures: (E,) edge curvature values for color coding.
        title:            Plot title.
        save_path:        If provided, save to this path.
        figsize:          Figure size.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Cannot visualize.")
        return

    n = data.x.size(0)
    names = asset_names or [str(i) for i in range(n)]

    G = nx.Graph()
    G.add_nodes_from(range(n))

    ei = data.edge_index.numpy()
    for e_idx in range(ei.shape[1]):
        src, dst = int(ei[0, e_idx]), int(ei[1, e_idx])
        if src < dst:
            w = float(data.edge_attr[e_idx, 0]) if data.edge_attr is not None else 1.0
            G.add_edge(src, dst, weight=w, edge_idx=e_idx)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    pos = nx.spring_layout(G, seed=42, weight="weight")

    # Node color: node risk (from node features)
    if data.x.size(1) > 0:
        node_strength = data.x[:, 0].numpy()
        vmin, vmax = node_strength.min(), node_strength.max()
        node_colors = plt.cm.RdYlGn(
            (node_strength - vmin) / (vmax - vmin + 1e-8)
        )
    else:
        node_colors = ["lightblue"] * n

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: names[i] for i in range(n)}, ax=ax,
                            font_size=8)

    # Edge color: Ricci curvature (blue = positive, red = negative)
    edges = list(G.edges(data=True))
    if len(edges) > 0:
        edge_weights = [d["weight"] for _, _, d in edges]

        if ricci_curvatures is not None:
            # Map edges to curvature values
            edge_curv = []
            for src, dst, d in edges:
                eidx = d.get("edge_idx", 0)
                curv = float(ricci_curvatures[eidx]) if eidx < len(ricci_curvatures) else 0.0
                edge_curv.append(curv)
            edge_colors = plt.cm.coolwarm(
                np.array([(c + 1) / 2 for c in edge_curv])
            )
            edge_widths = [w * 3 for w in edge_weights]
        else:
            edge_colors = [plt.cm.Reds(w) for w in edge_weights]
            edge_widths = [w * 3 for w in edge_weights]

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(s, d) for s, d, _ in edges],
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            ax=ax,
        )

    ax.set_title(title or f"Financial Graph (N={n}, E={G.number_of_edges()})")
    ax.axis("off")

    if ricci_curvatures is not None:
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-1, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Ollivier-Ricci Curvature", shrink=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def animate_graph_evolution(
    evolution: GraphEvolution,
    save_path: str = "graph_evolution.mp4",
    interval: int = 200,
    max_frames: int = 50,
) -> None:
    """Create an animated video of graph evolution.

    Args:
        evolution:  GraphEvolution instance with snapshots.
        save_path:  Output file path (.mp4 or .gif).
        interval:   Milliseconds between frames.
        max_frames: Maximum number of frames.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available.")
        return

    n_snaps = len(evolution.snapshots)
    n_frames = min(n_snaps, max_frames)
    step_size = max(1, n_snaps // n_frames)
    frame_indices = list(range(0, n_snaps, step_size))[:n_frames]

    fig, (ax_graph, ax_stats) = plt.subplots(1, 2, figsize=(16, 7))

    # Pre-compute consistent layout from the densest snapshot
    densest = max(range(n_snaps), key=lambda i: evolution.edge_counts[i])
    G_dense = evolution.get_networkx_graph(densest)
    pos = nx.spring_layout(G_dense, seed=42)

    def update(frame_idx: int):
        ax_graph.clear()
        ax_stats.clear()

        t = frame_indices[frame_idx]
        snap = evolution.snapshots[t]
        G = evolution.get_networkx_graph(t)

        # Graph panel
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        node_colors = []
        x_feat = snap.x.numpy()
        for i in range(evolution.N):
            strength = float(x_feat[i, 0]) if x_feat.shape[1] > 0 else 0.5
            node_colors.append(plt.cm.RdYlGn(np.clip((strength + 1) / 2, 0, 1)))

        nx.draw_networkx(
            G, pos=pos,
            node_color=node_colors,
            edge_color="gray",
            width=[w * 2 for w in edge_weights] if edge_weights else [],
            labels={i: evolution.asset_names[i] for i in range(evolution.N)},
            font_size=7,
            node_size=200,
            ax=ax_graph,
        )
        ax_graph.set_title(f"Step {t} | Edges: {G.number_of_edges()}")
        ax_graph.axis("off")

        # Stats panel
        ax_stats.plot(
            evolution.timestamps[:t+1],
            evolution.edge_counts[:t+1],
            'b-', label="Edge count"
        )
        ax_stats.axvline(evolution.timestamps[t], color='r', linestyle='--', alpha=0.5)
        ax_stats.set_xlabel("Time step")
        ax_stats.set_ylabel("Edge count")
        ax_stats.set_title("Graph Evolution")
        ax_stats.legend()

        if evolution.ricci_curvatures[t] is not None:
            rc = evolution.ricci_curvatures[t]
            ax_stats.set_title(f"Edges: {G.number_of_edges()}  Ricci: {np.mean(rc):.3f}")

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices),
        interval=interval, blit=False,
    )

    writer_kwargs = {}
    if save_path.endswith(".gif"):
        writer = animation.PillowWriter(fps=5)
    else:
        writer = animation.FFMpegWriter(fps=5)

    try:
        anim.save(save_path, writer=writer, **writer_kwargs)
        print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Could not save animation: {e}")
    finally:
        plt.close(fig)
