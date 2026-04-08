"""
Graph Neural Network for financial asset correlation graphs.

Implements (numpy-only, no external DL framework):
  - GraphConv layer: message passing aggregation
  - GATConv: Graph Attention Network layer
  - GraphSAGE: inductive representation learning
  - Temporal GNN: evolving graph over time
  - Node embedding training (supervised + unsupervised)
  - Portfolio GNN: learn optimal weights from graph structure
  - Sector/industry graph construction
  - Community-aware embeddings
  - Link prediction: forecast new correlations
  - Graph-level readout: aggregate to portfolio signal
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-10)

def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sigma + eps)


# ── Graph Construction ────────────────────────────────────────────────────────

def correlation_graph(
    returns: np.ndarray,          # (T, N) return matrix
    threshold: float = 0.3,
    method: str = "pearson",
) -> dict:
    """
    Build adjacency matrix from return correlations.
    Edges: |corr| > threshold.
    """
    T, N = returns.shape

    if method == "pearson":
        corr = np.corrcoef(returns.T)
    elif method == "spearman":
        ranks = np.argsort(np.argsort(returns, axis=0), axis=0).astype(float)
        corr = np.corrcoef(ranks.T)
    elif method == "partial":
        # Partial correlation via precision matrix
        corr = np.corrcoef(returns.T)
        try:
            prec = np.linalg.inv(corr + np.eye(N) * 1e-6)
            D = np.diag(1 / np.sqrt(np.abs(np.diag(prec)) + 1e-10))
            corr = -D @ prec @ D
            np.fill_diagonal(corr, 1.0)
        except np.linalg.LinAlgError:
            pass
    else:
        corr = np.corrcoef(returns.T)

    np.fill_diagonal(corr, 0)
    adj = (np.abs(corr) > threshold).astype(float)

    # Weighted adjacency
    adj_weighted = corr * adj

    # Degree
    degree = adj.sum(axis=1)

    return {
        "adj": adj,
        "adj_weighted": adj_weighted,
        "corr_matrix": corr,
        "n_edges": int(adj.sum() / 2),
        "avg_degree": float(degree.mean()),
        "degree": degree,
    }


def build_sector_graph(
    sector_labels: list[str],
    within_sector_weight: float = 0.8,
    cross_sector_weight: float = 0.1,
) -> np.ndarray:
    """
    Build adjacency matrix from sector labels.
    Within-sector nodes have stronger connections.
    """
    N = len(sector_labels)
    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                if sector_labels[i] == sector_labels[j]:
                    adj[i, j] = within_sector_weight
                else:
                    adj[i, j] = cross_sector_weight
    return adj


# ── GraphConv Layer ───────────────────────────────────────────────────────────

class GraphConvLayer:
    """
    Graph Convolutional Network layer (Kipf & Welling 2017).
    H' = sigma(D^{-1/2} A_hat D^{-1/2} H W)
    where A_hat = A + I (self-loops added).
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, math.sqrt(2 / in_dim), (in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def normalize_adj(self, adj: np.ndarray) -> np.ndarray:
        """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
        A_hat = adj + np.eye(len(adj))
        degree = A_hat.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-10))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def forward(self, H: np.ndarray, adj: np.ndarray, activation: bool = True) -> np.ndarray:
        A_norm = self.normalize_adj(adj)
        out = A_norm @ H @ self.W + self.b
        return _relu(out) if activation else out

    def update(self, grad_W: np.ndarray, grad_b: np.ndarray, lr: float = 0.01) -> None:
        self.W -= lr * grad_W
        self.b -= lr * grad_b


# ── Graph Attention Network ───────────────────────────────────────────────────

class GATLayer:
    """
    Graph Attention Network layer (Veličković et al. 2018).
    Learns attention coefficients alpha_{ij} for each edge.
    h'_i = sigma(sum_j alpha_ij * W * h_j)
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        # Per-head weight matrices
        self.W = rng.normal(0, math.sqrt(1 / in_dim), (n_heads, in_dim, self.head_dim))
        # Attention vectors [W_i || W_j] -> scalar
        self.a = rng.normal(0, 0.1, (n_heads, 2 * self.head_dim))

    def forward(self, H: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        H: (N, in_dim)
        adj: (N, N) adjacency (0/1 or weighted)
        Returns: (N, out_dim)
        """
        N = H.shape[0]
        head_outputs = []

        for k in range(self.n_heads):
            # Linear transform
            Wh = H @ self.W[k]  # (N, head_dim)

            # Attention logits
            # e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
            Wh_i = np.repeat(Wh[:, np.newaxis, :], N, axis=1)  # (N, N, head_dim)
            Wh_j = np.repeat(Wh[np.newaxis, :, :], N, axis=0)  # (N, N, head_dim)
            concat = np.concatenate([Wh_i, Wh_j], axis=-1)     # (N, N, 2*head_dim)
            e = concat @ self.a[k]   # (N, N)
            # LeakyReLU
            e = np.where(e >= 0, e, 0.2 * e)

            # Mask non-edges
            mask = (adj > 0).astype(float)
            e = e * mask + (1 - mask) * (-1e9)

            # Softmax over neighbors
            alpha = _softmax(e, axis=-1)   # (N, N)
            alpha = alpha * mask           # zero out non-neighbors

            # Aggregate
            h_prime = alpha @ Wh   # (N, head_dim)
            head_outputs.append(h_prime)

        # Concatenate heads
        out = np.concatenate(head_outputs, axis=-1)   # (N, out_dim)
        return _relu(out)


# ── GraphSAGE Layer ───────────────────────────────────────────────────────────

class GraphSAGELayer:
    """
    GraphSAGE (Hamilton et al. 2017) — inductive node representation.
    h'_i = sigma(W * [h_i || AGGREGATE(h_j for j in N(i))])
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregator: str = "mean",
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.aggregator = aggregator
        # Self + neighbor combined
        self.W = rng.normal(0, math.sqrt(2 / (2 * in_dim)), (2 * in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def forward(self, H: np.ndarray, adj: np.ndarray) -> np.ndarray:
        N = H.shape[0]
        # Neighborhood aggregation
        if self.aggregator == "mean":
            degree = adj.sum(axis=1, keepdims=True) + 1e-10
            neigh_agg = (adj @ H) / degree
        elif self.aggregator == "max":
            # Masked max pooling
            neigh_agg = np.zeros_like(H)
            for i in range(N):
                neighbors = np.where(adj[i] > 0)[0]
                if len(neighbors) > 0:
                    neigh_agg[i] = H[neighbors].max(axis=0)
                else:
                    neigh_agg[i] = H[i]
        elif self.aggregator == "sum":
            neigh_agg = adj @ H
        else:
            degree = adj.sum(axis=1, keepdims=True) + 1e-10
            neigh_agg = (adj @ H) / degree

        # Concatenate self + neighborhood
        combined = np.concatenate([H, neigh_agg], axis=-1)  # (N, 2*in_dim)
        out = combined @ self.W + self.b
        # L2 normalize
        norm = np.linalg.norm(out, axis=-1, keepdims=True) + 1e-10
        return _relu(out / norm)


# ── Temporal GNN ─────────────────────────────────────────────────────────────

class TemporalGNN:
    """
    Temporal Graph Neural Network: evolving graph over time.
    At each timestep: GNN update then GRU-like temporal update.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.gc1 = GraphConvLayer(in_dim, hidden_dim, seed)
        self.gc2 = GraphConvLayer(hidden_dim, hidden_dim, seed + 1)

        # GRU-style temporal update
        scale = math.sqrt(1 / hidden_dim)
        self.W_z = rng.normal(0, scale, (hidden_dim * 2, hidden_dim))  # update gate
        self.W_r = rng.normal(0, scale, (hidden_dim * 2, hidden_dim))  # reset gate
        self.W_h = rng.normal(0, scale, (hidden_dim * 2, hidden_dim))  # candidate

        # Output
        self.W_out = rng.normal(0, scale, (hidden_dim, out_dim))
        self.b_out = np.zeros(out_dim)

        self._hidden: Optional[np.ndarray] = None

    def step(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Process one timestep.
        x: (N, in_dim) node features
        adj: (N, N) adjacency
        Returns: (N, out_dim) node embeddings
        """
        N = x.shape[0]

        # GNN layers
        h1 = self.gc1.forward(x, adj)
        h_gnn = self.gc2.forward(h1, adj)

        # Initialize hidden if needed
        if self._hidden is None or self._hidden.shape[0] != N:
            self._hidden = np.zeros((N, self.hidden_dim))

        # GRU temporal update
        combined = np.concatenate([h_gnn, self._hidden], axis=-1)
        z = _sigmoid(combined @ self.W_z)   # update gate
        r = _sigmoid(combined @ self.W_r)   # reset gate
        combined_reset = np.concatenate([h_gnn, r * self._hidden], axis=-1)
        h_cand = np.tanh(combined_reset @ self.W_h)
        self._hidden = (1 - z) * self._hidden + z * h_cand

        # Output
        out = _relu(self._hidden @ self.W_out + self.b_out)
        return out

    def reset(self) -> None:
        self._hidden = None


# ── Portfolio GNN ─────────────────────────────────────────────────────────────

class PortfolioGNN:
    """
    GNN-based portfolio weight learner.
    Learns to map node embeddings to portfolio weights.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.gcn1 = GraphConvLayer(feature_dim, hidden_dim, seed)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim // 2, seed + 1)

        # Weight predictor
        h = hidden_dim // 2
        self.W1 = rng.normal(0, math.sqrt(2 / h), (h, h))
        self.W2 = rng.normal(0, math.sqrt(2 / h), (h, 1))
        self.b1 = np.zeros(h)
        self.b2 = np.zeros(1)

        self.lr = 0.001
        self.params = [self.gcn1.W, self.gcn2.W, self.W1, self.W2]

    def forward(self, features: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        features: (N, feature_dim)
        Returns: (N,) portfolio weights (softmax normalized)
        """
        h1 = self.gcn1.forward(features, adj)
        h2 = self.gcn2.forward(h1, adj)
        h3 = _relu(h2 @ self.W1 + self.b1)
        logits = h3 @ self.W2 + self.b2  # (N, 1)
        weights = _softmax(logits.squeeze(-1))
        return weights

    def sharpe_loss(
        self,
        weights: np.ndarray,
        returns: np.ndarray,  # (T, N)
    ) -> float:
        """Negative Sharpe ratio as loss."""
        port_returns = returns @ weights
        mu = float(port_returns.mean())
        sigma = float(port_returns.std() + 1e-10)
        return -float(mu / sigma * math.sqrt(252))

    def train_step(
        self,
        features: np.ndarray,
        adj: np.ndarray,
        returns: np.ndarray,
        eps: float = 1e-4,
    ) -> float:
        """Numerical gradient step."""
        weights = self.forward(features, adj)
        base_loss = self.sharpe_loss(weights, returns)

        # Gradient via finite differences on output weights
        grad = np.zeros_like(self.W2)
        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                self.W2[i, j] += eps
                w_plus = self.forward(features, adj)
                loss_plus = self.sharpe_loss(w_plus, returns)
                self.W2[i, j] -= eps
                grad[i, j] = (loss_plus - base_loss) / eps

        self.W2 -= self.lr * grad
        return float(base_loss)


# ── Link Prediction ───────────────────────────────────────────────────────────

def link_prediction_score(
    embeddings: np.ndarray,   # (N, d)
    method: str = "dot",
) -> np.ndarray:
    """
    Predict probability of edges from node embeddings.
    Returns (N, N) score matrix.
    """
    N = embeddings.shape[0]

    if method == "dot":
        scores = embeddings @ embeddings.T
    elif method == "l2":
        diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
        scores = -np.sum(diff**2, axis=-1)
    elif method == "hadamard":
        scores = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                scores[i, j] = float(np.sum(embeddings[i] * embeddings[j]))
    else:
        scores = embeddings @ embeddings.T

    # Normalize to [0, 1]
    scores = _sigmoid(scores)
    np.fill_diagonal(scores, 0)
    return scores


def forecast_correlation_change(
    current_adj: np.ndarray,
    embeddings_t: np.ndarray,
    embeddings_t1: np.ndarray,
) -> dict:
    """
    Forecast which correlations will increase/decrease using embedding drift.
    """
    N = current_adj.shape[0]
    scores_t = link_prediction_score(embeddings_t)
    scores_t1 = link_prediction_score(embeddings_t1)

    delta = scores_t1 - scores_t

    # New edges (correlation likely to increase)
    new_edges = (delta > 0.2) & (current_adj == 0)
    # Disappearing edges (correlation likely to decrease)
    disappearing = (delta < -0.2) & (current_adj > 0)

    return {
        "score_delta": delta,
        "new_edge_count": int(new_edges.sum() / 2),
        "disappearing_edge_count": int(disappearing.sum() / 2),
        "new_edges": list(zip(*np.where(new_edges & (np.triu(np.ones((N, N)), 1) > 0)))),
        "disappearing_edges": list(zip(*np.where(disappearing & (np.triu(np.ones((N, N)), 1) > 0)))),
    }


# ── Graph Readout / Pooling ───────────────────────────────────────────────────

def graph_readout(
    node_embeddings: np.ndarray,   # (N, d)
    method: str = "attention",
    adj: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Aggregate node embeddings to graph-level representation.
    methods: 'mean', 'max', 'sum', 'attention', 'hierarchical'
    """
    if method == "mean":
        return float(node_embeddings.mean(axis=0))

    elif method == "max":
        return node_embeddings.max(axis=0)

    elif method == "sum":
        return node_embeddings.sum(axis=0)

    elif method == "attention":
        # Self-attention pooling
        scores = node_embeddings @ node_embeddings.mean(axis=0)
        attn = _softmax(scores)
        return (attn[:, np.newaxis] * node_embeddings).sum(axis=0)

    elif method == "hierarchical" and adj is not None:
        # Aggregate high-degree nodes more heavily
        degree = adj.sum(axis=1)
        degree_norm = degree / (degree.sum() + 1e-10)
        return (degree_norm[:, np.newaxis] * node_embeddings).sum(axis=0)

    return node_embeddings.mean(axis=0)


# ── Full GNN Pipeline ─────────────────────────────────────────────────────────

class FinancialGNNPipeline:
    """
    End-to-end GNN pipeline for financial graphs.
    Inputs: return matrix + optional sector labels.
    Outputs: node embeddings, portfolio weights, regime signal.
    """

    def __init__(
        self,
        n_assets: int,
        feature_dim: int,
        hidden_dim: int = 32,
        corr_threshold: float = 0.3,
        use_temporal: bool = True,
    ):
        self.n_assets = n_assets
        self.corr_threshold = corr_threshold
        self.gcn1 = GraphConvLayer(feature_dim, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gat = GATLayer(hidden_dim, hidden_dim, n_heads=4)
        self.temporal = TemporalGNN(feature_dim, hidden_dim, hidden_dim) if use_temporal else None
        self.portfolio_gnn = PortfolioGNN(hidden_dim, hidden_dim)

    def build_graph(self, returns: np.ndarray) -> dict:
        """Build correlation graph from recent returns."""
        return correlation_graph(returns, self.corr_threshold)

    def embed(self, features: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Get node embeddings."""
        h1 = self.gcn1.forward(features, adj)
        h2 = self.gat.forward(h1, adj)
        h3 = self.gcn2.forward(h2, adj)
        return _layer_norm(h3)

    def get_portfolio_weights(
        self,
        features: np.ndarray,
        adj: np.ndarray,
    ) -> np.ndarray:
        embeddings = self.embed(features, adj)
        return self.portfolio_gnn.forward(embeddings, adj)

    def market_regime_signal(
        self,
        embeddings: np.ndarray,
        adj: np.ndarray,
    ) -> dict:
        """
        Derive market regime from graph structure.
        Dense, highly connected graph = correlated market = risk-off.
        """
        n_edges = int((adj > 0).sum() / 2)
        max_edges = self.n_assets * (self.n_assets - 1) / 2
        density = float(n_edges / max_edges) if max_edges > 0 else 0.0

        # Average embedding dispersion = diversity of market
        emb_std = float(embeddings.std(axis=0).mean())
        avg_corr = float(np.corrcoef(embeddings.T).mean())

        # High density + low dispersion = herding/crisis
        regime_score = float(1 - density * (1 - emb_std))

        if density > 0.6:
            regime = "crisis_herding"
        elif density > 0.4:
            regime = "correlated"
        elif density < 0.2:
            regime = "uncorrelated_dispersed"
        else:
            regime = "normal"

        return {
            "graph_density": density,
            "embedding_dispersion": emb_std,
            "regime": regime,
            "regime_score": float(np.clip(regime_score, 0, 1)),
            "n_edges": n_edges,
            "avg_embedding_corr": float(avg_corr),
        }

    def run(
        self,
        returns: np.ndarray,      # (T, N) historical returns
        current_features: np.ndarray,  # (N, feature_dim) current node features
    ) -> dict:
        """Full pipeline run."""
        graph = self.build_graph(returns)
        adj = graph["adj_weighted"]
        embeddings = self.embed(current_features, adj)
        weights = self.get_portfolio_weights(current_features, adj)
        regime = self.market_regime_signal(embeddings, adj)
        graph_repr = graph_readout(embeddings, method="attention")

        return {
            "embeddings": embeddings,
            "portfolio_weights": weights,
            "regime": regime,
            "graph": graph,
            "graph_repr": graph_repr,
            "top_assets": np.argsort(weights)[::-1][:5].tolist(),
        }
