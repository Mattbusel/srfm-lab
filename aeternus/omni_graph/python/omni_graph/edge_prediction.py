"""
edge_prediction.py — Edge weight prediction and anomaly detection.

Models:
    GraphDiffusion      Predict next-step edge weights via learned diffusion operator.
    LinkPredictor       MLP on node pair embeddings for binary edge existence prediction.
    WormholeDetector    Anomaly score for sudden high-weight edges (isolation forest).
    RicciFlowGNN        Uses Ollivier-Ricci curvature as edge features in GNN message passing.

Evaluation:
    evaluate_link_prediction   AUC-ROC, precision@k, average precision.
    evaluate_edge_weights      MSE, MAE, correlation metrics.
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score


# ── GraphDiffusion ────────────────────────────────────────────────────────────

class DiffusionOperator(nn.Module):
    """Learnable graph diffusion operator.

    Parameterizes the diffusion as a polynomial of the normalized adjacency:
        D(A) = sum_k alpha_k * T^k  where T = D^{-1/2} A D^{-1/2}
    The alpha_k coefficients are learned via a small MLP conditioned on
    the current graph state.

    Args:
        n_channels:   Number of channels to diffuse.
        poly_degree:  Degree of polynomial diffusion.
        hidden_dim:   Hidden dimension for coefficient predictor.
    """

    def __init__(self, n_channels: int, poly_degree: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.poly_degree = poly_degree

        # MLP to predict polynomial coefficients from graph summary statistics
        self.coeff_net = nn.Sequential(
            nn.Linear(n_channels + 4, hidden_dim),  # +4 for graph stats
            nn.ReLU(),
            nn.Linear(hidden_dim, poly_degree + 1),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply learned diffusion to node features.

        Args:
            x:           (N, n_channels) node features.
            edge_index:  (2, E) edge connectivity.
            edge_weight: (E,) edge weights.
        Returns:
            (N, n_channels) diffused features.
        """
        n = x.size(0)
        # Build transition matrix (sparse mul via scatter)
        row, col = edge_index

        # Degree normalization
        deg = torch.zeros(n, device=x.device)
        if edge_weight is not None:
            deg.scatter_add_(0, col, edge_weight)
        else:
            deg.scatter_add_(0, col, torch.ones(edge_index.size(1), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e6)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        if edge_weight is not None:
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Graph summary for coefficient prediction
        x_mean = x.mean(0)                                   # (n_channels,)
        graph_stats = torch.tensor([
            x.mean().item(), x.std().item(),
            float(n), float(edge_index.size(1))
        ], device=x.device)
        coeff_input = torch.cat([x_mean, graph_stats], dim=0)  # (n_channels+4,)
        coeffs = self.coeff_net(coeff_input.unsqueeze(0)).squeeze(0)  # (poly_degree+1,)

        # Apply polynomial diffusion: sum_k alpha_k * T^k * x
        result = coeffs[0] * x
        tx = x.clone()
        for k in range(1, self.poly_degree + 1):
            # tx = T * tx (sparse matrix-vector)
            new_tx = torch.zeros_like(tx)
            for c in range(self.n_channels):
                msg = norm * tx[col, c]
                new_tx[:, c].scatter_add_(0, row, msg)
            tx = new_tx
            result = result + coeffs[k] * tx

        return result


class GraphDiffusion(nn.Module):
    """Predict next-step edge weights via learned graph diffusion.

    Architecture:
        1. Encode node features with diffusion operator.
        2. For each edge (i, j), predict weight from h_i and h_j.
        3. Also predicts the full (dense) weight matrix for graph generation.

    Args:
        node_features:  Input node feature dimension.
        hidden_dim:     Hidden dimension.
        poly_degree:    Diffusion polynomial degree.
        n_layers:       Number of diffusion layers.
    """

    def __init__(
        self,
        node_features: int = 8,
        hidden_dim: int = 64,
        poly_degree: int = 5,
        n_layers: int = 3,
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(node_features, hidden_dim)

        self.diffusion_layers = nn.ModuleList([
            DiffusionOperator(hidden_dim, poly_degree)
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:           (N, node_features) node features.
            edge_index:  (2, E) edge connectivity.
            edge_weight: (E,) current edge weights.
        Returns:
            edge_preds:  (E,) predicted next-step edge weights.
            node_emb:    (N, hidden_dim) node embeddings.
        """
        h = self.input_proj(x)

        for diff, ln in zip(self.diffusion_layers, self.layer_norms):
            h_new = diff(h, edge_index, edge_weight)
            h = ln(h + h_new)  # residual connection

        # Predict edge weights
        row, col = edge_index
        edge_feat = torch.cat([h[row], h[col]], dim=-1)
        edge_preds = self.edge_predictor(edge_feat).squeeze(-1)

        return edge_preds, h

    def predict_all_edges(self, h: Tensor) -> Tensor:
        """Predict weights for ALL possible edges (dense N x N matrix).

        Args:
            h: (N, hidden_dim) node embeddings.
        Returns:
            (N, N) predicted weight matrix.
        """
        n = h.size(0)
        i_idx = torch.arange(n, device=h.device).unsqueeze(1).expand(n, n).reshape(-1)
        j_idx = torch.arange(n, device=h.device).unsqueeze(0).expand(n, n).reshape(-1)
        pairs = torch.cat([h[i_idx], h[j_idx]], dim=-1)
        weights = self.edge_predictor(pairs).reshape(n, n)
        return weights


# ── LinkPredictor ─────────────────────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """MLP-based link predictor for binary edge existence prediction.

    Uses node pair embeddings (and optionally edge history features) to predict
    whether an edge will exist in the next time step.

    Args:
        node_dim:        Node embedding dimension.
        hidden_dim:      Hidden dimension.
        history_features: Number of historical edge weight features (e.g., mean, std).
        dropout:         Dropout rate.
    """

    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        history_features: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.history_features = history_features

        # Three scoring functions (combined):
        # 1. Dot product of node embeddings (collaborative filtering style)
        # 2. Concatenation MLP
        # 3. Historical features
        in_dim = node_dim * 2 + history_features

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Hadamard component
        self.hadamard_proj = nn.Linear(node_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        h_i: Tensor,
        h_j: Tensor,
        history: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            h_i:     (E, node_dim) source node embeddings.
            h_j:     (E, node_dim) target node embeddings.
            history: (E, history_features) optional historical features.
        Returns:
            (E,) link existence logits (before sigmoid).
        """
        # Concatenation
        pair_feat = torch.cat([h_i, h_j], dim=-1)

        if history is not None and self.history_features > 0:
            pair_feat = torch.cat([pair_feat, history], dim=-1)
        elif self.history_features > 0:
            pad = torch.zeros(h_i.size(0), self.history_features, device=h_i.device)
            pair_feat = torch.cat([pair_feat, pad], dim=-1)

        mlp_score = self.mlp(pair_feat).squeeze(-1)

        # Hadamard
        hadamard = self.hadamard_proj(h_i * h_j).squeeze(-1)

        # Dot product
        dot = (h_i * h_j).sum(dim=-1)

        return mlp_score + 0.1 * hadamard + 0.1 * dot

    def predict_proba(self, h_i: Tensor, h_j: Tensor, history: Optional[Tensor] = None) -> Tensor:
        """Return sigmoid probabilities."""
        logits = self.forward(h_i, h_j, history)
        return torch.sigmoid(logits)


# ── WormholeDetector ──────────────────────────────────────────────────────────

class WormholeDetector:
    """Anomaly score for sudden high-weight edges (wormholes).

    A wormhole is an edge whose weight suddenly spikes far above its
    historical distribution, potentially indicating contagion propagation.

    Uses an Isolation Forest on edge weight time series features to detect
    anomalous edge behavior.

    Args:
        contamination:    Expected fraction of anomalies.
        window_size:      Rolling window for feature extraction.
        n_estimators:     Number of isolation trees.
        z_score_threshold: Z-score threshold for flagging (supplementary to IF).
    """

    def __init__(
        self,
        contamination: float = 0.05,
        window_size: int = 20,
        n_estimators: int = 100,
        z_score_threshold: float = 3.0,
    ):
        self.contamination = contamination
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.isolation_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.is_fitted = False
        # History: edge -> list of weights over time
        self._edge_history: Dict[Tuple[int, int], List[float]] = {}
        self._feature_history: List[np.ndarray] = []  # rows are feature vectors

    def _extract_features(
        self,
        edge: Tuple[int, int],
        current_weight: float,
        graph_density: float,
        n_nodes: int,
    ) -> np.ndarray:
        """Extract features for an edge at the current time step."""
        history = self._edge_history.get(edge, [])

        if len(history) >= 2:
            hist_arr = np.array(history[-self.window_size:])
            mean = hist_arr.mean()
            std = hist_arr.std() + 1e-8
            z_score = (current_weight - mean) / std
            trend = (history[-1] - history[0]) / (len(history) + 1e-8)
            max_hist = hist_arr.max()
            pct_rank = (current_weight > hist_arr).mean()
        else:
            mean = current_weight
            std = 1.0
            z_score = 0.0
            trend = 0.0
            max_hist = current_weight
            pct_rank = 0.5

        return np.array([
            current_weight,         # raw weight
            z_score,                # z-score vs history
            trend,                  # recent trend
            pct_rank,               # percentile rank in history
            current_weight / (max_hist + 1e-8),  # ratio to historical max
            graph_density,          # graph density
            float(n_nodes),         # graph size
            len(history),           # edge age (how long it has existed)
        ])

    def update(
        self,
        edges: List[Tuple[int, int, float]],  # (src, dst, weight)
        n_nodes: int,
    ) -> None:
        """Update edge histories with a new snapshot."""
        n_edges = len(edges)
        graph_density = n_edges / max(n_nodes * (n_nodes - 1), 1)

        for src, dst, w in edges:
            edge = (min(src, dst), max(src, dst))
            self._edge_history.setdefault(edge, []).append(w)

            feat = self._extract_features(edge, w, graph_density, n_nodes)
            self._feature_history.append(feat)

    def fit(self) -> None:
        """Fit the isolation forest on accumulated feature history."""
        if len(self._feature_history) < 10:
            return
        X = np.vstack(self._feature_history)
        self.isolation_forest.fit(X)
        self.is_fitted = True

    def score(
        self,
        edges: List[Tuple[int, int, float]],
        n_nodes: int,
    ) -> Dict[Tuple[int, int], float]:
        """Score edges in the current snapshot for wormhole anomaly.

        Returns:
            Dict mapping (src, dst) -> anomaly score in [0, 1]
            (higher = more anomalous).
        """
        results = {}
        n_edges = len(edges)
        graph_density = n_edges / max(n_nodes * (n_nodes - 1), 1)

        for src, dst, w in edges:
            edge = (min(src, dst), max(src, dst))
            feat = self._extract_features(edge, w, graph_density, n_nodes)

            # Z-score based score (always available)
            z = feat[1]  # z_score feature
            z_score_anom = 1.0 / (1.0 + math.exp(-0.5 * (abs(z) - self.z_score_threshold)))

            # Isolation forest score (if fitted)
            if self.is_fitted:
                # sklearn returns -1 (outlier) or 1 (inlier), score in (-inf, 0]
                if_score = self.isolation_forest.score_samples(feat.reshape(1, -1))[0]
                # Convert to [0, 1]: more negative = more anomalous
                if_anom = 1.0 / (1.0 + math.exp(5.0 * if_score + 2.0))
                anomaly_score = 0.5 * z_score_anom + 0.5 * if_anom
            else:
                anomaly_score = z_score_anom

            results[edge] = anomaly_score

        return results

    def detect_wormholes(
        self,
        edges: List[Tuple[int, int, float]],
        n_nodes: int,
        threshold: float = 0.7,
    ) -> List[Tuple[int, int, float, float]]:
        """Return edges classified as wormholes.

        Returns:
            List of (src, dst, weight, anomaly_score) for wormhole edges.
        """
        scores = self.score(edges, n_nodes)
        wormholes = []
        for (src, dst, w) in edges:
            edge = (min(src, dst), max(src, dst))
            score = scores.get(edge, 0.0)
            if score >= threshold:
                wormholes.append((src, dst, w, score))
        # Sort by anomaly score descending
        wormholes.sort(key=lambda x: x[3], reverse=True)
        return wormholes

    def rolling_anomaly_score(self, window: int = 5) -> List[float]:
        """Return rolling mean anomaly score over the edge history."""
        if not self._feature_history:
            return []
        all_feats = np.vstack(self._feature_history)
        z_scores = np.abs(all_feats[:, 1])  # z-score column
        scores = 1.0 / (1.0 + np.exp(-0.5 * (z_scores - self.z_score_threshold)))

        result = []
        for i in range(len(scores)):
            start = max(0, i - window + 1)
            result.append(float(scores[start:i+1].mean()))
        return result


# ── RicciFlowGNN ──────────────────────────────────────────────────────────────

class RicciMessagePassing(MessagePassing):
    """Message passing that uses Ricci curvature as an edge gate.

    Edges with higher (more positive) curvature are within-community
    and carry stronger messages. Edges with negative curvature are
    between-community (bridges) and are downweighted.

    Args:
        in_channels:   Input node feature dimension.
        out_channels:  Output node feature dimension.
        hidden_dim:    Hidden dimension.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int = 64):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Gate network: takes (h_i, h_j, curvature) -> gate in [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(2 * in_channels + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        ricci_curvature: Tensor,
    ) -> Tensor:
        """
        Args:
            x:               (N, in_channels) node features.
            edge_index:      (2, E) edge connectivity.
            ricci_curvature: (E,) Ollivier-Ricci curvature per edge.
        Returns:
            (N, out_channels) updated node features.
        """
        return self.propagate(edge_index, x=x, ricci=ricci_curvature)

    def message(self, x_i: Tensor, x_j: Tensor, ricci: Tensor) -> Tensor:
        # ricci: (E,) -> (E, 1)
        ricci_feat = ricci.unsqueeze(-1)

        # Gate: decides how much to weight this edge
        gate_input = torch.cat([x_i, x_j, ricci_feat], dim=-1)
        gate = self.gate_net(gate_input)  # (E, 1)

        # Message from neighbor j to i
        msg_input = torch.cat([x_j, ricci_feat], dim=-1)
        msg = self.msg_mlp(msg_input)  # (E, out_channels)

        return gate * msg

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        # Residual connection with update MLP
        combined = torch.cat([x[:aggr_out.size(0)], aggr_out], dim=-1)
        return self.update_mlp(combined)


class RicciFlowGNN(nn.Module):
    """GNN that uses Ricci curvature as edge features in message passing.

    Combines:
    - Node feature encoding
    - RicciMessagePassing layers with curvature-gated messages
    - Ricci flow: iteratively update curvatures during message passing
    - Output heads for edge weight and link prediction

    This model is particularly effective for detecting regime transitions
    because curvature collapse (all edges going negative) is a reliable
    leading indicator of financial crises.

    Args:
        node_features:  Input node feature dimension.
        hidden_dim:     Hidden dimension.
        n_layers:       Number of Ricci message passing layers.
        n_flow_steps:   Number of Ricci flow iterations per forward pass.
        dropout:        Dropout rate.
    """

    def __init__(
        self,
        node_features: int = 8,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_flow_steps: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.n_flow_steps = n_flow_steps

        self.input_proj = nn.Linear(node_features, hidden_dim)

        self.ricci_layers = nn.ModuleList([
            RicciMessagePassing(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        # Curvature updater: predict curvature from node embeddings
        # This simulates the Ricci flow update during inference
        self.curvature_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # curvature in [-1, 1]
        )

        # Output heads
        self.edge_weight_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.node_risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.graph_summary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # mean_ricci, fiedler_proxy, risk_score
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        ricci_curvature: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            x:               (N, node_features) node features.
            edge_index:      (2, E) edge connectivity.
            ricci_curvature: (E,) initial Ricci curvature values (or None for uniform).
            edge_weight:     (E,) edge weights.
        Returns:
            dict with: node_emb, edge_weight_pred, node_risk, graph_summary, final_curvature.
        """
        n = x.size(0)
        e = edge_index.size(1)
        h = self.input_proj(x)

        # Initialize Ricci curvature
        if ricci_curvature is None:
            ricci = torch.zeros(e, device=x.device)
        else:
            ricci = ricci_curvature.float()

        # Ricci flow + message passing
        for layer, ln in zip(self.ricci_layers, self.layer_norms):
            h_new = layer(h, edge_index, ricci)
            h = ln(h + h_new)
            h = self.dropout(h)

            # Update Ricci curvature based on current embeddings (simulated Ricci flow)
            row, col = edge_index
            edge_emb = torch.cat([h[row], h[col]], dim=-1)
            ricci_update = self.curvature_predictor(edge_emb).squeeze(-1)
            # Ricci flow: mix old and new curvature
            ricci = 0.7 * ricci + 0.3 * ricci_update

        # Edge weight prediction
        row, col = edge_index
        edge_feat = torch.cat([h[row], h[col], ricci.unsqueeze(-1)], dim=-1)
        edge_weight_pred = self.edge_weight_head(edge_feat).squeeze(-1)

        # Node risk scores
        node_risk = self.node_risk_head(h).squeeze(-1)

        # Graph-level summary
        graph_emb = h.mean(0, keepdim=True)
        graph_summary = self.graph_summary_head(graph_emb).squeeze(0)

        return {
            "node_emb": h,
            "edge_weight_pred": edge_weight_pred,
            "node_risk": node_risk,
            "graph_summary": graph_summary,
            "final_curvature": ricci,
        }

    def get_crisis_score(self, outputs: Dict[str, Tensor]) -> float:
        """Extract scalar crisis score from model outputs.

        Crisis score = combination of:
        - Mean negative curvature (more negative = higher risk)
        - Max node risk
        - Graph summary risk score
        """
        ricci = outputs["final_curvature"]
        neg_curvature = (-ricci.mean()).item()  # positive when ricci is negative
        max_node_risk = outputs["node_risk"].max().item()
        graph_risk = torch.sigmoid(outputs["graph_summary"][2]).item()

        # Combine
        score = 0.4 * neg_curvature + 0.3 * max_node_risk + 0.3 * graph_risk
        return float(np.clip(score, 0.0, 1.0))


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_link_prediction(
    model: nn.Module,
    node_embeddings: Tensor,
    pos_edges: Tensor,   # (2, E_pos) positive edges
    neg_edges: Tensor,   # (2, E_neg) negative edges
) -> Dict[str, float]:
    """Evaluate link prediction performance.

    Args:
        model:            A LinkPredictor model.
        node_embeddings:  (N, D) node embeddings.
        pos_edges:        Positive edge indices.
        neg_edges:        Negative edge indices.
    Returns:
        dict with auc_roc, avg_precision, precision_at_k metrics.
    """
    model.eval()
    with torch.no_grad():
        # Positive scores
        h_i_pos = node_embeddings[pos_edges[0]]
        h_j_pos = node_embeddings[pos_edges[1]]
        pos_scores = torch.sigmoid(model(h_i_pos, h_j_pos)).cpu().numpy()

        # Negative scores
        h_i_neg = node_embeddings[neg_edges[0]]
        h_j_neg = node_embeddings[neg_edges[1]]
        neg_scores = torch.sigmoid(model(h_i_neg, h_j_neg)).cpu().numpy()

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores))
    ])

    # AUC-ROC
    auc = roc_auc_score(all_labels, all_scores)

    # Average precision
    ap = average_precision_score(all_labels, all_scores)

    # Precision @ k (k = number of positive edges)
    k = len(pos_scores)
    top_k_idx = np.argsort(-all_scores)[:k]
    precision_at_k = all_labels[top_k_idx].mean()

    return {
        "auc_roc": float(auc),
        "avg_precision": float(ap),
        "precision_at_k": float(precision_at_k),
        "n_pos": len(pos_scores),
        "n_neg": len(neg_scores),
    }


def evaluate_edge_weight_prediction(
    true_weights: np.ndarray,
    pred_weights: np.ndarray,
) -> Dict[str, float]:
    """Evaluate edge weight prediction quality."""
    mse = float(np.mean((true_weights - pred_weights) ** 2))
    mae = float(np.mean(np.abs(true_weights - pred_weights)))
    rmse = math.sqrt(mse)

    # Correlation
    if true_weights.std() > 1e-8 and pred_weights.std() > 1e-8:
        corr = float(np.corrcoef(true_weights, pred_weights)[0, 1])
    else:
        corr = 0.0

    # Direction accuracy (did sign of change match?)
    # (Only meaningful if comparing consecutive graphs)
    return {"mse": mse, "mae": mae, "rmse": rmse, "correlation": corr}


# ── Training functions ────────────────────────────────────────────────────────

def train_graph_diffusion(
    model: GraphDiffusion,
    snapshots: List[Data],
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train GraphDiffusion to predict next-step edge weights.

    Pairs consecutive snapshots as (input, target).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {"loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for t in range(len(snapshots) - 1):
            snap_t = snapshots[t].to(device)
            snap_t1 = snapshots[t + 1].to(device)

            optimizer.zero_grad()
            edge_preds, _ = model(
                snap_t.x, snap_t.edge_index,
                snap_t.edge_attr[:, 0] if snap_t.edge_attr is not None else None,
            )

            # Target: edge weights in next snapshot (matched by edge_index)
            if snap_t1.edge_attr is not None:
                # Simple MSE on the same edge positions
                min_e = min(edge_preds.size(0), snap_t1.edge_attr.size(0))
                target = snap_t1.edge_attr[:min_e, 0]
                pred = edge_preds[:min_e]
                loss = F.mse_loss(pred, target)
            else:
                # Without target, minimize reconstruction loss on current weights
                if snap_t.edge_attr is not None:
                    loss = F.mse_loss(edge_preds, snap_t.edge_attr[:, 0])
                else:
                    loss = edge_preds.mean()  # dummy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        history["loss"].append(avg_loss)

        if verbose and epoch % 10 == 0:
            print(f"GraphDiffusion epoch {epoch}/{epochs}  loss={avg_loss:.6f}")

    return history


def train_ricci_gnn(
    model: RicciFlowGNN,
    snapshots: List[Data],
    ricci_per_snap: List[np.ndarray],
    epochs: int = 50,
    lr: float = 3e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train RicciFlowGNN.

    Supervises the model to reproduce the Ricci curvature values
    computed by the Rust engine as edge targets, while also predicting
    next-step edge weights.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {"loss": [], "curvature_loss": [], "weight_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        e_loss, e_curv, e_wt = [], [], []

        for t, (snap, ricci_vals) in enumerate(zip(snapshots, ricci_per_snap)):
            snap = snap.to(device)
            ricci_tensor = torch.tensor(ricci_vals, dtype=torch.float, device=device)

            # Pad/truncate Ricci to match edge count
            n_edges = snap.edge_index.size(1)
            if ricci_tensor.size(0) < n_edges:
                ricci_tensor = F.pad(ricci_tensor, (0, n_edges - ricci_tensor.size(0)))
            else:
                ricci_tensor = ricci_tensor[:n_edges]

            optimizer.zero_grad()
            outputs = model(snap.x, snap.edge_index, ricci_tensor)

            # Curvature reproduction loss
            curv_loss = F.mse_loss(outputs["final_curvature"], ricci_tensor)

            # Edge weight prediction loss (target = current weights)
            wt_loss = torch.tensor(0.0, device=device)
            if snap.edge_attr is not None:
                wt_loss = F.mse_loss(outputs["edge_weight_pred"], snap.edge_attr[:, 0])

            loss = curv_loss + 0.5 * wt_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            e_loss.append(loss.item())
            e_curv.append(curv_loss.item())
            e_wt.append(wt_loss.item())

        scheduler.step()
        history["loss"].append(sum(e_loss) / max(len(e_loss), 1))
        history["curvature_loss"].append(sum(e_curv) / max(len(e_curv), 1))
        history["weight_loss"].append(sum(e_wt) / max(len(e_wt), 1))

        if verbose and epoch % 10 == 0:
            print(f"RicciFlowGNN epoch {epoch}/{epochs}  "
                  f"loss={history['loss'][-1]:.4f}  "
                  f"curv={history['curvature_loss'][-1]:.4f}  "
                  f"wt={history['weight_loss'][-1]:.4f}")

    return history
