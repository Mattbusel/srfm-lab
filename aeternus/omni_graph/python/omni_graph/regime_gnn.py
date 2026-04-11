"""
regime_gnn.py — Graph-based market regime detection and crisis early warning.

Models:
    GraphRegimeDetector        Clusters graph snapshots into k regimes via graph kernel k-means.
    WassersteinGraphKernel     Distance between graphs via node embedding OT.
    RegimeTransitionPredictor  Predicts which regime comes next given current graph state.
    CrisisEarlyWarning         Combines Ricci curvature trend + regime transition probability
                               into a scalar crisis alarm signal.

Integration:
    - Page-Hinkley drift detection for regime change detection.
    - Integrates with Ricci curvature from the Rust engine.
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── Page-Hinkley Drift Detection ──────────────────────────────────────────────

class PageHinkley:
    """Page-Hinkley test for sequential drift detection.

    Detects a persistent increase in the monitored variable.
    Call update() with each new observation; check_alarm() returns True
    when a drift is detected.

    Args:
        delta:     Tolerance threshold (minimum detectable change).
        lambda_:   Detection threshold (larger = fewer false alarms).
        alpha:     Forgetting factor for running mean.
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 1.0):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0
        self._mean = 0.0
        self._ph_values: List[float] = []

    def update(self, value: float) -> bool:
        """Update with a new observation. Returns True if drift detected."""
        self._n += 1
        self._mean = self.alpha * self._mean + (1.0 - self.alpha) * value
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        ph = self._sum - self._min_sum
        self._ph_values.append(ph)
        return ph > self.lambda_

    def check_alarm(self) -> bool:
        """Check if the most recent update triggered an alarm."""
        if not self._ph_values:
            return False
        return self._ph_values[-1] > self.lambda_

    def reset(self):
        """Reset the detector state."""
        self._sum = 0.0
        self._min_sum = 0.0
        self._mean = 0.0
        self._ph_values.clear()

    @property
    def ph_statistic(self) -> float:
        """Current Page-Hinkley statistic."""
        return self._ph_values[-1] if self._ph_values else 0.0

    def get_history(self) -> List[float]:
        return self._ph_values.copy()


# ── Graph Kernel ──────────────────────────────────────────────────────────────

class WassersteinGraphKernel:
    """Compute Wasserstein-1 distance between two graphs via node embedding OT.

    For each pair of graphs, encodes nodes via a simple GNN, then computes the
    optimal transport distance between the resulting node distribution.
    This is used as a graph-level similarity metric for clustering.

    Args:
        embed_dim:     Dimension of node embeddings used for transport.
        n_layers:      GNN depth.
        reg:           Entropic regularization (Sinkhorn; 0 = exact OT via Hungarian).
        max_iter:      Max Sinkhorn iterations (if reg > 0).
    """

    def __init__(
        self,
        embed_dim: int = 32,
        n_layers: int = 2,
        reg: float = 0.1,
        max_iter: int = 100,
    ):
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.reg = reg
        self.max_iter = max_iter

    def embed_graph(self, data: Data) -> np.ndarray:
        """Embed graph nodes using simple spectral features + GCN-lite.

        Returns (N, embed_dim) numpy array.
        """
        n = data.x.size(0)
        x = data.x.float().numpy()

        # Build adjacency matrix (numpy, for small graphs)
        adj = np.zeros((n, n))
        ei = data.edge_index.numpy()
        ew = data.edge_attr[:, 0].numpy() if data.edge_attr is not None else np.ones(ei.shape[1])
        for e_idx in range(ei.shape[1]):
            i, j = ei[0, e_idx], ei[1, e_idx]
            if i < n and j < n:
                adj[i, j] += ew[e_idx]
                adj[j, i] += ew[e_idx]

        # Degree-normalized adjacency
        deg = adj.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        a_norm = adj / deg

        # Simple GCN-lite propagation
        h = x
        for _ in range(self.n_layers):
            h = a_norm @ h
            # Normalize features
            norm = np.linalg.norm(h, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            h = h / norm

        # Project to embed_dim via random features
        np.random.seed(42)
        if h.shape[1] >= self.embed_dim:
            # Random projection
            proj = np.random.randn(h.shape[1], self.embed_dim) / math.sqrt(h.shape[1])
            h = h @ proj
        else:
            # Pad
            pad = np.zeros((n, self.embed_dim - h.shape[1]))
            h = np.concatenate([h, pad], axis=1)

        return h

    def distance(self, g1: Data, g2: Data) -> float:
        """Compute Wasserstein-1 distance between two graphs."""
        emb1 = self.embed_graph(g1)
        emb2 = self.embed_graph(g2)

        # Uniform weights (graph-level distributions)
        n1, n2 = len(emb1), len(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0

        # Ground cost matrix: pairwise L2 distances
        cost = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                cost[i, j] = np.linalg.norm(emb1[i] - emb2[j])

        if self.reg > 0.0:
            return self._sinkhorn(cost, n1, n2)
        else:
            return self._exact_wasserstein(cost, n1, n2)

    def _sinkhorn(self, cost: np.ndarray, n1: int, n2: int) -> float:
        """Entropic-regularized OT via Sinkhorn iterations."""
        a = np.ones(n1) / n1
        b = np.ones(n2) / n2
        K = np.exp(-cost / self.reg)

        u = np.ones(n1) / n1
        for _ in range(self.max_iter):
            v = b / (K.T @ u + 1e-15)
            u = a / (K @ v + 1e-15)

        transport = np.diag(u) @ K @ np.diag(v)
        return float((transport * cost).sum())

    def _exact_wasserstein(self, cost: np.ndarray, n1: int, n2: int) -> float:
        """Exact Wasserstein via Hungarian algorithm (for uniform distributions)."""
        n = max(n1, n2)
        cost_sq = np.zeros((n, n))
        cost_sq[:n1, :n2] = cost
        row_ind, col_ind = linear_sum_assignment(cost_sq)
        return float(cost_sq[row_ind, col_ind].sum()) / max(min(n1, n2), 1)

    def pairwise_distances(self, graphs: List[Data]) -> np.ndarray:
        """Compute pairwise distance matrix between a list of graphs."""
        n = len(graphs)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(graphs[i], graphs[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        return dist_matrix


# ── GraphRegimeDetector ───────────────────────────────────────────────────────

class GraphRegimeDetector:
    """Cluster graph snapshots into k regimes via graph kernel k-means.

    Uses WassersteinGraphKernel as the distance metric and kernel k-means
    as the clustering algorithm. Graph-level features are extracted via
    a lightweight GNN encoder.

    Args:
        n_regimes:   Number of market regimes.
        kernel:      WassersteinGraphKernel instance.
        n_init:      Number of k-means initializations.
        max_iter:    Maximum k-means iterations.
        random_state: Random seed.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        kernel: Optional[WassersteinGraphKernel] = None,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.kernel = kernel or WassersteinGraphKernel()
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.kmeans: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_prototypes: Optional[np.ndarray] = None
        self.feature_history: List[np.ndarray] = []

    def _extract_graph_features(
        self,
        data: Data,
        ricci_curvature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract a fixed-size feature vector from a graph snapshot.

        Features: graph-level statistics, spectral features, optional Ricci.
        """
        n = data.x.size(0)
        e = data.edge_index.size(1)

        # Edge weight statistics
        if data.edge_attr is not None and data.edge_attr.size(0) > 0:
            ew = data.edge_attr[:, 0].float().numpy()
            ew_mean = float(ew.mean())
            ew_std = float(ew.std())
            ew_max = float(ew.max())
            ew_min = float(ew.min())
            ew_skew = float(np.mean(((ew - ew_mean) / (ew_std + 1e-8)) ** 3)) if ew_std > 0 else 0.0
        else:
            ew_mean = ew_std = ew_max = ew_min = ew_skew = 0.0

        # Ricci curvature statistics
        if ricci_curvature is not None and len(ricci_curvature) > 0:
            rc_mean = float(np.mean(ricci_curvature))
            rc_std = float(np.std(ricci_curvature))
            rc_min = float(np.min(ricci_curvature))
            rc_max = float(np.max(ricci_curvature))
            frac_negative = float((ricci_curvature < 0).mean())
        else:
            rc_mean = rc_std = rc_min = rc_max = frac_negative = 0.0

        # Structural features
        density = e / max(n * (n - 1), 1)

        # Node feature statistics
        x_np = data.x.float().numpy()
        x_mean = float(x_np.mean())
        x_std = float(x_np.std())

        features = np.array([
            n, e, density,
            ew_mean, ew_std, ew_max, ew_min, ew_skew,
            rc_mean, rc_std, rc_min, rc_max, frac_negative,
            x_mean, x_std,
        ])
        return features

    def fit(
        self,
        graphs: List[Data],
        ricci_curvatures: Optional[List[np.ndarray]] = None,
    ) -> "GraphRegimeDetector":
        """Fit regime detector on a list of graph snapshots.

        Args:
            graphs:           List of PyG Data objects.
            ricci_curvatures: Optional list of Ricci curvature arrays (same length).
        """
        features = []
        for i, g in enumerate(graphs):
            rc = ricci_curvatures[i] if ricci_curvatures is not None else None
            feat = self._extract_graph_features(g, rc)
            features.append(feat)

        X = np.vstack(features)
        self.feature_history = features

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # K-means clustering
        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.kmeans.fit(X_scaled)
        self.regime_prototypes = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self.is_fitted = True
        return self

    def predict(
        self,
        graph: Data,
        ricci_curvature: Optional[np.ndarray] = None,
    ) -> int:
        """Predict regime for a single graph snapshot."""
        if not self.is_fitted:
            raise RuntimeError("GraphRegimeDetector is not fitted. Call fit() first.")
        feat = self._extract_graph_features(graph, ricci_curvature)
        X = self.scaler.transform(feat.reshape(1, -1))
        return int(self.kmeans.predict(X)[0])

    def predict_proba(
        self,
        graph: Data,
        ricci_curvature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict soft regime probabilities via distance to cluster centers."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        feat = self._extract_graph_features(graph, ricci_curvature)
        X = self.scaler.transform(feat.reshape(1, -1))
        centers = self.kmeans.cluster_centers_

        # Softmax over negative distances
        dists = np.linalg.norm(X - centers, axis=1)
        scores = np.exp(-dists / (dists.mean() + 1e-8))
        return scores / scores.sum()

    def get_regime_name(self, regime_id: int) -> str:
        """Return a human-readable name for a regime based on its features."""
        if self.regime_prototypes is None:
            return f"Regime_{regime_id}"
        proto = self.regime_prototypes[regime_id]
        # proto[8] = rc_mean (index in feature vector)
        rc_mean = proto[8] if len(proto) > 8 else 0.0
        ew_mean = proto[3] if len(proto) > 3 else 0.0
        density = proto[2] if len(proto) > 2 else 0.0

        if rc_mean < -0.3:
            return f"Crisis/Stress_{regime_id}"
        elif rc_mean > 0.3 and density > 0.4:
            return f"Bubble/Exuberance_{regime_id}"
        elif ew_mean < 0.3 and density < 0.2:
            return f"Bear/Fragmented_{regime_id}"
        else:
            return f"Normal/Bull_{regime_id}"


# ── RegimeTransitionPredictor ─────────────────────────────────────────────────

class RegimeTransitionPredictor(nn.Module):
    """Predicts which regime comes next given current graph state and regime history.

    Uses a GRU to model regime transition dynamics and outputs a probability
    distribution over next-step regimes.

    Args:
        n_regimes:   Number of regimes.
        feature_dim: Dimension of graph feature vector.
        hidden_dim:  GRU hidden dimension.
        seq_len:     Length of regime history to consider.
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        feature_dim: int = 15,
        hidden_dim: int = 64,
        seq_len: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Combine graph features + one-hot regime embedding
        self.regime_emb = nn.Embedding(n_regimes, 16)
        in_dim = feature_dim + 16

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_regimes),
        )

        # Transition probability head (directly models the transition matrix)
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim + n_regimes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.regime_emb.weight, std=0.1)

    def forward(
        self,
        graph_features: Tensor,  # (B, T, feature_dim)
        regime_history: Tensor,  # (B, T) regime labels
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            graph_features: (B, T, feature_dim) graph feature sequences.
            regime_history: (B, T) regime label sequences.
            hidden:         Optional GRU initial hidden state.
        Returns:
            next_regime_logits: (B, n_regimes) logits for next-step regime.
            transition_probs:   (B, n_regimes) transition probability distribution.
            hidden_out:         Updated GRU hidden state.
        """
        regime_emb = self.regime_emb(regime_history)  # (B, T, 16)
        x = torch.cat([graph_features, regime_emb], dim=-1)  # (B, T, feature_dim+16)

        out, hidden_out = self.gru(x, hidden)  # (B, T, hidden_dim)
        last_out = out[:, -1, :]  # (B, hidden_dim)

        next_logits = self.output_head(last_out)  # (B, n_regimes)

        # Transition: conditioned on current regime distribution
        current_probs = F.softmax(next_logits, dim=-1)
        transition_input = torch.cat([last_out, current_probs], dim=-1)
        transition_probs = F.softmax(self.transition_head(transition_input), dim=-1)

        return next_logits, transition_probs, hidden_out

    def predict_sequence(
        self,
        graph_features: Tensor,
        regime_history: Tensor,
        n_steps: int = 5,
    ) -> List[Tensor]:
        """Autoregressively predict `n_steps` future regimes."""
        self.eval()
        with torch.no_grad():
            predictions = []
            feats = graph_features.clone()
            regs = regime_history.clone()
            hidden = None

            for _ in range(n_steps):
                logits, probs, hidden = self.forward(feats, regs, hidden)
                next_regime = logits.argmax(dim=-1)  # (B,)
                predictions.append(probs)

                # Update sequences (roll forward)
                feats = torch.cat([feats[:, 1:, :], feats[:, -1:, :]], dim=1)
                regs = torch.cat([regs[:, 1:], next_regime.unsqueeze(1)], dim=1)

        return predictions

    def compute_transition_matrix(
        self,
        regime_sequence: List[int],
    ) -> np.ndarray:
        """Estimate empirical regime transition matrix from observed sequence."""
        k = self.n_regimes
        T = np.zeros((k, k))
        for i in range(len(regime_sequence) - 1):
            from_r = regime_sequence[i]
            to_r = regime_sequence[i + 1]
            T[from_r, to_r] += 1

        # Normalize rows
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return T / row_sums


# ── CrisisEarlyWarning ────────────────────────────────────────────────────────

class CrisisEarlyWarning:
    """Combined crisis early warning system.

    Combines:
    1. Ollivier-Ricci curvature trend (from Rust engine)
    2. Regime transition probability to crisis states
    3. Page-Hinkley drift detection on the combined signal
    4. Wormhole detection frequency

    Produces a scalar alarm signal in [0, 1] where values > 0.7 indicate
    high probability of imminent market crisis.

    Args:
        n_regimes:          Number of regimes.
        crisis_regimes:     Set of regime IDs classified as "crisis".
        ricci_weight:       Weight of Ricci curvature component.
        transition_weight:  Weight of regime transition component.
        wormhole_weight:    Weight of wormhole anomaly component.
        ph_delta:           Page-Hinkley delta parameter.
        ph_lambda:          Page-Hinkley lambda threshold.
        ema_alpha:          EMA smoothing for the alarm signal.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        crisis_regimes: Optional[List[int]] = None,
        ricci_weight: float = 0.4,
        transition_weight: float = 0.35,
        wormhole_weight: float = 0.25,
        ph_delta: float = 0.005,
        ph_lambda: float = 30.0,
        ema_alpha: float = 0.2,
    ):
        self.n_regimes = n_regimes
        self.crisis_regimes = set(crisis_regimes or [0])  # default: regime 0 is crisis
        self.ricci_weight = ricci_weight
        self.transition_weight = transition_weight
        self.wormhole_weight = wormhole_weight
        self.ema_alpha = ema_alpha

        # Internal state
        self.ph_detector = PageHinkley(delta=ph_delta, lambda_=ph_lambda)
        self.alarm_history: List[float] = []
        self.ema_alarm: float = 0.0
        self.drift_events: List[int] = []  # timestep indices where drift detected

        # Components
        self.ricci_history: List[float] = []
        self.regime_history: List[int] = []
        self.wormhole_history: List[float] = []
        self.transition_probs_history: List[np.ndarray] = []

    def update(
        self,
        mean_ricci: float,
        current_regime: int,
        transition_probs: Optional[np.ndarray] = None,
        wormhole_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Update alarm with new observations.

        Args:
            mean_ricci:       Mean Ollivier-Ricci curvature from current graph.
            current_regime:   Predicted current regime index.
            transition_probs: (n_regimes,) probability distribution over next regime.
            wormhole_score:   Wormhole anomaly score in [0, 1].
        Returns:
            dict with alarm_score, ph_alarm, components.
        """
        self.ricci_history.append(mean_ricci)
        self.regime_history.append(current_regime)
        self.wormhole_history.append(wormhole_score)

        # 1. Ricci component: convert to [0, 1] where negative ricci = high risk
        # Typical range: [-0.5, 0.5]. Map so that -0.5 -> 1.0, 0.5 -> 0.0.
        ricci_component = 1.0 / (1.0 + math.exp(5.0 * mean_ricci))

        # Add trend component: is Ricci decreasing rapidly?
        if len(self.ricci_history) >= 5:
            recent = self.ricci_history[-5:]
            # Linear trend slope
            x = np.arange(len(recent))
            slope = float(np.polyfit(x, recent, 1)[0])
            # Negative slope (decreasing curvature) = higher risk
            trend_component = 1.0 / (1.0 + math.exp(10.0 * slope))
        else:
            trend_component = 0.5

        ricci_signal = 0.6 * ricci_component + 0.4 * trend_component

        # 2. Transition component: probability of transitioning to a crisis regime
        if transition_probs is not None:
            self.transition_probs_history.append(transition_probs)
            crisis_prob = sum(transition_probs[r] for r in self.crisis_regimes
                              if r < len(transition_probs))
        elif current_regime in self.crisis_regimes:
            crisis_prob = 1.0
        else:
            crisis_prob = 0.0
        transition_signal = float(np.clip(crisis_prob, 0.0, 1.0))

        # 3. Wormhole component
        wormhole_signal = float(np.clip(wormhole_score, 0.0, 1.0))

        # Combined alarm score
        raw_alarm = (
            self.ricci_weight * ricci_signal
            + self.transition_weight * transition_signal
            + self.wormhole_weight * wormhole_signal
        )
        raw_alarm = float(np.clip(raw_alarm, 0.0, 1.0))

        # EMA smoothing
        if not self.alarm_history:
            self.ema_alarm = raw_alarm
        else:
            self.ema_alarm = self.ema_alpha * raw_alarm + (1.0 - self.ema_alpha) * self.ema_alarm

        self.alarm_history.append(self.ema_alarm)

        # Page-Hinkley drift detection on the alarm signal
        ph_alarm = self.ph_detector.update(self.ema_alarm)
        if ph_alarm:
            self.drift_events.append(len(self.alarm_history) - 1)

        return {
            "alarm_score": self.ema_alarm,
            "raw_alarm": raw_alarm,
            "ph_alarm": ph_alarm,
            "ph_statistic": self.ph_detector.ph_statistic,
            "components": {
                "ricci": ricci_signal,
                "ricci_raw": ricci_component,
                "trend": trend_component,
                "transition": transition_signal,
                "wormhole": wormhole_signal,
            },
        }

    def is_crisis(self, threshold: float = 0.65) -> bool:
        """Return True if current alarm score exceeds crisis threshold."""
        return self.ema_alarm >= threshold

    def time_to_crisis(
        self,
        threshold: float = 0.65,
        forecast_horizon: int = 10,
    ) -> Optional[int]:
        """Estimate steps until alarm crosses threshold via linear extrapolation.

        Returns None if no crossing expected within forecast_horizon steps.
        """
        if len(self.alarm_history) < 3:
            return None

        recent = self.alarm_history[-min(5, len(self.alarm_history)):]
        x = np.arange(len(recent))
        if recent[-1] >= threshold:
            return 0
        slope = float(np.polyfit(x, recent, 1)[0])
        if slope <= 0:
            return None

        steps_needed = int(math.ceil((threshold - recent[-1]) / slope))
        if steps_needed > forecast_horizon:
            return None
        return max(1, steps_needed)

    def get_report(self) -> Dict[str, Any]:
        """Generate a summary report of the warning system state."""
        n = len(self.alarm_history)
        if n == 0:
            return {"status": "no_data"}

        current_alarm = self.alarm_history[-1]
        peak_alarm = max(self.alarm_history)
        peak_idx = self.alarm_history.index(peak_alarm)

        # Current Ricci trend
        ricci_trend = "stable"
        if len(self.ricci_history) >= 5:
            recent_rc = self.ricci_history[-5:]
            slope = float(np.polyfit(np.arange(5), recent_rc, 1)[0])
            if slope < -0.02:
                ricci_trend = "declining_fast"
            elif slope < 0:
                ricci_trend = "declining"
            elif slope > 0.02:
                ricci_trend = "improving"

        current_regime = self.regime_history[-1] if self.regime_history else -1
        in_crisis_regime = current_regime in self.crisis_regimes

        status = "high_alert" if current_alarm > 0.7 else \
                 "elevated" if current_alarm > 0.5 else \
                 "normal"

        return {
            "status": status,
            "current_alarm": current_alarm,
            "ema_alarm": self.ema_alarm,
            "peak_alarm": peak_alarm,
            "peak_at_step": peak_idx,
            "n_drift_events": len(self.drift_events),
            "drift_at_steps": self.drift_events.copy(),
            "current_regime": current_regime,
            "in_crisis_regime": in_crisis_regime,
            "ricci_trend": ricci_trend,
            "current_ricci": self.ricci_history[-1] if self.ricci_history else None,
            "ph_statistic": self.ph_detector.ph_statistic,
            "n_history": n,
        }


# ── Training regime transition predictor ─────────────────────────────────────

def train_regime_transition_predictor(
    model: RegimeTransitionPredictor,
    graph_feature_sequences: List[np.ndarray],
    regime_sequences: List[List[int]],
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train regime transition predictor.

    Args:
        model:                   RegimeTransitionPredictor instance.
        graph_feature_sequences: List of (T, feature_dim) numpy arrays.
        regime_sequences:        List of (T,) integer regime label sequences.
        epochs:                  Training epochs.
        lr:                      Learning rate.
        device:                  PyTorch device.
        verbose:                 Print progress.
    Returns:
        Training history.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    seq_len = model.seq_len
    history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        e_loss, e_correct, e_total = [], 0, 0

        for feat_seq, reg_seq in zip(graph_feature_sequences, regime_sequences):
            T = len(reg_seq)
            if T < seq_len + 1:
                continue

            # Create sliding window samples
            for start in range(T - seq_len):
                feats = torch.tensor(
                    feat_seq[start:start + seq_len],
                    dtype=torch.float, device=device
                ).unsqueeze(0)

                regs = torch.tensor(
                    reg_seq[start:start + seq_len],
                    dtype=torch.long, device=device
                ).unsqueeze(0)

                target = torch.tensor(
                    [reg_seq[start + seq_len]],
                    dtype=torch.long, device=device
                )

                optimizer.zero_grad()
                logits, probs, _ = model(feats, regs)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                e_loss.append(loss.item())
                pred = logits.argmax(dim=-1)
                e_correct += (pred == target).sum().item()
                e_total += 1

        scheduler.step()
        avg_loss = sum(e_loss) / max(len(e_loss), 1)
        acc = e_correct / max(e_total, 1)
        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)

        if verbose and epoch % 10 == 0:
            print(f"RegimePredictor epoch {epoch}/{epochs}  "
                  f"loss={avg_loss:.4f}  acc={acc:.3f}")

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_regime_detector(
    detector: GraphRegimeDetector,
    graphs: List[Data],
    true_labels: List[int],
    ricci_curvatures: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """Evaluate GraphRegimeDetector clustering quality."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pred_labels = [
        detector.predict(
            graphs[i],
            ricci_curvatures[i] if ricci_curvatures is not None else None,
        )
        for i in range(len(graphs))
    ]

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    # Purity
    from collections import Counter
    n = len(true_labels)
    k = detector.n_regimes
    purity = 0.0
    for r in range(k):
        indices = [i for i, l in enumerate(pred_labels) if l == r]
        if not indices:
            continue
        true_in_cluster = [true_labels[i] for i in indices]
        most_common = Counter(true_in_cluster).most_common(1)[0][1]
        purity += most_common
    purity /= max(n, 1)

    return {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
        "purity": float(purity),
        "n_predicted_regimes": len(set(pred_labels)),
    }
