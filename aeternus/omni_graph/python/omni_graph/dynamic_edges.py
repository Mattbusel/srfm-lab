"""
dynamic_edges.py
================
Dynamic edge management for evolving financial graphs.

Implements:
  - Correlation threshold updating
  - Exponential moving average (EMA) edge weights
  - Regime-conditioned edge rewiring
  - Edge birth/death process
  - Temporal edge attention
  - Graph topology change detection (graph edit distance proxy)
  - Structural break detection on graph Laplacian eigenvalues
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected, remove_self_loops, coalesce
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ---------------------------------------------------------------------------
# EMA edge weight manager
# ---------------------------------------------------------------------------

class EMAEdgeWeightManager:
    """
    Maintain exponential moving average (EMA) weights for graph edges.

    For each edge (i, j), the EMA weight is updated at each time step:
        w_ema(t) = alpha * w_new(t) + (1 - alpha) * w_ema(t-1)

    Edges with weight below `death_threshold` are removed (edge death).
    New edges with weight above `birth_threshold` are added (edge birth).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        birth_threshold: float = 0.3,
        death_threshold: float = 0.05,
        max_edges: Optional[int] = None,
    ):
        self.alpha = alpha
        self.birth_threshold = birth_threshold
        self.death_threshold = death_threshold
        self.max_edges = max_edges

        # State: dict (i, j) -> ema_weight  (i < j convention)
        self._weights: Dict[Tuple[int, int], float] = {}
        self._age: Dict[Tuple[int, int], int] = {}
        self._t: int = 0

    def _key(self, i: int, j: int) -> Tuple[int, int]:
        return (min(i, j), max(i, j))

    def update(
        self,
        new_edge_index: Tensor,
        new_weights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Update EMA weights from new observations.

        Parameters
        ----------
        new_edge_index : (2, E_new)
        new_weights    : (E_new,) raw edge weights at this time step

        Returns
        -------
        edge_index : (2, E_active) with currently live edges
        ema_weights : (E_active,) EMA-smoothed weights
        """
        self._t += 1

        # Decay existing weights
        for k in list(self._weights.keys()):
            self._weights[k] *= (1 - self.alpha)

        # Update/insert new observations
        for idx in range(new_edge_index.shape[1]):
            i, j = int(new_edge_index[0, idx]), int(new_edge_index[1, idx])
            k = self._key(i, j)
            w = float(new_weights[idx])
            if k in self._weights:
                self._weights[k] = self.alpha * w + (1 - self.alpha) * self._weights[k]
            else:
                self._weights[k] = self.alpha * w  # born with partial weight
            self._age[k] = self._t

        # Kill edges below death threshold
        dead = [k for k, v in self._weights.items() if abs(v) < self.death_threshold]
        for k in dead:
            del self._weights[k]
            del self._age[k]

        # Trim to max_edges if needed
        if self.max_edges is not None and len(self._weights) > self.max_edges:
            sorted_edges = sorted(self._weights.items(), key=lambda x: abs(x[1]), reverse=True)
            self._weights = dict(sorted_edges[: self.max_edges])

        return self._to_tensors()

    def _to_tensors(self) -> Tuple[Tensor, Tensor]:
        if not self._weights:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

        edges = list(self._weights.keys())
        weights = [self._weights[e] for e in edges]

        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
        wt = torch.tensor(weights, dtype=torch.float32)

        # Make undirected
        src_bi = torch.cat([src, dst])
        dst_bi = torch.cat([dst, src])
        wt_bi = torch.cat([wt, wt])

        return torch.stack([src_bi, dst_bi], dim=0), wt_bi

    def state_dict(self) -> Dict:
        return {"weights": dict(self._weights), "age": dict(self._age), "t": self._t}

    def load_state_dict(self, state: Dict) -> None:
        self._weights = state["weights"]
        self._age = state["age"]
        self._t = state["t"]


# ---------------------------------------------------------------------------
# Correlation threshold updater
# ---------------------------------------------------------------------------

class AdaptiveCorrelationThreshold:
    """
    Dynamically adjust the correlation threshold used for edge construction.

    Strategy: maintain a rolling percentile of edge weight distribution.
    Threshold tracks the `percentile`-th percentile of recent correlations.

    Also handles high-volatility regimes where threshold should relax.
    """

    def __init__(
        self,
        base_threshold: float = 0.3,
        percentile: float = 70.0,
        window: int = 50,
        regime_sensitivity: float = 0.5,
    ):
        self.base_threshold = base_threshold
        self.percentile = percentile
        self.window = window
        self.regime_sensitivity = regime_sensitivity

        self._corr_history: deque = deque(maxlen=window)
        self._vol_history: deque = deque(maxlen=window)
        self._threshold = base_threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def update(
        self,
        correlations: np.ndarray,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Update threshold given new correlation observations.

        Parameters
        ----------
        correlations : 1D array of pairwise correlation values
        volatility   : current market volatility estimate (optional)

        Returns
        -------
        new threshold
        """
        self._corr_history.extend(correlations.tolist())

        if len(self._corr_history) >= 10:
            self._threshold = float(
                np.percentile(np.abs(list(self._corr_history)), self.percentile)
            )

        # Regime adjustment: in high-vol regimes, relax threshold
        if volatility is not None:
            self._vol_history.append(volatility)
            if len(self._vol_history) >= 5:
                vol_z = (volatility - np.mean(list(self._vol_history))) / (
                    np.std(list(self._vol_history)) + 1e-8
                )
                vol_factor = 1.0 - self.regime_sensitivity * max(vol_z, 0) * 0.1
                self._threshold = self._threshold * vol_factor

        self._threshold = float(np.clip(self._threshold, 0.01, 0.95))
        return self._threshold


# ---------------------------------------------------------------------------
# Regime-conditioned edge rewiring
# ---------------------------------------------------------------------------

class RegimeConditionedRewiring:
    """
    Rewire graph edges based on detected market regime.

    In different regimes (bull, bear, crisis, neutral), the correlation
    structure of markets changes dramatically. This module adjusts:
      - Which edges are active (by threshold)
      - Edge weight scaling per regime
      - Whether directed (lead-lag) edges are emphasised

    Regimes are expected as integer labels (0=neutral, 1=bull, 2=bear, 3=crisis).
    """

    REGIME_CONFIGS = {
        0: {"threshold_mult": 1.0, "weight_scale": 1.0, "emphasise_lead_lag": False},   # neutral
        1: {"threshold_mult": 0.9, "weight_scale": 1.1, "emphasise_lead_lag": False},   # bull
        2: {"threshold_mult": 1.1, "weight_scale": 0.9, "emphasise_lead_lag": True},    # bear
        3: {"threshold_mult": 0.7, "weight_scale": 1.5, "emphasise_lead_lag": True},    # crisis
    }

    def __init__(
        self,
        base_threshold: float = 0.3,
        n_regimes: int = 4,
    ):
        self.base_threshold = base_threshold
        self.n_regimes = n_regimes

    def rewire(
        self,
        edge_index: Tensor,
        edge_weights: Tensor,
        regime: int,
        lead_lag_index: Optional[Tensor] = None,
        lead_lag_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply regime-specific rewiring to graph.

        Parameters
        ----------
        edge_index    : (2, E) current edge index
        edge_weights  : (E,) current edge weights
        regime        : current regime label
        lead_lag_index  : optional (2, E_ll) lead-lag edges
        lead_lag_weights : optional (E_ll,)

        Returns
        -------
        new_edge_index : (2, E')
        new_weights    : (E',)
        """
        cfg = self.REGIME_CONFIGS.get(regime, self.REGIME_CONFIGS[0])
        threshold = self.base_threshold * cfg["threshold_mult"]
        scale = cfg["weight_scale"]

        # Filter by threshold
        mask = edge_weights.abs() >= threshold
        filtered_ei = edge_index[:, mask]
        filtered_w = edge_weights[mask] * scale

        # Optionally add lead-lag edges
        if cfg["emphasise_lead_lag"] and lead_lag_index is not None and lead_lag_weights is not None:
            filtered_ei = torch.cat([filtered_ei, lead_lag_index], dim=1)
            filtered_w = torch.cat([filtered_w, lead_lag_weights * scale * 1.5])

        return filtered_ei, filtered_w

    def smooth_regime_transition(
        self,
        prev_edge_index: Tensor,
        prev_weights: Tensor,
        next_edge_index: Tensor,
        next_weights: Tensor,
        alpha: float = 0.3,
    ) -> Tuple[Tensor, Tensor]:
        """
        Blend edge weights between two regimes for smooth transition.

        Only edges present in both graphs are blended; others are taken
        from the target regime.
        """
        # Build weight dicts for each graph
        prev_dict: Dict[Tuple[int, int], float] = {}
        for k in range(prev_edge_index.shape[1]):
            i, j = int(prev_edge_index[0, k]), int(prev_edge_index[1, k])
            prev_dict[(i, j)] = float(prev_weights[k])

        blended_ei_rows, blended_ei_cols, blended_ws = [], [], []
        for k in range(next_edge_index.shape[1]):
            i, j = int(next_edge_index[0, k]), int(next_edge_index[1, k])
            nw = float(next_weights[k])
            pw = prev_dict.get((i, j), 0.0)
            bw = alpha * pw + (1 - alpha) * nw
            blended_ei_rows.append(i)
            blended_ei_cols.append(j)
            blended_ws.append(bw)

        if not blended_ei_rows:
            return next_edge_index, next_weights

        ei = torch.stack([
            torch.tensor(blended_ei_rows, dtype=torch.long),
            torch.tensor(blended_ei_cols, dtype=torch.long),
        ], dim=0)
        w = torch.tensor(blended_ws, dtype=torch.float32)
        return ei, w


# ---------------------------------------------------------------------------
# Edge birth/death process
# ---------------------------------------------------------------------------

class EdgeBirthDeathProcess:
    """
    Stochastic model for edge birth and death in financial networks.

    Uses a continuous-time Markov chain:
      - Dead edges are born with rate lambda_b (proportional to current correlation)
      - Live edges die with rate lambda_d (inversely proportional to |weight|)

    Discrete-time approximation: at each step, birth/death probabilities
    are applied independently.
    """

    def __init__(
        self,
        birth_rate_base: float = 0.05,
        death_rate_base: float = 0.02,
        correlation_birth_scale: float = 2.0,
        weight_death_scale: float = 1.0,
        min_weight_to_survive: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.birth_rate_base = birth_rate_base
        self.death_rate_base = death_rate_base
        self.correlation_birth_scale = correlation_birth_scale
        self.weight_death_scale = weight_death_scale
        self.min_weight_to_survive = min_weight_to_survive
        self.rng = np.random.default_rng(seed)

        self._live_edges: Dict[Tuple[int, int], float] = {}

    def step(
        self,
        num_nodes: int,
        corr_matrix: np.ndarray,
        dt: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Advance the birth/death process by one time step.

        Parameters
        ----------
        num_nodes   : N
        corr_matrix : (N, N) current correlation matrix
        dt          : time step size

        Returns
        -------
        edge_index : (2, E)
        weights    : (E,)
        """
        # Death step for existing edges
        dead = []
        for (i, j), w in list(self._live_edges.items()):
            lambda_d = self.death_rate_base / (abs(w) + self.min_weight_to_survive) * self.weight_death_scale
            p_death = 1.0 - math.exp(-lambda_d * dt)
            if self.rng.random() < p_death:
                dead.append((i, j))

        for k in dead:
            del self._live_edges[k]

        # Birth step for non-existing edges
        existing = set(self._live_edges.keys())
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                k = (i, j)
                if k not in existing:
                    c = corr_matrix[i, j]
                    lambda_b = self.birth_rate_base * abs(c) * self.correlation_birth_scale
                    p_birth = 1.0 - math.exp(-lambda_b * dt)
                    if self.rng.random() < p_birth:
                        self._live_edges[k] = c

        return self._to_tensors()

    def _to_tensors(self) -> Tuple[Tensor, Tensor]:
        if not self._live_edges:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

        edges = list(self._live_edges.keys())
        weights = [self._live_edges[e] for e in edges]

        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
        wt = torch.tensor(weights, dtype=torch.float32)

        src_bi = torch.cat([src, dst])
        dst_bi = torch.cat([dst, src])
        wt_bi = torch.cat([wt, wt])

        return torch.stack([src_bi, dst_bi], dim=0), wt_bi

    @property
    def n_live_edges(self) -> int:
        return len(self._live_edges)


# ---------------------------------------------------------------------------
# Temporal edge attention
# ---------------------------------------------------------------------------

class TemporalEdgeAttention(nn.Module):
    """
    Compute attention weights for edges across time, combining:
      1. Edge weight magnitude
      2. Recency (recent edges get higher weight)
      3. Learned attention from edge feature MLP

    Used to aggregate multi-hop temporal neighbourhoods.
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
        time_decay: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.time_decay = time_decay

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim + 1, hidden_dim),  # +1 for time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, n_heads),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim / n_heads)

    def forward(
        self,
        edge_attr_seq: Tensor,
        time_deltas: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        edge_attr_seq : (T, E, F) edge features across T time steps
        time_deltas   : (T,) time difference to current step

        Returns
        -------
        attn_weights : (E, T) normalised attention weights
        """
        T, E, F = edge_attr_seq.shape

        # Time decay embedding: (T, 1)
        time_emb = torch.exp(-self.time_decay * time_deltas.float()).unsqueeze(-1)  # (T, 1)

        # Broadcast: (T, E, 1)
        time_emb = time_emb.unsqueeze(1).expand(T, E, 1)

        # Concat edge features with time embedding: (T, E, F+1)
        x = torch.cat([edge_attr_seq, time_emb], dim=-1)

        # Compute attention logits: (T, E, n_heads)
        logits = self.edge_mlp(x)

        # Pool over heads by averaging: (T, E)
        logits = logits.mean(dim=-1)

        # Normalise over time dimension: (E, T)
        attn = F.softmax(logits.transpose(0, 1), dim=-1)
        attn = self.dropout(attn)

        return attn

    def aggregate(
        self,
        edge_attr_seq: Tensor,
        time_deltas: Tensor,
    ) -> Tensor:
        """
        Compute time-aggregated edge features.

        Parameters
        ----------
        edge_attr_seq : (T, E, F)
        time_deltas   : (T,)

        Returns
        -------
        aggregated : (E, F)
        """
        attn = self.forward(edge_attr_seq, time_deltas)  # (E, T)
        # Weighted sum: (E, F)
        aggregated = torch.einsum("et,tef->ef", attn, edge_attr_seq)
        return aggregated


# ---------------------------------------------------------------------------
# Graph topology change detector
# ---------------------------------------------------------------------------

class GraphTopologyChangeDetector:
    """
    Detect significant topology changes between consecutive graph snapshots.

    Methods:
      1. Approximate graph edit distance via node/edge Jaccard similarity
      2. Spectral distance (Frobenius norm of Laplacian eigenvalue difference)
      3. Degree distribution KL divergence
    """

    def __init__(
        self,
        method: str = "spectral",
        change_threshold: float = 0.2,
        window: int = 5,
    ):
        assert method in ("jaccard", "spectral", "degree_kl", "composite")
        self.method = method
        self.change_threshold = change_threshold
        self.window = window

        self._history: deque = deque(maxlen=window)
        self._scores: deque = deque(maxlen=window)

    def compute_change_score(
        self,
        prev_ei: Tensor,
        curr_ei: Tensor,
        num_nodes: int,
    ) -> float:
        """
        Compute a scalar change score in [0, 1] between two graph topologies.
        """
        if self.method == "jaccard":
            return self._jaccard_distance(prev_ei, curr_ei)
        elif self.method == "spectral":
            return self._spectral_distance(prev_ei, curr_ei, num_nodes)
        elif self.method == "degree_kl":
            return self._degree_kl(prev_ei, curr_ei, num_nodes)
        elif self.method == "composite":
            j = self._jaccard_distance(prev_ei, curr_ei)
            s = self._spectral_distance(prev_ei, curr_ei, num_nodes)
            d = self._degree_kl(prev_ei, curr_ei, num_nodes)
            return float((j + s + d) / 3.0)
        return 0.0

    def update(
        self,
        prev_ei: Tensor,
        curr_ei: Tensor,
        num_nodes: int,
    ) -> Tuple[float, bool]:
        """
        Update detector state and return (score, is_change_detected).
        """
        score = self.compute_change_score(prev_ei, curr_ei, num_nodes)
        self._scores.append(score)

        # Baseline from history
        if len(self._scores) >= 3:
            baseline = float(np.mean(list(self._scores)[:-1]))
            is_change = score > baseline + self.change_threshold
        else:
            is_change = score > self.change_threshold

        return score, is_change

    def _jaccard_distance(self, prev_ei: Tensor, curr_ei: Tensor) -> float:
        def edge_set(ei: Tensor) -> set:
            s = set()
            for k in range(ei.shape[1]):
                i, j = int(ei[0, k]), int(ei[1, k])
                s.add((min(i, j), max(i, j)))
            return s

        prev_s = edge_set(prev_ei)
        curr_s = edge_set(curr_ei)
        intersection = len(prev_s & curr_s)
        union = len(prev_s | curr_s)
        if union == 0:
            return 0.0
        return 1.0 - intersection / union

    def _spectral_distance(
        self,
        prev_ei: Tensor,
        curr_ei: Tensor,
        num_nodes: int,
    ) -> float:
        def laplacian_spectrum(ei: Tensor, n: int) -> np.ndarray:
            A = np.zeros((n, n))
            for k in range(ei.shape[1]):
                i, j = int(ei[0, k]), int(ei[1, k])
                if 0 <= i < n and 0 <= j < n and i != j:
                    A[i, j] = 1.0
            D = np.diag(A.sum(axis=1))
            L = D - A
            eigvals = np.linalg.eigvalsh(L)
            return eigvals

        if num_nodes > 200:
            # Skip expensive computation for large graphs
            return self._jaccard_distance(prev_ei, curr_ei)

        prev_eigs = laplacian_spectrum(prev_ei, num_nodes)
        curr_eigs = laplacian_spectrum(curr_ei, num_nodes)
        dist = float(np.linalg.norm(prev_eigs - curr_eigs)) / (num_nodes + 1e-8)
        return float(np.clip(dist, 0.0, 1.0))

    def _degree_kl(
        self,
        prev_ei: Tensor,
        curr_ei: Tensor,
        num_nodes: int,
    ) -> float:
        def degree_dist(ei: Tensor, n: int) -> np.ndarray:
            deg = np.zeros(n)
            for k in range(ei.shape[1]):
                i = int(ei[0, k])
                if 0 <= i < n:
                    deg[i] += 1
            deg = deg / (deg.sum() + 1e-8)
            return deg + 1e-8  # avoid log(0)

        p = degree_dist(prev_ei, num_nodes)
        q = degree_dist(curr_ei, num_nodes)
        kl = float(np.sum(p * np.log(p / q)))
        return float(np.clip(kl / 10.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Structural break detector (Laplacian eigenvalues)
# ---------------------------------------------------------------------------

class LaplacianStructuralBreakDetector:
    """
    Detect structural breaks in graph topology by monitoring Laplacian
    eigenvalue time series.

    Specifically tracks:
      - Fiedler value (λ₂): algebraic connectivity / liquidity proxy
      - Spectral gap: λ₂ - λ₁
      - Largest eigenvalue: λ_max (related to maximum degree)
      - Rank of L (number of non-trivial eigenvalues)

    Uses CUSUM (cumulative sum) change-point detection on eigenvalue series.
    """

    def __init__(
        self,
        n_eigenvalues: int = 5,
        cusum_threshold: float = 3.0,
        cusum_drift: float = 0.5,
        window: int = 20,
    ):
        self.n_eigenvalues = n_eigenvalues
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self.window = window

        self._eigenvalue_history: List[np.ndarray] = []
        self._cusum_pos: np.ndarray = np.zeros(n_eigenvalues)
        self._cusum_neg: np.ndarray = np.zeros(n_eigenvalues)
        self._mu_hat: Optional[np.ndarray] = None
        self._sigma_hat: Optional[np.ndarray] = None

    def update(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weights: Optional[Tensor] = None,
    ) -> Dict[str, Union[float, bool, np.ndarray]]:
        """
        Update the detector with a new graph snapshot.

        Returns
        -------
        dict with:
          fiedler_value: float
          spectral_gap: float
          lambda_max: float
          eigenvalues: np.ndarray
          cusum_alarm: bool
          break_components: list of which eigenvalues triggered alarm
        """
        eigvals = self._compute_eigenvalues(edge_index, num_nodes, edge_weights)
        self._eigenvalue_history.append(eigvals)

        # Initialise statistics from first `window` steps
        if len(self._eigenvalue_history) == self.window:
            hist = np.array(self._eigenvalue_history)
            self._mu_hat = hist.mean(axis=0)
            self._sigma_hat = hist.std(axis=0) + 1e-8
            self._cusum_pos = np.zeros(self.n_eigenvalues)
            self._cusum_neg = np.zeros(self.n_eigenvalues)

        cusum_alarm = False
        break_components: List[int] = []

        if self._mu_hat is not None:
            z = (eigvals - self._mu_hat) / self._sigma_hat
            self._cusum_pos = np.maximum(0, self._cusum_pos + z - self.cusum_drift)
            self._cusum_neg = np.maximum(0, self._cusum_neg - z - self.cusum_drift)

            alarm_mask = (self._cusum_pos > self.cusum_threshold) | (
                self._cusum_neg > self.cusum_threshold
            )
            if alarm_mask.any():
                cusum_alarm = True
                break_components = list(np.where(alarm_mask)[0])
                # Reset CUSUM after alarm
                self._cusum_pos[alarm_mask] = 0.0
                self._cusum_neg[alarm_mask] = 0.0

        # Update rolling mean/std with exponential decay
        if self._mu_hat is not None and len(self._eigenvalue_history) > self.window:
            alpha = 2.0 / (self.window + 1)
            self._mu_hat = (1 - alpha) * self._mu_hat + alpha * eigvals
            residual = (eigvals - self._mu_hat) ** 2
            self._sigma_hat = np.sqrt((1 - alpha) * self._sigma_hat ** 2 + alpha * residual) + 1e-8

        fiedler = float(eigvals[1]) if len(eigvals) > 1 else 0.0
        spectral_gap = float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else 0.0
        lambda_max = float(eigvals[-1]) if len(eigvals) > 0 else 0.0

        return {
            "fiedler_value": fiedler,
            "spectral_gap": spectral_gap,
            "lambda_max": lambda_max,
            "eigenvalues": eigvals,
            "cusum_alarm": cusum_alarm,
            "break_components": break_components,
            "cusum_pos": self._cusum_pos.copy(),
            "cusum_neg": self._cusum_neg.copy(),
        }

    def _compute_eigenvalues(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weights: Optional[Tensor],
    ) -> np.ndarray:
        n = min(num_nodes, 500)  # cap for efficiency

        A = np.zeros((n, n), dtype=np.float32)
        E = edge_index.shape[1]
        for k in range(E):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if 0 <= i < n and 0 <= j < n and i != j:
                w = float(edge_weights[k]) if edge_weights is not None else 1.0
                A[i, j] = abs(w)
                A[j, i] = abs(w)

        deg = A.sum(axis=1)
        D = np.diag(deg)
        L = D - A

        if n <= 100:
            eigvals = np.linalg.eigvalsh(L)
        else:
            # Partial eigenvalue decomposition (first k)
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix
            L_sparse = csr_matrix(L)
            k = min(self.n_eigenvalues, n - 2)
            eigvals, _ = eigsh(L_sparse, k=k, which="SM")
            eigvals = np.sort(eigvals)

        # Pad or trim to n_eigenvalues
        if len(eigvals) < self.n_eigenvalues:
            eigvals = np.pad(eigvals, (0, self.n_eigenvalues - len(eigvals)))
        else:
            eigvals = eigvals[: self.n_eigenvalues]

        return eigvals.astype(np.float32)

    def get_fiedler_series(self) -> np.ndarray:
        """Return the full history of Fiedler values."""
        if len(self._eigenvalue_history) < 2:
            return np.array([])
        hist = np.array(self._eigenvalue_history)
        if hist.shape[1] > 1:
            return hist[:, 1]
        return np.zeros(len(self._eigenvalue_history))


# ---------------------------------------------------------------------------
# Dynamic graph state manager
# ---------------------------------------------------------------------------

class DynamicGraphStateManager:
    """
    Central manager for a dynamic financial graph.

    Coordinates:
      - EMA edge weights
      - Adaptive threshold
      - Regime-conditioned rewiring
      - Edge birth/death process
      - Structural break detection

    Provides a unified `.step()` API used by the streaming pipeline.
    """

    def __init__(
        self,
        num_nodes: int,
        ema_alpha: float = 0.1,
        birth_threshold: float = 0.3,
        death_threshold: float = 0.05,
        n_regimes: int = 4,
        detect_breaks: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_nodes = num_nodes
        self.detect_breaks = detect_breaks

        self.ema_manager = EMAEdgeWeightManager(
            alpha=ema_alpha,
            birth_threshold=birth_threshold,
            death_threshold=death_threshold,
        )
        self.threshold_adapter = AdaptiveCorrelationThreshold(
            base_threshold=birth_threshold,
        )
        self.rewirer = RegimeConditionedRewiring(
            base_threshold=birth_threshold,
            n_regimes=n_regimes,
        )
        self.birth_death = EdgeBirthDeathProcess(seed=seed)
        if detect_breaks:
            self.break_detector = LaplacianStructuralBreakDetector()
        self.change_detector = GraphTopologyChangeDetector()

        self._prev_edge_index: Optional[Tensor] = None
        self._current_regime: int = 0
        self._t: int = 0

    def step(
        self,
        new_edge_index: Tensor,
        new_edge_weights: Tensor,
        corr_matrix: np.ndarray,
        regime: int = 0,
        volatility: Optional[float] = None,
    ) -> Dict:
        """
        Advance dynamic graph by one time step.

        Parameters
        ----------
        new_edge_index  : (2, E) observed edges this period
        new_edge_weights : (E,) observed weights this period
        corr_matrix     : (N, N) current correlation matrix
        regime          : current market regime label
        volatility      : current volatility estimate

        Returns
        -------
        dict with:
          edge_index: (2, E') active edges after all updates
          edge_weights: (E',) smoothed weights
          change_score: float
          break_detected: bool
          break_info: dict (if detect_breaks)
          regime: int
        """
        self._t += 1
        self._current_regime = regime

        # 1. Update adaptive threshold
        threshold = self.threshold_adapter.update(
            np.abs(new_edge_weights.numpy()) if isinstance(new_edge_weights, Tensor)
            else np.abs(new_edge_weights),
            volatility,
        )

        # 2. EMA update
        ema_ei, ema_w = self.ema_manager.update(new_edge_index, new_edge_weights)

        # 3. Regime rewiring
        if ema_ei.shape[1] > 0:
            ema_ei, ema_w = self.rewirer.rewire(ema_ei, ema_w, regime)

        # 4. Birth/death update
        bd_ei, bd_w = self.birth_death.step(
            self.num_nodes, corr_matrix, dt=1.0
        )

        # Merge EMA and birth/death edges
        if ema_ei.shape[1] > 0 and bd_ei.shape[1] > 0:
            merged_ei = torch.cat([ema_ei, bd_ei], dim=1)
            merged_w = torch.cat([ema_w, bd_w])
        elif ema_ei.shape[1] > 0:
            merged_ei, merged_w = ema_ei, ema_w
        else:
            merged_ei, merged_w = bd_ei, bd_w

        # 5. Topology change detection
        change_score = 0.0
        change_detected = False
        if self._prev_edge_index is not None and merged_ei.shape[1] > 0:
            change_score, change_detected = self.change_detector.update(
                self._prev_edge_index, merged_ei, self.num_nodes
            )

        # 6. Structural break detection
        break_info = {}
        break_detected = False
        if self.detect_breaks and merged_ei.shape[1] > 0:
            break_info = self.break_detector.update(merged_ei, self.num_nodes, merged_w)
            break_detected = break_info.get("cusum_alarm", False)

        self._prev_edge_index = merged_ei.clone() if merged_ei.shape[1] > 0 else None

        return {
            "edge_index": merged_ei,
            "edge_weights": merged_w,
            "change_score": change_score,
            "change_detected": change_detected,
            "break_detected": break_detected,
            "break_info": break_info,
            "regime": regime,
            "threshold": threshold,
            "t": self._t,
        }


# ---------------------------------------------------------------------------
# Temporal edge feature extractor
# ---------------------------------------------------------------------------

class TemporalEdgeFeatureExtractor(nn.Module):
    """
    Extract temporally-aware edge features from a sequence of edge attribute tensors.

    Uses:
      1. LSTM over edge feature time series
      2. Temporal edge attention for context aggregation
      3. Edge-level positional encoding
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=edge_feat_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=False,
            dropout=dropout if n_layers > 1 else 0,
        )

        if use_attention:
            self.attn = TemporalEdgeAttention(
                edge_feat_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
            )

        self.output_proj = nn.Linear(hidden_dim * 2 if use_attention else hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        edge_attr_seq: Tensor,
        time_deltas: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        edge_attr_seq : (T, E, F)
        time_deltas   : (T,) optional

        Returns
        -------
        edge_repr : (E, hidden_dim)
        """
        T, E, F = edge_attr_seq.shape

        # Reshape for LSTM: (T, E, F) → (T, E, F)
        lstm_out, (h_n, _) = self.lstm(edge_attr_seq)  # lstm_out: (T, E, H)

        # Take last hidden state
        h_last = lstm_out[-1]  # (E, H)

        if self.use_attention and time_deltas is not None:
            attn_feat = self.attn.aggregate(edge_attr_seq, time_deltas)  # (E, F)
            # Project attn_feat to hidden_dim before concat
            combined = torch.cat([h_last, attn_feat[:, :self.hidden_dim]], dim=-1)
        else:
            combined = h_last

        out = self.output_proj(self.dropout(combined))
        return out


# ---------------------------------------------------------------------------
# Graph edit distance (approximate)
# ---------------------------------------------------------------------------

def approximate_graph_edit_distance(
    ei_a: Tensor,
    ei_b: Tensor,
    num_nodes: int,
    normalise: bool = True,
) -> float:
    """
    Compute an approximate graph edit distance between two graphs
    defined by their edge sets.

    GED ≈ |E_a Δ E_b| (symmetric edge difference).

    Parameters
    ----------
    ei_a, ei_b : (2, E) edge index tensors
    num_nodes  : N
    normalise  : divide by max possible edges

    Returns
    -------
    ged : float
    """
    def edge_set(ei: Tensor) -> set:
        s = set()
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            s.add((min(i, j), max(i, j)))
        return s

    set_a = edge_set(ei_a)
    set_b = edge_set(ei_b)
    symmetric_diff = len(set_a.symmetric_difference(set_b))

    if normalise:
        max_edges = num_nodes * (num_nodes - 1) / 2
        return symmetric_diff / (max_edges + 1e-8)
    return float(symmetric_diff)


# ---------------------------------------------------------------------------
# Edge weight time series analyser
# ---------------------------------------------------------------------------

class EdgeWeightTimeSeriesAnalyser:
    """
    Analyse temporal evolution of specific edges' weights.

    Tracks selected edges and computes statistics on their weight trajectories:
      - Trend (linear regression slope)
      - Volatility (std dev of weight changes)
      - Mean-reversion speed (AR(1) coefficient)
      - Stationarity test (simplified ADF proxy)
    """

    def __init__(self, tracked_edges: List[Tuple[int, int]], max_history: int = 100):
        self.tracked_edges = [(min(i, j), max(i, j)) for i, j in tracked_edges]
        self.max_history = max_history
        self._histories: Dict[Tuple[int, int], deque] = {
            e: deque(maxlen=max_history) for e in self.tracked_edges
        }

    def update(
        self,
        edge_index: Tensor,
        edge_weights: Tensor,
    ) -> None:
        """Record current weights for tracked edges."""
        weight_map: Dict[Tuple[int, int], float] = {}
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            key = (min(i, j), max(i, j))
            weight_map[key] = float(edge_weights[k])

        for e in self.tracked_edges:
            w = weight_map.get(e, 0.0)  # 0 if edge died
            self._histories[e].append(w)

    def analyse_edge(self, edge: Tuple[int, int]) -> Dict:
        """Compute statistics for one tracked edge."""
        key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        hist = np.array(list(self._histories.get(key, [])), dtype=np.float32)

        if len(hist) < 5:
            return {"error": "insufficient_data"}

        # Trend
        t = np.arange(len(hist))
        slope = float(np.polyfit(t, hist, 1)[0])

        # Volatility
        changes = np.diff(hist)
        vol = float(np.std(changes))

        # AR(1) coefficient (mean reversion speed)
        if len(hist) > 2:
            ar1 = float(np.corrcoef(hist[:-1], hist[1:])[0, 1])
        else:
            ar1 = 0.0

        # Half-life of mean reversion (if AR1 < 1)
        if ar1 < 1.0 and ar1 > 0.0:
            half_life = -math.log(2) / math.log(ar1)
        else:
            half_life = float("inf")

        return {
            "mean": float(hist.mean()),
            "std": float(hist.std()),
            "trend_slope": slope,
            "volatility_of_changes": vol,
            "ar1_coefficient": ar1,
            "half_life": half_life,
            "n_observations": len(hist),
            "live_fraction": float((hist != 0.0).mean()),
        }

    def analyse_all(self) -> Dict[Tuple[int, int], Dict]:
        return {e: self.analyse_edge(e) for e in self.tracked_edges}


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "EMAEdgeWeightManager",
    "AdaptiveCorrelationThreshold",
    "RegimeConditionedRewiring",
    "EdgeBirthDeathProcess",
    "TemporalEdgeAttention",
    "GraphTopologyChangeDetector",
    "LaplacianStructuralBreakDetector",
    "DynamicGraphStateManager",
    "TemporalEdgeFeatureExtractor",
    "EdgeWeightTimeSeriesAnalyser",
    "approximate_graph_edit_distance",
]
