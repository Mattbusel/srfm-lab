"""
temporal_gnn.py
===============
Temporal Graph Neural Networks for dynamic financial networks.

Implements:
  - TGNN with memory module (TGN-style)
  - Temporal attention with time encoding
  - Harmonic time features (time2vec-style)
  - Graph snapshot interpolation
  - Continuous-Time Dynamic Graph (CTDG) convolution
  - Temporal node classification and link prediction heads
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Time encoding (harmonic / time2vec)
# ---------------------------------------------------------------------------

class HarmonicTimeEncoder(nn.Module):
    """
    Encode continuous timestamps using harmonic (Fourier) features.

    Following Time2Vec: f(t)_i = sin(w_i * t + phi_i) for i > 0,
    and f(t)_0 = w_0 * t + phi_0 (linear).

    Reference: Kazemi et al., "Time2Vec: Learning a Vector Representation of Time", 2019.
    """

    def __init__(self, out_dim: int, learnable: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.learnable = learnable

        if learnable:
            self.w = nn.Parameter(torch.randn(out_dim))
            self.phi = nn.Parameter(torch.zeros(out_dim))
        else:
            # Fixed Fourier frequencies (powers of 2)
            freqs = torch.zeros(out_dim)
            freqs[0] = 1.0  # linear
            for i in range(1, out_dim):
                freqs[i] = 2.0 ** (i - 1)
            self.register_buffer("w", freqs)
            self.register_buffer("phi", torch.zeros(out_dim))

    def forward(self, t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        t : (...) scalar timestamps

        Returns
        -------
        enc : (..., out_dim)
        """
        t = t.unsqueeze(-1)  # (..., 1)
        wt = t * self.w.unsqueeze(0) + self.phi.unsqueeze(0)  # (..., out_dim)

        # First feature: linear; rest: sinusoidal
        linear = wt[..., :1]
        sinusoidal = torch.sin(wt[..., 1:])
        return torch.cat([linear, sinusoidal], dim=-1)


class ExponentialTimeDecay(nn.Module):
    """
    Encode time elapsed as exponential decay features.

    enc(dt)_i = exp(-gamma_i * dt)

    Useful for recency weighting in temporal graphs.
    """

    def __init__(self, out_dim: int, init_gamma: float = 1.0):
        super().__init__()
        self.gammas = nn.Parameter(torch.full((out_dim,), init_gamma))

    def forward(self, dt: Tensor) -> Tensor:
        """
        Parameters
        ----------
        dt : (...) time deltas (non-negative)

        Returns
        -------
        enc : (..., out_dim)
        """
        dt = dt.unsqueeze(-1).clamp(min=0.0)
        return torch.exp(-F.softplus(self.gammas) * dt)


# ---------------------------------------------------------------------------
# Temporal memory module (TGN-style)
# ---------------------------------------------------------------------------

class TemporalMemoryModule(nn.Module):
    """
    Persistent node memory for temporal graph neural networks.

    Maintains a memory vector per node that evolves over time
    based on interactions (edges). Uses GRU-style update.

    Based on TGN: Rossi et al., "Temporal Graph Networks", 2020.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        message_dim: int,
        time_dim: int = 16,
        aggregator: str = "last",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.aggregator = aggregator

        # Memory storage (not a parameter — updated via detached operations)
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update_time", torch.zeros(num_nodes))

        # Message function: concatenate src_memory, dst_memory, edge_feat, time_enc
        self.time_encoder = HarmonicTimeEncoder(time_dim)

        msg_in_dim = memory_dim * 2 + message_dim + time_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_in_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

        # Memory updater (GRU)
        self.memory_updater = nn.GRUCell(
            input_size=message_dim,
            hidden_size=memory_dim,
        )

    def get_memory(self, node_ids: Tensor) -> Tensor:
        return self.memory[node_ids]

    def update_memory(
        self,
        src_ids: Tensor,       # (E,) source node IDs
        dst_ids: Tensor,       # (E,) destination node IDs
        edge_feat: Tensor,     # (E, message_dim)
        timestamps: Tensor,    # (E,)
    ) -> None:
        """
        Update memory for all involved nodes.

        For each event (src, dst, t, edge_feat):
        1. Compute message using current memories + time encoding
        2. Aggregate messages per node (last / mean)
        3. Update memory via GRU
        """
        # Time encoding for elapsed time
        dt_src = timestamps - self.last_update_time[src_ids]
        dt_dst = timestamps - self.last_update_time[dst_ids]

        time_enc_src = self.time_encoder(dt_src)
        time_enc_dst = self.time_encoder(dt_dst)

        # Messages from src→dst and dst→src
        src_mem = self.memory[src_ids]
        dst_mem = self.memory[dst_ids]

        msg_src = self.message_mlp(
            torch.cat([src_mem, dst_mem, edge_feat, time_enc_src], dim=-1)
        )
        msg_dst = self.message_mlp(
            torch.cat([dst_mem, src_mem, edge_feat, time_enc_dst], dim=-1)
        )

        # Aggregate messages per node
        all_nodes = torch.cat([src_ids, dst_ids])
        all_msgs = torch.cat([msg_src, msg_dst], dim=0)

        unique_nodes = torch.unique(all_nodes)
        aggregated = torch.zeros(len(unique_nodes), self.message_dim, device=edge_feat.device)

        for k, node in enumerate(unique_nodes):
            mask = all_nodes == node
            if self.aggregator == "last":
                # Last message (by index)
                aggregated[k] = all_msgs[mask][-1]
            elif self.aggregator == "mean":
                aggregated[k] = all_msgs[mask].mean(dim=0)
            else:
                aggregated[k] = all_msgs[mask][-1]

        # GRU update
        old_memory = self.memory[unique_nodes]
        new_memory = self.memory_updater(aggregated, old_memory)

        # Detach and write back
        self.memory[unique_nodes] = new_memory.detach()
        self.last_update_time[src_ids] = timestamps
        self.last_update_time[dst_ids] = timestamps

    def reset(self, node_ids: Optional[Tensor] = None) -> None:
        """Reset memory (all or specified nodes)."""
        if node_ids is None:
            self.memory.zero_()
            self.last_update_time.zero_()
        else:
            self.memory[node_ids] = 0.0
            self.last_update_time[node_ids] = 0.0


# ---------------------------------------------------------------------------
# Temporal graph attention convolution
# ---------------------------------------------------------------------------

class TemporalGraphAttention(nn.Module):
    """
    Temporal graph attention: aggregate messages from temporal neighbours,
    weighting by both attention scores and time decay.

    For each target node u at time t:
      h(u, t) = AGG_v∈N(u,t) [ alpha(u, v, t-t_e) * (W_v * h_v || time_enc(t-t_e)) ]

    where alpha is a learned attention weight.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int = 16,
        out_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0

        self.time_encoder = HarmonicTimeEncoder(time_dim)

        msg_dim = node_dim + edge_dim + time_dim

        self.W_q = nn.Linear(node_dim + time_dim, out_dim)
        self.W_k = nn.Linear(msg_dim, out_dim)
        self.W_v = nn.Linear(msg_dim, out_dim)

        self.attn_proj = nn.Linear(out_dim, out_dim)
        self.out_proj = nn.Linear(out_dim + node_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        src_feat: Tensor,    # (E, node_dim) source node features
        dst_feat: Tensor,    # (E, node_dim) target node features
        edge_feat: Tensor,   # (E, edge_dim)
        time_deltas: Tensor, # (E,) t_current - t_edge
        dst_ids: Tensor,     # (E,) global IDs of target nodes
        n_dst: int,          # total number of target nodes
        query_time: Tensor,  # (n_dst,) current query time per node
    ) -> Tensor:
        """
        Compute updated representations for n_dst target nodes.

        Returns
        -------
        out : (n_dst, out_dim)
        """
        E = src_feat.shape[0]

        # Time encodings
        time_enc = self.time_encoder(time_deltas)     # (E, time_dim)

        # Messages (key and value) from source neighbours
        msg = torch.cat([src_feat, edge_feat, time_enc], dim=-1)  # (E, node_dim+edge_dim+time_dim)

        # Query from destination nodes (use zero time delta for query)
        query_time_enc = self.time_encoder(torch.zeros(n_dst, device=src_feat.device))
        q_in = torch.cat([
            dst_feat[dst_ids] if dst_feat.shape[0] > dst_ids.max() else src_feat,
            query_time_enc[dst_ids] if n_dst > 0 else query_time_enc[:E],
        ], dim=-1)

        Q = self.W_q(q_in).view(E, self.n_heads, self.head_dim)
        K = self.W_k(msg).view(E, self.n_heads, self.head_dim)
        V = self.W_v(msg).view(E, self.n_heads, self.head_dim)

        # Attention scores
        scores = (Q * K).sum(dim=-1) / self.scale  # (E, n_heads)
        attn = torch.exp(scores.clamp(-5, 5))       # (E, n_heads)

        # Aggregate to dst nodes
        V_flat = V.reshape(E, -1)    # (E, out_dim)
        attn_flat = attn.mean(dim=-1, keepdim=True)  # (E, 1)

        dst_agg = torch.zeros(n_dst, self.n_heads * self.head_dim, device=src_feat.device)
        attn_norm = torch.zeros(n_dst, 1, device=src_feat.device)

        dst_exp = dst_ids.unsqueeze(-1).expand(-1, V_flat.shape[-1])
        dst_agg.scatter_add_(0, dst_exp, V_flat * attn_flat)
        attn_norm.scatter_add_(0, dst_ids.unsqueeze(-1), attn_flat)

        out = dst_agg / (attn_norm + 1e-8)
        return self.norm(self.dropout(out))


# ---------------------------------------------------------------------------
# TGN (Temporal Graph Network) full model
# ---------------------------------------------------------------------------

class TGN(nn.Module):
    """
    Temporal Graph Network for financial event streams.

    Architecture:
      1. Memory module: tracks per-node state over time
      2. Graph attention over temporal neighbourhood
      3. Node-level task heads (classification, regression)

    Reference: Rossi et al., "Temporal Graph Networks for Deep Learning on
    Dynamic Graphs", 2020.
    """

    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        memory_dim: int = 128,
        time_dim: int = 16,
        hidden_dim: int = 256,
        out_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        message_agg: str = "last",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Memory module
        self.memory = TemporalMemoryModule(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            message_dim=hidden_dim,
            time_dim=time_dim,
            aggregator=message_agg,
        )

        # Input projection: combines node features with memory
        self.input_proj = nn.Linear(node_feat_dim + memory_dim, hidden_dim)

        # Temporal attention layers
        self.temporal_attns = nn.ModuleList([
            TemporalGraphAttention(
                node_dim=hidden_dim,
                edge_dim=edge_feat_dim,
                time_dim=time_dim,
                out_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_feat: Tensor,     # (N, node_feat_dim)
        edge_index: Tensor,    # (2, E)
        edge_feat: Tensor,     # (E, edge_feat_dim)
        edge_times: Tensor,    # (E,) timestamps
        query_time: float = 0.0,
    ) -> Tensor:
        """
        Parameters
        ----------
        node_feat  : (N, F) raw node features
        edge_index : (2, E)
        edge_feat  : (E, FE)
        edge_times : (E,) event timestamps
        query_time : scalar, current time for prediction

        Returns
        -------
        node_embeds : (N, out_dim)
        """
        N = node_feat.shape[0]

        # Combine node features with memory
        memory = self.memory.get_memory(torch.arange(N, device=node_feat.device))
        h = self.input_proj(torch.cat([node_feat, memory], dim=-1))

        src_ids, dst_ids = edge_index[0], edge_index[1]
        time_deltas = query_time - edge_times

        for attn, norm in zip(self.temporal_attns, self.layer_norms):
            h_src = h[src_ids]
            h_dst = h[dst_ids]

            h_new = attn(
                src_feat=h_src,
                dst_feat=h_dst,
                edge_feat=edge_feat,
                time_deltas=time_deltas,
                dst_ids=dst_ids,
                n_dst=N,
                query_time=torch.full((N,), query_time, device=h.device),
            )
            h = norm(h + h_new)

        return self.output_proj(self.dropout(h))

    def update_memory(
        self,
        src_ids: Tensor,
        dst_ids: Tensor,
        edge_feat: Tensor,
        timestamps: Tensor,
    ) -> None:
        """Update node memory after observing new events."""
        self.memory.update_memory(src_ids, dst_ids, edge_feat, timestamps)

    def reset_memory(self) -> None:
        self.memory.reset()


# ---------------------------------------------------------------------------
# Graph snapshot interpolation
# ---------------------------------------------------------------------------

class GraphSnapshotInterpolator(nn.Module):
    """
    Interpolate between two graph snapshots to generate intermediate states.

    Useful for:
      - Up-sampling sparse temporal graphs
      - Smooth transition between graph states for visualisation
      - Data augmentation

    Supports linear, cubic spline, and learned interpolation.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: Optional[int] = None,
        method: str = "linear",
        n_steps: int = 5,
    ):
        super().__init__()
        assert method in ("linear", "slerp", "learned")
        self.method = method
        self.n_steps = n_steps

        if method == "learned":
            in_dim = node_dim * 2
            self.interp_net = nn.Sequential(
                nn.Linear(in_dim + 1, node_dim * 2),
                nn.GELU(),
                nn.Linear(node_dim * 2, node_dim),
            )

    def interpolate_nodes(
        self,
        x_a: Tensor,
        x_b: Tensor,
        alphas: Optional[List[float]] = None,
    ) -> List[Tensor]:
        """
        Interpolate node feature tensors between x_a (t=0) and x_b (t=1).

        Parameters
        ----------
        x_a, x_b : (N, D) node features at two time points
        alphas    : interpolation coefficients in [0, 1]

        Returns
        -------
        List of (N, D) tensors, one per alpha
        """
        if alphas is None:
            alphas = [i / (self.n_steps + 1) for i in range(1, self.n_steps + 1)]

        results = []
        for alpha in alphas:
            if self.method == "linear":
                x_interp = (1 - alpha) * x_a + alpha * x_b
            elif self.method == "slerp":
                x_interp = self._slerp(x_a, x_b, alpha)
            elif self.method == "learned":
                alpha_t = torch.full((x_a.shape[0], 1), alpha, device=x_a.device)
                x_interp = self.interp_net(torch.cat([x_a, x_b, alpha_t], dim=-1))
            else:
                x_interp = (1 - alpha) * x_a + alpha * x_b
            results.append(x_interp)

        return results

    def _slerp(self, a: Tensor, b: Tensor, t: float) -> Tensor:
        """Spherical linear interpolation."""
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        dot = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        safe = sin_omega.abs() > 1e-6
        w_a = torch.where(safe, torch.sin((1 - t) * omega) / (sin_omega + 1e-8),
                          torch.full_like(sin_omega, 1 - t))
        w_b = torch.where(safe, torch.sin(t * omega) / (sin_omega + 1e-8),
                          torch.full_like(sin_omega, t))

        norm_a = a.norm(dim=-1, keepdim=True)
        norm_b = b.norm(dim=-1, keepdim=True)
        norm_interp = (1 - t) * norm_a + t * norm_b

        return F.normalize(w_a * a + w_b * b, dim=-1) * norm_interp

    def interpolate_edges(
        self,
        ei_a: Tensor,
        w_a: Tensor,
        ei_b: Tensor,
        w_b: Tensor,
        n_nodes: int,
        alpha: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Interpolate edge sets between two snapshots.

        Edges present only in one snapshot are weighted by (1-alpha) or alpha.
        """
        def edge_dict(ei: Tensor, w: Tensor) -> Dict[Tuple[int, int], float]:
            d = {}
            for k in range(ei.shape[1]):
                i, j = int(ei[0, k]), int(ei[1, k])
                d[(min(i, j), max(i, j))] = float(w[k]) if w.shape[0] > k else 1.0
            return d

        dict_a = edge_dict(ei_a, w_a)
        dict_b = edge_dict(ei_b, w_b)
        all_edges = set(dict_a.keys()) | set(dict_b.keys())

        rows, cols, ws = [], [], []
        for (i, j) in all_edges:
            wa = dict_a.get((i, j), 0.0)
            wb = dict_b.get((i, j), 0.0)
            w_interp = (1 - alpha) * wa + alpha * wb
            if abs(w_interp) > 1e-6:
                rows.append(i); cols.append(j); ws.append(w_interp)

        if not rows:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

        ei = torch.tensor([rows, cols], dtype=torch.long)
        w_out = torch.tensor(ws, dtype=torch.float32)
        return ei, w_out


# ---------------------------------------------------------------------------
# Continuous-Time Dynamic Graph (CTDG) convolution
# ---------------------------------------------------------------------------

class CTDGConv(nn.Module):
    """
    Continuous-Time Dynamic Graph convolution layer.

    For each node u at query time t, aggregates messages from
    its temporal neighbourhood N(u, t) = {v : (v, u, t_e) exists, t_e <= t}.

    Messages are weighted by time-decay and learned edge attention.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int = 16,
        out_dim: int = 128,
        n_heads: int = 4,
        time_decay: str = "exponential",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads

        if time_decay == "exponential":
            self.time_enc = ExponentialTimeDecay(time_dim)
        else:
            self.time_enc = HarmonicTimeEncoder(time_dim)

        in_dim = node_dim + edge_dim + time_dim
        self.msg_proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(out_dim * 2, n_heads)

        self.out_proj = nn.Sequential(
            nn.Linear(out_dim + node_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: Tensor,             # (N, node_dim)
        edge_index: Tensor,    # (2, E) — only past edges (t_e <= query_t)
        edge_feat: Tensor,     # (E, edge_dim)
        time_deltas: Tensor,   # (E,) query_t - t_edge
    ) -> Tensor:
        """
        Parameters
        ----------
        x           : (N, node_dim)
        edge_index  : (2, E)
        edge_feat   : (E, edge_dim)
        time_deltas : (E,) elapsed time since each edge event

        Returns
        -------
        h : (N, out_dim)
        """
        N = x.shape[0]
        E = edge_index.shape[1]

        src_ids = edge_index[0]
        dst_ids = edge_index[1]

        time_enc = self.time_enc(time_deltas)  # (E, time_dim)
        msg_in = torch.cat([x[src_ids], edge_feat, time_enc], dim=-1)
        msgs = self.msg_proj(msg_in)  # (E, out_dim)

        # Attention
        attn_in = torch.cat([msgs, x[dst_ids].expand_as(msgs)], dim=-1)
        attn_logits = self.attn(attn_in)           # (E, n_heads)
        attn_weight = F.softmax(attn_logits, dim=0).mean(dim=-1, keepdim=True)  # (E, 1)

        # Aggregate
        weighted = msgs * attn_weight  # (E, out_dim)
        agg = torch.zeros(N, self.out_dim, device=x.device)
        agg.scatter_add_(0, dst_ids.unsqueeze(-1).expand(-1, self.out_dim), weighted)

        out = self.out_proj(torch.cat([agg, x], dim=-1))
        return self.norm(out)


# ---------------------------------------------------------------------------
# Temporal node classification head
# ---------------------------------------------------------------------------

class TemporalNodeClassifier(nn.Module):
    """
    Temporal node classification: classify each node at each time step.

    Combines TGN embeddings with an MLP classifier for regime detection,
    asset state classification, etc.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, node_embeds: Tensor) -> Tensor:
        """
        Parameters
        ----------
        node_embeds : (N, in_dim) or (T, N, in_dim)

        Returns
        -------
        logits : (N, n_classes) or (T, N, n_classes)
        """
        return self.mlp(node_embeds)

    def predict(self, node_embeds: Tensor) -> Tensor:
        """Return class predictions (argmax of logits)."""
        with torch.no_grad():
            logits = self.forward(node_embeds)
            return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Temporal link prediction head
# ---------------------------------------------------------------------------

class TemporalLinkPredictor(nn.Module):
    """
    Temporal link prediction: predict whether an edge exists at a given time.

    Uses learned similarity function between node pair embeddings,
    combined with time encoding.
    """

    def __init__(
        self,
        node_dim: int,
        time_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_enc = HarmonicTimeEncoder(time_dim)

        in_dim = node_dim * 2 + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        src_embeds: Tensor,  # (B, node_dim)
        dst_embeds: Tensor,  # (B, node_dim)
        time_deltas: Tensor, # (B,)
    ) -> Tensor:
        """
        Returns
        -------
        scores : (B,) link existence probabilities
        """
        time_enc = self.time_enc(time_deltas)
        x = torch.cat([src_embeds, dst_embeds, time_enc], dim=-1)
        return torch.sigmoid(self.mlp(x).squeeze(-1))

    def compute_loss(
        self,
        src_embeds: Tensor,
        dst_embeds: Tensor,
        time_deltas: Tensor,
        labels: Tensor,
    ) -> Tensor:
        probs = self.forward(src_embeds, dst_embeds, time_deltas)
        return F.binary_cross_entropy(probs, labels.float())


# ---------------------------------------------------------------------------
# Full TGN pipeline for financial graphs
# ---------------------------------------------------------------------------

class FinancialTGNPipeline(nn.Module):
    """
    End-to-end temporal GNN pipeline for financial time series.

    Supports:
      - Node classification (regime per asset)
      - Link prediction (correlation edge existence)
      - Graph regression (portfolio-level metrics)
    """

    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        n_node_classes: int = 4,
        memory_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 64,
        time_dim: int = 16,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.tgn = TGN(
            num_nodes=num_nodes,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.node_classifier = TemporalNodeClassifier(
            in_dim=out_dim,
            n_classes=n_node_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.link_predictor = TemporalLinkPredictor(
            node_dim=out_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout,
        )

        # Graph-level regression head (e.g., portfolio return prediction)
        self.graph_regressor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_feat: Tensor,
        edge_times: Tensor,
        query_time: float = 0.0,
    ) -> Dict[str, Tensor]:
        # TGN encoding
        node_embeds = self.tgn(
            node_feat, edge_index, edge_feat, edge_times, query_time
        )

        # Node classification
        node_logits = self.node_classifier(node_embeds)

        # Graph-level embedding (mean pooling)
        graph_embed = node_embeds.mean(dim=0)
        graph_pred = self.graph_regressor(graph_embed.unsqueeze(0)).squeeze()

        return {
            "node_embeds": node_embeds,
            "node_logits": node_logits,
            "graph_pred": graph_pred,
        }

    def predict_links(
        self,
        node_embeds: Tensor,
        src_ids: Tensor,
        dst_ids: Tensor,
        time_deltas: Tensor,
    ) -> Tensor:
        return self.link_predictor(
            node_embeds[src_ids],
            node_embeds[dst_ids],
            time_deltas,
        )

    def update_memory(
        self,
        src_ids: Tensor,
        dst_ids: Tensor,
        edge_feat: Tensor,
        timestamps: Tensor,
    ) -> None:
        self.tgn.update_memory(src_ids, dst_ids, edge_feat, timestamps)


# ---------------------------------------------------------------------------
# Temporal graph dataset (streaming)
# ---------------------------------------------------------------------------

class StreamingTemporalGraphDataset:
    """
    Streaming dataset for temporal graph events.

    Events are (src, dst, timestamp, edge_feat) tuples.
    Supports efficient temporal neighbourhood queries.
    """

    def __init__(self, max_events: int = 100_000):
        self.max_events = max_events
        self._events: List[Tuple[int, int, float, Optional[Tensor]]] = []

    def add_event(
        self,
        src: int,
        dst: int,
        timestamp: float,
        feat: Optional[Tensor] = None,
    ) -> None:
        self._events.append((src, dst, timestamp, feat))
        if len(self._events) > self.max_events:
            self._events.pop(0)

    def get_events_before(
        self,
        t: float,
        node: Optional[int] = None,
    ) -> List[Tuple[int, int, float, Optional[Tensor]]]:
        """Return all events before time t, optionally filtered by node."""
        events = [(s, d, ts, f) for s, d, ts, f in self._events if ts <= t]
        if node is not None:
            events = [(s, d, ts, f) for s, d, ts, f in events if s == node or d == node]
        return events

    def to_tensors(
        self,
        t: float,
        edge_feat_dim: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Convert events before t to tensors for GNN input.

        Returns
        -------
        edge_index : (2, E)
        edge_feat  : (E, edge_feat_dim)
        edge_times : (E,)
        """
        events = self.get_events_before(t)
        if not events:
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, edge_feat_dim),
                torch.zeros(0),
            )

        src_list, dst_list, ts_list, feat_list = zip(*events)
        ei = torch.tensor([list(src_list), list(dst_list)], dtype=torch.long)
        times = torch.tensor(ts_list, dtype=torch.float32)

        if feat_list[0] is not None:
            feats = torch.stack([f for f in feat_list if f is not None])
        else:
            feats = torch.ones(len(events), edge_feat_dim)

        return ei, feats, times


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "HarmonicTimeEncoder",
    "ExponentialTimeDecay",
    "TemporalMemoryModule",
    "TemporalGraphAttention",
    "TGN",
    "GraphSnapshotInterpolator",
    "CTDGConv",
    "TemporalNodeClassifier",
    "TemporalLinkPredictor",
    "FinancialTGNPipeline",
    "StreamingTemporalGraphDataset",
]
