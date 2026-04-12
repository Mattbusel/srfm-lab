"""
graph_transformer.py
====================
Graph Transformer architectures for financial graph neural networks.

Implements:
  - Global attention with structural bias (Graphormer-style)
  - Spatial encoding (shortest path distances)
  - Edge feature attention (bilinear edge-enhanced attention)
  - Multi-scale graph transformer
  - Hybrid local/global attention (local GCN + global transformer)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional encodings for graph structure
# ---------------------------------------------------------------------------

class SpatialEncoding(nn.Module):
    """
    Graphormer-style spatial encoding.

    Encodes shortest path distance (SPD) between node pairs as a
    learnable bias added to attention logits.

    SPD(i, j) = k → attention_bias[k] is added to A[i, j].

    Reference: Ying et al., "Do Transformers Really Perform Bad for Graph Representation?", NeurIPS 2021.
    """

    def __init__(self, max_dist: int = 8, n_heads: int = 8):
        super().__init__()
        self.max_dist = max_dist
        self.n_heads = n_heads
        # +2: 0=unreachable/same, 1..max_dist=dist, max_dist+1=unreachable-sentinel
        self.dist_bias = nn.Embedding(max_dist + 2, n_heads)
        nn.init.zeros_(self.dist_bias.weight)

    def forward(self, dist_matrix: Tensor) -> Tensor:
        """
        Parameters
        ----------
        dist_matrix : (N, N) integer SPD matrix (0 = self, -1 = unreachable)

        Returns
        -------
        bias : (N, N, n_heads) attention bias
        """
        N = dist_matrix.shape[0]
        clamped = dist_matrix.clamp(0, self.max_dist + 1).long()
        bias = self.dist_bias(clamped)  # (N, N, n_heads)
        return bias

    @staticmethod
    def compute_spd_matrix(edge_index: Tensor, num_nodes: int, max_dist: int = 8) -> Tensor:
        """
        Compute all-pairs shortest path distances via BFS.

        Returns int tensor of shape (N, N); unreachable = max_dist+1.
        """
        N = num_nodes
        adj = [[] for _ in range(N)]
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if 0 <= i < N and 0 <= j < N:
                adj[i].append(j)
                adj[j].append(i)

        dist = torch.full((N, N), max_dist + 1, dtype=torch.long)
        dist.fill_diagonal_(0)

        for start in range(N):
            visited = {start: 0}
            queue = [start]
            head = 0
            while head < len(queue):
                node = queue[head]; head += 1
                d = visited[node]
                if d >= max_dist:
                    continue
                for nbr in adj[node]:
                    if nbr not in visited:
                        visited[nbr] = d + 1
                        queue.append(nbr)
                        dist[start, nbr] = d + 1

        return dist


class DegreeEncoding(nn.Module):
    """
    Encode node degree as a learnable embedding (Graphormer).

    Adds a node-level bias to query and key projections.
    """

    def __init__(self, max_degree: int = 64, embed_dim: int = 128):
        super().__init__()
        self.max_degree = max_degree
        self.in_degree_embed = nn.Embedding(max_degree + 1, embed_dim)
        self.out_degree_embed = nn.Embedding(max_degree + 1, embed_dim)
        nn.init.zeros_(self.in_degree_embed.weight)
        nn.init.zeros_(self.out_degree_embed.weight)

    def forward(self, in_degree: Tensor, out_degree: Tensor) -> Tensor:
        """
        Parameters
        ----------
        in_degree, out_degree : (N,) integer degree tensors

        Returns
        -------
        degree_embed : (N, embed_dim)
        """
        in_d = in_degree.clamp(0, self.max_degree)
        out_d = out_degree.clamp(0, self.max_degree)
        return self.in_degree_embed(in_d) + self.out_degree_embed(out_d)


class EdgeFeatureEncoding(nn.Module):
    """
    Encode edge features as attention bias (Graphormer edge encoding).

    For each pair (i, j), encodes the mean of edge features along the
    shortest path (or direct edge feature if adjacent).
    """

    def __init__(self, edge_feat_dim: int, n_heads: int = 8):
        super().__init__()
        self.edge_proj = nn.Linear(edge_feat_dim, n_heads)

    def forward(
        self,
        edge_index: Tensor,
        edge_attr: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """
        Produce a dense (N, N, n_heads) edge bias matrix.

        Parameters
        ----------
        edge_index : (2, E)
        edge_attr  : (E, edge_feat_dim)
        num_nodes  : N

        Returns
        -------
        edge_bias : (N, N, n_heads) — 0 for non-adjacent pairs
        """
        N = num_nodes
        n_heads = self.edge_proj.out_features
        bias = torch.zeros(N, N, n_heads, device=edge_attr.device)

        proj = self.edge_proj(edge_attr)  # (E, n_heads)
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if 0 <= i < N and 0 <= j < N:
                bias[i, j] += proj[k]

        return bias


# ---------------------------------------------------------------------------
# Graph Transformer layer (Graphormer-style)
# ---------------------------------------------------------------------------

class GraphormerLayer(nn.Module):
    """
    Graphormer attention layer with structural biases.

    Full O(N²) attention over all node pairs, with:
      - Spatial encoding (SPD bias)
      - Edge feature encoding (path encoding)
      - Degree encoding (node-level bias)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x         : (N, d_model)
        attn_bias : (N, N, n_heads) optional structural bias

        Returns
        -------
        out : (N, d_model)
        """
        N = x.shape[0]

        # Pre-norm
        h = self.norm1(x)

        Q = self.q_proj(h).view(N, self.n_heads, self.head_dim).transpose(0, 1)  # (H, N, D)
        K = self.k_proj(h).view(N, self.n_heads, self.head_dim).transpose(0, 1)  # (H, N, D)
        V = self.v_proj(h).view(N, self.n_heads, self.head_dim).transpose(0, 1)  # (H, N, D)

        # Attention logits: (H, N, N)
        attn_logits = torch.bmm(Q, K.transpose(-1, -2)) / self.scale  # (H, N, N)

        # Add structural bias
        if attn_bias is not None:
            attn_bias_t = attn_bias.permute(2, 0, 1)  # (H, N, N)
            attn_logits = attn_logits + attn_bias_t

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Aggregate: (H, N, D) → (N, d_model)
        out = torch.bmm(attn_weights, V)  # (H, N, D)
        out = out.transpose(0, 1).contiguous().view(N, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual + FFN
        x = x + out
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Graphormer model
# ---------------------------------------------------------------------------

class GraphormerModel(nn.Module):
    """
    Full Graphormer model for financial graphs.

    Architecture:
      1. Input embedding (node features + degree encoding)
      2. Stack of Graphormer layers
      3. [CLS] token for graph-level tasks (optional)
      4. Output heads
    """

    def __init__(
        self,
        node_feat_dim: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 512,
        max_degree: int = 64,
        max_dist: int = 8,
        edge_feat_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.max_dist = max_dist

        self.node_proj = nn.Linear(node_feat_dim, d_model)
        self.degree_enc = DegreeEncoding(max_degree, d_model)
        self.spatial_enc = SpatialEncoding(max_dist, n_heads)

        if edge_feat_dim is not None:
            self.edge_enc = EdgeFeatureEncoding(edge_feat_dim, n_heads)
        else:
            self.edge_enc = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            GraphormerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,              # (N, node_feat_dim)
        edge_index: Tensor,     # (2, E)
        edge_attr: Optional[Tensor] = None,  # (E, edge_feat_dim)
        dist_matrix: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x           : (N, node_feat_dim)
        edge_index  : (2, E)
        edge_attr   : (E, edge_feat_dim) optional
        dist_matrix : (N, N) precomputed SPD optional

        Returns
        -------
        node_repr  : (N, d_model) or (N+1, d_model) with CLS
        graph_repr : (d_model,) — CLS token or mean pooled
        """
        N = x.shape[0]

        # Node embedding
        h = self.node_proj(x)

        # Degree encoding
        in_deg = torch.zeros(N, dtype=torch.long, device=x.device)
        out_deg = torch.zeros(N, dtype=torch.long, device=x.device)
        if edge_index.shape[1] > 0:
            dst = edge_index[1]
            src = edge_index[0]
            in_deg.scatter_add_(0, dst, torch.ones(edge_index.shape[1], dtype=torch.long, device=x.device))
            out_deg.scatter_add_(0, src, torch.ones(edge_index.shape[1], dtype=torch.long, device=x.device))
        h = h + self.degree_enc(in_deg, out_deg)

        # CLS token
        if self.use_cls_token:
            cls = self.cls_token.expand(1, -1)
            h = torch.cat([cls, h], dim=0)
            N_full = N + 1
        else:
            N_full = N

        # Structural biases
        attn_bias = None
        if dist_matrix is not None:
            if self.use_cls_token:
                # Pad dist_matrix for CLS token (distance 0 to all)
                pad_row = torch.zeros(1, N, dtype=torch.long, device=x.device)
                pad_col = torch.zeros(N + 1, 1, dtype=torch.long, device=x.device)
                dist_padded = torch.cat([pad_row, dist_matrix], dim=0)
                dist_padded = torch.cat([pad_col, dist_padded], dim=1)
            else:
                dist_padded = dist_matrix
            attn_bias = self.spatial_enc(dist_padded)

        if self.edge_enc is not None and edge_attr is not None:
            edge_bias = self.edge_enc(edge_index, edge_attr, N)
            if self.use_cls_token:
                pad_row = torch.zeros(1, N, attn_bias.shape[-1], device=x.device)
                pad_col = torch.zeros(N + 1, 1, attn_bias.shape[-1], device=x.device)
                edge_bias = torch.cat([pad_row, edge_bias], dim=0)
                edge_bias = torch.cat([pad_col, edge_bias], dim=1)

            if attn_bias is not None:
                attn_bias = attn_bias + edge_bias
            else:
                attn_bias = edge_bias

        # Transformer layers
        for layer in self.layers:
            h = layer(h, attn_bias)

        h = self.final_norm(h)

        if self.use_cls_token:
            graph_repr = h[0]
            node_repr = h[1:]
        else:
            graph_repr = h.mean(dim=0)
            node_repr = h

        return node_repr, graph_repr


# ---------------------------------------------------------------------------
# Local GCN layer (for hybrid model)
# ---------------------------------------------------------------------------

class LocalGCNLayer(nn.Module):
    """
    Standard message-passing GCN layer for local neighbourhood aggregation.

    Used as the "local" component in HybridGraphTransformer.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x           : (N, in_dim)
        edge_index  : (2, E)
        edge_weight : (E,) optional

        Returns
        -------
        out : (N, out_dim)
        """
        N = x.shape[0]
        E = edge_index.shape[1]

        src_ids = edge_index[0]
        dst_ids = edge_index[1]

        # Normalisation: D^{-1/2} A D^{-1/2}
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst_ids, torch.ones(E, device=x.device))
        deg_inv_sqrt = (deg + 1.0).pow(-0.5)

        # Propagation
        msg = x[src_ids]  # (E, in_dim)
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)
        msg = msg * deg_inv_sqrt[src_ids].unsqueeze(-1)

        agg = torch.zeros(N, x.shape[-1], device=x.device)
        agg.scatter_add_(0, dst_ids.unsqueeze(-1).expand(-1, x.shape[-1]), msg)
        agg = agg * deg_inv_sqrt.unsqueeze(-1)

        # Self-loop
        agg = agg + x

        out = self.lin(agg)
        return self.norm(self.dropout(F.gelu(out)))


# ---------------------------------------------------------------------------
# Hybrid local/global graph transformer
# ---------------------------------------------------------------------------

class HybridGraphTransformer(nn.Module):
    """
    Hybrid local/global attention graph transformer.

    At each layer:
      1. Local GCN: message passing over sparse financial graph edges
      2. Global transformer: full O(N²) attention (or top-k approximation)
      3. Gated fusion: learned blend of local and global representations

    This hybrid approach captures both:
      - Direct market relationships (local)
      - Long-range dependencies and systemic effects (global)
    """

    def __init__(
        self,
        node_feat_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 512,
        max_nodes_for_global: int = 256,
        edge_feat_dim: Optional[int] = None,
        dropout: float = 0.1,
        top_k_global: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_nodes_for_global = max_nodes_for_global
        self.top_k_global = top_k_global

        self.input_proj = nn.Linear(node_feat_dim, d_model)
        if edge_feat_dim is not None:
            self.edge_proj = nn.Linear(edge_feat_dim, d_model)
        else:
            self.edge_proj = None

        self.local_layers = nn.ModuleList([
            LocalGCNLayer(d_model, d_model, dropout) for _ in range(n_layers)
        ])
        self.global_layers = nn.ModuleList([
            GraphormerLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])

        # Gating: blend local + global
        self.gates = nn.ModuleList([
            nn.Linear(d_model * 2, 1) for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,              # (N, node_feat_dim)
        edge_index: Tensor,     # (2, E)
        edge_attr: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        node_repr  : (N, d_model)
        graph_repr : (d_model,) mean-pooled
        """
        N = x.shape[0]
        h = self.input_proj(x)

        edge_w = None
        if self.edge_proj is not None and edge_attr is not None:
            edge_emb = self.edge_proj(edge_attr)  # (E, d_model)
            edge_w = edge_emb.norm(dim=-1)        # (E,) — scalar weight

        for local_layer, global_layer, gate in zip(
            self.local_layers, self.global_layers, self.gates
        ):
            # Local
            h_local = local_layer(h, edge_index, edge_w)

            # Global (skip if too many nodes)
            if N <= self.max_nodes_for_global:
                h_global = global_layer(h, attn_bias)
            else:
                # Fall back to local only for large graphs
                h_global = h_local

            # Gated fusion
            g = torch.sigmoid(gate(torch.cat([h_local, h_global], dim=-1)))
            h = g * h_local + (1 - g) * h_global

        h = self.final_norm(h)
        return h, h.mean(dim=0)


# ---------------------------------------------------------------------------
# Multi-scale graph transformer
# ---------------------------------------------------------------------------

class MultiScaleGraphTransformer(nn.Module):
    """
    Multi-scale graph transformer for financial networks.

    Operates at multiple graph resolutions:
      - Fine scale: full asset graph
      - Coarse scale: sector/cluster-level graph
      - Cross-scale: learned attention between scales

    Useful for capturing both micro (asset-level) and macro (sector-level)
    market dynamics simultaneously.
    """

    def __init__(
        self,
        node_feat_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_scales: int = 2,
        pooling_ratios: Optional[List[float]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model
        pooling_ratios = pooling_ratios or [0.5] * (n_scales - 1)

        self.input_proj = nn.Linear(node_feat_dim, d_model)

        # Per-scale transformer layers
        self.scale_transformers = nn.ModuleList([
            nn.ModuleList([
                GraphormerLayer(d_model, n_heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ])
            for _ in range(n_scales)
        ])

        # Pooling layers (coarsen graph)
        self.pool_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_scales - 1)
        ])
        self.pooling_ratios = pooling_ratios

        # Cross-scale attention
        self.cross_scale_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_scales - 1)
        ])

        self.final_proj = nn.Linear(d_model * n_scales, d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,           # (N, node_feat_dim)
        edge_index: Tensor,  # (2, E)
        attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        node_repr  : (N, d_model)
        graph_repr : (d_model,)
        """
        h = self.input_proj(x)
        N = x.shape[0]

        scale_reprs = []
        h_current = h

        for scale in range(self.n_scales):
            # Apply transformer layers at this scale
            for layer in self.scale_transformers[scale]:
                h_current = layer(h_current, attn_bias if scale == 0 else None)

            scale_reprs.append(h_current.mean(dim=0))

            # Coarsen for next scale
            if scale < self.n_scales - 1:
                ratio = self.pooling_ratios[scale]
                k = max(1, int(N * ratio))

                # Simple average pooling over groups
                pool_proj = self.pool_layers[scale](h_current)  # (N, d_model)
                n_groups = k
                group_size = max(1, N // n_groups)
                groups = []
                for g in range(n_groups):
                    start = g * group_size
                    end = min(start + group_size, N)
                    groups.append(pool_proj[start:end].mean(dim=0))
                h_coarse = torch.stack(groups, dim=0)  # (k, d_model)

                # Cross-scale attention: coarse attends to fine
                h_coarse_expanded = h_coarse.unsqueeze(0)   # (1, k, d_model)
                h_fine_expanded = h_current.unsqueeze(0)    # (1, N, d_model)
                h_coarse_updated, _ = self.cross_scale_attn[scale](
                    h_coarse_expanded, h_fine_expanded, h_fine_expanded
                )
                h_current = h_coarse_updated.squeeze(0)  # (k, d_model)
                N = h_current.shape[0]
                attn_bias = None  # reset bias for coarser scale

        # Aggregate across scales
        graph_repr_multi = torch.cat(scale_reprs, dim=-1)  # (d_model * n_scales,)
        graph_repr = self.final_norm(self.final_proj(graph_repr_multi))

        # Node representations from finest scale
        node_repr = scale_reprs[0].unsqueeze(0).expand(x.shape[0], -1)

        return node_repr, graph_repr


# ---------------------------------------------------------------------------
# Edge-feature multi-head attention (bilinear)
# ---------------------------------------------------------------------------

class EdgeEnhancedMultiheadAttention(nn.Module):
    """
    Multi-head attention with explicit edge feature incorporation.

    For edge (i, j) with feature e_ij:
      attn(i, j) = (Q_i + f(e_ij))^T K_j / sqrt(d)

    And the value is enhanced:
      out_i = sum_j attn(i,j) * (V_j + g(e_ij))
    """

    def __init__(
        self,
        d_model: int,
        edge_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Edge feature projections for Q bias and V enhancement
        self.edge_q_bias = nn.Linear(edge_dim, d_model)
        self.edge_v_enhance = nn.Linear(edge_dim, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,             # (N, d_model)
        edge_index: Tensor,    # (2, E)
        edge_attr: Tensor,     # (E, edge_dim)
    ) -> Tensor:
        """
        Sparse edge-enhanced attention.

        Returns
        -------
        out : (N, d_model)
        """
        N = x.shape[0]
        E = edge_index.shape[1]

        Q = self.q_proj(x)   # (N, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        src_ids = edge_index[0]
        dst_ids = edge_index[1]

        # Edge feature bias
        e_q_bias = self.edge_q_bias(edge_attr)   # (E, d_model)
        e_v_enh = self.edge_v_enhance(edge_attr)  # (E, d_model)

        # Q for source + edge bias
        Q_src = Q[src_ids] + e_q_bias  # (E, d_model)
        K_dst = K[dst_ids]             # (E, d_model)
        V_src = V[src_ids] + e_v_enh  # (E, d_model)

        # Attention score (dot product along head dim)
        q_h = Q_src.view(E, self.n_heads, self.head_dim)
        k_h = K_dst.view(E, self.n_heads, self.head_dim)
        v_h = V_src.view(E, self.n_heads, self.head_dim)

        scores = (q_h * k_h).sum(dim=-1) / self.scale  # (E, n_heads)
        attn = torch.exp(scores.clamp(-5, 5)).mean(dim=-1, keepdim=True)  # (E, 1)

        # Weighted values
        weighted = (v_h * attn.unsqueeze(-1)).view(E, -1)  # (E, d_model)

        # Aggregate at dst nodes
        out = torch.zeros(N, self.n_heads * self.head_dim, device=x.device)
        norm = torch.zeros(N, 1, device=x.device)
        out.scatter_add_(0, dst_ids.unsqueeze(-1).expand(-1, weighted.shape[-1]), weighted)
        norm.scatter_add_(0, dst_ids.unsqueeze(-1), attn)
        out = out / (norm + 1e-8)

        out = self.out_proj(out)
        return self.norm(self.dropout(out) + x)


# ---------------------------------------------------------------------------
# Financial graph transformer (production model)
# ---------------------------------------------------------------------------

class FinancialGraphTransformer(nn.Module):
    """
    Production-grade graph transformer for financial graph classification.

    Designed for:
      - Market regime classification
      - Systemic risk scoring
      - Portfolio-level graph features

    Architecture:
      1. Node embedding with degree bias
      2. Multi-layer Graphormer
      3. CLS token for graph classification
      4. Task-specific output heads
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: Optional[int] = None,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        n_classes: int = 4,
        max_degree: int = 64,
        max_dist: int = 8,
        dropout: float = 0.1,
        use_hybrid: bool = False,
    ):
        super().__init__()
        self.use_hybrid = use_hybrid

        if use_hybrid:
            self.backbone = HybridGraphTransformer(
                node_feat_dim=node_feat_dim,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ffn_dim=ffn_dim,
                edge_feat_dim=edge_feat_dim,
                dropout=dropout,
            )
        else:
            self.backbone = GraphormerModel(
                node_feat_dim=node_feat_dim,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ffn_dim=ffn_dim,
                max_degree=max_degree,
                max_dist=max_dist,
                edge_feat_dim=edge_feat_dim,
                dropout=dropout,
                use_cls_token=True,
            )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self.risk_regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        dist_matrix: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Returns
        -------
        dict with:
          node_repr  : (N, d_model)
          graph_repr : (d_model,)
          logits     : (n_classes,)
          risk_score : scalar
        """
        node_repr, graph_repr = self.backbone(
            x, edge_index, edge_attr,
            dist_matrix if not self.use_hybrid else None,
        )

        logits = self.classifier(graph_repr)
        risk_score = self.risk_regressor(graph_repr).squeeze(-1)

        return {
            "node_repr": node_repr,
            "graph_repr": graph_repr,
            "logits": logits,
            "risk_score": risk_score,
        }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "SpatialEncoding",
    "DegreeEncoding",
    "EdgeFeatureEncoding",
    "GraphormerLayer",
    "GraphormerModel",
    "LocalGCNLayer",
    "HybridGraphTransformer",
    "MultiScaleGraphTransformer",
    "EdgeEnhancedMultiheadAttention",
    "FinancialGraphTransformer",
]
