"""
heterogeneous_graph.py
======================
Heterogeneous graph support for multi-type financial networks.

Implements:
  - Multiple node types: assets, sectors, exchanges, macro factors
  - Multiple edge types: correlation, causal, hierarchical, cross-asset
  - Heterogeneous Attention Network (HAN)
  - Heterogeneous Graph Transformer (HGT)
  - Type-specific feature transforms
  - Knowledge graph embedding (TransE, RotatE) for financial relationships
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HGTConv, HANConv, Linear
    from torch_geometric.utils import to_undirected
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("torch_geometric not found; heterogeneous GNN layers unavailable.")


# ---------------------------------------------------------------------------
# Node type definitions for financial heterogeneous graphs
# ---------------------------------------------------------------------------

FINANCIAL_NODE_TYPES = ["asset", "sector", "exchange", "macro_factor"]

FINANCIAL_EDGE_TYPES = [
    ("asset", "correlation", "asset"),
    ("asset", "causal", "asset"),
    ("asset", "belongs_to", "sector"),
    ("sector", "contains", "asset"),
    ("asset", "listed_on", "exchange"),
    ("exchange", "hosts", "asset"),
    ("macro_factor", "influences", "asset"),
    ("asset", "influenced_by", "macro_factor"),
    ("sector", "hierarchical", "sector"),
    ("asset", "cross_asset", "asset"),
]


# ---------------------------------------------------------------------------
# Heterogeneous financial graph builder
# ---------------------------------------------------------------------------

class HeterogeneousFinancialGraphBuilder:
    """
    Build a PyG HeteroData object from financial data with multiple node/edge types.
    """

    def __init__(
        self,
        corr_threshold: float = 0.3,
        causal_threshold: float = 0.2,
    ):
        self.corr_threshold = corr_threshold
        self.causal_threshold = causal_threshold

    def build(
        self,
        asset_returns: np.ndarray,         # (T, N_assets)
        sector_returns: np.ndarray,        # (T, N_sectors)
        macro_factors: np.ndarray,         # (T, N_macro)
        asset_sector_map: Dict[int, int],  # asset_id → sector_id
        asset_exchange_map: Dict[int, int], # asset_id → exchange_id
        n_exchanges: int = 3,
        asset_features: Optional[np.ndarray] = None,
        sector_features: Optional[np.ndarray] = None,
        macro_features: Optional[np.ndarray] = None,
    ) -> "HeteroData":
        if not HAS_PYG:
            raise ImportError("torch_geometric required for HeteroData")

        from torch_geometric.data import HeteroData

        data = HeteroData()
        n_assets = asset_returns.shape[1]
        n_sectors = sector_returns.shape[1]
        n_macro = macro_factors.shape[1]

        # Node features
        data["asset"].x = torch.tensor(
            asset_features if asset_features is not None
            else self._default_node_feat(asset_returns),
            dtype=torch.float32,
        )
        data["sector"].x = torch.tensor(
            sector_features if sector_features is not None
            else self._default_node_feat(sector_returns),
            dtype=torch.float32,
        )
        data["exchange"].x = torch.zeros(n_exchanges, 4, dtype=torch.float32)
        data["macro_factor"].x = torch.tensor(
            macro_features if macro_features is not None
            else self._default_node_feat(macro_factors),
            dtype=torch.float32,
        )

        # Asset-Asset correlation edges
        corr = np.corrcoef(asset_returns.T)
        src, dst, ws = [], [], []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if abs(corr[i, j]) >= self.corr_threshold:
                    src.append(i); dst.append(j); ws.append(corr[i, j])
                    src.append(j); dst.append(i); ws.append(corr[j, i])
        if src:
            data["asset", "correlation", "asset"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )
            data["asset", "correlation", "asset"].edge_attr = torch.tensor(
                ws, dtype=torch.float32
            ).unsqueeze(1)

        # Asset-Sector membership edges
        a_src, s_dst = [], []
        for asset_id, sector_id in asset_sector_map.items():
            if asset_id < n_assets and sector_id < n_sectors:
                a_src.append(asset_id)
                s_dst.append(sector_id)
        if a_src:
            data["asset", "belongs_to", "sector"].edge_index = torch.tensor(
                [a_src, s_dst], dtype=torch.long
            )
            data["sector", "contains", "asset"].edge_index = torch.tensor(
                [s_dst, a_src], dtype=torch.long
            )

        # Asset-Exchange listing edges
        ae_src, ae_dst = [], []
        for asset_id, exch_id in asset_exchange_map.items():
            if asset_id < n_assets and exch_id < n_exchanges:
                ae_src.append(asset_id)
                ae_dst.append(exch_id)
        if ae_src:
            data["asset", "listed_on", "exchange"].edge_index = torch.tensor(
                [ae_src, ae_dst], dtype=torch.long
            )

        # Macro → Asset influence edges (via cross-correlation)
        ma_src, ma_dst, ma_ws = [], [], []
        for m in range(n_macro):
            for a in range(n_assets):
                c = float(np.corrcoef(macro_factors[:, m], asset_returns[:, a])[0, 1])
                if abs(c) >= self.causal_threshold:
                    ma_src.append(m)
                    ma_dst.append(a)
                    ma_ws.append(c)
        if ma_src:
            data["macro_factor", "influences", "asset"].edge_index = torch.tensor(
                [ma_src, ma_dst], dtype=torch.long
            )
            data["macro_factor", "influences", "asset"].edge_attr = torch.tensor(
                ma_ws, dtype=torch.float32
            ).unsqueeze(1)

        return data

    def _default_node_feat(self, returns: np.ndarray) -> np.ndarray:
        """Compute simple node features (mean return, std, skew, kurt)."""
        n = returns.shape[1]
        feats = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            r = returns[:, i]
            feats[i, 0] = float(np.mean(r))
            feats[i, 1] = float(np.std(r))
            from scipy import stats as scipy_stats
            if len(r) > 2:
                feats[i, 2] = float(scipy_stats.skew(r))
                feats[i, 3] = float(scipy_stats.kurtosis(r))
        return feats


# ---------------------------------------------------------------------------
# Type-specific feature transform
# ---------------------------------------------------------------------------

class TypeSpecificFeatureTransform(nn.Module):
    """
    Apply type-specific linear projections to heterogeneous node features.

    Maps each node type's raw features into a common embedding space.
    """

    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.transforms = nn.ModuleDict({
            ntype: nn.Linear(dim, out_dim, bias=bias)
            for ntype, dim in node_feat_dims.items()
        })
        self.out_dim = out_dim

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        x_dict : dict mapping node_type → (N_type, feat_dim)

        Returns
        -------
        out_dict : dict mapping node_type → (N_type, out_dim)
        """
        out = {}
        for ntype, x in x_dict.items():
            if ntype in self.transforms:
                out[ntype] = F.relu(self.transforms[ntype](x))
            else:
                # If unknown type, use identity (zero-pad or project from first layer)
                if x.shape[-1] == self.out_dim:
                    out[ntype] = x
                else:
                    out[ntype] = F.pad(x, (0, max(0, self.out_dim - x.shape[-1])))[:, : self.out_dim]
        return out


# ---------------------------------------------------------------------------
# Heterogeneous Attention Network (HAN)
# ---------------------------------------------------------------------------

class HANLayer(nn.Module):
    """
    Heterogeneous Attention Network layer.

    For each target node type, performs:
    1. Message passing along each meta-path (type-specific)
    2. Meta-path-level attention to aggregate across meta-paths

    Reference: Wang et al., "Heterogeneous Graph Attention Network", WWW 2019.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        meta_paths: List[Tuple[str, str, str]],
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.meta_paths = meta_paths
        self.n_heads = n_heads

        # Per-meta-path GCN (simplified: linear transform + attention)
        self.path_transforms = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in meta_paths
        ])

        # Node-level attention (GAT-style)
        self.node_attn = nn.Parameter(torch.Tensor(1, n_heads, out_dim // n_heads * 2))
        nn.init.xavier_normal_(self.node_attn)

        # Semantic attention for meta-path aggregation
        self.semantic_attn_vec = nn.Parameter(torch.Tensor(out_dim))
        nn.init.normal_(self.semantic_attn_vec)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        x_dict          : dict ntype → (N, in_dim)
        edge_index_dict : dict (src, rel, dst) → (2, E)

        Returns
        -------
        out_dict : dict ntype → (N, out_dim)
        """
        # Collect per-meta-path aggregations for each target type
        type_embeddings: Dict[str, List[Tensor]] = {}

        for path_idx, (src_type, rel, dst_type) in enumerate(self.meta_paths):
            etype = (src_type, rel, dst_type)
            if etype not in edge_index_dict:
                continue
            ei = edge_index_dict[etype]
            if src_type not in x_dict or dst_type not in x_dict:
                continue

            src_x = x_dict[src_type]
            dst_x = x_dict[dst_type]
            N_dst = dst_x.shape[0]

            # Simple mean aggregation with transform
            proj_src = self.path_transforms[path_idx](src_x)  # (N_src, out_dim)

            # Aggregate: for each dst node, mean of connected src
            if ei.shape[1] == 0:
                agg = torch.zeros(N_dst, self.out_dim, device=dst_x.device)
            else:
                dst_ids = ei[1]
                src_ids = ei[0]
                src_msgs = proj_src[src_ids]  # (E, out_dim)
                agg = torch.zeros(N_dst, self.out_dim, device=dst_x.device)
                agg.scatter_add_(0, dst_ids.unsqueeze(-1).expand(-1, self.out_dim), src_msgs)
                # Count for mean
                count = torch.zeros(N_dst, device=dst_x.device)
                count.scatter_add_(0, dst_ids, torch.ones(ei.shape[1], device=dst_x.device))
                count = count.clamp(min=1).unsqueeze(-1)
                agg = agg / count

            if dst_type not in type_embeddings:
                type_embeddings[dst_type] = []
            type_embeddings[dst_type].append(agg)

        # Semantic attention aggregation
        out_dict: Dict[str, Tensor] = {}
        for ntype, embeds in type_embeddings.items():
            if len(embeds) == 1:
                out_dict[ntype] = self.norm(F.relu(embeds[0]))
                continue

            # Semantic attention: score each meta-path embedding
            stack = torch.stack(embeds, dim=0)  # (n_paths, N, out_dim)
            scores = torch.einsum("pnd,d->pn", torch.tanh(stack), self.semantic_attn_vec)  # (n_paths, N)
            weights = F.softmax(scores, dim=0)  # (n_paths, N)
            fused = (stack * weights.unsqueeze(-1)).sum(dim=0)  # (N, out_dim)
            out_dict[ntype] = self.norm(F.relu(fused))

        # For node types with no meta-paths, pass through zeros
        for ntype in x_dict:
            if ntype not in out_dict:
                N = x_dict[ntype].shape[0]
                out_dict[ntype] = torch.zeros(N, self.out_dim, device=x_dict[ntype].device)

        return out_dict


class HANModel(nn.Module):
    """
    Full Heterogeneous Attention Network for financial graphs.

    Stacks multiple HAN layers with residual connections.
    """

    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        hidden_dim: int = 128,
        out_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        meta_paths: Optional[List[Tuple[str, str, str]]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.meta_paths = meta_paths or [
            ("asset", "correlation", "asset"),
            ("asset", "belongs_to", "sector"),
            ("macro_factor", "influences", "asset"),
        ]

        self.feature_transform = TypeSpecificFeatureTransform(node_feat_dims, hidden_dim)

        self.han_layers = nn.ModuleList([
            HANLayer(
                in_dim=hidden_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                meta_paths=self.meta_paths,
                n_heads=n_heads,
                dropout=dropout,
            )
            for i in range(n_layers)
        ])

        self.output_heads = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, out_dim)
            for ntype in node_feat_dims.keys()
        })
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        h = self.feature_transform(x_dict)

        for layer in self.han_layers:
            h_new = layer(h, edge_index_dict)
            # Residual connection (where dims match)
            for ntype in h_new:
                if ntype in h and h[ntype].shape == h_new[ntype].shape:
                    h_new[ntype] = h_new[ntype] + h[ntype]
            h = h_new

        out = {
            ntype: self.output_heads[ntype](self.dropout(h[ntype]))
            for ntype in h if ntype in self.output_heads
        }
        return out


# ---------------------------------------------------------------------------
# Heterogeneous Graph Transformer (HGT)
# ---------------------------------------------------------------------------

class HGTLayer(nn.Module):
    """
    Heterogeneous Graph Transformer layer.

    Uses type-specific key/query/value projections and relation-specific
    attention mechanisms.

    Reference: Hu et al., "Heterogeneous Graph Transformer", WWW 2020.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert out_dim % n_heads == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        # Type-specific K, Q, V projections
        self.K_proj = nn.ModuleDict({t: nn.Linear(in_dim, out_dim) for t in node_types})
        self.Q_proj = nn.ModuleDict({t: nn.Linear(in_dim, out_dim) for t in node_types})
        self.V_proj = nn.ModuleDict({t: nn.Linear(in_dim, out_dim) for t in node_types})

        # Relation-specific attention prior
        rel_names = [et[1] for et in edge_types]
        self.rel_attn = nn.ParameterDict({
            r: nn.Parameter(torch.Tensor(n_heads, self.head_dim, self.head_dim))
            for r in set(rel_names)
        })
        for p in self.rel_attn.values():
            nn.init.xavier_uniform_(p)

        self.out_proj = nn.ModuleDict({t: nn.Linear(out_dim, out_dim) for t in node_types})
        self.norms = nn.ModuleDict({t: nn.LayerNorm(out_dim) for t in node_types})
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        x_dict : dict ntype → (N, in_dim)
        edge_index_dict : dict (src, rel, dst) → (2, E)

        Returns
        -------
        out_dict : dict ntype → (N, out_dim)
        """
        # Compute K, Q, V for all types
        K: Dict[str, Tensor] = {}
        Q: Dict[str, Tensor] = {}
        V: Dict[str, Tensor] = {}
        for ntype, x in x_dict.items():
            if ntype in self.K_proj:
                K[ntype] = self.K_proj[ntype](x).view(-1, self.n_heads, self.head_dim)
                Q[ntype] = self.Q_proj[ntype](x).view(-1, self.n_heads, self.head_dim)
                V[ntype] = self.V_proj[ntype](x).view(-1, self.n_heads, self.head_dim)

        # Accumulate messages for each target node type
        accum: Dict[str, Tensor] = {ntype: torch.zeros(x.shape[0], self.out_dim, device=x.device)
                                     for ntype, x in x_dict.items()}
        attn_norm: Dict[str, Tensor] = {ntype: torch.zeros(x.shape[0], device=x.device)
                                         for ntype, x in x_dict.items()}

        for (src_type, rel, dst_type), ei in edge_index_dict.items():
            if ei.shape[1] == 0:
                continue
            if src_type not in K or dst_type not in Q:
                continue

            src_ids, dst_ids = ei[0], ei[1]
            rel_name = rel

            k = K[src_type][src_ids]    # (E, H, D)
            q = Q[dst_type][dst_ids]    # (E, H, D)
            v = V[src_type][src_ids]    # (E, H, D)

            # Relation-specific attention: A = (Q W_r K^T) / sqrt(D)
            if rel_name in self.rel_attn:
                W_r = self.rel_attn[rel_name]  # (H, D, D)
                # q: (E, H, D), W_r: (H, D, D) → (E, H, D)
                qW = torch.einsum("ehd,hdf->ehf", q, W_r)
                attn_score = (qW * k).sum(dim=-1) / self.scale  # (E, H)
            else:
                attn_score = (q * k).sum(dim=-1) / self.scale  # (E, H)

            # Per-edge softmax approximation: exp(score) and accumulate
            attn_exp = torch.exp(attn_score.clamp(-5, 5))  # (E, H)
            attn_exp_mean = attn_exp.mean(dim=-1)  # (E,)

            # Weighted values
            msg = (v * attn_exp.unsqueeze(-1)).view(-1, self.out_dim)  # (E, out_dim)

            # Accumulate at dst
            dst_ids_exp = dst_ids.unsqueeze(-1).expand(-1, self.out_dim)
            accum[dst_type].scatter_add_(0, dst_ids_exp, msg)
            attn_norm[dst_type].scatter_add_(0, dst_ids, attn_exp_mean)

        # Normalise and project
        out_dict: Dict[str, Tensor] = {}
        for ntype, x in x_dict.items():
            norm = attn_norm[ntype].clamp(min=1e-8).unsqueeze(-1)
            agg = accum[ntype] / norm

            if ntype in self.out_proj:
                out = self.out_proj[ntype](self.dropout(agg))
                if ntype in self.norms:
                    out = self.norms[ntype](out + x if x.shape == out.shape else out)
                out_dict[ntype] = out
            else:
                out_dict[ntype] = agg

        return out_dict


class HGTModel(nn.Module):
    """
    Full Heterogeneous Graph Transformer for financial networks.
    """

    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 128,
        out_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_types = node_types

        # Input projection
        self.input_proj = TypeSpecificFeatureTransform(node_feat_dims, hidden_dim)

        # HGT layers
        self.hgt_layers = nn.ModuleList([
            HGTLayer(hidden_dim, hidden_dim, node_types, edge_types, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
            for ntype in node_types
        })

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        h = self.input_proj(x_dict)

        for layer in self.hgt_layers:
            h = layer(h, edge_index_dict)

        return {
            ntype: self.output_proj[ntype](h[ntype])
            for ntype in self.node_types
            if ntype in h and ntype in self.output_proj
        }


# ---------------------------------------------------------------------------
# Knowledge graph embedding for financial relationships
# ---------------------------------------------------------------------------

class FinancialKGEmbedding(nn.Module):
    """
    Knowledge graph embedding for financial entity relationships.

    Supports:
      - TransE: h + r ≈ t
      - RotatE: h ∘ r = t (complex number rotation)

    Entities: assets, sectors, macro factors, etc.
    Relations: correlation, causality, membership, cross-asset, etc.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embed_dim: int = 64,
        model: str = "transe",
        margin: float = 1.0,
        norm: int = 1,
    ):
        super().__init__()
        assert model in ("transe", "rotate")
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = embed_dim
        self.model = model
        self.margin = margin
        self.norm = norm

        if model == "transe":
            self.entity_embed = nn.Embedding(n_entities, embed_dim)
            self.relation_embed = nn.Embedding(n_relations, embed_dim)
        elif model == "rotate":
            # Complex embeddings (2 * embed_dim for real+imaginary)
            self.entity_embed = nn.Embedding(n_entities, embed_dim * 2)
            self.relation_embed = nn.Embedding(n_relations, embed_dim)  # only phase

        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        self.entity_embed.weight.data = F.normalize(self.entity_embed.weight.data, dim=-1)

    def score(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        """
        Compute score for (head, relation, tail) triples.

        Parameters
        ----------
        head, relation, tail : (B,) index tensors

        Returns
        -------
        scores : (B,) — lower = more likely valid triple
        """
        if self.model == "transe":
            return self._transe_score(head, relation, tail)
        elif self.model == "rotate":
            return self._rotate_score(head, relation, tail)
        raise ValueError(f"Unknown model: {self.model}")

    def _transe_score(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        h = F.normalize(self.entity_embed(head), dim=-1)
        r = self.relation_embed(relation)
        t = F.normalize(self.entity_embed(tail), dim=-1)
        if self.norm == 1:
            return (h + r - t).norm(p=1, dim=-1)
        return (h + r - t).norm(p=2, dim=-1)

    def _rotate_score(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        d = self.embed_dim
        h_full = self.entity_embed(head)
        h_re, h_im = h_full[:, :d], h_full[:, d:]

        t_full = self.entity_embed(tail)
        t_re, t_im = t_full[:, :d], t_full[:, d:]

        # Relation as phase: r ∈ [0, 2π]
        phase = self.relation_embed(relation)  # (B, d)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)

        # Complex multiplication: (h_re + ih_im)(r_re + ir_im)
        res_re = h_re * r_re - h_im * r_im
        res_im = h_re * r_im + h_im * r_re

        # Distance to tail
        dist = torch.stack([res_re - t_re, res_im - t_im], dim=-1)
        return dist.norm(dim=-1).sum(dim=-1)

    def margin_loss(
        self,
        pos_head: Tensor,
        pos_rel: Tensor,
        pos_tail: Tensor,
        neg_head: Tensor,
        neg_rel: Tensor,
        neg_tail: Tensor,
    ) -> Tensor:
        """Margin-based (TransE-style) contrastive loss."""
        pos_score = self.score(pos_head, pos_rel, pos_tail)
        neg_score = self.score(neg_head, neg_rel, neg_tail)
        loss = F.relu(self.margin + pos_score - neg_score).mean()
        return loss

    def get_entity_embeddings(self) -> Tensor:
        return self.entity_embed.weight.detach()

    def get_relation_embeddings(self) -> Tensor:
        return self.relation_embed.weight.detach()


# ---------------------------------------------------------------------------
# Heterogeneous graph aggregator
# ---------------------------------------------------------------------------

class HeterogeneousGraphAggregator(nn.Module):
    """
    Aggregate heterogeneous node embeddings into a single graph-level embedding.

    Uses type-specific attention to weight the contribution of each node type.
    """

    def __init__(
        self,
        node_types: List[str],
        node_dim: int,
        out_dim: int,
        pool: str = "attention",
    ):
        super().__init__()
        assert pool in ("mean", "max", "attention", "sum")
        self.pool = pool
        self.node_types = node_types

        if pool == "attention":
            self.type_attn = nn.ModuleDict({
                ntype: nn.Linear(node_dim, 1) for ntype in node_types
            })
            self.node_attn = nn.ModuleDict({
                ntype: nn.Linear(node_dim, 1) for ntype in node_types
            })

        self.output = nn.Linear(node_dim, out_dim)

    def forward(self, x_dict: Dict[str, Tensor]) -> Tensor:
        """
        Parameters
        ----------
        x_dict : dict ntype → (N_type, node_dim)

        Returns
        -------
        graph_embed : (out_dim,)
        """
        type_embeds = []
        for ntype in self.node_types:
            if ntype not in x_dict:
                continue
            x = x_dict[ntype]  # (N, D)

            if self.pool == "mean":
                embed = x.mean(dim=0)
            elif self.pool == "max":
                embed = x.max(dim=0).values
            elif self.pool == "sum":
                embed = x.sum(dim=0)
            elif self.pool == "attention":
                node_scores = self.node_attn[ntype](x)  # (N, 1)
                node_weights = F.softmax(node_scores, dim=0)
                embed = (x * node_weights).sum(dim=0)  # (D,)
            else:
                embed = x.mean(dim=0)

            type_embeds.append(embed)

        if not type_embeds:
            return torch.zeros(self.output.out_features)

        # Type-level aggregation
        stacked = torch.stack(type_embeds, dim=0)  # (n_types, D)

        if self.pool == "attention":
            type_scores = torch.stack([
                self.type_attn[ntype](x_dict[ntype]).mean()
                for ntype in self.node_types if ntype in x_dict
            ])
            type_weights = F.softmax(type_scores, dim=0).unsqueeze(-1)
            fused = (stacked * type_weights).sum(dim=0)
        else:
            fused = stacked.mean(dim=0)

        return self.output(fused)


# ---------------------------------------------------------------------------
# Heterogeneous regime classifier
# ---------------------------------------------------------------------------

class HeterogeneousRegimeClassifier(nn.Module):
    """
    Full pipeline: HeteroData → HGT embeddings → regime classification.

    Suitable for multi-class market regime detection using heterogeneous
    financial networks.
    """

    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        n_regimes: int = 4,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        n_hgt_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_kg_embed: bool = False,
        n_entities: int = 0,
        n_relations: int = 0,
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.use_kg_embed = use_kg_embed

        self.hgt = HGTModel(
            node_feat_dims=node_feat_dims,
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            n_layers=n_hgt_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.aggregator = HeterogeneousGraphAggregator(
            node_types=node_types,
            node_dim=embed_dim,
            out_dim=embed_dim,
            pool="attention",
        )

        if use_kg_embed and n_entities > 0:
            self.kg_embed = FinancialKGEmbedding(n_entities, n_relations, embed_dim)
            classifier_in = embed_dim * 2
        else:
            classifier_in = embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_regimes),
        )

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        entity_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x_dict          : dict ntype → (N, feat_dim)
        edge_index_dict : dict etype → (2, E)
        entity_ids      : optional (N,) for KG embedding lookup

        Returns
        -------
        logits : (n_regimes,)
        """
        node_embeds = self.hgt(x_dict, edge_index_dict)
        graph_embed = self.aggregator(node_embeds)  # (embed_dim,)

        if self.use_kg_embed and entity_ids is not None and hasattr(self, "kg_embed"):
            kg_emb = self.kg_embed.get_entity_embeddings()[entity_ids].mean(dim=0)
            graph_embed = torch.cat([graph_embed, kg_emb])

        logits = self.classifier(graph_embed.unsqueeze(0)).squeeze(0)
        return logits

    def predict_regime(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> int:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index_dict)
            return int(logits.argmax().item())


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "FINANCIAL_NODE_TYPES",
    "FINANCIAL_EDGE_TYPES",
    "HeterogeneousFinancialGraphBuilder",
    "TypeSpecificFeatureTransform",
    "HANLayer",
    "HANModel",
    "HGTLayer",
    "HGTModel",
    "FinancialKGEmbedding",
    "HeterogeneousGraphAggregator",
    "HeterogeneousRegimeClassifier",
]
