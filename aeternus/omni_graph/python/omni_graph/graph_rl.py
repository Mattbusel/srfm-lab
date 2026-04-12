"""
graph_rl.py
===========
Graph-based Reinforcement Learning for financial portfolio management.

Implements:
  - Graph observation encoder for RL agents
  - GNN policy network (actor)
  - GNN value network (critic)
  - Graph-aware exploration strategy
  - Asset selection as node classification
  - Portfolio optimization as graph partitioning
  - PPO training loop for graph-based RL
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal, MultivariateNormal


# ---------------------------------------------------------------------------
# Graph observation encoder
# ---------------------------------------------------------------------------

class GraphObservationEncoder(nn.Module):
    """
    Encode a financial graph snapshot as a fixed-size observation vector
    suitable for RL agent consumption.

    Takes:
      - Node features (per-asset statistics)
      - Edge index and weights
      - Optional graph-level features (VIX, market regime, etc.)

    Outputs:
      - Node-level embeddings (for per-asset decisions)
      - Graph-level embedding (for portfolio-level decisions)
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 1,
        graph_feat_dim: int = 0,
        hidden_dim: int = 128,
        out_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        gnn_type: str = "gat",
    ):
        super().__init__()
        self.gnn_type = gnn_type

        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GNN layers
        if gnn_type == "gat":
            self.gnn_layers = nn.ModuleList([
                GATConvLayer(hidden_dim, hidden_dim, n_heads=n_heads, edge_dim=edge_feat_dim, dropout=dropout)
                for _ in range(n_layers)
            ])
        else:  # gcn
            self.gnn_layers = nn.ModuleList([
                GCNLayer(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(n_layers)
            ])

        self.node_out = nn.Linear(hidden_dim, out_dim)

        # Graph-level pooling
        pool_in = out_dim + graph_feat_dim if graph_feat_dim > 0 else out_dim
        self.graph_pool = nn.Sequential(
            nn.Linear(pool_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        if graph_feat_dim > 0:
            self.graph_feat_proj = nn.Linear(graph_feat_dim, out_dim)

        self.graph_feat_dim = graph_feat_dim
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        graph_feat: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        node_feat  : (N, node_feat_dim)
        edge_index : (2, E)
        edge_attr  : (E, edge_feat_dim) optional
        graph_feat : (graph_feat_dim,) optional global features
        batch      : (N,) batch assignment for multiple graphs

        Returns
        -------
        node_embeds  : (N, out_dim)
        graph_embed  : (out_dim,) or (B, out_dim) if batched
        """
        h = F.gelu(self.node_proj(node_feat))

        for layer in self.gnn_layers:
            if self.gnn_type == "gat":
                h = layer(h, edge_index, edge_attr)
            else:
                h = layer(h, edge_index,
                          edge_attr[:, 0] if edge_attr is not None and edge_attr.dim() > 1
                          else edge_attr)
            h = self.dropout(h)

        node_embeds = self.node_out(h)  # (N, out_dim)

        # Graph-level: mean pooling
        if batch is not None:
            n_graphs = batch.max().item() + 1
            graph_embed = torch.zeros(n_graphs, node_embeds.shape[-1], device=node_embeds.device)
            count = torch.zeros(n_graphs, 1, device=node_embeds.device)
            graph_embed.scatter_add_(0, batch.unsqueeze(-1).expand_as(node_embeds), node_embeds)
            count.scatter_add_(0, batch.unsqueeze(-1), torch.ones(node_embeds.shape[0], 1, device=node_embeds.device))
            graph_embed = graph_embed / (count + 1e-8)
        else:
            graph_embed = node_embeds.mean(dim=0, keepdim=True)  # (1, out_dim)

        if self.graph_feat_dim > 0 and graph_feat is not None:
            gf = self.graph_feat_proj(graph_feat)
            if graph_embed.dim() == 1:
                graph_embed = torch.cat([graph_embed, gf.unsqueeze(0)], dim=-1)
            else:
                gf_exp = gf.unsqueeze(0).expand(graph_embed.shape[0], -1)
                graph_embed = torch.cat([graph_embed, gf_exp], dim=-1)

        graph_embed = self.graph_pool(graph_embed).squeeze(0)
        return node_embeds, graph_embed


# ---------------------------------------------------------------------------
# Lightweight GNN layers for RL
# ---------------------------------------------------------------------------

class GATConvLayer(nn.Module):
    """
    Graph Attention convolution layer (single-layer).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        edge_dim: int = 1,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.lin_src = nn.Linear(in_dim, out_dim)
        self.lin_dst = nn.Linear(in_dim, out_dim)
        self.lin_edge = nn.Linear(edge_dim, n_heads) if edge_dim > 0 else None

        self.attn_src = nn.Parameter(torch.Tensor(1, n_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.Tensor(1, n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src.view(1, -1).unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.view(1, -1).unsqueeze(0))

        self.leaky = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        N = x.shape[0]
        E = edge_index.shape[1]

        src_ids, dst_ids = edge_index[0], edge_index[1]

        x_src = self.lin_src(x).view(N, self.n_heads, self.head_dim)
        x_dst = self.lin_dst(x).view(N, self.n_heads, self.head_dim)

        # Attention: (E, n_heads)
        alpha_src = (x_src[src_ids] * self.attn_src).sum(dim=-1)
        alpha_dst = (x_dst[dst_ids] * self.attn_dst).sum(dim=-1)
        alpha = self.leaky(alpha_src + alpha_dst)  # (E, n_heads)

        if self.lin_edge is not None and edge_attr is not None:
            ea = edge_attr if edge_attr.dim() > 1 else edge_attr.unsqueeze(-1)
            edge_bias = self.lin_edge(ea.float())  # (E, n_heads)
            alpha = alpha + edge_bias

        # Softmax over source for each dst
        alpha_exp = torch.exp(alpha.clamp(-5, 5))  # (E, n_heads)
        alpha_norm = torch.zeros(N, self.n_heads, device=x.device)
        alpha_norm.scatter_add_(
            0,
            dst_ids.unsqueeze(-1).expand(-1, self.n_heads),
            alpha_exp,
        )
        alpha_norm = alpha_norm[dst_ids]  # (E, n_heads)
        attn = alpha_exp / (alpha_norm + 1e-8)  # (E, n_heads)
        attn = self.dropout(attn)

        # Aggregate
        msgs = x_src[src_ids] * attn.unsqueeze(-1)  # (E, n_heads, head_dim)
        msgs_flat = msgs.reshape(E, -1)  # (E, out_dim)

        out = torch.zeros(N, self.n_heads * self.head_dim, device=x.device)
        out.scatter_add_(0, dst_ids.unsqueeze(-1).expand(-1, msgs_flat.shape[-1]), msgs_flat)

        return self.norm(F.gelu(out) + x if out.shape == x.shape else F.gelu(out))


class GCNLayer(nn.Module):
    """Simple GCN layer."""

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
        N = x.shape[0]
        src_ids, dst_ids = edge_index[0], edge_index[1]
        deg = torch.zeros(N, device=x.device).scatter_add_(
            0, dst_ids, torch.ones(edge_index.shape[1], device=x.device)
        )
        deg_inv = (deg + 1.0).pow(-0.5)

        msgs = x[src_ids] * deg_inv[src_ids].unsqueeze(-1)
        if edge_weight is not None:
            msgs = msgs * edge_weight.unsqueeze(-1)

        agg = torch.zeros(N, x.shape[-1], device=x.device)
        agg.scatter_add_(0, dst_ids.unsqueeze(-1).expand(-1, x.shape[-1]), msgs)
        agg = agg * deg_inv.unsqueeze(-1) + x

        return self.norm(self.dropout(F.gelu(self.lin(agg))))


# ---------------------------------------------------------------------------
# GNN policy network (actor)
# ---------------------------------------------------------------------------

class GNNActor(nn.Module):
    """
    GNN-based actor (policy network) for financial RL.

    Supports:
      1. Continuous action: portfolio weights ∈ Δ^N (simplex)
      2. Discrete action: asset selection (binary per node)

    Input: graph observation (node + edge features)
    Output: action distribution parameters
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 1,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        action_type: str = "continuous",
        n_assets: Optional[int] = None,
        n_actions: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert action_type in ("continuous", "discrete", "per_asset_discrete")
        self.action_type = action_type
        self.n_assets = n_assets

        self.encoder = GraphObservationEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            dropout=dropout,
        )

        if action_type == "continuous":
            # Output: mean and log_std for each asset weight
            self.mean_head = nn.Linear(embed_dim, n_assets or 1)
            self.log_std = nn.Parameter(torch.zeros(n_assets or 1))

        elif action_type == "discrete":
            # Output: logits for n_actions portfolio states
            self.logit_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_actions),
            )

        elif action_type == "per_asset_discrete":
            # Per-node binary decision (include/exclude from portfolio)
            self.node_classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2),  # include / exclude
            )

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Union[Normal, Categorical]:
        """
        Returns
        -------
        dist : action distribution (Normal for continuous, Categorical for discrete)
        """
        node_embeds, graph_embed = self.encoder(node_feat, edge_index, edge_attr)

        if self.action_type == "continuous":
            mean = self.mean_head(graph_embed)
            # Portfolio weights via softmax (simplex)
            mean = F.softmax(mean, dim=-1)
            std = torch.exp(self.log_std.clamp(-4, 2))
            return Normal(mean, std)

        elif self.action_type == "discrete":
            logits = self.logit_head(graph_embed)
            return Categorical(logits=logits)

        elif self.action_type == "per_asset_discrete":
            logits = self.node_classifier(node_embeds)  # (N, 2)
            return Categorical(logits=logits)

    def get_action(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample an action and return (action, log_prob).
        """
        dist = self.forward(node_feat, edge_index, edge_attr)

        if deterministic:
            if isinstance(dist, Normal):
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
            log_prob = dist.log_prob(action).sum()
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

        return action, log_prob


# ---------------------------------------------------------------------------
# GNN value network (critic)
# ---------------------------------------------------------------------------

class GNNCritic(nn.Module):
    """
    GNN-based critic (value network) for financial RL.

    Estimates V(s) — the expected return from the current graph state.
    Can also estimate Q(s, a) for actor-critic methods.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 1,
        action_dim: int = 0,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphObservationEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            dropout=dropout,
        )

        value_in = embed_dim + action_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        action: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns
        -------
        value : scalar
        """
        _, graph_embed = self.encoder(node_feat, edge_index, edge_attr)

        if action is not None:
            x = torch.cat([graph_embed, action.flatten()])
        else:
            x = graph_embed

        return self.value_head(x.unsqueeze(0)).squeeze()


# ---------------------------------------------------------------------------
# Graph-aware exploration strategy
# ---------------------------------------------------------------------------

class GraphAwareExploration:
    """
    Graph-aware exploration strategy for financial RL.

    Uses graph structure to guide exploration:
      1. Novelty bonus: prefer visiting graph states with high structural dissimilarity
         to previously seen states
      2. Community-based exploration: explore nodes within each community
      3. Uncertainty-based: prefer assets with high embedding uncertainty

    Inspired by: curiosity-driven exploration and count-based methods.
    """

    def __init__(
        self,
        novelty_bonus_scale: float = 0.1,
        community_explore_prob: float = 0.2,
        memory_size: int = 100,
        embed_dim: int = 64,
    ):
        self.novelty_bonus_scale = novelty_bonus_scale
        self.community_explore_prob = community_explore_prob
        self._state_memory: deque = deque(maxlen=memory_size)
        self._visit_counts: Dict[int, int] = {}

    def compute_novelty_bonus(self, graph_embed: Tensor) -> float:
        """
        Compute novelty bonus for current graph state.
        High novelty → high bonus.
        """
        if not self._state_memory:
            self._state_memory.append(graph_embed.detach())
            return float(self.novelty_bonus_scale)

        # Distance to nearest neighbour in memory
        memory_tensor = torch.stack(list(self._state_memory))
        dists = torch.norm(memory_tensor - graph_embed.unsqueeze(0), dim=-1)
        min_dist = float(dists.min())

        self._state_memory.append(graph_embed.detach())
        return float(min_dist * self.novelty_bonus_scale)

    def ucb_node_bonus(
        self,
        node_ids: List[int],
        t: int,
        c: float = 1.0,
    ) -> np.ndarray:
        """
        UCB-style exploration bonus per node.

        bonus_i = c * sqrt(2 * log(t) / (n_visits_i + 1))
        """
        bonuses = np.zeros(len(node_ids))
        for k, nid in enumerate(node_ids):
            n_visits = self._visit_counts.get(nid, 0)
            bonuses[k] = c * math.sqrt(2 * math.log(max(t, 1)) / (n_visits + 1))
        return bonuses

    def record_visit(self, node_ids: List[int]) -> None:
        for nid in node_ids:
            self._visit_counts[nid] = self._visit_counts.get(nid, 0) + 1

    def select_exploration_nodes(
        self,
        node_embeds: Tensor,
        n_select: int,
        t: int,
    ) -> List[int]:
        """
        Select nodes for exploration based on UCB scores.
        """
        N = node_embeds.shape[0]
        bonuses = self.ucb_node_bonus(list(range(N)), t)
        selected = np.argsort(bonuses)[::-1][:n_select].tolist()
        return selected


# ---------------------------------------------------------------------------
# Asset selection as node classification
# ---------------------------------------------------------------------------

class AssetSelectionPolicy(nn.Module):
    """
    Asset selection policy: treat each asset as a node and decide
    whether to include it in the portfolio (binary classification per node).

    Action space: {0, 1}^N — binary portfolio mask.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 1,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        long_short: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.long_short = long_short

        self.encoder = GraphObservationEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            dropout=dropout,
        )

        n_classes = 3 if long_short else 2  # long / neutral / short OR include / exclude
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        logits    : (N, n_classes)
        portfolio_weights : (N,) — softmax over selected assets
        """
        node_embeds, _ = self.encoder(node_feat, edge_index, edge_attr)
        logits = self.classifier(node_embeds)  # (N, 2 or 3)

        # Soft portfolio weights
        probs = F.softmax(logits, dim=-1)
        if self.long_short:
            # long_weight - short_weight
            weights = probs[:, 0] - probs[:, 2]
        else:
            weights = probs[:, 0]  # P(include)

        # Normalise to valid portfolio (L1 norm)
        weights = weights / (weights.abs().sum() + 1e-8)
        return logits, weights

    def get_portfolio(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Return portfolio weights, optionally top-k by magnitude.
        """
        _, weights = self.forward(node_feat, edge_index, edge_attr)
        if top_k is not None:
            topk_idx = weights.abs().topk(top_k).indices
            mask = torch.zeros_like(weights)
            mask[topk_idx] = 1.0
            weights = weights * mask
            weights = weights / (weights.abs().sum() + 1e-8)
        return weights


# ---------------------------------------------------------------------------
# Portfolio optimization as graph partitioning
# ---------------------------------------------------------------------------

class GraphPartitionPortfolioOptimizer(nn.Module):
    """
    Frame portfolio optimization as a graph partitioning problem.

    The financial graph is partitioned into K sub-portfolios (clusters).
    Weights are allocated across partitions based on risk/return profile.

    Uses a GNN-based partition score followed by softmax allocation.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 1,
        n_partitions: int = 5,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_partitions = n_partitions

        self.encoder = GraphObservationEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            dropout=dropout,
        )

        # Assign each node to a partition (soft assignment)
        self.partition_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_partitions),
        )

        # Partition-level weight allocation
        self.allocation_head = nn.Sequential(
            nn.Linear(embed_dim + n_partitions, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_partitions),
        )

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Returns
        -------
        dict with:
          node_partition_probs : (N, K) soft assignment
          partition_weights    : (K,) allocation weights
          node_weights         : (N,) final portfolio weights
        """
        node_embeds, graph_embed = self.encoder(node_feat, edge_index, edge_attr)

        # Node partition assignment: (N, K)
        partition_logits = self.partition_head(node_embeds)
        partition_probs = F.softmax(partition_logits, dim=-1)

        # Partition representatives: weighted mean of node embeds
        partition_embeds = partition_probs.T @ node_embeds  # (K, embed_dim)
        partition_embeds_mean = partition_embeds.mean(dim=0)  # (embed_dim,)

        # Allocation across partitions
        alloc_in = torch.cat([graph_embed, partition_probs.mean(dim=0)], dim=-1)
        partition_weights = F.softmax(self.allocation_head(alloc_in), dim=-1)  # (K,)

        # Node weights: expectation under partition assignment
        node_weights = (partition_probs * partition_weights.unsqueeze(0)).sum(dim=-1)  # (N,)
        node_weights = node_weights / (node_weights.sum() + 1e-8)

        return {
            "node_partition_probs": partition_probs,
            "partition_weights": partition_weights,
            "node_weights": node_weights,
        }


# ---------------------------------------------------------------------------
# Replay buffer for graph RL
# ---------------------------------------------------------------------------

@dataclass
class GraphTransition:
    """Single transition for graph RL replay buffer."""
    node_feat: Tensor
    edge_index: Tensor
    edge_attr: Optional[Tensor]
    action: Tensor
    reward: float
    next_node_feat: Tensor
    next_edge_index: Tensor
    next_edge_attr: Optional[Tensor]
    done: bool


class GraphReplayBuffer:
    """
    Replay buffer for graph-based RL.

    Stores graph transitions with potentially different topologies
    (handles variable-size graphs).
    """

    def __init__(self, capacity: int = 10_000, seed: Optional[int] = None):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, transition: GraphTransition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[GraphTransition]:
        idx = self.rng.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), replace=False)
        return [self._buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) >= 64


# ---------------------------------------------------------------------------
# PPO trainer for graph RL
# ---------------------------------------------------------------------------

class GraphPPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer for graph-based RL agents.

    Designed for portfolio management tasks where:
      - States are financial graph snapshots
      - Actions are portfolio weight vectors or asset selections
      - Rewards are risk-adjusted portfolio returns

    Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
    """

    def __init__(
        self,
        actor: GNNActor,
        critic: GNNCritic,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        gamma: float = 0.99,
        lam: float = 0.95,
        n_epochs: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
    ):
        self.actor = actor
        self.critic = critic
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.gamma = gamma
        self.lam = lam
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=lr_critic)

        self._rollout_buffer: List[Dict] = []

    def collect_rollout_step(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        reward: float,
        done: bool,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Collect one environment step.

        Returns (action, log_prob, value).
        """
        with torch.no_grad():
            action, log_prob = self.actor.get_action(node_feat, edge_index, edge_attr)
            value = self.critic(node_feat, edge_index, edge_attr)

        self._rollout_buffer.append({
            "node_feat": node_feat,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
            "done": done,
        })

        return action, log_prob, value

    def compute_advantages(
        self,
        values: List[float],
        rewards: List[float],
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE (Generalized Advantage Estimation)."""
        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            not_done = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * not_done - values[t]
            gae = delta + self.gamma * self.lam * not_done * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """
        Run PPO update on collected rollout.

        Returns dict of training metrics.
        """
        if not self._rollout_buffer:
            return {}

        buffer = self._rollout_buffer
        n = len(buffer)

        values = [float(b["value"]) for b in buffer]
        rewards = [b["reward"] for b in buffer]
        dones = [b["done"] for b in buffer]

        advantages, returns = self.compute_advantages(values, rewards, dones)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        for epoch in range(self.n_epochs):
            indices = torch.randperm(n)

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                for idx in batch_idx:
                    b = buffer[int(idx)]
                    adv = advantages_t[idx]
                    ret = returns_t[idx]

                    # Actor update
                    dist = self.actor(b["node_feat"], b["edge_index"], b.get("edge_attr"))
                    new_log_prob = dist.log_prob(b["action"]).sum()

                    ratio = torch.exp(new_log_prob - b["log_prob"])
                    clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    actor_loss = -torch.min(ratio * adv, clipped * adv)

                    entropy = dist.entropy().sum() if hasattr(dist, "entropy") else torch.tensor(0.0)

                    # Critic update
                    value = self.critic(b["node_feat"], b["edge_index"], b.get("edge_attr"))
                    critic_loss = F.mse_loss(value, ret)

                    total_loss = (
                        actor_loss
                        - self.entropy_coeff * entropy
                        + self.value_coeff * critic_loss
                    )

                    self.actor_optim.zero_grad()
                    self.critic_optim.zero_grad()
                    total_loss.backward()

                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                    self.actor_optim.step()
                    self.critic_optim.step()

                    metrics["actor_loss"] += float(actor_loss) / (n * self.n_epochs)
                    metrics["critic_loss"] += float(critic_loss) / (n * self.n_epochs)
                    metrics["entropy"] += float(entropy) / (n * self.n_epochs)

        self._rollout_buffer.clear()
        return metrics


# ---------------------------------------------------------------------------
# Portfolio environment (graph observation wrapper)
# ---------------------------------------------------------------------------

class PortfolioGraphEnvironment:
    """
    Gym-style environment for portfolio management with graph observations.

    State: financial graph snapshot at time t
    Action: portfolio weights w ∈ Δ^N
    Reward: risk-adjusted return (Sharpe ratio proxy)
    """

    def __init__(
        self,
        returns: np.ndarray,
        graph_sequence: List[Tuple[Tensor, Tensor, Optional[Tensor]]],
        risk_free_rate: float = 0.0,
        transaction_cost: float = 0.001,
        risk_penalty: float = 0.1,
    ):
        self.returns = returns
        self.graph_sequence = graph_sequence
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty

        self.T = len(graph_sequence)
        self.N = returns.shape[1]
        self.t = 0
        self.prev_weights = np.ones(self.N) / self.N  # equal weight initially
        self.portfolio_values = [1.0]

    def reset(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        self.t = 0
        self.prev_weights = np.ones(self.N) / self.N
        self.portfolio_values = [1.0]
        return self.graph_sequence[0]

    def step(
        self,
        weights: np.ndarray,
    ) -> Tuple[Tuple[Tensor, Tensor, Optional[Tensor]], float, bool, Dict]:
        """
        Execute action (portfolio weights) and return (next_obs, reward, done, info).
        """
        weights = np.clip(weights, 0, None)
        weights = weights / (weights.sum() + 1e-8)

        # Compute portfolio return
        r = self.returns[self.t]
        port_return = float(np.dot(weights, r))

        # Transaction costs
        turnover = float(np.abs(weights - self.prev_weights).sum())
        tc = turnover * self.transaction_cost

        # Risk penalty (variance proxy)
        volatility = float(np.std(r))
        risk_pen = self.risk_penalty * volatility

        reward = port_return - tc - risk_pen

        # Update portfolio value
        new_val = self.portfolio_values[-1] * (1 + port_return - tc)
        self.portfolio_values.append(new_val)
        self.prev_weights = weights.copy()

        self.t += 1
        done = self.t >= self.T - 1

        if not done:
            next_obs = self.graph_sequence[self.t]
        else:
            next_obs = self.graph_sequence[-1]

        info = {
            "port_return": port_return,
            "turnover": turnover,
            "transaction_cost": tc,
            "portfolio_value": new_val,
        }

        return next_obs, reward, done, info

    def episode_stats(self) -> Dict[str, float]:
        vals = np.array(self.portfolio_values)
        returns = np.diff(vals) / (vals[:-1] + 1e-8)
        total_return = float(vals[-1] / vals[0] - 1)
        sharpe = float(
            returns.mean() / (returns.std() + 1e-8) * math.sqrt(252)
            if len(returns) > 1 else 0.0
        )
        cum = vals
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / (peak + 1e-8)
        max_dd = float(np.min(dd))

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "n_steps": self.t,
        }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "GraphObservationEncoder",
    "GATConvLayer",
    "GCNLayer",
    "GNNActor",
    "GNNCritic",
    "GraphAwareExploration",
    "AssetSelectionPolicy",
    "GraphPartitionPortfolioOptimizer",
    "GraphTransition",
    "GraphReplayBuffer",
    "GraphPPOTrainer",
    "PortfolioGraphEnvironment",
]
