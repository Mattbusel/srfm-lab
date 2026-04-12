"""
communication.py — Agent communication protocols for MARL.

Implements:
- Attention-based message passing (CommNet variant)
- TarMAC: targeted multi-agent communication
- Gated communication with bandwidth constraints
- Noisy channel simulation
- Graph neural network message passing
- Communication topology management
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import layer_init, EPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention-based message passing
# ---------------------------------------------------------------------------

class MultiHeadAttentionComm(nn.Module):
    """
    Multi-head attention for agent-to-agent communication.
    Each agent attends to messages from all other agents.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        msg_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.num_heads = num_heads
        assert msg_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, msg_dim)
        self.k_proj = nn.Linear(hidden_dim, msg_dim)
        self.v_proj = nn.Linear(hidden_dim, msg_dim)
        self.out_proj = layer_init(nn.Linear(msg_dim, hidden_dim), std=0.01)

        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = math.sqrt(msg_dim // num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (N, hidden_dim)
        mask: Optional[torch.Tensor] = None,  # (N, N) communication topology
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (N, hidden_dim) agent hidden states
            mask: (N, N) boolean mask (True = allow communication)
        Returns:
            updated_hidden: (N, hidden_dim)
            attention_weights: (N, N) or (num_heads, N, N)
        """
        N, D = hidden_states.shape
        H = self.num_heads
        head_dim = self.msg_dim // H

        Q = self.q_proj(hidden_states).view(N, H, head_dim).permute(1, 0, 2)  # (H, N, head_dim)
        K = self.k_proj(hidden_states).view(N, H, head_dim).permute(1, 0, 2)
        V = self.v_proj(hidden_states).view(N, H, head_dim).permute(1, 0, 2)

        # Attention scores: (H, N, N)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask: (N, N), True = allowed
            attn_mask = ~mask.unsqueeze(0).expand(H, -1, -1)
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (H, N, N)
        attn_weights = self.attn_drop(attn_weights)

        # Message aggregation: (H, N, head_dim)
        agg = torch.bmm(attn_weights, V)
        agg = agg.permute(1, 0, 2).contiguous().view(N, self.msg_dim)

        # Output projection
        msg_out = self.out_proj(agg)

        # Residual + LayerNorm
        updated = self.norm(hidden_states + msg_out)

        return updated, attn_weights


# ---------------------------------------------------------------------------
# CommNet
# ---------------------------------------------------------------------------

class CommNet(nn.Module):
    """
    CommNet: Communication Neural Net.
    Agents exchange mean-pooled hidden states iteratively.

    Reference: Sukhbaatar et al. 2016.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_comm_steps: int = 2,
        comm_dropout: float = 0.0,
        use_skip: bool = True,
    ):
        super().__init__()
        self.num_comm_steps = num_comm_steps
        self.use_skip = use_skip

        # Communication layers: one per step
        self.comm_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )
            for _ in range(num_comm_steps)
        ])

        self.dropout = nn.Dropout(comm_dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (N, hidden_dim)
        mask: Optional[torch.Tensor] = None,  # (N, N)
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (N, hidden_dim)
            mask: (N, N) communication allowed (1 = yes)
        Returns:
            updated: (N, hidden_dim)
        """
        N = hidden_states.shape[0]
        h = hidden_states

        for comm_layer in self.comm_layers:
            # Compute mean field: aggregate messages from neighbors
            if mask is not None:
                # Masked mean pooling
                mask_f = mask.float()  # (N, N)
                h_expanded = h.unsqueeze(0).expand(N, -1, -1)  # (N, N, D)
                masked = h_expanded * mask_f.unsqueeze(-1)
                denom = mask_f.sum(dim=-1, keepdim=True) + EPS
                mean_field = masked.sum(dim=1) / denom  # (N, D)
            else:
                # Global mean
                mean_field = h.mean(dim=0, keepdim=True).expand(N, -1)

            mean_field = self.dropout(mean_field)
            combined = torch.cat([h, mean_field], dim=-1)
            h_new = comm_layer(combined)
            if self.use_skip:
                h = h + h_new
            else:
                h = h_new

        return h


# ---------------------------------------------------------------------------
# TarMAC: Targeted Multi-Agent Communication
# ---------------------------------------------------------------------------

class TarMACProtocol(nn.Module):
    """
    TarMAC: Signature-based targeted communication.

    Each agent generates a (message, key) pair.
    Receivers use soft attention over keys to aggregate messages.

    Reference: Das et al. 2019.
    """

    def __init__(
        self,
        hidden_dim: int,
        msg_dim: int = 64,
        key_dim: int = 16,
        num_rounds: int = 2,
    ):
        super().__init__()
        self.msg_dim = msg_dim
        self.key_dim = key_dim
        self.num_rounds = num_rounds

        # Message generators
        self.msg_encoder = nn.Linear(hidden_dim, msg_dim)
        self.key_encoder = nn.Linear(hidden_dim, key_dim)

        # Query (receiver side)
        self.query_encoder = nn.Linear(hidden_dim, key_dim)

        # Aggregation
        self.agg_net = nn.Sequential(
            nn.Linear(hidden_dim + msg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (N, hidden_dim)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            updated: (N, hidden_dim)
            attention_weights: (N, N)
        """
        N = hidden_states.shape[0]

        h = hidden_states
        last_attn = torch.zeros(N, N, device=h.device)

        for _ in range(self.num_rounds):
            # Generate messages and keys for each agent (sender)
            msgs = self.msg_encoder(h)      # (N, msg_dim)
            keys = self.key_encoder(h)      # (N, key_dim)
            queries = self.query_encoder(h) # (N, key_dim)

            # Attention: (N_recv, N_send)
            scores = torch.mm(queries, keys.T) / math.sqrt(self.key_dim)

            if mask is not None:
                scores = scores.masked_fill(~mask.bool(), float("-inf"))

            attn = F.softmax(scores, dim=-1)  # (N, N)
            last_attn = attn

            # Weighted message aggregation
            agg_msg = torch.mm(attn, msgs)  # (N, msg_dim)

            # Update hidden states
            combined = torch.cat([h, agg_msg], dim=-1)
            h = self.agg_net(combined)

        return h, last_attn


# ---------------------------------------------------------------------------
# Noisy channel
# ---------------------------------------------------------------------------

class NoisyChannel(nn.Module):
    """
    Simulates a noisy communication channel between agents.

    - Gaussian additive noise
    - Random dropout (packet loss)
    - Bandwidth constraints (quantization)
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        dropout_prob: float = 0.05,
        quantization_bits: Optional[int] = None,
        bandwidth_constraint: Optional[float] = None,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.quantization_bits = quantization_bits
        self.bandwidth_constraint = bandwidth_constraint

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """
        Args:
            messages: (N, msg_dim) or (B, N, msg_dim)
        Returns:
            noisy_messages: same shape
        """
        if not self.training:
            return messages

        result = messages.clone()

        # Additive Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(messages) * self.noise_std
            result = result + noise

        # Packet loss (random dropout of entire messages)
        if self.dropout_prob > 0:
            if messages.dim() == 2:
                mask = torch.rand(messages.shape[0], 1, device=messages.device) > self.dropout_prob
            else:
                mask = torch.rand(messages.shape[0], messages.shape[1], 1, device=messages.device) > self.dropout_prob
            result = result * mask.float()

        # Quantization
        if self.quantization_bits is not None and self.quantization_bits > 0:
            n_levels = 2 ** self.quantization_bits
            result = torch.round(result * n_levels) / n_levels

        # Bandwidth constraint: L2 norm clipping
        if self.bandwidth_constraint is not None:
            norms = result.norm(dim=-1, keepdim=True)
            result = result * torch.clamp(self.bandwidth_constraint / (norms + EPS), max=1.0)

        return result


# ---------------------------------------------------------------------------
# Communication topology
# ---------------------------------------------------------------------------

class CommunicationTopology:
    """
    Manages the communication graph between agents.
    Supports various topologies: full, sparse, ring, k-nearest-neighbor.
    """

    def __init__(
        self,
        num_agents: int,
        topology: str = "full",
        k_nearest: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.num_agents = num_agents
        self.topology = topology
        self.k_nearest = k_nearest
        self.device = device or torch.device("cpu")

        self._adjacency = self._build_topology(topology)

    def _build_topology(self, topology: str) -> np.ndarray:
        N = self.num_agents
        adj = np.zeros((N, N), dtype=bool)

        if topology == "full":
            adj = np.ones((N, N), dtype=bool)
            np.fill_diagonal(adj, False)
        elif topology == "ring":
            for i in range(N):
                adj[i, (i - 1) % N] = True
                adj[i, (i + 1) % N] = True
        elif topology == "star":
            adj[0, 1:] = True
            adj[1:, 0] = True
        elif topology == "knn":
            # Each agent connects to k nearest neighbors (circular)
            for i in range(N):
                for j in range(1, self.k_nearest + 1):
                    adj[i, (i + j) % N] = True
                    adj[i, (i - j) % N] = True
        elif topology == "random":
            p = min(0.5, 2 * self.k_nearest / max(N - 1, 1))
            rand = np.random.random((N, N)) < p
            adj = rand | rand.T
            np.fill_diagonal(adj, False)
        elif topology == "none":
            pass  # No communication
        else:
            # Default: full
            adj = np.ones((N, N), dtype=bool)
            np.fill_diagonal(adj, False)

        return adj

    def get_mask(self) -> torch.Tensor:
        return torch.tensor(self._adjacency, dtype=torch.bool, device=self.device)

    def update_dynamic(self, agent_positions: Optional[np.ndarray] = None, radius: float = 1.0) -> None:
        """Update topology based on agent positions (distance-based)."""
        if agent_positions is None:
            return
        N = self.num_agents
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = float(np.linalg.norm(agent_positions[i] - agent_positions[j]))
                    adj[i, j] = dist <= radius
        self._adjacency = adj

    def get_neighbors(self, agent_id: int) -> List[int]:
        return [j for j in range(self.num_agents) if self._adjacency[agent_id, j]]

    def communication_load(self) -> Dict[str, float]:
        N = self.num_agents
        total_edges = int(self._adjacency.sum())
        return {
            "total_edges": float(total_edges),
            "density": float(total_edges) / (N * (N - 1) + EPS),
            "avg_neighbors": float(total_edges) / (N + EPS),
        }


# ---------------------------------------------------------------------------
# Graph attention network for agent relations
# ---------------------------------------------------------------------------

class AgentGraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer for agent relations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        if concat:
            assert out_features % num_heads == 0
            head_out = out_features // num_heads
        else:
            head_out = out_features

        self.W = nn.Linear(in_features, num_heads * head_out, bias=False)
        self.a = nn.Parameter(torch.empty(num_heads, 2 * head_out))
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.data)

        self.head_out = head_out

    def forward(
        self,
        x: torch.Tensor,              # (N, in_features)
        adjacency: torch.Tensor,       # (N, N) bool mask
    ) -> torch.Tensor:
        N = x.shape[0]
        H = self.num_heads
        d = self.head_out

        # Linear transform
        Wx = self.W(x).view(N, H, d)  # (N, H, d)

        # Attention coefficients
        # Concatenate for each pair
        Wx_i = Wx.unsqueeze(1).expand(N, N, H, d)  # (N, N, H, d)
        Wx_j = Wx.unsqueeze(0).expand(N, N, H, d)

        pair = torch.cat([Wx_i, Wx_j], dim=-1)  # (N, N, H, 2d)
        # a: (H, 2d) -> (1, 1, H, 2d)
        a = self.a.unsqueeze(0).unsqueeze(0)
        e = self.leaky_relu((pair * a).sum(dim=-1))  # (N, N, H)

        # Mask
        if adjacency is not None:
            mask = ~adjacency.unsqueeze(-1)  # (N, N, 1)
            e = e.masked_fill(mask, float("-inf"))

        attn = F.softmax(e, dim=1)  # (N, N, H)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.einsum("ijh,jhd->ihd", attn, Wx)  # (N, H, d)

        if self.concat:
            out = out.view(N, H * d)  # (N, out_features)
        else:
            out = out.mean(dim=1)  # (N, d)

        return self.norm(out + x if out.shape[-1] == x.shape[-1] else out)


class AgentRelationGNN(nn.Module):
    """
    Multi-layer Graph Attention Network for agent relationship modeling.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        out_features: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = in_features
        for i in range(num_layers - 1):
            layers.append(
                AgentGraphAttentionLayer(in_d, hidden_dim, num_heads=num_heads)
            )
            in_d = hidden_dim
        layers.append(
            AgentGraphAttentionLayer(in_d, out_features, num_heads=1, concat=False)
        )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_features)
        adjacency: torch.Tensor,  # (N, N)
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, adjacency))
        return h


# ---------------------------------------------------------------------------
# Complete communication module
# ---------------------------------------------------------------------------

class AgentCommunicationModule(nn.Module):
    """
    Complete communication module supporting multiple protocols.

    Integrates: attention message passing, CommNet, TarMAC, noisy channel,
    and topology management.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_agents: int,
        protocol: str = "attention",  # "attention", "commnet", "tarmac", "gnn"
        msg_dim: int = 64,
        num_comm_rounds: int = 2,
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        topology: str = "full",
        k_nearest: int = 3,
        bandwidth_constraint: Optional[float] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.protocol = protocol

        dev = torch.device(device) if device else torch.device("cpu")

        # Topology
        self.topology = CommunicationTopology(
            num_agents=num_agents,
            topology=topology,
            k_nearest=k_nearest,
            device=dev,
        )

        # Noisy channel
        self.channel = NoisyChannel(
            noise_std=noise_std,
            dropout_prob=dropout_prob,
            bandwidth_constraint=bandwidth_constraint,
        )

        # Communication protocol
        if protocol == "attention":
            self.comm_layer = MultiHeadAttentionComm(
                hidden_dim=hidden_dim, msg_dim=msg_dim, num_heads=4
            )
        elif protocol == "commnet":
            self.comm_layer = CommNet(
                hidden_dim=hidden_dim, num_comm_steps=num_comm_rounds
            )
        elif protocol == "tarmac":
            self.comm_layer = TarMACProtocol(
                hidden_dim=hidden_dim, msg_dim=msg_dim, num_rounds=num_comm_rounds
            )
        elif protocol == "gnn":
            self.comm_layer = AgentRelationGNN(
                in_features=hidden_dim, hidden_dim=hidden_dim, out_features=hidden_dim
            )
        else:
            self.comm_layer = CommNet(hidden_dim=hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (N, hidden_dim) or (B, N, hidden_dim)
        dynamic_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply communication to agent hidden states.

        Args:
            hidden_states: agent representations
            dynamic_mask: optional override topology mask
        Returns:
            updated_states: same shape as input
            info: dict with attention weights etc.
        """
        mask = dynamic_mask if dynamic_mask is not None else self.topology.get_mask()
        info = {}

        if hidden_states.dim() == 3:
            # Batch mode: (B, N, D)
            B, N, D = hidden_states.shape
            outs = []
            for b in range(B):
                h = hidden_states[b]  # (N, D)
                h = self.channel(h)
                if self.protocol in ("attention", "tarmac"):
                    h, attn = self.comm_layer(h, mask)
                    if b == 0:
                        info["attention"] = attn
                elif self.protocol == "gnn":
                    h = self.comm_layer(h, mask.float())
                else:
                    h = self.comm_layer(h, mask)
                outs.append(h)
            return torch.stack(outs, dim=0), info
        else:
            # Single (N, D)
            h = self.channel(hidden_states)
            if self.protocol in ("attention", "tarmac"):
                h, attn = self.comm_layer(h, mask)
                info["attention"] = attn
            elif self.protocol == "gnn":
                h = self.comm_layer(h, mask.float())
            else:
                h = self.comm_layer(h, mask)
            return h, info

    def update_topology(self, agent_positions: Optional[np.ndarray] = None) -> None:
        self.topology.update_dynamic(agent_positions)

    def communication_stats(self) -> Dict[str, float]:
        return self.topology.communication_load()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "MultiHeadAttentionComm",
    "CommNet",
    "TarMACProtocol",
    "NoisyChannel",
    "CommunicationTopology",
    "AgentGraphAttentionLayer",
    "AgentRelationGNN",
    "AgentCommunicationModule",
]
