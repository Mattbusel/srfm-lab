"""
Message Passing infrastructure for agent-to-agent communication.

Components:
  - AgentCommunicationGraph: directed graph of communication links
  - MessagePassing:          agents broadcast signals to neighbors
  - AttentionAggregation:    agents aggregate neighbor messages via attention
  - CommunicationPolicy:     learned policy for what to communicate
  - PrivateCommunication:    simulated encrypted pair channels
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Message dataclass
# ============================================================

@dataclass
class Message:
    """
    A single communication message from one agent to neighbors.

    signal: (signal_dim,) continuous vector
    signal_softmax: (3,) softmax over [bullish, neutral, bearish]
    confidence: scalar in [0, 1]
    """
    sender_id:  str
    timestep:   int
    signal:     np.ndarray               # raw signal vector
    sentiment:  np.ndarray               # softmax([bullish, neutral, bearish])
    confidence: float                    # agent's self-assessed confidence
    ttl:        int = 3                  # time-to-live (hops)

    def decay(self) -> "Message":
        """Return copy with TTL decremented."""
        return Message(
            sender_id  = self.sender_id,
            timestep   = self.timestep,
            signal     = self.signal.copy(),
            sentiment  = self.sentiment.copy(),
            confidence = self.confidence * 0.8,
            ttl        = self.ttl - 1,
        )


# ============================================================
# Communication Graph
# ============================================================

class AgentCommunicationGraph:
    """
    Directed graph of communication links between agents.

    Supports:
      - Static topology (ring, fully-connected, random)
      - Dynamic topology (edges added/removed based on agent behavior)
      - Centrality metrics for analyzing communication structure
    """

    TOPOLOGIES = ("ring", "fully_connected", "random", "small_world", "scale_free")

    def __init__(
        self,
        agent_ids:  List[str],
        topology:   str   = "random",
        p_edge:     float = 0.3,      # edge probability for random graph
        k_nearest:  int   = 3,         # for ring/small-world
        seed:       int   = 42,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.n         = len(agent_ids)
        self.topology  = topology
        self.seed      = seed

        self._id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        self.graph = self._build_graph(topology, p_edge, k_nearest)

        # Track communication activity
        self.message_counts: Dict[str, int] = {a: 0 for a in agent_ids}

    def _build_graph(
        self, topology: str, p_edge: float, k: int
    ) -> nx.DiGraph:
        rng = np.random.default_rng(self.seed)
        if topology == "ring":
            g = nx.cycle_graph(self.n, create_using=nx.DiGraph())
        elif topology == "fully_connected":
            g = nx.complete_graph(self.n, create_using=nx.DiGraph())
        elif topology == "small_world":
            g = nx.watts_strogatz_graph(self.n, k, 0.1, seed=self.seed)
            g = g.to_directed()
        elif topology == "scale_free":
            g = nx.scale_free_graph(self.n, seed=self.seed)
        else:  # random
            g = nx.gnp_random_graph(self.n, p_edge, seed=self.seed, directed=True)

        # Relabel nodes with agent IDs
        mapping = {i: self.agent_ids[i] for i in range(self.n)}
        return nx.relabel_nodes(g, mapping)

    def neighbors(self, agent_id: str) -> List[str]:
        """Return agents that agent_id broadcasts TO."""
        if agent_id not in self.graph:
            return []
        return list(self.graph.successors(agent_id))

    def sources(self, agent_id: str) -> List[str]:
        """Return agents that broadcast TO agent_id."""
        if agent_id not in self.graph:
            return []
        return list(self.graph.predecessors(agent_id))

    def add_edge(self, src: str, dst: str) -> None:
        self.graph.add_edge(src, dst)

    def remove_edge(self, src: str, dst: str) -> None:
        if self.graph.has_edge(src, dst):
            self.graph.remove_edge(src, dst)

    def degree_centrality(self) -> Dict[str, float]:
        return nx.out_degree_centrality(self.graph)

    def clustering(self) -> float:
        """Average clustering coefficient."""
        ug = self.graph.to_undirected()
        return float(nx.average_clustering(ug))

    def diameter(self) -> Optional[int]:
        ug = self.graph.to_undirected()
        try:
            return nx.diameter(ug)
        except nx.NetworkXError:
            return None

    def adjacency_matrix(self) -> np.ndarray:
        return nx.to_numpy_array(self.graph, nodelist=self.agent_ids, dtype=np.float32)

    def get_neighbor_map(self) -> Dict[str, List[str]]:
        return {a: self.neighbors(a) for a in self.agent_ids}


# ============================================================
# Message Passing engine
# ============================================================

class MessagePassing:
    """
    Manages broadcast and routing of messages across the communication graph.

    Each step:
      1. Agents produce messages via CommunicationPolicy
      2. Messages are routed to graph neighbors
      3. Messages in transit decay (TTL, confidence)
      4. Recipients receive aggregated neighbor messages
    """

    def __init__(
        self,
        graph:      AgentCommunicationGraph,
        signal_dim: int   = 8,
        max_queue:  int   = 10,
    ) -> None:
        self.graph      = graph
        self.signal_dim = signal_dim
        # Inbox: agent_id → deque of pending Messages
        self._inbox: Dict[str, deque] = {
            a: deque(maxlen=max_queue) for a in graph.agent_ids
        }
        self._step = 0
        self.stats: Dict[str, Any] = defaultdict(list)

    def broadcast(self, sender_id: str, message: Message) -> int:
        """
        Deliver message to all graph neighbors of sender.
        Returns number of recipients.
        """
        neighbors = self.graph.neighbors(sender_id)
        for nb in neighbors:
            self._inbox[nb].append(message)
        self.graph.message_counts[sender_id] = (
            self.graph.message_counts.get(sender_id, 0) + 1
        )
        return len(neighbors)

    def step(self) -> None:
        """Advance message TTL; discard expired messages."""
        self._step += 1
        for aid in self.graph.agent_ids:
            queue = self._inbox[aid]
            # Filter expired
            live = [m for m in queue if m.ttl > 0]
            self._inbox[aid] = deque(live, maxlen=queue.maxlen)

    def get_inbox(self, agent_id: str) -> List[Message]:
        """Return and clear inbox of agent_id."""
        msgs = list(self._inbox[agent_id])
        self._inbox[agent_id].clear()
        return msgs

    def peek_inbox(self, agent_id: str) -> List[Message]:
        """Return inbox without clearing."""
        return list(self._inbox[agent_id])

    def clear_all(self) -> None:
        for q in self._inbox.values():
            q.clear()

    def network_activity(self) -> Dict[str, float]:
        """Summary of network communication activity."""
        total_msgs = sum(self.graph.message_counts.values())
        return {
            "total_messages": float(total_msgs),
            "mean_msgs_per_agent": float(total_msgs) / max(len(self.graph.agent_ids), 1),
            "clustering": self.graph.clustering(),
        }


# ============================================================
# Attention Aggregation
# ============================================================

class AttentionAggregation(nn.Module):
    """
    Multi-head attention aggregation of neighbor messages.

    Each agent aggregates incoming messages from neighbors using
    query-key-value attention where:
      - Query:  own hidden state (query_dim,)
      - Keys:   neighbor message signals (signal_dim,) each
      - Values: neighbor message signals (signal_dim,) each

    Output: weighted sum of neighbor signals (signal_dim,)
    """

    def __init__(
        self,
        query_dim:  int,
        signal_dim: int,
        n_heads:    int = 4,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads    = n_heads
        self.signal_dim = signal_dim
        head_dim        = signal_dim // n_heads
        assert signal_dim % n_heads == 0, "signal_dim must be divisible by n_heads"

        self.q_proj = nn.Linear(query_dim,  signal_dim, bias=False)
        self.k_proj = nn.Linear(signal_dim, signal_dim, bias=False)
        self.v_proj = nn.Linear(signal_dim, signal_dim, bias=False)
        self.out_proj = nn.Linear(signal_dim, signal_dim, bias=False)
        self.dropout  = nn.Dropout(p=dropout)
        self._scale   = math.sqrt(head_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        query:   torch.Tensor,  # (batch, query_dim)
        signals: torch.Tensor,  # (batch, n_neighbors, signal_dim)
        mask:    Optional[torch.Tensor] = None,  # (batch, n_neighbors) bool True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            aggregated: (batch, signal_dim)
            attn_weights: (batch, n_heads, n_neighbors)
        """
        B, N, D = signals.shape
        H       = self.n_heads
        d_k     = D // H

        q = self.q_proj(query).view(B, H, d_k)         # (B, H, d_k)
        k = self.k_proj(signals).view(B, N, H, d_k).permute(0, 2, 1, 3)  # (B, H, N, d_k)
        v = self.v_proj(signals).view(B, N, H, d_k).permute(0, 2, 1, 3)  # (B, H, N, d_k)

        # Scaled dot-product attention
        scores = torch.einsum("bhd,bhnd->bhn", q, k) / self._scale  # (B, H, N)

        if mask is not None:
            invalid = ~mask.unsqueeze(1).expand_as(scores)  # (B, H, N)
            scores  = scores.masked_fill(invalid, -1e9)

        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        # Weighted sum of values
        out = torch.einsum("bhn,bhnd->bhd", attn, v)   # (B, H, d_k)
        out = out.reshape(B, D)
        out = self.out_proj(out)

        return out, attn

    def aggregate_messages(
        self,
        own_hidden:  torch.Tensor,  # (query_dim,)
        messages:    List[Message],
        signal_dim:  int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper that converts Message objects → tensor.
        """
        if not messages:
            dummy_signal = torch.zeros(1, signal_dim)
            dummy_attn   = torch.ones(1, 1, 1) / 1
            return dummy_signal.squeeze(0), dummy_attn

        signals_np = np.stack([
            np.pad(m.signal, (0, max(0, signal_dim - len(m.signal))))[:signal_dim]
            for m in messages
        ])
        signals_t = torch.FloatTensor(signals_np).unsqueeze(0)  # (1, N, D)
        query_t   = own_hidden.unsqueeze(0) if own_hidden.dim() == 1 else own_hidden

        agg, attn = self.forward(query_t, signals_t)
        return agg.squeeze(0), attn


# ============================================================
# Communication Policy (learned)
# ============================================================

class CommunicationPolicy(nn.Module):
    """
    Learned policy for WHAT to communicate.

    Input:  own hidden state (hidden_dim,)
    Output: message signal vector (signal_dim,)
            + sentiment distribution over [bullish, neutral, bearish]
            + confidence scalar

    Trained end-to-end with the main agent policy via gradient flow
    through the attention aggregation module.
    """

    def __init__(
        self,
        hidden_dim: int,
        signal_dim: int = 8,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.signal_dim = signal_dim

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Signal head: free continuous vector
        self.signal_head = nn.Linear(hidden_dim, signal_dim)
        # Tanh: keep signals bounded in [-1, 1]
        nn.init.uniform_(self.signal_head.weight, -0.01, 0.01)

        # Sentiment head: softmax over [bullish, neutral, bearish]
        self.sentiment_head = nn.Linear(hidden_dim, 3)

        # Confidence head: sigmoid → [0, 1]
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (batch, hidden_dim)

        Returns:
            signal:     (batch, signal_dim)  in [-1, 1]
            sentiment:  (batch, 3)           probabilities
            confidence: (batch,)             in [0, 1]
        """
        h          = self.encoder(hidden)
        signal     = torch.tanh(self.signal_head(h))
        sentiment  = F.softmax(self.sentiment_head(h), dim=-1)
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)
        return signal, sentiment, confidence

    def produce_message(
        self,
        sender_id: str,
        timestep:  int,
        hidden:    np.ndarray,
    ) -> Message:
        """
        Produce a Message object from an agent's hidden state.
        """
        h_t = torch.FloatTensor(hidden).unsqueeze(0)
        with torch.no_grad():
            signal, sentiment, conf = self.forward(h_t)

        return Message(
            sender_id  = sender_id,
            timestep   = timestep,
            signal     = signal.squeeze(0).numpy(),
            sentiment  = sentiment.squeeze(0).numpy(),
            confidence = float(conf.item()),
        )


# ============================================================
# Private Communication (simulated encryption)
# ============================================================

class PrivateCommunication:
    """
    Simulated encrypted private channel between agent pairs.

    In reality this is just a direct message dict — the "encryption"
    is that only the intended recipient can read it (no broadcast).
    Models scenarios where agents form coalitions with private channels.
    """

    def __init__(self, agent_ids: List[str]) -> None:
        self.agent_ids  = agent_ids
        # (sender, receiver) → list of messages
        self._channels: Dict[Tuple[str, str], List[Message]] = {}
        # Active partnerships
        self._partnerships: Set[Tuple[str, str]] = set()

    def open_channel(self, a1: str, a2: str) -> None:
        """Open a private channel between a1 and a2 (bidirectional)."""
        self._partnerships.add((a1, a2))
        self._partnerships.add((a2, a1))
        self._channels[(a1, a2)] = []
        self._channels[(a2, a1)] = []

    def close_channel(self, a1: str, a2: str) -> None:
        self._partnerships.discard((a1, a2))
        self._partnerships.discard((a2, a1))

    def send(self, sender: str, receiver: str, message: Message) -> bool:
        """
        Send private message. Returns False if channel not open.
        """
        key = (sender, receiver)
        if key not in self._partnerships:
            return False
        self._channels.setdefault(key, []).append(message)
        return True

    def receive(self, receiver: str, sender: str) -> List[Message]:
        """Retrieve and clear messages from sender to receiver."""
        key  = (sender, receiver)
        msgs = self._channels.pop(key, [])
        if key in self._partnerships:
            self._channels[key] = []  # re-open empty queue
        return msgs

    def receive_all(self, receiver: str) -> Dict[str, List[Message]]:
        """Retrieve all private messages directed to receiver."""
        result = {}
        for sender in self.agent_ids:
            key  = (sender, receiver)
            if key in self._partnerships:
                result[sender] = self._channels.get(key, [])
                self._channels[key] = []
        return result

    def partnerships_of(self, agent_id: str) -> List[str]:
        """Return all agents that have a private channel with agent_id."""
        return [b for (a, b) in self._partnerships if a == agent_id]

    def n_partnerships(self) -> int:
        return len(self._partnerships) // 2  # bidirectional counted once
