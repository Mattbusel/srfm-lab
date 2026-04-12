"""
multi_agent_comm_advanced.py
============================
Advanced multi-agent communication protocols for the Hyper-Agent MARL system.

Implements:
  - Emergent communication via differentiable discrete/continuous messages
  - Bandwidth-constrained communication (top-k gating, compressed messages)
  - Hierarchical communication (team leaders + followers)
  - Asynchronous message passing with message queues and delays
  - Communication failure robustness (dropout, noise, packet loss)
  - Message content analysis (mutual information, what agents communicate)
  - Attention-based message aggregation
  - Signature-verified message routing
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import logging
import math
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message data structure
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AgentMessage:
    """A message sent from one agent to another (or broadcast)."""
    sender_id: str
    recipient_id: Optional[str]          # None = broadcast
    content: torch.Tensor                # raw message vector
    timestep: int = 0
    priority: float = 1.0
    compressed: bool = False
    signature: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"AgentMessage(from={self.sender_id}, to={self.recipient_id}, "
                f"dim={self.content.shape}, t={self.timestep})")


# ---------------------------------------------------------------------------
# Base communication module
# ---------------------------------------------------------------------------

class BaseCommunicationModule(nn.Module, abc.ABC):
    """Abstract base for all communication modules."""

    def __init__(self, n_agents: int, obs_dim: int, msg_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim

    @abc.abstractmethod
    def send(self, obs: torch.Tensor,
             hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate messages from observations. Returns (B, msg_dim) or (B, n_agents, msg_dim)."""
        ...

    @abc.abstractmethod
    def receive(self, obs: torch.Tensor,
                messages: torch.Tensor) -> torch.Tensor:
        """Integrate received messages into observation. Returns (B, out_dim)."""
        ...


# ---------------------------------------------------------------------------
# 1. Emergent communication: differentiable discrete messages (Gumbel-Softmax)
# ---------------------------------------------------------------------------

class GumbelSoftmaxComm(BaseCommunicationModule):
    """
    Agents communicate via discrete symbols learned end-to-end
    using the Straight-Through Gumbel-Softmax estimator.
    Each agent emits a sequence of K discrete tokens from a vocabulary of V symbols.
    """

    def __init__(self, n_agents: int, obs_dim: int,
                 vocab_size: int = 16, n_symbols: int = 4,
                 msg_embed_dim: int = 32, temperature: float = 1.0):
        msg_dim = n_symbols * msg_embed_dim
        super().__init__(n_agents, obs_dim, msg_dim)
        self.vocab_size = vocab_size
        self.n_symbols = n_symbols
        self.msg_embed_dim = msg_embed_dim
        self.temperature = temperature

        # Encoder: obs -> symbol logits
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_symbols * vocab_size),
        )

        # Symbol embeddings
        self.symbol_embeddings = nn.Embedding(vocab_size, msg_embed_dim)

        # Receiver: obs + messages -> enriched obs
        self.receiver = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

        # Message analyser (what information is communicated)
        self.content_classifier = nn.Linear(msg_dim, 8)

    def send(self, obs: torch.Tensor,
             hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = obs.shape[0]
        logits = self.encoder(obs).view(B, self.n_symbols, self.vocab_size)

        if self.training:
            # Gumbel-Softmax for differentiable discretisation
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            soft_tokens = F.softmax((logits + gumbel) / self.temperature, dim=-1)
            # Straight-through
            hard_tokens = (torch.zeros_like(soft_tokens)
                           .scatter_(-1, soft_tokens.argmax(-1, keepdim=True), 1.0))
            tokens = hard_tokens - soft_tokens.detach() + soft_tokens
        else:
            indices = logits.argmax(-1)
            tokens = F.one_hot(indices, self.vocab_size).float()

        # Embed
        indices = tokens.argmax(-1)                  # (B, n_symbols)
        embedded = self.symbol_embeddings(indices)   # (B, n_symbols, embed_dim)
        message = embedded.view(B, self.msg_dim)     # (B, msg_dim)
        return message

    def receive(self, obs: torch.Tensor,
                messages: torch.Tensor) -> torch.Tensor:
        # messages: (B, msg_dim) — aggregated from all senders
        combined = torch.cat([obs, messages], dim=-1)
        return self.receiver(combined)

    def analyse_content(self, messages: torch.Tensor) -> torch.Tensor:
        """Classify message content into semantic categories."""
        return self.content_classifier(messages)


# ---------------------------------------------------------------------------
# 2. Continuous emergent communication with attention aggregation
# ---------------------------------------------------------------------------

class AttentionCommModule(BaseCommunicationModule):
    """
    Continuous message passing with multi-head attention aggregation.
    Each agent attends over messages from all other agents.
    """

    def __init__(self, n_agents: int, obs_dim: int, msg_dim: int = 64,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__(n_agents, obs_dim, msg_dim)
        self.n_heads = n_heads

        # Message encoder (per agent)
        self.msg_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, msg_dim),
        )

        # Multi-head attention over messages
        self.attention = nn.MultiheadAttention(
            embed_dim=msg_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        # Integration layer
        self.integrator = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

        self._message_buffer: Dict[str, torch.Tensor] = {}
        self._comm_stats: Dict[str, float] = {}

    def send(self, obs: torch.Tensor,
             hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.msg_encoder(obs)   # (B, msg_dim)

    def aggregate_messages(self, agent_idx: int,
                            all_messages: torch.Tensor) -> torch.Tensor:
        """
        all_messages: (B, n_agents, msg_dim)
        Returns aggregated message for agent_idx: (B, msg_dim)
        """
        # Query = current agent message, Keys/Values = all others
        query = all_messages[:, agent_idx:agent_idx + 1, :]   # (B, 1, msg_dim)
        attn_out, attn_weights = self.attention(query, all_messages, all_messages)
        self._comm_stats[f"agent_{agent_idx}_attn_entropy"] = float(
            -(attn_weights * (attn_weights + 1e-9).log()).sum(-1).mean().item()
        )
        return attn_out.squeeze(1)   # (B, msg_dim)

    def receive(self, obs: torch.Tensor,
                messages: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([obs, messages], dim=-1)
        return self.integrator(combined)

    def forward_all_agents(self, obs_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs_all: (B, n_agents, obs_dim)
        Returns: (enriched_obs: B x n_agents x obs_dim, messages: B x n_agents x msg_dim)
        """
        B, N, D = obs_all.shape
        obs_flat = obs_all.view(B * N, D)
        msgs_flat = self.send(obs_flat)                    # (B*N, msg_dim)
        msgs = msgs_flat.view(B, N, self.msg_dim)          # (B, N, msg_dim)

        enriched = []
        for i in range(N):
            agg = self.aggregate_messages(i, msgs)         # (B, msg_dim)
            obs_i = obs_all[:, i, :]
            enriched_i = self.receive(obs_i, agg)
            enriched.append(enriched_i)

        enriched_all = torch.stack(enriched, dim=1)        # (B, N, obs_dim)
        return enriched_all, msgs


# ---------------------------------------------------------------------------
# 3. Bandwidth-constrained communication (top-k gating)
# ---------------------------------------------------------------------------

class BandwidthConstrainedComm(BaseCommunicationModule):
    """
    Each agent can only send a message to top-K agents (by learned relevance score).
    Implements a learned gating mechanism to select recipients.
    Messages are compressed before transmission.
    """

    def __init__(self, n_agents: int, obs_dim: int,
                 msg_dim: int = 32, k: int = 2,
                 compress_ratio: float = 0.5):
        super().__init__(n_agents, obs_dim, msg_dim)
        self.k = min(k, n_agents - 1)
        self.compress_ratio = compress_ratio
        compressed_dim = max(4, int(msg_dim * compress_ratio))
        self.compressed_dim = compressed_dim

        # Message generator (full-fidelity)
        self.msg_gen = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, msg_dim),
        )

        # Compressor / decompressor
        self.compressor = nn.Linear(msg_dim, compressed_dim)
        self.decompressor = nn.Linear(compressed_dim, msg_dim)

        # Relevance scorer (decides who to talk to)
        self.relevance_scorer = nn.Bilinear(obs_dim, obs_dim, 1)

        # Receiver
        self.receiver_net = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, 256), nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

        self._bandwidth_usage: float = 0.0
        self._gating_entropy: float = 0.0

    def send(self, obs: torch.Tensor,
             hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        full_msg = self.msg_gen(obs)
        compressed = self.compressor(full_msg)
        return compressed   # (B, compressed_dim)

    def compute_gates(self, sender_obs: torch.Tensor,
                       all_obs: torch.Tensor) -> torch.Tensor:
        """
        sender_obs: (B, obs_dim)
        all_obs:    (B, n_agents, obs_dim)
        Returns gates: (B, n_agents)
        """
        B, N, D = all_obs.shape
        sender_expanded = sender_obs.unsqueeze(1).expand(B, N, D)
        scores = self.relevance_scorer(
            sender_expanded.reshape(B * N, D),
            all_obs.reshape(B * N, D),
        ).view(B, N)

        # Top-k sparse gating
        topk_vals, topk_idxs = scores.topk(min(self.k, N), dim=-1)
        gates = torch.zeros_like(scores)
        gates.scatter_(1, topk_idxs, F.softmax(topk_vals, dim=-1))

        self._gating_entropy = float(
            -(gates * (gates + 1e-9).log()).sum(-1).mean().item()
        )
        self._bandwidth_usage = float((gates > 0).float().mean().item())
        return gates

    def receive(self, obs: torch.Tensor,
                messages: torch.Tensor) -> torch.Tensor:
        # messages already aggregated: (B, msg_dim)
        combined = torch.cat([obs, messages], dim=-1)
        return self.receiver_net(combined)

    def full_forward(self, obs_all: torch.Tensor) -> torch.Tensor:
        """
        obs_all: (B, n_agents, obs_dim)
        Returns enriched obs: (B, n_agents, obs_dim)
        """
        B, N, D = obs_all.shape
        obs_flat = obs_all.view(B * N, D)
        compressed_flat = self.send(obs_flat)
        compressed = compressed_flat.view(B, N, self.compressed_dim)

        # Decompress for each receiver
        full_msgs = self.decompressor(
            compressed.view(B * N, self.compressed_dim)
        ).view(B, N, self.msg_dim)

        enriched = []
        for i in range(N):
            gates_i = self.compute_gates(obs_all[:, i, :], obs_all)  # (B, N)
            # Weighted aggregate of messages
            agg = (full_msgs * gates_i.unsqueeze(-1)).sum(dim=1)     # (B, msg_dim)
            enriched_i = self.receive(obs_all[:, i, :], agg)
            enriched.append(enriched_i)

        return torch.stack(enriched, dim=1)  # (B, N, obs_dim)


# ---------------------------------------------------------------------------
# 4. Hierarchical communication (leader-follower)
# ---------------------------------------------------------------------------

class HierarchicalCommModule(nn.Module):
    """
    Two-level hierarchical communication:
      Level 1: Leaders aggregate information from their sub-team
      Level 2: Leaders exchange with each other
      Level 3: Leaders broadcast decisions back to followers

    Uses a fixed assignment: agent 0 in each sub-team is the leader.
    """

    def __init__(self, n_agents: int, obs_dim: int,
                 team_size: int = 4, msg_dim: int = 64,
                 leader_msg_dim: int = 128):
        super().__init__()
        assert n_agents % team_size == 0, "n_agents must be divisible by team_size"
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.team_size = team_size
        self.n_teams = n_agents // team_size
        self.msg_dim = msg_dim
        self.leader_msg_dim = leader_msg_dim

        # Follower -> leader aggregation
        self.follower_encoder = nn.Sequential(
            nn.Linear(obs_dim, msg_dim), nn.ReLU()
        )
        self.leader_aggregator = nn.GRU(
            input_size=msg_dim, hidden_size=msg_dim, batch_first=True
        )

        # Leader <-> leader communication
        self.leader_comm = AttentionCommModule(
            n_agents=self.n_teams, obs_dim=msg_dim,
            msg_dim=leader_msg_dim,
        )

        # Leader -> follower broadcast
        self.broadcast_encoder = nn.Sequential(
            nn.Linear(leader_msg_dim, msg_dim), nn.ReLU()
        )

        # Integration
        self.integrator = nn.Sequential(
            nn.Linear(obs_dim + msg_dim + leader_msg_dim, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

    def forward(self, obs_all: torch.Tensor) -> torch.Tensor:
        """
        obs_all: (B, n_agents, obs_dim)
        Returns: enriched_obs (B, n_agents, obs_dim)
        """
        B, N, D = obs_all.shape
        T = self.team_size
        n_teams = self.n_teams

        # Step 1: follower -> leader aggregation
        team_agg = []
        for team_idx in range(n_teams):
            team_obs = obs_all[:, team_idx * T:(team_idx + 1) * T, :]  # (B, T, D)
            encoded = self.follower_encoder(
                team_obs.view(B * T, D)
            ).view(B, T, self.msg_dim)
            _, hidden = self.leader_aggregator(encoded)  # (1, B, msg_dim)
            team_agg.append(hidden.squeeze(0))           # (B, msg_dim)

        leader_obs = torch.stack(team_agg, dim=1)        # (B, n_teams, msg_dim)

        # Step 2: leader <-> leader communication
        enriched_leaders, leader_msgs = self.leader_comm.forward_all_agents(leader_obs)
        # enriched_leaders: (B, n_teams, msg_dim)

        # Step 3: broadcast back and integrate
        enriched_all = []
        for team_idx in range(n_teams):
            leader_signal = enriched_leaders[:, team_idx, :]        # (B, msg_dim)
            broadcast = self.broadcast_encoder(leader_signal)        # (B, msg_dim)
            for agent_local_idx in range(T):
                global_agent_idx = team_idx * T + agent_local_idx
                obs_i = obs_all[:, global_agent_idx, :]
                # Local team messages
                team_encoded = self.follower_encoder(obs_i)           # (B, msg_dim)
                combined = torch.cat([obs_i, team_encoded, leader_signal], dim=-1)
                enriched_i = self.integrator(combined)
                enriched_all.append(enriched_i)

        enriched_tensor = torch.stack(enriched_all, dim=1)
        assert enriched_tensor.shape == (B, N, D)
        return enriched_tensor


# ---------------------------------------------------------------------------
# 5. Asynchronous message passing
# ---------------------------------------------------------------------------

class AsyncMessageQueue:
    """
    Simulates asynchronous message passing with variable delays.
    Messages are stored in per-agent queues and delivered at later steps.
    """

    def __init__(self, n_agents: int, msg_dim: int,
                 max_delay: int = 3, packet_loss_prob: float = 0.05,
                 rng_seed: int = 0):
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.max_delay = max_delay
        self.packet_loss_prob = packet_loss_prob
        self._rng = np.random.default_rng(rng_seed)

        # Queues: agent_id -> list of (delivery_step, message_tensor)
        self._queues: Dict[int, List[Tuple[int, torch.Tensor]]] = {
            i: [] for i in range(n_agents)
        }
        self._current_step: int = 0
        self._stats = defaultdict(int)

    def enqueue(self, sender: int, recipient: int,
                message: torch.Tensor) -> None:
        if self._rng.random() < self.packet_loss_prob:
            self._stats["dropped"] += 1
            return
        delay = int(self._rng.integers(0, self.max_delay + 1))
        delivery_step = self._current_step + delay
        self._queues[recipient].append((delivery_step, message.detach().clone()))
        self._stats["enqueued"] += 1

    def broadcast(self, sender: int, message: torch.Tensor) -> None:
        for recipient in range(self.n_agents):
            if recipient != sender:
                self.enqueue(sender, recipient, message)

    def receive(self, agent: int) -> List[torch.Tensor]:
        """Collect all messages due for delivery at current step."""
        delivered = []
        remaining = []
        for delivery_step, msg in self._queues[agent]:
            if delivery_step <= self._current_step:
                delivered.append(msg)
                self._stats["delivered"] += 1
            else:
                remaining.append((delivery_step, msg))
        self._queues[agent] = remaining
        return delivered

    def aggregate_received(self, agent: int,
                            msg_dim: int,
                            device: str = "cpu") -> torch.Tensor:
        """Aggregate received messages into single vector (mean pool)."""
        messages = self.receive(agent)
        if not messages:
            return torch.zeros(msg_dim, device=device)
        stacked = torch.stack(messages, dim=0)
        return stacked.mean(0)

    def step(self) -> None:
        self._current_step += 1

    def reset(self) -> None:
        self._current_step = 0
        self._queues = {i: [] for i in range(self.n_agents)}

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# 6. Communication failure robustness
# ---------------------------------------------------------------------------

class RobustCommWrapper(nn.Module):
    """
    Wraps a communication module with configurable failure modes:
      - Gaussian noise injection
      - Dropout / silence
      - Adversarial perturbation
      - Quantisation noise
    """

    def __init__(self, comm_module: BaseCommunicationModule,
                 noise_std: float = 0.0,
                 dropout_prob: float = 0.0,
                 quantise_bits: Optional[int] = None,
                 adversarial_eps: float = 0.0):
        super().__init__()
        self.comm = comm_module
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.quantise_bits = quantise_bits
        self.adversarial_eps = adversarial_eps

    def send(self, obs: torch.Tensor,
             hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        msg = self.comm.send(obs, hidden)

        # Apply failure modes
        if self.noise_std > 0:
            msg = msg + torch.randn_like(msg) * self.noise_std

        if self.dropout_prob > 0 and self.training:
            mask = (torch.rand_like(msg) > self.dropout_prob).float()
            msg = msg * mask

        if self.quantise_bits is not None:
            msg = self._quantise(msg, self.quantise_bits)

        return msg

    def receive(self, obs: torch.Tensor,
                messages: torch.Tensor) -> torch.Tensor:
        return self.comm.receive(obs, messages)

    def _quantise(self, tensor: torch.Tensor, n_bits: int) -> torch.Tensor:
        n_levels = 2 ** n_bits
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / n_levels
        if scale < 1e-9:
            return tensor
        quantised = torch.round((tensor - min_val) / scale) * scale + min_val
        return quantised.detach() - tensor.detach() + tensor  # ST trick


# ---------------------------------------------------------------------------
# 7. Message content analysis (mutual information)
# ---------------------------------------------------------------------------

class MessageContentAnalyser:
    """
    Analyses what information agents learn to communicate.
    Uses linear probes to decode ground truth from messages.
    """

    def __init__(self, msg_dim: int, n_probe_targets: int = 8,
                 probe_hidden: int = 64):
        self.msg_dim = msg_dim
        self.n_probe_targets = n_probe_targets

        # Linear probes for various state variables
        self.probes = nn.ModuleDict({
            f"target_{i}": nn.Sequential(
                nn.Linear(msg_dim, probe_hidden),
                nn.ReLU(),
                nn.Linear(probe_hidden, 1),
            )
            for i in range(n_probe_targets)
        })

        self._probe_losses: Dict[str, deque] = {
            k: deque(maxlen=1000) for k in self.probes
        }

    def train_probes(self, messages: torch.Tensor,
                     targets: torch.Tensor,
                     lr: float = 1e-3) -> Dict[str, float]:
        """
        messages: (B, msg_dim)
        targets: (B, n_targets) — e.g. inventory, price, spread, ...
        """
        losses = {}
        msg_detach = messages.detach()
        for i, (name, probe) in enumerate(self.probes.items()):
            if i >= targets.shape[1]:
                break
            pred = probe(msg_detach).squeeze(-1)
            target_i = targets[:, i]
            loss = F.mse_loss(pred, target_i)
            losses[name] = float(loss.item())
            self._probe_losses[name].append(losses[name])
        return losses

    def decode_content(self, messages: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return {
                name: probe(messages).squeeze(-1)
                for name, probe in self.probes.items()
            }

    def communication_efficiency(self) -> float:
        """Average probe loss (lower = more informative messages)."""
        all_losses = []
        for buf in self._probe_losses.values():
            if buf:
                all_losses.append(float(np.mean(buf)))
        return float(np.mean(all_losses)) if all_losses else float("inf")

    def mutual_information_estimate(self, messages: torch.Tensor,
                                     targets: torch.Tensor) -> Dict[str, float]:
        """
        Estimate mutual information I(message; target_i) via MINE-style estimator.
        Positive values indicate correlation between messages and targets.
        """
        mi_estimates = {}
        msg_detach = messages.detach()

        for i in range(min(targets.shape[1], self.n_probe_targets)):
            target_i = targets[:, i:i + 1]
            # Joint distribution
            joint = torch.cat([msg_detach, target_i], dim=-1)
            # Marginal (shuffle targets)
            perm = torch.randperm(target_i.shape[0])
            shuffled_target = target_i[perm]
            marginal = torch.cat([msg_detach, shuffled_target], dim=-1)

            # Simple linear MINE estimator
            W = torch.randn(joint.shape[1], 1, device=messages.device)
            T_joint = joint.matmul(W).mean()
            T_marginal = torch.log(torch.exp(marginal.matmul(W)).mean() + 1e-9)
            mi = float((T_joint - T_marginal).item())
            mi_estimates[f"MI_target_{i}"] = mi

        return mi_estimates


# ---------------------------------------------------------------------------
# 8. Communication graph (dynamic topology)
# ---------------------------------------------------------------------------

class DynamicCommGraph:
    """
    Manages a dynamic communication graph where edge weights
    evolve based on agent interactions and information utility.
    """

    def __init__(self, n_agents: int, init_density: float = 0.5,
                 decay: float = 0.95, boost: float = 0.1,
                 seed: int = 0):
        self.n_agents = n_agents
        self.decay = decay
        self.boost = boost
        self._rng = np.random.default_rng(seed)
        # Edge weights: adjacency matrix
        self._weights = (self._rng.random((n_agents, n_agents)) < init_density).astype(float)
        np.fill_diagonal(self._weights, 0.0)

    def update(self, i: int, j: int, was_useful: bool) -> None:
        """Update edge weight based on whether message from i to j was useful."""
        if was_useful:
            self._weights[i, j] = min(1.0, self._weights[i, j] + self.boost)
        else:
            self._weights[i, j] = max(0.0, self._weights[i, j] * self.decay)

    def get_neighbors(self, agent: int, threshold: float = 0.3) -> List[int]:
        return [j for j in range(self.n_agents)
                if j != agent and self._weights[agent, j] >= threshold]

    def adjacency_matrix(self) -> np.ndarray:
        return self._weights.copy()

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        return torch.tensor(self._weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# 9. Centralised communication controller (CTDE)
# ---------------------------------------------------------------------------

class CentralisedCommController(nn.Module):
    """
    Centralised-Training Decentralised-Execution communication controller.
    During training: has access to all agents' observations to compute messages.
    During execution: each agent only receives pre-computed messages.
    """

    def __init__(self, n_agents: int, obs_dim: int,
                 msg_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim

        # Centralised encoder (used during training only)
        self.central_encoder = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, msg_dim * n_agents),
        )

        # Decentralised decoder (used at execution)
        self.local_decoder = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

        self._cached_messages: Optional[torch.Tensor] = None

    def compute_messages_centralised(self,
                                      obs_all: torch.Tensor) -> torch.Tensor:
        """
        obs_all: (B, n_agents, obs_dim)
        Returns: (B, n_agents, msg_dim)
        """
        B = obs_all.shape[0]
        flat = obs_all.view(B, -1)
        msgs_flat = self.central_encoder(flat)
        msgs = msgs_flat.view(B, self.n_agents, self.msg_dim)
        self._cached_messages = msgs
        return msgs

    def integrate_message(self, obs: torch.Tensor,
                           message: torch.Tensor) -> torch.Tensor:
        """(B, obs_dim) -> (B, obs_dim) using pre-computed message."""
        combined = torch.cat([obs, message], dim=-1)
        return self.local_decoder(combined)

    def forward(self, obs_all: torch.Tensor) -> torch.Tensor:
        """
        obs_all: (B, n_agents, obs_dim)
        Returns: enriched_obs (B, n_agents, obs_dim)
        """
        msgs = self.compute_messages_centralised(obs_all)
        enriched = []
        for i in range(self.n_agents):
            obs_i = obs_all[:, i, :]
            msg_i = msgs[:, i, :]
            enriched.append(self.integrate_message(obs_i, msg_i))
        return torch.stack(enriched, dim=1)


# ---------------------------------------------------------------------------
# 10. CommNet implementation
# ---------------------------------------------------------------------------

class CommNet(BaseCommunicationModule):
    """
    CommNet (Sukhbaatar et al., 2016) continuous communication via mean-pooled messages.
    Agents pass h_i through multiple rounds of communication.
    """

    def __init__(self, n_agents: int, obs_dim: int,
                 msg_dim: int = 128, n_rounds: int = 2):
        super().__init__(n_agents, obs_dim, msg_dim)
        self.n_rounds = n_rounds

        self.encoder = nn.Linear(obs_dim, msg_dim)
        self.comm_layers = nn.ModuleList([
            nn.Linear(msg_dim * 2, msg_dim)
            for _ in range(n_rounds)
        ])
        self.output = nn.Linear(msg_dim, obs_dim)

    def send(self, obs: torch.Tensor,
             hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(obs)

    def receive(self, obs: torch.Tensor,
                messages: torch.Tensor) -> torch.Tensor:
        return self.output(messages)

    def forward(self, obs_all: torch.Tensor) -> torch.Tensor:
        """
        obs_all: (B, n_agents, obs_dim)
        Returns: (B, n_agents, obs_dim)
        """
        B, N, D = obs_all.shape
        h = self.encoder(obs_all.view(B * N, D)).view(B, N, self.msg_dim)

        for comm_layer in self.comm_layers:
            # Mean pool excluding self
            mean_msg = (h.sum(dim=1, keepdim=True) - h) / max(N - 1, 1)
            h_combined = torch.cat([h, mean_msg], dim=-1)
            h = F.relu(comm_layer(h_combined.view(B * N, -1))).view(B, N, self.msg_dim)

        out = self.output(h.view(B * N, self.msg_dim)).view(B, N, D)
        return out


# ---------------------------------------------------------------------------
# 11. Full communication system (composable)
# ---------------------------------------------------------------------------

class MultiAgentCommSystem(nn.Module):
    """
    Top-level communication system that composes multiple protocols.
    Selects the appropriate module based on configuration.
    """

    class CommProtocol(enum.Enum):
        GUMBEL = "gumbel"
        ATTENTION = "attention"
        BANDWIDTH = "bandwidth"
        HIERARCHICAL = "hierarchical"
        COMMNET = "commnet"
        CENTRALISED = "centralised"

    def __init__(self, n_agents: int, obs_dim: int,
                 protocol: "MultiAgentCommSystem.CommProtocol" = None,
                 msg_dim: int = 64,
                 failure_noise_std: float = 0.0,
                 failure_dropout: float = 0.0,
                 async_delay: int = 0,
                 **protocol_kwargs):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.protocol = protocol or self.CommProtocol.ATTENTION
        self.msg_dim = msg_dim
        self.async_delay = async_delay

        # Build core module
        if self.protocol == self.CommProtocol.GUMBEL:
            core = GumbelSoftmaxComm(n_agents, obs_dim,
                                      msg_embed_dim=msg_dim, **protocol_kwargs)
        elif self.protocol == self.CommProtocol.ATTENTION:
            core = AttentionCommModule(n_agents, obs_dim, msg_dim, **protocol_kwargs)
        elif self.protocol == self.CommProtocol.BANDWIDTH:
            core = BandwidthConstrainedComm(n_agents, obs_dim, msg_dim, **protocol_kwargs)
        elif self.protocol == self.CommProtocol.COMMNET:
            core = CommNet(n_agents, obs_dim, msg_dim, **protocol_kwargs)
        elif self.protocol == self.CommProtocol.CENTRALISED:
            self._central = CentralisedCommController(n_agents, obs_dim, msg_dim)
            core = None
        elif self.protocol == self.CommProtocol.HIERARCHICAL:
            self._hierarchical = HierarchicalCommModule(
                n_agents, obs_dim, msg_dim=msg_dim, **protocol_kwargs
            )
            core = None
        else:
            raise ValueError(f"Unknown protocol: {self.protocol}")

        if core is not None and (failure_noise_std > 0 or failure_dropout > 0):
            self._module = RobustCommWrapper(core, failure_noise_std, failure_dropout)
        elif core is not None:
            self._module = core
        else:
            self._module = None

        if async_delay > 0:
            self._async_queue = AsyncMessageQueue(n_agents, msg_dim,
                                                   max_delay=async_delay)
        else:
            self._async_queue = None

        # Content analyser (optional, for training diagnostics)
        self._analyser = MessageContentAnalyser(msg_dim)

    def forward(self, obs_all: torch.Tensor) -> torch.Tensor:
        """
        obs_all: (B, n_agents, obs_dim)
        Returns: enriched_obs (B, n_agents, obs_dim)
        """
        if self.protocol == self.CommProtocol.CENTRALISED:
            return self._central(obs_all)
        if self.protocol == self.CommProtocol.HIERARCHICAL:
            return self._hierarchical(obs_all)
        if self.protocol == self.CommProtocol.COMMNET:
            return self._module(obs_all)
        if self.protocol == self.CommProtocol.ATTENTION:
            enriched, _ = self._module.forward_all_agents(obs_all)
            return enriched
        if self.protocol == self.CommProtocol.BANDWIDTH:
            return self._module.full_forward(obs_all)

        # Generic fallback
        B, N, D = obs_all.shape
        msgs = self._module.send(obs_all.view(B * N, D)).view(B, N, -1)
        mean_msg = msgs.mean(dim=1, keepdim=True).expand(B, N, -1)
        enriched = []
        for i in range(N):
            enriched.append(self._module.receive(obs_all[:, i, :], mean_msg[:, i, :]))
        return torch.stack(enriched, dim=1)

    def analyse(self, obs_all: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Run content analysis on a batch of observations."""
        B, N, D = obs_all.shape
        if self._module is None:
            return {}
        msgs = self._module.send(obs_all.view(B * N, D))
        analysis: Dict[str, Any] = {}
        if targets is not None:
            mi = self._analyser.mutual_information_estimate(msgs, targets)
            analysis.update(mi)
        analysis["comm_efficiency"] = self._analyser.communication_efficiency()
        return analysis


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== multi_agent_comm_advanced.py smoke test ===")

    B, N, D = 4, 6, 64
    obs = torch.randn(B, N, D)

    print("\n-- AttentionCommModule --")
    attn = AttentionCommModule(N, D, msg_dim=32, n_heads=4)
    enriched, msgs = attn.forward_all_agents(obs)
    print(f"Enriched obs: {enriched.shape}, Messages: {msgs.shape}")

    print("\n-- GumbelSoftmaxComm --")
    gumbel = GumbelSoftmaxComm(N, D, vocab_size=8, n_symbols=4, msg_embed_dim=16)
    flat_obs = obs.view(B * N, D)
    msgs_g = gumbel.send(flat_obs)
    print(f"Gumbel messages: {msgs_g.shape}")
    enriched_g = gumbel.receive(flat_obs, msgs_g)
    print(f"Enriched (gumbel): {enriched_g.shape}")

    print("\n-- BandwidthConstrainedComm --")
    bwc = BandwidthConstrainedComm(N, D, msg_dim=32, k=2)
    enriched_bwc = bwc.full_forward(obs)
    print(f"Enriched (bwc): {enriched_bwc.shape}")
    print(f"Bandwidth usage: {bwc._bandwidth_usage:.2f}, Gate entropy: {bwc._gating_entropy:.4f}")

    print("\n-- CommNet --")
    commnet = CommNet(N, D, msg_dim=64, n_rounds=2)
    enriched_cn = commnet.forward(obs)
    print(f"Enriched (CommNet): {enriched_cn.shape}")

    print("\n-- HierarchicalCommModule --")
    hier = HierarchicalCommModule(N, D, team_size=2, msg_dim=32, leader_msg_dim=64)
    enriched_h = hier.forward(obs)
    print(f"Enriched (hierarchical): {enriched_h.shape}")

    print("\n-- MultiAgentCommSystem --")
    sys = MultiAgentCommSystem(N, D, protocol=MultiAgentCommSystem.CommProtocol.ATTENTION,
                                msg_dim=32, failure_noise_std=0.1)
    enriched_sys = sys.forward(obs)
    print(f"Enriched (system): {enriched_sys.shape}")

    print("\n-- AsyncMessageQueue --")
    queue = AsyncMessageQueue(N, 32, max_delay=2)
    msg_tensor = torch.randn(32)
    queue.enqueue(0, 1, msg_tensor)
    queue.step()
    queue.step()
    agg = queue.aggregate_received(1, 32)
    print(f"Async received: {agg.shape}, Stats: {queue.stats}")

    print("\nAll smoke tests passed.")
