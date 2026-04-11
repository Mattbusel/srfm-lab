"""
dynamic_gnn.py — Temporal Graph Neural Networks for financial market dynamics.

Models:
    TemporalGraphConv   GCN with exponential decay by edge age
    DynamicEdgeConv     Edge feature learning from node pairs + edge age
    TemporalAttention   Multi-head attention over graph snapshots
    GraphRNN            GRU over sequence of graph states
    EvolutionaryGNN     Full model combining all above components

Training:
    train_evolutionary_gnn  AdamW + cosine LR + gradient clipping
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree, softmax


# ── Utility ───────────────────────────────────────────────────────────────────

def temporal_decay(edge_ages: Tensor, half_life: float = 20.0) -> Tensor:
    """Exponential decay weight: w(age) = exp(-ln2 * age / half_life)."""
    return torch.exp(-math.log(2.0) * edge_ages / half_life)


def sinusoidal_time_encoding(t: Tensor, d_model: int) -> Tensor:
    """Sinusoidal position encoding for scalar timestamps."""
    device = t.device
    t = t.unsqueeze(-1).float()  # (..., 1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() *
        -(math.log(10000.0) / d_model)
    )
    enc = torch.zeros(*t.shape[:-1], d_model, device=device)
    enc[..., 0::2] = torch.sin(t * div)
    enc[..., 1::2] = torch.cos(t * div)
    return enc


# ── TemporalGraphConv ─────────────────────────────────────────────────────────

class TemporalGraphConv(MessagePassing):
    """GCN layer with time-aware edge weights.

    Each edge carries an `age` (in steps since it appeared). The message
    is modulated by an exponential decay function of the edge age, so
    recent edges have more influence than old ones.

    Args:
        in_channels:   Input node feature dimension.
        out_channels:  Output node feature dimension.
        half_life:     Age in steps at which edge influence halves.
        bias:          Whether to add a bias term.
        normalize:     If True, apply D^{-1/2} A D^{-1/2} normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        half_life: float = 20.0,
        bias: bool = True,
        normalize: bool = True,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.half_life = half_life
        self.normalize = normalize

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        edge_age: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:           (N, in_channels) node features.
            edge_index:  (2, E) edge connectivity.
            edge_weight: (E,) optional edge weights.
            edge_age:    (E,) optional edge ages (in steps).
        Returns:
            (N, out_channels) updated node features.
        """
        n = x.size(0)
        edge_index, edge_weight = add_self_loops(
            edge_index,
            edge_attr=edge_weight,
            fill_value=1.0,
            num_nodes=n,
        )

        # Apply temporal decay
        if edge_age is not None:
            # Self-loops have age 0
            self_loop_ages = torch.zeros(n, device=x.device)
            all_ages = torch.cat([edge_age, self_loop_ages], dim=0)
            decay = temporal_decay(all_ages, self.half_life)
            if edge_weight is None:
                edge_weight = decay
            else:
                edge_weight = edge_weight * decay

        # Degree normalization
        if self.normalize:
            row, col = edge_index
            deg = degree(col, n, dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            if edge_weight is None:
                edge_weight = norm
            else:
                edge_weight = edge_weight * norm

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        if edge_weight is not None:
            return edge_weight.unsqueeze(-1) * x_j
        return x_j

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"{self.in_channels}, {self.out_channels}, "
                f"half_life={self.half_life})")


# ── DynamicEdgeConv ───────────────────────────────────────────────────────────

class DynamicEdgeConv(MessagePassing):
    """Edge convolution that learns edge features from node pair embeddings + age.

    For each edge (i, j) with age a:
        m_{ij} = MLP(h_i || h_j || phi(a))
    where phi(a) is a time encoding.

    Args:
        in_channels:   Node feature dimension.
        out_channels:  Output dimension.
        time_dim:      Dimension of time encoding.
        hidden_dim:    Hidden dimension of edge MLP.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__(aggr="max")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim

        mlp_in = 2 * in_channels + time_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
            nn.ReLU(),
        )

        self.node_update = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_age: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:          (N, in_channels) node features.
            edge_index: (2, E) edge connectivity.
            edge_age:   (E,) edge ages.
        Returns:
            (N, out_channels) updated node features.
        """
        n = x.size(0)
        if edge_age is None:
            edge_age = torch.zeros(edge_index.size(1), device=x.device)

        # Time encoding for edge ages
        time_enc = sinusoidal_time_encoding(edge_age, self.time_dim)  # (E, time_dim)

        agg = self.propagate(edge_index, x=x, time_enc=time_enc)
        out = self.node_update(torch.cat([x, agg], dim=-1))
        return out

    def message(self, x_i: Tensor, x_j: Tensor, time_enc: Tensor) -> Tensor:
        # time_enc shape: (E, time_dim)
        edge_feat = torch.cat([x_i, x_j, time_enc], dim=-1)
        return self.edge_mlp(edge_feat)


# ── TemporalAttention ─────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Multi-head attention over a sequence of graph embeddings (snapshots).

    Given a sequence of graph-level embeddings [h_0, h_1, ..., h_T],
    returns a context-aware representation via scaled dot-product attention,
    where queries are the most recent snapshot and keys/values are all snapshots.

    Args:
        embed_dim:   Embedding dimension.
        num_heads:   Number of attention heads.
        dropout:     Dropout on attention weights.
        causal:      If True, mask future snapshots.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Time encoding added to keys
        self.time_enc_proj = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        snapshots: Tensor,
        timestamps: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            snapshots:  (B, T, D) batch of snapshot embedding sequences.
            timestamps: (B, T) optional timestamp values for time encoding.
        Returns:
            (B, D) attended output, (B, num_heads, T, T) attention weights.
        """
        B, T, D = snapshots.shape

        # Add time encodings to keys
        if timestamps is not None:
            t_flat = timestamps.reshape(-1)
            t_enc = sinusoidal_time_encoding(t_flat, D).reshape(B, T, D)
            keys_input = snapshots + self.time_enc_proj(t_enc)
        else:
            keys_input = snapshots

        # Project
        q = self.q_proj(snapshots[:, -1:, :])   # (B, 1, D) — query is most recent
        k = self.k_proj(keys_input)              # (B, T, D)
        v = self.v_proj(snapshots)               # (B, T, D)

        # Reshape for multi-head
        q = q.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)      # (B, H, 1, d)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # (B, H, T, d)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # (B, H, T, d)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, T)

        if self.causal:
            # Mask: only attend to past (no future leakage)
            # Here T is the sequence, query is position T-1, all past is valid
            pass  # query is already last position, all T keys are valid past

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v)  # (B, H, 1, d)
        out = out.transpose(1, 2).reshape(B, 1, D)  # (B, 1, D)
        out = self.out_proj(out).squeeze(1)          # (B, D)

        return out, attn


# ── GraphRNN ──────────────────────────────────────────────────────────────────

class GraphRNN(nn.Module):
    """GRU over a sequence of graph states for next-graph prediction.

    Given a sequence of graph-level embeddings (from a GNN), uses a GRU to
    model graph evolution and predict the next-step graph embedding.

    Args:
        input_dim:   Dimension of graph-level embedding.
        hidden_dim:  GRU hidden dimension.
        output_dim:  Output dimension (next-graph embedding).
        num_layers:  Number of GRU layers.
        dropout:     Dropout between GRU layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Predict graph-level scalar metrics (e.g., mean Ricci curvature)
        self.scalar_head = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        seq: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            seq:    (B, T, input_dim) sequence of graph embeddings.
            hidden: (num_layers, B, hidden_dim) optional initial hidden state.
        Returns:
            next_embed:   (B, output_dim) predicted next-step embedding.
            hidden_out:   (num_layers, B, hidden_dim) final hidden state.
            scalar_pred:  (B, T) per-step scalar predictions.
        """
        out, hidden_out = self.gru(seq, hidden)          # (B, T, hidden_dim)
        next_embed = self.output_proj(out[:, -1, :])     # (B, output_dim)
        scalar_pred = self.scalar_head(out).squeeze(-1)  # (B, T)
        return next_embed, hidden_out, scalar_pred

    def unroll(
        self,
        seed: Tensor,
        hidden: Optional[Tensor],
        steps: int,
    ) -> List[Tensor]:
        """Autoregressively predict `steps` future graph embeddings."""
        predictions = []
        x = seed.unsqueeze(1)  # (B, 1, input_dim)
        h = hidden
        for _ in range(steps):
            out, h, _ = self.forward(x, h)
            predictions.append(out)
            x = out.unsqueeze(1)
        return predictions


# ── EvolutionaryGNN ───────────────────────────────────────────────────────────

class EvolutionaryGNN(nn.Module):
    """Full Evolutionary GNN model combining temporal convolution, edge features,
    temporal attention over snapshots, and GRU-based graph evolution prediction.

    Architecture:
        1. TemporalGraphConv x2  — encode node features with temporal edge weights
        2. DynamicEdgeConv       — refine node features with learned edge attributes
        3. Global pooling        — aggregate to graph-level embedding
        4. TemporalAttention     — attend over recent K snapshots
        5. GraphRNN              — model graph evolution, predict next state
        6. Heads                 — predict edge weights, Ricci curvature, regime

    Args:
        node_features:   Input node feature dimension.
        edge_features:   Input edge feature dimension (optional).
        hidden_dim:      Hidden dimension throughout model.
        seq_len:         Number of graph snapshots in input sequence.
        n_regimes:       Number of market regimes to classify.
        half_life:       Edge age half-life for temporal decay.
        n_heads:         Number of attention heads.
        dropout:         Dropout rate.
    """

    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 4,
        hidden_dim: int = 64,
        seq_len: int = 10,
        n_regimes: int = 4,
        half_life: float = 20.0,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.n_regimes = n_regimes

        # Node encoding
        self.input_proj = nn.Linear(node_features, hidden_dim)

        # Temporal graph convolutions
        self.tconv1 = TemporalGraphConv(hidden_dim, hidden_dim, half_life=half_life)
        self.tconv2 = TemporalGraphConv(hidden_dim, hidden_dim, half_life=half_life)

        # Dynamic edge convolution
        self.edge_conv = DynamicEdgeConv(hidden_dim, hidden_dim, time_dim=16)

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Graph-level projection
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Temporal attention over snapshots
        self.temporal_attn = TemporalAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            causal=True,
        )

        # Graph RNN for evolution prediction
        self.graph_rnn = GraphRNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Output heads
        self.edge_weight_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.ricci_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Ricci curvature in [-1, 1]
        )

        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_regimes),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def encode_snapshot(
        self,
        data: Data,
        edge_age: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode a single graph snapshot to a graph-level embedding.

        Args:
            data:     PyG Data with x (node features), edge_index, edge_attr (optional).
            edge_age: (E,) edge ages.
        Returns:
            (1, hidden_dim) graph embedding.
        """
        x = self.input_proj(data.x)          # (N, hidden_dim)
        edge_weight = data.edge_attr[:, 0] if data.edge_attr is not None else None

        # First temporal convolution
        h = self.tconv1(x, data.edge_index, edge_weight, edge_age)
        h = F.relu(self.ln1(h))
        h = self.dropout(h)

        # Second temporal convolution (residual)
        h2 = self.tconv2(h, data.edge_index, edge_weight, edge_age)
        h = self.ln2(h + h2)
        h = self.dropout(h)

        # Dynamic edge convolution
        h3 = self.edge_conv(h, data.edge_index, edge_age)
        h = self.ln3(h + h3)

        # Global mean pooling
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
            torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_emb = global_mean_pool(h, batch)  # (B, hidden_dim)
        return self.graph_proj(graph_emb)

    def forward(
        self,
        snapshot_list: List[Data],
        edge_ages_list: Optional[List[Tensor]] = None,
        timestamps: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            snapshot_list: List of T PyG Data objects (sequence of snapshots).
            edge_ages_list: List of T edge-age tensors.
            timestamps:    (T,) timestamps for positional encoding.
        Returns:
            dict with keys: next_embedding, ricci_pred, regime_logits, attn_weights
        """
        T = len(snapshot_list)
        embeddings = []

        for t, snap in enumerate(snapshot_list):
            ea = edge_ages_list[t] if edge_ages_list is not None else None
            emb = self.encode_snapshot(snap, ea)   # (1, hidden_dim) or (B, hidden_dim)
            embeddings.append(emb)

        # Stack to (B, T, hidden_dim)
        seq = torch.stack(embeddings, dim=1)

        # Temporal attention
        ts_tensor = timestamps.unsqueeze(0).float() if timestamps is not None else None
        attn_out, attn_weights = self.temporal_attn(seq, ts_tensor)

        # Graph RNN
        next_emb, _, scalar_preds = self.graph_rnn(seq)

        # Predictions
        ricci_pred = self.ricci_head(attn_out)        # (B, 1)
        regime_logits = self.regime_head(attn_out)    # (B, n_regimes)

        return {
            "next_embedding": next_emb,
            "ricci_pred": ricci_pred.squeeze(-1),
            "regime_logits": regime_logits,
            "attn_weights": attn_weights,
            "scalar_preds": scalar_preds,
            "graph_embedding": attn_out,
        }

    def predict_edge_weight(
        self,
        node_i: Tensor,
        node_j: Tensor,
    ) -> Tensor:
        """Predict edge weight between two node embeddings."""
        pair = torch.cat([node_i, node_j], dim=-1)
        return self.edge_weight_head(pair).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

class TemporalGraphDataset(torch.utils.data.Dataset):
    """Dataset of (snapshot_sequence, targets) pairs for training EvolutionaryGNN.

    Args:
        snapshots:      List of PyG Data objects ordered by time.
        edge_ages:      List of edge-age tensors (same length as snapshots).
        ricci_targets:  (T,) tensor of Ricci curvature scalar targets.
        regime_targets: (T,) tensor of regime labels.
        seq_len:        Length of input sequence.
    """

    def __init__(
        self,
        snapshots: List[Data],
        edge_ages: List[Tensor],
        ricci_targets: Tensor,
        regime_targets: Tensor,
        seq_len: int = 10,
    ):
        self.snapshots = snapshots
        self.edge_ages = edge_ages
        self.ricci_targets = ricci_targets
        self.regime_targets = regime_targets
        self.seq_len = seq_len
        # Valid starting positions: need seq_len steps + at least 1 step ahead
        self.valid_starts = list(range(len(snapshots) - seq_len))

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start = self.valid_starts[idx]
        end = start + self.seq_len
        return {
            "snapshots": self.snapshots[start:end],
            "edge_ages": self.edge_ages[start:end],
            "ricci_target": self.ricci_targets[end].item(),
            "regime_target": self.regime_targets[end].item(),
            "timestamps": torch.arange(start, end, dtype=torch.float),
        }


def temporal_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for TemporalGraphDataset (handles variable graph sizes)."""
    result = {
        "snapshots": [b["snapshots"] for b in batch],
        "edge_ages": [b["edge_ages"] for b in batch],
        "ricci_target": torch.tensor([b["ricci_target"] for b in batch]),
        "regime_target": torch.tensor([b["regime_target"] for b in batch], dtype=torch.long),
        "timestamps": torch.stack([b["timestamps"] for b in batch]),
    }
    return result


def compute_loss(
    outputs: Dict[str, Tensor],
    batch: Dict[str, Any],
    regime_weight: float = 1.0,
    ricci_weight: float = 0.5,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute combined loss for EvolutionaryGNN training.

    Args:
        outputs:        Model output dict.
        batch:          Batch dict with targets.
        regime_weight:  Weight for regime classification loss.
        ricci_weight:   Weight for Ricci curvature MSE loss.
    Returns:
        total_loss, loss_components dict.
    """
    regime_loss = F.cross_entropy(
        outputs["regime_logits"],
        batch["regime_target"].to(outputs["regime_logits"].device),
    )

    ricci_loss = F.mse_loss(
        outputs["ricci_pred"],
        batch["ricci_target"].to(outputs["ricci_pred"].device).float(),
    )

    # Contrastive loss on graph embeddings: push different-regime embeddings apart
    emb = outputs["graph_embedding"]
    labels = batch["regime_target"].to(emb.device)
    contrastive_loss = _supervised_contrastive_loss(emb, labels)

    total = regime_weight * regime_loss + ricci_weight * ricci_loss + 0.1 * contrastive_loss

    return total, {
        "regime": regime_loss.item(),
        "ricci": ricci_loss.item(),
        "contrastive": contrastive_loss.item(),
        "total": total.item(),
    }


def _supervised_contrastive_loss(embeddings: Tensor, labels: Tensor, temperature: float = 0.1) -> Tensor:
    """Supervised contrastive loss (Khosla et al. 2020)."""
    n = embeddings.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=embeddings.device)

    emb_norm = F.normalize(embeddings, dim=-1)
    sim = (emb_norm @ emb_norm.T) / temperature  # (N, N)

    # Mask self
    mask_self = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    sim = sim.masked_fill(mask_self, float("-inf"))

    # Positive mask: same label
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
    labels_eq = labels_eq & ~mask_self

    log_probs = F.log_softmax(sim, dim=-1)  # (N, N)

    # Mean log-prob of positive pairs
    n_pos = labels_eq.float().sum(dim=-1).clamp(min=1)
    loss = -(log_probs * labels_eq.float()).sum(dim=-1) / n_pos
    return loss.mean()


def train_evolutionary_gnn(
    model: EvolutionaryGNN,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 50,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Training loop for EvolutionaryGNN.

    Uses AdamW optimizer, cosine LR schedule, and gradient clipping.

    Args:
        model:          EvolutionaryGNN instance.
        train_loader:   DataLoader yielding temporal_collate batches.
        val_loader:     Optional validation DataLoader.
        epochs:         Number of training epochs.
        learning_rate:  Peak learning rate.
        weight_decay:   L2 regularization.
        grad_clip:      Max gradient norm.
        device:         torch device string.
        verbose:        Print progress.
    Returns:
        History dict with train_loss, val_loss lists.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_regime_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_correct = 0
        epoch_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Process each sample in the batch (graphs can have different sizes)
            batch_outputs = []
            for b_idx in range(len(batch["snapshots"])):
                snaps = [s.to(device) for s in batch["snapshots"][b_idx]]
                ages = [a.to(device) for a in batch["edge_ages"][b_idx]]
                ts = batch["timestamps"][b_idx].to(device)
                out = model(snaps, ages, ts)
                batch_outputs.append(out)

            # Stack outputs
            combined = {
                "regime_logits": torch.stack([o["regime_logits"].squeeze(0) for o in batch_outputs]),
                "ricci_pred": torch.cat([o["ricci_pred"] for o in batch_outputs]),
                "graph_embedding": torch.stack([o["graph_embedding"].squeeze(0) for o in batch_outputs]),
            }

            loss, components = compute_loss(combined, batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_losses.append(components["total"])

            # Regime accuracy
            preds = combined["regime_logits"].argmax(dim=-1)
            targets = batch["regime_target"].to(device)
            epoch_correct += (preds == targets).sum().item()
            epoch_total += targets.size(0)

        scheduler.step()

        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        regime_acc = epoch_correct / max(epoch_total, 1)
        history["train_loss"].append(avg_train_loss)
        history["train_regime_acc"].append(regime_acc)

        # Validation
        val_loss = None
        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, device)
            history["val_loss"].append(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            val_str = f"  val_loss={val_loss:.4f}" if val_loss is not None else ""
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_train_loss:.4f}  "
                  f"regime_acc={regime_acc:.3f}{val_str}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    return history


def _evaluate(
    model: EvolutionaryGNN,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch_outputs = []
            for b_idx in range(len(batch["snapshots"])):
                snaps = [s.to(device) for s in batch["snapshots"][b_idx]]
                ages = [a.to(device) for a in batch["edge_ages"][b_idx]]
                ts = batch["timestamps"][b_idx].to(device)
                out = model(snaps, ages, ts)
                batch_outputs.append(out)

            combined = {
                "regime_logits": torch.stack([o["regime_logits"].squeeze(0) for o in batch_outputs]),
                "ricci_pred": torch.cat([o["ricci_pred"] for o in batch_outputs]),
                "graph_embedding": torch.stack([o["graph_embedding"].squeeze(0) for o in batch_outputs]),
            }
            _, components = compute_loss(combined, batch)
            total_loss += components["total"]
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Utility: make synthetic snapshot ─────────────────────────────────────────

def make_synthetic_snapshot(
    n_nodes: int,
    node_features: int,
    edge_density: float = 0.3,
    seed: int = 0,
) -> Tuple[Data, Tensor]:
    """Generate a synthetic graph snapshot for testing."""
    torch.manual_seed(seed)
    x = torch.randn(n_nodes, node_features)

    # Random edges
    max_edges = n_nodes * (n_nodes - 1)
    n_edges = max(1, int(max_edges * edge_density))
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst])
    edge_attr = torch.rand(edge_index.size(1), 1)
    edge_age = torch.randint(0, 50, (edge_index.size(1),)).float()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), edge_age
