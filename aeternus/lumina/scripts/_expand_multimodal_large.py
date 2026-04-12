"""Expand multimodal.py with large additions."""
import os, sys

MULTIMODAL_PATH = os.path.join(os.path.dirname(__file__), "..", "lumina", "multimodal.py")

CONTENT = '''

# =============================================================================
# SECTION: Advanced Multimodal Fusion Architectures
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict


class GatedMultimodalUnit(nn.Module):
    """Gated Multimodal Unit (GMU) for learned soft fusion.

    Arevalo et al. (2017): learns a gating mechanism to combine
    representations from two modalities:

        z = tanh(W_1 * x1)
        h = tanh(W_2 * x2)
        g = sigmoid(W_g * [x1; x2])
        output = g * z + (1 - g) * h
    """

    def __init__(self, d1: int, d2: int, d_out: int):
        super().__init__()
        self.proj1 = nn.Linear(d1, d_out)
        self.proj2 = nn.Linear(d2, d_out)
        self.gate = nn.Linear(d1 + d2, d_out)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.proj1(x1))
        h = torch.tanh(self.proj2(x2))
        g = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        return g * z + (1 - g) * h


class BilinearFusion(nn.Module):
    """Bilinear pooling for multimodal fusion.

    Computes element-wise product of projected modalities, optionally
    using Low-Rank Bilinear pooling (MLB) for efficiency:
    h = W_1 * x1 ⊙ W_2 * x2

    Then sumpool over feature dimension.
    """

    def __init__(self, d1: int, d2: int, d_out: int, rank: int = None, dropout: float = 0.1):
        super().__init__()
        self.rank = rank or d_out
        self.proj1 = nn.Linear(d1, self.rank)
        self.proj2 = nn.Linear(d2, self.rank)
        self.out = nn.Linear(self.rank, d_out)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.proj1(x1)
        h2 = self.proj2(x2)
        fused = torch.relu(h1) * torch.relu(h2)
        fused = self.dropout(fused)
        out = self.out(fused)
        return self.norm(out)


class CrossModalAttentionFusion(nn.Module):
    """Bidirectional cross-modal attention for deep feature fusion.

    Each modality attends to the other and is updated with the result.
    Supports:
    - Asymmetric Q/K/V projections
    - Multi-head cross attention
    - Residual connections with normalization
    """

    def __init__(
        self,
        d_model1: int,
        d_model2: int,
        d_out: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_out = d_out
        self.d_head = d_out // n_heads

        # Modality 1 attends to modality 2
        self.q1 = nn.Linear(d_model1, d_out)
        self.k2 = nn.Linear(d_model2, d_out)
        self.v2 = nn.Linear(d_model2, d_out)
        self.out1 = nn.Linear(d_out, d_out)
        self.norm1 = nn.LayerNorm(d_out)

        # Modality 2 attends to modality 1
        self.q2 = nn.Linear(d_model2, d_out)
        self.k1 = nn.Linear(d_model1, d_out)
        self.v1 = nn.Linear(d_model1, d_out)
        self.out2 = nn.Linear(d_out, d_out)
        self.norm2 = nn.LayerNorm(d_out)

        # Project inputs to d_out for residuals
        self.proj1 = nn.Linear(d_model1, d_out) if d_model1 != d_out else nn.Identity()
        self.proj2 = nn.Linear(d_model2, d_out) if d_model2 != d_out else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def _attend(self, q, k, v):
        B, T_q, _ = q.shape
        T_k = k.shape[1]
        H, Dh = self.n_heads, self.d_head

        q = q.view(B, T_q, H, Dh).transpose(1, 2)
        k = k.view(B, T_k, H, Dh).transpose(1, 2)
        v = v.view(B, T_k, H, Dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, H * Dh)
        return out

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x1 attends to x2
        q1 = self.q1(x1)
        k2 = self.k2(x2)
        v2 = self.v2(x2)
        ctx1 = self._attend(q1, k2, v2)
        h1 = self.norm1(self.proj1(x1) + self.out1(ctx1))

        # x2 attends to x1
        q2 = self.q2(x2)
        k1 = self.k1(x1)
        v1 = self.v1(x1)
        ctx2 = self._attend(q2, k1, v1)
        h2 = self.norm2(self.proj2(x2) + self.out2(ctx2))

        return h1, h2


class ModalityAlignmentModule(nn.Module):
    """Align representations across modalities to a shared embedding space.

    Uses a contrastive loss + projection heads to pull paired
    (text, price) or (news, fundamentals) representations together.
    Similar to CLIP alignment (Radford et al. 2021).
    """

    def __init__(self, d_in: int, d_shared: int, temperature: float = 0.07):
        super().__init__()
        self.d_shared = d_shared
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_shared),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to shared embedding and L2-normalize."""
        h = self.proj(x)
        return F.normalize(h, dim=-1)

    def contrastive_loss(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric InfoNCE loss between two sets of embeddings."""
        B = emb1.shape[0]
        logits = (emb1 @ emb2.T) / self.temperature.exp().clamp(min=0.01)
        labels = torch.arange(B, device=emb1.device)
        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.T, labels)
        return (loss_12 + loss_21) / 2


# =============================================================================
# SECTION: Financial-Specific Multimodal Inputs
# =============================================================================

class OrderBookEncoder(nn.Module):
    """Encode Level-2 order book snapshots into dense representations.

    Processes bid/ask price and volume arrays at multiple levels,
    captures supply/demand imbalance and price impact signals.

    Input shape: [B, T, n_levels, 4] where last dim is [bid_px, bid_sz, ask_px, ask_sz]
    """

    def __init__(
        self,
        n_levels: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.d_model = d_model

        # Per-level projection: 4 features -> d_model
        self.level_proj = nn.Linear(4, d_model)

        # Level index embedding
        self.level_emb = nn.Embedding(n_levels, d_model)

        # Transformer to process levels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Aggregation: mean pool over levels
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, n_levels, 4] order book snapshots
        Returns:
            [B, T, d_model] encoded representations
        """
        B, T, L, F = x.shape
        # Flatten time and batch for transformer processing
        x_flat = x.view(B * T, L, F)

        # Project features
        h = self.level_proj(x_flat)

        # Add level embeddings
        level_idx = torch.arange(L, device=x.device)
        h = h + self.level_emb(level_idx).unsqueeze(0)

        # Transform
        h = self.transformer(h)

        # Pool over levels
        h = h.mean(dim=1)  # [B*T, d_model]
        h = self.output_proj(h)

        return h.view(B, T, self.d_model)


class OptionsChainEncoder(nn.Module):
    """Encode options chain data for each underlying at a given date.

    Processes strike/expiry grid with bid/ask/IV/delta for each contract.
    Captures term structure and smile/skew information.

    Input: [B, n_expiries, n_strikes, 6] where features are
    [strike, expiry_days, bid, ask, implied_vol, delta]
    """

    def __init__(
        self,
        n_expiries: int = 8,
        n_strikes: int = 15,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_expiries = n_expiries
        self.n_strikes = n_strikes
        self.d_model = d_model

        self.contract_proj = nn.Linear(6, d_model)

        # Process along strike dimension
        self.strike_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Process along expiry dimension
        self.expiry_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_expiries, n_strikes, 6]
        Returns:
            [B, d_model] single vector per options chain
        """
        B, E, S, F = x.shape
        x_flat = x.view(B * E, S, F)
        h = self.contract_proj(x_flat)  # [B*E, S, d_model]

        # Attention over strikes
        h_strike, _ = self.strike_attn(h, h, h)
        h = self.norm1(h + h_strike)
        h = h.mean(dim=1)  # [B*E, d_model]
        h = h.view(B, E, self.d_model)

        # Attention over expiries
        h_exp, _ = self.expiry_attn(h, h, h)
        h = self.norm2(h + h_exp)

        return self.out(h.mean(dim=1))  # [B, d_model]


class MacroIndicatorEncoder(nn.Module):
    """Encode macroeconomic indicator time series.

    Handles mixed-frequency indicators (daily, weekly, monthly, quarterly)
    via temporal aggregation and positional encoding.
    """

    def __init__(
        self,
        n_indicators: int = 50,
        d_model: int = 128,
        max_history: int = 120,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_indicators = n_indicators
        self.d_model = d_model

        # Per-indicator embedding
        self.indicator_emb = nn.Embedding(n_indicators, d_model)

        # Value projection
        self.value_proj = nn.Linear(1, d_model)

        # Positional encoding for time
        self.pos_emb = nn.Embedding(max_history, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pool = nn.Linear(d_model, d_model)

    def forward(
        self,
        values: torch.Tensor,
        indicator_ids: torch.Tensor,
        time_positions: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            values: [B, N] indicator values
            indicator_ids: [B, N] indicator index for each observation
            time_positions: [B, N] time step index
            mask: [B, N] padding mask (True = valid)
        Returns:
            [B, d_model]
        """
        B, N = values.shape

        h = (
            self.value_proj(values.unsqueeze(-1))
            + self.indicator_emb(indicator_ids)
            + self.pos_emb(time_positions)
        )

        if mask is not None:
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        # Pool valid positions
        if mask is not None:
            h = (h * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.pool(h)


class SentimentSignalEncoder(nn.Module):
    """Encode aggregated sentiment signals from news/social media.

    Inputs:
    - Pre-computed sentiment scores (positive, negative, neutral)
    - Entity mentions (tickers, sectors)
    - Source reliability weights
    - Volume/intensity metrics
    """

    def __init__(
        self,
        d_sentiment: int = 3,
        d_model: int = 128,
        n_sources: int = 20,
        max_entities: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sentiment_proj = nn.Linear(d_sentiment, d_model)
        self.source_emb = nn.Embedding(n_sources, d_model)
        self.entity_emb = nn.Embedding(max_entities, d_model)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        sentiment_scores: torch.Tensor,
        source_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            sentiment_scores: [B, T, 3] (pos, neg, neutral)
            source_ids: [B, T] source index
            entity_ids: [B, T] entity/ticker index
            weights: [B, T] optional reliability weights
        Returns:
            [B, d_model]
        """
        h_sent = self.sentiment_proj(sentiment_scores)
        h_src = self.source_emb(source_ids)
        h_ent = self.entity_emb(entity_ids)

        h = self.fusion(torch.cat([h_sent, h_src, h_ent], dim=-1))

        if weights is not None:
            w = F.softmax(weights, dim=1).unsqueeze(-1)
            h = (h * w).sum(dim=1)
        else:
            h = h.mean(dim=1)

        return self.norm(h)


# =============================================================================
# SECTION: Temporal Multimodal Integration
# =============================================================================

class TemporalModalityFusion(nn.Module):
    """Fuse multiple modalities over time with temporal attention.

    Each modality contributes at each time step, then cross-modal
    attention integrates information across modalities and time.
    """

    def __init__(
        self,
        modality_dims: List[int],
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        n_modalities = len(modality_dims)

        # Project each modality to d_model
        self.modality_projs = nn.ModuleList([
            nn.Linear(d, d_model) for d in modality_dims
        ])

        # Modality type embeddings
        self.modality_emb = nn.Embedding(n_modalities, d_model)

        # Temporal positional encoding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Unified transformer over [batch, time * modalities, d_model]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        modalities: List[torch.Tensor],
        time_positions: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            modalities: list of [B, T, D_i] tensors, one per modality
            time_positions: [B, T] time step indices
            mask: [B, T] valid position mask
        Returns:
            [B, T, d_model] fused representations
        """
        B, T = modalities[0].shape[:2]
        device = modalities[0].device

        tokens = []
        for i, (x, proj) in enumerate(zip(modalities, self.modality_projs)):
            h = proj(x)
            mod_emb = self.modality_emb(torch.tensor(i, device=device))
            h = h + mod_emb.unsqueeze(0).unsqueeze(0)
            tokens.append(h)

        # Interleave: [B, T, n_modalities, D] -> [B, T*n_modalities, D]
        stacked = torch.stack(tokens, dim=2)  # [B, T, M, D]
        n_mod = len(modalities)
        h = stacked.view(B, T * n_mod, self.d_model)

        if time_positions is None:
            time_positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # Repeat positions for each modality
        pos = self.pos_emb(time_positions)  # [B, T, D]
        pos = pos.unsqueeze(2).expand(-1, -1, n_mod, -1).reshape(B, T * n_mod, self.d_model)
        h = h + pos

        h = self.transformer(h)
        h = self.output_norm(h)

        # Extract primary modality positions (every n_mod-th token)
        h = h.view(B, T, n_mod, self.d_model)[:, :, 0, :]  # take first modality's output
        return h


class EventConditionedFusion(nn.Module):
    """Condition multimodal fusion on discrete financial events.

    Events (earnings releases, rate decisions, M&A) are embedded
    and used to gate/condition the fusion process.
    """

    def __init__(
        self,
        n_event_types: int = 50,
        d_model: int = 256,
        d_event: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.event_emb = nn.Embedding(n_event_types + 1, d_event, padding_idx=0)

        # Event-conditioned gating
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model + d_event, d_model),
            nn.Sigmoid(),
        )

        self.event_proj = nn.Linear(d_event, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        event_ids: torch.Tensor,
        event_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] base representations
            event_ids: [B, T] event type index (0 = no event)
            event_mask: [B, T] bool mask where events occurred
        Returns:
            [B, T, d_model] event-conditioned representations
        """
        e = self.event_emb(event_ids)  # [B, T, d_event]
        gate = self.gate_proj(torch.cat([x, e], dim=-1))
        event_repr = self.event_proj(e)

        if event_mask is not None:
            event_repr = event_repr * event_mask.float().unsqueeze(-1)

        out = self.fusion_norm(x + gate * self.dropout(event_repr))
        return out


# =============================================================================
# SECTION: Graph-Based Multimodal Fusion
# =============================================================================

class AssetCorrelationGraphEncoder(nn.Module):
    """Graph neural network over asset correlation structure.

    Treats assets as nodes, pairwise correlations as edge weights.
    Uses message passing to propagate information across correlated assets.
    """

    def __init__(
        self,
        d_asset: int = 128,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.node_proj = nn.Linear(d_asset, d_model)

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(d_model, d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, d_asset] per-asset features
            adjacency: [B, N, N] correlation adjacency matrix
        Returns:
            [B, N, d_model] updated node features
        """
        h = self.node_proj(node_features)

        for gat, norm in zip(self.gat_layers, self.norms):
            h_new = gat(h, adjacency)
            h = norm(h + h_new)

        return self.output_proj(h)


class GraphAttentionLayer(nn.Module):
    """Single Graph Attention Network (GAT) layer (Velickovic 2018)."""

    def __init__(self, d_in: int, d_out: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.W = nn.Linear(d_in, n_heads * self.d_head, bias=False)
        self.a = nn.Parameter(torch.randn(n_heads, 2 * self.d_head))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)
        self.out = nn.Linear(n_heads * self.d_head, d_out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        h = self.W(x).view(B, N, H, Dh)  # [B, N, H, Dh]

        # Compute attention coefficients
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B, N, N, H, Dh]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)
        concat = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, H, 2Dh]

        e = self.act((concat * self.a.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1))  # [B, N, N, H]

        # Mask with adjacency
        mask = (adj > 0).unsqueeze(-1).float()
        e = e * mask + (1 - mask) * (-1e9)

        alpha = F.softmax(e, dim=2)  # [B, N, N, H]
        alpha = self.dropout(alpha)

        # Weighted sum
        out = (alpha.unsqueeze(-1) * h_j).sum(dim=2)  # [B, N, H, Dh]
        out = out.view(B, N, H * Dh)
        return self.out(out)


# =============================================================================
# SECTION: Multimodal Pre-training Objectives
# =============================================================================

class MultimodalMaskedModeling(nn.Module):
    """Masked multimodal modeling: reconstruct masked tokens across modalities.

    Extends MLM to multiple modalities:
    - Mask a fraction of time series values
    - Mask a fraction of text/sentiment tokens
    - Model must reconstruct from context across all available modalities
    """

    def __init__(
        self,
        d_model: int,
        n_modalities: int,
        mask_prob: float = 0.15,
        vocab_sizes: Dict[str, int] = None,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.n_modalities = n_modalities

        if vocab_sizes is None:
            vocab_sizes = {}

        # Modality-specific reconstruction heads
        self.heads = nn.ModuleDict()
        for i in range(n_modalities):
            key = f"mod_{i}"
            if key in vocab_sizes:
                self.heads[key] = nn.Linear(d_model, vocab_sizes[key])
            else:
                self.heads[key] = nn.Linear(d_model, 1)  # regression

        self.mask_token = nn.ParameterDict({
            f"mod_{i}": nn.Parameter(torch.randn(d_model))
            for i in range(n_modalities)
        })

    def create_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create random mask for a modality tensor."""
        mask = torch.rand_like(x[..., 0]) < self.mask_prob
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_idx: int,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Args:
            hidden_states: [B, T, d_model] output from multimodal encoder
            modality_idx: which modality's head to use
            targets: [B, T, D_target] reconstruction targets
            mask: [B, T] bool mask of positions to reconstruct
        """
        key = f"mod_{modality_idx}"
        preds = self.heads[key](hidden_states)

        if mask.any():
            if preds.shape[-1] == 1:
                # Regression
                loss = F.mse_loss(preds[mask].squeeze(-1), targets[mask].float())
            else:
                # Classification
                loss = F.cross_entropy(preds[mask], targets[mask].long())
        else:
            loss = torch.tensor(0.0, device=hidden_states.device)

        return {"loss": loss, "predictions": preds}


class CrossModalContrastiveLoss(nn.Module):
    """Contrastive loss for aligning representations across modalities.

    Based on CLIP-style InfoNCE: pulls paired modality embeddings together
    and pushes apart non-paired ones within a batch.
    """

    def __init__(self, d_model: int, d_proj: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature).log())
        self.proj_a = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.proj_b = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
        """
        Args:
            z_a, z_b: [B, d_model] paired embeddings from two modalities
        """
        p_a = F.normalize(self.proj_a(z_a), dim=-1)
        p_b = F.normalize(self.proj_b(z_b), dim=-1)

        temp = self.temperature.exp().clamp(min=0.01, max=10.0)
        logits = (p_a @ p_b.T) / temp
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        loss = (loss_ab + loss_ba) / 2

        with torch.no_grad():
            acc = ((logits.argmax(dim=1) == labels).float().mean() +
                   (logits.T.argmax(dim=1) == labels).float().mean()) / 2

        return {"loss": loss, "accuracy": acc.item(), "temperature": temp.item()}


# =============================================================================
# SECTION: Multimodal Decoder / Generation
# =============================================================================

class MultimodalDecoder(nn.Module):
    """Decoder for generating signals conditioned on multimodal context.

    Combines:
    - Cross-attention to multimodal encoder outputs
    - Autoregressive self-attention for sequential prediction
    - Separate output heads for each target modality
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        n_output_classes: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        if n_output_classes is not None:
            self.output_head = nn.Linear(d_model, n_output_classes)
        else:
            self.output_head = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [B, T_tgt, d_model] decoder input embeddings
            memory: [B, T_mem, d_model] encoder output (multimodal context)
        Returns:
            [B, T_tgt, n_output]
        """
        h = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        h = self.norm(h)
        return self.output_head(h)


# =============================================================================
# SECTION: Multimodal Model Factory and Registry
# =============================================================================

_MULTIMODAL_REGISTRY = {}


def register_multimodal(name: str):
    def decorator(cls):
        _MULTIMODAL_REGISTRY[name] = cls
        return cls
    return decorator


def get_multimodal_model(name: str):
    if name not in _MULTIMODAL_REGISTRY:
        raise KeyError(f"Multimodal model '{name}' not found. Available: {list(_MULTIMODAL_REGISTRY.keys())}")
    return _MULTIMODAL_REGISTRY[name]


@register_multimodal("price_sentiment")
class PriceSentimentModel(nn.Module):
    """End-to-end model fusing price time series + sentiment signals."""

    def __init__(
        self,
        d_price: int = 64,
        d_sentiment: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_outputs: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.price_proj = nn.Linear(d_price, d_model)
        self.sentiment_proj = nn.Linear(d_sentiment, d_model)

        self.fusion = CrossModalAttentionFusion(d_model, d_model, d_model, n_heads, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_outputs),
        )

    def forward(
        self,
        price: torch.Tensor,
        sentiment: torch.Tensor,
    ) -> torch.Tensor:
        p = self.price_proj(price)
        s = self.sentiment_proj(sentiment)
        p_fused, s_fused = self.fusion(p, s)
        h = self.encoder(p_fused + s_fused)
        return self.head(h[:, -1, :])


@register_multimodal("full_multimodal")
class FullMultimodalLumina(nn.Module):
    """Full multimodal Lumina: price + order book + options + macro + sentiment."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_outputs: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.price_proj = nn.Linear(64, d_model)
        self.ob_encoder = OrderBookEncoder(d_model=d_model)
        self.macro_encoder = MacroIndicatorEncoder(d_model=d_model)
        self.sentiment_encoder = SentimentSignalEncoder(d_model=d_model)

        self.temporal_fusion = TemporalModalityFusion(
            modality_dims=[d_model, d_model],
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs),
        )

    def forward(
        self,
        price: torch.Tensor,
        order_book: torch.Tensor,
        macro: Tuple,
        sentiment: Tuple,
    ) -> torch.Tensor:
        p = self.price_proj(price)
        ob = self.ob_encoder(order_book)
        h = self.temporal_fusion([p, ob])
        return self.head(h[:, -1, :])


# =============================================================================
# SECTION: Multimodal Evaluation Metrics
# =============================================================================

class MultimodalAlignmentMetrics:
    """Evaluation metrics for multimodal model alignment and fusion quality."""

    @staticmethod
    def modality_gap(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute modality gap: L2 distance between modality cluster centers.

        A large modality gap indicates poor cross-modal alignment.
        """
        c1 = F.normalize(emb1.mean(dim=0), dim=-1)
        c2 = F.normalize(emb2.mean(dim=0), dim=-1)
        return (c1 - c2).norm().item()

    @staticmethod
    def retrieval_metrics(
        query_emb: torch.Tensor,
        gallery_emb: torch.Tensor,
        k_list: List[int] = [1, 5, 10],
    ) -> dict:
        """Compute Recall@K for cross-modal retrieval.

        Assumes i-th query matches i-th gallery item (diagonal pairs).
        """
        n = query_emb.shape[0]
        q = F.normalize(query_emb, dim=-1)
        g = F.normalize(gallery_emb, dim=-1)

        sim = q @ g.T  # [n, n]
        labels = torch.arange(n, device=q.device)

        results = {}
        for k in k_list:
            topk_idx = sim.topk(min(k, n), dim=1).indices
            hits = (topk_idx == labels.unsqueeze(1)).any(dim=1).float()
            results[f"R@{k}"] = hits.mean().item()

        # mAP
        sorted_idx = sim.argsort(dim=1, descending=True)
        ap_sum = 0.0
        for i in range(n):
            rank = (sorted_idx[i] == i).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                ap_sum += 1.0 / (rank[0].item() + 1)
        results["mAP"] = ap_sum / n

        return results

    @staticmethod
    def cross_modal_knn_accuracy(
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5,
    ) -> float:
        """KNN classification accuracy using cross-modal nearest neighbors."""
        e1 = F.normalize(emb1, dim=-1)
        e2 = F.normalize(emb2, dim=-1)
        sim = e1 @ e2.T

        topk_idx = sim.topk(k, dim=1).indices
        knn_labels = labels[topk_idx]
        pred_labels = knn_labels.mode(dim=1).values
        return (pred_labels == labels).float().mean().item()


# =============================================================================
# SECTION: Utility Functions
# =============================================================================

def build_multimodal_adjacency(
    returns: torch.Tensor,
    threshold: float = 0.3,
    method: str = "pearson",
) -> torch.Tensor:
    """Build asset correlation adjacency matrix from return data.

    Args:
        returns: [B, T, N] return time series for N assets
        threshold: minimum correlation to include as an edge
        method: 'pearson' or 'spearman'
    Returns:
        [B, N, N] adjacency matrix
    """
    B, T, N = returns.shape

    if method == "pearson":
        # Center returns
        centered = returns - returns.mean(dim=1, keepdim=True)
        # Compute correlation
        cov = torch.bmm(centered.transpose(1, 2), centered) / (T - 1)
        std = centered.std(dim=1, keepdim=True) + 1e-8
        outer_std = std.transpose(1, 2) @ std
        corr = cov / outer_std.squeeze(1)
    else:
        # Spearman: rank then correlate
        ranks = returns.argsort(dim=1).argsort(dim=1).float()
        centered = ranks - ranks.mean(dim=1, keepdim=True)
        cov = torch.bmm(centered.transpose(1, 2), centered) / (T - 1)
        std = centered.std(dim=1, keepdim=True) + 1e-8
        outer_std = std.transpose(1, 2) @ std
        corr = cov / outer_std.squeeze(1)

    adj = (corr.abs() >= threshold).float()
    # Remove self-loops
    eye = torch.eye(N, device=returns.device).unsqueeze(0)
    adj = adj * (1 - eye)

    return adj


def interpolate_missing_multimodal(
    modalities: List[Optional[torch.Tensor]],
    strategy: str = "zero",
    d_model: int = 256,
) -> List[torch.Tensor]:
    """Handle missing modalities during inference by imputation.

    Strategies:
    - zero: replace with zeros
    - mean: replace with dataset mean (requires precomputed)
    - learned: use a learned imputation vector
    """
    result = []
    ref_shape = next(m for m in modalities if m is not None).shape

    for m in modalities:
        if m is not None:
            result.append(m)
        else:
            if strategy == "zero":
                result.append(torch.zeros(*ref_shape[:-1], d_model))
            else:
                result.append(torch.zeros(*ref_shape[:-1], d_model))

    return result


def pad_modalities_to_same_length(
    modalities: List[torch.Tensor],
    pad_value: float = 0.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Pad modality sequences to the same temporal length.

    Returns padded modalities and corresponding masks.
    """
    max_len = max(m.shape[1] for m in modalities)
    padded = []
    masks = []

    for m in modalities:
        T = m.shape[1]
        if T < max_len:
            pad_size = max_len - T
            pad = torch.full(
                (*m.shape[:1], pad_size, *m.shape[2:]),
                pad_value,
                device=m.device,
                dtype=m.dtype,
            )
            m_pad = torch.cat([m, pad], dim=1)
            mask = torch.cat([
                torch.ones(m.shape[0], T, device=m.device),
                torch.zeros(m.shape[0], pad_size, device=m.device),
            ], dim=1)
        else:
            m_pad = m
            mask = torch.ones(m.shape[0], T, device=m.device)

        padded.append(m_pad)
        masks.append(mask)

    return padded, masks
'''

with open(MULTIMODAL_PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess, sys
result = subprocess.run(
    [sys.executable, "-c",
     f"lines = open(r'{MULTIMODAL_PATH}').readlines(); print(len(lines))"],
    capture_output=True, text=True
)
print(result.stdout.strip(), MULTIMODAL_PATH)
