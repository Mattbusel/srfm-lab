"""
lumina/multimodal.py

Multi-modal fusion components for Lumina:

  - CrossModalAttention
  - ModalityFusion (concat / cross-attn / gated)
  - TemporalAlignment
  - MultiModalLumina (full multi-modal model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .transformer import (
    RMSNorm, SwiGLUFFN, TransformerBlock, BidirectionalTransformer,
    CausalTransformer, LuminaModel, LuminaConfig,
)
from .positional_encoding import RotaryPositionalEncoding


# ---------------------------------------------------------------------------
# CrossModalAttention
# ---------------------------------------------------------------------------
class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention: Q from query modality, K/V from context modality.

    Used to let price representations attend to news, and vice versa.
    Supports optional RoPE on Q and K.
    """

    def __init__(
        self,
        d_query: int,
        d_context: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_query, d_model, bias=False)
        self.k_proj = nn.Linear(d_context, d_model, bias=False)
        self.v_proj = nn.Linear(d_context, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_query, bias=False)  # project back to query dim

        self.dropout = nn.Dropout(dropout)
        self.norm_q = RMSNorm(d_query)
        self.norm_c = RMSNorm(d_context)
        self.out_norm = RMSNorm(d_query)

        if use_rope:
            self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query:        (B, T_q, d_query)
            context:      (B, T_c, d_context)
            query_mask:   (B, T_q)
            context_mask: (B, T_c)

        Returns:
            attended: (B, T_q, d_query) — query enriched with context info
        """
        B, T_q, _ = query.shape
        T_c = context.shape[1]

        # Pre-norm
        q_norm = self.norm_q(query)
        c_norm = self.norm_c(context)

        Q = rearrange(self.q_proj(q_norm), "b t (h d) -> b h t d", h=self.n_heads)
        K = rearrange(self.k_proj(c_norm), "b t (h d) -> b h t d", h=self.n_heads)
        V = rearrange(self.v_proj(c_norm), "b t (h d) -> b h t d", h=self.n_heads)

        if self.use_rope:
            Q, K = self.rope(Q, K, seq_len=max(T_q, T_c))

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, n_heads, T_q, T_c)

        if context_mask is not None:
            # Mask out padding in context
            mask = context_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_c)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Handle all-masked rows (results in NaN from softmax)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, V)           # (B, n_heads, T_q, head_dim)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(out)              # (B, T_q, d_query)

        # Residual + norm
        result = self.out_norm(query + out)
        return result


# ---------------------------------------------------------------------------
# ModalityFusion
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """
    Gated fusion: learns a gate per modality per position.
    gate = sigmoid(W[x_1; x_2; ...; x_M])
    output = sum_i(gate_i * x_i)
    """

    def __init__(self, d_model: int, n_modalities: int):
        super().__init__()
        self.n_modalities = n_modalities
        self.gate_proj = nn.Linear(d_model * n_modalities, n_modalities)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, modality_reps: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_reps: list of (B, T, d_model) — same T for all

        Returns:
            fused: (B, T, d_model)
        """
        assert len(modality_reps) == self.n_modalities
        stacked = torch.stack(modality_reps, dim=2)  # (B, T, M, d_model)
        B, T, M, D = stacked.shape

        flat = stacked.view(B, T, M * D)  # (B, T, M*d_model)
        gates = torch.sigmoid(self.gate_proj(flat))  # (B, T, M)
        gates = gates.unsqueeze(-1)  # (B, T, M, 1)

        fused = (stacked * gates).sum(dim=2)  # (B, T, d_model)
        out = self.norm(self.out_proj(fused))
        return out


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: each modality queries all others.
    Each modality becomes a better representation by attending to the others.
    """

    def __init__(self, d_model: int, n_modalities: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_modalities = n_modalities
        # One cross-attn layer per (query, context) pair
        self.cross_attns = nn.ModuleDict({
            f"{i}_{j}": CrossModalAttention(d_model, d_model, d_model, n_heads, dropout)
            for i in range(n_modalities)
            for j in range(n_modalities)
            if i != j
        })
        self.merge_proj = nn.ModuleList([
            nn.Linear(d_model * n_modalities, d_model)
            for _ in range(n_modalities)
        ])
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_modalities)])

    def forward(
        self,
        modality_reps: List[torch.Tensor],
        modality_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            modality_reps:  list of (B, T_i, d_model)
            modality_masks: list of optional (B, T_i) masks

        Returns:
            updated_reps: list of (B, T_i, d_model) enriched representations
        """
        M = self.n_modalities
        if modality_masks is None:
            modality_masks = [None] * M

        updated = []
        for i in range(M):
            qi = modality_reps[i]  # (B, T_i, d_model)
            attended_parts = [qi]
            for j in range(M):
                if i == j:
                    continue
                key = f"{i}_{j}"
                if key in self.cross_attns:
                    att = self.cross_attns[key](
                        qi, modality_reps[j],
                        context_mask=modality_masks[j],
                    )
                    attended_parts.append(att)

            # If we only have the self part (no cross attention), just use it
            if len(attended_parts) == 1:
                updated.append(qi)
                continue

            # Pad all attended_parts to same T (they should match T_i already)
            combined = torch.cat(attended_parts, dim=-1)  # (B, T_i, d_model * M)
            # Project back to d_model
            # attended_parts has length M (1 self + M-1 cross)
            # But merge_proj[i] expects d_model * n_modalities input
            # Pad if fewer than M parts
            while len(attended_parts) < M:
                attended_parts.append(torch.zeros_like(qi))
            combined = torch.cat(attended_parts[:M], dim=-1)

            proj = self.merge_proj[i]
            out = self.norms[i](proj(combined))
            updated.append(out)

        return updated


class ModalityFusion(nn.Module):
    """
    Combines representations from all modalities.

    Three modes:
      - 'concat':     project concatenated reps to d_model
      - 'cross_attn': each modality attends to all others
      - 'gated':      learnable gate per modality per position
    """

    def __init__(
        self,
        d_model: int,
        n_modalities: int,
        n_heads: int = 8,
        mode: str = "gated",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert mode in ("concat", "cross_attn", "gated")
        self.mode = mode
        self.n_modalities = n_modalities
        self.d_model = d_model

        if mode == "concat":
            self.proj = nn.Linear(d_model * n_modalities, d_model)
            self.norm = RMSNorm(d_model)
        elif mode == "cross_attn":
            self.fusion = CrossAttentionFusion(d_model, n_modalities, n_heads, dropout)
            # Final concatenation + projection
            self.proj = nn.Linear(d_model * n_modalities, d_model)
            self.norm = RMSNorm(d_model)
        elif mode == "gated":
            self.fusion = GatedFusion(d_model, n_modalities)

    def forward(
        self,
        modality_reps: List[torch.Tensor],
        modality_masks: Optional[List[Optional[torch.Tensor]]] = None,
        target_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            modality_reps:  list of (B, T_i, d_model) — potentially different T
            modality_masks: optional masks
            target_len:     if set, all reps are interpolated/padded to this length

        Returns:
            fused: (B, target_len or T_0, d_model)
        """
        # Align all modalities to the same T (use first modality's length)
        T_ref = target_len if target_len is not None else modality_reps[0].shape[1]
        aligned = []
        for rep in modality_reps:
            if rep.shape[1] == T_ref:
                aligned.append(rep)
            elif rep.shape[1] > T_ref:
                aligned.append(rep[:, :T_ref])
            else:
                pad = torch.zeros(
                    rep.shape[0], T_ref - rep.shape[1], rep.shape[2],
                    device=rep.device, dtype=rep.dtype
                )
                aligned.append(torch.cat([rep, pad], dim=1))

        if self.mode == "concat":
            combined = torch.cat(aligned, dim=-1)  # (B, T, d_model * M)
            return self.norm(self.proj(combined))

        elif self.mode == "cross_attn":
            updated = self.fusion(aligned, modality_masks)
            combined = torch.cat(updated, dim=-1)
            return self.norm(self.proj(combined))

        elif self.mode == "gated":
            return self.fusion(aligned)


# ---------------------------------------------------------------------------
# TemporalAlignment
# ---------------------------------------------------------------------------
class TemporalAlignment(nn.Module):
    """
    Aligns asynchronous modalities to a common price-bar timeline.

    Methods:
      - nearest: snap each event timestamp to nearest price bar timestamp
      - interpolation: linear interpolation for continuous signals
    """

    def __init__(self, method: str = "nearest"):
        super().__init__()
        assert method in ("nearest", "interpolate")
        self.method = method

    def forward(
        self,
        price_timestamps: torch.Tensor,    # (B, T_price) unix seconds
        event_timestamps: torch.Tensor,    # (B, T_events) unix seconds
        event_values: torch.Tensor,        # (B, T_events, D) values to align
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align event_values to the price_timestamps grid.

        Returns:
            aligned_values: (B, T_price, D)
            coverage_mask:  (B, T_price) bool — True where at least one event mapped
        """
        B, T_price = price_timestamps.shape
        T_events = event_timestamps.shape[1]
        D = event_values.shape[2]
        device = price_timestamps.device

        aligned = torch.zeros(B, T_price, D, device=device, dtype=event_values.dtype)
        coverage = torch.zeros(B, T_price, dtype=torch.bool, device=device)
        counts = torch.zeros(B, T_price, device=device)

        if self.method == "nearest":
            for b in range(B):
                pt = price_timestamps[b]   # (T_price,)
                et = event_timestamps[b]   # (T_events,)

                # For each event, find nearest price bar
                # (T_events, 1) - (1, T_price) = (T_events, T_price) distances
                dist = (et.unsqueeze(1) - pt.unsqueeze(0)).abs()  # (T_events, T_price)
                nearest_idx = dist.argmin(dim=1)  # (T_events,)

                for ev_i in range(T_events):
                    bar_i = nearest_idx[ev_i].item()
                    aligned[b, bar_i] += event_values[b, ev_i]
                    counts[b, bar_i] += 1.0
                    coverage[b, bar_i] = True

                # Average multiple events at same bar
                nonzero = counts[b] > 0
                aligned[b, nonzero] = aligned[b, nonzero] / counts[b, nonzero].unsqueeze(-1)

        elif self.method == "interpolate":
            # Linear interpolation between events
            for b in range(B):
                pt = price_timestamps[b].float()  # (T_price,)
                et = event_timestamps[b].float()  # (T_events,)
                ev = event_values[b].float()       # (T_events, D)

                if T_events == 1:
                    # Single event: replicate
                    aligned[b] = ev[0].unsqueeze(0).expand(T_price, -1)
                    coverage[b] = True
                else:
                    # For each price bar, interpolate
                    for p_i in range(T_price):
                        t = pt[p_i]
                        # Find surrounding events
                        left_mask = et <= t
                        right_mask = et >= t

                        if left_mask.any() and right_mask.any():
                            left_idx = left_mask.nonzero()[-1].item()
                            right_idx = right_mask.nonzero()[0].item()

                            if left_idx == right_idx:
                                aligned[b, p_i] = ev[left_idx]
                            else:
                                t_left = et[left_idx]
                                t_right = et[right_idx]
                                alpha = (t - t_left) / (t_right - t_left + 1e-8)
                                aligned[b, p_i] = (1 - alpha) * ev[left_idx] + alpha * ev[right_idx]
                            coverage[b, p_i] = True
                        elif left_mask.any():
                            aligned[b, p_i] = ev[left_mask.nonzero()[-1].item()]
                            coverage[b, p_i] = True
                        elif right_mask.any():
                            aligned[b, p_i] = ev[right_mask.nonzero()[0].item()]
                            coverage[b, p_i] = True

        return aligned, coverage


# ---------------------------------------------------------------------------
# MultiModalLumina — Full Multi-Modal Model
# ---------------------------------------------------------------------------
@dataclass
class MultiModalLuminaConfig:
    # Per-modality encoder configs
    price_encoder: LuminaConfig = field(default_factory=lambda: LuminaConfig(
        d_model=256, n_layers=4, n_heads=4, n_kv_heads=2,
        use_moe=False, arch="bidirectional", lm_head=False, pool_head=False,
    ))
    onchain_encoder: LuminaConfig = field(default_factory=lambda: LuminaConfig(
        d_model=128, n_layers=2, n_heads=4, n_kv_heads=None,
        use_moe=False, arch="bidirectional", lm_head=False, pool_head=False,
    ))
    news_encoder: LuminaConfig = field(default_factory=lambda: LuminaConfig(
        d_model=256, n_layers=4, n_heads=4, n_kv_heads=2,
        use_moe=False, arch="bidirectional", lm_head=False, pool_head=False,
    ))

    # Shared encoder (alternative to per-modality): uses single LuminaModel
    use_shared_encoder: bool = False
    shared_encoder: LuminaConfig = field(default_factory=lambda: LuminaConfig(
        d_model=512, n_layers=12, n_heads=8, n_kv_heads=2,
        use_moe=True, arch="bidirectional", lm_head=True, pool_head=True,
    ))

    # Fusion
    d_fusion: int = 512
    fusion_mode: str = "gated"  # "concat", "cross_attn", "gated"
    fusion_n_heads: int = 8

    # Cross-modal attention layers
    n_cross_attn_layers: int = 2

    # Task heads
    n_regimes: int = 8
    forecast_horizon: int = 5  # bars ahead for volatility forecast

    # Input dims (must match tokenizer output)
    unified_token_dim: int = 256


class MultiModalLumina(nn.Module):
    """
    Full multi-modal Lumina model.

    Architecture:
      1. Per-modality encoders (or shared encoder)
      2. Cross-modal attention between modality pairs
      3. Unified fusion layer
      4. Task heads: next-return, regime classification, volatility forecast
    """

    def __init__(self, config: MultiModalLuminaConfig):
        super().__init__()
        self.config = config

        if config.use_shared_encoder:
            # Single shared encoder for all modalities
            self.shared_encoder = LuminaModel(config.shared_encoder)
            d_enc = config.shared_encoder.d_model
        else:
            # Per-modality encoders
            self.price_encoder = LuminaModel(config.price_encoder)
            self.onchain_encoder = LuminaModel(config.onchain_encoder)
            self.news_encoder = LuminaModel(config.news_encoder)

            # Project all encoder outputs to d_fusion
            self.price_proj = nn.Linear(config.price_encoder.d_model, config.d_fusion)
            self.onchain_proj = nn.Linear(config.onchain_encoder.d_model, config.d_fusion)
            self.news_proj = nn.Linear(config.news_encoder.d_model, config.d_fusion)
            d_enc = config.d_fusion

        # Cross-modal attention layers (applied after per-modality encoding)
        # Price ← News, News ← Price, OnChain ← Price
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "price_from_news": CrossModalAttention(
                    d_enc, d_enc, d_enc, config.fusion_n_heads,
                ),
                "news_from_price": CrossModalAttention(
                    d_enc, d_enc, d_enc, config.fusion_n_heads,
                ),
                "onchain_from_price": CrossModalAttention(
                    d_enc, d_enc, d_enc, config.fusion_n_heads,
                ),
            })
            for _ in range(config.n_cross_attn_layers)
        ])

        # Modality fusion
        self.fusion = ModalityFusion(
            d_model=config.d_fusion,
            n_modalities=3,  # price, onchain, news
            n_heads=config.fusion_n_heads,
            mode=config.fusion_mode,
        )

        # Final norm after fusion
        self.fusion_norm = RMSNorm(config.d_fusion)

        # Task heads
        # 1. Next-return prediction (regression)
        self.return_head = nn.Sequential(
            RMSNorm(config.d_fusion),
            nn.Linear(config.d_fusion, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # 2. Regime classification (8-class)
        self.regime_head = nn.Sequential(
            RMSNorm(config.d_fusion),
            nn.Linear(config.d_fusion, 64),
            nn.SiLU(),
            nn.Linear(64, config.n_regimes),
        )

        # 3. Volatility forecast (N-bar ahead)
        self.vol_head = nn.Sequential(
            RMSNorm(config.d_fusion),
            nn.Linear(config.d_fusion, 64),
            nn.SiLU(),
            nn.Linear(64, config.forecast_horizon),  # one per horizon bar
        )

        # 4. Crisis detection (binary)
        self.crisis_head = nn.Sequential(
            RMSNorm(config.d_fusion),
            nn.Linear(config.d_fusion, 32),
            nn.SiLU(),
            nn.Linear(32, 2),  # logits for [no_crisis, crisis]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_per_modality(
        self,
        price_tokens: Optional[torch.Tensor],
        onchain_tokens: Optional[torch.Tensor],
        news_tokens: Optional[torch.Tensor],
        price_mask: Optional[torch.Tensor],
        onchain_mask: Optional[torch.Tensor],
        news_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run per-modality encoders."""
        if self.config.use_shared_encoder:
            enc_fn = lambda tok, mask: self.shared_encoder(
                tok, attention_mask=mask
            )["hidden"] if tok is not None else None
            price_h = enc_fn(price_tokens, price_mask)
            onchain_h = enc_fn(onchain_tokens, onchain_mask)
            news_h = enc_fn(news_tokens, news_mask)
        else:
            price_h = self.price_encoder(
                price_tokens, attention_mask=price_mask
            )["hidden"] if price_tokens is not None else None
            onchain_h = self.onchain_encoder(
                onchain_tokens, attention_mask=onchain_mask
            )["hidden"] if onchain_tokens is not None else None
            news_h = self.news_encoder(
                news_tokens, attention_mask=news_mask
            )["hidden"] if news_tokens is not None else None

            # Project to d_fusion
            if price_h is not None:
                price_h = self.price_proj(price_h)
            if onchain_h is not None:
                onchain_h = self.onchain_proj(onchain_h)
            if news_h is not None:
                news_h = self.news_proj(news_h)

        return price_h, onchain_h, news_h

    def forward(
        self,
        price_tokens: Optional[torch.Tensor] = None,
        onchain_tokens: Optional[torch.Tensor] = None,
        news_tokens: Optional[torch.Tensor] = None,
        price_mask: Optional[torch.Tensor] = None,
        onchain_mask: Optional[torch.Tensor] = None,
        news_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            price_tokens:   (B, T_price, unified_token_dim)
            onchain_tokens: (B, T_onchain, unified_token_dim)
            news_tokens:    (B, T_news, unified_token_dim)
            *_mask:         corresponding boolean masks

        Returns:
            dict with: 'return_pred', 'regime_logits', 'vol_forecast',
                       'crisis_logits', 'fused', 'aux_loss'
        """
        # Step 1: Per-modality encoding
        price_h, onchain_h, news_h = self._encode_per_modality(
            price_tokens, onchain_tokens, news_tokens,
            price_mask, onchain_mask, news_mask,
        )

        # Determine reference T from available modalities
        T_ref = None
        for h in [price_h, onchain_h, news_h]:
            if h is not None:
                T_ref = h.shape[1]
                break

        if T_ref is None:
            raise ValueError("At least one modality must be provided.")

        # Create dummy reps for missing modalities
        B = next(h for h in [price_h, onchain_h, news_h] if h is not None).shape[0]
        d_fusion = self.config.d_fusion
        device = next(h for h in [price_h, onchain_h, news_h] if h is not None).device

        def _zero_rep():
            return torch.zeros(B, T_ref, d_fusion, device=device)

        if price_h is None:
            price_h = _zero_rep()
        if onchain_h is None:
            onchain_h = _zero_rep()
        if news_h is None:
            news_h = _zero_rep()

        # Step 2: Cross-modal attention (stacked layers)
        for layer in self.cross_attn_layers:
            price_h = layer["price_from_news"](price_h, news_h, context_mask=news_mask)
            news_h = layer["news_from_price"](news_h, price_h, context_mask=price_mask)
            onchain_h = layer["onchain_from_price"](onchain_h, price_h, context_mask=price_mask)

        # Step 3: Fusion
        fused = self.fusion(
            [price_h, onchain_h, news_h],
            modality_masks=[price_mask, onchain_mask, news_mask],
            target_len=T_ref,
        )
        fused = self.fusion_norm(fused)  # (B, T_ref, d_fusion)

        # Step 4: Global pooling for task heads
        # Use mean pooling with mask
        if price_mask is not None:
            mask_f = price_mask.float().unsqueeze(-1)
            # Truncate/pad mask to T_ref
            if mask_f.shape[1] >= T_ref:
                mask_f = mask_f[:, :T_ref]
            else:
                pad = torch.ones(B, T_ref - mask_f.shape[1], 1, device=device)
                mask_f = torch.cat([mask_f, pad], dim=1)
            pooled = (fused * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
        else:
            pooled = fused.mean(dim=1)  # (B, d_fusion)

        # Also use last token for autoregressive tasks
        last_token = fused[:, -1, :]  # (B, d_fusion)

        results = {
            "fused": fused,
            "pooled": pooled,
            "return_pred": self.return_head(last_token),        # (B, 1)
            "regime_logits": self.regime_head(pooled),           # (B, n_regimes)
            "vol_forecast": self.vol_head(pooled).abs(),         # (B, forecast_horizon) — positive
            "crisis_logits": self.crisis_head(pooled),           # (B, 2)
            "aux_loss": torch.tensor(0.0, device=device),
        }

        return results

    def save_pretrained(self, path: str):
        import os, json, pickle
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/model.pt")
        with open(f"{path}/config.pkl", "wb") as f:
            pickle.dump(self.config, f)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "MultiModalLumina":
        import pickle
        with open(f"{path}/config.pkl", "rb") as f:
            config = pickle.load(f)
        model = cls(config)
        model.load_state_dict(torch.load(f"{path}/model.pt", map_location=device))
        return model
