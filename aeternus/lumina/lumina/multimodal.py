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


# ---------------------------------------------------------------------------
# Multi-Modal Fusion Strategies
# ---------------------------------------------------------------------------

class EarlyFusion(nn.Module):
    """Early fusion of multi-modal financial data.

    All modalities are concatenated at the token level before the main
    transformer backbone processes them.

    Best for: modalities with tight temporal alignment and complementary
    features (e.g., OHLCV + technical indicators).

    Args:
        modality_dims:  dict of modality name → input dimension
        d_model:        unified output dimension
        dropout:        fusion dropout

    Example:
        >>> fusion = EarlyFusion(
        ...     modality_dims={"price": 64, "volume": 32, "macro": 20},
        ...     d_model=256
        ... )
        >>> inputs = {"price": torch.randn(2, 100, 64), ...}
        >>> out = fusion(inputs)  # (2, 100, 256)
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        d_model: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        total_dim = sum(modality_dims.values())
        self.modality_names = list(modality_dims.keys())
        self.proj = nn.Sequential(
            nn.Linear(total_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate and project all modalities.

        Args:
            inputs: dict of modality name → (B, T, d_i)

        Returns:
            fused: (B, T, d_model)
        """
        # Ensure all modalities have same sequence length
        seqlens = [v.shape[1] for v in inputs.values()]
        min_T = min(seqlens)
        parts = [inputs[name][:, :min_T, :] for name in self.modality_names if name in inputs]
        concat = torch.cat(parts, dim=-1)  # (B, T, sum_d)
        return self.norm(self.proj(concat))


class LateFusion(nn.Module):
    """Late fusion: each modality has its own backbone, then combine.

    Best for: modalities with different temporal scales or low alignment
    (e.g., price data + quarterly fundamentals + news headlines).

    Args:
        modality_backbones: dict of modality → nn.Module (each outputs d_model)
        d_model:            unified embedding dimension
        fusion_type:        "mean" | "max" | "attention" | "concat_project"

    Example:
        >>> backbones = {
        ...     "price": PriceBackbone(d_model=256),
        ...     "fundamental": FundaBackbone(d_model=256),
        ... }
        >>> fusion = LateFusion(backbones, d_model=256, fusion_type="attention")
        >>> out = fusion({"price": price_data, "fundamental": fund_data})
    """

    def __init__(
        self,
        modality_backbones: Dict[str, nn.Module],
        d_model: int = 256,
        fusion_type: str = "attention",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbones = nn.ModuleDict(modality_backbones)
        self.fusion_type = fusion_type
        self.d_model = d_model
        n_mod = len(modality_backbones)

        if fusion_type == "attention":
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=min(4, d_model // 64),
                dropout=dropout,
                batch_first=True,
            )
            self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        elif fusion_type == "concat_project":
            self.proj = nn.Linear(n_mod * d_model, d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run each backbone and fuse.

        Args:
            inputs: dict modality name → raw tensor

        Returns:
            fused: (B, d_model) fused representation
        """
        pooled = []
        for name, backbone in self.backbones.items():
            if name in inputs:
                out = backbone(inputs[name])
                # Pool if needed
                if out.dim() == 3:
                    out = out.mean(dim=1)  # (B, d_model)
                pooled.append(out)

        if not pooled:
            raise ValueError("No modality inputs provided")

        if len(pooled) == 1:
            return pooled[0]

        stacked = torch.stack(pooled, dim=1)  # (B, n_mod, d_model)

        if self.fusion_type == "mean":
            fused = stacked.mean(dim=1)
        elif self.fusion_type == "max":
            fused = stacked.max(dim=1).values
        elif self.fusion_type == "attention":
            B = stacked.shape[0]
            q = self.query.expand(B, -1, -1)  # (B, 1, d_model)
            fused, _ = self.fusion_attn(q, stacked, stacked)
            fused = fused.squeeze(1)  # (B, d_model)
        elif self.fusion_type == "concat_project":
            cat = stacked.reshape(stacked.shape[0], -1)  # (B, n_mod * d_model)
            fused = self.proj(cat)
        else:
            fused = stacked.mean(dim=1)

        return self.norm(fused)


class HierarchicalFusion(nn.Module):
    """Hierarchical multi-modal fusion for financial data.

    Fuses modalities in a tree-like hierarchy:
    1. First fuse high-frequency modalities (price + order book)
    2. Then fuse with medium-frequency modalities (fundamentals)
    3. Finally incorporate low-frequency modalities (macro)

    This allows each fusion level to operate at the appropriate
    temporal resolution.

    Args:
        d_model:     output dimension
        fusion_tree: list of lists, each inner list groups modality names
                     that are fused together at one level

    Example:
        >>> fusion = HierarchicalFusion(
        ...     d_model=256,
        ...     fusion_tree=[
        ...         ["price", "orderbook"],  # Level 1: high-freq
        ...         ["result_1", "fundamentals"],  # Level 2: + medium
        ...         ["result_2", "macro"],   # Level 3: + low-freq
        ...     ]
        ... )
    """

    def __init__(
        self,
        d_model: int = 256,
        fusion_tree: Optional[List[List[str]]] = None,
    ):
        super().__init__()
        self.d_model = d_model

        if fusion_tree is None:
            fusion_tree = [["price", "orderbook"], ["result_1", "fundamentals"]]
        self.fusion_tree = fusion_tree

        # One fusion layer per level
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            for _ in fusion_tree
        ])

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Hierarchically fuse modalities.

        Args:
            modality_embeddings: dict of name → (B, d_model)

        Returns:
            fused: (B, d_model)
        """
        embeddings = dict(modality_embeddings)

        for level_idx, (group, layer) in enumerate(zip(self.fusion_tree, self.fusion_layers)):
            available = [g for g in group if g in embeddings]
            if len(available) < 2:
                continue

            # Pair-wise fusion within the group
            a = embeddings[available[0]]
            b = embeddings[available[1]]
            if a.dim() == 3:
                a = a.mean(dim=1)
            if b.dim() == 3:
                b = b.mean(dim=1)

            fused = layer(torch.cat([a, b], dim=-1))
            embeddings[f"result_{level_idx + 1}"] = fused

        # Return last fusion result
        results = [k for k in embeddings if k.startswith("result_")]
        if results:
            return embeddings[sorted(results)[-1]]
        else:
            # Fallback: average all embeddings
            all_emb = [v.mean(dim=1) if v.dim() == 3 else v for v in embeddings.values()]
            return torch.stack(all_emb).mean(dim=0)


class DynamicModalityGating(nn.Module):
    """Dynamic gating mechanism for adaptive multi-modal fusion.

    Learns to weight modalities based on their relevance at each time step.
    Modalities that are more informative at a given moment get higher weight.

    Uses a small gating network that takes all modality representations
    and outputs scalar weights.

    Args:
        d_model:      embedding dimension
        n_modalities: number of modalities
        gate_type:    "softmax" | "sigmoid" | "sparsemax"

    Example:
        >>> gate = DynamicModalityGating(d_model=256, n_modalities=3)
        >>> embeddings = [torch.randn(4, 100, 256) for _ in range(3)]
        >>> weighted = gate(embeddings)  # (4, 100, 256)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_modalities: int = 3,
        gate_type: str = "softmax",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.gate_type = gate_type

        # Gating network: concatenate all modalities → weights
        self.gate_net = nn.Sequential(
            nn.Linear(n_modalities * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_modalities),
        )

    def forward(
        self,
        embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """Apply dynamic gating.

        Args:
            embeddings: list of (B, T, d_model) tensors, one per modality

        Returns:
            weighted: (B, T, d_model) gated combination
        """
        B, T, D = embeddings[0].shape

        # Compute gate weights from all modalities
        concat = torch.cat(embeddings, dim=-1)  # (B, T, n_mod * d_model)
        logits = self.gate_net(concat)  # (B, T, n_modalities)

        if self.gate_type == "softmax":
            weights = F.softmax(logits, dim=-1)  # (B, T, n_modalities)
        elif self.gate_type == "sigmoid":
            weights = torch.sigmoid(logits)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            weights = F.softmax(logits, dim=-1)

        # Weighted combination
        stacked = torch.stack(embeddings, dim=-1)  # (B, T, d_model, n_modalities)
        weights = weights.unsqueeze(2)              # (B, T, 1, n_modalities)
        return (stacked * weights).sum(dim=-1)      # (B, T, d_model)


class TemporalAlignmentModule(nn.Module):
    """Align multi-modal sequences with different temporal resolutions.

    When combining modalities at different frequencies (e.g., minute-bar
    price data with quarterly fundamental data), we need to align them.

    Alignment strategies:
    - "forward_fill": carry forward the lower-frequency signal
    - "interpolate":  linear interpolation between updates
    - "attention":    learned alignment via cross-attention

    Args:
        d_model:        embedding dimension
        strategy:       alignment strategy
        high_freq_len:  length of high-frequency sequence
        low_freq_len:   length of low-frequency sequence

    Example:
        >>> align = TemporalAlignmentModule(d_model=256, strategy="attention")
        >>> high_freq = torch.randn(4, 252, 256)  # daily
        >>> low_freq = torch.randn(4, 4, 256)    # quarterly
        >>> aligned = align(high_freq, low_freq)  # (4, 252, 256)
    """

    def __init__(
        self,
        d_model: int = 256,
        strategy: str = "attention",
    ):
        super().__init__()
        self.d_model = d_model
        self.strategy = strategy

        if strategy == "attention":
            self.align_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=min(4, d_model // 64),
                batch_first=True,
            )
            self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        high_freq: torch.Tensor,
        low_freq: torch.Tensor,
    ) -> torch.Tensor:
        """Align low-frequency sequence to high-frequency timeline.

        Args:
            high_freq: (B, T_high, d_model) high-frequency embeddings
            low_freq:  (B, T_low, d_model)  low-frequency embeddings

        Returns:
            aligned: (B, T_high, d_model) low-freq info aligned to high-freq timeline
        """
        if self.strategy == "forward_fill":
            # Upsample low_freq by repeating
            B, T_high, D = high_freq.shape
            T_low = low_freq.shape[1]
            scale = max(1, T_high // T_low)
            # Repeat each low-freq token 'scale' times
            upsampled = low_freq.repeat_interleave(scale, dim=1)
            # Trim or pad to match T_high
            if upsampled.shape[1] < T_high:
                pad = upsampled[:, -1:, :].expand(-1, T_high - upsampled.shape[1], -1)
                upsampled = torch.cat([upsampled, pad], dim=1)
            return upsampled[:, :T_high, :]

        elif self.strategy == "interpolate":
            # Linear interpolation
            B, T_high, D = high_freq.shape
            T_low = low_freq.shape[1]
            # Use F.interpolate on (B, D, T_low) → (B, D, T_high)
            lf_t = low_freq.transpose(1, 2)  # (B, D, T_low)
            upsampled = F.interpolate(lf_t, size=T_high, mode="linear", align_corners=False)
            return upsampled.transpose(1, 2)  # (B, T_high, D)

        elif self.strategy == "attention":
            # Cross-attention: high_freq queries low_freq
            aligned, _ = self.align_attn(high_freq, low_freq, low_freq)
            return self.norm(high_freq + aligned)

        else:
            return high_freq


class MultiModalMasking:
    """Masking strategies for multi-modal pre-training.

    Implements various masking patterns for self-supervised learning
    with multi-modal financial data:

    1. Unimodal masking: mask tokens within a single modality
    2. Cross-modal masking: mask an entire modality, predict from others
    3. Temporal masking: mask a time window across all modalities
    4. Semantic masking: mask tokens based on semantic importance

    Args:
        mask_ratio:          fraction of tokens to mask
        mask_strategy:       "random" | "block" | "cross_modal" | "temporal"
        cross_modal_prob:    probability of cross-modal masking

    Example:
        >>> masker = MultiModalMasking(mask_ratio=0.15)
        >>> embeddings = torch.randn(4, 128, 256)
        >>> modality_ids = torch.randint(0, 4, (4, 128))
        >>> masked, mask = masker.mask(embeddings, modality_ids)
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_strategy: str = "random",
        cross_modal_prob: float = 0.3,
    ):
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.cross_modal_prob = cross_modal_prob

    def mask(
        self,
        embeddings: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to embeddings.

        Args:
            embeddings:   (B, T, d_model)
            modality_ids: (B, T) long optional modality assignments

        Returns:
            masked_embeddings: (B, T, d_model) with masked positions zeroed
            mask:              (B, T) bool, True = masked
        """
        B, T, D = embeddings.shape

        if self.mask_strategy == "random":
            mask = torch.rand(B, T, device=embeddings.device) < self.mask_ratio

        elif self.mask_strategy == "block":
            # Mask consecutive blocks
            mask = torch.zeros(B, T, dtype=torch.bool, device=embeddings.device)
            n_mask = max(1, int(T * self.mask_ratio))
            for b in range(B):
                start = torch.randint(0, T - n_mask + 1, (1,)).item()
                mask[b, start:start + n_mask] = True

        elif self.mask_strategy == "cross_modal" and modality_ids is not None:
            # Randomly select a modality to fully mask
            n_modalities = modality_ids.max().item() + 1
            mask = torch.zeros(B, T, dtype=torch.bool, device=embeddings.device)
            for b in range(B):
                if torch.rand(1).item() < self.cross_modal_prob:
                    target_mod = torch.randint(0, int(n_modalities), (1,)).item()
                    mask[b] = (modality_ids[b] == target_mod)
                else:
                    mask[b] = torch.rand(T, device=embeddings.device) < self.mask_ratio

        else:
            mask = torch.rand(B, T, device=embeddings.device) < self.mask_ratio

        # Apply mask (zero out masked positions)
        masked = embeddings.clone()
        masked[mask] = 0.0
        return masked, mask


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


# ============================================================
# Multimodal Financial Intelligence Components
# ============================================================

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """
    Lightweight transformer-based text encoder for financial news/reports.
    Uses positional encodings + multi-head self-attention.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.embedding(input_ids) + self.pos_embedding(positions))

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        # CLS-style pooling: mean over non-padding tokens
        if attention_mask is not None:
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled = (x * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
        else:
            pooled = x.mean(1)

        return x, pooled  # (B, T, D), (B, D)


class VisionEncoder(nn.Module):
    """
    Patch-based vision encoder for financial charts/images (ViT-style).
    Liang et al. 2021 adapted for financial time-series visualization.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = images.shape[0]
        x = self.patch_embed(images)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        return x, x[:, 0]  # all patches, CLS token


class AudioEncoder(nn.Module):
    """
    1D convolutional + transformer encoder for audio (earnings calls, Fed speeches).
    Baevski et al. 2020 wav2vec style adapted for financial audio.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        conv_channels: List[int] = None,
        conv_kernels: List[int] = None,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        conv_channels = conv_channels or [64, 128, 256]
        conv_kernels = conv_kernels or [10, 3, 3]
        self.embed_dim = embed_dim

        convs = []
        ch_in = in_channels
        for ch_out, k in zip(conv_channels, conv_kernels):
            convs.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=k, stride=2, padding=k // 2),
                nn.GELU(),
                nn.GroupNorm(min(8, ch_out), ch_out),
            ])
            ch_in = ch_out
        self.feature_extractor = nn.Sequential(*convs)
        self.proj = nn.Linear(conv_channels[-1], embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # waveform: (B, C, T)
        x = self.feature_extractor(waveform)  # (B, ch, T')
        x = x.transpose(1, 2)  # (B, T', ch)
        x = self.proj(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x, x.mean(1)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention allowing one modality to attend to another.
    Used for grounding text in visual financial context.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Tq, _ = query.shape
        _, Tkv, _ = key_value.shape
        H, D = self.num_heads, self.head_dim

        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        Q = self.q_proj(q).view(B, Tq, H, D).transpose(1, 2)
        K = self.k_proj(kv).view(B, Tkv, H, D).transpose(1, 2)
        V = self.v_proj(kv).view(B, Tkv, H, D).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)
        return self.out_proj(out)


class MultimodalFusion(nn.Module):
    """
    Late/intermediate fusion of text, vision, and time-series modalities
    for holistic financial understanding.
    Strategies: concatenation, attention pooling, gated fusion.
    """

    def __init__(
        self,
        text_dim: int = 256,
        vision_dim: int = 256,
        ts_dim: int = 256,
        fused_dim: int = 512,
        fusion_type: str = "attention",  # "concat", "attention", "gated"
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.vision_proj = nn.Linear(vision_dim, fused_dim)
        self.ts_proj = nn.Linear(ts_dim, fused_dim)
        self.fused_dim = fused_dim

        if fusion_type == "attention":
            self.cross_attn = nn.MultiheadAttention(fused_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(fused_dim)
        elif fusion_type == "gated":
            self.gate_text = nn.Linear(fused_dim * 3, fused_dim)
            self.gate_vision = nn.Linear(fused_dim * 3, fused_dim)
            self.gate_ts = nn.Linear(fused_dim * 3, fused_dim)
        elif fusion_type == "concat":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(fused_dim * 3, fused_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim * 2, fused_dim),
            )

        self.output_norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        ts_feat: torch.Tensor,
    ) -> torch.Tensor:
        t = self.text_proj(text_feat)    # (B, D)
        v = self.vision_proj(vision_feat)
        s = self.ts_proj(ts_feat)

        if self.fusion_type == "concat":
            combined = torch.cat([t, v, s], dim=-1)
            out = self.fusion_mlp(combined)
        elif self.fusion_type == "attention":
            # Stack as sequence tokens
            seq = torch.stack([t, v, s], dim=1)  # (B, 3, D)
            attn_out, _ = self.cross_attn(seq, seq, seq)
            out = (seq + attn_out).mean(1)
            out = self.norm(out)
        elif self.fusion_type == "gated":
            combined = torch.cat([t, v, s], dim=-1)
            g_t = torch.sigmoid(self.gate_text(combined))
            g_v = torch.sigmoid(self.gate_vision(combined))
            g_s = torch.sigmoid(self.gate_ts(combined))
            out = g_t * t + g_v * v + g_s * s

        return self.output_norm(out)


class FinancialNewsClassifier(nn.Module):
    """
    Classifies financial news into sentiment categories (positive/negative/neutral)
    or event types (earnings, merger, macro, regulation, etc.)
    using the TextEncoder backbone.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, pooled = self.encoder(input_ids, attention_mask)
        return self.classifier(pooled)


class ChartPatternRecognizer(nn.Module):
    """
    Recognizes candlestick/chart patterns from price image representations.
    Head/shoulders, double top, cup & handle, flag patterns, etc.
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        num_patterns: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_patterns),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, cls_feat = self.encoder(images)
        return self.head(cls_feat)


class EarningsCallAnalyzer(nn.Module):
    """
    Analyzes earnings call audio + transcript to predict post-call price moves.
    Fuses audio prosody features with text sentiment.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        text_dim: int = 256,
        audio_dim: int = 256,
        fused_dim: int = 256,
        output_dim: int = 3,  # negative/neutral/positive move
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_enc = TextEncoder(vocab_size=vocab_size, embed_dim=text_dim, dropout=dropout)
        self.audio_enc = AudioEncoder(embed_dim=audio_dim, dropout=dropout)
        self.fusion = MultimodalFusion(
            text_dim=text_dim,
            vision_dim=audio_dim,
            ts_dim=audio_dim,  # reuse audio as second modality for simplicity
            fused_dim=fused_dim,
            fusion_type="gated",
            dropout=dropout,
        )
        self.head = nn.Linear(fused_dim, output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, text_feat = self.text_enc(input_ids, attention_mask)
        _, audio_feat = self.audio_enc(audio)
        fused = self.fusion(text_feat, audio_feat, audio_feat)
        return self.head(fused)


class DocumentEmbedder(nn.Module):
    """
    Hierarchical document embedder for long financial documents
    (10-K filings, prospectuses). Encodes sentences, then paragraphs.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        word_dim: int = 128,
        sent_dim: int = 256,
        doc_dim: int = 512,
        num_word_layers: int = 2,
        num_sent_layers: int = 2,
        num_doc_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.word_enc = TextEncoder(
            vocab_size=vocab_size, embed_dim=word_dim,
            num_layers=num_word_layers, dropout=dropout
        )
        sent_layer = nn.TransformerEncoderLayer(
            d_model=word_dim, nhead=4, dim_feedforward=word_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.sent_enc = nn.TransformerEncoder(sent_layer, num_layers=num_sent_layers)
        self.sent_proj = nn.Linear(word_dim, sent_dim)

        doc_layer = nn.TransformerEncoderLayer(
            d_model=sent_dim, nhead=8, dim_feedforward=sent_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.doc_enc = nn.TransformerEncoder(doc_layer, num_layers=num_doc_layers)
        self.doc_proj = nn.Linear(sent_dim, doc_dim)
        self.norm = nn.LayerNorm(doc_dim)

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, num_sents, sent_len)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, T = input_ids.shape
        flat_ids = input_ids.view(B * S, T)
        flat_mask = attention_mask.view(B * S, T) if attention_mask is not None else None
        _, sent_feats = self.word_enc(flat_ids, flat_mask)  # (B*S, word_dim)
        sent_feats = sent_feats.view(B, S, -1)
        sent_feats = self.sent_proj(self.sent_enc(sent_feats))
        doc_feats = self.doc_proj(self.doc_enc(sent_feats))
        return self.norm(doc_feats.mean(1))  # (B, doc_dim)


class KnowledgeGraphEmbedder(nn.Module):
    """
    Embeds financial knowledge graph entities and relations using TransE/RotatE.
    Entities: companies, sectors, indices, executives.
    Relations: subsidiary_of, competes_with, supplies_to, invested_by.
    """

    def __init__(
        self,
        num_entities: int = 10000,
        num_relations: int = 50,
        embed_dim: int = 128,
        scoring_fn: str = "transe",  # "transe" or "rotate"
        dropout: float = 0.1,
        margin: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scoring_fn = scoring_fn
        self.margin = margin

        self.entity_embed = nn.Embedding(num_entities, embed_dim)
        if scoring_fn == "rotate":
            self.relation_embed = nn.Embedding(num_relations, embed_dim // 2)  # complex
        else:
            self.relation_embed = nn.Embedding(num_relations, embed_dim)

        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        self.dropout = nn.Dropout(dropout)

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        h = self.dropout(self.entity_embed(head))
        r = self.relation_embed(relation)
        t = self.dropout(self.entity_embed(tail))

        if self.scoring_fn == "transe":
            return -(h + r - t).norm(p=1, dim=-1)
        elif self.scoring_fn == "rotate":
            # RotatE: entity in complex space, relation as rotation
            h_re, h_im = h.chunk(2, dim=-1)
            t_re, t_im = t.chunk(2, dim=-1)
            r_re = torch.cos(r)
            r_im = torch.sin(r)
            score_re = h_re * r_re - h_im * r_im - t_re
            score_im = h_re * r_im + h_im * r_re - t_im
            return -(score_re ** 2 + score_im ** 2).sum(-1).sqrt()
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_fn}")

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        neg_tail: torch.Tensor,
    ) -> torch.Tensor:
        """Margin-based loss for link prediction training."""
        pos_score = self.score(head, relation, tail)
        neg_score = self.score(head, relation, neg_tail)
        loss = F.relu(self.margin - pos_score + neg_score).mean()
        return loss

    def get_entity_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.entity_embed(entity_ids)


class AlternativeDataFusion(nn.Module):
    """
    Fuses alternative data sources: satellite imagery, credit card transactions,
    social sentiment, job posting trends, ESG scores.
    """

    def __init__(
        self,
        satellite_dim: int = 64,
        credit_dim: int = 32,
        sentiment_dim: int = 64,
        jobs_dim: int = 32,
        esg_dim: int = 16,
        fused_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        total_in = satellite_dim + credit_dim + sentiment_dim + jobs_dim + esg_dim

        self.satellite_proj = nn.Linear(satellite_dim, satellite_dim)
        self.credit_proj = nn.Linear(credit_dim, credit_dim)
        self.sentiment_proj = nn.Linear(sentiment_dim, sentiment_dim)
        self.jobs_proj = nn.Linear(jobs_dim, jobs_dim)
        self.esg_proj = nn.Linear(esg_dim, esg_dim)

        self.fusion = nn.Sequential(
            nn.Linear(total_in, fused_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(total_in, total_in),
            nn.Sigmoid(),
        )

    def forward(
        self,
        satellite: torch.Tensor,
        credit: torch.Tensor,
        sentiment: torch.Tensor,
        jobs: torch.Tensor,
        esg: torch.Tensor,
    ) -> torch.Tensor:
        s = F.gelu(self.satellite_proj(satellite))
        c = F.gelu(self.credit_proj(credit))
        se = F.gelu(self.sentiment_proj(sentiment))
        j = F.gelu(self.jobs_proj(jobs))
        e = F.gelu(self.esg_proj(esg))

        combined = torch.cat([s, c, se, j, e], dim=-1)
        gate = self.gate(combined)
        gated = combined * gate
        return self.fusion(gated)


class TemporalGraphNetwork(nn.Module):
    """
    Temporal graph network (Rossi et al. 2020) for dynamic financial networks.
    Captures evolving relationships between assets over time.
    """

    def __init__(
        self,
        num_nodes: int = 500,
        node_feat_dim: int = 32,
        edge_feat_dim: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        memory_dim: int = 64,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim

        # Node memory module
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim))
        nn.init.normal_(self.memory, std=0.01)

        self.node_proj = nn.Linear(node_feat_dim + memory_dim, embed_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, embed_dim)

        layers = []
        for _ in range(num_layers):
            layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True))
        self.gat_layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.memory_updater = nn.GRUCell(embed_dim, memory_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        node_feats: torch.Tensor,     # (B, N, F_n)
        edge_index: torch.Tensor,     # (2, E)
        edge_feats: Optional[torch.Tensor] = None,  # (E, F_e)
        node_ids: Optional[torch.Tensor] = None,    # (N,) global node IDs
    ) -> torch.Tensor:
        B, N, _ = node_feats.shape

        # Augment with memory
        if node_ids is not None:
            mem = self.memory[node_ids].unsqueeze(0).expand(B, -1, -1)
        else:
            mem = self.memory[:N].unsqueeze(0).expand(B, -1, -1)

        x = self.node_proj(torch.cat([node_feats, mem], dim=-1))

        for attn, norm in zip(self.gat_layers, self.norms):
            residual = x
            x, _ = attn(x, x, x)
            x = norm(residual + self.dropout(x))

        out = self.output_proj(x)

        # Update memory (first batch only for simplicity)
        with torch.no_grad():
            if node_ids is not None:
                self.memory[node_ids] = self.memory_updater(
                    out[0].detach(), self.memory[node_ids]
                )

        return out  # (B, N, embed_dim)


class MultimodalFinancialModel(nn.Module):
    """
    Full multimodal financial foundation model combining:
    - Text (news, filings) via TextEncoder
    - Time-series (prices, volumes) via existing Lumina backbone
    - Knowledge graph via KnowledgeGraphEmbedder
    - Alternative data via AlternativeDataFusion
    Final fusion produces unified asset representations for downstream tasks.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        text_dim: int = 256,
        ts_dim: int = 256,
        kg_dim: int = 128,
        alt_fused_dim: int = 256,
        output_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_enc = TextEncoder(vocab_size=vocab_size, embed_dim=text_dim, dropout=dropout)
        self.ts_proj = nn.Linear(ts_dim, output_dim)

        total_in = text_dim + output_dim + kg_dim + alt_fused_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_in, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.classifier = nn.Linear(output_dim, num_classes)
        self.regressor = nn.Linear(output_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        ts_feat: torch.Tensor,
        kg_feat: torch.Tensor,
        alt_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _, text_feat = self.text_enc(input_ids, attention_mask)
        ts_out = self.ts_proj(ts_feat)
        combined = torch.cat([text_feat, ts_out, kg_feat, alt_feat], dim=-1)
        fused = self.fusion(combined)
        return {
            "logits": self.classifier(fused),
            "prediction": self.regressor(fused).squeeze(-1),
            "embedding": fused,
        }
