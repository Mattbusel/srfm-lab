"""
lumina/tokenizer.py

Multi-modal financial tokenizer for Lumina Foundation Model.

Covers:
  - PriceSeriesTokenizer  : patch tokenization of OHLCV, learnable patch embeddings,
                            normalization, multi-asset support
  - OrderBookTokenizer    : LOB snapshots, delta encoding, depth aggregation
  - OnChainTokenizer      : whale flow, DEX volume, LP depth signals
  - NewsTokenizer         : financial text with special tokens
  - MultiModalTokenizer   : unified sequence with modality-type embeddings
  - TokenizerRegistry     : factory / serialization helpers
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ---------------------------------------------------------------------------
# Modality IDs
# ---------------------------------------------------------------------------
MODALITY_PRICE      = 0
MODALITY_ORDERBOOK  = 1
MODALITY_ONCHAIN    = 2
MODALITY_NEWS       = 3

# ---------------------------------------------------------------------------
# Special financial tokens
# ---------------------------------------------------------------------------
SPECIAL_FINANCIAL_TOKENS: List[str] = [
    "$TICKER", "#EARNINGS", "#FDA", "#MERGER", "#IPO",
    "#BANKRUPTCY", "#DIVIDEND", "#BUYBACK", "#GUIDANCE",
    "[PRICE]", "[ORDERBOOK]", "[ONCHAIN]", "[NEWS]",
    "[SEP_MOD]", "[CLS_LUMINA]", "[MASK_PRICE]", "[MASK_OB]",
    "[PAD_LUMINA]", "[SOT]", "[EOT]",
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def log_return(prices: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute log returns from a price tensor of shape (..., T)."""
    shifted = prices[..., :-1].clamp(min=eps)
    return torch.log(prices[..., 1:].clamp(min=eps) / shifted)


def z_score_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score normalize along `dim`, return (normalized, mean, std)."""
    mu  = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp(min=eps)
    return (x - mu) / std, mu, std


def min_max_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max normalize along `dim`, return (normalized, min_val, range)."""
    mn  = x.amin(dim=dim, keepdim=True)
    mx  = x.amax(dim=dim, keepdim=True)
    rng = (mx - mn).clamp(min=eps)
    return (x - mn) / rng, mn, rng


def rolling_zscore(x: torch.Tensor, window: int, eps: float = 1e-6) -> torch.Tensor:
    """Apply rolling Z-score normalization along last dimension."""
    B, T = x.shape[0], x.shape[-1]
    out = torch.zeros_like(x)
    for t in range(T):
        start = max(0, t - window + 1)
        window_slice = x[..., start:t+1]
        mu  = window_slice.mean(dim=-1, keepdim=True)
        std = window_slice.std(dim=-1, keepdim=True).clamp(min=eps)
        out[..., t] = ((x[..., t:t+1] - mu) / std).squeeze(-1)
    return out


def rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Rank normalize each feature independently (cross-sectional)."""
    B, T, F = x.shape
    ranks = x.argsort(dim=1).argsort(dim=1).float()
    return ranks / (T - 1 + 1e-8) * 2 - 1  # scale to [-1, 1]


def compute_atr(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, period: int = 14) -> torch.Tensor:
    """Average True Range indicator."""
    prev_close = F.pad(close[:, :-1], (1, 0), value=float('nan'))
    prev_close[:, 0] = close[:, 0]
    tr = torch.stack([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], dim=-1).amax(dim=-1)
    # Simple rolling mean for ATR
    atr = tr.unfold(-1, period, 1).mean(-1)
    atr = F.pad(atr, (period - 1, 0), value=float('nan'))
    return atr


def compute_rsi(close: torch.Tensor, period: int = 14) -> torch.Tensor:
    """Relative Strength Index."""
    delta = close[:, 1:] - close[:, :-1]
    gain  = delta.clamp(min=0)
    loss  = (-delta).clamp(min=0)
    avg_gain = gain.unfold(-1, period, 1).mean(-1)
    avg_loss = loss.unfold(-1, period, 1).mean(-1)
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    rsi = F.pad(rsi, (period, 0), value=50.0)  # neutral fill
    return rsi


def compute_bollinger(close: torch.Tensor, period: int = 20, num_std: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bollinger Bands: (upper, middle, lower)."""
    mid = close.unfold(-1, period, 1).mean(-1)
    std = close.unfold(-1, period, 1).std(-1)
    mid = F.pad(mid, (period - 1, 0), value=float('nan'))
    std = F.pad(std, (period - 1, 0), value=float('nan'))
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PriceTokenizerConfig:
    """Configuration for PriceSeriesTokenizer."""
    patch_size:        int   = 16        # bars per patch
    d_model:           int   = 512       # embedding dimension
    n_channels:        int   = 5         # OHLCV = 5 channels
    max_seq_len:       int   = 1024      # max number of patches
    norm_mode:         str   = "zscore"  # "zscore" | "minmax" | "returns" | "rank"
    use_log_returns:   bool  = True      # convert close prices to log returns
    add_technical:     bool  = True      # append technical indicators
    n_quantize_bins:   int   = 0         # 0 = no quantization
    dropout:           float = 0.1
    patch_overlap:     int   = 0         # overlapping patches
    multi_asset:       bool  = False     # multi-asset joint tokenization
    n_assets:          int   = 1
    use_cls_token:     bool  = True
    use_sep_token:     bool  = True
    patch_merge_mode:  str   = "linear"  # "linear" | "conv" | "mean"
    add_volume_feat:   bool  = True
    add_return_feat:   bool  = True
    learnable_norm:    bool  = False     # learnable affine norm parameters
    positional_type:   str   = "learned" # "learned" | "sinusoidal" | "rope"


@dataclass
class OrderBookTokenizerConfig:
    n_levels:    int   = 10
    d_model:     int   = 512
    patch_size:  int   = 8
    max_seq_len: int   = 512
    use_delta:   bool  = True
    dropout:     float = 0.1


@dataclass
class OnChainTokenizerConfig:
    n_features:  int   = 16
    d_model:     int   = 512
    patch_size:  int   = 8
    max_seq_len: int   = 256
    dropout:     float = 0.1


@dataclass
class NewsTokenizerConfig:
    vocab_size:    int   = 32000
    d_model:       int   = 512
    max_length:    int   = 512
    pretrained:    Optional[str] = None
    dropout:       float = 0.1


@dataclass
class MultiModalTokenizerConfig:
    price_cfg:    PriceTokenizerConfig    = field(default_factory=PriceTokenizerConfig)
    ob_cfg:       OrderBookTokenizerConfig = field(default_factory=OrderBookTokenizerConfig)
    onchain_cfg:  OnChainTokenizerConfig   = field(default_factory=OnChainTokenizerConfig)
    news_cfg:     NewsTokenizerConfig      = field(default_factory=NewsTokenizerConfig)
    d_model:      int   = 512
    max_total_len: int  = 2048
    dropout:      float = 0.1


# ---------------------------------------------------------------------------
# Patch Embedding implementations
# ---------------------------------------------------------------------------

class LinearPatchEmbedding(nn.Module):
    """Project a raw patch (patch_size * n_channels) → d_model using a linear layer."""

    def __init__(self, patch_size: int, n_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size  = patch_size
        self.n_channels  = n_channels
        self.d_model     = d_model
        in_dim           = patch_size * n_channels
        self.proj        = nn.Linear(in_dim, d_model, bias=True)
        self.norm        = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_patches, patch_size, n_channels) → (B, n_patches, d_model)."""
        B, N, P, C = x.shape
        x = x.reshape(B, N, P * C)
        x = self.proj(x)
        x = self.norm(x)
        return self.dropout(x)


class ConvPatchEmbedding(nn.Module):
    """1-D convolutional patch embedding operating along the time axis."""

    def __init__(self, patch_size: int, n_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.d_model    = d_model
        # Conv1d: in_channels=n_channels, out_channels=d_model, kernel=patch_size, stride=patch_size
        self.conv  = nn.Conv1d(n_channels, d_model, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, n_patches, d_model)."""
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv(x)                   # (B, d_model, n_patches)
        x = rearrange(x, 'b d n -> b n d')
        x = self.norm(x)
        return self.drop(x)


class MeanPatchEmbedding(nn.Module):
    """Simple mean aggregation over patch followed by linear projection."""

    def __init__(self, patch_size: int, n_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.proj  = nn.Linear(n_channels, d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_patches, patch_size, C) → (B, n_patches, d_model)."""
        x = x.mean(dim=2)   # (B, n_patches, C)
        x = self.proj(x)
        x = self.norm(x)
        return self.drop(x)


# ---------------------------------------------------------------------------
# Technical indicator feature extractor
# ---------------------------------------------------------------------------

class TechnicalFeatureExtractor(nn.Module):
    """Compute a fixed set of technical indicators and project to an embedding."""

    def __init__(self, d_out: int, rsi_period: int = 14, bb_period: int = 20,
                 atr_period: int = 14, dropout: float = 0.1):
        super().__init__()
        self.rsi_period = rsi_period
        self.bb_period  = bb_period
        self.atr_period = atr_period
        # Features: RSI, BB_position, ATR_norm, momentum_5, momentum_10, vol_ratio
        n_tech = 6
        self.proj  = nn.Linear(n_tech, d_out)
        self.norm  = nn.LayerNorm(d_out)
        self.drop  = nn.Dropout(dropout)

    def _safe_extract(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """ohlcv: (B, T, 5) with columns [O,H,L,C,V] → (B, T, n_tech)."""
        B, T, _ = ohlcv.shape
        o, h, l, c, v = ohlcv.unbind(dim=-1)  # each (B, T)

        # RSI
        rsi = compute_rsi(c, self.rsi_period) / 100.0  # scale [0,1]

        # Bollinger band position
        upper, mid, lower = compute_bollinger(c, self.bb_period)
        bb_pos = (c - lower) / (upper - lower + 1e-8)  # [0,1]

        # ATR normalized by close
        atr = compute_atr(h, l, c, self.atr_period)
        atr_norm = atr / (c + 1e-8)

        # Momentum
        mom5  = c / (c.roll(5, dims=1) + 1e-8) - 1
        mom10 = c / (c.roll(10, dims=1) + 1e-8) - 1
        mom5[:, :5]   = 0
        mom10[:, :10] = 0

        # Volume ratio vs rolling mean
        vol_mean = v.unfold(-1, 10, 1).mean(-1)
        vol_mean = F.pad(vol_mean, (9, 0), value=float(v.mean()))
        vol_ratio = v / (vol_mean + 1e-8)

        feats = torch.stack([rsi, bb_pos, atr_norm, mom5, mom10, vol_ratio], dim=-1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=2.0, neginf=-2.0)
        return feats

    def forward(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """ohlcv: (B, T, 5) → (B, T, d_out)."""
        feats = self._safe_extract(ohlcv)   # (B, T, n_tech)
        out   = self.proj(feats)
        out   = self.norm(out)
        return self.drop(out)


# ---------------------------------------------------------------------------
# Quantized patch tokenizer (VQ-like discrete bins)
# ---------------------------------------------------------------------------

class QuantizedPatchEmbedding(nn.Module):
    """Soft quantization: map each patch scalar to a learned codebook embedding."""

    def __init__(self, n_bins: int, n_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_bins     = n_bins
        self.n_channels = n_channels
        self.d_model    = d_model
        # Per-channel codebook
        self.codebook = nn.Embedding(n_bins * n_channels, d_model)
        self.proj     = nn.Linear(n_channels * d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)
        # Bin boundaries (uniform initially, learnable)
        self.bin_edges = nn.Parameter(torch.linspace(-3, 3, n_bins + 1))

    def _quantize(self, x: torch.Tensor, channel_idx: int) -> torch.Tensor:
        """x: (B, N) scalar values → (B, N) bin indices."""
        edges = self.bin_edges.sort().values  # ensure sorted
        indices = torch.bucketize(x, edges[1:-1])  # [0, n_bins-1]
        indices = indices.clamp(0, self.n_bins - 1)
        return indices + channel_idx * self.n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, patch_size, n_channels) → (B, N, d_model).
        Aggregates patch by mean before quantizing.
        """
        patch_mean = x.mean(dim=2)  # (B, N, C)
        embeddings = []
        for c in range(self.n_channels):
            ids   = self._quantize(patch_mean[..., c], c)   # (B, N)
            emb   = self.codebook(ids)                        # (B, N, d_model)
            embeddings.append(emb)
        out = torch.cat(embeddings, dim=-1)   # (B, N, C*d_model)
        out = self.proj(out)
        out = self.norm(out)
        return self.drop(out)


# ---------------------------------------------------------------------------
# Main PriceSeriesTokenizer
# ---------------------------------------------------------------------------

class PriceSeriesTokenizer(nn.Module):
    """
    Tokenizes raw OHLCV time series into a sequence of patch embeddings.

    Pipeline:
      1. Normalize the raw OHLCV using z-score / min-max / log-returns
      2. Optionally compute technical indicators
      3. Split into non-overlapping (or overlapping) patches
      4. Project each patch → d_model via linear / conv / mean embedding
      5. Prepend CLS token, append SEP token (optional)
      6. Add positional embeddings
    """

    def __init__(self, cfg: PriceTokenizerConfig):
        super().__init__()
        self.cfg = cfg
        P, C, D = cfg.patch_size, cfg.n_channels, cfg.d_model

        # ---- patch embedding ----
        if cfg.patch_merge_mode == "linear":
            self.patch_embed = LinearPatchEmbedding(P, C, D, cfg.dropout)
        elif cfg.patch_merge_mode == "conv":
            self.patch_embed = ConvPatchEmbedding(P, C, D, cfg.dropout)
        else:
            self.patch_embed = MeanPatchEmbedding(P, C, D, cfg.dropout)

        # ---- optional quantized embedding ----
        self.quant_embed: Optional[QuantizedPatchEmbedding] = None
        if cfg.n_quantize_bins > 0:
            self.quant_embed = QuantizedPatchEmbedding(cfg.n_quantize_bins, C, D, cfg.dropout)
            self.quant_mix   = nn.Parameter(torch.tensor(0.5))

        # ---- technical feature branch ----
        self.tech_extractor: Optional[TechnicalFeatureExtractor] = None
        if cfg.add_technical and C == 5:
            self.tech_extractor = TechnicalFeatureExtractor(D, dropout=cfg.dropout)
            self.tech_proj = nn.Linear(D + D, D)

        # ---- special tokens ----
        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        if cfg.use_sep_token:
            self.sep_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        # ---- positional encoding ----
        if cfg.positional_type == "learned":
            self.pos_emb = nn.Embedding(cfg.max_seq_len + 2, D)
        else:
            self.register_buffer(
                "pos_emb_buf",
                self._build_sinusoidal(cfg.max_seq_len + 2, D),
            )

        # ---- learnable normalization ----
        if cfg.learnable_norm:
            self.norm_scale = nn.Parameter(torch.ones(C))
            self.norm_bias  = nn.Parameter(torch.zeros(C))

        # ---- multi-asset ----
        if cfg.multi_asset:
            self.asset_embeddings = nn.Embedding(cfg.n_assets, D)

        self.output_norm = nn.LayerNorm(D)
        self.dropout     = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe.unsqueeze(0)  # (1, max_len, D)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → normalized (B, T, C)."""
        cfg = self.cfg
        if cfg.use_log_returns and x.shape[1] > 1:
            # Apply log returns to close (channel 3)
            close = x[:, :, 3]  # (B, T)
            lr    = torch.zeros_like(close)
            lr[:, 1:] = log_return(close.unsqueeze(1).squeeze(1).unsqueeze(-1).squeeze(-1), eps=1e-8).squeeze(-1) \
                        if False else (torch.log(close[:, 1:].clamp(1e-8) / close[:, :-1].clamp(1e-8)))
            x = x.clone()
            x[:, :, 3] = lr

        if cfg.norm_mode == "zscore":
            x, _, _ = z_score_normalize(x, dim=1)
        elif cfg.norm_mode == "minmax":
            x, _, _ = min_max_normalize(x, dim=1)
        elif cfg.norm_mode == "rank":
            x = rank_normalize(x)

        if cfg.learnable_norm:
            x = x * self.norm_scale + self.norm_bias

        return x

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, n_patches, patch_size, C)."""
        P   = self.cfg.patch_size
        ovl = self.cfg.patch_overlap
        B, T, C = x.shape
        if ovl == 0:
            n_patches = T // P
            x = x[:, :n_patches * P, :]
            return x.reshape(B, n_patches, P, C)
        else:
            stride = P - ovl
            indices = []
            t = 0
            while t + P <= T:
                indices.append(t)
                t += stride
            patches = torch.stack([x[:, i:i+P, :] for i in indices], dim=1)
            return patches

    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D) with positional encoding added."""
        N = x.shape[1]
        if self.cfg.positional_type == "learned":
            pos = torch.arange(N, device=x.device)
            x   = x + self.pos_emb(pos).unsqueeze(0)
        else:
            x = x + self.pos_emb_buf[:, :N, :]
        return x

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ohlcv: torch.Tensor,
        asset_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ohlcv:        (B, T, 5) raw OHLCV tensor
            asset_ids:    (B,) integer asset identifiers (multi-asset mode)
            padding_mask: (B, T) boolean mask (True = pad)

        Returns dict with:
            embeddings:   (B, N_tokens, D)
            patch_mask:   (B, N_tokens) boolean
            n_patches:    int
        """
        cfg = self.cfg
        B, T, C = ohlcv.shape

        # 1. Normalize
        x_norm = self._normalize(ohlcv)   # (B, T, C)

        # 2. Technical indicators (before patchifying)
        tech_seq: Optional[torch.Tensor] = None
        if self.tech_extractor is not None:
            tech_seq = self.tech_extractor(ohlcv)  # (B, T, D)

        # 3. Patchify
        if cfg.patch_merge_mode == "conv":
            patch_emb = self.patch_embed(x_norm)   # (B, n_patches, D)
        else:
            patches   = self._patchify(x_norm)     # (B, n_patches, P, C)
            patch_emb = self.patch_embed(patches)  # (B, n_patches, D)

        n_patches = patch_emb.shape[1]

        # 3b. Quantized branch
        if self.quant_embed is not None:
            if cfg.patch_merge_mode == "conv":
                patches_for_q = self._patchify(x_norm)
            else:
                patches_for_q = patches  # already computed
            q_emb     = self.quant_embed(patches_for_q)
            alpha     = torch.sigmoid(self.quant_mix)
            patch_emb = alpha * patch_emb + (1 - alpha) * q_emb

        # 3c. Fuse technical features
        if tech_seq is not None:
            P = cfg.patch_size
            # Aggregate tech features per patch
            tech_patches = tech_seq[:, :n_patches * P, :].reshape(B, n_patches, P, -1).mean(2)
            combined  = torch.cat([patch_emb, tech_patches], dim=-1)
            patch_emb = self.tech_proj(combined)

        # 4. Multi-asset embedding
        if cfg.multi_asset and asset_ids is not None:
            asset_emb = self.asset_embeddings(asset_ids)  # (B, D)
            patch_emb = patch_emb + asset_emb.unsqueeze(1)

        # 5. CLS / SEP tokens
        tokens = patch_emb
        if cfg.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        if cfg.use_sep_token:
            sep = self.sep_token.expand(B, -1, -1)
            tokens = torch.cat([tokens, sep], dim=1)

        # 6. Positional encoding
        tokens = self._add_positional(tokens)

        # 7. Build patch mask
        n_tokens = tokens.shape[1]
        patch_mask = torch.zeros(B, n_tokens, dtype=torch.bool, device=ohlcv.device)
        if padding_mask is not None:
            # Aggregate padding mask per patch
            P = cfg.patch_size
            pm = padding_mask[:, :n_patches * P].reshape(B, n_patches, P)
            pm = pm.all(dim=-1)   # patch is padding if all timesteps are padding
            offset = 1 if cfg.use_cls_token else 0
            patch_mask[:, offset:offset + n_patches] = pm

        tokens = self.output_norm(tokens)
        tokens = self.dropout(tokens)

        return {
            "embeddings": tokens,
            "patch_mask": patch_mask,
            "n_patches":  n_patches,
        }

    def encode_series(self, ohlcv: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Convenience wrapper accepting numpy arrays."""
        t = torch.from_numpy(ohlcv).float().unsqueeze(0).to(device)
        out = self.forward(t)
        return out["embeddings"]

    def get_n_patches(self, seq_len: int) -> int:
        """Calculate number of patches for a given sequence length."""
        P = self.cfg.patch_size
        ovl = self.cfg.patch_overlap
        if ovl == 0:
            return seq_len // P
        stride = P - ovl
        count  = 0
        t      = 0
        while t + P <= seq_len:
            count += 1
            t     += stride
        return count


# ---------------------------------------------------------------------------
# MultiAssetPriceTokenizer
# ---------------------------------------------------------------------------

class MultiAssetPriceTokenizer(nn.Module):
    """
    Tokenize multiple assets simultaneously and produce a joint embedding.
    Supports cross-asset positional encoding and per-asset type embeddings.
    """

    def __init__(self, cfg: PriceTokenizerConfig, n_assets: int, asset_names: Optional[List[str]] = None):
        super().__init__()
        cfg.multi_asset = True
        cfg.n_assets    = n_assets
        self.cfg        = cfg
        self.n_assets   = n_assets
        self.asset_names = asset_names or [f"asset_{i}" for i in range(n_assets)]

        self.tokenizers = nn.ModuleList([
            PriceSeriesTokenizer(cfg) for _ in range(n_assets)
        ])
        # Cross-asset position embedding
        self.cross_asset_pos = nn.Embedding(n_assets, cfg.d_model)
        self.output_norm     = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        ohlcv_list: List[torch.Tensor],
        padding_masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            ohlcv_list:    List of (B, T, 5) tensors, one per asset
            padding_masks: List of (B, T) boolean masks

        Returns:
            embeddings: (B, n_assets * n_tokens_per_asset, D)
            asset_ids:  (n_assets * n_tokens_per_asset,) asset index per token
            masks:      (B, n_assets * n_tokens_per_asset)
        """
        assert len(ohlcv_list) == self.n_assets
        all_embs  = []
        all_masks = []
        asset_ids_list = []

        for i, (tok, ohlcv) in enumerate(zip(self.tokenizers, ohlcv_list)):
            pmask = padding_masks[i] if padding_masks else None
            asset_ids = torch.full((ohlcv.shape[0],), i, dtype=torch.long, device=ohlcv.device)
            out = tok(ohlcv, asset_ids=asset_ids, padding_mask=pmask)
            emb = out["embeddings"]   # (B, N_i, D)
            # Add cross-asset position
            ca_pos = self.cross_asset_pos(torch.tensor(i, device=ohlcv.device))
            emb    = emb + ca_pos.unsqueeze(0).unsqueeze(0)
            all_embs.append(emb)
            all_masks.append(out["patch_mask"])
            n_i = emb.shape[1]
            asset_ids_list.append(torch.full((n_i,), i, dtype=torch.long, device=ohlcv.device))

        embeddings = torch.cat(all_embs,  dim=1)
        masks      = torch.cat(all_masks, dim=1)
        asset_ids_out = torch.cat(asset_ids_list, dim=0)

        return {
            "embeddings": self.output_norm(embeddings),
            "masks":      masks,
            "asset_ids":  asset_ids_out,
        }


# ---------------------------------------------------------------------------
# OrderBookTokenizer
# ---------------------------------------------------------------------------

class OrderBookFeatureExtractor(nn.Module):
    """Extract features from a raw LOB snapshot."""

    def __init__(self, n_levels: int, d_out: int):
        super().__init__()
        # Raw: bid_prices (n_levels), bid_sizes (n_levels), ask_prices (n_levels), ask_sizes (n_levels)
        in_dim = 4 * n_levels
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_out * 2),
            nn.GELU(),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        """lob: (B, T, 4, n_levels) → (B, T, d_out)."""
        B, T, _, L = lob.shape
        x = lob.reshape(B, T, -1)
        return self.net(x)


class OrderBookDeltaEncoder(nn.Module):
    """Encode order book changes (deltas) between consecutive snapshots."""

    def __init__(self, n_levels: int, d_out: int):
        super().__init__()
        in_dim = 4 * n_levels
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_out),
            nn.Tanh(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        """lob: (B, T, 4*n_levels) → (B, T-1, d_out) delta features."""
        delta = lob[:, 1:, :] - lob[:, :-1, :]
        return self.net(delta)


class OrderBookTokenizer(nn.Module):
    """
    Tokenizes limit order book (LOB) snapshots.
    Supports raw snapshot embedding and delta (change) encoding.
    """

    def __init__(self, cfg: OrderBookTokenizerConfig):
        super().__init__()
        self.cfg = cfg
        L, D, P = cfg.n_levels, cfg.d_model, cfg.patch_size

        self.feat_extractor = OrderBookFeatureExtractor(L, D)
        if cfg.use_delta:
            self.delta_encoder = OrderBookDeltaEncoder(L, D)
            self.delta_proj    = nn.Linear(D + D, D)

        # Patch over time
        self.patch_proj = nn.Linear(P * D, D)
        self.norm       = nn.LayerNorm(D)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.pos_emb    = nn.Embedding(cfg.max_seq_len + 1, D)
        self.dropout    = nn.Dropout(cfg.dropout)

        # Mid-price spread features
        self.spread_proj = nn.Linear(2, D // 4)

    def _compute_spread_features(self, lob: torch.Tensor) -> torch.Tensor:
        """Compute bid-ask spread and mid price. lob: (B,T,4,L)."""
        best_bid = lob[:, :, 0, 0]  # highest bid price
        best_ask = lob[:, :, 2, 0]  # lowest ask price
        mid      = (best_bid + best_ask) / 2
        spread   = best_ask - best_bid
        return torch.stack([mid, spread], dim=-1)  # (B, T, 2)

    def forward(
        self,
        lob: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            lob: (B, T, 4, n_levels) — [bid_prices, bid_sizes, ask_prices, ask_sizes]
            padding_mask: (B, T)
        Returns:
            embeddings: (B, n_patches+1, D)
            patch_mask: (B, n_patches+1)
        """
        cfg     = self.cfg
        B, T, _, L = lob.shape
        P       = cfg.patch_size

        # Feature extraction
        lob_flat = lob.reshape(B, T, 4 * L)
        feats    = self.feat_extractor(lob)   # (B, T, D)

        if cfg.use_delta:
            deltas    = self.delta_encoder(lob_flat)           # (B, T-1, D)
            deltas    = F.pad(deltas, (0, 0, 1, 0), value=0)  # (B, T, D)
            combined  = torch.cat([feats, deltas], dim=-1)    # (B, T, 2D)
            feats     = self.delta_proj(combined)              # (B, T, D)

        # Patchify over time
        n_patches = T // P
        feats_t   = feats[:, :n_patches * P, :].reshape(B, n_patches, P * cfg.d_model)
        patch_emb = self.patch_proj(feats_t)  # (B, n_patches, D)

        # CLS + positional
        cls       = self.cls_token.expand(B, -1, -1)
        tokens    = torch.cat([cls, patch_emb], dim=1)
        N         = tokens.shape[1]
        pos       = torch.arange(N, device=lob.device)
        tokens    = tokens + self.pos_emb(pos).unsqueeze(0)

        # Mask
        patch_mask = torch.zeros(B, N, dtype=torch.bool, device=lob.device)
        if padding_mask is not None:
            pm = padding_mask[:, :n_patches * P].reshape(B, n_patches, P).all(-1)
            patch_mask[:, 1:] = pm

        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)
        return {"embeddings": tokens, "patch_mask": patch_mask}


# ---------------------------------------------------------------------------
# OnChainTokenizer
# ---------------------------------------------------------------------------

class OnChainFeatureNormalizer(nn.Module):
    """Normalize on-chain features which can have heavy-tailed distributions."""

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        # Learnable log-transform scale
        self.log_scales = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_features) → normalized."""
        # Signed log transform: sign(x) * log(1 + |x|)
        x_log = torch.sign(x) * torch.log1p(x.abs())
        # Scale
        x_log = x_log * self.log_scales.unsqueeze(0).unsqueeze(0)
        # Z-score
        mu  = x_log.mean(dim=1, keepdim=True)
        std = x_log.std(dim=1, keepdim=True).clamp(1e-6)
        return (x_log - mu) / std


class OnChainTokenizer(nn.Module):
    """
    Tokenizes on-chain DeFi signals (whale flows, DEX volumes, LP depth, etc.)
    """

    def __init__(self, cfg: OnChainTokenizerConfig):
        super().__init__()
        self.cfg = cfg
        F_in, D, P = cfg.n_features, cfg.d_model, cfg.patch_size

        self.normalizer = OnChainFeatureNormalizer(F_in)
        self.feat_proj  = nn.Sequential(
            nn.Linear(F_in, D),
            nn.GELU(),
            nn.LayerNorm(D),
        )
        self.patch_proj  = nn.Linear(P * D, D)
        self.norm        = nn.LayerNorm(D)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.pos_emb     = nn.Embedding(cfg.max_seq_len + 1, D)
        self.dropout     = nn.Dropout(cfg.dropout)

    def forward(
        self,
        onchain: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            onchain: (B, T, n_features)
        Returns:
            embeddings: (B, n_patches+1, D)
        """
        cfg     = self.cfg
        B, T, _ = onchain.shape
        P       = cfg.patch_size

        x = self.normalizer(onchain)   # (B, T, F)
        x = self.feat_proj(x)         # (B, T, D)

        n_patches = T // P
        x_p = x[:, :n_patches * P, :].reshape(B, n_patches, P * cfg.d_model)
        patch_emb = self.patch_proj(x_p)   # (B, n_patches, D)

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patch_emb], dim=1)
        N      = tokens.shape[1]
        pos    = torch.arange(N, device=onchain.device)
        tokens = tokens + self.pos_emb(pos).unsqueeze(0)

        patch_mask = torch.zeros(B, N, dtype=torch.bool, device=onchain.device)
        if padding_mask is not None:
            pm = padding_mask[:, :n_patches * P].reshape(B, n_patches, P).all(-1)
            patch_mask[:, 1:] = pm

        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)
        return {"embeddings": tokens, "patch_mask": patch_mask}


# ---------------------------------------------------------------------------
# NewsTokenizer
# ---------------------------------------------------------------------------

FINANCIAL_VOCAB_EXTENSION = [
    "FOMC", "CPI", "NFP", "GDP", "PCE", "JOLTS", "PMI", "ISM",
    "QE", "QT", "tapering", "hawkish", "dovish", "stagflation",
    "recession", "expansion", "contraction", "inversion", "spread",
    "basis", "carry", "alpha", "beta", "gamma", "delta", "theta",
    "vega", "implied", "realized", "skew", "kurtosis", "tail",
    "drawdown", "Sharpe", "Sortino", "Calmar", "VaR", "CVaR",
    "ETF", "futures", "options", "swap", "CDS", "MBS", "CLO",
    "IPO", "SPO", "buyback", "dividend", "EBITDA", "EPS", "PE",
    "PB", "ROE", "ROIC", "FCF", "capex", "WACC", "DCF",
    "semiconductor", "semiconductor", "biotech", "fintech",
    "cryptocurrency", "defi", "NFT", "blockchain", "tokenize",
]


class FinancialSentimentHead(nn.Module):
    """Simple sentiment classification head on top of text embeddings."""

    def __init__(self, d_model: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """cls_emb: (B, D) → (B, n_classes) logits."""
        return self.net(cls_emb)


class SimpleFinancialTokenizer:
    """
    A simple BPE-inspired tokenizer for financial text.
    Falls back gracefully if transformers is not installed.
    """

    def __init__(self, vocab_size: int = 32000, max_length: int = 512):
        self.vocab_size  = vocab_size
        self.max_length  = max_length
        self.special_tokens = SPECIAL_FINANCIAL_TOKENS
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self._build_base_vocab()

    def _build_base_vocab(self):
        special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        all_special = special + self.special_tokens + FINANCIAL_VOCAB_EXTENSION
        for i, tok in enumerate(all_special):
            self.word2id[tok] = i
            self.id2word[i]   = tok

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = text.lower().split()
        ids    = []
        if add_special_tokens:
            ids.append(self.word2id.get("[CLS]", 2))
        for t in tokens[:self.max_length - 2]:
            ids.append(self.word2id.get(t, self.word2id.get("[UNK]", 1)))
        if add_special_tokens:
            ids.append(self.word2id.get("[SEP]", 3))
        # Pad
        while len(ids) < self.max_length:
            ids.append(self.word2id.get("[PAD]", 0))
        return ids[:self.max_length]

    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        all_ids  = [self.encode(t) for t in texts]
        ids_t    = torch.tensor(all_ids, dtype=torch.long)
        mask     = (ids_t != self.word2id.get("[PAD]", 0)).long()
        return {"input_ids": ids_t, "attention_mask": mask}


class NewsTokenizer(nn.Module):
    """
    Tokenizes financial news headlines / articles.
    Uses a learned token embedding + positional encoding.
    """

    def __init__(self, cfg: NewsTokenizerConfig):
        super().__init__()
        self.cfg  = cfg
        V, D, L   = cfg.vocab_size, cfg.d_model, cfg.max_length

        self.text_tokenizer = SimpleFinancialTokenizer(V, L)
        self.token_emb      = nn.Embedding(V, D, padding_idx=0)
        self.pos_emb        = nn.Embedding(L, D)
        self.seg_emb        = nn.Embedding(2, D)       # text vs special token
        self.norm           = nn.LayerNorm(D)
        self.dropout        = nn.Dropout(cfg.dropout)
        self.sentiment_head = FinancialSentimentHead(D)

        # Modality marker
        self.modality_emb = nn.Embedding(4, D)         # 4 modalities

        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (B, L) integer token ids
            attention_mask: (B, L) 1=real 0=pad
        Returns:
            embeddings: (B, L, D)
            cls_emb:    (B, D)
            sentiment:  (B, 3)
        """
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)

        tok_emb = self.token_emb(input_ids)   # (B, L, D)
        pos_emb = self.pos_emb(pos)            # (1, L, D)
        mod_emb = self.modality_emb(
            torch.full((1, 1), MODALITY_NEWS, device=input_ids.device)
        )

        x = tok_emb + pos_emb + mod_emb
        x = self.norm(x)
        x = self.dropout(x)

        cls_emb   = x[:, 0, :]               # (B, D)
        sentiment = self.sentiment_head(cls_emb)

        return {
            "embeddings": x,
            "cls_emb":    cls_emb,
            "sentiment":  sentiment,
        }

    def tokenize_texts(self, texts: List[str], device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Tokenize a list of strings and return tensors."""
        out = self.text_tokenizer.batch_encode(texts)
        return {k: v.to(device) for k, v in out.items()}


# ---------------------------------------------------------------------------
# MultiModalTokenizer
# ---------------------------------------------------------------------------

class ModalityTypeEmbedding(nn.Module):
    """Learnable embedding for each modality type."""

    def __init__(self, n_modalities: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_modalities, d_model)

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(modality_ids)


class SequenceFusion(nn.Module):
    """Fuse sequences from multiple modalities by concatenation + projection."""

    def __init__(self, d_model: int, n_modalities: int, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(d_model * n_modalities, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """sequences: list of (B, D) CLS tokens → (B, D)."""
        x = torch.cat(sequences, dim=-1)
        x = self.proj(x)
        x = self.norm(x)
        return self.dropout(x)


class MultiModalTokenizer(nn.Module):
    """
    Unified multi-modal tokenizer that combines all financial modalities into
    a single token sequence suitable for transformer input.

    Token layout:
      [CLS_GLOBAL] [PRICE tokens...] [SEP] [OB tokens...] [SEP] [ONCHAIN tokens...] [SEP] [NEWS tokens...] [SEP]
    """

    def __init__(self, cfg: MultiModalTokenizerConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        self.price_tokenizer   = PriceSeriesTokenizer(cfg.price_cfg)
        self.ob_tokenizer      = OrderBookTokenizer(cfg.ob_cfg)
        self.onchain_tokenizer = OnChainTokenizer(cfg.onchain_cfg)
        self.news_tokenizer    = NewsTokenizer(cfg.news_cfg)

        self.modality_emb      = ModalityTypeEmbedding(4, D)

        # Global CLS token
        self.global_cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        # SEP tokens per modality
        self.sep_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, D) * 0.02) for _ in range(4)
        ])

        # Global positional encoding for the combined sequence
        self.global_pos_emb = nn.Embedding(cfg.max_total_len + 10, D)

        self.output_norm = nn.LayerNorm(D)
        self.dropout     = nn.Dropout(cfg.dropout)

        # Alignment projection (in case sub-model d_model differs)
        price_d  = cfg.price_cfg.d_model
        ob_d     = cfg.ob_cfg.d_model
        onchain_d = cfg.onchain_cfg.d_model
        news_d   = cfg.news_cfg.d_model

        self.align_projs = nn.ModuleDict({
            "price":   nn.Linear(price_d, D)   if price_d   != D else nn.Identity(),
            "ob":      nn.Linear(ob_d, D)      if ob_d      != D else nn.Identity(),
            "onchain": nn.Linear(onchain_d, D) if onchain_d != D else nn.Identity(),
            "news":    nn.Linear(news_d, D)    if news_d    != D else nn.Identity(),
        })

    def forward(
        self,
        ohlcv:          Optional[torch.Tensor] = None,
        lob:            Optional[torch.Tensor] = None,
        onchain:        Optional[torch.Tensor] = None,
        input_ids:      Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        price_pad_mask: Optional[torch.Tensor] = None,
        ob_pad_mask:    Optional[torch.Tensor] = None,
        onchain_pad_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            embeddings:     (B, total_tokens, D)
            padding_mask:   (B, total_tokens)
            modality_map:   (total_tokens,) modality id per token
        """
        parts      = []
        masks      = []
        mod_ids    = []
        B          = None

        # Determine batch size
        for x in [ohlcv, lob, onchain, input_ids]:
            if x is not None:
                B = x.shape[0]
                break
        assert B is not None, "At least one modality must be provided"

        device = next(iter([ohlcv, lob, onchain, input_ids] if True else [None]
                           .__class__.__iter__([ohlcv, lob, onchain, input_ids])),
                      torch.zeros(1)).device if False else (
            ohlcv.device if ohlcv is not None else
            lob.device   if lob   is not None else
            onchain.device if onchain is not None else
            input_ids.device
        )

        # Global CLS
        global_cls = self.global_cls.expand(B, -1, -1)
        parts.append(global_cls)
        masks.append(torch.zeros(B, 1, dtype=torch.bool, device=device))
        mod_ids.append(torch.full((1,), -1, dtype=torch.long, device=device))

        # Price
        if ohlcv is not None:
            out     = self.price_tokenizer(ohlcv, padding_mask=price_pad_mask)
            emb     = self.align_projs["price"](out["embeddings"])
            mod_id  = torch.full((emb.shape[1],), MODALITY_PRICE, dtype=torch.long, device=device)
            emb     = emb + self.modality_emb(mod_id).unsqueeze(0)
            parts.append(emb)
            masks.append(out["patch_mask"])
            mod_ids.append(mod_id)
            # SEP
            sep = self.sep_tokens[MODALITY_PRICE].expand(B, -1, -1)
            parts.append(sep)
            masks.append(torch.zeros(B, 1, dtype=torch.bool, device=device))
            mod_ids.append(torch.full((1,), MODALITY_PRICE, dtype=torch.long, device=device))

        # Order book
        if lob is not None:
            out    = self.ob_tokenizer(lob, padding_mask=ob_pad_mask)
            emb    = self.align_projs["ob"](out["embeddings"])
            mod_id = torch.full((emb.shape[1],), MODALITY_ORDERBOOK, dtype=torch.long, device=device)
            emb    = emb + self.modality_emb(mod_id).unsqueeze(0)
            parts.append(emb)
            masks.append(out["patch_mask"])
            mod_ids.append(mod_id)
            sep = self.sep_tokens[MODALITY_ORDERBOOK].expand(B, -1, -1)
            parts.append(sep)
            masks.append(torch.zeros(B, 1, dtype=torch.bool, device=device))
            mod_ids.append(torch.full((1,), MODALITY_ORDERBOOK, dtype=torch.long, device=device))

        # On-chain
        if onchain is not None:
            out    = self.onchain_tokenizer(onchain, padding_mask=onchain_pad_mask)
            emb    = self.align_projs["onchain"](out["embeddings"])
            mod_id = torch.full((emb.shape[1],), MODALITY_ONCHAIN, dtype=torch.long, device=device)
            emb    = emb + self.modality_emb(mod_id).unsqueeze(0)
            parts.append(emb)
            masks.append(out["patch_mask"])
            mod_ids.append(mod_id)
            sep = self.sep_tokens[MODALITY_ONCHAIN].expand(B, -1, -1)
            parts.append(sep)
            masks.append(torch.zeros(B, 1, dtype=torch.bool, device=device))
            mod_ids.append(torch.full((1,), MODALITY_ONCHAIN, dtype=torch.long, device=device))

        # News
        if input_ids is not None:
            out    = self.news_tokenizer(input_ids, attention_mask)
            emb    = self.align_projs["news"](out["embeddings"])
            mod_id = torch.full((emb.shape[1],), MODALITY_NEWS, dtype=torch.long, device=device)
            emb    = emb + self.modality_emb(mod_id).unsqueeze(0)
            parts.append(emb)
            news_mask = (attention_mask == 0) if attention_mask is not None else \
                        torch.zeros(B, emb.shape[1], dtype=torch.bool, device=device)
            masks.append(news_mask)
            mod_ids.append(mod_id)

        # Concatenate
        embeddings   = torch.cat(parts, dim=1)          # (B, N_total, D)
        padding_mask = torch.cat(masks, dim=1)           # (B, N_total)
        modality_map = torch.cat(mod_ids, dim=0)         # (N_total,)

        # Truncate to max_total_len
        max_len = self.cfg.max_total_len
        if embeddings.shape[1] > max_len:
            embeddings   = embeddings[:, :max_len, :]
            padding_mask = padding_mask[:, :max_len]
            modality_map = modality_map[:max_len]

        # Global positional encoding
        N   = embeddings.shape[1]
        pos = torch.arange(N, device=device).unsqueeze(0)
        embeddings = embeddings + self.global_pos_emb(pos)

        embeddings = self.output_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return {
            "embeddings":   embeddings,
            "padding_mask": padding_mask,
            "modality_map": modality_map,
        }

    def get_modality_slices(
        self,
        ohlcv:     Optional[torch.Tensor] = None,
        lob:       Optional[torch.Tensor] = None,
        onchain:   Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tuple[int, int]]:
        """Return start/end indices for each modality in the combined sequence."""
        slices = {}
        pos    = 1  # after global CLS
        if ohlcv is not None:
            cfg_p = self.cfg.price_cfg
            n     = self.price_tokenizer.get_n_patches(ohlcv.shape[1])
            extra = (1 if cfg_p.use_cls_token else 0) + (1 if cfg_p.use_sep_token else 0)
            slices["price"] = (pos, pos + n + extra)
            pos += n + extra + 1  # +1 for SEP
        if lob is not None:
            n = lob.shape[1] // self.cfg.ob_cfg.patch_size + 1  # +1 for CLS
            slices["ob"] = (pos, pos + n)
            pos += n + 1
        if onchain is not None:
            n = onchain.shape[1] // self.cfg.onchain_cfg.patch_size + 1
            slices["onchain"] = (pos, pos + n)
            pos += n + 1
        if input_ids is not None:
            n = input_ids.shape[1]
            slices["news"] = (pos, pos + n)
        return slices


# ---------------------------------------------------------------------------
# TokenizerRegistry — factory and serialization
# ---------------------------------------------------------------------------

class TokenizerRegistry:
    """Central registry for tokenizer instances with save/load support."""

    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, tokenizer: nn.Module) -> None:
        cls._registry[name] = tokenizer

    @classmethod
    def get(cls, name: str) -> Optional[nn.Module]:
        return cls._registry.get(name)

    @classmethod
    def list_names(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def save(cls, name: str, path: Union[str, Path]) -> None:
        tok = cls._registry.get(name)
        if tok is None:
            raise KeyError(f"Tokenizer '{name}' not in registry")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": tok.state_dict(), "class": tok.__class__.__name__}, path)

    @classmethod
    def load_state(cls, name: str, path: Union[str, Path]) -> None:
        tok = cls._registry.get(name)
        if tok is None:
            raise KeyError(f"Tokenizer '{name}' not in registry")
        data = torch.load(path, map_location="cpu")
        tok.load_state_dict(data["state_dict"], strict=False)

    @classmethod
    def build_price_tokenizer(
        cls,
        d_model: int = 512,
        patch_size: int = 16,
        n_channels: int = 5,
        **kwargs,
    ) -> PriceSeriesTokenizer:
        cfg = PriceTokenizerConfig(d_model=d_model, patch_size=patch_size, n_channels=n_channels, **kwargs)
        return PriceSeriesTokenizer(cfg)

    @classmethod
    def build_multimodal_tokenizer(
        cls,
        d_model: int = 512,
        **kwargs,
    ) -> MultiModalTokenizer:
        cfg = MultiModalTokenizerConfig(d_model=d_model, **kwargs)
        return MultiModalTokenizer(cfg)


# ---------------------------------------------------------------------------
# Normalization statistics tracker (for online normalization)
# ---------------------------------------------------------------------------

class RunningStats(nn.Module):
    """
    Track running mean and variance for online normalization.
    Uses Welford's algorithm for numerical stability.
    """

    def __init__(self, n_features: int, momentum: float = 0.01):
        super().__init__()
        self.n_features = n_features
        self.momentum   = momentum
        self.register_buffer("running_mean", torch.zeros(n_features))
        self.register_buffer("running_var",  torch.ones(n_features))
        self.register_buffer("count",        torch.tensor(0, dtype=torch.long))

    def update(self, x: torch.Tensor) -> None:
        """x: (B, T, F) or (B, F)."""
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        batch_mean = x.mean(0)
        batch_var  = x.var(0)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * batch_var
        self.count       += x.shape[0]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using running statistics."""
        return (x - self.running_mean) / (self.running_var.sqrt() + 1e-6)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        return x * (self.running_var.sqrt() + 1e-6) + self.running_mean


# ---------------------------------------------------------------------------
# Patch masking (for masked pre-training)
# ---------------------------------------------------------------------------

class PatchMasker(nn.Module):
    """
    Randomly mask patches in the token sequence for masked pre-training.
    Uses a learnable MASK token.
    """

    def __init__(self, d_model: int, mask_ratio: float = 0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens:       (B, N, D)
            padding_mask: (B, N) True = padding

        Returns:
            masked_tokens: (B, N, D) with mask_ratio fraction replaced
            mask_indices:  (B, N) True = was masked
        """
        B, N, D = tokens.shape
        device  = tokens.device

        # Don't mask padding or CLS (position 0)
        eligible = torch.ones(B, N, dtype=torch.bool, device=device)
        eligible[:, 0] = False  # CLS
        if padding_mask is not None:
            eligible = eligible & ~padding_mask

        # Sample mask
        n_mask    = max(1, int(self.mask_ratio * eligible.float().sum(1).mean().item()))
        rand_vals = torch.rand(B, N, device=device)
        rand_vals[~eligible] = 1.0  # never mask these

        # Top-k smallest random values → masked
        threshold       = rand_vals.kthvalue(n_mask, dim=1).values.unsqueeze(1)
        mask_indices    = rand_vals <= threshold

        masked_tokens   = tokens.clone()
        masked_tokens[mask_indices] = self.mask_token.to(device)

        return masked_tokens, mask_indices


# ---------------------------------------------------------------------------
# Augmented tokenizer wrappers (for training-time augmentation)
# ---------------------------------------------------------------------------

class AugmentedPriceTokenizer(nn.Module):
    """
    Wraps PriceSeriesTokenizer with training-time data augmentation:
      - Gaussian jitter on prices
      - Random scaling
      - Time-shift (circular shift)
    """

    def __init__(self, cfg: PriceTokenizerConfig, jitter_std: float = 0.01,
                 scale_range: Tuple[float, float] = (0.9, 1.1)):
        super().__init__()
        self.tokenizer  = PriceSeriesTokenizer(cfg)
        self.jitter_std = jitter_std
        self.scale_range = scale_range

    def _augment(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """Apply augmentation during training."""
        if not self.training:
            return ohlcv
        # Gaussian jitter
        noise = torch.randn_like(ohlcv) * self.jitter_std
        ohlcv = ohlcv + noise

        # Random scale (preserve volume separately)
        lo, hi  = self.scale_range
        scale   = torch.empty(ohlcv.shape[0], 1, 1, device=ohlcv.device).uniform_(lo, hi)
        ohlcv_aug = ohlcv.clone()
        ohlcv_aug[:, :, :4] = ohlcv[:, :, :4] * scale  # OHLC only

        return ohlcv_aug

    def forward(self, ohlcv: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        ohlcv = self._augment(ohlcv)
        return self.tokenizer(ohlcv, **kwargs)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config dataclasses
    "PriceTokenizerConfig",
    "OrderBookTokenizerConfig",
    "OnChainTokenizerConfig",
    "NewsTokenizerConfig",
    "MultiModalTokenizerConfig",
    # Patch embedding modules
    "LinearPatchEmbedding",
    "ConvPatchEmbedding",
    "MeanPatchEmbedding",
    "QuantizedPatchEmbedding",
    # Feature extractors
    "TechnicalFeatureExtractor",
    "OnChainFeatureNormalizer",
    "OrderBookFeatureExtractor",
    "OrderBookDeltaEncoder",
    # Main tokenizers
    "PriceSeriesTokenizer",
    "MultiAssetPriceTokenizer",
    "OrderBookTokenizer",
    "OnChainTokenizer",
    "NewsTokenizer",
    "MultiModalTokenizer",
    # Utilities
    "TokenizerRegistry",
    "RunningStats",
    "PatchMasker",
    "AugmentedPriceTokenizer",
    "SimpleFinancialTokenizer",
    "FinancialSentimentHead",
    # Helper functions
    "log_return",
    "z_score_normalize",
    "min_max_normalize",
    "rolling_zscore",
    "rank_normalize",
    "compute_atr",
    "compute_rsi",
    "compute_bollinger",
    # Constants
    "MODALITY_PRICE",
    "MODALITY_ORDERBOOK",
    "MODALITY_ONCHAIN",
    "MODALITY_NEWS",
    "SPECIAL_FINANCIAL_TOKENS",
    "FINANCIAL_VOCAB_EXTENSION",
]
