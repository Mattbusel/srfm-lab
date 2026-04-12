"""
lumina/tokenizer_advanced.py

Advanced tokenization for Lumina financial foundation model.

Covers:
  - Learned quantization via VQ-VAE for price series
  - Wavelet packet tokenization of financial signals
  - Tick-level microstructure tokenizer (order book snapshots)
  - Options chain tokenizer (strike/expiry grid)
  - Cross-asset vocabulary construction
  - Tokenizer training from scratch
  - Byte-pair encoding (BPE) for financial event text
"""

from __future__ import annotations

import logging
import math
import os
import json
import pathlib
import struct
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import pywt
    _PYWAVELETS_AVAILABLE = True
except ImportError:
    _PYWAVELETS_AVAILABLE = False

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    _TOKENIZERS_AVAILABLE = True
except ImportError:
    _TOKENIZERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Base tokenizer interface
# ---------------------------------------------------------------------------

class FinancialTokenizerBase:
    """Abstract base class for financial tokenizers."""

    def encode(self, data: Any) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> Any:
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError

    def save(self, path: Union[str, pathlib.Path]) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "FinancialTokenizerBase":
        raise NotImplementedError


# ---------------------------------------------------------------------------
# VQ-VAE for price series quantization
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer (van den Oord et al., 2017).

    Maintains a codebook of embedding vectors.
    During forward pass, maps each continuous embedding to the
    nearest codebook entry (straight-through estimator for gradients).
    """

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        ema_update: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_update = ema_update

        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)

        if ema_update:
            self.register_buffer("ema_cluster_size", torch.zeros(n_embeddings))
            self.register_buffer("ema_w", torch.zeros(n_embeddings, embedding_dim))
            self.ema_decay = ema_decay
            self.ema_epsilon = ema_epsilon
            self.ema_w.data.copy_(self.embedding.weight.data)

    def forward(
        self, z: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            z: Continuous encoder output, shape (..., embedding_dim).

        Returns:
            quantized: Same shape as z, but quantized.
            loss: VQ + commitment loss.
            encoding_indices: Integer token IDs.
        """
        flat_z = z.view(-1, self.embedding_dim)

        # Distances to all codebook entries
        dist = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        encoding_indices = dist.argmin(dim=1)
        quantized = self.embedding(encoding_indices).view_as(z)

        if self.training and self.ema_update:
            # EMA update of codebook
            encodings_oh = F.one_hot(encoding_indices, self.n_embeddings).float()
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                encodings_oh.sum(dim=0), alpha=1 - self.ema_decay
            )
            dw = encodings_oh.T @ flat_z.detach()
            self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.ema_epsilon)
                / (n + self.n_embeddings * self.ema_epsilon) * n
            )
            self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        # Straight-through gradient estimator
        quantized_st = z + (quantized - z).detach()

        # VQ loss: codebook loss + commitment loss
        codebook_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized, z.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        return quantized_st, loss, encoding_indices.view(z.shape[:-1])

    def get_codebook_usage(self) -> Dict[str, Any]:
        if hasattr(self, "ema_cluster_size"):
            usage = self.ema_cluster_size.cpu().numpy()
            used = (usage > 0.5).sum()
            return {
                "total_codes": self.n_embeddings,
                "used_codes": int(used),
                "usage_pct": float(used / self.n_embeddings),
                "perplexity": float(np.exp(-np.sum((usage / usage.sum() + 1e-10) * np.log(usage / usage.sum() + 1e-10)))),
            }
        return {}


class PriceSeriesEncoder(nn.Module):
    """Encoder for price series: maps raw prices to latent embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(n_layers):
            out_ch = hidden_dim * (2 ** i) if i < n_layers - 1 else hidden_dim
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            in_ch = out_ch
        layers.append(nn.Conv1d(hidden_dim, latent_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, feature_dim, T) -> (B, T, latent_dim)"""
        out = self.net(x)        # (B, latent_dim, T)
        return out.permute(0, 2, 1)  # (B, T, latent_dim)


class PriceSeriesDecoder(nn.Module):
    """Decoder: maps quantized embeddings back to price series."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 5,   # OHLCV
        n_layers: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        layers = [nn.Conv1d(latent_dim, hidden_dim, 1)]
        in_ch = hidden_dim
        for i in range(n_layers - 1, -1, -1):
            out_ch = hidden_dim // (2 ** (n_layers - 1 - i)) if i > 0 else output_dim
            layers.extend([
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch) if i > 0 else nn.Identity(),
                nn.GELU() if i > 0 else nn.Identity(),
            ])
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, T, latent_dim) -> (B, output_dim, T)"""
        z = z.permute(0, 2, 1)  # (B, latent_dim, T)
        return self.net(z)


class VQVAETokenizer(nn.Module, FinancialTokenizerBase):
    """
    VQ-VAE tokenizer for financial price series.

    Learns a discrete vocabulary of price patterns.
    Each token represents a recurring price microstructure pattern.
    """

    def __init__(
        self,
        input_dim: int = 5,        # OHLCV
        latent_dim: int = 64,
        n_codes: int = 512,
        hidden_dim: int = 128,
        commitment_cost: float = 0.25,
        ema_update: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_codes = n_codes

        self.encoder = PriceSeriesEncoder(input_dim, hidden_dim, latent_dim)
        self.vq = VectorQuantizer(n_codes, latent_dim, commitment_cost, ema_update)
        self.decoder = PriceSeriesDecoder(latent_dim, hidden_dim, input_dim)

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x: (B, input_dim, T) — OHLCV sequence.
        Returns: (reconstruction, vq_loss, token_ids).
        """
        z = self.encoder(x)                        # (B, T, latent_dim)
        z_q, vq_loss, token_ids = self.vq(z)       # quantize
        recon = self.decoder(z_q)                  # (B, input_dim, T)
        return recon, vq_loss, token_ids

    def encode(self, x: np.ndarray) -> List[int]:
        """Encode numpy OHLCV array to token IDs."""
        self.eval()
        t = torch.from_numpy(x).float()
        if t.ndim == 2:
            t = t.T.unsqueeze(0)   # (1, input_dim, T)
        with torch.no_grad():
            z = self.encoder(t)
            _, _, ids = self.vq(z)
        return ids.squeeze().tolist()

    def decode(self, tokens: List[int]) -> np.ndarray:
        """Decode token IDs back to OHLCV sequence."""
        self.eval()
        ids = torch.tensor(tokens).long()
        with torch.no_grad():
            z_q = self.vq.embedding(ids).unsqueeze(0)  # (1, T, latent_dim)
            recon = self.decoder(z_q)
        return recon.squeeze(0).T.cpu().numpy()

    def vocab_size(self) -> int:
        return self.n_codes

    def training_loss(self, x: Tensor) -> Tensor:
        """Full VQ-VAE loss for training."""
        recon, vq_loss, _ = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        return recon_loss + vq_loss

    def save(self, path: Union[str, pathlib.Path]) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "n_codes": self.n_codes,
            },
        }, path)

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "VQVAETokenizer":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def train_tokenizer(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_epochs: int = 50,
        lr: float = 3e-4,
        device: Optional[torch.device] = None,
    ) -> List[float]:
        """Train the VQ-VAE tokenizer from scratch."""
        device = device or torch.device("cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n = 0
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = batch.get("input_ids", batch.get("features"))
                else:
                    x = batch

                x = x.float().to(device)
                if x.ndim == 3:
                    x = x.permute(0, 2, 1)  # (B, F, T)

                optimizer.zero_grad()
                loss = self.training_loss(x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1

            avg_loss = epoch_loss / max(1, n)
            losses.append(avg_loss)
            if epoch % 10 == 0:
                usage = self.vq.get_codebook_usage()
                logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}, codebook usage={usage.get('usage_pct', 0):.1%}")

        return losses


# ---------------------------------------------------------------------------
# Wavelet packet tokenization
# ---------------------------------------------------------------------------

class WaveletPacketTokenizer(FinancialTokenizerBase):
    """
    Wavelet Packet Decomposition tokenizer for financial time series.

    Decomposes price series into multi-scale frequency bands using
    wavelet packets, then quantizes each subband independently.

    Captures multi-frequency structure: trend, cycles, noise.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 3,
        n_bins: int = 64,         # Number of quantization bins per subband
        features: str = "all",    # "all" | "energy" | "coeff"
    ):
        self.wavelet = wavelet
        self.level = level
        self.n_bins = n_bins
        self.features = features
        self.n_subbands = 2 ** level
        self._total_vocab = n_bins * self.n_subbands
        self._bin_edges: Optional[np.ndarray] = None

    def _decompose(self, prices: np.ndarray) -> List[np.ndarray]:
        """Wavelet packet decomposition. Returns list of subband coefficients."""
        if not _PYWAVELETS_AVAILABLE:
            return self._fallback_decompose(prices)

        wp = pywt.WaveletPacket(data=prices, wavelet=self.wavelet, mode="periodization")
        nodes = [wp[node.path] for node in wp.get_level(self.level, "freq")]
        return [node.data for node in nodes]

    def _fallback_decompose(self, prices: np.ndarray) -> List[np.ndarray]:
        """Fallback: simple Haar-like decomposition."""
        subbands = []
        current = prices.copy()
        for _ in range(self.level):
            n = len(current)
            if n < 2:
                break
            even = current[::2]
            odd = current[1::2]
            min_len = min(len(even), len(odd))
            approx = (even[:min_len] + odd[:min_len]) / math.sqrt(2)
            detail = (even[:min_len] - odd[:min_len]) / math.sqrt(2)
            subbands.append(detail)
            current = approx
        subbands.append(current)
        return subbands

    def fit(self, price_series_list: List[np.ndarray]) -> "WaveletPacketTokenizer":
        """Fit quantization bin edges from training data."""
        all_coeffs: List[List[float]] = [[] for _ in range(self.n_subbands)]
        for prices in price_series_list:
            subbands = self._decompose(prices)
            for i, sb in enumerate(subbands[:self.n_subbands]):
                all_coeffs[i].extend(sb.tolist())

        self._bin_edges = np.zeros((self.n_subbands, self.n_bins + 1))
        for i in range(self.n_subbands):
            if all_coeffs[i]:
                self._bin_edges[i] = np.percentile(
                    all_coeffs[i],
                    np.linspace(0, 100, self.n_bins + 1)
                )
        return self

    def _quantize_subband(self, coeffs: np.ndarray, band_idx: int) -> np.ndarray:
        """Map subband coefficients to bin indices."""
        if self._bin_edges is None:
            # Default uniform bins
            min_v, max_v = coeffs.min(), coeffs.max()
            bins = np.linspace(min_v, max_v, self.n_bins + 1)
        else:
            bins = self._bin_edges[band_idx]
        return np.digitize(coeffs, bins[1:-1]).astype(np.int32)

    def encode(self, prices: np.ndarray) -> List[int]:
        """Encode price series to token sequence."""
        subbands = self._decompose(prices)
        tokens = []
        for i, sb in enumerate(subbands[:self.n_subbands]):
            quantized = self._quantize_subband(sb, i)
            # Offset tokens by band index * n_bins
            offset_tokens = quantized + i * self.n_bins
            tokens.extend(offset_tokens.tolist())
        return tokens

    def decode(self, tokens: List[int]) -> np.ndarray:
        """Approximate reconstruction from tokens (lossy)."""
        # Split tokens back into subbands
        subbands_recon = []
        for i in range(self.n_subbands):
            band_tokens = [t - i * self.n_bins for t in tokens
                           if i * self.n_bins <= t < (i + 1) * self.n_bins]
            if not band_tokens:
                continue
            if self._bin_edges is not None:
                bins = self._bin_edges[i]
                # Map bin index back to approximate coefficient value
                values = [(bins[b] + bins[b + 1]) / 2 for b in band_tokens
                          if b < len(bins) - 1]
            else:
                values = [float(b) / self.n_bins for b in band_tokens]
            subbands_recon.append(np.array(values))

        if not subbands_recon:
            return np.array([])

        # Reconstruct using IDWT if available
        if _PYWAVELETS_AVAILABLE and len(subbands_recon) == self.n_subbands:
            try:
                wp = pywt.WaveletPacket(data=None, wavelet=self.wavelet, mode="periodization")
                for i, node in enumerate(wp.get_level(self.level, "freq")):
                    if i < len(subbands_recon):
                        wp[node.path].data = subbands_recon[i]
                return wp.reconstruct(update=False)
            except Exception:
                pass

        # Fallback: concatenate subbands
        return np.concatenate(subbands_recon)

    def vocab_size(self) -> int:
        return self._total_vocab

    def save(self, path: Union[str, pathlib.Path]) -> None:
        data = {
            "wavelet": self.wavelet,
            "level": self.level,
            "n_bins": self.n_bins,
            "bin_edges": self._bin_edges.tolist() if self._bin_edges is not None else None,
        }
        pathlib.Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "WaveletPacketTokenizer":
        data = json.loads(pathlib.Path(path).read_text())
        tok = cls(data["wavelet"], data["level"], data["n_bins"])
        if data.get("bin_edges") is not None:
            tok._bin_edges = np.array(data["bin_edges"])
        return tok


# ---------------------------------------------------------------------------
# LOB microstructure tokenizer
# ---------------------------------------------------------------------------

@dataclass
class LOBTokenizerConfig:
    depth: int = 5           # LOB depth (levels)
    n_price_bins: int = 256  # Price quantization bins
    n_size_bins: int = 64    # Size quantization bins
    n_imbalance_bins: int = 32
    time_bins: int = 48      # Time of day bins (30-min intervals)


class LOBMicrostructureTokenizer(FinancialTokenizerBase):
    """
    Tokenizer for Limit Order Book snapshots.

    Each LOB snapshot is converted to a sequence of tokens:
      [bid_level_1, ask_level_1, bid_level_2, ask_level_2, ..., imbalance, spread]

    Price/size are quantized into discrete bins.
    """

    # Special tokens
    PAD = 0
    BOS = 1    # Beginning of LOB snapshot
    EOS = 2    # End of snapshot
    SEP = 3    # Separator between bid/ask
    SPECIAL_TOKENS = 4

    def __init__(self, config: LOBTokenizerConfig):
        self.config = config
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build token vocabulary."""
        cfg = self.config
        # Vocab structure:
        # [SPECIAL_TOKENS][price_tokens][size_tokens][imbalance_tokens][time_tokens]
        offset = self.SPECIAL_TOKENS
        self._price_offset = offset
        offset += cfg.n_price_bins
        self._size_offset = offset
        offset += cfg.n_size_bins
        self._imbalance_offset = offset
        offset += cfg.n_imbalance_bins
        self._time_offset = offset
        offset += cfg.time_bins
        self._vocab_size = offset

        # Bin edges (will be set during fit)
        self._price_bins: Optional[np.ndarray] = None
        self._size_bins: Optional[np.ndarray] = None

    def fit(
        self,
        lob_data: List[Dict[str, np.ndarray]],
    ) -> "LOBMicrostructureTokenizer":
        """
        Fit quantization bins from LOB data.

        Args:
            lob_data: List of dicts with keys "bid_prices", "ask_prices",
                      "bid_sizes", "ask_sizes".
        """
        all_prices, all_sizes = [], []
        for snap in lob_data:
            all_prices.extend(snap["bid_prices"].tolist())
            all_prices.extend(snap["ask_prices"].tolist())
            all_sizes.extend(snap["bid_sizes"].tolist())
            all_sizes.extend(snap["ask_sizes"].tolist())

        self._price_bins = np.percentile(
            all_prices, np.linspace(0, 100, self.config.n_price_bins + 1)
        )
        self._size_bins = np.percentile(
            all_sizes, np.linspace(0, 100, self.config.n_size_bins + 1)
        )
        return self

    def _quantize(self, value: float, bins: np.ndarray, offset: int) -> int:
        idx = np.digitize(value, bins[1:-1]).item()
        return min(idx, len(bins) - 2) + offset

    def encode_snapshot(
        self,
        bid_prices: np.ndarray,
        bid_sizes: np.ndarray,
        ask_prices: np.ndarray,
        ask_sizes: np.ndarray,
        time_of_day_min: float = 0.0,
    ) -> List[int]:
        """Encode a single LOB snapshot to token sequence."""
        tokens = [self.BOS]
        depth = min(self.config.depth, len(bid_prices), len(ask_prices))

        p_bins = self._price_bins if self._price_bins is not None else np.linspace(0, 1e6, self.config.n_price_bins + 1)
        s_bins = self._size_bins if self._size_bins is not None else np.linspace(0, 1e6, self.config.n_size_bins + 1)

        for i in range(depth):
            # Bid level
            bp_tok = self._quantize(bid_prices[i], p_bins, self._price_offset)
            bs_tok = self._quantize(bid_sizes[i], s_bins, self._size_offset)
            # Ask level
            ap_tok = self._quantize(ask_prices[i], p_bins, self._price_offset)
            as_tok = self._quantize(ask_sizes[i], s_bins, self._size_offset)
            tokens.extend([bp_tok, bs_tok, self.SEP, ap_tok, as_tok])

        # Imbalance token
        total_bid = bid_sizes[:depth].sum()
        total_ask = ask_sizes[:depth].sum()
        imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-10)
        imb_bin = int((imbalance + 1) / 2 * (self.config.n_imbalance_bins - 1))
        tokens.append(imb_bin + self._imbalance_offset)

        # Time of day token
        time_bin = int(time_of_day_min / (24 * 60) * self.config.time_bins)
        time_bin = min(time_bin, self.config.time_bins - 1)
        tokens.append(time_bin + self._time_offset)

        tokens.append(self.EOS)
        return tokens

    def encode(self, data: Any) -> List[int]:
        """data: dict with bid_prices, bid_sizes, ask_prices, ask_sizes"""
        return self.encode_snapshot(
            data["bid_prices"],
            data["bid_sizes"],
            data["ask_prices"],
            data["ask_sizes"],
            data.get("time_of_day_min", 0.0),
        )

    def decode(self, tokens: List[int]) -> Dict[str, Any]:
        """Approximate decode of tokens (lossy)."""
        p_bins = self._price_bins if self._price_bins is not None else np.linspace(0, 1e6, self.config.n_price_bins + 1)
        s_bins = self._size_bins if self._size_bins is not None else np.linspace(0, 1e6, self.config.n_size_bins + 1)

        bid_prices, bid_sizes, ask_prices, ask_sizes = [], [], [], []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == self.BOS or tok == self.EOS:
                i += 1
                continue
            if self._price_offset <= tok < self._price_offset + self.config.n_price_bins:
                bin_idx = tok - self._price_offset
                price = (p_bins[bin_idx] + p_bins[min(bin_idx + 1, len(p_bins) - 1)]) / 2
                # Alternate bid/ask
                if len(bid_prices) <= len(ask_prices):
                    bid_prices.append(price)
                else:
                    ask_prices.append(price)
            i += 1

        return {
            "bid_prices": np.array(bid_prices),
            "ask_prices": np.array(ask_prices),
            "bid_sizes": np.array(bid_sizes),
            "ask_sizes": np.array(ask_sizes),
        }

    def vocab_size(self) -> int:
        return self._vocab_size

    def save(self, path: Union[str, pathlib.Path]) -> None:
        data = {
            "config": asdict(self.config),
            "price_bins": self._price_bins.tolist() if self._price_bins is not None else None,
            "size_bins": self._size_bins.tolist() if self._size_bins is not None else None,
        }
        pathlib.Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "LOBMicrostructureTokenizer":
        data = json.loads(pathlib.Path(path).read_text())
        tok = cls(LOBTokenizerConfig(**data["config"]))
        if data.get("price_bins") is not None:
            tok._price_bins = np.array(data["price_bins"])
        if data.get("size_bins") is not None:
            tok._size_bins = np.array(data["size_bins"])
        return tok


# ---------------------------------------------------------------------------
# Options chain tokenizer
# ---------------------------------------------------------------------------

@dataclass
class OptionsTokenizerConfig:
    n_strike_bins: int = 128     # Relative strike (moneyness) bins
    n_expiry_bins: int = 32      # Days-to-expiry bins
    n_vol_bins: int = 128        # Implied volatility bins
    n_delta_bins: int = 64       # Delta bins
    n_gamma_bins: int = 64       # Gamma bins
    n_oi_bins: int = 64          # Open interest bins
    max_expiry_days: int = 730   # 2 years


class OptionsChainTokenizer(FinancialTokenizerBase):
    """
    Tokenizer for options chain data.

    Each option contract is tokenized as:
    [CALL/PUT][strike_bin][expiry_bin][iv_bin][delta_bin][gamma_bin][oi_bin]

    Supports full option surface (all strikes x expiries) tokenization.
    """

    CALL_TOKEN = 0
    PUT_TOKEN = 1
    SPECIAL_COUNT = 4   # PAD, BOS, EOS, SEP

    def __init__(self, config: OptionsTokenizerConfig):
        self.config = config
        self._build_vocab()

    def _build_vocab(self) -> None:
        cfg = self.config
        offset = self.SPECIAL_COUNT + 2  # 2 for CALL/PUT
        self._strike_offset = offset; offset += cfg.n_strike_bins
        self._expiry_offset = offset; offset += cfg.n_expiry_bins
        self._vol_offset = offset; offset += cfg.n_vol_bins
        self._delta_offset = offset; offset += cfg.n_delta_bins
        self._gamma_offset = offset; offset += cfg.n_gamma_bins
        self._oi_offset = offset; offset += cfg.n_oi_bins
        self._total_vocab = offset

    def _bin(self, value: float, min_val: float, max_val: float, n_bins: int) -> int:
        """Uniform binning."""
        normalized = (value - min_val) / (max_val - min_val + 1e-10)
        bin_idx = int(normalized * n_bins)
        return max(0, min(bin_idx, n_bins - 1))

    def encode_option(
        self,
        is_call: bool,
        moneyness: float,      # (strike - spot) / spot
        dte: float,            # Days to expiry
        iv: float,             # Implied volatility (annualized)
        delta: float,          # Option delta [-1, 1]
        gamma: float,          # Option gamma
        open_interest: float,  # Open interest
    ) -> List[int]:
        """Encode a single option contract."""
        cfg = self.config
        tokens = []

        # Option type
        tokens.append(self.CALL_TOKEN if is_call else self.PUT_TOKEN + self.SPECIAL_COUNT)

        # Strike (moneyness)
        tokens.append(self._bin(moneyness + 1, 0, 2, cfg.n_strike_bins) + self._strike_offset)

        # Expiry
        tokens.append(self._bin(dte, 0, cfg.max_expiry_days, cfg.n_expiry_bins) + self._expiry_offset)

        # Implied vol (0 to 3 = 0% to 300%)
        tokens.append(self._bin(iv, 0, 3.0, cfg.n_vol_bins) + self._vol_offset)

        # Delta (-1 to 1)
        tokens.append(self._bin(delta + 1, 0, 2, cfg.n_delta_bins) + self._delta_offset)

        # Gamma (0 to reasonable max)
        tokens.append(self._bin(gamma, 0, 0.1, cfg.n_gamma_bins) + self._gamma_offset)

        # Open interest (log-scale)
        log_oi = math.log1p(max(0, open_interest))
        tokens.append(self._bin(log_oi, 0, 15, cfg.n_oi_bins) + self._oi_offset)

        return tokens

    def encode_surface(
        self,
        chain_df: "pd.DataFrame",
        spot_price: float,
    ) -> List[int]:
        """
        Encode full options chain (surface).

        chain_df columns: is_call, strike, dte, iv, delta, gamma, open_interest.
        Returns flattened token sequence.
        """
        tokens = [self.SPECIAL_COUNT - 2]   # BOS (=2 in our layout)
        for _, row in chain_df.iterrows():
            moneyness = (row["strike"] - spot_price) / (spot_price + 1e-10)
            opt_tokens = self.encode_option(
                is_call=bool(row.get("is_call", True)),
                moneyness=moneyness,
                dte=float(row.get("dte", 30)),
                iv=float(row.get("iv", 0.2)),
                delta=float(row.get("delta", 0.5)),
                gamma=float(row.get("gamma", 0.01)),
                open_interest=float(row.get("open_interest", 0)),
            )
            tokens.extend(opt_tokens)
        tokens.append(self.SPECIAL_COUNT - 1)   # EOS (=3)
        return tokens

    def encode(self, data: Any) -> List[int]:
        if isinstance(data, dict):
            return self.encode_option(**data)
        return []

    def decode(self, tokens: List[int]) -> Dict[str, Any]:
        """Lossy decode of token sequence."""
        options = []
        i = 0
        while i + 7 <= len(tokens):
            chunk = tokens[i: i + 7]
            is_call = chunk[0] == self.CALL_TOKEN + self.SPECIAL_COUNT
            moneyness = (chunk[1] - self._strike_offset) / self.config.n_strike_bins * 2 - 1
            dte = (chunk[2] - self._expiry_offset) / self.config.n_expiry_bins * self.config.max_expiry_days
            iv = (chunk[3] - self._vol_offset) / self.config.n_vol_bins * 3.0
            delta = (chunk[4] - self._delta_offset) / self.config.n_delta_bins * 2 - 1
            options.append({
                "is_call": is_call,
                "moneyness": moneyness,
                "dte": dte,
                "iv": iv,
                "delta": delta,
            })
            i += 7
        return {"options": options}

    def vocab_size(self) -> int:
        return self._total_vocab

    def save(self, path: Union[str, pathlib.Path]) -> None:
        pathlib.Path(path).write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "OptionsChainTokenizer":
        cfg = json.loads(pathlib.Path(path).read_text())
        return cls(OptionsTokenizerConfig(**cfg))


# ---------------------------------------------------------------------------
# Cross-asset vocabulary
# ---------------------------------------------------------------------------

@dataclass
class CrossAssetVocabConfig:
    """Configuration for cross-asset vocabulary."""
    # Per-asset tokenizers
    n_equity_bins: int = 256
    n_fx_bins: int = 128
    n_rate_bins: int = 128
    n_commodity_bins: int = 128
    n_crypto_bins: int = 256

    # Asset type tokens
    n_asset_types: int = 16   # equity, fx, rate, commodity, crypto, etc.

    # Event tokens
    n_event_tokens: int = 64  # earnings, Fed, macro releases, etc.

    # Padding and special tokens
    pad_token: int = 0
    bos_token: int = 1
    eos_token: int = 2
    unk_token: int = 3
    sep_token: int = 4


class CrossAssetVocabulary:
    """
    Unified vocabulary for multi-asset financial models.

    Maps tokens from different asset classes into a single shared
    embedding space with asset-type conditioning.
    """

    ASSET_TYPES = {
        "equity": 0,
        "fx": 1,
        "rate": 2,
        "commodity": 3,
        "crypto": 4,
        "index": 5,
        "etf": 6,
        "option": 7,
        "future": 8,
        "credit": 9,
    }

    def __init__(self, config: CrossAssetVocabConfig):
        self.config = config
        self._build_vocab()

    def _build_vocab(self) -> None:
        cfg = self.config
        self.special_count = 5   # PAD, BOS, EOS, UNK, SEP

        offset = self.special_count
        self._asset_type_offset = offset; offset += cfg.n_asset_types
        self._equity_offset = offset; offset += cfg.n_equity_bins
        self._fx_offset = offset; offset += cfg.n_fx_bins
        self._rate_offset = offset; offset += cfg.n_rate_bins
        self._commodity_offset = offset; offset += cfg.n_commodity_bins
        self._crypto_offset = offset; offset += cfg.n_crypto_bins
        self._event_offset = offset; offset += cfg.n_event_tokens
        self._total_vocab = offset

        self._offsets = {
            "equity": (self._equity_offset, cfg.n_equity_bins),
            "fx": (self._fx_offset, cfg.n_fx_bins),
            "rate": (self._rate_offset, cfg.n_rate_bins),
            "commodity": (self._commodity_offset, cfg.n_commodity_bins),
            "crypto": (self._crypto_offset, cfg.n_crypto_bins),
        }

    def encode_return(
        self,
        log_return: float,
        asset_type: str,
        min_ret: float = -0.5,
        max_ret: float = 0.5,
    ) -> int:
        """Encode a log return as a token for the given asset type."""
        offset, n_bins = self._offsets.get(asset_type, (self._equity_offset, self.config.n_equity_bins))
        normalized = (log_return - min_ret) / (max_ret - min_ret + 1e-10)
        bin_idx = int(normalized * n_bins)
        bin_idx = max(0, min(bin_idx, n_bins - 1))
        return bin_idx + offset

    def encode_asset_type(self, asset_type: str) -> int:
        type_idx = self.ASSET_TYPES.get(asset_type, 0)
        return type_idx + self._asset_type_offset

    def encode_event(self, event_type: int) -> int:
        return (event_type % self.config.n_event_tokens) + self._event_offset

    def encode_sequence(
        self,
        returns: List[float],
        asset_type: str = "equity",
    ) -> List[int]:
        """Encode a return series to token sequence."""
        tokens = [self.config.bos_token, self.encode_asset_type(asset_type)]
        for ret in returns:
            tokens.append(self.encode_return(ret, asset_type))
        tokens.append(self.config.eos_token)
        return tokens

    def vocab_size(self) -> int:
        return self._total_vocab

    def save(self, path: Union[str, pathlib.Path]) -> None:
        pathlib.Path(path).write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "CrossAssetVocabulary":
        cfg = json.loads(pathlib.Path(path).read_text())
        return cls(CrossAssetVocabConfig(**cfg))


# ---------------------------------------------------------------------------
# BPE tokenizer for financial text
# ---------------------------------------------------------------------------

class FinancialBPETokenizer:
    """
    Byte-Pair Encoding tokenizer adapted for financial event text.

    Pre-trained on financial corpora: earnings transcripts, news, filings.
    Special tokens for financial entities, numbers, tickers.
    """

    SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[SEP]", "[MASK]"]
    FINANCIAL_SPECIALS = [
        "[TICKER]", "[DATE]", "[PRICE]", "[PCT]", "[EARNINGS]",
        "[GUIDANCE]", "[REVENUE]", "[EPS]", "[BUYBACK]", "[DIVIDEND]",
    ]

    def __init__(self, vocab_size: int = 32_000):
        self.vocab_size = vocab_size
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}
        self._merges: List[Tuple[str, str]] = []
        self._initialized = False
        self._build_special_tokens()

    def _build_special_tokens(self) -> None:
        all_specials = self.SPECIAL_TOKENS + self.FINANCIAL_SPECIALS
        for i, tok in enumerate(all_specials):
            self._vocab[tok] = i
            self._reverse_vocab[i] = tok

    def train(
        self,
        texts: List[str],
        min_frequency: int = 2,
    ) -> None:
        """Train BPE from a list of financial texts."""
        if _TOKENIZERS_AVAILABLE:
            self._train_with_library(texts, min_frequency)
        else:
            self._train_native(texts, min_frequency)

    def _train_with_library(self, texts: List[str], min_frequency: int) -> None:
        """Train using the tokenizers library."""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        special_tokens = self.SPECIAL_TOKENS + self.FINANCIAL_SPECIALS
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=False,
        )

        from io import StringIO
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text in texts:
                f.write(text + "\n")
            tmp_path = f.name

        tokenizer.train([tmp_path], trainer)
        os.unlink(tmp_path)

        self._tokenizer_lib = tokenizer
        self._initialized = True

    def _train_native(self, texts: List[str], min_frequency: int) -> None:
        """Native Python BPE implementation."""
        # Build initial character-level vocab from training data
        word_freq: Dict[str, int] = {}
        for text in texts:
            for word in text.lower().split():
                word_spaced = " ".join(list(word)) + " </w>"
                word_freq[word_spaced] = word_freq.get(word_spaced, 0) + 1

        # Filter by min frequency
        word_freq = {w: c for w, c in word_freq.items() if c >= min_frequency}

        # BPE: iteratively merge most frequent pairs
        vocab_offset = len(self._vocab)
        n_merges = self.vocab_size - vocab_offset

        for _ in range(n_merges):
            pairs = self._get_stats(word_freq)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_freq = self._merge_vocab(best_pair, word_freq)
            merged = "".join(best_pair)
            if merged not in self._vocab:
                idx = len(self._vocab)
                self._vocab[merged] = idx
                self._reverse_vocab[idx] = merged
                self._merges.append(best_pair)

        self._initialized = True

    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs: Dict[Tuple[str, str], int] = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_vocab(
        self,
        pair: Tuple[str, str],
        vocab: Dict[str, int],
    ) -> Dict[str, int]:
        import re
        new_vocab = {}
        pattern = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
        for word in vocab:
            new_word = pattern.sub("".join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self._initialized:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        if hasattr(self, "_tokenizer_lib"):
            output = self._tokenizer_lib.encode(text)
            return output.ids

        # Native encoding
        tokens = []
        for word in text.lower().split():
            word_spaced = " ".join(list(word)) + " </w>"
            # Apply merges
            for pair in self._merges:
                bigram = " ".join(pair)
                word_spaced = word_spaced.replace(bigram, "".join(pair))
            for sym in word_spaced.split():
                tokens.append(self._vocab.get(sym, self._vocab.get("[UNK]", 3)))
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self, "_tokenizer_lib"):
            return self._tokenizer_lib.decode(ids)
        tokens = [self._reverse_vocab.get(i, "[UNK]") for i in ids]
        return " ".join(tokens).replace(" </w>", "").strip()

    def vocab_size_actual(self) -> int:
        return len(self._vocab)

    def save(self, path: Union[str, pathlib.Path]) -> None:
        path = pathlib.Path(path)
        if hasattr(self, "_tokenizer_lib"):
            self._tokenizer_lib.save(str(path))
        else:
            data = {
                "vocab": self._vocab,
                "merges": self._merges,
                "vocab_size": self.vocab_size,
            }
            path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "FinancialBPETokenizer":
        path = pathlib.Path(path)
        tok = cls()
        if _TOKENIZERS_AVAILABLE and str(path).endswith(".json"):
            try:
                from tokenizers import Tokenizer
                tok._tokenizer_lib = Tokenizer.from_file(str(path))
                tok._initialized = True
                return tok
            except Exception:
                pass
        data = json.loads(path.read_text())
        tok._vocab = data["vocab"]
        tok._reverse_vocab = {int(v): k for k, v in data["vocab"].items()}
        tok._merges = [tuple(m) for m in data["merges"]]
        tok._initialized = True
        return tok


# ---------------------------------------------------------------------------
# Unified tokenizer factory
# ---------------------------------------------------------------------------

class TokenizerFactory:
    """Factory for creating tokenizers by type."""

    @staticmethod
    def create(tokenizer_type: str, **kwargs) -> FinancialTokenizerBase:
        if tokenizer_type == "vqvae":
            return VQVAETokenizer(**kwargs)
        elif tokenizer_type == "wavelet":
            return WaveletPacketTokenizer(**kwargs)
        elif tokenizer_type == "lob":
            cfg = LOBTokenizerConfig(**kwargs)
            return LOBMicrostructureTokenizer(cfg)
        elif tokenizer_type == "options":
            cfg = OptionsTokenizerConfig(**kwargs)
            return OptionsChainTokenizer(cfg)
        elif tokenizer_type == "cross_asset":
            cfg = CrossAssetVocabConfig(**kwargs)
            return CrossAssetVocabulary(cfg)
        elif tokenizer_type == "bpe":
            return FinancialBPETokenizer(**kwargs)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Base
    "FinancialTokenizerBase",
    # VQ-VAE
    "VectorQuantizer",
    "PriceSeriesEncoder",
    "PriceSeriesDecoder",
    "VQVAETokenizer",
    # Wavelet
    "WaveletPacketTokenizer",
    # LOB
    "LOBTokenizerConfig",
    "LOBMicrostructureTokenizer",
    # Options
    "OptionsTokenizerConfig",
    "OptionsChainTokenizer",
    # Cross-asset
    "CrossAssetVocabConfig",
    "CrossAssetVocabulary",
    # BPE
    "FinancialBPETokenizer",
    # Factory
    "TokenizerFactory",
]
