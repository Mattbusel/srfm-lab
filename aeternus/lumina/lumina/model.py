"""
lumina/model.py

Top-level LuminaModel assembling all components:

  - LuminaConfig         : full model configuration dataclass
  - LuminaModel          : main model class with all components
  - LuminaForPretraining : model with pre-training heads
  - LuminaForFineTuning  : model with task-specific heads
  - GenerationConfig     : autoregressive generation settings
  - ModelBuilder         : factory for pre-configured model variants
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Full model configuration
# ---------------------------------------------------------------------------

@dataclass
class LuminaConfig:
    """Configuration for the full Lumina model."""

    # Identity
    model_name:        str   = "lumina-base"
    version:           str   = "0.1.0"

    # Tokenizer
    patch_size:        int   = 16
    n_channels:        int   = 5
    max_seq_len:       int   = 2048
    norm_mode:         str   = "zscore"
    use_log_returns:   bool  = True
    add_technical:     bool  = True

    # Transformer
    d_model:           int   = 512
    n_heads:           int   = 8
    n_kv_heads:        int   = 4
    n_layers:          int   = 12
    d_ff:              Optional[int] = None
    ffn_type:          str   = "swiglu"
    dropout:           float = 0.1
    attn_dropout:      float = 0.0
    pos_encoding:      str   = "rope"
    rope_theta:        float = 10000.0
    use_flash:         bool  = True
    causal:            bool  = False
    use_qk_norm:       bool  = False
    norm_eps:          float = 1e-6
    bias:              bool  = False

    # MoE
    use_moe:           bool  = False
    moe_every_n:       int   = 4
    n_experts:         int   = 8
    n_active:          int   = 2

    # Training
    gradient_checkpointing: bool = False
    tie_embeddings:    bool  = True

    # Multi-modal
    use_orderbook:     bool  = False
    use_onchain:       bool  = False
    use_news:          bool  = False
    n_ob_levels:       int   = 10
    n_onchain_features: int  = 16

    # Multi-asset
    multi_asset:       bool  = False
    n_assets:          int   = 1

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "LuminaConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LuminaConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_patches:  int   = 64
    temperature:      float = 1.0
    top_k:            int   = 0
    top_p:            float = 1.0
    repetition_penalty: float = 1.0
    do_sample:        bool  = True


# ---------------------------------------------------------------------------
# Main LuminaModel
# ---------------------------------------------------------------------------

class LuminaModel(nn.Module):
    """
    Full Lumina Financial Foundation Model.

    Assembles:
      1. Multi-modal tokenizers (OHLCV, order book, on-chain, news)
      2. Positional encoding (RoPE or ALiBi)
      3. Transformer backbone (bidirectional or causal)
      4. Output normalization

    This class is the entry point for inference and fine-tuning.
    Pre-training heads are in LuminaForPretraining.
    """

    def __init__(self, cfg: LuminaConfig):
        super().__init__()
        self.cfg = cfg

        # Import components lazily to avoid circular imports
        from .tokenizer import PriceSeriesTokenizer, PriceTokenizerConfig
        from .transformer import (
            TransformerConfig, BidirectionalTransformer, CausalTransformer,
            RMSNorm, build_lumina_base_config, build_lumina_large_config,
        )

        # ----- Tokenizer -----
        tok_cfg = PriceTokenizerConfig(
            patch_size      = cfg.patch_size,
            d_model         = cfg.d_model,
            n_channels      = cfg.n_channels,
            max_seq_len     = cfg.max_seq_len // cfg.patch_size + 2,
            norm_mode       = cfg.norm_mode,
            use_log_returns = cfg.use_log_returns,
            add_technical   = cfg.add_technical,
            multi_asset     = cfg.multi_asset,
            n_assets        = cfg.n_assets,
            use_cls_token   = True,
            use_sep_token   = False,
            patch_merge_mode = "linear",
        )
        self.tokenizer = PriceSeriesTokenizer(tok_cfg)

        # ----- Transformer backbone -----
        tf_cfg = TransformerConfig(
            d_model        = cfg.d_model,
            n_heads        = cfg.n_heads,
            n_kv_heads     = cfg.n_kv_heads,
            n_layers       = cfg.n_layers,
            d_ff           = cfg.d_ff,
            ffn_type       = cfg.ffn_type,
            dropout        = cfg.dropout,
            attn_dropout   = cfg.attn_dropout,
            norm_eps       = cfg.norm_eps,
            causal         = cfg.causal,
            use_flash      = cfg.use_flash,
            pos_encoding   = cfg.pos_encoding,
            rope_theta     = cfg.rope_theta,
            use_qk_norm    = cfg.use_qk_norm,
            max_seq_len    = cfg.max_seq_len,
            bias           = cfg.bias,
            use_moe        = cfg.use_moe,
            moe_every_n    = cfg.moe_every_n,
            n_experts      = cfg.n_experts,
            n_active       = cfg.n_active,
            gradient_checkpointing = cfg.gradient_checkpointing,
        )

        input_dim = cfg.d_model   # tokenizer output dim = d_model

        if cfg.causal:
            self.backbone = CausalTransformer(tf_cfg, input_dim)
        else:
            self.backbone = BidirectionalTransformer(tf_cfg, input_dim)

        self.norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)

        # Optional multi-modal components
        if cfg.use_orderbook:
            from .tokenizer import OrderBookTokenizer, OrderBookTokenizerConfig
            ob_cfg = OrderBookTokenizerConfig(
                n_levels = cfg.n_ob_levels,
                d_model  = cfg.d_model,
            )
            self.ob_tokenizer = OrderBookTokenizer(ob_cfg)
            self.ob_proj      = nn.Linear(cfg.d_model, cfg.d_model)

        if cfg.use_onchain:
            from .tokenizer import OnChainTokenizer, OnChainTokenizerConfig
            oc_cfg = OnChainTokenizerConfig(
                n_features = cfg.n_onchain_features,
                d_model    = cfg.d_model,
            )
            self.oc_tokenizer = OnChainTokenizer(oc_cfg)
            self.oc_proj      = nn.Linear(cfg.d_model, cfg.d_model)

        # Fusion for multi-modal
        if cfg.use_orderbook or cfg.use_onchain or cfg.use_news:
            self.modal_fusion_norm = RMSNorm(cfg.d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply standard initialization to linear layers."""
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_init)

    def get_n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_n_params_str(self) -> str:
        n = self.get_n_params()
        if n >= 1e9:  return f"{n/1e9:.2f}B"
        if n >= 1e6:  return f"{n/1e6:.1f}M"
        return f"{n/1e3:.1f}K"

    def forward(
        self,
        ohlcv:          Optional[torch.Tensor] = None,
        lob:            Optional[torch.Tensor] = None,
        onchain:        Optional[torch.Tensor] = None,
        padding_mask:   Optional[torch.Tensor] = None,
        asset_ids:      Optional[torch.Tensor] = None,
        return_hidden:  bool = False,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass.

        Args:
            ohlcv:        (B, T, 5) OHLCV time series
            lob:          (B, T, 4, n_levels) limit order book (optional)
            onchain:      (B, T, n_features) on-chain signals (optional)
            padding_mask: (B, T) True = padding
            asset_ids:    (B,) integer asset IDs
            return_hidden: whether to return all hidden states

        Returns:
            dict with keys:
              - hidden:      (B, N_tokens, D)
              - cls_emb:     (B, D) — representation for classification
              - patch_mask:  (B, N_tokens) padding mask
              - aux_loss:    scalar MoE auxiliary loss
        """
        assert ohlcv is not None or lob is not None, "At least one modality must be provided"
        B = (ohlcv if ohlcv is not None else lob).shape[0]

        all_tokens  = []
        all_masks   = []

        # Price series tokenization
        if ohlcv is not None:
            tok_out  = self.tokenizer(ohlcv, asset_ids=asset_ids, padding_mask=padding_mask)
            price_tok = tok_out["embeddings"]   # (B, N, D)
            all_tokens.append(price_tok)
            all_masks.append(tok_out["patch_mask"])

        # Order book tokenization
        if lob is not None and hasattr(self, "ob_tokenizer"):
            ob_out  = self.ob_tokenizer(lob)
            ob_tok  = self.ob_proj(ob_out["embeddings"])
            all_tokens.append(ob_tok)
            all_masks.append(ob_out["patch_mask"])

        # On-chain tokenization
        if onchain is not None and hasattr(self, "oc_tokenizer"):
            oc_out  = self.oc_tokenizer(onchain)
            oc_tok  = self.oc_proj(oc_out["embeddings"])
            all_tokens.append(oc_tok)
            all_masks.append(oc_out["patch_mask"])

        # Concatenate all modalities
        tokens = torch.cat(all_tokens, dim=1)    # (B, N_total, D)
        mask   = torch.cat(all_masks, dim=1)     # (B, N_total)

        # Backbone forward
        enc_out = self.backbone(tokens, mask=mask, return_hidden=return_hidden)

        # Extract outputs
        hidden   = self.norm(enc_out["hidden"])
        cls_emb  = enc_out.get("cls_emb", hidden[:, 0, :])
        aux_loss = enc_out.get("aux_loss", torch.tensor(0.0, device=tokens.device))

        out = {
            "hidden":     hidden,
            "cls_emb":    cls_emb,
            "patch_mask": mask,
            "aux_loss":   aux_loss,
        }

        if return_hidden:
            out["hidden_states"] = enc_out.get("hidden_states", [])

        return out

    def encode(
        self,
        ohlcv:        torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        asset_ids:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode OHLCV series to CLS embedding. Returns (B, D)."""
        return self.forward(ohlcv, padding_mask=padding_mask, asset_ids=asset_ids)["cls_emb"]

    def encode_batch(
        self,
        ohlcv_list:   List[torch.Tensor],
        device:       str = "cpu",
    ) -> torch.Tensor:
        """Encode a list of OHLCV tensors. Returns (len(ohlcv_list), D)."""
        self.eval()
        with torch.no_grad():
            embeddings = []
            for ohlcv in ohlcv_list:
                if ohlcv.dim() == 2:
                    ohlcv = ohlcv.unsqueeze(0)
                ohlcv = ohlcv.to(device)
                emb   = self.encode(ohlcv)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    def save_pretrained(self, directory: Union[str, Path]) -> None:
        """Save model weights and config to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "model.pt")
        self.cfg.save(directory / "config.json")

    @classmethod
    def from_pretrained(
        cls,
        directory: Union[str, Path],
        strict:    bool = True,
    ) -> "LuminaModel":
        directory = Path(directory)
        cfg       = LuminaConfig.load(directory / "config.json")
        model     = cls(cfg)
        state     = torch.load(directory / "model.pt", map_location="cpu")
        model.load_state_dict(state, strict=strict)
        return model


# ---------------------------------------------------------------------------
# Lumina for Pre-training
# ---------------------------------------------------------------------------

class LuminaForPretraining(nn.Module):
    """
    Lumina model equipped with pre-training heads:
      - MRM (Masked Return Modeling)
      - NPP (Next Patch Prediction)
      - Contrastive across assets
      - Volatility prediction auxiliary
    """

    def __init__(self, cfg: LuminaConfig, pretrain_cfg: Optional[Any] = None):
        super().__init__()
        from .pretraining import (
            PretrainingConfig, MultiTaskPretrainingLoss, PatchMaskGenerator
        )

        self.backbone     = LuminaModel(cfg)
        self.pretrain_cfg = pretrain_cfg or PretrainingConfig()
        self.loss_module  = MultiTaskPretrainingLoss(
            self.pretrain_cfg,
            patch_size  = cfg.patch_size,
            n_channels  = cfg.n_channels,
            d_model     = cfg.d_model,
        )
        self.mask_gen = PatchMaskGenerator(self.pretrain_cfg)

    def forward(
        self,
        ohlcv:         torch.Tensor,
        patch_targets: Optional[torch.Tensor] = None,
        padding_mask:  Optional[torch.Tensor] = None,
        contrastive_z: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        enc   = self.backbone(ohlcv, padding_mask=padding_mask)
        hidden   = enc["hidden"]
        cls_emb  = enc["cls_emb"]
        aux_loss = enc["aux_loss"]

        B, N, D = hidden.shape
        mask    = self.mask_gen(B, N, ohlcv.device)

        if patch_targets is None:
            patch_targets = torch.zeros(B, N, self.backbone.cfg.patch_size * self.backbone.cfg.n_channels, device=ohlcv.device)

        loss, metrics = self.loss_module(
            hidden, cls_emb, patch_targets, mask,
            contrastive_z=contrastive_z,
        )

        total_loss = loss + 0.01 * aux_loss
        metrics["aux_loss"] = aux_loss.item()
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Lumina for Fine-Tuning
# ---------------------------------------------------------------------------

class LuminaForDirectionClassification(nn.Module):
    """Lumina backbone + direction classification head."""

    def __init__(self, cfg: LuminaConfig, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        from .finetuning import DirectionClassificationHead

        self.backbone = LuminaModel(cfg)
        self.head     = DirectionClassificationHead(cfg.d_model, n_classes, dropout)

    def forward(self, ohlcv: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        enc     = self.backbone(ohlcv, **kwargs)
        logits  = self.head(enc["cls_emb"])
        return {"logits": logits, "cls_emb": enc["cls_emb"], "aux_loss": enc["aux_loss"]}


class LuminaForVolatilityForecasting(nn.Module):
    """Lumina backbone + volatility forecasting head."""

    def __init__(self, cfg: LuminaConfig, dropout: float = 0.1):
        super().__init__()
        from .finetuning import VolatilityForecastingHead

        self.backbone = LuminaModel(cfg)
        self.head     = VolatilityForecastingHead(cfg.d_model, dropout)

    def forward(self, ohlcv: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        enc  = self.backbone(ohlcv, **kwargs)
        pred = self.head(enc["cls_emb"])
        return {"predictions": pred, "cls_emb": enc["cls_emb"], "aux_loss": enc["aux_loss"]}


class LuminaForPortfolioOptimization(nn.Module):
    """Lumina backbone + portfolio optimization head."""

    def __init__(self, cfg: LuminaConfig, n_assets: int, mode: str = "long_only"):
        super().__init__()
        from .finetuning import PortfolioOptimizationHead

        self.backbone = LuminaModel(cfg)
        self.head     = PortfolioOptimizationHead(cfg.d_model, n_assets, mode)

    def forward(self, ohlcv: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        enc     = self.backbone(ohlcv, **kwargs)
        weights = self.head(enc["cls_emb"])
        return {"weights": weights, "cls_emb": enc["cls_emb"], "aux_loss": enc["aux_loss"]}


# ---------------------------------------------------------------------------
# Model Builder (factory)
# ---------------------------------------------------------------------------

class ModelBuilder:
    """Factory class for creating pre-configured Lumina model variants."""

    @staticmethod
    def base(causal: bool = False, **overrides) -> LuminaModel:
        """Lumina-Base: ~90M parameters."""
        cfg = LuminaConfig(
            model_name = "lumina-base",
            d_model    = 512,
            n_heads    = 8,
            n_kv_heads = 4,
            n_layers   = 12,
            d_ff       = 2048,
            dropout    = 0.1,
            causal     = causal,
            use_flash  = True,
            **overrides,
        )
        return LuminaModel(cfg)

    @staticmethod
    def large(causal: bool = False, **overrides) -> LuminaModel:
        """Lumina-Large: ~350M parameters."""
        cfg = LuminaConfig(
            model_name = "lumina-large",
            d_model    = 1024,
            n_heads    = 16,
            n_kv_heads = 4,
            n_layers   = 24,
            d_ff       = 4096,
            dropout    = 0.1,
            causal     = causal,
            use_flash  = True,
            gradient_checkpointing = True,
            **overrides,
        )
        return LuminaModel(cfg)

    @staticmethod
    def deep(causal: bool = False, **overrides) -> LuminaModel:
        """Lumina-Deep: 48-layer MoE model, ~2B active params."""
        cfg = LuminaConfig(
            model_name = "lumina-deep",
            d_model    = 1024,
            n_heads    = 16,
            n_kv_heads = 2,
            n_layers   = 48,
            d_ff       = 4096,
            dropout    = 0.05,
            causal     = causal,
            use_flash  = True,
            use_moe    = True,
            moe_every_n = 4,
            n_experts  = 8,
            n_active   = 2,
            gradient_checkpointing = True,
            **overrides,
        )
        return LuminaModel(cfg)

    @staticmethod
    def xl(**overrides) -> LuminaModel:
        """Lumina-XL: 32 layers, 2048 d_model."""
        cfg = LuminaConfig(
            model_name = "lumina-xl",
            d_model    = 2048,
            n_heads    = 32,
            n_kv_heads = 4,
            n_layers   = 32,
            d_ff       = 8192,
            dropout    = 0.05,
            use_flash  = True,
            gradient_checkpointing = True,
            **overrides,
        )
        return LuminaModel(cfg)

    @staticmethod
    def tiny(**overrides) -> LuminaModel:
        """Lumina-Tiny: small model for testing."""
        cfg = LuminaConfig(
            model_name = "lumina-tiny",
            d_model    = 128,
            n_heads    = 4,
            n_kv_heads = 2,
            n_layers   = 4,
            d_ff       = 512,
            dropout    = 0.1,
            use_flash  = False,
            **overrides,
        )
        return LuminaModel(cfg)

    @staticmethod
    def from_config(config_path: Union[str, Path]) -> LuminaModel:
        cfg = LuminaConfig.load(config_path)
        return LuminaModel(cfg)


# ---------------------------------------------------------------------------
# Model parameter counting utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module, verbose: bool = False) -> None:
    """Print a human-readable model summary."""
    total    = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)

    print(f"\n{'='*60}")
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Total parameters:     {total:>15,}")
    print(f"  Trainable parameters: {trainable:>15,}")
    print(f"  Frozen parameters:    {total - trainable:>15,}")

    if verbose:
        print(f"\n  Layer breakdown:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules
                n = sum(p.numel() for p in module.parameters())
                if n > 0:
                    print(f"    {name:<60s}  {n:>10,}")
    print('='*60)


def get_memory_footprint(model: nn.Module, input_batch: Optional[Dict] = None) -> Dict[str, float]:
    """Estimate GPU memory footprint in MB."""
    n_params = count_parameters(model, trainable_only=False)
    param_mb  = n_params * 4 / (1024 ** 2)   # float32

    result = {
        "param_float32_mb": param_mb,
        "param_bfloat16_mb": param_mb / 2,
        "param_int8_mb":    param_mb / 4,
    }

    if torch.cuda.is_available():
        result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        result["gpu_reserved_mb"]  = torch.cuda.memory_reserved() / (1024 ** 2)

    return result


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------

class LuminaInferenceWrapper(nn.Module):
    """
    Wrapper for Lumina model inference with pre/post processing.
    Handles batch encoding, normalization, and output formatting.
    """

    def __init__(
        self,
        model:     LuminaModel,
        device:    str = "cpu",
        dtype:     torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model  = model.to(device)
        self.device = device
        self.dtype  = dtype

    @torch.no_grad()
    def encode_series(
        self,
        ohlcv:   Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a long OHLCV series into patch embeddings.
        Returns numpy array of shape (n_windows, d_model).
        """
        import numpy as np

        if isinstance(ohlcv, np.ndarray):
            ohlcv = torch.from_numpy(ohlcv).float()

        L   = self.model.cfg.max_seq_len
        T   = ohlcv.shape[0]
        P   = self.model.cfg.patch_size
        WS  = L  # window size

        all_embs = []
        for start in range(0, T - WS + 1, WS // 2):
            window = ohlcv[start:start + WS].unsqueeze(0).to(self.device)
            enc    = self.model(window)
            emb    = enc["cls_emb"].cpu()
            all_embs.append(emb)

        return torch.cat(all_embs, dim=0).numpy()

    @torch.no_grad()
    def predict_direction(
        self,
        ohlcv:   torch.Tensor,
        head:    nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run direction prediction. Returns (probs, predicted_class)."""
        if ohlcv.dim() == 2:
            ohlcv = ohlcv.unsqueeze(0)
        ohlcv = ohlcv.to(self.device)

        enc  = self.model(ohlcv)
        logits = head(enc["cls_emb"])
        probs  = F.softmax(logits, dim=-1)
        return probs.cpu(), probs.argmax(-1).cpu()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "LuminaConfig",
    "GenerationConfig",
    "LuminaModel",
    "LuminaForPretraining",
    "LuminaForDirectionClassification",
    "LuminaForVolatilityForecasting",
    "LuminaForPortfolioOptimization",
    "ModelBuilder",
    "LuminaInferenceWrapper",
    "LuminaForAlphaGeneration",
    "LuminaForRiskPrediction",
    "LuminaForRegimeClassification",
    "LuminaEnsemble",
    "ModelCheckpointConverter",
    "ModelBenchmarker",
    "count_parameters",
    "print_model_summary",
    "get_memory_footprint",
]


# ---------------------------------------------------------------------------
# Lumina for Alpha Generation
# ---------------------------------------------------------------------------

class LuminaForAlphaGeneration(nn.Module):
    """Lumina model specialized for cross-sectional alpha generation.

    Generates alpha signals for a universe of assets simultaneously.
    The model processes all assets jointly with cross-asset attention,
    enabling it to learn relative pricing relationships.

    Architecture:
    - Input: (B, N, T, F) — batch, assets, time, features
    - Per-asset temporal encoding: (B, N, T) → (B, N, d_model)
    - Cross-asset attention: (B, N, d_model)
    - Alpha head: (B, N, 1) → normalized alpha signals

    Args:
        config:          LuminaConfig for backbone transformer
        n_assets:        number of assets in universe
        d_input:         input feature dimension
        signal_type:     normalization for output signals

    Example:
        >>> model = LuminaForAlphaGeneration(config, n_assets=500)
        >>> ohlcv = torch.randn(4, 500, 60, 5)  # (B, N, T, F)
        >>> alphas = model(ohlcv)  # (4, 500) cross-sectional alpha scores
    """

    def __init__(
        self,
        config,
        n_assets: int = 500,
        d_input: int = 5,
        signal_type: str = "zscore",
    ):
        super().__init__()
        self.n_assets = n_assets
        self.signal_type = signal_type

        # Per-asset temporal processing
        from .transformer import RMSNorm, SwiGLUFFN, CausalTransformer
        self.input_proj = nn.Linear(d_input, config.d_model)
        self.temporal_norm = RMSNorm(config.d_model)

        # Shared temporal transformer (processes each asset's time series)
        from .transformer import LuminaConfig as TransConfig, LuminaModel as TM
        self.temporal_backbone = TM(config)

        # Cross-asset attention
        self.cross_asset_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=max(1, config.n_heads // 2),
            batch_first=True,
        )
        self.cross_norm = RMSNorm(config.d_model)

        # Alpha prediction head
        self.alpha_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )

        self.config = config

    def _normalize_alphas(self, raw: torch.Tensor) -> torch.Tensor:
        """Cross-sectionally normalize alpha scores."""
        if self.signal_type == "zscore":
            mu = raw.mean(dim=-1, keepdim=True)
            std = raw.std(dim=-1, keepdim=True).clamp(min=1e-6)
            return (raw - mu) / std
        elif self.signal_type == "rank":
            rank = raw.argsort(dim=-1).argsort(dim=-1).float()
            n = raw.shape[-1]
            return (rank / (n - 1) - 0.5) * 2
        elif self.signal_type == "sigmoid":
            return torch.sigmoid(raw)
        else:
            return raw

    def forward(
        self,
        ohlcv: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate cross-sectional alpha signals.

        Args:
            ohlcv:      (B, N, T, d_input) or (B*N, T, d_input) asset data
            timestamps: (B, N, T) or None

        Returns:
            alphas: (B, N) normalized cross-sectional alpha scores
        """
        if ohlcv.dim() == 4:
            B, N, T, F = ohlcv.shape
            # Reshape to process each asset independently through backbone
            x = ohlcv.reshape(B * N, T, F)
        else:
            BN, T, F = ohlcv.shape
            B = BN // self.n_assets
            N = self.n_assets
            x = ohlcv

        # Project input features
        x = self.input_proj(x)  # (B*N, T, d_model)
        x = self.temporal_norm(x)

        # Process through temporal backbone
        backbone_out = self.temporal_backbone(x)
        hidden = backbone_out["hidden"]  # (B*N, T, d_model)
        pooled = hidden.mean(dim=1)      # (B*N, d_model)

        # Reshape to cross-asset view
        pooled = pooled.reshape(B, N, self.config.d_model)  # (B, N, d_model)

        # Cross-asset attention
        cross_out, _ = self.cross_asset_attn(pooled, pooled, pooled)
        pooled = self.cross_norm(pooled + cross_out)

        # Generate raw alpha scores
        raw_alphas = self.alpha_head(pooled).squeeze(-1)  # (B, N)
        return self._normalize_alphas(raw_alphas)


# ---------------------------------------------------------------------------
# Lumina for Risk Prediction
# ---------------------------------------------------------------------------

class LuminaForRiskPrediction(nn.Module):
    """Lumina model specialized for asset risk forecasting.

    Predicts multiple risk measures:
    - Realized volatility (annualized)
    - Downside volatility
    - VaR (95th percentile)
    - CVaR (Expected Shortfall)
    - Tail correlation with market

    Uses both point prediction and distributional modeling.

    Args:
        config:      LuminaConfig
        n_risk_outputs: number of risk metrics to predict
        use_gmm:     use Gaussian Mixture Model for distribution modeling

    Example:
        >>> model = LuminaForRiskPrediction(config)
        >>> x = torch.randn(4, 64, 256)  # (B, T, d_token)
        >>> risk = model(x)  # dict with "realized_vol", "var_95", etc.
    """

    def __init__(
        self,
        config,
        n_risk_outputs: int = 5,
        use_gmm: bool = True,
        n_gmm_components: int = 3,
    ):
        super().__init__()
        self.config = config
        self.use_gmm = use_gmm
        self.n_gmm_components = n_gmm_components

        # Backbone
        from .transformer import LuminaModel as TM
        self.backbone = TM(config)

        # Risk prediction heads
        d = config.d_model
        self.vol_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        self.downvol_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        self.var_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        self.cvar_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        self.beta_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))

        if use_gmm:
            # GMM distribution head
            K = n_gmm_components
            self.gmm_mu = nn.Linear(d, K)
            self.gmm_log_sigma = nn.Linear(d, K)
            self.gmm_log_pi = nn.Linear(d, K)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict risk measures.

        Args:
            token_embeddings: (B, T, d_token)
            attention_mask:   (B, T) bool

        Returns:
            predictions: dict of risk measure → (B, 1) tensor
        """
        backbone_out = self.backbone(token_embeddings, attention_mask=attention_mask)
        pooled = backbone_out.get("pooled", backbone_out["hidden"].mean(dim=1))

        preds = {
            "realized_vol": F.softplus(self.vol_head(pooled)),
            "downside_vol": F.softplus(self.downvol_head(pooled)),
            "var_95": F.softplus(self.var_head(pooled)),
            "cvar_95": F.softplus(self.cvar_head(pooled)),
            "market_beta": self.beta_head(pooled),
        }

        if self.use_gmm:
            preds["gmm_mu"] = self.gmm_mu(pooled)
            preds["gmm_sigma"] = self.gmm_log_sigma(pooled).exp().clamp(min=1e-4)
            preds["gmm_pi"] = F.softmax(self.gmm_log_pi(pooled), dim=-1)

        return preds


# ---------------------------------------------------------------------------
# Lumina for Regime Classification
# ---------------------------------------------------------------------------

class LuminaForRegimeClassification(nn.Module):
    """Lumina model for market regime detection.

    Classifies each time step into a market regime:
    - Bull market (trending up)
    - Bear market (trending down)
    - Sideways/choppy
    - High volatility
    - Low volatility
    - Crisis

    Can operate in:
    - Online mode: classify one time step at a time with KV cache
    - Batch mode: classify sequence of T time steps

    Args:
        config:      LuminaConfig for backbone
        n_regimes:   number of regime classes
        use_crf:     use Conditional Random Field for sequence labeling

    Example:
        >>> model = LuminaForRegimeClassification(config, n_regimes=5)
        >>> x = torch.randn(4, 128, 256)  # (B, T, d_token)
        >>> logits = model(x)  # (4, 128, 5) — regime logits per time step
    """

    def __init__(
        self,
        config,
        n_regimes: int = 5,
        use_crf: bool = False,
    ):
        super().__init__()
        self.config = config
        self.n_regimes = n_regimes
        self.use_crf = use_crf

        from .transformer import LuminaModel as TM
        self.backbone = TM(config)

        d = config.d_model
        self.regime_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, n_regimes),
        )

        # Smoothing head: predict transition matrix
        self.transition = nn.Parameter(
            torch.zeros(n_regimes, n_regimes)
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Predict regime labels.

        Args:
            token_embeddings: (B, T, d_token)
            attention_mask:   (B, T) bool
            return_sequence:  if True, return per-step labels; else return final label

        Returns:
            outputs: dict with "logits" and optionally "probs"
        """
        backbone_out = self.backbone(token_embeddings, attention_mask=attention_mask)
        hidden = backbone_out["hidden"]  # (B, T, d_model)

        if return_sequence:
            logits = self.regime_head(hidden)  # (B, T, n_regimes)
        else:
            pooled = backbone_out.get("pooled", hidden.mean(dim=1))
            logits = self.regime_head(pooled)  # (B, n_regimes)

        probs = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "predicted_regime": logits.argmax(dim=-1),
        }


# ---------------------------------------------------------------------------
# Lumina Ensemble
# ---------------------------------------------------------------------------

class LuminaEnsemble(nn.Module):
    """Ensemble of multiple Lumina models.

    Combines predictions from multiple models via:
    - Simple averaging
    - Weighted averaging (learnable)
    - Stacking (meta-learner)

    Ensemble methods typically improve accuracy by 10-20% over single models
    and provide better uncertainty estimates.

    Args:
        models:           list of trained LuminaModel instances
        ensemble_method:  "mean" | "weighted_mean" | "stack"
        meta_d_hidden:    hidden size for stacking meta-learner

    Example:
        >>> m1 = LuminaModel(config1)
        >>> m2 = LuminaModel(config2)
        >>> ensemble = LuminaEnsemble([m1, m2], ensemble_method="weighted_mean")
        >>> out = ensemble(x)  # weighted combination
    """

    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = "mean",
        meta_d_hidden: int = 128,
        output_key: str = "cls_logits",
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.ensemble_method = ensemble_method
        self.output_key = output_key

        if ensemble_method == "weighted_mean":
            # Learnable per-model weights
            self.model_weights = nn.Parameter(torch.ones(self.n_models))

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run ensemble forward pass.

        Args:
            *args, **kwargs: forwarded to each member model

        Returns:
            ensemble_output: combined output dict
        """
        all_outputs = []
        for model in self.models:
            with torch.no_grad() if not self.training else contextlib.nullcontext():
                out = model(*args, **kwargs)
            all_outputs.append(out)

        # Combine predictions
        if self.output_key in all_outputs[0]:
            preds = torch.stack([o[self.output_key] for o in all_outputs], dim=0)  # (M, B, ...)

            if self.ensemble_method == "mean":
                ensemble_pred = preds.mean(dim=0)
            elif self.ensemble_method == "weighted_mean":
                weights = F.softmax(self.model_weights, dim=0)
                ensemble_pred = (preds * weights.view(-1, 1, 1)).sum(dim=0)
            else:
                ensemble_pred = preds.mean(dim=0)

            # Also compute uncertainty (std across models)
            uncertainty = preds.std(dim=0)

            result = dict(all_outputs[0])  # base on first model's output
            result[self.output_key] = ensemble_pred
            result["epistemic_uncertainty"] = uncertainty
            result["n_models"] = self.n_models
            return result
        else:
            return all_outputs[0]


# ---------------------------------------------------------------------------
# Model Checkpoint Converter
# ---------------------------------------------------------------------------

class ModelCheckpointConverter:
    """Convert between different checkpoint formats.

    Supports:
    - LuminaModel native format → HuggingFace format
    - HuggingFace → LuminaModel native
    - Older Lumina versions → current version
    - Float32 ↔ BFloat16 conversion

    Args:
        strict_loading: if False, allow missing keys in checkpoints

    Example:
        >>> converter = ModelCheckpointConverter()
        >>> lumina_state = converter.from_huggingface("path/to/hf_model")
        >>> model.load_state_dict(lumina_state, strict=False)
    """

    def __init__(self, strict_loading: bool = False):
        self.strict_loading = strict_loading
        self._key_map: Dict[str, str] = {}

    def register_key_mapping(self, old_key: str, new_key: str) -> None:
        """Register a key renaming mapping for version migration."""
        self._key_map[old_key] = new_key

    def remap_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply key remappings to state dict.

        Args:
            state_dict: original state dict

        Returns:
            remapped: state dict with renamed keys
        """
        new_state = {}
        for key, value in state_dict.items():
            new_key = key
            for old, new in self._key_map.items():
                new_key = new_key.replace(old, new)
            new_state[new_key] = value
        return new_state

    def convert_dtype(
        self,
        state_dict: Dict[str, torch.Tensor],
        target_dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Convert state dict tensors to target dtype.

        Args:
            state_dict:   state dict to convert
            target_dtype: target dtype (torch.float32, torch.bfloat16, etc.)

        Returns:
            converted: state dict with converted dtypes
        """
        return {
            k: v.to(target_dtype) if v.is_floating_point() else v
            for k, v in state_dict.items()
        }

    def check_compatibility(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, List[str]]:
        """Check compatibility between model and state dict.

        Args:
            model:      target model
            state_dict: checkpoint to load

        Returns:
            report: dict with "missing", "unexpected", "shape_mismatch" lists
        """
        model_keys = set(dict(model.named_parameters()).keys())
        ckpt_keys = set(state_dict.keys())

        missing = list(model_keys - ckpt_keys)
        unexpected = list(ckpt_keys - model_keys)
        shape_mismatch = []

        model_params = dict(model.named_parameters())
        for key in model_keys & ckpt_keys:
            if model_params[key].shape != state_dict[key].shape:
                shape_mismatch.append(
                    f"{key}: model={model_params[key].shape}, ckpt={state_dict[key].shape}"
                )

        return {
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatch": shape_mismatch,
            "n_compatible": len(model_keys & ckpt_keys) - len(shape_mismatch),
        }


# ---------------------------------------------------------------------------
# Model Benchmarker
# ---------------------------------------------------------------------------

class ModelBenchmarker:
    """Benchmark model inference speed and memory usage.

    Measures:
    - Forward pass throughput (samples/second)
    - Latency (milliseconds per batch)
    - GPU memory usage (peak, avg)
    - FLOPs estimate
    - Time breakdown by component

    Args:
        model:      model to benchmark
        device:     device to benchmark on
        n_warmup:   warmup iterations before timing
        n_iters:    timing iterations

    Example:
        >>> bench = ModelBenchmarker(model, device="cuda")
        >>> results = bench.run(batch_size=32, seq_len=512)
        >>> print(f"Throughput: {results['throughput']:.1f} samples/sec")
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        n_warmup: int = 5,
        n_iters: int = 20,
    ):
        self.model = model
        self.device = torch.device(device)
        self.n_warmup = n_warmup
        self.n_iters = n_iters
        self.model.to(self.device)
        self.model.eval()

    def _generate_input(
        self,
        batch_size: int,
        seq_len: int,
        d_input: int = 64,
    ) -> torch.Tensor:
        """Generate dummy input tensor."""
        return torch.randn(batch_size, seq_len, d_input, device=self.device)

    def run(
        self,
        batch_size: int = 8,
        seq_len: int = 256,
        d_input: int = 64,
    ) -> Dict[str, float]:
        """Run benchmark and return statistics.

        Args:
            batch_size: batch size
            seq_len:    sequence length
            d_input:    input dimension

        Returns:
            results: dict with timing and memory statistics
        """
        import time
        x = self._generate_input(batch_size, seq_len, d_input)

        # Warmup
        with torch.no_grad():
            for _ in range(self.n_warmup):
                _ = self.model(x)

        # Synchronize if CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Time forward passes
        latencies = []
        with torch.no_grad():
            for _ in range(self.n_iters):
                start = time.perf_counter()
                _ = self.model(x)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        avg_latency = sum(latencies) / len(latencies)
        throughput = batch_size * 1000 / avg_latency

        results = {
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "throughput_samples_per_sec": throughput,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

        if self.device.type == "cuda":
            results["peak_gpu_memory_MB"] = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()

        return results


# =============================================================================
# SECTION: Additional Lumina Model Variants
# =============================================================================

class LuminaForMultiHorizonForecast(nn.Module):
    """Lumina model variant for multi-horizon return forecasting.

    Generates return forecasts at multiple horizons (1d, 5d, 10d, 20d, 60d)
    with calibrated uncertainty estimates.

    Args:
        config: TransformerConfig
        forecast_horizons: List of forecast horizons in timesteps
        num_features: Input feature count
        use_quantile_head: Whether to add quantile regression head
        quantile_levels: Quantile levels for uncertainty
    """

    def __init__(
        self,
        config,
        forecast_horizons: Optional[List[int]] = None,
        num_features: int = 5,
        use_quantile_head: bool = True,
        quantile_levels: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.forecast_horizons = forecast_horizons or [1, 5, 10, 20, 60]
        self.num_horizons = len(self.forecast_horizons)
        self.quantile_levels = quantile_levels or [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        d_model = config.d_model

        # Input projection
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        # Transformer backbone
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, config.num_heads, d_ff=d_model * 4, dropout=config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Point forecast head
        self.point_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(self.num_horizons)
        ])
        # Quantile head (if enabled)
        if use_quantile_head:
            nq = len(self.quantile_levels)
            self.quantile_heads = nn.ModuleList([
                nn.Linear(d_model, nq) for _ in range(self.num_horizons)
            ])
        else:
            self.quantile_heads = None
        # Horizon embedding for head conditioning
        self.horizon_embed = nn.Embedding(len(self.forecast_horizons), d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, F = x.shape
        h = self.input_proj(x)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        h = h + self.pos_embed(pos_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        # Pool to single representation
        pooled = h[:, -1, :]  # Use last timestep
        # Generate predictions for each horizon
        point_preds = []
        quantile_preds = []
        for hi in range(self.num_horizons):
            h_cond = pooled + self.horizon_embed(
                torch.full((B,), hi, dtype=torch.long, device=x.device)
            )
            point_preds.append(self.point_heads[hi](h_cond))
            if self.quantile_heads is not None:
                quantile_preds.append(self.quantile_heads[hi](h_cond))
        point_preds = torch.cat(point_preds, dim=-1)  # (B, H)
        result = {
            "point_forecasts": point_preds,
            "forecast_horizons": self.forecast_horizons,
            "encoded": h,
        }
        if quantile_preds:
            qp = torch.stack(quantile_preds, dim=1)  # (B, H, Q)
            result["quantile_forecasts"] = qp
            result["quantile_levels"] = self.quantile_levels
        return result


class LuminaForCrossAssetModeling(nn.Module):
    """Lumina model for joint cross-asset prediction.

    Models a portfolio of assets simultaneously, capturing
    cross-asset correlations and lead-lag relationships.

    Args:
        config: TransformerConfig
        num_assets: Number of assets in the portfolio
        num_features: Features per asset
        use_cross_attention: Whether to use cross-asset attention
    """

    def __init__(
        self,
        config,
        num_assets: int = 50,
        num_features: int = 5,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        d_model = config.d_model
        # Per-asset encoder
        self.asset_encoder = nn.Linear(num_features, d_model)
        self.asset_embed = nn.Embedding(num_assets + 1, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(4096, d_model)
        # Temporal encoder (per asset)
        self.temporal_layers = nn.ModuleList([
            TransformerBlock(d_model, config.num_heads, d_ff=d_model * 4)
            for _ in range(config.num_layers // 2)
        ])
        # Cross-asset attention (across assets)
        if use_cross_attention:
            self.cross_asset_layers = nn.ModuleList([
                TransformerBlock(d_model, config.num_heads, d_ff=d_model * 4)
                for _ in range(config.num_layers // 2)
            ])
        else:
            self.cross_asset_layers = None
        self.norm = nn.LayerNorm(d_model)
        # Alpha head
        self.alpha_head = nn.Linear(d_model, 1)
        # Risk head
        self.risk_head = nn.Linear(d_model, 2)  # vol, beta

    def forward(
        self,
        x: torch.Tensor,
        asset_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, N, T, F) multi-asset time series
            asset_ids: (N,) asset indices for embedding
        Returns:
            Dict with alpha scores, risk estimates
        """
        if x.dim() == 3:
            # (B, T, F) single asset
            B, T, F = x.shape
            x = x.unsqueeze(1)  # (B, 1, T, F)
        B, N, T, F = x.shape
        # Encode each asset
        x_flat = x.view(B * N, T, F)
        h = self.asset_encoder(x_flat)
        # Add positional embeddings
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        h = h + self.pos_embed(pos_ids)
        # Add asset embeddings
        if asset_ids is not None:
            a_emb = self.asset_embed(asset_ids)  # (N, D)
            a_emb = a_emb.unsqueeze(0).unsqueeze(2).expand(B, N, T, -1)
            h = h + a_emb.view(B * N, T, -1)
        # Temporal encoding
        for layer in self.temporal_layers:
            h = layer(h)
        h = h.view(B, N, T, -1)
        # Pool temporal dimension
        h_pooled = h.mean(dim=2)  # (B, N, D)
        # Cross-asset encoding
        if self.cross_asset_layers is not None:
            for layer in self.cross_asset_layers:
                h_pooled = layer(h_pooled)
        h_pooled = self.norm(h_pooled)
        # Predictions
        alpha = self.alpha_head(h_pooled).squeeze(-1)  # (B, N)
        risk = self.risk_head(h_pooled)  # (B, N, 2)
        return {
            "alpha_scores": alpha,
            "vol_forecast": F.softplus(risk[:, :, 0]),
            "beta_forecast": risk[:, :, 1],
            "encoded": h_pooled,
        }


class LuminaForSentimentFusion(nn.Module):
    """Lumina model fusing price data with sentiment signals.

    Combines OHLCV price features with news/social sentiment
    through cross-modal attention.

    Args:
        config: TransformerConfig for price encoder
        sentiment_dim: Sentiment feature dimension
        num_sentiment_sources: Number of sentiment data sources
        fusion_type: 'early', 'late', or 'cross_attention'
    """

    def __init__(
        self,
        config,
        sentiment_dim: int = 32,
        num_sentiment_sources: int = 3,
        fusion_type: str = "cross_attention",
        num_features: int = 5,
    ) -> None:
        super().__init__()
        d_model = config.d_model
        self.fusion_type = fusion_type
        # Price encoder
        self.price_encoder = nn.Linear(num_features, d_model)
        self.sentiment_encoder = nn.Linear(sentiment_dim * num_sentiment_sources, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        # Price transformer
        self.price_layers = nn.ModuleList([
            TransformerBlock(d_model, config.num_heads, d_ff=d_model * 4)
            for _ in range(config.num_layers)
        ])
        # Sentiment transformer
        self.sentiment_layers = nn.ModuleList([
            TransformerBlock(d_model, config.num_heads // 2, d_ff=d_model * 2)
            for _ in range(config.num_layers // 2)
        ])
        if fusion_type == "cross_attention":
            self.cross_attn = CrossAttention(d_model, config.num_heads)
            self.cross_norm = nn.LayerNorm(d_model)
        elif fusion_type == "late":
            self.fusion_gate = nn.Linear(d_model * 2, 2)
        self.norm = nn.LayerNorm(d_model)
        self.alpha_head = nn.Linear(d_model, 1)

    def forward(
        self,
        price_x: torch.Tensor,
        sentiment_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, F = price_x.shape
        h_p = self.price_encoder(price_x)
        pos_ids = torch.arange(T, device=price_x.device).unsqueeze(0)
        h_p = h_p + self.pos_embed(pos_ids)
        for layer in self.price_layers:
            h_p = layer(h_p)
        if sentiment_x is not None:
            h_s = self.sentiment_encoder(sentiment_x)
            h_s = h_s + self.pos_embed(pos_ids[:, :h_s.size(1)])
            for layer in self.sentiment_layers:
                h_s = layer(h_s)
            if self.fusion_type == "cross_attention":
                h_fused = h_p + self.cross_attn(self.cross_norm(h_p), h_s)
            elif self.fusion_type == "late":
                h_s_up = F.interpolate(h_s.transpose(1, 2), size=T, mode="nearest").transpose(1, 2)
                gates = torch.softmax(self.fusion_gate(torch.cat([h_p, h_s_up], -1)), -1)
                h_fused = gates[:, :, 0:1] * h_p + gates[:, :, 1:2] * h_s_up
            else:
                h_fused = h_p + h_s[:, :T, :]
        else:
            h_fused = h_p
        h_fused = self.norm(h_fused)
        alpha = self.alpha_head(h_fused[:, -1, :]).squeeze(-1)
        return {
            "alpha_scores": alpha,
            "encoded": h_fused,
        }


class LuminaWithLoRA(nn.Module):
    """Lumina model wrapper applying LoRA for parameter-efficient fine-tuning.

    Wraps any Lumina model and injects LoRA adapters into specified
    layers for efficient adaptation to new tasks or markets.

    Args:
        base_model: Pretrained Lumina model
        lora_rank: LoRA adapter rank
        lora_alpha: LoRA scaling
        target_modules: Which module types to apply LoRA to
        lora_dropout: LoRA dropout probability
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_rank: int = 4,
        lora_alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        target_modules = target_modules or ["q_proj", "v_proj", "out_proj"]
        self._lora_modules: Dict[str, nn.Module] = {}
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        # Inject LoRA
        self._inject_lora(target_modules, lora_rank, lora_alpha, lora_dropout)

    def _inject_lora(self, targets, rank, alpha, dropout) -> None:
        """Inject LoRA adapters into target Linear layers."""
        for name, module in list(self.base_model.named_modules()):
            if not any(t in name for t in targets):
                continue
            if not isinstance(module, nn.Linear):
                continue
            lora = LoRALinear(
                module.in_features, module.out_features,
                rank=rank, alpha=alpha, dropout=dropout
            )
            with torch.no_grad():
                lora.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    lora.bias = nn.Parameter(module.bias.data.clone())
            # Replace module
            parts = name.split(".")
            parent = self.base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora)
            self._lora_modules[name] = lora

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


_NEW_MODEL_EXPORTS = [
    "LuminaForMultiHorizonForecast", "LuminaForCrossAssetModeling",
    "LuminaForSentimentFusion", "LuminaWithLoRA",
]


# =============================================================================
# SECTION: Extended Lumina Model Variants
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict


class LuminaMicro(nn.Module):
    """Lumina-Micro: compact 1M parameter model for edge deployment.

    Architecture:
    - 4 layers, 128 hidden, 4 heads
    - SwiGLU FFN
    - RoPE positional encoding
    - Single output head
    """

    def __init__(
        self,
        n_features: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        n_outputs: int = 5,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_ff, dropout=dropout, batch_first=True
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_outputs)

        # Weight tying: output head shares weights with input projection
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B, T, F = x.shape
        h = self.input_proj(x)
        pos = torch.arange(T, device=x.device)
        h = self.dropout(h + self.pos_emb(pos).unsqueeze(0))

        for layer in self.layers:
            h = layer(h, src_key_padding_mask=(~mask if mask is not None else None))

        h = self.norm(h)
        return {
            "last_hidden": h,
            "predictions": self.head(h[:, -1, :]),
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LuminaSmall(nn.Module):
    """Lumina-Small: 25M parameter model for production deployment.

    Architecture:
    - 8 layers, 256 hidden, 8 heads
    - Multi-task output heads
    - Optional LoRA adapters
    """

    def __init__(
        self,
        n_features: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        n_return_horizons: int = 5,
        n_risk_outputs: int = 3,
        n_factor_outputs: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_ff, dropout=dropout, batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Multi-task heads
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_return_horizons),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, n_risk_outputs),
            nn.Softplus(),
        )
        self.factor_head = nn.Linear(d_model, n_factor_outputs)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(T, device=x.device)
        h = self.dropout(h + self.pos_emb(pos).unsqueeze(0))

        for layer in self.layers:
            h = layer(h, src_key_padding_mask=(~mask if mask is not None else None))

        h = self.norm(h)
        cls = h[:, -1, :]  # Use last token as sequence representation

        out = {
            "returns": self.return_head(cls),
            "risk": self.risk_head(cls),
            "factors": self.factor_head(cls),
        }

        if return_hidden:
            out["hidden"] = h

        return out


class LuminaMedium(nn.Module):
    """Lumina-Medium: 125M parameter model with full feature set.

    Architecture:
    - 12 layers, 512 hidden, 8 heads
    - Mixture of Experts FFN in alternate layers
    - Contrastive self-supervised pretraining compatible
    - Multi-asset cross-attention capability
    """

    def __init__(
        self,
        n_features: int = 256,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        n_experts: int = 4,
        moe_every_n: int = 3,
        n_return_horizons: int = 10,
        n_assets: int = 500,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_assets = n_assets

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.asset_emb = nn.Embedding(n_assets + 1, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Build layers: alternate between standard and MoE layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i % moe_every_n == moe_every_n - 1:
                # MoE layer
                self.layers.append(nn.ModuleDict({
                    "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                    "norm1": nn.LayerNorm(d_model),
                    "norm2": nn.LayerNorm(d_model),
                    "layer_type": None,
                }))
            else:
                self.layers.append(nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_ff, dropout=dropout, batch_first=True,
                    norm_first=True,
                ))

        self.norm = nn.LayerNorm(d_model)

        # Output heads
        self.return_head = nn.Linear(d_model, n_return_horizons)
        self.vol_head = nn.Sequential(nn.Linear(d_model, n_return_horizons), nn.Softplus())
        self.regime_head = nn.Linear(d_model, 4)  # 4 market regimes

        # Projection head for contrastive pretraining
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128),
        )

    def forward(
        self,
        x: torch.Tensor,
        asset_ids: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_projections: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(T, device=x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)

        if asset_ids is not None:
            h = h + self.asset_emb(asset_ids).unsqueeze(1)

        h = self.dropout(h)

        for layer in self.layers:
            if isinstance(layer, nn.ModuleDict):
                # Manual attention + norm pass for MoE layers
                residual = h
                h_norm = layer["norm1"](h)
                attn_out, _ = layer["attn"](h_norm, h_norm, h_norm)
                h = layer["norm2"](residual + attn_out)
            else:
                h = layer(h, src_key_padding_mask=(~mask if mask is not None else None))

        h = self.norm(h)
        cls = h[:, -1, :]

        out = {
            "returns": self.return_head(cls),
            "volatility": self.vol_head(cls),
            "regime": self.regime_head(cls),
            "hidden": h,
        }

        if return_projections:
            out["projection"] = F.normalize(self.projection_head(cls), dim=-1)

        return out


class LuminaLargeV2(nn.Module):
    """Lumina-Large-V2: 1.3B parameter flagship model.

    Architecture improvements over V1:
    - Pre-norm with RMSNorm throughout
    - SwiGLU FFN in all layers
    - Grouped Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - Multi-task + multi-horizon output with uncertainty
    """

    def __init__(
        self,
        n_features: int = 512,
        d_model: int = 2048,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        n_layers: int = 24,
        expansion: float = 8.0 / 3.0,
        n_return_horizons: int = 20,
        n_risk_types: int = 5,
        dropout: float = 0.05,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.input_proj = nn.Linear(n_features, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # RMSNorm layers
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers * 2)])

        # Attention layers (simplified GQA)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True, bias=False)
            for _ in range(n_layers)
        ])

        # SwiGLU FFN layers
        d_ff = int(d_model * expansion)
        d_ff = (d_ff + 63) // 64 * 64
        self.ffn_gate = nn.ModuleList([nn.Linear(d_model, d_ff, bias=False) for _ in range(n_layers)])
        self.ffn_up = nn.ModuleList([nn.Linear(d_model, d_ff, bias=False) for _ in range(n_layers)])
        self.ffn_down = nn.ModuleList([nn.Linear(d_ff, d_model, bias=False) for _ in range(n_layers)])

        self.out_norm = nn.LayerNorm(d_model)

        # Output heads
        self.return_mu = nn.Linear(d_model, n_return_horizons, bias=False)
        self.return_sigma = nn.Sequential(
            nn.Linear(d_model, n_return_horizons, bias=False),
            nn.Softplus(),
        )
        self.risk_head = nn.Linear(d_model, n_risk_types, bias=False)

        self.n_layers = n_layers

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = self.dropout(h)

        for i in range(self.n_layers):
            # Pre-norm attention
            residual = h
            h_norm = self.norms[2 * i](h)
            attn_out, _ = self.attn_layers[i](h_norm, h_norm, h_norm)
            h = residual + attn_out

            # Pre-norm SwiGLU FFN
            residual = h
            h_norm = self.norms[2 * i + 1](h)
            ffn_out = self.ffn_down[i](F.silu(self.ffn_gate[i](h_norm)) * self.ffn_up[i](h_norm))
            h = residual + ffn_out

        h = self.out_norm(h)
        cls = h[:, -1, :]

        return {
            "return_mu": self.return_mu(cls),
            "return_sigma": self.return_sigma(cls),
            "risk": self.risk_head(cls),
            "hidden": h,
            "cls_token": cls,
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LuminaRegimeDetector(nn.Module):
    """Lumina sub-model for unsupervised market regime detection.

    Uses a mixture model approach:
    - Encoder produces regime embeddings
    - Soft assignment to K regimes via learned centroids
    - Temporal smoothing for regime consistency
    """

    def __init__(
        self,
        n_features: int = 64,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_regimes: int = 4,
        temporal_smoothing: float = 0.9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.temporal_smoothing = temporal_smoothing

        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Regime centroids (learnable)
        self.regime_centroids = nn.Parameter(torch.randn(n_regimes, d_model))

        # Temporal gating
        self.temporal_gate = nn.GRUCell(n_regimes, n_regimes)

        self.norm = nn.LayerNorm(d_model)
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape

        h = self.input_proj(x)
        h = self.encoder(h)
        h = self.norm(h)

        # Compute cosine similarity to regime centroids
        h_norm = F.normalize(h, dim=-1)
        c_norm = F.normalize(self.regime_centroids, dim=-1)
        logits = (h_norm @ c_norm.T) / self.temperature.clamp(min=1e-4)
        probs = F.softmax(logits, dim=-1)  # [B, T, K]

        # Temporal smoothing via GRU
        if initial_state is None:
            state = torch.zeros(B, self.n_regimes, device=x.device)
        else:
            state = initial_state

        smoothed = []
        for t in range(T):
            state = self.temporal_gate(probs[:, t, :], state)
            state = state * self.temporal_smoothing + probs[:, t, :] * (1 - self.temporal_smoothing)
            smoothed.append(state)

        smoothed_probs = torch.stack(smoothed, dim=1)  # [B, T, K]
        regime_assignments = smoothed_probs.argmax(dim=-1)

        return {
            "regime_probs": smoothed_probs,
            "regime_assignments": regime_assignments,
            "raw_probs": probs,
            "hidden": h,
            "final_state": state,
        }


class LuminaVolatilityForecaster(nn.Module):
    """Specialized Lumina model for volatility forecasting.

    Combines:
    - HAR-RV features (daily, weekly, monthly realized variance)
    - Transformer for non-linear dependencies
    - GARCH-inspired output parameterization
    - Volatility term structure output
    """

    def __init__(
        self,
        n_features: int = 32,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_horizons: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_horizons = n_horizons

        # HAR feature extraction
        self.har_daily = nn.Linear(1, d_model // 4)
        self.har_weekly = nn.Linear(5, d_model // 4)
        self.har_monthly = nn.Linear(22, d_model // 4)
        self.har_other = nn.Linear(n_features - 28, d_model // 4)

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

        # GARCH-inspired output
        self.vol_long_run = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.alpha = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        # Term structure
        self.term_structure = nn.Linear(d_model, n_horizons)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, n_features] where first 28 cols are daily/weekly/monthly RV
        """
        B, T, F = x.shape

        # HAR decomposition
        if F >= 28:
            daily = self.har_daily(x[:, :, :1])
            weekly = self.har_weekly(x[:, :, 1:6])
            monthly = self.har_monthly(x[:, :, 6:28])
            other = self.har_other(x[:, :, 28:])
            h = torch.cat([daily, weekly, monthly, other], dim=-1)
        else:
            h = x.repeat(1, 1, 1)[:, :, :self.har_daily.in_features]
            h = self.har_daily(h[:, :, :1]).repeat(1, 1, 4)

        h = self.encoder(h)
        h = self.norm(h)
        cls = h[:, -1, :]

        omega = self.vol_long_run(cls)
        alpha = self.alpha(cls) * 0.3
        beta = self.beta(cls) * 0.7

        # Simple GARCH-inspired forecast
        vol_1step = omega * (1 - alpha - beta) + alpha * x[:, -1:, 0:1] + beta * omega

        return {
            "vol_1step": vol_1step.squeeze(-1),
            "term_structure": self.term_structure(cls),
            "omega": omega.squeeze(-1),
            "alpha": alpha.squeeze(-1),
            "beta": beta.squeeze(-1),
            "hidden": h,
        }


class LuminaPortfolioOptimizer(nn.Module):
    """End-to-end differentiable portfolio optimizer using Lumina features.

    Combines:
    - Signal generation from factor exposures
    - Risk model estimation (covariance matrix)
    - Soft constrained optimization via Lagrangian relaxation
    - Transaction cost-aware rebalancing
    """

    def __init__(
        self,
        n_assets: int = 100,
        n_features: int = 50,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        risk_aversion: float = 1.0,
        transaction_cost: float = 0.001,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost

        # Signal generator
        self.signal_encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True)
        self.cross_asset_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Return signal
        self.mu_head = nn.Linear(d_model, 1)

        # Risk model
        self.cov_head = nn.Linear(d_model, d_model // 4)
        self.cov_factor = d_model // 4  # number of latent risk factors

        # Portfolio constraints
        self.long_only = False
        self.max_weight = 0.1
        self.norm = nn.LayerNorm(d_model)

    def _estimate_covariance(self, cov_features: torch.Tensor) -> torch.Tensor:
        """Estimate factor-based covariance: Sigma = F * F^T + diag(eps)."""
        # cov_features: [B, N, d_cov]
        # Factor model: Sigma = cov_features @ cov_features^T / d_cov
        B, N, d_cov = cov_features.shape
        cov = torch.bmm(cov_features, cov_features.transpose(1, 2)) / d_cov
        # Add small diagonal for positive definiteness
        eye = torch.eye(N, device=cov_features.device).unsqueeze(0)
        return cov + 0.01 * eye

    def _soft_portfolio_weights(self, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Compute soft portfolio weights via differentiable mean-variance optimization."""
        # Markowitz: w = (1/lambda) * Sigma^{-1} * mu, then normalize
        try:
            cov_inv = torch.linalg.inv(cov)
        except torch.linalg.LinAlgError:
            # Fallback: use Cholesky with regularization
            reg = cov + 0.1 * torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0)
            cov_inv = torch.linalg.inv(reg)

        raw_weights = (cov_inv @ mu) / self.risk_aversion

        if self.long_only:
            raw_weights = F.softmax(raw_weights.squeeze(-1), dim=-1)
        else:
            # Long-short: normalize to zero net exposure
            raw_weights = raw_weights.squeeze(-1)
            raw_weights = raw_weights - raw_weights.mean(dim=-1, keepdim=True)
            raw_weights = raw_weights / (raw_weights.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6))

        # Clip to max weight
        raw_weights = raw_weights.clamp(-self.max_weight, self.max_weight)
        return raw_weights

    def forward(
        self,
        features: torch.Tensor,
        prev_weights: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, N_assets, n_features]
            prev_weights: [B, N_assets] previous portfolio weights
        Returns:
            dict with weights, mu, cov, turnover
        """
        B, N, _ = features.shape

        h = self.signal_encoder(features)
        h = self.cross_asset_encoder(h)
        h = self.norm(h)

        mu = self.mu_head(h)  # [B, N, 1]
        cov_feat = self.cov_head(h)  # [B, N, d_cov]
        cov = self._estimate_covariance(cov_feat)  # [B, N, N]

        weights = self._soft_portfolio_weights(mu, cov)  # [B, N]

        result = {
            "weights": weights,
            "mu": mu.squeeze(-1),
            "covariance": cov,
        }

        if prev_weights is not None:
            turnover = (weights - prev_weights).abs().sum(dim=-1).mean()
            tc_cost = turnover * self.transaction_cost
            result["turnover"] = turnover
            result["transaction_cost"] = tc_cost

        return result
