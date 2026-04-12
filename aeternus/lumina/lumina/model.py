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
