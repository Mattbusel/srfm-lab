"""
Lumina — Financial Multi-Modal Foundation Model
Module 5 of Project AETERNUS, SRFM-lab

Lumina is a transformer-based foundation model designed to ingest and reason
over heterogeneous financial data streams: price/OHLCV time series, limit
order book snapshots, on-chain DeFi signals, and financial news text.

Architecture highlights:
  - Patch-based + quantized hybrid tokenization for price series
  - Rotary positional encoding (RoPE) + temporal sinusoidal features
  - Grouped-query attention (GQA) for efficient inference
  - SwiGLU feed-forward networks
  - Optional Mixture-of-Experts (MoE) layers
  - Cross-modal attention for multi-modal fusion
  - Pre-training objectives: MRM, NBP, contrastive, regime prediction
  - Fine-tuning with LoRA for parameter-efficient adaptation
"""

from .tokenizer import (
    PriceSeriesTokenizer,
    OrderBookTokenizer,
    OnChainTokenizer,
    NewsTokenizer,
    MultiModalTokenizer,
)
from .positional_encoding import (
    RotaryPositionalEncoding,
    ALiBiPositionalBias,
    TemporalEncoding,
    FourierTimeEncoding,
    CrossModalPositionalEncoding,
)
from .transformer import (
    MultiHeadSelfAttention,
    GroupedQueryAttention,
    SwiGLUFFN,
    RMSNorm,
    TransformerBlock,
    CausalTransformer,
    BidirectionalTransformer,
    MixtureOfExpertsLayer,
    LuminaModel,
)
from .multimodal import (
    CrossModalAttention,
    ModalityFusion,
    TemporalAlignment,
    MultiModalLumina,
)
from .inference import LuminaInference
from .evaluation import (
    crisis_detection_benchmark,
    volatility_forecast_benchmark,
    return_direction_benchmark,
    perplexity,
)

__version__ = "0.1.0"
__author__ = "SRFM-lab"
__all__ = [
    "PriceSeriesTokenizer",
    "OrderBookTokenizer",
    "OnChainTokenizer",
    "NewsTokenizer",
    "MultiModalTokenizer",
    "RotaryPositionalEncoding",
    "ALiBiPositionalBias",
    "TemporalEncoding",
    "FourierTimeEncoding",
    "CrossModalPositionalEncoding",
    "MultiHeadSelfAttention",
    "GroupedQueryAttention",
    "SwiGLUFFN",
    "RMSNorm",
    "TransformerBlock",
    "CausalTransformer",
    "BidirectionalTransformer",
    "MixtureOfExpertsLayer",
    "LuminaModel",
    "CrossModalAttention",
    "ModalityFusion",
    "TemporalAlignment",
    "MultiModalLumina",
    "LuminaInference",
    "crisis_detection_benchmark",
    "volatility_forecast_benchmark",
    "return_direction_benchmark",
    "perplexity",
]
