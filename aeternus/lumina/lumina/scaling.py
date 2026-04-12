"""
lumina/scaling.py

Model scaling utilities for Lumina financial foundation model.

Covers:
  - Parameter count estimation (theoretical and empirical)
  - FLOPs estimation for transformer forward pass
  - Chinchilla scaling law calculator (optimal tokens for model size)
  - Compute-optimal model size for a given FLOP budget
  - Hyperparameter scaling rules (LR, batch size, warmup)
  - Training efficiency metrics (MFU, HFU)
  - Model card generation
  - Scaling experiment planner
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model architecture parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Describes the architecture of a transformer-style model."""
    vocab_size: int = 50_000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None   # GQA if != n_heads
    ffn_dim_multiplier: float = 4.0
    use_moe: bool = False
    n_experts: int = 8
    n_active_experts: int = 2
    use_rope: bool = True
    max_seq_len: int = 2048
    tie_word_embeddings: bool = True
    use_bias: bool = False
    activation: str = "swiglu"          # "gelu" | "relu" | "swiglu"
    norm_type: str = "rmsnorm"          # "layernorm" | "rmsnorm"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        """Actual FFN hidden dim, possibly adjusted for SwiGLU (2/3 factor)."""
        base = int(self.d_model * self.ffn_dim_multiplier)
        if self.activation == "swiglu":
            # SwiGLU projects to 2/3 of the nominal to keep param count equal
            base = int(base * 2 / 3)
            # Round to multiple of 64 for efficiency
            base = ((base + 63) // 64) * 64
        return base

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads


# ---------------------------------------------------------------------------
# Parameter count estimation
# ---------------------------------------------------------------------------

class ParameterCounter:
    """
    Estimates parameter counts for transformer models analytically.
    Matches empirical counts from instantiated models.
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config

    def embedding_params(self) -> int:
        """Token embedding + position embedding (if learned)."""
        token_embed = self.cfg.vocab_size * self.cfg.d_model
        # RoPE has no learnable position params; learned PE does
        pos_embed = 0 if self.cfg.use_rope else self.cfg.max_seq_len * self.cfg.d_model
        return token_embed + pos_embed

    def attention_params_per_layer(self) -> int:
        """Q, K, V, O projection parameters for one attention layer."""
        d = self.cfg.d_model
        h = self.cfg.n_heads
        kv = self.cfg.kv_heads
        head_dim = self.cfg.head_dim

        q_params = d * (h * head_dim)
        k_params = d * (kv * head_dim)
        v_params = d * (kv * head_dim)
        o_params = (h * head_dim) * d

        bias_params = 0
        if self.cfg.use_bias:
            bias_params = (h + kv + kv) * head_dim + d

        return q_params + k_params + v_params + o_params + bias_params

    def ffn_params_per_layer(self) -> int:
        """FFN parameters for one layer."""
        d = self.cfg.d_model
        f = self.cfg.ffn_dim

        if self.cfg.activation == "swiglu":
            # SwiGLU: W_gate, W_up, W_down
            params = d * f + d * f + f * d
        else:
            # Standard 2-layer MLP
            params = d * f + f * d

        if self.cfg.use_bias:
            params += f + d

        return params

    def moe_params_per_layer(self) -> int:
        """MoE layer: n_experts * ffn_params + router params."""
        router = self.cfg.d_model * self.cfg.n_experts
        experts = self.cfg.n_experts * self.ffn_params_per_layer()
        return router + experts

    def norm_params_per_layer(self) -> int:
        """Pre-norm (2 norms per layer: before attn and before ffn)."""
        if self.cfg.norm_type in ("rmsnorm", "layernorm"):
            return 2 * self.cfg.d_model
        return 0

    def transformer_block_params(self) -> int:
        """Total params for one transformer block."""
        attn = self.attention_params_per_layer()
        if self.cfg.use_moe:
            ffn = self.moe_params_per_layer()
        else:
            ffn = self.ffn_params_per_layer()
        norm = self.norm_params_per_layer()
        return attn + ffn + norm

    def output_head_params(self) -> int:
        """LM head (unembedding) params."""
        if self.cfg.tie_word_embeddings:
            return 0  # Tied to token embeddings
        return self.cfg.d_model * self.cfg.vocab_size

    def total_params(self) -> Dict[str, int]:
        embed = self.embedding_params()
        per_block = self.transformer_block_params()
        total_blocks = per_block * self.cfg.n_layers
        lm_head = self.output_head_params()
        # Final norm
        final_norm = self.cfg.d_model

        total = embed + total_blocks + lm_head + final_norm
        return {
            "embedding": embed,
            "per_block": per_block,
            "all_blocks": total_blocks,
            "attention_all": self.attention_params_per_layer() * self.cfg.n_layers,
            "ffn_all": (self.moe_params_per_layer() if self.cfg.use_moe
                        else self.ffn_params_per_layer()) * self.cfg.n_layers,
            "lm_head": lm_head,
            "final_norm": final_norm,
            "total": total,
            "total_B": total / 1e9,
            "total_M": total / 1e6,
        }

    def count_empirical(self, model: nn.Module, trainable_only: bool = False) -> int:
        """Count params from an actual instantiated model."""
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())

    def compare(self, model: nn.Module) -> Dict[str, Any]:
        """Compare analytical estimate vs empirical count."""
        theoretical = self.total_params()["total"]
        empirical = self.count_empirical(model)
        diff = empirical - theoretical
        return {
            "theoretical": theoretical,
            "empirical": empirical,
            "difference": diff,
            "difference_pct": diff / theoretical * 100 if theoretical > 0 else 0,
        }


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

class FLOPsEstimator:
    """
    Estimates FLOPs for a transformer forward pass.

    Uses the standard formula:
      FLOPs ≈ 6 * N * D  (N = params, D = tokens, factor 6 = 2 fwd + 4 bwd)

    More detailed per-operation breakdown also available.
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.param_counter = ParameterCounter(config)

    def attention_flops_per_token(self) -> int:
        """FLOPs for attention computation for a single token in a sequence."""
        d = self.cfg.d_model
        h = self.cfg.n_heads
        kv = self.cfg.kv_heads
        head_dim = self.cfg.head_dim
        seq = self.cfg.max_seq_len

        # QKV projections: 2 * d * (h + kv + kv) * head_dim
        qkv_flops = 2 * d * (h + 2 * kv) * head_dim
        # Attention scores: 2 * seq * h * head_dim
        score_flops = 2 * seq * h * head_dim
        # Weighted sum: 2 * seq * h * head_dim
        wsum_flops = 2 * seq * h * head_dim
        # Output projection: 2 * d * d
        out_flops = 2 * d * d
        return qkv_flops + score_flops + wsum_flops + out_flops

    def ffn_flops_per_token(self) -> int:
        """FLOPs for FFN for a single token."""
        d = self.cfg.d_model
        f = self.cfg.ffn_dim
        if self.cfg.activation == "swiglu":
            return 2 * d * f + 2 * d * f + 2 * f * d  # gate + up + down
        return 2 * d * f + 2 * f * d

    def total_flops_forward(self, seq_len: int, batch_size: int = 1) -> int:
        """
        Total FLOPs for one forward pass.
        Approximately 2 * N * T where N = non-embedding params, T = seq_len.
        """
        L = self.cfg.n_layers
        per_token_attn = self.attention_flops_per_token()
        per_token_ffn = self.ffn_flops_per_token()
        per_token = (per_token_attn + per_token_ffn) * L
        # Embedding lookup: negligible
        return per_token * seq_len * batch_size

    def total_flops_training_step(self, seq_len: int, batch_size: int = 1) -> int:
        """
        FLOPs for one training step (forward + backward ≈ 3x forward).
        """
        return 3 * self.total_flops_forward(seq_len, batch_size)

    def chinchilla_flops(self, n_params: int, n_tokens: int) -> int:
        """
        FLOPs estimate using Chinchilla approximation:
        C ≈ 6 * N * D
        """
        return 6 * n_params * n_tokens

    def throughput(
        self,
        tokens_per_second: float,
        seq_len: int,
    ) -> Dict[str, float]:
        """Compute MFU and HFU given measured throughput."""
        forward_flops_per_token = self.total_flops_forward(seq_len) / seq_len
        # Per-step FLOPs: forward + backward
        training_flops_per_token = forward_flops_per_token * 3

        # Hardware peak FLOPs (A100 bfloat16: ~312 TFLOPS)
        gpu_peak_flops = 312e12  # A100 BF16

        mfu = tokens_per_second * training_flops_per_token / gpu_peak_flops
        return {
            "forward_flops_per_token": forward_flops_per_token,
            "training_flops_per_token": training_flops_per_token,
            "mfu": mfu,
            "mfu_pct": mfu * 100,
        }

    def estimate_training_cost(
        self,
        n_tokens: int,
        gpu_flops: float = 312e12,     # A100 BF16 peak
        mfu: float = 0.40,             # Typical 40% utilization
        gpu_price_per_hour: float = 2.0,
    ) -> Dict[str, float]:
        """Estimate training cost in GPU-hours and dollars."""
        n_params = self.param_counter.total_params()["total"]
        total_flops = self.chinchilla_flops(n_params, n_tokens)
        effective_flops_per_sec = gpu_flops * mfu
        gpu_seconds = total_flops / effective_flops_per_sec
        gpu_hours = gpu_seconds / 3600
        cost = gpu_hours * gpu_price_per_hour

        return {
            "total_flops": total_flops,
            "total_flops_PF": total_flops / 1e15,
            "gpu_hours": gpu_hours,
            "gpu_days": gpu_hours / 24,
            "estimated_cost_usd": cost,
        }


# ---------------------------------------------------------------------------
# Chinchilla scaling laws
# ---------------------------------------------------------------------------

@dataclass
class ChinchillaResult:
    """Result of Chinchilla compute-optimal analysis."""
    n_params: int            # Optimal number of parameters
    n_tokens: int            # Optimal number of training tokens
    compute_flops: float     # Total FLOPs = 6 * N * D
    tokens_per_param: float  # ≈ 20 for Chinchilla-optimal
    model_config: Optional[ModelConfig] = None


class ChinchillaScalingLaw:
    """
    Implements Chinchilla scaling laws (Hoffmann et al., 2022).

    Key finding: For compute-optimal training,
      N_opt(C) = (C / (6 * 20)) ^ 0.5
      D_opt(C) = 20 * N_opt

    More precisely (from the paper's fit):
      N_opt ≈ 0.1715 * C ^ 0.4945
      D_opt ≈ 0.2990 * C ^ 0.5055

    Also implements:
      - Neural scaling law loss prediction
      - Extrapolation to larger compute budgets
    """

    # Chinchilla fit constants (from Table 3, Hoffmann et al.)
    # L(N, D) = E + A/N^alpha + B/D^beta
    E: float = 1.69     # Irreducible loss
    A: float = 406.4
    B: float = 410.7
    ALPHA: float = 0.34
    BETA: float = 0.28

    # Compute-optimal scaling exponents
    N_COEFF: float = 0.1715
    N_EXPONENT: float = 0.4945
    D_COEFF: float = 0.2990
    D_EXPONENT: float = 0.5055

    @classmethod
    def optimal_for_compute(cls, compute_flops: float) -> ChinchillaResult:
        """
        Given a compute budget C (FLOPs), return compute-optimal N and D.
        """
        n_opt = cls.N_COEFF * (compute_flops ** cls.N_EXPONENT)
        d_opt = cls.D_COEFF * (compute_flops ** cls.D_EXPONENT)
        return ChinchillaResult(
            n_params=int(n_opt),
            n_tokens=int(d_opt),
            compute_flops=compute_flops,
            tokens_per_param=d_opt / n_opt,
        )

    @classmethod
    def optimal_for_params(cls, n_params: int) -> ChinchillaResult:
        """
        Given a fixed parameter count, find optimal token count and compute budget.
        Using the simple ≈20 tokens/param rule.
        """
        n_tokens = 20 * n_params
        compute = 6 * n_params * n_tokens
        return ChinchillaResult(
            n_params=n_params,
            n_tokens=n_tokens,
            compute_flops=compute,
            tokens_per_param=20.0,
        )

    @classmethod
    def predict_loss(cls, n_params: int, n_tokens: int) -> float:
        """
        Predict cross-entropy loss given model size and training tokens.
        L(N, D) = E + A/N^alpha + B/D^beta
        """
        return cls.E + cls.A / (n_params ** cls.ALPHA) + cls.B / (n_tokens ** cls.BETA)

    @classmethod
    def loss_curve(
        cls,
        n_params: int,
        token_counts: List[int],
    ) -> List[float]:
        """Predict loss at multiple token counts for a fixed model size."""
        return [cls.predict_loss(n_params, d) for d in token_counts]

    @classmethod
    def compute_efficient_frontier(
        cls,
        compute_budgets: List[float],
    ) -> List[ChinchillaResult]:
        """Compute the Pareto frontier of (N, D) for given compute budgets."""
        return [cls.optimal_for_compute(c) for c in compute_budgets]

    @classmethod
    def tokens_to_match_loss(cls, target_loss: float, n_params: int) -> int:
        """
        How many tokens are needed to reach target_loss with n_params?
        Solve: target_loss = E + A/N^alpha + B/D^beta for D.
        """
        residual = target_loss - cls.E - cls.A / (n_params ** cls.ALPHA)
        if residual <= 0:
            return int(1e15)  # Practically impossible
        d = (cls.B / residual) ** (1.0 / cls.BETA)
        return int(d)

    @classmethod
    def iso_flop_curve(
        cls,
        compute_flops: float,
        n_param_range: Tuple[int, int],
        n_points: int = 50,
    ) -> List[Tuple[int, int, float]]:
        """
        Compute loss for all (N, D) pairs on the iso-FLOPs curve C = 6*N*D.
        Returns list of (n_params, n_tokens, predicted_loss).
        """
        results = []
        n_values = np.logspace(
            math.log10(n_param_range[0]),
            math.log10(n_param_range[1]),
            n_points,
        )
        for n in n_values:
            d = compute_flops / (6 * n)
            if d < 1:
                continue
            loss = cls.predict_loss(int(n), int(d))
            results.append((int(n), int(d), loss))
        return results


# ---------------------------------------------------------------------------
# Hyperparameter scaling rules
# ---------------------------------------------------------------------------

class HyperparameterScaler:
    """
    Scales hyperparameters as model/batch size changes.

    Based on:
      - Linear scaling rule (Goyal et al., 2017): LR ∝ batch_size
      - Square root scaling rule: LR ∝ sqrt(batch_size)
      - µP (maximal update parameterization) transfer rules
      - Warmup: warmup_steps ∝ sqrt(n_params)
    """

    @staticmethod
    def scale_lr_linear(
        base_lr: float,
        base_batch: int,
        new_batch: int,
    ) -> float:
        """Linear scaling rule: scale LR proportionally to batch size."""
        return base_lr * (new_batch / base_batch)

    @staticmethod
    def scale_lr_sqrt(
        base_lr: float,
        base_batch: int,
        new_batch: int,
    ) -> float:
        """Square root scaling rule."""
        return base_lr * math.sqrt(new_batch / base_batch)

    @staticmethod
    def warmup_steps(
        n_params: int,
        base_params: int = 117_000_000,   # GPT-2 small
        base_warmup: int = 2000,
    ) -> int:
        """Scale warmup steps as sqrt of parameter count ratio."""
        ratio = n_params / base_params
        return int(base_warmup * math.sqrt(ratio))

    @staticmethod
    def learning_rate_mup(
        d_model: int,
        base_d_model: int = 256,
        base_lr: float = 3e-4,
    ) -> float:
        """
        µP (maximal update parameterization) learning rate scaling.
        LR ∝ 1/d_model (for attention and ffn layers).
        """
        return base_lr * (base_d_model / d_model)

    @staticmethod
    def optimal_batch_size(n_params: int) -> int:
        """
        Heuristic optimal batch size (in tokens) for a given parameter count.
        From empirical observations across large model training runs.
        """
        # Roughly: batch_tokens ≈ sqrt(n_params) * some_constant
        return int(math.sqrt(n_params) * 4)

    @staticmethod
    def optimal_lr(n_params: int, batch_size_tokens: int) -> float:
        """
        Heuristic: optimal peak LR ≈ 0.003 / (N/1e9)^0.1
        Approximate fit to published training runs.
        """
        n_b = n_params / 1e9  # in billions
        scale = 0.003 / (max(n_b, 1e-3) ** 0.1)
        # Further adjust for batch size
        scale *= math.sqrt(batch_size_tokens / 1_000_000)
        return min(scale, 0.01)  # Cap at 1%

    @staticmethod
    def suggest_hyperparameters(config: ModelConfig) -> Dict[str, Any]:
        """
        Suggest a full set of hyperparameters for a given model config.
        """
        pc = ParameterCounter(config)
        total_params = pc.total_params()["total"]

        batch_tokens = HyperparameterScaler.optimal_batch_size(total_params)
        # Typical batch: 2M tokens = 256 sequences * 8192 tokens
        seq_len = config.max_seq_len
        batch_size = max(1, batch_tokens // seq_len)

        lr = HyperparameterScaler.optimal_lr(total_params, batch_tokens)
        warmup = HyperparameterScaler.warmup_steps(total_params)

        chinchilla = ChinchillaScalingLaw.optimal_for_params(total_params)

        return {
            "model_params": total_params,
            "model_params_B": total_params / 1e9,
            "recommended_batch_size": batch_size,
            "recommended_batch_tokens": batch_tokens,
            "recommended_peak_lr": lr,
            "recommended_warmup_steps": warmup,
            "recommended_weight_decay": 0.1,
            "recommended_beta1": 0.9,
            "recommended_beta2": 0.95,
            "recommended_grad_clip": 1.0,
            "chinchilla_optimal_tokens": chinchilla.n_tokens,
            "chinchilla_tokens_per_param": chinchilla.tokens_per_param,
            "predicted_final_loss": ChinchillaScalingLaw.predict_loss(
                total_params, chinchilla.n_tokens
            ),
        }


# ---------------------------------------------------------------------------
# Model architecture family scaling
# ---------------------------------------------------------------------------

_MODEL_FAMILIES: Dict[str, List[ModelConfig]] = {
    "lumina": [
        ModelConfig(d_model=256,  n_layers=6,  n_heads=4,  ffn_dim_multiplier=4.0),   # 20M
        ModelConfig(d_model=512,  n_layers=8,  n_heads=8,  ffn_dim_multiplier=4.0),   # 85M
        ModelConfig(d_model=768,  n_layers=12, n_heads=12, ffn_dim_multiplier=4.0),   # 117M (GPT-2)
        ModelConfig(d_model=1024, n_layers=24, n_heads=16, ffn_dim_multiplier=4.0),   # 345M
        ModelConfig(d_model=1280, n_layers=36, n_heads=20, ffn_dim_multiplier=4.0),   # 760M
        ModelConfig(d_model=1600, n_layers=48, n_heads=25, ffn_dim_multiplier=4.0),   # 1.5B
        ModelConfig(d_model=2048, n_layers=24, n_heads=16, ffn_dim_multiplier=4.0),   # 3B (shallow+wide)
        ModelConfig(d_model=4096, n_layers=32, n_heads=32, ffn_dim_multiplier=4.0),   # 7B (LLaMA-style)
        ModelConfig(d_model=4096, n_layers=48, n_heads=32, n_kv_heads=8, ffn_dim_multiplier=4.0),  # 13B GQA
    ],
}


def get_model_family_scaling(family: str = "lumina") -> List[Dict[str, Any]]:
    """Return scaling analysis for a family of model configs."""
    configs = _MODEL_FAMILIES.get(family, [])
    results = []
    for cfg in configs:
        pc = ParameterCounter(cfg)
        fe = FLOPsEstimator(cfg)
        params = pc.total_params()
        flops = fe.total_flops_forward(cfg.max_seq_len)
        hps = HyperparameterScaler.suggest_hyperparameters(cfg)
        results.append({
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "params_M": params["total_M"],
            "params_B": params["total_B"],
            "forward_flops": flops,
            "forward_flops_T": flops / 1e12,
            **hps,
        })
    return results


# ---------------------------------------------------------------------------
# Training efficiency metrics
# ---------------------------------------------------------------------------

class TrainingEfficiencyMonitor:
    """
    Monitors and reports training efficiency metrics:
      - Model FLOPs Utilization (MFU)
      - Hardware FLOPs Utilization (HFU)
      - Tokens per second
      - Gradient norm trends
    """

    # Peak FLOPs for various GPUs (BF16/FP16)
    GPU_PEAK_FLOPS: Dict[str, float] = {
        "A100_80GB_BF16": 312e12,
        "A100_40GB_BF16": 312e12,
        "H100_BF16": 989e12,
        "H100_FP8": 1979e12,
        "A6000_BF16": 154e12,
        "V100_FP16": 112e12,
        "RTX4090_BF16": 330e12,
        "RTX3090_BF16": 71e12,
    }

    def __init__(self, config: ModelConfig, gpu_type: str = "A100_80GB_BF16"):
        self.config = config
        self.gpu_type = gpu_type
        self.gpu_peak = self.GPU_PEAK_FLOPS.get(gpu_type, 312e12)
        self.flops_estimator = FLOPsEstimator(config)
        self._step_times: List[float] = []
        self._tokens_per_step: List[int] = []

    def record_step(self, step_time_sec: float, tokens_in_batch: int) -> None:
        self._step_times.append(step_time_sec)
        self._tokens_per_step.append(tokens_in_batch)

    def compute_mfu(self, seq_len: int) -> float:
        """Model FLOPs Utilization."""
        if not self._step_times:
            return 0.0
        avg_time = sum(self._step_times) / len(self._step_times)
        avg_tokens = sum(self._tokens_per_step) / len(self._tokens_per_step)
        tokens_per_sec = avg_tokens / avg_time
        info = self.flops_estimator.throughput(tokens_per_sec, seq_len)
        return info["mfu"]

    def summary(self, seq_len: int) -> Dict[str, float]:
        if not self._step_times:
            return {}
        avg_time = sum(self._step_times) / len(self._step_times)
        avg_tokens = sum(self._tokens_per_step) / len(self._tokens_per_step)
        tokens_per_sec = avg_tokens / avg_time

        training_flops_per_token = (
            self.flops_estimator.total_flops_training_step(seq_len) / seq_len
        )
        mfu = tokens_per_sec * training_flops_per_token / self.gpu_peak

        return {
            "avg_step_time_sec": avg_time,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_day": tokens_per_sec * 86400,
            "mfu": mfu,
            "mfu_pct": mfu * 100,
            "gpu_type": self.gpu_type,
            "gpu_peak_flops_T": self.gpu_peak / 1e12,
        }


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """Structured model card following Hugging Face conventions."""
    model_name: str
    model_type: str = "financial_foundation"
    architecture: str = "decoder-only transformer"
    n_params: int = 0
    training_data: str = "Financial time series (OHLCV, LOB, text)"
    training_tokens: int = 0
    context_length: int = 2048
    vocab_size: int = 50_000
    languages: List[str] = field(default_factory=lambda: ["en"])
    license: str = "MIT"
    authors: List[str] = field(default_factory=list)
    base_model: Optional[str] = None
    finetuned_from: Optional[str] = None
    hardware: str = "NVIDIA A100"
    training_time_gpu_days: float = 0.0
    intended_uses: List[str] = field(default_factory=lambda: [
        "Financial time series prediction",
        "Market regime classification",
        "Volatility forecasting",
        "Portfolio optimization signals",
    ])
    limitations: List[str] = field(default_factory=lambda: [
        "Not financial advice",
        "Performance degrades in unprecedented market regimes",
        "Requires domain-specific fine-tuning for production use",
    ])
    evaluation_results: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_config: Optional[ModelConfig] = None

    def to_markdown(self) -> str:
        """Generate model card as Markdown string."""
        lines = [
            f"# {self.model_name}",
            "",
            "## Model Description",
            f"- **Architecture**: {self.architecture}",
            f"- **Type**: {self.model_type}",
            f"- **Parameters**: {self.n_params:,} ({self.n_params/1e9:.2f}B)",
            f"- **Context Length**: {self.context_length}",
            f"- **Vocabulary Size**: {self.vocab_size:,}",
            f"- **License**: {self.license}",
            "",
            "## Training",
            f"- **Training Data**: {self.training_data}",
            f"- **Training Tokens**: {self.training_tokens:,}",
            f"- **Hardware**: {self.hardware}",
            f"- **Training Time**: {self.training_time_gpu_days:.1f} GPU-days",
            "",
        ]

        if self.model_config:
            pc = ParameterCounter(self.model_config)
            params = pc.total_params()
            lines.extend([
                "## Architecture Details",
                f"| Parameter | Value |",
                f"|-----------|-------|",
                f"| d_model | {self.model_config.d_model} |",
                f"| n_layers | {self.model_config.n_layers} |",
                f"| n_heads | {self.model_config.n_heads} |",
                f"| FFN dim | {self.model_config.ffn_dim} |",
                f"| Activation | {self.model_config.activation} |",
                f"| Norm | {self.model_config.norm_type} |",
                f"| Use RoPE | {self.model_config.use_rope} |",
                f"| Use MoE | {self.model_config.use_moe} |",
                "",
            ])

        lines.extend([
            "## Hyperparameters",
        ])
        for k, v in self.hyperparameters.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

        lines.extend([
            "## Evaluation Results",
        ])
        for metric, value in self.evaluation_results.items():
            lines.append(f"- **{metric}**: {value:.4f}")
        lines.append("")

        lines.extend([
            "## Intended Uses",
        ])
        for use in self.intended_uses:
            lines.append(f"- {use}")
        lines.append("")

        lines.extend([
            "## Limitations",
        ])
        for lim in self.limitations:
            lines.append(f"- {lim}")

        if self.authors:
            lines.extend(["", "## Authors", ", ".join(self.authors)])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.model_config:
            d["model_config"] = asdict(self.model_config)
        return d

    def save(self, path: Union[str, pathlib.Path]) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".md":
            path.write_text(self.to_markdown())
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_config(cls, config: ModelConfig, name: str = "lumina-base") -> "ModelCard":
        pc = ParameterCounter(config)
        params = pc.total_params()
        hps = HyperparameterScaler.suggest_hyperparameters(config)
        return cls(
            model_name=name,
            n_params=params["total"],
            context_length=config.max_seq_len,
            vocab_size=config.vocab_size,
            model_config=config,
            hyperparameters={
                "peak_lr": hps["recommended_peak_lr"],
                "batch_size": hps["recommended_batch_size"],
                "warmup_steps": hps["recommended_warmup_steps"],
                "weight_decay": hps["recommended_weight_decay"],
                "optimizer": "AdamW",
            },
        )


# ---------------------------------------------------------------------------
# Scaling experiment planner
# ---------------------------------------------------------------------------

class ScalingExperimentPlanner:
    """
    Plans a series of scaling experiments to verify scaling laws
    before committing to a large training run.

    Strategy:
      1. Train small models to verify the loss-vs-compute relationship
      2. Extrapolate to predict large model performance
      3. Recommend optimal compute allocation
    """

    def __init__(
        self,
        compute_budget_flops: float,
        min_model_params: int = 10_000_000,
        n_pilot_runs: int = 5,
    ):
        self.compute_budget = compute_budget_flops
        self.min_params = min_model_params
        self.n_pilots = n_pilot_runs

    def suggest_pilot_runs(self) -> List[Dict[str, Any]]:
        """
        Suggest a series of pilot training runs to measure scaling behavior.
        Each run uses ~1/n_pilots^2 of the total compute budget.
        """
        pilot_computes = [
            self.compute_budget / (10 ** (self.n_pilots - i))
            for i in range(self.n_pilots)
        ]
        results = []
        for compute in pilot_computes:
            chinchilla = ChinchillaScalingLaw.optimal_for_compute(compute)
            if chinchilla.n_params < self.min_params:
                chinchilla = ChinchillaResult(
                    n_params=self.min_params,
                    n_tokens=compute // (6 * self.min_params),
                    compute_flops=compute,
                    tokens_per_param=compute // (6 * self.min_params) / self.min_params,
                )
            fe = FLOPsEstimator(
                ModelConfig(d_model=512, n_layers=8, n_heads=8)  # placeholder
            )
            cost_est = fe.estimate_training_cost(
                chinchilla.n_tokens,
                mfu=0.40,
                gpu_price_per_hour=2.0,
            )
            results.append({
                "pilot_compute_flops": compute,
                "pilot_compute_T": compute / 1e12,
                "n_params": chinchilla.n_params,
                "n_params_M": chinchilla.n_params / 1e6,
                "n_tokens": chinchilla.n_tokens,
                "n_tokens_B": chinchilla.n_tokens / 1e9,
                "predicted_loss": ChinchillaScalingLaw.predict_loss(
                    chinchilla.n_params, chinchilla.n_tokens
                ),
                "estimated_cost_usd": cost_est["estimated_cost_usd"],
                "estimated_gpu_hours": cost_est["gpu_hours"],
            })
        return results

    def extrapolate_to_budget(self, pilot_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Given pilot run results, fit a scaling law and extrapolate to full budget.
        Fits: L = a * C^(-b) where C is compute.
        """
        if len(pilot_results) < 2:
            return {}

        computes = np.array([r["pilot_compute_flops"] for r in pilot_results])
        losses = np.array([r.get("measured_loss", r["predicted_loss"]) for r in pilot_results])

        # Log-linear fit: log L = log a - b * log C
        log_c = np.log(computes)
        log_l = np.log(losses)
        coeffs = np.polyfit(log_c, log_l, 1)
        b = -coeffs[0]
        a = math.exp(coeffs[1])

        predicted_full_budget_loss = a * (self.compute_budget ** (-b))
        chinchilla_full = ChinchillaScalingLaw.optimal_for_compute(self.compute_budget)

        return {
            "scaling_exponent_b": b,
            "scaling_coeff_a": a,
            "predicted_loss_at_full_budget": predicted_full_budget_loss,
            "predicted_perplexity": math.exp(predicted_full_budget_loss),
            "chinchilla_optimal_params": chinchilla_full.n_params,
            "chinchilla_optimal_tokens": chinchilla_full.n_tokens,
            "r2_fit": float(np.corrcoef(log_c, log_l)[0, 1] ** 2),
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def estimate_model_size(
    d_model: int,
    n_layers: int,
    n_heads: int = 8,
    ffn_mult: float = 4.0,
    vocab_size: int = 50_000,
) -> Dict[str, Any]:
    """Quick helper to estimate model size without building a full config."""
    cfg = ModelConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim_multiplier=ffn_mult,
        vocab_size=vocab_size,
    )
    pc = ParameterCounter(cfg)
    fe = FLOPsEstimator(cfg)
    params = pc.total_params()
    flops = fe.total_flops_forward(cfg.max_seq_len)
    hps = HyperparameterScaler.suggest_hyperparameters(cfg)
    return {
        "config": asdict(cfg),
        "params": params,
        "forward_flops": flops,
        "hyperparameters": hps,
    }


def print_scaling_table(family: str = "lumina") -> None:
    """Print a scaling table for a model family."""
    results = get_model_family_scaling(family)
    header = f"{'d_model':>8} {'n_layers':>9} {'n_heads':>8} {'Params (M)':>12} {'Tokens (B)':>12} {'LR':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['d_model']:>8} {r['n_layers']:>9} {r['n_heads']:>8} "
            f"{r['params_M']:>12.0f} {r['chinchilla_optimal_tokens']/1e9:>12.1f} "
            f"{r['recommended_peak_lr']:>10.2e}"
        )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "ModelConfig",
    "ParameterCounter",
    "FLOPsEstimator",
    "ChinchillaResult",
    "ChinchillaScalingLaw",
    "HyperparameterScaler",
    "TrainingEfficiencyMonitor",
    "ModelCard",
    "ScalingExperimentPlanner",
    "get_model_family_scaling",
    "estimate_model_size",
    "print_scaling_table",
]


# =============================================================================
# SECTION: Neural Scaling Law Estimators
# =============================================================================

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, List, Tuple, Dict


class PowerLawFitter:
    """Fit power-law scaling curves: L = a * N^b + c.

    Used to estimate how loss scales with model size or data size
    following Kaplan et al. (2020) and Hoffmann et al. (2022).
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.a = None
        self.b = None
        self.c = None
        self._fitted = False

    def fit(self, ns: np.ndarray, losses: np.ndarray) -> "PowerLawFitter":
        """Fit L = a * N^b + c to (N, L) data points using log-linear regression."""
        log_n = np.log(ns)
        log_l = np.log(losses)

        if self.fit_intercept:
            A = np.vstack([log_n, np.ones_like(log_n)]).T
            result = np.linalg.lstsq(A, log_l, rcond=None)
            self.b, log_a = result[0]
            self.a = np.exp(log_a)
            self.c = 0.0
        else:
            A = log_n.reshape(-1, 1)
            result = np.linalg.lstsq(A, log_l, rcond=None)
            self.b = result[0][0]
            self.a = 1.0
            self.c = 0.0

        self._fitted = True
        return self

    def predict(self, n: float) -> float:
        """Predict loss for a given N."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.a * (n ** self.b) + self.c

    def optimal_allocation(self, total_flops: float) -> Tuple[float, float]:
        """Estimate optimal (N_params, D_tokens) for a given FLOP budget.

        Uses Chinchilla scaling: N_opt ≈ sqrt(FLOPs / 6), D_opt ≈ sqrt(FLOPs * 6).
        """
        n_opt = math.sqrt(total_flops / 6.0)
        d_opt = total_flops / (6.0 * n_opt)
        return n_opt, d_opt

    def extrapolate_loss(self, current_n: float, target_n: float, current_loss: float) -> float:
        """Extrapolate expected loss at target_n given current (n, loss)."""
        if self.b is not None:
            return current_loss * (target_n / current_n) ** self.b
        # Fallback: assume exponent of -0.076 (Kaplan 2020)
        return current_loss * (target_n / current_n) ** (-0.076)


class ScalingLawPredictor:
    """Predict model performance at scale using fitted scaling laws.

    Combines:
    - Compute-optimal scaling (Chinchilla)
    - Data scaling exponents
    - Architecture-specific adjustments
    """

    def __init__(
        self,
        compute_exponent: float = -0.050,
        data_exponent: float = -0.095,
        param_exponent: float = -0.076,
        irreducible_loss: float = 1.69,
    ):
        self.compute_exponent = compute_exponent
        self.data_exponent = data_exponent
        self.param_exponent = param_exponent
        self.irreducible_loss = irreducible_loss

    def loss_from_params(self, n_params: float, n_tokens: float) -> float:
        """Predict cross-entropy loss from parameter count and training tokens."""
        param_term = (5.4e13 / n_params) ** (-self.param_exponent)
        data_term = (1.8e13 / n_tokens) ** (-self.data_exponent)
        return self.irreducible_loss + param_term + data_term

    def loss_from_flops(self, flops: float) -> float:
        """Predict loss from total training FLOPs."""
        n_opt, d_opt = PowerLawFitter().optimal_allocation(flops)
        return self.loss_from_params(n_opt, d_opt)

    def compute_budget_for_loss(self, target_loss: float) -> float:
        """Estimate compute (FLOPs) required to achieve target_loss."""
        from scipy.optimize import brentq
        try:
            def objective(log_flops):
                return self.loss_from_flops(math.exp(log_flops)) - target_loss
            log_flops = brentq(objective, 20, 40)
            return math.exp(log_flops)
        except Exception:
            return float("nan")

    def params_for_target_loss(self, target_loss: float, n_tokens: float) -> float:
        """Estimate parameter count needed for target_loss given token budget."""
        from scipy.optimize import brentq
        try:
            def objective(log_n):
                return self.loss_from_params(math.exp(log_n), n_tokens) - target_loss
            log_n = brentq(objective, 10, 30)
            return math.exp(log_n)
        except Exception:
            return float("nan")


# =============================================================================
# SECTION: Model Size Calculator
# =============================================================================

class TransformerSizeCalculator:
    """Calculate parameter counts and FLOPs for transformer architectures."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_ratio: float = 4.0,
        max_seq_len: int = 2048,
        tie_embeddings: bool = True,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_ratio = ffn_ratio
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        self.d_ff = int(d_model * ffn_ratio)
        self.d_head = d_model // n_heads

    def embedding_params(self) -> int:
        token_emb = self.vocab_size * self.d_model
        pos_emb = self.max_seq_len * self.d_model
        return token_emb + pos_emb

    def attention_params_per_layer(self) -> int:
        qkv = 3 * self.d_model * self.d_model
        out_proj = self.d_model * self.d_model
        return qkv + out_proj

    def ffn_params_per_layer(self) -> int:
        fc1 = self.d_model * self.d_ff
        fc2 = self.d_ff * self.d_model
        bias1 = self.d_ff
        bias2 = self.d_model
        return fc1 + fc2 + bias1 + bias2

    def layernorm_params_per_layer(self) -> int:
        return 4 * self.d_model  # 2 layer norms, each with weight + bias

    def total_params(self) -> int:
        emb = self.embedding_params()
        if self.tie_embeddings:
            lm_head = 0
        else:
            lm_head = self.vocab_size * self.d_model

        per_layer = (
            self.attention_params_per_layer()
            + self.ffn_params_per_layer()
            + self.layernorm_params_per_layer()
        )
        return emb + self.n_layers * per_layer + lm_head

    def flops_per_token(self) -> int:
        """Estimate FLOPs per token for a forward pass (approximate)."""
        attn_flops = 4 * self.d_model * self.d_model  # QKV + out proj
        ffn_flops = 8 * self.d_model * self.d_ff
        return self.n_layers * (attn_flops + ffn_flops)

    def training_flops(self, n_tokens: int) -> int:
        """Total training FLOPs ≈ 6 * N_params * N_tokens (Chinchilla approximation)."""
        return 6 * self.total_params() * n_tokens

    def memory_footprint_gb(self, batch_size: int = 1, precision: str = "fp32") -> float:
        """Estimate GPU memory for activations + parameters + gradients."""
        bytes_per_param = 4 if precision == "fp32" else 2
        param_bytes = self.total_params() * bytes_per_param
        grad_bytes = param_bytes  # same size for gradients
        # Activations: rough estimate
        act_bytes = (
            batch_size
            * self.max_seq_len
            * self.d_model
            * self.n_layers
            * 2  # forward + backward
            * bytes_per_param
        )
        return (param_bytes + grad_bytes + act_bytes) / (1024 ** 3)

    def summary(self) -> dict:
        return {
            "total_params": self.total_params(),
            "total_params_M": self.total_params() / 1e6,
            "embedding_params": self.embedding_params(),
            "params_per_layer": (
                self.attention_params_per_layer()
                + self.ffn_params_per_layer()
                + self.layernorm_params_per_layer()
            ),
            "flops_per_token": self.flops_per_token(),
            "training_flops_100B": self.training_flops(100_000_000_000),
            "memory_fp32_gb": self.memory_footprint_gb(precision="fp32"),
            "memory_fp16_gb": self.memory_footprint_gb(precision="fp16"),
        }


# =============================================================================
# SECTION: Dynamic Architecture Scaling
# =============================================================================

class DepthScaler:
    """Scale model depth (number of layers) dynamically.

    Supports:
    - Progressive layer dropping (training regularization)
    - Layer freezing for staged fine-tuning
    - Layer-wise adaptive learning rates
    """

    def __init__(self, model: nn.Module, layer_attr: str = "layers"):
        self.model = model
        self.layer_attr = layer_attr
        self._frozen_layers = set()
        self._drop_rates = {}

    def _get_layers(self) -> nn.ModuleList:
        return getattr(self.model, self.layer_attr)

    def freeze_bottom_k(self, k: int):
        """Freeze the bottom k layers."""
        layers = self._get_layers()
        for i in range(min(k, len(layers))):
            for p in layers[i].parameters():
                p.requires_grad = False
            self._frozen_layers.add(i)

    def unfreeze_layer(self, layer_idx: int):
        """Unfreeze a specific layer."""
        layers = self._get_layers()
        if layer_idx < len(layers):
            for p in layers[layer_idx].parameters():
                p.requires_grad = True
            self._frozen_layers.discard(layer_idx)

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for p in self.model.parameters():
            p.requires_grad = True
        self._frozen_layers.clear()

    def set_progressive_drop_rate(self, max_drop_rate: float = 0.2):
        """Set stochastic depth drop rates: deeper layers drop more often."""
        layers = self._get_layers()
        n = len(layers)
        for i, layer in enumerate(layers):
            rate = max_drop_rate * i / max(n - 1, 1)
            self._drop_rates[i] = rate
            if hasattr(layer, "drop_rate"):
                layer.drop_rate = rate

    def get_trainable_params(self) -> int:
        """Count currently trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_frozen_params(self) -> int:
        """Count frozen parameters."""
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)


class WidthScaler:
    """Scale model width (hidden dimensions) dynamically.

    Supports:
    - Width multipliers for each component
    - Attention head pruning
    - FFN intermediate dimension adjustment
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        width_multiplier: float = 1.0,
    ):
        self.d_model = int(d_model * width_multiplier)
        self.n_heads = n_heads
        self.d_ff = int(d_ff * width_multiplier)
        self.d_head = self.d_model // self.n_heads
        self.width_multiplier = width_multiplier

    def get_config(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "d_head": self.d_head,
            "width_multiplier": self.width_multiplier,
        }

    def scale_to(self, new_multiplier: float) -> "WidthScaler":
        """Return a new WidthScaler with a different multiplier."""
        base_d = int(self.d_model / self.width_multiplier)
        base_ff = int(self.d_ff / self.width_multiplier)
        return WidthScaler(base_d, self.n_heads, base_ff, new_multiplier)


# =============================================================================
# SECTION: Mixture of Depth (MoD)
# =============================================================================

class TokenRouterMoD(nn.Module):
    """Token routing for Mixture of Depth (MoD).

    Selects which tokens pass through a given layer vs. being skipped.
    Based on Raposo et al. (2024): each layer only processes the top-k tokens
    by routing weight; the rest are passed through via residual connection.

    This allows variable compute per token, similar to MoE but along depth.
    """

    def __init__(self, d_model: int, capacity_factor: float = 0.5):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        layer: nn.Module,
    ) -> torch.Tensor:
        """Route tokens through layer: top-k processed, rest skip."""
        B, T, D = x.shape
        k = max(1, int(T * self.capacity_factor))

        scores = self.router(x).squeeze(-1)  # [B, T]
        topk_vals, topk_idx = torch.topk(scores, k, dim=1)

        # Gather selected tokens
        expanded_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        selected = x.gather(1, expanded_idx)  # [B, k, D]

        # Process selected tokens through layer
        processed = layer(selected)

        # Scatter back
        out = x.clone()
        weights = torch.sigmoid(topk_vals).unsqueeze(-1)
        out.scatter_add_(1, expanded_idx, processed * weights - selected * weights)

        return out


class MixtureOfDepthTransformer(nn.Module):
    """Transformer with Mixture of Depth routing.

    Alternates between full layers (all tokens) and MoD layers (top-k tokens).
    Reduces total FLOPs while maintaining model capacity.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        capacity_factor: float = 0.5,
        mod_every_n: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList()
        self.routers = nn.ModuleList()
        self.is_mod_layer = []

        for i in range(n_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.layers.append(layer)
            use_mod = (i % mod_every_n == 1)
            self.is_mod_layer.append(use_mod)
            if use_mod:
                self.routers.append(TokenRouterMoD(d_model, capacity_factor))
            else:
                self.routers.append(None)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        for layer, router, is_mod in zip(self.layers, self.routers, self.is_mod_layer):
            if is_mod and router is not None:
                x = router(x, layer)
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def estimate_effective_flops(self, seq_len: int) -> float:
        """Estimate effective FLOPs considering MoD routing."""
        n_full = sum(1 for b in self.is_mod_layer if not b)
        n_mod = sum(1 for b in self.is_mod_layer if b)
        cap = self.routers[0].capacity_factor if n_mod > 0 else 1.0

        full_flop = n_full * seq_len
        mod_flop = n_mod * seq_len * cap
        return full_flop + mod_flop


# =============================================================================
# SECTION: Efficient Attention Approximations
# =============================================================================

class LinearAttentionKernel(nn.Module):
    """Linear attention using kernel feature maps (Katharopoulos 2020).

    Approximates softmax attention with O(n) complexity using:
    Attention(Q, K, V) ≈ phi(Q) * (phi(K)^T * V) / (phi(Q) * phi(K)^T * 1)

    where phi(x) = elu(x) + 1 is the feature map.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int = None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        Q = self._feature_map(Q)
        K = self._feature_map(K)

        # Compute KV product: [B, H, Dh, Dh]
        KV = torch.einsum("bhnd,bhnm->bhdm", K, V)
        # Compute QKV: [B, H, T, Dh]
        QKV = torch.einsum("bhnd,bhdm->bhnm", Q, KV)
        # Normalizer
        K_sum = K.sum(dim=2)  # [B, H, Dh]
        Z = torch.einsum("bhnd,bhd->bhn", Q, K_sum).unsqueeze(-1).clamp(min=1e-6)

        out = (QKV / Z).transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.out_proj(out)


class PerformerAttention(nn.Module):
    """PERFORMER: random feature attention for O(n) complexity (Choromanski 2021).

    Uses random orthogonal features to approximate the softmax kernel:
    K(q, k) = E[phi(q) * phi(k)]

    where phi uses sin/cos random projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_random_features: int = 256,
        orthogonal_random: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_random_features = n_random_features
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Random projection matrix
        self._init_random_features(orthogonal_random)

    def _init_random_features(self, orthogonal: bool):
        m = self.n_random_features
        d = self.d_head
        if orthogonal:
            W = torch.zeros(m, d)
            for i in range(0, m, d):
                block = torch.randn(min(d, m - i), d)
                q, _ = torch.linalg.qr(block)
                W[i:i+min(d, m-i)] = q[:min(d, m-i)]
        else:
            W = torch.randn(m, d) / math.sqrt(d)
        self.register_buffer("random_features", W)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Random feature map: phi(x) = exp(-||x||^2/2) * [cos(Wx), sin(Wx)] / sqrt(m)."""
        norms_sq = (x ** 2).sum(dim=-1, keepdim=True)
        projections = x @ self.random_features.T  # [..., m]
        cos_proj = torch.cos(projections)
        sin_proj = torch.sin(projections)
        scale = torch.exp(-norms_sq / 2) / math.sqrt(self.n_random_features)
        return torch.cat([cos_proj, sin_proj], dim=-1) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        m = self.n_random_features * 2

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        Q_feat = self._phi(Q)  # [B, H, T, 2m]
        K_feat = self._phi(K)

        KV = torch.einsum("bhnd,bhnv->bhdv", K_feat, V)  # [B, H, 2m, Dh]
        QKV = torch.einsum("bhnd,bhdv->bhnv", Q_feat, KV)  # [B, H, T, Dh]

        K_sum = K_feat.sum(dim=2)
        Z = torch.einsum("bhnd,bhd->bhn", Q_feat, K_sum).unsqueeze(-1).clamp(min=1e-6)

        out = (QKV / Z).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# =============================================================================
# SECTION: Gradient Checkpointing Strategies
# =============================================================================

class SelectiveGradientCheckpointing:
    """Selectively apply gradient checkpointing to reduce memory.

    Strategies:
    - every_n: checkpoint every n-th layer
    - memory_budget: checkpoint until memory target is met
    - sensitivity: checkpoint layers with smallest gradient norms
    """

    def __init__(self, model: nn.Module, strategy: str = "every_n", n: int = 2):
        self.model = model
        self.strategy = strategy
        self.n = n
        self._checkpointed = []

    def apply_every_n(self, layer_attr: str = "layers"):
        """Apply checkpointing to every n-th layer."""
        from torch.utils.checkpoint import checkpoint

        layers = getattr(self.model, layer_attr, None)
        if layers is None:
            return

        for i, layer in enumerate(layers):
            if i % self.n == 0:
                original_forward = layer.forward

                def make_checkpointed(orig_fwd):
                    def checkpointed_forward(*args, **kwargs):
                        return checkpoint(orig_fwd, *args, use_reentrant=False, **kwargs)
                    return checkpointed_forward

                layer.forward = make_checkpointed(original_forward)
                self._checkpointed.append(i)

    def estimate_memory_savings(self, d_model: int, seq_len: int, n_layers: int) -> dict:
        """Estimate memory savings from gradient checkpointing."""
        bytes_per_activation = 4  # fp32
        full_activation_bytes = n_layers * seq_len * d_model * bytes_per_activation
        n_checkpointed = len(self._checkpointed)
        saved = n_checkpointed * seq_len * d_model * bytes_per_activation
        return {
            "full_activation_mb": full_activation_bytes / 1e6,
            "saved_mb": saved / 1e6,
            "remaining_mb": (full_activation_bytes - saved) / 1e6,
            "checkpointed_layers": self._checkpointed,
        }


# =============================================================================
# SECTION: Knowledge Distillation for Model Compression
# =============================================================================

class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard and soft targets.

    L = alpha * CE(student, hard_labels) + (1 - alpha) * KL(student, teacher) * T^2

    Following Hinton et al. (2015) and DistilBERT (Sanh 2019).
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: torch.Tensor = None,
    ) -> dict:
        import torch.nn.functional as F

        T = self.temperature
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)

        distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

        if hard_labels is not None and self.alpha > 0:
            ce_loss = F.cross_entropy(student_logits, hard_labels)
            total = self.alpha * ce_loss + (1 - self.alpha) * distill_loss
        else:
            ce_loss = torch.tensor(0.0)
            total = distill_loss

        return {
            "total": total,
            "distillation": distill_loss,
            "classification": ce_loss,
        }


class LayerMatchingDistillation(nn.Module):
    """Intermediate layer matching distillation (PKD, TinyBERT).

    Aligns intermediate hidden states between teacher and student
    using MSE or cosine similarity losses.
    """

    def __init__(
        self,
        student_d_model: int,
        teacher_d_model: int,
        layer_pairs: List[Tuple[int, int]],
        loss_type: str = "mse",
    ):
        super().__init__()
        self.layer_pairs = layer_pairs
        self.loss_type = loss_type

        # Projection layers to match dimensions
        self.projections = nn.ModuleList([
            nn.Linear(student_d_model, teacher_d_model, bias=False)
            for _ in layer_pairs
        ])

    def forward(
        self,
        student_hiddens: List[torch.Tensor],
        teacher_hiddens: List[torch.Tensor],
    ) -> torch.Tensor:
        import torch.nn.functional as F

        total_loss = torch.tensor(0.0, device=student_hiddens[0].device)

        for (s_idx, t_idx), proj in zip(self.layer_pairs, self.projections):
            s_h = proj(student_hiddens[s_idx])
            t_h = teacher_hiddens[t_idx].detach()

            if self.loss_type == "mse":
                loss = F.mse_loss(s_h, t_h)
            elif self.loss_type == "cosine":
                loss = 1.0 - F.cosine_similarity(s_h, t_h, dim=-1).mean()
            elif self.loss_type == "huber":
                loss = F.huber_loss(s_h, t_h)
            else:
                loss = F.mse_loss(s_h, t_h)

            total_loss = total_loss + loss

        return total_loss / max(len(self.layer_pairs), 1)


class AttentionTransferDistillation(nn.Module):
    """Attention map transfer distillation (Zagoruyko & Komodakis 2017).

    Transfers attention patterns from teacher to student using
    sum-of-squares normalization.
    """

    def __init__(self, beta: float = 1000.0):
        super().__init__()
        self.beta = beta

    def _attention_map(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute sum-of-squares attention map: F(A) = ||A||_2 normalized."""
        return torch.nn.functional.normalize(attention.pow(2).sum(dim=1), dim=1)

    def forward(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor],
    ) -> torch.Tensor:
        import torch.nn.functional as F

        total = torch.tensor(0.0)
        for s_att, t_att in zip(student_attentions, teacher_attentions):
            s_map = self._attention_map(s_att)
            t_map = self._attention_map(t_att).detach()
            total = total + F.mse_loss(s_map, t_map) * self.beta

        return total / max(len(student_attentions), 1)


# =============================================================================
# SECTION: Model Pruning
# =============================================================================

class MagnitudePruner:
    """Prune weights by magnitude (unstructured pruning).

    Creates binary masks for each parameter where small-magnitude
    weights are zeroed out.
    """

    def __init__(self, model: nn.Module, sparsity: float = 0.5):
        self.model = model
        self.sparsity = sparsity
        self._masks = {}

    def compute_masks(self, global_threshold: bool = True):
        """Compute pruning masks based on weight magnitudes."""
        if global_threshold:
            all_weights = torch.cat([
                p.data.abs().flatten()
                for p in self.model.parameters()
                if p.requires_grad and p.ndim >= 2
            ])
            threshold = torch.quantile(all_weights, self.sparsity)

            for name, p in self.model.named_parameters():
                if p.requires_grad and p.ndim >= 2:
                    self._masks[name] = (p.data.abs() > threshold).float()
        else:
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.ndim >= 2:
                    threshold = torch.quantile(p.data.abs(), self.sparsity)
                    self._masks[name] = (p.data.abs() > threshold).float()

    def apply_masks(self):
        """Zero out pruned weights using stored masks."""
        for name, p in self.model.named_parameters():
            if name in self._masks:
                p.data.mul_(self._masks[name])

    def sparsity_stats(self) -> dict:
        """Report actual sparsity per parameter."""
        stats = {}
        for name, mask in self._masks.items():
            total = mask.numel()
            zeros = (mask == 0).sum().item()
            stats[name] = {"sparsity": zeros / total, "total": total, "pruned": zeros}
        return stats

    def global_sparsity(self) -> float:
        """Overall fraction of zeroed weights."""
        total_zeros = sum((m == 0).sum().item() for m in self._masks.values())
        total_params = sum(m.numel() for m in self._masks.values())
        return total_zeros / max(total_params, 1)


class StructuredPruner:
    """Structured pruning: remove entire attention heads or FFN neurons.

    More hardware-friendly than unstructured pruning since it directly
    reduces matrix dimensions.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._head_importance = {}
        self._neuron_importance = {}

    def compute_head_importance(self, dataloader, loss_fn, n_batches: int = 10) -> dict:
        """Estimate attention head importance via gradient norms."""
        importance = {}

        for name, module in self.model.named_modules():
            if hasattr(module, "q_proj") and hasattr(module, "n_heads"):
                importance[name] = torch.zeros(module.n_heads)

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            self.model.zero_grad()
            out = self.model(**batch)
            loss = loss_fn(out)
            loss.backward()

            for name, module in self.model.named_modules():
                if name in importance and hasattr(module, "q_proj"):
                    grad = module.q_proj.weight.grad
                    if grad is not None:
                        n_heads = module.n_heads
                        d_head = grad.shape[0] // n_heads
                        head_grads = grad.view(n_heads, d_head, -1)
                        importance[name] += head_grads.abs().mean(dim=(1, 2)).detach()

        self._head_importance = importance
        return importance

    def prune_heads(self, heads_to_prune: Dict[str, List[int]]):
        """Prune specific attention heads from modules."""
        for name, head_indices in heads_to_prune.items():
            parts = name.split(".")
            module = self.model
            for part in parts:
                module = getattr(module, part)

            if hasattr(module, "prune_heads"):
                module.prune_heads(head_indices)


# =============================================================================
# SECTION: Adaptive Compute
# =============================================================================

class AdaptiveComputeWrapper(nn.Module):
    """Adaptive computation time (ACT) wrapper for variable-depth processing.

    Based on Graves (2016): each token decides how many steps of computation
    it needs via a halting probability.
    """

    def __init__(
        self,
        layer: nn.Module,
        d_model: int,
        max_steps: int = 10,
        halt_threshold: float = 0.99,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.layer = layer
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        self.epsilon = epsilon
        self.halting_unit = nn.Linear(d_model, 1)
        self.ponder_cost_weight = 0.01

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run layer with adaptive computation.

        Returns:
            (output, ponder_cost) where ponder_cost is the mean steps taken.
        """
        B, T, D = x.shape
        halting_probs = torch.zeros(B, T, device=x.device)
        remainders = torch.zeros(B, T, device=x.device)
        n_updates = torch.zeros(B, T, device=x.device)
        cumulative = torch.zeros(B, T, 1, device=x.device)
        output = torch.zeros_like(x)
        state = x.clone()

        for step in range(self.max_steps):
            p = torch.sigmoid(self.halting_unit(state).squeeze(-1))

            still_running = (halting_probs < self.halt_threshold).float()

            new_halted = ((halting_probs + p * still_running) >= self.halt_threshold).float() * still_running
            remainder = (1.0 - halting_probs) * new_halted

            halting_probs = halting_probs + p * still_running
            remainders = remainders + remainder
            n_updates = n_updates + still_running

            update_weights = (p * still_running + remainder).unsqueeze(-1)

            state_new = self.layer(state)
            output = output + update_weights * state_new
            state = state_new

            if (still_running.sum() == 0):
                break

        ponder_cost = (n_updates + remainders).mean()
        return output, ponder_cost * self.ponder_cost_weight


# =============================================================================
# SECTION: Efficient Training Utilities
# =============================================================================

class GradientNormMonitor:
    """Monitor and clip gradient norms during training."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self._history = []

    def clip_and_record(self, parameters) -> float:
        """Clip gradients and record the norm."""
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)
        self._history.append(total_norm.item() if torch.is_tensor(total_norm) else total_norm)
        return total_norm

    def recent_stats(self, window: int = 100) -> dict:
        """Statistics over recent gradient norms."""
        recent = self._history[-window:]
        if not recent:
            return {}
        import statistics
        return {
            "mean": statistics.mean(recent),
            "max": max(recent),
            "min": min(recent),
            "clipped_fraction": sum(1 for n in recent if n > self.max_norm) / len(recent),
        }


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_step: int = -1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = last_step + 1

    def step(self):
        """Update learning rate for current step."""
        s = self._step
        if s < self.warmup_steps:
            factor = s / max(self.warmup_steps, 1)
        else:
            progress = (s - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        for lr, group in zip(self._base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * factor

        self._step += 1
        return factor

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


class CyclicLRScheduler:
    """Cyclic LR with triangular or exp range policy (Smith 2017)."""

    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        step_size: int = 2000,
        mode: str = "triangular",
        gamma: float = 0.999,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self._cycle = 0
        self._step = 0

    def step(self):
        cycle = math.floor(1 + self._step / (2 * self.step_size))
        x = abs(self._step / self.step_size - 2 * cycle + 1)
        scale = max(0, 1 - x)

        if self.mode == "triangular":
            lr = self.base_lr + (self.max_lr - self.base_lr) * scale
        elif self.mode == "triangular2":
            lr = self.base_lr + (self.max_lr - self.base_lr) * scale / (2 ** (cycle - 1))
        elif self.mode == "exp_range":
            lr = self.base_lr + (self.max_lr - self.base_lr) * scale * (self.gamma ** self._step)
        else:
            lr = self.base_lr

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self._step += 1
        return lr


class SAM(torch.optim.Optimizer):
    """Sharpness Aware Minimization (Foret et al. 2021).

    SAM perturbs weights toward higher loss regions, then optimizes
    the perturbed loss to find flat minima. Two-step update:
    1. Gradient step to find perturbation delta
    2. Unperturb and take gradient step at original weights
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Compute and apply perturbation."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Remove perturbation and take gradient step."""
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group.get("adaptive") else torch.ones_like(p)) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM requires two-step: first_step + second_step")


# =============================================================================
# SECTION: Mixed Precision Training Utilities
# =============================================================================

class DynamicLossScaler:
    """Dynamic loss scaling for mixed precision training.

    Automatically adjusts loss scale based on gradient overflow detection.
    Equivalent to GradScaler but with more control over scaling schedule.
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_scale: float = 2 ** 24,
        min_scale: float = 1.0,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_scale = max_scale
        self.min_scale = min_scale
        self._steps_since_overflow = 0
        self._total_overflows = 0
        self._total_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    def check_overflow(self, parameters) -> bool:
        """Check for inf/nan gradients."""
        for p in parameters:
            if p.grad is not None:
                if torch.any(torch.isinf(p.grad)) or torch.any(torch.isnan(p.grad)):
                    return True
        return False

    def update(self, overflow: bool):
        """Update scale based on overflow status."""
        self._total_steps += 1
        if overflow:
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self._steps_since_overflow = 0
            self._total_overflows += 1
        else:
            self._steps_since_overflow += 1
            if self._steps_since_overflow >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self._steps_since_overflow = 0

    def state_dict(self) -> dict:
        return {
            "scale": self.scale,
            "steps_since_overflow": self._steps_since_overflow,
            "total_overflows": self._total_overflows,
        }

    def load_state_dict(self, state: dict):
        self.scale = state["scale"]
        self._steps_since_overflow = state.get("steps_since_overflow", 0)
        self._total_overflows = state.get("total_overflows", 0)


# =============================================================================
# SECTION: Scaling Registry
# =============================================================================

_SCALING_REGISTRY = {}


def register_scaling_strategy(name: str):
    def decorator(cls):
        _SCALING_REGISTRY[name] = cls
        return cls
    return decorator


def get_scaling_strategy(name: str):
    if name not in _SCALING_REGISTRY:
        raise KeyError(f"Scaling strategy '{name}' not found. Available: {list(_SCALING_REGISTRY.keys())}")
    return _SCALING_REGISTRY[name]


@register_scaling_strategy("depth")
class DepthScalingStrategy:
    """Scale model by increasing number of layers."""

    def __init__(self, base_n_layers: int = 6, scale_factor: float = 2.0):
        self.base_n_layers = base_n_layers
        self.scale_factor = scale_factor

    def get_config(self, target_scale: float) -> dict:
        return {"n_layers": int(self.base_n_layers * target_scale)}


@register_scaling_strategy("width")
class WidthScalingStrategy:
    """Scale model by increasing hidden dimensions."""

    def __init__(self, base_d_model: int = 256, scale_factor: float = 2.0):
        self.base_d_model = base_d_model
        self.scale_factor = scale_factor

    def get_config(self, target_scale: float) -> dict:
        d = int(self.base_d_model * target_scale)
        return {"d_model": d, "d_ff": d * 4}


@register_scaling_strategy("balanced")
class BalancedScalingStrategy:
    """Balanced depth/width scaling following compound scaling (EfficientNet style)."""

    def __init__(self, depth_coeff: float = 1.2, width_coeff: float = 1.1):
        self.depth_coeff = depth_coeff
        self.width_coeff = width_coeff

    def get_config(self, base_config: dict, target_scale: float) -> dict:
        cfg = dict(base_config)
        cfg["n_layers"] = int(cfg.get("n_layers", 6) * (self.depth_coeff ** target_scale))
        cfg["d_model"] = int(cfg.get("d_model", 256) * (self.width_coeff ** target_scale))
        cfg["d_ff"] = cfg["d_model"] * 4
        return cfg
