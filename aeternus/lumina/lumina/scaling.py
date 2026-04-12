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
