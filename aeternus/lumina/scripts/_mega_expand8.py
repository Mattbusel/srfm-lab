#!/usr/bin/env python3
"""Mega expansion 8 - more content across multiple files."""
import os, subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def append(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    n = open(path, encoding="utf-8").read().count("\n")
    print(f"  {os.path.basename(path)}: {n} lines")
    return n

def write_new(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    n = open(path, encoding="utf-8").read().count("\n")
    print(f"  {os.path.basename(path)}: {n} lines (new)")
    return n

# ════════════════════════════════════════════════════════════════════════════════
# 1. tests/test_stress_comprehensive.py - stress tests
# ════════════════════════════════════════════════════════════════════════════════
def build_stress_tests():
    lines = [
        '"""Stress tests and edge case tests for Lumina components."""',
        "import pytest",
        "import torch",
        "import torch.nn as nn",
        "import math",
        "",
        "# ═══ Memory and batch stress ══════════════════════════════════════════════",
        "",
    ]

    # Large batch stress tests for attention
    attention_classes = [
        "CosineAttention",
        "RetentiveAttention",
        "MultiQueryAttention",
        "ALiBiAttention",
        "RoPEAttention",
    ]

    for cls in attention_classes:
        lines += [
            f"class TestStress{cls}:",
            f"    @pytest.mark.parametrize('B,T', [(1, 128), (2, 64), (4, 32), (8, 16)])",
            f"    def test_various_batch_seq(self, B, T):",
            f"        from attention import {cls}",
            f"        model = {cls}(d_model=64, num_heads=4)",
            f"        x = torch.randn(B, T, 64)",
            f"        with torch.no_grad():",
            f"            out = model(x)",
            f"            if isinstance(out, tuple): out = out[0]",
            f"        assert out.shape == (B, T, 64)",
            f"        assert not torch.isnan(out).any()",
            "",
            f"    @pytest.mark.parametrize('d_model,num_heads', [(32, 4), (64, 8), (128, 8), (256, 8)])",
            f"    def test_various_sizes(self, d_model, num_heads):",
            f"        if d_model % num_heads != 0:",
            f"            pytest.skip('incompatible dims')",
            f"        from attention import {cls}",
            f"        model = {cls}(d_model=d_model, num_heads=num_heads)",
            f"        x = torch.randn(2, 8, d_model)",
            f"        with torch.no_grad():",
            f"            out = model(x)",
            f"            if isinstance(out, tuple): out = out[0]",
            f"        assert out.shape == (2, 8, d_model)",
            "",
        ]

    # Edge case tests
    lines += [
        "# ═══ Edge case tests ═════════════════════════════════════════════════════",
        "",
        "class TestEdgeCases:",
        "    def test_single_token_attention(self):",
        "        from attention import RoPEAttention",
        "        model = RoPEAttention(64, 4)",
        "        x = torch.randn(2, 1, 64)",
        "        out = model(x)",
        "        assert out.shape == (2, 1, 64)",
        "",
        "    def test_batch_size_one(self):",
        "        from attention import ALiBiAttention",
        "        model = ALiBiAttention(64, 4)",
        "        x = torch.randn(1, 16, 64)",
        "        out = model(x)",
        "        assert out.shape == (1, 16, 64)",
        "",
        "    def test_zero_input(self):",
        "        from attention import CosineAttention",
        "        model = CosineAttention(32, 4)",
        "        x = torch.zeros(2, 8, 32)",
        "        out = model(x)",
        "        assert not torch.isnan(out).any()",
        "        assert not torch.isinf(out).any()",
        "",
        "    def test_large_values(self):",
        "        from attention import MultiQueryAttention",
        "        model = MultiQueryAttention(32, 4)",
        "        x = torch.randn(2, 8, 32) * 10",
        "        out = model(x)",
        "        assert not torch.isnan(out).any()",
        "",
        "    def test_gradient_clipping_safe(self):",
        "        from lora import LoRALinear",
        "        layer = LoRALinear(32, 64, rank=4)",
        "        x = torch.randn(2, 8, 32)",
        "        out = layer(x).sum()",
        "        out.backward()",
        "        nn.utils.clip_grad_norm_(layer.parameters(), 1.0)",
        "        for p in layer.parameters():",
        "            if p.grad is not None:",
        "                assert not torch.isnan(p.grad).any()",
        "",
        "    def test_transformer_single_layer(self):",
        "        from transformer import NormFormerBlock",
        "        block = NormFormerBlock(32, 4)",
        "        x = torch.randn(2, 4, 32)",
        "        out = block(x)",
        "        assert out.shape == (2, 4, 32)",
        "",
        "    def test_moe_all_same_expert(self):",
        "        from moe import SparseMoELayer",
        "        moe = SparseMoELayer(32, num_experts=4, top_k=1)",
        "        x = torch.zeros(2, 4, 32)  # identical tokens -> same routing",
        "        out = moe(x)",
        "        assert not torch.isnan(out).any()",
        "",
        "    def test_lora_zero_rank(self):",
        "        # rank=1 minimal case",
        "        from lora import LoRALinear",
        "        layer = LoRALinear(16, 32, rank=1, alpha=2.0)",
        "        x = torch.randn(2, 4, 16)",
        "        out = layer(x)",
        "        assert out.shape == (2, 4, 32)",
        "",
        "    def test_seq_len_equals_one_transformer(self):",
        "        from transformer import SandwichTransformerBlock",
        "        block = SandwichTransformerBlock(64, 4)",
        "        x = torch.randn(4, 1, 64)",
        "        out = block(x)",
        "        assert out.shape == (4, 1, 64)",
        "",
    ]

    # Numerical stability tests
    lines += [
        "# ═══ Numerical stability ════════════════════════════════════════════════",
        "",
        "class TestNumericalStability:",
        "    def test_attention_no_overflow_fp32(self):",
        "        from attention import RoPEAttention",
        "        model = RoPEAttention(64, 4)",
        "        x = torch.randn(2, 16, 64) * 100",
        "        out = model(x)",
        "        assert not torch.isinf(out).any()",
        "",
        "    def test_lora_with_large_rank(self):",
        "        from lora import LoRALinear",
        "        layer = LoRALinear(64, 128, rank=32, alpha=64.0)",
        "        x = torch.randn(2, 8, 64)",
        "        out = layer(x)",
        "        assert not torch.isnan(out).any()",
        "",
        "    def test_moe_router_no_nan(self):",
        "        from moe import TopKRouter",
        "        router = TopKRouter(64, 16, top_k=4)",
        "        x = torch.randn(64, 64)",
        "        out = router(x)",
        "        assert not torch.isnan(out.router_probs).any()",
        "",
        "    def test_conformer_with_bn(self):",
        "        from transformer import ConformerBlock",
        "        block = ConformerBlock(64, 4, conv_kernel=31)",
        "        block.train()",
        "        x = torch.randn(8, 64, 64)  # larger batch for BN",
        "        out = block(x)",
        "        assert not torch.isnan(out).any()",
        "",
        "    def test_continual_norm_no_div_zero(self):",
        "        from continual_learning import ContinualNormalization",
        "        cn = ContinualNormalization(32, num_tasks=2)",
        "        cn.set_task(0)",
        "        x = torch.ones(8, 32)  # constant input -> var=0",
        "        cn.train()",
        "        out = cn(x)",
        "        assert not torch.isnan(out).any()",
        "        assert not torch.isinf(out).any()",
        "",
        "    def test_dpo_symmetric_gives_zero_reward_margin(self):",
        "        from rlhf import DirectPreferenceOptimization",
        "        import torch.nn as nn",
        "        # When policy==reference, reward margin should be ~0",
        "        class PassThrough(nn.Module):",
        "            def __init__(self):",
        "                super().__init__()",
        "                self.emb = nn.Embedding(50, 16)",
        "                self.head = nn.Linear(16, 50, bias=False)",
        "            def forward(self, ids): return self.head(self.emb(ids))",
        "        m = PassThrough()",
        "        dpo = DirectPreferenceOptimization(m, m, beta=0.1)",
        "        prompt = torch.randint(0, 50, (2, 4))",
        "        chosen = torch.randint(0, 50, (2, 4))",
        "        rejected = torch.randint(0, 50, (2, 4))",
        "        loss, metrics = dpo(prompt, chosen, rejected)",
        "        assert abs(metrics['reward_margin']) < 1e-3",
        "",
    ]

    # Performance/throughput tests
    lines += [
        "# ═══ Throughput benchmarks ═══════════════════════════════════════════════",
        "",
        "class TestThroughput:",
        "    def test_attention_forward_backward_perf(self):",
        "        import time",
        "        from attention import RoPEAttention",
        "        model = RoPEAttention(64, 4)",
        "        x = torch.randn(4, 64, 64)",
        "        # Warm up",
        "        for _ in range(3):",
        "            model(x).sum().backward()",
        "        # Time",
        "        t0 = time.time()",
        "        for _ in range(20):",
        "            out = model(x)",
        "            out.sum().backward()",
        "        elapsed = time.time() - t0",
        "        print(f'RoPEAttention 20 fwd+bwd: {elapsed:.3f}s')",
        "        assert elapsed < 60.0  # should be fast",
        "",
        "    def test_lora_vs_dense_overhead(self):",
        "        import time",
        "        from lora import LoRALinear",
        "        dense = nn.Linear(256, 512)",
        "        lora = LoRALinear(256, 512, rank=16)",
        "        x = torch.randn(32, 64, 256)",
        "        # Warm up",
        "        for _ in range(3):",
        "            dense(x)",
        "            lora(x)",
        "        t0 = time.time()",
        "        for _ in range(50):",
        "            dense(x)",
        "        t_dense = time.time() - t0",
        "        t0 = time.time()",
        "        for _ in range(50):",
        "            lora(x)",
        "        t_lora = time.time() - t0",
        "        # LoRA should not be more than 3x slower than dense",
        "        assert t_lora < 3 * t_dense + 1.0",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_stress_comprehensive.py", build_stress_tests())

# ════════════════════════════════════════════════════════════════════════════════
# 2. Append large content to scaling.py
# ════════════════════════════════════════════════════════════════════════════════
SCALING_ADD = '''

# ============================================================
# Extended Scaling Components - Part 2
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ModelSizeProfile:
    """Complete size profile of a transformer model."""
    n_params_total: int
    n_params_trainable: int
    n_params_embedding: int
    n_params_attention: int
    n_params_ffn: int
    n_params_other: int
    memory_fp32_mb: float
    memory_fp16_mb: float
    flops_per_token: float
    attention_fraction: float
    ffn_fraction: float

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelSizeProfile":
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        emb = att = ffn = other = 0
        for name, param in model.named_parameters():
            n = param.numel()
            if "embed" in name or "pos_" in name:
                emb += n
            elif any(k in name for k in ["attn", "query", "key", "value", "q_proj", "k_proj", "v_proj"]):
                att += n
            elif any(k in name for k in ["ffn", "fc", "mlp", "feedforward", "intermediate"]):
                ffn += n
            else:
                other += n

        bytes_per_param_fp32 = 4
        bytes_per_param_fp16 = 2
        return cls(
            n_params_total=total,
            n_params_trainable=trainable,
            n_params_embedding=emb,
            n_params_attention=att,
            n_params_ffn=ffn,
            n_params_other=other,
            memory_fp32_mb=total * bytes_per_param_fp32 / 1e6,
            memory_fp16_mb=total * bytes_per_param_fp16 / 1e6,
            flops_per_token=6 * total,  # approximate 6N FLOPs per token
            attention_fraction=att / max(total, 1),
            ffn_fraction=ffn / max(total, 1),
        )

    def __str__(self) -> str:
        return (
            f"ModelSizeProfile:\n"
            f"  Total params: {self.n_params_total:,}\n"
            f"  Trainable:    {self.n_params_trainable:,}\n"
            f"  Embedding:    {self.n_params_embedding:,} ({100*self.n_params_embedding/max(self.n_params_total,1):.1f}%)\n"
            f"  Attention:    {self.n_params_attention:,} ({100*self.attention_fraction:.1f}%)\n"
            f"  FFN:          {self.n_params_ffn:,} ({100*self.ffn_fraction:.1f}%)\n"
            f"  Memory FP32:  {self.memory_fp32_mb:.1f} MB\n"
            f"  Memory FP16:  {self.memory_fp16_mb:.1f} MB\n"
        )


class ChinchillaPredictor:
    """Chinchilla compute-optimal scaling (Hoffmann et al. 2022).

    Predicts optimal model size and token budget given compute budget.
    Chinchilla law: N_opt = C / (2 * D_opt), D_opt = C / (2 * N_opt)
    where C = compute budget in FLOPs.
    """

    # Fitted coefficients from Chinchilla paper
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28
    E = 1.69  # irreducible loss

    @classmethod
    def optimal_allocation(cls, compute_budget: float) -> Dict[str, float]:
        """Given FLOPs budget C, return optimal (N, D) pair."""
        # From Appendix D of Chinchilla: N* = G * sqrt(C), D* = C / (6 * N*)
        # Simplified version: N ~ 0.1 * sqrt(C/6), D ~ 20 * N
        G = (cls.A * cls.alpha / (cls.B * cls.beta)) ** (1 / (cls.alpha + cls.beta))
        N_opt = G * (compute_budget / 6) ** (cls.beta / (cls.alpha + cls.beta))
        D_opt = (cls.A * cls.alpha / (cls.B * cls.beta)) ** (-1 / (cls.alpha + cls.beta)) \
                * (compute_budget / 6) ** (cls.alpha / (cls.alpha + cls.beta))
        L_opt = cls.E + cls.A / N_opt ** cls.alpha + cls.B / D_opt ** cls.beta

        return {
            "n_params": N_opt,
            "n_tokens": D_opt,
            "compute_flops": compute_budget,
            "predicted_loss": L_opt,
            "token_to_param_ratio": D_opt / N_opt,
        }

    @classmethod
    def predict_loss(cls, n_params: float, n_tokens: float) -> float:
        """Predict loss given model size and tokens."""
        return cls.E + cls.A / n_params ** cls.alpha + cls.B / n_tokens ** cls.beta

    @classmethod
    def required_tokens(cls, n_params: float, target_loss: float) -> float:
        """Compute tokens needed to reach target_loss with model of n_params."""
        residual = target_loss - cls.E - cls.A / n_params ** cls.alpha
        if residual <= 0:
            return float("inf")
        return (cls.B / residual) ** (1 / cls.beta)

    @classmethod
    def iso_flop_frontier(
        cls,
        compute_budget: float,
        n_param_range: Tuple[float, float] = (1e6, 1e10),
        n_points: int = 50,
    ) -> Dict[str, List[float]]:
        """Returns (N, L) curve on iso-FLOP frontier."""
        ns = [n_param_range[0] * (n_param_range[1] / n_param_range[0]) ** (i / (n_points - 1))
              for i in range(n_points)]
        results = {"n_params": [], "n_tokens": [], "loss": []}
        for N in ns:
            D = compute_budget / (6 * N)
            if D < 1:
                continue
            L = cls.predict_loss(N, D)
            results["n_params"].append(N)
            results["n_tokens"].append(D)
            results["loss"].append(L)
        return results


class NeuralScalingLawFitter:
    """Fits neural scaling laws to empirical (N, D, L) measurements.

    Supports both Chinchilla-style and power-law fitting.
    """

    def __init__(self, model_type: str = "chinchilla"):
        self.model_type = model_type
        self._params: Optional[Dict[str, float]] = None

    def fit(
        self,
        n_params: List[float],
        n_tokens: List[float],
        losses: List[float],
    ) -> Dict[str, float]:
        """Fit scaling law to observed data."""
        import torch

        N = torch.tensor(n_params, dtype=torch.float64)
        D = torch.tensor(n_tokens, dtype=torch.float64)
        L_obs = torch.tensor(losses, dtype=torch.float64)

        # Parametric fit: L = E + A/N^alpha + B/D^beta
        log_A = nn.Parameter(torch.tensor(6.0))
        log_B = nn.Parameter(torch.tensor(6.0))
        log_E = nn.Parameter(torch.tensor(0.5))
        log_alpha = nn.Parameter(torch.tensor(-1.2))
        log_beta = nn.Parameter(torch.tensor(-1.3))

        opt = torch.optim.Adam([log_A, log_B, log_E, log_alpha, log_beta], lr=0.01)

        for _ in range(500):
            opt.zero_grad()
            A = log_A.exp()
            B = log_B.exp()
            E = log_E.exp()
            alpha = log_alpha.exp()
            beta = log_beta.exp()
            L_pred = E + A / N ** alpha + B / D ** beta
            loss = ((L_pred - L_obs) ** 2).mean()
            loss.backward()
            opt.step()

        self._params = {
            "E": log_E.exp().item(),
            "A": log_A.exp().item(),
            "B": log_B.exp().item(),
            "alpha": log_alpha.exp().item(),
            "beta": log_beta.exp().item(),
            "fit_loss": loss.item(),
        }
        return self._params

    def predict(self, n_params: float, n_tokens: float) -> float:
        if self._params is None:
            raise RuntimeError("Must fit() before predict()")
        p = self._params
        return p["E"] + p["A"] / n_params ** p["alpha"] + p["B"] / n_tokens ** p["beta"]


class ComputeOptimalTrainer:
    """Manages compute budget allocation for optimal training.

    Tracks FLOPs consumed and recommends stopping/continuing.
    """

    def __init__(
        self,
        n_params: int,
        compute_budget_flops: float,
        flops_per_step_estimate: float,
        target_loss: Optional[float] = None,
    ):
        self.n_params = n_params
        self.compute_budget = compute_budget_flops
        self.flops_per_step = flops_per_step_estimate
        self.target_loss = target_loss
        self._steps = 0
        self._flops_used = 0.0
        self._loss_history: List[Tuple[int, float]] = []

        # Chinchilla optimal allocation
        self._optimal = ChinchillaPredictor.optimal_allocation(compute_budget_flops)

    def record_step(self, loss: float, flops_override: Optional[float] = None):
        f = flops_override or self.flops_per_step
        self._flops_used += f
        self._steps += 1
        self._loss_history.append((self._steps, loss))

    def budget_fraction_used(self) -> float:
        return min(self._flops_used / self.compute_budget, 1.0)

    def should_stop(self, current_loss: float) -> bool:
        budget_exceeded = self._flops_used >= self.compute_budget
        if self.target_loss and current_loss <= self.target_loss:
            return True
        return budget_exceeded

    def training_summary(self) -> Dict[str, Any]:
        if not self._loss_history:
            return {}
        first_loss = self._loss_history[0][1]
        last_loss = self._loss_history[-1][1]
        opt_n = self._optimal["n_params"]
        opt_d = self._optimal["n_tokens"]
        actual_tokens = self._steps * 1  # placeholder
        return {
            "steps": self._steps,
            "flops_used": self._flops_used,
            "budget_fraction": self.budget_fraction_used(),
            "first_loss": first_loss,
            "last_loss": last_loss,
            "loss_reduction": first_loss - last_loss,
            "chinchilla_optimal_n": opt_n,
            "chinchilla_optimal_d": opt_d,
            "model_is_optimal": abs(math.log(self.n_params / opt_n)) < 0.3,
        }


class ModelEfficencyAnalyzer:
    """Analyzes model efficiency across multiple dimensions."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.profile = ModelSizeProfile.from_model(model)

    def compute_efficiency(
        self,
        target_loss: float,
        benchmark_n_params: float = 1e9,
        benchmark_loss: float = 2.5,
    ) -> Dict[str, float]:
        """Compute efficiency relative to benchmark."""
        actual_n = self.profile.n_params_total
        # Efficiency: how much smaller is the model vs benchmark at same quality
        size_efficiency = benchmark_n_params / max(actual_n, 1)
        mem_efficiency = self.profile.memory_fp32_mb
        return {
            "size_ratio_vs_benchmark": size_efficiency,
            "memory_fp32_mb": mem_efficiency,
            "memory_fp16_mb": self.profile.memory_fp16_mb,
            "params_per_flop": actual_n / max(self.profile.flops_per_token, 1),
            "attention_fraction": self.profile.attention_fraction,
            "ffn_fraction": self.profile.ffn_fraction,
        }

    def layer_param_breakdown(self) -> Dict[str, int]:
        """Per-module parameter count."""
        breakdown = {}
        for name, module in self.model.named_modules():
            if not list(module.children()):  # leaf module
                n = sum(p.numel() for p in module.parameters(recurse=False))
                if n > 0:
                    breakdown[name] = n
        return dict(sorted(breakdown.items(), key=lambda x: -x[1])[:20])

    def suggest_compression(self, target_compression: float = 0.5) -> Dict[str, Any]:
        """Suggest compression strategies to reach target size."""
        current_n = self.profile.n_params_total
        target_n = current_n * (1 - target_compression)
        suggestions = []

        if self.profile.attention_fraction > 0.4:
            suggestions.append({
                "method": "MultiQueryAttention",
                "description": "Replace MHA with MQA to reduce KV params by ~num_heads",
                "estimated_reduction": 0.2,
            })
        if self.profile.n_params_embedding > 0.3 * current_n:
            suggestions.append({
                "method": "EmbeddingFactorization",
                "description": "Factorize embedding into E=V*H^T (Albert-style)",
                "estimated_reduction": 0.15,
            })
        suggestions.append({
            "method": "LoRA",
            "description": f"Apply LoRA with rank={min(64, int(math.sqrt(current_n/1e6)))}",
            "estimated_reduction": 0.9,  # 90% of params frozen
        })
        suggestions.append({
            "method": "INT8_Quantization",
            "description": "Quantize weights to INT8",
            "estimated_reduction": 0.5,  # 50% memory reduction
        })
        return {
            "current_params": current_n,
            "target_params": target_n,
            "target_compression": target_compression,
            "suggestions": suggestions,
        }


class GradientNoiseScaleMonitor:
    """Monitors gradient noise scale (McCandlish et al. 2018) to determine optimal batch size.

    Gradient noise scale B_simple = Sigma / G, where:
    - Sigma is variance of gradient estimates
    - G is squared gradient norm
    """

    def __init__(self, model: nn.Module, ema_decay: float = 0.99):
        self.model = model
        self.ema_decay = ema_decay
        self._grad_ema: Optional[torch.Tensor] = None
        self._grad_sq_ema: Optional[torch.Tensor] = None
        self._step = 0

    def update(self):
        """Update gradient statistics after backward()."""
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.flatten())
        if not grads:
            return

        g = torch.cat(grads)
        g_sq = g ** 2

        if self._grad_ema is None:
            self._grad_ema = g.clone()
            self._grad_sq_ema = g_sq.clone()
        else:
            self._grad_ema = self.ema_decay * self._grad_ema + (1 - self.ema_decay) * g
            self._grad_sq_ema = self.ema_decay * self._grad_sq_ema + (1 - self.ema_decay) * g_sq

        self._step += 1

    def noise_scale(self) -> float:
        """Returns estimated gradient noise scale."""
        if self._grad_ema is None:
            return float("nan")
        G_sq = (self._grad_ema ** 2).sum().item()
        Sigma = (self._grad_sq_ema - self._grad_ema ** 2).sum().item()
        return Sigma / max(G_sq, 1e-12)

    def optimal_batch_size(self, current_batch_size: int) -> int:
        """Suggests optimal batch size based on gradient noise scale."""
        B_simple = self.noise_scale()
        if math.isnan(B_simple):
            return current_batch_size
        # Optimal batch ≈ B_simple (scale at which gradient noise = signal)
        suggested = int(B_simple)
        # Clip to reasonable range
        return max(1, min(suggested, 16384))


class LearningRateRangeTest:
    """LR range test (Smith 2017) to find optimal learning rate."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_cls=torch.optim.Adam,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        n_steps: int = 100,
    ):
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.n_steps = n_steps
        self._losses: List[Tuple[float, float]] = []

    def run(self, data_iterator, loss_fn: Callable, device: str = "cpu") -> Dict[str, Any]:
        """Run LR range test and return LR-loss curve."""
        optimizer = self.optimizer_cls(self.model.parameters(), lr=self.start_lr)
        lr_multiplier = (self.end_lr / self.start_lr) ** (1 / self.n_steps)

        best_loss = float("inf")
        smoothed_loss = 0.0
        beta = 0.98

        for step, batch in enumerate(data_iterator):
            if step >= self.n_steps:
                break

            # Set LR
            current_lr = self.start_lr * lr_multiplier ** step
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            smoothed_loss = beta * smoothed_loss + (1 - beta) * loss_val
            debiased = smoothed_loss / (1 - beta ** (step + 1))

            self._losses.append((current_lr, debiased))

            if debiased < best_loss:
                best_loss = debiased

            # Stop if loss has diverged
            if debiased > 4 * best_loss:
                break

        # Find optimal LR: steepest descent point
        if len(self._losses) > 2:
            lrs = [l[0] for l in self._losses]
            losses = [l[1] for l in self._losses]
            # Find min loss LR
            min_idx = losses.index(min(losses))
            # Suggest LR ~10x before min
            opt_idx = max(0, min_idx - int(len(losses) * 0.1))
            suggested_lr = lrs[opt_idx]
        else:
            suggested_lr = self.start_lr

        return {
            "lr_curve": self._losses,
            "suggested_lr": suggested_lr,
            "best_loss": best_loss,
        }
'''

append("scaling.py", SCALING_ADD)

# ════════════════════════════════════════════════════════════════════════════════
# 3. tests/test_scaling_extra.py
# ════════════════════════════════════════════════════════════════════════════════
def build_scaling_extra_tests():
    lines = [
        '"""Tests for scaling.py extended components."""',
        "import pytest",
        "import torch",
        "import torch.nn as nn",
        "",
        "class TestModelSizeProfile:",
        "    def test_from_model(self):",
        "        from scaling import ModelSizeProfile",
        "        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))",
        "        profile = ModelSizeProfile.from_model(model)",
        "        assert profile.n_params_total > 0",
        "        assert profile.memory_fp32_mb > 0",
        "",
        "    def test_trainable_le_total(self):",
        "        from scaling import ModelSizeProfile",
        "        model = nn.Linear(64, 64)",
        "        model.weight.requires_grad_(False)",
        "        profile = ModelSizeProfile.from_model(model)",
        "        assert profile.n_params_trainable <= profile.n_params_total",
        "",
        "    def test_str_representation(self):",
        "        from scaling import ModelSizeProfile",
        "        model = nn.Linear(32, 64)",
        "        profile = ModelSizeProfile.from_model(model)",
        "        s = str(profile)",
        "        assert 'params' in s.lower() or 'Memory' in s",
        "",
        "",
        "class TestChinchillaPredictor:",
        "    def test_optimal_allocation_returns_dict(self):",
        "        from scaling import ChinchillaPredictor",
        "        result = ChinchillaPredictor.optimal_allocation(1e21)",
        "        assert 'n_params' in result",
        "        assert 'n_tokens' in result",
        "        assert 'predicted_loss' in result",
        "",
        "    def test_predict_loss_positive(self):",
        "        from scaling import ChinchillaPredictor",
        "        loss = ChinchillaPredictor.predict_loss(1e9, 2e10)",
        "        assert loss > 0",
        "",
        "    def test_more_params_less_loss(self):",
        "        from scaling import ChinchillaPredictor",
        "        L_small = ChinchillaPredictor.predict_loss(1e8, 1e10)",
        "        L_large = ChinchillaPredictor.predict_loss(1e10, 1e10)",
        "        assert L_large < L_small",
        "",
        "    def test_iso_flop_frontier(self):",
        "        from scaling import ChinchillaPredictor",
        "        result = ChinchillaPredictor.iso_flop_frontier(1e21, n_points=10)",
        "        assert len(result['n_params']) > 0",
        "        assert len(result['loss']) == len(result['n_params'])",
        "",
        "    def test_required_tokens(self):",
        "        from scaling import ChinchillaPredictor",
        "        tokens = ChinchillaPredictor.required_tokens(1e9, target_loss=3.0)",
        "        assert tokens > 0",
        "",
        "",
        "class TestComputeOptimalTrainer:",
        "    def test_budget_fraction(self):",
        "        from scaling import ComputeOptimalTrainer",
        "        trainer = ComputeOptimalTrainer(",
        "            n_params=1e9,",
        "            compute_budget_flops=1e21,",
        "            flops_per_step_estimate=1e18,",
        "        )",
        "        trainer.record_step(loss=3.5)",
        "        assert 0 < trainer.budget_fraction_used() < 1",
        "",
        "    def test_should_stop_on_budget(self):",
        "        from scaling import ComputeOptimalTrainer",
        "        trainer = ComputeOptimalTrainer(",
        "            n_params=1e6,",
        "            compute_budget_flops=5.0,",
        "            flops_per_step_estimate=6.0,",
        "        )",
        "        trainer.record_step(loss=3.0)",
        "        assert trainer.should_stop(current_loss=3.0)",
        "",
        "    def test_summary(self):",
        "        from scaling import ComputeOptimalTrainer",
        "        trainer = ComputeOptimalTrainer(1e8, 1e18, 1e15)",
        "        for i in range(10):",
        "            trainer.record_step(loss=3.0 - i * 0.1)",
        "        s = trainer.training_summary()",
        "        assert 'steps' in s",
        "        assert s['steps'] == 10",
        "",
        "",
        "class TestGradientNoiseScaleMonitor:",
        "    def test_update_and_noise_scale(self):",
        "        from scaling import GradientNoiseScaleMonitor",
        "        model = nn.Linear(16, 4)",
        "        monitor = GradientNoiseScaleMonitor(model)",
        "        # Simulate a backward pass",
        "        x = torch.randn(4, 16)",
        "        loss = model(x).sum()",
        "        loss.backward()",
        "        monitor.update()",
        "        ns = monitor.noise_scale()",
        "        assert not (ns != ns)  # not NaN",
        "",
        "    def test_optimal_batch_size(self):",
        "        from scaling import GradientNoiseScaleMonitor",
        "        model = nn.Linear(16, 4)",
        "        monitor = GradientNoiseScaleMonitor(model)",
        "        x = torch.randn(4, 16)",
        "        model(x).sum().backward()",
        "        monitor.update()",
        "        bs = monitor.optimal_batch_size(current_batch_size=32)",
        "        assert isinstance(bs, int)",
        "        assert bs >= 1",
        "",
        "",
        "class TestModelEfficiencyAnalyzer:",
        "    def test_efficiency_metrics(self):",
        "        from scaling import ModelEfficencyAnalyzer",
        "        model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))",
        "        analyzer = ModelEfficencyAnalyzer(model)",
        "        metrics = analyzer.compute_efficiency(target_loss=2.5)",
        "        assert 'memory_fp32_mb' in metrics",
        "",
        "    def test_layer_breakdown(self):",
        "        from scaling import ModelEfficencyAnalyzer",
        "        model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 32))",
        "        analyzer = ModelEfficencyAnalyzer(model)",
        "        breakdown = analyzer.layer_param_breakdown()",
        "        assert len(breakdown) > 0",
        "        assert sum(breakdown.values()) > 0",
        "",
        "    def test_suggest_compression(self):",
        "        from scaling import ModelEfficencyAnalyzer",
        "        model = nn.Embedding(1000, 64)",
        "        analyzer = ModelEfficencyAnalyzer(model)",
        "        suggestions = analyzer.suggest_compression(target_compression=0.5)",
        "        assert 'suggestions' in suggestions",
        "        assert len(suggestions['suggestions']) > 0",
        "",
        "",
        "class TestNeuralScalingLawFitter:",
        "    def test_fit_and_predict(self):",
        "        from scaling import NeuralScalingLawFitter",
        "        fitter = NeuralScalingLawFitter()",
        "        # Synthetic data from Chinchilla",
        "        from scaling import ChinchillaPredictor as CP",
        "        ns = [1e6, 1e7, 1e8, 1e9]",
        "        ds = [2e7, 2e8, 2e9, 2e10]",
        "        ls = [CP.predict_loss(n, d) for n, d in zip(ns, ds)]",
        "        params = fitter.fit(ns, ds, ls)",
        "        assert 'E' in params",
        "        assert 'alpha' in params",
        "        # Test predict",
        "        pred = fitter.predict(1e8, 2e9)",
        "        assert pred > 0",
        "",
        "",
        "@pytest.mark.parametrize('n_params,n_tokens', [",
        "    (1e6, 2e7),",
        "    (1e7, 2e8),",
        "    (1e8, 2e9),",
        "    (1e9, 2e10),",
        "    (7e9, 1.4e11),  # Chinchilla 70B equivalent",
        "])",
        "def test_chinchilla_loss_positive(n_params, n_tokens):",
        "    from scaling import ChinchillaPredictor",
        "    loss = ChinchillaPredictor.predict_loss(n_params, n_tokens)",
        "    assert loss > 0",
        "    assert loss < 10.0",
        "",
    ]
    return "\n".join(lines)

write_new("tests/test_scaling_extra.py", build_scaling_extra_tests())

# ════════════════════════════════════════════════════════════════════════════════
# 4. More content for evaluation.py
# ════════════════════════════════════════════════════════════════════════════════
EVAL_ADD = '''

# ============================================================
# Extended Evaluation Components - Part 2
# ============================================================

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    """Complete backtest result with all performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in periods
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    hit_rate: float
    information_ratio: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    r_squared: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_trades": self.total_trades,
            "hit_rate": self.hit_rate,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


def compute_backtest_result(
    returns: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.02,
) -> BacktestResult:
    """Compute comprehensive backtest result from return series."""
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n == 0:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Basic metrics
    total_return = float(np.prod(1 + r) - 1)
    ann_return = float((1 + total_return) ** (periods_per_year / n) - 1)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year))

    rfr_period = risk_free_rate / periods_per_year
    excess_r = r - rfr_period
    sharpe = float(np.mean(excess_r) / (np.std(excess_r, ddof=1) + 1e-10) * np.sqrt(periods_per_year))

    # Sortino
    downside = r[r < rfr_period] - rfr_period
    sortino_denom = float(np.sqrt(np.mean(downside ** 2) + 1e-10) * np.sqrt(periods_per_year))
    sortino = float(np.mean(excess_r) * periods_per_year / max(sortino_denom, 1e-10))

    # Max drawdown
    cumret = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cumret)
    drawdown = (cumret - running_max) / (running_max + 1e-10)
    max_dd = float(drawdown.min())
    calmar = ann_return / (abs(max_dd) + 1e-10)

    # Max drawdown duration
    in_dd = drawdown < -1e-6
    dd_dur = 0
    max_dd_dur = 0
    for x in in_dd:
        if x:
            dd_dur += 1
            max_dd_dur = max(max_dd_dur, dd_dur)
        else:
            dd_dur = 0

    # Win/loss statistics
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(len(wins) / max(n, 1))
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    profit_factor = float(np.sum(wins) / (abs(np.sum(losses)) + 1e-10))

    # Benchmark metrics
    info_ratio = alpha = beta = r_sq = None
    if benchmark is not None:
        bm = np.asarray(benchmark, dtype=np.float64)
        bm = bm[:n]
        if len(bm) == n and np.std(bm) > 1e-10:
            cov_mat = np.cov(r, bm, ddof=1)
            beta_val = float(cov_mat[0, 1] / (cov_mat[1, 1] + 1e-10))
            alpha_val = float(np.mean(r) - beta_val * np.mean(bm))
            active_r = r - bm
            info_ratio = float(np.mean(active_r) / (np.std(active_r, ddof=1) + 1e-10) * np.sqrt(periods_per_year))
            # R-squared
            r_pred = alpha_val + beta_val * bm
            ss_res = np.sum((r - r_pred) ** 2)
            ss_tot = np.sum((r - np.mean(r)) ** 2)
            r_sq = float(1 - ss_res / (ss_tot + 1e-10))
            alpha = alpha_val
            beta = beta_val

    # Tail metrics
    var_95 = float(np.percentile(r, 5))
    cvar_95 = float(np.mean(r[r <= var_95]))

    # Moments
    skew = float(((r - np.mean(r)) ** 3).mean() / (np.std(r) + 1e-10) ** 3)
    kurt = float(((r - np.mean(r)) ** 4).mean() / (np.std(r) + 1e-10) ** 4 - 3)

    return BacktestResult(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_dur,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_trades=n,
        hit_rate=win_rate,
        information_ratio=info_ratio,
        alpha=alpha,
        beta=beta,
        r_squared=r_sq,
        var_95=var_95,
        cvar_95=cvar_95,
        skewness=skew,
        kurtosis=kurt,
    )


class StrategyEvaluationSuite:
    """Complete evaluation suite for trading strategy comparison."""

    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self._results: Dict[str, BacktestResult] = {}

    def add_strategy(
        self,
        name: str,
        returns: np.ndarray,
        benchmark: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        result = compute_backtest_result(
            returns, benchmark, self.periods_per_year, self.risk_free_rate
        )
        self._results[name] = result
        return result

    def rank_strategies(self, metric: str = "sharpe_ratio") -> List[Tuple[str, float]]:
        """Rank strategies by given metric (higher is better)."""
        ranked = []
        for name, result in self._results.items():
            val = getattr(result, metric, None)
            if val is not None:
                ranked.append((name, float(val)))
        ranked.sort(key=lambda x: -x[1])
        return ranked

    def pairwise_comparison(self, s1: str, s2: str) -> Dict[str, Any]:
        """Compare two strategies head-to-head."""
        r1 = self._results.get(s1)
        r2 = self._results.get(s2)
        if r1 is None or r2 is None:
            return {}
        return {
            "better_sharpe": s1 if r1.sharpe_ratio > r2.sharpe_ratio else s2,
            "better_sortino": s1 if r1.sortino_ratio > r2.sortino_ratio else s2,
            "better_calmar": s1 if r1.calmar_ratio > r2.calmar_ratio else s2,
            "better_max_drawdown": s1 if r1.max_drawdown > r2.max_drawdown else s2,
            "sharpe_diff": r1.sharpe_ratio - r2.sharpe_ratio,
            "return_diff": r1.annualized_return - r2.annualized_return,
            "vol_diff": r1.annualized_volatility - r2.annualized_volatility,
        }

    def full_report(self) -> Dict[str, Dict[str, Any]]:
        return {name: result.to_dict() for name, result in self._results.items()}


class MLModelEvaluator:
    """Evaluates ML model predictions for financial forecasting tasks."""

    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}

    def compute_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Comprehensive regression metrics for return prediction."""
        r = y_true
        p = y_pred
        n = len(r)

        if sample_weight is None:
            w = np.ones(n)
        else:
            w = sample_weight / sample_weight.sum()

        # Weighted metrics
        wmse = float(np.sum(w * (r - p) ** 2))
        wmae = float(np.sum(w * np.abs(r - p)))
        r2 = float(1 - np.sum((r - p) ** 2) / (np.sum((r - np.mean(r)) ** 2) + 1e-10))

        # Correlation metrics
        ic = float(np.corrcoef(r, p)[0, 1]) if np.std(r) > 1e-10 and np.std(p) > 1e-10 else 0.0
        rank_ic = float(np.corrcoef(np.argsort(np.argsort(r)), np.argsort(np.argsort(p)))[0, 1])

        # Directional accuracy
        direction_acc = float(np.mean((r > 0) == (p > 0)))

        # Quantile metrics
        q_returns = []
        n_quantiles = 5
        pred_ranks = np.argsort(np.argsort(p))
        for q in range(n_quantiles):
            q_mask = (pred_ranks >= q * n // n_quantiles) & (pred_ranks < (q + 1) * n // n_quantiles)
            if q_mask.any():
                q_returns.append(float(r[q_mask].mean()))
            else:
                q_returns.append(0.0)

        q_spread = q_returns[-1] - q_returns[0] if len(q_returns) >= 2 else 0.0

        return {
            "mse": wmse,
            "mae": wmae,
            "r2": r2,
            "ic": ic,
            "rank_ic": rank_ic,
            "direction_accuracy": direction_acc,
            "quantile_spread": q_spread,
            "quantile_returns": q_returns,
        }

    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Classification metrics for direction prediction."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        y_true_bin = (y_true > 0).astype(int)

        tp = int(((y_pred == 1) & (y_true_bin == 1)).sum())
        tn = int(((y_pred == 0) & (y_true_bin == 0)).sum())
        fp = int(((y_pred == 1) & (y_true_bin == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_bin == 1)).sum())

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        specificity = tn / max(tn + fp, 1)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

    def information_coefficient_series(
        self,
        forecasts: List[np.ndarray],
        realizations: List[np.ndarray],
    ) -> Dict[str, float]:
        """Compute IC across time periods and return summary stats."""
        ics = []
        for f, r in zip(forecasts, realizations):
            if len(f) > 1 and np.std(f) > 1e-10 and np.std(r) > 1e-10:
                ic = np.corrcoef(f, r)[0, 1]
                ics.append(float(ic))

        if not ics:
            return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "ic_positive_frac": 0.0}

        ic_arr = np.array(ics)
        return {
            "ic_mean": float(np.mean(ic_arr)),
            "ic_std": float(np.std(ic_arr)),
            "icir": float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-10)),
            "ic_positive_frac": float((ic_arr > 0).mean()),
            "ic_t_stat": float(np.mean(ic_arr) / (np.std(ic_arr) / max(np.sqrt(len(ic_arr)), 1))),
            "ic_series": ics,
        }


class PortfolioConstructionEvaluator:
    """Evaluates portfolio construction algorithms."""

    def __init__(self, n_assets: int, risk_free_rate: float = 0.02):
        self.n_assets = n_assets
        self.risk_free_rate = risk_free_rate

    def evaluate_weights(
        self,
        weights: np.ndarray,
        asset_returns: np.ndarray,
        benchmark_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate portfolio weights against asset return matrix.

        weights: (n_assets,) or (T, n_assets)
        asset_returns: (T, n_assets)
        """
        if weights.ndim == 1:
            w = np.broadcast_to(weights[None, :], asset_returns.shape)
        else:
            w = weights[:len(asset_returns)]
            asset_returns = asset_returns[:len(w)]

        port_returns = (w * asset_returns).sum(-1)
        result = compute_backtest_result(port_returns, risk_free_rate=self.risk_free_rate)

        metrics = result.to_dict()

        # Weight-specific metrics
        metrics["weight_herfindahl"] = float((weights.flatten() ** 2).sum())
        metrics["effective_n_assets"] = float(1 / max(metrics["weight_herfindahl"], 1e-10))
        metrics["max_weight"] = float(abs(weights).max())
        metrics["long_exposure"] = float(weights[weights > 0].sum() if (weights > 0).any() else 0.0)
        metrics["short_exposure"] = float(abs(weights[weights < 0].sum()) if (weights < 0).any() else 0.0)
        metrics["gross_exposure"] = metrics["long_exposure"] + metrics["short_exposure"]
        metrics["net_exposure"] = metrics["long_exposure"] - metrics["short_exposure"]

        if benchmark_weights is not None:
            active_w = weights.flatten()[:len(benchmark_weights.flatten())] - benchmark_weights.flatten()
            metrics["tracking_error_ex_ante"] = float(np.sqrt((active_w ** 2).sum()))

        return metrics

    def mean_variance_efficiency(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        gamma: float = 1.0,
    ) -> float:
        """Compute mean-variance utility: E[r] - gamma/2 * Var[r]."""
        mu = float(expected_returns @ weights)
        var = float(weights @ cov_matrix @ weights)
        return mu - gamma / 2 * var

    def turnover_cost(
        self,
        weights_before: np.ndarray,
        weights_after: np.ndarray,
        transaction_cost_bps: float = 10.0,
    ) -> float:
        """Estimate transaction cost from rebalancing."""
        turnover = float(np.abs(weights_after - weights_before).sum() / 2)
        return turnover * transaction_cost_bps / 10000.0
'''

append("evaluation.py", EVAL_ADD)

# Final count
result = subprocess.run(
    ["bash", "-c",
     "find /c/Users/Matthew/srfm-lab/aeternus/lumina -name '*.py' -o -name '*.yaml' | xargs wc -l 2>/dev/null | tail -1"],
    capture_output=True, text=True
)
print("GRAND TOTAL:", result.stdout.strip())
