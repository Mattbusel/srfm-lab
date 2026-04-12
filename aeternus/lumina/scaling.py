

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
            f"ModelSizeProfile:
"
            f"  Total params: {self.n_params_total:,}
"
            f"  Trainable:    {self.n_params_trainable:,}
"
            f"  Embedding:    {self.n_params_embedding:,} ({100*self.n_params_embedding/max(self.n_params_total,1):.1f}%)
"
            f"  Attention:    {self.n_params_attention:,} ({100*self.attention_fraction:.1f}%)
"
            f"  FFN:          {self.n_params_ffn:,} ({100*self.ffn_fraction:.1f}%)
"
            f"  Memory FP32:  {self.memory_fp32_mb:.1f} MB
"
            f"  Memory FP16:  {self.memory_fp16_mb:.1f} MB
"
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
        D_opt = (cls.A * cls.alpha / (cls.B * cls.beta)) ** (-1 / (cls.alpha + cls.beta))                 * (compute_budget / 6) ** (cls.alpha / (cls.alpha + cls.beta))
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
