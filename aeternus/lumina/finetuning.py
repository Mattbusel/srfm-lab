

# ============================================================
# Extended Finetuning Components - Part 2
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import copy
import math


@dataclass
class FinetuningConfig:
    """Configuration for fine-tuning Lumina models."""
    method: str = "full"  # full | lora | prefix | prompt | ia3 | freeze_lower
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    batch_size: int = 32
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0

    # LoRA specific
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Prefix/prompt specific
    prefix_length: int = 10
    n_soft_prompts: int = 20

    # Layer freezing
    freeze_n_layers: int = 0
    freeze_embedding: bool = True

    # Regularization
    l2_lambda: float = 0.0
    ewc_lambda: float = 0.0
    mixout_p: float = 0.0

    # Evaluation
    eval_metric: str = "loss"
    eval_every_n_steps: int = 100


class LayerFreezer:
    """Selectively freezes model layers for efficient fine-tuning."""

    def __init__(self, model: nn.Module, config: FinetuningConfig):
        self.model = model
        self.config = config
        self._frozen_params: List[nn.Parameter] = []

    def freeze_embeddings(self):
        """Freeze all embedding layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                for param in module.parameters():
                    param.requires_grad_(False)
                    self._frozen_params.append(param)

    def freeze_lower_n_layers(self, n: int):
        """Freeze first n transformer layers."""
        # Detect layer structure
        layer_modules = []
        for name, module in self.model.named_modules():
            if hasattr(module, "__class__") and "Layer" in type(module).__name__:
                layer_modules.append((name, module))

        for i, (name, module) in enumerate(layer_modules[:n]):
            for param in module.parameters():
                param.requires_grad_(False)
                self._frozen_params.append(param)

    def freeze_all_except_head(self):
        """Freeze everything except the final classification/regression head."""
        for name, param in self.model.named_parameters():
            if "head" not in name and "classifier" not in name and "output" not in name:
                param.requires_grad_(False)
                self._frozen_params.append(param)

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self._frozen_params:
            param.requires_grad_(True)
        self._frozen_params.clear()

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def num_frozen(self) -> int:
        return sum(p.numel() for p in self._frozen_params)


class MixoutRegularizer(nn.Module):
    """Mixout regularization (Lee et al. 2020): randomly reverts weights to pretrained values.

    Acts as a regularizer to prevent catastrophic forgetting during fine-tuning.
    """

    def __init__(self, model: nn.Module, pretrained_model: nn.Module, p: float = 0.1):
        super().__init__()
        self.model = model
        self.p = p

        # Store pretrained weights
        self._pretrained: Dict[str, torch.Tensor] = {}
        for name, param in pretrained_model.named_parameters():
            self._pretrained[name] = param.data.clone()

    def apply_mixout(self):
        """Randomly revert fraction p of weights to pretrained values."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._pretrained:
                    pretrained_w = self._pretrained[name].to(param.device)
                    mask = torch.bernoulli(torch.full_like(param.data, 1 - self.p))
                    param.data = mask * param.data + (1 - mask) * pretrained_w

    def forward(self, *args, **kwargs):
        if self.training:
            self.apply_mixout()
        return self.model(*args, **kwargs)


class TaskVectorFinetuner:
    """Task vectors (Ilharco et al. 2023): arithmetic on model weights for task combination.

    task_vector = finetuned_weights - pretrained_weights
    combined = pretrained + alpha * task_vector
    """

    def __init__(self, pretrained_model: nn.Module):
        self._pretrained: Dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in pretrained_model.named_parameters()
        }

    def compute_task_vector(self, finetuned_model: nn.Module) -> Dict[str, torch.Tensor]:
        """Compute task vector = finetuned - pretrained."""
        task_vector = {}
        for name, param in finetuned_model.named_parameters():
            if name in self._pretrained:
                task_vector[name] = param.data.clone() - self._pretrained[name]
        return task_vector

    def apply_task_vector(
        self,
        target_model: nn.Module,
        task_vector: Dict[str, torch.Tensor],
        alpha: float = 1.0,
    ):
        """Apply task vector to target model with scaling alpha."""
        with torch.no_grad():
            for name, param in target_model.named_parameters():
                if name in task_vector:
                    param.data = self._pretrained[name].to(param.device) + alpha * task_vector[name].to(param.device)

    def combine_task_vectors(
        self,
        task_vectors: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Combine multiple task vectors with optional weights."""
        if weights is None:
            weights = [1.0 / len(task_vectors)] * len(task_vectors)

        combined = {}
        for tv, w in zip(task_vectors, weights):
            for name, vec in tv.items():
                if name not in combined:
                    combined[name] = torch.zeros_like(vec)
                combined[name] += w * vec
        return combined


class WiSEFT(nn.Module):
    """WiSE-FT (Wortsman et al. 2022): weight-space ensemble of pretrained and fine-tuned.

    Interpolates between pretrained and fine-tuned weights for robust transfer.
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        finetuned_model: nn.Module,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.pretrained = pretrained_model
        self.finetuned = finetuned_model
        self.alpha = alpha

        # Create merged model
        self.merged = copy.deepcopy(finetuned_model)
        self._interpolate(alpha)

    def _interpolate(self, alpha: float):
        """Interpolate weights: alpha * finetuned + (1-alpha) * pretrained."""
        with torch.no_grad():
            pretrained_sd = dict(self.pretrained.named_parameters())
            for name, param in self.merged.named_parameters():
                if name in pretrained_sd:
                    pre = pretrained_sd[name].to(param.device)
                    fine = param.data.clone()
                    param.data = alpha * fine + (1 - alpha) * pre

    def set_alpha(self, alpha: float):
        """Update interpolation coefficient."""
        self.alpha = alpha
        self._interpolate(alpha)

    def forward(self, *args, **kwargs):
        return self.merged(*args, **kwargs)


class FisherMergingFinetuner:
    """Fisher merging (Matena & Raffel 2022): merge models weighted by Fisher information."""

    def __init__(self, models: List[nn.Module], dataloaders: List[Any]):
        self.models = models
        self.dataloaders = dataloaders

    def _estimate_fisher(self, model: nn.Module, dataloader: Any, loss_fn: Callable, n_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Estimate diagonal Fisher information."""
        fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
        model.eval()
        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break
            model.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            count += 1
        for name in fisher:
            fisher[name] /= max(count, 1)
        return fisher

    def merge(
        self,
        loss_fns: List[Callable],
        n_fisher_samples: int = 50,
    ) -> nn.Module:
        """Create Fisher-weighted merged model."""
        fishers = []
        for model, dl, lf in zip(self.models, self.dataloaders, loss_fns):
            fishers.append(self._estimate_fisher(model, dl, lf, n_fisher_samples))

        merged = copy.deepcopy(self.models[0])
        with torch.no_grad():
            for name, param in merged.named_parameters():
                numerator = torch.zeros_like(param)
                denominator = torch.zeros_like(param)
                for model, fisher in zip(self.models, fishers):
                    model_param = dict(model.named_parameters()).get(name)
                    if model_param is not None and name in fisher:
                        F_i = fisher[name]
                        numerator += F_i * model_param.data
                        denominator += F_i
                param.data = numerator / (denominator + 1e-8)

        return merged


class RegExFinetuner:
    """Regularized fine-tuning with multiple regularization options."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        l2_lambda: float = 0.0,
        l1_lambda: float = 0.0,
        elastic_lambda: float = 0.0,
        pretrained_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.elastic_lambda = elastic_lambda
        self.pretrained_params = pretrained_params
        self.step_count = 0

    def regularization_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self.l2_lambda > 0:
                loss = loss + self.l2_lambda * param.norm(2)
            if self.l1_lambda > 0:
                loss = loss + self.l1_lambda * param.norm(1)
            if self.elastic_lambda > 0 and self.pretrained_params and name in self.pretrained_params:
                pre = self.pretrained_params[name].to(param.device)
                loss = loss + self.elastic_lambda * (param - pre).norm(2)
        return loss

    def train_step(self, batch_loss: torch.Tensor) -> float:
        total_loss = batch_loss + self.regularization_loss()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1
        return total_loss.item()
