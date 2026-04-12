

# ============================================================
# Extended Pretraining Components - Part 2
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import copy


@dataclass
class PretrainingConfig:
    """Complete configuration for Lumina pretraining."""
    model_name: str = "lumina-medium"
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 1024
    vocab_size: int = 32000
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    max_steps: int = 100000
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Pretraining objectives
    use_mlm: bool = True
    mlm_probability: float = 0.15
    use_nsp: bool = False
    use_clm: bool = True
    use_contrastive: bool = False
    contrastive_temp: float = 0.07

    # Data
    data_path: str = "./data"
    num_workers: int = 4
    prefetch_factor: int = 2

    # Regularization
    label_smoothing: float = 0.1
    use_mixup: bool = False
    mixup_alpha: float = 0.2

    # Checkpointing
    save_every_n_steps: int = 5000
    eval_every_n_steps: int = 1000
    checkpoint_dir: str = "./checkpoints"

    # Financial-specific
    use_financial_mlm: bool = True
    financial_mask_probability: float = 0.2
    use_return_prediction: bool = True
    return_horizons: List[int] = field(default_factory=lambda: [1, 5, 21])


class MaskedLanguageModeling(nn.Module):
    """MLM objective for financial sequence modeling.

    Randomly masks tokens and trains model to predict them.
    Adapted for financial time series: masks price/volume values.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        d_model: int,
        mask_prob: float = 0.15,
        mask_token_id: int = 0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.ignore_index = ignore_index

        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def _mask_tokens(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT-style masking."""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob, device=input_ids.device)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% mask, 10% random, 10% unchanged
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced

        input_ids = input_ids.clone()
        input_ids[indices_replaced] = self.mask_token_id
        random_words = torch.randint(self.vocab_size, labels.shape, device=input_ids.device)
        input_ids[indices_random] = random_words[indices_random]

        labels[~masked_indices] = self.ignore_index
        return input_ids, labels

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_ids, labels = self._mask_tokens(input_ids)
        hidden = self.model(masked_ids)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        logits = self.mlm_head(hidden)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1),
                               ignore_index=self.ignore_index)
        return loss, logits


class CausalLanguageModeling(nn.Module):
    """CLM (autoregressive) objective for financial sequence modeling."""

    def __init__(self, model: nn.Module, vocab_size: int, d_model: int, label_smoothing: float = 0.0):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CLM loss: predict next token from all previous."""
        # Shift: input = [0..T-2], target = [1..T-1]
        input_ids_shifted = input_ids[:, :-1]
        target_ids = input_ids[:, 1:]

        hidden = self.model(input_ids_shifted)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        logits = self.lm_head(hidden)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            label_smoothing=self.label_smoothing,
        )
        return loss, logits


class FinancialMaskedModeling(nn.Module):
    """Financial-specific masked modeling: masks returns/volumes and predicts them.

    Unlike NLP MLM which masks tokens, this masks continuous financial values
    and requires regression prediction of masked values.
    """

    def __init__(
        self,
        model: nn.Module,
        d_model: int,
        input_dim: int,
        mask_prob: float = 0.2,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.mask_prob = mask_prob
        self.mask_value = mask_value

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_dim),
        )

    def _mask_values(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask a fraction of timestep-features."""
        mask = torch.bernoulli(torch.full(x.shape, self.mask_prob, device=x.device)).bool()
        x_masked = x.clone()
        x_masked[mask] = self.mask_value
        return x_masked, mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked, mask = self._mask_values(x)
        hidden = self.model(x_masked)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        predictions = self.regression_head(hidden)

        # Only compute loss on masked positions
        if mask.any():
            loss = F.mse_loss(predictions[mask], x[mask])
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        return loss, predictions


class TimeSeriesContrastiveLearning(nn.Module):
    """Contrastive learning for time series representations.

    Creates positive pairs by augmenting the same time series differently.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_model: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
        augmentation_strength: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.augmentation_strength = augmentation_strength

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
        )

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random augmentation: jitter + time shift."""
        noise = torch.randn_like(x) * self.augmentation_strength
        x_aug = x + noise
        # Random temporal crop/shift (simple version)
        T = x.shape[1]
        shift = torch.randint(0, max(T // 8, 1), (1,)).item()
        if shift > 0:
            x_aug = torch.roll(x_aug, shifts=shift, dims=1)
        return x_aug

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        pooled = hidden.mean(dim=1) if hidden.dim() == 3 else hidden
        return F.normalize(self.projector(pooled), p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute contrastive loss (NT-Xent)."""
        B = x.shape[0]

        z1 = self._encode(self._augment(x))
        z2 = self._encode(self._augment(x))

        # NT-Xent loss
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=x.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=x.device),
            torch.arange(0, B, device=x.device),
        ])

        loss = F.cross_entropy(sim, labels)
        acc = (sim.argmax(dim=-1) == labels).float().mean().item()

        return loss, {"contrastive_loss": loss.item(), "contrastive_accuracy": acc}


class MultiObjectivePretrainer(nn.Module):
    """Combines multiple pretraining objectives with adaptive loss weighting.

    Supports: MLM, CLM, contrastive, return prediction, financial MLM.
    Uses GradNorm-style dynamic loss weighting.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PretrainingConfig,
        d_model: int,
        input_dim: int = 1,
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.model = model
        self.config = config

        self.objectives = nn.ModuleDict()
        self.loss_weights = nn.ParameterDict()
        self.initial_losses: Dict[str, float] = {}

        if config.use_clm:
            self.objectives["clm"] = CausalLanguageModeling(model, vocab_size, d_model)
            self.loss_weights["clm"] = nn.Parameter(torch.tensor(1.0))

        if config.use_financial_mlm:
            self.objectives["fmlm"] = FinancialMaskedModeling(model, d_model, input_dim)
            self.loss_weights["fmlm"] = nn.Parameter(torch.tensor(1.0))

        if config.use_contrastive:
            self.objectives["contrastive"] = TimeSeriesContrastiveLearning(
                model, d_model, temperature=config.contrastive_temp
            )
            self.loss_weights["contrastive"] = nn.Parameter(torch.tensor(1.0))

        self._step = 0

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, requires_grad=True, device=next(self.model.parameters()).device)
        metrics = {}

        for name, obj in self.objectives.items():
            weight = F.softplus(self.loss_weights[name])

            if name == "clm" and "input_ids" in batch:
                loss, _ = obj(batch["input_ids"])
            elif name == "fmlm" and "features" in batch:
                loss, _ = obj(batch["features"])
            elif name == "contrastive" and "features" in batch:
                loss, extra = obj(batch["features"])
                metrics.update(extra)
            else:
                continue

            metrics[f"{name}_loss"] = loss.item()
            total_loss = total_loss + weight * loss

        self._step += 1
        return total_loss, metrics


class CurriculumPretrainer:
    """Curriculum learning for pretraining: starts with easy samples, progresses to hard."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        difficulty_fn: Callable,
        n_stages: int = 3,
        stage_steps: int = 10000,
    ):
        self.model = model
        self.optimizer = optimizer
        self.difficulty_fn = difficulty_fn
        self.n_stages = n_stages
        self.stage_steps = stage_steps
        self._global_step = 0
        self._stage = 0

    def current_difficulty_threshold(self) -> float:
        """Returns max difficulty allowed at current stage (0 to 1)."""
        return min(1.0, (self._stage + 1) / self.n_stages)

    def filter_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Filter batch to only include samples within current difficulty."""
        if "difficulties" not in batch:
            return batch

        threshold = self.current_difficulty_threshold()
        mask = batch["difficulties"] <= threshold
        if not mask.any():
            return batch  # fallback to full batch

        return {k: v[mask] if v.shape[0] == mask.shape[0] else v for k, v in batch.items()}

    def step(self, batch: Dict[str, torch.Tensor], loss_fn: Callable) -> float:
        filtered = self.filter_batch(batch)
        self.optimizer.zero_grad()
        loss = loss_fn(self.model, filtered)
        loss.backward()
        self.optimizer.step()
        self._global_step += 1

        if self._global_step % self.stage_steps == 0:
            self._stage = min(self._stage + 1, self.n_stages - 1)

        return loss.item()


class DataAugmentationPipeline(nn.Module):
    """Comprehensive data augmentation for financial time series pretraining."""

    def __init__(
        self,
        jitter_std: float = 0.01,
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        time_warp_range: Tuple[float, float] = (0.9, 1.1),
        window_warp_prob: float = 0.3,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.jitter_std = jitter_std
        self.scaling_range = scaling_range
        self.time_warp_range = time_warp_range
        self.window_warp_prob = window_warp_prob
        self.dropout_prob = dropout_prob

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        return x + torch.randn_like(x) * self.jitter_std

    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Random scaling of the entire sequence."""
        lo, hi = self.scaling_range
        scale = (lo + (hi - lo) * torch.rand(x.shape[0], 1, x.shape[-1], device=x.device))
        return x * scale

    def temporal_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out timesteps."""
        mask = torch.bernoulli(torch.full(x.shape[:2], 1 - self.dropout_prob, device=x.device))
        return x * mask.unsqueeze(-1)

    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Warp the magnitude of the time series using a smooth curve."""
        B, T, D = x.shape
        # Generate smooth warp: knots at regular intervals
        knots = torch.rand(B, 4, D, device=x.device) * 0.4 + 0.8  # range [0.8, 1.2]
        # Interpolate to full sequence
        knots_expanded = F.interpolate(knots.transpose(1, 2), size=T, mode='linear', align_corners=True)
        warp = knots_expanded.transpose(1, 2)
        return x * warp

    def window_slice(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly select a window and resize to original length."""
        B, T, D = x.shape
        lo, hi = self.time_warp_range
        window_frac = lo + (hi - lo) * torch.rand(1).item()
        window_size = max(2, int(T * window_frac))
        start = torch.randint(0, max(1, T - window_size), (1,)).item()
        sliced = x[:, start:start + window_size, :]
        # Resize to T
        resized = F.interpolate(sliced.transpose(1, 2), size=T, mode='linear', align_corners=True)
        return resized.transpose(1, 2)

    def forward(self, x: torch.Tensor, augmentations: Optional[List[str]] = None) -> torch.Tensor:
        """Apply augmentation pipeline."""
        if augmentations is None:
            augmentations = ["jitter", "scaling"]

        aug_map = {
            "jitter": self.jitter,
            "scaling": self.scaling,
            "temporal_dropout": self.temporal_dropout,
            "magnitude_warp": self.magnitude_warp,
            "window_slice": self.window_slice,
        }

        result = x
        for name in augmentations:
            if name in aug_map and self.training:
                result = aug_map[name](result)
        return result


class WarmupCosineDecayScheduler:
    """Learning rate scheduler with warmup and cosine decay + final constant phase."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-5,
        final_lr_fraction: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.final_lr_fraction = final_lr_fraction
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def get_lr(self, step: int) -> List[float]:
        lrs = []
        for base_lr in self._base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * step / max(self.warmup_steps, 1)
            else:
                progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                final_lr = max(base_lr * self.final_lr_fraction, self.min_lr)
                lr = final_lr + (base_lr - final_lr) * cosine_factor
            lrs.append(lr)
        return lrs

    def step(self):
        self._step += 1
        lrs = self.get_lr(self._step)
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr

    def state_dict(self) -> Dict[str, Any]:
        return {"step": self._step, "base_lrs": self._base_lrs}

    def load_state_dict(self, state: Dict[str, Any]):
        self._step = state["step"]
        self._base_lrs = state["base_lrs"]


class OnlineLossMonitor:
    """Monitors training loss with exponential moving averages and trend detection."""

    def __init__(self, ema_alpha: float = 0.99, patience: int = 1000, min_improvement: float = 1e-4):
        self.ema_alpha = ema_alpha
        self.patience = patience
        self.min_improvement = min_improvement

        self._ema_loss: Optional[float] = None
        self._raw_losses: List[float] = []
        self._best_loss: float = float("inf")
        self._steps_since_improvement = 0
        self._step = 0

    def update(self, loss: float):
        self._step += 1
        self._raw_losses.append(loss)
        if len(self._raw_losses) > 10000:
            self._raw_losses.pop(0)

        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self.ema_alpha * self._ema_loss + (1 - self.ema_alpha) * loss

        if self._ema_loss < self._best_loss - self.min_improvement:
            self._best_loss = self._ema_loss
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

    def is_plateauing(self) -> bool:
        return self._steps_since_improvement >= self.patience

    def loss_trend(self, window: int = 100) -> float:
        """Returns slope of loss trend over last `window` steps (negative = improving)."""
        if len(self._raw_losses) < window:
            return 0.0
        recent = self._raw_losses[-window:]
        x = list(range(len(recent)))
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(recent) / n
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, recent))
        denominator = sum((xi - mean_x) ** 2 for xi in x) + 1e-10
        return numerator / denominator

    def summary(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "ema_loss": self._ema_loss,
            "best_loss": self._best_loss,
            "steps_since_improvement": self._steps_since_improvement,
            "is_plateauing": self.is_plateauing(),
            "loss_trend": self.loss_trend(),
        }


class FinancialPretrainingDataset:
    """Dataset for financial pretraining with configurable objectives."""

    def __init__(
        self,
        data_path: str,
        seq_len: int = 252,
        stride: int = 21,
        features: Optional[List[str]] = None,
        normalize: bool = True,
        cache_in_memory: bool = True,
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.stride = stride
        self.features = features or ["open", "high", "low", "close", "volume"]
        self.normalize = normalize
        self._cache: Optional[torch.Tensor] = None

        # Stats for normalization
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

    def compute_stats(self, data: torch.Tensor):
        """Compute normalization statistics."""
        self._mean = data.mean(dim=(0, 1))
        self._std = data.std(dim=(0, 1)).clamp(min=1e-8)

    def normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        if self._mean is None:
            self.compute_stats(data)
        return (data - self._mean) / self._std

    def create_windows(self, data: torch.Tensor) -> torch.Tensor:
        """Create sliding windows from time series data."""
        T = data.shape[0]
        windows = []
        for start in range(0, T - self.seq_len + 1, self.stride):
            windows.append(data[start:start + self.seq_len])
        if not windows:
            return data.unsqueeze(0)
        return torch.stack(windows)

    def __len__(self) -> int:
        if self._cache is not None:
            return self._cache.shape[0]
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None:
            return {"features": self._cache[idx]}
        return {"features": torch.zeros(self.seq_len, len(self.features))}


class GradientAccumulatorWithWarmup:
    """Gradient accumulator that properly handles warmup with accumulation steps."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        scaler: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scaler = scaler
        self._micro_step = 0
        self._global_step = 0

    def step(self, loss: torch.Tensor) -> bool:
        """Returns True if optimizer step was taken (every accumulation_steps)."""
        scaled_loss = loss / self.accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._micro_step += 1

        if self._micro_step % self.accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self._global_step += 1
            return True

        return False

    def global_step(self) -> int:
        return self._global_step
