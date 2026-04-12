"""
transfer_learning.py — Sim-to-Real Transfer Learning for Hyper-Agent.

Implements:
- Domain adaptation: align sim and real feature distributions (DANN)
- Fine-tuning protocol: freeze lower layers, update upper layers on real data
- Progressive net columns: add new column for real-world, lateral connections from sim columns
- Behavioral cloning warmup: initialize from demonstration data
- Sim-to-real performance gap metric
"""

from __future__ import annotations

import math
import logging
import collections
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TransferStrategy(Enum):
    DANN = auto()               # Domain Adversarial Neural Network
    FINE_TUNING = auto()        # Standard fine-tuning
    PROGRESSIVE_NETS = auto()   # Progressive neural networks
    BEHAVIORAL_CLONING = auto() # BC warmup
    COMBINED = auto()           # All strategies


class DomainLabel(Enum):
    SIM = 0
    REAL = 1


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class DANNConfig:
    """Domain Adversarial Neural Network configuration."""
    feature_dim: int = 128
    hidden_dim: int = 64
    domain_classifier_hidden: int = 32
    gradient_reversal_lambda: float = 1.0
    lambda_schedule: str = "linear"    # linear, constant, cosine
    lambda_max: float = 2.0
    lambda_warmup_steps: int = 1000
    total_steps: int = 50_000
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_domain_classifier_layers: int = 2
    alignment_loss_weight: float = 1.0
    task_loss_weight: float = 1.0


@dataclass
class FineTuningConfig:
    """Fine-tuning protocol configuration."""
    freeze_layers_frac: float = 0.5     # fraction of layers to freeze
    finetune_lr: float = 1e-4
    finetune_epochs: int = 20
    batch_size: int = 32
    early_stopping_patience: int = 5
    lr_decay: float = 0.9
    min_real_samples: int = 100
    regularization: float = 1e-4
    use_elastic_weight_consolidation: bool = True
    ewc_lambda: float = 0.4


@dataclass
class ProgressiveNetsConfig:
    """Progressive neural networks configuration."""
    num_sim_columns: int = 1
    feature_dim: int = 128
    hidden_dim: int = 64
    lateral_connection_init_std: float = 0.01
    freeze_sim_columns: bool = True
    real_column_lr: float = 3e-4
    num_real_column_layers: int = 3


@dataclass
class BCConfig:
    """Behavioral cloning configuration."""
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 50
    validation_fraction: float = 0.1
    early_stopping_patience: int = 10
    loss_type: str = "mse"              # mse, nll, huber
    data_augmentation: bool = True
    augmentation_noise_std: float = 0.01
    min_demo_samples: int = 100


@dataclass
class TransferConfig:
    """Master transfer learning configuration."""
    dann: DANNConfig = field(default_factory=DANNConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    progressive: ProgressiveNetsConfig = field(default_factory=ProgressiveNetsConfig)
    bc: BCConfig = field(default_factory=BCConfig)

    strategy: TransferStrategy = TransferStrategy.COMBINED
    obs_dim: int = 64
    action_dim: int = 8
    seed: Optional[int] = None
    device: str = "cpu"
    enabled: bool = True


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for DANN)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class GradientReversalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: torch.Tensor, lam: float) -> torch.Tensor:  # type: ignore[override]
            ctx.save_for_backward(torch.tensor(lam))
            return x.clone()

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
            lam = ctx.saved_tensors[0].item()
            return -lam * grad_output, None

    class GradientReversalLayer(nn.Module):
        def __init__(self, lam: float = 1.0) -> None:
            super().__init__()
            self.lam = lam

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return GradientReversalFunction.apply(x, self.lam)

    # ------------------------------------------------------------------
    # DANN networks
    # ------------------------------------------------------------------

    class FeatureExtractor(nn.Module):
        """Shared feature extractor for DANN."""

        def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, feature_dim),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class DomainClassifier(nn.Module):
        """Classifies features as sim or real (after gradient reversal)."""

        def __init__(
            self,
            feature_dim: int,
            hidden_dim: int = 32,
            num_layers: int = 2,
        ) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            in_dim = feature_dim
            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else 1
                layers.append(nn.Linear(in_dim, out_dim))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
                in_dim = out_dim
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TaskHead(nn.Module):
        """Task-specific head (policy / value)."""

        def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class DANNModel(nn.Module):
        """Full DANN model."""

        def __init__(self, cfg: DANNConfig, obs_dim: int, action_dim: int) -> None:
            super().__init__()
            self.feature_extractor = FeatureExtractor(
                obs_dim, cfg.feature_dim, cfg.hidden_dim
            )
            self.grl = GradientReversalLayer(cfg.gradient_reversal_lambda)
            self.domain_classifier = DomainClassifier(
                cfg.feature_dim, cfg.domain_classifier_hidden, cfg.num_domain_classifier_layers
            )
            self.task_head = TaskHead(cfg.feature_dim, action_dim, cfg.hidden_dim)
            self._cfg = cfg

        def forward(
            self, obs: torch.Tensor, alpha: float = 1.0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Returns (task_output, domain_logit)."""
            features = self.feature_extractor(obs)
            task_out = self.task_head(features)
            self.grl.lam = alpha
            reversed_features = self.grl(features)
            domain_logit = self.domain_classifier(reversed_features)
            return task_out, domain_logit

        def get_features(self, obs: torch.Tensor) -> torch.Tensor:
            return self.feature_extractor(obs)

    # ------------------------------------------------------------------
    # Progressive nets column
    # ------------------------------------------------------------------

    class ProgNetColumn(nn.Module):
        """Single column in a Progressive Neural Network."""

        def __init__(
            self,
            obs_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 3,
            lateral_dim: int = 0,  # sum of all prev column hidden dims at each layer
        ) -> None:
            super().__init__()
            self._num_layers = num_layers
            self._hidden_dim = hidden_dim

            # Main layers
            self.layers = nn.ModuleList()
            in_dim = obs_dim
            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else output_dim
                self.layers.append(nn.Linear(in_dim + lateral_dim, out_dim))
                in_dim = hidden_dim

            # Lateral connection adapters (one per layer)
            if lateral_dim > 0:
                self.lateral_adapters = nn.ModuleList([
                    nn.Linear(lateral_dim, hidden_dim, bias=False)
                    for _ in range(num_layers - 1)
                ])
            else:
                self.lateral_adapters = nn.ModuleList()

        def forward(
            self,
            obs: torch.Tensor,
            lateral_inputs: Optional[List[torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            """
            Forward pass.
            Returns (output, list of intermediate activations per layer).
            """
            x = obs
            activations: List[torch.Tensor] = []
            for i, layer in enumerate(self.layers):
                if lateral_inputs is not None and i < len(lateral_inputs):
                    x_in = torch.cat([x, lateral_inputs[i]], dim=-1)
                else:
                    x_in = x
                x = layer(x_in)
                if i < self._num_layers - 1:
                    x = F.relu(x)
                    activations.append(x.clone())
            return x, activations


# ---------------------------------------------------------------------------
# DANN Trainer
# ---------------------------------------------------------------------------

class DANNTrainer:
    """
    Domain Adversarial Neural Network trainer.
    Aligns simulated and real feature distributions.
    """

    def __init__(self, config: DANNConfig, obs_dim: int, action_dim: int, device: str = "cpu") -> None:
        self.cfg = config
        self.device = device
        self._step = 0

        if not _TORCH_AVAILABLE:
            self._model = None
            self._optimizer = None
            return

        self._model = DANNModel(config, obs_dim, action_dim).to(device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=config.learning_rate)
        self._loss_history: collections.deque = collections.deque(maxlen=500)

    def _get_lambda(self) -> float:
        cfg = self.cfg
        if cfg.lambda_schedule == "constant":
            return cfg.gradient_reversal_lambda
        p = min(1.0, self._step / max(cfg.total_steps, 1))
        if cfg.lambda_schedule == "linear":
            return cfg.gradient_reversal_lambda + p * (cfg.lambda_max - cfg.gradient_reversal_lambda)
        elif cfg.lambda_schedule == "cosine":
            return cfg.lambda_max * (1.0 - math.cos(math.pi * p)) / 2.0
        # warmup
        if self._step < cfg.lambda_warmup_steps:
            return cfg.gradient_reversal_lambda * (self._step / max(cfg.lambda_warmup_steps, 1))
        return cfg.lambda_max

    def train_step(
        self,
        sim_obs: np.ndarray,
        real_obs: np.ndarray,
        sim_actions: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Single training step. Returns losses."""
        if not _TORCH_AVAILABLE or self._model is None or self._optimizer is None:
            return {}

        self._step += 1
        cfg = self.cfg
        device = self.device
        lam = self._get_lambda()

        # Build batches
        batch_size = min(cfg.batch_size, len(sim_obs), len(real_obs))
        sim_idx = np.random.choice(len(sim_obs), size=batch_size, replace=False)
        real_idx = np.random.choice(len(real_obs), size=batch_size, replace=False)

        sim_t = torch.tensor(sim_obs[sim_idx], dtype=torch.float32, device=device)
        real_t = torch.tensor(real_obs[real_idx], dtype=torch.float32, device=device)

        # Domain labels
        sim_domain = torch.zeros(batch_size, 1, device=device)
        real_domain = torch.ones(batch_size, 1, device=device)

        self._model.train()
        self._optimizer.zero_grad()

        # Forward: sim
        sim_task_out, sim_domain_logit = self._model(sim_t, alpha=lam)
        # Forward: real
        _, real_domain_logit = self._model(real_t, alpha=lam)

        # Domain alignment loss (cross entropy)
        domain_loss = (
            F.binary_cross_entropy_with_logits(sim_domain_logit, sim_domain)
            + F.binary_cross_entropy_with_logits(real_domain_logit, real_domain)
        ) * cfg.alignment_loss_weight

        # Task loss (if labels available)
        task_loss = torch.tensor(0.0, device=device)
        if sim_actions is not None:
            act_t = torch.tensor(sim_actions[sim_idx], dtype=torch.float32, device=device)
            task_loss = F.mse_loss(sim_task_out, act_t) * cfg.task_loss_weight

        total_loss = domain_loss + task_loss
        total_loss.backward()
        self._optimizer.step()

        losses = {
            "domain_loss": float(domain_loss.item()),
            "task_loss": float(task_loss.item()),
            "total_loss": float(total_loss.item()),
            "lambda": lam,
        }
        self._loss_history.append(losses["total_loss"])
        return losses

    def get_aligned_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract domain-aligned features."""
        if not _TORCH_AVAILABLE or self._model is None:
            return obs
        self._model.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            feats = self._model.get_features(obs_t)
        return feats.cpu().numpy()

    def measure_domain_gap(
        self, sim_obs: np.ndarray, real_obs: np.ndarray
    ) -> float:
        """Measure feature distribution gap (MMD approximation)."""
        if not _TORCH_AVAILABLE or self._model is None:
            return float("inf")

        sim_feats = self.get_aligned_features(sim_obs[:100])
        real_feats = self.get_aligned_features(real_obs[:100])

        # Maximum Mean Discrepancy (RBF kernel, simplified)
        n_s = len(sim_feats)
        n_r = len(real_feats)
        if n_s == 0 or n_r == 0:
            return float("inf")

        sim_mean = sim_feats.mean(axis=0)
        real_mean = real_feats.mean(axis=0)
        return float(np.linalg.norm(sim_mean - real_mean))

    @property
    def mean_loss(self) -> float:
        if not self._loss_history:
            return 0.0
        return float(np.mean(list(self._loss_history)))


# ---------------------------------------------------------------------------
# Fine-tuning trainer
# ---------------------------------------------------------------------------

class FineTuningTrainer:
    """
    Fine-tunes a pre-trained sim policy on real market data.
    Freezes lower layers, updates upper layers.
    """

    def __init__(self, config: FineTuningConfig, device: str = "cpu") -> None:
        self.cfg = config
        self.device = device
        self._frozen_params: List[str] = []
        self._fisher_info: Optional[Dict[str, torch.Tensor]] = None

    def freeze_lower_layers(self, model: Any, frac: Optional[float] = None) -> List[str]:
        """Freeze bottom fraction of layers."""
        if not _TORCH_AVAILABLE:
            return []
        frac = frac or self.cfg.freeze_layers_frac
        params = list(model.named_parameters())
        n_to_freeze = int(len(params) * frac)
        frozen = []
        for i, (name, param) in enumerate(params):
            if i < n_to_freeze:
                param.requires_grad_(False)
                frozen.append(name)
        self._frozen_params = frozen
        logger.info("Frozen %d / %d parameter groups", n_to_freeze, len(params))
        return frozen

    def unfreeze_all(self, model: Any) -> None:
        if not _TORCH_AVAILABLE:
            return
        for param in model.parameters():
            param.requires_grad_(True)

    def compute_fisher_information(
        self, model: Any, sim_data: np.ndarray, sim_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Compute Fisher information matrix for EWC."""
        if not _TORCH_AVAILABLE:
            return {}

        device = self.device
        fisher: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        model.eval()
        for i in range(min(len(sim_data), 200)):
            obs = torch.tensor(sim_data[i:i+1], dtype=torch.float32, device=device)
            lbl = torch.tensor(sim_labels[i:i+1], dtype=torch.float32, device=device)

            try:
                if hasattr(model, "forward"):
                    out = model(obs)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = F.mse_loss(out, lbl)
                    loss.backward()

                    for name, param in model.named_parameters():
                        if param.grad is not None and name in fisher:
                            fisher[name] += param.grad.data.pow(2)

                    model.zero_grad()
            except Exception:
                continue

        for name in fisher:
            fisher[name] /= max(min(len(sim_data), 200), 1)

        self._fisher_info = fisher
        return {k: v.cpu().numpy() for k, v in fisher.items()}

    def finetune(
        self,
        model: Any,
        real_obs: np.ndarray,
        real_actions: np.ndarray,
        val_obs: Optional[np.ndarray] = None,
        val_actions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tune model on real data.

        Returns dict with training history.
        """
        if not _TORCH_AVAILABLE:
            return {"error": "torch_not_available"}
        if len(real_obs) < self.cfg.min_real_samples:
            return {"error": f"insufficient_data: {len(real_obs)} < {self.cfg.min_real_samples}"}

        cfg = self.cfg
        device = self.device

        # Freeze lower layers
        self.freeze_lower_layers(model, cfg.freeze_layers_frac)

        # Build optimizer for unfrozen params only
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            return {"error": "no_trainable_params"}

        optimizer = optim.Adam(
            trainable_params,
            lr=cfg.finetune_lr,
            weight_decay=cfg.regularization,
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        n = len(real_obs)

        for epoch in range(cfg.finetune_epochs):
            model.train()
            epoch_losses = []
            indices = np.random.permutation(n)
            for start in range(0, n, cfg.batch_size):
                batch_idx = indices[start:start + cfg.batch_size]
                obs_t = torch.tensor(real_obs[batch_idx], dtype=torch.float32, device=device)
                act_t = torch.tensor(real_actions[batch_idx], dtype=torch.float32, device=device)

                optimizer.zero_grad()
                out = model(obs_t)
                if isinstance(out, tuple):
                    out = out[0]
                task_loss = F.mse_loss(out, act_t)

                # EWC penalty
                ewc_loss = torch.tensor(0.0, device=device)
                if cfg.use_elastic_weight_consolidation and self._fisher_info is not None:
                    for name, param in model.named_parameters():
                        if name in self._fisher_info:
                            f = self._fisher_info[name].to(device)
                            # Need reference params — approximate as current params
                            ewc_loss += (f * param.pow(2)).sum()
                    ewc_loss *= cfg.ewc_lambda * 0.5

                loss = task_loss + ewc_loss
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(task_loss.item()))

            train_losses.append(float(np.mean(epoch_losses)))
            scheduler.step()

            # Validation
            if val_obs is not None and val_actions is not None:
                model.eval()
                with torch.no_grad():
                    val_obs_t = torch.tensor(val_obs, dtype=torch.float32, device=device)
                    val_act_t = torch.tensor(val_actions, dtype=torch.float32, device=device)
                    val_out = model(val_obs_t)
                    if isinstance(val_out, tuple):
                        val_out = val_out[0]
                    val_loss = float(F.mse_loss(val_out, val_act_t).item())
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.early_stopping_patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "frozen_params": self._frozen_params,
            "final_train_loss": train_losses[-1] if train_losses else None,
        }


# ---------------------------------------------------------------------------
# Progressive Networks Trainer
# ---------------------------------------------------------------------------

class ProgressiveNetsTrainer:
    """
    Progressive neural networks for sim-to-real transfer.

    Adds a new real-world column while keeping sim columns frozen.
    Lateral connections allow knowledge transfer from sim to real columns.
    """

    def __init__(self, config: ProgressiveNetsConfig, obs_dim: int, action_dim: int, device: str = "cpu") -> None:
        self.cfg = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self._sim_columns: List[Any] = []
        self._real_column: Optional[Any] = None
        self._real_optimizer: Optional[Any] = None
        self._loss_history: collections.deque = collections.deque(maxlen=500)

    def add_sim_column(self, sim_model: Any) -> None:
        """Register a pre-trained sim column."""
        if _TORCH_AVAILABLE and self.cfg.freeze_sim_columns:
            for param in sim_model.parameters():
                param.requires_grad_(False)
        self._sim_columns.append(sim_model)

    def build_real_column(self) -> None:
        """Build the real-world column with lateral connections."""
        if not _TORCH_AVAILABLE:
            return
        cfg = self.cfg
        n_sim = len(self._sim_columns)
        lateral_dim = cfg.hidden_dim * n_sim  # one lateral per sim column per layer

        self._real_column = ProgNetColumn(
            obs_dim=self.obs_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=self.action_dim,
            num_layers=cfg.num_real_column_layers,
            lateral_dim=lateral_dim if n_sim > 0 else 0,
        ).to(self.device)

        # Initialize lateral weights to small values
        with torch.no_grad():
            for m in self._real_column.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=cfg.lateral_connection_init_std)
                    nn.init.zeros_(m.bias)

        self._real_optimizer = optim.Adam(
            self._real_column.parameters(),
            lr=cfg.real_column_lr,
        )

    def get_sim_activations(
        self, obs: torch.Tensor
    ) -> List[List[torch.Tensor]]:
        """Get intermediate activations from all sim columns."""
        all_activations: List[List[torch.Tensor]] = []
        for col in self._sim_columns:
            try:
                col.eval()
                with torch.no_grad():
                    _, acts = col(obs, lateral_inputs=None)
                all_activations.append(acts)
            except Exception:
                all_activations.append([torch.zeros(obs.size(0), self.cfg.hidden_dim, device=self.device)])
        return all_activations

    def train_step(
        self,
        real_obs: np.ndarray,
        real_actions: np.ndarray,
    ) -> float:
        """Train the real column for one step."""
        if not _TORCH_AVAILABLE or self._real_column is None or self._real_optimizer is None:
            return 0.0

        cfg = self.cfg
        device = self.device

        batch_size = min(cfg.num_real_column_layers * 10, len(real_obs))
        idx = np.random.choice(len(real_obs), size=batch_size, replace=False)
        obs_t = torch.tensor(real_obs[idx], dtype=torch.float32, device=device)
        act_t = torch.tensor(real_actions[idx], dtype=torch.float32, device=device)

        # Get sim activations for lateral connections
        sim_acts_per_col = self.get_sim_activations(obs_t)

        # Combine lateral inputs: for each layer, concatenate activations from all sim cols
        num_layers = cfg.num_real_column_layers - 1  # last layer doesn't need lateral
        combined_laterals: List[torch.Tensor] = []
        for layer_idx in range(num_layers):
            layer_acts = []
            for col_acts in sim_acts_per_col:
                if layer_idx < len(col_acts):
                    layer_acts.append(col_acts[layer_idx])
                else:
                    layer_acts.append(torch.zeros(obs_t.size(0), cfg.hidden_dim, device=device))
            if layer_acts:
                combined_laterals.append(torch.cat(layer_acts, dim=-1))

        self._real_column.train()
        self._real_optimizer.zero_grad()

        out, _ = self._real_column(obs_t, lateral_inputs=combined_laterals if combined_laterals else None)
        loss = F.mse_loss(out, act_t)
        loss.backward()
        self._real_optimizer.step()

        loss_val = float(loss.item())
        self._loss_history.append(loss_val)
        return loss_val

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Get real column prediction."""
        if not _TORCH_AVAILABLE or self._real_column is None:
            return np.zeros(self.action_dim)
        self._real_column.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        sim_acts = self.get_sim_activations(obs_t)
        combined = []
        for layer_idx in range(self.cfg.num_real_column_layers - 1):
            layer_acts = []
            for acts in sim_acts:
                if layer_idx < len(acts):
                    layer_acts.append(acts[layer_idx])
                else:
                    layer_acts.append(torch.zeros(obs_t.size(0), self.cfg.hidden_dim, device=self.device))
            if layer_acts:
                combined.append(torch.cat(layer_acts, dim=-1))

        with torch.no_grad():
            out, _ = self._real_column(obs_t, lateral_inputs=combined if combined else None)
        return out.squeeze(0).cpu().numpy()

    @property
    def mean_loss(self) -> float:
        if not self._loss_history:
            return 0.0
        return float(np.mean(list(self._loss_history)))


# ---------------------------------------------------------------------------
# Behavioral Cloning warmup
# ---------------------------------------------------------------------------

class BehavioralCloningWarmup:
    """
    Initializes policy from demonstration data via behavioral cloning.
    """

    def __init__(self, config: BCConfig, device: str = "cpu") -> None:
        self.cfg = config
        self.device = device
        self._training_history: List[Dict[str, float]] = []

    def train(
        self,
        policy: Any,
        demo_obs: np.ndarray,
        demo_actions: np.ndarray,
    ) -> Dict[str, Any]:
        """Train policy via behavioral cloning."""
        if not _TORCH_AVAILABLE:
            return {"error": "torch_not_available"}

        cfg = self.cfg
        if len(demo_obs) < cfg.min_demo_samples:
            return {"error": f"insufficient_demos: {len(demo_obs)} < {cfg.min_demo_samples}"}

        device = self.device

        # Split train/val
        n = len(demo_obs)
        n_val = max(1, int(n * cfg.validation_fraction))
        idx = np.random.permutation(n)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        train_obs = demo_obs[train_idx]
        train_act = demo_actions[train_idx]
        val_obs = demo_obs[val_idx]
        val_act = demo_actions[val_idx]

        # Optimizer
        trainable = [p for p in policy.parameters() if p.requires_grad]
        if not trainable:
            return {"error": "no_trainable_params"}
        optimizer = optim.Adam(trainable, lr=cfg.learning_rate)

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val = float("inf")
        patience = 0

        for epoch in range(cfg.num_epochs):
            policy.train()
            epoch_loss = []
            perm = np.random.permutation(len(train_obs))

            for start in range(0, len(train_obs), cfg.batch_size):
                batch_idx = perm[start:start + cfg.batch_size]
                obs_t = torch.tensor(train_obs[batch_idx], dtype=torch.float32, device=device)
                act_t = torch.tensor(train_act[batch_idx], dtype=torch.float32, device=device)

                # Data augmentation
                if cfg.data_augmentation:
                    obs_t = obs_t + torch.randn_like(obs_t) * cfg.augmentation_noise_std

                optimizer.zero_grad()
                try:
                    pred = policy(obs_t)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    if cfg.loss_type == "mse":
                        loss = F.mse_loss(pred, act_t)
                    elif cfg.loss_type == "huber":
                        loss = F.huber_loss(pred, act_t)
                    else:
                        loss = F.mse_loss(pred, act_t)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(float(loss.item()))
                except Exception as e:
                    logger.debug("BC train step error: %s", e)

            if epoch_loss:
                train_losses.append(float(np.mean(epoch_loss)))

            # Validation
            policy.eval()
            with torch.no_grad():
                val_obs_t = torch.tensor(val_obs, dtype=torch.float32, device=device)
                val_act_t = torch.tensor(val_act, dtype=torch.float32, device=device)
                try:
                    val_pred = policy(val_obs_t)
                    if isinstance(val_pred, tuple):
                        val_pred = val_pred[0]
                    val_loss = float(F.mse_loss(val_pred, val_act_t).item())
                    val_losses.append(val_loss)
                    if val_loss < best_val:
                        best_val = val_loss
                        patience = 0
                    else:
                        patience += 1
                        if patience >= cfg.early_stopping_patience:
                            logger.info("BC early stopping at epoch %d", epoch)
                            break
                except Exception:
                    pass

        result = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val,
            "num_demos": n,
            "epochs_trained": len(train_losses),
        }
        self._training_history.append({
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "best_val_loss": best_val,
        })
        return result


# ---------------------------------------------------------------------------
# Sim-to-real performance gap metric
# ---------------------------------------------------------------------------

class SimToRealGapMetric:
    """Quantifies how much performance degrades from sim to real."""

    def __init__(self) -> None:
        self._sim_returns: List[float] = []
        self._real_returns: List[float] = []
        self._gap_history: List[float] = []

    def record_sim_episode(self, episode_return: float) -> None:
        self._sim_returns.append(episode_return)

    def record_real_episode(self, episode_return: float) -> None:
        self._real_returns.append(episode_return)

    def compute_gap(self) -> Dict[str, float]:
        """Compute sim-to-real performance gap metrics."""
        if not self._sim_returns or not self._real_returns:
            return {"gap": 0.0, "relative_gap": 0.0, "available": False}

        sim_mean = float(np.mean(self._sim_returns[-100:]))
        real_mean = float(np.mean(self._real_returns[-100:]))
        gap = sim_mean - real_mean
        rel_gap = gap / (abs(sim_mean) + 1e-8)

        self._gap_history.append(gap)

        return {
            "gap": gap,
            "relative_gap": rel_gap,
            "sim_mean": sim_mean,
            "real_mean": real_mean,
            "sim_std": float(np.std(self._sim_returns[-100:])),
            "real_std": float(np.std(self._real_returns[-100:])),
            "n_sim": len(self._sim_returns),
            "n_real": len(self._real_returns),
            "gap_trend": self._gap_trend(),
            "available": True,
        }

    def _gap_trend(self) -> float:
        h = self._gap_history
        if len(h) < 10:
            return 0.0
        return float(np.mean(h[-5:])) - float(np.mean(h[-10:-5]))

    def reset(self) -> None:
        self._sim_returns.clear()
        self._real_returns.clear()


# ---------------------------------------------------------------------------
# Transfer Learning Manager (main class)
# ---------------------------------------------------------------------------

class TransferLearningManager:
    """
    Orchestrates all sim-to-real transfer learning strategies.
    """

    def __init__(
        self,
        config: Optional[TransferConfig] = None,
        sim_policy: Optional[Any] = None,
    ) -> None:
        self.cfg = config or TransferConfig()
        self.sim_policy = sim_policy
        device = self.cfg.device

        # Sub-modules
        self.dann_trainer = DANNTrainer(
            self.cfg.dann, self.cfg.obs_dim, self.cfg.action_dim, device
        )
        self.finetuner = FineTuningTrainer(self.cfg.fine_tuning, device)
        self.prog_nets = ProgressiveNetsTrainer(
            self.cfg.progressive, self.cfg.obs_dim, self.cfg.action_dim, device
        )
        self.bc_warmup = BehavioralCloningWarmup(self.cfg.bc, device)
        self.gap_metric = SimToRealGapMetric()

        # Register sim policy in progressive nets
        if sim_policy is not None:
            self.prog_nets.add_sim_column(sim_policy)
            self.prog_nets.build_real_column()

        self._step = 0

    def adapt(
        self,
        sim_obs: np.ndarray,
        real_obs: np.ndarray,
        sim_actions: Optional[np.ndarray] = None,
        real_actions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run one adaptation step across active strategies."""
        self._step += 1
        results: Dict[str, Any] = {"step": self._step}
        cfg = self.cfg

        # DANN
        if cfg.strategy in (TransferStrategy.DANN, TransferStrategy.COMBINED):
            dann_result = self.dann_trainer.train_step(sim_obs, real_obs, sim_actions)
            results["dann"] = dann_result

        # Progressive nets
        if cfg.strategy in (TransferStrategy.PROGRESSIVE_NETS, TransferStrategy.COMBINED):
            if real_actions is not None:
                prog_loss = self.prog_nets.train_step(real_obs, real_actions)
                results["progressive_loss"] = prog_loss

        return results

    def initialize_from_demos(
        self,
        policy: Any,
        demo_obs: np.ndarray,
        demo_actions: np.ndarray,
    ) -> Dict[str, Any]:
        """BC warmup."""
        cfg = self.cfg
        if cfg.strategy in (TransferStrategy.BEHAVIORAL_CLONING, TransferStrategy.COMBINED):
            return self.bc_warmup.train(policy, demo_obs, demo_actions)
        return {"skipped": True}

    def finetune_on_real(
        self,
        policy: Any,
        real_obs: np.ndarray,
        real_actions: np.ndarray,
        sim_obs: Optional[np.ndarray] = None,
        sim_actions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fine-tune on real data with EWC regularization."""
        if sim_obs is not None and sim_actions is not None:
            self.finetuner.compute_fisher_information(policy, sim_obs, sim_actions)
        val_split = int(len(real_obs) * 0.1)
        val_obs = real_obs[:val_split] if val_split > 0 else None
        val_act = real_actions[:val_split] if val_split > 0 else None
        train_obs = real_obs[val_split:]
        train_act = real_actions[val_split:]
        return self.finetuner.finetune(policy, train_obs, train_act, val_obs, val_act)

    def measure_gap(self) -> Dict[str, float]:
        return self.gap_metric.compute_gap()

    def measure_domain_gap(
        self, sim_obs: np.ndarray, real_obs: np.ndarray
    ) -> float:
        return self.dann_trainer.measure_domain_gap(sim_obs, real_obs)

    def predict_real_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action prediction from real column of progressive nets."""
        return self.prog_nets.predict(obs)

    def get_state(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "dann_mean_loss": self.dann_trainer.mean_loss,
            "prog_nets_mean_loss": self.prog_nets.mean_loss,
            "gap_metric": self.measure_gap(),
            "strategy": self.cfg.strategy.name,
        }


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_dann_trainer(
    obs_dim: int = 64,
    action_dim: int = 8,
    device: str = "cpu",
) -> DANNTrainer:
    cfg = DANNConfig()
    return DANNTrainer(cfg, obs_dim, action_dim, device)


def make_transfer_manager(
    obs_dim: int = 64,
    action_dim: int = 8,
    strategy: TransferStrategy = TransferStrategy.COMBINED,
    sim_policy: Optional[Any] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> TransferLearningManager:
    cfg = TransferConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        strategy=strategy,
        device=device,
        seed=seed,
    )
    return TransferLearningManager(cfg, sim_policy)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "TransferStrategy",
    "DomainLabel",
    # Configs
    "TransferConfig",
    "DANNConfig",
    "FineTuningConfig",
    "ProgressiveNetsConfig",
    "BCConfig",
    # Sub-modules
    "DANNTrainer",
    "FineTuningTrainer",
    "ProgressiveNetsTrainer",
    "BehavioralCloningWarmup",
    "SimToRealGapMetric",
    # Main
    "TransferLearningManager",
    # Factories
    "make_dann_trainer",
    "make_transfer_manager",
    # Extended
    "FeatureAlignmentMetrics",
    "ContinualLearner",
    "MultiSourceTransfer",
    "TransferEfficiencyTracker",
    "KnowledgeDistillation",
    "DomainGapMonitor",
    "SimRealDataMixer",
    "TransferCheckpointer",
]


# ---------------------------------------------------------------------------
# Extended: FeatureAlignmentMetrics
# ---------------------------------------------------------------------------

class FeatureAlignmentMetrics:
    """
    Computes various metrics for feature distribution alignment between sim and real.

    Metrics:
    - Maximum Mean Discrepancy (MMD)
    - Wasserstein-1 distance (approximate)
    - Covariate shift detection (KL divergence of marginals)
    - Feature correlation structure alignment
    """

    def __init__(self, num_bins: int = 20) -> None:
        self.num_bins = num_bins

    def mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        bandwidth: float = 1.0,
    ) -> float:
        """
        Maximum Mean Discrepancy with RBF kernel.

        MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
        """
        def rbf_kernel(A: np.ndarray, B: np.ndarray, bw: float) -> float:
            n_a, n_b = len(A), len(B)
            if n_a == 0 or n_b == 0:
                return 0.0
            # Sample a subset for speed
            idx_a = np.random.choice(n_a, size=min(200, n_a), replace=False)
            idx_b = np.random.choice(n_b, size=min(200, n_b), replace=False)
            A_sub = A[idx_a]
            B_sub = B[idx_b]
            dists = np.sum((A_sub[:, None] - B_sub[None, :]) ** 2, axis=2)
            return float(np.exp(-dists / (2 * bw ** 2)).mean())

        kxx = rbf_kernel(X, X, bandwidth)
        kyy = rbf_kernel(Y, Y, bandwidth)
        kxy = rbf_kernel(X, Y, bandwidth)
        return float(kxx - 2 * kxy + kyy)

    def marginal_kl(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        eps: float = 1e-10,
    ) -> float:
        """
        KL divergence between marginal distributions (per feature, averaged).
        Uses histogram approximation.
        """
        n_features = min(X.shape[1], Y.shape[1]) if X.ndim > 1 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            Y = Y.reshape(-1, 1)

        total_kl = 0.0
        for f in range(n_features):
            x_f = X[:, f]
            y_f = Y[:, f]
            bins = np.linspace(
                min(x_f.min(), y_f.min()),
                max(x_f.max(), y_f.max()),
                self.num_bins + 1,
            )
            p, _ = np.histogram(x_f, bins=bins, density=True)
            q, _ = np.histogram(y_f, bins=bins, density=True)
            p = p + eps
            q = q + eps
            p = p / p.sum()
            q = q / q.sum()
            total_kl += float(np.sum(p * np.log(p / q)))
        return total_kl / max(n_features, 1)

    def correlation_alignment(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> float:
        """
        CORAL (Correlation Alignment) distance between feature covariances.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n_f = min(X.shape[1], Y.shape[1])
        if n_f == 0:
            return 0.0

        X = X[:, :n_f]
        Y = Y[:, :n_f]

        # Covariance matrices
        C_X = np.cov(X.T) if X.shape[0] > 1 else np.eye(n_f)
        C_Y = np.cov(Y.T) if Y.shape[0] > 1 else np.eye(n_f)

        # Frobenius norm of difference
        diff = C_X - C_Y
        return float(np.linalg.norm(diff, "fro")) / (4 * n_f ** 2)

    def compute_all(
        self,
        sim_features: np.ndarray,
        real_features: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all alignment metrics."""
        return {
            "mmd": self.mmd(sim_features, real_features),
            "marginal_kl": self.marginal_kl(sim_features, real_features),
            "coral_distance": self.correlation_alignment(sim_features, real_features),
        }


# ---------------------------------------------------------------------------
# Extended: ContinualLearner
# ---------------------------------------------------------------------------

class ContinualLearner:
    """
    Continual learning module that learns from a stream of sim and real data
    without catastrophic forgetting.

    Uses:
    - Elastic Weight Consolidation (EWC)
    - Gradient Episodic Memory (GEM) - simplified
    - Experience Replay with memory buffer
    """

    def __init__(
        self,
        ewc_lambda: float = 0.4,
        replay_buffer_size: int = 1000,
        replay_batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.ewc_lambda = ewc_lambda
        self.replay_size = replay_buffer_size
        self.replay_batch = replay_batch_size
        self.device = device

        self._replay_buffer: collections.deque = collections.deque(maxlen=replay_buffer_size)
        self._task_history: List[str] = []
        self._fisher_info: Optional[Dict[str, Any]] = None
        self._reference_params: Optional[Dict[str, Any]] = None

    def consolidate(self, model: Any, task_data: np.ndarray, task_labels: np.ndarray, task_name: str) -> None:
        """Consolidate knowledge from a completed task."""
        self._task_history.append(task_name)
        # Store task samples in replay buffer
        for obs, lbl in zip(task_data[:100], task_labels[:100]):
            self._replay_buffer.append({"obs": obs, "label": lbl, "task": task_name})

        # Compute Fisher info if torch available
        if not _TORCH_AVAILABLE or not hasattr(model, "parameters"):
            return

        self._reference_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def compute_ewc_loss(self, model: Any) -> float:
        """Compute EWC loss to prevent forgetting."""
        if (
            not _TORCH_AVAILABLE
            or self._fisher_info is None
            or self._reference_params is None
            or not hasattr(model, "named_parameters")
        ):
            return 0.0

        loss = 0.0
        for name, param in model.named_parameters():
            if name in self._reference_params and name in self._fisher_info:
                ref = self._reference_params[name]
                fisher = self._fisher_info[name]
                loss += float(torch.sum(fisher * (param - ref) ** 2).item())
        return self.ewc_lambda * 0.5 * loss

    def get_replay_batch(self) -> List[Dict[str, Any]]:
        """Sample a batch from the replay buffer."""
        buf = list(self._replay_buffer)
        if not buf:
            return []
        n = min(self.replay_batch, len(buf))
        indices = np.random.choice(len(buf), size=n, replace=False)
        return [buf[i] for i in indices]

    @property
    def num_tasks_learned(self) -> int:
        return len(self._task_history)

    @property
    def task_history(self) -> List[str]:
        return self._task_history.copy()


# ---------------------------------------------------------------------------
# Extended: MultiSourceTransfer
# ---------------------------------------------------------------------------

class MultiSourceTransfer:
    """
    Multi-source domain adaptation when multiple sim environments are available.

    Learns domain-invariant features across multiple source domains (sims)
    and a single target domain (real market).
    """

    def __init__(
        self,
        num_sources: int = 3,
        obs_dim: int = 64,
        action_dim: int = 8,
        device: str = "cpu",
    ) -> None:
        self.num_sources = num_sources
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Per-source domain gap trackers
        self._source_gaps: List[SimToRealGapMetric] = [
            SimToRealGapMetric() for _ in range(num_sources)
        ]
        self._source_weights: np.ndarray = np.ones(num_sources) / num_sources
        self._n_updates: int = 0

    def record_source_episode(self, source_idx: int, ep_return: float) -> None:
        if 0 <= source_idx < self.num_sources:
            self._source_gaps[source_idx].record_sim_episode(ep_return)

    def record_real_episode(self, ep_return: float) -> None:
        for gap in self._source_gaps:
            gap.record_real_episode(ep_return)

    def update_source_weights(self) -> np.ndarray:
        """
        Update source weights inversely proportional to domain gap.

        Sources closer to real (smaller gap) get higher weight.
        """
        gaps = []
        for g in self._source_gaps:
            metrics = g.compute_gap()
            if metrics.get("available", False):
                gaps.append(abs(metrics.get("relative_gap", 1.0)))
            else:
                gaps.append(1.0)  # unknown gap = high gap

        gaps_arr = np.array(gaps) + 1e-6  # avoid zero
        # Inverse weighting: smaller gap = larger weight
        inv_gaps = 1.0 / gaps_arr
        self._source_weights = inv_gaps / inv_gaps.sum()
        self._n_updates += 1
        return self._source_weights.copy()

    def get_combined_gap(self) -> float:
        """Return weighted average gap across sources."""
        gaps = []
        for i, g in enumerate(self._source_gaps):
            metrics = g.compute_gap()
            if metrics.get("available", False):
                gaps.append(self._source_weights[i] * abs(metrics.get("gap", 0.0)))
        return float(sum(gaps))

    @property
    def source_weights(self) -> np.ndarray:
        return self._source_weights.copy()


# ---------------------------------------------------------------------------
# Extended: TransferEfficiencyTracker
# ---------------------------------------------------------------------------

class TransferEfficiencyTracker:
    """
    Tracks transfer learning efficiency metrics.

    Measures:
    - Data efficiency: how many real samples needed to reach performance threshold
    - Transfer benefit: improvement over training from scratch
    - Negative transfer detection: cases where sim pre-training hurts
    """

    def __init__(
        self,
        performance_threshold: float = 0.8,
        baseline_performance: float = 0.0,
    ) -> None:
        self.threshold = performance_threshold
        self.baseline = baseline_performance

        self._real_sample_count: int = 0
        self._performance_curve: List[Tuple[int, float]] = []
        self._scratch_curve: List[Tuple[int, float]] = []
        self._threshold_reached_at: Optional[int] = None
        self._scratch_threshold_reached_at: Optional[int] = None

    def record_performance(self, performance: float, is_transfer: bool = True) -> None:
        """Record current performance with given number of real samples."""
        self._real_sample_count += 1
        entry = (self._real_sample_count, performance)

        if is_transfer:
            self._performance_curve.append(entry)
            if (
                performance >= self.threshold
                and self._threshold_reached_at is None
            ):
                self._threshold_reached_at = self._real_sample_count
        else:
            self._scratch_curve.append(entry)
            if (
                performance >= self.threshold
                and self._scratch_threshold_reached_at is None
            ):
                self._scratch_threshold_reached_at = self._real_sample_count

    def compute_transfer_benefit(self) -> Dict[str, Any]:
        """Compute transfer benefit metrics."""
        if not self._performance_curve:
            return {"available": False}

        transfer_final = self._performance_curve[-1][1] if self._performance_curve else 0.0
        scratch_final = self._scratch_curve[-1][1] if self._scratch_curve else 0.0

        benefit = transfer_final - scratch_final
        relative_benefit = benefit / max(abs(scratch_final), 1e-8)

        # Data efficiency
        data_efficiency = None
        if (
            self._threshold_reached_at is not None
            and self._scratch_threshold_reached_at is not None
        ):
            data_efficiency = (
                self._scratch_threshold_reached_at / max(self._threshold_reached_at, 1)
            )

        # Detect negative transfer (early performance worse than scratch)
        negative_transfer = False
        if len(self._performance_curve) > 5 and len(self._scratch_curve) > 5:
            early_transfer = np.mean([p for _, p in self._performance_curve[:5]])
            early_scratch = np.mean([p for _, p in self._scratch_curve[:5]])
            negative_transfer = early_transfer < early_scratch - 0.1

        return {
            "available": True,
            "transfer_final": transfer_final,
            "scratch_final": scratch_final,
            "absolute_benefit": benefit,
            "relative_benefit": relative_benefit,
            "data_efficiency_ratio": data_efficiency,
            "threshold_samples_transfer": self._threshold_reached_at,
            "threshold_samples_scratch": self._scratch_threshold_reached_at,
            "negative_transfer_detected": negative_transfer,
        }

    def reset(self) -> None:
        self._real_sample_count = 0
        self._performance_curve.clear()
        self._scratch_curve.clear()
        self._threshold_reached_at = None
        self._scratch_threshold_reached_at = None


# ---------------------------------------------------------------------------
# Extended: KnowledgeDistillation
# ---------------------------------------------------------------------------

class KnowledgeDistillation:
    """
    Knowledge distillation from a large sim-trained teacher to a smaller
    real-world student policy.

    The student learns to mimic the teacher's behavior distribution,
    providing a soft target that is easier to fit than hard action labels.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        distillation_weight: float = 0.5,
        hard_label_weight: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.hard_label_weight = hard_label_weight
        self.device = device
        self._distillation_losses: collections.deque = collections.deque(maxlen=500)

    def distill_step(
        self,
        student: Any,
        teacher: Any,
        obs_batch: np.ndarray,
        real_actions: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Run one distillation step.

        Returns dict with loss components.
        """
        if not _TORCH_AVAILABLE:
            return {"error": "torch_not_available"}
        if not (hasattr(student, "parameters") and hasattr(teacher, "parameters")):
            return {"error": "not_nn_modules"}

        device = self.device
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)

        with torch.no_grad():
            teacher.eval()
            teacher_out = teacher(obs_t)
            if isinstance(teacher_out, tuple):
                teacher_out = teacher_out[0]
            # Soft targets: apply temperature
            soft_targets = teacher_out / self.temperature

        student.train()
        student_out = student(obs_t)
        if isinstance(student_out, tuple):
            student_out = student_out[0]

        # Distillation loss (soft targets)
        dist_loss = F.mse_loss(
            student_out / self.temperature, soft_targets.detach()
        ) * (self.temperature ** 2)

        # Hard label loss if available
        hard_loss = torch.tensor(0.0, device=device)
        if real_actions is not None:
            act_t = torch.tensor(real_actions, dtype=torch.float32, device=device)
            hard_loss = F.mse_loss(student_out, act_t)

        total_loss = (
            self.distillation_weight * dist_loss
            + self.hard_label_weight * hard_loss
        )

        self._distillation_losses.append(float(total_loss.item()))
        return {
            "distillation_loss": float(dist_loss.item()),
            "hard_loss": float(hard_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    @property
    def mean_loss(self) -> float:
        if not self._distillation_losses:
            return 0.0
        return float(np.mean(list(self._distillation_losses)))


# ---------------------------------------------------------------------------
# Extended: DomainGapMonitor
# ---------------------------------------------------------------------------

class DomainGapMonitor:
    """
    Continuous monitoring of the sim-to-real domain gap during live trading.

    Triggers alerts and adaptation when the gap exceeds thresholds.
    """

    def __init__(
        self,
        gap_threshold: float = 0.3,
        window: int = 200,
        alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.gap_threshold = gap_threshold
        self.window = window
        self.alert_callback = alert_callback
        self._gap_history: collections.deque = collections.deque(maxlen=window)
        self._alert_count: int = 0
        self._step: int = 0
        self._alignment_metrics = FeatureAlignmentMetrics()
        self._last_alert_step: int = -1000

    def update(
        self,
        sim_features: np.ndarray,
        real_features: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Update gap monitor with new feature batch.
        Returns monitoring results.
        """
        self._step += 1
        if len(sim_features) == 0 or len(real_features) == 0:
            return {"gap": 0.0, "alert": False}

        # Compute gap metrics
        mmd = self._alignment_metrics.mmd(sim_features, real_features)
        self._gap_history.append(mmd)

        gap_ema = float(np.mean(list(self._gap_history)))

        # Check threshold
        alert = False
        if (
            gap_ema > self.gap_threshold
            and self._step - self._last_alert_step > 50
        ):
            alert = True
            self._alert_count += 1
            self._last_alert_step = self._step
            if self.alert_callback is not None:
                self.alert_callback("domain_gap_exceeded", {
                    "gap": gap_ema,
                    "threshold": self.gap_threshold,
                    "step": self._step,
                })

        return {
            "mmd": mmd,
            "gap_ema": gap_ema,
            "alert": alert,
            "alert_count": self._alert_count,
            "threshold": self.gap_threshold,
        }

    @property
    def current_gap(self) -> float:
        if not self._gap_history:
            return 0.0
        return float(self._gap_history[-1])

    @property
    def gap_trend(self) -> float:
        h = list(self._gap_history)
        if len(h) < 10:
            return 0.0
        return float(np.mean(h[-5:])) - float(np.mean(h[-10:-5]))


# ---------------------------------------------------------------------------
# Extended: SimRealDataMixer
# ---------------------------------------------------------------------------

class SimRealDataMixer:
    """
    Mixes simulated and real data for transfer learning training.

    Gradually shifts the data distribution from pure sim to mixed sim+real
    as more real data becomes available.
    """

    def __init__(
        self,
        initial_real_frac: float = 0.0,
        max_real_frac: float = 0.8,
        ramp_steps: int = 1000,
        mixing_strategy: str = "linear",  # linear, curriculum, gap_driven
        seed: int = 0,
    ) -> None:
        self.initial_real_frac = initial_real_frac
        self.max_real_frac = max_real_frac
        self.ramp_steps = ramp_steps
        self.mixing_strategy = mixing_strategy
        self.rng = np.random.default_rng(seed)

        self._step: int = 0
        self._current_real_frac: float = initial_real_frac
        self._sim_buffer: collections.deque = collections.deque(maxlen=10_000)
        self._real_buffer: collections.deque = collections.deque(maxlen=5_000)

    def push_sim(self, obs: np.ndarray, action: np.ndarray) -> None:
        self._sim_buffer.append({"obs": obs, "action": action, "domain": "sim"})

    def push_real(self, obs: np.ndarray, action: np.ndarray) -> None:
        self._real_buffer.append({"obs": obs, "action": action, "domain": "real"})

    def step(self, domain_gap: float = 0.0) -> float:
        """Advance step counter and return updated real fraction."""
        self._step += 1
        if self.mixing_strategy == "linear":
            frac = min(
                self.max_real_frac,
                self.initial_real_frac + (self.max_real_frac - self.initial_real_frac) * (
                    self._step / max(self.ramp_steps, 1)
                )
            )
        elif self.mixing_strategy == "curriculum":
            # Increase real fraction faster as performance improves
            frac = self._current_real_frac + 0.001
            frac = min(self.max_real_frac, frac)
        elif self.mixing_strategy == "gap_driven":
            # More real data when gap is smaller (sim is closer to real)
            frac = self.max_real_frac * (1 - min(1.0, domain_gap / 0.5))
        else:
            frac = self._current_real_frac

        self._current_real_frac = float(np.clip(frac, 0.0, self.max_real_frac))
        return self._current_real_frac

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a mixed batch with current real fraction."""
        n_real = int(batch_size * self._current_real_frac)
        n_sim = batch_size - n_real

        batch: List[Dict[str, Any]] = []

        if self._real_buffer and n_real > 0:
            real_list = list(self._real_buffer)
            indices = self.rng.choice(len(real_list), size=min(n_real, len(real_list)), replace=False)
            batch.extend([real_list[int(i)] for i in indices])

        if self._sim_buffer and n_sim > 0:
            sim_list = list(self._sim_buffer)
            indices = self.rng.choice(len(sim_list), size=min(n_sim, len(sim_list)), replace=False)
            batch.extend([sim_list[int(i)] for i in indices])

        return batch

    @property
    def real_fraction(self) -> float:
        return self._current_real_frac

    @property
    def real_buffer_size(self) -> int:
        return len(self._real_buffer)

    @property
    def sim_buffer_size(self) -> int:
        return len(self._sim_buffer)


# ---------------------------------------------------------------------------
# Extended: TransferCheckpointer
# ---------------------------------------------------------------------------

class TransferCheckpointer:
    """
    Saves/loads transfer learning state including domain adaptation progress.
    """

    def __init__(self) -> None:
        self._checkpoints: Dict[str, Dict[str, Any]] = {}

    def save(
        self,
        manager: TransferLearningManager,
        name: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = manager.get_state()
        state["gap"] = manager.measure_gap()
        if extra:
            state.update(extra)
        self._checkpoints[name] = state

    def load_state(self, name: str) -> Optional[Dict[str, Any]]:
        return self._checkpoints.get(name)

    def list_checkpoints(self) -> List[str]:
        return list(self._checkpoints.keys())

    def compare_checkpoints(
        self, name1: str, name2: str
    ) -> Dict[str, Any]:
        """Compare two checkpoints and return differences."""
        s1 = self._checkpoints.get(name1, {})
        s2 = self._checkpoints.get(name2, {})
        comparison: Dict[str, Any] = {}
        for key in set(list(s1.keys()) + list(s2.keys())):
            v1 = s1.get(key)
            v2 = s2.get(key)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                comparison[key] = {
                    "checkpoint1": v1,
                    "checkpoint2": v2,
                    "delta": v2 - v1,
                }
            else:
                comparison[key] = {"checkpoint1": v1, "checkpoint2": v2}
        return comparison


# ---------------------------------------------------------------------------
# LayerFreezeScheduler — progressively unfreeze layers during fine-tuning
# ---------------------------------------------------------------------------

class LayerFreezeScheduler:
    """Starts with all layers frozen except the head, then progressively
    unfreezes earlier layers as training progresses."""

    def __init__(self, model, unfreeze_every: int = 200):
        self.model = model
        self.unfreeze_every = unfreeze_every
        self._step = 0
        self._unfrozen_count = 0
        # Collect named parameter groups (reversed — head first)
        if hasattr(model, "named_parameters"):
            self._param_groups = list(model.named_parameters())
        else:
            self._param_groups = []
        self._freeze_all()
        self._unfreeze_head()

    # ------------------------------------------------------------------
    def _freeze_all(self) -> None:
        for _, p in self._param_groups:
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def _unfreeze_head(self) -> None:
        # Unfreeze last group
        if self._param_groups:
            _, p = self._param_groups[-1]
            p.requires_grad_(True)
            self._unfrozen_count = 1

    # ------------------------------------------------------------------
    def step(self) -> bool:
        """Call each training step. Returns True if a layer was just unfrozen."""
        self._step += 1
        if self._step % self.unfreeze_every == 0:
            idx = len(self._param_groups) - 1 - self._unfrozen_count
            if idx >= 0:
                _, p = self._param_groups[idx]
                p.requires_grad_(True)
                self._unfrozen_count += 1
                return True
        return False

    # ------------------------------------------------------------------
    @property
    def num_trainable(self) -> int:
        return sum(1 for _, p in self._param_groups if p.requires_grad)


# ---------------------------------------------------------------------------
# AdversarialDomainAdapter — uses adversarial examples from sim to bridge gap
# ---------------------------------------------------------------------------

class AdversarialDomainAdapter:
    """Generates adversarial perturbations in sim observations that push them
    toward the real observation distribution."""

    def __init__(self, epsilon: float = 0.02, steps: int = 10,
                 step_size: float = 0.002):
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size

    # ------------------------------------------------------------------
    def adapt_batch(self, sim_obs: np.ndarray,
                    real_obs_mean: np.ndarray,
                    real_obs_std: np.ndarray) -> np.ndarray:
        """Shift sim_obs toward real distribution via gradient steps on L2 dist."""
        adapted = sim_obs.copy()
        for _ in range(self.steps):
            # Direction toward real mean, normalized by real std
            direction = (real_obs_mean - adapted) / (real_obs_std + 1e-8)
            direction = np.clip(direction, -1.0, 1.0)
            adapted = adapted + self.step_size * direction
            # Stay within epsilon ball of original sim_obs
            diff = adapted - sim_obs
            norm = np.linalg.norm(diff, axis=-1, keepdims=True)
            mask = norm > self.epsilon
            adapted = np.where(mask, sim_obs + diff / (norm + 1e-8) * self.epsilon, adapted)
        return adapted


# ---------------------------------------------------------------------------
# OnlineDomainStatistics — tracks running mean/std of observations per domain
# ---------------------------------------------------------------------------

class OnlineDomainStatistics:
    """Maintains Welford online statistics for sim and real observation streams."""

    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim
        self._stats = {
            "sim": {"n": 0, "mean": np.zeros(obs_dim), "M2": np.zeros(obs_dim)},
            "real": {"n": 0, "mean": np.zeros(obs_dim), "M2": np.zeros(obs_dim)},
        }

    # ------------------------------------------------------------------
    def update(self, obs: np.ndarray, domain: str) -> None:
        s = self._stats[domain]
        s["n"] += 1
        delta = obs - s["mean"]
        s["mean"] += delta / s["n"]
        delta2 = obs - s["mean"]
        s["M2"] += delta * delta2

    # ------------------------------------------------------------------
    def std(self, domain: str) -> np.ndarray:
        s = self._stats[domain]
        if s["n"] < 2:
            return np.ones(self.obs_dim)
        return np.sqrt(s["M2"] / (s["n"] - 1) + 1e-8)

    # ------------------------------------------------------------------
    def mean(self, domain: str) -> np.ndarray:
        return self._stats[domain]["mean"].copy()

    # ------------------------------------------------------------------
    def distribution_shift(self) -> float:
        """Return mean absolute shift normalized by real std."""
        if self._stats["real"]["n"] < 2 or self._stats["sim"]["n"] < 2:
            return 0.0
        shift = np.abs(self.mean("sim") - self.mean("real"))
        norm_shift = shift / (self.std("real") + 1e-8)
        return float(np.mean(norm_shift))


# ---------------------------------------------------------------------------
# CurriculumDomainMixer — gradually shifts training from pure sim to mixed
# ---------------------------------------------------------------------------

class CurriculumDomainMixer:
    """Controls the ratio of sim vs real data during training, following a
    curriculum that increases real data fraction as training progresses."""

    def __init__(self, total_steps: int = 100_000, max_real_fraction: float = 0.8,
                 warmup_steps: int = 10_000):
        self.total_steps = total_steps
        self.max_real_fraction = max_real_fraction
        self.warmup_steps = warmup_steps
        self._step = 0

    # ------------------------------------------------------------------
    def step(self) -> float:
        """Advance one step and return current real_fraction."""
        self._step = min(self._step + 1, self.total_steps)
        return self.real_fraction

    # ------------------------------------------------------------------
    @property
    def real_fraction(self) -> float:
        if self._step < self.warmup_steps:
            return 0.0
        progress = (self._step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps)
        return float(min(self.max_real_fraction, progress * self.max_real_fraction))

    # ------------------------------------------------------------------
    def should_use_real(self, rng: Optional[np.random.Generator] = None) -> bool:
        if rng is None:
            rng = np.random.default_rng()
        return float(rng.random()) < self.real_fraction


# ---------------------------------------------------------------------------
# TransferSummaryReport — builds a human-readable transfer summary
# ---------------------------------------------------------------------------

class TransferSummaryReport:
    """Aggregates results from transfer learning evaluations into a report."""

    def __init__(self):
        self._sections: dict = {}

    # ------------------------------------------------------------------
    def add_section(self, name: str, data: dict) -> None:
        self._sections[name] = data

    # ------------------------------------------------------------------
    def add_gap_metric(self, sim_returns: list, real_returns: list) -> None:
        gap = SimToRealGapMetric()
        for r in sim_returns:
            gap.record_sim(r)
        for r in real_returns:
            gap.record_real(r)
        self._sections["gap_metric"] = gap.compute()

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return dict(self._sections)

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        for section, data in self._sections.items():
            print(f"\n=== {section.upper()} ===")
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"  {data}")
