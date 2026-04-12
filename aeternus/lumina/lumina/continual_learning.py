"""
lumina/continual_learning.py

Continual and online learning for Lumina financial foundation model.

Covers:
  - Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting
  - Progressive Neural Networks (PNN) for sequential task learning
  - Experience replay buffer with financial regime awareness
  - Catastrophic forgetting metrics
  - Concept drift detection (DDM, ADWIN, Page-Hinkley)
  - Online LoRA adaptation to new market regimes
  - Streaming model updates with memory constraints
  - Meta-learning (MAML) for fast adaptation
  - Gradient Episodic Memory (GEM)
"""

from __future__ import annotations

import collections
import copy
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market regime definitions
# ---------------------------------------------------------------------------

class MarketRegime:
    """Enum-like class for market regime labels."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"

    ALL_REGIMES = [BULL, BEAR, SIDEWAYS, HIGH_VOL, LOW_VOL, CRISIS, RECOVERY]


@dataclass
class RegimeEpisode:
    """A sequence of data from a single market regime."""
    regime: str
    features: Tensor          # shape: (T, feature_dim)
    labels: Tensor            # shape: (T,)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Elastic Weight Consolidation (EWC)
# ---------------------------------------------------------------------------

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Prevents catastrophic forgetting by adding a quadratic penalty on the
    distance from the old task's optimal parameters, weighted by the
    Fisher Information Matrix.

    Loss = L_new(theta) + lambda/2 * sum_i F_i * (theta_i - theta_old_i)^2
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 400.0,
        online: bool = True,
        gamma: float = 1.0,     # Decay for online EWC (1.0 = no decay)
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.online = online
        self.gamma = gamma

        self._fisher: Optional[Dict[str, Tensor]] = None
        self._optimal_params: Optional[Dict[str, Tensor]] = None
        self._task_count: int = 0

    def compute_fisher(
        self,
        dataloader: DataLoader,
        n_samples: int = 1000,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute diagonal Fisher Information Matrix via empirical Fisher.
        F_i = E[ (d log p(y|x; theta) / d theta_i)^2 ]
        """
        device = device or next(self.model.parameters()).device
        self.model.eval()

        fisher: Dict[str, Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        n_seen = 0
        for batch in dataloader:
            if n_seen >= n_samples:
                break
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
                y = batch[1].to(device) if len(batch) > 1 else None
            elif isinstance(batch, dict):
                x = batch.get("input_ids", batch.get("features")).to(device)
                y = batch.get("labels", None)
                if y is not None:
                    y = y.to(device)
            else:
                continue

            self.model.zero_grad()

            with torch.enable_grad():
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    loss = outputs.get("loss")
                    if loss is None:
                        logits = outputs.get("logits", outputs.get("output"))
                        if y is not None:
                            loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)), y.view(-1)
                            )
                        else:
                            # Sample from model's distribution
                            probs = F.softmax(logits, dim=-1)
                            sampled = torch.multinomial(
                                probs.view(-1, probs.size(-1)), 1
                            ).squeeze()
                            loss = F.nll_loss(
                                F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1),
                                sampled,
                            )
                else:
                    logits = outputs
                    if y is not None:
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    else:
                        loss = logits.mean()

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.clone() ** 2

            n_seen += x.shape[0]

        # Normalize
        for name in fisher:
            fisher[name] /= max(1, n_seen)

        return fisher

    def consolidate(
        self,
        dataloader: DataLoader,
        n_samples: int = 1000,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Store optimal params and Fisher after training on a task.
        Call this AFTER training on a task, BEFORE training on the next task.
        """
        new_fisher = self.compute_fisher(dataloader, n_samples, device)
        new_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        if self.online and self._fisher is not None:
            # Online EWC: accumulate Fisher across tasks
            for name in new_fisher:
                if name in self._fisher:
                    self._fisher[name] = self.gamma * self._fisher[name] + new_fisher[name]
                else:
                    self._fisher[name] = new_fisher[name]
        else:
            self._fisher = new_fisher

        self._optimal_params = new_params
        self._task_count += 1
        logger.info(f"EWC consolidated: task {self._task_count}, lambda={self.lambda_ewc}")

    def penalty(self) -> Tensor:
        """
        Compute EWC penalty term.
        Returns scalar tensor.
        """
        if self._fisher is None or self._optimal_params is None:
            return torch.tensor(0.0)

        device = next(self.model.parameters()).device
        penalty = torch.tensor(0.0, device=device)

        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self._fisher:
                continue
            fisher = self._fisher[name].to(device)
            optimal = self._optimal_params[name].to(device)
            penalty += (fisher * (param - optimal) ** 2).sum()

        return self.lambda_ewc / 2.0 * penalty

    def ewc_loss(self, task_loss: Tensor) -> Tensor:
        """Add EWC penalty to task loss."""
        return task_loss + self.penalty()

    def forgetting_measure(self) -> float:
        """Measure how much current params have drifted from optimal."""
        if self._optimal_params is None:
            return 0.0
        drift = 0.0
        for name, param in self.model.named_parameters():
            if name in self._optimal_params:
                optimal = self._optimal_params[name]
                drift += (param.data - optimal).norm(2).item() ** 2
        return math.sqrt(drift)


# ---------------------------------------------------------------------------
# Progressive Neural Networks
# ---------------------------------------------------------------------------

class PNNColumn(nn.Module):
    """
    One column of a Progressive Neural Network.
    Each column is a small MLP that can receive lateral connections
    from all previous columns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        n_lateral_inputs: int = 0,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_lateral = n_lateral_inputs

        act_fn = nn.ReLU() if activation == "relu" else nn.GELU()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), act_fn])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.column = nn.Sequential(*layers)

        # Lateral connections from previous columns
        if n_lateral_inputs > 0:
            # One lateral adapter per hidden layer per previous column
            self.laterals = nn.ModuleList([
                nn.Linear(n_lateral_inputs, h) for h in hidden_dims
            ])

    def forward(
        self,
        x: Tensor,
        lateral_inputs: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass with optional lateral connections.

        Args:
            x: Input tensor
            lateral_inputs: List of activation tensors from previous columns
                           at each hidden layer.

        Returns:
            output, list of intermediate activations
        """
        activations = []
        h = x
        for i, layer in enumerate(self.column):
            h = layer(h)
            if isinstance(layer, (nn.ReLU, nn.GELU)):
                # Add lateral contributions
                if lateral_inputs and i < len(self.laterals):
                    lat = torch.cat(lateral_inputs, dim=-1)
                    h = h + self.laterals[i](lat)
                activations.append(h)
        return h, activations


class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Networks (Rusu et al., 2016).

    Adds new columns for new tasks without modifying old columns.
    Old columns are frozen; new column receives lateral connections
    from all previous columns.

    Designed for financial regime adaptation: train a new column
    for each new market regime while preserving knowledge of old regimes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.columns: nn.ModuleList = nn.ModuleList()
        self._regime_to_col: Dict[str, int] = {}

    def add_column(self, regime: Optional[str] = None) -> int:
        """Add a new column for a new task/regime."""
        n_existing = len(self.columns)
        # Freeze all existing columns
        for col in self.columns:
            for param in col.parameters():
                param.requires_grad = False

        # Calculate lateral input size: sum of hidden dims of all previous columns
        n_lateral = n_existing * sum(self.hidden_dims) if n_existing > 0 else 0

        new_col = PNNColumn(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            n_lateral_inputs=n_lateral,
            activation=self.activation,
        )
        self.columns.append(new_col)
        col_idx = len(self.columns) - 1

        if regime is not None:
            self._regime_to_col[regime] = col_idx

        logger.info(f"PNN: added column {col_idx} for regime '{regime}', total columns={col_idx+1}")
        return col_idx

    def forward(self, x: Tensor, col_idx: Optional[int] = None) -> Tensor:
        """Forward through the latest (or specified) column."""
        if not self.columns:
            raise RuntimeError("No columns added yet. Call add_column() first.")

        target_col = col_idx if col_idx is not None else len(self.columns) - 1

        # Collect activations from all previous columns
        all_activations: List[Tensor] = []
        for i in range(target_col):
            _, acts = self.columns[i](x, lateral_inputs=None)
            all_activations.extend(acts)

        # Forward through target column with lateral connections
        output, _ = self.columns[target_col](x, lateral_inputs=all_activations if all_activations else None)
        return output

    def forward_for_regime(self, x: Tensor, regime: str) -> Tensor:
        col_idx = self._regime_to_col.get(regime)
        if col_idx is None:
            raise ValueError(f"No column registered for regime '{regime}'.")
        return self.forward(x, col_idx=col_idx)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Only return parameters of the last (active) column."""
        if not self.columns:
            return
        yield from self.columns[-1].parameters()


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

class RegimeAwareReplayBuffer:
    """
    Experience replay buffer that stratifies memory by market regime.

    Ensures balanced representation of all historical regimes
    during continual learning, preventing the model from forgetting
    rare but important regimes (e.g., 2008 crisis).
    """

    def __init__(
        self,
        capacity: int = 10_000,
        n_regimes: int = len(MarketRegime.ALL_REGIMES),
        regime_names: Optional[List[str]] = None,
    ):
        self.capacity = capacity
        self.n_regimes = n_regimes
        self.regime_names = regime_names or MarketRegime.ALL_REGIMES
        self.per_regime_capacity = capacity // max(1, n_regimes)

        self._buffers: Dict[str, List[Tuple[Tensor, Tensor]]] = {
            r: [] for r in self.regime_names
        }
        self._counts: Dict[str, int] = {r: 0 for r in self.regime_names}

    def add(
        self,
        features: Tensor,
        labels: Tensor,
        regime: str,
    ) -> None:
        """Add experience(s) to the regime's buffer."""
        if regime not in self._buffers:
            self._buffers[regime] = []
            self._counts[regime] = 0

        buf = self._buffers[regime]
        if len(buf) >= self.per_regime_capacity:
            # Reservoir sampling: replace random element
            idx = random.randint(0, self._counts[regime])
            if idx < self.per_regime_capacity:
                buf[idx] = (features.detach().cpu(), labels.detach().cpu())
        else:
            buf.append((features.detach().cpu(), labels.detach().cpu()))
        self._counts[regime] += 1

    def add_episode(self, episode: RegimeEpisode) -> None:
        for i in range(len(episode.features)):
            self.add(episode.features[i], episode.labels[i], episode.regime)

    def sample(
        self,
        n_samples: int,
        regimes: Optional[List[str]] = None,
        balanced: bool = True,
    ) -> Tuple[Tensor, Tensor, List[str]]:
        """
        Sample experiences from buffer.

        Args:
            n_samples: Total number of samples.
            regimes: Which regimes to sample from (None = all).
            balanced: If True, sample equally from each regime.

        Returns:
            features, labels, regime_labels
        """
        active_regimes = regimes or [r for r in self.regime_names if self._buffers.get(r)]
        active_regimes = [r for r in active_regimes if self._buffers.get(r)]

        if not active_regimes:
            raise ValueError("No experiences in buffer.")

        all_features, all_labels, all_regime_labels = [], [], []

        if balanced:
            per_regime = max(1, n_samples // len(active_regimes))
            for regime in active_regimes:
                buf = self._buffers[regime]
                if not buf:
                    continue
                samples = random.choices(buf, k=min(per_regime, len(buf)))
                for feat, lbl in samples:
                    all_features.append(feat)
                    all_labels.append(lbl)
                    all_regime_labels.append(regime)
        else:
            all_buf = [(f, l, r) for r in active_regimes for (f, l) in self._buffers[r]]
            samples = random.choices(all_buf, k=min(n_samples, len(all_buf)))
            for feat, lbl, reg in samples:
                all_features.append(feat)
                all_labels.append(lbl)
                all_regime_labels.append(reg)

        features = torch.stack(all_features)
        labels = torch.stack(all_labels)
        return features, labels, all_regime_labels

    def as_dataset(
        self,
        n_samples: int = 1000,
        balanced: bool = True,
    ) -> TensorDataset:
        features, labels, _ = self.sample(n_samples, balanced=balanced)
        return TensorDataset(features, labels)

    def stats(self) -> Dict[str, int]:
        return {r: len(buf) for r, buf in self._buffers.items()}

    def total_size(self) -> int:
        return sum(len(buf) for buf in self._buffers.values())


# ---------------------------------------------------------------------------
# Catastrophic forgetting metrics
# ---------------------------------------------------------------------------

class ForgettingMetrics:
    """
    Measures catastrophic forgetting across tasks/regimes.

    Tracks performance on all previously seen tasks after training
    on new tasks. Reports forgetting as the average accuracy drop.
    """

    def __init__(self):
        # performance[task_id][after_task_j] = metric
        self._performance: Dict[str, Dict[int, float]] = collections.defaultdict(dict)
        self._task_sequence: List[str] = []

    def record(self, task_id: str, after_task: int, metric: float) -> None:
        """Record metric (e.g., accuracy, loss) for task_id measured after training on task after_task."""
        self._performance[task_id][after_task] = metric
        if task_id not in self._task_sequence:
            self._task_sequence.append(task_id)

    def backward_transfer(self) -> Dict[str, float]:
        """
        Backward Transfer (BWT): how much new task training helps/hurts old tasks.
        BWT_i = R_{i,T} - R_{i,i}
        Negative = forgetting; positive = backward knowledge transfer.
        """
        bwt = {}
        T = len(self._task_sequence)
        for task in self._task_sequence:
            perfs = self._performance[task]
            # Performance right after task was trained
            task_idx = self._task_sequence.index(task)
            if task_idx in perfs and T - 1 in perfs and task_idx != T - 1:
                bwt[task] = perfs[T - 1] - perfs[task_idx]
        return bwt

    def forward_transfer(self) -> Dict[str, float]:
        """
        Forward Transfer (FWT): how much past training helps future tasks.
        FWT_i = R_{i,i} - b_i (b_i = random init performance)
        """
        # Without random init baselines, approximate as perf at task training time
        return {task: self._performance[task].get(self._task_sequence.index(task), 0.0)
                for task in self._task_sequence}

    def average_forgetting(self) -> float:
        """Average forgetting across all tasks."""
        bwt = self.backward_transfer()
        if not bwt:
            return 0.0
        forgetting = [-v for v in bwt.values() if v < 0]
        return sum(forgetting) / len(bwt) if bwt else 0.0

    def intransigence(self) -> float:
        """
        Intransigence: inability to learn new tasks.
        Measured as performance deficit on new task vs independent training.
        """
        # Approximate: last task performance after all training
        if not self._task_sequence:
            return 0.0
        last = self._task_sequence[-1]
        T = len(self._task_sequence)
        return self._performance[last].get(T - 1, 0.0)

    def summary(self) -> Dict[str, Any]:
        bwt = self.backward_transfer()
        avg_forget = self.average_forgetting()
        return {
            "avg_forgetting": avg_forget,
            "backward_transfer": bwt,
            "forward_transfer": self.forward_transfer(),
            "task_sequence": self._task_sequence,
            "n_tasks": len(self._task_sequence),
        }


# ---------------------------------------------------------------------------
# Concept drift detection
# ---------------------------------------------------------------------------

class DDMDriftDetector:
    """
    Drift Detection Method (Gama et al., 2004).
    Monitors error rate and detects when it increases significantly.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
    ):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self._n = 0
        self._errors = 0.0
        self._min_p_plus_s = float("inf")
        self.in_warning: bool = False
        self.drift_detected: bool = False

    def update(self, error: float) -> str:
        """
        Update with new error observation.
        Returns: "normal" | "warning" | "drift"
        """
        self._n += 1
        self._errors += error
        p = self._errors / self._n
        s = math.sqrt(p * (1 - p) / self._n)
        p_plus_s = p + s

        if p_plus_s < self._min_p_plus_s:
            self._min_p_plus_s = p_plus_s
            self.in_warning = False
            self.drift_detected = False

        p_min, s_min = self._split_min()

        if p + s >= p_min + self.drift_level * s_min:
            self.drift_detected = True
            self._reset()
            return "drift"
        elif p + s >= p_min + self.warning_level * s_min:
            self.in_warning = True
            return "warning"
        else:
            return "normal"

    def _split_min(self) -> Tuple[float, float]:
        if self._min_p_plus_s == float("inf"):
            return 0.0, 0.0
        # Approximate p_min and s_min from p_min + s_min
        p_min = self._min_p_plus_s / 2
        s_min = p_min
        return p_min, s_min

    def _reset(self) -> None:
        self._n = 0
        self._errors = 0.0
        self._min_p_plus_s = float("inf")
        self.in_warning = False


class ADWINDriftDetector:
    """
    ADWIN (Adaptive Windowing) drift detector (Bifet & Gavalda, 2007).
    Maintains a sliding window and detects mean shifts.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._window: collections.deque = collections.deque()
        self._total: float = 0.0
        self._variance: float = 0.0
        self.drift_detected: bool = False

    def update(self, value: float) -> bool:
        """
        Add new value. Returns True if drift detected.
        """
        self._window.append(value)
        self._total += value
        n = len(self._window)

        if n < 5:
            return False

        # Check all possible splits
        left_sum = 0.0
        for i, v in enumerate(self._window):
            left_sum += v
            left_n = i + 1
            right_n = n - left_n
            if right_n < 1:
                break
            right_sum = self._total - left_sum

            mu_l = left_sum / left_n
            mu_r = right_sum / right_n

            threshold = math.sqrt(
                (1 / (2 * left_n) + 1 / (2 * right_n))
                * math.log(4 * n / self.delta)
            )
            if abs(mu_l - mu_r) >= threshold:
                # Drift: remove old part of window
                self.drift_detected = True
                # Remove left side
                for _ in range(left_n):
                    if self._window:
                        removed = self._window.popleft()
                        self._total -= removed
                return True

        self.drift_detected = False
        return False

    def mean(self) -> float:
        n = len(self._window)
        return self._total / n if n > 0 else 0.0

    def window_size(self) -> int:
        return len(self._window)


class PageHinkleyDriftDetector:
    """
    Page-Hinkley test for detecting upward shifts in mean.
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 0.9999,
    ):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self._sum = 0.0
        self._min_sum = float("inf")
        self._n = 0
        self._mean = 0.0
        self.drift_detected: bool = False

    def update(self, value: float) -> bool:
        self._n += 1
        self._mean = self.alpha * self._mean + (1 - self.alpha) * value
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        if self._sum - self._min_sum > self.lambda_:
            self.drift_detected = True
            self._reset()
            return True

        self.drift_detected = False
        return False

    def _reset(self) -> None:
        self._sum = 0.0
        self._min_sum = float("inf")
        self._n = 0


class RegimeDriftDetector:
    """
    Higher-level regime drift detector for financial time series.
    Monitors multiple signals simultaneously.
    """

    def __init__(self):
        self._vol_detector = ADWINDriftDetector(delta=0.01)
        self._return_detector = PageHinkleyDriftDetector(lambda_=30.0)
        self._error_detector = DDMDriftDetector()
        self._drift_timestamps: List[float] = []
        self._current_regime: str = MarketRegime.UNKNOWN

    def update(
        self,
        realized_vol: float,
        return_val: float,
        prediction_error: float,
    ) -> Dict[str, Any]:
        vol_drift = self._vol_detector.update(realized_vol)
        ret_drift = self._return_detector.update(abs(return_val))
        err_status = self._error_detector.update(prediction_error)

        any_drift = vol_drift or ret_drift or err_status == "drift"
        if any_drift:
            self._drift_timestamps.append(time.time())

        # Classify regime
        new_regime = self._classify_regime(realized_vol, return_val)
        regime_changed = new_regime != self._current_regime
        if regime_changed:
            self._current_regime = new_regime

        return {
            "vol_drift": vol_drift,
            "return_drift": ret_drift,
            "error_status": err_status,
            "drift_detected": any_drift,
            "regime": self._current_regime,
            "regime_changed": regime_changed,
            "n_drifts": len(self._drift_timestamps),
        }

    def _classify_regime(self, vol: float, ret: float) -> str:
        if vol > 0.4:
            return MarketRegime.CRISIS
        elif vol > 0.25:
            return MarketRegime.HIGH_VOL
        elif vol < 0.1:
            if ret > 0.0001:
                return MarketRegime.BULL
            elif ret < -0.0001:
                return MarketRegime.BEAR
            else:
                return MarketRegime.LOW_VOL
        else:
            return MarketRegime.SIDEWAYS


# ---------------------------------------------------------------------------
# Online LoRA adaptation
# ---------------------------------------------------------------------------

class OnlineLoRAAdapter(nn.Module):
    """
    Online LoRA (Low-Rank Adaptation) that can be updated quickly
    when a regime change is detected.

    Each regime has its own LoRA adapter; the base model is frozen.
    Adapters are lightweight and can be swapped instantly.
    """

    def __init__(
        self,
        base_module: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base = base_module
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = base_module.weight.shape[1]
        d_out = base_module.weight.shape[0]

        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.dropout = nn.Dropout(dropout)

        # Regime-specific adapter library
        self._adapters: Dict[str, Tuple[Tensor, Tensor]] = {}

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def save_adapter(self, regime: str) -> None:
        """Save current adapter weights for a regime."""
        self._adapters[regime] = (
            self.lora_A.data.clone(),
            self.lora_B.data.clone(),
        )

    def load_adapter(self, regime: str) -> None:
        """Load adapter weights for a regime."""
        if regime not in self._adapters:
            raise ValueError(f"No adapter saved for regime '{regime}'.")
        lora_A, lora_B = self._adapters[regime]
        self.lora_A.data.copy_(lora_A)
        self.lora_B.data.copy_(lora_B)

    def reset_adapter(self) -> None:
        """Reset to initialization (before regime-specific fine-tuning)."""
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def list_regimes(self) -> List[str]:
        return list(self._adapters.keys())


class OnlineLoRAModel(nn.Module):
    """
    Wraps a transformer model and adds online LoRA adapters to all
    linear layers. Adapters can be regime-switched in O(1).
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_model = model
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "v_proj", "out_proj", "fc1", "fc2"]

        self._lora_modules: Dict[str, OnlineLoRAAdapter] = {}
        self._install_adapters()
        self._current_regime: Optional[str] = None

    def _install_adapters(self) -> None:
        """Replace target linear layers with LoRA-wrapped versions."""
        for name, module in self.base_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in self.target_modules):
                continue
            adapter = OnlineLoRAAdapter(module, rank=self.rank, alpha=self.alpha)
            self._lora_modules[name] = adapter
            # Replace in parent module
            parts = name.split(".")
            parent = self.base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], adapter)
        logger.info(f"Installed {len(self._lora_modules)} LoRA adapters.")

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def save_regime_adapters(self, regime: str) -> None:
        """Save all adapter states for current regime."""
        for name, adapter in self._lora_modules.items():
            adapter.save_adapter(regime)
        self._current_regime = regime
        logger.info(f"Saved adapters for regime '{regime}'.")

    def load_regime_adapters(self, regime: str) -> None:
        """Load all adapter states for a regime."""
        for name, adapter in self._lora_modules.items():
            try:
                adapter.load_adapter(regime)
            except ValueError:
                adapter.reset_adapter()
        self._current_regime = regime
        logger.info(f"Loaded adapters for regime '{regime}'.")

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for adapter in self._lora_modules.values()
                for p in adapter.parameters() if p.requires_grad]

    def regime_adapter_size_bytes(self) -> int:
        total = 0
        for adapter in self._lora_modules.values():
            total += adapter.lora_A.numel() + adapter.lora_B.numel()
        return total * 4  # float32


# ---------------------------------------------------------------------------
# Gradient Episodic Memory (GEM)
# ---------------------------------------------------------------------------

class GEM:
    """
    Gradient Episodic Memory (Lopez-Paz & Ranzato, 2017).

    Stores a small episodic memory per task.
    During training on a new task, projects gradients to ensure
    no increase in loss on previously seen tasks.

    This prevents catastrophic forgetting by enforcing a gradient constraint.
    """

    def __init__(
        self,
        model: nn.Module,
        memory_per_task: int = 100,
        margin: float = 0.5,
    ):
        self.model = model
        self.memory_per_task = memory_per_task
        self.margin = margin
        self._memories: Dict[str, Tuple[Tensor, Tensor]] = {}
        self._grad_dims: List[int] = []
        self._compute_grad_dims()

    def _compute_grad_dims(self) -> None:
        for param in self.model.parameters():
            if param.requires_grad:
                self._grad_dims.append(param.data.numel())

    def _store_grad(self, grads: Tensor, grad_vec: Tensor, task: str) -> None:
        """Store current gradients for a task."""
        offset = 0
        for i, dim in enumerate(self._grad_dims):
            grads[i].fill_(0.0)
            grad_vec[offset:offset + dim].copy_(grads[i].view(-1))
            offset += dim

    def add_task_memory(
        self,
        task_id: str,
        features: Tensor,
        labels: Tensor,
    ) -> None:
        """Store a random subset of task data as episodic memory."""
        n = min(self.memory_per_task, len(features))
        indices = random.sample(range(len(features)), n)
        self._memories[task_id] = (
            features[indices].detach().cpu(),
            labels[indices].detach().cpu(),
        )

    def _get_task_gradient(
        self,
        task_id: str,
        device: torch.device,
        loss_fn: Callable,
    ) -> Tensor:
        """Compute gradient of memory loss for a task."""
        features, labels = self._memories[task_id]
        features = features.to(device)
        labels = labels.to(device)

        self.model.zero_grad()
        outputs = self.model(features)
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = loss_fn(outputs, labels)
        loss.backward()

        grad_vec = torch.zeros(sum(self._grad_dims), device=device)
        offset = 0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_vec[offset: offset + param.grad.numel()].copy_(param.grad.view(-1))
            offset += self._grad_dims[0] if self._grad_dims else 0
        return grad_vec

    def project_gradients(
        self,
        device: torch.device,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """
        Project current gradients to not violate old task constraints.
        Uses quadratic programming (simplified via gradient projection).
        """
        if not self._memories or loss_fn is None:
            return

        # Current gradient
        cur_grad = torch.zeros(sum(self._grad_dims), device=device)
        offset = 0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                cur_grad[offset: offset + param.grad.numel()].copy_(param.grad.view(-1))
            offset += param.grad.numel() if (param.requires_grad and param.grad is not None) else 0

        # Memory gradients
        for task_id in self._memories:
            mem_grad = self._get_task_gradient(task_id, device, loss_fn)
            # Check if violation
            dot = (cur_grad * mem_grad).sum()
            if dot < -self.margin:
                # Project: g_new = g - (g.m / (m.m)) * m
                denom = (mem_grad * mem_grad).sum() + 1e-10
                cur_grad = cur_grad - (dot / denom) * mem_grad

        # Write projected gradient back
        self.model.zero_grad()
        offset = 0
        for param in self.model.parameters():
            if param.requires_grad:
                n = param.numel()
                param.grad = cur_grad[offset: offset + n].view_as(param.data).clone()
                offset += n


# ---------------------------------------------------------------------------
# MAML - Model Agnostic Meta-Learning
# ---------------------------------------------------------------------------

class MAML:
    """
    Model-Agnostic Meta-Learning (Finn et al., 2017).

    Learns an initialization that can be quickly adapted to new tasks
    (market regimes) with just a few gradient steps.

    Inner loop: adapt to each regime task
    Outer loop: meta-update to improve adaptability
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        n_inner_steps: int = 5,
        first_order: bool = True,    # FOMAML: no second-order gradients
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_inner_steps = n_inner_steps
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    def _clone_params(self) -> Dict[str, Tensor]:
        return {name: param.clone() for name, param in self.model.named_parameters()}

    def _inner_step(
        self,
        support_x: Tensor,
        support_y: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, Tensor]:
        """Run n_inner_steps of gradient descent on support set."""
        if params is None:
            params = self._clone_params()

        for _ in range(self.n_inner_steps):
            # Forward with current params
            outputs = self.model(support_x)
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = loss_fn(outputs, support_y) if loss_fn else outputs.mean()

            grads = torch.autograd.grad(
                loss,
                list(params.values()),
                create_graph=not self.first_order,
                allow_unused=True,
            )
            # SGD update
            new_params = {}
            for (name, param), grad in zip(params.items(), grads):
                if grad is not None:
                    new_params[name] = param - self.inner_lr * grad
                else:
                    new_params[name] = param
            params = new_params

        return params

    def meta_train_step(
        self,
        tasks: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
        loss_fn: Optional[Callable] = None,
    ) -> float:
        """
        One meta-training step.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples.
            loss_fn: Loss function.

        Returns:
            Meta loss value.
        """
        self.meta_optimizer.zero_grad()
        meta_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop adaptation
            adapted_params = self._inner_step(support_x, support_y, loss_fn=loss_fn)

            # Query loss with adapted params (for meta-update)
            # Load adapted params temporarily
            original_params = {}
            for name, param in self.model.named_parameters():
                original_params[name] = param.data.clone()
                if name in adapted_params:
                    param.data.copy_(adapted_params[name])

            query_output = self.model(query_x)
            if isinstance(query_output, dict):
                query_loss = query_output["loss"]
            else:
                query_loss = loss_fn(query_output, query_y) if loss_fn else query_output.mean()

            meta_loss = meta_loss + query_loss

            # Restore original params
            for name, param in self.model.named_parameters():
                param.data.copy_(original_params[name])

        meta_loss = meta_loss / max(1, len(tasks))
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()

    def adapt(
        self,
        support_x: Tensor,
        support_y: Tensor,
        loss_fn: Optional[Callable] = None,
        n_steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt model to new regime using support set.
        Returns adapted model (does not modify original).
        """
        n_steps = n_steps or self.n_inner_steps
        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            outputs = adapted_model(support_x)
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = loss_fn(outputs, support_y) if loss_fn else outputs.mean()
            loss.backward()
            optimizer.step()

        return adapted_model


# ---------------------------------------------------------------------------
# Continual learning trainer
# ---------------------------------------------------------------------------

class ContinualLearningTrainer:
    """
    Orchestrates continual learning across market regimes.

    Combines:
      - EWC for parameter regularization
      - Replay buffer for memory
      - LoRA adapters for regime-specific adaptation
      - Drift detection for automatic regime change
    """

    def __init__(
        self,
        model: nn.Module,
        ewc: Optional[EWC] = None,
        replay_buffer: Optional[RegimeAwareReplayBuffer] = None,
        lora_model: Optional[OnlineLoRAModel] = None,
        drift_detector: Optional[RegimeDriftDetector] = None,
        learning_rate: float = 1e-4,
        replay_coeff: float = 0.5,
        ewc_coeff: float = 400.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.ewc = ewc
        self.replay_buffer = replay_buffer
        self.lora_model = lora_model
        self.drift_detector = drift_detector or RegimeDriftDetector()
        self.learning_rate = learning_rate
        self.replay_coeff = replay_coeff
        self.ewc_coeff = ewc_coeff
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        active_model = lora_model if lora_model else model
        params = lora_model.trainable_parameters() if lora_model else list(model.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
        self.forgetting_metrics = ForgettingMetrics()
        self._current_regime = MarketRegime.UNKNOWN
        self._regime_steps: Dict[str, int] = {}

    def train_on_regime(
        self,
        regime: str,
        dataloader: DataLoader,
        n_epochs: int = 3,
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train on a new regime. Applies EWC and replay to prevent forgetting.
        """
        model = self.lora_model if self.lora_model else self.model
        model.train()

        if self.lora_model and regime != self._current_regime:
            # Load regime-specific adapters if available
            try:
                self.lora_model.load_regime_adapters(regime)
            except ValueError:
                pass  # New regime; start fresh
            self._current_regime = regime

        total_loss = 0.0
        step = 0

        for epoch in range(n_epochs):
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    x = batch.get("input_ids", batch.get("features")).to(self.device)
                    y = batch.get("labels", None)
                    if y is not None:
                        y = y.to(self.device)
                else:
                    continue

                self.optimizer.zero_grad()

                outputs = model(x)
                if isinstance(outputs, dict):
                    task_loss = outputs["loss"]
                else:
                    task_loss = F.mse_loss(outputs.squeeze(), y.float()) if y is not None else outputs.mean()

                loss = task_loss

                # EWC penalty
                if self.ewc is not None:
                    loss = loss + self.ewc.penalty()

                # Replay loss
                if self.replay_buffer is not None and self.replay_buffer.total_size() > 0:
                    try:
                        r_feat, r_lbl, _ = self.replay_buffer.sample(
                            n_samples=min(32, self.replay_buffer.total_size()),
                        )
                        r_feat = r_feat.to(self.device)
                        r_lbl = r_lbl.to(self.device)
                        r_out = model(r_feat)
                        if isinstance(r_out, dict):
                            r_loss = r_out["loss"]
                        else:
                            r_loss = F.mse_loss(r_out.squeeze(), r_lbl.float())
                        loss = loss + self.replay_coeff * r_loss
                    except Exception:
                        pass

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                # Add to replay buffer
                if self.replay_buffer is not None:
                    self.replay_buffer.add(x.detach(), y.detach() if y is not None else x.detach(), regime)

                total_loss += task_loss.item()
                step += 1

        # Consolidate EWC after training on this regime
        if self.ewc is not None:
            self.ewc.consolidate(dataloader, device=self.device)

        # Save regime adapters
        if self.lora_model is not None:
            self.lora_model.save_regime_adapters(regime)

        avg_loss = total_loss / max(1, step)
        self._regime_steps[regime] = self._regime_steps.get(regime, 0) + step

        return {
            "regime": regime,
            "avg_loss": avg_loss,
            "steps": step,
            "ewc_forgetting": self.ewc.forgetting_measure() if self.ewc else 0.0,
        }

    def online_update(
        self,
        features: Tensor,
        labels: Tensor,
        vol: float = 0.0,
        ret: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Single-step online update. Checks for drift and switches regime if needed.
        """
        self.model.train()
        features = features.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(features)
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = F.mse_loss(outputs.squeeze(), labels.float())

        error = loss.item()
        loss.backward()
        self.optimizer.step()

        # Check for drift
        drift_info = self.drift_detector.update(vol, ret, error)

        if drift_info["regime_changed"]:
            new_regime = drift_info["regime"]
            logger.info(f"Regime changed: {self._current_regime} -> {new_regime}")
            if self.lora_model:
                try:
                    self.lora_model.load_regime_adapters(new_regime)
                except ValueError:
                    pass
            self._current_regime = new_regime

        return {
            "loss": error,
            **drift_info,
        }

    def evaluate_forgetting(
        self,
        regime_dataloaders: Dict[str, DataLoader],
        metric_fn: Optional[Callable] = None,
        current_task: int = 0,
    ) -> Dict[str, float]:
        """Evaluate model on all regimes to measure forgetting."""
        self.model.eval()
        metrics = {}
        with torch.no_grad():
            for regime, loader in regime_dataloaders.items():
                total_loss = 0.0
                n = 0
                for batch in loader:
                    if isinstance(batch, (list, tuple)):
                        x, y = batch[0].to(self.device), batch[1].to(self.device)
                    elif isinstance(batch, dict):
                        x = batch.get("input_ids", batch.get("features")).to(self.device)
                        y = batch.get("labels", None)
                        if y is not None:
                            y = y.to(self.device)
                    else:
                        continue
                    outputs = self.model(x)
                    if isinstance(outputs, dict):
                        loss = outputs["loss"]
                    else:
                        loss = F.mse_loss(outputs.squeeze(), y.float()) if y is not None else outputs.mean()
                    total_loss += loss.item()
                    n += 1
                avg = total_loss / max(1, n)
                metrics[regime] = avg
                self.forgetting_metrics.record(regime, current_task, avg)
        return metrics


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Regimes
    "MarketRegime",
    "RegimeEpisode",
    # EWC
    "EWC",
    # PNN
    "PNNColumn",
    "ProgressiveNeuralNetwork",
    # Replay
    "RegimeAwareReplayBuffer",
    # Metrics
    "ForgettingMetrics",
    # Drift detection
    "DDMDriftDetector",
    "ADWINDriftDetector",
    "PageHinkleyDriftDetector",
    "RegimeDriftDetector",
    # LoRA adaptation
    "OnlineLoRAAdapter",
    "OnlineLoRAModel",
    # GEM
    "GEM",
    # MAML
    "MAML",
    # Trainer
    "ContinualLearningTrainer",
]
