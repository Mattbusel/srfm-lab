

# ============================================================
# Extended Continual Learning Components
# ============================================================

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


class ElasticWeightConsolidation(nn.Module):
    """EWC (Kirkpatrick et al. 2017): penalizes changes to important weights.

    Importance estimated by diagonal Fisher information matrix.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0, n_fisher_samples: int = 200):
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.n_fisher_samples = n_fisher_samples

        # Stores: {param_name: (optimal_param, fisher_diag)}
        self._anchors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def estimate_fisher(self, dataloader, loss_fn: Callable, device: str = "cpu"):
        """Estimate Fisher information diagonal on current task data."""
        self.model.eval()
        fisher_dict: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)

        count = 0
        for batch in dataloader:
            if count >= self.n_fisher_samples:
                break
            self.model.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data.pow(2)
            count += 1

        # Average
        for name in fisher_dict:
            fisher_dict[name] /= count

        # Store anchors
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._anchors[name] = (
                    param.data.clone(),
                    fisher_dict.get(name, torch.zeros_like(param.data)),
                )

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty term."""
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._anchors:
                optimal, fisher = self._anchors[name]
                optimal = optimal.to(param.device)
                fisher = fisher.to(param.device)
                loss = loss + (fisher * (param - optimal).pow(2)).sum()
        return 0.5 * self.ewc_lambda * loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class OnlineEWC(ElasticWeightConsolidation):
    """Online EWC (Schwarz et al. 2018): running average of Fisher matrices across tasks."""

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0, gamma: float = 0.95):
        super().__init__(model, ewc_lambda)
        self.gamma = gamma
        self._task_count = 0

    def consolidate(self, dataloader, loss_fn: Callable, device: str = "cpu"):
        """Consolidate current task into running EWC anchor."""
        self.estimate_fisher(dataloader, loss_fn, device)
        self._task_count += 1

        if self._task_count > 1:
            # Decay old Fisher estimates
            for name in self._anchors:
                optimal, fisher = self._anchors[name]
                self._anchors[name] = (optimal, self.gamma * fisher)


class ProgressiveNeuralNetworks(nn.Module):
    """Progressive Neural Networks (Rusu et al. 2016).

    Adds new columns for each task; lateral connections from all previous columns.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.columns: nn.ModuleList = nn.ModuleList()
        self.lateral_connections: nn.ModuleList = nn.ModuleList()
        self._add_column()

    def _add_column(self):
        """Add a new network column."""
        k = len(self.columns)
        col = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
        ])
        self.columns.append(col)

        if k > 0:
            # Lateral adapters from all prev columns at each hidden layer
            laterals = nn.ModuleList()
            for prev_k in range(k):
                lat = nn.ModuleList([
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                ])
                laterals.append(lat)
            self.lateral_connections.append(laterals)

        # Freeze all previous columns
        for prev_k in range(k):
            for param in self.columns[prev_k].parameters():
                param.requires_grad_(False)

    def add_task(self):
        """Add a new column for a new task."""
        self._add_column()

    def forward(self, x: torch.Tensor, column_idx: Optional[int] = None) -> torch.Tensor:
        k = column_idx if column_idx is not None else len(self.columns) - 1
        col = self.columns[k]

        # Layer 1
        h_prev = [None] * k  # hidden states from prev columns at each layer
        h = F.relu(col[0](x))

        # Collect from prev columns layer 0
        prev_h = []
        for pk in range(k):
            prev_h.append(F.relu(self.columns[pk][0](x)))

        # Layer 2 with laterals
        h2_input = h
        if k > 0:
            laterals_for_k = self.lateral_connections[k - 1]
            for pk, lat in enumerate(laterals_for_k):
                h2_input = h2_input + lat[0](prev_h[pk])
        h = F.relu(col[1](h2_input))

        # Output layer
        out = col[2](h)
        return out


class PacketNetworks(nn.Module):
    """PackNet (Mallya & Lazebnik 2018): hard parameter isolation via binary masks."""

    def __init__(self, model: nn.Module, prune_ratio: float = 0.5):
        super().__init__()
        self.model = model
        self.prune_ratio = prune_ratio
        self._masks: Dict[str, torch.Tensor] = {}
        self._task_masks: List[Dict[str, torch.Tensor]] = []
        self._current_task = 0

    def _get_prunable_params(self):
        return {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.dim() >= 2
        }

    def pack_task(self):
        """After training task k, prune low-magnitude weights and assign to task k+1."""
        params = self._get_prunable_params()

        # Find free parameters (not assigned to any previous task)
        task_mask = {}
        for name, param in params.items():
            existing_mask = self._masks.get(name, torch.zeros_like(param.data, dtype=torch.bool))
            free = ~existing_mask
            # Rank free weights by magnitude
            magnitudes = param.data.abs() * free.float()
            flat_mag = magnitudes.flatten()
            # Keep top (1-prune_ratio) fraction of free weights
            k = max(1, int(free.sum().item() * (1 - self.prune_ratio)))
            threshold = flat_mag.topk(k).values.min()
            new_mask = (magnitudes >= threshold) & free
            task_mask[name] = new_mask
            # Update global free mask
            self._masks[name] = existing_mask | new_mask

        self._task_masks.append(task_mask)
        self._current_task += 1

    def apply_task_mask(self, task_id: int):
        """Zero out parameters not belonging to task_id."""
        if task_id >= len(self._task_masks):
            return
        mask = self._task_masks[task_id]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in mask:
                    param.data *= mask[name].float()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ContinualNormalization(nn.Module):
    """Task-specific batch normalization statistics for continual learning."""

    def __init__(self, num_features: int, num_tasks: int = 10, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.num_tasks = num_tasks
        self.eps = eps
        self.momentum = momentum
        self.current_task = 0

        # Per-task affine params
        self.weight = nn.Parameter(torch.ones(num_tasks, num_features))
        self.bias = nn.Parameter(torch.zeros(num_tasks, num_features))

        # Per-task running stats (not parameters)
        self.register_buffer("running_mean", torch.zeros(num_tasks, num_features))
        self.register_buffer("running_var", torch.ones(num_tasks, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.current_task
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean[t] = (1 - self.momentum) * self.running_mean[t] + self.momentum * mean.detach()
            self.running_var[t] = (1 - self.momentum) * self.running_var[t] + self.momentum * var.detach()
        else:
            mean = self.running_mean[t]
            var = self.running_var[t]

        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.weight[t] * x_norm + self.bias[t]

    def set_task(self, task_id: int):
        assert 0 <= task_id < self.num_tasks
        self.current_task = task_id


class GradientEpisodicMemory(nn.Module):
    """GEM (Lopez-Paz & Ranzato 2017): project gradients to avoid forgetting.

    Maintains episodic memory per task and projects current gradient
    to satisfy dot-product constraints with memory gradients.
    """

    def __init__(self, model: nn.Module, memory_size_per_task: int = 100, n_tasks: int = 10):
        super().__init__()
        self.model = model
        self.memory_size_per_task = memory_size_per_task
        self.n_tasks = n_tasks

        self.memory_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._prev_gradients: List[torch.Tensor] = []
        self._current_task = 0

    def store_memory(self, x: torch.Tensor, y: torch.Tensor):
        """Store a sample into episodic memory."""
        idx = torch.randperm(x.shape[0])[:self.memory_size_per_task]
        self.memory_data.append((x[idx].detach().clone(), y[idx].detach().clone()))

    def compute_memory_gradients(self, loss_fn: Callable, device: str = "cpu") -> List[torch.Tensor]:
        """Compute gradients on all past task memories."""
        past_grads = []
        params = [p for p in self.model.parameters() if p.requires_grad]
        for mem_x, mem_y in self.memory_data[:-1]:  # exclude current task
            self.model.zero_grad()
            mem_x, mem_y = mem_x.to(device), mem_y.to(device)
            loss = loss_fn(self.model(mem_x), mem_y)
            loss.backward()
            grad = torch.cat([p.grad.data.flatten() for p in params if p.grad is not None])
            past_grads.append(grad.clone())
        return past_grads

    def project_gradient(self, current_grad: torch.Tensor, past_grads: List[torch.Tensor]) -> torch.Tensor:
        """Project current_grad so dot product with all past_grads is >= 0."""
        g = current_grad.clone()
        for pg in past_grads:
            dot = (g * pg).sum()
            if dot < 0:
                # Project: g = g - (g·pg / pg·pg) * pg
                g = g - (dot / (pg * pg).sum()) * pg
        return g

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MemoryReplayBuffer:
    """Experience replay buffer for continual learning."""

    def __init__(self, capacity: int = 5000, strategy: str = "reservoir"):
        self.capacity = capacity
        self.strategy = strategy  # reservoir | fifo | class_balanced
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
        self._n_seen = 0

    def add(self, x: torch.Tensor, y: torch.Tensor):
        """Add samples to replay buffer."""
        n = x.shape[0]
        for i in range(n):
            self._n_seen += 1
            if self.strategy == "reservoir":
                if len(self.buffer_x) < self.capacity:
                    self.buffer_x.append(x[i].clone())
                    self.buffer_y.append(y[i].clone())
                else:
                    j = torch.randint(0, self._n_seen, (1,)).item()
                    if j < self.capacity:
                        self.buffer_x[j] = x[i].clone()
                        self.buffer_y[j] = y[i].clone()
            else:  # fifo
                self.buffer_x.append(x[i].clone())
                self.buffer_y.append(y[i].clone())
                if len(self.buffer_x) > self.capacity:
                    self.buffer_x.pop(0)
                    self.buffer_y.pop(0)

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n items from buffer."""
        idx = torch.randperm(len(self.buffer_x))[:n]
        x = torch.stack([self.buffer_x[i] for i in idx])
        y = torch.stack([self.buffer_y[i] for i in idx])
        return x, y

    def __len__(self) -> int:
        return len(self.buffer_x)


class DualMemorySystem(nn.Module):
    """Complementary Learning System (CLS): hippocampus + neocortex dual memory.

    Fast-learning episodic memory (hippocampus) + slow-learning semantic memory (neocortex).
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        semantic_dim: int,
        hippo_memory_size: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.semantic_dim = semantic_dim

        # Hippocampus: fast-learning autoassociative memory
        self.hippo_keys = nn.Parameter(torch.randn(hippo_memory_size, input_dim))
        self.hippo_values = nn.Parameter(torch.randn(hippo_memory_size, memory_dim))

        # Neocortex: slow-learning semantic model
        self.neocortex = nn.Sequential(
            nn.Linear(input_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, memory_dim),
        )

        # Integration gate
        self.gate = nn.Linear(2 * memory_dim, 1)

    def hippocampal_recall(self, x: torch.Tensor) -> torch.Tensor:
        """Soft attention-based recall from episodic memory."""
        # x: (B, D)
        sim = F.cosine_similarity(x.unsqueeze(1), self.hippo_keys.unsqueeze(0), dim=-1)  # (B, M)
        weights = F.softmax(sim / 0.1, dim=-1)  # (B, M)
        recalled = weights @ self.hippo_values  # (B, memory_dim)
        return recalled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hippo_out = self.hippocampal_recall(x)         # (B, memory_dim)
        neo_out = self.neocortex(x)                    # (B, memory_dim)
        # Gate between systems
        gate_input = torch.cat([hippo_out, neo_out], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input))   # (B, 1)
        return alpha * hippo_out + (1 - alpha) * neo_out


class SynapticIntelligence(nn.Module):
    """Synaptic Intelligence (Zenke et al. 2017): online importance estimation via path integral."""

    def __init__(self, model: nn.Module, si_lambda: float = 0.1, damping: float = 0.1):
        super().__init__()
        self.model = model
        self.si_lambda = si_lambda
        self.damping = damping

        self._prev_params: Dict[str, torch.Tensor] = {}
        self._running_importance: Dict[str, torch.Tensor] = {}
        self._old_params: Dict[str, torch.Tensor] = {}
        self._W: Dict[str, torch.Tensor] = {}

        self._initialize()

    def _initialize(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._prev_params[name] = param.data.clone()
                self._running_importance[name] = torch.zeros_like(param.data)
                self._W[name] = torch.zeros_like(param.data)

    def update_w(self):
        """Update path integral W after each step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                delta = param.data - self._prev_params[name]
                self._W[name] += -param.grad.data * delta
                self._prev_params[name] = param.data.clone()

    def consolidate(self):
        """Consolidate importance after task ends."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                omega = self._W[name] / ((param.data - self._old_params.get(name, param.data)).pow(2) + self.damping)
                self._running_importance[name] = self._running_importance.get(name, torch.zeros_like(omega)) + F.relu(omega)
                self._old_params[name] = param.data.clone()
                self._W[name] = torch.zeros_like(param.data)

    def si_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._old_params and name in self._running_importance:
                loss = loss + (self._running_importance[name] * (param - self._old_params[name]).pow(2)).sum()
        return self.si_lambda * loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ============================================================
# Extended Continual Learning Components
# ============================================================

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


class ElasticWeightConsolidation(nn.Module):
    """EWC (Kirkpatrick et al. 2017): penalizes changes to important weights.

    Importance estimated by diagonal Fisher information matrix.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0, n_fisher_samples: int = 200):
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.n_fisher_samples = n_fisher_samples

        # Stores: {param_name: (optimal_param, fisher_diag)}
        self._anchors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def estimate_fisher(self, dataloader, loss_fn: Callable, device: str = "cpu"):
        """Estimate Fisher information diagonal on current task data."""
        self.model.eval()
        fisher_dict: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)

        count = 0
        for batch in dataloader:
            if count >= self.n_fisher_samples:
                break
            self.model.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data.pow(2)
            count += 1

        # Average
        for name in fisher_dict:
            fisher_dict[name] /= count

        # Store anchors
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._anchors[name] = (
                    param.data.clone(),
                    fisher_dict.get(name, torch.zeros_like(param.data)),
                )

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty term."""
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._anchors:
                optimal, fisher = self._anchors[name]
                optimal = optimal.to(param.device)
                fisher = fisher.to(param.device)
                loss = loss + (fisher * (param - optimal).pow(2)).sum()
        return 0.5 * self.ewc_lambda * loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class OnlineEWC(ElasticWeightConsolidation):
    """Online EWC (Schwarz et al. 2018): running average of Fisher matrices across tasks."""

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0, gamma: float = 0.95):
        super().__init__(model, ewc_lambda)
        self.gamma = gamma
        self._task_count = 0

    def consolidate(self, dataloader, loss_fn: Callable, device: str = "cpu"):
        """Consolidate current task into running EWC anchor."""
        self.estimate_fisher(dataloader, loss_fn, device)
        self._task_count += 1

        if self._task_count > 1:
            # Decay old Fisher estimates
            for name in self._anchors:
                optimal, fisher = self._anchors[name]
                self._anchors[name] = (optimal, self.gamma * fisher)


class ProgressiveNeuralNetworks(nn.Module):
    """Progressive Neural Networks (Rusu et al. 2016).

    Adds new columns for each task; lateral connections from all previous columns.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.columns: nn.ModuleList = nn.ModuleList()
        self.lateral_connections: nn.ModuleList = nn.ModuleList()
        self._add_column()

    def _add_column(self):
        """Add a new network column."""
        k = len(self.columns)
        col = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
        ])
        self.columns.append(col)

        if k > 0:
            # Lateral adapters from all prev columns at each hidden layer
            laterals = nn.ModuleList()
            for prev_k in range(k):
                lat = nn.ModuleList([
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                ])
                laterals.append(lat)
            self.lateral_connections.append(laterals)

        # Freeze all previous columns
        for prev_k in range(k):
            for param in self.columns[prev_k].parameters():
                param.requires_grad_(False)

    def add_task(self):
        """Add a new column for a new task."""
        self._add_column()

    def forward(self, x: torch.Tensor, column_idx: Optional[int] = None) -> torch.Tensor:
        k = column_idx if column_idx is not None else len(self.columns) - 1
        col = self.columns[k]

        # Layer 1
        h_prev = [None] * k  # hidden states from prev columns at each layer
        h = F.relu(col[0](x))

        # Collect from prev columns layer 0
        prev_h = []
        for pk in range(k):
            prev_h.append(F.relu(self.columns[pk][0](x)))

        # Layer 2 with laterals
        h2_input = h
        if k > 0:
            laterals_for_k = self.lateral_connections[k - 1]
            for pk, lat in enumerate(laterals_for_k):
                h2_input = h2_input + lat[0](prev_h[pk])
        h = F.relu(col[1](h2_input))

        # Output layer
        out = col[2](h)
        return out


class PacketNetworks(nn.Module):
    """PackNet (Mallya & Lazebnik 2018): hard parameter isolation via binary masks."""

    def __init__(self, model: nn.Module, prune_ratio: float = 0.5):
        super().__init__()
        self.model = model
        self.prune_ratio = prune_ratio
        self._masks: Dict[str, torch.Tensor] = {}
        self._task_masks: List[Dict[str, torch.Tensor]] = []
        self._current_task = 0

    def _get_prunable_params(self):
        return {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.dim() >= 2
        }

    def pack_task(self):
        """After training task k, prune low-magnitude weights and assign to task k+1."""
        params = self._get_prunable_params()

        # Find free parameters (not assigned to any previous task)
        task_mask = {}
        for name, param in params.items():
            existing_mask = self._masks.get(name, torch.zeros_like(param.data, dtype=torch.bool))
            free = ~existing_mask
            # Rank free weights by magnitude
            magnitudes = param.data.abs() * free.float()
            flat_mag = magnitudes.flatten()
            # Keep top (1-prune_ratio) fraction of free weights
            k = max(1, int(free.sum().item() * (1 - self.prune_ratio)))
            threshold = flat_mag.topk(k).values.min()
            new_mask = (magnitudes >= threshold) & free
            task_mask[name] = new_mask
            # Update global free mask
            self._masks[name] = existing_mask | new_mask

        self._task_masks.append(task_mask)
        self._current_task += 1

    def apply_task_mask(self, task_id: int):
        """Zero out parameters not belonging to task_id."""
        if task_id >= len(self._task_masks):
            return
        mask = self._task_masks[task_id]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in mask:
                    param.data *= mask[name].float()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ContinualNormalization(nn.Module):
    """Task-specific batch normalization statistics for continual learning."""

    def __init__(self, num_features: int, num_tasks: int = 10, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.num_tasks = num_tasks
        self.eps = eps
        self.momentum = momentum
        self.current_task = 0

        # Per-task affine params
        self.weight = nn.Parameter(torch.ones(num_tasks, num_features))
        self.bias = nn.Parameter(torch.zeros(num_tasks, num_features))

        # Per-task running stats (not parameters)
        self.register_buffer("running_mean", torch.zeros(num_tasks, num_features))
        self.register_buffer("running_var", torch.ones(num_tasks, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.current_task
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean[t] = (1 - self.momentum) * self.running_mean[t] + self.momentum * mean.detach()
            self.running_var[t] = (1 - self.momentum) * self.running_var[t] + self.momentum * var.detach()
        else:
            mean = self.running_mean[t]
            var = self.running_var[t]

        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.weight[t] * x_norm + self.bias[t]

    def set_task(self, task_id: int):
        assert 0 <= task_id < self.num_tasks
        self.current_task = task_id


class GradientEpisodicMemory(nn.Module):
    """GEM (Lopez-Paz & Ranzato 2017): project gradients to avoid forgetting.

    Maintains episodic memory per task and projects current gradient
    to satisfy dot-product constraints with memory gradients.
    """

    def __init__(self, model: nn.Module, memory_size_per_task: int = 100, n_tasks: int = 10):
        super().__init__()
        self.model = model
        self.memory_size_per_task = memory_size_per_task
        self.n_tasks = n_tasks

        self.memory_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._prev_gradients: List[torch.Tensor] = []
        self._current_task = 0

    def store_memory(self, x: torch.Tensor, y: torch.Tensor):
        """Store a sample into episodic memory."""
        idx = torch.randperm(x.shape[0])[:self.memory_size_per_task]
        self.memory_data.append((x[idx].detach().clone(), y[idx].detach().clone()))

    def compute_memory_gradients(self, loss_fn: Callable, device: str = "cpu") -> List[torch.Tensor]:
        """Compute gradients on all past task memories."""
        past_grads = []
        params = [p for p in self.model.parameters() if p.requires_grad]
        for mem_x, mem_y in self.memory_data[:-1]:  # exclude current task
            self.model.zero_grad()
            mem_x, mem_y = mem_x.to(device), mem_y.to(device)
            loss = loss_fn(self.model(mem_x), mem_y)
            loss.backward()
            grad = torch.cat([p.grad.data.flatten() for p in params if p.grad is not None])
            past_grads.append(grad.clone())
        return past_grads

    def project_gradient(self, current_grad: torch.Tensor, past_grads: List[torch.Tensor]) -> torch.Tensor:
        """Project current_grad so dot product with all past_grads is >= 0."""
        g = current_grad.clone()
        for pg in past_grads:
            dot = (g * pg).sum()
            if dot < 0:
                # Project: g = g - (g·pg / pg·pg) * pg
                g = g - (dot / (pg * pg).sum()) * pg
        return g

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MemoryReplayBuffer:
    """Experience replay buffer for continual learning."""

    def __init__(self, capacity: int = 5000, strategy: str = "reservoir"):
        self.capacity = capacity
        self.strategy = strategy  # reservoir | fifo | class_balanced
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
        self._n_seen = 0

    def add(self, x: torch.Tensor, y: torch.Tensor):
        """Add samples to replay buffer."""
        n = x.shape[0]
        for i in range(n):
            self._n_seen += 1
            if self.strategy == "reservoir":
                if len(self.buffer_x) < self.capacity:
                    self.buffer_x.append(x[i].clone())
                    self.buffer_y.append(y[i].clone())
                else:
                    j = torch.randint(0, self._n_seen, (1,)).item()
                    if j < self.capacity:
                        self.buffer_x[j] = x[i].clone()
                        self.buffer_y[j] = y[i].clone()
            else:  # fifo
                self.buffer_x.append(x[i].clone())
                self.buffer_y.append(y[i].clone())
                if len(self.buffer_x) > self.capacity:
                    self.buffer_x.pop(0)
                    self.buffer_y.pop(0)

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n items from buffer."""
        idx = torch.randperm(len(self.buffer_x))[:n]
        x = torch.stack([self.buffer_x[i] for i in idx])
        y = torch.stack([self.buffer_y[i] for i in idx])
        return x, y

    def __len__(self) -> int:
        return len(self.buffer_x)


class DualMemorySystem(nn.Module):
    """Complementary Learning System (CLS): hippocampus + neocortex dual memory.

    Fast-learning episodic memory (hippocampus) + slow-learning semantic memory (neocortex).
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        semantic_dim: int,
        hippo_memory_size: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.semantic_dim = semantic_dim

        # Hippocampus: fast-learning autoassociative memory
        self.hippo_keys = nn.Parameter(torch.randn(hippo_memory_size, input_dim))
        self.hippo_values = nn.Parameter(torch.randn(hippo_memory_size, memory_dim))

        # Neocortex: slow-learning semantic model
        self.neocortex = nn.Sequential(
            nn.Linear(input_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, memory_dim),
        )

        # Integration gate
        self.gate = nn.Linear(2 * memory_dim, 1)

    def hippocampal_recall(self, x: torch.Tensor) -> torch.Tensor:
        """Soft attention-based recall from episodic memory."""
        # x: (B, D)
        sim = F.cosine_similarity(x.unsqueeze(1), self.hippo_keys.unsqueeze(0), dim=-1)  # (B, M)
        weights = F.softmax(sim / 0.1, dim=-1)  # (B, M)
        recalled = weights @ self.hippo_values  # (B, memory_dim)
        return recalled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hippo_out = self.hippocampal_recall(x)         # (B, memory_dim)
        neo_out = self.neocortex(x)                    # (B, memory_dim)
        # Gate between systems
        gate_input = torch.cat([hippo_out, neo_out], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input))   # (B, 1)
        return alpha * hippo_out + (1 - alpha) * neo_out


class SynapticIntelligence(nn.Module):
    """Synaptic Intelligence (Zenke et al. 2017): online importance estimation via path integral."""

    def __init__(self, model: nn.Module, si_lambda: float = 0.1, damping: float = 0.1):
        super().__init__()
        self.model = model
        self.si_lambda = si_lambda
        self.damping = damping

        self._prev_params: Dict[str, torch.Tensor] = {}
        self._running_importance: Dict[str, torch.Tensor] = {}
        self._old_params: Dict[str, torch.Tensor] = {}
        self._W: Dict[str, torch.Tensor] = {}

        self._initialize()

    def _initialize(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._prev_params[name] = param.data.clone()
                self._running_importance[name] = torch.zeros_like(param.data)
                self._W[name] = torch.zeros_like(param.data)

    def update_w(self):
        """Update path integral W after each step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                delta = param.data - self._prev_params[name]
                self._W[name] += -param.grad.data * delta
                self._prev_params[name] = param.data.clone()

    def consolidate(self):
        """Consolidate importance after task ends."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                omega = self._W[name] / ((param.data - self._old_params.get(name, param.data)).pow(2) + self.damping)
                self._running_importance[name] = self._running_importance.get(name, torch.zeros_like(omega)) + F.relu(omega)
                self._old_params[name] = param.data.clone()
                self._W[name] = torch.zeros_like(param.data)

    def si_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._old_params and name in self._running_importance:
                loss = loss + (self._running_importance[name] * (param - self._old_params[name]).pow(2)).sum()
        return self.si_lambda * loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
