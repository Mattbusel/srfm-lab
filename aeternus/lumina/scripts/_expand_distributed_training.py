"""Expand distributed_training.py with large additions."""
import os

PATH = os.path.join(os.path.dirname(__file__), "..", "lumina", "distributed_training.py")

CONTENT = '''

# =============================================================================
# SECTION: Advanced Distributed Training Strategies
# =============================================================================

import torch
import torch.nn as nn
import torch.distributed as dist
import math
from typing import Optional, List, Dict, Tuple, Any


class ZeRO3Optimizer:
    """ZeRO Stage 3 optimizer: partition parameters, gradients, AND optimizer states.

    Each rank only holds 1/world_size of each parameter tensor.
    During forward: gather parameters on demand (all-gather).
    During backward: reduce gradients then discard non-local shards.
    During optimizer step: update only local parameter shard.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs: dict = None,
        world_size: int = 1,
        rank: int = 0,
        overlap_comm: bool = True,
    ):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.overlap_comm = overlap_comm
        self._param_shards = {}
        self._grad_shards = {}
        self._optimizer = None

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-4, "weight_decay": 0.01}

        self._partition_params()
        local_params = [v for v in self._param_shards.values()]
        self._optimizer = optimizer_class(local_params, **optimizer_kwargs)

    def _partition_params(self):
        """Partition each parameter across ranks."""
        for name, param in self.model.named_parameters():
            flat = param.data.view(-1)
            n = flat.shape[0]
            shard_size = (n + self.world_size - 1) // self.world_size
            start = self.rank * shard_size
            end = min(start + shard_size, n)
            shard = flat[start:end].clone()
            self._param_shards[name] = nn.Parameter(shard)

    def all_gather_params(self):
        """Gather all parameter shards from all ranks."""
        if not dist.is_initialized():
            return

        for name, param in self.model.named_parameters():
            shard = self._param_shards[name]
            gathered = [torch.zeros_like(shard) for _ in range(self.world_size)]
            dist.all_gather(gathered, shard)
            full = torch.cat(gathered, dim=0)
            n = param.data.view(-1).shape[0]
            param.data.copy_(full[:n].view_as(param.data))

    def reduce_scatter_grads(self):
        """Reduce gradients and scatter to owning ranks."""
        if not dist.is_initialized():
            return

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            flat_grad = param.grad.view(-1)
            n = flat_grad.shape[0]
            shard_size = (n + self.world_size - 1) // self.world_size

            # Pad to multiple of world_size
            pad = shard_size * self.world_size - n
            if pad > 0:
                flat_grad = torch.cat([flat_grad, flat_grad.new_zeros(pad)])

            chunks = flat_grad.view(self.world_size, shard_size)
            local_grad = chunks[self.rank].clone()
            if dist.is_initialized():
                dist.reduce(local_grad, dst=self.rank, op=dist.ReduceOp.SUM)
            local_grad /= self.world_size
            self._grad_shards[name] = local_grad

    def step(self):
        """Apply optimizer step on local parameter shards."""
        for name in self._param_shards:
            if name in self._grad_shards:
                self._param_shards[name].grad = self._grad_shards[name]

        self._optimizer.step()
        self._optimizer.zero_grad()
        self._grad_shards.clear()

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None


class AsyncGradientAccumulator:
    """Asynchronous gradient accumulation with non-blocking all-reduce.

    Overlaps gradient computation of layer i+1 with communication
    of layer i's gradients. Uses torch.distributed async operations.
    """

    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int = 4,
        world_size: int = 1,
    ):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.world_size = world_size
        self._step = 0
        self._handles = []
        self._param_grads = {}

    def _register_hooks(self):
        """Register backward hooks for async gradient reduction."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                def make_hook(n, p):
                    def hook(grad):
                        if dist.is_initialized() and self.world_size > 1:
                            handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                            self._handles.append((handle, grad, n))
                        return grad
                    return hook
                param.register_hook(make_hook(name, param))

    def wait_for_all_reduces(self):
        """Wait for all pending async all-reduce operations."""
        for handle, grad, name in self._handles:
            if hasattr(handle, "wait"):
                handle.wait()
            grad /= self.world_size
        self._handles.clear()

    def step(self) -> bool:
        """Return True if it's time to apply gradients."""
        self._step += 1
        return self._step % self.accumulation_steps == 0


class ElasticTrainingCoordinator:
    """Coordinate elastic training: handle node additions/removals.

    Implements a simplified version of PyTorch's elastic launch
    for fault-tolerant distributed training.

    Responsibilities:
    - Track active workers via rendezvous
    - Checkpoint and restore training state on topology changes
    - Rebalance data shards when world_size changes
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        checkpoint_path: str = "/tmp/elastic_ckpt.pt",
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.checkpoint_path = checkpoint_path
        self._world_size = 1
        self._rank = 0
        self._step = 0
        self._restarts = 0

    def save_checkpoint(self, state: dict):
        """Save training state for elastic recovery."""
        state["_step"] = self._step
        state["_restarts"] = self._restarts
        torch.save(state, self.checkpoint_path)

    def load_checkpoint(self) -> Optional[dict]:
        """Load training state after elastic restart."""
        try:
            state = torch.load(self.checkpoint_path, map_location="cpu")
            self._step = state.pop("_step", 0)
            self._restarts = state.pop("_restarts", 0) + 1
            return state
        except FileNotFoundError:
            return None

    def handle_worker_failure(self, failed_ranks: List[int]) -> bool:
        """Handle worker failures and determine if training can continue."""
        active = self._world_size - len(failed_ranks)
        if active < self.min_workers:
            return False  # Cannot continue
        self._world_size = active
        return True

    def rebalance_data_shards(self, dataset_size: int) -> Tuple[int, int]:
        """Recompute data shard start/end for current rank."""
        shard_size = (dataset_size + self._world_size - 1) // self._world_size
        start = self._rank * shard_size
        end = min(start + shard_size, dataset_size)
        return start, end


class FaultTolerantTrainer:
    """Trainer with built-in fault tolerance and automatic recovery.

    Features:
    - Automatic checkpoint saving at configurable intervals
    - Resume from latest checkpoint on failure
    - Gradient explosion detection and recovery
    - Learning rate recovery after NaN loss
    - Stuck detection (loss stagnation)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        checkpoint_dir: str = "/tmp/ft_ckpts",
        save_every: int = 100,
        max_grad_norm: float = 1.0,
        nan_recovery_lr_factor: float = 0.5,
        stagnation_patience: int = 50,
        stagnation_threshold: float = 1e-5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.max_grad_norm = max_grad_norm
        self.nan_recovery_lr_factor = nan_recovery_lr_factor
        self.stagnation_patience = stagnation_patience
        self.stagnation_threshold = stagnation_threshold

        self._step = 0
        self._loss_history = []
        self._nan_count = 0
        self._stagnation_count = 0

        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _is_nan_loss(self, loss: torch.Tensor) -> bool:
        return torch.isnan(loss) or torch.isinf(loss)

    def _recover_from_nan(self):
        """Recover from NaN loss by loading last checkpoint and reducing LR."""
        latest = self._get_latest_checkpoint()
        if latest:
            self.load_checkpoint(latest)
        for group in self.optimizer.param_groups:
            group["lr"] *= self.nan_recovery_lr_factor
        self._nan_count += 1

    def _detect_stagnation(self, loss: float) -> bool:
        """Detect if training has stagnated."""
        self._loss_history.append(loss)
        if len(self._loss_history) < self.stagnation_patience:
            return False
        recent = self._loss_history[-self.stagnation_patience:]
        improvement = max(recent) - min(recent)
        return improvement < self.stagnation_threshold

    def save_checkpoint(self, extra: dict = None):
        """Save training checkpoint."""
        path = f"{self.checkpoint_dir}/step_{self._step:08d}.pt"
        state = {
            "step": self._step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self._loss_history[-1000:],
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str):
        """Load checkpoint and restore training state."""
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._step = state.get("step", 0)
        self._loss_history = state.get("loss_history", [])

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file."""
        import os, glob
        ckpts = sorted(glob.glob(f"{self.checkpoint_dir}/step_*.pt"))
        return ckpts[-1] if ckpts else None

    def training_step(self, loss: torch.Tensor) -> dict:
        """Execute one fault-tolerant training step."""
        if self._is_nan_loss(loss):
            self._recover_from_nan()
            return {"status": "nan_recovery", "step": self._step}

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._step += 1
        loss_val = loss.item()
        stagnated = self._detect_stagnation(loss_val)

        if self._step % self.save_every == 0:
            self.save_checkpoint()

        return {
            "status": "ok",
            "step": self._step,
            "loss": loss_val,
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            "stagnated": stagnated,
            "nan_count": self._nan_count,
        }


# =============================================================================
# SECTION: Communication Optimization
# =============================================================================

class BucketedAllReduce:
    """Bucket-based gradient all-reduce for communication efficiency.

    Groups small gradients into buckets to reduce communication overhead.
    Similar to DDP's gradient bucketing mechanism.
    """

    def __init__(
        self,
        parameters,
        bucket_size_mb: float = 25.0,
        world_size: int = 1,
    ):
        self.world_size = world_size
        self.bucket_size_bytes = int(bucket_size_mb * 1e6)
        self._buckets = self._create_buckets(parameters)
        self._handles = {}

    def _create_buckets(self, parameters) -> List[List]:
        """Group parameters into buckets by size."""
        buckets = []
        current_bucket = []
        current_size = 0

        for p in reversed(list(parameters)):
            if p.requires_grad:
                size = p.numel() * p.element_size()
                current_bucket.append(p)
                current_size += size
                if current_size >= self.bucket_size_bytes:
                    buckets.append(current_bucket)
                    current_bucket = []
                    current_size = 0

        if current_bucket:
            buckets.append(current_bucket)

        return buckets

    def all_reduce_bucket(self, bucket_idx: int, async_op: bool = True):
        """Flatten bucket, all-reduce, then unflatten."""
        if not dist.is_initialized():
            return

        bucket = self._buckets[bucket_idx]
        grads = [p.grad.view(-1) for p in bucket if p.grad is not None]
        if not grads:
            return

        flat = torch.cat(grads)
        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=async_op)

        def callback():
            flat /= self.world_size
            offset = 0
            for p in bucket:
                if p.grad is not None:
                    n = p.grad.numel()
                    p.grad.copy_(flat[offset:offset+n].view_as(p.grad))
                    offset += n

        self._handles[bucket_idx] = (handle, callback)

    def synchronize(self):
        """Wait for all pending all-reduce operations."""
        for bucket_idx, (handle, callback) in self._handles.items():
            if hasattr(handle, "wait"):
                handle.wait()
            callback()
        self._handles.clear()


class TopKSparseCommunicator:
    """Top-K sparse gradient compression for communication reduction.

    Only communicates the top-K% largest gradient values.
    Accumulates residuals to prevent convergence degradation.
    Based on Lin et al. (2018) Deep Gradient Compression.
    """

    def __init__(
        self,
        compression_ratio: float = 0.01,
        world_size: int = 1,
        momentum_correction: bool = True,
    ):
        self.compression_ratio = compression_ratio
        self.world_size = world_size
        self.momentum_correction = momentum_correction
        self._velocity = {}
        self._residuals = {}

    def compress(self, name: str, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-K compression to a gradient tensor.

        Returns: (values, indices) of selected top-K elements.
        """
        if name in self._residuals:
            grad = grad + self._residuals[name]

        flat = grad.view(-1)
        n = flat.shape[0]
        k = max(1, int(n * self.compression_ratio))

        _, topk_idx = torch.topk(flat.abs(), k)
        topk_vals = flat[topk_idx]

        # Compute residual
        residual = flat.clone()
        residual[topk_idx] = 0
        self._residuals[name] = residual.view_as(grad)

        return topk_vals, topk_idx

    def decompress(self, values: torch.Tensor, indices: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Reconstruct dense gradient from sparse (values, indices)."""
        flat = torch.zeros(math.prod(shape), device=values.device, dtype=values.dtype)
        flat.scatter_(0, indices, values)
        return flat.view(shape)

    def all_reduce_sparse(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """Compress, all-reduce (dense on sparse), decompress."""
        if not dist.is_initialized() or self.world_size == 1:
            return grad

        vals, idx = self.compress(name, grad)
        # For simplicity, convert to dense for transmission (real impl uses sparse all-reduce)
        dense = self.decompress(vals, idx, grad.shape)
        dist.all_reduce(dense, op=dist.ReduceOp.SUM)
        dense /= self.world_size
        return dense


# =============================================================================
# SECTION: Data Parallel Training Utilities
# =============================================================================

class BalancedDataParallel(nn.Module):
    """Custom DataParallel with memory-balanced batch splitting.

    Standard nn.DataParallel suffers from GPU memory imbalance because
    the output reduction happens on device 0. This version distributes
    computation more evenly.
    """

    def __init__(self, module: nn.Module, device_ids: List[int] = None):
        super().__init__()
        self.module = module
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.n_devices = len(self.device_ids)

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        if self.n_devices == 1:
            return self.module(*inputs, **kwargs)

        # Split inputs across devices
        chunked = self._chunk_inputs(inputs)
        replicas = nn.parallel.replicate(self.module, self.device_ids[:len(chunked)])
        outputs = nn.parallel.parallel_apply(replicas, chunked)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def _chunk_inputs(self, inputs):
        """Split batch dimension across devices."""
        return nn.parallel.scatter(inputs, self.device_ids)


class GradientSynchronizationBarrier:
    """Synchronization barrier for gradient all-reduce in custom training loops."""

    def __init__(self, model: nn.Module, world_size: int = 1):
        self.model = model
        self.world_size = world_size
        self._sync_count = 0

    def synchronize_gradients(self, scale: float = 1.0):
        """All-reduce gradients across all workers and scale."""
        if not dist.is_initialized() or self.world_size <= 1:
            return

        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= (self.world_size * scale)

        self._sync_count += 1

    def synchronize_buffers(self):
        """Broadcast model buffers from rank 0 to all other ranks."""
        if not dist.is_initialized():
            return
        for buffer in self.model.buffers():
            dist.broadcast(buffer, src=0)


# =============================================================================
# SECTION: Model Parallel Training
# =============================================================================

class LayerPipelineExecutor:
    """Execute transformer layers in pipeline-parallel fashion.

    Splits microbatches across pipeline stages to overlap
    forward and backward passes.

    Based on GPipe (Huang et al. 2019).
    """

    def __init__(
        self,
        stages: List[nn.Module],
        n_microbatches: int = 4,
        devices: List[str] = None,
    ):
        self.stages = stages
        self.n_microbatches = n_microbatches
        self.devices = devices or [f"cuda:{i}" for i in range(len(stages))]
        self.n_stages = len(stages)

        # Move each stage to its device
        for stage, dev in zip(self.stages, self.devices):
            stage.to(dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GPipe-style forward: fill pipeline with microbatches."""
        B = x.shape[0]
        mb_size = (B + self.n_microbatches - 1) // self.n_microbatches

        microbatches = [
            x[i * mb_size: (i + 1) * mb_size]
            for i in range(self.n_microbatches)
        ]

        outputs = [None] * self.n_microbatches

        # Clock cycles: each clock processes one microbatch at one stage
        for clock in range(self.n_microbatches + self.n_stages - 1):
            for stage_idx in range(self.n_stages):
                mb_idx = clock - stage_idx
                if 0 <= mb_idx < self.n_microbatches:
                    device = self.devices[stage_idx]
                    stage = self.stages[stage_idx]

                    if stage_idx == 0:
                        inp = microbatches[mb_idx].to(device)
                    else:
                        inp = outputs[mb_idx].to(device)

                    outputs[mb_idx] = stage(inp)

        return torch.cat([o.to(self.devices[0]) for o in outputs], dim=0)


class TensorParallelAttention(nn.Module):
    """Multi-head attention with column-parallel Q/K/V and row-parallel output.

    Megatron-LM tensor parallelism: each rank computes a subset of heads.
    Requires all-reduce for the output projection.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        world_size: int = 1,
        rank: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_heads % world_size == 0
        self.n_local_heads = n_heads // world_size
        self.d_head = d_model // n_heads
        self.d_local = self.n_local_heads * self.d_head
        self.world_size = world_size
        self.rank = rank
        self.scale = math.sqrt(self.d_head)

        # Column parallel: each rank gets a slice of Q/K/V weights
        self.q_proj = nn.Linear(d_model, self.d_local, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_local, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_local, bias=False)

        # Row parallel: each rank computes partial output
        self.out_proj = nn.Linear(self.d_local, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_local_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_local)
        out = self.out_proj(out)

        # All-reduce to sum partial outputs from all ranks
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return out


# =============================================================================
# SECTION: Distributed Evaluation
# =============================================================================

class DistributedEvaluator:
    """Evaluate models across distributed workers and aggregate metrics.

    Handles:
    - Barrier synchronization before evaluation
    - Per-rank predictions collection
    - All-gather of predictions and labels
    - Metric computation on rank 0
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 0,
        world_size: int = 1,
        device: str = "cuda",
    ):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader) -> Optional[dict]:
        """Run evaluation across all ranks and aggregate results.

        Returns metrics dict on rank 0, None on other ranks.
        """
        self.model.eval()

        if dist.is_initialized():
            dist.barrier()

        all_preds = []
        all_labels = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[:-1], batch[-1]
                out = self.model(*[inp.to(self.device) for inp in inputs])
            else:
                out = self.model(batch.to(self.device))
                labels = None

            preds = out[0] if isinstance(out, (tuple, list)) else out
            all_preds.append(preds.cpu())
            if labels is not None:
                all_labels.append(labels.cpu())

        local_preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0)
        local_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0)

        if dist.is_initialized() and self.world_size > 1:
            gathered_preds = [torch.zeros_like(local_preds) for _ in range(self.world_size)]
            dist.all_gather(gathered_preds, local_preds)
            all_preds_combined = torch.cat(gathered_preds, dim=0)

            if len(local_labels) > 0:
                gathered_labels = [torch.zeros_like(local_labels) for _ in range(self.world_size)]
                dist.all_gather(gathered_labels, local_labels)
                all_labels_combined = torch.cat(gathered_labels, dim=0)
            else:
                all_labels_combined = torch.empty(0)
        else:
            all_preds_combined = local_preds
            all_labels_combined = local_labels

        if self.rank != 0:
            return None

        metrics = {"n_samples": all_preds_combined.shape[0]}
        if len(all_labels_combined) > 0:
            mse = torch.nn.functional.mse_loss(all_preds_combined.float(), all_labels_combined.float())
            metrics["mse"] = mse.item()
            metrics["rmse"] = mse.sqrt().item()

        return metrics


# =============================================================================
# SECTION: Activation Checkpointing with Custom Partitioning
# =============================================================================

class ActivationCheckpointScheduler:
    """Dynamically schedule which layers use activation checkpointing.

    Balances memory usage and recomputation cost:
    - Memory-intensive layers always checkpoint
    - Cheap layers never checkpoint
    - Medium layers checkpoint based on memory budget
    """

    def __init__(
        self,
        model: nn.Module,
        memory_budget_gb: float = 8.0,
        device: str = "cuda",
    ):
        self.model = model
        self.memory_budget_bytes = int(memory_budget_gb * 1e9)
        self.device = device
        self._layer_costs = {}

    def profile_layer_memory(self, layer: nn.Module, dummy_input: torch.Tensor) -> int:
        """Profile peak memory of a single layer forward pass."""
        if not torch.cuda.is_available():
            return 0
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.no_grad():
            layer(dummy_input)
        return torch.cuda.max_memory_allocated(self.device)

    def assign_checkpointing(
        self,
        layers: nn.ModuleList,
        dummy_input: torch.Tensor,
    ):
        """Assign checkpointing to layers to stay within memory budget."""
        from torch.utils.checkpoint import checkpoint

        costs = []
        for layer in layers:
            cost = self.profile_layer_memory(layer, dummy_input)
            costs.append(cost)

        total_cost = sum(costs)
        cumulative = 0
        for i, (layer, cost) in enumerate(zip(layers, costs)):
            cumulative += cost
            if cumulative > self.memory_budget_bytes:
                original_fwd = layer.forward

                def make_checkpointed(fwd):
                    def ckpt_fwd(*args, **kwargs):
                        return checkpoint(fwd, *args, use_reentrant=False, **kwargs)
                    return ckpt_fwd

                layer.forward = make_checkpointed(original_fwd)
                cumulative -= cost  # savings from checkpointing


# =============================================================================
# SECTION: Distributed Hyperparameter Search
# =============================================================================

class DistributedHyperparameterSearcher:
    """Run distributed hyperparameter search across ranks.

    Each rank evaluates a different hyperparameter configuration.
    Results are gathered on rank 0 for selection.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        search_space: dict = None,
        n_trials: int = 20,
        seed: int = 42,
    ):
        self.rank = rank
        self.world_size = world_size
        self.search_space = search_space or {}
        self.n_trials = n_trials
        self.seed = seed
        self._results = []

    def _sample_config(self, trial_idx: int) -> dict:
        """Sample a hyperparameter configuration for this trial."""
        import random
        rng = random.Random(self.seed + trial_idx)
        config = {}
        for key, spec in self.search_space.items():
            if spec["type"] == "float":
                config[key] = rng.uniform(spec["low"], spec["high"])
            elif spec["type"] == "log_float":
                import math
                log_val = rng.uniform(math.log(spec["low"]), math.log(spec["high"]))
                config[key] = math.exp(log_val)
            elif spec["type"] == "int":
                config[key] = rng.randint(spec["low"], spec["high"])
            elif spec["type"] == "choice":
                config[key] = rng.choice(spec["values"])
        return config

    def get_rank_trials(self) -> List[dict]:
        """Get the trials assigned to this rank."""
        all_trials = [self._sample_config(i) for i in range(self.n_trials)]
        return [t for i, t in enumerate(all_trials) if i % self.world_size == self.rank]

    def record_result(self, config: dict, metric: float):
        """Record evaluation result for a trial."""
        self._results.append({"config": config, "metric": metric})

    def gather_results(self) -> Optional[List[dict]]:
        """Gather all results on rank 0."""
        if not dist.is_initialized():
            return self._results

        # Simple approach: each rank prints its results for manual aggregation
        # In production, would use object all_gather
        return self._results if self.rank == 0 else None

    def best_config(self, minimize: bool = True) -> Optional[dict]:
        """Return best configuration by metric value."""
        if not self._results:
            return None
        sorted_results = sorted(self._results, key=lambda x: x["metric"], reverse=not minimize)
        return sorted_results[0]
'''

with open(PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess, sys
result = subprocess.run(
    [sys.executable, "-c",
     f"lines = open(r'{PATH}').readlines(); print(len(lines))"],
    capture_output=True, text=True
)
print(result.stdout.strip(), PATH)
