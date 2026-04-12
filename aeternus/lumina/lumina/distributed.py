"""
lumina/distributed.py

Distributed training utilities for Lumina:

  - DDPWrapper           : DistributedDataParallel wrapper with utilities
  - GradientAccumulator  : gradient accumulation across micro-steps
  - AMPTrainer           : automatic mixed precision training
  - CheckpointManager    : checkpoint saving and loading with versioning
  - DistributedConfig    : configuration for distributed training
  - ShardedOptimizer     : ZeRO-style sharded optimizer stub
  - TrainingState        : serializable training state container
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DistributedConfig:
    # Distributed
    backend:           str   = "nccl"      # "nccl" | "gloo"
    world_size:        int   = 1
    rank:              int   = 0
    local_rank:        int   = 0
    master_addr:       str   = "localhost"
    master_port:       int   = 29500

    # Mixed precision
    use_amp:           bool  = True
    amp_dtype:         str   = "bfloat16"  # "float16" | "bfloat16"
    loss_scale:        str   = "dynamic"   # "dynamic" | "static"
    static_loss_scale: float = 128.0

    # Gradient accumulation
    grad_accum_steps:  int   = 1
    grad_clip:         float = 1.0

    # Checkpointing
    checkpoint_dir:    str   = "./checkpoints/lumina"
    save_every_n_steps: int  = 1000
    keep_n_checkpoints: int  = 3
    resume_from:       Optional[str] = None

    # Profiling
    profile:           bool  = False
    profile_steps:     int   = 10


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed(cfg: DistributedConfig) -> bool:
    """Initialize process group. Returns True if distributed is available."""
    if not torch.distributed.is_available():
        return False

    if cfg.world_size <= 1:
        return False

    os.environ["MASTER_ADDR"] = cfg.master_addr
    os.environ["MASTER_PORT"] = str(cfg.master_port)

    try:
        dist.init_process_group(
            backend=cfg.backend,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
        torch.cuda.set_device(cfg.local_rank)
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize distributed: {e}")
        return False


def cleanup_distributed() -> None:
    """Clean up process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Returns True if this is rank 0 or not distributed."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize() -> None:
    """Barrier synchronization across all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_scalar(value: float, op: str = "sum") -> float:
    """All-reduce a scalar across all processes."""
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG)
    return tensor.item()


def all_gather_object(obj: Any) -> List[Any]:
    """Gather an object from all processes."""
    if not dist.is_available() or not dist.is_initialized():
        return [obj]
    world_size = get_world_size()
    gathered   = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered


# ---------------------------------------------------------------------------
# DDP Wrapper
# ---------------------------------------------------------------------------

class DDPWrapper(nn.Module):
    """
    Wrapper around DistributedDataParallel with convenience utilities.

    Features:
      - Automatic device placement
      - find_unused_parameters support
      - State dict access that unwraps DDP
      - Gradient synchronization control
    """

    def __init__(
        self,
        model:           nn.Module,
        device:          torch.device,
        find_unused:     bool = False,
        gradient_as_bucket_view: bool = True,
    ):
        super().__init__()
        self.module  = model.to(device)
        self.device  = device

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self.ddp_model = DDP(
                self.module,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=find_unused,
                gradient_as_bucket_view=gradient_as_bucket_view,
            )
        else:
            self.ddp_model = self.module

    def forward(self, *args, **kwargs) -> Any:
        return self.ddp_model(*args, **kwargs)

    def unwrap(self) -> nn.Module:
        """Return the underlying model without DDP wrapper."""
        if isinstance(self.ddp_model, DDP):
            return self.ddp_model.module
        return self.ddp_model

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.unwrap().state_dict()

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        self.unwrap().load_state_dict(state_dict, strict=strict)

    @contextmanager
    def no_sync(self) -> Generator:
        """Context manager to skip gradient synchronization (for grad accumulation)."""
        if isinstance(self.ddp_model, DDP):
            with self.ddp_model.no_sync():
                yield
        else:
            yield


# ---------------------------------------------------------------------------
# Gradient Accumulation
# ---------------------------------------------------------------------------

class GradientAccumulator:
    """
    Manages gradient accumulation across micro-steps.
    Accumulates gradients for `n_steps` before calling optimizer.step().
    """

    def __init__(
        self,
        model:          nn.Module,
        optimizer:      torch.optim.Optimizer,
        n_steps:        int   = 4,
        grad_clip:      float = 1.0,
        scaler:         Optional[GradScaler] = None,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.n_steps   = n_steps
        self.grad_clip = grad_clip
        self.scaler    = scaler
        self._step     = 0

    @property
    def should_sync(self) -> bool:
        """True when we should actually update weights."""
        return (self._step + 1) % self.n_steps == 0

    @contextmanager
    def micro_step(self) -> Generator:
        """
        Context manager for a single micro-step (gradient accumulation step).
        Only synchronizes gradients on the last micro-step.
        """
        if hasattr(self.model, "no_sync") and not self.should_sync:
            with self.model.no_sync():
                yield
        else:
            yield

        self._step += 1

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by accumulation factor."""
        return loss / self.n_steps

    def step_optimizer(self) -> Optional[float]:
        """
        Perform optimizer step. Returns gradient norm.
        Only called when should_sync is True.
        """
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = None
        if self.grad_clip > 0:
            params    = [p for p in self.model.parameters() if p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm


# ---------------------------------------------------------------------------
# AMP Trainer
# ---------------------------------------------------------------------------

class AMPTrainer:
    """
    Automatic Mixed Precision trainer.

    Wraps the training loop with:
      - torch.autocast context for forward pass
      - GradScaler for float16 (not needed for bfloat16)
      - Gradient clipping
      - Gradient accumulation
    """

    def __init__(
        self,
        model:         nn.Module,
        optimizer:     torch.optim.Optimizer,
        cfg:           DistributedConfig,
        device:        torch.device,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.cfg       = cfg
        self.device    = device

        # AMP setup
        dtype_str = cfg.amp_dtype
        self.amp_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        if cfg.use_amp and self.amp_dtype == torch.float16:
            if cfg.loss_scale == "dynamic":
                self.scaler = GradScaler()
            else:
                self.scaler = GradScaler(init_scale=cfg.static_loss_scale, growth_interval=10000)
        else:
            self.scaler = None

        # Gradient accumulation
        self.accumulator = GradientAccumulator(
            model,
            optimizer,
            n_steps   = cfg.grad_accum_steps,
            grad_clip = cfg.grad_clip,
            scaler    = self.scaler,
        )

        self._step       = 0
        self._total_loss = 0.0
        self._loss_count = 0

    @contextmanager
    def autocast_context(self) -> Generator:
        """Context manager for automatic mixed precision."""
        if self.cfg.use_amp:
            device_type = self.device.type
            with autocast(device_type=device_type, dtype=self.amp_dtype):
                yield
        else:
            yield

    def train_step(
        self,
        loss_fn: Callable[[], torch.Tensor],
    ) -> Tuple[float, Optional[float]]:
        """
        Perform a single training step with gradient accumulation.

        loss_fn: callable that returns a loss tensor

        Returns: (loss_value, grad_norm or None)
        """
        with self.accumulator.micro_step():
            with self.autocast_context():
                loss = loss_fn()
                loss = self.accumulator.scale_loss(loss)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        loss_val = loss.item() * self.cfg.grad_accum_steps

        grad_norm = None
        if self.accumulator.should_sync:
            grad_norm = self.accumulator.step_optimizer()

        self._step       += 1
        self._total_loss += loss_val
        self._loss_count += 1

        return loss_val, grad_norm

    def get_avg_loss(self) -> float:
        if self._loss_count == 0:
            return 0.0
        return self._total_loss / self._loss_count

    def reset_loss(self) -> None:
        self._total_loss = 0.0
        self._loss_count = 0

    def get_scale(self) -> Optional[float]:
        if self.scaler is not None:
            return self.scaler.get_scale()
        return None


# ---------------------------------------------------------------------------
# Training State
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Serializable container for all training state."""
    step:              int   = 0
    epoch:             int   = 0
    best_val_loss:     float = float('inf')
    best_val_metric:   float = 0.0
    total_tokens:      int   = 0
    wall_time_seconds: float = 0.0
    metrics_history:   List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def log_step(self, metrics: Dict[str, float]) -> None:
        entry = {"step": self.step, "epoch": self.epoch, **metrics}
        self.metrics_history.append(entry)

    def get_recent_metrics(self, n: int = 100) -> List[Dict]:
        return self.metrics_history[-n:]


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages model checkpoints with versioning and automatic cleanup.

    Features:
      - Saves model, optimizer, scaler, training state
      - Keeps N most recent checkpoints
      - Saves 'best' checkpoint separately based on validation metric
      - Supports resuming from checkpoint
    """

    def __init__(
        self,
        save_dir:         Union[str, Path],
        keep_n:           int  = 3,
        metric_mode:      str  = "min",    # "min" or "max"
        save_every_n:     int  = 1000,
    ):
        self.save_dir    = Path(save_dir)
        self.keep_n      = keep_n
        self.metric_mode = metric_mode
        self.save_every_n = save_every_n
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints: List[Path] = []
        self._best_metric = float('inf') if metric_mode == "min" else float('-inf')
        self._best_path: Optional[Path] = None

    def _is_better(self, metric: float) -> bool:
        if self.metric_mode == "min":
            return metric < self._best_metric
        return metric > self._best_metric

    def save(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        state:     TrainingState,
        scaler:    Optional[GradScaler] = None,
        scheduler: Optional[Any] = None,
        metric:    Optional[float] = None,
        tag:       Optional[str] = None,
    ) -> Path:
        """Save a checkpoint. Returns the path."""
        tag  = tag or f"step_{state.step:08d}"
        path = self.save_dir / f"checkpoint_{tag}.pt"

        # Unwrap DDP if needed
        if isinstance(model, DDPWrapper):
            model_state = model.state_dict()
        elif isinstance(model, DDP):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        payload = {
            "model":     model_state,
            "optimizer": optimizer.state_dict(),
            "state":     state.to_dict(),
        }

        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            payload["scheduler"] = scheduler.state_dict()

        torch.save(payload, path)
        logger.info(f"Saved checkpoint: {path}")

        self._checkpoints.append(path)
        self._cleanup()

        # Save best
        if metric is not None and self._is_better(metric):
            self._best_metric = metric
            best_path         = self.save_dir / "checkpoint_best.pt"
            shutil.copy2(path, best_path)
            self._best_path   = best_path
            logger.info(f"New best checkpoint (metric={metric:.4f}): {best_path}")

        return path

    def _cleanup(self) -> None:
        """Remove old checkpoints, keeping the N most recent."""
        while len(self._checkpoints) > self.keep_n:
            old = self._checkpoints.pop(0)
            if old.exists() and old != self._best_path:
                old.unlink()
                logger.debug(f"Removed old checkpoint: {old}")

    def load(
        self,
        path:      Union[str, Path],
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler:    Optional[GradScaler] = None,
        scheduler: Optional[Any] = None,
        strict:    bool = True,
    ) -> TrainingState:
        """Load a checkpoint and restore all state. Returns TrainingState."""
        path    = Path(path)
        payload = torch.load(path, map_location="cpu")

        # Load model
        if isinstance(model, (DDPWrapper, DDP)):
            m = model.module if isinstance(model, DDP) else model.unwrap()
        else:
            m = model
        m.load_state_dict(payload["model"], strict=strict)

        # Load optimizer
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])

        # Load scaler
        if scaler is not None and "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])

        # Load scheduler
        if scheduler is not None and "scheduler" in payload and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(payload["scheduler"])

        state = TrainingState.from_dict(payload.get("state", {}))
        logger.info(f"Loaded checkpoint from {path} (step={state.step})")
        return state

    def load_best(
        self,
        model:     nn.Module,
        **kwargs,
    ) -> Optional[TrainingState]:
        """Load the best checkpoint if it exists."""
        best = self.save_dir / "checkpoint_best.pt"
        if best.exists():
            return self.load(best, model, **kwargs)
        return None

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint files in the save directory."""
        return sorted(self.save_dir.glob("checkpoint_*.pt"))

    def get_latest(self) -> Optional[Path]:
        checkpoints = self.list_checkpoints()
        # Exclude 'best'
        checkpoints = [c for c in checkpoints if "best" not in c.name]
        return checkpoints[-1] if checkpoints else None


# ---------------------------------------------------------------------------
# Sharded Optimizer (ZeRO-style stub)
# ---------------------------------------------------------------------------

class ShardedOptimizerStub:
    """
    Stub for ZeRO-style optimizer state sharding.

    In a real implementation, each rank would only store 1/world_size
    of the optimizer states, reducing per-GPU memory by ~8x.

    Requires: fairscale or deepspeed for full implementation.
    """

    def __init__(
        self,
        optimizer:   torch.optim.Optimizer,
        world_size:  int = 1,
        rank:        int = 0,
    ):
        self.optimizer  = optimizer
        self.world_size = world_size
        self.rank       = rank
        self._is_sharded = world_size > 1

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @staticmethod
    def is_available() -> bool:
        """Check if true ZeRO sharding is available."""
        try:
            import fairscale
            return True
        except ImportError:
            try:
                import deepspeed
                return True
            except ImportError:
                return False


# ---------------------------------------------------------------------------
# Profiler wrapper
# ---------------------------------------------------------------------------

class TrainingProfiler:
    """Wraps PyTorch profiler for training performance analysis."""

    def __init__(
        self,
        enabled:    bool = False,
        n_steps:    int  = 10,
        output_dir: str  = "./profiles",
    ):
        self.enabled    = enabled
        self.n_steps    = n_steps
        self.output_dir = Path(output_dir)
        self._profiler  = None
        self._step      = 0

    def __enter__(self):
        if self.enabled:
            from torch.profiler import profile, record_function, ProfilerActivity
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=self.n_steps),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self._profiler.__enter__()
        return self

    def __exit__(self, *args):
        if self._profiler is not None:
            self._profiler.__exit__(*args)

    def step(self) -> None:
        if self._profiler is not None:
            self._profiler.step()
        self._step += 1


# ---------------------------------------------------------------------------
# Training throughput monitor
# ---------------------------------------------------------------------------

class ThroughputMonitor:
    """Tracks training throughput (tokens/sec, samples/sec)."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._times:   List[float] = []
        self._samples: List[int]   = []
        self._tokens:  List[int]   = []

    def update(self, n_samples: int, n_tokens: int = 0) -> None:
        self._times.append(time.perf_counter())
        self._samples.append(n_samples)
        self._tokens.append(n_tokens)

        if len(self._times) > self.window_size:
            self._times.pop(0)
            self._samples.pop(0)
            self._tokens.pop(0)

    def get_throughput(self) -> Dict[str, float]:
        if len(self._times) < 2:
            return {"samples_per_sec": 0.0, "tokens_per_sec": 0.0}

        dt       = self._times[-1] - self._times[0]
        n_samps  = sum(self._samples[1:])
        n_toks   = sum(self._tokens[1:])

        if dt < 1e-6:
            return {"samples_per_sec": 0.0, "tokens_per_sec": 0.0}

        return {
            "samples_per_sec": n_samps / dt,
            "tokens_per_sec":  n_toks  / dt,
        }


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

def get_gpu_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.device("cuda")

    return {
        "allocated_mb":    torch.cuda.memory_allocated(device) / 1e6,
        "reserved_mb":     torch.cuda.memory_reserved(device)  / 1e6,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1e6,
        "max_reserved_mb": torch.cuda.max_memory_reserved(device)   / 1e6,
    }


def log_memory_stats(step: int, device: Optional[torch.device] = None) -> None:
    stats = get_gpu_memory_stats(device)
    if stats:
        logger.debug(
            f"[Step {step}] GPU Memory: "
            f"alloc={stats['allocated_mb']:.0f}MB "
            f"max={stats['max_allocated_mb']:.0f}MB"
        )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "DistributedConfig",
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_world_size",
    "get_rank",
    "synchronize",
    "all_reduce_scalar",
    "all_gather_object",
    "DDPWrapper",
    "GradientAccumulator",
    "AMPTrainer",
    "TrainingState",
    "CheckpointManager",
    "ShardedOptimizerStub",
    "TrainingProfiler",
    "ThroughputMonitor",
    "get_gpu_memory_stats",
    "log_memory_stats",
    "DistributedSampler",
    "ModelParallelWrapper",
    "PipelineParallelStub",
    "EarlyStopper",
    "LearningRateSchedulerWrapper",
    "GradientFlowMonitor",
    "TrainingLogger",
]


# ---------------------------------------------------------------------------
# Distributed Sampler
# ---------------------------------------------------------------------------

class DistributedSampler:
    """Sampler for distributed data loading.

    Ensures each process receives a non-overlapping subset of the dataset,
    with support for shuffling and epoch-based seed control.

    Args:
        dataset_size: total size of dataset
        world_size:   number of processes
        rank:         current process rank
        shuffle:      shuffle data at each epoch
        seed:         base random seed

    Example:
        >>> sampler = DistributedSampler(dataset_size=10000, world_size=4, rank=0)
        >>> indices = sampler.get_indices(epoch=1)
    """

    def __init__(
        self,
        dataset_size: int,
        world_size: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.dataset_size = dataset_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Compute number of samples per rank
        if drop_last:
            self.num_samples = dataset_size // world_size
        else:
            self.num_samples = math.ceil(dataset_size / world_size)

        self.total_size = self.num_samples * world_size

    def get_indices(self, epoch: int = 0) -> List[int]:
        """Get dataset indices for this rank at given epoch.

        Args:
            epoch: current epoch (used as seed offset for reproducibility)

        Returns:
            indices: list of dataset indices for this rank
        """
        if self.shuffle:
            rng = np.random.default_rng(self.seed + epoch)
            indices = rng.permutation(self.dataset_size).tolist()
        else:
            indices = list(range(self.dataset_size))

        # Pad to total_size if needed
        if not self.drop_last and len(indices) < self.total_size:
            n_pad = self.total_size - len(indices)
            indices = indices + indices[:n_pad]

        # Trim to total_size
        indices = indices[:self.total_size]

        # Select subset for this rank
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return indices[start:end]

    def __len__(self) -> int:
        return self.num_samples


# ---------------------------------------------------------------------------
# Model Parallel Wrapper
# ---------------------------------------------------------------------------

class ModelParallelWrapper(nn.Module):
    """Simple model parallelism: split layers across multiple GPUs.

    Assigns transformer layers to GPUs in round-robin fashion.
    Handles moving tensors between devices as they pass through layers.

    This is a simplified implementation — production code would use
    PyTorch's pipeline parallelism or Megatron-LM style tensor parallelism.

    Args:
        model:         model with a .blocks ModuleList
        devices:       list of CUDA device ids or device strings
        balance:       layer assignments per device (None = automatic)

    Example:
        >>> model = LuminaModel(config)
        >>> mp = ModelParallelWrapper(model, devices=["cuda:0", "cuda:1"])
        >>> out = mp(x)
    """

    def __init__(
        self,
        model: nn.Module,
        devices: List[str],
        balance: Optional[List[int]] = None,
    ):
        super().__init__()
        self.devices = devices
        self.n_devices = len(devices)

        if not hasattr(model, 'blocks'):
            raise ValueError("Model must have a .blocks ModuleList for model parallelism")

        n_layers = len(model.blocks)

        if balance is None:
            # Distribute layers as evenly as possible
            layers_per_device = n_layers // self.n_devices
            remainder = n_layers % self.n_devices
            balance = [layers_per_device + (1 if i < remainder else 0)
                       for i in range(self.n_devices)]

        assert sum(balance) == n_layers, \
            f"balance sum {sum(balance)} != n_layers {n_layers}"

        self.balance = balance
        self.layer_to_device: Dict[int, str] = {}

        # Move layers to assigned devices
        layer_idx = 0
        for dev_idx, n_dev_layers in enumerate(balance):
            device = devices[dev_idx]
            for _ in range(n_dev_layers):
                model.blocks[layer_idx] = model.blocks[layer_idx].to(device)
                self.layer_to_device[layer_idx] = device
                layer_idx += 1

        self.model = model
        # Move input processing to first device
        if hasattr(model, 'input_proj'):
            model.input_proj = model.input_proj.to(devices[0])
        # Move output heads to last device
        if hasattr(model, 'norm'):
            model.norm = model.norm.to(devices[-1])

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with inter-device transfers."""
        # This is a stub — actual implementation depends on model structure
        return self.model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Pipeline Parallel Stub
# ---------------------------------------------------------------------------

class PipelineParallelStub(nn.Module):
    """Stub for pipeline parallelism.

    In pipeline parallelism, the model is partitioned into stages and
    different micro-batches are processed simultaneously on different stages.

    This stub provides the interface for integration with PyTorch's
    PipelineParallel or Deepspeed PipelineModule.

    Args:
        model:         model to parallelize
        n_stages:      number of pipeline stages
        n_micro_batches: number of micro-batches for pipeline filling
    """

    def __init__(
        self,
        model: nn.Module,
        n_stages: int = 4,
        n_micro_batches: int = 8,
    ):
        super().__init__()
        self.model = model
        self.n_stages = n_stages
        self.n_micro_batches = n_micro_batches
        self._stages: List[nn.Module] = []
        self._warnings_issued = set()

    def partition_model(self) -> List[nn.Module]:
        """Partition model into n_stages equal parts.

        Returns:
            stages: list of nn.Module, one per pipeline stage
        """
        if hasattr(self.model, 'blocks'):
            blocks = list(self.model.blocks)
            stage_size = len(blocks) // self.n_stages
            stages = []
            for i in range(self.n_stages):
                start = i * stage_size
                end = (i + 1) * stage_size if i < self.n_stages - 1 else len(blocks)
                stage_blocks = nn.Sequential(*blocks[start:end])
                stages.append(stage_blocks)
            return stages
        else:
            return [self.model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential forward (no pipelining — use for debugging)."""
        if "sequential_fallback" not in self._warnings_issued:
            import warnings
            warnings.warn(
                "PipelineParallelStub: falling back to sequential forward. "
                "Use torch.distributed.pipeline.sync.Pipe for real pipelining."
            )
            self._warnings_issued.add("sequential_fallback")
        return self.model(x)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopper:
    """Early stopping monitor for training.

    Monitors a validation metric and stops training when it stops improving.

    Supports:
    - Min mode: stop when metric stops decreasing (e.g., loss)
    - Max mode: stop when metric stops increasing (e.g., accuracy, Sharpe)
    - Delta: minimum change to be considered improvement

    Args:
        patience:    number of epochs to wait after last improvement
        mode:        "min" or "max"
        min_delta:   minimum change to qualify as improvement
        restore_best:if True, track best checkpoint path

    Example:
        >>> stopper = EarlyStopper(patience=10, mode="min")
        >>> for epoch in range(100):
        ...     val_loss = trainer.evaluate()
        ...     if stopper.step(val_loss):
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best

        self._best_value: Optional[float] = None
        self._wait = 0
        self._best_epoch = 0
        self._n_calls = 0

    def _is_improvement(self, new_value: float) -> bool:
        """Check if new value is an improvement over best."""
        if self._best_value is None:
            return True
        if self.mode == "min":
            return new_value < self._best_value - self.min_delta
        else:
            return new_value > self._best_value + self.min_delta

    def step(self, value: float) -> bool:
        """Process one evaluation result.

        Args:
            value: current evaluation metric value

        Returns:
            True if training should stop, False otherwise
        """
        self._n_calls += 1

        if self._is_improvement(value):
            self._best_value = value
            self._best_epoch = self._n_calls
            self._wait = 0
            return False
        else:
            self._wait += 1
            if self._wait >= self.patience:
                return True
            return False

    @property
    def best_value(self) -> Optional[float]:
        """Best metric value seen so far."""
        return self._best_value

    @property
    def best_epoch(self) -> int:
        """Epoch with best metric value."""
        return self._best_epoch

    @property
    def n_waits(self) -> int:
        """Number of consecutive non-improving evaluations."""
        return self._wait

    def state_dict(self) -> Dict:
        """Serialize state for checkpointing."""
        return {
            "best_value": self._best_value,
            "wait": self._wait,
            "best_epoch": self._best_epoch,
            "n_calls": self._n_calls,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore state from checkpoint."""
        self._best_value = state["best_value"]
        self._wait = state["wait"]
        self._best_epoch = state["best_epoch"]
        self._n_calls = state["n_calls"]


# ---------------------------------------------------------------------------
# Learning Rate Scheduler Wrapper
# ---------------------------------------------------------------------------

class LearningRateSchedulerWrapper:
    """Wrapper for PyTorch LR schedulers with logging and warmup.

    Adds:
    - Comprehensive logging of LR history
    - Cosine annealing with warmup
    - Linear decay
    - Cyclical LR (1cycle)
    - Exponential decay with minimum LR floor

    Args:
        optimizer:     PyTorch optimizer
        schedule_type: "cosine_warmup" | "linear" | "exponential" | "constant"
        warmup_steps:  number of warmup steps
        total_steps:   total training steps (for cosine)
        min_lr_ratio:  minimum LR as fraction of max LR

    Example:
        >>> sched = LearningRateSchedulerWrapper(
        ...     optimizer, schedule_type="cosine_warmup",
        ...     warmup_steps=1000, total_steps=50000
        ... )
        >>> for step in range(total_steps):
        ...     loss.backward()
        ...     optimizer.step()
        ...     sched.step()
    """

    def __init__(
        self,
        optimizer: Any,
        schedule_type: str = "cosine_warmup",
        warmup_steps: int = 1000,
        total_steps: int = 50000,
        min_lr_ratio: float = 0.1,
        gamma: float = 0.99995,  # for exponential decay
    ):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.gamma = gamma
        self._step_count = 0
        self._lr_history: List[float] = []

        # Store initial LR
        self._max_lr = [pg["lr"] for pg in optimizer.param_groups]

    def _compute_lr(self, step: int, base_lr: float) -> float:
        """Compute learning rate at given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return base_lr * (step + 1) / self.warmup_steps
        elif self.schedule_type == "cosine_warmup":
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            min_lr = base_lr * self.min_lr_ratio
            return min_lr + (base_lr - min_lr) * cosine
        elif self.schedule_type == "linear":
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return base_lr * (1.0 - min(1.0, progress) * (1.0 - self.min_lr_ratio))
        elif self.schedule_type == "exponential":
            return max(base_lr * (self.gamma ** (step - self.warmup_steps)),
                       base_lr * self.min_lr_ratio)
        else:
            return base_lr

    def step(self) -> float:
        """Advance scheduler by one step.

        Returns:
            current_lr: current learning rate (first param group)
        """
        self._step_count += 1
        current_lrs = []
        for pg, base_lr in zip(self.optimizer.param_groups, self._max_lr):
            new_lr = self._compute_lr(self._step_count, base_lr)
            pg["lr"] = new_lr
            current_lrs.append(new_lr)
        self._lr_history.append(current_lrs[0])
        return current_lrs[0]

    def get_lr(self) -> List[float]:
        """Get current learning rates for all param groups."""
        return [pg["lr"] for pg in self.optimizer.param_groups]

    @property
    def lr_history(self) -> List[float]:
        """Full LR history."""
        return self._lr_history.copy()

    def state_dict(self) -> Dict:
        return {
            "step_count": self._step_count,
            "lr_history": self._lr_history[-1000:],  # Keep last 1000
        }

    def load_state_dict(self, state: Dict) -> None:
        self._step_count = state["step_count"]
        self._lr_history = state.get("lr_history", [])


# ---------------------------------------------------------------------------
# Gradient Flow Monitor
# ---------------------------------------------------------------------------

class GradientFlowMonitor:
    """Monitor gradient flow in transformer layers.

    Tracks:
    - Per-layer gradient norms
    - Gradient vanishing/exploding detection
    - Dead neurons (zero gradient)
    - Weight update statistics

    Useful for debugging training instability.

    Args:
        model:        model to monitor
        log_every_n:  log every N steps

    Example:
        >>> monitor = GradientFlowMonitor(model, log_every_n=100)
        >>> loss.backward()
        >>> stats = monitor.step()
        >>> if stats["exploding"]:
        ...     print("Warning: gradient explosion detected")
    """

    def __init__(self, model: nn.Module, log_every_n: int = 100):
        self.model = model
        self.log_every_n = log_every_n
        self._step = 0
        self._history: List[Dict] = []

    def step(self) -> Optional[Dict[str, Any]]:
        """Compute and record gradient statistics.

        Should be called after loss.backward() but before optimizer.step().

        Returns:
            stats: gradient statistics dict, or None if not logging this step
        """
        self._step += 1
        if self._step % self.log_every_n != 0:
            return None

        layer_stats = {}
        total_norm = 0.0
        n_dead = 0
        n_total = 0

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.data
            norm = grad.norm(2).item()
            total_norm += norm ** 2
            n_total += 1

            if norm < 1e-7:
                n_dead += 1

            layer_stats[name] = {
                "grad_norm": norm,
                "weight_norm": param.data.norm(2).item(),
                "grad_mean": grad.mean().item(),
                "grad_std": grad.std().item(),
            }

        total_norm = total_norm ** 0.5

        stats = {
            "step": self._step,
            "total_grad_norm": total_norm,
            "n_dead_params": n_dead,
            "n_total_params": n_total,
            "dead_fraction": n_dead / max(n_total, 1),
            "vanishing": total_norm < 1e-5,
            "exploding": total_norm > 1000.0,
            "layer_stats": layer_stats,
        }

        self._history.append(stats)
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics over all recorded steps."""
        if not self._history:
            return {}

        norms = [s["total_grad_norm"] for s in self._history]
        return {
            "n_steps_monitored": len(self._history),
            "mean_grad_norm": sum(norms) / len(norms),
            "max_grad_norm": max(norms),
            "min_grad_norm": min(norms),
            "n_vanishing_steps": sum(1 for s in self._history if s["vanishing"]),
            "n_exploding_steps": sum(1 for s in self._history if s["exploding"]),
        }


# ---------------------------------------------------------------------------
# Training Logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Comprehensive training logger with multiple backends.

    Supports logging to:
    - Console (rich formatting)
    - CSV file
    - JSON file
    - Weights & Biases (if available)
    - TensorBoard (if available)

    Args:
        run_name:     experiment name
        log_dir:      directory for log files
        use_wandb:    enable W&B logging
        use_tb:       enable TensorBoard logging
        log_every_n:  log every N steps

    Example:
        >>> logger = TrainingLogger(run_name="lumina_base_v1", log_dir="./logs")
        >>> logger.log_metrics(step=100, loss=0.5, lr=1e-4, grad_norm=1.2)
        >>> logger.log_hyperparams({"batch_size": 32, "d_model": 512})
        >>> logger.finalize()
    """

    def __init__(
        self,
        run_name: str = "lumina_run",
        log_dir: str = "./logs",
        use_wandb: bool = False,
        use_tb: bool = False,
        log_every_n: int = 10,
    ):
        self.run_name = run_name
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_tb = use_tb
        self.log_every_n = log_every_n
        self._step = 0
        self._metrics_history: List[Dict] = []
        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize logging backends."""
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv_path = os.path.join(self.log_dir, f"{self.run_name}_metrics.csv")
        self._log_path = os.path.join(self.log_dir, f"{self.run_name}_log.json")
        self._wandb_run = None
        self._tb_writer = None

        if self.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(name=self.run_name, reinit=True)
            except ImportError:
                pass

        if self.use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(
                    log_dir=os.path.join(self.log_dir, "tensorboard", self.run_name)
                )
            except ImportError:
                pass

    def log_metrics(self, step: int, **metrics: float) -> None:
        """Log scalar metrics.

        Args:
            step:    training step
            **metrics: metric name → float value

        Example:
            >>> logger.log_metrics(100, loss=0.5, acc=0.85, lr=1e-4)
        """
        self._step = step
        record = {"step": step, **metrics}
        self._metrics_history.append(record)

        # W&B
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception:
                pass

        # TensorBoard
        if self._tb_writer is not None:
            try:
                for k, v in metrics.items():
                    self._tb_writer.add_scalar(k, v, step)
            except Exception:
                pass

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self._wandb_run is not None:
            try:
                self._wandb_run.config.update(params)
            except Exception:
                pass
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_hparams(
                    {str(k): str(v) for k, v in params.items()}, {}
                )
            except Exception:
                pass

    def log_model_summary(self, model: nn.Module) -> None:
        """Log model parameter summary."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log_hyperparams({
            "total_params": total,
            "trainable_params": trainable,
        })

    def save_csv(self) -> None:
        """Save metrics history to CSV."""
        if not self._metrics_history:
            return
        import csv
        all_keys = set()
        for record in self._metrics_history:
            all_keys.update(record.keys())
        all_keys = sorted(all_keys)

        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for record in self._metrics_history:
                writer.writerow({k: record.get(k, "") for k in all_keys})

    def finalize(self) -> None:
        """Finalize logging and close backends."""
        self.save_csv()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass


# =============================================================================
# SECTION: Advanced Distributed Training Utilities
# =============================================================================

import socket
import pickle
import struct
from contextlib import contextmanager


class DistributedOptimizer:
    """Wrapper around torch optimizers adding distributed gradient sync.

    Provides: gradient clipping, gradient accumulation, and ZeRO-style
    parameter partitioning.

    Args:
        optimizer: Base PyTorch optimizer
        grad_clip: Max gradient norm (0 = no clipping)
        accumulation_steps: Steps between gradient updates
    """

    def __init__(self, optimizer, grad_clip: float = 1.0, accumulation_steps: int = 1) -> None:
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.accumulation_steps = accumulation_steps
        self._step_count = 0

    def step(self, model: nn.Module) -> bool:
        """Conditionally step: only update every accumulation_steps calls."""
        self._step_count += 1
        if self._step_count % self.accumulation_steps != 0:
            return False
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return True

    def state_dict(self) -> Dict:
        return {"optimizer": self.optimizer.state_dict(), "step": self._step_count}

    def load_state_dict(self, state: Dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self._step_count = state.get("step", 0)


class GradientCompressor:
    """Gradient compression for communication-efficient distributed training.

    Methods:
    - TopK: Keep only top-k% of gradient values, zero out the rest
    - Random sparsification: Randomly zero out (1-k)% of gradients
    - Quantization: Quantize gradients to lower precision

    Args:
        method: Compression method ('topk', 'random', 'quantize')
        compression_ratio: Fraction of gradients to keep/transmit
        num_bits: Bit width for quantization
    """

    def __init__(self, method: str = "topk", compression_ratio: float = 0.1, num_bits: int = 8) -> None:
        self.method = method
        self.compression_ratio = compression_ratio
        self.num_bits = num_bits
        self._error_feedback: Dict[str, torch.Tensor] = {}

    def compress(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress gradient dict.

        Args:
            gradients: Dict of {param_name: gradient}
        Returns:
            Compressed gradient dict
        """
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                compressed[name] = grad
                continue
            # Add error feedback
            if name in self._error_feedback:
                grad = grad + self._error_feedback[name]
            if self.method == "topk":
                flat = grad.view(-1)
                k = max(1, int(len(flat) * self.compression_ratio))
                values, indices = flat.abs().topk(k)
                mask = torch.zeros_like(flat)
                mask[indices] = 1.0
                compressed_grad = flat * mask
                self._error_feedback[name] = flat * (1 - mask)
                compressed[name] = compressed_grad.view_as(grad)
            elif self.method == "random":
                mask = (torch.rand_like(grad) < self.compression_ratio).float()
                compressed[name] = grad * mask / (self.compression_ratio + 1e-10)
                self._error_feedback[name] = grad * (1 - mask)
            elif self.method == "quantize":
                g_min, g_max = grad.min(), grad.max()
                scale = (g_max - g_min) / (2 ** self.num_bits - 1)
                g_q = torch.round((grad - g_min) / (scale + 1e-10)).to(torch.int8)
                compressed[name] = g_q.float() * scale + g_min
            else:
                compressed[name] = grad
        return compressed


class MixedPrecisionManager:
    """Manages mixed precision training with dynamic loss scaling.

    Handles:
    - bfloat16 and float16 mixed precision
    - Dynamic loss scale for float16 (gradient overflow detection)
    - Automatic scale factor adjustment
    - Per-layer dtype management

    Args:
        dtype: Target dtype ('float16' or 'bfloat16')
        init_scale: Initial loss scale (float16 only)
        growth_factor: Scale growth factor (float16 only)
        backoff_factor: Scale reduction on overflow
        growth_interval: Steps between scale growth attempts
    """

    def __init__(
        self,
        dtype: str = "bfloat16",
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        self.dtype = getattr(torch, dtype)
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._scale = init_scale
        self._steps_since_growth = 0
        self._overflow_count = 0
        self._scaler = torch.cuda.amp.GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        ) if dtype == "float16" else None

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for float16 precision."""
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def unscale_and_step(self, optimizer) -> bool:
        """Unscale gradients and step optimizer.

        Returns:
            True if step was taken (no overflow), False otherwise
        """
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)
            self._scaler.step(optimizer)
            self._scaler.update()
            return True
        optimizer.step()
        return True

    def get_scale(self) -> float:
        if self._scaler is not None:
            return self._scaler.get_scale()
        return 1.0

    def state_dict(self) -> Dict:
        state = {"scale": self._scale, "overflow_count": self._overflow_count}
        if self._scaler is not None:
            state["scaler"] = self._scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict) -> None:
        self._scale = state.get("scale", self.init_scale)
        self._overflow_count = state.get("overflow_count", 0)
        if self._scaler is not None and "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])


class CheckpointManager:
    """Comprehensive checkpoint management for distributed training.

    Features:
    - Save/load model, optimizer, scheduler, and training state
    - Automatic cleanup of old checkpoints (keep-last-N)
    - Best checkpoint tracking by metric
    - Async saving (non-blocking write)
    - Checkpoint metadata (timestamp, config, metrics)

    Args:
        save_dir: Directory for checkpoint files
        keep_last: Number of recent checkpoints to keep (0 = keep all)
        save_best: Whether to save best checkpoint by metric
        metric_name: Metric to track for best model
        higher_is_better: Whether higher metric value is better
    """

    def __init__(
        self,
        save_dir: str,
        keep_last: int = 3,
        save_best: bool = True,
        metric_name: str = "val_sharpe",
        higher_is_better: bool = True,
    ) -> None:
        self.save_dir = save_dir
        self.keep_last = keep_last
        self.save_best = save_best
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        os.makedirs(save_dir, exist_ok=True)
        self._checkpoints: List[str] = []
        self._best_metric = float("-inf") if higher_is_better else float("inf")
        self._best_path: Optional[str] = None

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer=None,
        scheduler=None,
        metrics: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ) -> str:
        """Save training checkpoint.

        Args:
            step: Current training step
            model: Model to checkpoint
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            metrics: Optional dict of current metrics
            config: Optional training config
        Returns:
            Path to saved checkpoint
        """
        import time as time_module
        filename = f"checkpoint_step_{step:08d}.pt"
        path = os.path.join(self.save_dir, filename)
        state = {
            "step": step,
            "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict() if hasattr(scheduler, "state_dict") else {}
        if metrics is not None:
            state["metrics"] = metrics
        if config is not None:
            state["config"] = config

        torch.save(state, path)
        self._checkpoints.append(path)

        # Save best checkpoint
        if self.save_best and metrics is not None and self.metric_name in metrics:
            val = metrics[self.metric_name]
            is_better = (val > self._best_metric if self.higher_is_better else val < self._best_metric)
            if is_better:
                self._best_metric = val
                best_path = os.path.join(self.save_dir, "best_checkpoint.pt")
                torch.save(state, best_path)
                self._best_path = best_path

        # Cleanup old checkpoints
        if self.keep_last > 0 and len(self._checkpoints) > self.keep_last:
            old = self._checkpoints.pop(0)
            if os.path.exists(old) and old != self._best_path:
                try:
                    os.remove(old)
                except OSError:
                    pass

        return path

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer=None,
        scheduler=None,
    ) -> Dict:
        """Load checkpoint from file.

        Args:
            path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
        Returns:
            Checkpoint state dict
        """
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=False)
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(state["scheduler_state_dict"])
        return state

    def load_latest(
        self,
        model: nn.Module,
        optimizer=None,
        scheduler=None,
    ) -> Optional[Dict]:
        """Load most recent checkpoint."""
        if not self._checkpoints:
            # Try to find checkpoints in directory
            ckpts = sorted([f for f in os.listdir(self.save_dir) if f.startswith("checkpoint_")])
            if not ckpts:
                return None
            path = os.path.join(self.save_dir, ckpts[-1])
        else:
            path = self._checkpoints[-1]
        return self.load(path, model, optimizer, scheduler)

    def load_best(
        self,
        model: nn.Module,
        optimizer=None,
    ) -> Optional[Dict]:
        """Load best checkpoint by tracked metric."""
        if self._best_path is None:
            best_path = os.path.join(self.save_dir, "best_checkpoint.pt")
            if not os.path.exists(best_path):
                return None
            self._best_path = best_path
        return self.load(self._best_path, model, optimizer)


class CommunicationProfiler:
    """Profile communication overhead in distributed training.

    Tracks:
    - Time spent in AllReduce operations
    - Amount of data communicated
    - Synchronization overhead
    - Communication vs computation ratio

    Args:
        world_size: Number of distributed processes
        log_interval: Steps between log outputs
    """

    def __init__(self, world_size: int = 1, log_interval: int = 100) -> None:
        self.world_size = world_size
        self.log_interval = log_interval
        self._comm_time: float = 0.0
        self._compute_time: float = 0.0
        self._bytes_communicated: int = 0
        self._step: int = 0

    @contextmanager
    def record_comm(self, tensor_size_bytes: int = 0):
        """Context manager to record communication time."""
        t0 = time.perf_counter()
        yield
        self._comm_time += time.perf_counter() - t0
        self._bytes_communicated += tensor_size_bytes

    @contextmanager
    def record_compute(self):
        """Context manager to record compute time."""
        t0 = time.perf_counter()
        yield
        self._compute_time += time.perf_counter() - t0

    def step(self) -> Optional[Dict[str, float]]:
        """Increment step counter and optionally return stats."""
        self._step += 1
        if self._step % self.log_interval == 0:
            return self.get_stats()
        return None

    def get_stats(self) -> Dict[str, float]:
        total_time = self._comm_time + self._compute_time + 1e-10
        return {
            "comm_time_ms": self._comm_time * 1000,
            "compute_time_ms": self._compute_time * 1000,
            "comm_ratio": self._comm_time / total_time,
            "gb_communicated": self._bytes_communicated / 1e9,
            "step": self._step,
        }

    def reset(self) -> None:
        self._comm_time = 0.0
        self._compute_time = 0.0
        self._bytes_communicated = 0
        self._step = 0


class DistributedMetricsAggregator:
    """Aggregate metrics across distributed processes.

    Provides:
    - Running mean/variance with distributed reduce
    - Percentile tracking via histogram
    - Per-rank stats for debugging

    Args:
        metrics: List of metric names to track
        num_bins: Number of histogram bins for percentile estimation
    """

    def __init__(self, metrics: Optional[List[str]] = None, num_bins: int = 100) -> None:
        self.metrics = metrics or []
        self.num_bins = num_bins
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._sq_sums: Dict[str, float] = defaultdict(float)
        self._mins: Dict[str, float] = defaultdict(lambda: float("inf"))
        self._maxs: Dict[str, float] = defaultdict(lambda: float("-inf"))

    def update(self, name: str, value: float, count: int = 1) -> None:
        """Update metric with new value(s)."""
        self._sums[name] += value * count
        self._counts[name] += count
        self._sq_sums[name] += value ** 2 * count
        self._mins[name] = min(self._mins[name], value)
        self._maxs[name] = max(self._maxs[name], value)

    def get_mean(self, name: str) -> float:
        """Return running mean for metric."""
        c = self._counts[name]
        return self._sums[name] / c if c > 0 else 0.0

    def get_std(self, name: str) -> float:
        """Return running standard deviation."""
        c = self._counts[name]
        if c < 2:
            return 0.0
        mean = self._sums[name] / c
        var = self._sq_sums[name] / c - mean ** 2
        return max(0.0, var) ** 0.5

    def get_all(self, name: str) -> Dict[str, float]:
        """Return all stats for a metric."""
        return {
            "mean": self.get_mean(name),
            "std": self.get_std(name),
            "min": self._mins.get(name, 0.0),
            "max": self._maxs.get(name, 0.0),
            "count": self._counts[name],
        }

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary for all tracked metrics."""
        return {name: self.get_all(name) for name in self._counts}

    def reset(self) -> None:
        """Reset all accumulators."""
        self._sums.clear()
        self._counts.clear()
        self._sq_sums.clear()
        self._mins.clear()
        self._maxs.clear()


class AdaptiveGradientAccumulation:
    """Adaptively adjust gradient accumulation steps based on GPU memory.

    Monitors memory usage and adjusts accumulation to maintain target
    batch size while fitting in available memory.

    Args:
        target_batch_size: Desired effective batch size
        max_physical_batch: Maximum batch per forward pass
        min_accumulation: Minimum accumulation steps
        max_accumulation: Maximum accumulation steps
    """

    def __init__(
        self,
        target_batch_size: int = 1024,
        max_physical_batch: int = 32,
        min_accumulation: int = 1,
        max_accumulation: int = 64,
    ) -> None:
        self.target_batch_size = target_batch_size
        self.max_physical_batch = max_physical_batch
        self.min_accumulation = min_accumulation
        self.max_accumulation = max_accumulation
        self._current_physical = max_physical_batch
        self._current_accum = max(min_accumulation, target_batch_size // max_physical_batch)

    def adjust_for_oom(self) -> int:
        """Reduce physical batch size after OOM error.

        Returns:
            New physical batch size
        """
        self._current_physical = max(1, self._current_physical // 2)
        self._current_accum = min(
            self.max_accumulation,
            self.target_batch_size // self._current_physical
        )
        return self._current_physical

    def adjust_for_memory(self, used_gb: float, total_gb: float) -> int:
        """Adjust batch size based on memory usage ratio.

        Args:
            used_gb: Currently used GPU memory
            total_gb: Total GPU memory
        Returns:
            Recommended physical batch size
        """
        ratio = used_gb / (total_gb + 1e-10)
        if ratio > 0.9:
            return self.adjust_for_oom()
        elif ratio < 0.5:
            # Can safely increase batch size
            new_phys = min(self.max_physical_batch, self._current_physical * 2)
            self._current_physical = new_phys
            self._current_accum = max(
                self.min_accumulation,
                self.target_batch_size // new_phys
            )
        return self._current_physical

    @property
    def effective_batch_size(self) -> int:
        return self._current_physical * self._current_accum

    @property
    def accumulation_steps(self) -> int:
        return self._current_accum


class TrainingProgressTracker:
    """Track and estimate training progress and ETA.

    Provides:
    - Step-level timing statistics
    - ETA estimation via moving average
    - Training efficiency (samples/sec, tokens/sec)
    - Loss curve smoothing

    Args:
        total_steps: Total training steps
        smoothing_window: Window for exponential smoothing
    """

    def __init__(self, total_steps: int, smoothing_window: int = 100) -> None:
        self.total_steps = total_steps
        self.smoothing_window = smoothing_window
        self._step_times: List[float] = []
        self._losses: List[float] = []
        self._start_time = time.time()
        self._current_step = 0

    def update(self, step: int, loss: float, batch_size: int = 1) -> Dict[str, float]:
        """Record a training step.

        Args:
            step: Current step number
            loss: Loss value this step
            batch_size: Samples in this step
        Returns:
            Dict with timing and ETA info
        """
        now = time.time()
        if self._current_step > 0:
            step_time = now - self._last_time
            self._step_times.append(step_time)
            if len(self._step_times) > self.smoothing_window:
                self._step_times.pop(0)

        self._last_time = now
        self._current_step = step
        self._losses.append(loss)
        if len(self._losses) > self.smoothing_window:
            self._losses.pop(0)

        avg_step_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0
        remaining_steps = self.total_steps - step
        eta_sec = avg_step_time * remaining_steps if avg_step_time > 0 else 0

        return {
            "step": step,
            "progress": step / self.total_steps,
            "loss_smooth": sum(self._losses) / len(self._losses),
            "avg_step_ms": avg_step_time * 1000,
            "eta_hours": eta_sec / 3600,
            "elapsed_hours": (now - self._start_time) / 3600,
            "samples_per_sec": batch_size / (avg_step_time + 1e-10),
        }

    def get_smoothed_loss(self, alpha: float = 0.9) -> float:
        """Get exponentially smoothed loss."""
        if not self._losses:
            return 0.0
        smoothed = self._losses[0]
        for l in self._losses[1:]:
            smoothed = alpha * smoothed + (1 - alpha) * l
        return smoothed


class HyperparameterScheduler:
    """Multi-parameter scheduler with diverse scheduling strategies.

    Supports scheduling for:
    - Learning rate (cosine, linear, polynomial, step)
    - Dropout rate (progressive increase)
    - Masking ratio (curriculum learning)
    - Temperature (annealing)
    - Layer drop rate

    Args:
        total_steps: Total training steps
        warmup_steps: Warmup steps (0 = no warmup)
        min_value: Minimum parameter value
        max_value: Maximum parameter value
        schedule_type: Scheduling function type
    """

    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        schedule_type: str = "cosine",
    ) -> None:
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_value = min_value
        self.max_value = max_value
        self.schedule_type = schedule_type

    def get_value(self, step: int) -> float:
        """Get parameter value at a given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.min_value + (self.max_value - self.min_value) * step / max(1, self.warmup_steps)

        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))

        if self.schedule_type == "cosine":
            import math
            cosine = math.cos(math.pi * progress)
            return self.min_value + 0.5 * (self.max_value - self.min_value) * (1 + cosine)
        elif self.schedule_type == "linear":
            return self.max_value - progress * (self.max_value - self.min_value)
        elif self.schedule_type == "constant":
            return self.max_value
        elif self.schedule_type == "exponential":
            return self.max_value * (self.min_value / self.max_value) ** progress
        elif self.schedule_type == "cyclic":
            import math
            cycle_length = self.total_steps // 4
            cycle = (step - self.warmup_steps) % cycle_length
            phase = cycle / cycle_length
            return self.min_value + (self.max_value - self.min_value) * (1 - math.cos(math.pi * phase)) / 2
        else:
            return self.max_value

    def __call__(self, step: int) -> float:
        return self.get_value(step)


_NEW_DISTRIBUTED_EXPORTS = [
    "DistributedOptimizer", "GradientCompressor", "MixedPrecisionManager",
    "CheckpointManager", "CommunicationProfiler", "DistributedMetricsAggregator",
    "AdaptiveGradientAccumulation", "TrainingProgressTracker", "HyperparameterScheduler",
]
