"""
lumina/distributed_training.py

Full distributed training infrastructure for Lumina financial foundation model.

Covers:
  - DistributedDataParallel (DDP) wrapper with gradient compression
  - FullyShardedDataParallel (FSDP) with auto-wrapping and mixed sharding strategies
  - ZeRO-1/2/3 optimizer state sharding (via DeepSpeed-style logic on top of FSDP)
  - Gradient accumulation with correct loss normalization
  - Automatic Mixed Precision (AMP) with GradScaler and loss scaling
  - Gradient checkpointing at the TransformerBlock level
  - Activation checkpointing via torch.utils.checkpoint
  - Tensor / pipeline model parallelism scaffolding
  - Checkpoint saving with atomic rename (orbax-style) and resuming
  - Learning-rate scheduler helpers compatible with distributed state
  - DistributedSampler helpers
  - Profiler integration (torch.profiler)
  - Communication utilities: all_reduce, all_gather, broadcast
"""

from __future__ import annotations

import contextlib
import copy
import functools
import io
import json
import logging
import math
import os
import pathlib
import pickle
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        BackwardPrefetch,
        CPUOffload,
        StateDictType,
        FullStateDictConfig,
        LocalStateDictConfig,
        ShardedStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig
    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False
    logger.warning("FSDP not available in this PyTorch version.")

try:
    from torch.distributed.algorithms.ddp_comm_hooks import default as ddp_hooks
    _DDP_HOOKS_AVAILABLE = True
except ImportError:
    _DDP_HOOKS_AVAILABLE = False

try:
    import apex
    from apex import amp as apex_amp
    _APEX_AVAILABLE = True
except ImportError:
    _APEX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DistributedConfig:
    """Master configuration for distributed training."""
    # Backend
    backend: str = "nccl"               # "nccl" | "gloo" | "mpi"
    init_method: str = "env://"

    # Strategy
    strategy: str = "ddp"               # "ddp" | "fsdp" | "none"
    fsdp_sharding: str = "full_shard"   # "full_shard" | "shard_grad_op" | "no_shard" | "hybrid_shard"
    zero_stage: int = 3                 # 1 | 2 | 3 (maps to FSDP sharding strategy)

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"         # "float16" | "bfloat16"
    loss_scale: float = 2.0 ** 15
    loss_scale_window: int = 1000

    # Gradient accumulation
    grad_accum_steps: int = 1

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 500
    keep_last_n_checkpoints: int = 3
    resume_from: Optional[str] = None

    # Activation / gradient checkpointing
    gradient_checkpointing: bool = True
    activation_checkpointing: bool = False

    # CPU offload (FSDP)
    cpu_offload: bool = False

    # Prefetching (FSDP)
    backward_prefetch: bool = True

    # Profiling
    enable_profiler: bool = False
    profiler_schedule_wait: int = 1
    profiler_schedule_warmup: int = 1
    profiler_schedule_active: int = 5
    profiler_schedule_repeat: int = 0

    # Seeds
    seed: int = 42

    # Logging
    log_every_n_steps: int = 10


@dataclass
class TrainingState:
    """Tracks mutable state across training steps."""
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    total_tokens_seen: int = 0
    loss_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        return cls(**d)


# ---------------------------------------------------------------------------
# Process group utilities
# ---------------------------------------------------------------------------

class ProcessGroupManager:
    """Manages process groups for distributed training."""

    _instance: Optional["ProcessGroupManager"] = None

    def __init__(self, config: DistributedConfig):
        self.config = config
        self._rank: int = 0
        self._local_rank: int = 0
        self._world_size: int = 1
        self._initialized: bool = False

    @classmethod
    def get_instance(cls, config: Optional[DistributedConfig] = None) -> "ProcessGroupManager":
        if cls._instance is None:
            if config is None:
                config = DistributedConfig()
            cls._instance = cls(config)
        return cls._instance

    def initialize(self) -> None:
        if self._initialized:
            return
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self._rank = int(os.environ["RANK"])
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._world_size = int(os.environ["WORLD_SIZE"])
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
            )
            self._initialized = True
            logger.info(
                f"Initialized process group: rank={self._rank}, "
                f"local_rank={self._local_rank}, world_size={self._world_size}"
            )
        else:
            logger.info("No distributed env vars found; running in single-process mode.")

    def destroy(self) -> None:
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    @property
    def is_distributed(self) -> bool:
        return self._world_size > 1


def setup_distributed(config: DistributedConfig) -> ProcessGroupManager:
    """Convenience: create and initialize a ProcessGroupManager."""
    mgr = ProcessGroupManager(config)
    mgr.initialize()
    if torch.cuda.is_available():
        torch.cuda.set_device(mgr.local_rank)
    set_seed(config.seed + mgr.rank)
    return mgr


def cleanup_distributed(mgr: ProcessGroupManager) -> None:
    mgr.destroy()


# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolutions (can hurt performance)
    # torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Communication primitives
# ---------------------------------------------------------------------------

def all_reduce_mean(tensor: Tensor, world_size: int) -> Tensor:
    """In-place all-reduce averaging."""
    if not dist.is_initialized() or world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(world_size)
    return tensor


def all_reduce_sum(tensor: Tensor) -> Tensor:
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_tensor(tensor: Tensor) -> List[Tensor]:
    """Gather tensor from all ranks; return list of tensors."""
    if not dist.is_initialized():
        return [tensor]
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a Python object from rank src to all ranks."""
    if not dist.is_initialized():
        return obj
    objects = [obj]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# DDP Wrapper
# ---------------------------------------------------------------------------

class LuminaDDP(nn.Module):
    """
    Thin wrapper around DistributedDataParallel with:
      - FP16 gradient compression hook (PowerSGD or simple compression)
      - find_unused_parameters detection
      - gradient bucketing configuration
    """

    def __init__(
        self,
        module: nn.Module,
        device_ids: Optional[List[int]] = None,
        find_unused_parameters: bool = False,
        bucket_cap_mb: float = 25.0,
        gradient_as_bucket_view: bool = True,
        compress_gradients: bool = False,
        static_graph: bool = False,
    ):
        super().__init__()
        self.module = module
        if dist.is_initialized():
            self.ddp = DDP(
                module,
                device_ids=device_ids,
                find_unused_parameters=find_unused_parameters,
                bucket_cap_mb=bucket_cap_mb,
                gradient_as_bucket_view=gradient_as_bucket_view,
            )
            if static_graph:
                self.ddp._set_static_graph()
            if compress_gradients and _DDP_HOOKS_AVAILABLE:
                self.ddp.register_comm_hook(
                    state=None,
                    hook=ddp_hooks.fp16_compress_hook,
                )
        else:
            self.ddp = module

    def forward(self, *args, **kwargs):
        return self.ddp(*args, **kwargs)

    def no_sync(self):
        """Context manager to skip gradient sync (for gradient accumulation)."""
        if isinstance(self.ddp, DDP):
            return self.ddp.no_sync()
        return contextlib.nullcontext()

    @property
    def unwrapped(self) -> nn.Module:
        if isinstance(self.ddp, DDP):
            return self.ddp.module
        return self.ddp


# ---------------------------------------------------------------------------
# FSDP Wrapper
# ---------------------------------------------------------------------------

_SHARDING_STRATEGY_MAP = {
    "full_shard": "FULL_SHARD",
    "shard_grad_op": "SHARD_GRAD_OP",
    "no_shard": "NO_SHARD",
    "hybrid_shard": "HYBRID_SHARD",
    "_hybrid_shard_zero2": "_HYBRID_SHARD_ZERO2",
}


def _get_fsdp_sharding_strategy(name: str):
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP not available.")
    strategy_attr = _SHARDING_STRATEGY_MAP.get(name.lower(), "FULL_SHARD")
    return getattr(ShardingStrategy, strategy_attr)


def _get_fsdp_mixed_precision(dtype_str: str) -> "MixedPrecision":
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP not available.")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
        cast_forward_inputs=True,
    )


def wrap_model_fsdp(
    model: nn.Module,
    config: DistributedConfig,
    transformer_block_cls: Optional[Type[nn.Module]] = None,
) -> "FSDP":
    """
    Wrap model with FSDP using auto-wrap policy.

    Args:
        model: The PyTorch model to wrap.
        config: DistributedConfig.
        transformer_block_cls: If provided, use transformer_auto_wrap_policy
                               targeting this class.
    Returns:
        FSDP-wrapped model.
    """
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP requires PyTorch >= 1.12")

    sharding_strategy = _get_fsdp_sharding_strategy(config.fsdp_sharding)
    mixed_precision = _get_fsdp_mixed_precision(config.amp_dtype) if config.use_amp else None
    cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE if config.backward_prefetch else None

    if transformer_block_cls is not None:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_block_cls},
        )
    else:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1_000_000,
        )

    wrapped = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        sync_module_states=True,
        limit_all_gathers=True,
    )
    return wrapped


# ---------------------------------------------------------------------------
# Gradient checkpointing helpers
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(model: nn.Module, block_cls: Optional[Type[nn.Module]] = None) -> None:
    """
    Enable gradient checkpointing on all instances of block_cls inside model.
    If block_cls is None, try to detect TransformerBlock automatically.
    """
    count = 0
    for name, module in model.named_modules():
        # Heuristic: any module with 'Block' in its class name
        cls_name = type(module).__name__
        if block_cls is not None:
            match = isinstance(module, block_cls)
        else:
            match = "Block" in cls_name or "Layer" in cls_name
        if match:
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
            elif hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True
            count += 1
    logger.info(f"Enabled gradient checkpointing on {count} modules.")


def checkpointed_forward(module: nn.Module, *args, use_reentrant: bool = False, **kwargs):
    """
    Run module forward with activation checkpointing.
    Wraps torch.utils.checkpoint.checkpoint.
    """
    return gradient_checkpoint(module, *args, use_reentrant=use_reentrant, **kwargs)


class CheckpointedSequential(nn.Sequential):
    """Sequential module where each sub-module uses activation checkpointing."""

    def __init__(self, *modules: nn.Module, use_reentrant: bool = False):
        super().__init__(*modules)
        self.use_reentrant = use_reentrant

    def forward(self, x: Tensor) -> Tensor:
        for module in self:
            x = gradient_checkpoint(module, x, use_reentrant=self.use_reentrant)
        return x


# ---------------------------------------------------------------------------
# Mixed Precision context manager
# ---------------------------------------------------------------------------

class AMPContext:
    """
    Wraps torch.cuda.amp autocast + GradScaler.

    Usage:
        amp = AMPContext(config)
        with amp.autocast():
            loss = model(batch)
        amp.scaler.scale(loss).backward()
        amp.step(optimizer)
        amp.update()
    """

    def __init__(self, config: DistributedConfig):
        self.enabled = config.use_amp and torch.cuda.is_available()
        self.dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        # GradScaler is mainly used for float16; bfloat16 rarely needs it
        use_scaler = self.enabled and self.dtype == torch.float16
        self.scaler = GradScaler(
            init_scale=config.loss_scale,
            growth_interval=config.loss_scale_window,
            enabled=use_scaler,
        )

    @contextlib.contextmanager
    def autocast(self):
        with autocast(enabled=self.enabled, dtype=self.dtype):
            yield

    def scale(self, loss: Tensor) -> Tensor:
        return self.scaler.scale(loss)

    def step(self, optimizer: Optimizer) -> None:
        self.scaler.step(optimizer)

    def update(self) -> None:
        self.scaler.update()

    def unscale_(self, optimizer: Optimizer) -> None:
        self.scaler.unscale_(optimizer)

    @property
    def loss_scale(self) -> float:
        return self.scaler.get_scale()


# ---------------------------------------------------------------------------
# Gradient accumulation manager
# ---------------------------------------------------------------------------

class GradAccumManager:
    """
    Manages gradient accumulation steps with correct loss normalization.

    Handles:
      - DDP no_sync() to avoid premature all-reduce on accumulation steps
      - FSDP no_sync() equivalent
      - Loss division by accumulation steps
    """

    def __init__(
        self,
        model: nn.Module,
        accum_steps: int,
        amp: Optional[AMPContext] = None,
    ):
        self.model = model
        self.accum_steps = max(1, accum_steps)
        self.amp = amp
        self._step = 0

    @property
    def should_sync(self) -> bool:
        return (self._step + 1) % self.accum_steps == 0

    @contextlib.contextmanager
    def accumulate(self):
        """
        Context manager for one micro-batch. Skips gradient sync if not last
        accumulation step.
        """
        sync = self.should_sync
        if not sync and hasattr(self.model, "no_sync"):
            ctx = self.model.no_sync()
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            yield
        self._step += 1

    def normalize_loss(self, loss: Tensor) -> Tensor:
        """Divide loss by accumulation steps for correct gradient magnitude."""
        return loss / self.accum_steps

    def reset(self) -> None:
        self._step = 0


# ---------------------------------------------------------------------------
# Optimizer builder with ZeRO support
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
    use_fused: bool = True,
) -> Optimizer:
    """
    Build an optimizer, separating weight-decayed and non-decayed params.
    Weight decay is not applied to biases, norms, and 1D parameters.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    fused_available = (
        use_fused
        and optimizer_type == "adamw"
        and torch.cuda.is_available()
        and hasattr(torch.optim, "AdamW")
    )
    # Check fused kwarg support
    try:
        import inspect
        sig = inspect.signature(torch.optim.AdamW)
        fused_available = fused_available and "fused" in sig.parameters
    except Exception:
        fused_available = False

    extra_kwargs = {"fused": True} if fused_available else {}

    if optimizer_type == "adamw":
        opt = torch.optim.AdamW(
            param_groups, lr=lr, betas=(beta1, beta2), eps=eps, **extra_kwargs
        )
    elif optimizer_type == "adam":
        opt = torch.optim.Adam(
            param_groups, lr=lr, betas=(beta1, beta2), eps=eps
        )
    elif optimizer_type == "sgd":
        opt = torch.optim.SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    logger.info(
        f"Built {optimizer_type} optimizer: "
        f"{len(decay_params)} decay params, {len(no_decay_params)} no-decay params, "
        f"lr={lr}, wd={weight_decay}, fused={fused_available}"
    )
    return opt


# ---------------------------------------------------------------------------
# Learning rate schedulers
# ---------------------------------------------------------------------------

def cosine_with_warmup_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Linear warmup then cosine decay to min_lr_ratio * base_lr.
    Standard schedule for large language / foundation models.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def wsd_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Warmup-Stable-Decay (WSD) schedule used in MiniCPM and other LLMs.
    """
    total_steps = warmup_steps + stable_steps + decay_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            decay_progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class CyclicCosineScheduler:
    """
    Multi-cycle cosine annealing with restarts (SGDR-style).
    Each restart multiplies the cycle length by cycle_mult.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 2.0,
        max_lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_steps: int = 100,
        gamma: float = 1.0,
    ):
        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = -1
        self._step_count = 0

    def get_lr(self) -> List[float]:
        if self.step_in_cycle == -1:
            return [self.min_lr] * len(self.optimizer.param_groups)
        elif self.step_in_cycle < self.warmup_steps:
            frac = self.step_in_cycle / self.warmup_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * frac
            return [lr] * len(self.optimizer.param_groups)
        else:
            progress = (self.step_in_cycle - self.warmup_steps) / (
                self.cur_cycle_steps - self.warmup_steps
            )
            lr = self.min_lr + (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            ) / 2
            return [lr] * len(self.optimizer.param_groups)

    def step(self) -> None:
        self._step_count += 1
        self.step_in_cycle += 1
        if self.step_in_cycle >= self.cur_cycle_steps:
            self.cycle += 1
            self.step_in_cycle = 0
            self.cur_cycle_steps = int(
                (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                + self.warmup_steps
            )
            self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)

        lrs = self.get_lr()
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cur_cycle_steps": self.cur_cycle_steps,
            "cycle": self.cycle,
            "step_in_cycle": self.step_in_cycle,
            "_step_count": self._step_count,
            "max_lr": self.max_lr,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# Gradient norm utilities
# ---------------------------------------------------------------------------

def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
    amp: Optional[AMPContext] = None,
    optimizer: Optional[Optimizer] = None,
) -> float:
    """
    Clip gradient norm. If AMP is enabled, unscales first.
    Returns the pre-clip gradient norm.
    """
    if amp is not None and optimizer is not None:
        amp.unscale_(optimizer)
    if _FSDP_AVAILABLE and isinstance(model, FSDP):
        grad_norm = model.clip_grad_norm_(max_norm).item()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm
        ).item()
    return grad_norm


def compute_grad_norm(model: nn.Module) -> float:
    """Compute L2 norm of all gradients without modifying them."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    return math.sqrt(total_norm)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Atomic checkpoint saving and loading.

    Atomic writes: write to a temp file, then rename (POSIX atomic).
    Tracks last N checkpoints and deletes old ones.
    Supports FSDP sharded state dicts and regular state dicts.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, pathlib.Path],
        keep_last_n: int = 3,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.rank = rank
        self.world_size = world_size
        self._checkpoints: List[pathlib.Path] = []

        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt_path(self, step: int) -> pathlib.Path:
        return self.checkpoint_dir / f"checkpoint_step_{step:08d}"

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        state: TrainingState,
        amp: Optional[AMPContext] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> pathlib.Path:
        """
        Save checkpoint. For FSDP, uses FULL_STATE_DICT gathered on rank 0.
        Writes atomically via temp dir + rename.
        """
        ckpt_dir = self._ckpt_path(step)

        if _FSDP_AVAILABLE and isinstance(model, FSDP):
            self._save_fsdp(ckpt_dir, step, model, optimizer, scheduler, state, amp, extra)
        else:
            self._save_regular(ckpt_dir, step, model, optimizer, scheduler, state, amp, extra)

        barrier()
        if self.rank == 0:
            self._checkpoints.append(ckpt_dir)
            self._prune_old_checkpoints()
            logger.info(f"Saved checkpoint at step {step}: {ckpt_dir}")
        return ckpt_dir

    def _save_regular(
        self,
        ckpt_dir: pathlib.Path,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        state: TrainingState,
        amp: Optional[AMPContext],
        extra: Optional[Dict[str, Any]],
    ) -> None:
        if self.rank != 0:
            return

        tmp_dir = pathlib.Path(str(ckpt_dir) + ".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Unwrap DDP/FSDP to get raw state dict
        raw_model = model
        if isinstance(model, LuminaDDP):
            raw_model = model.unwrapped
        elif isinstance(model, DDP):
            raw_model = model.module

        payload: Dict[str, Any] = {
            "step": step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_state": state.to_dict(),
        }
        if scheduler is not None:
            if hasattr(scheduler, "state_dict"):
                payload["scheduler_state_dict"] = scheduler.state_dict()
        if amp is not None:
            payload["scaler_state_dict"] = amp.scaler.state_dict()
        if extra:
            payload["extra"] = extra

        torch.save(payload, tmp_dir / "checkpoint.pt")
        # Write metadata JSON
        meta = {
            "step": step,
            "world_size": self.world_size,
            "timestamp": time.time(),
        }
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Atomic rename
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        tmp_dir.rename(ckpt_dir)

    def _save_fsdp(
        self,
        ckpt_dir: pathlib.Path,
        step: int,
        model: "FSDP",
        optimizer: Optimizer,
        scheduler: Any,
        state: TrainingState,
        amp: Optional[AMPContext],
        extra: Optional[Dict[str, Any]],
    ) -> None:
        """Save FSDP model using FULL_STATE_DICT (gathered to rank 0)."""
        if not _FSDP_AVAILABLE:
            self._save_regular(ckpt_dir, step, model, optimizer, scheduler, state, amp, extra)
            return

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        opt_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg, opt_cfg):
            model_state = model.state_dict()
            opt_state = FSDP.optim_state_dict(model, optimizer)

        if self.rank == 0:
            tmp_dir = pathlib.Path(str(ckpt_dir) + ".tmp")
            tmp_dir.mkdir(parents=True, exist_ok=True)

            payload = {
                "step": step,
                "model_state_dict": model_state,
                "optimizer_state_dict": opt_state,
                "training_state": state.to_dict(),
            }
            if scheduler is not None and hasattr(scheduler, "state_dict"):
                payload["scheduler_state_dict"] = scheduler.state_dict()
            if amp is not None:
                payload["scaler_state_dict"] = amp.scaler.state_dict()
            if extra:
                payload["extra"] = extra

            torch.save(payload, tmp_dir / "checkpoint.pt")
            meta = {"step": step, "world_size": self.world_size, "fsdp": True, "timestamp": time.time()}
            (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
            tmp_dir.rename(ckpt_dir)

    def load(
        self,
        ckpt_path: Union[str, pathlib.Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        amp: Optional[AMPContext] = None,
        strict: bool = True,
        map_location: Optional[str] = None,
    ) -> TrainingState:
        """Load checkpoint. Returns TrainingState."""
        ckpt_path = pathlib.Path(ckpt_path)
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_path}")

        if map_location is None:
            map_location = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

        payload = torch.load(ckpt_path / "checkpoint.pt", map_location=map_location)

        # Load model weights
        raw_model = model
        if isinstance(model, LuminaDDP):
            raw_model = model.unwrapped
        elif isinstance(model, DDP):
            raw_model = model.module
        elif _FSDP_AVAILABLE and isinstance(model, FSDP):
            raw_model = model

        if _FSDP_AVAILABLE and isinstance(raw_model, FSDP):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(raw_model, StateDictType.FULL_STATE_DICT, cfg):
                raw_model.load_state_dict(payload["model_state_dict"], strict=strict)
        else:
            raw_model.load_state_dict(payload["model_state_dict"], strict=strict)

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(payload["scheduler_state_dict"])
        if amp is not None and "scaler_state_dict" in payload:
            amp.scaler.load_state_dict(payload["scaler_state_dict"])

        state_dict = payload.get("training_state", {})
        state = TrainingState.from_dict(state_dict) if state_dict else TrainingState()
        logger.info(f"Loaded checkpoint from {ckpt_path}, step={state.step}")
        return state

    def _prune_old_checkpoints(self) -> None:
        """Delete oldest checkpoints beyond keep_last_n."""
        while len(self._checkpoints) > self.keep_last_n:
            old = self._checkpoints.pop(0)
            if old.exists():
                shutil.rmtree(old)
                logger.info(f"Deleted old checkpoint: {old}")

    def list_checkpoints(self) -> List[pathlib.Path]:
        """List all checkpoint directories sorted by step."""
        ckpts = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*"),
            key=lambda p: int(p.name.split("_")[-1]),
        )
        return ckpts

    def latest_checkpoint(self) -> Optional[pathlib.Path]:
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Model parallelism (tensor parallelism scaffolding)
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-wise tensor parallelism.
    Each rank holds a shard of output features.

    In a world_size=N setup, each rank holds out_features/N output columns.
    Forward: local matmul on full input.
    Backward: gradients are local; no communication needed on forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        assert out_features % world_size == 0, (
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.local_out = out_features // world_size

        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.bias_ = nn.Parameter(torch.zeros(self.local_out)) if bias else None
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_ is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias_)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row-wise tensor parallelism.
    Each rank holds a shard of input features.

    Forward: local matmul on local input shard, then all-reduce sum.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.local_in = in_features // world_size

        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        self.bias_ = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_ is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight)
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        if self.bias_ is not None:
            out = out + self.bias_
        return out


class TensorParallelAttention(nn.Module):
    """
    Multi-head attention with tensor parallelism across the head dimension.
    Each rank handles num_heads/world_size heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        world_size: int = 1,
        rank: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_heads % world_size == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        self.local_heads = num_heads // world_size
        self.head_dim = d_model // num_heads

        self.q_proj = ColumnParallelLinear(d_model, d_model, world_size=world_size, rank=rank)
        self.k_proj = ColumnParallelLinear(d_model, d_model, world_size=world_size, rank=rank)
        self.v_proj = ColumnParallelLinear(d_model, d_model, world_size=world_size, rank=rank)
        self.out_proj = RowParallelLinear(d_model, d_model, world_size=world_size, rank=rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.local_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        return out


# ---------------------------------------------------------------------------
# Profiler integration
# ---------------------------------------------------------------------------

class TrainingProfiler:
    """
    Wraps torch.profiler.profile for training loops.
    Activates for a few steps to collect traces without overhead.
    """

    def __init__(self, config: DistributedConfig, output_dir: str = "profiler_traces"):
        self.config = config
        self.enabled = config.enable_profiler
        self.output_dir = pathlib.Path(output_dir)
        self._profiler: Optional[torch.profiler.profile] = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.output_dir.mkdir(parents=True, exist_ok=True)
        schedule = torch.profiler.schedule(
            wait=self.config.profiler_schedule_wait,
            warmup=self.config.profiler_schedule_warmup,
            active=self.config.profiler_schedule_active,
            repeat=self.config.profiler_schedule_repeat,
        )
        self._profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self._profiler.__enter__()
        return self

    def __exit__(self, *args):
        if self._profiler is not None:
            self._profiler.__exit__(*args)

    def step(self) -> None:
        if self._profiler is not None:
            self._profiler.step()

    def key_averages(self):
        if self._profiler is not None:
            return self._profiler.key_averages()
        return []


# ---------------------------------------------------------------------------
# Distributed sampler utilities
# ---------------------------------------------------------------------------

class InfiniteDistributedSampler:
    """
    Infinite iterator over a dataset for distributed training.
    Reshuffles at the start of each epoch, ensures each rank sees distinct samples.
    """

    def __init__(
        self,
        dataset_size: int,
        rank: int,
        world_size: int,
        seed: int = 42,
    ):
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[int]:
        while True:
            rng = np.random.RandomState(self.seed + self._epoch)
            indices = rng.permutation(self.dataset_size).tolist()
            # Pad to be divisible by world_size
            pad = (self.world_size - len(indices) % self.world_size) % self.world_size
            indices = indices + indices[:pad]
            # Take rank's slice
            rank_indices = indices[self.rank::self.world_size]
            for idx in rank_indices:
                yield idx
            self._epoch += 1


def build_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    epoch: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """Build a DataLoader with DistributedSampler for multi-process training."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=drop_last,
    )
    sampler.set_epoch(epoch)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

class DistributedTrainer:
    """
    Full-featured distributed trainer that ties everything together.

    Supports:
      - DDP and FSDP strategies
      - Gradient accumulation
      - AMP
      - Gradient clipping
      - Checkpoint save/load
      - LR scheduling
      - Profiling
      - Logging with metrics averaging across ranks
    """

    def __init__(
        self,
        config: DistributedConfig,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.pgm = setup_distributed(config)

        self.device = (
            torch.device(f"cuda:{self.pgm.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Wrap model
        if config.strategy == "fsdp" and _FSDP_AVAILABLE and self.pgm.is_distributed:
            model = model.to(self.device)
            self.model = wrap_model_fsdp(model, config)
        elif config.strategy == "ddp" and self.pgm.is_distributed:
            model = model.to(self.device)
            self.model = LuminaDDP(model, device_ids=[self.pgm.local_rank])
        else:
            self.model = model.to(self.device)

        if config.gradient_checkpointing:
            enable_gradient_checkpointing(
                self.model.unwrapped if isinstance(self.model, LuminaDDP) else self.model
            )

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.amp = AMPContext(config)
        self.accum = GradAccumManager(self.model, config.grad_accum_steps, self.amp)
        self.ckpt_mgr = CheckpointManager(
            config.checkpoint_dir,
            keep_last_n=config.keep_last_n_checkpoints,
            rank=self.pgm.rank,
            world_size=self.pgm.world_size,
        )
        self.state = TrainingState()
        self.profiler = TrainingProfiler(config)

        # Resume if requested
        if config.resume_from:
            self.state = self.ckpt_mgr.load(
                config.resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.amp,
            )

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Run a single gradient accumulation step.
        Returns metrics dict.
        """
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, Tensor) else v
                 for k, v in batch.items()}

        with self.accum.accumulate():
            with self.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss = self.accum.normalize_loss(loss)

            scaled_loss = self.amp.scale(loss)
            scaled_loss.backward()

        metrics: Dict[str, float] = {}

        if self.accum.should_sync:
            grad_norm = clip_grad_norm(self.model, self.config.max_grad_norm, self.amp, self.optimizer)
            self.amp.step(self.optimizer)
            self.amp.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    self.scheduler.step()

            self.state.step += 1
            metrics["loss"] = loss.item() * self.accum.accum_steps
            metrics["grad_norm"] = grad_norm
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            metrics["loss_scale"] = self.amp.loss_scale

            self.state.loss_history.append(metrics["loss"])
            self.state.lr_history.append(metrics["lr"])
            self.state.grad_norm_history.append(grad_norm)

        return metrics

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.eval()
        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, Tensor) else v
                 for k, v in batch.items()}
        with self.amp.autocast():
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return {"val_loss": loss.item()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one full training epoch."""
        if self.train_loader is None:
            raise ValueError("train_loader not set.")

        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        n_steps = 0
        self.accum.reset()

        with self.profiler:
            for batch in self.train_loader:
                metrics = self.train_step(batch)
                if metrics:
                    total_loss += metrics.get("loss", 0.0)
                    n_steps += 1

                    if self.pgm.is_main_process and self.state.step % self.config.log_every_n_steps == 0:
                        logger.info(
                            f"Step {self.state.step} | loss={metrics.get('loss', 0):.4f} | "
                            f"lr={metrics.get('lr', 0):.2e} | grad_norm={metrics.get('grad_norm', 0):.3f}"
                        )

                    if self.state.step % self.config.save_every_n_steps == 0:
                        self.ckpt_mgr.save(
                            self.state.step,
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.state,
                            self.amp,
                        )

                self.profiler.step()

        avg_loss = total_loss / max(1, n_steps)
        # Sync across ranks
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        all_reduce_mean(loss_tensor, self.pgm.world_size)
        self.state.epoch = epoch
        return {"train_loss": loss_tensor.item(), "epoch": epoch}

    def evaluate(self) -> Dict[str, float]:
        """Run full validation loop."""
        if self.val_loader is None:
            raise ValueError("val_loader not set.")
        total_loss = 0.0
        n_steps = 0
        for batch in self.val_loader:
            m = self.eval_step(batch)
            total_loss += m["val_loss"]
            n_steps += 1

        avg_loss = total_loss / max(1, n_steps)
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        all_reduce_mean(loss_tensor, self.pgm.world_size)
        val_loss = loss_tensor.item()

        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            logger.info(f"New best val loss: {val_loss:.4f}")
        return {"val_loss": val_loss}

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        """Main training loop over multiple epochs."""
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(epoch)
            history["train_loss"].append(train_metrics["train_loss"])
            if self.val_loader is not None:
                val_metrics = self.evaluate()
                history["val_loss"].append(val_metrics["val_loss"])
                if self.pgm.is_main_process:
                    logger.info(
                        f"Epoch {epoch} | train_loss={train_metrics['train_loss']:.4f} | "
                        f"val_loss={val_metrics['val_loss']:.4f}"
                    )
            else:
                if self.pgm.is_main_process:
                    logger.info(f"Epoch {epoch} | train_loss={train_metrics['train_loss']:.4f}")
        # Final checkpoint
        self.ckpt_mgr.save(
            self.state.step, self.model, self.optimizer, self.scheduler, self.state, self.amp
        )
        return history

    def teardown(self) -> None:
        cleanup_distributed(self.pgm)


# ---------------------------------------------------------------------------
# ZeRO-stage emulation helpers
# ---------------------------------------------------------------------------

class ZeROGradientSharder:
    """
    Emulates ZeRO-1 (optimizer state sharding) and ZeRO-2 (+ gradient sharding)
    behavior on top of standard PyTorch via manual all-reduce of gradients
    and optimizer state partitioning.

    Note: In practice, FSDP full_shard IS ZeRO-3.
    This class demonstrates the concepts for educational clarity and for
    environments where FSDP is not available.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        parameters: List[nn.Parameter],
        rank: int,
        world_size: int,
        zero_stage: int = 1,
    ):
        self.optimizer = optimizer
        self.parameters = [p for p in parameters if p.requires_grad]
        self.rank = rank
        self.world_size = world_size
        self.zero_stage = zero_stage
        self._param_to_rank: Dict[int, int] = {}
        self._assign_params_to_ranks()

    def _assign_params_to_ranks(self) -> None:
        """Round-robin assignment of parameters to ranks for state sharding."""
        for i, p in enumerate(self.parameters):
            self._param_to_rank[id(p)] = i % self.world_size

    def reduce_gradients(self) -> None:
        """
        ZeRO-2: reduce gradients, keeping only local shard.
        ZeRO-1: full all-reduce (optimizer states are sharded at step time).
        """
        if not dist.is_initialized():
            return
        for p in self.parameters:
            if p.grad is None:
                continue
            if self.zero_stage >= 2:
                # Reduce-scatter: only rank that owns this param keeps full gradient
                owner = self._param_to_rank[id(p)]
                dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.SUM)
                if self.rank != owner:
                    p.grad = None
                else:
                    p.grad.div_(self.world_size)
            else:
                # ZeRO-1: all-reduce gradients fully
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(self.world_size)

    def step(self) -> None:
        """
        Optimizer step. For ZeRO-1/2, each rank only steps its owned params.
        Then broadcast updated params to all ranks.
        """
        if self.zero_stage >= 1:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if id(p) not in self._param_to_rank:
                        continue
                    owner = self._param_to_rank[id(p)]
                    if self.rank == owner:
                        # Step this parameter
                        pass  # optimizer.step() will handle it
            self.optimizer.step()
            # Broadcast updated params
            for p in self.parameters:
                owner = self._param_to_rank[id(p)]
                if dist.is_initialized():
                    dist.broadcast(p.data, src=owner)
        else:
            self.optimizer.step()


# ---------------------------------------------------------------------------
# Activation offloading (CPU offload for activations)
# ---------------------------------------------------------------------------

class ActivationOffloadHook:
    """
    Registers forward hooks on specified modules to offload activations
    to CPU during forward pass and reload during backward.
    This trades compute (H2D copy) for memory (GPU VRAM).
    """

    def __init__(self, modules: List[nn.Module]):
        self._stash: Dict[int, List[Tensor]] = {}
        self._hooks: List[Any] = []
        for mod in modules:
            h_fwd = mod.register_forward_hook(self._forward_hook)
            h_bwd = mod.register_full_backward_hook(self._backward_hook)
            self._hooks.extend([h_fwd, h_bwd])

    def _forward_hook(self, module: nn.Module, inputs: Tuple, output: Tensor) -> Tensor:
        mid = id(module)
        cpu_out = output.detach().cpu()
        self._stash[mid] = cpu_out
        return output

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[Tensor], ...],
        grad_output: Tuple[Optional[Tensor], ...],
    ) -> None:
        mid = id(module)
        if mid in self._stash:
            del self._stash[mid]

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Gradient accumulation with micro-batching helper
# ---------------------------------------------------------------------------

class MicroBatchAccumulator:
    """
    Splits a large batch into micro-batches and accumulates gradients,
    handling DDP no_sync correctly.

    Usage:
        accum = MicroBatchAccumulator(model, optimizer, amp, accum_steps=4)
        total_loss = accum.forward_backward_accumulate(batch, loss_fn)
        accum.optimizer_step(scheduler)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        amp: AMPContext,
        accum_steps: int = 4,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.amp = amp
        self.accum_steps = accum_steps
        self.max_grad_norm = max_grad_norm

    def split_batch(self, batch: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
        """Split batch dict along batch dimension into accum_steps micro-batches."""
        total = next(iter(batch.values())).shape[0]
        micro_size = total // self.accum_steps
        micros = []
        for i in range(self.accum_steps):
            start = i * micro_size
            end = start + micro_size if i < self.accum_steps - 1 else total
            micros.append({k: v[start:end] for k, v in batch.items()})
        return micros

    def forward_backward_accumulate(
        self,
        batch: Dict[str, Tensor],
        loss_fn: Optional[Callable] = None,
    ) -> float:
        """
        Run forward+backward for all micro-batches.
        Returns mean loss across micro-batches.
        """
        micros = self.split_batch(batch)
        total_loss = 0.0

        self.optimizer.zero_grad(set_to_none=True)

        for i, micro in enumerate(micros):
            is_last = (i == self.accum_steps - 1)
            sync_ctx = (
                contextlib.nullcontext()
                if is_last
                else (self.model.no_sync() if hasattr(self.model, "no_sync") else contextlib.nullcontext())
            )
            with sync_ctx:
                with self.amp.autocast():
                    if loss_fn is not None:
                        outputs = self.model(**micro)
                        loss = loss_fn(outputs)
                    else:
                        outputs = self.model(**micro)
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    loss = loss / self.accum_steps
                self.amp.scale(loss).backward()
            total_loss += loss.item()

        return total_loss * self.accum_steps

    def optimizer_step(self, scheduler: Optional[Any] = None) -> float:
        grad_norm = clip_grad_norm(self.model, self.max_grad_norm, self.amp, self.optimizer)
        self.amp.step(self.optimizer)
        self.amp.update()
        if scheduler is not None and hasattr(scheduler, "step"):
            scheduler.step()
        return grad_norm


# ---------------------------------------------------------------------------
# Training metrics aggregator
# ---------------------------------------------------------------------------

class MetricsAggregator:
    """
    Thread/process-safe metric aggregation across distributed ranks.
    Supports mean, sum, max aggregations.
    """

    def __init__(self, world_size: int = 1, device: Optional[torch.device] = None):
        self.world_size = world_size
        self.device = device or torch.device("cpu")
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def compute(self, sync: bool = True) -> Dict[str, float]:
        result = {k: self._sums[k] / max(1, self._counts[k]) for k in self._sums}
        if sync and dist.is_initialized():
            for k in result:
                t = torch.tensor(result[k], device=self.device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                result[k] = t.item() / self.world_size
        return result

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# LAMB optimizer (Layer-wise Adaptive Moments for Batch training)
# Useful for very large batch distributed training
# ---------------------------------------------------------------------------

class LAMB(Optimizer):
    """
    LAMB optimizer as described in:
    "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
    Yang You et al., 2019.

    Combines Adam with a per-layer trust ratio for large-batch stability.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        clamp_value: float = 10.0,
        adam: bool = False,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.clamp_value = clamp_value
        self.adam = adam  # If True, disable trust ratio (becomes Adam)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                step = state["step"]
                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step

                exp_avg_corr = exp_avg / bias1
                exp_avg_sq_corr = exp_avg_sq / bias2

                adam_update = exp_avg_corr / (exp_avg_sq_corr.sqrt() + group["eps"])
                adam_update.add_(p, alpha=group["weight_decay"])

                if self.adam:
                    trust_ratio = 1.0
                else:
                    w_norm = p.norm(2.0)
                    g_norm = adam_update.norm(2.0)
                    if w_norm == 0 or g_norm == 0:
                        trust_ratio = 1.0
                    else:
                        trust_ratio = (w_norm / g_norm).clamp(0, self.clamp_value).item()

                p.add_(adam_update, alpha=-group["lr"] * trust_ratio)

        return loss


# ---------------------------------------------------------------------------
# Utilities for model state inspection
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_memory_estimate(model: nn.Module, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """Estimate GPU memory usage for model parameters."""
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)

    total_params = count_parameters(model, trainable_only=False)
    param_mb = total_params * bytes_per_param / 1024 ** 2
    # Adam: 2 extra fp32 tensors per param = 8 bytes/param
    optimizer_mb = total_params * 8 / 1024 ** 2
    # Gradients: same as params
    grad_mb = param_mb

    return {
        "param_mb": param_mb,
        "optimizer_mb": optimizer_mb,
        "grad_mb": grad_mb,
        "total_mb": param_mb + optimizer_mb + grad_mb,
        "total_gb": (param_mb + optimizer_mb + grad_mb) / 1024,
    }


def print_model_summary(model: nn.Module, input_shape: Optional[Tuple] = None) -> None:
    """Print a summary of model architecture and parameter counts."""
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    frozen = total - trainable

    print(f"\n{'='*60}")
    print(f"Model: {type(model).__name__}")
    print(f"{'='*60}")
    print(f"Total parameters:     {total:>15,}")
    print(f"Trainable parameters: {trainable:>15,}")
    print(f"Frozen parameters:    {frozen:>15,}")
    mem = model_memory_estimate(model)
    print(f"Param memory (fp32):  {mem['param_mb']:>12.1f} MB")
    print(f"Total w/ opt+grad:    {mem['total_gb']:>12.2f} GB")
    print(f"{'='*60}")

    # Per-module summary
    print(f"\n{'Module':<50} {'Params':>12} {'Trainable':>12}")
    print("-" * 76)
    for name, module in model.named_children():
        p = sum(x.numel() for x in module.parameters())
        t = sum(x.numel() for x in module.parameters() if x.requires_grad)
        print(f"{name:<50} {p:>12,} {t:>12,}")
    print("-" * 76)


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA) of model weights
# ---------------------------------------------------------------------------

class ModelEMA:
    """
    Maintains an exponential moving average of model parameters.
    Useful for stable evaluation during training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[str] = None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.module.to(device=device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_params = dict(model.named_parameters())
        ema_params = dict(self.module.named_parameters())
        for name, ema_p in ema_params.items():
            if name in model_params:
                model_p = model_params[name].to(ema_p.device)
                ema_p.copy_(ema_p * self.decay + model_p * (1.0 - self.decay))

        # Also update buffers (running stats, etc.)
        model_bufs = dict(model.named_buffers())
        for name, buf in self.module.named_buffers():
            if name in model_bufs:
                buf.copy_(model_bufs[name].to(buf.device))

    def state_dict(self) -> Dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.module.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# torchrun launcher helper
# ---------------------------------------------------------------------------

def launch_distributed(
    fn: Callable,
    nproc_per_node: int,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    **kwargs,
) -> None:
    """
    Programmatically launch multi-process distributed training.
    Sets up environment variables and spawns processes.
    """
    import torch.multiprocessing as mp

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(nproc_per_node * nnodes)

    def worker(rank: int):
        os.environ["RANK"] = str(node_rank * nproc_per_node + rank)
        os.environ["LOCAL_RANK"] = str(rank)
        fn(**kwargs)

    mp.spawn(worker, nprocs=nproc_per_node, join=True)


# ---------------------------------------------------------------------------
# Straggler detection (for large cluster training)
# ---------------------------------------------------------------------------

class StragglerDetector:
    """
    Detects straggler processes in distributed training by comparing
    step times across ranks via all-reduce of timing tensors.
    """

    def __init__(self, threshold_factor: float = 2.0):
        self.threshold_factor = threshold_factor
        self._step_times: List[float] = []

    @contextlib.contextmanager
    def time_step(self, device: torch.device):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self._step_times.append(elapsed)
        self._check_stragglers(elapsed, device)

    def _check_stragglers(self, my_time: float, device: torch.device) -> None:
        if not dist.is_initialized():
            return
        t = torch.tensor(my_time, device=device)
        gathered = all_gather_tensor(t)
        times = [g.item() for g in gathered]
        mean_t = sum(times) / len(times)
        max_t = max(times)
        if max_t > self.threshold_factor * mean_t:
            straggler_rank = times.index(max_t)
            if dist.get_rank() == 0:
                logger.warning(
                    f"Straggler detected: rank {straggler_rank} took {max_t:.2f}s "
                    f"vs mean {mean_t:.2f}s"
                )


# ---------------------------------------------------------------------------
# Pipeline parallelism scaffold
# ---------------------------------------------------------------------------

class PipelineStage(nn.Module):
    """
    Represents one stage in a pipeline-parallel model.
    Handles micro-batch pipelining (GPipe-style).
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
        device: torch.device,
    ):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.device = device
        self.is_first = (stage_id == 0)
        self.is_last = (stage_id == num_stages - 1)
        self.layers.to(device)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GPipePipeline(nn.Module):
    """
    Naive GPipe-style pipeline parallelism.
    Splits model into stages, splits batch into micro-batches,
    runs forward pass stage by stage with micro-batch pipelining.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        chunks: int = 4,
    ):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.chunks = chunks

    def forward(self, x: Tensor) -> Tensor:
        """Run GPipe forward pass."""
        micro_batches = x.chunk(self.chunks, dim=0)
        outputs = []
        for micro in micro_batches:
            for stage in self.stages:
                micro = micro.to(stage.device)
                micro = stage(micro)
            outputs.append(micro)
        return torch.cat(outputs, dim=0)


def partition_model_for_pipeline(
    model: nn.Module,
    num_stages: int,
    devices: Optional[List[torch.device]] = None,
) -> List[PipelineStage]:
    """
    Evenly partition a sequential model into pipeline stages.
    """
    if devices is None:
        devices = [torch.device(f"cuda:{i}") for i in range(num_stages)]

    all_layers = list(model.children())
    layers_per_stage = max(1, len(all_layers) // num_stages)

    stages = []
    for i in range(num_stages):
        start = i * layers_per_stage
        end = start + layers_per_stage if i < num_stages - 1 else len(all_layers)
        stage_layers = nn.ModuleList(all_layers[start:end])
        stage = PipelineStage(stage_layers, i, num_stages, devices[i % len(devices)])
        stages.append(stage)

    return stages


# ---------------------------------------------------------------------------
# Logging utility
# ---------------------------------------------------------------------------

def setup_logging(rank: int = 0, level: int = logging.INFO) -> None:
    """Configure logging; only rank 0 prints at INFO level."""
    fmt = f"[rank={rank}] %(asctime)s | %(name)s | %(levelname)s | %(message)s"
    logging.basicConfig(
        level=level if rank == 0 else logging.WARNING,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Module-level convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "DistributedConfig",
    "TrainingState",
    # Setup
    "ProcessGroupManager",
    "setup_distributed",
    "cleanup_distributed",
    "set_seed",
    # Communication
    "all_reduce_mean",
    "all_reduce_sum",
    "all_gather_tensor",
    "broadcast_object",
    "barrier",
    # Wrappers
    "LuminaDDP",
    "wrap_model_fsdp",
    # AMP
    "AMPContext",
    # Gradient accumulation
    "GradAccumManager",
    "MicroBatchAccumulator",
    # Checkpointing
    "enable_gradient_checkpointing",
    "CheckpointedSequential",
    "CheckpointManager",
    # Optimizers / schedulers
    "build_optimizer",
    "cosine_with_warmup_schedule",
    "wsd_schedule",
    "get_linear_schedule_with_warmup",
    "CyclicCosineScheduler",
    "LAMB",
    # Grad utilities
    "clip_grad_norm",
    "compute_grad_norm",
    # Data
    "build_distributed_dataloader",
    "InfiniteDistributedSampler",
    # Tensor parallelism
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelAttention",
    # Pipeline parallelism
    "PipelineStage",
    "GPipePipeline",
    "partition_model_for_pipeline",
    # ZeRO
    "ZeROGradientSharder",
    # EMA
    "ModelEMA",
    # Profiling
    "TrainingProfiler",
    "StragglerDetector",
    # Metrics
    "MetricsAggregator",
    # Utilities
    "count_parameters",
    "model_memory_estimate",
    "print_model_summary",
    "setup_logging",
    # Main trainer
    "DistributedTrainer",
]
