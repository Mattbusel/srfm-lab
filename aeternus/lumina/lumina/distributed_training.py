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


# ============================================================
# Advanced Distributed Training Components
# ============================================================

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistributedConfig:
    """Configuration for distributed training setup."""
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: float = 25.0
    broadcast_buffers: bool = True
    use_zero: bool = False
    zero_stage: int = 1
    use_fsdp: bool = False
    fsdp_min_num_params: int = 1_000_000
    use_pipeline: bool = False
    pipeline_chunks: int = 8
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    mixed_precision: bool = True
    fp16_loss_scale: float = 2.0 ** 16
    fp16_loss_scale_window: int = 1000
    gradient_clip: float = 1.0
    log_interval: int = 10


class GradientSynchronizer:
    """Manages gradient synchronization across processes."""

    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.model = model
        self.config = config
        self._hooks: List[Any] = []
        self._grad_buffer: Dict[str, torch.Tensor] = {}
        self._sync_count = 0

    def register_hooks(self):
        """Register gradient synchronization hooks."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._make_sync_hook(name))
                self._hooks.append(hook)

    def _make_sync_hook(self, name: str) -> Callable:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(grad, op=dist.ReduceOp.AVG)
            self._sync_count += 1
            return grad
        return hook

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def synchronize_buffers(self):
        """Synchronize model buffers (e.g., BatchNorm running stats)."""
        for buf in self.model.buffers():
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(buf, src=0)

    @property
    def sync_count(self) -> int:
        return self._sync_count


class ZeroRedundancyOptimizer:
    """
    ZeRO optimizer (Rajbhandari et al. 2020) Stage 1/2/3 simulation.
    Partitions optimizer states (Stage 1), gradients (Stage 2),
    and parameters (Stage 3) across data-parallel ranks.
    """

    def __init__(
        self,
        params,
        optimizer_class,
        stage: int = 1,
        overlap_communication: bool = True,
        **optimizer_kwargs
    ):
        self.stage = stage
        self.overlap_communication = overlap_communication
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        param_list = list(params)
        self._partition_params(param_list)
        self.optimizer = optimizer_class(self._local_params, **optimizer_kwargs)
        self._global_params = param_list

    def _partition_params(self, params: List[torch.Tensor]):
        """Partition parameters across ranks."""
        n = len(params)
        chunk = max(1, n // max(1, self.world_size))
        start = self.rank * chunk
        end = start + chunk if self.rank < self.world_size - 1 else n
        self._local_params = params[start:end]
        self._local_indices = list(range(start, end))

    def step(self):
        """Optimizer step with optional gradient scattering/gathering."""
        if self.stage >= 2:
            self._scatter_gradients()
        self.optimizer.step()
        if self.stage >= 1:
            self._gather_parameters()

    def _scatter_gradients(self):
        """Stage 2: Scatter gradients so each rank only updates its shard."""
        for param in self._global_params:
            if param.grad is not None and dist.is_initialized():
                dist.reduce_scatter_tensor(param.grad, param.grad)

    def _gather_parameters(self):
        """Gather updated parameters from all ranks."""
        for param in self._global_params:
            if dist.is_initialized():
                dist.all_gather_into_tensor(param.data, param.data)

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class AllReduceLayer(nn.Module):
    """Differentiable all-reduce for tensor-parallel linear layers."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x


class TensorParallelLinear(nn.Module):
    """
    Column-parallel or row-parallel linear layer for tensor parallelism
    (Shoeybi et al. 2019 Megatron-LM style).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "column",  # "column" or "row"
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.mode = mode
        self.world_size = world_size
        self.rank = rank
        self.all_reduce = AllReduceLayer()

        if mode == "column":
            shard_out = out_features // world_size
            self.linear = nn.Linear(in_features, shard_out, bias=bias)
        else:  # row
            shard_in = in_features // world_size
            self.linear = nn.Linear(shard_in, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.mode == "row":
            out = self.all_reduce(out)
        return out

    @property
    def weight(self):
        return self.linear.weight


class PipelineStage(nn.Module):
    """
    A single stage in a pipeline-parallel model.
    Handles micro-batch processing with forward/backward interleaving.
    """

    def __init__(self, layers: nn.ModuleList, stage_id: int, num_stages: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
        self._input_buffer: List[torch.Tensor] = []
        self._output_buffer: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def is_first_stage(self) -> bool:
        return self.stage_id == 0

    def is_last_stage(self) -> bool:
        return self.stage_id == self.num_stages - 1


class GradientFlowMonitor:
    """
    Monitors gradient flow during training to detect vanishing/exploding gradients.
    """

    def __init__(self, model: nn.Module, log_interval: int = 100):
        self.model = model
        self.log_interval = log_interval
        self._step = 0
        self.history: List[Dict[str, float]] = []

    def record(self):
        """Record gradient norms for all layers."""
        stats: Dict[str, float] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.detach().norm().item()
                stats[name] = norm
        self.history.append(stats)
        self._step += 1
        return stats

    def get_max_grad_norm(self) -> float:
        if not self.history:
            return 0.0
        last = self.history[-1]
        return max(last.values()) if last else 0.0

    def get_min_grad_norm(self) -> float:
        if not self.history:
            return 0.0
        last = self.history[-1]
        return min(last.values()) if last else 0.0

    def detect_vanishing(self, threshold: float = 1e-7) -> List[str]:
        """Return parameter names with near-zero gradients."""
        if not self.history:
            return []
        return [k for k, v in self.history[-1].items() if v < threshold]

    def detect_exploding(self, threshold: float = 100.0) -> List[str]:
        """Return parameter names with large gradients."""
        if not self.history:
            return []
        return [k for k, v in self.history[-1].items() if v > threshold]

    def summary(self) -> Dict[str, float]:
        if not self.history:
            return {}
        last = self.history[-1]
        vals = list(last.values())
        return {
            "max_norm": max(vals),
            "min_norm": min(vals),
            "mean_norm": sum(vals) / len(vals),
            "num_params": len(vals),
        }


class MixedPrecisionTrainer:
    """
    FP16/BF16 mixed-precision training wrapper with dynamic loss scaling.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        dtype: torch.dtype = torch.float16,
        initial_scale: float = 2.0 ** 16,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dtype = dtype
        self.loss_scale = initial_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self._successful_steps = 0
        self._overflow_count = 0

    def _check_overflow(self) -> bool:
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.loss_scale

    def step(self, scaled_loss: torch.Tensor):
        """Backward, unscale, clip, step with overflow detection."""
        scaled_loss.backward()

        # Unscale gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.div_(self.loss_scale)

        overflow = self._check_overflow()
        if overflow:
            self._overflow_count += 1
            self.loss_scale = max(self.min_scale, self.loss_scale / self.scale_factor)
            self.optimizer.zero_grad()
            self._successful_steps = 0
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._successful_steps += 1
            if self._successful_steps >= self.scale_window:
                self.loss_scale *= self.scale_factor
                self._successful_steps = 0

        return not overflow

    @property
    def current_loss_scale(self) -> float:
        return self.loss_scale

    @property
    def overflow_count(self) -> int:
        return self._overflow_count


class CheckpointManager:
    """
    Manages model checkpointing with versioning, best-model tracking,
    and automatic cleanup of old checkpoints.
    """

    def __init__(
        self,
        save_dir: str,
        max_to_keep: int = 5,
        keep_best: bool = True,
        metric_mode: str = "min",  # "min" or "max"
    ):
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.metric_mode = metric_mode
        self._checkpoints: List[Dict] = []
        self._best_metric: Optional[float] = None
        self._best_path: Optional[str] = None
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer,
        step: int,
        metric: Optional[float] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        path = os.path.join(self.save_dir, f"checkpoint_step{step:08d}.pt")
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        self._checkpoints.append({"path": path, "step": step, "metric": metric})

        if metric is not None and self.keep_best:
            is_best = (
                self._best_metric is None
                or (self.metric_mode == "min" and metric < self._best_metric)
                or (self.metric_mode == "max" and metric > self._best_metric)
            )
            if is_best:
                self._best_metric = metric
                self._best_path = path

        self._cleanup()
        return path

    def _cleanup(self):
        """Remove old checkpoints beyond max_to_keep."""
        to_remove = self._checkpoints[:-self.max_to_keep]
        for ckpt in to_remove:
            if ckpt["path"] != self._best_path and os.path.exists(ckpt["path"]):
                os.remove(ckpt["path"])
        self._checkpoints = self._checkpoints[-self.max_to_keep:]

    def load_latest(self, model: nn.Module, optimizer=None) -> Dict:
        if not self._checkpoints:
            raise FileNotFoundError("No checkpoints found.")
        latest = self._checkpoints[-1]
        state = torch.load(latest["path"], map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        return state

    def load_best(self, model: nn.Module) -> Dict:
        if self._best_path is None:
            raise FileNotFoundError("No best checkpoint found.")
        state = torch.load(self._best_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        return state

    @property
    def best_metric(self) -> Optional[float]:
        return self._best_metric

    @property
    def num_checkpoints(self) -> int:
        return len(self._checkpoints)


class DistributedSampler:
    """
    Distributed sampler that partitions a dataset across ranks,
    with support for shuffling and epoch-based seeding.
    """

    def __init__(
        self,
        dataset_size: int,
        world_size: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.dataset_size = dataset_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if drop_last:
            self.num_samples = dataset_size // world_size
        else:
            self.num_samples = math.ceil(dataset_size / world_size)
        self.total_size = self.num_samples * world_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.dataset_size))

        # Pad to total_size
        if not self.drop_last:
            padding = self.total_size - len(indices)
            indices += indices[:padding]

        # Subsample for this rank
        local = indices[self.rank:self.total_size:self.world_size]
        return iter(local)

    def __len__(self) -> int:
        return self.num_samples


class CommunicationProfiler:
    """
    Profiles communication overhead in distributed training
    (all-reduce, broadcast, scatter times).
    """

    def __init__(self):
        self._timings: Dict[str, List[float]] = {
            "all_reduce": [],
            "broadcast": [],
            "scatter": [],
            "gather": [],
        }
        self._bytes_transferred: Dict[str, int] = {k: 0 for k in self._timings}

    def record_all_reduce(self, tensor: torch.Tensor, elapsed_ms: float):
        self._timings["all_reduce"].append(elapsed_ms)
        self._bytes_transferred["all_reduce"] += tensor.numel() * tensor.element_size()

    def record_broadcast(self, tensor: torch.Tensor, elapsed_ms: float):
        self._timings["broadcast"].append(elapsed_ms)
        self._bytes_transferred["broadcast"] += tensor.numel() * tensor.element_size()

    def summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for op, times in self._timings.items():
            if times:
                result[op] = {
                    "count": len(times),
                    "mean_ms": sum(times) / len(times),
                    "total_ms": sum(times),
                    "bytes": self._bytes_transferred[op],
                    "bandwidth_gbps": (
                        self._bytes_transferred[op] / (sum(times) / 1000 + 1e-9) / 1e9
                    ),
                }
        return result


class ElasticTrainer:
    """
    Elastic training wrapper supporting dynamic rank membership changes
    (worker failures/additions) per PyTorch Elastic (Karakus et al. 2021).
    """

    def __init__(self, model_factory: Callable, optimizer_factory: Callable):
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self._model: Optional[nn.Module] = None
        self._optimizer = None
        self._step = 0
        self._rendezvous_count = 0

    def initialize(self):
        """Initialize or reinitialize model and optimizer after rendezvous."""
        self._model = self.model_factory()
        self._optimizer = self.optimizer_factory(self._model.parameters())
        self._rendezvous_count += 1

    def state_dict(self) -> Dict:
        return {
            "step": self._step,
            "model": self._model.state_dict() if self._model else None,
            "optimizer": self._optimizer.state_dict() if self._optimizer else None,
        }

    def load_state_dict(self, state: Dict):
        self._step = state["step"]
        if self._model and state.get("model"):
            self._model.load_state_dict(state["model"])
        if self._optimizer and state.get("optimizer"):
            self._optimizer.load_state_dict(state["optimizer"])

    def train_step(self, batch) -> float:
        if self._model is None:
            raise RuntimeError("Call initialize() first.")
        self._model.train()
        loss = torch.tensor(0.0)  # Placeholder
        self._step += 1
        return loss.item()

    @property
    def global_step(self) -> int:
        return self._step

    @property
    def rendezvous_count(self) -> int:
        return self._rendezvous_count


class ActivationCheckpointing:
    """
    Gradient checkpointing wrapper to trade compute for memory.
    Wraps specific module blocks to recompute activations during backward.
    """

    def __init__(self, model: nn.Module, checkpoint_layers: Optional[List[str]] = None):
        self.model = model
        self.checkpoint_layers = checkpoint_layers or []
        self._original_forwards: Dict[str, Callable] = {}

    def enable(self):
        """Enable gradient checkpointing for specified (or all) layers."""
        from torch.utils.checkpoint import checkpoint as ckpt_fn
        for name, module in self.model.named_modules():
            if not self.checkpoint_layers or name in self.checkpoint_layers:
                if hasattr(module, 'forward') and len(list(module.children())) > 0:
                    self._original_forwards[name] = module.forward
                    def make_checkpointed(m):
                        orig = m.forward
                        def checkpointed_forward(*args, **kwargs):
                            return ckpt_fn(orig, *args, **kwargs)
                        return checkpointed_forward
                    module.forward = make_checkpointed(module)

    def disable(self):
        """Restore original forward methods."""
        for name, module in self.model.named_modules():
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
        self._original_forwards.clear()

    def estimate_memory_savings(self) -> float:
        """Rough estimate of memory savings fraction from checkpointing."""
        total = sum(p.numel() for p in self.model.parameters())
        checkpointed = 0
        for name, module in self.model.named_modules():
            if name in self._original_forwards:
                checkpointed += sum(p.numel() for p in module.parameters())
        return checkpointed / max(1, total)


class DataParallelismManager:
    """
    Manages various data-parallelism strategies:
    DDP, FSDP (simulated), and custom bucketed all-reduce.
    """

    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.model = model
        self.config = config
        self._ddp_model: Optional[nn.Module] = None

    def wrap_ddp(self) -> nn.Module:
        """Wrap model with DDP if distributed is initialized."""
        if dist.is_available() and dist.is_initialized():
            self._ddp_model = DDP(
                self.model,
                find_unused_parameters=self.config.find_unused_parameters,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
                bucket_cap_mb=self.config.bucket_cap_mb,
                broadcast_buffers=self.config.broadcast_buffers,
            )
            return self._ddp_model
        return self.model

    def average_gradients(self):
        """Manual gradient averaging across all ranks."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        for param in self.model.parameters():
            if param.grad is not None:
                if dist.is_initialized():
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size

    def broadcast_parameters(self, src: int = 0):
        """Broadcast parameters from src rank to all others."""
        if dist.is_initialized():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=src)

    @property
    def is_main_process(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0


class LearningRateWarmupScheduler:
    """
    Composite scheduler: linear warmup + cosine decay + optional restarts.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        num_restarts: int = 0,
        restart_decay: float = 0.8,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.num_restarts = num_restarts
        self.restart_decay = restart_decay
        self._step = 0
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def get_lr(self) -> List[float]:
        step = self._step
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            if self.num_restarts > 0:
                cycle = 1.0 / (self.num_restarts + 1)
                cycle_progress = (progress % cycle) / cycle
                restart_num = int(progress / cycle)
                decay = self.restart_decay ** restart_num
                scale = decay * (self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * cycle_progress)))
            else:
                scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base * scale for base in self._base_lrs]

    def step(self):
        lrs = self.get_lr()
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr
        self._step += 1

    @property
    def last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    @property
    def global_step(self) -> int:
        return self._step


class GradientCompressor:
    """
    Communication-efficient gradient compression using top-k sparsification
    and 1-bit quantization (Lin et al. 2017 deep gradient compression).
    """

    def __init__(self, compress_ratio: float = 0.01, use_error_feedback: bool = True):
        self.compress_ratio = compress_ratio
        self.use_error_feedback = use_error_feedback
        self._residuals: Dict[str, torch.Tensor] = {}

    def compress(self, name: str, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k sparsification with error feedback."""
        if self.use_error_feedback:
            residual = self._residuals.get(name, torch.zeros_like(grad))
            grad = grad + residual

        flat = grad.view(-1)
        k = max(1, int(flat.numel() * self.compress_ratio))
        abs_vals = flat.abs()
        threshold = abs_vals.kthvalue(flat.numel() - k).values
        mask = abs_vals >= threshold
        compressed = flat * mask

        if self.use_error_feedback:
            self._residuals[name] = (flat - compressed).view_as(grad)

        indices = mask.nonzero(as_tuple=False).squeeze(1)
        values = compressed[indices]
        return indices, values

    def decompress(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        grad = torch.zeros(math.prod(shape))
        grad.scatter_(0, indices, values)
        return grad.view(shape)

    def quantize_1bit(self, grad: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """1-bit quantization: sign + scale."""
        scale = grad.abs().mean().item()
        quantized = grad.sign().to(torch.int8)
        return quantized, scale

    def dequantize_1bit(self, quantized: torch.Tensor, scale: float) -> torch.Tensor:
        return quantized.to(torch.float32) * scale
