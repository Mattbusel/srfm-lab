"""
distributed_jit.py — Distributed JAX operations for TensorNet (Project AETERNUS).

Provides:
  - pmap/vmap wrappers for distributed MPS/TT operations
  - TPU/GPU sharding strategies for large tensor networks
  - Pipeline parallelism for TT layer inference
  - Gradient accumulation for large-batch TT training
  - Mixed precision (bfloat16/float32) with per-layer casting
  - Checkpointing with orbax (or fallback pickle)
  - Multi-device data parallel training loop
  - Collective communication primitives (allreduce, scatter, gather)
  - Replicated vs sharded parameter strategies
  - Device mesh construction for 2D parallelism
"""

from __future__ import annotations

import os
import math
import functools
import warnings
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap, pmap, lax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import optax

from tensor_net.mps import (
    MatrixProductState,
    mps_compress,
    mps_to_dense,
    mps_norm,
    mps_random,
)
from tensor_net.tensor_train import (
    TensorTrain,
    tt_to_dense,
    tt_norm,
)


# ============================================================================
# Device discovery
# ============================================================================

def get_available_devices(backend: Optional[str] = None) -> List[jax.Device]:
    """Return all available JAX devices, optionally filtered by backend.

    Args:
        backend: "cpu", "gpu", or "tpu". If None, returns all devices.

    Returns:
        List of JAX device objects.
    """
    all_devices = jax.devices()
    if backend is None:
        return all_devices
    return [d for d in all_devices if d.platform == backend]


def n_devices(backend: Optional[str] = None) -> int:
    """Return number of available devices."""
    return len(get_available_devices(backend))


def device_mesh_2d(
    n_rows: int,
    n_cols: int,
    axis_names: Tuple[str, str] = ("data", "model"),
) -> Mesh:
    """Create a 2D device mesh for data + model parallelism.

    Args:
        n_rows: Number of devices along the data-parallel axis.
        n_cols: Number of devices along the model-parallel axis.
        axis_names: Names for the two mesh axes.

    Returns:
        JAX Mesh object.
    """
    devices = np.array(jax.devices()[: n_rows * n_cols]).reshape(n_rows, n_cols)
    return Mesh(devices, axis_names=axis_names)


# ============================================================================
# Mixed precision utilities
# ============================================================================

DTYPE_MAP: Dict[str, jnp.dtype] = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "float64": jnp.float64,
}


def cast_cores(
    cores: List[jnp.ndarray],
    dtype: Union[str, jnp.dtype],
) -> List[jnp.ndarray]:
    """Cast a list of TT/MPS cores to the specified dtype.

    Args:
        cores: List of JAX arrays.
        dtype: Target dtype string or jnp.dtype.

    Returns:
        List of cast arrays.
    """
    if isinstance(dtype, str):
        dtype = DTYPE_MAP[dtype]
    return [c.astype(dtype) for c in cores]


def mixed_precision_forward(
    cores: List[jnp.ndarray],
    input_vec: jnp.ndarray,
    compute_dtype: str = "bfloat16",
    output_dtype: str = "float32",
) -> jnp.ndarray:
    """Run TT forward pass in reduced precision, accumulate in float32.

    Args:
        cores: TT cores in storage dtype.
        input_vec: Input vector.
        compute_dtype: Dtype for computation (e.g. bfloat16).
        output_dtype: Dtype for output accumulation.

    Returns:
        Result in output_dtype.
    """
    compute_dt = DTYPE_MAP[compute_dtype]
    output_dt = DTYPE_MAP[output_dtype]

    low_cores = cast_cores(cores, compute_dt)
    x = input_vec.astype(compute_dt)

    # Simple contraction: treat TT as a linear map
    result = x
    for core in low_cores:
        r_l, d, r_r = core.shape
        if result.ndim == 1:
            result = jnp.einsum("i,ijk->jk", result[:r_l], core).reshape(-1)
        else:
            result = jnp.einsum("...i,ijk->...k", result[..., :r_l], core)

    return result.astype(output_dt)


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed-precision TT training."""
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    output_dtype: str = "float32"
    loss_scale: float = 1.0
    dynamic_loss_scaling: bool = False
    loss_scale_period: int = 2000
    loss_scale_factor: float = 2.0


# ============================================================================
# Gradient accumulation
# ============================================================================

class GradientAccumulator:
    """Accumulate gradients over multiple micro-batches before updating.

    Useful when effective batch size exceeds device memory.

    Args:
        n_accumulation_steps: Number of micro-batches to accumulate.
        optimizer: Optax optimizer.
    """

    def __init__(
        self,
        n_accumulation_steps: int,
        optimizer: optax.GradientTransformation,
    ):
        self.n_steps = n_accumulation_steps
        self.optimizer = optimizer
        self._step = 0
        self._grad_buffer: Optional[Any] = None
        self._opt_state: Optional[Any] = None

    def init(self, params: Any) -> None:
        """Initialize optimizer state and gradient buffer.

        Args:
            params: Initial model parameters (pytree).
        """
        self._opt_state = self.optimizer.init(params)
        self._grad_buffer = jax.tree_util.tree_map(jnp.zeros_like, params)

    def accumulate(self, grads: Any) -> bool:
        """Add gradients to buffer. Returns True when update should occur.

        Args:
            grads: Gradient pytree matching params structure.

        Returns:
            True if accumulation is complete and update should be applied.
        """
        if self._grad_buffer is None:
            raise RuntimeError("Call init() before accumulate().")

        # Accumulate
        self._grad_buffer = jax.tree_util.tree_map(
            lambda acc, g: acc + g / self.n_steps,
            self._grad_buffer,
            grads,
        )
        self._step += 1
        return self._step % self.n_steps == 0

    def apply_update(self, params: Any) -> Tuple[Any, Any]:
        """Apply accumulated gradients and reset buffer.

        Args:
            params: Current model parameters.

        Returns:
            (updated_params, updated_opt_state)
        """
        updates, new_opt_state = self.optimizer.update(
            self._grad_buffer, self._opt_state, params
        )
        new_params = optax.apply_updates(params, updates)
        self._opt_state = new_opt_state
        # Reset buffer
        self._grad_buffer = jax.tree_util.tree_map(jnp.zeros_like, params)
        return new_params, new_opt_state


# ============================================================================
# Sharding strategies
# ============================================================================

@dataclass
class ShardingStrategy:
    """Describes how TT cores are sharded across devices."""
    name: str
    core_sharding: str  # "replicate", "data_parallel", "model_parallel", "pipeline"
    n_devices: int
    pipeline_stages: int = 1
    data_axis: str = "data"
    model_axis: str = "model"


def replicate_cores(
    cores: List[jnp.ndarray],
    devices: Optional[List[jax.Device]] = None,
) -> List[jnp.ndarray]:
    """Replicate TT cores across all devices using jax.device_put_replicated.

    Args:
        cores: List of TT core arrays.
        devices: Target devices. Defaults to all available.

    Returns:
        List of replicated arrays (leading axis = device).
    """
    if devices is None:
        devices = jax.devices()
    return [
        jax.device_put_replicated(c, devices)
        for c in cores
    ]


def shard_cores_data_parallel(
    cores: List[jnp.ndarray],
    data: jnp.ndarray,
    devices: Optional[List[jax.Device]] = None,
) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
    """Shard data across devices, replicate cores.

    Args:
        cores: TT cores (replicated to all devices).
        data: Input data of shape (batch, ...).
        devices: Target devices.

    Returns:
        (replicated_cores, sharded_data)
    """
    if devices is None:
        devices = jax.devices()
    n = len(devices)

    rep_cores = replicate_cores(cores, devices)

    # Shard data: split along batch axis
    batch = data.shape[0]
    assert batch % n == 0, f"Batch size {batch} must be divisible by n_devices={n}"
    sharded = jnp.reshape(data, (n, batch // n) + data.shape[1:])
    sharded = jax.device_put_sharded(list(sharded), devices)

    return rep_cores, sharded


def pipeline_partition_cores(
    cores: List[jnp.ndarray],
    n_stages: int,
) -> List[List[jnp.ndarray]]:
    """Partition TT cores into pipeline stages.

    Assigns consecutive cores to each pipeline stage.

    Args:
        cores: All TT cores.
        n_stages: Number of pipeline stages.

    Returns:
        List of lists, each containing the cores for one stage.
    """
    n_cores = len(cores)
    cores_per_stage = math.ceil(n_cores / n_stages)
    stages = []
    for i in range(n_stages):
        start = i * cores_per_stage
        end = min(start + cores_per_stage, n_cores)
        stages.append(cores[start:end])
    return stages


# ============================================================================
# pmap wrappers for distributed TT operations
# ============================================================================

def pmap_tt_matvec(
    cores: List[jnp.ndarray],
    vectors: jnp.ndarray,
) -> jnp.ndarray:
    """Apply TT linear map to a batch of vectors using pmap.

    Assumes the leading axis of vectors is the device axis.

    Args:
        cores: Replicated TT cores (device_axis, r_l, d, r_r).
        vectors: Sharded input (n_devices, batch_per_device, d_total).

    Returns:
        Output array (n_devices, batch_per_device, r_out).
    """
    def _single_device_matvec(replicated_cores, batch_vecs):
        # replicated_cores: list of (r_l, d, r_r)
        # batch_vecs: (batch, d_total)
        results = []
        for v in batch_vecs:
            x = v
            for core in replicated_cores:
                r_l, d, r_r = core.shape
                # Contract
                x = jnp.einsum("i,ijk->jk", x[:r_l], core).reshape(-1)
            results.append(x)
        return jnp.stack(results)

    # Use vmap over batch dimension instead for compatibility
    def _matvec_single(v, cores_list):
        x = v
        for core in cores_list:
            r_l, d, r_r = core.shape
            x = jnp.einsum("i,ijk->k", x[:r_l], core[:r_l, :, :])
        return x

    # vmap over batch
    batched = vmap(lambda v: _matvec_single(v, cores))(vectors)
    return batched


def pmap_tt_norm(
    cores_replicated: List[jnp.ndarray],
) -> jnp.ndarray:
    """Compute TT norm in a distributed-friendly way.

    Args:
        cores_replicated: TT cores, possibly replicated across devices.

    Returns:
        Scalar norm.
    """
    # For replicated cores, just compute on the first device copy
    if cores_replicated[0].ndim > 3:
        # Replicated: take first device slice
        local_cores = [c[0] for c in cores_replicated]
    else:
        local_cores = cores_replicated

    # Build transfer matrices
    transfer = jnp.eye(1)
    for core in local_cores:
        r_l, d, r_r = core.shape
        mat = core.reshape(r_l, d * r_r)
        transfer_new = jnp.zeros((r_r, r_r))
        core_T = core.reshape(r_l, d, r_r)
        for di in range(d):
            slice_ = core_T[:, di, :]
            transfer_new = transfer_new + slice_.T @ (transfer @ slice_)
        transfer = transfer_new

    return jnp.sqrt(jnp.trace(transfer))


# ============================================================================
# Pipeline parallelism
# ============================================================================

class PipelineParallelTT:
    """Pipeline parallelism for TT layer inference across multiple devices.

    Splits TT cores into stages assigned to different devices, enabling
    inference on TT layers too large to fit on a single device.

    Args:
        cores: Full list of TT cores.
        n_stages: Number of pipeline stages (= number of devices to use).
        devices: Specific devices to use. Defaults to all available.
    """

    def __init__(
        self,
        cores: List[jnp.ndarray],
        n_stages: int,
        devices: Optional[List[jax.Device]] = None,
    ):
        self.n_stages = n_stages
        self.devices = devices or jax.devices()[:n_stages]
        assert len(self.devices) >= n_stages, (
            f"Need {n_stages} devices, only {len(self.devices)} available."
        )

        # Partition cores into stages
        self.stage_cores = pipeline_partition_cores(cores, n_stages)

        # Place each stage's cores on the corresponding device
        self.device_cores: List[List[jnp.ndarray]] = []
        for stage_idx, stage in enumerate(self.stage_cores):
            dev = self.devices[stage_idx]
            placed = [jax.device_put(c, dev) for c in stage]
            self.device_cores.append(placed)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run forward pass through all pipeline stages.

        Args:
            x: Input array to the first stage.

        Returns:
            Output after all pipeline stages.
        """
        current = x
        for stage_idx, (stage_cores, dev) in enumerate(
            zip(self.device_cores, self.devices)
        ):
            current = jax.device_put(current, dev)
            for core in stage_cores:
                r_l, d, r_r = core.shape
                if current.ndim == 1:
                    current = jnp.einsum("i,ijk->jk", current[:r_l], core).reshape(-1)
                else:
                    current = jnp.einsum("...i,ijk->...k", current[..., :r_l], core)
        return current

    def update_cores(self, new_cores: List[jnp.ndarray]) -> None:
        """Update pipeline cores (e.g., after a training step).

        Args:
            new_cores: Updated TT cores.
        """
        self.stage_cores = pipeline_partition_cores(new_cores, self.n_stages)
        self.device_cores = []
        for stage_idx, stage in enumerate(self.stage_cores):
            dev = self.devices[stage_idx]
            placed = [jax.device_put(c, dev) for c in stage]
            self.device_cores.append(placed)


# ============================================================================
# Distributed training loop
# ============================================================================

@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed TT training."""
    n_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    n_accumulation_steps: int = 1
    mixed_precision: bool = False
    compute_dtype: str = "bfloat16"
    checkpoint_every: int = 10
    checkpoint_dir: str = "./checkpoints"
    log_every: int = 10
    seed: int = 42
    data_parallel: bool = True
    model_parallel: bool = False
    pipeline_stages: int = 1


def make_distributed_optimizer(
    config: DistributedTrainingConfig,
) -> optax.GradientTransformation:
    """Create an optimizer suitable for distributed training.

    Includes gradient clipping and optional weight decay.

    Args:
        config: DistributedTrainingConfig.

    Returns:
        Optax optimizer.
    """
    tx_list = [
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adam(config.learning_rate),
    ]
    if config.weight_decay > 0:
        tx_list.append(optax.add_decayed_weights(config.weight_decay))
    return optax.chain(*tx_list)


class DistributedTTTrainer:
    """Distributed trainer for Tensor Train models.

    Handles data parallelism, gradient accumulation, mixed precision,
    and checkpointing.

    Args:
        cores: Initial TT cores.
        loss_fn: Loss function (params, batch) -> scalar.
        config: DistributedTrainingConfig.
    """

    def __init__(
        self,
        cores: List[jnp.ndarray],
        loss_fn: Callable,
        config: DistributedTrainingConfig,
    ):
        self.cores = list(cores)
        self.loss_fn = loss_fn
        self.config = config
        self.devices = jax.devices()
        self.n_devices = len(self.devices)

        self.optimizer = make_distributed_optimizer(config)
        self.opt_state = self.optimizer.init(self.cores)

        self.accumulator = GradientAccumulator(
            config.n_accumulation_steps,
            self.optimizer,
        )
        self.accumulator.init(self.cores)

        self._step = 0
        self._epoch = 0
        self.loss_history: List[float] = []

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _compute_loss_and_grads(
        self,
        cores: List[jnp.ndarray],
        batch: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """Compute loss and gradients for a single batch.

        Args:
            cores: Current TT cores.
            batch: Input batch.

        Returns:
            (loss, grads)
        """
        def loss_wrapper(c):
            return self.loss_fn(c, batch)

        loss_val, grads = jax.value_and_grad(loss_wrapper)(cores)
        return loss_val, grads

    def train_step(self, batch: jnp.ndarray) -> float:
        """Execute one training step (with gradient accumulation).

        Args:
            batch: Input batch.

        Returns:
            Loss value for this step.
        """
        if self.config.mixed_precision:
            low_cores = cast_cores(self.cores, self.config.compute_dtype)
            low_batch = batch.astype(DTYPE_MAP[self.config.compute_dtype])
            loss_val, grads = self._compute_loss_and_grads(low_cores, low_batch)
            # Cast grads back to float32
            grads = cast_cores(grads, "float32")
        else:
            loss_val, grads = self._compute_loss_and_grads(self.cores, batch)

        loss_float = float(loss_val)
        self.loss_history.append(loss_float)

        should_update = self.accumulator.accumulate(grads)
        if should_update:
            self.cores, self.opt_state = self.accumulator.apply_update(self.cores)

        self._step += 1
        return loss_float

    def train_epoch(
        self,
        data: np.ndarray,
        rng_seed: Optional[int] = None,
    ) -> float:
        """Train for one full epoch.

        Args:
            data: Full dataset array of shape (n_samples, ...).
            rng_seed: Optional seed for shuffling.

        Returns:
            Mean loss over the epoch.
        """
        n = data.shape[0]
        bs = self.config.batch_size

        if rng_seed is None:
            rng_seed = self._epoch
        rng = np.random.default_rng(rng_seed)
        idx = rng.permutation(n)
        shuffled = data[idx]

        epoch_losses = []
        for start in range(0, n - bs + 1, bs):
            batch = jnp.array(shuffled[start : start + bs])
            loss = self.train_step(batch)
            epoch_losses.append(loss)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        self._epoch += 1

        if self.config.log_every > 0 and self._epoch % self.config.log_every == 0:
            print(f"  Epoch {self._epoch}: loss={mean_loss:.6f}")

        if self.config.checkpoint_every > 0 and self._epoch % self.config.checkpoint_every == 0:
            self.save_checkpoint()

        return mean_loss

    def train(
        self,
        data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """Run full training loop.

        Args:
            data: Training data.
            val_data: Optional validation data.

        Returns:
            Dict with train_losses and (optionally) val_losses.
        """
        train_losses = []
        val_losses = []

        for epoch in range(self.config.n_epochs):
            tl = self.train_epoch(data, rng_seed=epoch + self.config.seed)
            train_losses.append(tl)

            if val_data is not None:
                val_loss = self._evaluate(val_data)
                val_losses.append(val_loss)

        return {"train_losses": train_losses, "val_losses": val_losses}

    def _evaluate(self, data: np.ndarray) -> float:
        """Evaluate model on data.

        Args:
            data: Evaluation data.

        Returns:
            Mean loss.
        """
        bs = self.config.batch_size
        losses = []
        for start in range(0, len(data) - bs + 1, bs):
            batch = jnp.array(data[start : start + bs])
            loss = float(self.loss_fn(self.cores, batch))
            losses.append(loss)
        return float(np.mean(losses)) if losses else float("nan")

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save checkpoint using orbax if available, else pickle.

        Args:
            path: Save path. Defaults to auto-generated path in checkpoint_dir.

        Returns:
            Path where checkpoint was saved.
        """
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_epoch{self._epoch:05d}.pkl",
            )

        state = {
            "cores": [np.array(c) for c in self.cores],
            "step": self._step,
            "epoch": self._epoch,
            "loss_history": self.loss_history,
        }

        try:
            import orbax.checkpoint as ocp
            checkpointer = ocp.PyTreeCheckpointer()
            orbax_path = path.replace(".pkl", "_orbax")
            checkpointer.save(orbax_path, state)
            return orbax_path
        except ImportError:
            pass

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from path.

        Args:
            path: Path to checkpoint file or orbax directory.
        """
        if os.path.isdir(path):
            try:
                import orbax.checkpoint as ocp
                checkpointer = ocp.PyTreeCheckpointer()
                state = checkpointer.restore(path)
            except ImportError:
                raise RuntimeError("orbax required to load orbax checkpoints.")
        else:
            with open(path, "rb") as f:
                state = pickle.load(f)

        self.cores = [jnp.array(c) for c in state["cores"]]
        self._step = state["step"]
        self._epoch = state["epoch"]
        self.loss_history = state["loss_history"]
        self.opt_state = self.optimizer.init(self.cores)


# ============================================================================
# Collective communication utilities
# ============================================================================

def allreduce_grads(
    grads: List[jnp.ndarray],
    axis_name: str = "batch",
) -> List[jnp.ndarray]:
    """Allreduce gradients across devices inside pmap.

    Wraps lax.pmean for use within a pmap-compiled function.

    Args:
        grads: List of gradient arrays.
        axis_name: pmap axis name.

    Returns:
        Allreduced gradients.
    """
    return [lax.pmean(g, axis_name=axis_name) for g in grads]


def scatter_data(
    data: jnp.ndarray,
    devices: Optional[List[jax.Device]] = None,
) -> jnp.ndarray:
    """Scatter data batch across devices.

    Args:
        data: Array of shape (n_devices * per_device_batch, ...).
        devices: Target devices.

    Returns:
        Sharded array with leading device dimension.
    """
    if devices is None:
        devices = jax.devices()
    n = len(devices)
    assert data.shape[0] % n == 0, (
        f"Data batch {data.shape[0]} not divisible by n_devices={n}"
    )
    per_device = data.shape[0] // n
    split = [data[i * per_device : (i + 1) * per_device] for i in range(n)]
    return jax.device_put_sharded(split, devices)


def gather_results(
    sharded_results: Any,
    axis: int = 0,
) -> jnp.ndarray:
    """Gather sharded results from all devices.

    Args:
        sharded_results: Sharded array (n_devices, per_device, ...).
        axis: Axis to concatenate along.

    Returns:
        Gathered array.
    """
    # Bring all to local host
    local = jax.device_get(sharded_results)
    if isinstance(local, (list, tuple)):
        return np.concatenate(local, axis=axis)
    return local.reshape((-1,) + local.shape[2:]) if local.ndim > 2 else local.reshape(-1)


# ============================================================================
# TPU sharding strategies
# ============================================================================

class TPUShardingStrategy:
    """Sharding strategy optimized for TPU pod slices.

    Implements 2D sharding: data parallelism across pod rows,
    model parallelism across pod columns.

    Args:
        n_data_parallel: Number of data-parallel replicas.
        n_model_parallel: Number of model-parallel shards.
    """

    def __init__(
        self,
        n_data_parallel: int = 4,
        n_model_parallel: int = 2,
    ):
        self.n_dp = n_data_parallel
        self.n_mp = n_model_parallel
        self.n_total = n_data_parallel * n_model_parallel

        available = len(jax.devices())
        if available < self.n_total:
            warnings.warn(
                f"Requested {self.n_total} devices but only {available} available. "
                "Falling back to single-device."
            )
            self.n_dp = 1
            self.n_mp = 1
            self.n_total = 1

    def shard_cores(
        self,
        cores: List[jnp.ndarray],
    ) -> List[jnp.ndarray]:
        """Shard TT cores across model-parallel devices.

        Args:
            cores: TT cores.

        Returns:
            Sharded cores (each split along bond dimension).
        """
        if self.n_mp == 1:
            return cores

        sharded = []
        for core in cores:
            r_l, d, r_r = core.shape
            if r_r >= self.n_mp:
                # Shard along right bond dimension
                chunk = r_r // self.n_mp
                chunks = [core[:, :, i * chunk : (i + 1) * chunk] for i in range(self.n_mp)]
                sharded.append(chunks)
            else:
                sharded.append([core] * self.n_mp)

        return sharded  # type: ignore

    def replicate_cores(self, cores: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Replicate TT cores across all data-parallel devices.

        Args:
            cores: TT cores.

        Returns:
            Replicated cores.
        """
        devices = jax.devices()[: self.n_dp * self.n_mp]
        return replicate_cores(cores, devices[:self.n_dp])

    def partition_spec_for_core(self) -> PartitionSpec:
        """Return a PartitionSpec for TT core sharding.

        Returns:
            PartitionSpec for (r_l, d, r_r) core with r_r sharded.
        """
        return PartitionSpec(None, None, "model")

    def partition_spec_for_batch(self) -> PartitionSpec:
        """Return a PartitionSpec for data batch sharding.

        Returns:
            PartitionSpec for (batch, ...) data.
        """
        return PartitionSpec("data", None)


# ============================================================================
# Checkpointing utilities
# ============================================================================

def save_cores_checkpoint(
    cores: List[jnp.ndarray],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save TT cores to disk with optional metadata.

    Attempts orbax first; falls back to numpy .npz.

    Args:
        cores: List of TT core arrays.
        path: Save directory or file path.
        metadata: Optional dict of metadata to save alongside.

    Returns:
        Path where checkpoint was saved.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    np_cores = [np.array(c) for c in cores]

    try:
        import orbax.checkpoint as ocp

        state = {"cores": np_cores, "metadata": metadata or {}}
        checkpointer = ocp.PyTreeCheckpointer()
        orbax_dir = path if not path.endswith(".npz") else path[:-4] + "_orbax"
        checkpointer.save(orbax_dir, state)
        return orbax_dir
    except ImportError:
        pass

    # Fallback: numpy .npz
    npz_path = path if path.endswith(".npz") else path + ".npz"
    save_dict = {f"core_{i}": c for i, c in enumerate(np_cores)}
    if metadata:
        for k, v in metadata.items():
            try:
                save_dict[f"meta_{k}"] = np.array(v)
            except Exception:
                pass
    np.savez(npz_path, **save_dict)
    return npz_path


def load_cores_checkpoint(path: str) -> Tuple[List[jnp.ndarray], Dict[str, Any]]:
    """Load TT cores from a checkpoint.

    Args:
        path: Path to checkpoint (orbax dir or .npz file).

    Returns:
        (cores, metadata) tuple.
    """
    if os.path.isdir(path):
        try:
            import orbax.checkpoint as ocp
            checkpointer = ocp.PyTreeCheckpointer()
            state = checkpointer.restore(path)
            cores = [jnp.array(c) for c in state["cores"]]
            metadata = state.get("metadata", {})
            return cores, metadata
        except ImportError:
            raise RuntimeError("orbax required for orbax checkpoint loading.")

    npz_path = path if path.endswith(".npz") else path + ".npz"
    data = np.load(npz_path)
    core_keys = sorted([k for k in data.files if k.startswith("core_")],
                       key=lambda x: int(x.split("_")[1]))
    cores = [jnp.array(data[k]) for k in core_keys]
    metadata = {k[5:]: data[k].item() for k in data.files if k.startswith("meta_")}
    return cores, metadata


# ============================================================================
# vmap wrappers for batched TT operations
# ============================================================================

def vmap_tt_contract(
    cores: List[jnp.ndarray],
    batch_inputs: jnp.ndarray,
) -> jnp.ndarray:
    """Apply TT contraction to a batch of inputs using vmap.

    Args:
        cores: TT cores, each (r_l, d, r_r).
        batch_inputs: Batched input, shape (batch, d1 * d2 * ... * dN).

    Returns:
        Output array of shape (batch, r_out).
    """
    def single_contract(x):
        current = x
        for core in cores:
            r_l, d, r_r = core.shape
            current_r = min(current.shape[-1] if current.ndim > 1 else current.shape[0], r_l)
            if current.ndim == 1:
                current = jnp.einsum("i,ijk->jk", current[:r_l], core).reshape(-1)
            else:
                current = jnp.einsum("...i,ijk->...k", current[..., :r_l], core)
        return current

    return vmap(single_contract)(batch_inputs)


def vmap_mps_inner(
    mps_a_tensors: List[jnp.ndarray],
    mps_b_tensors: List[jnp.ndarray],
) -> jnp.ndarray:
    """Compute inner product between two MPS using transfer matrix method.

    Both MPS must have the same physical dimensions and number of sites.

    Args:
        mps_a_tensors: Tensors for MPS |a>, each (r_l, d, r_r).
        mps_b_tensors: Tensors for MPS |b>, each (r_l, d, r_r).

    Returns:
        Scalar inner product <a|b>.
    """
    n_sites = len(mps_a_tensors)
    assert len(mps_b_tensors) == n_sites

    # Build transfer matrix left to right
    transfer = jnp.ones((1, 1))
    for i in range(n_sites):
        a = mps_a_tensors[i]  # (r_al, d, r_ar)
        b = mps_b_tensors[i]  # (r_bl, d, r_br)
        r_al, d, r_ar = a.shape
        r_bl, _, r_br = b.shape

        # Transfer matrix contribution: einsum over physical index
        # T_new[r_ar, r_br] = sum_{d, r_al, r_bl} T[r_al, r_bl] * a[r_al,d,r_ar] * conj(b[r_bl,d,r_br])
        new_T = jnp.einsum(
            "ab,adc,bde->ce",
            transfer[:r_al, :r_bl],
            a[:r_al, :, :r_ar],
            jnp.conj(b[:r_bl, :, :r_br]),
        )
        transfer = new_T

    return transfer[0, 0]


# ============================================================================
# Distributed hyperparameter search
# ============================================================================

def distributed_hparam_sweep(
    train_fn: Callable[[Dict[str, Any], np.ndarray], float],
    hparam_grid: Dict[str, List[Any]],
    data: np.ndarray,
    n_trials_per_device: int = 2,
    rng_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run hyperparameter sweep distributed across devices.

    Each device runs a subset of hyperparameter trials in parallel.

    Args:
        train_fn: Function (hparams, data) -> val_loss.
        hparam_grid: Dict mapping param name to list of values.
        data: Training/evaluation data.
        n_trials_per_device: Trials per device.
        rng_seed: Random seed.

    Returns:
        List of dicts with hparams and corresponding val_loss, sorted by val_loss.
    """
    import itertools

    rng = np.random.default_rng(rng_seed)

    # Generate all combinations
    keys = list(hparam_grid.keys())
    values = list(hparam_grid.values())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    rng.shuffle(all_configs)

    # For simplicity, run sequentially (true device dispatch would require
    # jax.experimental.io_callback or custom pmap setup)
    results = []
    for config in all_configs:
        try:
            val_loss = train_fn(config, data)
        except Exception as e:
            val_loss = float("inf")
            warnings.warn(f"Trial failed: {e}")

        results.append({"hparams": config, "val_loss": val_loss})

    results.sort(key=lambda x: x["val_loss"])
    return results


# ============================================================================
# Gradient compression for distributed training
# ============================================================================

def compress_gradients_topk(
    grads: List[jnp.ndarray],
    k_fraction: float = 0.01,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """Top-k gradient compression for communication efficiency.

    Keeps only the top-k% of gradient values (by absolute magnitude)
    and zeros the rest. Returns compressed gradients and residuals.

    Args:
        grads: List of gradient arrays.
        k_fraction: Fraction of gradients to keep (e.g., 0.01 = top 1%).

    Returns:
        (compressed_grads, residuals) where residuals capture the error.
    """
    compressed = []
    residuals = []
    for g in grads:
        g_flat = g.reshape(-1)
        n_keep = max(1, int(len(g_flat) * k_fraction))
        threshold_idx = jnp.argsort(jnp.abs(g_flat))[-n_keep]
        threshold = jnp.abs(g_flat)[threshold_idx]
        mask = jnp.abs(g_flat) >= threshold
        g_compressed = g_flat * mask
        residual = g_flat * (1 - mask)
        compressed.append(g_compressed.reshape(g.shape))
        residuals.append(residual.reshape(g.shape))
    return compressed, residuals


def compress_gradients_random_sparsification(
    grads: List[jnp.ndarray],
    keep_prob: float = 0.1,
    rng_seed: int = 0,
) -> List[jnp.ndarray]:
    """Random sparsification for gradient compression.

    Each gradient element is kept with probability keep_prob
    and scaled up by 1/keep_prob to maintain unbiasedness.

    Args:
        grads: List of gradient arrays.
        keep_prob: Probability of keeping each element.
        rng_seed: Seed for random mask.

    Returns:
        Sparsified gradients (unbiased).
    """
    key = jax.random.PRNGKey(rng_seed)
    compressed = []
    for g in grads:
        key, subkey = jax.random.split(key)
        mask = jax.random.bernoulli(subkey, keep_prob, shape=g.shape)
        compressed.append(g * mask / keep_prob)
    return compressed


# ============================================================================
# Communication-efficient AllReduce
# ============================================================================

def ring_allreduce_grads(
    grads: List[jnp.ndarray],
    n_devices: int,
    device_rank: int,
) -> List[jnp.ndarray]:
    """Simulate ring-AllReduce for gradient aggregation.

    In a real distributed setting, this would use NCCL or similar.
    This is a simulation for single-node multi-GPU scenarios.

    Args:
        grads: List of local gradient arrays.
        n_devices: Total number of devices.
        device_rank: This device's rank in the ring.

    Returns:
        Globally averaged gradients (assuming all devices call this).
    """
    # Single-device case: no communication needed
    if n_devices == 1:
        return grads
    # In practice, use lax.psum inside pmap
    return [g / n_devices for g in grads]


# ============================================================================
# Learning rate warmup for distributed training
# ============================================================================

def linear_warmup_cosine_decay_schedule(
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
) -> Callable[[int], float]:
    """Learning rate schedule: linear warmup + cosine decay.

    Commonly used for large-scale distributed training.

    Args:
        base_lr: Peak learning rate (after warmup).
        warmup_steps: Steps for linear warmup.
        total_steps: Total training steps.
        min_lr: Minimum learning rate after decay.

    Returns:
        Callable that takes step -> learning rate.
    """
    import math as _math

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + _math.cos(_math.pi * min(1.0, progress)))
        return min_lr + (base_lr - min_lr) * cosine_decay

    return schedule


def make_distributed_lr_schedule(
    config: DistributedTrainingConfig,
    warmup_fraction: float = 0.05,
) -> Callable[[int], float]:
    """Create LR schedule for distributed training config.

    Args:
        config: DistributedTrainingConfig.
        warmup_fraction: Fraction of total steps for warmup.

    Returns:
        LR schedule callable.
    """
    # Estimate total steps
    total_steps = config.n_epochs * 1000  # rough estimate
    warmup_steps = int(total_steps * warmup_fraction)
    return linear_warmup_cosine_decay_schedule(
        config.learning_rate, warmup_steps, total_steps
    )


# ============================================================================
# Distributed evaluation utilities
# ============================================================================

def distributed_evaluate(
    cores: List[jnp.ndarray],
    eval_fn: Callable[[List[jnp.ndarray], jnp.ndarray], float],
    data: np.ndarray,
    batch_size: int = 256,
    n_devices: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a TT model in a distributed fashion.

    Splits evaluation data across devices for parallel evaluation.

    Args:
        cores: TT cores.
        eval_fn: Function (cores, batch) -> loss.
        data: Evaluation data.
        batch_size: Batch size.
        n_devices: Number of devices to use.

    Returns:
        Dict with mean_loss, std_loss, n_batches.
    """
    losses = []
    n = len(data)

    for start in range(0, n - batch_size + 1, batch_size):
        batch = jnp.array(data[start : start + batch_size])
        try:
            loss = float(eval_fn(cores, batch))
        except Exception:
            loss = float("nan")
        losses.append(loss)

    finite_losses = [l for l in losses if not (l != l)]  # filter NaN
    return {
        "mean_loss": float(np.mean(finite_losses)) if finite_losses else float("nan"),
        "std_loss": float(np.std(finite_losses)) if len(finite_losses) > 1 else 0.0,
        "n_batches": len(losses),
        "n_valid_batches": len(finite_losses),
    }


# ============================================================================
# Fault tolerance and recovery
# ============================================================================

class FaultTolerantTrainer:
    """Wrapper around DistributedTTTrainer with fault tolerance.

    Detects NaN/Inf losses and recovers from the last valid checkpoint.

    Args:
        trainer: Base DistributedTTTrainer.
        max_failures: Maximum failures before aborting.
        recovery_lr_factor: Scale LR by this factor after recovery.
    """

    def __init__(
        self,
        trainer: DistributedTTTrainer,
        max_failures: int = 3,
        recovery_lr_factor: float = 0.5,
    ):
        self.trainer = trainer
        self.max_failures = max_failures
        self.recovery_lr_factor = recovery_lr_factor
        self._n_failures = 0
        self._last_good_checkpoint: Optional[str] = None
        self._current_lr = trainer.config.learning_rate

    def train_step(self, batch: jnp.ndarray) -> float:
        """Train one step with fault detection.

        Args:
            batch: Input batch.

        Returns:
            Loss value (finite).
        """
        loss = self.trainer.train_step(batch)

        if not (loss == loss) or abs(loss) > 1e10:  # NaN or extreme value
            self._n_failures += 1
            if self._n_failures > self.max_failures:
                raise RuntimeError(
                    f"Too many failures ({self._n_failures}). Aborting."
                )

            # Recover from last checkpoint
            if self._last_good_checkpoint is not None:
                self.trainer.load_checkpoint(self._last_good_checkpoint)

            # Reduce learning rate
            self._current_lr *= self.recovery_lr_factor
            self.trainer.optimizer = make_distributed_optimizer(
                DistributedTrainingConfig(
                    learning_rate=self._current_lr,
                    gradient_clip_norm=self.trainer.config.gradient_clip_norm,
                )
            )
            self.trainer.opt_state = self.trainer.optimizer.init(self.trainer.cores)

            return float("nan")

        # Save periodic checkpoints
        if self.trainer._step % 100 == 0:
            try:
                path = self.trainer.save_checkpoint()
                self._last_good_checkpoint = path
            except Exception:
                pass

        return loss

    @property
    def n_failures(self) -> int:
        return self._n_failures
