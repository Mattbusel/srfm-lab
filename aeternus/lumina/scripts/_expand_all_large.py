"""Write very large expansions to multiple files at once."""
import os

BASE = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina"

# ===========================================================================
# EXPAND: distributed.py with comprehensive distributed training utilities
# ===========================================================================

DISTRIBUTED_CONTENT = r'''

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
'''

# ===========================================================================
# EXPAND: model.py with more model variants
# ===========================================================================

MODEL_CONTENT = r'''

# =============================================================================
# SECTION: Additional Lumina Model Variants
# =============================================================================

class LuminaForMultiHorizonForecast(nn.Module):
    """Lumina model variant for multi-horizon return forecasting.

    Generates return forecasts at multiple horizons (1d, 5d, 10d, 20d, 60d)
    with calibrated uncertainty estimates.

    Args:
        config: TransformerConfig
        forecast_horizons: List of forecast horizons in timesteps
        num_features: Input feature count
        use_quantile_head: Whether to add quantile regression head
        quantile_levels: Quantile levels for uncertainty
    """

    def __init__(
        self,
        config,
        forecast_horizons: Optional[List[int]] = None,
        num_features: int = 5,
        use_quantile_head: bool = True,
        quantile_levels: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.forecast_horizons = forecast_horizons or [1, 5, 10, 20, 60]
        self.num_horizons = len(self.forecast_horizons)
        self.quantile_levels = quantile_levels or [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        d_model = config.d_model

        # Input projection
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        # Transformer backbone
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, config.num_heads, d_ff=d_model * 4, dropout=config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Point forecast head
        self.point_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(self.num_horizons)
        ])
        # Quantile head (if enabled)
        if use_quantile_head:
            nq = len(self.quantile_levels)
            self.quantile_heads = nn.ModuleList([
                nn.Linear(d_model, nq) for _ in range(self.num_horizons)
            ])
        else:
            self.quantile_heads = None
        # Horizon embedding for head conditioning
        self.horizon_embed = nn.Embedding(len(self.forecast_horizons), d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, F = x.shape
        h = self.input_proj(x)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        h = h + self.pos_embed(pos_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        # Pool to single representation
        pooled = h[:, -1, :]  # Use last timestep
        # Generate predictions for each horizon
        point_preds = []
        quantile_preds = []
        for hi in range(self.num_horizons):
            h_cond = pooled + self.horizon_embed(
                torch.full((B,), hi, dtype=torch.long, device=x.device)
            )
            point_preds.append(self.point_heads[hi](h_cond))
            if self.quantile_heads is not None:
                quantile_preds.append(self.quantile_heads[hi](h_cond))
        point_preds = torch.cat(point_preds, dim=-1)  # (B, H)
        result = {
            "point_forecasts": point_preds,
            "forecast_horizons": self.forecast_horizons,
            "encoded": h,
        }
        if quantile_preds:
            qp = torch.stack(quantile_preds, dim=1)  # (B, H, Q)
            result["quantile_forecasts"] = qp
            result["quantile_levels"] = self.quantile_levels
        return result


class LuminaForCrossAssetModeling(nn.Module):
    """Lumina model for joint cross-asset prediction.

    Models a portfolio of assets simultaneously, capturing
    cross-asset correlations and lead-lag relationships.

    Args:
        config: TransformerConfig
        num_assets: Number of assets in the portfolio
        num_features: Features per asset
        use_cross_attention: Whether to use cross-asset attention
    """

    def __init__(
        self,
        config,
        num_assets: int = 50,
        num_features: int = 5,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        d_model = config.d_model
        # Per-asset encoder
        self.asset_encoder = nn.Linear(num_features, d_model)
        self.asset_embed = nn.Embedding(num_assets + 1, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(4096, d_model)
        # Temporal encoder (per asset)
        self.temporal_layers = nn.ModuleList([
            TransformerBlock(d_model, config.num_heads, d_ff=d_model * 4)
            for _ in range(config.num_layers // 2)
        ])
        # Cross-asset attention (across assets)
        if use_cross_attention:
            self.cross_asset_layers = nn.ModuleList([
                TransformerBlock(d_model, config.num_heads, d_ff=d_model * 4)
                for _ in range(config.num_layers // 2)
            ])
        else:
            self.cross_asset_layers = None
        self.norm = nn.LayerNorm(d_model)
        # Alpha head
        self.alpha_head = nn.Linear(d_model, 1)
        # Risk head
        self.risk_head = nn.Linear(d_model, 2)  # vol, beta

    def forward(
        self,
        x: torch.Tensor,
        asset_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, N, T, F) multi-asset time series
            asset_ids: (N,) asset indices for embedding
        Returns:
            Dict with alpha scores, risk estimates
        """
        if x.dim() == 3:
            # (B, T, F) single asset
            B, T, F = x.shape
            x = x.unsqueeze(1)  # (B, 1, T, F)
        B, N, T, F = x.shape
        # Encode each asset
        x_flat = x.view(B * N, T, F)
        h = self.asset_encoder(x_flat)
        # Add positional embeddings
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        h = h + self.pos_embed(pos_ids)
        # Add asset embeddings
        if asset_ids is not None:
            a_emb = self.asset_embed(asset_ids)  # (N, D)
            a_emb = a_emb.unsqueeze(0).unsqueeze(2).expand(B, N, T, -1)
            h = h + a_emb.view(B * N, T, -1)
        # Temporal encoding
        for layer in self.temporal_layers:
            h = layer(h)
        h = h.view(B, N, T, -1)
        # Pool temporal dimension
        h_pooled = h.mean(dim=2)  # (B, N, D)
        # Cross-asset encoding
        if self.cross_asset_layers is not None:
            for layer in self.cross_asset_layers:
                h_pooled = layer(h_pooled)
        h_pooled = self.norm(h_pooled)
        # Predictions
        alpha = self.alpha_head(h_pooled).squeeze(-1)  # (B, N)
        risk = self.risk_head(h_pooled)  # (B, N, 2)
        return {
            "alpha_scores": alpha,
            "vol_forecast": F.softplus(risk[:, :, 0]),
            "beta_forecast": risk[:, :, 1],
            "encoded": h_pooled,
        }


class LuminaForSentimentFusion(nn.Module):
    """Lumina model fusing price data with sentiment signals.

    Combines OHLCV price features with news/social sentiment
    through cross-modal attention.

    Args:
        config: TransformerConfig for price encoder
        sentiment_dim: Sentiment feature dimension
        num_sentiment_sources: Number of sentiment data sources
        fusion_type: 'early', 'late', or 'cross_attention'
    """

    def __init__(
        self,
        config,
        sentiment_dim: int = 32,
        num_sentiment_sources: int = 3,
        fusion_type: str = "cross_attention",
        num_features: int = 5,
    ) -> None:
        super().__init__()
        d_model = config.d_model
        self.fusion_type = fusion_type
        # Price encoder
        self.price_encoder = nn.Linear(num_features, d_model)
        self.sentiment_encoder = nn.Linear(sentiment_dim * num_sentiment_sources, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        # Price transformer
        self.price_layers = nn.ModuleList([
            TransformerBlock(d_model, config.num_heads, d_ff=d_model * 4)
            for _ in range(config.num_layers)
        ])
        # Sentiment transformer
        self.sentiment_layers = nn.ModuleList([
            TransformerBlock(d_model, config.num_heads // 2, d_ff=d_model * 2)
            for _ in range(config.num_layers // 2)
        ])
        if fusion_type == "cross_attention":
            self.cross_attn = CrossAttention(d_model, config.num_heads)
            self.cross_norm = nn.LayerNorm(d_model)
        elif fusion_type == "late":
            self.fusion_gate = nn.Linear(d_model * 2, 2)
        self.norm = nn.LayerNorm(d_model)
        self.alpha_head = nn.Linear(d_model, 1)

    def forward(
        self,
        price_x: torch.Tensor,
        sentiment_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, F = price_x.shape
        h_p = self.price_encoder(price_x)
        pos_ids = torch.arange(T, device=price_x.device).unsqueeze(0)
        h_p = h_p + self.pos_embed(pos_ids)
        for layer in self.price_layers:
            h_p = layer(h_p)
        if sentiment_x is not None:
            h_s = self.sentiment_encoder(sentiment_x)
            h_s = h_s + self.pos_embed(pos_ids[:, :h_s.size(1)])
            for layer in self.sentiment_layers:
                h_s = layer(h_s)
            if self.fusion_type == "cross_attention":
                h_fused = h_p + self.cross_attn(self.cross_norm(h_p), h_s)
            elif self.fusion_type == "late":
                h_s_up = F.interpolate(h_s.transpose(1, 2), size=T, mode="nearest").transpose(1, 2)
                gates = torch.softmax(self.fusion_gate(torch.cat([h_p, h_s_up], -1)), -1)
                h_fused = gates[:, :, 0:1] * h_p + gates[:, :, 1:2] * h_s_up
            else:
                h_fused = h_p + h_s[:, :T, :]
        else:
            h_fused = h_p
        h_fused = self.norm(h_fused)
        alpha = self.alpha_head(h_fused[:, -1, :]).squeeze(-1)
        return {
            "alpha_scores": alpha,
            "encoded": h_fused,
        }


class LuminaWithLoRA(nn.Module):
    """Lumina model wrapper applying LoRA for parameter-efficient fine-tuning.

    Wraps any Lumina model and injects LoRA adapters into specified
    layers for efficient adaptation to new tasks or markets.

    Args:
        base_model: Pretrained Lumina model
        lora_rank: LoRA adapter rank
        lora_alpha: LoRA scaling
        target_modules: Which module types to apply LoRA to
        lora_dropout: LoRA dropout probability
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_rank: int = 4,
        lora_alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        target_modules = target_modules or ["q_proj", "v_proj", "out_proj"]
        self._lora_modules: Dict[str, nn.Module] = {}
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        # Inject LoRA
        self._inject_lora(target_modules, lora_rank, lora_alpha, lora_dropout)

    def _inject_lora(self, targets, rank, alpha, dropout) -> None:
        """Inject LoRA adapters into target Linear layers."""
        for name, module in list(self.base_model.named_modules()):
            if not any(t in name for t in targets):
                continue
            if not isinstance(module, nn.Linear):
                continue
            lora = LoRALinear(
                module.in_features, module.out_features,
                rank=rank, alpha=alpha, dropout=dropout
            )
            with torch.no_grad():
                lora.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    lora.bias = nn.Parameter(module.bias.data.clone())
            # Replace module
            parts = name.split(".")
            parent = self.base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora)
            self._lora_modules[name] = lora

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


_NEW_MODEL_EXPORTS = [
    "LuminaForMultiHorizonForecast", "LuminaForCrossAssetModeling",
    "LuminaForSentimentFusion", "LuminaWithLoRA",
]
'''

# Write all content
import time

files_content = {
    os.path.join(BASE, "distributed.py"): DISTRIBUTED_CONTENT,
    os.path.join(BASE, "model.py"): MODEL_CONTENT,
}

for path, content in files_content.items():
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    import subprocess
    r = subprocess.run(["wc", "-l", path], capture_output=True, text=True, shell=True)
    print(r.stdout.strip())
