#!/usr/bin/env python3
"""
scripts/distributed_pretrain.py

torchrun-compatible distributed pretraining script for Lumina.

Usage:
    # Single GPU:
    python scripts/distributed_pretrain.py --config configs/pretrain_base.yaml

    # Multi-GPU (4 GPUs, 1 node):
    torchrun --nproc_per_node=4 scripts/distributed_pretrain.py --config configs/pretrain_base.yaml

    # Multi-node (2 nodes, 4 GPUs each):
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
             --master_addr=master_host --master_port=29500 \
             scripts/distributed_pretrain.py --config configs/pretrain_base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# ── Lumina imports ──────────────────────────────────────────────────────────
# Adjust path so we can import from the package
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lumina.distributed_training import (
    DistributedConfig,
    DistributedTrainer,
    ProcessGroupManager,
    setup_distributed,
    cleanup_distributed,
    cosine_with_warmup_schedule,
    wsd_schedule,
    build_optimizer,
    CheckpointManager,
    TrainingState,
    AMPContext,
    ModelEMA,
    setup_logging,
    count_parameters,
    print_model_summary,
    LAMB,
)
from lumina.experiment_tracker import (
    RunConfig,
    ExperimentTracker,
    EarlyStoppingCallback,
)
from lumina.scaling import (
    ModelConfig,
    ParameterCounter,
    HyperparameterScaler,
    ChinchillaScalingLaw,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    # Data
    data_root: str = "data/pretrain"
    val_data_root: str = "data/val"
    seq_len: int = 2048
    feature_dim: int = 64

    # Model
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    ffn_mult: float = 4.0
    use_moe: bool = False
    n_experts: int = 8
    n_active_experts: int = 2
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    vocab_size: int = 50_000

    # Training
    n_steps: int = 100_000
    n_epochs: Optional[int] = None
    batch_size: int = 16        # Per-GPU batch size
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0

    # Optimizer
    optimizer: str = "adamw"    # "adamw" | "lamb"
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"   # "cosine" | "wsd"
    warmup_steps: int = 2000
    stable_steps: int = 80_000
    decay_steps: int = 18_000

    # AMP / precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # Distribution
    strategy: str = "ddp"       # "ddp" | "fsdp" | "none"
    fsdp_sharding: str = "full_shard"
    gradient_checkpointing: bool = True

    # Checkpoints
    checkpoint_dir: str = "checkpoints/pretrain"
    resume_from: Optional[str] = None
    save_every_n_steps: int = 1000
    keep_last_n: int = 3

    # Logging
    log_every_n_steps: int = 10
    experiment_name: str = "lumina_pretrain"
    run_name: Optional[str] = None

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Profiler
    enable_profiler: bool = False

    # Seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Synthetic dataset (placeholder — replace with real data pipeline)
# ---------------------------------------------------------------------------

class SyntheticFinancialDataset(Dataset):
    """
    Synthetic financial time-series dataset for testing.
    In production, replace with MemoryMappedFinancialDataset or
    StreamingFinancialDataset from lumina.data_lake.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        feature_dim: int,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)

        # Pre-generate data to avoid repeated RNG calls during training
        logger.info(f"Generating synthetic dataset: {n_samples} samples x ({seq_len}, {feature_dim})")
        self.data = self.rng.randn(n_samples, seq_len, feature_dim).astype(np.float32)
        # Labels: direction of the mean feature at next step
        self.labels = (self.data[:, -1, 0] > 0).astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.data[idx])
        y = torch.tensor(self.labels[idx])
        return {"input_ids": x, "labels": y}


# ---------------------------------------------------------------------------
# Minimal Lumina model for pretraining (replaces importing full model)
# ---------------------------------------------------------------------------

class LuminaPretrainModel(nn.Module):
    """
    Simplified Lumina model suitable for pretraining.
    In production, import LuminaModel from lumina.transformer.
    """

    def __init__(self, config: PretrainConfig):
        super().__init__()
        d = config.d_model
        self.input_proj = nn.Linear(config.feature_dim, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=int(d * config.ffn_mult),
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)    # Binary direction prediction

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        x = self.input_proj(input_ids)
        x = self.encoder(x)
        x = self.norm(x)
        logits = self.head(x[:, -1, :]).squeeze(-1)    # Last token
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return output


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: PretrainConfig,
    rank: int,
    world_size: int,
) -> tuple:
    """Build train and validation DataLoaders."""
    train_ds = SyntheticFinancialDataset(
        n_samples=50_000,
        seq_len=config.seq_len,
        feature_dim=config.feature_dim,
        seed=config.seed,
    )
    val_ds = SyntheticFinancialDataset(
        n_samples=5_000,
        seq_len=config.seq_len,
        feature_dim=config.feature_dim,
        seed=config.seed + 1,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_shuffle = False
        val_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True
        val_shuffle = False

    n_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=n_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=val_shuffle,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def compute_tokens_per_second(
    n_tokens: int,
    elapsed_sec: float,
) -> float:
    return n_tokens / elapsed_sec if elapsed_sec > 0 else 0.0


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(config: PretrainConfig) -> None:
    # ── Setup distributed ────────────────────────────────────────────────
    dist_config = DistributedConfig(
        strategy=config.strategy,
        use_amp=config.use_amp,
        amp_dtype=config.amp_dtype,
        grad_accum_steps=config.grad_accum_steps,
        max_grad_norm=config.max_grad_norm,
        checkpoint_dir=config.checkpoint_dir,
        save_every_n_steps=config.save_every_n_steps,
        keep_last_n_checkpoints=config.keep_last_n,
        resume_from=config.resume_from,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        log_every_n_steps=config.log_every_n_steps,
        fsdp_sharding=config.fsdp_sharding,
        enable_profiler=config.enable_profiler,
    )

    pgm = setup_distributed(dist_config)
    rank = pgm.rank
    world_size = pgm.world_size
    is_main = pgm.is_main_process
    device = torch.device(f"cuda:{pgm.local_rank}" if torch.cuda.is_available() else "cpu")

    setup_logging(rank)

    if is_main:
        logger.info(f"Starting distributed pretraining | rank={rank}, world_size={world_size}")
        logger.info(f"Config: {asdict(config)}")

    # ── Experiment tracking ──────────────────────────────────────────────
    tracker_config = RunConfig(
        experiment_name=config.experiment_name,
        run_name=config.run_name,
        use_local=is_main,
        use_mlflow=False,
        use_wandb=False,
    )
    tracker = ExperimentTracker(tracker_config)

    if is_main:
        tracker.start_run(config.run_name or "pretrain")
        tracker.log_params(asdict(config))

    # ── Build model ──────────────────────────────────────────────────────
    model = LuminaPretrainModel(config).to(device)

    if is_main:
        print_model_summary(model)
        n_params = count_parameters(model)
        logger.info(f"Total parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ── Wrap with DDP / FSDP ─────────────────────────────────────────────
    if config.strategy == "ddp" and world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[pgm.local_rank], find_unused_parameters=False)
        raw_model = model.module
    elif config.strategy == "fsdp" and world_size > 1:
        try:
            from lumina.distributed_training import wrap_model_fsdp
            model = wrap_model_fsdp(model, dist_config)
            raw_model = model
        except Exception as e:
            logger.warning(f"FSDP wrapping failed: {e}; using DDP.")
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[pgm.local_rank])
            raw_model = model.module
    else:
        raw_model = model

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        optimizer_type=config.optimizer,
    )

    # ── Data loaders ─────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(config, rank, world_size)
    n_steps_per_epoch = len(train_loader) // config.grad_accum_steps

    total_steps = config.n_steps
    if config.n_epochs is not None:
        total_steps = config.n_epochs * n_steps_per_epoch

    # ── Scheduler ────────────────────────────────────────────────────────
    if config.scheduler == "cosine":
        scheduler = cosine_with_warmup_schedule(
            optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=config.min_lr / config.lr,
        )
    elif config.scheduler == "wsd":
        scheduler = wsd_schedule(
            optimizer,
            warmup_steps=config.warmup_steps,
            stable_steps=config.stable_steps,
            decay_steps=config.decay_steps,
            min_lr_ratio=config.min_lr / config.lr,
        )
    else:
        scheduler = None

    # ── AMP ──────────────────────────────────────────────────────────────
    amp = AMPContext(dist_config)

    # ── EMA ──────────────────────────────────────────────────────────────
    ema = ModelEMA(raw_model, decay=config.ema_decay) if (config.use_ema and is_main) else None

    # ── Checkpoint manager ───────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(
        config.checkpoint_dir,
        keep_last_n=config.keep_last_n,
        rank=rank,
        world_size=world_size,
    )
    state = TrainingState()

    # Resume from checkpoint
    if config.resume_from:
        try:
            state = ckpt_mgr.load(config.resume_from, model, optimizer, scheduler, amp)
            logger.info(f"Resumed from step {state.step}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    # ── Training loop ────────────────────────────────────────────────────
    early_stopper = EarlyStoppingCallback(monitor="val_loss", patience=5)

    if is_main:
        logger.info(f"Training for {total_steps} steps ({config.n_epochs or '?'} epochs)")

    global_step = state.step
    epoch = state.epoch
    accum_step = 0
    train_start = time.perf_counter()

    while global_step < total_steps:
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()

        for batch in train_loader:
            if global_step >= total_steps:
                break

            # Move to device
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Determine if this is a gradient sync step
            is_sync_step = (accum_step + 1) % config.grad_accum_steps == 0

            # No-sync context for intermediate accum steps
            if not is_sync_step and hasattr(model, "no_sync"):
                ctx = model.no_sync()
            else:
                ctx = torch.no_grad.__class__.__new__(torch.no_grad.__class__)  # null ctx
                import contextlib
                ctx = contextlib.nullcontext()

            with ctx:
                with amp.autocast():
                    outputs = model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    loss = loss / config.grad_accum_steps

                amp.scale(loss).backward()

            accum_step += 1

            if is_sync_step:
                # Clip and step
                amp.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                ).item()
                amp.step(optimizer)
                amp.update()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                # EMA update
                if ema is not None:
                    ema.update(raw_model)

                global_step += 1
                batch_loss = loss.item() * config.grad_accum_steps
                epoch_loss += batch_loss
                n_batches += 1

                # Logging
                if is_main and global_step % config.log_every_n_steps == 0:
                    elapsed = time.perf_counter() - train_start
                    tokens_per_sec = (
                        global_step * config.batch_size * config.seq_len * world_size
                    ) / elapsed
                    current_lr = optimizer.param_groups[0]["lr"]
                    metrics = {
                        "train/loss": batch_loss,
                        "train/grad_norm": grad_norm,
                        "train/lr": current_lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/loss_scale": amp.loss_scale,
                    }
                    tracker.log_metrics(metrics, step=global_step)
                    logger.info(
                        f"Step {global_step}/{total_steps} | "
                        f"loss={batch_loss:.4f} | lr={current_lr:.2e} | "
                        f"grad_norm={grad_norm:.3f} | tok/s={tokens_per_sec:.0f}"
                    )

                # Checkpoint
                if global_step % config.save_every_n_steps == 0:
                    state.step = global_step
                    state.epoch = epoch
                    ckpt_mgr.save(
                        global_step, model, optimizer, scheduler, state, amp
                    )

        # Epoch end
        state.epoch = epoch
        epoch += 1

        # Validation
        if is_main:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = {
                        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in val_batch.items()
                    }
                    with amp.autocast():
                        val_out = model(**val_batch)
                        val_loss = val_out["loss"] if isinstance(val_out, dict) else val_out[0]
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            avg_train_loss = epoch_loss / max(1, n_batches)
            epoch_time = time.perf_counter() - epoch_start

            metrics = {
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": avg_val_loss,
                "epoch/duration_sec": epoch_time,
            }
            tracker.log_metrics(metrics, step=global_step)

            logger.info(
                f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | time={epoch_time:.1f}s"
            )

            # Early stopping check
            stop = early_stopper.on_epoch_end(epoch, {"val_loss": avg_val_loss})
            if stop:
                logger.info("Early stopping triggered.")
                break

            if avg_val_loss < state.best_val_loss:
                state.best_val_loss = avg_val_loss

    # ── Final checkpoint ─────────────────────────────────────────────────
    state.step = global_step
    ckpt_mgr.save(global_step, model, optimizer, scheduler, state, amp)

    if is_main:
        total_time = time.perf_counter() - train_start
        logger.info(f"Training complete in {total_time/3600:.2f}h | best_val_loss={state.best_val_loss:.4f}")
        tracker.log_metrics({
            "final/best_val_loss": state.best_val_loss,
            "final/total_steps": global_step,
            "final/total_time_hr": total_time / 3600,
        })
        tracker.end_run("FINISHED")

    cleanup_distributed(pgm)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> PretrainConfig:
    parser = argparse.ArgumentParser(description="Lumina distributed pretraining")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    # Load config from file
    config = PretrainConfig()
    if args.config:
        cfg_path = pathlib.Path(args.config)
        if cfg_path.exists():
            if cfg_path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    with open(cfg_path) as f:
                        cfg_dict = yaml.safe_load(f)
                    for k, v in cfg_dict.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
                except ImportError:
                    logger.warning("PyYAML not available; ignoring config file.")
            elif cfg_path.suffix == ".json":
                cfg_dict = json.loads(cfg_path.read_text())
                for k, v in cfg_dict.items():
                    if hasattr(config, k):
                        setattr(config, k, v)

    # Override with CLI args
    for attr in ["d_model", "n_layers", "n_heads", "lr", "batch_size", "n_steps",
                 "strategy", "checkpoint_dir", "resume_from", "experiment_name",
                 "run_name", "grad_accum_steps", "seed"]:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(config, attr, val)

    if args.no_amp:
        config.use_amp = False
    if args.gradient_checkpointing:
        config.gradient_checkpointing = True

    return config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = parse_args()

    # Set up logging before distributed init
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main(config)
