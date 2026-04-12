#!/usr/bin/env python3
"""
scripts/train_lumina.py

Training script for Lumina Financial Foundation Model.

Usage:
    # Pre-training from scratch
    python scripts/train_lumina.py \
        --config configs/lumina_base.yaml \
        --mode pretrain \
        --data_dir ./data/crypto \
        --output_dir ./runs/lumina_base_pretrain

    # Fine-tuning for direction classification
    python scripts/train_lumina.py \
        --config configs/lumina_base.yaml \
        --mode finetune \
        --task direction \
        --pretrained_weights ./runs/lumina_base_pretrain/checkpoints/checkpoint_best.pt \
        --output_dir ./runs/lumina_direction_ft

    # Distributed pre-training (torchrun)
    torchrun --nproc_per_node=4 scripts/train_lumina.py \
        --config configs/lumina_large.yaml \
        --mode pretrain \
        --grad_accum 8

Author: SRFM-lab
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lumina.model import LuminaConfig, LuminaModel, ModelBuilder
from lumina.pretraining import (
    PretrainingConfig, MultiTaskPretrainingLoss, PretrainingTrainer,
    CosineWithWarmupScheduler,
)
from lumina.data_pipeline import (
    DataConfig, DataLoaderFactory, LuminaDataModule,
    generate_synthetic_ohlcv, FinancialDataset,
)
from lumina.distributed import (
    DistributedConfig, setup_distributed, cleanup_distributed,
    is_main_process, get_rank, get_world_size,
    DDPWrapper, AMPTrainer, CheckpointManager, TrainingState, ThroughputMonitor,
)
from lumina.evaluation import (
    return_direction_benchmark, volatility_forecast_benchmark,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_lumina")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Lumina Financial Foundation Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--mode",        type=str, default="pretrain",
                   choices=["pretrain", "finetune", "continue"],
                   help="Training mode")
    p.add_argument("--task",        type=str, default="direction",
                   choices=["direction", "volatility", "regime", "portfolio", "return"],
                   help="Fine-tuning task (finetune mode)")

    # Config and paths
    p.add_argument("--config",       type=str, default=None, help="YAML config file")
    p.add_argument("--data_dir",     type=str, default="./data", help="Data directory")
    p.add_argument("--output_dir",   type=str, default="./runs/lumina", help="Output directory")
    p.add_argument("--pretrained_weights", type=str, default=None,
                   help="Path to pretrained weights for fine-tuning")

    # Model
    p.add_argument("--model_size",   type=str, default="base",
                   choices=["tiny", "base", "large", "deep", "xl"],
                   help="Model size preset")
    p.add_argument("--d_model",      type=int,   default=None)
    p.add_argument("--n_layers",     type=int,   default=None)
    p.add_argument("--n_heads",      type=int,   default=None)
    p.add_argument("--patch_size",   type=int,   default=16)
    p.add_argument("--max_seq_len",  type=int,   default=2048)
    p.add_argument("--use_moe",      action="store_true")
    p.add_argument("--causal",       action="store_true")

    # Data
    p.add_argument("--assets",       type=str, nargs="+", default=["BTC"],
                   help="Asset symbols to train on")
    p.add_argument("--lookback",     type=int,   default=256)
    p.add_argument("--freq",         type=str,   default="1h")
    p.add_argument("--norm_mode",    type=str,   default="zscore")
    p.add_argument("--no_augmentation", action="store_true")
    p.add_argument("--synthetic",    action="store_true",
                   help="Use synthetic data (for testing)")

    # Training
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--max_steps",    type=int,   default=100000)
    p.add_argument("--max_epochs",   type=int,   default=None)
    p.add_argument("--warmup_steps", type=int,   default=1000)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--dropout",      type=float, default=0.1)

    # Pretraining objectives
    p.add_argument("--mask_ratio",   type=float, default=0.15)
    p.add_argument("--no_mrm",       action="store_true")
    p.add_argument("--no_npp",       action="store_true")
    p.add_argument("--no_contrastive", action="store_true")

    # AMP / distributed
    p.add_argument("--use_amp",      action="store_true", default=True)
    p.add_argument("--amp_dtype",    type=str,   default="bfloat16")
    p.add_argument("--no_amp",       action="store_true")
    p.add_argument("--local_rank",   type=int,   default=int(os.environ.get("LOCAL_RANK", 0)))
    p.add_argument("--grad_checkpoint", action="store_true")

    # Checkpointing
    p.add_argument("--save_every",   type=int,   default=5000)
    p.add_argument("--keep_n_ckpts", type=int,   default=3)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--eval_every",   type=int,   default=1000)

    # Logging
    p.add_argument("--log_every",    type=int,   default=50)
    p.add_argument("--wandb",        action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb_project", type=str,  default="lumina-srfm")
    p.add_argument("--run_name",     type=str,   default=None)

    # Misc
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--compile",      action="store_true", help="torch.compile the model")
    p.add_argument("--profile",      action="store_true")
    p.add_argument("--dry_run",      action="store_true", help="Run 1 step and exit")

    return p


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_lumina_config(args: argparse.Namespace) -> LuminaConfig:
    """Build LuminaConfig from args, optionally overriding with YAML config."""
    if args.config is not None:
        cfg_dict  = load_yaml_config(args.config)
        model_cfg = cfg_dict.get("model", {})
    else:
        model_cfg = {}

    # Presets
    size_presets = {
        "tiny":  {"d_model": 128, "n_heads": 4, "n_kv_heads": 2, "n_layers": 4,  "d_ff": 512},
        "base":  {"d_model": 512, "n_heads": 8, "n_kv_heads": 4, "n_layers": 12, "d_ff": 2048},
        "large": {"d_model": 1024,"n_heads":16, "n_kv_heads": 4, "n_layers": 24, "d_ff": 4096},
        "deep":  {"d_model": 1024,"n_heads":16, "n_kv_heads": 2, "n_layers": 48, "d_ff": 4096},
        "xl":    {"d_model": 2048,"n_heads":32, "n_kv_heads": 4, "n_layers": 32, "d_ff": 8192},
    }

    preset = size_presets.get(args.model_size, size_presets["base"])
    preset.update(model_cfg)

    # CLI overrides
    if args.d_model:   preset["d_model"]  = args.d_model
    if args.n_layers:  preset["n_layers"] = args.n_layers
    if args.n_heads:   preset["n_heads"]  = args.n_heads

    cfg = LuminaConfig(
        patch_size    = args.patch_size,
        max_seq_len   = args.max_seq_len,
        dropout       = args.dropout,
        use_moe       = args.use_moe,
        causal        = args.causal,
        gradient_checkpointing = args.grad_checkpoint,
        **preset,
    )
    return cfg


def build_data_config(args: argparse.Namespace) -> DataConfig:
    """Build DataConfig from args."""
    if args.config is not None:
        cfg_dict  = load_yaml_config(args.config)
        data_cfg  = cfg_dict.get("data", {})
    else:
        data_cfg = {}

    return DataConfig(
        assets        = args.assets,
        lookback      = args.lookback,
        freq          = args.freq,
        norm_mode     = args.norm_mode,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        use_augmentation = not args.no_augmentation,
        **{k: v for k, v in data_cfg.items()
           if k not in ["assets", "lookback", "freq", "norm_mode", "batch_size"]},
    )


def build_pretrain_config(args: argparse.Namespace) -> PretrainingConfig:
    """Build PretrainingConfig from args."""
    if args.config is not None:
        cfg_dict     = load_yaml_config(args.config)
        pretrain_cfg = cfg_dict.get("pretraining", {})
    else:
        pretrain_cfg = {}

    return PretrainingConfig(
        mask_ratio        = args.mask_ratio,
        use_mrm           = not args.no_mrm,
        use_npp           = not args.no_npp,
        use_contrastive   = not args.no_contrastive,
        max_steps         = args.max_steps,
        lr                = args.lr,
        weight_decay      = args.weight_decay,
        warmup_steps      = args.warmup_steps,
        log_every         = args.log_every,
        eval_every        = args.eval_every,
        save_every        = args.save_every,
        **{k: v for k, v in pretrain_cfg.items()
           if k not in ["mask_ratio", "max_steps", "lr"]},
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args: argparse.Namespace, data_cfg: DataConfig) -> Dict[str, np.ndarray]:
    """Load OHLCV data from files or generate synthetic data."""
    if args.synthetic:
        logger.info("Using synthetic data")
        return {"main": generate_synthetic_ohlcv(T=50000)}

    data = {}
    data_dir = Path(args.data_dir)

    for asset in args.assets:
        # Try common file formats
        for ext in [".npy", ".npz", ".csv"]:
            path = data_dir / f"{asset}_{args.freq}{ext}"
            if path.exists():
                if ext == ".npy":
                    arr = np.load(path)
                elif ext == ".npz":
                    npz = np.load(path)
                    arr = npz["ohlcv"] if "ohlcv" in npz else npz[list(npz.keys())[0]]
                else:  # CSV
                    import pandas as pd
                    df  = pd.read_csv(path)
                    arr = df[["open", "high", "low", "close", "volume"]].values
                data[asset] = arr.astype(np.float32)
                logger.info(f"Loaded {asset}: {arr.shape}")
                break
        else:
            logger.warning(f"Data for {asset} not found in {data_dir}, using synthetic")
            data[asset] = generate_synthetic_ohlcv(T=50000)

    if not data:
        logger.warning("No data loaded, using synthetic")
        data["main"] = generate_synthetic_ohlcv(T=50000)

    # Use first asset as primary if "main" not set
    if "main" not in data:
        first_key   = list(data.keys())[0]
        data["main"] = data[first_key]

    return data


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(
    lumina_cfg: LuminaConfig,
    pretrained_weights: Optional[str] = None,
) -> LuminaModel:
    model = LuminaModel(lumina_cfg)

    if pretrained_weights is not None and Path(pretrained_weights).exists():
        ckpt = torch.load(pretrained_weights, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained weights from {pretrained_weights}")
        if missing:
            logger.info(f"  Missing keys: {len(missing)}")
        if unexpected:
            logger.info(f"  Unexpected keys: {len(unexpected)}")
    else:
        logger.info("Training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    return model


# ---------------------------------------------------------------------------
# Pre-training loop
# ---------------------------------------------------------------------------

def pretrain(args: argparse.Namespace) -> None:
    """Main pre-training loop."""
    logger.info("Starting pre-training")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Build configs
    lumina_cfg  = build_lumina_config(args)
    data_cfg    = build_data_config(args)
    pretrain_cfg = build_pretrain_config(args)

    # Load data
    data = load_data(args, data_cfg)

    # Build loaders
    train_loader, val_loader, _ = DataLoaderFactory.build_all(
        data["main"], data_cfg, task="pretrain"
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    model = build_model(lumina_cfg, args.pretrained_weights).to(device)

    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap with DDP if distributed
    if get_world_size() > 1:
        model = DDPWrapper(model, device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pretrain_cfg.lr,
        weight_decay=pretrain_cfg.weight_decay,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )

    # Scheduler
    scheduler = CosineWithWarmupScheduler(
        optimizer,
        warmup_steps=pretrain_cfg.warmup_steps,
        max_steps=pretrain_cfg.max_steps,
    )

    # Loss module
    loss_module = MultiTaskPretrainingLoss(
        pretrain_cfg,
        patch_size = lumina_cfg.patch_size,
        n_channels = lumina_cfg.n_channels,
        d_model    = lumina_cfg.d_model,
    ).to(device)

    # Checkpoint manager
    output_dir = Path(args.output_dir)
    ckpt_mgr   = CheckpointManager(
        output_dir / "checkpoints",
        keep_n        = args.keep_n_ckpts,
        save_every_n  = args.save_every,
    )

    # AMP trainer
    dist_cfg = DistributedConfig(
        use_amp        = args.use_amp and not args.no_amp,
        amp_dtype      = args.amp_dtype,
        grad_clip      = args.grad_clip,
        grad_accum_steps = args.grad_accum,
        checkpoint_dir = str(output_dir / "checkpoints"),
    )

    amp_trainer = AMPTrainer(model, optimizer, dist_cfg, device)
    state       = TrainingState()
    throughput  = ThroughputMonitor()

    # Resume
    if args.resume is not None:
        state = ckpt_mgr.load(args.resume, model, optimizer)
        logger.info(f"Resumed from step {state.step}")

    # W&B
    wandb_run = None
    if args.wandb and is_main_process():
        try:
            import wandb
            run_name   = args.run_name or f"lumina-pretrain-{args.model_size}"
            wandb_run  = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={"lumina": lumina_cfg.to_dict(), "pretrain": asdict_safe(pretrain_cfg)},
            )
        except ImportError:
            logger.warning("wandb not installed, skipping")

    # Training loop
    logger.info("Starting training loop...")
    model.train()
    train_iter  = iter(train_loader)
    step        = state.step
    t0          = time.perf_counter()

    while step < pretrain_cfg.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)
            state.epoch += 1

        ohlcv = batch["ohlcv"].to(device)
        B, T, C = ohlcv.shape

        def loss_fn():
            enc      = model(ohlcv)
            hidden   = enc["hidden"]
            cls_emb  = enc["cls_emb"]
            aux_loss = enc["aux_loss"]

            B_l, N, D = hidden.shape
            # Simple patch targets (zeros for now, real impl would use tokenizer targets)
            targets    = torch.zeros(B_l, N, lumina_cfg.patch_size * lumina_cfg.n_channels, device=device)
            mask       = torch.rand(B_l, N, device=device) < pretrain_cfg.mask_ratio

            z1 = cls_emb
            z2 = cls_emb + 0.01 * torch.randn_like(cls_emb)

            loss, metrics = loss_module(hidden, cls_emb, targets, mask, contrastive_z=(z1, z2))
            return loss + 0.01 * aux_loss

        loss_val, grad_norm = amp_trainer.train_step(loss_fn)
        scheduler.step()

        throughput.update(B, B * T)
        step    += 1
        state.step = step

        if step % args.log_every == 0 and is_main_process():
            tput   = throughput.get_throughput()
            lr     = scheduler.get_last_lr()[0]
            t1     = time.perf_counter()
            dt     = t1 - t0
            t0     = t1

            log_str = (
                f"Step {step:6d}/{pretrain_cfg.max_steps} | "
                f"Loss: {loss_val:.4f} | "
                f"LR: {lr:.2e} | "
                f"Grad: {grad_norm:.2f} | "
                f"Tput: {tput['samples_per_sec']:.0f} samp/s | "
                f"dt: {dt:.2f}s"
            )
            logger.info(log_str)

            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": loss_val,
                    "train/lr":   lr,
                    "train/grad_norm": grad_norm or 0.0,
                    "train/samples_per_sec": tput["samples_per_sec"],
                    "step": step,
                })

        # Evaluation
        if step % args.eval_every == 0 and is_main_process():
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_ohlcv = val_batch["ohlcv"].to(device)
                    enc       = model(val_ohlcv)
                    val_losses.append(enc["aux_loss"].item())
            avg_val = np.mean(val_losses)
            logger.info(f"[Eval] Step {step}: val_aux_loss={avg_val:.4f}")
            model.train()

        # Checkpoint
        if step % args.save_every == 0 and is_main_process():
            m = model.unwrap() if isinstance(model, DDPWrapper) else model
            ckpt_mgr.save(m, optimizer, state, amp_trainer.scaler, scheduler)

        if args.dry_run:
            logger.info("Dry run complete, exiting")
            break

    logger.info("Pre-training complete!")

    # Final save
    if is_main_process():
        m = model.unwrap() if isinstance(model, DDPWrapper) else model
        ckpt_mgr.save(m, optimizer, state, amp_trainer.scaler, scheduler, tag="final")
        m.save_pretrained(str(output_dir / "final_model"))

    if wandb_run is not None:
        wandb_run.finish()

    cleanup_distributed()


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(args: argparse.Namespace) -> None:
    """Fine-tuning loop."""
    from lumina.finetuning import (
        FineTuningConfig, FineTuner,
        DirectionClassificationHead, VolatilityForecastingHead,
        FinancialFineTuningDataset,
    )
    from torch.utils.data import DataLoader as TorchLoader

    logger.info(f"Starting fine-tuning: task={args.task}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cpu")

    # Build configs
    lumina_cfg = build_lumina_config(args)
    data_cfg   = build_data_config(args)

    ft_cfg = FineTuningConfig(
        task           = args.task,
        d_model        = lumina_cfg.d_model,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        max_epochs     = args.max_epochs or 20,
        batch_size     = args.batch_size,
        dropout        = args.dropout,
        freeze_backbone = True,
    )

    # Load data
    raw_data = load_data(args, data_cfg)
    ohlcv    = raw_data["main"]

    # Build model
    backbone = build_model(lumina_cfg, args.pretrained_weights).to(device)

    # Task head
    if args.task in ["direction", "regime"]:
        n_classes = 3 if args.task == "direction" else 4
        head      = DirectionClassificationHead(lumina_cfg.d_model, n_classes).to(device)
    else:
        head = VolatilityForecastingHead(lumina_cfg.d_model).to(device)

    # Trainer
    finetuner = FineTuner(backbone, head, ft_cfg, device)

    # Datasets
    train_ds = FinancialFineTuningDataset(ohlcv, ft_cfg, split="train")
    val_ds   = FinancialFineTuningDataset(ohlcv, ft_cfg, split="val")

    train_loader = TorchLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader   = TorchLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    ckpt_mgr   = CheckpointManager(
        output_dir / "checkpoints",
        keep_n       = args.keep_n_ckpts,
        metric_mode  = "max" if args.task in ["direction", "regime"] else "min",
    )

    best_metric = 0.0

    for epoch in range(ft_cfg.max_epochs):
        train_metrics = finetuner.train_epoch(train_loader)
        val_metrics   = finetuner.evaluate(val_loader)

        if is_main_process():
            logger.info(
                f"Epoch {epoch+1:3d}/{ft_cfg.max_epochs} | "
                f"Train loss: {train_metrics['train_loss']:.4f} | "
                f"Val: {json.dumps({k: f'{v:.4f}' for k, v in val_metrics.items()})}"
            )

        # Early stopping check
        if finetuner.check_early_stopping(val_metrics):
            logger.info("Early stopping triggered")
            break

        # Save checkpoint
        metric_key = f"val_{ft_cfg.eval_metric}"
        cur_metric = val_metrics.get(metric_key, 0.0)
        state      = TrainingState(step=epoch)
        ckpt_mgr.save(backbone, finetuner.optimizer, state, metric=cur_metric)

        if args.dry_run:
            break

    logger.info("Fine-tuning complete!")

    if is_main_process():
        best_state = ckpt_mgr.load_best(backbone)
        if best_state:
            logger.info(f"Loaded best model (step={best_state.step})")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def asdict_safe(obj) -> Dict:
    try:
        from dataclasses import asdict
        return asdict(obj)
    except Exception:
        return {}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_system_info() -> None:
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log system info
    if is_main_process():
        log_system_info()
        logger.info(f"Arguments: {vars(args)}")

    # Set seed
    set_seed(args.seed)

    # Setup distributed
    dist_cfg = DistributedConfig(
        local_rank  = args.local_rank,
        world_size  = int(os.environ.get("WORLD_SIZE", 1)),
        rank        = int(os.environ.get("RANK", 0)),
        use_amp     = args.use_amp and not args.no_amp,
        amp_dtype   = args.amp_dtype,
        grad_clip   = args.grad_clip,
        grad_accum_steps = args.grad_accum,
    )
    setup_distributed(dist_cfg)

    try:
        if args.mode == "pretrain":
            pretrain(args)
        elif args.mode == "finetune":
            finetune(args)
        elif args.mode == "continue":
            if args.resume is None:
                # Find latest checkpoint
                ckpt_dir = Path(args.output_dir) / "checkpoints"
                ckpts    = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
                if ckpts:
                    args.resume = str(ckpts[-1])
            pretrain(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
