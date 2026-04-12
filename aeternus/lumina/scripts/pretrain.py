#!/usr/bin/env python3
"""
scripts/pretrain.py

CLI entry point for pretraining Lumina.

Usage:
    python scripts/pretrain.py --config configs/base_config.yaml
    python scripts/pretrain.py --config configs/small_config.yaml --resume checkpoints/step_5000
    python scripts/pretrain.py --d_model 512 --n_layers 12 --max_steps 100000
"""

import argparse
import os
import sys

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain Lumina Financial Foundation Model"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (base_config.yaml or small_config.yaml)"
    )
    # Model overrides
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_kv_heads", type=int, default=None)
    parser.add_argument("--unified_dim", type=int, default=None)
    parser.add_argument("--use_moe", action="store_true", default=None)
    parser.add_argument("--n_experts", type=int, default=None)
    parser.add_argument("--arch", type=str, default=None, choices=["causal", "bidirectional"])

    # Training overrides
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--n_train_samples", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--mixed_precision", action="store_true", default=None)
    parser.add_argument("--no_mixed_precision", action="store_true")

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="lumina-pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_pretrain_config(args, yaml_cfg: dict = None) -> "PretrainingConfig":
    from lumina.pretraining import PretrainingConfig

    if yaml_cfg is None:
        yaml_cfg = {}

    pt_cfg = yaml_cfg.get("pretraining", {})
    model_cfg = yaml_cfg.get("model", {})

    config = PretrainingConfig(
        n_train_samples=pt_cfg.get("n_train_samples", 50000),
        n_val_samples=pt_cfg.get("n_val_samples", 2000),
        seq_len=pt_cfg.get("seq_len", 256),
        n_workers=pt_cfg.get("n_workers", 4),
        d_model=model_cfg.get("d_model", 512),
        n_layers=model_cfg.get("n_layers", 12),
        n_heads=model_cfg.get("n_heads", 8),
        n_kv_heads=model_cfg.get("n_kv_heads", 2),
        unified_dim=model_cfg.get("unified_token_dim", 256),
        use_moe=model_cfg.get("use_moe", True),
        n_experts=model_cfg.get("n_experts", 8),
        arch=model_cfg.get("arch", "causal"),
        batch_size=pt_cfg.get("batch_size", 32),
        grad_accum_steps=pt_cfg.get("grad_accum_steps", 4),
        max_steps=pt_cfg.get("max_steps", 100000),
        warmup_steps=pt_cfg.get("warmup_steps", 2000),
        lr=pt_cfg.get("lr", 3e-4),
        weight_decay=pt_cfg.get("weight_decay", 0.1),
        max_grad_norm=pt_cfg.get("max_grad_norm", 1.0),
        mixed_precision=pt_cfg.get("mixed_precision", True),
        checkpoint_dir=pt_cfg.get("checkpoint_dir", "checkpoints/lumina_pretrain"),
        checkpoint_every_n_steps=pt_cfg.get("checkpoint_every_n_steps", 5000),
        eval_every_n_steps=pt_cfg.get("eval_every_n_steps", 1000),
        use_wandb=pt_cfg.get("use_wandb", False),
        wandb_project=pt_cfg.get("wandb_project", "lumina-pretrain"),
        wandb_run_name=pt_cfg.get("wandb_run_name", "run_01"),
        n_regimes=pt_cfg.get("n_regimes", 8),
    )

    # Apply CLI overrides
    if args.d_model is not None: config.d_model = args.d_model
    if args.n_layers is not None: config.n_layers = args.n_layers
    if args.n_heads is not None: config.n_heads = args.n_heads
    if args.n_kv_heads is not None: config.n_kv_heads = args.n_kv_heads
    if args.unified_dim is not None: config.unified_dim = args.unified_dim
    if args.use_moe is not None: config.use_moe = args.use_moe
    if args.n_experts is not None: config.n_experts = args.n_experts
    if args.arch is not None: config.arch = args.arch
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.lr is not None: config.lr = args.lr
    if args.max_steps is not None: config.max_steps = args.max_steps
    if args.warmup_steps is not None: config.warmup_steps = args.warmup_steps
    if args.seq_len is not None: config.seq_len = args.seq_len
    if args.n_train_samples is not None: config.n_train_samples = args.n_train_samples
    if args.grad_accum_steps is not None: config.grad_accum_steps = args.grad_accum_steps
    if args.no_mixed_precision: config.mixed_precision = False
    if args.mixed_precision: config.mixed_precision = True
    if args.checkpoint_dir is not None: config.checkpoint_dir = args.checkpoint_dir
    if args.wandb: config.use_wandb = True
    if args.wandb_project: config.wandb_project = args.wandb_project
    if args.wandb_run_name: config.wandb_run_name = args.wandb_run_name

    return config


def main():
    args = parse_args()

    print("=" * 60)
    print("  Lumina Foundation Model — Pretraining")
    print("=" * 60)

    # Load YAML config if provided
    yaml_cfg = {}
    if args.config is not None:
        print(f"Loading config from: {args.config}")
        yaml_cfg = load_config(args.config)

    # Build config
    from lumina.pretraining import pretrain, PretrainingConfig
    config = build_pretrain_config(args, yaml_cfg)

    print("\nPretraining configuration:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    print(f"\nDevice: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nStarting pretraining...\n")

    # Run pretraining
    model = pretrain(config, resume_from=args.resume)

    print("\nPretraining complete!")
    print(f"Model saved to: {config.checkpoint_dir}/final")


if __name__ == "__main__":
    main()
