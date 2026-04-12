#!/usr/bin/env python3
"""
scripts/finetune.py

CLI for fine-tuning Lumina on a specific task.

Usage:
    python scripts/finetune.py --pretrained checkpoints/lumina_pretrain/final \
        --task crisis_detection --config configs/base_config.yaml
    python scripts/finetune.py --pretrained checkpoints/final \
        --task volatility_forecast --use_lora --lora_rank 16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Lumina on a downstream task")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path to pretrained model checkpoint directory")
    parser.add_argument("--task", type=str, required=True,
                        choices=["crisis_detection", "volatility_forecast",
                                 "regime_classification", "return_direction"],
                        help="Fine-tuning task")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lumina_finetune")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--n_train_samples", type=int, default=5000)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"  Lumina Fine-Tuning: {args.task}")
    print("=" * 60)

    yaml_cfg = {}
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)

    ft_yaml = yaml_cfg.get("finetuning", {})
    model_cfg = yaml_cfg.get("model", {})

    from lumina.finetuning import FineTuningConfig, FineTuner
    from lumina.data_pipeline import SyntheticFinancialDataset, DataCollator, FinancialDataConfig

    seq_len = args.seq_len or ft_yaml.get("seq_len", 256) or model_cfg.get("max_seq_len", 256)
    use_lora = True
    if args.no_lora:
        use_lora = False
    elif args.use_lora:
        use_lora = True
    else:
        use_lora = ft_yaml.get("use_lora", True)

    config = FineTuningConfig(
        task=args.task,
        pretrained_path=args.pretrained,
        output_dir=f"{args.output_dir}/{args.task}",
        n_epochs=args.n_epochs or ft_yaml.get("n_epochs", 10),
        batch_size=args.batch_size or ft_yaml.get("batch_size", 32),
        lr=args.lr or ft_yaml.get("lr", 1e-4),
        pretrained_lr_factor=ft_yaml.get("pretrained_lr_factor", 0.1),
        weight_decay=ft_yaml.get("weight_decay", 0.01),
        max_grad_norm=ft_yaml.get("max_grad_norm", 1.0),
        warmup_steps=ft_yaml.get("warmup_steps", 100),
        mixed_precision=args.mixed_precision or ft_yaml.get("mixed_precision", False),
        use_lora=use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=ft_yaml.get("lora_dropout", 0.05),
        use_wandb=args.wandb or ft_yaml.get("use_wandb", False),
        n_regimes=ft_yaml.get("n_regimes", 8),
        forecast_horizon=ft_yaml.get("forecast_horizon", 5),
        seq_len=seq_len,
        unified_dim=model_cfg.get("unified_token_dim", 256),
    )

    print("\nFine-tuning configuration:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    # Build datasets
    data_cfg_train = FinancialDataConfig(
        n_samples=args.n_train_samples,
        seq_len=seq_len,
        crisis_prob=0.2 if args.task == "crisis_detection" else 0.05,
    )
    data_cfg_val = FinancialDataConfig(
        n_samples=max(100, args.n_train_samples // 5),
        seq_len=seq_len,
        crisis_prob=0.2 if args.task == "crisis_detection" else 0.05,
    )

    print(f"\nGenerating {args.n_train_samples} training samples...")
    train_ds = SyntheticFinancialDataset(data_cfg_train)
    val_ds = SyntheticFinancialDataset(data_cfg_val)

    # Add task labels
    def add_labels(ds):
        """Attach task-specific labels to dataset samples."""
        original_getitem = ds.__getitem__

        def getitem_with_labels(idx):
            item = original_getitem(idx)
            if args.task == "crisis_detection":
                item["labels"] = item["is_crisis"]
            elif args.task == "regime_classification":
                item["labels"] = item["regime"]
            elif args.task == "volatility_forecast":
                vol_proxy = item["price_tokens"].std(dim=-1).mean()
                item["labels"] = vol_proxy.unsqueeze(0).expand(config.forecast_horizon)
            elif args.task == "return_direction":
                last_return = item["price_tokens"][-1, 0]
                if last_return > 0.001:
                    label = 2
                elif last_return < -0.001:
                    label = 0
                else:
                    label = 1
                item["labels"] = torch.tensor(label, dtype=torch.long)
            return item

        import types
        ds.__getitem__ = types.MethodType(lambda self, idx: getitem_with_labels(idx), ds)
        return ds

    import types, torch
    for ds in [train_ds, val_ds]:
        orig = ds.__getitem__.__func__ if hasattr(ds.__getitem__, '__func__') else None
        task_name = args.task
        forecast_horizon = config.forecast_horizon

        def make_labeled_getitem(orig_fn, task, horizon):
            def labeled_getitem(self, idx):
                item = orig_fn(self, idx)
                if task == "crisis_detection":
                    item["labels"] = item["is_crisis"]
                elif task == "regime_classification":
                    item["labels"] = item["regime"]
                elif task == "volatility_forecast":
                    vol = item["price_tokens"].std(dim=-1).mean()
                    item["labels"] = vol.unsqueeze(0).expand(horizon).clone()
                elif task == "return_direction":
                    ret = item["price_tokens"][-1, 0].item() if item["price_tokens"].shape[-1] > 0 else 0.0
                    if ret > 0.001:
                        label = 2
                    elif ret < -0.001:
                        label = 0
                    else:
                        label = 1
                    item["labels"] = torch.tensor(label, dtype=torch.long)
                return item
            return labeled_getitem

        # Patch each dataset
        original_fn = SyntheticFinancialDataset.__getitem__
        ds.__class__ = type(
            "LabeledFinancialDataset",
            (SyntheticFinancialDataset,),
            {"__getitem__": make_labeled_getitem(original_fn, task_name, forecast_horizon)},
        )

    collator = DataCollator(seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

    print(f"\nTraining samples: {len(train_ds)}, Validation: {len(val_ds)}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    # Fine-tune
    finetuner = FineTuner(config)
    history = finetuner.fit(train_loader, val_loader)

    print("\nFine-tuning complete!")
    print(f"Final model saved to: {config.output_dir}/final")
    if history["train_loss"]:
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
