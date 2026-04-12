#!/usr/bin/env python3
"""
scripts/evaluate.py

CLI for evaluating a pretrained or fine-tuned Lumina model.

Usage:
    python scripts/evaluate.py --model checkpoints/final --task all
    python scripts/evaluate.py --model checkpoints/final --task crisis_detection
    python scripts/evaluate.py --model checkpoints/final --task perplexity
    python scripts/evaluate.py --model checkpoints/final --task probing
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Lumina model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--task_heads", type=str, default=None,
                        help="Path to task head checkpoints")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "crisis_detection", "volatility_forecast",
                                 "return_direction", "regime_transfer", "perplexity",
                                 "attention", "probing"])
    parser.add_argument("--n_test_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="aeternus/lumina/results")
    parser.add_argument("--held_out_regime", type=int, default=7,
                        help="Regime index held out for zero-shot transfer test")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"  Lumina Evaluation: {args.task}")
    print("=" * 60)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Load model
    from lumina.transformer import LuminaModel
    from lumina.inference import LuminaInference
    from lumina.data_pipeline import SyntheticFinancialDataset, DataCollator, FinancialDataConfig
    from lumina.evaluation import (
        crisis_detection_benchmark,
        volatility_forecast_benchmark,
        return_direction_benchmark,
        zero_shot_regime_transfer,
        perplexity,
        attention_visualization,
        probing_analysis,
    )

    print(f"\nLoading model from: {args.model}")
    model = LuminaModel.from_pretrained(args.model, device=device_str)
    model = model.to(device)
    model.eval()
    print(f"Model parameters: {model.get_num_params():,}")

    # Build inference wrapper
    infer = LuminaInference(model, device=device_str)
    if args.task_heads:
        infer._load_task_heads(args.task_heads, model.config.d_model)

    # Build test dataset
    print(f"\nGenerating {args.n_test_samples} test samples (seq_len={args.seq_len})...")
    test_cfg = FinancialDataConfig(
        n_samples=args.n_test_samples,
        seq_len=args.seq_len,
        crisis_prob=0.15,
    )
    test_ds = SyntheticFinancialDataset(test_cfg, seed=999)
    collator = DataCollator(seq_len=args.seq_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    tasks_to_run = (
        ["crisis_detection", "volatility_forecast", "return_direction",
         "regime_transfer", "perplexity", "probing"]
        if args.task == "all"
        else [args.task]
    )

    for task in tasks_to_run:
        print(f"\n--- Running: {task} ---")
        try:
            if task == "crisis_detection":
                results = crisis_detection_benchmark(
                    infer, test_loader, device,
                    results_name="crisis_detection_benchmark"
                )
                all_results["crisis_detection"] = results
                print(f"  Lumina F1: {results['lumina']['f1']:.4f}")
                print(f"  VIX-threshold F1: {results['vix_threshold']['f1']:.4f}")
                print(f"  CUSUM F1: {results['cusum']['f1']:.4f}")

            elif task == "volatility_forecast":
                results = volatility_forecast_benchmark(
                    infer, test_loader, device, horizon=5,
                    results_name="volatility_forecast_benchmark"
                )
                all_results["volatility_forecast"] = results
                print(f"  Lumina RMSE: {results['lumina']['rmse']:.6f}")
                print(f"  GARCH RMSE: {results['garch_1_1']['rmse']:.6f}")

            elif task == "return_direction":
                results = return_direction_benchmark(
                    infer, test_loader, device,
                    results_name="return_direction_benchmark"
                )
                all_results["return_direction"] = results
                print(f"  Lumina accuracy: {results['lumina']['accuracy']:.4f}")
                print(f"  Momentum baseline: {results['momentum_baseline']['accuracy']:.4f}")

            elif task == "regime_transfer":
                results = zero_shot_regime_transfer(
                    infer, test_loader, device,
                    held_out_regime=args.held_out_regime,
                    results_name="zero_shot_regime_transfer"
                )
                all_results["regime_transfer"] = results
                if "overall_accuracy" in results:
                    print(f"  Overall accuracy: {results['overall_accuracy']:.4f}")
                print(f"  Held-out regime {args.held_out_regime} mean prob: "
                      f"{results['mean_prob_held_out_regime']:.4f}")

            elif task == "perplexity":
                ppl = perplexity(model, test_loader, device)
                all_results["perplexity"] = ppl
                print(f"  Perplexity: {ppl:.2f}")

            elif task == "attention":
                # Get one sample batch
                sample_batch = next(iter(test_loader))
                attn_map = attention_visualization(
                    model, sample_batch, device,
                    results_name="attention_maps"
                )
                all_results["attention"] = {"shape": list(attn_map.shape)}
                print(f"  Attention map shape: {attn_map.shape}")

            elif task == "probing":
                results = probing_analysis(
                    model, test_loader, device,
                    probe_tasks=["regime", "crisis"],
                    results_name="probing_analysis"
                )
                all_results["probing"] = results
                for probe_task, metrics in results.items():
                    print(f"  Probe '{probe_task}': {metrics}")

        except Exception as e:
            print(f"  ERROR in {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task] = {"error": str(e)}

    # Save combined results
    combined_path = f"{args.output_dir}/evaluation_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {combined_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
