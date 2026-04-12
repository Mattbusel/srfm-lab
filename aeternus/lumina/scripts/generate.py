#!/usr/bin/env python3
"""
scripts/generate.py

CLI for generating synthetic financial scenarios with Lumina.

Usage:
    python scripts/generate.py --model checkpoints/final --n_paths 100 --steps 200
    python scripts/generate.py --model checkpoints/final --scenario crisis --n_paths 50
    python scripts/generate.py --model checkpoints/final --export_onnx outputs/lumina.onnx
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic scenarios with Lumina")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--n_paths", type=int, default=10,
                        help="Number of paths to generate")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of generation steps (tokens)")
    parser.add_argument("--prompt_len", type=int, default=64,
                        help="Length of prompt (conditioning window)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--scenario", type=str, default="normal",
                        choices=["normal", "crisis", "bull", "bear"],
                        help="Scenario type for prompt conditioning")
    parser.add_argument("--output_dir", type=str, default="aeternus/lumina/outputs")
    parser.add_argument("--export_onnx", type=str, default=None,
                        help="If set, export model to ONNX at this path")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def generate_scenario_prompt(
    scenario: str,
    prompt_len: int,
    d_model: int,
    n_paths: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a conditioning prompt embedding based on scenario type.

    Returns:
        prompt: (n_paths, prompt_len, d_model)
    """
    prompt = torch.randn(n_paths, prompt_len, d_model, device=device) * 0.1

    if scenario == "crisis":
        # High variance, trending downward
        t = torch.linspace(0, 1, prompt_len, device=device)
        trend = -2.0 * t.unsqueeze(0).unsqueeze(-1)
        vol = torch.randn(n_paths, prompt_len, d_model, device=device) * 0.5
        prompt = prompt + trend + vol

    elif scenario == "bull":
        # Low variance, trending upward
        t = torch.linspace(0, 1, prompt_len, device=device)
        trend = 1.0 * t.unsqueeze(0).unsqueeze(-1)
        vol = torch.randn(n_paths, prompt_len, d_model, device=device) * 0.05
        prompt = prompt + trend + vol

    elif scenario == "bear":
        # Low-medium variance, trending downward
        t = torch.linspace(0, 1, prompt_len, device=device)
        trend = -0.5 * t.unsqueeze(0).unsqueeze(-1)
        vol = torch.randn(n_paths, prompt_len, d_model, device=device) * 0.1
        prompt = prompt + trend + vol

    # Normal: standard noise (already set above)
    return prompt


def main():
    args = parse_args()

    print("=" * 60)
    print("  Lumina — Synthetic Scenario Generation")
    print("=" * 60)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Load model
    from lumina.transformer import LuminaModel
    from lumina.inference import LuminaInference

    print(f"\nLoading model from: {args.model}")
    model = LuminaModel.from_pretrained(args.model, device=device_str)
    d_model = model.config.unified_token_dim
    infer = LuminaInference(model, device=device_str)

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Unified token dim: {d_model}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ONNX export (if requested)
    if args.export_onnx:
        print(f"\nExporting to ONNX: {args.export_onnx}")
        try:
            infer.export_onnx(
                args.export_onnx,
                seq_len=args.prompt_len + args.steps,
                d_model=d_model,
                batch_size=1,
            )
        except Exception as e:
            print(f"ONNX export failed: {e}")

    # Generate paths
    print(f"\nGenerating {args.n_paths} paths of length {args.steps} "
          f"under scenario: '{args.scenario}'")
    print(f"Prompt length: {args.prompt_len}, temperature: {args.temperature}, "
          f"top_k: {args.top_k}, top_p: {args.top_p}")

    prompt = generate_scenario_prompt(
        args.scenario, args.prompt_len, d_model, args.n_paths, device
    )
    prompt = prompt.to(torch.float32)

    print("\nRunning generation...")
    with torch.no_grad():
        generated = infer.generate_return_sequence(
            prompt_embeddings=prompt,
            n_steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    print(f"Generated sequence shape: {generated.shape}")  # (B, T_prompt + steps, D)

    # Post-process: extract the generated suffix and compute summary stats
    gen_suffix = generated[:, args.prompt_len:, :]  # (B, steps, D)

    # Compute proxy return series (first dimension of embedding as proxy)
    proxy_returns = gen_suffix[:, :, 0].cpu().numpy()  # (B, steps)

    # Statistics
    stats = {
        "scenario": args.scenario,
        "n_paths": args.n_paths,
        "steps": args.steps,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "mean_return": float(proxy_returns.mean()),
        "std_return": float(proxy_returns.std()),
        "min_return": float(proxy_returns.min()),
        "max_return": float(proxy_returns.max()),
        "cumulative_return_mean": float(proxy_returns.cumsum(axis=1)[:, -1].mean()),
    }

    print("\nGeneration statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save outputs
    np.save(f"{args.output_dir}/generated_embeddings_{args.scenario}.npy",
            generated.cpu().numpy())
    np.save(f"{args.output_dir}/proxy_returns_{args.scenario}.npy", proxy_returns)

    with open(f"{args.output_dir}/generation_stats_{args.scenario}.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Compute crisis probability for each generated path
    print("\nComputing crisis probabilities for generated paths...")
    with torch.no_grad():
        crisis_probs = infer.crisis_score(
            generated[:, args.prompt_len:, :].to(device)
        ).cpu().numpy()

    print(f"  Mean crisis probability: {crisis_probs.mean():.4f}")
    print(f"  Max crisis probability: {crisis_probs.max():.4f}")
    print(f"  Fraction with P(crisis) > 0.5: {(crisis_probs > 0.5).mean():.4f}")

    np.save(f"{args.output_dir}/crisis_probs_{args.scenario}.npy", crisis_probs)

    print(f"\nOutputs saved to: {args.output_dir}")
    print("Generation complete!")


if __name__ == "__main__":
    main()
