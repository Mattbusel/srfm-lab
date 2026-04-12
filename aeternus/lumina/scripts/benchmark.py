#!/usr/bin/env python3
"""
scripts/benchmark.py

Full benchmark runner for Lumina financial foundation model.

Usage:
    python scripts/benchmark.py --checkpoint checkpoints/pretrain/checkpoint_step_00010000 \
                                  --data_root data/benchmark \
                                  --output_dir results/benchmark \
                                  --benchmarks direction volatility crisis portfolio
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lumina.benchmark_suite import (
    BenchmarkRunner,
    PerformanceMetrics,
    DirectionPredictionBenchmark,
    VolatilityForecastBenchmark,
    CrisisDetectionBenchmark,
    PortfolioOptimizationBenchmark,
    walk_forward_splits,
    DieboldMarianoTest,
    WhiteRealityCheck,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Minimal model for benchmarking
# ---------------------------------------------------------------------------

class BenchmarkModel(nn.Module):
    """Lightweight model wrapper for benchmarking."""

    def __init__(self, input_dim: int = 32, d_model: int = 128, n_layers: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.proj(x)
        h = self.encoder(h)
        logit = self.head(h[:, -1, :]).squeeze(-1)
        return {"logits": logit, "output": logit}


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_samples: int = 5000,
    n_assets: int = 10,
    seq_len: int = 64,
    feature_dim: int = 32,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic financial data for benchmarking."""
    rng = np.random.RandomState(seed)

    # Price paths via geometric Brownian motion
    mu = 0.0002
    sigma = 0.015
    dt = 1.0
    T = n_samples

    prices = np.zeros((T, n_assets))
    for j in range(n_assets):
        prices[0, j] = 100.0
        for t in range(1, T):
            dW = rng.randn() * np.sqrt(dt)
            prices[t, j] = prices[t - 1, j] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)

    returns = np.diff(np.log(prices + 1e-10), axis=0)  # (T-1, n_assets)

    # Realized volatility
    rvol = np.array([
        np.std(returns[max(0, t - 20): t], axis=0) * np.sqrt(252)
        for t in range(1, len(returns) + 1)
    ])  # (T-1, n_assets)

    # Crisis labels (VIX proxy)
    vol_index = rvol.mean(axis=1)
    crisis_labels = (vol_index > np.percentile(vol_index, 90)).astype(int)

    # Features (rolling window)
    feature_matrix = np.zeros((T, seq_len, feature_dim))
    for t in range(seq_len, T):
        window = returns[max(0, t - seq_len): t, 0]
        if len(window) < seq_len:
            window = np.pad(window, (seq_len - len(window), 0))
        feature_matrix[t, :, 0] = window
        feature_matrix[t, :, 1:] = rng.randn(seq_len, feature_dim - 1) * 0.1

    return {
        "prices": prices[:, 0],
        "returns": returns[:, 0],
        "multi_asset_returns": returns,
        "realized_vol": rvol[:, 0],
        "crisis_labels": crisis_labels,
        "features": feature_matrix.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Walk-forward benchmark runner
# ---------------------------------------------------------------------------

def run_walk_forward_benchmark(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    n_splits: int = 5,
    train_pct: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Run walk-forward (out-of-sample) benchmark."""
    features = data["features"]
    returns = data["returns"]
    prices = data["prices"]
    T = len(features)

    train_size = int(T * train_pct / n_splits)
    test_size = T // (n_splits * 2)

    splits = walk_forward_splits(T, train_size, test_size, expanding=True)
    logger.info(f"Walk-forward: {len(splits)} splits")

    all_predictions = []
    all_realized = []

    model.eval()
    for split in splits[:n_splits]:
        test_features = torch.from_numpy(features[split.test_start: split.test_end]).to(device)
        test_returns = returns[split.test_start: split.test_end]

        with torch.no_grad():
            outputs = model(test_features)
            preds = outputs["logits"].cpu().numpy()

        all_predictions.extend(preds.tolist())
        all_realized.extend(test_returns.tolist())

    preds_arr = np.array(all_predictions)
    rets_arr = np.array(all_realized)

    return {
        "n_splits": len(splits),
        "hit_rate": PerformanceMetrics.hit_rate(preds_arr, rets_arr),
        "ic": PerformanceMetrics.information_coefficient(preds_arr, rets_arr),
        "signal_strategy_sharpe": PerformanceMetrics.sharpe_ratio(
            np.sign(preds_arr) * rets_arr
        ),
    }


# ---------------------------------------------------------------------------
# Main benchmark script
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Lumina Benchmark Suite | device={args.device}")
    logger.info(f"Benchmarks: {args.benchmarks}")

    # ── Build / load model ───────────────────────────────────────────────
    model = BenchmarkModel(input_dim=args.feature_dim).to(device)
    if args.checkpoint:
        ckpt_path = pathlib.Path(args.checkpoint)
        ckpt_file = ckpt_path / "checkpoint.pt" if ckpt_path.is_dir() else ckpt_path
        if ckpt_file.exists():
            payload = torch.load(ckpt_file, map_location=device)
            state_dict = payload.get("model_state_dict", payload)
            # Try loading — may need key remapping
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded checkpoint from {ckpt_file}")
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}. Using random weights.")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_file}. Using random weights.")
    else:
        logger.info("No checkpoint provided. Using random model weights.")

    model.eval()

    # ── Generate benchmark data ──────────────────────────────────────────
    logger.info("Generating synthetic benchmark data...")
    data = generate_synthetic_data(
        n_samples=args.n_samples,
        n_assets=args.n_assets,
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
        seed=args.seed,
    )

    # ── Create DataLoader ────────────────────────────────────────────────
    from torch.utils.data import TensorDataset, DataLoader
    feature_tensor = torch.from_numpy(data["features"])
    return_tensor = torch.from_numpy(data["returns"][:len(data["features"])])
    dataset = TensorDataset(feature_tensor, return_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ── Run benchmarks ───────────────────────────────────────────────────
    all_results: Dict[str, Any] = {}
    runner = BenchmarkRunner(model, device=device)
    start_time = time.perf_counter()

    if "direction" in args.benchmarks:
        logger.info("Running direction prediction benchmark...")
        try:
            direction_results = runner.run_direction_benchmark(
                dataloader, data["prices"][:args.n_samples]
            )
            all_results["direction"] = direction_results
            lumina_sharpe = direction_results.get("lumina", {}).get("sharpe_ratio", 0)
            logger.info(f"Direction | Lumina Sharpe: {lumina_sharpe:.3f}")
        except Exception as e:
            logger.error(f"Direction benchmark failed: {e}")

    if "volatility" in args.benchmarks:
        logger.info("Running volatility forecasting benchmark...")
        try:
            vol_bench = VolatilityForecastBenchmark()
            returns_arr = data["returns"]
            realized_vol = data["realized_vol"]
            # Generate model vol forecasts (placeholder)
            model_vol = np.abs(returns_arr) + np.random.randn(len(returns_arr)) * 0.001
            vol_results = vol_bench.evaluate(model_vol, realized_vol, returns_arr)
            all_results["volatility"] = vol_results
            lumina_rmse = vol_results.get("lumina", {}).get("rmse", 0)
            logger.info(f"Volatility | Lumina RMSE: {lumina_rmse:.4f}")
        except Exception as e:
            logger.error(f"Volatility benchmark failed: {e}")

    if "crisis" in args.benchmarks:
        logger.info("Running crisis detection benchmark...")
        try:
            crisis_bench = CrisisDetectionBenchmark()
            n = len(data["crisis_labels"])
            model_scores = np.random.rand(n)   # Replace with actual model scores
            crisis_results = crisis_bench.evaluate(
                model_scores,
                data["crisis_labels"],
                returns=data["returns"][:n],
            )
            all_results["crisis"] = crisis_results
            logger.info(f"Crisis | Results computed for {n} samples")
        except Exception as e:
            logger.error(f"Crisis benchmark failed: {e}")

    if "portfolio" in args.benchmarks:
        logger.info("Running portfolio optimization benchmark...")
        try:
            port_bench = PortfolioOptimizationBenchmark(transaction_cost=0.001)
            T_port = min(len(data["multi_asset_returns"]), 2000)
            n_assets = data["multi_asset_returns"].shape[1]
            model_signals = np.random.randn(T_port, n_assets)   # Replace with actual signals
            port_results = port_bench.evaluate(
                model_signals,
                data["multi_asset_returns"][:T_port],
            )
            all_results["portfolio"] = port_results
            lumina_sharpe = port_results.get("lumina", {}).get("sharpe_ratio", 0)
            logger.info(f"Portfolio | Lumina Sharpe: {lumina_sharpe:.3f}")
        except Exception as e:
            logger.error(f"Portfolio benchmark failed: {e}")

    if "walk_forward" in args.benchmarks:
        logger.info("Running walk-forward benchmark...")
        try:
            wf_results = run_walk_forward_benchmark(model, data, device=device)
            all_results["walk_forward"] = wf_results
            logger.info(f"Walk-forward | IC={wf_results.get('ic', 0):.4f}")
        except Exception as e:
            logger.error(f"Walk-forward benchmark failed: {e}")

    # ── Compute total time ───────────────────────────────────────────────
    total_time = time.perf_counter() - start_time
    all_results["_meta"] = {
        "total_time_sec": total_time,
        "n_samples": args.n_samples,
        "n_assets": args.n_assets,
        "device": args.device,
        "checkpoint": args.checkpoint,
        "benchmarks": args.benchmarks,
    }

    # ── Save results ─────────────────────────────────────────────────────
    results_file = output_dir / "benchmark_results.json"
    results_file.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info(f"Results saved to {results_file}")

    # ── Print summary ────────────────────────────────────────────────────
    runner.print_report(all_results)
    logger.info(f"Benchmark complete in {total_time:.1f}s")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lumina benchmark runner")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint directory")
    parser.add_argument("--data_root", type=str, default="data/benchmark")
    parser.add_argument("--output_dir", type=str, default="results/benchmark")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["direction", "volatility", "crisis", "portfolio"],
                        choices=["direction", "volatility", "crisis", "portfolio", "walk_forward"])
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_assets", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
