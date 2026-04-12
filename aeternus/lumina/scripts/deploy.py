#!/usr/bin/env python3
"""
scripts/deploy.py

Deployment script for Lumina financial foundation model.

Usage:
    python scripts/deploy.py --checkpoint checkpoints/pretrain/checkpoint_step_00100000 \
                               --output_dir deployment/v1 \
                               --quantize int8 \
                               --export onnx torchscript \
                               --triton \
                               --benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lumina.deployment import (
    DeploymentConfig,
    ONNXExporter,
    TorchScriptCompiler,
    ModelQuantizer,
    TritonConfigGenerator,
    BatchedInferenceEngine,
    ModelRegistry,
    LuminaDeploymentPipeline,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Minimal model for deployment
# ---------------------------------------------------------------------------

class DeployModel(nn.Module):
    def __init__(self, input_dim: int = 32, d_model: int = 256, n_layers: int = 6):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.head(h)


# ---------------------------------------------------------------------------
# Main deployment function
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Lumina Deployment | output={output_dir}")

    # ── Build / load model ───────────────────────────────────────────────
    model = DeployModel(input_dim=args.feature_dim).eval()

    if args.checkpoint:
        ckpt_path = pathlib.Path(args.checkpoint)
        ckpt_file = ckpt_path / "checkpoint.pt" if ckpt_path.is_dir() else ckpt_path
        if ckpt_file.exists():
            payload = torch.load(ckpt_file, map_location="cpu")
            state_dict = payload.get("model_state_dict", payload)
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded checkpoint: {ckpt_file}")
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_file}. Using random weights.")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    example_input = torch.randn(1, args.seq_len, args.feature_dim)

    # ── Quantization ────────────────────────────────────────────────────
    deploy_model = model
    if args.quantize:
        if args.quantize == "int8":
            logger.info("Applying INT8 dynamic quantization...")
            quantized = ModelQuantizer.quantize_dynamic_int8(model)
            size_info = ModelQuantizer.compare_size(model, quantized)
            logger.info(
                f"Quantization: {size_info['original_size_mb']:.1f}MB -> "
                f"{size_info['quantized_size_mb']:.1f}MB "
                f"({size_info['compression_ratio']:.1f}x)"
            )
            deploy_model = quantized
        elif args.quantize == "fp16":
            logger.info("Converting to FP16...")
            deploy_model = ModelQuantizer.to_fp16(model)
        elif args.quantize == "bf16":
            logger.info("Converting to BF16...")
            deploy_model = ModelQuantizer.to_bfloat16(model)

    artifacts: Dict[str, Any] = {}

    # ── ONNX Export ──────────────────────────────────────────────────────
    if "onnx" in (args.export or []):
        logger.info("Exporting to ONNX...")
        try:
            dep_config = DeploymentConfig(
                model_name=args.model_name,
                version=args.version,
                output_dir=str(output_dir),
                onnx_opset=17,
                onnx_dynamic_axes=True,
            )
            exporter = ONNXExporter(dep_config)
            # Use float32 model for ONNX export (int8 may not be ONNX-compatible)
            onnx_model = model.float().eval()
            onnx_path = exporter.export(onnx_model, example_input, output_dir / f"{args.model_name}.onnx")
            artifacts["onnx_path"] = str(onnx_path)
            logger.info(f"ONNX exported: {onnx_path}")
        except ImportError:
            logger.warning("ONNX not available. Skipping ONNX export.")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

    # ── TorchScript Export ───────────────────────────────────────────────
    if "torchscript" in (args.export or []):
        logger.info("Compiling to TorchScript...")
        try:
            ts_path = output_dir / f"{args.model_name}.pt"
            ts_model = TorchScriptCompiler.trace(
                deploy_model.float().eval(),
                example_input,
                ts_path,
                optimize_for_inference=True,
            )
            artifacts["torchscript_path"] = str(ts_path)
            logger.info(f"TorchScript saved: {ts_path}")
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")

    # ── Triton Config ────────────────────────────────────────────────────
    if args.triton:
        logger.info("Generating Triton config...")
        dep_config = DeploymentConfig(
            model_name=args.model_name,
            version=args.version,
            output_dir=str(output_dir),
            max_batch_size=64,
            preferred_batch_size=16,
            dynamic_batching=True,
        )
        triton_gen = TritonConfigGenerator(dep_config)
        onnx_path_obj = pathlib.Path(artifacts.get("onnx_path", ""))
        triton_dir = triton_gen.write_repository(
            onnx_path=onnx_path_obj if onnx_path_obj.exists() else None,
            model=deploy_model.float() if not onnx_path_obj.exists() else None,
            input_shape=(1, args.seq_len, args.feature_dim),
            output_shape=(1, 1),
        )
        artifacts["triton_dir"] = str(triton_dir)
        logger.info(f"Triton repository: {triton_dir}")

    # ── Latency Benchmark ────────────────────────────────────────────────
    if args.benchmark:
        logger.info("Running latency benchmark...")
        device = torch.device(args.device)
        bench_model = deploy_model.to(device)
        engine = BatchedInferenceEngine(bench_model, device=device)
        bench_results = engine.benchmark(
            input_shape=(1, args.seq_len, args.feature_dim),
            batch_sizes=[1, 4, 8, 16, 32],
            n_warmup=10,
            n_iters=100,
        )
        artifacts["benchmark"] = bench_results

        logger.info("\nLatency benchmark results:")
        for bs, metrics in bench_results.items():
            logger.info(
                f"  Batch {bs:2d}: mean={metrics['mean_ms']:.2f}ms  "
                f"p99={metrics['p99_ms']:.2f}ms  "
                f"QPS={metrics['throughput_qps']:.0f}"
            )

    # ── Model Registry ───────────────────────────────────────────────────
    if args.register:
        logger.info("Registering to model registry...")
        registry = ModelRegistry(output_dir / "registry")
        mv = registry.register(
            model=model,
            name=args.model_name,
            version=args.version,
            tags={"checkpoint": args.checkpoint or "none"},
        )
        artifacts["registry_entry"] = {
            "name": mv.name,
            "version": mv.version,
            "artifact_path": mv.artifact_path,
        }
        logger.info(f"Registered {mv.name}:{mv.version}")

    # ── Save artifacts manifest ──────────────────────────────────────────
    manifest_path = output_dir / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2, default=str))
    logger.info(f"\nDeployment manifest saved: {manifest_path}")
    logger.info("Deployment complete.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lumina deployment")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="deployment/v1")
    parser.add_argument("--model_name", type=str, default="lumina")
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["int8", "fp16", "bf16", None])
    parser.add_argument("--export", nargs="+", default=["onnx"],
                        choices=["onnx", "torchscript"])
    parser.add_argument("--triton", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
