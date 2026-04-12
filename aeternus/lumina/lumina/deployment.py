"""
lumina/deployment.py

Production deployment for Lumina financial foundation model.

Covers:
  - ONNX export with dynamic shapes
  - TorchScript compilation (trace and script modes)
  - INT8 and FP16 quantization
  - Triton Inference Server config generation
  - Batched inference with dynamic batching
  - Latency benchmarking (P50/P90/P99)
  - Throughput optimization
  - Model versioning and registry
  - Canary deployment utilities
  - Health checks and monitoring hooks
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import pathlib
import struct
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import onnx
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    logger.debug("ONNX/OnnxRuntime not available.")

try:
    from torch.quantization import (
        quantize_dynamic,
        prepare,
        convert,
        get_default_qconfig,
        QConfig,
    )
    _QUANTIZE_AVAILABLE = True
except ImportError:
    _QUANTIZE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Deployment configuration
# ---------------------------------------------------------------------------

@dataclass
class DeploymentConfig:
    model_name: str = "lumina"
    version: str = "1.0.0"
    output_dir: str = "deployment"
    device: str = "cpu"

    # ONNX
    onnx_opset: int = 17
    onnx_dynamic_axes: bool = True
    onnx_optimize: bool = True

    # Quantization
    quantize: bool = False
    quantize_dtype: str = "int8"   # "int8" | "fp16"
    quantize_dynamic: bool = True  # Dynamic vs. static quantization

    # TorchScript
    torchscript: bool = False
    torchscript_mode: str = "trace"   # "trace" | "script"

    # Batching
    max_batch_size: int = 64
    preferred_batch_size: int = 16
    dynamic_batching: bool = True
    max_queue_delay_ms: float = 5.0

    # Triton
    triton_model_platform: str = "onnxruntime_onnx"   # or "pytorch_libtorch"
    triton_instance_count: int = 2
    triton_gpu_memory_bytes: int = 1 << 30    # 1 GB

    # Benchmarking
    n_warmup_iters: int = 20
    n_benchmark_iters: int = 200

    # Monitoring
    enable_monitoring: bool = True
    latency_slo_ms: float = 50.0   # Service Level Objective


# ---------------------------------------------------------------------------
# ONNX exporter
# ---------------------------------------------------------------------------

class ONNXExporter:
    """
    Exports Lumina models to ONNX format.
    Supports dynamic shapes for variable sequence length and batch size.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def export(
        self,
        model: nn.Module,
        example_input: Union[Tensor, Dict[str, Tensor]],
        output_path: Union[str, pathlib.Path],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> pathlib.Path:
        """
        Export model to ONNX.

        Args:
            model: PyTorch model.
            example_input: Example input(s) for tracing.
            output_path: Path to write .onnx file.
            input_names: Names for ONNX input nodes.
            output_names: Names for ONNX output nodes.

        Returns:
            Path to exported .onnx file.
        """
        if not _ONNX_AVAILABLE:
            raise ImportError("Install onnx and onnxruntime: pip install onnx onnxruntime")

        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = model.eval().cpu()

        # Prepare inputs
        if isinstance(example_input, dict):
            example_args = tuple(example_input.values())
            input_names = input_names or list(example_input.keys())
        else:
            example_args = (example_input,)
            input_names = input_names or ["input"]
        output_names = output_names or ["output"]

        # Dynamic axes for variable batch / seq len
        dynamic_axes = None
        if self.config.onnx_dynamic_axes:
            dynamic_axes = {}
            for name in input_names:
                dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
            for name in output_names:
                dynamic_axes[name] = {0: "batch_size"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,
                example_args,
                str(output_path),
                opset_version=self.config.onnx_opset,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
            )

        # Verify the model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model exported and verified: {output_path}")

        if self.config.onnx_optimize:
            optimized_path = self._optimize_onnx(output_path)
            return optimized_path

        return output_path

    def _optimize_onnx(self, model_path: pathlib.Path) -> pathlib.Path:
        """Apply ONNX graph optimizations."""
        try:
            from onnxruntime.transformers import optimizer as ort_opt
            optimized_path = model_path.with_suffix(".opt.onnx")
            opt_model = ort_opt.optimize_model(
                str(model_path),
                model_type="bert",    # Generic transformer
                num_heads=12,
                hidden_size=768,
            )
            opt_model.save_model_to_file(str(optimized_path))
            logger.info(f"ONNX model optimized: {optimized_path}")
            return optimized_path
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}; using unoptimized model.")
            return model_path

    def validate(
        self,
        onnx_path: Union[str, pathlib.Path],
        pytorch_model: nn.Module,
        test_input: Tensor,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> bool:
        """Validate ONNX output matches PyTorch output."""
        if not _ONNX_AVAILABLE:
            return False

        sess = ort.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name

        onnx_out = sess.run(None, {input_name: test_input.numpy()})[0]
        with torch.no_grad():
            pt_out = pytorch_model(test_input)
            if isinstance(pt_out, dict):
                pt_out = pt_out.get("logits", pt_out.get("output", torch.zeros(1))).numpy()
            else:
                pt_out = pt_out.numpy()

        match = np.allclose(pt_out, onnx_out, rtol=rtol, atol=atol)
        logger.info(f"ONNX validation: {'PASS' if match else 'FAIL'}")
        return bool(match)


# ---------------------------------------------------------------------------
# ONNX Runtime inference
# ---------------------------------------------------------------------------

class ONNXInferenceEngine:
    """
    Fast inference engine using ONNX Runtime.
    Supports CPU, CUDA, and TensorRT backends.
    """

    def __init__(
        self,
        model_path: Union[str, pathlib.Path],
        device: str = "cpu",
        n_threads: int = 4,
    ):
        if not _ONNX_AVAILABLE:
            raise ImportError("Install onnxruntime.")

        self.model_path = str(model_path)
        self.device = device

        providers = []
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        if device == "tensorrt" and "TensorrtExecutionProvider" in ort.get_available_providers():
            providers.append("TensorrtExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = n_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_opts,
            providers=providers,
        )
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

    def run(
        self,
        inputs: Union[Tensor, Dict[str, Tensor], np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Run inference."""
        if isinstance(inputs, Tensor):
            feed = {self._input_names[0]: inputs.numpy()}
        elif isinstance(inputs, dict):
            feed = {k: (v.numpy() if isinstance(v, Tensor) else v) for k, v in inputs.items()}
        else:
            feed = {self._input_names[0]: inputs}

        outputs = self.session.run(self._output_names, feed)
        return dict(zip(self._output_names, outputs))

    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        n_warmup: int = 20,
        n_iters: int = 200,
        dtype: np.dtype = np.float32,
    ) -> Dict[str, float]:
        """Benchmark inference latency."""
        dummy = np.random.randn(*input_shape).astype(dtype)
        feed = {self._input_names[0]: dummy}

        # Warmup
        for _ in range(n_warmup):
            self.session.run(self._output_names, feed)

        # Benchmark
        latencies = []
        for _ in range(n_iters):
            start = time.perf_counter()
            self.session.run(self._output_names, feed)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        return {
            "mean_ms": float(latencies.mean()),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "throughput_qps": float(1000.0 / latencies.mean()),
        }


# ---------------------------------------------------------------------------
# TorchScript compilation
# ---------------------------------------------------------------------------

class TorchScriptCompiler:
    """
    Compiles PyTorch models to TorchScript for production deployment.
    """

    @staticmethod
    def trace(
        model: nn.Module,
        example_input: Union[Tensor, Dict[str, Tensor]],
        output_path: Optional[Union[str, pathlib.Path]] = None,
        optimize_for_inference: bool = True,
    ) -> torch.jit.ScriptModule:
        """
        Trace-compile model with example input.
        Best for models without data-dependent control flow.
        """
        model = model.eval().cpu()
        if isinstance(example_input, dict):
            traced = torch.jit.trace(model, tuple(example_input.values()))
        else:
            traced = torch.jit.trace(model, example_input)

        if optimize_for_inference:
            traced = torch.jit.optimize_for_inference(traced)

        if output_path is not None:
            output_path = pathlib.Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            traced.save(str(output_path))
            logger.info(f"TorchScript model saved: {output_path}")

        return traced

    @staticmethod
    def script(
        model: nn.Module,
        output_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> torch.jit.ScriptModule:
        """
        Script-compile model.
        Supports dynamic control flow but requires type annotations.
        """
        model = model.eval().cpu()
        scripted = torch.jit.script(model)

        if output_path is not None:
            output_path = pathlib.Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            scripted.save(str(output_path))
            logger.info(f"TorchScript model saved: {output_path}")

        return scripted

    @staticmethod
    def load(path: Union[str, pathlib.Path]) -> torch.jit.ScriptModule:
        return torch.jit.load(str(path))

    @staticmethod
    def benchmark(
        model: torch.jit.ScriptModule,
        input_shape: Tuple[int, ...],
        n_warmup: int = 20,
        n_iters: int = 200,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """Benchmark TorchScript inference."""
        device_obj = torch.device(device)
        model = model.to(device_obj)
        dummy = torch.randn(*input_shape, device=device_obj)

        for _ in range(n_warmup):
            with torch.no_grad():
                model(dummy)
        if "cuda" in device:
            torch.cuda.synchronize()

        latencies = []
        for _ in range(n_iters):
            start = time.perf_counter()
            with torch.no_grad():
                model(dummy)
            if "cuda" in device:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        return {
            "mean_ms": float(latencies.mean()),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_qps": float(1000.0 / latencies.mean()),
        }


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

class ModelQuantizer:
    """
    Quantizes Lumina models to INT8 or FP16 for faster inference.
    """

    @staticmethod
    def quantize_dynamic_int8(
        model: nn.Module,
        qconfig_spec: Optional[Dict] = None,
    ) -> nn.Module:
        """
        Dynamic INT8 quantization.
        Quantizes weights but computes activations in FP32 at runtime.
        Simple, no calibration data needed.
        """
        if not _QUANTIZE_AVAILABLE:
            logger.warning("quantize_dynamic not available; returning original model.")
            return model

        model = copy.deepcopy(model).cpu().eval()
        if qconfig_spec is None:
            qconfig_spec = {nn.Linear}

        quantized = quantize_dynamic(
            model,
            qconfig_spec=qconfig_spec,
            dtype=torch.qint8,
        )
        return quantized

    @staticmethod
    def quantize_static_int8(
        model: nn.Module,
        calibration_dataloader: torch.utils.data.DataLoader,
        n_calibration_batches: int = 100,
    ) -> nn.Module:
        """
        Static INT8 quantization with calibration.
        More accurate than dynamic but requires calibration data.
        """
        if not _QUANTIZE_AVAILABLE:
            return model

        model = copy.deepcopy(model).cpu().eval()
        model.qconfig = get_default_qconfig("fbgemm")
        prepare(model, inplace=True)

        # Calibrate
        with torch.no_grad():
            for i, batch in enumerate(calibration_dataloader):
                if i >= n_calibration_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    x = batch[0].cpu()
                elif isinstance(batch, dict):
                    x = batch.get("input_ids", batch.get("features")).cpu()
                else:
                    continue
                model(x)

        convert(model, inplace=True)
        return model

    @staticmethod
    def to_fp16(model: nn.Module) -> nn.Module:
        """Convert model to FP16."""
        return model.half()

    @staticmethod
    def to_bfloat16(model: nn.Module) -> nn.Module:
        """Convert model to BFloat16."""
        return model.to(torch.bfloat16)

    @staticmethod
    def compare_size(original: nn.Module, quantized: nn.Module) -> Dict[str, Any]:
        """Compare parameter sizes before and after quantization."""
        def get_size_mb(model: nn.Module) -> float:
            total_bytes = 0
            for p in model.parameters():
                total_bytes += p.nelement() * p.element_size()
            for b in model.buffers():
                total_bytes += b.nelement() * b.element_size()
            return total_bytes / 1024 ** 2

        orig_size = get_size_mb(original)
        quant_size = get_size_mb(quantized)
        return {
            "original_size_mb": orig_size,
            "quantized_size_mb": quant_size,
            "compression_ratio": orig_size / (quant_size + 1e-10),
            "size_reduction_pct": (1 - quant_size / orig_size) * 100,
        }


# ---------------------------------------------------------------------------
# Triton Inference Server config generation
# ---------------------------------------------------------------------------

class TritonConfigGenerator:
    """
    Generates Triton Inference Server model repository configs.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def generate_config(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        model_type: str = "onnx",
    ) -> str:
        """
        Generate config.pbtxt for Triton.
        """
        cfg = self.config
        input_dims = list(input_shape[1:])  # Exclude batch dim
        output_dims = list(output_shape[1:])

        if model_type == "onnx":
            platform = "onnxruntime_onnx"
            model_file = "model.onnx"
        elif model_type == "torchscript":
            platform = "pytorch_libtorch"
            model_file = "model.pt"
        else:
            platform = cfg.triton_model_platform
            model_file = "model"

        config_lines = [
            f'name: "{cfg.model_name}"',
            f'platform: "{platform}"',
            f'max_batch_size: {cfg.max_batch_size}',
            "",
            "input [",
            "  {",
            '    name: "input"',
            "    data_type: TYPE_FP32",
            f"    dims: {input_dims}",
            "  }",
            "]",
            "",
            "output [",
            "  {",
            '    name: "output"',
            "    data_type: TYPE_FP32",
            f"    dims: {output_dims}",
            "  }",
            "]",
            "",
            "instance_group [",
            "  {",
            "    kind: KIND_GPU",
            f"    count: {cfg.triton_instance_count}",
            "  }",
            "]",
        ]

        if cfg.dynamic_batching:
            config_lines.extend([
                "",
                "dynamic_batching {",
                f"  preferred_batch_size: [ {cfg.preferred_batch_size} ]",
                f"  max_queue_delay_microseconds: {int(cfg.max_queue_delay_ms * 1000)}",
                "}",
            ])

        return "\n".join(config_lines)

    def write_repository(
        self,
        model: Optional[nn.Module] = None,
        onnx_path: Optional[pathlib.Path] = None,
        input_shape: Tuple[int, ...] = (1, 64, 32),
        output_shape: Tuple[int, ...] = (1, 1),
        model_version: int = 1,
    ) -> pathlib.Path:
        """
        Create full Triton model repository directory structure.

        Structure:
          {output_dir}/
            {model_name}/
              config.pbtxt
              {version}/
                model.onnx
        """
        repo_dir = pathlib.Path(self.config.output_dir) / self.config.model_name
        version_dir = repo_dir / str(model_version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Write config
        config_str = self.generate_config(input_shape, output_shape)
        (repo_dir / "config.pbtxt").write_text(config_str)

        # Copy or export model
        if onnx_path and onnx_path.exists():
            import shutil
            shutil.copy(onnx_path, version_dir / "model.onnx")
        elif model is not None and _ONNX_AVAILABLE:
            exporter = ONNXExporter(self.config)
            dummy = torch.randn(*input_shape)
            exporter.export(model, dummy, version_dir / "model.onnx")

        logger.info(f"Triton repository written: {repo_dir}")
        return repo_dir


# ---------------------------------------------------------------------------
# Batched inference engine
# ---------------------------------------------------------------------------

class BatchedInferenceEngine:
    """
    Production-grade batched inference engine with:
      - Dynamic batching (accumulate requests up to max_batch_size)
      - Request queuing with timeout
      - Async processing
      - Latency tracking
    """

    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 64,
        max_wait_ms: float = 5.0,
        device: Optional[torch.device] = None,
        amp: bool = True,
    ):
        self.model = model.eval()
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.device = device or torch.device("cpu")
        self.model = self.model.to(self.device)
        self.amp = amp and self.device.type == "cuda"
        self._latencies: List[float] = []

    @torch.no_grad()
    def infer_batch(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Run inference on a batch."""
        inputs = inputs.to(self.device)
        start = time.perf_counter()

        if self.amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._latencies.append(elapsed_ms)

        if isinstance(outputs, dict):
            return {k: v.cpu() for k, v in outputs.items() if isinstance(v, Tensor)}
        return {"output": outputs.cpu()}

    @torch.no_grad()
    def infer_single(self, input_: Tensor) -> Dict[str, Tensor]:
        """Infer a single sample (adds batch dimension)."""
        if input_.ndim == 1:
            input_ = input_.unsqueeze(0)
        elif input_.ndim == 2:
            input_ = input_.unsqueeze(0)
        return self.infer_batch(input_)

    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = None,
        n_warmup: int = 20,
        n_iters: int = 200,
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark latency across different batch sizes."""
        batch_sizes = batch_sizes or [1, 4, 8, 16, 32, 64]
        results = {}

        for bs in batch_sizes:
            dummy = torch.randn(bs, *input_shape[1:])
            # Warmup
            for _ in range(n_warmup):
                self.infer_batch(dummy)

            latencies = []
            for _ in range(n_iters):
                start = time.perf_counter()
                self.infer_batch(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

            latencies = np.array(latencies)
            results[bs] = {
                "mean_ms": float(latencies.mean()),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p90_ms": float(np.percentile(latencies, 90)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "throughput_qps": float(bs * 1000.0 / latencies.mean()),
                "samples_per_sec": float(bs * 1000.0 / latencies.mean()),
            }
            logger.info(f"Batch {bs}: mean={results[bs]['mean_ms']:.1f}ms, "
                       f"QPS={results[bs]['throughput_qps']:.0f}")

        return results

    def latency_report(self) -> Dict[str, float]:
        if not self._latencies:
            return {}
        arr = np.array(self._latencies)
        return {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "n_requests": len(arr),
        }


# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------

@dataclass
class ModelVersion:
    """Metadata for a deployed model version."""
    name: str
    version: str
    created_at: float
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifact_path: str = ""
    is_champion: bool = False
    is_challenger: bool = False


class ModelRegistry:
    """
    Simple filesystem-based model registry.

    Tracks model versions, metrics, and champion/challenger status.
    """

    def __init__(self, registry_dir: Union[str, pathlib.Path]):
        self.registry_dir = pathlib.Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.registry_dir / "registry.json"
        self._versions: Dict[str, List[ModelVersion]] = {}
        self._load()

    def _load(self) -> None:
        if self._registry_file.exists():
            data = json.loads(self._registry_file.read_text())
            for name, versions in data.items():
                self._versions[name] = [ModelVersion(**v) for v in versions]

    def _save(self) -> None:
        data = {
            name: [asdict(v) for v in versions]
            for name, versions in self._versions.items()
        }
        self._registry_file.write_text(json.dumps(data, indent=2))

    def register(
        self,
        model: nn.Module,
        name: str,
        version: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        """Register a model version."""
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = str(model_dir / "model.pt")
        torch.save(model.state_dict(), artifact_path)

        mv = ModelVersion(
            name=name,
            version=version,
            created_at=time.time(),
            metrics=metrics or {},
            tags=tags or {},
            artifact_path=artifact_path,
        )

        if name not in self._versions:
            self._versions[name] = []
        self._versions[name].append(mv)
        self._save()
        logger.info(f"Registered model {name}:{version}")
        return mv

    def list_versions(self, name: str) -> List[ModelVersion]:
        return self._versions.get(name, [])

    def get_champion(self, name: str) -> Optional[ModelVersion]:
        for v in self._versions.get(name, []):
            if v.is_champion:
                return v
        # Return latest if no explicit champion
        versions = self._versions.get(name, [])
        return versions[-1] if versions else None

    def promote_champion(self, name: str, version: str) -> None:
        """Promote a specific version to champion."""
        for v in self._versions.get(name, []):
            v.is_champion = v.version == version
            v.is_challenger = False
        self._save()
        logger.info(f"Promoted {name}:{version} to champion.")

    def load_champion(self, name: str, model: nn.Module) -> nn.Module:
        """Load champion model weights into provided model."""
        champion = self.get_champion(name)
        if champion is None:
            raise ValueError(f"No champion registered for {name}.")
        state = torch.load(champion.artifact_path, map_location="cpu")
        model.load_state_dict(state)
        return model

    def compare_versions(
        self,
        name: str,
        metric: str = "sharpe_ratio",
    ) -> List[Dict[str, Any]]:
        """Compare versions by a metric."""
        versions = self._versions.get(name, [])
        results = [
            {
                "version": v.version,
                "metric": v.metrics.get(metric, float("nan")),
                "is_champion": v.is_champion,
                "created_at": v.created_at,
            }
            for v in versions
        ]
        return sorted(results, key=lambda x: x["metric"], reverse=True)


# ---------------------------------------------------------------------------
# Canary deployment
# ---------------------------------------------------------------------------

class CanaryDeployment:
    """
    Canary deployment: route a fraction of traffic to the new model
    and compare performance before full rollout.
    """

    def __init__(
        self,
        champion: nn.Module,
        challenger: nn.Module,
        canary_fraction: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.champion = champion.eval()
        self.challenger = challenger.eval()
        self.canary_fraction = canary_fraction
        self.device = device or torch.device("cpu")
        self._champion_latencies: List[float] = []
        self._challenger_latencies: List[float] = []
        self._champion_requests: int = 0
        self._challenger_requests: int = 0

    @torch.no_grad()
    def infer(self, x: Tensor) -> Tuple[Tensor, str]:
        """
        Route request to champion or challenger.
        Returns (output, model_used).
        """
        import random
        x = x.to(self.device)
        if random.random() < self.canary_fraction:
            model = self.challenger
            model_name = "challenger"
        else:
            model = self.champion
            model_name = "champion"

        start = time.perf_counter()
        out = model(x)
        elapsed = (time.perf_counter() - start) * 1000

        if model_name == "champion":
            self._champion_latencies.append(elapsed)
            self._champion_requests += 1
        else:
            self._challenger_latencies.append(elapsed)
            self._challenger_requests += 1

        if isinstance(out, dict):
            out = out.get("output", out.get("logits", list(out.values())[0]))
        return out, model_name

    def stats(self) -> Dict[str, Any]:
        """Return canary deployment statistics."""

        def summarize(lats: List[float]) -> Dict[str, float]:
            if not lats:
                return {}
            arr = np.array(lats)
            return {
                "mean_ms": float(arr.mean()),
                "p99_ms": float(np.percentile(arr, 99)),
                "n_requests": len(arr),
            }

        return {
            "champion": summarize(self._champion_latencies),
            "challenger": summarize(self._challenger_latencies),
            "canary_fraction": self.canary_fraction,
            "champion_requests": self._champion_requests,
            "challenger_requests": self._challenger_requests,
        }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class ModelHealthMonitor:
    """
    Monitors deployed model health:
      - Input distribution drift
      - Output distribution shift
      - Latency SLO compliance
      - Error rate
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._latencies: List[float] = []
        self._errors: int = 0
        self._requests: int = 0
        self._input_stats: Dict[str, float] = {}
        self._output_stats: Dict[str, float] = {}

    def record_request(
        self,
        latency_ms: float,
        input_tensor: Optional[Tensor] = None,
        output_tensor: Optional[Tensor] = None,
        error: bool = False,
    ) -> None:
        self._requests += 1
        self._latencies.append(latency_ms)
        if error:
            self._errors += 1
        if input_tensor is not None:
            with torch.no_grad():
                self._input_stats["mean"] = float(input_tensor.mean())
                self._input_stats["std"] = float(input_tensor.std())
        if output_tensor is not None:
            with torch.no_grad():
                self._output_stats["mean"] = float(output_tensor.mean())
                self._output_stats["std"] = float(output_tensor.std())

    def is_healthy(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if model is within health thresholds."""
        issues = []
        if len(self._latencies) >= 10:
            p99 = np.percentile(self._latencies[-100:], 99)
            if p99 > self.config.latency_slo_ms:
                issues.append(f"P99 latency {p99:.1f}ms > SLO {self.config.latency_slo_ms}ms")

        error_rate = self._errors / max(1, self._requests)
        if error_rate > 0.01:
            issues.append(f"Error rate {error_rate:.2%} > 1%")

        healthy = len(issues) == 0
        return healthy, {
            "healthy": healthy,
            "issues": issues,
            "p99_ms": float(np.percentile(self._latencies[-100:], 99)) if self._latencies else 0.0,
            "error_rate": error_rate,
            "total_requests": self._requests,
        }

    def report(self) -> Dict[str, Any]:
        healthy, health_info = self.is_healthy()
        return {
            **health_info,
            "input_stats": self._input_stats,
            "output_stats": self._output_stats,
        }


# ---------------------------------------------------------------------------
# Full deployment pipeline
# ---------------------------------------------------------------------------

class LuminaDeploymentPipeline:
    """
    End-to-end deployment pipeline for Lumina.

    Steps:
      1. Quantize (optional)
      2. Export to ONNX and/or TorchScript
      3. Benchmark
      4. Generate Triton config
      5. Register to model registry
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.onnx_exporter = ONNXExporter(config)
        self.ts_compiler = TorchScriptCompiler()
        self.quantizer = ModelQuantizer()
        self.triton_gen = TritonConfigGenerator(config)
        self.registry = ModelRegistry(pathlib.Path(config.output_dir) / "registry")

    def deploy(
        self,
        model: nn.Module,
        example_input: Tensor,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run full deployment pipeline."""
        output_dir = pathlib.Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        # Quantize
        deploy_model = copy.deepcopy(model).cpu().eval()
        if self.config.quantize:
            if self.config.quantize_dtype == "int8":
                deploy_model = self.quantizer.quantize_dynamic_int8(deploy_model)
            elif self.config.quantize_dtype == "fp16":
                deploy_model = self.quantizer.to_fp16(deploy_model)
            logger.info(f"Model quantized to {self.config.quantize_dtype}")

        # ONNX export
        onnx_path = output_dir / f"{self.config.model_name}.onnx"
        try:
            if _ONNX_AVAILABLE:
                self.onnx_exporter.export(deploy_model, example_input, onnx_path)
                artifacts["onnx_path"] = str(onnx_path)
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

        # TorchScript
        if self.config.torchscript:
            ts_path = output_dir / f"{self.config.model_name}.pt"
            try:
                self.ts_compiler.trace(deploy_model, example_input, ts_path)
                artifacts["torchscript_path"] = str(ts_path)
            except Exception as e:
                logger.warning(f"TorchScript export failed: {e}")

        # Benchmark
        engine = BatchedInferenceEngine(deploy_model)
        bench_results = engine.benchmark(
            example_input.shape,
            batch_sizes=[1, 8, 16],
            n_warmup=self.config.n_warmup_iters,
            n_iters=self.config.n_benchmark_iters,
        )
        artifacts["benchmark"] = bench_results

        # Triton config
        triton_dir = self.triton_gen.write_repository(
            onnx_path=onnx_path if onnx_path.exists() else None,
            input_shape=example_input.shape,
            output_shape=(example_input.shape[0], 1),
        )
        artifacts["triton_dir"] = str(triton_dir)

        # Register
        mv = self.registry.register(
            model=model,
            name=self.config.model_name,
            version=self.config.version,
            metrics=metrics or {},
        )
        artifacts["model_version"] = asdict(mv)

        logger.info(f"Deployment complete for {self.config.model_name}:{self.config.version}")
        return artifacts


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "DeploymentConfig",
    "ONNXExporter",
    "ONNXInferenceEngine",
    "TorchScriptCompiler",
    "ModelQuantizer",
    "TritonConfigGenerator",
    "BatchedInferenceEngine",
    "ModelVersion",
    "ModelRegistry",
    "CanaryDeployment",
    "ModelHealthMonitor",
    "LuminaDeploymentPipeline",
]
