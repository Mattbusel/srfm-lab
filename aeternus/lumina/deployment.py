

# ============================================================
# Extended Deployment Components
# ============================================================

import os
import time
import json
import math
import hashlib
import threading
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class ServingConfig:
    """Configuration for model serving."""
    max_batch_size: int = 32
    max_seq_len: int = 512
    timeout_ms: float = 100.0
    num_workers: int = 4
    device: str = "cpu"
    use_fp16: bool = False
    use_dynamic_batching: bool = True
    cache_size: int = 1000
    warmup_steps: int = 10
    log_requests: bool = True
    rate_limit_qps: Optional[float] = None


@dataclass
class Request:
    """Single inference request."""
    request_id: str
    inputs: Dict[str, torch.Tensor]
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Single inference response."""
    request_id: str
    outputs: Dict[str, torch.Tensor]
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class RequestQueue:
    """Priority queue for inference requests with timeout support."""

    def __init__(self, maxsize: int = 1000):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize)
        self._counter = 0

    def put(self, request: Request, block: bool = True, timeout: Optional[float] = None):
        # Negate priority for max-heap behavior
        item = (-request.priority, self._counter, request)
        self._counter += 1
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Request:
        _, _, request = self._queue.get(block=block, timeout=timeout)
        return request

    def get_batch(self, max_size: int, timeout_ms: float = 10.0) -> List[Request]:
        """Collect up to max_size requests within timeout."""
        batch = []
        deadline = time.time() + timeout_ms / 1000.0
        while len(batch) < max_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                req = self.get(block=True, timeout=remaining)
                batch.append(req)
            except queue.Empty:
                break
        return batch

    def qsize(self) -> int:
        return self._queue.qsize()


class DynamicBatcher:
    """Dynamic batching: accumulates requests until max_batch or timeout."""

    def __init__(self, max_batch_size: int = 32, timeout_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self._pending: List[Request] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

    def add(self, request: Request) -> Optional[List[Request]]:
        """Add request; returns batch to process if ready."""
        with self._lock:
            self._pending.append(request)
            should_flush = (
                len(self._pending) >= self.max_batch_size or
                (time.time() - self._last_flush) * 1000 >= self.timeout_ms
            )
            if should_flush:
                batch = self._pending.copy()
                self._pending.clear()
                self._last_flush = time.time()
                return batch
        return None

    def flush(self) -> List[Request]:
        with self._lock:
            batch = self._pending.copy()
            self._pending.clear()
            self._last_flush = time.time()
        return batch


class InferenceCache:
    """LRU cache for inference results keyed by input hash."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._cache: Dict[str, Response] = {}
        self._access_order: deque = deque()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _hash_inputs(self, inputs: Dict[str, torch.Tensor]) -> str:
        parts = []
        for k in sorted(inputs.keys()):
            v = inputs[k]
            parts.append(f"{k}:{v.shape}:{v.sum().item():.6f}")
        return hashlib.md5(":".join(parts).encode()).hexdigest()

    def get(self, inputs: Dict[str, torch.Tensor]) -> Optional[Response]:
        key = self._hash_inputs(inputs)
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            self.misses += 1
        return None

    def put(self, inputs: Dict[str, torch.Tensor], response: Response):
        key = self._hash_inputs(inputs)
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.capacity:
                oldest = self._access_order.popleft()
                del self._cache[oldest]
            self._cache[key] = response
            self._access_order.append(key)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ModelServer:
    """High-performance model server with batching, caching, and rate limiting."""

    def __init__(self, model: nn.Module, config: ServingConfig):
        self.model = model
        self.config = config
        self.model.eval()
        if config.use_fp16:
            self.model = self.model.half()
        self.model = self.model.to(config.device)

        self.request_queue = RequestQueue(maxsize=10000)
        self.batcher = DynamicBatcher(config.max_batch_size, config.timeout_ms)
        self.cache = InferenceCache(config.cache_size)

        self._response_futures: Dict[str, Any] = {}
        self._workers: List[threading.Thread] = []
        self._running = False
        self._metrics = ServerMetrics()

        # Rate limiter
        if config.rate_limit_qps:
            self._rate_limiter = TokenBucketRateLimiter(config.rate_limit_qps)
        else:
            self._rate_limiter = None

    def start(self):
        self._running = True
        for i in range(self.config.num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True, name=f"server-worker-{i}")
            t.start()
            self._workers.append(t)

    def stop(self):
        self._running = False
        for t in self._workers:
            t.join(timeout=5.0)

    def _worker_loop(self):
        while self._running:
            try:
                requests = self.request_queue.get_batch(
                    self.config.max_batch_size, self.config.timeout_ms
                )
                if requests:
                    self._process_batch(requests)
                else:
                    time.sleep(0.001)
            except Exception as e:
                pass  # Log in production

    def _process_batch(self, requests: List[Request]):
        start = time.time()
        try:
            # Check cache for each request
            uncached = []
            results = {}
            for req in requests:
                cached = self.cache.get(req.inputs)
                if cached is not None:
                    results[req.request_id] = cached
                else:
                    uncached.append(req)

            if uncached:
                # Collate inputs
                keys = list(uncached[0].inputs.keys())
                batch_inputs = {}
                for k in keys:
                    batch_inputs[k] = torch.stack([r.inputs[k] for r in uncached]).to(self.config.device)

                # Inference
                with torch.no_grad():
                    if self.config.use_fp16:
                        with torch.autocast(device_type=self.config.device.split(":")[0]):
                            batch_out = self.model(**batch_inputs)
                    else:
                        batch_out = self.model(**batch_inputs)

                # Unbatch and cache
                latency_ms = (time.time() - start) * 1000 / len(uncached)
                for i, req in enumerate(uncached):
                    if isinstance(batch_out, torch.Tensor):
                        out_i = {"output": batch_out[i]}
                    elif isinstance(batch_out, dict):
                        out_i = {k: v[i] for k, v in batch_out.items()}
                    else:
                        out_i = {"output": batch_out[0][i]}
                    resp = Response(req.request_id, out_i, latency_ms)
                    results[req.request_id] = resp
                    self.cache.put(req.inputs, resp)

            total_latency = (time.time() - start) * 1000
            self._metrics.record_batch(len(requests), total_latency)

        except Exception as e:
            for req in requests:
                results[req.request_id] = Response(req.request_id, {}, error=str(e))

    def infer(self, inputs: Dict[str, torch.Tensor], priority: int = 0) -> str:
        """Submit inference request, returns request_id."""
        import uuid
        req_id = str(uuid.uuid4())[:8]
        req = Request(req_id, inputs, priority)
        self.request_queue.put(req)
        return req_id

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics.summary(),
            "cache_hit_rate": self.cache.hit_rate,
            "queue_depth": self.request_queue.qsize(),
        }


class TokenBucketRateLimiter:
    """Token bucket rate limiter for QPS control."""

    def __init__(self, rate_qps: float, burst: Optional[float] = None):
        self.rate = rate_qps
        self.burst = burst or rate_qps * 2
        self._tokens = self.burst
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, n_tokens: float = 1.0, block: bool = True) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now

            if self._tokens >= n_tokens:
                self._tokens -= n_tokens
                return True
            elif block:
                wait_time = (n_tokens - self._tokens) / self.rate
                time.sleep(wait_time)
                self._tokens = 0
                return True
            return False


class ServerMetrics:
    """Tracks server performance metrics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: deque = deque(maxlen=window_size)
        self._batch_sizes: deque = deque(maxlen=window_size)
        self._total_requests = 0
        self._total_batches = 0
        self._start_time = time.time()

    def record_batch(self, batch_size: int, latency_ms: float):
        self._latencies.append(latency_ms)
        self._batch_sizes.append(batch_size)
        self._total_requests += batch_size
        self._total_batches += 1

    def summary(self) -> Dict[str, float]:
        uptime = time.time() - self._start_time
        lats = list(self._latencies)
        if not lats:
            return {"uptime_s": uptime, "total_requests": self._total_requests}
        return {
            "uptime_s": uptime,
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "throughput_qps": self._total_requests / max(uptime, 1e-8),
            "avg_batch_size": statistics.mean(self._batch_sizes) if self._batch_sizes else 0.0,
            "p50_latency_ms": statistics.median(lats),
            "p95_latency_ms": sorted(lats)[int(0.95 * len(lats))],
            "p99_latency_ms": sorted(lats)[int(0.99 * len(lats))],
            "max_latency_ms": max(lats),
        }


class ModelVersionManager:
    """Manages multiple model versions with A/B testing and canary deployments."""

    def __init__(self):
        self._versions: Dict[str, nn.Module] = {}
        self._traffic_weights: Dict[str, float] = {}
        self._metrics: Dict[str, ServerMetrics] = {}
        self._active_version: str = ""

    def register(self, version_id: str, model: nn.Module, traffic_weight: float = 1.0):
        self._versions[version_id] = model
        self._traffic_weights[version_id] = traffic_weight
        self._metrics[version_id] = ServerMetrics()
        if not self._active_version:
            self._active_version = version_id
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(self._traffic_weights.values())
        for k in self._traffic_weights:
            self._traffic_weights[k] /= total

    def route_request(self) -> str:
        """Select version based on traffic weights."""
        r = torch.rand(1).item()
        cumulative = 0.0
        for version_id, weight in self._traffic_weights.items():
            cumulative += weight
            if r <= cumulative:
                return version_id
        return self._active_version

    def get_model(self, version_id: Optional[str] = None) -> nn.Module:
        if version_id is None:
            version_id = self.route_request()
        return self._versions[version_id]

    def set_canary(self, canary_id: str, canary_fraction: float = 0.05):
        """Route canary_fraction of traffic to canary model."""
        for version_id in self._traffic_weights:
            if version_id == canary_id:
                self._traffic_weights[version_id] = canary_fraction
            else:
                self._traffic_weights[version_id] = (1 - canary_fraction) / max(
                    1, len(self._versions) - 1
                )
        self._normalize_weights()

    def promote_canary(self, canary_id: str):
        """Make canary the primary version."""
        self._active_version = canary_id
        for version_id in self._traffic_weights:
            self._traffic_weights[version_id] = 1.0 if version_id == canary_id else 0.0

    def version_comparison(self) -> Dict[str, Dict[str, float]]:
        return {vid: self._metrics[vid].summary() for vid in self._versions}


class GradientFreeShadowMode(nn.Module):
    """Shadow mode deployment: runs new model alongside production, logs discrepancies."""

    def __init__(self, production_model: nn.Module, shadow_model: nn.Module, log_dir: str = "./shadow_logs"):
        super().__init__()
        self.production = production_model
        self.shadow = shadow_model
        self.log_dir = log_dir
        self._discrepancies: List[float] = []
        os.makedirs(log_dir, exist_ok=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            prod_out = self.production(x, **kwargs)
            try:
                shadow_out = self.shadow(x, **kwargs)
                if isinstance(prod_out, torch.Tensor) and isinstance(shadow_out, torch.Tensor):
                    disc = (prod_out - shadow_out).abs().mean().item()
                    self._discrepancies.append(disc)
            except Exception:
                pass
        return prod_out

    def discrepancy_stats(self) -> Dict[str, float]:
        if not self._discrepancies:
            return {}
        return {
            "mean_discrepancy": statistics.mean(self._discrepancies),
            "max_discrepancy": max(self._discrepancies),
            "n_comparisons": len(self._discrepancies),
        }


class TorchServeAdapter:
    """Adapter to export Lumina models for TorchServe deployment."""

    def __init__(self, model: nn.Module, model_name: str, version: str = "1.0"):
        self.model = model
        self.model_name = model_name
        self.version = version

    def create_handler_script(self, output_path: str) -> str:
        lines = [
            chr(34)*3,
            "TorchServe handler for " + self.model_name + " v" + self.version,
            "Auto-generated by Lumina deployment module.",
            chr(34)*3,
            "import torch",
            "from ts.torch_handler.base_handler import BaseHandler",
            "",
            "",
            "class LuminaHandler(BaseHandler):",
            "    def initialize(self, context):",
            "        self.model = torch.jit.load(properties.get("model_dir") + "/model.pt")",
            "    def preprocess(self, data): return torch.stack([torch.tensor(r.get("body")) for r in data])",
            "    def inference(self, data): return self.model(data)",
            "    def postprocess(self, data): return data.tolist()",
        ]
        handler_content = chr(10).join(lines)
        with open(output_path, "w") as f:
            f.write(handler_content)
        return output_path

    def export_torchscript(self, example_input: torch.Tensor, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        traced = torch.jit.trace(self.model, example_input)
        path = os.path.join(output_dir, f"{self.model_name}.pt")
        traced.save(path)
        return path
