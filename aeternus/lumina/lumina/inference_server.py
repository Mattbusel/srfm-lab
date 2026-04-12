"""
inference_server.py
===================
Production inference server for Lumina MoE.

Features:
  - Async request handling with dynamic batching (max 10ms accumulation)
  - Request priority queue: trading signals > analytics
  - Response streaming for autoregressive generation
  - Concurrent request handling via torch.multiprocessing
  - Health-check / readiness probe endpoints
  - Metrics: req/s, latency p50/p95/p99, batch size dist, expert utilization
  - Integration with RTEL shm-bus: read LOB features from shared memory,
    publish predictions
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import enum
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import signal
import statistics
import struct
import threading
import time
import traceback
import uuid
import warnings
from collections import deque
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as torch_mp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BATCH_WAIT_MS = 10.0          # dynamic batching window
DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_WORKER_PROCESSES = 2
DEFAULT_PORT = 8765
STREAM_CHUNK_SIZE = 1             # tokens per stream chunk
SHM_LOB_FEATURE_DIM = 64         # LOB feature dimension from RTEL shm-bus
SHM_KEY = "lumina_rtel_shm"
METRICS_WINDOW = 1000             # rolling window for metric computation

# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------


class RequestPriority(enum.IntEnum):
    REALTIME_TRADING = 0    # highest priority
    RISK_MANAGEMENT = 1
    ANALYTICS = 2
    BATCH_BACKFILL = 3      # lowest priority


# ---------------------------------------------------------------------------
# Request / Response data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class InferenceRequest:
    request_id: str
    features: np.ndarray                   # (seq_len, feature_dim)
    priority: RequestPriority = RequestPriority.ANALYTICS
    stream: bool = False
    max_new_tokens: int = 1
    timeout_ms: float = 100.0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    enqueue_time: float = dataclasses.field(default_factory=time.monotonic)

    def __lt__(self, other: "InferenceRequest") -> bool:
        """Enable priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.enqueue_time < other.enqueue_time


@dataclasses.dataclass
class InferenceResponse:
    request_id: str
    output: Optional[np.ndarray]           # (seq_len, hidden_dim)
    latency_ms: float
    batch_size: int
    success: bool
    error: Optional[str] = None
    expert_utilization: Optional[List[float]] = None
    stream_chunk: bool = False
    is_final_chunk: bool = True


@dataclasses.dataclass
class HealthStatus:
    healthy: bool
    ready: bool
    uptime_sec: float
    requests_processed: int
    errors: int
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    current_batch_size: int
    queue_depth: int
    model_version: str


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    Thread-safe metrics collector with rolling windows.
    Tracks: requests/sec, latency percentiles, batch sizes, expert utilization.
    """

    def __init__(self, window: int = METRICS_WINDOW):
        self.window = window
        self._lock = threading.Lock()

        self._latencies: deque = deque(maxlen=window)
        self._batch_sizes: deque = deque(maxlen=window)
        self._timestamps: deque = deque(maxlen=window)
        self._errors: int = 0
        self._total_requests: int = 0
        self._expert_util: Dict[int, deque] = {}
        self._dropped_tokens: deque = deque(maxlen=window)

    def record_request(
        self,
        latency_ms: float,
        batch_size: int,
        expert_util: Optional[List[float]] = None,
        dropped_tokens: int = 0,
        error: bool = False,
    ) -> None:
        with self._lock:
            now = time.monotonic()
            self._latencies.append(latency_ms)
            self._batch_sizes.append(batch_size)
            self._timestamps.append(now)
            self._dropped_tokens.append(dropped_tokens)
            self._total_requests += 1
            if error:
                self._errors += 1

            if expert_util:
                for i, u in enumerate(expert_util):
                    if i not in self._expert_util:
                        self._expert_util[i] = deque(maxlen=self.window)
                    self._expert_util[i].append(u)

    def requests_per_sec(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed <= 0:
                return 0.0
            return len(self._timestamps) / elapsed

    def latency_percentiles(self) -> Dict[str, float]:
        with self._lock:
            if not self._latencies:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
            lats = list(self._latencies)
        lats_np = np.array(lats)
        return {
            "p50": float(np.percentile(lats_np, 50)),
            "p95": float(np.percentile(lats_np, 95)),
            "p99": float(np.percentile(lats_np, 99)),
            "mean": float(lats_np.mean()),
        }

    def batch_size_distribution(self) -> Dict[str, float]:
        with self._lock:
            if not self._batch_sizes:
                return {"mean": 0.0, "p50": 0.0, "p99": 0.0}
            bs = list(self._batch_sizes)
        bs_np = np.array(bs)
        return {
            "mean": float(bs_np.mean()),
            "p50": float(np.percentile(bs_np, 50)),
            "p99": float(np.percentile(bs_np, 99)),
        }

    def expert_utilization(self) -> Dict[int, float]:
        with self._lock:
            return {i: float(np.mean(list(v))) for i, v in self._expert_util.items()}

    def summary(self) -> Dict[str, Any]:
        return {
            "requests_per_sec": self.requests_per_sec(),
            "latency": self.latency_percentiles(),
            "batch_size": self.batch_size_distribution(),
            "expert_utilization": self.expert_utilization(),
            "total_requests": self._total_requests,
            "errors": self._errors,
            "error_rate": self._errors / max(self._total_requests, 1),
        }


# ---------------------------------------------------------------------------
# Shared Memory Bus (RTEL shm-bus integration)
# ---------------------------------------------------------------------------


class RTELShmBus:
    """
    Interface to the RTEL (Real-Time Event Loop) shared memory bus.

    Reads LOB (Level Order Book) feature snapshots from a shared memory
    segment and publishes Lumina model predictions back.

    Layout of the shared memory block:
      - 4 bytes: header magic (0xDEADC0DE)
      - 4 bytes: uint32 sequence number (monotonically increasing)
      - 4 bytes: uint32 feature_dim
      - 4 * feature_dim bytes: float32 feature vector
      - 4 bytes: float32 latest_prediction (written by Lumina)
    """

    MAGIC = 0xDEADC0DE
    HEADER_SIZE = 12   # magic + seq_num + feature_dim
    PRED_OFFSET_FROM_END = 4

    def __init__(
        self,
        shm_key: str = SHM_KEY,
        feature_dim: int = SHM_LOB_FEATURE_DIM,
        create: bool = False,
    ):
        self.shm_key = shm_key
        self.feature_dim = feature_dim
        self._shm = None
        self._buf = None
        self._last_seq = -1
        self._enabled = False
        self._lock = threading.Lock()

        try:
            self._init_shm(create)
            self._enabled = True
            logger.info(f"RTELShmBus initialized (shm_key={shm_key})")
        except Exception as e:
            warnings.warn(
                f"RTELShmBus: could not attach to shared memory '{shm_key}': {e}. "
                "Running without RTEL shm-bus integration.",
                RuntimeWarning,
            )

    def _init_shm(self, create: bool) -> None:
        """Attempt to attach to or create the shared memory segment."""
        try:
            from multiprocessing import shared_memory
            total_bytes = (
                self.HEADER_SIZE
                + self.feature_dim * 4
                + self.PRED_OFFSET_FROM_END
            )
            if create:
                self._shm = shared_memory.SharedMemory(
                    name=self.shm_key, create=True, size=total_bytes
                )
                # Write magic header
                struct.pack_into(">I", self._shm.buf, 0, self.MAGIC)
                struct.pack_into(">I", self._shm.buf, 4, 0)
                struct.pack_into(">I", self._shm.buf, 8, self.feature_dim)
            else:
                self._shm = shared_memory.SharedMemory(name=self.shm_key, create=False)
            self._buf = self._shm.buf
        except Exception:
            raise

    def read_lob_features(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Read the latest LOB features from shared memory.
        Returns (features_array, seq_num) or None if no new data or SHM unavailable.
        """
        if not self._enabled or self._buf is None:
            return None

        with self._lock:
            try:
                magic = struct.unpack_from(">I", self._buf, 0)[0]
                if magic != self.MAGIC:
                    return None

                seq_num = struct.unpack_from(">I", self._buf, 4)[0]
                if seq_num <= self._last_seq:
                    return None  # No new data

                feature_dim = struct.unpack_from(">I", self._buf, 8)[0]
                feat_offset = self.HEADER_SIZE
                features = np.frombuffer(
                    self._buf[feat_offset: feat_offset + feature_dim * 4],
                    dtype=np.float32,
                ).copy()

                self._last_seq = seq_num
                return features, seq_num
            except Exception as e:
                logger.debug(f"RTELShmBus read error: {e}")
                return None

    def publish_prediction(self, prediction: float) -> bool:
        """Write a scalar prediction back to shared memory."""
        if not self._enabled or self._buf is None:
            return False
        with self._lock:
            try:
                pred_offset = self.HEADER_SIZE + self.feature_dim * 4
                struct.pack_into("f", self._buf, pred_offset, float(prediction))
                return True
            except Exception as e:
                logger.debug(f"RTELShmBus write error: {e}")
                return False

    def close(self) -> None:
        if self._shm is not None:
            with contextlib.suppress(Exception):
                self._shm.close()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Dynamic Batcher
# ---------------------------------------------------------------------------


class DynamicBatcher:
    """
    Accumulates requests for up to MAX_BATCH_WAIT_MS milliseconds,
    then releases a batch for processing.

    Also enforces a max batch size cap.
    """

    def __init__(
        self,
        max_wait_ms: float = MAX_BATCH_WAIT_MS,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ):
        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._batch_lock = asyncio.Lock()

    async def enqueue(self, request: InferenceRequest) -> None:
        """Add a request to the priority queue."""
        await self._queue.put((request.priority.value, request.enqueue_time, request))

    async def get_batch(self) -> List[InferenceRequest]:
        """
        Collect up to max_batch_size requests, waiting up to max_wait_ms.
        Returns a list of InferenceRequest sorted by priority.
        """
        batch: List[InferenceRequest] = []
        deadline = time.monotonic() + self.max_wait_ms / 1000.0

        # Block until we have at least one request
        try:
            _, _, req = await asyncio.wait_for(
                self._queue.get(),
                timeout=self.max_wait_ms / 1000.0,
            )
            batch.append(req)
        except asyncio.TimeoutError:
            return []

        # Drain queue up to max_batch_size or deadline
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                _, _, req = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=max(remaining, 0),
                )
                batch.append(req)
            except asyncio.TimeoutError:
                break

        return batch

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# Model worker (runs in a separate process)
# ---------------------------------------------------------------------------


class ModelWorkerProcess:
    """
    Wraps the Lumina model in a worker process communicating via queues.
    Enables true CPU/GPU isolation and avoids GIL contention.
    """

    def __init__(
        self,
        model_config_dict: Dict[str, Any],
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        worker_id: int = 0,
    ):
        self.config_dict = model_config_dict
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        self._process: Optional[mp.Process] = None

    def start(self) -> None:
        self._process = mp.Process(
            target=self._worker_loop,
            args=(
                self.config_dict,
                self.request_queue,
                self.response_queue,
                self.worker_id,
            ),
            daemon=True,
            name=f"lumina-worker-{self.worker_id}",
        )
        self._process.start()
        logger.info(f"Worker process {self.worker_id} started (pid={self._process.pid})")

    def stop(self) -> None:
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            logger.info(f"Worker process {self.worker_id} stopped")

    @staticmethod
    def _worker_loop(
        config_dict: Dict[str, Any],
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        worker_id: int,
    ) -> None:
        """Worker process main loop."""
        try:
            from lumina.moe_inference_engine import MoEConfig, LuminaMoEModel, MoEInferenceEngine

            device = config_dict.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            config = MoEConfig(**{k: v for k, v in config_dict.items() if k in MoEConfig.__dataclass_fields__})
            model = LuminaMoEModel(config, num_layers=config_dict.get("num_layers", 2))
            engine = MoEInferenceEngine(model, config)
            engine.warmup()

            logger.info(f"Worker {worker_id}: model loaded on {device}")

            while True:
                try:
                    item = request_queue.get(timeout=1.0)
                    if item is None:
                        break  # shutdown signal

                    req_id, features_np = item
                    features = torch.from_numpy(features_np).unsqueeze(0)  # (1, S, H)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        output = engine.infer(features)
                    t1 = time.perf_counter()

                    response_queue.put((
                        req_id,
                        output.squeeze(0).cpu().numpy(),
                        (t1 - t0) * 1000.0,
                        True,
                        None,
                    ))
                except queue.Empty:
                    continue
                except Exception as e:
                    response_queue.put((req_id, None, 0.0, False, str(e)))
        except Exception as e:
            logger.error(f"Worker {worker_id} fatal error: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Response Streamer
# ---------------------------------------------------------------------------


class ResponseStreamer:
    """
    Streams autoregressive generation token-by-token to the client.
    For financial use: streams updated predictions at each time step.
    """

    def __init__(self, request_id: str, max_tokens: int = 64):
        self.request_id = request_id
        self.max_tokens = max_tokens
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done = False
        self._n_streamed = 0

    async def push(self, chunk: np.ndarray, is_final: bool = False) -> None:
        """Push a response chunk to the stream."""
        response = InferenceResponse(
            request_id=self.request_id,
            output=chunk,
            latency_ms=0.0,
            batch_size=1,
            success=True,
            stream_chunk=True,
            is_final_chunk=is_final,
        )
        await self._queue.put(response)
        self._n_streamed += 1
        if is_final:
            self._done = True

    async def __aiter__(self) -> AsyncGenerator[InferenceResponse, None]:
        """Async iterator over stream chunks."""
        while True:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                yield chunk
                if chunk.is_final_chunk:
                    break
            except asyncio.TimeoutError:
                break


# ---------------------------------------------------------------------------
# Core inference server
# ---------------------------------------------------------------------------


class LuminaInferenceServer:
    """
    Production inference server for Lumina MoE.

    Usage:
        server = LuminaInferenceServer(model, config)
        asyncio.run(server.serve())
    """

    def __init__(
        self,
        model,
        config,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_batch_wait_ms: float = MAX_BATCH_WAIT_MS,
        num_workers: int = DEFAULT_WORKER_PROCESSES,
        enable_shm_bus: bool = True,
    ):
        self.model = model
        self.config = config
        self.host = host
        self.port = port
        self.num_workers = num_workers

        # Engine
        from lumina.moe_inference_engine import MoEInferenceEngine
        self._engine = MoEInferenceEngine(model, config)
        self._engine.warmup()

        # Batching
        self.batcher = DynamicBatcher(max_batch_wait_ms, max_batch_size)

        # Metrics
        self.metrics = MetricsCollector()

        # shm-bus
        self.shm_bus = RTELShmBus(SHM_KEY, SHM_LOB_FEATURE_DIM) if enable_shm_bus else None

        # State
        self._start_time = time.monotonic()
        self._running = False
        self._request_futures: Dict[str, asyncio.Future] = {}
        self._streamers: Dict[str, ResponseStreamer] = {}
        self._lock = asyncio.Lock()

        # Stats
        self._total_requests = 0
        self._total_errors = 0
        self._current_batch_size = 0
        self.model_version = "lumina-moe-v1"

        logger.info(
            f"LuminaInferenceServer initialized: host={host}:{port}, "
            f"max_batch={max_batch_size}, wait_ms={max_batch_wait_ms}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def infer_async(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Submit a single request and await the response."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        async with self._lock:
            self._request_futures[request.request_id] = future

        await self.batcher.enqueue(request)

        try:
            response = await asyncio.wait_for(future, timeout=request.timeout_ms / 1000.0)
            return response
        except asyncio.TimeoutError:
            async with self._lock:
                self._request_futures.pop(request.request_id, None)
            return InferenceResponse(
                request_id=request.request_id,
                output=None,
                latency_ms=request.timeout_ms,
                batch_size=0,
                success=False,
                error="timeout",
            )

    async def stream_infer(
        self,
        request: InferenceRequest,
    ) -> AsyncGenerator[InferenceResponse, None]:
        """Stream inference results token-by-token."""
        request.stream = True
        streamer = ResponseStreamer(request.request_id, request.max_new_tokens)

        async with self._lock:
            self._streamers[request.request_id] = streamer

        await self.batcher.enqueue(request)

        async for chunk in streamer:
            yield chunk

        async with self._lock:
            self._streamers.pop(request.request_id, None)

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------

    async def serve(self) -> None:
        """Main server coroutine. Runs batch processing and shm polling."""
        self._running = True
        logger.info(f"Lumina inference server starting on {self.host}:{self.port}")

        tasks = [
            asyncio.create_task(self._batch_processing_loop()),
            asyncio.create_task(self._metrics_reporting_loop()),
        ]

        if self.shm_bus is not None and self.shm_bus._enabled:
            tasks.append(asyncio.create_task(self._shm_polling_loop()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Server shutting down...")
        finally:
            self._running = False
            for task in tasks:
                task.cancel()

    def shutdown(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Batch processing loop
    # ------------------------------------------------------------------

    async def _batch_processing_loop(self) -> None:
        """Continuously drain the request queue and process batches."""
        while self._running:
            batch = await self.batcher.get_batch()
            if not batch:
                await asyncio.sleep(0.001)
                continue

            await self._process_batch(batch)

    async def _process_batch(self, batch: List[InferenceRequest]) -> None:
        """Process a batch of requests through the model."""
        t_start = time.perf_counter()
        self._current_batch_size = len(batch)

        try:
            # Separate streaming and non-streaming requests
            stream_reqs = [r for r in batch if r.stream]
            sync_reqs = [r for r in batch if not r.stream]

            # Process sync requests as a batch
            if sync_reqs:
                outputs = await self._run_model_batch(sync_reqs)
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000.0

                for req, output in zip(sync_reqs, outputs):
                    response = InferenceResponse(
                        request_id=req.request_id,
                        output=output,
                        latency_ms=latency_ms,
                        batch_size=len(sync_reqs),
                        success=output is not None,
                        error=None if output is not None else "model_error",
                        expert_utilization=self._get_expert_util(),
                    )
                    async with self._lock:
                        future = self._request_futures.pop(req.request_id, None)
                    if future and not future.done():
                        future.set_result(response)

                self.metrics.record_request(
                    latency_ms=latency_ms,
                    batch_size=len(sync_reqs),
                    expert_util=self._get_expert_util(),
                )

            # Process streaming requests one by one
            for req in stream_reqs:
                await self._process_streaming_request(req)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self._total_errors += 1
            # Fail all futures in this batch
            for req in batch:
                async with self._lock:
                    future = self._request_futures.pop(req.request_id, None)
                if future and not future.done():
                    future.set_exception(RuntimeError(str(e)))

        finally:
            self._total_requests += len(batch)

    async def _run_model_batch(
        self, requests: List[InferenceRequest]
    ) -> List[Optional[np.ndarray]]:
        """
        Collate features, run model, split outputs.
        Runs in executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._model_forward_batch, requests)

    def _model_forward_batch(
        self, requests: List[InferenceRequest]
    ) -> List[Optional[np.ndarray]]:
        """Synchronous model forward pass for a batch."""
        try:
            # Find max seq_len for padding
            max_seq = max(r.features.shape[0] for r in requests)
            feat_dim = requests[0].features.shape[-1]

            # Pad and stack
            padded = []
            for r in requests:
                f = r.features
                if f.shape[0] < max_seq:
                    pad = np.zeros((max_seq - f.shape[0], f.shape[-1]), dtype=np.float32)
                    f = np.concatenate([f, pad], axis=0)
                padded.append(f)

            batch_np = np.stack(padded, axis=0)  # (B, S, feat_dim)
            batch_t = torch.from_numpy(batch_np).to(self._engine.device)

            # Project to model hidden dim if needed
            H = self.config.hidden_dim
            if batch_t.shape[-1] != H:
                # Simple linear projection (would be a learned layer in production)
                proj = torch.zeros(
                    batch_t.shape[0], batch_t.shape[1], H,
                    device=batch_t.device, dtype=torch.bfloat16,
                )
                min_dim = min(batch_t.shape[-1], H)
                proj[..., :min_dim] = batch_t[..., :min_dim].to(torch.bfloat16)
                batch_t = proj
            else:
                batch_t = batch_t.to(torch.bfloat16)

            with torch.no_grad():
                output = self._engine.infer(batch_t)  # (B, S, H)

            # Split outputs (un-pad)
            results = []
            for i, r in enumerate(requests):
                out = output[i, :r.features.shape[0]].cpu().float().numpy()
                results.append(out)
            return results
        except Exception as e:
            logger.error(f"Model forward error: {e}")
            return [None] * len(requests)

    async def _process_streaming_request(self, req: InferenceRequest) -> None:
        """Process a streaming inference request step-by-step."""
        loop = asyncio.get_event_loop()
        streamer = self._streamers.get(req.request_id)
        if streamer is None:
            return

        for step in range(req.max_new_tokens):
            output = await loop.run_in_executor(
                None,
                self._model_forward_single,
                req.features,
            )
            is_final = (step == req.max_new_tokens - 1)
            if streamer:
                await streamer.push(output, is_final=is_final)
            if is_final:
                break

    def _model_forward_single(self, features: np.ndarray) -> np.ndarray:
        """Single-sample model forward pass."""
        feat_t = torch.from_numpy(features).unsqueeze(0)  # (1, S, feat_dim)
        H = self.config.hidden_dim
        if feat_t.shape[-1] != H:
            proj = torch.zeros(1, feat_t.shape[1], H, dtype=torch.bfloat16)
            min_d = min(feat_t.shape[-1], H)
            proj[..., :min_d] = feat_t[..., :min_d].to(torch.bfloat16)
            feat_t = proj

        with torch.no_grad():
            out = self._engine.infer(feat_t.to(self._engine.device))
        return out.squeeze(0).cpu().float().numpy()

    # ------------------------------------------------------------------
    # RTEL shm-bus polling loop
    # ------------------------------------------------------------------

    async def _shm_polling_loop(self) -> None:
        """Poll RTEL shm-bus for new LOB features and publish predictions."""
        logger.info("RTEL shm-bus polling started")
        loop = asyncio.get_event_loop()

        while self._running:
            result = await loop.run_in_executor(None, self.shm_bus.read_lob_features)
            if result is not None:
                features_np, seq_num = result
                # Create a high-priority request
                req = InferenceRequest(
                    request_id=f"rtel_{seq_num}",
                    features=features_np.reshape(1, -1),  # (1, feat_dim)
                    priority=RequestPriority.REALTIME_TRADING,
                    timeout_ms=5.0,
                )
                try:
                    response = await asyncio.wait_for(
                        self.infer_async(req), timeout=0.010
                    )
                    if response.success and response.output is not None:
                        # Publish scalar prediction (e.g., predicted return)
                        pred_scalar = float(response.output.mean())
                        self.shm_bus.publish_prediction(pred_scalar)
                except asyncio.TimeoutError:
                    logger.debug(f"shm_bus: timeout on seq {seq_num}")
                except Exception as e:
                    logger.debug(f"shm_bus: error on seq {seq_num}: {e}")

            await asyncio.sleep(0.001)  # poll at ~1 kHz

    # ------------------------------------------------------------------
    # Metrics reporting loop
    # ------------------------------------------------------------------

    async def _metrics_reporting_loop(self) -> None:
        """Periodically log metrics."""
        while self._running:
            await asyncio.sleep(10.0)
            summary = self.metrics.summary()
            logger.info(
                f"Metrics: rps={summary['requests_per_sec']:.1f} "
                f"p50={summary['latency']['p50']:.2f}ms "
                f"p95={summary['latency']['p95']:.2f}ms "
                f"p99={summary['latency']['p99']:.2f}ms "
                f"batch_mean={summary['batch_size']['mean']:.1f} "
                f"errors={summary['errors']}"
            )

    # ------------------------------------------------------------------
    # Health check / readiness probe
    # ------------------------------------------------------------------

    def health_check(self) -> HealthStatus:
        """Return current health status."""
        gpu_used = 0.0
        gpu_total = 0.0
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

        return HealthStatus(
            healthy=self._running,
            ready=self._running and self._engine is not None,
            uptime_sec=time.monotonic() - self._start_time,
            requests_processed=self._total_requests,
            errors=self._total_errors,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
            current_batch_size=self._current_batch_size,
            queue_depth=self.batcher.queue_depth,
            model_version=self.model_version,
        )

    def readiness_probe(self) -> bool:
        """Simple readiness probe: returns True if server can handle requests."""
        return self._running and self._engine is not None

    def liveness_probe(self) -> bool:
        """Liveness probe: returns True if server is not deadlocked."""
        return self._running

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_expert_util(self) -> Optional[List[float]]:
        """Get per-expert utilization from the model."""
        try:
            return self.model.moe_layers[0].expert_utilization()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# HTTP server adapter (minimal, for health checks)
# ---------------------------------------------------------------------------


class HTTPHealthServer:
    """
    Minimal async HTTP server that exposes:
      GET /health   — health check (200 OK or 503)
      GET /ready    — readiness probe
      GET /metrics  — JSON metrics dump
    """

    def __init__(
        self,
        inference_server: LuminaInferenceServer,
        host: str = "0.0.0.0",
        port: int = 8766,
    ):
        self.inference_server = inference_server
        self.host = host
        self.port = port

    async def handle_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            data = await asyncio.wait_for(reader.read(1024), timeout=2.0)
            if not data:
                writer.close()
                return

            request_line = data.decode("utf-8", errors="replace").split("\r\n")[0]
            parts = request_line.split()
            if len(parts) < 2:
                writer.close()
                return

            method, path = parts[0], parts[1]

            if path == "/health":
                status = self.inference_server.health_check()
                body = json.dumps(dataclasses.asdict(status))
                code = "200 OK" if status.healthy else "503 Service Unavailable"
            elif path == "/ready":
                ready = self.inference_server.readiness_probe()
                body = json.dumps({"ready": ready})
                code = "200 OK" if ready else "503 Service Unavailable"
            elif path == "/metrics":
                body = json.dumps(self.inference_server.metrics.summary())
                code = "200 OK"
            else:
                body = '{"error": "not found"}'
                code = "404 Not Found"

            response = (
                f"HTTP/1.1 {code}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n\r\n"
                f"{body}"
            )
            writer.write(response.encode())
            await writer.drain()
        except Exception as e:
            logger.debug(f"HTTP handler error: {e}")
        finally:
            writer.close()

    async def serve(self) -> None:
        server = await asyncio.start_server(
            self.handle_request, self.host, self.port
        )
        logger.info(f"HTTP health server listening on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()


# ---------------------------------------------------------------------------
# Request rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Token-bucket rate limiter for the inference server.
    Prevents single clients from overwhelming the queue.
    """

    def __init__(self, rate: float = 100.0, burst: float = 200.0):
        self.rate = rate
        self.burst = burst
        self._tokens: Dict[str, float] = {}
        self._last_refill: Dict[str, float] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        with self._lock:
            now = time.monotonic()
            if client_id not in self._tokens:
                self._tokens[client_id] = self.burst
                self._last_refill[client_id] = now

            elapsed = now - self._last_refill[client_id]
            self._tokens[client_id] = min(
                self.burst,
                self._tokens[client_id] + elapsed * self.rate,
            )
            self._last_refill[client_id] = now

            if self._tokens[client_id] >= 1.0:
                self._tokens[client_id] -= 1.0
                return True
            return False


# ---------------------------------------------------------------------------
# Load balancer for multi-GPU setups
# ---------------------------------------------------------------------------


class MultiGPULoadBalancer:
    """
    Routes inference requests across multiple GPU workers using round-robin
    with health-based weighting.
    """

    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self._counters = [0] * num_gpus
        self._healthy = [True] * num_gpus
        self._latencies: List[deque] = [deque(maxlen=50) for _ in range(num_gpus)]
        self._lock = threading.Lock()

    def select_gpu(self) -> int:
        """Select the best available GPU."""
        with self._lock:
            healthy = [i for i in range(self.num_gpus) if self._healthy[i]]
            if not healthy:
                return 0

            # Prefer GPU with lowest mean latency
            def score(i: int) -> float:
                if not self._latencies[i]:
                    return 0.0
                return float(np.mean(self._latencies[i]))

            return min(healthy, key=score)

    def record_latency(self, gpu_id: int, latency_ms: float) -> None:
        with self._lock:
            self._latencies[gpu_id].append(latency_ms)

    def mark_unhealthy(self, gpu_id: int) -> None:
        with self._lock:
            self._healthy[gpu_id] = False
            logger.warning(f"GPU {gpu_id} marked unhealthy")

    def mark_healthy(self, gpu_id: int) -> None:
        with self._lock:
            self._healthy[gpu_id] = True

    def status(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "gpu_id": i,
                    "healthy": self._healthy[i],
                    "mean_latency_ms": float(np.mean(self._latencies[i])) if self._latencies[i] else 0.0,
                }
                for i in range(self.num_gpus)
            ]


# ---------------------------------------------------------------------------
# Graceful shutdown manager
# ---------------------------------------------------------------------------


class GracefulShutdown:
    """
    Handles SIGTERM/SIGINT for graceful server shutdown.
    Waits for in-flight requests to complete before exiting.
    """

    def __init__(self, server: LuminaInferenceServer, timeout_sec: float = 30.0):
        self.server = server
        self.timeout_sec = timeout_sec
        self._shutdown_event = asyncio.Event()

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(Exception):
                signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_event.set()

    async def wait_for_shutdown(self) -> None:
        await self._shutdown_event.wait()
        logger.info("Draining in-flight requests...")
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            if self.server.batcher.queue_depth == 0:
                break
            await asyncio.sleep(0.1)
        self.server.shutdown()
        logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def build_inference_server(
    num_experts: int = 8,
    hidden_dim: int = 512,
    ffn_dim: int = 2048,
    num_layers: int = 2,
    device: str = "cuda",
    port: int = DEFAULT_PORT,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> LuminaInferenceServer:
    """Factory function to build and return a ready inference server."""
    from lumina.moe_inference_engine import MoEConfig, LuminaMoEModel

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    config = MoEConfig(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        device=device if torch.cuda.is_available() else "cpu",
        dtype=dtype,
    )
    model = LuminaMoEModel(config, num_layers=num_layers)

    server = LuminaInferenceServer(
        model=model,
        config=config,
        port=port,
        max_batch_size=max_batch_size,
    )
    return server


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = build_inference_server()
    health_server = HTTPHealthServer(server)
    shutdown = GracefulShutdown(server)

    await asyncio.gather(
        server.serve(),
        health_server.serve(),
        shutdown.wait_for_shutdown(),
    )


if __name__ == "__main__":
    asyncio.run(_main())


# ---------------------------------------------------------------------------
# Extended: Request deduplication cache
# ---------------------------------------------------------------------------


class RequestDeduplicationCache:
    """
    Caches recent inference results for identical inputs to avoid
    redundant model execution. Especially useful when the same LOB
    snapshot is submitted multiple times (e.g., from multiple clients).

    Uses content-addressable storage: hash(features) -> response.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        ttl_sec: float = 1.0,
    ):
        self.max_entries = max_entries
        self.ttl_sec = ttl_sec
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _hash_features(self, features: np.ndarray) -> str:
        """Compute a fast hash of the feature array."""
        import hashlib
        return hashlib.md5(features.tobytes()).hexdigest()

    def get(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Return cached output or None."""
        key = self._hash_features(features)
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            ts = self._timestamps.get(key, 0.0)
            if time.monotonic() - ts > self.ttl_sec:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def put(self, features: np.ndarray, output: np.ndarray) -> None:
        """Store a result in the cache."""
        key = self._hash_features(features)
        with self._lock:
            self._cache[key] = output
            self._timestamps[key] = time.monotonic()
            self._cache.move_to_end(key)
            # Evict LRU if over capacity
            while len(self._cache) > self.max_entries:
                oldest_key, _ = self._cache.popitem(last=False)
                self._timestamps.pop(oldest_key, None)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    def stats(self) -> Dict[str, Any]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "cache_size": len(self._cache),
            "max_entries": self.max_entries,
        }


# ---------------------------------------------------------------------------
# Extended: Adaptive batching strategy
# ---------------------------------------------------------------------------


class AdaptiveBatchingStrategy:
    """
    Adapts the batching window and max batch size based on observed
    system load and latency SLO compliance.

    If p99 latency > SLO: reduce batch size / window
    If p99 latency < 50% SLO: increase batch size / window (better throughput)
    """

    def __init__(
        self,
        slo_p99_ms: float = 50.0,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        min_wait_ms: float = 1.0,
        max_wait_ms: float = 20.0,
        adjustment_interval: int = 50,
    ):
        self.slo_p99_ms = slo_p99_ms
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_wait_ms = min_wait_ms
        self.max_wait_ms = max_wait_ms
        self.adjustment_interval = adjustment_interval

        self._current_batch_size = 32
        self._current_wait_ms = 5.0
        self._latency_history: deque = deque(maxlen=adjustment_interval)
        self._n_adjustments = 0
        self._adjustment_history: List[Dict[str, Any]] = []

    def record_latency(self, latency_ms: float) -> None:
        self._latency_history.append(latency_ms)
        if len(self._latency_history) >= self.adjustment_interval:
            self._adjust()

    def _adjust(self) -> None:
        lats = np.array(self._latency_history)
        p99 = float(np.percentile(lats, 99))

        if p99 > self.slo_p99_ms:
            # Under pressure: reduce batch size and wait time
            new_bs = max(self.min_batch_size, int(self._current_batch_size * 0.75))
            new_wait = max(self.min_wait_ms, self._current_wait_ms * 0.8)
        elif p99 < self.slo_p99_ms * 0.5:
            # Headroom: increase batch size and wait time for better throughput
            new_bs = min(self.max_batch_size, int(self._current_batch_size * 1.25))
            new_wait = min(self.max_wait_ms, self._current_wait_ms * 1.2)
        else:
            new_bs = self._current_batch_size
            new_wait = self._current_wait_ms

        if new_bs != self._current_batch_size or new_wait != self._current_wait_ms:
            logger.debug(
                f"AdaptiveBatching: bs {self._current_batch_size}->{new_bs}, "
                f"wait {self._current_wait_ms:.1f}->{new_wait:.1f}ms "
                f"(p99={p99:.1f}ms, SLO={self.slo_p99_ms}ms)"
            )
            self._adjustment_history.append({
                "step": self._n_adjustments,
                "p99_ms": p99,
                "old_batch_size": self._current_batch_size,
                "new_batch_size": new_bs,
                "old_wait_ms": self._current_wait_ms,
                "new_wait_ms": new_wait,
            })
            self._current_batch_size = new_bs
            self._current_wait_ms = new_wait

        self._n_adjustments += 1

    @property
    def current_batch_size(self) -> int:
        return self._current_batch_size

    @property
    def current_wait_ms(self) -> float:
        return self._current_wait_ms


# ---------------------------------------------------------------------------
# Extended: Circuit breaker for model errors
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """
    Circuit breaker pattern for the inference engine.
    Opens the circuit (stops forwarding requests) if too many errors occur.

    States:
      CLOSED  — normal operation
      OPEN    — rejecting requests, waiting for recovery
      HALF_OPEN — testing if the service recovered
    """

    class State(enum.Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.half_open_max_calls = half_open_max_calls

        self._state = self.State.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """Execute fn through the circuit breaker."""
        with self._lock:
            state = self._check_state()

        if state == self.State.OPEN:
            raise RuntimeError("Circuit breaker OPEN: service unavailable")

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _check_state(self) -> "CircuitBreaker.State":
        if self._state == self.State.OPEN:
            if self._last_failure_time and (
                time.monotonic() - self._last_failure_time > self.recovery_timeout_sec
            ):
                self._state = self.State.HALF_OPEN
                self._half_open_calls = 0
                logger.info("CircuitBreaker: OPEN -> HALF_OPEN")
        return self._state

    def _on_success(self) -> None:
        with self._lock:
            if self._state == self.State.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = self.State.CLOSED
                    self._failure_count = 0
                    logger.info("CircuitBreaker: HALF_OPEN -> CLOSED (recovered)")
            elif self._state == self.State.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                if self._state != self.State.OPEN:
                    self._state = self.State.OPEN
                    logger.warning(
                        f"CircuitBreaker: CLOSED -> OPEN "
                        f"(failures={self._failure_count})"
                    )

    @property
    def state(self) -> "CircuitBreaker.State":
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == self.State.OPEN


# ---------------------------------------------------------------------------
# Extended: Request trace collector (distributed tracing)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RequestTrace:
    """Distributed trace for a single inference request."""
    request_id: str
    spans: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    start_time: float = dataclasses.field(default_factory=time.monotonic)
    end_time: Optional[float] = None

    def add_span(self, name: str, duration_ms: float, metadata: Optional[Dict] = None) -> None:
        self.spans.append({
            "name": name,
            "duration_ms": duration_ms,
            "metadata": metadata or {},
            "timestamp": time.monotonic(),
        })

    def finish(self) -> None:
        self.end_time = time.monotonic()

    @property
    def total_latency_ms(self) -> float:
        if self.end_time is None:
            return (time.monotonic() - self.start_time) * 1000.0
        return (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_latency_ms": self.total_latency_ms,
            "spans": self.spans,
        }


class TraceCollector:
    """Collects and stores request traces for observability."""

    def __init__(self, max_traces: int = 10000):
        self.max_traces = max_traces
        self._traces: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def start_trace(self, request_id: str) -> RequestTrace:
        trace = RequestTrace(request_id=request_id)
        with self._lock:
            self._traces[request_id] = trace
            if len(self._traces) > self.max_traces:
                self._traces.popitem(last=False)
        return trace

    def finish_trace(self, request_id: str) -> Optional[RequestTrace]:
        with self._lock:
            trace = self._traces.get(request_id)
            if trace:
                trace.finish()
            return trace

    def get_trace(self, request_id: str) -> Optional[RequestTrace]:
        return self._traces.get(request_id)

    def latency_percentiles(self) -> Dict[str, float]:
        with self._lock:
            finished = [t for t in self._traces.values() if t.end_time is not None]
        if not finished:
            return {}
        lats = np.array([t.total_latency_ms for t in finished])
        return {
            "p50": float(np.percentile(lats, 50)),
            "p95": float(np.percentile(lats, 95)),
            "p99": float(np.percentile(lats, 99)),
            "mean": float(lats.mean()),
        }


# ---------------------------------------------------------------------------
# Extended: Webhook notifier for model events
# ---------------------------------------------------------------------------


class WebhookNotifier:
    """
    Sends webhook notifications for important server events:
    - Model swap (new version promoted)
    - Circuit breaker opened
    - SLO violation
    - Expert collapse detected
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self._enabled = webhook_url is not None

    async def notify(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Send a webhook notification asynchronously."""
        if not self._enabled:
            return False

        data = {
            "event_type": event_type,
            "timestamp": time.time(),
            "payload": payload,
        }

        try:
            import urllib.request
            import json as _json
            body = _json.dumps(data).encode()
            req = urllib.request.Request(
                self.webhook_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, urllib.request.urlopen, req)
            return True
        except Exception as e:
            logger.debug(f"Webhook notification failed: {e}")
            return False

    async def notify_model_swap(self, old_version: str, new_version: str) -> None:
        await self.notify("model_swap", {
            "old_version": old_version,
            "new_version": new_version,
        })

    async def notify_slo_violation(self, p99_ms: float, slo_ms: float) -> None:
        await self.notify("slo_violation", {
            "p99_latency_ms": p99_ms,
            "slo_ms": slo_ms,
            "violation_factor": p99_ms / slo_ms,
        })

    async def notify_circuit_open(self, error_count: int) -> None:
        await self.notify("circuit_breaker_open", {"error_count": error_count})


# ---------------------------------------------------------------------------
# Extended: Prometheus metrics exporter (text format)
# ---------------------------------------------------------------------------


class PrometheusMetricsExporter:
    """
    Exports Lumina inference server metrics in Prometheus text format.
    Suitable for scraping by a Prometheus/Grafana stack.
    """

    def __init__(self, metrics: MetricsCollector, server_name: str = "lumina"):
        self.metrics = metrics
        self.server_name = server_name

    def render(self) -> str:
        """Render metrics in Prometheus text format."""
        lines = []
        summary = self.metrics.summary()
        prefix = f"lumina_{self.server_name}"

        def gauge(name: str, value: float, labels: str = "") -> None:
            label_str = f"{{{labels}}}" if labels else ""
            lines.append(f"# TYPE {prefix}_{name} gauge")
            lines.append(f"{prefix}_{name}{label_str} {value:.6f}")

        gauge("requests_total", float(summary["total_requests"]))
        gauge("errors_total", float(summary["errors"]))
        gauge("error_rate", summary["error_rate"])
        gauge("requests_per_second", summary["requests_per_sec"])

        lat = summary["latency"]
        gauge("latency_p50_ms", lat["p50"])
        gauge("latency_p95_ms", lat["p95"])
        gauge("latency_p99_ms", lat["p99"])
        gauge("latency_mean_ms", lat["mean"])

        bs = summary["batch_size"]
        gauge("batch_size_mean", bs["mean"])
        gauge("batch_size_p99", bs["p99"])

        eu = summary.get("expert_utilization", {})
        for expert_id, util in eu.items():
            gauge("expert_utilization", util, labels=f'expert="{expert_id}"')

        return "\n".join(lines) + "\n"

    async def serve_metrics_endpoint(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve Prometheus metrics on a simple HTTP endpoint."""
        try:
            await asyncio.wait_for(reader.read(1024), timeout=2.0)
            body = self.render()
            response = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: text/plain; version=0.0.4\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n\r\n"
                f"{body}"
            )
            writer.write(response.encode())
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()


# ---------------------------------------------------------------------------
# Extended: Warm pool of pre-allocated tensors
# ---------------------------------------------------------------------------


class TensorPool:
    """
    Pre-allocates a pool of tensors to avoid repeated allocation/deallocation
    during inference. Reduces memory fragmentation and allocation latency.
    """

    def __init__(
        self,
        shapes: List[Tuple],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        pool_size: int = 4,
    ):
        self._pool: Dict[Tuple, queue.Queue] = {}
        self._dtype = dtype
        self._device = device

        for shape in shapes:
            q: queue.Queue = queue.Queue()
            for _ in range(pool_size):
                t = torch.zeros(shape, dtype=dtype, device=device)
                q.put(t)
            self._pool[shape] = q

    def acquire(self, shape: Tuple) -> Tensor:
        """Acquire a tensor from the pool (or allocate a new one)."""
        q = self._pool.get(shape)
        if q is not None:
            try:
                return q.get_nowait()
            except queue.Empty:
                pass
        return torch.zeros(shape, dtype=self._dtype, device=self._device)

    def release(self, tensor: Tensor) -> None:
        """Return a tensor to the pool."""
        shape = tuple(tensor.shape)
        q = self._pool.get(shape)
        if q is not None:
            tensor.zero_()
            try:
                q.put_nowait(tensor)
            except queue.Full:
                pass  # Pool is full, let it be garbage collected

    @contextlib.contextmanager
    def borrow(self, shape: Tuple):
        """Context manager for borrowing a tensor."""
        t = self.acquire(shape)
        try:
            yield t
        finally:
            self.release(t)


# ---------------------------------------------------------------------------
# Extended: Connection pool for downstream services
# ---------------------------------------------------------------------------


class DownstreamServicePool:
    """
    Manages a pool of connections to downstream services that consume
    Lumina model predictions (e.g., order management system, risk engine).

    Provides:
    - Connection health checking
    - Automatic reconnection on failure
    - Round-robin load balancing
    """

    def __init__(
        self,
        endpoints: List[str],
        pool_size: int = 3,
        connect_timeout_sec: float = 2.0,
        request_timeout_sec: float = 1.0,
    ):
        self.endpoints = endpoints
        self.pool_size = pool_size
        self.connect_timeout = connect_timeout_sec
        self.request_timeout = request_timeout_sec

        self._healthy: Dict[str, bool] = {ep: True for ep in endpoints}
        self._round_robin_idx = 0
        self._lock = threading.Lock()
        self._send_counts: Dict[str, int] = {ep: 0 for ep in endpoints}
        self._error_counts: Dict[str, int] = {ep: 0 for ep in endpoints}

    def get_endpoint(self) -> Optional[str]:
        """Get the next available healthy endpoint."""
        with self._lock:
            healthy = [ep for ep in self.endpoints if self._healthy[ep]]
            if not healthy:
                return None
            ep = healthy[self._round_robin_idx % len(healthy)]
            self._round_robin_idx += 1
            return ep

    async def send_prediction(
        self,
        endpoint: str,
        prediction: float,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Send a prediction to a downstream service."""
        try:
            payload = json.dumps({
                "prediction": prediction,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }).encode()

            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, self._send_http, endpoint, payload),
                timeout=self.request_timeout,
            )
            with self._lock:
                self._send_counts[endpoint] = self._send_counts.get(endpoint, 0) + 1
            return True
        except Exception as e:
            logger.debug(f"Failed to send to {endpoint}: {e}")
            with self._lock:
                self._error_counts[endpoint] = self._error_counts.get(endpoint, 0) + 1
                # Mark unhealthy after 3 consecutive errors
                if self._error_counts[endpoint] >= 3:
                    self._healthy[endpoint] = False
                    logger.warning(f"Endpoint {endpoint} marked unhealthy")
            return False

    def _send_http(self, endpoint: str, payload: bytes) -> None:
        """Synchronous HTTP send (runs in executor)."""
        import urllib.request
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            _ = resp.read()

    async def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on all endpoints."""
        results = {}
        for ep in self.endpoints:
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._ping, ep),
                    timeout=self.connect_timeout,
                )
                results[ep] = True
                with self._lock:
                    self._healthy[ep] = True
                    self._error_counts[ep] = 0
            except Exception:
                results[ep] = False
                with self._lock:
                    self._healthy[ep] = False
        return results

    def _ping(self, endpoint: str) -> None:
        import urllib.request
        urllib.request.urlopen(
            endpoint.rstrip("/") + "/health",
            timeout=self.connect_timeout,
        )

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "endpoints": self.endpoints,
                "healthy": dict(self._healthy),
                "send_counts": dict(self._send_counts),
                "error_counts": dict(self._error_counts),
            }


# ---------------------------------------------------------------------------
# Extended: Prediction publisher
# ---------------------------------------------------------------------------


class PredictionPublisher:
    """
    Publishes Lumina model predictions to multiple downstream channels:
    - RTEL shared memory bus
    - HTTP downstream services
    - In-memory ring buffer (for local consumers)
    - Async queue (for async consumers)
    """

    def __init__(
        self,
        shm_bus: Optional[RTELShmBus] = None,
        downstream_pool: Optional[DownstreamServicePool] = None,
        ring_buffer_size: int = 10000,
    ):
        self.shm_bus = shm_bus
        self.downstream_pool = downstream_pool
        self._ring_buffer: deque = deque(maxlen=ring_buffer_size)
        self._async_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._n_published = 0
        self._lock = threading.Lock()

    async def publish(
        self,
        prediction: float,
        seq_num: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Publish a prediction to all channels."""
        payload = {
            "prediction": prediction,
            "seq_num": seq_num,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        with self._lock:
            self._ring_buffer.append(payload)
            self._n_published += 1

        # Async queue
        try:
            self._async_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # Drop oldest item

        # Shared memory
        if self.shm_bus is not None:
            self.shm_bus.publish_prediction(prediction)

        # Downstream services
        if self.downstream_pool is not None:
            endpoint = self.downstream_pool.get_endpoint()
            if endpoint:
                asyncio.create_task(
                    self.downstream_pool.send_prediction(endpoint, prediction, metadata)
                )

    async def subscribe(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Async generator that yields predictions as they arrive."""
        while True:
            try:
                payload = await asyncio.wait_for(
                    self._async_queue.get(), timeout=1.0
                )
                yield payload
            except asyncio.TimeoutError:
                continue

    def recent_predictions(self, n: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent n predictions from the ring buffer."""
        with self._lock:
            all_preds = list(self._ring_buffer)
        return all_preds[-n:]

    @property
    def n_published(self) -> int:
        return self._n_published
