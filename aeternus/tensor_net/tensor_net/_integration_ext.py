"""
Extension content for integration.py — appended programmatically.
This file is NOT imported directly; its contents are appended to integration.py.
"""
# ---------------------------------------------------------------------------
# Section 2: Real-time streaming integration utilities
# ---------------------------------------------------------------------------

import threading
import queue as _queue
import time as _time
from collections import deque as _deque
from dataclasses import field as _field


@dataclass
class StreamConfig:
    """Configuration for real-time data stream integration."""
    max_queue_size: int = 10_000
    batch_timeout_ms: float = 50.0
    min_batch_size: int = 1
    max_batch_size: int = 512
    backpressure_threshold: int = 8_000
    drop_policy: str = "oldest"
    enable_metrics: bool = True
    heartbeat_interval_s: float = 5.0


@dataclass
class StreamMetrics:
    """Metrics collected by a streaming integration bus."""
    packets_received: int = 0
    packets_dropped: int = 0
    packets_processed: int = 0
    batches_flushed: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    queue_high_watermark: int = 0

    def record_latency(self, ms: float) -> None:
        self.total_latency_ms += ms
        if ms > self.max_latency_ms:
            self.max_latency_ms = ms

    @property
    def avg_latency_ms(self) -> float:
        if self.packets_processed == 0:
            return 0.0
        return self.total_latency_ms / self.packets_processed

    def to_dict(self) -> dict:
        return {
            "packets_received": self.packets_received,
            "packets_dropped": self.packets_dropped,
            "packets_processed": self.packets_processed,
            "batches_flushed": self.batches_flushed,
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "max_latency_ms": round(self.max_latency_ms, 4),
            "queue_high_watermark": self.queue_high_watermark,
            "drop_rate": round(
                self.packets_dropped / max(1, self.packets_received), 6
            ),
        }


class RealTimeIntegrationBus:
    """
    Thread-safe integration bus that accepts DataPackets from multiple
    producers, batches them, and delivers to registered consumer callbacks.

    Usage::

        def my_consumer(batch: list) -> None:
            for pkt in batch:
                process(pkt)

        bus = RealTimeIntegrationBus(config=StreamConfig())
        bus.register_consumer(my_consumer)
        bus.start()
        bus.push(packet)
        bus.stop()
    """

    def __init__(self, config=None) -> None:
        self.config = config or StreamConfig()
        self._queue = _queue.Queue(maxsize=self.config.max_queue_size)
        self._consumers: list = []
        self._metrics = StreamMetrics()
        self._running = False
        self._worker_thread = None
        self._lock = threading.Lock()

    def push(self, packet) -> bool:
        """Push a DataPacket onto the bus. Returns True if accepted."""
        qsize = self._queue.qsize()
        if self.config.enable_metrics:
            self._metrics.packets_received += 1
            if qsize > self._metrics.queue_high_watermark:
                self._metrics.queue_high_watermark = qsize

        if qsize >= self.config.backpressure_threshold:
            if self.config.drop_policy == "oldest":
                try:
                    self._queue.get_nowait()
                except _queue.Empty:
                    pass
            else:
                if self.config.enable_metrics:
                    self._metrics.packets_dropped += 1
                return False

        try:
            self._queue.put_nowait(packet)
            return True
        except _queue.Full:
            if self.config.enable_metrics:
                self._metrics.packets_dropped += 1
            return False

    def push_many(self, packets: list) -> int:
        """Push multiple packets; returns count of accepted packets."""
        accepted = 0
        for p in packets:
            if self.push(p):
                accepted += 1
        return accepted

    def register_consumer(self, fn) -> None:
        """Register a callable(batch: list[DataPacket]) -> None."""
        self._consumers.append(fn)

    def unregister_consumer(self, fn) -> None:
        self._consumers = [c for c in self._consumers if c is not fn]

    def start(self) -> None:
        """Start background dispatch thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="integration-bus"
        )
        self._worker_thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the dispatch thread gracefully."""
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout_s)

    def _dispatch_loop(self) -> None:
        timeout_s = self.config.batch_timeout_ms / 1000.0
        while self._running:
            batch: list = []
            deadline = _time.monotonic() + timeout_s
            while len(batch) < self.config.max_batch_size:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    break
                try:
                    pkt = self._queue.get(timeout=min(remaining, 0.005))
                    batch.append(pkt)
                except _queue.Empty:
                    if len(batch) >= self.config.min_batch_size:
                        break
            if batch:
                self._deliver(batch)

    def _deliver(self, batch: list) -> None:
        t0 = _time.monotonic()
        for consumer in self._consumers:
            try:
                consumer(batch)
            except Exception as exc:
                import warnings
                warnings.warn(f"Integration bus consumer error: {exc}")
        if self.config.enable_metrics:
            latency_ms = (_time.monotonic() - t0) * 1000.0
            self._metrics.record_latency(latency_ms)
            self._metrics.packets_processed += len(batch)
            self._metrics.batches_flushed += 1

    @property
    def metrics(self) -> StreamMetrics:
        return self._metrics

    def reset_metrics(self) -> None:
        self._metrics = StreamMetrics()


# ---------------------------------------------------------------------------
# Section 3: Cross-module bridge helpers
# ---------------------------------------------------------------------------


def chronos_lob_to_data_packet(records, asset_id: str, source_id: str = "chronos"):
    """
    Convert LOB records to a DataPacket with shape (T, 4*levels).

    Parameters
    ----------
    records : sequence of dict or ChronosLOBRecord
        LOB records in chronological order.
    asset_id : str
        Ticker / instrument identifier.
    source_id : str
        Source tag (default ``"chronos"``).
    """
    rows = []
    timestamps = []
    for r in records:
        if isinstance(r, dict):
            bids_p = r.get("bid_prices", [])
            bids_q = r.get("bid_quantities", [])
            asks_p = r.get("ask_prices", [])
            asks_q = r.get("ask_quantities", [])
            ts = r.get("timestamp", 0.0)
        else:
            bids_p = getattr(r, "bid_prices", [])
            bids_q = getattr(r, "bid_quantities", [])
            asks_p = getattr(r, "ask_prices", [])
            asks_q = getattr(r, "ask_quantities", [])
            ts = getattr(r, "timestamp", 0.0)
        row = list(bids_p) + list(bids_q) + list(asks_p) + list(asks_q)
        rows.append(row)
        timestamps.append(float(ts))

    payload = np.array(rows, dtype=np.float32)
    return DataPacket(
        source_id=source_id,
        packet_type="lob_snapshot",
        timestamp=timestamps[-1] if timestamps else 0.0,
        payload=payload,
        metadata={"asset_id": asset_id, "n_records": len(records), "timestamps": timestamps},
    )


def omni_graph_to_data_packet(
    edge_weights: np.ndarray,
    node_ids: list,
    timestamp: float,
    source_id: str = "omni_graph",
):
    """Wrap Omni-Graph edge-weight matrix as a DataPacket."""
    if edge_weights.ndim != 2 or edge_weights.shape[0] != edge_weights.shape[1]:
        raise ValueError("edge_weights must be a square 2-D array")
    return DataPacket(
        source_id=source_id,
        packet_type="edge_weights",
        timestamp=timestamp,
        payload=edge_weights.astype(np.float32),
        metadata={"node_ids": node_ids, "n_nodes": len(node_ids)},
    )


def lumina_tokens_to_data_packet(
    token_ids: np.ndarray,
    timestamp: float,
    vocab_size: int,
    source_id: str = "lumina",
):
    """Wrap Lumina token IDs as a DataPacket suitable for TT-embedding lookup."""
    return DataPacket(
        source_id=source_id,
        packet_type="token_ids",
        timestamp=timestamp,
        payload=token_ids.astype(np.int32),
        metadata={"vocab_size": vocab_size, "shape": list(token_ids.shape)},
    )


# ---------------------------------------------------------------------------
# Section 4: Packet serialization (JSON-Lines)
# ---------------------------------------------------------------------------


def packets_to_jsonlines(packets: list, path: str) -> None:
    """Serialize a list of DataPackets to a JSON-Lines file."""
    import base64
    import json
    with open(path, "w") as fh:
        for pkt in packets:
            arr = np.array(pkt.payload, dtype=np.float32)
            obj = {
                "source_id": pkt.source_id,
                "packet_type": pkt.packet_type,
                "timestamp": pkt.timestamp,
                "payload_b64": base64.b64encode(arr.tobytes()).decode(),
                "payload_shape": list(arr.shape),
                "payload_dtype": str(arr.dtype),
                "metadata": pkt.metadata,
            }
            fh.write(json.dumps(obj) + "\n")


def packets_from_jsonlines(path: str) -> list:
    """Deserialize DataPackets from a JSON-Lines file written by packets_to_jsonlines."""
    import base64
    import json
    packets = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = base64.b64decode(obj["payload_b64"])
            dtype = obj.get("payload_dtype", "float32")
            shape = tuple(obj["payload_shape"])
            payload = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
            pkt = DataPacket(
                source_id=obj["source_id"],
                packet_type=obj["packet_type"],
                timestamp=obj["timestamp"],
                payload=payload,
                metadata=obj.get("metadata", {}),
            )
            packets.append(pkt)
    return packets


# ---------------------------------------------------------------------------
# Section 5: Integrated multi-source aggregator
# ---------------------------------------------------------------------------


@dataclass
class AggregatorConfig:
    """Configuration for the MultiSourceAggregator."""
    window_size: int = 128
    align_timestamps: bool = True
    fill_method: str = "ffill"
    sources: list = _field(default_factory=list)
    output_shape: tuple = (128, 64)


class MultiSourceAggregator:
    """
    Aggregates DataPackets from multiple sources (Chronos, Omni-Graph, Lumina)
    into a unified tensor for downstream TT processing.

    Maintains a rolling buffer per source and produces aligned multi-source
    output tensors on demand.
    """

    def __init__(self, config=None) -> None:
        self.config = config or AggregatorConfig()
        self._buffers: dict = {
            src: _deque(maxlen=self.config.window_size)
            for src in self.config.sources
        }
        self._lock = threading.Lock()

    def ingest(self, packet) -> None:
        """Accept a DataPacket and append to its source buffer."""
        with self._lock:
            src = packet.source_id
            if src not in self._buffers:
                self._buffers[src] = _deque(maxlen=self.config.window_size)
            self._buffers[src].append(packet)

    def ingest_many(self, packets: list) -> None:
        for p in packets:
            self.ingest(p)

    def get_source_tensor(self, source_id: str):
        """
        Return latest buffered data for *source_id* as (T, D) float32 array.
        Returns None if empty.
        """
        with self._lock:
            buf = self._buffers.get(source_id, _deque())
            if not buf:
                return None
            arrays = []
            for pkt in buf:
                arr = np.array(pkt.payload, dtype=np.float32)
                if arr.ndim == 1:
                    arrays.append(arr)
                elif arr.ndim == 2:
                    arrays.append(arr[-1])
                else:
                    arrays.append(arr.ravel()[:self.config.output_shape[1]])
            stacked = np.stack(arrays, axis=0)
            T, D = stacked.shape
            target_D = self.config.output_shape[1]
            if D < target_D:
                pad = np.zeros((T, target_D - D), dtype=np.float32)
                stacked = np.concatenate([stacked, pad], axis=1)
            else:
                stacked = stacked[:, :target_D]
            return stacked

    def get_unified_tensor(self) -> dict:
        """Return dict of source_id -> (T, D) arrays for all sources."""
        return {src: self.get_source_tensor(src) for src in self._buffers}

    def get_concatenated_tensor(self):
        """Concatenate all sources along feature dimension -> (T, D*S)."""
        tensors = {
            src: t for src, t in self.get_unified_tensor().items()
            if t is not None
        }
        if not tensors:
            return None
        min_T = min(t.shape[0] for t in tensors.values())
        aligned = [t[-min_T:] for t in tensors.values()]
        return np.concatenate(aligned, axis=1)

    def buffer_sizes(self) -> dict:
        with self._lock:
            return {src: len(buf) for src, buf in self._buffers.items()}

    def clear(self, source_id=None) -> None:
        with self._lock:
            if source_id is not None:
                if source_id in self._buffers:
                    self._buffers[source_id].clear()
            else:
                for buf in self._buffers.values():
                    buf.clear()


# ---------------------------------------------------------------------------
# Section 6: AeternusIntegrationPipelineV2
# ---------------------------------------------------------------------------


@dataclass
class PipelineV2Config:
    """Configuration for AeternusIntegrationPipelineV2."""
    stream_config: StreamConfig = _field(default_factory=StreamConfig)
    aggregator_config: AggregatorConfig = _field(default_factory=AggregatorConfig)
    enable_validation: bool = True
    tensor_rank: int = 8
    output_dtype: str = "float32"
    log_every_n_batches: int = 100


class AeternusIntegrationPipelineV2:
    """
    Extended integration pipeline combining RealTimeIntegrationBus,
    MultiSourceAggregator, optional packet validation, and tensor extraction.

    This is the recommended entry-point for connecting Chronos, Omni-Graph,
    and Lumina outputs to TensorNet decomposition routines.
    """

    def __init__(self, config=None) -> None:
        self.config = config or PipelineV2Config()
        self.bus = RealTimeIntegrationBus(self.config.stream_config)
        self.aggregator = MultiSourceAggregator(self.config.aggregator_config)
        self._batch_count = 0
        self.bus.register_consumer(self._on_batch)
        self.bus.start()

    def _on_batch(self, batch: list) -> None:
        valid_batch = []
        for pkt in batch:
            if self.config.enable_validation:
                result = validate_packet(pkt)
                if not result["valid"]:
                    continue
            valid_batch.append(pkt)
        self.aggregator.ingest_many(valid_batch)
        self._batch_count += 1
        if self._batch_count % self.config.log_every_n_batches == 0:
            import warnings
            warnings.warn(
                f"[AeternusPipelineV2] processed {self._batch_count} batches; "
                f"bus metrics: {self.bus.metrics.to_dict()}",
                stacklevel=1,
            )

    def push(self, packet) -> bool:
        """Push a packet onto the integration bus."""
        return self.bus.push(packet)

    def push_chronos_lob(self, records, asset_id: str) -> bool:
        pkt = chronos_lob_to_data_packet(records, asset_id)
        return self.push(pkt)

    def push_omni_graph(self, edge_weights, node_ids, timestamp) -> bool:
        pkt = omni_graph_to_data_packet(edge_weights, node_ids, timestamp)
        return self.push(pkt)

    def push_lumina_tokens(self, token_ids, timestamp, vocab_size) -> bool:
        pkt = lumina_tokens_to_data_packet(token_ids, timestamp, vocab_size)
        return self.push(pkt)

    def get_tensor_for_tt(self, source_id: str):
        """Get a (T, D) array ready for TT decomposition."""
        return self.aggregator.get_source_tensor(source_id)

    def get_concatenated_tensor(self):
        return self.aggregator.get_concatenated_tensor()

    def stop(self) -> None:
        self.bus.stop()

    @property
    def metrics(self) -> StreamMetrics:
        return self.bus.metrics


# ---------------------------------------------------------------------------
# Section 7: Schema registry and versioning
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.2.0"

PACKET_TYPE_REGISTRY: dict = {
    "price_returns": {
        "version": "1.0",
        "required_keys": ["asset_id"],
        "payload_ndim": [2, 3],
        "value_range": (-10.0, 10.0),
        "description": "Normalised log-return tensors from price data.",
    },
    "lob_snapshot": {
        "version": "1.0",
        "required_keys": ["asset_id", "n_records"],
        "payload_ndim": [2],
        "value_range": (-1e6, 1e6),
        "description": "Limit order book bid/ask price+quantity snapshots.",
    },
    "edge_weights": {
        "version": "1.0",
        "required_keys": ["node_ids", "n_nodes"],
        "payload_ndim": [2],
        "value_range": (-1e4, 1e4),
        "description": "Omni-Graph edge weight adjacency matrix.",
    },
    "token_ids": {
        "version": "1.0",
        "required_keys": ["vocab_size", "shape"],
        "payload_ndim": [1, 2],
        "value_range": (0, 1e6),
        "description": "Lumina tokenizer output integer token IDs.",
    },
    "regime_state": {
        "version": "1.0",
        "required_keys": ["n_states"],
        "payload_ndim": [1, 2],
        "value_range": (0.0, 1.0),
        "description": "HMM regime posterior probabilities.",
    },
    "compressed_factors": {
        "version": "1.0",
        "required_keys": ["rank", "method"],
        "payload_ndim": [2, 3],
        "value_range": (-1e3, 1e3),
        "description": "Compressed factor matrices from TT/Tucker decomp.",
    },
    "attention_weights": {
        "version": "1.0",
        "required_keys": ["n_heads"],
        "payload_ndim": [3, 4],
        "value_range": (0.0, 1.0),
        "description": "Multi-head TT attention weight tensors.",
    },
}


def get_packet_type_info(packet_type: str) -> dict:
    """Return registry entry for *packet_type*, or empty dict if unknown."""
    return PACKET_TYPE_REGISTRY.get(packet_type, {})


def list_packet_types() -> list:
    """Return all registered packet type names."""
    return list(PACKET_TYPE_REGISTRY.keys())


def register_packet_type(
    packet_type: str,
    version: str,
    required_keys: list,
    payload_ndim: list,
    value_range: tuple,
    description: str = "",
) -> None:
    """
    Register a new packet type in the global registry.

    Parameters
    ----------
    packet_type : str
        Unique string identifier.
    version : str
        Semantic version string.
    required_keys : list of str
        Keys required in DataPacket.metadata.
    payload_ndim : list of int
        Accepted ndim values for payload.
    value_range : tuple (lo, hi)
        Soft bounds for validation.
    description : str
        Human-readable description.
    """
    PACKET_TYPE_REGISTRY[packet_type] = {
        "version": version,
        "required_keys": required_keys,
        "payload_ndim": payload_ndim,
        "value_range": value_range,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Section 8: Replay and simulation utilities
# ---------------------------------------------------------------------------


class PacketReplayEngine:
    """
    Replays a sequence of DataPackets at wall-clock speed (or faster/slower),
    emitting them to registered callbacks.

    Useful for back-testing and offline simulation of live pipelines.

    Parameters
    ----------
    packets : list of DataPacket
        Packets to replay in order of ``timestamp``.
    speed_factor : float
        1.0 = real-time, 2.0 = 2x faster, 0.5 = half speed.
    loop : bool
        If True, restart from beginning after exhausting packets.
    """

    def __init__(self, packets: list, speed_factor: float = 1.0, loop: bool = False) -> None:
        self._packets = sorted(packets, key=lambda p: p.timestamp)
        self.speed_factor = max(1e-6, speed_factor)
        self.loop = loop
        self._running = False
        self._thread = None
        self._callbacks: list = []

    def register_callback(self, fn) -> None:
        """Register callable(pkt: DataPacket) -> None."""
        self._callbacks.append(fn)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._replay_loop, daemon=True, name="packet-replay"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _replay_loop(self) -> None:
        while self._running:
            packets = self._packets
            if not packets:
                break
            t0_real = _time.monotonic()
            t0_pkt = packets[0].timestamp
            for pkt in packets:
                if not self._running:
                    return
                elapsed_pkt = pkt.timestamp - t0_pkt
                target_real = t0_real + elapsed_pkt / self.speed_factor
                now = _time.monotonic()
                if target_real > now:
                    _time.sleep(target_real - now)
                for cb in self._callbacks:
                    try:
                        cb(pkt)
                    except Exception as exc:
                        import warnings
                        warnings.warn(f"PacketReplayEngine callback error: {exc}")
            if not self.loop:
                break
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Section 9: Integration health-check
# ---------------------------------------------------------------------------


def integration_health_check(pipeline=None) -> dict:
    """
    Run a lightweight self-test of the integration layer.

    Pushes synthetic packets through the bus, verifies arrival in the
    aggregator, and returns a dict with a ``"status"`` key.
    """
    import time as _t

    results: dict = {"status": "ok", "checks": []}

    # 1. Packet construction
    try:
        pkt = make_packet(
            source_id="health_check",
            packet_type="price_returns",
            payload=np.zeros((8, 4), dtype=np.float32),
            metadata={"asset_id": "TEST"},
        )
        assert pkt.source_id == "health_check"
        results["checks"].append({"name": "packet_construction", "passed": True})
    except Exception as exc:
        results["checks"].append({"name": "packet_construction", "passed": False, "error": str(exc)})
        results["status"] = "degraded"

    # 2. Validation
    try:
        val = validate_packet(pkt)
        results["checks"].append({"name": "packet_validation", "passed": val["valid"]})
    except Exception as exc:
        results["checks"].append({"name": "packet_validation", "passed": False, "error": str(exc)})
        results["status"] = "degraded"

    # 3. Serialization round-trip
    try:
        js = pkt.to_json()
        pkt2 = DataPacket.from_json(js)
        assert pkt2.source_id == pkt.source_id
        results["checks"].append({"name": "json_roundtrip", "passed": True})
    except Exception as exc:
        results["checks"].append({"name": "json_roundtrip", "passed": False, "error": str(exc)})
        results["status"] = "degraded"

    # 4. Pipeline push (optional)
    if pipeline is not None:
        try:
            ok = pipeline.push(pkt)
            _t.sleep(0.15)
            sizes = pipeline.aggregator.buffer_sizes()
            results["checks"].append({"name": "pipeline_push", "passed": ok, "buffer_sizes": sizes})
        except Exception as exc:
            results["checks"].append({"name": "pipeline_push", "passed": False, "error": str(exc)})
            results["status"] = "degraded"

    failed = [c for c in results["checks"] if not c["passed"]]
    if failed:
        results["status"] = "degraded"
        results["failed_checks"] = [c["name"] for c in failed]

    return results


# ---------------------------------------------------------------------------
# Section 10: PacketFilter and PacketRouter
# ---------------------------------------------------------------------------


class PacketFilter:
    """
    Applies a chain of predicate functions to DataPackets, passing only
    those that satisfy ALL predicates to the wrapped consumer.

    Parameters
    ----------
    consumer : callable(list[DataPacket]) -> None
    predicates : list of callable(DataPacket) -> bool
    """

    def __init__(self, consumer, predicates: list) -> None:
        self._consumer = consumer
        self._predicates = predicates
        self.n_passed = 0
        self.n_rejected = 0

    def __call__(self, batch: list) -> None:
        accepted = []
        for pkt in batch:
            if all(pred(pkt) for pred in self._predicates):
                accepted.append(pkt)
            else:
                self.n_rejected += 1
        if accepted:
            self.n_passed += len(accepted)
            self._consumer(accepted)

    @staticmethod
    def by_source(source_id: str):
        """Predicate: keep only packets from *source_id*."""
        return lambda pkt: pkt.source_id == source_id

    @staticmethod
    def by_type(packet_type: str):
        """Predicate: keep only packets of *packet_type*."""
        return lambda pkt: pkt.packet_type == packet_type

    @staticmethod
    def by_timestamp_range(lo: float, hi: float):
        """Predicate: keep only packets with timestamp in [lo, hi]."""
        return lambda pkt: lo <= pkt.timestamp <= hi

    @staticmethod
    def no_nan():
        """Predicate: reject packets whose payload contains NaN."""
        return lambda pkt: not np.any(np.isnan(np.array(pkt.payload, dtype=np.float32)))


class PacketRouter:
    """
    Routes DataPackets to different consumer callables based on packet_type.

    Usage::

        router = PacketRouter()
        router.route("lob_snapshot", lob_handler)
        router.route("edge_weights", graph_handler)
        router.set_default(fallback_handler)
        bus.register_consumer(router)
    """

    def __init__(self) -> None:
        self._routes: dict = {}
        self._default = None

    def route(self, packet_type: str, consumer) -> None:
        """Bind *packet_type* to *consumer*."""
        self._routes[packet_type] = consumer

    def set_default(self, consumer) -> None:
        """Consumer for unmatched packet types."""
        self._default = consumer

    def __call__(self, batch: list) -> None:
        buckets: dict = {}
        for pkt in batch:
            key = pkt.packet_type
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(pkt)
        for key, pkts in buckets.items():
            consumer = self._routes.get(key, self._default)
            if consumer is not None:
                try:
                    consumer(pkts)
                except Exception as exc:
                    import warnings
                    warnings.warn(f"PacketRouter consumer error for type={key}: {exc}")


# ---------------------------------------------------------------------------
# Section 11: Adaptive sampling and throttling
# ---------------------------------------------------------------------------


class AdaptiveSampler:
    """
    Stochastically down-samples an incoming packet stream to a target rate,
    adjusting the keep probability based on measured arrival rate.

    Parameters
    ----------
    target_rate_hz : float
        Desired downstream packet rate (packets per second).
    window_s : float
        Measurement window for arrival rate estimation.
    """

    def __init__(self, target_rate_hz: float = 100.0, window_s: float = 1.0) -> None:
        self.target_rate_hz = target_rate_hz
        self.window_s = window_s
        self._arrival_times: _deque = _deque()
        self._rng = np.random.default_rng(seed=42)
        self._keep_prob = 1.0

    def should_keep(self) -> bool:
        """Call for each packet; returns True if it should be forwarded."""
        now = _time.monotonic()
        self._arrival_times.append(now)
        # Evict old entries
        cutoff = now - self.window_s
        while self._arrival_times and self._arrival_times[0] < cutoff:
            self._arrival_times.popleft()
        arrival_rate = len(self._arrival_times) / self.window_s
        if arrival_rate > 0:
            self._keep_prob = min(1.0, self.target_rate_hz / arrival_rate)
        return self._rng.random() < self._keep_prob

    @property
    def keep_probability(self) -> float:
        """Current keep probability."""
        return self._keep_prob


class TokenBucketThrottle:
    """
    Token-bucket rate limiter for DataPacket streams.

    Parameters
    ----------
    rate_hz : float
        Sustained rate limit in packets per second.
    burst : int
        Maximum burst size.
    """

    def __init__(self, rate_hz: float = 100.0, burst: int = 50) -> None:
        self.rate_hz = rate_hz
        self.burst = burst
        self._tokens: float = float(burst)
        self._last_refill = _time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: int = 1) -> bool:
        """Try to acquire *n* tokens. Returns True if acquired, False if throttled."""
        with self._lock:
            now = _time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate_hz)
            self._last_refill = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


# ---------------------------------------------------------------------------
# Section 12: Metrics export helpers
# ---------------------------------------------------------------------------


def export_metrics_csv(metrics: StreamMetrics, path: str) -> None:
    """
    Write StreamMetrics to a CSV file with a single header row and data row.

    Parameters
    ----------
    metrics : StreamMetrics
        Metrics object from a RealTimeIntegrationBus.
    path : str
        Output file path.
    """
    import csv
    d = metrics.to_dict()
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(d.keys()))
        writer.writeheader()
        writer.writerow(d)


def export_metrics_json(metrics: StreamMetrics, path: str) -> None:
    """Write StreamMetrics to a JSON file."""
    import json
    with open(path, "w") as fh:
        json.dump(metrics.to_dict(), fh, indent=2)


def aggregate_metrics(metrics_list: list) -> dict:
    """
    Aggregate a list of StreamMetrics objects into summary statistics.

    Returns a dict with keys:
    ``total_packets_received``, ``total_packets_processed``,
    ``total_packets_dropped``, ``mean_avg_latency_ms``,
    ``max_max_latency_ms``, ``total_batches_flushed``.
    """
    if not metrics_list:
        return {}
    total_received = sum(m.packets_received for m in metrics_list)
    total_processed = sum(m.packets_processed for m in metrics_list)
    total_dropped = sum(m.packets_dropped for m in metrics_list)
    total_batches = sum(m.batches_flushed for m in metrics_list)
    latencies = [m.avg_latency_ms for m in metrics_list if m.packets_processed > 0]
    max_latencies = [m.max_latency_ms for m in metrics_list]
    return {
        "total_packets_received": total_received,
        "total_packets_processed": total_processed,
        "total_packets_dropped": total_dropped,
        "total_batches_flushed": total_batches,
        "mean_avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "max_max_latency_ms": float(np.max(max_latencies)) if max_latencies else 0.0,
        "overall_drop_rate": total_dropped / max(1, total_received),
    }


# ---------------------------------------------------------------------------
# Section 13: Convenience re-exports and module __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Core
    "DataPacket",
    "make_packet",
    "validate_packet",
    "PACKET_SCHEMAS",
    # Consumers
    "ChronosLOBConsumer",
    "OmniGraphConsumer",
    "LuminaTokenizerBridge",
    "AeternusIntegrationPipeline",
    # V2
    "StreamConfig",
    "StreamMetrics",
    "RealTimeIntegrationBus",
    "AggregatorConfig",
    "MultiSourceAggregator",
    "PipelineV2Config",
    "AeternusIntegrationPipelineV2",
    # Converters
    "chronos_lob_to_data_packet",
    "omni_graph_to_data_packet",
    "lumina_tokens_to_data_packet",
    # Serialization
    "packets_to_jsonlines",
    "packets_from_jsonlines",
    # Registry
    "SCHEMA_VERSION",
    "PACKET_TYPE_REGISTRY",
    "get_packet_type_info",
    "list_packet_types",
    "register_packet_type",
    # Replay
    "PacketReplayEngine",
    # Routing/filtering
    "PacketFilter",
    "PacketRouter",
    # Rate control
    "AdaptiveSampler",
    "TokenBucketThrottle",
    # Metrics
    "export_metrics_csv",
    "export_metrics_json",
    "aggregate_metrics",
    # Health
    "integration_health_check",
]
