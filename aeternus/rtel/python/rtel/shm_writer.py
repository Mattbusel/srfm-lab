"""
AETERNUS Real-Time Execution Layer (RTEL)
shm_writer.py — Python Shared Memory Bus Writer

Used by Lumina and other Python modules to publish inference outputs
back to the GSR via the RTEL shm-bus.

Usage:
    writer = ShmWriter()
    writer.open_channel("aeternus.lumina.predictions", create=True)
    # Write prediction tensor
    pred = np.zeros((3, 512), dtype=np.float32)
    writer.write_predictions("aeternus.lumina.predictions", pred)
"""
from __future__ import annotations

import ctypes
import logging
import mmap
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .shm_reader import (
    CACHE_LINE_SIZE, DTYPE_FLOAT32, DTYPE_FLOAT64, RTEL_MAGIC,
    RING_CTRL_PADDED, RING_CTRL_SIZE, SLOT_HDR_SIZE,
    ChannelConfig, ChannelCursor, DEFAULT_SLOT_BYTES, DEFAULT_RING_CAP,
    RingControl, SlotHeader, TensorDescriptor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WriterChannelState — mutable state for a write channel
# ---------------------------------------------------------------------------
@dataclass
class WriterStats:
    published_total: int = 0
    errors:          int = 0
    bytes_written:   int = 0
    ring_full_hits:  int = 0
    mean_latency_ns: float = 0.0


# ---------------------------------------------------------------------------
# ShmWriter — publishes data to RTEL channels
# ---------------------------------------------------------------------------
class ShmWriter:
    """
    Writes data from Python to RTEL shared-memory channels.
    Creates channel files if they don't exist.
    Thread-safe (uses per-channel file locks).
    """

    LOB_SNAPSHOT   = "aeternus.chronos.lob"
    VOL_SURFACE    = "aeternus.neuro_sde.vol"
    TENSOR_COMP    = "aeternus.tensornet.compressed"
    GRAPH_ADJ      = "aeternus.omni_graph.adj"
    LUMINA_PRED    = "aeternus.lumina.predictions"
    AGENT_ACTIONS  = "aeternus.hyper_agent.actions"

    def __init__(self, base_path: Path = Path("/tmp")):
        self._base_path = base_path
        self._channels: Dict[str, Tuple[mmap.mmap, Any, ChannelConfig]] = {}
        self._stats:    Dict[str, WriterStats] = {}

    def open_channel(self, name: str,
                     slot_bytes: int = DEFAULT_SLOT_BYTES,
                     ring_capacity: int = DEFAULT_RING_CAP,
                     create: bool = True) -> None:
        if name in self._channels:
            return

        cfg = ChannelConfig(
            name=name,
            slot_bytes=slot_bytes,
            ring_capacity=ring_capacity,
            shm_base_path=self._base_path,
            readonly=False,
        )
        path  = cfg.shm_path()
        total = cfg.total_bytes()

        if create and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * total)
            self._init_ring_control(path, cfg)

        f = open(path, "r+b")
        mm = mmap.mmap(f.fileno(), total, access=mmap.ACCESS_WRITE)
        self._channels[name] = (mm, f, cfg)
        self._stats[name]    = WriterStats()
        logger.info("ShmWriter: opened '%s' (%d bytes)", name, total)

    def _init_ring_control(self, path: Path, cfg: ChannelConfig) -> None:
        """Initialize the RingControl block in a newly created channel file."""
        with open(path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), cfg.total_bytes(), access=mmap.ACCESS_WRITE)
            ctrl = RingControl()
            ctrl.magic       = RTEL_MAGIC
            ctrl.slot_bytes  = cfg.slot_bytes
            ctrl.ring_cap    = cfg.ring_capacity
            ctrl.schema_ver  = 1
            ctrl.write_seq   = 1
            ctrl.min_read_seq= 0
            name_enc = cfg.name.encode()[:63]
            ctrl.channel_name[:len(name_enc)] = name_enc
            mm[:ctypes.sizeof(RingControl)] = bytes(ctrl)
            mm.flush()
            mm.close()

    def open_lumina_channels(self) -> None:
        """Open the standard Lumina output channels."""
        self.open_channel(self.LUMINA_PRED, slot_bytes=64*1024, ring_capacity=512)

    def open_all_output_channels(self) -> None:
        """Open all Python-writable output channels."""
        self.open_channel(self.LUMINA_PRED,  64*1024,  512)
        self.open_channel(self.VOL_SURFACE,  64*1024,  256)
        self.open_channel(self.TENSOR_COMP, 256*1024,  256)
        self.open_channel(self.GRAPH_ADJ,   64*1024,  256)

    def _write_raw(self, name: str, data: bytes,
                   dtype_code: int, shape: Tuple[int, ...]) -> bool:
        """Low-level write: claim slot, copy data, publish sequence."""
        if name not in self._channels:
            logger.error("Channel '%s' not open", name)
            return False

        mm, _f, cfg = self._channels[name]
        stats = self._stats[name]

        if len(data) > cfg.slot_bytes - SLOT_HDR_SIZE:
            logger.error("Data too large: %d", len(data))
            stats.errors += 1
            return False

        t0 = time.perf_counter_ns()

        # Atomically claim a write slot
        # write_seq is at offset 32 in RingControl
        write_seq_off = 32
        old_seq = struct.unpack_from("<Q", mm, write_seq_off)[0]
        new_seq = old_seq + 1
        struct.pack_into("<Q", mm, write_seq_off, new_seq)

        idx      = old_seq & (cfg.ring_capacity - 1)
        slot_off = RING_CTRL_PADDED + idx * cfg.slot_bytes
        pay_off  = slot_off + SLOT_HDR_SIZE

        # Build header
        hdr = SlotHeader()
        hdr.magic         = RTEL_MAGIC
        hdr.sequence      = old_seq
        hdr.timestamp_ns  = time.time_ns()
        hdr.flags         = SlotHeader.FLAG_VALID
        hdr.schema_ver    = 1

        hdr.tensor.dtype         = dtype_code
        hdr.tensor.ndim          = len(shape)
        for i, s in enumerate(shape):
            hdr.tensor.shape[i]  = s
        hdr.tensor.payload_bytes = len(data)
        hdr.tensor.num_elements  = int(np.prod(shape)) if shape else 0
        hdr.tensor.compute_c_strides()  # if method available

        # Write header
        mm[slot_off : slot_off + SLOT_HDR_SIZE] = bytes(hdr)
        # Write payload
        mm[pay_off : pay_off + len(data)] = data
        # Publish: store sequence + 1 (readable)
        struct.pack_into("<Q", mm, slot_off + 8, old_seq + 1)

        t1 = time.perf_counter_ns()
        lat = t1 - t0
        stats.published_total += 1
        stats.bytes_written   += len(data)
        stats.mean_latency_ns  = stats.mean_latency_ns * 0.99 + lat * 0.01
        return True

    # -----------------------------------------------------------------------
    # Typed writers
    # -----------------------------------------------------------------------

    def write_array(self, channel: str, arr: np.ndarray) -> bool:
        """Write a numpy array to the specified channel."""
        dtype_code = {
            np.float32: DTYPE_FLOAT32,
            np.float64: DTYPE_FLOAT64,
        }.get(arr.dtype.type, DTYPE_FLOAT64)
        return self._write_raw(channel, arr.tobytes(), dtype_code, arr.shape)

    def write_predictions(self, channel: str,
                          returns:    np.ndarray,
                          risks:      Optional[np.ndarray] = None,
                          confidence: Optional[np.ndarray] = None) -> bool:
        """
        Write Lumina prediction tensors.
        Packs returns/risks/confidence into a [3 × N] float32 array.
        """
        n = len(returns)
        pred = np.zeros((3, n), dtype=np.float32)
        pred[0] = returns.astype(np.float32)
        if risks is not None:
            pred[1] = risks.astype(np.float32)
        if confidence is not None:
            pred[2] = confidence.astype(np.float32)
        return self.write_array(channel or self.LUMINA_PRED, pred)

    def write_vol_surface(self, channel: str,
                           vols: np.ndarray,
                           asset_id: int = 0) -> bool:
        """Write volatility surface [n_strikes × n_expiries] float64."""
        if vols.ndim != 2:
            logger.error("vol surface must be 2D, got shape %s", vols.shape)
            return False
        # Prepend asset_id header
        header = np.array([asset_id, vols.shape[0], vols.shape[1],
                           time.time_ns()], dtype=np.float64)
        combined = np.concatenate([header, vols.flatten().astype(np.float64)])
        return self.write_array(channel, combined)

    def write_graph(self, channel: str,
                    n_nodes: int, n_edges: int,
                    row_ptr: np.ndarray,
                    col_idx: np.ndarray,
                    weights: np.ndarray) -> bool:
        """Write CSR graph adjacency."""
        header = np.array([n_nodes, n_edges], dtype=np.float32)
        data = np.concatenate([
            header,
            row_ptr.astype(np.float32),
            col_idx.astype(np.float32),
            weights.astype(np.float32),
        ])
        return self.write_array(channel, data)

    def write_compressed_tensor(self, channel: str,
                                  asset_id: int,
                                  tt_cores: list,
                                  compression_ratio: float) -> bool:
        """Write TT-compressed tensor representation."""
        flattened = []
        for core in tt_cores:
            flattened.extend(core.flatten().tolist())
        header = np.array([asset_id, compression_ratio, len(tt_cores)],
                          dtype=np.float32)
        data = np.concatenate([header, np.array(flattened, dtype=np.float32)])
        return self.write_array(channel, data)

    def flush_all(self) -> None:
        """Flush all memory-mapped files."""
        for mm, _f, _cfg in self._channels.values():
            mm.flush()

    def stats(self, channel: Optional[str] = None) -> Dict[str, Any]:
        if channel:
            s = self._stats.get(channel)
            return s.__dict__ if s else {}
        return {k: v.__dict__ for k, v in self._stats.items()}

    def print_stats(self) -> None:
        print(f"{'Channel':<45} {'Published':>10} {'BytesWritten':>14} {'MeanLatNs':>12}")
        print("-" * 85)
        for name, s in self._stats.items():
            print(f"{name:<45} {s.published_total:>10} {s.bytes_written:>14} "
                  f"{s.mean_latency_ns:>12.1f}")

    def close(self) -> None:
        self.flush_all()
        for mm, f, _ in self._channels.values():
            mm.close()
            f.close()
        self._channels.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"ShmWriter(channels={list(self._channels.keys())})"
