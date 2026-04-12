"""
AETERNUS Real-Time Execution Layer (RTEL)
shm_reader.py — Python Shared Memory Bus Reader

Zero-copy reading of RTEL shm-bus channels using mmap + ctypes.
Provides numpy array views directly into shared memory.

Usage:
    reader = ShmReader()
    reader.open_channel("aeternus.chronos.lob")
    cursor = reader.new_cursor("aeternus.chronos.lob")
    while True:
        snap = reader.read_lob(cursor)
        if snap is not None:
            print(f"mid={snap.mid_price:.4f}")
"""
from __future__ import annotations

import ctypes
import logging
import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match C++ / Rust)
# ---------------------------------------------------------------------------
RTEL_MAGIC         = 0xAE7E_4E55_5254_4C00
CACHE_LINE_SIZE    = 64
DEFAULT_SLOT_BYTES = 64 * 1024
DEFAULT_RING_CAP   = 1024
MAX_ASSETS         = 512
MAX_LOB_LEVELS     = 10
MAX_STRIKES        = 50
MAX_EXPIRIES       = 12

# DType codes
DTYPE_FLOAT32  = 0x01
DTYPE_FLOAT16  = 0x02
DTYPE_FLOAT64  = 0x03
DTYPE_INT32    = 0x04
DTYPE_INT64    = 0x05
DTYPE_UINT8    = 0x06

DTYPE_TO_NUMPY = {
    DTYPE_FLOAT32: np.float32,
    DTYPE_FLOAT16: np.float16,
    DTYPE_FLOAT64: np.float64,
    DTYPE_INT32:   np.int32,
    DTYPE_INT64:   np.int64,
    DTYPE_UINT8:   np.uint8,
}

DTYPE_ELEMENT_SIZE = {
    DTYPE_FLOAT32: 4,
    DTYPE_FLOAT16: 2,
    DTYPE_FLOAT64: 8,
    DTYPE_INT32:   4,
    DTYPE_INT64:   8,
    DTYPE_UINT8:   1,
}

# ---------------------------------------------------------------------------
# ctypes struct layouts (must match C++ exact layout)
# ---------------------------------------------------------------------------

class TensorDescriptor(ctypes.Structure):
    """Mirrors C++ TensorDescriptor — 256 bytes, cache-line aligned."""
    _pack_ = 8
    _fields_ = [
        ("dtype",         ctypes.c_uint8),
        ("ndim",          ctypes.c_uint8),
        ("_pad",          ctypes.c_uint8 * 6),
        ("shape",         ctypes.c_uint64 * 8),
        ("strides",       ctypes.c_uint64 * 8),
        ("num_elements",  ctypes.c_uint64),
        ("payload_bytes", ctypes.c_uint64),
        ("name",          ctypes.c_char * 32),
    ]

    def numpy_dtype(self) -> np.dtype:
        return np.dtype(DTYPE_TO_NUMPY.get(self.dtype, np.uint8))

    def shape_tuple(self) -> Tuple[int, ...]:
        return tuple(self.shape[i] for i in range(self.ndim))

    def strides_tuple(self) -> Tuple[int, ...]:
        return tuple(self.strides[i] for i in range(self.ndim))


class RingControl(ctypes.Structure):
    """Mirrors C++ RingControl — at offset 0 in shared memory."""
    _pack_ = 8
    _fields_ = [
        ("magic",         ctypes.c_uint64),
        ("slot_bytes",    ctypes.c_uint64),
        ("ring_cap",      ctypes.c_uint64),
        ("schema_ver",    ctypes.c_uint64),
        ("write_seq",     ctypes.c_uint64),   # atomic in C++
        ("min_read_seq",  ctypes.c_uint64),
        ("channel_name",  ctypes.c_char * 64),
        ("_pad",          ctypes.c_uint8 * 64),
    ]


class SlotHeader(ctypes.Structure):
    """Mirrors C++ SlotHeader — at offset 0 of each ring slot."""
    _pack_ = 8
    _fields_ = [
        ("magic",        ctypes.c_uint64),
        ("sequence",     ctypes.c_uint64),    # atomic in C++
        ("timestamp_ns", ctypes.c_uint64),
        ("producer_id",  ctypes.c_uint64),
        ("flags",        ctypes.c_uint32),
        ("schema_ver",   ctypes.c_uint32),
        ("tensor",       TensorDescriptor),
        ("_pad",         ctypes.c_uint8 * 64),
    ]

    FLAG_VALID      = 0x01
    FLAG_COMPRESSED = 0x02
    FLAG_LAST       = 0x04
    FLAG_HEARTBEAT  = 0x08

    def is_valid(self) -> bool:
        return bool(self.flags & self.FLAG_VALID)


RING_CTRL_SIZE   = ctypes.sizeof(RingControl)
RING_CTRL_PADDED = (RING_CTRL_SIZE + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1)
SLOT_HDR_SIZE    = ctypes.sizeof(SlotHeader)


# ---------------------------------------------------------------------------
# ChannelCursor — per-consumer read cursor
# ---------------------------------------------------------------------------
@dataclass
class ChannelCursor:
    """Tracks consumer position in a ring channel."""
    channel_name: str
    next_seq: int = 1

    def advance(self) -> None:
        self.next_seq += 1


# ---------------------------------------------------------------------------
# ChannelConfig
# ---------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    name:           str
    slot_bytes:     int = DEFAULT_SLOT_BYTES
    ring_capacity:  int = DEFAULT_RING_CAP
    shm_base_path:  Path = Path("/tmp")
    readonly:       bool = True

    def shm_path(self) -> Path:
        safe = self.name.replace(".", "_")
        return self.shm_base_path / f"aeternus_rtel_{safe}"

    def total_bytes(self) -> int:
        cap = self.ring_capacity
        return RING_CTRL_PADDED + cap * self.slot_bytes


# ---------------------------------------------------------------------------
# ShmChannel — a single mapped channel
# ---------------------------------------------------------------------------
class ShmChannel:
    """Memory-mapped RTEL shm channel."""

    def __init__(self, cfg: ChannelConfig):
        self.cfg = cfg
        self._mmap: Optional[mmap.mmap] = None
        self._file  = None
        self._lock  = Lock()

        self._stat_consumed:  int = 0
        self._stat_bytes_read:int = 0

        self._open()

    def _open(self) -> None:
        path = self.cfg.shm_path()
        total = self.cfg.total_bytes()

        if not path.exists():
            if self.cfg.readonly:
                raise FileNotFoundError(f"ShmChannel: {path} not found")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * total)

        access = mmap.ACCESS_READ if self.cfg.readonly else mmap.ACCESS_WRITE
        self._file = open(path, "r+b" if not self.cfg.readonly else "rb")
        self._mmap = mmap.mmap(self._file.fileno(), total, access=access)
        logger.debug("ShmChannel '%s' opened (%d bytes)", self.cfg.name, total)

        self._validate()

    def _validate(self) -> None:
        ctrl = self._ring_control()
        if ctrl.magic != RTEL_MAGIC and ctrl.magic != 0:
            logger.warning("ShmChannel '%s': unexpected magic 0x%016X",
                           self.cfg.name, ctrl.magic)

    def _ring_control(self) -> RingControl:
        ctrl = RingControl.from_buffer_copy(self._mmap, 0)
        return ctrl

    def _write_ring_control(self) -> RingControl:
        """Returns a mutable reference into mmap (requires write access)."""
        return RingControl.from_buffer(self._mmap, 0)

    def _slot_offset(self, idx: int) -> int:
        idx = idx & (self.cfg.ring_capacity - 1)
        return RING_CTRL_PADDED + idx * self.cfg.slot_bytes

    def _slot_header(self, idx: int) -> SlotHeader:
        off = self._slot_offset(idx)
        return SlotHeader.from_buffer_copy(self._mmap, off)

    def _payload_offset(self, idx: int) -> int:
        return self._slot_offset(idx) + SLOT_HDR_SIZE

    def payload_capacity(self) -> int:
        return self.cfg.slot_bytes - SLOT_HDR_SIZE

    # -----------------------------------------------------------------------
    # Reader API
    # -----------------------------------------------------------------------

    def current_write_seq(self) -> int:
        ctrl = self._ring_control()
        return ctrl.write_seq

    def new_cursor(self) -> ChannelCursor:
        seq = self.current_write_seq()
        return ChannelCursor(channel_name=self.cfg.name, next_seq=seq)

    def peek(self, cursor: ChannelCursor) -> Optional[SlotHeader]:
        """Return the SlotHeader at cursor if data is ready."""
        idx  = cursor.next_seq & (self.cfg.ring_capacity - 1)
        off  = self._slot_offset(idx)
        # Read sequence atomically (best-effort in Python)
        seq_bytes = self._mmap[off + 8 : off + 16]  # sequence field offset
        seq = struct.unpack_from("<Q", seq_bytes)[0]
        if seq == cursor.next_seq + 1:
            return SlotHeader.from_buffer_copy(self._mmap, off)
        return None

    def consume(self, cursor: ChannelCursor
                ) -> Optional[Tuple[SlotHeader, np.ndarray]]:
        """
        Consume next available slot.
        Returns (SlotHeader, numpy_array) or None if no data.
        The numpy array is a COPY (safe to use after cursor advance).
        """
        hdr = self.peek(cursor)
        if hdr is None:
            return None

        idx = cursor.next_seq & (self.cfg.ring_capacity - 1)
        payload_off = self._payload_offset(idx)
        payload_len = min(
            int(hdr.tensor.payload_bytes),
            self.payload_capacity(),
        )

        # Zero-copy numpy view into mmap
        raw = self._mmap[payload_off : payload_off + payload_len]
        dtype = DTYPE_TO_NUMPY.get(hdr.tensor.dtype, np.uint8)
        arr = np.frombuffer(raw, dtype=dtype).copy()

        # Reshape if tensor has shape info
        if hdr.tensor.ndim > 1:
            shape = tuple(hdr.tensor.shape[i] for i in range(hdr.tensor.ndim))
            try:
                arr = arr.reshape(shape)
            except ValueError:
                pass

        cursor.advance()
        self._stat_consumed     += 1
        self._stat_bytes_read   += payload_len
        return hdr, arr

    def consume_raw_view(self, cursor: ChannelCursor
                         ) -> Optional[Tuple[SlotHeader, memoryview]]:
        """
        Zero-copy consume: returns a memoryview into shared memory.
        Caller MUST finish using the view before the slot is overwritten
        (i.e. before the ring wraps around).
        """
        hdr = self.peek(cursor)
        if hdr is None:
            return None

        idx = cursor.next_seq & (self.cfg.ring_capacity - 1)
        payload_off = self._payload_offset(idx)
        payload_len = min(
            int(hdr.tensor.payload_bytes),
            self.payload_capacity(),
        )
        view = memoryview(self._mmap)[payload_off : payload_off + payload_len]
        cursor.advance()
        return hdr, view

    # -----------------------------------------------------------------------
    # Writer API (for Python writing back to shm)
    # -----------------------------------------------------------------------

    def write(self, data: np.ndarray, tensor_desc: Optional[TensorDescriptor] = None
              ) -> bool:
        """Write numpy array to next slot. Returns True on success."""
        if self.cfg.readonly:
            raise PermissionError("Channel opened in readonly mode")

        raw = data.tobytes()
        if len(raw) > self.payload_capacity():
            logger.error("Data too large: %d > %d", len(raw), self.payload_capacity())
            return False

        # Claim slot
        ctrl = self._write_ring_control()
        seq = ctrl.write_seq
        # Increment write_seq
        struct.pack_into("<Q", self._mmap, 32, seq + 1)  # write_seq offset

        idx = seq & (self.cfg.ring_capacity - 1)
        slot_off = self._slot_offset(idx)

        # Build and write SlotHeader
        hdr = SlotHeader()
        hdr.magic         = RTEL_MAGIC
        hdr.sequence      = seq
        hdr.timestamp_ns  = int(time.time_ns())
        hdr.flags         = SlotHeader.FLAG_VALID
        hdr.schema_ver    = 1

        if tensor_desc is not None:
            hdr.tensor = tensor_desc
        else:
            hdr.tensor.dtype         = DTYPE_FLOAT32 if data.dtype == np.float32 else DTYPE_FLOAT64
            hdr.tensor.ndim          = data.ndim
            for i, s in enumerate(data.shape):
                hdr.tensor.shape[i]  = s
            hdr.tensor.payload_bytes = len(raw)
            hdr.tensor.num_elements  = data.size

        # Write header
        self._mmap[slot_off : slot_off + SLOT_HDR_SIZE] = bytes(hdr)
        # Write payload
        pay_off = slot_off + SLOT_HDR_SIZE
        self._mmap[pay_off : pay_off + len(raw)] = raw
        # Publish: store sequence + 1
        struct.pack_into("<Q", self._mmap, slot_off + 8, seq + 1)

        return True

    def stats(self) -> Dict[str, Any]:
        ctrl = self._ring_control()
        return {
            "name":          self.cfg.name,
            "consumed":      self._stat_consumed,
            "bytes_read":    self._stat_bytes_read,
            "write_seq":     ctrl.write_seq,
            "slot_bytes":    ctrl.slot_bytes,
            "ring_capacity": ctrl.ring_cap,
        }

    def close(self) -> None:
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# LOB Snapshot (Python representation)
# ---------------------------------------------------------------------------
@dataclass
class LobSnapshot:
    """Order book snapshot parsed from RTEL flat f64 format."""
    asset_id:      int = 0
    exchange_ts_ns:int = 0
    sequence:      int = 0
    bids:          List[Tuple[float, float]] = field(default_factory=list)
    asks:          List[Tuple[float, float]] = field(default_factory=list)
    mid_price:     float = 0.0
    spread:        float = 0.0
    bid_imbalance: float = 0.0
    vwap_bid:      float = 0.0
    vwap_ask:      float = 0.0

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "LobSnapshot":
        """Deserialize from flat f64 array (must match Rust/C++ layout)."""
        if arr.ndim > 1:
            arr = arr.flatten()
        snap = cls()
        snap.asset_id       = int(arr[0])
        n_bids              = int(arr[1])
        n_asks              = int(arr[2])
        snap.exchange_ts_ns = int(arr[3])
        snap.sequence       = int(arr[4])

        base_bid_p = 5
        base_bid_s = 5 + MAX_LOB_LEVELS
        base_ask_p = 5 + 2 * MAX_LOB_LEVELS
        base_ask_s = 5 + 3 * MAX_LOB_LEVELS

        snap.bids = [
            (float(arr[base_bid_p + i]), float(arr[base_bid_s + i]))
            for i in range(min(n_bids, MAX_LOB_LEVELS))
        ]
        snap.asks = [
            (float(arr[base_ask_p + i]), float(arr[base_ask_s + i]))
            for i in range(min(n_asks, MAX_LOB_LEVELS))
        ]

        derived_base = 5 + 4 * MAX_LOB_LEVELS
        snap.mid_price      = float(arr[derived_base])
        snap.spread         = float(arr[derived_base + 1])
        snap.bid_imbalance  = float(arr[derived_base + 2])
        snap.vwap_bid       = float(arr[derived_base + 3])
        snap.vwap_ask       = float(arr[derived_base + 4])
        return snap


# ---------------------------------------------------------------------------
# ShmReader — high-level reader managing multiple channels
# ---------------------------------------------------------------------------
class ShmReader:
    """
    High-level interface for reading from multiple RTEL shm-bus channels.
    Provides structured accessors for all standard AETERNUS data types.
    """

    # Standard channel names
    LOB_SNAPSHOT    = "aeternus.chronos.lob"
    VOL_SURFACE     = "aeternus.neuro_sde.vol"
    TENSOR_COMP     = "aeternus.tensornet.compressed"
    GRAPH_ADJ       = "aeternus.omni_graph.adj"
    LUMINA_PRED     = "aeternus.lumina.predictions"
    AGENT_ACTIONS   = "aeternus.hyper_agent.actions"

    def __init__(self, base_path: Path = Path("/tmp"),
                 readonly: bool = True,
                 auto_open: bool = True):
        self._base_path = base_path
        self._readonly  = readonly
        self._channels: Dict[str, ShmChannel] = {}
        self._lock = Lock()

        if auto_open:
            self._try_open_standard_channels()

    def _try_open_standard_channels(self) -> None:
        standard = [
            (self.LOB_SNAPSHOT,  128 * 1024, 512),
            (self.VOL_SURFACE,   64  * 1024, 256),
            (self.TENSOR_COMP,   256 * 1024, 256),
            (self.GRAPH_ADJ,     64  * 1024, 256),
            (self.LUMINA_PRED,   64  * 1024, 512),
            (self.AGENT_ACTIONS, 16  * 1024, 1024),
        ]
        for name, slot_bytes, ring_cap in standard:
            try:
                self.open_channel(name, slot_bytes=slot_bytes, ring_capacity=ring_cap)
            except FileNotFoundError:
                logger.debug("Channel not available (not yet created): %s", name)
            except Exception as e:
                logger.warning("Failed to open channel '%s': %s", name, e)

    def open_channel(self, name: str, slot_bytes: int = DEFAULT_SLOT_BYTES,
                     ring_capacity: int = DEFAULT_RING_CAP) -> ShmChannel:
        with self._lock:
            if name in self._channels:
                return self._channels[name]
            cfg = ChannelConfig(
                name=name,
                slot_bytes=slot_bytes,
                ring_capacity=ring_capacity,
                shm_base_path=self._base_path,
                readonly=self._readonly,
            )
            ch = ShmChannel(cfg)
            self._channels[name] = ch
            return ch

    def channel(self, name: str) -> Optional[ShmChannel]:
        return self._channels.get(name)

    def new_cursor(self, channel_name: str) -> ChannelCursor:
        ch = self._channels.get(channel_name)
        if ch is None:
            return ChannelCursor(channel_name=channel_name, next_seq=1)
        return ch.new_cursor()

    # -----------------------------------------------------------------------
    # Typed readers
    # -----------------------------------------------------------------------

    def read_lob(self, cursor: ChannelCursor) -> Optional[LobSnapshot]:
        """Read next LOB snapshot."""
        ch = self._channels.get(self.LOB_SNAPSHOT)
        if ch is None:
            return None
        result = ch.consume(cursor)
        if result is None:
            return None
        _, arr = result
        try:
            return LobSnapshot.from_array(arr.astype(np.float64))
        except (IndexError, ValueError) as e:
            logger.debug("LOB parse error: %s", e)
            return None

    def read_vol_surface(self, cursor: ChannelCursor
                         ) -> Optional[np.ndarray]:
        """Read next volatility surface (returns [n_strikes, n_expiries] float64 array)."""
        ch = self._channels.get(self.VOL_SURFACE)
        if ch is None:
            return None
        result = ch.consume(cursor)
        if result is None:
            return None
        hdr, arr = result
        shape = hdr.tensor.shape_tuple()
        if len(shape) >= 2:
            try:
                return arr.reshape(shape[-2], shape[-1])
            except ValueError:
                pass
        return arr

    def read_graph_adjacency(self, cursor: ChannelCursor
                              ) -> Optional[Dict[str, np.ndarray]]:
        """Read graph adjacency (returns dict with row_ptr, col_idx, edge_weights)."""
        ch = self._channels.get(self.GRAPH_ADJ)
        if ch is None:
            return None
        result = ch.consume(cursor)
        if result is None:
            return None
        _, arr = result
        # Parse CSR: first 2 values are n_nodes, n_edges
        if len(arr) < 4:
            return None
        arr_f64 = arr.astype(np.float64)
        n_nodes = int(arr_f64[0])
        n_edges = int(arr_f64[1])
        return {
            "n_nodes":     n_nodes,
            "n_edges":     n_edges,
            "raw":         arr_f64,
        }

    def read_lumina_predictions(self, cursor: ChannelCursor
                                 ) -> Optional[np.ndarray]:
        """Read Lumina predictions [3 × MAX_ASSETS] (return, risk, confidence)."""
        ch = self._channels.get(self.LUMINA_PRED)
        if ch is None:
            return None
        result = ch.consume(cursor)
        if result is None:
            return None
        _, arr = result
        return arr.astype(np.float32)

    def read_agent_actions(self, cursor: ChannelCursor
                            ) -> Optional[np.ndarray]:
        """Read HyperAgent position delta actions [MAX_ASSETS] float32."""
        ch = self._channels.get(self.AGENT_ACTIONS)
        if ch is None:
            return None
        result = ch.consume(cursor)
        if result is None:
            return None
        _, arr = result
        return arr.astype(np.float32)

    def read_raw(self, channel_name: str, cursor: ChannelCursor
                 ) -> Optional[Tuple[SlotHeader, np.ndarray]]:
        """Low-level: read raw bytes from any channel."""
        ch = self._channels.get(channel_name)
        if ch is None:
            return None
        return ch.consume(cursor)

    # -----------------------------------------------------------------------
    # Bulk / snapshot readers (read all pending updates at once)
    # -----------------------------------------------------------------------

    def drain_lob(self, cursor: ChannelCursor, max_items: int = 100
                  ) -> List[LobSnapshot]:
        """Drain all available LOB snapshots up to max_items."""
        snaps = []
        for _ in range(max_items):
            snap = self.read_lob(cursor)
            if snap is None:
                break
            snaps.append(snap)
        return snaps

    def read_latest_lob(self, channel_name: Optional[str] = None
                        ) -> Optional[LobSnapshot]:
        """Read the LATEST available LOB snapshot (skips stale items)."""
        ch_name = channel_name or self.LOB_SNAPSHOT
        ch = self._channels.get(ch_name)
        if ch is None:
            return None
        # Create temporary cursor at current write head minus ring capacity
        # to read the latest available slot
        ctrl = ch._ring_control()
        latest_seq = max(1, ctrl.write_seq - 1)
        cur = ChannelCursor(channel_name=ch_name, next_seq=latest_seq)
        for _ in range(ch.cfg.ring_capacity):
            snap = self.read_lob(cur)
            if snap is not None:
                return snap
        return None

    def all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {name: ch.stats() for name, ch in self._channels.items()}

    def close(self) -> None:
        for ch in self._channels.values():
            ch.close()
        self._channels.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"ShmReader(channels={list(self._channels.keys())})"
