"""
AETERNUS Real-Time Execution Layer (RTEL)
feature_store.py — Unified Feature Registry and Store

Named feature arrays with schema (dtype, shape, description).
Versioned feature snapshots for reproducibility.
Time-indexed feature history for lookback windows.
Canonical data interface between all Python modules.

Usage:
    store = FeatureStore()

    # Define feature schemas
    store.register("lob_mid_price", shape=(512,), dtype=np.float64,
                   description="Mid price per asset")
    store.register("vol_atm", shape=(512,), dtype=np.float64,
                   description="ATM implied vol per asset")

    # Update features
    store.update("lob_mid_price", mid_prices)

    # Read features (returns numpy array)
    arr = store.get("lob_mid_price")

    # Get all features as a flat feature vector
    fv = store.feature_vector(["lob_mid_price", "vol_atm"])
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FeatureSchema
# ---------------------------------------------------------------------------
@dataclass
class FeatureSchema:
    """Schema definition for a named feature."""
    name:        str
    shape:       Tuple[int, ...]
    dtype:       np.dtype
    description: str = ""
    group:       str = "default"
    version:     int = 1
    tags:        List[str] = field(default_factory=list)
    min_val:     Optional[float] = None
    max_val:     Optional[float] = None
    units:       str = ""

    @classmethod
    def float32(cls, name: str, shape: Tuple[int, ...],
                description: str = "", **kwargs) -> "FeatureSchema":
        return cls(name, shape, np.dtype(np.float32), description, **kwargs)

    @classmethod
    def float64(cls, name: str, shape: Tuple[int, ...],
                description: str = "", **kwargs) -> "FeatureSchema":
        return cls(name, shape, np.dtype(np.float64), description, **kwargs)

    @classmethod
    def int64(cls, name: str, shape: Tuple[int, ...],
              description: str = "", **kwargs) -> "FeatureSchema":
        return cls(name, shape, np.dtype(np.int64), description, **kwargs)

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def nbytes(self) -> int:
        return self.numel() * self.dtype.itemsize

    def validate(self, arr: np.ndarray) -> bool:
        if arr.shape != self.shape:
            return False
        if arr.dtype != self.dtype:
            return False
        return True


# ---------------------------------------------------------------------------
# FeatureSnapshot — a versioned point-in-time snapshot of all features
# ---------------------------------------------------------------------------
@dataclass
class FeatureSnapshot:
    version:       int
    timestamp_ns:  int
    features:      Dict[str, np.ndarray]
    metadata:      Dict[str, Any] = field(default_factory=dict)
    lob_sequence:  int = 0
    pipeline_id:   int = 0

    def checksum(self) -> str:
        h = hashlib.md5()
        for name in sorted(self.features):
            h.update(name.encode())
            h.update(self.features[name].tobytes())
        return h.hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version":      self.version,
            "timestamp_ns": self.timestamp_ns,
            "checksum":     self.checksum(),
            "features":     {k: v.tolist() for k, v in self.features.items()},
        }

    def __repr__(self) -> str:
        return (f"FeatureSnapshot(v={self.version}, "
                f"t={self.timestamp_ns/1e9:.3f}s, "
                f"n_features={len(self.features)})")


# ---------------------------------------------------------------------------
# FeatureHistory — rolling time-indexed history for one feature
# ---------------------------------------------------------------------------
class FeatureHistory:
    """Circular buffer of (timestamp_ns, value) pairs for a single feature."""

    def __init__(self, schema: FeatureSchema, max_history: int = 1000):
        self.schema      = schema
        self.max_history = max_history
        self._values: deque = deque(maxlen=max_history)
        self._times:  deque = deque(maxlen=max_history)

    def record(self, value: np.ndarray, timestamp_ns: Optional[int] = None) -> None:
        ts = timestamp_ns or time.time_ns()
        self._values.append(value.copy())
        self._times.append(ts)

    def latest(self) -> Optional[np.ndarray]:
        return self._values[-1] if self._values else None

    def latest_ts(self) -> Optional[int]:
        return self._times[-1] if self._times else None

    def at_index(self, idx: int) -> Optional[np.ndarray]:
        """idx=0 is latest, idx=1 is one step back, etc."""
        if idx >= len(self._values):
            return None
        return self._values[-(idx + 1)]

    def window(self, n: int) -> np.ndarray:
        """Get last n values as array [n × *shape]."""
        n = min(n, len(self._values))
        if n == 0:
            return np.zeros((0,) + self.schema.shape, dtype=self.schema.dtype)
        values = [self._values[-(n - i)] for i in range(n)]
        return np.stack(values)

    def timestamps_window(self, n: int) -> np.ndarray:
        n = min(n, len(self._times))
        return np.array(list(self._times)[-n:], dtype=np.int64)

    def count(self) -> int:
        return len(self._values)

    def as_dataframe(self):
        """Convert to pandas DataFrame (if pandas available)."""
        try:
            import pandas as pd
            data = {f"t{self.schema.name}_{i}": [v[i] for v in self._values]
                    for i in range(min(self.schema.numel(), 10))}
            data["timestamp_ns"] = list(self._times)
            return pd.DataFrame(data)
        except ImportError:
            return None


# ---------------------------------------------------------------------------
# FeatureGroup — logically grouped set of features
# ---------------------------------------------------------------------------
@dataclass
class FeatureGroup:
    name:    str
    members: List[str] = field(default_factory=list)
    description: str = ""

    def add(self, feature_name: str) -> None:
        if feature_name not in self.members:
            self.members.append(feature_name)


# ---------------------------------------------------------------------------
# FeatureStore — the main feature registry
# ---------------------------------------------------------------------------
class FeatureStore:
    """
    Unified feature registry for AETERNUS.

    Thread-safe. Supports:
    - Named feature registration with schema validation
    - Versioned snapshots
    - Rolling history per feature
    - Feature groups
    - Flat feature vector construction
    - Checkpointing to disk
    """

    def __init__(self,
                 max_history: int = 1000,
                 snapshot_history: int = 100,
                 validate_on_update: bool = True):
        self._schemas:  Dict[str, FeatureSchema] = {}
        self._data:     Dict[str, np.ndarray]    = {}
        self._history:  Dict[str, FeatureHistory] = {}
        self._groups:   Dict[str, FeatureGroup]  = {}
        self._snapshots: deque = deque(maxlen=snapshot_history)

        self._max_history        = max_history
        self._validate_on_update = validate_on_update
        self._version:   int = 0
        self._lock              = RLock()

        # Register standard AETERNUS features
        self._register_standard_features()

    def _register_standard_features(self) -> None:
        """Register the canonical AETERNUS feature set."""
        from .shm_reader import MAX_ASSETS, MAX_STRIKES, MAX_EXPIRIES

        # LOB features
        lob_features = [
            ("lob_mid_price",    (MAX_ASSETS,), np.float64, "Mid price per asset"),
            ("lob_spread",       (MAX_ASSETS,), np.float64, "Bid-ask spread"),
            ("lob_bid_imbalance",(MAX_ASSETS,), np.float64, "Order book imbalance"),
            ("lob_vwap_bid",     (MAX_ASSETS,), np.float64, "VWAP bid"),
            ("lob_vwap_ask",     (MAX_ASSETS,), np.float64, "VWAP ask"),
            ("lob_bid_depth",    (MAX_ASSETS, 10), np.float64, "Bid depth levels"),
            ("lob_ask_depth",    (MAX_ASSETS, 10), np.float64, "Ask depth levels"),
            ("lob_sequence",     (MAX_ASSETS,), np.int64,   "Sequence numbers"),
        ]
        for name, shape, dtype, desc in lob_features:
            self.register(FeatureSchema(name, shape, np.dtype(dtype), desc,
                                         group="lob"))

        # Volatility features
        vol_features = [
            ("vol_atm",        (MAX_ASSETS,),                 np.float64, "ATM vol"),
            ("vol_surface",    (MAX_ASSETS, MAX_STRIKES, MAX_EXPIRIES),
                                                               np.float64, "Vol surface"),
            ("vol_skew",       (MAX_ASSETS,),                 np.float64, "Vol skew"),
            ("vol_term_slope", (MAX_ASSETS,),                 np.float64, "Vol term structure slope"),
        ]
        for name, shape, dtype, desc in vol_features:
            self.register(FeatureSchema(name, shape, np.dtype(dtype), desc,
                                         group="vol"))

        # Graph features
        graph_features = [
            ("graph_degree",      (512,), np.int32,   "Node degree"),
            ("graph_clustering",  (512,), np.float32, "Clustering coefficient"),
            ("graph_centrality",  (512,), np.float32, "Betweenness centrality"),
            ("graph_corr_matrix", (512, 512), np.float32, "Correlation matrix"),
        ]
        for name, shape, dtype, desc in graph_features:
            self.register(FeatureSchema(name, shape, np.dtype(dtype), desc,
                                         group="graph"))

        # Lumina prediction features
        pred_features = [
            ("lumina_return_forecast", (MAX_ASSETS,), np.float32, "Return forecast"),
            ("lumina_risk_forecast",   (MAX_ASSETS,), np.float32, "Risk forecast"),
            ("lumina_confidence",      (MAX_ASSETS,), np.float32, "Prediction confidence"),
        ]
        for name, shape, dtype, desc in pred_features:
            self.register(FeatureSchema(name, shape, np.dtype(dtype), desc,
                                         group="lumina"))

        # Agent features
        agent_features = [
            ("agent_position_delta", (MAX_ASSETS,), np.float32, "Target position change"),
            ("agent_action_type",    (MAX_ASSETS,), np.int32,   "Action type (0=hold,1=buy,2=sell)"),
            ("agent_confidence",     (1,),          np.float32, "Agent confidence"),
        ]
        for name, shape, dtype, desc in agent_features:
            self.register(FeatureSchema(name, shape, np.dtype(dtype), desc,
                                         group="agent"))

    def register(self, schema: FeatureSchema) -> None:
        """Register a feature schema. Idempotent if schema unchanged."""
        with self._lock:
            existing = self._schemas.get(schema.name)
            if existing is not None and existing != schema:
                logger.warning("Replacing schema for '%s'", schema.name)

            self._schemas[schema.name] = schema
            # Initialize with zeros
            if schema.name not in self._data:
                self._data[schema.name] = np.zeros(schema.shape, dtype=schema.dtype)
            # Initialize history
            if schema.name not in self._history:
                self._history[schema.name] = FeatureHistory(schema, self._max_history)
            # Add to group
            if schema.group not in self._groups:
                self._groups[schema.group] = FeatureGroup(schema.group)
            self._groups[schema.group].add(schema.name)

    def update(self, name: str, value: np.ndarray,
               timestamp_ns: Optional[int] = None,
               validate: bool = True) -> bool:
        """Update a feature value. Returns True if successful."""
        with self._lock:
            schema = self._schemas.get(name)
            if schema is None:
                logger.error("Feature '%s' not registered", name)
                return False

            # Coerce dtype and shape
            try:
                arr = np.asarray(value, dtype=schema.dtype)
                if arr.shape != schema.shape:
                    arr = arr.reshape(schema.shape)
            except (ValueError, TypeError) as e:
                logger.error("Feature '%s' shape/dtype error: %s", name, e)
                return False

            if validate and self._validate_on_update:
                if not schema.validate(arr):
                    logger.error("Feature '%s' validation failed", name)
                    return False

            # Apply value clipping if bounds specified
            if schema.min_val is not None or schema.max_val is not None:
                arr = np.clip(arr, schema.min_val, schema.max_val)

            self._data[name] = arr
            self._history[name].record(arr, timestamp_ns)
            return True

    def get(self, name: str) -> Optional[np.ndarray]:
        """Get the current value of a feature (returns a copy)."""
        with self._lock:
            arr = self._data.get(name)
            return arr.copy() if arr is not None else None

    def get_view(self, name: str) -> Optional[np.ndarray]:
        """Get a read-only view (faster, no copy — caller must not modify)."""
        with self._lock:
            return self._data.get(name)

    def get_history(self, name: str, n: int) -> Optional[np.ndarray]:
        """Get last n historical values as [n × *shape] array."""
        hist = self._history.get(name)
        if hist is None:
            return None
        return hist.window(n)

    def has(self, name: str) -> bool:
        return name in self._schemas

    def schema(self, name: str) -> Optional[FeatureSchema]:
        return self._schemas.get(name)

    def group_features(self, group: str) -> List[str]:
        g = self._groups.get(group)
        return g.members.copy() if g else []

    def feature_vector(self, names: Sequence[str],
                        flatten: bool = True) -> np.ndarray:
        """Concatenate named features into a flat feature vector."""
        parts = []
        for name in names:
            arr = self.get(name)
            if arr is None:
                schema = self._schemas.get(name)
                if schema:
                    arr = np.zeros(schema.shape, dtype=schema.dtype)
                else:
                    logger.warning("Feature '%s' not found", name)
                    continue
            parts.append(arr.flatten() if flatten else arr)
        return np.concatenate(parts) if parts else np.array([])

    def group_vector(self, group: str) -> np.ndarray:
        """Feature vector for all features in a group."""
        return self.feature_vector(self.group_features(group))

    def snapshot(self, pipeline_id: int = 0,
                 lob_seq: int = 0) -> FeatureSnapshot:
        """Create a versioned snapshot of current feature values."""
        with self._lock:
            self._version += 1
            snap = FeatureSnapshot(
                version=self._version,
                timestamp_ns=time.time_ns(),
                features={k: v.copy() for k, v in self._data.items()},
                pipeline_id=pipeline_id,
                lob_sequence=lob_seq,
            )
            self._snapshots.append(snap)
            return snap

    def latest_snapshot(self) -> Optional[FeatureSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def snapshot_at_version(self, version: int) -> Optional[FeatureSnapshot]:
        for snap in reversed(self._snapshots):
            if snap.version == version:
                return snap
        return None

    def list_features(self, group: Optional[str] = None) -> List[str]:
        if group:
            return self.group_features(group)
        return sorted(self._schemas.keys())

    def list_groups(self) -> List[str]:
        return sorted(self._groups.keys())

    def update_from_lob(self, lob_snap) -> None:
        """Update LOB features from a LobSnapshot object."""
        n = 512  # MAX_ASSETS
        mid = np.zeros(n, dtype=np.float64)
        spread = np.zeros(n, dtype=np.float64)
        imbal = np.zeros(n, dtype=np.float64)
        vwap_bid = np.zeros(n, dtype=np.float64)
        vwap_ask = np.zeros(n, dtype=np.float64)
        seq_arr = np.zeros(n, dtype=np.int64)

        ai = lob_snap.asset_id if hasattr(lob_snap, "asset_id") else 0
        mid[ai]     = lob_snap.mid_price
        spread[ai]  = lob_snap.spread
        imbal[ai]   = lob_snap.bid_imbalance
        vwap_bid[ai]= lob_snap.vwap_bid
        vwap_ask[ai]= lob_snap.vwap_ask
        seq_arr[ai] = lob_snap.sequence

        self.update("lob_mid_price",     mid)
        self.update("lob_spread",        spread)
        self.update("lob_bid_imbalance", imbal)
        self.update("lob_vwap_bid",      vwap_bid)
        self.update("lob_vwap_ask",      vwap_ask)
        self.update("lob_sequence",      seq_arr)

    def update_from_predictions(self, returns: np.ndarray,
                                  risks: Optional[np.ndarray] = None,
                                  confidence: Optional[np.ndarray] = None) -> None:
        self.update("lumina_return_forecast", returns)
        if risks is not None:
            self.update("lumina_risk_forecast", risks)
        if confidence is not None:
            self.update("lumina_confidence", confidence)

    def save(self, path: Union[str, Path]) -> None:
        """Checkpoint all features to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "version":  self._version,
                "schemas":  {k: {
                    "shape": s.shape,
                    "dtype": str(s.dtype),
                    "description": s.description,
                    "group": s.group,
                } for k, s in self._schemas.items()},
                "data": {k: v.tolist() for k, v in self._data.items()},
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("FeatureStore saved to %s (%d features)", path, len(self._data))

    def load(self, path: Union[str, Path]) -> None:
        """Restore features from a checkpoint."""
        with open(Path(path)) as f:
            data = json.load(f)
        with self._lock:
            self._version = data.get("version", 0)
            for name, d in data.get("data", {}).items():
                schema = self._schemas.get(name)
                if schema:
                    arr = np.array(d, dtype=schema.dtype).reshape(schema.shape)
                    self._data[name] = arr
        logger.info("FeatureStore loaded from %s", path)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "n_features":   len(self._schemas),
                "n_groups":     len(self._groups),
                "version":      self._version,
                "n_snapshots":  len(self._snapshots),
                "total_bytes":  sum(v.nbytes for v in self._data.values()),
                "groups":       {g: len(grp.members)
                                 for g, grp in self._groups.items()},
            }

    def print_stats(self) -> None:
        s = self.stats()
        print("=== FeatureStore Stats ===")
        print(f"  Features:    {s['n_features']}")
        print(f"  Groups:      {s['n_groups']}")
        print(f"  Version:     {s['version']}")
        print(f"  Snapshots:   {s['n_snapshots']}")
        print(f"  Total bytes: {s['total_bytes']:,}")
        print(f"  Groups: {s['groups']}")

    def __repr__(self) -> str:
        return (f"FeatureStore(n_features={len(self._schemas)}, "
                f"version={self._version})")
