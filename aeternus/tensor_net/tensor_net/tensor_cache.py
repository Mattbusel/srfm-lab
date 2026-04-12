"""
tensor_cache.py — Intelligent tensor caching layer for AETERNUS TT decompositions.

Provides:
  - LRU cache for expensive TT decompositions keyed by (input_hash, compression_params)
  - Staleness tracking: entries expire after N ticks or when input changes by >epsilon
  - GPU tensor cache: keep frequently-used compressed tensors pinned in device memory
  - Cache hit-rate monitoring and eviction policy tuning
  - Warm-up: pre-compute decompositions for common correlation regimes at startup
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, Hashable, Iterator, List,
    Optional, Set, Tuple, TypeVar, Union
)

import numpy as np

logger = logging.getLogger(__name__)

V = TypeVar("V")


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

def _array_hash(arr: np.ndarray, n_bytes: int = 8192) -> str:
    """
    Compute a fast hash of an array.
    For large arrays, hash only the first n_bytes of the raw buffer.
    """
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    flat = arr.ravel()
    byte_view = flat.view(np.uint8)
    h.update(byte_view[:min(n_bytes, byte_view.nbytes)].tobytes())
    return h.hexdigest()[:20]


@dataclass(frozen=True)
class CompressionParams:
    """Hashable compression parameter set used as part of cache keys."""
    max_rank: int = 8
    target_ratio: float = 10.0
    method: str = "tt_svd"
    tol: float = 1e-4

    def to_key(self) -> str:
        return f"{self.method}_r{self.max_rank}_cr{self.target_ratio:.2f}_tol{self.tol:.2e}"


@dataclass(frozen=True)
class CacheKey:
    input_hash: str
    params_key: str
    schema_name: str

    def __str__(self) -> str:
        return f"{self.schema_name}:{self.input_hash}:{self.params_key}"


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry(Generic[V]):
    key: CacheKey
    value: V
    created_tick: int
    created_time: float  # perf_counter
    access_count: int = 0
    last_access_tick: int = 0
    last_access_time: float = 0.0
    input_norm: float = 0.0   # L2 norm of original input for change detection
    nbytes: int = 0

    def touch(self, tick_id: int) -> None:
        self.access_count += 1
        self.last_access_tick = tick_id
        self.last_access_time = time.perf_counter()

    def age_ticks(self, current_tick: int) -> int:
        return current_tick - self.created_tick

    def is_stale(
        self,
        current_tick: int,
        max_age_ticks: int,
        current_input_norm: Optional[float] = None,
        epsilon: float = 0.05,
    ) -> bool:
        if self.age_ticks(current_tick) > max_age_ticks:
            return True
        if current_input_norm is not None:
            rel_change = abs(current_input_norm - self.input_norm) / (self.input_norm + 1e-12)
            if rel_change > epsilon:
                return True
        return False


# ---------------------------------------------------------------------------
# Eviction policies
# ---------------------------------------------------------------------------

class EvictionPolicy(Enum):
    LRU    = auto()   # Least Recently Used
    LFU    = auto()   # Least Frequently Used
    FIFO   = auto()   # First In, First Out
    RANDOM = auto()   # Random eviction


# ---------------------------------------------------------------------------
# LRU Cache implementation
# ---------------------------------------------------------------------------

class LRUTensorCache(Generic[V]):
    """
    Thread-safe LRU cache for tensor computation results.

    Generic over value type V (e.g. TT decomposition result).

    Parameters
    ----------
    max_entries:
        Maximum number of cache entries.
    max_bytes:
        Maximum total bytes of cached values. 0 = unlimited.
    max_age_ticks:
        Entries older than this many ticks are considered stale.
    staleness_epsilon:
        Relative input norm change threshold for staleness.
    eviction_policy:
        Which eviction policy to use when the cache is full.
    """

    def __init__(
        self,
        max_entries: int = 512,
        max_bytes: int = 0,
        max_age_ticks: int = 100,
        staleness_epsilon: float = 0.05,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._max_age_ticks = max_age_ticks
        self._epsilon = staleness_epsilon
        self._policy = eviction_policy
        self._store: OrderedDict[CacheKey, CacheEntry[V]] = OrderedDict()
        self._total_bytes: int = 0
        self._lock = threading.RLock()

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._stale_evictions: int = 0

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #

    def get(
        self,
        key: CacheKey,
        current_tick: int = 0,
        current_input_norm: Optional[float] = None,
    ) -> Optional[V]:
        """Retrieve a cached value, or None on miss / stale entry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_stale(current_tick, self._max_age_ticks, current_input_norm, self._epsilon):
                self._evict_entry(key)
                self._stale_evictions += 1
                self._misses += 1
                return None
            entry.touch(current_tick)
            self._hits += 1
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return entry.value

    def put(
        self,
        key: CacheKey,
        value: V,
        current_tick: int = 0,
        input_norm: float = 0.0,
        nbytes: int = 0,
    ) -> None:
        """Insert a new entry, evicting old ones if needed."""
        with self._lock:
            if key in self._store:
                old = self._store.pop(key)
                self._total_bytes -= old.nbytes

            entry = CacheEntry(
                key=key,
                value=value,
                created_tick=current_tick,
                created_time=time.perf_counter(),
                input_norm=input_norm,
                nbytes=nbytes,
            )
            entry.touch(current_tick)
            self._store[key] = entry
            self._store.move_to_end(key)
            self._total_bytes += nbytes

            # Evict if necessary
            while len(self._store) > self._max_entries or (
                self._max_bytes > 0 and self._total_bytes > self._max_bytes
            ):
                self._evict_one()

    def contains(self, key: CacheKey) -> bool:
        with self._lock:
            return key in self._store

    def invalidate(self, key: CacheKey) -> bool:
        with self._lock:
            if key in self._store:
                self._evict_entry(key)
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._total_bytes = 0

    # ------------------------------------------------------------------ #
    # Eviction internals
    # ------------------------------------------------------------------ #

    def _evict_entry(self, key: CacheKey) -> None:
        entry = self._store.pop(key, None)
        if entry is not None:
            self._total_bytes -= entry.nbytes
            self._evictions += 1

    def _evict_one(self) -> None:
        if not self._store:
            return
        if self._policy == EvictionPolicy.LRU:
            # Least recently used = first in OrderedDict
            key = next(iter(self._store))
        elif self._policy == EvictionPolicy.LFU:
            key = min(self._store, key=lambda k: self._store[k].access_count)
        elif self._policy == EvictionPolicy.FIFO:
            key = min(self._store, key=lambda k: self._store[k].created_time)
        else:  # RANDOM
            import random
            key = random.choice(list(self._store.keys()))
        self._evict_entry(key)

    # ------------------------------------------------------------------ #
    # Staleness sweep
    # ------------------------------------------------------------------ #

    def sweep_stale(
        self,
        current_tick: int,
        input_norms: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Evict all stale entries. Returns number of entries evicted.
        *input_norms* is an optional mapping from schema_name -> current L2 norm.
        """
        stale_keys = []
        with self._lock:
            for key, entry in self._store.items():
                norm = (input_norms or {}).get(key.schema_name)
                if entry.is_stale(current_tick, self._max_age_ticks, norm, self._epsilon):
                    stale_keys.append(key)
            for k in stale_keys:
                self._evict_entry(k)
            self._stale_evictions += len(stale_keys)
        return len(stale_keys)

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "evictions": self._evictions,
            "stale_evictions": self._stale_evictions,
            "total_bytes": self._total_bytes,
            "policy": self._policy.name,
        }

    def tune_policy(self, new_policy: EvictionPolicy) -> None:
        """Switch eviction policy (takes effect on next eviction)."""
        with self._lock:
            self._policy = new_policy
        logger.info("Cache eviction policy changed to %s.", new_policy.name)


# ---------------------------------------------------------------------------
# GPU tensor cache stub
# ---------------------------------------------------------------------------

class GPUTensorCache:
    """
    Cache for compressed tensors in GPU memory.

    This is a stub that falls back to CPU memory when GPU is unavailable.
    In production, replace numpy arrays with JAX DeviceArrays or CuPy arrays.
    """

    def __init__(
        self,
        max_entries: int = 64,
        device_id: int = 0,
    ) -> None:
        self._max_entries = max_entries
        self._device_id = device_id
        self._store: OrderedDict[CacheKey, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self._gpu_available = self._detect_gpu()
        self._hits = 0
        self._misses = 0

        if not self._gpu_available:
            logger.warning(
                "GPU not available; GPUTensorCache will use CPU memory as fallback."
            )

    def _detect_gpu(self) -> bool:
        try:
            import jax
            backends = jax.devices()
            gpu_devs = [d for d in backends if "gpu" in d.device_kind.lower()]
            return len(gpu_devs) > 0
        except Exception:
            return False

    def pin(self, key: CacheKey, arr: np.ndarray) -> None:
        """Upload (or keep) *arr* on GPU memory under *key*."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return
            if self._gpu_available:
                try:
                    import jax
                    import jax.numpy as jnp
                    gpu_arr = jax.device_put(arr)
                    # We store as-is; in practice this would be a DeviceArray
                    self._store[key] = gpu_arr  # type: ignore[assignment]
                except Exception:
                    self._store[key] = arr.copy()
            else:
                self._store[key] = arr.copy()
            self._store.move_to_end(key)
            # Evict if over capacity
            while len(self._store) > self._max_entries:
                evict_key = next(iter(self._store))
                del self._store[evict_key]

    def get(self, key: CacheKey) -> Optional[np.ndarray]:
        with self._lock:
            val = self._store.get(key)
            if val is None:
                self._misses += 1
                return None
            self._hits += 1
            self._store.move_to_end(key)
            return val

    def evict(self, key: CacheKey) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "gpu_available": self._gpu_available,
            "size": len(self._store),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# ---------------------------------------------------------------------------
# Regime-based warm-up
# ---------------------------------------------------------------------------

@dataclass
class RegimeCorrelation:
    """Pre-defined correlation regime for warm-up."""
    name: str
    base_corr: float    # average pairwise correlation
    spread: float       # spread around base_corr
    n_assets: int = 10

    def generate_adjacency(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        adj = np.full((self.n_assets, self.n_assets), self.base_corr, dtype=np.float32)
        noise = rng.uniform(-self.spread, self.spread, (self.n_assets, self.n_assets)).astype(np.float32)
        adj = np.clip(adj + noise, 0.0, 1.0).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        return adj


COMMON_REGIMES: List[RegimeCorrelation] = [
    RegimeCorrelation("crisis",        base_corr=0.85, spread=0.05),
    RegimeCorrelation("normal",        base_corr=0.35, spread=0.15),
    RegimeCorrelation("low_vol",       base_corr=0.20, spread=0.10),
    RegimeCorrelation("sector_rotate", base_corr=0.45, spread=0.25),
    RegimeCorrelation("decorrelated",  base_corr=0.05, spread=0.05),
]


def warmup_cache(
    cache: LRUTensorCache,
    decompose_fn: Callable[[np.ndarray, CompressionParams], Any],
    params: CompressionParams,
    regimes: Optional[List[RegimeCorrelation]] = None,
    n_assets: int = 10,
    tick_id: int = 0,
) -> int:
    """
    Pre-populate cache with decompositions for common correlation regimes.

    Parameters
    ----------
    cache:
        The LRU cache to warm up.
    decompose_fn:
        Function (arr, params) -> decomposed_value to call for each regime.
    params:
        CompressionParams to use for all warm-up decompositions.
    regimes:
        List of RegimeCorrelation objects. Defaults to COMMON_REGIMES.
    n_assets:
        Number of assets for generated adjacency matrices.
    tick_id:
        Starting tick id for cache entries.

    Returns
    -------
    Number of entries inserted.
    """
    regimes = regimes or COMMON_REGIMES
    inserted = 0
    for regime in regimes:
        adj = regime.generate_adjacency()
        h = _array_hash(adj)
        key = CacheKey(
            input_hash=h,
            params_key=params.to_key(),
            schema_name=f"OmniGraphAdjacency_warmup_{regime.name}",
        )
        if not cache.contains(key):
            try:
                value = decompose_fn(adj, params)
                nbytes = _estimate_value_bytes(value)
                norm = float(np.linalg.norm(adj))
                cache.put(key, value, current_tick=tick_id, input_norm=norm, nbytes=nbytes)
                inserted += 1
                logger.debug("Warm-up: inserted regime '%s' into cache.", regime.name)
            except Exception as exc:
                logger.warning("Warm-up failed for regime '%s': %s", regime.name, exc)
    return inserted


def _estimate_value_bytes(value: Any) -> int:
    """Best-effort estimate of bytes used by a cached value."""
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (list, tuple)):
        return sum(_estimate_value_bytes(v) for v in value)
    if hasattr(value, "__dict__"):
        total = 0
        for v in vars(value).values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
            elif isinstance(v, (list, tuple)):
                total += _estimate_value_bytes(v)
        return total
    return 0


# ---------------------------------------------------------------------------
# Composite cache manager
# ---------------------------------------------------------------------------

class TensorCacheManager:
    """
    Composite cache manager combining LRU + GPU cache.

    Workflow:
      1. Check GPU cache (fastest) on get.
      2. Check LRU CPU cache on miss.
      3. On LRU miss, compute decomposition.
      4. Store in both LRU and (if pinnable) GPU cache.
    """

    def __init__(
        self,
        lru_max_entries: int = 512,
        lru_max_bytes: int = 0,
        lru_max_age_ticks: int = 100,
        staleness_epsilon: float = 0.05,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        gpu_max_entries: int = 32,
        device_id: int = 0,
    ) -> None:
        self._lru = LRUTensorCache(
            max_entries=lru_max_entries,
            max_bytes=lru_max_bytes,
            max_age_ticks=lru_max_age_ticks,
            staleness_epsilon=staleness_epsilon,
            eviction_policy=eviction_policy,
        )
        self._gpu = GPUTensorCache(max_entries=gpu_max_entries, device_id=device_id)
        self._decompose_fn: Optional[Callable[[np.ndarray, CompressionParams], Any]] = None
        self._current_tick: int = 0

    def set_decompose_fn(
        self,
        fn: Callable[[np.ndarray, CompressionParams], Any],
    ) -> None:
        self._decompose_fn = fn

    def set_tick(self, tick_id: int) -> None:
        self._current_tick = tick_id

    def get_or_compute(
        self,
        arr: np.ndarray,
        params: CompressionParams,
        schema_name: str = "unknown",
        pin_to_gpu: bool = False,
    ) -> Any:
        """
        Return cached decomposition of *arr* with *params*, computing if needed.

        Parameters
        ----------
        arr:
            Input tensor to decompose.
        params:
            Compression parameters.
        schema_name:
            UTR schema name (for key namespacing).
        pin_to_gpu:
            If True, store result in GPU cache after computation.

        Returns
        -------
        Decomposed value (type depends on decompose_fn).
        """
        h = _array_hash(arr)
        key = CacheKey(input_hash=h, params_key=params.to_key(), schema_name=schema_name)
        norm = float(np.linalg.norm(arr))

        # 1. GPU cache lookup
        gpu_val = self._gpu.get(key)
        if gpu_val is not None:
            return gpu_val

        # 2. LRU CPU cache lookup
        lru_val = self._lru.get(key, current_tick=self._current_tick, current_input_norm=norm)
        if lru_val is not None:
            if pin_to_gpu:
                self._maybe_pin_to_gpu(key, lru_val)
            return lru_val

        # 3. Compute
        if self._decompose_fn is None:
            raise RuntimeError("No decompose_fn set. Call set_decompose_fn() first.")
        value = self._decompose_fn(arr, params)
        nbytes = _estimate_value_bytes(value)
        self._lru.put(key, value, current_tick=self._current_tick, input_norm=norm, nbytes=nbytes)
        if pin_to_gpu:
            self._maybe_pin_to_gpu(key, value)
        return value

    def _maybe_pin_to_gpu(self, key: CacheKey, value: Any) -> None:
        # Only attempt to pin numpy arrays directly
        if isinstance(value, np.ndarray):
            self._gpu.pin(key, value)

    def sweep_stale(self, input_norms: Optional[Dict[str, float]] = None) -> int:
        return self._lru.sweep_stale(self._current_tick, input_norms)

    def warmup(
        self,
        params: CompressionParams,
        regimes: Optional[List[RegimeCorrelation]] = None,
        n_assets: int = 10,
    ) -> int:
        if self._decompose_fn is None:
            logger.warning("Warm-up skipped: no decompose_fn set.")
            return 0
        return warmup_cache(
            self._lru,
            self._decompose_fn,
            params,
            regimes=regimes,
            n_assets=n_assets,
            tick_id=self._current_tick,
        )

    def tune_eviction_policy(self, policy: EvictionPolicy) -> None:
        self._lru.tune_policy(policy)

    def clear_all(self) -> None:
        self._lru.clear()
        self._gpu.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "lru": self._lru.stats(),
            "gpu": self._gpu.stats(),
            "current_tick": self._current_tick,
        }

    def summary(self) -> str:
        lru = self._lru.stats()
        gpu = self._gpu.stats()
        lines = [
            "TensorCacheManager",
            f"  LRU: size={lru['size']}/{lru['max_entries']}  "
            f"hit_rate={lru['hit_rate']:.2%}  "
            f"evictions={lru['evictions']}  "
            f"stale={lru['stale_evictions']}  "
            f"bytes={lru['total_bytes']:,}",
            f"  GPU: size={gpu['size']}/{gpu['max_entries']}  "
            f"hit_rate={gpu['hit_rate']:.2%}  "
            f"available={gpu['gpu_available']}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cached decomposition decorator
# ---------------------------------------------------------------------------

def cached_decomposition(
    cache_manager: TensorCacheManager,
    params: CompressionParams,
    schema_name: str = "unknown",
    pin_to_gpu: bool = False,
) -> Callable[[Callable[[np.ndarray], Any]], Callable[[np.ndarray], Any]]:
    """
    Decorator that wraps a decomposition function with the TensorCacheManager.

    Usage
    -----
    >>> mgr = TensorCacheManager()
    >>> @cached_decomposition(mgr, CompressionParams(max_rank=8), "OmniGraphAdjacency")
    ... def my_decompose(arr: np.ndarray) -> Any:
    ...     ...
    >>> result = my_decompose(my_array)  # transparently cached
    """
    def decorator(fn: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], Any]:
        # Adapt fn to accept (arr, params) as required by get_or_compute
        def _decompose_adapter(arr: np.ndarray, _params: CompressionParams) -> Any:
            return fn(arr)
        cache_manager.set_decompose_fn(_decompose_adapter)

        def wrapper(arr: np.ndarray) -> Any:
            return cache_manager.get_or_compute(
                arr, params, schema_name=schema_name, pin_to_gpu=pin_to_gpu
            )
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Cache hit-rate monitor
# ---------------------------------------------------------------------------

@dataclass
class HitRateSnapshot:
    tick_id: int
    lru_hit_rate: float
    gpu_hit_rate: float
    lru_size: int
    gpu_size: int
    timestamp: float = field(default_factory=time.perf_counter)


class CacheHitRateMonitor:
    """
    Periodic snapshot logger for cache hit rates.
    Call record() once per tick; retrieve history with get_history().
    """

    def __init__(
        self,
        manager: TensorCacheManager,
        snapshot_interval_ticks: int = 50,
    ) -> None:
        self._manager = manager
        self._interval = snapshot_interval_ticks
        self._history: List[HitRateSnapshot] = []
        self._last_snapshot_tick: int = -1

    def record(self, tick_id: int) -> Optional[HitRateSnapshot]:
        """Record a snapshot if the interval has elapsed."""
        if tick_id - self._last_snapshot_tick < self._interval:
            return None
        stats = self._manager.stats()
        snap = HitRateSnapshot(
            tick_id=tick_id,
            lru_hit_rate=stats["lru"]["hit_rate"],
            gpu_hit_rate=stats["gpu"]["hit_rate"],
            lru_size=stats["lru"]["size"],
            gpu_size=stats["gpu"]["size"],
        )
        self._history.append(snap)
        self._last_snapshot_tick = tick_id
        return snap

    def get_history(self) -> List[HitRateSnapshot]:
        return list(self._history)

    def average_lru_hit_rate(self) -> float:
        if not self._history:
            return 0.0
        return float(np.mean([s.lru_hit_rate for s in self._history]))

    def trend(self, last_n: int = 10) -> str:
        """Return 'improving', 'degrading', or 'stable' based on recent snapshots."""
        if len(self._history) < 2:
            return "stable"
        recent = [s.lru_hit_rate for s in self._history[-last_n:]]
        if len(recent) < 2:
            return "stable"
        slope = (recent[-1] - recent[0]) / len(recent)
        if slope > 0.02:
            return "improving"
        if slope < -0.02:
            return "degrading"
        return "stable"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Key helpers
    "_array_hash",
    "CompressionParams",
    "CacheKey",
    "CacheEntry",
    # Eviction
    "EvictionPolicy",
    # Core cache
    "LRUTensorCache",
    "GPUTensorCache",
    # Regime warm-up
    "RegimeCorrelation",
    "COMMON_REGIMES",
    "warmup_cache",
    # Composite manager
    "TensorCacheManager",
    # Decorator
    "cached_decomposition",
    # Monitoring
    "HitRateSnapshot",
    "CacheHitRateMonitor",
]
