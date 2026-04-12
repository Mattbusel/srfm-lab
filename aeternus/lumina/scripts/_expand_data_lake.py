"""Expand data_lake.py with large additions."""
import os

PATH = os.path.join(os.path.dirname(__file__), "..", "lumina", "data_lake.py")

CONTENT = '''

# =============================================================================
# SECTION: Advanced Data Lake Management
# =============================================================================

import os
import json
import hashlib
import datetime
from typing import Optional, List, Dict, Tuple, Any, Iterator
import torch
import numpy as np


class DataCatalog:
    """Catalog for tracking all datasets in the financial data lake.

    Maintains a registry of dataset metadata: schema, provenance,
    quality stats, versioning, and access patterns.
    """

    def __init__(self, catalog_path: str = None):
        self.catalog_path = catalog_path or os.path.join(os.getcwd(), "catalog.json")
        self._entries: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load catalog from disk if it exists."""
        if os.path.exists(self.catalog_path):
            try:
                with open(self.catalog_path, "r") as f:
                    self._entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._entries = {}

    def _save(self):
        """Persist catalog to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.catalog_path)), exist_ok=True)
        with open(self.catalog_path, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)

    def register_dataset(
        self,
        name: str,
        path: str,
        schema: dict,
        description: str = "",
        tags: List[str] = None,
        owner: str = "lumina",
    ) -> dict:
        """Register a new dataset in the catalog."""
        entry = {
            "name": name,
            "path": path,
            "schema": schema,
            "description": description,
            "tags": tags or [],
            "owner": owner,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "version": 1,
            "quality_score": None,
            "n_records": None,
            "size_bytes": None,
        }
        self._entries[name] = entry
        self._save()
        return entry

    def get(self, name: str) -> Optional[dict]:
        """Get dataset entry by name."""
        return self._entries.get(name)

    def search(self, tags: List[str] = None, owner: str = None) -> List[dict]:
        """Search catalog by tags and/or owner."""
        results = list(self._entries.values())
        if tags:
            results = [e for e in results if any(t in e.get("tags", []) for t in tags)]
        if owner:
            results = [e for e in results if e.get("owner") == owner]
        return results

    def update_quality_stats(self, name: str, stats: dict):
        """Update quality statistics for a dataset."""
        if name in self._entries:
            self._entries[name]["quality_score"] = stats.get("quality_score")
            self._entries[name]["n_records"] = stats.get("n_records")
            self._entries[name]["size_bytes"] = stats.get("size_bytes")
            self._entries[name]["updated_at"] = datetime.datetime.utcnow().isoformat()
            self._save()

    def increment_version(self, name: str):
        """Increment dataset version after an update."""
        if name in self._entries:
            self._entries[name]["version"] += 1
            self._entries[name]["updated_at"] = datetime.datetime.utcnow().isoformat()
            self._save()

    def list_all(self) -> List[str]:
        """List all registered dataset names."""
        return list(self._entries.keys())

    def delete(self, name: str):
        """Remove a dataset from the catalog."""
        self._entries.pop(name, None)
        self._save()


class DataLineageTracker:
    """Track data transformation lineage for reproducibility and audit.

    Records:
    - Source datasets
    - Transformation operations applied
    - Output datasets
    - Timestamps and user attribution
    """

    def __init__(self):
        self._graph: Dict[str, dict] = {}

    def record_transform(
        self,
        output_name: str,
        input_names: List[str],
        operation: str,
        params: dict = None,
        user: str = "system",
    ):
        """Record a data transformation event."""
        node = {
            "output": output_name,
            "inputs": input_names,
            "operation": operation,
            "params": params or {},
            "user": user,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        self._graph[output_name] = node

    def get_lineage(self, dataset_name: str, depth: int = 10) -> List[dict]:
        """Get full lineage chain for a dataset up to given depth."""
        lineage = []
        current = dataset_name

        for _ in range(depth):
            if current not in self._graph:
                break
            node = self._graph[current]
            lineage.append(node)
            if not node["inputs"]:
                break
            current = node["inputs"][0]  # Follow first input

        return lineage

    def get_descendants(self, dataset_name: str) -> List[str]:
        """Get all datasets derived from the given dataset."""
        descendants = []
        for name, node in self._graph.items():
            if dataset_name in node["inputs"]:
                descendants.append(name)
                descendants.extend(self.get_descendants(name))
        return list(set(descendants))

    def export_dot(self) -> str:
        """Export lineage graph as DOT format for visualization."""
        lines = ["digraph DataLineage {"]
        for name, node in self._graph.items():
            for inp in node["inputs"]:
                lines.append(f'  "{inp}" -> "{name}" [label="{node["operation"]}"];')
        lines.append("}")
        return "\\n".join(lines)


class PartitionedDataStore:
    """Partitioned storage for time-series financial data.

    Organizes data by (symbol, date, frequency) with efficient
    range queries and lazy loading.

    Directory structure:
    root/
      {symbol}/
        {frequency}/
          {year}/
            {month}/
              data.pt
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        self._cache = {}
        self._cache_maxsize = 256

    def _partition_path(
        self,
        symbol: str,
        frequency: str,
        year: int,
        month: int,
    ) -> str:
        return os.path.join(
            self.root_dir, symbol, frequency,
            str(year), f"{month:02d}"
        )

    def write(
        self,
        symbol: str,
        frequency: str,
        year: int,
        month: int,
        data: torch.Tensor,
        metadata: dict = None,
    ):
        """Write a data partition to disk."""
        partition_dir = self._partition_path(symbol, frequency, year, month)
        os.makedirs(partition_dir, exist_ok=True)
        path = os.path.join(partition_dir, "data.pt")
        payload = {"data": data}
        if metadata:
            payload["metadata"] = metadata
        torch.save(payload, path)

    def read(
        self,
        symbol: str,
        frequency: str,
        year: int,
        month: int,
    ) -> Optional[Tuple[torch.Tensor, dict]]:
        """Read a data partition from disk."""
        path = os.path.join(
            self._partition_path(symbol, frequency, year, month), "data.pt"
        )

        if path in self._cache:
            return self._cache[path]

        if not os.path.exists(path):
            return None

        payload = torch.load(path, map_location="cpu")
        data = payload["data"]
        meta = payload.get("metadata", {})

        # LRU-style cache management
        if len(self._cache) >= self._cache_maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[path] = (data, meta)

        return data, meta

    def read_range(
        self,
        symbol: str,
        frequency: str,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> List[Tuple[torch.Tensor, dict]]:
        """Read all partitions within a date range."""
        results = []
        y, m = start_year, start_month

        while (y, m) <= (end_year, end_month):
            result = self.read(symbol, frequency, y, m)
            if result is not None:
                results.append(result)

            m += 1
            if m > 12:
                m = 1
                y += 1

        return results

    def list_partitions(self, symbol: str, frequency: str) -> List[Tuple[int, int]]:
        """List available (year, month) partitions for a symbol."""
        base = os.path.join(self.root_dir, symbol, frequency)
        if not os.path.exists(base):
            return []

        partitions = []
        for year_dir in sorted(os.listdir(base)):
            try:
                year = int(year_dir)
                month_dir = os.path.join(base, year_dir)
                for month_str in sorted(os.listdir(month_dir)):
                    try:
                        month = int(month_str)
                        data_file = os.path.join(month_dir, month_str, "data.pt")
                        if os.path.exists(data_file):
                            partitions.append((year, month))
                    except (ValueError, NotADirectoryError):
                        pass
            except (ValueError, NotADirectoryError):
                pass

        return partitions


class DataVersionControl:
    """Version control for financial datasets.

    Tracks dataset snapshots, supports branching, and provides
    rollback capabilities for reproducible experiments.
    """

    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.versions_dir = os.path.join(repo_dir, ".versions")
        os.makedirs(self.versions_dir, exist_ok=True)
        self._history: List[dict] = self._load_history()

    def _history_path(self) -> str:
        return os.path.join(self.versions_dir, "history.json")

    def _load_history(self) -> List[dict]:
        if os.path.exists(self._history_path()):
            try:
                with open(self._history_path(), "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_history(self):
        with open(self._history_path(), "w") as f:
            json.dump(self._history, f, indent=2, default=str)

    def _compute_hash(self, data: torch.Tensor) -> str:
        """Compute content hash for a data tensor."""
        return hashlib.sha256(data.numpy().tobytes()).hexdigest()[:16]

    def commit(
        self,
        data: torch.Tensor,
        message: str,
        branch: str = "main",
        author: str = "lumina",
    ) -> str:
        """Commit a new version of dataset data."""
        content_hash = self._compute_hash(data)
        commit_id = f"{branch}_{len(self._history):06d}_{content_hash}"

        # Save snapshot
        snapshot_path = os.path.join(self.versions_dir, f"{commit_id}.pt")
        torch.save(data, snapshot_path)

        # Record commit
        commit = {
            "id": commit_id,
            "branch": branch,
            "message": message,
            "author": author,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "hash": content_hash,
            "snapshot_path": snapshot_path,
        }
        self._history.append(commit)
        self._save_history()

        return commit_id

    def checkout(self, commit_id: str) -> Optional[torch.Tensor]:
        """Load data from a specific commit."""
        for commit in self._history:
            if commit["id"] == commit_id:
                if os.path.exists(commit["snapshot_path"]):
                    return torch.load(commit["snapshot_path"], map_location="cpu")
        return None

    def log(self, branch: str = None, n: int = 10) -> List[dict]:
        """Show recent commit history."""
        commits = self._history
        if branch:
            commits = [c for c in commits if c["branch"] == branch]
        return commits[-n:]

    def diff(self, commit_id_a: str, commit_id_b: str) -> dict:
        """Compute statistics about differences between two commits."""
        a = self.checkout(commit_id_a)
        b = self.checkout(commit_id_b)

        if a is None or b is None:
            return {"error": "One or both commits not found"}

        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "mean_diff": (b.float() - a.float()).mean().item() if a.shape == b.shape else None,
            "max_diff": (b.float() - a.float()).abs().max().item() if a.shape == b.shape else None,
        }


# =============================================================================
# SECTION: Real-Time Data Ingestion
# =============================================================================

class StreamingDataBuffer:
    """Thread-safe buffer for real-time streaming data.

    Accumulates incoming data points and provides batch-level access
    for online learning scenarios.
    """

    def __init__(
        self,
        feature_dim: int,
        max_size: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        self.feature_dim = feature_dim
        self.max_size = max_size
        self.dtype = dtype
        self._buffer = torch.zeros(max_size, feature_dim, dtype=dtype)
        self._labels = torch.zeros(max_size, dtype=dtype)
        self._timestamps = []
        self._head = 0
        self._size = 0

        import threading
        self._lock = threading.Lock()

    def push(self, features: torch.Tensor, label: float = 0.0, timestamp=None):
        """Add one data point to the buffer (circular)."""
        with self._lock:
            idx = self._head % self.max_size
            self._buffer[idx] = features.detach()
            self._labels[idx] = label
            if len(self._timestamps) < self.max_size:
                self._timestamps.append(timestamp or datetime.datetime.utcnow())
            else:
                self._timestamps[idx] = timestamp or datetime.datetime.utcnow()
            self._head += 1
            self._size = min(self._size + 1, self.max_size)

    def get_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get the most recent batch_size data points."""
        with self._lock:
            if self._size < batch_size:
                return None
            if self._head <= self.max_size:
                features = self._buffer[:self._size][-batch_size:]
                labels = self._labels[:self._size][-batch_size:]
            else:
                start = (self._head - batch_size) % self.max_size
                end = self._head % self.max_size
                if end > start:
                    features = self._buffer[start:end]
                    labels = self._labels[start:end]
                else:
                    features = torch.cat([self._buffer[start:], self._buffer[:end]])
                    labels = torch.cat([self._labels[start:], self._labels[:end]])
            return features.clone(), labels.clone()

    def current_size(self) -> int:
        with self._lock:
            return self._size

    def clear(self):
        with self._lock:
            self._head = 0
            self._size = 0


class TickDataProcessor:
    """Process raw tick data (trade-by-trade) into OHLCV bars.

    Supports:
    - Time-based bars (1min, 5min, 1h)
    - Volume bars (equal volume per bar)
    - Dollar bars (equal dollar volume per bar)
    - Tick bars (equal number of trades per bar)
    - Information-ratio bars (Advances Marcos Lopez de Prado)
    """

    def __init__(
        self,
        bar_type: str = "time",
        bar_size: float = 60.0,
    ):
        self.bar_type = bar_type
        self.bar_size = bar_size
        self._pending_trades = []
        self._current_bar_accum = 0.0
        self._bars = []

    def process_tick(self, price: float, volume: float, timestamp: float) -> Optional[dict]:
        """Process one tick and return a completed bar if threshold met."""
        self._pending_trades.append({"price": price, "volume": volume, "ts": timestamp})

        if self.bar_type == "volume":
            self._current_bar_accum += volume
        elif self.bar_type == "dollar":
            self._current_bar_accum += price * volume
        elif self.bar_type == "tick":
            self._current_bar_accum += 1
        elif self.bar_type == "time":
            if len(self._pending_trades) > 1:
                span = timestamp - self._pending_trades[0]["ts"]
                if span >= self.bar_size:
                    return self._finalize_bar()
            return None

        if self._current_bar_accum >= self.bar_size:
            return self._finalize_bar()

        return None

    def _finalize_bar(self) -> dict:
        """Compute OHLCV from pending trades."""
        trades = self._pending_trades
        prices = [t["price"] for t in trades]
        volumes = [t["volume"] for t in trades]

        bar = {
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(volumes),
            "vwap": sum(p * v for p, v in zip(prices, volumes)) / max(sum(volumes), 1e-10),
            "n_trades": len(trades),
            "timestamp": trades[0]["ts"],
        }

        self._pending_trades = []
        self._current_bar_accum = 0.0
        self._bars.append(bar)
        return bar

    def get_bars_as_tensor(self) -> Optional[torch.Tensor]:
        """Convert completed bars to tensor format."""
        if not self._bars:
            return None
        keys = ["open", "high", "low", "close", "volume", "vwap", "n_trades"]
        rows = [[b[k] for k in keys] for b in self._bars]
        return torch.tensor(rows, dtype=torch.float32)


class CorpActionAdjuster:
    """Adjust price/volume data for corporate actions.

    Handles:
    - Stock splits and reverse splits
    - Dividends (cash and stock)
    - Spinoffs and mergers
    - Rights offerings

    Applies backward-adjusted prices using cumulative adjustment factors.
    """

    def __init__(self):
        self._actions: Dict[str, List[dict]] = {}

    def record_split(self, symbol: str, ex_date: str, ratio: float):
        """Record a stock split (ratio = new_shares / old_shares)."""
        self._actions.setdefault(symbol, []).append({
            "type": "split",
            "ex_date": ex_date,
            "factor": 1.0 / ratio,
            "volume_factor": ratio,
        })
        self._actions[symbol].sort(key=lambda x: x["ex_date"])

    def record_dividend(self, symbol: str, ex_date: str, amount: float, price_on_ex: float):
        """Record a cash dividend adjustment factor."""
        factor = 1.0 - amount / max(price_on_ex, 1e-10)
        self._actions.setdefault(symbol, []).append({
            "type": "dividend",
            "ex_date": ex_date,
            "factor": factor,
            "volume_factor": 1.0,
        })
        self._actions[symbol].sort(key=lambda x: x["ex_date"])

    def get_adjustment_factor(
        self,
        symbol: str,
        price_date: str,
        reference_date: str,
    ) -> Tuple[float, float]:
        """Compute cumulative price and volume adjustment factors.

        Returns (price_factor, volume_factor) to apply to prices/volumes
        at price_date to make them comparable to reference_date.
        """
        actions = self._actions.get(symbol, [])
        price_factor = 1.0
        volume_factor = 1.0

        for action in actions:
            if price_date <= action["ex_date"] <= reference_date:
                price_factor *= action["factor"]
                volume_factor *= action["volume_factor"]

        return price_factor, volume_factor

    def adjust_series(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        dates: List[str],
        reference_date: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply adjustment factors to entire price/volume series."""
        adj_prices = prices.copy().astype(float)
        adj_volumes = volumes.copy().astype(float)

        for i, date in enumerate(dates):
            pf, vf = self.get_adjustment_factor(symbol, date, reference_date)
            adj_prices[i] *= pf
            adj_volumes[i] *= vf

        return adj_prices, adj_volumes


# =============================================================================
# SECTION: Feature Store
# =============================================================================

class FeatureStore:
    """Centralized feature store for financial ML features.

    Provides:
    - Feature materialization and caching
    - Point-in-time correct feature retrieval
    - Feature drift monitoring
    - Online/offline serving with consistent logic
    """

    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self._feature_defs: Dict[str, dict] = {}
        self._feature_cache: Dict[str, torch.Tensor] = {}

    def register_feature(
        self,
        name: str,
        compute_fn,
        inputs: List[str],
        dtype: torch.dtype = torch.float32,
        description: str = "",
    ):
        """Register a feature definition."""
        self._feature_defs[name] = {
            "name": name,
            "compute_fn": compute_fn,
            "inputs": inputs,
            "dtype": dtype,
            "description": description,
        }

    def materialize(self, feature_names: List[str], data: dict) -> Dict[str, torch.Tensor]:
        """Compute and cache feature values for given data."""
        results = {}

        for fname in feature_names:
            if fname in self._feature_cache:
                results[fname] = self._feature_cache[fname]
                continue

            if fname not in self._feature_defs:
                raise KeyError(f"Feature '{fname}' not registered")

            defn = self._feature_defs[fname]
            input_data = {k: data[k] for k in defn["inputs"] if k in data}
            result = defn["compute_fn"](**input_data)

            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, dtype=defn["dtype"])

            self._feature_cache[fname] = result
            results[fname] = result

        return results

    def save_feature(self, name: str, values: torch.Tensor, timestamp: str = None):
        """Save materialized feature values to disk."""
        ts = timestamp or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.store_dir, f"{name}_{ts}.pt")
        torch.save({"feature": name, "values": values, "timestamp": ts}, path)

    def load_feature(self, name: str, timestamp: str = None) -> Optional[torch.Tensor]:
        """Load feature values from disk."""
        import glob

        if timestamp:
            path = os.path.join(self.store_dir, f"{name}_{timestamp}.pt")
        else:
            # Load most recent
            paths = sorted(glob.glob(os.path.join(self.store_dir, f"{name}_*.pt")))
            if not paths:
                return None
            path = paths[-1]

        if not os.path.exists(path):
            return None

        payload = torch.load(path, map_location="cpu")
        return payload["values"]

    def monitor_drift(
        self,
        feature_name: str,
        reference: torch.Tensor,
        current: torch.Tensor,
        method: str = "psi",
    ) -> dict:
        """Monitor feature distribution drift.

        Methods:
        - psi: Population Stability Index
        - ks: Kolmogorov-Smirnov test statistic
        - wasserstein: 1-Wasserstein distance
        """
        ref = reference.float().numpy()
        cur = current.float().numpy()

        if method == "psi":
            n_bins = 10
            ref_hist, bin_edges = np.histogram(ref, bins=n_bins)
            cur_hist, _ = np.histogram(cur, bins=bin_edges)

            ref_pct = ref_hist / (ref_hist.sum() + 1e-10)
            cur_pct = cur_hist / (cur_hist.sum() + 1e-10)

            # Avoid log(0)
            ref_pct = np.clip(ref_pct, 1e-6, None)
            cur_pct = np.clip(cur_pct, 1e-6, None)

            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return {
                "method": "psi",
                "value": float(psi),
                "alert": psi > 0.2,
                "feature": feature_name,
            }

        elif method == "ks":
            from scipy import stats
            ks_stat, p_value = stats.ks_2samp(ref, cur)
            return {
                "method": "ks",
                "value": float(ks_stat),
                "p_value": float(p_value),
                "alert": p_value < 0.05,
                "feature": feature_name,
            }

        else:
            # Wasserstein distance
            sorted_ref = np.sort(ref)
            sorted_cur = np.sort(cur)
            n = min(len(sorted_ref), len(sorted_cur))
            dist = float(np.abs(sorted_ref[:n] - sorted_cur[:n]).mean())
            return {
                "method": "wasserstein",
                "value": dist,
                "feature": feature_name,
            }


# =============================================================================
# SECTION: Data Pipeline Orchestration
# =============================================================================

class DataPipelineStep:
    """A single step in a data pipeline DAG."""

    def __init__(
        self,
        name: str,
        fn,
        inputs: List[str],
        outputs: List[str],
        retries: int = 3,
        timeout_s: int = 300,
    ):
        self.name = name
        self.fn = fn
        self.inputs = inputs
        self.outputs = outputs
        self.retries = retries
        self.timeout_s = timeout_s
        self._status = "pending"
        self._error = None
        self._result = None

    def execute(self, context: dict) -> dict:
        """Execute this step with retry logic."""
        for attempt in range(self.retries):
            try:
                input_data = {k: context[k] for k in self.inputs if k in context}
                result = self.fn(**input_data)
                self._status = "completed"
                self._result = result
                if isinstance(result, dict):
                    return result
                elif len(self.outputs) == 1:
                    return {self.outputs[0]: result}
                return {}
            except Exception as e:
                self._error = str(e)
                if attempt == self.retries - 1:
                    self._status = "failed"
                    raise

        return {}

    @property
    def status(self) -> str:
        return self._status


class DataPipelineDAG:
    """DAG-based data pipeline for financial data processing.

    Steps execute in topological order. Failed steps are retried.
    Results are stored in a shared context dictionary.
    """

    def __init__(self, name: str):
        self.name = name
        self.steps: Dict[str, DataPipelineStep] = {}
        self._context: dict = {}

    def add_step(self, step: DataPipelineStep):
        """Add a step to the pipeline."""
        self.steps[step.name] = step

    def _topological_sort(self) -> List[str]:
        """Return steps in dependency order."""
        visited = set()
        order = []

        def dfs(name):
            if name in visited:
                return
            visited.add(name)
            step = self.steps[name]
            for inp in step.inputs:
                # Find steps that produce this input
                for other_name, other_step in self.steps.items():
                    if inp in other_step.outputs:
                        dfs(other_name)
            order.append(name)

        for name in self.steps:
            dfs(name)

        return order

    def run(self, initial_context: dict = None) -> dict:
        """Execute all pipeline steps in topological order."""
        self._context = dict(initial_context or {})
        order = self._topological_sort()

        results = {
            "pipeline": self.name,
            "steps": {},
            "final_context_keys": [],
        }

        for step_name in order:
            step = self.steps[step_name]
            try:
                step_results = step.execute(self._context)
                self._context.update(step_results)
                results["steps"][step_name] = {"status": "completed"}
            except Exception as e:
                results["steps"][step_name] = {"status": "failed", "error": str(e)}
                break

        results["final_context_keys"] = list(self._context.keys())
        return results

    def get_output(self, key: str):
        """Get a specific output from the pipeline context."""
        return self._context.get(key)


# =============================================================================
# SECTION: Temporal Join and Asof Merge
# =============================================================================

class TemporalJoiner:
    """Join datasets on timestamps using asof (last-value-carry-forward) semantics.

    Essential for point-in-time correct feature assembly in financial ML.
    """

    @staticmethod
    def asof_join(
        left_timestamps: List[float],
        left_values: np.ndarray,
        right_timestamps: List[float],
        right_values: np.ndarray,
        tolerance: float = None,
    ) -> np.ndarray:
        """For each left timestamp, find the most recent right value.

        Args:
            left_timestamps: sorted list of timestamps for query points
            left_values: [N_left, D] values for left dataset (unused, returned)
            right_timestamps: sorted list of timestamps for right dataset
            right_values: [N_right, D_right] values for right dataset
            tolerance: maximum allowable staleness (in same units as timestamps)

        Returns:
            [N_left, D_right] right values aligned to left timestamps
        """
        right_ts = np.array(right_timestamps)
        result = np.full((len(left_timestamps), right_values.shape[1]), np.nan)

        for i, ts in enumerate(left_timestamps):
            # Find rightmost right_timestamp <= ts
            idx = np.searchsorted(right_ts, ts, side="right") - 1
            if idx >= 0:
                if tolerance is None or (ts - right_ts[idx]) <= tolerance:
                    result[i] = right_values[idx]

        return result

    @staticmethod
    def point_in_time_correct_join(
        event_dates: List[str],
        feature_release_dates: List[str],
        feature_values: np.ndarray,
    ) -> np.ndarray:
        """Enforce point-in-time correctness for fundamental data.

        Only uses features that were available (released) before each event date.
        """
        result = np.full((len(event_dates), feature_values.shape[1]), np.nan)

        for i, event_date in enumerate(event_dates):
            # Find features released before event date
            valid_mask = [d < event_date for d in feature_release_dates]
            valid_indices = [j for j, v in enumerate(valid_mask) if v]

            if valid_indices:
                # Use most recently released features
                latest_idx = max(valid_indices, key=lambda j: feature_release_dates[j])
                result[i] = feature_values[latest_idx]

        return result


# =============================================================================
# SECTION: Dataset Quality Monitor
# =============================================================================

class OnlineDataQualityMonitor:
    """Monitor data quality in real-time streaming scenarios.

    Tracks:
    - Missing value rates
    - Out-of-range values
    - Distribution shifts via running statistics
    - Duplicate detection
    - Temporal gap anomalies
    """

    def __init__(
        self,
        n_features: int,
        expected_ranges: Dict[int, Tuple[float, float]] = None,
        max_gap_seconds: float = 300.0,
    ):
        self.n_features = n_features
        self.expected_ranges = expected_ranges or {}
        self.max_gap_seconds = max_gap_seconds

        # Running statistics (Welford's online algorithm)
        self._n = 0
        self._mean = np.zeros(n_features)
        self._M2 = np.zeros(n_features)
        self._missing_count = np.zeros(n_features)
        self._range_violation_count = np.zeros(n_features)
        self._last_timestamp = None
        self._gap_violations = 0
        self._duplicate_hashes = set()
        self._total_records = 0

    def update(
        self,
        x: np.ndarray,
        timestamp: float = None,
    ) -> dict:
        """Process one data record and return quality metrics."""
        assert len(x) == self.n_features
        self._total_records += 1

        # Hash for duplicate detection
        row_hash = hashlib.md5(x.tobytes()).hexdigest()
        is_duplicate = row_hash in self._duplicate_hashes
        self._duplicate_hashes.add(row_hash)

        # Temporal gap check
        gap_violation = False
        if timestamp is not None and self._last_timestamp is not None:
            gap = timestamp - self._last_timestamp
            if gap > self.max_gap_seconds:
                self._gap_violations += 1
                gap_violation = True
        self._last_timestamp = timestamp

        # Feature-wise checks
        missing = np.isnan(x)
        self._missing_count += missing.astype(float)

        range_violations = np.zeros(self.n_features, dtype=bool)
        for feat_idx, (lo, hi) in self.expected_ranges.items():
            if not missing[feat_idx]:
                if x[feat_idx] < lo or x[feat_idx] > hi:
                    range_violations[feat_idx] = True
                    self._range_violation_count[feat_idx] += 1

        # Update running statistics (Welford)
        valid_mask = ~missing
        self._n += 1
        if valid_mask.any():
            delta = np.where(valid_mask, x - self._mean, 0)
            self._mean += delta / self._n
            delta2 = np.where(valid_mask, x - self._mean, 0)
            self._M2 += delta * delta2

        return {
            "missing": missing.tolist(),
            "range_violations": range_violations.tolist(),
            "is_duplicate": is_duplicate,
            "gap_violation": gap_violation,
            "record_id": self._total_records,
        }

    def summary(self) -> dict:
        """Return quality summary statistics."""
        variance = self._M2 / max(self._n - 1, 1)
        return {
            "total_records": self._total_records,
            "missing_rate": (self._missing_count / max(self._total_records, 1)).tolist(),
            "range_violation_rate": (self._range_violation_count / max(self._total_records, 1)).tolist(),
            "gap_violations": self._gap_violations,
            "duplicate_count": len(self._duplicate_hashes),
            "running_mean": self._mean.tolist(),
            "running_std": np.sqrt(variance).tolist(),
        }
'''

with open(PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess, sys
result = subprocess.run(
    [sys.executable, "-c", f"lines = open(r'{PATH}').readlines(); print(len(lines))"],
    capture_output=True, text=True
)
print(result.stdout.strip(), PATH)
