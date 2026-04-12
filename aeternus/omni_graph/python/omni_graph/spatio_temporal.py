"""
spatio_temporal.py
==================
Spatio-temporal graph sampling for dynamic financial GNNs.

Implements:
  - GraphSAINT random-walk sampler
  - ClusterGCN partition sampler
  - Temporal snapshot sampling
  - Sliding window graph batches
  - Mini-batch construction for dynamic graphs
  - Neighbourhood sampling with temporal constraints
"""

from __future__ import annotations

import math
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.utils import (
        subgraph, k_hop_subgraph, to_undirected,
        remove_self_loops, add_self_loops,
        degree as pyg_degree,
    )
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("torch_geometric not found; some samplers will be limited.")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ---------------------------------------------------------------------------
# Data containers for sampled subgraphs
# ---------------------------------------------------------------------------

@dataclass
class SampledSubgraph:
    """
    Represents a subgraph sampled from the full financial graph.
    """
    node_ids: Tensor              # global node IDs in the sampled subgraph
    edge_index: Tensor            # local edge index (0-indexed within subgraph)
    edge_attr: Optional[Tensor]   # edge features
    node_attr: Optional[Tensor]   # node features
    t_start: int = 0
    t_end: int = 0
    sampler_type: str = "unknown"


@dataclass
class TemporalBatch:
    """
    A batch of temporal graph snapshots.
    """
    snapshots: List[Data]
    timestamps: List[int]
    batch_size: int = 0

    def __post_init__(self):
        self.batch_size = len(self.snapshots)


# ---------------------------------------------------------------------------
# Base sampler
# ---------------------------------------------------------------------------

class BaseSampler:
    """Abstract base for graph samplers."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

    def sample(self, *args, **kwargs) -> SampledSubgraph:
        raise NotImplementedError

    def _local_edge_index(
        self,
        global_node_ids: Tensor,
        global_edge_index: Tensor,
        global_edge_attr: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Extract subgraph edge_index for a set of global node IDs.
        Returns local edge_index and corresponding edge_attr.
        """
        node_set = set(global_node_ids.tolist())
        mask = torch.zeros(global_edge_index.shape[1], dtype=torch.bool)

        for k in range(global_edge_index.shape[1]):
            s, d = int(global_edge_index[0, k]), int(global_edge_index[1, k])
            if s in node_set and d in node_set:
                mask[k] = True

        sub_ei = global_edge_index[:, mask]

        # Remap to local indices
        id_map = {int(v): i for i, v in enumerate(global_node_ids.tolist())}
        local_src = torch.tensor([id_map[int(v)] for v in sub_ei[0]], dtype=torch.long)
        local_dst = torch.tensor([id_map[int(v)] for v in sub_ei[1]], dtype=torch.long)
        local_ei = torch.stack([local_src, local_dst], dim=0)

        local_ea = global_edge_attr[mask] if global_edge_attr is not None else None
        return local_ei, local_ea


# ---------------------------------------------------------------------------
# GraphSAINT random-walk sampler
# ---------------------------------------------------------------------------

class GraphSAINTRandomWalkSampler(BaseSampler):
    """
    GraphSAINT sampler using random walks.

    Samples a connected subgraph by performing multiple random walks
    starting from random root nodes.

    Reference: Zeng et al., "GraphSAINT: Graph Sampling Based Inductive
    Learning Method", ICLR 2020.
    """

    def __init__(
        self,
        walk_length: int = 10,
        n_walks: int = 20,
        restart_prob: float = 0.1,
        normalise_loss: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.walk_length = walk_length
        self.n_walks = n_walks
        self.restart_prob = restart_prob
        self.normalise_loss = normalise_loss

    def sample(
        self,
        edge_index: Tensor,
        node_attr: Optional[Tensor],
        edge_attr: Optional[Tensor],
        num_nodes: int,
        root_nodes: Optional[Tensor] = None,
    ) -> SampledSubgraph:
        """
        Parameters
        ----------
        edge_index : (2, E) global edge index
        node_attr  : (N, F) global node features
        edge_attr  : (E, FE) global edge features
        num_nodes  : N
        root_nodes : optional seed nodes for walks
        """
        adj = self._build_adjacency(edge_index, num_nodes)
        visited = set()

        for _ in range(self.n_walks):
            if root_nodes is not None:
                root = int(root_nodes[self.rng.integers(len(root_nodes))])
            else:
                root = int(self.rng.integers(num_nodes))

            node = root
            for _ in range(self.walk_length):
                visited.add(node)
                if self.rng.random() < self.restart_prob:
                    node = root
                    continue
                neighbours = adj.get(node, [])
                if not neighbours:
                    break
                node = int(self.rng.choice(neighbours))

        sampled_ids = torch.tensor(sorted(visited), dtype=torch.long)

        # Extract subgraph
        local_ei, local_ea = self._local_edge_index(sampled_ids, edge_index, edge_attr)

        # Extract node features
        local_nf = node_attr[sampled_ids] if node_attr is not None else None

        # Importance weights for loss normalisation (node sampling probability)
        node_weights = None
        if self.normalise_loss:
            total_visited_count = len(visited)
            node_weights = torch.ones(len(sampled_ids)) / (total_visited_count + 1e-8)

        subgraph = SampledSubgraph(
            node_ids=sampled_ids,
            edge_index=local_ei,
            edge_attr=local_ea,
            node_attr=local_nf,
            sampler_type="graphsaint_rw",
        )
        return subgraph

    def _build_adjacency(
        self, edge_index: Tensor, num_nodes: int
    ) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = defaultdict(list)
        for k in range(edge_index.shape[1]):
            s, d = int(edge_index[0, k]), int(edge_index[1, k])
            adj[s].append(d)
        return dict(adj)

    def iter_batches(
        self,
        edge_index: Tensor,
        node_attr: Optional[Tensor],
        edge_attr: Optional[Tensor],
        num_nodes: int,
        n_batches: int = 10,
    ) -> Iterator[SampledSubgraph]:
        """Yield n_batches sampled subgraphs."""
        for _ in range(n_batches):
            yield self.sample(edge_index, node_attr, edge_attr, num_nodes)


# ---------------------------------------------------------------------------
# GraphSAINT node sampler
# ---------------------------------------------------------------------------

class GraphSAINTNodeSampler(BaseSampler):
    """
    GraphSAINT node sampler: sample nodes with probability proportional
    to their degree (or uniform), then extract induced subgraph.
    """

    def __init__(
        self,
        budget: int = 100,
        use_degree_prob: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.budget = budget
        self.use_degree_prob = use_degree_prob

    def sample(
        self,
        edge_index: Tensor,
        node_attr: Optional[Tensor],
        edge_attr: Optional[Tensor],
        num_nodes: int,
    ) -> SampledSubgraph:
        budget = min(self.budget, num_nodes)

        if self.use_degree_prob:
            # Compute degree-based sampling probability
            deg = torch.zeros(num_nodes, dtype=torch.float32)
            src = edge_index[0]
            deg.scatter_add_(0, src, torch.ones(src.shape[0]))
            deg = deg + 1.0  # avoid zero prob
            prob = (deg / deg.sum()).numpy()
        else:
            prob = np.ones(num_nodes) / num_nodes

        sampled = self.rng.choice(num_nodes, size=budget, replace=False, p=prob)
        sampled_ids = torch.tensor(sorted(sampled.tolist()), dtype=torch.long)

        local_ei, local_ea = self._local_edge_index(sampled_ids, edge_index, edge_attr)
        local_nf = node_attr[sampled_ids] if node_attr is not None else None

        return SampledSubgraph(
            node_ids=sampled_ids,
            edge_index=local_ei,
            edge_attr=local_ea,
            node_attr=local_nf,
            sampler_type="graphsaint_node",
        )


# ---------------------------------------------------------------------------
# GraphSAINT edge sampler
# ---------------------------------------------------------------------------

class GraphSAINTEdgeSampler(BaseSampler):
    """
    GraphSAINT edge sampler: sample a fixed number of edges, then
    keep all endpoint nodes.
    """

    def __init__(
        self,
        budget: int = 200,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.budget = budget

    def sample(
        self,
        edge_index: Tensor,
        node_attr: Optional[Tensor],
        edge_attr: Optional[Tensor],
        num_nodes: int,
    ) -> SampledSubgraph:
        E = edge_index.shape[1]
        budget = min(self.budget, E)

        chosen = self.rng.choice(E, size=budget, replace=False)
        chosen_t = torch.tensor(chosen, dtype=torch.long)

        sub_ei = edge_index[:, chosen_t]
        sub_ea = edge_attr[chosen_t] if edge_attr is not None else None

        # Collect unique nodes
        unique_nodes = torch.unique(sub_ei)
        id_map = {int(v): i for i, v in enumerate(unique_nodes.tolist())}

        local_src = torch.tensor([id_map[int(v)] for v in sub_ei[0]], dtype=torch.long)
        local_dst = torch.tensor([id_map[int(v)] for v in sub_ei[1]], dtype=torch.long)
        local_ei = torch.stack([local_src, local_dst], dim=0)

        local_nf = node_attr[unique_nodes] if node_attr is not None else None

        return SampledSubgraph(
            node_ids=unique_nodes,
            edge_index=local_ei,
            edge_attr=sub_ea,
            node_attr=local_nf,
            sampler_type="graphsaint_edge",
        )


# ---------------------------------------------------------------------------
# ClusterGCN sampler
# ---------------------------------------------------------------------------

class ClusterGCNSampler(BaseSampler):
    """
    ClusterGCN: partition the graph via METIS-style spectral clustering,
    then sample one or more clusters per mini-batch.

    Since METIS is not always available, falls back to Louvain/modularity
    clustering via networkx.

    Reference: Chiang et al., "Cluster-GCN", KDD 2019.
    """

    def __init__(
        self,
        n_parts: int = 10,
        n_parts_per_batch: int = 2,
        use_metis: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.n_parts = n_parts
        self.n_parts_per_batch = n_parts_per_batch
        self.use_metis = use_metis
        self._partitions: Optional[List[List[int]]] = None

    def partition(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> List[List[int]]:
        """
        Partition the graph into self.n_parts clusters.
        Returns list of lists of node IDs.
        """
        if self.use_metis:
            return self._metis_partition(edge_index, num_nodes)
        else:
            return self._spectral_partition(edge_index, num_nodes)

    def _spectral_partition(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> List[List[int]]:
        """Spectral clustering using graph Laplacian eigenvectors."""
        # Build adjacency matrix (sparse-like via numpy)
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            A[i, j] = 1.0

        # Symmetric normalised Laplacian
        deg = A.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(deg) + 1e-8))
        L = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

        k = min(self.n_parts, num_nodes - 1)
        _, evecs = np.linalg.eigh(L)
        embedding = evecs[:, 1 : k + 1]  # skip trivial eigenvector

        # K-means clustering on eigenvectors
        partitions = self._kmeans(embedding, k)
        return partitions

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 100) -> List[List[int]]:
        """Simple k-means implementation."""
        n = X.shape[0]
        k = min(k, n)
        # Random initialisation
        centres = X[self.rng.choice(n, size=k, replace=False)]
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign
            dists = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            # Update centres
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    centres[c] = X[mask].mean(axis=0)

        partitions = [[] for _ in range(k)]
        for node, lbl in enumerate(labels):
            partitions[lbl].append(node)
        return [p for p in partitions if p]

    def _metis_partition(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> List[List[int]]:
        """Use pymetis for graph partitioning if available."""
        try:
            import pymetis
        except ImportError:
            warnings.warn("pymetis not found; falling back to spectral partition.")
            return self._spectral_partition(edge_index, num_nodes)

        adj_list = [[] for _ in range(num_nodes)]
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if j not in adj_list[i]:
                adj_list[i].append(j)

        _, membership = pymetis.part_graph(self.n_parts, adjacency=adj_list)
        partitions: Dict[int, List[int]] = defaultdict(list)
        for node, part in enumerate(membership):
            partitions[part].append(node)
        return list(partitions.values())

    def sample(
        self,
        edge_index: Tensor,
        node_attr: Optional[Tensor],
        edge_attr: Optional[Tensor],
        num_nodes: int,
    ) -> SampledSubgraph:
        if self._partitions is None:
            self._partitions = self.partition(edge_index, num_nodes)

        n_parts = len(self._partitions)
        chosen_parts = self.rng.choice(
            n_parts,
            size=min(self.n_parts_per_batch, n_parts),
            replace=False,
        )
        sampled_nodes: List[int] = []
        for p in chosen_parts:
            sampled_nodes.extend(self._partitions[p])
        sampled_nodes = sorted(set(sampled_nodes))

        sampled_ids = torch.tensor(sampled_nodes, dtype=torch.long)
        local_ei, local_ea = self._local_edge_index(sampled_ids, edge_index, edge_attr)
        local_nf = node_attr[sampled_ids] if node_attr is not None else None

        return SampledSubgraph(
            node_ids=sampled_ids,
            edge_index=local_ei,
            edge_attr=local_ea,
            node_attr=local_nf,
            sampler_type="cluster_gcn",
        )

    def iter_epoch(
        self,
        edge_index: Tensor,
        node_attr: Optional[Tensor],
        edge_attr: Optional[Tensor],
        num_nodes: int,
    ) -> Iterator[SampledSubgraph]:
        """Iterate over all clusters (one epoch)."""
        if self._partitions is None:
            self._partitions = self.partition(edge_index, num_nodes)

        parts = list(range(len(self._partitions)))
        self.rng.shuffle(parts)

        for start in range(0, len(parts), self.n_parts_per_batch):
            batch_parts = parts[start : start + self.n_parts_per_batch]
            sampled_nodes: List[int] = []
            for p in batch_parts:
                sampled_nodes.extend(self._partitions[p])
            sampled_nodes = sorted(set(sampled_nodes))
            sampled_ids = torch.tensor(sampled_nodes, dtype=torch.long)
            local_ei, local_ea = self._local_edge_index(sampled_ids, edge_index, edge_attr)
            local_nf = node_attr[sampled_ids] if node_attr is not None else None
            yield SampledSubgraph(
                node_ids=sampled_ids,
                edge_index=local_ei,
                edge_attr=local_ea,
                node_attr=local_nf,
                sampler_type="cluster_gcn",
            )


# ---------------------------------------------------------------------------
# Temporal snapshot sampler
# ---------------------------------------------------------------------------

class TemporalSnapshotSampler:
    """
    Generate temporal snapshots of a dynamic graph.

    Operates on a list of (edge_index, edge_attr, node_attr) tuples,
    one per time step. Produces fixed-size windows suitable for
    temporal GNN training.
    """

    def __init__(
        self,
        window_size: int = 10,
        stride: int = 1,
        pad_short: bool = True,
    ):
        self.window_size = window_size
        self.stride = stride
        self.pad_short = pad_short

    def make_windows(
        self,
        snapshots: List[Data],
        target_t: Optional[int] = None,
    ) -> List[List[Data]]:
        """
        Split a list of temporal snapshots into sliding windows.

        Parameters
        ----------
        snapshots : list of PyG Data objects
        target_t  : if specified, predict this time step (not used here)

        Returns
        -------
        List of windows, each a list of `window_size` Data objects.
        """
        T = len(snapshots)
        windows = []
        t = 0
        while t + self.window_size <= T:
            windows.append(snapshots[t : t + self.window_size])
            t += self.stride

        if self.pad_short and T < self.window_size:
            # Pad with first snapshot
            pad_count = self.window_size - T
            windows.append([snapshots[0]] * pad_count + snapshots)

        return windows

    def sample_window(
        self,
        snapshots: List[Data],
        t: int,
    ) -> List[Data]:
        """Return a single window ending at time t."""
        start = max(0, t - self.window_size + 1)
        window = snapshots[start : t + 1]
        if len(window) < self.window_size and self.pad_short:
            pad = self.window_size - len(window)
            window = [window[0]] * pad + window
        return window


# ---------------------------------------------------------------------------
# Sliding window graph batch builder
# ---------------------------------------------------------------------------

class SlidingWindowGraphBatch:
    """
    Produces mini-batches of graph snapshots using a sliding window.

    Each batch contains `batch_size` consecutive windows of temporal graphs.
    Supports both fixed-graph (static topology, dynamic features) and
    dynamic-graph (topology changes per step) modes.
    """

    def __init__(
        self,
        window_size: int = 10,
        batch_size: int = 8,
        stride: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.window_size = window_size
        self.batch_size = batch_size
        self.stride = stride
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def build_dataset(
        self,
        snapshots: List[Data],
        labels: Optional[Tensor] = None,
    ) -> List[Tuple[List[Data], Optional[Tensor]]]:
        """
        Parameters
        ----------
        snapshots : list of PyG Data (one per time step)
        labels    : optional (T,) tensor of labels per step

        Returns
        -------
        List of (window, label) tuples
        """
        T = len(snapshots)
        dataset = []
        t = self.window_size
        while t <= T:
            window = snapshots[t - self.window_size : t]
            lbl = labels[t - 1] if labels is not None else None
            dataset.append((window, lbl))
            t += self.stride

        return dataset

    def get_loader(
        self,
        dataset: List[Tuple[List[Data], Optional[Tensor]]],
    ) -> Iterator[Tuple[List[Data], Optional[Tensor]]]:
        """Yield batches from the dataset."""
        indices = list(range(len(dataset)))
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_windows = [dataset[i][0] for i in batch_idx]
            batch_labels = (
                torch.stack([dataset[i][1] for i in batch_idx if dataset[i][1] is not None])
                if dataset[0][1] is not None
                else None
            )
            yield batch_windows, batch_labels


# ---------------------------------------------------------------------------
# Neighbourhood sampler with temporal constraints
# ---------------------------------------------------------------------------

class TemporalNeighbourSampler(BaseSampler):
    """
    Sample k-hop neighbourhoods with temporal constraints.

    For each target node, expand to k hops in the graph but only
    include neighbours that have been "active" within a time window.

    Activity is defined by: a node is active at time t if it has
    at least one edge in snapshots [t - activity_window, t].
    """

    def __init__(
        self,
        num_hops: int = 2,
        max_neighbours: int = 20,
        activity_window: int = 10,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.num_hops = num_hops
        self.max_neighbours = max_neighbours
        self.activity_window = activity_window

    def sample_neighbourhood(
        self,
        target_node: int,
        edge_index: Tensor,
        node_attr: Optional[Tensor],
        edge_attr: Optional[Tensor],
        num_nodes: int,
        active_nodes: Optional[Tensor] = None,
    ) -> SampledSubgraph:
        """
        Parameters
        ----------
        target_node  : root node ID
        edge_index   : (2, E) global edges
        active_nodes : set of node IDs considered active at current time
        """
        adj = self._build_adjacency(edge_index, num_nodes)
        active_set = set(active_nodes.tolist()) if active_nodes is not None else set(range(num_nodes))

        # BFS up to num_hops
        visited = {target_node}
        frontier = {target_node}

        for _ in range(self.num_hops):
            next_frontier = set()
            for node in frontier:
                nbrs = [n for n in adj.get(node, []) if n in active_set and n not in visited]
                # Sample up to max_neighbours per node
                if len(nbrs) > self.max_neighbours:
                    nbrs = list(self.rng.choice(nbrs, size=self.max_neighbours, replace=False))
                next_frontier.update(nbrs)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        sampled_ids = torch.tensor(sorted(visited), dtype=torch.long)
        local_ei, local_ea = self._local_edge_index(sampled_ids, edge_index, edge_attr)
        local_nf = node_attr[sampled_ids] if node_attr is not None else None

        return SampledSubgraph(
            node_ids=sampled_ids,
            edge_index=local_ei,
            edge_attr=local_ea,
            node_attr=local_nf,
            sampler_type="temporal_neighbourhood",
        )

    def _build_adjacency(
        self, edge_index: Tensor, num_nodes: int
    ) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = defaultdict(list)
        for k in range(edge_index.shape[1]):
            s, d = int(edge_index[0, k]), int(edge_index[1, k])
            adj[s].append(d)
            adj[d].append(s)
        return dict(adj)

    def compute_active_nodes(
        self,
        snapshot_sequence: List[Tensor],
        current_t: int,
    ) -> Tensor:
        """
        Compute active nodes from a sequence of edge_index tensors.

        A node is active if it appears in any edge in the last activity_window snapshots.
        """
        t_start = max(0, current_t - self.activity_window)
        active = set()
        for t in range(t_start, current_t + 1):
            if t < len(snapshot_sequence):
                ei = snapshot_sequence[t]
                active.update(ei[0].tolist())
                active.update(ei[1].tolist())
        return torch.tensor(sorted(active), dtype=torch.long)


# ---------------------------------------------------------------------------
# Mini-batch constructor for dynamic graphs
# ---------------------------------------------------------------------------

class DynamicGraphMiniBatchConstructor:
    """
    Constructs PyTorch mini-batches for training on dynamic financial graphs.

    Supports two modes:
      1. Node classification: each node at each time step is a sample
      2. Link prediction: each potential edge at each time step is a sample
      3. Graph classification: each graph snapshot is a sample

    Also handles negative sampling for link prediction.
    """

    def __init__(
        self,
        task: str = "node_classification",
        batch_size: int = 64,
        neg_ratio: float = 1.0,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        assert task in ("node_classification", "link_prediction", "graph_classification")
        self.task = task
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def build_node_batches(
        self,
        snapshots: List[Data],
        labels: Tensor,  # (T, N) — label per (time, node)
    ) -> Iterator[Tuple[List[Data], Tensor, Tensor]]:
        """
        Yield (snapshots_window, target_node_ids, target_labels).
        """
        T = snapshots[0].num_nodes if snapshots else 0
        all_samples = [
            (t, n) for t in range(len(snapshots)) for n in range(T)
        ]
        if self.shuffle:
            self.rng.shuffle(all_samples)

        for start in range(0, len(all_samples), self.batch_size):
            batch = all_samples[start : start + self.batch_size]
            times = [s[0] for s in batch]
            nodes = [s[1] for s in batch]
            batch_labels = labels[times, nodes]
            yield ([snapshots[t] for t in times], torch.tensor(nodes), batch_labels)

    def build_link_batches(
        self,
        snapshots: List[Data],
        positive_edges: Optional[List[Tensor]] = None,
    ) -> Iterator[Tuple[Data, Tensor, Tensor, Tensor]]:
        """
        Yield (snapshot, pos_src, pos_dst, labels) with negatives.
        """
        for t, snap in enumerate(snapshots):
            ei = snap.edge_index
            n_pos = ei.shape[1] // 2  # undirected → half

            pos_src = ei[0, :n_pos]
            pos_dst = ei[1, :n_pos]

            # Negative sampling
            n_neg = int(n_pos * self.neg_ratio)
            neg_src = torch.randint(0, snap.num_nodes, (n_neg,))
            neg_dst = torch.randint(0, snap.num_nodes, (n_neg,))

            src = torch.cat([pos_src, neg_src])
            dst = torch.cat([pos_dst, neg_dst])
            labels = torch.cat([
                torch.ones(n_pos, dtype=torch.float32),
                torch.zeros(n_neg, dtype=torch.float32),
            ])

            # Shuffle
            perm = torch.randperm(len(labels))

            for start in range(0, len(perm), self.batch_size):
                idx = perm[start : start + self.batch_size]
                yield snap, src[idx], dst[idx], labels[idx]

    def build_graph_batches(
        self,
        snapshots: List[Data],
        graph_labels: Tensor,
    ) -> Iterator[Tuple[List[Data], Tensor]]:
        """
        Yield batches of full graph snapshots with graph-level labels.
        """
        N = len(snapshots)
        indices = list(range(N))
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, N, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_snaps = [snapshots[i] for i in batch_idx]
            batch_labels = graph_labels[batch_idx]
            yield batch_snaps, batch_labels


# ---------------------------------------------------------------------------
# Temporal graph batching utilities
# ---------------------------------------------------------------------------

def collate_temporal_snapshots(
    snapshots: List[Data],
    stack_node_features: bool = True,
) -> Dict[str, Tensor]:
    """
    Collate a list of temporal graph snapshots into a single tensor dict.

    Assumes all snapshots share the same topology (static graph, dynamic features).

    Returns
    -------
    dict with keys:
      - 'node_features' : (T, N, F) if stack_node_features
      - 'edge_index'    : (2, E) from last snapshot
      - 'edge_attr'     : (T, E, FE) or (E, FE) from last
    """
    T = len(snapshots)
    result = {}

    if stack_node_features and all(s.x is not None for s in snapshots):
        result["node_features"] = torch.stack([s.x for s in snapshots], dim=0)

    if snapshots:
        result["edge_index"] = snapshots[-1].edge_index
        if snapshots[-1].edge_attr is not None:
            result["edge_attr"] = torch.stack(
                [s.edge_attr for s in snapshots if s.edge_attr is not None], dim=0
            )

    result["T"] = T
    return result


def temporal_train_test_split(
    snapshots: List[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Split temporal snapshots into train/val/test sets (chronological split).
    """
    T = len(snapshots)
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    n_test = T - n_train - n_val

    train = snapshots[:n_train]
    val = snapshots[n_train : n_train + n_val]
    test = snapshots[n_train + n_val :]
    return train, val, test


# ---------------------------------------------------------------------------
# Feature standardiser for temporal graphs
# ---------------------------------------------------------------------------

class TemporalFeatureStandardiser(nn.Module):
    """
    Standardise node features across a temporal sequence.

    Computes running mean and variance per feature dimension,
    suitable for online / streaming graphs.
    """

    def __init__(self, n_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.n_features = n_features
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(n_features))
        self.register_buffer("running_var", torch.ones(n_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (N, F) or (T, N, F)
        """
        if self.training:
            if x.dim() == 3:
                flat = x.reshape(-1, self.n_features)
            else:
                flat = x

            batch_mean = flat.mean(dim=0)
            batch_var = flat.var(dim=0)

            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
            )
            self.num_batches_tracked += 1

            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        return (x - mean) / (var.sqrt() + self.eps)


# ---------------------------------------------------------------------------
# Adaptive sampling rate controller
# ---------------------------------------------------------------------------

class AdaptiveSamplingController:
    """
    Dynamically adjust sampling parameters based on graph statistics.

    Monitors graph density, volatility regime, and model loss to
    adapt sample budget and walk length over training.
    """

    def __init__(
        self,
        base_budget: int = 100,
        min_budget: int = 30,
        max_budget: int = 500,
        base_walk_length: int = 10,
    ):
        self.base_budget = base_budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.base_walk_length = base_walk_length

        self._loss_history: List[float] = []
        self._density_history: List[float] = []

    def update(self, loss: float, graph_density: float) -> None:
        self._loss_history.append(loss)
        self._density_history.append(graph_density)

    def get_budget(self) -> int:
        if len(self._loss_history) < 5:
            return self.base_budget

        recent_loss = np.mean(self._loss_history[-5:])
        prev_loss = np.mean(self._loss_history[-10:-5]) if len(self._loss_history) >= 10 else recent_loss

        # If loss is decreasing, maintain budget; if stagnant, increase
        if recent_loss >= prev_loss * 0.99:
            new_budget = min(int(self.base_budget * 1.2), self.max_budget)
        else:
            new_budget = max(int(self.base_budget * 0.9), self.min_budget)

        return new_budget

    def get_walk_length(self) -> int:
        if not self._density_history:
            return self.base_walk_length
        density = self._density_history[-1]
        # Sparse graphs need longer walks
        if density < 0.05:
            return int(self.base_walk_length * 2)
        elif density > 0.3:
            return max(self.base_walk_length // 2, 3)
        return self.base_walk_length


# ---------------------------------------------------------------------------
# Spatio-temporal graph dataset wrapper
# ---------------------------------------------------------------------------

class SpatioTemporalGraphDataset:
    """
    Dataset class for spatio-temporal financial graphs.

    Stores a list of temporal graph sequences with labels,
    supports batching with various samplers.
    """

    def __init__(
        self,
        sequences: List[List[Data]],
        labels: Optional[Tensor] = None,
        window_size: int = 10,
        sampler: Optional[BaseSampler] = None,
        transform=None,
    ):
        self.sequences = sequences
        self.labels = labels
        self.window_size = window_size
        self.sampler = sampler
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[Data], Optional[Tensor]]:
        seq = self.sequences[idx]
        label = self.labels[idx] if self.labels is not None else None

        if self.transform is not None:
            seq = [self.transform(s) for s in seq]

        return seq, label

    def get_dataloader(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[List[List[Data]], Optional[Tensor]]]:
        """Yield batches of (sequences, labels)."""
        rng = np.random.default_rng(seed)
        indices = list(range(len(self)))
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_seqs = []
            batch_labels = []
            for i in batch_idx:
                seq, lbl = self[i]
                batch_seqs.append(seq)
                if lbl is not None:
                    batch_labels.append(lbl)

            labels_out = torch.stack(batch_labels) if batch_labels else None
            yield batch_seqs, labels_out


# ---------------------------------------------------------------------------
# Heterogeneous temporal graph sampler
# ---------------------------------------------------------------------------

class HeterogeneousTemporalSampler(BaseSampler):
    """
    Sample subgraphs from heterogeneous temporal financial graphs.

    Handles multiple node/edge types and temporal constraints.
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        budget_per_type: Dict[str, int],
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.node_types = node_types
        self.edge_types = edge_types
        self.budget_per_type = budget_per_type

    def sample(
        self,
        hetero_data,  # HeteroData
        t: int = 0,
    ) -> Dict[str, SampledSubgraph]:
        """
        Parameters
        ----------
        hetero_data : PyG HeteroData
        t           : current time step

        Returns
        -------
        Dict mapping node_type → SampledSubgraph
        """
        results = {}
        for ntype in self.node_types:
            budget = self.budget_per_type.get(ntype, 50)
            if not hasattr(hetero_data, ntype):
                continue

            n = hetero_data[ntype].num_nodes
            budget = min(budget, n)
            chosen = self.rng.choice(n, size=budget, replace=False)
            sampled_ids = torch.tensor(sorted(chosen.tolist()), dtype=torch.long)

            x = hetero_data[ntype].x
            local_nf = x[sampled_ids] if x is not None else None

            results[ntype] = SampledSubgraph(
                node_ids=sampled_ids,
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=None,
                node_attr=local_nf,
                t_start=t,
                t_end=t,
                sampler_type="hetero_temporal",
            )

        return results


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "SampledSubgraph",
    "TemporalBatch",
    "BaseSampler",
    "GraphSAINTRandomWalkSampler",
    "GraphSAINTNodeSampler",
    "GraphSAINTEdgeSampler",
    "ClusterGCNSampler",
    "TemporalSnapshotSampler",
    "SlidingWindowGraphBatch",
    "TemporalNeighbourSampler",
    "DynamicGraphMiniBatchConstructor",
    "TemporalFeatureStandardiser",
    "AdaptiveSamplingController",
    "SpatioTemporalGraphDataset",
    "HeterogeneousTemporalSampler",
    "collate_temporal_snapshots",
    "temporal_train_test_split",
]
