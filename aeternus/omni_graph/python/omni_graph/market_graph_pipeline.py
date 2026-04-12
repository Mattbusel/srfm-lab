"""
market_graph_pipeline.py
========================
End-to-end pipeline: raw returns CSV → graph construction → GNN inference → regime output.

Supports:
  - CSV ingestion with return computation
  - Integration with Chronos LOB data
  - Streaming graph updates
  - Regime classification output
  - Full pipeline with logging and checkpointing
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from .graph_topology import (
    CorrelationGraphBuilder,
    GraphBuildConfig,
    FinancialGraphData,
    compute_node_features,
    pearson_correlation_matrix,
    MSTGraphBuilder,
    AdaptiveGraphBuilder,
)
from .dynamic_edges import DynamicGraphStateManager, EMAEdgeWeightManager
from .graph_integrity import GraphHealthMonitor
from .temporal_gnn import FinancialTGNPipeline, HarmonicTimeEncoder
from .heterogeneous_graph import HGTModel, TypeSpecificFeatureTransform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the market graph pipeline."""
    # Data
    return_window: int = 1           # return computation: log(P_t / P_{t-window})
    lookback_window: int = 60        # window for graph construction
    stride: int = 5                  # steps between graph updates
    min_observations: int = 30       # minimum obs for valid graph

    # Graph construction
    graph_method: str = "adaptive"   # pearson | mst | adaptive | knn
    corr_threshold: float = 0.3
    k_neighbours: int = 5
    max_lag: int = 5

    # GNN model
    model_type: str = "tgn"          # tgn | hgt | transformer
    node_feat_dim: int = 14
    edge_feat_dim: int = 3
    hidden_dim: int = 128
    out_dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    n_regimes: int = 4

    # Dynamic edges
    ema_alpha: float = 0.1
    birth_threshold: float = 0.3
    death_threshold: float = 0.05

    # Health monitoring
    enable_health_check: bool = True
    health_check_interval: int = 10

    # Streaming
    streaming_mode: bool = False
    buffer_size: int = 1000

    # Output
    output_format: str = "json"      # json | pandas | tensor
    checkpoint_dir: Optional[str] = None
    device: str = "cpu"


@dataclass
class PipelineOutput:
    """Output from a single pipeline step."""
    timestamp: int
    asset_names: List[str]
    regime: int
    regime_probs: np.ndarray
    node_embeddings: np.ndarray
    graph_stats: Dict[str, Any]
    health_report: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "asset_names": self.asset_names,
            "regime": self.regime,
            "regime_probs": self.regime_probs.tolist(),
            "graph_stats": self.graph_stats,
            "health_report": self.health_report,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

class MarketDataIngester:
    """
    Ingest market data from various sources.

    Supports:
      - CSV files (prices or returns)
      - Pandas DataFrames
      - Chronos LOB snapshots (streaming)
    """

    def __init__(
        self,
        return_window: int = 1,
        fillna_method: str = "ffill",
        min_price: float = 1e-6,
    ):
        self.return_window = return_window
        self.fillna_method = fillna_method
        self.min_price = min_price

    def from_csv(
        self,
        path: Union[str, Path],
        price_col_prefix: str = "",
        date_col: Optional[str] = "date",
        is_returns: bool = False,
    ) -> pd.DataFrame:
        """
        Load market data from CSV.

        Parameters
        ----------
        path          : path to CSV file
        price_col_prefix : prefix for price columns (e.g. "close_")
        date_col      : name of date/timestamp column
        is_returns    : whether data is already returns (vs prices)

        Returns
        -------
        DataFrame of log-returns, shape (T, N)
        """
        df = pd.read_csv(path)

        if date_col and date_col in df.columns:
            df = df.set_index(date_col)

        # Select price columns
        if price_col_prefix:
            cols = [c for c in df.columns if c.startswith(price_col_prefix)]
            df = df[cols]
            df.columns = [c[len(price_col_prefix):] for c in df.columns]

        # Handle missing data
        if self.fillna_method == "ffill":
            df = df.ffill().bfill()
        elif self.fillna_method == "zero":
            df = df.fillna(0)
        elif self.fillna_method == "mean":
            df = df.fillna(df.mean())

        if not is_returns:
            df = df.clip(lower=self.min_price)
            returns = np.log(df.shift(-self.return_window) / df).dropna()
            return returns
        else:
            return df.fillna(0)

    def from_dataframe(
        self,
        prices: pd.DataFrame,
        is_returns: bool = False,
    ) -> pd.DataFrame:
        """Convert price DataFrame to log-returns."""
        if is_returns:
            return prices.fillna(0)
        prices = prices.clip(lower=self.min_price)
        returns = np.log(prices.shift(-self.return_window) / prices).dropna()
        return returns

    def from_numpy(
        self,
        data: np.ndarray,
        is_returns: bool = False,
        asset_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert numpy array to returns.

        Returns (returns_array, asset_names).
        """
        if not is_returns:
            data = np.log(
                np.roll(data, -self.return_window, axis=0) / (data + 1e-10)
            )[:-self.return_window]

        n = data.shape[1]
        names = asset_names or [f"asset_{i}" for i in range(n)]
        return data.astype(np.float32), names


# ---------------------------------------------------------------------------
# Chronos LOB data adapter
# ---------------------------------------------------------------------------

class ChronosLOBAdapter:
    """
    Adapter for Chronos LOB streaming data format.

    Converts Chronos tick data (bid/ask price-level arrays) into
    graph-ready node features and edge indices.
    """

    def __init__(
        self,
        depth: int = 10,
        feature_config: Optional[Dict] = None,
    ):
        self.depth = depth
        self.feature_config = feature_config or {
            "use_mid_price": True,
            "use_spread": True,
            "use_imbalance": True,
            "use_vwap": True,
        }

    def parse_tick(
        self,
        tick: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Parse a Chronos tick dict into arrays.

        Expected keys: bid_prices, bid_sizes, ask_prices, ask_sizes, timestamp, symbol
        """
        bid_p = np.array(tick.get("bid_prices", []), dtype=np.float32)[: self.depth]
        bid_s = np.array(tick.get("bid_sizes", []), dtype=np.float32)[: self.depth]
        ask_p = np.array(tick.get("ask_prices", []), dtype=np.float32)[: self.depth]
        ask_s = np.array(tick.get("ask_sizes", []), dtype=np.float32)[: self.depth]

        # Pad if shorter than depth
        def pad(arr: np.ndarray, length: int, fill: float = 0.0) -> np.ndarray:
            if len(arr) < length:
                return np.concatenate([arr, np.full(length - len(arr), fill)])
            return arr[:length]

        bid_p = pad(bid_p, self.depth)
        bid_s = pad(bid_s, self.depth)
        ask_p = pad(ask_p, self.depth, fill=bid_p[-1] if len(bid_p) > 0 else 1.0)
        ask_s = pad(ask_s, self.depth)

        features = self._compute_lob_features(bid_p, bid_s, ask_p, ask_s)
        return {
            "bid_prices": bid_p,
            "bid_sizes": bid_s,
            "ask_prices": ask_p,
            "ask_sizes": ask_s,
            "features": features,
            "timestamp": float(tick.get("timestamp", 0)),
            "symbol": tick.get("symbol", "unknown"),
        }

    def _compute_lob_features(
        self,
        bid_p: np.ndarray,
        bid_s: np.ndarray,
        ask_p: np.ndarray,
        ask_s: np.ndarray,
    ) -> np.ndarray:
        """Compute scalar LOB features for node feature vector."""
        mid = (bid_p[0] + ask_p[0]) / 2.0 if len(bid_p) > 0 and len(ask_p) > 0 else 0.0
        spread = float(ask_p[0] - bid_p[0]) if len(bid_p) > 0 and len(ask_p) > 0 else 0.0

        total_bid = float(bid_s.sum())
        total_ask = float(ask_s.sum())
        imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)

        # VWAP for bid/ask
        bid_vwap = float(
            np.dot(bid_p, bid_s) / (total_bid + 1e-8)
        ) if total_bid > 0 else 0.0
        ask_vwap = float(
            np.dot(ask_p, ask_s) / (total_ask + 1e-8)
        ) if total_ask > 0 else 0.0

        return np.array([mid, spread, imbalance, bid_vwap, ask_vwap, total_bid, total_ask, math.log(mid + 1e-8)], dtype=np.float32)

    def ticks_to_returns(
        self,
        tick_history: List[Dict],
    ) -> np.ndarray:
        """
        Convert a history of LOB ticks to per-step mid-price returns.

        Returns shape (T-1, 1) — returns from sequential mid-prices.
        """
        mids = []
        for tick in tick_history:
            parsed = self.parse_tick(tick)
            bid_p = parsed["bid_prices"]
            ask_p = parsed["ask_prices"]
            if len(bid_p) > 0 and len(ask_p) > 0:
                mids.append((bid_p[0] + ask_p[0]) / 2.0)
            else:
                mids.append(mids[-1] if mids else 1.0)

        mids = np.array(mids, dtype=np.float32)
        returns = np.log(mids[1:] / (mids[:-1] + 1e-10))
        return returns.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Streaming graph update manager
# ---------------------------------------------------------------------------

class StreamingGraphUpdateManager:
    """
    Manage real-time graph updates from streaming data.

    Maintains a rolling buffer of returns and updates the graph
    whenever new data arrives.
    """

    def __init__(
        self,
        num_assets: int,
        config: Optional[PipelineConfig] = None,
    ):
        self.num_assets = num_assets
        self.config = config or PipelineConfig()
        self.buffer_size = config.buffer_size if config else 1000

        self._return_buffer = np.zeros((self.buffer_size, num_assets), dtype=np.float32)
        self._head = 0
        self._n_filled = 0

        cfg = GraphBuildConfig(
            corr_threshold=self.config.corr_threshold,
            k_neighbours=self.config.k_neighbours,
        )
        self._graph_builder = CorrelationGraphBuilder(cfg)
        self._ema_manager = EMAEdgeWeightManager(
            alpha=self.config.ema_alpha,
            birth_threshold=self.config.birth_threshold,
            death_threshold=self.config.death_threshold,
        )

    def push(self, returns: np.ndarray) -> Optional[FinancialGraphData]:
        """
        Push new returns observation and optionally rebuild graph.

        Parameters
        ----------
        returns : (N,) returns at current time step

        Returns
        -------
        FinancialGraphData if graph was updated, else None
        """
        self._return_buffer[self._head] = returns
        self._head = (self._head + 1) % self.buffer_size
        self._n_filled = min(self._n_filled + 1, self.buffer_size)

        if self._n_filled >= self.config.lookback_window:
            if self._n_filled % self.config.stride == 0:
                return self._rebuild_graph()

        return None

    def _rebuild_graph(self) -> FinancialGraphData:
        """Rebuild graph from current buffer."""
        if self._n_filled < self.config.lookback_window:
            start = 0
        else:
            # Get last `lookback_window` observations
            start = (self._head - self.config.lookback_window) % self.buffer_size

        if start < self._head:
            window = self._return_buffer[start : self._head]
        else:
            window = np.vstack([
                self._return_buffer[start :],
                self._return_buffer[: self._head],
            ])

        window = window[-self.config.lookback_window :]
        return self._graph_builder.build(window)


# ---------------------------------------------------------------------------
# GNN model factory
# ---------------------------------------------------------------------------

def build_gnn_model(config: PipelineConfig, num_nodes: int) -> nn.Module:
    """
    Build GNN model based on config.

    Parameters
    ----------
    config    : PipelineConfig
    num_nodes : number of asset nodes

    Returns
    -------
    nn.Module
    """
    if config.model_type == "tgn":
        return FinancialTGNPipeline(
            num_nodes=num_nodes,
            node_feat_dim=config.node_feat_dim,
            edge_feat_dim=config.edge_feat_dim,
            n_node_classes=config.n_regimes,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
    elif config.model_type == "transformer":
        from .graph_transformer import FinancialGraphTransformer
        return FinancialGraphTransformer(
            node_feat_dim=config.node_feat_dim,
            edge_feat_dim=config.edge_feat_dim,
            d_model=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_classes=config.n_regimes,
            dropout=config.dropout,
        )
    elif config.model_type == "causal_gnn":
        # Simple MLP baseline for causal graph output
        return nn.Sequential(
            nn.Linear(config.node_feat_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.out_dim),
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


# ---------------------------------------------------------------------------
# Regime classifier head
# ---------------------------------------------------------------------------

class RegimeClassificationHead(nn.Module):
    """
    Graph-level regime classification from node embeddings.

    Aggregates node embeddings and produces softmax regime probabilities.
    """

    def __init__(
        self,
        in_dim: int,
        n_regimes: int = 4,
        hidden_dim: int = 64,
        pool: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = pool

        if pool == "attention":
            self.attn_key = nn.Linear(in_dim, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_regimes),
        )

    def forward(self, node_embeds: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        node_embeds : (N, in_dim)

        Returns
        -------
        logits : (n_regimes,)
        probs  : (n_regimes,)
        """
        if self.pool == "attention":
            scores = self.attn_key(node_embeds)  # (N, 1)
            weights = torch.softmax(scores, dim=0)
            graph_embed = (node_embeds * weights).sum(dim=0)
        elif self.pool == "mean":
            graph_embed = node_embeds.mean(dim=0)
        elif self.pool == "max":
            graph_embed = node_embeds.max(dim=0).values
        else:
            graph_embed = node_embeds.mean(dim=0)

        logits = self.mlp(graph_embed)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class MarketGraphPipeline:
    """
    End-to-end market graph pipeline.

    Flow:
      1. Ingest raw returns (CSV / DataFrame / streaming)
      2. Build dynamic graph (correlation, MST, adaptive)
      3. Run GNN inference
      4. Classify market regime
      5. Output results (JSON, DataFrame, tensors)

    Supports both batch (historical) and streaming (real-time) modes.
    """

    def __init__(
        self,
        num_assets: int,
        asset_names: Optional[List[str]] = None,
        config: Optional[PipelineConfig] = None,
        pretrained_model: Optional[nn.Module] = None,
    ):
        self.num_assets = num_assets
        self.asset_names = asset_names or [f"asset_{i}" for i in range(num_assets)]
        self.config = config or PipelineConfig()

        # Data ingestion
        self.ingester = MarketDataIngester(return_window=self.config.return_window)
        self.lob_adapter = ChronosLOBAdapter()

        # Graph construction
        graph_cfg = GraphBuildConfig(
            corr_method="pearson",
            corr_threshold=self.config.corr_threshold,
            k_neighbours=self.config.k_neighbours,
            max_lag=self.config.max_lag,
        )
        if self.config.graph_method == "adaptive":
            self.graph_builder = AdaptiveGraphBuilder(graph_cfg)
        elif self.config.graph_method == "mst":
            self.graph_builder = MSTGraphBuilder(graph_cfg)
        else:
            self.graph_builder = CorrelationGraphBuilder(graph_cfg)

        # Dynamic state management
        self.dynamic_state = DynamicGraphStateManager(
            num_nodes=num_assets,
            ema_alpha=self.config.ema_alpha,
            birth_threshold=self.config.birth_threshold,
            death_threshold=self.config.death_threshold,
        )

        # Health monitoring
        if self.config.enable_health_check:
            self.health_monitor = GraphHealthMonitor(
                num_nodes=num_assets,
                window=20,
            )
        else:
            self.health_monitor = None

        # GNN model
        if pretrained_model is not None:
            self.model = pretrained_model
        else:
            self.model = build_gnn_model(self.config, num_assets)

        # Regime classification head
        self.regime_head = RegimeClassificationHead(
            in_dim=self.config.out_dim,
            n_regimes=self.config.n_regimes,
            hidden_dim=self.config.hidden_dim // 2,
        )

        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.regime_head.to(self.device)

        self._t = 0
        self._output_history: List[PipelineOutput] = []
        self._streaming_buffer = StreamingGraphUpdateManager(num_assets, self.config)

        logger.info(
            f"MarketGraphPipeline initialised: {num_assets} assets, "
            f"model={self.config.model_type}, device={self.config.device}"
        )

    def run_batch(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        timestamps: Optional[List] = None,
    ) -> List[PipelineOutput]:
        """
        Run pipeline on historical batch of returns.

        Parameters
        ----------
        returns    : (T, N) returns array or DataFrame
        timestamps : optional list of T timestamps

        Returns
        -------
        list of PipelineOutput, one per time window
        """
        if isinstance(returns, pd.DataFrame):
            asset_names = list(returns.columns)
            returns_np = returns.values.astype(np.float32)
        else:
            returns_np = returns.astype(np.float32)
            asset_names = self.asset_names

        T = returns_np.shape[0]
        outputs = []
        timestamps = timestamps or list(range(T))

        t = self.config.lookback_window
        while t <= T:
            window = returns_np[t - self.config.lookback_window : t]
            ts = timestamps[t - 1]

            output = self._process_window(window, asset_names, ts)
            outputs.append(output)
            self._output_history.append(output)

            t += self.config.stride

        logger.info(f"Batch run complete: {len(outputs)} graph snapshots processed.")
        return outputs

    def push_tick(
        self,
        returns: np.ndarray,
        timestamp: Optional[int] = None,
    ) -> Optional[PipelineOutput]:
        """
        Push single time step of returns (streaming mode).

        Returns PipelineOutput if graph was updated, else None.
        """
        self._t += 1
        ts = timestamp or self._t

        graph = self._streaming_buffer.push(returns)
        if graph is None:
            return None

        window_returns = self._streaming_buffer._return_buffer[
            max(0, self._streaming_buffer._head - self.config.lookback_window) : self._streaming_buffer._head
        ]
        if len(window_returns) < self.config.min_observations:
            return None

        output = self._process_window(window_returns, self.asset_names, ts)
        self._output_history.append(output)
        return output

    def _process_window(
        self,
        window_returns: np.ndarray,
        asset_names: List[str],
        ts: Any,
    ) -> PipelineOutput:
        """Process a single window of returns through the full pipeline."""
        # 1. Build graph
        try:
            if isinstance(self.graph_builder, AdaptiveGraphBuilder):
                graph_data = self.graph_builder.build(window_returns, asset_names)
            else:
                graph_data = self.graph_builder.build(window_returns, asset_names)
        except Exception as e:
            logger.warning(f"Graph construction failed at t={ts}: {e}")
            graph_data = self._empty_graph(len(asset_names))

        # 2. Dynamic state update
        corr = pearson_correlation_matrix(window_returns)
        if graph_data.edge_index.shape[1] > 0:
            ew = graph_data.edge_attr[:, 0] if graph_data.edge_attr is not None else torch.ones(graph_data.edge_index.shape[1])
            state_result = self.dynamic_state.step(
                graph_data.edge_index, ew, corr, regime=0
            )
            active_ei = state_result["edge_index"]
            active_ew = state_result["edge_weights"]
        else:
            active_ei = graph_data.edge_index
            active_ew = torch.zeros(0)

        # 3. Node features
        node_feat = compute_node_features(window_returns)

        # 4. GNN inference
        with torch.no_grad():
            node_feat_d = node_feat.to(self.device)
            edge_index_d = active_ei.to(self.device)
            edge_attr_d = active_ew.unsqueeze(-1).to(self.device) if active_ew.shape[0] > 0 else None

            # Pad edge_attr to match expected edge_feat_dim
            if edge_attr_d is not None and edge_attr_d.shape[-1] < self.config.edge_feat_dim:
                pad = torch.zeros(edge_attr_d.shape[0], self.config.edge_feat_dim - edge_attr_d.shape[-1], device=self.device)
                edge_attr_d = torch.cat([edge_attr_d, pad], dim=-1)

            try:
                model_out = self._run_model(node_feat_d, edge_index_d, edge_attr_d)
                node_embeds = model_out.get("node_embeds", node_feat_d)
            except Exception as e:
                logger.warning(f"GNN inference failed at t={ts}: {e}")
                node_embeds = node_feat_d

            # 5. Regime classification
            try:
                logits, probs = self.regime_head(node_embeds)
                regime = int(logits.argmax().item())
                regime_probs_np = probs.cpu().numpy()
            except Exception:
                regime = 0
                regime_probs_np = np.ones(self.config.n_regimes) / self.config.n_regimes

        # 6. Graph statistics
        graph_stats = self._compute_graph_stats(active_ei, self.num_assets)

        # 7. Health check
        health_info = None
        if self.health_monitor is not None and self._t % self.config.health_check_interval == 0:
            report = self.health_monitor.check(active_ei, active_ew.unsqueeze(-1) if active_ew.shape[0] > 0 else None, t=self._t)
            health_info = {"is_healthy": report.is_healthy, "alerts": report.alerts, "anomaly_score": report.anomaly_score}

        return PipelineOutput(
            timestamp=int(ts) if isinstance(ts, (int, np.integer)) else hash(str(ts)) % 2**31,
            asset_names=asset_names,
            regime=regime,
            regime_probs=regime_probs_np,
            node_embeddings=node_embeds.cpu().numpy(),
            graph_stats=graph_stats,
            health_report=health_info,
            metadata={"t": self._t, "n_active_edges": int(active_ei.shape[1])},
        )

    def _run_model(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """Run the GNN model and return output dict."""
        if isinstance(self.model, FinancialTGNPipeline):
            # TGN needs edge timestamps
            T_edge = edge_index.shape[1]
            edge_times = torch.zeros(T_edge, device=node_feat.device)
            edge_feat = edge_attr if edge_attr is not None else torch.ones(T_edge, self.config.edge_feat_dim, device=node_feat.device)
            return self.model(node_feat, edge_index, edge_feat, edge_times, query_time=float(self._t))
        else:
            # For transformer: forward returns dict
            result = self.model(
                node_feat, edge_index, edge_attr
            )
            if isinstance(result, dict):
                return result
            # If model returns tensor directly
            return {"node_embeds": result if isinstance(result, Tensor) else result[0]}

    def _compute_graph_stats(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Dict[str, Any]:
        E = int(edge_index.shape[1])
        density = 2 * E / max(num_nodes * (num_nodes - 1), 1)
        return {
            "n_nodes": num_nodes,
            "n_edges": E,
            "density": float(density),
        }

    def _empty_graph(self, n: int) -> FinancialGraphData:
        from .graph_topology import FinancialGraphData
        return FinancialGraphData(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            num_nodes=n,
            asset_names=self.asset_names,
        )

    def get_regime_series(self) -> pd.Series:
        """Return time series of regime labels."""
        return pd.Series(
            [o.regime for o in self._output_history],
            index=[o.timestamp for o in self._output_history],
            name="regime",
        )

    def get_graph_stats_df(self) -> pd.DataFrame:
        """Return DataFrame of graph statistics over time."""
        rows = []
        for o in self._output_history:
            row = {"timestamp": o.timestamp}
            row.update(o.graph_stats)
            rows.append(row)
        return pd.DataFrame(rows).set_index("timestamp")

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save model checkpoint and pipeline state."""
        path = path or self.config.checkpoint_dir or "."
        os.makedirs(path, exist_ok=True)
        ckpt_path = os.path.join(path, f"omni_graph_ckpt_{int(time.time())}.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "regime_head_state": self.regime_head.state_dict(),
            "config": self.config,
            "t": self._t,
        }, ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")
        return ckpt_path

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.regime_head.load_state_dict(ckpt["regime_head_state"])
        self._t = ckpt.get("t", 0)
        logger.info(f"Checkpoint loaded from {path}")

    def run_from_csv(
        self,
        csv_path: Union[str, Path],
        date_col: Optional[str] = "date",
        is_returns: bool = False,
    ) -> List[PipelineOutput]:
        """Convenience method: load CSV and run full pipeline."""
        returns_df = self.ingester.from_csv(csv_path, date_col=date_col, is_returns=is_returns)
        timestamps = list(returns_df.index)
        self.asset_names = list(returns_df.columns)
        return self.run_batch(returns_df.values, timestamps=timestamps)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "PipelineConfig",
    "PipelineOutput",
    "MarketDataIngester",
    "ChronosLOBAdapter",
    "StreamingGraphUpdateManager",
    "build_gnn_model",
    "RegimeClassificationHead",
    "MarketGraphPipeline",
]
