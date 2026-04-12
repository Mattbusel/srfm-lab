"""
AETERNUS Real-Time Execution Layer (RTEL)
neuro_interface.py — Interface layer between RTEL and Neuro-SDE / TensorNet

Provides Python-side wrappers for:
- Reading Neuro-SDE vol surfaces from RTEL shared memory
- Feeding TensorNet compressed tensors
- Lumina prediction consumption
- HyperAgent action dispatching
- Thin adapter classes for each AETERNUS module
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Neuro-SDE vol surface interface
# ---------------------------------------------------------------------------

@dataclass
class VolSurfaceMessage:
    asset_id:    int
    timestamp:   float
    n_strikes:   int
    n_expiries:  int
    atm_vol:     float
    skew:        float
    term_slope:  float
    surface:     np.ndarray       # [n_strikes × n_expiries]
    model_params: Dict[str, float] = field(default_factory=dict)

    def iv(self, strike_idx: int, expiry_idx: int) -> float:
        if (0 <= strike_idx < self.n_strikes and
                0 <= expiry_idx < self.n_expiries):
            return float(self.surface[strike_idx, expiry_idx])
        return self.atm_vol

    def risk_reversal(self, delta_pct: float = 25.0) -> float:
        """Approximate delta-risk-reversal from surface."""
        k_lo = max(0, int((50 - delta_pct) / 100.0 * self.n_strikes))
        k_hi = min(self.n_strikes-1, int((50 + delta_pct) / 100.0 * self.n_strikes))
        if self.n_expiries < 1:
            return 0.0
        return float(self.surface[k_hi, 0] - self.surface[k_lo, 0])

    def butterfly(self, delta_pct: float = 25.0) -> float:
        k_lo = max(0, int((50 - delta_pct) / 100.0 * self.n_strikes))
        k_hi = min(self.n_strikes-1, int((50 + delta_pct) / 100.0 * self.n_strikes))
        k_atm= self.n_strikes // 2
        if self.n_expiries < 1:
            return 0.0
        return float(0.5*(self.surface[k_hi,0] + self.surface[k_lo,0]) -
                     self.surface[k_atm, 0])


class NeuroSDEInterface:
    """Adapter for consuming Neuro-SDE volatility surface outputs."""

    def __init__(self, n_assets: int, n_strikes: int = 21, n_expiries: int = 8):
        self.n_assets   = n_assets
        self.n_strikes  = n_strikes
        self.n_expiries = n_expiries
        self._surfaces: Dict[int, VolSurfaceMessage] = {}
        self._history:  Dict[int, deque] = {}
        self._callbacks: List[Callable[[VolSurfaceMessage], None]] = []

    def on_vol_surface(self, msg: VolSurfaceMessage) -> None:
        """Handle incoming vol surface from Neuro-SDE."""
        self._surfaces[msg.asset_id] = msg
        if msg.asset_id not in self._history:
            self._history[msg.asset_id] = deque(maxlen=100)
        self._history[msg.asset_id].append(msg)
        for cb in self._callbacks:
            try:
                cb(msg)
            except Exception as e:
                logger.warning("Vol surface callback error: %s", e)

    def add_callback(self, fn: Callable[[VolSurfaceMessage], None]) -> None:
        self._callbacks.append(fn)

    def latest_surface(self, asset_id: int) -> Optional[VolSurfaceMessage]:
        return self._surfaces.get(asset_id)

    def atm_vols(self) -> Dict[int, float]:
        return {aid: msg.atm_vol for aid, msg in self._surfaces.items()}

    def generate_synthetic(self, asset_id: int, base_vol: float = 0.25) -> VolSurfaceMessage:
        """Generate a synthetic Heston-like vol surface."""
        strikes  = np.linspace(0.7, 1.3, self.n_strikes)
        expiries = np.array([0.083, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])[:self.n_expiries]
        surface  = np.zeros((self.n_strikes, self.n_expiries), dtype=np.float32)

        for j, T in enumerate(expiries):
            for i, K in enumerate(strikes):
                log_K = math.log(K)
                # Heston-like smile: vol ≈ atm + skew*logK + curvature*logK^2
                skew      = -0.05 / math.sqrt(T + 0.1)
                curvature = 0.10 / (T + 0.1)
                surface[i, j] = max(0.05,
                    base_vol + skew * log_K + curvature * log_K**2 +
                    0.01 * math.sqrt(T))

        atm_idx = self.n_strikes // 2
        return VolSurfaceMessage(
            asset_id   = asset_id,
            timestamp  = time.time(),
            n_strikes  = self.n_strikes,
            n_expiries = self.n_expiries,
            atm_vol    = float(surface[atm_idx, 0]),
            skew       = float(surface[-1, 0] - surface[0, 0]),
            term_slope = float(surface[atm_idx, -1] - surface[atm_idx, 0]),
            surface    = surface,
            model_params = {"kappa": 2.0, "theta": base_vol**2,
                            "xi": 0.3, "rho": -0.7},
        )

    def seed_all(self, base_vols: Optional[Dict[int, float]] = None) -> None:
        """Seed all assets with synthetic surfaces."""
        for i in range(self.n_assets):
            bv = (base_vols or {}).get(i, 0.20 + 0.05 * (i % 3))
            msg = self.generate_synthetic(i, bv)
            self.on_vol_surface(msg)


# ---------------------------------------------------------------------------
# TensorNet interface
# ---------------------------------------------------------------------------

@dataclass
class CompressedTensor:
    asset_id:   int
    timestamp:  float
    shape:      Tuple[int, ...]
    cores:      List[np.ndarray]   # TT-cores
    dtype:      str = "float32"
    compression_ratio: float = 1.0
    recon_error: float = 0.0

    def decompress(self) -> np.ndarray:
        """Reconstruct tensor from TT-cores."""
        if not self.cores:
            return np.zeros(self.shape)
        result = self.cores[0][0]
        for k in range(1, len(self.cores)):
            r_prev, n_k, r_k = self.cores[k].shape
            shape_in = result.shape
            result   = result.reshape(-1, r_prev)
            mat      = self.cores[k].reshape(r_prev, n_k * r_k)
            result   = result @ mat
            result   = result.reshape(*shape_in[:-1], n_k, r_k) \
                       if len(shape_in) > 1 else result.reshape(n_k, r_k)
        return result[..., 0] if result.ndim > 1 else result


class TensorNetInterface:
    """Adapter for consuming TensorNet compressed tensor outputs."""

    def __init__(self, n_assets: int):
        self.n_assets    = n_assets
        self._tensors:   Dict[int, CompressedTensor] = {}
        self._decode_cache: Dict[int, np.ndarray] = {}
        self._callbacks: List[Callable[[CompressedTensor], None]] = []

    def on_tensor(self, tensor: CompressedTensor) -> None:
        self._tensors[tensor.asset_id] = tensor
        # Invalidate decode cache
        self._decode_cache.pop(tensor.asset_id, None)
        for cb in self._callbacks:
            try:
                cb(tensor)
            except Exception as e:
                logger.warning("Tensor callback error: %s", e)

    def add_callback(self, fn: Callable[[CompressedTensor], None]) -> None:
        self._callbacks.append(fn)

    def get_tensor(self, asset_id: int, decode: bool = False) -> Optional[np.ndarray]:
        t = self._tensors.get(asset_id)
        if t is None:
            return None
        if not decode:
            return None  # Return raw cores
        if asset_id not in self._decode_cache:
            self._decode_cache[asset_id] = t.decompress()
        return self._decode_cache[asset_id]

    def compression_stats(self) -> dict:
        if not self._tensors:
            return {}
        ratios = [t.compression_ratio for t in self._tensors.values()]
        errors = [t.recon_error for t in self._tensors.values()]
        return {
            "n_tensors":   len(self._tensors),
            "mean_ratio":  float(np.mean(ratios)),
            "mean_error":  float(np.mean(errors)),
            "min_ratio":   float(min(ratios)),
            "max_ratio":   float(max(ratios)),
        }


# ---------------------------------------------------------------------------
# Lumina prediction interface
# ---------------------------------------------------------------------------

@dataclass
class LuminaPrediction:
    asset_id:    int
    timestamp:   float
    direction:   float    # [-1, +1]
    confidence:  float    # [0, 1]
    horizon_s:   float    # prediction horizon in seconds
    features:    Optional[np.ndarray] = None
    model_version: str = "v1"


class LuminaInterface:
    """Adapter for consuming Lumina alpha prediction outputs."""

    def __init__(self, n_assets: int):
        self.n_assets    = n_assets
        self._predictions: Dict[int, LuminaPrediction] = {}
        self._history:   Dict[int, deque] = {}
        self._callbacks: List[Callable[[LuminaPrediction], None]] = []
        self._ic_tracker: Dict[int, deque] = {}  # rolling IC per asset

    def on_prediction(self, pred: LuminaPrediction) -> None:
        self._predictions[pred.asset_id] = pred
        if pred.asset_id not in self._history:
            self._history[pred.asset_id]    = deque(maxlen=500)
            self._ic_tracker[pred.asset_id] = deque(maxlen=100)
        self._history[pred.asset_id].append(pred)
        for cb in self._callbacks:
            try:
                cb(pred)
            except Exception as e:
                logger.warning("Lumina callback error: %s", e)

    def add_callback(self, fn: Callable[[LuminaPrediction], None]) -> None:
        self._callbacks.append(fn)

    def update_realized_return(self, asset_id: int, ret: float) -> None:
        """Update IC tracking with realized return."""
        hist = self._history.get(asset_id)
        if hist and len(hist) >= 2:
            prev_pred = hist[-2]
            ic = prev_pred.direction * np.sign(ret) if abs(ret) > _EPS else 0.0
            self._ic_tracker[asset_id].append(ic)

    def signal(self, asset_id: int) -> Optional[float]:
        pred = self._predictions.get(asset_id)
        if pred is None:
            return None
        return pred.direction * pred.confidence

    def all_signals(self) -> Dict[int, float]:
        return {
            aid: pred.direction * pred.confidence
            for aid, pred in self._predictions.items()
        }

    def rolling_ic(self, asset_id: int) -> float:
        tracker = self._ic_tracker.get(asset_id)
        if not tracker:
            return 0.0
        return float(np.mean(list(tracker)))

    def generate_synthetic(self, asset_id: int, price_change: float) -> LuminaPrediction:
        """Generate synthetic Lumina prediction based on price momentum."""
        direction  = math.tanh(price_change * 50.0)
        confidence = min(1.0, abs(price_change) * 100.0)
        return LuminaPrediction(
            asset_id    = asset_id,
            timestamp   = time.time(),
            direction   = direction,
            confidence  = confidence,
            horizon_s   = 60.0,
        )


# ---------------------------------------------------------------------------
# HyperAgent action interface
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    asset_id:    int
    timestamp:   float
    position_target: float  # target position fraction [-1, +1]
    urgency:     float      # [0, 1] — how urgently to execute
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None
    max_slippage_bps: float = 10.0
    agent_id:    int = 0
    episode:     int = 0


class HyperAgentInterface:
    """Adapter for dispatching HyperAgent actions to execution."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self._actions: Dict[int, AgentAction] = {}
        self._action_history: deque = deque(maxlen=10000)
        self._callbacks: List[Callable[[AgentAction], None]] = []
        self._n_actions = 0

    def on_action(self, action: AgentAction) -> None:
        self._actions[action.asset_id] = action
        self._action_history.append(action)
        self._n_actions += 1
        for cb in self._callbacks:
            try:
                cb(action)
            except Exception as e:
                logger.warning("Agent action callback error: %s", e)

    def add_callback(self, fn: Callable[[AgentAction], None]) -> None:
        self._callbacks.append(fn)

    def get_target_positions(self) -> Dict[int, float]:
        return {aid: a.position_target for aid, a in self._actions.items()}

    def get_urgency(self, asset_id: int) -> float:
        a = self._actions.get(asset_id)
        return a.urgency if a else 0.0

    def n_actions(self) -> int:
        return self._n_actions

    def generate_synthetic(self, asset_id: int, signal: float) -> AgentAction:
        """Generate synthetic agent action from signal."""
        pos_target = math.tanh(signal * 2.0)
        urgency    = min(1.0, abs(signal))
        return AgentAction(
            asset_id        = asset_id,
            timestamp       = time.time(),
            position_target = pos_target,
            urgency         = urgency,
            max_slippage_bps= 5.0 + urgency * 10.0,
        )


# ---------------------------------------------------------------------------
# OmniGraph adjacency interface
# ---------------------------------------------------------------------------

@dataclass
class GraphAdjacency:
    timestamp:  float
    n_nodes:    int
    adj_matrix: np.ndarray   # [n_nodes × n_nodes] weighted adjacency
    communities: Optional[List[int]] = None   # community assignment per node
    pagerank:   Optional[np.ndarray] = None

    def density(self) -> float:
        n = self.n_nodes
        max_edges = n * (n-1) / 2.0
        if max_edges < _EPS:
            return 0.0
        edges = float((self.adj_matrix > 0).sum() - self.n_nodes) / 2.0
        return edges / max_edges

    def mean_weight(self) -> float:
        n = self.n_nodes
        if n < 2:
            return 0.0
        upper = self.adj_matrix[np.triu_indices(n, k=1)]
        return float(upper.mean()) if len(upper) > 0 else 0.0


class OmniGraphInterface:
    """Adapter for consuming OmniGraph adjacency updates."""

    def __init__(self, n_assets: int):
        self.n_assets   = n_assets
        self._latest:   Optional[GraphAdjacency] = None
        self._history:  deque = deque(maxlen=100)
        self._callbacks: List[Callable[[GraphAdjacency], None]] = []

    def on_adjacency(self, adj: GraphAdjacency) -> None:
        self._latest = adj
        self._history.append(adj)
        for cb in self._callbacks:
            try:
                cb(adj)
            except Exception as e:
                logger.warning("Graph callback error: %s", e)

    def add_callback(self, fn: Callable[[GraphAdjacency], None]) -> None:
        self._callbacks.append(fn)

    def latest(self) -> Optional[GraphAdjacency]:
        return self._latest

    def correlation_to_asset(self, asset_id: int) -> Optional[np.ndarray]:
        if self._latest is None or asset_id >= self.n_assets:
            return None
        return self._latest.adj_matrix[asset_id].copy()

    def generate_synthetic(self) -> GraphAdjacency:
        """Generate synthetic correlation graph."""
        n   = self.n_assets
        rng = np.random.default_rng(int(time.time() * 1000) % (2**32))
        raw = rng.uniform(0.2, 0.8, (n, n))
        adj = (raw + raw.T) / 2.0
        np.fill_diagonal(adj, 1.0)
        # Simple community assignment
        communities = [i % max(1, n // 3) for i in range(n)]
        # PageRank (uniform as placeholder)
        pr = np.ones(n) / n
        return GraphAdjacency(
            timestamp   = time.time(),
            n_nodes     = n,
            adj_matrix  = adj.astype(np.float32),
            communities = communities,
            pagerank    = pr,
        )


# ---------------------------------------------------------------------------
# RTELModuleHub — integrates all module interfaces
# ---------------------------------------------------------------------------

class RTELModuleHub:
    """
    Central hub connecting all AETERNUS module interfaces.
    Coordinates data flow: Chronos LOB → signals → portfolio → execution.
    """

    def __init__(self, n_assets: int):
        self.n_assets      = n_assets
        self.neuro_sde     = NeuroSDEInterface(n_assets)
        self.tensornet     = TensorNetInterface(n_assets)
        self.lumina        = LuminaInterface(n_assets)
        self.hyper_agent   = HyperAgentInterface(n_assets)
        self.omni_graph    = OmniGraphInterface(n_assets)

        self._step         = 0
        self._last_prices: Dict[int, float] = {}

    def update_all(self, prices: Dict[int, float],
                   price_changes: Optional[Dict[int, float]] = None) -> None:
        """Simulate a full pipeline step."""
        changes = price_changes or {}

        # Update Lumina predictions
        for aid, price in prices.items():
            change = changes.get(aid, 0.0)
            pred   = self.lumina.generate_synthetic(aid, change)
            self.lumina.on_prediction(pred)

        # Update realized returns for IC
        if self._last_prices:
            for aid, price in prices.items():
                prev = self._last_prices.get(aid, price)
                if prev > _EPS:
                    ret = (price - prev) / prev
                    self.lumina.update_realized_return(aid, ret)

        # Update OmniGraph periodically
        if self._step % 10 == 0:
            self.omni_graph.on_adjacency(self.omni_graph.generate_synthetic())

        # Update Neuro-SDE vol surfaces periodically
        if self._step % 5 == 0:
            for aid in range(self.n_assets):
                msg = self.neuro_sde.generate_synthetic(aid)
                self.neuro_sde.on_vol_surface(msg)

        # Get combined signals and generate agent actions
        signals = self.lumina.all_signals()
        for aid, sig in signals.items():
            action = self.hyper_agent.generate_synthetic(aid, sig)
            self.hyper_agent.on_action(action)

        self._last_prices = dict(prices)
        self._step += 1

    def target_positions(self) -> Dict[int, float]:
        return self.hyper_agent.get_target_positions()

    def atm_vols(self) -> Dict[int, float]:
        return self.neuro_sde.atm_vols()

    def diagnostics(self) -> dict:
        return {
            "step":           self._step,
            "n_assets":       self.n_assets,
            "n_vol_surfaces": len(self.neuro_sde._surfaces),
            "n_predictions":  len(self.lumina._predictions),
            "n_actions":      self.hyper_agent.n_actions(),
            "graph_density":  self.omni_graph.latest().density()
                               if self.omni_graph.latest() else 0.0,
            "mean_ic":        float(np.mean([
                                   self.lumina.rolling_ic(i)
                                   for i in range(self.n_assets)])),
        }
