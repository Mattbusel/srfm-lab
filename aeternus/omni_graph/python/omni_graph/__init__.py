"""
omni_graph — Module 4 of Project AETERNUS
==========================================
Dynamic Temporal Graph Neural Network system for financial market analysis.

Combines:
- PyTorch Geometric GNNs with temporal awareness
- Ollivier-Ricci curvature (via Rust network-graph crate)
- Wormhole detection (sudden high-weight cross-community edges)
- Regime classification from graph topology
- Crisis early warning from curvature trajectories

Architecture:
    financial_graphs  ─ Build graphs from returns (correlation, Granger, TE, GLASSO)
    dynamic_gnn       ─ Temporal GCN, DynamicEdgeConv, GraphRNN, EvolutionaryGNN
    edge_prediction   ─ GraphDiffusion, LinkPredictor, WormholeDetector, RicciFlowGNN
    regime_gnn        ─ GraphRegimeDetector, WassersteinKernel, CrisisEarlyWarning
    experiments       ─ Benchmark experiments, saved results

Version history:
    0.1.0  2026-04-11  Initial Module 4 implementation
"""

__version__ = "0.1.0"
__author__ = "SRFM Lab — Project AETERNUS"

# Core graph construction
from omni_graph.financial_graphs import (
    build_correlation_graph,
    build_granger_graph,
    build_partial_correlation_graph,
    build_transfer_entropy_graph,
    GraphEvolution,
)

# Dynamic GNN models
from omni_graph.dynamic_gnn import (
    TemporalGraphConv,
    DynamicEdgeConv,
    TemporalAttention,
    GraphRNN,
    EvolutionaryGNN,
    train_evolutionary_gnn,
)

# Edge prediction and anomaly detection
from omni_graph.edge_prediction import (
    GraphDiffusion,
    LinkPredictor,
    WormholeDetector,
    RicciFlowGNN,
)

# Regime detection
from omni_graph.regime_gnn import (
    GraphRegimeDetector,
    WassersteinGraphKernel,
    RegimeTransitionPredictor,
    CrisisEarlyWarning,
)

__all__ = [
    "__version__",
    # Financial graph construction
    "build_correlation_graph",
    "build_granger_graph",
    "build_partial_correlation_graph",
    "build_transfer_entropy_graph",
    "GraphEvolution",
    # GNN models
    "TemporalGraphConv",
    "DynamicEdgeConv",
    "TemporalAttention",
    "GraphRNN",
    "EvolutionaryGNN",
    "train_evolutionary_gnn",
    # Edge prediction
    "GraphDiffusion",
    "LinkPredictor",
    "WormholeDetector",
    "RicciFlowGNN",
    # Regime detection
    "GraphRegimeDetector",
    "WassersteinGraphKernel",
    "RegimeTransitionPredictor",
    "CrisisEarlyWarning",
]
