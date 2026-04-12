# AETERNUS RTEL Python package
from .shm_reader import ShmReader, ShmChannel, ChannelCursor
from .shm_writer import ShmWriter
from .pipeline_client import PipelineClient
from .feature_store import FeatureStore, FeatureSchema, FeatureSnapshot
from .experiment_orchestrator import ExperimentOrchestrator, ExperimentConfig
from .signal_engine import SignalEngine, MomentumSignal, MeanReversionSignal
from .portfolio_optimizer import PortfolioOptimizationEngine, ERCOptimizer, MVOptimizer
from .data_pipeline import DataPipeline, SyntheticDataSource, RawTick
from .backtest_runner import BacktestRunner, BacktestConfig, BacktestStats
from .neuro_interface import (RTELModuleHub, NeuroSDEInterface, LuminaInterface,
                               HyperAgentInterface, OmniGraphInterface, TensorNetInterface)
from .config import RTELConfig
from .monitoring import MonitoringSystem, HealthRegistry, MetricsAggregator

__version__ = "0.1.0"
__all__ = [
    # Core SHM layer
    "ShmReader", "ShmChannel", "ChannelCursor",
    "ShmWriter",
    "PipelineClient",
    "FeatureStore", "FeatureSchema", "FeatureSnapshot",
    "ExperimentOrchestrator", "ExperimentConfig",
    # Signal and portfolio
    "SignalEngine", "MomentumSignal", "MeanReversionSignal",
    "PortfolioOptimizationEngine", "ERCOptimizer", "MVOptimizer",
    # Data pipeline
    "DataPipeline", "SyntheticDataSource", "RawTick",
    # Backtesting
    "BacktestRunner", "BacktestConfig", "BacktestStats",
    # Module interfaces
    "RTELModuleHub", "NeuroSDEInterface", "LuminaInterface",
    "HyperAgentInterface", "OmniGraphInterface", "TensorNetInterface",
    # Config and monitoring
    "RTELConfig", "MonitoringSystem", "HealthRegistry", "MetricsAggregator",
]

# Standard channel names
LOB_SNAPSHOT    = "aeternus.chronos.lob"
VOL_SURFACE     = "aeternus.neuro_sde.vol"
TENSOR_COMP     = "aeternus.tensornet.compressed"
GRAPH_ADJ       = "aeternus.omni_graph.adj"
LUMINA_PRED     = "aeternus.lumina.predictions"
AGENT_ACTIONS   = "aeternus.hyper_agent.actions"
AGENT_WEIGHTS   = "aeternus.hyper_agent.weights"
PIPELINE_EVENTS = "aeternus.rtel.pipeline_events"
HEARTBEAT       = "aeternus.rtel.heartbeat"
