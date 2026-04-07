"""
research/validation -- model validation and hypothesis testing framework.

Provides statistical tests, model validation, hypothesis lifecycle management,
and bootstrap inference methods for quantitative trading research.
"""

from research.validation.statistical_tests import (
    NormalityTests, StationarityTests, SignalTests, ModelDiagnostics, TestResult
)
from research.validation.model_validator import (
    ModelSpec, ValidationReport, SignalModelValidator, RiskModelValidator, MLModelValidator
)
from research.validation.hypothesis_engine import (
    Hypothesis, HypothesisTest, HypothesisLibrary, HypothesisTestResult
)
from research.validation.bootstrap_analyzer import (
    StationaryBootstrap, CircularBlockBootstrap, BootstrapTests
)
from research.validation.causal_inference import (
    CausalAnalyzer, GrangerResult, PSMResult, DiffInDiffResult, IVResult
)
from research.validation.out_of_sample_validator import (
    OutOfSampleValidator, OOSResult, WalkForwardResult, CPCVResult
)
from research.validation.market_efficiency_tests import (
    MarketEfficiencyTests, VRTestResult, RunsTestResult, LjungBoxResult,
    LongMemoryResult, ThresholdCointResult
)
from research.validation.performance_persistence import (
    PerformancePersistenceAnalyzer, ContingencyResult, IRStabilityResult,
    RegimePersistenceResult
)

__all__ = [
    "NormalityTests", "StationarityTests", "SignalTests", "ModelDiagnostics", "TestResult",
    "ModelSpec", "ValidationReport", "SignalModelValidator", "RiskModelValidator", "MLModelValidator",
    "Hypothesis", "HypothesisTest", "HypothesisLibrary", "HypothesisTestResult",
    "StationaryBootstrap", "CircularBlockBootstrap", "BootstrapTests",
    # Causal inference
    "CausalAnalyzer", "GrangerResult", "PSMResult", "DiffInDiffResult", "IVResult",
    # OOS validation
    "OutOfSampleValidator", "OOSResult", "WalkForwardResult", "CPCVResult",
    # Market efficiency
    "MarketEfficiencyTests", "VRTestResult", "RunsTestResult", "LjungBoxResult",
    "LongMemoryResult", "ThresholdCointResult",
    # Performance persistence
    "PerformancePersistenceAnalyzer", "ContingencyResult", "IRStabilityResult",
    "RegimePersistenceResult",
]
