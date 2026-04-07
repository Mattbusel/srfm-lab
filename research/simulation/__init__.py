"""
research/simulation -- synthetic market data generation and stress testing.

Modules:
  market_simulator         -- GBM, OU, regime-switching OHLCV generation
  bh_signal_injector       -- BH mass / quaternion-nav signal injection
  stress_scenarios         -- pre-built LARSA stress test scenarios
  parameter_sensitivity_sim -- Monte Carlo parameter sensitivity analysis
"""

from research.simulation.market_simulator import (
    MarketRegime,
    SimConfig,
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    RegimeSwitchingMarket,
    CorrelatedAssetSimulator,
)
from research.simulation.bh_signal_injector import (
    BHMassSimulator,
    QuatNavSignalInjector,
    SignalQualityInjector,
)
from research.simulation.stress_scenarios import (
    StressScenario,
    StressResult,
    StressTester,
    STRESS_SCENARIOS,
)
from research.simulation.parameter_sensitivity_sim import (
    ParameterSensitivitySimulator,
    SensitivityResult,
)

__all__ = [
    "MarketRegime", "SimConfig", "GeometricBrownianMotion", "OrnsteinUhlenbeck",
    "RegimeSwitchingMarket", "CorrelatedAssetSimulator",
    "BHMassSimulator", "QuatNavSignalInjector", "SignalQualityInjector",
    "StressScenario", "StressResult", "StressTester", "STRESS_SCENARIOS",
    "ParameterSensitivitySimulator", "SensitivityResult",
]
