"""
microstructure/__init__.py

Market microstructure analysis for the IAE.

Detects when the market is healthy for the BH signal vs when microstructure
conditions would impair execution quality.

Models
------
- Amihud illiquidity ratio (price impact from volume)
- Kyle's lambda (order flow price impact)
- Roll spread estimator (effective spread from serial covariance)
- Hasbrouck information share (price discovery leadership)
- Adverse selection / PIN proxy (informed trading risk)
- Intraday hourly profiles (structural hour-of-day patterns)

Outputs
-------
- MicrostructureSignal: composite health score + recommended size multiplier
- IntradayMicrostructureProfile: 24-hour spread/volume/score profile
- CalibrationResult: IC between health score and next-bar returns

Quick start
-----------
from microstructure.live_monitor import LiveMicrostructureMonitor, MonitorConfig
from microstructure.hypothesis_generator import MicrostructureHypothesisGenerator

config = MonitorConfig(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
monitor = LiveMicrostructureMonitor(config, data_fn=my_ohlcv_fetcher)
monitor.start()
"""

from microstructure.signals.microstructure_signal import (
    MicrostructureSignal,
    MicrostructureHealth,
)
from microstructure.models.amihud import AmihudCalculator, AmihudReading
from microstructure.models.roll_spread import RollSpreadCalculator, RollSpreadReading
from microstructure.models.kyle_lambda import KyleLambdaCalculator, KyleLambdaReading
from microstructure.models.adverse_selection import (
    AdverseSelectionCalculator,
    AdverseSelectionReading,
    AdverseSelectionRisk,
)
from microstructure.models.intraday_patterns import (
    IntradayPatternAnalyzer,
    IntradayMicrostructureProfile,
)
from microstructure.calibrator import MicrostructureCalibrator
from microstructure.live_monitor import LiveMicrostructureMonitor, MonitorConfig
from microstructure.hypothesis_generator import MicrostructureHypothesisGenerator

__all__ = [
    "MicrostructureSignal", "MicrostructureHealth",
    "AmihudCalculator", "AmihudReading",
    "RollSpreadCalculator", "RollSpreadReading",
    "KyleLambdaCalculator", "KyleLambdaReading",
    "AdverseSelectionCalculator", "AdverseSelectionReading", "AdverseSelectionRisk",
    "IntradayPatternAnalyzer", "IntradayMicrostructureProfile",
    "MicrostructureCalibrator",
    "LiveMicrostructureMonitor", "MonitorConfig",
    "MicrostructureHypothesisGenerator",
]
