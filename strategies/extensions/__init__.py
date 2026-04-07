"""
strategies/extensions -- SRFM extension strategies.
Wave4, MeanReversion, Volatility, and RegimeAdaptive sub-strategies.
"""
from .wave4_strategy import Wave4Detector, Wave4Signal, Wave4StrategyAdapter
from .mean_reversion_strategy import MeanReversionEnsemble, OUParams, MRSignal
from .volatility_strategy import VolatilityBreakoutStrategy, GARCHVolForecast, GARCHParams
from .regime_adaptive_strategy import RegimeAdaptiveStrategy, RegimeDetector, Regime
