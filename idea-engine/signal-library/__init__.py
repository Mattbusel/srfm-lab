"""
idea-engine/signal-library/__init__.py
========================================
Public API for the SRFM Signal Library.

Exports
-------
SIGNAL_REGISTRY : dict[str, type]
    Flat mapping of signal_name → Signal class for all 60+ signals.

REGISTRY : SignalRegistry
    The full SignalRegistry object (supports list_by_category, summary, etc.).

All individual signal classes are also importable from this package.

Usage
-----
    from signal_library import SIGNAL_REGISTRY, REGISTRY

    # Instantiate and compute
    cls  = SIGNAL_REGISTRY["rsi"]
    inst = cls(period=14)
    vals = inst.compute(df)

    # List all momentum signals
    print(REGISTRY.list_by_category("momentum"))

    # Registry summary table
    print(REGISTRY.summary())
"""

from __future__ import annotations

# ── Base ──────────────────────────────────────────────────────────────────────
from .base import (
    Signal,
    SignalResult,
    SignalRegistry,
    REGISTRY,
    CATEGORIES,
    SIGNAL_TYPES,
    REQUIRED_OHLCV_COLS,
)

# ── Momentum ──────────────────────────────────────────────────────────────────
from .momentum import (
    EMAMomentum,
    ROC,
    RSI,
    MACD,
    ADX,
    AroonOscillator,
    TrendIntensity,
    MomentumDivergence,
    AccelerationMomentum,
    DualMomentum,
    BHMassSignal,
    VolAdjMomentum,
)

# ── Mean Reversion ────────────────────────────────────────────────────────────
from .mean_reversion import (
    OUZScore,
    BollingerBand,
    RSIMeanReversion,
    KeltnerChannel,
    DonchianBreakout,
    StatArb,
    HalfLifeReversionSpeed,
    MeanReversionVelocity,
    HighLowMeanReversion,
    PriceToMovingAverage,
)

# ── Volatility ────────────────────────────────────────────────────────────────
from .volatility import (
    GARCHVolForecast,
    RealizedVol,
    ParkinsonsVol,
    GarmanKlassVol,
    ATR,
    VolRegime,
    VolOfVol,
    VolCone,
    VolBreakout,
    VolMeanReversion,
)

# ── Cross-Asset ───────────────────────────────────────────────────────────────
from .cross_asset import (
    BTCDominance,
    BTCLead,
    AltSeasonIndex,
    CorrelationBreakdown,
    BetaAdjusted,
    FlightToQuality,
    CryptoGlobalCap,
    DefiVsCex,
)

# ── Macro ─────────────────────────────────────────────────────────────────────
from .macro import (
    MayerMultiple,
    StockToFlowDeviation,
    PiCycleTop,
    NUPLProxy,
    PuellMultiple,
    FearGreedProxy,
)

# ── Microstructure ────────────────────────────────────────────────────────────
from .microstructure import (
    VolumeWeightedMomentum,
    VolumeSpike,
    BuyPressure,
    OBV,
    VPT,
    VolumeMomentumDivergence,
)

# ── Composite ─────────────────────────────────────────────────────────────────
from .composite import (
    SignalEnsemble,
    SignalVoter,
    RegimeConditionalSignal,
    SignalStack,
    AdaptiveSignalWeighter,
)

# ---------------------------------------------------------------------------
# Auto-register all concrete Signal subclasses into REGISTRY
# ---------------------------------------------------------------------------

_ALL_SIGNAL_CLASSES = [
    # Momentum
    EMAMomentum,
    ROC,
    RSI,
    MACD,
    ADX,
    AroonOscillator,
    TrendIntensity,
    MomentumDivergence,
    AccelerationMomentum,
    DualMomentum,
    BHMassSignal,
    VolAdjMomentum,
    # Mean reversion
    OUZScore,
    BollingerBand,
    RSIMeanReversion,
    KeltnerChannel,
    DonchianBreakout,
    StatArb,
    HalfLifeReversionSpeed,
    MeanReversionVelocity,
    HighLowMeanReversion,
    PriceToMovingAverage,
    # Volatility
    GARCHVolForecast,
    RealizedVol,
    ParkinsonsVol,
    GarmanKlassVol,
    ATR,
    VolRegime,
    VolOfVol,
    VolCone,
    VolBreakout,
    VolMeanReversion,
    # Cross-asset
    BTCDominance,
    BTCLead,
    AltSeasonIndex,
    CorrelationBreakdown,
    BetaAdjusted,
    FlightToQuality,
    CryptoGlobalCap,
    DefiVsCex,
    # Macro
    MayerMultiple,
    StockToFlowDeviation,
    PiCycleTop,
    NUPLProxy,
    PuellMultiple,
    FearGreedProxy,
    # Microstructure
    VolumeWeightedMomentum,
    VolumeSpike,
    BuyPressure,
    OBV,
    VPT,
    VolumeMomentumDivergence,
    # Composite
    SignalEnsemble,
    SignalVoter,
    RegimeConditionalSignal,
    SignalStack,
    AdaptiveSignalWeighter,
]

for _cls in _ALL_SIGNAL_CLASSES:
    if _cls.name:
        REGISTRY.register(_cls)

# ---------------------------------------------------------------------------
# Public flat dict for quick lookups (as specified in the module contract)
# ---------------------------------------------------------------------------

SIGNAL_REGISTRY: dict[str, type] = REGISTRY.all_classes()

__all__ = [
    # Core
    "Signal",
    "SignalResult",
    "SignalRegistry",
    "REGISTRY",
    "SIGNAL_REGISTRY",
    "CATEGORIES",
    "SIGNAL_TYPES",
    "REQUIRED_OHLCV_COLS",
    # Momentum
    "EMAMomentum",
    "ROC",
    "RSI",
    "MACD",
    "ADX",
    "AroonOscillator",
    "TrendIntensity",
    "MomentumDivergence",
    "AccelerationMomentum",
    "DualMomentum",
    "BHMassSignal",
    "VolAdjMomentum",
    # Mean reversion
    "OUZScore",
    "BollingerBand",
    "RSIMeanReversion",
    "KeltnerChannel",
    "DonchianBreakout",
    "StatArb",
    "HalfLifeReversionSpeed",
    "MeanReversionVelocity",
    "HighLowMeanReversion",
    "PriceToMovingAverage",
    # Volatility
    "GARCHVolForecast",
    "RealizedVol",
    "ParkinsonsVol",
    "GarmanKlassVol",
    "ATR",
    "VolRegime",
    "VolOfVol",
    "VolCone",
    "VolBreakout",
    "VolMeanReversion",
    # Cross-asset
    "BTCDominance",
    "BTCLead",
    "AltSeasonIndex",
    "CorrelationBreakdown",
    "BetaAdjusted",
    "FlightToQuality",
    "CryptoGlobalCap",
    "DefiVsCex",
    # Macro
    "MayerMultiple",
    "StockToFlowDeviation",
    "PiCycleTop",
    "NUPLProxy",
    "PuellMultiple",
    "FearGreedProxy",
    # Microstructure
    "VolumeWeightedMomentum",
    "VolumeSpike",
    "BuyPressure",
    "OBV",
    "VPT",
    "VolumeMomentumDivergence",
    # Composite
    "SignalEnsemble",
    "SignalVoter",
    "RegimeConditionalSignal",
    "SignalStack",
    "AdaptiveSignalWeighter",
]
