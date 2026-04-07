"""
srfm-lab options analytics library.

Modules
-------
pricing         -- BlackScholes, BjerksundStensland2002, HestonModel,
                   BinomialTree, MonteCarloPricer
vol_surface     -- SVIModel, SVIParams, SABRModel, VolSurface, VolSmile,
                   LocalVolSurface
greeks          -- GreeksResult, AnalyticalGreeks, NumericalGreeks,
                   GreeksAggregator, ImpliedGreeks
risk            -- OptionsPosition, OptionsPortfolio, DeltaHedger,
                   VegaHedger, RiskLimits
term_structure  -- YieldCurve, DividendSchedule, Dividend, BorrowRate,
                   ForwardPrice
"""

# Pricing models
from lib.options.pricing import (
    BlackScholes,
    BjerksundStensland2002,
    HestonModel,
    BinomialTree,
    MonteCarloPricer,
)

# Volatility surface
from lib.options.vol_surface import (
    SVIModel,
    SVIParams,
    SABRModel,
    VolSurface,
    VolSmile,
    LocalVolSurface,
    build_vol_surface_from_svi,
)

# Greeks engine
from lib.options.greeks import (
    GreeksResult,
    AnalyticalGreeks,
    NumericalGreeks,
    GreeksAggregator,
    ImpliedGreeks,
    StressScenario,
)

# Risk management
from lib.options.risk import (
    OptionsPosition,
    OptionsPortfolio,
    DeltaHedger,
    VegaHedger,
    RiskLimits,
)

# Term structure
from lib.options.term_structure import (
    YieldCurve,
    DividendSchedule,
    Dividend,
    BorrowRate,
    ForwardPrice,
)

__all__ = [
    # Pricing
    "BlackScholes",
    "BjerksundStensland2002",
    "HestonModel",
    "BinomialTree",
    "MonteCarloPricer",
    # Vol surface
    "SVIModel",
    "SVIParams",
    "SABRModel",
    "VolSurface",
    "VolSmile",
    "LocalVolSurface",
    "build_vol_surface_from_svi",
    # Greeks
    "GreeksResult",
    "AnalyticalGreeks",
    "NumericalGreeks",
    "GreeksAggregator",
    "ImpliedGreeks",
    "StressScenario",
    # Risk
    "OptionsPosition",
    "OptionsPortfolio",
    "DeltaHedger",
    "VegaHedger",
    "RiskLimits",
    # Term structure
    "YieldCurve",
    "DividendSchedule",
    "Dividend",
    "BorrowRate",
    "ForwardPrice",
]
