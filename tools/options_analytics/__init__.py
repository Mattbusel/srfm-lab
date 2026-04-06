"""
options_analytics — quantitative options analysis suite for srfm-lab.

Modules
-------
vol_surface     Implied volatility surface construction, SVI fitting, Greeks
greeks_monitor  Live portfolio Greeks dashboard with Rich terminal display
skew_analyzer   Vol skew, risk-reversals, butterflies, term-structure
options_flow    Unusual-activity scanner, sweep detection, put/call ratios
pricing_models  BS, Binomial, Monte-Carlo/Heston, SABR, Local-Vol engines
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("srfm-lab")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "vol_surface",
    "greeks_monitor",
    "skew_analyzer",
    "options_flow",
    "pricing_models",
]
