"""
alt_data -- Alternative data and macro regime signals for quantitative crypto trading.

Modules:
  macro_regime        -- Macro regime classification from multi-source signals
  on_chain_advanced   -- Advanced on-chain analytics (realized price, exchange flows, miners)
  satellite_web       -- Web proxy signals (trends, GitHub activity, app store proxy)
  cross_asset_signals -- Cross-asset signals informing crypto positioning
"""

from .macro_regime import MacroRegimeClassifier, MacroIndicator, MacroRegime, YieldCurveMonitor, CreditSpreadMonitor
from .on_chain_advanced import RealizedPriceBands, ExchangeFlowAnalyzer, MinerSignal, StablecoinRatio, NVTSignal
from .satellite_web import GoogleTrendsProxy, GitHubActivitySignal, AppStoreProxy
from .cross_asset_signals import EquityCryptoCorrelation, DollarCycleSignal, RateImpact, CrossAssetMomentum

__all__ = [
    "MacroRegimeClassifier", "MacroIndicator", "MacroRegime", "YieldCurveMonitor", "CreditSpreadMonitor",
    "RealizedPriceBands", "ExchangeFlowAnalyzer", "MinerSignal", "StablecoinRatio", "NVTSignal",
    "GoogleTrendsProxy", "GitHubActivitySignal", "AppStoreProxy",
    "EquityCryptoCorrelation", "DollarCycleSignal", "RateImpact", "CrossAssetMomentum",
]
