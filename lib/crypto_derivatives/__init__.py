"""
crypto_derivatives -- analytics module for perpetuals, options, and DeFi signals.

Provides market signal extraction from derivatives markets to inform spot
crypto trading decisions on BTC, ETH, and SOL via Alpaca.
"""

from lib.crypto_derivatives.perpetuals import (
    FundingRate,
    FundingRateAggregator,
    OpenInterestAnalyzer,
    BasisTracker,
)
from lib.crypto_derivatives.options_market import (
    DeribitOptionChain,
    OptionQuote,
    CryptoImpliedVol,
    CryptoVolRegime,
    DerivativesSignal,
    PerpOptionsComposite,
)
from lib.crypto_derivatives.defi_analytics import (
    AMMLiquidityAnalyzer,
    LendingProtocolSignal,
    BridgeFlowTracker,
)

__all__ = [
    "FundingRate",
    "FundingRateAggregator",
    "OpenInterestAnalyzer",
    "BasisTracker",
    "DeribitOptionChain",
    "OptionQuote",
    "CryptoImpliedVol",
    "CryptoVolRegime",
    "DerivativesSignal",
    "PerpOptionsComposite",
    "AMMLiquidityAnalyzer",
    "LendingProtocolSignal",
    "BridgeFlowTracker",
]
