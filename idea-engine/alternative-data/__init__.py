"""
alternative-data — Component of the Idea Automation Engine (IAE)
=================================================================
Non-price leading indicator pipeline for crypto strategy signals.

Modules
-------
google_trends     : Google Trends search volume acceleration signals
github_activity   : Developer commit/PR/star activity on core protocol repos
futures_oi        : Binance futures open interest tracking
funding_rates     : Binance perpetual funding rate extremes
exchange_flows    : Simulated on-chain exchange inflow/outflow signals
liquidations      : Liquidation cascade volume tracking
derivatives_signal: Composite DerivativesSignal (OI + funding + liq)
pipeline          : Orchestrates all alt-data fetches and emits to IAE

Public API
----------
    from alternative_data import AltDataPipeline, DerivativesSignal
    from alternative_data.google_trends import GoogleTrendsFetcher
    from alternative_data.futures_oi    import FuturesOIFetcher
    from alternative_data.funding_rates import FundingRateFetcher

Typical usage::

    pipeline = AltDataPipeline(db_path="idea_engine.db")
    results  = pipeline.run_cycle()
"""

from .pipeline           import AltDataPipeline
from .derivatives_signal import DerivativesSignal, DerivativesSignalBuilder

__all__ = [
    "AltDataPipeline",
    "DerivativesSignal",
    "DerivativesSignalBuilder",
]

__version__ = "0.1.0"
