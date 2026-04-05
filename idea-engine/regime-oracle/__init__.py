"""
regime-oracle/__init__.py
──────────────────────────
Public API for the Regime Oracle subsystem.

The Regime Oracle classifies the current market into one of six regimes:
  BULL | BEAR | NEUTRAL | CRISIS | RECOVERY | TOPPING

It tags genomes and hypotheses with their optimal regime, monitors for
transitions, and fires alerts to the existing Narrative Intelligence layer.

Typical usage
-------------
    from regime_oracle import RegimeOracle, RegimeFeatureBuilder, GenomeTagger
    from regime_oracle import RegimeAlertMonitor

    oracle = RegimeOracle(db_path="idea_engine.db")
    features = RegimeFeatureBuilder().build_features(ohlcv_df)
    state = oracle.classify(features)
    print(state.regime, state.bull_prob, state.bear_prob)

    tagger = GenomeTagger(db_path="idea_engine.db")
    tagger.update_all_tags()
    routing = tagger.regime_routing_table()
"""

from __future__ import annotations

from .classifier import RegimeOracle, RegimeState, Regime
from .feature_builder import RegimeFeatureBuilder
from .genome_tagger import GenomeTagger
from .alert_monitor import RegimeAlertMonitor

__all__ = [
    # classifier
    "RegimeOracle",
    "RegimeState",
    "Regime",
    # feature_builder
    "RegimeFeatureBuilder",
    # genome_tagger
    "GenomeTagger",
    # alert_monitor
    "RegimeAlertMonitor",
]
