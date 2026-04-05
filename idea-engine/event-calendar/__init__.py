"""
event-calendar: Crypto and macro event tracking for the SRFM idea engine.

Tracks token unlocks, exchange listings, protocol upgrades, and macro
announcements.  Converts events into trading signals and IAE hypotheses.
"""

from .aggregator import EventAggregator
from .signal_generator import EventSignalGenerator
from .store import EventStore

__all__ = ["EventAggregator", "EventSignalGenerator", "EventStore"]
