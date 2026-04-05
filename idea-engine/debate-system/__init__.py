"""
debate-system/__init__.py

Multi-agent debate system for IAE hypothesis validation.

A hypothesis must survive a gauntlet of 6 specialized analyst agents before
being promoted to the backtest queue.  Majority + weighted vote gates promotion.

Quick start
-----------
from debate_system.debate.chamber import DebateChamber
from debate_system.promoter import HypothesisPromoter
from debate_system.track_record import AgentTrackRecordManager

chamber = DebateChamber()
result = chamber.debate(hypothesis, market_data)

promoter = HypothesisPromoter()
promoter.process(result)

tracker = AgentTrackRecordManager(chamber)
tracker.record_debate(result)
# ... later, after backtest:
tracker.resolve(hypothesis_id, backtest_confirmed=True)
"""

from debate_system.debate.chamber import DebateChamber, DebateOutcome, DebateResult
from debate_system.debate.transcript import DebateTranscript, TranscriptStore
from debate_system.promoter import HypothesisPromoter
from debate_system.track_record import AgentTrackRecordManager

__all__ = [
    "DebateChamber",
    "DebateOutcome",
    "DebateResult",
    "DebateTranscript",
    "TranscriptStore",
    "HypothesisPromoter",
    "AgentTrackRecordManager",
]
