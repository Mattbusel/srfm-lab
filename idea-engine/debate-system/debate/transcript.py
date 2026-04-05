"""
debate-system/debate/transcript.py

Structured debate transcript: records every round of argument, counterargument,
and final position.  Stored as JSON in SQLite for human review and audit.

Schema
------
debate_transcripts table:
  debate_id        TEXT PRIMARY KEY
  hypothesis_id    TEXT
  started_at       TEXT (ISO 8601)
  completed_at     TEXT
  result           TEXT (APPROVED | REJECTED | NEEDS_MORE_DATA)
  transcript_json  TEXT (full structured transcript as JSON)

Transcript JSON structure:
  {
    "debate_id": "...",
    "hypothesis_id": "...",
    "hypothesis_description": "...",
    "rounds": [
      {
        "round": 1,
        "phase": "initial_analysis",
        "arguments": [
          {
            "agent": "StatisticalAnalyst",
            "vote": "FOR",
            "confidence": 0.75,
            "reasoning": "...",
            "key_concerns": [...]
          }, ...
        ]
      },
      {
        "round": 2,
        "phase": "cross_examination",
        "rebuttals": [...]
      }
    ],
    "final_positions": [...],
    "vote_summary": {
      "weighted_for": 0.72,
      "weighted_against": 0.28,
      "total_weight": 1.0,
      "result": "APPROVED"
    }
  }
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from debate_system.agents.base_agent import AnalystVerdict

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS debate_transcripts (
    debate_id        TEXT PRIMARY KEY,
    hypothesis_id    TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    completed_at     TEXT,
    result           TEXT,
    transcript_json  TEXT NOT NULL
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_dt_hypothesis_id
ON debate_transcripts (hypothesis_id);
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DebateArgument:
    """One agent's argument in one round of the debate."""
    agent: str
    vote: str
    confidence: float
    reasoning: str
    key_concerns: list[str] = field(default_factory=list)

    @classmethod
    def from_verdict(cls, verdict: AnalystVerdict) -> "DebateArgument":
        return cls(
            agent=verdict.agent_name,
            vote=verdict.vote.value,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            key_concerns=verdict.key_concerns,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "vote": self.vote,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_concerns": self.key_concerns,
        }


@dataclass
class DebateRound:
    """One round of the multi-round debate."""
    round_number: int
    phase: str           # 'initial_analysis' | 'cross_examination' | 'final_positions'
    arguments: list[DebateArgument] = field(default_factory=list)
    rebuttals: list[dict[str, Any]] = field(default_factory=list)

    def add_argument(self, verdict: AnalystVerdict) -> None:
        self.arguments.append(DebateArgument.from_verdict(verdict))

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "phase": self.phase,
            "arguments": [a.to_dict() for a in self.arguments],
            "rebuttals": self.rebuttals,
        }


@dataclass
class VoteSummary:
    weighted_for: float
    weighted_against: float
    total_weight: float
    result: str
    agent_votes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weighted_for": round(self.weighted_for, 4),
            "weighted_against": round(self.weighted_against, 4),
            "total_weight": round(self.total_weight, 4),
            "result": self.result,
            "agent_votes": self.agent_votes,
        }


@dataclass
class DebateTranscript:
    """Complete structured record of a debate session."""
    debate_id: str
    hypothesis_id: str
    hypothesis_description: str
    started_at: str
    rounds: list[DebateRound] = field(default_factory=list)
    final_positions: list[DebateArgument] = field(default_factory=list)
    vote_summary: VoteSummary | None = None
    completed_at: str | None = None

    @classmethod
    def new(cls, hypothesis_id: str, description: str) -> "DebateTranscript":
        return cls(
            debate_id=str(uuid.uuid4()),
            hypothesis_id=hypothesis_id,
            hypothesis_description=description,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def add_round(self, phase: str) -> DebateRound:
        r = DebateRound(round_number=len(self.rounds) + 1, phase=phase)
        self.rounds.append(r)
        return r

    def set_final_positions(self, verdicts: list[AnalystVerdict]) -> None:
        self.final_positions = [DebateArgument.from_verdict(v) for v in verdicts]

    def close(self, vote_summary: VoteSummary) -> None:
        self.vote_summary = vote_summary
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_description": self.hypothesis_description,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_positions": [fp.to_dict() for fp in self.final_positions],
            "vote_summary": self.vote_summary.to_dict() if self.vote_summary else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# TranscriptStore
# ---------------------------------------------------------------------------

class TranscriptStore:
    """
    Persists and retrieves DebateTranscripts from SQLite.
    Uses the shared idea_engine.db so all IAE components stay in sync.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.execute(CREATE_INDEX_SQL)
            conn.commit()

    def save(self, transcript: DebateTranscript) -> None:
        """Insert or replace the transcript record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO debate_transcripts
                    (debate_id, hypothesis_id, started_at, completed_at,
                     result, transcript_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    transcript.debate_id,
                    transcript.hypothesis_id,
                    transcript.started_at,
                    transcript.completed_at,
                    transcript.vote_summary.result if transcript.vote_summary else None,
                    transcript.to_json(),
                ),
            )
            conn.commit()

    def load(self, debate_id: str) -> DebateTranscript | None:
        """Load a transcript by debate_id. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT transcript_json FROM debate_transcripts WHERE debate_id = ?",
                (debate_id,),
            ).fetchone()
        if row is None:
            return None
        return self._from_json(row["transcript_json"])

    def load_for_hypothesis(self, hypothesis_id: str) -> list[DebateTranscript]:
        """Load all transcripts for a given hypothesis (re-debates over time)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT transcript_json FROM debate_transcripts
                   WHERE hypothesis_id = ? ORDER BY started_at""",
                (hypothesis_id,),
            ).fetchall()
        return [self._from_json(r["transcript_json"]) for r in rows]

    def latest_for_hypothesis(self, hypothesis_id: str) -> DebateTranscript | None:
        """Return the most recent transcript for a hypothesis."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT transcript_json FROM debate_transcripts
                   WHERE hypothesis_id = ?
                   ORDER BY started_at DESC LIMIT 1""",
                (hypothesis_id,),
            ).fetchone()
        if row is None:
            return None
        return self._from_json(row["transcript_json"])

    @staticmethod
    def _from_json(json_str: str) -> DebateTranscript:
        """Reconstruct a DebateTranscript from persisted JSON."""
        data = json.loads(json_str)
        rounds = []
        for rd in data.get("rounds", []):
            round_obj = DebateRound(
                round_number=rd["round"],
                phase=rd["phase"],
                rebuttals=rd.get("rebuttals", []),
            )
            for arg in rd.get("arguments", []):
                round_obj.arguments.append(
                    DebateArgument(
                        agent=arg["agent"],
                        vote=arg["vote"],
                        confidence=arg["confidence"],
                        reasoning=arg["reasoning"],
                        key_concerns=arg.get("key_concerns", []),
                    )
                )
            rounds.append(round_obj)

        vs_data = data.get("vote_summary")
        vote_summary = None
        if vs_data:
            vote_summary = VoteSummary(
                weighted_for=vs_data["weighted_for"],
                weighted_against=vs_data["weighted_against"],
                total_weight=vs_data["total_weight"],
                result=vs_data["result"],
                agent_votes=vs_data.get("agent_votes", []),
            )

        final_positions = [
            DebateArgument(
                agent=fp["agent"],
                vote=fp["vote"],
                confidence=fp["confidence"],
                reasoning=fp["reasoning"],
                key_concerns=fp.get("key_concerns", []),
            )
            for fp in data.get("final_positions", [])
        ]

        transcript = DebateTranscript(
            debate_id=data["debate_id"],
            hypothesis_id=data["hypothesis_id"],
            hypothesis_description=data["hypothesis_description"],
            started_at=data["started_at"],
            rounds=rounds,
            final_positions=final_positions,
            vote_summary=vote_summary,
            completed_at=data.get("completed_at"),
        )
        return transcript
