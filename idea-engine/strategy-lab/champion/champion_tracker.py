"""
champion_tracker.py
-------------------
Tracks the current CHAMPION strategy version and its entire lineage.

ChampionTenure
--------------
Records how long each version held the champion title, what displaced it,
and what its terminal performance was.

ChampionTracker
---------------
* Log when a version becomes champion.
* Log when it is displaced and why.
* Performance attribution: which IAE ideas contributed most to champion
  improvements over time (delta Sharpe per IAE idea).
* Query history: all champion versions, sorted by tenure.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np

from ..versioning.strategy_version import StrategyVersion
from ..versioning.version_store import VersionStore

_DEFAULT_DB = Path(__file__).parent.parent / "strategy_lab.db"


# ---------------------------------------------------------------------------
# ChampionTenure
# ---------------------------------------------------------------------------

@dataclass
class ChampionTenure:
    """
    Records the period in which a strategy version held the CHAMPION title.

    Attributes
    ----------
    version_id       : strategy version
    promoted_at      : ISO timestamp when it became champion
    demoted_at       : ISO timestamp when it was displaced (None = current champion)
    tenure_days      : number of days as champion (None if still active)
    replaced_by_id   : version_id of the version that replaced it
    reason_promoted  : why it was promoted
    reason_demoted   : why it was demoted
    sharpe_at_promo  : estimated Sharpe at promotion time (from backtest / A/B)
    sharpe_at_demotion: live Sharpe when demoted
    iae_idea_ids     : IAE ideas that contributed to this champion
    """
    version_id: str
    promoted_at: str
    demoted_at: str | None
    tenure_days: int | None
    replaced_by_id: str | None
    reason_promoted: str
    reason_demoted: str
    sharpe_at_promo: float
    sharpe_at_demotion: float
    iae_idea_ids: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChampionTenure":
        return cls(**d)

    def __str__(self) -> str:
        dur = f"{self.tenure_days}d" if self.tenure_days is not None else "ongoing"
        return (
            f"Champion {self.version_id[:8]} | "
            f"tenure={dur} | "
            f"Sharpe {self.sharpe_at_promo:.3f} -> {self.sharpe_at_demotion:.3f}"
        )


# ---------------------------------------------------------------------------
# ChampionTracker
# ---------------------------------------------------------------------------

class ChampionTracker:
    """
    Persistent log of every version that has ever been champion.

    Parameters
    ----------
    version_store : VersionStore for loading version metadata
    db_path       : SQLite path (reuses strategy_lab.db)
    """

    def __init__(
        self,
        version_store: VersionStore,
        db_path: str | Path = _DEFAULT_DB,
    ) -> None:
        self.version_store = version_store
        self.db_path       = Path(db_path)
        self._init_db()

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    def record_promotion(
        self,
        version_id: str,
        *,
        reason: str = "",
        sharpe_at_promo: float = 0.0,
    ) -> ChampionTenure:
        """Record that version_id became CHAMPION."""
        version = self.version_store.load(version_id)
        iae_ids = version.iae_idea_ids if version else []

        tenure = ChampionTenure(
            version_id=version_id,
            promoted_at=_now_iso(),
            demoted_at=None,
            tenure_days=None,
            replaced_by_id=None,
            reason_promoted=reason,
            reason_demoted="",
            sharpe_at_promo=sharpe_at_promo,
            sharpe_at_demotion=0.0,
            iae_idea_ids=iae_ids,
        )
        self._save_tenure(tenure)
        return tenure

    def record_demotion(
        self,
        version_id: str,
        replaced_by_id: str,
        *,
        reason: str = "",
        sharpe_at_demotion: float = 0.0,
    ) -> ChampionTenure | None:
        """Record that version_id was dethroned by replaced_by_id."""
        tenure = self._load_tenure(version_id)
        if tenure is None:
            return None

        promoted_dt = datetime.fromisoformat(tenure.promoted_at)
        now_dt      = datetime.now(timezone.utc)
        tenure_days = (now_dt - promoted_dt).days

        tenure.demoted_at        = _now_iso()
        tenure.tenure_days       = tenure_days
        tenure.replaced_by_id    = replaced_by_id
        tenure.reason_demoted    = reason
        tenure.sharpe_at_demotion = sharpe_at_demotion
        self._save_tenure(tenure)
        return tenure

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def current_champion_tenure(self) -> ChampionTenure | None:
        """Return the tenure record for the currently active champion."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT tenure_json FROM champion_tenures WHERE demoted_at IS NULL ORDER BY promoted_at DESC LIMIT 1"
            ).fetchone()
        return ChampionTenure.from_dict(json.loads(row[0])) if row else None

    def all_tenures(self) -> list[ChampionTenure]:
        """Return all champion tenures, ordered from oldest to newest."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT tenure_json FROM champion_tenures ORDER BY promoted_at"
            ).fetchall()
        return [ChampionTenure.from_dict(json.loads(r[0])) for r in rows]

    def by_tenure_length(self) -> list[ChampionTenure]:
        """Return all tenures sorted by duration (longest first)."""
        tenures = [t for t in self.all_tenures() if t.tenure_days is not None]
        return sorted(tenures, key=lambda t: t.tenure_days or 0, reverse=True)

    # ------------------------------------------------------------------
    # Performance attribution
    # ------------------------------------------------------------------

    def iae_attribution(self) -> dict[str, float]:
        """
        Compute Sharpe delta attributed to each IAE idea across champion history.

        Methodology
        -----------
        For each champion tenure, compute delta_sharpe = sharpe_at_demotion - sharpe_at_promo
        (or most recent vs promotion if still active).
        Assign that delta evenly across all iae_idea_ids of that version.
        Return a dict {idea_id: cumulative_attributed_delta_sharpe}.
        """
        attribution: dict[str, float] = {}
        tenures = self.all_tenures()

        for tenure in tenures:
            if not tenure.iae_idea_ids:
                continue
            delta = tenure.sharpe_at_demotion - tenure.sharpe_at_promo
            per_idea = delta / len(tenure.iae_idea_ids)
            for iid in tenure.iae_idea_ids:
                attribution[iid] = attribution.get(iid, 0.0) + per_idea

        return dict(sorted(attribution.items(), key=lambda x: x[1], reverse=True))

    def sharpe_trajectory(self) -> list[tuple[str, float]]:
        """
        Return (version_id_short, sharpe_at_promo) sorted chronologically.
        Useful for plotting the champion Sharpe over time.
        """
        return [
            (t.version_id[:8], t.sharpe_at_promo)
            for t in self.all_tenures()
        ]

    def improvement_summary(self) -> str:
        """Human-readable summary of champion improvements over time."""
        tenures = self.all_tenures()
        if not tenures:
            return "No champion history yet."

        lines = ["Champion Improvement History", "=" * 50]
        for i, t in enumerate(tenures):
            is_current = t.demoted_at is None
            dur_str = f"{t.tenure_days}d" if t.tenure_days is not None else "ongoing"
            flag = " ← CURRENT" if is_current else ""
            lines.append(
                f"{i+1:2d}. {t.version_id[:8]}  "
                f"Sharpe={t.sharpe_at_promo:.3f}  "
                f"tenure={dur_str}  "
                f"IAE={t.iae_idea_ids or '—'}{flag}"
            )

        # Overall improvement
        first_sharpe = tenures[0].sharpe_at_promo
        last_sharpe  = tenures[-1].sharpe_at_promo
        improvement  = last_sharpe - first_sharpe
        lines.append("-" * 50)
        lines.append(
            f"Total Sharpe improvement: {first_sharpe:.3f} → {last_sharpe:.3f} "
            f"({improvement:+.3f})"
        )

        # Top IAE ideas
        attr = self.iae_attribution()
        if attr:
            lines.append("\nTop IAE ideas by Sharpe attribution:")
            for rank, (iid, delta) in enumerate(list(attr.items())[:5], 1):
                lines.append(f"  {rank}. #{iid}  +{delta:.4f} Sharpe")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_tenure(self, tenure: ChampionTenure) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO champion_tenures
                   (version_id, promoted_at, demoted_at, tenure_json)
                   VALUES (?,?,?,?)""",
                (
                    tenure.version_id,
                    tenure.promoted_at,
                    tenure.demoted_at,
                    json.dumps(tenure.to_dict(), default=str),
                ),
            )

    def _load_tenure(self, version_id: str) -> ChampionTenure | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT tenure_json FROM champion_tenures WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        return ChampionTenure.from_dict(json.loads(row[0])) if row else None

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS champion_tenures (
                    version_id   TEXT PRIMARY KEY,
                    promoted_at  TEXT NOT NULL,
                    demoted_at   TEXT,
                    tenure_json  TEXT NOT NULL
                );
            """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
