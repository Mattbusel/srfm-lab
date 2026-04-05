"""
changelog.py
------------
Auto-generate a human-readable changelog from a sequence of StrategyVersion objects.

How it works
------------
1. Walk the version lineage from oldest to newest.
2. For each adjacent pair, compute a VersionDiff.
3. Convert each ParameterDelta into a natural-language sentence.
4. If the version has iae_idea_ids, attempt to pull hypothesis descriptions
   from the IAE hypothesis table (SQLite) and append them as attribution.
5. Render as Markdown or plaintext.

IAE integration
---------------
Looks for an ``idea_engine.db`` two levels up from this file.
Table: hypotheses(id TEXT, description TEXT, ...)
If not found, idea IDs are cited without descriptions.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .strategy_version import ParameterDelta, StrategyVersion, VersionStatus
from .version_diff import VersionDiff

# Where the IAE stores hypotheses (best-effort; graceful failure if absent)
_IAE_DB_CANDIDATES = [
    Path(__file__).parent.parent.parent.parent / "idea_engine.db",
    Path(__file__).parent.parent.parent / "idea_engine.db",
]


@dataclass
class ChangelogEntry:
    """A single entry in the generated changelog."""
    version_id: str
    version_short: str
    created_at: str
    author: str
    description: str
    status: VersionStatus
    category: str          # MAJOR | MINOR | IDENTICAL | ROOT
    sentences: list[str]   # natural-language change sentences
    iae_attributions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines: list[str] = []
        tag_str = "  `" + "`  `".join(self.tags) + "`" if self.tags else ""
        lines.append(
            f"### [{self.version_short}] {self.description}  "
            f"_{self.created_at[:10]}_ — {self.author}{tag_str}"
        )
        lines.append(f"**{self.category}**")
        for s in self.sentences:
            lines.append(f"- {s}")
        for attr in self.iae_attributions:
            lines.append(f"  > *{attr}*")
        return "\n".join(lines)

    def to_plaintext(self) -> str:
        lines: list[str] = []
        lines.append(
            f"[{self.version_short}] {self.description}  "
            f"({self.created_at[:10]}, {self.author})"
        )
        lines.append(f"  Type: {self.category}")
        for s in self.sentences:
            lines.append(f"  * {s}")
        for attr in self.iae_attributions:
            lines.append(f"    > {attr}")
        return "\n".join(lines)


class ChangelogGenerator:
    """
    Generates a changelog for a sequence of StrategyVersion objects.

    Parameters
    ----------
    iae_db_path : optional override for the IAE SQLite database path.
    """

    def __init__(self, iae_db_path: str | Path | None = None) -> None:
        self._iae_db: Path | None = None
        if iae_db_path:
            self._iae_db = Path(iae_db_path)
        else:
            for candidate in _IAE_DB_CANDIDATES:
                if candidate.exists():
                    self._iae_db = candidate
                    break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, versions: list[StrategyVersion]) -> list[ChangelogEntry]:
        """
        Generate changelog entries for an ordered list of versions (oldest first).
        """
        if not versions:
            return []

        entries: list[ChangelogEntry] = []
        # First version is the root
        root = versions[0]
        entries.append(self._root_entry(root))

        for i in range(1, len(versions)):
            parent = versions[i - 1]
            child = versions[i]
            diff = VersionDiff.compute(parent, child)
            entry = self._entry_from_diff(diff, child)
            entries.append(entry)

        return entries

    def to_markdown(self, versions: list[StrategyVersion], title: str = "Strategy Changelog") -> str:
        entries = self.generate(versions)
        lines = [f"# {title}", ""]
        for e in reversed(entries):  # newest first
            lines.append(e.to_markdown())
            lines.append("")
        return "\n".join(lines)

    def to_plaintext(self, versions: list[StrategyVersion], title: str = "Strategy Changelog") -> str:
        entries = self.generate(versions)
        sep = "-" * 70
        lines = [title, sep]
        for e in reversed(entries):
            lines.append(e.to_plaintext())
            lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _root_entry(self, v: StrategyVersion) -> ChangelogEntry:
        return ChangelogEntry(
            version_id=v.version_id,
            version_short=v.version_id[:8],
            created_at=v.created_at,
            author=v.author,
            description=v.description or "Initial version",
            status=v.status,
            category="ROOT",
            sentences=["Initial strategy version created."],
            iae_attributions=self._fetch_iae_attributions(v.iae_idea_ids),
            tags=v.tags,
        )

    def _entry_from_diff(self, diff: VersionDiff, child: StrategyVersion) -> ChangelogEntry:
        sentences = [self._delta_to_sentence(d) for d in diff.deltas]
        if not sentences:
            sentences = ["No parameter changes from previous version."]

        return ChangelogEntry(
            version_id=child.version_id,
            version_short=child.version_id[:8],
            created_at=child.created_at,
            author=child.author,
            description=child.description,
            status=child.status,
            category=diff.category,
            sentences=sentences,
            iae_attributions=self._fetch_iae_attributions(child.iae_idea_ids),
            tags=child.tags,
        )

    # ------------------------------------------------------------------
    # Delta -> natural language
    # ------------------------------------------------------------------

    @staticmethod
    def _delta_to_sentence(d: ParameterDelta) -> str:
        key_human = d.key.replace("_", " ").title()

        # Key added
        if d.old_value is None:
            return f"Added new parameter '{key_human}' = {d.new_value!r}."

        # Key removed
        if d.new_value is None:
            return f"Removed parameter '{key_human}' (was {d.old_value!r})."

        # List change (e.g. instrument universe)
        if isinstance(d.old_value, list) and isinstance(d.new_value, list):
            added   = [x for x in d.new_value if x not in d.old_value]
            removed = [x for x in d.old_value if x not in d.new_value]
            parts: list[str] = []
            if added:
                parts.append(f"Added {added} to '{key_human}'")
            if removed:
                parts.append(f"Removed {removed} from '{key_human}'")
            if not parts:
                return f"Reordered '{key_human}' list."
            return "; ".join(parts) + "."

        # Numeric change
        if d.pct_change is not None:
            direction = "Raised" if d.pct_change > 0 else "Lowered"
            pct_str   = f"{abs(d.pct_change):.1f}%"
            qualifier = " (large shift)" if d.is_major else ""
            return (
                f"{direction} '{key_human}' from {d.old_value} to {d.new_value} "
                f"({pct_str}{qualifier})."
            )

        # Generic
        return f"Changed '{key_human}' from {d.old_value!r} to {d.new_value!r}."

    # ------------------------------------------------------------------
    # IAE attribution
    # ------------------------------------------------------------------

    def _fetch_iae_attributions(self, idea_ids: list[str]) -> list[str]:
        if not idea_ids or not self._iae_db:
            return [f"IAE idea #{iid}" for iid in idea_ids]

        attributions: list[str] = []
        try:
            conn = sqlite3.connect(self._iae_db)
            conn.row_factory = sqlite3.Row
            for iid in idea_ids:
                desc = self._lookup_idea(conn, iid)
                if desc:
                    attributions.append(f"IAE idea #{iid}: {desc}")
                else:
                    attributions.append(f"IAE idea #{iid}")
            conn.close()
        except Exception:
            attributions = [f"IAE idea #{iid}" for iid in idea_ids]

        return attributions

    @staticmethod
    def _lookup_idea(conn: sqlite3.Connection, idea_id: str) -> str | None:
        """
        Attempt to look up an IAE idea description.
        Tries several common table/column naming conventions gracefully.
        """
        candidates = [
            ("hypotheses", "description"),
            ("hypotheses", "summary"),
            ("ideas",       "description"),
            ("ideas",       "hypothesis"),
        ]
        for table, col in candidates:
            try:
                row = conn.execute(
                    f"SELECT {col} FROM {table} WHERE id = ? LIMIT 1", (idea_id,)
                ).fetchone()
                if row:
                    return str(row[0])[:200]
            except sqlite3.OperationalError:
                continue
        return None


# ---------------------------------------------------------------------------
# Quick helper
# ---------------------------------------------------------------------------

def generate_changelog(
    versions: list[StrategyVersion],
    fmt: str = "markdown",
    iae_db_path: str | Path | None = None,
) -> str:
    """
    Convenience wrapper.

    Parameters
    ----------
    versions     : ordered list (oldest first)
    fmt          : "markdown" | "plaintext"
    iae_db_path  : optional override for IAE DB
    """
    gen = ChangelogGenerator(iae_db_path=iae_db_path)
    if fmt == "markdown":
        return gen.to_markdown(versions)
    return gen.to_plaintext(versions)
