"""
version_diff.py
---------------
Diff two StrategyVersion objects: show exactly what changed, in what direction,
and by how much.

Outputs
-------
* VersionDiff.to_plaintext()  -- terminal-friendly text
* VersionDiff.to_html()       -- HTML table for browser / notebook display
* VersionDiff.summary()       -- one-line classification (MAJOR / MINOR / IDENTICAL)

Change classification
---------------------
IDENTICAL : param_hash matches — no diff at all
MINOR     : all deltas have |pct_change| <= 50 % AND no keys added/removed
MAJOR     : any delta with |pct_change| > 50 % OR any key added/removed
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Sequence

from .strategy_version import ParameterDelta, StrategyVersion


# ---------------------------------------------------------------------------
# Change category
# ---------------------------------------------------------------------------

class ChangeCategory:
    IDENTICAL = "IDENTICAL"
    MINOR     = "MINOR"
    MAJOR     = "MAJOR"


# ---------------------------------------------------------------------------
# VersionDiff
# ---------------------------------------------------------------------------

@dataclass
class VersionDiff:
    """
    Full diff between two StrategyVersion objects.

    Attributes
    ----------
    version_a     : the "from" (older / control) version
    version_b     : the "to"   (newer / challenger) version
    deltas        : list of ParameterDelta for every changed parameter
    category      : IDENTICAL | MINOR | MAJOR
    """
    version_a: StrategyVersion
    version_b: StrategyVersion
    deltas: list[ParameterDelta]
    category: str

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def compute(
        cls,
        version_a: StrategyVersion,
        version_b: StrategyVersion,
    ) -> "VersionDiff":
        """
        Compute the diff between version_a and version_b.

        If the two versions share the same param_hash the diff is IDENTICAL
        and deltas will be empty.
        """
        if version_a.param_hash == version_b.param_hash:
            return cls(
                version_a=version_a,
                version_b=version_b,
                deltas=[],
                category=ChangeCategory.IDENTICAL,
            )

        deltas = version_a.compute_delta(version_b)
        any_major = any(d.is_major for d in deltas)
        category = ChangeCategory.MAJOR if any_major else ChangeCategory.MINOR

        return cls(
            version_a=version_a,
            version_b=version_b,
            deltas=deltas,
            category=category,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def major_changes(self) -> list[ParameterDelta]:
        return [d for d in self.deltas if d.is_major]

    @property
    def minor_changes(self) -> list[ParameterDelta]:
        return [d for d in self.deltas if not d.is_major]

    @property
    def added_keys(self) -> list[ParameterDelta]:
        return [d for d in self.deltas if d.old_value is None]

    @property
    def removed_keys(self) -> list[ParameterDelta]:
        return [d for d in self.deltas if d.new_value is None]

    @property
    def modified_keys(self) -> list[ParameterDelta]:
        return [d for d in self.deltas if d.old_value is not None and d.new_value is not None]

    # ------------------------------------------------------------------
    # Summary line
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if self.category == ChangeCategory.IDENTICAL:
            return "IDENTICAL — no parameter changes"
        a_short = self.version_a.version_id[:8]
        b_short = self.version_b.version_id[:8]
        n = len(self.deltas)
        n_major = len(self.major_changes)
        return (
            f"{self.category}: {a_short} -> {b_short} | "
            f"{n} param(s) changed ({n_major} major)"
        )

    # ------------------------------------------------------------------
    # Plaintext output
    # ------------------------------------------------------------------

    def to_plaintext(self) -> str:
        lines: list[str] = []
        a = self.version_a
        b = self.version_b

        lines.append("=" * 72)
        lines.append(f"Strategy Version Diff")
        lines.append(f"  FROM: {a.version_id[:8]}  [{a.status.value}]  {a.description}")
        lines.append(f"  TO:   {b.version_id[:8]}  [{b.status.value}]  {b.description}")
        lines.append(f"  Classification: {self.category}")
        lines.append("=" * 72)

        if self.category == ChangeCategory.IDENTICAL:
            lines.append("  (no changes)")
            return "\n".join(lines)

        if self.added_keys:
            lines.append("\nADDED PARAMETERS:")
            for d in self.added_keys:
                lines.append(f"  + {d.key}: {d.new_value!r}  [NEW KEY]")

        if self.removed_keys:
            lines.append("\nREMOVED PARAMETERS:")
            for d in self.removed_keys:
                lines.append(f"  - {d.key}: {d.old_value!r}  [REMOVED]")

        if self.major_changes:
            lines.append("\nMAJOR CHANGES (>50% shift):")
            for d in [x for x in self.major_changes if x.old_value is not None and x.new_value is not None]:
                arrow = _arrow(d.pct_change)
                pct_str = f"{d.pct_change:+.1f}%" if d.pct_change is not None else "N/A"
                lines.append(f"  {arrow} {d.key}: {d.old_value!r} -> {d.new_value!r}  ({pct_str})")

        if self.minor_changes:
            lines.append("\nMINOR CHANGES (<=50% shift):")
            for d in self.minor_changes:
                arrow = _arrow(d.pct_change)
                pct_str = f"{d.pct_change:+.1f}%" if d.pct_change is not None else "N/A"
                lines.append(f"  {arrow} {d.key}: {d.old_value!r} -> {d.new_value!r}  ({pct_str})")

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML output
    # ------------------------------------------------------------------

    def to_html(self) -> str:
        a = self.version_a
        b = self.version_b

        rows_html = ""
        for d in self.deltas:
            if d.old_value is None:
                row_class = "added"
                change_type = "ADDED"
            elif d.new_value is None:
                row_class = "removed"
                change_type = "REMOVED"
            elif d.is_major:
                row_class = "major"
                change_type = "MAJOR"
            else:
                row_class = "minor"
                change_type = "minor"

            pct_str = f"{d.pct_change:+.1f}%" if d.pct_change is not None else "—"
            arrow_html = _arrow_html(d.pct_change)

            rows_html += f"""
            <tr class="{row_class}">
                <td>{html.escape(d.key)}</td>
                <td>{html.escape(str(d.old_value))}</td>
                <td>{html.escape(str(d.new_value))}</td>
                <td>{arrow_html} {pct_str}</td>
                <td><b>{change_type}</b></td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Strategy Diff: {a.version_id[:8]} → {b.version_id[:8]}</title>
<style>
  body {{ font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
  h2 {{ color: #58a6ff; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #161b22; padding: 8px 12px; text-align: left; color: #8b949e; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #21262d; }}
  tr.added   td {{ background: #0d2a0d; color: #56d364; }}
  tr.removed td {{ background: #2a0d0d; color: #f85149; }}
  tr.major   td {{ background: #2a1f0d; color: #e3b341; }}
  tr.minor   td {{ }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:.8em; }}
  .IDENTICAL {{ background:#21262d; }}
  .MINOR     {{ background:#1f3326; color:#56d364; }}
  .MAJOR     {{ background:#3a1f0d; color:#e3b341; }}
</style>
</head><body>
<h2>Strategy Version Diff</h2>
<p><b>FROM:</b> <code>{a.version_id[:8]}</code> [{a.status.value}] — {html.escape(a.description)}<br>
   <b>TO:</b>   <code>{b.version_id[:8]}</code> [{b.status.value}] — {html.escape(b.description)}</p>
<p>Classification: <span class="pill {self.category}">{self.category}</span>
   &nbsp;|&nbsp; {len(self.deltas)} parameter(s) changed</p>
<table>
  <thead><tr><th>Parameter</th><th>Before</th><th>After</th><th>Change</th><th>Type</th></tr></thead>
  <tbody>{rows_html}
  </tbody>
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arrow(pct: float | None) -> str:
    if pct is None:
        return "~"
    if pct > 0:
        return "↑"
    if pct < 0:
        return "↓"
    return "="


def _arrow_html(pct: float | None) -> str:
    if pct is None:
        return "~"
    if pct > 0:
        return '<span style="color:#56d364">▲</span>'
    if pct < 0:
        return '<span style="color:#f85149">▼</span>'
    return "="


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def diff(a: StrategyVersion, b: StrategyVersion) -> VersionDiff:
    """Shorthand: ``diff(v1, v2)`` returns a VersionDiff."""
    return VersionDiff.compute(a, b)
