"""
strategy_history.py
-------------------
Full strategy lineage report: tree of all versions, performance at each node,
what changed at each branch.

Key analysis
------------
* ASCII tree of the version lineage
* Performance table (Sharpe, win rate, return) at each node
* Identifying the single parameter change with the biggest historical impact
* Branch comparison: which branches are live, archived, champion
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..versioning.strategy_version import StrategyVersion, VersionStatus
from ..versioning.version_store import VersionStore
from ..versioning.version_diff import VersionDiff
from ..champion.champion_tracker import ChampionTracker


# ---------------------------------------------------------------------------
# NodePerf — performance metadata attached to a version node
# ---------------------------------------------------------------------------

@dataclass
class NodePerf:
    version_id: str
    sharpe: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    n_trades: int = 0
    was_champion: bool = False
    tenure_days: int | None = None

    def __str__(self) -> str:
        flag = " ★" if self.was_champion else ""
        return (
            f"Sharpe={self.sharpe:+.3f}  "
            f"WR={self.win_rate:.1%}  "
            f"Ret={self.total_return:+.1%}  "
            f"N={self.n_trades}{flag}"
        )


# ---------------------------------------------------------------------------
# StrategyHistory
# ---------------------------------------------------------------------------

class StrategyHistory:
    """
    Builds and renders the full strategy version lineage.

    Parameters
    ----------
    version_store    : VersionStore with all versions
    champion_tracker : optional ChampionTracker for tenure data
    perf_data        : optional dict {version_id: NodePerf} for performance overlays
    """

    def __init__(
        self,
        version_store: VersionStore,
        champion_tracker: ChampionTracker | None = None,
        perf_data: dict[str, NodePerf] | None = None,
    ) -> None:
        self.store           = version_store
        self.champion_tracker = champion_tracker
        self.perf_data       = perf_data or {}
        self._versions       = {v.version_id: v for v in version_store.all()}
        self._children       = self._build_child_map()
        self._perf           = self._merge_champion_perf()

    # ------------------------------------------------------------------
    # ASCII tree
    # ------------------------------------------------------------------

    def render_tree(self, show_params: bool = False) -> str:
        roots = [v for v in self._versions.values() if v.parent_id is None]
        if not roots:
            return "(empty lineage)"
        lines: list[str] = ["STRATEGY LINEAGE TREE", "=" * 72]
        for root in roots:
            lines.extend(self._render_node(root, prefix="", is_last=True, show_params=show_params))
        return "\n".join(lines)

    def _render_node(
        self,
        version: StrategyVersion,
        prefix: str,
        is_last: bool,
        show_params: bool,
    ) -> list[str]:
        connector = "└── " if is_last else "├── "
        lines: list[str] = []

        vid   = version.version_id[:8]
        date  = version.created_at[:10]
        status_sym = {
            VersionStatus.CHAMPION: "★",
            VersionStatus.ACTIVE:   "●",
            VersionStatus.ARCHIVED: "○",
            VersionStatus.DRAFT:    "◌",
        }.get(version.status, "?")

        perf_str = ""
        if version.version_id in self._perf:
            perf_str = f"  [{self._perf[version.version_id]}]"

        tag_str = ""
        if version.tags:
            tag_str = f"  tags={version.tags}"

        header = (
            f"{prefix}{connector}{status_sym} {vid}  {date}  "
            f"{version.status.value:<8}  {version.description[:35]}"
            f"{perf_str}{tag_str}"
        )
        lines.append(header)

        if show_params and version.parent_id:
            parent = self._versions.get(version.parent_id)
            if parent:
                diff = VersionDiff.compute(parent, version)
                child_prefix = prefix + ("    " if is_last else "│   ")
                for delta in diff.deltas[:5]:
                    lines.append(
                        f"{child_prefix}  Δ {delta.key}: "
                        f"{delta.old_value!r} → {delta.new_value!r}"
                    )
                if len(diff.deltas) > 5:
                    lines.append(f"{child_prefix}  ... ({len(diff.deltas) - 5} more changes)")

        child_prefix = prefix + ("    " if is_last else "│   ")
        children     = self._children.get(version.version_id, [])
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            lines.extend(
                self._render_node(child, child_prefix, is_child_last, show_params)
            )

        return lines

    # ------------------------------------------------------------------
    # Performance table
    # ------------------------------------------------------------------

    def performance_table(self) -> str:
        versions = list(self._versions.values())
        versions.sort(key=lambda v: v.created_at)

        lines = [
            "PERFORMANCE TABLE",
            f"{'ID':>8}  {'Date':>10}  {'Status':>8}  "
            f"{'Sharpe':>8}  {'WinRate':>8}  {'Return':>8}  "
            f"{'Trades':>7}  Description",
            "-" * 85,
        ]
        for v in versions:
            p = self._perf.get(v.version_id)
            champ = "★" if v.status == VersionStatus.CHAMPION else " "
            if p:
                lines.append(
                    f"{v.version_id[:8]}  {v.created_at[:10]}  "
                    f"{v.status.value:<8}  "
                    f"{p.sharpe:+8.3f}  {p.win_rate:7.1%}  "
                    f"{p.total_return:+7.1%}  {p.n_trades:7d}  "
                    f"{champ}{v.description[:35]}"
                )
            else:
                lines.append(
                    f"{v.version_id[:8]}  {v.created_at[:10]}  "
                    f"{v.status.value:<8}  "
                    f"{'—':>8}  {'—':>8}  {'—':>8}  {'—':>7}  "
                    f"{champ}{v.description[:35]}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Highest-impact change identification
    # ------------------------------------------------------------------

    def most_impactful_changes(self, top_n: int = 10) -> str:
        """
        Identify the parameter changes (from parent to child) that produced
        the largest improvement in Sharpe ratio historically.

        Requires perf_data to be populated.
        """
        if not self._perf:
            return "No performance data available. Populate perf_data to enable this analysis."

        impacts: list[tuple[str, ParameterImpact]] = []

        for vid, version in self._versions.items():
            if version.parent_id is None:
                continue
            parent = self._versions.get(version.parent_id)
            if parent is None:
                continue

            p_perf = self._perf.get(parent.version_id)
            c_perf = self._perf.get(version.version_id)
            if p_perf is None or c_perf is None:
                continue

            sharpe_delta = c_perf.sharpe - p_perf.sharpe
            if abs(sharpe_delta) < 0.001:
                continue

            diff = VersionDiff.compute(parent, version)
            for delta in diff.deltas:
                impacts.append((
                    vid[:8],
                    ParameterImpact(
                        param_key=delta.key,
                        old_value=delta.old_value,
                        new_value=delta.new_value,
                        pct_change=delta.pct_change,
                        sharpe_delta=sharpe_delta,
                        version_from=parent.version_id[:8],
                        version_to=version.version_id[:8],
                    )
                ))

        impacts.sort(key=lambda x: x[1].sharpe_delta, reverse=True)

        lines = [f"MOST IMPACTFUL PARAMETER CHANGES (top {top_n})", "-" * 70]
        for rank, (_, imp) in enumerate(impacts[:top_n], 1):
            dir_sym = "▲" if imp.sharpe_delta > 0 else "▼"
            pct_str = f"({imp.pct_change:+.1f}%)" if imp.pct_change is not None else ""
            lines.append(
                f"{rank:2d}. {dir_sym} {imp.param_key:<30} "
                f"{imp.old_value!r} → {imp.new_value!r} {pct_str:<10}  "
                f"ΔSharpe={imp.sharpe_delta:+.4f}  "
                f"({imp.version_from} → {imp.version_to})"
            )

        if not impacts:
            lines.append("  No parameter-change / Sharpe data pairs found.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Branch comparison
    # ------------------------------------------------------------------

    def branch_comparison(self) -> str:
        """Summary table of all branches (one row per leaf node)."""
        leaves = [
            v for v in self._versions.values()
            if not self._children.get(v.version_id)
        ]
        leaves.sort(key=lambda v: v.created_at)

        lines = ["BRANCH COMPARISON (leaf nodes)", "-" * 70]
        for v in leaves:
            depth = len(self.store.ancestors_of(v.version_id))
            p = self._perf.get(v.version_id)
            perf_str = str(p) if p else "(no perf data)"
            lines.append(
                f"  {v.version_id[:8]}  depth={depth}  "
                f"{v.status.value:<8}  {perf_str}  {v.description[:30]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self, show_params: bool = True) -> str:
        sections = [
            self.render_tree(show_params=show_params),
            self.performance_table(),
            self.most_impactful_changes(),
            self.branch_comparison(),
        ]
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_child_map(self) -> dict[str, list[StrategyVersion]]:
        children: dict[str, list[StrategyVersion]] = defaultdict(list)
        for v in self._versions.values():
            if v.parent_id:
                children[v.parent_id].append(v)
        # Sort children by creation date for deterministic output
        for vid in children:
            children[vid].sort(key=lambda v: v.created_at)
        return dict(children)

    def _merge_champion_perf(self) -> dict[str, NodePerf]:
        perf = dict(self.perf_data)
        if not self.champion_tracker:
            return perf
        for tenure in self.champion_tracker.all_tenures():
            vid = tenure.version_id
            if vid not in perf:
                perf[vid] = NodePerf(version_id=vid)
            perf[vid].was_champion = True
            perf[vid].tenure_days  = tenure.tenure_days
            if tenure.sharpe_at_promo:
                perf[vid].sharpe = tenure.sharpe_at_promo
        return perf


# ---------------------------------------------------------------------------
# ParameterImpact — internal helper
# ---------------------------------------------------------------------------

@dataclass
class ParameterImpact:
    param_key: str
    old_value: Any
    new_value: Any
    pct_change: float | None
    sharpe_delta: float
    version_from: str
    version_to: str
