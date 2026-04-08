"""
strategy_library.py
-------------------
Persistent strategy registry with versioning, genealogy tracking,
performance ranking, and full JSON serialization.
"""

from __future__ import annotations

import json
import uuid
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_version(v: str) -> tuple[int, int, int]:
    parts = v.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semver: {v!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _bump_version(v: str, part: str = "patch") -> str:
    major, minor, patch = _parse_version(v)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerformanceSnapshot:
    """A single timestamped snapshot of strategy performance metrics."""
    timestamp: str
    sharpe: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    avg_hold_days: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PerformanceSnapshot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ChangelogEntry:
    timestamp: str
    version: str
    author: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChangelogEntry":
        return cls(**d)


@dataclass
class StrategyRecord:
    """Full representation of a strategy in the library."""
    id: str
    name: str
    version: str
    created_at: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)          # momentum/reversion/arbitrage/macro/micro
    status: str = "active"                                  # active | deprecated | experimental
    parent_id: Optional[str] = None
    forked_from: Optional[str] = None                      # id:version string
    performance_history: list[PerformanceSnapshot] = field(default_factory=list)
    changelog: list[ChangelogEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def latest_performance(self) -> Optional[PerformanceSnapshot]:
        if not self.performance_history:
            return None
        return self.performance_history[-1]

    def performance_array(self, metric: str) -> np.ndarray:
        return np.array([getattr(s, metric, 0.0) for s in self.performance_history], dtype=float)

    def is_decaying(self, metric: str = "sharpe", window: int = 3, threshold: float = 0.10) -> bool:
        """
        Return True if the most recent *window* snapshots show a declining
        trend exceeding *threshold* fraction of the historic mean.
        """
        arr = self.performance_array(metric)
        if len(arr) < window + 1:
            return False
        recent = arr[-window:]
        historic_mean = np.mean(arr[:-window])
        if historic_mean == 0:
            return False
        recent_mean = np.mean(recent)
        return (historic_mean - recent_mean) / abs(historic_mean) > threshold

    def to_dict(self) -> dict:
        d = asdict(self)
        d["performance_history"] = [s.to_dict() for s in self.performance_history]
        d["changelog"] = [c.to_dict() for c in self.changelog]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyRecord":
        perf = [PerformanceSnapshot.from_dict(p) for p in d.pop("performance_history", [])]
        changelog = [ChangelogEntry.from_dict(c) for c in d.pop("changelog", [])]
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        obj.performance_history = perf
        obj.changelog = changelog
        return obj


# ---------------------------------------------------------------------------
# Strategy Library
# ---------------------------------------------------------------------------

class StrategyLibrary:
    """
    Persistent, versioned registry of trading strategies.

    Usage
    -----
    lib = StrategyLibrary()
    sid = lib.add("MomFactor", params={"lookback": 20}, tags=["momentum"])
    lib.record_performance(sid, sharpe=1.2, win_rate=0.55)
    lib.save("strategies.json")
    """

    VALID_TAGS = {"momentum", "reversion", "arbitrage", "macro", "micro",
                  "stat_arb", "ml", "options", "crypto", "equities", "futures"}

    def __init__(self) -> None:
        self._records: dict[str, StrategyRecord] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        description: str = "",
        params: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        version: str = "1.0.0",
        author: str = "system",
        parent_id: Optional[str] = None,
        forked_from: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Register a new strategy. Returns the new strategy id."""
        sid = str(uuid.uuid4())
        tags = [t.lower() for t in (tags or [])]
        record = StrategyRecord(
            id=sid,
            name=name,
            version=version,
            created_at=_utcnow(),
            description=description,
            params=params or {},
            tags=tags,
            parent_id=parent_id,
            forked_from=forked_from,
            metadata=metadata or {},
        )
        record.changelog.append(ChangelogEntry(
            timestamp=_utcnow(),
            version=version,
            author=author,
            description=f"Initial registration of '{name}'",
        ))
        self._records[sid] = record
        return sid

    def get(self, sid: str) -> StrategyRecord:
        if sid not in self._records:
            raise KeyError(f"Strategy {sid!r} not found")
        return self._records[sid]

    def update(
        self,
        sid: str,
        params: Optional[dict] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        bump: str = "patch",
        author: str = "system",
        changelog_note: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """Update a strategy, bump its version, and log the change."""
        rec = self.get(sid)
        new_version = _bump_version(rec.version, bump)
        if params is not None:
            rec.params.update(params)
        if description is not None:
            rec.description = description
        if tags is not None:
            rec.tags = [t.lower() for t in tags]
        if metadata is not None:
            rec.metadata.update(metadata)
        rec.version = new_version
        rec.changelog.append(ChangelogEntry(
            timestamp=_utcnow(),
            version=new_version,
            author=author,
            description=changelog_note or f"Updated to {new_version}",
        ))
        return new_version

    def deprecate(self, sid: str, reason: str = "", author: str = "system") -> None:
        rec = self.get(sid)
        rec.status = "deprecated"
        rec.changelog.append(ChangelogEntry(
            timestamp=_utcnow(),
            version=rec.version,
            author=author,
            description=f"Deprecated. Reason: {reason}",
        ))

    def fork(
        self,
        sid: str,
        new_name: Optional[str] = None,
        param_overrides: Optional[dict] = None,
        author: str = "system",
    ) -> str:
        """Create a child strategy forked from *sid*."""
        parent = self.get(sid)
        merged_params = {**parent.params, **(param_overrides or {})}
        child_name = new_name or f"{parent.name}_fork"
        child_id = self.add(
            name=child_name,
            description=f"Forked from {parent.name} v{parent.version}",
            params=merged_params,
            tags=list(parent.tags),
            version="1.0.0",
            author=author,
            parent_id=sid,
            forked_from=f"{sid}:{parent.version}",
        )
        return child_id

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def record_performance(
        self,
        sid: str,
        sharpe: float = 0.0,
        calmar: float = 0.0,
        win_rate: float = 0.0,
        avg_hold_days: float = 0.0,
        total_return: float = 0.0,
        max_drawdown: float = 0.0,
        num_trades: int = 0,
        note: str = "",
    ) -> None:
        snap = PerformanceSnapshot(
            timestamp=_utcnow(),
            sharpe=sharpe,
            calmar=calmar,
            win_rate=win_rate,
            avg_hold_days=avg_hold_days,
            total_return=total_return,
            max_drawdown=max_drawdown,
            num_trades=num_trades,
            note=note,
        )
        self.get(sid).performance_history.append(snap)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(
        self,
        metric: str = "sharpe",
        active_only: bool = True,
        top_n: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Return list of (sid, metric_value) sorted descending.
        Uses latest performance snapshot.
        """
        results = []
        for sid, rec in self._records.items():
            if active_only and rec.status != "active":
                continue
            snap = rec.latest_performance()
            val = getattr(snap, metric, 0.0) if snap else 0.0
            results.append((sid, val))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n] if top_n else results

    # ------------------------------------------------------------------
    # Search & filter
    # ------------------------------------------------------------------

    def search(self, query: str, active_only: bool = False) -> list[StrategyRecord]:
        """Full-text search over name, description, and tags."""
        q = query.lower()
        out = []
        for rec in self._records.values():
            if active_only and rec.status != "active":
                continue
            text = " ".join([rec.name, rec.description] + rec.tags).lower()
            if q in text:
                out.append(rec)
        return out

    def filter_by_tags(self, tags: list[str], match_all: bool = False) -> list[StrategyRecord]:
        tags_lower = {t.lower() for t in tags}
        out = []
        for rec in self._records.values():
            rec_tags = set(rec.tags)
            if match_all:
                if tags_lower.issubset(rec_tags):
                    out.append(rec)
            else:
                if tags_lower & rec_tags:
                    out.append(rec)
        return out

    def decaying_strategies(
        self, metric: str = "sharpe", window: int = 3, threshold: float = 0.10
    ) -> list[StrategyRecord]:
        return [
            rec for rec in self._records.values()
            if rec.status == "active" and rec.is_decaying(metric, window, threshold)
        ]

    # ------------------------------------------------------------------
    # Genealogy
    # ------------------------------------------------------------------

    def ancestry(self, sid: str) -> list[str]:
        """Return chain of parent ids from *sid* up to the root."""
        chain = []
        current = sid
        visited = set()
        while current:
            if current in visited:
                break
            visited.add(current)
            rec = self._records.get(current)
            if rec is None:
                break
            chain.append(current)
            current = rec.parent_id
        return chain

    def descendants(self, sid: str) -> list[str]:
        """Return all strategies that trace their lineage back to *sid*."""
        result = []
        for child_id, rec in self._records.items():
            if child_id == sid:
                continue
            if sid in self.ancestry(child_id):
                result.append(child_id)
        return result

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, sid_a: str, sid_b: str) -> dict:
        """Head-to-head stats and return correlation between two strategies."""
        rec_a = self.get(sid_a)
        rec_b = self.get(sid_b)

        metrics = ["sharpe", "calmar", "win_rate", "avg_hold_days", "total_return", "max_drawdown"]
        snap_a = rec_a.latest_performance()
        snap_b = rec_b.latest_performance()

        head_to_head = {}
        for m in metrics:
            va = getattr(snap_a, m, 0.0) if snap_a else 0.0
            vb = getattr(snap_b, m, 0.0) if snap_b else 0.0
            head_to_head[m] = {"a": va, "b": vb, "winner": "a" if va >= vb else "b"}

        # Return correlation over shared history length
        ret_a = rec_a.performance_array("total_return")
        ret_b = rec_b.performance_array("total_return")
        min_len = min(len(ret_a), len(ret_b))
        if min_len > 1:
            correlation = float(np.corrcoef(ret_a[-min_len:], ret_b[-min_len:])[0, 1])
        else:
            correlation = float("nan")

        return {
            "strategy_a": {"id": sid_a, "name": rec_a.name, "version": rec_a.version},
            "strategy_b": {"id": sid_b, "name": rec_b.name, "version": rec_b.version},
            "head_to_head": head_to_head,
            "return_correlation": correlation,
            "shared_snapshots": min_len,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "version": "1.0",
            "exported_at": _utcnow(),
            "strategies": {sid: rec.to_dict() for sid, rec in self._records.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str, merge: bool = False) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not merge:
            self._records.clear()
        for sid, rec_dict in data.get("strategies", {}).items():
            self._records[sid] = StrategyRecord.from_dict(rec_dict)

    def export_strategy(self, sid: str) -> dict:
        return self.get(sid).to_dict()

    def import_strategy(self, data: dict, overwrite: bool = False) -> str:
        rec = StrategyRecord.from_dict(data)
        if rec.id in self._records and not overwrite:
            raise ValueError(f"Strategy {rec.id!r} already exists. Use overwrite=True.")
        self._records[rec.id] = rec
        return rec.id

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        total = len(self._records)
        active = sum(1 for r in self._records.values() if r.status == "active")
        deprecated = sum(1 for r in self._records.values() if r.status == "deprecated")
        tag_counts: dict[str, int] = {}
        for rec in self._records.values():
            for t in rec.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        return {
            "total": total,
            "active": active,
            "deprecated": deprecated,
            "experimental": total - active - deprecated,
            "tag_distribution": tag_counts,
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"StrategyLibrary(total={s['total']}, active={s['active']}, "
            f"deprecated={s['deprecated']})"
        )


# ---------------------------------------------------------------------------
# Performance decay detector (standalone utility)
# ---------------------------------------------------------------------------

class DecayDetector:
    """
    Scan a StrategyLibrary and produce a decay report.
    """

    def __init__(
        self,
        library: StrategyLibrary,
        metric: str = "sharpe",
        window: int = 3,
        threshold: float = 0.10,
    ) -> None:
        self.library = library
        self.metric = metric
        self.window = window
        self.threshold = threshold

    def run(self) -> list[dict]:
        results = []
        for rec in self.library._records.values():
            if rec.status != "active":
                continue
            arr = rec.performance_array(self.metric)
            if len(arr) < self.window + 1:
                continue
            recent_mean = float(np.mean(arr[-self.window:]))
            historic_mean = float(np.mean(arr[:-self.window]))
            if historic_mean == 0:
                continue
            decay_pct = (historic_mean - recent_mean) / abs(historic_mean)
            if decay_pct > self.threshold:
                results.append({
                    "id": rec.id,
                    "name": rec.name,
                    "metric": self.metric,
                    "historic_mean": round(historic_mean, 4),
                    "recent_mean": round(recent_mean, 4),
                    "decay_pct": round(decay_pct * 100, 2),
                })
        results.sort(key=lambda x: x["decay_pct"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    lib = StrategyLibrary()

    sid1 = lib.add("MomentumAlpha", description="12-1 mom factor", tags=["momentum", "equities"],
                   params={"lookback": 252, "skip": 21})
    sid2 = lib.add("MeanReversionBeta", description="Z-score reversion", tags=["reversion"],
                   params={"zscore_entry": 2.0, "zscore_exit": 0.0})

    for sharpe, wr in [(1.1, 0.54), (1.3, 0.56), (1.0, 0.53), (0.7, 0.50)]:
        lib.record_performance(sid1, sharpe=sharpe, win_rate=wr, calmar=sharpe * 0.8,
                               total_return=sharpe * 0.1)
    for sharpe, wr in [(0.9, 0.51), (1.1, 0.52), (1.2, 0.54), (1.3, 0.55)]:
        lib.record_performance(sid2, sharpe=sharpe, win_rate=wr, calmar=sharpe * 0.7,
                               total_return=sharpe * 0.08)

    sid3 = lib.fork(sid1, new_name="MomentumAlpha_LT", param_overrides={"lookback": 504})

    lib.update(sid1, params={"skip": 5}, bump="minor", changelog_note="Shorter skip period")
    lib.deprecate(sid2, reason="Regime shift rendered ineffective")

    print(lib)
    print("Ranked by Sharpe:", lib.rank("sharpe"))
    print("Decay report:", DecayDetector(lib).run())
    print("Comparison:", lib.compare(sid1, sid3))
    print("Ancestry of fork:", lib.ancestry(sid3))
    print("Descendants of sid1:", lib.descendants(sid1))
    print("Search 'momentum':", [r.name for r in lib.search("momentum")])


if __name__ == "__main__":
    _smoke_test()
