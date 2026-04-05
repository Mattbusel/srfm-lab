"""
strategy_version.py
-------------------
Core dataclasses for strategy versioning in the SRFM trading lab.

StrategyVersion  -- Immutable snapshot of all BH strategy parameters at a point in time.
ParameterDelta   -- What changed from parent to child version.

Versions are content-addressed: the parameter hash deduplicates identical configs.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Status lifecycle
# ---------------------------------------------------------------------------

class VersionStatus(str, Enum):
    """Lifecycle state of a strategy version."""
    DRAFT    = "DRAFT"      # being edited, not yet deployed
    ACTIVE   = "ACTIVE"     # deployed in A/B test or shadow
    ARCHIVED = "ARCHIVED"   # retired, kept for lineage
    CHAMPION = "CHAMPION"   # current live production version


# ---------------------------------------------------------------------------
# ParameterDelta -- diff between two versions
# ---------------------------------------------------------------------------

@dataclass
class ParameterDelta:
    """
    Records what changed from parent_id -> child_id for a single parameter key.

    Attributes
    ----------
    key        : parameter name
    old_value  : value in the parent version (None if the key was added)
    new_value  : value in the child version (None if the key was removed)
    pct_change : signed percentage change (None for non-numeric or additions/removals)
    is_major   : True if |pct_change| > 50 % or the key was added/removed entirely
    """
    key: str
    old_value: Any
    new_value: Any
    pct_change: float | None = None
    is_major: bool = False

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "pct_change": self.pct_change,
            "is_major": self.is_major,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParameterDelta":
        return cls(**d)

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        direction = ""
        if self.pct_change is not None:
            sign = "+" if self.pct_change >= 0 else ""
            direction = f" ({sign}{self.pct_change:.1f}%)"
        tag = " [MAJOR]" if self.is_major else ""
        return f"  {self.key}: {self.old_value!r} -> {self.new_value!r}{direction}{tag}"


# ---------------------------------------------------------------------------
# StrategyVersion
# ---------------------------------------------------------------------------

@dataclass
class StrategyVersion:
    """
    Immutable snapshot of an SRFM/BH strategy configuration.

    Attributes
    ----------
    version_id   : UUID4 string — globally unique identifier
    parent_id    : UUID4 of the parent version; None for the root version
    created_at   : UTC ISO-8601 timestamp
    parameters   : full flat dict of every BH strategy parameter
    param_hash   : SHA-256 of the canonical JSON of parameters (deduplication key)
    description  : human-readable summary of why this version was created
    author       : who or what created this version (e.g., "IAE", "manual", "auto-tune")
    tags         : list of labels such as "iae-wave-1", "pre-corr-fix", "baseline"
    status       : VersionStatus lifecycle state
    iae_idea_ids : list of IAE idea IDs that motivated this version (for attribution)
    notes        : free-form notes (hypothesis text, backtest summary, etc.)
    """
    version_id: str
    parent_id: str | None
    created_at: str
    parameters: dict[str, Any]
    param_hash: str
    description: str
    author: str
    tags: list[str]
    status: VersionStatus
    iae_idea_ids: list[str] = field(default_factory=list)
    notes: str = ""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def new(
        cls,
        parameters: dict[str, Any],
        *,
        parent_id: str | None = None,
        description: str = "",
        author: str = "manual",
        tags: list[str] | None = None,
        status: VersionStatus = VersionStatus.DRAFT,
        iae_idea_ids: list[str] | None = None,
        notes: str = "",
    ) -> "StrategyVersion":
        """Create a new StrategyVersion with auto-generated ID, timestamp, and hash."""
        return cls(
            version_id=str(uuid.uuid4()),
            parent_id=parent_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            parameters=parameters,
            param_hash=cls._hash_parameters(parameters),
            description=description,
            author=author,
            tags=tags or [],
            status=status,
            iae_idea_ids=iae_idea_ids or [],
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Parameter hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_parameters(params: dict[str, Any]) -> str:
        """
        Compute a stable SHA-256 fingerprint of the parameter dict.

        Keys are sorted to ensure identical configs produce identical hashes
        regardless of insertion order.
        """
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    def compute_delta(self, child: "StrategyVersion") -> list[ParameterDelta]:
        """
        Return the list of ParameterDelta between self (parent) and child.

        Only keys that differ are included; unchanged keys are omitted.
        """
        deltas: list[ParameterDelta] = []
        all_keys = set(self.parameters) | set(child.parameters)

        for key in sorted(all_keys):
            old_val = self.parameters.get(key)
            new_val = child.parameters.get(key)

            if old_val == new_val:
                continue  # unchanged

            # Compute % change for numeric scalars
            pct = None
            is_major = False
            if old_val is None or new_val is None:
                is_major = True  # key added or removed
            elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if old_val != 0:
                    pct = (new_val - old_val) / abs(old_val) * 100.0
                    is_major = abs(pct) > 50.0
                else:
                    is_major = new_val != 0

            deltas.append(ParameterDelta(
                key=key,
                old_value=old_val,
                new_value=new_val,
                pct_change=pct,
                is_major=is_major,
            ))

        return deltas

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyVersion":
        d = dict(d)
        d["status"] = VersionStatus(d["status"])
        return cls(**d)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, s: str) -> "StrategyVersion":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_duplicate_of(self, other: "StrategyVersion") -> bool:
        """Two versions with the same param_hash have identical parameter sets."""
        return self.param_hash == other.param_hash

    def promote(self) -> "StrategyVersion":
        """Return a copy of this version with status set to CHAMPION."""
        from dataclasses import replace
        return replace(self, status=VersionStatus.CHAMPION)

    def archive(self) -> "StrategyVersion":
        """Return a copy with status ARCHIVED."""
        from dataclasses import replace
        return replace(self, status=VersionStatus.ARCHIVED)

    def activate(self) -> "StrategyVersion":
        """Return a copy with status ACTIVE."""
        from dataclasses import replace
        return replace(self, status=VersionStatus.ACTIVE)

    def __repr__(self) -> str:
        short_id = self.version_id[:8]
        return (
            f"StrategyVersion(id={short_id}, status={self.status.value}, "
            f"hash={self.param_hash[:8]}, desc={self.description!r})"
        )


# ---------------------------------------------------------------------------
# Default BH/LARSA parameter template
# ---------------------------------------------------------------------------

DEFAULT_BH_PARAMETERS: dict[str, Any] = {
    # --- Universe ---
    "instruments": ["ES", "NQ", "YM", "CL", "GC", "ZB", "NG", "VX"],
    # --- Position sizing ---
    "portfolio_daily_risk": 0.01,
    "inst_correlation": 0.35,
    "per_inst_risk": 0.00181,
    "tail_fixed_capital": 3_000_000.0,
    # --- BH physics ---
    "min_hold_bars": 4,
    "bh_form_override_cl": 1.8,
    "bh_form_override_ng": 1.8,
    "bh_form_override_vx": 1.8,
    # --- Harvest mode ---
    "harvest_risk_per_inst": 0.02,
    "harvest_z_entry": 1.5,
    "harvest_z_exit": 0.3,
    "harvest_z_stop": 2.8,
    "harvest_lookback": 20,
    # --- Instrument caps (notional) ---
    "inst_cap_nq": 400_000,
    "inst_cap_ng": 200_000,
    "inst_cap_vx": 150_000,
    # --- Regime ---
    "regime_lookback": 50,
    "regime_vol_threshold": 0.015,
    # --- Execution ---
    "slippage_bps": 2,
    "commission_per_contract": 2.0,
}
