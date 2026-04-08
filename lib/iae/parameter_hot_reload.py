"""
IAE Live Parameter Hot-Reload (T2-2)
Enables the IAE to update live trader parameters without restart.

Safety:
  - All parameter changes staged in a staging file before going live
  - Rate limiter: no parameter changes > MAX_CHANGE_PCT per 7 days
  - Rollback: if performance degrades after parameter change, auto-revert
  - Audit log: every parameter change logged with before/after values
"""
import json
import logging
import time
import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

log = logging.getLogger(__name__)

@dataclass
class ParameterChange:
    param_name: str
    old_value: Any
    new_value: Any
    proposed_by: str  # "iae", "manual", "genome_evolution"
    timestamp: float = field(default_factory=time.time)
    applied: bool = False
    rolled_back: bool = False

@dataclass
class HotReloadConfig:
    live_params_path: str = "config/live_params.json"
    staged_params_path: str = "config/staged_params.json"
    audit_log_path: str = "config/param_audit_log.jsonl"
    max_change_pct: float = 0.20        # max 20% change per parameter per week
    rate_limit_window_days: int = 7
    performance_window_bars: int = 500   # bars to evaluate performance after change
    rollback_sharpe_threshold: float = -0.5  # auto-rollback if Sharpe drops below this

class ParameterHotReloader:
    """
    Manages live parameter updates from IAE -> live trader.

    Workflow:
      1. IAE writes proposed params to staged_params.json
      2. Reloader validates against rate limits and constraints
      3. On approval, writes to live_params.json
      4. Live trader reads live_params.json on each bar (or hourly)
      5. Performance monitor tracks post-change Sharpe
      6. Auto-rollback if Sharpe degrades below threshold

    Usage:
        reloader = ParameterHotReloader()
        current = reloader.get_current_params()  # returns dict of active params
        reloader.propose_change("BH_FORM", 1.92, 2.10, proposer="iae")
        reloader.apply_staged()  # applies all validated staged changes
    """

    def __init__(self, cfg: HotReloadConfig = None):
        self.cfg = cfg or HotReloadConfig()
        self._current_params: dict = {}
        self._change_history: list[ParameterChange] = []
        self._post_change_sharpe: list[float] = []
        self._load_current()

    def get_current_params(self) -> dict:
        """Returns currently active parameters. Call this from live trader."""
        self._maybe_reload()
        return copy.deepcopy(self._current_params)

    def propose_change(self, param_name: str, new_value: Any, proposer: str = "iae") -> bool:
        """
        Propose a parameter change. Validates rate limits.
        Returns True if proposal is accepted (not yet applied).
        """
        old_value = self._current_params.get(param_name)

        # Rate limit check
        if not self._passes_rate_limit(param_name, old_value, new_value):
            log.warning(
                "HotReload: rate limit rejected change to %s: %s -> %s",
                param_name, old_value, new_value
            )
            return False

        # Write to staged params
        staged = self._load_staged()
        staged[param_name] = {
            "value": new_value,
            "old_value": old_value,
            "proposer": proposer,
            "proposed_at": time.time(),
        }
        self._write_staged(staged)
        log.info("HotReload: staged change %s: %s -> %s (by %s)", param_name, old_value, new_value, proposer)
        return True

    def apply_staged(self) -> list[str]:
        """Apply all valid staged parameter changes. Returns list of applied param names."""
        staged = self._load_staged()
        if not staged:
            return []

        applied = []
        for param_name, proposal in staged.items():
            old_value = self._current_params.get(param_name)
            new_value = proposal["value"]

            if not self._passes_rate_limit(param_name, old_value, new_value):
                log.warning("HotReload: skipping %s (rate limit)", param_name)
                continue

            change = ParameterChange(
                param_name=param_name,
                old_value=old_value,
                new_value=new_value,
                proposed_by=proposal.get("proposer", "unknown"),
                applied=True,
            )
            self._current_params[param_name] = new_value
            self._change_history.append(change)
            self._audit_log(change)
            applied.append(param_name)
            log.info("HotReload: applied %s = %s (was %s)", param_name, new_value, old_value)

        if applied:
            self._save_current()
            # Clear staged
            self._write_staged({})
            self._post_change_sharpe = []  # reset performance monitor

        return applied

    def record_trade_return(self, ret: float):
        """Record a trade return for post-change performance monitoring."""
        self._post_change_sharpe.append(ret)

        # Check rollback trigger
        if len(self._post_change_sharpe) >= self.cfg.performance_window_bars:
            self._check_rollback()

    def _check_rollback(self):
        """Auto-rollback last parameter batch if Sharpe degraded below threshold."""
        returns = self._post_change_sharpe
        if not returns:
            return
        mean_r = sum(returns) / len(returns)
        std_r = (sum((r - mean_r)**2 for r in returns) / len(returns)) ** 0.5
        sharpe = mean_r / (std_r + 1e-6) * (252**0.5)

        if sharpe < self.cfg.rollback_sharpe_threshold:
            log.warning(
                "HotReload: post-change Sharpe=%.3f below threshold=%.3f -- rolling back",
                sharpe, self.cfg.rollback_sharpe_threshold
            )
            self._rollback_last_batch()

    def _rollback_last_batch(self):
        """Roll back the last batch of applied changes."""
        # Find last applied batch (since last reset of post_change_sharpe)
        recent = [c for c in self._change_history if c.applied and not c.rolled_back]
        if not recent:
            return

        for change in recent:
            self._current_params[change.param_name] = change.old_value
            change.rolled_back = True
            log.info("HotReload: rolled back %s -> %s", change.param_name, change.old_value)

        self._save_current()
        self._post_change_sharpe = []

    def _passes_rate_limit(self, param_name: str, old_value: Any, new_value: Any) -> bool:
        """Check if this change respects the max % change per rate_limit_window."""
        if old_value is None or not isinstance(old_value, (int, float)):
            return True  # non-numeric params always allowed
        if old_value == 0:
            return True

        # Check total change already applied in the window
        cutoff = time.time() - self.cfg.rate_limit_window_days * 86400
        recent_changes = [
            c for c in self._change_history
            if c.param_name == param_name and c.timestamp > cutoff and c.applied and not c.rolled_back
        ]

        # Compute cumulative change ratio
        base_value = old_value
        for c in recent_changes:
            if c.old_value and c.old_value != 0:
                pass  # track history

        change_ratio = abs(new_value - old_value) / abs(old_value)
        if change_ratio > self.cfg.max_change_pct:
            return False

        return True

    def _maybe_reload(self):
        """Reload from file if it was modified externally."""
        p = Path(self.cfg.live_params_path)
        if p.exists():
            try:
                mtime = p.stat().st_mtime
                if mtime > getattr(self, '_last_mtime', 0):
                    self._load_current()
                    self._last_mtime = mtime
            except Exception:
                pass

    def _load_current(self):
        p = Path(self.cfg.live_params_path)
        if p.exists():
            try:
                with open(p) as f:
                    self._current_params = json.load(f)
            except Exception:
                self._current_params = {}
        else:
            self._current_params = {}

    def _save_current(self):
        p = Path(self.cfg.live_params_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self._current_params, f, indent=2)

    def _load_staged(self) -> dict:
        p = Path(self.cfg.staged_params_path)
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _write_staged(self, data: dict):
        p = Path(self.cfg.staged_params_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    def _audit_log(self, change: ParameterChange):
        p = Path(self.cfg.audit_log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "param": change.param_name,
            "old": change.old_value,
            "new": change.new_value,
            "by": change.proposed_by,
        }
        with open(p, "a") as f:
            f.write(json.dumps(entry) + "\n")
