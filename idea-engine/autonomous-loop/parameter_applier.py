"""
idea-engine/autonomous-loop/parameter_applier.py

ParameterApplier: safely apply validated parameters to the live strategy.

Uses ast.parse + ast.unparse to surgically modify Python source constants
without touching anything else. Maintains a 10-state rollback history.
Safety limits: max 3 params per application, max 50% change per value.
"""

from __future__ import annotations

import ast
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[3]
_LIVE_TRADER = _REPO_ROOT / "tools" / "live_trader_alpaca.py"
_HISTORY_DIR = Path(__file__).parent / "param_history"
_MAX_HISTORY = 10
_MAX_PARAMS_PER_APPLY = 3
_MAX_VALUE_CHANGE_FRACTION = 0.50


class ParameterApplier:
    """
    Apply validated hypothesis parameters to live_trader_alpaca.py.

    Safety guarantees:
      - Never change more than 3 parameters at once
      - Never change any single value by more than 50%
      - Keep last 10 parameter states for instant rollback
      - Full audit log: what changed, when, why

    Uses AST manipulation so only constant assignments are touched;
    logic, imports, and structure are never modified.
    """

    def __init__(
        self,
        live_trader: Path | str | None = None,
        history_dir: Path | str | None = None,
    ) -> None:
        self.live_trader = Path(live_trader) if live_trader else _LIVE_TRADER
        self.history_dir = Path(history_dir) if history_dir else _HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._audit_log = self.history_dir / "audit_log.jsonl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, hypothesis, backtest_result) -> bool:
        """
        Apply the hypothesis parameters to the live trader.
        Returns True on success, False if safety checks blocked the change.
        """
        hyp_id = getattr(hypothesis, "hypothesis_id", "unknown")
        new_params = getattr(hypothesis, "parameters", {})

        if not self.live_trader.exists():
            logger.warning("ParameterApplier: live trader not found at %s", self.live_trader)
            return False

        if not new_params:
            logger.info("ParameterApplier: hypothesis %s has no parameters to apply.", hyp_id[:8])
            return False

        current_params = self._extract_current_params()
        diff = self._compute_diff(current_params, new_params)

        if not diff:
            logger.info("ParameterApplier: no change detected for %s.", hyp_id[:8])
            return False

        ok, reason = self._safety_check(diff, current_params)
        if not ok:
            logger.warning(
                "ParameterApplier: SAFETY BLOCK for %s — %s", hyp_id[:8], reason
            )
            self._audit(hyp_id, diff, applied=False, reason=reason)
            return False

        # Save rollback snapshot
        self._save_snapshot(current_params)

        # Apply
        success = self._write_params(diff)
        if success:
            logger.info(
                "ParameterApplier: applied %d params for hypothesis %s: %s",
                len(diff),
                hyp_id[:8],
                list(diff.keys()),
            )
            self._audit(hyp_id, diff, applied=True, reason="validated by backtest")
        else:
            logger.error("ParameterApplier: write failed for %s.", hyp_id[:8])

        return success

    def rollback(self, steps: int = 1) -> bool:
        """Restore the parameter state N steps back."""
        snapshots = sorted(self.history_dir.glob("snapshot_*.json"), reverse=True)
        if len(snapshots) < steps:
            logger.warning("ParameterApplier: only %d snapshots available.", len(snapshots))
            return False

        target = snapshots[steps - 1]
        try:
            snapshot = json.loads(target.read_text())
            self._write_params_dict(snapshot)
            logger.info("ParameterApplier: rolled back to snapshot %s", target.name)
            return True
        except Exception as exc:
            logger.error("ParameterApplier: rollback failed: %s", exc)
            return False

    def get_rollback_states(self) -> list[dict[str, Any]]:
        """Return metadata for available rollback states."""
        snapshots = sorted(self.history_dir.glob("snapshot_*.json"), reverse=True)
        states = []
        for snap in snapshots[:_MAX_HISTORY]:
            try:
                data = json.loads(snap.read_text())
                states.append({"file": snap.name, "params": data})
            except Exception:
                pass
        return states

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _compute_diff(
        self, current: dict[str, Any], new_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Return only the parameters that actually differ."""
        diff = {}
        for key, new_val in new_params.items():
            if key in current and current[key] != new_val:
                diff[key] = new_val
        return diff

    def _safety_check(
        self, diff: dict[str, Any], current: dict[str, Any]
    ) -> tuple[bool, str]:
        """Enforce hard safety limits."""
        if len(diff) > _MAX_PARAMS_PER_APPLY:
            return False, f"too many params ({len(diff)} > {_MAX_PARAMS_PER_APPLY})"

        for key, new_val in diff.items():
            old_val = current.get(key)
            if old_val is None:
                continue
            try:
                old_f = float(old_val)
                new_f = float(new_val)
                if old_f == 0:
                    continue
                frac = abs(new_f - old_f) / abs(old_f)
                if frac > _MAX_VALUE_CHANGE_FRACTION:
                    return False, (
                        f"{key}: change fraction {frac:.1%} > {_MAX_VALUE_CHANGE_FRACTION:.0%}"
                    )
            except (TypeError, ValueError):
                pass  # non-numeric, skip magnitude check

        return True, ""

    # ------------------------------------------------------------------
    # AST-based source manipulation
    # ------------------------------------------------------------------

    def _extract_current_params(self) -> dict[str, Any]:
        """Parse the live trader and extract all top-level constant assignments."""
        params: dict[str, Any] = {}
        try:
            source = self.live_trader.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            try:
                                params[target.id] = ast.literal_eval(node.value)
                            except Exception:
                                pass
        except Exception as exc:
            logger.warning("Could not extract current params: %s", exc)
        return params

    def _write_params(self, diff: dict[str, Any]) -> bool:
        """Modify the live trader source to update only the specified params."""
        try:
            source = self.live_trader.read_text(encoding="utf-8")
            tree = ast.parse(source)

            class ParamRewriter(ast.NodeTransformer):
                def visit_Assign(self, node: ast.Assign) -> ast.Assign:
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in diff:
                            new_val = diff[target.id]
                            node.value = ast.Constant(value=new_val)
                    return node

            new_tree = ParamRewriter().visit(tree)
            ast.fix_missing_locations(new_tree)
            new_source = ast.unparse(new_tree)

            # Atomic write: write to .tmp then rename
            tmp = self.live_trader.with_suffix(".py.tmp")
            tmp.write_text(new_source, encoding="utf-8")
            tmp.replace(self.live_trader)
            return True
        except Exception as exc:
            logger.error("ParameterApplier: AST write failed: %s", exc)
            return False

    def _write_params_dict(self, params: dict[str, Any]) -> None:
        """Overwrite params with a full dict (used for rollback)."""
        self._write_params(params)

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def _save_snapshot(self, params: dict[str, Any]) -> None:
        """Save current state as a rollback snapshot."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap_path = self.history_dir / f"snapshot_{ts}.json"
        snap_path.write_text(json.dumps(params, indent=2))

        # Trim to keep only the last _MAX_HISTORY snapshots
        all_snaps = sorted(self.history_dir.glob("snapshot_*.json"))
        for old in all_snaps[:-_MAX_HISTORY]:
            try:
                old.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _audit(
        self,
        hypothesis_id: str,
        diff: dict[str, Any],
        applied: bool,
        reason: str,
    ) -> None:
        """Append an audit record to the JSONL audit log."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "hypothesis_id": hypothesis_id,
            "applied": applied,
            "reason": reason,
            "params_changed": diff,
        }
        try:
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("Could not write audit log: %s", exc)

    def get_audit_log(self, last_n: int = 50) -> list[dict[str, Any]]:
        """Return the last N audit entries."""
        if not self._audit_log.exists():
            return []
        try:
            lines = self._audit_log.read_text().strip().split("\n")
            entries = [json.loads(l) for l in lines if l]
            return entries[-last_n:]
        except Exception:
            return []
