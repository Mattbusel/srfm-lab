"""
idea-engine/autonomous-loop/backtest_bridge.py

BacktestBridge: runs the Python backtester in a subprocess and validates
whether a hypothesis produces a real, statistically significant improvement.

Baseline is cached from the last run and updated after each validated promotion.
"""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[3]
_BACKTEST_SCRIPT = _REPO_ROOT / "tools" / "crypto_backtest_mc.py"
_BASELINE_CACHE = Path(__file__).parent / "backtest_baseline.pkl"
_SUBPROCESS_TIMEOUT = 600  # 10 minutes
_MIN_SHARPE_IMPROVEMENT = 0.05    # must beat baseline Sharpe by at least 0.05
_MIN_DD_IMPROVEMENT = 0.005       # max drawdown must improve by 0.5%
_SIGNIFICANCE_ALPHA = 0.05


@dataclass
class BacktestResult:
    """Results from a single backtest run."""

    hypothesis_id: str
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_pnl: float
    total_trades: int
    params_tested: dict[str, Any]
    raw_returns: list[float] = field(default_factory=list)
    passed_validation: bool = False
    validation_reason: str = ""
    run_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "avg_pnl": self.avg_pnl,
            "total_trades": self.total_trades,
            "passed_validation": self.passed_validation,
            "validation_reason": self.validation_reason,
            "run_at": self.run_at,
        }


@dataclass
class BaselineResult:
    """Cached baseline performance from the last accepted backtest."""

    sharpe_ratio: float = 1.0
    max_drawdown: float = 0.15
    win_rate: float = 0.422
    avg_pnl: float = -6.0
    raw_returns: list[float] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    cached_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BacktestBridge:
    """
    Bridge between the autonomous loop and the Python backtester.

    1. Translates a hypothesis into a modified param set.
    2. Runs crypto_backtest_mc.py as a subprocess.
    3. Parses results and validates via statistical tests.
    4. Updates the cached baseline on validated promotion.
    """

    def __init__(self) -> None:
        self._baseline: BaselineResult = self._load_baseline()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_for_hypothesis(self, hypothesis) -> BacktestResult | None:
        """Full pipeline: translate hypothesis -> run backtest -> validate."""
        hyp_id = getattr(hypothesis, "hypothesis_id", "unknown")
        logger.info("BacktestBridge: running backtest for hypothesis %s", hyp_id[:8])

        try:
            new_params = self.apply_hypothesis_to_params(hypothesis)
            raw = self.run_backtest_subprocess(new_params)
            if raw is None:
                return None

            improvement = self.compare_to_baseline(raw, self._baseline)
            raw.passed_validation = self.validate_improvement(improvement)
            raw.validation_reason = improvement.get("reason", "")
            return raw
        except Exception as exc:
            logger.error("BacktestBridge: backtest failed for %s: %s", hyp_id[:8], exc)
            return None

    def apply_hypothesis_to_params(self, hypothesis) -> dict[str, Any]:
        """
        Translate hypothesis.parameters into a full param dict for the backtester.
        Starts from the current live params and merges the hypothesis deltas.
        """
        base_params = self._read_current_live_params()
        hyp_params = getattr(hypothesis, "parameters", {})

        merged = dict(base_params)
        for key, value in hyp_params.items():
            if key in merged:
                merged[key] = value
            else:
                logger.debug("Ignoring unknown hypothesis param: %s", key)

        return merged

    def run_backtest_subprocess(self, params: dict[str, Any]) -> BacktestResult | None:
        """
        Write params to a temp JSON file and run crypto_backtest_mc.py as a subprocess.
        Parse stdout JSON for results.
        """
        if not _BACKTEST_SCRIPT.exists():
            logger.warning("Backtest script not found: %s", _BACKTEST_SCRIPT)
            return self._stub_result(params)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="bt_params_"
        ) as f:
            json.dump(params, f)
            param_file = f.name

        result_file = param_file.replace("bt_params_", "bt_result_").replace(".json", "_out.json")

        cmd = [
            sys.executable,
            str(_BACKTEST_SCRIPT),
            "--params", param_file,
            "--output", result_file,
            "--quiet",
        ]

        logger.info("BacktestBridge: launching subprocess: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.error("BacktestBridge: subprocess timed out after %ds", _SUBPROCESS_TIMEOUT)
            return None
        except Exception as exc:
            logger.error("BacktestBridge: subprocess error: %s", exc)
            return self._stub_result(params)

        if proc.returncode != 0:
            logger.warning(
                "BacktestBridge: backtest exited %d.\nstderr: %s",
                proc.returncode,
                proc.stderr[:500],
            )
            return self._stub_result(params)

        return self._parse_result(result_file, params)

    def compare_to_baseline(
        self, result: BacktestResult, baseline: BaselineResult
    ) -> dict[str, Any]:
        """
        Compute improvement metrics vs the cached baseline.
        Runs a t-test on raw returns if available.
        """
        sharpe_delta = result.sharpe_ratio - baseline.sharpe_ratio
        dd_delta = baseline.max_drawdown - result.max_drawdown  # positive = improved
        wr_delta = result.win_rate - baseline.win_rate

        p_value = 1.0
        if result.raw_returns and baseline.raw_returns:
            try:
                from scipy import stats  # type: ignore
                _, p_value = stats.ttest_ind(result.raw_returns, baseline.raw_returns)
            except Exception:
                pass

        passes = (
            sharpe_delta >= _MIN_SHARPE_IMPROVEMENT
            and dd_delta >= _MIN_DD_IMPROVEMENT
            and p_value < _SIGNIFICANCE_ALPHA
        )

        reason_parts = []
        if sharpe_delta < _MIN_SHARPE_IMPROVEMENT:
            reason_parts.append(f"sharpe_delta={sharpe_delta:.3f} < {_MIN_SHARPE_IMPROVEMENT}")
        if dd_delta < _MIN_DD_IMPROVEMENT:
            reason_parts.append(f"dd_delta={dd_delta:.4f} < {_MIN_DD_IMPROVEMENT}")
        if p_value >= _SIGNIFICANCE_ALPHA:
            reason_parts.append(f"p_value={p_value:.3f} >= {_SIGNIFICANCE_ALPHA}")

        return {
            "passes": passes,
            "sharpe_delta": sharpe_delta,
            "dd_delta": dd_delta,
            "wr_delta": wr_delta,
            "p_value": p_value,
            "reason": "; ".join(reason_parts) if reason_parts else "all checks passed",
        }

    def validate_improvement(self, improvement: dict[str, Any]) -> bool:
        """Simple gate: improvement dict must have passes=True."""
        return bool(improvement.get("passes", False))

    def update_baseline(self, result: BacktestResult) -> None:
        """Update the cached baseline after a validated promotion."""
        self._baseline = BaselineResult(
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            avg_pnl=result.avg_pnl,
            raw_returns=result.raw_returns,
            params=result.params_tested,
        )
        self._save_baseline()
        logger.info(
            "BacktestBridge: baseline updated — Sharpe=%.3f, MaxDD=%.3f",
            self._baseline.sharpe_ratio,
            self._baseline.max_drawdown,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_current_live_params(self) -> dict[str, Any]:
        """Read current parameter values from the live trader source."""
        live_trader = _REPO_ROOT / "tools" / "live_trader_alpaca.py"
        if not live_trader.exists():
            return {}
        try:
            import ast
            tree = ast.parse(live_trader.read_text(encoding="utf-8"))
            params: dict[str, Any] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and isinstance(
                            node.value, (ast.Constant, ast.UnaryOp)
                        ):
                            try:
                                params[target.id] = ast.literal_eval(node.value)
                            except Exception:
                                pass
            return params
        except Exception as exc:
            logger.warning("Could not read live params: %s", exc)
            return {}

    def _parse_result(
        self, result_file: str, params: dict[str, Any]
    ) -> BacktestResult | None:
        """Parse the JSON output file written by the backtest subprocess."""
        try:
            data = json.loads(Path(result_file).read_text())
            return BacktestResult(
                hypothesis_id=params.get("hypothesis_id", "unknown"),
                sharpe_ratio=float(data.get("sharpe_ratio", 0.0)),
                max_drawdown=float(data.get("max_drawdown", 1.0)),
                win_rate=float(data.get("win_rate", 0.0)),
                avg_pnl=float(data.get("avg_pnl", 0.0)),
                total_trades=int(data.get("total_trades", 0)),
                params_tested=params,
                raw_returns=data.get("returns", []),
            )
        except Exception as exc:
            logger.warning("BacktestBridge: could not parse result file: %s", exc)
            return self._stub_result(params)

    def _stub_result(self, params: dict[str, Any]) -> BacktestResult:
        """Return a placeholder result when the subprocess is unavailable."""
        return BacktestResult(
            hypothesis_id=params.get("hypothesis_id", "stub"),
            sharpe_ratio=self._baseline.sharpe_ratio,
            max_drawdown=self._baseline.max_drawdown,
            win_rate=self._baseline.win_rate,
            avg_pnl=self._baseline.avg_pnl,
            total_trades=0,
            params_tested=params,
            passed_validation=False,
            validation_reason="backtest script unavailable",
        )

    def _load_baseline(self) -> BaselineResult:
        if _BASELINE_CACHE.exists():
            try:
                with open(_BASELINE_CACHE, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, BaselineResult):
                    logger.info(
                        "BacktestBridge: loaded cached baseline (Sharpe=%.3f)", obj.sharpe_ratio
                    )
                    return obj
            except Exception as exc:
                logger.warning("BacktestBridge: could not load baseline cache: %s", exc)
        return BaselineResult()

    def _save_baseline(self) -> None:
        try:
            with open(_BASELINE_CACHE, "wb") as f:
                pickle.dump(self._baseline, f)
        except Exception as exc:
            logger.warning("BacktestBridge: could not save baseline: %s", exc)
