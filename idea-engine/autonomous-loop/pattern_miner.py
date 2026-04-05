"""
idea-engine/autonomous-loop/pattern_miner.py

PatternMiner: mine patterns from recent live trade data and compare them
against backtest expectations.

Loads the last N live trades from execution/live_trades.db, runs the full
suite of IAE statistical miners, and returns a prioritised list of
MinedPattern objects with p-values and effect sizes.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Paths relative to this file
_REPO_ROOT = Path(__file__).parents[3]
_LIVE_TRADES_DB = _REPO_ROOT / "execution" / "live_trades.db"
_DEFAULT_N_TRADES = 500
_MIN_TRADES_FOR_MINING = 30
_SIGNIFICANCE_THRESHOLD = 0.05


class PatternMiner:
    """
    Mine statistically significant patterns from live trade history.

    Runs four miners in sequence:
      1. Hourly pattern miner  — best/worst hours of day
      2. Day-of-week miner     — calendar seasonality
      3. Hold-duration miner   — optimal holding time
      4. Symbol P&L miner      — per-symbol alpha decomposition

    Then compares live patterns against backtest baselines. Returns only
    patterns that appear in both datasets (higher conviction) or are
    novel and statistically robust.
    """

    def __init__(
        self,
        live_trades_db: Path | str | None = None,
        n_trades: int = _DEFAULT_N_TRADES,
    ) -> None:
        self.live_trades_db = Path(live_trades_db) if live_trades_db else _LIVE_TRADES_DB
        self.n_trades = n_trades
        self._backtest_baseline: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def mine(self) -> list:
        """
        Run all miners and return List[MinedPattern] sorted by priority.

        If fewer than _MIN_TRADES_FOR_MINING live trades exist, returns [].
        """
        trades = self._load_live_trades()
        if len(trades) < _MIN_TRADES_FOR_MINING:
            logger.info(
                "PatternMiner: only %d live trades — need %d minimum. Skipping.",
                len(trades),
                _MIN_TRADES_FOR_MINING,
            )
            return []

        logger.info("PatternMiner: mining %d live trades …", len(trades))

        patterns: list = []
        patterns.extend(self._mine_hourly(trades))
        patterns.extend(self._mine_day_of_week(trades))
        patterns.extend(self._mine_hold_duration(trades))
        patterns.extend(self._mine_symbol_pnl(trades))

        # Prioritise: patterns confirmed in backtest baseline come first
        patterns = self._prioritise(patterns)

        logger.info("PatternMiner: found %d significant patterns.", len(patterns))
        return patterns

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_live_trades(self) -> list[dict[str, Any]]:
        """Load the last N trades from live_trades.db."""
        if not self.live_trades_db.exists():
            logger.warning("live_trades.db not found at %s", self.live_trades_db)
            return []

        try:
            with sqlite3.connect(self.live_trades_db) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT symbol, side, entry_price, exit_price, pnl_pct,
                           entry_time, exit_time, hold_seconds
                    FROM trades
                    ORDER BY entry_time DESC
                    LIMIT ?
                    """,
                    (self.n_trades,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not load live trades: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Miner implementations
    # ------------------------------------------------------------------

    def _mine_hourly(self, trades: list[dict]) -> list:
        """Find hours of day with statistically different P&L distributions."""
        from scipy import stats  # type: ignore
        from hypothesis.types import MinedPattern

        hour_pnl: dict[int, list[float]] = {h: [] for h in range(24)}
        for t in trades:
            try:
                entry_str = t.get("entry_time", "")
                hour = datetime.fromisoformat(entry_str).hour
                pnl = float(t.get("pnl_pct", 0.0))
                hour_pnl[hour].append(pnl)
            except Exception:
                continue

        all_pnl = [p for ps in hour_pnl.values() for p in ps]
        if len(all_pnl) < 10:
            return []

        patterns = []
        for hour, pnls in hour_pnl.items():
            if len(pnls) < 5:
                continue
            other = [p for h, ps in hour_pnl.items() if h != hour for p in ps]
            if not other:
                continue
            t_stat, p_val = stats.ttest_ind(pnls, other, equal_var=False)
            mean_diff = np.mean(pnls) - np.mean(other)
            effect = mean_diff / (np.std(all_pnl) + 1e-9)
            if p_val < _SIGNIFICANCE_THRESHOLD and abs(effect) > 0.2:
                patterns.append(
                    MinedPattern.create(
                        pattern_type="time_of_day",
                        instruments=list({t["symbol"] for t in trades if t.get("symbol")}),
                        p_value=float(p_val),
                        effect_size=float(effect),
                        ci_lower=float(np.mean(pnls) - 1.96 * np.std(pnls) / max(len(pnls) ** 0.5, 1)),
                        ci_upper=float(np.mean(pnls) + 1.96 * np.std(pnls) / max(len(pnls) ** 0.5, 1)),
                        evidence={
                            "hour": hour,
                            "n_trades": len(pnls),
                            "mean_pnl": float(np.mean(pnls)),
                            "baseline_mean": float(np.mean(other)),
                            "t_stat": float(t_stat),
                        },
                        regime_context={},
                    )
                )
        return patterns

    def _mine_day_of_week(self, trades: list[dict]) -> list:
        """Find days of week with anomalous P&L."""
        from scipy import stats  # type: ignore
        from hypothesis.types import MinedPattern

        dow_pnl: dict[int, list[float]] = {d: [] for d in range(7)}
        for t in trades:
            try:
                entry_str = t.get("entry_time", "")
                dow = datetime.fromisoformat(entry_str).weekday()
                pnl = float(t.get("pnl_pct", 0.0))
                dow_pnl[dow].append(pnl)
            except Exception:
                continue

        all_pnl = [p for ps in dow_pnl.values() for p in ps]
        if len(all_pnl) < 10:
            return []

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        patterns = []
        for dow, pnls in dow_pnl.items():
            if len(pnls) < 5:
                continue
            other = [p for d, ps in dow_pnl.items() if d != dow for p in ps]
            if not other:
                continue
            _, p_val = stats.ttest_ind(pnls, other, equal_var=False)
            effect = (np.mean(pnls) - np.mean(other)) / (np.std(all_pnl) + 1e-9)
            if p_val < _SIGNIFICANCE_THRESHOLD and abs(effect) > 0.25:
                patterns.append(
                    MinedPattern.create(
                        pattern_type="regime_cluster",
                        instruments=list({t["symbol"] for t in trades if t.get("symbol")}),
                        p_value=float(p_val),
                        effect_size=float(effect),
                        ci_lower=float(np.mean(pnls) - 2 * np.std(pnls)),
                        ci_upper=float(np.mean(pnls) + 2 * np.std(pnls)),
                        evidence={
                            "day": day_names[dow],
                            "dow_index": dow,
                            "n_trades": len(pnls),
                            "mean_pnl": float(np.mean(pnls)),
                        },
                        regime_context={},
                    )
                )
        return patterns

    def _mine_hold_duration(self, trades: list[dict]) -> list:
        """Find optimal hold duration buckets via Pearson correlation."""
        from scipy import stats  # type: ignore
        from hypothesis.types import MinedPattern

        pairs = []
        for t in trades:
            try:
                hold = float(t.get("hold_seconds", 0))
                pnl = float(t.get("pnl_pct", 0.0))
                if hold > 0:
                    pairs.append((hold, pnl))
            except Exception:
                continue

        if len(pairs) < 20:
            return []

        holds, pnls = zip(*pairs)
        r, p_val = stats.pearsonr(holds, pnls)
        effect = r

        if p_val < _SIGNIFICANCE_THRESHOLD and abs(effect) > 0.15:
            return [
                MinedPattern.create(
                    pattern_type="anomaly",
                    instruments=list({t["symbol"] for t in trades if t.get("symbol")}),
                    p_value=float(p_val),
                    effect_size=float(effect),
                    ci_lower=float(np.percentile(pnls, 25)),
                    ci_upper=float(np.percentile(pnls, 75)),
                    evidence={
                        "pearson_r": float(r),
                        "n_pairs": len(pairs),
                        "median_hold_h": float(np.median(holds) / 3600),
                        "best_quartile_mean_pnl": float(np.mean(
                            [p for h, p in pairs if h > np.median(holds)]
                        )),
                    },
                    regime_context={},
                )
            ]
        return []

    def _mine_symbol_pnl(self, trades: list[dict]) -> list:
        """Identify symbols with consistently different P&L vs the rest."""
        from scipy import stats  # type: ignore
        from hypothesis.types import MinedPattern

        symbol_pnl: dict[str, list[float]] = {}
        for t in trades:
            sym = t.get("symbol", "")
            pnl = t.get("pnl_pct")
            if sym and pnl is not None:
                symbol_pnl.setdefault(sym, []).append(float(pnl))

        all_pnl = [p for ps in symbol_pnl.values() for p in ps]
        if not all_pnl:
            return []

        patterns = []
        for sym, pnls in symbol_pnl.items():
            if len(pnls) < 10:
                continue
            other = [p for s, ps in symbol_pnl.items() if s != sym for p in ps]
            if not other:
                continue
            _, p_val = stats.ttest_ind(pnls, other, equal_var=False)
            effect = (np.mean(pnls) - np.mean(other)) / (np.std(all_pnl) + 1e-9)
            if p_val < _SIGNIFICANCE_THRESHOLD and abs(effect) > 0.3:
                patterns.append(
                    MinedPattern.create(
                        pattern_type="cross_asset",
                        instruments=[sym],
                        p_value=float(p_val),
                        effect_size=float(effect),
                        ci_lower=float(np.mean(pnls) - 1.96 * np.std(pnls) / max(len(pnls) ** 0.5, 1)),
                        ci_upper=float(np.mean(pnls) + 1.96 * np.std(pnls) / max(len(pnls) ** 0.5, 1)),
                        evidence={
                            "symbol": sym,
                            "n_trades": len(pnls),
                            "mean_pnl": float(np.mean(pnls)),
                            "win_rate": float(sum(1 for p in pnls if p > 0) / len(pnls)),
                        },
                        regime_context={},
                    )
                )
        return patterns

    # ------------------------------------------------------------------
    # Prioritisation
    # ------------------------------------------------------------------

    def _prioritise(self, patterns: list) -> list:
        """
        Sort patterns by priority score: -log(p_value) * |effect_size|.
        Patterns also found in backtest baseline get a 1.5x boost.
        """
        def _score(p) -> float:
            import math
            base = -math.log(max(p.p_value, 1e-10)) * abs(p.effect_size)
            if self._is_confirmed_in_backtest(p):
                base *= 1.5
            return base

        return sorted(patterns, key=_score, reverse=True)

    def _is_confirmed_in_backtest(self, pattern) -> bool:
        """Check if a pattern type/instrument combination exists in the backtest baseline."""
        key = f"{pattern.pattern_type}:{','.join(pattern.instruments)}"
        return key in self._backtest_baseline

    def update_backtest_baseline(self, baseline: dict[str, Any]) -> None:
        """Called by BacktestBridge after each backtest run to update baseline patterns."""
        self._backtest_baseline = baseline
        logger.info(
            "PatternMiner: backtest baseline updated with %d patterns.", len(baseline)
        )
