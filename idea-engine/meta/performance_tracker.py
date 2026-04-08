"""
idea-engine/meta/performance_tracker.py

Comprehensive performance tracking for the idea engine.

Tracks PnL, Sharpe, win rates, and regime performance across hypothesis templates.

Public API:
  - HypothesisPerformance dataclass: per-hypothesis performance record
  - PerformanceTracker:
      record_trade(hypothesis_id, entry_price, exit_price, size, direction, regime)
      rolling_metrics(window=30) → recent performance stats
      by_template_type() → aggregate stats per template type
      by_regime() → performance breakdown by market regime
      best_performers(n=10) → top hypothesis templates
      worst_performers(n=10) → hypothesis templates to prune
      regime_conditional_sharpe() → Sharpe by regime × template
      attribution_report() → full attribution breakdown
      export_json() → serializable performance record
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """A single completed trade."""
    trade_id: str
    hypothesis_id: str
    template_type: str
    entry_price: float
    exit_price: float
    size: float              # position size (notional or units)
    direction: int           # +1 = long, -1 = short
    regime: str              # market regime at entry
    entry_time: float        # unix timestamp
    exit_time: float         # unix timestamp
    hold_bars: int           # number of periods held

    @property
    def pnl(self) -> float:
        """Profit / loss from this trade."""
        raw = (self.exit_price - self.entry_price) / self.entry_price
        return raw * self.direction * self.size

    @property
    def return_pct(self) -> float:
        """Return as a percentage of entry price."""
        return (self.exit_price - self.entry_price) / self.entry_price * self.direction * 100.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class HypothesisPerformance:
    """Performance summary for a single hypothesis template."""
    hypothesis_id: str
    template_type: str

    # PnL metrics
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    gross_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0

    # Trade statistics
    n_trades: int = 0
    n_winners: int = 0
    n_losers: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0    # gross_win / gross_loss

    # Risk-adjusted metrics
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0           # annualized return / max drawdown
    information_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Regime breakdown
    regime_pnl: Dict[str, float] = field(default_factory=dict)
    best_regime: str = ""
    worst_regime: str = ""

    # Time statistics
    avg_hold_bars: float = 0.0
    first_trade_time: float = 0.0
    last_trade_time: float = 0.0

    # Metadata
    last_updated: float = field(default_factory=time.time)


@dataclass
class RollingMetrics:
    """Performance metrics over a rolling window."""
    window: int
    n_trades: int
    pnl: float
    win_rate: float
    sharpe: float
    sortino: float
    avg_return: float
    vol_of_returns: float
    max_drawdown: float


@dataclass
class AttributionReport:
    """Full performance attribution breakdown."""
    total_pnl: float
    total_trades: int
    overall_sharpe: float
    overall_win_rate: float
    overall_max_drawdown: float

    by_template: Dict[str, Dict[str, Any]]
    by_regime: Dict[str, Dict[str, Any]]
    by_template_regime: Dict[str, Dict[str, float]]   # template × regime → Sharpe

    top_contributors: List[Tuple[str, float]]         # (hypothesis_id, pnl)
    top_detractors: List[Tuple[str, float]]

    generated_at: str


# ── Helper Functions ──────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio from a return series."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = excess.std(ddof=1)
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, mar: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < mar]
    if len(downside) < 2:
        return float(returns.mean() * periods_per_year / 1e-10)
    downside_vol = downside.std(ddof=1)
    if downside_vol < 1e-10:
        return 0.0
    return float((returns.mean() - mar / periods_per_year) / downside_vol * math.sqrt(periods_per_year))


def _max_drawdown(cumulative_pnl: np.ndarray) -> Tuple[float, int]:
    """
    Maximum drawdown and duration from cumulative PnL series.
    Returns (max_drawdown, max_drawdown_duration_bars).
    """
    if len(cumulative_pnl) < 2:
        return 0.0, 0
    peak = cumulative_pnl[0]
    max_dd = 0.0
    max_dur = 0
    dd_start = 0
    for i, val in enumerate(cumulative_pnl):
        if val > peak:
            peak = val
            dd_start = i
        dd = (peak - val) / max(abs(peak), 1e-10)
        if dd > max_dd:
            max_dd = dd
            max_dur = i - dd_start
    return float(max_dd), max_dur


def _compute_hypothesis_performance(
    hypothesis_id: str,
    template_type: str,
    trades: List[TradeRecord],
) -> HypothesisPerformance:
    """Compute full performance metrics for a hypothesis from its trade list."""
    if not trades:
        return HypothesisPerformance(
            hypothesis_id=hypothesis_id,
            template_type=template_type,
        )

    pnls = np.array([t.pnl for t in trades])
    rets = np.array([t.return_pct / 100.0 for t in trades])

    total_pnl = float(pnls.sum())
    n_trades = len(trades)
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    n_win = len(winners)
    n_loss = len(losers)
    win_rate = n_win / n_trades if n_trades > 0 else 0.0

    avg_win = float(np.mean([t.pnl for t in winners])) if winners else 0.0
    avg_loss = float(np.mean([t.pnl for t in losers])) if losers else 0.0

    gross_win = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    profit_factor = gross_win / max(gross_loss, 1e-10)

    sharpe = _sharpe(rets)
    sortino = _sortino(rets)

    cum_pnl = np.cumsum(pnls)
    max_dd, dd_dur = _max_drawdown(cum_pnl)
    avg_dd = float(np.mean(np.maximum(np.maximum.accumulate(cum_pnl) - cum_pnl, 0.0)))

    ann_return = float(rets.mean() * 252)
    calmar = ann_return / max(max_dd, 1e-6)

    # Regime breakdown
    regime_pnl: Dict[str, float] = defaultdict(float)
    for t in trades:
        regime_pnl[t.regime] += t.pnl
    regime_pnl = dict(regime_pnl)

    best_regime = max(regime_pnl, key=regime_pnl.get, default="")
    worst_regime = min(regime_pnl, key=regime_pnl.get, default="")

    avg_hold = float(np.mean([t.hold_bars for t in trades]))
    times = [t.entry_time for t in trades]

    return HypothesisPerformance(
        hypothesis_id=hypothesis_id,
        template_type=template_type,
        total_pnl=total_pnl,
        total_return_pct=float(rets.sum() * 100.0),
        gross_pnl=float(gross_win),
        max_pnl=float(pnls.max()),
        min_pnl=float(pnls.min()),
        n_trades=n_trades,
        n_winners=n_win,
        n_losers=n_loss,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        max_drawdown_duration=dd_dur,
        regime_pnl=regime_pnl,
        best_regime=best_regime,
        worst_regime=worst_regime,
        avg_hold_bars=avg_hold,
        first_trade_time=min(times) if times else 0.0,
        last_trade_time=max(times) if times else 0.0,
        last_updated=time.time(),
    )


# ── Performance Tracker ───────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Comprehensive performance tracker for the idea engine.
    Maintains a ledger of all trades across hypothesis templates,
    computes rolling and cumulative performance metrics, and produces
    attribution reports.
    """

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self._trades: List[TradeRecord] = []
        self._trade_counter: int = 0
        self._hypothesis_cache: Dict[str, HypothesisPerformance] = {}
        self._cache_dirty: bool = True

    # ── Trade Recording ───────────────────────────────────────────────────────

    def record_trade(
        self,
        hypothesis_id: str,
        entry_price: float,
        exit_price: float,
        size: float,
        direction: int,
        regime: str,
        template_type: Optional[str] = None,
        hold_bars: int = 1,
        entry_time: Optional[float] = None,
        exit_time: Optional[float] = None,
    ) -> str:
        """
        Record a completed trade.

        Parameters
        ----------
        hypothesis_id : unique ID for the hypothesis template instance
        entry_price, exit_price : trade prices
        size : notional position size
        direction : +1 = long, -1 = short
        regime : market regime label at entry time
        template_type : optional template family (e.g. "entropy_minimization")
        hold_bars : number of periods held
        entry_time, exit_time : unix timestamps (defaults to now)

        Returns
        -------
        trade_id : unique identifier for this trade
        """
        now = time.time()
        self._trade_counter += 1
        trade_id = f"trade_{self._trade_counter:06d}"

        if template_type is None:
            # Infer template type from hypothesis_id prefix
            template_type = hypothesis_id.split("_")[0] if "_" in hypothesis_id else hypothesis_id

        trade = TradeRecord(
            trade_id=trade_id,
            hypothesis_id=hypothesis_id,
            template_type=template_type,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            size=float(size),
            direction=int(direction),
            regime=str(regime),
            entry_time=entry_time or now,
            exit_time=exit_time or now,
            hold_bars=max(hold_bars, 1),
        )
        self._trades.append(trade)
        self._cache_dirty = True
        return trade_id

    # ── Metrics Computation ───────────────────────────────────────────────────

    def _rebuild_cache(self) -> None:
        """Rebuild hypothesis performance cache."""
        by_hypothesis: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            by_hypothesis[t.hypothesis_id].append(t)

        self._hypothesis_cache = {
            hid: _compute_hypothesis_performance(
                hid,
                trades[0].template_type if trades else "unknown",
                trades,
            )
            for hid, trades in by_hypothesis.items()
        }
        self._cache_dirty = False

    def _get_cache(self) -> Dict[str, HypothesisPerformance]:
        if self._cache_dirty:
            self._rebuild_cache()
        return self._hypothesis_cache

    def rolling_metrics(self, window: int = 30) -> RollingMetrics:
        """
        Compute performance metrics over the most recent `window` trades.
        """
        recent = self._trades[-window:]
        if not recent:
            return RollingMetrics(
                window=window, n_trades=0, pnl=0.0, win_rate=0.0,
                sharpe=0.0, sortino=0.0, avg_return=0.0,
                vol_of_returns=0.0, max_drawdown=0.0,
            )

        pnls = np.array([t.pnl for t in recent])
        rets = np.array([t.return_pct / 100.0 for t in recent])
        win_rate = float(sum(1 for t in recent if t.is_winner) / len(recent))
        cum_pnl = np.cumsum(pnls)
        max_dd, _ = _max_drawdown(cum_pnl)

        return RollingMetrics(
            window=window,
            n_trades=len(recent),
            pnl=float(pnls.sum()),
            win_rate=win_rate,
            sharpe=_sharpe(rets, self.risk_free_rate, self.periods_per_year),
            sortino=_sortino(rets, periods_per_year=self.periods_per_year),
            avg_return=float(rets.mean()),
            vol_of_returns=float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
            max_drawdown=max_dd,
        )

    def by_template_type(self) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate performance stats grouped by template type.
        Returns dict: template_type → {pnl, sharpe, win_rate, n_trades, ...}
        """
        by_type: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            by_type[t.template_type].append(t)

        results = {}
        for ttype, trades in by_type.items():
            pnls = np.array([t.pnl for t in trades])
            rets = np.array([t.return_pct / 100.0 for t in trades])
            win_rate = sum(1 for t in trades if t.is_winner) / len(trades)
            cum_pnl = np.cumsum(pnls)
            max_dd, _ = _max_drawdown(cum_pnl)
            results[ttype] = {
                "n_trades": len(trades),
                "total_pnl": float(pnls.sum()),
                "win_rate": float(win_rate),
                "sharpe": _sharpe(rets, self.risk_free_rate, self.periods_per_year),
                "sortino": _sortino(rets, periods_per_year=self.periods_per_year),
                "avg_pnl": float(pnls.mean()),
                "max_drawdown": max_dd,
                "profit_factor": (
                    float(pnls[pnls > 0].sum() / max(abs(pnls[pnls < 0].sum()), 1e-10))
                    if len(pnls) > 0 else 0.0
                ),
            }
        return results

    def by_regime(self) -> Dict[str, Dict[str, Any]]:
        """
        Performance breakdown by market regime.
        Returns dict: regime → {pnl, sharpe, win_rate, n_trades, ...}
        """
        by_regime: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            by_regime[t.regime].append(t)

        results = {}
        for regime, trades in by_regime.items():
            pnls = np.array([t.pnl for t in trades])
            rets = np.array([t.return_pct / 100.0 for t in trades])
            win_rate = sum(1 for t in trades if t.is_winner) / len(trades)
            results[regime] = {
                "n_trades": len(trades),
                "total_pnl": float(pnls.sum()),
                "win_rate": float(win_rate),
                "sharpe": _sharpe(rets, self.risk_free_rate, self.periods_per_year),
                "avg_pnl": float(pnls.mean()),
                "pnl_per_trade": float(pnls.mean()),
            }
        return results

    def best_performers(self, n: int = 10) -> List[HypothesisPerformance]:
        """Return top N hypothesis templates by Sharpe ratio."""
        cache = self._get_cache()
        sorted_hyps = sorted(cache.values(), key=lambda h: h.sharpe, reverse=True)
        return sorted_hyps[:n]

    def worst_performers(self, n: int = 10) -> List[HypothesisPerformance]:
        """Return bottom N hypothesis templates by Sharpe ratio (candidates to prune)."""
        cache = self._get_cache()
        sorted_hyps = sorted(cache.values(), key=lambda h: h.sharpe)
        return sorted_hyps[:n]

    def regime_conditional_sharpe(self) -> Dict[str, Dict[str, float]]:
        """
        Compute Sharpe ratio for each (template_type, regime) combination.
        Returns dict: template_type → {regime → sharpe}
        """
        by_template_regime: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for t in self._trades:
            by_template_regime[t.template_type][t.regime].append(t.return_pct / 100.0)

        results: Dict[str, Dict[str, float]] = {}
        for ttype, regimes in by_template_regime.items():
            results[ttype] = {}
            for regime, rets in regimes.items():
                r = np.array(rets)
                results[ttype][regime] = _sharpe(r, self.risk_free_rate, self.periods_per_year)

        return results

    def attribution_report(self) -> AttributionReport:
        """
        Full attribution breakdown across all dimensions.
        """
        cache = self._get_cache()
        all_pnls = np.array([t.pnl for t in self._trades])
        all_rets = np.array([t.return_pct / 100.0 for t in self._trades])

        total_pnl = float(all_pnls.sum()) if len(all_pnls) > 0 else 0.0
        overall_sharpe = _sharpe(all_rets, self.risk_free_rate, self.periods_per_year)
        overall_wr = float(sum(1 for t in self._trades if t.is_winner) / max(len(self._trades), 1))
        cum_pnl = np.cumsum(all_pnls) if len(all_pnls) > 0 else np.array([0.0])
        overall_mdd, _ = _max_drawdown(cum_pnl)

        by_template = self.by_template_type()
        by_regime = self.by_regime()
        rc_sharpe = self.regime_conditional_sharpe()

        # Top contributors and detractors by hypothesis
        hyp_pnls = [(hid, hp.total_pnl) for hid, hp in cache.items()]
        hyp_pnls_sorted = sorted(hyp_pnls, key=lambda x: x[1], reverse=True)
        top_contributors = hyp_pnls_sorted[:10]
        top_detractors = hyp_pnls_sorted[-10:][::-1]

        return AttributionReport(
            total_pnl=total_pnl,
            total_trades=len(self._trades),
            overall_sharpe=overall_sharpe,
            overall_win_rate=overall_wr,
            overall_max_drawdown=overall_mdd,
            by_template=by_template,
            by_regime=by_regime,
            by_template_regime=rc_sharpe,
            top_contributors=top_contributors,
            top_detractors=top_detractors,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    def cumulative_pnl_series(self) -> np.ndarray:
        """Return cumulative PnL series over all recorded trades."""
        if not self._trades:
            return np.array([0.0])
        return np.cumsum([t.pnl for t in self._trades])

    def trade_return_series(self) -> np.ndarray:
        """Return array of per-trade returns (as decimals)."""
        return np.array([t.return_pct / 100.0 for t in self._trades])

    def equity_curve(self, initial_equity: float = 10000.0) -> np.ndarray:
        """Return equity curve starting from initial_equity."""
        pnls = np.array([t.pnl for t in self._trades])
        return initial_equity + np.concatenate([[0.0], np.cumsum(pnls)])

    # ── Summary Tables ────────────────────────────────────────────────────────

    def summary_table(self) -> List[Dict[str, Any]]:
        """Return list of dicts suitable for tabular display (one row per hypothesis)."""
        cache = self._get_cache()
        rows = []
        for hid, hp in cache.items():
            rows.append({
                "hypothesis_id": hp.hypothesis_id,
                "template_type": hp.template_type,
                "n_trades": hp.n_trades,
                "total_pnl": round(hp.total_pnl, 4),
                "win_rate": round(hp.win_rate * 100, 1),
                "sharpe": round(hp.sharpe, 3),
                "sortino": round(hp.sortino, 3),
                "max_drawdown": round(hp.max_drawdown * 100, 2),
                "profit_factor": round(hp.profit_factor, 3),
                "best_regime": hp.best_regime,
                "worst_regime": hp.worst_regime,
                "avg_hold_bars": round(hp.avg_hold_bars, 1),
            })
        return sorted(rows, key=lambda r: r["sharpe"], reverse=True)

    # ── Pruning Recommendations ───────────────────────────────────────────────

    def prune_candidates(
        self,
        min_trades: int = 10,
        min_sharpe: float = -0.5,
        max_drawdown: float = 0.3,
    ) -> List[str]:
        """
        Return list of hypothesis IDs that should be considered for pruning.
        Criteria: min_trades met AND (Sharpe below threshold OR max drawdown exceeded).
        """
        cache = self._get_cache()
        candidates = []
        for hid, hp in cache.items():
            if hp.n_trades >= min_trades:
                if hp.sharpe < min_sharpe or hp.max_drawdown > max_drawdown:
                    candidates.append(hid)
        return sorted(candidates, key=lambda h: cache[h].sharpe)

    def promote_candidates(
        self,
        min_trades: int = 20,
        min_sharpe: float = 1.0,
        min_win_rate: float = 0.55,
    ) -> List[str]:
        """
        Return list of hypothesis IDs that qualify for promotion to live trading.
        Criteria: meets all thresholds with sufficient sample size.
        """
        cache = self._get_cache()
        candidates = []
        for hid, hp in cache.items():
            if (hp.n_trades >= min_trades and
                    hp.sharpe >= min_sharpe and
                    hp.win_rate >= min_win_rate):
                candidates.append(hid)
        return sorted(candidates, key=lambda h: cache[h].sharpe, reverse=True)

    # ── Serialization ─────────────────────────────────────────────────────────

    def export_json(self, include_trades: bool = False) -> str:
        """
        Export complete performance record as JSON.
        Suitable for logging, API responses, or persistence.
        """
        cache = self._get_cache()
        doc = {
            "metadata": {
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "total_trades": len(self._trades),
                "n_hypotheses": len(cache),
                "risk_free_rate": self.risk_free_rate,
                "periods_per_year": self.periods_per_year,
            },
            "overall": {
                "total_pnl": float(sum(t.pnl for t in self._trades)),
                "overall_sharpe": _sharpe(self.trade_return_series()),
                "overall_win_rate": float(sum(1 for t in self._trades if t.is_winner) / max(len(self._trades), 1)),
            },
            "by_template": self.by_template_type(),
            "by_regime": self.by_regime(),
            "regime_conditional_sharpe": self.regime_conditional_sharpe(),
            "hypothesis_performance": {
                hid: {
                    "hypothesis_id": hp.hypothesis_id,
                    "template_type": hp.template_type,
                    "n_trades": hp.n_trades,
                    "total_pnl": hp.total_pnl,
                    "win_rate": hp.win_rate,
                    "sharpe": hp.sharpe,
                    "sortino": hp.sortino,
                    "calmar": hp.calmar,
                    "max_drawdown": hp.max_drawdown,
                    "profit_factor": hp.profit_factor,
                    "regime_pnl": hp.regime_pnl,
                    "best_regime": hp.best_regime,
                    "worst_regime": hp.worst_regime,
                    "avg_hold_bars": hp.avg_hold_bars,
                }
                for hid, hp in cache.items()
            },
        }
        if include_trades:
            doc["trades"] = [
                {
                    "trade_id": t.trade_id,
                    "hypothesis_id": t.hypothesis_id,
                    "template_type": t.template_type,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "direction": t.direction,
                    "regime": t.regime,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "hold_bars": t.hold_bars,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                }
                for t in self._trades
            ]
        return json.dumps(doc, indent=2, default=str)

    def import_json(self, json_str: str) -> int:
        """
        Import trades from a JSON export. Returns number of trades imported.
        Only imports trades not already present (by trade_id).
        """
        doc = json.loads(json_str)
        trades_data = doc.get("trades", [])
        existing_ids = {t.trade_id for t in self._trades}
        n_imported = 0
        for td in trades_data:
            if td["trade_id"] not in existing_ids:
                trade = TradeRecord(
                    trade_id=td["trade_id"],
                    hypothesis_id=td["hypothesis_id"],
                    template_type=td.get("template_type", "unknown"),
                    entry_price=td["entry_price"],
                    exit_price=td["exit_price"],
                    size=td["size"],
                    direction=td["direction"],
                    regime=td["regime"],
                    entry_time=td.get("entry_time", 0.0),
                    exit_time=td.get("exit_time", 0.0),
                    hold_bars=td.get("hold_bars", 1),
                )
                self._trades.append(trade)
                n_imported += 1
        if n_imported > 0:
            self._cache_dirty = True
        return n_imported

    def reset(self) -> None:
        """Clear all trades and reset the tracker."""
        self._trades.clear()
        self._hypothesis_cache.clear()
        self._cache_dirty = True
        self._trade_counter = 0

    def __len__(self) -> int:
        return len(self._trades)

    def __repr__(self) -> str:
        return (
            f"PerformanceTracker("
            f"trades={len(self._trades)}, "
            f"hypotheses={len(self._get_cache())})"
        )
