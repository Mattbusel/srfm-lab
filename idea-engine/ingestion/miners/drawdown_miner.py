"""
idea-engine/ingestion/miners/drawdown_miner.py
───────────────────────────────────────────────
Identifies significant drawdown events, clusters them, and finds their
common preceding conditions (regime state, BH mass, instruments held).

Method
──────
  1. Compute the equity drawdown series from equity_series.
  2. Identify "significant" drawdown events: troughs where DD < threshold
     (default: -5%).
  3. For each DD event: look up which trades were open/recently closed,
     the regime state, and BH mass around that time.
  4. Cluster nearby DD events (within DD_CLUSTER_DISTANCE_DAYS days) to
     avoid double-counting.
  5. Find "drawdown predictors": conditions that were consistently present
     N bars before major DDs but not present during good periods.
     Uses a simple logistic-regression / chi-squared test.
  6. Return MinedPattern objects for each significant predictor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from ..config import (
    DD_CLUSTER_DISTANCE_DAYS,
    DD_SIGNIFICANT_THRESHOLD,
    MIN_EFFECT_SIZE,
    MIN_GROUP_SAMPLE,
    RAW_P_VALUE_THRESHOLD,
)
from ..types import (
    EffectSizeType,
    LiveTradeData,
    MinedPattern,
    PatternStatus,
    PatternType,
    safe_float,
)

logger = logging.getLogger(__name__)


# ── drawdown event dataclass ──────────────────────────────────────────────────

@dataclass
class DrawdownEvent:
    trough_ts:     pd.Timestamp
    trough_equity: float
    peak_equity:   float
    drawdown_pct:  float           # negative, e.g. -0.08 = 8 % DD
    duration_bars: int
    cluster_id:    int = -1
    regime_at_trough: Dict         = field(default_factory=dict)
    trades_during:    List[Dict]   = field(default_factory=list)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _bh_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    n = len(p_values)
    if n == 0:
        return []
    order  = sorted(range(n), key=lambda i: p_values[i])
    reject = [False] * n
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / n:
            reject[idx] = True
    return reject


# ── drawdown identification ───────────────────────────────────────────────────

def _compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Return fractional drawdown series (0 at peak, negative in troughs)."""
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return dd


def _find_trough_timestamps(
    equity: pd.Series,
    dd_series: pd.Series,
    threshold: float = DD_SIGNIFICANT_THRESHOLD,
) -> List[pd.Timestamp]:
    """
    Find local minima in drawdown below the threshold.

    Uses a simple approach: find all points where DD < threshold,
    then pick the worst point in each contiguous segment.
    """
    below = dd_series < -abs(threshold)
    if not below.any():
        return []

    # Find contiguous segments
    transitions = below.astype(int).diff().fillna(0)
    starts = dd_series.index[transitions == 1].tolist()
    ends   = dd_series.index[transitions == -1].tolist()

    # Handle case where we're below at the very beginning
    if below.iloc[0]:
        starts = [dd_series.index[0]] + starts

    # Handle case where we end below the threshold
    if below.iloc[-1]:
        ends.append(dd_series.index[-1])

    troughs: List[pd.Timestamp] = []
    for start, end in zip(starts, ends):
        segment = dd_series[start:end]
        trough_ts = segment.idxmin()
        troughs.append(trough_ts)

    return troughs


def _cluster_drawdowns(
    events: List[DrawdownEvent],
    cluster_days: int = DD_CLUSTER_DISTANCE_DAYS,
) -> List[DrawdownEvent]:
    """
    Merge DD events within cluster_days of each other into a single cluster.

    Sets cluster_id on each event. Returns events with cluster_id set.
    """
    if not events:
        return events

    events = sorted(events, key=lambda e: e.trough_ts)
    cluster_id = 0
    events[0].cluster_id = cluster_id
    prev_ts = events[0].trough_ts

    for ev in events[1:]:
        delta = (ev.trough_ts - prev_ts).total_seconds() / 86400.0
        if delta > cluster_days:
            cluster_id += 1
        ev.cluster_id = cluster_id
        prev_ts = ev.trough_ts

    return events


# ── regime / trade context extraction ────────────────────────────────────────

def _get_regime_at(
    ts: pd.Timestamp,
    regime_log: pd.DataFrame,
    window_bars: int = 3,
) -> Dict:
    """Return median regime features in the window leading up to ts."""
    if regime_log is None or regime_log.empty:
        return {}
    rl = regime_log.copy()
    if "_ts" not in rl.columns:
        if "ts" in rl.columns:
            rl["_ts"] = pd.to_datetime(rl["ts"], errors="coerce")
        else:
            return {}
    window = rl[rl["_ts"] <= ts].tail(window_bars)
    if window.empty:
        return {}
    numeric_cols = window.select_dtypes(include=[np.number]).columns.tolist()
    return {c: float(window[c].median()) for c in numeric_cols if c not in ("id",)}


def _get_trades_during(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    trades: pd.DataFrame,
) -> List[Dict]:
    """Return trades that were closed during [start_ts, end_ts]."""
    if trades is None or trades.empty:
        return []
    ts_col = "ts" if "ts" in trades.columns else "exit_time"
    if ts_col not in trades.columns:
        return []
    df = trades.copy()
    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    mask = (df["_ts"] >= start_ts) & (df["_ts"] <= end_ts)
    subset = df[mask]
    if subset.empty:
        return []
    return subset[["_ts", "symbol", "pnl"]].dropna(subset=["pnl"]).to_dict("records")


# ── drawdown predictor mining ─────────────────────────────────────────────────

def _mine_bh_mass_predictor(
    events: List[DrawdownEvent],
    regime_log: pd.DataFrame,
    equity: pd.Series,
    alpha: float = RAW_P_VALUE_THRESHOLD,
    min_effect: float = MIN_EFFECT_SIZE,
    source: str = "live",
) -> List[MinedPattern]:
    """
    Test whether BH mass level before a drawdown is a predictor.

    For each regime feature, compare its value in the pre-DD window
    vs during non-DD periods.
    """
    patterns: List[MinedPattern] = []
    if regime_log is None or regime_log.empty:
        return patterns

    feature_cols = [
        c for c in ["d_bh_mass", "h_bh_mass", "m15_bh_mass", "tf_score",
                    "garch_vol", "ou_zscore", "atr"]
        if c in regime_log.columns
    ]
    if not feature_cols:
        return patterns

    rl = regime_log.copy()
    if "_ts" not in rl.columns:
        rl["_ts"] = pd.to_datetime(rl["ts"], errors="coerce")

    # Collect pre-DD windows and non-DD windows
    dd_ts_set = {ev.trough_ts for ev in events}
    lookback  = pd.Timedelta(hours=48)

    pre_dd_mask   = pd.Series(False, index=rl.index)
    for ev in events:
        window_mask = (rl["_ts"] >= ev.trough_ts - lookback) & (rl["_ts"] < ev.trough_ts)
        pre_dd_mask = pre_dd_mask | window_mask

    pre_dd_rows    = rl[pre_dd_mask]
    non_dd_rows    = rl[~pre_dd_mask]

    if len(pre_dd_rows) < MIN_GROUP_SAMPLE or len(non_dd_rows) < MIN_GROUP_SAMPLE:
        return patterns

    p_values:   List[float] = []
    candidates: List[Tuple] = []

    for col in feature_cols:
        a = pre_dd_rows[col].dropna().values.astype(float)
        b = non_dd_rows[col].dropna().values.astype(float)
        if len(a) < MIN_GROUP_SAMPLE or len(b) < MIN_GROUP_SAMPLE:
            continue
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = 1.0
        p_values.append(float(p))
        candidates.append((col, a, b, float(p)))

    if not candidates:
        return patterns

    rejected = _bh_correction([c[3] for c in candidates], alpha=alpha)

    for (col, a, b, p), rej in zip(candidates, rejected):
        d = _cohens_d(a, b)
        if not rej and abs(d) < min_effect:
            continue

        direction = "higher" if a.mean() > b.mean() else "lower"
        patterns.append(MinedPattern(
            source           = source,
            miner            = "DrawdownMiner",
            pattern_type     = PatternType.DRAWDOWN,
            label            = f"Pre-DD predictor: {col} is {direction}",
            description      = (
                f"'{col}' is significantly {direction} in the 48h before major drawdowns "
                f"(threshold={DD_SIGNIFICANT_THRESHOLD*100:.0f}%): "
                f"pre-DD median={np.median(a):.3f} vs normal={np.median(b):.3f}; "
                f"p={p:.4f}, Cohen's d={d:.3f}"
            ),
            feature_dict     = {
                "predictor_col":       col,
                "pre_dd_median":       float(np.median(a)),
                "normal_median":       float(np.median(b)),
                "direction":           direction,
                "n_dd_events":         len(events),
                "dd_threshold":        DD_SIGNIFICANT_THRESHOLD,
            },
            sample_size      = int(len(a)),
            p_value          = p,
            effect_size      = abs(d),
            effect_size_type = EffectSizeType.COHENS_D,
            status           = PatternStatus.NEW,
            tags             = ["drawdown", "predictor", col],
            raw_group        = pd.Series(a),
            raw_baseline     = pd.Series(b),
        ))

    return patterns


def _mine_instrument_predictor(
    events: List[DrawdownEvent],
    trades: pd.DataFrame,
    equity: pd.Series,
    alpha: float = RAW_P_VALUE_THRESHOLD,
    source: str = "live",
) -> List[MinedPattern]:
    """
    Test whether certain instruments are disproportionately present during DDs.
    """
    patterns: List[MinedPattern] = []
    if trades is None or trades.empty:
        return patterns

    sym_col = "symbol" if "symbol" in trades.columns else "sym"
    if sym_col not in trades.columns or "pnl" not in trades.columns:
        return patterns

    ts_col = "ts" if "ts" in trades.columns else "exit_time"
    if ts_col not in trades.columns:
        return patterns

    trades = trades.copy()
    trades["_ts"] = pd.to_datetime(trades[ts_col], errors="coerce")
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")

    # Flag trades during DD periods
    dd_flags = pd.Series(False, index=trades.index)
    for ev in events:
        start = ev.trough_ts - pd.Timedelta(hours=48)
        end   = ev.trough_ts
        mask  = (trades["_ts"] >= start) & (trades["_ts"] <= end)
        dd_flags = dd_flags | mask

    trades["during_dd"] = dd_flags.astype(int)
    all_syms = trades[sym_col].dropna().unique()

    p_vals:     List[float] = []
    candidates: List[Tuple] = []

    for sym in all_syms:
        sym_mask  = trades[sym_col] == sym
        contingency = pd.crosstab(sym_mask.astype(int), trades["during_dd"])
        if contingency.shape != (2, 2):
            continue
        try:
            chi2, p, dof, expected = chi2_contingency(contingency, correction=True)
        except Exception:
            continue
        if np.any(expected < 5):
            continue
        p_vals.append(float(p))
        candidates.append((sym, float(p), float(chi2), sym_mask, trades["during_dd"]))

    if not candidates:
        return patterns

    rejected = _bh_correction([c[1] for c in candidates], alpha=alpha)

    for (sym, p, chi2, sym_mask, during_dd), rej in zip(candidates, rejected):
        if not rej:
            continue
        dd_sym_count    = int((sym_mask & (during_dd == 1)).sum())
        total_dd_trades = int((during_dd == 1).sum())
        total_sym       = int(sym_mask.sum())
        total_trades    = len(trades)

        patterns.append(MinedPattern(
            source           = source,
            miner            = "DrawdownMiner",
            pattern_type     = PatternType.DRAWDOWN,
            label            = f"Instrument {sym} disproportionately in drawdowns",
            description      = (
                f"{sym}: {dd_sym_count}/{total_dd_trades} ({100*dd_sym_count/max(total_dd_trades,1):.1f}%) "
                f"DD trades vs {total_sym}/{total_trades} ({100*total_sym/max(total_trades,1):.1f}%) overall; "
                f"χ²={chi2:.2f}, p={p:.4f}"
            ),
            feature_dict     = {
                "instrument":           sym,
                "dd_trade_count":       dd_sym_count,
                "total_dd_trades":      total_dd_trades,
                "total_sym_trades":     total_sym,
                "chi2":                 chi2,
            },
            sample_size      = int(total_sym),
            p_value          = p,
            effect_size      = float(chi2 / max(total_trades, 1)),
            effect_size_type = EffectSizeType.ETA_SQUARED,
            status           = PatternStatus.NEW,
            tags             = ["drawdown", "instrument", sym],
        ))

    return patterns


# ── public API ────────────────────────────────────────────────────────────────

class DrawdownMiner:
    """
    Identifies significant drawdown events and their predictors.
    """

    def __init__(
        self,
        source:         str   = "live",
        dd_threshold:   float = DD_SIGNIFICANT_THRESHOLD,
        cluster_days:   int   = DD_CLUSTER_DISTANCE_DAYS,
        alpha:          float = RAW_P_VALUE_THRESHOLD,
        min_effect:     float = MIN_EFFECT_SIZE,
        min_sample:     int   = MIN_GROUP_SAMPLE,
    ):
        self.source       = source
        self.dd_threshold = dd_threshold
        self.cluster_days = cluster_days
        self.alpha        = alpha
        self.min_effect   = min_effect
        self.min_sample   = min_sample

    def _build_events(
        self,
        equity: pd.Series,
        trades: Optional[pd.DataFrame],
        regime_log: Optional[pd.DataFrame],
    ) -> List[DrawdownEvent]:
        """Identify and enrich drawdown events."""
        dd_series = _compute_drawdown_series(equity)
        trough_tss = _find_trough_timestamps(equity, dd_series, threshold=self.dd_threshold)

        if not trough_tss:
            logger.info("DrawdownMiner: no drawdowns found below %.1f%%", self.dd_threshold * 100)
            return []

        events: List[DrawdownEvent] = []
        for ts in trough_tss:
            # Find peak before this trough
            pre_dd = equity[:ts]
            if pre_dd.empty:
                continue
            peak_ts    = pre_dd.idxmax()
            peak_eq    = float(equity[peak_ts])
            trough_eq  = float(equity[ts])
            dd_pct     = (trough_eq - peak_eq) / peak_eq  # negative

            # Duration in bars
            peak_idx   = equity.index.get_loc(peak_ts)
            trough_idx = equity.index.get_loc(ts)
            duration   = max(int(trough_idx) - int(peak_idx), 1)

            ev = DrawdownEvent(
                trough_ts      = ts,
                trough_equity  = trough_eq,
                peak_equity    = peak_eq,
                drawdown_pct   = dd_pct,
                duration_bars  = duration,
            )

            # Enrich with regime state
            if regime_log is not None and not regime_log.empty:
                ev.regime_at_trough = _get_regime_at(ts, regime_log)

            # Enrich with trades during the drawdown
            if trades is not None and not trades.empty:
                ev.trades_during = _get_trades_during(
                    pd.Timestamp(peak_ts), ts, trades
                )

            events.append(ev)

        logger.info("DrawdownMiner: found %d drawdown events", len(events))
        events = _cluster_drawdowns(events, self.cluster_days)
        return events

    def mine(self, live_data: LiveTradeData) -> List[MinedPattern]:
        """
        Run drawdown mining on LiveTradeData.

        Returns
        -------
        List[MinedPattern]
        """
        if live_data.equity_series is None or len(live_data.equity_series) < 10:
            logger.warning("DrawdownMiner: equity_series unavailable or too short")
            return []

        equity = live_data.equity_series
        events = self._build_events(equity, live_data.trades, live_data.regime_log)

        if not events:
            return []

        patterns: List[MinedPattern] = []

        # BH mass predictor patterns
        if live_data.regime_log is not None:
            patterns.extend(
                _mine_bh_mass_predictor(
                    events, live_data.regime_log, equity,
                    alpha=self.alpha, min_effect=self.min_effect, source=self.source,
                )
            )

        # Instrument predictor patterns
        if live_data.trades is not None:
            patterns.extend(
                _mine_instrument_predictor(
                    events, live_data.trades, equity,
                    alpha=self.alpha, source=self.source,
                )
            )

        # Add a summary pattern for each cluster
        n_clusters = max((ev.cluster_id for ev in events), default=0) + 1
        for cid in range(n_clusters):
            cluster_evs = [ev for ev in events if ev.cluster_id == cid]
            worst       = min(cluster_evs, key=lambda e: e.drawdown_pct)
            instruments = list({
                t.get("symbol", "") for ev in cluster_evs for t in ev.trades_during
                if t.get("symbol")
            })
            regime_desc = {
                k: round(v, 4) for k, v in worst.regime_at_trough.items()
                if k in ["d_bh_mass", "h_bh_mass", "m15_bh_mass", "tf_score"]
            }
            patterns.append(MinedPattern(
                source           = self.source,
                miner            = "DrawdownMiner",
                pattern_type     = PatternType.DRAWDOWN,
                label            = f"Drawdown cluster {cid}: {worst.drawdown_pct*100:.1f}% worst event",
                description      = (
                    f"Cluster {cid} with {len(cluster_evs)} event(s); "
                    f"worst trough at {worst.trough_ts.date()} "
                    f"({worst.drawdown_pct*100:.1f}%); "
                    f"regime at trough: {regime_desc}; "
                    f"instruments: {instruments}"
                ),
                feature_dict     = {
                    "cluster_id":    cid,
                    "n_events":      len(cluster_evs),
                    "worst_dd_pct":  worst.drawdown_pct,
                    "duration_bars": worst.duration_bars,
                    "regime":        regime_desc,
                    "instruments":   instruments,
                },
                sample_size      = len(cluster_evs),
                instruments      = instruments,
                status           = PatternStatus.NEW,
                tags             = ["drawdown", "cluster", f"dd_cluster_{cid}"],
            ))

        logger.info("DrawdownMiner produced %d pattern(s)", len(patterns))
        return patterns


def mine_drawdowns(live_data: LiveTradeData, source: str = "live", **kwargs) -> List[MinedPattern]:
    """Shortcut function."""
    return DrawdownMiner(source=source, **kwargs).mine(live_data)
