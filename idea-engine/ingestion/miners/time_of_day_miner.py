"""
idea-engine/ingestion/miners/time_of_day_miner.py
──────────────────────────────────────────────────
Mines statistically anomalous time-of-day and day-of-week windows from
trade data using non-parametric tests.

Method
──────
  1. Bin trades by hour-of-day (0-23) and day-of-week (0=Mon … 6=Sun).
  2. Kruskal-Wallis H-test across bins for both win_rate and avg PnL.
  3. If KW is significant, run pairwise Dunn post-hoc test (with BH correction).
  4. Report each significantly anomalous bin as a MinedPattern.

Dependencies
────────────
  scipy ≥ 1.7  (kruskal, mannwhitneyu)
  scikit_posthocs (optional, falls back to pairwise Mann-Whitney if absent)
"""

from __future__ import annotations

import importlib
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from ..config import (
    MIN_GROUP_SAMPLE,
    TOD_DUNN_ALPHA,
    TOD_KRUSKAL_ALPHA,
    TOD_MIN_TRADES_PER_BIN,
)
from ..types import EffectSizeType, MinedPattern, PatternStatus, PatternType

logger = logging.getLogger(__name__)

_HAS_SCIKIT_POSTHOCS = importlib.util.find_spec("scikit_posthocs") is not None


# ── statistics helpers ────────────────────────────────────────────────────────

def _cohens_d(group: np.ndarray, baseline: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group), len(baseline)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2   = group.mean(), baseline.mean()
    s1, s2   = group.std(ddof=1), baseline.std(ddof=1)
    pooled_s = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_s == 0:
        return 0.0
    return float((m1 - m2) / pooled_s)


def _bh_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Benjamini-Hochberg FDR correction.  Returns list of rejection booleans."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject  = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        if p <= alpha * rank / n:
            reject[orig_idx] = True
    # BH is monotone: once we stop rejecting, later ones are also not rejected
    # (already handled by the sorted structure above)
    return reject


def _dunn_posthoc_mannwhitney(
    groups: dict[str, np.ndarray],
    alpha: float = TOD_DUNN_ALPHA,
) -> dict[tuple[str, str], float]:
    """
    Pairwise Mann-Whitney U test with BH correction (Dunn-like fallback).

    Returns dict of (label_a, label_b) → adjusted_p_value for rejected pairs.
    """
    from scipy.stats import mannwhitneyu

    keys  = list(groups.keys())
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    p_vals = []
    for a, b in pairs:
        if len(groups[a]) < 3 or len(groups[b]) < 3:
            p_vals.append(1.0)
            continue
        _, p = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        p_vals.append(float(p))
    rejected = _bh_correction(p_vals, alpha=alpha)
    return {pair: p for pair, p, rej in zip(pairs, p_vals, rejected) if rej}


def _dunn_scikit(
    data: pd.Series,
    groups: pd.Series,
    alpha: float = TOD_DUNN_ALPHA,
) -> dict[tuple, float]:
    """Dunn test via scikit_posthocs."""
    import scikit_posthocs as sp  # type: ignore

    df = pd.DataFrame({"val": data, "grp": groups})
    try:
        result = sp.posthoc_dunn(df, val_col="val", group_col="grp", p_adjust="fdr_bh")
    except Exception as exc:
        logger.warning("scikit_posthocs Dunn test failed: %s", exc)
        return {}
    significant = {}
    for col in result.columns:
        for idx in result.index:
            if idx < col:
                p = result.loc[idx, col]
                if p < alpha:
                    significant[(idx, col)] = float(p)
    return significant


# ── main miner ────────────────────────────────────────────────────────────────

class TimeOfDayMiner:
    """
    Mines time-of-day and day-of-week patterns from a trades DataFrame.

    Expected columns in trades_df:
        ts / exit_time  — timestamp
        pnl             — trade profit/loss in dollars
        symbol (opt.)   — symbol filter
    """

    def __init__(
        self,
        min_trades_per_bin: int = TOD_MIN_TRADES_PER_BIN,
        kruskal_alpha:       float = TOD_KRUSKAL_ALPHA,
        dunn_alpha:          float = TOD_DUNN_ALPHA,
        source:              str = "live",
    ):
        self.min_trades_per_bin = min_trades_per_bin
        self.kruskal_alpha      = kruskal_alpha
        self.dunn_alpha         = dunn_alpha
        self.source             = source

    # ── internal ─────────────────────────────────────────────────────────────

    def _prep(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        df = trades_df.copy()
        ts_col = "ts" if "ts" in df.columns else "exit_time" if "exit_time" in df.columns else None
        if ts_col is None:
            raise ValueError("trades_df must have a 'ts' or 'exit_time' column")
        df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
        df         = df.dropna(subset=["_ts", "pnl"])
        df["_pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
        df          = df.dropna(subset=["_pnl"])
        df["_hour"] = df["_ts"].dt.hour.astype(int)
        df["_dow"]  = df["_ts"].dt.dayofweek.astype(int)
        df["_win"]  = (df["_pnl"] > 0).astype(float)
        return df

    def _kw_and_dunn(
        self,
        df: pd.DataFrame,
        bin_col: str,
        value_col: str,
        dimension_label: str,
    ) -> List[MinedPattern]:
        """Run KW + Dunn for one dimension (hour or dow) and one metric (pnl/win)."""
        patterns: List[MinedPattern] = []

        bin_labels = sorted(df[bin_col].unique())
        groups_dict: dict[int, np.ndarray] = {}
        for b in bin_labels:
            g = df[df[bin_col] == b][value_col].values
            if len(g) >= self.min_trades_per_bin:
                groups_dict[b] = g

        if len(groups_dict) < 2:
            return patterns

        all_groups = list(groups_dict.values())
        try:
            h_stat, p_kw = kruskal(*all_groups)
        except Exception as exc:
            logger.debug("KW test failed for %s/%s: %s", dimension_label, value_col, exc)
            return patterns

        if p_kw >= self.kruskal_alpha:
            logger.debug(
                "KW not significant for %s/%s (p=%.4f)", dimension_label, value_col, p_kw
            )
            return patterns

        logger.info(
            "KW significant for %s/%s: H=%.3f, p=%.4f — running post-hoc",
            dimension_label, value_col, h_stat, p_kw,
        )

        # Baseline = all trades
        baseline = df[value_col].values
        baseline_mean = baseline.mean()

        # Dunn post-hoc
        str_groups = {str(k): v for k, v in groups_dict.items()}
        if _HAS_SCIKIT_POSTHOCS:
            pnl_series    = pd.Series(
                [v for vals in groups_dict.values() for v in vals]
            )
            group_series  = pd.Series(
                [k for k, vals in groups_dict.items() for _ in vals]
            )
            sig_pairs = _dunn_scikit(pnl_series, group_series, self.dunn_alpha)
        else:
            sig_pairs = _dunn_mannwhitney = _dunn_posthoc_mannwhitney(str_groups, self.dunn_alpha)

        # Build pattern for each significantly anomalous bin
        reported_bins: set = set()
        for pair in sig_pairs:
            for b_str in [str(p) for p in pair]:
                if b_str in reported_bins:
                    continue
                try:
                    b_int = int(float(b_str))
                except ValueError:
                    continue
                if b_int not in groups_dict:
                    continue
                reported_bins.add(b_str)

                grp     = groups_dict[b_int]
                d       = _cohens_d(grp, baseline)
                win_r   = float((grp > 0).sum() / len(grp)) if value_col == "_pnl" else None
                avg_pnl = float(grp.mean()) if value_col == "_pnl" else None

                if dimension_label == "hour":
                    label = f"Hour {b_int:02d}: anomalous {value_col}"
                    feat  = {"hour": b_int, "metric": value_col}
                    win_s = f"{b_int:02d}:00"
                    win_e = f"{b_int:02d}:59"
                else:
                    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    label = f"{dow_names[b_int]}: anomalous {value_col}"
                    feat  = {"day_of_week": b_int, "metric": value_col}
                    win_s = win_e = dow_names[b_int]

                patterns.append(MinedPattern(
                    source           = self.source,
                    miner            = self.__class__.__name__,
                    pattern_type     = PatternType.TIME_OF_DAY,
                    label            = label,
                    description      = (
                        f"KW H={h_stat:.2f} (p={p_kw:.4f}); bin {b_int} "
                        f"mean={grp.mean():.4f} vs baseline={baseline_mean:.4f}; "
                        f"Cohen's d={d:.3f}"
                    ),
                    feature_dict     = feat,
                    window_start     = win_s,
                    window_end       = win_e,
                    sample_size      = len(grp),
                    p_value          = p_kw,
                    effect_size      = abs(d),
                    effect_size_type = EffectSizeType.COHENS_D,
                    win_rate         = win_r,
                    avg_pnl          = avg_pnl,
                    avg_pnl_baseline = float(baseline_mean),
                    status           = PatternStatus.NEW,
                    tags             = [dimension_label, value_col, "time_of_day"],
                    raw_group        = pd.Series(grp),
                    raw_baseline     = pd.Series(baseline),
                ))

        return patterns

    # ── public ───────────────────────────────────────────────────────────────

    def mine(self, trades_df: pd.DataFrame) -> List[MinedPattern]:
        """
        Run all time-of-day analyses and return a list of MinedPattern.

        Parameters
        ----------
        trades_df : DataFrame with at minimum 'ts'/'exit_time' and 'pnl' columns.

        Returns
        -------
        List[MinedPattern]
        """
        if trades_df is None or trades_df.empty:
            logger.warning("TimeOfDayMiner: empty trades_df")
            return []

        try:
            df = self._prep(trades_df)
        except ValueError as exc:
            logger.error("TimeOfDayMiner prep failed: %s", exc)
            return []

        patterns: List[MinedPattern] = []

        # Hour-of-day × PnL
        patterns.extend(self._kw_and_dunn(df, "_hour", "_pnl", "hour"))
        # Hour-of-day × Win indicator
        patterns.extend(self._kw_and_dunn(df, "_hour", "_win", "hour"))
        # Day-of-week × PnL
        patterns.extend(self._kw_and_dunn(df, "_dow", "_pnl", "dow"))
        # Day-of-week × Win indicator
        patterns.extend(self._kw_and_dunn(df, "_dow", "_win", "dow"))

        logger.info("TimeOfDayMiner produced %d pattern(s)", len(patterns))
        return patterns


# ── convenience function ──────────────────────────────────────────────────────

def mine_time_of_day(
    trades_df: pd.DataFrame,
    source: str = "live",
    **kwargs,
) -> List[MinedPattern]:
    """Shortcut: instantiate TimeOfDayMiner and run it."""
    miner = TimeOfDayMiner(source=source, **kwargs)
    return miner.mine(trades_df)
