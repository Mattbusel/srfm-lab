"""
idea-engine/ingestion/miners/mass_physics_miner.py
───────────────────────────────────────────────────
SRFM-specific miner: analyses BH (Black-Hole) mass trajectories before,
during, and after trades to find the "optimal activation pattern".

Key concepts
────────────
  d_bh_mass / h_bh_mass / m15_bh_mass
      BH mass values on daily / hourly / 15-minute timeframes.
      A mass > 1.0 indicates an activated black-hole attractor.
      Mass > 1.92 is considered "fully activated" (near singularity threshold).
      Mass in [1.50, 1.92) is an "early warning" — approaching activation.

  bh_active (d/h/m15)
      Binary flag: 1 = BH is currently pulling price, 0 = inactive.

  tf_score
      Combined multi-timeframe score; higher = stronger trend alignment.

What this miner does
─────────────────────
  1. For each trade, look up the BH mass trajectory for the N bars before entry.
  2. Classify each trade's "entry context" into mass-trajectory categories:
       - COLD     : all masses < 1.0 before trade
       - WARMING  : avg mass in [1.0, 1.50)
       - EARLY_WRN: at least one mass in [1.50, 1.92)
       - ACTIVE   : at least one mass > 1.92
       - DECAYING : mass was > 1.92 and has declined significantly
  3. Compare PnL statistics across categories.
  4. Look for "optimal activation pattern": mass ramps steadily to > 1.92
     over the lookback window, then stays there — find whether this leads
     to outsized positive PnL.
  5. Report each statistically significant category as a MinedPattern.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from ..config import (
    BH_EARLY_WARNING_HIGH,
    BH_EARLY_WARNING_LOW,
    BH_TRAJECTORY_LOOKBACK,
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


# ── BH mass categories ────────────────────────────────────────────────────────

class MassContext(str, Enum):
    COLD      = "COLD"       # all masses < 1.0
    WARMING   = "WARMING"    # avg mass ∈ [1.0, 1.50)
    EARLY_WRN = "EARLY_WRN"  # any mass ∈ [1.50, 1.92)
    ACTIVE    = "ACTIVE"     # any mass ≥ 1.92
    DECAYING  = "DECAYING"   # was ACTIVE, now declining
    UNKNOWN   = "UNKNOWN"


# ── statistics helpers ────────────────────────────────────────────────────────

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    dom = sum(1 if xi > xj else (-1 if xi < xj else 0) for xi in a for xj in b)
    return float(dom / (n1 * n2))


def _bh_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    n = len(p_values)
    if n == 0:
        return []
    order   = sorted(range(n), key=lambda i: p_values[i])
    reject  = [False] * n
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / n:
            reject[idx] = True
    return reject


# ── BH trajectory feature extraction ─────────────────────────────────────────

def _extract_trajectory(
    ts: pd.Timestamp,
    regime_log: pd.DataFrame,
    lookback: int = BH_TRAJECTORY_LOOKBACK,
) -> Optional[pd.DataFrame]:
    """
    Return the N regime_log rows immediately before (and including) ts.

    Returns None if no rows are available.
    """
    if regime_log is None or regime_log.empty or "_ts" not in regime_log.columns:
        return None
    before = regime_log[regime_log["_ts"] <= ts].tail(lookback)
    if len(before) == 0:
        return None
    return before


def _classify_context(traj: Optional[pd.DataFrame]) -> MassContext:
    """
    Classify the BH mass context from a trajectory window.

    Precedence: ACTIVE > DECAYING > EARLY_WRN > WARMING > COLD
    """
    if traj is None or traj.empty:
        return MassContext.UNKNOWN

    mass_cols = [c for c in ["d_bh_mass", "h_bh_mass", "m15_bh_mass"] if c in traj.columns]
    if not mass_cols:
        return MassContext.UNKNOWN

    masses = traj[mass_cols].values.flatten()
    masses = masses[~np.isnan(masses)]
    if len(masses) == 0:
        return MassContext.UNKNOWN

    avg_mass   = float(np.mean(masses))
    max_mass   = float(np.max(masses))
    last_row   = traj.iloc[-1]

    last_masses_arr = np.array([
        float(last_row.get(c, 0) or 0) for c in mass_cols
    ])
    last_avg = float(last_masses_arr.mean())

    # Check for DECAYING: peak was ACTIVE but current avg has dropped
    if traj.shape[0] >= 2:
        first_masses = traj.iloc[0][mass_cols].values.astype(float)
        first_max    = float(np.nanmax(first_masses)) if len(first_masses) > 0 else 0.0
        if first_max >= BH_EARLY_WARNING_HIGH and last_avg < first_max * 0.75:
            return MassContext.DECAYING

    if max_mass >= BH_EARLY_WARNING_HIGH:
        return MassContext.ACTIVE
    if max_mass >= BH_EARLY_WARNING_LOW:
        return MassContext.EARLY_WRN
    if avg_mass >= 1.0:
        return MassContext.WARMING
    return MassContext.COLD


def _compute_mass_ramp(traj: Optional[pd.DataFrame]) -> Optional[float]:
    """
    Compute the slope of BH mass over the lookback window (avg across TFs).

    Returns the linear regression slope (mass units per bar) or None.
    """
    if traj is None or len(traj) < 2:
        return None
    mass_cols = [c for c in ["d_bh_mass", "h_bh_mass", "m15_bh_mass"] if c in traj.columns]
    if not mass_cols:
        return None
    avg_mass = traj[mass_cols].mean(axis=1).values.astype(float)
    nan_mask = np.isnan(avg_mass)
    if nan_mask.all():
        return None
    x = np.arange(len(avg_mass), dtype=float)
    x, y = x[~nan_mask], avg_mass[~nan_mask]
    if len(x) < 2:
        return None
    coef = np.polyfit(x, y, 1)
    return float(coef[0])


def _compute_tf_alignment(traj: Optional[pd.DataFrame]) -> Optional[float]:
    """Mean tf_score over the trajectory window."""
    if traj is None or traj.empty or "tf_score" not in traj.columns:
        return None
    vals = pd.to_numeric(traj["tf_score"], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) > 0 else None


# ── main miner ────────────────────────────────────────────────────────────────

class MassPhysicsMiner:
    """
    Mines BH mass trajectory patterns from live trade data.

    For each trade, extracts:
      - The mass context category (COLD, WARMING, EARLY_WRN, ACTIVE, DECAYING)
      - The mass ramp slope (is mass accelerating?)
      - tf_score alignment

    Then tests which contexts produce significantly different PnL.
    """

    def __init__(
        self,
        source:      str   = "live",
        lookback:    int   = BH_TRAJECTORY_LOOKBACK,
        alpha:       float = RAW_P_VALUE_THRESHOLD,
        min_effect:  float = MIN_EFFECT_SIZE,
        min_sample:  int   = MIN_GROUP_SAMPLE,
    ):
        self.source     = source
        self.lookback   = lookback
        self.alpha      = alpha
        self.min_effect = min_effect
        self.min_sample = min_sample

    # ── feature labelling ─────────────────────────────────────────────────

    def _label_trades(
        self,
        trades: pd.DataFrame,
        regime_log: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each trade, compute BH context features.

        Adds columns: mass_context, mass_ramp, tf_align
        """
        regime_log = regime_log.copy()
        regime_log["_ts"] = pd.to_datetime(regime_log["ts"], errors="coerce")
        regime_log = regime_log.sort_values("_ts").reset_index(drop=True)

        ts_col = "ts" if "ts" in trades.columns else "exit_time"
        trades = trades.copy()
        trades["_ts"] = pd.to_datetime(trades[ts_col], errors="coerce")
        trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")
        trades = trades.dropna(subset=["_ts", "pnl"]).copy()

        contexts: List[str]            = []
        ramps:    List[Optional[float]] = []
        tf_aligns: List[Optional[float]] = []

        for _, row in trades.iterrows():
            traj = _extract_trajectory(row["_ts"], regime_log, self.lookback)
            ctx  = _classify_context(traj)
            ramp = _compute_mass_ramp(traj)
            tfa  = _compute_tf_alignment(traj)
            contexts.append(ctx.value)
            ramps.append(ramp)
            tf_aligns.append(tfa)

        trades["mass_context"] = contexts
        trades["mass_ramp"]    = ramps
        trades["tf_align"]     = tf_aligns
        return trades

    # ── context-based patterns ────────────────────────────────────────────

    def _mine_context_patterns(self, labelled: pd.DataFrame) -> List[MinedPattern]:
        patterns: List[MinedPattern] = []
        all_pnl = labelled["pnl"].dropna().values

        unique_ctxs = [c for c in MassContext if c != MassContext.UNKNOWN]
        groups      = {}
        for ctx in unique_ctxs:
            g = labelled[labelled["mass_context"] == ctx.value]["pnl"].dropna().values
            if len(g) >= self.min_sample:
                groups[ctx.value] = g

        if len(groups) < 2:
            return patterns

        # Kruskal-Wallis across contexts
        try:
            h_stat, p_kw = kruskal(*list(groups.values()))
        except Exception as exc:
            logger.debug("KW test for mass contexts failed: %s", exc)
            return patterns

        # Even if KW not significant, still report ACTIVE vs baseline if effect is large
        # Pairwise with BH correction
        ctx_keys = list(groups.keys())
        p_vals: List[float]     = []
        pairs:  List[Tuple]     = []
        for i, ka in enumerate(ctx_keys):
            for kb in ctx_keys[i + 1:]:
                try:
                    _, p = mannwhitneyu(groups[ka], groups[kb], alternative="two-sided")
                except Exception:
                    p = 1.0
                p_vals.append(float(p))
                pairs.append((ka, kb))

        rejected = _bh_correction(p_vals, alpha=self.alpha)

        # Build a pattern for each context that appears in at least one significant pair
        flagged_contexts: set = set()
        for (ka, kb), p, rej in zip(pairs, p_vals, rejected):
            if rej:
                flagged_contexts.add(ka)
                flagged_contexts.add(kb)

        # Also flag any context where effect size vs full baseline is large
        for ctx_name, grp in groups.items():
            d = _cohens_d(grp, all_pnl)
            if abs(d) >= self.min_effect:
                flagged_contexts.add(ctx_name)

        for ctx_name in flagged_contexts:
            if ctx_name not in groups:
                continue
            grp  = groups[ctx_name]
            base = all_pnl
            d    = _cohens_d(grp, base)
            delt = _cliffs_delta(grp, base)

            # Find best pairwise p-value for this context
            best_p = 1.0
            for (ka, kb), p in zip(pairs, p_vals):
                if ctx_name in (ka, kb):
                    best_p = min(best_p, p)

            win_r = float((grp > 0).sum() / len(grp))
            pf_g  = grp[grp > 0].sum()
            pf_l  = abs(grp[grp < 0].sum())
            pf    = float(pf_g / pf_l) if pf_l > 0 else None

            # Describe the context
            ctx_descriptions = {
                MassContext.COLD.value:      "All BH masses < 1.0 before trade entry",
                MassContext.WARMING.value:   "Avg BH mass in [1.0, 1.50) — BH is warming up",
                MassContext.EARLY_WRN.value: "BH mass in early-warning zone [1.50, 1.92) — approaching activation",
                MassContext.ACTIVE.value:    "BH fully activated (mass ≥ 1.92) before trade",
                MassContext.DECAYING.value:  "BH was active but mass is declining — momentum fading",
            }

            patterns.append(MinedPattern(
                source           = self.source,
                miner            = "MassPhysicsMiner",
                pattern_type     = PatternType.BH_PHYSICS,
                label            = f"BH context {ctx_name}: anomalous PnL",
                description      = (
                    f"{ctx_descriptions.get(ctx_name, ctx_name)}. "
                    f"N={len(grp)}, mean PnL={grp.mean():.2f} vs {base.mean():.2f}; "
                    f"Cohen's d={d:.3f}, Cliff's δ={delt:.3f}, p={best_p:.4f}"
                ),
                feature_dict     = {
                    "mass_context":  ctx_name,
                    "kw_h_stat":     float(h_stat),
                    "kw_p_value":    float(p_kw),
                    "low_threshold": BH_EARLY_WARNING_LOW,
                    "high_threshold": BH_EARLY_WARNING_HIGH,
                },
                sample_size      = int(len(grp)),
                p_value          = best_p,
                effect_size      = abs(delt),
                effect_size_type = EffectSizeType.CLIFFS_DELTA,
                win_rate         = win_r,
                avg_pnl          = float(grp.mean()),
                avg_pnl_baseline = float(base.mean()),
                profit_factor    = pf,
                status           = PatternStatus.NEW,
                tags             = ["bh_physics", "mass_context", ctx_name.lower()],
                raw_group        = pd.Series(grp),
                raw_baseline     = pd.Series(base),
            ))

        return patterns

    # ── ramp / slope patterns ─────────────────────────────────────────────

    def _mine_ramp_patterns(self, labelled: pd.DataFrame) -> List[MinedPattern]:
        """
        Find whether a positive mass ramp (accelerating BH) predicts better PnL.

        Splits trades into: negative ramp, near-zero ramp, positive ramp.
        """
        patterns: List[MinedPattern] = []

        df = labelled.dropna(subset=["mass_ramp", "pnl"]).copy()
        if len(df) < self.min_sample * 3:
            return patterns

        q33 = df["mass_ramp"].quantile(0.33)
        q67 = df["mass_ramp"].quantile(0.67)

        neg  = df[df["mass_ramp"] <= q33]["pnl"].values
        mid  = df[(df["mass_ramp"] > q33) & (df["mass_ramp"] < q67)]["pnl"].values
        pos  = df[df["mass_ramp"] >= q67]["pnl"].values

        if any(len(g) < self.min_sample for g in [neg, mid, pos]):
            return patterns

        try:
            h_stat, p_kw = kruskal(neg, mid, pos)
        except Exception:
            return patterns

        if p_kw >= self.alpha:
            return patterns

        for label, grp, ramp_label in [
            ("negative ramp", neg, f"≤{q33:.3f}"),
            ("flat ramp", mid, f"({q33:.3f},{q67:.3f})"),
            ("positive ramp", pos, f"≥{q67:.3f}"),
        ]:
            all_pnl = labelled["pnl"].dropna().values
            d    = _cohens_d(grp, all_pnl)
            delt = _cliffs_delta(grp, all_pnl)
            if abs(delt) < self.min_effect:
                continue

            win_r = float((grp > 0).sum() / len(grp))
            pf_g  = grp[grp > 0].sum()
            pf_l  = abs(grp[grp < 0].sum())
            pf    = float(pf_g / pf_l) if pf_l > 0 else None

            patterns.append(MinedPattern(
                source           = self.source,
                miner            = "MassPhysicsMiner",
                pattern_type     = PatternType.BH_PHYSICS,
                label            = f"BH mass {label}: anomalous PnL",
                description      = (
                    f"Trades with {label} (ramp {ramp_label} mass/bar): "
                    f"mean PnL={grp.mean():.2f} vs {all_pnl.mean():.2f}; "
                    f"KW H={h_stat:.2f} (p={p_kw:.4f}), Cliff's δ={delt:.3f}"
                ),
                feature_dict     = {
                    "ramp_label":     label,
                    "ramp_threshold": ramp_label,
                    "kw_p":           float(p_kw),
                },
                sample_size      = int(len(grp)),
                p_value          = float(p_kw),
                effect_size      = abs(delt),
                effect_size_type = EffectSizeType.CLIFFS_DELTA,
                win_rate         = win_r,
                avg_pnl          = float(grp.mean()),
                avg_pnl_baseline = float(all_pnl.mean()),
                profit_factor    = pf,
                status           = PatternStatus.NEW,
                tags             = ["bh_physics", "mass_ramp", label.replace(" ", "_")],
                raw_group        = pd.Series(grp),
                raw_baseline     = pd.Series(all_pnl),
            ))

        return patterns

    # ── early-warning signal detection ───────────────────────────────────

    def _mine_early_warning_patterns(self, labelled: pd.DataFrame) -> List[MinedPattern]:
        """
        Find the "optimal early-warning" signal: mass enters [1.50, 1.92) and
        then crosses 1.92 during the hold period → predict outsized gain.
        """
        patterns: List[MinedPattern] = []

        ew_trades    = labelled[labelled["mass_context"] == MassContext.EARLY_WRN.value]
        other_trades = labelled[labelled["mass_context"] != MassContext.EARLY_WRN.value]

        if len(ew_trades) < self.min_sample or len(other_trades) < self.min_sample:
            return patterns

        ew_pnl    = ew_trades["pnl"].dropna().values
        other_pnl = other_trades["pnl"].dropna().values

        try:
            _, p = mannwhitneyu(ew_pnl, other_pnl, alternative="two-sided")
        except Exception:
            return patterns

        d    = _cohens_d(ew_pnl, other_pnl)
        delt = _cliffs_delta(ew_pnl, other_pnl)

        if abs(delt) < self.min_effect and p >= self.alpha:
            return patterns

        win_r = float((ew_pnl > 0).sum() / len(ew_pnl))
        pf_g  = ew_pnl[ew_pnl > 0].sum()
        pf_l  = abs(ew_pnl[ew_pnl < 0].sum())
        pf    = float(pf_g / pf_l) if pf_l > 0 else None

        patterns.append(MinedPattern(
            source           = self.source,
            miner            = "MassPhysicsMiner",
            pattern_type     = PatternType.BH_PHYSICS,
            label            = f"BH early-warning entry signal (mass [{BH_EARLY_WARNING_LOW},{BH_EARLY_WARNING_HIGH}))",
            description      = (
                f"Trades entered when BH mass is in early-warning zone "
                f"[{BH_EARLY_WARNING_LOW}, {BH_EARLY_WARNING_HIGH}): "
                f"mean PnL={ew_pnl.mean():.2f} vs others={other_pnl.mean():.2f}; "
                f"p={p:.4f}, Cliff's δ={delt:.3f}, Cohen's d={d:.3f}"
            ),
            feature_dict     = {
                "mass_context":       MassContext.EARLY_WRN.value,
                "low_threshold":      BH_EARLY_WARNING_LOW,
                "high_threshold":     BH_EARLY_WARNING_HIGH,
                "early_warning_signal": True,
            },
            sample_size      = int(len(ew_pnl)),
            p_value          = float(p),
            effect_size      = abs(delt),
            effect_size_type = EffectSizeType.CLIFFS_DELTA,
            win_rate         = win_r,
            avg_pnl          = float(ew_pnl.mean()),
            avg_pnl_baseline = float(other_pnl.mean()),
            profit_factor    = pf,
            status           = PatternStatus.NEW,
            tags             = ["bh_physics", "early_warning", "activation_signal"],
            raw_group        = pd.Series(ew_pnl),
            raw_baseline     = pd.Series(other_pnl),
        ))

        return patterns

    # ── public ───────────────────────────────────────────────────────────

    def mine(self, live_data: LiveTradeData) -> List[MinedPattern]:
        """
        Run all BH physics analyses and return MinedPattern objects.

        Parameters
        ----------
        live_data : LiveTradeData (must have trades and regime_log)

        Returns
        -------
        List[MinedPattern]
        """
        if live_data.trades is None or live_data.trades.empty:
            logger.warning("MassPhysicsMiner: no trades data")
            return []
        if live_data.regime_log is None or live_data.regime_log.empty:
            logger.warning("MassPhysicsMiner: no regime_log data")
            return []
        if "pnl" not in live_data.trades.columns:
            logger.warning("MassPhysicsMiner: trades missing 'pnl' column")
            return []

        logger.info("MassPhysicsMiner: labelling %d trades with BH context …", len(live_data.trades))
        labelled = self._label_trades(live_data.trades, live_data.regime_log)

        patterns: List[MinedPattern] = []
        patterns.extend(self._mine_context_patterns(labelled))
        patterns.extend(self._mine_ramp_patterns(labelled))
        patterns.extend(self._mine_early_warning_patterns(labelled))

        logger.info("MassPhysicsMiner produced %d pattern(s)", len(patterns))
        return patterns


def mine_mass_physics(live_data: LiveTradeData, source: str = "live", **kwargs) -> List[MinedPattern]:
    """Shortcut function."""
    return MassPhysicsMiner(source=source, **kwargs).mine(live_data)
