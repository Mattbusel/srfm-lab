"""
SignalAnalysisTemplate: analyse a new trading signal's predictive quality.

Computes:
  - Information Coefficient (IC) vs N-bar forward returns
  - Signal value distribution statistics
  - By-regime breakdown of IC and win rate
  - t-stat and p-value for significance
  - Decile analysis (does the top decile outperform the bottom?)

Output: analysis_report dict suitable for storage and display.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SignalAnalysisReport:
    """Full output of a signal analysis run."""

    signal_name: str
    n_observations: int
    ic_mean: float                # mean Information Coefficient
    ic_std: float                 # std of IC
    ic_ir: float                  # IC / std (information ratio)
    t_stat: float
    p_value: float
    is_significant: bool          # p < 0.05 two-tailed
    by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)
    decile_returns: List[float] = field(default_factory=list)  # 10 values
    signal_percentiles: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "n_observations": self.n_observations,
            "ic_mean": round(self.ic_mean, 4),
            "ic_std": round(self.ic_std, 4),
            "ic_ir": round(self.ic_ir, 4),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "by_regime": {k: {m: round(v, 4) for m, v in vd.items()} for k, vd in self.by_regime.items()},
            "decile_returns": [round(x, 4) for x in self.decile_returns],
            "signal_percentiles": {k: round(v, 4) for k, v in self.signal_percentiles.items()},
            "notes": self.notes,
        }


class SignalAnalysisTemplate:
    """
    Template for signal quality analysis.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Columns required: signal_value, forward_return (pct), regime (optional)
    signal_name : str
        Human-readable name for the signal
    forward_bar : int
        Number of bars the forward return is computed over (for labelling)

    Usage::

        template = SignalAnalysisTemplate()
        report   = template.run(trades_df=df, signal_name="BH_score", forward_bar=4)
        print(report.to_dict())
    """

    def run(
        self,
        trades_df: pd.DataFrame,
        signal_name: str = "signal",
        forward_bar: int = 1,
    ) -> SignalAnalysisReport:
        """Run the full analysis and return a SignalAnalysisReport."""
        df = trades_df.copy().dropna(subset=["signal_value", "forward_return"])
        n = len(df)
        notes: List[str] = []

        if n < 30:
            notes.append(f"WARNING: Only {n} observations — results may be unreliable.")

        # ── IC computation ───────────────────────────────────────────────────────
        # Compute rolling IC per day if a date column is present; else single IC
        if "date" in df.columns or "timestamp" in df.columns:
            date_col = "date" if "date" in df.columns else "timestamp"
            df[date_col] = pd.to_datetime(df[date_col])
            df["period"] = df[date_col].dt.to_period("D")
            ic_series = (
                df.groupby("period")
                .apply(lambda g: g["signal_value"].corr(g["forward_return"], method="spearman"))
                .dropna()
            )
            ic_values = ic_series.values
        else:
            # Single IC for the whole dataset
            ic_scalar = df["signal_value"].corr(df["forward_return"], method="spearman")
            ic_values = np.array([ic_scalar])
            notes.append("No date column — computing single IC for entire dataset.")

        ic_mean = float(np.nanmean(ic_values))
        ic_std = float(np.nanstd(ic_values)) if len(ic_values) > 1 else 0.0
        ic_ir = ic_mean / ic_std if ic_std > 1e-9 else 0.0

        # ── t-test on IC ─────────────────────────────────────────────────────────
        if len(ic_values) > 1:
            t_stat, p_value = stats.ttest_1samp(ic_values, 0.0)
        else:
            t_stat = ic_mean * math.sqrt(n) / (df["forward_return"].std() + 1e-9)
            p_value = 2 * stats.t.sf(abs(t_stat), df=max(n - 1, 1))

        # ── by-regime breakdown ───────────────────────────────────────────────────
        by_regime: Dict[str, Dict[str, float]] = {}
        if "regime" in df.columns:
            for regime, grp in df.groupby("regime"):
                if len(grp) < 10:
                    continue
                regime_ic = float(grp["signal_value"].corr(grp["forward_return"], method="spearman"))
                wr = float((grp["forward_return"] > 0).mean())
                avg_ret = float(grp["forward_return"].mean())
                by_regime[str(regime)] = {
                    "ic": regime_ic,
                    "win_rate": wr,
                    "avg_return": avg_ret,
                    "n": len(grp),
                }
        else:
            notes.append("No 'regime' column — skipping by-regime breakdown.")

        # ── decile analysis ───────────────────────────────────────────────────────
        try:
            df["decile"] = pd.qcut(df["signal_value"], q=10, labels=False, duplicates="drop")
            decile_returns = (
                df.groupby("decile")["forward_return"].mean().reindex(range(10), fill_value=float("nan")).tolist()
            )
        except Exception:
            decile_returns = []
            notes.append("Could not compute decile returns (insufficient distinct signal values?).")

        # ── distribution stats ────────────────────────────────────────────────────
        sv = df["signal_value"]
        signal_percentiles = {
            "p5": float(sv.quantile(0.05)),
            "p25": float(sv.quantile(0.25)),
            "p50": float(sv.quantile(0.50)),
            "p75": float(sv.quantile(0.75)),
            "p95": float(sv.quantile(0.95)),
            "mean": float(sv.mean()),
            "std": float(sv.std()),
            "skew": float(sv.skew()),
        }

        return SignalAnalysisReport(
            signal_name=signal_name,
            n_observations=n,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            t_stat=float(t_stat),
            p_value=float(p_value),
            is_significant=float(p_value) < 0.05,
            by_regime=by_regime,
            decile_returns=[float(x) for x in decile_returns],
            signal_percentiles=signal_percentiles,
            notes=notes,
        )

    @staticmethod
    def generate_synthetic_trades(n: int = 500, seed: int = 42) -> pd.DataFrame:
        """Generate a synthetic trades DataFrame for testing."""
        rng = np.random.default_rng(seed)
        signal = rng.normal(0, 1, n)
        noise = rng.normal(0, 2, n)
        forward_return = 0.15 * signal + noise   # weak positive IC ~0.075
        regimes = rng.choice(["bull", "bear", "high_vol"], size=n)
        dates = pd.date_range("2023-01-01", periods=n, freq="1h")
        return pd.DataFrame({
            "signal_value": signal,
            "forward_return": forward_return,
            "regime": regimes,
            "date": dates,
        })
