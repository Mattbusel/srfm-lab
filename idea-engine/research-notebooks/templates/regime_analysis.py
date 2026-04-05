"""
RegimeAnalysisTemplate: split trades by regime and compare performance metrics.

Metrics computed per segment:
  - Win rate, avg P&L, Sharpe ratio, max drawdown, trade count
  - Chi-squared test for win-rate independence between regime and non-regime
  - Recommendation: is the regime filter worth applying?

Output: RegimeAnalysis dataclass with recommendation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RegimeSegmentStats:
    """Performance statistics for a single regime segment."""

    regime_label: str
    n_trades: int
    win_rate: float
    avg_pnl: float
    median_pnl: float
    sharpe: float
    max_drawdown: float
    total_pnl: float
    pnl_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class RegimeAnalysis:
    """Full output of a regime analysis run."""

    regime_name: str
    regime_stats: RegimeSegmentStats
    baseline_stats: RegimeSegmentStats
    chi2_stat: float
    chi2_p_value: float
    win_rate_is_independent: bool   # True if not significantly different
    sharpe_lift: float              # regime Sharpe - baseline Sharpe
    recommendation: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_name": self.regime_name,
            "regime_stats": self.regime_stats.to_dict(),
            "baseline_stats": self.baseline_stats.to_dict(),
            "chi2_stat": round(self.chi2_stat, 4),
            "chi2_p_value": round(self.chi2_p_value, 6),
            "win_rate_is_independent": self.win_rate_is_independent,
            "sharpe_lift": round(self.sharpe_lift, 4),
            "recommendation": self.recommendation,
            "notes": self.notes,
        }


class RegimeAnalysisTemplate:
    """
    Regime analysis: compare performance inside vs outside a regime.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Columns required: pnl (numeric), is_regime (bool/int 0-1)
        Optional: timestamp, instrument

    Usage::

        template = RegimeAnalysisTemplate()
        result   = template.run(df, regime_name="high_vol", regime_column="is_high_vol")
    """

    def run(
        self,
        trades_df: pd.DataFrame,
        regime_name: str = "regime",
        regime_column: str = "is_regime",
    ) -> RegimeAnalysis:
        df = trades_df.copy().dropna(subset=["pnl"])
        notes: List[str] = []

        if regime_column not in df.columns:
            raise ValueError(f"Column '{regime_column}' not found in trades_df.")

        in_regime = df[df[regime_column].astype(bool)]
        out_regime = df[~df[regime_column].astype(bool)]

        if len(in_regime) < 10:
            notes.append(f"WARNING: Only {len(in_regime)} trades in regime — results unreliable.")
        if len(out_regime) < 10:
            notes.append(f"WARNING: Only {len(out_regime)} trades outside regime — results unreliable.")

        regime_stats = self._compute_stats(in_regime["pnl"].values, regime_name)
        baseline_stats = self._compute_stats(out_regime["pnl"].values, "non_" + regime_name)

        # Chi-squared test on win/loss counts
        chi2_stat, chi2_p = self._chi2_win_rate(in_regime["pnl"], out_regime["pnl"])
        win_rate_independent = chi2_p >= 0.05

        sharpe_lift = regime_stats.sharpe - baseline_stats.sharpe

        recommendation = self._make_recommendation(
            regime_stats, baseline_stats, chi2_p, sharpe_lift, notes
        )

        return RegimeAnalysis(
            regime_name=regime_name,
            regime_stats=regime_stats,
            baseline_stats=baseline_stats,
            chi2_stat=chi2_stat,
            chi2_p_value=chi2_p,
            win_rate_is_independent=win_rate_independent,
            sharpe_lift=sharpe_lift,
            recommendation=recommendation,
            notes=notes,
        )

    # ── helpers ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_stats(pnl: np.ndarray, label: str) -> RegimeSegmentStats:
        if len(pnl) == 0:
            return RegimeSegmentStats(label, 0, 0, 0, 0, 0, 0, 0, 0)
        wins = int((pnl > 0).sum())
        n = len(pnl)
        win_rate = wins / n
        avg = float(np.mean(pnl))
        median = float(np.median(pnl))
        std = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
        sharpe = (avg / std * math.sqrt(252)) if std > 1e-9 else 0.0
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = float(np.min(cumulative - running_max))
        return RegimeSegmentStats(
            regime_label=label,
            n_trades=n,
            win_rate=win_rate,
            avg_pnl=avg,
            median_pnl=median,
            sharpe=sharpe,
            max_drawdown=drawdown,
            total_pnl=float(np.sum(pnl)),
            pnl_std=std,
        )

    @staticmethod
    def _chi2_win_rate(
        pnl_in: pd.Series, pnl_out: pd.Series
    ) -> Tuple[float, float]:
        wins_in = (pnl_in > 0).sum()
        losses_in = (pnl_in <= 0).sum()
        wins_out = (pnl_out > 0).sum()
        losses_out = (pnl_out <= 0).sum()
        contingency = np.array([[wins_in, losses_in], [wins_out, losses_out]])
        if contingency.min() == 0:
            return 0.0, 1.0
        chi2, p, _, _ = stats.chi2_contingency(contingency, correction=False)
        return float(chi2), float(p)

    @staticmethod
    def _make_recommendation(
        regime: RegimeSegmentStats,
        baseline: RegimeSegmentStats,
        chi2_p: float,
        sharpe_lift: float,
        notes: List[str],
    ) -> str:
        if chi2_p < 0.05 and sharpe_lift > 0.3:
            return (
                f"APPLY FILTER: Regime shows significantly higher win rate "
                f"({regime.win_rate:.1%} vs {baseline.win_rate:.1%}, p={chi2_p:.3f}) "
                f"and +{sharpe_lift:.2f} Sharpe lift."
            )
        if chi2_p < 0.05 and sharpe_lift < -0.3:
            return (
                f"AVOID REGIME: Win rate is significantly lower in regime "
                f"({regime.win_rate:.1%} vs {baseline.win_rate:.1%}, p={chi2_p:.3f}). "
                "Consider pausing entries or reducing size."
            )
        if chi2_p >= 0.05:
            notes.append("Win rates not statistically different (chi2 p >= 0.05).")
            return "NEUTRAL: No statistically significant regime effect detected."
        return "INCONCLUSIVE: Marginal signal — collect more data."

    @staticmethod
    def generate_synthetic_trades(n: int = 600, regime_fraction: float = 0.3, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        is_regime = rng.random(n) < regime_fraction
        pnl = np.where(is_regime,
                       rng.normal(-0.002, 0.015, n),   # worse in regime
                       rng.normal(0.004, 0.012, n))    # better outside regime
        return pd.DataFrame({"pnl": pnl, "is_regime": is_regime.astype(int)})
