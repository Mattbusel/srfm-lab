"""
IdeaValidationTemplate: end-to-end validation of an IAE hypothesis.

Steps:
  1. Apply the hypothesis (parameter delta) to the last 90 days of trades
  2. Compare vs baseline on: WR, avg P&L, Sharpe, max drawdown
  3. Compute statistical significance (t-test on P&L difference)
  4. Overfitting check: compare in-sample (days 0-60) vs OOS (days 61-90)
  5. Render: APPROVED / REJECTED / NEEDS_MORE_DATA

Output: ValidationResult
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

APPROVED = "APPROVED"
REJECTED = "REJECTED"
NEEDS_MORE_DATA = "NEEDS_MORE_DATA"


@dataclass
class ValidationResult:
    """Result of idea validation."""

    hypothesis_id: str
    verdict: str                     # APPROVED | REJECTED | NEEDS_MORE_DATA
    baseline_sharpe: float
    hypothesis_sharpe: float
    sharpe_delta: float
    baseline_wr: float
    hypothesis_wr: float
    wr_delta: float
    t_stat: float
    p_value: float
    is_significant: bool
    is_sample_is_oos_consistent: bool   # True if IS and OOS point same direction
    is_sharpe_improvement: float
    oos_sharpe_improvement: float
    n_trades_total: int
    n_trades_oos: int
    reason: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "verdict": self.verdict,
            "baseline_sharpe": round(self.baseline_sharpe, 4),
            "hypothesis_sharpe": round(self.hypothesis_sharpe, 4),
            "sharpe_delta": round(self.sharpe_delta, 4),
            "baseline_wr": round(self.baseline_wr, 4),
            "hypothesis_wr": round(self.hypothesis_wr, 4),
            "wr_delta": round(self.wr_delta, 4),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "is_sample_is_oos_consistent": self.is_sample_is_oos_consistent,
            "is_sharpe_improvement": round(self.is_sharpe_improvement, 4),
            "oos_sharpe_improvement": round(self.oos_sharpe_improvement, 4),
            "n_trades_total": self.n_trades_total,
            "n_trades_oos": self.n_trades_oos,
            "reason": self.reason,
            "warnings": self.warnings,
        }


class IdeaValidationTemplate:
    """
    Validates an IAE hypothesis against recent trade data.

    The hypothesis is encoded as a Python callable ``apply_fn`` that takes a
    DataFrame and returns a filtered/modified DataFrame representing the
    trades that would occur under the new hypothesis.

    Usage::

        def hypothesis_fn(df):
            # Block trades at hour 13
            return df[df["entry_hour"] != 13]

        template = IdeaValidationTemplate()
        result   = template.run(
            trades_df=df,
            hypothesis_id="hyp_block_hour_13",
            apply_fn=hypothesis_fn,
        )
    """

    def run(
        self,
        trades_df: pd.DataFrame,
        hypothesis_id: str,
        apply_fn: Callable[[pd.DataFrame], pd.DataFrame],
        window_days: int = 90,
        is_split_days: int = 60,
    ) -> ValidationResult:
        warnings: List[str] = []

        df = trades_df.copy().dropna(subset=["pnl"])
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
            cutoff = df["entry_time"].max() - pd.Timedelta(days=window_days)
            df = df[df["entry_time"] >= cutoff].copy()

        if len(df) < 20:
            return ValidationResult(
                hypothesis_id=hypothesis_id,
                verdict=NEEDS_MORE_DATA,
                baseline_sharpe=0, hypothesis_sharpe=0, sharpe_delta=0,
                baseline_wr=0, hypothesis_wr=0, wr_delta=0,
                t_stat=0, p_value=1, is_significant=False,
                is_sample_is_oos_consistent=False,
                is_sharpe_improvement=0, oos_sharpe_improvement=0,
                n_trades_total=len(df), n_trades_oos=0,
                reason=f"Only {len(df)} trades in window — need ≥ 20.",
                warnings=warnings,
            )

        # ── Apply hypothesis ──────────────────────────────────────────────────────
        try:
            df_hyp = apply_fn(df.copy())
        except Exception as exc:
            warnings.append(f"apply_fn raised: {exc}")
            df_hyp = df.copy()

        # ── IS / OOS split ────────────────────────────────────────────────────────
        if "entry_time" in df.columns and df["entry_time"].notna().any():
            is_cutoff = df["entry_time"].max() - pd.Timedelta(days=window_days - is_split_days)
            df_is = df[df["entry_time"] < is_cutoff]
            df_oos = df[df["entry_time"] >= is_cutoff]
            df_hyp_is = df_hyp[df_hyp.index.isin(df_is.index)] if len(df_hyp) > 0 else df_hyp
            df_hyp_oos = df_hyp[df_hyp.index.isin(df_oos.index)] if len(df_hyp) > 0 else df_hyp
        else:
            split = int(len(df) * is_split_days / window_days)
            df_is, df_oos = df.iloc[:split], df.iloc[split:]
            df_hyp_is = df_hyp.iloc[:split] if len(df_hyp) > split else df_hyp
            df_hyp_oos = df_hyp.iloc[split:] if len(df_hyp) > split else pd.DataFrame(columns=df.columns)

        # ── Full-window comparison ────────────────────────────────────────────────
        base_pnl = df["pnl"].values
        hyp_pnl = df_hyp["pnl"].values if len(df_hyp) > 0 else np.array([0.0])

        base_sharpe = self._sharpe(base_pnl)
        hyp_sharpe = self._sharpe(hyp_pnl)
        base_wr = float((base_pnl > 0).mean())
        hyp_wr = float((hyp_pnl > 0).mean()) if len(hyp_pnl) > 0 else 0.0

        # Pad to same length for t-test
        n_min = min(len(base_pnl), len(hyp_pnl))
        if n_min >= 5:
            t_stat, p_value = stats.ttest_rel(hyp_pnl[:n_min], base_pnl[:n_min])
        else:
            t_stat, p_value = 0.0, 1.0
            warnings.append("Insufficient paired observations for t-test.")

        # ── IS vs OOS consistency ─────────────────────────────────────────────────
        is_imp = self._sharpe(df_hyp_is["pnl"].values) - self._sharpe(df_is["pnl"].values) if len(df_is) > 0 else 0.0
        oos_imp = self._sharpe(df_hyp_oos["pnl"].values) - self._sharpe(df_oos["pnl"].values) if len(df_oos) > 0 else 0.0
        consistent = (is_imp > 0) == (oos_imp > 0)

        if len(df_oos) < 10:
            warnings.append(f"OOS window has only {len(df_oos)} trades.")

        # ── Verdict logic ─────────────────────────────────────────────────────────
        sharpe_delta = hyp_sharpe - base_sharpe
        significant = float(p_value) < 0.05

        if len(hyp_pnl) < 10:
            verdict = NEEDS_MORE_DATA
            reason = "Hypothesis filters too many trades — insufficient sample post-filter."
        elif significant and sharpe_delta > 0.1 and consistent:
            verdict = APPROVED
            reason = (
                f"Significant improvement (p={p_value:.3f}), Sharpe +{sharpe_delta:.3f}, "
                "consistent IS/OOS."
            )
        elif significant and sharpe_delta < -0.1:
            verdict = REJECTED
            reason = f"Significant degradation (p={p_value:.3f}), Sharpe {sharpe_delta:.3f}."
        elif not significant and abs(sharpe_delta) > 0.05:
            verdict = NEEDS_MORE_DATA
            reason = f"Direction is {'positive' if sharpe_delta > 0 else 'negative'} but not significant (p={p_value:.3f})."
        elif not consistent:
            verdict = NEEDS_MORE_DATA
            reason = "IS and OOS point in different directions — possible overfitting."
        else:
            verdict = NEEDS_MORE_DATA
            reason = f"Marginal result (Sharpe delta={sharpe_delta:.3f}, p={p_value:.3f})."

        return ValidationResult(
            hypothesis_id=hypothesis_id,
            verdict=verdict,
            baseline_sharpe=base_sharpe,
            hypothesis_sharpe=hyp_sharpe,
            sharpe_delta=sharpe_delta,
            baseline_wr=base_wr,
            hypothesis_wr=hyp_wr,
            wr_delta=hyp_wr - base_wr,
            t_stat=float(t_stat),
            p_value=float(p_value),
            is_significant=significant,
            is_sample_is_oos_consistent=consistent,
            is_sharpe_improvement=is_imp,
            oos_sharpe_improvement=oos_imp,
            n_trades_total=len(df),
            n_trades_oos=len(df_oos),
            reason=reason,
            warnings=warnings,
        )

    @staticmethod
    def _sharpe(pnl: np.ndarray, annualise_factor: float = math.sqrt(252 * 24)) -> float:
        if len(pnl) < 2:
            return 0.0
        std = float(np.std(pnl, ddof=1))
        return float(np.mean(pnl) / std * annualise_factor) if std > 1e-9 else 0.0

    @staticmethod
    def generate_synthetic_trades(n: int = 400, seed: int = 21) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        entry_times = pd.date_range("2023-07-01", periods=n, freq="6h")
        hours = entry_times.hour
        # Simulate hour 13 being slightly worse
        pnl = rng.normal(0.003, 0.015, n) - 0.008 * (hours == 13)
        return pd.DataFrame({
            "pnl": pnl,
            "entry_time": entry_times,
            "entry_hour": hours,
        })
