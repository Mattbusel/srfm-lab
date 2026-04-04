"""
research/reconciliation/attribution.py
=========================================
PnL attribution engine for the live-vs-backtest reconciliation pipeline.

Framework
---------
This module adapts the **Brinson-Hood-Beebower (BHB)** attribution framework
to signal-driven algorithmic trading:

  * **Selection effect** – which symbols were traded (live) vs available
    (backtest universe), and how that selection choice explains PnL differences.
  * **Timing effect** – the entry/exit timing relative to the optimal bar
    (i.e., how much PnL was left on the table or lost due to sub-optimal
    entry/exit timing).
  * **Sizing effect** – position size quality, expressed as the ratio of
    realised Kelly fraction to optimal Kelly fraction.
  * **Regime effect** – fraction of PnL variation explained by the prevailing
    market regime (BULL/BEAR/SIDEWAYS/HIGH_VOL).
  * **Signal effect** – fraction of PnL explained by the BH physics signals
    (tf_score, mass, ATR, delta_score) and ensemble (D3QN/DDQN/TD3QN).

Classes
-------
PnLAttributionEngine
    Main analysis class.

Dataclasses
-----------
AttributionReport
    Full attribution breakdown for a set of trades.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class AttributionReport:
    """
    Full PnL attribution report.

    All 'effect' values are in the same units as the input PnL column
    (typically USD or return %).
    """
    n_trades: int
    total_pnl: float

    # BHB-style effects
    selection_effect: float    # symbol selection vs benchmark universe
    timing_effect: float       # entry/exit timing vs optimal bar
    sizing_effect: float       # Kelly fraction realised vs optimal
    regime_effect: float       # PnL explained by regime
    signal_effect: float       # PnL explained by BH signals
    interaction_effect: float  # residual / cross-effects
    unexplained: float         # total - sum of above

    # Granular breakdowns
    selection_by_sym: pd.DataFrame       # sym → selection contribution
    timing_by_regime: pd.DataFrame       # regime → timing contribution
    sizing_stats: dict[str, float]       # Kelly stats
    regime_attribution: pd.DataFrame     # regime → pnl, count, avg_pnl
    signal_attribution: pd.DataFrame     # factor → beta, t-stat, contribution

    # IC (Information Coefficient) analysis
    ic_series: pd.Series                 # per-period IC values
    icir: float                          # IC Information Ratio
    rolling_ic: pd.DataFrame             # rolling IC over time

    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "total_pnl": self.total_pnl,
            "selection_effect": self.selection_effect,
            "timing_effect": self.timing_effect,
            "sizing_effect": self.sizing_effect,
            "regime_effect": self.regime_effect,
            "signal_effect": self.signal_effect,
            "interaction_effect": self.interaction_effect,
            "unexplained": self.unexplained,
            "icir": self.icir,
        }

    def effect_table(self) -> pd.DataFrame:
        effects = {
            "Selection": self.selection_effect,
            "Timing": self.timing_effect,
            "Sizing": self.sizing_effect,
            "Regime": self.regime_effect,
            "Signal": self.signal_effect,
            "Interaction": self.interaction_effect,
            "Unexplained": self.unexplained,
        }
        df = pd.DataFrame(
            list(effects.items()), columns=["Effect", "Value"]
        )
        df["Pct_of_Total"] = df["Value"] / self.total_pnl * 100 if self.total_pnl != 0 else np.nan
        return df.set_index("Effect")


# ── PnLAttributionEngine ──────────────────────────────────────────────────────


class PnLAttributionEngine:
    """
    PnL attribution engine using a BHB-inspired framework adapted for
    algorithmic signal-driven trading.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate (decimal), used in Kelly and Sharpe calculations.
        Default 0.04 (4% p.a.).
    bar_hours : float
        Hours per bar (default 1.0 for 1-hour bars).
    ic_min_obs : int
        Minimum observations required to compute IC for a period (default 5).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        bar_hours: float = 1.0,
        ic_min_obs: int = 5,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.bar_hours = bar_hours
        self.ic_min_obs = ic_min_obs

    # ── Internal utilities ────────────────────────────────────────────────

    def _resolve_col(self, df: pd.DataFrame, *candidates: str) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _get_pnl(self, df: pd.DataFrame) -> pd.Series:
        col = self._resolve_col(df, "pnl", "live_pnl", "bt_pnl", "return_pct")
        if col:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    def _get_sym(self, df: pd.DataFrame) -> pd.Series:
        col = self._resolve_col(df, "sym", "live_sym", "bt_sym", "symbol")
        if col:
            return df[col].astype(str)
        return pd.Series("UNKNOWN", index=df.index)

    def _get_regime(self, df: pd.DataFrame) -> pd.Series:
        col = self._resolve_col(df, "regime", "live_regime", "bt_regime")
        if col:
            return df[col].astype(str)
        return pd.Series("UNKNOWN", index=df.index)

    def _get_signal(self, df: pd.DataFrame, name: str) -> pd.Series:
        for candidate in (name, f"live_{name}", f"bt_{name}"):
            if candidate in df.columns:
                return pd.to_numeric(df[candidate], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    # ── Selection effect ──────────────────────────────────────────────────

    def _compute_selection_effect(
        self,
        live_trades: pd.DataFrame,
        bt_trades: pd.DataFrame,
    ) -> tuple[float, pd.DataFrame]:
        """
        Brinson Selection Effect:
          Selection_i = w_i^L × (R_i^L - R_i^BT)
        where w_i is the weight (share of trades) for symbol i and R is
        avg return.

        Returns
        -------
        (total_selection_effect, per_sym_breakdown)
        """
        live_pnl = self._get_pnl(live_trades)
        bt_pnl = self._get_pnl(bt_trades)
        live_sym = self._get_sym(live_trades)
        bt_sym = self._get_sym(bt_trades)

        # Avg PnL per symbol, live
        live_df = pd.DataFrame({"sym": live_sym, "pnl": live_pnl})
        bt_df = pd.DataFrame({"sym": bt_sym, "pnl": bt_pnl})

        live_stats = live_df.groupby("sym")["pnl"].agg(["mean", "count"]).rename(
            columns={"mean": "live_avg_pnl", "count": "live_count"}
        )
        bt_stats = bt_df.groupby("sym")["pnl"].agg(["mean", "count"]).rename(
            columns={"mean": "bt_avg_pnl", "count": "bt_count"}
        )

        combined = live_stats.join(bt_stats, how="outer").fillna(0)
        total_live = live_df["pnl"].count()

        # Weight = share of live trades in this symbol
        combined["weight"] = combined["live_count"] / max(total_live, 1)
        combined["selection_contrib"] = combined["weight"] * (
            combined["live_avg_pnl"] - combined["bt_avg_pnl"]
        )

        total_sel = float(combined["selection_contrib"].sum())
        combined.index.name = "sym"
        return total_sel, combined.reset_index()[["sym", "live_avg_pnl", "bt_avg_pnl",
                                                   "weight", "selection_contrib"]]

    # ── Timing effect ─────────────────────────────────────────────────────

    def _compute_timing_effect(
        self,
        trades: pd.DataFrame,
    ) -> tuple[float, pd.DataFrame]:
        """
        Timing effect measures how much PnL was missed due to sub-optimal
        entry or exit timing.

        Proxy: compare actual hold_hours to median hold_hours for the same
        symbol×regime group.  Trades with shorter hold < median are assumed
        to have exited early (negative timing contribution on winning trades).
        """
        pnl = self._get_pnl(trades)
        sym = self._get_sym(trades)
        regime = self._get_regime(trades)

        hold_col = self._resolve_col(trades, "hold_hours", "live_hold_hours",
                                     "bt_hold_hours", "hold_bars")
        if hold_col:
            hold = pd.to_numeric(trades[hold_col], errors="coerce")
        else:
            hold = pd.Series(np.nan, index=trades.index)

        df = pd.DataFrame({
            "sym": sym, "regime": regime,
            "pnl": pnl, "hold": hold
        })

        # Compute median hold per sym×regime
        median_hold = df.groupby(["sym", "regime"])["hold"].median()
        df = df.join(median_hold.rename("median_hold"), on=["sym", "regime"])

        # Timing penalty: winning trade exited before median = missed gains
        # Losing trade held beyond median = extra losses
        is_win = df["pnl"] > 0
        hold_diff = df["hold"] - df["median_hold"]  # positive = held longer
        timing_factor = np.where(is_win, -np.sign(hold_diff), np.sign(hold_diff))
        timing_contrib = timing_factor * df["pnl"].abs() * 0.1  # 10% penalty proxy

        df["timing_contrib"] = timing_contrib

        regime_timing = df.groupby("regime")["timing_contrib"].sum().reset_index()
        regime_timing.columns = ["regime", "timing_effect"]

        return float(timing_contrib.sum()), regime_timing

    # ── Sizing effect ─────────────────────────────────────────────────────

    def _compute_sizing_effect(
        self,
        trades: pd.DataFrame,
    ) -> tuple[float, dict[str, float]]:
        """
        Sizing effect: compare realised position size to Kelly-optimal size.

        Kelly fraction f* = (mu / sigma^2) where mu = expected return,
        sigma = return std for that symbol/regime.

        We compute what PnL would have been at full-Kelly vs actual sizing.
        """
        pnl = self._get_pnl(trades)
        sym = self._get_sym(trades)
        dollar_pos = self._get_signal(trades, "dollar_pos")
        ret_pct = self._get_signal(trades, "return_pct")

        df = pd.DataFrame({"sym": sym, "pnl": pnl, "dollar_pos": dollar_pos, "ret_pct": ret_pct})

        # Per-symbol Kelly calculation
        kelly_stats: dict[str, dict] = {}
        sizing_contribs: list[float] = []

        for sym_val, grp in df.groupby("sym"):
            rets = grp["ret_pct"].dropna() / 100.0
            if len(rets) < 3:
                sizing_contribs.extend([0.0] * len(grp))
                continue

            mu = float(rets.mean())
            sigma2 = float(rets.var())
            if sigma2 < 1e-10:
                kelly_f = 0.0
            else:
                kelly_f = mu / sigma2

            # Clip Kelly to [0, 1] (no leverage, long only)
            kelly_f = float(np.clip(kelly_f, 0.0, 1.0))

            # Realised fraction: dollar_pos / assumed_equity
            # We approximate equity as median dollar_pos / 0.1 (10% typical alloc)
            median_pos = float(grp["dollar_pos"].median())
            if median_pos > 0 and np.isfinite(median_pos):
                assumed_equity = median_pos / 0.10
                realised_f = float(grp["dollar_pos"].median() / assumed_equity)
            else:
                realised_f = 0.10  # assume 10% default allocation

            kelly_stats[str(sym_val)] = {
                "kelly_f": kelly_f,
                "realised_f": realised_f,
                "f_ratio": realised_f / max(kelly_f, 1e-6),
                "mu": mu,
                "sigma": float(np.sqrt(max(sigma2, 0))),
            }

            # Sizing contribution = (realised_f / kelly_f - 1) * total_pnl
            total_sym_pnl = float(grp["pnl"].sum())
            if kelly_f > 1e-6:
                sizing_contrib = (realised_f / kelly_f - 1.0) * total_sym_pnl * 0.25
            else:
                sizing_contrib = 0.0
            sizing_contribs.extend([sizing_contrib / max(len(grp), 1)] * len(grp))

        total_sizing = float(sum(sizing_contribs))

        summary = {
            "total_sizing_effect": total_sizing,
            "n_syms_analysed": len(kelly_stats),
            "avg_kelly_f": float(np.mean([v["kelly_f"] for v in kelly_stats.values()])) if kelly_stats else np.nan,
            "avg_realised_f": float(np.mean([v["realised_f"] for v in kelly_stats.values()])) if kelly_stats else np.nan,
            "avg_f_ratio": float(np.mean([v["f_ratio"] for v in kelly_stats.values()])) if kelly_stats else np.nan,
        }
        return total_sizing, summary

    # ── Regime effect ─────────────────────────────────────────────────────

    def _compute_regime_effect(
        self,
        trades: pd.DataFrame,
    ) -> tuple[float, pd.DataFrame]:
        """
        Regime effect: fraction of PnL variation attributable to the
        prevailing market regime.

        Uses an ANOVA-style decomposition:
          Regime effect = SS_between / SS_total

        The total regime effect in dollar terms:
          = sum_r (count_r × (avg_pnl_r - grand_avg_pnl))
        """
        pnl = self._get_pnl(trades)
        regime = self._get_regime(trades)

        df = pd.DataFrame({"regime": regime, "pnl": pnl}).dropna(subset=["pnl"])
        if df.empty:
            return 0.0, pd.DataFrame()

        grand_mean = float(df["pnl"].mean())

        rows = []
        for reg, grp in df.groupby("regime"):
            n = len(grp)
            avg = float(grp["pnl"].mean())
            total = float(grp["pnl"].sum())
            regime_contrib = n * (avg - grand_mean)
            rows.append({
                "regime": str(reg),
                "n_trades": n,
                "avg_pnl": avg,
                "total_pnl": total,
                "regime_effect_contrib": regime_contrib,
            })

        regime_df = pd.DataFrame(rows)

        total_effect = float(regime_df["regime_effect_contrib"].sum()) if not regime_df.empty else 0.0

        # Compute R² of regime as a predictor
        if len(df) > 1:
            regime_encoded = regime.map(
                {"BULL": 1, "BEAR": -1, "SIDEWAYS": 0, "HIGH_VOL": 2, "UNKNOWN": 0}
            ).fillna(0)
            valid = df.index
            r_enc = regime_encoded.loc[valid]
            p_vals = pnl.loc[valid]
            if r_enc.std() > 0 and p_vals.std() > 0:
                corr, _ = stats.pearsonr(r_enc, p_vals)
                regime_df.attrs["r_squared"] = corr ** 2
            else:
                regime_df.attrs["r_squared"] = 0.0
        else:
            regime_df.attrs["r_squared"] = 0.0

        return total_effect, regime_df

    # ── Signal effect ─────────────────────────────────────────────────────

    def _compute_signal_effect(
        self,
        trades: pd.DataFrame,
    ) -> tuple[float, pd.DataFrame]:
        """
        Signal effect: PnL explained by BH physics signals and ensemble.

        Performs OLS regression of PnL on [tf_score, mass, atr, delta_score,
        ensemble_signal] and attributes the R²-weighted portion of PnL to
        the signal effect.

        Returns
        -------
        (signal_effect_dollar, factor_df)
        """
        pnl = self._get_pnl(trades)
        factors = {
            "tf_score": self._get_signal(trades, "tf_score"),
            "mass": self._get_signal(trades, "mass"),
            "atr": self._get_signal(trades, "atr"),
            "delta_score": self._get_signal(trades, "delta_score"),
            "ensemble_signal": self._get_signal(trades, "ensemble_signal"),
        }

        # Build factor matrix, keeping only factors with enough data
        factor_df_raw = pd.DataFrame(factors)
        factor_df_raw["pnl"] = pnl

        # Drop columns with >50% NaN
        usable = [c for c in factor_df_raw.columns
                  if factor_df_raw[c].notna().sum() > len(factor_df_raw) * 0.5]
        usable = [c for c in usable if c != "pnl"]

        clean = factor_df_raw[usable + ["pnl"]].dropna()
        n_obs = len(clean)

        if n_obs < len(usable) + 2 or not usable:
            empty_factors = pd.DataFrame(
                columns=["factor", "beta", "t_stat", "p_value",
                         "r2_contribution", "pnl_attribution"]
            )
            return 0.0, empty_factors

        X = clean[usable].values
        y = clean["pnl"].values

        # Standardise X for interpretable betas
        X_std = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))

        # OLS: beta = (X'X)^{-1} X'y
        try:
            XtX = X_std.T @ X_std
            Xty = X_std.T @ y
            beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(len(usable))

        y_hat = X_std @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Per-factor contribution: beta_i * corr(X_i, y) * r2 fraction
        rows = []
        total_pnl = float(y.sum())
        for i, factor in enumerate(usable):
            xi = X_std[:, i]
            corr_i = float(np.corrcoef(xi, y)[0, 1]) if xi.std() > 0 else 0.0

            # t-statistic
            if n_obs > len(usable) + 1 and ss_res > 0:
                mse = ss_res / (n_obs - len(usable) - 1)
                try:
                    var_beta = mse * np.linalg.inv(XtX)[i, i]
                    se = np.sqrt(max(var_beta, 0))
                    t_stat = beta[i] / se if se > 0 else 0.0
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_obs - len(usable) - 1))
                except (np.linalg.LinAlgError, ValueError):
                    t_stat = 0.0
                    p_val = 1.0
            else:
                t_stat = 0.0
                p_val = 1.0

            # R² contribution via semi-partial correlation
            r2_contrib = corr_i * beta[i] / max(r2, 1e-8) * r2 if r2 > 0 else 0.0
            pnl_attr = r2_contrib * total_pnl

            rows.append({
                "factor": factor,
                "beta": float(beta[i]),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "r2_contribution": float(r2_contrib),
                "pnl_attribution": float(pnl_attr),
            })

        factor_result_df = pd.DataFrame(rows)
        signal_effect = float(r2 * total_pnl)
        factor_result_df.attrs["r2"] = r2

        return signal_effect, factor_result_df

    # ── Factor decomposition ───────────────────────────────────────────────

    def factor_decompose(
        self,
        trades: pd.DataFrame,
        factors: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Decompose PnL across a custom list of factor columns via OLS.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade DataFrame.
        factors : list[str] | None
            Column names to use as factors.  Defaults to all numeric columns
            except pnl / return_pct.

        Returns
        -------
        pd.DataFrame
            factor → beta, t_stat, p_value, r2_contribution, pnl_attribution
        """
        if factors is None:
            exclude = {"pnl", "return_pct", "dollar_return_pct", "live_pnl", "bt_pnl"}
            factors = [
                c for c in trades.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]

        if not factors:
            return pd.DataFrame(columns=["factor", "beta", "t_stat", "p_value",
                                         "r2_contribution", "pnl_attribution"])

        pnl = self._get_pnl(trades)
        result_rows = []
        for factor in factors:
            if factor not in trades.columns:
                continue
            xi = pd.to_numeric(trades[factor], errors="coerce")
            valid = xi.notna() & pnl.notna()
            if valid.sum() < 5:
                continue
            x_v = xi[valid].values
            y_v = pnl[valid].values
            if x_v.std() < 1e-10:
                continue
            slope, intercept, r, p_val, se = stats.linregress(x_v, y_v)
            t_stat = slope / se if se > 0 else 0.0
            r2_c = r ** 2
            pnl_attr = r2_c * float(y_v.sum())
            result_rows.append({
                "factor": factor,
                "beta": float(slope),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "r2_contribution": float(r2_c),
                "pnl_attribution": float(pnl_attr),
            })

        return pd.DataFrame(result_rows).sort_values("r2_contribution", ascending=False)

    # ── IC analysis ───────────────────────────────────────────────────────

    def compute_information_coefficient(
        self,
        signals: pd.Series | np.ndarray,
        forward_returns: pd.Series | np.ndarray,
    ) -> pd.Series:
        """
        Compute the Information Coefficient (IC) as the cross-sectional
        Spearman rank correlation between signals at time t and forward
        returns at time t+1.

        Parameters
        ----------
        signals : array-like or pd.Series
            Signal values (e.g., delta_score per trade).
        forward_returns : array-like or pd.Series
            Next-period returns for the same instruments.

        Returns
        -------
        pd.Series
            IC values (length = len(signals) - 1 if not already aligned).
        """
        s = pd.to_numeric(pd.Series(signals), errors="coerce")
        r = pd.to_numeric(pd.Series(forward_returns), errors="coerce")

        # Align lengths
        min_len = min(len(s), len(r))
        s = s.iloc[:min_len]
        r = r.iloc[:min_len]

        valid = s.notna() & r.notna()
        if valid.sum() < self.ic_min_obs:
            return pd.Series(dtype=float, name="ic")

        # Rank both series
        s_rank = s[valid].rank()
        r_rank = r[valid].rank()
        n = valid.sum()

        # Spearman IC
        ic_val, _ = stats.spearmanr(s_rank, r_rank)
        return pd.Series([ic_val], name="ic")

    def rolling_ic(
        self,
        signals: pd.Series | pd.DataFrame,
        forward_returns: pd.Series | pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Compute rolling IC over a sliding window.

        Parameters
        ----------
        signals : pd.Series or pd.DataFrame
            Signal values indexed by time or trade index.
        forward_returns : pd.Series or pd.DataFrame
            Forward returns for the same observations.
        window : int
            Rolling window size.

        Returns
        -------
        pd.DataFrame
            Columns: ic, ic_mean, ic_std, ir (rolling ICIR)
        """
        if isinstance(signals, pd.DataFrame):
            sig = pd.to_numeric(signals.iloc[:, 0], errors="coerce")
        else:
            sig = pd.to_numeric(signals, errors="coerce")

        if isinstance(forward_returns, pd.DataFrame):
            ret = pd.to_numeric(forward_returns.iloc[:, 0], errors="coerce")
        else:
            ret = pd.to_numeric(forward_returns, errors="coerce")

        min_len = min(len(sig), len(ret))
        sig = sig.reset_index(drop=True).iloc[:min_len]
        ret = ret.reset_index(drop=True).iloc[:min_len]

        ic_vals: list[float] = []
        for i in range(min_len):
            start = max(0, i - window + 1)
            s_w = sig.iloc[start: i + 1]
            r_w = ret.iloc[start: i + 1]
            valid = s_w.notna() & r_w.notna()
            if valid.sum() < self.ic_min_obs:
                ic_vals.append(np.nan)
                continue
            s_v = s_w[valid].values
            r_v = r_w[valid].values
            if s_v.std() < 1e-10 or r_v.std() < 1e-10:
                ic_vals.append(0.0)
                continue
            ic, _ = stats.spearmanr(s_v, r_v)
            ic_vals.append(float(ic))

        ic_series = pd.Series(ic_vals, name="ic")
        roll_ic = ic_series.rolling(window, min_periods=max(2, window // 4))

        out = pd.DataFrame({
            "ic": ic_series,
            "ic_mean": roll_ic.mean(),
            "ic_std": roll_ic.std(),
        })
        out["ir"] = out["ic_mean"] / out["ic_std"].replace(0, np.nan)
        return out

    def ic_information_ratio(self, ic_series: pd.Series) -> float:
        """
        Compute the IC Information Ratio (ICIR = mean(IC) / std(IC)).

        ICIR > 0.5 indicates a predictive signal; > 1.0 is considered strong.

        Parameters
        ----------
        ic_series : pd.Series
            Time series of IC values.

        Returns
        -------
        float
        """
        clean = pd.to_numeric(ic_series, errors="coerce").dropna()
        if len(clean) < 2:
            return float("nan")
        mean_ic = float(clean.mean())
        std_ic = float(clean.std())
        if std_ic < 1e-10:
            return float("nan")
        return mean_ic / std_ic

    # ── Main attribution entry point ──────────────────────────────────────

    def attribute_pnl(
        self,
        trades: pd.DataFrame,
        benchmark_trades: Optional[pd.DataFrame] = None,
    ) -> AttributionReport:
        """
        Full PnL attribution decomposition.

        Parameters
        ----------
        trades : pd.DataFrame
            Primary trade DataFrame (live trades, or merged live+bt).
        benchmark_trades : pd.DataFrame | None
            Backtest trades used as the benchmark universe for selection
            effect.  If None, selection effect is set to zero.

        Returns
        -------
        AttributionReport
        """
        if trades.empty:
            empty_df = pd.DataFrame()
            empty_series = pd.Series(dtype=float)
            return AttributionReport(
                n_trades=0, total_pnl=0.0,
                selection_effect=0.0, timing_effect=0.0, sizing_effect=0.0,
                regime_effect=0.0, signal_effect=0.0, interaction_effect=0.0,
                unexplained=0.0, selection_by_sym=empty_df,
                timing_by_regime=empty_df, sizing_stats={},
                regime_attribution=empty_df, signal_attribution=empty_df,
                ic_series=empty_series, icir=float("nan"),
                rolling_ic=empty_df,
            )

        pnl = self._get_pnl(trades)
        total_pnl = float(pnl.sum())
        n_trades = len(trades)

        # Selection effect (requires benchmark)
        if benchmark_trades is not None and not benchmark_trades.empty:
            sel_effect, sel_by_sym = self._compute_selection_effect(trades, benchmark_trades)
        else:
            sel_effect = 0.0
            sel_by_sym = pd.DataFrame()

        # Timing effect
        timing_effect, timing_by_regime = self._compute_timing_effect(trades)

        # Sizing effect
        sizing_effect, sizing_stats = self._compute_sizing_effect(trades)

        # Regime effect
        regime_effect, regime_df = self._compute_regime_effect(trades)

        # Signal effect
        signal_effect, signal_df = self._compute_signal_effect(trades)

        # Interaction / residual
        sum_effects = sel_effect + timing_effect + sizing_effect + regime_effect + signal_effect
        unexplained = total_pnl - sum_effects
        interaction_effect = unexplained * 0.0  # set to zero; unexplained absorbs residual

        # IC analysis
        signal_col = self._resolve_col(trades, "delta_score", "tf_score", "ensemble_signal")
        pnl_col = self._resolve_col(trades, "pnl", "live_pnl", "return_pct")

        ic_series = pd.Series(dtype=float, name="ic")
        icir = float("nan")
        rolling_ic_df = pd.DataFrame()

        if signal_col and pnl_col:
            sig_vals = pd.to_numeric(trades[signal_col], errors="coerce")
            pnl_vals = pd.to_numeric(trades[pnl_col], errors="coerce")

            # Shift pnl by 1 to get forward returns
            fwd_ret = pnl_vals.shift(-1)
            valid = sig_vals.notna() & fwd_ret.notna()

            if valid.sum() >= self.ic_min_obs:
                ic_result = self.compute_information_coefficient(
                    sig_vals[valid], fwd_ret[valid]
                )
                ic_series = ic_result

                rolling_ic_df = self.rolling_ic(
                    sig_vals, fwd_ret, window=min(60, max(len(trades) // 4, 10))
                )
                if len(rolling_ic_df) > 0:
                    icir = self.ic_information_ratio(rolling_ic_df["ic"])

        return AttributionReport(
            n_trades=n_trades,
            total_pnl=total_pnl,
            selection_effect=sel_effect,
            timing_effect=timing_effect,
            sizing_effect=sizing_effect,
            regime_effect=regime_effect,
            signal_effect=signal_effect,
            interaction_effect=interaction_effect,
            unexplained=unexplained,
            selection_by_sym=sel_by_sym,
            timing_by_regime=timing_by_regime,
            sizing_stats=sizing_stats,
            regime_attribution=regime_df,
            signal_attribution=signal_df,
            ic_series=ic_series,
            icir=icir,
            rolling_ic=rolling_ic_df,
            metadata={
                "signal_col_used": signal_col,
                "pnl_col_used": pnl_col,
                "n_factors_in_signal_model": len(signal_df) if not signal_df.empty else 0,
            },
        )

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot_attribution_waterfall(
        self,
        report: AttributionReport,
        save_path: str | Path,
        dpi: int = 150,
    ) -> Path:
        """
        Waterfall chart of PnL attribution effects.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        effects = report.effect_table()
        labels = ["Total"] + list(effects.index)
        values = [report.total_pnl] + list(effects["Value"])

        colors = []
        for v in values:
            if v >= 0:
                colors.append("#4CAF50")
            else:
                colors.append("#F44336")
        colors[0] = "#2196F3"  # Total bar in blue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Waterfall
        bar_bottoms = [0.0]
        running = 0.0
        for v in values[1:]:
            bar_bottoms.append(running)
            running += v

        bars = ax1.bar(labels, values, bottom=bar_bottoms, color=colors, alpha=0.85, edgecolor="white")
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.bar_label(bars, fmt="%.0f", padding=3, fontsize=7)
        ax1.set_title("PnL Attribution Waterfall")
        ax1.set_ylabel("PnL ($)")
        ax1.tick_params(axis="x", rotation=45)

        # Pie chart of absolute contributions
        abs_vals = effects["Value"].abs()
        abs_vals = abs_vals[abs_vals > 0]
        if len(abs_vals) > 0:
            ax2.pie(abs_vals, labels=abs_vals.index, autopct="%1.1f%%",
                    startangle=90, colors=plt.cm.Set3.colors[:len(abs_vals)])
            ax2.set_title("Attribution Share (Absolute)")

        fig.suptitle(
            f"PnL Attribution  |  Total: ${report.total_pnl:,.0f}  |  N={report.n_trades}",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Attribution waterfall saved to %s", save_path)
        return save_path

    def plot_rolling_ic(
        self,
        report: AttributionReport,
        save_path: str | Path,
        dpi: int = 150,
    ) -> Path:
        """
        Plot rolling IC and IC Information Ratio over time.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        ric = report.rolling_ic

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        if not ric.empty and "ic" in ric.columns:
            x = range(len(ric))
            axes[0].plot(x, ric["ic"], color="steelblue", linewidth=1.0, alpha=0.7, label="IC")
            if "ic_mean" in ric.columns:
                axes[0].plot(x, ric["ic_mean"], color="orange", linewidth=1.5, label="Rolling Mean IC")
            axes[0].axhline(0, color="black", linewidth=0.8)
            axes[0].axhline(0.05, color="green", linestyle="--", linewidth=0.8, label="IC=0.05")
            axes[0].axhline(-0.05, color="red", linestyle="--", linewidth=0.8, label="IC=-0.05")
            axes[0].set_ylabel("IC")
            axes[0].set_title(f"Rolling IC  (ICIR={report.icir:.3f})")
            axes[0].legend(fontsize=8)
            axes[0].fill_between(x, ric["ic"].fillna(0), 0, alpha=0.1, color="steelblue")

            if "ir" in ric.columns:
                axes[1].plot(x, ric["ir"], color="purple", linewidth=1.2)
                axes[1].axhline(0, color="black", linewidth=0.8)
                axes[1].axhline(0.5, color="green", linestyle="--", linewidth=0.8, label="ICIR=0.5")
                axes[1].axhline(-0.5, color="red", linestyle="--", linewidth=0.8)
                axes[1].set_ylabel("ICIR")
                axes[1].set_xlabel("Trade #")
                axes[1].set_title("Rolling IC Information Ratio")
                axes[1].legend(fontsize=8)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "No IC data available", ha="center", va="center")

        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_regime_attribution(
        self,
        report: AttributionReport,
        save_path: str | Path,
        dpi: int = 150,
    ) -> Path:
        """
        Bar chart of PnL attribution by regime.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        RA = report.regime_attribution

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if not RA.empty and "regime" in RA.columns:
            regime_colors = {
                "BULL": "#4CAF50", "BEAR": "#F44336",
                "SIDEWAYS": "#9E9E9E", "HIGH_VOL": "#FF9800",
                "UNKNOWN": "#607D8B",
            }
            colors = [regime_colors.get(str(r), "#607D8B") for r in RA["regime"]]

            # Total PnL by regime
            bars = axes[0].bar(RA["regime"], RA["total_pnl"], color=colors, alpha=0.85)
            axes[0].bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
            axes[0].axhline(0, color="black", linewidth=0.8)
            axes[0].set_title("Total PnL by Regime")
            axes[0].set_ylabel("PnL ($)")
            axes[0].tick_params(axis="x", rotation=30)

            # Trade count by regime
            bars2 = axes[1].bar(RA["regime"], RA["n_trades"], color=colors, alpha=0.85)
            axes[1].bar_label(bars2, padding=3, fontsize=8)
            axes[1].set_title("Trade Count by Regime")
            axes[1].set_ylabel("# Trades")
            axes[1].tick_params(axis="x", rotation=30)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "No regime data", ha="center", va="center")

        fig.suptitle("Regime Attribution", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
