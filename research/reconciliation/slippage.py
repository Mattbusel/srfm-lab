"""
research/reconciliation/slippage.py
=====================================
Slippage and market-impact analysis for the live-vs-backtest reconciliation
pipeline.

The backtest fills at exact OHLCV prices; the live trader fills at the
next-bar open (or market order execution price).  The difference between the
two constitutes *slippage*.  This module decomposes total slippage into:

  1. **Spread component** – half the bid-ask spread paid on entry and exit.
  2. **Timing component** – price movement between the signal bar close and
     the actual fill (latency + queue position).
  3. **Market-impact component** – price impact proportional to order size
     relative to Average Daily Volume (ADV), modelled via an Almgren-Chriss
     square-root law.

Classes
-------
SlippageAnalyzer
    Main workhorse: accepts a merged live+backtest DataFrame and produces
    FillReport and SlippageStats objects.

Dataclasses
-----------
FillReport      – aggregate fill-quality statistics
SlippageStats   – per-stratum slippage statistics
ImpactEstimate  – output of the market-impact model
"""

from __future__ import annotations

import logging
import math
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
class SlippageStats:
    """
    Slippage statistics for a particular stratum (regime, symbol,
    time-of-day bucket, etc.).
    """
    stratum: str
    n_trades: int
    mean_bps: float          # mean slippage in basis points
    median_bps: float
    p95_bps: float
    p99_bps: float
    std_bps: float
    total_cost_usd: float    # cumulative dollar cost from slippage
    entry_mean_bps: float    # entry-side only
    exit_mean_bps: float     # exit-side only
    spread_component_bps: float
    timing_component_bps: float
    impact_component_bps: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class FillReport:
    """
    Aggregate fill-quality report across all trades in a dataset.
    """
    n_trades: int
    overall: SlippageStats
    by_regime: dict[str, SlippageStats] = field(default_factory=dict)
    by_sym: dict[str, SlippageStats] = field(default_factory=dict)
    by_hour: dict[int, SlippageStats] = field(default_factory=dict)
    worst_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        """Return a summary table of slippage by regime."""
        rows = [v.to_dict() for v in self.by_regime.values()]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("stratum")


@dataclass
class ImpactEstimate:
    """Output of ``estimate_market_impact``."""
    dollar_size: float
    adv: float
    sigma: float           # daily volatility (decimal)
    participation_rate: float
    linear_impact_bps: float
    sqrt_impact_bps: float
    total_impact_bps: float
    total_impact_usd: float


# ── SlippageAnalyzer ──────────────────────────────────────────────────────────


class SlippageAnalyzer:
    """
    Analyse fill quality and slippage between live execution and backtest
    ideal fills.

    Parameters
    ----------
    spread_bps : float
        Assumed half-spread in basis points per side (default 5 bps).
        Used when actual bid-ask data is unavailable.
    impact_coeff : float
        Almgren-Chriss square-root impact coefficient η (default 0.1).
        Impact ≈ η × σ × √(X / ADV) in bps.
    linear_coeff : float
        Linear market-impact coefficient (default 0.01).
        Linear impact ≈ γ × (X / ADV) × σ in bps.
    min_adv_usd : float
        Minimum ADV assumed when ADV is unknown (default 1,000,000 USD).
    """

    # Typical ADV by asset class (USD) – fallback when not provided
    _DEFAULT_ADV: dict[str, float] = {
        "BTC": 20_000_000_000.0,
        "ETH": 8_000_000_000.0,
        "BNB": 1_000_000_000.0,
        "SOL": 2_000_000_000.0,
        "XRP": 2_500_000_000.0,
        "ADA": 600_000_000.0,
        "DOGE": 800_000_000.0,
        "DOT": 300_000_000.0,
        "LINK": 400_000_000.0,
        "UNI": 150_000_000.0,
        "SUSHI": 50_000_000.0,
        "DEFAULT": 200_000_000.0,
    }

    def __init__(
        self,
        spread_bps: float = 5.0,
        impact_coeff: float = 0.1,
        linear_coeff: float = 0.01,
        min_adv_usd: float = 1_000_000.0,
    ) -> None:
        self.spread_bps = spread_bps
        self.impact_coeff = impact_coeff
        self.linear_coeff = linear_coeff
        self.min_adv_usd = min_adv_usd

    # ── internal helpers ──────────────────────────────────────────────────

    def _get_adv(self, sym: str) -> float:
        base = sym.replace("USDT", "").replace("USD", "").replace("/", "")
        return self._DEFAULT_ADV.get(base, self._DEFAULT_ADV["DEFAULT"])

    def _compute_slippage_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a merged DataFrame (live_ and bt_ prefixed columns), compute
        per-trade slippage in basis points.

        Slippage definition
        -------------------
        entry_slippage_bps = (live_entry - bt_entry) / bt_entry * 10_000
          (positive = paid more than backtest, i.e. worse fill)
        exit_slippage_bps  = (bt_exit - live_exit) / bt_exit * 10_000
          (positive = received less than backtest, i.e. worse fill)
        total_slippage_bps = entry_slippage_bps + exit_slippage_bps

        When live entry/exit prices are unavailable (no match), we infer
        slippage from the PnL difference scaled by position size.
        """
        out = df.copy()

        # --- Column resolution -------------------------------------------------
        # Prefer live_ prefixed columns from merged frame; fall back to raw cols
        def _col(name: str, prefix: str = "live") -> pd.Series:
            prefixed = f"{prefix}_{name}"
            if prefixed in out.columns:
                return pd.to_numeric(out[prefixed], errors="coerce")
            if name in out.columns:
                return pd.to_numeric(out[name], errors="coerce")
            return pd.Series(np.nan, index=out.index)

        live_entry = _col("entry_price", "live")
        live_exit  = _col("exit_price", "live")
        bt_entry   = _col("entry_price", "bt")
        bt_exit    = _col("exit_price", "bt")
        live_pnl   = _col("pnl", "live")
        bt_pnl     = _col("pnl", "bt")
        dollar_pos = _col("dollar_pos", "live").fillna(_col("dollar_pos", "bt"))

        # --- Entry slippage ---------------------------------------------------
        valid_entry = bt_entry != 0
        out["entry_slippage_bps"] = np.where(
            valid_entry & bt_entry.notna() & live_entry.notna(),
            (live_entry - bt_entry) / bt_entry.replace(0, np.nan) * 10_000,
            np.nan,
        )

        # --- Exit slippage ----------------------------------------------------
        valid_exit = bt_exit != 0
        out["exit_slippage_bps"] = np.where(
            valid_exit & bt_exit.notna() & live_exit.notna(),
            (bt_exit - live_exit) / bt_exit.replace(0, np.nan) * 10_000,
            np.nan,
        )

        # --- Fallback: use PnL difference ------------------------------------
        pnl_diff = live_pnl - bt_pnl
        pnl_based_slip = np.where(
            dollar_pos != 0,
            -pnl_diff / dollar_pos.replace(0, np.nan) * 10_000,
            np.nan,
        )

        out["entry_slippage_bps"] = out["entry_slippage_bps"].fillna(
            pd.Series(pnl_based_slip / 2, index=out.index)
        )
        out["exit_slippage_bps"] = out["exit_slippage_bps"].fillna(
            pd.Series(pnl_based_slip / 2, index=out.index)
        )

        # --- Total slippage ---------------------------------------------------
        out["total_slippage_bps"] = (
            out["entry_slippage_bps"].fillna(0) + out["exit_slippage_bps"].fillna(0)
        )

        # --- Dollar cost of slippage -----------------------------------------
        out["slippage_cost_usd"] = out["total_slippage_bps"] / 10_000 * dollar_pos.fillna(0)

        # --- Decomposition ---------------------------------------------------
        out["spread_component_bps"] = self.spread_bps * 2  # entry + exit side

        sym_col = self._resolve_sym(out)
        sigma_col = self._resolve_sigma(out)

        impact_bps = []
        timing_bps = []
        for idx, row in out.iterrows():
            sym = str(row.get(sym_col, "UNKNOWN") or "UNKNOWN")
            adv = self._get_adv(sym)
            dpos = float(dollar_pos.at[idx]) if not pd.isna(dollar_pos.at[idx]) else 0.0
            sigma = float(sigma_col.at[idx]) if not pd.isna(sigma_col.at[idx]) else 0.02

            imp = self._impact_bps(dpos, adv, sigma)
            total_slip = float(out.at[idx, "total_slippage_bps"])
            spread = self.spread_bps * 2

            timing = total_slip - spread - imp
            impact_bps.append(imp)
            timing_bps.append(timing)

        out["impact_component_bps"] = impact_bps
        out["timing_component_bps"] = timing_bps

        return out

    def _resolve_sym(self, df: pd.DataFrame) -> str:
        for candidate in ("live_sym", "bt_sym", "sym"):
            if candidate in df.columns:
                return candidate
        return "sym"

    def _resolve_sigma(self, df: pd.DataFrame) -> pd.Series:
        for candidate in ("live_atr", "bt_atr", "atr"):
            if candidate in df.columns:
                col = pd.to_numeric(df[candidate], errors="coerce")
                # ATR as fraction of price → approximate daily vol
                entry_col = "live_entry_price" if "live_entry_price" in df.columns else "entry_price"
                if entry_col in df.columns:
                    price = pd.to_numeric(df[entry_col], errors="coerce")
                    return (col / price.replace(0, np.nan)).fillna(0.02)
                return col.fillna(0.02)
        return pd.Series(0.02, index=df.index)

    def _impact_bps(self, dollar_size: float, adv: float, sigma: float) -> float:
        """
        Almgren-Chriss market impact in basis points.
        Total = linear + sqrt components.
        """
        adv = max(adv, self.min_adv_usd)
        sigma = max(sigma, 1e-6)
        part_rate = dollar_size / adv
        linear = self.linear_coeff * part_rate * sigma * 10_000
        sqrt = self.impact_coeff * sigma * math.sqrt(part_rate) * 10_000
        return linear + sqrt

    def _build_stats(
        self,
        df: pd.DataFrame,
        stratum: str,
    ) -> SlippageStats:
        """Construct a SlippageStats from a slippage-annotated DataFrame."""
        slip = pd.to_numeric(df["total_slippage_bps"], errors="coerce").dropna()
        entry_slip = pd.to_numeric(df["entry_slippage_bps"], errors="coerce").dropna()
        exit_slip = pd.to_numeric(df["exit_slippage_bps"], errors="coerce").dropna()
        cost = pd.to_numeric(df["slippage_cost_usd"], errors="coerce").dropna()
        spread = pd.to_numeric(df.get("spread_component_bps", pd.Series(dtype=float)), errors="coerce").dropna()
        timing = pd.to_numeric(df.get("timing_component_bps", pd.Series(dtype=float)), errors="coerce").dropna()
        impact = pd.to_numeric(df.get("impact_component_bps", pd.Series(dtype=float)), errors="coerce").dropna()

        n = len(slip)
        if n == 0:
            return SlippageStats(
                stratum=stratum, n_trades=0,
                mean_bps=np.nan, median_bps=np.nan,
                p95_bps=np.nan, p99_bps=np.nan,
                std_bps=np.nan, total_cost_usd=np.nan,
                entry_mean_bps=np.nan, exit_mean_bps=np.nan,
                spread_component_bps=np.nan,
                timing_component_bps=np.nan,
                impact_component_bps=np.nan,
            )

        return SlippageStats(
            stratum=stratum,
            n_trades=n,
            mean_bps=float(slip.mean()),
            median_bps=float(slip.median()),
            p95_bps=float(np.percentile(slip, 95)),
            p99_bps=float(np.percentile(slip, 99)),
            std_bps=float(slip.std()),
            total_cost_usd=float(cost.sum()),
            entry_mean_bps=float(entry_slip.mean()) if len(entry_slip) > 0 else np.nan,
            exit_mean_bps=float(exit_slip.mean()) if len(exit_slip) > 0 else np.nan,
            spread_component_bps=float(spread.mean()) if len(spread) > 0 else np.nan,
            timing_component_bps=float(timing.mean()) if len(timing) > 0 else np.nan,
            impact_component_bps=float(impact.mean()) if len(impact) > 0 else np.nan,
        )

    # ── public API ────────────────────────────────────────────────────────

    def analyze_fill_quality(self, trades: pd.DataFrame) -> FillReport:
        """
        Compute a comprehensive fill-quality report.

        Parameters
        ----------
        trades : pd.DataFrame
            Either the merged live-vs-backtest DataFrame (with live_ / bt_
            prefixed columns) or a raw live trade DataFrame with
            entry_price, exit_price, pnl, dollar_pos columns.

        Returns
        -------
        FillReport
        """
        if trades.empty:
            return FillReport(
                n_trades=0,
                overall=self._build_stats(trades, "ALL"),
            )

        annotated = self._compute_slippage_series(trades)

        overall = self._build_stats(annotated, "ALL")

        # --- By regime -------------------------------------------------------
        by_regime: dict[str, SlippageStats] = {}
        regime_col = self._resolve_regime(annotated)
        if regime_col:
            for regime, grp in annotated.groupby(regime_col):
                by_regime[str(regime)] = self._build_stats(grp, str(regime))

        # --- By symbol -------------------------------------------------------
        by_sym: dict[str, SlippageStats] = {}
        sym_col = self._resolve_sym(annotated)
        if sym_col in annotated.columns:
            for sym, grp in annotated.groupby(sym_col):
                by_sym[str(sym)] = self._build_stats(grp, str(sym))

        # --- By hour of day --------------------------------------------------
        by_hour: dict[int, SlippageStats] = {}
        exit_col = self._resolve_exit_time(annotated)
        if exit_col and exit_col in annotated.columns:
            hours = pd.to_datetime(annotated[exit_col], utc=True, errors="coerce").dt.hour
            annotated["_hour"] = hours
            for hr, grp in annotated.groupby("_hour"):
                by_hour[int(hr)] = self._build_stats(grp, f"hour_{int(hr):02d}")
            annotated.drop(columns=["_hour"], inplace=True, errors="ignore")

        # --- Worst / best trades ---------------------------------------------
        slip_col = "total_slippage_bps"
        sort_cols = [c for c in [sym_col, slip_col, "slippage_cost_usd"] if c in annotated.columns]
        if slip_col in annotated.columns:
            sorted_slip = annotated.sort_values(slip_col, ascending=False, na_position="last")
            worst_trades = sorted_slip.head(20)[sort_cols] if sort_cols else pd.DataFrame()
            best_trades = sorted_slip.tail(20)[sort_cols] if sort_cols else pd.DataFrame()
        else:
            worst_trades = pd.DataFrame()
            best_trades = pd.DataFrame()

        return FillReport(
            n_trades=len(annotated),
            overall=overall,
            by_regime=by_regime,
            by_sym=by_sym,
            by_hour=by_hour,
            worst_trades=worst_trades,
            best_trades=best_trades,
            metadata={
                "spread_assumption_bps": self.spread_bps,
                "impact_coeff": self.impact_coeff,
                "linear_coeff": self.linear_coeff,
            },
        )

    def _resolve_regime(self, df: pd.DataFrame) -> Optional[str]:
        for candidate in ("live_regime", "bt_regime", "regime"):
            if candidate in df.columns:
                return candidate
        return None

    def _resolve_exit_time(self, df: pd.DataFrame) -> Optional[str]:
        for candidate in ("live_exit_time", "bt_exit_time", "exit_time"):
            if candidate in df.columns:
                return candidate
        return None

    def estimate_market_impact(
        self,
        dollar_size: float,
        adv: float,
        sigma: float,
        sym: str = "UNKNOWN",
    ) -> ImpactEstimate:
        """
        Estimate market impact for a given order using the Almgren-Chriss
        square-root model.

        Parameters
        ----------
        dollar_size : float
            Order size in USD.
        adv : float
            Average daily volume in USD.
        sigma : float
            Daily volatility (decimal, e.g. 0.03 for 3%).
        sym : str
            Symbol (for ADV lookup if adv=0).

        Returns
        -------
        ImpactEstimate
        """
        if adv <= 0:
            adv = self._get_adv(sym)
        adv = max(adv, self.min_adv_usd)
        sigma = max(sigma, 1e-8)

        part_rate = dollar_size / adv
        linear_bps = self.linear_coeff * part_rate * sigma * 10_000
        sqrt_bps = self.impact_coeff * sigma * math.sqrt(part_rate) * 10_000
        total_bps = linear_bps + sqrt_bps
        total_usd = total_bps / 10_000 * dollar_size

        return ImpactEstimate(
            dollar_size=dollar_size,
            adv=adv,
            sigma=sigma,
            participation_rate=part_rate,
            linear_impact_bps=linear_bps,
            sqrt_impact_bps=sqrt_bps,
            total_impact_bps=total_bps,
            total_impact_usd=total_usd,
        )

    def slippage_by_regime(self, trades: pd.DataFrame) -> dict[str, SlippageStats]:
        """
        Convenience wrapper: compute and return slippage broken down by regime.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade DataFrame (merged or raw live).

        Returns
        -------
        dict[str, SlippageStats]
        """
        report = self.analyze_fill_quality(trades)
        return report.by_regime

    def slippage_decomposition(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame showing the three-way decomposition of slippage
        (spread, timing, market impact) for each trade.

        Returns
        -------
        pd.DataFrame
            Columns: total_slippage_bps, spread_component_bps,
            timing_component_bps, impact_component_bps,
            spread_pct, timing_pct, impact_pct
        """
        if trades.empty:
            return pd.DataFrame()

        annotated = self._compute_slippage_series(trades)
        cols = [
            "total_slippage_bps",
            "spread_component_bps",
            "timing_component_bps",
            "impact_component_bps",
        ]
        available = [c for c in cols if c in annotated.columns]
        out = annotated[available].copy()

        total = out["total_slippage_bps"].replace(0, np.nan)
        for comp in ("spread_component_bps", "timing_component_bps", "impact_component_bps"):
            if comp in out.columns:
                pct_name = comp.replace("_bps", "_pct")
                out[pct_name] = out[comp] / total * 100

        return out

    def compute_turnover_cost(
        self,
        trades: pd.DataFrame,
        annual_factor: float = 252.0,
    ) -> dict[str, float]:
        """
        Estimate annualised turnover cost from slippage.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade DataFrame.
        annual_factor : float
            Trading days per year.

        Returns
        -------
        dict with keys: total_cost_usd, avg_cost_per_trade_bps,
            annualised_drag_bps, annualised_drag_pct
        """
        if trades.empty:
            return {}

        annotated = self._compute_slippage_series(trades)
        total_cost = float(pd.to_numeric(annotated["slippage_cost_usd"], errors="coerce").sum())
        avg_pos = float(
            pd.to_numeric(
                annotated.get("live_dollar_pos", annotated.get("dollar_pos", pd.Series([0]))),
                errors="coerce"
            ).mean()
        )
        mean_slip = float(pd.to_numeric(annotated["total_slippage_bps"], errors="coerce").mean())

        n_trades = len(annotated)
        exit_times = pd.to_datetime(
            annotated.get("live_exit_time", annotated.get("exit_time")),
            utc=True,
            errors="coerce",
        ).dropna()

        if len(exit_times) > 1:
            span_days = (exit_times.max() - exit_times.min()).total_seconds() / 86400
            trades_per_day = n_trades / max(span_days, 1)
        else:
            trades_per_day = 1.0

        annualised_drag_bps = mean_slip * trades_per_day * annual_factor
        annualised_drag_pct = annualised_drag_bps / 100

        return {
            "total_cost_usd": total_cost,
            "n_trades": n_trades,
            "avg_cost_per_trade_bps": mean_slip,
            "annualised_drag_bps": annualised_drag_bps,
            "annualised_drag_pct": annualised_drag_pct,
        }

    def plot_slippage_distribution(
        self,
        trades: pd.DataFrame,
        save_path: str | Path,
        title: str = "Slippage Distribution",
        dpi: int = 150,
    ) -> Path:
        """
        Plot slippage distribution with decomposition breakdown and save
        to disk.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade DataFrame.
        save_path : str | Path
            File path for the output PNG/PDF.
        title : str
            Chart title.
        dpi : int
            Image resolution.

        Returns
        -------
        Path
            Path to the saved figure.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if trades.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            return save_path

        annotated = self._compute_slippage_series(trades)
        slip = pd.to_numeric(annotated["total_slippage_bps"], errors="coerce").dropna()

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 1. Histogram of total slippage
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(slip, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
        ax1.axvline(slip.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={slip.mean():.1f}")
        ax1.axvline(float(np.percentile(slip, 50)), color="orange", linestyle="--", linewidth=1.2, label=f"P50={np.percentile(slip,50):.1f}")
        ax1.axvline(float(np.percentile(slip, 95)), color="purple", linestyle=":", linewidth=1.2, label=f"P95={np.percentile(slip,95):.1f}")
        ax1.set_xlabel("Total Slippage (bps)")
        ax1.set_ylabel("Count")
        ax1.set_title("Total Slippage Distribution")
        ax1.legend(fontsize=7)

        # 2. QQ-plot
        ax2 = fig.add_subplot(gs[0, 1])
        (osm, osr), (slope, intercept, r) = stats.probplot(slip, dist="norm", fit=True)
        ax2.scatter(osm, osr, s=8, alpha=0.5, color="steelblue")
        x_line = np.linspace(min(osm), max(osm), 100)
        ax2.plot(x_line, slope * x_line + intercept, color="red", linewidth=1.5)
        ax2.set_title(f"Q-Q Plot (r={r:.3f})")
        ax2.set_xlabel("Theoretical Quantiles")
        ax2.set_ylabel("Sample Quantiles")

        # 3. Decomposition bar chart
        ax3 = fig.add_subplot(gs[0, 2])
        comps = ["spread", "timing", "impact"]
        comp_cols = ["spread_component_bps", "timing_component_bps", "impact_component_bps"]
        means = []
        for cc in comp_cols:
            if cc in annotated.columns:
                means.append(float(pd.to_numeric(annotated[cc], errors="coerce").mean()))
            else:
                means.append(0.0)
        colors = ["#2196F3", "#FF9800", "#E91E63"]
        bars = ax3.bar(comps, means, color=colors, alpha=0.85)
        ax3.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax3.set_ylabel("Mean (bps)")
        ax3.set_title("Slippage Decomposition")

        # 4. By regime box plot
        ax4 = fig.add_subplot(gs[1, 0])
        regime_col = self._resolve_regime(annotated)
        if regime_col and regime_col in annotated.columns:
            regime_groups = {}
            for reg, grp in annotated.groupby(regime_col):
                s = pd.to_numeric(grp["total_slippage_bps"], errors="coerce").dropna()
                if len(s) > 0:
                    regime_groups[str(reg)] = s.values
            if regime_groups:
                ax4.boxplot(
                    list(regime_groups.values()),
                    labels=list(regime_groups.keys()),
                    patch_artist=True,
                    medianprops={"color": "red"},
                )
                ax4.set_ylabel("Slippage (bps)")
                ax4.set_title("Slippage by Regime")
                ax4.tick_params(axis="x", rotation=30)
        else:
            ax4.text(0.5, 0.5, "No regime data", ha="center", va="center")
            ax4.set_title("Slippage by Regime")

        # 5. Entry vs exit scatter
        ax5 = fig.add_subplot(gs[1, 1])
        entry_slip = pd.to_numeric(annotated.get("entry_slippage_bps", pd.Series(dtype=float)), errors="coerce")
        exit_slip = pd.to_numeric(annotated.get("exit_slippage_bps", pd.Series(dtype=float)), errors="coerce")
        valid_mask = entry_slip.notna() & exit_slip.notna()
        if valid_mask.sum() > 0:
            ax5.scatter(
                entry_slip[valid_mask], exit_slip[valid_mask],
                s=10, alpha=0.4, color="steelblue"
            )
            lims = [
                min(entry_slip[valid_mask].min(), exit_slip[valid_mask].min()),
                max(entry_slip[valid_mask].max(), exit_slip[valid_mask].max()),
            ]
            ax5.plot(lims, lims, "r--", alpha=0.5, linewidth=1)
            ax5.set_xlabel("Entry Slippage (bps)")
            ax5.set_ylabel("Exit Slippage (bps)")
            ax5.set_title("Entry vs Exit Slippage")
        else:
            ax5.text(0.5, 0.5, "No price data", ha="center", va="center")
            ax5.set_title("Entry vs Exit Slippage")

        # 6. Cumulative cost over time
        ax6 = fig.add_subplot(gs[1, 2])
        exit_col = self._resolve_exit_time(annotated)
        if exit_col and exit_col in annotated.columns:
            times = pd.to_datetime(annotated[exit_col], utc=True, errors="coerce")
            cost = pd.to_numeric(annotated["slippage_cost_usd"], errors="coerce").fillna(0)
            sorted_idx = times.argsort()
            cum_cost = cost.iloc[sorted_idx].cumsum().values
            ax6.plot(range(len(cum_cost)), cum_cost, color="steelblue", linewidth=1.5)
            ax6.set_xlabel("Trade #")
            ax6.set_ylabel("Cumulative Slippage Cost (USD)")
            ax6.set_title("Cumulative Slippage Cost")
            ax6.fill_between(range(len(cum_cost)), cum_cost, alpha=0.2, color="steelblue")
        else:
            ax6.text(0.5, 0.5, "No exit time data", ha="center", va="center")
            ax6.set_title("Cumulative Slippage Cost")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Slippage distribution plot saved to %s", save_path)
        return save_path

    def plot_slippage_by_symbol(
        self,
        trades: pd.DataFrame,
        save_path: str | Path,
        top_n: int = 20,
        dpi: int = 150,
    ) -> Path:
        """
        Bar chart of mean slippage per symbol (top N by trade count).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if trades.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            return save_path

        annotated = self._compute_slippage_series(trades)
        sym_col = self._resolve_sym(annotated)

        if sym_col not in annotated.columns:
            log.warning("No symbol column found; cannot plot by symbol.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No symbol column", ha="center", va="center")
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            return save_path

        grp = (
            annotated
            .groupby(sym_col)["total_slippage_bps"]
            .agg(["mean", "count"])
            .nlargest(top_n, "count")
            .sort_values("mean", ascending=False)
        )

        fig, ax = plt.subplots(figsize=(max(8, len(grp) * 0.5), 5))
        bars = ax.bar(grp.index, grp["mean"], color="steelblue", alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Symbol")
        ax.set_ylabel("Mean Slippage (bps)")
        ax.set_title(f"Mean Slippage by Symbol (Top {top_n} by trade count)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def compute_vwap_slippage(
        self,
        trades: pd.DataFrame,
        vwap_col: str = "vwap",
    ) -> pd.Series:
        """
        Compute slippage relative to VWAP when VWAP data is available.

        Parameters
        ----------
        trades : pd.DataFrame
            Must contain a VWAP column.
        vwap_col : str
            Name of the VWAP column.

        Returns
        -------
        pd.Series
            VWAP slippage in bps for each row (NaN if VWAP unavailable).
        """
        if vwap_col not in trades.columns:
            log.warning("No VWAP column '%s' found; returning NaN series.", vwap_col)
            return pd.Series(np.nan, index=trades.index)

        vwap = pd.to_numeric(trades[vwap_col], errors="coerce")
        live_exit = pd.to_numeric(
            trades.get("live_exit_price", trades.get("exit_price")),
            errors="coerce",
        )
        slip = (live_exit - vwap) / vwap.replace(0, np.nan) * 10_000
        return slip

    def rolling_slippage(
        self,
        trades: pd.DataFrame,
        window: int = 30,
    ) -> pd.DataFrame:
        """
        Compute rolling mean slippage over a sliding window of trades.

        Returns
        -------
        pd.DataFrame
            Columns: rolling_mean_bps, rolling_std_bps, rolling_p95_bps
        """
        if trades.empty:
            return pd.DataFrame()

        annotated = self._compute_slippage_series(trades)
        slip = pd.to_numeric(annotated["total_slippage_bps"], errors="coerce")

        roll = slip.rolling(window, min_periods=max(2, window // 4))
        out = pd.DataFrame(
            {
                "rolling_mean_bps": roll.mean(),
                "rolling_std_bps": roll.std(),
                "rolling_p95_bps": roll.quantile(0.95),
            },
            index=annotated.index,
        )

        exit_col = self._resolve_exit_time(annotated)
        if exit_col and exit_col in annotated.columns:
            out["exit_time"] = annotated[exit_col]

        return out

    def regime_impact_matrix(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Build a regime × symbol matrix of mean slippage (bps).

        Returns
        -------
        pd.DataFrame
            Index: regime labels; columns: symbol; values: mean slippage bps.
        """
        if trades.empty:
            return pd.DataFrame()

        annotated = self._compute_slippage_series(trades)
        regime_col = self._resolve_regime(annotated)
        sym_col = self._resolve_sym(annotated)

        if not regime_col or regime_col not in annotated.columns:
            return pd.DataFrame()

        pivot = (
            annotated
            .groupby([regime_col, sym_col])["total_slippage_bps"]
            .mean()
            .unstack(level=sym_col)
        )
        return pivot
