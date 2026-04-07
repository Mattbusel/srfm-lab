"""
research/portfolio_lab/return_attribution_lab.py

Return attribution analysis for SRFM-Lab.
Supports Brinson-Hood-Beebower (BHB) sector attribution,
factor attribution, transaction cost attribution, and time attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BrinsonResult:
    """
    Brinson-Hood-Beebower attribution results.

    Decomposes active return into:
        total = allocation + selection + interaction

    Attributes
    ----------
    allocation_effect  : value added by overweighting/underweighting sectors
    selection_effect   : value added by picking stocks within sectors
    interaction_effect : joint effect of allocation and selection
    total              : sum of above; equals active return
    by_sector          : per-sector breakdown dict with keys
                         {allocation, selection, interaction, total}
    """

    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total: float
    by_sector: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_series(self) -> pd.Series:
        return pd.Series(
            {
                "allocation": self.allocation_effect,
                "selection": self.selection_effect,
                "interaction": self.interaction_effect,
                "total": self.total,
            }
        )

    def check_consistency(self, tol: float = 1e-8) -> bool:
        """Verify allocation + selection + interaction == total."""
        computed = self.allocation_effect + self.selection_effect + self.interaction_effect
        return abs(computed - self.total) < tol


@dataclass
class FactorResult:
    """Factor attribution of portfolio active return."""

    # total active return explained by each factor
    factor_contributions: Dict[str, float]
    # contribution from alpha (unexplained by factors)
    alpha_contribution: float
    # total return from factors + alpha
    total_explained: float
    # residual not captured by factor model
    unexplained: float
    active_return: float

    def as_series(self) -> pd.Series:
        data: Dict[str, float] = dict(self.factor_contributions)
        data["alpha"] = self.alpha_contribution
        data["unexplained"] = self.unexplained
        data["total_explained"] = self.total_explained
        return pd.Series(data)


@dataclass
class CostImpact:
    """Transaction cost drag analysis."""

    gross_return: float
    total_cost: float
    net_return: float
    # annualised cost drag (assumes 252 days input is already annualised)
    annualised_drag: float
    cost_as_pct_gross: float
    daily_costs: Optional[pd.Series] = None

    @property
    def cost_bps(self) -> float:
        return self.total_cost * 10_000


@dataclass
class TimeAttribution:
    """Attribution of returns across time buckets (e.g. by month, quarter)."""

    bucket_returns: Dict[str, float]  # bucket label -> portfolio return
    bucket_contributions: Dict[str, float]  # bucket label -> contribution to total
    total_return: float
    # number of periods per bucket
    bucket_sizes: Dict[str, int]


# ---------------------------------------------------------------------------
# Main lab
# ---------------------------------------------------------------------------


class ReturnAttributionLab:
    """
    Multi-method return attribution toolkit.

    Methods
    -------
    brinson_attribution          -- BHB sector attribution
    factor_attribution           -- factor model based attribution
    transaction_cost_attribution -- cost drag analysis
    time_attribution             -- attribution by time period
    """

    # ------------------------------------------------------------------
    # Brinson attribution
    # ------------------------------------------------------------------

    def brinson_attribution(
        self,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        portfolio_returns: Dict[str, float],
        benchmark_returns: Dict[str, float],
        sectors: Dict[str, str],
    ) -> BrinsonResult:
        """
        Brinson-Hood-Beebower single-period attribution.

        Parameters
        ----------
        portfolio_weights  : asset -> portfolio weight
        benchmark_weights  : asset -> benchmark weight
        portfolio_returns  : asset -> return over period
        benchmark_returns  : asset -> return over period
        sectors            : asset -> sector label

        BHB formulas (per sector s):
            Rb_s   = benchmark return for sector s (weighted avg of assets in s)
            Rp_s   = portfolio return for sector s
            Rb_tot = total benchmark return
            wp_s   = portfolio weight in sector s
            wb_s   = benchmark weight in sector s

            allocation   = (wp_s - wb_s) * (Rb_s - Rb_tot)
            selection    = wb_s * (Rp_s - Rb_s)
            interaction  = (wp_s - wb_s) * (Rp_s - Rb_s)
        """
        assets = set(portfolio_weights) | set(benchmark_weights)

        # gather unique sectors
        sector_set = set(sectors.get(a, "Unknown") for a in assets)

        # compute benchmark total return (weighted average)
        Rb_tot = sum(
            benchmark_weights.get(a, 0.0) * benchmark_returns.get(a, 0.0)
            for a in assets
        )

        sector_alloc: Dict[str, float] = {}
        sector_sel: Dict[str, float] = {}
        sector_inter: Dict[str, float] = {}
        by_sector: Dict[str, Dict[str, float]] = {}

        for s in sector_set:
            s_assets = [a for a in assets if sectors.get(a, "Unknown") == s]

            wb_s = sum(benchmark_weights.get(a, 0.0) for a in s_assets)
            wp_s = sum(portfolio_weights.get(a, 0.0) for a in s_assets)

            # sector benchmark return (weighted avg within sector)
            if wb_s > 1e-12:
                Rb_s = sum(
                    benchmark_weights.get(a, 0.0) * benchmark_returns.get(a, 0.0)
                    for a in s_assets
                ) / wb_s
            else:
                # no benchmark weight: use equal weight of benchmark returns
                ret_list = [benchmark_returns.get(a, 0.0) for a in s_assets]
                Rb_s = float(np.mean(ret_list)) if ret_list else 0.0

            # sector portfolio return (weighted avg within sector)
            if wp_s > 1e-12:
                Rp_s = sum(
                    portfolio_weights.get(a, 0.0) * portfolio_returns.get(a, 0.0)
                    for a in s_assets
                ) / wp_s
            else:
                ret_list = [portfolio_returns.get(a, 0.0) for a in s_assets]
                Rp_s = float(np.mean(ret_list)) if ret_list else 0.0

            alloc = (wp_s - wb_s) * (Rb_s - Rb_tot)
            sel = wb_s * (Rp_s - Rb_s)
            inter = (wp_s - wb_s) * (Rp_s - Rb_s)

            sector_alloc[s] = alloc
            sector_sel[s] = sel
            sector_inter[s] = inter

            by_sector[s] = {
                "allocation": alloc,
                "selection": sel,
                "interaction": inter,
                "total": alloc + sel + inter,
                "wp": wp_s,
                "wb": wb_s,
                "Rp_s": Rp_s,
                "Rb_s": Rb_s,
            }

        total_alloc = float(sum(sector_alloc.values()))
        total_sel = float(sum(sector_sel.values()))
        total_inter = float(sum(sector_inter.values()))
        total_active = total_alloc + total_sel + total_inter

        return BrinsonResult(
            allocation_effect=total_alloc,
            selection_effect=total_sel,
            interaction_effect=total_inter,
            total=total_active,
            by_sector=by_sector,
        )

    # ------------------------------------------------------------------
    # Factor attribution
    # ------------------------------------------------------------------

    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_exposures: Dict[str, float],
        factor_returns: pd.DataFrame,
    ) -> FactorResult:
        """
        Attribute portfolio returns to factors.

        Parameters
        ----------
        portfolio_returns : daily return series for the portfolio
        factor_exposures  : dict of {factor_name: beta}
        factor_returns    : DataFrame of factor return series

        Returns total factor contributions over the period and residual.
        """
        aligned_factors = factor_returns.dropna()
        aligned = pd.concat([portfolio_returns, aligned_factors], axis=1).dropna()
        port = aligned.iloc[:, 0]
        factors = aligned.iloc[:, 1:]

        total_port_return = float((1.0 + port).prod() - 1.0)

        factor_contributions: Dict[str, float] = {}
        total_factor_contrib = 0.0

        for fname, beta in factor_exposures.items():
            if fname in factors.columns:
                factor_series = factors[fname]
                # total factor return over period
                factor_total = float((1.0 + factor_series).prod() - 1.0)
                contrib = beta * factor_total
                factor_contributions[fname] = contrib
                total_factor_contrib += contrib

        # approximate alpha contribution (assume equal daily alpha)
        residual_daily = port - sum(
            factor_exposures.get(f, 0.0) * factors[f]
            for f in factor_exposures
            if f in factors.columns
        )
        alpha_contribution = float((1.0 + residual_daily).prod() - 1.0)

        total_explained = total_factor_contrib + alpha_contribution
        unexplained = total_port_return - total_explained

        return FactorResult(
            factor_contributions=factor_contributions,
            alpha_contribution=alpha_contribution,
            total_explained=total_explained,
            unexplained=unexplained,
            active_return=total_port_return,
        )

    # ------------------------------------------------------------------
    # Transaction cost attribution
    # ------------------------------------------------------------------

    def transaction_cost_attribution(
        self,
        gross_returns: pd.Series,
        costs: pd.Series,
    ) -> CostImpact:
        """
        Measure the drag of transaction costs on portfolio returns.

        Parameters
        ----------
        gross_returns : daily gross returns (before costs)
        costs         : daily cost (as a positive fraction, e.g. 0.0005 = 5bps)

        Net return = gross_return - cost for each day.
        """
        aligned = pd.concat([gross_returns, costs], axis=1).dropna()
        aligned.columns = ["gross", "cost"]

        net_daily = aligned["gross"] - aligned["cost"]

        gross_total = float((1.0 + aligned["gross"]).prod() - 1.0)
        net_total = float((1.0 + net_daily).prod() - 1.0)
        total_cost = gross_total - net_total

        n_days = len(aligned)
        annualised_drag = total_cost * (252.0 / n_days) if n_days > 0 else 0.0

        cost_as_pct_gross = (
            total_cost / abs(gross_total) if abs(gross_total) > 1e-12 else 0.0
        )

        return CostImpact(
            gross_return=gross_total,
            total_cost=total_cost,
            net_return=net_total,
            annualised_drag=annualised_drag,
            cost_as_pct_gross=cost_as_pct_gross,
            daily_costs=aligned["cost"].copy(),
        )

    # ------------------------------------------------------------------
    # Time attribution
    # ------------------------------------------------------------------

    def time_attribution(
        self,
        returns: pd.Series,
        time_buckets: str = "ME",
    ) -> TimeAttribution:
        """
        Attribute cumulative return by time bucket.

        Parameters
        ----------
        returns      : daily return series with DatetimeIndex
        time_buckets : pandas frequency string for resampling
                       e.g. "ME" = month-end, "QE" = quarter-end,
                       "YE" = year-end, "W" = weekly

        Returns bucket-level returns and each bucket's contribution
        to total portfolio return via geometric compounding.
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns must have a DatetimeIndex")

        resampled = (1.0 + returns).resample(time_buckets).prod() - 1.0

        total_return = float((1.0 + returns).prod() - 1.0)

        # each bucket's contribution: approximate additive share
        # use log-returns for decomposition
        log_total = float(np.log1p(total_return))

        bucket_returns: Dict[str, float] = {}
        bucket_contributions: Dict[str, float] = {}
        bucket_sizes: Dict[str, int] = {}

        for period, bret in resampled.items():
            key = str(period.date() if hasattr(period, "date") else period)
            bucket_returns[key] = float(bret)
            # log contribution as fraction of total log-return
            log_bucket = float(np.log1p(bret))
            contrib = log_bucket / log_total if abs(log_total) > 1e-12 else 0.0
            bucket_contributions[key] = contrib
            # count observations in this bucket
            mask = returns.resample(time_buckets).count()
            bucket_sizes[key] = int(mask.loc[period]) if period in mask.index else 0

        return TimeAttribution(
            bucket_returns=bucket_returns,
            bucket_contributions=bucket_contributions,
            total_return=total_return,
            bucket_sizes=bucket_sizes,
        )


# ---------------------------------------------------------------------------
# Attribution chain
# ---------------------------------------------------------------------------


class AttributionChain:
    """
    Chain multiple attribution results over time using geometric compounding.

    Handles multi-period Brinson attribution by geometrically compounding
    single-period effects.
    """

    @staticmethod
    def compound_effects(effects: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Geometrically compound a list of single-period attribution dicts.

        For each key, returns prod(1 + r_t) - 1 across periods.
        All dicts must share the same keys.
        """
        if not effects:
            return {}

        keys = list(effects[0].keys())
        out: Dict[str, float] = {}
        for k in keys:
            compound = 1.0
            for d in effects:
                compound *= 1.0 + d.get(k, 0.0)
            out[k] = compound - 1.0
        return out

    @staticmethod
    def cumulative_attribution(
        daily_results: List[BrinsonResult],
    ) -> BrinsonResult:
        """
        Compound a list of single-period BrinsonResults into a
        multi-period summary.

        Uses geometric compounding for each effect.
        """
        if not daily_results:
            raise ValueError("daily_results is empty")

        # compound top-level effects
        effects_list = [
            {
                "allocation": r.allocation_effect,
                "selection": r.selection_effect,
                "interaction": r.interaction_effect,
                "total": r.total,
            }
            for r in daily_results
        ]
        compounded = AttributionChain.compound_effects(effects_list)

        # compound by-sector effects
        # collect all sector names
        all_sectors = set()
        for r in daily_results:
            all_sectors.update(r.by_sector.keys())

        by_sector_compound: Dict[str, Dict[str, float]] = {}
        for s in all_sectors:
            sector_effects = []
            for r in daily_results:
                if s in r.by_sector:
                    sector_effects.append(r.by_sector[s])
                else:
                    sector_effects.append(
                        {"allocation": 0.0, "selection": 0.0, "interaction": 0.0, "total": 0.0}
                    )
            by_sector_compound[s] = AttributionChain.compound_effects(sector_effects)

        return BrinsonResult(
            allocation_effect=compounded["allocation"],
            selection_effect=compounded["selection"],
            interaction_effect=compounded["interaction"],
            total=compounded["total"],
            by_sector=by_sector_compound,
        )

    @staticmethod
    def rolling_attribution(
        portfolio_weights_ts: pd.DataFrame,
        benchmark_weights_ts: pd.DataFrame,
        portfolio_returns_ts: pd.DataFrame,
        benchmark_returns_ts: pd.DataFrame,
        sectors: Dict[str, str],
        lab: Optional["ReturnAttributionLab"] = None,
    ) -> pd.DataFrame:
        """
        Compute Brinson attribution on a rolling single-period basis.

        Each row of *_ts represents one period (e.g. daily).
        Returns a DataFrame with columns: date, allocation, selection,
        interaction, total.
        """
        if lab is None:
            lab = ReturnAttributionLab()

        dates = portfolio_weights_ts.index
        records: List[Dict] = []

        for date in dates:
            pw = portfolio_weights_ts.loc[date].to_dict()
            bw = benchmark_weights_ts.loc[date].to_dict()
            pr = portfolio_returns_ts.loc[date].to_dict()
            br = benchmark_returns_ts.loc[date].to_dict()

            result = lab.brinson_attribution(pw, bw, pr, br, sectors)
            records.append(
                {
                    "date": date,
                    "allocation": result.allocation_effect,
                    "selection": result.selection_effect,
                    "interaction": result.interaction_effect,
                    "total": result.total,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("date")
        return df
