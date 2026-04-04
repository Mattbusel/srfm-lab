"""
tca.py — Transaction Cost Analysis Engine
==========================================

Full TCA for the srfm-lab Alpaca live-trading system.

Key Classes
-----------
TCAEngine   : Stateless computation of individual cost components
TCAReport   : Dataclass holding all cost metrics for one trade
TCAAnalyzer : High-level analysis of trade portfolios
"""

from __future__ import annotations

import math
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    """A single execution fill from the broker."""
    sym: str
    side: str                    # "buy" | "sell"
    qty: float
    fill_price: float
    fill_time: datetime
    order_id: str = ""
    exchange: str = ""
    notional: float = field(init=False)

    def __post_init__(self) -> None:
        self.notional = self.qty * self.fill_price
        self.side = self.side.lower()
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")


@dataclass
class Trade:
    """Aggregated trade — may consist of multiple fills."""
    sym: str
    side: str
    fills: list[Fill]
    decision_time: datetime
    decision_price: float
    arrival_price: float | None = None   # price at order submission
    market_vwap: float | None = None     # market VWAP over trade horizon
    market_twap: float | None = None     # market TWAP over trade horizon
    bid_ask_spread: float | None = None  # at decision time (absolute)
    adv: float | None = None             # average daily volume in USD
    daily_vol: float | None = None       # daily return volatility (fractional)
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> int:
        """+1 for buys, -1 for sells."""
        return 1 if self.side.lower() == "buy" else -1

    @property
    def total_qty(self) -> float:
        return sum(f.qty for f in self.fills)

    @property
    def total_notional(self) -> float:
        return sum(f.notional for f in self.fills)

    @property
    def vwap_fill(self) -> float:
        """Volume-weighted average fill price."""
        total_notional = sum(f.notional for f in self.fills)
        total_qty = sum(f.qty for f in self.fills)
        return total_notional / total_qty if total_qty > 0 else 0.0

    @property
    def start_fill_time(self) -> datetime:
        return min(f.fill_time for f in self.fills)

    @property
    def end_fill_time(self) -> datetime:
        return max(f.fill_time for f in self.fills)


@dataclass
class TCAReport:
    """Full TCA result for a single trade."""
    sym: str
    side: str
    total_notional: float
    decision_price: float
    vwap_fill: float

    # Cost components (all in basis points)
    implementation_shortfall_bps: float = 0.0
    vwap_slippage_bps: float = 0.0
    twap_slippage_bps: float = 0.0
    arrival_price_slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    spread_cost_bps: float = 0.0
    timing_alpha_bps: float = 0.0
    total_cost_bps: float = 0.0

    # Derived
    total_cost_dollars: float = 0.0
    num_fills: int = 1
    trade_horizon_seconds: float = 0.0
    data_quality_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sym": self.sym,
            "side": self.side,
            "total_notional": self.total_notional,
            "decision_price": self.decision_price,
            "vwap_fill": self.vwap_fill,
            "implementation_shortfall_bps": self.implementation_shortfall_bps,
            "vwap_slippage_bps": self.vwap_slippage_bps,
            "twap_slippage_bps": self.twap_slippage_bps,
            "arrival_price_slippage_bps": self.arrival_price_slippage_bps,
            "market_impact_bps": self.market_impact_bps,
            "spread_cost_bps": self.spread_cost_bps,
            "timing_alpha_bps": self.timing_alpha_bps,
            "total_cost_bps": self.total_cost_bps,
            "total_cost_dollars": self.total_cost_dollars,
            "num_fills": self.num_fills,
            "trade_horizon_seconds": self.trade_horizon_seconds,
            "data_quality_flags": self.data_quality_flags,
        }


@dataclass
class PortfolioTCAReport:
    """Aggregated TCA across a set of trades."""
    n_trades: int
    total_notional: float
    mean_is_bps: float
    median_is_bps: float
    p95_is_bps: float
    mean_vwap_slippage_bps: float
    mean_market_impact_bps: float
    mean_spread_cost_bps: float
    mean_timing_alpha_bps: float
    mean_total_cost_bps: float
    total_cost_dollars: float
    notional_weighted_total_cost_bps: float
    by_sym: dict[str, "SymTCAReport"] = field(default_factory=dict)
    individual_reports: list[TCAReport] = field(default_factory=list)


@dataclass
class SymTCAReport:
    """Per-symbol TCA summary."""
    sym: str
    n_trades: int
    total_notional: float
    mean_is_bps: float
    mean_total_cost_bps: float
    total_cost_dollars: float


# ---------------------------------------------------------------------------
# Core engine — stateless cost calculations
# ---------------------------------------------------------------------------

class TCAEngine:
    """
    Stateless engine for computing individual TCA cost components.

    All return values are in basis points (bps) unless otherwise stated.
    1 bps = 0.01% = 0.0001

    Conventions
    -----------
    - direction: +1 for buys, -1 for sells
    - Higher cost = worse execution (costs are positive for bad fills)
    """

    # -----------------------------------------------------------------------
    # Implementation Shortfall
    # -----------------------------------------------------------------------

    @staticmethod
    def implementation_shortfall(
        decision_price: float,
        fill_price: float,
        direction: int,
    ) -> float:
        """
        Implementation Shortfall (IS) in basis points.

        IS measures the difference between the paper portfolio (trading at
        decision price) and the actual portfolio (trading at fill price).

        Parameters
        ----------
        decision_price : float
            Price at time of investment decision.
        fill_price : float
            Actual volume-weighted average fill price.
        direction : int
            +1 for buy, -1 for sell.

        Returns
        -------
        float
            IS in bps. Positive = cost (bad), Negative = alpha (lucky fill).

        Notes
        -----
        IS = direction × (fill_price − decision_price) / decision_price × 10_000
        For a buy: bad if fill_price > decision_price.
        For a sell: bad if fill_price < decision_price.
        """
        if decision_price <= 0:
            raise ValueError(f"decision_price must be positive, got {decision_price}")
        if direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {direction}")

        return direction * (fill_price - decision_price) / decision_price * 10_000

    # -----------------------------------------------------------------------
    # VWAP benchmark
    # -----------------------------------------------------------------------

    @staticmethod
    def vwap_benchmark(
        fills: list[Fill],
        market_vwap: float,
    ) -> float:
        """
        Slippage relative to market VWAP in bps.

        Parameters
        ----------
        fills : list[Fill]
        market_vwap : float
            Market VWAP over the trading horizon.

        Returns
        -------
        float
            VWAP slippage in bps.  Positive = worse than market.
        """
        if not fills:
            return 0.0
        if market_vwap <= 0:
            raise ValueError(f"market_vwap must be positive, got {market_vwap}")

        # Determine direction from fills
        side = fills[0].side
        direction = 1 if side == "buy" else -1

        total_notional = sum(f.notional for f in fills)
        total_qty = sum(f.qty for f in fills)
        if total_qty == 0:
            return 0.0
        our_vwap = total_notional / total_qty

        return direction * (our_vwap - market_vwap) / market_vwap * 10_000

    # -----------------------------------------------------------------------
    # TWAP benchmark
    # -----------------------------------------------------------------------

    @staticmethod
    def twap_benchmark(
        fills: list[Fill],
        start_time: datetime,
        end_time: datetime,
        market_prices: pd.Series | None = None,
    ) -> float:
        """
        Slippage relative to market TWAP in bps.

        If `market_prices` is a time-indexed Series of market prices, the
        market TWAP is computed from it over [start_time, end_time].
        Otherwise, a midpoint approximation is used.

        Parameters
        ----------
        fills : list[Fill]
        start_time, end_time : datetime
        market_prices : optional pd.Series indexed by datetime

        Returns
        -------
        float
            TWAP slippage in bps.
        """
        if not fills:
            return 0.0

        side = fills[0].side
        direction = 1 if side == "buy" else -1

        total_notional = sum(f.notional for f in fills)
        total_qty = sum(f.qty for f in fills)
        if total_qty == 0:
            return 0.0
        our_twap = total_notional / total_qty

        if market_prices is not None and len(market_prices) > 0:
            mask = (market_prices.index >= start_time) & (market_prices.index <= end_time)
            relevant = market_prices[mask]
            if len(relevant) > 0:
                market_twap = float(relevant.mean())
            else:
                # fall back to fill-based estimate
                market_twap = float(market_prices.iloc[-1])
        else:
            # Approximate: use first and last fill midpoint
            prices_sorted = sorted(fills, key=lambda f: f.fill_time)
            market_twap = (prices_sorted[0].fill_price + prices_sorted[-1].fill_price) / 2

        if market_twap <= 0:
            return 0.0

        return direction * (our_twap - market_twap) / market_twap * 10_000

    # -----------------------------------------------------------------------
    # Arrival price benchmark
    # -----------------------------------------------------------------------

    @staticmethod
    def arrival_price_benchmark(
        fill_price: float,
        arrival_price: float,
        direction: int,
    ) -> float:
        """
        Slippage vs the mid-quote at order arrival time, in bps.

        Parameters
        ----------
        fill_price : float
            VWAP fill price.
        arrival_price : float
            Mid-quote at order submission.
        direction : int
            +1 buy, -1 sell.

        Returns
        -------
        float
            Arrival price slippage in bps.
        """
        if arrival_price <= 0:
            raise ValueError(f"arrival_price must be positive, got {arrival_price}")

        return direction * (fill_price - arrival_price) / arrival_price * 10_000

    # -----------------------------------------------------------------------
    # Market impact model (Almgren-Chriss simplified)
    # -----------------------------------------------------------------------

    @staticmethod
    def market_impact_model(
        dollar_size: float,
        adv: float,
        daily_vol: float,
        side: str,
    ) -> float:
        """
        Estimate market impact using a simplified Almgren-Chriss model.

        The model separates temporary and permanent impact:
          temporary = σ × sqrt(Q / V) × 0.5
          permanent  = 0.1 × (Q / V)
          total = temporary + permanent   (in bps)

        Parameters
        ----------
        dollar_size : float
            Trade size in USD notional.
        adv : float
            Average daily volume in USD.
        daily_vol : float
            Daily return volatility (e.g. 0.02 = 2%).
        side : str
            "buy" | "sell"

        Returns
        -------
        float
            Expected market impact in bps (always non-negative).
        """
        if adv <= 0:
            raise ValueError(f"adv must be positive, got {adv}")
        if daily_vol < 0:
            raise ValueError(f"daily_vol must be non-negative, got {daily_vol}")
        if dollar_size < 0:
            raise ValueError(f"dollar_size must be non-negative, got {dollar_size}")

        participation_rate = dollar_size / adv
        # Almgren-Chriss: temporary + permanent components
        temporary_impact = daily_vol * math.sqrt(participation_rate) * 0.5
        permanent_impact = 0.1 * participation_rate
        # Convert to bps (model returns fraction, multiply by 10000)
        return (temporary_impact + permanent_impact) * 10_000

    # -----------------------------------------------------------------------
    # Spread cost
    # -----------------------------------------------------------------------

    @staticmethod
    def spread_cost(
        bid_ask_spread: float,
        fill_price: float,
    ) -> float:
        """
        Half-spread cost in bps.

        Parameters
        ----------
        bid_ask_spread : float
            Absolute bid-ask spread (ask − bid) in price units.
        fill_price : float
            Reference price (typically mid).

        Returns
        -------
        float
            Half-spread cost in bps.
        """
        if fill_price <= 0:
            raise ValueError(f"fill_price must be positive, got {fill_price}")
        if bid_ask_spread < 0:
            raise ValueError(f"bid_ask_spread must be non-negative, got {bid_ask_spread}")

        half_spread = bid_ask_spread / 2
        return (half_spread / fill_price) * 10_000

    # -----------------------------------------------------------------------
    # Timing alpha
    # -----------------------------------------------------------------------

    @staticmethod
    def timing_alpha(
        fill_price: float,
        decision_price: float,
        market_return: float,
    ) -> float:
        """
        Timing component of implementation shortfall.

        Timing alpha measures how much of the IS is explained by general
        market movement vs alpha-seeking timing decisions.

        timing_alpha = direction × market_return × 10_000

        Parameters
        ----------
        fill_price : float
        decision_price : float
        market_return : float
            Market (SPY/BTC index) return between decision and fill.

        Returns
        -------
        float
            Timing component in bps. Positive = market moved against us.
        """
        direction = 1 if fill_price >= decision_price else -1
        return direction * market_return * 10_000


# ---------------------------------------------------------------------------
# High-level analyzer
# ---------------------------------------------------------------------------

class TCAAnalyzer:
    """
    High-level TCA analysis over collections of trades.

    Wraps TCAEngine to provide per-trade, portfolio, and sliced analyses.
    """

    def __init__(self) -> None:
        self.engine = TCAEngine()

    def analyze_trade(
        self,
        trade: Trade,
        market_data: dict[str, Any] | None = None,
    ) -> TCAReport:
        """
        Produce a TCAReport for a single trade.

        Parameters
        ----------
        trade : Trade
        market_data : dict, optional
            May contain keys: 'market_return', 'market_prices' (pd.Series)

        Returns
        -------
        TCAReport
        """
        if not trade.fills:
            raise ValueError(f"Trade {trade.sym} has no fills")

        vwap_fill = trade.vwap_fill
        direction = trade.direction
        md = market_data or {}
        flags: list[str] = []

        # --- IS ---
        is_bps = self.engine.implementation_shortfall(
            trade.decision_price, vwap_fill, direction
        )

        # --- VWAP ---
        if trade.market_vwap:
            vwap_slip = self.engine.vwap_benchmark(trade.fills, trade.market_vwap)
        else:
            vwap_slip = 0.0
            flags.append("no_market_vwap")

        # --- TWAP ---
        market_prices: pd.Series | None = md.get("market_prices")
        twap_slip = self.engine.twap_benchmark(
            trade.fills,
            trade.start_fill_time,
            trade.end_fill_time,
            market_prices=market_prices,
        )

        # --- Arrival price ---
        if trade.arrival_price:
            arr_slip = self.engine.arrival_price_benchmark(
                vwap_fill, trade.arrival_price, direction
            )
        else:
            arr_slip = 0.0
            flags.append("no_arrival_price")

        # --- Market impact ---
        if trade.adv and trade.daily_vol is not None:
            impact_bps = self.engine.market_impact_model(
                trade.total_notional, trade.adv, trade.daily_vol, trade.side
            )
        else:
            impact_bps = 0.0
            flags.append("no_adv_or_vol")

        # --- Spread cost ---
        if trade.bid_ask_spread:
            spread_bps = self.engine.spread_cost(trade.bid_ask_spread, vwap_fill)
        else:
            spread_bps = 0.0
            flags.append("no_spread_data")

        # --- Timing alpha ---
        market_return: float = md.get("market_return", 0.0)
        timing_bps = self.engine.timing_alpha(vwap_fill, trade.decision_price, market_return)

        # --- Total cost ---
        # Total = IS is the ground truth; others are attribution
        total_cost_bps = is_bps
        total_cost_dollars = total_cost_bps / 10_000 * trade.total_notional

        horizon_seconds = (
            (trade.end_fill_time - trade.start_fill_time).total_seconds()
            if len(trade.fills) > 1
            else 0.0
        )

        return TCAReport(
            sym=trade.sym,
            side=trade.side,
            total_notional=trade.total_notional,
            decision_price=trade.decision_price,
            vwap_fill=vwap_fill,
            implementation_shortfall_bps=is_bps,
            vwap_slippage_bps=vwap_slip,
            twap_slippage_bps=twap_slip,
            arrival_price_slippage_bps=arr_slip,
            market_impact_bps=impact_bps,
            spread_cost_bps=spread_bps,
            timing_alpha_bps=timing_bps,
            total_cost_bps=total_cost_bps,
            total_cost_dollars=total_cost_dollars,
            num_fills=len(trade.fills),
            trade_horizon_seconds=horizon_seconds,
            data_quality_flags=flags,
        )

    def analyze_portfolio(
        self,
        trades: list[Trade],
        market_data: dict[str, Any] | None = None,
    ) -> PortfolioTCAReport:
        """
        Analyze a list of trades and return portfolio-level TCAReport.

        Parameters
        ----------
        trades : list[Trade]
        market_data : dict, optional

        Returns
        -------
        PortfolioTCAReport
        """
        if not trades:
            raise ValueError("trades list is empty")

        reports: list[TCAReport] = []
        for t in trades:
            try:
                r = self.analyze_trade(t, market_data)
                reports.append(r)
            except Exception as exc:
                logger.warning("Skipping trade %s %s: %s", t.sym, t.decision_time, exc)

        if not reports:
            raise RuntimeError("All trades failed TCA analysis")

        is_vals = np.array([r.implementation_shortfall_bps for r in reports])
        notionals = np.array([r.total_notional for r in reports])
        total_notional = float(notionals.sum())

        # Notional-weighted cost
        if total_notional > 0:
            nw_cost = float(np.average([r.total_cost_bps for r in reports], weights=notionals))
        else:
            nw_cost = float(np.mean([r.total_cost_bps for r in reports]))

        by_sym = self._build_sym_reports(reports)

        return PortfolioTCAReport(
            n_trades=len(reports),
            total_notional=total_notional,
            mean_is_bps=float(np.mean(is_vals)),
            median_is_bps=float(np.median(is_vals)),
            p95_is_bps=float(np.percentile(is_vals, 95)),
            mean_vwap_slippage_bps=float(np.mean([r.vwap_slippage_bps for r in reports])),
            mean_market_impact_bps=float(np.mean([r.market_impact_bps for r in reports])),
            mean_spread_cost_bps=float(np.mean([r.spread_cost_bps for r in reports])),
            mean_timing_alpha_bps=float(np.mean([r.timing_alpha_bps for r in reports])),
            mean_total_cost_bps=float(np.mean([r.total_cost_bps for r in reports])),
            total_cost_dollars=float(sum(r.total_cost_dollars for r in reports)),
            notional_weighted_total_cost_bps=nw_cost,
            by_sym=by_sym,
            individual_reports=reports,
        )

    def _build_sym_reports(self, reports: list[TCAReport]) -> dict[str, SymTCAReport]:
        """Group individual reports by symbol."""
        groups: dict[str, list[TCAReport]] = {}
        for r in reports:
            groups.setdefault(r.sym, []).append(r)

        out: dict[str, SymTCAReport] = {}
        for sym, sym_reports in groups.items():
            out[sym] = SymTCAReport(
                sym=sym,
                n_trades=len(sym_reports),
                total_notional=sum(r.total_notional for r in sym_reports),
                mean_is_bps=float(np.mean([r.implementation_shortfall_bps for r in sym_reports])),
                mean_total_cost_bps=float(np.mean([r.total_cost_bps for r in sym_reports])),
                total_cost_dollars=sum(r.total_cost_dollars for r in sym_reports),
            )
        return out

    def cost_breakdown_by_sym(
        self,
        trades: list[Trade],
        market_data: dict[str, Any] | None = None,
    ) -> dict[str, TCAReport]:
        """
        Return a dict of sym → aggregated TCAReport.

        Rather than returning individual trades, this groups fills by symbol
        and returns a representative report per symbol.
        """
        groups: dict[str, list[Trade]] = {}
        for t in trades:
            groups.setdefault(t.sym, []).append(t)

        result: dict[str, TCAReport] = {}
        for sym, sym_trades in groups.items():
            # Build an aggregate pseudo-report by averaging
            reps = [self.analyze_trade(t, market_data) for t in sym_trades]
            avg = TCAReport(
                sym=sym,
                side="mixed" if len({t.side for t in sym_trades}) > 1 else sym_trades[0].side,
                total_notional=sum(r.total_notional for r in reps),
                decision_price=float(np.mean([r.decision_price for r in reps])),
                vwap_fill=float(np.mean([r.vwap_fill for r in reps])),
                implementation_shortfall_bps=float(np.mean([r.implementation_shortfall_bps for r in reps])),
                vwap_slippage_bps=float(np.mean([r.vwap_slippage_bps for r in reps])),
                twap_slippage_bps=float(np.mean([r.twap_slippage_bps for r in reps])),
                arrival_price_slippage_bps=float(np.mean([r.arrival_price_slippage_bps for r in reps])),
                market_impact_bps=float(np.mean([r.market_impact_bps for r in reps])),
                spread_cost_bps=float(np.mean([r.spread_cost_bps for r in reps])),
                timing_alpha_bps=float(np.mean([r.timing_alpha_bps for r in reps])),
                total_cost_bps=float(np.mean([r.total_cost_bps for r in reps])),
                total_cost_dollars=sum(r.total_cost_dollars for r in reps),
                num_fills=sum(r.num_fills for r in reps),
            )
            result[sym] = avg
        return result

    def cost_breakdown_by_size(
        self,
        trades: list[Trade],
        n_buckets: int = 5,
        market_data: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Return size-bucket vs cost DataFrame.

        Parameters
        ----------
        trades : list[Trade]
        n_buckets : int
            Number of size quantile buckets.
        market_data : dict, optional

        Returns
        -------
        pd.DataFrame
            Columns: size_bucket_label, mean_notional, mean_is_bps,
                     mean_impact_bps, mean_total_cost_bps, count
        """
        reports = [self.analyze_trade(t, market_data) for t in trades]
        df = pd.DataFrame([r.as_dict() for r in reports])

        if df.empty:
            return pd.DataFrame()

        df["size_bucket"] = pd.qcut(
            df["total_notional"], q=n_buckets, labels=False, duplicates="drop"
        )
        labels = {
            i: f"Q{i+1}" for i in range(n_buckets)
        }
        df["size_bucket_label"] = df["size_bucket"].map(labels)

        summary = (
            df.groupby("size_bucket_label")
            .agg(
                mean_notional=("total_notional", "mean"),
                mean_is_bps=("implementation_shortfall_bps", "mean"),
                mean_impact_bps=("market_impact_bps", "mean"),
                mean_total_cost_bps=("total_cost_bps", "mean"),
                count=("sym", "count"),
            )
            .reset_index()
        )
        return summary

    # -----------------------------------------------------------------------
    # Plotting helpers
    # -----------------------------------------------------------------------

    def plot_is_distribution(
        self,
        trades: list[Trade],
        save_path: str | Path,
        market_data: dict[str, Any] | None = None,
    ) -> None:
        """
        Plot histogram of implementation shortfall across trades.

        Parameters
        ----------
        trades : list[Trade]
        save_path : str | Path
            Where to save the figure (PNG or PDF).
        market_data : dict, optional
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed — cannot plot")
            return

        reports = [self.analyze_trade(t, market_data) for t in trades]
        is_vals = [r.implementation_shortfall_bps for r in reports]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(is_vals, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Zero IS")
        mean_is = float(np.mean(is_vals))
        ax.axvline(mean_is, color="red", linewidth=1.5, linestyle="-", label=f"Mean IS = {mean_is:.1f} bps")
        ax.set_xlabel("Implementation Shortfall (bps)")
        ax.set_ylabel("Trade Count")
        ax.set_title("Implementation Shortfall Distribution")
        ax.legend()
        fig.tight_layout()
        plt.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("IS distribution saved to %s", save_path)

    def plot_cost_vs_size(
        self,
        trades: list[Trade],
        save_path: str | Path,
        market_data: dict[str, Any] | None = None,
    ) -> None:
        """
        Scatter plot of trade notional vs total cost (bps).

        Parameters
        ----------
        trades : list[Trade]
        save_path : str | Path
        market_data : dict, optional
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed — cannot plot")
            return

        reports = [self.analyze_trade(t, market_data) for t in trades]
        notionals = [r.total_notional for r in reports]
        costs = [r.total_cost_bps for r in reports]

        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(notionals, costs, alpha=0.5, c=costs, cmap="RdYlGn_r", s=30)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Trade Notional (USD)")
        ax.set_ylabel("Total Cost (bps)")
        ax.set_title("Execution Cost vs Trade Size")
        plt.colorbar(scatter, ax=ax, label="Cost (bps)")

        # Optional: fit log-linear trend
        if len(notionals) > 5:
            log_n = np.log(np.array(notionals, dtype=float) + 1)
            c_arr = np.array(costs, dtype=float)
            poly = np.polyfit(log_n, c_arr, 1)
            x_range = np.linspace(min(notionals), max(notionals), 200)
            trend = np.polyval(poly, np.log(x_range + 1))
            ax.plot(x_range, trend, color="blue", linewidth=1.5, label="Log-linear fit")
            ax.legend()

        fig.tight_layout()
        plt.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("Cost vs size plot saved to %s", save_path)

    # -----------------------------------------------------------------------
    # Utility — load from DB
    # -----------------------------------------------------------------------

    @staticmethod
    def load_trades_from_db(db_path: str | Path) -> list[Trade]:
        """
        Load trades from live_trades.db SQLite database.

        Expected table: trades
        Columns: sym, side, fill_price, qty, fill_time, order_id,
                 decision_time, decision_price, arrival_price, adv, daily_vol,
                 bid_ask_spread, market_vwap

        Returns
        -------
        list[Trade]
        """
        import sqlite3

        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"DB not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql_query("SELECT * FROM trades", conn)
        finally:
            conn.close()

        if df.empty:
            return []

        # Parse timestamps
        for col in ("fill_time", "decision_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        trades: list[Trade] = []
        for order_id, group in df.groupby("order_id"):
            row0 = group.iloc[0]
            fills = [
                Fill(
                    sym=row["sym"],
                    side=row["side"],
                    qty=float(row["qty"]),
                    fill_price=float(row["fill_price"]),
                    fill_time=row["fill_time"].to_pydatetime(),
                    order_id=str(row.get("order_id", "")),
                )
                for _, row in group.iterrows()
            ]
            trades.append(
                Trade(
                    sym=str(row0["sym"]),
                    side=str(row0["side"]),
                    fills=fills,
                    decision_time=row0["decision_time"].to_pydatetime(),
                    decision_price=float(row0["decision_price"]),
                    arrival_price=float(row0["arrival_price"]) if "arrival_price" in row0 and pd.notna(row0["arrival_price"]) else None,
                    market_vwap=float(row0["market_vwap"]) if "market_vwap" in row0 and pd.notna(row0["market_vwap"]) else None,
                    adv=float(row0["adv"]) if "adv" in row0 and pd.notna(row0["adv"]) else None,
                    daily_vol=float(row0["daily_vol"]) if "daily_vol" in row0 and pd.notna(row0["daily_vol"]) else None,
                    bid_ask_spread=float(row0["bid_ask_spread"]) if "bid_ask_spread" in row0 and pd.notna(row0["bid_ask_spread"]) else None,
                )
            )
        return trades


# ---------------------------------------------------------------------------
# HTML Report generation
# ---------------------------------------------------------------------------

def generate_html_report(
    portfolio_report: PortfolioTCAReport,
    output_path: str | Path,
) -> None:
    """
    Generate a self-contained HTML TCA report.

    Parameters
    ----------
    portfolio_report : PortfolioTCAReport
    output_path : str | Path
    """
    output_path = Path(output_path)
    pr = portfolio_report

    rows_html = "\n".join(
        f"""
        <tr>
          <td>{sym}</td>
          <td>{sr.n_trades}</td>
          <td>${sr.total_notional:,.0f}</td>
          <td>{sr.mean_is_bps:.1f}</td>
          <td>{sr.mean_total_cost_bps:.1f}</td>
          <td>${sr.total_cost_dollars:,.0f}</td>
        </tr>
        """
        for sym, sr in sorted(pr.by_sym.items())
    )

    individual_rows = "\n".join(
        f"""
        <tr>
          <td>{r.sym}</td>
          <td>{r.side}</td>
          <td>${r.total_notional:,.0f}</td>
          <td>{r.implementation_shortfall_bps:.1f}</td>
          <td>{r.vwap_slippage_bps:.1f}</td>
          <td>{r.market_impact_bps:.1f}</td>
          <td>{r.spread_cost_bps:.1f}</td>
          <td>{r.total_cost_bps:.1f}</td>
          <td>{r.num_fills}</td>
        </tr>
        """
        for r in pr.individual_reports
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>TCA Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2em; background: #f8f9fa; }}
  h1,h2 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
  th {{ background: #4a90d9; color: white; text-align: center; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap:1em; margin-bottom:2em; }}
  .card {{ background: white; border-radius: 8px; padding: 1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .card h3 {{ margin: 0 0 0.5em; font-size: 0.85em; color: #666; }}
  .card p {{ margin: 0; font-size: 1.4em; font-weight: bold; color: #333; }}
  .pos {{ color: #c0392b; }} .neg {{ color: #27ae60; }}
</style>
</head>
<body>
<h1>Transaction Cost Analysis Report</h1>
<p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>

<div class="summary-grid">
  <div class="card"><h3>Trades Analyzed</h3><p>{pr.n_trades}</p></div>
  <div class="card"><h3>Total Notional</h3><p>${pr.total_notional:,.0f}</p></div>
  <div class="card"><h3>Mean IS (bps)</h3><p class="{'pos' if pr.mean_is_bps > 0 else 'neg'}">{pr.mean_is_bps:.2f}</p></div>
  <div class="card"><h3>Total Cost ($)</h3><p class="pos">${pr.total_cost_dollars:,.0f}</p></div>
  <div class="card"><h3>Median IS (bps)</h3><p>{pr.median_is_bps:.2f}</p></div>
  <div class="card"><h3>P95 IS (bps)</h3><p>{pr.p95_is_bps:.2f}</p></div>
  <div class="card"><h3>Mean Impact (bps)</h3><p>{pr.mean_market_impact_bps:.2f}</p></div>
  <div class="card"><h3>NW Total Cost (bps)</h3><p>{pr.notional_weighted_total_cost_bps:.2f}</p></div>
</div>

<h2>By Symbol</h2>
<table>
<thead><tr>
  <th>Symbol</th><th>Trades</th><th>Notional</th>
  <th>Mean IS (bps)</th><th>Mean Cost (bps)</th><th>Total Cost ($)</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>

<h2>Individual Trades</h2>
<table>
<thead><tr>
  <th>Symbol</th><th>Side</th><th>Notional</th>
  <th>IS (bps)</th><th>VWAP Slip (bps)</th><th>Impact (bps)</th>
  <th>Spread (bps)</th><th>Total (bps)</th><th>Fills</th>
</tr></thead>
<tbody>{individual_rows}</tbody>
</table>

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)
