"""
execution_analyzer.py — Execution quality analysis for SRFM live trading.

Measures effective spread, implementation shortfall, market impact, slippage,
and generates optimal trade scheduling recommendations.
"""

from __future__ import annotations

import json
import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Average bid-ask half-spread by instrument class (fraction of price)
DEFAULT_HALF_SPREADS: Dict[str, float] = {
    "ES": 0.00010,     # E-mini S&P 500 futures, ~0.25 points / 4500
    "NQ": 0.00008,     # Nasdaq futures
    "CL": 0.00020,     # Crude oil futures
    "GC": 0.00015,     # Gold futures
    "ZB": 0.00012,     # Treasury bond futures
    "BTC": 0.00050,    # Bitcoin
    "ETH": 0.00060,    # Ethereum
    "DEFAULT": 0.00025,
}

# Approximate daily volume (contracts/shares) for market impact
DEFAULT_ADV: Dict[str, float] = {
    "ES": 1_500_000,
    "NQ": 750_000,
    "CL": 500_000,
    "GC": 250_000,
    "ZB": 800_000,
    "BTC": 50_000,
    "ETH": 80_000,
    "DEFAULT": 300_000,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRecord:
    """Single execution event with pre/post trade data."""
    trade_id: str
    instrument: str
    direction: str          # LONG or SHORT
    quantity: float         # number of contracts/shares
    decision_time: pd.Timestamp
    arrival_price: float    # midpoint at decision time
    executed_price: float   # actual fill price
    exit_price: float       # midpoint at exit decision
    filled_time: pd.Timestamp
    benchmark_twap: float = 0.0   # TWAP benchmark
    benchmark_vwap: float = 0.0   # VWAP benchmark

    @property
    def direction_sign(self) -> int:
        return 1 if self.direction == "LONG" else -1

    @property
    def slippage_bps(self) -> float:
        """Signed slippage: positive = cost, negative = benefit."""
        if self.arrival_price <= 0:
            return 0.0
        return (self.executed_price - self.arrival_price) / self.arrival_price * self.direction_sign * 10_000

    @property
    def fill_latency_ms(self) -> float:
        """Milliseconds from decision to fill."""
        return (self.filled_time - self.decision_time).total_seconds() * 1000


@dataclass
class SpreadMetrics:
    instrument: str
    n_observations: int
    quoted_half_spread_bps: float
    effective_half_spread_bps: float
    realized_half_spread_bps: float
    price_impact_bps: float

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "n_obs": self.n_observations,
            "quoted_half_spread_bps": round(self.quoted_half_spread_bps, 3),
            "effective_half_spread_bps": round(self.effective_half_spread_bps, 3),
            "realized_half_spread_bps": round(self.realized_half_spread_bps, 3),
            "price_impact_bps": round(self.price_impact_bps, 3),
        }


@dataclass
class ImplementationShortfall:
    trade_id: str
    instrument: str
    quantity: float
    arrival_cost_bps: float      # vs arrival price
    market_impact_bps: float     # estimated permanent impact
    timing_cost_bps: float       # from delayed execution
    opportunity_cost_bps: float  # from unfilled portion (0 if fully filled)
    total_is_bps: float          # sum of components

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "instrument": self.instrument,
            "quantity": self.quantity,
            "arrival_cost_bps": round(self.arrival_cost_bps, 3),
            "market_impact_bps": round(self.market_impact_bps, 3),
            "timing_cost_bps": round(self.timing_cost_bps, 3),
            "opportunity_cost_bps": round(self.opportunity_cost_bps, 3),
            "total_is_bps": round(self.total_is_bps, 3),
        }


@dataclass
class MarketImpactModel:
    """Almgren-Chriss style linear + nonlinear impact model."""
    instrument: str
    adv: float              # average daily volume
    spread_bps: float       # quoted half-spread in bps
    eta: float              # temporary impact coefficient
    gamma: float            # permanent impact coefficient
    sigma: float            # daily return volatility

    def temporary_impact(self, quantity: float, time_horizon_days: float) -> float:
        """Temporary market impact in bps."""
        participation_rate = quantity / (self.adv * time_horizon_days + 1e-9)
        return self.eta * participation_rate * 10_000

    def permanent_impact(self, quantity: float) -> float:
        """Permanent price impact in bps."""
        participation_rate = quantity / (self.adv + 1e-9)
        return self.gamma * np.sqrt(participation_rate) * 10_000

    def total_impact(self, quantity: float, time_horizon_days: float) -> float:
        return (
            self.temporary_impact(quantity, time_horizon_days)
            + self.permanent_impact(quantity)
            + self.spread_bps
        )


@dataclass
class OptimalSchedule:
    """Almgren-Chriss optimal execution schedule."""
    instrument: str
    total_quantity: float
    time_horizon_days: float
    n_slices: int
    slice_quantities: List[float]
    slice_times: List[float]      # fraction of horizon
    expected_cost_bps: float
    expected_variance: float
    risk_aversion: float

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "total_quantity": self.total_quantity,
            "time_horizon_days": self.time_horizon_days,
            "n_slices": self.n_slices,
            "slice_quantities": [round(q, 2) for q in self.slice_quantities],
            "slice_times": [round(t, 4) for t in self.slice_times],
            "expected_cost_bps": round(self.expected_cost_bps, 3),
            "expected_variance": round(self.expected_variance, 6),
            "risk_aversion": self.risk_aversion,
        }


# ---------------------------------------------------------------------------
# ExecutionAnalyzer
# ---------------------------------------------------------------------------

class ExecutionAnalyzer:
    """
    Execution quality analysis for SRFM live trading strategies.

    Analyzes slippage, market impact, implementation shortfall, and
    generates optimal execution schedules.
    """

    def __init__(
        self,
        executions: Optional[List[ExecutionRecord]] = None,
        instrument_configs: Optional[Dict[str, dict]] = None,
    ):
        self.executions = executions or []
        self.instrument_configs = instrument_configs or {}
        self._impact_models: Dict[str, MarketImpactModel] = {}
        self._build_impact_models()

    def _build_impact_models(self) -> None:
        """Build Almgren-Chriss impact models for each known instrument."""
        known = set(DEFAULT_HALF_SPREADS.keys()) | set(self.instrument_configs.keys())
        for inst in known:
            if inst == "DEFAULT":
                continue
            cfg = self.instrument_configs.get(inst, {})
            adv = cfg.get("adv", DEFAULT_ADV.get(inst, DEFAULT_ADV["DEFAULT"]))
            spread_bps = cfg.get("spread_bps", DEFAULT_HALF_SPREADS.get(inst, DEFAULT_HALF_SPREADS["DEFAULT"]) * 10_000)
            sigma = cfg.get("sigma", 0.015)  # 1.5% daily vol default

            # Calibrate eta, gamma from spread and volatility
            # Simplified: eta = spread/2, gamma = volatility * sqrt_adv_factor
            eta = spread_bps / 2
            gamma = sigma * 0.01

            self._impact_models[inst] = MarketImpactModel(
                instrument=inst,
                adv=adv,
                spread_bps=spread_bps,
                eta=eta,
                gamma=gamma,
                sigma=sigma,
            )

    def _get_impact_model(self, instrument: str) -> MarketImpactModel:
        if instrument in self._impact_models:
            return self._impact_models[instrument]
        # Build a default
        return MarketImpactModel(
            instrument=instrument,
            adv=DEFAULT_ADV["DEFAULT"],
            spread_bps=DEFAULT_HALF_SPREADS["DEFAULT"] * 10_000,
            eta=1.25,
            gamma=0.005,
            sigma=0.015,
        )

    # ------------------------------------------------------------------
    # Effective spread
    # ------------------------------------------------------------------

    def effective_spread(
        self,
        instrument: Optional[str] = None,
    ) -> SpreadMetrics:
        """
        Compute quoted, effective, and realized half-spread metrics.

        Effective half-spread = |executed_price - midpoint| / midpoint
        Realized half-spread  = sign * (post_trade_mid - executed_price) / midpoint
        Price impact          = effective - realized
        """
        recs = [e for e in self.executions if instrument is None or e.instrument == instrument]
        if not recs:
            inst_label = instrument or "ALL"
            return SpreadMetrics(inst_label, 0, 0.0, 0.0, 0.0, 0.0)

        inst_label = instrument or "ALL"
        quoted_model = self._get_impact_model(recs[0].instrument if instrument else "DEFAULT")

        quoted_bps = quoted_model.spread_bps

        eff_spreads = []
        realized_spreads = []

        for r in recs:
            if r.arrival_price <= 0:
                continue
            # Effective half-spread (cost of trading vs midpoint)
            eff = abs(r.executed_price - r.arrival_price) / r.arrival_price * 10_000
            eff_spreads.append(eff)

            # Realized half-spread (revenue to liquidity provider)
            # = sign * (exit_price - executed_price) / arrival_price
            if r.exit_price > 0:
                realized = r.direction_sign * (r.exit_price - r.executed_price) / r.arrival_price * 10_000
                # realized spread = effective - price impact; negative = gave alpha away
                realized_spreads.append(-realized)  # cost perspective

        n = len(eff_spreads)
        if n == 0:
            return SpreadMetrics(inst_label, 0, quoted_bps, 0.0, 0.0, 0.0)

        eff_mean = float(np.mean(eff_spreads))
        real_mean = float(np.mean(realized_spreads)) if realized_spreads else 0.0
        price_impact = max(0.0, eff_mean - real_mean)

        return SpreadMetrics(
            instrument=inst_label,
            n_observations=n,
            quoted_half_spread_bps=quoted_bps,
            effective_half_spread_bps=eff_mean,
            realized_half_spread_bps=real_mean,
            price_impact_bps=price_impact,
        )

    # ------------------------------------------------------------------
    # Realized spread
    # ------------------------------------------------------------------

    def realized_spread(
        self,
        instrument: Optional[str] = None,
        horizon_ms: float = 5000.0,
    ) -> pd.DataFrame:
        """
        Compute realized spread at various time horizons.
        Returns DataFrame with columns: horizon_ms, mean_realized_bps, median_realized_bps.
        """
        recs = [e for e in self.executions if instrument is None or e.instrument == instrument]
        if not recs:
            return pd.DataFrame(columns=["horizon_ms", "mean_realized_bps", "median_realized_bps"])

        horizons = [500, 1000, 2000, 5000, 10000, 30000, 60000]
        rows = []

        for h in horizons:
            rs_list = []
            for r in recs:
                if r.arrival_price <= 0 or r.exit_price <= 0:
                    continue
                # Approximate: use exit_price as the post-trade mid
                # Real implementation would interpolate tick data
                latency_ms = r.fill_latency_ms
                if latency_ms < h:
                    rs = r.direction_sign * (r.exit_price - r.executed_price) / r.arrival_price * 10_000
                    rs_list.append(-rs)  # cost perspective

            if rs_list:
                rows.append({
                    "horizon_ms": h,
                    "mean_realized_bps": float(np.mean(rs_list)),
                    "median_realized_bps": float(np.median(rs_list)),
                    "n_obs": len(rs_list),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Implementation shortfall
    # ------------------------------------------------------------------

    def implementation_shortfall(
        self,
        record: ExecutionRecord,
    ) -> ImplementationShortfall:
        """
        Compute Perold's implementation shortfall decomposition.

        IS = arrival cost + market impact + timing cost + opportunity cost
        """
        if record.arrival_price <= 0:
            return ImplementationShortfall(
                record.trade_id, record.instrument, record.quantity,
                0.0, 0.0, 0.0, 0.0, 0.0
            )

        # Arrival cost: cost vs decision price
        arrival_cost = (
            (record.executed_price - record.arrival_price)
            / record.arrival_price
            * record.direction_sign
            * 10_000
        )

        # Market impact: permanent component
        model = self._get_impact_model(record.instrument)
        market_impact = model.permanent_impact(record.quantity)

        # Timing cost: TWAP/VWAP comparison
        if record.benchmark_twap > 0:
            timing = (
                (record.executed_price - record.benchmark_twap)
                / record.arrival_price
                * record.direction_sign
                * 10_000
            )
        else:
            # Estimate from latency
            latency_s = record.fill_latency_ms / 1000.0
            timing = latency_s * model.sigma / np.sqrt(252 * 6.5 * 3600) * 10_000

        # Opportunity cost: assume fully filled → 0
        opportunity_cost = 0.0

        total = arrival_cost + market_impact + timing + opportunity_cost

        return ImplementationShortfall(
            trade_id=record.trade_id,
            instrument=record.instrument,
            quantity=record.quantity,
            arrival_cost_bps=float(arrival_cost),
            market_impact_bps=float(market_impact),
            timing_cost_bps=float(timing),
            opportunity_cost_bps=float(opportunity_cost),
            total_is_bps=float(total),
        )

    def implementation_shortfall_summary(self) -> pd.DataFrame:
        """Compute IS for all executions, return summary DataFrame."""
        if not self.executions:
            return pd.DataFrame()

        rows = []
        for rec in self.executions:
            is_result = self.implementation_shortfall(rec)
            rows.append(is_result.to_dict())

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    # Timing analysis
    # ------------------------------------------------------------------

    def timing_analysis(self) -> pd.DataFrame:
        """
        Analyze execution timing: slippage vs hour of day, day of week.
        Returns DataFrame with timing breakdown.
        """
        if not self.executions:
            return pd.DataFrame()

        rows = []
        for r in self.executions:
            rows.append({
                "hour": r.decision_time.hour,
                "day_of_week": r.decision_time.dayofweek,
                "slippage_bps": r.slippage_bps,
                "fill_latency_ms": r.fill_latency_ms,
                "quantity": r.quantity,
                "instrument": r.instrument,
            })

        df = pd.DataFrame(rows)

        # By hour
        by_hour = df.groupby("hour").agg(
            mean_slippage_bps=("slippage_bps", "mean"),
            median_slippage_bps=("slippage_bps", "median"),
            mean_latency_ms=("fill_latency_ms", "mean"),
            n_trades=("slippage_bps", "count"),
        ).reset_index()

        return by_hour

    def timing_analysis_dayofweek(self) -> pd.DataFrame:
        """Slippage breakdown by day of week."""
        if not self.executions:
            return pd.DataFrame()

        rows = []
        for r in self.executions:
            rows.append({
                "day_of_week": r.decision_time.dayofweek,
                "slippage_bps": r.slippage_bps,
                "fill_latency_ms": r.fill_latency_ms,
            })

        df = pd.DataFrame(rows)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        by_day = df.groupby("day_of_week").agg(
            mean_slippage_bps=("slippage_bps", "mean"),
            mean_latency_ms=("fill_latency_ms", "mean"),
            n_trades=("slippage_bps", "count"),
        ).reset_index()
        by_day["day_name"] = by_day["day_of_week"].map(
            lambda d: day_names[d] if d < len(day_names) else str(d)
        )

        return by_day

    # ------------------------------------------------------------------
    # Market impact model calibration
    # ------------------------------------------------------------------

    def market_impact_model(
        self,
        instrument: str,
        size_range: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Compute expected market impact across a range of order sizes.
        Returns DataFrame with size, temporary_impact_bps, permanent_impact_bps, total_impact_bps.
        """
        model = self._get_impact_model(instrument)

        if size_range is None:
            size_range = [
                model.adv * f for f in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20]
            ]

        rows = []
        for size in size_range:
            for horizon in [0.125, 0.5, 1.0]:  # 1h, 4h, 1day
                rows.append({
                    "size": size,
                    "size_pct_adv": size / model.adv * 100,
                    "horizon_days": horizon,
                    "temporary_impact_bps": model.temporary_impact(size, horizon),
                    "permanent_impact_bps": model.permanent_impact(size),
                    "total_impact_bps": model.total_impact(size, horizon),
                })

        return pd.DataFrame(rows)

    def calibrate_impact_model(
        self,
        instrument: str,
    ) -> MarketImpactModel:
        """
        Calibrate market impact model from actual execution data via OLS.
        Returns fitted MarketImpactModel for instrument.
        """
        recs = [e for e in self.executions if e.instrument == instrument]
        if len(recs) < 5:
            return self._get_impact_model(instrument)

        model = self._get_impact_model(instrument)

        # Build regression: slippage ~ f(participation_rate, sqrt_participation)
        X_rows = []
        y_vals = []

        for r in recs:
            if r.arrival_price <= 0:
                continue
            participation = r.quantity / (model.adv + 1e-9)
            sqrt_part = np.sqrt(participation)
            slippage = r.slippage_bps
            X_rows.append([participation, sqrt_part, 1.0])
            y_vals.append(slippage)

        if len(y_vals) < 5:
            return model

        X = np.array(X_rows)
        y = np.array(y_vals)

        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            eta_fit = max(0.0, beta[0])
            gamma_fit = max(0.0, beta[1])
        except (np.linalg.LinAlgError, ValueError):
            eta_fit = model.eta
            gamma_fit = model.gamma

        return MarketImpactModel(
            instrument=instrument,
            adv=model.adv,
            spread_bps=model.spread_bps,
            eta=float(eta_fit),
            gamma=float(gamma_fit),
            sigma=model.sigma,
        )

    # ------------------------------------------------------------------
    # Slippage by size
    # ------------------------------------------------------------------

    def slippage_by_size(
        self,
        instrument: Optional[str] = None,
        n_buckets: int = 5,
    ) -> pd.DataFrame:
        """
        Analyze slippage as function of order size (in quantile buckets).
        Returns DataFrame with size_bucket, mean_size, mean_slippage_bps, n_trades.
        """
        recs = [e for e in self.executions if instrument is None or e.instrument == instrument]
        if not recs:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "quantity": r.quantity,
            "slippage_bps": r.slippage_bps,
        } for r in recs])

        df["size_bucket"] = pd.qcut(df["quantity"], n_buckets, labels=False, duplicates="drop")

        result = df.groupby("size_bucket").agg(
            mean_size=("quantity", "mean"),
            median_size=("quantity", "median"),
            mean_slippage_bps=("slippage_bps", "mean"),
            median_slippage_bps=("slippage_bps", "median"),
            std_slippage_bps=("slippage_bps", "std"),
            n_trades=("slippage_bps", "count"),
        ).reset_index()

        return result

    # ------------------------------------------------------------------
    # Optimal trade schedule (Almgren-Chriss)
    # ------------------------------------------------------------------

    def optimal_trade_schedule(
        self,
        instrument: str,
        total_quantity: float,
        time_horizon_days: float = 1.0,
        n_slices: int = 8,
        risk_aversion: float = 1e-6,
    ) -> OptimalSchedule:
        """
        Compute optimal execution schedule using Almgren-Chriss framework.

        Minimizes: E[cost] + lambda * Var[cost]

        The optimal trajectory is linear (TWAP) in the risk-neutral case,
        and front-loaded when risk aversion is high.

        Returns OptimalSchedule with slice quantities and times.
        """
        model = self._get_impact_model(instrument)

        tau = time_horizon_days / n_slices  # time per slice in days
        sigma_daily = model.sigma

        # Almgren-Chriss: kappa^2 = risk_aversion * sigma^2 / eta
        # where eta = temporary impact coefficient (per unit)
        eta_per_unit = model.eta / (model.adv * 10_000)  # convert bps/% to absolute
        if eta_per_unit < 1e-15:
            eta_per_unit = 1e-8

        kappa2 = risk_aversion * sigma_daily**2 / (2 * eta_per_unit)
        kappa = np.sqrt(max(kappa2, 0.0))

        # Optimal holdings trajectory: x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
        T = time_horizon_days
        times_frac = np.linspace(0, 1, n_slices + 1)
        times_abs = times_frac * T

        if kappa * T < 1e-8:
            # Risk-neutral limit: linear (TWAP)
            holdings = total_quantity * (1 - times_frac)
        else:
            sinh_kT = np.sinh(kappa * T)
            if sinh_kT < 1e-10:
                holdings = total_quantity * (1 - times_frac)
            else:
                holdings = total_quantity * np.sinh(kappa * (T - times_abs)) / sinh_kT

        holdings = np.clip(holdings, 0, total_quantity)
        holdings[-1] = 0.0  # must be fully sold by end

        # Slice quantities = reduction in holdings
        slice_qty = np.diff(-holdings)  # how much sold each period
        # Normalize to sum to total_quantity
        total_sold = slice_qty.sum()
        if total_sold > 1e-9:
            slice_qty = slice_qty * total_quantity / total_sold
        slice_qty = np.clip(slice_qty, 0, None)

        # Compute expected cost and variance
        total_cost_bps = 0.0
        total_var = 0.0
        remaining = total_quantity

        for i, q in enumerate(slice_qty):
            t_start = times_abs[i]
            t_end = times_abs[i + 1]
            dt = t_end - t_start

            # Temporary impact for this slice
            tmp_impact = model.eta * (q / (model.adv * dt + 1e-9))
            # Permanent impact
            perm_impact = model.gamma * np.sqrt(q / (model.adv + 1e-9)) * 10_000

            slice_cost = (tmp_impact + perm_impact + model.spread_bps) * q
            total_cost_bps += slice_cost

            # Variance from remaining inventory
            var_slice = risk_aversion * sigma_daily**2 * remaining**2 * dt
            total_var += var_slice
            remaining -= q

        # Normalize cost to per-unit bps
        if total_quantity > 0:
            total_cost_bps /= total_quantity

        return OptimalSchedule(
            instrument=instrument,
            total_quantity=total_quantity,
            time_horizon_days=time_horizon_days,
            n_slices=n_slices,
            slice_quantities=slice_qty.tolist(),
            slice_times=times_frac[1:].tolist(),
            expected_cost_bps=float(total_cost_bps),
            expected_variance=float(total_var),
            risk_aversion=risk_aversion,
        )

    # ------------------------------------------------------------------
    # TWAP / VWAP benchmarks
    # ------------------------------------------------------------------

    def compute_twap(
        self,
        price_series: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> float:
        """Compute TWAP between start and end from price series."""
        mask = (price_series.index >= start) & (price_series.index <= end)
        segment = price_series.loc[mask]
        if segment.empty:
            return float(price_series.iloc[0]) if len(price_series) > 0 else 0.0
        return float(segment.mean())

    def compute_vwap(
        self,
        price_series: pd.Series,
        volume_series: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> float:
        """Compute VWAP between start and end."""
        mask = (price_series.index >= start) & (price_series.index <= end)
        prices = price_series.loc[mask]
        volumes = volume_series.loc[mask]
        if prices.empty or volumes.sum() <= 0:
            return float(price_series.iloc[0]) if len(price_series) > 0 else 0.0
        return float((prices * volumes).sum() / volumes.sum())

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary_statistics(self) -> dict:
        """Compute overall execution quality summary."""
        if not self.executions:
            return {"error": "no executions"}

        slippages = [e.slippage_bps for e in self.executions]
        latencies = [e.fill_latency_ms for e in self.executions]
        quantities = [e.quantity for e in self.executions]

        instruments = list({e.instrument for e in self.executions})
        spread_metrics = [self.effective_spread(inst) for inst in instruments]

        is_results = self.implementation_shortfall_summary()
        mean_is = float(is_results["total_is_bps"].mean()) if not is_results.empty else 0.0

        return {
            "n_executions": len(self.executions),
            "instruments": instruments,
            "mean_slippage_bps": round(float(np.mean(slippages)), 3),
            "median_slippage_bps": round(float(np.median(slippages)), 3),
            "std_slippage_bps": round(float(np.std(slippages, ddof=1)), 3),
            "p95_slippage_bps": round(float(np.percentile(slippages, 95)), 3),
            "mean_latency_ms": round(float(np.mean(latencies)), 1),
            "median_latency_ms": round(float(np.median(latencies)), 1),
            "mean_quantity": round(float(np.mean(quantities)), 2),
            "mean_is_bps": round(mean_is, 3),
            "spread_metrics": [s.to_dict() for s in spread_metrics],
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_execution_report(
        self,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """Generate comprehensive execution quality report plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib not available; skipping plots")
            return

        fig = plt.figure(figsize=(20, 22))
        fig.suptitle("SRFM Execution Quality Report", fontsize=16, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

        has_data = len(self.executions) > 0
        slippages = [e.slippage_bps for e in self.executions] if has_data else []
        latencies = [e.fill_latency_ms for e in self.executions] if has_data else []

        # ---- 1. Slippage distribution ----
        ax1 = fig.add_subplot(gs[0, 0])
        if slippages:
            ax1.hist(slippages, bins=min(30, len(slippages) // 2 + 1),
                     color="steelblue", edgecolor="white", alpha=0.8)
            ax1.axvline(np.mean(slippages), color="red", linestyle="--",
                        label=f"Mean: {np.mean(slippages):.2f} bps")
            ax1.axvline(0, color="k", linewidth=0.8)
            ax1.set_title("Slippage Distribution", fontweight="bold")
            ax1.set_xlabel("Slippage (bps)")
            ax1.set_ylabel("Frequency")
            ax1.legend(fontsize=8)
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)

        # ---- 2. Fill latency distribution ----
        ax2 = fig.add_subplot(gs[0, 1])
        if latencies:
            ax2.hist(latencies, bins=min(30, len(latencies) // 2 + 1),
                     color="darkorange", edgecolor="white", alpha=0.8)
            ax2.axvline(np.mean(latencies), color="red", linestyle="--",
                        label=f"Mean: {np.mean(latencies):.0f} ms")
            ax2.set_title("Fill Latency Distribution", fontweight="bold")
            ax2.set_xlabel("Latency (ms)")
            ax2.set_ylabel("Frequency")
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)

        # ---- 3. Slippage over time ----
        ax3 = fig.add_subplot(gs[0, 2])
        if has_data:
            times = [e.decision_time for e in self.executions]
            ax3.scatter(times, slippages, alpha=0.4, s=10, color="navy")
            window = min(20, len(slippages))
            if len(slippages) >= window:
                roll_mean = pd.Series(slippages, index=times).rolling(window).mean()
                ax3.plot(times, roll_mean.values, color="red", linewidth=1.5,
                         label=f"{window}-trade rolling mean")
            ax3.axhline(0, color="k", linewidth=0.8)
            ax3.set_title("Slippage Over Time", fontweight="bold")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Slippage (bps)")
            ax3.tick_params(axis="x", rotation=45)
            ax3.legend(fontsize=8)
        else:
            ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)

        # ---- 4. Slippage by hour ----
        ax4 = fig.add_subplot(gs[1, 0])
        timing_df = self.timing_analysis()
        if not timing_df.empty:
            ax4.bar(timing_df["hour"], timing_df["mean_slippage_bps"],
                    color="purple", alpha=0.7)
            ax4.axhline(0, color="k", linewidth=0.8)
            ax4.set_title("Mean Slippage by Hour of Day", fontweight="bold")
            ax4.set_xlabel("Hour (UTC)")
            ax4.set_ylabel("Mean Slippage (bps)")
        else:
            ax4.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax4.transAxes)

        # ---- 5. Slippage by size ----
        ax5 = fig.add_subplot(gs[1, 1])
        size_df = self.slippage_by_size(n_buckets=min(5, len(self.executions) // 3 + 1))
        if not size_df.empty:
            ax5.errorbar(
                size_df["mean_size"],
                size_df["mean_slippage_bps"],
                yerr=size_df["std_slippage_bps"].fillna(0),
                marker="o",
                color="darkgreen",
                capsize=4,
            )
            ax5.axhline(0, color="k", linewidth=0.8)
            ax5.set_title("Slippage vs Order Size", fontweight="bold")
            ax5.set_xlabel("Mean Order Size")
            ax5.set_ylabel("Mean Slippage (bps)")
        else:
            ax5.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax5.transAxes)

        # ---- 6. Market impact model ----
        ax6 = fig.add_subplot(gs[1, 2])
        if self.executions:
            inst = self.executions[0].instrument
        else:
            inst = "ES"
        impact_df = self.market_impact_model(inst)
        sub = impact_df[impact_df["horizon_days"] == 1.0]
        if not sub.empty:
            ax6.plot(sub["size_pct_adv"], sub["temporary_impact_bps"],
                     label="Temporary", color="blue", marker="o", markersize=4)
            ax6.plot(sub["size_pct_adv"], sub["permanent_impact_bps"],
                     label="Permanent", color="red", marker="s", markersize=4)
            ax6.plot(sub["size_pct_adv"], sub["total_impact_bps"],
                     label="Total", color="black", linewidth=2)
            ax6.set_title(f"Market Impact Model ({inst})", fontweight="bold")
            ax6.set_xlabel("Order Size (% of ADV)")
            ax6.set_ylabel("Impact (bps)")
            ax6.legend(fontsize=8)
            ax6.set_xscale("log")
        else:
            ax6.text(0.5, 0.5, "No model data", ha="center", va="center",
                     transform=ax6.transAxes)

        # ---- 7. Implementation shortfall breakdown ----
        ax7 = fig.add_subplot(gs[2, :2])
        is_df = self.implementation_shortfall_summary()
        if not is_df.empty:
            components = ["arrival_cost_bps", "market_impact_bps", "timing_cost_bps"]
            labels = ["Arrival Cost", "Market Impact", "Timing Cost"]
            colors = ["steelblue", "darkorange", "purple"]
            means = [is_df[c].mean() for c in components]
            bars = ax7.bar(labels, means, color=colors, alpha=0.8)
            ax7.axhline(0, color="k", linewidth=0.8)
            for bar, mean in zip(bars, means):
                ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{mean:.2f} bps", ha="center", va="bottom", fontsize=9)
            ax7.set_title("Implementation Shortfall Components", fontweight="bold")
            ax7.set_ylabel("Mean Cost (bps)")
        else:
            ax7.text(0.5, 0.5, "No IS data", ha="center", va="center",
                     transform=ax7.transAxes)

        # ---- 8. Optimal schedule ----
        ax8 = fig.add_subplot(gs[2, 2])
        inst = self.executions[0].instrument if self.executions else "ES"
        model = self._get_impact_model(inst)
        for ra, label, color in [
            (1e-8, "Low risk aversion", "green"),
            (1e-6, "Medium risk aversion", "orange"),
            (1e-4, "High risk aversion", "red"),
        ]:
            schedule = self.optimal_trade_schedule(inst, model.adv * 0.01, 1.0, 8, ra)
            cum_pct = np.cumsum(schedule.slice_quantities) / max(sum(schedule.slice_quantities), 1e-9) * 100
            ax8.plot([0] + list(np.array(schedule.slice_times) * 100), [0] + list(cum_pct),
                     label=label, color=color, marker="o", markersize=3)
        ax8.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="TWAP (linear)")
        ax8.set_title(f"Optimal Execution Trajectory ({inst})", fontweight="bold")
        ax8.set_xlabel("% of Time Horizon")
        ax8.set_ylabel("% Quantity Executed")
        ax8.legend(fontsize=7)

        # ---- 9. Spread metrics table ----
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis("off")
        instruments = list({e.instrument for e in self.executions}) if self.executions else ["ES"]
        spread_data = []
        for inst_name in instruments[:6]:
            sm = self.effective_spread(inst_name)
            spread_data.append([
                inst_name,
                f"{sm.n_observations}",
                f"{sm.quoted_half_spread_bps:.2f}",
                f"{sm.effective_half_spread_bps:.2f}",
                f"{sm.realized_half_spread_bps:.2f}",
                f"{sm.price_impact_bps:.2f}",
            ])

        if spread_data:
            col_labels = [
                "Instrument", "N Obs", "Quoted HS (bps)",
                "Effective HS (bps)", "Realized HS (bps)", "Price Impact (bps)"
            ]
            table = ax9.table(
                cellText=spread_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            ax9.set_title("Spread Metrics by Instrument", fontweight="bold", y=0.95)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Execution report saved to {output_path}")
        elif show:
            plt.show()
        plt.close(fig)

    def export_results(self, output_path: str) -> dict:
        """Export all execution analysis results to JSON."""
        summary = self.summary_statistics()
        is_df = self.implementation_shortfall_summary()
        timing_df = self.timing_analysis()

        results = {
            "summary": summary,
            "implementation_shortfall": is_df.to_dict(orient="records") if not is_df.empty else [],
            "timing_by_hour": timing_df.to_dict(orient="records") if not timing_df.empty else [],
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Execution analysis results saved to {output_path}")
        return results


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_executions(
    n_trades: int = 200,
    instruments: Optional[List[str]] = None,
    seed: int = 42,
) -> List[ExecutionRecord]:
    """
    Generate realistic synthetic execution records for testing.
    Includes intraday patterns: higher slippage at open/close, latency variation.
    """
    if instruments is None:
        instruments = ["ES", "NQ", "CL"]

    rng = np.random.default_rng(seed)
    records = []
    base_time = pd.Timestamp("2024-01-02 09:30:00", tz="UTC")

    for i in range(n_trades):
        inst = rng.choice(instruments)
        direction = rng.choice(["LONG", "SHORT"])
        quantity = float(rng.integers(1, 20))

        # Simulate intraday time (market hours)
        day_offset = i // 10
        hour_offset = float(rng.uniform(0, 6.5))  # 0 to 6.5 hours into session
        decision_time = base_time + pd.Timedelta(days=day_offset, hours=hour_offset)

        # Base price by instrument
        base_prices = {"ES": 4500.0, "NQ": 15000.0, "CL": 75.0}
        base_price = base_prices.get(inst, 100.0)
        arrival_price = base_price * (1 + rng.normal(0, 0.001))

        # Slippage: higher at open (h < 1) and close (h > 6), scales with size
        hour_frac = hour_offset / 6.5
        intraday_factor = 1.0 + 2.0 * abs(hour_frac - 0.5)  # U-shaped
        half_spread = DEFAULT_HALF_SPREADS.get(inst, DEFAULT_HALF_SPREADS["DEFAULT"])
        base_slippage_bps = half_spread * 10_000 * intraday_factor
        size_factor = 1.0 + 0.05 * quantity
        slippage_bps = float(rng.normal(base_slippage_bps * size_factor, base_slippage_bps))

        dir_sign = 1 if direction == "LONG" else -1
        executed_price = arrival_price * (1 + dir_sign * slippage_bps / 10_000)

        # Exit price: simulated with P&L
        pnl_bps = float(rng.normal(5.0, 50.0))
        exit_price = executed_price * (1 + dir_sign * pnl_bps / 10_000)

        # Fill latency
        latency_ms = float(rng.exponential(50.0) + 5.0) * intraday_factor
        filled_time = decision_time + pd.Timedelta(milliseconds=latency_ms)

        # TWAP benchmark (add some deviation)
        twap_dev_bps = float(rng.normal(0, half_spread * 5000))
        twap = arrival_price * (1 + twap_dev_bps / 10_000)

        records.append(ExecutionRecord(
            trade_id=f"T{i:05d}",
            instrument=inst,
            direction=direction,
            quantity=quantity,
            decision_time=decision_time,
            arrival_price=float(arrival_price),
            executed_price=float(executed_price),
            exit_price=float(exit_price),
            filled_time=filled_time,
            benchmark_twap=float(twap),
            benchmark_vwap=float(twap * (1 + rng.normal(0, 0.0001))),
        ))

    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SRFM Execution Quality Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python execution_analyzer.py --synthetic --n-trades 500 --output exec_report.png
  python execution_analyzer.py --synthetic --export exec_results.json
  python execution_analyzer.py --synthetic --schedule ES --quantity 100
""",
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--n-trades", type=int, default=200, help="Number of synthetic trades")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="execution_report.png",
                        help="Output plot path")
    parser.add_argument("--export", type=str, help="Export JSON results to path")
    parser.add_argument("--schedule", type=str, help="Run optimal schedule for instrument")
    parser.add_argument("--quantity", type=float, default=100, help="Schedule quantity")
    parser.add_argument("--horizon", type=float, default=1.0, help="Schedule horizon (days)")
    parser.add_argument("--risk-aversion", type=float, default=1e-6, help="Risk aversion lambda")
    args = parser.parse_args()

    # Generate or load executions
    if args.synthetic:
        print(f"Generating {args.n_trades} synthetic execution records...")
        executions = generate_synthetic_executions(args.n_trades, seed=args.seed)
    else:
        print("No data source specified; use --synthetic")
        return

    analyzer = ExecutionAnalyzer(executions)
    summary = analyzer.summary_statistics()

    print(f"\n--- Execution Summary ({len(executions)} trades) ---")
    print(f"  Mean slippage:    {summary['mean_slippage_bps']:+.3f} bps")
    print(f"  Median slippage:  {summary['median_slippage_bps']:+.3f} bps")
    print(f"  95th pct slipp:   {summary['p95_slippage_bps']:+.3f} bps")
    print(f"  Mean latency:     {summary['mean_latency_ms']:.1f} ms")
    print(f"  Mean IS:          {summary['mean_is_bps']:+.3f} bps")

    print("\n--- Spread Metrics ---")
    for sm in summary["spread_metrics"]:
        print(f"  {sm['instrument']}: quoted={sm['quoted_half_spread_bps']:.2f} bps, "
              f"effective={sm['effective_half_spread_bps']:.2f} bps, "
              f"impact={sm['price_impact_bps']:.2f} bps  (n={sm['n_obs']})")

    if args.schedule:
        print(f"\n--- Optimal Schedule: {args.schedule} ---")
        schedule = analyzer.optimal_trade_schedule(
            args.schedule, args.quantity, args.horizon, 8, args.risk_aversion
        )
        print(f"  Total quantity:   {schedule.total_quantity}")
        print(f"  Time horizon:     {schedule.time_horizon_days} days")
        print(f"  Expected cost:    {schedule.expected_cost_bps:.3f} bps")
        print(f"  Slice quantities: {[round(q, 2) for q in schedule.slice_quantities]}")

    print(f"\nGenerating execution report → {args.output}")
    analyzer.plot_execution_report(output_path=args.output)

    if args.export:
        analyzer.export_results(args.export)


if __name__ == "__main__":
    main()
