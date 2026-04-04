"""
order_splitting.py — Order Splitting and Optimal Execution Scheduling
=====================================================================

Handles:
  - Alpaca $200K max-notional-per-order limit
  - TWAP / VWAP / Almgren-Chriss optimal schedule generation
  - Adaptive (live-feed aware) scheduling
  - Post-hoc schedule evaluation
"""

from __future__ import annotations

import math
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Generator, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """A single order to be submitted to Alpaca."""
    sym: str
    side: str           # "buy" | "sell"
    notional: float     # USD notional (≤ 200_000 for Alpaca)
    qty: float          # shares/units (0 if notional-based)
    limit_price: float | None
    schedule_time: datetime
    order_id: str = ""
    slice_index: int = 0
    total_slices: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.notional > 200_000:
            raise ValueError(
                f"Order notional {self.notional:,.0f} exceeds Alpaca $200K limit. "
                f"Use OrderSplitter.split_order() first."
            )


@dataclass
class ScheduledOrder:
    """An order with additional scheduling metadata."""
    order: Order
    expected_time: datetime
    expected_impact_bps: float = 0.0
    expected_participation_rate: float = 0.0
    cumulative_qty_pct: float = 0.0     # fraction of total done after this slice
    notes: str = ""


@dataclass
class ScheduleEvaluation:
    """Comparison of a planned schedule vs actual fills."""
    sym: str
    planned_n_slices: int
    actual_n_fills: int
    planned_vwap: float
    actual_vwap: float
    vwap_slippage_bps: float
    planned_horizon_minutes: float
    actual_horizon_minutes: float
    schedule_adherence_pct: float       # 0-100: how close were fills to planned times
    total_slippage_bps: float
    total_slippage_dollars: float
    timing_errors: list[float]          # actual - planned time in seconds per slice
    unfilled_slices: int
    notes: str = ""

    def is_good_execution(self, threshold_bps: float = 5.0) -> bool:
        return abs(self.total_slippage_bps) < threshold_bps


# ---------------------------------------------------------------------------
# Volume profile helper
# ---------------------------------------------------------------------------

DEFAULT_EQUITY_VOLUME_PROFILE: dict[int, float] = {
    9: 0.12, 10: 0.10, 11: 0.09, 12: 0.08, 13: 0.07,
    14: 0.08, 15: 0.11, 16: 0.15,
}
"""Approximate fraction of daily equity volume per hour (market hours)."""

DEFAULT_CRYPTO_VOLUME_PROFILE: dict[int, float] = {
    h: 1/24 for h in range(24)
}
"""Uniform 24-hour crypto volume profile."""


def normalize_volume_profile(profile: dict[int, float]) -> dict[int, float]:
    """Normalize a volume profile so fractions sum to 1.0."""
    total = sum(profile.values())
    if total == 0:
        raise ValueError("Volume profile sums to zero")
    return {k: v / total for k, v in profile.items()}


# ---------------------------------------------------------------------------
# Main splitter class
# ---------------------------------------------------------------------------

class OrderSplitter:
    """
    Splits large orders to comply with Alpaca's $200K max-notional limit
    and generates optimal execution schedules.

    Parameters
    ----------
    max_per_order : float
        Maximum notional per child order (default 200_000).
    """

    ALPACA_MAX_NOTIONAL = 200_000.0

    def __init__(self, max_per_order: float = 200_000.0) -> None:
        if max_per_order <= 0 or max_per_order > self.ALPACA_MAX_NOTIONAL:
            raise ValueError(
                f"max_per_order must be in (0, {self.ALPACA_MAX_NOTIONAL}], "
                f"got {max_per_order}"
            )
        self.max_per_order = max_per_order

    # -----------------------------------------------------------------------
    # Simple split
    # -----------------------------------------------------------------------

    def split_order(
        self,
        sym: str,
        side: str,
        target_notional: float,
        max_per_order: float | None = None,
        start_time: datetime | None = None,
        interval_minutes: float = 1.0,
        limit_price: float | None = None,
    ) -> list[Order]:
        """
        Split a large order into child orders ≤ max_per_order.

        Uses an even split (each child has the same notional).

        Parameters
        ----------
        sym : str
        side : str
            "buy" | "sell"
        target_notional : float
            Total USD notional to trade.
        max_per_order : float, optional
            Override per-order limit. Defaults to self.max_per_order.
        start_time : datetime, optional
            When to start. Defaults to now.
        interval_minutes : float
            Minutes between child orders.
        limit_price : float, optional
            Limit price for each child order.

        Returns
        -------
        list[Order]
        """
        cap = max_per_order or self.max_per_order
        if target_notional <= 0:
            raise ValueError(f"target_notional must be positive, got {target_notional}")

        n_orders = math.ceil(target_notional / cap)
        per_order_notional = target_notional / n_orders  # even split

        now = start_time or datetime.utcnow()
        orders: list[Order] = []

        for i in range(n_orders):
            sched_time = now + timedelta(minutes=i * interval_minutes)
            orders.append(
                Order(
                    sym=sym,
                    side=side,
                    notional=per_order_notional,
                    qty=0.0,  # notional-based
                    limit_price=limit_price,
                    schedule_time=sched_time,
                    slice_index=i,
                    total_slices=n_orders,
                    metadata={"split_type": "even", "target_notional": target_notional},
                )
            )

        logger.info(
            "split_order: %s %s $%.0f → %d child orders of $%.0f each",
            sym, side, target_notional, n_orders, per_order_notional,
        )
        return orders

    # -----------------------------------------------------------------------
    # TWAP schedule
    # -----------------------------------------------------------------------

    def twap_schedule(
        self,
        sym: str,
        side: str,
        total_qty: float,
        n_slices: int,
        interval_minutes: float = 5.0,
        start_time: datetime | None = None,
        price_estimate: float | None = None,
        adv: float | None = None,
        daily_vol: float | None = None,
    ) -> list[ScheduledOrder]:
        """
        Generate a TWAP (Time-Weighted Average Price) schedule.

        Divides total_qty equally across n_slices, spaced interval_minutes apart.

        Parameters
        ----------
        sym : str
        side : str
        total_qty : float
            Total quantity to trade.
        n_slices : int
            Number of equal time slices.
        interval_minutes : float
        start_time : datetime, optional
        price_estimate : float, optional
            Estimate of current price (used for notional check and impact).
        adv : float, optional
            Average daily volume in USD (for impact estimation).
        daily_vol : float, optional
            Daily return volatility (for impact estimation).

        Returns
        -------
        list[ScheduledOrder]
        """
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        if total_qty <= 0:
            raise ValueError(f"total_qty must be positive, got {total_qty}")

        slice_qty = total_qty / n_slices
        now = start_time or datetime.utcnow()
        scheduled: list[ScheduledOrder] = []

        for i in range(n_slices):
            sched_time = now + timedelta(minutes=i * interval_minutes)

            # Estimate notional for this slice
            notional = slice_qty * price_estimate if price_estimate else 0.0
            # Ensure we don't exceed the $200K limit per order
            if notional > self.max_per_order:
                raise ValueError(
                    f"TWAP slice {i} notional ${notional:,.0f} exceeds $200K limit. "
                    f"Reduce price_estimate or increase n_slices."
                )

            # Impact estimate
            impact_bps = 0.0
            participation_rate = 0.0
            if adv and daily_vol and price_estimate and adv > 0:
                slice_notional = slice_qty * price_estimate
                participation_rate = slice_notional / adv
                impact_bps = (
                    daily_vol * math.sqrt(participation_rate) * 0.5
                    + 0.1 * participation_rate
                ) * 10_000

            order = Order(
                sym=sym,
                side=side,
                notional=notional if notional > 0 else 0.0,
                qty=slice_qty,
                limit_price=None,
                schedule_time=sched_time,
                slice_index=i,
                total_slices=n_slices,
                metadata={"schedule_type": "TWAP", "interval_min": interval_minutes},
            )

            scheduled.append(
                ScheduledOrder(
                    order=order,
                    expected_time=sched_time,
                    expected_impact_bps=impact_bps,
                    expected_participation_rate=participation_rate,
                    cumulative_qty_pct=(i + 1) / n_slices,
                )
            )

        logger.info(
            "twap_schedule: %s %s qty=%.4f over %d slices @ %g min intervals",
            sym, side, total_qty, n_slices, interval_minutes,
        )
        return scheduled

    # -----------------------------------------------------------------------
    # VWAP schedule
    # -----------------------------------------------------------------------

    def vwap_schedule(
        self,
        sym: str,
        side: str,
        total_qty: float,
        volume_profile: dict[int, float],
        start_time: datetime | None = None,
        price_estimate: float | None = None,
        max_participation_rate: float = 0.10,
    ) -> list[ScheduledOrder]:
        """
        Generate a VWAP schedule aligned with expected volume profile.

        Parameters
        ----------
        sym : str
        side : str
        total_qty : float
        volume_profile : dict[int, float]
            Mapping of hour → fraction of daily volume. Will be normalized.
        start_time : datetime, optional
        price_estimate : float, optional
        max_participation_rate : float
            Cap per-period participation rate.

        Returns
        -------
        list[ScheduledOrder]
        """
        if not volume_profile:
            raise ValueError("volume_profile cannot be empty")

        profile = normalize_volume_profile(volume_profile)
        now = start_time or datetime.utcnow()
        scheduled: list[ScheduledOrder] = []

        hours_sorted = sorted(profile.keys())
        cumulative_qty = 0.0

        for idx, hour in enumerate(hours_sorted):
            fraction = profile[hour]
            slice_qty = total_qty * fraction

            # Clamp to max participation rate
            if price_estimate and slice_qty * price_estimate > self.max_per_order:
                orig = slice_qty
                slice_qty = self.max_per_order / price_estimate
                logger.warning(
                    "VWAP slice hour=%d clamped qty %.4f → %.4f (notional limit)",
                    hour, orig, slice_qty,
                )

            cumulative_qty += slice_qty
            sched_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if sched_time < now:
                sched_time += timedelta(days=1)

            notional = slice_qty * price_estimate if price_estimate else 0.0
            participation_rate = fraction * max_participation_rate

            order = Order(
                sym=sym,
                side=side,
                notional=min(notional, self.max_per_order),
                qty=slice_qty,
                limit_price=None,
                schedule_time=sched_time,
                slice_index=idx,
                total_slices=len(hours_sorted),
                metadata={
                    "schedule_type": "VWAP",
                    "hour": hour,
                    "volume_fraction": fraction,
                },
            )

            scheduled.append(
                ScheduledOrder(
                    order=order,
                    expected_time=sched_time,
                    expected_impact_bps=0.0,
                    expected_participation_rate=participation_rate,
                    cumulative_qty_pct=cumulative_qty / total_qty,
                )
            )

        logger.info(
            "vwap_schedule: %s %s qty=%.4f over %d hourly buckets",
            sym, side, total_qty, len(hours_sorted),
        )
        return scheduled

    # -----------------------------------------------------------------------
    # Almgren-Chriss optimal schedule
    # -----------------------------------------------------------------------

    def almgren_chriss_schedule(
        self,
        sym: str,
        side: str,
        total_qty: float,
        T: float,
        n_slices: int | None = None,
        risk_aversion: float = 1e-6,
        sigma: float = 0.01,
        eta: float = 0.1,
        gamma: float = 0.1,
        start_time: datetime | None = None,
        price_estimate: float | None = None,
    ) -> list[ScheduledOrder]:
        """
        Generate an Almgren-Chriss optimal liquidation schedule.

        The A-C model minimizes the expected cost + lambda × variance of cost,
        yielding a closed-form optimal trajectory.

        Model parameters
        ----------------
        sigma    : daily volatility (fraction)
        eta      : temporary impact (price per share per unit time)
        gamma    : permanent impact (price per share traded)
        T        : trading horizon in days
        lambda_  : risk aversion (= risk_aversion parameter)

        Closed-form solution
        --------------------
        kappa² = lambda × sigma² / eta
        kappa  = sqrt(kappa²)
        x(t_j) = X × sinh(kappa(T - t_j)) / sinh(kappa × T)

        where x(t_j) is the remaining inventory at time t_j.

        Parameters
        ----------
        sym : str
        side : str
        total_qty : float
            Total quantity X to liquidate.
        T : float
            Trading horizon in DAYS.
        n_slices : int, optional
            Number of trading intervals. Defaults to int(T * 24) for crypto.
        risk_aversion : float
            Lambda in A-C model. Higher → more risk averse (faster execution).
        sigma : float
            Annualised or daily volatility of the asset.
        eta : float
            Temporary market impact coefficient.
        gamma : float
            Permanent market impact coefficient.
        start_time : datetime, optional
        price_estimate : float, optional

        Returns
        -------
        list[ScheduledOrder]
        """
        if T <= 0:
            raise ValueError(f"T (trading horizon) must be positive, got {T}")
        if total_qty <= 0:
            raise ValueError(f"total_qty must be positive, got {total_qty}")

        if n_slices is None:
            n_slices = max(2, int(T * 24))  # hourly for crypto by default

        # A-C closed-form parameters
        kappa_sq = risk_aversion * (sigma ** 2) / eta
        kappa = math.sqrt(max(kappa_sq, 1e-12))

        # Time grid: t_0=0, ..., t_N=T
        tau = T / n_slices  # time step (in days)

        now = start_time or datetime.utcnow()
        scheduled: list[ScheduledOrder] = []

        prev_inventory = total_qty
        cumulative_sold = 0.0

        for j in range(n_slices):
            t_j = j * tau
            t_j1 = (j + 1) * tau

            # Remaining inventory at t_j and t_{j+1}
            denom = math.sinh(kappa * T)
            if denom < 1e-10:
                # kappa → 0: linear (TWAP) schedule
                x_j = total_qty * (T - t_j) / T
                x_j1 = total_qty * (T - t_j1) / T
            else:
                x_j = total_qty * math.sinh(kappa * (T - t_j)) / denom
                x_j1 = total_qty * math.sinh(kappa * (T - t_j1)) / denom

            slice_qty = max(0.0, x_j - x_j1)
            cumulative_sold += slice_qty

            sched_time = now + timedelta(days=t_j)

            # Market impact estimate for this slice
            impact_bps = 0.0
            participation_rate = 0.0
            if price_estimate and price_estimate > 0:
                slice_notional = slice_qty * price_estimate
                # Temporary impact in price units: eta * (slice_qty / tau)
                temp_impact_price = eta * (slice_qty / tau) if tau > 0 else 0.0
                impact_bps = (temp_impact_price / price_estimate) * 10_000
                participation_rate = slice_notional / (price_estimate * 1e6)  # vs $1M ADV

            notional = min(slice_qty * price_estimate, self.max_per_order) if price_estimate else 0.0

            order = Order(
                sym=sym,
                side=side,
                notional=notional,
                qty=slice_qty,
                limit_price=None,
                schedule_time=sched_time,
                slice_index=j,
                total_slices=n_slices,
                metadata={
                    "schedule_type": "AlmgrenChriss",
                    "t_j": t_j,
                    "remaining_inventory": x_j1,
                    "kappa": kappa,
                    "tau": tau,
                },
            )

            scheduled.append(
                ScheduledOrder(
                    order=order,
                    expected_time=sched_time,
                    expected_impact_bps=impact_bps,
                    expected_participation_rate=participation_rate,
                    cumulative_qty_pct=cumulative_sold / total_qty,
                    notes=f"A-C kappa={kappa:.4f}, tau={tau:.4f}d",
                )
            )

        logger.info(
            "almgren_chriss_schedule: %s %s qty=%.4f T=%.2fd %d slices "
            "(kappa=%.4f, lambda=%.2e)",
            sym, side, total_qty, T, n_slices, kappa, risk_aversion,
        )
        return scheduled

    # -----------------------------------------------------------------------
    # Adaptive schedule (generator)
    # -----------------------------------------------------------------------

    def adaptive_schedule(
        self,
        sym: str,
        side: str,
        total_qty: float,
        live_feed_fn: Callable[[], dict[str, float]],
        urgency: float = 0.5,
        min_slice_qty: float | None = None,
        max_slice_notional: float | None = None,
        price_key: str = "mid",
        volume_key: str = "volume_rate",
    ) -> Generator[Order, dict[str, Any], None]:
        """
        Adaptive execution schedule as a generator.

        The generator yields Order objects one at a time. After each yield,
        the caller should send back a dict with actual fill info:
          {"filled_qty": float, "fill_price": float, "timestamp": datetime}

        The schedule adapts based on live market data from live_feed_fn().

        Parameters
        ----------
        sym : str
        side : str
        total_qty : float
        live_feed_fn : Callable[[], dict[str, float]]
            Called to get current market state. Must return at minimum
            {price_key: float}.
        urgency : float
            0.0 = very passive (wait for good prices)
            1.0 = very aggressive (trade immediately)
        min_slice_qty : float, optional
        max_slice_notional : float, optional

        Yields
        ------
        Order

        Receives (via send)
        -------------------
        dict with fill info
        """
        remaining_qty = total_qty
        max_slice_notional = max_slice_notional or self.max_per_order
        slice_index = 0
        now = datetime.utcnow()

        while remaining_qty > (min_slice_qty or total_qty * 0.01):
            # Poll live market data
            try:
                market = live_feed_fn()
            except Exception as exc:
                logger.warning("live_feed_fn failed: %s — using defaults", exc)
                market = {}

            price = float(market.get(price_key, 0.0))
            volume_rate = float(market.get(volume_key, 1.0))  # relative to normal

            # Adaptive sizing: scale by urgency and volume rate
            base_fraction = urgency * (1 + volume_rate * 0.5)
            base_fraction = max(0.05, min(0.50, base_fraction))
            slice_qty = remaining_qty * base_fraction

            if price > 0:
                slice_notional = slice_qty * price
                if slice_notional > max_slice_notional:
                    slice_qty = max_slice_notional / price

            if min_slice_qty and slice_qty < min_slice_qty:
                slice_qty = min(min_slice_qty, remaining_qty)

            slice_qty = min(slice_qty, remaining_qty)
            if slice_qty <= 0:
                break

            order = Order(
                sym=sym,
                side=side,
                notional=slice_qty * price if price > 0 else 0.0,
                qty=slice_qty,
                limit_price=None,
                schedule_time=datetime.utcnow(),
                slice_index=slice_index,
                total_slices=-1,  # unknown
                metadata={
                    "schedule_type": "adaptive",
                    "urgency": urgency,
                    "volume_rate": volume_rate,
                    "remaining_before": remaining_qty,
                },
            )

            fill_info: dict[str, Any] = yield order

            if fill_info is None:
                # No fill info provided — assume full fill
                filled_qty = slice_qty
            else:
                filled_qty = float(fill_info.get("filled_qty", slice_qty))

            remaining_qty -= filled_qty
            slice_index += 1

            logger.debug(
                "adaptive_schedule: slice %d filled %.4f, remaining %.4f",
                slice_index, filled_qty, remaining_qty,
            )

        logger.info(
            "adaptive_schedule: completed %s %s qty=%.4f in %d slices",
            sym, side, total_qty, slice_index,
        )

    # -----------------------------------------------------------------------
    # Evaluate schedule
    # -----------------------------------------------------------------------

    def evaluate_schedule(
        self,
        schedule: list[ScheduledOrder],
        actual_fills: list[dict[str, Any]],
    ) -> ScheduleEvaluation:
        """
        Compare a planned schedule against actual fills.

        Parameters
        ----------
        schedule : list[ScheduledOrder]
        actual_fills : list[dict]
            Each dict must have: qty, fill_price, fill_time (datetime or str).

        Returns
        -------
        ScheduleEvaluation
        """
        if not schedule:
            raise ValueError("schedule is empty")

        sym = schedule[0].order.sym
        side = schedule[0].order.side
        total_planned_qty = sum(s.order.qty for s in schedule)

        # Planned VWAP (using estimated price if available)
        planned_vwap = float(
            np.mean([s.order.limit_price or 0.0 for s in schedule if s.order.limit_price])
        ) or 0.0

        # Actual fills processing
        total_filled_qty = 0.0
        total_notional = 0.0
        fill_times: list[datetime] = []

        for f in actual_fills:
            qty = float(f.get("qty", 0.0))
            price = float(f.get("fill_price", 0.0))
            fill_time = f.get("fill_time")
            if isinstance(fill_time, str):
                fill_time = datetime.fromisoformat(fill_time)
            total_filled_qty += qty
            total_notional += qty * price
            if fill_time:
                fill_times.append(fill_time)

        actual_vwap = total_notional / total_filled_qty if total_filled_qty > 0 else 0.0

        # VWAP slippage
        direction = 1 if side == "buy" else -1
        vwap_slippage = 0.0
        if actual_vwap > 0 and planned_vwap > 0:
            vwap_slippage = direction * (actual_vwap - planned_vwap) / planned_vwap * 10_000

        # Horizon
        planned_start = schedule[0].expected_time
        planned_end = schedule[-1].expected_time
        planned_horizon = (planned_end - planned_start).total_seconds() / 60

        actual_horizon = 0.0
        if len(fill_times) >= 2:
            actual_horizon = (max(fill_times) - min(fill_times)).total_seconds() / 60

        # Timing errors: compare planned vs actual fill times per slice
        timing_errors: list[float] = []
        n_matched = min(len(schedule), len(actual_fills))
        for i in range(n_matched):
            planned_t = schedule[i].expected_time
            fill_time = actual_fills[i].get("fill_time")
            if fill_time:
                if isinstance(fill_time, str):
                    fill_time = datetime.fromisoformat(fill_time)
                timing_errors.append((fill_time - planned_t).total_seconds())

        # Schedule adherence: fraction of slices where timing error < 1 interval
        interval_s = (
            (schedule[1].expected_time - schedule[0].expected_time).total_seconds()
            if len(schedule) > 1 else 300.0
        )
        good_timing = sum(1 for e in timing_errors if abs(e) < interval_s)
        adherence_pct = (good_timing / len(timing_errors) * 100) if timing_errors else 0.0

        unfilled = max(0, len(schedule) - len(actual_fills))
        total_slip_dollars = vwap_slippage / 10_000 * total_notional

        return ScheduleEvaluation(
            sym=sym,
            planned_n_slices=len(schedule),
            actual_n_fills=len(actual_fills),
            planned_vwap=planned_vwap,
            actual_vwap=actual_vwap,
            vwap_slippage_bps=vwap_slippage,
            planned_horizon_minutes=planned_horizon,
            actual_horizon_minutes=actual_horizon,
            schedule_adherence_pct=adherence_pct,
            total_slippage_bps=vwap_slippage,
            total_slippage_dollars=total_slip_dollars,
            timing_errors=timing_errors,
            unfilled_slices=unfilled,
        )

    # -----------------------------------------------------------------------
    # Convenience: check whether an order needs splitting
    # -----------------------------------------------------------------------

    def needs_split(self, notional: float) -> bool:
        """Return True if the order notional exceeds the Alpaca limit."""
        return notional > self.max_per_order

    def n_orders_required(self, notional: float) -> int:
        """How many child orders are needed for this notional?"""
        return math.ceil(notional / self.max_per_order)

    # -----------------------------------------------------------------------
    # Summary printing
    # -----------------------------------------------------------------------

    @staticmethod
    def print_schedule(scheduled: list[ScheduledOrder]) -> None:
        """Pretty-print a schedule to stdout."""
        if not scheduled:
            print("Empty schedule")
            return

        print(f"\n{'Idx':>4} {'Time':>22} {'Qty':>12} {'Notional':>14} "
              f"{'Impact':>10} {'Cum%':>8} {'Notes'}")
        print("-" * 90)
        for s in scheduled:
            o = s.order
            print(
                f"{o.slice_index:>4d} "
                f"{s.expected_time.strftime('%Y-%m-%d %H:%M:%S'):>22} "
                f"{o.qty:>12.4f} "
                f"${o.notional:>12,.0f} "
                f"{s.expected_impact_bps:>9.2f}b "
                f"{s.cumulative_qty_pct*100:>7.1f}% "
                f"{s.notes}"
            )
        total_qty = sum(s.order.qty for s in scheduled)
        total_impact = sum(s.expected_impact_bps * s.order.qty for s in scheduled) / max(total_qty, 1)
        print("-" * 90)
        print(
            f"{'TOTAL':>4} {'':>22} {total_qty:>12.4f} "
            f"${sum(s.order.notional for s in scheduled):>12,.0f} "
            f"{total_impact:>9.2f}b"
        )
        print()
