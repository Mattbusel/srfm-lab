"""
execution/cost_model.py
=======================
Transaction cost modeling for equities and crypto.

Covers:
  - Venue fee schedules (commission in bps)
  - Square-root market impact (Almgren et al.)
  - Temporary / permanent impact decomposition
  - Spread + timing slippage
  - Full CostEstimator with Almgren-Chriss optimal execution schedule
  - CostTracker for recording actual fills and computing cost efficiency

Usage::

    from execution.cost_model import CostEstimator, VENUES

    est = CostEstimator()
    result = est.estimate(
        symbol="SPY",
        order_size_usd=50_000,
        side="buy",
        venue="alpaca_equity",
        adv_usd=25_000_000,
        sigma_daily=0.012,
    )
    print(result.total_bps)
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("execution.cost_model")

# ---------------------------------------------------------------------------
# Venue configuration
# ---------------------------------------------------------------------------

@dataclass
class VenueConfig:
    """Fee schedule and size limits for a single trading venue.

    Attributes
    ----------
    name : str
        Human-readable venue identifier.
    maker_fee_bps : float
        Commission (in basis points) paid on limit orders that add liquidity.
    taker_fee_bps : float
        Commission (in basis points) paid on market/aggressive limit orders.
    min_order_size : float
        Minimum notional order size in USD.
    max_order_pct_adv : float
        Maximum order size as a fraction of 30-day average daily volume.
        E.g. 0.05 = 5 % ADV.
    """
    name:               str
    maker_fee_bps:      float
    taker_fee_bps:      float
    min_order_size:     float   = 1.0
    max_order_pct_adv:  float   = 0.05


VENUES: dict[str, VenueConfig] = {
    "alpaca_equity": VenueConfig(
        name="alpaca_equity",
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        min_order_size=1.0,
        max_order_pct_adv=0.05,
    ),
    "alpaca_crypto": VenueConfig(
        name="alpaca_crypto",
        maker_fee_bps=15.0,
        taker_fee_bps=25.0,
        min_order_size=1.0,
        max_order_pct_adv=0.05,
    ),
    "binance_spot": VenueConfig(
        name="binance_spot",
        maker_fee_bps=7.0,
        taker_fee_bps=10.0,
        min_order_size=10.0,
        max_order_pct_adv=0.10,
    ),
    "coinbase": VenueConfig(
        name="coinbase",
        maker_fee_bps=50.0,
        taker_fee_bps=50.0,
        min_order_size=1.0,
        max_order_pct_adv=0.03,
    ),
}


# ---------------------------------------------------------------------------
# Market impact models
# ---------------------------------------------------------------------------

class ImpactModel:
    """
    Square-root market impact model (Almgren et al.).

    impact_bps = eta * sigma * sqrt(order_size / ADV) * 10_000

    Parameters
    ----------
    eta : float
        Empirical constant calibrated from historical fills.
        Default 0.1 is a widely-used industry estimate.
    """

    ETA_DEFAULT: float = 0.1

    def __init__(self, eta: float = ETA_DEFAULT) -> None:
        self.eta = eta

    def estimate_bps(
        self,
        order_size_usd: float,
        adv_usd:        float,
        sigma_daily:    float,
    ) -> float:
        """Return expected market impact in basis points.

        Parameters
        ----------
        order_size_usd : float
            Notional value of the order in USD.
        adv_usd : float
            30-day average daily volume in USD.
        sigma_daily : float
            Daily return volatility as a fraction (e.g. 0.02 = 2 %).

        Returns
        -------
        float
            Expected impact in bps.
        """
        if adv_usd <= 0 or sigma_daily <= 0:
            return 0.0
        participation = order_size_usd / adv_usd
        impact_frac   = self.eta * sigma_daily * math.sqrt(participation)
        return impact_frac * 10_000.0

    def calibrate(self, fills: list[tuple[float, float]]) -> None:
        """OLS-calibrate eta from a list of (model_impact_bps, actual_impact_bps) pairs."""
        if len(fills) < 5:
            return
        # model_impact = eta * sigma * sqrt(Q/V) * 1e4
        # actual = eta_new * model / eta_old  (linear, no intercept)
        num = sum(m * a for m, a in fills)
        den = sum(m * m for m, _ in fills)
        if den < 1e-12:
            return
        # ratio scales eta
        scale      = num / den
        old_eta    = self.eta
        self.eta   = old_eta * scale
        log.info("ImpactModel: eta recalibrated %.4f -> %.4f (n=%d)", old_eta, self.eta, len(fills))


class TemporaryImpact:
    """
    Temporary (transient) component of market impact.

    Models the price reversion after order completion.
    Decay follows an exponential with the given half-life in bars.

    decay(t) = initial_impact * exp(-lambda * t)
    where lambda = ln(2) / half_life_bars
    """

    def __init__(self, half_life_bars: float = 5.0) -> None:
        self.half_life_bars = half_life_bars
        self._lambda        = math.log(2.0) / half_life_bars

    def estimate_bps(
        self,
        order_size_usd: float,
        adv_usd:        float,
        sigma_daily:    float,
        eta:            float = ImpactModel.ETA_DEFAULT,
    ) -> float:
        """Temporary impact at execution time (t=0) in bps."""
        if adv_usd <= 0 or sigma_daily <= 0:
            return 0.0
        participation = order_size_usd / adv_usd
        impact_frac   = eta * sigma_daily * math.sqrt(participation)
        return impact_frac * 10_000.0

    def decay(self, initial_bps: float, elapsed_bars: float) -> float:
        """Return residual temporary impact after *elapsed_bars* bars."""
        if elapsed_bars < 0:
            return initial_bps
        return initial_bps * math.exp(-self._lambda * elapsed_bars)

    def remaining_fraction(self, elapsed_bars: float) -> float:
        """Fraction of temporary impact still present after *elapsed_bars*."""
        return math.exp(-self._lambda * elapsed_bars)


class PermanentImpact:
    """
    Permanent (lasting) component of market impact.

    Convention: permanent impact = permanent_fraction * temporary_impact.
    Default: 50 % of temporary impact persists permanently.
    """

    def __init__(self, permanent_fraction: float = 0.5) -> None:
        self.permanent_fraction = permanent_fraction

    def estimate_bps(
        self,
        temporary_impact_bps: float,
    ) -> float:
        """Permanent impact given the temporary impact estimate."""
        return self.permanent_fraction * temporary_impact_bps

    def total_impact_bps(
        self,
        order_size_usd: float,
        adv_usd:        float,
        sigma_daily:    float,
        eta:            float = ImpactModel.ETA_DEFAULT,
    ) -> tuple[float, float]:
        """Return (permanent_bps, temporary_bps) decomposition."""
        temp = TemporaryImpact().estimate_bps(order_size_usd, adv_usd, sigma_daily, eta)
        perm = self.estimate_bps(temp)
        return perm, temp


# ---------------------------------------------------------------------------
# Slippage model
# ---------------------------------------------------------------------------

class SlippageModel:
    """
    Spread + timing slippage estimator.

    Components
    ----------
    1. Spread cost: half bid-ask spread paid as taker.
    2. Timing slippage: 0.3 * sigma * sqrt(execution_time_bars)
       -- adverse price drift while working the order.

    Both are expressed in basis points.
    """

    TIMING_COEFF: float = 0.3

    def spread_cost_bps(self, half_spread_bps: float) -> float:
        """Cost in bps of crossing half the bid-ask spread (taker only)."""
        return half_spread_bps

    def timing_slippage_bps(
        self,
        sigma_daily:          float,
        execution_time_bars:  float,
        bars_per_day:         float = 26.0,   # 6.5 h / 15-min bars
    ) -> float:
        """
        Timing slippage from adverse drift while the order is being worked.

        sigma_daily is converted to per-bar vol using sqrt(1/bars_per_day).
        """
        if sigma_daily <= 0 or execution_time_bars <= 0:
            return 0.0
        sigma_bar    = sigma_daily / math.sqrt(bars_per_day)
        slippage_frac = self.TIMING_COEFF * sigma_bar * math.sqrt(execution_time_bars)
        return slippage_frac * 10_000.0

    def estimate_bps(
        self,
        half_spread_bps:      float,
        sigma_daily:          float,
        execution_time_bars:  float,
        bars_per_day:         float = 26.0,
    ) -> float:
        """Total slippage in bps (spread + timing)."""
        return (
            self.spread_cost_bps(half_spread_bps)
            + self.timing_slippage_bps(sigma_daily, execution_time_bars, bars_per_day)
        )


# ---------------------------------------------------------------------------
# CostEstimate value object
# ---------------------------------------------------------------------------

@dataclass
class CostEstimate:
    """
    Complete pre-trade cost estimate for a single order.

    All cost components are in basis points (1 bp = 0.01 %).
    dollar_cost is the absolute expected cost in USD.
    """
    symbol:          str
    venue:           str
    side:            str
    order_size_usd:  float

    commission_bps:  float = 0.0
    spread_cost_bps: float = 0.0
    impact_bps:      float = 0.0
    timing_bps:      float = 0.0
    total_bps:       float = 0.0
    dollar_cost:     float = 0.0

    # Breakdown for debugging
    permanent_impact_bps:  float = 0.0
    temporary_impact_bps:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol":               self.symbol,
            "venue":                self.venue,
            "side":                 self.side,
            "order_size_usd":       self.order_size_usd,
            "commission_bps":       round(self.commission_bps,  4),
            "spread_cost_bps":      round(self.spread_cost_bps, 4),
            "impact_bps":           round(self.impact_bps,      4),
            "timing_bps":           round(self.timing_bps,      4),
            "total_bps":            round(self.total_bps,       4),
            "dollar_cost":          round(self.dollar_cost,     4),
            "permanent_impact_bps": round(self.permanent_impact_bps, 4),
            "temporary_impact_bps": round(self.temporary_impact_bps, 4),
        }


# ---------------------------------------------------------------------------
# Almgren-Chriss optimal trajectory
# ---------------------------------------------------------------------------

def _almgren_chriss_trajectory(
    total_shares:    float,
    n_bars:          int,
    sigma:           float,
    eta:             float,
    gamma:           float,
    lam:             float,
) -> list[float]:
    """
    Compute the Almgren-Chriss optimal liquidation schedule.

    Minimises E[cost] + lambda * Var[cost] over n_bars periods.

    The closed-form solution gives trade sizes::

        x_k = X * sinh(kappa * (n - k)) / sinh(kappa * n)

    where kappa = arccosh(1 + 0.5 * kappa_tilde^2) and
    kappa_tilde^2 = (lam * sigma^2) / (0.5 * eta) (continuous limit scaling).

    Parameters
    ----------
    total_shares : float    Total quantity to execute.
    n_bars : int            Number of execution intervals.
    sigma : float           Per-bar volatility (fraction).
    eta : float             Temporary impact coefficient.
    gamma : float           Permanent impact coefficient.
    lam : float             Risk-aversion parameter (lambda).

    Returns
    -------
    list[float]
        Trade size for each bar, length n_bars, summing to total_shares.
    """
    if n_bars <= 1 or total_shares <= 0:
        return [total_shares]

    # Avoid division-by-zero
    eta_eff = max(eta, 1e-9)

    # kappa_tilde^2 in discrete-time Almgren-Chriss
    kappa_tilde_sq = (lam * sigma ** 2) / (0.5 * eta_eff)
    # Discrete kappa via arccosh
    arg = 1.0 + 0.5 * kappa_tilde_sq
    kappa = math.acosh(max(arg, 1.0 + 1e-12))

    # Optimal trajectory: remaining inventory at bar k
    # x(k) = X * sinh(kappa * (n - k)) / sinh(kappa * n)
    n = n_bars
    denom_sinh = math.sinh(kappa * n)
    if denom_sinh < 1e-12:
        # Degenerate: uniform schedule
        uniform = total_shares / n
        return [uniform] * n

    inventories = [
        total_shares * math.sinh(kappa * (n - k)) / denom_sinh
        for k in range(n + 1)
    ]
    # Trade sizes = difference between consecutive inventories
    trades = [inventories[k] - inventories[k + 1] for k in range(n)]

    # Normalise to ensure sum == total_shares exactly
    total = sum(trades)
    if total > 0:
        scale = total_shares / total
        trades = [t * scale for t in trades]

    return trades


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------

class CostEstimator:
    """
    Unified pre-trade cost estimator.

    Combines commission, spread, market impact, and timing slippage into
    a single CostEstimate.  Also provides an Almgren-Chriss optimal
    execution schedule for large orders.

    Parameters
    ----------
    half_spread_bps : float
        Default assumed half bid-ask spread in bps (used when live
        quote is not available).
    execution_time_bars : float
        Default assumed execution duration in 15-min bars.
    risk_aversion : float
        Almgren-Chriss risk-aversion parameter lambda.  Higher values
        produce faster (more aggressive) execution schedules.
    """

    DEFAULT_HALF_SPREAD_BPS:   float = 2.0
    DEFAULT_EXEC_BARS:         float = 1.0
    DEFAULT_RISK_AVERSION:     float = 1e-6

    def __init__(
        self,
        half_spread_bps:      float = DEFAULT_HALF_SPREAD_BPS,
        execution_time_bars:  float = DEFAULT_EXEC_BARS,
        risk_aversion:        float = DEFAULT_RISK_AVERSION,
    ) -> None:
        self._half_spread_bps     = half_spread_bps
        self._execution_time_bars = execution_time_bars
        self._risk_aversion       = risk_aversion

        self._impact_model    = ImpactModel()
        self._slippage_model  = SlippageModel()
        self._temp_impact     = TemporaryImpact()
        self._perm_impact     = PermanentImpact()

    # ------------------------------------------------------------------
    # Primary estimate method
    # ------------------------------------------------------------------

    def estimate(
        self,
        symbol:          str,
        order_size_usd:  float,
        side:            str,
        venue:           str,
        adv_usd:         float,
        sigma_daily:     float,
        half_spread_bps: Optional[float] = None,
        exec_bars:       Optional[float] = None,
        is_maker:        bool = False,
    ) -> CostEstimate:
        """
        Compute a complete pre-trade cost estimate.

        Parameters
        ----------
        symbol : str
        order_size_usd : float   Notional value in USD.
        side : str               "buy" or "sell".
        venue : str              Key into VENUES dict.
        adv_usd : float          30-day average daily volume in USD.
        sigma_daily : float      Daily volatility fraction.
        half_spread_bps : float | None  Override default half spread.
        exec_bars : float | None        Override default exec time.
        is_maker : bool          Use maker fee if True.

        Returns
        -------
        CostEstimate
        """
        venue_cfg = VENUES.get(venue)
        if venue_cfg is None:
            raise ValueError(f"Unknown venue '{venue}'. Known: {list(VENUES)}")

        spread    = half_spread_bps if half_spread_bps is not None else self._half_spread_bps
        exec_t    = exec_bars if exec_bars is not None else self._execution_time_bars

        # 1. Commission
        commission_bps = venue_cfg.maker_fee_bps if is_maker else venue_cfg.taker_fee_bps

        # 2. Spread cost
        spread_bps = self._slippage_model.spread_cost_bps(spread)

        # 3. Market impact (temporary + permanent)
        temp_bps = self._temp_impact.estimate_bps(order_size_usd, adv_usd, sigma_daily)
        perm_bps = self._perm_impact.estimate_bps(temp_bps)
        impact_bps = temp_bps + perm_bps

        # 4. Timing slippage
        timing_bps = self._slippage_model.timing_slippage_bps(sigma_daily, exec_t)

        total_bps  = commission_bps + spread_bps + impact_bps + timing_bps
        dollar_cost = order_size_usd * total_bps / 10_000.0

        return CostEstimate(
            symbol=symbol,
            venue=venue,
            side=side,
            order_size_usd=order_size_usd,
            commission_bps=commission_bps,
            spread_cost_bps=spread_bps,
            impact_bps=impact_bps,
            timing_bps=timing_bps,
            total_bps=total_bps,
            dollar_cost=dollar_cost,
            permanent_impact_bps=perm_bps,
            temporary_impact_bps=temp_bps,
        )

    # ------------------------------------------------------------------
    # Execution schedule optimisation
    # ------------------------------------------------------------------

    def optimize_execution_schedule(
        self,
        order_size:   float,
        adv:          float,
        target_bars:  int   = 8,
        sigma_daily:  float = 0.02,
        strategy:     str   = "almgren_chriss",
    ) -> list[float]:
        """
        Split an order across *target_bars* bars to minimise total cost.

        Strategies
        ----------
        "twap"
            Equal-sized slices (naive TWAP).
        "vwap"
            Volume-weighted slices using a typical intraday volume profile.
        "almgren_chriss"
            Optimal trajectory via Almgren-Chriss closed-form solution.
            Minimises E[cost] + lambda * Var[cost].

        Parameters
        ----------
        order_size : float   Total order notional (USD) or shares.
        adv : float          Average daily volume (same units as order_size).
        target_bars : int    Number of execution intervals.
        sigma_daily : float  Daily volatility fraction.
        strategy : str       One of "twap", "vwap", "almgren_chriss".

        Returns
        -------
        list[float]
            Fraction of total order to execute in each bar.
            Sums to 1.0.
        """
        if target_bars <= 0:
            return [1.0]

        if strategy == "twap":
            fracs = [1.0 / target_bars] * target_bars
            return fracs

        if strategy == "vwap":
            # Typical U-shaped intraday volume profile (simplified)
            # Higher volume at open and close, lower in the middle
            profile = _intraday_volume_profile(target_bars)
            total_v = sum(profile)
            return [v / total_v for v in profile]

        # Almgren-Chriss
        # Convert sigma_daily to per-bar sigma
        bars_per_day = 26.0  # 15-min bars in a 6.5h trading day
        sigma_bar    = sigma_daily / math.sqrt(bars_per_day)

        # Participation rate as proxy for "shares"
        total_shares = order_size / adv if adv > 0 else 1.0

        # eta and gamma in terms of participation
        eta   = self._impact_model.eta
        gamma = self._perm_impact.permanent_fraction * eta

        trades = _almgren_chriss_trajectory(
            total_shares   = total_shares,
            n_bars         = target_bars,
            sigma          = sigma_bar,
            eta            = eta,
            gamma          = gamma,
            lam            = self._risk_aversion,
        )

        # Normalise to fractions
        total = sum(trades)
        if total <= 0:
            return [1.0 / target_bars] * target_bars
        return [t / total for t in trades]

    # ------------------------------------------------------------------
    # Venue comparison
    # ------------------------------------------------------------------

    def cheapest_venue(
        self,
        symbol:         str,
        order_size_usd: float,
        side:           str,
        adv_usd:        float,
        sigma_daily:    float,
        candidates:     Optional[list[str]] = None,
    ) -> tuple[str, CostEstimate]:
        """
        Return the (venue_name, CostEstimate) with the lowest total cost.

        Parameters
        ----------
        candidates : list[str] | None
            Subset of VENUES to consider.  None means all.
        """
        venues = candidates if candidates else list(VENUES.keys())
        best_venue  = venues[0]
        best_est    = self.estimate(symbol, order_size_usd, side, venues[0], adv_usd, sigma_daily)
        for v in venues[1:]:
            est = self.estimate(symbol, order_size_usd, side, v, adv_usd, sigma_daily)
            if est.total_bps < best_est.total_bps:
                best_venue = v
                best_est   = est
        return best_venue, best_est


# ---------------------------------------------------------------------------
# Intraday volume profile helper
# ---------------------------------------------------------------------------

def _intraday_volume_profile(n_bars: int) -> list[float]:
    """
    Return a synthetic U-shaped intraday volume profile for *n_bars* bars.

    Volume is highest at the open (bar 0) and close (bar n-1) and lowest
    around mid-session.  Uses a cosine transformation.
    """
    if n_bars == 1:
        return [1.0]
    profile = []
    for i in range(n_bars):
        # Normalised position in [0, pi]
        x = math.pi * i / (n_bars - 1)
        # cos(x) == 1 at x=0 (open), -1 at x=pi (close); shift+scale to [0,1]
        # U-shape: high at endpoints, low in the middle
        vol = 0.5 * (1.0 + math.cos(x) ** 2)
        profile.append(max(vol, 0.05))
    return profile


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

_FILL_RECORD_FIELDS = (
    "ts", "symbol", "venue", "estimated_bps", "actual_bps", "slippage_bps"
)


@dataclass
class FillRecord:
    ts:            float
    symbol:        str
    venue:         str
    estimated_bps: float
    actual_bps:    float
    slippage_bps:  float   = field(init=False)

    def __post_init__(self) -> None:
        self.slippage_bps = self.actual_bps - self.estimated_bps


class CostTracker:
    """
    Records actual fills vs pre-trade cost estimates and generates reports.

    Maintains an in-memory ring buffer (last *window* fills) and optionally
    persists to SQLite.

    Metrics
    -------
    - Per-venue average realised vs estimated cost
    - Per-symbol average realised vs estimated cost
    - Rolling 30-day cost efficiency ratio (estimated / actual)
    - Implementation shortfall (actual_bps - estimated_bps)
    """

    _WINDOW_DAYS = 30

    def __init__(
        self,
        db_path:    Optional[Path] = None,
        max_memory: int = 10_000,
    ) -> None:
        self._db_path   = db_path
        self._records:  deque[FillRecord] = deque(maxlen=max_memory)
        self._lock      = threading.RLock()

        if db_path is not None:
            self._init_db(db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_fill(
        self,
        symbol:        str,
        estimated_bps: float,
        actual_bps:    float,
        venue:         str,
        ts:            Optional[float] = None,
    ) -> None:
        """
        Record a completed fill for cost tracking.

        Parameters
        ----------
        symbol : str
        estimated_bps : float   Pre-trade estimate total cost in bps.
        actual_bps : float      Realised total cost in bps (measured from
                                decision-time mid to fill price).
        venue : str
        ts : float | None       Unix timestamp; defaults to now.
        """
        record = FillRecord(
            ts=ts if ts is not None else time.time(),
            symbol=symbol,
            venue=venue,
            estimated_bps=estimated_bps,
            actual_bps=actual_bps,
        )
        with self._lock:
            self._records.append(record)

        if self._db_path is not None:
            self._persist_record(record)

    def get_slippage_report(
        self,
        since_ts: Optional[float] = None,
    ) -> dict:
        """
        Generate a cost efficiency report.

        Returns a dict with per-venue and per-symbol summaries plus
        a rolling 30-day efficiency ratio.

        Parameters
        ----------
        since_ts : float | None
            Filter fills to those recorded after this Unix timestamp.
            Defaults to 30 days ago.

        Returns
        -------
        dict with keys:
            "per_venue"  : dict[venue, {avg_estimated, avg_actual, avg_slippage, n}]
            "per_symbol" : dict[symbol, {avg_estimated, avg_actual, avg_slippage, n}]
            "efficiency_ratio" : float   (estimated / actual; 1.0 = perfect)
            "n_fills"    : int
            "since_ts"   : float
        """
        cutoff = since_ts if since_ts is not None else time.time() - self._WINDOW_DAYS * 86400.0

        with self._lock:
            records = [r for r in self._records if r.ts >= cutoff]

        if not records:
            return {
                "per_venue":        {},
                "per_symbol":       {},
                "efficiency_ratio": 1.0,
                "n_fills":          0,
                "since_ts":         cutoff,
            }

        # Per-venue
        venue_buckets: dict[str, list[FillRecord]] = defaultdict(list)
        sym_buckets:   dict[str, list[FillRecord]] = defaultdict(list)
        for r in records:
            venue_buckets[r.venue].append(r)
            sym_buckets[r.symbol].append(r)

        def _summarise(recs: list[FillRecord]) -> dict:
            n   = len(recs)
            avg_est = sum(r.estimated_bps for r in recs) / n
            avg_act = sum(r.actual_bps    for r in recs) / n
            avg_slp = sum(r.slippage_bps  for r in recs) / n
            return {
                "avg_estimated_bps": round(avg_est, 4),
                "avg_actual_bps":    round(avg_act, 4),
                "avg_slippage_bps":  round(avg_slp, 4),
                "n": n,
            }

        per_venue  = {v: _summarise(recs) for v, recs in venue_buckets.items()}
        per_symbol = {s: _summarise(recs) for s, recs in sym_buckets.items()}

        total_est = sum(r.estimated_bps for r in records)
        total_act = sum(r.actual_bps    for r in records)
        eff_ratio = (total_est / total_act) if total_act != 0 else 1.0

        return {
            "per_venue":        per_venue,
            "per_symbol":       per_symbol,
            "efficiency_ratio": round(eff_ratio, 6),
            "n_fills":          len(records),
            "since_ts":         cutoff,
        }

    def efficiency_ratio(self, window_days: int = 30) -> float:
        """Rolling cost efficiency ratio: estimated_bps / actual_bps."""
        cutoff = time.time() - window_days * 86400.0
        with self._lock:
            recent = [r for r in self._records if r.ts >= cutoff]
        if not recent:
            return 1.0
        total_est = sum(r.estimated_bps for r in recent)
        total_act = sum(r.actual_bps    for r in recent)
        return (total_est / total_act) if total_act > 0 else 1.0

    def clear(self) -> None:
        """Remove all in-memory records (does not affect SQLite)."""
        with self._lock:
            self._records.clear()

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(db_path))
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS fill_records (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            REAL    NOT NULL,
                symbol        TEXT    NOT NULL,
                venue         TEXT    NOT NULL,
                estimated_bps REAL    NOT NULL,
                actual_bps    REAL    NOT NULL,
                slippage_bps  REAL    NOT NULL
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_fr_ts ON fill_records (ts)")
        con.commit()
        con.close()

    def _persist_record(self, r: FillRecord) -> None:
        try:
            con = sqlite3.connect(str(self._db_path))
            con.execute(
                "INSERT INTO fill_records (ts,symbol,venue,estimated_bps,actual_bps,slippage_bps) "
                "VALUES (?,?,?,?,?,?)",
                (r.ts, r.symbol, r.venue, r.estimated_bps, r.actual_bps, r.slippage_bps),
            )
            con.commit()
            con.close()
        except Exception as exc:
            log.warning("CostTracker: failed to persist record: %s", exc)

    def load_from_db(self, since_ts: Optional[float] = None) -> int:
        """
        Load records from SQLite into the in-memory deque.

        Returns number of records loaded.
        """
        if self._db_path is None or not self._db_path.exists():
            return 0
        cutoff = since_ts if since_ts is not None else 0.0
        con = sqlite3.connect(str(self._db_path))
        rows = con.execute(
            "SELECT ts, symbol, venue, estimated_bps, actual_bps "
            "FROM fill_records WHERE ts >= ? ORDER BY ts",
            (cutoff,),
        ).fetchall()
        con.close()
        with self._lock:
            for ts, symbol, venue, est, act in rows:
                self._records.append(
                    FillRecord(ts=ts, symbol=symbol, venue=venue,
                               estimated_bps=est, actual_bps=act)
                )
        return len(rows)
