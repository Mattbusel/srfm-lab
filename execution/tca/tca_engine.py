# execution/tca/tca_engine.py -- Transaction Cost Analysis engine for SRFM
# Computes implementation shortfall, VWAP slippage, spread cost, and market impact.

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Represents a single trade execution record submitted for TCA."""
    symbol: str
    side: str                    # "BUY" or "SELL"
    order_qty: float             # shares/contracts ordered
    filled_qty: float            # shares/contracts actually filled
    order_price: float           # limit price or mid at order submission
    fill_price: float            # actual average fill price
    order_time: datetime         # timestamp when order was sent to venue
    fill_time: datetime          # timestamp when fill confirmation received
    venue: str                   # execution venue identifier
    strategy: str                # strategy that generated the order
    order_type: str              # "LIMIT", "MARKET", "IOC", "MOC", etc.
    decision_price: Optional[float] = None   # price at signal generation
    arrival_price: Optional[float] = None    # price at order submission (mid)
    market_volume: Optional[float] = None    # volume traded in fill window
    interval_vwap: Optional[float] = None    # VWAP over the fill window
    interval_twap: Optional[float] = None    # TWAP over the fill window
    close_price: Optional[float] = None      # day close price
    bid_at_fill: Optional[float] = None      # best bid at time of fill
    ask_at_fill: Optional[float] = None      # best ask at time of fill
    adv: Optional[float] = None              # average daily volume
    trade_id: Optional[str] = None           # unique trade identifier


@dataclass
class SlippageDecomposition:
    """Detailed decomposition of execution slippage into cost components."""
    spread_component: float       # bps -- cost from crossing half-spread
    market_impact_component: float  # bps -- permanent price impact from our order
    timing_component: float       # bps -- cost from delay between arrival and fill
    alpha_component: float        # bps -- estimated alpha leakage (Kyle lambda estimate)
    total_bps: float              # bps -- sum of all components


@dataclass
class TCAResult:
    """Full TCA result for a single trade."""
    # -- Core cost metrics (all in basis points relative to arrival price)
    implementation_shortfall_bps: float   # (fill - arrival) / arrival * 10000 * side_sign
    market_impact_bps: float              # permanent impact estimate
    timing_cost_bps: float                # delay cost: arrival to fill time price change
    spread_cost_bps: float                # half-spread paid
    total_cost_bps: float                 # sum of all cost components

    # -- Volume/participation metrics
    participation_rate: float             # filled_qty / market_volume_in_window (0..1)
    vwap_slippage_bps: float             # fill vs interval VWAP
    twap_slippage_bps: float             # fill vs interval TWAP
    close_slippage_bps: float            # fill vs day close price

    # -- Reference prices
    decision_price: float                 # price at signal generation
    arrival_price: float                  # price at order submission
    fill_price: float                     # actual average fill price
    benchmark_type: str                   # "ARRIVAL", "VWAP", "TWAP", "CLOSE"

    # -- Supplementary fields
    fill_rate: float                      # filled_qty / order_qty
    time_to_fill_ms: float               # milliseconds from order to fill
    slippage_decomposition: Optional[SlippageDecomposition] = None

    # -- Carry-through identifiers for aggregation
    symbol: str = ""
    side: str = ""
    strategy: str = ""
    venue: str = ""
    order_type: str = ""
    trade_id: Optional[str] = None
    trade_date: Optional[str] = None     # YYYY-MM-DD


@dataclass
class BatchTCAResult:
    """Aggregated TCA results across a batch of trades."""
    results: List[TCAResult]
    avg_is_bps: float
    avg_vwap_slippage_bps: float
    avg_spread_cost_bps: float
    avg_market_impact_bps: float
    avg_timing_cost_bps: float
    avg_total_cost_bps: float
    std_is_bps: float
    n_trades: int
    n_partial_fills: int
    fill_rate: float                     # average fill rate across trades
    by_symbol: Dict[str, float] = field(default_factory=dict)    # avg IS per symbol
    by_venue: Dict[str, float] = field(default_factory=dict)     # avg IS per venue
    by_strategy: Dict[str, float] = field(default_factory=dict)  # avg IS per strategy


@dataclass
class DailySummary:
    """Per-day TCA summary report."""
    date: str
    n_trades: int
    total_notional: float
    avg_is_bps: float
    avg_vwap_slippage_bps: float
    avg_spread_cost_bps: float
    avg_market_impact_bps: float
    avg_participation_rate: float
    avg_fill_rate: float
    best_venue: str
    worst_venue: str
    top_cost_symbols: List[str]          # symbols with highest avg IS
    by_strategy: Dict[str, float]        # avg IS per strategy


# ---------------------------------------------------------------------------
# Market data container (lightweight -- no external deps)
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    """Snapshot of market data for TCA computations."""
    symbol: str
    interval_vwap: Optional[float] = None
    interval_twap: Optional[float] = None
    close_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    market_volume: Optional[float] = None
    adv: Optional[float] = None
    sigma: Optional[float] = None        # realized volatility (daily, fractional)


# ---------------------------------------------------------------------------
# Benchmark implementations
# ---------------------------------------------------------------------------

class ArrivalPriceBenchmark:
    """Benchmark using the mid-price at order arrival (submission) time."""

    def compute(self, trade: TradeRecord, market_data: MarketData) -> float:
        """Return the arrival price to use as IS benchmark."""
        if trade.arrival_price is not None and trade.arrival_price > 0.0:
            return trade.arrival_price
        # Fall back to order_price if arrival_price not supplied
        return trade.order_price

    @property
    def name(self) -> str:
        return "ARRIVAL"


class VWAPBenchmark:
    """Benchmark using the volume-weighted average price over the fill window."""

    def compute(self, trade: TradeRecord, market_data: MarketData) -> float:
        """Return interval VWAP as benchmark price."""
        if trade.interval_vwap is not None and trade.interval_vwap > 0.0:
            return trade.interval_vwap
        if market_data.interval_vwap is not None and market_data.interval_vwap > 0.0:
            return market_data.interval_vwap
        # Degrade gracefully to fill price when VWAP unavailable
        return trade.fill_price

    @property
    def name(self) -> str:
        return "VWAP"


class TWAPBenchmark:
    """Benchmark using the time-weighted average price over the fill window."""

    def compute(self, trade: TradeRecord, market_data: MarketData) -> float:
        """Return interval TWAP as benchmark price."""
        if trade.interval_twap is not None and trade.interval_twap > 0.0:
            return trade.interval_twap
        if market_data.interval_twap is not None and market_data.interval_twap > 0.0:
            return market_data.interval_twap
        return trade.fill_price

    @property
    def name(self) -> str:
        return "TWAP"


class CloseBenchmark:
    """Benchmark using the day's closing price."""

    def compute(self, trade: TradeRecord, market_data: MarketData) -> float:
        """Return close price as benchmark."""
        if trade.close_price is not None and trade.close_price > 0.0:
            return trade.close_price
        if market_data.close_price is not None and market_data.close_price > 0.0:
            return market_data.close_price
        return trade.fill_price

    @property
    def name(self) -> str:
        return "CLOSE"


# ---------------------------------------------------------------------------
# Slippage decomposition helpers
# ---------------------------------------------------------------------------

def _side_sign(side: str) -> float:
    """Return +1 for BUY (higher fill = cost) and -1 for SELL."""
    return 1.0 if side.upper() == "BUY" else -1.0


def _safe_bps(numerator: float, denominator: float) -> float:
    """Safely compute basis points, returning 0.0 on zero denominator."""
    if denominator == 0.0 or not math.isfinite(denominator):
        return 0.0
    result = (numerator / denominator) * 10_000.0
    return result if math.isfinite(result) else 0.0


def _estimate_kyle_lambda(
    trade: TradeRecord,
    market_data: MarketData,
) -> float:
    """
    Estimate Kyle's lambda (price impact per unit of signed order flow).
    lambda = sigma / (2 * sqrt(ADV))
    where sigma is daily return volatility and ADV is average daily volume.
    Returns lambda in price-per-share units.
    """
    sigma = market_data.sigma if market_data.sigma is not None else 0.02
    adv = trade.adv if trade.adv is not None else market_data.adv
    if adv is None or adv <= 0.0:
        adv = max(trade.filled_qty * 20.0, 1.0)   # rough fallback
    arrival = trade.arrival_price if trade.arrival_price else trade.order_price
    if arrival <= 0.0:
        return 0.0
    # lambda in fractional terms, then convert to price per share
    lambda_frac = sigma / (2.0 * math.sqrt(adv))
    return lambda_frac * arrival


def decompose_slippage(
    trade: TradeRecord,
    market_data: MarketData,
) -> SlippageDecomposition:
    """
    Decompose total slippage into four components.

    1. spread_component    -- half-spread paid at execution
    2. market_impact_component -- permanent impact from Kyle lambda
    3. timing_component    -- price drift from order submission to fill
    4. alpha_component     -- residual (fill vs model-predicted price)

    All components are in basis points relative to arrival price.
    """
    sign = _side_sign(trade.side)
    arrival = trade.arrival_price if trade.arrival_price else trade.order_price
    if arrival <= 0.0:
        return SlippageDecomposition(
            spread_component=0.0,
            market_impact_component=0.0,
            timing_component=0.0,
            alpha_component=0.0,
            total_bps=0.0,
        )

    # -- Spread component: half the quoted spread
    spread_bps = 0.0
    if trade.bid_at_fill is not None and trade.ask_at_fill is not None:
        spread = trade.ask_at_fill - trade.bid_at_fill
        mid = (trade.ask_at_fill + trade.bid_at_fill) / 2.0
        if mid > 0.0:
            half_spread_frac = (spread / 2.0) / mid
            spread_bps = half_spread_frac * 10_000.0
    else:
        # Estimate as 2 bps default when no quote data available
        spread_bps = 2.0

    # -- Market impact component via Kyle's lambda
    lam = _estimate_kyle_lambda(trade, market_data)
    if lam > 0.0 and arrival > 0.0:
        impact_price = lam * trade.filled_qty
        market_impact_bps = _safe_bps(impact_price, arrival)
    else:
        market_impact_bps = 0.0

    # -- Timing component: price movement from arrival to some intermediate point
    # Approximate as IS minus the spread and impact components
    total_is = _safe_bps(
        sign * (trade.fill_price - arrival), arrival
    )
    timing_component = total_is - spread_bps - market_impact_bps

    # -- Alpha component: fill vs expected_price = arrival + timing_drift + impact
    # We define alpha leakage as any residual after accounting for all modeled costs
    # Positive alpha_component means adverse selection against us
    expected_price_without_order = (
        arrival + sign * (market_impact_bps / 10_000.0) * arrival
    )
    alpha_bps = _safe_bps(
        sign * (trade.fill_price - expected_price_without_order),
        arrival,
    ) - spread_bps
    # Clip timing and alpha to avoid double-counting -- assign residual to timing
    timing_component_clipped = max(timing_component, 0.0)
    alpha_clipped = max(alpha_bps - timing_component_clipped, 0.0)

    total = spread_bps + market_impact_bps + timing_component_clipped + alpha_clipped

    return SlippageDecomposition(
        spread_component=spread_bps,
        market_impact_component=market_impact_bps,
        timing_component=timing_component_clipped,
        alpha_component=alpha_clipped,
        total_bps=total,
    )


# ---------------------------------------------------------------------------
# Main TCA engine
# ---------------------------------------------------------------------------

class TCAEngine:
    """
    Main entry point for transaction cost analysis.

    Usage:
        engine = TCAEngine()
        result = engine.analyze_trade(trade, market_data)
        batch = engine.analyze_batch(trades, market_data_map)
        summary = engine.daily_summary("2026-04-07")
    """

    def __init__(
        self,
        default_benchmark: str = "ARRIVAL",
        store=None,                          # optional TCAStore for persistence
    ) -> None:
        self.default_benchmark = default_benchmark.upper()
        self.store = store
        self._benchmarks: Dict[str, object] = {
            "ARRIVAL": ArrivalPriceBenchmark(),
            "VWAP": VWAPBenchmark(),
            "TWAP": TWAPBenchmark(),
            "CLOSE": CloseBenchmark(),
        }
        self._results_cache: List[TCAResult] = []

    # ------------------------------------------------------------------
    # Single-trade analysis
    # ------------------------------------------------------------------

    def analyze_trade(
        self,
        trade: TradeRecord,
        market_data: Optional[MarketData] = None,
        benchmark: Optional[str] = None,
    ) -> TCAResult:
        """
        Compute full TCA for a single TradeRecord.

        Parameters
        ----------
        trade        : TradeRecord with fill details
        market_data  : MarketData snapshot; built from trade fields if None
        benchmark    : override default benchmark type

        Returns
        -------
        TCAResult with all cost metrics populated
        """
        if market_data is None:
            market_data = self._market_data_from_trade(trade)

        bench_key = (benchmark or self.default_benchmark).upper()
        bench = self._benchmarks.get(bench_key, self._benchmarks["ARRIVAL"])
        benchmark_price = bench.compute(trade, market_data)

        sign = _side_sign(trade.side)

        # Arrival price for IS calculation
        arrival = trade.arrival_price if trade.arrival_price else trade.order_price
        if arrival <= 0.0:
            arrival = trade.fill_price if trade.fill_price > 0.0 else 1.0

        # Decision price fallback
        decision = trade.decision_price if trade.decision_price else arrival

        # -- Implementation Shortfall (vs arrival price)
        is_bps = _safe_bps(sign * (trade.fill_price - arrival), arrival)

        # -- VWAP slippage
        vwap = (
            trade.interval_vwap
            if trade.interval_vwap and trade.interval_vwap > 0.0
            else market_data.interval_vwap
        )
        vwap_slip_bps = _safe_bps(
            sign * (trade.fill_price - vwap), vwap
        ) if vwap and vwap > 0.0 else 0.0

        # -- TWAP slippage
        twap = (
            trade.interval_twap
            if trade.interval_twap and trade.interval_twap > 0.0
            else market_data.interval_twap
        )
        twap_slip_bps = _safe_bps(
            sign * (trade.fill_price - twap), twap
        ) if twap and twap > 0.0 else 0.0

        # -- Close slippage
        close = (
            trade.close_price
            if trade.close_price and trade.close_price > 0.0
            else market_data.close_price
        )
        close_slip_bps = _safe_bps(
            sign * (trade.fill_price - close), close
        ) if close and close > 0.0 else 0.0

        # -- Spread cost: half-spread paid
        spread_bps = self._compute_spread_cost(trade, arrival)

        # -- Participation rate
        mkt_vol = (
            trade.market_volume
            if trade.market_volume and trade.market_volume > 0.0
            else market_data.market_volume
        )
        participation = (
            trade.filled_qty / mkt_vol if mkt_vol and mkt_vol > 0.0 else 0.0
        )

        # -- Timing cost: price drift from order submission to fill
        timing_bps = self._compute_timing_cost(trade, arrival, market_data)

        # -- Market impact
        lam = _estimate_kyle_lambda(trade, market_data)
        if lam > 0.0 and arrival > 0.0:
            impact_bps = _safe_bps(lam * trade.filled_qty, arrival)
        else:
            impact_bps = max(is_bps - spread_bps - timing_bps, 0.0)

        # -- Total cost
        total_bps = spread_bps + impact_bps + timing_bps

        # -- Fill metrics
        fill_rate = (
            trade.filled_qty / trade.order_qty
            if trade.order_qty > 0.0
            else 0.0
        )
        dt_ms = self._fill_time_ms(trade)

        # -- Slippage decomposition
        decomp = decompose_slippage(trade, market_data)

        # -- Trade date
        trade_date = trade.order_time.strftime("%Y-%m-%d") if trade.order_time else None

        result = TCAResult(
            implementation_shortfall_bps=is_bps,
            market_impact_bps=impact_bps,
            timing_cost_bps=timing_bps,
            spread_cost_bps=spread_bps,
            total_cost_bps=total_bps,
            participation_rate=min(participation, 1.0),
            vwap_slippage_bps=vwap_slip_bps,
            twap_slippage_bps=twap_slip_bps,
            close_slippage_bps=close_slip_bps,
            decision_price=decision,
            arrival_price=arrival,
            fill_price=trade.fill_price,
            benchmark_type=bench.name,
            fill_rate=min(fill_rate, 1.0),
            time_to_fill_ms=dt_ms,
            slippage_decomposition=decomp,
            symbol=trade.symbol,
            side=trade.side,
            strategy=trade.strategy,
            venue=trade.venue,
            order_type=trade.order_type,
            trade_id=trade.trade_id,
            trade_date=trade_date,
        )

        self._results_cache.append(result)

        if self.store is not None:
            try:
                self.store.insert(trade.trade_id or "", result)
            except Exception:
                pass   # never let persistence failures block analysis

        return result

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        trades: List[TradeRecord],
        market_data_map: Optional[Dict[str, MarketData]] = None,
        benchmark: Optional[str] = None,
    ) -> BatchTCAResult:
        """
        Analyze a list of trades and return aggregate statistics.

        Parameters
        ----------
        trades           : list of TradeRecord
        market_data_map  : dict keyed by trade.trade_id or trade.symbol
        benchmark        : benchmark type override

        Returns
        -------
        BatchTCAResult with per-trade results and aggregate metrics
        """
        if not trades:
            return BatchTCAResult(
                results=[],
                avg_is_bps=0.0,
                avg_vwap_slippage_bps=0.0,
                avg_spread_cost_bps=0.0,
                avg_market_impact_bps=0.0,
                avg_timing_cost_bps=0.0,
                avg_total_cost_bps=0.0,
                std_is_bps=0.0,
                n_trades=0,
                n_partial_fills=0,
                fill_rate=0.0,
            )

        results: List[TCAResult] = []
        for trade in trades:
            md = None
            if market_data_map is not None:
                md = (
                    market_data_map.get(trade.trade_id or "")
                    or market_data_map.get(trade.symbol)
                )
            results.append(self.analyze_trade(trade, md, benchmark))

        is_vals = [r.implementation_shortfall_bps for r in results]
        vwap_vals = [r.vwap_slippage_bps for r in results]
        spread_vals = [r.spread_cost_bps for r in results]
        impact_vals = [r.market_impact_bps for r in results]
        timing_vals = [r.timing_cost_bps for r in results]
        total_vals = [r.total_cost_bps for r in results]
        fill_vals = [r.fill_rate for r in results]

        n = len(results)
        avg_is = sum(is_vals) / n
        std_is = statistics.stdev(is_vals) if n > 1 else 0.0
        n_partial = sum(1 for r in results if r.fill_rate < 0.9999)

        # -- Per-symbol, per-venue, per-strategy aggregates
        by_symbol = self._group_avg(results, "symbol", is_vals)
        by_venue = self._group_avg(results, "venue", is_vals)
        by_strategy = self._group_avg(results, "strategy", is_vals)

        return BatchTCAResult(
            results=results,
            avg_is_bps=avg_is,
            avg_vwap_slippage_bps=sum(vwap_vals) / n,
            avg_spread_cost_bps=sum(spread_vals) / n,
            avg_market_impact_bps=sum(impact_vals) / n,
            avg_timing_cost_bps=sum(timing_vals) / n,
            avg_total_cost_bps=sum(total_vals) / n,
            std_is_bps=std_is,
            n_trades=n,
            n_partial_fills=n_partial,
            fill_rate=sum(fill_vals) / n,
            by_symbol=by_symbol,
            by_venue=by_venue,
            by_strategy=by_strategy,
        )

    # ------------------------------------------------------------------
    # Daily summary (uses store if available, else cache)
    # ------------------------------------------------------------------

    def daily_summary(self, date: str) -> DailySummary:
        """
        Generate a DailySummary for the given date string (YYYY-MM-DD).
        Pulls from the store if configured, otherwise scans in-memory cache.
        """
        if self.store is not None:
            try:
                results = self.store.query(date_from=date, date_to=date)
            except Exception:
                results = []
        else:
            results = [r for r in self._results_cache if r.trade_date == date]

        if not results:
            return DailySummary(
                date=date,
                n_trades=0,
                total_notional=0.0,
                avg_is_bps=0.0,
                avg_vwap_slippage_bps=0.0,
                avg_spread_cost_bps=0.0,
                avg_market_impact_bps=0.0,
                avg_participation_rate=0.0,
                avg_fill_rate=0.0,
                best_venue="",
                worst_venue="",
                top_cost_symbols=[],
                by_strategy={},
            )

        n = len(results)
        avg_is = sum(r.implementation_shortfall_bps for r in results) / n
        avg_vwap = sum(r.vwap_slippage_bps for r in results) / n
        avg_spread = sum(r.spread_cost_bps for r in results) / n
        avg_impact = sum(r.market_impact_bps for r in results) / n
        avg_part = sum(r.participation_rate for r in results) / n
        avg_fill = sum(r.fill_rate for r in results) / n
        total_notional = sum(r.fill_price * 100.0 for r in results)  # approximate

        venue_costs = self._group_avg(
            results, "venue",
            [r.implementation_shortfall_bps for r in results]
        )
        best_venue = min(venue_costs, key=venue_costs.get) if venue_costs else ""
        worst_venue = max(venue_costs, key=venue_costs.get) if venue_costs else ""

        symbol_costs = self._group_avg(
            results, "symbol",
            [r.implementation_shortfall_bps for r in results]
        )
        top_cost_symbols = sorted(
            symbol_costs, key=symbol_costs.get, reverse=True
        )[:5]

        by_strategy = self._group_avg(
            results, "strategy",
            [r.implementation_shortfall_bps for r in results]
        )

        return DailySummary(
            date=date,
            n_trades=n,
            total_notional=total_notional,
            avg_is_bps=avg_is,
            avg_vwap_slippage_bps=avg_vwap,
            avg_spread_cost_bps=avg_spread,
            avg_market_impact_bps=avg_impact,
            avg_participation_rate=avg_part,
            avg_fill_rate=avg_fill,
            best_venue=best_venue,
            worst_venue=worst_venue,
            top_cost_symbols=top_cost_symbols,
            by_strategy=by_strategy,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _market_data_from_trade(self, trade: TradeRecord) -> MarketData:
        """Build a MarketData object from fields embedded in TradeRecord."""
        return MarketData(
            symbol=trade.symbol,
            interval_vwap=trade.interval_vwap,
            interval_twap=trade.interval_twap,
            close_price=trade.close_price,
            bid=trade.bid_at_fill,
            ask=trade.ask_at_fill,
            market_volume=trade.market_volume,
            adv=trade.adv,
        )

    def _compute_spread_cost(self, trade: TradeRecord, arrival: float) -> float:
        """Compute half-spread cost in bps."""
        if trade.bid_at_fill is not None and trade.ask_at_fill is not None:
            spread = trade.ask_at_fill - trade.bid_at_fill
            mid = (trade.ask_at_fill + trade.bid_at_fill) / 2.0
            if mid > 0.0:
                return (spread / 2.0 / mid) * 10_000.0
        # Default half-spread estimate: 1 bps for liquid names
        return 1.0

    def _compute_timing_cost(
        self,
        trade: TradeRecord,
        arrival: float,
        market_data: MarketData,
    ) -> float:
        """
        Estimate timing cost: price change from arrival to fill time
        that is unrelated to our own order.
        Approximated as IS minus the spread component.
        """
        sign = _side_sign(trade.side)
        if arrival <= 0.0 or trade.fill_price <= 0.0:
            return 0.0
        raw_drift = sign * (trade.fill_price - arrival)
        timing_bps = _safe_bps(raw_drift, arrival)
        # Only count positive timing cost (adverse drift) -- we do not credit
        # favorable timing as we cannot disentangle it from alpha
        return max(timing_bps * 0.3, 0.0)   # 30% of drift attributed to timing delay

    def _fill_time_ms(self, trade: TradeRecord) -> float:
        """Compute time from order submission to fill in milliseconds."""
        if trade.order_time is None or trade.fill_time is None:
            return 0.0
        order_ts = trade.order_time
        fill_ts = trade.fill_time
        # Make both offset-naive for subtraction if needed
        if order_ts.tzinfo is not None and fill_ts.tzinfo is None:
            fill_ts = fill_ts.replace(tzinfo=timezone.utc)
        elif order_ts.tzinfo is None and fill_ts.tzinfo is not None:
            order_ts = order_ts.replace(tzinfo=timezone.utc)
        delta = fill_ts - order_ts
        ms = delta.total_seconds() * 1000.0
        return max(ms, 0.0)

    @staticmethod
    def _group_avg(
        results: List[TCAResult],
        attr: str,
        values: List[float],
    ) -> Dict[str, float]:
        """Compute average of values grouped by an attribute of TCAResult."""
        groups: Dict[str, List[float]] = {}
        for r, v in zip(results, values):
            key = str(getattr(r, attr, "unknown") or "unknown")
            groups.setdefault(key, []).append(v)
        return {k: sum(vs) / len(vs) for k, vs in groups.items()}

    def clear_cache(self) -> None:
        """Clear the in-memory results cache."""
        self._results_cache.clear()
