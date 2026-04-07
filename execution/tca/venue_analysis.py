# execution/tca/venue_analysis.py -- Per-venue performance analysis for SRFM TCA
# Ranks venues by composite score and recommends optimal routing.

from __future__ import annotations

import csv
import io
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class VenueScore:
    """Composite performance score for a single execution venue."""
    venue: str
    avg_slippage_bps: float     # average implementation shortfall
    fill_rate: float            # fraction of orders fully filled (0..1)
    avg_fill_time_ms: float     # average time from order to fill in ms
    spread_paid_bps: float      # average spread cost in bps
    score: float                # composite 0-100 (higher = better)
    n_trades: int               # number of trades in the scoring window


@dataclass
class VenueComparison:
    """Comparison of multiple venues across a batch of trades."""
    scores: Dict[str, VenueScore]          # venue -> VenueScore
    best_venue: str                        # venue with highest composite score
    worst_venue: str                       # venue with lowest composite score
    ranking: List[str]                     # venues sorted best to worst
    spread: float                          # bps difference best vs worst avg IS
    n_venues: int


@dataclass
class RoutingRecommendation:
    """Venue routing recommendation for a prospective order."""
    recommended_venue: str
    reason: str
    alternatives: List[str]
    confidence: float    # 0..1 based on data quality
    order_type: str      # "LIMIT", "MARKET", "IOC", etc.


# ---------------------------------------------------------------------------
# Scoring weights (tunable)
# ---------------------------------------------------------------------------

_WEIGHT_SLIPPAGE = 0.40     # IS / slippage -- most important
_WEIGHT_FILL_RATE = 0.25    # fill quality
_WEIGHT_SPEED = 0.20        # fill speed
_WEIGHT_SPREAD = 0.15       # spread paid


def _score_metric(
    value: float,
    all_values: List[float],
    lower_is_better: bool = True,
) -> float:
    """
    Normalize a metric to 0-100 range given distribution across venues.
    Returns 50.0 if all venues have the same value (no differentiation).
    """
    if len(all_values) <= 1 or max(all_values) == min(all_values):
        return 50.0
    lo = min(all_values)
    hi = max(all_values)
    normalized = (value - lo) / (hi - lo)   # 0 = best_raw, 1 = worst_raw if lower_is_better
    if lower_is_better:
        return (1.0 - normalized) * 100.0
    return normalized * 100.0


def _composite_score(
    slippage: float,
    fill_rate: float,
    fill_time_ms: float,
    spread_paid: float,
    all_slippages: List[float],
    all_fill_rates: List[float],
    all_fill_times: List[float],
    all_spreads: List[float],
) -> float:
    """Compute composite 0-100 score from venue metrics."""
    s_slip = _score_metric(slippage, all_slippages, lower_is_better=True)
    s_fill = _score_metric(fill_rate, all_fill_rates, lower_is_better=False)
    s_speed = _score_metric(fill_time_ms, all_fill_times, lower_is_better=True)
    s_spread = _score_metric(spread_paid, all_spreads, lower_is_better=True)

    return (
        _WEIGHT_SLIPPAGE * s_slip
        + _WEIGHT_FILL_RATE * s_fill
        + _WEIGHT_SPEED * s_speed
        + _WEIGHT_SPREAD * s_spread
    )


# ---------------------------------------------------------------------------
# Venue analyzer
# ---------------------------------------------------------------------------

class VenueAnalyzer:
    """
    Analyzes per-venue execution quality from TCA results and produces
    routing recommendations.
    """

    def __init__(self) -> None:
        # symbol -> list of TCAResult for recent trades
        self._history: List = []

    def add_results(self, results: List) -> None:
        """Add TCAResult objects to the analyzer's history."""
        self._history.extend(results)

    def compare_venues(self, trades: List) -> VenueComparison:
        """
        Compare all venues represented in a list of TCAResult objects.

        Parameters
        ----------
        trades : List[TCAResult]

        Returns
        -------
        VenueComparison with scores and rankings
        """
        if not trades:
            return VenueComparison(
                scores={},
                best_venue="",
                worst_venue="",
                ranking=[],
                spread=0.0,
                n_venues=0,
            )

        # Group by venue
        venue_groups: Dict[str, List] = {}
        for t in trades:
            v = getattr(t, "venue", "UNKNOWN") or "UNKNOWN"
            venue_groups.setdefault(v, []).append(t)

        # Compute per-venue aggregate metrics
        venue_metrics: Dict[str, Tuple[float, float, float, float, int]] = {}
        for venue, group in venue_groups.items():
            n = len(group)
            avg_slip = sum(r.implementation_shortfall_bps for r in group) / n
            avg_fill = sum(r.fill_rate for r in group) / n
            avg_time = sum(r.time_to_fill_ms for r in group) / n
            avg_spread = sum(r.spread_cost_bps for r in group) / n
            venue_metrics[venue] = (avg_slip, avg_fill, avg_time, avg_spread, n)

        venues = list(venue_metrics.keys())
        all_slippages = [venue_metrics[v][0] for v in venues]
        all_fill_rates = [venue_metrics[v][1] for v in venues]
        all_fill_times = [venue_metrics[v][2] for v in venues]
        all_spreads = [venue_metrics[v][3] for v in venues]

        scores: Dict[str, VenueScore] = {}
        for venue in venues:
            slip, fill, speed, spread, n = venue_metrics[venue]
            score = _composite_score(
                slip, fill, speed, spread,
                all_slippages, all_fill_rates, all_fill_times, all_spreads,
            )
            scores[venue] = VenueScore(
                venue=venue,
                avg_slippage_bps=slip,
                fill_rate=fill,
                avg_fill_time_ms=speed,
                spread_paid_bps=spread,
                score=score,
                n_trades=n,
            )

        ranking = sorted(venues, key=lambda v: scores[v].score, reverse=True)
        best_venue = ranking[0] if ranking else ""
        worst_venue = ranking[-1] if ranking else ""
        spread_bps = (
            scores[worst_venue].avg_slippage_bps - scores[best_venue].avg_slippage_bps
            if len(ranking) >= 2
            else 0.0
        )

        return VenueComparison(
            scores=scores,
            best_venue=best_venue,
            worst_venue=worst_venue,
            ranking=ranking,
            spread=spread_bps,
            n_venues=len(venues),
        )

    def best_venue_for(
        self,
        symbol: str,
        side: str,
        qty: float,
        urgency: str,          # "HIGH", "MEDIUM", "LOW"
        adv: Optional[float] = None,
    ) -> str:
        """
        Recommend the best venue for a given order using routing rules.

        Routing logic:
        - Large orders (>1% ADV): prefer dark pools (DARKPOOL, IEX, BATS_DARK)
        - Urgent orders: prefer direct exchange with aggressive limit (NYSE, NASDAQ)
        - Normal orders: prefer venue with best recent composite score for symbol

        Parameters
        ----------
        symbol  : ticker symbol
        side    : "BUY" or "SELL"
        qty     : order quantity
        urgency : "HIGH", "MEDIUM", "LOW"
        adv     : average daily volume (used for large order detection)

        Returns
        -------
        Venue identifier string
        """
        rec = self._route_order(symbol, side, qty, urgency, adv)
        return rec.recommended_venue

    def route_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        urgency: str,
        adv: Optional[float] = None,
    ) -> RoutingRecommendation:
        """Full routing recommendation with reason and alternatives."""
        return self._route_order(symbol, side, qty, urgency, adv)

    def _route_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        urgency: str,
        adv: Optional[float],
    ) -> RoutingRecommendation:
        """Internal routing logic."""
        urgency = urgency.upper()

        # Large order rule: >1% ADV goes to dark pool
        if adv is not None and adv > 0.0 and qty / adv > 0.01:
            return RoutingRecommendation(
                recommended_venue="IEX",
                reason="Large order (>1% ADV) -- prefer dark pool to minimize market impact",
                alternatives=["BATS_DARK", "CROSSFINDER"],
                confidence=0.80,
                order_type="LIMIT",
            )

        # High urgency: direct exchange with aggressive IOC
        if urgency == "HIGH":
            return RoutingRecommendation(
                recommended_venue="NASDAQ",
                reason="High urgency -- direct exchange for fastest fill",
                alternatives=["NYSE", "CBOE"],
                confidence=0.75,
                order_type="IOC",
            )

        # Normal orders: use best scored venue from recent history
        symbol_trades = [
            r for r in self._history
            if getattr(r, "symbol", "") == symbol
        ]
        if symbol_trades:
            comparison = self.compare_venues(symbol_trades)
            if comparison.best_venue:
                alts = comparison.ranking[1:3] if len(comparison.ranking) > 1 else []
                n_best = comparison.scores[comparison.best_venue].n_trades
                confidence = min(0.5 + n_best / 200.0, 0.95)
                return RoutingRecommendation(
                    recommended_venue=comparison.best_venue,
                    reason=f"Best composite score for {symbol} in recent history",
                    alternatives=alts,
                    confidence=confidence,
                    order_type="LIMIT",
                )

        # Fallback: return a reasonable default
        return RoutingRecommendation(
            recommended_venue="NASDAQ",
            reason="No historical data for symbol -- default to primary exchange",
            alternatives=["NYSE", "IEX"],
            confidence=0.40,
            order_type="LIMIT",
        )

    def venue_scorecard(
        self,
        window_days: int = 30,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, VenueScore]:
        """
        Compute venue scorecard using trades within the last window_days.

        Parameters
        ----------
        window_days     : lookback window in calendar days
        reference_date  : date to measure window from (default: today)

        Returns
        -------
        Dict[venue_name -> VenueScore]
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)

        # Normalize to naive UTC for comparison with parsed trade_date strings
        if reference_date.tzinfo is not None:
            reference_date = reference_date.replace(tzinfo=None)
        cutoff = reference_date - timedelta(days=window_days)

        # Filter to window -- use trade_date field if available
        window_trades = []
        for r in self._history:
            trade_date_str = getattr(r, "trade_date", None)
            if trade_date_str:
                try:
                    td = datetime.strptime(trade_date_str, "%Y-%m-%d")
                    if td >= cutoff:
                        window_trades.append(r)
                except ValueError:
                    window_trades.append(r)
            else:
                window_trades.append(r)

        if not window_trades:
            return {}

        comparison = self.compare_venues(window_trades)
        return comparison.scores


# ---------------------------------------------------------------------------
# Venue report generator
# ---------------------------------------------------------------------------

class VenueReportGenerator:
    """Produces markdown and CSV venue scorecards from VenueScore data."""

    def to_markdown(
        self,
        scores: Dict[str, VenueScore],
        title: str = "Venue Scorecard",
    ) -> str:
        """
        Render a VenueScore dict as a Markdown table.

        Returns a multi-line string with a formatted table.
        """
        if not scores:
            return f"## {title}\n\nNo data available.\n"

        ranked = sorted(scores.values(), key=lambda s: s.score, reverse=True)

        lines = [
            f"## {title}",
            "",
            f"| Rank | Venue | Score | Avg IS (bps) | Fill Rate | Avg Fill (ms) | Spread (bps) | N Trades |",
            f"|------|-------|-------|-------------|-----------|---------------|-------------|---------|",
        ]
        for rank, vs in enumerate(ranked, 1):
            lines.append(
                f"| {rank} | {vs.venue} | {vs.score:.1f} | {vs.avg_slippage_bps:.2f} |"
                f" {vs.fill_rate:.1%} | {vs.avg_fill_time_ms:.0f} |"
                f" {vs.spread_paid_bps:.2f} | {vs.n_trades} |"
            )

        lines.append("")
        lines.append(
            f"*Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        )
        return "\n".join(lines)

    def to_csv(
        self,
        scores: Dict[str, VenueScore],
    ) -> str:
        """
        Render a VenueScore dict as a CSV string.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "venue", "score", "avg_slippage_bps", "fill_rate",
            "avg_fill_time_ms", "spread_paid_bps", "n_trades",
        ])
        ranked = sorted(scores.values(), key=lambda s: s.score, reverse=True)
        for vs in ranked:
            writer.writerow([
                vs.venue,
                f"{vs.score:.4f}",
                f"{vs.avg_slippage_bps:.4f}",
                f"{vs.fill_rate:.4f}",
                f"{vs.avg_fill_time_ms:.2f}",
                f"{vs.spread_paid_bps:.4f}",
                vs.n_trades,
            ])
        return output.getvalue()

    def write_csv(
        self,
        scores: Dict[str, VenueScore],
        path: str,
    ) -> int:
        """Write CSV scorecard to file. Returns number of rows written."""
        content = self.to_csv(scores)
        rows = max(content.count("\n") - 1, 0)   # subtract header row
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(content)
        return rows

    def daily_venue_summary(
        self,
        scores: Dict[str, VenueScore],
        date: str,
    ) -> Dict:
        """
        Return a dict summary suitable for logging or JSON serialization.
        """
        if not scores:
            return {"date": date, "n_venues": 0, "venues": []}

        ranked = sorted(scores.values(), key=lambda s: s.score, reverse=True)
        return {
            "date": date,
            "n_venues": len(scores),
            "best_venue": ranked[0].venue,
            "worst_venue": ranked[-1].venue,
            "avg_score": sum(v.score for v in scores.values()) / len(scores),
            "venues": [
                {
                    "venue": vs.venue,
                    "score": vs.score,
                    "avg_slippage_bps": vs.avg_slippage_bps,
                    "fill_rate": vs.fill_rate,
                    "n_trades": vs.n_trades,
                }
                for vs in ranked
            ],
        }
