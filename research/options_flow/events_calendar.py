"""
events_calendar.py — Earnings / FOMC / CPI event vol analysis.

Covers:
  - Implied move calculation from straddle pricing
  - Historical realized vs implied move comparison
  - Straddle P&L backtest by event type
  - Event-driven position sizing (Kelly / vol-scaled)
  - FOMC and CPI calendar + expected move
  - Pre-event IV expansion and post-event crush
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from flow_scanner import OptionDataFeed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Event:
    event_type: str          # "earnings", "fomc", "cpi", "ppi", "jobs"
    ticker: Optional[str]    # None for macro events
    date: date
    time_of_day: str         # "pre_market", "after_market", "intraday"
    description: str
    importance: str          # "high", "medium", "low"


@dataclass
class ImpliedMove:
    ticker: str
    event: Event
    spot_price: float
    front_straddle_price: float
    implied_move_pct: float      # ≈ straddle / spot
    implied_move_usd: float
    call_price: float
    put_price: float
    expiry_used: date
    dte: int
    atm_iv: float


@dataclass
class EventHistory:
    event_type: str
    ticker: Optional[str]
    events: List[Dict]   # {date, implied_move_pct, actual_move_pct, straddle_pnl_pct}

    @property
    def n_events(self) -> int:
        return len(self.events)

    def beat_rate(self) -> float:
        """Fraction of events where actual move > implied move."""
        beats = [e for e in self.events if abs(e.get("actual_move_pct", 0)) > abs(e.get("implied_move_pct", 1))]
        return len(beats) / len(self.events) if self.events else 0.5

    def avg_ratio(self) -> float:
        """Average actual/implied move ratio."""
        ratios = [abs(e.get("actual_move_pct", 0)) / abs(e.get("implied_move_pct", 1))
                  for e in self.events if e.get("implied_move_pct", 0) != 0]
        return float(np.mean(ratios)) if ratios else 1.0

    def avg_straddle_pnl(self) -> float:
        pnls = [e.get("straddle_pnl_pct", 0) for e in self.events]
        return float(np.mean(pnls)) if pnls else 0.0


@dataclass
class StraddlePnLResult:
    event: Event
    ticker: str
    entry_price: float       # straddle cost
    exit_price: float        # straddle value after event
    pnl_usd: float
    pnl_pct: float           # return on premium spent
    actual_move_pct: float
    implied_move_pct: float
    beat_implied: bool       # actual > implied
    iv_crush_pct: float      # post-event IV drop


@dataclass
class EventSizing:
    ticker: str
    event: Event
    base_size_usd: float
    kelly_fraction: float
    vol_scaled_size_usd: float
    recommended_size_usd: float
    max_loss_usd: float
    breakeven_move_pct: float
    expected_value_usd: float


# ---------------------------------------------------------------------------
# Event calendar
# ---------------------------------------------------------------------------

class EventCalendar:
    """
    Maintains upcoming event schedule.
    In production: integrate with earnings APIs and FOMC calendar.
    """

    # Fixed macro events (would be updated dynamically)
    FOMC_DATES_2025 = [
        date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
        date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
        date(2025, 10, 29), date(2025, 12, 10),
    ]
    CPI_DATES_2025 = [
        date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
        date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
        date(2025, 7, 11), date(2025, 8, 12), date(2025, 9, 10),
        date(2025, 10, 15), date(2025, 11, 13), date(2025, 12, 10),
    ]

    # Simulated earnings calendar (in production: fetch from Nasdaq/WSJ)
    EARNINGS_SIMULATED = {
        "AAPL":  [date(2025, 1, 30), date(2025, 5, 1), date(2025, 7, 31), date(2025, 10, 30)],
        "NVDA":  [date(2025, 2, 26), date(2025, 5, 28), date(2025, 8, 27), date(2025, 11, 19)],
        "MSFT":  [date(2025, 1, 29), date(2025, 4, 30), date(2025, 7, 30), date(2025, 10, 29)],
        "GOOGL": [date(2025, 2, 4), date(2025, 4, 29), date(2025, 7, 29), date(2025, 10, 28)],
        "META":  [date(2025, 1, 29), date(2025, 4, 30), date(2025, 7, 30), date(2025, 10, 29)],
        "TSLA":  [date(2025, 1, 22), date(2025, 4, 22), date(2025, 7, 23), date(2025, 10, 22)],
        "AMZN":  [date(2025, 2, 6), date(2025, 5, 1), date(2025, 7, 31), date(2025, 10, 30)],
        "SPY":   [],
        "QQQ":   [],
    }

    def upcoming_events(
        self,
        days_ahead: int = 30,
        tickers: List[str] = None,
        include_macro: bool = True,
    ) -> List[Event]:
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        events = []

        if include_macro:
            for d in self.FOMC_DATES_2025:
                if today <= d <= cutoff:
                    events.append(Event("fomc", None, d, "intraday", "FOMC Rate Decision", "high"))
            for d in self.CPI_DATES_2025:
                if today <= d <= cutoff:
                    events.append(Event("cpi", None, d, "pre_market", "CPI Inflation Report", "high"))

        if tickers:
            for ticker in tickers:
                dates = self.EARNINGS_SIMULATED.get(ticker.upper(), [])
                for d in dates:
                    if today <= d <= cutoff:
                        events.append(Event("earnings", ticker, d, "after_market",
                                          f"{ticker} Q Earnings", "high"))

        return sorted(events, key=lambda e: e.date)

    def nearest_event(self, ticker: str = None) -> Optional[Event]:
        upcoming = self.upcoming_events(90, [ticker] if ticker else None, include_macro=True)
        if ticker:
            ticker_events = [e for e in upcoming if e.ticker == ticker]
            return ticker_events[0] if ticker_events else (upcoming[0] if upcoming else None)
        return upcoming[0] if upcoming else None

    def days_to_event(self, event: Event) -> int:
        return (event.date - date.today()).days


# ---------------------------------------------------------------------------
# Implied move calculator
# ---------------------------------------------------------------------------

class ImpliedMoveCalculator:
    """Calculates expected move from ATM straddle pricing."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
        self.feed = OptionDataFeed()
        self.calendar = EventCalendar()

    def calculate(
        self,
        ticker: str,
        event: Event = None,
    ) -> ImpliedMove:
        q = self.feed.get_quotes(ticker)
        spot = float(q.get("last", 100.0))

        exps = self.feed.get_expirations(ticker)
        if not exps:
            raise RuntimeError(f"No expirations for {ticker}")

        # Find expiration just after the event
        if event:
            event_date = event.date
            exp_str = self._find_post_event_expiry(exps, event_date)
        else:
            exp_str = exps[0]

        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = max((exp_date - date.today()).days, 1)
        T = dte / 365.0

        chain = self.feed.get_option_chain(ticker, exp_str)
        atm_call, atm_put = self._get_atm_straddle(chain, spot)

        straddle_price = atm_call["mid"] + atm_put["mid"]
        atm_iv = (float(atm_call.get("implied_volatility", 0.25)) +
                  float(atm_put.get("implied_volatility", 0.25))) / 2

        # Implied move ≈ straddle / spot (simplified)
        implied_move_pct = straddle_price / spot

        if event is None:
            event = self.calendar.nearest_event(ticker) or Event(
                "unknown", ticker, exp_date, "unknown", "Unknown", "medium"
            )

        return ImpliedMove(
            ticker=ticker,
            event=event,
            spot_price=spot,
            front_straddle_price=straddle_price,
            implied_move_pct=implied_move_pct,
            implied_move_usd=straddle_price,
            call_price=atm_call["mid"],
            put_price=atm_put["mid"],
            expiry_used=exp_date,
            dte=dte,
            atm_iv=atm_iv,
        )

    def _find_post_event_expiry(self, expirations: List[str], event_date: date) -> str:
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if exp_date >= event_date:
                return exp_str
        return expirations[0]

    def _get_atm_straddle(
        self, chain: List[Dict], spot: float
    ) -> Tuple[Dict, Dict]:
        calls = [o for o in chain if o.get("option_type","").lower() == "call"]
        puts  = [o for o in chain if o.get("option_type","").lower() == "put"]

        if not calls or not puts:
            dummy = {"implied_volatility": 0.25, "bid": 1.0, "ask": 1.2, "mid": 1.1}
            return dummy, dummy

        def _mid(o):
            return (float(o.get("bid",0)) + float(o.get("ask",0))) / 2

        atm_call = min(calls, key=lambda o: abs(float(o.get("strike",0)) - spot))
        atm_put  = min(puts,  key=lambda o: abs(float(o.get("strike",0)) - spot))

        for o in [atm_call, atm_put]:
            o["mid"] = _mid(o)

        return atm_call, atm_put


# ---------------------------------------------------------------------------
# Event history and straddle P&L backtest
# ---------------------------------------------------------------------------

class EventHistoryAnalyzer:
    """
    Analyzes historical straddle P&L around earnings/FOMC/CPI.
    Uses synthetic historical data when live data not available.
    """

    # Historical earnings move data (approximate actual averages)
    HISTORICAL_EARNINGS_MOVES = {
        "AAPL":  {"avg_implied": 0.048, "avg_actual": 0.042, "std_actual": 0.03, "beat_rate": 0.45},
        "NVDA":  {"avg_implied": 0.090, "avg_actual": 0.112, "std_actual": 0.06, "beat_rate": 0.65},
        "MSFT":  {"avg_implied": 0.055, "avg_actual": 0.050, "std_actual": 0.025, "beat_rate": 0.48},
        "GOOGL": {"avg_implied": 0.058, "avg_actual": 0.055, "std_actual": 0.03, "beat_rate": 0.50},
        "META":  {"avg_implied": 0.078, "avg_actual": 0.085, "std_actual": 0.05, "beat_rate": 0.58},
        "TSLA":  {"avg_implied": 0.092, "avg_actual": 0.096, "std_actual": 0.07, "beat_rate": 0.55},
        "AMZN":  {"avg_implied": 0.062, "avg_actual": 0.057, "std_actual": 0.035, "beat_rate": 0.47},
    }
    FOMC_MOVE_SPY = {"avg_implied": 0.012, "avg_actual": 0.013, "std_actual": 0.008, "beat_rate": 0.52}
    CPI_MOVE_SPY  = {"avg_implied": 0.010, "avg_actual": 0.009, "std_actual": 0.006, "beat_rate": 0.45}

    def build_history(
        self,
        event_type: str,
        ticker: str = None,
        n_simulated: int = 20,
    ) -> EventHistory:
        """Build (synthetic) event history."""
        rng = random.Random(hash(f"{event_type}{ticker}"))

        if event_type == "earnings" and ticker:
            params = self.HISTORICAL_EARNINGS_MOVES.get(ticker.upper(), {
                "avg_implied": 0.06, "avg_actual": 0.06, "std_actual": 0.04, "beat_rate": 0.50,
            })
        elif event_type == "fomc":
            params = self.FOMC_MOVE_SPY
        elif event_type == "cpi":
            params = self.CPI_MOVE_SPY
        else:
            params = {"avg_implied": 0.03, "avg_actual": 0.028, "std_actual": 0.02, "beat_rate": 0.48}

        events = []
        today = date.today()
        for i in range(n_simulated):
            # Simulate historical event
            implied = abs(rng.gauss(params["avg_implied"], params["avg_implied"] * 0.3))
            actual = abs(rng.gauss(params["avg_actual"], params["std_actual"]))
            direction = 1 if rng.random() > 0.5 else -1
            actual_signed = actual * direction

            # Straddle P&L: (actual_move - implied) / implied  (rough approximation)
            straddle_pnl = (actual - implied) / implied

            event_date = today - timedelta(days=i * 90 + rng.randint(0, 30))
            events.append({
                "date": event_date.isoformat(),
                "implied_move_pct": implied,
                "actual_move_pct": actual_signed,
                "straddle_pnl_pct": straddle_pnl,
                "beat_implied": actual > implied,
            })

        return EventHistory(event_type=event_type, ticker=ticker, events=events)

    def event_summary(self, history: EventHistory) -> Dict:
        if not history.events:
            return {}

        implied = [e["implied_move_pct"] for e in history.events]
        actual = [abs(e["actual_move_pct"]) for e in history.events]
        pnls = [e["straddle_pnl_pct"] for e in history.events]

        return {
            "n_events": history.n_events,
            "avg_implied_pct": float(np.mean(implied) * 100),
            "avg_actual_pct": float(np.mean(actual) * 100),
            "actual_implied_ratio": history.avg_ratio(),
            "beat_rate": history.beat_rate(),
            "avg_straddle_pnl_pct": float(np.mean(pnls) * 100),
            "straddle_win_rate": float(np.mean([1 for p in pnls if p > 0])),
            "best_pnl_pct": float(np.max(pnls) * 100),
            "worst_pnl_pct": float(np.min(pnls) * 100),
            "straddle_sharpe": float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Event-driven position sizer
# ---------------------------------------------------------------------------

class EventPositionSizer:
    """
    Sizes event-driven straddle positions using:
    - Expected value (from historical beat rate and avg ratio)
    - Kelly criterion
    - Volatility scaling
    """

    def __init__(self):
        self.history_analyzer = EventHistoryAnalyzer()

    def size(
        self,
        ticker: str,
        event: Event,
        implied_move: ImpliedMove,
        portfolio_size_usd: float,
        max_risk_pct: float = 0.02,    # max 2% of portfolio per event
    ) -> EventSizing:
        history = self.history_analyzer.build_history(event.event_type, ticker)
        summary = self.history_analyzer.event_summary(history)

        actual_implied_ratio = summary.get("actual_implied_ratio", 1.0)
        beat_rate = summary.get("beat_rate", 0.5)

        # Expected value of straddle
        # If actual > implied: profit ≈ (actual - implied)
        # If actual < implied: loss ≈ -(implied - actual)
        ev_per_unit = beat_rate * (actual_implied_ratio - 1) - (1 - beat_rate) * (1 - actual_implied_ratio * 0.5)
        ev_unit = max(-1.0, min(1.0, ev_per_unit))

        # Kelly fraction: EV / (win-loss odds)
        win_amt  = actual_implied_ratio - 1.0
        loss_amt = 1.0   # lose premium
        kelly = (beat_rate * win_amt - (1 - beat_rate) * loss_amt) / win_amt if win_amt > 0 else 0.0
        kelly = max(0.0, min(0.5, kelly))  # cap at 50%

        base_size = portfolio_size_usd * max_risk_pct
        kelly_size = portfolio_size_usd * kelly * max_risk_pct
        vol_scale = implied_move.implied_move_pct / 0.05   # relative to 5% implied move
        vol_scaled_size = base_size / max(0.5, vol_scale)

        recommended = min(base_size, kelly_size * 2, vol_scaled_size)

        # Expected value in USD
        ev_usd = recommended * ev_unit

        # Breakeven: straddle needs to move by straddle cost / delta
        breakeven_pct = implied_move.implied_move_pct  # approximately

        max_loss = recommended  # long straddle: lose full premium

        return EventSizing(
            ticker=ticker,
            event=event,
            base_size_usd=base_size,
            kelly_fraction=kelly,
            vol_scaled_size_usd=vol_scaled_size,
            recommended_size_usd=recommended,
            max_loss_usd=max_loss,
            breakeven_move_pct=breakeven_pct,
            expected_value_usd=ev_usd,
        )


# ---------------------------------------------------------------------------
# IV crush model
# ---------------------------------------------------------------------------

class IVCrushEstimator:
    """
    Estimates post-event IV crush.
    IV typically drops 30-60% after earnings/FOMC.
    """

    CRUSH_ESTIMATES = {
        "earnings": {"mean": 0.45, "std": 0.15, "range": (0.25, 0.70)},
        "fomc":     {"mean": 0.30, "std": 0.10, "range": (0.15, 0.50)},
        "cpi":      {"mean": 0.25, "std": 0.08, "range": (0.10, 0.40)},
    }

    def estimate_crush(self, event_type: str, pre_event_iv: float) -> Dict:
        params = self.CRUSH_ESTIMATES.get(event_type, self.CRUSH_ESTIMATES["earnings"])
        crush_frac = params["mean"]
        post_event_iv = pre_event_iv * (1 - crush_frac)

        return {
            "pre_event_iv": pre_event_iv,
            "estimated_crush_pct": crush_frac * 100,
            "post_event_iv": post_event_iv,
            "iv_pts_drop": (pre_event_iv - post_event_iv) * 100,
            "confidence_range": (
                pre_event_iv * (1 - params["range"][1]),
                pre_event_iv * (1 - params["range"][0]),
            ),
        }

    def time_to_crush(
        self,
        event: Event,
        current_iv: float,
        normal_iv: float,
    ) -> Dict:
        """How much IV is event-elevated vs normal?"""
        event_premium_iv = current_iv - normal_iv
        dte = (event.date - date.today()).days
        decay_per_day = event_premium_iv / dte if dte > 0 else 0

        return {
            "event_date": event.date.isoformat(),
            "dte": dte,
            "current_iv": current_iv,
            "normal_iv": normal_iv,
            "event_premium_iv": event_premium_iv,
            "daily_decay": decay_per_day,
            "note": "IV should approach normal IV as event approaches, then crush after",
        }


# ---------------------------------------------------------------------------
# Main EventsCalendarAnalytics facade
# ---------------------------------------------------------------------------

class EventsCalendarAnalytics:
    """Unified events calendar analytics."""

    def __init__(self):
        self.calendar = EventCalendar()
        self.implied_move_calc = ImpliedMoveCalculator()
        self.history_analyzer = EventHistoryAnalyzer()
        self.sizer = EventPositionSizer()
        self.crush_estimator = IVCrushEstimator()

    def event_preview(
        self,
        ticker: str,
        portfolio_size: float = 1_000_000,
    ) -> str:
        event = self.calendar.nearest_event(ticker)
        if not event:
            return f"No upcoming events found for {ticker}"

        im = self.implied_move_calc.calculate(ticker, event)
        history = self.history_analyzer.build_history(event.event_type, ticker)
        summary = self.history_analyzer.event_summary(history)
        sizing = self.sizer.size(ticker, event, im, portfolio_size)
        crush = self.crush_estimator.estimate_crush(event.event_type, im.atm_iv)

        lines = [
            f"=== Event Preview: {ticker} ===",
            f"Event: {event.description} on {event.date} ({self.calendar.days_to_event(event)} days away)",
            f"Spot: ${im.spot_price:,.2f} | ATM IV: {im.atm_iv:.1%}",
            "",
            "Implied Move:",
            f"  Straddle Cost:    ${im.front_straddle_price:.2f}",
            f"  Implied Move:     ±{im.implied_move_pct:.1%} (±${im.implied_move_usd:.2f})",
            f"  Expiry Used:      {im.expiry_used} ({im.dte} DTE)",
            "",
            "Historical Context:",
            f"  Avg Implied Move: {summary.get('avg_implied_pct', 0):.1f}%",
            f"  Avg Actual Move:  {summary.get('avg_actual_pct', 0):.1f}%",
            f"  Beat Rate:        {summary.get('beat_rate', 0.5):.0%}",
            f"  Avg Straddle P&L: {summary.get('avg_straddle_pnl_pct', 0):.1f}%",
            f"  Straddle Sharpe:  {summary.get('straddle_sharpe', 0):.2f}",
            "",
            "IV Crush Estimate:",
            f"  Pre-event IV:     {crush['pre_event_iv']:.1%}",
            f"  Expected Crush:   {crush['estimated_crush_pct']:.0f}%",
            f"  Post-event IV:    {crush['post_event_iv']:.1%}",
            "",
            "Position Sizing ($1M portfolio):",
            f"  Recommended:      ${sizing.recommended_size_usd:,.0f}",
            f"  Kelly Fraction:   {sizing.kelly_fraction:.1%}",
            f"  Max Loss:         ${sizing.max_loss_usd:,.0f}",
            f"  Expected Value:   ${sizing.expected_value_usd:,.0f}",
        ]
        return "\n".join(lines)

    def upcoming_event_grid(self, tickers: List[str] = None, days: int = 30) -> str:
        all_tickers = tickers or list(EventCalendar.EARNINGS_SIMULATED.keys())[:5]
        events = self.calendar.upcoming_events(days, all_tickers, include_macro=True)

        lines = [
            f"=== Upcoming Events (next {days} days) ===",
            f"{'Date':>12} {'Type':>10} {'Ticker':>8} {'DTE':>4} {'Description':>30}",
            "-" * 70,
        ]
        for e in events[:15]:
            dte = self.calendar.days_to_event(e)
            lines.append(
                f"{e.date.strftime('%Y-%m-%d'):>12} {e.event_type:>10} "
                f"{e.ticker or 'MACRO':>8} {dte:>4} {e.description[:30]:>30}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Events calendar CLI")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--action", choices=["preview", "grid", "history"], default="grid")
    parser.add_argument("--portfolio", type=float, default=1_000_000)
    args = parser.parse_args()

    analytics = EventsCalendarAnalytics()

    if args.action == "grid":
        print(analytics.upcoming_event_grid())
    elif args.action == "preview":
        print(analytics.event_preview(args.ticker, args.portfolio))
    elif args.action == "history":
        import json as _json
        history = analytics.history_analyzer.build_history("earnings", args.ticker)
        summary = analytics.history_analyzer.event_summary(history)
        print(f"Event history summary for {args.ticker}:")
        print(_json.dumps(summary, indent=2))
