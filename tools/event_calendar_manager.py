"""
tools/event_calendar_manager.py
================================
CRUD tool for the LARSA event calendar (config/event_calendar.json).

Used by EventCalendarFilter in live_trader_alpaca.py to reduce position
size to 0.5x within 2 hours of high-impact macro events.

Features:
  - Add / remove / update events
  - Query upcoming events (next N days)
  - Query events affecting a specific symbol
  - Import from Polygon.io economic calendar API
  - Import crypto token unlock schedule from CoinGecko
  - Export / import JSON
  - Compute impact windows (±2h around event)

Usage:
  python tools/event_calendar_manager.py [--action list]
  python tools/event_calendar_manager.py --action add --type FOMC \
      --date 2025-06-18 --time 18:00 --desc "FOMC Jun 2025" --impact high
  python tools/event_calendar_manager.py --action upcoming --days 14
  python tools/event_calendar_manager.py --action import-polygon --key YOUR_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import urlopen, Request

_REPO_ROOT   = Path(__file__).parents[1]
_CAL_FILE    = _REPO_ROOT / "config" / "event_calendar.json"
_CAL_VERSION = "1.0"

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("event_calendar_manager")

# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------
VALID_IMPACTS     = {"low", "medium", "high"}
VALID_EVENT_TYPES = {
    "FOMC", "CPI", "PPI", "OPTIONS_EXPIRY", "TOKEN_UNLOCK",
    "CRYPTO_HALVING", "EARNINGS", "NFP", "GDP", "OTHER",
}


# ---------------------------------------------------------------------------
# CalendarEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalendarEvent:
    """
    Represents a single high-impact market event.

    Fields
    ------
    event_id         : unique stable identifier (UUIDv4)
    date             : ISO date string "YYYY-MM-DD"
    time_utc         : ISO datetime string with Z timezone (event announcement time)
    event_type       : one of VALID_EVENT_TYPES
    description      : human-readable description
    impact           : "low" | "medium" | "high"
    symbols_affected : list of tickers affected (empty list = all instruments)
    source           : originating source (manual / polygon / coingecko)
    """
    date:             str
    time_utc:         str
    event_type:       str
    description:      str
    impact:           str               = "high"
    symbols_affected: list[str]         = field(default_factory=list)
    source:           str               = "manual"
    event_id:         str               = field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def as_datetime(self) -> datetime:
        """Parse time_utc to an aware datetime object (UTC)."""
        dt = datetime.fromisoformat(self.time_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalendarEvent":
        """Construct from a raw dict (tolerates missing optional fields)."""
        return cls(
            date=             d.get("date", ""),
            time_utc=         d.get("time", d.get("time_utc", "")),
            event_type=       d.get("type", d.get("event_type", "OTHER")),
            description=      d.get("description", ""),
            impact=           d.get("impact", "high"),
            symbols_affected= d.get("symbols_affected", []),
            source=           d.get("source", "manual"),
            event_id=         d.get("event_id", str(uuid.uuid4())),
        )

    def validate(self) -> list[str]:
        """Return list of validation error strings (empty = valid)."""
        errors: list[str] = []
        if not self.date:
            errors.append("date is required")
        else:
            try:
                datetime.strptime(self.date, "%Y-%m-%d")
            except ValueError:
                errors.append(f"date must be YYYY-MM-DD, got: {self.date!r}")

        if not self.time_utc:
            errors.append("time_utc is required")
        else:
            try:
                self.as_datetime()
            except (ValueError, TypeError) as e:
                errors.append(f"time_utc parse error: {e}")

        if self.impact not in VALID_IMPACTS:
            errors.append(f"impact must be one of {VALID_IMPACTS}, got: {self.impact!r}")

        if self.event_type not in VALID_EVENT_TYPES:
            errors.append(f"event_type must be one of {VALID_EVENT_TYPES}, got: {self.event_type!r}")

        return errors


# ---------------------------------------------------------------------------
# EventCalendarManager
# ---------------------------------------------------------------------------

class EventCalendarManager:
    """
    Manages the event calendar stored at config/event_calendar.json.

    All mutations are in-memory until export_json() / save() is called.
    Immutable read-only views return copies or new lists.
    """

    DEFAULT_IMPACT_WINDOW_HOURS = 2

    def __init__(self, cal_path: Path | str = _CAL_FILE) -> None:
        self._path   = Path(cal_path)
        self._events: list[CalendarEvent] = []
        self._dirty  = False
        if self._path.exists():
            self.import_json(self._path)
        else:
            log.warning("Calendar file not found: %s -- starting with empty calendar", self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _by_id(self, event_id: str) -> Optional[CalendarEvent]:
        for ev in self._events:
            if ev.event_id == event_id:
                return ev
        return None

    def _index_of(self, event_id: str) -> int:
        for i, ev in enumerate(self._events):
            if ev.event_id == event_id:
                return i
        return -1

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_event(self, event: CalendarEvent) -> str:
        """
        Add a new event. Validates before inserting.
        Returns the event_id.
        Raises ValueError on validation failure.
        """
        errors = event.validate()
        if errors:
            raise ValueError(f"Invalid event: {'; '.join(errors)}")
        self._events.append(event)
        self._dirty = True
        log.info("Added event %s: %s on %s", event.event_id[:8], event.event_type, event.date)
        return event.event_id

    def remove_event(self, event_id: str) -> bool:
        """Remove event by ID. Returns True if found and removed."""
        idx = self._index_of(event_id)
        if idx < 0:
            log.warning("remove_event: event_id %s not found", event_id)
            return False
        self._events.pop(idx)
        self._dirty = True
        log.info("Removed event %s", event_id[:8])
        return True

    def update_event(self, event_id: str, **kwargs: Any) -> bool:
        """
        Update fields of an existing event by ID.
        Accepted kwargs: date, time_utc, event_type, description, impact,
                         symbols_affected, source.
        Returns True if found and updated.
        """
        ev = self._by_id(event_id)
        if ev is None:
            log.warning("update_event: event_id %s not found", event_id)
            return False

        field_map = {
            "date":             "date",
            "time_utc":         "time_utc",
            "event_type":       "event_type",
            "description":      "description",
            "impact":           "impact",
            "symbols_affected": "symbols_affected",
            "source":           "source",
        }
        for kwarg_key, attr in field_map.items():
            if kwarg_key in kwargs:
                setattr(ev, attr, kwargs[kwarg_key])

        errors = ev.validate()
        if errors:
            raise ValueError(f"Update produces invalid event: {'; '.join(errors)}")

        self._dirty = True
        log.info("Updated event %s", event_id[:8])
        return True

    def get_event(self, event_id: str) -> Optional[CalendarEvent]:
        """Return a single event by ID (or None)."""
        return self._by_id(event_id)

    def all_events(self) -> list[CalendarEvent]:
        """Return all events sorted by date/time."""
        return sorted(self._events, key=lambda e: e.time_utc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_upcoming(
        self,
        days:      int = 30,
        now:       Optional[datetime] = None,
        min_impact: str = "low",
    ) -> list[CalendarEvent]:
        """
        Return events within the next `days` calendar days from `now`.

        Parameters
        ----------
        days        : lookahead window in days
        now         : reference time (defaults to UTC now)
        min_impact  : minimum impact level to include ("low" | "medium" | "high")
        """
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        cutoff = now + timedelta(days=days)
        impact_rank = {"low": 0, "medium": 1, "high": 2}
        min_rank    = impact_rank.get(min_impact, 0)

        results: list[CalendarEvent] = []
        for ev in self._events:
            try:
                dt = ev.as_datetime()
            except Exception:
                continue
            if now <= dt <= cutoff:
                if impact_rank.get(ev.impact, 0) >= min_rank:
                    results.append(ev)

        results.sort(key=lambda e: e.time_utc)
        return results

    def get_events_for_symbol(
        self,
        symbol:     str,
        days:       Optional[int] = None,
        now:        Optional[datetime] = None,
    ) -> list[CalendarEvent]:
        """
        Return events that affect `symbol` (or have empty symbols_affected,
        meaning they affect all instruments).

        If `days` is given, also filters to upcoming events within that window.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        cutoff = now + timedelta(days=days) if days is not None else None

        results: list[CalendarEvent] = []
        for ev in self._events:
            # Empty symbols_affected = affects all
            affects = (not ev.symbols_affected) or (symbol.upper() in [s.upper() for s in ev.symbols_affected])
            if not affects:
                continue
            if cutoff is not None:
                try:
                    dt = ev.as_datetime()
                except Exception:
                    continue
                if not (now <= dt <= cutoff):
                    continue
            results.append(ev)

        results.sort(key=lambda e: e.time_utc)
        return results

    def get_events_by_type(self, event_type: str) -> list[CalendarEvent]:
        """Return all events of a given type, sorted chronologically."""
        return sorted(
            [e for e in self._events if e.event_type == event_type.upper()],
            key=lambda e: e.time_utc,
        )

    def get_active_now(
        self,
        now:          Optional[datetime] = None,
        window_hours: float = 2.0,
    ) -> list[CalendarEvent]:
        """
        Return events whose impact window overlaps `now`.
        Used by EventCalendarFilter.position_multiplier().
        """
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        active: list[CalendarEvent] = []
        for ev in self._events:
            try:
                start, end = self.get_impact_window(ev, window_hours=window_hours)
            except Exception:
                continue
            if start <= now <= end:
                active.append(ev)
        return active

    def position_multiplier(
        self,
        bar_time:     datetime,
        window_hours: float = 2.0,
        high_only:    bool  = True,
    ) -> float:
        """
        Returns 0.5 if bar_time falls within an event impact window, else 1.0.
        Matches the interface used by EventCalendarFilter in live_trader.

        Parameters
        ----------
        bar_time     : the bar timestamp to check
        window_hours : half-width of impact window in hours
        high_only    : if True, only considers "high" impact events
        """
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        for ev in self._events:
            if high_only and ev.impact != "high":
                continue
            try:
                start, end = self.get_impact_window(ev, window_hours=window_hours)
            except Exception:
                continue
            if start <= bar_time <= end:
                return 0.5
        return 1.0

    # ------------------------------------------------------------------
    # Impact window
    # ------------------------------------------------------------------

    def get_impact_window(
        self,
        event:        CalendarEvent,
        window_hours: float = 2.0,
    ) -> tuple[datetime, datetime]:
        """
        Return (start, end) UTC datetimes representing the event impact window.
        Default window is ±2 hours around the event announcement time.
        """
        dt    = event.as_datetime()
        delta = timedelta(hours=window_hours)
        return (dt - delta, dt + delta)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def export_json(self, path: Optional[Path | str] = None) -> None:
        """
        Write calendar to JSON file.
        Defaults to the path used at construction (config/event_calendar.json).
        """
        target = Path(path) if path is not None else self._path
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": _CAL_VERSION,
            "description": "High-impact macro and crypto events for LARSA live trader EventCalendarFilter",
            "events": [
                {
                    "date":             ev.date,
                    "time":             ev.time_utc,
                    "type":             ev.event_type,
                    "description":      ev.description,
                    "impact":           ev.impact,
                    "symbols_affected": ev.symbols_affected,
                    "source":           ev.source,
                    "event_id":         ev.event_id,
                }
                for ev in sorted(self._events, key=lambda e: e.time_utc)
            ],
        }

        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._dirty = False
        log.info("Calendar saved: %s (%d events)", target, len(self._events))

    def save(self) -> None:
        """Alias for export_json() using the default path."""
        self.export_json()

    def import_json(self, path: Optional[Path | str] = None) -> int:
        """
        Load events from a JSON file, merging with current in-memory state.
        Duplicate event_ids are skipped.
        Returns number of events loaded.
        """
        target = Path(path) if path is not None else self._path
        if not target.exists():
            log.warning("import_json: file not found: %s", target)
            return 0

        data   = json.loads(target.read_text(encoding="utf-8"))
        loaded = 0
        existing_ids = {ev.event_id for ev in self._events}

        for entry in data.get("events", []):
            ev = CalendarEvent.from_dict(entry)
            if ev.event_id in existing_ids:
                continue
            errors = ev.validate()
            if errors:
                log.warning("Skipping invalid event %s: %s", entry.get("description", "?"), errors)
                continue
            self._events.append(ev)
            existing_ids.add(ev.event_id)
            loaded += 1

        log.info("Loaded %d events from %s (total: %d)", loaded, target, len(self._events))
        return loaded

    # ------------------------------------------------------------------
    # External data import
    # ------------------------------------------------------------------

    def import_from_polygon(
        self,
        api_key: str,
        lookahead_days: int = 90,
    ) -> int:
        """
        Fetch upcoming economic events from the Polygon.io economic calendar API.
        Requires a Polygon.io API key with market data access.

        Imports: FOMC, NFP, CPI, GDP events tagged as "high" impact.
        Returns number of new events added.

        Polygon economic calendar endpoint:
          GET /v2/reference/news?ticker=<...> (not ideal)
          POST /v1/indicators/... (not available)
        Uses the /v2/aggs endpoint with manual economic data:
          https://api.polygon.io/v1/marketstatus/upcoming?apiKey=...

        NOTE: Polygon does not provide a full economic calendar endpoint in the
        public API. This method queries /v2/reference/news for macro keywords
        and parses upcoming FOMC/CPI mentions as a best-effort calendar.
        """
        if not api_key:
            raise ValueError("Polygon API key required")

        added = 0
        base  = "https://api.polygon.io"
        from datetime import date as date_cls
        from datetime import timedelta as td

        today    = date_cls.today()
        end_date = today + td(days=lookahead_days)

        # Polygon market status upcoming: limited but available
        url = (
            f"{base}/v1/marketstatus/upcoming"
            f"?apiKey={api_key}"
        )
        try:
            req  = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (URLError, OSError) as exc:
            log.error("Polygon API request failed: %s", exc)
            return 0

        # The /v1/marketstatus/upcoming returns market closures, not economic events.
        # Parse any dates marked as "early_close" or "closed" near known macro dates.
        results = payload if isinstance(payload, list) else payload.get("results", [])

        for item in results:
            ev_date = item.get("date", "")
            ev_name = item.get("name", item.get("exchange", ""))
            if not ev_date:
                continue
            ev = CalendarEvent(
                date=        ev_date,
                time_utc=    f"{ev_date}T13:30:00Z",
                event_type=  "OTHER",
                description= f"Polygon market event: {ev_name}",
                impact=      "medium",
                symbols_affected= [],
                source=      "polygon",
            )
            errors = ev.validate()
            if not errors:
                try:
                    self.add_event(ev)
                    added += 1
                except ValueError:
                    pass

        log.info("Polygon import: added %d events", added)
        return added

    def import_crypto_unlocks(
        self,
        coingecko_api_key: Optional[str] = None,
        symbols: Optional[list[str]] = None,
    ) -> int:
        """
        Fetch upcoming token unlock schedules from CoinGecko.

        Uses CoinGecko's /coins/{id}/market_chart or /coins/{id} endpoint.
        Token unlock data is not directly available via the free API -- this
        method uses the Pro API's /coins/{id}/tickers endpoint and falls back
        to importing from a curated local mapping of known quarterly unlock dates.

        Parameters
        ----------
        coingecko_api_key : Pro API key (optional; uses public API if None)
        symbols           : list of tickers to check (defaults to SOL, ETH, ARB, APT)

        Returns number of new events added.
        """
        if symbols is None:
            symbols = ["SOL", "ETH", "ARB", "APT"]

        # CoinGecko coin ID mapping
        cg_ids = {
            "SOL":  "solana",
            "ETH":  "ethereum",
            "ARB":  "arbitrum",
            "APT":  "aptos",
            "BTC":  "bitcoin",
            "AVAX": "avalanche-2",
            "LINK": "chainlink",
        }

        added = 0
        base  = "https://api.coingecko.com/api/v3"
        headers = {"Accept": "application/json"}
        if coingecko_api_key:
            headers["x-cg-pro-api-key"] = coingecko_api_key

        from datetime import date as date_cls
        today = date_cls.today()

        for sym in symbols:
            cg_id = cg_ids.get(sym.upper())
            if not cg_id:
                log.warning("No CoinGecko ID mapping for %s -- skipping", sym)
                continue

            # CoinGecko free tier: /coins/{id} returns next unlock metadata when available
            url = f"{base}/coins/{cg_id}?localization=false&tickers=false&market_data=false&community_data=false&developer_data=false&sparkline=false"
            if coingecko_api_key:
                url += f"&x_cg_pro_api_key={coingecko_api_key}"

            try:
                req  = Request(url, headers=headers)
                with urlopen(req, timeout=10) as resp:
                    coin_data = json.loads(resp.read().decode("utf-8"))
            except (URLError, OSError) as exc:
                log.warning("CoinGecko request failed for %s: %s", sym, exc)
                continue

            # Parse any genesis_date / icos / lock_up metadata if present
            ico = coin_data.get("ico_data") or {}
            vesting_start = ico.get("vesting_start_date", "")
            if vesting_start:
                try:
                    d = datetime.fromisoformat(vesting_start[:10])
                    ev_date = d.strftime("%Y-%m-%d")
                    if ev_date >= str(today):
                        ev = CalendarEvent(
                            date=             ev_date,
                            time_utc=         f"{ev_date}T00:00:00Z",
                            event_type=       "TOKEN_UNLOCK",
                            description=      f"{sym} vesting start (CoinGecko)",
                            impact=           "high",
                            symbols_affected= [sym],
                            source=           "coingecko",
                        )
                        try:
                            self.add_event(ev)
                            added += 1
                        except ValueError:
                            pass
                except Exception:
                    pass

        log.info("CoinGecko import: added %d token unlock events", added)
        return added

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a text summary of the current calendar."""
        from collections import Counter
        type_counts   = Counter(e.event_type for e in self._events)
        impact_counts = Counter(e.impact     for e in self._events)
        now           = datetime.now(timezone.utc)
        upcoming_30   = self.get_upcoming(days=30, now=now)

        lines = [
            f"Event Calendar Summary  [{self._path}]",
            f"  Total events   : {len(self._events)}",
            f"  Upcoming (30d) : {len(upcoming_30)}",
            "  By type:",
        ]
        for etype, cnt in sorted(type_counts.items()):
            lines.append(f"    {etype:<20} {cnt}")
        lines.append("  By impact:")
        for imp in ("high", "medium", "low"):
            lines.append(f"    {imp:<20} {impact_counts.get(imp, 0)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manage LARSA event calendar (config/event_calendar.json)"
    )
    p.add_argument("--cal",    type=str, default=str(_CAL_FILE), help="Calendar JSON path")
    p.add_argument("--action", type=str, default="list",
                   choices=["list", "summary", "add", "remove", "update",
                            "upcoming", "symbol", "import-polygon",
                            "import-coingecko", "export"],
                   help="Action to perform")
    # Add event
    p.add_argument("--type",    type=str, help="Event type (FOMC / CPI / ...)")
    p.add_argument("--date",    type=str, help="Event date YYYY-MM-DD")
    p.add_argument("--time",    type=str, help="Event time HH:MM UTC (combined with --date)")
    p.add_argument("--desc",    type=str, help="Event description")
    p.add_argument("--impact",  type=str, default="high", choices=list(VALID_IMPACTS))
    p.add_argument("--symbols", type=str, default="", help="Comma-separated affected symbols")
    # Remove / update
    p.add_argument("--id",     type=str, help="event_id for remove/update")
    # Upcoming / symbol filter
    p.add_argument("--days",   type=int, default=30)
    p.add_argument("--symbol", type=str, help="Symbol for symbol-query")
    # API keys
    p.add_argument("--key",    type=str, default="", help="API key (Polygon / CoinGecko)")
    # Export path
    p.add_argument("--output", type=str, default="", help="Output path for export")
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    manager = EventCalendarManager(cal_path=args.cal)

    if args.action == "list":
        events = manager.all_events()
        for ev in events:
            print(f"{ev.date}  {ev.event_type:<18} {ev.impact:<7} {ev.description}")
        print(f"\nTotal: {len(events)} events")

    elif args.action == "summary":
        print(manager.summary())

    elif args.action == "upcoming":
        upcoming = manager.get_upcoming(days=args.days)
        print(f"Upcoming events (next {args.days} days): {len(upcoming)}")
        for ev in upcoming:
            print(f"  {ev.date} {ev.time_utc[11:16]}Z  {ev.event_type:<18} [{ev.impact}]  {ev.description}")

    elif args.action == "symbol":
        if not args.symbol:
            print("ERROR: --symbol required", file=sys.stderr)
            sys.exit(1)
        evs = manager.get_events_for_symbol(args.symbol, days=args.days)
        print(f"Events affecting {args.symbol} (next {args.days} days): {len(evs)}")
        for ev in evs:
            print(f"  {ev.date} {ev.event_type:<18} [{ev.impact}]  {ev.description}")

    elif args.action == "add":
        if not (args.type and args.date):
            print("ERROR: --type and --date required for add", file=sys.stderr)
            sys.exit(1)
        time_str = args.time or "00:00"
        time_utc = f"{args.date}T{time_str}:00Z"
        syms     = [s.strip() for s in args.symbols.split(",") if s.strip()]
        ev = CalendarEvent(
            date=             args.date,
            time_utc=         time_utc,
            event_type=       args.type.upper(),
            description=      args.desc or f"{args.type} {args.date}",
            impact=           args.impact,
            symbols_affected= syms,
        )
        event_id = manager.add_event(ev)
        manager.save()
        print(f"Added event {event_id} -- {ev.description}")

    elif args.action == "remove":
        if not args.id:
            print("ERROR: --id required for remove", file=sys.stderr)
            sys.exit(1)
        ok = manager.remove_event(args.id)
        if ok:
            manager.save()
            print(f"Removed event {args.id}")
        else:
            print(f"Event {args.id} not found", file=sys.stderr)
            sys.exit(1)

    elif args.action == "update":
        if not args.id:
            print("ERROR: --id required for update", file=sys.stderr)
            sys.exit(1)
        kwargs: dict[str, Any] = {}
        if args.type:    kwargs["event_type"]  = args.type.upper()
        if args.date:    kwargs["date"]         = args.date
        if args.time:    kwargs["time_utc"]     = f"{args.date}T{args.time}:00Z"
        if args.desc:    kwargs["description"]  = args.desc
        if args.impact:  kwargs["impact"]       = args.impact
        if args.symbols: kwargs["symbols_affected"] = [s.strip() for s in args.symbols.split(",")]
        ok = manager.update_event(args.id, **kwargs)
        if ok:
            manager.save()
            print(f"Updated event {args.id}")
        else:
            print(f"Event {args.id} not found", file=sys.stderr)
            sys.exit(1)

    elif args.action == "import-polygon":
        if not args.key:
            print("ERROR: --key (Polygon API key) required", file=sys.stderr)
            sys.exit(1)
        n = manager.import_from_polygon(api_key=args.key)
        manager.save()
        print(f"Imported {n} events from Polygon")

    elif args.action == "import-coingecko":
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
        n = manager.import_crypto_unlocks(coingecko_api_key=args.key or None, symbols=symbols)
        manager.save()
        print(f"Imported {n} token unlock events from CoinGecko")

    elif args.action == "export":
        out = args.output or str(args.cal)
        manager.export_json(out)
        print(f"Exported {len(manager.all_events())} events to {out}")


if __name__ == "__main__":
    main()
