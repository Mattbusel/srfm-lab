"""
tools/tests/test_event_calendar.py
====================================
Unit tests for EventCalendarManager and CalendarEvent.

Run:
  python -m pytest tools/tests/test_event_calendar.py -v
  python tools/tests/test_event_calendar.py   (standalone)

Test coverage: 25+ test cases across:
  - JSON loading and serialization
  - Impact window calculation
  - Upcoming events filter (date range, impact level)
  - Symbol-specific event queries
  - Add / remove event CRUD
  - Validation rules
  - Position multiplier calculation
  - Duplicate ID deduplication
  - Edge cases (empty calendar, timezone handling, boundary conditions)
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Allow running from repo root or tools/tests/
sys.path.insert(0, str(Path(__file__).parents[2]))

from tools.event_calendar_manager import (
    CalendarEvent,
    EventCalendarManager,
    VALID_IMPACTS,
    VALID_EVENT_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    date:    str = "2025-06-18",
    time:    str = "2025-06-18T18:00:00Z",
    etype:   str = "FOMC",
    desc:    str = "Test FOMC",
    impact:  str = "high",
    symbols: list[str] | None = None,
    source:  str = "manual",
) -> CalendarEvent:
    return CalendarEvent(
        date=             date,
        time_utc=         time,
        event_type=       etype,
        description=      desc,
        impact=           impact,
        symbols_affected= symbols or [],
        source=           source,
    )


def _manager_with_events(events: list[CalendarEvent]) -> EventCalendarManager:
    """Create an in-memory manager pre-populated with given events (no file I/O)."""
    mgr = EventCalendarManager.__new__(EventCalendarManager)
    mgr._path   = Path("/dev/null")
    mgr._events = list(events)
    mgr._dirty  = False
    return mgr


def _sample_events() -> list[CalendarEvent]:
    """Return a small but representative event set for query tests."""
    now = datetime.now(timezone.utc)
    return [
        CalendarEvent(
            date=        (now + timedelta(days=1)).strftime("%Y-%m-%d"),
            time_utc=    (now + timedelta(days=1)).strftime("%Y-%m-%dT18:00:00Z"),
            event_type=  "FOMC",
            description= "Upcoming FOMC",
            impact=      "high",
            symbols_affected= [],
        ),
        CalendarEvent(
            date=        (now + timedelta(days=3)).strftime("%Y-%m-%d"),
            time_utc=    (now + timedelta(days=3)).strftime("%Y-%m-%dT13:30:00Z"),
            event_type=  "CPI",
            description= "Upcoming CPI",
            impact=      "high",
            symbols_affected= [],
        ),
        CalendarEvent(
            date=        (now + timedelta(days=5)).strftime("%Y-%m-%d"),
            time_utc=    (now + timedelta(days=5)).strftime("%Y-%m-%dT00:00:00Z"),
            event_type=  "TOKEN_UNLOCK",
            description= "SOL unlock",
            impact=      "high",
            symbols_affected= ["SOL"],
        ),
        CalendarEvent(
            date=        (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            time_utc=    (now + timedelta(days=7)).strftime("%Y-%m-%dT21:00:00Z"),
            event_type=  "OPTIONS_EXPIRY",
            description= "Monthly OpEx",
            impact=      "medium",
            symbols_affected= ["SPY", "QQQ", "NVDA"],
        ),
        CalendarEvent(
            date=        (now - timedelta(days=5)).strftime("%Y-%m-%d"),
            time_utc=    (now - timedelta(days=5)).strftime("%Y-%m-%dT18:00:00Z"),
            event_type=  "FOMC",
            description= "Past FOMC",
            impact=      "high",
            symbols_affected= [],
        ),
        CalendarEvent(
            date=        (now + timedelta(days=60)).strftime("%Y-%m-%d"),
            time_utc=    (now + timedelta(days=60)).strftime("%Y-%m-%dT18:00:00Z"),
            event_type=  "FOMC",
            description= "Far future FOMC",
            impact=      "high",
            symbols_affected= [],
        ),
    ]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestCalendarEventConstruction(unittest.TestCase):

    def test_basic_construction(self):
        ev = _make_event()
        self.assertEqual(ev.event_type, "FOMC")
        self.assertEqual(ev.impact,     "high")
        self.assertIsNotNone(ev.event_id)
        self.assertTrue(len(ev.event_id) > 0)

    def test_event_id_unique(self):
        ev1 = _make_event()
        ev2 = _make_event()
        self.assertNotEqual(ev1.event_id, ev2.event_id)

    def test_from_dict_full(self):
        d = {
            "date":             "2025-03-19",
            "time":             "2025-03-19T18:00:00Z",
            "type":             "FOMC",
            "description":      "FOMC Mar 2025",
            "impact":           "high",
            "symbols_affected": [],
            "source":           "manual",
            "event_id":         "abc-123",
        }
        ev = CalendarEvent.from_dict(d)
        self.assertEqual(ev.date,       "2025-03-19")
        self.assertEqual(ev.event_type, "FOMC")
        self.assertEqual(ev.event_id,   "abc-123")

    def test_from_dict_minimal(self):
        d = {"date": "2025-01-15", "time": "2025-01-15T13:30:00Z",
             "type": "CPI", "description": "CPI Jan"}
        ev = CalendarEvent.from_dict(d)
        self.assertEqual(ev.impact, "high")       # default
        self.assertEqual(ev.symbols_affected, []) # default

    def test_as_datetime_utc(self):
        ev = _make_event(time="2025-06-18T18:00:00Z")
        dt = ev.as_datetime()
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt.year, 2025)
        self.assertEqual(dt.hour, 18)

    def test_as_datetime_no_tz_defaults_utc(self):
        ev = _make_event(time="2025-06-18T18:00:00")
        dt = ev.as_datetime()
        self.assertIsNotNone(dt.tzinfo)

    def test_to_dict_roundtrip(self):
        ev1 = _make_event(symbols=["BTC", "ETH"])
        d   = ev1.to_dict()
        ev2 = CalendarEvent.from_dict({
            **d,
            "time": d["time_utc"],
            "type": d["event_type"],
        })
        self.assertEqual(ev2.date,             ev1.date)
        self.assertEqual(ev2.symbols_affected, ev1.symbols_affected)


class TestCalendarEventValidation(unittest.TestCase):

    def test_valid_event_no_errors(self):
        ev = _make_event()
        self.assertEqual(ev.validate(), [])

    def test_invalid_date_format(self):
        ev = _make_event(date="18/06/2025")
        errors = ev.validate()
        self.assertTrue(any("date" in e for e in errors))

    def test_empty_date(self):
        ev = _make_event(date="")
        errors = ev.validate()
        self.assertTrue(any("date" in e for e in errors))

    def test_invalid_impact(self):
        ev = _make_event(impact="extreme")
        errors = ev.validate()
        self.assertTrue(any("impact" in e for e in errors))

    def test_invalid_event_type(self):
        ev = _make_event(etype="MYSTERY")
        errors = ev.validate()
        self.assertTrue(any("event_type" in e for e in errors))

    def test_empty_time_utc(self):
        ev = _make_event(time="")
        errors = ev.validate()
        self.assertTrue(len(errors) > 0)

    def test_all_valid_impacts(self):
        for imp in VALID_IMPACTS:
            ev = _make_event(impact=imp)
            self.assertEqual(ev.validate(), [], f"Expected no errors for impact={imp}")

    def test_all_valid_event_types(self):
        for etype in VALID_EVENT_TYPES:
            ev = _make_event(etype=etype)
            errors = ev.validate()
            self.assertEqual(errors, [], f"Expected no errors for type={etype}")


class TestCalendarJsonLoad(unittest.TestCase):

    def test_calendar_load_json(self):
        """Loading the real config/event_calendar.json should succeed."""
        cal_path = Path(__file__).parents[2] / "config" / "event_calendar.json"
        if not cal_path.exists():
            self.skipTest("config/event_calendar.json not found")
        mgr = EventCalendarManager(cal_path=cal_path)
        self.assertGreater(len(mgr.all_events()), 0)

    def test_calendar_load_json_event_types(self):
        """All loaded events should have valid types and impacts."""
        cal_path = Path(__file__).parents[2] / "config" / "event_calendar.json"
        if not cal_path.exists():
            self.skipTest("config/event_calendar.json not found")
        mgr = EventCalendarManager(cal_path=cal_path)
        for ev in mgr.all_events():
            self.assertIn(ev.event_type, VALID_EVENT_TYPES, f"Bad type: {ev.event_type}")
            self.assertIn(ev.impact,     VALID_IMPACTS,     f"Bad impact: {ev.impact}")

    def test_import_export_roundtrip(self):
        """Export then re-import should preserve all events."""
        mgr = _manager_with_events(_sample_events())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = Path(f.name)
        try:
            mgr.export_json(tmp_path)
            mgr2 = EventCalendarManager(cal_path=tmp_path)
            self.assertEqual(len(mgr2.all_events()), len(mgr.all_events()))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_import_skips_duplicates(self):
        """Re-importing the same JSON does not double the event count."""
        mgr = _manager_with_events(_sample_events())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = Path(f.name)
        try:
            mgr.export_json(tmp_path)
            n_before = len(mgr.all_events())
            mgr.import_json(tmp_path)
            self.assertEqual(len(mgr.all_events()), n_before)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_missing_file_starts_empty(self):
        """Manager with a non-existent path should start with no events."""
        mgr = EventCalendarManager(cal_path="/nonexistent/path/calendar.json")
        self.assertEqual(len(mgr.all_events()), 0)


class TestImpactWindowCalculation(unittest.TestCase):

    def test_impact_window_default_2h(self):
        ev = _make_event(time="2025-06-18T18:00:00Z")
        mgr = _manager_with_events([ev])
        start, end = mgr.get_impact_window(ev)
        self.assertEqual(start, datetime(2025, 6, 18, 16, 0, tzinfo=timezone.utc))
        self.assertEqual(end,   datetime(2025, 6, 18, 20, 0, tzinfo=timezone.utc))

    def test_impact_window_custom_hours(self):
        ev  = _make_event(time="2025-03-19T18:00:00Z")
        mgr = _manager_with_events([ev])
        start, end = mgr.get_impact_window(ev, window_hours=4.0)
        self.assertEqual(end - start, timedelta(hours=8))

    def test_impact_window_zero_hours(self):
        ev  = _make_event(time="2025-01-15T13:30:00Z")
        mgr = _manager_with_events([ev])
        start, end = mgr.get_impact_window(ev, window_hours=0.0)
        self.assertEqual(start, end)

    def test_impact_window_midnight_event(self):
        ev  = _make_event(time="2025-04-01T00:00:00Z")
        mgr = _manager_with_events([ev])
        start, end = mgr.get_impact_window(ev)
        # Window should span into previous day
        self.assertEqual(start.day, 31)  # March 31
        self.assertEqual(end.day,   1)   # April 1


class TestUpcomingEventsFilter(unittest.TestCase):

    def setUp(self):
        self._now = datetime.now(timezone.utc)
        self._mgr = _manager_with_events(_sample_events())

    def test_upcoming_events_filter(self):
        upcoming = self._mgr.get_upcoming(days=30, now=self._now)
        for ev in upcoming:
            dt = ev.as_datetime()
            self.assertGreaterEqual(dt, self._now)
            self.assertLessEqual(dt, self._now + timedelta(days=30))

    def test_upcoming_excludes_past(self):
        upcoming = self._mgr.get_upcoming(days=30, now=self._now)
        descs = [ev.description for ev in upcoming]
        self.assertNotIn("Past FOMC", descs)

    def test_upcoming_excludes_far_future(self):
        upcoming = self._mgr.get_upcoming(days=30, now=self._now)
        descs = [ev.description for ev in upcoming]
        self.assertNotIn("Far future FOMC", descs)

    def test_upcoming_sorted_chronologically(self):
        upcoming = self._mgr.get_upcoming(days=30, now=self._now)
        times = [ev.time_utc for ev in upcoming]
        self.assertEqual(times, sorted(times))

    def test_upcoming_min_impact_high(self):
        upcoming = self._mgr.get_upcoming(days=30, now=self._now, min_impact="high")
        for ev in upcoming:
            self.assertEqual(ev.impact, "high")

    def test_upcoming_min_impact_medium_includes_medium(self):
        upcoming_med = self._mgr.get_upcoming(days=30, now=self._now, min_impact="medium")
        upcoming_high = self._mgr.get_upcoming(days=30, now=self._now, min_impact="high")
        # Medium filter should return >= high filter count
        self.assertGreaterEqual(len(upcoming_med), len(upcoming_high))

    def test_upcoming_empty_calendar(self):
        mgr = _manager_with_events([])
        self.assertEqual(mgr.get_upcoming(days=30), [])

    def test_upcoming_days_zero(self):
        upcoming = self._mgr.get_upcoming(days=0, now=self._now)
        # No events should be strictly in [now, now+0d] since we use <
        self.assertEqual(upcoming, [])


class TestEventAffectsSymbol(unittest.TestCase):

    def setUp(self):
        self._mgr = _manager_with_events(_sample_events())
        self._now = datetime.now(timezone.utc)

    def test_event_affects_symbol(self):
        results = self._mgr.get_events_for_symbol("SOL")
        descs = [ev.description for ev in results]
        self.assertIn("SOL unlock", descs)

    def test_event_does_not_affect_unrelated_symbol(self):
        # TOKEN_UNLOCK for SOL should not appear under BTC
        results = self._mgr.get_events_for_symbol("BTC")
        descs = [ev.description for ev in results]
        # "SOL unlock" has symbols_affected=["SOL"] -- should NOT appear for BTC
        self.assertNotIn("SOL unlock", descs)

    def test_global_events_affect_all_symbols(self):
        # FOMC has empty symbols_affected -- should appear for any symbol
        results = self._mgr.get_events_for_symbol("ETH")
        types = [ev.event_type for ev in results]
        self.assertIn("FOMC", types)

    def test_opex_affects_spy(self):
        results = self._mgr.get_events_for_symbol("SPY")
        descs = [ev.description for ev in results]
        self.assertIn("Monthly OpEx", descs)

    def test_opex_does_not_affect_btc(self):
        results = self._mgr.get_events_for_symbol("BTC")
        descs = [ev.description for ev in results]
        # Monthly OpEx has symbols_affected=["SPY","QQQ","NVDA"] -- should not include BTC
        self.assertNotIn("Monthly OpEx", descs)

    def test_symbol_query_with_days_filter(self):
        results_30d = self._mgr.get_events_for_symbol("SOL", days=30, now=self._now)
        results_1d  = self._mgr.get_events_for_symbol("SOL", days=1,  now=self._now)
        self.assertGreaterEqual(len(results_30d), len(results_1d))


class TestAddRemoveEvent(unittest.TestCase):

    def setUp(self):
        self._mgr = _manager_with_events(_sample_events())

    def test_add_remove_event(self):
        n_before = len(self._mgr.all_events())
        ev = _make_event(desc="New test FOMC")
        eid = self._mgr.add_event(ev)
        self.assertEqual(len(self._mgr.all_events()), n_before + 1)
        ok = self._mgr.remove_event(eid)
        self.assertTrue(ok)
        self.assertEqual(len(self._mgr.all_events()), n_before)

    def test_add_invalid_event_raises(self):
        ev = _make_event(impact="unknown_impact")
        with self.assertRaises(ValueError):
            self._mgr.add_event(ev)

    def test_remove_nonexistent_returns_false(self):
        ok = self._mgr.remove_event("does-not-exist-id")
        self.assertFalse(ok)

    def test_update_event_field(self):
        ev  = _make_event(desc="Original description")
        eid = self._mgr.add_event(ev)
        ok  = self._mgr.update_event(eid, description="Updated description")
        self.assertTrue(ok)
        fetched = self._mgr.get_event(eid)
        self.assertEqual(fetched.description, "Updated description")

    def test_update_nonexistent_returns_false(self):
        ok = self._mgr.update_event("ghost-id", description="X")
        self.assertFalse(ok)

    def test_update_invalid_impact_raises(self):
        ev  = _make_event()
        eid = self._mgr.add_event(ev)
        with self.assertRaises(ValueError):
            self._mgr.update_event(eid, impact="nuclear")

    def test_add_preserves_event_id(self):
        ev  = _make_event()
        original_id = ev.event_id
        returned_id = self._mgr.add_event(ev)
        self.assertEqual(returned_id, original_id)

    def test_get_event_by_id(self):
        ev  = _make_event(desc="Findable event")
        eid = self._mgr.add_event(ev)
        found = self._mgr.get_event(eid)
        self.assertIsNotNone(found)
        self.assertEqual(found.description, "Findable event")

    def test_get_event_nonexistent(self):
        found = self._mgr.get_event("no-such-id")
        self.assertIsNone(found)


class TestPositionMultiplier(unittest.TestCase):

    def test_position_multiplier_during_event(self):
        ev  = _make_event(time="2025-06-18T18:00:00Z", impact="high")
        mgr = _manager_with_events([ev])
        # 1 hour before = inside ±2h window
        bar_time = datetime(2025, 6, 18, 17, 0, tzinfo=timezone.utc)
        self.assertEqual(mgr.position_multiplier(bar_time), 0.5)

    def test_position_multiplier_outside_event(self):
        ev  = _make_event(time="2025-06-18T18:00:00Z", impact="high")
        mgr = _manager_with_events([ev])
        # 3 hours before = outside ±2h window
        bar_time = datetime(2025, 6, 18, 15, 0, tzinfo=timezone.utc)
        self.assertEqual(mgr.position_multiplier(bar_time), 1.0)

    def test_position_multiplier_at_boundary(self):
        ev  = _make_event(time="2025-06-18T18:00:00Z", impact="high")
        mgr = _manager_with_events([ev])
        # Exactly 2 hours before = boundary should be included
        bar_time = datetime(2025, 6, 18, 16, 0, tzinfo=timezone.utc)
        self.assertEqual(mgr.position_multiplier(bar_time), 0.5)

    def test_position_multiplier_high_only_ignores_medium(self):
        ev  = _make_event(time="2025-06-18T18:00:00Z", impact="medium")
        mgr = _manager_with_events([ev])
        bar_time = datetime(2025, 6, 18, 17, 0, tzinfo=timezone.utc)
        # high_only=True (default) should ignore medium events
        self.assertEqual(mgr.position_multiplier(bar_time, high_only=True),  1.0)
        self.assertEqual(mgr.position_multiplier(bar_time, high_only=False), 0.5)

    def test_position_multiplier_no_timezone_bar_time(self):
        ev  = _make_event(time="2025-06-18T18:00:00Z", impact="high")
        mgr = _manager_with_events([ev])
        # Naive datetime (no tz) should be treated as UTC
        bar_time = datetime(2025, 6, 18, 17, 30)  # naive
        result = mgr.position_multiplier(bar_time)
        self.assertEqual(result, 0.5)

    def test_position_multiplier_empty_calendar(self):
        mgr = _manager_with_events([])
        bar_time = datetime.now(timezone.utc)
        self.assertEqual(mgr.position_multiplier(bar_time), 1.0)


class TestGetActiveNow(unittest.TestCase):

    def test_get_active_now_during_event(self):
        now = datetime.now(timezone.utc)
        ev  = CalendarEvent(
            date=        now.strftime("%Y-%m-%d"),
            time_utc=    now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            event_type=  "FOMC",
            description= "Active FOMC",
            impact=      "high",
        )
        mgr = _manager_with_events([ev])
        active = mgr.get_active_now(now=now)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].description, "Active FOMC")

    def test_get_active_now_outside_window(self):
        now = datetime.now(timezone.utc)
        ev  = CalendarEvent(
            date=        (now + timedelta(hours=5)).strftime("%Y-%m-%d"),
            time_utc=    (now + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            event_type=  "FOMC",
            description= "Future FOMC",
            impact=      "high",
        )
        mgr = _manager_with_events([ev])
        active = mgr.get_active_now(now=now)
        self.assertEqual(len(active), 0)


class TestEventsByType(unittest.TestCase):

    def test_get_events_by_type(self):
        mgr    = _manager_with_events(_sample_events())
        fomc   = mgr.get_events_by_type("FOMC")
        for ev in fomc:
            self.assertEqual(ev.event_type, "FOMC")

    def test_get_events_by_type_case_insensitive(self):
        mgr  = _manager_with_events(_sample_events())
        cpi1 = mgr.get_events_by_type("CPI")
        cpi2 = mgr.get_events_by_type("cpi")
        self.assertEqual(len(cpi1), len(cpi2))

    def test_get_events_by_unknown_type_returns_empty(self):
        mgr    = _manager_with_events(_sample_events())
        result = mgr.get_events_by_type("MYSTERY")
        self.assertEqual(result, [])


class TestCalendarSummary(unittest.TestCase):

    def test_summary_contains_total(self):
        mgr     = _manager_with_events(_sample_events())
        summary = mgr.summary()
        self.assertIn("Total events", summary)

    def test_summary_contains_type_name(self):
        mgr     = _manager_with_events(_sample_events())
        summary = mgr.summary()
        self.assertIn("FOMC", summary)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(__import__(__name__))
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
