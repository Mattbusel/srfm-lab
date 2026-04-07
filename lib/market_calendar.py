"""
market_calendar.py -- US equity market calendar and macro event tracking.

Covers NYSE trading hours, half-days 2020-2026, holidays, and key macro
event dates (FOMC, CPI, NFP, OpEx). Also supports crypto 24/7 sessions.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# -- US/Eastern offset from UTC (no DST handling; caller should pass ET-aware datetimes)
ET_UTC_OFFSET = datetime.timedelta(hours=-5)   # EST; -4 during EDT -- approximate


@dataclass
class Event:
    name: str
    date: datetime.date
    category: str      # FOMC | CPI | NFP | OPEX | EARNINGS | HOLIDAY


class MarketCalendar:
    """
    US equity market calendar: trading hours, half days, holidays.
    Also tracks crypto market sessions (24/7 with lower liquidity windows).
    """

    NYSE_OPEN        = (9, 30)    # (hour, minute) ET
    NYSE_CLOSE       = (16, 0)
    HALF_DAY_CLOSE   = (13, 0)

    # -- NYSE holidays 2020-2026 (observed dates)
    NYSE_HOLIDAYS: set[datetime.date] = {
        # 2020
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 20),
        datetime.date(2020, 2, 17),
        datetime.date(2020, 4, 10),
        datetime.date(2020, 5, 25),
        datetime.date(2020, 7, 3),
        datetime.date(2020, 9, 7),
        datetime.date(2020, 11, 26),
        datetime.date(2020, 12, 25),
        # 2021
        datetime.date(2021, 1, 1),
        datetime.date(2021, 1, 18),
        datetime.date(2021, 2, 15),
        datetime.date(2021, 4, 2),
        datetime.date(2021, 5, 31),
        datetime.date(2021, 7, 5),
        datetime.date(2021, 9, 6),
        datetime.date(2021, 11, 25),
        datetime.date(2021, 12, 24),
        # 2022
        datetime.date(2022, 1, 17),
        datetime.date(2022, 2, 21),
        datetime.date(2022, 4, 15),
        datetime.date(2022, 5, 30),
        datetime.date(2022, 6, 20),
        datetime.date(2022, 7, 4),
        datetime.date(2022, 9, 5),
        datetime.date(2022, 11, 24),
        datetime.date(2022, 12, 26),
        # 2023
        datetime.date(2023, 1, 2),
        datetime.date(2023, 1, 16),
        datetime.date(2023, 2, 20),
        datetime.date(2023, 4, 7),
        datetime.date(2023, 5, 29),
        datetime.date(2023, 6, 19),
        datetime.date(2023, 7, 4),
        datetime.date(2023, 9, 4),
        datetime.date(2023, 11, 23),
        datetime.date(2023, 12, 25),
        # 2024
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 15),
        datetime.date(2024, 2, 19),
        datetime.date(2024, 3, 29),
        datetime.date(2024, 5, 27),
        datetime.date(2024, 6, 19),
        datetime.date(2024, 7, 4),
        datetime.date(2024, 9, 2),
        datetime.date(2024, 11, 28),
        datetime.date(2024, 12, 25),
        # 2025
        datetime.date(2025, 1, 1),
        datetime.date(2025, 1, 9),   # national day of mourning -- Carter
        datetime.date(2025, 1, 20),
        datetime.date(2025, 2, 17),
        datetime.date(2025, 4, 18),
        datetime.date(2025, 5, 26),
        datetime.date(2025, 6, 19),
        datetime.date(2025, 7, 4),
        datetime.date(2025, 9, 1),
        datetime.date(2025, 11, 27),
        datetime.date(2025, 12, 25),
        # 2026
        datetime.date(2026, 1, 1),
        datetime.date(2026, 1, 19),
        datetime.date(2026, 2, 16),
        datetime.date(2026, 4, 3),
        datetime.date(2026, 5, 25),
        datetime.date(2026, 6, 19),
        datetime.date(2026, 7, 3),
        datetime.date(2026, 9, 7),
        datetime.date(2026, 11, 26),
        datetime.date(2026, 12, 25),
    }

    # -- NYSE half-day dates (early close at 13:00 ET) 2020-2026
    HALF_DAYS: set[datetime.date] = {
        datetime.date(2020, 11, 27),
        datetime.date(2020, 12, 24),
        datetime.date(2021, 11, 26),
        datetime.date(2022, 11, 25),
        datetime.date(2023, 7, 3),
        datetime.date(2023, 11, 24),
        datetime.date(2024, 7, 3),
        datetime.date(2024, 11, 29),
        datetime.date(2024, 12, 24),
        datetime.date(2025, 7, 3),
        datetime.date(2025, 11, 28),
        datetime.date(2025, 12, 24),
        datetime.date(2026, 11, 27),
        datetime.date(2026, 12, 24),
    }

    def is_holiday(self, d: datetime.date) -> bool:
        return d in self.NYSE_HOLIDAYS

    def is_half_day(self, d: datetime.date) -> bool:
        return d in self.HALF_DAYS

    def is_trading_day(self, d: datetime.date) -> bool:
        """Returns True if NYSE is open on date d (weekday, non-holiday)."""
        return d.weekday() < 5 and d not in self.NYSE_HOLIDAYS

    def _open_time(self, d: datetime.date) -> datetime.datetime:
        return datetime.datetime(d.year, d.month, d.day, *self.NYSE_OPEN)

    def _close_time(self, d: datetime.date) -> datetime.datetime:
        close = self.HALF_DAY_CLOSE if self.is_half_day(d) else self.NYSE_CLOSE
        return datetime.datetime(d.year, d.month, d.day, *close)

    def is_market_open(self, dt: datetime.datetime) -> bool:
        """
        Returns True if NYSE is open at the given datetime.
        Caller must pass a datetime in ET. Handles holidays and half-days.
        """
        d = dt.date()
        if not self.is_trading_day(d):
            return False
        open_dt  = self._open_time(d)
        close_dt = self._close_time(d)
        return open_dt <= dt < close_dt

    def next_open(self, dt: datetime.datetime) -> datetime.datetime:
        """
        Return the next NYSE open datetime after dt (ET-local).
        Skips weekends and holidays.
        """
        candidate = dt.date()
        # -- if we are before open today and today is a trading day, return today's open
        if self.is_trading_day(candidate) and dt < self._open_time(candidate):
            return self._open_time(candidate)
        # -- otherwise advance to the next trading day
        candidate += datetime.timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += datetime.timedelta(days=1)
        return self._open_time(candidate)

    def bars_until_close(self, dt: datetime.datetime, bar_minutes: int) -> int:
        """
        Return the number of full bars remaining until NYSE close.
        Returns 0 if market is not open.
        """
        d = dt.date()
        if not self.is_market_open(dt):
            return 0
        close_dt = self._close_time(d)
        remaining_seconds = (close_dt - dt).total_seconds()
        if remaining_seconds <= 0:
            return 0
        return int(remaining_seconds // (bar_minutes * 60))

    def session_minutes(self, d: datetime.date) -> int:
        """Return total trading minutes for date d."""
        if not self.is_trading_day(d):
            return 0
        open_dt  = self._open_time(d)
        close_dt = self._close_time(d)
        return int((close_dt - open_dt).total_seconds() // 60)

    # ------------------------------------------------------------------
    # Earnings blackout
    # ------------------------------------------------------------------

    def is_earnings_blackout(self, symbol: str, dt: datetime.datetime) -> bool:
        """
        Returns True if dt falls within 2 calendar days pre/post an
        earnings event for symbol. Earnings dates must be registered via
        register_earnings().
        """
        d = dt.date()
        for ed in self._earnings.get(symbol, []):
            delta = abs((d - ed).days)
            if delta <= 2:
                return True
        return False

    def register_earnings(self, symbol: str, dates: list[datetime.date]) -> None:
        """Register known earnings dates for symbol."""
        self._earnings.setdefault(symbol, []).extend(dates)

    # -- lazy-init earnings dict so __init__ stays simple
    @property
    def _earnings(self) -> dict:
        if not hasattr(self, "_earnings_data"):
            self._earnings_data: dict[str, list[datetime.date]] = {}
        return self._earnings_data

    # ------------------------------------------------------------------
    # Crypto session helpers
    # ------------------------------------------------------------------

    # -- crypto liquidity low windows (UTC hour ranges, approximate)
    CRYPTO_LOW_LIQ_UTC = [(0, 6)]   # 00:00-06:00 UTC weekends especially

    def is_crypto_low_liquidity(self, dt: datetime.datetime) -> bool:
        """
        Returns True if dt (UTC) falls in typical low-liquidity crypto hours.
        """
        hour = dt.hour
        for start, end in self.CRYPTO_LOW_LIQ_UTC:
            if start <= hour < end:
                return True
        return False


class EventCalendar:
    """
    Known macro event dates 2020-2026: FOMC, CPI, NFP, OpEx.
    Provides proximity checks and upcoming event queries.
    """

    # -- FOMC meeting end dates (policy announcement day) 2020-2026
    FOMC_DATES: list[datetime.date] = [
        # 2020
        datetime.date(2020, 1, 29), datetime.date(2020, 3, 3),
        datetime.date(2020, 3, 15), datetime.date(2020, 4, 29),
        datetime.date(2020, 6, 10), datetime.date(2020, 7, 29),
        datetime.date(2020, 9, 16), datetime.date(2020, 11, 5),
        datetime.date(2020, 12, 16),
        # 2021
        datetime.date(2021, 1, 27), datetime.date(2021, 3, 17),
        datetime.date(2021, 4, 28), datetime.date(2021, 6, 16),
        datetime.date(2021, 7, 28), datetime.date(2021, 9, 22),
        datetime.date(2021, 11, 3), datetime.date(2021, 12, 15),
        # 2022
        datetime.date(2022, 1, 26), datetime.date(2022, 3, 16),
        datetime.date(2022, 5, 4),  datetime.date(2022, 6, 15),
        datetime.date(2022, 7, 27), datetime.date(2022, 9, 21),
        datetime.date(2022, 11, 2), datetime.date(2022, 12, 14),
        # 2023
        datetime.date(2023, 2, 1),  datetime.date(2023, 3, 22),
        datetime.date(2023, 5, 3),  datetime.date(2023, 6, 14),
        datetime.date(2023, 7, 26), datetime.date(2023, 9, 20),
        datetime.date(2023, 11, 1), datetime.date(2023, 12, 13),
        # 2024
        datetime.date(2024, 1, 31), datetime.date(2024, 3, 20),
        datetime.date(2024, 5, 1),  datetime.date(2024, 6, 12),
        datetime.date(2024, 7, 31), datetime.date(2024, 9, 18),
        datetime.date(2024, 11, 7), datetime.date(2024, 12, 18),
        # 2025
        datetime.date(2025, 1, 29), datetime.date(2025, 3, 19),
        datetime.date(2025, 5, 7),  datetime.date(2025, 6, 18),
        datetime.date(2025, 7, 30), datetime.date(2025, 9, 17),
        datetime.date(2025, 11, 5), datetime.date(2025, 12, 17),
        # 2026
        datetime.date(2026, 1, 28), datetime.date(2026, 3, 18),
        datetime.date(2026, 4, 29), datetime.date(2026, 6, 17),
        datetime.date(2026, 7, 29), datetime.date(2026, 9, 16),
        datetime.date(2026, 11, 4), datetime.date(2026, 12, 16),
    ]

    # -- CPI release dates (typically 2nd or 3rd Tue/Wed of month)
    CPI_DATES: list[datetime.date] = [
        datetime.date(2024, 1, 11),  datetime.date(2024, 2, 13),
        datetime.date(2024, 3, 12),  datetime.date(2024, 4, 10),
        datetime.date(2024, 5, 15),  datetime.date(2024, 6, 12),
        datetime.date(2024, 7, 11),  datetime.date(2024, 8, 14),
        datetime.date(2024, 9, 11),  datetime.date(2024, 10, 10),
        datetime.date(2024, 11, 13), datetime.date(2024, 12, 11),
        datetime.date(2025, 1, 15),  datetime.date(2025, 2, 12),
        datetime.date(2025, 3, 12),  datetime.date(2025, 4, 10),
        datetime.date(2025, 5, 13),  datetime.date(2025, 6, 11),
        datetime.date(2025, 7, 15),  datetime.date(2025, 8, 12),
        datetime.date(2025, 9, 10),  datetime.date(2025, 10, 15),
        datetime.date(2025, 11, 12), datetime.date(2025, 12, 10),
        datetime.date(2026, 1, 14),  datetime.date(2026, 2, 11),
        datetime.date(2026, 3, 11),  datetime.date(2026, 4, 8),
        datetime.date(2026, 5, 13),  datetime.date(2026, 6, 10),
    ]

    # -- NFP (non-farm payrolls) -- first Friday of each month
    NFP_DATES: list[datetime.date] = [
        datetime.date(2024, 1, 5),  datetime.date(2024, 2, 2),
        datetime.date(2024, 3, 8),  datetime.date(2024, 4, 5),
        datetime.date(2024, 5, 3),  datetime.date(2024, 6, 7),
        datetime.date(2024, 7, 5),  datetime.date(2024, 8, 2),
        datetime.date(2024, 9, 6),  datetime.date(2024, 10, 4),
        datetime.date(2024, 11, 1), datetime.date(2024, 12, 6),
        datetime.date(2025, 1, 10), datetime.date(2025, 2, 7),
        datetime.date(2025, 3, 7),  datetime.date(2025, 4, 4),
        datetime.date(2025, 5, 2),  datetime.date(2025, 6, 6),
        datetime.date(2025, 7, 3),  datetime.date(2025, 8, 1),
        datetime.date(2025, 9, 5),  datetime.date(2025, 10, 3),
        datetime.date(2025, 11, 7), datetime.date(2025, 12, 5),
        datetime.date(2026, 1, 9),  datetime.date(2026, 2, 6),
        datetime.date(2026, 3, 6),  datetime.date(2026, 4, 3),
        datetime.date(2026, 5, 1),  datetime.date(2026, 6, 5),
    ]

    @staticmethod
    def _third_friday(year: int, month: int) -> datetime.date:
        """Return the 3rd Friday of a given month."""
        first = datetime.date(year, month, 1)
        # -- weekday 4 = Friday
        day_offset = (4 - first.weekday()) % 7
        first_friday = first + datetime.timedelta(days=day_offset)
        return first_friday + datetime.timedelta(weeks=2)

    @classmethod
    def _build_opex_dates(cls) -> list[datetime.date]:
        dates = []
        for year in range(2020, 2027):
            for month in range(1, 13):
                dates.append(cls._third_friday(year, month))
        return dates

    @property
    def OPEX_DATES(self) -> list[datetime.date]:
        if not hasattr(self, "_opex_cache"):
            self._opex_cache = self._build_opex_dates()
        return self._opex_cache

    def _all_events(self) -> list[Event]:
        events: list[Event] = []
        for d in self.FOMC_DATES:
            events.append(Event("FOMC", d, "FOMC"))
        for d in self.CPI_DATES:
            events.append(Event("CPI", d, "CPI"))
        for d in self.NFP_DATES:
            events.append(Event("NFP", d, "NFP"))
        for d in self.OPEX_DATES:
            events.append(Event("OPEX", d, "OPEX"))
        return events

    def is_near_event(self, dt: datetime.datetime, window_hours: float = 2.0) -> bool:
        """
        Returns True if dt falls within window_hours of any macro event.
        Assumes events occur at 08:30 ET (CPI/NFP) or 14:00 ET (FOMC).
        """
        d = dt.date()
        window = datetime.timedelta(hours=window_hours)

        category_times = {
            "FOMC": datetime.time(14, 0),
            "CPI":  datetime.time(8, 30),
            "NFP":  datetime.time(8, 30),
            "OPEX": datetime.time(9, 30),
        }

        for event in self._all_events():
            if abs((event.date - d).days) > 1:
                continue
            release_time = category_times.get(event.category, datetime.time(9, 30))
            event_dt = datetime.datetime.combine(event.date, release_time)
            if abs((dt - event_dt).total_seconds()) <= window.total_seconds():
                return True
        return False

    def get_upcoming_events(self, dt: datetime.datetime, days: int = 7) -> list[Event]:
        """Return all macro events within the next `days` calendar days."""
        cutoff = dt.date() + datetime.timedelta(days=days)
        today = dt.date()
        return sorted(
            [e for e in self._all_events() if today <= e.date <= cutoff],
            key=lambda e: e.date,
        )
