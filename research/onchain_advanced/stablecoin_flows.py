"""
stablecoin_flows.py -- Stablecoin supply and DEX flow analysis for SRFM.

Covers:
  - Aggregate stablecoin supply tracking (USDT, USDC, BUSD, DAI, FRAX)
  - Supply change signals: rising supply = dry powder = potential bullish
  - Stablecoin dominance and crypto-to-stablecoin ratio
  - DEX stablecoin-to-crypto flow monitoring
  - Composite signal methods: supply_change, dominance, dex_flow, blended

Signal conventions:
  - +1.0 = strong bullish (stablecoins flowing into crypto)
  - -1.0 = strong bearish (crypto rotating into stablecoins)
  - 0.0  = neutral
"""

from __future__ import annotations

import logging
import math
import sqlite3
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported stablecoins
STABLECOINS: Tuple[str, ...] = ("USDT", "USDC", "BUSD", "DAI", "FRAX", "TUSD", "USDP")

# Rough historical average stablecoin dominance for z-score baseline
HISTORICAL_STABLE_DOMINANCE_MEAN: float = 0.08   # 8% of total crypto
HISTORICAL_STABLE_DOMINANCE_STD:  float = 0.03   # 3% std

# DEX pair name conventions
DEX_STABLE_PAIRS: Tuple[str, ...] = (
    "USDC-ETH", "USDT-ETH", "DAI-ETH",
    "USDC-BTC", "USDT-BTC",
    "USDC-BNB", "USDT-BNB",
)

# Signal thresholds
SUPPLY_CHANGE_ZSCORE_CLIP: float = 3.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StablecoinSupplySnapshot:
    """Point-in-time snapshot of stablecoin supplies."""
    date:       str               # ISO date "YYYY-MM-DD"
    supplies:   Dict[str, float]  # token -> circulating supply (USD)
    timestamp:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total(self) -> float:
        return sum(self.supplies.values())

    def get(self, token: str, default: float = 0.0) -> float:
        return self.supplies.get(token.upper(), default)


@dataclass
class DexTrade:
    """A single DEX swap record."""
    pair:          str      # e.g. "USDC-ETH"
    stable_amount: float    # stablecoin side of the swap (USD)
    crypto_amount: float    # crypto side (native units)
    direction:     str      # "BUY" (stable -> crypto) or "SELL" (crypto -> stable)
    timestamp:     datetime
    tx_hash:       str = ""

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        if self.direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY or SELL, got {self.direction!r}")


# ---------------------------------------------------------------------------
# Stablecoin flow analyzer
# ---------------------------------------------------------------------------

class StablecoinFlowAnalyzer:
    """
    Analyze aggregate stablecoin supply trends and produce market signals.

    Methods:
      total_stablecoin_supply  -- sum of all tracked stablecoin supplies on a date
      supply_change            -- net USD change over a rolling window
      stablecoin_dominance     -- stablecoins / total crypto market cap
      crypto_to_stablecoin_ratio -- crypto market cap / stable supply
      signal                   -- composite [-1, +1] directional signal
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._snapshots:   Dict[str, StablecoinSupplySnapshot] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db(db_path)

    # ------------------------------------------------------------------
    # DB initialization
    # ------------------------------------------------------------------

    def _init_db(self, db_path: str) -> None:
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS stablecoin_supply (
                    date      TEXT NOT NULL,
                    token     TEXT NOT NULL,
                    supply    REAL NOT NULL,
                    cached_at TEXT NOT NULL,
                    PRIMARY KEY (date, token)
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stable_date ON stablecoin_supply (date)"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("Stablecoin DB init failed: %s", exc)
            self._conn = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_supply(
        self,
        date:     str,
        token:    str,
        supply:   float,
    ) -> None:
        """
        Record a stablecoin supply observation.

        Parameters
        ----------
        date:   ISO date string "YYYY-MM-DD"
        token:  Stablecoin ticker (e.g. "USDC")
        supply: Circulating supply in USD
        """
        token_upper = token.upper()
        snap = self._snapshots.setdefault(
            date,
            StablecoinSupplySnapshot(date=date, supplies={})
        )
        snap.supplies[token_upper] = supply

        if self._conn is not None:
            try:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO stablecoin_supply (date, token, supply, cached_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (date, token_upper, supply, datetime.now(timezone.utc).isoformat()),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.warning("Supply DB write failed: %s", exc)

    def load_from_db(self) -> int:
        """Reload all supply snapshots from SQLite. Returns row count."""
        if self._conn is None:
            return 0
        count = 0
        try:
            rows = self._conn.execute(
                "SELECT date, token, supply FROM stablecoin_supply ORDER BY date"
            ).fetchall()
            for date, token, supply in rows:
                snap = self._snapshots.setdefault(
                    date,
                    StablecoinSupplySnapshot(date=date, supplies={})
                )
                snap.supplies[token] = supply
                count += 1
        except sqlite3.Error as exc:
            logger.error("Supply DB load failed: %s", exc)
        return count

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def total_stablecoin_supply(self, date: str) -> float:
        """
        Return total stablecoin supply (USD) on a given date.

        Sums all tracked tokens. Returns 0.0 if no data available.
        """
        snap = self._snapshots.get(date)
        if snap is None:
            logger.debug("No stablecoin supply data for date %s", date)
            return 0.0
        return snap.total

    def supply_series(self, token: Optional[str] = None) -> pd.Series:
        """
        Return a date-indexed pd.Series of supply totals (or per-token).

        Parameters
        ----------
        token:
            If provided, return supply for that token only.
            If None, return aggregate total supply.
        """
        dates = sorted(self._snapshots.keys())
        if not dates:
            return pd.Series(dtype=float)

        if token is not None:
            t = token.upper()
            values = [self._snapshots[d].get(t) for d in dates]
        else:
            values = [self._snapshots[d].total for d in dates]

        idx = pd.to_datetime(dates)
        return pd.Series(values, index=idx, name=token or "total_stablecoin")

    def supply_change(
        self,
        window_days: int = 7,
        token: Optional[str] = None,
    ) -> float:
        """
        Compute net stablecoin supply change over the last N calendar days.

        A positive value means supply increased (more dry powder = potential bullish).
        A negative value means supply decreased (rotation out of stablecoins).

        Parameters
        ----------
        window_days:
            Number of days to look back.
        token:
            Specific stablecoin; if None, uses aggregate total.

        Returns
        -------
        float
            Net USD supply change.
        """
        series = self.supply_series(token)
        if len(series) < 2:
            return 0.0

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
        series = series[series.index >= cutoff] if series.index.tz is not None else series.iloc[-window_days:]

        if len(series) < 2:
            return 0.0

        return float(series.iloc[-1] - series.iloc[0])

    def stablecoin_dominance(self, crypto_market_cap: float) -> float:
        """
        Compute stablecoin dominance: total stable supply / total crypto market cap.

        Parameters
        ----------
        crypto_market_cap:
            Total crypto market cap in USD (including stablecoins).

        Returns
        -------
        float
            Ratio in [0, 1].
        """
        if crypto_market_cap <= 0:
            raise ValueError("crypto_market_cap must be positive")

        latest_date = max(self._snapshots.keys()) if self._snapshots else None
        if latest_date is None:
            return 0.0

        total_stable = self._snapshots[latest_date].total
        return total_stable / crypto_market_cap

    def crypto_to_stablecoin_ratio(
        self,
        crypto_market_cap: float,
    ) -> float:
        """
        Return the ratio of total crypto market cap to stablecoin supply.

        Lower ratio = more stablecoins relative to crypto = more dry powder.
        Higher ratio = stablecoins are a small fraction = less buying power.

        Parameters
        ----------
        crypto_market_cap:
            Total crypto market cap in USD (including stablecoins).
        """
        latest_date = max(self._snapshots.keys()) if self._snapshots else None
        if latest_date is None:
            return float("inf")

        total_stable = self._snapshots[latest_date].total
        if total_stable < 1e-6:
            return float("inf")

        return crypto_market_cap / total_stable

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def signal(
        self,
        method: str = "supply_change",
        crypto_market_cap: float = 0.0,
        window_days: int = 7,
    ) -> float:
        """
        Produce a [-1, +1] directional signal.

        Methods
        -------
        "supply_change":
            Normalize supply change by rolling std of daily changes.
            Rising supply -> positive signal (dry powder accumulating).
            Falling supply -> negative signal (rotating into crypto already occurred
            or exiting crypto).

        "dominance":
            Compare current dominance to historical baseline.
            Above historical mean -> negative signal (fear/risk-off).
            Below historical mean -> positive signal (risk-on appetite).

        "blended":
            Average of supply_change and dominance signals.
        """
        if method == "supply_change":
            return self._supply_change_signal(window_days)
        elif method == "dominance":
            if crypto_market_cap <= 0:
                logger.warning("dominance signal requires positive crypto_market_cap")
                return 0.0
            return self._dominance_signal(crypto_market_cap)
        elif method == "blended":
            sc = self._supply_change_signal(window_days)
            if crypto_market_cap > 0:
                dom = self._dominance_signal(crypto_market_cap)
                return float(np.clip(0.6 * sc + 0.4 * dom, -1.0, 1.0))
            return sc
        else:
            raise ValueError(f"Unknown signal method: {method!r}. Choose supply_change, dominance, blended")

    def _supply_change_signal(self, window_days: int = 7) -> float:
        """
        Z-score of window_days supply change vs recent daily changes.
        Clipped to [-SUPPLY_CHANGE_ZSCORE_CLIP, +SUPPLY_CHANGE_ZSCORE_CLIP]
        then divided to yield [-1, +1].
        """
        series = self.supply_series()
        if len(series) < 3:
            return 0.0

        daily_changes = series.diff().dropna()
        if len(daily_changes) < 2:
            return 0.0

        mu  = float(daily_changes.mean())
        std = float(daily_changes.std())

        # Use last window_days sum as the signal value
        recent_change = self.supply_change(window_days)

        if std < 1e-9:
            # Zero historical variance -- if supply is moving, it's a strong signal
            if abs(recent_change) < 1e-3:
                return 0.0
            # Return capped directional signal proportional to the change direction
            return float(np.clip(recent_change / (abs(mu) * window_days + 1e-9), -1.0, 1.0))

        z = (recent_change - mu * window_days) / (std * math.sqrt(window_days))
        clipped = float(np.clip(z, -SUPPLY_CHANGE_ZSCORE_CLIP, SUPPLY_CHANGE_ZSCORE_CLIP))
        return clipped / SUPPLY_CHANGE_ZSCORE_CLIP

    def _dominance_signal(self, crypto_market_cap: float) -> float:
        """
        Signal based on stablecoin dominance deviation from historical mean.
        High dominance (risk-off) -> negative signal.
        Low dominance (risk-on) -> positive signal.
        """
        dominance = self.stablecoin_dominance(crypto_market_cap)
        z = (dominance - HISTORICAL_STABLE_DOMINANCE_MEAN) / HISTORICAL_STABLE_DOMINANCE_STD
        # Invert: high dominance = bearish
        clipped = float(np.clip(-z, -SUPPLY_CHANGE_ZSCORE_CLIP, SUPPLY_CHANGE_ZSCORE_CLIP))
        return clipped / SUPPLY_CHANGE_ZSCORE_CLIP

    # ------------------------------------------------------------------
    # Time-series signal
    # ------------------------------------------------------------------

    def supply_change_signal_series(
        self,
        window_days: int = 7,
        zscore_window: int = 90,
    ) -> pd.Series:
        """
        Compute rolling supply change signal across the full history.

        Returns
        -------
        pd.Series
            Signal in [-1, +1] indexed by date.
        """
        series = self.supply_series()
        if len(series) < window_days + 2:
            return pd.Series(dtype=float)

        changes = series.diff(window_days).dropna()
        rolling_mean = changes.rolling(zscore_window, min_periods=10).mean()
        rolling_std  = changes.rolling(zscore_window, min_periods=10).std()

        z = (changes - rolling_mean) / (rolling_std + 1e-9)
        signal = (z / SUPPLY_CHANGE_ZSCORE_CLIP).clip(-1.0, 1.0)
        signal.name = "stablecoin_supply_signal"
        return signal


# ---------------------------------------------------------------------------
# DEX stablecoin monitor
# ---------------------------------------------------------------------------

class DexStablecoinMonitor:
    """
    Monitor DEX trading of stablecoin-to-crypto pairs.

    Aggregates buy/sell flow across pairs and produces short-term
    directional signals based on net stablecoin inflow to crypto.

    Signal logic:
      - Net stablecoin buys (stable -> crypto) = bullish signal
      - Net stablecoin sells (crypto -> stable) = bearish signal
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        # pair -> deque of DexTrade
        self._trades:   Dict[str, Deque[DexTrade]] = defaultdict(
            lambda: deque(maxlen=50_000)
        )
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db(db_path)

    def _init_db(self, db_path: str) -> None:
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS dex_trades (
                    tx_hash        TEXT PRIMARY KEY,
                    pair           TEXT NOT NULL,
                    stable_amount  REAL NOT NULL,
                    crypto_amount  REAL NOT NULL,
                    direction      TEXT NOT NULL,
                    timestamp      TEXT NOT NULL
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dex_pair_ts ON dex_trades (pair, timestamp)"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("DEX monitor DB init failed: %s", exc)
            self._conn = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_trade(self, trade: DexTrade) -> None:
        """Record a single DEX trade."""
        self._trades[trade.pair.upper()].append(trade)
        self._persist_trade(trade)

    def record_trades(self, trades: List[DexTrade]) -> None:
        """Record a batch of DEX trades."""
        for trade in trades:
            self.record_trade(trade)

    def _persist_trade(self, trade: DexTrade) -> None:
        if self._conn is None or not trade.tx_hash:
            return
        try:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO dex_trades
                  (tx_hash, pair, stable_amount, crypto_amount, direction, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.tx_hash,
                    trade.pair.upper(),
                    trade.stable_amount,
                    trade.crypto_amount,
                    trade.direction,
                    trade.timestamp.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("DEX trade persist failed: %s", exc)

    # ------------------------------------------------------------------
    # Flow metrics
    # ------------------------------------------------------------------

    def net_stablecoin_buys(
        self,
        pair: str,
        window_hours: int = 24,
    ) -> float:
        """
        Compute net USD value of stablecoins spent on crypto (buys - sells)
        over the last window_hours.

        Parameters
        ----------
        pair:
            DEX trading pair, e.g. "USDC-ETH".
        window_hours:
            Lookback window in hours.

        Returns
        -------
        float
            Positive = net buying pressure (stables flowing into crypto).
            Negative = net selling pressure (crypto being sold for stables).
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)
        trades = [
            t for t in self._trades[pair.upper()]
            if t.timestamp >= cutoff
        ]

        net = 0.0
        for t in trades:
            if t.direction == "BUY":
                net += t.stable_amount
            else:
                net -= t.stable_amount
        return net

    def net_buys_all_pairs(self, window_hours: int = 24) -> Dict[str, float]:
        """Return net stablecoin buys for every monitored pair."""
        return {
            pair: self.net_stablecoin_buys(pair, window_hours)
            for pair in self._trades
        }

    def aggregate_net_buys(self, window_hours: int = 24) -> float:
        """Sum net stablecoin buys across all pairs."""
        return sum(self.net_buys_all_pairs(window_hours).values())

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def stable_to_crypto_flow_signal(
        self,
        window_hours: int = 4,
        normalization_window_hours: int = 168,
    ) -> float:
        """
        Produce a [-1, +1] signal based on recent net stablecoin buys.

        Method:
          1. Compute net_buys for window_hours
          2. Compute rolling hourly std over normalization_window_hours
          3. z-score the net_buys and clip to [-1, +1]

        Large net stablecoin buys -> +1 (bullish).
        Large net stablecoin sells -> -1 (bearish).
        """
        net_recent = self.aggregate_net_buys(window_hours)
        hourly_std = self._rolling_hourly_std(normalization_window_hours)

        if hourly_std < 1.0:
            # Insufficient data -- return sign only
            if abs(net_recent) < 1e-3:
                return 0.0
            return 1.0 if net_recent > 0 else -1.0

        z = net_recent / (hourly_std * window_hours)
        return float(np.clip(z / 3.0, -1.0, 1.0))

    def _rolling_hourly_std(self, window_hours: int) -> float:
        """
        Compute the std of per-hour net buy aggregates over the last N hours.
        Used for normalizing the flow signal.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)
        hourly_buckets: Dict[int, float] = defaultdict(float)

        for pair_deque in self._trades.values():
            for t in pair_deque:
                if t.timestamp < cutoff:
                    continue
                hour_idx = int((t.timestamp - cutoff).total_seconds() // 3600)
                if t.direction == "BUY":
                    hourly_buckets[hour_idx] += t.stable_amount
                else:
                    hourly_buckets[hour_idx] -= t.stable_amount

        if len(hourly_buckets) < 2:
            return 0.0
        values = list(hourly_buckets.values())
        return float(statistics.stdev(values)) if len(values) >= 2 else 0.0

    def pair_signal(
        self,
        pair: str,
        window_hours: int = 4,
        normalization_window_hours: int = 168,
    ) -> float:
        """
        Signal for a specific pair only.
        """
        net_recent = self.net_stablecoin_buys(pair, window_hours)
        # Compute per-pair std
        hourly_buckets: Dict[int, float] = defaultdict(float)
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=normalization_window_hours)
        for t in self._trades[pair.upper()]:
            if t.timestamp < cutoff:
                continue
            hour_idx = int((t.timestamp - cutoff).total_seconds() // 3600)
            hourly_buckets[hour_idx] += t.stable_amount if t.direction == "BUY" else -t.stable_amount

        if len(hourly_buckets) < 2:
            return 0.0

        hourly_std = float(statistics.stdev(hourly_buckets.values()))
        if hourly_std < 1.0:
            return 0.0

        z = net_recent / (hourly_std * window_hours)
        return float(np.clip(z / 3.0, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, window_hours: int = 24) -> Dict[str, object]:
        """Return a summary of DEX stablecoin flow activity."""
        pair_flows = self.net_buys_all_pairs(window_hours)
        total_volume = sum(
            t.stable_amount
            for pair_deque in self._trades.values()
            for t in pair_deque
            if t.timestamp >= datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)
        )
        return {
            "window_hours":    window_hours,
            "aggregate_flow":  self.aggregate_net_buys(window_hours),
            "flow_signal_4h":  self.stable_to_crypto_flow_signal(window_hours=4),
            "total_volume_usd": total_volume,
            "pair_flows":      pair_flows,
        }
