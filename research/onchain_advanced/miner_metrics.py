"""
miner_metrics.py -- Bitcoin miner behavior analysis for SRFM.

Covers:
  - Miner revenue estimation (block rewards + fees)
  - Miner selling pressure (estimated BTC sold to cover opex)
  - Hash rate signal: rising = miner confidence, declining = capitulation risk
  - Puell Multiple: daily issuance vs 365-day MA (valuation indicator)
  - Miner capitulation risk: price below breakeven + declining hash rate
  - Hash rate estimation from difficulty and block time
  - Electricity-cost-based breakeven price calculation

Signal conventions:
  - All signals normalized to [-1, +1] where available
  - Puell Multiple > 4 -> overbought (negative signal)
  - Puell Multiple < 0.5 -> oversold / capitulation (positive signal)
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bitcoin block subsidy schedule (halving epochs)
BLOCK_SUBSIDY_SCHEDULE: List[Tuple[int, float]] = [
    (0,       50.0),
    (210_000, 25.0),
    (420_000, 12.5),
    (630_000, 6.25),
    (840_000, 3.125),
    (1_050_000, 1.5625),
]

# Average blocks per day (10-minute target)
BLOCKS_PER_DAY: float = 144.0

# Approximate average network hash rate in EH/s at various difficulties
# Used for hash-rate estimation fallback
DIFFICULTY_TO_HASHRATE_COEFFICIENT: float = 7.158278826e-12  # EH/s per diff unit

# Puell Multiple thresholds
PUELL_OVERBOUGHT:  float = 4.0
PUELL_OVERSOLD:    float = 0.5
PUELL_NEUTRAL_LOW: float = 0.5
PUELL_NEUTRAL_HIGH: float = 4.0

# Estimated average miner breakeven -- mid-tier ASIC at 5 cents/kWh
# Updated periodically; used as a default when no electricity cost is provided
DEFAULT_BREAKEVEN_USD: float = 20_000.0

# Typical fraction of miner revenue sold to cover operating expenses
DEFAULT_SELL_FRACTION: float = 0.60

# Electricity consumption per TH/s in watts (Antminer S19 Pro equivalent)
EFFICIENCY_W_PER_TH: float = 29.5

# Seconds per day
SECONDS_PER_DAY: float = 86_400.0


# ---------------------------------------------------------------------------
# Hash rate estimation utility
# ---------------------------------------------------------------------------

class HashRateEstimator:
    """
    Estimate network hash rate from difficulty and block time.

    Formula:
        hash_rate (H/s) = difficulty * 2^32 / block_time_s

    The 2^32 factor comes from the SHA-256 double-hash probability math:
    one "share" of work requires on average 2^32 hashes at difficulty=1.
    """

    @staticmethod
    def estimate_from_difficulty(
        difficulty: float,
        block_time_s: float = 600.0,
    ) -> float:
        """
        Return estimated network hash rate in hashes/second.

        Parameters
        ----------
        difficulty:
            Bitcoin network difficulty (dimensionless).
        block_time_s:
            Mean inter-block time in seconds (default 600 for 10-minute target).

        Returns
        -------
        float
            Estimated hash rate in hashes per second (H/s).
        """
        if difficulty <= 0:
            raise ValueError(f"Difficulty must be positive, got {difficulty}")
        if block_time_s <= 0:
            raise ValueError(f"Block time must be positive, got {block_time_s}")
        return difficulty * (2.0 ** 32) / block_time_s

    @staticmethod
    def to_exahash(hashrate_hs: float) -> float:
        """Convert H/s to EH/s."""
        return hashrate_hs / 1e18

    @staticmethod
    def to_terahash(hashrate_hs: float) -> float:
        """Convert H/s to TH/s."""
        return hashrate_hs / 1e12

    @staticmethod
    def breakeven_price(
        electricity_cost_kwh: float = 0.05,
        efficiency_w_per_th:  float = EFFICIENCY_W_PER_TH,
        difficulty:           float = 5e13,
        block_reward_btc:     float = 3.125,
    ) -> float:
        """
        Estimate the USD price at which mining breaks even.

        Derivation:
          Revenue per TH/s per day (BTC) =
              (BLOCKS_PER_DAY * block_reward_btc) / (network_hashrate_ths)
          Cost per TH/s per day (USD) =
              efficiency_w_per_th * 24 / 1000 * electricity_cost_kwh
          Breakeven price = cost / revenue_in_btc

        Parameters
        ----------
        electricity_cost_kwh:
            Electricity cost in USD per kWh.
        efficiency_w_per_th:
            ASIC power consumption in watts per TH/s.
        difficulty:
            Current network difficulty.
        block_reward_btc:
            Current block subsidy in BTC (post-halving value).

        Returns
        -------
        float
            USD/BTC price at which mining is break-even.
        """
        if electricity_cost_kwh <= 0:
            raise ValueError("Electricity cost must be positive")

        network_hashrate_ths = HashRateEstimator.to_terahash(
            HashRateEstimator.estimate_from_difficulty(difficulty)
        )

        # BTC revenue per TH/s per day
        revenue_btc_per_ths_day = (BLOCKS_PER_DAY * block_reward_btc) / network_hashrate_ths

        # USD cost per TH/s per day
        cost_usd_per_ths_day = (efficiency_w_per_th / 1000.0) * 24.0 * electricity_cost_kwh

        if revenue_btc_per_ths_day < 1e-18:
            return float("inf")

        return cost_usd_per_ths_day / revenue_btc_per_ths_day

    @staticmethod
    def current_block_subsidy(block_height: int) -> float:
        """Return block subsidy in BTC at the given block height."""
        subsidy = 50.0
        for height, reward in reversed(BLOCK_SUBSIDY_SCHEDULE):
            if block_height >= height:
                subsidy = reward
                break
        return subsidy


# ---------------------------------------------------------------------------
# Miner metrics analyzer
# ---------------------------------------------------------------------------

class MinerMetricsAnalyzer:
    """
    Analyze miner behavior metrics to generate directional market signals.

    All methods accept either raw float inputs or pandas Series for
    vectorized time-series computation.

    Signal interpretation:
      - Positive values -> bullish (accumulation / healthy miner economics)
      - Negative values -> bearish (distribution / capitulation pressure)
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        default_sell_fraction: float = DEFAULT_SELL_FRACTION,
        default_breakeven_usd: float = DEFAULT_BREAKEVEN_USD,
    ) -> None:
        self._sell_fraction    = default_sell_fraction
        self._breakeven_usd    = default_breakeven_usd
        self._estimator        = HashRateEstimator()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db(db_path)

    def _init_db(self, db_path: str) -> None:
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS miner_revenue_cache (
                    date           TEXT PRIMARY KEY,
                    revenue_usd    REAL NOT NULL,
                    block_rewards  REAL,
                    fees_usd       REAL,
                    cached_at      TEXT NOT NULL
                )
            """)
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("Miner metrics DB init failed: %s", exc)
            self._conn = None

    # ------------------------------------------------------------------
    # Revenue
    # ------------------------------------------------------------------

    def miner_revenue(
        self,
        date: str,
        btc_price_usd: float = 0.0,
        block_count:   int   = 144,
        avg_fee_btc:   float = 0.0002,
        block_height:  int   = 840_000,
    ) -> float:
        """
        Estimate total miner revenue (USD) for a given date.

        Revenue = (block_subsidy + avg_fees_per_block) * blocks_per_day * btc_price

        Parameters
        ----------
        date:
            ISO date string "YYYY-MM-DD" (used as cache key).
        btc_price_usd:
            BTC spot price in USD on that date.
        block_count:
            Actual number of blocks mined that day (default 144).
        avg_fee_btc:
            Average transaction fees per block in BTC.
        block_height:
            Block height on that date (for subsidy lookup).

        Returns
        -------
        float
            Estimated miner revenue in USD.
        """
        # Check cache
        cached = self._read_revenue_cache(date)
        if cached is not None:
            return cached

        subsidy = HashRateEstimator.current_block_subsidy(block_height)
        revenue_btc = (subsidy + avg_fee_btc) * block_count
        revenue_usd = revenue_btc * btc_price_usd

        self._write_revenue_cache(date, revenue_usd, subsidy * block_count * btc_price_usd, avg_fee_btc * block_count * btc_price_usd)
        return revenue_usd

    def _read_revenue_cache(self, date: str) -> Optional[float]:
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT revenue_usd FROM miner_revenue_cache WHERE date = ?", (date,)
            ).fetchone()
            return float(row[0]) if row else None
        except sqlite3.Error:
            return None

    def _write_revenue_cache(
        self,
        date: str,
        revenue_usd: float,
        block_rewards: float,
        fees_usd: float,
    ) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO miner_revenue_cache
                  (date, revenue_usd, block_rewards, fees_usd, cached_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (date, revenue_usd, block_rewards, fees_usd, datetime.utcnow().isoformat()),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Revenue cache write failed: %s", exc)

    # ------------------------------------------------------------------
    # Selling pressure
    # ------------------------------------------------------------------

    def miner_selling_pressure(
        self,
        daily_revenue_usd: pd.Series,
        window_days: int = 7,
        sell_fraction: Optional[float] = None,
    ) -> pd.Series:
        """
        Estimate the USD value of BTC sold by miners to cover operating costs.

        Estimated sold = revenue * sell_fraction (rolling mean over window_days).

        Parameters
        ----------
        daily_revenue_usd:
            Time series of daily miner revenue in USD.
        window_days:
            Rolling window for smoothing.
        sell_fraction:
            Fraction of revenue sold (default: instance default).

        Returns
        -------
        pd.Series
            Estimated USD value of BTC sold per day.
        """
        sf = sell_fraction if sell_fraction is not None else self._sell_fraction
        if not (0.0 <= sf <= 1.0):
            raise ValueError(f"sell_fraction must be in [0, 1], got {sf}")
        smoothed = daily_revenue_usd.rolling(window=window_days, min_periods=1).mean()
        return smoothed * sf

    # ------------------------------------------------------------------
    # Hash rate signal
    # ------------------------------------------------------------------

    def hash_rate_signal(
        self,
        hash_rate: pd.Series,
        short_window: int = 14,
        long_window:  int = 90,
    ) -> pd.Series:
        """
        Produce a directional signal from hash rate momentum.

        Logic:
          - signal = (short_MA - long_MA) / long_MA
          - Positive = hash rate rising -> miner confidence -> bullish
          - Negative = hash rate falling -> capitulation risk -> bearish
          - Clipped to [-1, +1] via tanh normalization

        Parameters
        ----------
        hash_rate:
            Daily network hash rate time series (any consistent unit).
        short_window:
            Fast MA window in days.
        long_window:
            Slow MA window in days.

        Returns
        -------
        pd.Series
            Signal in [-1, +1].
        """
        if len(hash_rate) < 2:
            return pd.Series(np.zeros(len(hash_rate)), index=hash_rate.index)

        ma_short = hash_rate.rolling(window=short_window, min_periods=1).mean()
        ma_long  = hash_rate.rolling(window=long_window,  min_periods=1).mean()

        # Relative momentum: avoid division by zero
        momentum = (ma_short - ma_long) / (ma_long + 1e-12)

        # tanh normalization: maps momentum to (-1, +1) smoothly
        signal = np.tanh(momentum * 5.0)  # scale factor 5 chosen empirically
        return pd.Series(signal, index=hash_rate.index, name="hash_rate_signal")

    # ------------------------------------------------------------------
    # Puell Multiple
    # ------------------------------------------------------------------

    def puell_multiple(
        self,
        miner_revenue: pd.Series,
        window: int = 365,
    ) -> pd.Series:
        """
        Compute the Puell Multiple.

        Formula:
            Puell = daily_issuance_usd / rolling_mean_365d(daily_issuance_usd)

        Interpretation:
          - > 4.0: miners earning far above average -> likely selling -> bearish
          - < 0.5: miners earning far below average -> capitulation zone -> bullish
          - 0.5 - 4.0: neutral

        Parameters
        ----------
        miner_revenue:
            Daily miner revenue (USD) time series.
        window:
            Rolling mean window in days (default 365).

        Returns
        -------
        pd.Series
            Puell Multiple values (dimensionless ratio).
        """
        rolling_mean = miner_revenue.rolling(window=window, min_periods=14).mean()
        puell = miner_revenue / (rolling_mean + 1e-9)
        puell.name = "puell_multiple"
        return puell

    def puell_signal(self, puell: pd.Series) -> pd.Series:
        """
        Convert Puell Multiple to a [-1, +1] signal.

        Mapping:
          Puell <= 0.5  -> +1.0 (oversold / capitulation = buy signal)
          Puell >= 4.0  -> -1.0 (overbought = sell signal)
          Between 0.5 and 4.0 -> linear interpolation
        """
        # Normalize Puell to [0, 1] range then flip (low Puell = bullish)
        normalized = (puell - PUELL_NEUTRAL_LOW) / (PUELL_NEUTRAL_HIGH - PUELL_NEUTRAL_LOW)
        clipped    = normalized.clip(0.0, 1.0)
        # Invert: low Puell -> high signal
        signal = 1.0 - 2.0 * clipped
        signal.name = "puell_signal"
        return signal

    # ------------------------------------------------------------------
    # Miner capitulation risk
    # ------------------------------------------------------------------

    def miner_capitulation_risk(
        self,
        hash_rate:     pd.Series,
        price:         pd.Series,
        breakeven_usd: Optional[float] = None,
        hash_rate_decline_window: int = 30,
        hash_rate_decline_pct: float = 0.10,
    ) -> pd.Series:
        """
        Estimate miner capitulation risk as a value in [0, 1].

        Capitulation occurs when:
          1. Price is below the estimated miner breakeven cost
          2. Hash rate is declining (miners are shutting off machines)

        Parameters
        ----------
        hash_rate:
            Daily network hash rate.
        price:
            Daily BTC/USD closing price.
        breakeven_usd:
            Miner breakeven price in USD (defaults to DEFAULT_BREAKEVEN_USD).
        hash_rate_decline_window:
            Rolling window to measure hash rate trend (days).
        hash_rate_decline_pct:
            Threshold decline percentage to flag hash rate as falling.

        Returns
        -------
        pd.Series
            Risk score in [0, 1] where 1 = high capitulation risk.
        """
        be = breakeven_usd if breakeven_usd is not None else self._breakeven_usd

        if price.index.equals(hash_rate.index) is False:
            # Reindex to price series
            hash_rate = hash_rate.reindex(price.index, method="ffill")

        # Price stress: how far below breakeven?
        # 0 when price >= breakeven, 1 when price = 0
        price_stress = ((be - price) / be).clip(0.0, 1.0)

        # Hash rate trend: rolling % change
        hr_ma = hash_rate.rolling(window=hash_rate_decline_window, min_periods=5).mean()
        hr_change = hr_ma.pct_change(periods=hash_rate_decline_window).fillna(0.0)
        # Declining hash rate -> positive stress signal
        hr_stress = (-hr_change).clip(0.0, 1.0)

        # Combined risk: both conditions together amplify the signal
        risk = (price_stress + hr_stress) / 2.0
        risk.name = "capitulation_risk"
        return risk.clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # Composite signal
    # ------------------------------------------------------------------

    def composite_miner_signal(
        self,
        hash_rate:     pd.Series,
        price:         pd.Series,
        miner_revenue: pd.Series,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Blend hash rate signal and Puell signal into a single [-1, +1] output.

        Default weights: hash_rate=0.5, puell=0.5.
        """
        w = weights or {"hash_rate": 0.5, "puell": 0.5}
        total_w = sum(w.values())
        if abs(total_w - 1.0) > 1e-6:
            # Normalize weights
            w = {k: v / total_w for k, v in w.items()}

        hr_sig   = self.hash_rate_signal(hash_rate)
        puell    = self.puell_multiple(miner_revenue)
        pu_sig   = self.puell_signal(puell)

        # Align indexes
        common_idx = hr_sig.index.intersection(pu_sig.index)
        blended = (
            w.get("hash_rate", 0.5) * hr_sig.reindex(common_idx)
            + w.get("puell",      0.5) * pu_sig.reindex(common_idx)
        )
        blended.name = "miner_composite_signal"
        return blended.clip(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Difficulty ribbon
    # ------------------------------------------------------------------

    @staticmethod
    def difficulty_ribbon(
        hash_rate: pd.Series,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Compute the difficulty ribbon -- multiple MAs of hash rate.

        When short MAs cross below long MAs, indicates miner capitulation.
        Returns a DataFrame with one column per MA window.
        """
        ws = windows or [9, 14, 25, 40, 60, 90, 128, 200]
        ribbon: Dict[str, pd.Series] = {}
        for w in ws:
            ribbon[f"hr_ma{w}"] = hash_rate.rolling(window=w, min_periods=1).mean()
        return pd.DataFrame(ribbon, index=hash_rate.index)

    @staticmethod
    def ribbon_compression(ribbon: pd.DataFrame) -> pd.Series:
        """
        Compute ribbon compression: std across MA columns at each timestep.
        Low compression (all MAs converging) = potential inflection point.
        """
        comp = ribbon.std(axis=1)
        normalized = comp / (comp.rolling(90, min_periods=14).mean() + 1e-12)
        normalized.name = "ribbon_compression"
        return normalized
