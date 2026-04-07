"""
bridge/microstructure_bridge.py

Real-time microstructure analytics bridge.

Wraps the Rust microstructure-engine crate (via subprocess) and adds Python-level
analytics. Provides standalone Python implementations as fallback when the Rust
binary is unavailable or the market-data service is unreachable.

Key output: sizing_multiplier in [0.5, 1.0] that reduces position size when
adverse microstructure conditions are detected (high VPIN, toxic flow, illiquidity).

Market-data REST endpoint: http://localhost:8780/snapshot
SQLite output: microstructure_signals table in data/live_trades.db
"""

from __future__ import annotations

import logging
import math
import sqlite3
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MARKET_DATA_URL = "http://localhost:8780/snapshot"
_DB_PATH = Path(__file__).parent.parent / "data" / "live_trades.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_POLL_INTERVAL = 900   # 15 minutes in seconds
_REQUEST_TIMEOUT = 5

# OFI thresholds
_OFI_BUY_THRESHOLD = 0.6
_OFI_SELL_THRESHOLD = -0.6

# VPIN thresholds
_VPIN_TOXIC_THRESHOLD = 0.7
_VPIN_BUCKET_COUNT = 50

# Adverse selection threshold
_ADVERSE_REALIZED_QUOTED_THRESHOLD = 0.3

# Kyle lambda validation tolerance
_KYLE_LAMBDA_TOLERANCE = 0.1

# Amihud threshold (ratio above which we consider market illiquid)
_AMIHUD_ILLIQUID_THRESHOLD = 1.5   # relative to historical average

# Composite signal weights
_MICRO_WEIGHTS = {
    "ofi": 0.40,
    "vpin": 0.30,
    "adverse_selection": 0.20,
    "amihud": 0.10,
}

# Sizing multiplier range
_SIZING_MIN = 0.5
_SIZING_MAX = 1.0

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class L2Snapshot:
    """L2 order book snapshot from market-data service."""
    symbol: str
    timestamp: float
    bids: List[Tuple[float, float]]   # (price, size) sorted descending
    asks: List[Tuple[float, float]]   # (price, size) sorted ascending
    last_price: float
    volume_24h: float


@dataclass
class MicrostructureReading:
    """Composite microstructure reading for a symbol."""
    symbol: str
    composite_signal: float          # [-1, 1] (bullish/bearish)
    toxicity_score: float            # [0, 1] higher = more toxic flow
    liquidity_score: float           # [0, 1] higher = more liquid
    sizing_multiplier: float         # [0.5, 1.0] applied to position size
    ofi_signal: float
    vpin: float
    adverse_selection_ratio: float
    kyle_lambda: float
    amihud_ratio: float
    bid_ask_spread: float
    roll_spread_estimate: float
    timestamp: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_request(url: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("HTTP request failed for %s: %s", url, exc)
        return None


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Order Flow Imbalance
# ---------------------------------------------------------------------------


class OrderFlowImbalance:
    """
    Bid/ask volume imbalance from L2 book snapshots.

    OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    Rolling 20-bar OFI:
      OFI > 0.6 = strong buy pressure
      OFI < -0.6 = strong sell pressure

    Fetches snapshots from localhost:8780/snapshot.
    """

    _WINDOW = 20

    def __init__(self) -> None:
        self._history: Dict[str, Deque[float]] = {}

    def compute_from_snapshot(self, snapshot: L2Snapshot) -> float:
        """Compute single-bar OFI from L2 snapshot. Returns value in [-1, 1]."""
        bid_vol = sum(size for _, size in snapshot.bids[:10])
        ask_vol = sum(size for _, size in snapshot.asks[:10])
        total = bid_vol + ask_vol
        if total < 1e-12:
            return 0.0
        return (bid_vol - ask_vol) / total

    def update(self, symbol: str, snapshot: L2Snapshot) -> float:
        """Update rolling OFI and return rolling average."""
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._WINDOW)
        bar_ofi = self.compute_from_snapshot(snapshot)
        self._history[symbol].append(bar_ofi)
        history = list(self._history[symbol])
        return statistics.mean(history) if history else 0.0

    def get_signal(self, symbol: str) -> float:
        """Return OFI-based directional signal in [-1, 1]."""
        history = self._history.get(symbol)
        if not history:
            return 0.0
        rolling_ofi = statistics.mean(history)
        if rolling_ofi > _OFI_BUY_THRESHOLD:
            return _clamp(rolling_ofi)
        elif rolling_ofi < _OFI_SELL_THRESHOLD:
            return _clamp(rolling_ofi)
        else:
            return _clamp(rolling_ofi / _OFI_BUY_THRESHOLD)

    def compute_from_volumes(self, bid_vol: float, ask_vol: float) -> float:
        """Compute OFI directly from volumes."""
        total = bid_vol + ask_vol
        if total < 1e-12:
            return 0.0
        return (bid_vol - ask_vol) / total


# ---------------------------------------------------------------------------
# VPIN Calculator
# ---------------------------------------------------------------------------


class VPINCalculator:
    """
    Volume-synchronized Probability of Informed Trading.

    Algorithm:
      1. Divide cumulative volume into buckets of size V_bar / 50 (where V_bar
         is the average daily volume over the lookback period).
      2. Classify each trade as buy or sell using the tick rule:
         price up from last = buy, price down = sell, unchanged = previous.
      3. VPIN = mean(|buy_vol - sell_vol| / V_bar) over rolling 50 buckets.

    High VPIN (> 0.7) = toxic order flow warning.
    Output is in [0, 1].
    """

    _BUCKET_COUNT = _VPIN_BUCKET_COUNT
    _ROLLING_WINDOW = 50

    def __init__(self) -> None:
        self._buckets: Dict[str, Deque[float]] = {}
        self._current_bucket: Dict[str, Dict] = {}
        self._vbar: Dict[str, float] = {}

    def set_vbar(self, symbol: str, vbar: float) -> None:
        """Set the average volume per bucket unit."""
        self._vbar[symbol] = max(vbar, 1.0)
        if symbol not in self._buckets:
            self._buckets[symbol] = deque(maxlen=self._ROLLING_WINDOW)
        if symbol not in self._current_bucket:
            self._current_bucket[symbol] = {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0}

    def process_trade(self, symbol: str, price: float, volume: float, prev_price: float) -> Optional[float]:
        """
        Process a single trade. Returns VPIN estimate when a bucket is complete,
        otherwise None.
        """
        if symbol not in self._vbar:
            return None

        bucket_size = self._vbar[symbol] / self._BUCKET_COUNT
        direction = _sign(price - prev_price)
        if direction == 0.0:
            direction = 1.0   # default to buy on no movement

        bucket = self._current_bucket[symbol]
        if direction > 0:
            bucket["buy_vol"] += volume
        else:
            bucket["sell_vol"] += volume
        bucket["total_vol"] += volume

        if bucket["total_vol"] >= bucket_size:
            imbalance = abs(bucket["buy_vol"] - bucket["sell_vol"]) / self._vbar[symbol]
            self._buckets[symbol].append(imbalance)
            self._current_bucket[symbol] = {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0}
            return self.get_vpin(symbol)

        return None

    def process_trades(self, symbol: str, trades: List[Tuple[float, float]], vbar: float) -> float:
        """
        Process a batch of (price, volume) trades and return VPIN estimate.
        Automatically sets vbar.
        """
        self.set_vbar(symbol, vbar)
        vpin = 0.0
        prev_price = trades[0][0] if trades else 0.0
        for price, volume in trades:
            result = self.process_trade(symbol, price, volume, prev_price)
            if result is not None:
                vpin = result
            prev_price = price
        return vpin

    def get_vpin(self, symbol: str) -> float:
        """Return current VPIN estimate in [0, 1]."""
        buckets = self._buckets.get(symbol)
        if not buckets:
            return 0.0
        return min(1.0, statistics.mean(buckets))

    def compute_from_trades(self, prices: List[float], volumes: List[float]) -> float:
        """Stateless VPIN computation from price/volume arrays."""
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0
        vbar = statistics.mean(volumes) if volumes else 1.0
        bucket_size = vbar / self._BUCKET_COUNT
        buckets: List[float] = []
        buy_vol = 0.0
        sell_vol = 0.0
        total_vol = 0.0

        for i in range(1, len(prices)):
            direction = _sign(prices[i] - prices[i - 1])
            if direction == 0:
                direction = 1.0
            vol = volumes[i]
            if direction > 0:
                buy_vol += vol
            else:
                sell_vol += vol
            total_vol += vol

            if total_vol >= bucket_size:
                imbalance = abs(buy_vol - sell_vol) / vbar
                buckets.append(imbalance)
                buy_vol = 0.0
                sell_vol = 0.0
                total_vol = 0.0

        if not buckets:
            return 0.0
        return min(1.0, statistics.mean(buckets[-self._ROLLING_WINDOW:]))


# ---------------------------------------------------------------------------
# Adverse Selection Monitor
# ---------------------------------------------------------------------------


class AdverseSelectionMonitor:
    """
    Tracks realized spread vs quoted spread ratio.

    Realized spread = 2 * d_i * (p_{i+k} - p_i) where d_i is trade direction,
    k is a fixed horizon (e.g., 5 trades ahead).
    Quoted spread = ask - bid at time of trade.

    When realized/quoted < 0.3, informed trading is suspected and the
    sizing multiplier is reduced to 0.7x.

    Also computes effective spread and price impact.
    """

    _HORIZON = 5

    def __init__(self) -> None:
        self._history: Dict[str, Deque[float]] = {}

    def compute_realized_spread(
        self,
        prices: List[float],
        directions: List[float],
        quoted_spread: float,
    ) -> float:
        """
        Compute average realized spread from price series.
        directions: +1 for buy, -1 for sell.
        Returns ratio of realized to quoted spread.
        """
        if len(prices) < self._HORIZON + 2 or quoted_spread < 1e-10:
            return 1.0   # assume normal conditions

        realized_spreads = []
        for i in range(len(prices) - self._HORIZON):
            d = directions[i]
            midpoint_change = d * (prices[i + self._HORIZON] - prices[i])
            realized = 2.0 * midpoint_change
            realized_spreads.append(realized)

        if not realized_spreads:
            return 1.0

        avg_realized = statistics.mean(realized_spreads)
        return avg_realized / quoted_spread if quoted_spread > 0 else 1.0

    def compute_effective_spread(
        self, trade_price: float, mid_price: float, direction: float
    ) -> float:
        """Effective spread = 2 * |trade_price - mid_price|."""
        return 2.0 * abs(trade_price - mid_price)

    def compute_price_impact(
        self, pre_trade_mid: float, post_trade_mid: float, direction: float
    ) -> float:
        """Price impact = direction * (post_trade_mid - pre_trade_mid)."""
        return direction * (post_trade_mid - pre_trade_mid)

    def get_sizing_adjustment(self, realized_quoted_ratio: float) -> float:
        """
        Returns sizing multiplier based on adverse selection level.
        ratio < 0.3 -> suspected informed trading -> 0.7x
        ratio >= 0.3 -> normal conditions -> 1.0x
        """
        if realized_quoted_ratio < _ADVERSE_REALIZED_QUOTED_THRESHOLD:
            return 0.7
        return 1.0

    def is_informed_trading(self, realized_quoted_ratio: float) -> bool:
        return realized_quoted_ratio < _ADVERSE_REALIZED_QUOTED_THRESHOLD


# ---------------------------------------------------------------------------
# Kyle Lambda Estimator
# ---------------------------------------------------------------------------


class KyleLambdaEstimator:
    """
    Estimates Kyle's lambda: price impact per unit of signed order flow.

    Method: OLS regression of signed price changes on signed volume.
      delta_p = lambda * signed_volume + epsilon

    Rolling 100-trade window, updated every bar.
    lambda > 0 always (price increases on net buying pressure).

    Optionally validates against QuatNav MI proxy from nav_state table.
    """

    _WINDOW = 100

    def __init__(self) -> None:
        self._price_changes: Dict[str, Deque[float]] = {}
        self._signed_volumes: Dict[str, Deque[float]] = {}

    def update(self, symbol: str, price_change: float, signed_volume: float) -> None:
        if symbol not in self._price_changes:
            self._price_changes[symbol] = deque(maxlen=self._WINDOW)
            self._signed_volumes[symbol] = deque(maxlen=self._WINDOW)
        self._price_changes[symbol].append(price_change)
        self._signed_volumes[symbol].append(signed_volume)

    def estimate(self, symbol: str) -> float:
        """Return Kyle's lambda estimate. Always >= 0."""
        dp = list(self._price_changes.get(symbol, []))
        sv = list(self._signed_volumes.get(symbol, []))
        return self._ols_lambda(dp, sv)

    def estimate_from_arrays(
        self, price_changes: List[float], signed_volumes: List[float]
    ) -> float:
        """Stateless OLS lambda estimation."""
        return self._ols_lambda(price_changes, signed_volumes)

    @staticmethod
    def _ols_lambda(dp: List[float], sv: List[float]) -> float:
        """OLS: lambda = cov(dp, sv) / var(sv). Clamp to >= 0."""
        if len(dp) < 5 or len(sv) < 5:
            return 0.0
        n = len(dp)
        mean_dp = sum(dp) / n
        mean_sv = sum(sv) / n
        cov = sum((dp[i] - mean_dp) * (sv[i] - mean_sv) for i in range(n)) / n
        var_sv = sum((sv[i] - mean_sv) ** 2 for i in range(n)) / n
        if var_sv < 1e-12:
            return 0.0
        lam = cov / var_sv
        return max(0.0, lam)   # lambda must be non-negative

    def validate_against_nav(self, symbol: str, db_conn: sqlite3.Connection) -> Optional[float]:
        """
        Compare estimated lambda to QuatNav MI proxy from nav_state table.
        Returns relative difference or None if nav_state not available.
        """
        try:
            row = db_conn.execute(
                "SELECT angular_velocity FROM nav_state WHERE symbol=? ORDER BY ts DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if row is None:
                return None
            nav_mi_proxy = abs(row[0])
            estimated = self.estimate(symbol)
            if nav_mi_proxy < 1e-12:
                return None
            return abs(estimated - nav_mi_proxy) / nav_mi_proxy
        except Exception as exc:
            logger.debug("Kyle lambda nav validation failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Amihud Illiquidity
# ---------------------------------------------------------------------------


class AmihudIlliquidity:
    """
    Amihud illiquidity ratio: |return| / dollar_volume.

    Rolling 20-day Amihud ratio. Elevated Amihud indicates an illiquid market
    with wider effective spreads -> reduce position sizing.

    Output is normalized relative to the rolling mean so it is dimensionless.
    """

    _WINDOW = 20

    def __init__(self) -> None:
        self._history: Dict[str, Deque[float]] = {}

    def compute_single(self, abs_return: float, dollar_volume: float) -> float:
        """Single Amihud observation."""
        if dollar_volume < 1.0:
            return 0.0
        return abs_return / dollar_volume

    def update(self, symbol: str, abs_return: float, dollar_volume: float) -> float:
        """Update rolling Amihud and return current rolling mean."""
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._WINDOW)
        obs = self.compute_single(abs_return, dollar_volume)
        self._history[symbol].append(obs)
        history = list(self._history[symbol])
        return statistics.mean(history) if history else 0.0

    def get_ratio(self, symbol: str) -> float:
        """
        Return current Amihud ratio normalized to rolling mean (dimensionless).
        1.0 = average conditions; > 1.5 = elevated illiquidity.
        """
        history = list(self._history.get(symbol, []))
        if not history:
            return 1.0
        current = history[-1]
        mean = statistics.mean(history)
        if mean < 1e-20:
            return 1.0
        return current / mean

    def compute_from_arrays(
        self, returns: List[float], volumes: List[float]
    ) -> float:
        """Stateless: compute rolling Amihud ratio from arrays."""
        if len(returns) < 2 or len(volumes) < 2:
            return 0.0
        obs = [
            abs(returns[i]) / volumes[i]
            for i in range(len(returns))
            if volumes[i] > 1.0
        ]
        return statistics.mean(obs) if obs else 0.0

    def get_sizing_adjustment(self, symbol: str) -> float:
        """Return sizing multiplier: reduced when Amihud is elevated."""
        ratio = self.get_ratio(symbol)
        if ratio > _AMIHUD_ILLIQUID_THRESHOLD:
            # Linearly reduce from 1.0 to 0.7 as ratio goes from 1.5 to 3.0
            reduction = (ratio - _AMIHUD_ILLIQUID_THRESHOLD) / 1.5 * 0.3
            return max(_SIZING_MIN, 1.0 - reduction)
        return _SIZING_MAX


# ---------------------------------------------------------------------------
# Bid-Ask Bounce (Roll's Estimator)
# ---------------------------------------------------------------------------


class BidAskBounce:
    """
    Roll's estimator of bid-ask spread from serial covariance of returns.

    Spread = 2 * sqrt(-cov(r_t, r_{t-1})) if cov < 0, else 0.

    Compares to observed L2 spread. Excess bounce (Roll > observed) indicates
    market quality is degrading.
    """

    _WINDOW = 60

    def __init__(self) -> None:
        self._returns: Dict[str, Deque[float]] = {}

    def update(self, symbol: str, price_return: float) -> None:
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self._WINDOW)
        self._returns[symbol].append(price_return)

    def estimate_spread(self, symbol: str) -> float:
        """Estimate bid-ask spread via Roll's method."""
        returns = list(self._returns.get(symbol, []))
        return self._roll_spread(returns)

    def estimate_spread_from_returns(self, returns: List[float]) -> float:
        return self._roll_spread(returns)

    @staticmethod
    def _roll_spread(returns: List[float]) -> float:
        if len(returns) < 3:
            return 0.0
        n = len(returns)
        mean_r = statistics.mean(returns)
        cov = sum(
            (returns[i] - mean_r) * (returns[i - 1] - mean_r)
            for i in range(1, n)
        ) / (n - 1)
        if cov >= 0:
            return 0.0
        return 2.0 * math.sqrt(-cov)

    def is_degrading(self, symbol: str, observed_spread: float) -> bool:
        """True if Roll estimate exceeds observed L2 spread (market quality issue)."""
        roll = self.estimate_spread(symbol)
        return roll > observed_spread * 1.5


# ---------------------------------------------------------------------------
# Microstructure Signal Combiner
# ---------------------------------------------------------------------------


class MicrostructureSignal:
    """
    Combines OFI + VPIN + adverse selection + Kyle's lambda into composite signal.

    Weights: OFI 40%, VPIN 30%, adverse selection 20%, Amihud 10%.

    Outputs MicrostructureReading with:
      - composite_signal [-1, 1]
      - toxicity_score [0, 1]
      - liquidity_score [0, 1]
      - sizing_multiplier [0.5, 1.0]
    """

    def __init__(self) -> None:
        self._ofi = OrderFlowImbalance()
        self._vpin = VPINCalculator()
        self._adverse = AdverseSelectionMonitor()
        self._kyle = KyleLambdaEstimator()
        self._amihud = AmihudIlliquidity()
        self._roll = BidAskBounce()

    def compute(
        self,
        symbol: str,
        ofi_signal: float,
        vpin: float,
        realized_quoted_ratio: float,
        kyle_lambda: float,
        amihud_ratio: float,
        bid_ask_spread: float,
        roll_spread: float,
    ) -> MicrostructureReading:
        # VPIN contribution: high VPIN is bearish/toxic -> negative signal
        vpin_signal = -_clamp(vpin * 2.0 - 1.0)  # map [0,1] -> [-1, 1], high=bearish

        # Adverse selection: low ratio = suspected informed trading -> negative
        adverse_signal = _clamp((realized_quoted_ratio - 0.5) * 2.0)

        # Amihud: high = illiquid = less confidence in direction -> dampen
        amihud_signal = _clamp(1.0 - (amihud_ratio - 1.0))

        composite = (
            _MICRO_WEIGHTS["ofi"] * ofi_signal
            + _MICRO_WEIGHTS["vpin"] * vpin_signal
            + _MICRO_WEIGHTS["adverse_selection"] * adverse_signal
            + _MICRO_WEIGHTS["amihud"] * amihud_signal
        )

        # Toxicity: driven by VPIN and adverse selection
        toxicity = _clamp(
            0.5 * vpin + 0.5 * max(0.0, 1.0 - realized_quoted_ratio),
            0.0, 1.0
        )

        # Liquidity: high Amihud = low liquidity, small Roll spread = liquid
        spread_penalty = min(1.0, roll_spread / (bid_ask_spread + 1e-10)) if bid_ask_spread > 0 else 0.0
        liquidity = _clamp(1.0 - 0.5 * (amihud_ratio - 1.0) / _AMIHUD_ILLIQUID_THRESHOLD - 0.5 * spread_penalty, 0.0, 1.0)

        # Sizing multiplier: reduced by VPIN and adverse selection
        sizing = _SIZING_MAX
        if vpin > _VPIN_TOXIC_THRESHOLD:
            sizing = min(sizing, 0.5 + 0.2 * (1.0 - vpin))
        if realized_quoted_ratio < _ADVERSE_REALIZED_QUOTED_THRESHOLD:
            sizing = min(sizing, 0.7)
        if amihud_ratio > _AMIHUD_ILLIQUID_THRESHOLD:
            reduction = (amihud_ratio - _AMIHUD_ILLIQUID_THRESHOLD) / 1.5 * 0.3
            sizing = min(sizing, 1.0 - reduction)
        sizing = max(_SIZING_MIN, min(_SIZING_MAX, sizing))

        return MicrostructureReading(
            symbol=symbol,
            composite_signal=_clamp(composite),
            toxicity_score=toxicity,
            liquidity_score=liquidity,
            sizing_multiplier=sizing,
            ofi_signal=ofi_signal,
            vpin=vpin,
            adverse_selection_ratio=realized_quoted_ratio,
            kyle_lambda=kyle_lambda,
            amihud_ratio=amihud_ratio,
            bid_ask_spread=bid_ask_spread,
            roll_spread_estimate=roll_spread,
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS microstructure_signals (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol               TEXT    NOT NULL,
    composite_signal     REAL    NOT NULL,
    toxicity_score       REAL    NOT NULL,
    liquidity_score      REAL    NOT NULL,
    sizing_multiplier    REAL    NOT NULL,
    ofi_signal           REAL,
    vpin                 REAL,
    adverse_selection    REAL,
    kyle_lambda          REAL,
    amihud_ratio         REAL,
    bid_ask_spread       REAL,
    roll_spread          REAL,
    timestamp            REAL    NOT NULL,
    created_at           REAL    NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_ms_symbol_ts
    ON microstructure_signals (symbol, timestamp DESC);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_CREATE_TABLE_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Snapshot parser
# ---------------------------------------------------------------------------


def _parse_snapshot(data: dict) -> Optional[L2Snapshot]:
    """Parse JSON response from localhost:8780/snapshot into L2Snapshot."""
    try:
        bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])[:20]]
        asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])[:20]]
        return L2Snapshot(
            symbol=str(data.get("symbol", "UNKNOWN")),
            timestamp=float(data.get("timestamp", time.time())),
            bids=sorted(bids, key=lambda x: -x[0]),
            asks=sorted(asks, key=lambda x: x[0]),
            last_price=float(data.get("last_price", 0.0)),
            volume_24h=float(data.get("volume_24h", 0.0)),
        )
    except Exception as exc:
        logger.debug("Snapshot parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# MicrostructureBridge (main entry point)
# ---------------------------------------------------------------------------


class MicrostructureBridge:
    """
    Polls the market-data service every 15 minutes, updates all microstructure
    estimators, writes to SQLite, and provides get_sizing_multiplier(sym) for
    the live trader.

    Falls back to Python implementations if the Rust microstructure-engine
    subprocess is unavailable.

    Never raises on get_sizing_multiplier() -- returns 1.0 (no reduction) on error.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        market_data_url: str = _MARKET_DATA_URL,
    ) -> None:
        self._db_path = db_path or _DB_PATH
        self._market_data_url = market_data_url
        self._conn: Optional[sqlite3.Connection] = None

        # Estimators
        self._ofi = OrderFlowImbalance()
        self._vpin = VPINCalculator()
        self._adverse = AdverseSelectionMonitor()
        self._kyle = KyleLambdaEstimator()
        self._amihud = AmihudIlliquidity()
        self._roll = BidAskBounce()
        self._signal = MicrostructureSignal()

        # Per-symbol state
        self._price_history: Dict[str, Deque[float]] = {}
        self._volume_history: Dict[str, Deque[float]] = {}
        self._last_reading: Dict[str, MicrostructureReading] = {}
        self._last_poll: Dict[str, float] = {}

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            _init_db(self._conn)
        return self._conn

    def poll(self, symbol: str) -> Optional[MicrostructureReading]:
        """
        Fetch snapshot, update all estimators, persist reading.
        Returns the reading or None on failure.
        """
        try:
            data = _safe_request(self._market_data_url, {"symbol": symbol})
            if data is None:
                return self._load_latest(symbol)

            snapshot = _parse_snapshot(data)
            if snapshot is None:
                return self._load_latest(symbol)

            reading = self._update_all(symbol, snapshot)
            self._persist(reading)
            self._last_reading[symbol] = reading
            self._last_poll[symbol] = time.time()
            return reading
        except Exception as exc:
            logger.error("MicrostructureBridge.poll failed for %s: %s", symbol, exc)
            return self._load_latest(symbol)

    def _update_all(self, symbol: str, snapshot: L2Snapshot) -> MicrostructureReading:
        """Update all estimators from a fresh snapshot."""
        # Price and volume history
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=200)
            self._volume_history[symbol] = deque(maxlen=200)

        self._price_history[symbol].append(snapshot.last_price)
        self._volume_history[symbol].append(snapshot.volume_24h)

        prices = list(self._price_history[symbol])
        volumes = list(self._volume_history[symbol])

        # OFI
        ofi_signal = self._ofi.update(symbol, snapshot)

        # VPIN
        if len(prices) >= 2:
            vbar = statistics.mean(volumes) if volumes else 1.0
            vpin_val = self._vpin.compute_from_trades(prices, volumes)
        else:
            vpin_val = 0.0

        # Bid-ask spread from L2
        best_bid = snapshot.bids[0][0] if snapshot.bids else 0.0
        best_ask = snapshot.asks[0][0] if snapshot.asks else 0.0
        quoted_spread = best_ask - best_bid if best_ask > best_bid else 0.0
        mid_price = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else snapshot.last_price

        # Adverse selection (realized spread proxy)
        if len(prices) >= 10:
            returns = [prices[i] / prices[i-1] - 1.0 for i in range(1, len(prices))]
            directions = [_sign(r) for r in returns]
            realized_ratio = self._adverse.compute_realized_spread(
                prices[-20:], directions[-19:] if len(directions) >= 19 else directions, quoted_spread
            )
        else:
            realized_ratio = 1.0

        # Kyle lambda
        if len(prices) >= 10:
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            signed_vols = [
                volumes[i] * _sign(prices[i] - prices[i-1]) if i < len(volumes) else 0.0
                for i in range(1, len(prices))
            ]
            kyle_lambda = self._kyle.estimate_from_arrays(
                price_changes[-100:], signed_vols[-100:]
            )
        else:
            kyle_lambda = 0.0

        # Amihud
        if len(prices) >= 2:
            abs_ret = abs(snapshot.last_price / prices[-2] - 1.0) if prices[-2] > 0 else 0.0
            self._amihud.update(symbol, abs_ret, snapshot.volume_24h)
        amihud_ratio = self._amihud.get_ratio(symbol)

        # Roll spread
        if len(prices) >= 3:
            returns = [prices[i] / prices[i-1] - 1.0 for i in range(1, len(prices))]
            for r in returns[-5:]:
                self._roll.update(symbol, r)
        roll_spread = self._roll.estimate_spread(symbol)

        return self._signal.compute(
            symbol=symbol,
            ofi_signal=ofi_signal,
            vpin=vpin_val,
            realized_quoted_ratio=realized_ratio,
            kyle_lambda=kyle_lambda,
            amihud_ratio=amihud_ratio,
            bid_ask_spread=quoted_spread,
            roll_spread=roll_spread,
        )

    def get_sizing_multiplier(self, symbol: str) -> float:
        """
        Return sizing multiplier in [0.5, 1.0] for symbol.
        Refreshes if data is stale (> _POLL_INTERVAL). Returns 1.0 on error.
        """
        try:
            last_poll = self._last_poll.get(symbol, 0.0)
            if time.time() - last_poll > _POLL_INTERVAL:
                reading = self.poll(symbol)
            else:
                reading = self._last_reading.get(symbol)

            if reading is None:
                return 1.0
            return max(_SIZING_MIN, min(_SIZING_MAX, reading.sizing_multiplier))
        except Exception as exc:
            logger.error("get_sizing_multiplier error for %s: %s", symbol, exc)
            return 1.0

    def get_reading(self, symbol: str) -> Optional[MicrostructureReading]:
        """Return full MicrostructureReading, refreshing if stale."""
        try:
            last_poll = self._last_poll.get(symbol, 0.0)
            if time.time() - last_poll > _POLL_INTERVAL:
                return self.poll(symbol)
            return self._last_reading.get(symbol)
        except Exception as exc:
            logger.error("get_reading error for %s: %s", symbol, exc)
            return None

    def _persist(self, reading: MicrostructureReading) -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO microstructure_signals
                   (symbol, composite_signal, toxicity_score, liquidity_score,
                    sizing_multiplier, ofi_signal, vpin, adverse_selection,
                    kyle_lambda, amihud_ratio, bid_ask_spread, roll_spread,
                    timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    reading.symbol,
                    reading.composite_signal,
                    reading.toxicity_score,
                    reading.liquidity_score,
                    reading.sizing_multiplier,
                    reading.ofi_signal,
                    reading.vpin,
                    reading.adverse_selection_ratio,
                    reading.kyle_lambda,
                    reading.amihud_ratio,
                    reading.bid_ask_spread,
                    reading.roll_spread_estimate,
                    reading.timestamp,
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("DB persist failed: %s", exc)

    def _load_latest(self, symbol: str) -> Optional[MicrostructureReading]:
        try:
            conn = self._get_conn()
            row = conn.execute(
                """SELECT symbol, composite_signal, toxicity_score, liquidity_score,
                          sizing_multiplier, ofi_signal, vpin, adverse_selection,
                          kyle_lambda, amihud_ratio, bid_ask_spread, roll_spread, timestamp
                   FROM microstructure_signals WHERE symbol=?
                   ORDER BY timestamp DESC LIMIT 1""",
                (symbol,),
            ).fetchone()
            if row is None:
                return None
            return MicrostructureReading(
                symbol=row[0],
                composite_signal=row[1],
                toxicity_score=row[2],
                liquidity_score=row[3],
                sizing_multiplier=row[4],
                ofi_signal=row[5],
                vpin=row[6],
                adverse_selection_ratio=row[7],
                kyle_lambda=row[8],
                amihud_ratio=row[9],
                bid_ask_spread=row[10],
                roll_spread_estimate=row[11],
                timestamp=row[12],
            )
        except Exception as exc:
            logger.warning("DB load failed: %s", exc)
            return None

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "MicrostructureBridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bridge_singleton: Optional[MicrostructureBridge] = None


def get_bridge(db_path: Optional[Path] = None) -> MicrostructureBridge:
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = MicrostructureBridge(db_path=db_path)
    return _bridge_singleton


def get_sizing_multiplier(symbol: str) -> float:
    """Convenience function for live trader integration."""
    return get_bridge().get_sizing_multiplier(symbol)
