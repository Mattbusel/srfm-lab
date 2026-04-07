"""
bridge/onchain_bridge.py

Real-time on-chain signal aggregator for Bitcoin and Ethereum.
Pulls from free public APIs: CoinGecko, alternative.me Fear & Greed,
Binance/Bybit funding rate endpoints. Produces composite signals
consumed by the IAE loop.

All HTTP calls are synchronous (requests library). Every fetcher
is wrapped in try/except; on failure the last cached value is returned
so callers never receive an exception.

Signal convention: positive = bullish, negative = bearish, range [-1, 1].
"""

from __future__ import annotations

import logging
import math
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent / "data" / "onchain_signals.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_FEAR_GREED_URL = "https://api.alternative.me/fng/"
_BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_BYBIT_FUNDING_URL = "https://api.bybit.com/v5/market/funding/history"

_TTL_DEFAULT = 3600       # 1 hour
_TTL_FUNDING = 900        # 15 minutes
_TTL_FEAR_GREED = 3600    # 1 hour

_REQUEST_TIMEOUT = 15     # seconds

# Composite weights (must sum to 1.0)
_WEIGHTS = {
    "mvrv": 0.25,
    "sopr": 0.20,
    "funding": 0.20,
    "oi": 0.15,
    "exchange_reserve": 0.10,
    "fear_greed": 0.10,
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OnChainMetric:
    name: str
    value: float
    zscore: float
    signal_direction: float   # [-1, 1]
    timestamp: float
    source: str


@dataclass
class OnChainComposite:
    symbol: str
    composite_signal: float        # [-1, 1]
    mvrv_signal: float
    sopr_signal: float
    funding_signal: float
    oi_signal: float
    exchange_reserve_signal: float
    fear_greed_signal: float
    timestamp: float
    confidence: float              # 0..1 based on data freshness


# ---------------------------------------------------------------------------
# Internal cache helper
# ---------------------------------------------------------------------------


class _TTLCache:
    """Simple in-process TTL cache keyed by string."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str, ttl: float) -> Optional[object]:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > ttl:
            return None
        return value

    def set(self, key: str, value: object) -> None:
        self._store[key] = (time.time(), value)

    def get_or_stale(self, key: str) -> Optional[object]:
        """Return cached value regardless of age (stale fallback)."""
        entry = self._store.get(key)
        if entry is None:
            return None
        return entry[1]


_cache = _TTLCache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_get(url: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("HTTP request failed for %s: %s", url, exc)
        return None


def _zscore_of(value: float, history: List[float]) -> float:
    if len(history) < 2:
        return 0.0
    mu = statistics.mean(history)
    sd = statistics.stdev(history)
    if sd < 1e-12:
        return 0.0
    return (value - mu) / sd


# ---------------------------------------------------------------------------
# MVRV Calculator
# ---------------------------------------------------------------------------


class MVRVCalculator:
    """
    Market-Value-to-Realized-Value Z-score proxy using CoinGecko price history.

    True MVRV requires on-chain realized cap data. This proxy uses the ratio
    of current market cap to the 365-day average market cap as a stand-in,
    then computes the Z-score against the trailing standard deviation.

    Thresholds: zscore > 3 = extreme greed (short signal -1),
                zscore < -1 = extreme fear (long signal +1).
    """

    _CACHE_KEY = "mvrv_{symbol}"
    _DAYS_HISTORY = 365

    def fetch(self, symbol: str = "bitcoin") -> OnChainMetric:
        key = self._CACHE_KEY.format(symbol=symbol)
        cached = _cache.get(key, _TTL_DEFAULT)
        if cached is not None:
            return cached  # type: ignore[return-value]

        metric = self._compute(symbol)
        _cache.set(key, metric)
        return metric

    def _compute(self, symbol: str) -> OnChainMetric:
        url = f"{_COINGECKO_BASE}/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": str(self._DAYS_HISTORY), "interval": "daily"}
        data = _safe_get(url, params)

        if data is None:
            stale = _cache.get_or_stale(self._CACHE_KEY.format(symbol=symbol))
            if stale is not None:
                return stale  # type: ignore[return-value]
            return OnChainMetric(
                name="mvrv_zscore", value=0.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="coingecko"
            )

        market_caps = [pt[1] for pt in data.get("market_caps", []) if pt[1]]
        if len(market_caps) < 30:
            return OnChainMetric(
                name="mvrv_zscore", value=0.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="coingecko"
            )

        current_mc = market_caps[-1]
        realized_cap_proxy = statistics.mean(market_caps)  # 365d average as realized cap proxy
        sd = statistics.stdev(market_caps)
        if sd < 1.0:
            sd = 1.0

        mvrv_raw = current_mc / realized_cap_proxy if realized_cap_proxy > 0 else 1.0
        zscore = (current_mc - realized_cap_proxy) / sd

        # Signal: zscore > 3 is extreme greed -> short (-1)
        #         zscore < -1 is extreme fear  -> long (+1)
        if zscore > 7:
            signal = -1.0
        elif zscore > 3:
            signal = _clamp(-0.5 * (zscore - 3) / 4)
        elif zscore < -1:
            signal = _clamp(0.5 * abs(zscore + 1) / 2)
        else:
            # Neutral zone: mild signal proportional to z
            signal = _clamp(-zscore * 0.1)

        return OnChainMetric(
            name="mvrv_zscore",
            value=mvrv_raw,
            zscore=zscore,
            signal_direction=signal,
            timestamp=time.time(),
            source="coingecko",
        )


# ---------------------------------------------------------------------------
# SOPR Tracker
# ---------------------------------------------------------------------------


class SOPRTracker:
    """
    Spent Output Profit Ratio proxy via price vs 30-day moving average.

    True SOPR = (price_spent_at_output_creation / current_price) for on-chain UTXOs.
    Proxy: current_price / MA30_price. Values > 1 means recent buyers are in profit
    (continuation); < 1 means recent buyers are underwater (capitulation).
    """

    _CACHE_KEY = "sopr_{symbol}"
    _MA_DAYS = 30

    def fetch(self, symbol: str = "bitcoin") -> OnChainMetric:
        key = self._CACHE_KEY.format(symbol=symbol)
        cached = _cache.get(key, _TTL_DEFAULT)
        if cached is not None:
            return cached  # type: ignore[return-value]

        metric = self._compute(symbol)
        _cache.set(key, metric)
        return metric

    def _compute(self, symbol: str) -> OnChainMetric:
        url = f"{_COINGECKO_BASE}/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": "60", "interval": "daily"}
        data = _safe_get(url, params)

        if data is None:
            stale = _cache.get_or_stale(self._CACHE_KEY.format(symbol=symbol))
            if stale is not None:
                return stale  # type: ignore[return-value]
            return OnChainMetric(
                name="sopr", value=1.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="coingecko"
            )

        prices = [pt[1] for pt in data.get("prices", []) if pt[1]]
        if len(prices) < self._MA_DAYS + 1:
            return OnChainMetric(
                name="sopr", value=1.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="coingecko"
            )

        current_price = prices[-1]
        ma30 = statistics.mean(prices[-self._MA_DAYS:])
        sopr_proxy = current_price / ma30 if ma30 > 0 else 1.0

        # Compute z-score over recent history
        sopr_history = [
            prices[i] / statistics.mean(prices[max(0, i - self._MA_DAYS):i])
            for i in range(self._MA_DAYS, len(prices))
            if statistics.mean(prices[max(0, i - self._MA_DAYS):i]) > 0
        ]
        zscore = _zscore_of(sopr_proxy, sopr_history)

        # SOPR > 1 and rising -> continuation (mild bullish if already trending)
        # SOPR < 1 -> capitulation -> potential reversal long signal
        if sopr_proxy < 0.97:
            signal = 0.6   # capitulation -> buying opportunity
        elif sopr_proxy < 1.0:
            signal = 0.2
        elif sopr_proxy > 1.10:
            signal = -0.4  # extended profit taking -> distribution risk
        elif sopr_proxy > 1.05:
            signal = -0.1
        else:
            signal = 0.1 * (sopr_proxy - 1.0) / 0.05

        return OnChainMetric(
            name="sopr",
            value=sopr_proxy,
            zscore=zscore,
            signal_direction=_clamp(signal),
            timestamp=time.time(),
            source="coingecko",
        )


# ---------------------------------------------------------------------------
# Exchange Reserve Monitor
# ---------------------------------------------------------------------------


class ExchangeReserveMonitor:
    """
    Exchange inflow/outflow proxy via volume patterns.

    Uses CoinGecko 24h volume trend. Rising volume + price decline = inflow
    (distribution, bearish). Rising volume + price rise = accumulation
    (bullish). Net outflow proxy: volume decreasing = accumulation phase.
    """

    _CACHE_KEY = "exchange_reserve_{symbol}"

    def fetch(self, symbol: str = "bitcoin") -> OnChainMetric:
        key = self._CACHE_KEY.format(symbol=symbol)
        cached = _cache.get(key, _TTL_DEFAULT)
        if cached is not None:
            return cached  # type: ignore[return-value]

        metric = self._compute(symbol)
        _cache.set(key, metric)
        return metric

    def _compute(self, symbol: str) -> OnChainMetric:
        url = f"{_COINGECKO_BASE}/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": "14", "interval": "daily"}
        data = _safe_get(url, params)

        if data is None:
            stale = _cache.get_or_stale(self._CACHE_KEY.format(symbol=symbol))
            if stale is not None:
                return stale  # type: ignore[return-value]
            return OnChainMetric(
                name="exchange_reserve", value=0.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="coingecko"
            )

        prices = [pt[1] for pt in data.get("prices", []) if pt[1]]
        volumes = [pt[1] for pt in data.get("total_volumes", []) if pt[1]]

        if len(prices) < 7 or len(volumes) < 7:
            return OnChainMetric(
                name="exchange_reserve", value=0.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="coingecko"
            )

        # Compare last 3 days to prior 4 days
        recent_prices = prices[-3:]
        prior_prices = prices[-7:-3]
        recent_volumes = volumes[-3:]
        prior_volumes = volumes[-7:-3]

        price_change = (statistics.mean(recent_prices) / statistics.mean(prior_prices)) - 1.0
        volume_change = (statistics.mean(recent_volumes) / statistics.mean(prior_volumes)) - 1.0

        # Volume rising + price rising = accumulation (outflow from exchanges) -> bullish
        # Volume rising + price falling = distribution (inflow to exchanges) -> bearish
        # Volume falling = accumulation phase -> mildly bullish
        if price_change > 0 and volume_change > 0.1:
            signal = 0.5 * min(1.0, volume_change)
        elif price_change < 0 and volume_change > 0.1:
            signal = -0.5 * min(1.0, volume_change)
        elif volume_change < -0.1:
            signal = 0.3  # declining volume = accumulation
        else:
            signal = 0.0

        zscore = _zscore_of(volume_change, [
            (volumes[i] / volumes[i-1]) - 1.0
            for i in range(1, len(volumes))
            if volumes[i-1] > 0
        ])

        return OnChainMetric(
            name="exchange_reserve",
            value=volume_change,
            zscore=zscore,
            signal_direction=_clamp(signal),
            timestamp=time.time(),
            source="coingecko",
        )


# ---------------------------------------------------------------------------
# Funding Rate Tracker
# ---------------------------------------------------------------------------


class FundingRateTracker:
    """
    Composite funding rate across major perpetual contracts.
    Sources: Binance USDT-perp and Bybit.

    Funding > +0.05% per 8h = crowded long -> fade (bearish signal).
    Funding < -0.05% per 8h = crowded short -> fade (bullish signal).
    """

    _CACHE_KEY = "funding_{symbol}"
    _SYMBOLS_BINANCE = ["BTCUSDT", "ETHUSDT"]
    _SYMBOLS_BYBIT = ["BTCUSDT", "ETHUSDT"]

    def fetch(self, symbol: str = "BTC") -> OnChainMetric:
        key = self._CACHE_KEY.format(symbol=symbol)
        cached = _cache.get(key, _TTL_FUNDING)
        if cached is not None:
            return cached  # type: ignore[return-value]

        metric = self._compute(symbol)
        _cache.set(key, metric)
        return metric

    def _compute(self, symbol: str) -> OnChainMetric:
        rates: List[float] = []

        # Binance
        bn_sym = f"{symbol}USDT"
        bn_data = _safe_get(self._binance_url(bn_sym))
        if bn_data and isinstance(bn_data, list):
            for entry in bn_data[-5:]:
                try:
                    rates.append(float(entry["fundingRate"]))
                except (KeyError, ValueError, TypeError):
                    pass

        # Bybit
        by_data = _safe_get(_BYBIT_FUNDING_URL, {
            "category": "linear", "symbol": f"{symbol}USDT", "limit": "5"
        })
        if by_data and isinstance(by_data, dict):
            result_list = by_data.get("result", {}).get("list", [])
            for entry in result_list:
                try:
                    rates.append(float(entry["fundingRate"]))
                except (KeyError, ValueError, TypeError):
                    pass

        if not rates:
            stale = _cache.get_or_stale(self._CACHE_KEY.format(symbol=symbol))
            if stale is not None:
                return stale  # type: ignore[return-value]
            return OnChainMetric(
                name="funding_rate", value=0.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="binance/bybit"
            )

        avg_rate = statistics.mean(rates)
        # Normalize: typical range roughly [-0.001, 0.001] per 8h
        # High positive funding -> crowded long -> fade signal (bearish)
        # High negative funding -> crowded short -> fade signal (bullish)
        threshold = 0.0003
        if avg_rate > threshold:
            signal = _clamp(-avg_rate / 0.001)
        elif avg_rate < -threshold:
            signal = _clamp(-avg_rate / 0.001)   # negative rate -> positive signal
        else:
            signal = 0.0

        zscore = _zscore_of(avg_rate, rates) if len(rates) > 1 else 0.0

        return OnChainMetric(
            name="funding_rate",
            value=avg_rate,
            zscore=zscore,
            signal_direction=_clamp(signal),
            timestamp=time.time(),
            source="binance/bybit",
        )

    @staticmethod
    def _binance_url(symbol: str) -> str:
        return f"{_BINANCE_FUNDING_URL}?symbol={symbol}&limit=5"


# ---------------------------------------------------------------------------
# Open Interest Analyzer
# ---------------------------------------------------------------------------


class OpenInterestAnalyzer:
    """
    Open interest change relative to price movement.

    OI rising + price rising = trend confirmation (bullish signal).
    OI rising + price falling = potential short squeeze / reversal (bearish).
    OI falling = deleveraging, reduces conviction.

    Fetches from Binance futures OI endpoint.
    """

    _CACHE_KEY = "oi_{symbol}"
    _BINANCE_OI_URL = "https://fapi.binance.com/fapi/v1/openInterestHist"

    def fetch(self, symbol: str = "BTC") -> OnChainMetric:
        key = self._CACHE_KEY.format(symbol=symbol)
        cached = _cache.get(key, _TTL_DEFAULT)
        if cached is not None:
            return cached  # type: ignore[return-value]

        metric = self._compute(symbol)
        _cache.set(key, metric)
        return metric

    def _compute(self, symbol: str) -> OnChainMetric:
        bn_sym = f"{symbol}USDT"
        data = _safe_get(self._BINANCE_OI_URL, {
            "symbol": bn_sym, "period": "1h", "limit": "24"
        })

        if data is None or not isinstance(data, list) or len(data) < 4:
            stale = _cache.get_or_stale(self._CACHE_KEY.format(symbol=symbol))
            if stale is not None:
                return stale  # type: ignore[return-value]
            return OnChainMetric(
                name="open_interest", value=0.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="binance"
            )

        oi_values = []
        for entry in data:
            try:
                oi_values.append(float(entry["sumOpenInterest"]))
            except (KeyError, ValueError, TypeError):
                pass

        if len(oi_values) < 4:
            return OnChainMetric(
                name="open_interest", value=0.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="binance"
            )

        recent_oi = statistics.mean(oi_values[-4:])
        prior_oi = statistics.mean(oi_values[:4])
        oi_change = (recent_oi / prior_oi - 1.0) if prior_oi > 0 else 0.0

        zscore = _zscore_of(oi_change, [
            oi_values[i] / oi_values[i-1] - 1.0
            for i in range(1, len(oi_values))
            if oi_values[i-1] > 0
        ])

        # OI rising strongly -> trend has legs (mildly bullish if no price divergence)
        # OI falling -> deleveraging -> neutral to bearish
        if oi_change > 0.05:
            signal = 0.4
        elif oi_change > 0.02:
            signal = 0.2
        elif oi_change < -0.05:
            signal = -0.3
        elif oi_change < -0.02:
            signal = -0.1
        else:
            signal = 0.0

        return OnChainMetric(
            name="open_interest",
            value=oi_change,
            zscore=zscore,
            signal_direction=_clamp(signal),
            timestamp=time.time(),
            source="binance",
        )


# ---------------------------------------------------------------------------
# Whale Signal Detector
# ---------------------------------------------------------------------------


class WhaleSignalDetector:
    """
    Large transaction proxy via volume spikes greater than 3 sigma above 24h MA.
    Filters noise using 24h moving average. Sudden volume spike without
    corresponding price move = whale accumulation (bullish). Spike with
    price drop = whale distribution (bearish).
    """

    _CACHE_KEY = "whale_{symbol}"

    def fetch(self, symbol: str = "bitcoin") -> OnChainMetric:
        key = self._CACHE_KEY.format(symbol=symbol)
        cached = _cache.get(key, _TTL_DEFAULT)
        if cached is not None:
            return cached  # type: ignore[return-value]

        metric = self._compute(symbol)
        _cache.set(key, metric)
        return metric

    def _compute(self, symbol: str) -> OnChainMetric:
        url = f"{_COINGECKO_BASE}/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": "7", "interval": "hourly"}
        data = _safe_get(url, params)

        if data is None:
            stale = _cache.get_or_stale(self._CACHE_KEY.format(symbol=symbol))
            if stale is not None:
                return stale  # type: ignore[return-value]
            return OnChainMetric(
                name="whale_signal", value=0.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="coingecko"
            )

        prices_raw = [(pt[0], pt[1]) for pt in data.get("prices", []) if pt[1]]
        volumes_raw = [(pt[0], pt[1]) for pt in data.get("total_volumes", []) if pt[1]]

        if len(prices_raw) < 25 or len(volumes_raw) < 25:
            return OnChainMetric(
                name="whale_signal", value=0.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="coingecko"
            )

        prices = [p[1] for p in prices_raw[-25:]]
        volumes = [v[1] for v in volumes_raw[-25:]]

        current_vol = volumes[-1]
        vol_ma24 = statistics.mean(volumes[-25:-1])
        vol_sd24 = statistics.stdev(volumes[-25:-1]) if len(volumes) >= 3 else 1.0

        zscore = (current_vol - vol_ma24) / (vol_sd24 + 1e-10)

        price_change = (prices[-1] / prices[-2] - 1.0) if prices[-2] > 0 else 0.0

        if zscore > 3.0:
            # Whale-level volume
            if price_change > 0.005:
                signal = 0.5   # volume spike + price up = accumulation
            elif price_change < -0.005:
                signal = -0.5  # volume spike + price down = distribution
            else:
                signal = 0.2   # neutral spike = ambiguous, slight bullish bias
        elif zscore > 2.0:
            signal = 0.1 * math.copysign(1.0, price_change)
        else:
            signal = 0.0

        return OnChainMetric(
            name="whale_signal",
            value=zscore,
            zscore=zscore,
            signal_direction=_clamp(signal),
            timestamp=time.time(),
            source="coingecko",
        )


# ---------------------------------------------------------------------------
# Fear and Greed Composite
# ---------------------------------------------------------------------------


class FearGreedComposite:
    """
    Fetches the alternative.me Crypto Fear & Greed Index.
    Normalizes index (0-100) to [-1, 1]:
      0 = extreme fear -> bullish contrarian signal (+1)
      100 = extreme greed -> bearish contrarian signal (-1)
    """

    _CACHE_KEY = "fear_greed"

    def fetch(self) -> OnChainMetric:
        cached = _cache.get(self._CACHE_KEY, _TTL_FEAR_GREED)
        if cached is not None:
            return cached  # type: ignore[return-value]

        metric = self._compute()
        _cache.set(self._CACHE_KEY, metric)
        return metric

    def _compute(self) -> OnChainMetric:
        data = _safe_get(_FEAR_GREED_URL, {"limit": "1"})

        if data is None:
            stale = _cache.get_or_stale(self._CACHE_KEY)
            if stale is not None:
                return stale  # type: ignore[return-value]
            return OnChainMetric(
                name="fear_greed", value=50.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="alternative.me"
            )

        try:
            raw_value = float(data["data"][0]["value"])
        except (KeyError, IndexError, ValueError, TypeError):
            return OnChainMetric(
                name="fear_greed", value=50.0, zscore=0.0,
                signal_direction=0.0, timestamp=time.time(), source="alternative.me"
            )

        # Normalize: 0 = extreme fear -> +1, 100 = extreme greed -> -1
        # signal = 1 - (2 * value / 100)
        signal = 1.0 - (2.0 * raw_value / 100.0)

        return OnChainMetric(
            name="fear_greed",
            value=raw_value,
            zscore=(raw_value - 50.0) / 25.0,   # rough z relative to neutral
            signal_direction=_clamp(signal),
            timestamp=time.time(),
            source="alternative.me",
        )


# ---------------------------------------------------------------------------
# On-Chain Signal Aggregator
# ---------------------------------------------------------------------------


class OnChainSignalAggregator:
    """
    Combines all individual signals into a single composite per asset.
    Weights as defined in _WEIGHTS at module level.
    """

    def __init__(self) -> None:
        self._mvrv = MVRVCalculator()
        self._sopr = SOPRTracker()
        self._exchange = ExchangeReserveMonitor()
        self._funding = FundingRateTracker()
        self._oi = OpenInterestAnalyzer()
        self._whale = WhaleSignalDetector()
        self._fear_greed = FearGreedComposite()

    def compute(self, symbol: str = "BTC") -> OnChainComposite:
        cg_symbol = "bitcoin" if symbol.upper() in ("BTC", "BITCOIN") else "ethereum"
        fn_symbol = symbol.upper() if symbol.upper() in ("BTC", "ETH") else "BTC"

        mvrv_m = self._mvrv.fetch(cg_symbol)
        sopr_m = self._sopr.fetch(cg_symbol)
        exchange_m = self._exchange.fetch(cg_symbol)
        funding_m = self._funding.fetch(fn_symbol)
        oi_m = self._oi.fetch(fn_symbol)
        fear_greed_m = self._fear_greed.fetch()

        # Weighted composite
        composite = (
            _WEIGHTS["mvrv"] * mvrv_m.signal_direction
            + _WEIGHTS["sopr"] * sopr_m.signal_direction
            + _WEIGHTS["funding"] * funding_m.signal_direction
            + _WEIGHTS["oi"] * oi_m.signal_direction
            + _WEIGHTS["exchange_reserve"] * exchange_m.signal_direction
            + _WEIGHTS["fear_greed"] * fear_greed_m.signal_direction
        )

        # Confidence: fraction of signals that are non-zero (have data)
        signals = [
            mvrv_m.signal_direction, sopr_m.signal_direction,
            funding_m.signal_direction, oi_m.signal_direction,
            exchange_m.signal_direction, fear_greed_m.signal_direction,
        ]
        confidence = sum(1 for s in signals if s != 0.0) / len(signals)

        return OnChainComposite(
            symbol=symbol.upper(),
            composite_signal=_clamp(composite),
            mvrv_signal=mvrv_m.signal_direction,
            sopr_signal=sopr_m.signal_direction,
            funding_signal=funding_m.signal_direction,
            oi_signal=oi_m.signal_direction,
            exchange_reserve_signal=exchange_m.signal_direction,
            fear_greed_signal=fear_greed_m.signal_direction,
            timestamp=time.time(),
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _init_db(conn: sqlite3.Connection) -> None:
    schema_path = Path(__file__).parent / "onchain_schema.sql"
    if schema_path.exists():
        conn.executescript(schema_path.read_text())
    else:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS onchain_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                composite_signal REAL NOT NULL,
                mvrv_signal REAL,
                sopr_signal REAL,
                funding_signal REAL,
                oi_signal REAL,
                exchange_reserve_signal REAL,
                fear_greed_signal REAL,
                confidence REAL,
                timestamp REAL NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_onchain_symbol_ts
                ON onchain_signals(symbol, timestamp);
        """)
    conn.commit()


# ---------------------------------------------------------------------------
# OnChainBridge (main entry point)
# ---------------------------------------------------------------------------


class OnChainBridge:
    """
    Orchestrates all fetchers, manages SQLite persistence, and provides
    `get_signal(sym)` -> float in [-1, 1] for the IAE loop.

    Cache TTL:
      - Most signals: 1 hour
      - Funding rate: 15 minutes

    On HTTP failure, returns last cached composite from DB.
    Never raises.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._aggregator = OnChainSignalAggregator()
        self._conn: Optional[sqlite3.Connection] = None
        self._last_composite: Dict[str, OnChainComposite] = {}

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            _init_db(self._conn)
        return self._conn

    def refresh(self, symbol: str = "BTC") -> OnChainComposite:
        """Fetch all signals and persist to DB. Returns the composite."""
        try:
            composite = self._aggregator.compute(symbol)
            self._last_composite[symbol.upper()] = composite
            self._persist(composite)
            return composite
        except Exception as exc:
            logger.error("OnChainBridge.refresh failed for %s: %s", symbol, exc)
            return self._load_latest(symbol) or OnChainComposite(
                symbol=symbol.upper(),
                composite_signal=0.0,
                mvrv_signal=0.0, sopr_signal=0.0, funding_signal=0.0,
                oi_signal=0.0, exchange_reserve_signal=0.0, fear_greed_signal=0.0,
                timestamp=time.time(), confidence=0.0,
            )

    def get_signal(self, symbol: str = "BTC") -> float:
        """
        Return composite on-chain signal for symbol in [-1, 1].
        Refreshes if data is stale (>1h). Returns 0.0 on any error.
        """
        sym = symbol.upper()
        try:
            last = self._last_composite.get(sym)
            if last is None or (time.time() - last.timestamp) > _TTL_DEFAULT:
                last = self.refresh(sym)
            return _clamp(last.composite_signal)
        except Exception as exc:
            logger.error("get_signal error for %s: %s", sym, exc)
            return 0.0

    def get_full_composite(self, symbol: str = "BTC") -> Optional[OnChainComposite]:
        """Return full composite dataclass. Refreshes if stale."""
        sym = symbol.upper()
        try:
            last = self._last_composite.get(sym)
            if last is None or (time.time() - last.timestamp) > _TTL_DEFAULT:
                last = self.refresh(sym)
            return last
        except Exception as exc:
            logger.error("get_full_composite error for %s: %s", sym, exc)
            return None

    def _persist(self, composite: OnChainComposite) -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO onchain_signals
                   (symbol, composite_signal, mvrv_signal, sopr_signal,
                    funding_signal, oi_signal, exchange_reserve_signal,
                    fear_greed_signal, confidence, timestamp, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    composite.symbol,
                    composite.composite_signal,
                    composite.mvrv_signal,
                    composite.sopr_signal,
                    composite.funding_signal,
                    composite.oi_signal,
                    composite.exchange_reserve_signal,
                    composite.fear_greed_signal,
                    composite.confidence,
                    composite.timestamp,
                    time.time(),
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("DB persist failed: %s", exc)

    def _load_latest(self, symbol: str) -> Optional[OnChainComposite]:
        try:
            conn = self._get_conn()
            row = conn.execute(
                """SELECT symbol, composite_signal, mvrv_signal, sopr_signal,
                          funding_signal, oi_signal, exchange_reserve_signal,
                          fear_greed_signal, confidence, timestamp
                   FROM onchain_signals WHERE symbol=?
                   ORDER BY timestamp DESC LIMIT 1""",
                (symbol.upper(),),
            ).fetchone()
            if row is None:
                return None
            return OnChainComposite(
                symbol=row[0],
                composite_signal=row[1],
                mvrv_signal=row[2],
                sopr_signal=row[3],
                funding_signal=row[4],
                oi_signal=row[5],
                exchange_reserve_signal=row[6],
                fear_greed_signal=row[7],
                confidence=row[8],
                timestamp=row[9],
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

    def __enter__(self) -> "OnChainBridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_bridge_singleton: Optional[OnChainBridge] = None


def get_bridge() -> OnChainBridge:
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = OnChainBridge()
    return _bridge_singleton


def get_onchain_signal(symbol: str = "BTC") -> float:
    """Convenience function: returns composite on-chain signal in [-1, 1]."""
    return get_bridge().get_signal(symbol)
