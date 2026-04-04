"""
funding_rates.py — Perpetual funding rate arbitrage analytics.

Covers:
  - Cross-exchange funding rate data ingestion
  - Basis trade P&L model (spot long + perp short)
  - Carry calculator (annualized funding rate)
  - Historical funding distribution analysis
  - Funding rate regime detection
  - Arbitrage opportunity scanner
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from data_fetchers import DiskCache, GlassnodeClient, OnChainDataProvider, RateLimiter, _build_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exchange REST clients for funding rates
# ---------------------------------------------------------------------------

FUNDING_INTERVAL_HOURS = 8    # most exchanges: 8h
PERIODS_PER_YEAR = 365 * 24 / FUNDING_INTERVAL_HOURS   # = 1095


@dataclass
class FundingRateSnapshot:
    exchange: str
    symbol: str
    funding_rate: float          # 8-hour rate (decimal, not %)
    predicted_rate: float        # next period prediction
    timestamp: int
    open_interest_usd: float

    @property
    def annualized_rate(self) -> float:
        return self.funding_rate * PERIODS_PER_YEAR

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass
class FundingHistory:
    exchange: str
    symbol: str
    rates: List[Tuple[datetime, float]]   # (time, 8h rate)

    @property
    def values(self) -> List[float]:
        return [r for _, r in self.rates]

    @property
    def mean_8h(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def mean_annual(self) -> float:
        return self.mean_8h * PERIODS_PER_YEAR

    @property
    def std_8h(self) -> float:
        return float(np.std(self.values)) if self.values else 0.0

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.values, p)) if self.values else 0.0

    def days_positive(self) -> float:
        pos = sum(1 for r in self.values if r > 0)
        return pos / len(self.values) if self.values else 0.5


# ---------------------------------------------------------------------------
# Exchange-specific clients
# ---------------------------------------------------------------------------

class BinanceFundingClient:
    BASE = "https://fapi.binance.com"

    def __init__(self, cache: DiskCache = None):
        self.session = _build_session()
        self.cache = cache or DiskCache()
        self.limiter = RateLimiter(20)

    def get_current_funding(self, symbol: str = "BTCUSDT") -> FundingRateSnapshot:
        self.limiter.wait()
        resp = self.session.get(f"{self.BASE}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=15)
        resp.raise_for_status()
        d = resp.json()
        return FundingRateSnapshot(
            exchange="binance",
            symbol=symbol,
            funding_rate=float(d.get("lastFundingRate", 0)),
            predicted_rate=float(d.get("interestRate", 0)),
            timestamp=int(d.get("time", time.time() * 1000)) // 1000,
            open_interest_usd=0.0,
        )

    def get_open_interest(self, symbol: str = "BTCUSDT") -> float:
        resp = self.session.get(f"{self.BASE}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=15)
        resp.raise_for_status()
        d = resp.json()
        return float(d.get("openInterest", 0))

    def get_funding_history(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 500,
    ) -> FundingHistory:
        resp = self.session.get(
            f"{self.BASE}/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        records = resp.json()
        rates = [
            (datetime.fromtimestamp(r["fundingTime"] / 1000, tz=timezone.utc), float(r["fundingRate"]))
            for r in records
        ]
        return FundingHistory("binance", symbol, sorted(rates))

    def get_all_funding_rates(self) -> List[Dict]:
        resp = self.session.get(f"{self.BASE}/fapi/v1/premiumIndex", timeout=15)
        resp.raise_for_status()
        return resp.json()


class BybitFundingClient:
    BASE = "https://api.bybit.com"

    def __init__(self, cache: DiskCache = None):
        self.session = _build_session()
        self.cache = cache or DiskCache()

    def get_current_funding(self, symbol: str = "BTCUSDT") -> FundingRateSnapshot:
        resp = self.session.get(f"{self.BASE}/v5/market/tickers", params={"category": "linear", "symbol": symbol}, timeout=15)
        resp.raise_for_status()
        result = resp.json().get("result", {}).get("list", [{}])[0]
        return FundingRateSnapshot(
            exchange="bybit",
            symbol=symbol,
            funding_rate=float(result.get("fundingRate", 0)),
            predicted_rate=float(result.get("nextFundingTime", 0)),
            timestamp=int(time.time()),
            open_interest_usd=float(result.get("openInterest", 0)),
        )

    def get_funding_history(self, symbol: str = "BTCUSDT", limit: int = 200) -> FundingHistory:
        resp = self.session.get(
            f"{self.BASE}/v5/market/funding/history",
            params={"category": "linear", "symbol": symbol, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        records = resp.json().get("result", {}).get("list", [])
        rates = [
            (datetime.fromtimestamp(int(r["fundingRateTimestamp"]) / 1000, tz=timezone.utc), float(r["fundingRate"]))
            for r in records
        ]
        return FundingHistory("bybit", symbol, sorted(rates))


class OKXFundingClient:
    BASE = "https://www.okx.com"

    def __init__(self, cache: DiskCache = None):
        self.session = _build_session()

    def get_current_funding(self, inst_id: str = "BTC-USDT-SWAP") -> FundingRateSnapshot:
        resp = self.session.get(
            f"{self.BASE}/api/v5/public/funding-rate",
            params={"instId": inst_id},
            timeout=15,
        )
        resp.raise_for_status()
        d = resp.json().get("data", [{}])[0]
        return FundingRateSnapshot(
            exchange="okx",
            symbol=inst_id,
            funding_rate=float(d.get("fundingRate", 0)),
            predicted_rate=float(d.get("nextFundingRate", 0)),
            timestamp=int(d.get("fundingTime", time.time() * 1000)) // 1000,
            open_interest_usd=0.0,
        )

    def get_funding_history(self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> FundingHistory:
        resp = self.session.get(
            f"{self.BASE}/api/v5/public/funding-rate-history",
            params={"instId": inst_id, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        records = resp.json().get("data", [])
        rates = [
            (datetime.fromtimestamp(int(r["fundingTime"]) / 1000, tz=timezone.utc), float(r["fundingRate"]))
            for r in records
        ]
        return FundingHistory("okx", inst_id, sorted(rates))


# ---------------------------------------------------------------------------
# Cross-exchange aggregator
# ---------------------------------------------------------------------------

class CrossExchangeFundingAggregator:
    """Fetches and aggregates funding rates across exchanges."""

    SYMBOL_MAP = {
        "BTC": {"binance": "BTCUSDT", "bybit": "BTCUSDT", "okx": "BTC-USDT-SWAP"},
        "ETH": {"binance": "ETHUSDT", "bybit": "ETHUSDT", "okx": "ETH-USDT-SWAP"},
        "SOL": {"binance": "SOLUSDT", "bybit": "SOLUSDT", "okx": "SOL-USDT-SWAP"},
        "BNB": {"binance": "BNBUSDT", "bybit": "BNBUSDT", "okx": "BNB-USDT-SWAP"},
    }

    def __init__(self, cache: DiskCache = None):
        cache = cache or DiskCache()
        self.binance = BinanceFundingClient(cache)
        self.bybit   = BybitFundingClient(cache)
        self.okx     = OKXFundingClient(cache)

    def get_all_current(self, asset: str = "BTC") -> Dict[str, FundingRateSnapshot]:
        symbols = self.SYMBOL_MAP.get(asset.upper(), {})
        results = {}
        for exchange, symbol in symbols.items():
            try:
                if exchange == "binance":
                    results[exchange] = self.binance.get_current_funding(symbol)
                elif exchange == "bybit":
                    results[exchange] = self.bybit.get_current_funding(symbol)
                elif exchange == "okx":
                    results[exchange] = self.okx.get_current_funding(symbol)
            except Exception as exc:
                logger.warning("Failed to fetch %s funding from %s: %s", asset, exchange, exc)
        return results

    def get_spread(self, asset: str = "BTC") -> Dict:
        """Returns the funding spread between highest and lowest exchange."""
        snapshots = self.get_all_current(asset)
        if not snapshots:
            return {}
        rates = {ex: s.funding_rate for ex, s in snapshots.items()}
        max_ex = max(rates, key=rates.get)
        min_ex = min(rates, key=rates.get)
        spread_8h = rates[max_ex] - rates[min_ex]
        return {
            "asset": asset,
            "max_exchange": max_ex,
            "max_rate_8h": rates[max_ex],
            "max_rate_annual": rates[max_ex] * PERIODS_PER_YEAR,
            "min_exchange": min_ex,
            "min_rate_8h": rates[min_ex],
            "min_rate_annual": rates[min_ex] * PERIODS_PER_YEAR,
            "spread_8h": spread_8h,
            "spread_annual": spread_8h * PERIODS_PER_YEAR,
            "all_rates": rates,
        }

    def get_history(self, asset: str = "BTC", exchange: str = "binance") -> FundingHistory:
        symbols = self.SYMBOL_MAP.get(asset.upper(), {})
        sym = symbols.get(exchange)
        if not sym:
            raise ValueError(f"No symbol mapping for {asset}/{exchange}")
        if exchange == "binance":
            return self.binance.get_funding_history(sym)
        if exchange == "bybit":
            return self.bybit.get_funding_history(sym)
        if exchange == "okx":
            return self.okx.get_funding_history(sym)
        raise ValueError(f"Unknown exchange: {exchange}")


# ---------------------------------------------------------------------------
# Basis trade P&L model
# ---------------------------------------------------------------------------

@dataclass
class BasisTradeSetup:
    asset: str
    exchange: str
    spot_price: float
    perp_price: float
    funding_rate_8h: float
    position_size_usd: float
    entry_timestamp: datetime
    # Costs
    spot_fee_pct: float = 0.001      # 0.1%
    perp_fee_pct: float = 0.0005     # 0.05% maker
    spot_slippage_pct: float = 0.0005
    perp_slippage_pct: float = 0.0005
    borrow_rate_annual: float = 0.05  # 5% annual if spot is borrowed/margined


@dataclass
class BasisTradePnL:
    setup: BasisTradeSetup
    days_held: float
    funding_periods: int
    gross_funding_usd: float     # total funding received
    entry_cost_usd: float
    exit_cost_usd: float
    borrow_cost_usd: float
    net_pnl_usd: float
    net_pnl_pct: float           # vs notional
    annualized_return: float
    basis_pnl_usd: float         # from spot-perp convergence


class BasisTradeCalculator:
    """
    Models the P&L of a cash-and-carry basis trade:
    Buy spot + short perpetual, collect funding.
    """

    def calculate(
        self,
        setup: BasisTradeSetup,
        days_held: float = 30.0,
        avg_funding_rate_8h: Optional[float] = None,
    ) -> BasisTradePnL:
        if avg_funding_rate_8h is None:
            avg_funding_rate_8h = setup.funding_rate_8h

        # Number of 8-hour funding periods
        periods = int(days_held * 24 / FUNDING_INTERVAL_HOURS)
        notional = setup.position_size_usd

        # Entry costs: pay taker fee on spot + perp entry
        entry_cost = notional * (setup.spot_fee_pct + setup.perp_fee_pct +
                                  setup.spot_slippage_pct + setup.perp_slippage_pct)

        # Exit costs (mirror of entry)
        exit_cost = entry_cost

        # Funding income: short perp receives funding when rate > 0
        gross_funding = notional * avg_funding_rate_8h * periods

        # Borrow/carry cost if capital is deployed
        borrow_cost = notional * setup.borrow_rate_annual * days_held / 365

        # Basis P&L: difference between spot and perp prices at entry/exit
        # At expiry (or close), spot-perp spread should converge to 0
        basis_at_entry = setup.perp_price - setup.spot_price
        basis_pnl = basis_at_entry * (notional / setup.spot_price)   # long spot, short perp

        net_pnl = gross_funding + basis_pnl - entry_cost - exit_cost - borrow_cost
        net_pct = net_pnl / notional
        annualized = (1 + net_pct) ** (365.25 / days_held) - 1 if days_held > 0 else 0.0

        return BasisTradePnL(
            setup=setup,
            days_held=days_held,
            funding_periods=periods,
            gross_funding_usd=gross_funding,
            entry_cost_usd=entry_cost,
            exit_cost_usd=exit_cost,
            borrow_cost_usd=borrow_cost,
            net_pnl_usd=net_pnl,
            net_pnl_pct=net_pct,
            annualized_return=annualized,
            basis_pnl_usd=basis_pnl,
        )

    def breakeven_funding_rate(
        self,
        notional: float,
        days: float,
        spot_fee: float = 0.001,
        perp_fee: float = 0.0005,
        slippage: float = 0.001,
        borrow_annual: float = 0.05,
    ) -> float:
        """Minimum 8h funding rate to break even."""
        periods = int(days * 24 / FUNDING_INTERVAL_HOURS)
        if periods == 0:
            return float("inf")
        total_cost = notional * (2 * spot_fee + 2 * perp_fee + 2 * slippage + borrow_annual * days / 365)
        return total_cost / (notional * periods)

    def scenario_table(
        self,
        setup: BasisTradeSetup,
        rate_scenarios: List[float] = None,
        days_scenarios: List[float] = None,
    ) -> List[Dict]:
        if rate_scenarios is None:
            rate_scenarios = [-0.001, 0.0, 0.0001, 0.0003, 0.001, 0.003]
        if days_scenarios is None:
            days_scenarios = [7, 14, 30, 60, 90]

        rows = []
        for rate in rate_scenarios:
            for days in days_scenarios:
                pnl = self.calculate(setup, days, rate)
                rows.append({
                    "rate_8h": rate,
                    "rate_annual_pct": rate * PERIODS_PER_YEAR * 100,
                    "days": days,
                    "net_pnl_usd": pnl.net_pnl_usd,
                    "net_pnl_pct": pnl.net_pnl_pct * 100,
                    "annualized_return_pct": pnl.annualized_return * 100,
                })
        return rows


# ---------------------------------------------------------------------------
# Carry calculator
# ---------------------------------------------------------------------------

class CarryCalculator:
    """
    Annualizes funding rates and computes carry across assets and exchanges.
    """

    def annualize_funding(self, rate_8h: float, compounding: bool = True) -> float:
        """Convert 8h funding rate to annualized rate."""
        if compounding:
            return (1 + rate_8h) ** PERIODS_PER_YEAR - 1
        return rate_8h * PERIODS_PER_YEAR

    def effective_carry(
        self,
        funding_rate_8h: float,
        borrow_rate_annual: float = 0.05,
        fee_drag_annual: float = 0.02,
    ) -> float:
        """Net carry after borrow + fee costs."""
        gross = self.annualize_funding(funding_rate_8h)
        return gross - borrow_rate_annual - fee_drag_annual

    def rank_assets_by_carry(
        self,
        rates: Dict[str, float],    # {asset: 8h_rate}
        borrow_rate: float = 0.05,
    ) -> List[Dict]:
        """Rank assets by effective carry."""
        rows = []
        for asset, rate in rates.items():
            annual = self.annualize_funding(rate)
            carry = self.effective_carry(rate, borrow_rate)
            rows.append({
                "asset": asset,
                "rate_8h": rate,
                "rate_8h_pct": rate * 100,
                "annualized_pct": annual * 100,
                "effective_carry_pct": carry * 100,
                "is_attractive": carry > 0.02,   # > 2% net carry threshold
            })
        return sorted(rows, key=lambda r: r["effective_carry_pct"], reverse=True)

    def compute_historical_carry(
        self,
        history: FundingHistory,
        borrow_rate: float = 0.05,
    ) -> Dict:
        vals = history.values
        if not vals:
            return {}
        annual_rates = [self.annualize_funding(r) for r in vals]
        carries = [r - borrow_rate for r in annual_rates]
        return {
            "mean_annual": float(np.mean(annual_rates)),
            "median_annual": float(np.median(annual_rates)),
            "mean_carry": float(np.mean(carries)),
            "std_carry": float(np.std(carries)),
            "sharpe_carry": float(np.mean(carries) / np.std(carries)) if np.std(carries) > 0 else 0.0,
            "pct_positive": float(np.mean([1 for c in carries if c > 0])),
            "worst_annual": float(np.min(annual_rates)),
            "best_annual": float(np.max(annual_rates)),
        }


# ---------------------------------------------------------------------------
# Historical distribution analyzer
# ---------------------------------------------------------------------------

class FundingDistributionAnalyzer:
    """
    Analyzes statistical properties of historical funding rates.
    Fits distributions, computes tail risks, detects regime changes.
    """

    def analyze(self, history: FundingHistory) -> Dict:
        vals = np.array(history.values)
        if len(vals) == 0:
            return {}

        annual_vals = vals * PERIODS_PER_YEAR

        # Basic stats
        stats = {
            "count": len(vals),
            "mean_8h": float(np.mean(vals)),
            "median_8h": float(np.median(vals)),
            "std_8h": float(np.std(vals)),
            "skewness": float(self._skewness(vals)),
            "kurtosis": float(self._kurtosis(vals)),
            "mean_annual_pct": float(np.mean(annual_vals) * 100),
            "median_annual_pct": float(np.median(annual_vals) * 100),
        }

        # Percentiles
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            stats[f"p{p}_annual_pct"] = float(np.percentile(annual_vals, p) * 100)

        # Regime detection: split into positive and negative funding regimes
        pos_vals = vals[vals > 0]
        neg_vals = vals[vals < 0]
        stats["pct_positive_funding"] = float(len(pos_vals) / len(vals)) if len(vals) > 0 else 0.5
        stats["avg_positive_annual_pct"] = float(np.mean(pos_vals * PERIODS_PER_YEAR) * 100) if len(pos_vals) > 0 else 0.0
        stats["avg_negative_annual_pct"] = float(np.mean(neg_vals * PERIODS_PER_YEAR) * 100) if len(neg_vals) > 0 else 0.0

        # Consecutive runs analysis
        runs = self._count_runs(vals)
        stats["max_positive_run"] = runs["max_positive"]
        stats["max_negative_run"] = runs["max_negative"]
        stats["avg_positive_run"] = runs["avg_positive"]
        stats["avg_negative_run"] = runs["avg_negative"]

        return stats

    @staticmethod
    def _skewness(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    @staticmethod
    def _kurtosis(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 4:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4) - 3)

    @staticmethod
    def _count_runs(vals: np.ndarray) -> Dict:
        positive_runs, negative_runs = [], []
        current_run = 0
        current_sign = None
        for v in vals:
            s = "pos" if v > 0 else "neg"
            if s == current_sign:
                current_run += 1
            else:
                if current_sign == "pos":
                    positive_runs.append(current_run)
                elif current_sign == "neg":
                    negative_runs.append(current_run)
                current_sign = s
                current_run = 1
        return {
            "max_positive": max(positive_runs) if positive_runs else 0,
            "max_negative": max(negative_runs) if negative_runs else 0,
            "avg_positive": float(np.mean(positive_runs)) if positive_runs else 0.0,
            "avg_negative": float(np.mean(negative_runs)) if negative_runs else 0.0,
        }

    def detect_regime_change(
        self,
        history: FundingHistory,
        window: int = 21,
        threshold_std: float = 2.0,
    ) -> List[Dict]:
        """Detect periods where funding rate deviates significantly from rolling mean."""
        vals = history.values
        if len(vals) < window + 1:
            return []

        changes = []
        for i in range(window, len(vals)):
            segment = vals[i - window : i]
            mean = float(np.mean(segment))
            std = float(np.std(segment))
            current = vals[i]
            z = (current - mean) / std if std > 0 else 0.0
            if abs(z) >= threshold_std:
                dt, _ = history.rates[i]
                changes.append({
                    "dt": dt.isoformat(),
                    "rate_8h": current,
                    "rate_annual_pct": current * PERIODS_PER_YEAR * 100,
                    "z_score": z,
                    "rolling_mean_annual_pct": mean * PERIODS_PER_YEAR * 100,
                    "regime": "positive_spike" if z > 0 else "negative_spike",
                })
        return changes


# ---------------------------------------------------------------------------
# Funding rate regime classifier
# ---------------------------------------------------------------------------

class FundingRegimeClassifier:
    """Classifies the current funding rate environment."""

    THRESHOLDS = {
        "extreme_positive": 0.005,    # > 0.5% per 8h (> ~197% annual) — extreme longs paying
        "high_positive":    0.001,    # 0.1-0.5%
        "mild_positive":    0.0001,   # 0-0.1%
        "neutral":          0.0,      # ~0
        "mild_negative":   -0.0001,
        "high_negative":   -0.001,
    }

    def classify(self, rate_8h: float) -> str:
        if rate_8h >= self.THRESHOLDS["extreme_positive"]:
            return "extreme_positive"
        if rate_8h >= self.THRESHOLDS["high_positive"]:
            return "high_positive"
        if rate_8h >= self.THRESHOLDS["mild_positive"]:
            return "mild_positive"
        if rate_8h <= self.THRESHOLDS["high_negative"]:
            return "high_negative"
        if rate_8h <= self.THRESHOLDS["mild_negative"]:
            return "mild_negative"
        return "neutral"

    def regime_signal(self, regime: str) -> str:
        mapping = {
            "extreme_positive": "crowded_long — strong mean reversion short signal",
            "high_positive":    "overextended long — consider delta-neutral short",
            "mild_positive":    "slight long bias — attractive basis trade",
            "neutral":          "balanced market — no strong signal",
            "mild_negative":    "slight short bias — consider spot accumulation",
            "high_negative":    "capitulation — longs being rewarded — buy signal",
        }
        return mapping.get(regime, "unknown")


# ---------------------------------------------------------------------------
# Arbitrage opportunity scanner
# ---------------------------------------------------------------------------

@dataclass
class FundingArbitrageOpportunity:
    asset: str
    long_exchange: str
    short_exchange: str
    long_rate_8h: float
    short_rate_8h: float
    spread_8h: float
    spread_annual_pct: float
    estimated_net_carry_pct: float   # after fees
    min_holding_days: float          # to break even
    is_attractive: bool


class FundingArbitrageScanner:
    """
    Scans for cross-exchange funding arbitrage opportunities:
    Long perp on low-funding exchange, short perp on high-funding exchange.
    """

    FEE_DRAG_ANNUAL = 0.03   # 3% total fee drag (entry + exit roundtrip / year)
    MIN_ATTRACTIVE_SPREAD = 0.0003   # 0.03% per 8h = ~33% annual

    def __init__(self, aggregator: CrossExchangeFundingAggregator):
        self.agg = aggregator
        self.carry_calc = CarryCalculator()
        self.regime_classifier = FundingRegimeClassifier()

    def scan(self, assets: List[str] = None) -> List[FundingArbitrageOpportunity]:
        if assets is None:
            assets = ["BTC", "ETH", "SOL"]
        opportunities = []
        for asset in assets:
            try:
                opp = self._scan_asset(asset)
                if opp:
                    opportunities.append(opp)
            except Exception as exc:
                logger.warning("Arb scan failed for %s: %s", asset, exc)
        return sorted(opportunities, key=lambda o: o.spread_annual_pct, reverse=True)

    def _scan_asset(self, asset: str) -> Optional[FundingArbitrageOpportunity]:
        snapshots = self.agg.get_all_current(asset)
        if len(snapshots) < 2:
            return None

        rates = {ex: s.funding_rate for ex, s in snapshots.items()}
        long_ex  = min(rates, key=rates.get)
        short_ex = max(rates, key=rates.get)

        long_rate  = rates[long_ex]
        short_rate = rates[short_ex]
        spread_8h  = short_rate - long_rate

        if spread_8h <= 0:
            return None

        spread_annual = spread_8h * PERIODS_PER_YEAR
        net_carry = spread_annual - self.FEE_DRAG_ANNUAL

        # Days to break even: fee drag / daily spread
        daily_spread = spread_annual / 365
        min_days = self.FEE_DRAG_ANNUAL / daily_spread if daily_spread > 0 else float("inf")

        return FundingArbitrageOpportunity(
            asset=asset,
            long_exchange=long_ex,
            short_exchange=short_ex,
            long_rate_8h=long_rate,
            short_rate_8h=short_rate,
            spread_8h=spread_8h,
            spread_annual_pct=spread_annual * 100,
            estimated_net_carry_pct=net_carry * 100,
            min_holding_days=min_days,
            is_attractive=spread_8h >= self.MIN_ATTRACTIVE_SPREAD,
        )

    def full_report(self, assets: List[str] = None) -> str:
        opportunities = self.scan(assets)
        lines = [
            "=== Funding Rate Arbitrage Scanner ===",
            f"Scan time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            "",
            f"{'Asset':>6} {'Long Ex':>10} {'Short Ex':>10} {'Spread 8h':>11} "
            f"{'Spread Ann%':>12} {'Net Carry%':>11} {'Min Days':>9} {'Signal':>8}",
            "-" * 80,
        ]
        for o in opportunities:
            signal = "✓ YES" if o.is_attractive else "  no"
            lines.append(
                f"{o.asset:>6} {o.long_exchange:>10} {o.short_exchange:>10} "
                f"{o.spread_8h*100:>10.4f}% {o.spread_annual_pct:>11.1f}% "
                f"{o.estimated_net_carry_pct:>10.1f}% {o.min_holding_days:>9.1f} {signal:>8}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main FundingRateAnalytics facade
# ---------------------------------------------------------------------------

class FundingRateAnalytics:
    """Unified entry point for funding rate analysis."""

    def __init__(self, provider: OnChainDataProvider = None):
        self.provider = provider or OnChainDataProvider()
        self.agg = CrossExchangeFundingAggregator()
        self.carry = CarryCalculator()
        self.basis_calc = BasisTradeCalculator()
        self.dist_analyzer = FundingDistributionAnalyzer()
        self.arb_scanner = FundingArbitrageScanner(self.agg)
        self.regime = FundingRegimeClassifier()

    def current_snapshot(self, asset: str = "BTC") -> Dict:
        snapshots = self.agg.get_all_current(asset)
        spread = self.agg.get_spread(asset)

        by_exchange = {}
        for ex, snap in snapshots.items():
            by_exchange[ex] = {
                "rate_8h": snap.funding_rate,
                "rate_8h_pct": snap.funding_rate * 100,
                "annualized_pct": snap.annualized_rate * 100,
                "regime": self.regime.classify(snap.funding_rate),
                "signal": self.regime.regime_signal(self.regime.classify(snap.funding_rate)),
            }

        return {
            "asset": asset,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "by_exchange": by_exchange,
            "spread": spread,
        }

    def historical_analysis(self, asset: str = "BTC", exchange: str = "binance") -> Dict:
        history = self.agg.get_history(asset, exchange)
        dist = self.dist_analyzer.analyze(history)
        carry_stats = self.carry.compute_historical_carry(history)
        regime_changes = self.dist_analyzer.detect_regime_change(history)
        return {
            "asset": asset,
            "exchange": exchange,
            "distribution": dist,
            "carry_stats": carry_stats,
            "recent_regime_changes": regime_changes[-5:],
        }

    def basis_trade_analysis(
        self,
        asset: str = "BTC",
        spot_price: float = 50000.0,
        notional: float = 100_000.0,
        days: float = 30.0,
    ) -> Dict:
        snapshots = self.agg.get_all_current(asset)
        results = {}
        for ex, snap in snapshots.items():
            setup = BasisTradeSetup(
                asset=asset,
                exchange=ex,
                spot_price=spot_price,
                perp_price=spot_price * (1 + snap.funding_rate * 3),  # rough premium
                funding_rate_8h=snap.funding_rate,
                position_size_usd=notional,
            )
            pnl = self.basis_calc.calculate(setup, days)
            breakeven = self.basis_calc.breakeven_funding_rate(notional, days)
            results[ex] = {
                "net_pnl_usd": pnl.net_pnl_usd,
                "net_pnl_pct": pnl.net_pnl_pct * 100,
                "annualized_return_pct": pnl.annualized_return * 100,
                "gross_funding_usd": pnl.gross_funding_usd,
                "breakeven_rate_8h": breakeven,
                "is_profitable": pnl.net_pnl_usd > 0,
            }
        return {
            "asset": asset,
            "notional_usd": notional,
            "days": days,
            "by_exchange": results,
        }

    def arb_opportunities(self) -> List[FundingArbitrageOpportunity]:
        return self.arb_scanner.scan()

    def full_dashboard(self, assets: List[str] = None) -> str:
        if assets is None:
            assets = ["BTC", "ETH"]
        lines = ["=" * 70, "FUNDING RATE DASHBOARD", "=" * 70]
        for asset in assets:
            snap = self.current_snapshot(asset)
            lines.append(f"\n{asset}:")
            for ex, d in snap["by_exchange"].items():
                lines.append(f"  {ex:12s}: {d['rate_8h_pct']:+.4f}%/8h "
                             f"({d['annualized_pct']:+.1f}%/yr) — {d['regime']}")
        lines.append("")
        lines.append(self.arb_scanner.full_report(assets))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Funding rates CLI")
    parser.add_argument("--action", choices=["dashboard", "snapshot", "arb", "basis"], default="dashboard")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--notional", type=float, default=100_000)
    parser.add_argument("--days", type=float, default=30)
    args = parser.parse_args()

    analytics = FundingRateAnalytics()

    if args.action == "dashboard":
        print(analytics.full_dashboard())
    elif args.action == "snapshot":
        snap = analytics.current_snapshot(args.asset)
        print(_json.dumps(snap, indent=2))
    elif args.action == "arb":
        opps = analytics.arb_opportunities()
        for o in opps:
            print(f"{o.asset}: {o.long_exchange} vs {o.short_exchange} — "
                  f"spread {o.spread_annual_pct:.1f}% annual, net carry {o.estimated_net_carry_pct:.1f}%")
    elif args.action == "basis":
        result = analytics.basis_trade_analysis(args.asset, notional=args.notional, days=args.days)
        print(_json.dumps(result, indent=2))
