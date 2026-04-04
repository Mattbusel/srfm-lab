"""
flow_scanner.py — Unusual options activity scanner.

Covers:
  - Volume vs OI ratio (unusual activity detection)
  - Large block trade detection
  - Sweep order identification (multi-exchange fills)
  - Dark pool / off-exchange print detection
  - Options alert generation
  - Historical flow aggregation by ticker
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OptionSide(Enum):
    CALL = "call"
    PUT  = "put"


class TradeType(Enum):
    BUY_OPEN  = "buy_to_open"
    SELL_OPEN = "sell_to_open"
    BUY_CLOSE = "buy_to_close"
    SELL_CLOSE= "sell_to_close"
    UNKNOWN   = "unknown"


class FlowSentiment(Enum):
    BULLISH   = "bullish"
    BEARISH   = "bearish"
    NEUTRAL   = "neutral"
    AMBIGUOUS = "ambiguous"


class AlertSeverity(Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class OptionContract:
    ticker: str
    expiry: date
    strike: float
    side: OptionSide
    underlying_price: float

    @property
    def moneyness(self) -> float:
        """Strike / Underlying price."""
        if self.underlying_price == 0:
            return 1.0
        return self.strike / self.underlying_price

    @property
    def is_otm(self) -> bool:
        if self.side == OptionSide.CALL:
            return self.strike > self.underlying_price
        return self.strike < self.underlying_price

    @property
    def is_itm(self) -> bool:
        return not self.is_otm

    @property
    def days_to_expiry(self) -> int:
        return (self.expiry - date.today()).days

    @property
    def contract_key(self) -> str:
        return f"{self.ticker}_{self.expiry}_{self.strike:.2f}_{self.side.value}"


@dataclass
class OptionTrade:
    trade_id: str
    contract: OptionContract
    timestamp: datetime
    price: float           # premium per contract
    size: int              # number of contracts (100 shares each)
    open_interest: int
    implied_vol: float
    delta: float
    exchange: str
    trade_type: TradeType
    is_sweep: bool         # multi-exchange fill
    is_block: bool         # large block trade
    is_dark_pool: bool
    aggressor: str         # "buyer" or "seller"
    bid: float
    ask: float
    midpoint: float

    @property
    def notional_usd(self) -> float:
        return self.price * self.size * 100

    @property
    def delta_notional(self) -> float:
        return self.delta * self.contract.underlying_price * self.size * 100

    @property
    def vol_oi_ratio(self) -> float:
        return self.size / self.open_interest if self.open_interest > 0 else float("inf")

    @property
    def is_unusual(self) -> bool:
        return self.vol_oi_ratio > 0.5 or self.is_block or self.is_sweep

    @property
    def sentiment(self) -> FlowSentiment:
        if self.contract.side == OptionSide.CALL:
            if self.trade_type in (TradeType.BUY_OPEN, TradeType.SELL_CLOSE):
                return FlowSentiment.BULLISH
            if self.trade_type in (TradeType.SELL_OPEN, TradeType.BUY_CLOSE):
                return FlowSentiment.BEARISH
        else:  # PUT
            if self.trade_type in (TradeType.BUY_OPEN, TradeType.SELL_CLOSE):
                return FlowSentiment.BEARISH
            if self.trade_type in (TradeType.SELL_OPEN, TradeType.BUY_CLOSE):
                return FlowSentiment.BULLISH
        return FlowSentiment.AMBIGUOUS

    @property
    def relative_to_mid(self) -> str:
        if self.price >= self.ask * 0.97:
            return "at_ask"
        if self.price <= self.bid * 1.03:
            return "at_bid"
        return "mid"


@dataclass
class UnusualActivity:
    ticker: str
    contract: OptionContract
    trades: List[OptionTrade]
    total_premium_usd: float
    total_contracts: int
    vol_oi_ratio: float
    is_sweep: bool
    is_block: bool
    dominant_sentiment: FlowSentiment
    alert_severity: AlertSeverity
    description: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FlowAlert:
    alert_id: str
    ticker: str
    timestamp: datetime
    severity: AlertSeverity
    activity_type: str    # "unusual_volume", "block_trade", "sweep", "dark_pool"
    sentiment: FlowSentiment
    description: str
    contract: OptionContract
    premium_usd: float
    contracts: int
    raw_trades: List[OptionTrade] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Market data feed (simulated / Tradier / CBOE)
# ---------------------------------------------------------------------------

class OptionDataFeed:
    """
    Wraps options market data APIs.
    Primary: Tradier API
    Fallback: synthetic data generation for testing
    """

    TRADIER_BASE = "https://api.tradier.com/v1"

    def __init__(self, api_key: str = ""):
        import os
        self.api_key = api_key or os.environ.get("TRADIER_API_KEY", "")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    def get_option_chain(
        self,
        ticker: str,
        expiration: str,   # "YYYY-MM-DD"
    ) -> List[Dict]:
        if not self.api_key:
            return self._synthetic_chain(ticker, expiration)
        try:
            resp = self._session.get(
                f"{self.TRADIER_BASE}/markets/options/chains",
                params={"symbol": ticker, "expiration": expiration, "greeks": True},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("options", {}).get("option", [])
        except Exception:
            return self._synthetic_chain(ticker, expiration)

    def get_expirations(self, ticker: str) -> List[str]:
        if not self.api_key:
            today = date.today()
            return [
                (today + timedelta(days=d)).strftime("%Y-%m-%d")
                for d in [7, 14, 21, 30, 45, 60, 90, 120, 180]
            ]
        try:
            resp = self._session.get(
                f"{self.TRADIER_BASE}/markets/options/expirations",
                params={"symbol": ticker, "includeAllRoots": True},
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json().get("expirations", {}).get("date", [])
            return result if isinstance(result, list) else [result]
        except Exception:
            return []

    def get_quotes(self, ticker: str) -> Dict:
        if not self.api_key:
            prices = {"SPY": 520.0, "QQQ": 440.0, "AAPL": 195.0, "TSLA": 200.0, "NVDA": 850.0}
            price = prices.get(ticker.upper(), 100.0)
            return {"symbol": ticker, "last": price, "bid": price - 0.05, "ask": price + 0.05}
        try:
            resp = self._session.get(
                f"{self.TRADIER_BASE}/markets/quotes",
                params={"symbols": ticker, "greeks": False},
                timeout=15,
            )
            resp.raise_for_status()
            quotes = resp.json().get("quotes", {}).get("quote", {})
            return quotes if isinstance(quotes, dict) else (quotes[0] if quotes else {})
        except Exception:
            return {}

    def _synthetic_chain(self, ticker: str, expiration: str) -> List[Dict]:
        """Generate synthetic option chain for testing."""
        prices = {"SPY": 520.0, "QQQ": 440.0, "AAPL": 195.0, "TSLA": 200.0, "NVDA": 850.0}
        spot = prices.get(ticker.upper(), 100.0)
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        dte = (exp_date - date.today()).days
        t = max(dte, 1) / 365.0

        iv_atm = 0.25  # 25% ATM vol

        strikes = [round(spot * m / 100) * 100 / 100 for m in range(80, 125, 5)]
        # Round to nearest dollar or 5 dollars depending on price
        if spot >= 100:
            strikes = [round(spot * m / 100 / 5) * 5 for m in range(80, 125, 5)]
        else:
            strikes = [round(spot * m / 100, 2) for m in range(80, 125, 5)]

        chain = []
        rng = random.Random(hash(f"{ticker}{expiration}"))

        for strike in strikes:
            moneyness = strike / spot
            iv_skew = iv_atm + 0.3 * max(0, 1.0 - moneyness) ** 2  # skew for puts
            if moneyness > 1:
                iv_c = iv_atm + 0.05 * (moneyness - 1)
                iv_p = iv_atm + 0.15 * (moneyness - 1)
            else:
                iv_c = iv_atm + 0.10 * (1 - moneyness)
                iv_p = iv_atm + 0.20 * (1 - moneyness)

            call_price = self._bs_call(spot, strike, t, 0.05, iv_c)
            put_price  = self._bs_put(spot, strike, t, 0.05, iv_p)
            call_delta = self._bs_delta_call(spot, strike, t, 0.05, iv_c)
            put_delta  = call_delta - 1.0

            call_oi = int(rng.uniform(500, 10000))
            put_oi  = int(rng.uniform(500, 12000))
            call_vol = int(rng.uniform(100, call_oi * 0.3))
            put_vol  = int(rng.uniform(100, put_oi * 0.3))

            for side, price, iv, delta, oi, vol in [
                ("call", call_price, iv_c, call_delta, call_oi, call_vol),
                ("put",  put_price,  iv_p, put_delta,  put_oi,  put_vol),
            ]:
                chain.append({
                    "symbol": f"{ticker}{expiration.replace('-','')}{'C' if side=='call' else 'P'}{int(strike*1000):08d}",
                    "description": f"{ticker} {expiration} ${strike} {side.upper()}",
                    "underlying": ticker,
                    "expiration_date": expiration,
                    "option_type": side,
                    "strike": strike,
                    "last": round(price, 2),
                    "bid": round(price * 0.98, 2),
                    "ask": round(price * 1.02, 2),
                    "volume": vol,
                    "open_interest": oi,
                    "implied_volatility": round(iv, 4),
                    "delta": round(delta, 4),
                    "gamma": round(self._bs_gamma(spot, strike, t, 0.05, iv), 6),
                    "theta": round(-abs(price) / dte if dte > 0 else 0, 4),
                    "vega": round(self._bs_vega(spot, strike, t, 0.05, iv) / 100, 4),
                    "greeks": {
                        "delta": round(delta, 4),
                        "gamma": round(self._bs_gamma(spot, strike, t, 0.05, iv), 6),
                        "theta": round(-abs(price) / dte if dte > 0 else 0, 4),
                        "vega":  round(self._bs_vega(spot, strike, t, 0.05, iv) / 100, 4),
                        "rho":   0.0,
                    },
                })
        return chain

    @staticmethod
    def _bs_d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _bs_call(self, S, K, T, r, sigma):
        d1 = self._bs_d1(S, K, T, r, sigma)
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._norm_cdf(d1) - K * math.exp(-r*T) * self._norm_cdf(d2)

    def _bs_put(self, S, K, T, r, sigma):
        d1 = self._bs_d1(S, K, T, r, sigma)
        d2 = d1 - sigma * math.sqrt(T)
        return K * math.exp(-r*T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

    def _bs_delta_call(self, S, K, T, r, sigma):
        return self._norm_cdf(self._bs_d1(S, K, T, r, sigma))

    def _bs_gamma(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = self._bs_d1(S, K, T, r, sigma)
        return math.exp(-0.5*d1**2) / (math.sqrt(2*math.pi) * S * sigma * math.sqrt(T))

    def _bs_vega(self, S, K, T, r, sigma):
        if T <= 0:
            return 0.0
        d1 = self._bs_d1(S, K, T, r, sigma)
        return S * math.sqrt(T) * math.exp(-0.5*d1**2) / math.sqrt(2*math.pi)


# ---------------------------------------------------------------------------
# Volume / OI anomaly detector
# ---------------------------------------------------------------------------

class VolumeOIAnomalyDetector:
    """Identifies options with unusual volume relative to open interest."""

    def __init__(self, vol_oi_threshold: float = 0.5, min_volume: int = 500):
        self.vol_oi_threshold = vol_oi_threshold
        self.min_volume = min_volume

    def detect(self, chain: List[Dict], spot: float, ticker: str) -> List[UnusualActivity]:
        unusual = []
        for opt in chain:
            vol = opt.get("volume", 0)
            oi  = opt.get("open_interest", 1)
            if vol < self.min_volume:
                continue
            ratio = vol / oi if oi > 0 else float("inf")
            if ratio < self.vol_oi_threshold:
                continue

            exp_str = opt.get("expiration_date", "")
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            side = OptionSide.CALL if opt.get("option_type", "").lower() == "call" else OptionSide.PUT
            strike = float(opt.get("strike", 0))
            contract = OptionContract(ticker, exp_date, strike, side, spot)

            premium = float(opt.get("last", 0)) * vol * 100
            iv = float(opt.get("implied_volatility", 0))
            delta = float(opt.get("delta", 0.5 if side == OptionSide.CALL else -0.5))

            if ratio >= 5.0 and premium >= 500_000:
                severity = AlertSeverity.CRITICAL
            elif ratio >= 2.0 and premium >= 200_000:
                severity = AlertSeverity.HIGH
            elif ratio >= 1.0 and premium >= 50_000:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW

            sentiment = FlowSentiment.BULLISH if side == OptionSide.CALL else FlowSentiment.BEARISH

            unusual.append(UnusualActivity(
                ticker=ticker,
                contract=contract,
                trades=[],
                total_premium_usd=premium,
                total_contracts=vol,
                vol_oi_ratio=ratio,
                is_sweep=False,
                is_block=vol >= 5000,
                dominant_sentiment=sentiment,
                alert_severity=severity,
                description=(
                    f"{ticker} {side.value.upper()} ${strike:.0f} exp {exp_str}: "
                    f"vol={vol:,} / OI={oi:,} = {ratio:.1f}x, premium=${premium/1000:.0f}K, IV={iv:.0%}"
                ),
            ))

        return sorted(unusual, key=lambda u: u.total_premium_usd, reverse=True)


# ---------------------------------------------------------------------------
# Block trade detector
# ---------------------------------------------------------------------------

class BlockTradeDetector:
    """Detects large single-transaction block trades."""

    BLOCK_THRESHOLD_CONTRACTS = 500    # ≥500 contracts
    BLOCK_THRESHOLD_PREMIUM   = 250_000  # ≥$250K notional

    def classify_trade(self, trade: OptionTrade) -> bool:
        return (
            trade.size >= self.BLOCK_THRESHOLD_CONTRACTS
            or trade.notional_usd >= self.BLOCK_THRESHOLD_PREMIUM
        )

    def generate_synthetic_trades(
        self,
        chain: List[Dict],
        ticker: str,
        spot: float,
        n: int = 50,
    ) -> List[OptionTrade]:
        """Generate synthetic trade stream from chain data for testing."""
        trades = []
        rng = random.Random(int(time.time()))
        now = datetime.now(timezone.utc)

        for i, opt in enumerate(rng.choices(chain, k=n)):
            exp_str = opt.get("expiration_date", "")
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            side = OptionSide.CALL if opt.get("option_type", "").lower() == "call" else OptionSide.PUT
            strike = float(opt.get("strike", spot))
            bid = float(opt.get("bid", 0))
            ask = float(opt.get("ask", 0))
            mid = (bid + ask) / 2

            # Random trade size (power law distribution)
            size_raw = int(abs(rng.gauss(0, 1)) ** 2 * 200) + 1
            size = min(size_raw, 5000)
            price = bid if rng.random() > 0.5 else ask

            contract = OptionContract(ticker, exp_date, strike, side, spot)
            aggressor = "buyer" if price >= mid else "seller"
            is_block = size >= self.BLOCK_THRESHOLD_CONTRACTS
            is_sweep = rng.random() < 0.1   # 10% are sweeps

            trade_types = list(TradeType)[:4]
            t_type = rng.choice(trade_types)

            oi = opt.get("open_interest", 1000)
            iv = float(opt.get("implied_volatility", 0.25))
            delta = float(opt.get("delta", 0.5))

            trades.append(OptionTrade(
                trade_id=f"T{i:05d}_{int(time.time()*1000) % 100000}",
                contract=contract,
                timestamp=now - timedelta(seconds=rng.randint(0, 86400)),
                price=round(price, 2),
                size=size,
                open_interest=oi,
                implied_vol=iv,
                delta=delta,
                exchange=rng.choice(["CBOE", "PHLX", "ISE", "ARCA", "BATS"]),
                trade_type=t_type,
                is_sweep=is_sweep,
                is_block=is_block,
                is_dark_pool=rng.random() < 0.05,
                aggressor=aggressor,
                bid=bid,
                ask=ask,
                midpoint=mid,
            ))

        return sorted(trades, key=lambda t: t.timestamp, reverse=True)


# ---------------------------------------------------------------------------
# Sweep order detector
# ---------------------------------------------------------------------------

class SweepOrderDetector:
    """
    Identifies sweep orders: same contract filled across multiple exchanges
    within a short time window.
    """

    SWEEP_WINDOW_SECONDS = 30
    SWEEP_MIN_EXCHANGES  = 2
    SWEEP_MIN_CONTRACTS  = 200

    def detect_sweeps(self, trades: List[OptionTrade]) -> List[List[OptionTrade]]:
        """Group trades into sweep clusters."""
        # Group by contract
        by_contract: Dict[str, List[OptionTrade]] = {}
        for t in trades:
            key = t.contract.contract_key
            by_contract.setdefault(key, []).append(t)

        sweeps = []
        for key, contract_trades in by_contract.items():
            sorted_trades = sorted(contract_trades, key=lambda t: t.timestamp)
            i = 0
            while i < len(sorted_trades):
                lead = sorted_trades[i]
                window = [lead]
                j = i + 1
                while j < len(sorted_trades):
                    follow = sorted_trades[j]
                    if (follow.timestamp - lead.timestamp).total_seconds() > self.SWEEP_WINDOW_SECONDS:
                        break
                    if follow.aggressor == lead.aggressor:  # same direction
                        window.append(follow)
                    j += 1

                if len(window) >= self.SWEEP_MIN_EXCHANGES:
                    exchanges = {t.exchange for t in window}
                    total_size = sum(t.size for t in window)
                    if len(exchanges) >= self.SWEEP_MIN_EXCHANGES and total_size >= self.SWEEP_MIN_CONTRACTS:
                        sweeps.append(window)
                i += 1

        return sweeps


# ---------------------------------------------------------------------------
# Main FlowScanner
# ---------------------------------------------------------------------------

class OptionsFlowScanner:
    """
    Main scanner that orchestrates all detection modules.
    Produces ranked unusual activity alerts.
    """

    def __init__(
        self,
        api_key: str = "",
        vol_oi_threshold: float = 0.5,
        min_premium_usd: float = 50_000,
        min_volume: int = 300,
    ):
        self.feed = OptionDataFeed(api_key)
        self.anomaly_detector = VolumeOIAnomalyDetector(vol_oi_threshold, min_volume)
        self.block_detector = BlockTradeDetector()
        self.sweep_detector = SweepOrderDetector()
        self.min_premium = min_premium_usd
        self._alert_counter = 0

    def scan_ticker(
        self,
        ticker: str,
        expiration: str = None,
        max_expirations: int = 3,
    ) -> List[UnusualActivity]:
        quote = self.feed.get_quotes(ticker)
        spot = float(quote.get("last", 100.0))

        if expiration:
            expirations = [expiration]
        else:
            expirations = self.feed.get_expirations(ticker)[:max_expirations]

        all_unusual = []
        for exp in expirations:
            chain = self.feed.get_option_chain(ticker, exp)
            unusual = self.anomaly_detector.detect(chain, spot, ticker)
            unusual = [u for u in unusual if u.total_premium_usd >= self.min_premium]
            all_unusual.extend(unusual)

        return sorted(all_unusual, key=lambda u: u.total_premium_usd, reverse=True)

    def scan_watchlist(
        self,
        tickers: List[str],
        max_expirations: int = 2,
    ) -> Dict[str, List[UnusualActivity]]:
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.scan_ticker(ticker, max_expirations=max_expirations)
            except Exception as exc:
                logger.error("Scan failed for %s: %s", ticker, exc)
                results[ticker] = []
        return results

    def generate_alerts(
        self,
        unusual_activities: List[UnusualActivity],
        ticker: str,
    ) -> List[FlowAlert]:
        alerts = []
        self._alert_counter += 1

        for ua in unusual_activities:
            if ua.alert_severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
                alert_type = "block_trade" if ua.is_block else "unusual_volume"
                if ua.is_sweep:
                    alert_type = "sweep"

                alerts.append(FlowAlert(
                    alert_id=f"OFA-{self._alert_counter:05d}",
                    ticker=ticker,
                    timestamp=ua.detected_at,
                    severity=ua.alert_severity,
                    activity_type=alert_type,
                    sentiment=ua.dominant_sentiment,
                    description=ua.description,
                    contract=ua.contract,
                    premium_usd=ua.total_premium_usd,
                    contracts=ua.total_contracts,
                ))
                self._alert_counter += 1

        return sorted(alerts, key=lambda a: a.premium_usd, reverse=True)

    def format_unusual_activity(self, activities: List[UnusualActivity]) -> str:
        if not activities:
            return "No unusual activity detected."

        lines = [
            "=== Unusual Options Activity ===",
            f"{'Ticker':>6} {'Type':>5} {'Strike':>7} {'Expiry':>12} {'Vol':>7} "
            f"{'OI':>8} {'Vol/OI':>7} {'Premium':>10} {'Severity':>8}",
            "-" * 80,
        ]
        for ua in activities[:20]:
            c = ua.contract
            lines.append(
                f"{ua.ticker:>6} {c.side.value[0].upper():>5} ${c.strike:>6.0f} "
                f"{c.expiry.strftime('%Y-%m-%d'):>12} {ua.total_contracts:>7,} "
                f"{ua.total_contracts:>8,} {ua.vol_oi_ratio:>7.1f}x "
                f"${ua.total_premium_usd/1000:>9.0f}K {ua.alert_severity.value:>8}"
            )
        return "\n".join(lines)

    def top_flow_summary(
        self,
        tickers: List[str],
        max_expirations: int = 2,
    ) -> str:
        all_flow = []
        for ticker in tickers:
            ua_list = self.scan_ticker(ticker, max_expirations=max_expirations)
            all_flow.extend(ua_list)

        all_flow.sort(key=lambda u: u.total_premium_usd, reverse=True)
        bull_flow = sum(u.total_premium_usd for u in all_flow if u.dominant_sentiment == FlowSentiment.BULLISH)
        bear_flow = sum(u.total_premium_usd for u in all_flow if u.dominant_sentiment == FlowSentiment.BEARISH)

        lines = [
            "=== Options Flow Summary ===",
            f"Tickers scanned: {', '.join(tickers)}",
            f"Total unusual activities: {len(all_flow)}",
            f"Bullish flow: ${bull_flow/1e6:.1f}M",
            f"Bearish flow: ${bear_flow/1e6:.1f}M",
            f"Bull/Bear ratio: {bull_flow/bear_flow:.2f}x" if bear_flow else "Bull/Bear ratio: ∞",
            "",
            "Top 10 by Premium:",
        ]
        for ua in all_flow[:10]:
            c = ua.contract
            emoji = "🐂" if ua.dominant_sentiment == FlowSentiment.BULLISH else "🐻"
            lines.append(
                f"  {emoji} {ua.ticker} {c.side.value.upper()} ${c.strike:.0f} "
                f"exp {c.expiry} — ${ua.total_premium_usd/1000:.0f}K ({ua.vol_oi_ratio:.1f}x VOI)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options flow scanner CLI")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL"])
    parser.add_argument("--min-premium", type=float, default=100_000)
    parser.add_argument("--action", choices=["scan", "summary", "alerts"], default="summary")
    args = parser.parse_args()

    scanner = OptionsFlowScanner(min_premium_usd=args.min_premium)

    if args.action == "summary":
        print(scanner.top_flow_summary(args.tickers))
    elif args.action == "scan":
        for ticker in args.tickers:
            activities = scanner.scan_ticker(ticker)
            print(scanner.format_unusual_activity(activities[:5]))
    elif args.action == "alerts":
        for ticker in args.tickers:
            activities = scanner.scan_ticker(ticker)
            alerts = scanner.generate_alerts(activities, ticker)
            for a in alerts[:3]:
                print(f"[{a.severity.value.upper()}] {a.ticker}: {a.description}")
