"""
flow_signals.py — Smart money options flow signals.

Covers:
  - Call/put skew changes (directional sentiment shifts)
  - IV term structure steepening/flattening
  - Risk reversal signals (25-delta RR)
  - Box spread detection (carry trade arbitrage)
  - Smart money footprint scoring
  - Composite flow signal for each ticker
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from flow_scanner import OptionDataFeed, FlowSentiment, OptionSide

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-Scholes utilities (local to avoid circular imports)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))

def bs_delta_call(S, K, T, r, sigma):
    return _norm_cdf(_bs_d1(S, K, T, r, sigma))

def delta_to_strike(spot, T, r, sigma, target_delta, option_type="call") -> float:
    """Solve for the strike corresponding to a target delta."""
    # Newton-Raphson iteration
    K = spot  # initial guess
    for _ in range(50):
        d1 = _bs_d1(spot, K, T, r, sigma)
        if option_type == "call":
            delta = _norm_cdf(d1)
        else:
            delta = _norm_cdf(d1) - 1.0
        # dDelta/dK
        from math import exp, pi, sqrt
        pdf_d1 = exp(-0.5 * d1**2) / sqrt(2*pi)
        ddelta_dK = -pdf_d1 / (K * sigma * sqrt(T)) if (K > 0 and T > 0 and sigma > 0) else 0.0
        if abs(ddelta_dK) < 1e-12:
            break
        K = K - (delta - target_delta) / ddelta_dK
        K = max(K, 0.01)
    return K


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SkewMetrics:
    ticker: str
    expiry: date
    spot_price: float
    iv_atm: float
    iv_25d_call: float    # 25-delta call IV
    iv_25d_put: float     # 25-delta put IV
    risk_reversal_25d: float   # call IV - put IV (positive = call premium)
    butterfly_25d: float       # (call + put) / 2 - atm (wing premium)
    put_call_iv_ratio: float   # put/call IV ratio (>1 = put skew)
    skew_slope: float          # d(IV)/d(strike) around ATM
    skew_change_1d: float      # 1-day change in risk reversal
    skew_change_5d: float      # 5-day change

    @property
    def sentiment(self) -> str:
        if self.risk_reversal_25d > 0.02:
            return "bullish"
        if self.risk_reversal_25d < -0.02:
            return "bearish"
        return "neutral"


@dataclass
class TermStructure:
    ticker: str
    spot_price: float
    maturities: List[Tuple[date, float]]  # (expiry, ATM IV)
    contango_slope: float     # dIV/dT (positive = contango = normal)
    backwardation: bool
    front_back_spread: float  # front_month_IV - back_month_IV
    steepness: float          # normalized slope
    vix_proxy: float          # 30-day interpolated ATM IV

    @property
    def term_structure_signal(self) -> str:
        if self.backwardation:
            return "backwardation_stressed"
        if self.contango_slope < 0.01:
            return "flat_cautious"
        return "contango_normal"


@dataclass
class RiskReversal:
    ticker: str
    expiry: date
    spot: float
    delta_target: float       # e.g., 0.25
    strike_call: float        # 25-delta call strike
    strike_put: float         # 25-delta put strike
    iv_call: float
    iv_put: float
    rr_value: float           # iv_call - iv_put
    rr_percentile: float      # vs historical
    signal: str               # "bullish", "bearish", "neutral"
    confidence: float


@dataclass
class BoxSpread:
    """
    A box spread = bull call spread + bear put spread at same strikes.
    Should equal PV of the strike difference (risk-free).
    Mispricing = carry trade opportunity.
    """
    ticker: str
    expiry: date
    low_strike: float
    high_strike: float
    call_spread_price: float   # long K1 call + short K2 call
    put_spread_price: float    # long K2 put + short K1 put
    box_price: float           # call_spread + put_spread (should = K2-K1 discounted)
    fair_value: float          # (K2 - K1) * exp(-r*T)
    mispricing_usd: float      # box_price - fair_value
    mispricing_bps: float      # in bps of fair value
    is_arbitrage: bool         # |mispricing| > transaction costs
    implied_rate: float        # carry rate implied by box price


@dataclass
class SmartMoneySignal:
    ticker: str
    timestamp: datetime
    composite_score: float        # -1 (bearish) to +1 (bullish)
    direction: str
    confidence: float
    components: Dict[str, float]   # sub-signal contributions
    alert_message: str


# ---------------------------------------------------------------------------
# Skew analyzer
# ---------------------------------------------------------------------------

class SkewAnalyzer:
    """Analyzes option IV skew for directional sentiment."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = r = risk_free_rate
        self._skew_history: Dict[str, List[float]] = {}

    def compute(
        self,
        ticker: str,
        spot: float,
        chain: List[Dict],
        expiry: date,
    ) -> SkewMetrics:
        T = max((expiry - date.today()).days, 1) / 365.0

        # Separate calls and puts, get strikes and IVs
        calls = {float(o["strike"]): float(o.get("implied_volatility", 0.25))
                 for o in chain if o.get("option_type", "").lower() == "call"
                 and float(o.get("implied_volatility", 0)) > 0}
        puts  = {float(o["strike"]): float(o.get("implied_volatility", 0.25))
                 for o in chain if o.get("option_type", "").lower() == "put"
                 and float(o.get("implied_volatility", 0)) > 0}

        if not calls or not puts:
            return self._empty_skew(ticker, expiry, spot)

        # ATM IV: interpolate to spot
        atm_call_iv = self._interp_iv(calls, spot)
        atm_put_iv  = self._interp_iv(puts, spot)
        iv_atm = (atm_call_iv + atm_put_iv) / 2

        # 25-delta strike IVs
        k_call_25d = delta_to_strike(spot, T, self.r, iv_atm, 0.25, "call")
        k_put_25d  = delta_to_strike(spot, T, self.r, iv_atm, 0.25, "put")

        iv_call_25d = self._interp_iv(calls, k_call_25d)
        iv_put_25d  = self._interp_iv(puts, k_put_25d)

        rr = iv_call_25d - iv_put_25d
        bf = (iv_call_25d + iv_put_25d) / 2 - iv_atm

        put_call_ratio = iv_put_25d / iv_call_25d if iv_call_25d > 0 else 1.0

        # Skew slope: linear fit of IV vs normalized strike
        all_data = list(calls.items()) + list(puts.items())
        if len(all_data) >= 3:
            xs = [(k / spot - 1) for k, _ in all_data]
            ys = [iv for _, iv in all_data]
            try:
                slope = float(np.polyfit(xs, ys, 1)[0])
            except Exception:
                slope = 0.0
        else:
            slope = 0.0

        # Historical skew change
        history_key = f"{ticker}_{expiry}"
        hist = self._skew_history.setdefault(history_key, [])
        hist.append(rr)
        if len(hist) > 30:
            hist.pop(0)

        change_1d = (hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0
        change_5d = (hist[-1] - hist[-6]) if len(hist) >= 6 else 0.0

        return SkewMetrics(
            ticker=ticker,
            expiry=expiry,
            spot_price=spot,
            iv_atm=iv_atm,
            iv_25d_call=iv_call_25d,
            iv_25d_put=iv_put_25d,
            risk_reversal_25d=rr,
            butterfly_25d=bf,
            put_call_iv_ratio=put_call_ratio,
            skew_slope=slope,
            skew_change_1d=change_1d,
            skew_change_5d=change_5d,
        )

    @staticmethod
    def _interp_iv(iv_by_strike: Dict[float, float], target: float) -> float:
        if not iv_by_strike:
            return 0.25
        strikes = sorted(iv_by_strike.keys())
        if target <= strikes[0]:
            return iv_by_strike[strikes[0]]
        if target >= strikes[-1]:
            return iv_by_strike[strikes[-1]]
        for i in range(len(strikes) - 1):
            if strikes[i] <= target <= strikes[i+1]:
                frac = (target - strikes[i]) / (strikes[i+1] - strikes[i])
                return iv_by_strike[strikes[i]] * (1 - frac) + iv_by_strike[strikes[i+1]] * frac
        return iv_by_strike[strikes[len(strikes)//2]]

    def _empty_skew(self, ticker, expiry, spot) -> SkewMetrics:
        return SkewMetrics(ticker, expiry, spot, 0.25, 0.275, 0.30, -0.025, 0.025, 1.09, -0.05, 0.0, 0.0)

    def skew_signal(self, skew: SkewMetrics) -> str:
        """Generate signal from skew change."""
        if skew.skew_change_5d > 0.02:
            return "call_skew_rising: bullish rotation"
        if skew.skew_change_5d < -0.02:
            return "put_skew_rising: bearish rotation"
        if skew.risk_reversal_25d > 0.03:
            return "elevated_call_premium: bullish bias"
        if skew.risk_reversal_25d < -0.05:
            return "elevated_put_premium: fear / bearish"
        return "neutral_skew"


# ---------------------------------------------------------------------------
# Term structure analyzer
# ---------------------------------------------------------------------------

class TermStructureAnalyzer:
    """Analyzes IV term structure across expirations."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
        self.feed = OptionDataFeed()

    def build(self, ticker: str, spot: float, expirations: List[str]) -> TermStructure:
        maturities: List[Tuple[date, float]] = []

        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                chain = self.feed.get_option_chain(ticker, exp_str)
                atm_iv = self._get_atm_iv(chain, spot)
                if atm_iv > 0:
                    maturities.append((exp_date, atm_iv))
            except Exception as exc:
                logger.debug("Failed to get IV for %s %s: %s", ticker, exp_str, exc)

        maturities.sort(key=lambda x: x[0])

        if len(maturities) < 2:
            return TermStructure(
                ticker=ticker, spot_price=spot, maturities=maturities,
                contango_slope=0.0, backwardation=False, front_back_spread=0.0,
                steepness=0.0, vix_proxy=0.25,
            )

        # Linear slope of IV vs time
        xs = [(m - maturities[0][0]).days / 365.0 for m, _ in maturities]
        ys = [iv for _, iv in maturities]

        if len(xs) >= 2 and max(xs) > 0:
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = 0.0

        front_iv = maturities[0][1]
        back_iv  = maturities[-1][1]
        fb_spread = front_iv - back_iv

        vix_proxy = self._interpolate_30d_iv(maturities)
        steepness = slope / front_iv if front_iv > 0 else 0.0

        return TermStructure(
            ticker=ticker,
            spot_price=spot,
            maturities=maturities,
            contango_slope=slope,
            backwardation=slope < 0,
            front_back_spread=fb_spread,
            steepness=steepness,
            vix_proxy=vix_proxy,
        )

    @staticmethod
    def _get_atm_iv(chain: List[Dict], spot: float) -> float:
        if not chain:
            return 0.0
        atm_opt = min(chain, key=lambda o: abs(float(o.get("strike", 0)) - spot))
        return float(atm_opt.get("implied_volatility", 0))

    @staticmethod
    def _interpolate_30d_iv(maturities: List[Tuple[date, float]]) -> float:
        today = date.today()
        target_days = 30
        for i in range(len(maturities) - 1):
            d1, iv1 = maturities[i]
            d2, iv2 = maturities[i+1]
            t1 = (d1 - today).days
            t2 = (d2 - today).days
            if t1 <= target_days <= t2 and t2 > t1:
                frac = (target_days - t1) / (t2 - t1)
                return iv1 * (1 - frac) + iv2 * frac
        return maturities[0][1] if maturities else 0.25


# ---------------------------------------------------------------------------
# Risk reversal signal builder
# ---------------------------------------------------------------------------

class RiskReversalSignalBuilder:
    """Builds and interprets 25-delta risk reversal signals."""

    def __init__(self, history_window: int = 30):
        self.history_window = history_window
        self._history: Dict[str, List[float]] = {}

    def build(
        self,
        ticker: str,
        skew: "SkewMetrics",
        expiry: date,
        spot: float,
    ) -> RiskReversal:
        iv_atm = skew.iv_atm
        T = max((expiry - date.today()).days, 1) / 365.0
        r = 0.05

        k_call = delta_to_strike(spot, T, r, iv_atm, 0.25, "call")
        k_put  = delta_to_strike(spot, T, r, iv_atm, 0.25, "put")

        rr = skew.risk_reversal_25d

        hist = self._history.setdefault(ticker, [])
        hist.append(rr)
        if len(hist) > self.history_window:
            hist.pop(0)

        arr = np.array(hist)
        if len(arr) >= 5:
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            z = (rr - mean) / std if std > 0 else 0.0
            pct = float(np.searchsorted(np.sort(arr), rr)) / len(arr) * 100
        else:
            z, pct = 0.0, 50.0

        if z > 1.5:
            signal, conf = "bullish", min(1.0, abs(z) / 3)
        elif z < -1.5:
            signal, conf = "bearish", min(1.0, abs(z) / 3)
        else:
            signal, conf = "neutral", 0.3

        return RiskReversal(
            ticker=ticker,
            expiry=expiry,
            spot=spot,
            delta_target=0.25,
            strike_call=k_call,
            strike_put=k_put,
            iv_call=skew.iv_25d_call,
            iv_put=skew.iv_25d_put,
            rr_value=rr,
            rr_percentile=pct,
            signal=signal,
            confidence=conf,
        )


# ---------------------------------------------------------------------------
# Box spread detector
# ---------------------------------------------------------------------------

class BoxSpreadDetector:
    """
    Identifies box spreads from option chains.
    Box spread price should equal PV of strike differential.
    Deviations imply either mispricing (rare) or implied carry rates.
    """

    TRANSACTION_COST_BPS = 20   # 20bps transaction cost estimate

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def scan_chain(self, chain: List[Dict], expiry: date, spot: float) -> List[BoxSpread]:
        """Find all box spreads in a chain and flag mispricings."""
        ticker = chain[0].get("underlying", "?") if chain else "?"
        T = max((expiry - date.today()).days, 1) / 365.0

        # Get all available strikes
        call_map: Dict[float, Dict] = {}
        put_map:  Dict[float, Dict] = {}

        for opt in chain:
            try:
                exp_str = opt.get("expiration_date", "")
                exp_d = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_d != expiry:
                    continue
            except ValueError:
                pass

            strike = float(opt.get("strike", 0))
            ot = opt.get("option_type", "").lower()
            if ot == "call":
                call_map[strike] = opt
            else:
                put_map[strike] = opt

        common_strikes = sorted(set(call_map) & set(put_map))
        boxes = []

        for i in range(len(common_strikes) - 1):
            K1 = common_strikes[i]
            K2 = common_strikes[i + 1]

            # Prices (use midpoint)
            c1 = (float(call_map[K1].get("bid", 0)) + float(call_map[K1].get("ask", 0))) / 2
            c2 = (float(call_map[K2].get("bid", 0)) + float(call_map[K2].get("ask", 0))) / 2
            p1 = (float(put_map[K1].get("bid", 0)) + float(put_map[K1].get("ask", 0))) / 2
            p2 = (float(put_map[K2].get("bid", 0)) + float(put_map[K2].get("ask", 0))) / 2

            call_spread = c1 - c2   # long K1 call, short K2 call
            put_spread  = p2 - p1   # long K2 put, short K1 put
            box_price   = call_spread + put_spread

            fair_value = (K2 - K1) * math.exp(-self.r * T)
            mispricing = box_price - fair_value
            misprice_bps = mispricing / fair_value * 10000 if fair_value > 0 else 0.0

            implied_rate = -math.log(box_price / (K2 - K1)) / T if box_price > 0 and T > 0 else 0.0

            is_arb = abs(misprice_bps) > self.TRANSACTION_COST_BPS

            boxes.append(BoxSpread(
                ticker=ticker,
                expiry=expiry,
                low_strike=K1,
                high_strike=K2,
                call_spread_price=call_spread,
                put_spread_price=put_spread,
                box_price=box_price,
                fair_value=fair_value,
                mispricing_usd=mispricing,
                mispricing_bps=misprice_bps,
                is_arbitrage=is_arb,
                implied_rate=implied_rate,
            ))

        return [b for b in boxes if abs(b.mispricing_bps) > 0]


# ---------------------------------------------------------------------------
# Smart money flow scorer
# ---------------------------------------------------------------------------

class SmartMoneyFlowScorer:
    """
    Builds composite smart money signal from multiple flow metrics.
    """

    def __init__(self, feed: OptionDataFeed = None):
        self.feed = feed or OptionDataFeed()
        self.skew_analyzer = SkewAnalyzer()
        self.ts_analyzer = TermStructureAnalyzer()
        self.rr_builder = RiskReversalSignalBuilder()
        self.box_detector = BoxSpreadDetector()

    def compute_signal(self, ticker: str) -> SmartMoneySignal:
        quote = self.feed.get_quotes(ticker)
        spot = float(quote.get("last", 100.0))
        expirations = self.feed.get_expirations(ticker)[:4]

        if not expirations:
            return SmartMoneySignal(
                ticker=ticker, timestamp=datetime.now(timezone.utc),
                composite_score=0.0, direction="neutral", confidence=0.0,
                components={}, alert_message="No expiration data",
            )

        # Use nearest expiration for skew/RR
        front_exp = expirations[0]
        front_chain = self.feed.get_option_chain(ticker, front_exp)
        front_exp_date = datetime.strptime(front_exp, "%Y-%m-%d").date()

        # Skew signal
        skew = self.skew_analyzer.compute(ticker, spot, front_chain, front_exp_date)
        skew_sig = skew.skew_signal(skew)

        # RR signal
        rr = self.rr_builder.build(ticker, skew, front_exp_date, spot)

        # Term structure
        ts = self.ts_analyzer.build(ticker, spot, expirations)

        # Scoring
        components: Dict[str, float] = {}

        # Skew component: negative put skew = bearish
        rr_score = math.tanh(skew.risk_reversal_25d / 0.03)
        components["risk_reversal"] = rr_score

        # Skew change 5d: rising call skew = bullish
        skew_change_score = math.tanh(skew.skew_change_5d / 0.02)
        components["skew_change_5d"] = skew_change_score

        # Term structure: backwardation = fear = bearish
        ts_score = -1.0 if ts.backwardation else (0.3 if ts.contango_slope > 0.02 else 0.0)
        components["term_structure"] = ts_score

        # Butterfly: elevated wings = tail fear = bearish
        bf_score = -math.tanh(skew.butterfly_25d / 0.02)
        components["butterfly"] = bf_score

        weights = {"risk_reversal": 0.40, "skew_change_5d": 0.25, "term_structure": 0.20, "butterfly": 0.15}
        composite = sum(components[k] * weights[k] for k in components)

        if composite > 0.35:
            direction, conf = "strong_bullish", min(1.0, abs(composite))
        elif composite > 0.10:
            direction, conf = "bullish", min(1.0, abs(composite))
        elif composite < -0.35:
            direction, conf = "strong_bearish", min(1.0, abs(composite))
        elif composite < -0.10:
            direction, conf = "bearish", min(1.0, abs(composite))
        else:
            direction, conf = "neutral", 0.3

        alert = (
            f"{ticker}: {direction.upper()} | RR={skew.risk_reversal_25d:+.1%} "
            f"| 5d chg={skew.skew_change_5d:+.1%} | TS={'BACK' if ts.backwardation else 'CONTANGO'}"
        )

        return SmartMoneySignal(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            composite_score=round(composite, 4),
            direction=direction,
            confidence=round(conf, 3),
            components=components,
            alert_message=alert,
        )


# ---------------------------------------------------------------------------
# Multi-ticker flow signal scanner
# ---------------------------------------------------------------------------

class FlowSignalScanner:
    """Scans a watchlist and ranks by composite smart money signal."""

    def __init__(self):
        self.scorer = SmartMoneyFlowScorer()

    def scan(self, tickers: List[str]) -> List[SmartMoneySignal]:
        results = []
        for ticker in tickers:
            try:
                sig = self.scorer.compute_signal(ticker)
                results.append(sig)
            except Exception as exc:
                logger.warning("Flow signal failed for %s: %s", ticker, exc)
        return sorted(results, key=lambda s: s.composite_score, reverse=True)

    def format_table(self, signals: List[SmartMoneySignal]) -> str:
        lines = [
            "=== Smart Money Flow Signals ===",
            f"{'Ticker':>8} {'Score':>7} {'Direction':>14} {'Conf':>6} {'RR':>8} {'TS Chg':>8}",
            "-" * 60,
        ]
        for s in signals:
            rr = s.components.get("risk_reversal", 0)
            ts = s.components.get("term_structure", 0)
            lines.append(
                f"{s.ticker:>8} {s.composite_score:>+7.3f} {s.direction:>14} "
                f"{s.confidence:>6.0%} {rr:>+8.3f} {ts:>+8.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options flow signals CLI")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "TSLA"])
    parser.add_argument("--action", choices=["scan", "skew", "ts"], default="scan")
    args = parser.parse_args()

    scanner = FlowSignalScanner()

    if args.action == "scan":
        signals = scanner.scan(args.tickers)
        print(scanner.format_table(signals))
        print()
        for s in signals:
            print(s.alert_message)
    elif args.action == "skew":
        feed = OptionDataFeed()
        skew_analyzer = SkewAnalyzer()
        for ticker in args.tickers:
            q = feed.get_quotes(ticker)
            spot = float(q.get("last", 100))
            exps = feed.get_expirations(ticker)
            if exps:
                chain = feed.get_option_chain(ticker, exps[0])
                exp_date = datetime.strptime(exps[0], "%Y-%m-%d").date()
                skew = skew_analyzer.compute(ticker, spot, chain, exp_date)
                print(f"{ticker}: ATM={skew.iv_atm:.1%} RR={skew.risk_reversal_25d:+.1%} BF={skew.butterfly_25d:.1%} → {skew.sentiment}")
    elif args.action == "ts":
        feed = OptionDataFeed()
        ts_analyzer = TermStructureAnalyzer()
        for ticker in args.tickers:
            q = feed.get_quotes(ticker)
            spot = float(q.get("last", 100))
            exps = feed.get_expirations(ticker)[:5]
            ts = ts_analyzer.build(ticker, spot, exps)
            print(f"{ticker}: VIX-proxy={ts.vix_proxy:.1%} slope={ts.contango_slope:.3f} "
                  f"{'BACKWARDATION' if ts.backwardation else 'contango'}")
