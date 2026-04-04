"""
vol_regime.py — Volatility regime classifier.

Covers:
  - VIX term structure analysis (contango/backwardation)
  - VVIX (VIX of VIX) — vol-of-vol signal
  - Realized vs implied vol spread (vol risk premium)
  - Vol risk premium (VRP) over rolling windows
  - Regime classification: low/rising/stressed/collapsing vol
  - Vol mean reversion signal
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from flow_scanner import OptionDataFeed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VIX data fetcher
# ---------------------------------------------------------------------------

class VIXDataFetcher:
    """
    Fetches VIX, VIX3M, VIX6M, and VVIX from CBOE / Yahoo Finance.
    Falls back to synthetic data if live feeds unavailable.
    """

    CBOE_BASE = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical"
    YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0"})
        self._cache: Dict[str, Tuple[float, List]] = {}   # {ticker: (ts, data)}
        self._cache_ttl = 300  # 5 minutes

    def get_quote(self, symbol: str) -> float:
        """Get current VIX-family level."""
        cached = self._cache.get(symbol)
        if cached and time.time() - cached[0] < self._cache_ttl:
            return cached[1][-1][1] if cached[1] else 20.0

        try:
            resp = self._session.get(
                f"{self.YAHOO_BASE}/{symbol}",
                params={"interval": "1d", "range": "5d"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            valid = [c for c in closes if c is not None]
            return valid[-1] if valid else self._synthetic_level(symbol)
        except Exception:
            return self._synthetic_level(symbol)

    def get_history(self, symbol: str, days: int = 252) -> List[Tuple[datetime, float]]:
        """Historical daily closes for VIX/VVIX etc."""
        try:
            resp = self._session.get(
                f"{self.YAHOO_BASE}/{symbol}",
                params={"interval": "1d", "range": f"{days}d"},
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()["chart"]["result"][0]
            timestamps = result["timestamp"]
            closes = result["indicators"]["quote"][0]["close"]
            series = [
                (datetime.fromtimestamp(ts, tz=timezone.utc), c)
                for ts, c in zip(timestamps, closes) if c is not None
            ]
            return series
        except Exception:
            return self._synthetic_history(symbol, days)

    @staticmethod
    def _synthetic_level(symbol: str) -> float:
        defaults = {"^VIX": 18.5, "^VIX3M": 19.2, "^VIX6M": 20.1, "^VVIX": 85.0}
        return defaults.get(symbol, 20.0)

    def _synthetic_history(self, symbol: str, days: int) -> List[Tuple[datetime, float]]:
        """Generate realistic synthetic VIX history."""
        level = self._synthetic_level(symbol)
        np.random.seed(42)
        vols = [level]
        for _ in range(days - 1):
            # Mean-reverting process
            speed = 5.0
            vol_of_vol = 0.8
            dt = 1/252
            dW = np.random.normal(0, math.sqrt(dt))
            dV = speed * (level - vols[-1]) * dt + vol_of_vol * vols[-1] * dW
            vols.append(max(8.0, vols[-1] + dV))

        start = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            (start + timedelta(days=i), v)
            for i, v in enumerate(vols)
        ]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VIXTermStructure:
    vix_1m: float
    vix_3m: float
    vix_6m: float
    vix_9m: float       # approximated
    contango_slope: float   # (vix_6m - vix_1m) / 5  normalized
    is_backwardation: bool
    m1_m3_spread: float     # vix_1m - vix_3m (negative = normal contango)
    m1_m6_spread: float
    structure_signal: str   # "steep_contango", "flat", "backwardation", "inverted"

    @property
    def is_stressed(self) -> bool:
        return self.is_backwardation and self.vix_1m > 25


@dataclass
class VVIXAnalysis:
    current_vvix: float
    mean_vvix: float
    vvix_z: float
    regime: str         # "calm", "elevated", "stressed"
    vol_of_vol_premium: float
    signal: str


@dataclass
class VolRiskPremium:
    lookback_days: int
    realized_vol: float      # 21-day historical vol
    implied_vol: float       # ATM IV (30-day VIX proxy)
    vrp: float               # implied - realized (positive = normal)
    vrp_z: float             # vs history
    vrp_percentile: float
    signal: str              # "attractive_short_vol", "fair", "cheap_long_vol"
    rolling_vrp: List[Tuple[datetime, float]] = field(default_factory=list)


@dataclass
class VolRegimeState:
    timestamp: datetime
    regime: str              # "low_stable", "rising", "stressed", "collapsing", "transition"
    vix_level: float
    vix_ts: VIXTermStructure
    vvix: VVIXAnalysis
    vrp: VolRiskPremium
    regime_duration_days: int
    transition_probability: float  # prob of regime change in next 5 days
    recommended_strategy: str


# ---------------------------------------------------------------------------
# VIX term structure analyzer
# ---------------------------------------------------------------------------

class VIXTermStructureAnalyzer:
    """Analyzes the VIX futures term structure for vol regime signals."""

    def __init__(self):
        self.fetcher = VIXDataFetcher()

    def compute(self) -> VIXTermStructure:
        vix_1m = self.fetcher.get_quote("^VIX")
        vix_3m = self.fetcher.get_quote("^VIX3M")
        vix_6m = self.fetcher.get_quote("^VIX6M")

        # VIX9M not always available; estimate from slope
        slope_3_6 = (vix_6m - vix_3m) / 3   # per month
        vix_9m = vix_6m + slope_3_6 * 3

        m1_m3 = vix_1m - vix_3m
        m1_m6 = vix_1m - vix_6m
        contango_slope = (vix_6m - vix_1m) / 5   # per month

        is_backwardation = vix_1m > vix_3m

        if contango_slope > 1.5:
            structure_signal = "steep_contango"
        elif contango_slope > 0.3:
            structure_signal = "normal_contango"
        elif contango_slope > -0.5:
            structure_signal = "flat"
        elif is_backwardation and vix_1m > 25:
            structure_signal = "inverted_stressed"
        else:
            structure_signal = "backwardation"

        return VIXTermStructure(
            vix_1m=vix_1m, vix_3m=vix_3m, vix_6m=vix_6m, vix_9m=vix_9m,
            contango_slope=contango_slope,
            is_backwardation=is_backwardation,
            m1_m3_spread=m1_m3,
            m1_m6_spread=m1_m6,
            structure_signal=structure_signal,
        )

    def historical_structure(self, days: int = 90) -> List[Dict]:
        vix1m_hist = self.fetcher.get_history("^VIX", days)
        vix3m_hist = self.fetcher.get_history("^VIX3M", days)

        # Align by date
        v1_map = {dt.date(): v for dt, v in vix1m_hist}
        v3_map = {dt.date(): v for dt, v in vix3m_hist}
        common = sorted(set(v1_map) & set(v3_map))

        return [
            {
                "date": d.isoformat(),
                "vix_1m": v1_map[d],
                "vix_3m": v3_map[d],
                "m1_m3_spread": v1_map[d] - v3_map[d],
                "is_backwardation": v1_map[d] > v3_map[d],
            }
            for d in common
        ]


# ---------------------------------------------------------------------------
# VVIX analyzer
# ---------------------------------------------------------------------------

class VVIXAnalyzer:
    """Analyzes VVIX (volatility of VIX) for second-order vol signals."""

    VVIX_HIGH_THRESHOLD = 110
    VVIX_LOW_THRESHOLD  = 75

    def __init__(self):
        self.fetcher = VIXDataFetcher()

    def compute(self, lookback_days: int = 90) -> VVIXAnalysis:
        current = self.fetcher.get_quote("^VVIX")
        history = self.fetcher.get_history("^VVIX", lookback_days)
        hist_vals = [v for _, v in history]

        mean_vvix = float(np.mean(hist_vals)) if hist_vals else 85.0
        std_vvix  = float(np.std(hist_vals))  if hist_vals else 10.0
        z = (current - mean_vvix) / std_vvix if std_vvix > 0 else 0.0

        if current >= self.VVIX_HIGH_THRESHOLD:
            regime = "stressed"
            signal = "extreme vol-of-vol: expect VIX spike risk"
        elif current >= 95:
            regime = "elevated"
            signal = "elevated VVIX: tail risk pricing active"
        else:
            regime = "calm"
            signal = "low VVIX: stable vol environment"

        vol_of_vol_premium = max(0.0, current - mean_vvix)

        return VVIXAnalysis(
            current_vvix=current,
            mean_vvix=mean_vvix,
            vvix_z=z,
            regime=regime,
            vol_of_vol_premium=vol_of_vol_premium,
            signal=signal,
        )


# ---------------------------------------------------------------------------
# Realized vs implied vol (VRP)
# ---------------------------------------------------------------------------

class VolRiskPremiumCalculator:
    """
    Computes the volatility risk premium: implied vol - realized vol.
    Positive VRP means options are priced rich → sell vol.
    Negative VRP (rare) → buy vol.
    """

    def __init__(self):
        self.feed = OptionDataFeed()
        self.fetcher = VIXDataFetcher()
        self._vrp_history: List[float] = []

    def compute(
        self,
        ticker: str = "SPY",
        realized_window: int = 21,
    ) -> VolRiskPremium:
        # Realized vol: from historical price returns
        rv = self._realized_vol(ticker, realized_window)

        # Implied vol: current ATM IV (use VIX as proxy for SPY)
        if ticker.upper() in ("SPY", "^SPX", "SPX"):
            iv = self.fetcher.get_quote("^VIX") / 100.0
        else:
            iv = self._atm_iv_from_chain(ticker)

        vrp = iv - rv

        self._vrp_history.append(vrp)
        if len(self._vrp_history) > 252:
            self._vrp_history.pop(0)

        arr = np.array(self._vrp_history)
        z = float((vrp - np.mean(arr)) / np.std(arr)) if len(arr) > 10 and np.std(arr) > 0 else 0.0
        pct = float(np.searchsorted(np.sort(arr), vrp)) / len(arr) * 100 if arr.size > 0 else 50.0

        if vrp > 0.05 and z > 1.0:
            signal = "attractive_short_vol"
        elif vrp < 0.01:
            signal = "cheap_long_vol"
        else:
            signal = "fair_vol"

        return VolRiskPremium(
            lookback_days=realized_window,
            realized_vol=rv,
            implied_vol=iv,
            vrp=vrp,
            vrp_z=z,
            vrp_percentile=pct,
            signal=signal,
        )

    def _realized_vol(self, ticker: str, window: int) -> float:
        """Compute realized vol from Yahoo Finance price history."""
        try:
            resp = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"interval": "1d", "range": f"{window+5}d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            resp.raise_for_status()
            closes = resp.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if len(closes) < 2:
                return 0.18
            returns = [math.log(closes[i+1]/closes[i]) for i in range(len(closes)-1)]
            return float(np.std(returns[-window:])) * math.sqrt(252)
        except Exception:
            return 0.18   # fallback: 18% realized vol

    def _atm_iv_from_chain(self, ticker: str) -> float:
        try:
            q = self.feed.get_quotes(ticker)
            spot = float(q.get("last", 100))
            exps = self.feed.get_expirations(ticker)
            if not exps:
                return 0.25
            chain = self.feed.get_option_chain(ticker, exps[0])
            if not chain:
                return 0.25
            atm = min(chain, key=lambda o: abs(float(o.get("strike",0)) - spot))
            return max(float(atm.get("implied_volatility", 0.25)), 0.01)
        except Exception:
            return 0.25

    def historical_vrp(self, ticker: str = "SPY", days: int = 90) -> List[Tuple[datetime, float]]:
        """Compute rolling daily VRP."""
        vix_hist = self.fetcher.get_history("^VIX", days + 30)
        vix_map = {dt.date(): v / 100.0 for dt, v in vix_hist}

        try:
            resp = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"interval": "1d", "range": f"{days+30}d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()["chart"]["result"][0]
            ts = result["timestamp"]
            closes = result["indicators"]["quote"][0]["close"]
            price_data = [(datetime.fromtimestamp(t, tz=timezone.utc).date(), c)
                         for t, c in zip(ts, closes) if c is not None]
        except Exception:
            return []

        rv_series = []
        window = 21
        for i in range(window, len(price_data)):
            segment = [price_data[j][1] for j in range(i-window, i+1) if price_data[j][1]]
            if len(segment) < 2:
                continue
            rets = [math.log(segment[j+1]/segment[j]) for j in range(len(segment)-1)]
            rv = float(np.std(rets)) * math.sqrt(252)
            d = price_data[i][0]
            iv = vix_map.get(d, None)
            if iv:
                vrp = iv - rv
                rv_series.append((datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc), vrp))

        return rv_series


# ---------------------------------------------------------------------------
# Vol regime classifier
# ---------------------------------------------------------------------------

class VolRegimeClassifier:
    """Classifies current vol regime and recommends strategy."""

    REGIMES = {
        "low_stable":    "VIX < 15, contango, low VVIX",
        "rising":        "VIX 15-25, flattening TS, elevated VVIX",
        "stressed":      "VIX > 25, backwardation, high VVIX",
        "collapsing":    "VIX falling from peak, steep contango recovering",
        "transition":    "Mixed signals",
    }

    def classify(
        self,
        ts: VIXTermStructure,
        vvix: VVIXAnalysis,
        vrp: VolRiskPremium,
    ) -> VolRegimeState:
        vix = ts.vix_1m

        # Classify regime
        if vix < 14 and not ts.is_backwardation and vvix.regime == "calm":
            regime = "low_stable"
            strategy = "Sell premium: short strangles, covered calls. VRP attractive."
            transition_prob = 0.10

        elif vix > 30 and ts.is_backwardation and vvix.regime == "stressed":
            regime = "stressed"
            strategy = "Buy protection or go long vol. Avoid short vol. Trade vol mean reversion."
            transition_prob = 0.25

        elif vix > 20 and ts.is_backwardation:
            regime = "rising"
            strategy = "Reduce short vol exposure. Consider long vol / put spreads for hedging."
            transition_prob = 0.30

        elif vix < 18 and ts.contango_slope > 1.0 and vrp.vrp > 0.04:
            regime = "collapsing"
            strategy = "Sell near-term vol into the collapse. VRP highest in contango recovery."
            transition_prob = 0.15

        else:
            regime = "transition"
            strategy = "Neutral. Reduce position size. Monitor for regime clarification."
            transition_prob = 0.35

        return VolRegimeState(
            timestamp=datetime.now(timezone.utc),
            regime=regime,
            vix_level=vix,
            vix_ts=ts,
            vvix=vvix,
            vrp=vrp,
            regime_duration_days=0,   # would need historical tracking
            transition_probability=transition_prob,
            recommended_strategy=strategy,
        )


# ---------------------------------------------------------------------------
# Main VolRegimeAnalytics facade
# ---------------------------------------------------------------------------

class VolRegimeAnalytics:
    """Unified volatility regime analytics."""

    def __init__(self):
        self.ts_analyzer = VIXTermStructureAnalyzer()
        self.vvix_analyzer = VVIXAnalyzer()
        self.vrp_calc = VolRiskPremiumCalculator()
        self.classifier = VolRegimeClassifier()

    def current_regime(self) -> VolRegimeState:
        ts   = self.ts_analyzer.compute()
        vvix = self.vvix_analyzer.compute()
        vrp  = self.vrp_calc.compute()
        return self.classifier.classify(ts, vvix, vrp)

    def format_report(self) -> str:
        state = self.current_regime()
        ts   = state.vix_ts
        vvix = state.vvix
        vrp  = state.vrp

        lines = [
            "=== Volatility Regime Report ===",
            f"Timestamp: {state.timestamp.strftime('%Y-%m-%d %H:%M')} UTC",
            f"REGIME: {state.regime.replace('_',' ').upper()}",
            f"Transition probability (5d): {state.transition_probability:.0%}",
            "",
            "VIX Term Structure:",
            f"  VIX 1M:   {ts.vix_1m:.2f}",
            f"  VIX 3M:   {ts.vix_3m:.2f}",
            f"  VIX 6M:   {ts.vix_6m:.2f}",
            f"  M1-M3:    {ts.m1_m3_spread:+.2f}",
            f"  Slope:    {ts.contango_slope:+.2f}/month",
            f"  Structure:{ts.structure_signal.replace('_',' ').upper()}",
            "",
            "VVIX (Vol of VIX):",
            f"  Current:  {vvix.current_vvix:.1f}",
            f"  Mean:     {vvix.mean_vvix:.1f}",
            f"  Z-score:  {vvix.vvix_z:+.2f}",
            f"  Regime:   {vvix.regime.upper()}",
            f"  Signal:   {vvix.signal}",
            "",
            "Vol Risk Premium:",
            f"  Realized (21d): {vrp.realized_vol:.1%}",
            f"  Implied (VIX):  {vrp.implied_vol:.1%}",
            f"  VRP:            {vrp.vrp:+.1%}",
            f"  VRP Z-score:    {vrp.vrp_z:+.2f}",
            f"  Signal:         {vrp.signal.replace('_',' ').upper()}",
            "",
            "Strategy Recommendation:",
            f"  {state.recommended_strategy}",
        ]
        return "\n".join(lines)

    def vol_regime_signal(self) -> Dict:
        state = self.current_regime()
        return {
            "regime": state.regime,
            "vix": state.vix_level,
            "vix_structure": state.vix_ts.structure_signal,
            "vvix_z": state.vvix.vvix_z,
            "vrp": state.vrp.vrp,
            "vrp_signal": state.vrp.signal,
            "transition_prob": state.transition_probability,
            "strategy": state.recommended_strategy,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vol regime CLI")
    parser.add_argument("--action", choices=["report", "signal", "ts"], default="report")
    args = parser.parse_args()

    import json as _json
    analytics = VolRegimeAnalytics()

    if args.action == "report":
        print(analytics.format_report())
    elif args.action == "signal":
        sig = analytics.vol_regime_signal()
        print(_json.dumps(sig, indent=2))
    elif args.action == "ts":
        ts_data = analytics.ts_analyzer.historical_structure(60)
        backwardation_days = sum(1 for d in ts_data if d["is_backwardation"])
        print(f"Last 60 sessions: {backwardation_days} in backwardation ({backwardation_days/len(ts_data):.0%})")
        for row in ts_data[-10:]:
            flag = "⚠️ BACK" if row["is_backwardation"] else "     "
            print(f"  {row['date']}: VIX={row['vix_1m']:.1f} VIX3M={row['vix_3m']:.1f} spread={row['m1_m3_spread']:+.2f} {flag}")
