"""
gamma_exposure.py — Dealer gamma exposure (GEX) analytics.

Covers:
  - Net gamma by strike (call GEX - put GEX)
  - Gamma flip level (price where dealer gamma changes sign)
  - Spot-vol correlation from GEX regime
  - Pin risk near expiry (maximum gamma zones)
  - GEX surface over strikes and expirations
  - Delta hedging flow estimation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-Scholes Greeks library
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def bs_delta(S, K, T, r, sigma, option_type="call") -> float:
    d1 = _bs_d1(S, K, T, r, sigma)
    if option_type == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S, K, T, r, sigma) -> float:
    if T <= 0 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return S * math.sqrt(T) * _norm_pdf(d1)


def bs_theta(S, K, T, r, sigma, option_type="call") -> float:
    if T <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    term1 = -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type == "call":
        term2 = -r * K * math.exp(-r * T) * _norm_cdf(d2)
        return (term1 + term2) / 365
    else:
        term2 = r * K * math.exp(-r * T) * _norm_cdf(-d2)
        return (term1 + term2) / 365


def bs_charm(S, K, T, r, sigma, option_type="call") -> float:
    """Delta decay (dDelta/dT)."""
    if T <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    charm_base = -_norm_pdf(d1) * (2*r*T - d2*sigma*math.sqrt(T)) / (2*T*sigma*math.sqrt(T))
    return charm_base


def bs_speed(S, K, T, r, sigma) -> float:
    """Third derivative of option price w.r.t. spot (dGamma/dS)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    gamma = bs_gamma(S, K, T, r, sigma)
    return -gamma / S * (d1 / (sigma * math.sqrt(T)) + 1)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StrikeGEX:
    strike: float
    call_gamma: float         # positive: long call gamma
    put_gamma: float          # negative convention: dealers short puts
    call_oi: int
    put_oi: int
    net_gamma: float          # call_gex - put_gex (dealer perspective)
    call_gex_usd: float       # dollar gamma: gamma * OI * spot^2 * 0.01
    put_gex_usd: float
    net_gex_usd: float
    distance_from_spot_pct: float

    @property
    def is_net_long(self) -> bool:
        """Dealer is net long gamma at this strike."""
        return self.net_gex_usd > 0

    @property
    def pin_magnetic(self) -> float:
        """Pin magnetism score: large absolute GEX near expiry attracts price."""
        return abs(self.net_gex_usd)


@dataclass
class GEXSurface:
    ticker: str
    spot_price: float
    timestamp: datetime
    strikes: List[StrikeGEX]
    total_call_gex: float
    total_put_gex: float
    total_net_gex: float
    gamma_flip_level: Optional[float]    # price where net GEX changes sign
    max_gamma_strike: float              # strike with highest absolute GEX
    pin_risk_strikes: List[float]        # strikes with high pin potential
    gex_regime: str                      # "long_gamma" or "short_gamma"

    @property
    def dealer_is_long_gamma(self) -> bool:
        return self.total_net_gex > 0

    @property
    def implied_vol_bias(self) -> str:
        """When dealers are long gamma, they sell vol → lower realized vol."""
        if self.dealer_is_long_gamma:
            return "dampened_vol"   # dealers hedge → pin effect → low vol
        return "amplified_vol"       # dealers chase → trend extension → high vol


@dataclass
class GammaFlipAnalysis:
    current_price: float
    flip_level: Optional[float]
    distance_to_flip_pct: Optional[float]
    is_above_flip: bool
    regime: str
    implication: str
    flip_strikes_detail: List[Dict]


@dataclass
class SpotVolCorrelation:
    current_gex: float
    gex_regime: str
    expected_spot_vol_corr: float    # estimated correlation coefficient
    vol_dampening_factor: float      # how much vol is suppressed
    mean_reversion_strength: float   # 0-1
    trend_following_strength: float  # 0-1


# ---------------------------------------------------------------------------
# GEX calculator
# ---------------------------------------------------------------------------

class GEXCalculator:
    """
    Computes dealer gamma exposure from option chain data.

    Convention (market maker / dealer perspective):
    - Dealers are typically SHORT options (they sell to clients)
    - Long calls → dealers sold calls → dealer is SHORT call gamma
    - Long puts → dealers sold puts → dealer is SHORT put gamma
    - When client BOUGHT calls: dealer is -Gamma (short gamma at that strike)
    - When client BOUGHT puts: dealer is -Gamma too
    - Net long call OI > put OI at a strike → dealers net short gamma there

    Standard GEX formula:
    Call GEX = call_OI * gamma * spot^2 * 0.01 (per 1% move)
    Put GEX  = -put_OI * gamma * spot^2 * 0.01  (dealers short puts)
    Net GEX  = Call GEX + Put GEX
    """

    CONTRACTS_PER_LOT = 100   # shares per contract

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def compute_strike_gex(
        self,
        spot: float,
        strike: float,
        expiry: date,
        call_oi: int,
        put_oi: int,
        call_iv: float,
        put_iv: float,
    ) -> StrikeGEX:
        T = max((expiry - date.today()).days, 0) / 365.0
        if T == 0:
            T = 1 / 365.0   # same-day expiry: very short T

        g_call = bs_gamma(spot, strike, T, self.r, call_iv)
        g_put  = bs_gamma(spot, strike, T, self.r, put_iv)

        # Dollar gamma: gamma * OI * contracts * spot^2 * 0.01
        call_gex_usd = call_oi * g_call * self.CONTRACTS_PER_LOT * spot**2 * 0.01
        put_gex_usd  = -put_oi * g_put  * self.CONTRACTS_PER_LOT * spot**2 * 0.01

        net_gex_usd  = call_gex_usd + put_gex_usd
        dist_pct     = (strike - spot) / spot * 100

        return StrikeGEX(
            strike=strike,
            call_gamma=g_call,
            put_gamma=g_put,
            call_oi=call_oi,
            put_oi=put_oi,
            net_gamma=g_call - g_put,
            call_gex_usd=call_gex_usd,
            put_gex_usd=put_gex_usd,
            net_gex_usd=net_gex_usd,
            distance_from_spot_pct=dist_pct,
        )

    def compute_surface(
        self,
        ticker: str,
        spot: float,
        chain: List[Dict],
        expiry: date = None,
    ) -> GEXSurface:
        """Build complete GEX surface from option chain data."""
        # Group by strike
        by_strike: Dict[float, Dict] = {}
        for opt in chain:
            if expiry is not None:
                try:
                    opt_exp = datetime.strptime(opt.get("expiration_date", ""), "%Y-%m-%d").date()
                    if opt_exp != expiry:
                        continue
                except ValueError:
                    continue

            strike = float(opt.get("strike", 0))
            ot = opt.get("option_type", "").lower()
            iv = float(opt.get("implied_volatility", 0.25))
            oi = int(opt.get("open_interest", 0))

            rec = by_strike.setdefault(strike, {
                "call_oi": 0, "put_oi": 0, "call_iv": 0.25, "put_iv": 0.25,
                "expiry": opt.get("expiration_date", ""),
            })
            if ot == "call":
                rec["call_oi"] = oi
                rec["call_iv"] = max(iv, 0.01)
            else:
                rec["put_oi"] = oi
                rec["put_iv"] = max(iv, 0.01)

        strike_gex_list = []
        for strike, data in by_strike.items():
            exp_str = data["expiry"]
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                exp_date = date.today() + __import__("datetime").timedelta(days=30)

            sgex = self.compute_strike_gex(
                spot, strike, exp_date,
                data["call_oi"], data["put_oi"],
                data["call_iv"], data["put_iv"],
            )
            strike_gex_list.append(sgex)

        strike_gex_list.sort(key=lambda s: s.strike)

        total_call_gex = sum(s.call_gex_usd for s in strike_gex_list)
        total_put_gex  = sum(s.put_gex_usd  for s in strike_gex_list)
        total_net_gex  = total_call_gex + total_put_gex

        # Gamma flip level
        flip_level = self._find_gamma_flip(strike_gex_list, spot)

        # Max gamma strike
        max_strike = max(strike_gex_list, key=lambda s: abs(s.net_gex_usd), default=None)
        max_gamma_s = max_strike.strike if max_strike else spot

        # Pin risk: top strikes by absolute net GEX near expiry
        near_atm = [s for s in strike_gex_list if abs(s.distance_from_spot_pct) <= 10]
        pin_risks = [
            s.strike for s in sorted(near_atm, key=lambda s: abs(s.net_gex_usd), reverse=True)[:3]
        ]

        regime = "long_gamma" if total_net_gex > 0 else "short_gamma"

        return GEXSurface(
            ticker=ticker,
            spot_price=spot,
            timestamp=datetime.now(timezone.utc),
            strikes=strike_gex_list,
            total_call_gex=total_call_gex,
            total_put_gex=total_put_gex,
            total_net_gex=total_net_gex,
            gamma_flip_level=flip_level,
            max_gamma_strike=max_gamma_s,
            pin_risk_strikes=pin_risks,
            gex_regime=regime,
        )

    @staticmethod
    def _find_gamma_flip(
        strikes: List[StrikeGEX],
        spot: float,
    ) -> Optional[float]:
        """
        Find the price level where cumulative net GEX changes sign.
        At the flip level, dealers transition from long to short gamma.
        """
        below = sorted([s for s in strikes if s.strike <= spot], key=lambda s: s.strike, reverse=True)
        above = sorted([s for s in strikes if s.strike >  spot], key=lambda s: s.strike)

        # Accumulate GEX from spot outward
        cum_gex = 0.0
        last_sign = None
        flip = None

        for s in below:
            cum_gex += s.net_gex_usd
            sign = 1 if cum_gex > 0 else -1
            if last_sign is not None and sign != last_sign:
                flip = (s.strike + list(below)[list(below).index(s) - 1].strike) / 2 if list(below).index(s) > 0 else s.strike
                return flip
            last_sign = sign

        return None


# ---------------------------------------------------------------------------
# Gamma flip analyzer
# ---------------------------------------------------------------------------

class GammaFlipAnalyzer:
    """Analyzes the gamma flip level and its market implications."""

    def analyze(self, surface: GEXSurface) -> GammaFlipAnalysis:
        spot = surface.spot_price
        flip = surface.gamma_flip_level

        if flip is None:
            return GammaFlipAnalysis(
                current_price=spot,
                flip_level=None,
                distance_to_flip_pct=None,
                is_above_flip=True,
                regime=surface.gex_regime,
                implication="No clear flip level detected",
                flip_strikes_detail=[],
            )

        dist_pct = (spot - flip) / flip * 100
        is_above = spot > flip

        if is_above and surface.gex_regime == "long_gamma":
            regime = "positive_gamma"
            implication = (
                f"Dealers long gamma above ${flip:,.0f}. "
                "Expect mean-reversion / dampened volatility. "
                "Spot tends to be 'pinned' between gamma clusters."
            )
        elif not is_above:
            regime = "negative_gamma"
            implication = (
                f"Spot below gamma flip ${flip:,.0f}. "
                "Dealers short gamma — forced to buy dips AND sell rips, "
                "amplifying moves. Trend-following environment."
            )
        else:
            regime = "transitioning"
            implication = "Near gamma flip level — unstable regime."

        # Detail for strikes near flip
        flip_strikes = [
            {"strike": s.strike, "net_gex_usd": s.net_gex_usd}
            for s in surface.strikes
            if abs(s.strike - (flip or spot)) < spot * 0.02
        ]

        return GammaFlipAnalysis(
            current_price=spot,
            flip_level=flip,
            distance_to_flip_pct=dist_pct,
            is_above_flip=is_above,
            regime=regime,
            implication=implication,
            flip_strikes_detail=flip_strikes[:5],
        )


# ---------------------------------------------------------------------------
# Spot-vol correlation estimator
# ---------------------------------------------------------------------------

class SpotVolCorrelationEstimator:
    """
    Estimates how GEX regime affects spot-vol correlation.
    Long gamma → dealers sell rallies, buy dips → spot-vol correlation negative
    Short gamma → dealers buy rallies, sell dips → spot-vol correlation positive (crash risk)
    """

    def estimate(self, surface: GEXSurface) -> SpotVolCorrelation:
        net_gex = surface.total_net_gex
        max_gex = sum(abs(s.net_gex_usd) for s in surface.strikes)

        if max_gex == 0:
            normalized_gex = 0.0
        else:
            normalized_gex = net_gex / max_gex   # -1 to +1

        # Long gamma → spot-vol correlation negative (vol compresses as price rises)
        # Short gamma → spot-vol correlation positive (vol spikes with price moves)
        expected_corr = -normalized_gex * 0.7  # empirical scaling

        # Vol dampening: how much gamma hedging suppresses realized vol
        # Rough model: 1% vol reduction per $1B net long GEX per day
        vol_dampening = max(0.0, min(0.3, net_gex / 1e10))

        # Mean reversion strength (from dealer hedging)
        mr_strength = max(0.0, min(1.0, normalized_gex * 0.8 + 0.5))

        # Trend following (from dealer chasing in short gamma)
        tf_strength = max(0.0, min(1.0, -normalized_gex * 0.8 + 0.5))

        regime = "long_gamma" if net_gex > 0 else "short_gamma"

        return SpotVolCorrelation(
            current_gex=net_gex,
            gex_regime=regime,
            expected_spot_vol_corr=round(expected_corr, 3),
            vol_dampening_factor=round(vol_dampening, 3),
            mean_reversion_strength=round(mr_strength, 3),
            trend_following_strength=round(tf_strength, 3),
        )


# ---------------------------------------------------------------------------
# Pin risk near expiry
# ---------------------------------------------------------------------------

class PinRiskAnalyzer:
    """
    Identifies strikes where price is likely to 'pin' near expiry.
    Max pain theory: price moves to minimize aggregate option value.
    """

    def max_pain_strike(
        self,
        chain: List[Dict],
        spot: float,
        expiry: date = None,
    ) -> float:
        """Compute the max pain strike using total OI value minimization."""
        by_strike: Dict[float, Dict] = {}
        for opt in chain:
            if expiry is not None:
                try:
                    opt_exp = datetime.strptime(opt.get("expiration_date", ""), "%Y-%m-%d").date()
                    if opt_exp != expiry:
                        continue
                except ValueError:
                    pass
            strike = float(opt.get("strike", 0))
            ot = opt.get("option_type", "").lower()
            oi = int(opt.get("open_interest", 0))
            rec = by_strike.setdefault(strike, {"call_oi": 0, "put_oi": 0})
            if ot == "call":
                rec["call_oi"] += oi
            else:
                rec["put_oi"] += oi

        if not by_strike:
            return spot

        strikes = sorted(by_strike.keys())
        min_pain = float("inf")
        min_pain_strike = spot

        for test_price in strikes:
            # Total pain at test_price
            total_pain = 0.0
            for k, data in by_strike.items():
                # Call holders lose if test_price < strike
                if test_price < k:
                    total_pain += (k - test_price) * data["call_oi"]
                # Put holders lose if test_price > strike
                if test_price > k:
                    total_pain += (test_price - k) * data["put_oi"]

            if total_pain < min_pain:
                min_pain = total_pain
                min_pain_strike = test_price

        return min_pain_strike

    def pin_probability(
        self,
        spot: float,
        strike: float,
        dte: int,
        net_gex_at_strike: float,
        total_gex: float,
    ) -> float:
        """Estimate probability that spot pins at this strike by expiry."""
        distance = abs(strike - spot) / spot
        time_decay = math.exp(-dte / 5)   # stronger pin near expiry

        if total_gex == 0:
            return 0.0
        gex_share = abs(net_gex_at_strike) / abs(total_gex)

        # Pin probability: high if close, near expiry, large GEX
        prob = gex_share * time_decay * math.exp(-distance * 20)
        return min(1.0, prob)


# ---------------------------------------------------------------------------
# GEX time series tracker
# ---------------------------------------------------------------------------

class GEXTimeSeriesTracker:
    """Tracks GEX over time to identify regime changes."""

    def __init__(self):
        self._history: List[Tuple[datetime, float]] = []

    def record(self, ts: datetime, net_gex: float) -> None:
        self._history.append((ts, net_gex))

    def get_series(self) -> List[Tuple[datetime, float]]:
        return list(self._history)

    def current_regime(self) -> str:
        if not self._history:
            return "unknown"
        return "long_gamma" if self._history[-1][1] > 0 else "short_gamma"

    def regime_change_points(self) -> List[Dict]:
        changes = []
        for i in range(1, len(self._history)):
            prev_sign = self._history[i-1][1] > 0
            curr_sign = self._history[i][1] > 0
            if prev_sign != curr_sign:
                ts, gex = self._history[i]
                changes.append({
                    "timestamp": ts.isoformat(),
                    "gex": gex,
                    "new_regime": "long_gamma" if curr_sign else "short_gamma",
                })
        return changes

    def gex_momentum(self, window: int = 5) -> float:
        """Rate of change in GEX (derivative)."""
        if len(self._history) < window + 1:
            return 0.0
        vals = [gex for _, gex in self._history[-window-1:]]
        return (vals[-1] - vals[0]) / window


# ---------------------------------------------------------------------------
# Main GEX analytics facade
# ---------------------------------------------------------------------------

class GammaExposureAnalytics:
    """Unified API for all GEX analytics."""

    def __init__(self):
        from flow_scanner import OptionDataFeed
        self.feed = OptionDataFeed()
        self.calculator = GEXCalculator()
        self.flip_analyzer = GammaFlipAnalyzer()
        self.sv_estimator = SpotVolCorrelationEstimator()
        self.pin_analyzer = PinRiskAnalyzer()
        self.tracker = GEXTimeSeriesTracker()

    def build_surface(self, ticker: str) -> GEXSurface:
        quote = self.feed.get_quotes(ticker)
        spot = float(quote.get("last", 100.0))
        exps = self.feed.get_expirations(ticker)[:1]

        if not exps:
            raise RuntimeError(f"No expirations for {ticker}")

        all_chain = []
        for exp in exps[:3]:  # top 3 near-term expirations
            all_chain.extend(self.feed.get_option_chain(ticker, exp))

        surface = self.calculator.compute_surface(ticker, spot, all_chain)
        self.tracker.record(surface.timestamp, surface.total_net_gex)
        return surface

    def full_analysis(self, ticker: str) -> Dict:
        surface = self.build_surface(ticker)
        flip_analysis = self.flip_analyzer.analyze(surface)
        sv_corr = self.sv_estimator.estimate(surface)

        quote = self.feed.get_quotes(ticker)
        spot = float(quote.get("last", 100.0))
        exps = self.feed.get_expirations(ticker)
        chain = self.feed.get_option_chain(ticker, exps[0]) if exps else []
        max_pain = self.pin_analyzer.max_pain_strike(chain, spot)

        return {
            "ticker": ticker,
            "spot": spot,
            "timestamp": surface.timestamp.isoformat(),
            "gex": {
                "total_net_gex": surface.total_net_gex,
                "total_call_gex": surface.total_call_gex,
                "total_put_gex": surface.total_put_gex,
                "regime": surface.gex_regime,
                "implied_vol_bias": surface.implied_vol_bias,
                "max_gamma_strike": surface.max_gamma_strike,
                "pin_risk_strikes": surface.pin_risk_strikes,
            },
            "flip": {
                "level": flip_analysis.flip_level,
                "distance_pct": flip_analysis.distance_to_flip_pct,
                "is_above_flip": flip_analysis.is_above_flip,
                "regime": flip_analysis.regime,
                "implication": flip_analysis.implication,
            },
            "spot_vol_corr": {
                "expected_corr": sv_corr.expected_spot_vol_corr,
                "vol_dampening": sv_corr.vol_dampening_factor,
                "mean_reversion_strength": sv_corr.mean_reversion_strength,
            },
            "max_pain": max_pain,
        }

    def format_report(self, ticker: str) -> str:
        data = self.full_analysis(ticker)
        gex = data["gex"]
        flip = data["flip"]
        sv = data["spot_vol_corr"]

        lines = [
            f"=== Gamma Exposure Analysis: {ticker} ===",
            f"Spot: ${data['spot']:,.2f} | Time: {data['timestamp']}",
            "",
            f"Net GEX:    ${gex['total_net_gex']/1e9:.2f}B  ({gex['regime'].replace('_', ' ').upper()})",
            f"Call GEX:   ${gex['total_call_gex']/1e9:.2f}B",
            f"Put GEX:    ${gex['total_put_gex']/1e9:.2f}B",
            f"Vol Bias:   {gex['implied_vol_bias'].replace('_', ' ')}",
            f"Max Gamma Strike: ${gex['max_gamma_strike']:,.0f}",
            f"Pin Risk Levels:  {', '.join(f'${s:,.0f}' for s in gex['pin_risk_strikes'])}",
            "",
            "Gamma Flip:",
            f"  Flip Level:  {'${:,.0f}'.format(flip['level']) if flip['level'] else 'N/A'}",
            f"  Distance:    {'{:+.1f}%'.format(flip['distance_pct']) if flip['distance_pct'] else 'N/A'}",
            f"  Regime:      {flip['regime'].replace('_', ' ').upper()}",
            f"  Implication: {flip['implication'][:80]}",
            "",
            "Spot-Vol Correlation (estimated):",
            f"  Spot-Vol Corr:       {sv['expected_corr']:+.3f}",
            f"  Vol Dampening:       {sv['vol_dampening']:.1%}",
            f"  Mean Rev Strength:   {sv['mean_reversion_strength']:.1%}",
            "",
            f"Max Pain Strike: ${data['max_pain']:,.0f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gamma exposure CLI")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--action", choices=["report", "surface", "flip"], default="report")
    args = parser.parse_args()

    gex = GammaExposureAnalytics()

    if args.action == "report":
        print(gex.format_report(args.ticker))
    elif args.action == "surface":
        surface = gex.build_surface(args.ticker)
        print(f"GEX Surface: {len(surface.strikes)} strikes")
        print(f"Total Net GEX: ${surface.total_net_gex/1e9:.2f}B")
        print(f"Flip Level: ${surface.gamma_flip_level:,.0f}" if surface.gamma_flip_level else "No flip found")
        print("\nTop 10 strikes by |GEX|:")
        sorted_s = sorted(surface.strikes, key=lambda s: abs(s.net_gex_usd), reverse=True)
        for s in sorted_s[:10]:
            print(f"  ${s.strike:,.0f}: net_gex=${s.net_gex_usd/1e6:+.0f}M  ({s.distance_from_spot_pct:+.1f}% from spot)")
    elif args.action == "flip":
        surface = gex.build_surface(args.ticker)
        flip = gex.flip_analyzer.analyze(surface)
        print(f"Gamma Flip Level: ${flip.flip_level:,.0f}" if flip.flip_level else "No flip found")
        print(f"Distance from spot: {flip.distance_to_flip_pct:+.1f}%" if flip.distance_to_flip_pct else "N/A")
        print(f"Regime: {flip.regime}")
        print(f"Implication: {flip.implication}")
