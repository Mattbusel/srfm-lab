"""
positioning.py — Options-implied positioning analytics.

Covers:
  - 25-delta risk reversal for directional bias
  - Butterfly spread for tail risk premium
  - Variance swap replication (model-free IV)
  - OI-weighted delta exposure
  - Gamma/vega exposure by strike
  - Positioning snapshot vs historical
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from flow_scanner import OptionDataFeed, OptionSide

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _n_cdf(x):
    return 0.5*(1+math.erf(x/math.sqrt(2)))

def _n_pdf(x):
    return math.exp(-0.5*x**2)/math.sqrt(2*math.pi)

def _d1(S,K,T,r,s):
    if T<=0 or s<=0: return 0.
    return (math.log(S/K)+(r+.5*s**2)*T)/(s*math.sqrt(T))

def _bs_call(S,K,T,r,s):
    d1=_d1(S,K,T,r,s); d2=d1-s*math.sqrt(T)
    return S*_n_cdf(d1)-K*math.exp(-r*T)*_n_cdf(d2)

def _bs_put(S,K,T,r,s):
    d1=_d1(S,K,T,r,s); d2=d1-s*math.sqrt(T)
    return K*math.exp(-r*T)*_n_cdf(-d2)-S*_n_cdf(-d1)

def _bs_vega(S,K,T,r,s):
    if T<=0: return 0.
    d1=_d1(S,K,T,r,s)
    return S*math.sqrt(T)*_n_pdf(d1)

def _bs_gamma(S,K,T,r,s):
    if T<=0 or s<=0 or S<=0: return 0.
    d1=_d1(S,K,T,r,s)
    return _n_pdf(d1)/(S*s*math.sqrt(T))

def _bs_delta_call(S,K,T,r,s):
    return _n_cdf(_d1(S,K,T,r,s))

def _bs_delta_put(S,K,T,r,s):
    return _n_cdf(_d1(S,K,T,r,s))-1.


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DirectionalBias:
    ticker: str
    expiry: date
    spot: float
    rr_25d: float             # risk reversal (call IV - put IV)
    rr_10d: float             # 10-delta RR for tails
    net_delta_oi: float       # OI-weighted net delta ($)
    net_delta_z: float        # z-score vs history
    directional_tilt: str     # "long_bias", "short_bias", "neutral"
    confidence: float


@dataclass
class TailRiskPremium:
    ticker: str
    expiry: date
    spot: float
    butterfly_25d: float      # (c25d_iv + p25d_iv)/2 - atm_iv
    left_tail_iv: float       # 10-delta put IV
    right_tail_iv: float      # 10-delta call IV
    skew_asymmetry: float     # left_tail - right_tail (positive = put skew dominant)
    tail_premium_annualized: float
    fear_gauge: float         # 0-1


@dataclass
class VarianceSwapReplication:
    """
    Model-free implied variance (CBOE VIX-style).
    VarSwap rate = (2/T) * sum_i [C(K_i)/K_i^2 + P(K_i)/K_i^2] * dK - (F/K_0 - 1)^2
    """
    ticker: str
    expiry: date
    T: float                # years to expiry
    var_swap_rate: float    # annualized variance
    vol_swap_rate: float    # sqrt(var_swap) ≈ fair strike for vol swap
    atm_iv: float
    convexity_premium: float  # vol_swap - atm_iv (should be positive)
    n_strikes_used: int


@dataclass
class PositioningSnapshot:
    ticker: str
    spot: float
    expiry: date
    timestamp: datetime
    directional: DirectionalBias
    tail_risk: TailRiskPremium
    var_swap: VarianceSwapReplication
    oi_call_total: int
    oi_put_total: int
    pc_ratio: float           # put/call OI ratio
    pc_ratio_z: float
    max_pain: float
    net_gamma_usd: float      # aggregate dealer gamma exposure
    net_vega_usd: float       # aggregate dealer vega exposure


# ---------------------------------------------------------------------------
# Directional bias estimator
# ---------------------------------------------------------------------------

class DirectionalBiasEstimator:
    """Estimates net directional positioning from OI-weighted deltas."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
        self._rr_history: Dict[str, List[float]] = {}

    def estimate(
        self,
        ticker: str,
        spot: float,
        chain: List[Dict],
        expiry: date,
    ) -> DirectionalBias:
        T = max((expiry - date.today()).days, 1) / 365.0

        # OI-weighted delta
        net_delta_oi = 0.0
        total_oi = 0

        # Build IV maps
        calls_iv: Dict[float, float] = {}
        puts_iv:  Dict[float, float] = {}

        for opt in chain:
            strike = float(opt.get("strike", 0))
            ot = opt.get("option_type","").lower()
            oi = int(opt.get("open_interest", 0))
            iv = max(float(opt.get("implied_volatility", 0.25)), 0.01)

            if ot == "call":
                delta = _bs_delta_call(spot, strike, T, self.r, iv)
                calls_iv[strike] = iv
            else:
                delta = _bs_delta_put(spot, strike, T, self.r, iv)
                puts_iv[strike] = iv

            # Dealer is short options → negative delta for dealer = positive for market
            net_delta_oi += delta * oi * 100 * spot  # dollar delta
            total_oi += oi

        # 25-delta RR
        atm_iv = self._atm_iv(calls_iv, puts_iv, spot)
        rr_25d = self._rr_at_delta(spot, T, atm_iv, calls_iv, puts_iv, 0.25)
        rr_10d = self._rr_at_delta(spot, T, atm_iv, calls_iv, puts_iv, 0.10)

        hist = self._rr_history.setdefault(ticker, [])
        hist.append(rr_25d)
        if len(hist) > 60:
            hist.pop(0)

        arr = np.array(hist)
        z = float((rr_25d - np.mean(arr)) / np.std(arr)) if len(arr) > 5 and np.std(arr) > 0 else 0.0

        if net_delta_oi > spot * total_oi * 0.1:
            tilt, conf = "long_bias", min(1., abs(z)/2)
        elif net_delta_oi < -spot * total_oi * 0.1:
            tilt, conf = "short_bias", min(1., abs(z)/2)
        else:
            tilt, conf = "neutral", 0.3

        return DirectionalBias(
            ticker=ticker, expiry=expiry, spot=spot,
            rr_25d=rr_25d, rr_10d=rr_10d,
            net_delta_oi=net_delta_oi,
            net_delta_z=z,
            directional_tilt=tilt,
            confidence=conf,
        )

    @staticmethod
    def _atm_iv(calls, puts, spot):
        combined = dict(calls)
        combined.update(puts)
        if not combined:
            return 0.25
        k = min(combined, key=lambda k: abs(k - spot))
        return combined[k]

    def _rr_at_delta(self, spot, T, atm_iv, calls_iv, puts_iv, target_delta) -> float:
        k_call = self._find_strike_for_delta(spot, T, atm_iv, target_delta, "call", calls_iv)
        k_put  = self._find_strike_for_delta(spot, T, atm_iv, target_delta, "put", puts_iv)
        iv_call = self._interp(calls_iv, k_call, atm_iv)
        iv_put  = self._interp(puts_iv, k_put, atm_iv)
        return iv_call - iv_put

    def _find_strike_for_delta(self, spot, T, sigma, delta, side, iv_map):
        from flow_signals import delta_to_strike
        try:
            return delta_to_strike(spot, T, self.r, sigma, delta, side)
        except Exception:
            return spot * (1.05 if side == "call" else 0.95)

    @staticmethod
    def _interp(iv_map, target, default):
        if not iv_map:
            return default
        strikes = sorted(iv_map.keys())
        if target <= strikes[0]:
            return iv_map[strikes[0]]
        if target >= strikes[-1]:
            return iv_map[strikes[-1]]
        for i in range(len(strikes)-1):
            if strikes[i] <= target <= strikes[i+1]:
                f = (target - strikes[i]) / (strikes[i+1] - strikes[i])
                return iv_map[strikes[i]]*(1-f) + iv_map[strikes[i+1]]*f
        return default


# ---------------------------------------------------------------------------
# Tail risk premium estimator
# ---------------------------------------------------------------------------

class TailRiskPremiumEstimator:
    """Measures how much the market pays for tail risk protection."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def estimate(
        self,
        ticker: str,
        spot: float,
        chain: List[Dict],
        expiry: date,
    ) -> TailRiskPremium:
        T = max((expiry - date.today()).days, 1) / 365.0

        calls_iv: Dict[float, float] = {}
        puts_iv:  Dict[float, float] = {}

        for opt in chain:
            k = float(opt.get("strike", 0))
            ot = opt.get("option_type","").lower()
            iv = max(float(opt.get("implied_volatility", 0.25)), 0.01)
            if ot == "call":
                calls_iv[k] = iv
            else:
                puts_iv[k] = iv

        atm_iv = DirectionalBiasEstimator._atm_iv(calls_iv, puts_iv, spot)

        # 25-delta levels
        from flow_signals import delta_to_strike
        k_c25 = delta_to_strike(spot, T, self.r, atm_iv, 0.25, "call")
        k_p25 = delta_to_strike(spot, T, self.r, atm_iv, 0.25, "put")
        k_c10 = delta_to_strike(spot, T, self.r, atm_iv, 0.10, "call")
        k_p10 = delta_to_strike(spot, T, self.r, atm_iv, 0.10, "put")

        iv_c25 = DirectionalBiasEstimator._interp(calls_iv, k_c25, atm_iv * 1.05)
        iv_p25 = DirectionalBiasEstimator._interp(puts_iv, k_p25, atm_iv * 1.10)
        iv_c10 = DirectionalBiasEstimator._interp(calls_iv, k_c10, atm_iv * 1.10)
        iv_p10 = DirectionalBiasEstimator._interp(puts_iv, k_p10, atm_iv * 1.20)

        butterfly_25d = (iv_c25 + iv_p25) / 2 - atm_iv
        skew_asym = iv_p10 - iv_c10   # positive = put tail heavier

        # Tail premium: annualized excess vol from wings
        tail_premium = butterfly_25d * math.sqrt(252)

        fear_gauge = min(1.0, (skew_asym / 0.10) * 0.5 + (butterfly_25d / 0.05) * 0.5)

        return TailRiskPremium(
            ticker=ticker, expiry=expiry, spot=spot,
            butterfly_25d=butterfly_25d,
            left_tail_iv=iv_p10, right_tail_iv=iv_c10,
            skew_asymmetry=skew_asym,
            tail_premium_annualized=tail_premium,
            fear_gauge=max(0.0, fear_gauge),
        )


# ---------------------------------------------------------------------------
# Variance swap replication
# ---------------------------------------------------------------------------

class VarianceSwapReplicator:
    """
    Model-free implied variance using the CBOE VIX methodology.
    VIX^2 = (2/T) * Σ [P(K)/K^2 * ΔK] for K < F
                  + (2/T) * Σ [C(K)/K^2 * ΔK] for K >= F
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def compute(
        self,
        ticker: str,
        spot: float,
        chain: List[Dict],
        expiry: date,
    ) -> VarianceSwapReplication:
        T = max((expiry - date.today()).days, 1) / 365.0
        F = spot * math.exp(self.r * T)   # forward price

        # Collect OTM options sorted by strike
        otm: List[Tuple[float, float, str]] = []  # (strike, price, type)

        for opt in chain:
            k = float(opt.get("strike", 0))
            ot = opt.get("option_type","").lower()
            bid = float(opt.get("bid", 0))
            ask = float(opt.get("ask", 0))
            mid = (bid + ask) / 2

            if mid <= 0:
                continue

            if ot == "put" and k < F:
                otm.append((k, mid, "put"))
            elif ot == "call" and k >= F:
                otm.append((k, mid, "call"))

        otm.sort(key=lambda x: x[0])

        if len(otm) < 3:
            atm_iv = 0.25
            return VarianceSwapReplication(
                ticker=ticker, expiry=expiry, T=T,
                var_swap_rate=atm_iv**2, vol_swap_rate=atm_iv,
                atm_iv=atm_iv, convexity_premium=0.0, n_strikes_used=0,
            )

        # Numerical integration: trapezoidal
        var_sum = 0.0
        for i, (k, price, ot) in enumerate(otm):
            if i == 0:
                dk = otm[1][0] - k
            elif i == len(otm) - 1:
                dk = k - otm[i-1][0]
            else:
                dk = (otm[i+1][0] - otm[i-1][0]) / 2

            var_sum += price / (k**2) * dk

        var_swap_rate = (2 / T) * math.exp(self.r * T) * var_sum
        var_swap_rate = max(0.0001, var_swap_rate)
        vol_swap_rate = math.sqrt(var_swap_rate)

        # ATM IV
        atm_idx = min(range(len(otm)), key=lambda i: abs(otm[i][0] - spot))
        atm_price = otm[atm_idx][1]
        atm_strike = otm[atm_idx][0]
        atm_iv = self._implied_vol(spot, atm_strike, T, atm_price, otm[atm_idx][2])

        convexity_premium = vol_swap_rate - atm_iv

        return VarianceSwapReplication(
            ticker=ticker, expiry=expiry, T=T,
            var_swap_rate=var_swap_rate,
            vol_swap_rate=vol_swap_rate,
            atm_iv=atm_iv,
            convexity_premium=convexity_premium,
            n_strikes_used=len(otm),
        )

    def _implied_vol(self, S, K, T, price, option_type, tol=1e-6, max_iter=100):
        """Newton-Raphson IV solver."""
        sigma = 0.25
        for _ in range(max_iter):
            if option_type == "call":
                model_price = _bs_call(S, K, T, self.r, sigma)
            else:
                model_price = _bs_put(S, K, T, self.r, sigma)
            vega = _bs_vega(S, K, T, self.r, sigma)
            if abs(vega) < 1e-10:
                break
            diff = model_price - price
            sigma -= diff / vega
            sigma = max(0.001, min(10.0, sigma))
            if abs(diff) < tol:
                break
        return sigma


# ---------------------------------------------------------------------------
# Positioning snapshot builder
# ---------------------------------------------------------------------------

class PositioningSnapshotBuilder:
    """Assembles a full positioning snapshot."""

    def __init__(self):
        self.feed = OptionDataFeed()
        self.directional_est = DirectionalBiasEstimator()
        self.tail_risk_est = TailRiskPremiumEstimator()
        self.var_swap = VarianceSwapReplicator()
        self._pc_history: Dict[str, List[float]] = {}

    def build(self, ticker: str, expiry_idx: int = 0) -> PositioningSnapshot:
        quote = self.feed.get_quotes(ticker)
        spot = float(quote.get("last", 100.0))
        exps = self.feed.get_expirations(ticker)

        if not exps:
            raise RuntimeError(f"No expirations available for {ticker}")

        exp_str = exps[min(expiry_idx, len(exps)-1)]
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        chain = self.feed.get_option_chain(ticker, exp_str)

        directional = self.directional_est.estimate(ticker, spot, chain, exp_date)
        tail_risk    = self.tail_risk_est.estimate(ticker, spot, chain, exp_date)
        var_swap     = self.var_swap.compute(ticker, spot, chain, exp_date)

        # Put/call OI ratio
        call_oi = sum(int(o.get("open_interest", 0)) for o in chain if o.get("option_type","").lower() == "call")
        put_oi  = sum(int(o.get("open_interest", 0)) for o in chain if o.get("option_type","").lower() == "put")
        pc_ratio = put_oi / call_oi if call_oi > 0 else 1.0

        hist = self._pc_history.setdefault(ticker, [])
        hist.append(pc_ratio)
        if len(hist) > 30:
            hist.pop(0)
        arr = np.array(hist)
        pc_z = float((pc_ratio - np.mean(arr)) / np.std(arr)) if len(arr) > 5 and np.std(arr) > 0 else 0.0

        # Max pain
        by_strike: Dict[float, Dict] = {}
        for o in chain:
            k = float(o.get("strike", 0))
            ot = o.get("option_type","").lower()
            oi = int(o.get("open_interest", 0))
            rec = by_strike.setdefault(k, {"c":0,"p":0})
            if ot == "call":
                rec["c"] += oi
            else:
                rec["p"] += oi

        T = max((exp_date - date.today()).days, 1) / 365.0

        max_pain = spot
        min_pain = float("inf")
        for test_k in sorted(by_strike):
            pain = sum(
                max(0, test_k - k) * d["c"] + max(0, k - test_k) * d["p"]
                for k, d in by_strike.items()
            )
            if pain < min_pain:
                min_pain = pain
                max_pain = test_k

        # Aggregate net gamma and vega
        net_gamma_usd = 0.0
        net_vega_usd = 0.0
        for o in chain:
            k = float(o.get("strike", 0))
            ot = o.get("option_type","").lower()
            oi = int(o.get("open_interest", 0))
            iv = max(float(o.get("implied_volatility", 0.25)), 0.01)
            g = _bs_gamma(spot, k, T, 0.05, iv) * oi * 100 * spot**2 * 0.01
            v = _bs_vega(spot, k, T, 0.05, iv) * oi / 100
            sign = 1 if ot == "call" else -1
            net_gamma_usd += g * sign
            net_vega_usd  += v * sign

        return PositioningSnapshot(
            ticker=ticker, spot=spot, expiry=exp_date,
            timestamp=datetime.now(timezone.utc),
            directional=directional,
            tail_risk=tail_risk,
            var_swap=var_swap,
            oi_call_total=call_oi,
            oi_put_total=put_oi,
            pc_ratio=pc_ratio,
            pc_ratio_z=pc_z,
            max_pain=max_pain,
            net_gamma_usd=net_gamma_usd,
            net_vega_usd=net_vega_usd,
        )

    def format_snapshot(self, snap: PositioningSnapshot) -> str:
        d = snap.directional
        tr = snap.tail_risk
        vs = snap.var_swap

        lines = [
            f"=== Positioning Snapshot: {snap.ticker} ===",
            f"Spot: ${snap.spot:,.2f} | Expiry: {snap.expiry} | {snap.timestamp.strftime('%Y-%m-%d %H:%M')} UTC",
            "",
            "Directional Bias:",
            f"  Tilt:          {d.directional_tilt.replace('_',' ').upper()}",
            f"  RR 25d:        {d.rr_25d:+.1%}",
            f"  RR 10d:        {d.rr_10d:+.1%}",
            f"  Net Delta OI:  ${d.net_delta_oi/1e6:+.1f}M",
            f"  Delta Z-score: {d.net_delta_z:+.2f}",
            "",
            "Tail Risk Premium:",
            f"  Butterfly 25d: {tr.butterfly_25d:.1%}",
            f"  Skew Asymmetry:{tr.skew_asymmetry:+.1%} (positive = put skew)",
            f"  Left Tail IV:  {tr.left_tail_iv:.1%}  |  Right Tail IV: {tr.right_tail_iv:.1%}",
            f"  Fear Gauge:    {tr.fear_gauge:.0%}",
            "",
            "Variance Swap (VIX-style):",
            f"  Var Swap Rate: {vs.var_swap_rate:.4f} ({math.sqrt(vs.var_swap_rate)*100:.1f}% vol)",
            f"  ATM IV:        {vs.atm_iv:.1%}",
            f"  Convexity:     {vs.convexity_premium:+.1%} (var swap - ATM)",
            f"  Strikes Used:  {vs.n_strikes_used}",
            "",
            "Put/Call:",
            f"  P/C OI Ratio:  {snap.pc_ratio:.2f}x (z={snap.pc_ratio_z:+.2f})",
            f"  Max Pain:      ${snap.max_pain:,.0f}",
            "",
            f"Net Gamma USD:  ${snap.net_gamma_usd/1e6:+.1f}M",
            f"Net Vega USD:   ${snap.net_vega_usd/1e3:+.1f}K",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options positioning CLI")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--expiry-idx", type=int, default=0)
    args = parser.parse_args()

    builder = PositioningSnapshotBuilder()
    snap = builder.build(args.ticker, args.expiry_idx)
    print(builder.format_snapshot(snap))
