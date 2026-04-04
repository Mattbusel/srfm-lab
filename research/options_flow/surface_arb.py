"""
surface_arb.py — Volatility surface arbitrage detection.

Covers:
  - Put-call parity violations
  - Calendar spread mispricing
  - Butterfly spread arbitrage detection
  - Vertical spread no-arbitrage bounds
  - Convexity / monotonicity violations
  - Vol surface smoothing and fitting
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from flow_scanner import OptionDataFeed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _ncd(x):
    return 0.5*(1+math.erf(x/math.sqrt(2)))

def _npd(x):
    return math.exp(-0.5*x**2)/math.sqrt(2*math.pi)

def _d1(S,K,T,r,s):
    if T<=0 or s<=0: return 0.
    return (math.log(S/K)+(r+.5*s**2)*T)/(s*math.sqrt(T))

def bs_call(S,K,T,r,s):
    d1=_d1(S,K,T,r,s); d2=d1-s*math.sqrt(T)
    return S*_ncd(d1)-K*math.exp(-r*T)*_ncd(d2)

def bs_put(S,K,T,r,s):
    d1=_d1(S,K,T,r,s); d2=d1-s*math.sqrt(T)
    return K*math.exp(-r*T)*_ncd(-d2)-S*_ncd(-d1)

def bs_vega(S,K,T,r,s):
    if T<=0: return 0.
    d1=_d1(S,K,T,r,s)
    return S*math.sqrt(T)*_npd(d1)

def implied_vol(S, K, T, r, price, option_type="call", tol=1e-6, max_iter=100) -> Optional[float]:
    """Newton-Raphson IV solver. Returns None if no solution."""
    sigma = 0.25
    for _ in range(max_iter):
        if option_type == "call":
            model = bs_call(S, K, T, r, sigma)
        else:
            model = bs_put(S, K, T, r, sigma)
        v = bs_vega(S, K, T, r, sigma)
        if abs(v) < 1e-10:
            return None
        sigma -= (model - price) / v
        sigma = max(0.001, min(20.0, sigma))
        if abs(model - price) < tol:
            return sigma
    return None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PutCallParityViolation:
    ticker: str
    strike: float
    expiry: date
    call_price: float
    put_price: float
    spot: float
    forward: float
    pcp_lhs: float      # C - P
    pcp_rhs: float      # F - K * e^(-rT) = PV(F-K)
    deviation: float    # lhs - rhs
    deviation_bps: float
    is_arbitrage: bool
    arb_direction: str  # "buy_call_sell_put" or "sell_call_buy_put"
    net_profit_per_lot: float


@dataclass
class CalendarSpreadViolation:
    ticker: str
    strike: float
    near_expiry: date
    far_expiry: date
    near_iv: float
    far_iv: float
    violation_type: str   # "backwardation" (near_iv > far_iv abnormally) or "excess_roll"
    deviation_pts: float
    is_arbitrage: bool
    expected_roll_up: float
    arb_trade: str


@dataclass
class ButterflyArb:
    ticker: str
    expiry: date
    low_strike: float
    mid_strike: float
    high_strike: float
    call_prices: Tuple[float, float, float]
    butterfly_price: float     # C(K1) - 2*C(K2) + C(K3)
    is_negative: bool          # negative butterfly = arbitrage
    free_money: float          # abs(butterfly_price) if negative
    arb_trade: str


@dataclass
class VerticalSpreadBound:
    ticker: str
    expiry: date
    low_strike: float
    high_strike: float
    call_spread_price: float    # C(K1) - C(K2)
    intrinsic_bound: float      # should be ≤ max strike diff * exp(-rT)
    intrinsic_bound_violation: bool
    positive_bound_violation: bool  # spread price < 0 (can't happen)
    deviation: float


@dataclass
class SurfaceArbitrageReport:
    ticker: str
    timestamp: datetime
    pcp_violations: List[PutCallParityViolation]
    calendar_violations: List[CalendarSpreadViolation]
    butterfly_arbs: List[ButterflyArb]
    vertical_bound_violations: List[VerticalSpreadBound]
    total_violations: int
    estimated_arb_value: float  # total value of detected arbitrages
    is_clean: bool


# ---------------------------------------------------------------------------
# Put-call parity checker
# ---------------------------------------------------------------------------

class PutCallParityChecker:
    """
    C - P = S - K*exp(-rT) = F*exp(-rT) - K*exp(-rT)
    where F = S*exp(rT) is the forward price.
    """

    TRANSACTION_COST_BPS = 20   # 20bps round-trip

    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        self.r = risk_free_rate
        self.q = dividend_yield   # continuous dividend yield

    def check_chain(
        self,
        ticker: str,
        spot: float,
        chain: List[Dict],
        expiry: date,
    ) -> List[PutCallParityViolation]:
        T = max((expiry - date.today()).days, 1) / 365.0

        # Build strike-indexed call/put price maps
        calls: Dict[float, Dict] = {}
        puts:  Dict[float, Dict] = {}

        for opt in chain:
            try:
                opt_exp = datetime.strptime(opt.get("expiration_date",""), "%Y-%m-%d").date()
                if opt_exp != expiry:
                    continue
            except ValueError:
                pass

            k = float(opt.get("strike", 0))
            mid = (float(opt.get("bid",0)) + float(opt.get("ask",0))) / 2
            if mid <= 0:
                mid = float(opt.get("last", 0))
            ot = opt.get("option_type","").lower()
            if ot == "call":
                calls[k] = {"mid": mid, **opt}
            else:
                puts[k] = {"mid": mid, **opt}

        common_strikes = set(calls) & set(puts)
        violations = []
        transaction_cost = self.TRANSACTION_COST_BPS / 10000

        for K in sorted(common_strikes):
            c = calls[K]["mid"]
            p = puts[K]["mid"]

            # Forward price
            F = spot * math.exp((self.r - self.q) * T)
            pv_F_K = (F - K) * math.exp(-self.r * T)   # = S*e^(-qT) - K*e^(-rT)

            lhs = c - p
            rhs = pv_F_K
            dev = lhs - rhs
            dev_bps = abs(dev / max(abs(rhs), 0.01)) * 10000

            is_arb = dev_bps > self.TRANSACTION_COST_BPS * 100

            if dev > 0:
                direction = "sell_call_buy_put_buy_forward"
            else:
                direction = "buy_call_sell_put_sell_forward"

            net_profit = (abs(dev) - spot * transaction_cost) * 100  # per contract

            violations.append(PutCallParityViolation(
                ticker=ticker, strike=K, expiry=expiry,
                call_price=c, put_price=p, spot=spot, forward=F,
                pcp_lhs=lhs, pcp_rhs=rhs, deviation=dev,
                deviation_bps=dev_bps,
                is_arbitrage=is_arb,
                arb_direction=direction,
                net_profit_per_lot=net_profit,
            ))

        return sorted(violations, key=lambda v: abs(v.deviation_bps), reverse=True)


# ---------------------------------------------------------------------------
# Calendar spread violation detector
# ---------------------------------------------------------------------------

class CalendarSpreadViolationDetector:
    """
    Detects calendar spread mispricings.
    For European options: IV(near) > IV(far) at same strike is unusual.
    For American options on dividend stocks: may have valid calendar backwardation.
    """

    def check(
        self,
        ticker: str,
        spot: float,
        chains: Dict[date, List[Dict]],  # {expiry: chain}
    ) -> List[CalendarSpreadViolation]:
        violations = []
        sorted_expiries = sorted(chains.keys())

        if len(sorted_expiries) < 2:
            return []

        for i in range(len(sorted_expiries) - 1):
            near_exp = sorted_expiries[i]
            far_exp  = sorted_expiries[i + 1]

            near_chain = chains[near_exp]
            far_chain  = chains[far_exp]

            near_strikes = {float(o.get("strike",0)): o for o in near_chain}
            far_strikes  = {float(o.get("strike",0)): o for o in far_chain}

            for K in set(near_strikes) & set(far_strikes):
                near_opt = near_strikes[K]
                far_opt  = far_strikes[K]

                for ot in ("call", "put"):
                    near_iv = float(near_opt.get("implied_volatility", 0.25)) if near_opt.get("option_type","").lower() == ot else None
                    far_iv  = float(far_opt.get("implied_volatility",  0.25)) if far_opt.get("option_type","").lower() == ot else None

                    if near_iv is None or far_iv is None:
                        continue
                    if near_iv <= 0 or far_iv <= 0:
                        continue

                    # Violation: near IV significantly exceeds far IV
                    if near_iv > far_iv * 1.15:  # >15% premium = suspicious
                        deviation = near_iv - far_iv
                        violations.append(CalendarSpreadViolation(
                            ticker=ticker, strike=K,
                            near_expiry=near_exp, far_expiry=far_exp,
                            near_iv=near_iv, far_iv=far_iv,
                            violation_type="near_backwardation",
                            deviation_pts=deviation,
                            is_arbitrage=deviation > 0.05,  # >5 vol pts
                            expected_roll_up=far_iv,
                            arb_trade=f"Buy far {ot} (sell near vol), sell near {ot} (buy near vol)",
                        ))

        return sorted(violations, key=lambda v: abs(v.deviation_pts), reverse=True)


# ---------------------------------------------------------------------------
# Butterfly arbitrage detector
# ---------------------------------------------------------------------------

class ButterflyArbDetector:
    """
    Detects butterfly arbitrages: C(K1) - 2*C(K2) + C(K3) must be ≥ 0
    for any K1 < K2 < K3 with K2 = (K1+K3)/2.
    A negative butterfly price implies a free lunch.
    """

    def detect(
        self,
        ticker: str,
        spot: float,
        chain: List[Dict],
        expiry: date,
    ) -> List[ButterflyArb]:
        calls = sorted(
            [(float(o.get("strike",0)), (float(o.get("bid",0))+float(o.get("ask",0)))/2)
             for o in chain if o.get("option_type","").lower() == "call"
             and o.get("bid") and o.get("ask")],
            key=lambda x: x[0]
        )

        if len(calls) < 3:
            return []

        violations = []
        for i in range(len(calls) - 2):
            K1, C1 = calls[i]
            K2, C2 = calls[i+1]
            K3, C3 = calls[i+2]

            # Only consider symmetric butterflies
            if abs((K1 + K3) / 2 - K2) > K2 * 0.01:
                continue

            bf_price = C1 - 2*C2 + C3

            if bf_price < -0.05:  # small tolerance for bid-ask
                violations.append(ButterflyArb(
                    ticker=ticker, expiry=expiry,
                    low_strike=K1, mid_strike=K2, high_strike=K3,
                    call_prices=(C1, C2, C3),
                    butterfly_price=bf_price,
                    is_negative=True,
                    free_money=abs(bf_price) * 100,
                    arb_trade=f"Buy K1=${K1:.0f} and K3=${K3:.0f} calls, sell 2x K2=${K2:.0f} call",
                ))

        return sorted(violations, key=lambda v: v.free_money, reverse=True)


# ---------------------------------------------------------------------------
# Vertical spread bounds checker
# ---------------------------------------------------------------------------

class VerticalSpreadBoundsChecker:
    """
    Checks call spread no-arbitrage bounds:
    0 ≤ C(K1) - C(K2) ≤ (K2-K1)*exp(-rT)  for K1 < K2
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def check(
        self,
        ticker: str,
        spot: float,
        chain: List[Dict],
        expiry: date,
    ) -> List[VerticalSpreadBound]:
        T = max((expiry - date.today()).days, 1) / 365.0

        calls = sorted(
            [(float(o.get("strike",0)),
              (float(o.get("bid",0))+float(o.get("ask",0)))/2)
             for o in chain if o.get("option_type","").lower() == "call"
             and float(o.get("bid",0)) > 0],
            key=lambda x: x[0]
        )

        violations = []
        for i in range(len(calls) - 1):
            K1, C1 = calls[i]
            K2, C2 = calls[i+1]

            spread = C1 - C2
            upper_bound = (K2 - K1) * math.exp(-self.r * T)

            upper_viol = spread > upper_bound * 1.01
            lower_viol = spread < -0.05

            if upper_viol or lower_viol:
                violations.append(VerticalSpreadBound(
                    ticker=ticker, expiry=expiry,
                    low_strike=K1, high_strike=K2,
                    call_spread_price=spread,
                    intrinsic_bound=upper_bound,
                    intrinsic_bound_violation=upper_viol,
                    positive_bound_violation=lower_viol,
                    deviation=max(spread - upper_bound, -spread, 0),
                ))

        return sorted(violations, key=lambda v: v.deviation, reverse=True)


# ---------------------------------------------------------------------------
# Vol surface smoother / fitter
# ---------------------------------------------------------------------------

class VolSurfaceFitter:
    """
    Fits a smooth vol surface (SVI or polynomial) to observed market IVs.
    Used for interpolation and detecting outliers.
    """

    def fit_svi(
        self,
        strikes: List[float],
        ivs: List[float],
        T: float,
        spot: float,
    ) -> Dict:
        """
        Fit Stochastic Volatility Inspired (SVI) parameterization:
        w(k) = a + b * [ρ(k-m) + sqrt((k-m)^2 + σ^2)]
        where k = log(K/F), w = IV^2 * T
        Returns params dict.
        """
        if len(strikes) < 5:
            return {"a": 0.04*T, "b": 0.04, "rho": -0.5, "m": 0.0, "sigma": 0.1}

        F = spot  # simplified: use spot as forward
        k = [math.log(K/F) for K in strikes]
        w = [iv**2 * T for iv in ivs]

        # Simple grid search for SVI parameters
        best_params = {"a": 0.04*T, "b": 0.04, "rho": -0.5, "m": 0.0, "sigma": 0.1}
        best_err = float("inf")

        for rho in [-0.8, -0.5, -0.3, 0.0]:
            for m in [-0.1, 0.0, 0.1]:
                for sigma in [0.05, 0.10, 0.20]:
                    # Fit a, b by LS
                    X = np.array([1.0 if True else 0 for _ in k])
                    psi = np.array([rho*(ki - m) + math.sqrt((ki-m)**2 + sigma**2) for ki in k])
                    A = np.column_stack([np.ones(len(k)), psi])
                    try:
                        res = np.linalg.lstsq(A, w, rcond=None)
                        a_, b_ = res[0]
                        if b_ <= 0:
                            continue
                        pred = a_ + b_ * psi
                        err = float(np.mean((np.array(w) - pred)**2))
                        if err < best_err:
                            best_err = err
                            best_params = {"a": float(a_), "b": float(b_), "rho": rho, "m": m, "sigma": sigma}
                    except Exception:
                        continue

        return best_params

    def svi_iv(self, K: float, F: float, T: float, params: Dict) -> float:
        """Evaluate SVI model at a given strike."""
        a, b, rho, m, sigma = params["a"], params["b"], params["rho"], params["m"], params["sigma"]
        k = math.log(K / F) if F > 0 else 0.0
        w = a + b * (rho * (k - m) + math.sqrt((k - m)**2 + sigma**2))
        w = max(0.0001 * T, w)
        return math.sqrt(w / T)

    def detect_outliers(
        self,
        strikes: List[float],
        market_ivs: List[float],
        T: float,
        spot: float,
        threshold_pts: float = 0.03,   # 3 vol pts
    ) -> List[Dict]:
        """Find strikes where market IV deviates significantly from SVI fit."""
        params = self.fit_svi(strikes, market_ivs, T, spot)
        outliers = []
        for K, iv in zip(strikes, market_ivs):
            model_iv = self.svi_iv(K, spot, T, params)
            dev = iv - model_iv
            if abs(dev) > threshold_pts:
                outliers.append({
                    "strike": K,
                    "market_iv": iv,
                    "model_iv": model_iv,
                    "deviation_pts": dev,
                    "direction": "expensive" if dev > 0 else "cheap",
                })
        return sorted(outliers, key=lambda x: abs(x["deviation_pts"]), reverse=True)


# ---------------------------------------------------------------------------
# Main surface arb facade
# ---------------------------------------------------------------------------

class VolSurfaceArbitrageScanner:
    """Unified scanner for vol surface arbitrage opportunities."""

    def __init__(self):
        self.feed = OptionDataFeed()
        self.pcp_checker  = PutCallParityChecker()
        self.cal_checker  = CalendarSpreadViolationDetector()
        self.bf_detector  = ButterflyArbDetector()
        self.vert_checker = VerticalSpreadBoundsChecker()
        self.fitter       = VolSurfaceFitter()

    def full_scan(self, ticker: str, n_expirations: int = 3) -> SurfaceArbitrageReport:
        q = self.feed.get_quotes(ticker)
        spot = float(q.get("last", 100.0))
        exps = self.feed.get_expirations(ticker)[:n_expirations]

        if not exps:
            return SurfaceArbitrageReport(
                ticker=ticker, timestamp=datetime.now(timezone.utc),
                pcp_violations=[], calendar_violations=[], butterfly_arbs=[],
                vertical_bound_violations=[], total_violations=0,
                estimated_arb_value=0.0, is_clean=True,
            )

        # Build chains
        chains: Dict[date, List[Dict]] = {}
        for exp_str in exps:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            chain = self.feed.get_option_chain(ticker, exp_str)
            chains[exp_date] = chain

        front_exp = sorted(chains.keys())[0]
        front_chain = chains[front_exp]

        pcp_viols   = self.pcp_checker.check_chain(ticker, spot, front_chain, front_exp)
        cal_viols   = self.cal_checker.check(ticker, spot, chains)
        bf_arbs     = self.bf_detector.detect(ticker, spot, front_chain, front_exp)
        vert_viols  = self.vert_checker.check(ticker, spot, front_chain, front_exp)

        pcp_arbs  = [v for v in pcp_viols  if v.is_arbitrage]
        bf_arbs_f = [b for b in bf_arbs    if b.is_negative]
        cal_arbs  = [c for c in cal_viols  if c.is_arbitrage]
        v_arbs    = [v for v in vert_viols if v.intrinsic_bound_violation or v.positive_bound_violation]

        total = len(pcp_arbs) + len(bf_arbs_f) + len(cal_arbs) + len(v_arbs)

        arb_value = (
            sum(abs(v.net_profit_per_lot) for v in pcp_arbs) +
            sum(b.free_money for b in bf_arbs_f)
        )

        return SurfaceArbitrageReport(
            ticker=ticker, timestamp=datetime.now(timezone.utc),
            pcp_violations=pcp_viols[:5],
            calendar_violations=cal_viols[:5],
            butterfly_arbs=bf_arbs_f[:5],
            vertical_bound_violations=v_arbs[:5],
            total_violations=total,
            estimated_arb_value=arb_value,
            is_clean=(total == 0),
        )

    def format_report(self, report: SurfaceArbitrageReport) -> str:
        lines = [
            f"=== Vol Surface Arbitrage Scan: {report.ticker} ===",
            f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M')} UTC",
            f"Surface Clean: {'YES ✓' if report.is_clean else 'NO — VIOLATIONS FOUND'}",
            f"Total Arbitrage Violations: {report.total_violations}",
            f"Estimated Arb Value: ${report.estimated_arb_value:,.0f}",
            "",
        ]

        if report.pcp_violations:
            lines.append("Put-Call Parity Violations:")
            for v in report.pcp_violations[:3]:
                lines.append(f"  ${v.strike:.0f} exp {v.expiry}: "
                            f"dev={v.deviation:+.3f} ({v.deviation_bps:.0f}bps) "
                            f"{'ARB!' if v.is_arbitrage else ''}")

        if report.butterfly_arbs:
            lines.append("\nButterfly Arbitrages:")
            for b in report.butterfly_arbs:
                lines.append(f"  ${b.low_strike:.0f}-${b.mid_strike:.0f}-${b.high_strike:.0f}: "
                            f"bf={b.butterfly_price:.3f} → ${b.free_money:.0f}/lot")

        if report.calendar_violations:
            lines.append("\nCalendar Spread Violations:")
            for c in report.calendar_violations[:3]:
                lines.append(f"  ${c.strike:.0f}: near_iv={c.near_iv:.1%} far_iv={c.far_iv:.1%} "
                            f"dev={c.deviation_pts:.1%} {'ARB!' if c.is_arbitrage else ''}")

        if report.is_clean:
            lines.append("\nNo significant arbitrage opportunities detected.")
            lines.append("Surface appears well-priced relative to no-arb bounds.")

        return "\n".join(lines)

    def outlier_strikes(self, ticker: str) -> List[Dict]:
        """Find strikes where IV is unusually rich or cheap vs SVI fit."""
        q = self.feed.get_quotes(ticker)
        spot = float(q.get("last", 100.0))
        exps = self.feed.get_expirations(ticker)[:1]
        if not exps:
            return []
        chain = self.feed.get_option_chain(ticker, exps[0])
        exp_date = datetime.strptime(exps[0], "%Y-%m-%d").date()
        T = max((exp_date - date.today()).days, 1) / 365.0

        calls = [(float(o.get("strike",0)), float(o.get("implied_volatility",0.25)))
                 for o in chain if o.get("option_type","").lower()=="call"
                 and float(o.get("implied_volatility",0)) > 0]
        calls.sort()
        if len(calls) < 5:
            return []

        strikes = [k for k,_ in calls]
        ivs     = [iv for _,iv in calls]
        return self.fitter.detect_outliers(strikes, ivs, T, spot)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vol surface arb CLI")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--action", choices=["scan", "outliers", "pcp"], default="scan")
    args = parser.parse_args()

    scanner = VolSurfaceArbitrageScanner()

    if args.action == "scan":
        report = scanner.full_scan(args.ticker)
        print(scanner.format_report(report))
    elif args.action == "outliers":
        outliers = scanner.outlier_strikes(args.ticker)
        print(f"IV outliers for {args.ticker}:")
        for o in outliers[:10]:
            print(f"  K=${o['strike']:.0f}: market={o['market_iv']:.1%} model={o['model_iv']:.1%} "
                  f"dev={o['deviation_pts']:+.1%} ({o['direction']})")
    elif args.action == "pcp":
        q = scanner.feed.get_quotes(args.ticker)
        spot = float(q.get("last", 100.0))
        exps = scanner.feed.get_expirations(args.ticker)[:1]
        if exps:
            exp_date = datetime.strptime(exps[0], "%Y-%m-%d").date()
            chain = scanner.feed.get_option_chain(args.ticker, exps[0])
            viols = scanner.pcp_checker.check_chain(args.ticker, spot, chain, exp_date)
            print(f"Put-Call Parity check for {args.ticker} (top 5):")
            for v in viols[:5]:
                print(f"  K=${v.strike:.0f}: lhs={v.pcp_lhs:.3f} rhs={v.pcp_rhs:.3f} "
                      f"dev={v.deviation_bps:.0f}bps {'⚡ARB' if v.is_arbitrage else ''}")
