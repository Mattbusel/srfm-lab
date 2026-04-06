"""
market_impact.py — Market impact estimation models.

Models
------
AlmgrenChrissModel    Optimal execution trajectory (Almgren & Chriss 2000)
SquareRootImpact      Simple σ * sqrt(Q/ADV) model
LinearProgramExecution  LP-based optimal slice schedule

Backtest mode
-------------
    python market_impact.py --backtest --symbol ETH --since 2024-01-01
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import yfinance as yf  # type: ignore
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    from scipy.optimize import linprog  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    log.warning("scipy not installed — LinearProgramExecution disabled")

# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

_CRYPTO_SYMS = {
    "BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK", "UNI", "AAVE",
    "CRV", "SUSHI", "BAT", "YFI", "DOT", "LTC", "BCH", "SHIB",
}


def _is_crypto(symbol: str) -> bool:
    return symbol.upper() in _CRYPTO_SYMS


def fetch_adv(symbol: str, lookback_days: int = 20) -> float:
    """
    Fetch Average Daily Volume in USD from yfinance.
    Falls back to a conservative default if unavailable.
    """
    if not _HAS_YF:
        return 50_000_000.0 if _is_crypto(symbol) else 200_000_000.0
    try:
        ticker_sym = f"{symbol.upper()}-USD" if _is_crypto(symbol) else symbol.upper()
        hist = yf.Ticker(ticker_sym).history(
            period=f"{lookback_days}d", interval="1d", auto_adjust=True
        )
        if hist.empty:
            raise ValueError("empty history")
        adv = float((hist["Close"] * hist["Volume"]).mean())
        log.debug("ADV(%s) = $%.0f", symbol, adv)
        return adv
    except Exception as exc:
        log.debug("ADV fetch for %s failed: %s", symbol, exc)
        return 50_000_000.0 if _is_crypto(symbol) else 200_000_000.0


def fetch_daily_volatility(symbol: str, lookback_days: int = 30) -> float:
    """
    Fetch daily return volatility from yfinance.
    Returns daily vol (e.g. 0.03 = 3%).
    """
    if not _HAS_YF:
        return 0.03 if _is_crypto(symbol) else 0.015
    try:
        ticker_sym = f"{symbol.upper()}-USD" if _is_crypto(symbol) else symbol.upper()
        hist = yf.Ticker(ticker_sym).history(
            period=f"{lookback_days}d", interval="1d", auto_adjust=True
        )
        if hist.empty or len(hist) < 5:
            raise ValueError("insufficient data")
        vol = float(hist["Close"].pct_change().dropna().std())
        log.debug("Vol(%s) = %.4f", symbol, vol)
        return vol
    except Exception as exc:
        log.debug("Vol fetch for %s failed: %s", symbol, exc)
        return 0.03 if _is_crypto(symbol) else 0.015


# ---------------------------------------------------------------------------
# Almgren-Chriss Model
# ---------------------------------------------------------------------------

@dataclass
class ACSlice:
    """One time-slice in an Almgren-Chriss execution schedule."""
    slice_index: int
    start_time_frac: float    # fraction of total horizon [0,1)
    end_time_frac: float
    shares_to_trade: float    # positive = trade
    shares_remaining: float
    expected_cost_bps: float
    cumulative_cost_bps: float


@dataclass
class ACResult:
    """Full result from Almgren-Chriss optimization."""
    symbol: str
    total_shares: float
    notional_usd: float
    horizon_periods: int
    risk_aversion: float       # lambda
    schedule: List[ACSlice] = field(default_factory=list)
    total_expected_cost_bps: float = 0.0
    total_variance_bps2: float = 0.0
    efficient_frontier: List[Tuple[float, float]] = field(default_factory=list)
    # (cost_bps, variance_bps2) pairs across lambda values


@dataclass
class EfficientFrontierPoint:
    urgency: float             # 0 = leisurely, 1 = urgent
    risk_aversion: float
    expected_cost_bps: float
    variance_bps2: float
    n_slices: int


class AlmgrenChrissModel:
    """
    Almgren-Chriss (2000) optimal execution model.

    The model minimizes:
        E[cost] + lambda * Var[cost]

    where the cost has temporary impact (eta * v_t) and permanent
    impact (gamma * rate_of_trading) components.

    Parameters
    ----------
    symbol       : ticker symbol
    notional_usd : order size in USD
    side         : "buy" or "sell"
    horizon_periods : number of trading periods (slices)
    sigma        : daily return volatility (decimal)
    adv_usd      : average daily volume in USD
    eta          : temporary impact coefficient
    gamma        : permanent impact coefficient
    risk_aversion: lambda — tradeoff between cost and variance
    """

    def __init__(
        self,
        symbol: str,
        notional_usd: float,
        side: str = "buy",
        horizon_periods: int = 10,
        sigma: Optional[float] = None,
        adv_usd: Optional[float] = None,
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
        risk_aversion: float = 1e-6,
    ) -> None:
        self.symbol = symbol.upper()
        self.notional_usd = notional_usd
        self.side = side.lower()
        self.horizon_periods = horizon_periods
        self.risk_aversion = risk_aversion

        # Market params — fetch if not provided
        self.sigma = sigma if sigma is not None else fetch_daily_volatility(symbol)
        self.adv_usd = adv_usd if adv_usd is not None else fetch_adv(symbol)

        # Impact params — defaults calibrated to equity markets
        # eta: temp impact = eta * (v_t / ADV_period)
        # gamma: perm impact = gamma * (V / ADV)
        self.eta = eta if eta is not None else 0.1
        self.gamma = gamma if gamma is not None else 0.05

        # Derived
        # shares ~ notional / mid_price; we work in USD-normalized units
        self._X = notional_usd  # total shares in USD equiv
        # per-period ADV
        self._adv_period = self.adv_usd / horizon_periods

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def optimal_schedule(self) -> ACResult:
        """
        Compute the optimal execution schedule.

        Uses the closed-form trajectory from Almgren-Chriss:
            x_j = X * sinh(kappa*(T-t_j)) / sinh(kappa*T)

        where kappa = arccosh(0.5*tau^2*(lambda*sigma^2/eta + 2/tau) + 1) / tau
        and tau = T/N (time per period).
        """
        N = self.horizon_periods
        X = self._X
        lam = self.risk_aversion
        sigma = self.sigma
        eta = self.eta
        gamma = self.gamma
        adv = self._adv_period

        # kappa calculation
        tau = 1.0  # normalized period length
        tilde_eta = eta / adv
        tilde_gamma = gamma / adv

        try:
            kappa_sq = (lam * sigma**2) / (tilde_eta)
            kappa_sq = max(kappa_sq, 1e-12)
            kappa = math.acosh(0.5 * tau**2 * kappa_sq + 1.0) / tau
        except (ValueError, ZeroDivisionError):
            kappa = 0.01  # degenerate fallback

        # trajectory: shares remaining at each period boundary
        trajectory = []
        for j in range(N + 1):
            denom = math.sinh(kappa * N * tau)
            if denom == 0:
                x_j = X * (1.0 - j / N)
            else:
                x_j = X * math.sinh(kappa * (N - j) * tau) / denom
            trajectory.append(x_j)

        # build slices
        slices = []
        cum_cost = 0.0
        for i in range(N):
            x_start = trajectory[i]
            x_end = trajectory[i + 1]
            trade_size = x_start - x_end  # shares sold in this period
            v = trade_size / tau           # trading rate

            # temporary cost
            temp_cost = tilde_eta * v * trade_size
            # permanent cost (simplified: accrued on all remaining shares)
            perm_cost = tilde_gamma * v * x_start
            slice_cost = temp_cost + perm_cost

            # convert to bps of notional
            slice_cost_bps = (slice_cost / X) * 10_000 if X > 0 else 0.0
            cum_cost += slice_cost_bps

            slices.append(ACSlice(
                slice_index=i,
                start_time_frac=i / N,
                end_time_frac=(i + 1) / N,
                shares_to_trade=round(trade_size, 4),
                shares_remaining=round(x_end, 4),
                expected_cost_bps=round(slice_cost_bps, 4),
                cumulative_cost_bps=round(cum_cost, 4),
            ))

        # total variance (simplified)
        # Var ≈ sigma^2 * sum_j(tau * x_j^2)
        total_var = sum(
            sigma**2 * tau * trajectory[j]**2
            for j in range(N)
        )
        var_bps2 = (total_var / X**2 * 1e8) if X > 0 else 0.0

        result = ACResult(
            symbol=self.symbol,
            total_shares=X,
            notional_usd=self.notional_usd,
            horizon_periods=N,
            risk_aversion=lam,
            schedule=slices,
            total_expected_cost_bps=round(cum_cost, 4),
            total_variance_bps2=round(var_bps2, 4),
        )
        return result

    def efficient_frontier(
        self, n_points: int = 20
    ) -> List[EfficientFrontierPoint]:
        """
        Compute the cost-variance efficient frontier by sweeping risk_aversion.

        Returns list of EfficientFrontierPoint sorted by urgency.
        """
        # lambda range: low = TWAP (uniform), high = aggressive
        lambda_vals = np.logspace(-10, -3, n_points)
        points = []
        for i, lam in enumerate(lambda_vals):
            orig_lam = self.risk_aversion
            self.risk_aversion = float(lam)
            try:
                res = self.optimal_schedule()
                points.append(EfficientFrontierPoint(
                    urgency=float(i) / (n_points - 1),
                    risk_aversion=float(lam),
                    expected_cost_bps=res.total_expected_cost_bps,
                    variance_bps2=res.total_variance_bps2,
                    n_slices=res.horizon_periods,
                ))
            finally:
                self.risk_aversion = orig_lam
        return points

    def print_schedule(self, result: Optional[ACResult] = None) -> None:
        r = result or self.optimal_schedule()
        print(f"\nAlmgren-Chriss Schedule -- {r.symbol}  "
              f"${r.notional_usd:,.0f}  N={r.horizon_periods}  lam={r.risk_aversion:.2e}")
        print(f"Total expected cost: {r.total_expected_cost_bps:.2f} bps  "
              f"Variance: {r.total_variance_bps2:.2f} bps^2")
        print(f"{'Slice':>5}  {'t_start':>7}  {'t_end':>7}  {'Trade$':>12}  "
              f"{'Remain$':>12}  {'Slice bps':>10}  {'Cum bps':>9}")
        for sl in r.schedule:
            print(f"  {sl.slice_index:>3}  {sl.start_time_frac:>7.2f}  "
                  f"{sl.end_time_frac:>7.2f}  {sl.shares_to_trade:>12,.2f}  "
                  f"{sl.shares_remaining:>12,.2f}  {sl.expected_cost_bps:>10.4f}  "
                  f"{sl.cumulative_cost_bps:>9.4f}")


# ---------------------------------------------------------------------------
# Square-Root Impact
# ---------------------------------------------------------------------------

class SquareRootImpact:
    """
    Square-root market impact model.

    I(Q) = sigma * sqrt(Q / ADV)

    where Q is order size, ADV is average daily volume (both in USD).

    Returns impact in basis points.
    """

    def __init__(
        self,
        symbol: str,
        sigma: Optional[float] = None,
        adv_usd: Optional[float] = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.sigma = sigma if sigma is not None else fetch_daily_volatility(symbol)
        self.adv_usd = adv_usd if adv_usd is not None else fetch_adv(symbol)

    def impact_bps(self, notional_usd: float) -> float:
        """Return market impact in basis points for an order of *notional_usd*."""
        if self.adv_usd <= 0:
            return 0.0
        participation = notional_usd / self.adv_usd
        return self.sigma * math.sqrt(participation) * 10_000

    def impact_curve(
        self, max_notional: float, n_points: int = 50
    ) -> List[Tuple[float, float]]:
        """
        Return (notional_usd, impact_bps) pairs over [0, max_notional].
        Useful for plotting the impact function.
        """
        sizes = np.linspace(0, max_notional, n_points)
        return [(float(s), self.impact_bps(float(s))) for s in sizes]

    def inverse_impact(self, target_bps: float) -> float:
        """
        Return the maximum order size that stays within *target_bps* of impact.
        """
        if target_bps <= 0 or self.sigma <= 0:
            return 0.0
        participation = (target_bps / (self.sigma * 10_000)) ** 2
        return participation * self.adv_usd


# ---------------------------------------------------------------------------
# Linear Program Execution
# ---------------------------------------------------------------------------

@dataclass
class LPConstraints:
    """Constraints for the LP execution optimizer."""
    max_participation_rate: float = 0.10   # max fraction of ADV per period
    min_slice_fraction: float = 0.0        # min trade per period as fraction of total
    max_slice_fraction: float = 1.0        # max trade per period
    urgency_deadline: Optional[int] = None # must complete by this period index


@dataclass
class LPResult:
    """Result from LP-based execution optimizer."""
    symbol: str
    notional_usd: float
    n_periods: int
    schedule_fractions: List[float]   # fraction of total to trade per period
    schedule_notional: List[float]    # USD to trade per period
    total_expected_cost_bps: float
    feasible: bool
    solver_message: str


class LinearProgramExecution:
    """
    LP-based optimal execution schedule.

    Minimizes total market impact + timing risk subject to:
    - trade exactly notional_usd over N periods
    - participation rate <= max_participation_rate per period
    - non-negativity
    - optional urgency deadline

    Parameters
    ----------
    symbol         : ticker
    notional_usd   : total order size in USD
    n_periods      : number of execution periods
    sigma          : daily vol per period (scaled to period)
    adv_usd        : average daily volume in USD
    constraints    : LPConstraints
    """

    def __init__(
        self,
        symbol: str,
        notional_usd: float,
        n_periods: int = 10,
        sigma: Optional[float] = None,
        adv_usd: Optional[float] = None,
        constraints: Optional[LPConstraints] = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.notional_usd = notional_usd
        self.n_periods = n_periods
        self.sigma = sigma if sigma is not None else fetch_daily_volatility(symbol)
        self.adv_usd = adv_usd if adv_usd is not None else fetch_adv(symbol)
        self.constraints = constraints or LPConstraints()

    def optimize(self) -> LPResult:
        """
        Solve the LP and return optimal execution fractions.

        If scipy is unavailable, falls back to uniform TWAP schedule.
        """
        if not _HAS_SCIPY:
            return self._twap_fallback()

        N = self.n_periods
        Q = self.notional_usd
        adv = self.adv_usd
        sigma = self.sigma
        c = self.constraints

        # Decision variables: q_i = fraction of Q traded in period i, i=0..N-1
        # Objective: minimize sum_i [ impact(q_i) + timing_risk(q_i) ]
        # We linearize impact as: alpha * q_i   (first-order sqrt approx near Q/N)
        # timing risk weight = sigma * sqrt(remaining) → approximate as beta*(N-i)*q_i

        # Linear objective coefficients (impact + timing)
        adv_period = adv / N if N > 0 else adv
        # impact slope at reference point Q/N
        q_ref = Q / N if N > 0 else Q
        if adv_period > 0 and q_ref > 0:
            impact_slope = sigma * 0.5 / math.sqrt(q_ref / adv_period) / adv_period
        else:
            impact_slope = 0.0

        obj = np.array([
            impact_slope * (i + 1) + sigma * (N - i) * 0.01
            for i in range(N)
        ])

        # Equality constraint: sum(q_i) = 1  (fractions sum to 1)
        A_eq = np.ones((1, N))
        b_eq = np.array([1.0])

        # Inequality constraints: q_i <= max_participation_rate * adv_period / Q
        max_q = c.max_participation_rate * adv_period / Q if Q > 0 else 1.0
        max_q = min(max_q, c.max_slice_fraction)

        bounds = [(c.min_slice_fraction, max_q)] * N

        # Urgency deadline: sum(q_i for i<=deadline) >= 1 after deadline
        A_ub = []
        b_ub_vec = []
        if c.urgency_deadline is not None and c.urgency_deadline < N:
            d = c.urgency_deadline
            # Must complete trading by period d: sum q_0..q_d >= 1
            row = [-1.0 if i <= d else 0.0 for i in range(N)]
            A_ub.append(row)
            b_ub_vec.append(-1.0)

        A_ub_arr = np.array(A_ub) if A_ub else None
        b_ub_arr = np.array(b_ub_vec) if b_ub_vec else None

        try:
            res = linprog(
                obj,
                A_ub=A_ub_arr,
                b_ub=b_ub_arr,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            feasible = res.success
            fracs = list(res.x) if feasible else [1.0 / N] * N
            msg = res.message
        except Exception as exc:
            feasible = False
            fracs = [1.0 / N] * N
            msg = str(exc)

        notional_schedule = [f * Q for f in fracs]
        # total cost estimate
        sqrt_impact = SquareRootImpact(self.symbol, self.sigma, self.adv_usd)
        total_cost_bps = sum(sqrt_impact.impact_bps(n) for n in notional_schedule) / N

        return LPResult(
            symbol=self.symbol,
            notional_usd=Q,
            n_periods=N,
            schedule_fractions=[round(f, 6) for f in fracs],
            schedule_notional=[round(n, 2) for n in notional_schedule],
            total_expected_cost_bps=round(total_cost_bps, 4),
            feasible=feasible,
            solver_message=msg,
        )

    def _twap_fallback(self) -> LPResult:
        N = self.n_periods
        Q = self.notional_usd
        fracs = [1.0 / N] * N
        notional_schedule = [Q / N] * N
        sqrt_impact = SquareRootImpact(self.symbol, self.sigma, self.adv_usd)
        total_cost_bps = sum(sqrt_impact.impact_bps(n) for n in notional_schedule) / N
        return LPResult(
            symbol=self.symbol,
            notional_usd=Q,
            n_periods=N,
            schedule_fractions=fracs,
            schedule_notional=notional_schedule,
            total_expected_cost_bps=round(total_cost_bps, 4),
            feasible=True,
            solver_message="TWAP fallback (scipy unavailable)",
        )


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------

def backtest_impact_model(
    symbol: str,
    since: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict:
    """
    Apply SquareRootImpact to historical trades and compare model prediction
    vs realized slippage.  Returns a dict of prediction error statistics.
    """
    import sqlite3
    import pandas as pd

    db = db_path or (_REPO_ROOT / "execution" / "live_trades.db")
    if not db.exists():
        log.error("DB not found: %s", db)
        return {}

    conn = sqlite3.connect(db)
    try:
        clauses = ["symbol = ?"]
        params: list = [symbol.upper()]
        if since:
            clauses.append("fill_time >= ?")
            params.append(since)
        where = "WHERE " + " AND ".join(clauses)
        df = pd.read_sql_query(
            f"SELECT * FROM live_trades {where} ORDER BY fill_time",
            conn,
            params=params,
        )
    finally:
        conn.close()

    if df.empty:
        return {"symbol": symbol, "n_trades": 0, "error": "no data"}

    model = SquareRootImpact(symbol)
    # compute arrival price (first-fill of order as proxy)
    df["order_id"] = df["order_id"].fillna("_single_")
    arrival: Dict[str, float] = {}
    for oid, grp in df.groupby("order_id"):
        first3 = grp.nsmallest(3, "fill_time")
        vwap = (first3["price"] * first3["qty"]).sum() / first3["qty"].sum()
        arrival[oid] = float(vwap)

    rows = []
    for _, row in df.iterrows():
        arr_px = arrival.get(str(row["order_id"]), float(row["price"]))
        fill_px = float(row["price"])
        notional = float(row["notional"])
        side = str(row["side"]).lower()

        if arr_px > 0:
            realized = abs(fill_px - arr_px) / arr_px * 10_000
        else:
            realized = 0.0

        predicted = model.impact_bps(notional)
        rows.append({
            "realized_bps": realized,
            "predicted_bps": predicted,
            "error_bps": predicted - realized,
            "notional": notional,
        })

    err_df = pd.DataFrame(rows)
    mae = float(err_df["error_bps"].abs().mean())
    mse = float((err_df["error_bps"] ** 2).mean())
    bias = float(err_df["error_bps"].mean())
    corr = float(err_df["realized_bps"].corr(err_df["predicted_bps"]))

    result = {
        "symbol": symbol,
        "n_trades": len(err_df),
        "mean_realized_bps": float(err_df["realized_bps"].mean()),
        "mean_predicted_bps": float(err_df["predicted_bps"].mean()),
        "mae_bps": mae,
        "rmse_bps": float(math.sqrt(mse)),
        "bias_bps": bias,
        "correlation": corr,
        "adv_usd": model.adv_usd,
        "sigma": model.sigma,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    p = argparse.ArgumentParser(description="Market Impact Models")
    sub = p.add_subparsers(dest="cmd")

    # ac subcommand
    ac_p = sub.add_parser("ac", help="Almgren-Chriss schedule")
    ac_p.add_argument("--symbol", required=True)
    ac_p.add_argument("--notional", type=float, required=True)
    ac_p.add_argument("--periods", type=int, default=10)
    ac_p.add_argument("--lambda", dest="lam", type=float, default=1e-6)
    ac_p.add_argument("--frontier", action="store_true")

    # sqrt subcommand
    sq_p = sub.add_parser("sqrt", help="Square-root impact curve")
    sq_p.add_argument("--symbol", required=True)
    sq_p.add_argument("--notional", type=float, required=True)

    # lp subcommand
    lp_p = sub.add_parser("lp", help="LP-based schedule")
    lp_p.add_argument("--symbol", required=True)
    lp_p.add_argument("--notional", type=float, required=True)
    lp_p.add_argument("--periods", type=int, default=10)
    lp_p.add_argument("--max-part", type=float, default=0.10)

    # backtest subcommand
    bt_p = sub.add_parser("backtest", help="Backtest impact model")
    bt_p.add_argument("--symbol", required=True)
    bt_p.add_argument("--since", default=None)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.cmd == "ac":
        model = AlmgrenChrissModel(
            symbol=args.symbol,
            notional_usd=args.notional,
            horizon_periods=args.periods,
            risk_aversion=args.lam,
        )
        result = model.optimal_schedule()
        model.print_schedule(result)
        if args.frontier:
            print("\nEfficient Frontier:")
            print(f"  {'Urgency':>8}  {'Lambda':>12}  {'Cost bps':>10}  {'Var bps2':>10}")
            for pt in model.efficient_frontier(20):
                print(f"  {pt.urgency:>8.2f}  {pt.risk_aversion:>12.2e}  "
                      f"{pt.expected_cost_bps:>10.4f}  {pt.variance_bps2:>10.4f}")

    elif args.cmd == "sqrt":
        model = SquareRootImpact(args.symbol)
        impact = model.impact_bps(args.notional)
        max_q = model.inverse_impact(10.0)  # 10 bps threshold
        print(f"\nSquareRoot Impact — {args.symbol}")
        print(f"  ADV: ${model.adv_usd:,.0f}   Sigma: {model.sigma:.4f}")
        print(f"  Impact for ${args.notional:,.0f}: {impact:.2f} bps")
        print(f"  Max size for <10 bps impact: ${max_q:,.0f}")

    elif args.cmd == "lp":
        lpe = LinearProgramExecution(
            symbol=args.symbol,
            notional_usd=args.notional,
            n_periods=args.periods,
            constraints=LPConstraints(max_participation_rate=args.max_part),
        )
        res = lpe.optimize()
        print(f"\nLP Execution Schedule -- {res.symbol}  ${res.notional_usd:,.0f}")
        print(f"  Feasible: {res.feasible}  ({res.solver_message})")
        print(f"  Total expected cost: {res.total_expected_cost_bps:.4f} bps")
        print(f"  {'Period':>6}  {'Fraction':>10}  {'Notional $':>14}")
        for i, (frac, notional) in enumerate(
            zip(res.schedule_fractions, res.schedule_notional)
        ):
            print(f"  {i:>6}  {frac:>10.4f}  {notional:>14,.2f}")

    elif args.cmd == "backtest":
        result = backtest_impact_model(args.symbol, since=args.since)
        if not result:
            print("No data returned.")
            return
        print(f"\nBacktest — {result['symbol']}  N={result['n_trades']}")
        print(f"  Mean realized bps: {result.get('mean_realized_bps', 0):.4f}")
        print(f"  Mean predicted bps: {result.get('mean_predicted_bps', 0):.4f}")
        print(f"  MAE: {result.get('mae_bps', 0):.4f} bps")
        print(f"  RMSE: {result.get('rmse_bps', 0):.4f} bps")
        print(f"  Bias: {result.get('bias_bps', 0):.4f} bps")
        print(f"  Correlation: {result.get('correlation', 0):.4f}")

    else:
        p.print_help()


if __name__ == "__main__":
    _cli_main()
