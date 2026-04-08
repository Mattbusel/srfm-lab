"""
fx_models.py
FX/currency-specific quantitative models.

Covers:
- Garman-Kohlhagen FX option pricing
- Vanna-Volga FX vol surface construction
- SABR calibration for FX
- Cross-rate triangulation and arbitrage detection
- PPP deviation scoring
- Interest rate parity (covered/uncovered) and forward premium anomaly
- Currency carry trade scoring
- FX momentum (12-1 month cross-sectional ranking)
- Regime detection per pair (trending vs mean-reverting)
- Dual-currency bond pricing with embedded FX option
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GKResult:
    """Result of Garman-Kohlhagen option pricing."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho_d: float   # sensitivity to domestic rate
    rho_f: float   # sensitivity to foreign rate
    d1: float
    d2: float


@dataclass
class VannaVolgaSurface:
    """Vanna-Volga implied vol surface at standard strikes."""
    atm_vol: float
    rr25: float            # 25-delta risk reversal
    bf25: float            # 25-delta butterfly
    sigma_25c: float       # implied vol at 25-delta call
    sigma_25p: float       # implied vol at 25-delta put
    sigma_atm: float       # ATM vol (should equal atm_vol)
    strikes: np.ndarray    # grid of strikes
    vols: np.ndarray       # vol at each strike


@dataclass
class SABRParams:
    """Calibrated SABR parameters for FX."""
    alpha: float   # initial vol
    beta: float    # CEV exponent (fixed for FX, often 0.5 or 1.0)
    rho: float     # correlation
    nu: float      # vol-of-vol
    rmse: float    # calibration error


@dataclass
class CrossRateResult:
    """Cross-rate triangulation result."""
    implied_cross: float        # S_AC = S_AB * S_BC
    market_cross: float
    arb_pct: float              # (market - implied) / implied
    arb_present: bool
    direction: str              # 'buy_cross' or 'sell_cross'


@dataclass
class PPPScore:
    """Purchasing Power Parity deviation."""
    pair: str
    spot: float
    ppp_rate: float
    deviation_pct: float        # (spot - ppp) / ppp * 100
    z_score: float              # deviation normalised by historical std
    signal: str                 # 'overvalued' / 'undervalued' / 'fair'


@dataclass
class IRPResult:
    """Interest rate parity diagnostics."""
    pair: str
    spot: float
    forward: float
    forward_pct: float          # (F - S) / S * 100
    covered_deviation: float    # actual fwd - CIP-implied fwd
    uncovered_premium: float    # annualised forward premium
    anomaly_score: float        # z-score vs historical


@dataclass
class CarryScore:
    """Currency carry trade score."""
    pair: str
    forward_premium: float      # annualised, high = earn by selling forward
    realised_vol: float
    vol_adjusted_carry: float   # forward_premium / realised_vol
    rank: int


@dataclass
class MomentumScore:
    """FX momentum score (12-1 month cross-sectional)."""
    pair: str
    return_12m: float
    return_1m: float
    momentum: float     # 12-1 month return
    rank: int


@dataclass
class RegimeResult:
    """FX regime classification for a currency pair."""
    pair: str
    hurst: float                    # Hurst exponent
    variance_ratio: float           # Lo-MacKinlay VR stat
    adf_stat: float                 # ADF test statistic
    regime: str                     # 'trending' / 'mean_reverting' / 'random_walk'
    confidence: float               # [0, 1]


@dataclass
class DualCurrencyBond:
    """Dual-currency bond pricing result."""
    straight_bond_pv: float
    embedded_option_value: float
    dcb_price: float            # straight_bond_pv - embedded_option_value
    yield_enhancement: float    # extra yield vs straight bond, annualised
    gk_result: GKResult


# ---------------------------------------------------------------------------
# 1. Garman-Kohlhagen FX option pricing
# ---------------------------------------------------------------------------

def garman_kohlhagen(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    option_type: str = "call",
) -> GKResult:
    """
    Garman-Kohlhagen model for European FX options.

    Parameters
    ----------
    S       : spot exchange rate (units of domestic per foreign)
    K       : strike
    T       : time to expiry in years
    r_d     : domestic risk-free rate (continuously compounded)
    r_f     : foreign risk-free rate (continuously compounded)
    sigma   : implied volatility (annualised)
    option_type : 'call' or 'put'

    Returns
    -------
    GKResult dataclass with price and all first-order Greeks.
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return GKResult(intrinsic, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), float("nan"))

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    phi = 1.0 if option_type == "call" else -1.0

    Nd1 = norm.cdf(phi * d1)
    Nd2 = norm.cdf(phi * d2)
    nd1 = norm.pdf(d1)  # standard normal PDF — same for call/put Greeks

    disc_d = math.exp(-r_d * T)
    disc_f = math.exp(-r_f * T)

    price = phi * (S * disc_f * Nd1 - K * disc_d * Nd2)

    delta = phi * disc_f * Nd1
    gamma = disc_f * nd1 / (S * sigma * sqrtT)
    vega = S * disc_f * nd1 * sqrtT          # per unit of vol (not %)
    theta = (
        -S * disc_f * nd1 * sigma / (2 * sqrtT)
        - phi * r_d * K * disc_d * Nd2
        + phi * r_f * S * disc_f * Nd1
    )
    rho_d = phi * K * T * disc_d * Nd2       # dV/dr_d
    rho_f = -phi * S * T * disc_f * Nd1      # dV/dr_f

    return GKResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho_d=rho_d,
        rho_f=rho_f,
        d1=d1,
        d2=d2,
    )


def gk_implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    option_type: str = "call",
    tol: float = 1e-8,
) -> float:
    """
    Invert GK formula to obtain implied volatility via Brent's method.

    Returns implied vol as a float, or np.nan if no solution found.
    """
    def obj(sigma):
        return garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type).price - market_price

    try:
        return brentq(obj, 1e-6, 5.0, xtol=tol)
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------------------
# 2. Vanna-Volga FX vol surface
# ---------------------------------------------------------------------------

def _delta_to_strike(delta: float, S: float, T: float, r_d: float, r_f: float, sigma: float, option_type: str = "call") -> float:
    """
    Invert Black-Scholes delta to obtain strike (spot delta convention).
    delta = phi * exp(-r_f * T) * N(phi * d1)
    """
    phi = 1.0 if option_type == "call" else -1.0
    sqrtT = math.sqrt(T)

    def obj(K):
        if K <= 0:
            return -1e10
        d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        return phi * math.exp(-r_f * T) * norm.cdf(phi * d1) - delta

    try:
        return brentq(obj, S * 0.01, S * 10.0)
    except ValueError:
        return np.nan


def vanna_volga_surface(
    S: float,
    T: float,
    r_d: float,
    r_f: float,
    atm_vol: float,
    rr25: float,
    bf25: float,
    n_strikes: int = 50,
) -> VannaVolgaSurface:
    """
    Build a Vanna-Volga FX vol smile from market quotes.

    The three liquid FX vol pillars are:
      - ATM (delta-neutral straddle)
      - 25-delta Risk Reversal  : RR = sigma_25C - sigma_25P
      - 25-delta Butterfly      : BF = (sigma_25C + sigma_25P) / 2 - ATM

    Vanna-Volga approximation adds first-order corrections from the
    three hedging instruments to the ATM Black-Scholes price.

    Parameters
    ----------
    S       : spot
    T       : tenor in years
    r_d     : domestic rate
    r_f     : foreign rate
    atm_vol : ATM implied vol
    rr25    : 25-delta risk reversal (call vol - put vol)
    bf25    : 25-delta butterfly ((call vol + put vol)/2 - ATM vol)
    n_strikes : number of strike grid points

    Returns
    -------
    VannaVolgaSurface dataclass
    """
    sigma_25c = atm_vol + bf25 + 0.5 * rr25
    sigma_25p = atm_vol + bf25 - 0.5 * rr25

    K_25c = _delta_to_strike(0.25, S, T, r_d, r_f, sigma_25c, "call")
    K_25p = _delta_to_strike(-0.25, S, T, r_d, r_f, sigma_25p, "put")
    K_atm = S * math.exp((r_d - r_f + 0.5 * atm_vol ** 2) * T)   # delta-neutral ATM strike

    pillar_strikes = np.array([K_25p, K_atm, K_25c])
    pillar_vols = np.array([sigma_25p, atm_vol, sigma_25c])

    K_lo = S * 0.6
    K_hi = S * 1.6
    strikes = np.linspace(K_lo, K_hi, n_strikes)
    vv_vols = np.empty(n_strikes)

    for i, K in enumerate(strikes):
        # Vanna-Volga weights: solve for x1, x2, x3 such that
        # sum(xj * vanna_j) = vanna(K), sum(xj * volga_j) = volga(K)
        # Full VV uses vega/vanna/volga system; simplified quadratic interp used here.
        # We use the closed-form approximation from Castagna & Mercurio (2007).
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r_d - r_f + 0.5 * atm_vol ** 2) * T) / (atm_vol * sqrtT)
        d2 = d1 - atm_vol * sqrtT

        # vega, vanna, volga at target strike under ATM vol
        nd1 = norm.pdf(d1)
        vega_K = S * math.exp(-r_f * T) * nd1 * sqrtT

        if abs(vega_K) < 1e-14:
            vv_vols[i] = atm_vol
            continue

        vanna_K = -math.exp(-r_f * T) * nd1 * d2 / atm_vol
        volga_K = vega_K * d1 * d2 / atm_vol

        # same for pillars
        def _vvg(Kp, sp):
            d1p = (math.log(S / Kp) + (r_d - r_f + 0.5 * sp ** 2) * T) / (sp * sqrtT)
            d2p = d1p - sp * sqrtT
            nd1p = norm.pdf(d1p)
            vega_p = S * math.exp(-r_f * T) * nd1p * sqrtT
            vanna_p = -math.exp(-r_f * T) * nd1p * d2p / sp
            volga_p = vega_p * d1p * d2p / sp
            return vega_p, vanna_p, volga_p

        vega1, vanna1, volga1 = _vvg(K_25p, sigma_25p)
        vega2, vanna2, volga2 = _vvg(K_atm, atm_vol)
        vega3, vanna3, volga3 = _vvg(K_25c, sigma_25c)

        A = np.array([
            [vega1, vega2, vega3],
            [vanna1, vanna2, vanna3],
            [volga1, volga2, volga3],
        ])
        b = np.array([vega_K, vanna_K, volga_K])

        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            vv_vols[i] = atm_vol
            continue

        # VV price correction
        gk_atm = garman_kohlhagen(S, K, T, r_d, r_f, atm_vol, "call")
        gk1 = garman_kohlhagen(S, K_25p, T, r_d, r_f, sigma_25p, "put")
        gk2 = garman_kohlhagen(S, K_atm, T, r_d, r_f, atm_vol, "call")
        gk3 = garman_kohlhagen(S, K_25c, T, r_d, r_f, sigma_25c, "call")

        # Intrinsic VV cost of pillar replication
        gk1_atm = garman_kohlhagen(S, K_25p, T, r_d, r_f, atm_vol, "put")
        gk2_atm = garman_kohlhagen(S, K_atm, T, r_d, r_f, atm_vol, "call")
        gk3_atm = garman_kohlhagen(S, K_25c, T, r_d, r_f, atm_vol, "call")

        correction = (
            x[0] * (gk1.price - gk1_atm.price)
            + x[1] * (gk2.price - gk2_atm.price)
            + x[2] * (gk3.price - gk3_atm.price)
        )

        corrected_price = gk_atm.price + correction

        # Back out implied vol from corrected price
        iv = gk_implied_vol(max(corrected_price, 1e-12), S, K, T, r_d, r_f, "call")
        vv_vols[i] = iv if not np.isnan(iv) else atm_vol

    return VannaVolgaSurface(
        atm_vol=atm_vol,
        rr25=rr25,
        bf25=bf25,
        sigma_25c=sigma_25c,
        sigma_25p=sigma_25p,
        sigma_atm=atm_vol,
        strikes=strikes,
        vols=vv_vols,
    )


# ---------------------------------------------------------------------------
# 3. SABR calibration for FX
# ---------------------------------------------------------------------------

def sabr_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Hagan et al. (2002) SABR implied vol approximation.

    Parameters
    ----------
    F     : forward price
    K     : strike
    T     : expiry in years
    alpha : initial vol (alpha > 0)
    beta  : CEV exponent in [0, 1]
    rho   : correlation in (-1, 1)
    nu    : vol-of-vol (nu > 0)

    Returns
    -------
    Implied Black vol as float.
    """
    if abs(F - K) < 1e-10:
        # ATM formula
        FK_mid = F ** (1.0 - beta)
        term1 = alpha / FK_mid
        term2 = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0) * alpha ** 2 / FK_mid ** 2
            + rho * beta * nu * alpha / (4.0 * FK_mid)
            + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        ) * T
        return term1 * term2

    log_fk = math.log(F / K)
    FK_beta = (F * K) ** (0.5 * (1.0 - beta))
    z = nu / alpha * FK_beta * log_fk
    x_z = math.log((math.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho))

    if abs(x_z) < 1e-10:
        x_z = 1.0

    num_front = alpha
    denom_front = FK_beta * (
        1.0
        + (1.0 - beta) ** 2 / 24.0 * log_fk ** 2
        + (1.0 - beta) ** 4 / 1920.0 * log_fk ** 4
    )

    correction = 1.0 + (
        (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / FK_beta ** 2
        + rho * beta * nu * alpha / (4.0 * FK_beta)
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    ) * T

    return num_front / denom_front * (z / x_z) * correction


def sabr_calibrate_fx(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float = 0.5,
    init_guess: Optional[Tuple[float, float, float]] = None,
) -> SABRParams:
    """
    Calibrate SABR alpha, rho, nu to market FX vol smile with fixed beta.

    Parameters
    ----------
    F           : forward rate at tenor T
    T           : tenor in years
    strikes     : array of strikes
    market_vols : array of market implied vols (same length as strikes)
    beta        : fixed CEV exponent (0.5 common for FX)
    init_guess  : (alpha0, rho0, nu0)

    Returns
    -------
    SABRParams dataclass with calibrated parameters and RMSE.
    """
    if init_guess is None:
        # Sensible defaults for FX
        atm_idx = np.argmin(np.abs(strikes - F))
        alpha0 = market_vols[atm_idx] * F ** (1.0 - beta)
        init_guess = (alpha0, 0.0, 0.3)

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1.0:
            return 1e10
        model_vols = np.array([sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
        return float(np.mean((model_vols - market_vols) ** 2))

    bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]
    res = minimize(
        objective,
        init_guess,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    alpha, rho, nu = res.x
    rmse = math.sqrt(res.fun)

    return SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu, rmse=rmse)


# ---------------------------------------------------------------------------
# 4. Cross-rate triangulation and arbitrage detection
# ---------------------------------------------------------------------------

def cross_rate_triangulate(
    S_AB: float,
    S_BC: float,
    S_AC_market: float,
    transaction_cost_pct: float = 0.0,
) -> CrossRateResult:
    """
    Detect triangular arbitrage across three currency pairs.

    Convention: S_XY = units of Y per 1 unit of X.
    Implied S_AC = S_AB * S_BC.

    Parameters
    ----------
    S_AB            : spot A/B (B per A)
    S_BC            : spot B/C (C per B)
    S_AC_market     : market quote of A/C (C per A)
    transaction_cost_pct : round-trip cost fraction

    Returns
    -------
    CrossRateResult with arbitrage flag and direction.
    """
    implied = S_AB * S_BC
    arb_pct = (S_AC_market - implied) / implied

    # Net of transaction costs
    net_arb = abs(arb_pct) - transaction_cost_pct
    arb_present = net_arb > 0.0

    if S_AC_market > implied:
        direction = "sell_cross"   # sell A/C directly, buy synthetically
    else:
        direction = "buy_cross"    # buy A/C directly, sell synthetically

    return CrossRateResult(
        implied_cross=implied,
        market_cross=S_AC_market,
        arb_pct=arb_pct * 100.0,
        arb_present=arb_present,
        direction=direction if arb_present else "none",
    )


def batch_triangulate(quotes: Dict[str, float], transaction_cost_pct: float = 0.001) -> List[CrossRateResult]:
    """
    Scan all triplets in a dict of FX quotes for triangular arbitrage.

    Parameters
    ----------
    quotes : dict mapping 'CCY1/CCY2' -> mid-price
    transaction_cost_pct : one-way cost (e.g. 0.001 = 10 bps)

    Returns
    -------
    List of CrossRateResult for triplets with arb_present=True.
    """
    pairs = list(quotes.keys())
    ccys = set()
    for p in pairs:
        a, b = p.split("/")
        ccys.update([a, b])
    ccys = sorted(ccys)

    results = []
    n = len(ccys)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if len({i, j, k}) < 3:
                    continue
                A, B, C = ccys[i], ccys[j], ccys[k]
                ab = quotes.get(f"{A}/{B}") or (1.0 / quotes.get(f"{B}/{A}", 0) or None)
                bc = quotes.get(f"{B}/{C}") or (1.0 / quotes.get(f"{C}/{B}", 0) or None)
                ac = quotes.get(f"{A}/{C}") or (1.0 / quotes.get(f"{C}/{A}", 0) or None)
                if ab is None or bc is None or ac is None:
                    continue
                r = cross_rate_triangulate(ab, bc, ac, transaction_cost_pct)
                if r.arb_present:
                    results.append(r)
    return results


# ---------------------------------------------------------------------------
# 5. PPP deviation scoring
# ---------------------------------------------------------------------------

def ppp_deviation_score(
    pair: str,
    spot: float,
    ppp_rate: float,
    historical_deviations: Optional[np.ndarray] = None,
) -> PPPScore:
    """
    Score a currency pair against its PPP equilibrium rate.

    Parameters
    ----------
    pair                  : label e.g. 'USD/EUR'
    spot                  : current spot rate
    ppp_rate              : PPP-implied equilibrium rate (e.g. from OECD/IMF)
    historical_deviations : array of past (spot - ppp)/ppp values for z-score

    Returns
    -------
    PPPScore dataclass.
    """
    dev_pct = (spot - ppp_rate) / ppp_rate * 100.0

    if historical_deviations is not None and len(historical_deviations) > 1:
        mu = float(np.mean(historical_deviations))
        sigma = float(np.std(historical_deviations, ddof=1))
        z = (dev_pct / 100.0 - mu) / sigma if sigma > 0 else 0.0
    else:
        z = dev_pct / 10.0   # rough normalisation absent history

    if dev_pct > 5.0:
        signal = "overvalued"
    elif dev_pct < -5.0:
        signal = "undervalued"
    else:
        signal = "fair"

    return PPPScore(pair=pair, spot=spot, ppp_rate=ppp_rate, deviation_pct=dev_pct, z_score=z, signal=signal)


# ---------------------------------------------------------------------------
# 6. Interest rate parity (covered/uncovered) and forward premium anomaly
# ---------------------------------------------------------------------------

def cip_implied_forward(spot: float, r_d: float, r_f: float, T: float) -> float:
    """Covered interest parity (CIP) implied forward rate."""
    return spot * math.exp((r_d - r_f) * T)


def irp_diagnostics(
    pair: str,
    spot: float,
    forward_market: float,
    r_d: float,
    r_f: float,
    T: float,
    historical_deviations: Optional[np.ndarray] = None,
) -> IRPResult:
    """
    Compute covered and uncovered interest rate parity metrics.

    The *forward premium anomaly* (Fama puzzle) refers to empirical evidence
    that currencies with high forward premiums tend to depreciate less than
    predicted, generating carry trade profits.

    Parameters
    ----------
    pair               : currency pair label
    spot               : current spot
    forward_market     : market-quoted outright forward rate
    r_d                : domestic (base) annual rate
    r_f                : foreign (quote) annual rate
    T                  : tenor in years
    historical_deviations : past covered deviations for z-score

    Returns
    -------
    IRPResult dataclass.
    """
    cip_fwd = cip_implied_forward(spot, r_d, r_f, T)
    covered_dev = (forward_market - cip_fwd) / spot      # in spot points normalised
    forward_pct = (forward_market - spot) / spot * 100.0
    uncovered_premium = (r_d - r_f) * 100.0              # annualised

    if historical_deviations is not None and len(historical_deviations) > 1:
        mu = float(np.mean(historical_deviations))
        sigma = float(np.std(historical_deviations, ddof=1))
        anomaly_score = (covered_dev - mu) / sigma if sigma > 0 else 0.0
    else:
        anomaly_score = covered_dev * 100.0

    return IRPResult(
        pair=pair,
        spot=spot,
        forward=forward_market,
        forward_pct=forward_pct,
        covered_deviation=covered_dev,
        uncovered_premium=uncovered_premium,
        anomaly_score=anomaly_score,
    )


# ---------------------------------------------------------------------------
# 7. Currency carry trade scoring
# ---------------------------------------------------------------------------

def carry_trade_scores(
    pairs: List[str],
    forward_premiums: np.ndarray,
    realised_vols: np.ndarray,
) -> List[CarryScore]:
    """
    Rank currencies by vol-adjusted carry (forward premium / realised vol).

    Long currencies with highest score, short currencies with lowest score.

    Parameters
    ----------
    pairs            : list of currency pair labels
    forward_premiums : annualised forward premium for each pair (r_d - r_f approx)
    realised_vols    : annualised realised vol for each pair

    Returns
    -------
    List of CarryScore sorted descending by vol_adjusted_carry.
    """
    assert len(pairs) == len(forward_premiums) == len(realised_vols)
    scores = []
    for i, pair in enumerate(pairs):
        vol = realised_vols[i]
        fp = forward_premiums[i]
        va_carry = fp / vol if vol > 0 else 0.0
        scores.append(CarryScore(
            pair=pair,
            forward_premium=float(fp),
            realised_vol=float(vol),
            vol_adjusted_carry=float(va_carry),
            rank=0,
        ))

    scores.sort(key=lambda s: s.vol_adjusted_carry, reverse=True)
    for rank, s in enumerate(scores, start=1):
        s.rank = rank

    return scores


# ---------------------------------------------------------------------------
# 8. FX momentum (12-1 month cross-sectional ranking)
# ---------------------------------------------------------------------------

def fx_momentum_scores(
    pairs: List[str],
    price_series: np.ndarray,
) -> List[MomentumScore]:
    """
    Compute 12-1 month cross-sectional FX momentum scores.

    Parameters
    ----------
    pairs        : list of currency pair labels (length N)
    price_series : array of shape (T, N) — monthly closing prices.
                   T must be >= 13 to compute 12-1 month momentum.

    Returns
    -------
    List of MomentumScore sorted descending by momentum.
    """
    T_obs, N = price_series.shape
    assert N == len(pairs), "pairs length must match second dimension of price_series"
    assert T_obs >= 13, "Need at least 13 months of data for 12-1 month momentum"

    ret_12m = (price_series[-2] - price_series[-13]) / price_series[-13]   # -13 to -2
    ret_1m  = (price_series[-1] - price_series[-2])  / price_series[-2]    # last month
    momentum = ret_12m - ret_1m   # skip last month (Jegadeesh-Titman)

    scores = []
    for i, pair in enumerate(pairs):
        scores.append(MomentumScore(
            pair=pair,
            return_12m=float(ret_12m[i]),
            return_1m=float(ret_1m[i]),
            momentum=float(momentum[i]),
            rank=0,
        ))

    scores.sort(key=lambda s: s.momentum, reverse=True)
    for rank, s in enumerate(scores, start=1):
        s.rank = rank

    return scores


# ---------------------------------------------------------------------------
# 9. Regime detection per pair (trending vs mean-reverting)
# ---------------------------------------------------------------------------

def _hurst_rs(series: np.ndarray) -> float:
    """Estimate Hurst exponent via rescaled range analysis."""
    n = len(series)
    if n < 20:
        return 0.5
    lags = np.unique(np.logspace(1, np.log10(n // 2), 20).astype(int))
    rs_vals = []
    for lag in lags:
        chunks = [series[i: i + lag] for i in range(0, n - lag + 1, lag)]
        rs_chunk = []
        for chunk in chunks:
            mean_c = np.mean(chunk)
            dev = np.cumsum(chunk - mean_c)
            R = np.max(dev) - np.min(dev)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))

    if len(rs_vals) < 4:
        return 0.5

    lags_used = lags[: len(rs_vals)]
    log_rs = np.log(rs_vals)
    log_lags = np.log(lags_used)
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


def _variance_ratio(series: np.ndarray, q: int = 5) -> float:
    """Lo-MacKinlay variance ratio statistic VR(q)."""
    n = len(series)
    if n < q * 4:
        return 1.0
    rets = np.diff(np.log(series))
    mu = np.mean(rets)
    var1 = np.var(rets, ddof=1)
    rets_q = np.array([np.sum(rets[i: i + q]) for i in range(n - q)])
    var_q = np.var(rets_q, ddof=1) / q
    if var1 < 1e-14:
        return 1.0
    return float(var_q / var1)


def _simple_adf(series: np.ndarray, lags: int = 1) -> float:
    """Minimal ADF test statistic (no critical values — use for relative comparison)."""
    y = np.diff(series)
    x = series[:-1]
    # OLS of dy on y_lag and dy_lags
    T = len(y)
    if T < lags + 5:
        return 0.0
    X = np.column_stack([x[lags:], np.ones(T - lags)])
    dy_main = y[lags:]
    for lag in range(1, lags + 1):
        X = np.column_stack([X, y[lags - lag: T - lag]])
    try:
        beta, res, _, _ = np.linalg.lstsq(X, dy_main, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    resid = dy_main - X @ beta
    se = math.sqrt(np.sum(resid ** 2) / (len(resid) - X.shape[1]))
    xtx_inv = np.linalg.pinv(X.T @ X)
    se_beta0 = se * math.sqrt(xtx_inv[0, 0])
    return float(beta[0] / se_beta0) if se_beta0 > 0 else 0.0


def regime_detect(
    pair: str,
    price_series: np.ndarray,
    vr_q: int = 5,
    adf_threshold: float = -2.86,
) -> RegimeResult:
    """
    Classify a currency pair as trending, mean-reverting, or random walk.

    Uses three complementary tests:
      - Hurst exponent (>0.55 trending, <0.45 mean-reverting)
      - Variance ratio (>1.1 trending, <0.9 mean-reverting)
      - ADF test statistic (< threshold → mean-reverting)

    Parameters
    ----------
    pair         : currency pair label
    price_series : array of prices (at least 50 observations)
    vr_q         : variance ratio horizon
    adf_threshold: ADF critical value at 5% for mean-reversion label

    Returns
    -------
    RegimeResult dataclass.
    """
    h = _hurst_rs(price_series)
    vr = _variance_ratio(price_series, vr_q)
    adf = _simple_adf(price_series)

    trending_votes = 0
    mr_votes = 0

    if h > 0.55:
        trending_votes += 1
    elif h < 0.45:
        mr_votes += 1

    if vr > 1.1:
        trending_votes += 1
    elif vr < 0.9:
        mr_votes += 1

    if adf < adf_threshold:
        mr_votes += 1
    elif adf > -1.5:
        trending_votes += 1

    total = trending_votes + mr_votes
    if total == 0:
        regime = "random_walk"
        confidence = 0.5
    elif trending_votes > mr_votes:
        regime = "trending"
        confidence = trending_votes / 3.0
    elif mr_votes > trending_votes:
        regime = "mean_reverting"
        confidence = mr_votes / 3.0
    else:
        regime = "random_walk"
        confidence = 0.5

    return RegimeResult(
        pair=pair,
        hurst=h,
        variance_ratio=vr,
        adf_stat=adf,
        regime=regime,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 10. Dual-currency bond pricing
# ---------------------------------------------------------------------------

def dual_currency_bond_price(
    face: float,
    coupon_rate: float,
    maturity: float,
    coupon_freq: int,
    r_d: float,
    S: float,
    K: float,
    r_f: float,
    sigma: float,
    option_type: str = "put",
) -> DualCurrencyBond:
    """
    Price a dual-currency bond (DCB): coupons in domestic, principal in foreign.

    The embedded option is an FX option at maturity that gives the issuer the
    right to repay principal in a weaker foreign currency (i.e. a put on the
    foreign currency from the bondholder's perspective).

    DCB Price = Straight Bond PV - Value of Embedded FX Option

    Parameters
    ----------
    face        : face value in domestic currency
    coupon_rate : annual coupon rate (e.g. 0.05 = 5%)
    maturity    : years to maturity
    coupon_freq : coupons per year (e.g. 2 = semi-annual)
    r_d         : domestic discount rate
    S           : current spot FX rate (domestic per foreign)
    K           : strike of embedded option (conversion rate at maturity)
    r_f         : foreign risk-free rate
    sigma       : FX implied vol
    option_type : 'put' (bondholder loses if FX depreciates below K)

    Returns
    -------
    DualCurrencyBond dataclass.
    """
    dt = 1.0 / coupon_freq
    n_periods = int(round(maturity * coupon_freq))
    coupon = face * coupon_rate / coupon_freq

    # Straight bond PV
    pv = 0.0
    for i in range(1, n_periods + 1):
        t = i * dt
        pv += coupon * math.exp(-r_d * t)
    pv += face * math.exp(-r_d * maturity)

    # Embedded FX option (European) at maturity on face/K units of foreign currency
    notional_fx = face / K   # foreign currency notional
    gk = garman_kohlhagen(S, K, maturity, r_d, r_f, sigma, option_type)
    option_value = gk.price * notional_fx

    dcb_price = pv - option_value

    # Implied yield of DCB (solve for y such that PV of cash flows = dcb_price)
    def pv_at_yield(y):
        pv_y = sum(coupon * math.exp(-y * i * dt) for i in range(1, n_periods + 1))
        pv_y += face * math.exp(-y * maturity)
        return pv_y - dcb_price

    try:
        dcb_yield = brentq(pv_at_yield, 0.0001, 0.5)
    except ValueError:
        dcb_yield = r_d

    yield_enhancement = dcb_yield - r_d

    return DualCurrencyBond(
        straight_bond_pv=pv,
        embedded_option_value=option_value,
        dcb_price=dcb_price,
        yield_enhancement=yield_enhancement,
        gk_result=gk,
    )
