"""
Statistical arbitrage hypothesis templates.

Implements:
  - Classic pairs trading (cointegration-based)
  - Basket/index arbitrage
  - ETF-NAV arbitrage
  - Cross-exchange arbitrage
  - Triangular arbitrage (FX/crypto)
  - Vol surface arbitrage (calendar spread, butterfly)
  - Dividend arbitrage
  - Convertible bond arbitrage
  - Merger arbitrage
  - Capital structure arbitrage
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Pairs Trade ───────────────────────────────────────────────────────────────

@dataclass
class PairsTradeSetup:
    """A fully parameterized pairs trade."""
    asset_a: str
    asset_b: str
    hedge_ratio: float          # shares of B per share of A
    spread_mean: float          # historical mean of spread
    spread_std: float           # historical std of spread
    current_spread: float
    z_score: float
    half_life_days: float       # mean reversion half-life
    cointegration_pvalue: float
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_threshold: float = 3.5


def pairs_zscore(
    price_a: np.ndarray,
    price_b: np.ndarray,
    hedge_ratio: float,
    lookback: int = 63,
) -> dict:
    """
    Compute rolling z-score of a pairs spread.
    spread = log(A) - hedge_ratio * log(B)
    """
    log_a = np.log(np.maximum(price_a, 1e-10))
    log_b = np.log(np.maximum(price_b, 1e-10))
    spread = log_a - hedge_ratio * log_b

    n = len(spread)
    z_scores = np.zeros(n)
    for i in range(lookback, n):
        window = spread[i - lookback: i]
        mu = float(window.mean())
        sigma = float(window.std() + 1e-10)
        z_scores[i] = (spread[i] - mu) / sigma

    current_z = float(z_scores[-1])
    spread_mean = float(spread[-lookback:].mean())
    spread_std = float(spread[-lookback:].std())

    return {
        "z_scores": z_scores,
        "current_z": current_z,
        "spread": spread,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "current_spread": float(spread[-1]),
    }


def dynamic_hedge_ratio(
    price_a: np.ndarray,
    price_b: np.ndarray,
    method: str = "ols_rolling",
    window: int = 63,
) -> np.ndarray:
    """
    Dynamic hedge ratio estimation.
    methods: 'ols_rolling', 'kalman', 'tls' (total least squares)
    """
    n = len(price_a)
    log_a = np.log(np.maximum(price_a, 1e-10))
    log_b = np.log(np.maximum(price_b, 1e-10))
    ratios = np.zeros(n)

    if method == "ols_rolling":
        for i in range(window, n):
            x = log_b[i - window: i]
            y = log_a[i - window: i]
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            denom = float(np.dot(x_centered, x_centered))
            if denom > 1e-10:
                ratios[i] = float(np.dot(x_centered, y_centered) / denom)
            else:
                ratios[i] = 1.0
        ratios[:window] = ratios[window]

    elif method == "kalman":
        # Simple Kalman filter hedge ratio
        beta = 1.0
        P = 1.0
        Q = 1e-5   # process noise
        R = 1e-3   # observation noise
        for i in range(n):
            # Predict
            P_pred = P + Q
            # Update
            h = log_b[i]
            innov = log_a[i] - beta * h
            S = P_pred * h**2 + R
            K = P_pred * h / max(S, 1e-10)
            beta += K * innov
            P = (1 - K * h) * P_pred
            ratios[i] = beta

    elif method == "tls":
        # Total Least Squares (orthogonal regression)
        for i in range(window, n):
            x = log_b[i - window: i]
            y = log_a[i - window: i]
            X = np.stack([x - x.mean(), y - y.mean()], axis=1)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            # Direction of minimum variance
            v = Vt[-1]
            if abs(v[0]) > 1e-10:
                ratios[i] = float(-v[1] / v[0])
            else:
                ratios[i] = 1.0
        ratios[:window] = ratios[window]

    return ratios


def pairs_half_life(spread: np.ndarray) -> float:
    """
    Estimate mean reversion half-life via OU-process regression.
    d(spread) = kappa * (mu - spread) * dt + sigma * dW
    """
    if len(spread) < 5:
        return float("inf")

    y = np.diff(spread)
    x = spread[:-1] - spread.mean()

    if x.std() < 1e-10:
        return float("inf")

    # OLS: y = a * x + b + eps, kappa = -a
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom < 1e-10:
        return float("inf")
    kappa_hat = -float(np.dot(x_centered, y - y.mean()) / denom)

    if kappa_hat <= 0:
        return float("inf")

    return float(math.log(2) / kappa_hat)


# ── Basket Arbitrage ──────────────────────────────────────────────────────────

def basket_nav_spread(
    basket_prices: np.ndarray,   # (n_assets,) current prices
    basket_weights: np.ndarray,  # (n_assets,) weights (sum to 1)
    etf_price: float,
) -> dict:
    """
    ETF-NAV arbitrage: compare ETF price to weighted basket NAV.
    Premium = (ETF price - NAV) / NAV
    """
    nav = float(np.dot(basket_weights, basket_prices))
    premium_bps = float((etf_price - nav) / max(nav, 1e-10) * 10000)
    fair_value_discount = float(nav - etf_price)

    return {
        "nav": nav,
        "etf_price": etf_price,
        "premium_bps": premium_bps,
        "fair_value_discount": fair_value_discount,
        "arb_direction": "buy_etf_sell_basket" if premium_bps < -5 else
                          "buy_basket_sell_etf" if premium_bps > 5 else "neutral",
        "is_actionable": bool(abs(premium_bps) > 5),
    }


def index_arb_signal(
    futures_price: float,
    spot_index: float,
    risk_free_rate: float,  # annualized
    dividend_yield: float,  # annualized
    days_to_expiry: int,
    transaction_cost_bps: float = 3.0,
) -> dict:
    """
    Index futures arbitrage: compare futures to fair value.
    Fair value = Spot * exp((r - d) * T)
    """
    T = days_to_expiry / 365.0
    fair_value = spot_index * math.exp((risk_free_rate - dividend_yield) * T)
    basis = futures_price - fair_value
    basis_bps = float(basis / max(spot_index, 1e-10) * 10000)

    net_bps = abs(basis_bps) - transaction_cost_bps * 2
    actionable = bool(net_bps > 1.0)

    return {
        "fair_value": float(fair_value),
        "basis": float(basis),
        "basis_bps": float(basis_bps),
        "net_arb_bps": float(net_bps),
        "direction": "buy_futures_sell_spot" if basis < 0 else "buy_spot_sell_futures",
        "is_actionable": actionable,
        "annualized_return_pct": float(net_bps / max(T * 365, 1) * 100 / 10000) if actionable else 0.0,
    }


# ── Triangular Arbitrage ──────────────────────────────────────────────────────

def triangular_arb_check(
    rate_ab: float,   # A/B exchange rate
    rate_bc: float,   # B/C exchange rate
    rate_ac: float,   # A/C exchange rate
    transaction_cost_bps: float = 2.0,
) -> dict:
    """
    Detect triangular arbitrage opportunities in FX/crypto.
    Implied A/C = A/B * B/C
    """
    implied_ac = rate_ab * rate_bc
    deviation_pct = float((implied_ac - rate_ac) / max(rate_ac, 1e-10) * 100)
    deviation_bps = float(deviation_pct * 100)

    # Round-trip: start with 1 A, buy B, buy C, sell for A
    round_trip_gain = rate_ab * rate_bc / rate_ac - 1
    cost = transaction_cost_bps * 3 / 10000  # 3 transactions
    net_gain = float(round_trip_gain - cost)

    return {
        "implied_rate_ac": float(implied_ac),
        "actual_rate_ac": float(rate_ac),
        "deviation_bps": float(deviation_bps),
        "round_trip_gain": float(round_trip_gain),
        "net_gain_after_costs": net_gain,
        "is_profitable": bool(net_gain > 0),
        "direction": "forward" if round_trip_gain > 0 else "reverse",
    }


# ── Volatility Arbitrage ──────────────────────────────────────────────────────

def calendar_spread_arb(
    front_iv: float,       # implied vol of front month
    back_iv: float,        # implied vol of back month
    front_days: int,
    back_days: int,
    realized_vol_21d: float,
    transaction_cost_bps: float = 5.0,
) -> dict:
    """
    Calendar spread vol arbitrage: trade term structure discrepancies.
    Fair value: back vol should account for forward vol.
    """
    T1 = front_days / 365
    T2 = back_days / 365

    # Forward variance = total variance delta
    front_var = front_iv**2 * T1
    back_var = back_iv**2 * T2
    forward_var = back_var - front_var
    forward_vol = float(math.sqrt(max(forward_var, 0) / max(T2 - T1, 1e-6)))

    # Implied vs realized comparison for front month
    front_vega_pnl_estimate = float((front_iv - realized_vol_21d) * 100)  # in vol points

    # Term structure slope (bps/day)
    slope = float((back_iv - front_iv) / max(back_days - front_days, 1) * 252)

    # Signal
    if back_iv > front_iv * 1.1:
        trade = "sell_back_buy_front"
    elif front_iv > back_iv * 1.05:
        trade = "buy_back_sell_front"
    else:
        trade = "neutral"

    return {
        "front_iv": float(front_iv),
        "back_iv": float(back_iv),
        "forward_vol": float(forward_vol),
        "term_structure_slope": float(slope),
        "front_vega_opportunity_bps": float(front_vega_pnl_estimate * 100),
        "recommended_trade": trade,
        "is_actionable": bool(abs(back_iv - front_iv) * 100 > transaction_cost_bps / 100),
    }


def vol_surface_butterfly(
    put_25d_iv: float,
    atm_iv: float,
    call_25d_iv: float,
) -> dict:
    """
    Butterfly (kurtosis) spread: deviation from lognormal distribution.
    Fly = (put_25d + call_25d) / 2 - atm
    """
    fly = (put_25d_iv + call_25d_iv) / 2 - atm_iv
    risk_reversal = call_25d_iv - put_25d_iv

    # Fair value fly (lognormal): small positive
    # Elevated fly = fat tails in market = protection premium
    fly_z = float(fly / max(atm_iv, 1e-10) * 100)  # fly as % of ATM vol

    return {
        "butterfly": float(fly),
        "risk_reversal": float(risk_reversal),
        "butterfly_pct_atm": float(fly_z),
        "skew_direction": "put_skew" if risk_reversal < 0 else "call_skew",
        "kurtosis_premium": float(max(fly, 0) / max(atm_iv, 1e-10)),
    }


# ── Merger Arbitrage ──────────────────────────────────────────────────────────

@dataclass
class MergerArbSetup:
    """Merger arbitrage trade parameters."""
    target_ticker: str
    acquirer_ticker: str
    deal_price: float               # per share cash consideration
    current_target_price: float
    deal_type: str                  # cash / stock / mixed
    expected_close_date_days: int
    break_probability: float        # estimated deal break probability
    regulatory_risk: str            # low/medium/high
    acquirer_ratio: float = 0.0     # shares of acquirer per target share (stock deal)


def merger_arb_analysis(setup: MergerArbSetup) -> dict:
    """
    Full merger arbitrage analysis.
    Expected return = spread * (1 - P_break) + downside * P_break
    """
    spread = setup.deal_price - setup.current_target_price
    gross_spread_pct = float(spread / max(setup.current_target_price, 1e-10) * 100)

    T = setup.expected_close_date_days / 365.0
    annualized_return = float(gross_spread_pct / max(T, 1/365) if T > 0 else 0)

    # Downside on break: typically 20-30% drop back to pre-announcement price
    break_downside_pct = -20.0  # conservative assumption
    expected_return = float(
        gross_spread_pct * (1 - setup.break_probability)
        + break_downside_pct * setup.break_probability
    )

    # Sharpe-like: expected return / uncertainty
    vol_estimate = float(setup.break_probability * abs(break_downside_pct - gross_spread_pct) * 0.5)
    sharpe_estimate = float(expected_return / max(vol_estimate, 1e-10))

    return {
        "gross_spread_pct": float(gross_spread_pct),
        "annualized_gross_pct": float(annualized_return),
        "expected_return_pct": float(expected_return),
        "vol_estimate_pct": float(vol_estimate),
        "sharpe_estimate": float(sharpe_estimate),
        "break_probability": float(setup.break_probability),
        "is_favorable": bool(expected_return > 2.0 and sharpe_estimate > 1.0),
        "days_to_close": setup.expected_close_date_days,
    }


# ── Capital Structure Arbitrage ───────────────────────────────────────────────

def capital_structure_arb(
    equity_price: float,
    credit_spread_bps: float,   # CDS or bond spread
    implied_vol: float,         # equity option vol
    debt_face_value: float,
    risk_free_rate: float,
    time_to_maturity_years: float = 5.0,
) -> dict:
    """
    Capital structure arbitrage: Merton model implied credit vs market credit.
    Equity = call option on firm assets.
    """
    # Rough asset value: equity + debt
    asset_value_est = equity_price + debt_face_value * math.exp(-risk_free_rate * time_to_maturity_years)

    # Merton: d1 = (ln(V/D) + (r + sigma^2/2)*T) / (sigma * sqrt(T))
    if asset_value_est <= 0 or implied_vol <= 0 or time_to_maturity_years <= 0:
        return {"model_credit_spread_bps": 0.0, "arb_opportunity_bps": 0.0}

    d1 = (math.log(asset_value_est / max(debt_face_value, 1e-10))
          + (risk_free_rate + implied_vol**2 / 2) * time_to_maturity_years) / \
         (implied_vol * math.sqrt(time_to_maturity_years))
    d2 = d1 - implied_vol * math.sqrt(time_to_maturity_years)

    # Risk-neutral probability of default
    from math import erf
    def ncdf(x):
        return 0.5 * (1 + erf(x / math.sqrt(2)))

    p_default = float(1 - ncdf(d2))

    # Model credit spread (continuous): -ln(1-PD) / T
    model_spread_bps = float(-math.log(max(1 - p_default, 1e-6)) / time_to_maturity_years * 10000)

    # Opportunity: market spread vs model spread
    arb_bps = float(credit_spread_bps - model_spread_bps)

    return {
        "model_credit_spread_bps": float(model_spread_bps),
        "market_credit_spread_bps": float(credit_spread_bps),
        "arb_opportunity_bps": float(arb_bps),
        "p_default": float(p_default),
        "direction": "short_credit_long_equity" if arb_bps > 50 else
                     "long_credit_short_equity" if arb_bps < -50 else "neutral",
        "is_actionable": bool(abs(arb_bps) > 50),
    }


# ── Convertible Bond Arbitrage ────────────────────────────────────────────────

def convertible_bond_arb(
    cb_price: float,           # convertible bond price (% of par)
    stock_price: float,
    conversion_ratio: float,   # shares per bond
    par_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    risk_free_rate: float,
    credit_spread_bps: float,
    implied_vol: float,
) -> dict:
    """
    Convertible bond arbitrage: model CB value vs market price.
    CB value = straight bond + embedded call option on stock.
    """
    # Straight bond value (simplified)
    discount_rate = risk_free_rate + credit_spread_bps / 10000
    T = years_to_maturity
    n_periods = max(int(T * 2), 1)  # semi-annual
    dt = T / n_periods
    coupon = coupon_rate * par_value / 2  # semi-annual coupon

    pv_coupons = sum(coupon * math.exp(-discount_rate * (i + 1) * dt)
                     for i in range(n_periods))
    pv_par = par_value * math.exp(-discount_rate * T)
    straight_bond_value = float((pv_coupons + pv_par) / par_value * 100)

    # Conversion value
    conversion_value = float(stock_price * conversion_ratio / par_value * 100)

    # Intrinsic value
    intrinsic = max(straight_bond_value, conversion_value)

    # Option value (Black-Scholes approximation)
    if years_to_maturity > 0 and implied_vol > 0:
        d1 = (math.log(stock_price / max(par_value / conversion_ratio, 1e-10))
              + (risk_free_rate + implied_vol**2 / 2) * T) / (implied_vol * math.sqrt(T))
        d2 = d1 - implied_vol * math.sqrt(T)

        def ncdf(x):
            from math import erf
            return 0.5 * (1 + erf(x / math.sqrt(2)))

        call_option = float(
            stock_price * ncdf(d1) - (par_value / conversion_ratio) * math.exp(-risk_free_rate * T) * ncdf(d2)
        )
        option_component = call_option * conversion_ratio / par_value * 100
    else:
        option_component = 0.0

    model_value = float(straight_bond_value + option_component)
    premium = float(cb_price - model_value)
    premium_pct = float(premium / max(model_value, 1e-10) * 100)

    return {
        "model_value_pct": float(model_value),
        "market_price_pct": float(cb_price),
        "straight_bond_value_pct": float(straight_bond_value),
        "conversion_value_pct": float(conversion_value),
        "option_component_pct": float(option_component),
        "premium_pct": float(premium_pct),
        "is_cheap": bool(premium_pct < -3),
        "is_rich": bool(premium_pct > 5),
    }
