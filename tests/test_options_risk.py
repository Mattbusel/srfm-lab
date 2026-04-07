"""
test_options_risk.py — Tests for Black-Scholes options pricing and VaR/CVaR.

~500 LOC. Standalone implementations (no external dependency on lib/risk.py).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sp_stats
from scipy.optimize import brentq

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "lib"))


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes implementation
# ─────────────────────────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(0.0, S - K)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * sp_stats.norm.cdf(d1) - K * math.exp(-r * T) * sp_stats.norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(0.0, K - S)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * sp_stats.norm.cdf(-d2) - S * sp_stats.norm.cdf(-d1)


def bs_delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(sp_stats.norm.cdf(d1))


def bs_delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(sp_stats.norm.cdf(d1) - 1.0)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(sp_stats.norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(S * sp_stats.norm.pdf(d1) * math.sqrt(T))


def bs_theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term1 = -(S * sp_stats.norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    term2 = -r * K * math.exp(-r * T) * sp_stats.norm.cdf(d2)
    return float(term1 + term2)


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = "call") -> float:
    fn = bs_call if option_type == "call" else bs_put
    try:
        iv = brentq(
            lambda sigma: fn(S, K, T, r, sigma) - market_price,
            1e-6, 10.0, xtol=1e-8, maxiter=200
        )
        return float(iv)
    except ValueError:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# VaR / CVaR
# ─────────────────────────────────────────────────────────────────────────────

def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    return float(-np.percentile(returns, (1.0 - confidence) * 100))


def historical_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    var = historical_var(returns, confidence)
    losses = returns[returns < -var + 1e-10]
    if len(losses) == 0:
        return var
    return float(-np.mean(losses))


def parametric_var(mu: float, sigma: float, confidence: float = 0.95) -> float:
    z = sp_stats.norm.ppf(1.0 - confidence)
    return float(-(mu + z * sigma))


def kupiec_test(n_obs: int, n_exc: int, confidence: float = 0.95) -> dict:
    p = 1.0 - confidence
    p_hat = n_exc / max(1, n_obs)
    if p_hat <= 0 or p_hat >= 1:
        return {"statistic": float("nan"), "p_value": 1.0, "reject_h0": False}
    lr = -2.0 * (
        n_exc * math.log(p / p_hat) + (n_obs - n_exc) * math.log((1 - p) / (1 - p_hat))
    )
    p_value = float(1.0 - sp_stats.chi2.cdf(lr, df=1))
    return {"statistic": lr, "p_value": p_value, "reject_h0": p_value < 0.05}


# ─────────────────────────────────────────────────────────────────────────────
# Class TestBlackScholes
# ─────────────────────────────────────────────────────────────────────────────

class TestBlackScholes:

    S = 100.0; K = 100.0; T = 1.0; r = 0.05; sigma = 0.20

    def test_put_call_parity(self):
        """C - P = S - K*e^(-rT)."""
        C   = bs_call(self.S, self.K, self.T, self.r, self.sigma)
        P   = bs_put( self.S, self.K, self.T, self.r, self.sigma)
        lhs = C - P
        rhs = self.S - self.K * math.exp(-self.r * self.T)
        assert abs(lhs - rhs) < 1e-8, f"PCP violated: {lhs:.8f} != {rhs:.8f}"

    def test_at_the_money_delta_near_half(self):
        """ATM call delta ≈ 0.5."""
        delta = bs_delta_call(self.S, self.K, self.T, self.r, self.sigma)
        assert 0.50 <= delta <= 0.65

    def test_put_delta_near_neg_half_atm(self):
        """ATM put delta between -0.65 and -0.35 (shifts toward -0.4 when r>0)."""
        delta = bs_delta_put(self.S, self.K, self.T, self.r, self.sigma)
        assert -0.65 <= delta <= -0.35

    def test_gamma_positive_always(self):
        """Gamma > 0 for all strikes and maturities."""
        for S in (80.0, 100.0, 120.0):
            for T in (0.1, 0.5, 1.0):
                g = bs_gamma(S, self.K, T, self.r, self.sigma)
                assert g > 0.0, f"gamma={g} for S={S},T={T}"

    def test_vega_positive_for_options(self):
        """Vega > 0."""
        v = bs_vega(self.S, self.K, self.T, self.r, self.sigma)
        assert v > 0.0

    def test_theta_negative_for_long(self):
        """Call theta < 0 (long option decays)."""
        th = bs_theta_call(self.S, self.K, self.T, self.r, self.sigma)
        assert th < 0.0

    def test_implied_vol_roundtrip(self):
        """BS_price(IV) == market_price."""
        true_sigma   = 0.25
        market_price = bs_call(self.S, self.K, self.T, self.r, true_sigma)
        iv = implied_vol(market_price, self.S, self.K, self.T, self.r, "call")
        assert math.isfinite(iv)
        assert abs(iv - true_sigma) < 1e-5, f"IV roundtrip: {iv:.6f} != {true_sigma:.6f}"

    def test_implied_vol_monotone_in_price(self):
        """Higher call price → higher IV."""
        prices = [5.0, 10.0, 15.0, 20.0]
        ivs = [implied_vol(p, self.S, self.K, self.T, self.r, "call") for p in prices]
        valid = [(p, iv) for p, iv in zip(prices, ivs) if math.isfinite(iv)]
        for i in range(1, len(valid)):
            assert valid[i][1] >= valid[i-1][1]

    def test_deep_itm_call_near_intrinsic(self):
        S_itm = 160.0
        C = bs_call(S_itm, self.K, self.T, self.r, self.sigma)
        intrinsic = S_itm - self.K * math.exp(-self.r * self.T)
        assert abs(C - intrinsic) < 10.0

    def test_deep_otm_call_near_zero(self):
        S_otm = 50.0
        C = bs_call(S_otm, self.K, self.T, self.r, self.sigma)
        assert C < 2.0

    def test_call_price_increases_with_sigma(self):
        sigmas = [0.10, 0.20, 0.30, 0.40]
        prices = [bs_call(self.S, self.K, self.T, self.r, s) for s in sigmas]
        for i in range(1, len(prices)):
            assert prices[i] > prices[i-1]

    def test_delta_bounded_call(self):
        """Call delta in (0,1)."""
        for S in (60.0, 80.0, 100.0, 120.0, 140.0):
            d = bs_delta_call(S, self.K, self.T, self.r, self.sigma)
            assert 0.0 <= d <= 1.0

    def test_delta_bounded_put(self):
        """Put delta in (-1,0)."""
        for S in (60.0, 80.0, 100.0, 120.0, 140.0):
            d = bs_delta_put(S, self.K, self.T, self.r, self.sigma)
            assert -1.0 <= d <= 0.0

    def test_put_call_parity_many_strikes(self):
        for K in (80.0, 90.0, 100.0, 110.0, 120.0):
            C = bs_call(self.S, K, self.T, self.r, self.sigma)
            P = bs_put( self.S, K, self.T, self.r, self.sigma)
            parity = C - P - (self.S - K * math.exp(-self.r * self.T))
            assert abs(parity) < 1e-7, f"K={K}: parity={parity:.2e}"

    def test_bs_call_nonnegative(self):
        """Call price should always be >= 0."""
        for S in (50.0, 100.0, 150.0):
            for K in (80.0, 100.0, 120.0):
                for T in (0.1, 1.0, 2.0):
                    C = bs_call(S, K, T, self.r, self.sigma)
                    assert C >= 0.0

    def test_bs_put_nonnegative(self):
        """Put price should always be >= 0."""
        for S in (50.0, 100.0, 150.0):
            for K in (80.0, 100.0, 120.0):
                P = bs_put(S, K, self.T, self.r, self.sigma)
                assert P >= 0.0

    def test_call_equals_put_atm_zero_rate(self):
        """At zero rate, ATM call == ATM put by symmetry."""
        C = bs_call(100.0, 100.0, 1.0, 0.0, 0.20)
        P = bs_put( 100.0, 100.0, 1.0, 0.0, 0.20)
        assert abs(C - P) < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Class TestVaR
# ─────────────────────────────────────────────────────────────────────────────

class TestVaR:

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(42)
        self.normal_returns = rng.normal(0.001, 0.02, 2_000)

    def test_historical_var_from_normal(self):
        """95% VaR from N(0.001, 0.02) should be ≈ 0.032."""
        var = historical_var(self.normal_returns, confidence=0.95)
        # theoretical: -(0.001 - 1.645*0.02) ≈ 0.0319
        assert 0.02 < var < 0.05, f"VaR {var:.4f} not in expected range"

    def test_var_decreases_with_confidence(self):
        """Higher confidence → larger VaR."""
        var_90 = historical_var(self.normal_returns, 0.90)
        var_95 = historical_var(self.normal_returns, 0.95)
        var_99 = historical_var(self.normal_returns, 0.99)
        assert var_90 <= var_95 <= var_99

    def test_cvar_greater_than_var(self):
        """CVaR >= VaR."""
        var  = historical_var(self.normal_returns, 0.95)
        cvar = historical_cvar(self.normal_returns, 0.95)
        assert cvar >= var - 1e-10

    def test_backtest_kupiec_test_size(self):
        """Well-calibrated VaR: don't reject H0."""
        n_obs = 500; n_exc = int(n_obs * 0.05)
        result = kupiec_test(n_obs, n_exc, confidence=0.95)
        assert result["p_value"] > 0.01

    def test_kupiec_rejects_bad_model(self):
        """20% exception rate should be rejected."""
        result = kupiec_test(500, 100, confidence=0.95)
        assert result["reject_h0"] is True

    def test_var_increases_with_vol(self):
        rng = np.random.default_rng(1)
        low  = rng.normal(0.0, 0.01, 1_000)
        high = rng.normal(0.0, 0.04, 1_000)
        assert historical_var(high, 0.95) > historical_var(low, 0.95)

    def test_var_finite(self):
        var = historical_var(self.normal_returns, 0.95)
        assert math.isfinite(var)

    def test_cvar_extreme_losses(self):
        """CVaR captures tail risk beyond VaR."""
        rng = np.random.default_rng(7)
        rets = rng.normal(0.0, 0.02, 1_000)
        var  = historical_var(rets, 0.95)
        cvar = historical_cvar(rets, 0.95)
        assert cvar >= var

    def test_parametric_var_formula(self):
        """Parametric VaR for N(0, 0.02): -(0 + z*0.02) where z=1.645."""
        z   = sp_stats.norm.ppf(0.05)  # -1.645
        mu  = 0.0
        sig = 0.02
        var = parametric_var(mu, sig, 0.95)
        expected = -(mu + z * sig)
        assert abs(var - expected) < 1e-10

    def test_kupiec_fields_present(self):
        result = kupiec_test(250, 12, 0.95)
        assert all(k in result for k in ("statistic", "p_value", "reject_h0"))
        assert isinstance(result["reject_h0"], bool)
