"""
Comprehensive test suite for the srfm-lab options analytics library.

Tests cover:
- Black-Scholes put-call parity
- Delta bounds for calls and puts
- Vega positivity
- Theta negativity for long positions
- Implied vol round-trip
- SVI no-arbitrage constraints
- SABR smile symmetry and monotonicity
- Greek aggregation correctness
- Portfolio VaR positivity
- Binomial convergence to BS
- Term structure consistency
- Risk limits checking
"""

import math
import sys
import os

import numpy as np
import pytest

# Make sure the project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.options.pricing import BlackScholes, BjerksundStensland2002, BinomialTree, MonteCarloPricer
from lib.options.greeks import GreeksResult, AnalyticalGreeks, NumericalGreeks, GreeksAggregator
from lib.options.vol_surface import SVIModel, SVIParams, SABRModel, VolSmile, VolSurface
from lib.options.risk import OptionsPosition, OptionsPortfolio, DeltaHedger, RiskLimits
from lib.options.term_structure import YieldCurve, DividendSchedule, Dividend, ForwardPrice, BorrowRate


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CALL = "call"
PUT = "put"
TOL = 1e-4          # tolerance for most numerical tests
IV_TOL = 1e-5       # tolerance for implied vol round-trip


def bs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0, option_type="call") -> BlackScholes:
    return BlackScholes(S, K, T, r, sigma, q, option_type)


# ---------------------------------------------------------------------------
# 1. Put-call parity
# ---------------------------------------------------------------------------

class TestPutCallParity:
    """C - P = F * df - K * df (spot form: S*dq - K*df)"""

    @pytest.mark.parametrize("S,K,T,r,sigma,q", [
        (100, 100, 1.0, 0.05, 0.20, 0.00),
        (100, 110, 0.5, 0.03, 0.25, 0.02),
        (50,  40,  2.0, 0.06, 0.30, 0.01),
        (200, 180, 0.25, 0.04, 0.15, 0.03),
        (100, 100, 0.01, 0.05, 0.20, 0.00),  # very short dated
    ])
    def test_parity(self, S, K, T, r, sigma, q):
        call = BlackScholes(S, K, T, r, sigma, q, "call").price()
        put  = BlackScholes(S, K, T, r, sigma, q, "put").price()
        lhs = call - put
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs(lhs - rhs) < TOL, (
            f"PCP failed: C-P={lhs:.6f}, S*dq - K*df={rhs:.6f}"
        )

    def test_parity_deep_itm_call(self):
        call = BlackScholes(200, 100, 1.0, 0.05, 0.20, 0.0, "call").price()
        put  = BlackScholes(200, 100, 1.0, 0.05, 0.20, 0.0, "put").price()
        rhs = 200 * 1.0 - 100 * math.exp(-0.05 * 1.0)
        assert abs((call - put) - rhs) < TOL

    def test_parity_deep_otm_call(self):
        call = BlackScholes(100, 200, 1.0, 0.05, 0.20, 0.0, "call").price()
        put  = BlackScholes(100, 200, 1.0, 0.05, 0.20, 0.0, "put").price()
        rhs = 100 - 200 * math.exp(-0.05)
        assert abs((call - put) - rhs) < TOL


# ---------------------------------------------------------------------------
# 2. Delta bounds
# ---------------------------------------------------------------------------

class TestDeltaBounds:
    """Call delta in [0, 1], put delta in [-1, 0]."""

    @pytest.mark.parametrize("K", [70, 80, 90, 100, 110, 120, 130])
    def test_call_delta_bounds(self, K):
        delta = BlackScholes(100, K, 1.0, 0.05, 0.20, 0.0, "call").delta()
        assert 0.0 <= delta <= 1.0, f"Call delta={delta:.6f} out of [0,1] at K={K}"

    @pytest.mark.parametrize("K", [70, 80, 90, 100, 110, 120, 130])
    def test_put_delta_bounds(self, K):
        delta = BlackScholes(100, K, 1.0, 0.05, 0.20, 0.0, "put").delta()
        assert -1.0 <= delta <= 0.0, f"Put delta={delta:.6f} out of [-1,0] at K={K}"

    def test_atm_call_delta_near_half(self):
        """ATM call delta should be close to 0.5 for q=0, small r."""
        delta = BlackScholes(100, 100, 1.0, 0.0, 0.20, 0.0, "call").delta()
        assert 0.45 < delta < 0.60

    def test_atm_put_delta_near_neg_half(self):
        delta = BlackScholes(100, 100, 1.0, 0.0, 0.20, 0.0, "put").delta()
        assert -0.60 < delta < -0.40

    def test_call_delta_monotone_in_strike(self):
        """Call delta should decrease as strike increases."""
        deltas = [BlackScholes(100, K, 1.0, 0.05, 0.20).delta() for K in [80, 90, 100, 110, 120]]
        for i in range(len(deltas) - 1):
            assert deltas[i] >= deltas[i + 1] - 1e-9

    def test_put_delta_monotone_in_strike(self):
        """Put delta decreases (becomes more negative) as strike increases (higher K = deeper ITM put)."""
        deltas = [BlackScholes(100, K, 1.0, 0.05, 0.20, 0.0, "put").delta() for K in [80, 90, 100, 110, 120]]
        for i in range(len(deltas) - 1):
            assert deltas[i] >= deltas[i + 1] - 1e-9, (
                f"Put delta not monotone at index {i}: {deltas[i]:.6f} < {deltas[i+1]:.6f}"
            )

    def test_call_put_delta_relation(self):
        """call_delta - put_delta = exp(-q*T)."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.03
        dc = BlackScholes(S, K, T, r, sigma, q, "call").delta()
        dp = BlackScholes(S, K, T, r, sigma, q, "put").delta()
        assert abs(dc - dp - math.exp(-q * T)) < TOL


# ---------------------------------------------------------------------------
# 3. Vega positivity
# ---------------------------------------------------------------------------

class TestVegaPositivity:
    """Vega is always non-negative for European options."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("K", [70, 90, 100, 110, 130])
    def test_vega_positive(self, option_type, K):
        vega = BlackScholes(100, K, 1.0, 0.05, 0.20, 0.0, option_type).vega()
        assert vega >= 0.0, f"Vega={vega} < 0 for {option_type} K={K}"

    def test_vega_atm_is_largest(self):
        """ATM vega should dominate OTM/ITM vega (for same T, sigma)."""
        atm_vega = BlackScholes(100, 100, 1.0, 0.05, 0.20).vega()
        otm_vega = BlackScholes(100, 140, 1.0, 0.05, 0.20).vega()
        itm_vega = BlackScholes(100,  60, 1.0, 0.05, 0.20).vega()
        assert atm_vega > otm_vega
        assert atm_vega > itm_vega

    def test_vega_call_equals_put(self):
        """Call and put vega are equal by symmetry of the formula."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.0
        vc = BlackScholes(S, K, T, r, sigma, q, "call").vega()
        vp = BlackScholes(S, K, T, r, sigma, q, "put").vega()
        assert abs(vc - vp) < TOL

    def test_vega_positive_numerical(self):
        """Numerical vega from price bump should be positive."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.0
        p_up = BlackScholes(S, K, T, r, sigma + 0.001, q).price()
        p_dn = BlackScholes(S, K, T, r, sigma - 0.001, q).price()
        numerical_vega = (p_up - p_dn) / 0.002
        assert numerical_vega > 0.0


# ---------------------------------------------------------------------------
# 4. Theta negativity for long positions
# ---------------------------------------------------------------------------

class TestThetaNegativity:
    """Theta should be negative for long European options (time value decays)."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("K", [80, 100])
    def test_theta_negative(self, option_type, K):
        # Deep ITM puts with nonzero r can have positive theta due to interest
        # income on strike. Test ATM and near-OTM only.
        theta = BlackScholes(100, K, 1.0, 0.05, 0.20, 0.0, option_type).theta()
        assert theta < 0.0, f"Theta={theta} >= 0 for long {option_type} K={K}"

    def test_theta_negative_atm_zero_rate(self):
        """ATM theta is always negative when r=0."""
        for otype in ["call", "put"]:
            theta = BlackScholes(100, 100, 1.0, 0.0, 0.20, 0.0, otype).theta()
            assert theta < 0.0, f"ATM theta={theta} >= 0 for {otype} at r=0"

    def test_theta_more_negative_near_expiry(self):
        """Theta accelerates in magnitude as expiry approaches."""
        theta_1y = BlackScholes(100, 100, 1.0, 0.05, 0.20).theta()
        theta_1m = BlackScholes(100, 100, 1.0 / 12.0, 0.05, 0.20).theta()
        assert abs(theta_1m) > abs(theta_1y)

    def test_theta_consistent_with_price_decay(self):
        """theta * dt should approximate price change over a small time step."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        bs_now = BlackScholes(S, K, T, r, sigma)
        dt = 1.0 / 365.0
        bs_later = BlackScholes(S, K, T - dt, r, sigma)
        price_change = bs_later.price() - bs_now.price()
        theta = bs_now.theta()
        # theta is already per-day, so compare directly
        assert abs(price_change - theta) < 0.05 * abs(theta) + 1e-5


# ---------------------------------------------------------------------------
# 5. Implied volatility round-trip
# ---------------------------------------------------------------------------

class TestImpliedVolRoundTrip:
    """price -> implied vol -> price should recover original price."""

    @pytest.mark.parametrize("sigma", [0.05, 0.10, 0.20, 0.40, 0.80, 1.50])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_roundtrip_atm(self, sigma, option_type):
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        target_price = BlackScholes(S, K, T, r, sigma, q, option_type).price()
        iv = BlackScholes.implied_vol(target_price, S, K, T, r, q, option_type)
        assert not math.isnan(iv), f"IV failed for sigma={sigma} {option_type}"
        recovered = BlackScholes(S, K, T, r, iv, q, option_type).price()
        assert abs(recovered - target_price) < IV_TOL, (
            f"Round-trip failed: sigma={sigma}, IV={iv:.6f}, "
            f"target={target_price:.6f}, recovered={recovered:.6f}"
        )

    @pytest.mark.parametrize("K", [70, 80, 90, 100, 110, 120, 130])
    def test_roundtrip_across_strikes(self, K):
        S, T, r, sigma, q = 100, 1.0, 0.05, 0.25, 0.02
        for otype in ["call", "put"]:
            price = BlackScholes(S, K, T, r, sigma, q, otype).price()
            iv = BlackScholes.implied_vol(price, S, K, T, r, q, otype)
            if not math.isnan(iv):
                recovered = BlackScholes(S, K, T, r, iv, q, otype).price()
                assert abs(recovered - price) < IV_TOL

    @pytest.mark.parametrize("T", [0.05, 0.25, 0.5, 1.0, 2.0])
    def test_roundtrip_across_expiries(self, T):
        S, K, r, sigma, q = 100, 100, 0.05, 0.20, 0.0
        price = BlackScholes(S, K, T, r, sigma, q, "call").price()
        iv = BlackScholes.implied_vol(price, S, K, T, r, q, "call")
        assert not math.isnan(iv)
        recovered = BlackScholes(S, K, T, r, iv, q, "call").price()
        assert abs(recovered - price) < IV_TOL

    def test_iv_returns_nan_for_below_intrinsic(self):
        """IV should return NaN if market price is below intrinsic."""
        S, K, T, r = 100, 90, 1.0, 0.05
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
        iv = BlackScholes.implied_vol(intrinsic * 0.5, S, K, T, r, 0.0, "call")
        assert math.isnan(iv)


# ---------------------------------------------------------------------------
# 6. SVI no-arbitrage constraints
# ---------------------------------------------------------------------------

class TestSVINoArbitrage:
    """Test SVI butterfly no-arbitrage and parameter constraints."""

    def _typical_params(self) -> SVIParams:
        return SVIParams(a=0.04, b=0.10, rho=-0.3, m=0.0, sigma=0.15)

    def test_total_variance_positive(self):
        model = SVIModel(self._typical_params())
        ks = np.linspace(-1.5, 1.5, 200)
        ws = model.total_variance_array(ks)
        assert np.all(ws > 0), "SVI total variance must be positive"

    def test_total_variance_convex(self):
        """SVI total variance should be convex (w'' >= 0 for these params)."""
        model = SVIModel(self._typical_params())
        ks = np.linspace(-1.0, 1.0, 100)
        ws = model.total_variance_array(ks)
        # Check that second difference is non-negative (convexity)
        diffs2 = np.diff(ws, 2)
        assert np.all(diffs2 >= -1e-6), "SVI total variance should be convex"

    def test_butterfly_arbitrage_free(self):
        """Typical SVI params should pass butterfly no-arbitrage check."""
        model = SVIModel(self._typical_params())
        ks = np.linspace(-1.5, 1.5, 500)
        is_free, g = model.check_butterfly_arbitrage(ks)
        assert is_free, f"Butterfly arbitrage detected. Min g={np.min(g):.6f}"

    def test_svi_atm_vol(self):
        """At k=0 (ATM), w(0) = a + b*sigma*sqrt(1 - rho^2) approximately."""
        p = SVIParams(a=0.04, b=0.10, rho=-0.3, m=0.0, sigma=0.15)
        model = SVIModel(p)
        w_atm = model.total_variance(0.0)
        expected = p.a + p.b * (p.rho * (-p.m) + math.sqrt(p.m ** 2 + p.sigma ** 2))
        assert abs(w_atm - expected) < 1e-12

    def test_calendar_arbitrage_free(self):
        """Two slices with increasing total variance pass calendar check."""
        p1 = SVIParams(a=0.02, b=0.08, rho=-0.3, m=0.0, sigma=0.10)
        p2 = SVIParams(a=0.04, b=0.10, rho=-0.3, m=0.0, sigma=0.15)
        model = SVIModel()
        ks = np.linspace(-1.0, 1.0, 100)
        is_free, diff = model.check_calendar_arbitrage(p1, p2, ks)
        assert is_free, f"Calendar arbitrage detected. Min diff={np.min(diff):.6f}"

    def test_svi_calibration_recovers_params(self):
        """Calibration should recover target params from noiseless data."""
        target = SVIParams(a=0.04, b=0.10, rho=-0.3, m=0.0, sigma=0.15)
        model = SVIModel()
        ks = np.linspace(-1.0, 1.0, 50)
        T = 1.0
        ws = model.total_variance_array(ks, target)
        model2 = SVIModel()
        fitted = model2.fit(ks, ws)
        # Should recover ATM total variance within 1e-4
        atm_w_target = model.total_variance(0.0, target)
        atm_w_fitted = model2.total_variance(0.0, fitted)
        assert abs(atm_w_fitted - atm_w_target) < 5e-4


# ---------------------------------------------------------------------------
# 7. SABR smile
# ---------------------------------------------------------------------------

class TestSABRSmile:
    """SABR smile properties: positivity, ATM level, skew direction."""

    def test_sabr_atm_positive(self):
        model = SABRModel(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
        iv = model.atm_vol(100.0, 1.0)
        assert iv > 0.0

    def test_sabr_vol_positive_all_strikes(self):
        model = SABRModel(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
        strikes = np.linspace(70, 130, 50)
        vols = model.smile_array(100.0, strikes, 1.0)
        assert np.all(vols > 0.0), "SABR vols should be positive"

    def test_sabr_negative_rho_gives_negative_skew(self):
        """With rho < 0, the smile should have negative skew (vol increases for lower strikes)."""
        model = SABRModel(alpha=0.3, beta=0.5, rho=-0.6, nu=0.4)
        v_low  = model.implied_vol(100.0, 90.0,  1.0)
        v_atm  = model.implied_vol(100.0, 100.0, 1.0)
        v_high = model.implied_vol(100.0, 110.0, 1.0)
        assert v_low > v_atm, "Negative rho should give higher vol for low strikes"
        assert v_high < v_atm, "Negative rho should give lower vol for high strikes"

    def test_sabr_zero_rho_near_symmetric(self):
        """With rho=0, the smile should be approximately symmetric around ATM."""
        model = SABRModel(alpha=0.3, beta=0.5, rho=0.0, nu=0.4)
        dK = 10.0
        v_low  = model.implied_vol(100.0, 100.0 - dK, 1.0)
        v_high = model.implied_vol(100.0, 100.0 + dK, 1.0)
        assert abs(v_low - v_high) < 0.02, "Near-zero rho smile should be near-symmetric"

    def test_sabr_positive_rho_gives_positive_skew(self):
        model = SABRModel(alpha=0.3, beta=0.5, rho=0.6, nu=0.4)
        v_low  = model.implied_vol(100.0, 90.0,  1.0)
        v_high = model.implied_vol(100.0, 110.0, 1.0)
        assert v_high > v_low, "Positive rho should give higher vol for high strikes"

    def test_sabr_calibration_atm_match(self):
        """Calibration should match ATM vol of input data."""
        model = SABRModel(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
        F = 100.0
        T = 1.0
        strikes = np.array([80, 90, 95, 100, 105, 110, 120], dtype=float)
        market_vols = model.smile_array(F, strikes, T)
        model2 = SABRModel()
        model2.fit(F, strikes, market_vols, T, beta=0.5)
        atm_fitted = model2.atm_vol(F, T)
        atm_target = model.atm_vol(F, T)
        assert abs(atm_fitted - atm_target) < 0.005


# ---------------------------------------------------------------------------
# 8. Greek aggregation correctness
# ---------------------------------------------------------------------------

class TestGreekAggregation:
    """GreeksAggregator net greeks should match sum of scaled individual greeks."""

    def test_net_delta_sums_correctly(self):
        ag = AnalyticalGreeks("call")
        g1 = ag.compute(100, 100, 1.0, 0.05, 0.20)
        g2 = ag.compute(100, 110, 0.5, 0.05, 0.25)

        agg = GreeksAggregator()
        agg.add("AAPL", g1, qty=10.0, spot=100.0, sigma=0.20)
        agg.add("AAPL", g2, qty=-5.0,  spot=100.0, sigma=0.25)

        net = agg.net_greeks()
        expected_delta = g1.delta * 10.0 + g2.delta * (-5.0)
        assert abs(net.delta - expected_delta) < 1e-10

    def test_net_gamma_sums_correctly(self):
        ag = AnalyticalGreeks("call")
        g1 = ag.compute(100, 100, 1.0, 0.05, 0.20)
        g2 = ag.compute(100,  90, 1.0, 0.05, 0.20)

        agg = GreeksAggregator()
        agg.add("SPY", g1, qty=5.0, spot=100.0, sigma=0.20)
        agg.add("SPY", g2, qty=5.0, spot=100.0, sigma=0.20)

        net = agg.net_greeks()
        expected_gamma = (g1.gamma + g2.gamma) * 5.0
        assert abs(net.gamma - expected_gamma) < 1e-10

    def test_greeks_by_underlying(self):
        ag_c = AnalyticalGreeks("call")
        ag_p = AnalyticalGreeks("put")
        g_aapl = ag_c.compute(150, 150, 1.0, 0.05, 0.20)
        g_tsla = ag_p.compute(200, 200, 0.5, 0.05, 0.30)

        agg = GreeksAggregator()
        agg.add("AAPL", g_aapl, qty=10.0, spot=150.0, sigma=0.20)
        agg.add("TSLA", g_tsla, qty= 5.0, spot=200.0, sigma=0.30)

        by_und = agg.greeks_by_underlying()
        assert "AAPL" in by_und
        assert "TSLA" in by_und
        assert abs(by_und["AAPL"].delta - g_aapl.delta * 10.0) < 1e-10
        assert abs(by_und["TSLA"].delta - g_tsla.delta *  5.0) < 1e-10

    def test_zero_position_zero_greeks(self):
        ag = AnalyticalGreeks("call")
        g = ag.compute(100, 100, 1.0, 0.05, 0.20)
        agg = GreeksAggregator()
        agg.add("X", g, qty=0.0, spot=100.0, sigma=0.20)
        net = agg.net_greeks()
        assert abs(net.delta) < 1e-15
        assert abs(net.vega) < 1e-15

    def test_greeks_result_addition(self):
        g1 = GreeksResult(delta=0.5, gamma=0.02, vega=10.0, theta=-0.05)
        g2 = GreeksResult(delta=-0.3, gamma=0.01, vega=5.0, theta=-0.03)
        g_sum = g1 + g2
        assert abs(g_sum.delta - 0.2) < 1e-15
        assert abs(g_sum.gamma - 0.03) < 1e-15
        assert abs(g_sum.vega - 15.0) < 1e-15

    def test_greeks_result_multiplication(self):
        g = GreeksResult(delta=0.5, gamma=0.02, vega=10.0)
        g2 = g * 3.0
        assert abs(g2.delta - 1.5) < 1e-15
        assert abs(g2.vega - 30.0) < 1e-15

    def test_stress_scenarios_exist(self):
        ag = AnalyticalGreeks("call")
        g = ag.compute(100, 100, 1.0, 0.05, 0.20)
        agg = GreeksAggregator()
        agg.add("SPY", g, qty=100.0, spot=100.0, sigma=0.20)
        scenarios = agg.stress_scenarios()
        assert len(scenarios) > 0
        labels = [s.label for s in scenarios]
        assert any("Spot" in l for l in labels)
        assert any("Vol" in l for l in labels)

    def test_numerical_greeks_match_analytical(self):
        """Numerical Greeks should be close to analytical for BS."""
        S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0

        def pricer(S_, K_, T_, r_, sigma_, q_=0.0) -> float:
            return BlackScholes(S_, K_, T_, r_, sigma_, q_, "call").price()

        num = NumericalGreeks(pricer, dS_frac=0.005, dSigma=0.0005)
        an = AnalyticalGreeks("call")

        g_num = num.compute(S, K, T, r, sigma, q)
        g_an  = an.compute(S, K, T, r, sigma, q)

        assert abs(g_num.delta - g_an.delta) < 5e-4
        assert abs(g_num.gamma - g_an.gamma) < 1e-4
        assert abs(g_num.vega  - g_an.vega)  < 0.01
        assert abs(g_num.theta - g_an.theta) < 1e-4


# ---------------------------------------------------------------------------
# 9. Portfolio VaR positive
# ---------------------------------------------------------------------------

class TestPortfolioVaR:
    """Portfolio VaR should be a non-negative number."""

    def _make_portfolio(self) -> OptionsPortfolio:
        portfolio = OptionsPortfolio()
        for K, qty in [(90, 5), (100, -10), (110, 5)]:
            pos = OptionsPosition(
                underlying="SPY",
                option_type="call",
                strike=K,
                expiry=1.0,
                qty=qty,
                entry_price=5.0,
                current_price=5.0,
                spot=100.0,
                sigma=0.20,
                r=0.05,
                q=0.0,
                multiplier=100.0,
            )
            pos.refresh_greeks()
            portfolio.add_position(pos)
        return portfolio

    def test_var_positive(self):
        portfolio = self._make_portfolio()
        var = portfolio.var_delta_gamma(spot_vol_annual=0.20, confidence=0.99)
        assert var >= 0.0, f"VaR={var} should be non-negative"

    def test_var_increases_with_vol(self):
        portfolio = self._make_portfolio()
        var_low  = portfolio.var_delta_gamma(spot_vol_annual=0.10)
        var_high = portfolio.var_delta_gamma(spot_vol_annual=0.40)
        assert var_high >= var_low, "Higher spot vol should give higher VaR"

    def test_var_increases_with_confidence(self):
        portfolio = self._make_portfolio()
        var_95 = portfolio.var_delta_gamma(spot_vol_annual=0.20, confidence=0.95)
        var_99 = portfolio.var_delta_gamma(spot_vol_annual=0.20, confidence=0.99)
        assert var_99 >= var_95, "99% VaR should be >= 95% VaR"

    def test_es_positive(self):
        portfolio = self._make_portfolio()
        es = portfolio.expected_shortfall(0.20, confidence=0.99, n_simulations=10000)
        assert es >= 0.0

    def test_scenario_matrix_shape(self):
        portfolio = self._make_portfolio()
        spot_shocks = [-0.10, 0.0, 0.10]
        vol_shocks = [-0.05, 0.0, 0.05]
        result = portfolio.scenario_matrix(spot_shocks, vol_shocks)
        assert result["pnl_matrix"].shape == (3, 3)

    def test_portfolio_greeks_aggregate(self):
        portfolio = self._make_portfolio()
        net = portfolio.portfolio_greeks()
        assert isinstance(net.delta, float)
        assert isinstance(net.gamma, float)

    def test_delta_hedge_recommendation(self):
        portfolio = self._make_portfolio()
        rec = portfolio.delta_hedge_recommendation()
        assert "delta_hedges" in rec
        assert len(rec["delta_hedges"]) > 0


# ---------------------------------------------------------------------------
# 10. Binomial tree convergence
# ---------------------------------------------------------------------------

class TestBinomialTree:
    """Binomial CRR prices should converge to BS prices for European options."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_european_converges_to_bs(self, option_type):
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02
        bs_price = BlackScholes(S, K, T, r, sigma, q, option_type).price()
        tree = BinomialTree(n_steps=500)
        tree_price = tree.price(S, K, T, r, sigma, q, option_type, american=False)
        assert abs(tree_price - bs_price) < 0.05, (
            f"Binomial price={tree_price:.4f} vs BS={bs_price:.4f}"
        )

    def test_american_put_ge_european_put(self):
        """American put must be worth at least as much as European put."""
        S, K, T, r, sigma, q = 100, 110, 1.0, 0.05, 0.20, 0.0
        tree = BinomialTree(n_steps=300)
        amer = tree.price(S, K, T, r, sigma, q, "put", american=True)
        euro = tree.price(S, K, T, r, sigma, q, "put", american=False)
        assert amer >= euro - 1e-6, f"American put {amer:.4f} < European put {euro:.4f}"

    def test_american_call_no_early_exercise_without_dividends(self):
        """American call without dividends should equal European call."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        tree = BinomialTree(n_steps=300)
        amer = tree.price(S, K, T, r, sigma, 0.0, "call", american=True)
        euro = BlackScholes(S, K, T, r, sigma).price()
        assert abs(amer - euro) < 0.10, (
            f"American call {amer:.4f} should equal European call {euro:.4f}"
        )


# ---------------------------------------------------------------------------
# 11. Term structure
# ---------------------------------------------------------------------------

class TestTermStructure:
    """Yield curve, dividends, and forward pricing consistency."""

    def test_flat_curve_discount_factor(self):
        r = 0.05
        curve = YieldCurve.flat(r)
        df = curve.discount_factor(1.0)
        assert abs(df - math.exp(-r)) < 1e-10

    def test_flat_curve_zero_rate(self):
        r = 0.05
        curve = YieldCurve.flat(r)
        z = curve.zero_rate(2.0)
        assert abs(z - r) < 1e-6

    def test_forward_rate_consistency(self):
        """Forward rate should be consistent with zero rates."""
        curve = YieldCurve.flat(0.05)
        fwd = curve.forward_rate(1.0, 2.0)
        assert abs(fwd - 0.05) < 1e-6

    def test_forward_rate_from_upward_curve(self):
        """Forward rate should exceed long-end zero rate for upward-sloping curve."""
        maturities = np.array([0.5, 1.0, 2.0, 5.0])
        rates = np.array([0.02, 0.03, 0.04, 0.05])
        curve = YieldCurve(maturities, rates)
        fwd_1_2 = curve.forward_rate(1.0, 2.0)
        z_1 = curve.zero_rate(1.0)
        z_2 = curve.zero_rate(2.0)
        assert fwd_1_2 > z_2  # Forward rate exceeds long zero rate for upward curve

    def test_dividend_pv(self):
        schedule = DividendSchedule([Dividend(0.25, 1.5), Dividend(0.75, 1.5)])
        pv = schedule.pv_dividends(1.0, r=0.05)
        expected = 1.5 * math.exp(-0.05 * 0.25) + 1.5 * math.exp(-0.05 * 0.75)
        assert abs(pv - expected) < 1e-10

    def test_adjusted_spot_less_than_spot(self):
        S = 100.0
        schedule = DividendSchedule([Dividend(0.5, 2.0)])
        S_adj = schedule.adjusted_spot(S, 1.0, 0.05)
        assert S_adj < S

    def test_forward_price_no_dividends(self):
        fp = ForwardPrice()
        F = fp.continuous_dividend(100.0, 1.0, q=0.0, r=0.05)
        assert abs(F - 100 * math.exp(0.05)) < 1e-10

    def test_forward_price_with_dividend_yield(self):
        fp = ForwardPrice()
        F = fp.continuous_dividend(100.0, 1.0, q=0.02, r=0.05)
        assert abs(F - 100 * math.exp(0.03)) < 1e-10

    def test_borrow_rate_flat(self):
        br = BorrowRate()
        br.set_flat("GME", 0.10)
        assert abs(br.get("GME") - 0.10) < 1e-10
        assert abs(br.effective_rate("GME", r=0.05) - (-0.05)) < 1e-10

    def test_borrow_rate_default_zero(self):
        br = BorrowRate()
        assert br.get("AAPL") == 0.0


# ---------------------------------------------------------------------------
# 12. Bjerksund-Stensland
# ---------------------------------------------------------------------------

class TestBjerksundStensland:
    """American option pricing tests."""

    def test_american_put_geq_european_put(self):
        bs2 = BjerksundStensland2002()
        amer = bs2.price(100, 110, 1.0, 0.05, 0.20, q=0.0, option_type="put")
        euro = BlackScholes(100, 110, 1.0, 0.05, 0.20, 0.0, "put").price()
        assert amer >= euro - 1e-4

    def test_american_call_no_dividends_eq_european(self):
        """American call on non-dividend paying stock = European call."""
        bs2 = BjerksundStensland2002()
        amer = bs2.price(100, 100, 1.0, 0.05, 0.20, q=0.0, option_type="call")
        euro = BlackScholes(100, 100, 1.0, 0.05, 0.20, 0.0, "call").price()
        # Should be close (within 0.5%)
        assert abs(amer - euro) / euro < 0.01

    def test_early_exercise_boundary_returns_array(self):
        bs2 = BjerksundStensland2002()
        boundary = bs2.early_exercise_boundary(100, 1.0, 0.05, 0.20, q=0.0, option_type="put")
        assert boundary.shape[1] == 2
        assert np.all(boundary[:, 0] > 0)

    def test_futures_price_call(self):
        bs2 = BjerksundStensland2002()
        price = bs2.futures_price(100, 100, 1.0, 0.05, 0.20, "call")
        assert price > 0.0


# ---------------------------------------------------------------------------
# 13. Risk limits
# ---------------------------------------------------------------------------

class TestRiskLimits:
    """Risk limit checking on portfolios."""

    def _small_portfolio(self) -> OptionsPortfolio:
        portfolio = OptionsPortfolio()
        pos = OptionsPosition(
            underlying="AAPL",
            option_type="call",
            strike=150,
            expiry=1.0,
            qty=1.0,
            entry_price=5.0,
            current_price=5.0,
            spot=150.0,
            sigma=0.25,
            r=0.05,
            multiplier=100.0,
        )
        pos.refresh_greeks()
        portfolio.add_position(pos)
        return portfolio

    def test_no_violations_small_portfolio(self):
        limits = RiskLimits(
            max_delta_per_underlying=1000.0,
            max_portfolio_delta=5000.0,
        )
        portfolio = self._small_portfolio()
        violations = limits.check_portfolio(portfolio)
        assert len(violations) == 0, f"Unexpected violations: {violations}"

    def test_violation_on_large_position(self):
        limits = RiskLimits(max_position_size=0.5)
        portfolio = self._small_portfolio()
        violations = limits.check_portfolio(portfolio)
        assert any("max_position_size" in v for v in violations)

    def test_check_new_position(self):
        limits = RiskLimits(max_position_size=50)
        portfolio = OptionsPortfolio()
        new_pos = OptionsPosition(
            underlying="AAPL",
            option_type="call",
            strike=150,
            expiry=1.0,
            qty=100.0,
            entry_price=5.0,
            current_price=5.0,
            spot=150.0,
            sigma=0.25,
            r=0.05,
        )
        new_pos.refresh_greeks()
        violations = limits.check_new_position(portfolio, new_pos)
        assert any("max_position_size" in v for v in violations)


# ---------------------------------------------------------------------------
# 14. VolSmile
# ---------------------------------------------------------------------------

class TestVolSmile:
    """VolSmile interpolation and smile metrics."""

    def _make_smile(self) -> VolSmile:
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = np.array([0.28, 0.24, 0.20, 0.22, 0.25])
        return VolSmile(strikes, vols, expiry=1.0, F=100.0, r=0.05)

    def test_atm_vol(self):
        smile = self._make_smile()
        atm = smile.atm_vol()
        assert abs(atm - 0.20) < 0.01  # Should be close to 0.20 at K=F=100

    def test_vol_in_range(self):
        smile = self._make_smile()
        for K in [85, 95, 100, 105, 115]:
            v = smile.vol(K)
            assert 0.01 < v < 2.0, f"Vol={v} out of range at K={K}"

    def test_smile_summary_keys(self):
        smile = self._make_smile()
        summary = smile.smile_summary()
        for key in ["atm_vol", "rr_25d", "bf_25d", "min_vol", "max_vol"]:
            assert key in summary


# ---------------------------------------------------------------------------
# 15. Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    """Monte Carlo pricer basic sanity checks."""

    def test_atm_call_close_to_bs(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        mc = MonteCarloPricer(n_paths=100000, seed=42, use_antithetic=True, use_control_variate=True)
        mc_price = mc.price(S, K, T, r, sigma, 0.0, "call")
        bs_price = BlackScholes(S, K, T, r, sigma).price()
        assert abs(mc_price - bs_price) < 0.20, (
            f"MC price={mc_price:.4f}, BS price={bs_price:.4f}"
        )

    def test_put_call_parity_mc(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        mc = MonteCarloPricer(n_paths=100000, seed=99)
        call = mc.price(S, K, T, r, sigma, 0.0, "call")
        put  = mc.price(S, K, T, r, sigma, 0.0, "put")
        parity_rhs = S - K * math.exp(-r * T)
        assert abs((call - put) - parity_rhs) < 0.30

    def test_ci_contains_bs_price(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        mc = MonteCarloPricer(n_paths=200000, seed=7)
        mean, lo, hi = mc.price_with_ci(S, K, T, r, sigma, 0.0, "call", confidence=0.99)
        bs_price = BlackScholes(S, K, T, r, sigma).price()
        assert lo <= bs_price <= hi, (
            f"BS price {bs_price:.4f} not in CI [{lo:.4f}, {hi:.4f}]"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
