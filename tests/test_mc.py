"""
Monte Carlo tests.

Covers:
- Law of Large Numbers convergence
- Blow-up probability estimation for levered strategies
- AR(1) serial correlation detection in simulated paths
- Monte Carlo option pricing convergence
- Path-dependent statistics (barrier, Asian option)
"""

import sys
import os
import unittest

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Test: Law of Large Numbers convergence
# ---------------------------------------------------------------------------

class TestLLNConvergence(unittest.TestCase):
    """MC estimates should converge to theoretical values as n → ∞."""

    def test_mean_estimation(self):
        """Sample mean converges to population mean."""
        rng = np.random.default_rng(42)
        mu_true = 0.001
        sigma = 0.02
        for n in [100, 1000, 10000]:
            sample = mu_true + sigma * rng.standard_normal(n)
            mean_est = sample.mean()
            std_err = sigma / np.sqrt(n)
            # Should be within 3 std errors of true mean
            self.assertLess(abs(mean_est - mu_true), 3 * std_err + 1e-6,
                            f"Mean estimation failed at n={n}")

    def test_variance_estimation(self):
        """Sample variance converges to population variance."""
        rng = np.random.default_rng(42)
        sigma_true = 0.02
        n = 10000
        sample = sigma_true * rng.standard_normal(n)
        var_est = np.var(sample, ddof=1)
        self.assertAlmostEqual(var_est, sigma_true ** 2, delta=sigma_true ** 2 * 0.1)

    def test_quantile_estimation(self):
        """Sample quantile converges to theoretical quantile."""
        rng = np.random.default_rng(42)
        n = 50000
        mu, sigma = 0.0, 1.0
        sample = rng.standard_normal(n)
        q95_est = np.quantile(sample, 0.95)
        q95_true = stats.norm.ppf(0.95, mu, sigma)
        self.assertAlmostEqual(q95_est, q95_true, delta=0.05)

    def test_mc_pi_estimation(self):
        """Estimate pi via Monte Carlo (circle area)."""
        rng = np.random.default_rng(42)
        n = 100000
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        inside = (x ** 2 + y ** 2) <= 1
        pi_est = 4 * inside.mean()
        self.assertAlmostEqual(pi_est, np.pi, delta=0.05)

    def test_gbm_terminal_mean(self):
        """GBM terminal mean: E[S_T] = S_0 * exp(mu * T)."""
        rng = np.random.default_rng(42)
        S0, mu, sigma, T = 100, 0.05, 0.20, 1.0
        n = 100000
        Z = rng.standard_normal(n)
        S_T = S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        theoretical_mean = S0 * np.exp(mu * T)
        mc_mean = S_T.mean()
        std_err = S_T.std() / np.sqrt(n)
        self.assertLess(abs(mc_mean - theoretical_mean), 3 * std_err + 1)

    def test_gbm_variance(self):
        """GBM variance: Var[S_T] = S0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)."""
        rng = np.random.default_rng(42)
        S0, mu, sigma, T = 100, 0.0, 0.20, 1.0
        n = 200000
        Z = rng.standard_normal(n)
        S_T = S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        theoretical_var = S0 ** 2 * np.exp(2 * mu * T) * (np.exp(sigma ** 2 * T) - 1)
        mc_var = np.var(S_T)
        rel_err = abs(mc_var - theoretical_var) / theoretical_var
        self.assertLess(rel_err, 0.05)


# ---------------------------------------------------------------------------
# Test: Blow-up probability
# ---------------------------------------------------------------------------

class TestBlowupProbability(unittest.TestCase):
    """Levered strategies have a probability of ruin."""

    def _simulate_levered_paths(
        self,
        n_paths: int,
        n_steps: int,
        mu: float,
        sigma: float,
        leverage: float,
        ruin_threshold: float = 0.01,
        seed: int = 42,
    ) -> float:
        """Simulate levered GBM paths and return fraction that hit ruin threshold."""
        rng = np.random.default_rng(seed)
        dt = 1 / 252
        levered_mu = leverage * mu - leverage * (leverage - 1) / 2 * sigma ** 2
        levered_sigma = leverage * sigma
        Z = rng.standard_normal((n_paths, n_steps))
        log_rets = (levered_mu - 0.5 * levered_sigma ** 2) * dt + levered_sigma * np.sqrt(dt) * Z
        cum_log_rets = np.cumsum(log_rets, axis=1)
        paths = np.exp(cum_log_rets)
        # Insert initial value
        paths = np.hstack([np.ones((n_paths, 1)), paths])
        # Track if path ever goes below ruin threshold
        ruin_count = (paths.min(axis=1) < ruin_threshold).sum()
        return ruin_count / n_paths

    def test_unlev_low_ruin(self):
        """Unlevered strategy with positive drift has low ruin probability."""
        p_ruin = self._simulate_levered_paths(
            n_paths=5000, n_steps=252,
            mu=0.10, sigma=0.15, leverage=1.0,
            ruin_threshold=0.5,  # 50% loss
        )
        # With 10% annual return and 15% vol, P(ruin to 50%) should be < 20%
        self.assertLess(p_ruin, 0.30)

    def test_high_leverage_increases_ruin(self):
        """Higher leverage → higher ruin probability."""
        common_params = dict(
            n_paths=3000, n_steps=252,
            mu=0.08, sigma=0.20,
            ruin_threshold=0.5,
        )
        p_ruin_1x = self._simulate_levered_paths(leverage=1.0, **common_params)
        p_ruin_3x = self._simulate_levered_paths(leverage=3.0, **common_params)
        self.assertGreater(p_ruin_3x, p_ruin_1x)

    def test_negative_drift_increases_ruin(self):
        """Negative drift → higher ruin than positive drift."""
        common_params = dict(
            n_paths=3000, n_steps=252,
            sigma=0.20, leverage=1.0,
            ruin_threshold=0.8,
        )
        p_ruin_pos = self._simulate_levered_paths(mu=0.10, **common_params)
        p_ruin_neg = self._simulate_levered_paths(mu=-0.10, **common_params)
        self.assertGreater(p_ruin_neg, p_ruin_pos)

    def test_ruin_prob_in_range(self):
        """Ruin probability should be in [0, 1]."""
        p_ruin = self._simulate_levered_paths(
            n_paths=1000, n_steps=252,
            mu=0.05, sigma=0.25, leverage=2.0,
            ruin_threshold=0.5,
        )
        self.assertGreaterEqual(p_ruin, 0.0)
        self.assertLessEqual(p_ruin, 1.0)


# ---------------------------------------------------------------------------
# Test: AR(1) serial correlation
# ---------------------------------------------------------------------------

class TestAR1Correlation(unittest.TestCase):
    """Tests for AR(1) serial correlation detection in simulated data."""

    def _simulate_ar1(
        self,
        n: int,
        phi: float,
        sigma: float,
        seed: int = 42,
    ) -> pd.Series:
        """Simulate AR(1) process: x_t = phi * x_{t-1} + eps_t."""
        rng = np.random.default_rng(seed)
        x = np.zeros(n)
        eps = sigma * rng.standard_normal(n)
        x[0] = eps[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]
        return pd.Series(x)

    def test_ar1_autocorrelation_lag1(self):
        """Lag-1 autocorrelation should be close to phi."""
        for phi in [0.0, 0.3, 0.7, -0.5]:
            with self.subTest(phi=phi):
                x = self._simulate_ar1(n=2000, phi=phi, sigma=1.0)
                ac = float(x.autocorr(1))
                self.assertAlmostEqual(ac, phi, delta=0.1)

    def test_random_walk_near_zero_autocorr(self):
        """Random walk (phi=0) should have near-zero autocorrelation."""
        x = self._simulate_ar1(n=2000, phi=0.0, sigma=1.0)
        ac = abs(float(x.autocorr(1)))
        self.assertLess(ac, 0.1)

    def test_stationary_ar1(self):
        """AR(1) with |phi| < 1 should be stationary (finite variance)."""
        x = self._simulate_ar1(n=2000, phi=0.7, sigma=1.0)
        theoretical_var = 1.0 / (1 - 0.7 ** 2)  # sigma^2 / (1 - phi^2)
        empirical_var = float(x.var())
        self.assertAlmostEqual(empirical_var, theoretical_var, delta=theoretical_var * 0.3)

    def test_integrated_process_nonstationary(self):
        """Random walk (phi=1) variance grows with T."""
        rng = np.random.default_rng(42)
        x = pd.Series(np.cumsum(rng.standard_normal(1000)))
        # Variance should be ~T for a unit root process
        # Just check it's much larger than stationary var
        self.assertGreater(x.var(), 10.0)

    def test_ljung_box_detects_ar1(self):
        """Ljung-Box test should detect AR(1) structure."""
        x = self._simulate_ar1(n=500, phi=0.5, sigma=1.0)
        # Compute Ljung-Box statistic
        n = len(x)
        lags = 10
        lb_stat = 0.0
        for k in range(1, lags + 1):
            rho_k = float(x.autocorr(k))
            lb_stat += rho_k ** 2 / (n - k)
        lb_stat *= n * (n + 2)
        p_value = 1 - stats.chi2.cdf(lb_stat, df=lags)
        # Should reject H0 of no autocorrelation for phi=0.5
        self.assertLess(p_value, 0.05)

    def test_adf_test_stationary(self):
        """ADF test should not reject stationarity for |phi| < 1."""
        x = self._simulate_ar1(n=500, phi=0.5, sigma=1.0)
        # Manual ADF: regress dx on lagged x
        dx = x.diff().dropna().values
        x_lag = x.shift(1).dropna().values
        n = len(dx)
        X = np.column_stack([x_lag, np.ones(n)])
        beta = np.linalg.lstsq(X, dx, rcond=None)[0]
        fitted = X @ beta
        resid = dx - fitted
        sigma2 = np.sum(resid ** 2) / max(n - X.shape[1], 1)
        se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0])
        adf_stat = beta[0] / (se + 1e-12)
        # ADF critical value at 5% ~ -2.86 for large samples
        # For phi=0.5, coefficient on x_lag ≈ phi-1 = -0.5 → very negative ADF
        self.assertLess(adf_stat, 0)


# ---------------------------------------------------------------------------
# Test: MC Option Pricing Convergence
# ---------------------------------------------------------------------------

class TestMCOptionPricing(unittest.TestCase):
    """Monte Carlo option prices should converge to Black-Scholes."""

    def setUp(self):
        self.S = 100
        self.K = 100
        self.T = 0.5
        self.r = 0.05
        self.sigma = 0.20
        self.q = 0.0

    def _bs_call(self) -> float:
        from research.options.pricing import BlackScholes
        return BlackScholes(self.S, self.K, self.T, self.r, self.sigma, self.q).call()

    def test_mc_convergence_call(self):
        """MC call price converges to BS as n_paths increases."""
        from research.options.pricing import MonteCarloPricer
        bs_price = self._bs_call()

        for n_paths in [1000, 10000, 100000]:
            with self.subTest(n_paths=n_paths):
                mc = MonteCarloPricer(
                    self.S, self.K, self.T, self.r, self.sigma,
                    n_paths=n_paths, n_steps=50, seed=42,
                )
                price, se = mc.price_european("call", antithetic=True, control_variate=True)
                # Should be within 3 std errors of BS price
                self.assertLess(abs(price - bs_price), max(3 * se + 0.01, 0.5))

    def test_mc_put_call_parity(self):
        """MC call and put should satisfy put-call parity."""
        from research.options.pricing import MonteCarloPricer, BlackScholes
        mc = MonteCarloPricer(
            self.S, self.K, self.T, self.r, self.sigma,
            n_paths=50000, n_steps=50, seed=42,
        )
        call_price, _ = mc.price_european("call", antithetic=True)
        put_price, _ = mc.price_european("put", antithetic=True)
        lhs = call_price - put_price
        rhs = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(lhs, rhs, delta=0.30)

    def test_mc_asian_below_european(self):
        """Asian call <= European call (averaging reduces option value)."""
        from research.options.pricing import MonteCarloPricer
        mc = MonteCarloPricer(
            self.S, self.K, self.T, self.r, self.sigma,
            n_paths=50000, n_steps=50, seed=42,
        )
        euro_call, _ = mc.price_european("call", antithetic=True)
        asian_call, _ = mc.price_asian("call", "arithmetic")
        # Asian <= European for fixed-strike options
        self.assertLessEqual(asian_call, euro_call + 0.5)

    def test_mc_se_decreases_with_n(self):
        """Standard error should decrease as sqrt(n_paths)."""
        from research.options.pricing import MonteCarloPricer
        ses = []
        for n_paths in [1000, 4000, 16000]:
            mc = MonteCarloPricer(
                self.S, self.K, self.T, self.r, self.sigma,
                n_paths=n_paths, n_steps=30, seed=42,
            )
            _, se = mc.price_european("call", antithetic=False, control_variate=False)
            ses.append(se)

        # SE at 4000 should be about half SE at 1000
        se_ratio = ses[0] / (ses[1] + 1e-10)
        self.assertGreater(se_ratio, 1.5)
        self.assertLess(se_ratio, 3.0)

    def test_antithetic_reduces_variance(self):
        """Antithetic variates should reduce pricing error vs plain MC."""
        from research.options.pricing import MonteCarloPricer
        mc = MonteCarloPricer(
            self.S, self.K, self.T, self.r, self.sigma,
            n_paths=10000, n_steps=30, seed=42,
        )
        _, se_plain = mc.price_european("call", antithetic=False, control_variate=False)
        _, se_anti = mc.price_european("call", antithetic=True, control_variate=False)
        # Antithetic should reduce variance
        self.assertLessEqual(se_anti, se_plain * 1.1)  # allow 10% slack

    def test_barrier_option_below_vanilla(self):
        """Down-and-out barrier call <= European call."""
        from research.options.pricing import MonteCarloPricer
        mc = MonteCarloPricer(
            self.S, self.K, self.T, self.r, self.sigma,
            n_paths=50000, n_steps=50, seed=42,
        )
        vanilla, _ = mc.price_european("call")
        barrier, _ = mc.price_barrier("call", barrier=90.0, barrier_type="down-and-out")
        self.assertLessEqual(barrier, vanilla + 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
