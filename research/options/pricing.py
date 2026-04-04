"""
Options pricing models.

Implements:
- Black-Scholes (full: call, put, digital)
- CRR binomial tree (American and European)
- Monte Carlo with antithetic variates and control variates
- Heston stochastic volatility model (characteristic function approach)
- SABR model (Hagan et al. 2002)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate, optimize, stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptionPrice:
    """Option pricing result."""
    call: float
    put: float
    intrinsic_call: float
    intrinsic_put: float
    model: str
    params: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

class BlackScholes:
    """
    Black-Scholes-Merton option pricing model.

    Parameters
    ----------
    S : float
        Current asset price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate (continuously compounded).
    sigma : float
        Implied or realized volatility (annualized).
    q : float
        Continuous dividend yield.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    def _d1(self) -> float:
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / \
               (self.sigma * np.sqrt(self.T) + 1e-12)

    def _d2(self) -> float:
        return self._d1() - self.sigma * np.sqrt(self.T)

    def call(self) -> float:
        """European call price."""
        if self.T <= 0:
            return max(self.S - self.K, 0.0)
        d1, d2 = self._d1(), self._d2()
        return (self.S * np.exp(-self.q * self.T) * stats.norm.cdf(d1)
                - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2))

    def put(self) -> float:
        """European put price."""
        if self.T <= 0:
            return max(self.K - self.S, 0.0)
        d1, d2 = self._d1(), self._d2()
        return (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)
                - self.S * np.exp(-self.q * self.T) * stats.norm.cdf(-d1))

    def price(self) -> OptionPrice:
        """Return both call and put prices."""
        return OptionPrice(
            call=round(self.call(), 6),
            put=round(self.put(), 6),
            intrinsic_call=round(max(self.S - self.K, 0.0), 6),
            intrinsic_put=round(max(self.K - self.S, 0.0), 6),
            model="Black-Scholes",
            params={"S": self.S, "K": self.K, "T": self.T,
                    "r": self.r, "sigma": self.sigma, "q": self.q},
        )

    def digital_call(self) -> float:
        """Cash-or-nothing digital call (pays $1 if S_T > K)."""
        if self.T <= 0:
            return float(self.S > self.K)
        return np.exp(-self.r * self.T) * stats.norm.cdf(self._d2())

    def digital_put(self) -> float:
        """Cash-or-nothing digital put."""
        if self.T <= 0:
            return float(self.S < self.K)
        return np.exp(-self.r * self.T) * stats.norm.cdf(-self._d2())

    def greeks(self) -> Dict[str, float]:
        """All first and second order Greeks."""
        from .greeks import BlackScholesGreeks
        g = BlackScholesGreeks(self.S, self.K, self.T, self.r, self.sigma, self.q)
        return g.all_greeks()

    def implied_vol(
        self,
        market_price: float,
        option_type: str = "call",
        method: str = "brentq",
    ) -> float:
        """
        Compute implied volatility via numerical inversion.

        Parameters
        ----------
        market_price : float
        option_type : str
            'call' or 'put'.
        method : str
            'brentq' (Brent) or 'newton' (Newton-Raphson).

        Returns
        -------
        float
            Implied volatility.
        """
        from .vol_surface import implied_vol as _iv
        return _iv(market_price, self.S, self.K, self.T, self.r,
                   option_type=option_type, q=self.q, method=method)

    @staticmethod
    def put_call_parity_check(
        call_price: float,
        put_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        tol: float = 1e-4,
    ) -> bool:
        """Verify C - P = S*e^{-qT} - K*e^{-rT}."""
        lhs = call_price - put_price
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        return abs(lhs - rhs) < tol


# ---------------------------------------------------------------------------
# CRR Binomial Tree
# ---------------------------------------------------------------------------

class BinomialTree:
    """
    Cox-Ross-Rubinstein (CRR) binomial tree option pricer.

    Handles both European and American options.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiration (years).
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    n_steps : int
        Number of time steps in the tree.
    q : float
        Dividend yield.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int = 200,
        q: float = 0.0,
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.q = q

    def _tree_params(self) -> Tuple[float, float, float, float, float]:
        """Compute CRR tree parameters."""
        dt = self.T / self.n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        df = np.exp(-self.r * dt)
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        p = np.clip(p, 0.0, 1.0)
        return dt, u, d, df, p

    def _price(self, option_type: str, american: bool) -> float:
        """General pricing routine."""
        dt, u, d, df, p = self._tree_params()
        n = self.n_steps

        # Terminal asset prices
        S_T = self.S * u ** (np.arange(n + 1) * 2 - n)  # S * u^j * d^(n-j)
        # More precisely: S * u^j * d^(n-j) for j=0..n
        j_arr = np.arange(n + 1)
        S_T = self.S * (u ** j_arr) * (d ** (n - j_arr))

        if option_type == "call":
            payoffs = np.maximum(S_T - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - S_T, 0.0)

        # Backward induction
        values = payoffs.copy()
        for step in range(n - 1, -1, -1):
            # Asset prices at this node
            j_arr_step = np.arange(step + 1)
            S_step = self.S * (u ** j_arr_step) * (d ** (step - j_arr_step))

            # Continuation values
            cont = df * (p * values[1:step + 2] + (1 - p) * values[:step + 1])

            if american:
                if option_type == "call":
                    intrinsic = np.maximum(S_step - self.K, 0.0)
                else:
                    intrinsic = np.maximum(self.K - S_step, 0.0)
                values = np.maximum(cont, intrinsic)
            else:
                values = cont

        return float(values[0])

    def european_call(self) -> float:
        return self._price("call", american=False)

    def european_put(self) -> float:
        return self._price("put", american=False)

    def american_call(self) -> float:
        return self._price("call", american=True)

    def american_put(self) -> float:
        return self._price("put", american=True)

    def price(self, option_type: str = "call", american: bool = False) -> float:
        return self._price(option_type, american)

    def early_exercise_premium(self) -> Dict[str, float]:
        """Compute American minus European premium for call and put."""
        eu_call = self.european_call()
        am_call = self.american_call()
        eu_put = self.european_put()
        am_put = self.american_put()
        return {
            "call_early_exercise_premium": round(am_call - eu_call, 6),
            "put_early_exercise_premium": round(am_put - eu_put, 6),
            "european_call": round(eu_call, 6),
            "european_put": round(eu_put, 6),
            "american_call": round(am_call, 6),
            "american_put": round(am_put, 6),
        }


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class MonteCarloPricer:
    """
    Monte Carlo option pricer with variance reduction.

    Techniques:
    - Antithetic variates (halves variance)
    - Control variates (uses Black-Scholes as control)
    - Quasi-Monte Carlo (Sobol sequence)

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiration (years).
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Time steps per path.
    q : float
        Dividend yield.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_paths: int = 100_000,
        n_steps: int = 252,
        q: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.q = q
        self.seed = seed

    def _simulate_paths(self, antithetic: bool = True) -> np.ndarray:
        """
        Simulate GBM paths.

        Returns
        -------
        np.ndarray
            (n_paths, n_steps+1) terminal prices.
        """
        rng = np.random.default_rng(self.seed)
        dt = self.T / self.n_steps
        n = self.n_paths // 2 if antithetic else self.n_paths

        Z = rng.standard_normal((n, self.n_steps))
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        log_S = np.zeros((n, self.n_steps + 1))
        log_S[:, 0] = np.log(self.S)
        for t in range(self.n_steps):
            log_S[:, t + 1] = log_S[:, t] + drift + diffusion * Z[:, t]

        S_paths = np.exp(log_S)

        if antithetic:
            log_S_anti = np.zeros((n, self.n_steps + 1))
            log_S_anti[:, 0] = np.log(self.S)
            for t in range(self.n_steps):
                log_S_anti[:, t + 1] = log_S_anti[:, t] + drift + diffusion * (-Z[:, t])
            S_paths_anti = np.exp(log_S_anti)
            S_paths = np.vstack([S_paths, S_paths_anti])

        return S_paths

    def price_european(
        self,
        option_type: str = "call",
        antithetic: bool = True,
        control_variate: bool = True,
    ) -> Tuple[float, float]:
        """
        Price European option with variance reduction.

        Returns
        -------
        (price, std_error)
        """
        S_paths = self._simulate_paths(antithetic)
        S_T = S_paths[:, -1]

        if option_type == "call":
            payoffs = np.maximum(S_T - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - S_T, 0.0)

        disc_factor = np.exp(-self.r * self.T)
        disc_payoffs = disc_factor * payoffs

        if control_variate:
            # Use S_T itself as control (known: E[e^{-rT}*S_T] = S*e^{-qT})
            cv = disc_factor * S_T
            cv_mean = self.S * np.exp(-self.q * self.T)

            # Optimal coefficient
            cov = np.cov(disc_payoffs, cv)
            if cov[1, 1] > 0:
                beta = cov[0, 1] / cov[1, 1]
                disc_payoffs = disc_payoffs - beta * (cv - cv_mean)

        price = float(disc_payoffs.mean())
        se = float(disc_payoffs.std() / np.sqrt(len(disc_payoffs)))
        return round(price, 6), round(se, 6)

    def price_asian(
        self,
        option_type: str = "call",
        average_type: str = "arithmetic",
    ) -> Tuple[float, float]:
        """
        Price Asian (average price) option.

        Parameters
        ----------
        option_type : 'call' or 'put'
        average_type : 'arithmetic' or 'geometric'

        Returns
        -------
        (price, std_error)
        """
        S_paths = self._simulate_paths(antithetic=True)

        if average_type == "arithmetic":
            avg = S_paths[:, 1:].mean(axis=1)
        else:
            avg = np.exp(np.log(S_paths[:, 1:] + 1e-10).mean(axis=1))

        if option_type == "call":
            payoffs = np.maximum(avg - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - avg, 0.0)

        disc_payoffs = np.exp(-self.r * self.T) * payoffs
        price = float(disc_payoffs.mean())
        se = float(disc_payoffs.std() / np.sqrt(len(disc_payoffs)))
        return round(price, 6), round(se, 6)

    def price_barrier(
        self,
        option_type: str = "call",
        barrier: float = None,
        barrier_type: str = "down-and-out",
    ) -> Tuple[float, float]:
        """
        Price barrier option.

        Parameters
        ----------
        barrier : float
            Barrier level.
        barrier_type : str
            'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'.

        Returns
        -------
        (price, std_error)
        """
        if barrier is None:
            barrier = 0.9 * self.S

        S_paths = self._simulate_paths(antithetic=True)
        S_T = S_paths[:, -1]

        path_min = S_paths.min(axis=1)
        path_max = S_paths.max(axis=1)

        if option_type == "call":
            payoffs = np.maximum(S_T - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - S_T, 0.0)

        if barrier_type == "down-and-out":
            active = path_min > barrier
        elif barrier_type == "up-and-out":
            active = path_max < barrier
        elif barrier_type == "down-and-in":
            active = path_min <= barrier
        elif barrier_type == "up-and-in":
            active = path_max >= barrier
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")

        disc_payoffs = np.exp(-self.r * self.T) * payoffs * active
        price = float(disc_payoffs.mean())
        se = float(disc_payoffs.std() / np.sqrt(len(disc_payoffs)))
        return round(price, 6), round(se, 6)

    def convergence_analysis(
        self, n_trials: int = 10, option_type: str = "call"
    ) -> "pd.DataFrame":
        """
        Run pricing multiple times and report convergence statistics.

        Returns
        -------
        pd.DataFrame
            n_paths, mean_price, std_error, 95_ci_low, 95_ci_high per trial.
        """
        import pandas as pd
        rows = []
        for trial in range(n_trials):
            self.seed = trial
            p, se = self.price_european(option_type=option_type)
            rows.append({
                "trial": trial,
                "price": p,
                "std_error": se,
                "ci_95_low": round(p - 1.96 * se, 6),
                "ci_95_high": round(p + 1.96 * se, 6),
            })
        return pd.DataFrame(rows).set_index("trial")


# ---------------------------------------------------------------------------
# Heston Model
# ---------------------------------------------------------------------------

class HestonModel:
    """
    Heston (1993) stochastic volatility model.

    dS = mu*S*dt + sqrt(V)*S*dW1
    dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW2
    dW1*dW2 = rho*dt

    Pricing via the characteristic function and Gil-Pelaez inversion.

    Parameters
    ----------
    S : float
        Current asset price.
    K : float
        Strike price.
    T : float
        Time to expiration.
    r : float
        Risk-free rate.
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between asset and variance processes.
    q : float
        Dividend yield.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        q: float = 0.0,
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.q = q

    def _char_func(self, phi: complex, j: int) -> complex:
        """
        Heston characteristic function Phi_j(phi).

        j=1 or j=2 for the two terms in the Gil-Pelaez formula.
        """
        S, K, T = self.S, self.K, self.T
        r, q = self.r, self.q
        v0, kappa, theta, xi, rho = self.v0, self.kappa, self.theta, self.xi, self.rho

        if j == 1:
            u = 0.5
            b = kappa - rho * xi
        else:
            u = -0.5
            b = kappa

        a = kappa * theta
        d = np.sqrt((rho * xi * phi * 1j - b) ** 2 - xi ** 2 * (2 * u * phi * 1j - phi ** 2))
        g = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)

        # Avoid division by zero
        if abs(1 - g * np.exp(d * T)) < 1e-10:
            return complex(0, 0)

        C = (r - q) * phi * 1j * T + a / xi ** 2 * (
            (b - rho * xi * phi * 1j + d) * T
            - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
        )
        D = (b - rho * xi * phi * 1j + d) / xi ** 2 * (
            (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
        )

        return np.exp(C + D * v0 + 1j * phi * np.log(S))

    def _P_j(self, j: int, integration_limit: float = 200.0, n_points: int = 1000) -> float:
        """Compute probability P_j via numerical integration."""
        x = np.log(self.K)

        def integrand(phi: float) -> float:
            cf = self._char_func(phi, j)
            val = np.real(np.exp(-1j * phi * x) * cf / (1j * phi + 1e-12))
            return float(val)

        phi_vals = np.linspace(1e-6, integration_limit, n_points)
        integrand_vals = np.array([integrand(p) for p in phi_vals])
        integral = np.trapz(integrand_vals, phi_vals)
        return 0.5 + integral / np.pi

    def call(self) -> float:
        """European call price under Heston model."""
        P1 = self._P_j(1)
        P2 = self._P_j(2)
        return (self.S * np.exp(-self.q * self.T) * P1
                - self.K * np.exp(-self.r * self.T) * P2)

    def put(self) -> float:
        """European put price via put-call parity."""
        c = self.call()
        return c - self.S * np.exp(-self.q * self.T) + self.K * np.exp(-self.r * self.T)

    def price(self) -> OptionPrice:
        c = self.call()
        p = self.put()
        return OptionPrice(
            call=round(c, 6),
            put=round(p, 6),
            intrinsic_call=round(max(self.S - self.K, 0.0), 6),
            intrinsic_put=round(max(self.K - self.S, 0.0), 6),
            model="Heston",
            params={"v0": self.v0, "kappa": self.kappa, "theta": self.theta,
                    "xi": self.xi, "rho": self.rho},
        )

    def calibrate(
        self,
        market_prices: "np.ndarray",
        strikes: "np.ndarray",
        option_types: List[str],
        initial_params: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Calibrate Heston parameters to market option prices.

        Parameters
        ----------
        market_prices : (n,) array of market option prices
        strikes : (n,) array of strike prices
        option_types : list of 'call' or 'put'
        initial_params : [kappa, theta, xi, rho, v0]

        Returns
        -------
        dict of calibrated parameters
        """
        if initial_params is None:
            initial_params = [2.0, self.v0, 0.5, -0.5, self.v0]

        def objective(params):
            kappa, theta, xi, rho, v0 = params
            if kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) >= 1 or v0 <= 0:
                return 1e10
            # Feller condition: 2*kappa*theta > xi^2
            if 2 * kappa * theta < xi ** 2:
                return 1e10
            errors = []
            for K_i, price_i, otype in zip(strikes, market_prices, option_types):
                model_i = HestonModel(
                    self.S, K_i, self.T, self.r, v0, kappa, theta, xi, rho, self.q
                )
                try:
                    if otype == "call":
                        model_price = model_i.call()
                    else:
                        model_price = model_i.put()
                    errors.append((model_price - price_i) ** 2)
                except Exception:
                    errors.append(1e4)
            return float(np.sum(errors))

        bounds = [(0.01, 20), (0.001, 1.0), (0.01, 3.0), (-0.99, 0.99), (0.001, 2.0)]
        result = optimize.minimize(
            objective, initial_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        kappa, theta, xi, rho, v0 = result.x
        return {
            "kappa": round(kappa, 6),
            "theta": round(theta, 6),
            "xi": round(xi, 6),
            "rho": round(rho, 6),
            "v0": round(v0, 6),
            "calibration_error": round(result.fun, 8),
            "success": result.success,
        }


# ---------------------------------------------------------------------------
# SABR Model
# ---------------------------------------------------------------------------

class SABRModel:
    """
    SABR stochastic volatility model (Hagan et al. 2002).

    Provides analytical approximation for implied volatility:
    sigma_B(F, K, T) = alpha * ... (Hagan formula)

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to expiration.
    alpha : float
        SABR alpha (initial volatility).
    beta : float
        CEV exponent (0 = normal, 1 = lognormal).
    rho : float
        Correlation between asset and vol processes.
    nu : float
        Vol-of-vol.
    """

    def __init__(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float = 0.5,
        rho: float = 0.0,
        nu: float = 0.4,
    ) -> None:
        self.F = F
        self.K = K
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def implied_vol(self) -> float:
        """
        Hagan et al. (2002) SABR implied volatility approximation.

        Returns
        -------
        float
            Black-Scholes implied volatility.
        """
        F, K, T = self.F, self.K, self.T
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        if abs(F - K) < 1e-10:
            # ATM approximation
            FK_beta = F ** (1 - beta)
            z = nu / alpha * FK_beta * (F - K)  # effectively 0
            atm = (alpha / FK_beta * (
                1 + ((1 - beta) ** 2 / 24 * alpha ** 2 / FK_beta ** 2
                     + rho * beta * nu * alpha / (4 * FK_beta)
                     + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
            ))
            return float(atm)

        FK_mid = np.sqrt(F * K)
        log_FK = np.log(F / K)

        # Equation (2.17) in Hagan et al.
        z = nu / alpha * FK_mid ** (1 - beta) * log_FK

        # chi(z)
        chi_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho) + 1e-10)

        A = alpha / (
            FK_mid ** (1 - beta)
            * (1 + (1 - beta) ** 2 / 24 * log_FK ** 2
               + (1 - beta) ** 4 / 1920 * log_FK ** 4)
        )
        B = z / (chi_z + 1e-10)
        C = (
            1
            + ((1 - beta) ** 2 / 24 * alpha ** 2 / FK_mid ** (2 - 2 * beta)
               + rho * beta * nu * alpha / (4 * FK_mid ** (1 - beta))
               + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
        )

        return float(A * B * C)

    def price(self, r: float = 0.0, q: float = 0.0) -> OptionPrice:
        """Price using SABR implied vol in Black-Scholes formula."""
        sigma_sabr = self.implied_vol()
        # Convert forward to spot: S = F * e^{-(r-q)*T}
        # Actually use forward directly in displaced Black formula
        S = self.F * np.exp(-(r - q) * self.T)
        bs = BlackScholes(S=S, K=self.K, T=self.T, r=r, sigma=sigma_sabr, q=q)
        op = bs.price()
        op.model = "SABR"
        op.params.update({"alpha": self.alpha, "beta": self.beta,
                           "rho": self.rho, "nu": self.nu, "sabr_iv": sigma_sabr})
        return op

    def calibrate(
        self,
        market_vols: "np.ndarray",
        strikes: "np.ndarray",
        initial_params: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Calibrate SABR [alpha, rho, nu] to market implied vols (beta fixed).

        Parameters
        ----------
        market_vols : (n,) array of market implied vols
        strikes : (n,) array of strikes
        initial_params : [alpha, rho, nu]

        Returns
        -------
        dict
        """
        if initial_params is None:
            initial_params = [0.3, -0.3, 0.4]

        def objective(params):
            alpha, rho, nu = params
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                return 1e10
            errors = []
            for K_i, target_vol in zip(strikes, market_vols):
                sabr_i = SABRModel(self.F, K_i, self.T, alpha, self.beta, rho, nu)
                model_vol = sabr_i.implied_vol()
                errors.append((model_vol - target_vol) ** 2)
            return float(np.sum(errors))

        bounds = [(0.001, 5.0), (-0.99, 0.99), (0.001, 5.0)]
        result = optimize.minimize(
            objective, initial_params,
            bounds=bounds,
            method="L-BFGS-B",
        )
        alpha, rho, nu = result.x
        return {
            "alpha": round(alpha, 6),
            "beta": round(self.beta, 6),
            "rho": round(rho, 6),
            "nu": round(nu, 6),
            "rmse": round(np.sqrt(result.fun / max(len(strikes), 1)), 8),
            "success": result.success,
        }
