"""
research/simulation/calibration_tools.py

Model calibration utilities for SRFM simulation research. Provides
moment-matching calibrators for GBM, OU, GARCH, and jump-diffusion models,
plus a suite of goodness-of-fit diagnostics.

Usage:
    cal = MomentCalibrator()
    params = cal.calibrate_gbm(returns)

    gof = GoodnessOfFit()
    result = gof.ks_test(empirical_returns, simulated_returns)
    score = gof.stylized_facts_score(returns)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GBMParams:
    """Geometric Brownian Motion parameters."""
    mu: float       # annualised drift
    sigma: float    # annualised volatility

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError(f"GBM sigma must be non-negative, got {self.sigma}")


@dataclass
class OUParams:
    """Ornstein-Uhlenbeck process parameters."""
    kappa: float    # mean-reversion speed (per year)
    theta: float    # long-run mean
    sigma: float    # diffusion coefficient

    def half_life(self) -> float:
        """Mean-reversion half-life in the same time units as kappa."""
        if self.kappa <= 0:
            return float("inf")
        return math.log(2.0) / self.kappa


@dataclass
class GARCHParams:
    """GARCH(1,1) parameters."""
    omega: float    # constant term
    alpha: float    # ARCH coefficient
    beta: float     # GARCH coefficient

    def persistence(self) -> float:
        """alpha + beta; < 1 implies covariance stationarity."""
        return self.alpha + self.beta

    def unconditional_variance(self) -> float:
        """Long-run variance implied by GARCH(1,1)."""
        denom = 1.0 - self.persistence()
        if denom <= 0:
            return float("inf")
        return self.omega / denom


@dataclass
class JumpParams:
    """Merton jump-diffusion parameters."""
    mu: float           # diffusion drift
    sigma: float        # diffusion volatility
    lambda_j: float     # Poisson jump intensity (jumps/year)
    mu_j: float         # mean jump size (log)
    sigma_j: float      # jump size std (log)

    def expected_return(self) -> float:
        """Expected log-return per unit time."""
        return (
            self.mu
            + self.lambda_j * (math.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0)
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KSResult:
    """Kolmogorov-Smirnov test result."""
    statistic: float
    p_value: float
    reject_h0: bool    # True if distributions differ at 5% level

    def __repr__(self) -> str:
        conclusion = "REJECT H0" if self.reject_h0 else "fail to reject H0"
        return (
            f"KSResult(D={self.statistic:.4f}, p={self.p_value:.4f}, {conclusion})"
        )


@dataclass
class ADResult:
    """Anderson-Darling test result."""
    statistic: float
    critical_values: List[float]
    significance_levels: List[float]
    reject_at_5pct: bool

    def __repr__(self) -> str:
        conclusion = "REJECT" if self.reject_at_5pct else "fail to reject"
        return (
            f"ADResult(A2={self.statistic:.4f}, {conclusion} at 5%)"
        )


# ---------------------------------------------------------------------------
# MomentCalibrator
# ---------------------------------------------------------------------------

class MomentCalibrator:
    """
    Calibrates stochastic process parameters to match empirical moments.

    All methods accept a numpy array of observed values and return a
    parameter dataclass. Calibration is done via closed-form moment matching
    where possible, and scipy.optimize.minimize otherwise.
    """

    # ------------------------------------------------------------------
    # GBM
    # ------------------------------------------------------------------
    def calibrate_gbm(
        self,
        returns: NDArray[np.float64],
        dt: float = 1.0 / 252,
    ) -> GBMParams:
        """
        Calibrate GBM parameters from log-return observations.

        Matches empirical mean and variance using closed-form estimators.
        For GBM: E[r] = (mu - 0.5*sigma^2)*dt, Var[r] = sigma^2 * dt.

        Parameters
        # --------
        returns : array of log-returns
        dt : float
            Time step in years (default 1/252 = daily).

        Returns
        # -----
        GBMParams
        """
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 2:
            raise ValueError("Need at least 2 return observations")

        mu_r = float(np.mean(returns))
        var_r = float(np.var(returns, ddof=1))

        sigma_ann = math.sqrt(max(var_r / dt, 0.0))
        mu_ann = mu_r / dt + 0.5 * sigma_ann**2

        logger.debug("GBM calibration: mu=%.4f sigma=%.4f", mu_ann, sigma_ann)
        return GBMParams(mu=mu_ann, sigma=sigma_ann)

    # ------------------------------------------------------------------
    # OU
    # ------------------------------------------------------------------
    def calibrate_ou(
        self,
        prices: NDArray[np.float64],
        dt: float = 1.0 / 252,
    ) -> OUParams:
        """
        Calibrate Ornstein-Uhlenbeck parameters from a price (or log-price)
        series using OLS regression on the discrete-time representation:
            X_{t+1} = a + b * X_t + epsilon

        where kappa = -log(b)/dt, theta = a/(1-b).

        Parameters
        # --------
        prices : array of prices or log-prices
        dt : float
            Time step in years.

        Returns
        # -----
        OUParams
        """
        prices = np.asarray(prices, dtype=np.float64)
        if len(prices) < 10:
            raise ValueError("Need at least 10 price observations for OU calibration")

        x = prices[:-1]
        y = prices[1:]

        # OLS: y = a + b*x
        n = len(x)
        sum_x = float(np.sum(x))
        sum_y = float(np.sum(y))
        sum_xx = float(np.sum(x * x))
        sum_xy = float(np.sum(x * y))

        denom = n * sum_xx - sum_x**2
        if abs(denom) < 1e-12:
            raise ValueError("Degenerate price series; cannot calibrate OU")

        b = (n * sum_xy - sum_x * sum_y) / denom
        a = (sum_y - b * sum_x) / n

        # convert to continuous-time parameters
        b_clipped = min(b, 1.0 - 1e-9)
        kappa = -math.log(max(b_clipped, 1e-9)) / dt
        theta = a / (1.0 - b) if abs(1.0 - b) > 1e-9 else float(np.mean(prices))

        # sigma from residual variance
        residuals = y - (a + b * x)
        var_eps = float(np.var(residuals, ddof=2))
        # sigma^2 = var_eps / ((1 - exp(-2*kappa*dt)) / (2*kappa))
        factor = (1.0 - math.exp(-2.0 * kappa * dt)) / (2.0 * kappa) if kappa > 0 else dt
        sigma = math.sqrt(max(var_eps / factor, 0.0))

        logger.debug(
            "OU calibration: kappa=%.4f theta=%.4f sigma=%.4f half-life=%.2f days",
            kappa, theta, sigma, math.log(2) / kappa / dt if kappa > 0 else float("inf"),
        )
        return OUParams(kappa=kappa, theta=theta, sigma=sigma)

    # ------------------------------------------------------------------
    # GARCH
    # ------------------------------------------------------------------
    def calibrate_garch(
        self,
        returns: NDArray[np.float64],
        max_iter: int = 500,
    ) -> GARCHParams:
        """
        Calibrate GARCH(1,1) parameters by quasi-maximum likelihood.

        Optimises the Gaussian log-likelihood of the GARCH(1,1) model:
            h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}

        Parameters
        # --------
        returns : array of log-returns (zero-mean recommended)
        max_iter : int
            Maximum optimizer iterations.

        Returns
        # -----
        GARCHParams
        """
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 20:
            raise ValueError("Need at least 20 observations for GARCH calibration")

        unconditional_var = float(np.var(returns, ddof=1))

        def neg_loglik(params: NDArray[np.float64]) -> float:
            omega, alpha, beta = params
            if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1.0:
                return 1e10
            n = len(returns)
            h = np.zeros(n, dtype=np.float64)
            h[0] = unconditional_var
            for t in range(1, n):
                h[t] = omega + alpha * returns[t - 1]**2 + beta * h[t - 1]
                h[t] = max(h[t], 1e-12)
            # Gaussian log-likelihood
            ll = -0.5 * np.sum(np.log(2 * math.pi * h) + returns**2 / h)
            return -ll

        # initial guess: target 5% persistence gap
        p0 = 0.90
        a0 = 0.05
        b0 = p0 - a0
        w0 = unconditional_var * (1.0 - p0)
        x0 = np.array([w0, a0, b0])

        bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.999)]
        constraints = [
            {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]},
        ]

        result = optimize.minimize(
            neg_loglik,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning("GARCH optimisation did not converge: %s", result.message)

        omega, alpha, beta = result.x
        logger.debug(
            "GARCH(1,1): omega=%.2e alpha=%.4f beta=%.4f persistence=%.4f",
            omega, alpha, beta, alpha + beta,
        )
        return GARCHParams(omega=float(omega), alpha=float(alpha), beta=float(beta))

    # ------------------------------------------------------------------
    # Jump diffusion
    # ------------------------------------------------------------------
    def calibrate_jump_diffusion(
        self,
        returns: NDArray[np.float64],
        dt: float = 1.0 / 252,
        max_iter: int = 300,
    ) -> JumpParams:
        """
        Calibrate Merton jump-diffusion parameters by moment matching.

        Matches: mean, variance, skewness, excess kurtosis of empirical returns.
        The first four moments of Merton JD are:
            E[r]    = (mu - 0.5*sigma^2 + lambda*(exp(mu_j+0.5*sigma_j^2)-1)) * dt
            Var[r]  = (sigma^2 + lambda*(sigma_j^2 + mu_j^2)) * dt
            Skew    = lambda * mu_j * (3*sigma_j^2 + mu_j^2) * dt / Var^1.5
            Kurt    = lambda * (3*sigma_j^4 + 6*sigma_j^2*mu_j^2 + mu_j^4) * dt / Var^2

        Parameters
        # --------
        returns : array of log-returns
        dt : float
            Time step in years.
        max_iter : int
            Optimiser iteration limit.

        Returns
        # -----
        JumpParams
        """
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 30:
            raise ValueError("Need at least 30 observations for jump-diffusion calibration")

        emp_mean = float(np.mean(returns))
        emp_var = float(np.var(returns, ddof=1))
        emp_skew = float(stats.skew(returns))
        emp_kurt = float(stats.kurtosis(returns))   # excess kurtosis

        emp_var = max(emp_var, 1e-12)

        def moment_residuals(params: NDArray[np.float64]) -> float:
            log_sigma, log_lambda, mu_j, log_sigma_j = params
            sigma = math.exp(log_sigma)
            lam = math.exp(log_lambda)
            sigma_j = math.exp(log_sigma_j)

            jump_mean = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
            model_var = (sigma**2 + lam * (sigma_j**2 + mu_j**2)) * dt
            model_var = max(model_var, 1e-12)

            model_skew_num = lam * mu_j * (3 * sigma_j**2 + mu_j**2) * dt
            model_skew = model_skew_num / model_var**1.5

            model_kurt_num = lam * (
                3 * sigma_j**4 + 6 * sigma_j**2 * mu_j**2 + mu_j**4
            ) * dt
            model_kurt = model_kurt_num / model_var**2

            res_var = (model_var - emp_var)**2 / emp_var**2
            res_skew = (model_skew - emp_skew)**2
            res_kurt = (model_kurt - emp_kurt)**2

            return res_var + 0.5 * res_skew + 0.25 * res_kurt

        # initial guess based on empirical stats
        sigma0 = math.sqrt(max(emp_var / dt * 0.7, 1e-8))
        x0 = np.array([
            math.log(sigma0),
            math.log(max(abs(emp_kurt) * 0.1 + 0.5, 0.1)),
            emp_skew * 0.01,
            math.log(max(math.sqrt(abs(emp_kurt) * emp_var / dt * 0.01), 0.001)),
        ])

        result = optimize.minimize(
            moment_residuals,
            x0,
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-6, "fatol": 1e-6},
        )

        if not result.success:
            logger.warning(
                "Jump-diffusion optimisation did not fully converge: %s", result.message
            )

        log_sigma, log_lambda, mu_j, log_sigma_j = result.x
        sigma = float(math.exp(log_sigma))
        lam = float(math.exp(log_lambda))
        sigma_j = float(math.exp(log_sigma_j))
        mu = emp_mean / dt - lam * (math.exp(mu_j + 0.5 * sigma_j**2) - 1.0)

        logger.debug(
            "JumpDiff: mu=%.4f sigma=%.4f lambda=%.2f mu_j=%.4f sigma_j=%.4f",
            mu, sigma, lam, mu_j, sigma_j,
        )
        return JumpParams(
            mu=float(mu),
            sigma=sigma,
            lambda_j=lam,
            mu_j=float(mu_j),
            sigma_j=sigma_j,
        )


# ---------------------------------------------------------------------------
# GoodnessOfFit
# ---------------------------------------------------------------------------

class GoodnessOfFit:
    """
    Statistical goodness-of-fit diagnostics for comparing empirical and
    simulated return distributions.
    """

    def ks_test(
        self,
        empirical: NDArray[np.float64],
        simulated: NDArray[np.float64],
    ) -> KSResult:
        """
        Two-sample Kolmogorov-Smirnov test.

        H0: both samples are drawn from the same distribution.

        Parameters
        # --------
        empirical : array of observed values
        simulated : array of simulated values

        Returns
        # -----
        KSResult
        """
        empirical = np.asarray(empirical, dtype=np.float64)
        simulated = np.asarray(simulated, dtype=np.float64)

        if len(empirical) < 2 or len(simulated) < 2:
            raise ValueError("Need at least 2 observations in each sample")

        result = stats.ks_2samp(empirical, simulated)
        return KSResult(
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            reject_h0=bool(result.pvalue < 0.05),
        )

    def anderson_darling(
        self,
        empirical: NDArray[np.float64],
        distribution: str = "norm",
    ) -> ADResult:
        """
        One-sample Anderson-Darling test against a theoretical distribution.

        Parameters
        # --------
        empirical : array of observed values
        distribution : str
            Distribution name as accepted by scipy.stats.anderson.
            Options: 'norm', 'expon', 'logistic', 'gumbel', 'extreme1'.

        Returns
        # -----
        ADResult
        """
        empirical = np.asarray(empirical, dtype=np.float64)
        if len(empirical) < 5:
            raise ValueError("Need at least 5 observations for Anderson-Darling test")

        valid_dists = {"norm", "expon", "logistic", "gumbel", "extreme1"}
        if distribution not in valid_dists:
            raise ValueError(
                f"distribution must be one of {valid_dists}, got '{distribution}'"
            )

        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", FutureWarning)
            result = stats.anderson(empirical, dist=distribution)
        crit_vals = list(result.critical_values)
        sig_levels = list(result.significance_level)

        # reject at 5% level: find the 5% critical value
        reject_at_5pct = False
        for cv, sl in zip(crit_vals, sig_levels):
            if abs(sl - 5.0) < 1.0:
                reject_at_5pct = bool(result.statistic > cv)
                break

        return ADResult(
            statistic=float(result.statistic),
            critical_values=crit_vals,
            significance_levels=sig_levels,
            reject_at_5pct=reject_at_5pct,
        )

    def qq_plot_stats(
        self,
        empirical: NDArray[np.float64],
        theoretical_quantiles: NDArray[np.float64],
    ) -> float:
        """
        Compute R^2 of the QQ plot (empirical vs theoretical quantiles).

        A value close to 1.0 indicates the empirical distribution matches
        the theoretical one well.

        Parameters
        # --------
        empirical : array of observed values
        theoretical_quantiles : precomputed theoretical quantiles at the
            same probability points as the empirical order statistics.

        Returns
        # -----
        float
            R^2 of the linear fit on the QQ plot.
        """
        empirical = np.asarray(empirical, dtype=np.float64)
        theoretical_quantiles = np.asarray(theoretical_quantiles, dtype=np.float64)

        if len(empirical) != len(theoretical_quantiles):
            raise ValueError(
                "empirical and theoretical_quantiles must have the same length"
            )

        emp_sorted = np.sort(empirical)

        # R^2 = 1 - SS_res / SS_tot
        ss_res = float(np.sum((emp_sorted - theoretical_quantiles)**2))
        ss_tot = float(np.sum((emp_sorted - np.mean(emp_sorted))**2))
        if ss_tot < 1e-16:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    def stylized_facts_score(
        self,
        returns: NDArray[np.float64],
        dt: float = 1.0 / 252,
    ) -> Dict[str, float]:
        """
        Compute a set of stylized-fact diagnostics for a return series.

        Checks:
          - heavy_tails: excess kurtosis > 1 (score = min(kurt/3, 1))
          - vol_clustering: autocorrelation of |r| at lag 1 (should be > 0)
          - no_autocorr: 1 - abs(AC1 of r) (raw returns near-zero AC)
          - negative_skew: captures left skew (common in equity)
          - leverage_effect: correlation between r_t and r^2_{t+1} (should be < 0)
          - overall: geometric mean of individual scores

        All scores are in [0, 1]; higher is better (closer to stylized facts).

        Parameters
        # --------
        returns : array of log-returns
        dt : float
            Time step in years (not used in scoring, for reference only).

        Returns
        # -----
        Dict[str, float] with keys: heavy_tails, vol_clustering,
            no_autocorr, negative_skew, leverage_effect, overall.
        """
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 20:
            raise ValueError("Need at least 20 observations for stylized facts scoring")

        results: Dict[str, float] = {}

        # 1. Heavy tails: excess kurtosis
        kurt = float(stats.kurtosis(returns))
        results["heavy_tails"] = float(np.clip(kurt / 3.0, 0.0, 1.0))

        # 2. Volatility clustering: AC1 of squared returns
        r2 = returns**2
        if len(r2) > 2:
            ac1_sq = float(np.corrcoef(r2[:-1], r2[1:])[0, 1])
        else:
            ac1_sq = 0.0
        results["vol_clustering"] = float(np.clip(ac1_sq, 0.0, 1.0))

        # 3. Absence of serial correlation in raw returns
        if len(returns) > 2:
            ac1_r = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        else:
            ac1_r = 0.0
        results["no_autocorr"] = float(np.clip(1.0 - abs(ac1_r), 0.0, 1.0))

        # 4. Negative skewness (equity returns tend to be left-skewed)
        skew = float(stats.skew(returns))
        results["negative_skew"] = float(np.clip(-skew / 2.0 + 0.5, 0.0, 1.0))

        # 5. Leverage effect: negative correlation r_t vs |r_{t+1}|
        if len(returns) > 2:
            corr_lev = float(np.corrcoef(returns[:-1], np.abs(returns[1:]))[0, 1])
        else:
            corr_lev = 0.0
        results["leverage_effect"] = float(np.clip((-corr_lev + 1.0) / 2.0, 0.0, 1.0))

        # overall: geometric mean of individual scores
        scores = [v for v in results.values()]
        if all(s >= 0 for s in scores) and len(scores) > 0:
            log_mean = np.mean(np.log(np.clip(scores, 1e-9, 1.0)))
            results["overall"] = float(math.exp(log_mean))
        else:
            results["overall"] = 0.0

        return results

    # ------------------------------------------------------------------
    # Convenience: simulate GBM returns and compare
    # ------------------------------------------------------------------
    def compare_to_gbm(
        self,
        empirical: NDArray[np.float64],
        gbm_params: "GBMParams",
        dt: float = 1.0 / 252,
        rng: Optional[np.random.Generator] = None,
    ) -> KSResult:
        """
        Generate a GBM sample of the same length as `empirical` and run a
        KS test.

        Parameters
        # --------
        empirical : observed log-returns
        gbm_params : calibrated GBM parameters
        dt : time step
        rng : optional random generator

        Returns
        # -----
        KSResult
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(empirical)
        sim = rng.normal(
            (gbm_params.mu - 0.5 * gbm_params.sigma**2) * dt,
            gbm_params.sigma * math.sqrt(dt),
            size=n,
        )
        return self.ks_test(empirical, sim)
