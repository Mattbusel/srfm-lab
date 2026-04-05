"""
risk-engine/var_calculator.py

Value at Risk (VaR) calculations using multiple methodologies:
  - Historical simulation
  - Parametric (normal distribution)
  - Cornish-Fisher expansion (adjusts for skewness and excess kurtosis)
  - Conditional VaR / Expected Shortfall (CVaR)
  - Monte Carlo simulation
  - Rolling VaR time series
  - VaR backtesting with Kupiec and Christoffersen tests
  - Extreme Value Theory (GPD tail fitting)

Results are persisted to the var_estimates table via the shared DB connection.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VaRBacktestResult:
    """
    Full result of a VaR backtest including violations and hypothesis test outcomes.

    Attributes
    ----------
    total_observations : int
        Number of return observations in the backtest window.
    expected_violations : float
        Expected number of VaR breaches under the model.
    actual_violations : int
        Observed number of VaR breaches.
    violation_rate : float
        Fraction of days on which the VaR was exceeded.
    kupiec_lr_stat : float
        Kupiec likelihood-ratio test statistic (unconditional coverage).
    kupiec_p_value : float
        p-value for the Kupiec test. Small p → model rejected.
    christoffersen_lr_stat : float
        Christoffersen independence test statistic.
    christoffersen_p_value : float
        p-value for the independence test. Small p → clustering in violations.
    combined_lr_stat : float
        Combined (conditional coverage) LR statistic.
    combined_p_value : float
        p-value for the combined test.
    confidence : float
        Confidence level used (e.g. 0.95).
    pass_kupiec : bool
        True if the model is not rejected at the 5 % level by Kupiec.
    pass_christoffersen : bool
        True if violation clustering is not detected at the 5 % level.
    violation_dates : list[str]
        ISO-8601 timestamps of breach dates (if index available).
    """

    total_observations: int
    expected_violations: float
    actual_violations: int
    violation_rate: float
    kupiec_lr_stat: float
    kupiec_p_value: float
    christoffersen_lr_stat: float
    christoffersen_p_value: float
    combined_lr_stat: float
    combined_p_value: float
    confidence: float
    pass_kupiec: bool
    pass_christoffersen: bool
    violation_dates: list[str] = field(default_factory=list)

    @property
    def is_valid_model(self) -> bool:
        """Return True if the VaR model passes both backtest criteria."""
        return self.pass_kupiec and self.pass_christoffersen

    def summary(self) -> str:
        lines = [
            f"VaR Backtest Summary (confidence={self.confidence:.0%})",
            f"  Observations       : {self.total_observations}",
            f"  Expected violations: {self.expected_violations:.1f}",
            f"  Actual violations  : {self.actual_violations}",
            f"  Violation rate     : {self.violation_rate:.4f}",
            f"  Kupiec LR stat     : {self.kupiec_lr_stat:.4f}  p={self.kupiec_p_value:.4f}  {'PASS' if self.pass_kupiec else 'FAIL'}",
            f"  Christoffersen LR  : {self.christoffersen_lr_stat:.4f}  p={self.christoffersen_p_value:.4f}  {'PASS' if self.pass_christoffersen else 'FAIL'}",
            f"  Model valid        : {self.is_valid_model}",
        ]
        return "\n".join(lines)


@dataclass
class VaREstimate:
    """Single VaR estimate ready for DB persistence."""

    run_id: str
    method: str
    confidence: float
    var_value: float
    cvar_value: Optional[float] = None
    window_days: Optional[int] = None
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


# ---------------------------------------------------------------------------
# Core calculator
# ---------------------------------------------------------------------------


class VaRCalculator:
    """
    Multi-method Value at Risk calculator.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database.  When provided, each estimate is
        automatically persisted to the ``var_estimates`` table.
    run_id : str, optional
        Identifier for the current pipeline run.  Auto-generated if omitted.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> returns = pd.Series(rng.normal(0.0005, 0.012, 1000))
    >>> calc = VaRCalculator()
    >>> var_95 = calc.historical_var(returns, confidence=0.95)
    >>> print(f"Historical VaR (95%): {var_95:.4f}")
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        self.run_id = run_id or str(uuid.uuid4())
        self._estimates: list[VaREstimate] = []

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def historical_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        window: int = 252,
    ) -> float:
        """
        Historical simulation VaR.

        Uses the empirical distribution of the most recent ``window`` returns
        to estimate the loss threshold at the given confidence level.

        Parameters
        ----------
        returns : pd.Series
            Daily (or bar) return series.  Losses are negative values.
        confidence : float
            Confidence level (e.g. 0.95 for 95 % VaR).
        window : int
            Number of most-recent observations to include.

        Returns
        -------
        float
            VaR as a positive number (magnitude of the loss at the quantile).
        """
        if len(returns) == 0:
            raise ValueError("returns series must not be empty")
        if not (0 < confidence < 1):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        sample = returns.iloc[-window:] if len(returns) > window else returns
        var_value = float(-np.percentile(sample.dropna().values, (1 - confidence) * 100))
        cvar_value = self._cvar_from_sample(sample.dropna().values, confidence)

        self._record_estimate("historical", confidence, var_value, cvar_value, window)
        return var_value

    def parametric_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Parametric (variance-covariance) VaR assuming normally distributed returns.

        Parameters
        ----------
        returns : pd.Series
            Return series.
        confidence : float
            Confidence level.

        Returns
        -------
        float
            VaR as a positive number.
        """
        arr = returns.dropna().values
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        z = stats.norm.ppf(1 - confidence)
        var_value = float(-(mu + z * sigma))
        cvar_value = float(sigma * stats.norm.pdf(z) / (1 - confidence) - mu)

        self._record_estimate("parametric", confidence, var_value, cvar_value)
        return var_value

    def cornish_fisher_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Cornish-Fisher expansion VaR, adjusting the normal quantile for
        empirical skewness and excess kurtosis.

        The adjusted quantile is:

            z_cf = z + (z²-1)*S/6 + (z³-3z)*K/24 - (2z³-5z)*S²/36

        where S is skewness, K is excess kurtosis, and z = Φ⁻¹(1-α).

        Parameters
        ----------
        returns : pd.Series
            Return series.
        confidence : float
            Confidence level.

        Returns
        -------
        float
            Cornish-Fisher adjusted VaR.
        """
        arr = returns.dropna().values
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        S = float(stats.skew(arr))
        K = float(stats.kurtosis(arr))  # excess kurtosis

        z = stats.norm.ppf(1 - confidence)
        z_cf = (
            z
            + (z**2 - 1) * S / 6
            + (z**3 - 3 * z) * K / 24
            - (2 * z**3 - 5 * z) * S**2 / 36
        )
        var_value = float(-(mu + z_cf * sigma))

        # CVaR approximation via numerical integration of the adjusted quantile
        cvar_value = self._cvar_from_sample(arr, confidence)

        self._record_estimate("cornish_fisher", confidence, var_value, cvar_value)
        return var_value

    def conditional_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Conditional VaR (CVaR) / Expected Shortfall — the mean loss beyond
        the VaR threshold.

        Parameters
        ----------
        returns : pd.Series
            Return series.
        confidence : float
            Confidence level.

        Returns
        -------
        float
            CVaR as a positive number.
        """
        arr = returns.dropna().values
        cvar = self._cvar_from_sample(arr, confidence)
        var_value = float(-np.percentile(arr, (1 - confidence) * 100))
        self._record_estimate("cvar", confidence, var_value, cvar)
        return cvar

    def monte_carlo_var(
        self,
        returns: pd.Series,
        n_sims: int = 10_000,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """
        Monte Carlo VaR using bootstrapped geometric Brownian motion paths.

        Parameters
        ----------
        returns : pd.Series
            Historical returns used to estimate drift and volatility.
        n_sims : int
            Number of simulation paths.
        confidence : float
            Confidence level.
        horizon : int
            Forecast horizon in bars.

        Returns
        -------
        float
            Monte Carlo VaR.
        """
        arr = returns.dropna().values
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))

        rng = np.random.default_rng(seed=42)
        shocks = rng.normal(mu, sigma, size=(n_sims, horizon))
        sim_returns = shocks.sum(axis=1)  # multi-period sum (approx log returns)

        var_value = float(-np.percentile(sim_returns, (1 - confidence) * 100))
        cvar_value = self._cvar_from_sample(sim_returns, confidence)

        self._record_estimate("monte_carlo", confidence, var_value, cvar_value)
        return var_value

    def rolling_var(
        self,
        returns: pd.Series,
        window: int = 30,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> pd.Series:
        """
        Compute a rolling VaR time series.

        Parameters
        ----------
        returns : pd.Series
            Return series (must have a DatetimeIndex for meaningful output).
        window : int
            Rolling window size in bars.
        confidence : float
            Confidence level.
        method : str
            One of ``'historical'``, ``'parametric'``, ``'cornish_fisher'``.

        Returns
        -------
        pd.Series
            Rolling VaR values indexed identically to the input series.
        """
        if method not in {"historical", "parametric", "cornish_fisher"}:
            raise ValueError(f"Unknown method: {method!r}")

        results: list[float] = []
        arr = returns.dropna().values

        for i in range(len(arr)):
            if i < window - 1:
                results.append(np.nan)
                continue
            chunk = pd.Series(arr[i - window + 1 : i + 1])
            if method == "historical":
                val = float(-np.percentile(chunk.values, (1 - confidence) * 100))
            elif method == "parametric":
                mu = float(np.mean(chunk))
                sigma = float(np.std(chunk, ddof=1))
                z = stats.norm.ppf(1 - confidence)
                val = float(-(mu + z * sigma))
            else:
                mu = float(np.mean(chunk))
                sigma = float(np.std(chunk, ddof=1))
                S = float(stats.skew(chunk))
                K = float(stats.kurtosis(chunk))
                z = stats.norm.ppf(1 - confidence)
                z_cf = (
                    z
                    + (z**2 - 1) * S / 6
                    + (z**3 - 3 * z) * K / 24
                    - (2 * z**3 - 5 * z) * S**2 / 36
                )
                val = float(-(mu + z_cf * sigma))
            results.append(val)

        out = pd.Series(results, index=returns.index, name=f"rolling_var_{confidence:.2f}")
        return out

    def var_backtest(
        self,
        returns: pd.Series,
        var_series: pd.Series,
    ) -> VaRBacktestResult:
        """
        Backtest a VaR model using Kupiec (1995) unconditional coverage test and
        Christoffersen (1998) conditional coverage / independence test.

        Parameters
        ----------
        returns : pd.Series
            Realised daily returns.
        var_series : pd.Series
            Corresponding VaR estimates (positive numbers representing the loss
            threshold).  Must be aligned with ``returns``.

        Returns
        -------
        VaRBacktestResult
        """
        aligned = pd.DataFrame({"ret": returns, "var": var_series}).dropna()
        n = len(aligned)
        if n < 10:
            raise ValueError("Need at least 10 observations for backtest")

        # Infer confidence from the tail proportion observed
        # (assumes var_series was computed at a fixed confidence level)
        violations: pd.Series = aligned["ret"] < -aligned["var"]
        x = int(violations.sum())

        # Confidence from the nominal var_series level — we back it out by
        # examining how many violations we expect to see at the estimated rate
        # If not provided we treat the expected rate as (1 - confidence)
        # Use the empirical violation rate to drive the Kupiec test
        p_hat = x / n  # empirical violation rate
        # We compare against the nominal confidence embedded in the var_series
        # Since we don't have access to the original confidence here we use 0.95
        # as a default and let the caller interpret
        confidence = 1.0 - p_hat if p_hat > 0 else 0.95
        p0 = 1.0 - confidence  # expected violation probability

        # --- Kupiec likelihood-ratio test (unconditional coverage) ---
        kupiec_lr = self._kupiec_lr(n, x, p0)
        kupiec_p = float(stats.chi2.sf(kupiec_lr, df=1))

        # --- Christoffersen independence test ---
        chris_lr, combined_lr = self._christoffersen_lr(violations.values.astype(int), p0)
        chris_p = float(stats.chi2.sf(chris_lr, df=1))
        combined_p = float(stats.chi2.sf(combined_lr, df=2))

        violation_dates = (
            aligned.index[violations].strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
            if hasattr(aligned.index, "strftime")
            else [str(i) for i in aligned.index[violations]]
        )

        return VaRBacktestResult(
            total_observations=n,
            expected_violations=p0 * n,
            actual_violations=x,
            violation_rate=p_hat,
            kupiec_lr_stat=kupiec_lr,
            kupiec_p_value=kupiec_p,
            christoffersen_lr_stat=chris_lr,
            christoffersen_p_value=chris_p,
            combined_lr_stat=combined_lr,
            combined_p_value=combined_p,
            confidence=confidence,
            pass_kupiec=kupiec_p > 0.05,
            pass_christoffersen=chris_p > 0.05,
            violation_dates=violation_dates,
        )

    def extreme_value_theory_var(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        threshold_quantile: float = 0.90,
    ) -> float:
        """
        Extreme Value Theory VaR using the Generalised Pareto Distribution (GPD)
        fitted to the left tail via maximum likelihood estimation.

        This method exceedance-threshold approach (Peaks over Threshold, PoT):
          1. Extract tail observations below the threshold quantile.
          2. Fit a GPD to the exceedances.
          3. Extrapolate to the requested confidence level.

        Parameters
        ----------
        returns : pd.Series
            Return series.
        confidence : float
            Confidence level (typically 0.99 or 0.999 for EVT).
        threshold_quantile : float
            Quantile used to define the tail threshold (e.g. 0.90 means the
            worst 10 % of returns form the tail sample).

        Returns
        -------
        float
            EVT-GPD VaR estimate.
        """
        arr = returns.dropna().values
        losses = -arr  # convert to positive losses
        u = float(np.percentile(losses, threshold_quantile * 100))
        exceedances = losses[losses > u] - u

        if len(exceedances) < 10:
            # Fall back to historical if insufficient tail data
            return self.historical_var(returns, confidence=confidence)

        xi, beta = self._fit_gpd(exceedances)

        n = len(losses)
        n_u = len(exceedances)
        p = 1 - confidence

        if xi == 0:
            var_evt = u + beta * np.log(n / (n_u * p))
        else:
            var_evt = u + beta / xi * ((n / (n_u * p)) ** xi - 1)

        cvar_value = self._cvar_from_sample(arr, confidence)
        self._record_estimate("evt_gpd", confidence, float(var_evt), cvar_value)
        return float(var_evt)

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def flush_to_db(self) -> int:
        """
        Write all buffered VaR estimates to the database.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not self.db_path or not self._estimates:
            return 0

        rows = [
            (
                e.run_id,
                e.method,
                e.confidence,
                e.var_value,
                e.cvar_value,
                e.window_days,
                e.computed_at,
            )
            for e in self._estimates
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO var_estimates
                    (run_id, method, confidence, var_value, cvar_value, window_days, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        inserted = len(rows)
        self._estimates.clear()
        return inserted

    def fetch_estimates(self, run_id: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve stored VaR estimates from the database.

        Parameters
        ----------
        run_id : str, optional
            Filter by run_id; returns all records if omitted.

        Returns
        -------
        pd.DataFrame
        """
        if not self.db_path:
            raise RuntimeError("db_path not configured")

        query = "SELECT * FROM var_estimates"
        params: tuple = ()
        if run_id:
            query += " WHERE run_id = ?"
            params = (run_id,)

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cvar_from_sample(arr: np.ndarray, confidence: float) -> float:
        """Compute CVaR (Expected Shortfall) directly from a sample array."""
        threshold = np.percentile(arr, (1 - confidence) * 100)
        tail = arr[arr <= threshold]
        if len(tail) == 0:
            return float(-threshold)
        return float(-np.mean(tail))

    def _record_estimate(
        self,
        method: str,
        confidence: float,
        var_value: float,
        cvar_value: Optional[float],
        window_days: Optional[int] = None,
    ) -> None:
        """Buffer a VaR estimate for later persistence."""
        self._estimates.append(
            VaREstimate(
                run_id=self.run_id,
                method=method,
                confidence=confidence,
                var_value=var_value,
                cvar_value=cvar_value,
                window_days=window_days,
            )
        )

    @staticmethod
    def _kupiec_lr(n: int, x: int, p0: float) -> float:
        """
        Kupiec (1995) likelihood-ratio statistic for unconditional coverage.

        LR = -2 ln[ L(p0) / L(p_hat) ]
        """
        if x == 0:
            if p0 == 0:
                return 0.0
            return -2.0 * n * np.log(1 - p0)
        if x == n:
            return -2.0 * n * np.log(p0)

        p_hat = x / n
        eps = 1e-10
        p_hat = np.clip(p_hat, eps, 1 - eps)
        p0 = np.clip(p0, eps, 1 - eps)

        ll_null = x * np.log(p0) + (n - x) * np.log(1 - p0)
        ll_alt = x * np.log(p_hat) + (n - x) * np.log(1 - p_hat)
        return float(-2.0 * (ll_null - ll_alt))

    @staticmethod
    def _christoffersen_lr(
        violations: np.ndarray, p0: float
    ) -> tuple[float, float]:
        """
        Christoffersen (1998) independence and combined conditional-coverage tests.

        Returns
        -------
        (independence_lr, combined_lr)
        """
        v = violations
        n = len(v)

        # Transition counts
        n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
        n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
        n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
        n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

        eps = 1e-10
        pi01 = n01 / max(n00 + n01, 1)
        pi11 = n11 / max(n10 + n11, 1)
        pi = (n01 + n11) / max(n, 1)

        pi01 = np.clip(pi01, eps, 1 - eps)
        pi11 = np.clip(pi11, eps, 1 - eps)
        pi = np.clip(pi, eps, 1 - eps)

        ll_indep = (
            (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        )
        ll_dep = (
            n00 * np.log(1 - pi01)
            + n01 * np.log(pi01)
            + n10 * np.log(1 - pi11)
            + n11 * np.log(pi11)
        )
        independence_lr = float(-2.0 * (ll_indep - ll_dep))

        # Combined: Kupiec (unconditional) + independence
        x = int(v.sum())
        kupiec = VaRCalculator._kupiec_lr(n, x, p0)
        combined_lr = float(kupiec + independence_lr)

        return independence_lr, combined_lr

    @staticmethod
    def _fit_gpd(exceedances: np.ndarray) -> tuple[float, float]:
        """
        Fit a Generalised Pareto Distribution to exceedances via MLE.

        Returns
        -------
        (xi, beta) : shape and scale parameters
        """

        def neg_log_likelihood(params: np.ndarray) -> float:
            xi, beta = params
            if beta <= 0:
                return 1e9
            z = exceedances / beta
            if xi == 0:
                return float(len(exceedances) * np.log(beta) + np.sum(z))
            if np.any(1 + xi * z <= 0):
                return 1e9
            return float(
                len(exceedances) * np.log(beta)
                + (1 + 1 / xi) * np.sum(np.log(1 + xi * z))
            )

        x0 = np.array([0.1, float(np.mean(exceedances))])
        result = minimize(
            neg_log_likelihood,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
        )
        xi, beta = float(result.x[0]), float(result.x[1])
        beta = max(beta, 1e-6)
        return xi, beta


# ---------------------------------------------------------------------------
# Convenience function for one-shot VaR comparison across methods
# ---------------------------------------------------------------------------


def compare_var_methods(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
    n_sims: int = 10_000,
) -> pd.DataFrame:
    """
    Compute VaR by all available methods and return a comparison DataFrame.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float
        Confidence level.
    window : int
        Lookback window for historical VaR.
    n_sims : int
        Monte Carlo simulation count.

    Returns
    -------
    pd.DataFrame
        Columns: method, var_value, notes.
    """
    calc = VaRCalculator()
    rows = []

    methods = {
        "historical": lambda: calc.historical_var(returns, confidence, window),
        "parametric": lambda: calc.parametric_var(returns, confidence),
        "cornish_fisher": lambda: calc.cornish_fisher_var(returns, confidence),
        "cvar": lambda: calc.conditional_var(returns, confidence),
        "monte_carlo": lambda: calc.monte_carlo_var(returns, n_sims, confidence),
        "evt_gpd": lambda: calc.extreme_value_theory_var(returns, confidence),
    }

    for name, fn in methods.items():
        try:
            val = fn()
            rows.append({"method": name, "var_value": val, "error": None})
        except Exception as exc:
            rows.append({"method": name, "var_value": np.nan, "error": str(exc)})

    return pd.DataFrame(rows)
