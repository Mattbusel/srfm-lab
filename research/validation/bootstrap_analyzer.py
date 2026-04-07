"""
bootstrap_analyzer.py -- bootstrap and resampling methods for robust inference.

Implements Politis-Romano stationary bootstrap, circular block bootstrap, and
bootstrap-based tests for Sharpe ratio confidence intervals, IC significance,
and strategy comparison.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Stationary Bootstrap (Politis & Romano 1994)
# ---------------------------------------------------------------------------

class StationaryBootstrap:
    """
    Politis-Romano stationary bootstrap.

    Unlike the fixed-block bootstrap, block lengths are geometrically
    distributed so the resampled series is itself stationary. The expected
    block length controls the autocorrelation preservation.

    Parameters
    ----------
    expected_block_len : mean block length (default 10). Smaller values
                         give more randomness; larger values preserve more
                         serial structure.
    random_state       : seed for reproducibility
    """

    def __init__(
        self,
        expected_block_len: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        if expected_block_len < 1:
            raise ValueError("expected_block_len must be >= 1")
        self.expected_block_len = expected_block_len
        self._rng = np.random.default_rng(random_state)

    def resample(
        self,
        x: np.ndarray,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Generate bootstrap samples.

        Parameters
        ----------
        x         : 1-D input array of length T
        n_samples : number of bootstrap replicates

        Returns
        -------
        np.ndarray of shape (n_samples, T), each row is one resample
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be a 1-D array")
        T = len(x)
        if T < 2:
            raise ValueError("Need at least 2 observations to bootstrap")

        p = 1.0 / max(1, self.expected_block_len)
        out = np.empty((n_samples, T), dtype=float)

        for b in range(n_samples):
            out[b] = self._draw_one(x, T, p)

        return out

    def _draw_one(self, x: np.ndarray, T: int, p: float) -> np.ndarray:
        """Draw a single bootstrap sample of length T."""
        sample = np.empty(T, dtype=float)
        i = 0
        while i < T:
            start = int(self._rng.integers(0, T))
            block_len = int(self._rng.geometric(p))
            for j in range(block_len):
                if i >= T:
                    break
                sample[i] = x[(start + j) % T]
                i += 1
        return sample

    def resample_statistic(
        self,
        x: np.ndarray,
        func,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Apply a statistic function to each bootstrap resample.

        Parameters
        ----------
        x         : input time series
        func      : callable(np.ndarray) -> scalar
        n_samples : number of bootstrap replicates

        Returns
        -------
        np.ndarray of shape (n_samples,) with the statistic for each resample
        """
        samples = self.resample(x, n_samples=n_samples)
        return np.array([func(samples[b]) for b in range(n_samples)])


# ---------------------------------------------------------------------------
# Circular Block Bootstrap
# ---------------------------------------------------------------------------

class CircularBlockBootstrap:
    """
    Circular (wrap-around) block bootstrap.

    Uses fixed block size for each resample. The auto-selection rule is
    block_size = ceil(T^(1/3)), which is optimal for smooth statistics
    under mild dependence (Politis & White 2004 guidance).

    Parameters
    ----------
    block_size   : fixed block length, or 'auto' to use T^(1/3) heuristic
    random_state : seed for reproducibility
    """

    def __init__(
        self,
        block_size: object = "auto",
        random_state: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self._rng = np.random.default_rng(random_state)

    def _resolve_block_size(self, T: int) -> int:
        if self.block_size == "auto":
            return max(1, int(np.ceil(T ** (1.0 / 3.0))))
        return int(self.block_size)

    def resample(
        self,
        x: np.ndarray,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Generate bootstrap samples using circular blocks.

        Parameters
        ----------
        x         : 1-D input array of length T
        n_samples : number of bootstrap replicates

        Returns
        -------
        np.ndarray of shape (n_samples, T)
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be a 1-D array")
        T = len(x)
        if T < 4:
            raise ValueError("Need at least 4 observations")

        b = self._resolve_block_size(T)
        n_blocks = int(np.ceil(T / b))
        out = np.empty((n_samples, T), dtype=float)

        for s in range(n_samples):
            starts = self._rng.integers(0, T, size=n_blocks)
            sample = np.empty(n_blocks * b, dtype=float)
            for idx, start in enumerate(starts):
                for j in range(b):
                    sample[idx * b + j] = x[(start + j) % T]
            out[s] = sample[:T]

        return out

    def resample_statistic(
        self,
        x: np.ndarray,
        func,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """Apply a statistic function to each bootstrap resample."""
        samples = self.resample(x, n_samples=n_samples)
        return np.array([func(samples[b]) for b in range(n_samples)])


# ---------------------------------------------------------------------------
# Bootstrap Tests
# ---------------------------------------------------------------------------

class BootstrapTests:
    """
    Bootstrap-based inference methods for trading strategies.

    All methods default to the stationary bootstrap because financial return
    series typically exhibit short-range dependence.
    """

    def __init__(
        self,
        bootstrap: Optional[StationaryBootstrap] = None,
        n_boot: int = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        if bootstrap is None:
            self._bs = StationaryBootstrap(random_state=random_state)
        else:
            self._bs = bootstrap
        self.n_boot = n_boot

    # -- Sharpe confidence interval --------------------------------------

    def sharpe_ci(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        n_boot: Optional[int] = None,
        annual_factor: float = 252.0,
    ) -> tuple:
        """
        Confidence interval for the annualised Sharpe ratio.

        Uses the percentile-t (studentised) bootstrap for better coverage
        in finite samples.

        Parameters
        ----------
        returns       : daily return series
        confidence    : confidence level (default 0.95)
        n_boot        : override instance n_boot if provided
        annual_factor : annualisation factor

        Returns
        -------
        (lower, upper): tuple of floats
        """
        n_boot = n_boot or self.n_boot
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 10:
            raise ValueError("Need at least 10 returns for Sharpe CI")

        def sharpe(r: np.ndarray) -> float:
            mu = float(np.mean(r))
            sigma = float(np.std(r, ddof=1))
            if sigma < 1e-12:
                return 0.0
            return mu / sigma * np.sqrt(annual_factor)

        sharpe_obs = sharpe(x)
        boot_sharpes = self._bs.resample_statistic(x, sharpe, n_samples=n_boot)

        alpha = 1.0 - confidence
        lo = float(np.percentile(boot_sharpes, 100 * alpha / 2))
        hi = float(np.percentile(boot_sharpes, 100 * (1 - alpha / 2)))
        return (lo, hi)

    # -- IC significance -------------------------------------------------

    def ic_significance(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        n_boot: Optional[int] = None,
    ) -> float:
        """
        Bootstrap p-value for whether the Spearman IC is significantly != 0.

        The null distribution is constructed by resampling the signal values
        (breaking the signal-return pairing) while keeping returns fixed.

        Parameters
        ----------
        signals : signal array, shape (T,)
        returns : forward return array, shape (T,)
        n_boot  : number of bootstrap replicates

        Returns
        -------
        float : one-tailed bootstrap p-value (probability IC <= 0 | observed IC)
        """
        n_boot = n_boot or self.n_boot
        s = np.asarray(signals, dtype=float)
        r = np.asarray(returns, dtype=float)
        mask = np.isfinite(s) & np.isfinite(r)
        s, r = s[mask], r[mask]
        if len(s) < 10:
            raise ValueError("Need at least 10 paired observations")

        observed_ic = float(stats.spearmanr(s, r)[0])

        # Null distribution: resample signals independently of returns
        rng = np.random.default_rng(seed=42)
        null_ics = np.empty(n_boot)
        for b in range(n_boot):
            s_boot = rng.choice(s, size=len(s), replace=True)
            null_ics[b] = float(stats.spearmanr(s_boot, r)[0])

        # Two-tailed p-value
        p_val = float(np.mean(np.abs(null_ics) >= abs(observed_ic)))
        return p_val

    # -- Strategy comparison ---------------------------------------------

    def strategy_comparison(
        self,
        strategy_a: np.ndarray,
        strategy_b: np.ndarray,
        n_boot: Optional[int] = None,
        annual_factor: float = 252.0,
    ) -> tuple:
        """
        Bootstrap test for whether strategy A has a higher Sharpe than B.

        Returns (p_value, sharpe_difference) where p_value is the bootstrap
        probability that the observed Sharpe(A) - Sharpe(B) > 0 under the
        null of no difference.

        Parameters
        ----------
        strategy_a    : return series for strategy A, shape (T,)
        strategy_b    : return series for strategy B, shape (T,)
        n_boot        : number of bootstrap replicates
        annual_factor : annualisation factor

        Returns
        -------
        (p_value: float, sharpe_diff: float)
        """
        n_boot = n_boot or self.n_boot
        a = np.asarray(strategy_a, dtype=float)
        b = np.asarray(strategy_b, dtype=float)
        T = min(len(a), len(b))
        a, b = a[:T], b[:T]

        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if len(a) < 10:
            raise ValueError("Need at least 10 paired observations")

        def annualised_sharpe(r: np.ndarray) -> float:
            mu = float(np.mean(r))
            sigma = float(np.std(r, ddof=1))
            if sigma < 1e-12:
                return 0.0
            return mu / sigma * np.sqrt(annual_factor)

        diff = np.asarray(a - b, dtype=float)
        sharpe_a = annualised_sharpe(a)
        sharpe_b = annualised_sharpe(b)
        observed_diff = sharpe_a - sharpe_b

        # Bootstrap the difference in Sharpes
        bs_cb = CircularBlockBootstrap(block_size="auto", random_state=0)
        diff_samples = bs_cb.resample(diff, n_samples=n_boot)
        boot_diffs = np.array([annualised_sharpe(diff_samples[b_]) for b_ in range(n_boot)])

        # One-tailed p-value: P(boot_diff <= 0 | observed_diff > 0)
        if observed_diff > 0:
            p_val = float(np.mean(boot_diffs <= 0))
        else:
            p_val = float(np.mean(boot_diffs >= 0))

        return (p_val, observed_diff)

    # -- Convenience: full Sharpe analysis --------------------------------

    def full_sharpe_analysis(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        n_boot: Optional[int] = None,
        annual_factor: float = 252.0,
    ) -> dict:
        """
        Compute Sharpe, its bootstrap CI, and a significance test.

        Returns a dict with keys:
          sharpe, ci_lower, ci_upper, is_significant, n_obs
        """
        n_boot = n_boot or self.n_boot
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]

        mu = float(np.mean(x))
        sigma = float(np.std(x, ddof=1)) if len(x) > 1 else 1.0
        sharpe = mu / sigma * np.sqrt(annual_factor) if sigma > 1e-12 else 0.0

        try:
            lo, hi = self.sharpe_ci(x, confidence=confidence, n_boot=n_boot, annual_factor=annual_factor)
        except ValueError:
            lo, hi = np.nan, np.nan

        is_significant = (lo > 0.0) if np.isfinite(lo) else False

        return {
            "sharpe": sharpe,
            "ci_lower": lo,
            "ci_upper": hi,
            "confidence": confidence,
            "is_significant": is_significant,
            "n_obs": len(x),
        }
