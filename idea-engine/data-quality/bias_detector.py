"""
BiasDetector
============
Statistical bias detection for backtests and hypothesis research.

Checks performed:
  1. Survivorship bias — coins present now that didn't exist at backtest start.
  2. Selection bias   — p-value inflation from testing many hypotheses.
  3. Overfitting score — Probabilistic Sharpe Ratio (Bailey-Lopez-de Prado).
  4. Data-snooping penalty — Benjamini-Hochberg corrected significance threshold.
  5. Minimum backtest length — minimum bars required to be meaningful.

Reference:
  Bailey, D.H. & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
  Journal of Risk, 15(2), pp. 3-44.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as scipy_stats
from scipy.special import ndtr  # standard normal CDF, same as Φ

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BiasReport:
    """Aggregated result from :meth:`BiasDetector.survivorship_bias_check`."""

    universe_now: int              # symbols in current universe
    universe_at_start: int         # symbols available at backtest start date
    n_new_symbols: int             # symbols not available at start
    survivorship_pct: float        # fraction of current universe missing at start
    is_biased: bool
    description: str
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe_now": self.universe_now,
            "universe_at_start": self.universe_at_start,
            "n_new_symbols": self.n_new_symbols,
            "survivorship_pct": self.survivorship_pct,
            "is_biased": self.is_biased,
            "description": self.description,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# BiasDetector
# ---------------------------------------------------------------------------

class BiasDetector:
    """
    Provides statistical tools for detecting various forms of backtest bias.

    Parameters
    ----------
    survivorship_threshold : fraction of new symbols above which survivorship
                             bias is flagged (default 0.20 = 20 %).
    skewness_sr : assumed return skewness for PSR computation (default 0).
    kurtosis_sr : assumed excess return kurtosis for PSR (default 0).
    """

    def __init__(
        self,
        survivorship_threshold: float = 0.20,
        skewness_sr: float = 0.0,
        kurtosis_sr: float = 0.0,
    ) -> None:
        self.survivorship_threshold = survivorship_threshold
        self.skewness_sr = skewness_sr
        self.kurtosis_sr = kurtosis_sr

    # ------------------------------------------------------------------
    # 1. Survivorship bias
    # ------------------------------------------------------------------

    def survivorship_bias_check(
        self,
        universe_now: list[str],
        universe_at_backtest_start: list[str],
    ) -> BiasReport:
        """
        Check how many coins in *universe_now* did not exist at backtest start.

        A universe that has grown substantially since the backtest start means
        the backtest looked only at "survivors" — which inflates results.

        Parameters
        ----------
        universe_now : list of symbols currently traded/available.
        universe_at_backtest_start : list of symbols available at the earliest
                                     bar of the backtest period.
        """
        set_now = set(universe_now)
        set_start = set(universe_at_backtest_start)

        # Symbols in current universe that did not exist at backtest start
        new_symbols = set_now - set_start
        n_new = len(new_symbols)
        n_now = len(set_now)
        n_start = len(set_start)
        survivorship_pct = n_new / n_now if n_now > 0 else 0.0

        is_biased = survivorship_pct > self.survivorship_threshold

        if is_biased:
            desc = (
                f"Survivorship bias detected: {n_new}/{n_now} symbols "
                f"({survivorship_pct:.1%}) in the current universe did not exist "
                f"at backtest start. Results likely overstated."
            )
        else:
            desc = (
                f"Survivorship bias appears manageable: {n_new}/{n_now} new symbols "
                f"({survivorship_pct:.1%}) — below {self.survivorship_threshold:.0%} threshold."
            )

        recs: list[str] = []
        if is_biased:
            recs.append(
                "Re-run the backtest using only symbols that existed at the start date."
            )
            recs.append(
                "Use a point-in-time universe constructed from historical listing dates."
            )
            recs.append(
                "Apply a delisted-coin penalty: assume delisted assets went to zero."
            )

        logger.info(desc)
        return BiasReport(
            universe_now=n_now,
            universe_at_start=n_start,
            n_new_symbols=n_new,
            survivorship_pct=survivorship_pct,
            is_biased=is_biased,
            description=desc,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # 2. Selection bias / p-value inflation
    # ------------------------------------------------------------------

    def selection_bias_check(
        self,
        hypotheses_tested: int,
        hypotheses_total: int,
    ) -> float:
        """
        Estimate the p-value inflation factor from testing *hypotheses_tested*
        out of *hypotheses_total* possible hypotheses.

        The naive p-value must be divided by this factor (or equivalently, a
        Bonferroni correction applied) to account for multiple comparisons.

        Parameters
        ----------
        hypotheses_tested : number of hypotheses actually evaluated.
        hypotheses_total  : total number of hypotheses in the search space
                            (including those not explicitly tested).

        Returns
        -------
        float : inflation factor.  A value of 5 means the effective p-value
                threshold is 1/5th of the nominal level.
        """
        if hypotheses_tested <= 0 or hypotheses_total <= 0:
            raise ValueError("Both arguments must be positive integers.")

        # Simple Bonferroni-style inflation: testing M hypotheses inflates
        # false positives by a factor of M.
        bonferroni_factor = float(hypotheses_tested)

        # If only a subset was tested, the "hidden" comparisons still inflate.
        # We scale proportionally: if tested/total = 0.1, the hidden factor is
        # log(total/tested) (a common heuristic).
        hidden_factor = math.log(max(hypotheses_total / hypotheses_tested, 1.0)) + 1.0

        inflation = bonferroni_factor * hidden_factor

        logger.info(
            "Selection bias: %d tested / %d total → inflation factor %.2f",
            hypotheses_tested, hypotheses_total, inflation,
        )
        return round(inflation, 4)

    # ------------------------------------------------------------------
    # 3. Overfitting score — Probabilistic Sharpe Ratio
    # ------------------------------------------------------------------

    def overfitting_score(
        self,
        n_params: int,
        n_trades: int,
        sharpe: float,
        benchmark_sharpe: float = 0.0,
    ) -> float:
        """
        Compute the Probabilistic Sharpe Ratio (PSR) — the probability that
        the true Sharpe ratio exceeds *benchmark_sharpe* given *n_trades*
        observations.

        Based on Bailey & Lopez de Prado (2012).

        PSR = Φ( (SR̂ - SR*) × √(n-1) / √(1 - γ₃ SR̂ + (γ₄ - 1)/4 × SR̂²) )

        where:
          SR̂  = observed Sharpe ratio
          SR* = benchmark Sharpe ratio
          n   = number of trades (observations)
          γ₃  = skewness of returns
          γ₄  = excess kurtosis of returns
          Φ   = standard normal CDF

        Parameters
        ----------
        n_params        : number of free parameters in the strategy.
        n_trades        : number of completed trades.
        sharpe          : annualised Sharpe ratio of the strategy.
        benchmark_sharpe: the null hypothesis Sharpe (default 0).

        Returns
        -------
        float : PSR in [0, 1].  Values above 0.95 are generally considered
                statistically meaningful.  The PSR is penalised implicitly
                through the finite-sample variance correction.
        """
        if n_trades < 2:
            logger.warning("PSR requires at least 2 trades; returning 0.")
            return 0.0

        sr_hat = sharpe
        sr_star = benchmark_sharpe
        n = n_trades
        gamma3 = self.skewness_sr
        gamma4 = self.kurtosis_sr   # excess kurtosis

        # Variance of the Sharpe ratio estimator (finite-sample correction)
        var_numerator = max(
            1.0 - gamma3 * sr_hat + (gamma4 - 1.0) / 4.0 * sr_hat ** 2,
            1e-12,
        )

        z = (sr_hat - sr_star) * math.sqrt(n - 1) / math.sqrt(var_numerator)
        psr = float(ndtr(z))

        # Additional penalty for n_params relative to n_trades
        # (degrees-of-freedom shrinkage)
        dof_ratio = max(n_trades - n_params, 1) / max(n_trades, 1)
        deflated_psr = psr * dof_ratio

        logger.info(
            "PSR: SR=%.3f n_trades=%d n_params=%d → PSR=%.4f deflated=%.4f",
            sharpe, n_trades, n_params, psr, deflated_psr,
        )
        return round(deflated_psr, 6)

    def deflated_sharpe_ratio(
        self,
        sharpe: float,
        n_trials: int,
        n_obs: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Compute the Deflated Sharpe Ratio (DSR) — adjusts the observed Sharpe
        for the number of trials performed, using the expected maximum of *n_trials*
        i.i.d. standard normal random variables as the benchmark.

        DSR = PSR( SR_expected_max )

        Parameters
        ----------
        sharpe     : observed Sharpe ratio.
        n_trials   : number of strategies / parameter sets evaluated.
        n_obs      : number of observations (trades or bars).
        skewness   : return distribution skewness.
        kurtosis   : return distribution kurtosis (total, not excess).

        Returns
        -------
        float : DSR in [0, 1].
        """
        if n_trials < 1 or n_obs < 2:
            return 0.0

        # Expected maximum of n_trials standard normals (approximation)
        # E[max(Z₁...Zₙ)] ≈ (1 - γ) × Φ⁻¹(1 - 1/n) + γ × Φ⁻¹(1 - 1/(n×e))
        # Simplified: use the harmonic-series approximation
        euler_gamma = 0.5772156649
        e_max = (
            (1.0 - euler_gamma) * scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
            + euler_gamma * scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )

        # Scale by trial standard deviation (1/√n_obs)
        sr_star = e_max / math.sqrt(n_obs)

        excess_kurt = kurtosis - 3.0
        var_num = max(
            1.0 - skewness * sharpe + (excess_kurt / 4.0) * sharpe ** 2,
            1e-12,
        )
        z = (sharpe - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(var_num)
        dsr = float(ndtr(z))

        logger.info(
            "DSR: SR=%.3f SR_benchmark=%.3f n_trials=%d n_obs=%d → DSR=%.4f",
            sharpe, sr_star, n_trials, n_obs, dsr,
        )
        return round(dsr, 6)

    # ------------------------------------------------------------------
    # 4. Data-snooping penalty — Benjamini-Hochberg
    # ------------------------------------------------------------------

    def data_snooping_penalty(
        self,
        n_hypotheses_tested: int,
        fdr_level: float = 0.05,
    ) -> float:
        """
        Compute the Benjamini-Hochberg (BH) corrected significance threshold
        for *n_hypotheses_tested* simultaneous tests.

        BH controls the False Discovery Rate (FDR) — the expected proportion
        of rejected null hypotheses that are false discoveries — at level *fdr_level*.

        The corrected threshold for the k-th ranked p-value is:
            α_BH = k / m × q

        where m = n_hypotheses_tested and q = fdr_level.

        This method returns the *minimum* corrected significance level (k=1),
        which is the tightest bound (equivalent to Bonferroni / m).

        Parameters
        ----------
        n_hypotheses_tested : total number of simultaneous hypothesis tests.
        fdr_level           : target FDR (default 0.05).

        Returns
        -------
        float : adjusted significance threshold.  Use this as your alpha
                instead of 0.05 when testing multiple hypotheses.
        """
        if n_hypotheses_tested <= 0:
            raise ValueError("n_hypotheses_tested must be a positive integer.")

        # BH threshold for the 1st-ranked p-value (most conservative)
        bh_threshold = fdr_level / n_hypotheses_tested

        logger.info(
            "BH correction: m=%d tests, FDR=%.3f → α_BH=%.6f",
            n_hypotheses_tested, fdr_level, bh_threshold,
        )
        return round(bh_threshold, 8)

    def bh_reject_hypotheses(
        self,
        p_values: list[float],
        fdr_level: float = 0.05,
    ) -> list[bool]:
        """
        Apply Benjamini-Hochberg procedure to a list of p-values.

        Returns a boolean list of the same length — True means "reject H₀"
        (i.e. the result is significant after FDR correction).

        Parameters
        ----------
        p_values  : list of p-values, one per hypothesis.
        fdr_level : target FDR level.
        """
        m = len(p_values)
        if m == 0:
            return []

        # Sort by p-value, keep track of original positions
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        reject = [False] * m

        for rank, (orig_idx, p) in enumerate(indexed, start=1):
            threshold = fdr_level * rank / m
            if p <= threshold:
                reject[orig_idx] = True
            else:
                # All remaining p-values are larger, so they also fail
                break

        return reject

    # ------------------------------------------------------------------
    # 5. Minimum backtest length
    # ------------------------------------------------------------------

    def minimum_backtest_length(
        self,
        n_params: int,
        target_sharpe: float = 1.5,
        confidence: float = 0.95,
        annual_bars: int = 252,
    ) -> int:
        """
        Compute the minimum number of bars (periods) required for a backtest
        to be statistically meaningful given *n_params* free parameters and
        a *target_sharpe*.

        Derivation follows from the requirement that the PSR ≥ *confidence*
        under the null hypothesis SR = 0:

            n ≥ 1 + (Z_conf / SR_target)² × (1 - γ₃ SR + (γ₄-1)/4 × SR²)

        where Z_conf = Φ⁻¹(confidence) and the variance term accounts for
        higher moments of the return distribution.

        An additional multiplier of *n_params* is applied to account for
        degrees-of-freedom consumed by fitting.

        Parameters
        ----------
        n_params      : number of free parameters in the strategy.
        target_sharpe : per-bar Sharpe ratio the strategy aims to achieve.
                        Note: this is NOT the annualised Sharpe.
        confidence    : desired PSR level (default 0.95).
        annual_bars   : bars per year, used only to convert to years in the
                        log output.

        Returns
        -------
        int : minimum number of bars required.
        """
        if target_sharpe <= 0:
            raise ValueError("target_sharpe must be positive.")

        z_conf = scipy_stats.norm.ppf(confidence)
        gamma3 = self.skewness_sr
        gamma4 = self.kurtosis_sr

        # Variance correction term
        var_term = max(
            1.0 - gamma3 * target_sharpe
            + (gamma4 - 1.0) / 4.0 * target_sharpe ** 2,
            1e-12,
        )

        # Base minimum observations (treating each bar as one observation)
        n_base = 1.0 + (z_conf / target_sharpe) ** 2 * var_term

        # Inflate by n_params for degrees-of-freedom usage
        n_required = int(math.ceil(n_base * max(n_params, 1)))

        years = n_required / annual_bars
        logger.info(
            "Min backtest length: n_params=%d target_SR=%.2f confidence=%.0f%% "
            "→ %d bars (%.1f years at %d bars/yr)",
            n_params, target_sharpe, confidence * 100,
            n_required, years, annual_bars,
        )
        return n_required

    # ------------------------------------------------------------------
    # Composite report
    # ------------------------------------------------------------------

    def full_bias_report(
        self,
        *,
        universe_now: list[str] | None = None,
        universe_at_start: list[str] | None = None,
        n_hypotheses: int = 1,
        n_params: int = 1,
        n_trades: int = 0,
        sharpe: float = 0.0,
        target_sharpe: float = 1.5,
    ) -> dict[str, Any]:
        """
        Run all available bias checks and return a summary dict.

        All parameters have sensible defaults so callers can pass only what
        they have available.
        """
        report: dict[str, Any] = {}

        if universe_now is not None and universe_at_start is not None:
            sb = self.survivorship_bias_check(universe_now, universe_at_start)
            report["survivorship_bias"] = sb.to_dict()

        if n_hypotheses > 1:
            report["selection_bias_inflation"] = self.selection_bias_check(
                n_hypotheses, n_hypotheses
            )
            report["bh_significance_threshold"] = self.data_snooping_penalty(
                n_hypotheses
            )

        if n_trades > 0 and sharpe != 0.0:
            report["probabilistic_sharpe_ratio"] = self.overfitting_score(
                n_params, n_trades, sharpe
            )
            report["deflated_sharpe_ratio"] = self.deflated_sharpe_ratio(
                sharpe, n_hypotheses, n_trades
            )

        report["minimum_backtest_bars"] = self.minimum_backtest_length(
            n_params, target_sharpe
        )

        return report

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_effective_n(
        returns: "np.ndarray",
        block_size: int = 10,
    ) -> int:
        """
        Estimate effective sample size using block-bootstrap logic.

        Autocorrelated returns reduce the effective N below the raw bar count.
        We estimate via the ratio of block variance to IID variance.

        Parameters
        ----------
        returns    : 1-D array of period returns.
        block_size : number of bars per block.

        Returns
        -------
        int : estimated effective number of independent observations.
        """
        import numpy as np

        n = len(returns)
        if n < block_size * 2:
            return n

        # Variance of block means
        n_blocks = n // block_size
        blocks = returns[: n_blocks * block_size].reshape(n_blocks, block_size)
        block_means = blocks.mean(axis=1)

        var_block = np.var(block_means, ddof=1) if len(block_means) > 1 else np.var(returns)
        var_iid = np.var(returns, ddof=1) / block_size

        if var_iid == 0:
            return n

        ratio = var_block / var_iid
        effective_n = int(n / max(ratio, 1.0))
        return max(effective_n, 1)
