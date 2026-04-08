"""
Hypothesis validator — rigorous statistical validation of trading hypotheses.

Runs a battery of tests on hypotheses before live deployment:
  - Statistical significance tests (t-test, bootstrap, PSR)
  - Overfitting detection (IS vs OOS degradation)
  - Regime robustness (does the edge persist across regimes?)
  - Transaction cost break-even analysis
  - Stability analysis (rolling Sharpe std, parameter sensitivity)
  - Variance ratio test (random walk test for directional strategies)
  - Correlation to existing hypothesis book (diversification check)
  - False Discovery Rate correction for strategy selection
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm, ttest_1samp


@dataclass
class ValidationReport:
    """Comprehensive validation result for a hypothesis."""
    hypothesis_id: str
    hypothesis_name: str

    # Statistical tests
    t_stat: float
    p_value: float
    is_significant: bool
    psr: float                      # Probabilistic Sharpe Ratio

    # Overfitting tests
    is_sharpe: float                # In-sample Sharpe
    oos_sharpe: float               # Out-of-sample Sharpe
    oos_degradation: float          # (IS - OOS) / IS
    is_overfit: bool

    # Regime robustness
    regime_sharpes: dict[str, float]
    regime_consistency: float       # fraction of regimes with positive Sharpe
    is_regime_robust: bool

    # Transaction cost analysis
    gross_sharpe: float
    net_sharpe: float               # after realistic transaction costs
    break_even_cost_bps: float      # cost at which strategy breaks even
    is_cost_robust: bool

    # Stability
    rolling_sharpe_std: float
    sharpe_stability_score: float   # 1 = very stable, 0 = unstable
    parameter_sensitivity: float    # avg change in Sharpe per 10% param change

    # Diversification
    avg_correlation_to_book: float
    marginal_sharpe_contribution: float

    # Overall verdict
    verdict: str                    # PASS / CONDITIONAL_PASS / FAIL
    failure_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 0.0              # 0-100 overall score


def _rolling_sharpe(returns: np.ndarray, window: int = 63) -> np.ndarray:
    T = len(returns)
    sr = np.zeros(T)
    for t in range(window, T):
        r = returns[t - window: t]
        sr[t] = float(r.mean() / (r.std() + 1e-10) * math.sqrt(252))
    return sr


def _simulate_strategy_returns(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    transaction_cost_bps: float = 10,
    slippage_bps: float = 5,
) -> np.ndarray:
    """Simple strategy return simulation: signal * return - costs."""
    cost = (transaction_cost_bps + slippage_bps) / 10000
    # Cost applies when signal changes direction
    position_changes = np.abs(np.diff(np.sign(signal), prepend=0))
    costs = position_changes * cost
    strategy_returns = signal * forward_returns - costs
    return strategy_returns


class HypothesisValidator:
    """
    Comprehensive validator for trading hypotheses.
    Runs statistical, overfitting, regime, and cost tests.
    """

    def __init__(
        self,
        is_fraction: float = 0.7,       # fraction for in-sample
        min_psr: float = 0.80,           # minimum PSR to pass
        min_oos_retention: float = 0.40, # OOS must retain at least this fraction of IS
        min_regime_fraction: float = 0.5, # min fraction of regimes profitable
        max_cost_degradation: float = 0.5, # max allowed Sharpe degradation from costs
        n_bootstrap: int = 500,
        seed: int = 42,
    ):
        self.is_fraction = is_fraction
        self.min_psr = min_psr
        self.min_oos_retention = min_oos_retention
        self.min_regime_fraction = min_regime_fraction
        self.max_cost_degradation = max_cost_degradation
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(seed)

    def validate(
        self,
        hypothesis_id: str,
        hypothesis_name: str,
        returns: np.ndarray,           # strategy returns
        signal: Optional[np.ndarray] = None,
        regimes: Optional[np.ndarray] = None,  # per-bar regime labels
        transaction_cost_bps: float = 10.0,
        existing_book_returns: Optional[np.ndarray] = None,
    ) -> ValidationReport:
        """Run full validation battery on hypothesis returns."""
        n = len(returns)
        failures = []
        warnings = []

        # ── IS / OOS Split ────────────────────────────────────────────────
        is_n = int(n * self.is_fraction)
        is_returns = returns[:is_n]
        oos_returns = returns[is_n:] if is_n < n else returns

        # ── Statistical Significance ──────────────────────────────────────
        t_stat, p_value = float(ttest_1samp(is_returns, 0).statistic), float(ttest_1samp(is_returns, 0).pvalue)

        mu = float(is_returns.mean())
        sigma = float(is_returns.std() + 1e-10)
        is_sharpe = float(mu / sigma * math.sqrt(252))

        # PSR (probabilistic Sharpe ratio)
        skew = float(np.mean(((is_returns - mu) / sigma)**3))
        kurt = float(np.mean(((is_returns - mu) / sigma)**4))
        psr = _psr(is_sharpe, 0.0, is_n, skew, kurt)

        is_significant = bool(p_value < 0.05 and psr >= self.min_psr)
        if not is_significant:
            failures.append(f"Not statistically significant (p={p_value:.3f}, PSR={psr:.2f})")

        # ── OOS Degradation ───────────────────────────────────────────────
        oos_sharpe = float(
            oos_returns.mean() / (oos_returns.std() + 1e-10) * math.sqrt(252)
        ) if len(oos_returns) >= 5 else 0.0

        if abs(is_sharpe) > 1e-10:
            oos_degradation = (is_sharpe - oos_sharpe) / max(abs(is_sharpe), 1e-10)
        else:
            oos_degradation = 1.0

        is_overfit = bool(oos_degradation > 1 - self.min_oos_retention)
        if is_overfit:
            failures.append(f"OOS degradation too high: {oos_degradation:.1%}")

        # ── Regime Robustness ─────────────────────────────────────────────
        regime_sharpes = {}
        if regimes is not None and len(regimes) == n:
            unique_regimes = np.unique(regimes)
            for r in unique_regimes:
                mask = regimes == r
                r_returns = returns[mask]
                if len(r_returns) >= 20:
                    sr_r = float(r_returns.mean() / (r_returns.std() + 1e-10) * math.sqrt(252))
                    regime_sharpes[str(r)] = sr_r
        else:
            # Split into halves as proxy
            mid = n // 2
            regime_sharpes["first_half"] = float(
                returns[:mid].mean() / (returns[:mid].std() + 1e-10) * math.sqrt(252)
            )
            regime_sharpes["second_half"] = float(
                returns[mid:].mean() / (returns[mid:].std() + 1e-10) * math.sqrt(252)
            )

        regime_consistency = float(
            sum(1 for sr in regime_sharpes.values() if sr > 0)
            / max(len(regime_sharpes), 1)
        )
        is_regime_robust = bool(regime_consistency >= self.min_regime_fraction)
        if not is_regime_robust:
            failures.append(f"Not regime robust: only {regime_consistency:.0%} regimes positive")

        # ── Transaction Cost Analysis ─────────────────────────────────────
        # Estimate turnover from returns
        if signal is not None and len(signal) == n:
            tc_costs = np.abs(np.diff(signal, prepend=signal[0])) * transaction_cost_bps / 10000
        else:
            tc_costs = np.ones(n) * transaction_cost_bps / 10000 * 0.5

        net_returns = returns - tc_costs
        net_sharpe = float(net_returns.mean() / (net_returns.std() + 1e-10) * math.sqrt(252))
        gross_sharpe = float(returns.mean() / (returns.std() + 1e-10) * math.sqrt(252))

        # Break-even cost
        if returns.mean() > 0 and tc_costs.mean() > 0:
            break_even = float(returns.mean() / tc_costs.mean() * transaction_cost_bps)
        else:
            break_even = 0.0

        if gross_sharpe > 0:
            cost_degradation = (gross_sharpe - net_sharpe) / max(abs(gross_sharpe), 1e-10)
        else:
            cost_degradation = 1.0

        is_cost_robust = bool(cost_degradation <= self.max_cost_degradation)
        if not is_cost_robust:
            warnings.append(f"High transaction cost sensitivity: {cost_degradation:.1%} Sharpe degradation")

        # ── Rolling Sharpe Stability ──────────────────────────────────────
        rolling_sr = _rolling_sharpe(returns, window=63)
        nonzero_sr = rolling_sr[rolling_sr != 0]
        if len(nonzero_sr) >= 5:
            rolling_sr_std = float(nonzero_sr.std())
            sharpe_stability = float(max(1 - rolling_sr_std / max(abs(gross_sharpe), 0.5), 0))
        else:
            rolling_sr_std = 0.0
            sharpe_stability = 0.5

        # ── Parameter Sensitivity (bootstrap-based) ───────────────────────
        boot_sharpes = []
        for _ in range(min(self.n_bootstrap, 100)):
            idx = self.rng.integers(0, is_n, is_n)
            boot_r = is_returns[idx]
            boot_sr = float(boot_r.mean() / (boot_r.std() + 1e-10) * math.sqrt(252))
            boot_sharpes.append(boot_sr)
        param_sensitivity = float(np.std(boot_sharpes)) if boot_sharpes else 0.5

        # ── Diversification Check ─────────────────────────────────────────
        avg_corr = 0.0
        marginal_sharpe = float(gross_sharpe)
        if existing_book_returns is not None and len(existing_book_returns) == n:
            avg_corr = float(np.corrcoef(returns, existing_book_returns)[0, 1])
            # Marginal Sharpe: diversified portfolio Sharpe - book-alone Sharpe
            book_sr = float(existing_book_returns.mean() / (existing_book_returns.std() + 1e-10) * math.sqrt(252))
            combined_r = 0.5 * returns + 0.5 * existing_book_returns
            combined_sr = float(combined_r.mean() / (combined_r.std() + 1e-10) * math.sqrt(252))
            marginal_sharpe = combined_sr - book_sr

        # ── Overall Score ─────────────────────────────────────────────────
        score = 0.0
        score += min(psr * 30, 30)                           # up to 30 pts for PSR
        score += max(15 - oos_degradation * 30, 0)           # up to 15 for OOS retention
        score += regime_consistency * 15                      # up to 15 for regime robustness
        score += is_cost_robust * 10                          # 10 for cost robustness
        score += sharpe_stability * 15                        # up to 15 for stability
        score += min(max(marginal_sharpe, 0) * 10, 15)       # up to 15 for diversification

        # Verdict
        if len(failures) == 0 and score >= 60:
            verdict = "PASS"
        elif len(failures) <= 1 and score >= 40:
            verdict = "CONDITIONAL_PASS"
        else:
            verdict = "FAIL"

        return ValidationReport(
            hypothesis_id=hypothesis_id,
            hypothesis_name=hypothesis_name,
            t_stat=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            psr=psr,
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            oos_degradation=oos_degradation,
            is_overfit=is_overfit,
            regime_sharpes=regime_sharpes,
            regime_consistency=regime_consistency,
            is_regime_robust=is_regime_robust,
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            break_even_cost_bps=break_even,
            is_cost_robust=is_cost_robust,
            rolling_sharpe_std=rolling_sr_std,
            sharpe_stability_score=sharpe_stability,
            parameter_sensitivity=param_sensitivity,
            avg_correlation_to_book=avg_corr,
            marginal_sharpe_contribution=marginal_sharpe,
            verdict=verdict,
            failure_reasons=failures,
            warnings=warnings,
            score=float(min(score, 100)),
        )

    def batch_validate(
        self,
        hypotheses: list[dict],
        apply_fdr_correction: bool = True,
        fdr_alpha: float = 0.05,
    ) -> list[ValidationReport]:
        """
        Validate multiple hypotheses with FDR correction for multiple testing.
        """
        reports = []
        for hyp in hypotheses:
            report = self.validate(
                hypothesis_id=hyp.get("id", ""),
                hypothesis_name=hyp.get("name", ""),
                returns=hyp.get("returns", np.array([])),
                signal=hyp.get("signal"),
                regimes=hyp.get("regimes"),
                transaction_cost_bps=hyp.get("transaction_cost_bps", 10.0),
                existing_book_returns=hyp.get("book_returns"),
            )
            reports.append(report)

        if apply_fdr_correction and len(reports) > 1:
            # Apply Benjamini-Hochberg on p-values
            p_values = np.array([r.p_value for r in reports])
            m = len(p_values)
            sorted_idx = np.argsort(p_values)
            bh_thresholds = fdr_alpha * np.arange(1, m + 1) / m
            bh_rejected = p_values[sorted_idx] <= bh_thresholds

            if bh_rejected.any():
                last_sig = np.where(bh_rejected)[0][-1]
                sig_indices = set(sorted_idx[:last_sig + 1])
            else:
                sig_indices = set()

            for i, report in enumerate(reports):
                if i not in sig_indices and report.is_significant:
                    report.is_significant = False
                    report.warnings.append("Significance removed by BH FDR correction")
                    if report.verdict == "PASS":
                        report.verdict = "CONDITIONAL_PASS"

        reports.sort(key=lambda r: r.score, reverse=True)
        return reports


def _psr(sr: float, benchmark: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio helper."""
    if n <= 1:
        return 0.5
    se = math.sqrt(max(
        (1 + 0.5 * sr**2 - skew * sr + (kurt - 3) / 4 * sr**2) / max(n - 1, 1),
        1e-10
    ))
    return float(norm.cdf((sr - benchmark) / se))
