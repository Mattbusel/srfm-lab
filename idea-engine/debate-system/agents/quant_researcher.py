"""
quant_researcher.py
===================
QuantResearcherAgent — debate-system participant that performs rigorous
quantitative validation of trading hypotheses.

Role in the debate
------------------
The quant researcher acts as the primary statistical sceptic.  It independently
validates hypotheses using multiple statistical lenses:

  1. Significance testing       — t-test, JB normality, Ljung-Box autocorrelation
  2. Multiple-testing awareness — estimates effective number of tests, adjusts
  3. Sample size adequacy       — minimum power analysis
  4. Out-of-sample validity     — walk-forward Sharpe degradation test
  5. Factor decomposition       — Fama-French-style alpha/beta decomposition
  6. Execution feasibility      — capacity estimate, slippage-adjusted Sharpe
  7. Performance analytics      — IR, Calmar, Omega, Sortino, max drawdown, recovery

Final verdict: STRONG_BUY_SIGNAL | BUY_SIGNAL | NEUTRAL | SELL_SIGNAL | REJECT

Dependencies: numpy only (pure Python statistical tests)
"""

from __future__ import annotations

import math
import time
import collections
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Verdict taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    STRONG_BUY_SIGNAL = "STRONG_BUY_SIGNAL"
    BUY_SIGNAL        = "BUY_SIGNAL"
    NEUTRAL           = "NEUTRAL"
    SELL_SIGNAL       = "SELL_SIGNAL"
    REJECT            = "REJECT"


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FactorExposure:
    """Fama-French-like factor exposures for a return series."""
    alpha: float            # annualised intercept (pure alpha)
    beta_market: float      # exposure to market factor
    beta_size: float        # exposure to size (SMB-like)
    beta_value: float       # exposure to value (HML-like)
    beta_momentum: float    # exposure to momentum (WML-like)
    r_squared: float        # in-sample R² of factor model
    residual_std: float     # annualised residual (unexplained) volatility


@dataclass
class PerformanceMetrics:
    """Comprehensive risk-adjusted performance metrics."""
    # Return metrics
    mean_return_ann: float      # annualised mean return
    std_return_ann: float       # annualised volatility
    total_return: float         # cumulative return
    # Risk-adjusted metrics
    information_ratio: float    # (mean excess return) / tracking error
    sharpe_ratio: float         # (mean - rf) / std (annualised)
    sortino_ratio: float        # mean / downside deviation
    calmar_ratio: float         # annualised return / max drawdown
    omega_ratio: float          # probability-weighted gain/loss ratio
    # Drawdown analysis
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  # in periods
    recovery_time: int          # periods to recover from max DD
    # Distribution
    skewness: float
    excess_kurtosis: float
    hit_rate: float             # fraction of positive periods
    # OOS
    oos_sharpe: float
    sharpe_decay: float         # IS Sharpe - OOS Sharpe


@dataclass
class StatisticalTests:
    """Results of key statistical tests."""
    t_stat: float
    t_pvalue: float
    jb_stat: float
    jb_pvalue: float            # JB normality test
    lb_stat: float
    lb_pvalue: float            # Ljung-Box autocorrelation test
    adf_stat: float             # Augmented Dickey-Fuller proxy
    adf_pvalue: float
    n_obs: int
    effective_n_tests: int      # estimated number of tests performed
    bonferroni_threshold: float # adjusted significance threshold


@dataclass
class ExecutionAssessment:
    """Practical execution feasibility analysis."""
    estimated_capacity_usd: float      # rough capacity estimate
    avg_daily_turnover: float          # fraction of portfolio turned over daily
    slippage_bps: float                # assumed one-way slippage
    slippage_adjusted_sharpe: float    # Sharpe after estimated costs
    net_ir: float                      # IR after costs
    feasible: bool                     # is the strategy worth executing?
    capacity_notes: str


@dataclass
class DebatePosition:
    """Full debate position from the QuantResearcherAgent."""
    hypothesis_id: str
    verdict: Verdict
    confidence_score: float          # 0–1
    performance: PerformanceMetrics
    statistics: StatisticalTests
    factor_exposure: FactorExposure
    execution: ExecutionAssessment
    supporting_arguments: list[str]
    objections: list[str]
    suggested_refinements: list[str]
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        p = self.performance
        s = self.statistics
        e = self.execution
        f = self.factor_exposure
        lines = [
            f"╔══ QuantResearcher Verdict: {self.verdict.value} ══",
            f"  Hypothesis  : {self.hypothesis_id}",
            f"  Confidence  : {self.confidence_score:.2f}",
            f"",
            f"  ─ Performance ─",
            f"  Ann. Return   : {p.mean_return_ann:.2%}",
            f"  Ann. Vol      : {p.std_return_ann:.2%}",
            f"  Sharpe (IS)   : {p.sharpe_ratio:.3f}",
            f"  Sharpe (OOS)  : {p.oos_sharpe:.3f}",
            f"  Sharpe decay  : {p.sharpe_decay:.3f}",
            f"  Sortino       : {p.sortino_ratio:.3f}",
            f"  Calmar        : {p.calmar_ratio:.3f}",
            f"  Omega         : {p.omega_ratio:.3f}",
            f"  Max Drawdown  : {p.max_drawdown:.2%}",
            f"  Max DD dur.   : {p.max_drawdown_duration} periods",
            f"  Recovery time : {p.recovery_time} periods",
            f"  Hit rate      : {p.hit_rate:.1%}",
            f"",
            f"  ─ Statistics ─",
            f"  t-stat        : {s.t_stat:.3f}  (p={s.t_pvalue:.4f})",
            f"  JB normality  : stat={s.jb_stat:.2f}  (p={s.jb_pvalue:.4f})",
            f"  Ljung-Box     : stat={s.lb_stat:.2f}  (p={s.lb_pvalue:.4f})",
            f"  N obs         : {s.n_obs}",
            f"  Bonferroni α  : {s.bonferroni_threshold:.4f}  "
              f"(N_tests≈{s.effective_n_tests})",
            f"",
            f"  ─ Factor Decomposition ─",
            f"  Alpha (ann.)  : {f.alpha:.2%}",
            f"  β_market      : {f.beta_market:.3f}",
            f"  β_size        : {f.beta_size:.3f}",
            f"  β_value       : {f.beta_value:.3f}",
            f"  β_momentum    : {f.beta_momentum:.3f}",
            f"  R²            : {f.r_squared:.3f}",
            f"  Residual vol  : {f.residual_std:.2%}",
            f"",
            f"  ─ Execution ─",
            f"  Capacity est. : ${e.estimated_capacity_usd:,.0f}",
            f"  Daily turnover: {e.avg_daily_turnover:.1%}",
            f"  Slippage      : {e.slippage_bps:.1f} bps",
            f"  Net Sharpe    : {e.slippage_adjusted_sharpe:.3f}",
            f"  Feasible      : {'Yes' if e.feasible else 'No'}",
        ]
        if self.supporting_arguments:
            lines += ["", "  ─ Supporting arguments ─"]
            lines += [f"  ✓ {a}" for a in self.supporting_arguments]
        if self.objections:
            lines += ["", "  ─ Objections ─"]
            lines += [f"  ✗ {o}" for o in self.objections]
        if self.suggested_refinements:
            lines += ["", "  ─ Suggested refinements ─"]
            lines += [f"  → {r}" for r in self.suggested_refinements]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Statistical primitives  (pure numpy, no scipy dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _t_pvalue(t: float, df: int) -> float:
    """Approximate two-sided p-value from t-distribution."""
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    p1 = 0.5 * _regularised_ibeta(x, df / 2.0, 0.5)
    return min(1.0, 2.0 * p1)


def _regularised_ibeta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta via continued fraction (Lentz method)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_ibeta(1.0 - x, b, a)
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
             + a * math.log(x) + b * math.log(1.0 - x + 1e-300))
    prefactor = math.exp(lbeta) / a
    fpmin = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return prefactor * h


def _chi2_sf(x: float, df: int) -> float:
    """Chi-squared survival function Q(x, df)."""
    if x <= 0:
        return 1.0
    return _regularised_upper_gamma(df / 2.0, x / 2.0)


def _regularised_upper_gamma(a: float, x: float) -> float:
    if x < a + 1.0:
        return 1.0 - _lower_gamma_series(a, x)
    return _upper_gamma_cf(a, x)


def _lower_gamma_series(a: float, x: float) -> float:
    if x <= 0:
        return 0.0
    ap = a
    delt = 1.0 / a
    total = delt
    for _ in range(300):
        ap += 1.0
        delt *= x / ap
        total += delt
        if abs(delt) < abs(total) * 1e-10:
            break
    return total * math.exp(a * math.log(x + 1e-300) - x - math.lgamma(a + 1.0))


def _upper_gamma_cf(a: float, x: float) -> float:
    fpmin = 1e-30
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / max(abs(b), fpmin)
    if b < 0:
        d = -d
    h = d
    for i in range(1, 301):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return math.exp(a * math.log(x + 1e-300) - x - math.lgamma(a)) * h


def t_test_one_sample(returns: np.ndarray) -> tuple[float, float]:
    """H₀: mean = 0.  Returns (t_stat, two-sided p-value)."""
    n = len(returns)
    if n < 4:
        return 0.0, 1.0
    mean = float(np.mean(returns))
    std  = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0, 1.0
    t = mean / (std / math.sqrt(n))
    return t, _t_pvalue(t, n - 1)


def jarque_bera_test(returns: np.ndarray) -> tuple[float, float]:
    """
    Jarque-Bera normality test.

    JB = n/6 * [S² + (K-3)²/4]  ~  χ²(2) under H₀: normal

    Returns (JB_stat, p_value).
    """
    n = len(returns)
    if n < 8:
        return 0.0, 1.0
    mean = float(np.mean(returns))
    std  = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0, 1.0
    z = (returns - mean) / std
    s = float(np.mean(z ** 3))           # skewness
    k = float(np.mean(z ** 4)) - 3.0    # excess kurtosis
    jb = n / 6.0 * (s ** 2 + k ** 2 / 4.0)
    p  = _chi2_sf(jb, 2)
    return round(jb, 4), round(max(0.0, min(1.0, p)), 6)


def ljung_box_test(returns: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """
    Ljung-Box autocorrelation test.

    Q = n(n+2) Σ_{k=1}^{lags} ρ_k² / (n-k)  ~  χ²(lags) under H₀: no AC

    Returns (Q_stat, p_value).
    """
    n = len(returns)
    if n < lags + 5:
        return 0.0, 1.0
    mean    = float(np.mean(returns))
    demeaned = returns - mean
    var     = float(np.var(demeaned))
    if var < 1e-12:
        return 0.0, 1.0
    q = 0.0
    for k in range(1, lags + 1):
        rho_k = float(np.mean(demeaned[k:] * demeaned[:-k])) / var
        q += rho_k ** 2 / (n - k)
    q *= n * (n + 2)
    p = _chi2_sf(q, lags)
    return round(q, 4), round(max(0.0, min(1.0, p)), 6)


def adf_proxy(returns: np.ndarray) -> tuple[float, float]:
    """
    Simplified ADF-style stationarity proxy.

    Regresses Δr_t on r_{t-1} and uses t-stat on the lag coefficient.
    Negative t-stat = tendency toward stationarity (mean-reversion).

    Returns (adf_stat, approx_pvalue).
    """
    n = len(returns)
    if n < 10:
        return 0.0, 0.5
    dy   = np.diff(returns)
    y_lag = returns[:-1]
    # OLS: dy = γ * y_lag + ε
    cov = float(np.mean((y_lag - y_lag.mean()) * (dy - dy.mean())))
    var = float(np.var(y_lag, ddof=1))
    if var < 1e-12:
        return 0.0, 0.5
    gamma = cov / var
    resid = dy - gamma * y_lag
    se    = float(np.std(resid, ddof=2)) / (math.sqrt(var * (n - 1)) + 1e-12)
    t     = gamma / (se + 1e-12)
    # Very rough p-value (ADF critical values differ; this is illustrative)
    p = _t_pvalue(t, df=n - 2)
    return round(t, 4), round(p, 6)


# ──────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_performance(
    returns: np.ndarray,
    oos_returns: np.ndarray,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    n = len(returns)
    if n < 2:
        return _empty_performance()

    ann = float(periods_per_year)
    mean = float(np.mean(returns))
    std  = float(np.std(returns, ddof=1))

    mean_ann = mean * ann
    std_ann  = std * math.sqrt(ann)

    excess = returns - rf / ann
    sharpe = float(np.mean(excess)) / (float(np.std(excess, ddof=1)) + 1e-12) * math.sqrt(ann)

    # Sortino
    downside = returns[returns < rf / ann] - rf / ann
    dd_std   = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-12
    sortino  = mean_ann / (dd_std * math.sqrt(ann) + 1e-12)

    # Drawdown
    equity = np.cumprod(1.0 + returns)
    equity = np.concatenate([[1.0], equity])
    max_dd, dd_dur, recovery = _drawdown_analysis(equity)

    # Calmar
    calmar = mean_ann / (max_dd + 1e-12)

    # Omega ratio (threshold = 0)
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    omega  = gains / (losses + 1e-12)

    # Distribution
    z = (returns - mean) / (std + 1e-12)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0

    # OOS Sharpe
    oos_sharpe = 0.0
    if len(oos_returns) > 4:
        oos_mean = float(np.mean(oos_returns))
        oos_std  = float(np.std(oos_returns, ddof=1))
        oos_sharpe = oos_mean / (oos_std + 1e-12) * math.sqrt(ann)

    # Average drawdown
    peak = equity[0]
    dd_vals = []
    for v in equity:
        peak = max(peak, v)
        dd_vals.append((peak - v) / (peak + 1e-12))
    avg_dd = float(np.mean(dd_vals))

    return PerformanceMetrics(
        mean_return_ann=round(mean_ann, 6),
        std_return_ann=round(std_ann, 6),
        total_return=round(float(equity[-1]) - 1.0, 6),
        information_ratio=round(sharpe, 4),     # IR ≈ Sharpe vs. cash
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        omega_ratio=round(omega, 4),
        max_drawdown=round(max_dd, 6),
        avg_drawdown=round(avg_dd, 6),
        max_drawdown_duration=dd_dur,
        recovery_time=recovery,
        skewness=round(skew, 4),
        excess_kurtosis=round(kurt, 4),
        hit_rate=round(float(np.mean(returns > 0)), 4),
        oos_sharpe=round(oos_sharpe, 4),
        sharpe_decay=round(sharpe - oos_sharpe, 4),
    )


def _drawdown_analysis(equity: np.ndarray) -> tuple[float, int, int]:
    """Returns (max_drawdown, max_drawdown_duration, recovery_time)."""
    peak = equity[0]
    max_dd = 0.0
    current_dd = 0.0
    current_dd_start = 0
    max_dd_dur = 0
    recovery = 0

    peak_idx = 0
    trough_idx = 0
    max_dd_peak_idx = 0

    for t, v in enumerate(equity):
        if v >= peak:
            peak = v
            peak_idx = t
            current_dd = 0.0
            current_dd_start = t
        else:
            dd = (peak - v) / (peak + 1e-12)
            if dd > max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                trough_idx = t
                max_dd_dur = t - peak_idx

    # Recovery: how many periods after trough to regain peak
    for t in range(trough_idx, len(equity)):
        if equity[t] >= equity[max_dd_peak_idx]:
            recovery = t - trough_idx
            break
    else:
        recovery = len(equity) - trough_idx  # not yet recovered

    return max_dd, max_dd_dur, recovery


def _empty_performance() -> PerformanceMetrics:
    return PerformanceMetrics(
        mean_return_ann=0.0, std_return_ann=0.0, total_return=0.0,
        information_ratio=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
        calmar_ratio=0.0, omega_ratio=1.0, max_drawdown=0.0,
        avg_drawdown=0.0, max_drawdown_duration=0, recovery_time=0,
        skewness=0.0, excess_kurtosis=0.0, hit_rate=0.5,
        oos_sharpe=0.0, sharpe_decay=0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Factor decomposition
# ──────────────────────────────────────────────────────────────────────────────

def _factor_decomposition(
    strategy_returns: np.ndarray,
    factor_returns: np.ndarray | None,
    periods_per_year: int = 252,
) -> FactorExposure:
    """
    Fama-French 4-factor OLS regression (or proxy factors if not supplied).

    factor_returns : (T, 4) matrix [market, SMB, HML, WML] or None
    """
    n = len(strategy_returns)

    if factor_returns is None or factor_returns.shape[0] < n:
        # Generate synthetic orthogonal proxy factors for illustration
        rng = np.random.default_rng(0)
        T_f = n
        raw = rng.standard_normal((T_f, 4)) * 0.01
        # Orthogonalise
        Q, _ = np.linalg.qr(raw)
        factor_returns = Q[:, :4] * 0.01

    T_f = min(n, factor_returns.shape[0])
    Y = strategy_returns[:T_f]
    X = factor_returns[:T_f]   # (T_f, 4)

    # Add intercept
    ones = np.ones((T_f, 1))
    X_aug = np.hstack([ones, X])   # (T_f, 5)

    # OLS: β = (XᵀX)⁻¹ Xᵀy
    try:
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ Y
        beta = np.linalg.solve(XtX + 1e-8 * np.eye(5), Xty)
    except np.linalg.LinAlgError:
        beta = np.zeros(5)

    alpha_daily = float(beta[0])
    alpha_ann   = alpha_daily * periods_per_year

    betas = beta[1:]

    y_hat = X_aug @ beta
    ss_res = float(np.sum((Y - y_hat) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    resid_std_ann = float(np.std(Y - y_hat, ddof=5)) * math.sqrt(periods_per_year)

    return FactorExposure(
        alpha=round(alpha_ann, 6),
        beta_market=round(float(betas[0]), 4),
        beta_size=round(float(betas[1]), 4),
        beta_value=round(float(betas[2]), 4),
        beta_momentum=round(float(betas[3]), 4),
        r_squared=round(max(0.0, r2), 4),
        residual_std=round(resid_std_ann, 6),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Execution assessment
# ──────────────────────────────────────────────────────────────────────────────

def _execution_assessment(
    performance: PerformanceMetrics,
    avg_daily_volume_usd: float = 1e8,
    participation_rate: float = 0.05,
    turnover_rate: float = 0.20,
    periods_per_year: int = 252,
) -> ExecutionAssessment:
    """
    Estimate execution feasibility and capacity.

    Capacity model
    --------------
    Participation rate × daily_volume × sqrt(turnover_rate × T) ≈ rough capacity.
    This is a simplified model; real capacity models are proprietary.

    Slippage model
    --------------
    slippage_bps = k * sqrt(turnover_rate / liquidity_fraction)
    k = 5  (typical empirical constant)
    """
    k_slippage = 5.0
    liquidity_fraction = min(1.0, avg_daily_volume_usd / 1e9)
    slippage_bps = k_slippage * math.sqrt(turnover_rate / (liquidity_fraction + 1e-6))
    slippage_bps = min(slippage_bps, 200.0)

    # Annual cost drag
    slippage_ann = slippage_bps / 1e4 * turnover_rate * periods_per_year
    std_ann = performance.std_return_ann
    slippage_sharpe = (performance.mean_return_ann - slippage_ann) / (std_ann + 1e-12)

    # Rough capacity: fraction of daily volume × trading days / turnover
    capacity = (participation_rate * avg_daily_volume_usd
                / (turnover_rate + 1e-6) * math.sqrt(252))

    net_ir = slippage_sharpe  # simplified

    feasible = (
        slippage_sharpe > 0.3
        and performance.max_drawdown < 0.30
        and capacity > 1e6
    )

    notes = (
        f"Capacity limited by estimated daily volume "
        f"(${avg_daily_volume_usd:,.0f}) and turnover ({turnover_rate:.1%}/day). "
        f"Slippage of {slippage_bps:.1f} bps one-way. "
        f"{'Feasible for deployment.' if feasible else 'Not feasible as-is.'}"
    )

    return ExecutionAssessment(
        estimated_capacity_usd=round(capacity, -3),
        avg_daily_turnover=round(turnover_rate, 4),
        slippage_bps=round(slippage_bps, 2),
        slippage_adjusted_sharpe=round(slippage_sharpe, 4),
        net_ir=round(net_ir, 4),
        feasible=feasible,
        capacity_notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# QuantResearcherAgent
# ──────────────────────────────────────────────────────────────────────────────

class QuantResearcherAgent:
    """
    Quantitative researcher debate agent.

    Evaluates a trading strategy's return series and produces a structured
    DebatePosition with a Verdict, supporting arguments, and objections.

    Parameters
    ----------
    alpha             : significance level (post-correction)
    min_sharpe        : minimum IS Sharpe for BUY_SIGNAL
    max_sharpe_decay  : maximum IS-OOS Sharpe decay before flagging OOS failure
    min_obs           : minimum observations required
    oos_fraction      : fraction of data used for OOS testing
    rf_ann            : risk-free rate (annualised)
    periods_per_year  : trading periods per year
    effective_n_tests : estimated number of parallel tests (for Bonferroni)
    seed              : reproducibility seed
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_sharpe: float = 0.5,
        max_sharpe_decay: float = 0.5,
        min_obs: int = 60,
        oos_fraction: float = 0.30,
        rf_ann: float = 0.04,
        periods_per_year: int = 252,
        effective_n_tests: int = 50,
        seed: int = 42,
    ):
        self.alpha             = alpha
        self.min_sharpe        = min_sharpe
        self.max_sharpe_decay  = max_sharpe_decay
        self.min_obs           = min_obs
        self.oos_fraction      = oos_fraction
        self.rf_ann            = rf_ann
        self.periods_per_year  = periods_per_year
        self.effective_n_tests = effective_n_tests
        self._rng              = np.random.default_rng(seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        hypothesis_id: str,
        strategy_returns: np.ndarray,
        factor_returns: np.ndarray | None = None,
        avg_daily_volume_usd: float = 1e8,
        turnover_rate: float = 0.20,
    ) -> DebatePosition:
        """
        Full quantitative evaluation of a trading strategy.

        Parameters
        ----------
        hypothesis_id         : identifier for the hypothesis under review
        strategy_returns      : (T,) daily (or periodic) strategy return series
        factor_returns        : (T, 4) factor return matrix [mkt, SMB, HML, WML]
                                       or None to use synthetic proxies
        avg_daily_volume_usd  : estimated daily tradeable volume
        turnover_rate         : estimated daily two-way turnover fraction

        Returns
        -------
        DebatePosition with full analysis and verdict
        """
        n = len(strategy_returns)

        # Split IS / OOS
        oos_split = max(int(n * (1.0 - self.oos_fraction)), self.min_obs)
        is_rets   = strategy_returns[:oos_split]
        oos_rets  = strategy_returns[oos_split:]

        if n < self.min_obs:
            return self._insufficient_data_position(hypothesis_id, n)

        # ── Statistical tests ──────────────────────────────────────────────
        t_stat, t_pval   = t_test_one_sample(is_rets)
        jb_stat, jb_pval = jarque_bera_test(is_rets)
        lb_stat, lb_pval = ljung_box_test(is_rets)
        adf_stat, adf_pval = adf_proxy(strategy_returns)

        bonf_alpha = self.alpha / self.effective_n_tests

        stats = StatisticalTests(
            t_stat=round(t_stat, 4),
            t_pvalue=round(t_pval, 6),
            jb_stat=jb_stat,
            jb_pvalue=jb_pval,
            lb_stat=lb_stat,
            lb_pvalue=lb_pval,
            adf_stat=adf_stat,
            adf_pvalue=adf_pval,
            n_obs=n,
            effective_n_tests=self.effective_n_tests,
            bonferroni_threshold=round(bonf_alpha, 6),
        )

        # ── Performance metrics ────────────────────────────────────────────
        perf = _compute_performance(
            is_rets, oos_rets,
            rf=self.rf_ann,
            periods_per_year=self.periods_per_year,
        )

        # ── Factor decomposition ───────────────────────────────────────────
        factor_exp = _factor_decomposition(
            is_rets, factor_returns,
            periods_per_year=self.periods_per_year,
        )

        # ── Execution assessment ───────────────────────────────────────────
        exec_assess = _execution_assessment(
            perf,
            avg_daily_volume_usd=avg_daily_volume_usd,
            turnover_rate=turnover_rate,
            periods_per_year=self.periods_per_year,
        )

        # ── Verdict & reasoning ────────────────────────────────────────────
        verdict, confidence, supports, objections, refinements = self._render_verdict(
            stats, perf, factor_exp, exec_assess
        )

        return DebatePosition(
            hypothesis_id=hypothesis_id,
            verdict=verdict,
            confidence_score=round(confidence, 3),
            performance=perf,
            statistics=stats,
            factor_exposure=factor_exp,
            execution=exec_assess,
            supporting_arguments=supports,
            objections=objections,
            suggested_refinements=refinements,
        )

    def quick_screen(
        self,
        returns: np.ndarray,
    ) -> tuple[float, float, bool]:
        """
        Fast screen: compute Sharpe, p-value, and pass/fail flag.

        Returns (sharpe, p_value, passes_screen).
        """
        if len(returns) < self.min_obs:
            return 0.0, 1.0, False
        t, p = t_test_one_sample(returns)
        sharpe = float(np.mean(returns)) / (float(np.std(returns, ddof=1)) + 1e-12)
        sharpe *= math.sqrt(self.periods_per_year)
        bonf_p = self.alpha / self.effective_n_tests
        passes = p < bonf_p and sharpe > self.min_sharpe
        return round(sharpe, 4), round(p, 6), passes

    def minimum_sample_size(
        self,
        expected_sharpe: float,
        power: float = 0.80,
    ) -> int:
        """
        Estimate minimum sample size for given power.

        Using the approximation: n ≈ (z_α + z_β)² / IR² * periods_per_year
        where z_α = 1.96 (two-tailed α=0.05), z_β = 0.84 (power=0.80).
        """
        if expected_sharpe < 0.01:
            return 10_000  # essentially infinite
        z_alpha = 1.96
        z_beta  = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}.get(power, 0.842)
        daily_ir = expected_sharpe / math.sqrt(self.periods_per_year)
        n = math.ceil(((z_alpha + z_beta) / daily_ir) ** 2)
        return n

    # ── Private helpers ───────────────────────────────────────────────────────

    def _render_verdict(
        self,
        stats: StatisticalTests,
        perf: PerformanceMetrics,
        factor: FactorExposure,
        execution: ExecutionAssessment,
    ) -> tuple[Verdict, float, list[str], list[str], list[str]]:
        """
        Multi-criteria verdict logic.

        Scoring system (each criterion adds/subtracts from base score 0.5):
        +0.20 : t-test significant at Bonferroni-corrected level
        +0.10 : IS Sharpe > min_sharpe
        +0.10 : OOS Sharpe > 0.3
        +0.10 : Sharpe decay < max_sharpe_decay
        +0.05 : Factor alpha > 2%
        +0.05 : Execution feasible
        -0.20 : t-test not significant
        -0.10 : High autocorrelation (overstated Sharpe)
        -0.10 : Severe non-normality (fat tails)
        -0.10 : Excess factor loading (not alpha)
        -0.05 : High drawdown > 25%
        -0.15 : Strong OOS decay (overfitting)
        """
        supports:    list[str] = []
        objections:  list[str] = []
        refinements: list[str] = []

        score = 0.5

        # ── Significance ──
        bonf_alpha = stats.bonferroni_threshold
        if stats.t_pvalue < bonf_alpha:
            score += 0.20
            supports.append(
                f"Mean return statistically significant at Bonferroni-corrected "
                f"α={bonf_alpha:.4f}: t={stats.t_stat:.2f}, p={stats.t_pvalue:.4f}"
            )
        elif stats.t_pvalue < self.alpha:
            score += 0.05
            objections.append(
                f"Marginally significant at unadjusted α={self.alpha} "
                f"(t={stats.t_stat:.2f}, p={stats.t_pvalue:.4f}) but fails "
                f"Bonferroni correction for {stats.effective_n_tests} tests. "
                f"Likely a multiple-testing artefact."
            )
            refinements.append(
                "Reduce search space and retest on held-out data before claiming significance."
            )
        else:
            score -= 0.20
            objections.append(
                f"Mean return is NOT statistically significant "
                f"(t={stats.t_stat:.2f}, p={stats.t_pvalue:.4f}). "
                f"Insufficient evidence to reject the null of zero alpha."
            )

        # ── Sharpe ──
        if perf.sharpe_ratio > self.min_sharpe:
            score += 0.10
            supports.append(f"In-sample Sharpe {perf.sharpe_ratio:.3f} exceeds hurdle {self.min_sharpe}.")
        else:
            objections.append(
                f"In-sample Sharpe {perf.sharpe_ratio:.3f} below hurdle {self.min_sharpe}. "
                f"Risk-adjusted returns are insufficient for deployment."
            )

        # ── OOS Sharpe ──
        if perf.oos_sharpe > 0.3:
            score += 0.10
            supports.append(f"OOS Sharpe {perf.oos_sharpe:.3f} is positive and meaningful.")
        elif perf.oos_sharpe > 0.0:
            score += 0.02
            objections.append(f"OOS Sharpe {perf.oos_sharpe:.3f} is positive but low.")
        else:
            score -= 0.10
            objections.append(
                f"OOS Sharpe is negative ({perf.oos_sharpe:.3f}). "
                f"Strategy likely does not generalise out-of-sample."
            )
            refinements.append("Investigate overfitting: simplify signal construction.")

        # ── Sharpe decay ──
        if perf.sharpe_decay < self.max_sharpe_decay:
            score += 0.10
            supports.append(
                f"Sharpe decay {perf.sharpe_decay:.3f} < {self.max_sharpe_decay}: "
                f"good IS→OOS stability."
            )
        else:
            score -= 0.15
            objections.append(
                f"Sharpe decay of {perf.sharpe_decay:.3f} exceeds threshold "
                f"{self.max_sharpe_decay}. Strong evidence of in-sample overfitting."
            )
            refinements.append("Walk-forward validation across 3+ windows before considering.")

        # ── Autocorrelation ──
        if stats.lb_pvalue < 0.05:
            score -= 0.10
            objections.append(
                f"Ljung-Box test rejects no-autocorrelation "
                f"(LB={stats.lb_stat:.2f}, p={stats.lb_pvalue:.4f}). "
                f"Return autocorrelation inflates Sharpe; effective N is lower."
            )
            refinements.append("Use Newey-West standard errors and effective-n-adjusted t-stats.")

        # ── Normality ──
        if stats.jb_pvalue < 0.01:
            score -= 0.10
            objections.append(
                f"Jarque-Bera rejects normality (JB={stats.jb_stat:.2f}, "
                f"p={stats.jb_pvalue:.4f}). Sharpe/Sortino understate tail risk."
            )
            refinements.append("Report CVaR and max drawdown alongside Sharpe.")
        else:
            supports.append(
                f"Return distribution is approximately normal "
                f"(JB p={stats.jb_pvalue:.3f} ≥ 0.01)."
            )

        # ── Factor exposure ──
        if factor.r_squared > 0.5 and factor.alpha < 0.01:
            score -= 0.10
            objections.append(
                f"High factor model R²={factor.r_squared:.2f} with low alpha "
                f"{factor.alpha:.2%}. Returns primarily reflect factor exposure "
                f"(β_mkt={factor.beta_market:.2f}), not genuine alpha."
            )
            refinements.append("Construct factor-neutral version of the signal.")
        elif factor.alpha > 0.02:
            score += 0.05
            supports.append(
                f"Factor-adjusted alpha is {factor.alpha:.2%} annualised "
                f"(R²={factor.r_squared:.2f}). Meaningful genuine alpha above factor premia."
            )

        # ── Drawdown ──
        if perf.max_drawdown > 0.25:
            score -= 0.05
            objections.append(
                f"Max drawdown of {perf.max_drawdown:.1%} is large. "
                f"Recovery time was {perf.recovery_time} periods."
            )
        else:
            supports.append(f"Max drawdown {perf.max_drawdown:.1%} is within acceptable range.")

        # ── Execution ──
        if execution.feasible:
            score += 0.05
            supports.append(
                f"Execution feasible: net Sharpe {execution.slippage_adjusted_sharpe:.3f} "
                f"after {execution.slippage_bps:.1f} bps slippage."
            )
        else:
            objections.append(
                f"Execution not feasible: net Sharpe after costs is "
                f"{execution.slippage_adjusted_sharpe:.3f}. {execution.capacity_notes}"
            )

        # ── Sample size ──
        min_n = self.minimum_sample_size(max(perf.sharpe_ratio, 0.1))
        if stats.n_obs < min_n:
            score -= 0.05
            objections.append(
                f"Sample size {stats.n_obs} is below the estimated minimum "
                f"{min_n} observations needed for 80% power at this Sharpe level."
            )
            refinements.append(
                f"Extend data history by {min_n - stats.n_obs} observations "
                f"before drawing conclusions."
            )

        # ── Clip score to [0, 1] ──
        score = max(0.0, min(1.0, score))

        # ── Verdict thresholds ──
        if score >= 0.80:
            verdict = Verdict.STRONG_BUY_SIGNAL
        elif score >= 0.65:
            verdict = Verdict.BUY_SIGNAL
        elif score >= 0.45:
            verdict = Verdict.NEUTRAL
        elif score >= 0.30:
            verdict = Verdict.SELL_SIGNAL
        else:
            verdict = Verdict.REJECT

        return verdict, score, supports, objections, refinements

    def _insufficient_data_position(
        self, hypothesis_id: str, n_obs: int
    ) -> DebatePosition:
        return DebatePosition(
            hypothesis_id=hypothesis_id,
            verdict=Verdict.REJECT,
            confidence_score=0.95,
            performance=_empty_performance(),
            statistics=StatisticalTests(
                t_stat=0.0, t_pvalue=1.0, jb_stat=0.0, jb_pvalue=1.0,
                lb_stat=0.0, lb_pvalue=1.0, adf_stat=0.0, adf_pvalue=0.5,
                n_obs=n_obs, effective_n_tests=self.effective_n_tests,
                bonferroni_threshold=self.alpha / self.effective_n_tests,
            ),
            factor_exposure=FactorExposure(0,0,0,0,0,0,0),
            execution=ExecutionAssessment(0,0,0,0,0,False,"Insufficient data."),
            supporting_arguments=[],
            objections=[
                f"REJECTED: Only {n_obs} observations provided; "
                f"minimum required is {self.min_obs}."
            ],
            suggested_refinements=["Provide at least 2 years of daily return data."],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    rng = np.random.default_rng(1)

    # Simulate a strategy with modest genuine alpha + noise
    T = 756  # 3 years
    alpha_daily  = 0.0003
    sigma_daily  = 0.01
    returns = alpha_daily + rng.standard_normal(T) * sigma_daily

    agent = QuantResearcherAgent(
        alpha=0.05,
        min_sharpe=0.5,
        max_sharpe_decay=0.5,
        oos_fraction=0.30,
        effective_n_tests=100,
    )

    # Quick screen
    sharpe, p, passes = agent.quick_screen(returns)
    print(f"Quick screen: Sharpe={sharpe:.3f}, p={p:.4f}, passes={passes}")

    # Min sample size estimate
    min_n = agent.minimum_sample_size(expected_sharpe=0.5)
    print(f"Min sample size for Sharpe=0.5, power=80%: {min_n} obs")

    print("\nFull evaluation...")
    position = agent.evaluate("HYP-DEMO-001", returns)
    print(position.summary())

    print("\n\n--- Scenario: noisy / insignificant strategy ---")
    noise_rets = rng.standard_normal(300) * 0.015   # no alpha, short history
    pos2 = agent.evaluate("HYP-DEMO-002", noise_rets)
    print(pos2.summary())


if __name__ == "__main__":
    _demo()
