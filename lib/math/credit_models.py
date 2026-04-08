"""
credit_models.py — Credit risk models for srfm-lab.

Covers:
  - Merton structural model
  - CreditMetrics transition matrix simulation
  - KMV distance-to-default
  - Jarrow-Turnbull reduced-form model
  - CDO tranche pricing (Gaussian copula)
  - CDS pricing (risky bond + spread)
  - Credit VaR (portfolio loss distribution)
  - Recovery rate modeling (Beta distribution)
  - Contagion default model
  - Credit spread decomposition
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import beta as beta_fn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return stats.norm.cdf(x)


def _norm_pdf(x: float) -> float:
    return stats.norm.pdf(x)


# ---------------------------------------------------------------------------
# 1. Merton Structural Model
# ---------------------------------------------------------------------------

@dataclass
class MertonResult:
    default_probability: float      # risk-neutral PD over horizon T
    distance_to_default: float      # (log(V/D) + (mu - 0.5*sigma^2)*T) / (sigma*sqrt(T))
    equity_value: float             # model equity value (call on assets)
    asset_volatility: float         # implied asset volatility
    credit_spread: float            # risky yield - risk-free rate (annualized)


def merton_model(
    equity_value: float,
    equity_vol: float,
    debt_face: float,
    risk_free: float,
    T: float,
    mu: Optional[float] = None,
) -> MertonResult:
    """
    Merton (1974) structural model.

    Solves for asset value V and asset volatility sigma_V simultaneously using:
      E = V*N(d1) - D*exp(-r*T)*N(d2)
      sigma_E * E = V * sigma_V * N(d1)

    Parameters
    ----------
    equity_value : observed market equity (E)
    equity_vol   : annualized equity return volatility
    debt_face    : face value of debt (D)
    risk_free    : continuously compounded risk-free rate
    T            : debt maturity in years
    mu           : real-world drift (defaults to risk_free)
    """
    if mu is None:
        mu = risk_free

    def equations(params):
        V, sigma_V = params
        if V <= 0 or sigma_V <= 0:
            return [1e10, 1e10]
        d1 = (math.log(V / debt_face) + (risk_free + 0.5 * sigma_V ** 2) * T) / (sigma_V * math.sqrt(T))
        d2 = d1 - sigma_V * math.sqrt(T)
        eq1 = V * _norm_cdf(d1) - debt_face * math.exp(-risk_free * T) * _norm_cdf(d2) - equity_value
        eq2 = V * sigma_V * _norm_cdf(d1) - equity_vol * equity_value
        return [eq1, eq2]

    from scipy.optimize import fsolve
    V0 = equity_value + debt_face * math.exp(-risk_free * T)
    sigma0 = equity_vol * equity_value / V0
    sol = fsolve(equations, [V0, sigma0], full_output=False)
    V_star, sigma_star = float(sol[0]), float(sol[1])

    d1 = (math.log(V_star / debt_face) + (risk_free + 0.5 * sigma_star ** 2) * T) / (sigma_star * math.sqrt(T))
    d2 = d1 - sigma_star * math.sqrt(T)

    # Real-world DD uses mu instead of r
    dd_rw = (math.log(V_star / debt_face) + (mu - 0.5 * sigma_star ** 2) * T) / (sigma_star * math.sqrt(T))
    pd_rw = _norm_cdf(-dd_rw)

    # Risk-neutral PD
    pd_rn = _norm_cdf(-d2)

    # Credit spread: s = -log(N(d2) + (V/D)*exp(rT)*N(-d1)) / T - r
    risky_bond = debt_face * math.exp(-risk_free * T) * _norm_cdf(d2) + V_star * _norm_cdf(-d1) * (1.0 - 0.0)
    risky_bond_price = risky_bond / debt_face
    if risky_bond_price > 0:
        yield_risky = -math.log(risky_bond_price) / T
    else:
        yield_risky = risk_free + 0.50
    spread = max(0.0, yield_risky - risk_free)

    return MertonResult(
        default_probability=pd_rw,
        distance_to_default=dd_rw,
        equity_value=V_star * _norm_cdf(d1) - debt_face * math.exp(-risk_free * T) * _norm_cdf(d2),
        asset_volatility=sigma_star,
        credit_spread=spread,
    )


# ---------------------------------------------------------------------------
# 2. CreditMetrics Transition Matrix Simulation
# ---------------------------------------------------------------------------

# Standard 8-state rating transition matrix (annual, simplified)
DEFAULT_TRANSITION_MATRIX = np.array([
    # AAA   AA    A     BBB   BB    B     CCC   D
    [0.9081, 0.0833, 0.0068, 0.0006, 0.0008, 0.0002, 0.0001, 0.0001],  # AAA
    [0.0070, 0.9065, 0.0779, 0.0064, 0.0006, 0.0013, 0.0002, 0.0001],  # AA
    [0.0009, 0.0227, 0.9105, 0.0552, 0.0074, 0.0026, 0.0001, 0.0006],  # A
    [0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0117, 0.0012, 0.0018],  # BBB
    [0.0003, 0.0014, 0.0067, 0.0773, 0.8053, 0.0884, 0.0100, 0.0106],  # BB
    [0.0000, 0.0011, 0.0024, 0.0043, 0.0648, 0.8346, 0.0407, 0.0521],  # B
    [0.0022, 0.0000, 0.0022, 0.0130, 0.0238, 0.1124, 0.6486, 0.1978],  # CCC
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],  # D
], dtype=float)

RATING_NAMES = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]


@dataclass
class CreditMetricsResult:
    loss_distribution: np.ndarray    # simulated portfolio losses
    expected_loss: float
    loss_std: float
    var_95: float
    var_99: float
    cvar_99: float
    rating_migrations: Dict[str, np.ndarray]  # asset -> final rating counts


def creditmetrics_simulation(
    exposures: np.ndarray,           # shape (N,) dollar exposure per obligor
    initial_ratings: np.ndarray,     # shape (N,) int indices into RATING_NAMES
    recovery_rates: np.ndarray,      # shape (N,) LGD = 1 - recovery
    correlation_matrix: np.ndarray,  # shape (N, N) asset-return correlation
    transition_matrix: Optional[np.ndarray] = None,
    n_simulations: int = 10_000,
    horizon_years: float = 1.0,
    seed: int = 42,
) -> CreditMetricsResult:
    """
    CreditMetrics-style Monte Carlo simulation using Gaussian copula to
    introduce correlated rating migrations.
    """
    rng = np.random.default_rng(seed)
    if transition_matrix is None:
        transition_matrix = DEFAULT_TRANSITION_MATRIX

    N = len(exposures)
    n_ratings = transition_matrix.shape[0]

    # Raise annual matrix to horizon power via eigendecomposition
    if abs(horizon_years - 1.0) > 1e-6:
        vals, vecs = np.linalg.eig(transition_matrix.T)
        # Use matrix power approximation
        P = np.linalg.matrix_power(transition_matrix, max(1, int(round(horizon_years))))
    else:
        P = transition_matrix

    # Cumulative transition thresholds in standard-normal space per rating
    thresholds = np.zeros((n_ratings, n_ratings))
    for i in range(n_ratings):
        cum = np.cumsum(P[i])
        cum = np.clip(cum, 1e-12, 1 - 1e-12)
        thresholds[i] = stats.norm.ppf(cum)

    # Cholesky decomposition for correlated normals
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # Fall back to nearest PSD
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(corr_psd)

    losses = np.zeros(n_simulations)
    final_ratings = np.zeros((n_simulations, N), dtype=int)

    for sim in range(n_simulations):
        z_indep = rng.standard_normal(N)
        z_corr = L @ z_indep

        for i in range(N):
            row = int(initial_ratings[i])
            thresh = thresholds[row]
            new_rating = 0
            for j in range(n_ratings):
                if z_corr[i] <= thresh[j]:
                    new_rating = j
                    break
            else:
                new_rating = n_ratings - 1
            final_ratings[sim, i] = new_rating
            if new_rating == n_ratings - 1:  # default
                losses[sim] += exposures[i] * (1.0 - recovery_rates[i])

    migration_counts: Dict[str, np.ndarray] = {}
    for i in range(N):
        counts = np.bincount(final_ratings[:, i], minlength=n_ratings)
        migration_counts[f"obligor_{i}"] = counts

    return CreditMetricsResult(
        loss_distribution=losses,
        expected_loss=float(np.mean(losses)),
        loss_std=float(np.std(losses)),
        var_95=float(np.percentile(losses, 95)),
        var_99=float(np.percentile(losses, 99)),
        cvar_99=float(np.mean(losses[losses >= np.percentile(losses, 99)])),
        rating_migrations=migration_counts,
    )


# ---------------------------------------------------------------------------
# 3. KMV Distance-to-Default
# ---------------------------------------------------------------------------

@dataclass
class KMVResult:
    distance_to_default: float
    expected_default_frequency: float   # EDF mapped from DD
    asset_value: float
    asset_volatility: float


def kmv_distance_to_default(
    market_cap: float,
    short_term_debt: float,
    long_term_debt: float,
    equity_vol: float,
    risk_free: float = 0.05,
    T: float = 1.0,
    edf_scale: float = 1.0,
) -> KMVResult:
    """
    KMV model: default point = STD + 0.5 * LTD (Moody's convention).
    Iteratively solves for asset value and volatility.
    """
    default_point = short_term_debt + 0.5 * long_term_debt

    # Iterative approach: start with equity_vol as proxy for asset vol
    V = market_cap + default_point * math.exp(-risk_free * T)
    sigma_A = equity_vol * market_cap / V

    for _ in range(200):
        d1 = (math.log(V / default_point) + (risk_free + 0.5 * sigma_A ** 2) * T) / (sigma_A * math.sqrt(T))
        nd1 = _norm_cdf(d1)
        sigma_A_new = equity_vol * market_cap / (V * nd1) if nd1 > 1e-10 else sigma_A
        d2 = d1 - sigma_A * math.sqrt(T)
        V_new = (market_cap + default_point * math.exp(-risk_free * T) * _norm_cdf(d2)) / _norm_cdf(d1)
        if abs(V_new - V) < 1e-4 and abs(sigma_A_new - sigma_A) < 1e-6:
            break
        V = 0.5 * V + 0.5 * V_new
        sigma_A = 0.5 * sigma_A + 0.5 * sigma_A_new

    dd = (math.log(V / default_point) + (risk_free - 0.5 * sigma_A ** 2) * T) / (sigma_A * math.sqrt(T))
    # Moody's EDF empirical mapping: approximately N(-DD) scaled
    edf = _norm_cdf(-dd) * edf_scale
    edf = min(max(edf, 0.0), 1.0)

    return KMVResult(
        distance_to_default=dd,
        expected_default_frequency=edf,
        asset_value=V,
        asset_volatility=sigma_A,
    )


# ---------------------------------------------------------------------------
# 4. Jarrow-Turnbull Reduced-Form (Intensity-Based) Model
# ---------------------------------------------------------------------------

@dataclass
class JarrowTurnbullResult:
    survival_probabilities: np.ndarray  # Q(tau > t_i) for each payment date
    risky_bond_price: float
    default_intensity: float            # constant lambda
    hazard_rates: np.ndarray


def jarrow_turnbull(
    payment_dates: np.ndarray,          # years from now: e.g. [0.5, 1.0, 1.5, 2.0]
    coupon_rate: float,                 # annual coupon / face
    face_value: float,
    risk_free_rates: np.ndarray,        # continuously compounded spot rates per payment date
    default_intensity: float,           # constant hazard rate lambda
    recovery_rate: float = 0.40,
) -> JarrowTurnbullResult:
    """
    Jarrow-Turnbull (1995) reduced-form model.
    Default arrives as Poisson process with intensity lambda.
    Risky bond price = sum of discounted cash flows weighted by survival prob
                     + recovery * (1 - survival) * discount factor.
    """
    T = payment_dates
    n = len(T)
    r = risk_free_rates
    lam = default_intensity

    # Survival probabilities: Q(tau > t) = exp(-lambda * t)
    survival = np.exp(-lam * T)
    hazard_rates = np.full(n, lam)

    # Discount factors
    df = np.exp(-r * T)

    # Coupon payment at each date (annualized coupon * period length)
    dt = np.diff(np.concatenate([[0.0], T]))
    coupon_cash_flows = face_value * coupon_rate * dt

    # Risky price = PV of coupons (if survived) + PV of face at maturity (if survived)
    # + PV of recovery on default
    risky_price = 0.0
    prev_survival = 1.0
    for i in range(n):
        s = survival[i]
        coupon_pv = coupon_cash_flows[i] * s * df[i]
        # Default in (t_{i-1}, t_i]: prob = prev_s - s
        default_prob_in_period = prev_survival - s
        recovery_pv = recovery_rate * face_value * default_prob_in_period * df[i]
        risky_price += coupon_pv + recovery_pv
        prev_survival = s

    # Face value at maturity if survived
    risky_price += face_value * survival[-1] * df[-1]

    return JarrowTurnbullResult(
        survival_probabilities=survival,
        risky_bond_price=risky_price,
        default_intensity=lam,
        hazard_rates=hazard_rates,
    )


# ---------------------------------------------------------------------------
# 5. CDO Tranche Pricing — Gaussian Copula
# ---------------------------------------------------------------------------

@dataclass
class CDOTrancheResult:
    tranche_spread: float       # par spread in bps
    tranche_el: float           # expected loss of tranche
    portfolio_el: float         # portfolio-level expected loss
    attachment_point: float
    detachment_point: float


def _gaussian_copula_portfolio_loss(
    n_names: int,
    pd_individual: np.ndarray,
    lgd: np.ndarray,
    rho: float,
    n_simulations: int = 50_000,
    seed: int = 0,
) -> np.ndarray:
    """Simulate portfolio losses via single-factor Gaussian copula."""
    rng = np.random.default_rng(seed)
    # Thresholds: N^{-1}(PD_i)
    thresholds = stats.norm.ppf(pd_individual)
    notional = np.ones(n_names)  # unit notional per name

    losses = np.zeros(n_simulations)
    for s in range(n_simulations):
        M = rng.standard_normal()  # market factor
        Z = rng.standard_normal(n_names)
        X = math.sqrt(rho) * M + math.sqrt(1 - rho) * Z
        defaults = X < thresholds
        losses[s] = np.sum(defaults * lgd * notional)

    return losses / n_names  # normalize to [0,1] as fraction of portfolio


def cdo_tranche_price(
    attachment: float,          # e.g. 0.03 (3%)
    detachment: float,          # e.g. 0.07 (7%)
    n_names: int,
    pd_individual: np.ndarray,
    lgd: np.ndarray,
    correlation: float,
    risk_free: float = 0.05,
    T: float = 5.0,
    n_simulations: int = 50_000,
) -> CDOTrancheResult:
    """
    Price a CDO tranche using the Gaussian copula.
    Tranche absorbs losses in [attachment, detachment].
    Par spread = E[tranche_loss] / risky_annuity.
    """
    portfolio_losses = _gaussian_copula_portfolio_loss(
        n_names, pd_individual, lgd, correlation, n_simulations
    )

    tranche_width = detachment - attachment
    tranche_losses = np.clip(portfolio_losses - attachment, 0, tranche_width) / tranche_width

    el_tranche = float(np.mean(tranche_losses))
    el_portfolio = float(np.mean(portfolio_losses))

    # Simple risky annuity: T * (1 - EL_tranche/2) * discount
    risky_annuity = T * (1.0 - el_tranche / 2.0) * math.exp(-risk_free * T / 2.0)
    if risky_annuity < 1e-10:
        spread_bps = 0.0
    else:
        spread_bps = (el_tranche / risky_annuity) * 10_000

    return CDOTrancheResult(
        tranche_spread=spread_bps,
        tranche_el=el_tranche,
        portfolio_el=el_portfolio,
        attachment_point=attachment,
        detachment_point=detachment,
    )


# ---------------------------------------------------------------------------
# 6. Credit Default Swap Pricing
# ---------------------------------------------------------------------------

@dataclass
class CDSResult:
    par_spread_bps: float       # fair CDS spread in basis points
    risky_annuity: float        # PV01: value of 1bp per annum protection
    protection_leg_pv: float    # PV of protection payments
    premium_leg_pv: float       # PV of premium at par spread
    mtm_pnl: float              # MTM if entered at contract_spread


def cds_price(
    payment_dates: np.ndarray,          # e.g. quarterly [0.25, 0.5, ..., 5.0]
    hazard_rate: float,                 # constant default intensity
    recovery_rate: float,
    risk_free: float,
    notional: float = 1_000_000.0,
    contract_spread_bps: float = 0.0,
) -> CDSResult:
    """
    Price a CDS using flat hazard rate and flat risk-free curve.
    Protection leg = integral of LGD * default prob density * discount.
    Premium leg = sum of period * survival * discount * spread.
    """
    lam = hazard_rate
    r = risk_free
    lgd = 1.0 - recovery_rate

    T = payment_dates
    n = len(T)
    dt = np.diff(np.concatenate([[0.0], T]))

    survival = np.exp(-lam * T)
    df = np.exp(-r * T)

    # Premium leg PV (per unit spread)
    premium_leg_unit = float(np.sum(dt * survival * df))

    # Protection leg PV: LGD * integral of lambda*exp(-(lambda+r)*t) dt
    # For constant lam, r: = LGD * lam / (lam + r) * (1 - exp(-(lam+r)*T_final))
    T_final = float(T[-1])
    lr = lam + r
    protection_leg = lgd * (lam / lr) * (1.0 - math.exp(-lr * T_final)) * notional

    par_spread = (protection_leg / notional) / premium_leg_unit if premium_leg_unit > 1e-12 else 0.0
    risky_annuity = premium_leg_unit * notional / 10_000  # PV of 1bp

    par_bps = par_spread * 10_000
    contract_spread = contract_spread_bps / 10_000
    mtm = (par_spread - contract_spread) * premium_leg_unit * notional

    return CDSResult(
        par_spread_bps=par_bps,
        risky_annuity=risky_annuity,
        protection_leg_pv=protection_leg,
        premium_leg_pv=par_spread * premium_leg_unit * notional,
        mtm_pnl=mtm,
    )


# ---------------------------------------------------------------------------
# 7. Credit VaR — Portfolio Loss Distribution
# ---------------------------------------------------------------------------

@dataclass
class CreditVaRResult:
    expected_loss: float
    unexpected_loss: float      # std dev of loss
    var_95: float
    var_99: float
    cvar_99: float
    loss_distribution: np.ndarray


def credit_var(
    exposures: np.ndarray,
    hazard_rates: np.ndarray,
    recovery_rates: np.ndarray,
    correlation_matrix: np.ndarray,
    T: float = 1.0,
    n_simulations: int = 50_000,
    seed: int = 42,
) -> CreditVaRResult:
    """
    Monte Carlo Credit VaR using Gaussian copula for correlated defaults.
    """
    rng = np.random.default_rng(seed)
    N = len(exposures)

    pd = 1.0 - np.exp(-hazard_rates * T)
    thresholds = stats.norm.ppf(np.clip(pd, 1e-10, 1 - 1e-10))
    lgd = 1.0 - recovery_rates

    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        ev, evec = np.linalg.eigh(correlation_matrix)
        ev = np.maximum(ev, 1e-8)
        L = np.linalg.cholesky(evec @ np.diag(ev) @ evec.T)

    losses = np.zeros(n_simulations)
    for s in range(n_simulations):
        z = L @ rng.standard_normal(N)
        defaults = z < thresholds
        losses[s] = np.sum(defaults * lgd * exposures)

    el = float(np.mean(losses))
    ul = float(np.std(losses))
    var95 = float(np.percentile(losses, 95))
    var99 = float(np.percentile(losses, 99))
    tail = losses[losses >= np.percentile(losses, 99)]
    cvar99 = float(np.mean(tail)) if len(tail) > 0 else var99

    return CreditVaRResult(
        expected_loss=el,
        unexpected_loss=ul,
        var_95=var95,
        var_99=var99,
        cvar_99=cvar99,
        loss_distribution=losses,
    )


# ---------------------------------------------------------------------------
# 8. Recovery Rate Modeling — Beta Distribution
# ---------------------------------------------------------------------------

@dataclass
class RecoveryRateModel:
    alpha: float        # Beta distribution shape parameter
    beta: float         # Beta distribution shape parameter
    mean: float
    std: float
    senior_adj: float   # seniority adjustment factor


def fit_recovery_rate_model(
    observed_recoveries: np.ndarray,
    seniority: str = "senior_secured",
) -> RecoveryRateModel:
    """
    Fit a Beta distribution to observed recovery rates.
    Adjust mean for seniority class.
    """
    seniority_adjustments = {
        "senior_secured": 1.0,
        "senior_unsecured": 0.75,
        "subordinated": 0.50,
        "equity": 0.20,
    }
    adj = seniority_adjustments.get(seniority, 1.0)

    obs = np.clip(observed_recoveries, 1e-4, 1 - 1e-4)
    mu = float(np.mean(obs))
    var = float(np.var(obs))

    if var <= 0 or var >= mu * (1 - mu):
        alpha_hat = 2.0
        beta_hat = 2.0
    else:
        alpha_hat = mu * (mu * (1 - mu) / var - 1)
        beta_hat = (1 - mu) * (mu * (1 - mu) / var - 1)

    fitted_mean = alpha_hat / (alpha_hat + beta_hat) * adj
    fitted_std = math.sqrt(alpha_hat * beta_hat / ((alpha_hat + beta_hat) ** 2 * (alpha_hat + beta_hat + 1)))

    return RecoveryRateModel(
        alpha=alpha_hat,
        beta=beta_hat,
        mean=fitted_mean,
        std=fitted_std,
        senior_adj=adj,
    )


def sample_recovery_rates(
    model: RecoveryRateModel,
    n: int,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.beta(model.alpha, model.beta, size=n)
    return np.clip(samples * model.senior_adj, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 9. Contagion Default Model
# ---------------------------------------------------------------------------

@dataclass
class ContagionResult:
    default_sequence: List[int]                 # order of defaults
    final_intensities: np.ndarray               # lambda at end of simulation
    num_defaults: int
    contagion_triggered: bool


def contagion_default_model(
    n_obligors: int,
    base_intensities: np.ndarray,       # shape (N,) lambda_0 for each obligor
    contagion_jump: float,              # delta_lambda on each default
    contagion_decay: float,             # how fast contagion intensity decays
    T: float = 1.0,
    dt: float = 1 / 252,
    seed: int = 42,
) -> ContagionResult:
    """
    Contagion model: when obligor j defaults, all surviving obligors'
    intensity jumps by contagion_jump, then decays exponentially.

    Uses discretized simulation with Euler step.
    """
    rng = np.random.default_rng(seed)
    intensities = base_intensities.copy().astype(float)
    n_steps = int(T / dt)
    defaulted = np.zeros(n_obligors, dtype=bool)
    default_sequence: List[int] = []
    contagion_extra = np.zeros(n_obligors)

    for step in range(n_steps):
        # Decay contagion intensity
        contagion_extra *= math.exp(-contagion_decay * dt)
        effective_lam = intensities + contagion_extra

        # Check for defaults this period
        u = rng.random(n_obligors)
        default_prob_dt = 1.0 - np.exp(-effective_lam * dt)
        new_defaults = (~defaulted) & (u < default_prob_dt)

        for j in np.where(new_defaults)[0]:
            defaulted[j] = True
            default_sequence.append(int(j))
            # Contagion: surviving obligors get intensity jump
            surviving = ~defaulted
            contagion_extra[surviving] += contagion_jump

    return ContagionResult(
        default_sequence=default_sequence,
        final_intensities=intensities + contagion_extra,
        num_defaults=len(default_sequence),
        contagion_triggered=len(default_sequence) > int(np.sum(base_intensities * T) * 1.5),
    )


# ---------------------------------------------------------------------------
# 10. Credit Spread Decomposition
# ---------------------------------------------------------------------------

@dataclass
class CreditSpreadDecomposition:
    total_spread_bps: float
    default_component_bps: float        # EL from hazard rate
    liquidity_premium_bps: float        # illiquidity compensation
    risk_premium_bps: float             # compensation for uncertainty
    residual_bps: float                 # unexplained
    default_fraction: float
    liquidity_fraction: float
    risk_premium_fraction: float


def decompose_credit_spread(
    observed_spread_bps: float,
    hazard_rate: float,
    recovery_rate: float,
    bid_ask_spread_bps: float,          # proxy for liquidity
    equity_vol: float,                  # proxy for uncertainty
    risk_free: float = 0.05,
    T: float = 1.0,
) -> CreditSpreadDecomposition:
    """
    Decompose observed credit spread into:
      1. Default component: (1 - R) * lambda  (expected loss per annum)
      2. Liquidity premium: proxy from bid-ask spread
      3. Risk premium: residual after removing EL and liquidity
    """
    lgd = 1.0 - recovery_rate
    default_component = lgd * hazard_rate * 10_000  # in bps

    # Liquidity premium: half the bid-ask spread as compensation
    liquidity_premium = bid_ask_spread_bps * 0.5

    # Risk premium proxy: scale by equity vol (uncertainty aversion)
    uncertainty_premium = default_component * (equity_vol / 0.20) * 0.15

    # Ensure components sum to at most observed spread
    total_model = default_component + liquidity_premium + uncertainty_premium
    residual = observed_spread_bps - total_model

    if observed_spread_bps > 1e-6:
        d_frac = default_component / observed_spread_bps
        l_frac = liquidity_premium / observed_spread_bps
        rp_frac = uncertainty_premium / observed_spread_bps
    else:
        d_frac = l_frac = rp_frac = 0.0

    return CreditSpreadDecomposition(
        total_spread_bps=observed_spread_bps,
        default_component_bps=default_component,
        liquidity_premium_bps=liquidity_premium,
        risk_premium_bps=uncertainty_premium,
        residual_bps=residual,
        default_fraction=d_frac,
        liquidity_fraction=l_frac,
        risk_premium_fraction=rp_frac,
    )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_credit_suite_demo() -> Dict:
    """Quick smoke test of all models. Returns a summary dict."""
    results = {}

    # Merton
    m = merton_model(
        equity_value=50.0, equity_vol=0.30, debt_face=100.0,
        risk_free=0.05, T=1.0
    )
    results["merton"] = {"pd": m.default_probability, "dd": m.distance_to_default}

    # KMV
    k = kmv_distance_to_default(
        market_cap=50.0, short_term_debt=30.0, long_term_debt=60.0,
        equity_vol=0.30, risk_free=0.05
    )
    results["kmv"] = {"dd": k.distance_to_default, "edf": k.expected_default_frequency}

    # CDS
    dates = np.arange(0.25, 5.25, 0.25)
    cds = cds_price(dates, hazard_rate=0.02, recovery_rate=0.40, risk_free=0.05)
    results["cds_spread_bps"] = cds.par_spread_bps

    # JT
    jt = jarrow_turnbull(dates, coupon_rate=0.05, face_value=100.0,
                          risk_free_rates=np.full(len(dates), 0.05),
                          default_intensity=0.02)
    results["jt_bond_price"] = jt.risky_bond_price

    # Spread decomposition
    dec = decompose_credit_spread(
        observed_spread_bps=150.0, hazard_rate=0.02, recovery_rate=0.40,
        bid_ask_spread_bps=20.0, equity_vol=0.30
    )
    results["spread_decomp"] = {
        "default_bps": dec.default_component_bps,
        "liquidity_bps": dec.liquidity_premium_bps,
        "risk_premium_bps": dec.risk_premium_bps,
    }

    return results
