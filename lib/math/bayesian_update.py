"""
bayesian_update.py — Bayesian inference for financial parameters.

Conjugate models: Beta-Binomial, Normal-Normal, Inverse-Gamma, Normal-Inverse-Gamma.
Sequential updating, Bayes factors, predictive distributions, credible intervals,
HDI, empirical Bayes (EB-Normal), and simple two-level hierarchical model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, special, stats


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BetaBinomialState:
    """Posterior state for Bayesian win-rate estimation."""
    alpha: float     # prior / posterior shape 1 (successes + prior)
    beta: float      # prior / posterior shape 2 (failures + prior)
    n_obs: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return a * b / (s ** 2 * (s + 1))

    @property
    def mode(self) -> Optional[float]:
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None


@dataclass
class NormalNormalState:
    """Posterior for Normal likelihood with Normal prior on mean (known variance)."""
    mu_0: float       # prior mean
    sigma_0_sq: float # prior variance on mean
    sigma_sq: float   # known likelihood variance
    mu_n: float = 0.0
    sigma_n_sq: float = 0.0
    n_obs: int = 0

    def __post_init__(self) -> None:
        self.mu_n = self.mu_0
        self.sigma_n_sq = self.sigma_0_sq


@dataclass
class InvGammaState:
    """Posterior for variance with Inverse-Gamma prior."""
    alpha_0: float    # prior shape
    beta_0: float     # prior scale
    alpha_n: float = 0.0
    beta_n: float = 0.0
    n_obs: int = 0

    def __post_init__(self) -> None:
        self.alpha_n = self.alpha_0
        self.beta_n = self.beta_0

    @property
    def mean_variance(self) -> float:
        if self.alpha_n > 1:
            return self.beta_n / (self.alpha_n - 1)
        return float('inf')

    @property
    def mode_variance(self) -> float:
        return self.beta_n / (self.alpha_n + 1)


@dataclass
class NIG_State:
    """Normal-Inverse-Gamma joint posterior for (mu, sigma^2)."""
    mu_0: float
    kappa_0: float    # prior pseudo-observations on mean
    alpha_0: float    # IG shape
    beta_0: float     # IG scale
    mu_n: float = 0.0
    kappa_n: float = 0.0
    alpha_n: float = 0.0
    beta_n: float = 0.0
    n_obs: int = 0

    def __post_init__(self) -> None:
        self.mu_n = self.mu_0
        self.kappa_n = self.kappa_0
        self.alpha_n = self.alpha_0
        self.beta_n = self.beta_0

    @property
    def posterior_mean_mu(self) -> float:
        return self.mu_n

    @property
    def posterior_mean_sigma2(self) -> float:
        if self.alpha_n > 1:
            return self.beta_n / (self.alpha_n - 1)
        return float('inf')


@dataclass
class CredibleInterval:
    lower: float
    upper: float
    level: float      # e.g. 0.95
    method: str       # 'equal_tailed' or 'hdi'


@dataclass
class BayesFactor:
    """Result of Bayesian hypothesis test H1 vs H0."""
    log_bf: float     # log Bayes factor: log P(data|H1) - log P(data|H0)
    bf: float         # Bayes factor = P(data|H1) / P(data|H0)
    interpretation: str


@dataclass
class HierarchicalModel:
    """Two-level hierarchical Normal model pooling alpha estimates."""
    asset_means: np.ndarray          # per-asset posterior means
    asset_variances: np.ndarray      # per-asset posterior variances
    group_mean: float                # hyperprior mean (estimated from data)
    group_variance: float            # hyperprior variance
    pooled_means: np.ndarray         # shrinkage-pooled estimates
    shrinkage_factors: np.ndarray    # how much each asset is shrunk


# ---------------------------------------------------------------------------
# 1. Beta-Binomial: Bayesian Win Rate
# ---------------------------------------------------------------------------

def beta_binomial_prior(alpha: float = 1.0, beta: float = 1.0) -> BetaBinomialState:
    """Initialise Beta(alpha, beta) prior. Default = uniform."""
    return BetaBinomialState(alpha=alpha, beta=beta, n_obs=0)


def beta_binomial_update(state: BetaBinomialState,
                          wins: int, losses: int) -> BetaBinomialState:
    """
    Conjugate update: Beta(alpha + wins, beta + losses).
    """
    return BetaBinomialState(alpha=state.alpha + wins,
                              beta=state.beta + losses,
                              n_obs=state.n_obs + wins + losses)


def beta_binomial_update_stream(state: BetaBinomialState,
                                 outcomes: List[int]) -> BetaBinomialState:
    """Update from a list of 0/1 outcomes sequentially."""
    wins = int(sum(1 for o in outcomes if o == 1))
    losses = len(outcomes) - wins
    return beta_binomial_update(state, wins, losses)


def beta_binomial_predictive(state: BetaBinomialState, n_future: int = 1) -> np.ndarray:
    """
    Beta-Binomial predictive PMF: P(Y = k | data) for k = 0, ..., n_future.
    """
    k_vals = np.arange(n_future + 1)
    log_probs = []
    for k in k_vals:
        log_p = (special.betaln(state.alpha + k, state.beta + n_future - k)
                 - special.betaln(state.alpha, state.beta)
                 + special.gammaln(n_future + 1)
                 - special.gammaln(k + 1)
                 - special.gammaln(n_future - k + 1))
        log_probs.append(log_p)
    probs = np.exp(np.array(log_probs))
    probs /= probs.sum()
    return probs


def beta_credible_interval(state: BetaBinomialState,
                            level: float = 0.95) -> CredibleInterval:
    """Equal-tailed credible interval from Beta posterior."""
    tail = (1.0 - level) / 2.0
    dist = stats.beta(state.alpha, state.beta)
    return CredibleInterval(lower=float(dist.ppf(tail)),
                             upper=float(dist.ppf(1.0 - tail)),
                             level=level, method='equal_tailed')


# ---------------------------------------------------------------------------
# 2. Normal-Normal Conjugate: Bayesian Mean Estimation
# ---------------------------------------------------------------------------

def normal_normal_prior(mu_0: float = 0.0, sigma_0_sq: float = 1.0,
                         sigma_sq: float = 1.0) -> NormalNormalState:
    """N(mu_0, sigma_0_sq) prior on mean, known likelihood variance sigma_sq."""
    return NormalNormalState(mu_0=mu_0, sigma_0_sq=sigma_0_sq, sigma_sq=sigma_sq)


def normal_normal_update(state: NormalNormalState,
                          observations: np.ndarray) -> NormalNormalState:
    """
    Conjugate Normal-Normal update.
    Posterior: mu | data ~ N(mu_n, sigma_n^2)
      1/sigma_n^2 = 1/sigma_0^2 + n/sigma^2
      mu_n = sigma_n^2 * (mu_0/sigma_0^2 + n*x_bar/sigma^2)
    """
    obs = np.asarray(observations, dtype=float)
    n = len(obs)
    xbar = float(obs.mean())
    prec_prior = 1.0 / state.sigma_0_sq
    prec_lik = n / state.sigma_sq
    prec_post = prec_prior + prec_lik
    sigma_n_sq = 1.0 / prec_post
    mu_n = sigma_n_sq * (state.mu_0 / state.sigma_0_sq + n * xbar / state.sigma_sq)
    new_state = NormalNormalState(mu_0=state.mu_0, sigma_0_sq=state.sigma_0_sq,
                                   sigma_sq=state.sigma_sq,
                                   mu_n=mu_n, sigma_n_sq=sigma_n_sq,
                                   n_obs=state.n_obs + n)
    return new_state


def normal_predictive(state: NormalNormalState) -> stats.norm:
    """
    Posterior predictive distribution for next observation: N(mu_n, sigma^2 + sigma_n^2).
    """
    pred_var = state.sigma_sq + state.sigma_n_sq
    return stats.norm(loc=state.mu_n, scale=math.sqrt(pred_var))


def normal_credible_interval(state: NormalNormalState,
                               level: float = 0.95) -> CredibleInterval:
    tail = (1.0 - level) / 2.0
    dist = stats.norm(loc=state.mu_n, scale=math.sqrt(state.sigma_n_sq))
    return CredibleInterval(lower=float(dist.ppf(tail)),
                             upper=float(dist.ppf(1.0 - tail)),
                             level=level, method='equal_tailed')


# ---------------------------------------------------------------------------
# 3. Inverse-Gamma Prior for Variance
# ---------------------------------------------------------------------------

def inv_gamma_prior(alpha_0: float = 2.0, beta_0: float = 1.0) -> InvGammaState:
    """IG(alpha_0, beta_0) prior on variance."""
    return InvGammaState(alpha_0=alpha_0, beta_0=beta_0)


def inv_gamma_update(state: InvGammaState, observations: np.ndarray,
                      known_mean: float = 0.0) -> InvGammaState:
    """
    Conjugate IG update with known mean.
    Posterior: sigma^2 | data ~ IG(alpha_0 + n/2, beta_0 + SS/2)
    where SS = Σ (x_i - mu)^2
    """
    obs = np.asarray(observations, dtype=float)
    n = len(obs)
    ss = float(np.sum((obs - known_mean) ** 2))
    alpha_n = state.alpha_0 + n / 2.0
    beta_n = state.beta_0 + ss / 2.0
    return InvGammaState(alpha_0=state.alpha_0, beta_0=state.beta_0,
                          alpha_n=alpha_n, beta_n=beta_n,
                          n_obs=state.n_obs + n)


def inv_gamma_credible_interval(state: InvGammaState,
                                 level: float = 0.95) -> CredibleInterval:
    """Equal-tailed CI on variance from IG posterior."""
    tail = (1.0 - level) / 2.0
    dist = stats.invgamma(a=state.alpha_n, scale=state.beta_n)
    return CredibleInterval(lower=float(dist.ppf(tail)),
                             upper=float(dist.ppf(1.0 - tail)),
                             level=level, method='equal_tailed')


# ---------------------------------------------------------------------------
# 4. Normal-Inverse-Gamma Joint Posterior
# ---------------------------------------------------------------------------

def nig_prior(mu_0: float = 0.0, kappa_0: float = 1.0,
               alpha_0: float = 2.0, beta_0: float = 1.0) -> NIG_State:
    """NIG(mu_0, kappa_0, alpha_0, beta_0) prior."""
    return NIG_State(mu_0=mu_0, kappa_0=kappa_0, alpha_0=alpha_0, beta_0=beta_0)


def nig_update(state: NIG_State, observations: np.ndarray) -> NIG_State:
    """
    Conjugate NIG update.
    Posterior hyperparameters:
      kappa_n = kappa_0 + n
      mu_n    = (kappa_0 * mu_0 + n * x_bar) / kappa_n
      alpha_n = alpha_0 + n/2
      beta_n  = beta_0 + 0.5*SS + kappa_0*n*(x_bar - mu_0)^2 / (2*kappa_n)
    """
    obs = np.asarray(observations, dtype=float)
    n = len(obs)
    xbar = float(obs.mean())
    ss = float(np.sum((obs - xbar) ** 2))
    kappa_n = state.kappa_n + n
    mu_n = (state.kappa_n * state.mu_n + n * xbar) / kappa_n
    alpha_n = state.alpha_n + n / 2.0
    beta_n = (state.beta_n + 0.5 * ss
              + state.kappa_n * n * (xbar - state.mu_n) ** 2 / (2.0 * kappa_n))
    new_state = NIG_State(mu_0=state.mu_0, kappa_0=state.kappa_0,
                           alpha_0=state.alpha_0, beta_0=state.beta_0,
                           mu_n=mu_n, kappa_n=kappa_n,
                           alpha_n=alpha_n, beta_n=beta_n,
                           n_obs=state.n_obs + n)
    return new_state


def nig_marginal_mu(state: NIG_State) -> stats.t:
    """
    Marginal distribution of mu: Student-t
    t_{2*alpha_n}(mu_n, beta_n / (alpha_n * kappa_n))
    """
    df = 2.0 * state.alpha_n
    scale = math.sqrt(state.beta_n / (state.alpha_n * state.kappa_n))
    return stats.t(df=df, loc=state.mu_n, scale=scale)


def nig_predictive(state: NIG_State) -> stats.t:
    """
    Posterior predictive for next observation: Student-t
    t_{2*alpha_n}(mu_n, beta_n*(kappa_n+1)/(alpha_n*kappa_n))
    """
    df = 2.0 * state.alpha_n
    scale = math.sqrt(state.beta_n * (state.kappa_n + 1.0)
                      / (state.alpha_n * state.kappa_n))
    return stats.t(df=df, loc=state.mu_n, scale=scale)


# ---------------------------------------------------------------------------
# 5. Sequential Bayesian Update (streaming)
# ---------------------------------------------------------------------------

class SequentialBayesianUpdater:
    """
    Maintains a NIG posterior that is updated one observation at a time.
    Useful for streaming / online inference.
    """

    def __init__(self, mu_0: float = 0.0, kappa_0: float = 1.0,
                 alpha_0: float = 2.0, beta_0: float = 1.0):
        self.state = nig_prior(mu_0, kappa_0, alpha_0, beta_0)
        self._history: List[float] = []

    def update(self, x: float) -> NIG_State:
        """Incorporate a single new observation."""
        self.state = nig_update(self.state, np.array([x]))
        self._history.append(x)
        return self.state

    def update_batch(self, xs: np.ndarray) -> NIG_State:
        self.state = nig_update(self.state, np.asarray(xs))
        self._history.extend(xs.tolist())
        return self.state

    def credible_interval(self, level: float = 0.95) -> CredibleInterval:
        dist = nig_marginal_mu(self.state)
        tail = (1.0 - level) / 2.0
        return CredibleInterval(lower=float(dist.ppf(tail)),
                                 upper=float(dist.ppf(1.0 - tail)),
                                 level=level, method='equal_tailed')

    def predictive_std(self) -> float:
        dist = nig_predictive(self.state)
        return float(dist.std())


# ---------------------------------------------------------------------------
# 6. Bayesian Hypothesis Testing (Bayes Factor)
# ---------------------------------------------------------------------------

def bayes_factor_normal_mean(observations: np.ndarray,
                              mu_H0: float = 0.0,
                              mu_H1: float = 0.01,
                              sigma: float = 0.02) -> BayesFactor:
    """
    Compute Bayes factor for H1: mu = mu_H1 vs H0: mu = mu_H0
    under Normal likelihood with known sigma.

    BF = P(data | H1) / P(data | H0) = product of likelihood ratios.
    """
    obs = np.asarray(observations, dtype=float)
    log_lh0 = float(np.sum(stats.norm.logpdf(obs, loc=mu_H0, scale=sigma)))
    log_lh1 = float(np.sum(stats.norm.logpdf(obs, loc=mu_H1, scale=sigma)))
    log_bf = log_lh1 - log_lh0
    bf = math.exp(min(log_bf, 700))  # avoid overflow
    if bf > 100:
        interp = "Very strong evidence for H1"
    elif bf > 10:
        interp = "Strong evidence for H1"
    elif bf > 3:
        interp = "Moderate evidence for H1"
    elif bf > 1:
        interp = "Weak evidence for H1"
    elif bf > 1 / 3:
        interp = "Weak evidence for H0"
    elif bf > 1 / 10:
        interp = "Moderate evidence for H0"
    else:
        interp = "Strong evidence for H0"
    return BayesFactor(log_bf=log_bf, bf=bf, interpretation=interp)


def bayes_factor_jeffreys(observations: np.ndarray,
                           mu_H0: float = 0.0,
                           sigma: float = 0.02) -> BayesFactor:
    """
    JZS / Jeffreys-Zellner-Siow-style Bayes factor for testing H0: mu = 0
    vs H1: mu ≠ 0 with Cauchy prior on mu (scale = r*sigma).
    """
    obs = np.asarray(observations, dtype=float)
    n = len(obs)
    xbar = float(obs.mean())
    t_stat = xbar / (sigma / math.sqrt(n))
    r = 0.707  # default JZS scale

    def _log_integrand(delta: float) -> float:
        log_lik = stats.norm.logpdf(t_stat, loc=delta * math.sqrt(n), scale=1.0)
        log_prior = stats.cauchy.logpdf(delta, loc=0.0, scale=r)
        return log_lik + log_prior

    # Numerical integration
    from scipy.integrate import quad
    lh0 = float(stats.norm.pdf(t_stat, loc=0.0, scale=1.0))
    # Integrate out delta for H1 marginal likelihood
    def _integrand(delta: float) -> float:
        return math.exp(_log_integrand(delta))
    lh1, _ = quad(_integrand, -20, 20)
    if lh0 < 1e-300:
        log_bf = 0.0
    else:
        log_bf = math.log(max(lh1, 1e-300)) - math.log(lh0)
    bf = math.exp(min(log_bf, 700))
    interp = "Strong evidence for H1" if bf > 10 else ("Moderate" if bf > 3 else "Weak/H0")
    return BayesFactor(log_bf=log_bf, bf=bf, interpretation=interp)


# ---------------------------------------------------------------------------
# 7. HDI (Highest Density Interval)
# ---------------------------------------------------------------------------

def hdi_from_samples(samples: np.ndarray, level: float = 0.95) -> CredibleInterval:
    """
    Compute Highest Density Interval (HDI) from MCMC or other samples.
    The HDI is the shortest interval containing `level` probability mass.
    """
    samples = np.sort(np.asarray(samples, dtype=float))
    n = len(samples)
    n_included = int(math.floor(level * n))
    if n_included < 1:
        return CredibleInterval(lower=samples[0], upper=samples[-1],
                                 level=level, method='hdi')
    # All possible intervals of width n_included
    widths = samples[n_included:] - samples[:n - n_included]
    idx = int(np.argmin(widths))
    return CredibleInterval(lower=float(samples[idx]),
                             upper=float(samples[idx + n_included]),
                             level=level, method='hdi')


def hdi_from_distribution(dist,
                            level: float = 0.95,
                            n_samples: int = 100_000) -> CredibleInterval:
    """HDI from a scipy.stats frozen distribution via sampling."""
    samples = dist.rvs(size=n_samples, random_state=42)
    return hdi_from_samples(samples, level=level)


# ---------------------------------------------------------------------------
# 8. Bayesian Information Criterion as Marginal Likelihood Proxy
# ---------------------------------------------------------------------------

def bic_marginal_likelihood(log_likelihood: float, n_params: int,
                              n_obs: int) -> float:
    """
    BIC approximation to log marginal likelihood:
      log P(data | model) ≈ log_likelihood - 0.5 * k * log(n)
    Useful for comparing models without full integration.
    """
    return log_likelihood - 0.5 * n_params * math.log(n_obs)


def bic_bayes_factor(ll1: float, k1: int, ll2: float, k2: int,
                      n_obs: int) -> BayesFactor:
    """
    Approximate Bayes factor between two models using BIC.
    BF ≈ exp(bic_ml1 - bic_ml2)
    """
    bml1 = bic_marginal_likelihood(ll1, k1, n_obs)
    bml2 = bic_marginal_likelihood(ll2, k2, n_obs)
    log_bf = bml1 - bml2
    bf = math.exp(min(log_bf, 700))
    interp = "Strong evidence for M1" if bf > 10 else ("Moderate" if bf > 3 else "Weak/M0")
    return BayesFactor(log_bf=log_bf, bf=bf, interpretation=interp)


# ---------------------------------------------------------------------------
# 9. Empirical Bayes: Estimate Hyperparameters from Data
# ---------------------------------------------------------------------------

def empirical_bayes_normal(observations: np.ndarray,
                            prior_alpha_0: float = 2.0,
                            prior_beta_0: float = 1.0) -> Tuple[float, float]:
    """
    Estimate NIG hyperparameters (mu_0, kappa_0) via Type-II MLE (marginal likelihood).

    Returns (mu_0_hat, kappa_0_hat).
    """
    obs = np.asarray(observations, dtype=float)
    n = len(obs)
    xbar = float(obs.mean())
    ss = float(np.sum((obs - xbar) ** 2))

    def _neg_log_marginal(params: np.ndarray) -> float:
        mu_0, log_kappa_0 = params
        kappa_0 = math.exp(log_kappa_0)
        alpha_n = prior_alpha_0 + n / 2.0
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + n * xbar) / kappa_n
        beta_n = (prior_beta_0 + 0.5 * ss
                  + kappa_0 * n * (xbar - mu_0) ** 2 / (2.0 * kappa_n))
        # Log marginal likelihood of NIG model
        log_ml = (special.gammaln(alpha_n) - special.gammaln(prior_alpha_0)
                  + prior_alpha_0 * math.log(prior_beta_0)
                  - alpha_n * math.log(beta_n)
                  + 0.5 * math.log(kappa_0 / kappa_n)
                  - n / 2.0 * math.log(2 * math.pi))
        return -log_ml

    x0 = np.array([xbar, 0.0])
    try:
        result = optimize.minimize(_neg_log_marginal, x0, method='Nelder-Mead',
                                    options={'maxiter': 2000, 'xatol': 1e-6})
        mu_0_hat = float(result.x[0])
        kappa_0_hat = float(math.exp(result.x[1]))
    except Exception:
        mu_0_hat = xbar
        kappa_0_hat = 1.0
    return mu_0_hat, kappa_0_hat


# ---------------------------------------------------------------------------
# 10. Hierarchical Model: Pooling Alpha Estimates Across Assets
# ---------------------------------------------------------------------------

def hierarchical_normal_pooling(asset_means: np.ndarray,
                                  asset_stds: np.ndarray,
                                  n_obs_per_asset: Optional[np.ndarray] = None) -> HierarchicalModel:
    """
    Two-level hierarchical Normal model (Stein-James shrinkage).

    Level 1: x_i | mu_i ~ N(mu_i, sigma_i^2)   (known sigma_i = asset_stds)
    Level 2: mu_i | theta, tau^2 ~ N(theta, tau^2)  (hyperprior)

    Returns shrinkage estimates of mu_i towards group mean theta.

    Parameters
    ----------
    asset_means : (n,) observed mean returns per asset
    asset_stds  : (n,) standard errors of the mean per asset
    n_obs_per_asset : (n,) number of observations per asset (for weighting)
    """
    means = np.asarray(asset_means, dtype=float)
    stds = np.asarray(asset_stds, dtype=float)
    n = len(means)
    if n_obs_per_asset is None:
        n_obs_per_asset = np.ones(n)
    n_obs = np.asarray(n_obs_per_asset, dtype=float)

    # Estimate hyperparameters: theta = weighted average, tau^2 via method of moments
    sigma_sq = stds ** 2
    # Precision weights
    prec = 1.0 / (sigma_sq + 1e-15)
    theta = float(np.sum(prec * means) / np.sum(prec))
    # Moment estimate of tau^2: tau^2 = max(0, (Q - (n-1)) / (Σ prec - Σ prec^2/Σ prec))
    Q = float(np.sum(prec * (means - theta) ** 2))
    denom_tau = float(np.sum(prec) - np.sum(prec ** 2) / np.sum(prec))
    tau_sq = max(0.0, (Q - (n - 1)) / (denom_tau + 1e-15))

    # Posterior means (shrinkage to group mean)
    # B_i = sigma_i^2 / (sigma_i^2 + tau^2)  — shrinkage factor
    B = sigma_sq / (sigma_sq + tau_sq + 1e-15)  # shrinkage towards prior
    pooled_means = B * theta + (1.0 - B) * means
    # Posterior variances
    post_vars = (1.0 - B) * sigma_sq

    return HierarchicalModel(
        asset_means=means,
        asset_variances=sigma_sq,
        group_mean=theta,
        group_variance=tau_sq,
        pooled_means=pooled_means,
        shrinkage_factors=B,
    )


# ---------------------------------------------------------------------------
# 11. Convenience: full Bayesian summary for a return series
# ---------------------------------------------------------------------------

@dataclass
class BayesianReturnSummary:
    posterior_mean: float
    posterior_std: float
    credible_interval_95: CredibleInterval
    hdi_95: CredibleInterval
    posterior_vol: float         # E[sigma | data]
    predictive_std: float
    n_obs: int
    nig_state: NIG_State


def bayesian_return_summary(returns: np.ndarray,
                              mu_0: float = 0.0,
                              kappa_0: float = 1.0,
                              alpha_0: float = 2.0,
                              beta_0: float = 0.01) -> BayesianReturnSummary:
    """
    Full Bayesian summary for a return series using NIG conjugate model.
    """
    obs = np.asarray(returns, dtype=float)
    state = nig_prior(mu_0, kappa_0, alpha_0, beta_0)
    state = nig_update(state, obs)
    mu_dist = nig_marginal_mu(state)
    ci = CredibleInterval(lower=float(mu_dist.ppf(0.025)),
                           upper=float(mu_dist.ppf(0.975)),
                           level=0.95, method='equal_tailed')
    # HDI via sampling
    samples = mu_dist.rvs(size=50_000, random_state=0)
    hdi = hdi_from_samples(samples, level=0.95)
    pred_dist = nig_predictive(state)
    return BayesianReturnSummary(
        posterior_mean=state.mu_n,
        posterior_std=float(mu_dist.std()),
        credible_interval_95=ci,
        hdi_95=hdi,
        posterior_vol=math.sqrt(max(state.posterior_mean_sigma2, 0.0)),
        predictive_std=float(pred_dist.std()),
        n_obs=state.n_obs,
        nig_state=state,
    )
