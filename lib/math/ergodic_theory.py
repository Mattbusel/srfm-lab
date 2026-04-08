"""
Ergodic economics and time-average returns.

Implements the key ideas from Ole Peters' ergodicity economics: the distinction
between ensemble and time averages, Kelly criterion, leverage optimization,
gamble evaluation, and wealth dynamics simulation.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy import stats


# ---------------------------------------------------------------------------
# Geometric vs arithmetic mean
# ---------------------------------------------------------------------------

def arithmetic_mean_return(returns: np.ndarray) -> float:
    """
    Ensemble (arithmetic) average of returns.
    E[r] = (1/N) sum r_i
    This is what you expect on average across many parallel realizations.
    """
    return np.mean(returns)


def geometric_mean_return(returns: np.ndarray) -> float:
    """
    Time-average (geometric) growth rate.
    g = exp(E[log(1+r)]) - 1
    This is what a single individual experiences over time.
    """
    log_growth = np.mean(np.log(1.0 + returns))
    return np.exp(log_growth) - 1.0


def log_growth_rate(returns: np.ndarray) -> float:
    """
    Continuously compounded growth rate: E[log(1+r)].
    This is the quantity that matters for long-run wealth.
    """
    return np.mean(np.log(1.0 + returns))


def volatility_drag(mu: float, sigma: float) -> float:
    """
    Volatility drag: the difference between arithmetic and geometric means.
    For lognormal returns: g ≈ mu - sigma^2/2

    Parameters
    ----------
    mu : arithmetic mean return
    sigma : standard deviation of returns

    Returns
    -------
    drag : approximate reduction in growth rate due to volatility
    """
    return sigma ** 2 / 2.0


def exact_geometric_growth(mu: float, sigma: float) -> float:
    """
    Exact geometric growth rate for lognormal returns.
    g = mu - sigma^2 / 2
    """
    return mu - sigma ** 2 / 2.0


# ---------------------------------------------------------------------------
# Ensemble vs time average demonstration
# ---------------------------------------------------------------------------

def ensemble_vs_time_average(n_agents: int = 10000, n_steps: int = 1000,
                             gain: float = 0.5, loss: float = 0.4,
                             prob_gain: float = 0.5,
                             seed: int = 42) -> dict:
    """
    Simulation proof that ensemble average and time average diverge for
    multiplicative dynamics.

    A fair-looking gamble: gain +50% or lose -40% with equal probability.
    Ensemble average: E[W] grows (multiplicative expectation > 1).
    Time average: individual wealth almost surely declines.

    Parameters
    ----------
    n_agents : number of parallel agents (ensemble size)
    n_steps : number of time steps
    gain : fractional gain (e.g. 0.5 for +50%)
    loss : fractional loss (e.g. 0.4 for -40%)
    prob_gain : probability of gain

    Returns
    -------
    dict with ensemble trajectory, individual trajectories, statistics
    """
    rng = np.random.RandomState(seed)

    wealth = np.ones(n_agents)
    ensemble_means = [1.0]
    ensemble_medians = [1.0]
    sample_trajectories = np.ones((min(20, n_agents), n_steps + 1))

    # multiplicative factors
    up = 1.0 + gain
    down = 1.0 - loss

    # expected growth rates
    arithmetic_growth = prob_gain * gain - (1 - prob_gain) * loss
    geometric_growth = prob_gain * np.log(up) + (1 - prob_gain) * np.log(down)

    for t in range(n_steps):
        outcomes = rng.random(n_agents) < prob_gain
        factors = np.where(outcomes, up, down)
        wealth *= factors

        ensemble_means.append(np.mean(wealth))
        ensemble_medians.append(np.median(wealth))

        for i in range(min(20, n_agents)):
            sample_trajectories[i, t + 1] = wealth[i]

    # fraction of agents that grew
    fraction_grew = np.mean(wealth > 1.0)

    return {
        'ensemble_means': np.array(ensemble_means),
        'ensemble_medians': np.array(ensemble_medians),
        'final_wealth': wealth,
        'sample_trajectories': sample_trajectories,
        'arithmetic_growth_rate': arithmetic_growth,
        'geometric_growth_rate': geometric_growth,
        'fraction_grew': fraction_grew,
        'mean_final_wealth': np.mean(wealth),
        'median_final_wealth': np.median(wealth),
        'ergodic': geometric_growth > 0,
    }


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------

def kelly_fraction(p: float, b: float) -> float:
    """
    Kelly criterion for a simple bet.

    Bet fraction f that maximizes E[log(1 + f*X)] where X = b with prob p,
    X = -1 with prob (1-p).

    f* = p/1 - (1-p)/b = p - (1-p)/b  (for odds b:1)

    Parameters
    ----------
    p : probability of winning
    b : net odds (win b for every 1 wagered)

    Returns
    -------
    Optimal fraction of wealth to bet
    """
    return p - (1.0 - p) / b


def kelly_fraction_general(returns: np.ndarray, probabilities: np.ndarray = None) -> float:
    """
    General Kelly criterion for discrete outcomes.

    Maximize E[log(1 + f * r)] over f.

    Parameters
    ----------
    returns : array of possible returns
    probabilities : array of probabilities (uniform if None)
    """
    if probabilities is None:
        probabilities = np.ones(len(returns)) / len(returns)

    def neg_growth(f):
        wealth_factors = 1.0 + f * returns
        # handle bankruptcy
        if np.any(wealth_factors <= 0):
            return 1e10
        return -np.sum(probabilities * np.log(wealth_factors))

    # find bounds: f must keep 1 + f*r > 0 for all r
    min_ret = np.min(returns)
    max_ret = np.max(returns)

    if min_ret >= 0:
        f_min = 0.0
        f_max = 10.0
    elif max_ret <= 0:
        f_min = -10.0
        f_max = 0.0
    else:
        f_min = -0.99 / max_ret if max_ret > 0 else -10.0
        f_max = -0.99 / min_ret if min_ret < 0 else 10.0

    result = minimize_scalar(neg_growth, bounds=(f_min, f_max), method='bounded')
    return result.x


def kelly_continuous(mu: float, sigma: float, rf: float = 0.0) -> float:
    """
    Continuous Kelly criterion for lognormal returns.
    f* = (mu - rf) / sigma^2

    This is the leverage that maximizes geometric growth rate.
    """
    return (mu - rf) / (sigma ** 2)


def kelly_portfolio(mu: np.ndarray, Sigma: np.ndarray,
                    rf: float = 0.0) -> np.ndarray:
    """
    Multi-asset Kelly portfolio (continuous case).
    f* = Sigma^{-1} (mu - rf)

    Note: This often produces aggressive leveraged positions.
    Practitioners typically use fractional Kelly (e.g. half-Kelly).
    """
    excess = mu - rf
    return np.linalg.solve(Sigma, excess)


def fractional_kelly(full_kelly: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """
    Fractional Kelly: reduce position by a fraction for safety.
    Common choices: 0.5 (half-Kelly), 0.25 (quarter-Kelly).
    """
    return fraction * full_kelly


# ---------------------------------------------------------------------------
# Leverage optimization
# ---------------------------------------------------------------------------

def optimal_leverage(mu: float, sigma: float, rf: float = 0.0,
                     higher_moments: bool = False,
                     skew: float = 0.0, kurt: float = 3.0) -> dict:
    """
    Find optimal leverage ratio maximizing geometric growth rate.

    For lognormal: g(f) = rf + f*(mu - rf) - f^2 * sigma^2 / 2
    Optimal: f* = (mu - rf) / sigma^2

    With higher moments (Edgeworth expansion):
    g(f) ≈ rf + f*(mu-rf) - f^2*sigma^2/2 + f^3*sigma^3*skew/6
           - f^4*sigma^4*(kurt-3)/24
    """
    if not higher_moments:
        f_star = (mu - rf) / (sigma ** 2)
        g_star = rf + f_star * (mu - rf) - f_star ** 2 * sigma ** 2 / 2.0
        g_unlev = mu - sigma ** 2 / 2.0

        return {
            'optimal_leverage': f_star,
            'growth_rate': g_star,
            'unleveraged_growth': g_unlev,
            'leverage_benefit': g_star - g_unlev,
        }

    # with higher moments: numerical optimization
    def neg_growth(f):
        g = (rf + f * (mu - rf)
             - f ** 2 * sigma ** 2 / 2.0
             + f ** 3 * sigma ** 3 * skew / 6.0
             - f ** 4 * sigma ** 4 * (kurt - 3.0) / 24.0)
        return -g

    result = minimize_scalar(neg_growth, bounds=(0, 10), method='bounded')
    f_star = result.x
    g_star = -result.fun

    return {
        'optimal_leverage': f_star,
        'growth_rate': g_star,
        'unleveraged_growth': mu - sigma ** 2 / 2.0,
    }


def leverage_growth_curve(mu: float, sigma: float, rf: float = 0.0,
                          max_leverage: float = 5.0,
                          n_points: int = 200) -> dict:
    """
    Compute growth rate as a function of leverage.
    Shows the parabolic shape and the danger of over-leveraging.
    """
    leverages = np.linspace(0, max_leverage, n_points)
    growth_rates = np.zeros(n_points)

    for i, f in enumerate(leverages):
        growth_rates[i] = rf + f * (mu - rf) - f ** 2 * sigma ** 2 / 2.0

    f_star = (mu - rf) / (sigma ** 2)
    g_star = rf + f_star * (mu - rf) - f_star ** 2 * sigma ** 2 / 2.0

    # leverage at which growth = 0
    # 0 = f*(mu-rf) - f^2*sigma^2/2 => f = 2*(mu-rf)/sigma^2
    f_zero = 2.0 * (mu - rf) / (sigma ** 2) if sigma > 0 else np.inf

    return {
        'leverages': leverages,
        'growth_rates': growth_rates,
        'optimal_leverage': f_star,
        'optimal_growth': g_star,
        'zero_growth_leverage': f_zero,
    }


# ---------------------------------------------------------------------------
# Peters' ergodic gamble criterion
# ---------------------------------------------------------------------------

def peters_criterion(gain: float, loss: float,
                     prob_gain: float = 0.5) -> dict:
    """
    Evaluate a gamble using Peters' ergodicity criterion.

    Accept the gamble if the time-average growth rate is positive:
    g = p * log(1 + gain) + (1-p) * log(1 - loss) > 0

    Compare with the ensemble criterion (expected value):
    E = p * gain - (1-p) * loss

    These can disagree: a gamble can have positive expectation but
    negative time-average growth (and vice versa is impossible for
    multiplicative dynamics).
    """
    time_avg = prob_gain * np.log(1.0 + gain) + (1.0 - prob_gain) * np.log(1.0 - loss)
    ensemble_avg = prob_gain * gain - (1.0 - prob_gain) * loss

    accept_ergodic = time_avg > 0
    accept_ensemble = ensemble_avg > 0

    # optimal Kelly fraction for this gamble
    f_kelly = kelly_fraction(prob_gain, gain / loss) if loss > 0 else np.inf

    return {
        'time_average_growth': time_avg,
        'ensemble_average': ensemble_avg,
        'accept_ergodic': accept_ergodic,
        'accept_ensemble': accept_ensemble,
        'disagree': accept_ergodic != accept_ensemble,
        'kelly_fraction': f_kelly,
    }


def evaluate_gamble_sequence(gains: np.ndarray, losses: np.ndarray,
                             probabilities: np.ndarray) -> dict:
    """
    Evaluate a sequence of gambles (compound game).

    Parameters
    ----------
    gains : array of potential gains per gamble
    losses : array of potential losses per gamble
    probabilities : array of win probabilities per gamble
    """
    n = len(gains)
    time_avg_total = 0.0
    ensemble_avg_total = 0.0
    individual_results = []

    for i in range(n):
        res = peters_criterion(gains[i], losses[i], probabilities[i])
        time_avg_total += res['time_average_growth']
        ensemble_avg_total += res['ensemble_average']
        individual_results.append(res)

    return {
        'total_time_average_growth': time_avg_total,
        'total_ensemble_average': ensemble_avg_total,
        'accept_compound': time_avg_total > 0,
        'individual_results': individual_results,
    }


# ---------------------------------------------------------------------------
# Non-ergodic portfolio construction
# ---------------------------------------------------------------------------

def max_geometric_growth_portfolio(mu: np.ndarray, Sigma: np.ndarray,
                                   rf: float = 0.0,
                                   allow_leverage: bool = False,
                                   max_leverage: float = 2.0) -> dict:
    """
    Construct portfolio maximizing geometric growth rate.

    max_w  mu^T w - 0.5 * w^T Sigma w
    s.t.   sum(w) = 1 (or <= max_leverage if leveraged)
           w >= 0 (long only) or unconstrained

    This is the Kelly-optimal portfolio, equivalent to maximizing
    the expected log return.
    """
    n = len(mu)
    excess = mu - rf

    if not allow_leverage:
        # constrained optimization
        from scipy.optimize import minimize as sp_minimize

        def neg_growth(w):
            return -(excess @ w - 0.5 * w @ Sigma @ w + rf)

        def neg_growth_grad(w):
            return -(excess - Sigma @ w)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]
        bounds = [(0, 1)] * n

        x0 = np.ones(n) / n
        result = sp_minimize(neg_growth, x0, jac=neg_growth_grad,
                             method='SLSQP', bounds=bounds,
                             constraints=constraints)
        w = result.x
    else:
        # unconstrained Kelly
        w = np.linalg.solve(Sigma, excess)
        # scale to max leverage
        total = np.sum(np.abs(w))
        if total > max_leverage:
            w = w * max_leverage / total

    growth = excess @ w - 0.5 * w @ Sigma @ w + rf
    vol = np.sqrt(w @ Sigma @ w)
    arithmetic_ret = mu @ w

    return {
        'weights': w,
        'geometric_growth': growth,
        'arithmetic_return': arithmetic_ret,
        'volatility': vol,
        'volatility_drag': 0.5 * w @ Sigma @ w,
        'sharpe_ratio': (arithmetic_ret - rf) / vol if vol > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Wealth dynamics simulation
# ---------------------------------------------------------------------------

def simulate_wealth_dynamics(strategies: dict, returns: np.ndarray,
                             n_simulations: int = 1000,
                             n_periods: int = 252,
                             seed: int = 42) -> dict:
    """
    Compare wealth trajectories of different strategies.

    Parameters
    ----------
    strategies : dict mapping name -> weight vector
    returns : (n_periods, n_assets) historical or simulated returns
    n_simulations : number of Monte Carlo paths
    n_periods : periods per path

    Returns
    -------
    dict with wealth paths, growth rates, and comparisons for each strategy
    """
    rng = np.random.RandomState(seed)
    n_assets = returns.shape[1] if returns.ndim > 1 else 1
    T = returns.shape[0]

    results = {}

    for name, weights in strategies.items():
        w = np.asarray(weights)
        wealth_paths = np.ones((n_simulations, n_periods + 1))

        for sim in range(n_simulations):
            # bootstrap returns
            indices = rng.choice(T, size=n_periods, replace=True)
            sim_returns = returns[indices]

            if sim_returns.ndim == 1:
                port_returns = sim_returns * w
            else:
                port_returns = sim_returns @ w

            # multiplicative wealth accumulation
            for t in range(n_periods):
                wealth_paths[sim, t + 1] = wealth_paths[sim, t] * (1.0 + port_returns[t])

        final_wealth = wealth_paths[:, -1]

        results[name] = {
            'wealth_paths': wealth_paths,
            'final_wealth': final_wealth,
            'mean_final': np.mean(final_wealth),
            'median_final': np.median(final_wealth),
            'geometric_growth': np.mean(np.log(final_wealth)) / n_periods,
            'arithmetic_growth': np.mean(final_wealth - 1.0) / n_periods,
            'prob_profit': np.mean(final_wealth > 1.0),
            'prob_ruin': np.mean(final_wealth < 0.1),
            'percentiles': {
                '5th': np.percentile(final_wealth, 5),
                '25th': np.percentile(final_wealth, 25),
                '50th': np.percentile(final_wealth, 50),
                '75th': np.percentile(final_wealth, 75),
                '95th': np.percentile(final_wealth, 95),
            },
        }

    return results


def compare_ergodic_strategies(mu: float = 0.10, sigma: float = 0.20,
                               rf: float = 0.02,
                               n_periods: int = 1000,
                               n_simulations: int = 5000,
                               seed: int = 42) -> dict:
    """
    Compare strategies that look identical under ensemble averaging
    but differ under time averaging.

    Strategies:
    1. Full Kelly: f* = (mu-rf)/sigma^2
    2. Half Kelly: f*/2
    3. Double Kelly: 2*f* (over-leveraged)
    4. Fixed: always invest 100%
    """
    rng = np.random.RandomState(seed)
    f_star = (mu - rf) / (sigma ** 2)

    strategies = {
        'full_kelly': f_star,
        'half_kelly': f_star / 2.0,
        'double_kelly': 2.0 * f_star,
        'fixed_100pct': 1.0,
    }

    results = {}

    for name, f in strategies.items():
        wealth = np.ones(n_simulations)
        log_wealth = np.zeros((n_simulations, n_periods + 1))

        for t in range(n_periods):
            r = rng.normal(mu / 252, sigma / np.sqrt(252), n_simulations)
            portfolio_r = rf / 252 + f * (r - rf / 252)
            wealth *= (1.0 + portfolio_r)
            wealth = np.maximum(wealth, 1e-15)  # prevent exact zero
            log_wealth[:, t + 1] = np.log(wealth)

        # theoretical growth rate
        g_theory = rf + f * (mu - rf) - 0.5 * f ** 2 * sigma ** 2

        results[name] = {
            'leverage': f,
            'mean_final_wealth': np.mean(wealth),
            'median_final_wealth': np.median(wealth),
            'geometric_growth_empirical': np.mean(np.log(wealth)) / n_periods,
            'geometric_growth_theory': g_theory / 252,
            'prob_profit': np.mean(wealth > 1.0),
            'prob_ruin_50pct': np.mean(wealth < 0.5),
            'mean_log_wealth': log_wealth[:, -1].mean(),
            'std_log_wealth': log_wealth[:, -1].std(),
        }

    return {
        'results': results,
        'kelly_fraction': f_star,
        'parameters': {'mu': mu, 'sigma': sigma, 'rf': rf},
    }


# ---------------------------------------------------------------------------
# Tail exponent and growth rate
# ---------------------------------------------------------------------------

def tail_exponent_growth(returns: np.ndarray, threshold_quantile: float = 0.95) -> dict:
    """
    Estimate the tail exponent and its relationship to growth rate.

    Heavy tails reduce geometric growth more than light tails for the
    same arithmetic mean and variance. The tail exponent alpha determines
    how severe the volatility drag is.
    """
    n = len(returns)

    # estimate tail exponent via Hill estimator
    sorted_abs = np.sort(np.abs(returns))[::-1]
    threshold_idx = max(1, int(n * (1 - threshold_quantile)))
    threshold = sorted_abs[threshold_idx]

    exceedances = sorted_abs[:threshold_idx]
    if len(exceedances) > 1 and threshold > 0:
        hill_estimate = np.mean(np.log(exceedances / threshold))
        alpha = 1.0 / hill_estimate if hill_estimate > 0 else np.inf
    else:
        alpha = np.inf

    # growth rates
    mu = np.mean(returns)
    sigma = np.std(returns)
    g_empirical = np.mean(np.log(1.0 + returns))
    g_gaussian_approx = mu - sigma ** 2 / 2.0

    # excess kurtosis contribution to growth drag
    kurt = stats.kurtosis(returns, fisher=True)
    g_kurtosis_correction = mu - sigma ** 2 / 2.0 - kurt * sigma ** 4 / 24.0

    return {
        'tail_exponent': alpha,
        'hill_estimate': 1.0 / alpha if alpha < np.inf else 0.0,
        'arithmetic_mean': mu,
        'volatility': sigma,
        'empirical_growth': g_empirical,
        'gaussian_growth_approx': g_gaussian_approx,
        'kurtosis_adjusted_growth': g_kurtosis_correction,
        'excess_kurtosis': kurt,
        'growth_drag': mu - g_empirical,
        'gaussian_drag': sigma ** 2 / 2.0,
    }


# ---------------------------------------------------------------------------
# Rebalancing premium
# ---------------------------------------------------------------------------

def rebalancing_premium(mu: np.ndarray, sigma: np.ndarray,
                        corr: np.ndarray, weights: np.ndarray) -> dict:
    """
    Compute the rebalancing premium (diversification return).

    The geometric return of a rebalanced portfolio exceeds the weighted
    average of component geometric returns. This "free lunch" arises from
    the concavity of log and is a purely ergodic phenomenon.

    Parameters
    ----------
    mu : arithmetic mean returns per asset
    sigma : volatilities per asset
    corr : correlation matrix
    weights : portfolio weights (sum to 1)
    """
    n = len(mu)
    Sigma = np.outer(sigma, sigma) * corr

    # portfolio arithmetic return
    port_mu = weights @ mu

    # portfolio variance
    port_var = weights @ Sigma @ weights
    port_sigma = np.sqrt(port_var)

    # portfolio geometric growth
    port_geo = port_mu - port_var / 2.0

    # weighted average of component geometric growths
    component_geo = mu - sigma ** 2 / 2.0
    weighted_component_geo = weights @ component_geo

    # rebalancing premium
    premium = port_geo - weighted_component_geo

    # diversification ratio
    weighted_vol = weights @ sigma
    div_ratio = weighted_vol / port_sigma if port_sigma > 0 else 1.0

    # excess growth rate (Fernholz)
    excess_growth = 0.5 * (np.sum(weights * sigma ** 2) - port_var)

    return {
        'portfolio_arithmetic_return': port_mu,
        'portfolio_geometric_growth': port_geo,
        'weighted_component_geometric': weighted_component_geo,
        'rebalancing_premium': premium,
        'excess_growth_rate': excess_growth,
        'diversification_ratio': div_ratio,
        'portfolio_volatility': port_sigma,
        'volatility_drag': port_var / 2.0,
    }


def rebalancing_premium_simulation(mu: np.ndarray, Sigma: np.ndarray,
                                   weights: np.ndarray,
                                   n_periods: int = 10000,
                                   n_simulations: int = 100,
                                   rebalance_freq: int = 1,
                                   seed: int = 42) -> dict:
    """
    Simulate the rebalancing premium: compare rebalanced vs buy-and-hold.
    """
    rng = np.random.RandomState(seed)
    n_assets = len(mu)
    daily_mu = mu / 252
    daily_Sigma = Sigma / 252

    L = np.linalg.cholesky(daily_Sigma)

    rebal_wealth = np.ones(n_simulations)
    bah_wealth = np.ones(n_simulations)

    for sim in range(n_simulations):
        w_rebal = weights.copy()
        w_bah = weights.copy()
        W_rebal = 1.0
        W_bah = 1.0

        for t in range(n_periods):
            z = rng.randn(n_assets)
            r = daily_mu + L @ z

            # rebalanced
            W_rebal *= (1.0 + w_rebal @ r)
            if (t + 1) % rebalance_freq == 0:
                w_rebal = weights.copy()  # rebalance to target

            # buy and hold: weights drift
            asset_values = w_bah * (1.0 + r)
            W_bah_new = np.sum(asset_values)
            w_bah = asset_values / W_bah_new if W_bah_new > 0 else w_bah
            W_bah *= (1.0 + w_bah @ r)

        rebal_wealth[sim] = W_rebal
        bah_wealth[sim] = W_bah

    rebal_growth = np.mean(np.log(rebal_wealth)) / n_periods
    bah_growth = np.mean(np.log(bah_wealth)) / n_periods

    return {
        'rebalanced_growth': rebal_growth,
        'buy_hold_growth': bah_growth,
        'rebalancing_premium_empirical': rebal_growth - bah_growth,
        'rebalanced_mean_wealth': np.mean(rebal_wealth),
        'buy_hold_mean_wealth': np.mean(bah_wealth),
        'rebalanced_median_wealth': np.median(rebal_wealth),
        'buy_hold_median_wealth': np.median(bah_wealth),
    }


# ---------------------------------------------------------------------------
# Volatility drag quantification
# ---------------------------------------------------------------------------

def volatility_drag_analysis(returns: np.ndarray,
                             window: int = 60) -> dict:
    """
    Detailed analysis of volatility drag: g ≈ mu - sigma^2/2.

    Computes rolling estimates of arithmetic mean, geometric mean,
    and the drag term, showing how they relate over time.
    """
    n = len(returns)
    arithmetic_means = np.full(n, np.nan)
    geometric_means = np.full(n, np.nan)
    volatilities = np.full(n, np.nan)
    drag_terms = np.full(n, np.nan)
    approximation_errors = np.full(n, np.nan)

    for t in range(window, n):
        r = returns[t - window:t]
        mu_hat = np.mean(r)
        sigma_hat = np.std(r)
        g_hat = np.mean(np.log(1.0 + r))

        arithmetic_means[t] = mu_hat
        geometric_means[t] = g_hat
        volatilities[t] = sigma_hat
        drag_terms[t] = sigma_hat ** 2 / 2.0
        approximation_errors[t] = g_hat - (mu_hat - sigma_hat ** 2 / 2.0)

    # overall statistics
    valid = ~np.isnan(arithmetic_means)
    mu_overall = np.mean(returns)
    sigma_overall = np.std(returns)
    g_overall = np.mean(np.log(1.0 + returns))

    return {
        'arithmetic_means': arithmetic_means,
        'geometric_means': geometric_means,
        'volatilities': volatilities,
        'drag_terms': drag_terms,
        'approximation_errors': approximation_errors,
        'overall': {
            'arithmetic_mean': mu_overall,
            'geometric_mean': g_overall,
            'volatility': sigma_overall,
            'drag': sigma_overall ** 2 / 2.0,
            'actual_drag': mu_overall - g_overall,
            'approximation_error': g_overall - (mu_overall - sigma_overall ** 2 / 2.0),
        },
        'window': window,
    }


# ---------------------------------------------------------------------------
# Ergodic utility and wealth dynamics
# ---------------------------------------------------------------------------

def isoelastic_utility_growth(returns: np.ndarray, eta: float = 1.0) -> float:
    """
    Growth rate under isoelastic (CRRA) utility.

    For eta = 1 (log utility): g = E[log(1+r)]  (Kelly criterion)
    For eta != 1: g = E[(1+r)^(1-eta)] / (1-eta)

    Peters shows that the appropriate utility function for a given
    wealth dynamic makes the corresponding growth rate ergodic.
    """
    if abs(eta - 1.0) < 1e-10:
        return np.mean(np.log(1.0 + returns))
    else:
        return np.mean((1.0 + returns) ** (1.0 - eta)) / (1.0 - eta)


def optimal_eta_for_dynamics(returns: np.ndarray,
                             dynamics: str = 'multiplicative') -> dict:
    """
    Find the risk aversion parameter eta that corresponds to ergodic
    optimization under given wealth dynamics.

    For multiplicative dynamics: eta = 1 (log utility)
    For additive dynamics: eta = 0 (linear/risk-neutral)
    For general dynamics: numerically determined
    """
    if dynamics == 'multiplicative':
        eta_star = 1.0
        growth = np.mean(np.log(1.0 + returns))
    elif dynamics == 'additive':
        eta_star = 0.0
        growth = np.mean(returns)
    else:
        # numerical search
        def neg_growth(eta):
            return -isoelastic_utility_growth(returns, eta)

        result = minimize_scalar(neg_growth, bounds=(0.1, 5.0), method='bounded')
        eta_star = result.x
        growth = -result.fun

    return {
        'optimal_eta': eta_star,
        'growth_rate': growth,
        'dynamics': dynamics,
    }
