"""
Point process models for event-driven finance.

Implements Poisson, Hawkes, Cox, self-correcting, and marked point processes
with simulation, estimation, and diagnostics.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import kstest, expon, weibull_min, lognorm
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Homogeneous Poisson process
# ---------------------------------------------------------------------------

def simulate_poisson(
    rate: float,
    T: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate a homogeneous Poisson process on [0, T].

    Returns array of event times.
    """
    rng = np.random.default_rng(seed)
    n_events = rng.poisson(rate * T)
    times = np.sort(rng.uniform(0, T, n_events))
    return times


def poisson_mle(event_times: np.ndarray, T: float) -> dict:
    """MLE for homogeneous Poisson process rate."""
    n = len(event_times)
    rate = n / T
    rate_se = np.sqrt(n) / T  # approximate SE
    return {"rate": rate, "std_err": rate_se, "n_events": n, "T": T}


# ---------------------------------------------------------------------------
# Inhomogeneous Poisson process
# ---------------------------------------------------------------------------

def simulate_inhomogeneous_poisson(
    rate_func,
    T: float,
    rate_upper: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate inhomogeneous Poisson process via thinning (Ogata).

    Parameters
    ----------
    rate_func : callable(t) -> intensity at time t
    T : time horizon
    rate_upper : upper bound on rate_func over [0, T]
    seed : random seed

    Returns
    -------
    Array of event times.
    """
    rng = np.random.default_rng(seed)
    times = []
    t = 0.0
    while t < T:
        dt = rng.exponential(1.0 / rate_upper)
        t += dt
        if t >= T:
            break
        u = rng.uniform()
        if u <= rate_func(t) / rate_upper:
            times.append(t)
    return np.array(times)


# ---------------------------------------------------------------------------
# Univariate Hawkes process
# ---------------------------------------------------------------------------

def simulate_hawkes_exp(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate univariate Hawkes process with exponential kernel via Ogata thinning.

    Intensity: lambda(t) = mu + alpha * sum_{t_i < t} beta * exp(-beta * (t - t_i))

    Parameters
    ----------
    mu : baseline intensity
    alpha : excitation magnitude (branching ratio = alpha; need alpha < 1 for stability)
    beta : decay rate
    T : time horizon

    Returns
    -------
    Array of event times.
    """
    rng = np.random.default_rng(seed)
    times = []
    t = 0.0
    lam = mu  # current intensity

    while t < T:
        lam_bar = lam  # upper bound (intensity is maximal right after an event)
        if lam_bar <= 0:
            lam_bar = mu
        dt = rng.exponential(1.0 / max(lam_bar, 1e-10))
        t += dt
        if t >= T:
            break

        # Compute actual intensity at t
        lam = mu
        for ti in times:
            lam += alpha * beta * np.exp(-beta * (t - ti))

        u = rng.uniform()
        if u <= lam / max(lam_bar, 1e-10):
            times.append(t)

    return np.array(times)


def hawkes_intensity(
    times: np.ndarray,
    event_times: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Compute Hawkes intensity at given times."""
    intensity = np.full(len(times), mu)
    for ti in event_times:
        mask = times > ti
        intensity[mask] += alpha * beta * np.exp(-beta * (times[mask] - ti))
    return intensity


def hawkes_loglik_exp(
    params: np.ndarray,
    event_times: np.ndarray,
    T: float,
) -> float:
    """
    Negative log-likelihood for univariate Hawkes with exponential kernel.

    params = [mu, alpha, beta]
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or alpha >= 1 or beta <= 0:
        return 1e15

    n = len(event_times)
    if n == 0:
        return mu * T

    # Log-likelihood: sum log(lambda(t_i)) - integral of lambda
    # Recursive computation for efficiency
    A = 0.0  # A_i = sum_{j<i} exp(-beta*(t_i - t_j))
    log_lik = 0.0

    for i in range(n):
        lam_i = mu + alpha * beta * A
        if lam_i <= 0:
            return 1e15
        log_lik += np.log(lam_i)

        if i < n - 1:
            A = np.exp(-beta * (event_times[i + 1] - event_times[i])) * (1 + A)

    # Integral of lambda: mu*T + alpha * sum_i (1 - exp(-beta*(T - t_i)))
    integral = mu * T + alpha * np.sum(1 - np.exp(-beta * (T - event_times)))
    log_lik -= integral

    return -log_lik


def fit_hawkes_exp(
    event_times: np.ndarray,
    T: float,
    x0: np.ndarray = None,
) -> dict:
    """
    MLE for univariate Hawkes with exponential kernel.

    Returns dict with mu, alpha, beta, branching_ratio, log_likelihood.
    """
    if x0 is None:
        n = len(event_times)
        mu0 = n / (2 * T)
        x0 = np.array([mu0, 0.5, 1.0])

    result = minimize(
        hawkes_loglik_exp,
        x0,
        args=(event_times, T),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    mu, alpha, beta = result.x
    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": alpha,
        "stable": alpha < 1.0,
        "log_likelihood": -result.fun,
        "converged": result.success,
    }


def hawkes_branching_ratio(alpha: float) -> float:
    """Branching ratio for exponential Hawkes. Must be < 1 for stationarity."""
    return alpha


# ---------------------------------------------------------------------------
# Multivariate Hawkes process
# ---------------------------------------------------------------------------

def simulate_multivariate_hawkes(
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    T: float,
    seed: int = 42,
) -> dict:
    """
    Simulate multivariate Hawkes process with exponential kernels.

    lambda_i(t) = mu_i + sum_j alpha_ij * beta_ij * sum_{t_k^j < t} exp(-beta_ij * (t - t_k^j))

    Parameters
    ----------
    mu : (d,) baseline intensities
    alpha : (d, d) excitation matrix (spectral radius < 1 for stability)
    beta : (d, d) decay rates
    T : time horizon

    Returns
    -------
    dict mapping dimension index to array of event times
    """
    rng = np.random.default_rng(seed)
    d = len(mu)
    events = {i: [] for i in range(d)}

    # Simple thinning with global upper bound
    t = 0.0
    max_lam = np.sum(mu) * 3  # rough initial upper bound

    while t < T:
        dt = rng.exponential(1.0 / max(max_lam, 1e-10))
        t += dt
        if t >= T:
            break

        # Compute all intensities
        lam = mu.copy().astype(float)
        for i in range(d):
            for j in range(d):
                for tk in events[j]:
                    lam[i] += alpha[i, j] * beta[i, j] * np.exp(-beta[i, j] * (t - tk))

        total_lam = np.sum(lam)
        u = rng.uniform()
        if u <= total_lam / max(max_lam, 1e-10):
            # Accept: assign to dimension proportional to intensity
            probs = lam / total_lam
            dim = rng.choice(d, p=probs)
            events[dim].append(t)

        max_lam = max(total_lam * 1.5, np.sum(mu))

    for i in range(d):
        events[i] = np.array(events[i])

    return events


def multivariate_hawkes_spectral_radius(alpha: np.ndarray) -> float:
    """Spectral radius of the excitation matrix. Must be < 1 for stationarity."""
    eigenvalues = np.linalg.eigvals(alpha)
    return float(np.max(np.abs(eigenvalues)))


# ---------------------------------------------------------------------------
# Cox process (doubly stochastic Poisson)
# ---------------------------------------------------------------------------

def simulate_cox_process(
    intensity_path: np.ndarray,
    dt: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate Cox process given a pre-computed stochastic intensity path.

    Parameters
    ----------
    intensity_path : 1-D array of intensity values at each time step
    dt : time step
    seed : random seed

    Returns
    -------
    Array of event times.
    """
    rng = np.random.default_rng(seed)
    T = len(intensity_path) * dt
    times = []

    for i, lam in enumerate(intensity_path):
        lam_pos = max(lam, 0.0)
        # Probability of event in [i*dt, (i+1)*dt] ≈ lam * dt (for small dt)
        n_events = rng.poisson(lam_pos * dt)
        for _ in range(n_events):
            times.append(i * dt + rng.uniform(0, dt))

    return np.sort(np.array(times))


def simulate_cox_ou_intensity(
    mu: float,
    theta: float,
    sigma: float,
    lam0: float,
    T: float,
    dt: float = 0.001,
    seed: int = 42,
) -> dict:
    """
    Cox process with OU stochastic intensity.

    d(lambda) = theta*(mu - lambda)*dt + sigma*dW
    lambda clamped to be non-negative.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    intensity = np.zeros(n_steps)
    intensity[0] = lam0

    for t in range(1, n_steps):
        dW = rng.standard_normal() * np.sqrt(dt)
        intensity[t] = intensity[t - 1] + theta * (mu - intensity[t - 1]) * dt + sigma * dW
        intensity[t] = max(intensity[t], 0.0)

    events = simulate_cox_process(intensity, dt, seed=seed + 1)

    return {"event_times": events, "intensity_path": intensity, "time_grid": np.arange(n_steps) * dt}


# ---------------------------------------------------------------------------
# Self-correcting process
# ---------------------------------------------------------------------------

def simulate_self_correcting(
    mu: float,
    alpha: float,
    T: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Self-correcting (inhibitory) process.

    lambda(t) = exp(mu*t - alpha*N(t-))

    Each event reduces the intensity, creating regular spacing.
    """
    rng = np.random.default_rng(seed)
    times = []
    t = 0.0
    N = 0

    while t < T:
        # Intensity at current time
        lam = np.exp(mu * t - alpha * N)
        lam_upper = np.exp(mu * T - alpha * N)  # upper bound on [t, T]
        if lam_upper < 1e-10:
            break

        dt = rng.exponential(1.0 / lam_upper)
        t += dt
        if t >= T:
            break

        lam_t = np.exp(mu * t - alpha * N)
        u = rng.uniform()
        if u <= lam_t / lam_upper:
            times.append(t)
            N += 1

    return np.array(times)


# ---------------------------------------------------------------------------
# Marked point process
# ---------------------------------------------------------------------------

def simulate_marked_hawkes(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    mark_dist: str = "exponential",
    mark_param: float = 1.0,
    impact_func=None,
    seed: int = 42,
) -> dict:
    """
    Marked Hawkes process where marks (e.g., trade sizes) influence future intensity.

    lambda(t) = mu + sum_{t_i < t} g(m_i) * alpha * beta * exp(-beta*(t - t_i))

    Parameters
    ----------
    mu, alpha, beta : Hawkes parameters
    T : time horizon
    mark_dist : 'exponential', 'lognormal', or 'pareto'
    mark_param : parameter for mark distribution
    impact_func : callable(mark) -> weight, default = identity

    Returns
    -------
    dict with event_times, marks
    """
    rng = np.random.default_rng(seed)
    if impact_func is None:
        impact_func = lambda m: m

    times = []
    marks = []
    t = 0.0

    def _draw_mark():
        if mark_dist == "exponential":
            return rng.exponential(mark_param)
        elif mark_dist == "lognormal":
            return rng.lognormal(0, mark_param)
        elif mark_dist == "pareto":
            return (rng.pareto(mark_param) + 1)
        return 1.0

    while t < T:
        # Current intensity
        lam = mu
        for i, ti in enumerate(times):
            lam += impact_func(marks[i]) * alpha * beta * np.exp(-beta * (t - ti))

        lam_bar = max(lam, mu) * 2
        dt = rng.exponential(1.0 / max(lam_bar, 1e-10))
        t += dt
        if t >= T:
            break

        # Recompute intensity
        lam = mu
        for i, ti in enumerate(times):
            lam += impact_func(marks[i]) * alpha * beta * np.exp(-beta * (t - ti))

        u = rng.uniform()
        if u <= lam / max(lam_bar, 1e-10):
            times.append(t)
            marks.append(_draw_mark())

    return {"event_times": np.array(times), "marks": np.array(marks)}


# ---------------------------------------------------------------------------
# Inter-event time distributions
# ---------------------------------------------------------------------------

def fit_inter_event_times(
    event_times: np.ndarray,
) -> dict:
    """
    Fit multiple distributions to inter-event times and compare.

    Fits: exponential, Weibull, log-normal.
    Returns AIC for each and best fit.
    """
    iet = np.diff(event_times)
    if len(iet) < 5:
        return {"error": "Too few events"}

    n = len(iet)
    results = {}

    # Exponential
    rate = 1.0 / np.mean(iet)
    ll_exp = np.sum(np.log(rate) - rate * iet)
    results["exponential"] = {"rate": rate, "loglik": ll_exp, "aic": -2 * ll_exp + 2}

    # Weibull
    try:
        shape, _, scale = weibull_min.fit(iet, floc=0)
        ll_weib = np.sum(weibull_min.logpdf(iet, shape, loc=0, scale=scale))
        results["weibull"] = {"shape": shape, "scale": scale, "loglik": ll_weib, "aic": -2 * ll_weib + 4}
    except Exception:
        results["weibull"] = {"error": "fit failed"}

    # Log-normal
    try:
        s, _, scale = lognorm.fit(iet, floc=0)
        mu_ln = np.log(scale)
        ll_ln = np.sum(lognorm.logpdf(iet, s, loc=0, scale=scale))
        results["lognormal"] = {"mu": mu_ln, "sigma": s, "loglik": ll_ln, "aic": -2 * ll_ln + 4}
    except Exception:
        results["lognormal"] = {"error": "fit failed"}

    # Best by AIC
    valid = {k: v for k, v in results.items() if "aic" in v}
    if valid:
        best = min(valid, key=lambda k: valid[k]["aic"])
    else:
        best = None

    return {"distributions": results, "best_fit": best}


# ---------------------------------------------------------------------------
# Temporal clustering detection
# ---------------------------------------------------------------------------

def temporal_clustering_test(
    event_times: np.ndarray,
    T: float,
    n_bins: int = 50,
) -> dict:
    """
    Test for temporal clustering via dispersion index and runs test.

    Dispersion index: Var(counts) / Mean(counts). For Poisson, this = 1.
    Values > 1 indicate clustering, < 1 indicate regularity.
    """
    bin_edges = np.linspace(0, T, n_bins + 1)
    counts, _ = np.histogram(event_times, bins=bin_edges)

    mean_count = np.mean(counts)
    var_count = np.var(counts, ddof=1)
    dispersion_index = var_count / max(mean_count, 1e-10)

    # Chi-squared test for dispersion
    chi2_stat = (n_bins - 1) * dispersion_index
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(chi2_stat, n_bins - 1)

    # Runs test on inter-event times
    iet = np.diff(event_times)
    if len(iet) > 1:
        median_iet = np.median(iet)
        binary = (iet > median_iet).astype(int)
        runs = 1 + np.sum(np.abs(np.diff(binary)))
        n1 = np.sum(binary)
        n0 = len(binary) - n1
        if n0 > 0 and n1 > 0:
            E_runs = 1 + 2 * n0 * n1 / (n0 + n1)
            Var_runs = 2 * n0 * n1 * (2 * n0 * n1 - n0 - n1) / ((n0 + n1)**2 * (n0 + n1 - 1))
            z_runs = (runs - E_runs) / max(np.sqrt(Var_runs), 1e-10)
        else:
            z_runs = 0.0
    else:
        runs = 0
        z_runs = 0.0

    return {
        "dispersion_index": float(dispersion_index),
        "chi2_stat": float(chi2_stat),
        "chi2_pvalue": float(p_value),
        "is_clustered": dispersion_index > 1 and p_value < 0.05,
        "runs_test_z": float(z_runs),
        "n_events": len(event_times),
    }


# ---------------------------------------------------------------------------
# Trade arrival modeling
# ---------------------------------------------------------------------------

def trade_arrival_hawkes(
    trade_times: np.ndarray,
    T: float,
) -> dict:
    """
    Fit a Hawkes model to trade arrival times and report diagnostics.
    """
    fit = fit_hawkes_exp(trade_times, T)

    # Compute compensator (integrated intensity) at each event
    n = len(trade_times)
    mu, alpha, beta = fit["mu"], fit["alpha"], fit["beta"]

    compensator = np.zeros(n)
    for i in range(n):
        t_prev = 0.0 if i == 0 else trade_times[i - 1]
        t_curr = trade_times[i]

        # Integral of lambda from t_prev to t_curr
        base = mu * (t_curr - t_prev)
        hawkes_part = 0.0
        for j in range(i):
            tj = trade_times[j]
            hawkes_part += alpha * (np.exp(-beta * (t_prev - tj)) - np.exp(-beta * (t_curr - tj)))

        compensator[i] = base + hawkes_part

    # Time-rescaling: compensator values should be Exp(1) if model is correct
    ks_stat, ks_pval = kstest(compensator[compensator > 0], "expon", args=(0, 1))

    fit["compensator"] = compensator
    fit["ks_statistic"] = float(ks_stat)
    fit["ks_pvalue"] = float(ks_pval)
    fit["model_adequate"] = ks_pval > 0.05

    return fit


# ---------------------------------------------------------------------------
# News event clustering
# ---------------------------------------------------------------------------

def news_clustering_analysis(
    event_times: np.ndarray,
    T: float,
    min_gap: float = 0.0,
) -> dict:
    """
    Identify clusters of news events using inter-event time thresholding.

    Events separated by less than the median inter-event time are grouped
    into clusters.
    """
    if len(event_times) < 2:
        return {"n_clusters": 0, "clusters": []}

    iet = np.diff(event_times)
    threshold = np.median(iet) if min_gap <= 0 else min_gap

    clusters = []
    current_cluster = [event_times[0]]

    for i in range(1, len(event_times)):
        if iet[i - 1] <= threshold:
            current_cluster.append(event_times[i])
        else:
            clusters.append(np.array(current_cluster))
            current_cluster = [event_times[i]]
    clusters.append(np.array(current_cluster))

    cluster_sizes = [len(c) for c in clusters]
    cluster_durations = [c[-1] - c[0] if len(c) > 1 else 0.0 for c in clusters]

    return {
        "n_clusters": len(clusters),
        "cluster_sizes": cluster_sizes,
        "cluster_durations": cluster_durations,
        "avg_cluster_size": float(np.mean(cluster_sizes)),
        "max_cluster_size": int(np.max(cluster_sizes)),
        "threshold": float(threshold),
        "clusters": clusters,
    }


# ---------------------------------------------------------------------------
# Residual analysis: time rescaling theorem
# ---------------------------------------------------------------------------

def time_rescaling_residuals(
    event_times: np.ndarray,
    intensity_func,
    dt: float = 0.001,
) -> dict:
    """
    Time rescaling theorem: transform event times by the compensator
    (integrated conditional intensity). Under correct model, transformed
    inter-event times are Exp(1).

    Parameters
    ----------
    event_times : 1-D array
    intensity_func : callable(t, history) -> intensity
    dt : integration step

    Returns
    -------
    dict with transformed_times, ks_stat, ks_pvalue, qq_data
    """
    n = len(event_times)
    if n < 3:
        return {"error": "Too few events"}

    transformed = np.zeros(n)
    for i in range(n):
        t_start = 0.0 if i == 0 else event_times[i - 1]
        t_end = event_times[i]
        history = event_times[:i]

        # Integrate intensity from t_start to t_end
        n_int = max(int((t_end - t_start) / dt), 1)
        t_grid = np.linspace(t_start, t_end, n_int + 1)
        intensities = np.array([intensity_func(t, history) for t in t_grid])
        integral = np.trapz(intensities, t_grid)
        transformed[i] = integral

    # KS test against Exp(1)
    ks_stat, ks_pval = kstest(transformed, "expon", args=(0, 1))

    # QQ data
    sorted_transformed = np.sort(transformed)
    theoretical = -np.log(1 - np.arange(1, n + 1) / (n + 1))

    return {
        "transformed_times": transformed,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "model_adequate": ks_pval > 0.05,
        "qq_empirical": sorted_transformed,
        "qq_theoretical": theoretical,
    }


# ---------------------------------------------------------------------------
# Intensity estimation via kernel smoothing
# ---------------------------------------------------------------------------

def kernel_intensity_estimate(
    event_times: np.ndarray,
    T: float,
    bandwidth: float = None,
    n_grid: int = 500,
    kernel: str = "gaussian",
) -> dict:
    """
    Non-parametric intensity estimation via kernel smoothing.

    Parameters
    ----------
    event_times : 1-D array
    T : observation window
    bandwidth : kernel bandwidth (default: Silverman rule)
    n_grid : evaluation grid size
    kernel : 'gaussian' or 'epanechnikov'

    Returns
    -------
    dict with grid, intensity_estimate
    """
    n = len(event_times)
    if bandwidth is None:
        if n > 1:
            iet = np.diff(event_times)
            bandwidth = 1.06 * np.std(iet) * n**(-0.2)
        else:
            bandwidth = T / 10

    bandwidth = max(bandwidth, 1e-6)
    grid = np.linspace(0, T, n_grid)
    intensity = np.zeros(n_grid)

    for i, t in enumerate(grid):
        diff = (t - event_times) / bandwidth
        if kernel == "gaussian":
            K = np.exp(-0.5 * diff**2) / np.sqrt(2 * np.pi)
        elif kernel == "epanechnikov":
            K = np.where(np.abs(diff) <= 1, 0.75 * (1 - diff**2), 0.0)
        else:
            K = np.exp(-0.5 * diff**2) / np.sqrt(2 * np.pi)

        intensity[i] = np.sum(K) / bandwidth

    return {
        "grid": grid,
        "intensity": intensity,
        "bandwidth": bandwidth,
        "total_events": n,
    }


# ---------------------------------------------------------------------------
# Hawkes process EM fitting
# ---------------------------------------------------------------------------

def hawkes_em_fit(
    event_times: np.ndarray,
    T: float,
    n_iter: int = 50,
    seed: int = 42,
) -> dict:
    """
    EM algorithm for Hawkes process with exponential kernel.

    E-step: compute P(event i triggered by event j) = p_ij
    M-step: update mu, alpha, beta from sufficient statistics.
    """
    rng = np.random.default_rng(seed)
    n = len(event_times)
    if n < 3:
        return {"error": "Too few events"}

    # Initialize
    mu = n / (2 * T)
    alpha = 0.3
    beta = 1.0

    log_liks = []

    for iteration in range(n_iter):
        # E-step: compute branching probabilities
        # p_ij = alpha*beta*exp(-beta*(t_i - t_j)) / lambda(t_i)  for j < i
        # p_i0 = mu / lambda(t_i)  (immigrant)

        p_immigrant = np.zeros(n)

        for i in range(n):
            lam_i = mu
            triggered = []
            for j in range(i):
                g = alpha * beta * np.exp(-beta * (event_times[i] - event_times[j]))
                triggered.append(g)
                lam_i += g

            p_immigrant[i] = mu / max(lam_i, 1e-15)

        total_immigrants = np.sum(p_immigrant)
        total_offspring = n - total_immigrants

        # M-step
        mu_new = total_immigrants / T
        alpha_new = total_offspring / max(n, 1)

        # Update beta via weighted MLE
        weighted_sum = 0.0
        weight_total = 0.0
        for i in range(1, n):
            lam_i = mu
            for j in range(i):
                g = alpha * beta * np.exp(-beta * (event_times[i] - event_times[j]))
                p_ij = g / max(lam_i + sum(
                    alpha * beta * np.exp(-beta * (event_times[i] - event_times[k]))
                    for k in range(i)
                ), 1e-15)
                w = p_ij  # approximate
                weighted_sum += w * (event_times[i] - event_times[j])
                weight_total += w
                lam_i += g

        if weight_total > 0:
            beta_new = weight_total / max(weighted_sum, 1e-15)
        else:
            beta_new = beta

        mu = max(mu_new, 1e-6)
        alpha = min(max(alpha_new, 1e-6), 0.999)
        beta = max(beta_new, 1e-3)

        # Log-likelihood
        ll = -hawkes_loglik_exp(np.array([mu, alpha, beta]), event_times, T)
        log_liks.append(ll)

    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": alpha,
        "stable": alpha < 1.0,
        "log_likelihoods": log_liks,
        "converged": len(log_liks) > 1 and abs(log_liks[-1] - log_liks[-2]) < 1e-4,
    }


# ---------------------------------------------------------------------------
# Utility: simulate and analyze trade arrivals
# ---------------------------------------------------------------------------

def analyze_point_process(
    event_times: np.ndarray,
    T: float,
) -> dict:
    """
    Comprehensive point process analysis: fit Poisson and Hawkes, compare,
    and provide diagnostics.
    """
    n = len(event_times)
    if n < 5:
        return {"error": "Too few events for analysis"}

    # Basic statistics
    iet = np.diff(event_times)
    basic = {
        "n_events": n,
        "mean_rate": n / T,
        "mean_iet": float(np.mean(iet)),
        "std_iet": float(np.std(iet)),
        "cv_iet": float(np.std(iet) / np.mean(iet)) if np.mean(iet) > 0 else 0,
    }

    # Poisson fit
    poisson = poisson_mle(event_times, T)

    # Hawkes fit
    hawkes = fit_hawkes_exp(event_times, T)

    # Clustering test
    clustering = temporal_clustering_test(event_times, T)

    # IET distributions
    iet_fits = fit_inter_event_times(event_times)

    # Intensity estimate
    intensity = kernel_intensity_estimate(event_times, T)

    return {
        "basic_stats": basic,
        "poisson_fit": poisson,
        "hawkes_fit": hawkes,
        "clustering": clustering,
        "iet_fits": iet_fits,
        "intensity_estimate": {
            "grid": intensity["grid"],
            "intensity": intensity["intensity"],
        },
    }
