"""
Optimal stopping theory for quantitative finance.

Implements classical and modern optimal stopping algorithms including
American option pricing, optimal entry/exit timing, secretary problems,
and change-point detection.
"""

import numpy as np
from scipy import stats
from scipy.special import eval_laguerre
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Longstaff-Schwartz (LSM) American option pricing
# ---------------------------------------------------------------------------

def _laguerre_basis(x: np.ndarray, n_basis: int) -> np.ndarray:
    """Weighted Laguerre polynomial basis L_k(x)*exp(-x/2) for k=0..n_basis-1."""
    x_pos = np.maximum(x, 0.0)
    basis = np.zeros((len(x_pos), n_basis))
    for k in range(n_basis):
        basis[:, k] = eval_laguerre(k, x_pos) * np.exp(-x_pos / 2.0)
    return basis


def lsm_american_put(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 50,
    n_paths: int = 10000,
    n_basis: int = 4,
    seed: int = 42,
) -> dict:
    """
    Longstaff-Schwartz Monte Carlo for American put option.

    Parameters
    ----------
    S0 : float – initial stock price
    K : float – strike price
    r : float – risk-free rate
    sigma : float – volatility
    T : float – time to maturity
    n_steps : int – number of time steps
    n_paths : int – number of simulation paths
    n_basis : int – number of Laguerre basis functions
    seed : int – random seed

    Returns
    -------
    dict with keys: price, std_err, exercise_boundary (array of shape (n_steps,))
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    # Simulate GBM paths
    Z = rng.standard_normal((n_paths, n_steps))
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

    # Intrinsic value
    payoff = np.maximum(K - S, 0.0)

    # Backward induction
    V = payoff[:, -1].copy()
    exercise_boundary = np.full(n_steps + 1, np.nan)

    for t in range(n_steps - 1, 0, -1):
        itm = payoff[:, t] > 0
        if np.sum(itm) < n_basis + 1:
            V *= discount
            continue

        X = S[itm, t] / K  # normalise for numerical stability
        basis = _laguerre_basis(X, n_basis)
        Y = V[itm] * discount

        # Least squares regression
        coeffs, _, _, _ = np.linalg.lstsq(basis, Y, rcond=None)
        continuation = basis @ coeffs

        exercise = payoff[itm, t] >= continuation
        V[itm] = np.where(exercise, payoff[itm, t], V[itm] * discount)
        V[~itm] *= discount

        # Exercise boundary: approximate as max S where exercise is optimal
        if np.any(exercise):
            exercise_boundary[t] = np.max(S[itm, t][exercise])

    price = np.mean(V * discount)
    std_err = np.std(V * discount) / np.sqrt(n_paths)

    return {
        "price": price,
        "std_err": std_err,
        "exercise_boundary": exercise_boundary,
    }


def lsm_american_call(
    S0: float, K: float, r: float, q: float, sigma: float,
    T: float, n_steps: int = 50, n_paths: int = 10000,
    n_basis: int = 4, seed: int = 42,
) -> dict:
    """LSM for American call with continuous dividend yield q."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    Z = rng.standard_normal((n_paths, n_steps))
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

    payoff = np.maximum(S - K, 0.0)
    V = payoff[:, -1].copy()
    exercise_boundary = np.full(n_steps + 1, np.nan)

    for t in range(n_steps - 1, 0, -1):
        itm = payoff[:, t] > 0
        if np.sum(itm) < n_basis + 1:
            V *= discount
            continue

        X = S[itm, t] / K
        basis = _laguerre_basis(X, n_basis)
        Y = V[itm] * discount
        coeffs, _, _, _ = np.linalg.lstsq(basis, Y, rcond=None)
        continuation = basis @ coeffs

        exercise = payoff[itm, t] >= continuation
        V[itm] = np.where(exercise, payoff[itm, t], V[itm] * discount)
        V[~itm] *= discount

        if np.any(exercise):
            exercise_boundary[t] = np.min(S[itm, t][exercise])

    price = np.mean(V * discount)
    std_err = np.std(V * discount) / np.sqrt(n_paths)
    return {"price": price, "std_err": std_err, "exercise_boundary": exercise_boundary}


# ---------------------------------------------------------------------------
# Optimal entry timing via OU threshold crossing
# ---------------------------------------------------------------------------

def ou_optimal_entry(
    mu: float,
    theta: float,
    sigma: float,
    current_x: float,
    r: float,
    reward_per_unit: float = 1.0,
    n_grid: int = 500,
    x_range: tuple = (-3.0, 3.0),
) -> dict:
    """
    Optimal entry timing: wait for a pullback in a mean-reverting (OU) process.

    The investor earns reward_per_unit * (mu - x) upon entry at level x,
    discounted at rate r. Solve for the optimal entry threshold x* that
    maximises E[e^{-r*tau} * (mu - X_tau)].

    Uses the free-boundary ODE approach:
        0.5*sigma^2 V'' + theta*(mu - x)*V' - r*V = 0
    with V(x*) = mu - x*, V'(x*) = -1 (smooth pasting).

    Returns dict with optimal_threshold, value_function (on grid), grid.
    """
    grid = np.linspace(x_range[0], x_range[1], n_grid)
    dx = grid[1] - grid[0]

    def _solve_value(x_star: float):
        """Solve BVP for a candidate threshold, return value at x=mu (should be positive)."""
        # Finite difference: 0.5*sig^2*V'' + theta*(mu-x)*V' - r*V = 0
        # on [x_star, x_range[1]] with V(x_star) = mu - x_star, V'(x_star) = -1
        sub_grid = grid[grid >= x_star]
        if len(sub_grid) < 5:
            return -1e10
        n = len(sub_grid)
        dx_l = sub_grid[1] - sub_grid[0]

        A = np.zeros((n, n))
        b_vec = np.zeros(n)

        # Boundary at x_star
        A[0, 0] = 1.0
        b_vec[0] = reward_per_unit * (mu - x_star)

        # Interior points
        for i in range(1, n - 1):
            xi = sub_grid[i]
            drift = theta * (mu - xi)
            diff = 0.5 * sigma**2
            A[i, i - 1] = diff / dx_l**2 - drift / (2 * dx_l)
            A[i, i] = -2 * diff / dx_l**2 - r
            A[i, i + 1] = diff / dx_l**2 + drift / (2 * dx_l)

        # Far boundary: V -> 0 as x -> inf
        A[-1, -1] = 1.0
        b_vec[-1] = 0.0

        try:
            V = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            return -1e10

        # Find V at x closest to current_x
        idx = np.argmin(np.abs(sub_grid - current_x))
        return V[idx] if idx < len(V) else 0.0

    # Search for x* that maximises value
    candidates = np.linspace(x_range[0], mu, 200)
    vals = np.array([_solve_value(xs) for xs in candidates])
    best_idx = np.argmax(vals)
    optimal_threshold = candidates[best_idx]

    # Re-solve for full value function at optimal threshold
    sub_grid = grid[grid >= optimal_threshold]
    n = len(sub_grid)
    dx_l = sub_grid[1] - sub_grid[0] if n > 1 else dx
    A = np.zeros((n, n))
    b_vec = np.zeros(n)
    A[0, 0] = 1.0
    b_vec[0] = reward_per_unit * (mu - optimal_threshold)
    for i in range(1, n - 1):
        xi = sub_grid[i]
        drift = theta * (mu - xi)
        diff = 0.5 * sigma**2
        A[i, i - 1] = diff / dx_l**2 - drift / (2 * dx_l)
        A[i, i] = -2 * diff / dx_l**2 - r
        A[i, i + 1] = diff / dx_l**2 + drift / (2 * dx_l)
    A[-1, -1] = 1.0
    b_vec[-1] = 0.0
    V_full = np.linalg.solve(A, b_vec)

    return {
        "optimal_threshold": optimal_threshold,
        "value_at_current": vals[best_idx],
        "grid": sub_grid,
        "value_function": V_full,
    }


# ---------------------------------------------------------------------------
# Trailing stop calibration via drawdown distribution
# ---------------------------------------------------------------------------

def trailing_stop_calibration(
    mu: float,
    sigma: float,
    dt: float,
    n_paths: int = 50000,
    n_steps: int = 1000,
    stop_levels: np.ndarray = None,
    seed: int = 42,
) -> dict:
    """
    Calibrate trailing stop-loss by simulating maximum drawdown distribution
    under GBM with drift mu and volatility sigma.

    Returns dict mapping each stop level to (prob_triggered, avg_pnl_if_triggered,
    avg_time_to_trigger).
    """
    if stop_levels is None:
        stop_levels = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20])

    rng = np.random.default_rng(seed)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    cum_returns = np.cumsum(log_returns, axis=1)
    running_max = np.maximum.accumulate(cum_returns, axis=1)
    drawdown = running_max - cum_returns  # always >= 0

    results = {}
    for sl in stop_levels:
        # First time drawdown exceeds sl (in log terms, approx pct)
        triggered = drawdown >= sl
        first_trigger = np.argmax(triggered, axis=1)  # 0 if never triggered
        ever_triggered = np.any(triggered, axis=1)

        prob = np.mean(ever_triggered)
        if np.any(ever_triggered):
            avg_time = np.mean(first_trigger[ever_triggered]) * dt
            avg_pnl = np.mean(cum_returns[ever_triggered, first_trigger[ever_triggered].clip(0, n_steps - 1)])
        else:
            avg_time = np.nan
            avg_pnl = np.nan

        results[float(sl)] = {
            "prob_triggered": prob,
            "avg_pnl_if_triggered": avg_pnl,
            "avg_time_to_trigger": avg_time,
        }

    # Also compute unconditional max drawdown distribution
    max_dd = np.max(drawdown, axis=1)
    dd_quantiles = np.quantile(max_dd, [0.5, 0.75, 0.9, 0.95, 0.99])

    return {
        "stop_levels": results,
        "max_drawdown_quantiles": {
            "50%": dd_quantiles[0],
            "75%": dd_quantiles[1],
            "90%": dd_quantiles[2],
            "95%": dd_quantiles[3],
            "99%": dd_quantiles[4],
        },
    }


# ---------------------------------------------------------------------------
# Secretary problem for best execution price
# ---------------------------------------------------------------------------

def secretary_problem_optimal_k(n: int) -> int:
    """Optimal number of candidates to skip in the classical secretary problem."""
    return max(1, int(np.round(n / np.e)))


def secretary_best_execution(
    prices: np.ndarray,
    is_buy: bool = True,
) -> dict:
    """
    Apply the secretary (best-choice) algorithm to find the best execution price.

    For a buy order, we want the minimum price; for sell, the maximum.
    We observe the first n/e prices, then pick the first price that beats all
    of those.

    Parameters
    ----------
    prices : 1-D array of observed prices in chronological order
    is_buy : True for buy (minimise), False for sell (maximise)

    Returns
    -------
    dict with selected_price, selected_index, optimal_price, was_optimal
    """
    n = len(prices)
    k = secretary_problem_optimal_k(n)

    if is_buy:
        reference = np.min(prices[:k])
        optimal = np.min(prices)
        selected_idx = None
        for i in range(k, n):
            if prices[i] <= reference:
                selected_idx = i
                break
        if selected_idx is None:
            selected_idx = n - 1  # forced to take last
    else:
        reference = np.max(prices[:k])
        optimal = np.max(prices)
        selected_idx = None
        for i in range(k, n):
            if prices[i] >= reference:
                selected_idx = i
                break
        if selected_idx is None:
            selected_idx = n - 1

    return {
        "selected_price": prices[selected_idx],
        "selected_index": int(selected_idx),
        "optimal_price": optimal,
        "was_optimal": bool(prices[selected_idx] == optimal),
        "skip_count": k,
    }


def secretary_problem_simulation(
    n: int = 100, n_trials: int = 100000, seed: int = 42,
) -> dict:
    """Monte Carlo estimate of success probability for secretary problem."""
    rng = np.random.default_rng(seed)
    k = secretary_problem_optimal_k(n)
    successes = 0
    for _ in range(n_trials):
        perm = rng.permutation(n)
        ref = np.max(perm[:k])
        chosen = None
        for i in range(k, n):
            if perm[i] > ref:
                chosen = perm[i]
                break
        if chosen is None:
            chosen = perm[-1]
        if chosen == n - 1:
            successes += 1
    return {"n": n, "k": k, "success_rate": successes / n_trials, "theoretical": 1.0 / np.e}


# ---------------------------------------------------------------------------
# Multiple stopping: k-best entry/exit points
# ---------------------------------------------------------------------------

def multiple_stopping_dp(
    values: np.ndarray,
    k: int,
    discount: float = 1.0,
) -> dict:
    """
    Find k optimal stopping points on a deterministic sequence to maximise
    total discounted reward sum_{j=1}^k discount^{t_j} * values[t_j].

    Uses DP: V_j(t) = max(discount^t * values[t] + V_{j-1}(t+1), V_j(t+1))
    for j = k, k-1, ..., 1.

    Parameters
    ----------
    values : 1-D array of rewards at each time
    k : number of stops allowed
    discount : per-period discount factor

    Returns
    -------
    dict with total_reward, stopping_times (list of indices)
    """
    n = len(values)
    if k >= n:
        idx = list(range(n))
        return {"total_reward": np.sum(values * discount ** np.arange(n)), "stopping_times": idx}

    disc = discount ** np.arange(n)
    discounted = values * disc

    # V[j][t] = max expected reward from t onward with j stops remaining
    V = np.zeros((k + 1, n + 1))
    policy = np.zeros((k + 1, n + 1), dtype=int)  # 1 = stop, 0 = continue

    for j in range(1, k + 1):
        for t in range(n - 1, -1, -1):
            stop_val = discounted[t] + V[j - 1, t + 1]
            cont_val = V[j, t + 1]
            if stop_val >= cont_val:
                V[j, t] = stop_val
                policy[j, t] = 1
            else:
                V[j, t] = cont_val
                policy[j, t] = 0

    # Trace optimal stops
    stops = []
    j_rem = k
    t = 0
    while j_rem > 0 and t < n:
        if policy[j_rem, t] == 1:
            stops.append(int(t))
            j_rem -= 1
        t += 1

    return {"total_reward": V[k, 0], "stopping_times": stops}


# ---------------------------------------------------------------------------
# Free-boundary problem for perpetual American put
# ---------------------------------------------------------------------------

def perpetual_american_put(
    K: float, r: float, sigma: float,
) -> dict:
    """
    Closed-form perpetual American put price and optimal exercise boundary.

    V(S) = (K - S*) * (S / S*)^gamma  for S >= S*
    V(S) = K - S                       for S < S*
    where gamma = -2r/sigma^2 and S* = K * gamma / (gamma - 1).
    """
    gamma = 2 * r / sigma**2
    # Exponent for the ODE solution
    lam = -(gamma - 1) / 2 - np.sqrt(((gamma - 1) / 2)**2 + gamma)
    # Actually use standard formula:
    # characteristic eqn: 0.5*sig^2*lam*(lam-1) + r*lam - r = 0
    # roots: lam = ( -(r/sig^2 - 0.5) +/- sqrt((r/sig^2 - 0.5)^2 + 2r/sig^2) )
    a = 0.5
    b_coeff = r / sigma**2 - 0.5
    c_coeff = -r / sigma**2
    disc = b_coeff**2 - 4 * a * c_coeff
    lam_neg = (-b_coeff - np.sqrt(disc)) / (2 * a)

    S_star = K * lam_neg / (lam_neg - 1)

    def price(S):
        S = np.atleast_1d(np.asarray(S, dtype=float))
        V = np.where(S <= S_star, K - S, (K - S_star) * (S / S_star) ** lam_neg)
        return V

    return {
        "exercise_boundary": S_star,
        "exponent": lam_neg,
        "price_func": price,
        "sample_prices": {
            "S": np.linspace(0.5 * S_star, 2 * K, 50).tolist(),
            "V": price(np.linspace(0.5 * S_star, 2 * K, 50)).tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Bayesian optimal stopping with unknown drift
# ---------------------------------------------------------------------------

def bayesian_optimal_stopping(
    observations: np.ndarray,
    prior_mu: float = 0.0,
    prior_var: float = 1.0,
    sigma: float = 1.0,
    discount: float = 0.99,
    reward_func=None,
) -> dict:
    """
    Bayesian optimal stopping: sequentially update belief about drift mu
    while deciding when to stop.

    At each step, observe X_t = mu + sigma*eps, update posterior on mu,
    then decide stop/continue.

    Uses backward induction on discretized (posterior_mean, posterior_var, t) state.
    Here we use a simpler forward heuristic: stop when
        E[reward | stop now] >= E[reward | continue one more step].

    Parameters
    ----------
    observations : 1-D array of sequential observations
    prior_mu, prior_var : prior on drift
    sigma : noise std
    discount : per-step discount
    reward_func : callable(posterior_mu, posterior_var) -> float, default = posterior_mu

    Returns
    -------
    dict with stopping_time, posterior_mu, posterior_var, expected_reward
    """
    if reward_func is None:
        reward_func = lambda m, v: m

    n = len(observations)
    mu_post = prior_mu
    var_post = prior_var

    stopping_time = n - 1
    posteriors = []

    for t in range(n):
        # Bayesian update (conjugate normal-normal)
        precision_prior = 1.0 / var_post
        precision_obs = 1.0 / sigma**2
        precision_post = precision_prior + precision_obs
        mu_post = (precision_prior * mu_post + precision_obs * observations[t]) / precision_post
        var_post = 1.0 / precision_post

        posteriors.append((mu_post, var_post))

        # Current reward if stop
        reward_stop = reward_func(mu_post, var_post)

        # Expected reward if continue one more step (integrate over next obs)
        # Next obs ~ N(mu_post, sigma^2 + var_post)
        # After update, posterior mean ~ N(mu_post, var_post * sigma^2 / (var_post + sigma^2))
        if t < n - 1:
            var_next = 1.0 / (precision_post + precision_obs)
            # Expected reward of continuing = discount * E[reward(mu_next, var_next)]
            # Under current posterior, mu_next has same mean mu_post
            reward_continue = discount * reward_func(mu_post, var_next)
        else:
            reward_continue = -np.inf

        if reward_stop >= reward_continue:
            stopping_time = t
            break

    return {
        "stopping_time": stopping_time,
        "posterior_mu": mu_post,
        "posterior_var": var_post,
        "expected_reward": reward_func(mu_post, var_post),
        "posteriors": posteriors,
    }


# ---------------------------------------------------------------------------
# Finite-horizon optimal stopping DP
# ---------------------------------------------------------------------------

def finite_horizon_optimal_stopping_dp(
    transition_matrix: np.ndarray,
    reward: np.ndarray,
    discount: float = 0.99,
    terminal_reward: np.ndarray = None,
) -> dict:
    """
    Finite-horizon optimal stopping via dynamic programming on a discrete state space.

    Parameters
    ----------
    transition_matrix : (n_states, n_states) stochastic matrix P[i,j] = P(X_{t+1}=j | X_t=i)
    reward : (n_states, n_steps) reward[i, t] = payoff from stopping at state i, time t
    discount : discount factor
    terminal_reward : (n_states,) payoff at terminal time, default = reward[:, -1]

    Returns
    -------
    dict with value_function (n_states, n_steps), optimal_policy (n_states, n_steps)
    """
    n_states, n_steps = reward.shape
    P = transition_matrix

    V = np.zeros((n_states, n_steps))
    policy = np.zeros((n_states, n_steps), dtype=int)  # 1 = stop

    if terminal_reward is None:
        terminal_reward = reward[:, -1]

    V[:, -1] = terminal_reward
    policy[:, -1] = 1  # must stop at terminal time

    for t in range(n_steps - 2, -1, -1):
        continuation = discount * P @ V[:, t + 1]
        stop_val = reward[:, t]
        V[:, t] = np.maximum(stop_val, continuation)
        policy[:, t] = (stop_val >= continuation).astype(int)

    return {"value_function": V, "optimal_policy": policy}


# ---------------------------------------------------------------------------
# Shiryaev-Roberts change-point detection
# ---------------------------------------------------------------------------

def shiryaev_roberts(
    data: np.ndarray,
    mu0: float = 0.0,
    mu1: float = 1.0,
    sigma: float = 1.0,
    threshold: float = 100.0,
) -> dict:
    """
    Shiryaev-Roberts procedure for sequential change-point detection.

    Assumes observations are N(mu0, sigma^2) before change and N(mu1, sigma^2) after.

    The SR statistic is:
        R_t = (1 + R_{t-1}) * L_t
    where L_t = f_1(X_t) / f_0(X_t) is the likelihood ratio.

    Stop at first time R_t >= threshold.

    Parameters
    ----------
    data : 1-D array of observations
    mu0, mu1, sigma : pre- and post-change parameters
    threshold : detection threshold

    Returns
    -------
    dict with alarm_time, sr_statistic (array), log_likelihood_ratios
    """
    n = len(data)
    R = np.zeros(n)
    log_lr = np.zeros(n)

    alarm_time = n  # no alarm if never triggered

    for t in range(n):
        # Log-likelihood ratio
        log_lr[t] = ((mu1 - mu0) * data[t] - 0.5 * (mu1**2 - mu0**2)) / sigma**2
        lr = np.exp(log_lr[t])

        if t == 0:
            R[t] = lr
        else:
            R[t] = (1.0 + R[t - 1]) * lr

        if R[t] >= threshold and alarm_time == n:
            alarm_time = t

    return {
        "alarm_time": alarm_time if alarm_time < n else None,
        "sr_statistic": R,
        "log_likelihood_ratios": log_lr,
    }


def cusum_detection(
    data: np.ndarray,
    mu0: float = 0.0,
    mu1: float = 1.0,
    sigma: float = 1.0,
    threshold: float = 5.0,
) -> dict:
    """
    CUSUM procedure for change-point detection (one-sided, upward shift).

    S_t = max(0, S_{t-1} + (X_t - (mu0+mu1)/2) * (mu1-mu0) / sigma^2)
    """
    n = len(data)
    S = np.zeros(n)
    ref = (mu0 + mu1) / 2.0
    scale = (mu1 - mu0) / sigma**2
    alarm_time = None

    for t in range(n):
        incr = (data[t] - ref) * scale
        S[t] = max(0.0, (S[t - 1] if t > 0 else 0.0) + incr)
        if S[t] >= threshold and alarm_time is None:
            alarm_time = t

    return {"alarm_time": alarm_time, "cusum_statistic": S}


# ---------------------------------------------------------------------------
# Snell envelope on binomial tree
# ---------------------------------------------------------------------------

def snell_envelope_binomial(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 100,
    option_type: str = "put",
) -> dict:
    """
    Compute the Snell envelope (smallest supermartingale majorant of the
    payoff process) on a CRR binomial tree. This gives the American option
    price at every node.

    Parameters
    ----------
    S0, K, r, sigma, T : option parameters
    n_steps : tree depth
    option_type : 'put' or 'call'

    Returns
    -------
    dict with price, tree (2-D array of Snell envelope values),
    exercise_region (boolean 2-D array)
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    if option_type == "put":
        payoff_fn = lambda s: max(K - s, 0.0)
    else:
        payoff_fn = lambda s: max(s - K, 0.0)

    # Build stock price tree
    S = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            S[j, i] = S0 * u**j * d**(i - j)

    # Payoff at each node
    H = np.zeros_like(S)
    for i in range(n_steps + 1):
        for j in range(i + 1):
            H[j, i] = payoff_fn(S[j, i])

    # Backward induction for Snell envelope
    V = np.zeros_like(S)
    exercise = np.zeros_like(S, dtype=bool)

    # Terminal
    for j in range(n_steps + 1):
        V[j, n_steps] = H[j, n_steps]
        exercise[j, n_steps] = H[j, n_steps] > 0

    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * V[j + 1, i + 1] + (1 - p) * V[j, i + 1])
            V[j, i] = max(H[j, i], cont)
            exercise[j, i] = H[j, i] >= cont and H[j, i] > 0

    return {
        "price": V[0, 0],
        "snell_envelope": V,
        "exercise_region": exercise,
        "stock_tree": S,
        "up_factor": u,
        "down_factor": d,
        "risk_neutral_prob": p,
    }


# ---------------------------------------------------------------------------
# Utility: simulate optimal stopping scenarios
# ---------------------------------------------------------------------------

def simulate_optimal_stopping_paths(
    n_paths: int = 1000,
    n_steps: int = 200,
    mu: float = 0.05,
    sigma: float = 0.2,
    dt: float = 1 / 252,
    stop_rule: str = "trailing",
    stop_param: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Simulate price paths and apply a simple stopping rule.

    stop_rule options:
    - 'trailing' : trailing stop at stop_param fraction drawdown
    - 'profit_target' : stop at stop_param fraction gain
    - 'combined' : stop at either trailing stop or profit target (stop_param is trailing)

    Returns dict with avg_return, avg_holding_period, paths_summary.
    """
    rng = np.random.default_rng(seed)
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    cum_ret = np.cumsum(log_ret, axis=1)
    prices = np.exp(cum_ret)  # normalised to start at 1

    returns = []
    holding_periods = []

    for i in range(n_paths):
        path = prices[i]
        running_max = 0.0
        stop_time = n_steps - 1

        for t in range(n_steps):
            running_max = max(running_max, path[t])
            drawdown = 1 - path[t] / running_max if running_max > 0 else 0

            if stop_rule == "trailing" and drawdown >= stop_param:
                stop_time = t
                break
            elif stop_rule == "profit_target" and path[t] >= 1 + stop_param:
                stop_time = t
                break
            elif stop_rule == "combined":
                if drawdown >= stop_param or path[t] >= 1 + 2 * stop_param:
                    stop_time = t
                    break

        returns.append(float(path[stop_time] - 1))
        holding_periods.append(stop_time)

    returns = np.array(returns)
    holding_periods = np.array(holding_periods)

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_holding_period": float(np.mean(holding_periods)),
        "median_holding_period": float(np.median(holding_periods)),
        "win_rate": float(np.mean(returns > 0)),
        "avg_win": float(np.mean(returns[returns > 0])) if np.any(returns > 0) else 0.0,
        "avg_loss": float(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0.0,
    }
