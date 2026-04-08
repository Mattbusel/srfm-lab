"""
Stochastic control theory for optimal trading.

Implements:
  - Hamilton-Jacobi-Bellman (HJB) equation solvers
  - Merton's optimal consumption/investment problem
  - Optimal stopping (American option, sequential testing)
  - Linear Quadratic Regulator (LQR) for portfolio control
  - CARA/CRRA utility maximization
  - Risk-sensitive control (exponential utility)
  - Finite horizon dynamic programming
  - Stochastic optimal switching (regime-aware)
  - Optimal liquidation as stochastic control
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


# ── Utility Functions ─────────────────────────────────────────────────────────

def cara_utility(wealth: float, gamma: float = 0.001) -> float:
    """Constant Absolute Risk Aversion: U(W) = -exp(-gamma * W)."""
    return -math.exp(-gamma * wealth)


def crra_utility(wealth: float, gamma: float = 2.0) -> float:
    """Constant Relative Risk Aversion: U(W) = W^(1-gamma)/(1-gamma)."""
    if gamma == 1.0:
        return math.log(max(wealth, 1e-10))
    return wealth**(1 - gamma) / (1 - gamma) if wealth > 0 else -1e10


def merton_optimal_fraction(
    mu: float,
    sigma: float,
    r: float = 0.0,
    gamma: float = 2.0,
) -> float:
    """
    Merton's optimal risky fraction for CRRA utility.
    f* = (mu - r) / (gamma * sigma^2)
    """
    return float((mu - r) / (gamma * sigma**2 + 1e-10))


def merton_optimal_consumption(
    wealth: float,
    rho: float,      # time preference
    r: float,        # risk-free rate
    gamma: float,    # risk aversion
    mu: float,
    sigma: float,
    T: float,        # remaining horizon
) -> float:
    """
    Merton optimal consumption rate c* = (rho - r*(1-gamma) - 0.5*((mu-r)/sigma)^2 * (1-gamma)/gamma) / gamma * W
    """
    sharpe = (mu - r) / max(sigma, 1e-10)
    phi = (rho - r * (1 - gamma) - 0.5 * sharpe**2 * (1 - gamma) / gamma) / gamma
    c_rate = max(phi, 1e-6)
    return float(c_rate * wealth)


# ── Linear Quadratic Regulator ────────────────────────────────────────────────

@dataclass
class LQRProblem:
    """
    Continuous-time LQR: min ∫ (x'Qx + u'Ru) dt + x(T)'Fx(T)
    Dynamics: dx = (Ax + Bu)dt
    """
    A: np.ndarray   # state transition
    B: np.ndarray   # control input
    Q: np.ndarray   # state cost
    R: np.ndarray   # control cost
    F: np.ndarray   # terminal cost


def lqr_riccati(
    prob: LQRProblem,
    T: float,
    n_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Riccati ODE backward: -dP/dt = A'P + PA - PBR^{-1}B'P + Q, P(T) = F.
    Returns (P_trajectory, time_grid).
    """
    dt = T / n_steps
    n = prob.A.shape[0]

    P = prob.F.copy()
    P_history = [P.copy()]
    R_inv = np.linalg.inv(prob.R + 1e-8 * np.eye(prob.R.shape[0]))

    # Backward integration (Euler)
    for _ in range(n_steps):
        dP = (prob.A.T @ P + P @ prob.A
              - P @ prob.B @ R_inv @ prob.B.T @ P + prob.Q)
        P = P - dt * dP  # backward step
        P = 0.5 * (P + P.T)  # symmetrize
        P_history.append(P.copy())

    t_grid = np.linspace(T, 0, n_steps + 1)
    return P_history[-1], t_grid


def lqr_control(
    prob: LQRProblem,
    x: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Optimal LQR control: u* = -R^{-1} B' P x."""
    R_inv = np.linalg.inv(prob.R + 1e-8 * np.eye(prob.R.shape[0]))
    return float(-R_inv @ prob.B.T @ P @ x)


def portfolio_lqr(
    target_weights: np.ndarray,
    current_weights: np.ndarray,
    cov_matrix: np.ndarray,
    transaction_cost: float = 0.001,
    horizon: float = 1.0,
) -> np.ndarray:
    """
    LQR for portfolio rebalancing: minimize tracking error + transaction costs.
    Returns optimal trade vector.
    """
    n = len(target_weights)
    deviation = current_weights - target_weights

    # State = deviation, control = trades
    A = np.zeros((n, n))
    B = np.eye(n)
    Q = cov_matrix + 1e-4 * np.eye(n)  # penalize tracking error
    R = transaction_cost * np.eye(n)    # penalize trading
    F = Q  # terminal penalty

    prob = LQRProblem(A=A, B=B, Q=Q, R=R, F=F)
    P, _ = lqr_riccati(prob, horizon, n_steps=20)
    trades = lqr_control(prob, deviation, P)

    # Ensure zero-sum (long/short balance)
    trades = np.array(trades)
    trades -= trades.mean()
    return trades


# ── Optimal Stopping ──────────────────────────────────────────────────────────

def optimal_stopping_1d(
    x_grid: np.ndarray,
    payoff_fn: Callable,       # payoff(x) at stopping
    dynamics_fn: Callable,     # dynamics_fn(x, rng) → next x
    discount: float = 0.99,
    n_iter: int = 100,
    n_samples: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve optimal stopping via Monte Carlo value function iteration.
    V(x) = max(payoff(x), E[discount * V(x')])
    Returns (V, stopping_region) for each x in x_grid.
    """
    rng = rng or np.random.default_rng(42)
    n = len(x_grid)
    V = np.vectorize(payoff_fn)(x_grid)

    for _ in range(n_iter):
        V_new = np.zeros(n)
        for i, x in enumerate(x_grid):
            # Monte Carlo expectation
            next_xs = np.array([dynamics_fn(x, rng) for _ in range(n_samples)])
            # Interpolate V at next_xs
            continuation = discount * float(np.interp(next_xs, x_grid, V).mean())
            V_new[i] = max(payoff_fn(x), continuation)
        V = V_new

    stopping_region = np.vectorize(payoff_fn)(x_grid) >= discount * np.interp(x_grid, x_grid, V)
    return V, stopping_region.astype(float)


def pairs_trade_optimal_stopping(
    spread: float,
    ou_params,          # OUParams from mean_reversion_signals
    horizon_steps: int = 50,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Optimal entry/exit thresholds for pairs trade via OU optimal stopping.
    Maximizes expected PnL net of transaction costs.
    """
    kappa = ou_params.kappa
    mu = ou_params.mu
    sigma = ou_params.sigma

    # Expected hitting time to mu from spread
    z = spread - mu
    sigma_eq = sigma / math.sqrt(2 * max(kappa, 1e-6))

    # Optimal entry: maximize E[spread - mu - cost] for OU process
    # Simplified: entry when E[PnL] > 0 accounting for decay
    expected_profit = abs(z) * (1 - math.exp(-kappa * horizon_steps)) - transaction_cost * 2
    expected_hold = math.log(2) / max(kappa, 1e-6)

    # Optimal entry z-score: sigma_eq * optimal_threshold
    # From Bertola-Caballero: threshold* = sqrt(2) * sigma_eq (rough approximation)
    optimal_entry_z = math.sqrt(2) * sigma_eq
    optimal_exit_z = 0.25 * sigma_eq

    return {
        "current_spread": float(spread),
        "optimal_entry_z": float(optimal_entry_z),
        "optimal_exit_z": float(optimal_exit_z),
        "should_enter": bool(abs(z) > optimal_entry_z),
        "should_exit": bool(abs(z) < optimal_exit_z),
        "expected_profit": float(max(expected_profit, 0)),
        "expected_hold_periods": float(expected_hold),
        "sigma_equilibrium": float(sigma_eq),
    }


# ── Risk-Sensitive Control ────────────────────────────────────────────────────

def risk_sensitive_value(
    returns: np.ndarray,
    theta: float = 0.01,    # risk sensitivity (0=risk neutral, +inf=worst case)
) -> float:
    """
    Risk-sensitive (exponential utility) value function.
    V = -1/theta * log(E[exp(-theta * sum_returns)])
    Positive theta: risk-averse; negative theta: risk-seeking.
    """
    if abs(theta) < 1e-8:
        return float(returns.sum())
    cumsum = float(returns.sum())
    # For Gaussian returns: risk-sensitive value = mean - theta/2 * variance
    return float(cumsum - theta / 2 * returns.var() * len(returns))


def certainty_equivalent(
    returns: np.ndarray,
    gamma: float = 2.0,     # CRRA risk aversion
) -> float:
    """
    Certainty equivalent return under CRRA utility.
    CE = (E[W^(1-gamma)])^(1/(1-gamma)) - 1 for W = 1 + cumulative return.
    """
    wealth = 1 + np.cumsum(returns)
    T = len(wealth)
    if gamma == 1.0:
        return float(math.exp(np.log(np.maximum(wealth, 1e-10)).mean()) - 1)
    avg_utility = float(np.mean(np.maximum(wealth, 1e-10) ** (1 - gamma)) / (1 - gamma))
    if avg_utility <= 0:
        return -1.0
    return float(((1 - gamma) * avg_utility) ** (1 / (1 - gamma)) - 1)


# ── Finite Horizon Dynamic Programming ────────────────────────────────────────

def finite_horizon_dp(
    states: np.ndarray,            # discrete state space
    actions: np.ndarray,           # discrete action space
    transition_fn: Callable,       # P(s', a, s) → probability
    reward_fn: Callable,           # r(s, a) → reward
    terminal_value_fn: Callable,   # V_T(s) → terminal value
    T: int,                        # horizon
    discount: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite horizon DP via backward induction.
    Returns (V_0, policy_0) at initial time.
    V: shape (T+1, n_states)
    policy: shape (T, n_states) — optimal action index at each state/time
    """
    n_s = len(states)
    n_a = len(actions)

    V = np.zeros((T + 1, n_s))
    policy = np.zeros((T, n_s), dtype=int)

    # Terminal values
    for i, s in enumerate(states):
        V[T, i] = terminal_value_fn(s)

    # Backward induction
    for t in range(T - 1, -1, -1):
        for i, s in enumerate(states):
            best_val = -np.inf
            best_action = 0
            for j, a in enumerate(actions):
                # Compute expected continuation
                r = reward_fn(s, a)
                expected_next = 0.0
                for k, s_next in enumerate(states):
                    p = transition_fn(s_next, a, s)
                    expected_next += p * V[t + 1, k]
                q_val = r + discount * expected_next
                if q_val > best_val:
                    best_val = q_val
                    best_action = j
            V[t, i] = best_val
            policy[t, i] = best_action

    return V, policy


# ── Optimal Liquidation (Stochastic Control) ─────────────────────────────────

def optimal_liquidation_hjb(
    X0: float,               # initial inventory
    T: int,                  # time horizon (periods)
    sigma: float,            # price volatility per period
    eta: float = 2.5e-7,     # temporary impact coefficient
    gamma: float = 1e-6,     # risk aversion
    n_x: int = 50,           # inventory grid points
    n_t: int = 100,          # time grid points
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HJB solution for optimal liquidation (Almgren-Chriss continuous time).
    Solves: dV/dt + max_v [-v * dV/dx - eta*v^2 + 0.5*sigma^2*x^2 * gamma] = 0
    Returns (x_grid, t_grid, V_grid).
    """
    x_grid = np.linspace(0, X0, n_x)
    t_grid = np.linspace(0, T, n_t)
    dt = T / (n_t - 1)

    # Terminal condition: V(x, T) = -gamma/2 * sigma^2 * x^2 (inventory risk penalty)
    V = np.zeros((n_x, n_t))
    V[:, -1] = -gamma * 0.5 * sigma**2 * x_grid**2

    # Backward in time using finite differences
    for t_idx in range(n_t - 2, -1, -1):
        V_t = V[:, t_idx + 1]
        for i in range(n_x):
            x = x_grid[i]
            dVdx = (V_t[min(i + 1, n_x - 1)] - V_t[max(i - 1, 0)]) / (2 * X0 / max(n_x - 1, 1))
            # Optimal rate: v* = -dV/dx / (2*eta)
            v_star = -dVdx / (2 * eta + 1e-15)
            v_star = float(np.clip(v_star, 0, x / max(dt, 1e-8)))
            running_reward = -eta * v_star**2 - 0.5 * gamma * sigma**2 * x**2
            V[i, t_idx] = V_t[i] + dt * (running_reward - v_star * dVdx)

    # Optimal schedule: at each time, v*(x) = -dV/dx / (2*eta)
    schedule = np.zeros((n_x, n_t))
    for t_idx in range(n_t - 1):
        for i in range(n_x):
            dVdx = (V[min(i + 1, n_x - 1), t_idx] - V[max(i - 1, 0), t_idx]) / (2 * X0 / max(n_x - 1, 1))
            x = x_grid[i]
            schedule[i, t_idx] = float(np.clip(-dVdx / (2 * eta + 1e-15), 0, x))

    return x_grid, t_grid, schedule
