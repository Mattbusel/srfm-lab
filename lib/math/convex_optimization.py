"""
Convex optimization solvers for portfolio problems.

Implements projected gradient descent, ADMM, proximal operators, interior point
methods, quadratic programming, CVaR optimization, and multi-objective methods.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------

def prox_l1(x: np.ndarray, lam: float) -> np.ndarray:
    """Soft-thresholding (proximal operator of lam * ||x||_1)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def prox_l2(x: np.ndarray, lam: float) -> np.ndarray:
    """Proximal operator of lam * ||x||_2 (block soft-thresholding)."""
    norm_x = np.linalg.norm(x)
    if norm_x <= lam:
        return np.zeros_like(x)
    return x * (1.0 - lam / norm_x)


def prox_l2_squared(x: np.ndarray, lam: float) -> np.ndarray:
    """Proximal operator of lam * ||x||_2^2."""
    return x / (1.0 + 2.0 * lam)


def prox_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Project onto box constraints [lb, ub]."""
    return np.clip(x, lb, ub)


def project_simplex(x: np.ndarray) -> np.ndarray:
    """
    Project x onto the probability simplex {w >= 0, sum(w) = 1}.
    Algorithm from Duchi et al. (2008).
    """
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_candidates = u - cssv / np.arange(1, n + 1)
    rho = np.max(np.where(rho_candidates > 0)[0]) + 1 if np.any(rho_candidates > 0) else 1
    theta = cssv[rho - 1] / float(rho)
    return np.maximum(x - theta, 0.0)


def project_l1_ball(x: np.ndarray, radius: float) -> np.ndarray:
    """Project x onto the L1 ball of given radius."""
    if np.linalg.norm(x, 1) <= radius:
        return x.copy()
    # Use simplex projection on |x|
    abs_x = np.abs(x)
    u = np.sort(abs_x)[::-1]
    cssv = np.cumsum(u) - radius
    n = len(x)
    rho_candidates = u - cssv / np.arange(1, n + 1)
    rho = np.max(np.where(rho_candidates > 0)[0]) + 1
    theta = cssv[rho - 1] / float(rho)
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0.0)


def project_nonneg(x: np.ndarray) -> np.ndarray:
    """Project onto non-negative orthant."""
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Projected gradient descent
# ---------------------------------------------------------------------------

class ProjectedGradientDescent:
    """
    Projected gradient descent for:
        min f(x)  s.t. x in C
    where C is a convex set with a known projection operator.
    """

    def __init__(self, grad_fn, proj_fn, lr: float = 1e-3,
                 max_iter: int = 5000, tol: float = 1e-8,
                 backtrack: bool = True, beta: float = 0.5, sigma: float = 0.3):
        self.grad_fn = grad_fn
        self.proj_fn = proj_fn
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.backtrack = backtrack
        self.beta = beta
        self.sigma = sigma

    def solve(self, x0: np.ndarray, f_fn=None) -> dict:
        """
        Run projected gradient descent.

        Parameters
        ----------
        x0 : initial point
        f_fn : objective function (needed for backtracking line search)

        Returns
        -------
        dict with 'x', 'iterations', 'converged', 'history'
        """
        x = self.proj_fn(x0.copy())
        history = []
        converged = False

        for k in range(self.max_iter):
            g = self.grad_fn(x)

            if self.backtrack and f_fn is not None:
                step = self._backtracking_line_search(x, g, f_fn)
            else:
                step = self.lr

            x_new = self.proj_fn(x - step * g)
            diff = np.linalg.norm(x_new - x)
            history.append(diff)

            if diff < self.tol:
                converged = True
                x = x_new
                break
            x = x_new

        return {'x': x, 'iterations': k + 1, 'converged': converged, 'history': history}

    def _backtracking_line_search(self, x, g, f_fn):
        """Armijo backtracking line search."""
        step = self.lr
        fx = f_fn(x)
        gnorm_sq = np.dot(g, g)

        for _ in range(30):
            x_trial = self.proj_fn(x - step * g)
            if f_fn(x_trial) <= fx - self.sigma * step * gnorm_sq:
                return step
            step *= self.beta
        return step


# ---------------------------------------------------------------------------
# ADMM general solver
# ---------------------------------------------------------------------------

class ADMM:
    """
    Alternating Direction Method of Multipliers for:
        min f(x) + g(z)
        s.t. Ax + Bz = c

    Default form (consensus): min f(x) + g(z) s.t. x - z = 0
    """

    def __init__(self, prox_f, prox_g, rho: float = 1.0,
                 max_iter: int = 5000, tol_abs: float = 1e-6,
                 tol_rel: float = 1e-4):
        """
        Parameters
        ----------
        prox_f : callable(v, rho) -> x minimizing f(x) + (rho/2)||x - v||^2
        prox_g : callable(v, rho) -> z minimizing g(z) + (rho/2)||z - v||^2
        rho : penalty parameter
        """
        self.prox_f = prox_f
        self.prox_g = prox_g
        self.rho = rho
        self.max_iter = max_iter
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel

    def solve(self, x0: np.ndarray) -> dict:
        """Run ADMM consensus: min f(x) + g(z) s.t. x = z."""
        n = len(x0)
        x = x0.copy()
        z = x0.copy()
        u = np.zeros(n)  # scaled dual variable

        primal_residuals = []
        dual_residuals = []
        converged = False

        for k in range(self.max_iter):
            # x-update
            x = self.prox_f(z - u, self.rho)

            # z-update
            z_old = z.copy()
            z = self.prox_g(x + u, self.rho)

            # dual update
            u = u + x - z

            # residuals
            r_primal = np.linalg.norm(x - z)
            r_dual = self.rho * np.linalg.norm(z - z_old)
            primal_residuals.append(r_primal)
            dual_residuals.append(r_dual)

            # convergence check
            eps_primal = np.sqrt(n) * self.tol_abs + self.tol_rel * max(
                np.linalg.norm(x), np.linalg.norm(z))
            eps_dual = np.sqrt(n) * self.tol_abs + self.tol_rel * np.linalg.norm(self.rho * u)

            if r_primal < eps_primal and r_dual < eps_dual:
                converged = True
                break

        return {
            'x': x, 'z': z, 'u': u,
            'iterations': k + 1,
            'converged': converged,
            'primal_residuals': primal_residuals,
            'dual_residuals': dual_residuals,
        }

    def solve_with_linear_constraint(self, x0: np.ndarray,
                                     A: np.ndarray, b: np.ndarray) -> dict:
        """
        ADMM for: min f(x) + g(z) s.t. Ax + z = b.
        Uses factorisation caching for efficiency.
        """
        n = len(x0)
        m = len(b)
        x = x0.copy()
        z = np.zeros(m)
        u = np.zeros(m)

        converged = False
        primal_res = []
        dual_res = []

        # precompute for x-update when f is quadratic
        for k in range(self.max_iter):
            # x-update: prox_f(v, rho) where v chosen so that penalty on Ax+z-b=0
            v = x - (1.0 / self.rho) * A.T @ (A @ x + z - b + u)
            x = self.prox_f(v, self.rho)

            # z-update
            z_old = z.copy()
            Az = A @ x - b + u
            z = self.prox_g(-Az, self.rho)

            # dual
            u = u + A @ x + z - b

            r_p = np.linalg.norm(A @ x + z - b)
            r_d = self.rho * np.linalg.norm(A.T @ (z - z_old))
            primal_res.append(r_p)
            dual_res.append(r_d)

            eps_p = np.sqrt(m) * self.tol_abs + self.tol_rel * max(
                np.linalg.norm(A @ x), np.linalg.norm(z), np.linalg.norm(b))
            eps_d = np.sqrt(n) * self.tol_abs + self.tol_rel * np.linalg.norm(A.T @ u)

            if r_p < eps_p and r_d < eps_d:
                converged = True
                break

        return {
            'x': x, 'z': z, 'u': u,
            'iterations': k + 1, 'converged': converged,
            'primal_residuals': primal_res, 'dual_residuals': dual_res,
        }


# ---------------------------------------------------------------------------
# Log-barrier interior point method
# ---------------------------------------------------------------------------

class LogBarrierMethod:
    """
    Log-barrier (interior point) method for:
        min f(x) s.t. g_i(x) <= 0, Ax = b

    Uses Newton steps on the barrier subproblem:
        min t * f(x) - sum log(-g_i(x))
    """

    def __init__(self, f_fn, grad_f, hess_f,
                 g_fns, grad_gs, hess_gs,
                 A=None, b=None,
                 mu: float = 10.0, t0: float = 1.0,
                 tol_outer: float = 1e-6, tol_inner: float = 1e-8,
                 max_outer: int = 50, max_inner: int = 100):
        self.f_fn = f_fn
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.g_fns = g_fns          # list of inequality constraint functions
        self.grad_gs = grad_gs      # list of gradient functions
        self.hess_gs = hess_gs      # list of hessian functions
        self.A = A
        self.b = b
        self.mu = mu
        self.t0 = t0
        self.tol_outer = tol_outer
        self.tol_inner = tol_inner
        self.max_outer = max_outer
        self.max_inner = max_inner

    def _barrier_val(self, x):
        """Evaluate sum of -log(-g_i(x))."""
        val = 0.0
        for g in self.g_fns:
            gi = g(x)
            if gi >= 0:
                return np.inf
            val -= np.log(-gi)
        return val

    def _barrier_grad(self, x):
        """Gradient of barrier: sum_i (1/(-g_i(x))) * grad g_i(x)."""
        grad = np.zeros_like(x)
        for g, dg in zip(self.g_fns, self.grad_gs):
            gi = g(x)
            grad += (1.0 / (-gi)) * dg(x)
        return grad

    def _barrier_hess(self, x):
        """Hessian of barrier term."""
        n = len(x)
        H = np.zeros((n, n))
        for g, dg, ddg in zip(self.g_fns, self.grad_gs, self.hess_gs):
            gi = g(x)
            dgi = dg(x)
            H += (1.0 / (gi ** 2)) * np.outer(dgi, dgi) + (1.0 / (-gi)) * ddg(x)
        return H

    def _newton_step(self, x, t):
        """Compute Newton step for barrier subproblem."""
        n = len(x)
        grad = t * self.grad_f(x) + self._barrier_grad(x)
        hess = t * self.hess_f(x) + self._barrier_hess(x)

        if self.A is not None:
            # KKT system with equality constraints
            m = self.A.shape[0]
            KKT = np.zeros((n + m, n + m))
            KKT[:n, :n] = hess
            KKT[:n, n:] = self.A.T
            KKT[n:, :n] = self.A
            rhs = np.zeros(n + m)
            rhs[:n] = -grad
            rhs[n:] = -(self.A @ x - self.b) if self.b is not None else -(self.A @ x)
            sol = linalg.solve(KKT, rhs, assume_a='sym')
            dx = sol[:n]
        else:
            dx = linalg.solve(hess, -grad, assume_a='pos')

        return dx

    def _line_search(self, x, dx, t, alpha=0.01, beta=0.5):
        """Backtracking line search ensuring feasibility."""
        s = 1.0
        m = len(self.g_fns)

        # ensure feasibility
        for _ in range(50):
            feasible = all(g(x + s * dx) < 0 for g in self.g_fns)
            if feasible:
                break
            s *= beta

        # Armijo condition
        f0 = t * self.f_fn(x) + self._barrier_val(x)
        grad0 = t * self.grad_f(x) + self._barrier_grad(x)
        slope = grad0 @ dx

        for _ in range(50):
            x_new = x + s * dx
            f_new = t * self.f_fn(x_new) + self._barrier_val(x_new)
            if f_new <= f0 + alpha * s * slope:
                break
            s *= beta
        return s

    def solve(self, x0: np.ndarray) -> dict:
        """Run the barrier method."""
        x = x0.copy()
        t = self.t0
        m = len(self.g_fns)
        history = []

        for outer in range(self.max_outer):
            # centering step: minimise t*f + barrier via Newton
            for inner in range(self.max_inner):
                dx = self._newton_step(x, t)
                decrement = -self.grad_f(x) @ dx
                if decrement / 2.0 < self.tol_inner:
                    break
                s = self._line_search(x, dx, t)
                x = x + s * dx

            history.append({'t': t, 'f': self.f_fn(x), 'inner_iters': inner + 1})

            # duality gap
            if m / t < self.tol_outer:
                return {'x': x, 'converged': True,
                        'outer_iterations': outer + 1, 'history': history}

            t *= self.mu

        return {'x': x, 'converged': False,
                'outer_iterations': self.max_outer, 'history': history}


# ---------------------------------------------------------------------------
# Active-set QP for portfolio optimization
# ---------------------------------------------------------------------------

class ActiveSetQP:
    """
    Active-set method for quadratic programming:
        min  0.5 * x^T Q x + c^T x
        s.t. A_eq x = b_eq
             x >= 0  (or general bounds)

    Specialized for portfolio problems where Q = covariance matrix.
    """

    def __init__(self, Q: np.ndarray, c: np.ndarray,
                 A_eq: np.ndarray = None, b_eq: np.ndarray = None,
                 lb: np.ndarray = None, ub: np.ndarray = None,
                 max_iter: int = 10000, tol: float = 1e-10):
        self.Q = Q
        self.c = c
        self.n = len(c)
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.lb = lb if lb is not None else np.zeros(self.n)
        self.ub = ub if ub is not None else np.full(self.n, np.inf)
        self.max_iter = max_iter
        self.tol = tol

    def solve(self) -> dict:
        """Solve the QP using a primal active-set method."""
        n = self.n
        x = self._find_feasible_start()
        if x is None:
            return {'x': None, 'converged': False, 'message': 'No feasible start found'}

        # active set: indices where bounds are active
        active_lower = set(np.where(np.abs(x - self.lb) < self.tol)[0])
        active_upper = set(np.where(np.abs(x - self.ub) < self.tol)[0])

        for k in range(self.max_iter):
            # gradient
            g = self.Q @ x + self.c

            # free variables
            free = [i for i in range(n) if i not in active_lower and i not in active_upper]

            if len(free) == 0:
                return {'x': x, 'converged': True, 'iterations': k,
                        'objective': 0.5 * x @ self.Q @ x + self.c @ x}

            # solve reduced QP for free variables
            Q_ff = self.Q[np.ix_(free, free)]
            g_f = g[free]

            if self.A_eq is not None:
                A_f = self.A_eq[:, free]
                m = A_f.shape[0]
                # KKT for reduced problem
                KKT = np.zeros((len(free) + m, len(free) + m))
                KKT[:len(free), :len(free)] = Q_ff
                KKT[:len(free), len(free):] = A_f.T
                KKT[len(free):, :len(free)] = A_f
                rhs = np.zeros(len(free) + m)
                rhs[:len(free)] = -g_f
                try:
                    sol = linalg.solve(KKT, rhs)
                except linalg.LinAlgError:
                    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
                p_free = sol[:len(free)]
            else:
                try:
                    p_free = linalg.solve(Q_ff, -g_f, assume_a='pos')
                except linalg.LinAlgError:
                    p_free = np.linalg.lstsq(Q_ff, -g_f, rcond=None)[0]

            p = np.zeros(n)
            for i, fi in enumerate(free):
                p[fi] = p_free[i]

            if np.linalg.norm(p) < self.tol:
                # check multipliers for active constraints
                drop_idx = None
                most_negative = 0.0

                for i in active_lower:
                    lam = g[i]  # multiplier for lower bound
                    if lam < most_negative:
                        most_negative = lam
                        drop_idx = ('lower', i)

                for i in active_upper:
                    lam = -g[i]  # multiplier for upper bound
                    if lam < most_negative:
                        most_negative = lam
                        drop_idx = ('upper', i)

                if drop_idx is None:
                    return {'x': x, 'converged': True, 'iterations': k,
                            'objective': 0.5 * x @ self.Q @ x + self.c @ x}
                else:
                    if drop_idx[0] == 'lower':
                        active_lower.discard(drop_idx[1])
                    else:
                        active_upper.discard(drop_idx[1])
                continue

            # step length
            alpha = 1.0
            blocking_idx = None
            blocking_type = None

            for i in free:
                if p[i] < -self.tol:
                    a_i = (self.lb[i] - x[i]) / p[i]
                    if a_i < alpha:
                        alpha = a_i
                        blocking_idx = i
                        blocking_type = 'lower'
                elif p[i] > self.tol:
                    a_i = (self.ub[i] - x[i]) / p[i]
                    if a_i < alpha:
                        alpha = a_i
                        blocking_idx = i
                        blocking_type = 'upper'

            x = x + alpha * p

            if blocking_idx is not None:
                if blocking_type == 'lower':
                    active_lower.add(blocking_idx)
                    x[blocking_idx] = self.lb[blocking_idx]
                else:
                    active_upper.add(blocking_idx)
                    x[blocking_idx] = self.ub[blocking_idx]

        return {'x': x, 'converged': False, 'iterations': self.max_iter,
                'objective': 0.5 * x @ self.Q @ x + self.c @ x}

    def _find_feasible_start(self) -> np.ndarray:
        """Find a feasible starting point."""
        n = self.n
        if self.A_eq is not None:
            # least-norm solution satisfying equality constraints
            x = self.A_eq.T @ linalg.solve(
                self.A_eq @ self.A_eq.T, self.b_eq)
        else:
            x = np.full(n, 0.5 * (np.mean(self.lb) + np.clip(np.mean(self.ub), 0, 1)))

        x = np.clip(x, self.lb, self.ub)

        if self.A_eq is not None:
            residual = self.A_eq @ x - self.b_eq
            if np.linalg.norm(residual) > 1e-6:
                # try to fix via nullspace
                try:
                    null = linalg.null_space(self.A_eq)
                    if null.shape[1] > 0:
                        correction = null @ linalg.lstsq(
                            self.A_eq @ null, -residual)[0]
                        x = x + correction
                        x = np.clip(x, self.lb, self.ub)
                except Exception:
                    pass
        return x


# ---------------------------------------------------------------------------
# CVaR optimization via LP reformulation
# ---------------------------------------------------------------------------

class CVaROptimizer:
    """
    Conditional Value at Risk (CVaR) portfolio optimization.

    min  CVaR_alpha(portfolio loss)
    s.t. w >= 0, sum(w) = 1

    Uses the Rockafellar-Uryasev LP reformulation:
        min_{w, alpha, u}  alpha + (1/(S*(1-beta))) * sum(u_s)
        s.t. u_s >= -r_s^T w - alpha,  u_s >= 0
             sum(w) = 1, w >= 0
    """

    def __init__(self, returns: np.ndarray, alpha: float = 0.95,
                 max_iter: int = 10000, tol: float = 1e-8):
        """
        Parameters
        ----------
        returns : (S, n) array of scenario returns
        alpha : confidence level (e.g. 0.95 for 95% CVaR)
        """
        self.returns = returns
        self.S, self.n = returns.shape
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def solve(self) -> dict:
        """
        Solve CVaR minimization via projected gradient descent on the
        smoothed LP formulation.
        """
        n = self.n
        S = self.S
        beta = self.alpha
        R = self.returns  # (S, n)

        # decision: w (n), plus auxiliary variable zeta (VaR threshold)
        w = np.ones(n) / n
        zeta = 0.0
        lr_w = 1e-3
        lr_z = 1e-3

        history = []

        for k in range(self.max_iter):
            # losses for each scenario
            losses = -R @ w  # (S,)

            # CVaR subgradient
            exceedances = losses - zeta
            indicator = (exceedances > 0).astype(float)

            # gradient w.r.t. zeta
            grad_zeta = 1.0 - (1.0 / (S * (1.0 - beta))) * np.sum(indicator)

            # gradient w.r.t. w
            grad_w = -(1.0 / (S * (1.0 - beta))) * R.T @ indicator

            # update
            zeta_new = zeta - lr_z * grad_zeta
            w_new = w - lr_w * grad_w
            w_new = project_simplex(w_new)

            diff = np.linalg.norm(w_new - w) + abs(zeta_new - zeta)
            cvar_val = zeta + (1.0 / (S * (1.0 - beta))) * np.sum(
                np.maximum(losses - zeta, 0.0))
            history.append(cvar_val)

            w = w_new
            zeta = zeta_new

            if diff < self.tol:
                break

        losses = -R @ w
        cvar = zeta + (1.0 / (S * (1.0 - beta))) * np.sum(
            np.maximum(losses - zeta, 0.0))
        var = zeta

        return {
            'weights': w,
            'VaR': var,
            'CVaR': cvar,
            'iterations': k + 1,
            'history': history,
        }

    def solve_with_return_constraint(self, min_return: float) -> dict:
        """
        min CVaR  s.t. E[r^T w] >= min_return, w in simplex.
        """
        n = self.n
        S = self.S
        beta = self.alpha
        R = self.returns
        mu = R.mean(axis=0)

        w = np.ones(n) / n
        zeta = 0.0
        lam = 0.0  # Lagrange multiplier for return constraint
        lr_w = 1e-3
        lr_z = 1e-3
        lr_lam = 1e-3

        for k in range(self.max_iter):
            losses = -R @ w
            exceedances = losses - zeta
            indicator = (exceedances > 0).astype(float)

            grad_zeta = 1.0 - (1.0 / (S * (1.0 - beta))) * np.sum(indicator)
            grad_w = -(1.0 / (S * (1.0 - beta))) * R.T @ indicator - lam * mu

            zeta = zeta - lr_z * grad_zeta
            w_new = w - lr_w * grad_w
            w_new = project_simplex(w_new)

            # dual update
            lam = max(0.0, lam - lr_lam * (mu @ w_new - min_return))

            diff = np.linalg.norm(w_new - w)
            w = w_new

            if diff < self.tol:
                break

        losses = -R @ w
        cvar = zeta + (1.0 / (S * (1.0 - beta))) * np.sum(
            np.maximum(losses - zeta, 0.0))

        return {
            'weights': w, 'VaR': zeta, 'CVaR': cvar,
            'expected_return': mu @ w, 'iterations': k + 1,
        }


# ---------------------------------------------------------------------------
# Bisection method for 1D optimization
# ---------------------------------------------------------------------------

def bisection_root(f, a: float, b: float, tol: float = 1e-10,
                   max_iter: int = 100) -> dict:
    """Find root of f in [a, b] via bisection. Requires f(a)*f(b) < 0."""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    for k in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return {'root': mid, 'iterations': k + 1, 'converged': True}
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

    return {'root': 0.5 * (a + b), 'iterations': max_iter, 'converged': False}


def bisection_minimize(f, a: float, b: float, tol: float = 1e-10,
                       max_iter: int = 100) -> dict:
    """Golden-section search for unimodal f on [a, b]."""
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    for k in range(max_iter):
        if (b - a) < tol:
            break
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)

    xmin = 0.5 * (a + b)
    return {'x': xmin, 'f': f(xmin), 'iterations': k + 1}


# ---------------------------------------------------------------------------
# SOCP solver (simplified interior point)
# ---------------------------------------------------------------------------

class SOCPSolver:
    """
    Simplified second-order cone program solver:
        min  c^T x
        s.t. ||A_i x + b_i|| <= c_i^T x + d_i   for each cone constraint
             Fx = g  (equality)

    Uses a log-barrier approach on the cone constraints.
    The barrier for SOC constraint ||u|| <= t is: -log(t^2 - ||u||^2).
    """

    def __init__(self, c: np.ndarray, cone_constraints: list,
                 F: np.ndarray = None, g: np.ndarray = None,
                 mu: float = 10.0, tol: float = 1e-6, max_iter: int = 50):
        """
        Parameters
        ----------
        c : linear objective
        cone_constraints : list of dicts with keys 'A', 'b', 'c_vec', 'd'
            representing ||A x + b|| <= c_vec^T x + d
        F, g : equality constraints Fx = g
        """
        self.c_obj = c
        self.cones = cone_constraints
        self.F = F
        self.g_eq = g
        self.mu = mu
        self.tol = tol
        self.max_iter = max_iter

    def _barrier(self, x):
        """Evaluate SOC barrier: -log(t^2 - ||u||^2) for each cone."""
        val = 0.0
        for cone in self.cones:
            u = cone['A'] @ x + cone['b']
            t = cone['c_vec'] @ x + cone['d']
            slack = t ** 2 - np.dot(u, u)
            if slack <= 0:
                return np.inf
            val -= np.log(slack)
        return val

    def _barrier_grad(self, x):
        """Gradient of SOC barrier."""
        grad = np.zeros_like(x)
        for cone in self.cones:
            u = cone['A'] @ x + cone['b']
            t = cone['c_vec'] @ x + cone['d']
            slack = t ** 2 - np.dot(u, u)
            # d/dx (-log(t^2 - ||u||^2))
            grad += (2.0 / slack) * (cone['A'].T @ u - t * cone['c_vec'])
        return grad

    def _barrier_hess(self, x):
        """Approximate Hessian of SOC barrier."""
        n = len(x)
        H = np.zeros((n, n))
        for cone in self.cones:
            A = cone['A']
            u = A @ x + cone['b']
            cv = cone['c_vec']
            t = cv @ x + cone['d']
            slack = t ** 2 - np.dot(u, u)

            # gradient components
            du = A.T @ u
            q = du - t * cv
            H += (4.0 / (slack ** 2)) * np.outer(q, q)
            H += (2.0 / slack) * (A.T @ A - np.outer(cv, cv))
        return H

    def solve(self, x0: np.ndarray) -> dict:
        """Solve via barrier method."""
        x = x0.copy()
        t = 1.0
        m = len(self.cones)

        for outer in range(self.max_iter):
            # Newton centering
            for _ in range(100):
                grad = t * self.c_obj + self._barrier_grad(x)
                hess = self._barrier_hess(x)

                if self.F is not None:
                    n = len(x)
                    p = self.F.shape[0]
                    KKT = np.zeros((n + p, n + p))
                    KKT[:n, :n] = hess
                    KKT[:n, n:] = self.F.T
                    KKT[n:, :n] = self.F
                    rhs = np.concatenate([-grad, np.zeros(p)])
                    try:
                        sol = linalg.solve(KKT, rhs)
                    except linalg.LinAlgError:
                        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
                    dx = sol[:n]
                else:
                    try:
                        dx = linalg.solve(hess, -grad)
                    except linalg.LinAlgError:
                        dx = np.linalg.lstsq(hess, -grad, rcond=None)[0]

                newton_dec = -grad @ dx
                if newton_dec / 2.0 < 1e-8:
                    break

                # backtracking
                s = 1.0
                for _ in range(30):
                    x_new = x + s * dx
                    if self._barrier(x_new) < np.inf:
                        break
                    s *= 0.5
                x = x + s * dx

            if m / t < self.tol:
                return {'x': x, 'converged': True, 'objective': self.c_obj @ x,
                        'outer_iterations': outer + 1}
            t *= self.mu

        return {'x': x, 'converged': False, 'objective': self.c_obj @ x,
                'outer_iterations': self.max_iter}


# ---------------------------------------------------------------------------
# Multi-objective optimization
# ---------------------------------------------------------------------------

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization via epsilon-constraint and weighted-sum methods.
    Designed for portfolio problems with competing objectives (risk vs return).
    """

    @staticmethod
    def weighted_sum(objectives: list, gradients: list,
                     weights: np.ndarray, proj_fn,
                     x0: np.ndarray, lr: float = 1e-3,
                     max_iter: int = 5000, tol: float = 1e-8) -> dict:
        """
        min  sum_k  w_k * f_k(x)  s.t. x in C

        Parameters
        ----------
        objectives : list of callables f_k(x)
        gradients : list of callables grad f_k(x)
        weights : convex combination weights (sum to 1)
        proj_fn : projection onto feasible set
        """
        x = proj_fn(x0.copy())
        history = []

        for k in range(max_iter):
            grad = sum(w * g(x) for w, g in zip(weights, gradients))
            x_new = proj_fn(x - lr * grad)
            diff = np.linalg.norm(x_new - x)
            vals = [f(x_new) for f in objectives]
            history.append(vals)
            if diff < tol:
                x = x_new
                break
            x = x_new

        return {
            'x': x,
            'objective_values': [f(x) for f in objectives],
            'iterations': k + 1,
            'history': history,
        }

    @staticmethod
    def epsilon_constraint(primary_obj, primary_grad,
                           secondary_objs, secondary_grads,
                           epsilons: list,
                           proj_fn, x0: np.ndarray,
                           penalty: float = 100.0,
                           lr: float = 1e-3, max_iter: int = 5000,
                           tol: float = 1e-8) -> dict:
        """
        min  f_1(x)
        s.t. f_k(x) <= eps_k  for k = 2,...,K
             x in C

        Uses penalty method for the epsilon constraints.
        """
        x = proj_fn(x0.copy())

        for iteration in range(max_iter):
            # augmented gradient: primary + penalty for violated constraints
            grad = primary_grad(x)
            for sec_f, sec_g, eps in zip(secondary_objs, secondary_grads, epsilons):
                violation = sec_f(x) - eps
                if violation > 0:
                    grad += penalty * violation * sec_g(x)

            x_new = proj_fn(x - lr * grad)
            diff = np.linalg.norm(x_new - x)
            if diff < tol:
                x = x_new
                break
            x = x_new

        return {
            'x': x,
            'primary_value': primary_obj(x),
            'secondary_values': [f(x) for f in secondary_objs],
            'iterations': iteration + 1,
        }

    @staticmethod
    def pareto_front(objectives: list, gradients: list,
                     proj_fn, x0: np.ndarray,
                     n_points: int = 20, **kwargs) -> list:
        """
        Trace out the Pareto front for 2 objectives using weighted sum
        with varying weights.
        """
        results = []
        for i in range(n_points + 1):
            w1 = i / n_points
            w2 = 1.0 - w1
            weights = np.array([w1, w2])
            res = MultiObjectiveOptimizer.weighted_sum(
                objectives, gradients, weights, proj_fn, x0.copy(), **kwargs)
            results.append({
                'weights': weights.copy(),
                'x': res['x'],
                'objective_values': res['objective_values'],
            })
        return results


# ---------------------------------------------------------------------------
# Portfolio optimization with transaction costs
# ---------------------------------------------------------------------------

class PortfolioTransactionCostOptimizer:
    """
    min  0.5 * w^T Sigma w - lambda * mu^T w + kappa * ||w - w_prev||_1
    s.t. sum(w) = 1, w >= 0

    Transaction costs modeled as L1 penalty on trades.
    Solved via ADMM splitting.
    """

    def __init__(self, Sigma: np.ndarray, mu: np.ndarray,
                 w_prev: np.ndarray, risk_aversion: float = 1.0,
                 transaction_cost: float = 0.001,
                 max_iter: int = 5000, tol: float = 1e-8):
        self.Sigma = Sigma
        self.mu = mu
        self.w_prev = w_prev
        self.lam = risk_aversion
        self.kappa = transaction_cost
        self.n = len(mu)
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, rho: float = 1.0) -> dict:
        """
        ADMM formulation:
            min f(w) + g(z)  s.t. w = z
        where f(w) = 0.5 w^T Sigma w - lam * mu^T w + kappa * ||w - w_prev||_1
              g(z) = indicator(z in simplex ∩ z >= 0)
        """
        n = self.n
        Sigma = self.Sigma
        mu_vec = self.mu
        lam = self.lam
        kappa = self.kappa
        w_prev = self.w_prev

        # precompute factorization for x-update
        # x-update: min 0.5 x^T Sigma x - lam mu^T x + kappa||x-w_prev||_1
        #           + (rho/2)||x - z + u||^2
        # We split: let x = v + w_prev, then L1 on v
        # Actually simpler: use prox of L1 + quadratic via linearization

        w = np.ones(n) / n
        z = w.copy()
        u = np.zeros(n)

        # factor for quadratic part: (Sigma + rho * I)
        L_factor = linalg.cho_factor(Sigma + rho * np.eye(n))

        history = []

        for k in range(self.max_iter):
            # w-update: solve quadratic + L1
            # min 0.5 w^T Sigma w - lam mu^T w + kappa ||w - w_prev||_1
            #     + (rho/2)||w - z + u||^2
            # Approximate: linearize L1 around current w, or use
            # splitting within the w-update itself.
            # Simple approach: iterate between quadratic solve and soft-threshold

            rhs = lam * mu_vec + rho * (z - u)
            w_quad = linalg.cho_solve(L_factor, rhs)

            # apply L1 prox for transaction cost term
            w_new = w_prev + prox_l1(w_quad - w_prev, kappa / rho)

            # z-update: project onto simplex
            z_old = z.copy()
            z = project_simplex(w_new + u)

            # dual update
            u = u + w_new - z

            r_primal = np.linalg.norm(w_new - z)
            r_dual = rho * np.linalg.norm(z - z_old)
            history.append((r_primal, r_dual))

            w = w_new

            if r_primal < self.tol and r_dual < self.tol:
                break

        trades = w - w_prev
        obj = (0.5 * w @ Sigma @ w - lam * mu_vec @ w
               + kappa * np.sum(np.abs(trades)))

        return {
            'weights': w,
            'trades': trades,
            'objective': obj,
            'risk': w @ Sigma @ w,
            'expected_return': mu_vec @ w,
            'transaction_cost': kappa * np.sum(np.abs(trades)),
            'iterations': k + 1,
            'history': history,
        }

    def efficient_frontier_with_costs(self, n_points: int = 20) -> dict:
        """Trace efficient frontier accounting for transaction costs."""
        risk_aversions = np.logspace(-2, 2, n_points)
        frontier = []

        for lam in risk_aversions:
            self.lam = lam
            res = self.solve()
            frontier.append({
                'risk_aversion': lam,
                'weights': res['weights'],
                'risk': res['risk'],
                'return': res['expected_return'],
                'transaction_cost': res['transaction_cost'],
            })

        return {
            'frontier': frontier,
            'risks': [f['risk'] for f in frontier],
            'returns': [f['return'] for f in frontier],
            'costs': [f['transaction_cost'] for f in frontier],
        }
