"""
Kalman filter suite for financial state estimation.

Implements:
  - Linear Kalman filter (pairs trading spread, latent factor tracking)
  - Extended Kalman Filter (EKF) for nonlinear dynamics
  - Unscented Kalman Filter (UKF) for highly nonlinear systems
  - Particle filter (Sequential Monte Carlo) for non-Gaussian noise
  - Kalman smoother (RTS backward pass)
  - Online hedge ratio estimation via Kalman
  - Dynamic beta estimation (time-varying CAPM)
  - Regime-switching Kalman (IMM filter)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Linear Kalman Filter ──────────────────────────────────────────────────────

@dataclass
class KalmanState:
    x: np.ndarray        # state mean
    P: np.ndarray        # state covariance

    def copy(self) -> "KalmanState":
        return KalmanState(self.x.copy(), self.P.copy())


class LinearKalmanFilter:
    """
    Linear Kalman filter: x_t = F*x_{t-1} + B*u_t + w_t,  z_t = H*x_t + v_t
    w ~ N(0, Q),  v ~ N(0, R)

    Primary use cases:
      - Latent factor tracking (BH mass, drift)
      - Pairs spread estimation with time-varying hedge ratio
      - Trend/level/season decomposition
    """

    def __init__(
        self,
        F: np.ndarray,     # state transition
        H: np.ndarray,     # observation matrix
        Q: np.ndarray,     # process noise covariance
        R: np.ndarray,     # observation noise covariance
        B: Optional[np.ndarray] = None,   # control input matrix
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B if B is not None else np.zeros((F.shape[0], 1))
        n = F.shape[0]
        self.state = KalmanState(
            x=x0 if x0 is not None else np.zeros(n),
            P=P0 if P0 is not None else np.eye(n) * 1e3,
        )
        self._history: list[KalmanState] = []

    def predict(self, u: Optional[np.ndarray] = None) -> KalmanState:
        """Predict step."""
        u = u if u is not None else np.zeros((self.B.shape[1],))
        x_pred = self.F @ self.state.x + self.B @ u
        P_pred = self.F @ self.state.P @ self.F.T + self.Q
        return KalmanState(x_pred, P_pred)

    def update(self, z: np.ndarray, pred: Optional[KalmanState] = None) -> tuple[KalmanState, float]:
        """Update step. Returns (state, log_likelihood)."""
        if pred is None:
            pred = self.predict()

        y = z - self.H @ pred.x                           # innovation
        S = self.H @ pred.P @ self.H.T + self.R           # innovation covariance
        K = pred.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        x_upd = pred.x + K @ y
        P_upd = (np.eye(len(pred.x)) - K @ self.H) @ pred.P

        # Log-likelihood contribution
        d = len(z)
        sign, log_det_S = np.linalg.slogdet(S)
        log_lk = -0.5 * (d * math.log(2 * math.pi) + log_det_S + float(y @ np.linalg.inv(S) @ y))

        self.state = KalmanState(x_upd, P_upd)
        self._history.append(self.state.copy())
        return self.state, log_lk

    def filter(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Run Kalman filter on observation sequence.
        observations: shape (T, m)
        Returns (x_filtered, P_filtered, total_log_lk)
        """
        T, m = observations.shape
        n = len(self.state.x)
        x_hist = np.zeros((T, n))
        P_hist = np.zeros((T, n, n))
        total_ll = 0.0

        for t in range(T):
            pred = self.predict()
            state, ll = self.update(observations[t], pred)
            x_hist[t] = state.x
            P_hist[t] = state.P
            total_ll += ll

        return x_hist, P_hist, total_ll

    def smooth(self, x_filt: np.ndarray, P_filt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """RTS backward smoother."""
        T, n = x_filt.shape
        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()

        for t in range(T - 2, -1, -1):
            P_pred = self.F @ P_filt[t] @ self.F.T + self.Q
            G = P_filt[t] @ self.F.T @ np.linalg.inv(P_pred)
            x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - self.F @ x_filt[t])
            P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred) @ G.T

        return x_smooth, P_smooth


# ── Pairs trading Kalman spread ────────────────────────────────────────────────

class KalmanPairsSpread:
    """
    Time-varying hedge ratio via Kalman filter.
    Model: y_t = beta_t * x_t + alpha_t + eps_t
    State: [alpha_t, beta_t] follows random walk.
    """

    def __init__(
        self,
        delta: float = 1e-4,    # state evolution noise (lower = slower beta changes)
        R_obs: float = 0.001,   # observation noise
    ):
        self.delta = delta
        self.R_obs = R_obs

        # State: [alpha, beta]
        self._x = np.zeros(2)           # [alpha, beta]
        self._P = np.eye(2) * 10.0

        # Process noise
        self._Q = np.eye(2) * delta

    def update(self, y: float, x: float) -> tuple[float, float, float]:
        """
        Update with new (y, x) observation.
        Returns (spread, beta, alpha).
        """
        H = np.array([1.0, x])
        # Predict
        P_pred = self._P + self._Q
        # Innovation
        y_pred = H @ self._x
        innov = y - y_pred
        S = H @ P_pred @ H + self.R_obs
        K = P_pred @ H / S
        # Update
        self._x = self._x + K * innov
        self._P = (np.eye(2) - np.outer(K, H)) @ P_pred

        alpha, beta = self._x
        spread = y - beta * x - alpha
        return float(spread), float(beta), float(alpha)

    def fit(self, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit on series. Returns (spreads, betas, alphas)."""
        T = min(len(y), len(x))
        spreads, betas, alphas = np.zeros(T), np.zeros(T), np.zeros(T)
        for t in range(T):
            s, b, a = self.update(float(y[t]), float(x[t]))
            spreads[t], betas[t], alphas[t] = s, b, a
        return spreads, betas, alphas


# ── Dynamic beta estimation ────────────────────────────────────────────────────

class DynamicBeta:
    """
    Time-varying CAPM beta via Kalman filter.
    beta_t+1 = beta_t + w_t,  r_t = alpha + beta_t * r_m_t + eps_t
    """

    def __init__(self, sigma_beta: float = 0.01, sigma_eps: float = 0.02):
        self._beta = 1.0
        self._P = 1.0
        self._sigma_beta = sigma_beta
        self._sigma_eps = sigma_eps

    def update(self, r_asset: float, r_market: float) -> float:
        """Update beta estimate. Returns current beta."""
        P_pred = self._P + self._sigma_beta ** 2
        H = r_market
        S = H ** 2 * P_pred + self._sigma_eps ** 2
        K = P_pred * H / S
        innov = r_asset - self._beta * H
        self._beta += K * innov
        self._P = (1 - K * H) * P_pred
        return float(self._beta)

    def fit(self, returns_asset: np.ndarray, returns_market: np.ndarray) -> np.ndarray:
        T = min(len(returns_asset), len(returns_market))
        betas = np.zeros(T)
        for t in range(T):
            betas[t] = self.update(float(returns_asset[t]), float(returns_market[t]))
        return betas


# ── Extended Kalman Filter ────────────────────────────────────────────────────

class ExtendedKalmanFilter:
    """
    EKF for nonlinear state-space models.
    x_t = f(x_{t-1}) + w_t
    z_t = h(x_t) + v_t
    """

    def __init__(
        self,
        f: Callable,               # state transition function
        h: Callable,               # observation function
        F_jac: Callable,           # Jacobian of f
        H_jac: Callable,           # Jacobian of h
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ):
        self.f = f
        self.h = h
        self.F_jac = F_jac
        self.H_jac = H_jac
        self.Q = Q
        self.R = R
        self.state = KalmanState(x0.copy(), P0.copy())

    def predict(self) -> KalmanState:
        x_pred = self.f(self.state.x)
        F = self.F_jac(self.state.x)
        P_pred = F @ self.state.P @ F.T + self.Q
        return KalmanState(x_pred, P_pred)

    def update(self, z: np.ndarray) -> KalmanState:
        pred = self.predict()
        H = self.H_jac(pred.x)
        y = z - self.h(pred.x)
        S = H @ pred.P @ H.T + self.R
        K = pred.P @ H.T @ np.linalg.inv(S)
        x_upd = pred.x + K @ y
        P_upd = (np.eye(len(pred.x)) - K @ H) @ pred.P
        self.state = KalmanState(x_upd, P_upd)
        return self.state


# ── Unscented Kalman Filter ───────────────────────────────────────────────────

class UnscentedKalmanFilter:
    """
    UKF via scaled sigma-point transform.
    Better than EKF for highly nonlinear systems.
    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.state = KalmanState(x0.copy(), P0.copy())
        n = len(x0)
        self.n = n
        self.lam = alpha ** 2 * (n + kappa) - n
        self.Wm = np.full(2 * n + 1, 0.5 / (n + self.lam))
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc = self.Wm.copy()
        self.Wc[0] += (1 - alpha ** 2 + beta)

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n = self.n
        try:
            L = np.linalg.cholesky((n + self.lam) * P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky((n + self.lam) * (P + 1e-6 * np.eye(n)))
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + L[:, i]
            sigmas[n + i + 1] = x - L[:, i]
        return sigmas

    def predict(self) -> KalmanState:
        sigmas = self._sigma_points(self.state.x, self.state.P)
        sigmas_f = np.array([self.f(s) for s in sigmas])
        x_pred = np.sum(self.Wm[:, None] * sigmas_f, axis=0)
        P_pred = sum(
            self.Wc[i] * np.outer(sigmas_f[i] - x_pred, sigmas_f[i] - x_pred)
            for i in range(2 * self.n + 1)
        ) + self.Q
        return KalmanState(x_pred, P_pred)

    def update(self, z: np.ndarray) -> KalmanState:
        pred = self.predict()
        sigmas = self._sigma_points(pred.x, pred.P)
        sigmas_h = np.array([self.h(s) for s in sigmas])
        z_pred = np.sum(self.Wm[:, None] * sigmas_h, axis=0)
        S = sum(
            self.Wc[i] * np.outer(sigmas_h[i] - z_pred, sigmas_h[i] - z_pred)
            for i in range(2 * self.n + 1)
        ) + self.R
        Pxz = sum(
            self.Wc[i] * np.outer(sigmas[i] - pred.x, sigmas_h[i] - z_pred)
            for i in range(2 * self.n + 1)
        )
        K = Pxz @ np.linalg.inv(S)
        x_upd = pred.x + K @ (z - z_pred)
        P_upd = pred.P - K @ S @ K.T
        self.state = KalmanState(x_upd, P_upd)
        return self.state


# ── Particle Filter ───────────────────────────────────────────────────────────

class ParticleFilter:
    """
    Sequential Monte Carlo (SIR particle filter).
    For non-Gaussian, nonlinear state estimation.
    """

    def __init__(
        self,
        f: Callable,                     # transition: x_{t+1} ~ p(x|x_t)
        g: Callable,                     # likelihood: p(z|x)
        n_particles: int = 1000,
        x0_sampler: Optional[Callable] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.f = f
        self.g = g
        self.n_particles = n_particles
        self.rng = rng or np.random.default_rng()
        if x0_sampler is not None:
            self.particles = np.array([x0_sampler() for _ in range(n_particles)])
        else:
            self.particles = self.rng.standard_normal((n_particles, 1))
        self.weights = np.ones(n_particles) / n_particles

    def update(self, z: Any) -> np.ndarray:
        """One step update. Returns posterior mean estimate."""
        # Propagate
        self.particles = np.array([self.f(p, self.rng) for p in self.particles])
        # Weight by likelihood
        lk = np.array([self.g(z, p) for p in self.particles])
        self.weights *= lk
        self.weights += 1e-300
        self.weights /= self.weights.sum()
        # Effective sample size
        ess = 1.0 / np.sum(self.weights ** 2)
        # Resample if ESS < n/2
        if ess < self.n_particles / 2:
            indices = self.rng.choice(self.n_particles, self.n_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
        return np.average(self.particles, axis=0, weights=self.weights)
