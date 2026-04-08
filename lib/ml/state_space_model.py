"""
State space models for sequential financial data.

Numpy-only implementation covering:
- Linear Gaussian SSM (Kalman filter / smoother)
- Deep State Space: neural-net-parameterized transitions/emissions
- Structured State Space (S4-inspired): diagonal + low-rank, HiPPO init
- Discretization (ZOH, bilinear), parallel scan
- Multi-variate SSM for portfolio of assets
- Online streaming inference, model comparison (BIC/AIC)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict

# ---------------------------------------------------------------------------
# Kalman Filter & Smoother
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Linear Gaussian state-space model:
        x_t = A x_{t-1} + B u_t + w_t,   w_t ~ N(0, Q)
        y_t = C x_t + D u_t + v_t,        v_t ~ N(0, R)
    """

    def __init__(self, state_dim: int, obs_dim: int, control_dim: int = 0):
        self.n = state_dim
        self.m = obs_dim
        self.k = control_dim
        self.A = np.eye(state_dim)
        self.B = np.zeros((state_dim, max(control_dim, 1)))
        self.C = np.eye(obs_dim, state_dim)
        self.D = np.zeros((obs_dim, max(control_dim, 1)))
        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(obs_dim) * 0.1
        self.x0 = np.zeros(state_dim)
        self.P0 = np.eye(state_dim)

    def filter(self, observations: np.ndarray,
               controls: Optional[np.ndarray] = None
               ) -> Dict[str, List[np.ndarray]]:
        """
        Run Kalman filter on a sequence of observations.

        observations: (T, obs_dim)
        controls: (T, control_dim) or None
        Returns dict with filtered means, covariances, predictions, log-likelihoods.
        """
        T = observations.shape[0]
        x = self.x0.copy()
        P = self.P0.copy()

        filtered_means = []
        filtered_covs = []
        predicted_means = []
        predicted_covs = []
        log_likelihoods = []

        for t in range(T):
            # Predict
            u = controls[t] if controls is not None else np.zeros(max(self.k, 1))
            x_pred = self.A @ x + self.B @ u
            P_pred = self.A @ P @ self.A.T + self.Q

            predicted_means.append(x_pred.copy())
            predicted_covs.append(P_pred.copy())

            # Innovation
            y = observations[t]
            y_pred = self.C @ x_pred + self.D @ u
            S = self.C @ P_pred @ self.C.T + self.R
            S_inv = np.linalg.inv(S)
            innovation = y - y_pred

            # Log-likelihood contribution
            sign, logdet = np.linalg.slogdet(S)
            ll = -0.5 * (self.m * np.log(2 * np.pi) + logdet +
                         innovation @ S_inv @ innovation)
            log_likelihoods.append(float(ll))

            # Kalman gain
            K = P_pred @ self.C.T @ S_inv

            # Update
            x = x_pred + K @ innovation
            P = (np.eye(self.n) - K @ self.C) @ P_pred

            filtered_means.append(x.copy())
            filtered_covs.append(P.copy())

        return {
            "filtered_means": filtered_means,
            "filtered_covs": filtered_covs,
            "predicted_means": predicted_means,
            "predicted_covs": predicted_covs,
            "log_likelihoods": log_likelihoods,
        }

    def smooth(self, observations: np.ndarray,
               controls: Optional[np.ndarray] = None
               ) -> Dict[str, List[np.ndarray]]:
        """Rauch-Tung-Striebel smoother."""
        filt = self.filter(observations, controls)
        T = len(filt["filtered_means"])

        smoothed_means = [None] * T
        smoothed_covs = [None] * T
        smoothed_means[T - 1] = filt["filtered_means"][T - 1]
        smoothed_covs[T - 1] = filt["filtered_covs"][T - 1]

        for t in range(T - 2, -1, -1):
            P_filt = filt["filtered_covs"][t]
            P_pred = filt["predicted_covs"][t + 1]
            P_pred_inv = np.linalg.inv(P_pred)
            G = P_filt @ self.A.T @ P_pred_inv

            smoothed_means[t] = (filt["filtered_means"][t] +
                                 G @ (smoothed_means[t + 1] - filt["predicted_means"][t + 1]))
            smoothed_covs[t] = (P_filt +
                                G @ (smoothed_covs[t + 1] - P_pred) @ G.T)

        return {
            "smoothed_means": smoothed_means,
            "smoothed_covs": smoothed_covs,
            "log_likelihoods": filt["log_likelihoods"],
        }

    def log_likelihood(self, observations: np.ndarray,
                       controls: Optional[np.ndarray] = None) -> float:
        filt = self.filter(observations, controls)
        return float(np.sum(filt["log_likelihoods"]))

    def predict_ahead(self, x: np.ndarray, P: np.ndarray,
                      n_steps: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Multi-step ahead prediction from a given state."""
        means = []
        covs = []
        for _ in range(n_steps):
            x = self.A @ x
            P = self.A @ P @ self.A.T + self.Q
            means.append(self.C @ x)
            covs.append(self.C @ P @ self.C.T + self.R)
        return means, covs

# ---------------------------------------------------------------------------
# EM for Kalman parameter learning
# ---------------------------------------------------------------------------

class KalmanEM:
    """Learn KF parameters via Expectation-Maximization."""

    def __init__(self, state_dim: int, obs_dim: int, seed: int = 42):
        self.kf = KalmanFilter(state_dim, obs_dim)
        self.rng = np.random.RandomState(seed)
        self.history: List[float] = []

    def fit(self, observations: np.ndarray, n_iter: int = 50,
            verbose: bool = False) -> List[float]:
        T = observations.shape[0]
        n = self.kf.n
        m = self.kf.m
        self.history = []

        for it in range(n_iter):
            # E-step
            result = self.kf.smooth(observations)
            sm = result["smoothed_means"]
            sP = result["smoothed_covs"]
            ll = sum(result["log_likelihoods"])
            self.history.append(ll)

            if verbose and it % max(1, n_iter // 10) == 0:
                print(f"EM iter {it:3d} | LL={ll:.4f}")

            # Sufficient statistics
            S_xx = np.zeros((n, n))
            S_xx_prev = np.zeros((n, n))
            S_cross = np.zeros((n, n))
            S_yx = np.zeros((m, n))
            S_yy = np.zeros((m, m))

            for t in range(T):
                xx = np.outer(sm[t], sm[t]) + sP[t]
                S_yx += np.outer(observations[t], sm[t])
                S_yy += np.outer(observations[t], observations[t])
                S_xx += xx
                if t > 0:
                    S_xx_prev += np.outer(sm[t - 1], sm[t - 1]) + sP[t - 1]
                    # Cross term: E[x_t x_{t-1}^T]
                    # Approximate: use smoother gain
                    P_pred = self.kf.A @ sP[t - 1] @ self.kf.A.T + self.kf.Q
                    P_pred_inv = np.linalg.inv(P_pred + 1e-8 * np.eye(n))
                    G = sP[t - 1] @ self.kf.A.T @ P_pred_inv
                    Pcross = G @ sP[t]  # approximate
                    S_cross += np.outer(sm[t], sm[t - 1]) + Pcross.T

            # M-step
            if T > 1:
                S_xx_prev_inv = np.linalg.inv(S_xx_prev + 1e-8 * np.eye(n))
                self.kf.A = S_cross @ S_xx_prev_inv
                self.kf.Q = (S_xx - S_cross @ S_xx_prev_inv @ S_cross.T) / (T - 1)
                self.kf.Q = 0.5 * (self.kf.Q + self.kf.Q.T)
                np.fill_diagonal(self.kf.Q, np.maximum(np.diag(self.kf.Q), 1e-6))

            S_xx_inv = np.linalg.inv(S_xx + 1e-8 * np.eye(n))
            self.kf.C = S_yx @ S_xx_inv
            self.kf.R = (S_yy - S_yx @ S_xx_inv @ S_yx.T) / T
            self.kf.R = 0.5 * (self.kf.R + self.kf.R.T)
            np.fill_diagonal(self.kf.R, np.maximum(np.diag(self.kf.R), 1e-6))

            self.kf.x0 = sm[0]
            self.kf.P0 = sP[0]

        return self.history

# ---------------------------------------------------------------------------
# Deep State Space Model
# ---------------------------------------------------------------------------

def _mlp_init(dims: List[int], rng: np.random.RandomState) -> List[Tuple[np.ndarray, np.ndarray]]:
    weights = []
    for i in range(len(dims) - 1):
        std = np.sqrt(2.0 / dims[i])
        W = rng.randn(dims[i], dims[i + 1]) * std
        b = np.zeros((1, dims[i + 1]))
        weights.append((W, b))
    return weights


def _mlp_forward(x: np.ndarray, weights: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    h = x
    for i, (W, b) in enumerate(weights):
        h = h @ W + b
        if i < len(weights) - 1:
            h = np.maximum(0, h)  # ReLU
    return h


class DeepStateSpace:
    """
    Deep State Space Model: transition and emission parameterized by MLPs.

        x_t = f_theta(x_{t-1}) + w_t
        y_t = g_phi(x_t) + v_t

    Uses EKF-like filtering with linearization around the MLP output.
    """

    def __init__(self, state_dim: int, obs_dim: int, hidden_dim: int = 32,
                 process_noise: float = 0.01, obs_noise: float = 0.1,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.f_net = _mlp_init([state_dim, hidden_dim, hidden_dim, state_dim], rng)
        self.g_net = _mlp_init([state_dim, hidden_dim, obs_dim], rng)
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(obs_dim) * obs_noise
        self.x0 = np.zeros(state_dim)
        self.P0 = np.eye(state_dim)

    def _numerical_jacobian(self, func, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute Jacobian of func at x via finite differences."""
        y0 = func(x.reshape(1, -1)).ravel()
        n_in = x.shape[0]
        n_out = y0.shape[0]
        J = np.zeros((n_out, n_in))
        for i in range(n_in):
            x_plus = x.copy()
            x_plus[i] += eps
            y_plus = func(x_plus.reshape(1, -1)).ravel()
            J[:, i] = (y_plus - y0) / eps
        return J

    def transition(self, x: np.ndarray) -> np.ndarray:
        return _mlp_forward(x.reshape(1, -1), self.f_net).ravel()

    def emission(self, x: np.ndarray) -> np.ndarray:
        return _mlp_forward(x.reshape(1, -1), self.g_net).ravel()

    def filter(self, observations: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """Extended Kalman filter with MLP transition/emission."""
        T = observations.shape[0]
        x = self.x0.copy()
        P = self.P0.copy()
        n = self.state_dim

        filtered_means = []
        filtered_covs = []
        log_likelihoods = []

        for t in range(T):
            # Predict
            x_pred = self.transition(x)
            F = self._numerical_jacobian(
                lambda z: _mlp_forward(z, self.f_net), x)
            P_pred = F @ P @ F.T + self.Q

            # Innovation
            y = observations[t]
            y_pred = self.emission(x_pred)
            H = self._numerical_jacobian(
                lambda z: _mlp_forward(z, self.g_net), x_pred)
            S = H @ P_pred @ H.T + self.R
            S_inv = np.linalg.inv(S + 1e-8 * np.eye(self.obs_dim))
            innovation = y - y_pred

            sign, logdet = np.linalg.slogdet(S)
            ll = -0.5 * (self.obs_dim * np.log(2 * np.pi) + logdet +
                         innovation @ S_inv @ innovation)
            log_likelihoods.append(float(ll))

            K = P_pred @ H.T @ S_inv
            x = x_pred + K @ innovation
            P = (np.eye(n) - K @ H) @ P_pred

            filtered_means.append(x.copy())
            filtered_covs.append(P.copy())

        return {
            "filtered_means": filtered_means,
            "filtered_covs": filtered_covs,
            "log_likelihoods": log_likelihoods,
        }

# ---------------------------------------------------------------------------
# HiPPO matrix initialization
# ---------------------------------------------------------------------------

def hippo_legs_matrix(N: int) -> np.ndarray:
    """
    HiPPO-LegS matrix for long-range dependencies.
    A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2} if n > k, else -(n+1) if n==k.
    """
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(n + 1):
            if n == k:
                A[n, k] = -(n + 1)
            else:
                A[n, k] = -np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
    return A


def hippo_b_vector(N: int) -> np.ndarray:
    """HiPPO B vector."""
    B = np.zeros((N, 1))
    for n in range(N):
        B[n, 0] = np.sqrt(2 * n + 1)
    return B

# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

def discretize_zoh(A: np.ndarray, B: np.ndarray,
                   dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-order hold discretization: A_d = exp(A*dt), B_d = A^{-1}(A_d - I)B."""
    N = A.shape[0]
    # Matrix exponential via Pade approximation (truncated)
    Ad = np.eye(N)
    Ak = np.eye(N)
    for k in range(1, 20):
        Ak = Ak @ (A * dt) / k
        Ad = Ad + Ak
    # B_d = A^{-1}(Ad - I)B, but A may be singular; use series
    Bd = np.zeros_like(B)
    Ak = np.eye(N) * dt
    for k in range(1, 20):
        Bd = Bd + Ak @ B / k
        Ak = Ak @ (A * dt) / (k + 1)
    return Ad, Bd


def discretize_bilinear(A: np.ndarray, B: np.ndarray,
                        dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear (Tustin) discretization."""
    N = A.shape[0]
    I = np.eye(N)
    Ah = A * dt / 2
    inv = np.linalg.inv(I - Ah)
    Ad = inv @ (I + Ah)
    Bd = inv @ B * dt
    return Ad, Bd

# ---------------------------------------------------------------------------
# Structured State Space (S4-inspired)
# ---------------------------------------------------------------------------

class StructuredSSM:
    """
    S4-inspired structured state space model.

    Uses diagonal + low-rank parameterization:
        A = diag(lambda) + P Q^T   (low rank correction)

    With HiPPO initialization and choice of discretization.
    """

    def __init__(self, state_dim: int, input_dim: int = 1, output_dim: int = 1,
                 rank: int = 1, dt: float = 1.0,
                 discretization: str = "bilinear", seed: int = 42):
        rng = np.random.RandomState(seed)
        self.N = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.dt = dt
        self.disc_method = discretization

        # Initialize from HiPPO
        A_hippo = hippo_legs_matrix(state_dim)
        B_hippo = hippo_b_vector(state_dim)

        # Diagonal part: eigenvalues of HiPPO (real parts)
        eigvals = np.linalg.eigvals(A_hippo).real
        self.Lambda = np.sort(eigvals)  # (N,)

        # Low-rank correction
        self.P = rng.randn(state_dim, rank) * 0.01
        self.Q = rng.randn(state_dim, rank) * 0.01

        # Input/output projections
        self.B = B_hippo[:, :1] if input_dim == 1 else rng.randn(state_dim, input_dim) * 0.1
        self.C = rng.randn(output_dim, state_dim) * 0.1
        self.D = np.zeros((output_dim, input_dim))

        self._discretize()

    def _get_A(self) -> np.ndarray:
        return np.diag(self.Lambda) + self.P @ self.Q.T

    def _discretize(self):
        A = self._get_A()
        if self.disc_method == "zoh":
            self.Ad, self.Bd = discretize_zoh(A, self.B, self.dt)
        else:
            self.Ad, self.Bd = discretize_bilinear(A, self.B, self.dt)

    def forward_scan(self, u: np.ndarray) -> np.ndarray:
        """
        Sequential scan: process input sequence u of shape (T, input_dim).
        Returns output y of shape (T, output_dim).
        """
        T = u.shape[0]
        x = np.zeros(self.N)
        ys = []
        for t in range(T):
            x = self.Ad @ x + self.Bd @ u[t]
            y = self.C @ x + self.D @ u[t]
            ys.append(y)
        return np.array(ys)

    def parallel_scan(self, u: np.ndarray) -> np.ndarray:
        """
        Parallel scan for efficient sequence processing.

        Uses the associative scan for linear recurrences:
        x_t = A x_{t-1} + B u_t  =>  (A_t, b_t) combined associatively.
        """
        T = u.shape[0]
        N = self.N
        # Initialize (A_i, b_i) tuples
        As = np.tile(self.Ad, (T, 1, 1))  # (T, N, N)
        bs = np.zeros((T, N))
        for t in range(T):
            bs[t] = self.Bd @ u[t]

        # Parallel prefix (associative scan) - iterative doubling
        # (A2, b2) o (A1, b1) = (A2 @ A1, A2 @ b1 + b2)
        stride = 1
        # We need to store intermediate results
        A_scan = As.copy()
        b_scan = bs.copy()
        while stride < T:
            new_A = A_scan.copy()
            new_b = b_scan.copy()
            for t in range(stride, T):
                new_A[t] = A_scan[t] @ A_scan[t - stride]
                new_b[t] = A_scan[t] @ b_scan[t - stride] + b_scan[t]
            A_scan = new_A
            b_scan = new_b
            stride *= 2

        # b_scan[t] now contains x_t (state at time t, assuming x_0 = 0)
        xs = b_scan  # (T, N)
        ys = xs @ self.C.T + u @ self.D.T
        return ys

    def convolutional_mode(self, u: np.ndarray) -> np.ndarray:
        """
        Compute output via convolution with the SSM kernel.
        Kernel K_t = C A^t B.
        """
        T = u.shape[0]
        # Precompute kernel
        kernel = np.zeros((T, self.output_dim, self.input_dim))
        At = np.eye(self.N)
        for t in range(T):
            kernel[t] = self.C @ At @ self.Bd
            At = self.Ad @ At

        # Convolve
        ys = np.zeros((T, self.output_dim))
        for t in range(T):
            for s in range(t + 1):
                ys[t] += kernel[t - s] @ u[s]
            ys[t] += self.D @ u[t]
        return ys

# ---------------------------------------------------------------------------
# Multi-variate SSM for portfolio of assets
# ---------------------------------------------------------------------------

class MultivariateSSM:
    """
    Multi-variate state-space model for a portfolio of assets.

    Each asset has its own latent state, plus shared factors.
    """

    def __init__(self, n_assets: int, state_per_asset: int = 2,
                 n_shared: int = 2, seed: int = 42):
        self.n_assets = n_assets
        self.state_per_asset = state_per_asset
        self.n_shared = n_shared
        total_state = n_assets * state_per_asset + n_shared
        self.kf = KalmanFilter(total_state, n_assets)
        rng = np.random.RandomState(seed)

        # Block-diagonal A for individual states + shared factor dynamics
        A = np.zeros((total_state, total_state))
        for i in range(n_assets):
            s = i * state_per_asset
            A[s:s + state_per_asset, s:s + state_per_asset] = (
                np.eye(state_per_asset) * 0.95 + rng.randn(state_per_asset, state_per_asset) * 0.02)
        sf = n_assets * state_per_asset
        A[sf:, sf:] = np.eye(n_shared) * 0.98
        self.kf.A = A

        # C: each asset observes its own state + shared factors
        C = np.zeros((n_assets, total_state))
        for i in range(n_assets):
            s = i * state_per_asset
            C[i, s] = 1.0
            C[i, sf:] = rng.randn(n_shared) * 0.3
        self.kf.C = C

        self.kf.Q = np.eye(total_state) * 0.01
        self.kf.R = np.eye(n_assets) * 0.05

    def filter(self, returns: np.ndarray) -> Dict[str, List[np.ndarray]]:
        return self.kf.filter(returns)

    def smooth(self, returns: np.ndarray) -> Dict[str, List[np.ndarray]]:
        return self.kf.smooth(returns)

    def extract_shared_factors(self, returns: np.ndarray) -> np.ndarray:
        result = self.smooth(returns)
        sf = self.n_assets * self.state_per_asset
        factors = np.array([m[sf:] for m in result["smoothed_means"]])
        return factors

    def extract_asset_states(self, returns: np.ndarray) -> np.ndarray:
        result = self.smooth(returns)
        states = np.array(result["smoothed_means"])
        n = self.n_assets * self.state_per_asset
        return states[:, :n].reshape(-1, self.n_assets, self.state_per_asset)

# ---------------------------------------------------------------------------
# Volatility state tracking
# ---------------------------------------------------------------------------

class VolatilityStateTracker:
    """
    Track latent volatility state via a log-volatility SSM.

    log_vol_t = a * log_vol_{t-1} + w_t
    r_t ~ N(0, exp(log_vol_t))

    Uses a KF on squared returns as a proxy.
    """

    def __init__(self, persistence: float = 0.98, vol_of_vol: float = 0.1,
                 obs_noise: float = 0.5):
        self.kf = KalmanFilter(1, 1)
        self.kf.A = np.array([[persistence]])
        self.kf.C = np.array([[1.0]])
        self.kf.Q = np.array([[vol_of_vol ** 2]])
        self.kf.R = np.array([[obs_noise ** 2]])
        self.kf.x0 = np.array([0.0])
        self.kf.P0 = np.array([[1.0]])

    def fit(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate latent log-volatility from return series."""
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        # Transform: use log(r^2 + eps) as observation proxy
        log_sq = np.log(returns ** 2 + 1e-8)
        result = self.kf.smooth(log_sq)
        log_vol = np.array([m[0] for m in result["smoothed_means"]])
        vol = np.exp(log_vol / 2)
        return {
            "log_volatility": log_vol,
            "volatility": vol,
            "smoothed_means": result["smoothed_means"],
            "smoothed_covs": result["smoothed_covs"],
        }

# ---------------------------------------------------------------------------
# Regime-switching SSM
# ---------------------------------------------------------------------------

class RegimeSwitchingSSM:
    """
    SSM with discrete regime switching.

    Maintains a KF per regime and uses a transition matrix to
    weight regime probabilities.
    """

    def __init__(self, n_regimes: int, state_dim: int, obs_dim: int,
                 seed: int = 42):
        self.n_regimes = n_regimes
        rng = np.random.RandomState(seed)
        self.kfs = [KalmanFilter(state_dim, obs_dim) for _ in range(n_regimes)]
        # Transition matrix (n_regimes x n_regimes)
        Z = rng.rand(n_regimes, n_regimes) + np.eye(n_regimes) * 5
        self.transition = Z / Z.sum(axis=1, keepdims=True)
        self.initial_probs = np.ones(n_regimes) / n_regimes

    def filter(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        IMM-style (Interacting Multiple Model) approximate filter.
        Returns regime probabilities and blended state estimates.
        """
        T = observations.shape[0]
        K = self.n_regimes
        probs = np.zeros((T, K))
        regime_lls = np.zeros(K)

        # Initialize regime states
        states = [(kf.x0.copy(), kf.P0.copy()) for kf in self.kfs]
        pi = self.initial_probs.copy()

        blended_means = []

        for t in range(T):
            # Predict regime probs
            pi_pred = self.transition.T @ pi

            new_states = []
            lls = np.zeros(K)
            for k in range(K):
                kf = self.kfs[k]
                x, P = states[k]
                # Predict
                x_pred = kf.A @ x
                P_pred = kf.A @ P @ kf.A.T + kf.Q
                # Innovation
                y_pred = kf.C @ x_pred
                S = kf.C @ P_pred @ kf.C.T + kf.R
                S_inv = np.linalg.inv(S + 1e-8 * np.eye(kf.m))
                inn = observations[t] - y_pred
                sign, logdet = np.linalg.slogdet(S)
                lls[k] = -0.5 * (kf.m * np.log(2 * np.pi) + logdet +
                                  inn @ S_inv @ inn)
                K_gain = P_pred @ kf.C.T @ S_inv
                x_new = x_pred + K_gain @ inn
                P_new = (np.eye(kf.n) - K_gain @ kf.C) @ P_pred
                new_states.append((x_new, P_new))

            # Update regime probs
            log_joint = np.log(pi_pred + 1e-12) + lls
            log_joint -= np.max(log_joint)  # numerical stability
            pi = np.exp(log_joint)
            pi /= pi.sum()
            probs[t] = pi
            states = new_states

            # Blended mean
            blend = sum(pi[k] * states[k][0] for k in range(K))
            blended_means.append(blend)

        return {
            "regime_probs": probs,
            "blended_means": np.array(blended_means),
        }

    def most_likely_regime(self, observations: np.ndarray) -> np.ndarray:
        result = self.filter(observations)
        return np.argmax(result["regime_probs"], axis=1)

# ---------------------------------------------------------------------------
# Online streaming inference
# ---------------------------------------------------------------------------

class StreamingKalman:
    """Online (streaming) Kalman filter for real-time state updates."""

    def __init__(self, kf: KalmanFilter):
        self.kf = kf
        self.x = kf.x0.copy()
        self.P = kf.P0.copy()
        self.t = 0
        self.history_x: List[np.ndarray] = []
        self.history_P: List[np.ndarray] = []

    def update(self, observation: np.ndarray,
               control: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single observation and return updated (mean, cov)."""
        kf = self.kf
        u = control if control is not None else np.zeros(max(kf.k, 1))
        x_pred = kf.A @ self.x + kf.B @ u
        P_pred = kf.A @ self.P @ kf.A.T + kf.Q
        y_pred = kf.C @ x_pred + kf.D @ u
        S = kf.C @ P_pred @ kf.C.T + kf.R
        S_inv = np.linalg.inv(S)
        K = P_pred @ kf.C.T @ S_inv
        inn = observation - y_pred
        self.x = x_pred + K @ inn
        self.P = (np.eye(kf.n) - K @ kf.C) @ P_pred
        self.t += 1
        self.history_x.append(self.x.copy())
        self.history_P.append(self.P.copy())
        return self.x.copy(), self.P.copy()

    def predict_next(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next observation without updating."""
        x_pred = self.kf.A @ self.x
        P_pred = self.kf.A @ self.P @ self.kf.A.T + self.kf.Q
        y_pred = self.kf.C @ x_pred
        y_cov = self.kf.C @ P_pred @ self.kf.C.T + self.kf.R
        return y_pred, y_cov

    def reset(self):
        self.x = self.kf.x0.copy()
        self.P = self.kf.P0.copy()
        self.t = 0
        self.history_x = []
        self.history_P = []

# ---------------------------------------------------------------------------
# Model comparison: AIC / BIC
# ---------------------------------------------------------------------------

def count_ssm_params(state_dim: int, obs_dim: int,
                     control_dim: int = 0) -> int:
    """Count free parameters in a linear Gaussian SSM."""
    n, m, k = state_dim, obs_dim, control_dim
    n_A = n * n
    n_C = m * n
    n_Q = n * (n + 1) // 2  # symmetric
    n_R = m * (m + 1) // 2
    n_B = n * k
    n_D = m * k
    n_init = n + n * (n + 1) // 2
    return n_A + n_C + n_Q + n_R + n_B + n_D + n_init


def aic(log_lik: float, n_params: int) -> float:
    return -2 * log_lik + 2 * n_params


def bic(log_lik: float, n_params: int, n_obs: int) -> float:
    return -2 * log_lik + n_params * np.log(n_obs)


def compare_state_dimensions(observations: np.ndarray,
                             state_dims: List[int],
                             n_em_iter: int = 30) -> Dict[int, Dict[str, float]]:
    """
    Compare SSMs with different state dimensions via AIC/BIC.
    """
    obs_dim = observations.shape[1] if observations.ndim > 1 else 1
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    T = observations.shape[0]
    results = {}
    for sd in state_dims:
        em = KalmanEM(sd, obs_dim)
        history = em.fit(observations, n_iter=n_em_iter)
        ll = em.kf.log_likelihood(observations)
        np_ = count_ssm_params(sd, obs_dim)
        results[sd] = {
            "log_likelihood": ll,
            "n_params": np_,
            "aic": aic(ll, np_),
            "bic": bic(ll, np_, T),
            "final_em_ll": history[-1] if history else float("-inf"),
        }
    return results


def select_best_model(comparison: Dict[int, Dict[str, float]],
                      criterion: str = "bic") -> int:
    """Select state dimension with lowest AIC or BIC."""
    best_dim = min(comparison, key=lambda d: comparison[d][criterion])
    return best_dim
