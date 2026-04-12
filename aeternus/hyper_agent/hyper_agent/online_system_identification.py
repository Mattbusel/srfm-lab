"""
online_system_identification.py — Online System Identification (OSI) module.

Monitors live market residuals and continuously updates the agent's internal
environment model (IEM) to track changing market dynamics.

Components:
- Residual monitor: predicted vs actual price impact / fill rate / spread
- Kalman filter for online volatility regime estimation (HMM + EM)
- Recursive Least Squares (RLS) for online Kyle lambda estimation
- Granger causality update: detect cross-asset lead-lag structural breaks
- Agent IEM: 2-layer MLP (obs, action) → next_obs
- Online IEM update: mini-batch update when residual > threshold
- Regime change alert: publish to RTEL GSR when structural break detected
- Latency budget: entire OSI forward pass < 5ms (torch.jit.script)
"""

from __future__ import annotations

import math
import time
import logging
import collections
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OSI_LATENCY_BUDGET_MS = 5.0   # maximum latency for OSI forward pass


# ---------------------------------------------------------------------------
# OSI Configuration
# ---------------------------------------------------------------------------

@dataclass
class ResidualMonitorConfig:
    """Configuration for the market residual monitor."""
    window_size: int = 200
    impact_residual_threshold: float = 0.1     # relative threshold
    fill_rate_residual_threshold: float = 0.15
    spread_residual_threshold: float = 0.2
    ewma_alpha: float = 0.05                   # exponential smoothing
    outlier_clip_sigma: float = 4.0
    min_samples_for_alert: int = 20


@dataclass
class KalmanVolConfig:
    """Kalman filter configuration for online volatility estimation."""
    num_regimes: int = 3
    observation_noise_var: float = 1e-4
    process_noise_var: float = 1e-5
    initial_vol: float = 0.01
    vol_mean_reversion: float = 0.95
    em_update_interval: int = 100
    em_max_iter: int = 5
    min_vol: float = 1e-5
    max_vol: float = 0.5
    smoothing_window: int = 20


@dataclass
class RLSConfig:
    """Recursive Least Squares configuration for Kyle lambda estimation."""
    forgetting_factor: float = 0.98
    initial_covariance: float = 1000.0
    regularization: float = 1e-6
    num_features: int = 3        # [signed_volume, volatility, spread]
    min_lambda: float = 1e-6
    max_lambda: float = 0.01
    update_every_n_steps: int = 1
    outlier_threshold: float = 5.0


@dataclass
class GrangerConfig:
    """Granger causality configuration for cross-asset lead-lag detection."""
    lag_order: int = 5
    significance_level: float = 0.05
    test_interval: int = 500
    min_samples: int = 100
    structural_break_threshold: float = 2.0   # z-score for break detection
    rolling_window: int = 200
    num_assets: int = 4


@dataclass
class IEMConfig:
    """Internal Environment Model configuration."""
    obs_dim: int = 64
    action_dim: int = 8
    hidden_dim: int = 128
    output_dim: int = 64
    learning_rate: float = 3e-4
    batch_size: int = 32
    replay_buffer_size: int = 10_000
    update_every_n_steps: int = 50
    residual_threshold: float = 0.05
    min_samples_for_update: int = 64
    grad_clip: float = 1.0
    weight_decay: float = 1e-5
    dropout: float = 0.1
    use_layer_norm: bool = True
    warm_start_steps: int = 200
    prediction_horizon: int = 1


@dataclass
class OSIConfig:
    """Master OSI configuration."""
    residual_monitor: ResidualMonitorConfig = field(default_factory=ResidualMonitorConfig)
    kalman_vol: KalmanVolConfig = field(default_factory=KalmanVolConfig)
    rls: RLSConfig = field(default_factory=RLSConfig)
    granger: GrangerConfig = field(default_factory=GrangerConfig)
    iem: IEMConfig = field(default_factory=IEMConfig)

    num_assets: int = 4
    seed: Optional[int] = None

    # Alert config
    alert_on_regime_change: bool = True
    alert_cooldown_steps: int = 100
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    # Performance
    use_torch_jit: bool = True
    profile_latency: bool = False

    # Logging
    log_level: str = "WARNING"


# ---------------------------------------------------------------------------
# Residual Monitor
# ---------------------------------------------------------------------------

class ResidualBuffer:
    """Circular buffer with EWMA tracking for market residuals."""

    def __init__(self, window: int, alpha: float = 0.05) -> None:
        self._window = window
        self._alpha = alpha
        self._buffer: collections.deque = collections.deque(maxlen=window)
        self._ewma: float = 0.0
        self._ewma_var: float = 0.0
        self._n: int = 0

    def push(self, value: float) -> None:
        self._buffer.append(value)
        if self._n == 0:
            self._ewma = value
            self._ewma_var = 0.0
        else:
            delta = value - self._ewma
            self._ewma += self._alpha * delta
            self._ewma_var = (1 - self._alpha) * (self._ewma_var + self._alpha * delta ** 2)
        self._n += 1

    @property
    def ewma(self) -> float:
        return self._ewma

    @property
    def ewma_std(self) -> float:
        return math.sqrt(max(self._ewma_var, 0.0))

    @property
    def mean(self) -> float:
        return float(np.mean(list(self._buffer))) if self._buffer else 0.0

    @property
    def std(self) -> float:
        return float(np.std(list(self._buffer))) if len(self._buffer) > 1 else 0.0

    @property
    def n(self) -> int:
        return self._n

    def z_score(self, value: float) -> float:
        s = self.ewma_std
        if s < 1e-10:
            return 0.0
        return (value - self._ewma) / s

    def to_array(self) -> np.ndarray:
        return np.array(list(self._buffer))


class MarketResidualMonitor:
    """
    Monitors residuals between predicted and actual market outcomes.

    Tracks:
    - Price impact residual: predicted_impact - actual_impact
    - Fill rate residual: predicted_fill_rate - actual_fill_rate
    - Spread residual: predicted_spread - actual_spread
    """

    def __init__(self, config: ResidualMonitorConfig, num_assets: int = 4) -> None:
        self.cfg = config
        self.num_assets = num_assets
        self._impact_buffers: List[ResidualBuffer] = [
            ResidualBuffer(config.window_size, config.ewma_alpha) for _ in range(num_assets)
        ]
        self._fill_buffers: List[ResidualBuffer] = [
            ResidualBuffer(config.window_size, config.ewma_alpha) for _ in range(num_assets)
        ]
        self._spread_buffers: List[ResidualBuffer] = [
            ResidualBuffer(config.window_size, config.ewma_alpha) for _ in range(num_assets)
        ]
        self._step = 0

    def reset(self) -> None:
        self.__init__(self.cfg, self.num_assets)

    def update_impact(
        self,
        asset_id: int,
        predicted_impact: float,
        actual_impact: float,
    ) -> float:
        """Record impact residual. Returns normalized residual."""
        if asset_id >= self.num_assets:
            return 0.0
        raw = predicted_impact - actual_impact
        # Clip outliers
        if abs(raw) > self.cfg.outlier_clip_sigma * max(self._impact_buffers[asset_id].ewma_std, 1e-8):
            raw = math.copysign(
                self.cfg.outlier_clip_sigma * max(self._impact_buffers[asset_id].ewma_std, 1e-8),
                raw
            )
        self._impact_buffers[asset_id].push(raw)
        return raw

    def update_fill_rate(
        self,
        asset_id: int,
        predicted_fill: float,
        actual_fill: float,
    ) -> float:
        if asset_id >= self.num_assets:
            return 0.0
        raw = predicted_fill - actual_fill
        self._fill_buffers[asset_id].push(raw)
        return raw

    def update_spread(
        self,
        asset_id: int,
        predicted_spread: float,
        actual_spread: float,
    ) -> float:
        if asset_id >= self.num_assets:
            return 0.0
        raw = (predicted_spread - actual_spread) / max(actual_spread, 1e-8)
        self._spread_buffers[asset_id].push(raw)
        return raw

    def should_alert(self, asset_id: int) -> Tuple[bool, str]:
        """Returns (alert, reason) if residuals exceed thresholds."""
        cfg = self.cfg
        if asset_id >= self.num_assets:
            return False, ""

        if self._impact_buffers[asset_id].n < cfg.min_samples_for_alert:
            return False, ""

        abs_impact = abs(self._impact_buffers[asset_id].ewma)
        if abs_impact > cfg.impact_residual_threshold:
            return True, f"price_impact_drift_{asset_id}: ewma={abs_impact:.4f}"

        abs_fill = abs(self._fill_buffers[asset_id].ewma)
        if abs_fill > cfg.fill_rate_residual_threshold:
            return True, f"fill_rate_drift_{asset_id}: ewma={abs_fill:.4f}"

        abs_spread = abs(self._spread_buffers[asset_id].ewma)
        if abs_spread > cfg.spread_residual_threshold:
            return True, f"spread_drift_{asset_id}: ewma={abs_spread:.4f}"

        return False, ""

    def get_summary(self) -> Dict[str, Any]:
        return {
            f"asset_{i}": {
                "impact_ewma": self._impact_buffers[i].ewma,
                "impact_std": self._impact_buffers[i].ewma_std,
                "fill_ewma": self._fill_buffers[i].ewma,
                "fill_std": self._fill_buffers[i].ewma_std,
                "spread_ewma": self._spread_buffers[i].ewma,
                "spread_std": self._spread_buffers[i].ewma_std,
                "n": self._impact_buffers[i].n,
            }
            for i in range(self.num_assets)
        }


# ---------------------------------------------------------------------------
# Kalman Filter for online volatility + HMM regime
# ---------------------------------------------------------------------------

class KalmanVolatilityFilter:
    """
    Kalman filter for online volatility estimation with HMM regime.

    State: log-volatility
    Observation: |return|
    Regime: K discrete states with EM-updated transition matrix
    """

    def __init__(self, config: KalmanVolConfig) -> None:
        self.cfg = config
        K = config.num_regimes
        self._K = K

        # Per-regime Kalman state: (mean, variance)
        self._means = np.linspace(-5.0, -3.0, K)     # log-vol
        self._variances = np.ones(K) * config.observation_noise_var

        # HMM state
        self._regime_probs = np.ones(K) / K           # posterior over regimes
        self._transition_matrix = (
            np.eye(K) * 0.9 + (1.0 - 0.9) / K * np.ones((K, K))
        )
        self._emission_params = np.exp(self._means)   # emission scale per regime

        # Observation history for EM
        self._obs_history: collections.deque = collections.deque(maxlen=500)
        self._filtered_history: List[np.ndarray] = []
        self._step = 0
        self._last_em_step = 0

        self._smoothed_vol: float = config.initial_vol
        self._vol_history: collections.deque = collections.deque(
            maxlen=config.smoothing_window
        )

    def reset(self) -> None:
        K = self._K
        self._means = np.linspace(-5.0, -3.0, K)
        self._variances = np.ones(K) * self.cfg.observation_noise_var
        self._regime_probs = np.ones(K) / K
        self._obs_history.clear()
        self._filtered_history.clear()
        self._step = 0
        self._smoothed_vol = self.cfg.initial_vol

    def update(self, log_return: float) -> Dict[str, Any]:
        """
        Update filter with a new log-return observation.

        Returns dict with estimated volatility and regime probabilities.
        """
        self._step += 1
        obs = abs(log_return)
        self._obs_history.append(obs)

        cfg = self.cfg
        K = self._K

        # Predict step: propagate regime probs through transition matrix
        predicted_probs = self._transition_matrix.T @ self._regime_probs

        # Kalman predict for each regime's log-vol
        predicted_means = cfg.vol_mean_reversion * self._means
        predicted_vars = (
            cfg.vol_mean_reversion ** 2 * self._variances + cfg.process_noise_var
        )

        # Likelihood of obs under each regime (half-normal / log-normal approx)
        likelihoods = np.zeros(K)
        for k in range(K):
            scale = max(np.exp(predicted_means[k]), cfg.min_vol)
            # Half-normal: p(|r|) = 2/scale * phi(|r|/scale)
            z = obs / max(scale, 1e-10)
            likelihoods[k] = (2.0 / max(scale, 1e-10)) * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

        # Update regime probs
        joint = predicted_probs * likelihoods
        total = joint.sum()
        if total > 1e-300:
            self._regime_probs = joint / total
        else:
            self._regime_probs = np.ones(K) / K

        # Kalman update for each regime
        for k in range(K):
            predicted_obs = np.exp(predicted_means[k])
            innovation = obs - predicted_obs
            S = predicted_vars[k] + cfg.observation_noise_var
            K_gain = predicted_vars[k] / max(S, 1e-10)
            self._means[k] = predicted_means[k] + K_gain * math.log1p(abs(innovation) / max(predicted_obs, 1e-10)) * math.copysign(1.0, innovation)
            self._variances[k] = (1.0 - K_gain) * predicted_vars[k]
            self._means[k] = float(np.clip(self._means[k], math.log(cfg.min_vol), math.log(cfg.max_vol)))

        # Store filtered state
        self._filtered_history.append(self._regime_probs.copy())
        if len(self._filtered_history) > 500:
            self._filtered_history.pop(0)

        # EM update
        if self._step - self._last_em_step >= cfg.em_update_interval:
            self._em_update()
            self._last_em_step = self._step

        # Compute posterior vol estimate
        posterior_vol = float(np.sum(self._regime_probs * np.exp(self._means)))
        posterior_vol = float(np.clip(posterior_vol, cfg.min_vol, cfg.max_vol))

        # Smooth
        self._vol_history.append(posterior_vol)
        self._smoothed_vol = float(np.mean(list(self._vol_history)))

        return {
            "vol_estimate": self._smoothed_vol,
            "regime_probs": self._regime_probs.copy(),
            "dominant_regime": int(np.argmax(self._regime_probs)),
            "regime_entropy": float(-np.sum(
                self._regime_probs * np.log(np.maximum(self._regime_probs, 1e-10))
            )),
        }

    def _em_update(self) -> None:
        """Run EM update on transition matrix using stored filtered states."""
        cfg = self.cfg
        if len(self._filtered_history) < 10:
            return
        K = self._K
        history = np.array(self._filtered_history[-100:])
        T = len(history)
        if T < 2:
            return

        # M-step for transition matrix: count transitions
        new_trans = np.zeros((K, K))
        for t in range(T - 1):
            # Outer product of consecutive regime posteriors
            new_trans += np.outer(history[t], history[t + 1])

        # Normalize rows
        row_sums = new_trans.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        new_trans /= row_sums

        # Smooth with prior
        prior = (np.eye(K) * 0.9 + (1.0 - 0.9) / K * np.ones((K, K)))
        alpha = 0.1
        for _ in range(cfg.em_max_iter):
            self._transition_matrix = (1.0 - alpha) * self._transition_matrix + alpha * new_trans
        # Ensure rows sum to 1
        self._transition_matrix /= self._transition_matrix.sum(axis=1, keepdims=True)

    @property
    def current_vol(self) -> float:
        return self._smoothed_vol

    @property
    def regime_probs(self) -> np.ndarray:
        return self._regime_probs.copy()

    @property
    def dominant_regime(self) -> int:
        return int(np.argmax(self._regime_probs))


# ---------------------------------------------------------------------------
# Recursive Least Squares for Kyle lambda
# ---------------------------------------------------------------------------

class RLSKyleLambda:
    """
    Recursive Least Squares estimator for Kyle's lambda (price impact).

    Regresses: delta_price = lambda * signed_volume + noise
    Extended with additional features: volatility, spread.
    """

    def __init__(self, config: RLSConfig) -> None:
        self.cfg = config
        d = config.num_features
        self._theta = np.zeros(d)           # parameter vector
        self._P = np.eye(d) * config.initial_covariance  # covariance matrix
        self._step = 0
        self._residuals: collections.deque = collections.deque(maxlen=200)

    def reset(self) -> None:
        d = self.cfg.num_features
        self._theta = np.zeros(d)
        self._P = np.eye(d) * self.cfg.initial_covariance
        self._step = 0
        self._residuals.clear()

    def update(
        self,
        features: np.ndarray,
        target: float,
    ) -> Dict[str, Any]:
        """
        Update RLS with a new (features, target) pair.

        features: [signed_volume, volatility, spread, ...]
        target: observed price change
        """
        self._step += 1
        lam = self.cfg.forgetting_factor
        reg = self.cfg.regularization

        # Prediction
        y_pred = float(np.dot(self._theta, features))
        residual = target - y_pred

        # Outlier detection
        res_arr = self._residuals
        if len(res_arr) > 10:
            res_std = float(np.std(list(res_arr)))
            if abs(residual) > self.cfg.outlier_threshold * max(res_std, 1e-10):
                # Skip outlier update
                return {
                    "lambda_estimate": self.kyle_lambda,
                    "residual": residual,
                    "outlier_skipped": True,
                }

        self._residuals.append(residual)

        # RLS gain
        Pf = self._P @ features
        denom = lam + features @ Pf + reg
        K_gain = Pf / denom

        # Update
        self._theta = self._theta + K_gain * residual
        self._P = (self._P - np.outer(K_gain, Pf)) / lam
        # Symmetrize
        self._P = 0.5 * (self._P + self._P.T)

        return {
            "lambda_estimate": self.kyle_lambda,
            "residual": residual,
            "theta": self._theta.copy(),
            "outlier_skipped": False,
        }

    @property
    def kyle_lambda(self) -> float:
        """First coefficient is the Kyle lambda estimate."""
        lam = float(self._theta[0])
        return float(np.clip(lam, self.cfg.min_lambda, self.cfg.max_lambda))

    @property
    def theta(self) -> np.ndarray:
        return self._theta.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def lambda_uncertainty(self) -> float:
        """Standard deviation of lambda estimate from covariance diagonal."""
        return float(math.sqrt(max(self._P[0, 0], 0.0)))

    @property
    def residual_std(self) -> float:
        if len(self._residuals) < 2:
            return 1.0
        return float(np.std(list(self._residuals)))


# ---------------------------------------------------------------------------
# Granger causality lead-lag detector
# ---------------------------------------------------------------------------

class GrangerLeadLagDetector:
    """
    Online Granger causality detector for cross-asset lead-lag structure.

    Uses rolling VAR estimation and monitors for structural breaks.
    """

    def __init__(self, config: GrangerConfig) -> None:
        self.cfg = config
        n = config.num_assets
        p = config.lag_order
        self._n = n
        self._p = p

        # Rolling return history
        self._returns: np.ndarray = np.zeros((config.rolling_window, n))
        self._idx: int = 0
        self._full: bool = False

        # Estimated VAR coefficients: shape (n, n*p)
        self._var_coefs: Optional[np.ndarray] = None
        self._prev_var_coefs: Optional[np.ndarray] = None

        # Break detection
        self._break_detected: bool = False
        self._break_z_scores: np.ndarray = np.zeros((n, n))
        self._break_history: List[Tuple[int, np.ndarray]] = []
        self._step = 0
        self._last_test_step = 0

    def reset(self) -> None:
        self._returns = np.zeros((self.cfg.rolling_window, self._n))
        self._idx = 0
        self._full = False
        self._var_coefs = None
        self._prev_var_coefs = None
        self._break_detected = False
        self._step = 0
        self._last_test_step = 0
        self._break_history.clear()

    def update(self, returns: np.ndarray) -> None:
        """Add a new cross-asset return observation."""
        assert len(returns) == self._n
        self._returns[self._idx % self.cfg.rolling_window] = returns
        self._idx += 1
        if self._idx >= self.cfg.rolling_window:
            self._full = True
        self._step += 1

    def run_test(self) -> Dict[str, Any]:
        """
        Run Granger causality test. Should be called every test_interval steps.

        Returns dict with test results and break detection.
        """
        self._last_test_step = self._step
        cfg = self.cfg
        n = self._n
        p = self._p

        if not self._full and self._idx < cfg.min_samples:
            return {
                "break_detected": False,
                "break_z_scores": np.zeros((n, n)),
                "var_coefs": None,
                "n_samples": self._idx,
            }

        # Get available data
        avail = min(self._idx, cfg.rolling_window)
        if self._full:
            # Reorder circular buffer
            data = np.roll(self._returns, -self._idx % cfg.rolling_window, axis=0)
        else:
            data = self._returns[:self._idx]

        # Estimate VAR(p) via OLS
        T = avail - p
        if T < n * p + 1:
            return {
                "break_detected": False,
                "break_z_scores": np.zeros((n, n)),
                "var_coefs": self._var_coefs,
                "n_samples": avail,
            }

        Y = data[p:].T           # (n, T)
        X_cols = []
        for lag in range(1, p + 1):
            X_cols.append(data[p - lag: p - lag + T].T)
        X = np.vstack(X_cols)    # (n*p, T)

        # OLS: B = Y X' (X X')^-1
        try:
            XX = X @ X.T + np.eye(n * p) * 1e-6
            XY = X @ Y.T
            B = np.linalg.solve(XX, XY).T   # (n, n*p)
        except np.linalg.LinAlgError:
            return {
                "break_detected": False,
                "break_z_scores": np.zeros((n, n)),
                "var_coefs": self._var_coefs,
                "n_samples": avail,
            }

        # Granger causality: F-test for each pair
        granger_matrix = np.zeros((n, n))
        for caused in range(n):
            for causing in range(n):
                if caused == causing:
                    continue
                # Indices of causing variable across all lags
                causing_indices = [causing + lag * n for lag in range(p)]
                b_causing = B[caused, causing_indices]
                granger_matrix[caused, causing] = float(np.sum(b_causing ** 2))

        # Detect structural break vs previous estimate
        break_z_scores = np.zeros((n, n))
        break_detected = False
        if self._prev_var_coefs is not None:
            delta = B - self._prev_var_coefs
            norm_delta = np.linalg.norm(delta)
            prev_norm = np.linalg.norm(self._prev_var_coefs)
            if prev_norm > 1e-10:
                relative_change = norm_delta / prev_norm
                if relative_change > cfg.structural_break_threshold * 0.1:
                    break_detected = True
                    self._break_history.append((self._step, B.copy()))

        self._prev_var_coefs = self._var_coefs
        self._var_coefs = B
        self._break_detected = break_detected
        self._break_z_scores = break_z_scores

        return {
            "break_detected": break_detected,
            "break_z_scores": break_z_scores,
            "var_coefs": B.copy(),
            "granger_matrix": granger_matrix,
            "n_samples": avail,
        }

    def should_run_test(self) -> bool:
        return self._step - self._last_test_step >= self.cfg.test_interval

    @property
    def break_detected(self) -> bool:
        return self._break_detected

    @property
    def var_coefs(self) -> Optional[np.ndarray]:
        return self._var_coefs.copy() if self._var_coefs is not None else None

    @property
    def break_history(self) -> List[Tuple[int, np.ndarray]]:
        return self._break_history[-10:]


# ---------------------------------------------------------------------------
# Agent Internal Environment Model (IEM)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class IEMNetwork(nn.Module):
        """
        2-layer MLP: (obs, action) → predicted_next_obs.

        Scripted for low-latency inference.
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
        ) -> None:
            super().__init__()
            in_dim = obs_dim + action_dim
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.ln1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.ln2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self._use_ln = use_layer_norm

        def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            x = torch.cat([obs, action], dim=-1)
            x = F.relu(self.ln1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.ln2(self.fc2(x)))
            x = self.dropout(x)
            return self.out(x)

    class IEMNetworkScripted(nn.Module):
        """torch.jit.script-compatible version (no conditional modules)."""

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int,
            output_dim: int,
        ) -> None:
            super().__init__()
            in_dim = obs_dim + action_dim
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, output_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

        def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            x = torch.cat([obs, action], dim=-1)
            x = F.relu(self.ln1(self.fc1(x)))
            x = F.relu(self.ln2(self.fc2(x)))
            return self.out(x)


class AgentInternalEnvironmentModel:
    """
    Online-updatable Internal Environment Model (IEM).

    Maps (obs, action) → next_obs. Trained online from experience.
    Triggers mini-batch updates when residual exceeds threshold.
    """

    def __init__(self, config: IEMConfig, device: str = "cpu") -> None:
        self.cfg = config
        self.device = device
        self._step = 0
        self._trained = False

        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available; IEM will be disabled.")
            self._model = None
            self._optimizer = None
            return

        # Build model
        self._model_base = IEMNetwork(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm,
        ).to(device)

        # Try to create scripted model for fast inference
        try:
            scripted = IEMNetworkScripted(
                config.obs_dim,
                config.action_dim,
                config.hidden_dim,
                config.output_dim,
            ).to(device)
            self._scripted = torch.jit.script(scripted)
        except Exception as e:
            logger.warning("Could not script IEM: %s", e)
            self._scripted = None

        self._optimizer = optim.Adam(
            self._model_base.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Replay buffer
        self._obs_buf: collections.deque = collections.deque(maxlen=config.replay_buffer_size)
        self._action_buf: collections.deque = collections.deque(maxlen=config.replay_buffer_size)
        self._next_obs_buf: collections.deque = collections.deque(maxlen=config.replay_buffer_size)

        # Metrics
        self._loss_history: collections.deque = collections.deque(maxlen=1000)
        self._residual_history: collections.deque = collections.deque(maxlen=500)
        self._last_update_step = 0

    def reset(self) -> None:
        self._obs_buf.clear()
        self._action_buf.clear()
        self._next_obs_buf.clear()
        self._step = 0
        self._last_update_step = 0

    def predict(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict next_obs given (obs, action). Returns (predicted_next_obs, latency_ms).
        Uses scripted model if available for <5ms budget.
        """
        if not _TORCH_AVAILABLE or self._model_base is None:
            return obs.copy(), 0.0

        t0 = time.perf_counter()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            act_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

            if self._scripted is not None:
                pred = self._scripted(obs_t, act_t)
            else:
                self._model_base.eval()
                pred = self._model_base(obs_t, act_t)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return pred.squeeze(0).cpu().numpy(), latency_ms

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
    ) -> float:
        """
        Push experience to replay buffer.
        Returns residual (|predicted - actual|).
        """
        self._step += 1
        self._obs_buf.append(obs.copy())
        self._action_buf.append(action.copy())
        self._next_obs_buf.append(next_obs.copy())

        if _TORCH_AVAILABLE and self._model_base is not None and self._trained:
            pred, _ = self.predict(obs, action)
            residual = float(np.mean(np.abs(pred - next_obs)))
            self._residual_history.append(residual)
            return residual
        return 0.0

    def should_update(self, residual: float) -> bool:
        """Return True if IEM should be updated."""
        cfg = self.cfg
        if len(self._obs_buf) < cfg.min_samples_for_update:
            return False
        if self._step < cfg.warm_start_steps:
            return False
        if self._step - self._last_update_step < cfg.update_every_n_steps:
            return False
        if residual > cfg.residual_threshold:
            return True
        # Also update periodically regardless
        return (self._step - self._last_update_step) >= cfg.update_every_n_steps * 5

    def update(self, force: bool = False) -> Optional[float]:
        """Run mini-batch update. Returns mean loss or None."""
        if not _TORCH_AVAILABLE or self._model_base is None:
            return None
        cfg = self.cfg
        n = len(self._obs_buf)
        if n < cfg.min_samples_for_update:
            return None

        self._last_update_step = self._step
        self._trained = True

        # Sample mini-batch
        batch_size = min(cfg.batch_size, n)
        indices = np.random.choice(n, size=batch_size, replace=False)

        obs_list = list(self._obs_buf)
        act_list = list(self._action_buf)
        nobs_list = list(self._next_obs_buf)

        obs_batch = np.array([obs_list[i] for i in indices])
        act_batch = np.array([act_list[i] for i in indices])
        nobs_batch = np.array([nobs_list[i] for i in indices])

        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_batch, dtype=torch.float32, device=self.device)
        target_t = torch.tensor(nobs_batch, dtype=torch.float32, device=self.device)

        self._model_base.train()
        self._optimizer.zero_grad()
        pred_t = self._model_base(obs_t, act_t)
        loss = F.mse_loss(pred_t, target_t)
        loss.backward()

        nn.utils.clip_grad_norm_(self._model_base.parameters(), cfg.grad_clip)
        self._optimizer.step()

        loss_val = float(loss.item())
        self._loss_history.append(loss_val)

        # Sync scripted model weights
        if self._scripted is not None:
            try:
                sd = self._model_base.state_dict()
                # Copy weights that match
                scripted_sd = self._scripted.state_dict()
                compatible = {k: v for k, v in sd.items() if k in scripted_sd}
                self._scripted.load_state_dict(compatible, strict=False)
            except Exception:
                pass

        return loss_val

    @property
    def mean_loss(self) -> float:
        if not self._loss_history:
            return 0.0
        return float(np.mean(list(self._loss_history)))

    @property
    def mean_residual(self) -> float:
        if not self._residual_history:
            return 0.0
        return float(np.mean(list(self._residual_history)))

    @property
    def buffer_size(self) -> int:
        return len(self._obs_buf)

    def get_model_state(self) -> Optional[Dict[str, Any]]:
        if not _TORCH_AVAILABLE or self._model_base is None:
            return None
        return self._model_base.state_dict()

    def load_model_state(self, state: Dict[str, Any]) -> None:
        if _TORCH_AVAILABLE and self._model_base is not None:
            self._model_base.load_state_dict(state)


# ---------------------------------------------------------------------------
# Regime change alert publisher
# ---------------------------------------------------------------------------

class RegimeChangeAlert:
    """
    Publishes regime change alerts to downstream consumers (e.g., RTEL GSR).

    Supports callback-based and queue-based alert delivery.
    """

    def __init__(
        self,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cooldown_steps: int = 100,
    ) -> None:
        self._callback = callback
        self._cooldown = cooldown_steps
        self._last_alert_step: Dict[str, int] = {}
        self._alert_history: List[Dict[str, Any]] = []
        self._step = 0
        self._alert_queue: collections.deque = collections.deque(maxlen=1000)
        self._lock = threading.Lock()

    def step(self) -> None:
        self._step += 1

    def publish(self, alert_type: str, payload: Dict[str, Any]) -> bool:
        """
        Publish an alert. Respects cooldown. Returns True if published.
        """
        with self._lock:
            last = self._last_alert_step.get(alert_type, -self._cooldown - 1)
            if self._step - last < self._cooldown:
                return False

            self._last_alert_step[alert_type] = self._step
            full_payload = {
                "alert_type": alert_type,
                "step": self._step,
                **payload,
            }
            self._alert_history.append(full_payload)
            self._alert_queue.append(full_payload)

            if self._callback is not None:
                try:
                    self._callback(alert_type, full_payload)
                except Exception as e:
                    logger.warning("Alert callback raised: %s", e)

            logger.info("OSI Alert [%s]: %s", alert_type, payload)
            return True

    def drain(self) -> List[Dict[str, Any]]:
        """Drain the alert queue and return all pending alerts."""
        with self._lock:
            alerts = list(self._alert_queue)
            self._alert_queue.clear()
            return alerts

    @property
    def alert_history(self) -> List[Dict[str, Any]]:
        return self._alert_history[-50:]

    @property
    def current_step(self) -> int:
        return self._step


# ---------------------------------------------------------------------------
# OSI Latency profiler
# ---------------------------------------------------------------------------

class LatencyProfiler:
    """Profiles latency of OSI components."""

    def __init__(self, window: int = 500) -> None:
        self._timings: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=window)
        )

    def record(self, component: str, duration_ms: float) -> None:
        self._timings[component].append(duration_ms)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for comp, times in self._timings.items():
            arr = list(times)
            if arr:
                stats[comp] = {
                    "mean_ms": float(np.mean(arr)),
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)),
                    "max_ms": float(np.max(arr)),
                    "budget_exceeded_pct": float(
                        100.0 * np.mean(np.array(arr) > OSI_LATENCY_BUDGET_MS)
                    ),
                }
        return stats

    def total_mean_ms(self) -> float:
        total = 0.0
        for times in self._timings.values():
            if times:
                total += float(np.mean(list(times)))
        return total


# ---------------------------------------------------------------------------
# Online System Identification — main class
# ---------------------------------------------------------------------------

class OnlineSystemIdentification:
    """
    Online System Identification (OSI) module for Hyper-Agent.

    Orchestrates:
    - MarketResidualMonitor
    - KalmanVolatilityFilter (per asset)
    - RLSKyleLambda (per asset)
    - GrangerLeadLagDetector
    - AgentInternalEnvironmentModel
    - RegimeChangeAlert
    - LatencyProfiler
    """

    def __init__(
        self,
        config: Optional[OSIConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = config or OSIConfig()
        self.device = device
        n = self.cfg.num_assets

        # Sub-modules
        self.residual_monitor = MarketResidualMonitor(self.cfg.residual_monitor, n)
        self.vol_filters: List[KalmanVolatilityFilter] = [
            KalmanVolatilityFilter(self.cfg.kalman_vol) for _ in range(n)
        ]
        self.rls_estimators: List[RLSKyleLambda] = [
            RLSKyleLambda(self.cfg.rls) for _ in range(n)
        ]
        self.granger = GrangerLeadLagDetector(self.cfg.granger)
        self.iem = AgentInternalEnvironmentModel(self.cfg.iem, device)
        self.alert_publisher = RegimeChangeAlert(
            callback=self.cfg.alert_callback,
            cooldown_steps=self.cfg.alert_cooldown_steps,
        )
        self.profiler = LatencyProfiler() if self.cfg.profile_latency else None

        self._step = 0
        self._vol_estimates: np.ndarray = np.zeros(n)
        self._lambda_estimates: np.ndarray = np.zeros(n)
        self._regime_probs_per_asset: List[np.ndarray] = [
            np.ones(self.cfg.kalman_vol.num_regimes) / self.cfg.kalman_vol.num_regimes
            for _ in range(n)
        ]

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all OSI sub-modules."""
        self.residual_monitor.reset()
        for f in self.vol_filters:
            f.reset()
        for r in self.rls_estimators:
            r.reset()
        self.granger.reset()
        self.iem.reset()
        self._step = 0
        n = self.cfg.num_assets
        self._vol_estimates = np.zeros(n)
        self._lambda_estimates = np.zeros(n)

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(
        self,
        log_returns: np.ndarray,
        signed_volumes: np.ndarray,
        price_changes: np.ndarray,
        predicted_impacts: np.ndarray,
        actual_impacts: np.ndarray,
        predicted_fill_rates: np.ndarray,
        actual_fill_rates: np.ndarray,
        predicted_spreads: np.ndarray,
        actual_spreads: np.ndarray,
        obs: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        next_obs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Full OSI update step.

        Must complete within OSI_LATENCY_BUDGET_MS = 5ms.

        Args:
            log_returns: per-asset log returns (shape: num_assets)
            signed_volumes: signed trade volumes (shape: num_assets)
            price_changes: observed delta prices (shape: num_assets)
            predicted_impacts: model-predicted price impacts
            actual_impacts: observed price impacts
            predicted_fill_rates: model fill rate predictions
            actual_fill_rates: actual fill rates
            predicted_spreads: model spread predictions
            actual_spreads: observed spreads
            obs: current agent observation (for IEM)
            action: agent action (for IEM)
            next_obs: next agent observation (for IEM)

        Returns:
            dict with all OSI outputs
        """
        t_start = time.perf_counter()
        self._step += 1
        self.alert_publisher.step()

        n = self.cfg.num_assets
        result: Dict[str, Any] = {"step": self._step}

        # ---- 1. Residual monitoring ----
        t0 = time.perf_counter()
        impact_residuals = np.zeros(n)
        fill_residuals = np.zeros(n)
        spread_residuals = np.zeros(n)
        for i in range(n):
            impact_residuals[i] = self.residual_monitor.update_impact(
                i, float(predicted_impacts[i]), float(actual_impacts[i])
            )
            fill_residuals[i] = self.residual_monitor.update_fill_rate(
                i, float(predicted_fill_rates[i]), float(actual_fill_rates[i])
            )
            spread_residuals[i] = self.residual_monitor.update_spread(
                i, float(predicted_spreads[i]), float(actual_spreads[i])
            )
            # Check for residual alerts
            alerted, reason = self.residual_monitor.should_alert(i)
            if alerted:
                self.alert_publisher.publish(
                    "residual_drift",
                    {"asset_id": i, "reason": reason},
                )

        if self.profiler:
            self.profiler.record("residual_monitor", (time.perf_counter() - t0) * 1000)

        # ---- 2. Kalman volatility filter ----
        t0 = time.perf_counter()
        vol_results = []
        for i in range(n):
            vr = self.vol_filters[i].update(float(log_returns[i]))
            self._vol_estimates[i] = vr["vol_estimate"]
            self._regime_probs_per_asset[i] = vr["regime_probs"]
            vol_results.append(vr)

            # Alert on regime entropy (uncertainty in regime)
            if vr["regime_entropy"] > 0.9 * math.log(self.cfg.kalman_vol.num_regimes):
                self.alert_publisher.publish(
                    "regime_uncertainty",
                    {"asset_id": i, "entropy": vr["regime_entropy"]},
                )

        if self.profiler:
            self.profiler.record("kalman_vol", (time.perf_counter() - t0) * 1000)

        # ---- 3. RLS Kyle lambda ----
        t0 = time.perf_counter()
        rls_results = []
        for i in range(n):
            features = np.array([
                float(signed_volumes[i]),
                self._vol_estimates[i],
                float(actual_spreads[i]),
            ])
            rr = self.rls_estimators[i].update(features, float(price_changes[i]))
            self._lambda_estimates[i] = self.rls_estimators[i].kyle_lambda
            rls_results.append(rr)

        if self.profiler:
            self.profiler.record("rls_lambda", (time.perf_counter() - t0) * 1000)

        # ---- 4. Granger causality ----
        t0 = time.perf_counter()
        granger_result: Optional[Dict[str, Any]] = None
        self.granger.update(log_returns)
        if self.granger.should_run_test():
            granger_result = self.granger.run_test()
            if granger_result.get("break_detected", False):
                self.alert_publisher.publish(
                    "lead_lag_structural_break",
                    {
                        "n_samples": granger_result.get("n_samples", 0),
                        "step": self._step,
                    },
                )

        if self.profiler:
            self.profiler.record("granger", (time.perf_counter() - t0) * 1000)

        # ---- 5. IEM update ----
        t0 = time.perf_counter()
        iem_loss: Optional[float] = None
        iem_residual = 0.0
        if obs is not None and action is not None and next_obs is not None:
            iem_residual = self.iem.push(obs, action, next_obs)
            if self.iem.should_update(iem_residual):
                iem_loss = self.iem.update()
                if iem_residual > self.cfg.iem.residual_threshold * 2:
                    self.alert_publisher.publish(
                        "iem_high_residual",
                        {
                            "residual": iem_residual,
                            "mean_loss": self.iem.mean_loss,
                        },
                    )

        if self.profiler:
            self.profiler.record("iem", (time.perf_counter() - t0) * 1000)

        # ---- Total latency check ----
        total_ms = (time.perf_counter() - t_start) * 1000.0
        if self.profiler:
            self.profiler.record("total", total_ms)
        if total_ms > OSI_LATENCY_BUDGET_MS:
            logger.debug("OSI latency budget exceeded: %.2f ms", total_ms)

        result.update({
            "vol_estimates": self._vol_estimates.copy(),
            "lambda_estimates": self._lambda_estimates.copy(),
            "regime_probs": [rp.copy() for rp in self._regime_probs_per_asset],
            "dominant_regimes": [vr["dominant_regime"] for vr in vol_results],
            "impact_residuals": impact_residuals,
            "fill_residuals": fill_residuals,
            "spread_residuals": spread_residuals,
            "granger_break": granger_result.get("break_detected", False) if granger_result else False,
            "iem_residual": iem_residual,
            "iem_loss": iem_loss,
            "pending_alerts": self.alert_publisher.drain(),
            "total_latency_ms": total_ms,
        })
        return result

    # ------------------------------------------------------------------
    # IEM inference
    # ------------------------------------------------------------------

    def predict_next_obs(
        self, obs: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Predict next observation using the IEM."""
        return self.iem.predict(obs, action)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def vol_estimates(self) -> np.ndarray:
        return self._vol_estimates.copy()

    @property
    def lambda_estimates(self) -> np.ndarray:
        return self._lambda_estimates.copy()

    @property
    def lambda_uncertainties(self) -> np.ndarray:
        return np.array([r.lambda_uncertainty for r in self.rls_estimators])

    def get_full_state(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "vol_estimates": self._vol_estimates.copy(),
            "lambda_estimates": self._lambda_estimates.copy(),
            "lambda_uncertainties": self.lambda_uncertainties.tolist(),
            "dominant_regimes": [f.dominant_regime for f in self.vol_filters],
            "iem_buffer_size": self.iem.buffer_size,
            "iem_mean_loss": self.iem.mean_loss,
            "iem_mean_residual": self.iem.mean_residual,
            "granger_break_detected": self.granger.break_detected,
            "residual_summary": self.residual_monitor.get_summary(),
            "alert_history": self.alert_publisher.alert_history[-5:],
            "profiler_stats": self.profiler.get_stats() if self.profiler else {},
        }

    def get_observation_features(self) -> np.ndarray:
        """
        Return a compact feature vector summarizing the OSI state.

        Useful for augmenting agent observations.
        """
        n = self.cfg.num_assets
        K = self.cfg.kalman_vol.num_regimes
        features = np.concatenate([
            self._vol_estimates,                        # n
            self._lambda_estimates,                     # n
            self.lambda_uncertainties,                  # n
            *self._regime_probs_per_asset[:n],          # n * K
        ])
        return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Multi-asset OSI
# ---------------------------------------------------------------------------

class MultiAssetOSI:
    """
    Wrapper that manages per-asset OSI and cross-asset correlation.

    Adds cross-asset correlation estimation via DCC-GARCH (simplified).
    """

    def __init__(self, config: Optional[OSIConfig] = None, device: str = "cpu") -> None:
        self.osi = OnlineSystemIdentification(config, device)
        self._cfg = config or OSIConfig()
        n = self._cfg.num_assets

        # DCC correlation state
        self._corr_matrix: np.ndarray = np.eye(n)
        self._dcc_alpha: float = 0.05
        self._dcc_beta: float = 0.90
        self._Q_bar: np.ndarray = np.eye(n)
        self._Q: np.ndarray = np.eye(n)
        self._standardized_returns: List[np.ndarray] = []
        self._n = n

    def reset(self) -> None:
        self.osi.reset()
        self._corr_matrix = np.eye(self._n)
        self._Q = np.eye(self._n)
        self._standardized_returns.clear()

    def update_correlation(self, log_returns: np.ndarray) -> np.ndarray:
        """Update DCC-GARCH correlation matrix. Returns updated correlation."""
        n = self._n
        vols = self.osi.vol_estimates
        # Standardize returns
        std_ret = log_returns / np.maximum(vols, 1e-8)
        self._standardized_returns.append(std_ret.copy())
        if len(self._standardized_returns) > 500:
            self._standardized_returns.pop(0)

        # DCC update
        if len(self._standardized_returns) > 20:
            # Update Q_bar as sample correlation of standardized returns
            data = np.array(self._standardized_returns[-100:])
            self._Q_bar = np.corrcoef(data.T) if data.shape[0] > 2 else np.eye(n)

            # DCC recursion: Qt = (1-a-b)*Q_bar + a*(z_{t-1} z_{t-1}') + b*Q_{t-1}
            z = std_ret
            self._Q = (
                (1 - self._dcc_alpha - self._dcc_beta) * self._Q_bar
                + self._dcc_alpha * np.outer(z, z)
                + self._dcc_beta * self._Q
            )

            # Normalize to correlation
            q_diag = np.sqrt(np.maximum(np.diag(self._Q), 1e-10))
            self._corr_matrix = self._Q / np.outer(q_diag, q_diag)

        return self._corr_matrix.copy()

    def full_update(
        self,
        log_returns: np.ndarray,
        signed_volumes: np.ndarray,
        price_changes: np.ndarray,
        predicted_impacts: np.ndarray,
        actual_impacts: np.ndarray,
        predicted_fill_rates: np.ndarray,
        actual_fill_rates: np.ndarray,
        predicted_spreads: np.ndarray,
        actual_spreads: np.ndarray,
        obs: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        next_obs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        result = self.osi.update(
            log_returns, signed_volumes, price_changes,
            predicted_impacts, actual_impacts,
            predicted_fill_rates, actual_fill_rates,
            predicted_spreads, actual_spreads,
            obs, action, next_obs,
        )
        corr = self.update_correlation(log_returns)
        result["correlation_matrix"] = corr
        return result

    @property
    def correlation_matrix(self) -> np.ndarray:
        return self._corr_matrix.copy()

    def get_observation_features(self) -> np.ndarray:
        base = self.osi.get_observation_features()
        corr_upper = self._corr_matrix[np.triu_indices(self._n, k=1)]
        return np.concatenate([base, corr_upper]).astype(np.float32)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_osi(
    num_assets: int = 4,
    obs_dim: int = 64,
    action_dim: int = 8,
    seed: Optional[int] = None,
    device: str = "cpu",
    use_jit: bool = True,
    alert_callback: Optional[Callable] = None,
) -> OnlineSystemIdentification:
    """Create an OSI instance with sensible defaults."""
    cfg = OSIConfig(
        num_assets=num_assets,
        seed=seed,
        use_torch_jit=use_jit,
        alert_callback=alert_callback,
        iem=IEMConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
        ),
        granger=GrangerConfig(num_assets=num_assets),
    )
    return OnlineSystemIdentification(cfg, device)


def make_multi_asset_osi(
    num_assets: int = 4,
    obs_dim: int = 64,
    action_dim: int = 8,
    device: str = "cpu",
) -> MultiAssetOSI:
    """Create a MultiAssetOSI with DCC correlation."""
    cfg = OSIConfig(
        num_assets=num_assets,
        iem=IEMConfig(obs_dim=obs_dim, action_dim=action_dim),
        granger=GrangerConfig(num_assets=num_assets),
    )
    return MultiAssetOSI(cfg, device)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Configs
    "OSIConfig",
    "ResidualMonitorConfig",
    "KalmanVolConfig",
    "RLSConfig",
    "GrangerConfig",
    "IEMConfig",
    # Sub-modules
    "MarketResidualMonitor",
    "ResidualBuffer",
    "KalmanVolatilityFilter",
    "RLSKyleLambda",
    "GrangerLeadLagDetector",
    "AgentInternalEnvironmentModel",
    "RegimeChangeAlert",
    "LatencyProfiler",
    # Main
    "OnlineSystemIdentification",
    "MultiAssetOSI",
    # Factories
    "make_osi",
    "make_multi_asset_osi",
    # Constants
    "OSI_LATENCY_BUDGET_MS",
    # Extended
    "FillRatePredictor",
    "SpreadPredictor",
    "OSIEnsemble",
    "OSIDiagnostics",
    "OnlineNormalizationTracker",
    "StructuralBreakDetector",
    "OSIPerformanceMonitor",
    "IEMEnsemble",
    "AdaptiveLearningRateScheduler",
    "OSIStateBuffer",
]


# ---------------------------------------------------------------------------
# Extended: FillRatePredictor
# ---------------------------------------------------------------------------

class FillRatePredictor:
    """
    Online predictor for fill rates using recursive exponential smoothing
    with regime-aware correction.

    Tracks:
    - Rolling fill rate by order side and price level
    - Fill rate sensitivity to spread, depth, and volatility
    - Intraday fill rate seasonality
    """

    def __init__(
        self,
        alpha: float = 0.05,
        num_price_levels: int = 5,
        num_regimes: int = 3,
    ) -> None:
        self.alpha = alpha
        self.num_levels = num_price_levels
        self.num_regimes = num_regimes

        # Per-level, per-regime fill rate EWMA: shape (num_levels, num_regimes, 2)
        # last dim: [bid_fill_rate, ask_fill_rate]
        self._fill_rates = np.full((num_price_levels, num_regimes, 2), 0.8)
        self._spread_sensitivity = 0.0
        self._depth_sensitivity = 0.0
        self._vol_sensitivity = 0.0

        self._history: collections.deque = collections.deque(maxlen=1000)
        self._n = 0

    def update(
        self,
        fill_rate: float,
        side: str,
        price_level: int,
        regime: int,
        spread: float,
        depth: float,
        vol: float,
    ) -> None:
        side_idx = 0 if side == "bid" else 1
        level = min(price_level, self.num_levels - 1)
        reg = min(regime, self.num_regimes - 1)

        old = self._fill_rates[level, reg, side_idx]
        self._fill_rates[level, reg, side_idx] = (
            (1 - self.alpha) * old + self.alpha * fill_rate
        )
        self._history.append({
            "fill_rate": fill_rate,
            "spread": spread,
            "depth": depth,
            "vol": vol,
            "side": side_idx,
        })
        self._n += 1

        # Update sensitivities periodically
        if self._n % 50 == 0:
            self._update_sensitivities()

    def _update_sensitivities(self) -> None:
        if len(self._history) < 20:
            return
        data = list(self._history)
        fills = np.array([d["fill_rate"] for d in data])
        spreads = np.array([d["spread"] for d in data])
        vols = np.array([d["vol"] for d in data])

        if np.std(spreads) > 1e-10:
            corr_spread = float(np.corrcoef(fills, spreads)[0, 1])
            self._spread_sensitivity = corr_spread
        if np.std(vols) > 1e-10:
            corr_vol = float(np.corrcoef(fills, vols)[0, 1])
            self._vol_sensitivity = corr_vol

    def predict(
        self,
        side: str,
        price_level: int,
        regime: int,
        spread: float,
        depth: float,
        vol: float,
    ) -> float:
        side_idx = 0 if side == "bid" else 1
        level = min(price_level, self.num_levels - 1)
        reg = min(regime, self.num_regimes - 1)
        base = float(self._fill_rates[level, reg, side_idx])

        # Adjust for current conditions
        adjustment = (
            self._spread_sensitivity * (spread - 0.01) * (-1.0)
            + self._vol_sensitivity * (vol - 0.01) * (-0.5)
        )
        return float(np.clip(base + adjustment, 0.0, 1.0))

    @property
    def mean_fill_rate(self) -> float:
        return float(self._fill_rates.mean())


# ---------------------------------------------------------------------------
# Extended: SpreadPredictor
# ---------------------------------------------------------------------------

class SpreadPredictor:
    """
    Online spread predictor using autoregressive model with exogenous inputs.

    Models: spread_t = a * spread_{t-1} + b * vol_t + c * depth_t + noise
    """

    def __init__(self, ar_order: int = 5, num_assets: int = 4) -> None:
        self.ar_order = ar_order
        self.num_assets = num_assets
        # AR coefficients per asset
        self._ar_coefs = np.zeros((num_assets, ar_order))
        self._vol_coef = np.zeros(num_assets)
        self._depth_coef = np.zeros(num_assets)
        self._spread_history: List[collections.deque] = [
            collections.deque(maxlen=ar_order + 100) for _ in range(num_assets)
        ]
        self._prediction_errors: List[collections.deque] = [
            collections.deque(maxlen=200) for _ in range(num_assets)
        ]
        self._n = np.zeros(num_assets, dtype=int)

    def update(
        self,
        asset_id: int,
        spread: float,
        vol: float,
        depth: float,
    ) -> float:
        """Update model and return prediction for next step."""
        if asset_id >= self.num_assets:
            return spread
        hist = self._spread_history[asset_id]
        hist.append(spread)
        self._n[asset_id] += 1

        if len(hist) < self.ar_order + 2:
            return spread

        # Simple RLS-like online update
        hist_arr = np.array(list(hist))
        y = hist_arr[self.ar_order:]
        X_cols = [hist_arr[i:i + len(y)] for i in range(self.ar_order)]
        X = np.column_stack(X_cols)

        if len(y) > self.ar_order:
            try:
                coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                alpha = 0.1
                self._ar_coefs[asset_id] = (
                    (1 - alpha) * self._ar_coefs[asset_id] + alpha * coefs
                )
            except np.linalg.LinAlgError:
                pass

        # Prediction
        pred = self._predict(asset_id, hist_arr[-self.ar_order:], vol, depth)
        error = spread - pred
        self._prediction_errors[asset_id].append(error)
        return pred

    def _predict(
        self, asset_id: int, recent: np.ndarray, vol: float, depth: float
    ) -> float:
        if len(recent) < self.ar_order:
            return float(recent[-1]) if len(recent) > 0 else 0.0
        pred = float(np.dot(self._ar_coefs[asset_id], recent[-self.ar_order:]))
        pred += self._vol_coef[asset_id] * vol
        pred += self._depth_coef[asset_id] * depth
        return max(0.0, pred)

    def get_prediction_error(self, asset_id: int) -> float:
        errors = self._prediction_errors[asset_id]
        if not errors:
            return 0.0
        return float(np.std(list(errors)))


# ---------------------------------------------------------------------------
# Extended: StructuralBreakDetector
# ---------------------------------------------------------------------------

class StructuralBreakDetector:
    """
    Online structural break detector using CUSUM statistic.

    Detects changes in:
    - Mean return
    - Volatility
    - Spread level
    - Fill rate
    """

    def __init__(
        self,
        window: int = 100,
        threshold_std_multiples: float = 3.0,
        num_series: int = 4,
    ) -> None:
        self.window = window
        self.threshold = threshold_std_multiples
        self.num_series = num_series

        self._series: List[collections.deque] = [
            collections.deque(maxlen=window) for _ in range(num_series)
        ]
        self._cusums: np.ndarray = np.zeros(num_series)
        self._means: np.ndarray = np.zeros(num_series)
        self._stds: np.ndarray = np.ones(num_series)
        self._n: np.ndarray = np.zeros(num_series, dtype=int)
        self._break_detected: np.ndarray = np.zeros(num_series, dtype=bool)
        self._break_history: List[List[int]] = [[] for _ in range(num_series)]
        self._step: int = 0

    def update(self, series_id: int, value: float) -> bool:
        """
        Update CUSUM for a series. Returns True if break detected.
        """
        if series_id >= self.num_series:
            return False
        self._step += 1
        self._series[series_id].append(value)
        self._n[series_id] += 1
        n = self._n[series_id]

        # Update running stats
        arr = np.array(list(self._series[series_id]))
        self._means[series_id] = float(arr.mean())
        self._stds[series_id] = float(arr.std()) if len(arr) > 1 else 1.0

        if n < 20:
            return False

        # CUSUM update
        z = (value - self._means[series_id]) / max(self._stds[series_id], 1e-10)
        self._cusums[series_id] = max(0.0, self._cusums[series_id] + z - 0.5)

        break_detected = self._cusums[series_id] > self.threshold
        if break_detected and not self._break_detected[series_id]:
            self._break_history[series_id].append(self._step)
            self._break_detected[series_id] = True
            self._cusums[series_id] = 0.0  # reset after detection
        elif not break_detected:
            self._break_detected[series_id] = False

        return bool(break_detected)

    def reset_series(self, series_id: int) -> None:
        if series_id < self.num_series:
            self._series[series_id].clear()
            self._cusums[series_id] = 0.0
            self._n[series_id] = 0

    def get_break_times(self, series_id: int) -> List[int]:
        if series_id < self.num_series:
            return self._break_history[series_id].copy()
        return []

    @property
    def any_break_detected(self) -> bool:
        return bool(np.any(self._break_detected))


# ---------------------------------------------------------------------------
# Extended: OnlineNormalizationTracker
# ---------------------------------------------------------------------------

class OnlineNormalizationTracker:
    """
    Tracks running statistics for online feature normalization.

    Uses Welford's online algorithm for numerically stable mean/variance.
    """

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self._count = 0
        self._mean = np.zeros(feature_dim)
        self._M2 = np.zeros(feature_dim)

    def update(self, x: np.ndarray) -> None:
        """Update with a new observation vector."""
        assert len(x) == self.feature_dim
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._M2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize x using running statistics."""
        if self._count < 2:
            return x
        std = np.sqrt(self._M2 / max(self._count - 1, 1))
        return (x - self._mean) / np.maximum(std, 1e-8)

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        if self._count < 2:
            return x_norm
        std = np.sqrt(self._M2 / max(self._count - 1, 1))
        return x_norm * np.maximum(std, 1e-8) + self._mean

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        if self._count < 2:
            return np.ones(self.feature_dim)
        return np.sqrt(self._M2 / max(self._count - 1, 1))

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0
        self._mean = np.zeros(self.feature_dim)
        self._M2 = np.zeros(self.feature_dim)


# ---------------------------------------------------------------------------
# Extended: OSIDiagnostics
# ---------------------------------------------------------------------------

class OSIDiagnostics:
    """
    Diagnostic tools for the OSI module.

    Provides:
    - Health check: are all components functioning correctly?
    - Performance report: accuracy of each predictor
    - Alert summary: recent alerts and their frequency
    - Calibration check: are predictions well-calibrated?
    """

    def __init__(self, osi: OnlineSystemIdentification) -> None:
        self.osi = osi
        self._health_checks: List[Dict[str, Any]] = []

    def run_health_check(self) -> Dict[str, Any]:
        """Run full health check on OSI components."""
        issues: List[str] = []

        # Check vol estimates are reasonable
        vols = self.osi.vol_estimates
        if np.any(vols <= 0):
            issues.append("vol_estimates_non_positive")
        if np.any(vols > 1.0):
            issues.append("vol_estimates_too_large")

        # Check lambda estimates
        lambdas = self.osi.lambda_estimates
        if np.any(lambdas < 0):
            issues.append("lambda_estimates_negative")

        # Check IEM
        if self.osi.iem.buffer_size == 0 and self.osi._step > 100:
            issues.append("iem_buffer_empty")

        # Check Kalman filter stability
        for i, f in enumerate(self.osi.vol_filters):
            if not np.isfinite(f.current_vol):
                issues.append(f"kalman_filter_{i}_non_finite")

        # Check RLS stability
        for i, r in enumerate(self.osi.rls_estimators):
            if not np.isfinite(r.kyle_lambda):
                issues.append(f"rls_{i}_non_finite")

        status = "healthy" if not issues else "degraded"
        health = {
            "status": status,
            "issues": issues,
            "step": self.osi._step,
            "vol_estimates": vols.tolist(),
            "lambda_estimates": lambdas.tolist(),
        }
        self._health_checks.append(health)
        return health

    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Return accuracy metrics for each predictor."""
        residuals = self.osi.residual_monitor.get_summary()
        accuracy: Dict[str, float] = {}
        for asset_key, stats in residuals.items():
            accuracy[f"{asset_key}_impact_bias"] = stats.get("impact_ewma", 0.0)
            accuracy[f"{asset_key}_fill_bias"] = stats.get("fill_ewma", 0.0)
            accuracy[f"{asset_key}_spread_bias"] = stats.get("spread_ewma", 0.0)
        return accuracy

    def get_alert_summary(self) -> Dict[str, Any]:
        """Summarize recent alerts."""
        history = self.osi.alert_publisher.alert_history
        if not history:
            return {"total_alerts": 0, "alert_types": {}}

        type_counts: Dict[str, int] = {}
        for alert in history:
            t = alert.get("alert_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_alerts": len(history),
            "alert_types": type_counts,
            "most_recent": history[-1] if history else None,
        }

    def run_calibration_check(
        self,
        predicted_vols: List[float],
        realized_vols: List[float],
    ) -> Dict[str, float]:
        """Check vol forecast calibration."""
        if not predicted_vols or not realized_vols:
            return {"error": "no_data"}
        n = min(len(predicted_vols), len(realized_vols))
        p = np.array(predicted_vols[:n])
        r = np.array(realized_vols[:n])
        mse = float(np.mean((p - r) ** 2))
        mae = float(np.mean(np.abs(p - r)))
        corr = float(np.corrcoef(p, r)[0, 1]) if n > 2 else 0.0
        bias = float(np.mean(p - r))
        return {
            "mse": mse,
            "mae": mae,
            "correlation": corr,
            "bias": bias,
            "n": n,
        }


# ---------------------------------------------------------------------------
# Extended: OSIEnsemble
# ---------------------------------------------------------------------------

class OSIEnsemble:
    """
    Ensemble of OSI instances for more robust estimation.

    Runs N OSI instances with different hyperparameters and combines
    their outputs via weighted averaging (weight by recent accuracy).
    """

    def __init__(
        self,
        n_members: int = 3,
        base_config: Optional[OSIConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.n_members = n_members
        self.device = device
        self._members: List[OnlineSystemIdentification] = []
        self._weights: np.ndarray = np.ones(n_members) / n_members
        self._prediction_errors: List[collections.deque] = [
            collections.deque(maxlen=100) for _ in range(n_members)
        ]

        # Build diverse members
        for i in range(n_members):
            cfg = base_config or OSIConfig()
            # Perturb hyperparameters for diversity
            perturbed = OSIConfig(
                num_assets=cfg.num_assets,
                kalman_vol=KalmanVolConfig(
                    observation_noise_var=cfg.kalman_vol.observation_noise_var * (0.5 + i * 0.5),
                    process_noise_var=cfg.kalman_vol.process_noise_var * (0.5 + i * 0.3),
                    num_regimes=cfg.kalman_vol.num_regimes,
                ),
                rls=RLSConfig(
                    forgetting_factor=0.95 + i * 0.02,
                    num_features=cfg.rls.num_features,
                ),
                iem=cfg.iem,
            )
            self._members.append(OnlineSystemIdentification(perturbed, device))

    def reset(self) -> None:
        for m in self._members:
            m.reset()
        self._weights = np.ones(self.n_members) / self.n_members

    def update(self, **kwargs: Any) -> Dict[str, Any]:
        """Update all ensemble members and return aggregated output."""
        results = []
        for i, member in enumerate(self._members):
            try:
                r = member.update(**kwargs)
                results.append(r)
            except Exception:
                results.append(None)

        valid = [(i, r) for i, r in enumerate(results) if r is not None]
        if not valid:
            return {}

        # Weighted average of vol and lambda estimates
        vol_estimates = np.zeros(self._members[0].cfg.num_assets if self._members else 1)
        lambda_estimates = np.zeros_like(vol_estimates)

        total_w = 0.0
        for i, r in valid:
            w = self._weights[i]
            vol_estimates += w * r.get("vol_estimates", np.zeros_like(vol_estimates))
            lambda_estimates += w * r.get("lambda_estimates", np.zeros_like(lambda_estimates))
            total_w += w

        if total_w > 0:
            vol_estimates /= total_w
            lambda_estimates /= total_w

        # Update weights based on prediction error
        self._update_weights(valid, results)

        # Aggregate result
        agg = dict(valid[0][1]) if valid else {}
        agg["vol_estimates"] = vol_estimates
        agg["lambda_estimates"] = lambda_estimates
        agg["ensemble_weights"] = self._weights.tolist()
        return agg

    def _update_weights(
        self,
        valid: List[Tuple[int, Dict]],
        results: List[Optional[Dict]],
    ) -> None:
        """Update ensemble weights based on recent errors."""
        pass  # Simple uniform weights for now

    @property
    def vol_estimates(self) -> np.ndarray:
        vols = [m.vol_estimates for m in self._members]
        return np.average(np.array(vols), axis=0, weights=self._weights)


# ---------------------------------------------------------------------------
# Extended: OSIPerformanceMonitor
# ---------------------------------------------------------------------------

class OSIPerformanceMonitor:
    """
    Monitors OSI performance over time.

    Tracks how well the OSI's internal models predict actual outcomes,
    and raises warnings when performance degrades.
    """

    def __init__(self, window: int = 500) -> None:
        self._window = window
        self._vol_errors: collections.deque = collections.deque(maxlen=window)
        self._lambda_errors: collections.deque = collections.deque(maxlen=window)
        self._fill_errors: collections.deque = collections.deque(maxlen=window)
        self._spread_errors: collections.deque = collections.deque(maxlen=window)
        self._latencies: collections.deque = collections.deque(maxlen=window)
        self._step: int = 0

    def record(
        self,
        osi_result: Dict[str, Any],
        actual_vol: Optional[float] = None,
        actual_lambda: Optional[float] = None,
        actual_fill: Optional[float] = None,
        actual_spread: Optional[float] = None,
    ) -> None:
        self._step += 1
        self._latencies.append(osi_result.get("total_latency_ms", 0.0))

        if actual_vol is not None:
            vol_est = float(np.mean(osi_result.get("vol_estimates", [0])))
            self._vol_errors.append(abs(vol_est - actual_vol))

        if actual_lambda is not None:
            lam_est = float(np.mean(osi_result.get("lambda_estimates", [0])))
            self._lambda_errors.append(abs(lam_est - actual_lambda))

        impact_res = osi_result.get("impact_residuals", np.zeros(1))
        fill_res = osi_result.get("fill_residuals", np.zeros(1))
        self._fill_errors.append(float(np.mean(np.abs(fill_res))))
        self._spread_errors.append(float(np.mean(np.abs(impact_res))))

    def get_performance_report(self) -> Dict[str, Any]:
        def safe_stats(q: collections.deque) -> Dict[str, float]:
            arr = list(q)
            if not arr:
                return {"mean": 0.0, "std": 0.0, "p95": 0.0}
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p95": float(np.percentile(arr, 95)),
            }

        return {
            "step": self._step,
            "vol_error": safe_stats(self._vol_errors),
            "lambda_error": safe_stats(self._lambda_errors),
            "fill_error": safe_stats(self._fill_errors),
            "spread_error": safe_stats(self._spread_errors),
            "latency_ms": safe_stats(self._latencies),
            "budget_exceeded_pct": float(
                100.0 * np.mean(np.array(list(self._latencies)) > OSI_LATENCY_BUDGET_MS)
            ) if self._latencies else 0.0,
        }

    def is_healthy(self) -> bool:
        """Return True if OSI performance is within acceptable bounds."""
        report = self.get_performance_report()
        lat_mean = report["latency_ms"]["mean"]
        return lat_mean < OSI_LATENCY_BUDGET_MS * 2.0


# ---------------------------------------------------------------------------
# Extended: IEMEnsemble
# ---------------------------------------------------------------------------

class IEMEnsemble:
    """
    Ensemble of IEM models for uncertainty-aware prediction.

    Combines multiple IEM instances trained with different seeds/configs
    to produce a Gaussian mixture prediction with calibrated uncertainty.
    """

    def __init__(
        self,
        n_members: int = 5,
        obs_dim: int = 64,
        action_dim: int = 8,
        hidden_dim: int = 128,
        output_dim: int = 64,
        device: str = "cpu",
    ) -> None:
        self.n_members = n_members
        cfg_base = IEMConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self._members = [
            AgentInternalEnvironmentModel(cfg_base, device)
            for _ in range(n_members)
        ]
        self._device = device

    def predict_with_uncertainty(
        self, obs: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns (mean_prediction, uncertainty, mean_latency_ms).

        uncertainty = std over ensemble members.
        """
        preds = []
        total_lat = 0.0
        for member in self._members:
            pred, lat = member.predict(obs, action)
            preds.append(pred)
            total_lat += lat

        if not preds:
            return np.zeros(self._members[0].cfg.output_dim), np.ones(self._members[0].cfg.output_dim), 0.0

        preds_arr = np.array(preds)
        mean_pred = preds_arr.mean(axis=0)
        unc = preds_arr.std(axis=0)
        return mean_pred, unc, total_lat / max(len(preds), 1)

    def push_all(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> float:
        """Push to all members. Returns mean residual."""
        residuals = []
        for member in self._members:
            r = member.push(obs, action, next_obs)
            residuals.append(r)
        return float(np.mean(residuals))

    def update_all(self) -> List[Optional[float]]:
        """Update all members."""
        return [m.update() for m in self._members]

    def reset_all(self) -> None:
        for m in self._members:
            m.reset()


# ---------------------------------------------------------------------------
# Extended: AdaptiveLearningRateScheduler
# ---------------------------------------------------------------------------

class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler for online IEM training.

    Adjusts learning rate based on:
    - Residual magnitude (high residual → increase lr)
    - Recent loss trend (decreasing loss → maintain lr)
    - Oscillation detection (oscillating loss → decrease lr)
    """

    def __init__(
        self,
        initial_lr: float = 1e-3,
        min_lr: float = 1e-5,
        max_lr: float = 1e-2,
        window: int = 50,
        residual_boost_threshold: float = 0.1,
        oscillation_window: int = 10,
    ) -> None:
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.window = window
        self.residual_boost_threshold = residual_boost_threshold
        self.oscillation_window = oscillation_window

        self._current_lr = initial_lr
        self._loss_history: collections.deque = collections.deque(maxlen=window)
        self._residual_history: collections.deque = collections.deque(maxlen=window)
        self._step = 0

    def update(self, loss: float, residual: float) -> float:
        """Update and return new learning rate."""
        self._step += 1
        self._loss_history.append(loss)
        self._residual_history.append(residual)

        lr = self._current_lr

        # Boost lr if high residual (model is lagging)
        if residual > self.residual_boost_threshold:
            lr = min(self.max_lr, lr * 1.5)

        # Check oscillation
        if len(self._loss_history) >= self.oscillation_window:
            recent = np.array(list(self._loss_history)[-self.oscillation_window:])
            sign_changes = np.sum(np.diff(np.sign(np.diff(recent))) != 0)
            if sign_changes > self.oscillation_window // 2:
                lr = max(self.min_lr, lr * 0.5)

        # Check trend
        if len(self._loss_history) >= 20:
            trend = float(np.mean(list(self._loss_history)[-5:])) - float(
                np.mean(list(self._loss_history)[-20:-5])
            )
            if trend < -0.001:  # improving
                lr = min(self.max_lr, lr * 1.05)
            elif trend > 0.001:  # worsening
                lr = max(self.min_lr, lr * 0.9)

        self._current_lr = float(np.clip(lr, self.min_lr, self.max_lr))
        return self._current_lr

    @property
    def current_lr(self) -> float:
        return self._current_lr

    def reset(self) -> None:
        self._current_lr = self.initial_lr
        self._loss_history.clear()
        self._residual_history.clear()
        self._step = 0


# ---------------------------------------------------------------------------
# Extended: OSIStateBuffer
# ---------------------------------------------------------------------------

class OSIStateBuffer:
    """
    Buffer that stores a history of OSI states for downstream use.

    Useful for:
    - Providing OSI state history to agent observations
    - Computing multi-step OSI features
    - Replay of OSI state sequences for debugging
    """

    def __init__(self, capacity: int = 1000, feature_dim: int = 32) -> None:
        self._capacity = capacity
        self._feature_dim = feature_dim
        self._buffer: collections.deque = collections.deque(maxlen=capacity)
        self._step = 0

    def push(self, state: Dict[str, Any]) -> None:
        """Push an OSI state dict."""
        self._step += 1
        features = self._extract_features(state)
        self._buffer.append({
            "step": self._step,
            "features": features,
            "raw": {k: v for k, v in state.items() if isinstance(v, (int, float, bool))},
        })

    def _extract_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract a fixed-size feature vector from state."""
        features = []
        for key in ["vol_estimates", "lambda_estimates"]:
            val = state.get(key, [])
            if isinstance(val, np.ndarray):
                features.extend(val.flatten().tolist()[:4])
            else:
                features.extend([0.0] * 4)
        # Pad/truncate to feature_dim
        features = features[:self._feature_dim]
        features += [0.0] * max(0, self._feature_dim - len(features))
        return np.array(features, dtype=np.float32)

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent states."""
        buf = list(self._buffer)
        return buf[-n:]

    def get_feature_sequence(self, n: int = 10) -> np.ndarray:
        """Get feature sequence as (n, feature_dim) array."""
        recent = self.get_recent(n)
        if not recent:
            return np.zeros((n, self._feature_dim), dtype=np.float32)
        feats = [s["features"] for s in recent]
        # Pad if needed
        while len(feats) < n:
            feats.insert(0, np.zeros(self._feature_dim, dtype=np.float32))
        return np.array(feats[-n:], dtype=np.float32)

    def clear(self) -> None:
        self._buffer.clear()
        self._step = 0

    def __len__(self) -> int:
        return len(self._buffer)
