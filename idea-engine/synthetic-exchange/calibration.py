"""
calibration.py - Calibrate the synthetic exchange to match real market data.

Provides HistoricalCalibrator, CalibrationReport, and RealisticExchangeFactory
for fitting simulation parameters from historical data and verifying statistical
fidelity of the synthetic exchange.
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats, optimize, signal as sp_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VolatilityParams:
    """Parameters for the volatility model."""
    realized_vol_daily: float = 0.0
    realized_vol_annual: float = 0.0
    garch_omega: float = 1e-6
    garch_alpha: float = 0.1
    garch_beta: float = 0.85
    garch_persistence: float = 0.95
    vol_of_vol: float = 0.0
    intraday_vol_pattern: np.ndarray = field(default_factory=lambda: np.ones(8))
    long_run_variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realized_vol_daily": self.realized_vol_daily,
            "realized_vol_annual": self.realized_vol_annual,
            "garch_omega": self.garch_omega,
            "garch_alpha": self.garch_alpha,
            "garch_beta": self.garch_beta,
            "garch_persistence": self.garch_persistence,
            "vol_of_vol": self.vol_of_vol,
            "intraday_vol_pattern": self.intraday_vol_pattern.tolist(),
            "long_run_variance": self.long_run_variance,
        }


@dataclass
class OrderBookParams:
    """Parameters describing order book shape."""
    depth_profile_bids: np.ndarray = field(default_factory=lambda: np.ones(20))
    depth_profile_asks: np.ndarray = field(default_factory=lambda: np.ones(20))
    avg_depth_per_level: float = 100.0
    depth_decay_rate: float = 0.1
    depth_asymmetry: float = 0.0  # positive = more bids
    resilience_rate: float = 0.5  # how fast book recovers after a trade
    queue_priority_factor: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth_profile_bids": self.depth_profile_bids.tolist(),
            "depth_profile_asks": self.depth_profile_asks.tolist(),
            "avg_depth_per_level": self.avg_depth_per_level,
            "depth_decay_rate": self.depth_decay_rate,
            "depth_asymmetry": self.depth_asymmetry,
            "resilience_rate": self.resilience_rate,
            "queue_priority_factor": self.queue_priority_factor,
        }


@dataclass
class HawkesParams:
    """Parameters for Hawkes process trade arrival model."""
    baseline_intensity: float = 1.0  # mu: base arrival rate
    self_excitation: float = 0.5  # alpha: jump in intensity per event
    decay_rate: float = 1.0  # beta: decay rate of excitation
    branching_ratio: float = 0.5  # alpha/beta: <1 for stationarity
    mean_inter_arrival: float = 1.0
    cluster_factor: float = 1.0

    @property
    def is_stationary(self) -> bool:
        return self.branching_ratio < 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_intensity": self.baseline_intensity,
            "self_excitation": self.self_excitation,
            "decay_rate": self.decay_rate,
            "branching_ratio": self.branching_ratio,
            "mean_inter_arrival": self.mean_inter_arrival,
            "cluster_factor": self.cluster_factor,
        }


@dataclass
class SpreadModelParams:
    """Parameters for the bid-ask spread model."""
    base_spread_bps: float = 10.0
    volume_sensitivity: float = -0.3  # spread decreases with volume
    volatility_sensitivity: float = 2.0  # spread increases with vol
    inventory_sensitivity: float = 0.5
    time_of_day_pattern: np.ndarray = field(default_factory=lambda: np.ones(8))
    min_spread_bps: float = 1.0
    max_spread_bps: float = 200.0
    mean_reversion_speed: float = 0.1

    def predict_spread(
        self, volume_z: float, vol_z: float, inventory_z: float = 0.0
    ) -> float:
        """Predict spread in bps given standardized inputs."""
        log_spread = (
            np.log(self.base_spread_bps)
            + self.volume_sensitivity * volume_z
            + self.volatility_sensitivity * vol_z
            + self.inventory_sensitivity * inventory_z
        )
        spread = np.exp(log_spread)
        return float(np.clip(spread, self.min_spread_bps, self.max_spread_bps))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_spread_bps": self.base_spread_bps,
            "volume_sensitivity": self.volume_sensitivity,
            "volatility_sensitivity": self.volatility_sensitivity,
            "inventory_sensitivity": self.inventory_sensitivity,
            "time_of_day_pattern": self.time_of_day_pattern.tolist(),
            "min_spread_bps": self.min_spread_bps,
            "max_spread_bps": self.max_spread_bps,
            "mean_reversion_speed": self.mean_reversion_speed,
        }


@dataclass
class RegimeParams:
    """Parameters for regime-switching model."""
    num_regimes: int = 3
    transition_matrix: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.9 + np.ones((3, 3)) * 0.1 / 3)
    regime_means: np.ndarray = field(default_factory=lambda: np.array([0.0005, 0.0, -0.001]))
    regime_vols: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.02, 0.04]))
    stationary_dist: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.2]))
    regime_labels: List[str] = field(default_factory=lambda: ["low_vol", "normal", "high_vol"])
    avg_regime_duration: np.ndarray = field(default_factory=lambda: np.array([50, 30, 10]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_regimes": self.num_regimes,
            "transition_matrix": self.transition_matrix.tolist(),
            "regime_means": self.regime_means.tolist(),
            "regime_vols": self.regime_vols.tolist(),
            "stationary_dist": self.stationary_dist.tolist(),
            "regime_labels": self.regime_labels,
            "avg_regime_duration": self.avg_regime_duration.tolist(),
        }


@dataclass
class JumpParams:
    """Parameters for the jump-diffusion model."""
    jump_intensity: float = 0.05  # lambda: expected jumps per bar
    jump_mean: float = 0.0  # mean jump size (log)
    jump_std: float = 0.03  # jump size std (log)
    positive_jump_prob: float = 0.45
    negative_jump_prob: float = 0.55
    avg_positive_jump: float = 0.02
    avg_negative_jump: float = -0.025
    jump_clustering: float = 0.3  # probability of second jump after first

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jump_intensity": self.jump_intensity,
            "jump_mean": self.jump_mean,
            "jump_std": self.jump_std,
            "positive_jump_prob": self.positive_jump_prob,
            "negative_jump_prob": self.negative_jump_prob,
            "avg_positive_jump": self.avg_positive_jump,
            "avg_negative_jump": self.avg_negative_jump,
            "jump_clustering": self.jump_clustering,
        }


@dataclass
class ExchangeConfig:
    """Complete exchange configuration derived from calibration."""
    volatility_params: VolatilityParams = field(default_factory=VolatilityParams)
    orderbook_params: OrderBookParams = field(default_factory=OrderBookParams)
    hawkes_params: HawkesParams = field(default_factory=HawkesParams)
    spread_params: SpreadModelParams = field(default_factory=SpreadModelParams)
    regime_params: RegimeParams = field(default_factory=RegimeParams)
    jump_params: JumpParams = field(default_factory=JumpParams)
    initial_price: float = 100.0
    tick_size: float = 0.01
    lot_size: float = 1.0
    num_market_makers: int = 5
    num_trend_followers: int = 8
    num_mean_reversion: int = 4
    num_noise_traders: int = 20
    num_hft_agents: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "volatility": self.volatility_params.to_dict(),
            "orderbook": self.orderbook_params.to_dict(),
            "hawkes": self.hawkes_params.to_dict(),
            "spread": self.spread_params.to_dict(),
            "regime": self.regime_params.to_dict(),
            "jump": self.jump_params.to_dict(),
            "initial_price": self.initial_price,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "num_market_makers": self.num_market_makers,
            "num_trend_followers": self.num_trend_followers,
            "num_mean_reversion": self.num_mean_reversion,
            "num_noise_traders": self.num_noise_traders,
            "num_hft_agents": self.num_hft_agents,
        }

    def to_simulation_config(self) -> Dict[str, Any]:
        """Convert to the format expected by the Go exchange API."""
        return {
            "symbols": ["SIM-USD"],
            "initial_price": self.initial_price,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "volatility": self.volatility_params.realized_vol_daily,
            "drift": float(self.regime_params.regime_means[0]),
            "mean_spread_bps": self.spread_params.base_spread_bps,
            "num_market_makers": self.num_market_makers,
            "num_trend_followers": self.num_trend_followers,
            "num_mean_reversion": self.num_mean_reversion,
            "num_noise_traders": self.num_noise_traders,
            "num_hft_agents": self.num_hft_agents,
            "book_depth_levels": len(self.orderbook_params.depth_profile_bids),
            "garch_omega": self.volatility_params.garch_omega,
            "garch_alpha": self.volatility_params.garch_alpha,
            "garch_beta": self.volatility_params.garch_beta,
            "hawkes_mu": self.hawkes_params.baseline_intensity,
            "hawkes_alpha": self.hawkes_params.self_excitation,
            "hawkes_beta": self.hawkes_params.decay_rate,
            "jump_intensity": self.jump_params.jump_intensity,
            "jump_mean": self.jump_params.jump_mean,
            "jump_std": self.jump_params.jump_std,
            "regime_transition_matrix": self.regime_params.transition_matrix.tolist(),
            "regime_means": self.regime_params.regime_means.tolist(),
            "regime_vols": self.regime_params.regime_vols.tolist(),
        }


# ---------------------------------------------------------------------------
# HistoricalCalibrator
# ---------------------------------------------------------------------------

class HistoricalCalibrator:
    """Fit simulation parameters from real historical market data."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # -- Volatility ---------------------------------------------------------

    def fit_volatility(self, returns: np.ndarray) -> VolatilityParams:
        """
        Fit volatility parameters from a return series.

        Fits:
          - Realized volatility (daily and annualized)
          - GARCH(1,1) parameters via MLE
          - Vol-of-vol
          - Intraday vol pattern (if enough data)
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            logger.warning("Too few returns for volatility calibration")
            return VolatilityParams()

        # realized vol
        daily_vol = float(np.std(returns, ddof=1))
        annual_vol = daily_vol * np.sqrt(252)

        # GARCH(1,1) via quasi-MLE
        omega, alpha, beta = self._fit_garch11(returns)
        persistence = alpha + beta
        long_run_var = omega / max(1.0 - persistence, 1e-10) if persistence < 1 else daily_vol ** 2

        # vol of vol: rolling 20-period vol, then std of that
        if len(returns) >= 40:
            window = 20
            rolling_vol = np.array([
                np.std(returns[i:i + window], ddof=1)
                for i in range(len(returns) - window + 1)
            ])
            vol_of_vol = float(np.std(rolling_vol, ddof=1))
        else:
            vol_of_vol = daily_vol * 0.2

        # intraday pattern: split returns into 8 buckets
        n_buckets = 8
        bucket_size = max(1, len(returns) // n_buckets)
        pattern = np.ones(n_buckets)
        for i in range(n_buckets):
            start = i * bucket_size
            end = min(start + bucket_size, len(returns))
            if end > start:
                bucket_vol = np.std(returns[start:end], ddof=1)
                pattern[i] = bucket_vol / (daily_vol + 1e-10)
        pattern = pattern / (np.mean(pattern) + 1e-10)

        return VolatilityParams(
            realized_vol_daily=daily_vol,
            realized_vol_annual=annual_vol,
            garch_omega=omega,
            garch_alpha=alpha,
            garch_beta=beta,
            garch_persistence=persistence,
            vol_of_vol=vol_of_vol,
            intraday_vol_pattern=pattern,
            long_run_variance=long_run_var,
        )

    def _fit_garch11(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """Fit GARCH(1,1) via MLE with constraints."""
        T = len(returns)
        var_sample = np.var(returns)

        def neg_loglik(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            sigma2 = np.empty(T)
            sigma2[0] = var_sample
            for t in range(1, T):
                sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
                if sigma2[t] <= 0:
                    return 1e10
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns ** 2 / sigma2)
            return -ll

        # initial guess
        x0 = np.array([var_sample * 0.05, 0.1, 0.85])
        bounds = [(1e-10, var_sample), (1e-6, 0.5), (0.3, 0.999)]

        try:
            result = optimize.minimize(
                neg_loglik, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if result.success:
                omega, alpha, beta = result.x
                if alpha + beta < 1:
                    return float(omega), float(alpha), float(beta)
        except Exception as exc:
            logger.debug("GARCH optimization failed: %s", exc)

        # fallback: method of moments
        omega = var_sample * 0.05
        alpha = 0.1
        beta = 0.85
        return omega, alpha, beta

    # -- Order Book Shape ---------------------------------------------------

    def fit_orderbook_shape(
        self,
        orderbook_snapshots: List[Dict[str, Any]],
    ) -> OrderBookParams:
        """
        Fit order book depth profile parameters from snapshots.

        Each snapshot should have keys: 'bids' and 'asks', each a list of
        [price, quantity] pairs.
        """
        if not orderbook_snapshots:
            logger.warning("No orderbook snapshots provided")
            return OrderBookParams()

        max_levels = 20
        bid_profiles = []
        ask_profiles = []
        asymmetries = []
        depths = []

        for snap in orderbook_snapshots:
            bids = snap.get("bids", [])
            asks = snap.get("asks", [])

            bid_qty = np.zeros(max_levels)
            ask_qty = np.zeros(max_levels)

            for i, level in enumerate(bids[:max_levels]):
                if isinstance(level, (list, tuple)) and len(level) >= 2:
                    bid_qty[i] = level[1]
                elif isinstance(level, dict):
                    bid_qty[i] = level.get("quantity", level.get("qty", 0))

            for i, level in enumerate(asks[:max_levels]):
                if isinstance(level, (list, tuple)) and len(level) >= 2:
                    ask_qty[i] = level[1]
                elif isinstance(level, dict):
                    ask_qty[i] = level.get("quantity", level.get("qty", 0))

            bid_sum = np.sum(bid_qty)
            ask_sum = np.sum(ask_qty)
            if bid_sum > 0:
                bid_profiles.append(bid_qty / bid_sum)
            if ask_sum > 0:
                ask_profiles.append(ask_qty / ask_sum)

            total = bid_sum + ask_sum
            if total > 0:
                asymmetries.append((bid_sum - ask_sum) / total)
            depths.append((bid_sum + ask_sum) / max_levels)

        avg_bid_profile = np.mean(bid_profiles, axis=0) if bid_profiles else np.ones(max_levels) / max_levels
        avg_ask_profile = np.mean(ask_profiles, axis=0) if ask_profiles else np.ones(max_levels) / max_levels

        # fit exponential decay to profile
        avg_profile = (avg_bid_profile + avg_ask_profile) / 2.0
        levels_x = np.arange(max_levels, dtype=float)
        log_profile = np.log(avg_profile + 1e-10)
        try:
            slope, intercept = np.polyfit(levels_x, log_profile, 1)
            decay_rate = -slope
        except Exception:
            decay_rate = 0.1

        # resilience: how fast the book recovers (estimated from variance of depth)
        depth_arr = np.array(depths)
        if len(depth_arr) > 1:
            depth_var = np.var(depth_arr)
            depth_mean = np.mean(depth_arr)
            resilience = 1.0 / (1.0 + depth_var / (depth_mean ** 2 + 1e-10))
        else:
            resilience = 0.5

        return OrderBookParams(
            depth_profile_bids=avg_bid_profile,
            depth_profile_asks=avg_ask_profile,
            avg_depth_per_level=float(np.mean(depths)) if depths else 100.0,
            depth_decay_rate=float(decay_rate),
            depth_asymmetry=float(np.mean(asymmetries)) if asymmetries else 0.0,
            resilience_rate=float(resilience),
        )

    # -- Trade Arrival (Hawkes) --------------------------------------------

    def fit_trade_arrival(self, trade_times: np.ndarray) -> HawkesParams:
        """
        Fit a Hawkes process to trade arrival times.

        Uses the EM algorithm for univariate Hawkes(mu, alpha, beta).
        """
        trade_times = np.asarray(trade_times, dtype=np.float64)
        trade_times = np.sort(trade_times)
        if len(trade_times) < 20:
            logger.warning("Too few trades for Hawkes calibration")
            return HawkesParams()

        # normalize to [0, T]
        t0 = trade_times[0]
        times = trade_times - t0
        T = times[-1]
        n = len(times)

        if T <= 0:
            return HawkesParams()

        # empirical mean rate
        mean_rate = n / T

        # fit via MLE
        def neg_loglik(params):
            mu, alpha, beta_h = params
            if mu <= 0 or alpha < 0 or beta_h <= 0 or alpha >= beta_h:
                return 1e10
            A = 0.0
            ll = 0.0
            for i in range(n):
                if i > 0:
                    dt = times[i] - times[i - 1]
                    A = np.exp(-beta_h * dt) * (1.0 + A)
                lam_i = mu + alpha * A
                if lam_i <= 0:
                    return 1e10
                ll += np.log(lam_i)

            # integral of lambda
            integral = mu * T
            A_sum = 0.0
            for i in range(n):
                A_sum += 1.0 - np.exp(-beta_h * (T - times[i]))
            integral += alpha / beta_h * A_sum
            ll -= integral
            return -ll

        x0 = np.array([mean_rate * 0.5, mean_rate * 0.3, mean_rate * 1.0])
        bounds = [(1e-6, mean_rate * 5), (1e-6, mean_rate * 10), (1e-4, mean_rate * 20)]

        try:
            result = optimize.minimize(
                neg_loglik, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 300},
            )
            if result.success:
                mu, alpha_h, beta_h = result.x
                branching = alpha_h / beta_h
                if branching < 1:
                    return HawkesParams(
                        baseline_intensity=float(mu),
                        self_excitation=float(alpha_h),
                        decay_rate=float(beta_h),
                        branching_ratio=float(branching),
                        mean_inter_arrival=float(T / n),
                        cluster_factor=float(1.0 / (1.0 - branching)),
                    )
        except Exception as exc:
            logger.debug("Hawkes MLE failed: %s", exc)

        # fallback: moment-based
        inter_arrivals = np.diff(times)
        if len(inter_arrivals) == 0:
            return HawkesParams()
        mean_ia = np.mean(inter_arrivals)
        var_ia = np.var(inter_arrivals)
        cv2 = var_ia / (mean_ia ** 2 + 1e-10)
        branching_est = max(0, min(0.95, 1.0 - 1.0 / (cv2 + 1e-10)))
        mu_est = mean_rate * (1.0 - branching_est)
        beta_est = mean_rate
        alpha_est = branching_est * beta_est

        return HawkesParams(
            baseline_intensity=float(mu_est),
            self_excitation=float(alpha_est),
            decay_rate=float(beta_est),
            branching_ratio=float(branching_est),
            mean_inter_arrival=float(mean_ia),
            cluster_factor=float(1.0 / (1.0 - branching_est + 1e-10)),
        )

    # -- Spread Model -------------------------------------------------------

    def fit_spread_model(
        self,
        spreads: np.ndarray,
        volumes: np.ndarray,
        volatilities: np.ndarray,
    ) -> SpreadModelParams:
        """
        Fit a spread model: log(spread) = a + b1*vol_z + b2*volume_z.

        Inputs should be arrays of equal length.
        """
        spreads = np.asarray(spreads, dtype=np.float64)
        volumes = np.asarray(volumes, dtype=np.float64)
        volatilities = np.asarray(volatilities, dtype=np.float64)

        mask = (spreads > 0) & np.isfinite(spreads) & np.isfinite(volumes) & np.isfinite(volatilities)
        spreads = spreads[mask]
        volumes = volumes[mask]
        volatilities = volatilities[mask]

        if len(spreads) < 10:
            logger.warning("Too few spread observations")
            return SpreadModelParams()

        log_spreads = np.log(spreads)

        vol_mean, vol_std = np.mean(volatilities), np.std(volatilities)
        volume_mean, volume_std = np.mean(volumes), np.std(volumes)

        vol_z = (volatilities - vol_mean) / (vol_std + 1e-10)
        volume_z = (volumes - volume_mean) / (volume_std + 1e-10)

        # OLS regression: log(spread) = intercept + b1*vol_z + b2*volume_z
        X = np.column_stack([np.ones(len(spreads)), vol_z, volume_z])
        try:
            beta_hat, residuals, rank, sv = np.linalg.lstsq(X, log_spreads, rcond=None)
            intercept = beta_hat[0]
            vol_sens = beta_hat[1]
            vol_trade_sens = beta_hat[2]
        except Exception:
            intercept = np.mean(log_spreads)
            vol_sens = 2.0
            vol_trade_sens = -0.3

        base_spread = np.exp(intercept)

        # time-of-day pattern
        n_buckets = 8
        bucket_size = max(1, len(spreads) // n_buckets)
        pattern = np.ones(n_buckets)
        for i in range(n_buckets):
            start = i * bucket_size
            end = min(start + bucket_size, len(spreads))
            if end > start:
                pattern[i] = np.mean(spreads[start:end]) / (np.mean(spreads) + 1e-10)
        pattern = pattern / (np.mean(pattern) + 1e-10)

        # mean reversion speed (half-life from autocorrelation)
        if len(log_spreads) > 2:
            autocorr = np.corrcoef(log_spreads[:-1], log_spreads[1:])[0, 1]
            if 0 < autocorr < 1:
                half_life = -np.log(2) / np.log(autocorr)
                mr_speed = 1.0 / max(half_life, 1.0)
            else:
                mr_speed = 0.1
        else:
            mr_speed = 0.1

        return SpreadModelParams(
            base_spread_bps=float(base_spread),
            volume_sensitivity=float(vol_trade_sens),
            volatility_sensitivity=float(vol_sens),
            time_of_day_pattern=pattern,
            min_spread_bps=float(np.percentile(spreads, 1)),
            max_spread_bps=float(np.percentile(spreads, 99)),
            mean_reversion_speed=float(mr_speed),
        )

    # -- Regime Model -------------------------------------------------------

    def fit_regime(self, returns: np.ndarray, num_regimes: int = 3) -> RegimeParams:
        """
        Fit a regime-switching model using EM for Gaussian mixture with Markov transitions.

        Simplified Baum-Welch for HMM with Gaussian emissions.
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        T = len(returns)
        K = num_regimes

        if T < 30:
            logger.warning("Too few returns for regime calibration")
            return RegimeParams()

        # initialize via quantile-based clustering
        sorted_rets = np.sort(returns)
        boundaries = np.linspace(0, T, K + 1, dtype=int)
        means = np.array([np.mean(sorted_rets[boundaries[i]:boundaries[i + 1]]) for i in range(K)])
        vols = np.array([np.std(sorted_rets[boundaries[i]:boundaries[i + 1]], ddof=1) + 1e-8 for i in range(K)])

        # sort regimes by vol
        order = np.argsort(vols)
        means = means[order]
        vols = vols[order]

        # transition matrix: high persistence
        trans = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(trans, 0.95)
        trans = trans / trans.sum(axis=1, keepdims=True)

        pi = np.ones(K) / K  # initial distribution

        # EM iterations
        for iteration in range(50):
            # E-step: forward-backward
            # emission probabilities
            B = np.zeros((T, K))
            for k in range(K):
                B[:, k] = stats.norm.pdf(returns, loc=means[k], scale=vols[k])
            B = np.maximum(B, 1e-300)

            # forward
            alpha = np.zeros((T, K))
            alpha[0] = pi * B[0]
            scale = np.zeros(T)
            scale[0] = np.sum(alpha[0])
            if scale[0] > 0:
                alpha[0] /= scale[0]

            for t in range(1, T):
                alpha[t] = B[t] * (alpha[t - 1] @ trans)
                scale[t] = np.sum(alpha[t])
                if scale[t] > 0:
                    alpha[t] /= scale[t]

            # backward
            beta = np.zeros((T, K))
            beta[-1] = 1.0

            for t in range(T - 2, -1, -1):
                beta[t] = trans @ (B[t + 1] * beta[t + 1])
                if scale[t + 1] > 0:
                    beta[t] /= scale[t + 1]

            # gamma and xi
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum = np.maximum(gamma_sum, 1e-300)
            gamma = gamma / gamma_sum

            # M-step
            for k in range(K):
                wk = gamma[:, k]
                wk_sum = np.sum(wk) + 1e-10
                means[k] = np.sum(wk * returns) / wk_sum
                diff = returns - means[k]
                vols[k] = np.sqrt(np.sum(wk * diff ** 2) / wk_sum + 1e-10)

            # transition matrix
            for i in range(K):
                for j in range(K):
                    num = 0.0
                    den = 0.0
                    for t in range(T - 1):
                        xi_ij = alpha[t, i] * trans[i, j] * B[t + 1, j] * beta[t + 1, j]
                        num += xi_ij
                        den += gamma[t, i]
                    trans[i, j] = num / (den + 1e-10)
                row_sum = np.sum(trans[i])
                if row_sum > 0:
                    trans[i] /= row_sum

            pi = gamma[0]

        # stationary distribution
        try:
            eigenvalues, eigenvectors = np.linalg.eig(trans.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx])
            stationary = np.abs(stationary)
            stationary = stationary / (np.sum(stationary) + 1e-10)
        except Exception:
            stationary = np.ones(K) / K

        # average regime duration
        avg_duration = 1.0 / (1.0 - np.diag(trans) + 1e-10)

        labels = []
        for k in range(K):
            if vols[k] < np.median(vols):
                labels.append(f"low_vol_{k}")
            elif vols[k] > np.percentile(vols, 75):
                labels.append(f"high_vol_{k}")
            else:
                labels.append(f"normal_{k}")

        return RegimeParams(
            num_regimes=K,
            transition_matrix=trans,
            regime_means=means,
            regime_vols=vols,
            stationary_dist=stationary,
            regime_labels=labels,
            avg_regime_duration=avg_duration,
        )

    # -- Jump Process -------------------------------------------------------

    def fit_jump_process(self, returns: np.ndarray) -> JumpParams:
        """
        Fit jump parameters from returns using a threshold-based approach.

        Jumps are identified as returns exceeding 3 * MAD (median absolute deviation).
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        if len(returns) < 20:
            return JumpParams()

        # identify jumps via MAD threshold
        median = np.median(returns)
        mad = np.median(np.abs(returns - median))
        threshold = 3.0 * max(mad, 1e-8) * 1.4826  # scale to std

        is_jump = np.abs(returns - median) > threshold
        jump_returns = returns[is_jump]
        n_jumps = np.sum(is_jump)
        n_total = len(returns)

        jump_intensity = n_jumps / n_total if n_total > 0 else 0.05

        if n_jumps >= 2:
            jump_mean = float(np.mean(jump_returns))
            jump_std = float(np.std(jump_returns, ddof=1))
        else:
            jump_mean = 0.0
            jump_std = float(threshold)

        pos_jumps = jump_returns[jump_returns > 0]
        neg_jumps = jump_returns[jump_returns < 0]

        pos_prob = len(pos_jumps) / max(n_jumps, 1)
        neg_prob = len(neg_jumps) / max(n_jumps, 1)
        avg_pos = float(np.mean(pos_jumps)) if len(pos_jumps) > 0 else 0.02
        avg_neg = float(np.mean(neg_jumps)) if len(neg_jumps) > 0 else -0.025

        # jump clustering: conditional probability of jump given previous jump
        jump_indices = np.where(is_jump)[0]
        consecutive = 0
        for i in range(1, len(jump_indices)):
            if jump_indices[i] == jump_indices[i - 1] + 1:
                consecutive += 1
        clustering = consecutive / max(n_jumps - 1, 1) if n_jumps > 1 else 0.0

        return JumpParams(
            jump_intensity=float(jump_intensity),
            jump_mean=jump_mean,
            jump_std=jump_std,
            positive_jump_prob=float(pos_prob),
            negative_jump_prob=float(neg_prob),
            avg_positive_jump=avg_pos,
            avg_negative_jump=avg_neg,
            jump_clustering=float(clustering),
        )


# ---------------------------------------------------------------------------
# CalibrationReport
# ---------------------------------------------------------------------------

@dataclass
class DistributionComparison:
    """Compare a simulated vs real distribution."""
    name: str
    real_mean: float
    sim_mean: float
    real_std: float
    sim_std: float
    real_skew: float
    sim_skew: float
    real_kurtosis: float
    sim_kurtosis: float
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float = 0.0
    wasserstein_distance: float = 0.0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "real_mean": self.real_mean,
            "sim_mean": self.sim_mean,
            "real_std": self.real_std,
            "sim_std": self.sim_std,
            "real_skew": self.real_skew,
            "sim_skew": self.sim_skew,
            "real_kurtosis": self.real_kurtosis,
            "sim_kurtosis": self.sim_kurtosis,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "ad_statistic": self.ad_statistic,
            "wasserstein_distance": self.wasserstein_distance,
            "passed": self.passed,
        }


class CalibrationReport:
    """Compare simulated vs real market statistics for calibration validation."""

    def __init__(
        self,
        real_returns: np.ndarray,
        sim_returns: np.ndarray,
        real_spreads: Optional[np.ndarray] = None,
        sim_spreads: Optional[np.ndarray] = None,
        real_trade_sizes: Optional[np.ndarray] = None,
        sim_trade_sizes: Optional[np.ndarray] = None,
        significance_level: float = 0.05,
    ):
        self.real_returns = np.asarray(real_returns, dtype=np.float64)
        self.sim_returns = np.asarray(sim_returns, dtype=np.float64)
        self.real_spreads = np.asarray(real_spreads, dtype=np.float64) if real_spreads is not None else None
        self.sim_spreads = np.asarray(sim_spreads, dtype=np.float64) if sim_spreads is not None else None
        self.real_trade_sizes = np.asarray(real_trade_sizes, dtype=np.float64) if real_trade_sizes is not None else None
        self.sim_trade_sizes = np.asarray(sim_trade_sizes, dtype=np.float64) if sim_trade_sizes is not None else None
        self.significance = significance_level
        self._comparisons: List[DistributionComparison] = []

    def run(self) -> Dict[str, Any]:
        """Run all comparisons and return the full report."""
        self._comparisons = []

        # Return distribution
        self._comparisons.append(
            self._compare_distributions("returns", self.real_returns, self.sim_returns)
        )

        # Spread distribution
        if self.real_spreads is not None and self.sim_spreads is not None:
            self._comparisons.append(
                self._compare_distributions("spreads", self.real_spreads, self.sim_spreads)
            )

        # Trade size distribution
        if self.real_trade_sizes is not None and self.sim_trade_sizes is not None:
            self._comparisons.append(
                self._compare_distributions("trade_sizes", self.real_trade_sizes, self.sim_trade_sizes)
            )

        # Autocorrelation structure
        ac_comparison = self._compare_autocorrelation()

        # Volatility clustering
        vol_cluster = self._compare_volatility_clustering()

        # Aggregate score
        all_passed = all(c.passed for c in self._comparisons)
        ks_pvalues = {c.name: c.ks_pvalue for c in self._comparisons}

        return {
            "distributions": [c.to_dict() for c in self._comparisons],
            "autocorrelation": ac_comparison,
            "volatility_clustering": vol_cluster,
            "all_ks_pvalues": ks_pvalues,
            "all_passed": all_passed,
            "num_comparisons": len(self._comparisons),
            "num_passed": sum(1 for c in self._comparisons if c.passed),
            "calibration_score": self._compute_score(),
        }

    def _compare_distributions(
        self, name: str, real: np.ndarray, sim: np.ndarray,
    ) -> DistributionComparison:
        """Compare two distributions with statistical tests."""
        real = real[np.isfinite(real)]
        sim = sim[np.isfinite(sim)]

        if len(real) == 0 or len(sim) == 0:
            return DistributionComparison(
                name=name,
                real_mean=0, sim_mean=0, real_std=0, sim_std=0,
                real_skew=0, sim_skew=0, real_kurtosis=0, sim_kurtosis=0,
                ks_statistic=1, ks_pvalue=0, passed=False,
            )

        real_mean = float(np.mean(real))
        sim_mean = float(np.mean(sim))
        real_std = float(np.std(real, ddof=1))
        sim_std = float(np.std(sim, ddof=1))
        real_skew = float(stats.skew(real))
        sim_skew = float(stats.skew(sim))
        real_kurt = float(stats.kurtosis(real))
        sim_kurt = float(stats.kurtosis(sim))

        ks_stat, ks_pval = stats.ks_2samp(real, sim)

        # Wasserstein distance
        try:
            wd = float(stats.wasserstein_distance(real, sim))
        except Exception:
            wd = 0.0

        # Anderson-Darling (approximate via KS)
        ad_stat = ks_stat * np.sqrt(len(real) * len(sim) / (len(real) + len(sim)))

        passed = ks_pval > self.significance

        return DistributionComparison(
            name=name,
            real_mean=real_mean,
            sim_mean=sim_mean,
            real_std=real_std,
            sim_std=sim_std,
            real_skew=real_skew,
            sim_skew=sim_skew,
            real_kurtosis=real_kurt,
            sim_kurtosis=sim_kurt,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            ad_statistic=float(ad_stat),
            wasserstein_distance=wd,
            passed=passed,
        )

    def _compare_autocorrelation(self, max_lag: int = 20) -> Dict[str, Any]:
        """Compare autocorrelation structure of returns and absolute returns."""
        real = self.real_returns[np.isfinite(self.real_returns)]
        sim = self.sim_returns[np.isfinite(self.sim_returns)]

        if len(real) < max_lag + 5 or len(sim) < max_lag + 5:
            return {"status": "insufficient_data"}

        def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
            n = len(x)
            x_centered = x - np.mean(x)
            var = np.var(x)
            if var == 0:
                return np.zeros(nlags + 1)
            acf_vals = np.zeros(nlags + 1)
            for lag in range(nlags + 1):
                acf_vals[lag] = np.sum(x_centered[:n - lag] * x_centered[lag:]) / (n * var)
            return acf_vals

        real_acf = _acf(real, max_lag)
        sim_acf = _acf(sim, max_lag)
        real_abs_acf = _acf(np.abs(real), max_lag)
        sim_abs_acf = _acf(np.abs(sim), max_lag)

        # distance between ACF curves
        acf_rmse = float(np.sqrt(np.mean((real_acf[1:] - sim_acf[1:]) ** 2)))
        abs_acf_rmse = float(np.sqrt(np.mean((real_abs_acf[1:] - sim_abs_acf[1:]) ** 2)))

        # stylized fact: returns ACF ~ 0, abs returns ACF > 0 (volatility clustering)
        real_has_clustering = bool(np.mean(real_abs_acf[1:6]) > 0.05)
        sim_has_clustering = bool(np.mean(sim_abs_acf[1:6]) > 0.05)

        return {
            "return_acf_rmse": acf_rmse,
            "abs_return_acf_rmse": abs_acf_rmse,
            "real_acf_first5": real_acf[1:6].tolist(),
            "sim_acf_first5": sim_acf[1:6].tolist(),
            "real_abs_acf_first5": real_abs_acf[1:6].tolist(),
            "sim_abs_acf_first5": sim_abs_acf[1:6].tolist(),
            "real_has_vol_clustering": real_has_clustering,
            "sim_has_vol_clustering": sim_has_clustering,
            "clustering_match": real_has_clustering == sim_has_clustering,
            "acf_quality": "good" if acf_rmse < 0.05 else ("fair" if acf_rmse < 0.1 else "poor"),
        }

    def _compare_volatility_clustering(self) -> Dict[str, Any]:
        """Compare volatility clustering via ARCH effects."""
        real = self.real_returns[np.isfinite(self.real_returns)]
        sim = self.sim_returns[np.isfinite(self.sim_returns)]

        def _arch_test(x: np.ndarray, lags: int = 5) -> Tuple[float, float]:
            """Engle's ARCH test (simplified)."""
            n = len(x)
            if n < lags + 10:
                return 0.0, 1.0
            e2 = (x - np.mean(x)) ** 2
            y = e2[lags:]
            X_mat = np.column_stack([e2[lags - i - 1:n - i - 1] for i in range(lags)])
            X_mat = np.column_stack([np.ones(len(y)), X_mat])
            try:
                beta_h, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
                fitted = X_mat @ beta_h
                ss_res = np.sum((y - fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                test_stat = len(y) * r2
                p_val = 1.0 - stats.chi2.cdf(test_stat, lags)
                return float(test_stat), float(p_val)
            except Exception:
                return 0.0, 1.0

        real_stat, real_pval = _arch_test(real)
        sim_stat, sim_pval = _arch_test(sim)

        real_has_arch = real_pval < 0.05
        sim_has_arch = sim_pval < 0.05

        # rolling vol comparison
        window = 20
        if len(real) >= window and len(sim) >= window:
            real_rolling = np.array([np.std(real[i:i + window]) for i in range(len(real) - window)])
            sim_rolling = np.array([np.std(sim[i:i + window]) for i in range(len(sim) - window)])
            vol_of_vol_real = float(np.std(real_rolling))
            vol_of_vol_sim = float(np.std(sim_rolling))
            vol_of_vol_ratio = vol_of_vol_sim / (vol_of_vol_real + 1e-10)
        else:
            vol_of_vol_real = 0.0
            vol_of_vol_sim = 0.0
            vol_of_vol_ratio = 1.0

        return {
            "real_arch_stat": real_stat,
            "real_arch_pval": real_pval,
            "sim_arch_stat": sim_stat,
            "sim_arch_pval": sim_pval,
            "real_has_arch_effects": real_has_arch,
            "sim_has_arch_effects": sim_has_arch,
            "arch_match": real_has_arch == sim_has_arch,
            "vol_of_vol_real": vol_of_vol_real,
            "vol_of_vol_sim": vol_of_vol_sim,
            "vol_of_vol_ratio": vol_of_vol_ratio,
        }

    def _compute_score(self) -> float:
        """Compute an overall calibration quality score [0, 1]."""
        if not self._comparisons:
            return 0.0

        scores: List[float] = []

        for comp in self._comparisons:
            s = 0.0
            # KS test contribution
            if comp.ks_pvalue > 0.1:
                s += 0.4
            elif comp.ks_pvalue > 0.05:
                s += 0.2

            # moment matching
            mean_err = abs(comp.real_mean - comp.sim_mean) / (abs(comp.real_mean) + 1e-10)
            if mean_err < 0.1:
                s += 0.15
            elif mean_err < 0.3:
                s += 0.07

            std_err = abs(comp.real_std - comp.sim_std) / (comp.real_std + 1e-10)
            if std_err < 0.1:
                s += 0.15
            elif std_err < 0.3:
                s += 0.07

            skew_err = abs(comp.real_skew - comp.sim_skew)
            if skew_err < 0.3:
                s += 0.15
            elif skew_err < 0.6:
                s += 0.07

            kurt_err = abs(comp.real_kurtosis - comp.sim_kurtosis)
            if kurt_err < 1.0:
                s += 0.15
            elif kurt_err < 3.0:
                s += 0.07

            scores.append(s)

        return float(np.mean(scores))

    def summary_text(self) -> str:
        """Generate human-readable summary."""
        report = self.run()
        lines = [
            "=" * 60,
            "CALIBRATION REPORT",
            "=" * 60,
            "",
        ]

        for comp_dict in report["distributions"]:
            lines.append(f"--- {comp_dict['name']} ---")
            lines.append(
                f"  Mean:     real={comp_dict['real_mean']:.6f}  sim={comp_dict['sim_mean']:.6f}"
            )
            lines.append(
                f"  Std:      real={comp_dict['real_std']:.6f}  sim={comp_dict['sim_std']:.6f}"
            )
            lines.append(
                f"  Skew:     real={comp_dict['real_skew']:.4f}  sim={comp_dict['sim_skew']:.4f}"
            )
            lines.append(
                f"  Kurtosis: real={comp_dict['real_kurtosis']:.4f}  sim={comp_dict['sim_kurtosis']:.4f}"
            )
            lines.append(
                f"  KS test:  stat={comp_dict['ks_statistic']:.4f}  p={comp_dict['ks_pvalue']:.4f}  "
                f"{'PASS' if comp_dict['passed'] else 'FAIL'}"
            )
            lines.append(f"  Wasserstein: {comp_dict['wasserstein_distance']:.6f}")
            lines.append("")

        ac = report.get("autocorrelation", {})
        if ac.get("status") != "insufficient_data":
            lines.append("--- Autocorrelation ---")
            lines.append(f"  Return ACF RMSE: {ac.get('acf_quality', 'N/A')} ({ac.get('return_acf_rmse', 0):.4f})")
            lines.append(f"  Abs return ACF RMSE: {ac.get('abs_return_acf_rmse', 0):.4f}")
            lines.append(f"  Vol clustering match: {ac.get('clustering_match', 'N/A')}")
            lines.append("")

        vc = report.get("volatility_clustering", {})
        lines.append("--- Volatility Clustering ---")
        lines.append(f"  ARCH match: {vc.get('arch_match', 'N/A')}")
        lines.append(f"  Vol-of-vol ratio: {vc.get('vol_of_vol_ratio', 0):.3f}")
        lines.append("")

        lines.append(f"Overall calibration score: {report['calibration_score']:.3f}")
        lines.append(f"Passed: {report['num_passed']}/{report['num_comparisons']}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RealisticExchangeFactory
# ---------------------------------------------------------------------------

class RealisticExchangeFactory:
    """
    Create a synthetic exchange calibrated to real historical data.

    The factory takes historical price, volume, and spread data, runs the
    full calibration pipeline, and produces an ExchangeConfig that should
    generate statistically indistinguishable synthetic data.
    """

    def __init__(self, seed: int = 42):
        self.calibrator = HistoricalCalibrator(seed=seed)
        self.seed = seed

    def from_historical(
        self,
        price_data: np.ndarray,
        volume_data: Optional[np.ndarray] = None,
        spread_data: Optional[np.ndarray] = None,
        trade_times: Optional[np.ndarray] = None,
        orderbook_snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> ExchangeConfig:
        """
        Create a fully calibrated ExchangeConfig from historical data.

        Args:
            price_data: Array of prices (OHLC or just close prices)
            volume_data: Array of volumes (same length as price_data)
            spread_data: Array of bid-ask spreads in bps
            trade_times: Array of trade timestamps for Hawkes calibration
            orderbook_snapshots: List of order book snapshots

        Returns:
            ExchangeConfig with all parameters calibrated to historical data
        """
        price_data = np.asarray(price_data, dtype=np.float64)
        if price_data.ndim == 2:
            closes = price_data[:, 3]  # assume OHLC
        else:
            closes = price_data

        closes = closes[np.isfinite(closes)]
        if len(closes) < 20:
            raise ValueError("Need at least 20 price observations")

        # compute returns
        log_returns = np.diff(np.log(closes + 1e-10))

        logger.info("Calibrating from %d price observations...", len(closes))

        # 1. Volatility calibration
        vol_params = self.calibrator.fit_volatility(log_returns)
        logger.info("  Volatility: daily=%.4f annual=%.4f GARCH(%.6f,%.3f,%.3f)",
                     vol_params.realized_vol_daily, vol_params.realized_vol_annual,
                     vol_params.garch_omega, vol_params.garch_alpha, vol_params.garch_beta)

        # 2. Order book calibration
        ob_params = OrderBookParams()
        if orderbook_snapshots:
            ob_params = self.calibrator.fit_orderbook_shape(orderbook_snapshots)
            logger.info("  Order book: decay=%.3f asymmetry=%.3f depth=%.1f",
                         ob_params.depth_decay_rate, ob_params.depth_asymmetry,
                         ob_params.avg_depth_per_level)

        # 3. Hawkes trade arrival
        hawkes_params = HawkesParams()
        if trade_times is not None:
            hawkes_params = self.calibrator.fit_trade_arrival(trade_times)
            logger.info("  Hawkes: mu=%.3f alpha=%.3f beta=%.3f branching=%.3f",
                         hawkes_params.baseline_intensity, hawkes_params.self_excitation,
                         hawkes_params.decay_rate, hawkes_params.branching_ratio)

        # 4. Spread model
        spread_params = SpreadModelParams()
        if volume_data is not None and spread_data is not None:
            vol_data = volume_data[:len(closes)]
            sp_data = spread_data[:len(closes)]
            # compute rolling volatility as feature
            window = 20
            rolling_vols = np.zeros(len(log_returns))
            for i in range(len(log_returns)):
                start = max(0, i - window)
                rolling_vols[i] = np.std(log_returns[start:i + 1], ddof=1) if i > 0 else vol_params.realized_vol_daily
            # align arrays
            min_len = min(len(sp_data), len(vol_data), len(rolling_vols))
            spread_params = self.calibrator.fit_spread_model(
                sp_data[:min_len], vol_data[:min_len], rolling_vols[:min_len]
            )
            logger.info("  Spread: base=%.2f bps, vol_sens=%.3f, volume_sens=%.3f",
                         spread_params.base_spread_bps,
                         spread_params.volatility_sensitivity,
                         spread_params.volume_sensitivity)
        else:
            # estimate spread from price data
            if price_data.ndim == 2 and price_data.shape[1] >= 4:
                # use high-low as proxy
                highs = price_data[:, 1]
                lows = price_data[:, 2]
                hl_spread = (highs - lows) / (closes + 1e-10) * 10000 * 0.3
                spread_params.base_spread_bps = float(np.median(hl_spread))

        # 5. Regime model
        regime_params = self.calibrator.fit_regime(log_returns, num_regimes=3)
        logger.info("  Regimes: %d states, persistence=%.3f",
                     regime_params.num_regimes,
                     float(np.mean(np.diag(regime_params.transition_matrix))))

        # 6. Jump process
        jump_params = self.calibrator.fit_jump_process(log_returns)
        logger.info("  Jumps: intensity=%.3f mean=%.4f std=%.4f clustering=%.3f",
                     jump_params.jump_intensity, jump_params.jump_mean,
                     jump_params.jump_std, jump_params.jump_clustering)

        # Determine agent counts based on market characteristics
        num_mm, num_tf, num_mr, num_noise, num_hft = self._estimate_agent_counts(
            vol_params, hawkes_params, regime_params
        )

        config = ExchangeConfig(
            volatility_params=vol_params,
            orderbook_params=ob_params,
            hawkes_params=hawkes_params,
            spread_params=spread_params,
            regime_params=regime_params,
            jump_params=jump_params,
            initial_price=float(closes[-1]),
            tick_size=self._estimate_tick_size(closes),
            lot_size=1.0,
            num_market_makers=num_mm,
            num_trend_followers=num_tf,
            num_mean_reversion=num_mr,
            num_noise_traders=num_noise,
            num_hft_agents=num_hft,
        )

        logger.info("Calibration complete. Config ready.")
        return config

    def _estimate_agent_counts(
        self,
        vol_params: VolatilityParams,
        hawkes_params: HawkesParams,
        regime_params: RegimeParams,
    ) -> Tuple[int, int, int, int, int]:
        """Estimate appropriate agent counts from market characteristics."""
        # higher vol -> more trend followers (momentum feeds on vol)
        vol_ratio = vol_params.realized_vol_annual / 0.20  # normalized to ~20%

        # higher Hawkes clustering -> more HFT
        cluster = hawkes_params.cluster_factor

        # low-vol regime fraction -> more mean reversion
        low_vol_frac = float(regime_params.stationary_dist[0])

        num_mm = max(3, min(10, int(5 / vol_ratio)))
        num_tf = max(3, min(15, int(8 * vol_ratio)))
        num_mr = max(2, min(10, int(4 + 6 * low_vol_frac)))
        num_noise = max(10, min(40, int(20 * vol_ratio)))
        num_hft = max(1, min(8, int(3 * cluster)))

        return num_mm, num_tf, num_mr, num_noise, num_hft

    def _estimate_tick_size(self, prices: np.ndarray) -> float:
        """Estimate appropriate tick size from price level."""
        median_price = np.median(prices)
        if median_price > 1000:
            return 0.1
        elif median_price > 100:
            return 0.01
        elif median_price > 10:
            return 0.001
        elif median_price > 1:
            return 0.0001
        else:
            return 0.00001

    def validate(
        self,
        config: ExchangeConfig,
        real_returns: np.ndarray,
        sim_returns: np.ndarray,
        real_spreads: Optional[np.ndarray] = None,
        sim_spreads: Optional[np.ndarray] = None,
    ) -> CalibrationReport:
        """Validate calibration by comparing real and simulated data."""
        return CalibrationReport(
            real_returns=real_returns,
            sim_returns=sim_returns,
            real_spreads=real_spreads,
            sim_spreads=sim_spreads,
        )

    def generate_synthetic_returns(
        self, config: ExchangeConfig, n_bars: int = 1000
    ) -> np.ndarray:
        """
        Generate synthetic returns using the calibrated parameters.
        Useful for quick validation without running the full exchange.
        """
        rng = np.random.default_rng(self.seed)
        vol = config.volatility_params
        regime = config.regime_params
        jump = config.jump_params

        returns = np.zeros(n_bars)
        current_regime = 0
        sigma2 = vol.realized_vol_daily ** 2

        for t in range(n_bars):
            # regime transition
            probs = regime.transition_matrix[current_regime]
            current_regime = rng.choice(regime.num_regimes, p=probs)

            # GARCH variance update
            if t > 0:
                sigma2 = (
                    vol.garch_omega
                    + vol.garch_alpha * returns[t - 1] ** 2
                    + vol.garch_beta * sigma2
                )

            # regime-adjusted volatility
            regime_vol = regime.regime_vols[current_regime]
            combined_vol = np.sqrt(sigma2) * (regime_vol / (vol.realized_vol_daily + 1e-10))
            combined_vol = max(combined_vol, 1e-8)

            # diffusion component
            regime_mean = regime.regime_means[current_regime]
            ret = regime_mean + combined_vol * rng.standard_normal()

            # jump component
            if rng.random() < jump.jump_intensity:
                jump_size = rng.normal(jump.jump_mean, jump.jump_std)
                ret += jump_size
                # clustering: possible second jump
                if rng.random() < jump.jump_clustering:
                    ret += rng.normal(jump.jump_mean, jump.jump_std)

            returns[t] = ret

        return returns

    def auto_calibrate_and_validate(
        self,
        price_data: np.ndarray,
        volume_data: Optional[np.ndarray] = None,
        spread_data: Optional[np.ndarray] = None,
        trade_times: Optional[np.ndarray] = None,
        orderbook_snapshots: Optional[List[Dict[str, Any]]] = None,
        n_sim_bars: int = 5000,
    ) -> Tuple[ExchangeConfig, Dict[str, Any]]:
        """
        Full pipeline: calibrate from historical data, generate synthetic data,
        and validate the calibration.

        Returns:
            (config, validation_report)
        """
        config = self.from_historical(
            price_data=price_data,
            volume_data=volume_data,
            spread_data=spread_data,
            trade_times=trade_times,
            orderbook_snapshots=orderbook_snapshots,
        )

        # generate synthetic returns
        sim_returns = self.generate_synthetic_returns(config, n_bars=n_sim_bars)

        # compute real returns
        prices = np.asarray(price_data, dtype=np.float64)
        if prices.ndim == 2:
            closes = prices[:, 3]
        else:
            closes = prices
        real_returns = np.diff(np.log(closes[np.isfinite(closes)] + 1e-10))

        # build report
        report = CalibrationReport(
            real_returns=real_returns,
            sim_returns=sim_returns,
            real_spreads=spread_data,
        )
        validation = report.run()

        logger.info("Calibration score: %.3f", validation["calibration_score"])
        logger.info("KS test results: %s",
                     {k: f"p={v:.4f}" for k, v in validation["all_ks_pvalues"].items()})

        return config, validation
